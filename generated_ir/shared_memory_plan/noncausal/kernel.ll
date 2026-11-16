; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/shared_memory_plan/noncausal/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/shared_memory_plan/noncausal/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !28 {
  %5 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %6 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %5), !dbg !8
  %7 = add nuw nsw i32 %6, 64, !dbg !8
  %8 = xor i32 %6, 16, !dbg !8
  %9 = and i32 %7, 192, !dbg !8
  %10 = icmp samesign ult i32 %8, %9, !dbg !8
  %11 = select i1 %10, i32 %8, i32 %6, !dbg !8
  %12 = shl nuw nsw i32 %11, 2, !dbg !8
  %13 = xor i32 %6, 32, !dbg !8
  %14 = icmp samesign ult i32 %13, %9, !dbg !8
  %15 = select i1 %14, i32 %13, i32 %6, !dbg !8
  %16 = shl nuw nsw i32 %15, 2, !dbg !8
  %17 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %18 = sext i32 %17 to i64, !dbg !5
  %19 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %20 = sext i32 %19 to i64, !dbg !5
  %21 = shl nsw i64 %20, 5, !dbg !5
  %22 = add nsw i64 %21, %18, !dbg !5
  %23 = lshr i64 %22, 5, !dbg !5
  %24 = and i64 %23, 576460752303423456, !dbg !5
  %25 = and i64 %18, 31, !dbg !5
  %26 = or disjoint i64 %24, %25, !dbg !5
  %27 = lshr i64 %22, 10, !dbg !9
  %28 = add nuw nsw i64 %23, %27, !dbg !9
  %29 = and i64 %28, 31, !dbg !9
  %30 = shl nsw i64 %26, 7, !dbg !9
  %31 = mul nsw i64 %27, -4096, !dbg !9
  %32 = add i64 %30, %31, !dbg !9
  %33 = shl nsw i64 %32, 7, !dbg !9
  %34 = call range(i32 0, 256) i32 @llvm.amdgcn.workitem.id.x(), !dbg !9
  %35 = zext nneg i32 %34 to i64, !dbg !9
  %.idx = shl nuw nsw i64 %29, 20, !dbg !9
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !9, !range !30, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !9
  %address.local874 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local875 = and i64 %address.local874, 1536, !dbg !9
  %address.local876 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !9
  %address.local877 = ptrtoint ptr addrspace(3) %address.local876 to i64, !dbg !9
  %address.local878 = trunc i64 %address.local877 to i32, !dbg !9
  %36 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local878), !dbg !9
  %address.local879 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local880 = and i64 %address.local879, 16256, !dbg !9
  %address.local882 = lshr i64 %address.local, 4, !dbg !9
  %address.local883 = and i64 %address.local882, 8, !dbg !9
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx4623 = shl i64 %32, 8, !dbg !9
  %38 = add i64 %.idx, %.idx4623, !dbg !9
  %address.local881.idx4624 = or disjoint i64 %address.local880, %address.local883, !dbg !9
  %39 = shl nuw nsw i64 %address.local881.idx4624, 1, !dbg !9
  %40 = or disjoint i64 %38, %39, !dbg !9
  %dma.buffer.offset4589 = trunc i64 %40 to i32, !dbg !9
  %41 = insertelement <2 x i64> poison, i64 %37, i64 0, !dbg !9
  %42 = bitcast <2 x i64> %41 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc4590 = shufflevector <4 x i32> %42, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %43 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset4589, <4 x i32> %dma.buffer.rsrc4590) #10, !dbg !9
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local876, i64 4096, !dbg !9
  %address.local889 = ptrtoint ptr addrspace(3) %address.local888 to i64, !dbg !9
  %address.local890 = trunc i64 %address.local889 to i32, !dbg !9
  %44 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local890), !dbg !9
  %dma.buffer.offset4587 = or disjoint i32 %dma.buffer.offset4589, 32, !dbg !9
  %45 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %44, i32 %dma.buffer.offset4587, <4 x i32> %dma.buffer.rsrc4590) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %46 = shl nuw nsw i64 %29, 19, !dbg !9
  %47 = add i64 %46, %33, !dbg !9
  %address.local898 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local875, !dbg !9
  %address.local899 = ptrtoint ptr addrspace(3) %address.local898 to i64, !dbg !9
  %address.local900 = trunc i64 %address.local899 to i32, !dbg !9
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local900), !dbg !9
  %49 = or disjoint i64 %47, 32, !dbg !9
  %50 = or disjoint i64 %address.local883, %49, !dbg !9
  %51 = or i64 %50, %address.local880, !dbg !9
  %.tr = trunc i64 %51 to i32, !dbg !9
  %dma.buffer.offset4585 = shl i32 %.tr, 1, !dbg !9
  %52 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %48, i32 %dma.buffer.offset4585, <4 x i32> %dma.buffer.rsrc4590) #10, !dbg !9
  %address.local910 = getelementptr i8, ptr addrspace(3) %address.local898, i64 4096, !dbg !9
  %address.local911 = ptrtoint ptr addrspace(3) %address.local910 to i64, !dbg !9
  %address.local912 = trunc i64 %address.local911 to i32, !dbg !9
  %53 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local912), !dbg !9
  %.idx46284629 = or disjoint i64 %47, %address.local880, !dbg !9
  %54 = or disjoint i64 %.idx46284629, %address.local883, !dbg !9
  %.tr4630 = trunc i64 %54 to i32, !dbg !9
  %55 = shl i32 %.tr4630, 1, !dbg !9
  %dma.buffer.offset4583 = or i32 %55, 96, !dbg !9
  %56 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %dma.buffer.offset4583, <4 x i32> %dma.buffer.rsrc4590) #10, !dbg !9
  %57 = lshr i64 %35, 6, !dbg !12
  %address.local917 = lshr i32 %address.thread, 2, !dbg !12
  %address.local918 = and i32 %address.local917, 4, !dbg !12
  %address.local919 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local920 = and i32 %address.local919, 384, !dbg !12
  %address.local921 = or disjoint i32 %address.local918, %address.local920, !dbg !12
  %address.local922 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local923 = and i32 %address.local922, 120, !dbg !12
  %address.local924 = or disjoint i32 %address.local921, %address.local923, !dbg !12
  %address.local925 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local926 = and i32 %address.local925, 1024, !dbg !12
  %address.local927 = or disjoint i32 %address.local924, %address.local926, !dbg !12
  %address.local928 = zext nneg i32 %address.local927 to i64, !dbg !12
  %address.local929 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local928, !dbg !12
  %58 = load <4 x half>, ptr addrspace(3) %address.local929, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local929, i64 4096, !dbg !12
  %59 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %60 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !12
  %lds.const.gep4423 = getelementptr i8, ptr addrspace(3) %address.local929, i64 1024, !dbg !12
  %61 = load <4 x half>, ptr addrspace(3) %lds.const.gep4423, align 8, !dbg !12
  %62 = fmul <4 x half> %59, splat (half 0xH3015), !dbg !12
  %lds.const.gep4424 = getelementptr i8, ptr addrspace(3) %address.local929, i64 5120, !dbg !12
  %63 = load <4 x half>, ptr addrspace(3) %lds.const.gep4424, align 8, !dbg !12
  %64 = fmul <4 x half> %61, splat (half 0xH3015), !dbg !12
  %65 = fmul <4 x half> %63, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %66 = or disjoint i64 %47, 64, !dbg !9
  %67 = or disjoint i64 %address.local883, %66, !dbg !9
  %68 = or i64 %67, %address.local880, !dbg !9
  %.tr4633 = trunc i64 %68 to i32, !dbg !9
  %dma.buffer.offset4581 = shl i32 %.tr4633, 1, !dbg !9
  %69 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset4581, <4 x i32> %dma.buffer.rsrc4590) #10, !dbg !9
  %dma.buffer.offset4579 = or i32 %55, 160, !dbg !9
  %70 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %44, i32 %dma.buffer.offset4579, <4 x i32> %dma.buffer.rsrc4590) #10, !dbg !9
  %lds.const.gep4425 = getelementptr i8, ptr addrspace(3) %address.local929, i64 8192, !dbg !12
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep4425, align 8, !dbg !12
  %lds.const.gep4426 = getelementptr i8, ptr addrspace(3) %address.local929, i64 12288, !dbg !12
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep4426, align 8, !dbg !12
  %lds.const.gep4427 = getelementptr i8, ptr addrspace(3) %address.local929, i64 9216, !dbg !12
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep4427, align 8, !dbg !12
  %74 = fmul <4 x half> %71, splat (half 0xH3015), !dbg !12
  %lds.const.gep4428 = getelementptr i8, ptr addrspace(3) %address.local929, i64 13312, !dbg !12
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep4428, align 8, !dbg !12
  %76 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !12
  %77 = fmul <4 x half> %73, splat (half 0xH3015), !dbg !12
  %78 = fmul <4 x half> %75, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %79 = or disjoint i64 %47, 96, !dbg !9
  %80 = or disjoint i64 %address.local883, %79, !dbg !9
  %81 = or i64 %80, %address.local880, !dbg !9
  %.tr4639 = trunc i64 %81 to i32, !dbg !9
  %dma.buffer.offset4577 = shl i32 %.tr4639, 1, !dbg !9
  %82 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %48, i32 %dma.buffer.offset4577, <4 x i32> %dma.buffer.rsrc4590) #10, !dbg !9
  %dma.buffer.offset4575 = or i32 %55, 224, !dbg !9
  %83 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %dma.buffer.offset4575, <4 x i32> %dma.buffer.rsrc4590) #10, !dbg !9
  %84 = load <4 x half>, ptr addrspace(3) %address.local929, align 8, !dbg !12
  %85 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep4423, align 8, !dbg !12
  %87 = fmul <4 x half> %84, splat (half 0xH3015), !dbg !12
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep4424, align 8, !dbg !12
  %89 = fmul <4 x half> %85, splat (half 0xH3015), !dbg !12
  %90 = fmul <4 x half> %86, splat (half 0xH3015), !dbg !12
  %91 = fmul <4 x half> %88, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep4425, align 8, !dbg !12
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep4426, align 8, !dbg !12
  %94 = load <4 x half>, ptr addrspace(3) %lds.const.gep4427, align 8, !dbg !12
  %95 = fmul <4 x half> %92, splat (half 0xH3015), !dbg !12
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep4428, align 8, !dbg !12
  %97 = fmul <4 x half> %93, splat (half 0xH3015), !dbg !12
  %98 = fmul <4 x half> %94, splat (half 0xH3015), !dbg !12
  %99 = fmul <4 x half> %96, splat (half 0xH3015), !dbg !12
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  %100 = shl nuw nsw i64 %35, 2, !dbg !15
  %k0.prime.address.thread1132 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !13, !range !30, !frisk.register_lifetimes !1
  %k0.prime.address.local1133 = zext nneg i32 %k0.prime.address.thread1132 to i64, !dbg !9
  %k0.prime.address.local1134 = shl nuw nsw i64 %k0.prime.address.local1133, 2, !dbg !13
  %k0.prime.address.local1135 = and i64 %k0.prime.address.local1134, 768, !dbg !13
  %k0.prime.address.local1136 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.prime.address.local1135, !dbg !13
  %k0.prime.address.local1137 = ptrtoint ptr addrspace(3) %k0.prime.address.local1136 to i64, !dbg !13
  %k0.prime.address.local1138 = trunc i64 %k0.prime.address.local1137 to i32, !dbg !13
  %k0.prime. = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local1138), !dbg !13
  %k0.prime.4786 = shl i64 %23, 19, !dbg !13
  %k0.prime.4787 = and i64 %k0.prime.4786, 16252928, !dbg !13
  %k0.prime.4788 = shl nuw nsw i64 0, 14, !dbg !13
  %k0.prime.4789 = or disjoint i64 %k0.prime.4787, %k0.prime.4788, !dbg !13
  %k0.prime.4790 = shl nuw nsw i64 %57, 2, !dbg !13
  %k0.prime.4791 = add nuw nsw i64 %100, %k0.prime.4790, !dbg !13
  %k0.prime.4792 = add nuw nsw i64 %57, %35, !dbg !13
  %k0.prime.4793 = shl nuw nsw i64 %k0.prime.4792, 2, !dbg !13
  %k0.prime.4794 = and i64 %k0.prime.4793, 2040, !dbg !13
  %k0.prime.4795 = sub nsw i64 %k0.prime.4791, %k0.prime.4794, !dbg !13
  %k0.prime.4796 = shl nuw nsw i32 %34, 4, !dbg !13
  %k0.prime.4797 = and i32 %k0.prime.4796, 3584, !dbg !13
  %k0.prime.4798 = shl nuw nsw i32 %34, 6, !dbg !13
  %k0.prime.4799 = and i32 %k0.prime.4798, 16256, !dbg !13
  %k0.prime.narrow805 = add nuw nsw i32 %k0.prime.4797, %k0.prime.4799, !dbg !13
  %k0.prime.4800 = zext nneg i32 %k0.prime.narrow805 to i64, !dbg !13
  %k0.prime.4801 = lshr i32 %34, 6, !dbg !13
  %k0.prime.4802 = lshr i32 %34, 5, !dbg !13
  %k0.prime.4803 = lshr i32 %34, 3, !dbg !13
  %k0.prime.4804 = and i32 %k0.prime.4803, 1, !dbg !13
  %k0.prime.4805 = and i32 %k0.prime.4802, %k0.prime.4804, !dbg !13
  %k0.prime.narrow804 = add nuw nsw i32 %k0.prime.4801, %k0.prime.4805, !dbg !13
  %k0.prime.4806 = shl nuw nsw i32 %k0.prime.narrow804, 10, !dbg !13
  %k0.prime.4807 = zext nneg i32 %k0.prime.4806 to i64, !dbg !13
  %k0.prime.4808 = sub nsw i64 %k0.prime.4800, %k0.prime.4807, !dbg !13
  %k0.prime.4809 = and i64 %k0.prime.4808, 16256, !dbg !13
  %k0.prime.4810 = add nsw i64 %k0.prime.4795, %k0.prime.4809, !dbg !13
  %k0.prime..idx46434646 = add nsw i64 %k0.prime.4789, %k0.prime.4810, !dbg !13
  %k0.prime..idx46434646.tr = trunc i64 %k0.prime..idx46434646 to i32, !dbg !13
  %k0.prime.dma.buffer.offset4621 = shl i32 %k0.prime..idx46434646.tr, 1, !dbg !13
  %k0.prime.4811 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %k0.prime.4812 = insertelement <2 x i64> poison, i64 %k0.prime.4811, i64 0, !dbg !13
  %k0.prime.4813 = bitcast <2 x i64> %k0.prime.4812 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc4622 = shufflevector <4 x i32> %k0.prime.4813, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.4814 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime., i32 %k0.prime.dma.buffer.offset4621, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.address.local1142 = getelementptr i8, ptr addrspace(3) %k0.prime.address.local1136, i64 2048, !dbg !13
  %k0.prime.address.local1143 = ptrtoint ptr addrspace(3) %k0.prime.address.local1142 to i64, !dbg !13
  %k0.prime.address.local1144 = trunc i64 %k0.prime.address.local1143 to i32, !dbg !13
  %k0.prime.4815 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local1144), !dbg !13
  %k0.prime..idx4647 = shl nuw nsw i64 %k0.prime.4789, 1, !dbg !13
  %k0.prime.4816 = add nuw nsw i64 %100, 1040, !dbg !13
  %k0.prime.4817 = add nuw nsw i64 %k0.prime.4816, %k0.prime.4790, !dbg !13
  %k0.prime.4818 = add nuw nsw i64 %35, 260, !dbg !13
  %k0.prime.4819 = add nuw nsw i64 %k0.prime.4818, %57, !dbg !13
  %k0.prime.4820 = shl nuw nsw i64 %k0.prime.4819, 2, !dbg !13
  %k0.prime.4821 = and i64 %k0.prime.4820, 8184, !dbg !13
  %k0.prime.4822 = sub nsw i64 %k0.prime.4817, %k0.prime.4821, !dbg !13
  %k0.prime..idx4648 = shl nsw i64 %k0.prime.4822, 1, !dbg !13
  %k0.prime.4823 = add nsw i64 %k0.prime..idx4648, 16, !dbg !13
  %k0.prime.4824 = shl nuw nsw i32 %34, 5, !dbg !13
  %k0.prime.4825 = and i32 %k0.prime.4824, 7168, !dbg !13
  %k0.prime.4826 = or disjoint i32 %k0.prime.4825, 8192, !dbg !13
  %k0.prime.4827 = shl nuw nsw i32 %34, 7, !dbg !13
  %k0.prime.4828 = and i32 %k0.prime.4827, 32512, !dbg !13
  %k0.prime.narrow807 = add nuw nsw i32 %k0.prime.4826, %k0.prime.4828, !dbg !13
  %k0.prime.4829 = or disjoint i32 %k0.prime.4801, 4, !dbg !13
  %k0.prime.narrow806 = add nuw nsw i32 %k0.prime.4829, %k0.prime.4805, !dbg !13
  %k0.prime.4830 = shl nuw nsw i32 %k0.prime.narrow806, 11, !dbg !13
  %k0.prime.4831 = sub nsw i32 %k0.prime.narrow807, %k0.prime.4830, !dbg !13
  %k0.prime.4832 = and i32 %k0.prime.4831, 32512, !dbg !13
  %k0.prime..idx4649 = zext nneg i32 %k0.prime.4832 to i64, !dbg !13
  %k0.prime.4833 = add i64 %k0.prime.4823, %k0.prime..idx4649, !dbg !13
  %k0.prime.4834 = add i64 %k0.prime..idx4647, %k0.prime.4833, !dbg !13
  %k0.prime.dma.buffer.offset4619 = trunc i64 %k0.prime.4834 to i32, !dbg !13
  %k0.prime.4835 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.4815, i32 %k0.prime.dma.buffer.offset4619, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.address.local1148 = getelementptr i8, ptr addrspace(3) %k0.prime.address.local1136, i64 4096, !dbg !13
  %k0.prime.address.local1149 = ptrtoint ptr addrspace(3) %k0.prime.address.local1148 to i64, !dbg !13
  %k0.prime.address.local1150 = trunc i64 %k0.prime.address.local1149 to i32, !dbg !13
  %k0.prime.4836 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local1150), !dbg !13
  %k0.prime.4837 = add nuw nsw i64 %100, 2080, !dbg !13
  %k0.prime.4838 = add nuw nsw i64 %k0.prime.4837, %k0.prime.4790, !dbg !13
  %k0.prime.4839 = add nuw nsw i64 %35, 520, !dbg !13
  %k0.prime.4840 = add nuw nsw i64 %k0.prime.4839, %57, !dbg !13
  %k0.prime.4841 = shl nuw nsw i64 %k0.prime.4840, 2, !dbg !13
  %k0.prime.4842 = and i64 %k0.prime.4841, 8184, !dbg !13
  %k0.prime.4843 = sub nsw i64 %k0.prime.4838, %k0.prime.4842, !dbg !13
  %k0.prime..idx4651 = shl nsw i64 %k0.prime.4843, 1, !dbg !13
  %k0.prime.4844 = add nsw i64 %k0.prime..idx4651, 32, !dbg !13
  %k0.prime.4845 = or disjoint i32 %k0.prime.4797, 8192, !dbg !13
  %k0.prime.narrow810 = add nuw nsw i32 %k0.prime.4845, %k0.prime.4799, !dbg !13
  %k0.prime.4846 = zext nneg i32 %k0.prime.narrow810 to i64, !dbg !13
  %k0.prime.4847 = or disjoint i32 %k0.prime.4801, 8, !dbg !13
  %k0.prime.narrow809 = add nuw nsw i32 %k0.prime.4847, %k0.prime.4805, !dbg !13
  %k0.prime.4848 = shl nuw nsw i32 %k0.prime.narrow809, 10, !dbg !13
  %k0.prime.4849 = zext nneg i32 %k0.prime.4848 to i64, !dbg !13
  %k0.prime.4850 = sub nsw i64 %k0.prime.4846, %k0.prime.4849, !dbg !13
  %k0.prime.4851 = shl nsw i64 %k0.prime.4850, 1, !dbg !13
  %k0.prime..idx4652 = and i64 %k0.prime.4851, 32512, !dbg !13
  %k0.prime.4852 = add i64 %k0.prime.4844, %k0.prime..idx4652, !dbg !13
  %k0.prime.4853 = add i64 %k0.prime..idx4647, %k0.prime.4852, !dbg !13
  %k0.prime.dma.buffer.offset4617 = trunc i64 %k0.prime.4853 to i32, !dbg !13
  %k0.prime.4854 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.4836, i32 %k0.prime.dma.buffer.offset4617, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.address.local1154 = getelementptr i8, ptr addrspace(3) %k0.prime.address.local1136, i64 6144, !dbg !13
  %k0.prime.address.local1155 = ptrtoint ptr addrspace(3) %k0.prime.address.local1154 to i64, !dbg !13
  %k0.prime.address.local1156 = trunc i64 %k0.prime.address.local1155 to i32, !dbg !13
  %k0.prime.4855 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local1156), !dbg !13
  %k0.prime.4856 = add nuw nsw i64 %100, 3120, !dbg !13
  %k0.prime.4857 = add nuw nsw i64 %k0.prime.4856, %k0.prime.4790, !dbg !13
  %k0.prime.4858 = add nuw nsw i64 %35, 780, !dbg !13
  %k0.prime.4859 = add nuw nsw i64 %k0.prime.4858, %57, !dbg !13
  %k0.prime.4860 = shl nuw nsw i64 %k0.prime.4859, 2, !dbg !13
  %k0.prime.4861 = and i64 %k0.prime.4860, 16376, !dbg !13
  %k0.prime.4862 = sub nsw i64 %k0.prime.4857, %k0.prime.4861, !dbg !13
  %k0.prime..idx46534656 = add nsw i64 %k0.prime.4789, %k0.prime.4862, !dbg !13
  %k0.prime..idx46534656.tr = trunc i64 %k0.prime..idx46534656 to i32, !dbg !13
  %k0.prime.4863 = shl i32 %k0.prime..idx46534656.tr, 1, !dbg !13
  %k0.prime.4864 = add i32 %k0.prime.4863, 48, !dbg !13
  %k0.prime.4865 = or disjoint i32 %k0.prime.4825, 24576, !dbg !13
  %k0.prime.narrow812 = add nuw nsw i32 %k0.prime.4865, %k0.prime.4828, !dbg !13
  %k0.prime.4866 = or disjoint i32 %k0.prime.4801, 12, !dbg !13
  %k0.prime.narrow811 = add nuw nsw i32 %k0.prime.4866, %k0.prime.4805, !dbg !13
  %k0.prime.4867 = shl nuw nsw i32 %k0.prime.narrow811, 11, !dbg !13
  %k0.prime.4868 = sub nsw i32 %k0.prime.narrow812, %k0.prime.4867, !dbg !13
  %k0.prime.4869 = and i32 %k0.prime.4868, 32512, !dbg !13
  %k0.prime.dma.buffer.offset4615 = add i32 %k0.prime.4864, %k0.prime.4869, !dbg !13
  %k0.prime.4870 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.4855, i32 %k0.prime.dma.buffer.offset4615, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %101, !dbg !6

101:                                              ; preds = %k0.join, %4
  %102 = phi i64 [ %1083, %k0.join ], [ 0, %4 ], !dbg !6
  %103 = phi [2 x <8 x float>] [ %1151, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %104 = phi [2 x <1 x float>] [ %924, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %105 = phi [2 x <1 x float>] [ %lazy.base.join, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %106 = phi [2 x <8 x float>] [ %1155, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %107 = phi [2 x <8 x float>] [ %1159, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %108 = phi [2 x <8 x float>] [ %1163, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %109 = icmp samesign ult i64 %102, 32, !dbg !6
  br i1 %109, label %110, label %1164, !dbg !6

110:                                              ; preds = %101
  %111 = shl i64 %23, 19, !dbg !13
  %112 = and i64 %111, 16252928, !dbg !13
  %113 = shl nuw nsw i64 %102, 14, !dbg !13
  %114 = or disjoint i64 %112, %113, !dbg !13
  %115 = shl nuw nsw i64 %57, 2, !dbg !13
  %116 = add nuw nsw i64 %100, %115, !dbg !13
  %117 = add nuw nsw i64 %57, %35, !dbg !13
  %118 = shl nuw nsw i64 %117, 2, !dbg !13
  %119 = and i64 %118, 2040, !dbg !13
  %120 = sub nsw i64 %116, %119, !dbg !13
  %121 = lshr i32 %34, 5, !dbg !13
  %122 = lshr i32 %34, 6, !dbg !13
  %123 = lshr i32 %34, 3, !dbg !13
  %124 = and i32 %123, 1, !dbg !13
  %125 = and i32 %121, %124, !dbg !13
  %narrow804 = add nuw nsw i32 %122, %125, !dbg !13
  %126 = shl nuw nsw i32 %narrow804, 10, !dbg !13
  %127 = zext nneg i32 %126 to i64, !dbg !13
  %128 = shl nuw nsw i32 %34, 4, !dbg !13
  %129 = and i32 %128, 3584, !dbg !13
  %130 = shl nuw nsw i32 %34, 6, !dbg !13
  %131 = and i32 %130, 16256, !dbg !13
  %narrow805 = add nuw nsw i32 %129, %131, !dbg !13
  %132 = zext nneg i32 %narrow805 to i64, !dbg !13
  %133 = sub nsw i64 %132, %127, !dbg !13
  %address.thread1132 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !13, !range !30, !frisk.register_lifetimes !1
  %address.local1133 = zext nneg i32 %address.thread1132 to i64, !dbg !9
  %address.local1134 = shl nuw nsw i64 %address.local1133, 2, !dbg !13
  %address.local1135 = and i64 %address.local1134, 768, !dbg !13
  %address.local1136 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1135, !dbg !13
  %address.local1137 = ptrtoint ptr addrspace(3) %address.local1136 to i64, !dbg !13
  %address.local1138 = trunc i64 %address.local1137 to i32, !dbg !13
  %134 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1138), !dbg !13
  %135 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %136 = and i64 %133, 16256, !dbg !13
  %137 = add nsw i64 %120, %136, !dbg !13
  %.idx46434646 = add nsw i64 %114, %137, !dbg !13
  %.idx46434646.tr = trunc i64 %.idx46434646 to i32, !dbg !13
  %dma.buffer.offset4621 = shl i32 %.idx46434646.tr, 1, !dbg !13
  %138 = insertelement <2 x i64> poison, i64 %135, i64 0, !dbg !13
  %139 = bitcast <2 x i64> %138 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc4622 = shufflevector <4 x i32> %139, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %140 = add nuw nsw i64 %100, 1040, !dbg !13
  %141 = add nuw nsw i64 %140, %115, !dbg !13
  %142 = add nuw nsw i64 %35, 260, !dbg !13
  %143 = add nuw nsw i64 %142, %57, !dbg !13
  %144 = shl nuw nsw i64 %143, 2, !dbg !13
  %145 = and i64 %144, 8184, !dbg !13
  %146 = sub nsw i64 %141, %145, !dbg !13
  %147 = or disjoint i32 %122, 4, !dbg !13
  %narrow806 = add nuw nsw i32 %147, %125, !dbg !13
  %148 = shl nuw nsw i32 %34, 5, !dbg !13
  %149 = and i32 %148, 7168, !dbg !13
  %150 = or disjoint i32 %149, 8192, !dbg !13
  %151 = shl nuw nsw i32 %34, 7, !dbg !13
  %152 = and i32 %151, 32512, !dbg !13
  %narrow807 = add nuw nsw i32 %150, %152, !dbg !13
  %address.local1142 = getelementptr i8, ptr addrspace(3) %address.local1136, i64 2048, !dbg !13
  %address.local1143 = ptrtoint ptr addrspace(3) %address.local1142 to i64, !dbg !13
  %address.local1144 = trunc i64 %address.local1143 to i32, !dbg !13
  %153 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1144), !dbg !13
  %.idx4647 = shl nuw nsw i64 %114, 1, !dbg !13
  %.idx4648 = shl nsw i64 %146, 1, !dbg !13
  %154 = add nsw i64 %.idx4648, 16, !dbg !13
  %155 = shl nuw nsw i32 %narrow806, 11, !dbg !13
  %156 = sub nsw i32 %narrow807, %155, !dbg !13
  %157 = and i32 %156, 32512, !dbg !13
  %.idx4649 = zext nneg i32 %157 to i64, !dbg !13
  %158 = add i64 %154, %.idx4649, !dbg !13
  %159 = add i64 %.idx4647, %158, !dbg !13
  %dma.buffer.offset4619 = trunc i64 %159 to i32, !dbg !13
  %160 = add nuw nsw i64 %100, 2080, !dbg !13
  %161 = add nuw nsw i64 %160, %115, !dbg !13
  %162 = add nuw nsw i64 %35, 520, !dbg !13
  %163 = add nuw nsw i64 %162, %57, !dbg !13
  %164 = shl nuw nsw i64 %163, 2, !dbg !13
  %165 = and i64 %164, 8184, !dbg !13
  %166 = sub nsw i64 %161, %165, !dbg !13
  %167 = or disjoint i32 %122, 8, !dbg !13
  %narrow809 = add nuw nsw i32 %167, %125, !dbg !13
  %168 = shl nuw nsw i32 %narrow809, 10, !dbg !13
  %169 = zext nneg i32 %168 to i64, !dbg !13
  %170 = or disjoint i32 %129, 8192, !dbg !13
  %narrow810 = add nuw nsw i32 %170, %131, !dbg !13
  %171 = zext nneg i32 %narrow810 to i64, !dbg !13
  %172 = sub nsw i64 %171, %169, !dbg !13
  %address.local1148 = getelementptr i8, ptr addrspace(3) %address.local1136, i64 4096, !dbg !13
  %address.local1149 = ptrtoint ptr addrspace(3) %address.local1148 to i64, !dbg !13
  %address.local1150 = trunc i64 %address.local1149 to i32, !dbg !13
  %173 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1150), !dbg !13
  %.idx4651 = shl nsw i64 %166, 1, !dbg !13
  %174 = add nsw i64 %.idx4651, 32, !dbg !13
  %175 = shl nsw i64 %172, 1, !dbg !13
  %.idx4652 = and i64 %175, 32512, !dbg !13
  %176 = add i64 %174, %.idx4652, !dbg !13
  %177 = add i64 %.idx4647, %176, !dbg !13
  %dma.buffer.offset4617 = trunc i64 %177 to i32, !dbg !13
  %178 = add nuw nsw i64 %100, 3120, !dbg !13
  %179 = add nuw nsw i64 %178, %115, !dbg !13
  %180 = add nuw nsw i64 %35, 780, !dbg !13
  %181 = add nuw nsw i64 %180, %57, !dbg !13
  %182 = shl nuw nsw i64 %181, 2, !dbg !13
  %183 = and i64 %182, 16376, !dbg !13
  %184 = sub nsw i64 %179, %183, !dbg !13
  %185 = or disjoint i32 %122, 12, !dbg !13
  %narrow811 = add nuw nsw i32 %185, %125, !dbg !13
  %186 = or disjoint i32 %149, 24576, !dbg !13
  %narrow812 = add nuw nsw i32 %186, %152, !dbg !13
  %address.local1154 = getelementptr i8, ptr addrspace(3) %address.local1136, i64 6144, !dbg !13
  %address.local1155 = ptrtoint ptr addrspace(3) %address.local1154 to i64, !dbg !13
  %address.local1156 = trunc i64 %address.local1155 to i32, !dbg !13
  %187 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1156), !dbg !13
  %.idx46534656 = add nsw i64 %114, %184, !dbg !13
  %188 = shl nuw nsw i32 %narrow811, 11, !dbg !13
  %189 = sub nsw i32 %narrow812, %188, !dbg !13
  %190 = and i32 %189, 32512, !dbg !13
  %.idx46534656.tr = trunc i64 %.idx46534656 to i32, !dbg !13
  %191 = shl i32 %.idx46534656.tr, 1, !dbg !13
  %192 = add i32 %191, 48, !dbg !13
  %dma.buffer.offset4615 = add i32 %192, %190, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %193 = or disjoint i64 %114, 32, !dbg !13
  %address.local1161 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1135, !dbg !13
  %address.local1162 = ptrtoint ptr addrspace(3) %address.local1161 to i64, !dbg !13
  %address.local1163 = trunc i64 %address.local1162 to i32, !dbg !13
  %194 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1163), !dbg !13
  %.idx46574660 = add nsw i64 %193, %137, !dbg !13
  %.idx46574660.tr = trunc i64 %.idx46574660 to i32, !dbg !13
  %dma.buffer.offset4613 = shl i32 %.idx46574660.tr, 1, !dbg !13
  %195 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %194, i32 %dma.buffer.offset4613, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13
  %address.local1167 = getelementptr i8, ptr addrspace(3) %address.local1161, i64 2048, !dbg !13
  %address.local1168 = ptrtoint ptr addrspace(3) %address.local1167 to i64, !dbg !13
  %address.local1169 = trunc i64 %address.local1168 to i32, !dbg !13
  %196 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1169), !dbg !13
  %.idx4661 = shl nuw nsw i64 %193, 1, !dbg !13
  %197 = add i64 %.idx4661, %158, !dbg !13
  %dma.buffer.offset4611 = trunc i64 %197 to i32, !dbg !13
  %198 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %196, i32 %dma.buffer.offset4611, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13
  %address.local1173 = getelementptr i8, ptr addrspace(3) %address.local1161, i64 4096, !dbg !13
  %address.local1174 = ptrtoint ptr addrspace(3) %address.local1173 to i64, !dbg !13
  %address.local1175 = trunc i64 %address.local1174 to i32, !dbg !13
  %199 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1175), !dbg !13
  %200 = add i64 %.idx4661, %176, !dbg !13
  %dma.buffer.offset4609 = trunc i64 %200 to i32, !dbg !13
  %201 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %199, i32 %dma.buffer.offset4609, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13
  %address.local1179 = getelementptr i8, ptr addrspace(3) %address.local1161, i64 6144, !dbg !13
  %address.local1180 = ptrtoint ptr addrspace(3) %address.local1179 to i64, !dbg !13
  %address.local1181 = trunc i64 %address.local1180 to i32, !dbg !13
  %202 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1181), !dbg !13
  %.idx46674670 = add nsw i64 %193, %184, !dbg !13
  %.idx46674670.tr = trunc i64 %.idx46674670 to i32, !dbg !13
  %203 = shl i32 %.idx46674670.tr, 1, !dbg !13
  %204 = add i32 %203, 48, !dbg !13
  %dma.buffer.offset4607 = add i32 %204, %190, !dbg !13
  %205 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %202, i32 %dma.buffer.offset4607, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13
  %address.local1184 = mul nuw nsw i64 %address.local1133, 160, !dbg !15
  %address.local1185 = lshr i64 %address.local1133, 2, !dbg !15
  %address.local1186 = mul nsw i64 %address.local1185, -640, !dbg !15
  %address.local1187 = add nsw i64 %address.local1184, %address.local1186, !dbg !15
  %address.local1188 = shl nuw nsw i64 %address.local1133, 1, !dbg !15
  %address.local1189 = and i64 %address.local1188, 24, !dbg !15
  %address.local1190 = or disjoint i64 %address.local1187, %address.local1189, !dbg !15
  %address.local1191 = shl nuw nsw i64 %address.local1133, 5, !dbg !15
  %address.local1192 = and i64 %address.local1191, 64, !dbg !15
  %address.local1193 = sub nsw i64 %address.local1190, %address.local1192, !dbg !15
  %address.local1194 = and i64 %address.local1133, 63, !dbg !12
  %address.local1195 = lshr i64 %address.local1194, 4, !dbg !12
  %address.local1196 = shl nuw nsw i64 %address.local1195, 2, !dbg !12
  %address.local1197 = add nsw i64 %address.local1193, %address.local1196, !dbg !15
  %address.local1198 = lshr i64 %address.local1194, 5, !dbg !12
  %address.local1199 = mul nuw nsw i64 %address.local1198, 1016, !dbg !12
  %address.local1200 = add i64 %address.local1197, %address.local1199, !dbg !15
  %address.local1201 = lshr i32 %address.thread1132, 1, !dbg !15
  %address.local1202 = and i32 %address.local1201, 16, !dbg !15
  %address.local1203 = shl nuw nsw i32 %address.thread1132, 1, !dbg !15
  %address.local1204 = and i32 %address.local1203, 4, !dbg !15
  %address.local1205 = or disjoint i32 %address.local1202, %address.local1204, !dbg !15
  %address.local1206 = zext nneg i32 %address.local1205 to i64, !dbg !15
  %address.local1207 = add i64 %address.local1200, %address.local1206, !dbg !15
  %address.local1209 = and i32 %address.local1201, 1, !dbg !15
  %address.local1210 = zext nneg i32 %address.local1209 to i64, !dbg !15
  %address.local1211 = and i64 %address.local1195, %address.local1210, !dbg !15
  %address.local1212 = lshr i32 %address.thread1132, 4, !dbg !15
  %address.local1213 = and i32 %address.local1212, 2, !dbg !15
  %address.local1214 = zext nneg i32 %address.local1213 to i64, !dbg !15
  %address.local1215 = or disjoint i64 %address.local1211, %address.local1214, !dbg !15
  %address.local1216 = shl nuw nsw i64 %address.local1215, 3, !dbg !15
  %address.local1217 = sub i64 %address.local1207, %address.local1216, !dbg !15
  %address.local1218 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1217, !dbg !15
  %206 = load <4 x half>, ptr addrspace(3) %address.local1218, align 8, !dbg !15, !frisk.k_lds_pair !32
  %lds.const.gep4439 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 128, !dbg !15
  %207 = load <4 x half>, ptr addrspace(3) %lds.const.gep4439, align 8, !dbg !15, !frisk.k_lds_pair !32
  %lds.const.gep4437 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 4096, !dbg !15
  %208 = load <4 x half>, ptr addrspace(3) %lds.const.gep4437, align 8, !dbg !15, !frisk.k_lds_pair !33
  %lds.const.gep4440 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 4224, !dbg !15
  %209 = load <4 x half>, ptr addrspace(3) %lds.const.gep4440, align 8, !dbg !15, !frisk.k_lds_pair !33
  %address.local1247 = and i32 %address.local1203, 24, !dbg !15
  %address.local1248 = shl nuw nsw i32 %address.thread1132, 8, !dbg !15
  %address.local1249 = and i32 %address.local1248, 768, !dbg !15
  %address.local1251 = shl nuw nsw i32 %address.thread1132, 5, !dbg !15
  %address.local1252 = and i32 %address.local1251, 96, !dbg !15
  %210 = or disjoint i32 %address.local1249, %address.local1252, !dbg !15
  %address.local1253 = or disjoint i32 %210, %address.local1247, !dbg !15
  %address.local1254 = zext nneg i32 %address.local1253 to i64, !dbg !15
  %211 = shl nuw nsw i64 %address.local1185, 4, !dbg !15
  %address.local1257 = sub nsw i64 %address.local1134, %211, !dbg !15
  %address.local1258 = and i32 %address.thread1132, 1, !dbg !15
  %address.local1259 = zext nneg i32 %address.local1258 to i64, !dbg !15
  %address.local1260 = or disjoint i64 %address.local1257, %address.local1259, !dbg !15
  %address.local1261 = add nuw nsw i64 %address.local1260, 1, !dbg !15
  %address.local1262 = sdiv i64 %address.local1261, 2, !dbg !15
  %address.local1266 = add nsw i64 %address.local1262, %address.local1210, !dbg !15
  %address.local1267 = shl i64 %address.local1266, 6, !dbg !15
  %address.local1268 = sub i64 %address.local1254, %address.local1267, !dbg !15
  %address.local1269 = add i64 %address.local1268, %address.local1196, !dbg !15
  %address.local1270 = add i64 %address.local1269, %address.local1199, !dbg !15
  %address.local1277 = add i64 %address.local1270, %address.local1206, !dbg !15
  %address.local1287 = sub i64 %address.local1277, %address.local1216, !dbg !15
  %212 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1287, !dbg !15
  %address.local1289 = getelementptr i8, ptr addrspace(3) %212, i64 64, !dbg !15
  %213 = load <4 x half>, ptr addrspace(3) %address.local1289, align 8, !dbg !15, !frisk.k_lds_pair !34
  %lds.const.gep4441 = getelementptr i8, ptr addrspace(3) %212, i64 192, !dbg !15
  %214 = load <4 x half>, ptr addrspace(3) %lds.const.gep4441, align 8, !dbg !15, !frisk.k_lds_pair !34
  %lds.const.gep4438 = getelementptr i8, ptr addrspace(3) %212, i64 4160, !dbg !15
  %215 = load <4 x half>, ptr addrspace(3) %lds.const.gep4438, align 8, !dbg !15, !frisk.k_lds_pair !35
  %lds.const.gep4442 = getelementptr i8, ptr addrspace(3) %212, i64 4288, !dbg !15
  %216 = load <4 x half>, ptr addrspace(3) %lds.const.gep4442, align 8, !dbg !15, !frisk.k_lds_pair !35
  %lds.const.gep4443 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 1024, !dbg !15
  %217 = load <4 x half>, ptr addrspace(3) %lds.const.gep4443, align 8, !dbg !15, !frisk.k_lds_pair !31
  %lds.const.gep4447 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 1152, !dbg !15
  %218 = load <4 x half>, ptr addrspace(3) %lds.const.gep4447, align 8, !dbg !15, !frisk.k_lds_pair !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %lazy.base = extractvalue [2 x <1 x float>] %105, 0, !dbg !15
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %lazy.base4880 = extractvalue [2 x <1 x float>] %105, 1, !dbg !15
  %lazy.seed4881 = shufflevector <1 x float> %lazy.base4880, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %206, <4 x float> %lazy.seed) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %lds.const.gep4444 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 5120, !dbg !15
  %220 = load <4 x half>, ptr addrspace(3) %lds.const.gep4444, align 8, !dbg !15, !frisk.k_lds_pair !36
  %lds.const.gep4448 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 5248, !dbg !15
  %221 = load <4 x half>, ptr addrspace(3) %lds.const.gep4448, align 8, !dbg !15, !frisk.k_lds_pair !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %208, <4 x float> %219) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %lds.const.gep4445 = getelementptr i8, ptr addrspace(3) %212, i64 1088, !dbg !15
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep4445, align 8, !dbg !15, !frisk.k_lds_pair !37
  %lds.const.gep4449 = getelementptr i8, ptr addrspace(3) %212, i64 1216, !dbg !15
  %224 = load <4 x half>, ptr addrspace(3) %lds.const.gep4449, align 8, !dbg !15, !frisk.k_lds_pair !37
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %213, <4 x float> %lazy.seed) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %lds.const.gep4446 = getelementptr i8, ptr addrspace(3) %212, i64 5184, !dbg !15
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep4446, align 8, !dbg !15, !frisk.k_lds_pair !38
  %lds.const.gep4450 = getelementptr i8, ptr addrspace(3) %212, i64 5312, !dbg !15
  %227 = load <4 x half>, ptr addrspace(3) %lds.const.gep4450, align 8, !dbg !15, !frisk.k_lds_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %215, <4 x float> %225) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %207, <4 x float> %lazy.seed) #9, !dbg !15
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %209, <4 x float> %229) #9, !dbg !15
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %214, <4 x float> %lazy.seed) #9, !dbg !15
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %216, <4 x float> %231) #9, !dbg !15
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %217, <4 x float> %lazy.seed) #9, !dbg !15
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %220, <4 x float> %233) #9, !dbg !15
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %223, <4 x float> %lazy.seed) #9, !dbg !15
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %226, <4 x float> %235) #9, !dbg !15
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %218, <4 x float> %lazy.seed) #9, !dbg !15
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %221, <4 x float> %237) #9, !dbg !15
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %224, <4 x float> %lazy.seed) #9, !dbg !15
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %227, <4 x float> %239) #9, !dbg !15
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %206, <4 x float> %lazy.seed4881) #9, !dbg !15
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %208, <4 x float> %241) #9, !dbg !15
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %213, <4 x float> %lazy.seed4881) #9, !dbg !15
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %215, <4 x float> %243) #9, !dbg !15
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %207, <4 x float> %lazy.seed4881) #9, !dbg !15
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %209, <4 x float> %245) #9, !dbg !15
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %214, <4 x float> %lazy.seed4881) #9, !dbg !15
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %216, <4 x float> %247) #9, !dbg !15
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %217, <4 x float> %lazy.seed4881) #9, !dbg !15
  %250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %220, <4 x float> %249) #9, !dbg !15
  %251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %223, <4 x float> %lazy.seed4881) #9, !dbg !15
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %226, <4 x float> %251) #9, !dbg !15
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %218, <4 x float> %lazy.seed4881) #9, !dbg !15
  %254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %221, <4 x float> %253) #9, !dbg !15
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %224, <4 x float> %lazy.seed4881) #9, !dbg !15
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %227, <4 x float> %255) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %257 = or disjoint i64 %114, 64, !dbg !13
  %.idx46714674 = add nsw i64 %257, %137, !dbg !13
  %.idx46714674.tr = trunc i64 %.idx46714674 to i32, !dbg !13
  %dma.buffer.offset4605 = shl i32 %.idx46714674.tr, 1, !dbg !13
  %258 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %134, i32 %dma.buffer.offset4605, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13
  %.idx4675 = shl nuw nsw i64 %257, 1, !dbg !13
  %259 = add i64 %.idx4675, %158, !dbg !13
  %dma.buffer.offset4603 = trunc i64 %259 to i32, !dbg !13
  %260 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %153, i32 %dma.buffer.offset4603, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13
  %261 = add i64 %.idx4675, %176, !dbg !13
  %dma.buffer.offset4601 = trunc i64 %261 to i32, !dbg !13
  %262 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %173, i32 %dma.buffer.offset4601, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13
  %.idx46814684 = add nsw i64 %257, %184, !dbg !13
  %.idx46814684.tr = trunc i64 %.idx46814684 to i32, !dbg !13
  %263 = shl i32 %.idx46814684.tr, 1, !dbg !13
  %264 = add i32 %263, 48, !dbg !13
  %dma.buffer.offset4599 = add i32 %264, %190, !dbg !13
  %265 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %187, i32 %dma.buffer.offset4599, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13
  %lds.const.gep4465 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 8192, !dbg !15
  %266 = load <4 x half>, ptr addrspace(3) %lds.const.gep4465, align 8, !dbg !15, !frisk.k_lds_pair !39
  %lds.const.gep4469 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 8320, !dbg !15
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep4469, align 8, !dbg !15, !frisk.k_lds_pair !39
  %lds.const.gep4466 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 12288, !dbg !15
  %268 = load <4 x half>, ptr addrspace(3) %lds.const.gep4466, align 8, !dbg !15, !frisk.k_lds_pair !40
  %lds.const.gep4470 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 12416, !dbg !15
  %269 = load <4 x half>, ptr addrspace(3) %lds.const.gep4470, align 8, !dbg !15, !frisk.k_lds_pair !40
  %lds.const.gep4467 = getelementptr i8, ptr addrspace(3) %212, i64 8256, !dbg !15
  %270 = load <4 x half>, ptr addrspace(3) %lds.const.gep4467, align 8, !dbg !15, !frisk.k_lds_pair !41
  %lds.const.gep4471 = getelementptr i8, ptr addrspace(3) %212, i64 8384, !dbg !15
  %271 = load <4 x half>, ptr addrspace(3) %lds.const.gep4471, align 8, !dbg !15, !frisk.k_lds_pair !41
  %lds.const.gep4468 = getelementptr i8, ptr addrspace(3) %212, i64 12352, !dbg !15
  %272 = load <4 x half>, ptr addrspace(3) %lds.const.gep4468, align 8, !dbg !15, !frisk.k_lds_pair !42
  %lds.const.gep4472 = getelementptr i8, ptr addrspace(3) %212, i64 12480, !dbg !15
  %273 = load <4 x half>, ptr addrspace(3) %lds.const.gep4472, align 8, !dbg !15, !frisk.k_lds_pair !42
  %lds.const.gep4473 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 9216, !dbg !15
  %274 = load <4 x half>, ptr addrspace(3) %lds.const.gep4473, align 8, !dbg !15, !frisk.k_lds_pair !43
  %lds.const.gep4477 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 9344, !dbg !15
  %275 = load <4 x half>, ptr addrspace(3) %lds.const.gep4477, align 8, !dbg !15, !frisk.k_lds_pair !43
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %266, <4 x float> %222) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %lds.const.gep4474 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 13312, !dbg !15
  %277 = load <4 x half>, ptr addrspace(3) %lds.const.gep4474, align 8, !dbg !15, !frisk.k_lds_pair !44
  %lds.const.gep4478 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 13440, !dbg !15
  %278 = load <4 x half>, ptr addrspace(3) %lds.const.gep4478, align 8, !dbg !15, !frisk.k_lds_pair !44
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %268, <4 x float> %276) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %lds.const.gep4475 = getelementptr i8, ptr addrspace(3) %212, i64 9280, !dbg !15
  %280 = load <4 x half>, ptr addrspace(3) %lds.const.gep4475, align 8, !dbg !15, !frisk.k_lds_pair !45
  %lds.const.gep4479 = getelementptr i8, ptr addrspace(3) %212, i64 9408, !dbg !15
  %281 = load <4 x half>, ptr addrspace(3) %lds.const.gep4479, align 8, !dbg !15, !frisk.k_lds_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %270, <4 x float> %228) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %lds.const.gep4476 = getelementptr i8, ptr addrspace(3) %212, i64 13376, !dbg !15
  %283 = load <4 x half>, ptr addrspace(3) %lds.const.gep4476, align 8, !dbg !15, !frisk.k_lds_pair !46
  %lds.const.gep4480 = getelementptr i8, ptr addrspace(3) %212, i64 13504, !dbg !15
  %284 = load <4 x half>, ptr addrspace(3) %lds.const.gep4480, align 8, !dbg !15, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %272, <4 x float> %282) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %267, <4 x float> %230) #9, !dbg !15
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %269, <4 x float> %286) #9, !dbg !15
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %271, <4 x float> %232) #9, !dbg !15
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %273, <4 x float> %288) #9, !dbg !15
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %274, <4 x float> %234) #9, !dbg !15
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %277, <4 x float> %290) #9, !dbg !15
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %280, <4 x float> %236) #9, !dbg !15
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %283, <4 x float> %292) #9, !dbg !15
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %275, <4 x float> %238) #9, !dbg !15
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %278, <4 x float> %294) #9, !dbg !15
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %281, <4 x float> %240) #9, !dbg !15
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %284, <4 x float> %296) #9, !dbg !15
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %266, <4 x float> %242) #9, !dbg !15
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %268, <4 x float> %298) #9, !dbg !15
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %270, <4 x float> %244) #9, !dbg !15
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %272, <4 x float> %300) #9, !dbg !15
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %267, <4 x float> %246) #9, !dbg !15
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %269, <4 x float> %302) #9, !dbg !15
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %271, <4 x float> %248) #9, !dbg !15
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %273, <4 x float> %304) #9, !dbg !15
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %274, <4 x float> %250) #9, !dbg !15
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %277, <4 x float> %306) #9, !dbg !15
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %280, <4 x float> %252) #9, !dbg !15
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %283, <4 x float> %308) #9, !dbg !15
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %275, <4 x float> %254) #9, !dbg !15
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %278, <4 x float> %310) #9, !dbg !15
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %281, <4 x float> %256) #9, !dbg !15
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %284, <4 x float> %312) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %314 = or disjoint i64 %114, 96, !dbg !13
  %.idx46854688 = add nsw i64 %314, %137, !dbg !13
  %.idx46854688.tr = trunc i64 %.idx46854688 to i32, !dbg !13
  %dma.buffer.offset4597 = shl i32 %.idx46854688.tr, 1, !dbg !13
  %315 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %194, i32 %dma.buffer.offset4597, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13
  %.idx4689 = shl nuw nsw i64 %314, 1, !dbg !13
  %316 = add i64 %.idx4689, %158, !dbg !13
  %dma.buffer.offset4595 = trunc i64 %316 to i32, !dbg !13
  %317 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %196, i32 %dma.buffer.offset4595, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13
  %318 = add i64 %.idx4689, %176, !dbg !13
  %dma.buffer.offset4593 = trunc i64 %318 to i32, !dbg !13
  %319 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %199, i32 %dma.buffer.offset4593, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13
  %.idx46954698 = add nsw i64 %314, %184, !dbg !13
  %.idx46954698.tr = trunc i64 %.idx46954698 to i32, !dbg !13
  %320 = shl i32 %.idx46954698.tr, 1, !dbg !13
  %321 = add i32 %320, 48, !dbg !13
  %dma.buffer.offset4591 = add i32 %321, %190, !dbg !13
  %322 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %202, i32 %dma.buffer.offset4591, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13
  %323 = load <4 x half>, ptr addrspace(3) %address.local1218, align 8, !dbg !15, !frisk.k_lds_pair !47
  %324 = load <4 x half>, ptr addrspace(3) %lds.const.gep4439, align 8, !dbg !15, !frisk.k_lds_pair !47
  %325 = load <4 x half>, ptr addrspace(3) %lds.const.gep4437, align 8, !dbg !15, !frisk.k_lds_pair !48
  %326 = load <4 x half>, ptr addrspace(3) %lds.const.gep4440, align 8, !dbg !15, !frisk.k_lds_pair !48
  %327 = load <4 x half>, ptr addrspace(3) %address.local1289, align 8, !dbg !15, !frisk.k_lds_pair !49
  %328 = load <4 x half>, ptr addrspace(3) %lds.const.gep4441, align 8, !dbg !15, !frisk.k_lds_pair !49
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep4438, align 8, !dbg !15, !frisk.k_lds_pair !50
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep4442, align 8, !dbg !15, !frisk.k_lds_pair !50
  %331 = load <4 x half>, ptr addrspace(3) %lds.const.gep4443, align 8, !dbg !15, !frisk.k_lds_pair !51
  %332 = load <4 x half>, ptr addrspace(3) %lds.const.gep4447, align 8, !dbg !15, !frisk.k_lds_pair !51
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %323, <4 x float> %279) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep4444, align 8, !dbg !15, !frisk.k_lds_pair !52
  %335 = load <4 x half>, ptr addrspace(3) %lds.const.gep4448, align 8, !dbg !15, !frisk.k_lds_pair !52
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %325, <4 x float> %333) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep4445, align 8, !dbg !15, !frisk.k_lds_pair !53
  %338 = load <4 x half>, ptr addrspace(3) %lds.const.gep4449, align 8, !dbg !15, !frisk.k_lds_pair !53
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %327, <4 x float> %285) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %340 = load <4 x half>, ptr addrspace(3) %lds.const.gep4446, align 8, !dbg !15, !frisk.k_lds_pair !54
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep4450, align 8, !dbg !15, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %329, <4 x float> %339) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %324, <4 x float> %287) #9, !dbg !15
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %326, <4 x float> %343) #9, !dbg !15
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %328, <4 x float> %289) #9, !dbg !15
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %330, <4 x float> %345) #9, !dbg !15
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %331, <4 x float> %291) #9, !dbg !15
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %334, <4 x float> %347) #9, !dbg !15
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %337, <4 x float> %293) #9, !dbg !15
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %340, <4 x float> %349) #9, !dbg !15
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %332, <4 x float> %295) #9, !dbg !15
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %335, <4 x float> %351) #9, !dbg !15
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %338, <4 x float> %297) #9, !dbg !15
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %341, <4 x float> %353) #9, !dbg !15
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %323, <4 x float> %299) #9, !dbg !15
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %325, <4 x float> %355) #9, !dbg !15
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %327, <4 x float> %301) #9, !dbg !15
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %329, <4 x float> %357) #9, !dbg !15
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %324, <4 x float> %303) #9, !dbg !15
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %326, <4 x float> %359) #9, !dbg !15
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %328, <4 x float> %305) #9, !dbg !15
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %330, <4 x float> %361) #9, !dbg !15
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %331, <4 x float> %307) #9, !dbg !15
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %334, <4 x float> %363) #9, !dbg !15
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %337, <4 x float> %309) #9, !dbg !15
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %340, <4 x float> %365) #9, !dbg !15
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %332, <4 x float> %311) #9, !dbg !15
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %335, <4 x float> %367) #9, !dbg !15
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %338, <4 x float> %313) #9, !dbg !15
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %341, <4 x float> %369) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local3187 = shl nuw nsw i64 %address.local1133, 3, !dbg !19
  %address.local3188 = and i64 %address.local3187, 1536, !dbg !19
  %address.local3189 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local3188, !dbg !19
  %address.local3190 = ptrtoint ptr addrspace(3) %address.local3189 to i64, !dbg !19
  %address.local3191 = trunc i64 %address.local3190 to i32, !dbg !19
  %371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3191), !dbg !19
  %address.local3192 = shl nuw nsw i64 %address.local1133, 7, !dbg !19
  %address.local3193 = and i64 %address.local3192, 16256, !dbg !19
  %address.local3195 = lshr i64 %address.local1133, 4, !dbg !19
  %address.local3196 = and i64 %address.local3195, 8, !dbg !19
  %372 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %address.local3194.idx4700 = or disjoint i64 %address.local3193, %address.local3196, !dbg !19
  %.idx46994701 = or disjoint i64 %114, %address.local3194.idx4700, !dbg !19
  %.idx46994701.tr = trunc nuw i64 %.idx46994701 to i32, !dbg !19
  %dma.buffer.offset4573 = shl nuw nsw i32 %.idx46994701.tr, 1, !dbg !19
  %373 = insertelement <2 x i64> poison, i64 %372, i64 0, !dbg !19
  %374 = bitcast <2 x i64> %373 to <4 x i32>, !dbg !19
  %dma.buffer.rsrc4574 = shufflevector <4 x i32> %374, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !19
  %375 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %371, i32 %dma.buffer.offset4573, <4 x i32> %dma.buffer.rsrc4574) #10, !dbg !19
  %address.local3201 = getelementptr i8, ptr addrspace(3) %address.local3189, i64 4096, !dbg !19
  %address.local3202 = ptrtoint ptr addrspace(3) %address.local3201 to i64, !dbg !19
  %address.local3203 = trunc i64 %address.local3202 to i32, !dbg !19
  %376 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3203), !dbg !19
  %.idx47024703 = or disjoint i64 %114, %address.local3193, !dbg !19
  %377 = or disjoint i64 %.idx47024703, %address.local3196, !dbg !19
  %.tr4704 = trunc nuw i64 %377 to i32, !dbg !19
  %378 = shl nuw nsw i32 %.tr4704, 1, !dbg !19
  %dma.buffer.offset4571 = or i32 %378, 32, !dbg !19
  %379 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %376, i32 %dma.buffer.offset4571, <4 x i32> %dma.buffer.rsrc4574) #10, !dbg !19
  %380 = load <4 x half>, ptr addrspace(3) %lds.const.gep4465, align 8, !dbg !15, !frisk.k_tail_pair !32
  %381 = load <4 x half>, ptr addrspace(3) %lds.const.gep4469, align 8, !dbg !15, !frisk.k_tail_pair !32
  %382 = load <4 x half>, ptr addrspace(3) %lds.const.gep4466, align 8, !dbg !15, !frisk.k_tail_pair !33
  %383 = load <4 x half>, ptr addrspace(3) %lds.const.gep4470, align 8, !dbg !15, !frisk.k_tail_pair !33
  %384 = load <4 x half>, ptr addrspace(3) %lds.const.gep4467, align 8, !dbg !15, !frisk.k_tail_pair !34
  %385 = load <4 x half>, ptr addrspace(3) %lds.const.gep4471, align 8, !dbg !15, !frisk.k_tail_pair !34
  %386 = load <4 x half>, ptr addrspace(3) %lds.const.gep4468, align 8, !dbg !15, !frisk.k_tail_pair !35
  %387 = load <4 x half>, ptr addrspace(3) %lds.const.gep4472, align 8, !dbg !15, !frisk.k_tail_pair !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %380, <4 x float> %336) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %382, <4 x float> %388) #9, !dbg !15
  %390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %384, <4 x float> %342) #9, !dbg !15
  %391 = extractelement <4 x float> %389, i64 0, !dbg !15
  %392 = call float @llvm.maxnum.f32(float %391, float 0xFFF0000000000000), !dbg !8
  %393 = extractelement <4 x float> %389, i64 1, !dbg !15
  %394 = extractelement <4 x float> %389, i64 2, !dbg !15
  %395 = extractelement <4 x float> %389, i64 3, !dbg !15
  %396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %386, <4 x float> %390) #9, !dbg !15
  %397 = load <4 x half>, ptr addrspace(3) %lds.const.gep4473, align 8, !dbg !15, !frisk.k_tail_pair !31
  %398 = load <4 x half>, ptr addrspace(3) %lds.const.gep4477, align 8, !dbg !15, !frisk.k_tail_pair !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %381, <4 x float> %344) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %400 = extractelement <4 x float> %396, i64 0, !dbg !15
  %401 = call float @llvm.maxnum.f32(float %392, float %400), !dbg !8
  %402 = extractelement <4 x float> %396, i64 1, !dbg !15
  %403 = call float @llvm.maxnum.f32(float %393, float %402), !dbg !8
  %404 = extractelement <4 x float> %396, i64 2, !dbg !15
  %405 = call float @llvm.maxnum.f32(float %394, float %404), !dbg !8
  %406 = extractelement <4 x float> %396, i64 3, !dbg !15
  %407 = call float @llvm.maxnum.f32(float %395, float %406), !dbg !8
  %408 = load <4 x half>, ptr addrspace(3) %lds.const.gep4474, align 8, !dbg !15, !frisk.k_tail_pair !36
  %409 = load <4 x half>, ptr addrspace(3) %lds.const.gep4478, align 8, !dbg !15, !frisk.k_tail_pair !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %383, <4 x float> %399) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %411 = load <4 x half>, ptr addrspace(3) %lds.const.gep4475, align 8, !dbg !15, !frisk.k_tail_pair !37
  %412 = load <4 x half>, ptr addrspace(3) %lds.const.gep4479, align 8, !dbg !15, !frisk.k_tail_pair !37
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %385, <4 x float> %346) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %414 = extractelement <4 x float> %410, i64 0, !dbg !15
  %415 = call float @llvm.maxnum.f32(float %401, float %414), !dbg !8
  %416 = extractelement <4 x float> %410, i64 1, !dbg !15
  %417 = call float @llvm.maxnum.f32(float %403, float %416), !dbg !8
  %418 = extractelement <4 x float> %410, i64 2, !dbg !15
  %419 = call float @llvm.maxnum.f32(float %405, float %418), !dbg !8
  %420 = extractelement <4 x float> %410, i64 3, !dbg !15
  %421 = call float @llvm.maxnum.f32(float %407, float %420), !dbg !8
  %422 = load <4 x half>, ptr addrspace(3) %lds.const.gep4476, align 8, !dbg !15, !frisk.k_tail_pair !38
  %423 = load <4 x half>, ptr addrspace(3) %lds.const.gep4480, align 8, !dbg !15, !frisk.k_tail_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %387, <4 x float> %413) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %397, <4 x float> %348) #9, !dbg !15
  %426 = extractelement <4 x float> %424, i64 0, !dbg !15
  %427 = call float @llvm.maxnum.f32(float %415, float %426), !dbg !8
  %428 = extractelement <4 x float> %424, i64 1, !dbg !15
  %429 = call float @llvm.maxnum.f32(float %417, float %428), !dbg !8
  %430 = extractelement <4 x float> %424, i64 2, !dbg !15
  %431 = call float @llvm.maxnum.f32(float %419, float %430), !dbg !8
  %432 = extractelement <4 x float> %424, i64 3, !dbg !15
  %433 = call float @llvm.maxnum.f32(float %421, float %432), !dbg !8
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %408, <4 x float> %425) #9, !dbg !15
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %411, <4 x float> %350) #9, !dbg !15
  %436 = extractelement <4 x float> %434, i64 0, !dbg !15
  %437 = call float @llvm.maxnum.f32(float %427, float %436), !dbg !8
  %438 = extractelement <4 x float> %434, i64 1, !dbg !15
  %439 = call float @llvm.maxnum.f32(float %429, float %438), !dbg !8
  %440 = extractelement <4 x float> %434, i64 2, !dbg !15
  %441 = call float @llvm.maxnum.f32(float %431, float %440), !dbg !8
  %442 = extractelement <4 x float> %434, i64 3, !dbg !15
  %443 = call float @llvm.maxnum.f32(float %433, float %442), !dbg !8
  %444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %422, <4 x float> %435) #9, !dbg !15
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %398, <4 x float> %352) #9, !dbg !15
  %446 = extractelement <4 x float> %444, i64 0, !dbg !15
  %447 = call float @llvm.maxnum.f32(float %437, float %446), !dbg !8
  %448 = extractelement <4 x float> %444, i64 1, !dbg !15
  %449 = call float @llvm.maxnum.f32(float %439, float %448), !dbg !8
  %450 = extractelement <4 x float> %444, i64 2, !dbg !15
  %451 = call float @llvm.maxnum.f32(float %441, float %450), !dbg !8
  %452 = extractelement <4 x float> %444, i64 3, !dbg !15
  %453 = call float @llvm.maxnum.f32(float %443, float %452), !dbg !8
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %409, <4 x float> %445) #9, !dbg !15
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %412, <4 x float> %354) #9, !dbg !15
  %456 = extractelement <4 x float> %454, i64 0, !dbg !15
  %457 = call float @llvm.maxnum.f32(float %447, float %456), !dbg !8
  %458 = extractelement <4 x float> %454, i64 1, !dbg !15
  %459 = call float @llvm.maxnum.f32(float %449, float %458), !dbg !8
  %460 = extractelement <4 x float> %454, i64 2, !dbg !15
  %461 = call float @llvm.maxnum.f32(float %451, float %460), !dbg !8
  %462 = extractelement <4 x float> %454, i64 3, !dbg !15
  %463 = call float @llvm.maxnum.f32(float %453, float %462), !dbg !8
  %464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %423, <4 x float> %455) #9, !dbg !15
  %465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %380, <4 x float> %356) #9, !dbg !15
  %466 = extractelement <4 x float> %464, i64 0, !dbg !15
  %467 = call float @llvm.maxnum.f32(float %457, float %466), !dbg !8
  %468 = extractelement <4 x float> %464, i64 1, !dbg !15
  %469 = call float @llvm.maxnum.f32(float %459, float %468), !dbg !8
  %470 = extractelement <4 x float> %464, i64 2, !dbg !15
  %471 = call float @llvm.maxnum.f32(float %461, float %470), !dbg !8
  %472 = extractelement <4 x float> %464, i64 3, !dbg !15
  %473 = call float @llvm.maxnum.f32(float %463, float %472), !dbg !8
  %474 = call float @llvm.maxnum.f32(float %467, float %469), !dbg !8
  %475 = call float @llvm.maxnum.f32(float %471, float %473), !dbg !8
  %476 = call float @llvm.maxnum.f32(float %474, float %475), !dbg !8
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %382, <4 x float> %465) #9, !dbg !15
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %384, <4 x float> %358) #9, !dbg !15
  %479 = extractelement <4 x float> %477, i64 0, !dbg !15
  %480 = extractelement <4 x float> %477, i64 1, !dbg !15
  %481 = extractelement <4 x float> %477, i64 2, !dbg !15
  %482 = extractelement <4 x float> %477, i64 3, !dbg !15
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %386, <4 x float> %478) #9, !dbg !15
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %381, <4 x float> %360) #9, !dbg !15
  %485 = extractelement <4 x float> %483, i64 0, !dbg !15
  %486 = extractelement <4 x float> %483, i64 1, !dbg !15
  %487 = extractelement <4 x float> %483, i64 2, !dbg !15
  %488 = extractelement <4 x float> %483, i64 3, !dbg !15
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %383, <4 x float> %484) #9, !dbg !15
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %385, <4 x float> %362) #9, !dbg !15
  %491 = extractelement <4 x float> %489, i64 0, !dbg !15
  %492 = extractelement <4 x float> %489, i64 1, !dbg !15
  %493 = extractelement <4 x float> %489, i64 2, !dbg !15
  %494 = extractelement <4 x float> %489, i64 3, !dbg !15
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %387, <4 x float> %490) #9, !dbg !15
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %397, <4 x float> %364) #9, !dbg !15
  %497 = extractelement <4 x float> %495, i64 0, !dbg !15
  %498 = extractelement <4 x float> %495, i64 1, !dbg !15
  %499 = extractelement <4 x float> %495, i64 2, !dbg !15
  %500 = extractelement <4 x float> %495, i64 3, !dbg !15
  %501 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %408, <4 x float> %496) #9, !dbg !15
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %411, <4 x float> %366) #9, !dbg !15
  %503 = extractelement <4 x float> %501, i64 0, !dbg !15
  %504 = extractelement <4 x float> %501, i64 1, !dbg !15
  %505 = extractelement <4 x float> %501, i64 2, !dbg !15
  %506 = extractelement <4 x float> %501, i64 3, !dbg !15
  %507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %422, <4 x float> %502) #9, !dbg !15
  %508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %398, <4 x float> %368) #9, !dbg !15
  %509 = extractelement <4 x float> %507, i64 0, !dbg !15
  %510 = extractelement <4 x float> %507, i64 1, !dbg !15
  %511 = extractelement <4 x float> %507, i64 2, !dbg !15
  %512 = extractelement <4 x float> %507, i64 3, !dbg !15
  %513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %409, <4 x float> %508) #9, !dbg !15
  %514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %412, <4 x float> %370) #9, !dbg !15
  %515 = extractelement <4 x float> %513, i64 0, !dbg !15
  %516 = extractelement <4 x float> %513, i64 1, !dbg !15
  %517 = extractelement <4 x float> %513, i64 2, !dbg !15
  %518 = extractelement <4 x float> %513, i64 3, !dbg !15
  %519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %423, <4 x float> %514) #9, !dbg !15
  %520 = extractelement <4 x float> %519, i64 0, !dbg !15
  %521 = extractelement <4 x float> %519, i64 1, !dbg !15
  %522 = extractelement <4 x float> %519, i64 2, !dbg !15
  %523 = extractelement <4 x float> %519, i64 3, !dbg !15
  %524 = bitcast float %476 to i32, !dbg !8
  %525 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %524), !dbg !8
  %526 = bitcast i32 %525 to float, !dbg !8
  %527 = call float @llvm.maxnum.f32(float %476, float %526), !dbg !8
  %528 = bitcast float %527 to i32, !dbg !8
  %529 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %528), !dbg !8
  %530 = bitcast i32 %529 to float, !dbg !8
  %531 = call float @llvm.maxnum.f32(float %527, float %530), !dbg !8
  %532 = call float @llvm.maxnum.f32(float %479, float 0xFFF0000000000000), !dbg !8
  %533 = call float @llvm.maxnum.f32(float %532, float %485), !dbg !8
  %534 = call float @llvm.maxnum.f32(float %480, float %486), !dbg !8
  %535 = call float @llvm.maxnum.f32(float %481, float %487), !dbg !8
  %536 = call float @llvm.maxnum.f32(float %482, float %488), !dbg !8
  %537 = call float @llvm.maxnum.f32(float %533, float %491), !dbg !8
  %538 = call float @llvm.maxnum.f32(float %534, float %492), !dbg !8
  %539 = call float @llvm.maxnum.f32(float %535, float %493), !dbg !8
  %540 = call float @llvm.maxnum.f32(float %536, float %494), !dbg !8
  %541 = call float @llvm.maxnum.f32(float %537, float %497), !dbg !8
  %542 = call float @llvm.maxnum.f32(float %538, float %498), !dbg !8
  %543 = call float @llvm.maxnum.f32(float %539, float %499), !dbg !8
  %544 = call float @llvm.maxnum.f32(float %540, float %500), !dbg !8
  %545 = call float @llvm.maxnum.f32(float %541, float %503), !dbg !8
  %546 = call float @llvm.maxnum.f32(float %542, float %504), !dbg !8
  %547 = call float @llvm.maxnum.f32(float %543, float %505), !dbg !8
  %548 = call float @llvm.maxnum.f32(float %544, float %506), !dbg !8
  %549 = call float @llvm.maxnum.f32(float %545, float %509), !dbg !8
  %550 = call float @llvm.maxnum.f32(float %546, float %510), !dbg !8
  %551 = call float @llvm.maxnum.f32(float %547, float %511), !dbg !8
  %552 = call float @llvm.maxnum.f32(float %548, float %512), !dbg !8
  %553 = call float @llvm.maxnum.f32(float %549, float %515), !dbg !8
  %554 = call float @llvm.maxnum.f32(float %550, float %516), !dbg !8
  %555 = call float @llvm.maxnum.f32(float %551, float %517), !dbg !8
  %556 = call float @llvm.maxnum.f32(float %552, float %518), !dbg !8
  %557 = call float @llvm.maxnum.f32(float %553, float %520), !dbg !8
  %558 = call float @llvm.maxnum.f32(float %554, float %521), !dbg !8
  %559 = call float @llvm.maxnum.f32(float %555, float %522), !dbg !8
  %560 = call float @llvm.maxnum.f32(float %556, float %523), !dbg !8
  %561 = call float @llvm.maxnum.f32(float %557, float %558), !dbg !8
  %562 = call float @llvm.maxnum.f32(float %559, float %560), !dbg !8
  %563 = call float @llvm.maxnum.f32(float %561, float %562), !dbg !8
  %564 = bitcast float %563 to i32, !dbg !8
  %565 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %564), !dbg !8
  %566 = bitcast i32 %565 to float, !dbg !8
  %567 = call float @llvm.maxnum.f32(float %563, float %566), !dbg !8
  %568 = bitcast float %567 to i32, !dbg !8
  %569 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %568), !dbg !8
  %570 = bitcast i32 %569 to float, !dbg !8
  %571 = call float @llvm.maxnum.f32(float %567, float %570), !dbg !8
  %lazy.peak = call float @llvm.maxnum.f32(float %531, float %571), !dbg !23
  %lazy.over = fcmp ogt float %lazy.peak, 0x4020000000000000, !dbg !23
  %lazy.mask = call i64 @llvm.amdgcn.ballot.i64(i1 %lazy.over), !dbg !23
  %lazy.any = icmp ne i64 %lazy.mask, 0, !dbg !23
  %lazy.first = icmp eq i64 %102, 0, !dbg !23
  %lazy.take = or i1 %lazy.first, %lazy.any, !dbg !23
  br i1 %lazy.take, label %lazy.slow, label %lazy.join, !dbg !23

lazy.slow:                                        ; preds = %110
  %lazy.positive = call float @llvm.maxnum.f32(float %531, float 0x0000000000000000), !dbg !23
  %lazy.delta = select i1 %lazy.first, float %531, float %lazy.positive, !dbg !23
  %lazy.negated = fsub float 0x0000000000000000, %lazy.delta, !dbg !23
  %lazy.exp = call float @llvm.amdgcn.exp2.f32(float %lazy.negated), !dbg !23
  %lazy.alpha = select i1 %lazy.first, float 0x3FF0000000000000, float %lazy.exp, !dbg !23
  %lazy.positive4882 = call float @llvm.maxnum.f32(float %571, float 0x0000000000000000), !dbg !23
  %lazy.delta4883 = select i1 %lazy.first, float %571, float %lazy.positive4882, !dbg !23
  %lazy.negated4884 = fsub float 0x0000000000000000, %lazy.delta4883, !dbg !23
  %lazy.exp4885 = call float @llvm.amdgcn.exp2.f32(float %lazy.negated4884), !dbg !23
  %lazy.alpha4886 = select i1 %lazy.first, float 0x3FF0000000000000, float %lazy.exp4885, !dbg !23
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted = fsub <4 x float> %389, %lazy.delta.v, !dbg !23
  %lazy.delta.v.lane4887 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v4888 = shufflevector <1 x float> %lazy.delta.v.lane4887, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4889 = fsub <4 x float> %396, %lazy.delta.v4888, !dbg !23
  %lazy.delta.v.lane4890 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v4891 = shufflevector <1 x float> %lazy.delta.v.lane4890, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4892 = fsub <4 x float> %410, %lazy.delta.v4891, !dbg !23
  %lazy.delta.v.lane4893 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v4894 = shufflevector <1 x float> %lazy.delta.v.lane4893, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4895 = fsub <4 x float> %424, %lazy.delta.v4894, !dbg !23
  %lazy.delta.v.lane4896 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v4897 = shufflevector <1 x float> %lazy.delta.v.lane4896, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4898 = fsub <4 x float> %434, %lazy.delta.v4897, !dbg !23
  %lazy.delta.v.lane4899 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v4900 = shufflevector <1 x float> %lazy.delta.v.lane4899, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4901 = fsub <4 x float> %444, %lazy.delta.v4900, !dbg !23
  %lazy.delta.v.lane4902 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v4903 = shufflevector <1 x float> %lazy.delta.v.lane4902, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4904 = fsub <4 x float> %454, %lazy.delta.v4903, !dbg !23
  %lazy.delta.v.lane4905 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v4906 = shufflevector <1 x float> %lazy.delta.v.lane4905, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4907 = fsub <4 x float> %464, %lazy.delta.v4906, !dbg !23
  %lazy.delta.v.lane4908 = insertelement <1 x float> poison, float %lazy.delta4883, i64 0, !dbg !23
  %lazy.delta.v4909 = shufflevector <1 x float> %lazy.delta.v.lane4908, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4910 = fsub <4 x float> %477, %lazy.delta.v4909, !dbg !23
  %lazy.delta.v.lane4911 = insertelement <1 x float> poison, float %lazy.delta4883, i64 0, !dbg !23
  %lazy.delta.v4912 = shufflevector <1 x float> %lazy.delta.v.lane4911, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4913 = fsub <4 x float> %483, %lazy.delta.v4912, !dbg !23
  %lazy.delta.v.lane4914 = insertelement <1 x float> poison, float %lazy.delta4883, i64 0, !dbg !23
  %lazy.delta.v4915 = shufflevector <1 x float> %lazy.delta.v.lane4914, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4916 = fsub <4 x float> %489, %lazy.delta.v4915, !dbg !23
  %lazy.delta.v.lane4917 = insertelement <1 x float> poison, float %lazy.delta4883, i64 0, !dbg !23
  %lazy.delta.v4918 = shufflevector <1 x float> %lazy.delta.v.lane4917, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4919 = fsub <4 x float> %495, %lazy.delta.v4918, !dbg !23
  %lazy.delta.v.lane4920 = insertelement <1 x float> poison, float %lazy.delta4883, i64 0, !dbg !23
  %lazy.delta.v4921 = shufflevector <1 x float> %lazy.delta.v.lane4920, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4922 = fsub <4 x float> %501, %lazy.delta.v4921, !dbg !23
  %lazy.delta.v.lane4923 = insertelement <1 x float> poison, float %lazy.delta4883, i64 0, !dbg !23
  %lazy.delta.v4924 = shufflevector <1 x float> %lazy.delta.v.lane4923, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4925 = fsub <4 x float> %507, %lazy.delta.v4924, !dbg !23
  %lazy.delta.v.lane4926 = insertelement <1 x float> poison, float %lazy.delta4883, i64 0, !dbg !23
  %lazy.delta.v4927 = shufflevector <1 x float> %lazy.delta.v.lane4926, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4928 = fsub <4 x float> %513, %lazy.delta.v4927, !dbg !23
  %lazy.delta.v.lane4929 = insertelement <1 x float> poison, float %lazy.delta4883, i64 0, !dbg !23
  %lazy.delta.v4930 = shufflevector <1 x float> %lazy.delta.v.lane4929, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted4931 = fsub <4 x float> %519, %lazy.delta.v4930, !dbg !23
  %lazy.state = extractvalue [2 x <1 x float>] %104, 0, !dbg !23
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !23
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !23
  %lazy.state4932 = extractvalue [2 x <1 x float>] %104, 1, !dbg !23
  %lazy.alpha.v.lane4933 = insertelement <1 x float> poison, float %lazy.alpha4886, i64 0, !dbg !23
  %lazy.scaled4934 = fmul <1 x float> %lazy.state4932, %lazy.alpha.v.lane4933, !dbg !23
  %lazy.state.next4935 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled4934, 1, !dbg !23
  %lazy.state4936 = extractvalue [2 x <8 x float>] %108, 0, !dbg !23
  %lazy.alpha.v.lane4937 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane4937, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled4938 = fmul <8 x float> %lazy.state4936, %lazy.alpha.v, !dbg !23
  %lazy.state.next4939 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled4938, 0, !dbg !23
  %lazy.state4940 = extractvalue [2 x <8 x float>] %108, 1, !dbg !23
  %lazy.alpha.v.lane4941 = insertelement <1 x float> poison, float %lazy.alpha4886, i64 0, !dbg !23
  %lazy.alpha.v4942 = shufflevector <1 x float> %lazy.alpha.v.lane4941, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled4943 = fmul <8 x float> %lazy.state4940, %lazy.alpha.v4942, !dbg !23
  %lazy.state.next4944 = insertvalue [2 x <8 x float>] %lazy.state.next4939, <8 x float> %lazy.scaled4943, 1, !dbg !23
  %lazy.state4945 = extractvalue [2 x <8 x float>] %107, 0, !dbg !23
  %lazy.alpha.v.lane4946 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v4947 = shufflevector <1 x float> %lazy.alpha.v.lane4946, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled4948 = fmul <8 x float> %lazy.state4945, %lazy.alpha.v4947, !dbg !23
  %lazy.state.next4949 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled4948, 0, !dbg !23
  %lazy.state4950 = extractvalue [2 x <8 x float>] %107, 1, !dbg !23
  %lazy.alpha.v.lane4951 = insertelement <1 x float> poison, float %lazy.alpha4886, i64 0, !dbg !23
  %lazy.alpha.v4952 = shufflevector <1 x float> %lazy.alpha.v.lane4951, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled4953 = fmul <8 x float> %lazy.state4950, %lazy.alpha.v4952, !dbg !23
  %lazy.state.next4954 = insertvalue [2 x <8 x float>] %lazy.state.next4949, <8 x float> %lazy.scaled4953, 1, !dbg !23
  %lazy.state4955 = extractvalue [2 x <8 x float>] %106, 0, !dbg !23
  %lazy.alpha.v.lane4956 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v4957 = shufflevector <1 x float> %lazy.alpha.v.lane4956, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled4958 = fmul <8 x float> %lazy.state4955, %lazy.alpha.v4957, !dbg !23
  %lazy.state.next4959 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled4958, 0, !dbg !23
  %lazy.state4960 = extractvalue [2 x <8 x float>] %106, 1, !dbg !23
  %lazy.alpha.v.lane4961 = insertelement <1 x float> poison, float %lazy.alpha4886, i64 0, !dbg !23
  %lazy.alpha.v4962 = shufflevector <1 x float> %lazy.alpha.v.lane4961, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled4963 = fmul <8 x float> %lazy.state4960, %lazy.alpha.v4962, !dbg !23
  %lazy.state.next4964 = insertvalue [2 x <8 x float>] %lazy.state.next4959, <8 x float> %lazy.scaled4963, 1, !dbg !23
  %lazy.state4965 = extractvalue [2 x <8 x float>] %103, 0, !dbg !23
  %lazy.alpha.v.lane4966 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v4967 = shufflevector <1 x float> %lazy.alpha.v.lane4966, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled4968 = fmul <8 x float> %lazy.state4965, %lazy.alpha.v4967, !dbg !23
  %lazy.state.next4969 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled4968, 0, !dbg !23
  %lazy.state4970 = extractvalue [2 x <8 x float>] %103, 1, !dbg !23
  %lazy.alpha.v.lane4971 = insertelement <1 x float> poison, float %lazy.alpha4886, i64 0, !dbg !23
  %lazy.alpha.v4972 = shufflevector <1 x float> %lazy.alpha.v.lane4971, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled4973 = fmul <8 x float> %lazy.state4970, %lazy.alpha.v4972, !dbg !23
  %lazy.state.next4974 = insertvalue [2 x <8 x float>] %lazy.state.next4969, <8 x float> %lazy.scaled4973, 1, !dbg !23
  %lazy.base.old = extractvalue [2 x <1 x float>] %105, 0, !dbg !23
  %lazy.delta.b.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.base.next = fsub <1 x float> %lazy.base.old, %lazy.delta.b.lane, !dbg !23
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !23
  %lazy.base.old4975 = extractvalue [2 x <1 x float>] %105, 1, !dbg !23
  %lazy.delta.b.lane4976 = insertelement <1 x float> poison, float %lazy.delta4883, i64 0, !dbg !23
  %lazy.base.next4977 = fsub <1 x float> %lazy.base.old4975, %lazy.delta.b.lane4976, !dbg !23
  %lazy.base.state4978 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next4977, 1, !dbg !23
  br label %lazy.join, !dbg !23

lazy.join:                                        ; preds = %lazy.slow, %110
  %lazy.score = phi <4 x float> [ %389, %110 ], [ %lazy.shifted, %lazy.slow ], !dbg !23
  %lazy.score4979 = phi <4 x float> [ %396, %110 ], [ %lazy.shifted4889, %lazy.slow ], !dbg !23
  %lazy.score4980 = phi <4 x float> [ %410, %110 ], [ %lazy.shifted4892, %lazy.slow ], !dbg !23
  %lazy.score4981 = phi <4 x float> [ %424, %110 ], [ %lazy.shifted4895, %lazy.slow ], !dbg !23
  %lazy.score4982 = phi <4 x float> [ %434, %110 ], [ %lazy.shifted4898, %lazy.slow ], !dbg !23
  %lazy.score4983 = phi <4 x float> [ %444, %110 ], [ %lazy.shifted4901, %lazy.slow ], !dbg !23
  %lazy.score4984 = phi <4 x float> [ %454, %110 ], [ %lazy.shifted4904, %lazy.slow ], !dbg !23
  %lazy.score4985 = phi <4 x float> [ %464, %110 ], [ %lazy.shifted4907, %lazy.slow ], !dbg !23
  %lazy.score4986 = phi <4 x float> [ %477, %110 ], [ %lazy.shifted4910, %lazy.slow ], !dbg !23
  %lazy.score4987 = phi <4 x float> [ %483, %110 ], [ %lazy.shifted4913, %lazy.slow ], !dbg !23
  %lazy.score4988 = phi <4 x float> [ %489, %110 ], [ %lazy.shifted4916, %lazy.slow ], !dbg !23
  %lazy.score4989 = phi <4 x float> [ %495, %110 ], [ %lazy.shifted4919, %lazy.slow ], !dbg !23
  %lazy.score4990 = phi <4 x float> [ %501, %110 ], [ %lazy.shifted4922, %lazy.slow ], !dbg !23
  %lazy.score4991 = phi <4 x float> [ %507, %110 ], [ %lazy.shifted4925, %lazy.slow ], !dbg !23
  %lazy.score4992 = phi <4 x float> [ %513, %110 ], [ %lazy.shifted4928, %lazy.slow ], !dbg !23
  %lazy.score4993 = phi <4 x float> [ %519, %110 ], [ %lazy.shifted4931, %lazy.slow ], !dbg !23
  %lazy.state4994 = phi [2 x <1 x float>] [ %104, %110 ], [ %lazy.state.next4935, %lazy.slow ], !dbg !23
  %lazy.state4995 = phi [2 x <8 x float>] [ %108, %110 ], [ %lazy.state.next4944, %lazy.slow ], !dbg !23
  %lazy.state4996 = phi [2 x <8 x float>] [ %107, %110 ], [ %lazy.state.next4954, %lazy.slow ], !dbg !23
  %lazy.state4997 = phi [2 x <8 x float>] [ %106, %110 ], [ %lazy.state.next4964, %lazy.slow ], !dbg !23
  %lazy.state4998 = phi [2 x <8 x float>] [ %103, %110 ], [ %lazy.state.next4974, %lazy.slow ], !dbg !23
  %lazy.base.join = phi [2 x <1 x float>] [ %105, %110 ], [ %lazy.base.state4978, %lazy.slow ], !dbg !23
  %572 = extractvalue [2 x <1 x float>] %lazy.state4994, 0, !dbg !23
  %573 = extractelement <1 x float> %572, i64 0, !dbg !23
  %574 = extractvalue [2 x <1 x float>] %lazy.state4994, 1, !dbg !23
  %575 = extractelement <1 x float> %574, i64 0, !dbg !23
  %576 = extractvalue [2 x <8 x float>] %lazy.state4998, 0, !dbg !27
  %577 = shufflevector <8 x float> %576, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %578 = shufflevector <8 x float> %576, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %579 = extractvalue [2 x <8 x float>] %lazy.state4998, 1, !dbg !27
  %580 = shufflevector <8 x float> %579, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %581 = shufflevector <8 x float> %579, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.local3862 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local3188, !dbg !19
  %address.local3863 = ptrtoint ptr addrspace(3) %address.local3862 to i64, !dbg !19
  %address.local3864 = trunc i64 %address.local3863 to i32, !dbg !19
  %582 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3864), !dbg !19
  %.idx47054707 = or disjoint i64 %193, %address.local3194.idx4700, !dbg !19
  %.idx47054707.tr = trunc nuw i64 %.idx47054707 to i32, !dbg !19
  %dma.buffer.offset4569 = shl nuw nsw i32 %.idx47054707.tr, 1, !dbg !19
  %583 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %582, i32 %dma.buffer.offset4569, <4 x i32> %dma.buffer.rsrc4574) #10, !dbg !19
  %address.local3874 = getelementptr i8, ptr addrspace(3) %address.local3862, i64 4096, !dbg !19
  %address.local3875 = ptrtoint ptr addrspace(3) %address.local3874 to i64, !dbg !19
  %address.local3876 = trunc i64 %address.local3875 to i32, !dbg !19
  %584 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3876), !dbg !19
  %.idx47084709 = or disjoint i64 %193, %address.local3193, !dbg !19
  %585 = or disjoint i64 %.idx47084709, %address.local3196, !dbg !19
  %.tr4710 = trunc nuw i64 %585 to i32, !dbg !19
  %586 = shl i32 %.tr4710, 1, !dbg !19
  %dma.buffer.offset4567 = or i32 %586, 32, !dbg !19
  %587 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %584, i32 %dma.buffer.offset4567, <4 x i32> %dma.buffer.rsrc4574) #10, !dbg !19
  %address.local3883 = shl nuw nsw i32 %address.thread1132, 3, !dbg !14
  %address.local3884 = and i32 %address.local3883, 384, !dbg !14
  %address.local3886 = shl nuw nsw i32 %address.thread1132, 10, !dbg !14
  %address.local3887 = and i32 %address.local3886, 3072, !dbg !14
  %588 = or disjoint i32 %address.local3884, %address.local3887, !dbg !14
  %address.local3888 = or disjoint i32 %588, %address.local1247, !dbg !14
  %address.local3889 = zext nneg i32 %address.local3888 to i64, !dbg !14
  %address.local3890 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local3889, !dbg !14
  %589 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3890, i32 0) #0, !dbg !14
  %590 = bitcast <4 x i32> %589 to <8 x half>, !dbg !14
  %address.local3901 = getelementptr i8, ptr addrspace(3) %address.local3890, i64 64, !dbg !14
  %591 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3901, i32 0) #0, !dbg !14
  %592 = bitcast <4 x i32> %591 to <8 x half>, !dbg !14
  %address.local3912 = getelementptr i8, ptr addrspace(3) %address.local3890, i64 128, !dbg !14
  %593 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3912, i32 0) #0, !dbg !14
  %594 = bitcast <4 x i32> %593 to <8 x half>, !dbg !14
  %address.local3923 = getelementptr i8, ptr addrspace(3) %address.local3890, i64 192, !dbg !14
  %595 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3923, i32 0) #0, !dbg !14
  %596 = bitcast <4 x i32> %595 to <8 x half>, !dbg !14
  %address.local3934 = getelementptr i8, ptr addrspace(3) %address.local3890, i64 1024, !dbg !14
  %597 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3934, i32 0) #0, !dbg !14
  %598 = bitcast <4 x i32> %597 to <8 x half>, !dbg !14
  %address.local3945 = getelementptr i8, ptr addrspace(3) %address.local3890, i64 1088, !dbg !14
  %599 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3945, i32 0) #0, !dbg !14
  %600 = bitcast <4 x i32> %599 to <8 x half>, !dbg !14
  %address.local3956 = getelementptr i8, ptr addrspace(3) %address.local3890, i64 1152, !dbg !14
  %601 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3956, i32 0) #0, !dbg !14
  %602 = bitcast <4 x i32> %601 to <8 x half>, !dbg !14
  %address.local3967 = getelementptr i8, ptr addrspace(3) %address.local3890, i64 1216, !dbg !14
  %603 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3967, i32 0) #0, !dbg !14
  %604 = shufflevector <8 x half> %590, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %605 = extractelement <4 x float> %lazy.score, i64 0, !dbg !24
  %606 = call float @llvm.amdgcn.exp2.f32(float %605), !dbg !24
  %607 = insertelement <4 x float> poison, float %606, i64 0, !dbg !26
  %608 = extractelement <4 x float> %lazy.score, i64 1, !dbg !24
  %609 = call float @llvm.amdgcn.exp2.f32(float %608), !dbg !24
  %610 = insertelement <4 x float> %607, float %609, i64 1, !dbg !26
  %611 = extractelement <4 x float> %lazy.score, i64 2, !dbg !24
  %612 = call float @llvm.amdgcn.exp2.f32(float %611), !dbg !24
  %613 = insertelement <4 x float> %610, float %612, i64 2, !dbg !26
  %614 = extractelement <4 x float> %lazy.score, i64 3, !dbg !24
  %615 = call float @llvm.amdgcn.exp2.f32(float %614), !dbg !24
  %616 = insertelement <4 x float> %613, float %615, i64 3, !dbg !26
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %616) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i32 0, !dbg !26
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i32 1, !dbg !26
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i32 2, !dbg !26
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i32 3, !dbg !26
  %.cvtpack.p0 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1), !dbg !26
  %.cvtpack.p1 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3), !dbg !26
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i32 0, !dbg !26
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i32 1, !dbg !26
  %617 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !26
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %617) #9, !dbg !24, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %604, <4 x float> %577) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %619 = shufflevector <8 x half> %590, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %619, <4 x float> %578) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %621 = extractelement <4 x float> %lazy.score4986, i64 0, !dbg !24
  %622 = call float @llvm.amdgcn.exp2.f32(float %621), !dbg !24
  %623 = insertelement <4 x float> poison, float %622, i64 0, !dbg !26
  %624 = extractelement <4 x float> %lazy.score4986, i64 1, !dbg !24
  %625 = call float @llvm.amdgcn.exp2.f32(float %624), !dbg !24
  %626 = insertelement <4 x float> %623, float %625, i64 1, !dbg !26
  %627 = extractelement <4 x float> %lazy.score4986, i64 2, !dbg !24
  %628 = call float @llvm.amdgcn.exp2.f32(float %627), !dbg !24
  %629 = insertelement <4 x float> %626, float %628, i64 2, !dbg !26
  %630 = extractelement <4 x float> %lazy.score4986, i64 3, !dbg !24
  %631 = call float @llvm.amdgcn.exp2.f32(float %630), !dbg !24
  %632 = insertelement <4 x float> %629, float %631, i64 3, !dbg !26
  %math.result858 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %632) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f04999 = extractelement <4 x float> %math.result858, i32 0, !dbg !26
  %.cvtpack.f15000 = extractelement <4 x float> %math.result858, i32 1, !dbg !26
  %.cvtpack.f25001 = extractelement <4 x float> %math.result858, i32 2, !dbg !26
  %.cvtpack.f35002 = extractelement <4 x float> %math.result858, i32 3, !dbg !26
  %.cvtpack.p05003 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04999, float %.cvtpack.f15000), !dbg !26
  %.cvtpack.p15004 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25001, float %.cvtpack.f35002), !dbg !26
  %.cvtpack.v05005 = insertelement <2 x i32> poison, i32 %.cvtpack.p05003, i32 0, !dbg !26
  %.cvtpack.v15006 = insertelement <2 x i32> %.cvtpack.v05005, i32 %.cvtpack.p15004, i32 1, !dbg !26
  %633 = bitcast <2 x i32> %.cvtpack.v15006 to <4 x half>, !dbg !26
  %math.packed859 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %633) #9, !dbg !24, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %604, <4 x float> %580) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %619, <4 x float> %581) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %636 = shufflevector <8 x half> %592, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %637 = extractelement <4 x float> %lazy.score4979, i64 0, !dbg !24
  %638 = call float @llvm.amdgcn.exp2.f32(float %637), !dbg !24
  %639 = fadd float %606, %638, !dbg !25
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %639) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %640 = insertelement <4 x float> poison, float %638, i64 0, !dbg !26
  %641 = extractelement <4 x float> %lazy.score4979, i64 1, !dbg !24
  %642 = call float @llvm.amdgcn.exp2.f32(float %641), !dbg !24
  %643 = fadd float %609, %642, !dbg !25
  %k0.sum.anchor4723 = call float asm sideeffect "", "=v,0"(float %643) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %644 = insertelement <4 x float> %640, float %642, i64 1, !dbg !26
  %645 = extractelement <4 x float> %lazy.score4979, i64 2, !dbg !24
  %646 = call float @llvm.amdgcn.exp2.f32(float %645), !dbg !24
  %647 = fadd float %612, %646, !dbg !25
  %k0.sum.anchor4724 = call float asm sideeffect "", "=v,0"(float %647) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %648 = insertelement <4 x float> %644, float %646, i64 2, !dbg !26
  %649 = extractelement <4 x float> %lazy.score4979, i64 3, !dbg !24
  %650 = call float @llvm.amdgcn.exp2.f32(float %649), !dbg !24
  %651 = fadd float %615, %650, !dbg !25
  %k0.sum.anchor4725 = call float asm sideeffect "", "=v,0"(float %651) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %652 = insertelement <4 x float> %648, float %650, i64 3, !dbg !26
  %math.result844 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %652) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f05007 = extractelement <4 x float> %math.result844, i32 0, !dbg !26
  %.cvtpack.f15008 = extractelement <4 x float> %math.result844, i32 1, !dbg !26
  %.cvtpack.f25009 = extractelement <4 x float> %math.result844, i32 2, !dbg !26
  %.cvtpack.f35010 = extractelement <4 x float> %math.result844, i32 3, !dbg !26
  %.cvtpack.p05011 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05007, float %.cvtpack.f15008), !dbg !26
  %.cvtpack.p15012 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25009, float %.cvtpack.f35010), !dbg !26
  %.cvtpack.v05013 = insertelement <2 x i32> poison, i32 %.cvtpack.p05011, i32 0, !dbg !26
  %.cvtpack.v15014 = insertelement <2 x i32> %.cvtpack.v05013, i32 %.cvtpack.p15012, i32 1, !dbg !26
  %653 = bitcast <2 x i32> %.cvtpack.v15014 to <4 x half>, !dbg !26
  %math.packed845 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %653) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %636, <4 x float> %618) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %655 = shufflevector <8 x half> %592, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %655, <4 x float> %620) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %657 = extractelement <4 x float> %lazy.score4987, i64 0, !dbg !24
  %658 = call float @llvm.amdgcn.exp2.f32(float %657), !dbg !24
  %659 = fadd float %622, %658, !dbg !25
  %k0.sum.anchor4726 = call float asm sideeffect "", "=v,0"(float %659) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %660 = insertelement <4 x float> poison, float %658, i64 0, !dbg !26
  %661 = extractelement <4 x float> %lazy.score4987, i64 1, !dbg !24
  %662 = call float @llvm.amdgcn.exp2.f32(float %661), !dbg !24
  %663 = fadd float %625, %662, !dbg !25
  %k0.sum.anchor4727 = call float asm sideeffect "", "=v,0"(float %663) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %664 = insertelement <4 x float> %660, float %662, i64 1, !dbg !26
  %665 = extractelement <4 x float> %lazy.score4987, i64 2, !dbg !24
  %666 = call float @llvm.amdgcn.exp2.f32(float %665), !dbg !24
  %667 = fadd float %628, %666, !dbg !25
  %k0.sum.anchor4728 = call float asm sideeffect "", "=v,0"(float %667) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %668 = insertelement <4 x float> %664, float %666, i64 2, !dbg !26
  %669 = extractelement <4 x float> %lazy.score4987, i64 3, !dbg !24
  %670 = call float @llvm.amdgcn.exp2.f32(float %669), !dbg !24
  %671 = fadd float %631, %670, !dbg !25
  %k0.sum.anchor4729 = call float asm sideeffect "", "=v,0"(float %671) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %672 = insertelement <4 x float> %668, float %670, i64 3, !dbg !26
  %math.result860 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %672) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f05015 = extractelement <4 x float> %math.result860, i32 0, !dbg !26
  %.cvtpack.f15016 = extractelement <4 x float> %math.result860, i32 1, !dbg !26
  %.cvtpack.f25017 = extractelement <4 x float> %math.result860, i32 2, !dbg !26
  %.cvtpack.f35018 = extractelement <4 x float> %math.result860, i32 3, !dbg !26
  %.cvtpack.p05019 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05015, float %.cvtpack.f15016), !dbg !26
  %.cvtpack.p15020 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25017, float %.cvtpack.f35018), !dbg !26
  %.cvtpack.v05021 = insertelement <2 x i32> poison, i32 %.cvtpack.p05019, i32 0, !dbg !26
  %.cvtpack.v15022 = insertelement <2 x i32> %.cvtpack.v05021, i32 %.cvtpack.p15020, i32 1, !dbg !26
  %673 = bitcast <2 x i32> %.cvtpack.v15022 to <4 x half>, !dbg !26
  %math.packed861 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %673) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %636, <4 x float> %634) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %655, <4 x float> %635) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %676 = shufflevector <8 x half> %594, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %677 = extractelement <4 x float> %lazy.score4980, i64 0, !dbg !24
  %678 = call float @llvm.amdgcn.exp2.f32(float %677), !dbg !24
  %679 = fadd float %k0.sum.anchor, %678, !dbg !25
  %k0.sum.anchor4730 = call float asm sideeffect "", "=v,0"(float %679) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %680 = insertelement <4 x float> poison, float %678, i64 0, !dbg !26
  %681 = extractelement <4 x float> %lazy.score4980, i64 1, !dbg !24
  %682 = call float @llvm.amdgcn.exp2.f32(float %681), !dbg !24
  %683 = fadd float %k0.sum.anchor4723, %682, !dbg !25
  %k0.sum.anchor4731 = call float asm sideeffect "", "=v,0"(float %683) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %684 = insertelement <4 x float> %680, float %682, i64 1, !dbg !26
  %685 = extractelement <4 x float> %lazy.score4980, i64 2, !dbg !24
  %686 = call float @llvm.amdgcn.exp2.f32(float %685), !dbg !24
  %687 = fadd float %k0.sum.anchor4724, %686, !dbg !25
  %k0.sum.anchor4732 = call float asm sideeffect "", "=v,0"(float %687) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %688 = insertelement <4 x float> %684, float %686, i64 2, !dbg !26
  %689 = extractelement <4 x float> %lazy.score4980, i64 3, !dbg !24
  %690 = call float @llvm.amdgcn.exp2.f32(float %689), !dbg !24
  %691 = fadd float %k0.sum.anchor4725, %690, !dbg !25
  %k0.sum.anchor4733 = call float asm sideeffect "", "=v,0"(float %691) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %692 = insertelement <4 x float> %688, float %690, i64 3, !dbg !26
  %math.result846 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %692) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f05023 = extractelement <4 x float> %math.result846, i32 0, !dbg !26
  %.cvtpack.f15024 = extractelement <4 x float> %math.result846, i32 1, !dbg !26
  %.cvtpack.f25025 = extractelement <4 x float> %math.result846, i32 2, !dbg !26
  %.cvtpack.f35026 = extractelement <4 x float> %math.result846, i32 3, !dbg !26
  %.cvtpack.p05027 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05023, float %.cvtpack.f15024), !dbg !26
  %.cvtpack.p15028 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25025, float %.cvtpack.f35026), !dbg !26
  %.cvtpack.v05029 = insertelement <2 x i32> poison, i32 %.cvtpack.p05027, i32 0, !dbg !26
  %.cvtpack.v15030 = insertelement <2 x i32> %.cvtpack.v05029, i32 %.cvtpack.p15028, i32 1, !dbg !26
  %693 = bitcast <2 x i32> %.cvtpack.v15030 to <4 x half>, !dbg !26
  %math.packed847 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %693) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %676, <4 x float> %654) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %695 = shufflevector <8 x half> %594, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %695, <4 x float> %656) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %697 = extractelement <4 x float> %lazy.score4988, i64 0, !dbg !24
  %698 = call float @llvm.amdgcn.exp2.f32(float %697), !dbg !24
  %699 = fadd float %k0.sum.anchor4726, %698, !dbg !25
  %k0.sum.anchor4734 = call float asm sideeffect "", "=v,0"(float %699) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %700 = insertelement <4 x float> poison, float %698, i64 0, !dbg !26
  %701 = extractelement <4 x float> %lazy.score4988, i64 1, !dbg !24
  %702 = call float @llvm.amdgcn.exp2.f32(float %701), !dbg !24
  %703 = fadd float %k0.sum.anchor4727, %702, !dbg !25
  %k0.sum.anchor4735 = call float asm sideeffect "", "=v,0"(float %703) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %704 = insertelement <4 x float> %700, float %702, i64 1, !dbg !26
  %705 = extractelement <4 x float> %lazy.score4988, i64 2, !dbg !24
  %706 = call float @llvm.amdgcn.exp2.f32(float %705), !dbg !24
  %707 = fadd float %k0.sum.anchor4728, %706, !dbg !25
  %k0.sum.anchor4736 = call float asm sideeffect "", "=v,0"(float %707) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %708 = insertelement <4 x float> %704, float %706, i64 2, !dbg !26
  %709 = extractelement <4 x float> %lazy.score4988, i64 3, !dbg !24
  %710 = call float @llvm.amdgcn.exp2.f32(float %709), !dbg !24
  %711 = fadd float %k0.sum.anchor4729, %710, !dbg !25
  %k0.sum.anchor4737 = call float asm sideeffect "", "=v,0"(float %711) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %712 = insertelement <4 x float> %708, float %710, i64 3, !dbg !26
  %math.result862 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %712) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f05031 = extractelement <4 x float> %math.result862, i32 0, !dbg !26
  %.cvtpack.f15032 = extractelement <4 x float> %math.result862, i32 1, !dbg !26
  %.cvtpack.f25033 = extractelement <4 x float> %math.result862, i32 2, !dbg !26
  %.cvtpack.f35034 = extractelement <4 x float> %math.result862, i32 3, !dbg !26
  %.cvtpack.p05035 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05031, float %.cvtpack.f15032), !dbg !26
  %.cvtpack.p15036 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25033, float %.cvtpack.f35034), !dbg !26
  %.cvtpack.v05037 = insertelement <2 x i32> poison, i32 %.cvtpack.p05035, i32 0, !dbg !26
  %.cvtpack.v15038 = insertelement <2 x i32> %.cvtpack.v05037, i32 %.cvtpack.p15036, i32 1, !dbg !26
  %713 = bitcast <2 x i32> %.cvtpack.v15038 to <4 x half>, !dbg !26
  %math.packed863 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %713) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %714 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %676, <4 x float> %674) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %715 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %695, <4 x float> %675) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %716 = shufflevector <8 x half> %596, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %717 = extractelement <4 x float> %lazy.score4981, i64 0, !dbg !24
  %718 = call float @llvm.amdgcn.exp2.f32(float %717), !dbg !24
  %719 = fadd float %k0.sum.anchor4730, %718, !dbg !25
  %k0.sum.anchor4738 = call float asm sideeffect "", "=v,0"(float %719) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %720 = insertelement <4 x float> poison, float %718, i64 0, !dbg !26
  %721 = extractelement <4 x float> %lazy.score4981, i64 1, !dbg !24
  %722 = call float @llvm.amdgcn.exp2.f32(float %721), !dbg !24
  %723 = fadd float %k0.sum.anchor4731, %722, !dbg !25
  %k0.sum.anchor4739 = call float asm sideeffect "", "=v,0"(float %723) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %724 = insertelement <4 x float> %720, float %722, i64 1, !dbg !26
  %725 = extractelement <4 x float> %lazy.score4981, i64 2, !dbg !24
  %726 = call float @llvm.amdgcn.exp2.f32(float %725), !dbg !24
  %727 = fadd float %k0.sum.anchor4732, %726, !dbg !25
  %k0.sum.anchor4740 = call float asm sideeffect "", "=v,0"(float %727) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %728 = insertelement <4 x float> %724, float %726, i64 2, !dbg !26
  %729 = extractelement <4 x float> %lazy.score4981, i64 3, !dbg !24
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !24
  %731 = fadd float %k0.sum.anchor4733, %730, !dbg !25
  %k0.sum.anchor4741 = call float asm sideeffect "", "=v,0"(float %731) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %732 = insertelement <4 x float> %728, float %730, i64 3, !dbg !26
  %math.result848 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %732) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f05039 = extractelement <4 x float> %math.result848, i32 0, !dbg !26
  %.cvtpack.f15040 = extractelement <4 x float> %math.result848, i32 1, !dbg !26
  %.cvtpack.f25041 = extractelement <4 x float> %math.result848, i32 2, !dbg !26
  %.cvtpack.f35042 = extractelement <4 x float> %math.result848, i32 3, !dbg !26
  %.cvtpack.p05043 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05039, float %.cvtpack.f15040), !dbg !26
  %.cvtpack.p15044 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25041, float %.cvtpack.f35042), !dbg !26
  %.cvtpack.v05045 = insertelement <2 x i32> poison, i32 %.cvtpack.p05043, i32 0, !dbg !26
  %.cvtpack.v15046 = insertelement <2 x i32> %.cvtpack.v05045, i32 %.cvtpack.p15044, i32 1, !dbg !26
  %733 = bitcast <2 x i32> %.cvtpack.v15046 to <4 x half>, !dbg !26
  %math.packed849 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %733) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %734 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %716, <4 x float> %694) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %735 = shufflevector <8 x half> %596, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %736 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %735, <4 x float> %696) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %737 = extractelement <4 x float> %lazy.score4989, i64 0, !dbg !24
  %738 = call float @llvm.amdgcn.exp2.f32(float %737), !dbg !24
  %739 = fadd float %k0.sum.anchor4734, %738, !dbg !25
  %k0.sum.anchor4742 = call float asm sideeffect "", "=v,0"(float %739) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %740 = insertelement <4 x float> poison, float %738, i64 0, !dbg !26
  %741 = extractelement <4 x float> %lazy.score4989, i64 1, !dbg !24
  %742 = call float @llvm.amdgcn.exp2.f32(float %741), !dbg !24
  %743 = fadd float %k0.sum.anchor4735, %742, !dbg !25
  %k0.sum.anchor4743 = call float asm sideeffect "", "=v,0"(float %743) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %744 = insertelement <4 x float> %740, float %742, i64 1, !dbg !26
  %745 = extractelement <4 x float> %lazy.score4989, i64 2, !dbg !24
  %746 = call float @llvm.amdgcn.exp2.f32(float %745), !dbg !24
  %747 = fadd float %k0.sum.anchor4736, %746, !dbg !25
  %k0.sum.anchor4744 = call float asm sideeffect "", "=v,0"(float %747) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %748 = insertelement <4 x float> %744, float %746, i64 2, !dbg !26
  %749 = extractelement <4 x float> %lazy.score4989, i64 3, !dbg !24
  %750 = call float @llvm.amdgcn.exp2.f32(float %749), !dbg !24
  %751 = fadd float %k0.sum.anchor4737, %750, !dbg !25
  %k0.sum.anchor4745 = call float asm sideeffect "", "=v,0"(float %751) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %752 = insertelement <4 x float> %748, float %750, i64 3, !dbg !26
  %math.result864 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %752) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f05047 = extractelement <4 x float> %math.result864, i32 0, !dbg !26
  %.cvtpack.f15048 = extractelement <4 x float> %math.result864, i32 1, !dbg !26
  %.cvtpack.f25049 = extractelement <4 x float> %math.result864, i32 2, !dbg !26
  %.cvtpack.f35050 = extractelement <4 x float> %math.result864, i32 3, !dbg !26
  %.cvtpack.p05051 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05047, float %.cvtpack.f15048), !dbg !26
  %.cvtpack.p15052 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25049, float %.cvtpack.f35050), !dbg !26
  %.cvtpack.v05053 = insertelement <2 x i32> poison, i32 %.cvtpack.p05051, i32 0, !dbg !26
  %.cvtpack.v15054 = insertelement <2 x i32> %.cvtpack.v05053, i32 %.cvtpack.p15052, i32 1, !dbg !26
  %753 = bitcast <2 x i32> %.cvtpack.v15054 to <4 x half>, !dbg !26
  %math.packed865 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %753) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %754 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %716, <4 x float> %714) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %735, <4 x float> %715) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %756 = shufflevector <8 x half> %598, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %757 = extractelement <4 x float> %lazy.score4982, i64 0, !dbg !24
  %758 = call float @llvm.amdgcn.exp2.f32(float %757), !dbg !24
  %759 = fadd float %k0.sum.anchor4738, %758, !dbg !25
  %k0.sum.anchor4746 = call float asm sideeffect "", "=v,0"(float %759) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %760 = insertelement <4 x float> poison, float %758, i64 0, !dbg !26
  %761 = extractelement <4 x float> %lazy.score4982, i64 1, !dbg !24
  %762 = call float @llvm.amdgcn.exp2.f32(float %761), !dbg !24
  %763 = fadd float %k0.sum.anchor4739, %762, !dbg !25
  %k0.sum.anchor4747 = call float asm sideeffect "", "=v,0"(float %763) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %764 = insertelement <4 x float> %760, float %762, i64 1, !dbg !26
  %765 = extractelement <4 x float> %lazy.score4982, i64 2, !dbg !24
  %766 = call float @llvm.amdgcn.exp2.f32(float %765), !dbg !24
  %767 = fadd float %k0.sum.anchor4740, %766, !dbg !25
  %k0.sum.anchor4748 = call float asm sideeffect "", "=v,0"(float %767) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %768 = insertelement <4 x float> %764, float %766, i64 2, !dbg !26
  %769 = extractelement <4 x float> %lazy.score4982, i64 3, !dbg !24
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !24
  %771 = fadd float %k0.sum.anchor4741, %770, !dbg !25
  %k0.sum.anchor4749 = call float asm sideeffect "", "=v,0"(float %771) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %772 = insertelement <4 x float> %768, float %770, i64 3, !dbg !26
  %math.result850 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %772) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f05055 = extractelement <4 x float> %math.result850, i32 0, !dbg !26
  %.cvtpack.f15056 = extractelement <4 x float> %math.result850, i32 1, !dbg !26
  %.cvtpack.f25057 = extractelement <4 x float> %math.result850, i32 2, !dbg !26
  %.cvtpack.f35058 = extractelement <4 x float> %math.result850, i32 3, !dbg !26
  %.cvtpack.p05059 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05055, float %.cvtpack.f15056), !dbg !26
  %.cvtpack.p15060 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25057, float %.cvtpack.f35058), !dbg !26
  %.cvtpack.v05061 = insertelement <2 x i32> poison, i32 %.cvtpack.p05059, i32 0, !dbg !26
  %.cvtpack.v15062 = insertelement <2 x i32> %.cvtpack.v05061, i32 %.cvtpack.p15060, i32 1, !dbg !26
  %773 = bitcast <2 x i32> %.cvtpack.v15062 to <4 x half>, !dbg !26
  %math.packed851 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %773) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %774 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %756, <4 x float> %734) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %775 = shufflevector <8 x half> %598, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %776 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %775, <4 x float> %736) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %777 = extractelement <4 x float> %lazy.score4990, i64 0, !dbg !24
  %778 = call float @llvm.amdgcn.exp2.f32(float %777), !dbg !24
  %779 = fadd float %k0.sum.anchor4742, %778, !dbg !25
  %k0.sum.anchor4750 = call float asm sideeffect "", "=v,0"(float %779) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %780 = insertelement <4 x float> poison, float %778, i64 0, !dbg !26
  %781 = extractelement <4 x float> %lazy.score4990, i64 1, !dbg !24
  %782 = call float @llvm.amdgcn.exp2.f32(float %781), !dbg !24
  %783 = fadd float %k0.sum.anchor4743, %782, !dbg !25
  %k0.sum.anchor4751 = call float asm sideeffect "", "=v,0"(float %783) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %784 = insertelement <4 x float> %780, float %782, i64 1, !dbg !26
  %785 = extractelement <4 x float> %lazy.score4990, i64 2, !dbg !24
  %786 = call float @llvm.amdgcn.exp2.f32(float %785), !dbg !24
  %787 = fadd float %k0.sum.anchor4744, %786, !dbg !25
  %k0.sum.anchor4752 = call float asm sideeffect "", "=v,0"(float %787) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %788 = insertelement <4 x float> %784, float %786, i64 2, !dbg !26
  %789 = extractelement <4 x float> %lazy.score4990, i64 3, !dbg !24
  %790 = call float @llvm.amdgcn.exp2.f32(float %789), !dbg !24
  %791 = fadd float %k0.sum.anchor4745, %790, !dbg !25
  %k0.sum.anchor4753 = call float asm sideeffect "", "=v,0"(float %791) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %792 = insertelement <4 x float> %788, float %790, i64 3, !dbg !26
  %math.result866 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %792) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f05063 = extractelement <4 x float> %math.result866, i32 0, !dbg !26
  %.cvtpack.f15064 = extractelement <4 x float> %math.result866, i32 1, !dbg !26
  %.cvtpack.f25065 = extractelement <4 x float> %math.result866, i32 2, !dbg !26
  %.cvtpack.f35066 = extractelement <4 x float> %math.result866, i32 3, !dbg !26
  %.cvtpack.p05067 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05063, float %.cvtpack.f15064), !dbg !26
  %.cvtpack.p15068 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25065, float %.cvtpack.f35066), !dbg !26
  %.cvtpack.v05069 = insertelement <2 x i32> poison, i32 %.cvtpack.p05067, i32 0, !dbg !26
  %.cvtpack.v15070 = insertelement <2 x i32> %.cvtpack.v05069, i32 %.cvtpack.p15068, i32 1, !dbg !26
  %793 = bitcast <2 x i32> %.cvtpack.v15070 to <4 x half>, !dbg !26
  %math.packed867 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %793) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %794 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %756, <4 x float> %754) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %795 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %775, <4 x float> %755) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %796 = shufflevector <8 x half> %600, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %797 = extractelement <4 x float> %lazy.score4983, i64 0, !dbg !24
  %798 = call float @llvm.amdgcn.exp2.f32(float %797), !dbg !24
  %799 = fadd float %k0.sum.anchor4746, %798, !dbg !25
  %k0.sum.anchor4754 = call float asm sideeffect "", "=v,0"(float %799) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %800 = insertelement <4 x float> poison, float %798, i64 0, !dbg !26
  %801 = extractelement <4 x float> %lazy.score4983, i64 1, !dbg !24
  %802 = call float @llvm.amdgcn.exp2.f32(float %801), !dbg !24
  %803 = fadd float %k0.sum.anchor4747, %802, !dbg !25
  %k0.sum.anchor4755 = call float asm sideeffect "", "=v,0"(float %803) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %804 = insertelement <4 x float> %800, float %802, i64 1, !dbg !26
  %805 = extractelement <4 x float> %lazy.score4983, i64 2, !dbg !24
  %806 = call float @llvm.amdgcn.exp2.f32(float %805), !dbg !24
  %807 = fadd float %k0.sum.anchor4748, %806, !dbg !25
  %k0.sum.anchor4756 = call float asm sideeffect "", "=v,0"(float %807) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %808 = insertelement <4 x float> %804, float %806, i64 2, !dbg !26
  %809 = extractelement <4 x float> %lazy.score4983, i64 3, !dbg !24
  %810 = call float @llvm.amdgcn.exp2.f32(float %809), !dbg !24
  %811 = fadd float %k0.sum.anchor4749, %810, !dbg !25
  %k0.sum.anchor4757 = call float asm sideeffect "", "=v,0"(float %811) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %812 = insertelement <4 x float> %808, float %810, i64 3, !dbg !26
  %math.result852 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %812) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f05071 = extractelement <4 x float> %math.result852, i32 0, !dbg !26
  %.cvtpack.f15072 = extractelement <4 x float> %math.result852, i32 1, !dbg !26
  %.cvtpack.f25073 = extractelement <4 x float> %math.result852, i32 2, !dbg !26
  %.cvtpack.f35074 = extractelement <4 x float> %math.result852, i32 3, !dbg !26
  %.cvtpack.p05075 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05071, float %.cvtpack.f15072), !dbg !26
  %.cvtpack.p15076 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25073, float %.cvtpack.f35074), !dbg !26
  %.cvtpack.v05077 = insertelement <2 x i32> poison, i32 %.cvtpack.p05075, i32 0, !dbg !26
  %.cvtpack.v15078 = insertelement <2 x i32> %.cvtpack.v05077, i32 %.cvtpack.p15076, i32 1, !dbg !26
  %813 = bitcast <2 x i32> %.cvtpack.v15078 to <4 x half>, !dbg !26
  %math.packed853 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %813) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %796, <4 x float> %774) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %815 = shufflevector <8 x half> %600, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %815, <4 x float> %776) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %817 = extractelement <4 x float> %lazy.score4991, i64 0, !dbg !24
  %818 = call float @llvm.amdgcn.exp2.f32(float %817), !dbg !24
  %819 = fadd float %k0.sum.anchor4750, %818, !dbg !25
  %k0.sum.anchor4758 = call float asm sideeffect "", "=v,0"(float %819) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %820 = insertelement <4 x float> poison, float %818, i64 0, !dbg !26
  %821 = extractelement <4 x float> %lazy.score4991, i64 1, !dbg !24
  %822 = call float @llvm.amdgcn.exp2.f32(float %821), !dbg !24
  %823 = fadd float %k0.sum.anchor4751, %822, !dbg !25
  %k0.sum.anchor4759 = call float asm sideeffect "", "=v,0"(float %823) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %824 = insertelement <4 x float> %820, float %822, i64 1, !dbg !26
  %825 = extractelement <4 x float> %lazy.score4991, i64 2, !dbg !24
  %826 = call float @llvm.amdgcn.exp2.f32(float %825), !dbg !24
  %827 = fadd float %k0.sum.anchor4752, %826, !dbg !25
  %k0.sum.anchor4760 = call float asm sideeffect "", "=v,0"(float %827) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %828 = insertelement <4 x float> %824, float %826, i64 2, !dbg !26
  %829 = extractelement <4 x float> %lazy.score4991, i64 3, !dbg !24
  %830 = call float @llvm.amdgcn.exp2.f32(float %829), !dbg !24
  %831 = fadd float %k0.sum.anchor4753, %830, !dbg !25
  %k0.sum.anchor4761 = call float asm sideeffect "", "=v,0"(float %831) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %832 = insertelement <4 x float> %828, float %830, i64 3, !dbg !26
  %math.result868 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %832) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f05079 = extractelement <4 x float> %math.result868, i32 0, !dbg !26
  %.cvtpack.f15080 = extractelement <4 x float> %math.result868, i32 1, !dbg !26
  %.cvtpack.f25081 = extractelement <4 x float> %math.result868, i32 2, !dbg !26
  %.cvtpack.f35082 = extractelement <4 x float> %math.result868, i32 3, !dbg !26
  %.cvtpack.p05083 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05079, float %.cvtpack.f15080), !dbg !26
  %.cvtpack.p15084 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25081, float %.cvtpack.f35082), !dbg !26
  %.cvtpack.v05085 = insertelement <2 x i32> poison, i32 %.cvtpack.p05083, i32 0, !dbg !26
  %.cvtpack.v15086 = insertelement <2 x i32> %.cvtpack.v05085, i32 %.cvtpack.p15084, i32 1, !dbg !26
  %833 = bitcast <2 x i32> %.cvtpack.v15086 to <4 x half>, !dbg !26
  %math.packed869 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %833) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %834 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %796, <4 x float> %794) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %815, <4 x float> %795) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %836 = shufflevector <8 x half> %602, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %837 = extractelement <4 x float> %lazy.score4984, i64 0, !dbg !24
  %838 = call float @llvm.amdgcn.exp2.f32(float %837), !dbg !24
  %839 = fadd float %k0.sum.anchor4754, %838, !dbg !25
  %k0.sum.anchor4762 = call float asm sideeffect "", "=v,0"(float %839) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %840 = insertelement <4 x float> poison, float %838, i64 0, !dbg !26
  %841 = extractelement <4 x float> %lazy.score4984, i64 1, !dbg !24
  %842 = call float @llvm.amdgcn.exp2.f32(float %841), !dbg !24
  %843 = fadd float %k0.sum.anchor4755, %842, !dbg !25
  %k0.sum.anchor4763 = call float asm sideeffect "", "=v,0"(float %843) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %844 = insertelement <4 x float> %840, float %842, i64 1, !dbg !26
  %845 = extractelement <4 x float> %lazy.score4984, i64 2, !dbg !24
  %846 = call float @llvm.amdgcn.exp2.f32(float %845), !dbg !24
  %847 = fadd float %k0.sum.anchor4756, %846, !dbg !25
  %k0.sum.anchor4764 = call float asm sideeffect "", "=v,0"(float %847) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %848 = insertelement <4 x float> %844, float %846, i64 2, !dbg !26
  %849 = extractelement <4 x float> %lazy.score4984, i64 3, !dbg !24
  %850 = call float @llvm.amdgcn.exp2.f32(float %849), !dbg !24
  %851 = fadd float %k0.sum.anchor4757, %850, !dbg !25
  %k0.sum.anchor4765 = call float asm sideeffect "", "=v,0"(float %851) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %852 = insertelement <4 x float> %848, float %850, i64 3, !dbg !26
  %math.result854 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %852) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f05087 = extractelement <4 x float> %math.result854, i32 0, !dbg !26
  %.cvtpack.f15088 = extractelement <4 x float> %math.result854, i32 1, !dbg !26
  %.cvtpack.f25089 = extractelement <4 x float> %math.result854, i32 2, !dbg !26
  %.cvtpack.f35090 = extractelement <4 x float> %math.result854, i32 3, !dbg !26
  %.cvtpack.p05091 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05087, float %.cvtpack.f15088), !dbg !26
  %.cvtpack.p15092 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25089, float %.cvtpack.f35090), !dbg !26
  %.cvtpack.v05093 = insertelement <2 x i32> poison, i32 %.cvtpack.p05091, i32 0, !dbg !26
  %.cvtpack.v15094 = insertelement <2 x i32> %.cvtpack.v05093, i32 %.cvtpack.p15092, i32 1, !dbg !26
  %853 = bitcast <2 x i32> %.cvtpack.v15094 to <4 x half>, !dbg !26
  %math.packed855 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %853) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %854 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %836, <4 x float> %814) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %855 = shufflevector <8 x half> %602, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %856 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %855, <4 x float> %816) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %857 = extractelement <4 x float> %lazy.score4992, i64 0, !dbg !24
  %858 = call float @llvm.amdgcn.exp2.f32(float %857), !dbg !24
  %859 = fadd float %k0.sum.anchor4758, %858, !dbg !25
  %k0.sum.anchor4766 = call float asm sideeffect "", "=v,0"(float %859) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %860 = insertelement <4 x float> poison, float %858, i64 0, !dbg !26
  %861 = extractelement <4 x float> %lazy.score4992, i64 1, !dbg !24
  %862 = call float @llvm.amdgcn.exp2.f32(float %861), !dbg !24
  %863 = fadd float %k0.sum.anchor4759, %862, !dbg !25
  %k0.sum.anchor4767 = call float asm sideeffect "", "=v,0"(float %863) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %864 = insertelement <4 x float> %860, float %862, i64 1, !dbg !26
  %865 = extractelement <4 x float> %lazy.score4992, i64 2, !dbg !24
  %866 = call float @llvm.amdgcn.exp2.f32(float %865), !dbg !24
  %867 = fadd float %k0.sum.anchor4760, %866, !dbg !25
  %k0.sum.anchor4768 = call float asm sideeffect "", "=v,0"(float %867) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %868 = insertelement <4 x float> %864, float %866, i64 2, !dbg !26
  %869 = extractelement <4 x float> %lazy.score4992, i64 3, !dbg !24
  %870 = call float @llvm.amdgcn.exp2.f32(float %869), !dbg !24
  %871 = fadd float %k0.sum.anchor4761, %870, !dbg !25
  %k0.sum.anchor4769 = call float asm sideeffect "", "=v,0"(float %871) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %872 = insertelement <4 x float> %868, float %870, i64 3, !dbg !26
  %math.result870 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %872) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f05095 = extractelement <4 x float> %math.result870, i32 0, !dbg !26
  %.cvtpack.f15096 = extractelement <4 x float> %math.result870, i32 1, !dbg !26
  %.cvtpack.f25097 = extractelement <4 x float> %math.result870, i32 2, !dbg !26
  %.cvtpack.f35098 = extractelement <4 x float> %math.result870, i32 3, !dbg !26
  %.cvtpack.p05099 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05095, float %.cvtpack.f15096), !dbg !26
  %.cvtpack.p15100 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25097, float %.cvtpack.f35098), !dbg !26
  %.cvtpack.v05101 = insertelement <2 x i32> poison, i32 %.cvtpack.p05099, i32 0, !dbg !26
  %.cvtpack.v15102 = insertelement <2 x i32> %.cvtpack.v05101, i32 %.cvtpack.p15100, i32 1, !dbg !26
  %873 = bitcast <2 x i32> %.cvtpack.v15102 to <4 x half>, !dbg !26
  %math.packed871 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %873) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %836, <4 x float> %834) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %875 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %855, <4 x float> %835) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %876 = bitcast <4 x i32> %603 to <8 x half>, !dbg !14
  %877 = shufflevector <8 x half> %876, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %878 = extractelement <4 x float> %lazy.score4985, i64 0, !dbg !24
  %879 = call float @llvm.amdgcn.exp2.f32(float %878), !dbg !24
  %880 = fadd float %k0.sum.anchor4762, %879, !dbg !25
  %k0.sum.anchor4770 = call float asm sideeffect "", "=v,0"(float %880) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %881 = insertelement <4 x float> poison, float %879, i64 0, !dbg !26
  %882 = extractelement <4 x float> %lazy.score4985, i64 1, !dbg !24
  %883 = call float @llvm.amdgcn.exp2.f32(float %882), !dbg !24
  %884 = fadd float %k0.sum.anchor4763, %883, !dbg !25
  %k0.sum.anchor4771 = call float asm sideeffect "", "=v,0"(float %884) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %885 = fadd float %k0.sum.anchor4770, %k0.sum.anchor4771, !dbg !25
  %k0.sum.anchor4772 = call float asm sideeffect "", "=v,0"(float %885) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %886 = insertelement <4 x float> %881, float %883, i64 1, !dbg !26
  %887 = extractelement <4 x float> %lazy.score4985, i64 2, !dbg !24
  %888 = call float @llvm.amdgcn.exp2.f32(float %887), !dbg !24
  %889 = fadd float %k0.sum.anchor4764, %888, !dbg !25
  %k0.sum.anchor4773 = call float asm sideeffect "", "=v,0"(float %889) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %890 = insertelement <4 x float> %886, float %888, i64 2, !dbg !26
  %891 = extractelement <4 x float> %lazy.score4985, i64 3, !dbg !24
  %892 = call float @llvm.amdgcn.exp2.f32(float %891), !dbg !24
  %893 = fadd float %k0.sum.anchor4765, %892, !dbg !25
  %k0.sum.anchor4774 = call float asm sideeffect "", "=v,0"(float %893) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %894 = fadd float %k0.sum.anchor4773, %k0.sum.anchor4774, !dbg !25
  %k0.sum.anchor4775 = call float asm sideeffect "", "=v,0"(float %894) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %895 = fadd float %k0.sum.anchor4772, %k0.sum.anchor4775, !dbg !25
  %k0.sum.anchor4776 = call float asm sideeffect "", "=v,0"(float %895) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %896 = fadd float %k0.sum.anchor4776, %573, !dbg !25
  %k0.sum.anchor4777 = call float asm sideeffect "", "=v,0"(float %896) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %897 = insertelement <1 x float> poison, float %k0.sum.anchor4777, i64 0, !dbg !6
  %898 = insertvalue [2 x <1 x float>] poison, <1 x float> %897, 0, !dbg !6
  %899 = insertelement <4 x float> %890, float %892, i64 3, !dbg !26
  %math.result856 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %899) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f05103 = extractelement <4 x float> %math.result856, i32 0, !dbg !26
  %.cvtpack.f15104 = extractelement <4 x float> %math.result856, i32 1, !dbg !26
  %.cvtpack.f25105 = extractelement <4 x float> %math.result856, i32 2, !dbg !26
  %.cvtpack.f35106 = extractelement <4 x float> %math.result856, i32 3, !dbg !26
  %.cvtpack.p05107 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05103, float %.cvtpack.f15104), !dbg !26
  %.cvtpack.p15108 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25105, float %.cvtpack.f35106), !dbg !26
  %.cvtpack.v05109 = insertelement <2 x i32> poison, i32 %.cvtpack.p05107, i32 0, !dbg !26
  %.cvtpack.v15110 = insertelement <2 x i32> %.cvtpack.v05109, i32 %.cvtpack.p15108, i32 1, !dbg !26
  %900 = bitcast <2 x i32> %.cvtpack.v15110 to <4 x half>, !dbg !26
  %math.packed857 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %900) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %877, <4 x float> %854) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %902 = shufflevector <8 x half> %876, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %902, <4 x float> %856) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %904 = extractelement <4 x float> %lazy.score4993, i64 0, !dbg !24
  %905 = call float @llvm.amdgcn.exp2.f32(float %904), !dbg !24
  %906 = fadd float %k0.sum.anchor4766, %905, !dbg !25
  %k0.sum.anchor4778 = call float asm sideeffect "", "=v,0"(float %906) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %907 = insertelement <4 x float> poison, float %905, i64 0, !dbg !26
  %908 = extractelement <4 x float> %lazy.score4993, i64 1, !dbg !24
  %909 = call float @llvm.amdgcn.exp2.f32(float %908), !dbg !24
  %910 = fadd float %k0.sum.anchor4767, %909, !dbg !25
  %k0.sum.anchor4779 = call float asm sideeffect "", "=v,0"(float %910) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %911 = fadd float %k0.sum.anchor4778, %k0.sum.anchor4779, !dbg !25
  %k0.sum.anchor4780 = call float asm sideeffect "", "=v,0"(float %911) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %912 = insertelement <4 x float> %907, float %909, i64 1, !dbg !26
  %913 = extractelement <4 x float> %lazy.score4993, i64 2, !dbg !24
  %914 = call float @llvm.amdgcn.exp2.f32(float %913), !dbg !24
  %915 = fadd float %k0.sum.anchor4768, %914, !dbg !25
  %k0.sum.anchor4781 = call float asm sideeffect "", "=v,0"(float %915) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %916 = insertelement <4 x float> %912, float %914, i64 2, !dbg !26
  %917 = extractelement <4 x float> %lazy.score4993, i64 3, !dbg !24
  %918 = call float @llvm.amdgcn.exp2.f32(float %917), !dbg !24
  %919 = fadd float %k0.sum.anchor4769, %918, !dbg !25
  %k0.sum.anchor4782 = call float asm sideeffect "", "=v,0"(float %919) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %920 = fadd float %k0.sum.anchor4781, %k0.sum.anchor4782, !dbg !25
  %k0.sum.anchor4783 = call float asm sideeffect "", "=v,0"(float %920) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %921 = fadd float %k0.sum.anchor4780, %k0.sum.anchor4783, !dbg !25
  %k0.sum.anchor4784 = call float asm sideeffect "", "=v,0"(float %921) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %922 = fadd float %k0.sum.anchor4784, %575, !dbg !25
  %k0.sum.anchor4785 = call float asm sideeffect "", "=v,0"(float %922) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %923 = insertelement <1 x float> poison, float %k0.sum.anchor4785, i64 0, !dbg !6
  %924 = insertvalue [2 x <1 x float>] %898, <1 x float> %923, 1, !dbg !6
  %925 = insertelement <4 x float> %916, float %918, i64 3, !dbg !26
  %math.result872 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %925) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f05111 = extractelement <4 x float> %math.result872, i32 0, !dbg !26
  %.cvtpack.f15112 = extractelement <4 x float> %math.result872, i32 1, !dbg !26
  %.cvtpack.f25113 = extractelement <4 x float> %math.result872, i32 2, !dbg !26
  %.cvtpack.f35114 = extractelement <4 x float> %math.result872, i32 3, !dbg !26
  %.cvtpack.p05115 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05111, float %.cvtpack.f15112), !dbg !26
  %.cvtpack.p15116 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25113, float %.cvtpack.f35114), !dbg !26
  %.cvtpack.v05117 = insertelement <2 x i32> poison, i32 %.cvtpack.p05115, i32 0, !dbg !26
  %.cvtpack.v15118 = insertelement <2 x i32> %.cvtpack.v05117, i32 %.cvtpack.p15116, i32 1, !dbg !26
  %926 = bitcast <2 x i32> %.cvtpack.v15118 to <4 x half>, !dbg !26
  %math.packed873 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %926) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %877, <4 x float> %874) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %902, <4 x float> %875) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %929 = extractvalue [2 x <8 x float>] %lazy.state4997, 0, !dbg !27
  %930 = shufflevector <8 x float> %929, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %931 = shufflevector <8 x float> %929, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %932 = extractvalue [2 x <8 x float>] %lazy.state4997, 1, !dbg !27
  %933 = shufflevector <8 x float> %932, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %934 = shufflevector <8 x float> %932, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %.idx47114713 = or disjoint i64 %257, %address.local3194.idx4700, !dbg !19
  %.idx47114713.tr = trunc nuw i64 %.idx47114713 to i32, !dbg !19
  %dma.buffer.offset4565 = shl nuw nsw i32 %.idx47114713.tr, 1, !dbg !19
  %935 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %371, i32 %dma.buffer.offset4565, <4 x i32> %dma.buffer.rsrc4574) #10, !dbg !19
  %.idx47144715 = or disjoint i64 %257, %address.local3193, !dbg !19
  %936 = or disjoint i64 %.idx47144715, %address.local3196, !dbg !19
  %.tr4716 = trunc nuw i64 %936 to i32, !dbg !19
  %937 = shl i32 %.tr4716, 1, !dbg !19
  %dma.buffer.offset4563 = or i32 %937, 32, !dbg !19
  %938 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %376, i32 %dma.buffer.offset4563, <4 x i32> %dma.buffer.rsrc4574) #10, !dbg !19
  %address.local4000 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local3889, !dbg !14
  %939 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4000, i32 0) #0, !dbg !14
  %940 = bitcast <4 x i32> %939 to <8 x half>, !dbg !14
  %address.local4011 = getelementptr i8, ptr addrspace(3) %address.local4000, i64 64, !dbg !14
  %941 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4011, i32 0) #0, !dbg !14
  %942 = bitcast <4 x i32> %941 to <8 x half>, !dbg !14
  %address.local4022 = getelementptr i8, ptr addrspace(3) %address.local4000, i64 128, !dbg !14
  %943 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4022, i32 0) #0, !dbg !14
  %944 = bitcast <4 x i32> %943 to <8 x half>, !dbg !14
  %address.local4033 = getelementptr i8, ptr addrspace(3) %address.local4000, i64 192, !dbg !14
  %945 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4033, i32 0) #0, !dbg !14
  %946 = bitcast <4 x i32> %945 to <8 x half>, !dbg !14
  %address.local4044 = getelementptr i8, ptr addrspace(3) %address.local4000, i64 1024, !dbg !14
  %947 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4044, i32 0) #0, !dbg !14
  %948 = bitcast <4 x i32> %947 to <8 x half>, !dbg !14
  %address.local4055 = getelementptr i8, ptr addrspace(3) %address.local4000, i64 1088, !dbg !14
  %949 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4055, i32 0) #0, !dbg !14
  %950 = bitcast <4 x i32> %949 to <8 x half>, !dbg !14
  %address.local4066 = getelementptr i8, ptr addrspace(3) %address.local4000, i64 1152, !dbg !14
  %951 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4066, i32 0) #0, !dbg !14
  %952 = bitcast <4 x i32> %951 to <8 x half>, !dbg !14
  %address.local4077 = getelementptr i8, ptr addrspace(3) %address.local4000, i64 1216, !dbg !14
  %953 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4077, i32 0) #0, !dbg !14
  %954 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %954, <4 x float> %930) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %956 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %957 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %956, <4 x float> %931) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %954, <4 x float> %933) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %956, <4 x float> %934) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %960 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %960, <4 x float> %955) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %962 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %962, <4 x float> %957) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %960, <4 x float> %958) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %962, <4 x float> %959) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %966 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %966, <4 x float> %961) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %968 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %968, <4 x float> %963) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %966, <4 x float> %964) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %968, <4 x float> %965) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %972 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %972, <4 x float> %967) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %974 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %974, <4 x float> %969) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %972, <4 x float> %970) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %974, <4 x float> %971) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %978 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %978, <4 x float> %973) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %980 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %980, <4 x float> %975) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %978, <4 x float> %976) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %980, <4 x float> %977) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %984 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %984, <4 x float> %979) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %986 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %986, <4 x float> %981) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %984, <4 x float> %982) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %986, <4 x float> %983) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %990 = shufflevector <8 x half> %952, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %990, <4 x float> %985) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %992 = shufflevector <8 x half> %952, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %992, <4 x float> %987) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %990, <4 x float> %988) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %992, <4 x float> %989) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %996 = bitcast <4 x i32> %953 to <8 x half>, !dbg !14
  %997 = shufflevector <8 x half> %996, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %997, <4 x float> %991) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %999 = shufflevector <8 x half> %996, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1000 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %999, <4 x float> %993) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %997, <4 x float> %994) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %999, <4 x float> %995) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %1003 = extractvalue [2 x <8 x float>] %lazy.state4996, 0, !dbg !27
  %1004 = shufflevector <8 x float> %1003, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1005 = shufflevector <8 x float> %1003, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %1006 = extractvalue [2 x <8 x float>] %lazy.state4996, 1, !dbg !27
  %1007 = shufflevector <8 x float> %1006, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1008 = shufflevector <8 x float> %1006, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %.idx47174719 = or disjoint i64 %314, %address.local3194.idx4700, !dbg !19
  %.idx47174719.tr = trunc nuw i64 %.idx47174719 to i32, !dbg !19
  %dma.buffer.offset4561 = shl nuw nsw i32 %.idx47174719.tr, 1, !dbg !19
  %1009 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %582, i32 %dma.buffer.offset4561, <4 x i32> %dma.buffer.rsrc4574) #10, !dbg !19
  %.idx47204721 = or disjoint i64 %314, %address.local3193, !dbg !19
  %1010 = or disjoint i64 %.idx47204721, %address.local3196, !dbg !19
  %.tr4722 = trunc nuw i64 %1010 to i32, !dbg !19
  %1011 = shl i32 %.tr4722, 1, !dbg !19
  %dma.buffer.offset = or i32 %1011, 32, !dbg !19
  %1012 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %584, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc4574) #10, !dbg !19
  %1013 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3890, i32 0) #0, !dbg !14
  %1014 = bitcast <4 x i32> %1013 to <8 x half>, !dbg !14
  %address.local4119 = or disjoint i32 %address.local3888, 32, !dbg !14
  %address.local4120 = zext nneg i32 %address.local4119 to i64, !dbg !14
  %address.local4121 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4120, !dbg !14
  %1015 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4121, i32 0) #0, !dbg !14
  %1016 = bitcast <4 x i32> %1015 to <8 x half>, !dbg !14
  %address.local4130 = or disjoint i32 %address.local3888, 64, !dbg !14
  %address.local4131 = zext nneg i32 %address.local4130 to i64, !dbg !14
  %address.local4132 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4131, !dbg !14
  %1017 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4132, i32 0) #0, !dbg !14
  %1018 = bitcast <4 x i32> %1017 to <8 x half>, !dbg !14
  %address.local4141 = or disjoint i32 %address.local3888, 96, !dbg !14
  %address.local4142 = zext nneg i32 %address.local4141 to i64, !dbg !14
  %address.local4143 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4142, !dbg !14
  %1019 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4143, i32 0) #0, !dbg !14
  %1020 = bitcast <4 x i32> %1019 to <8 x half>, !dbg !14
  %address.local4152 = or disjoint i32 %address.local3888, 512, !dbg !14
  %address.local4153 = zext nneg i32 %address.local4152 to i64, !dbg !14
  %address.local4154 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4153, !dbg !14
  %1021 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4154, i32 0) #0, !dbg !14
  %1022 = bitcast <4 x i32> %1021 to <8 x half>, !dbg !14
  %address.local4163 = or disjoint i32 %address.local3888, 544, !dbg !14
  %address.local4164 = zext nneg i32 %address.local4163 to i64, !dbg !14
  %address.local4165 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4164, !dbg !14
  %1023 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4165, i32 0) #0, !dbg !14
  %1024 = bitcast <4 x i32> %1023 to <8 x half>, !dbg !14
  %address.local4174 = or disjoint i32 %address.local3888, 576, !dbg !14
  %address.local4175 = zext nneg i32 %address.local4174 to i64, !dbg !14
  %address.local4176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4175, !dbg !14
  %1025 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4176, i32 0) #0, !dbg !14
  %1026 = bitcast <4 x i32> %1025 to <8 x half>, !dbg !14
  %address.local4185 = or disjoint i32 %address.local3888, 608, !dbg !14
  %address.local4186 = zext nneg i32 %address.local4185 to i64, !dbg !14
  %address.local4187 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4186, !dbg !14
  %1027 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4187, i32 0) #0, !dbg !14
  %1028 = shufflevector <8 x half> %1014, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1028, <4 x float> %1004) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1030 = shufflevector <8 x half> %1014, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1031 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1030, <4 x float> %1005) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1032 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %1028, <4 x float> %1007) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %1030, <4 x float> %1008) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1034 = shufflevector <8 x half> %1016, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %1034, <4 x float> %1029) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1036 = shufflevector <8 x half> %1016, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %1036, <4 x float> %1031) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %1034, <4 x float> %1032) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %1036, <4 x float> %1033) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1040 = shufflevector <8 x half> %1018, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1041 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %1040, <4 x float> %1035) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1042 = shufflevector <8 x half> %1018, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %1042, <4 x float> %1037) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %1040, <4 x float> %1038) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %1042, <4 x float> %1039) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1046 = shufflevector <8 x half> %1020, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %1046, <4 x float> %1041) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1048 = shufflevector <8 x half> %1020, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %1048, <4 x float> %1043) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %1046, <4 x float> %1044) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %1048, <4 x float> %1045) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1052 = shufflevector <8 x half> %1022, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %1052, <4 x float> %1047) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1054 = shufflevector <8 x half> %1022, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %1054, <4 x float> %1049) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %1052, <4 x float> %1050) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %1054, <4 x float> %1051) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1058 = shufflevector <8 x half> %1024, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %1058, <4 x float> %1053) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1060 = shufflevector <8 x half> %1024, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %1060, <4 x float> %1055) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %1058, <4 x float> %1056) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %1060, <4 x float> %1057) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1064 = shufflevector <8 x half> %1026, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %1064, <4 x float> %1059) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1066 = shufflevector <8 x half> %1026, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %1066, <4 x float> %1061) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %1064, <4 x float> %1062) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %1066, <4 x float> %1063) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1070 = bitcast <4 x i32> %1027 to <8 x half>, !dbg !14
  %1071 = shufflevector <8 x half> %1070, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %1071, <4 x float> %1065) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1073 = shufflevector <8 x half> %1070, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %1073, <4 x float> %1067) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %1071, <4 x float> %1068) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %1073, <4 x float> %1069) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %1077 = extractvalue [2 x <8 x float>] %lazy.state4995, 0, !dbg !27
  %1078 = shufflevector <8 x float> %1077, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1079 = shufflevector <8 x float> %1077, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %1080 = extractvalue [2 x <8 x float>] %lazy.state4995, 1, !dbg !27
  %1081 = shufflevector <8 x float> %1080, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1082 = shufflevector <8 x float> %1080, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1083 = add nuw nsw i64 %102, 1, !dbg !6
  %k0.has.next = icmp ult i64 %1083, 32, !dbg !14
  br i1 %k0.has.next, label %k0.issue, label %k0.join, !dbg !14

k0.issue:                                         ; preds = %lazy.join
  %k0.next. = shl nuw nsw i64 %1083, 14, !dbg !13
  %k0.next.4871 = or disjoint i64 %112, %k0.next., !dbg !13
  %k0.next..idx46434646 = add nsw i64 %k0.next.4871, %137, !dbg !13
  %k0.next..idx46434646.tr = trunc i64 %k0.next..idx46434646 to i32, !dbg !13
  %k0.next.dma.buffer.offset4621 = shl i32 %k0.next..idx46434646.tr, 1, !dbg !13
  %k0.next.4872 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %134, i32 %k0.next.dma.buffer.offset4621, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next..idx4647 = shl nuw nsw i64 %k0.next.4871, 1, !dbg !13
  %k0.next.4873 = add i64 %k0.next..idx4647, %158, !dbg !13
  %k0.next.dma.buffer.offset4619 = trunc i64 %k0.next.4873 to i32, !dbg !13
  %k0.next.4874 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %153, i32 %k0.next.dma.buffer.offset4619, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.4875 = add i64 %k0.next..idx4647, %176, !dbg !13
  %k0.next.dma.buffer.offset4617 = trunc i64 %k0.next.4875 to i32, !dbg !13
  %k0.next.4876 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %173, i32 %k0.next.dma.buffer.offset4617, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next..idx46534656 = add nsw i64 %k0.next.4871, %184, !dbg !13
  %k0.next..idx46534656.tr = trunc i64 %k0.next..idx46534656 to i32, !dbg !13
  %k0.next.4877 = shl i32 %k0.next..idx46534656.tr, 1, !dbg !13
  %k0.next.4878 = add i32 %k0.next.4877, 48, !dbg !13
  %k0.next.dma.buffer.offset4615 = add i32 %k0.next.4878, %190, !dbg !13
  %k0.next.4879 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %187, i32 %k0.next.dma.buffer.offset4615, <4 x i32> %dma.buffer.rsrc4622) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %k0.join

k0.join:                                          ; preds = %lazy.join, %k0.issue
  %1084 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4000, i32 0) #0, !dbg !14
  %1085 = bitcast <4 x i32> %1084 to <8 x half>, !dbg !14
  %address.local4189 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4120, !dbg !14
  %1086 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4189, i32 0) #0, !dbg !14
  %1087 = bitcast <4 x i32> %1086 to <8 x half>, !dbg !14
  %address.local4190 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4131, !dbg !14
  %1088 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4190, i32 0) #0, !dbg !14
  %1089 = bitcast <4 x i32> %1088 to <8 x half>, !dbg !14
  %address.local4191 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4142, !dbg !14
  %1090 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4191, i32 0) #0, !dbg !14
  %1091 = bitcast <4 x i32> %1090 to <8 x half>, !dbg !14
  %address.local4192 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4153, !dbg !14
  %1092 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4192, i32 0) #0, !dbg !14
  %1093 = bitcast <4 x i32> %1092 to <8 x half>, !dbg !14
  %address.local4193 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4164, !dbg !14
  %1094 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4193, i32 0) #0, !dbg !14
  %1095 = bitcast <4 x i32> %1094 to <8 x half>, !dbg !14
  %address.local4194 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4175, !dbg !14
  %1096 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4194, i32 0) #0, !dbg !14
  %1097 = bitcast <4 x i32> %1096 to <8 x half>, !dbg !14
  %address.local4195 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4186, !dbg !14
  %1098 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4195, i32 0) #0, !dbg !14
  %1099 = shufflevector <8 x half> %1085, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1099, <4 x float> %1078) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1101 = shufflevector <8 x half> %1085, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1101, <4 x float> %1079) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %1099, <4 x float> %1081) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %1101, <4 x float> %1082) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1105 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %1105, <4 x float> %1100) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1107 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %1107, <4 x float> %1102) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1109 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %1105, <4 x float> %1103) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %1107, <4 x float> %1104) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1111 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %1111, <4 x float> %1106) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1113 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %1113, <4 x float> %1108) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1115 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %1111, <4 x float> %1109) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %1113, <4 x float> %1110) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1117 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %1117, <4 x float> %1112) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1119 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %1119, <4 x float> %1114) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %1117, <4 x float> %1115) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %1119, <4 x float> %1116) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1123 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %1123, <4 x float> %1118) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1125 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %1125, <4 x float> %1120) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %1123, <4 x float> %1121) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %1125, <4 x float> %1122) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1129 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %1129, <4 x float> %1124) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1131 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %1131, <4 x float> %1126) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %1129, <4 x float> %1127) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %1131, <4 x float> %1128) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1135 = shufflevector <8 x half> %1097, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %1135, <4 x float> %1130) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1137 = shufflevector <8 x half> %1097, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %1137, <4 x float> %1132) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %1135, <4 x float> %1133) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %1137, <4 x float> %1134) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1141 = bitcast <4 x i32> %1098 to <8 x half>, !dbg !14
  %1142 = shufflevector <8 x half> %1141, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %1142, <4 x float> %1136) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1144 = shufflevector <8 x half> %1141, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %1144, <4 x float> %1138) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %1142, <4 x float> %1139) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %1144, <4 x float> %1140) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !6
  %1148 = shufflevector <4 x float> %901, <4 x float> %903, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1149 = insertvalue [2 x <8 x float>] poison, <8 x float> %1148, 0, !dbg !6
  %1150 = shufflevector <4 x float> %927, <4 x float> %928, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1151 = insertvalue [2 x <8 x float>] %1149, <8 x float> %1150, 1, !dbg !6
  %1152 = shufflevector <4 x float> %998, <4 x float> %1000, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1153 = insertvalue [2 x <8 x float>] poison, <8 x float> %1152, 0, !dbg !6
  %1154 = shufflevector <4 x float> %1001, <4 x float> %1002, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1155 = insertvalue [2 x <8 x float>] %1153, <8 x float> %1154, 1, !dbg !6
  %1156 = shufflevector <4 x float> %1072, <4 x float> %1074, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1157 = insertvalue [2 x <8 x float>] poison, <8 x float> %1156, 0, !dbg !6
  %1158 = shufflevector <4 x float> %1075, <4 x float> %1076, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1159 = insertvalue [2 x <8 x float>] %1157, <8 x float> %1158, 1, !dbg !6
  %1160 = shufflevector <4 x float> %1143, <4 x float> %1145, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1161 = insertvalue [2 x <8 x float>] poison, <8 x float> %1160, 0, !dbg !6
  %1162 = shufflevector <4 x float> %1146, <4 x float> %1147, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1163 = insertvalue [2 x <8 x float>] %1161, <8 x float> %1162, 1, !dbg !6
  br label %101, !dbg !6

1164:                                             ; preds = %101
  %1165 = extractvalue [2 x <1 x float>] %104, 1, !dbg !6
  %1166 = extractelement <1 x float> %1165, i64 0, !dbg !6
  %1167 = bitcast float %1166 to i32, !dbg !6
  %1168 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1167), !dbg !6
  %1169 = bitcast i32 %1168 to float, !dbg !6
  %1170 = fadd float %1166, %1169, !dbg !6
  %1171 = bitcast float %1170 to i32, !dbg !6
  %1172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1171), !dbg !6
  %1173 = bitcast i32 %1172 to float, !dbg !6
  %1174 = fadd float %1170, %1173, !dbg !6
  %1175 = extractvalue [2 x <1 x float>] %104, 0, !dbg !6
  %1176 = extractelement <1 x float> %1175, i64 0, !dbg !6
  %1177 = bitcast float %1176 to i32, !dbg !6
  %1178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1177), !dbg !6
  %1179 = bitcast i32 %1178 to float, !dbg !6
  %1180 = fadd float %1176, %1179, !dbg !6
  %1181 = bitcast float %1180 to i32, !dbg !6
  %1182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1181), !dbg !6
  %1183 = bitcast i32 %1182 to float, !dbg !6
  %1184 = fadd float %1180, %1183, !dbg !6
  %1185 = and i32 %34, 16, !dbg !6
  %.not = icmp eq i32 %1185, 0, !dbg !6
  %1186 = and i32 %34, 32, !dbg !6
  %.not84 = icmp eq i32 %1186, 0, !dbg !6
  %1187 = extractvalue [2 x <8 x float>] %108, 0, !dbg !6
  %1188 = extractelement <8 x float> %1187, i64 0, !dbg !6
  %1189 = extractelement <8 x float> %1187, i64 1, !dbg !6
  %1190 = extractelement <8 x float> %1187, i64 2, !dbg !6
  %1191 = extractelement <8 x float> %1187, i64 3, !dbg !6
  %1192 = select i1 %.not, float %1189, float %1188, !dbg !6
  %1193 = bitcast float %1192 to i32, !dbg !6
  %1194 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1193), !dbg !6
  %1195 = bitcast i32 %1194 to float, !dbg !6
  %1196 = select i1 %.not, float %1188, float %1195, !dbg !6
  %1197 = select i1 %.not, float %1195, float %1189, !dbg !6
  %1198 = select i1 %.not, float %1191, float %1190, !dbg !6
  %1199 = bitcast float %1198 to i32, !dbg !6
  %1200 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1199), !dbg !6
  %1201 = bitcast i32 %1200 to float, !dbg !6
  %1202 = select i1 %.not, float %1190, float %1201, !dbg !6
  %1203 = select i1 %.not, float %1201, float %1191, !dbg !6
  %1204 = select i1 %.not84, float %1202, float %1196, !dbg !6
  %1205 = bitcast float %1204 to i32, !dbg !6
  %1206 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1205), !dbg !6
  %1207 = bitcast i32 %1206 to float, !dbg !6
  %1208 = select i1 %.not84, float %1196, float %1207, !dbg !6
  %1209 = select i1 %.not84, float %1207, float %1202, !dbg !6
  %1210 = select i1 %.not84, float %1203, float %1197, !dbg !6
  %1211 = bitcast float %1210 to i32, !dbg !6
  %1212 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1211), !dbg !6
  %1213 = bitcast i32 %1212 to float, !dbg !6
  %1214 = select i1 %.not84, float %1197, float %1213, !dbg !6
  %1215 = select i1 %.not84, float %1213, float %1203, !dbg !6
  %1216 = extractelement <8 x float> %1187, i64 4, !dbg !6
  %1217 = extractelement <8 x float> %1187, i64 5, !dbg !6
  %1218 = extractelement <8 x float> %1187, i64 6, !dbg !6
  %1219 = extractelement <8 x float> %1187, i64 7, !dbg !6
  %1220 = select i1 %.not, float %1217, float %1216, !dbg !6
  %1221 = bitcast float %1220 to i32, !dbg !6
  %1222 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1221), !dbg !6
  %1223 = bitcast i32 %1222 to float, !dbg !6
  %1224 = select i1 %.not, float %1216, float %1223, !dbg !6
  %1225 = select i1 %.not, float %1223, float %1217, !dbg !6
  %1226 = select i1 %.not, float %1219, float %1218, !dbg !6
  %1227 = bitcast float %1226 to i32, !dbg !6
  %1228 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1227), !dbg !6
  %1229 = bitcast i32 %1228 to float, !dbg !6
  %1230 = select i1 %.not, float %1218, float %1229, !dbg !6
  %1231 = select i1 %.not, float %1229, float %1219, !dbg !6
  %1232 = select i1 %.not84, float %1230, float %1224, !dbg !6
  %1233 = bitcast float %1232 to i32, !dbg !6
  %1234 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1233), !dbg !6
  %1235 = bitcast i32 %1234 to float, !dbg !6
  %1236 = select i1 %.not84, float %1224, float %1235, !dbg !6
  %1237 = select i1 %.not84, float %1235, float %1230, !dbg !6
  %1238 = select i1 %.not84, float %1231, float %1225, !dbg !6
  %1239 = bitcast float %1238 to i32, !dbg !6
  %1240 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1239), !dbg !6
  %1241 = bitcast i32 %1240 to float, !dbg !6
  %1242 = select i1 %.not84, float %1225, float %1241, !dbg !6
  %1243 = select i1 %.not84, float %1241, float %1231, !dbg !6
  %1244 = extractvalue [2 x <8 x float>] %108, 1, !dbg !6
  %1245 = extractelement <8 x float> %1244, i64 0, !dbg !6
  %1246 = extractelement <8 x float> %1244, i64 1, !dbg !6
  %1247 = extractelement <8 x float> %1244, i64 2, !dbg !6
  %1248 = extractelement <8 x float> %1244, i64 3, !dbg !6
  %1249 = select i1 %.not, float %1246, float %1245, !dbg !6
  %1250 = bitcast float %1249 to i32, !dbg !6
  %1251 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1250), !dbg !6
  %1252 = bitcast i32 %1251 to float, !dbg !6
  %1253 = select i1 %.not, float %1245, float %1252, !dbg !6
  %1254 = select i1 %.not, float %1252, float %1246, !dbg !6
  %1255 = select i1 %.not, float %1248, float %1247, !dbg !6
  %1256 = bitcast float %1255 to i32, !dbg !6
  %1257 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1256), !dbg !6
  %1258 = bitcast i32 %1257 to float, !dbg !6
  %1259 = select i1 %.not, float %1247, float %1258, !dbg !6
  %1260 = select i1 %.not, float %1258, float %1248, !dbg !6
  %1261 = select i1 %.not84, float %1259, float %1253, !dbg !6
  %1262 = bitcast float %1261 to i32, !dbg !6
  %1263 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1262), !dbg !6
  %1264 = bitcast i32 %1263 to float, !dbg !6
  %1265 = select i1 %.not84, float %1253, float %1264, !dbg !6
  %1266 = select i1 %.not84, float %1264, float %1259, !dbg !6
  %1267 = select i1 %.not84, float %1260, float %1254, !dbg !6
  %1268 = bitcast float %1267 to i32, !dbg !6
  %1269 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1268), !dbg !6
  %1270 = bitcast i32 %1269 to float, !dbg !6
  %1271 = select i1 %.not84, float %1254, float %1270, !dbg !6
  %1272 = select i1 %.not84, float %1270, float %1260, !dbg !6
  %1273 = extractelement <8 x float> %1244, i64 4, !dbg !6
  %1274 = extractelement <8 x float> %1244, i64 5, !dbg !6
  %1275 = extractelement <8 x float> %1244, i64 6, !dbg !6
  %1276 = extractelement <8 x float> %1244, i64 7, !dbg !6
  %1277 = select i1 %.not, float %1274, float %1273, !dbg !6
  %1278 = bitcast float %1277 to i32, !dbg !6
  %1279 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1278), !dbg !6
  %1280 = bitcast i32 %1279 to float, !dbg !6
  %1281 = select i1 %.not, float %1273, float %1280, !dbg !6
  %1282 = select i1 %.not, float %1280, float %1274, !dbg !6
  %1283 = select i1 %.not, float %1276, float %1275, !dbg !6
  %1284 = bitcast float %1283 to i32, !dbg !6
  %1285 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1284), !dbg !6
  %1286 = bitcast i32 %1285 to float, !dbg !6
  %1287 = select i1 %.not, float %1275, float %1286, !dbg !6
  %1288 = select i1 %.not, float %1286, float %1276, !dbg !6
  %1289 = select i1 %.not84, float %1287, float %1281, !dbg !6
  %1290 = bitcast float %1289 to i32, !dbg !6
  %1291 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1290), !dbg !6
  %1292 = bitcast i32 %1291 to float, !dbg !6
  %1293 = select i1 %.not84, float %1281, float %1292, !dbg !6
  %1294 = select i1 %.not84, float %1292, float %1287, !dbg !6
  %1295 = select i1 %.not84, float %1288, float %1282, !dbg !6
  %1296 = bitcast float %1295 to i32, !dbg !6
  %1297 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1296), !dbg !6
  %1298 = bitcast i32 %1297 to float, !dbg !6
  %1299 = select i1 %.not84, float %1282, float %1298, !dbg !6
  %1300 = select i1 %.not84, float %1298, float %1288, !dbg !6
  %1301 = extractvalue [2 x <8 x float>] %107, 0, !dbg !6
  %1302 = extractelement <8 x float> %1301, i64 0, !dbg !6
  %1303 = extractelement <8 x float> %1301, i64 1, !dbg !6
  %1304 = extractelement <8 x float> %1301, i64 2, !dbg !6
  %1305 = extractelement <8 x float> %1301, i64 3, !dbg !6
  %1306 = select i1 %.not, float %1303, float %1302, !dbg !6
  %1307 = bitcast float %1306 to i32, !dbg !6
  %1308 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1307), !dbg !6
  %1309 = bitcast i32 %1308 to float, !dbg !6
  %1310 = select i1 %.not, float %1302, float %1309, !dbg !6
  %1311 = select i1 %.not, float %1309, float %1303, !dbg !6
  %1312 = select i1 %.not, float %1305, float %1304, !dbg !6
  %1313 = bitcast float %1312 to i32, !dbg !6
  %1314 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1313), !dbg !6
  %1315 = bitcast i32 %1314 to float, !dbg !6
  %1316 = select i1 %.not, float %1304, float %1315, !dbg !6
  %1317 = select i1 %.not, float %1315, float %1305, !dbg !6
  %1318 = select i1 %.not84, float %1316, float %1310, !dbg !6
  %1319 = bitcast float %1318 to i32, !dbg !6
  %1320 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1319), !dbg !6
  %1321 = bitcast i32 %1320 to float, !dbg !6
  %1322 = select i1 %.not84, float %1310, float %1321, !dbg !6
  %1323 = select i1 %.not84, float %1321, float %1316, !dbg !6
  %1324 = select i1 %.not84, float %1317, float %1311, !dbg !6
  %1325 = bitcast float %1324 to i32, !dbg !6
  %1326 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1325), !dbg !6
  %1327 = bitcast i32 %1326 to float, !dbg !6
  %1328 = select i1 %.not84, float %1311, float %1327, !dbg !6
  %1329 = select i1 %.not84, float %1327, float %1317, !dbg !6
  %1330 = extractelement <8 x float> %1301, i64 4, !dbg !6
  %1331 = extractelement <8 x float> %1301, i64 5, !dbg !6
  %1332 = extractelement <8 x float> %1301, i64 6, !dbg !6
  %1333 = extractelement <8 x float> %1301, i64 7, !dbg !6
  %1334 = select i1 %.not, float %1331, float %1330, !dbg !6
  %1335 = bitcast float %1334 to i32, !dbg !6
  %1336 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1335), !dbg !6
  %1337 = bitcast i32 %1336 to float, !dbg !6
  %1338 = select i1 %.not, float %1330, float %1337, !dbg !6
  %1339 = select i1 %.not, float %1337, float %1331, !dbg !6
  %1340 = select i1 %.not, float %1333, float %1332, !dbg !6
  %1341 = bitcast float %1340 to i32, !dbg !6
  %1342 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1341), !dbg !6
  %1343 = bitcast i32 %1342 to float, !dbg !6
  %1344 = select i1 %.not, float %1332, float %1343, !dbg !6
  %1345 = select i1 %.not, float %1343, float %1333, !dbg !6
  %1346 = select i1 %.not84, float %1344, float %1338, !dbg !6
  %1347 = bitcast float %1346 to i32, !dbg !6
  %1348 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1347), !dbg !6
  %1349 = bitcast i32 %1348 to float, !dbg !6
  %1350 = select i1 %.not84, float %1338, float %1349, !dbg !6
  %1351 = select i1 %.not84, float %1349, float %1344, !dbg !6
  %1352 = select i1 %.not84, float %1345, float %1339, !dbg !6
  %1353 = bitcast float %1352 to i32, !dbg !6
  %1354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1353), !dbg !6
  %1355 = bitcast i32 %1354 to float, !dbg !6
  %1356 = select i1 %.not84, float %1339, float %1355, !dbg !6
  %1357 = select i1 %.not84, float %1355, float %1345, !dbg !6
  %1358 = extractvalue [2 x <8 x float>] %107, 1, !dbg !6
  %1359 = extractelement <8 x float> %1358, i64 0, !dbg !6
  %1360 = extractelement <8 x float> %1358, i64 1, !dbg !6
  %1361 = extractelement <8 x float> %1358, i64 2, !dbg !6
  %1362 = extractelement <8 x float> %1358, i64 3, !dbg !6
  %1363 = select i1 %.not, float %1360, float %1359, !dbg !6
  %1364 = bitcast float %1363 to i32, !dbg !6
  %1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1364), !dbg !6
  %1366 = bitcast i32 %1365 to float, !dbg !6
  %1367 = select i1 %.not, float %1359, float %1366, !dbg !6
  %1368 = select i1 %.not, float %1366, float %1360, !dbg !6
  %1369 = select i1 %.not, float %1362, float %1361, !dbg !6
  %1370 = bitcast float %1369 to i32, !dbg !6
  %1371 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1370), !dbg !6
  %1372 = bitcast i32 %1371 to float, !dbg !6
  %1373 = select i1 %.not, float %1361, float %1372, !dbg !6
  %1374 = select i1 %.not, float %1372, float %1362, !dbg !6
  %1375 = select i1 %.not84, float %1373, float %1367, !dbg !6
  %1376 = bitcast float %1375 to i32, !dbg !6
  %1377 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1376), !dbg !6
  %1378 = bitcast i32 %1377 to float, !dbg !6
  %1379 = select i1 %.not84, float %1367, float %1378, !dbg !6
  %1380 = select i1 %.not84, float %1378, float %1373, !dbg !6
  %1381 = select i1 %.not84, float %1374, float %1368, !dbg !6
  %1382 = bitcast float %1381 to i32, !dbg !6
  %1383 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1382), !dbg !6
  %1384 = bitcast i32 %1383 to float, !dbg !6
  %1385 = select i1 %.not84, float %1368, float %1384, !dbg !6
  %1386 = select i1 %.not84, float %1384, float %1374, !dbg !6
  %1387 = extractelement <8 x float> %1358, i64 4, !dbg !6
  %1388 = extractelement <8 x float> %1358, i64 5, !dbg !6
  %1389 = extractelement <8 x float> %1358, i64 6, !dbg !6
  %1390 = extractelement <8 x float> %1358, i64 7, !dbg !6
  %1391 = select i1 %.not, float %1388, float %1387, !dbg !6
  %1392 = bitcast float %1391 to i32, !dbg !6
  %1393 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1392), !dbg !6
  %1394 = bitcast i32 %1393 to float, !dbg !6
  %1395 = select i1 %.not, float %1387, float %1394, !dbg !6
  %1396 = select i1 %.not, float %1394, float %1388, !dbg !6
  %1397 = select i1 %.not, float %1390, float %1389, !dbg !6
  %1398 = bitcast float %1397 to i32, !dbg !6
  %1399 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1398), !dbg !6
  %1400 = bitcast i32 %1399 to float, !dbg !6
  %1401 = select i1 %.not, float %1389, float %1400, !dbg !6
  %1402 = select i1 %.not, float %1400, float %1390, !dbg !6
  %1403 = select i1 %.not84, float %1401, float %1395, !dbg !6
  %1404 = bitcast float %1403 to i32, !dbg !6
  %1405 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1404), !dbg !6
  %1406 = bitcast i32 %1405 to float, !dbg !6
  %1407 = select i1 %.not84, float %1395, float %1406, !dbg !6
  %1408 = select i1 %.not84, float %1406, float %1401, !dbg !6
  %1409 = select i1 %.not84, float %1402, float %1396, !dbg !6
  %1410 = bitcast float %1409 to i32, !dbg !6
  %1411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1410), !dbg !6
  %1412 = bitcast i32 %1411 to float, !dbg !6
  %1413 = select i1 %.not84, float %1396, float %1412, !dbg !6
  %1414 = select i1 %.not84, float %1412, float %1402, !dbg !6
  %1415 = extractvalue [2 x <8 x float>] %106, 0, !dbg !6
  %1416 = extractelement <8 x float> %1415, i64 0, !dbg !6
  %1417 = extractelement <8 x float> %1415, i64 1, !dbg !6
  %1418 = extractelement <8 x float> %1415, i64 2, !dbg !6
  %1419 = extractelement <8 x float> %1415, i64 3, !dbg !6
  %1420 = select i1 %.not, float %1417, float %1416, !dbg !6
  %1421 = bitcast float %1420 to i32, !dbg !6
  %1422 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1421), !dbg !6
  %1423 = bitcast i32 %1422 to float, !dbg !6
  %1424 = select i1 %.not, float %1416, float %1423, !dbg !6
  %1425 = select i1 %.not, float %1423, float %1417, !dbg !6
  %1426 = select i1 %.not, float %1419, float %1418, !dbg !6
  %1427 = bitcast float %1426 to i32, !dbg !6
  %1428 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1427), !dbg !6
  %1429 = bitcast i32 %1428 to float, !dbg !6
  %1430 = select i1 %.not, float %1418, float %1429, !dbg !6
  %1431 = select i1 %.not, float %1429, float %1419, !dbg !6
  %1432 = select i1 %.not84, float %1430, float %1424, !dbg !6
  %1433 = bitcast float %1432 to i32, !dbg !6
  %1434 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1433), !dbg !6
  %1435 = bitcast i32 %1434 to float, !dbg !6
  %1436 = select i1 %.not84, float %1424, float %1435, !dbg !6
  %1437 = select i1 %.not84, float %1435, float %1430, !dbg !6
  %1438 = select i1 %.not84, float %1431, float %1425, !dbg !6
  %1439 = bitcast float %1438 to i32, !dbg !6
  %1440 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1439), !dbg !6
  %1441 = bitcast i32 %1440 to float, !dbg !6
  %1442 = select i1 %.not84, float %1425, float %1441, !dbg !6
  %1443 = select i1 %.not84, float %1441, float %1431, !dbg !6
  %1444 = extractelement <8 x float> %1415, i64 4, !dbg !6
  %1445 = extractelement <8 x float> %1415, i64 5, !dbg !6
  %1446 = extractelement <8 x float> %1415, i64 6, !dbg !6
  %1447 = extractelement <8 x float> %1415, i64 7, !dbg !6
  %1448 = select i1 %.not, float %1445, float %1444, !dbg !6
  %1449 = bitcast float %1448 to i32, !dbg !6
  %1450 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1449), !dbg !6
  %1451 = bitcast i32 %1450 to float, !dbg !6
  %1452 = select i1 %.not, float %1444, float %1451, !dbg !6
  %1453 = select i1 %.not, float %1451, float %1445, !dbg !6
  %1454 = select i1 %.not, float %1447, float %1446, !dbg !6
  %1455 = bitcast float %1454 to i32, !dbg !6
  %1456 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1455), !dbg !6
  %1457 = bitcast i32 %1456 to float, !dbg !6
  %1458 = select i1 %.not, float %1446, float %1457, !dbg !6
  %1459 = select i1 %.not, float %1457, float %1447, !dbg !6
  %1460 = select i1 %.not84, float %1458, float %1452, !dbg !6
  %1461 = bitcast float %1460 to i32, !dbg !6
  %1462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1461), !dbg !6
  %1463 = bitcast i32 %1462 to float, !dbg !6
  %1464 = select i1 %.not84, float %1452, float %1463, !dbg !6
  %1465 = select i1 %.not84, float %1463, float %1458, !dbg !6
  %1466 = select i1 %.not84, float %1459, float %1453, !dbg !6
  %1467 = bitcast float %1466 to i32, !dbg !6
  %1468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1467), !dbg !6
  %1469 = bitcast i32 %1468 to float, !dbg !6
  %1470 = select i1 %.not84, float %1453, float %1469, !dbg !6
  %1471 = select i1 %.not84, float %1469, float %1459, !dbg !6
  %1472 = extractvalue [2 x <8 x float>] %106, 1, !dbg !6
  %1473 = extractelement <8 x float> %1472, i64 0, !dbg !6
  %1474 = extractelement <8 x float> %1472, i64 1, !dbg !6
  %1475 = extractelement <8 x float> %1472, i64 2, !dbg !6
  %1476 = extractelement <8 x float> %1472, i64 3, !dbg !6
  %1477 = select i1 %.not, float %1474, float %1473, !dbg !6
  %1478 = bitcast float %1477 to i32, !dbg !6
  %1479 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1478), !dbg !6
  %1480 = bitcast i32 %1479 to float, !dbg !6
  %1481 = select i1 %.not, float %1473, float %1480, !dbg !6
  %1482 = select i1 %.not, float %1480, float %1474, !dbg !6
  %1483 = select i1 %.not, float %1476, float %1475, !dbg !6
  %1484 = bitcast float %1483 to i32, !dbg !6
  %1485 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1484), !dbg !6
  %1486 = bitcast i32 %1485 to float, !dbg !6
  %1487 = select i1 %.not, float %1475, float %1486, !dbg !6
  %1488 = select i1 %.not, float %1486, float %1476, !dbg !6
  %1489 = select i1 %.not84, float %1487, float %1481, !dbg !6
  %1490 = bitcast float %1489 to i32, !dbg !6
  %1491 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1490), !dbg !6
  %1492 = bitcast i32 %1491 to float, !dbg !6
  %1493 = select i1 %.not84, float %1481, float %1492, !dbg !6
  %1494 = select i1 %.not84, float %1492, float %1487, !dbg !6
  %1495 = select i1 %.not84, float %1488, float %1482, !dbg !6
  %1496 = bitcast float %1495 to i32, !dbg !6
  %1497 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1496), !dbg !6
  %1498 = bitcast i32 %1497 to float, !dbg !6
  %1499 = select i1 %.not84, float %1482, float %1498, !dbg !6
  %1500 = select i1 %.not84, float %1498, float %1488, !dbg !6
  %1501 = extractelement <8 x float> %1472, i64 4, !dbg !6
  %1502 = extractelement <8 x float> %1472, i64 5, !dbg !6
  %1503 = extractelement <8 x float> %1472, i64 6, !dbg !6
  %1504 = extractelement <8 x float> %1472, i64 7, !dbg !6
  %1505 = select i1 %.not, float %1502, float %1501, !dbg !6
  %1506 = bitcast float %1505 to i32, !dbg !6
  %1507 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1506), !dbg !6
  %1508 = bitcast i32 %1507 to float, !dbg !6
  %1509 = select i1 %.not, float %1501, float %1508, !dbg !6
  %1510 = select i1 %.not, float %1508, float %1502, !dbg !6
  %1511 = select i1 %.not, float %1504, float %1503, !dbg !6
  %1512 = bitcast float %1511 to i32, !dbg !6
  %1513 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1512), !dbg !6
  %1514 = bitcast i32 %1513 to float, !dbg !6
  %1515 = select i1 %.not, float %1503, float %1514, !dbg !6
  %1516 = select i1 %.not, float %1514, float %1504, !dbg !6
  %1517 = select i1 %.not84, float %1515, float %1509, !dbg !6
  %1518 = bitcast float %1517 to i32, !dbg !6
  %1519 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1518), !dbg !6
  %1520 = bitcast i32 %1519 to float, !dbg !6
  %1521 = select i1 %.not84, float %1509, float %1520, !dbg !6
  %1522 = select i1 %.not84, float %1520, float %1515, !dbg !6
  %1523 = select i1 %.not84, float %1516, float %1510, !dbg !6
  %1524 = bitcast float %1523 to i32, !dbg !6
  %1525 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1524), !dbg !6
  %1526 = bitcast i32 %1525 to float, !dbg !6
  %1527 = select i1 %.not84, float %1510, float %1526, !dbg !6
  %1528 = select i1 %.not84, float %1526, float %1516, !dbg !6
  %1529 = extractvalue [2 x <8 x float>] %103, 0, !dbg !6
  %1530 = extractelement <8 x float> %1529, i64 0, !dbg !6
  %1531 = extractelement <8 x float> %1529, i64 1, !dbg !6
  %1532 = extractelement <8 x float> %1529, i64 2, !dbg !6
  %1533 = extractelement <8 x float> %1529, i64 3, !dbg !6
  %1534 = select i1 %.not, float %1531, float %1530, !dbg !6
  %1535 = bitcast float %1534 to i32, !dbg !6
  %1536 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1535), !dbg !6
  %1537 = bitcast i32 %1536 to float, !dbg !6
  %1538 = select i1 %.not, float %1530, float %1537, !dbg !6
  %1539 = select i1 %.not, float %1537, float %1531, !dbg !6
  %1540 = select i1 %.not, float %1533, float %1532, !dbg !6
  %1541 = bitcast float %1540 to i32, !dbg !6
  %1542 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1541), !dbg !6
  %1543 = bitcast i32 %1542 to float, !dbg !6
  %1544 = select i1 %.not, float %1532, float %1543, !dbg !6
  %1545 = select i1 %.not, float %1543, float %1533, !dbg !6
  %1546 = select i1 %.not84, float %1544, float %1538, !dbg !6
  %1547 = bitcast float %1546 to i32, !dbg !6
  %1548 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1547), !dbg !6
  %1549 = select i1 %.not84, float %1545, float %1539, !dbg !6
  %1550 = bitcast float %1549 to i32, !dbg !6
  %1551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1550), !dbg !6
  %1552 = extractelement <8 x float> %1529, i64 4, !dbg !6
  %1553 = extractelement <8 x float> %1529, i64 5, !dbg !6
  %1554 = extractelement <8 x float> %1529, i64 6, !dbg !6
  %1555 = extractelement <8 x float> %1529, i64 7, !dbg !6
  %1556 = select i1 %.not, float %1553, float %1552, !dbg !6
  %1557 = bitcast float %1556 to i32, !dbg !6
  %1558 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1557), !dbg !6
  %1559 = bitcast i32 %1558 to float, !dbg !6
  %1560 = select i1 %.not, float %1552, float %1559, !dbg !6
  %1561 = select i1 %.not, float %1559, float %1553, !dbg !6
  %1562 = select i1 %.not, float %1555, float %1554, !dbg !6
  %1563 = bitcast float %1562 to i32, !dbg !6
  %1564 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1563), !dbg !6
  %1565 = bitcast i32 %1564 to float, !dbg !6
  %1566 = select i1 %.not, float %1554, float %1565, !dbg !6
  %1567 = select i1 %.not, float %1565, float %1555, !dbg !6
  %1568 = select i1 %.not84, float %1566, float %1560, !dbg !6
  %1569 = bitcast float %1568 to i32, !dbg !6
  %1570 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1569), !dbg !6
  %1571 = select i1 %.not84, float %1567, float %1561, !dbg !6
  %1572 = bitcast float %1571 to i32, !dbg !6
  %1573 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1572), !dbg !6
  %1574 = extractvalue [2 x <8 x float>] %103, 1, !dbg !6
  %1575 = extractelement <8 x float> %1574, i64 0, !dbg !6
  %1576 = extractelement <8 x float> %1574, i64 1, !dbg !6
  %1577 = extractelement <8 x float> %1574, i64 2, !dbg !6
  %1578 = extractelement <8 x float> %1574, i64 3, !dbg !6
  %1579 = select i1 %.not, float %1576, float %1575, !dbg !6
  %1580 = bitcast float %1579 to i32, !dbg !6
  %1581 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1580), !dbg !6
  %1582 = bitcast i32 %1581 to float, !dbg !6
  %1583 = select i1 %.not, float %1575, float %1582, !dbg !6
  %1584 = select i1 %.not, float %1582, float %1576, !dbg !6
  %1585 = select i1 %.not, float %1578, float %1577, !dbg !6
  %1586 = bitcast float %1585 to i32, !dbg !6
  %1587 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1586), !dbg !6
  %1588 = bitcast i32 %1587 to float, !dbg !6
  %1589 = select i1 %.not, float %1577, float %1588, !dbg !6
  %1590 = select i1 %.not, float %1588, float %1578, !dbg !6
  %1591 = select i1 %.not84, float %1589, float %1583, !dbg !6
  %1592 = bitcast float %1591 to i32, !dbg !6
  %1593 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1592), !dbg !6
  %1594 = select i1 %.not84, float %1590, float %1584, !dbg !6
  %1595 = bitcast float %1594 to i32, !dbg !6
  %1596 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1595), !dbg !6
  %1597 = extractelement <8 x float> %1574, i64 4, !dbg !6
  %1598 = extractelement <8 x float> %1574, i64 5, !dbg !6
  %1599 = extractelement <8 x float> %1574, i64 6, !dbg !6
  %1600 = extractelement <8 x float> %1574, i64 7, !dbg !6
  %1601 = select i1 %.not, float %1598, float %1597, !dbg !6
  %1602 = bitcast float %1601 to i32, !dbg !6
  %1603 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1602), !dbg !6
  %1604 = bitcast i32 %1603 to float, !dbg !6
  %1605 = select i1 %.not, float %1597, float %1604, !dbg !6
  %1606 = select i1 %.not, float %1604, float %1598, !dbg !6
  %1607 = select i1 %.not, float %1600, float %1599, !dbg !6
  %1608 = bitcast float %1607 to i32, !dbg !6
  %1609 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1608), !dbg !6
  %1610 = bitcast i32 %1609 to float, !dbg !6
  %1611 = select i1 %.not, float %1599, float %1610, !dbg !6
  %1612 = select i1 %.not, float %1610, float %1600, !dbg !6
  %1613 = select i1 %.not84, float %1611, float %1605, !dbg !6
  %1614 = bitcast float %1613 to i32, !dbg !6
  %1615 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1614), !dbg !6
  %1616 = select i1 %.not84, float %1612, float %1606, !dbg !6
  %1617 = bitcast float %1616 to i32, !dbg !6
  %1618 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1617), !dbg !6
  %1619 = insertelement <4 x float> poison, float %1184, i64 0, !dbg !16
  %1620 = fdiv arcp <4 x float> <float 0x3FF0000000000000, float poison, float poison, float poison>, %1619, !dbg !16
  %1621 = shufflevector <4 x float> %1620, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1622 = insertelement <4 x float> poison, float %1174, i64 0, !dbg !16
  %1623 = fdiv arcp <4 x float> <float 0x3FF0000000000000, float poison, float poison, float poison>, %1622, !dbg !16
  %1624 = shufflevector <4 x float> %1623, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1625 = bitcast i32 %1615 to float, !dbg !6
  %1626 = select i1 %.not84, float %1605, float %1625, !dbg !6
  %1627 = insertelement <4 x float> poison, float %1626, i64 0, !dbg !16
  %1628 = bitcast i32 %1618 to float, !dbg !6
  %1629 = select i1 %.not84, float %1606, float %1628, !dbg !6
  %1630 = insertelement <4 x float> %1627, float %1629, i64 1, !dbg !16
  %1631 = select i1 %.not84, float %1625, float %1611, !dbg !6
  %1632 = insertelement <4 x float> %1630, float %1631, i64 2, !dbg !16
  %1633 = select i1 %.not84, float %1628, float %1612, !dbg !6
  %1634 = insertelement <4 x float> %1632, float %1633, i64 3, !dbg !16
  %1635 = fmul <4 x float> %1634, %1624, !dbg !16
  %1636 = fptrunc <4 x float> %1635 to <4 x half>, !dbg !17
  %1637 = extractelement <4 x half> %1636, i64 3, !dbg !17
  %1638 = extractelement <4 x half> %1636, i64 2, !dbg !17
  %1639 = extractelement <4 x half> %1636, i64 1, !dbg !17
  %1640 = extractelement <4 x half> %1636, i64 0, !dbg !17
  %1641 = bitcast i32 %1593 to float, !dbg !6
  %1642 = select i1 %.not84, float %1583, float %1641, !dbg !6
  %1643 = insertelement <4 x float> poison, float %1642, i64 0, !dbg !16
  %1644 = bitcast i32 %1596 to float, !dbg !6
  %1645 = select i1 %.not84, float %1584, float %1644, !dbg !6
  %1646 = insertelement <4 x float> %1643, float %1645, i64 1, !dbg !16
  %1647 = select i1 %.not84, float %1641, float %1589, !dbg !6
  %1648 = insertelement <4 x float> %1646, float %1647, i64 2, !dbg !16
  %1649 = select i1 %.not84, float %1644, float %1590, !dbg !6
  %1650 = insertelement <4 x float> %1648, float %1649, i64 3, !dbg !16
  %1651 = fmul <4 x float> %1650, %1624, !dbg !16
  %1652 = fptrunc <4 x float> %1651 to <4 x half>, !dbg !17
  %1653 = extractelement <4 x half> %1652, i64 3, !dbg !17
  %1654 = extractelement <4 x half> %1652, i64 2, !dbg !17
  %1655 = extractelement <4 x half> %1652, i64 1, !dbg !17
  %1656 = extractelement <4 x half> %1652, i64 0, !dbg !17
  %1657 = bitcast i32 %1570 to float, !dbg !6
  %1658 = select i1 %.not84, float %1560, float %1657, !dbg !6
  %1659 = insertelement <4 x float> poison, float %1658, i64 0, !dbg !16
  %1660 = bitcast i32 %1573 to float, !dbg !6
  %1661 = select i1 %.not84, float %1561, float %1660, !dbg !6
  %1662 = insertelement <4 x float> %1659, float %1661, i64 1, !dbg !16
  %1663 = select i1 %.not84, float %1657, float %1566, !dbg !6
  %1664 = insertelement <4 x float> %1662, float %1663, i64 2, !dbg !16
  %1665 = select i1 %.not84, float %1660, float %1567, !dbg !6
  %1666 = insertelement <4 x float> %1664, float %1665, i64 3, !dbg !16
  %1667 = fmul <4 x float> %1666, %1621, !dbg !16
  %1668 = fptrunc <4 x float> %1667 to <4 x half>, !dbg !17
  %1669 = extractelement <4 x half> %1668, i64 3, !dbg !17
  %1670 = extractelement <4 x half> %1668, i64 2, !dbg !17
  %1671 = extractelement <4 x half> %1668, i64 1, !dbg !17
  %1672 = extractelement <4 x half> %1668, i64 0, !dbg !17
  %1673 = bitcast i32 %1548 to float, !dbg !6
  %1674 = select i1 %.not84, float %1538, float %1673, !dbg !6
  %1675 = insertelement <4 x float> poison, float %1674, i64 0, !dbg !16
  %1676 = bitcast i32 %1551 to float, !dbg !6
  %1677 = select i1 %.not84, float %1539, float %1676, !dbg !6
  %1678 = insertelement <4 x float> %1675, float %1677, i64 1, !dbg !16
  %1679 = select i1 %.not84, float %1673, float %1544, !dbg !6
  %1680 = insertelement <4 x float> %1678, float %1679, i64 2, !dbg !16
  %1681 = select i1 %.not84, float %1676, float %1545, !dbg !6
  %1682 = insertelement <4 x float> %1680, float %1681, i64 3, !dbg !16
  %1683 = fmul <4 x float> %1682, %1621, !dbg !16
  %1684 = fptrunc <4 x float> %1683 to <4 x half>, !dbg !17
  %1685 = extractelement <4 x half> %1684, i64 3, !dbg !17
  %1686 = extractelement <4 x half> %1684, i64 2, !dbg !17
  %1687 = extractelement <4 x half> %1684, i64 1, !dbg !17
  %1688 = extractelement <4 x half> %1684, i64 0, !dbg !17
  %address.thread4196 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !18, !range !30, !frisk.register_lifetimes !1
  %address.local4197 = zext nneg i32 %address.thread4196 to i64, !dbg !9
  %address.local4198 = add nuw i64 %30, %address.local4197, !dbg !18
  %1689 = lshr i64 %address.local4197, 2, !dbg !18
  %address.local4200 = and i64 %1689, 48, !dbg !18
  %address.local4201 = add i64 %address.local4198, %address.local4200, !dbg !18
  %address.local4202 = and i64 %address.local4197, 240, !dbg !18
  %address.local4203 = sub i64 %address.local4201, %address.local4202, !dbg !18
  %address.local4204 = add i64 %address.local4203, %31, !dbg !18
  %address.local4205 = shl nuw nsw i64 %address.local4204, 7, !dbg !18
  %1690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %46, !dbg !18
  %address.local4207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1690, i64 %address.local4205, !dbg !18
  %address.local4210 = and i64 %1689, 12, !dbg !12
  %address.local4211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4210, !dbg !18
  store half %1688, ptr addrspace(1) %address.local4211, align 2, !dbg !18
  %address.local4213 = or disjoint i64 %address.local4210, 1, !dbg !18
  %address.local4214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4213, !dbg !18
  store half %1687, ptr addrspace(1) %address.local4214, align 2, !dbg !18
  %address.local4216 = or disjoint i64 %address.local4210, 2, !dbg !18
  %address.local4217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4216, !dbg !18
  store half %1686, ptr addrspace(1) %address.local4217, align 2, !dbg !18
  %address.local4219 = or disjoint i64 %address.local4210, 3, !dbg !18
  %address.local4220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4219, !dbg !18
  store half %1685, ptr addrspace(1) %address.local4220, align 2, !dbg !18
  %address.local4222 = or disjoint i64 %address.local4210, 16, !dbg !18
  %address.local4223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4222, !dbg !18
  store half %1672, ptr addrspace(1) %address.local4223, align 2, !dbg !18
  %address.local4225 = or disjoint i64 %address.local4210, 17, !dbg !18
  %address.local4226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4225, !dbg !18
  store half %1671, ptr addrspace(1) %address.local4226, align 2, !dbg !18
  %address.local4228 = or disjoint i64 %address.local4210, 18, !dbg !18
  %address.local4229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4228, !dbg !18
  store half %1670, ptr addrspace(1) %address.local4229, align 2, !dbg !18
  %address.local4231 = or disjoint i64 %address.local4210, 19, !dbg !18
  %address.local4232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4231, !dbg !18
  store half %1669, ptr addrspace(1) %address.local4232, align 2, !dbg !18
  %address.local4236 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local4207, i64 16384, !dbg !18
  %address.local4237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4210, !dbg !18
  store half %1656, ptr addrspace(1) %address.local4237, align 2, !dbg !18
  %address.local4239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4213, !dbg !18
  store half %1655, ptr addrspace(1) %address.local4239, align 2, !dbg !18
  %address.local4241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4216, !dbg !18
  store half %1654, ptr addrspace(1) %address.local4241, align 2, !dbg !18
  %address.local4243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4219, !dbg !18
  store half %1653, ptr addrspace(1) %address.local4243, align 2, !dbg !18
  %address.local4245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4222, !dbg !18
  store half %1640, ptr addrspace(1) %address.local4245, align 2, !dbg !18
  %address.local4247 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4225, !dbg !18
  store half %1639, ptr addrspace(1) %address.local4247, align 2, !dbg !18
  %address.local4249 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4228, !dbg !18
  store half %1638, ptr addrspace(1) %address.local4249, align 2, !dbg !18
  %address.local4251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4231, !dbg !18
  store half %1637, ptr addrspace(1) %address.local4251, align 2, !dbg !18
  %1691 = insertelement <4 x float> poison, float %1521, i64 0, !dbg !16
  %1692 = insertelement <4 x float> %1691, float %1527, i64 1, !dbg !16
  %1693 = insertelement <4 x float> %1692, float %1522, i64 2, !dbg !16
  %1694 = insertelement <4 x float> %1693, float %1528, i64 3, !dbg !16
  %1695 = fmul <4 x float> %1694, %1624, !dbg !16
  %1696 = fptrunc <4 x float> %1695 to <4 x half>, !dbg !17
  %1697 = extractelement <4 x half> %1696, i64 3, !dbg !17
  %1698 = extractelement <4 x half> %1696, i64 2, !dbg !17
  %1699 = extractelement <4 x half> %1696, i64 1, !dbg !17
  %1700 = extractelement <4 x half> %1696, i64 0, !dbg !17
  %1701 = insertelement <4 x float> poison, float %1493, i64 0, !dbg !16
  %1702 = insertelement <4 x float> %1701, float %1499, i64 1, !dbg !16
  %1703 = insertelement <4 x float> %1702, float %1494, i64 2, !dbg !16
  %1704 = insertelement <4 x float> %1703, float %1500, i64 3, !dbg !16
  %1705 = fmul <4 x float> %1704, %1624, !dbg !16
  %1706 = fptrunc <4 x float> %1705 to <4 x half>, !dbg !17
  %1707 = extractelement <4 x half> %1706, i64 3, !dbg !17
  %1708 = extractelement <4 x half> %1706, i64 2, !dbg !17
  %1709 = extractelement <4 x half> %1706, i64 1, !dbg !17
  %1710 = extractelement <4 x half> %1706, i64 0, !dbg !17
  %1711 = insertelement <4 x float> poison, float %1464, i64 0, !dbg !16
  %1712 = insertelement <4 x float> %1711, float %1470, i64 1, !dbg !16
  %1713 = insertelement <4 x float> %1712, float %1465, i64 2, !dbg !16
  %1714 = insertelement <4 x float> %1713, float %1471, i64 3, !dbg !16
  %1715 = fmul <4 x float> %1714, %1621, !dbg !16
  %1716 = fptrunc <4 x float> %1715 to <4 x half>, !dbg !17
  %1717 = extractelement <4 x half> %1716, i64 3, !dbg !17
  %1718 = extractelement <4 x half> %1716, i64 2, !dbg !17
  %1719 = extractelement <4 x half> %1716, i64 1, !dbg !17
  %1720 = extractelement <4 x half> %1716, i64 0, !dbg !17
  %1721 = insertelement <4 x float> poison, float %1436, i64 0, !dbg !16
  %1722 = insertelement <4 x float> %1721, float %1442, i64 1, !dbg !16
  %1723 = insertelement <4 x float> %1722, float %1437, i64 2, !dbg !16
  %1724 = insertelement <4 x float> %1723, float %1443, i64 3, !dbg !16
  %1725 = fmul <4 x float> %1724, %1621, !dbg !16
  %1726 = fptrunc <4 x float> %1725 to <4 x half>, !dbg !17
  %1727 = extractelement <4 x half> %1726, i64 3, !dbg !17
  %1728 = extractelement <4 x half> %1726, i64 2, !dbg !17
  %1729 = extractelement <4 x half> %1726, i64 1, !dbg !17
  %1730 = extractelement <4 x half> %1726, i64 0, !dbg !17
  %address.local4267 = or disjoint i64 %address.local4210, 32, !dbg !18
  %address.local4268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4267, !dbg !18
  store half %1730, ptr addrspace(1) %address.local4268, align 2, !dbg !18
  %address.local4270 = or disjoint i64 %address.local4210, 33, !dbg !18
  %address.local4271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4270, !dbg !18
  store half %1729, ptr addrspace(1) %address.local4271, align 2, !dbg !18
  %address.local4273 = or disjoint i64 %address.local4210, 34, !dbg !18
  %address.local4274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4273, !dbg !18
  store half %1728, ptr addrspace(1) %address.local4274, align 2, !dbg !18
  %address.local4276 = or disjoint i64 %address.local4210, 35, !dbg !18
  %address.local4277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4276, !dbg !18
  store half %1727, ptr addrspace(1) %address.local4277, align 2, !dbg !18
  %address.local4279 = or disjoint i64 %address.local4210, 48, !dbg !18
  %address.local4280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4279, !dbg !18
  store half %1720, ptr addrspace(1) %address.local4280, align 2, !dbg !18
  %address.local4282 = or disjoint i64 %address.local4210, 49, !dbg !18
  %address.local4283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4282, !dbg !18
  store half %1719, ptr addrspace(1) %address.local4283, align 2, !dbg !18
  %address.local4285 = or disjoint i64 %address.local4210, 50, !dbg !18
  %address.local4286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4285, !dbg !18
  store half %1718, ptr addrspace(1) %address.local4286, align 2, !dbg !18
  %address.local4288 = or i64 %1689, 51, !dbg !18
  %address.local4289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4288, !dbg !18
  store half %1717, ptr addrspace(1) %address.local4289, align 2, !dbg !18
  %address.local4294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4267, !dbg !18
  store half %1710, ptr addrspace(1) %address.local4294, align 2, !dbg !18
  %address.local4296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4270, !dbg !18
  store half %1709, ptr addrspace(1) %address.local4296, align 2, !dbg !18
  %address.local4298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4273, !dbg !18
  store half %1708, ptr addrspace(1) %address.local4298, align 2, !dbg !18
  %address.local4300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4276, !dbg !18
  store half %1707, ptr addrspace(1) %address.local4300, align 2, !dbg !18
  %address.local4302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4279, !dbg !18
  store half %1700, ptr addrspace(1) %address.local4302, align 2, !dbg !18
  %address.local4304 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4282, !dbg !18
  store half %1699, ptr addrspace(1) %address.local4304, align 2, !dbg !18
  %address.local4306 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4285, !dbg !18
  store half %1698, ptr addrspace(1) %address.local4306, align 2, !dbg !18
  %address.local4308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4288, !dbg !18
  store half %1697, ptr addrspace(1) %address.local4308, align 2, !dbg !18
  %1731 = insertelement <4 x float> poison, float %1407, i64 0, !dbg !16
  %1732 = insertelement <4 x float> %1731, float %1413, i64 1, !dbg !16
  %1733 = insertelement <4 x float> %1732, float %1408, i64 2, !dbg !16
  %1734 = insertelement <4 x float> %1733, float %1414, i64 3, !dbg !16
  %1735 = fmul <4 x float> %1734, %1624, !dbg !16
  %1736 = fptrunc <4 x float> %1735 to <4 x half>, !dbg !17
  %1737 = extractelement <4 x half> %1736, i64 3, !dbg !17
  %1738 = extractelement <4 x half> %1736, i64 2, !dbg !17
  %1739 = extractelement <4 x half> %1736, i64 1, !dbg !17
  %1740 = extractelement <4 x half> %1736, i64 0, !dbg !17
  %1741 = insertelement <4 x float> poison, float %1379, i64 0, !dbg !16
  %1742 = insertelement <4 x float> %1741, float %1385, i64 1, !dbg !16
  %1743 = insertelement <4 x float> %1742, float %1380, i64 2, !dbg !16
  %1744 = insertelement <4 x float> %1743, float %1386, i64 3, !dbg !16
  %1745 = fmul <4 x float> %1744, %1624, !dbg !16
  %1746 = fptrunc <4 x float> %1745 to <4 x half>, !dbg !17
  %1747 = extractelement <4 x half> %1746, i64 3, !dbg !17
  %1748 = extractelement <4 x half> %1746, i64 2, !dbg !17
  %1749 = extractelement <4 x half> %1746, i64 1, !dbg !17
  %1750 = extractelement <4 x half> %1746, i64 0, !dbg !17
  %1751 = insertelement <4 x float> poison, float %1350, i64 0, !dbg !16
  %1752 = insertelement <4 x float> %1751, float %1356, i64 1, !dbg !16
  %1753 = insertelement <4 x float> %1752, float %1351, i64 2, !dbg !16
  %1754 = insertelement <4 x float> %1753, float %1357, i64 3, !dbg !16
  %1755 = fmul <4 x float> %1754, %1621, !dbg !16
  %1756 = fptrunc <4 x float> %1755 to <4 x half>, !dbg !17
  %1757 = extractelement <4 x half> %1756, i64 3, !dbg !17
  %1758 = extractelement <4 x half> %1756, i64 2, !dbg !17
  %1759 = extractelement <4 x half> %1756, i64 1, !dbg !17
  %1760 = extractelement <4 x half> %1756, i64 0, !dbg !17
  %1761 = insertelement <4 x float> poison, float %1322, i64 0, !dbg !16
  %1762 = insertelement <4 x float> %1761, float %1328, i64 1, !dbg !16
  %1763 = insertelement <4 x float> %1762, float %1323, i64 2, !dbg !16
  %1764 = insertelement <4 x float> %1763, float %1329, i64 3, !dbg !16
  %1765 = fmul <4 x float> %1764, %1621, !dbg !16
  %1766 = fptrunc <4 x float> %1765 to <4 x half>, !dbg !17
  %1767 = extractelement <4 x half> %1766, i64 3, !dbg !17
  %1768 = extractelement <4 x half> %1766, i64 2, !dbg !17
  %1769 = extractelement <4 x half> %1766, i64 1, !dbg !17
  %1770 = extractelement <4 x half> %1766, i64 0, !dbg !17
  %address.local4324 = or disjoint i64 %address.local4210, 64, !dbg !18
  %address.local4325 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4324, !dbg !18
  store half %1770, ptr addrspace(1) %address.local4325, align 2, !dbg !18
  %address.local4327 = or disjoint i64 %address.local4210, 65, !dbg !18
  %address.local4328 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4327, !dbg !18
  store half %1769, ptr addrspace(1) %address.local4328, align 2, !dbg !18
  %address.local4330 = or disjoint i64 %address.local4210, 66, !dbg !18
  %address.local4331 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4330, !dbg !18
  store half %1768, ptr addrspace(1) %address.local4331, align 2, !dbg !18
  %address.local4333 = or disjoint i64 %address.local4210, 67, !dbg !18
  %address.local4334 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4333, !dbg !18
  store half %1767, ptr addrspace(1) %address.local4334, align 2, !dbg !18
  %address.local4336 = or disjoint i64 %address.local4210, 80, !dbg !18
  %address.local4337 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4336, !dbg !18
  store half %1760, ptr addrspace(1) %address.local4337, align 2, !dbg !18
  %address.local4339 = or disjoint i64 %address.local4210, 81, !dbg !18
  %address.local4340 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4339, !dbg !18
  store half %1759, ptr addrspace(1) %address.local4340, align 2, !dbg !18
  %address.local4342 = or disjoint i64 %address.local4210, 82, !dbg !18
  %address.local4343 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4342, !dbg !18
  store half %1758, ptr addrspace(1) %address.local4343, align 2, !dbg !18
  %address.local4345 = or disjoint i64 %address.local4210, 83, !dbg !18
  %address.local4346 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4345, !dbg !18
  store half %1757, ptr addrspace(1) %address.local4346, align 2, !dbg !18
  %address.local4351 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4324, !dbg !18
  store half %1750, ptr addrspace(1) %address.local4351, align 2, !dbg !18
  %address.local4353 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4327, !dbg !18
  store half %1749, ptr addrspace(1) %address.local4353, align 2, !dbg !18
  %address.local4355 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4330, !dbg !18
  store half %1748, ptr addrspace(1) %address.local4355, align 2, !dbg !18
  %address.local4357 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4333, !dbg !18
  store half %1747, ptr addrspace(1) %address.local4357, align 2, !dbg !18
  %address.local4359 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4336, !dbg !18
  store half %1740, ptr addrspace(1) %address.local4359, align 2, !dbg !18
  %address.local4361 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4339, !dbg !18
  store half %1739, ptr addrspace(1) %address.local4361, align 2, !dbg !18
  %address.local4363 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4342, !dbg !18
  store half %1738, ptr addrspace(1) %address.local4363, align 2, !dbg !18
  %address.local4365 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4345, !dbg !18
  store half %1737, ptr addrspace(1) %address.local4365, align 2, !dbg !18
  %1771 = insertelement <4 x float> poison, float %1293, i64 0, !dbg !16
  %1772 = insertelement <4 x float> %1771, float %1299, i64 1, !dbg !16
  %1773 = insertelement <4 x float> %1772, float %1294, i64 2, !dbg !16
  %1774 = insertelement <4 x float> %1773, float %1300, i64 3, !dbg !16
  %1775 = fmul <4 x float> %1774, %1624, !dbg !16
  %1776 = fptrunc <4 x float> %1775 to <4 x half>, !dbg !17
  %1777 = extractelement <4 x half> %1776, i64 3, !dbg !17
  %1778 = extractelement <4 x half> %1776, i64 2, !dbg !17
  %1779 = extractelement <4 x half> %1776, i64 1, !dbg !17
  %1780 = extractelement <4 x half> %1776, i64 0, !dbg !17
  %1781 = insertelement <4 x float> poison, float %1265, i64 0, !dbg !16
  %1782 = insertelement <4 x float> %1781, float %1271, i64 1, !dbg !16
  %1783 = insertelement <4 x float> %1782, float %1266, i64 2, !dbg !16
  %1784 = insertelement <4 x float> %1783, float %1272, i64 3, !dbg !16
  %1785 = fmul <4 x float> %1784, %1624, !dbg !16
  %1786 = fptrunc <4 x float> %1785 to <4 x half>, !dbg !17
  %1787 = extractelement <4 x half> %1786, i64 3, !dbg !17
  %1788 = extractelement <4 x half> %1786, i64 2, !dbg !17
  %1789 = extractelement <4 x half> %1786, i64 1, !dbg !17
  %1790 = extractelement <4 x half> %1786, i64 0, !dbg !17
  %1791 = insertelement <4 x float> poison, float %1236, i64 0, !dbg !16
  %1792 = insertelement <4 x float> %1791, float %1242, i64 1, !dbg !16
  %1793 = insertelement <4 x float> %1792, float %1237, i64 2, !dbg !16
  %1794 = insertelement <4 x float> %1793, float %1243, i64 3, !dbg !16
  %1795 = fmul <4 x float> %1794, %1621, !dbg !16
  %1796 = fptrunc <4 x float> %1795 to <4 x half>, !dbg !17
  %1797 = extractelement <4 x half> %1796, i64 3, !dbg !17
  %1798 = extractelement <4 x half> %1796, i64 2, !dbg !17
  %1799 = extractelement <4 x half> %1796, i64 1, !dbg !17
  %1800 = extractelement <4 x half> %1796, i64 0, !dbg !17
  %1801 = insertelement <4 x float> poison, float %1208, i64 0, !dbg !16
  %1802 = insertelement <4 x float> %1801, float %1214, i64 1, !dbg !16
  %1803 = insertelement <4 x float> %1802, float %1209, i64 2, !dbg !16
  %1804 = insertelement <4 x float> %1803, float %1215, i64 3, !dbg !16
  %1805 = fmul <4 x float> %1804, %1621, !dbg !16
  %1806 = fptrunc <4 x float> %1805 to <4 x half>, !dbg !17
  %1807 = extractelement <4 x half> %1806, i64 3, !dbg !17
  %1808 = extractelement <4 x half> %1806, i64 2, !dbg !17
  %1809 = extractelement <4 x half> %1806, i64 1, !dbg !17
  %1810 = extractelement <4 x half> %1806, i64 0, !dbg !17
  %address.local4381 = or disjoint i64 %address.local4210, 96, !dbg !18
  %address.local4382 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4381, !dbg !18
  store half %1810, ptr addrspace(1) %address.local4382, align 2, !dbg !18
  %address.local4384 = or disjoint i64 %address.local4210, 97, !dbg !18
  %address.local4385 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4384, !dbg !18
  store half %1809, ptr addrspace(1) %address.local4385, align 2, !dbg !18
  %address.local4387 = or disjoint i64 %address.local4210, 98, !dbg !18
  %address.local4388 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4387, !dbg !18
  store half %1808, ptr addrspace(1) %address.local4388, align 2, !dbg !18
  %address.local4390 = or disjoint i64 %address.local4210, 99, !dbg !18
  %address.local4391 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4390, !dbg !18
  store half %1807, ptr addrspace(1) %address.local4391, align 2, !dbg !18
  %address.local4393 = or disjoint i64 %address.local4210, 112, !dbg !18
  %address.local4394 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4393, !dbg !18
  store half %1800, ptr addrspace(1) %address.local4394, align 2, !dbg !18
  %address.local4396 = or disjoint i64 %address.local4210, 113, !dbg !18
  %address.local4397 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4396, !dbg !18
  store half %1799, ptr addrspace(1) %address.local4397, align 2, !dbg !18
  %address.local4399 = or disjoint i64 %address.local4210, 114, !dbg !18
  %address.local4400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4399, !dbg !18
  store half %1798, ptr addrspace(1) %address.local4400, align 2, !dbg !18
  %address.local4402 = or i64 %1689, 115, !dbg !18
  %address.local4403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4402, !dbg !18
  store half %1797, ptr addrspace(1) %address.local4403, align 2, !dbg !18
  %address.local4408 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4381, !dbg !18
  store half %1790, ptr addrspace(1) %address.local4408, align 2, !dbg !18
  %address.local4410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4384, !dbg !18
  store half %1789, ptr addrspace(1) %address.local4410, align 2, !dbg !18
  %address.local4412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4387, !dbg !18
  store half %1788, ptr addrspace(1) %address.local4412, align 2, !dbg !18
  %address.local4414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4390, !dbg !18
  store half %1787, ptr addrspace(1) %address.local4414, align 2, !dbg !18
  %address.local4416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4393, !dbg !18
  store half %1780, ptr addrspace(1) %address.local4416, align 2, !dbg !18
  %address.local4418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4396, !dbg !18
  store half %1779, ptr addrspace(1) %address.local4418, align 2, !dbg !18
  %address.local4420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4399, !dbg !18
  store half %1778, ptr addrspace(1) %address.local4420, align 2, !dbg !18
  %address.local4422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4402, !dbg !18
  store half %1777, ptr addrspace(1) %address.local4422, align 2, !dbg !18
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

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.ballot.i64(i1) #4

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.lazy_softmax_base"="v1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_cvt_pack"="rtz" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
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
!llvm.module.flags = !{!29}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/shared_memory_plan/noncausal")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!6 = !DILocation(line: 57, column: 24, scope: !4)
!8 = !DILocation(line: 74, column: 20, scope: !4)
!9 = !DILocation(line: 67, column: 22, scope: !4)
!12 = !DILocation(line: 69, column: 24, scope: !4)
!13 = !DILocation(line: 68, column: 22, scope: !4)
!14 = !DILocation(line: 89, column: 20, scope: !4)
!15 = !DILocation(line: 71, column: 23, scope: !4)
!16 = !DILocation(line: 92, column: 23, scope: !4)
!17 = !DILocation(line: 93, column: 19, scope: !4)
!18 = !DILocation(line: 94, column: 9, scope: !4)
!19 = !DILocation(line: 86, column: 20, scope: !4)
!23 = !DILocation(line: 78, column: 22, scope: !4)
!24 = !DILocation(line: 79, column: 22, scope: !4)
!25 = !DILocation(line: 80, column: 20, scope: !4)
!26 = !DILocation(line: 81, column: 16, scope: !4)
!27 = !DILocation(line: 88, column: 22, scope: !4)
!28 = !{i32 256, i32 1, i32 1}
!29 = !{i32 2, !"Debug Info Version", i32 3}
!30 = !{i32 0, i32 256}
!31 = !{i32 4}
!32 = !{i32 0}
!33 = !{i32 1}
!34 = !{i32 2}
!35 = !{i32 3}
!36 = !{i32 5}
!37 = !{i32 6}
!38 = !{i32 7}
!39 = !{i32 8}
!40 = !{i32 9}
!41 = !{i32 10}
!42 = !{i32 11}
!43 = !{i32 12}
!44 = !{i32 13}
!45 = !{i32 14}
!46 = !{i32 15}
!47 = !{i32 16}
!48 = !{i32 17}
!49 = !{i32 18}
!50 = !{i32 19}
!51 = !{i32 20}
!52 = !{i32 21}
!53 = !{i32 22}
!54 = !{i32 23}
