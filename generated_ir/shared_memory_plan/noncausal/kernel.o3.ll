; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/shared_memory_plan/noncausal/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/shared_memory_plan/noncausal/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %5 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %6 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %5), !dbg !7
  %7 = add nuw nsw i32 %6, 64, !dbg !7
  %8 = xor i32 %6, 16, !dbg !7
  %9 = and i32 %7, 192, !dbg !7
  %10 = icmp samesign ult i32 %8, %9, !dbg !7
  %11 = select i1 %10, i32 %8, i32 %6, !dbg !7
  %12 = shl nuw nsw i32 %11, 2, !dbg !7
  %13 = xor i32 %6, 32, !dbg !7
  %14 = icmp samesign ult i32 %13, %9, !dbg !7
  %15 = select i1 %14, i32 %13, i32 %6, !dbg !7
  %16 = shl nuw nsw i32 %15, 2, !dbg !7
  %17 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !8
  %18 = sext i32 %17 to i64, !dbg !8
  %19 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !8
  %20 = sext i32 %19 to i64, !dbg !8
  %21 = shl nsw i64 %20, 5, !dbg !8
  %22 = add nsw i64 %21, %18, !dbg !8
  %23 = lshr i64 %22, 5, !dbg !8
  %24 = and i64 %23, 576460752303423456, !dbg !8
  %25 = and i64 %18, 31, !dbg !8
  %26 = or disjoint i64 %24, %25, !dbg !8
  %27 = lshr i64 %22, 10, !dbg !9
  %28 = add nuw nsw i64 %23, %27, !dbg !9
  %29 = and i64 %28, 31, !dbg !9
  %30 = shl nuw nsw i64 %26, 7, !dbg !9
  %31 = shl i64 %27, 12, !dbg !9
  %32 = sub i64 %30, %31, !dbg !9
  %33 = shl nsw i64 %32, 7, !dbg !9
  %34 = tail call range(i32 0, 256) i32 @llvm.amdgcn.workitem.id.x(), !dbg !9
  %35 = zext nneg i32 %34 to i64, !dbg !9
  %.idx = shl nuw nsw i64 %29, 20, !dbg !9
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %34) #8, !dbg !9, !range !10, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !9
  %address.local874 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local875 = and i64 %address.local874, 1536, !dbg !9
  %address.local876 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !9
  %address.local879 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local880 = and i64 %address.local879, 16256, !dbg !9
  %address.local882 = lshr i64 %address.local, 4, !dbg !9
  %address.local883 = and i64 %address.local882, 8, !dbg !9
  %36 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx4623 = shl i64 %32, 8, !dbg !9
  %37 = add i64 %.idx4623, %.idx, !dbg !9
  %address.local881.idx4624 = or disjoint i64 %address.local880, %address.local883, !dbg !9
  %38 = shl nuw nsw i64 %address.local881.idx4624, 1, !dbg !9
  %39 = or disjoint i64 %38, %37, !dbg !9
  %dma.buffer.offset4589 = trunc i64 %39 to i32, !dbg !9
  %40 = insertelement <2 x i64> poison, i64 %36, i64 0, !dbg !9
  %41 = bitcast <2 x i64> %40 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc4590 = shufflevector <4 x i32> %41, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local876, i64 4096, !dbg !9
  %dma.buffer.offset4587 = or disjoint i32 %dma.buffer.offset4589, 32, !dbg !9
  %42 = shl nuw nsw i64 %29, 19, !dbg !9
  %43 = add i64 %33, %42, !dbg !9
  %address.local898 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local875, !dbg !9
  %44 = or disjoint i64 %address.local881.idx4624, %43, !dbg !9
  %45 = trunc i64 %44 to i32, !dbg !9
  %.tr = shl i32 %45, 1, !dbg !9
  %dma.buffer.offset4585 = or disjoint i32 %.tr, 64, !dbg !9
  %address.local910 = getelementptr i8, ptr addrspace(3) %address.local898, i64 4096, !dbg !9
  %dma.buffer.offset4583 = or disjoint i32 %.tr, 96, !dbg !9
  %46 = lshr i64 %35, 6, !dbg !11
  %address.local917 = lshr i32 %address.thread, 2, !dbg !11
  %address.local918 = and i32 %address.local917, 4, !dbg !11
  %address.local919 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local920 = and i32 %address.local919, 384, !dbg !11
  %address.local921 = or disjoint i32 %address.local918, %address.local920, !dbg !11
  %address.local922 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local923 = and i32 %address.local922, 120, !dbg !11
  %address.local924 = or disjoint i32 %address.local921, %address.local923, !dbg !11
  %address.local925 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local926 = and i32 %address.local925, 1024, !dbg !11
  %address.local927 = or disjoint i32 %address.local924, %address.local926, !dbg !11
  %address.local928 = zext nneg i32 %address.local927 to i64, !dbg !11
  %address.local929 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local928, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local929, i64 4096, !dbg !11
  %lds.const.gep4423 = getelementptr i8, ptr addrspace(3) %address.local929, i64 1024, !dbg !11
  %lds.const.gep4424 = getelementptr i8, ptr addrspace(3) %address.local929, i64 5120, !dbg !11
  %dma.buffer.offset4581 = or disjoint i32 %.tr, 128, !dbg !9
  %dma.buffer.offset4579 = or disjoint i32 %.tr, 160, !dbg !9
  %lds.const.gep4425 = getelementptr i8, ptr addrspace(3) %address.local929, i64 8192, !dbg !11
  %lds.const.gep4426 = getelementptr i8, ptr addrspace(3) %address.local929, i64 12288, !dbg !11
  %lds.const.gep4427 = getelementptr i8, ptr addrspace(3) %address.local929, i64 9216, !dbg !11
  %lds.const.gep4428 = getelementptr i8, ptr addrspace(3) %address.local929, i64 13312, !dbg !11
  %dma.buffer.offset4577 = or disjoint i32 %.tr, 192, !dbg !9
  %dma.buffer.offset4575 = or disjoint i32 %.tr, 224, !dbg !9
  %k0.prime.address.local1134 = shl nuw nsw i64 %address.local, 2, !dbg !12
  %k0.prime.address.local1135 = and i64 %k0.prime.address.local1134, 768, !dbg !12
  %k0.prime.address.local1136 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.prime.address.local1135, !dbg !12
  %k0.prime.4786 = shl i64 %23, 19, !dbg !12
  %k0.prime.4787 = and i64 %k0.prime.4786, 16252928, !dbg !12
  %k0.prime.4790155 = add nuw nsw i64 %46, %35, !dbg !12
  %k0.prime.4791 = shl nuw nsw i64 %k0.prime.4790155, 2, !dbg !12
  %k0.prime.4792 = add nuw nsw i64 %46, %35, !dbg !12
  %k0.prime.4793 = shl nuw nsw i64 %k0.prime.4792, 2, !dbg !12
  %k0.prime.4794 = and i64 %k0.prime.4793, 2040, !dbg !12
  %k0.prime.4795 = sub nsw i64 %k0.prime.4791, %k0.prime.4794, !dbg !12
  %k0.prime.4796 = shl nuw nsw i32 %34, 4, !dbg !12
  %k0.prime.4797 = and i32 %k0.prime.4796, 3584, !dbg !12
  %k0.prime.4798 = shl nuw nsw i32 %34, 6, !dbg !12
  %k0.prime.4799 = and i32 %k0.prime.4798, 16256, !dbg !12
  %k0.prime.narrow805 = add nuw nsw i32 %k0.prime.4797, %k0.prime.4799, !dbg !12
  %k0.prime.4800 = zext nneg i32 %k0.prime.narrow805 to i64, !dbg !12
  %k0.prime.4801 = lshr i32 %34, 6, !dbg !12
  %47 = and i32 %34, 40, !dbg !12
  %48 = icmp eq i32 %47, 40, !dbg !12
  %.neg = sext i1 %48 to i32, !dbg !12
  %49 = zext i1 %48 to i32, !dbg !12
  %k0.prime.narrow804 = add nuw nsw i32 %k0.prime.4801, %49, !dbg !12
  %k0.prime.4806 = shl nuw nsw i32 %k0.prime.narrow804, 10, !dbg !12
  %k0.prime.4807 = zext nneg i32 %k0.prime.4806 to i64, !dbg !12
  %k0.prime.4808 = sub nsw i64 %k0.prime.4800, %k0.prime.4807, !dbg !12
  %k0.prime.4809 = and i64 %k0.prime.4808, 16256, !dbg !12
  %k0.prime.4810 = add nsw i64 %k0.prime.4795, %k0.prime.4809, !dbg !12
  %k0.prime..idx46434646 = add nsw i64 %k0.prime.4810, %k0.prime.4787, !dbg !12
  %k0.prime..idx46434646.tr = trunc nsw i64 %k0.prime..idx46434646 to i32, !dbg !12
  %k0.prime.dma.buffer.offset4621 = shl nsw i32 %k0.prime..idx46434646.tr, 1, !dbg !12
  %k0.prime.4811 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %k0.prime.4812 = insertelement <2 x i64> poison, i64 %k0.prime.4811, i64 0, !dbg !12
  %k0.prime.4813 = bitcast <2 x i64> %k0.prime.4812 to <4 x i32>, !dbg !12
  %k0.prime.dma.buffer.rsrc4622 = shufflevector <4 x i32> %k0.prime.4813, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %k0.prime.address.local1142 = getelementptr i8, ptr addrspace(3) %k0.prime.address.local1136, i64 2048, !dbg !12
  %k0.prime..idx4647 = shl nuw nsw i64 %k0.prime.4787, 1, !dbg !12
  %k0.prime.4820 = add nuw nsw i64 %k0.prime.4793, 1040, !dbg !12
  %k0.prime.4821 = and i64 %k0.prime.4820, 8184, !dbg !12
  %k0.prime.4817 = add nuw nsw i64 %k0.prime.4791, 1040, !dbg !12
  %k0.prime.4822 = sub nsw i64 %k0.prime.4817, %k0.prime.4821, !dbg !12
  %k0.prime..idx4648 = shl nsw i64 %k0.prime.4822, 1, !dbg !12
  %k0.prime.4823 = add nsw i64 %k0.prime..idx4648, 16, !dbg !12
  %k0.prime.4824 = shl nuw nsw i32 %34, 5, !dbg !12
  %k0.prime.4825 = and i32 %k0.prime.4824, 7168, !dbg !12
  %k0.prime.4826 = or disjoint i32 %k0.prime.4825, 8192, !dbg !12
  %k0.prime.4827 = shl nuw nsw i32 %34, 7, !dbg !12
  %k0.prime.4828 = and i32 %k0.prime.4827, 32512, !dbg !12
  %k0.prime.narrow807 = add nuw nsw i32 %k0.prime.4826, %k0.prime.4828, !dbg !12
  %reass.sub = sub nsw i32 %.neg, %k0.prime.4801, !dbg !12
  %k0.prime.narrow806.neg = shl nsw i32 %reass.sub, 11, !dbg !12
  %k0.prime.4830.neg7 = add nsw i32 %k0.prime.narrow806.neg, 24576, !dbg !12
  %k0.prime.4831 = add nuw nsw i32 %k0.prime.narrow807, %k0.prime.4830.neg7, !dbg !12
  %k0.prime.4832 = and i32 %k0.prime.4831, 32512, !dbg !12
  %k0.prime..idx4649 = zext nneg i32 %k0.prime.4832 to i64, !dbg !12
  %k0.prime.4833 = add nsw i64 %k0.prime.4823, %k0.prime..idx4649, !dbg !12
  %k0.prime.4834 = add nsw i64 %k0.prime.4833, %k0.prime..idx4647, !dbg !12
  %k0.prime.dma.buffer.offset4619 = trunc nsw i64 %k0.prime.4834 to i32, !dbg !12
  %k0.prime.address.local1148 = getelementptr i8, ptr addrspace(3) %k0.prime.address.local1136, i64 4096, !dbg !12
  %k0.prime.4841 = add nuw nsw i64 %k0.prime.4793, 2080, !dbg !12
  %k0.prime.4842 = and i64 %k0.prime.4841, 8184, !dbg !12
  %k0.prime.4838 = add nuw nsw i64 %k0.prime.4791, 2080, !dbg !12
  %k0.prime.4843 = sub nsw i64 %k0.prime.4838, %k0.prime.4842, !dbg !12
  %k0.prime..idx4651 = shl nsw i64 %k0.prime.4843, 1, !dbg !12
  %k0.prime.4844 = add nsw i64 %k0.prime..idx4651, 32, !dbg !12
  %k0.prime.4845 = or disjoint i32 %k0.prime.4797, 8192, !dbg !12
  %k0.prime.narrow810 = add nuw nsw i32 %k0.prime.4845, %k0.prime.4799, !dbg !12
  %k0.prime.4846 = zext nneg i32 %k0.prime.narrow810 to i64, !dbg !12
  %k0.prime.4847 = or disjoint i32 %k0.prime.4801, 8, !dbg !12
  %k0.prime.narrow809 = add nuw nsw i32 %k0.prime.4847, %49, !dbg !12
  %k0.prime.4848 = shl nuw nsw i32 %k0.prime.narrow809, 10, !dbg !12
  %k0.prime.4849 = zext nneg i32 %k0.prime.4848 to i64, !dbg !12
  %k0.prime.4850 = sub nsw i64 %k0.prime.4846, %k0.prime.4849, !dbg !12
  %k0.prime.4851 = shl nsw i64 %k0.prime.4850, 1, !dbg !12
  %k0.prime..idx4652 = and i64 %k0.prime.4851, 32512, !dbg !12
  %k0.prime.4852 = add nsw i64 %k0.prime.4844, %k0.prime..idx4652, !dbg !12
  %k0.prime.4853 = add nsw i64 %k0.prime.4852, %k0.prime..idx4647, !dbg !12
  %k0.prime.dma.buffer.offset4617 = trunc nsw i64 %k0.prime.4853 to i32, !dbg !12
  %k0.prime.address.local1154 = getelementptr i8, ptr addrspace(3) %k0.prime.address.local1136, i64 6144, !dbg !12
  %50 = insertelement <8 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local876, i64 0, !dbg !9
  %51 = insertelement <8 x ptr addrspace(3)> %50, ptr addrspace(3) %address.local888, i64 1, !dbg !9
  %52 = insertelement <8 x ptr addrspace(3)> %51, ptr addrspace(3) %address.local898, i64 2, !dbg !9
  %53 = insertelement <8 x ptr addrspace(3)> %52, ptr addrspace(3) %address.local910, i64 3, !dbg !9
  %54 = insertelement <8 x ptr addrspace(3)> %53, ptr addrspace(3) %k0.prime.address.local1136, i64 4, !dbg !9
  %55 = insertelement <8 x ptr addrspace(3)> %54, ptr addrspace(3) %k0.prime.address.local1142, i64 5, !dbg !9
  %56 = insertelement <8 x ptr addrspace(3)> %55, ptr addrspace(3) %k0.prime.address.local1148, i64 6, !dbg !9
  %57 = insertelement <8 x ptr addrspace(3)> %56, ptr addrspace(3) %k0.prime.address.local1154, i64 7, !dbg !9
  %58 = ptrtoint <8 x ptr addrspace(3)> %57 to <8 x i64>, !dbg !9
  %59 = trunc <8 x i64> %58 to <8 x i32>, !dbg !9
  %60 = extractelement <8 x i32> %59, i64 0, !dbg !9
  %61 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !9
  %62 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %61, i32 %dma.buffer.offset4589, <4 x i32> %dma.buffer.rsrc4590) #9, !dbg !9
  %63 = extractelement <8 x i32> %59, i64 1, !dbg !9
  %64 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %63), !dbg !9
  %65 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %64, i32 %dma.buffer.offset4587, <4 x i32> %dma.buffer.rsrc4590) #9, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %66 = extractelement <8 x i32> %59, i64 2, !dbg !9
  %67 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %66), !dbg !9
  %68 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %67, i32 %dma.buffer.offset4585, <4 x i32> %dma.buffer.rsrc4590) #9, !dbg !9
  %69 = extractelement <8 x i32> %59, i64 3, !dbg !9
  %70 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %69), !dbg !9
  %71 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %70, i32 %dma.buffer.offset4583, <4 x i32> %dma.buffer.rsrc4590) #9, !dbg !9
  %72 = load <4 x half>, ptr addrspace(3) %address.local929, align 8, !dbg !11
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %74 = fmul <4 x half> %72, splat (half 1.275630e-01), !dbg !11
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep4423, align 8, !dbg !11
  %76 = fmul <4 x half> %73, splat (half 1.275630e-01), !dbg !11
  %77 = load <4 x half>, ptr addrspace(3) %lds.const.gep4424, align 8, !dbg !11
  %78 = fmul <4 x half> %75, splat (half 1.275630e-01), !dbg !11
  %79 = fmul <4 x half> %77, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %80 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %61, i32 %dma.buffer.offset4581, <4 x i32> %dma.buffer.rsrc4590) #9, !dbg !9
  %81 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %64, i32 %dma.buffer.offset4579, <4 x i32> %dma.buffer.rsrc4590) #9, !dbg !9
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep4425, align 8, !dbg !11
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep4426, align 8, !dbg !11
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep4427, align 8, !dbg !11
  %85 = fmul <4 x half> %82, splat (half 1.275630e-01), !dbg !11
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep4428, align 8, !dbg !11
  %87 = fmul <4 x half> %83, splat (half 1.275630e-01), !dbg !11
  %88 = fmul <4 x half> %84, splat (half 1.275630e-01), !dbg !11
  %89 = fmul <4 x half> %86, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %90 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %67, i32 %dma.buffer.offset4577, <4 x i32> %dma.buffer.rsrc4590) #9, !dbg !9
  %91 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %70, i32 %dma.buffer.offset4575, <4 x i32> %dma.buffer.rsrc4590) #9, !dbg !9
  %92 = load <4 x half>, ptr addrspace(3) %address.local929, align 8, !dbg !11
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %94 = load <4 x half>, ptr addrspace(3) %lds.const.gep4423, align 8, !dbg !11
  %95 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !11
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep4424, align 8, !dbg !11
  %97 = fmul <4 x half> %93, splat (half 1.275630e-01), !dbg !11
  %98 = fmul <4 x half> %94, splat (half 1.275630e-01), !dbg !11
  %99 = fmul <4 x half> %96, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep4425, align 8, !dbg !11
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep4426, align 8, !dbg !11
  %102 = load <4 x half>, ptr addrspace(3) %lds.const.gep4427, align 8, !dbg !11
  %103 = fmul <4 x half> %100, splat (half 1.275630e-01), !dbg !11
  %104 = load <4 x half>, ptr addrspace(3) %lds.const.gep4428, align 8, !dbg !11
  %105 = fmul <4 x half> %101, splat (half 1.275630e-01), !dbg !11
  %106 = fmul <4 x half> %102, splat (half 1.275630e-01), !dbg !11
  %107 = fmul <4 x half> %104, splat (half 1.275630e-01), !dbg !11
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %108 = extractelement <8 x i32> %59, i64 4, !dbg !12
  %k0.prime. = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %108), !dbg !12
  %k0.prime.4814 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime., i32 %k0.prime.dma.buffer.offset4621, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %109 = extractelement <8 x i32> %59, i64 5, !dbg !12
  %k0.prime.4815 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %109), !dbg !12
  %k0.prime.4835 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.4815, i32 %k0.prime.dma.buffer.offset4619, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %110 = extractelement <8 x i32> %59, i64 6, !dbg !12
  %k0.prime.4836 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %110), !dbg !12
  %k0.prime.4854 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.4836, i32 %k0.prime.dma.buffer.offset4617, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %111 = extractelement <8 x i32> %59, i64 7, !dbg !12
  %k0.prime.4855 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %111), !dbg !12
  %k0.prime.4860 = add nuw nsw i64 %k0.prime.4793, 3120, !dbg !12
  %k0.prime.4861 = and i64 %k0.prime.4860, 16376, !dbg !12
  %k0.prime.4857 = add nuw nsw i64 %k0.prime.4791, 3120, !dbg !12
  %k0.prime.4862 = sub nsw i64 %k0.prime.4857, %k0.prime.4861, !dbg !12
  %k0.prime..idx46534656 = add nsw i64 %k0.prime.4787, %k0.prime.4862, !dbg !12
  %k0.prime..idx46534656.tr = trunc nsw i64 %k0.prime..idx46534656 to i32, !dbg !12
  %k0.prime.4863 = shl nsw i32 %k0.prime..idx46534656.tr, 1, !dbg !12
  %k0.prime.4865 = or disjoint i32 %k0.prime.4825, 24576, !dbg !12
  %k0.prime.narrow812 = add nuw nsw i32 %k0.prime.4865, %k0.prime.4828, !dbg !12
  %k0.prime.4867.neg9 = add nsw i32 %k0.prime.narrow806.neg, 8192, !dbg !12
  %k0.prime.4868 = add nuw nsw i32 %k0.prime.narrow812, %k0.prime.4867.neg9, !dbg !12
  %k0.prime.4869 = and i32 %k0.prime.4868, 32512, !dbg !12
  %k0.prime.4864 = or disjoint i32 %k0.prime.4869, 48, !dbg !12
  %k0.prime.dma.buffer.offset4615 = add nsw i32 %k0.prime.4864, %k0.prime.4863, !dbg !12
  %k0.prime.4870 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.4855, i32 %k0.prime.dma.buffer.offset4615, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %112 = ptrtoint ptr addrspace(1) %2 to i64
  %113 = insertelement <2 x i64> poison, i64 %112, i64 0
  %114 = bitcast <2 x i64> %113 to <4 x i32>
  %dma.buffer.rsrc4574 = shufflevector <4 x i32> %114, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  br label %115, !dbg !13

115:                                              ; preds = %4, %k0.join
  %116 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1043, %k0.join ]
  %117 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1039, %k0.join ]
  %118 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1035, %k0.join ]
  %119 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %lazy.base.join, %k0.join ]
  %120 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %808, %k0.join ]
  %121 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1031, %k0.join ]
  %122 = phi i64 [ 0, %4 ], [ %963, %k0.join ]
  %123 = shl nuw nsw i64 %122, 14, !dbg !12
  %124 = add nuw nsw i64 %123, %k0.prime.4787, !dbg !12
  %address.thread1132 = tail call i32 asm sideeffect "", "=v,0"(i32 %34) #8, !dbg !12, !range !10, !frisk.register_lifetimes !5
  %address.local1133 = zext nneg i32 %address.thread1132 to i64, !dbg !9
  %address.local1134 = shl nuw nsw i64 %address.local1133, 2, !dbg !12
  %address.local1135 = and i64 %address.local1134, 768, !dbg !12
  %address.local1136 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1135, !dbg !12
  %address.local1142 = getelementptr i8, ptr addrspace(3) %address.local1136, i64 2048, !dbg !12
  %address.local1148 = getelementptr i8, ptr addrspace(3) %address.local1136, i64 4096, !dbg !12
  %address.local1154 = getelementptr i8, ptr addrspace(3) %address.local1136, i64 6144, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %125 = or disjoint i64 %124, 32, !dbg !12
  %address.local1161 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1135, !dbg !12
  %.idx46574660 = add nsw i64 %125, %k0.prime.4810, !dbg !12
  %.idx46574660.tr = trunc nsw i64 %.idx46574660 to i32, !dbg !12
  %dma.buffer.offset4613 = shl nsw i32 %.idx46574660.tr, 1, !dbg !12
  %address.local1167 = getelementptr i8, ptr addrspace(3) %address.local1161, i64 2048, !dbg !12
  %.idx4661 = shl nuw nsw i64 %125, 1, !dbg !12
  %126 = add nsw i64 %.idx4661, %k0.prime.4833, !dbg !12
  %dma.buffer.offset4611 = trunc nsw i64 %126 to i32, !dbg !12
  %address.local1173 = getelementptr i8, ptr addrspace(3) %address.local1161, i64 4096, !dbg !12
  %127 = add nsw i64 %.idx4661, %k0.prime.4852, !dbg !12
  %dma.buffer.offset4609 = trunc nsw i64 %127 to i32, !dbg !12
  %address.local1179 = getelementptr i8, ptr addrspace(3) %address.local1161, i64 6144, !dbg !12
  %128 = insertelement <8 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local1136, i64 0, !dbg !12
  %129 = insertelement <8 x ptr addrspace(3)> %128, ptr addrspace(3) %address.local1142, i64 1, !dbg !12
  %130 = insertelement <8 x ptr addrspace(3)> %129, ptr addrspace(3) %address.local1148, i64 2, !dbg !12
  %131 = insertelement <8 x ptr addrspace(3)> %130, ptr addrspace(3) %address.local1154, i64 3, !dbg !12
  %132 = insertelement <8 x ptr addrspace(3)> %131, ptr addrspace(3) %address.local1161, i64 4, !dbg !12
  %133 = insertelement <8 x ptr addrspace(3)> %132, ptr addrspace(3) %address.local1167, i64 5, !dbg !12
  %134 = insertelement <8 x ptr addrspace(3)> %133, ptr addrspace(3) %address.local1173, i64 6, !dbg !12
  %135 = insertelement <8 x ptr addrspace(3)> %134, ptr addrspace(3) %address.local1179, i64 7, !dbg !12
  %136 = ptrtoint <8 x ptr addrspace(3)> %135 to <8 x i64>, !dbg !12
  %137 = trunc <8 x i64> %136 to <8 x i32>, !dbg !12
  %138 = extractelement <8 x i32> %137, i64 0, !dbg !12
  %139 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %138), !dbg !12
  %140 = extractelement <8 x i32> %137, i64 1, !dbg !12
  %141 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %140), !dbg !12
  %142 = extractelement <8 x i32> %137, i64 2, !dbg !12
  %143 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %142), !dbg !12
  %144 = extractelement <8 x i32> %137, i64 3, !dbg !12
  %145 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %144), !dbg !12
  %146 = extractelement <8 x i32> %137, i64 4, !dbg !12
  %147 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %146), !dbg !12
  %148 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %147, i32 %dma.buffer.offset4613, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12
  %149 = extractelement <8 x i32> %137, i64 5, !dbg !12
  %150 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %149), !dbg !12
  %151 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %150, i32 %dma.buffer.offset4611, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12
  %152 = extractelement <8 x i32> %137, i64 6, !dbg !12
  %153 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %152), !dbg !12
  %154 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %153, i32 %dma.buffer.offset4609, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12
  %155 = extractelement <8 x i32> %137, i64 7, !dbg !12
  %156 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %155), !dbg !12
  %.idx46674670 = add nsw i64 %125, %k0.prime.4862, !dbg !12
  %.idx46674670.tr = trunc nsw i64 %.idx46674670 to i32, !dbg !12
  %157 = shl nsw i32 %.idx46674670.tr, 1, !dbg !12
  %dma.buffer.offset4607 = add i32 %k0.prime.4864, %157, !dbg !12
  %158 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %156, i32 %dma.buffer.offset4607, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12
  %address.local1184 = mul nuw nsw i64 %address.local1133, 160, !dbg !14
  %address.local1185 = lshr i64 %address.local1133, 2, !dbg !14
  %address.local1186 = mul nsw i64 %address.local1185, -640, !dbg !14
  %address.local1188 = shl nuw nsw i64 %address.local1133, 1, !dbg !14
  %address.local1189 = and i64 %address.local1188, 24, !dbg !14
  %address.local1191 = shl nuw nsw i64 %address.local1133, 5, !dbg !14
  %address.local1192 = and i64 %address.local1191, 64, !dbg !14
  %address.local1194 = and i64 %address.local1133, 63, !dbg !11
  %address.local1195 = lshr i64 %address.local1194, 4, !dbg !11
  %address.local1196 = shl nuw nsw i64 %address.local1195, 2, !dbg !11
  %address.local1198 = lshr i64 %address.local1194, 5, !dbg !11
  %address.local1199 = mul nuw nsw i64 %address.local1198, 1016, !dbg !11
  %address.local1201 = lshr i32 %address.thread1132, 1, !dbg !14
  %address.local1202 = and i32 %address.local1201, 16, !dbg !14
  %address.local1203 = shl nuw nsw i32 %address.thread1132, 1, !dbg !14
  %address.local1204 = and i32 %address.local1203, 4, !dbg !14
  %address.local1205 = or disjoint i32 %address.local1202, %address.local1204, !dbg !14
  %address.local1206 = zext nneg i32 %address.local1205 to i64, !dbg !14
  %address.local1209 = and i32 %address.local1201, 1, !dbg !14
  %address.local1210 = zext nneg i32 %address.local1209 to i64, !dbg !14
  %address.local1211 = and i64 %address.local1195, %address.local1210, !dbg !14
  %address.local1212 = lshr i32 %address.thread1132, 4, !dbg !14
  %address.local1213 = and i32 %address.local1212, 2, !dbg !14
  %address.local1214 = zext nneg i32 %address.local1213 to i64, !dbg !14
  %address.local1215 = or disjoint i64 %address.local1211, %address.local1214, !dbg !14
  %address.local1216.neg11 = mul nsw i64 %address.local1215, -8, !dbg !14
  %address.local1190 = add nuw nsw i64 %address.local1199, %address.local1196, !dbg !14
  %address.local1193 = add nuw nsw i64 %address.local1190, %address.local1184, !dbg !14
  %address.local1197 = add nsw i64 %address.local1193, %address.local1186, !dbg !14
  %address.local1187 = add nsw i64 %address.local1197, %address.local1189, !dbg !14
  %address.local1200 = sub nsw i64 %address.local1187, %address.local1192, !dbg !14
  %159 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1200, !dbg !14
  %160 = getelementptr [2 x i8], ptr addrspace(3) %159, i64 %address.local1206, !dbg !14
  %address.local1218 = getelementptr [2 x i8], ptr addrspace(3) %160, i64 %address.local1216.neg11, !dbg !14
  %161 = load <4 x half>, ptr addrspace(3) %address.local1218, align 8, !dbg !14, !frisk.k_lds_pair !15
  %lds.const.gep4439 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 128, !dbg !14
  %162 = load <4 x half>, ptr addrspace(3) %lds.const.gep4439, align 8, !dbg !14, !frisk.k_lds_pair !15
  %lds.const.gep4437 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 4096, !dbg !14
  %163 = load <4 x half>, ptr addrspace(3) %lds.const.gep4437, align 8, !dbg !14, !frisk.k_lds_pair !16
  %lds.const.gep4440 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 4224, !dbg !14
  %164 = load <4 x half>, ptr addrspace(3) %lds.const.gep4440, align 8, !dbg !14, !frisk.k_lds_pair !16
  %address.local1247 = and i32 %address.local1203, 24, !dbg !14
  %address.local1248 = shl nuw nsw i32 %address.thread1132, 8, !dbg !14
  %address.local1249 = and i32 %address.local1248, 768, !dbg !14
  %address.local1251 = shl nuw nsw i32 %address.thread1132, 5, !dbg !14
  %address.local1252 = and i32 %address.local1251, 96, !dbg !14
  %165 = or disjoint i32 %address.local1249, %address.local1252, !dbg !14
  %address.local1253 = or disjoint i32 %165, %address.local1247, !dbg !14
  %address.local1254 = zext nneg i32 %address.local1253 to i64, !dbg !14
  %166 = shl nuw nsw i64 %address.local1185, 4, !dbg !14
  %address.local1257 = sub nsw i64 %address.local1134, %166, !dbg !14
  %address.local1258 = and i32 %address.thread1132, 1, !dbg !14
  %address.local1259 = zext nneg i32 %address.local1258 to i64, !dbg !14
  %address.local1260 = or disjoint i64 %address.local1257, %address.local1259, !dbg !14
  %167 = trunc nsw i64 %address.local1260 to i16, !dbg !14
  %address.local1262.lhs.trunc = add nuw nsw i16 %167, 1, !dbg !14
  %address.local12622 = sdiv i16 %address.local1262.lhs.trunc, 2, !dbg !14
  %address.local1262.sext = sext i16 %address.local12622 to i64, !dbg !14
  %address.local1266 = add nsw i64 %address.local1262.sext, %address.local1210, !dbg !14
  %address.local1269 = add nuw nsw i64 %address.local1190, %address.local1206, !dbg !14
  %address.local1270 = add nuw nsw i64 %address.local1269, %address.local1254, !dbg !14
  %address.local1277 = add nsw i64 %address.local1270, %address.local1216.neg11, !dbg !14
  %168 = shl nsw i64 %address.local1266, 6, !dbg !14
  %address.local1287 = sub nsw i64 %address.local1277, %168, !dbg !14
  %169 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1287, !dbg !14
  %address.local1289 = getelementptr i8, ptr addrspace(3) %169, i64 64, !dbg !14
  %170 = load <4 x half>, ptr addrspace(3) %address.local1289, align 8, !dbg !14, !frisk.k_lds_pair !17
  %lds.const.gep4441 = getelementptr i8, ptr addrspace(3) %169, i64 192, !dbg !14
  %171 = load <4 x half>, ptr addrspace(3) %lds.const.gep4441, align 8, !dbg !14, !frisk.k_lds_pair !17
  %lds.const.gep4438 = getelementptr i8, ptr addrspace(3) %169, i64 4160, !dbg !14
  %172 = load <4 x half>, ptr addrspace(3) %lds.const.gep4438, align 8, !dbg !14, !frisk.k_lds_pair !18
  %lds.const.gep4442 = getelementptr i8, ptr addrspace(3) %169, i64 4288, !dbg !14
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep4442, align 8, !dbg !14, !frisk.k_lds_pair !18
  %lds.const.gep4443 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 1024, !dbg !14
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep4443, align 8, !dbg !14, !frisk.k_lds_pair !19
  %lds.const.gep4447 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 1152, !dbg !14
  %175 = load <4 x half>, ptr addrspace(3) %lds.const.gep4447, align 8, !dbg !14, !frisk.k_lds_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %lazy.base = extractvalue [2 x <1 x float>] %119, 0, !dbg !14
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !14
  %lazy.base4880 = extractvalue [2 x <1 x float>] %119, 1, !dbg !14
  %lazy.seed4881 = shufflevector <1 x float> %lazy.base4880, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !14
  %176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %161, <4 x float> %lazy.seed) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %lds.const.gep4444 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 5120, !dbg !14
  %177 = load <4 x half>, ptr addrspace(3) %lds.const.gep4444, align 8, !dbg !14, !frisk.k_lds_pair !20
  %lds.const.gep4448 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 5248, !dbg !14
  %178 = load <4 x half>, ptr addrspace(3) %lds.const.gep4448, align 8, !dbg !14, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %163, <4 x float> %176) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %lds.const.gep4445 = getelementptr i8, ptr addrspace(3) %169, i64 1088, !dbg !14
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep4445, align 8, !dbg !14, !frisk.k_lds_pair !21
  %lds.const.gep4449 = getelementptr i8, ptr addrspace(3) %169, i64 1216, !dbg !14
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep4449, align 8, !dbg !14, !frisk.k_lds_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %170, <4 x float> %lazy.seed) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %lds.const.gep4446 = getelementptr i8, ptr addrspace(3) %169, i64 5184, !dbg !14
  %183 = load <4 x half>, ptr addrspace(3) %lds.const.gep4446, align 8, !dbg !14, !frisk.k_lds_pair !22
  %lds.const.gep4450 = getelementptr i8, ptr addrspace(3) %169, i64 5312, !dbg !14
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep4450, align 8, !dbg !14, !frisk.k_lds_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %172, <4 x float> %182) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %162, <4 x float> %lazy.seed) #8, !dbg !14
  %187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %164, <4 x float> %186) #8, !dbg !14
  %188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %171, <4 x float> %lazy.seed) #8, !dbg !14
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %173, <4 x float> %188) #8, !dbg !14
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %174, <4 x float> %lazy.seed) #8, !dbg !14
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %177, <4 x float> %190) #8, !dbg !14
  %192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %180, <4 x float> %lazy.seed) #8, !dbg !14
  %193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %183, <4 x float> %192) #8, !dbg !14
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %175, <4 x float> %lazy.seed) #8, !dbg !14
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %178, <4 x float> %194) #8, !dbg !14
  %196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %181, <4 x float> %lazy.seed) #8, !dbg !14
  %197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %184, <4 x float> %196) #8, !dbg !14
  %198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %161, <4 x float> %lazy.seed4881) #8, !dbg !14
  %199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %163, <4 x float> %198) #8, !dbg !14
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %170, <4 x float> %lazy.seed4881) #8, !dbg !14
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %172, <4 x float> %200) #8, !dbg !14
  %202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %162, <4 x float> %lazy.seed4881) #8, !dbg !14
  %203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %164, <4 x float> %202) #8, !dbg !14
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %171, <4 x float> %lazy.seed4881) #8, !dbg !14
  %205 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %173, <4 x float> %204) #8, !dbg !14
  %206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %174, <4 x float> %lazy.seed4881) #8, !dbg !14
  %207 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %177, <4 x float> %206) #8, !dbg !14
  %208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %180, <4 x float> %lazy.seed4881) #8, !dbg !14
  %209 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %183, <4 x float> %208) #8, !dbg !14
  %210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %175, <4 x float> %lazy.seed4881) #8, !dbg !14
  %211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %178, <4 x float> %210) #8, !dbg !14
  %212 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %181, <4 x float> %lazy.seed4881) #8, !dbg !14
  %213 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %184, <4 x float> %212) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %214 = or disjoint i64 %124, 64, !dbg !12
  %.idx46714674 = add nsw i64 %214, %k0.prime.4810, !dbg !12
  %.idx46714674.tr = trunc nsw i64 %.idx46714674 to i32, !dbg !12
  %dma.buffer.offset4605 = shl nsw i32 %.idx46714674.tr, 1, !dbg !12
  %215 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %139, i32 %dma.buffer.offset4605, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12
  %.idx4675 = shl nuw nsw i64 %214, 1, !dbg !12
  %216 = add nsw i64 %.idx4675, %k0.prime.4833, !dbg !12
  %dma.buffer.offset4603 = trunc nsw i64 %216 to i32, !dbg !12
  %217 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %141, i32 %dma.buffer.offset4603, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12
  %218 = add nsw i64 %.idx4675, %k0.prime.4852, !dbg !12
  %dma.buffer.offset4601 = trunc nsw i64 %218 to i32, !dbg !12
  %219 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %143, i32 %dma.buffer.offset4601, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12
  %.idx46814684 = add nsw i64 %214, %k0.prime.4862, !dbg !12
  %.idx46814684.tr = trunc nsw i64 %.idx46814684 to i32, !dbg !12
  %220 = shl nsw i32 %.idx46814684.tr, 1, !dbg !12
  %dma.buffer.offset4599 = add i32 %k0.prime.4864, %220, !dbg !12
  %221 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %145, i32 %dma.buffer.offset4599, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12
  %lds.const.gep4465 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 8192, !dbg !14
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep4465, align 8, !dbg !14, !frisk.k_lds_pair !23
  %lds.const.gep4469 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 8320, !dbg !14
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep4469, align 8, !dbg !14, !frisk.k_lds_pair !23
  %lds.const.gep4466 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 12288, !dbg !14
  %224 = load <4 x half>, ptr addrspace(3) %lds.const.gep4466, align 8, !dbg !14, !frisk.k_lds_pair !24
  %lds.const.gep4470 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 12416, !dbg !14
  %225 = load <4 x half>, ptr addrspace(3) %lds.const.gep4470, align 8, !dbg !14, !frisk.k_lds_pair !24
  %lds.const.gep4467 = getelementptr i8, ptr addrspace(3) %169, i64 8256, !dbg !14
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep4467, align 8, !dbg !14, !frisk.k_lds_pair !25
  %lds.const.gep4471 = getelementptr i8, ptr addrspace(3) %169, i64 8384, !dbg !14
  %227 = load <4 x half>, ptr addrspace(3) %lds.const.gep4471, align 8, !dbg !14, !frisk.k_lds_pair !25
  %lds.const.gep4468 = getelementptr i8, ptr addrspace(3) %169, i64 12352, !dbg !14
  %228 = load <4 x half>, ptr addrspace(3) %lds.const.gep4468, align 8, !dbg !14, !frisk.k_lds_pair !26
  %lds.const.gep4472 = getelementptr i8, ptr addrspace(3) %169, i64 12480, !dbg !14
  %229 = load <4 x half>, ptr addrspace(3) %lds.const.gep4472, align 8, !dbg !14, !frisk.k_lds_pair !26
  %lds.const.gep4473 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 9216, !dbg !14
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep4473, align 8, !dbg !14, !frisk.k_lds_pair !27
  %lds.const.gep4477 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 9344, !dbg !14
  %231 = load <4 x half>, ptr addrspace(3) %lds.const.gep4477, align 8, !dbg !14, !frisk.k_lds_pair !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %222, <4 x float> %179) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %lds.const.gep4474 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 13312, !dbg !14
  %233 = load <4 x half>, ptr addrspace(3) %lds.const.gep4474, align 8, !dbg !14, !frisk.k_lds_pair !28
  %lds.const.gep4478 = getelementptr i8, ptr addrspace(3) %address.local1218, i64 13440, !dbg !14
  %234 = load <4 x half>, ptr addrspace(3) %lds.const.gep4478, align 8, !dbg !14, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %224, <4 x float> %232) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %lds.const.gep4475 = getelementptr i8, ptr addrspace(3) %169, i64 9280, !dbg !14
  %236 = load <4 x half>, ptr addrspace(3) %lds.const.gep4475, align 8, !dbg !14, !frisk.k_lds_pair !29
  %lds.const.gep4479 = getelementptr i8, ptr addrspace(3) %169, i64 9408, !dbg !14
  %237 = load <4 x half>, ptr addrspace(3) %lds.const.gep4479, align 8, !dbg !14, !frisk.k_lds_pair !29
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %226, <4 x float> %185) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %lds.const.gep4476 = getelementptr i8, ptr addrspace(3) %169, i64 13376, !dbg !14
  %239 = load <4 x half>, ptr addrspace(3) %lds.const.gep4476, align 8, !dbg !14, !frisk.k_lds_pair !30
  %lds.const.gep4480 = getelementptr i8, ptr addrspace(3) %169, i64 13504, !dbg !14
  %240 = load <4 x half>, ptr addrspace(3) %lds.const.gep4480, align 8, !dbg !14, !frisk.k_lds_pair !30
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %228, <4 x float> %238) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %223, <4 x float> %187) #8, !dbg !14
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %225, <4 x float> %242) #8, !dbg !14
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %227, <4 x float> %189) #8, !dbg !14
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %229, <4 x float> %244) #8, !dbg !14
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %230, <4 x float> %191) #8, !dbg !14
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %233, <4 x float> %246) #8, !dbg !14
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %236, <4 x float> %193) #8, !dbg !14
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %239, <4 x float> %248) #8, !dbg !14
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %231, <4 x float> %195) #8, !dbg !14
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %234, <4 x float> %250) #8, !dbg !14
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %237, <4 x float> %197) #8, !dbg !14
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %240, <4 x float> %252) #8, !dbg !14
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %222, <4 x float> %199) #8, !dbg !14
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %224, <4 x float> %254) #8, !dbg !14
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %226, <4 x float> %201) #8, !dbg !14
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %228, <4 x float> %256) #8, !dbg !14
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %223, <4 x float> %203) #8, !dbg !14
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %225, <4 x float> %258) #8, !dbg !14
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %227, <4 x float> %205) #8, !dbg !14
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %229, <4 x float> %260) #8, !dbg !14
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %230, <4 x float> %207) #8, !dbg !14
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %233, <4 x float> %262) #8, !dbg !14
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %236, <4 x float> %209) #8, !dbg !14
  %265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %239, <4 x float> %264) #8, !dbg !14
  %266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %231, <4 x float> %211) #8, !dbg !14
  %267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %234, <4 x float> %266) #8, !dbg !14
  %268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %237, <4 x float> %213) #8, !dbg !14
  %269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %240, <4 x float> %268) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %270 = or disjoint i64 %124, 96, !dbg !12
  %.idx46854688 = add nsw i64 %270, %k0.prime.4810, !dbg !12
  %.idx46854688.tr = trunc nsw i64 %.idx46854688 to i32, !dbg !12
  %dma.buffer.offset4597 = shl nsw i32 %.idx46854688.tr, 1, !dbg !12
  %271 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %147, i32 %dma.buffer.offset4597, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12
  %.idx4689 = shl nuw nsw i64 %270, 1, !dbg !12
  %272 = add nsw i64 %.idx4689, %k0.prime.4833, !dbg !12
  %dma.buffer.offset4595 = trunc nsw i64 %272 to i32, !dbg !12
  %273 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %150, i32 %dma.buffer.offset4595, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12
  %274 = add nsw i64 %.idx4689, %k0.prime.4852, !dbg !12
  %dma.buffer.offset4593 = trunc nsw i64 %274 to i32, !dbg !12
  %275 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %153, i32 %dma.buffer.offset4593, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12
  %.idx46954698 = add nsw i64 %270, %k0.prime.4862, !dbg !12
  %.idx46954698.tr = trunc nsw i64 %.idx46954698 to i32, !dbg !12
  %276 = shl nsw i32 %.idx46954698.tr, 1, !dbg !12
  %dma.buffer.offset4591 = add i32 %k0.prime.4864, %276, !dbg !12
  %277 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %156, i32 %dma.buffer.offset4591, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12
  %278 = load <4 x half>, ptr addrspace(3) %address.local1218, align 8, !dbg !14, !frisk.k_lds_pair !31
  %279 = load <4 x half>, ptr addrspace(3) %lds.const.gep4439, align 8, !dbg !14, !frisk.k_lds_pair !31
  %280 = load <4 x half>, ptr addrspace(3) %lds.const.gep4437, align 8, !dbg !14, !frisk.k_lds_pair !32
  %281 = load <4 x half>, ptr addrspace(3) %lds.const.gep4440, align 8, !dbg !14, !frisk.k_lds_pair !32
  %282 = load <4 x half>, ptr addrspace(3) %address.local1289, align 8, !dbg !14, !frisk.k_lds_pair !33
  %283 = load <4 x half>, ptr addrspace(3) %lds.const.gep4441, align 8, !dbg !14, !frisk.k_lds_pair !33
  %284 = load <4 x half>, ptr addrspace(3) %lds.const.gep4438, align 8, !dbg !14, !frisk.k_lds_pair !34
  %285 = load <4 x half>, ptr addrspace(3) %lds.const.gep4442, align 8, !dbg !14, !frisk.k_lds_pair !34
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep4443, align 8, !dbg !14, !frisk.k_lds_pair !35
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep4447, align 8, !dbg !14, !frisk.k_lds_pair !35
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %278, <4 x float> %235) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %289 = load <4 x half>, ptr addrspace(3) %lds.const.gep4444, align 8, !dbg !14, !frisk.k_lds_pair !36
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep4448, align 8, !dbg !14, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %280, <4 x float> %288) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %292 = load <4 x half>, ptr addrspace(3) %lds.const.gep4445, align 8, !dbg !14, !frisk.k_lds_pair !37
  %293 = load <4 x half>, ptr addrspace(3) %lds.const.gep4449, align 8, !dbg !14, !frisk.k_lds_pair !37
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %282, <4 x float> %241) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep4446, align 8, !dbg !14, !frisk.k_lds_pair !38
  %296 = load <4 x half>, ptr addrspace(3) %lds.const.gep4450, align 8, !dbg !14, !frisk.k_lds_pair !38
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %284, <4 x float> %294) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %279, <4 x float> %243) #8, !dbg !14
  %299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %281, <4 x float> %298) #8, !dbg !14
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %283, <4 x float> %245) #8, !dbg !14
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %285, <4 x float> %300) #8, !dbg !14
  %302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %286, <4 x float> %247) #8, !dbg !14
  %303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %289, <4 x float> %302) #8, !dbg !14
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %292, <4 x float> %249) #8, !dbg !14
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %295, <4 x float> %304) #8, !dbg !14
  %306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %287, <4 x float> %251) #8, !dbg !14
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %290, <4 x float> %306) #8, !dbg !14
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %293, <4 x float> %253) #8, !dbg !14
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %296, <4 x float> %308) #8, !dbg !14
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %278, <4 x float> %255) #8, !dbg !14
  %311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %280, <4 x float> %310) #8, !dbg !14
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %282, <4 x float> %257) #8, !dbg !14
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %284, <4 x float> %312) #8, !dbg !14
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %279, <4 x float> %259) #8, !dbg !14
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %281, <4 x float> %314) #8, !dbg !14
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %283, <4 x float> %261) #8, !dbg !14
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %285, <4 x float> %316) #8, !dbg !14
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %286, <4 x float> %263) #8, !dbg !14
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %289, <4 x float> %318) #8, !dbg !14
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %292, <4 x float> %265) #8, !dbg !14
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %295, <4 x float> %320) #8, !dbg !14
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %287, <4 x float> %267) #8, !dbg !14
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %290, <4 x float> %322) #8, !dbg !14
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %293, <4 x float> %269) #8, !dbg !14
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %296, <4 x float> %324) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local3187 = shl nuw nsw i64 %address.local1133, 3, !dbg !39
  %address.local3188 = and i64 %address.local3187, 1536, !dbg !39
  %address.local3189 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local3188, !dbg !39
  %address.local3190 = ptrtoint ptr addrspace(3) %address.local3189 to i64, !dbg !39
  %address.local3191 = trunc i64 %address.local3190 to i32, !dbg !39
  %326 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3191), !dbg !39
  %address.local3192 = shl nuw nsw i64 %address.local1133, 7, !dbg !39
  %address.local3193 = and i64 %address.local3192, 16256, !dbg !39
  %address.local3195 = lshr i64 %address.local1133, 4, !dbg !39
  %address.local3196 = and i64 %address.local3195, 8, !dbg !39
  %address.local3194.idx4700 = or disjoint i64 %address.local3193, %address.local3196, !dbg !39
  %.idx46994701 = or disjoint i64 %address.local3194.idx4700, %124, !dbg !39
  %.idx46994701.tr = trunc nuw nsw i64 %.idx46994701 to i32, !dbg !39
  %dma.buffer.offset4573 = shl nuw nsw i32 %.idx46994701.tr, 1, !dbg !39
  %327 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %326, i32 %dma.buffer.offset4573, <4 x i32> %dma.buffer.rsrc4574) #9, !dbg !39
  %address.local3201 = getelementptr i8, ptr addrspace(3) %address.local3189, i64 4096, !dbg !39
  %address.local3202 = ptrtoint ptr addrspace(3) %address.local3201 to i64, !dbg !39
  %address.local3203 = trunc i64 %address.local3202 to i32, !dbg !39
  %328 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3203), !dbg !39
  %dma.buffer.offset4571 = or disjoint i32 %dma.buffer.offset4573, 32, !dbg !39
  %329 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %328, i32 %dma.buffer.offset4571, <4 x i32> %dma.buffer.rsrc4574) #9, !dbg !39
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep4465, align 8, !dbg !14, !frisk.k_tail_pair !15
  %331 = load <4 x half>, ptr addrspace(3) %lds.const.gep4469, align 8, !dbg !14, !frisk.k_tail_pair !15
  %332 = load <4 x half>, ptr addrspace(3) %lds.const.gep4466, align 8, !dbg !14, !frisk.k_tail_pair !16
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep4470, align 8, !dbg !14, !frisk.k_tail_pair !16
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep4467, align 8, !dbg !14, !frisk.k_tail_pair !17
  %335 = load <4 x half>, ptr addrspace(3) %lds.const.gep4471, align 8, !dbg !14, !frisk.k_tail_pair !17
  %336 = load <4 x half>, ptr addrspace(3) %lds.const.gep4468, align 8, !dbg !14, !frisk.k_tail_pair !18
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep4472, align 8, !dbg !14, !frisk.k_tail_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %330, <4 x float> %291) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %339 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %332, <4 x float> %338) #8, !dbg !14
  %340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %334, <4 x float> %297) #8, !dbg !14
  %341 = extractelement <4 x float> %339, i64 0, !dbg !14
  %342 = tail call float @llvm.maxnum.f32(float %341, float -inf), !dbg !7
  %343 = extractelement <4 x float> %339, i64 1, !dbg !14
  %344 = extractelement <4 x float> %339, i64 2, !dbg !14
  %345 = extractelement <4 x float> %339, i64 3, !dbg !14
  %346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %336, <4 x float> %340) #8, !dbg !14
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep4473, align 8, !dbg !14, !frisk.k_tail_pair !19
  %348 = load <4 x half>, ptr addrspace(3) %lds.const.gep4477, align 8, !dbg !14, !frisk.k_tail_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %331, <4 x float> %299) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %350 = extractelement <4 x float> %346, i64 0, !dbg !14
  %351 = tail call float @llvm.maxnum.f32(float %342, float %350), !dbg !7
  %352 = extractelement <4 x float> %346, i64 1, !dbg !14
  %353 = tail call float @llvm.maxnum.f32(float %343, float %352), !dbg !7
  %354 = extractelement <4 x float> %346, i64 2, !dbg !14
  %355 = tail call float @llvm.maxnum.f32(float %344, float %354), !dbg !7
  %356 = extractelement <4 x float> %346, i64 3, !dbg !14
  %357 = tail call float @llvm.maxnum.f32(float %345, float %356), !dbg !7
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep4474, align 8, !dbg !14, !frisk.k_tail_pair !20
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep4478, align 8, !dbg !14, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %333, <4 x float> %349) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep4475, align 8, !dbg !14, !frisk.k_tail_pair !21
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep4479, align 8, !dbg !14, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %335, <4 x float> %301) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %364 = extractelement <4 x float> %360, i64 0, !dbg !14
  %365 = tail call float @llvm.maxnum.f32(float %351, float %364), !dbg !7
  %366 = extractelement <4 x float> %360, i64 1, !dbg !14
  %367 = tail call float @llvm.maxnum.f32(float %353, float %366), !dbg !7
  %368 = extractelement <4 x float> %360, i64 2, !dbg !14
  %369 = tail call float @llvm.maxnum.f32(float %355, float %368), !dbg !7
  %370 = extractelement <4 x float> %360, i64 3, !dbg !14
  %371 = tail call float @llvm.maxnum.f32(float %357, float %370), !dbg !7
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep4476, align 8, !dbg !14, !frisk.k_tail_pair !22
  %373 = load <4 x half>, ptr addrspace(3) %lds.const.gep4480, align 8, !dbg !14, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %337, <4 x float> %363) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %375 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %347, <4 x float> %303) #8, !dbg !14
  %376 = extractelement <4 x float> %374, i64 0, !dbg !14
  %377 = tail call float @llvm.maxnum.f32(float %365, float %376), !dbg !7
  %378 = extractelement <4 x float> %374, i64 1, !dbg !14
  %379 = tail call float @llvm.maxnum.f32(float %367, float %378), !dbg !7
  %380 = extractelement <4 x float> %374, i64 2, !dbg !14
  %381 = tail call float @llvm.maxnum.f32(float %369, float %380), !dbg !7
  %382 = extractelement <4 x float> %374, i64 3, !dbg !14
  %383 = tail call float @llvm.maxnum.f32(float %371, float %382), !dbg !7
  %384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %358, <4 x float> %375) #8, !dbg !14
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %361, <4 x float> %305) #8, !dbg !14
  %386 = extractelement <4 x float> %384, i64 0, !dbg !14
  %387 = tail call float @llvm.maxnum.f32(float %377, float %386), !dbg !7
  %388 = extractelement <4 x float> %384, i64 1, !dbg !14
  %389 = tail call float @llvm.maxnum.f32(float %379, float %388), !dbg !7
  %390 = extractelement <4 x float> %384, i64 2, !dbg !14
  %391 = tail call float @llvm.maxnum.f32(float %381, float %390), !dbg !7
  %392 = extractelement <4 x float> %384, i64 3, !dbg !14
  %393 = tail call float @llvm.maxnum.f32(float %383, float %392), !dbg !7
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %372, <4 x float> %385) #8, !dbg !14
  %395 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %348, <4 x float> %307) #8, !dbg !14
  %396 = extractelement <4 x float> %394, i64 0, !dbg !14
  %397 = tail call float @llvm.maxnum.f32(float %387, float %396), !dbg !7
  %398 = extractelement <4 x float> %394, i64 1, !dbg !14
  %399 = tail call float @llvm.maxnum.f32(float %389, float %398), !dbg !7
  %400 = extractelement <4 x float> %394, i64 2, !dbg !14
  %401 = tail call float @llvm.maxnum.f32(float %391, float %400), !dbg !7
  %402 = extractelement <4 x float> %394, i64 3, !dbg !14
  %403 = tail call float @llvm.maxnum.f32(float %393, float %402), !dbg !7
  %404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %359, <4 x float> %395) #8, !dbg !14
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %362, <4 x float> %309) #8, !dbg !14
  %406 = extractelement <4 x float> %404, i64 0, !dbg !14
  %407 = tail call float @llvm.maxnum.f32(float %397, float %406), !dbg !7
  %408 = extractelement <4 x float> %404, i64 1, !dbg !14
  %409 = tail call float @llvm.maxnum.f32(float %399, float %408), !dbg !7
  %410 = extractelement <4 x float> %404, i64 2, !dbg !14
  %411 = tail call float @llvm.maxnum.f32(float %401, float %410), !dbg !7
  %412 = extractelement <4 x float> %404, i64 3, !dbg !14
  %413 = tail call float @llvm.maxnum.f32(float %403, float %412), !dbg !7
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %373, <4 x float> %405) #8, !dbg !14
  %415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %330, <4 x float> %311) #8, !dbg !14
  %416 = extractelement <4 x float> %414, i64 0, !dbg !14
  %417 = tail call float @llvm.maxnum.f32(float %407, float %416), !dbg !7
  %418 = extractelement <4 x float> %414, i64 1, !dbg !14
  %419 = tail call float @llvm.maxnum.f32(float %409, float %418), !dbg !7
  %420 = extractelement <4 x float> %414, i64 2, !dbg !14
  %421 = tail call float @llvm.maxnum.f32(float %411, float %420), !dbg !7
  %422 = extractelement <4 x float> %414, i64 3, !dbg !14
  %423 = tail call float @llvm.maxnum.f32(float %413, float %422), !dbg !7
  %424 = tail call float @llvm.maxnum.f32(float %417, float %419), !dbg !7
  %425 = tail call float @llvm.maxnum.f32(float %421, float %423), !dbg !7
  %426 = tail call float @llvm.maxnum.f32(float %424, float %425), !dbg !7
  %427 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %332, <4 x float> %415) #8, !dbg !14
  %428 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %334, <4 x float> %313) #8, !dbg !14
  %429 = extractelement <4 x float> %427, i64 0, !dbg !14
  %430 = extractelement <4 x float> %427, i64 1, !dbg !14
  %431 = extractelement <4 x float> %427, i64 2, !dbg !14
  %432 = extractelement <4 x float> %427, i64 3, !dbg !14
  %433 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %336, <4 x float> %428) #8, !dbg !14
  %434 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %331, <4 x float> %315) #8, !dbg !14
  %435 = extractelement <4 x float> %433, i64 0, !dbg !14
  %436 = extractelement <4 x float> %433, i64 1, !dbg !14
  %437 = extractelement <4 x float> %433, i64 2, !dbg !14
  %438 = extractelement <4 x float> %433, i64 3, !dbg !14
  %439 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %333, <4 x float> %434) #8, !dbg !14
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %335, <4 x float> %317) #8, !dbg !14
  %441 = extractelement <4 x float> %439, i64 0, !dbg !14
  %442 = extractelement <4 x float> %439, i64 1, !dbg !14
  %443 = extractelement <4 x float> %439, i64 2, !dbg !14
  %444 = extractelement <4 x float> %439, i64 3, !dbg !14
  %445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %337, <4 x float> %440) #8, !dbg !14
  %446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %347, <4 x float> %319) #8, !dbg !14
  %447 = extractelement <4 x float> %445, i64 0, !dbg !14
  %448 = extractelement <4 x float> %445, i64 1, !dbg !14
  %449 = extractelement <4 x float> %445, i64 2, !dbg !14
  %450 = extractelement <4 x float> %445, i64 3, !dbg !14
  %451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %358, <4 x float> %446) #8, !dbg !14
  %452 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %361, <4 x float> %321) #8, !dbg !14
  %453 = extractelement <4 x float> %451, i64 0, !dbg !14
  %454 = extractelement <4 x float> %451, i64 1, !dbg !14
  %455 = extractelement <4 x float> %451, i64 2, !dbg !14
  %456 = extractelement <4 x float> %451, i64 3, !dbg !14
  %457 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %372, <4 x float> %452) #8, !dbg !14
  %458 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %348, <4 x float> %323) #8, !dbg !14
  %459 = extractelement <4 x float> %457, i64 0, !dbg !14
  %460 = extractelement <4 x float> %457, i64 1, !dbg !14
  %461 = extractelement <4 x float> %457, i64 2, !dbg !14
  %462 = extractelement <4 x float> %457, i64 3, !dbg !14
  %463 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %359, <4 x float> %458) #8, !dbg !14
  %464 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %362, <4 x float> %325) #8, !dbg !14
  %465 = extractelement <4 x float> %463, i64 0, !dbg !14
  %466 = extractelement <4 x float> %463, i64 1, !dbg !14
  %467 = extractelement <4 x float> %463, i64 2, !dbg !14
  %468 = extractelement <4 x float> %463, i64 3, !dbg !14
  %469 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %373, <4 x float> %464) #8, !dbg !14
  %470 = extractelement <4 x float> %469, i64 0, !dbg !14
  %471 = extractelement <4 x float> %469, i64 1, !dbg !14
  %472 = extractelement <4 x float> %469, i64 2, !dbg !14
  %473 = extractelement <4 x float> %469, i64 3, !dbg !14
  %474 = bitcast float %426 to i32, !dbg !7
  %475 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %474), !dbg !7
  %476 = bitcast i32 %475 to float, !dbg !7
  %477 = tail call float @llvm.maxnum.f32(float %426, float %476), !dbg !7
  %478 = bitcast float %477 to i32, !dbg !7
  %479 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %478), !dbg !7
  %480 = bitcast i32 %479 to float, !dbg !7
  %481 = tail call float @llvm.maxnum.f32(float %477, float %480), !dbg !7
  %482 = tail call float @llvm.maxnum.f32(float %429, float -inf), !dbg !7
  %483 = tail call float @llvm.maxnum.f32(float %482, float %435), !dbg !7
  %484 = tail call float @llvm.maxnum.f32(float %430, float %436), !dbg !7
  %485 = tail call float @llvm.maxnum.f32(float %431, float %437), !dbg !7
  %486 = tail call float @llvm.maxnum.f32(float %432, float %438), !dbg !7
  %487 = tail call float @llvm.maxnum.f32(float %483, float %441), !dbg !7
  %488 = tail call float @llvm.maxnum.f32(float %484, float %442), !dbg !7
  %489 = tail call float @llvm.maxnum.f32(float %485, float %443), !dbg !7
  %490 = tail call float @llvm.maxnum.f32(float %486, float %444), !dbg !7
  %491 = tail call float @llvm.maxnum.f32(float %487, float %447), !dbg !7
  %492 = tail call float @llvm.maxnum.f32(float %488, float %448), !dbg !7
  %493 = tail call float @llvm.maxnum.f32(float %489, float %449), !dbg !7
  %494 = tail call float @llvm.maxnum.f32(float %490, float %450), !dbg !7
  %495 = tail call float @llvm.maxnum.f32(float %491, float %453), !dbg !7
  %496 = tail call float @llvm.maxnum.f32(float %492, float %454), !dbg !7
  %497 = tail call float @llvm.maxnum.f32(float %493, float %455), !dbg !7
  %498 = tail call float @llvm.maxnum.f32(float %494, float %456), !dbg !7
  %499 = tail call float @llvm.maxnum.f32(float %495, float %459), !dbg !7
  %500 = tail call float @llvm.maxnum.f32(float %496, float %460), !dbg !7
  %501 = tail call float @llvm.maxnum.f32(float %497, float %461), !dbg !7
  %502 = tail call float @llvm.maxnum.f32(float %498, float %462), !dbg !7
  %503 = tail call float @llvm.maxnum.f32(float %499, float %465), !dbg !7
  %504 = tail call float @llvm.maxnum.f32(float %500, float %466), !dbg !7
  %505 = tail call float @llvm.maxnum.f32(float %501, float %467), !dbg !7
  %506 = tail call float @llvm.maxnum.f32(float %502, float %468), !dbg !7
  %507 = tail call float @llvm.maxnum.f32(float %503, float %470), !dbg !7
  %508 = tail call float @llvm.maxnum.f32(float %504, float %471), !dbg !7
  %509 = tail call float @llvm.maxnum.f32(float %505, float %472), !dbg !7
  %510 = tail call float @llvm.maxnum.f32(float %506, float %473), !dbg !7
  %511 = tail call float @llvm.maxnum.f32(float %507, float %508), !dbg !7
  %512 = tail call float @llvm.maxnum.f32(float %509, float %510), !dbg !7
  %513 = tail call float @llvm.maxnum.f32(float %511, float %512), !dbg !7
  %514 = bitcast float %513 to i32, !dbg !7
  %515 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %514), !dbg !7
  %516 = bitcast i32 %515 to float, !dbg !7
  %517 = tail call float @llvm.maxnum.f32(float %513, float %516), !dbg !7
  %518 = bitcast float %517 to i32, !dbg !7
  %519 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %518), !dbg !7
  %520 = bitcast i32 %519 to float, !dbg !7
  %521 = tail call float @llvm.maxnum.f32(float %517, float %520), !dbg !7
  %lazy.peak = tail call float @llvm.maxnum.f32(float %481, float %521), !dbg !40
  %lazy.over = fcmp ogt float %lazy.peak, 8.000000e+00, !dbg !40
  %lazy.mask = tail call i64 @llvm.amdgcn.ballot.i64(i1 %lazy.over), !dbg !40
  %lazy.any = icmp ne i64 %lazy.mask, 0, !dbg !40
  %lazy.first = icmp eq i64 %122, 0, !dbg !40
  %lazy.take = or i1 %lazy.first, %lazy.any, !dbg !40
  br i1 %lazy.take, label %lazy.slow, label %lazy.join, !dbg !40

lazy.slow:                                        ; preds = %115
  %lazy.positive = tail call float @llvm.maxnum.f32(float %481, float 0.000000e+00), !dbg !40
  %lazy.delta = select i1 %lazy.first, float %481, float %lazy.positive, !dbg !40
  %lazy.negated = fsub float 0.000000e+00, %lazy.delta, !dbg !40
  %lazy.exp = tail call float @llvm.amdgcn.exp2.f32(float %lazy.negated), !dbg !40
  %lazy.alpha = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp, !dbg !40
  %lazy.positive4882 = tail call float @llvm.maxnum.f32(float %521, float 0.000000e+00), !dbg !40
  %lazy.delta4883 = select i1 %lazy.first, float %521, float %lazy.positive4882, !dbg !40
  %lazy.negated4884 = fsub float 0.000000e+00, %lazy.delta4883, !dbg !40
  %lazy.exp4885 = tail call float @llvm.amdgcn.exp2.f32(float %lazy.negated4884), !dbg !40
  %lazy.alpha4886 = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp4885, !dbg !40
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !40
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !40
  %lazy.shifted = fsub <4 x float> %339, %lazy.delta.v, !dbg !40
  %lazy.shifted4889 = fsub <4 x float> %346, %lazy.delta.v, !dbg !40
  %lazy.shifted4892 = fsub <4 x float> %360, %lazy.delta.v, !dbg !40
  %lazy.shifted4895 = fsub <4 x float> %374, %lazy.delta.v, !dbg !40
  %lazy.shifted4898 = fsub <4 x float> %384, %lazy.delta.v, !dbg !40
  %lazy.shifted4901 = fsub <4 x float> %394, %lazy.delta.v, !dbg !40
  %lazy.shifted4904 = fsub <4 x float> %404, %lazy.delta.v, !dbg !40
  %lazy.shifted4907 = fsub <4 x float> %414, %lazy.delta.v, !dbg !40
  %lazy.delta.v.lane4908 = insertelement <1 x float> poison, float %lazy.delta4883, i64 0, !dbg !40
  %lazy.delta.v4909 = shufflevector <1 x float> %lazy.delta.v.lane4908, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !40
  %lazy.shifted4910 = fsub <4 x float> %427, %lazy.delta.v4909, !dbg !40
  %lazy.shifted4913 = fsub <4 x float> %433, %lazy.delta.v4909, !dbg !40
  %lazy.shifted4916 = fsub <4 x float> %439, %lazy.delta.v4909, !dbg !40
  %lazy.shifted4919 = fsub <4 x float> %445, %lazy.delta.v4909, !dbg !40
  %lazy.shifted4922 = fsub <4 x float> %451, %lazy.delta.v4909, !dbg !40
  %lazy.shifted4925 = fsub <4 x float> %457, %lazy.delta.v4909, !dbg !40
  %lazy.shifted4928 = fsub <4 x float> %463, %lazy.delta.v4909, !dbg !40
  %lazy.shifted4931 = fsub <4 x float> %469, %lazy.delta.v4909, !dbg !40
  %lazy.state = extractvalue [2 x <1 x float>] %120, 0, !dbg !40
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !40
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !40
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !40
  %lazy.state4932 = extractvalue [2 x <1 x float>] %120, 1, !dbg !40
  %lazy.alpha.v.lane4933 = insertelement <1 x float> poison, float %lazy.alpha4886, i64 0, !dbg !40
  %lazy.scaled4934 = fmul <1 x float> %lazy.state4932, %lazy.alpha.v.lane4933, !dbg !40
  %lazy.state.next4935 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled4934, 1, !dbg !40
  %lazy.state4936 = extractvalue [2 x <8 x float>] %116, 0, !dbg !40
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !40
  %lazy.scaled4938 = fmul <8 x float> %lazy.state4936, %lazy.alpha.v, !dbg !40
  %lazy.state.next4939 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled4938, 0, !dbg !40
  %lazy.state4940 = extractvalue [2 x <8 x float>] %116, 1, !dbg !40
  %lazy.alpha.v4942 = shufflevector <1 x float> %lazy.alpha.v.lane4933, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !40
  %lazy.scaled4943 = fmul <8 x float> %lazy.state4940, %lazy.alpha.v4942, !dbg !40
  %lazy.state.next4944 = insertvalue [2 x <8 x float>] %lazy.state.next4939, <8 x float> %lazy.scaled4943, 1, !dbg !40
  %lazy.state4945 = extractvalue [2 x <8 x float>] %117, 0, !dbg !40
  %lazy.scaled4948 = fmul <8 x float> %lazy.state4945, %lazy.alpha.v, !dbg !40
  %lazy.state.next4949 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled4948, 0, !dbg !40
  %lazy.state4950 = extractvalue [2 x <8 x float>] %117, 1, !dbg !40
  %lazy.scaled4953 = fmul <8 x float> %lazy.state4950, %lazy.alpha.v4942, !dbg !40
  %lazy.state.next4954 = insertvalue [2 x <8 x float>] %lazy.state.next4949, <8 x float> %lazy.scaled4953, 1, !dbg !40
  %lazy.state4955 = extractvalue [2 x <8 x float>] %118, 0, !dbg !40
  %lazy.scaled4958 = fmul <8 x float> %lazy.state4955, %lazy.alpha.v, !dbg !40
  %lazy.state.next4959 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled4958, 0, !dbg !40
  %lazy.state4960 = extractvalue [2 x <8 x float>] %118, 1, !dbg !40
  %lazy.scaled4963 = fmul <8 x float> %lazy.state4960, %lazy.alpha.v4942, !dbg !40
  %lazy.state.next4964 = insertvalue [2 x <8 x float>] %lazy.state.next4959, <8 x float> %lazy.scaled4963, 1, !dbg !40
  %lazy.state4965 = extractvalue [2 x <8 x float>] %121, 0, !dbg !40
  %lazy.scaled4968 = fmul <8 x float> %lazy.state4965, %lazy.alpha.v, !dbg !40
  %lazy.state.next4969 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled4968, 0, !dbg !40
  %lazy.state4970 = extractvalue [2 x <8 x float>] %121, 1, !dbg !40
  %lazy.scaled4973 = fmul <8 x float> %lazy.state4970, %lazy.alpha.v4942, !dbg !40
  %lazy.state.next4974 = insertvalue [2 x <8 x float>] %lazy.state.next4969, <8 x float> %lazy.scaled4973, 1, !dbg !40
  %lazy.base.next = fsub <1 x float> %lazy.base, %lazy.delta.v.lane, !dbg !40
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !40
  %lazy.base.next4977 = fsub <1 x float> %lazy.base4880, %lazy.delta.v.lane4908, !dbg !40
  %lazy.base.state4978 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next4977, 1, !dbg !40
  %.pre = extractelement <4 x float> %lazy.shifted, i64 0, !dbg !41
  %.pre29 = extractelement <4 x float> %lazy.shifted, i64 1, !dbg !41
  %.pre31 = extractelement <4 x float> %lazy.shifted, i64 2, !dbg !41
  %.pre33 = extractelement <4 x float> %lazy.shifted, i64 3, !dbg !41
  %.pre35 = extractelement <4 x float> %lazy.shifted4910, i64 0, !dbg !41
  %.pre37 = extractelement <4 x float> %lazy.shifted4910, i64 1, !dbg !41
  %.pre39 = extractelement <4 x float> %lazy.shifted4910, i64 2, !dbg !41
  %.pre41 = extractelement <4 x float> %lazy.shifted4910, i64 3, !dbg !41
  %.pre43 = extractelement <4 x float> %lazy.shifted4889, i64 0, !dbg !41
  %.pre45 = extractelement <4 x float> %lazy.shifted4889, i64 1, !dbg !41
  %.pre47 = extractelement <4 x float> %lazy.shifted4889, i64 2, !dbg !41
  %.pre49 = extractelement <4 x float> %lazy.shifted4889, i64 3, !dbg !41
  %.pre51 = extractelement <4 x float> %lazy.shifted4913, i64 0, !dbg !41
  %.pre53 = extractelement <4 x float> %lazy.shifted4913, i64 1, !dbg !41
  %.pre55 = extractelement <4 x float> %lazy.shifted4913, i64 2, !dbg !41
  %.pre57 = extractelement <4 x float> %lazy.shifted4913, i64 3, !dbg !41
  %.pre59 = extractelement <4 x float> %lazy.shifted4892, i64 0, !dbg !41
  %.pre61 = extractelement <4 x float> %lazy.shifted4892, i64 1, !dbg !41
  %.pre63 = extractelement <4 x float> %lazy.shifted4892, i64 2, !dbg !41
  %.pre65 = extractelement <4 x float> %lazy.shifted4892, i64 3, !dbg !41
  %.pre67 = extractelement <4 x float> %lazy.shifted4916, i64 0, !dbg !41
  %.pre69 = extractelement <4 x float> %lazy.shifted4916, i64 1, !dbg !41
  %.pre71 = extractelement <4 x float> %lazy.shifted4916, i64 2, !dbg !41
  %.pre73 = extractelement <4 x float> %lazy.shifted4916, i64 3, !dbg !41
  %.pre75 = extractelement <4 x float> %lazy.shifted4895, i64 0, !dbg !41
  %.pre77 = extractelement <4 x float> %lazy.shifted4895, i64 1, !dbg !41
  %.pre79 = extractelement <4 x float> %lazy.shifted4895, i64 2, !dbg !41
  %.pre81 = extractelement <4 x float> %lazy.shifted4895, i64 3, !dbg !41
  %.pre83 = extractelement <4 x float> %lazy.shifted4919, i64 0, !dbg !41
  %.pre85 = extractelement <4 x float> %lazy.shifted4919, i64 1, !dbg !41
  %.pre87 = extractelement <4 x float> %lazy.shifted4919, i64 2, !dbg !41
  %.pre89 = extractelement <4 x float> %lazy.shifted4919, i64 3, !dbg !41
  %.pre91 = extractelement <4 x float> %lazy.shifted4898, i64 0, !dbg !41
  %.pre93 = extractelement <4 x float> %lazy.shifted4898, i64 1, !dbg !41
  %.pre95 = extractelement <4 x float> %lazy.shifted4898, i64 2, !dbg !41
  %.pre97 = extractelement <4 x float> %lazy.shifted4898, i64 3, !dbg !41
  %.pre99 = extractelement <4 x float> %lazy.shifted4922, i64 0, !dbg !41
  %.pre101 = extractelement <4 x float> %lazy.shifted4922, i64 1, !dbg !41
  %.pre103 = extractelement <4 x float> %lazy.shifted4922, i64 2, !dbg !41
  %.pre105 = extractelement <4 x float> %lazy.shifted4922, i64 3, !dbg !41
  %.pre107 = extractelement <4 x float> %lazy.shifted4901, i64 0, !dbg !41
  %.pre109 = extractelement <4 x float> %lazy.shifted4901, i64 1, !dbg !41
  %.pre111 = extractelement <4 x float> %lazy.shifted4901, i64 2, !dbg !41
  %.pre113 = extractelement <4 x float> %lazy.shifted4901, i64 3, !dbg !41
  %.pre115 = extractelement <4 x float> %lazy.shifted4925, i64 0, !dbg !41
  %.pre117 = extractelement <4 x float> %lazy.shifted4925, i64 1, !dbg !41
  %.pre119 = extractelement <4 x float> %lazy.shifted4925, i64 2, !dbg !41
  %.pre121 = extractelement <4 x float> %lazy.shifted4925, i64 3, !dbg !41
  %.pre123 = extractelement <4 x float> %lazy.shifted4904, i64 0, !dbg !41
  %.pre125 = extractelement <4 x float> %lazy.shifted4904, i64 1, !dbg !41
  %.pre127 = extractelement <4 x float> %lazy.shifted4904, i64 2, !dbg !41
  %.pre129 = extractelement <4 x float> %lazy.shifted4904, i64 3, !dbg !41
  %.pre131 = extractelement <4 x float> %lazy.shifted4928, i64 0, !dbg !41
  %.pre133 = extractelement <4 x float> %lazy.shifted4928, i64 1, !dbg !41
  %.pre135 = extractelement <4 x float> %lazy.shifted4928, i64 2, !dbg !41
  %.pre137 = extractelement <4 x float> %lazy.shifted4928, i64 3, !dbg !41
  %.pre139 = extractelement <4 x float> %lazy.shifted4907, i64 0, !dbg !41
  %.pre141 = extractelement <4 x float> %lazy.shifted4907, i64 1, !dbg !41
  %.pre143 = extractelement <4 x float> %lazy.shifted4907, i64 2, !dbg !41
  %.pre145 = extractelement <4 x float> %lazy.shifted4907, i64 3, !dbg !41
  %.pre147 = extractelement <4 x float> %lazy.shifted4931, i64 0, !dbg !41
  %.pre149 = extractelement <4 x float> %lazy.shifted4931, i64 1, !dbg !41
  %.pre151 = extractelement <4 x float> %lazy.shifted4931, i64 2, !dbg !41
  %.pre153 = extractelement <4 x float> %lazy.shifted4931, i64 3, !dbg !41
  br label %lazy.join, !dbg !40

lazy.join:                                        ; preds = %lazy.slow, %115
  %.pre-phi154 = phi float [ %.pre153, %lazy.slow ], [ %473, %115 ], !dbg !41
  %.pre-phi152 = phi float [ %.pre151, %lazy.slow ], [ %472, %115 ], !dbg !41
  %.pre-phi150 = phi float [ %.pre149, %lazy.slow ], [ %471, %115 ], !dbg !41
  %.pre-phi148 = phi float [ %.pre147, %lazy.slow ], [ %470, %115 ], !dbg !41
  %.pre-phi146 = phi float [ %.pre145, %lazy.slow ], [ %422, %115 ], !dbg !41
  %.pre-phi144 = phi float [ %.pre143, %lazy.slow ], [ %420, %115 ], !dbg !41
  %.pre-phi142 = phi float [ %.pre141, %lazy.slow ], [ %418, %115 ], !dbg !41
  %.pre-phi140 = phi float [ %.pre139, %lazy.slow ], [ %416, %115 ], !dbg !41
  %.pre-phi138 = phi float [ %.pre137, %lazy.slow ], [ %468, %115 ], !dbg !41
  %.pre-phi136 = phi float [ %.pre135, %lazy.slow ], [ %467, %115 ], !dbg !41
  %.pre-phi134 = phi float [ %.pre133, %lazy.slow ], [ %466, %115 ], !dbg !41
  %.pre-phi132 = phi float [ %.pre131, %lazy.slow ], [ %465, %115 ], !dbg !41
  %.pre-phi130 = phi float [ %.pre129, %lazy.slow ], [ %412, %115 ], !dbg !41
  %.pre-phi128 = phi float [ %.pre127, %lazy.slow ], [ %410, %115 ], !dbg !41
  %.pre-phi126 = phi float [ %.pre125, %lazy.slow ], [ %408, %115 ], !dbg !41
  %.pre-phi124 = phi float [ %.pre123, %lazy.slow ], [ %406, %115 ], !dbg !41
  %.pre-phi122 = phi float [ %.pre121, %lazy.slow ], [ %462, %115 ], !dbg !41
  %.pre-phi120 = phi float [ %.pre119, %lazy.slow ], [ %461, %115 ], !dbg !41
  %.pre-phi118 = phi float [ %.pre117, %lazy.slow ], [ %460, %115 ], !dbg !41
  %.pre-phi116 = phi float [ %.pre115, %lazy.slow ], [ %459, %115 ], !dbg !41
  %.pre-phi114 = phi float [ %.pre113, %lazy.slow ], [ %402, %115 ], !dbg !41
  %.pre-phi112 = phi float [ %.pre111, %lazy.slow ], [ %400, %115 ], !dbg !41
  %.pre-phi110 = phi float [ %.pre109, %lazy.slow ], [ %398, %115 ], !dbg !41
  %.pre-phi108 = phi float [ %.pre107, %lazy.slow ], [ %396, %115 ], !dbg !41
  %.pre-phi106 = phi float [ %.pre105, %lazy.slow ], [ %456, %115 ], !dbg !41
  %.pre-phi104 = phi float [ %.pre103, %lazy.slow ], [ %455, %115 ], !dbg !41
  %.pre-phi102 = phi float [ %.pre101, %lazy.slow ], [ %454, %115 ], !dbg !41
  %.pre-phi100 = phi float [ %.pre99, %lazy.slow ], [ %453, %115 ], !dbg !41
  %.pre-phi98 = phi float [ %.pre97, %lazy.slow ], [ %392, %115 ], !dbg !41
  %.pre-phi96 = phi float [ %.pre95, %lazy.slow ], [ %390, %115 ], !dbg !41
  %.pre-phi94 = phi float [ %.pre93, %lazy.slow ], [ %388, %115 ], !dbg !41
  %.pre-phi92 = phi float [ %.pre91, %lazy.slow ], [ %386, %115 ], !dbg !41
  %.pre-phi90 = phi float [ %.pre89, %lazy.slow ], [ %450, %115 ], !dbg !41
  %.pre-phi88 = phi float [ %.pre87, %lazy.slow ], [ %449, %115 ], !dbg !41
  %.pre-phi86 = phi float [ %.pre85, %lazy.slow ], [ %448, %115 ], !dbg !41
  %.pre-phi84 = phi float [ %.pre83, %lazy.slow ], [ %447, %115 ], !dbg !41
  %.pre-phi82 = phi float [ %.pre81, %lazy.slow ], [ %382, %115 ], !dbg !41
  %.pre-phi80 = phi float [ %.pre79, %lazy.slow ], [ %380, %115 ], !dbg !41
  %.pre-phi78 = phi float [ %.pre77, %lazy.slow ], [ %378, %115 ], !dbg !41
  %.pre-phi76 = phi float [ %.pre75, %lazy.slow ], [ %376, %115 ], !dbg !41
  %.pre-phi74 = phi float [ %.pre73, %lazy.slow ], [ %444, %115 ], !dbg !41
  %.pre-phi72 = phi float [ %.pre71, %lazy.slow ], [ %443, %115 ], !dbg !41
  %.pre-phi70 = phi float [ %.pre69, %lazy.slow ], [ %442, %115 ], !dbg !41
  %.pre-phi68 = phi float [ %.pre67, %lazy.slow ], [ %441, %115 ], !dbg !41
  %.pre-phi66 = phi float [ %.pre65, %lazy.slow ], [ %370, %115 ], !dbg !41
  %.pre-phi64 = phi float [ %.pre63, %lazy.slow ], [ %368, %115 ], !dbg !41
  %.pre-phi62 = phi float [ %.pre61, %lazy.slow ], [ %366, %115 ], !dbg !41
  %.pre-phi60 = phi float [ %.pre59, %lazy.slow ], [ %364, %115 ], !dbg !41
  %.pre-phi58 = phi float [ %.pre57, %lazy.slow ], [ %438, %115 ], !dbg !41
  %.pre-phi56 = phi float [ %.pre55, %lazy.slow ], [ %437, %115 ], !dbg !41
  %.pre-phi54 = phi float [ %.pre53, %lazy.slow ], [ %436, %115 ], !dbg !41
  %.pre-phi52 = phi float [ %.pre51, %lazy.slow ], [ %435, %115 ], !dbg !41
  %.pre-phi50 = phi float [ %.pre49, %lazy.slow ], [ %356, %115 ], !dbg !41
  %.pre-phi48 = phi float [ %.pre47, %lazy.slow ], [ %354, %115 ], !dbg !41
  %.pre-phi46 = phi float [ %.pre45, %lazy.slow ], [ %352, %115 ], !dbg !41
  %.pre-phi44 = phi float [ %.pre43, %lazy.slow ], [ %350, %115 ], !dbg !41
  %.pre-phi42 = phi float [ %.pre41, %lazy.slow ], [ %432, %115 ], !dbg !41
  %.pre-phi40 = phi float [ %.pre39, %lazy.slow ], [ %431, %115 ], !dbg !41
  %.pre-phi38 = phi float [ %.pre37, %lazy.slow ], [ %430, %115 ], !dbg !41
  %.pre-phi36 = phi float [ %.pre35, %lazy.slow ], [ %429, %115 ], !dbg !41
  %.pre-phi34 = phi float [ %.pre33, %lazy.slow ], [ %345, %115 ], !dbg !41
  %.pre-phi32 = phi float [ %.pre31, %lazy.slow ], [ %344, %115 ], !dbg !41
  %.pre-phi30 = phi float [ %.pre29, %lazy.slow ], [ %343, %115 ], !dbg !41
  %.pre-phi = phi float [ %.pre, %lazy.slow ], [ %341, %115 ], !dbg !41
  %lazy.state4994 = phi [2 x <1 x float>] [ %lazy.state.next4935, %lazy.slow ], [ %120, %115 ], !dbg !40
  %lazy.state4995 = phi [2 x <8 x float>] [ %lazy.state.next4944, %lazy.slow ], [ %116, %115 ], !dbg !40
  %lazy.state4996 = phi [2 x <8 x float>] [ %lazy.state.next4954, %lazy.slow ], [ %117, %115 ], !dbg !40
  %lazy.state4997 = phi [2 x <8 x float>] [ %lazy.state.next4964, %lazy.slow ], [ %118, %115 ], !dbg !40
  %lazy.state4998 = phi [2 x <8 x float>] [ %lazy.state.next4974, %lazy.slow ], [ %121, %115 ], !dbg !40
  %lazy.base.join = phi [2 x <1 x float>] [ %lazy.base.state4978, %lazy.slow ], [ %119, %115 ], !dbg !40
  %522 = extractvalue [2 x <1 x float>] %lazy.state4994, 0, !dbg !40
  %523 = extractelement <1 x float> %522, i64 0, !dbg !40
  %524 = extractvalue [2 x <1 x float>] %lazy.state4994, 1, !dbg !40
  %525 = extractelement <1 x float> %524, i64 0, !dbg !40
  %526 = extractvalue [2 x <8 x float>] %lazy.state4998, 0, !dbg !42
  %527 = shufflevector <8 x float> %526, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  %528 = shufflevector <8 x float> %526, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %529 = extractvalue [2 x <8 x float>] %lazy.state4998, 1, !dbg !42
  %530 = shufflevector <8 x float> %529, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  %531 = shufflevector <8 x float> %529, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %address.local3862 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local3188, !dbg !39
  %address.local3863 = ptrtoint ptr addrspace(3) %address.local3862 to i64, !dbg !39
  %address.local3864 = trunc i64 %address.local3863 to i32, !dbg !39
  %532 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3864), !dbg !39
  %.idx47054707 = or disjoint i64 %address.local3194.idx4700, %125, !dbg !39
  %.idx47054707.tr = trunc nuw nsw i64 %.idx47054707 to i32, !dbg !39
  %dma.buffer.offset4569 = shl nuw nsw i32 %.idx47054707.tr, 1, !dbg !39
  %533 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %532, i32 %dma.buffer.offset4569, <4 x i32> %dma.buffer.rsrc4574) #9, !dbg !39
  %address.local3874 = getelementptr i8, ptr addrspace(3) %address.local3862, i64 4096, !dbg !39
  %address.local3875 = ptrtoint ptr addrspace(3) %address.local3874 to i64, !dbg !39
  %address.local3876 = trunc i64 %address.local3875 to i32, !dbg !39
  %534 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3876), !dbg !39
  %dma.buffer.offset4567 = or disjoint i32 %dma.buffer.offset4569, 32, !dbg !39
  %535 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %534, i32 %dma.buffer.offset4567, <4 x i32> %dma.buffer.rsrc4574) #9, !dbg !39
  %address.local3883 = shl nuw nsw i32 %address.thread1132, 3, !dbg !43
  %address.local3884 = and i32 %address.local3883, 384, !dbg !43
  %address.local3886 = shl nuw nsw i32 %address.thread1132, 10, !dbg !43
  %address.local3887 = and i32 %address.local3886, 3072, !dbg !43
  %536 = or disjoint i32 %address.local3884, %address.local3887, !dbg !43
  %address.local3888 = or disjoint i32 %536, %address.local1247, !dbg !43
  %address.local3889 = zext nneg i32 %address.local3888 to i64, !dbg !43
  %address.local3890 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local3889, !dbg !43
  %537 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3890, i32 0) #11, !dbg !43
  %538 = bitcast <4 x i32> %537 to <8 x half>, !dbg !43
  %address.local3901 = getelementptr i8, ptr addrspace(3) %address.local3890, i64 64, !dbg !43
  %539 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3901, i32 0) #11, !dbg !43
  %540 = bitcast <4 x i32> %539 to <8 x half>, !dbg !43
  %address.local3912 = getelementptr i8, ptr addrspace(3) %address.local3890, i64 128, !dbg !43
  %541 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3912, i32 0) #11, !dbg !43
  %542 = bitcast <4 x i32> %541 to <8 x half>, !dbg !43
  %address.local3923 = getelementptr i8, ptr addrspace(3) %address.local3890, i64 192, !dbg !43
  %543 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3923, i32 0) #11, !dbg !43
  %544 = bitcast <4 x i32> %543 to <8 x half>, !dbg !43
  %address.local3934 = getelementptr i8, ptr addrspace(3) %address.local3890, i64 1024, !dbg !43
  %545 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3934, i32 0) #11, !dbg !43
  %546 = bitcast <4 x i32> %545 to <8 x half>, !dbg !43
  %address.local3945 = getelementptr i8, ptr addrspace(3) %address.local3890, i64 1088, !dbg !43
  %547 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3945, i32 0) #11, !dbg !43
  %548 = bitcast <4 x i32> %547 to <8 x half>, !dbg !43
  %address.local3956 = getelementptr i8, ptr addrspace(3) %address.local3890, i64 1152, !dbg !43
  %549 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3956, i32 0) #11, !dbg !43
  %550 = bitcast <4 x i32> %549 to <8 x half>, !dbg !43
  %address.local3967 = getelementptr i8, ptr addrspace(3) %address.local3890, i64 1216, !dbg !43
  %551 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3967, i32 0) #11, !dbg !43
  %552 = shufflevector <8 x half> %538, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %553 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi), !dbg !41
  %554 = insertelement <4 x float> poison, float %553, i64 0, !dbg !44
  %555 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi30), !dbg !41
  %556 = insertelement <4 x float> %554, float %555, i64 1, !dbg !44
  %557 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi32), !dbg !41
  %558 = insertelement <4 x float> %556, float %557, i64 2, !dbg !44
  %559 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi34), !dbg !41
  %560 = insertelement <4 x float> %558, float %559, i64 3, !dbg !44
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %560) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i64 0, !dbg !44
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i64 1, !dbg !44
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i64 2, !dbg !44
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i64 3, !dbg !44
  %.cvtpack.p0 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1) #12, !dbg !44
  %.cvtpack.p1 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3) #12, !dbg !44
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i64 0, !dbg !44
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i64 1, !dbg !44
  %561 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !44
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %561) #8, !dbg !41, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %562 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %552, <4 x float> %527) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %563 = shufflevector <8 x half> %538, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %564 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %563, <4 x float> %528) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %565 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi36), !dbg !41
  %566 = insertelement <4 x float> poison, float %565, i64 0, !dbg !44
  %567 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi38), !dbg !41
  %568 = insertelement <4 x float> %566, float %567, i64 1, !dbg !44
  %569 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi40), !dbg !41
  %570 = insertelement <4 x float> %568, float %569, i64 2, !dbg !44
  %571 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi42), !dbg !41
  %572 = insertelement <4 x float> %570, float %571, i64 3, !dbg !44
  %math.result858 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %572) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f04999 = extractelement <4 x float> %math.result858, i64 0, !dbg !44
  %.cvtpack.f15000 = extractelement <4 x float> %math.result858, i64 1, !dbg !44
  %.cvtpack.f25001 = extractelement <4 x float> %math.result858, i64 2, !dbg !44
  %.cvtpack.f35002 = extractelement <4 x float> %math.result858, i64 3, !dbg !44
  %.cvtpack.p05003 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04999, float %.cvtpack.f15000) #12, !dbg !44
  %.cvtpack.p15004 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25001, float %.cvtpack.f35002) #12, !dbg !44
  %.cvtpack.v05005 = insertelement <2 x i32> poison, i32 %.cvtpack.p05003, i64 0, !dbg !44
  %.cvtpack.v15006 = insertelement <2 x i32> %.cvtpack.v05005, i32 %.cvtpack.p15004, i64 1, !dbg !44
  %573 = bitcast <2 x i32> %.cvtpack.v15006 to <4 x half>, !dbg !44
  %math.packed859 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %573) #8, !dbg !41, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %574 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %552, <4 x float> %530) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %575 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %563, <4 x float> %531) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %576 = shufflevector <8 x half> %540, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %577 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi44), !dbg !41
  %578 = fadd float %553, %577, !dbg !45
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %578) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %579 = insertelement <4 x float> poison, float %577, i64 0, !dbg !44
  %580 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi46), !dbg !41
  %581 = fadd float %555, %580, !dbg !45
  %k0.sum.anchor4723 = tail call float asm sideeffect "", "=v,0"(float %581) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %582 = insertelement <4 x float> %579, float %580, i64 1, !dbg !44
  %583 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi48), !dbg !41
  %584 = fadd float %557, %583, !dbg !45
  %k0.sum.anchor4724 = tail call float asm sideeffect "", "=v,0"(float %584) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %585 = insertelement <4 x float> %582, float %583, i64 2, !dbg !44
  %586 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi50), !dbg !41
  %587 = fadd float %559, %586, !dbg !45
  %k0.sum.anchor4725 = tail call float asm sideeffect "", "=v,0"(float %587) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %588 = insertelement <4 x float> %585, float %586, i64 3, !dbg !44
  %math.result844 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %588) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f05007 = extractelement <4 x float> %math.result844, i64 0, !dbg !44
  %.cvtpack.f15008 = extractelement <4 x float> %math.result844, i64 1, !dbg !44
  %.cvtpack.f25009 = extractelement <4 x float> %math.result844, i64 2, !dbg !44
  %.cvtpack.f35010 = extractelement <4 x float> %math.result844, i64 3, !dbg !44
  %.cvtpack.p05011 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05007, float %.cvtpack.f15008) #12, !dbg !44
  %.cvtpack.p15012 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25009, float %.cvtpack.f35010) #12, !dbg !44
  %.cvtpack.v05013 = insertelement <2 x i32> poison, i32 %.cvtpack.p05011, i64 0, !dbg !44
  %.cvtpack.v15014 = insertelement <2 x i32> %.cvtpack.v05013, i32 %.cvtpack.p15012, i64 1, !dbg !44
  %589 = bitcast <2 x i32> %.cvtpack.v15014 to <4 x half>, !dbg !44
  %math.packed845 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %589) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %590 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %576, <4 x float> %562) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %591 = shufflevector <8 x half> %540, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %592 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %591, <4 x float> %564) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %593 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi52), !dbg !41
  %594 = fadd float %565, %593, !dbg !45
  %k0.sum.anchor4726 = tail call float asm sideeffect "", "=v,0"(float %594) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %595 = insertelement <4 x float> poison, float %593, i64 0, !dbg !44
  %596 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi54), !dbg !41
  %597 = fadd float %567, %596, !dbg !45
  %k0.sum.anchor4727 = tail call float asm sideeffect "", "=v,0"(float %597) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %598 = insertelement <4 x float> %595, float %596, i64 1, !dbg !44
  %599 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi56), !dbg !41
  %600 = fadd float %569, %599, !dbg !45
  %k0.sum.anchor4728 = tail call float asm sideeffect "", "=v,0"(float %600) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %601 = insertelement <4 x float> %598, float %599, i64 2, !dbg !44
  %602 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi58), !dbg !41
  %603 = fadd float %571, %602, !dbg !45
  %k0.sum.anchor4729 = tail call float asm sideeffect "", "=v,0"(float %603) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %604 = insertelement <4 x float> %601, float %602, i64 3, !dbg !44
  %math.result860 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %604) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f05015 = extractelement <4 x float> %math.result860, i64 0, !dbg !44
  %.cvtpack.f15016 = extractelement <4 x float> %math.result860, i64 1, !dbg !44
  %.cvtpack.f25017 = extractelement <4 x float> %math.result860, i64 2, !dbg !44
  %.cvtpack.f35018 = extractelement <4 x float> %math.result860, i64 3, !dbg !44
  %.cvtpack.p05019 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05015, float %.cvtpack.f15016) #12, !dbg !44
  %.cvtpack.p15020 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25017, float %.cvtpack.f35018) #12, !dbg !44
  %.cvtpack.v05021 = insertelement <2 x i32> poison, i32 %.cvtpack.p05019, i64 0, !dbg !44
  %.cvtpack.v15022 = insertelement <2 x i32> %.cvtpack.v05021, i32 %.cvtpack.p15020, i64 1, !dbg !44
  %605 = bitcast <2 x i32> %.cvtpack.v15022 to <4 x half>, !dbg !44
  %math.packed861 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %605) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %606 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %576, <4 x float> %574) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %607 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %591, <4 x float> %575) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %608 = shufflevector <8 x half> %542, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %609 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi60), !dbg !41
  %610 = fadd float %609, %k0.sum.anchor, !dbg !45
  %k0.sum.anchor4730 = tail call float asm sideeffect "", "=v,0"(float %610) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %611 = insertelement <4 x float> poison, float %609, i64 0, !dbg !44
  %612 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi62), !dbg !41
  %613 = fadd float %612, %k0.sum.anchor4723, !dbg !45
  %k0.sum.anchor4731 = tail call float asm sideeffect "", "=v,0"(float %613) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %614 = insertelement <4 x float> %611, float %612, i64 1, !dbg !44
  %615 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi64), !dbg !41
  %616 = fadd float %615, %k0.sum.anchor4724, !dbg !45
  %k0.sum.anchor4732 = tail call float asm sideeffect "", "=v,0"(float %616) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %617 = insertelement <4 x float> %614, float %615, i64 2, !dbg !44
  %618 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi66), !dbg !41
  %619 = fadd float %618, %k0.sum.anchor4725, !dbg !45
  %k0.sum.anchor4733 = tail call float asm sideeffect "", "=v,0"(float %619) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %620 = insertelement <4 x float> %617, float %618, i64 3, !dbg !44
  %math.result846 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %620) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f05023 = extractelement <4 x float> %math.result846, i64 0, !dbg !44
  %.cvtpack.f15024 = extractelement <4 x float> %math.result846, i64 1, !dbg !44
  %.cvtpack.f25025 = extractelement <4 x float> %math.result846, i64 2, !dbg !44
  %.cvtpack.f35026 = extractelement <4 x float> %math.result846, i64 3, !dbg !44
  %.cvtpack.p05027 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05023, float %.cvtpack.f15024) #12, !dbg !44
  %.cvtpack.p15028 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25025, float %.cvtpack.f35026) #12, !dbg !44
  %.cvtpack.v05029 = insertelement <2 x i32> poison, i32 %.cvtpack.p05027, i64 0, !dbg !44
  %.cvtpack.v15030 = insertelement <2 x i32> %.cvtpack.v05029, i32 %.cvtpack.p15028, i64 1, !dbg !44
  %621 = bitcast <2 x i32> %.cvtpack.v15030 to <4 x half>, !dbg !44
  %math.packed847 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %621) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %622 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %608, <4 x float> %590) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %623 = shufflevector <8 x half> %542, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %624 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %623, <4 x float> %592) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %625 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi68), !dbg !41
  %626 = fadd float %625, %k0.sum.anchor4726, !dbg !45
  %k0.sum.anchor4734 = tail call float asm sideeffect "", "=v,0"(float %626) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %627 = insertelement <4 x float> poison, float %625, i64 0, !dbg !44
  %628 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi70), !dbg !41
  %629 = fadd float %628, %k0.sum.anchor4727, !dbg !45
  %k0.sum.anchor4735 = tail call float asm sideeffect "", "=v,0"(float %629) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %630 = insertelement <4 x float> %627, float %628, i64 1, !dbg !44
  %631 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi72), !dbg !41
  %632 = fadd float %631, %k0.sum.anchor4728, !dbg !45
  %k0.sum.anchor4736 = tail call float asm sideeffect "", "=v,0"(float %632) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %633 = insertelement <4 x float> %630, float %631, i64 2, !dbg !44
  %634 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi74), !dbg !41
  %635 = fadd float %634, %k0.sum.anchor4729, !dbg !45
  %k0.sum.anchor4737 = tail call float asm sideeffect "", "=v,0"(float %635) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %636 = insertelement <4 x float> %633, float %634, i64 3, !dbg !44
  %math.result862 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %636) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f05031 = extractelement <4 x float> %math.result862, i64 0, !dbg !44
  %.cvtpack.f15032 = extractelement <4 x float> %math.result862, i64 1, !dbg !44
  %.cvtpack.f25033 = extractelement <4 x float> %math.result862, i64 2, !dbg !44
  %.cvtpack.f35034 = extractelement <4 x float> %math.result862, i64 3, !dbg !44
  %.cvtpack.p05035 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05031, float %.cvtpack.f15032) #12, !dbg !44
  %.cvtpack.p15036 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25033, float %.cvtpack.f35034) #12, !dbg !44
  %.cvtpack.v05037 = insertelement <2 x i32> poison, i32 %.cvtpack.p05035, i64 0, !dbg !44
  %.cvtpack.v15038 = insertelement <2 x i32> %.cvtpack.v05037, i32 %.cvtpack.p15036, i64 1, !dbg !44
  %637 = bitcast <2 x i32> %.cvtpack.v15038 to <4 x half>, !dbg !44
  %math.packed863 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %637) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %638 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %608, <4 x float> %606) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %639 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %623, <4 x float> %607) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %640 = shufflevector <8 x half> %544, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %641 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi76), !dbg !41
  %642 = fadd float %641, %k0.sum.anchor4730, !dbg !45
  %k0.sum.anchor4738 = tail call float asm sideeffect "", "=v,0"(float %642) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %643 = insertelement <4 x float> poison, float %641, i64 0, !dbg !44
  %644 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi78), !dbg !41
  %645 = fadd float %644, %k0.sum.anchor4731, !dbg !45
  %k0.sum.anchor4739 = tail call float asm sideeffect "", "=v,0"(float %645) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %646 = insertelement <4 x float> %643, float %644, i64 1, !dbg !44
  %647 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi80), !dbg !41
  %648 = fadd float %647, %k0.sum.anchor4732, !dbg !45
  %k0.sum.anchor4740 = tail call float asm sideeffect "", "=v,0"(float %648) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %649 = insertelement <4 x float> %646, float %647, i64 2, !dbg !44
  %650 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi82), !dbg !41
  %651 = fadd float %650, %k0.sum.anchor4733, !dbg !45
  %k0.sum.anchor4741 = tail call float asm sideeffect "", "=v,0"(float %651) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %652 = insertelement <4 x float> %649, float %650, i64 3, !dbg !44
  %math.result848 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %652) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f05039 = extractelement <4 x float> %math.result848, i64 0, !dbg !44
  %.cvtpack.f15040 = extractelement <4 x float> %math.result848, i64 1, !dbg !44
  %.cvtpack.f25041 = extractelement <4 x float> %math.result848, i64 2, !dbg !44
  %.cvtpack.f35042 = extractelement <4 x float> %math.result848, i64 3, !dbg !44
  %.cvtpack.p05043 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05039, float %.cvtpack.f15040) #12, !dbg !44
  %.cvtpack.p15044 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25041, float %.cvtpack.f35042) #12, !dbg !44
  %.cvtpack.v05045 = insertelement <2 x i32> poison, i32 %.cvtpack.p05043, i64 0, !dbg !44
  %.cvtpack.v15046 = insertelement <2 x i32> %.cvtpack.v05045, i32 %.cvtpack.p15044, i64 1, !dbg !44
  %653 = bitcast <2 x i32> %.cvtpack.v15046 to <4 x half>, !dbg !44
  %math.packed849 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %653) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %654 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %640, <4 x float> %622) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %655 = shufflevector <8 x half> %544, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %656 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %655, <4 x float> %624) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %657 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi84), !dbg !41
  %658 = fadd float %657, %k0.sum.anchor4734, !dbg !45
  %k0.sum.anchor4742 = tail call float asm sideeffect "", "=v,0"(float %658) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %659 = insertelement <4 x float> poison, float %657, i64 0, !dbg !44
  %660 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi86), !dbg !41
  %661 = fadd float %660, %k0.sum.anchor4735, !dbg !45
  %k0.sum.anchor4743 = tail call float asm sideeffect "", "=v,0"(float %661) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %662 = insertelement <4 x float> %659, float %660, i64 1, !dbg !44
  %663 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi88), !dbg !41
  %664 = fadd float %663, %k0.sum.anchor4736, !dbg !45
  %k0.sum.anchor4744 = tail call float asm sideeffect "", "=v,0"(float %664) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %665 = insertelement <4 x float> %662, float %663, i64 2, !dbg !44
  %666 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi90), !dbg !41
  %667 = fadd float %666, %k0.sum.anchor4737, !dbg !45
  %k0.sum.anchor4745 = tail call float asm sideeffect "", "=v,0"(float %667) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %668 = insertelement <4 x float> %665, float %666, i64 3, !dbg !44
  %math.result864 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %668) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f05047 = extractelement <4 x float> %math.result864, i64 0, !dbg !44
  %.cvtpack.f15048 = extractelement <4 x float> %math.result864, i64 1, !dbg !44
  %.cvtpack.f25049 = extractelement <4 x float> %math.result864, i64 2, !dbg !44
  %.cvtpack.f35050 = extractelement <4 x float> %math.result864, i64 3, !dbg !44
  %.cvtpack.p05051 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05047, float %.cvtpack.f15048) #12, !dbg !44
  %.cvtpack.p15052 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25049, float %.cvtpack.f35050) #12, !dbg !44
  %.cvtpack.v05053 = insertelement <2 x i32> poison, i32 %.cvtpack.p05051, i64 0, !dbg !44
  %.cvtpack.v15054 = insertelement <2 x i32> %.cvtpack.v05053, i32 %.cvtpack.p15052, i64 1, !dbg !44
  %669 = bitcast <2 x i32> %.cvtpack.v15054 to <4 x half>, !dbg !44
  %math.packed865 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %669) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %670 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %640, <4 x float> %638) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %671 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %655, <4 x float> %639) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %672 = shufflevector <8 x half> %546, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %673 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi92), !dbg !41
  %674 = fadd float %673, %k0.sum.anchor4738, !dbg !45
  %k0.sum.anchor4746 = tail call float asm sideeffect "", "=v,0"(float %674) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %675 = insertelement <4 x float> poison, float %673, i64 0, !dbg !44
  %676 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi94), !dbg !41
  %677 = fadd float %676, %k0.sum.anchor4739, !dbg !45
  %k0.sum.anchor4747 = tail call float asm sideeffect "", "=v,0"(float %677) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %678 = insertelement <4 x float> %675, float %676, i64 1, !dbg !44
  %679 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi96), !dbg !41
  %680 = fadd float %679, %k0.sum.anchor4740, !dbg !45
  %k0.sum.anchor4748 = tail call float asm sideeffect "", "=v,0"(float %680) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %681 = insertelement <4 x float> %678, float %679, i64 2, !dbg !44
  %682 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi98), !dbg !41
  %683 = fadd float %682, %k0.sum.anchor4741, !dbg !45
  %k0.sum.anchor4749 = tail call float asm sideeffect "", "=v,0"(float %683) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %684 = insertelement <4 x float> %681, float %682, i64 3, !dbg !44
  %math.result850 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %684) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f05055 = extractelement <4 x float> %math.result850, i64 0, !dbg !44
  %.cvtpack.f15056 = extractelement <4 x float> %math.result850, i64 1, !dbg !44
  %.cvtpack.f25057 = extractelement <4 x float> %math.result850, i64 2, !dbg !44
  %.cvtpack.f35058 = extractelement <4 x float> %math.result850, i64 3, !dbg !44
  %.cvtpack.p05059 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05055, float %.cvtpack.f15056) #12, !dbg !44
  %.cvtpack.p15060 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25057, float %.cvtpack.f35058) #12, !dbg !44
  %.cvtpack.v05061 = insertelement <2 x i32> poison, i32 %.cvtpack.p05059, i64 0, !dbg !44
  %.cvtpack.v15062 = insertelement <2 x i32> %.cvtpack.v05061, i32 %.cvtpack.p15060, i64 1, !dbg !44
  %685 = bitcast <2 x i32> %.cvtpack.v15062 to <4 x half>, !dbg !44
  %math.packed851 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %685) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %686 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %672, <4 x float> %654) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %687 = shufflevector <8 x half> %546, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %688 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %687, <4 x float> %656) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %689 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi100), !dbg !41
  %690 = fadd float %689, %k0.sum.anchor4742, !dbg !45
  %k0.sum.anchor4750 = tail call float asm sideeffect "", "=v,0"(float %690) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %691 = insertelement <4 x float> poison, float %689, i64 0, !dbg !44
  %692 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi102), !dbg !41
  %693 = fadd float %692, %k0.sum.anchor4743, !dbg !45
  %k0.sum.anchor4751 = tail call float asm sideeffect "", "=v,0"(float %693) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %694 = insertelement <4 x float> %691, float %692, i64 1, !dbg !44
  %695 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi104), !dbg !41
  %696 = fadd float %695, %k0.sum.anchor4744, !dbg !45
  %k0.sum.anchor4752 = tail call float asm sideeffect "", "=v,0"(float %696) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %697 = insertelement <4 x float> %694, float %695, i64 2, !dbg !44
  %698 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi106), !dbg !41
  %699 = fadd float %698, %k0.sum.anchor4745, !dbg !45
  %k0.sum.anchor4753 = tail call float asm sideeffect "", "=v,0"(float %699) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %700 = insertelement <4 x float> %697, float %698, i64 3, !dbg !44
  %math.result866 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %700) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f05063 = extractelement <4 x float> %math.result866, i64 0, !dbg !44
  %.cvtpack.f15064 = extractelement <4 x float> %math.result866, i64 1, !dbg !44
  %.cvtpack.f25065 = extractelement <4 x float> %math.result866, i64 2, !dbg !44
  %.cvtpack.f35066 = extractelement <4 x float> %math.result866, i64 3, !dbg !44
  %.cvtpack.p05067 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05063, float %.cvtpack.f15064) #12, !dbg !44
  %.cvtpack.p15068 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25065, float %.cvtpack.f35066) #12, !dbg !44
  %.cvtpack.v05069 = insertelement <2 x i32> poison, i32 %.cvtpack.p05067, i64 0, !dbg !44
  %.cvtpack.v15070 = insertelement <2 x i32> %.cvtpack.v05069, i32 %.cvtpack.p15068, i64 1, !dbg !44
  %701 = bitcast <2 x i32> %.cvtpack.v15070 to <4 x half>, !dbg !44
  %math.packed867 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %701) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %702 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %672, <4 x float> %670) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %703 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %687, <4 x float> %671) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %704 = shufflevector <8 x half> %548, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %705 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi108), !dbg !41
  %706 = fadd float %705, %k0.sum.anchor4746, !dbg !45
  %k0.sum.anchor4754 = tail call float asm sideeffect "", "=v,0"(float %706) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %707 = insertelement <4 x float> poison, float %705, i64 0, !dbg !44
  %708 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi110), !dbg !41
  %709 = fadd float %708, %k0.sum.anchor4747, !dbg !45
  %k0.sum.anchor4755 = tail call float asm sideeffect "", "=v,0"(float %709) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %710 = insertelement <4 x float> %707, float %708, i64 1, !dbg !44
  %711 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi112), !dbg !41
  %712 = fadd float %711, %k0.sum.anchor4748, !dbg !45
  %k0.sum.anchor4756 = tail call float asm sideeffect "", "=v,0"(float %712) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %713 = insertelement <4 x float> %710, float %711, i64 2, !dbg !44
  %714 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi114), !dbg !41
  %715 = fadd float %714, %k0.sum.anchor4749, !dbg !45
  %k0.sum.anchor4757 = tail call float asm sideeffect "", "=v,0"(float %715) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %716 = insertelement <4 x float> %713, float %714, i64 3, !dbg !44
  %math.result852 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %716) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f05071 = extractelement <4 x float> %math.result852, i64 0, !dbg !44
  %.cvtpack.f15072 = extractelement <4 x float> %math.result852, i64 1, !dbg !44
  %.cvtpack.f25073 = extractelement <4 x float> %math.result852, i64 2, !dbg !44
  %.cvtpack.f35074 = extractelement <4 x float> %math.result852, i64 3, !dbg !44
  %.cvtpack.p05075 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05071, float %.cvtpack.f15072) #12, !dbg !44
  %.cvtpack.p15076 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25073, float %.cvtpack.f35074) #12, !dbg !44
  %.cvtpack.v05077 = insertelement <2 x i32> poison, i32 %.cvtpack.p05075, i64 0, !dbg !44
  %.cvtpack.v15078 = insertelement <2 x i32> %.cvtpack.v05077, i32 %.cvtpack.p15076, i64 1, !dbg !44
  %717 = bitcast <2 x i32> %.cvtpack.v15078 to <4 x half>, !dbg !44
  %math.packed853 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %717) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %718 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %704, <4 x float> %686) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %719 = shufflevector <8 x half> %548, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %720 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %719, <4 x float> %688) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %721 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi116), !dbg !41
  %722 = fadd float %721, %k0.sum.anchor4750, !dbg !45
  %k0.sum.anchor4758 = tail call float asm sideeffect "", "=v,0"(float %722) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %723 = insertelement <4 x float> poison, float %721, i64 0, !dbg !44
  %724 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi118), !dbg !41
  %725 = fadd float %724, %k0.sum.anchor4751, !dbg !45
  %k0.sum.anchor4759 = tail call float asm sideeffect "", "=v,0"(float %725) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %726 = insertelement <4 x float> %723, float %724, i64 1, !dbg !44
  %727 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi120), !dbg !41
  %728 = fadd float %727, %k0.sum.anchor4752, !dbg !45
  %k0.sum.anchor4760 = tail call float asm sideeffect "", "=v,0"(float %728) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %729 = insertelement <4 x float> %726, float %727, i64 2, !dbg !44
  %730 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi122), !dbg !41
  %731 = fadd float %730, %k0.sum.anchor4753, !dbg !45
  %k0.sum.anchor4761 = tail call float asm sideeffect "", "=v,0"(float %731) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %732 = insertelement <4 x float> %729, float %730, i64 3, !dbg !44
  %math.result868 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %732) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f05079 = extractelement <4 x float> %math.result868, i64 0, !dbg !44
  %.cvtpack.f15080 = extractelement <4 x float> %math.result868, i64 1, !dbg !44
  %.cvtpack.f25081 = extractelement <4 x float> %math.result868, i64 2, !dbg !44
  %.cvtpack.f35082 = extractelement <4 x float> %math.result868, i64 3, !dbg !44
  %.cvtpack.p05083 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05079, float %.cvtpack.f15080) #12, !dbg !44
  %.cvtpack.p15084 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25081, float %.cvtpack.f35082) #12, !dbg !44
  %.cvtpack.v05085 = insertelement <2 x i32> poison, i32 %.cvtpack.p05083, i64 0, !dbg !44
  %.cvtpack.v15086 = insertelement <2 x i32> %.cvtpack.v05085, i32 %.cvtpack.p15084, i64 1, !dbg !44
  %733 = bitcast <2 x i32> %.cvtpack.v15086 to <4 x half>, !dbg !44
  %math.packed869 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %733) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %734 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %704, <4 x float> %702) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %735 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %719, <4 x float> %703) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %736 = shufflevector <8 x half> %550, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %737 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi124), !dbg !41
  %738 = fadd float %737, %k0.sum.anchor4754, !dbg !45
  %k0.sum.anchor4762 = tail call float asm sideeffect "", "=v,0"(float %738) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %739 = insertelement <4 x float> poison, float %737, i64 0, !dbg !44
  %740 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi126), !dbg !41
  %741 = fadd float %740, %k0.sum.anchor4755, !dbg !45
  %k0.sum.anchor4763 = tail call float asm sideeffect "", "=v,0"(float %741) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %742 = insertelement <4 x float> %739, float %740, i64 1, !dbg !44
  %743 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi128), !dbg !41
  %744 = fadd float %743, %k0.sum.anchor4756, !dbg !45
  %k0.sum.anchor4764 = tail call float asm sideeffect "", "=v,0"(float %744) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %745 = insertelement <4 x float> %742, float %743, i64 2, !dbg !44
  %746 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi130), !dbg !41
  %747 = fadd float %746, %k0.sum.anchor4757, !dbg !45
  %k0.sum.anchor4765 = tail call float asm sideeffect "", "=v,0"(float %747) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %748 = insertelement <4 x float> %745, float %746, i64 3, !dbg !44
  %math.result854 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %748) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f05087 = extractelement <4 x float> %math.result854, i64 0, !dbg !44
  %.cvtpack.f15088 = extractelement <4 x float> %math.result854, i64 1, !dbg !44
  %.cvtpack.f25089 = extractelement <4 x float> %math.result854, i64 2, !dbg !44
  %.cvtpack.f35090 = extractelement <4 x float> %math.result854, i64 3, !dbg !44
  %.cvtpack.p05091 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05087, float %.cvtpack.f15088) #12, !dbg !44
  %.cvtpack.p15092 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25089, float %.cvtpack.f35090) #12, !dbg !44
  %.cvtpack.v05093 = insertelement <2 x i32> poison, i32 %.cvtpack.p05091, i64 0, !dbg !44
  %.cvtpack.v15094 = insertelement <2 x i32> %.cvtpack.v05093, i32 %.cvtpack.p15092, i64 1, !dbg !44
  %749 = bitcast <2 x i32> %.cvtpack.v15094 to <4 x half>, !dbg !44
  %math.packed855 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %749) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %750 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %736, <4 x float> %718) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %751 = shufflevector <8 x half> %550, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %752 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %751, <4 x float> %720) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %753 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi132), !dbg !41
  %754 = fadd float %753, %k0.sum.anchor4758, !dbg !45
  %k0.sum.anchor4766 = tail call float asm sideeffect "", "=v,0"(float %754) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %755 = insertelement <4 x float> poison, float %753, i64 0, !dbg !44
  %756 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi134), !dbg !41
  %757 = fadd float %756, %k0.sum.anchor4759, !dbg !45
  %k0.sum.anchor4767 = tail call float asm sideeffect "", "=v,0"(float %757) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %758 = insertelement <4 x float> %755, float %756, i64 1, !dbg !44
  %759 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi136), !dbg !41
  %760 = fadd float %759, %k0.sum.anchor4760, !dbg !45
  %k0.sum.anchor4768 = tail call float asm sideeffect "", "=v,0"(float %760) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %761 = insertelement <4 x float> %758, float %759, i64 2, !dbg !44
  %762 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi138), !dbg !41
  %763 = fadd float %762, %k0.sum.anchor4761, !dbg !45
  %k0.sum.anchor4769 = tail call float asm sideeffect "", "=v,0"(float %763) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %764 = insertelement <4 x float> %761, float %762, i64 3, !dbg !44
  %math.result870 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %764) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f05095 = extractelement <4 x float> %math.result870, i64 0, !dbg !44
  %.cvtpack.f15096 = extractelement <4 x float> %math.result870, i64 1, !dbg !44
  %.cvtpack.f25097 = extractelement <4 x float> %math.result870, i64 2, !dbg !44
  %.cvtpack.f35098 = extractelement <4 x float> %math.result870, i64 3, !dbg !44
  %.cvtpack.p05099 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05095, float %.cvtpack.f15096) #12, !dbg !44
  %.cvtpack.p15100 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25097, float %.cvtpack.f35098) #12, !dbg !44
  %.cvtpack.v05101 = insertelement <2 x i32> poison, i32 %.cvtpack.p05099, i64 0, !dbg !44
  %.cvtpack.v15102 = insertelement <2 x i32> %.cvtpack.v05101, i32 %.cvtpack.p15100, i64 1, !dbg !44
  %765 = bitcast <2 x i32> %.cvtpack.v15102 to <4 x half>, !dbg !44
  %math.packed871 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %765) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %766 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %736, <4 x float> %734) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %767 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %751, <4 x float> %735) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %768 = bitcast <4 x i32> %551 to <8 x half>, !dbg !43
  %769 = shufflevector <8 x half> %768, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %770 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi140), !dbg !41
  %771 = fadd float %770, %k0.sum.anchor4762, !dbg !45
  %k0.sum.anchor4770 = tail call float asm sideeffect "", "=v,0"(float %771) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %772 = insertelement <4 x float> poison, float %770, i64 0, !dbg !44
  %773 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi142), !dbg !41
  %774 = fadd float %773, %k0.sum.anchor4763, !dbg !45
  %k0.sum.anchor4771 = tail call float asm sideeffect "", "=v,0"(float %774) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %775 = fadd float %k0.sum.anchor4770, %k0.sum.anchor4771, !dbg !45
  %k0.sum.anchor4772 = tail call float asm sideeffect "", "=v,0"(float %775) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %776 = insertelement <4 x float> %772, float %773, i64 1, !dbg !44
  %777 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi144), !dbg !41
  %778 = fadd float %777, %k0.sum.anchor4764, !dbg !45
  %k0.sum.anchor4773 = tail call float asm sideeffect "", "=v,0"(float %778) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %779 = insertelement <4 x float> %776, float %777, i64 2, !dbg !44
  %780 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi146), !dbg !41
  %781 = fadd float %780, %k0.sum.anchor4765, !dbg !45
  %k0.sum.anchor4774 = tail call float asm sideeffect "", "=v,0"(float %781) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %782 = fadd float %k0.sum.anchor4773, %k0.sum.anchor4774, !dbg !45
  %k0.sum.anchor4775 = tail call float asm sideeffect "", "=v,0"(float %782) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %783 = fadd float %k0.sum.anchor4772, %k0.sum.anchor4775, !dbg !45
  %k0.sum.anchor4776 = tail call float asm sideeffect "", "=v,0"(float %783) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %784 = fadd float %523, %k0.sum.anchor4776, !dbg !45
  %k0.sum.anchor4777 = tail call float asm sideeffect "", "=v,0"(float %784) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %785 = insertelement <1 x float> poison, float %k0.sum.anchor4777, i64 0, !dbg !13
  %786 = insertvalue [2 x <1 x float>] poison, <1 x float> %785, 0, !dbg !13
  %787 = insertelement <4 x float> %779, float %780, i64 3, !dbg !44
  %math.result856 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %787) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f05103 = extractelement <4 x float> %math.result856, i64 0, !dbg !44
  %.cvtpack.f15104 = extractelement <4 x float> %math.result856, i64 1, !dbg !44
  %.cvtpack.f25105 = extractelement <4 x float> %math.result856, i64 2, !dbg !44
  %.cvtpack.f35106 = extractelement <4 x float> %math.result856, i64 3, !dbg !44
  %.cvtpack.p05107 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05103, float %.cvtpack.f15104) #12, !dbg !44
  %.cvtpack.p15108 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25105, float %.cvtpack.f35106) #12, !dbg !44
  %.cvtpack.v05109 = insertelement <2 x i32> poison, i32 %.cvtpack.p05107, i64 0, !dbg !44
  %.cvtpack.v15110 = insertelement <2 x i32> %.cvtpack.v05109, i32 %.cvtpack.p15108, i64 1, !dbg !44
  %788 = bitcast <2 x i32> %.cvtpack.v15110 to <4 x half>, !dbg !44
  %math.packed857 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %788) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %789 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %769, <4 x float> %750) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %790 = shufflevector <8 x half> %768, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %791 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %790, <4 x float> %752) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %792 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi148), !dbg !41
  %793 = fadd float %792, %k0.sum.anchor4766, !dbg !45
  %k0.sum.anchor4778 = tail call float asm sideeffect "", "=v,0"(float %793) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %794 = insertelement <4 x float> poison, float %792, i64 0, !dbg !44
  %795 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi150), !dbg !41
  %796 = fadd float %795, %k0.sum.anchor4767, !dbg !45
  %k0.sum.anchor4779 = tail call float asm sideeffect "", "=v,0"(float %796) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %797 = fadd float %k0.sum.anchor4778, %k0.sum.anchor4779, !dbg !45
  %k0.sum.anchor4780 = tail call float asm sideeffect "", "=v,0"(float %797) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %798 = insertelement <4 x float> %794, float %795, i64 1, !dbg !44
  %799 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi152), !dbg !41
  %800 = fadd float %799, %k0.sum.anchor4768, !dbg !45
  %k0.sum.anchor4781 = tail call float asm sideeffect "", "=v,0"(float %800) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %801 = insertelement <4 x float> %798, float %799, i64 2, !dbg !44
  %802 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi154), !dbg !41
  %803 = fadd float %802, %k0.sum.anchor4769, !dbg !45
  %k0.sum.anchor4782 = tail call float asm sideeffect "", "=v,0"(float %803) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %804 = fadd float %k0.sum.anchor4781, %k0.sum.anchor4782, !dbg !45
  %k0.sum.anchor4783 = tail call float asm sideeffect "", "=v,0"(float %804) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %805 = fadd float %k0.sum.anchor4780, %k0.sum.anchor4783, !dbg !45
  %k0.sum.anchor4784 = tail call float asm sideeffect "", "=v,0"(float %805) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %806 = fadd float %525, %k0.sum.anchor4784, !dbg !45
  %k0.sum.anchor4785 = tail call float asm sideeffect "", "=v,0"(float %806) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %807 = insertelement <1 x float> poison, float %k0.sum.anchor4785, i64 0, !dbg !13
  %808 = insertvalue [2 x <1 x float>] %786, <1 x float> %807, 1, !dbg !13
  %809 = insertelement <4 x float> %801, float %802, i64 3, !dbg !44
  %math.result872 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %809) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f05111 = extractelement <4 x float> %math.result872, i64 0, !dbg !44
  %.cvtpack.f15112 = extractelement <4 x float> %math.result872, i64 1, !dbg !44
  %.cvtpack.f25113 = extractelement <4 x float> %math.result872, i64 2, !dbg !44
  %.cvtpack.f35114 = extractelement <4 x float> %math.result872, i64 3, !dbg !44
  %.cvtpack.p05115 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f05111, float %.cvtpack.f15112) #12, !dbg !44
  %.cvtpack.p15116 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f25113, float %.cvtpack.f35114) #12, !dbg !44
  %.cvtpack.v05117 = insertelement <2 x i32> poison, i32 %.cvtpack.p05115, i64 0, !dbg !44
  %.cvtpack.v15118 = insertelement <2 x i32> %.cvtpack.v05117, i32 %.cvtpack.p15116, i64 1, !dbg !44
  %810 = bitcast <2 x i32> %.cvtpack.v15118 to <4 x half>, !dbg !44
  %math.packed873 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %810) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43, !frisk.softmax_fragments !5
  %811 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %769, <4 x float> %766) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %812 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %790, <4 x float> %767) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %813 = extractvalue [2 x <8 x float>] %lazy.state4997, 0, !dbg !42
  %814 = shufflevector <8 x float> %813, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  %815 = shufflevector <8 x float> %813, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %816 = extractvalue [2 x <8 x float>] %lazy.state4997, 1, !dbg !42
  %817 = shufflevector <8 x float> %816, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  %818 = shufflevector <8 x float> %816, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %.idx47114713 = or disjoint i64 %address.local3194.idx4700, %214, !dbg !39
  %.idx47114713.tr = trunc nuw nsw i64 %.idx47114713 to i32, !dbg !39
  %dma.buffer.offset4565 = shl nuw nsw i32 %.idx47114713.tr, 1, !dbg !39
  %819 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %326, i32 %dma.buffer.offset4565, <4 x i32> %dma.buffer.rsrc4574) #9, !dbg !39
  %dma.buffer.offset4563 = or disjoint i32 %dma.buffer.offset4565, 32, !dbg !39
  %820 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %328, i32 %dma.buffer.offset4563, <4 x i32> %dma.buffer.rsrc4574) #9, !dbg !39
  %address.local4000 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local3889, !dbg !43
  %821 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4000, i32 0) #11, !dbg !43
  %822 = bitcast <4 x i32> %821 to <8 x half>, !dbg !43
  %address.local4011 = getelementptr i8, ptr addrspace(3) %address.local4000, i64 64, !dbg !43
  %823 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4011, i32 0) #11, !dbg !43
  %824 = bitcast <4 x i32> %823 to <8 x half>, !dbg !43
  %address.local4022 = getelementptr i8, ptr addrspace(3) %address.local4000, i64 128, !dbg !43
  %825 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4022, i32 0) #11, !dbg !43
  %826 = bitcast <4 x i32> %825 to <8 x half>, !dbg !43
  %address.local4033 = getelementptr i8, ptr addrspace(3) %address.local4000, i64 192, !dbg !43
  %827 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4033, i32 0) #11, !dbg !43
  %828 = bitcast <4 x i32> %827 to <8 x half>, !dbg !43
  %address.local4044 = getelementptr i8, ptr addrspace(3) %address.local4000, i64 1024, !dbg !43
  %829 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4044, i32 0) #11, !dbg !43
  %830 = bitcast <4 x i32> %829 to <8 x half>, !dbg !43
  %address.local4055 = getelementptr i8, ptr addrspace(3) %address.local4000, i64 1088, !dbg !43
  %831 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4055, i32 0) #11, !dbg !43
  %832 = bitcast <4 x i32> %831 to <8 x half>, !dbg !43
  %address.local4066 = getelementptr i8, ptr addrspace(3) %address.local4000, i64 1152, !dbg !43
  %833 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4066, i32 0) #11, !dbg !43
  %834 = bitcast <4 x i32> %833 to <8 x half>, !dbg !43
  %address.local4077 = getelementptr i8, ptr addrspace(3) %address.local4000, i64 1216, !dbg !43
  %835 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4077, i32 0) #11, !dbg !43
  %836 = shufflevector <8 x half> %822, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %837 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %836, <4 x float> %814) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %838 = shufflevector <8 x half> %822, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %839 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %838, <4 x float> %815) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %840 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %836, <4 x float> %817) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %841 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %838, <4 x float> %818) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %842 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %843 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %842, <4 x float> %837) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %844 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %845 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %844, <4 x float> %839) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %846 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %842, <4 x float> %840) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %847 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %844, <4 x float> %841) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %848 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %849 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %848, <4 x float> %843) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %850 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %850, <4 x float> %845) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %852 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %848, <4 x float> %846) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %853 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %850, <4 x float> %847) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %854 = shufflevector <8 x half> %828, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %855 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %854, <4 x float> %849) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %856 = shufflevector <8 x half> %828, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %857 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %856, <4 x float> %851) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %858 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %854, <4 x float> %852) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %859 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %856, <4 x float> %853) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %860 = shufflevector <8 x half> %830, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %861 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %860, <4 x float> %855) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %862 = shufflevector <8 x half> %830, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %863 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %862, <4 x float> %857) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %864 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %860, <4 x float> %858) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %865 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %862, <4 x float> %859) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %866 = shufflevector <8 x half> %832, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %867 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %866, <4 x float> %861) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %868 = shufflevector <8 x half> %832, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %869 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %868, <4 x float> %863) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %870 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %866, <4 x float> %864) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %868, <4 x float> %865) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %872 = shufflevector <8 x half> %834, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %872, <4 x float> %867) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %874 = shufflevector <8 x half> %834, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %874, <4 x float> %869) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %876 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %872, <4 x float> %870) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %877 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %874, <4 x float> %871) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %878 = bitcast <4 x i32> %835 to <8 x half>, !dbg !43
  %879 = shufflevector <8 x half> %878, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %880 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %879, <4 x float> %873) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %881 = shufflevector <8 x half> %878, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %882 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %881, <4 x float> %875) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %883 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %879, <4 x float> %876) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %881, <4 x float> %877) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %885 = extractvalue [2 x <8 x float>] %lazy.state4996, 0, !dbg !42
  %886 = shufflevector <8 x float> %885, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  %887 = shufflevector <8 x float> %885, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %888 = extractvalue [2 x <8 x float>] %lazy.state4996, 1, !dbg !42
  %889 = shufflevector <8 x float> %888, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  %890 = shufflevector <8 x float> %888, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %.idx47174719 = or disjoint i64 %address.local3194.idx4700, %270, !dbg !39
  %.idx47174719.tr = trunc nuw nsw i64 %.idx47174719 to i32, !dbg !39
  %dma.buffer.offset4561 = shl nuw nsw i32 %.idx47174719.tr, 1, !dbg !39
  %891 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %532, i32 %dma.buffer.offset4561, <4 x i32> %dma.buffer.rsrc4574) #9, !dbg !39
  %dma.buffer.offset = or disjoint i32 %dma.buffer.offset4561, 32, !dbg !39
  %892 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %534, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc4574) #9, !dbg !39
  %893 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3890, i32 0) #11, !dbg !43
  %894 = bitcast <4 x i32> %893 to <8 x half>, !dbg !43
  %address.local4119 = or disjoint i32 %address.local3888, 32, !dbg !43
  %address.local4120 = zext nneg i32 %address.local4119 to i64, !dbg !43
  %address.local4121 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4120, !dbg !43
  %895 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4121, i32 0) #11, !dbg !43
  %896 = bitcast <4 x i32> %895 to <8 x half>, !dbg !43
  %address.local4130 = or disjoint i32 %address.local3888, 64, !dbg !43
  %address.local4131 = zext nneg i32 %address.local4130 to i64, !dbg !43
  %address.local4132 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4131, !dbg !43
  %897 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4132, i32 0) #11, !dbg !43
  %898 = bitcast <4 x i32> %897 to <8 x half>, !dbg !43
  %address.local4141 = or disjoint i32 %address.local3888, 96, !dbg !43
  %address.local4142 = zext nneg i32 %address.local4141 to i64, !dbg !43
  %address.local4143 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4142, !dbg !43
  %899 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4143, i32 0) #11, !dbg !43
  %900 = bitcast <4 x i32> %899 to <8 x half>, !dbg !43
  %address.local4152 = or disjoint i32 %address.local3888, 512, !dbg !43
  %address.local4153 = zext nneg i32 %address.local4152 to i64, !dbg !43
  %address.local4154 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4153, !dbg !43
  %901 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4154, i32 0) #11, !dbg !43
  %902 = bitcast <4 x i32> %901 to <8 x half>, !dbg !43
  %address.local4163 = or disjoint i32 %address.local3888, 544, !dbg !43
  %address.local4164 = zext nneg i32 %address.local4163 to i64, !dbg !43
  %address.local4165 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4164, !dbg !43
  %903 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4165, i32 0) #11, !dbg !43
  %904 = bitcast <4 x i32> %903 to <8 x half>, !dbg !43
  %address.local4174 = or disjoint i32 %address.local3888, 576, !dbg !43
  %address.local4175 = zext nneg i32 %address.local4174 to i64, !dbg !43
  %address.local4176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4175, !dbg !43
  %905 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4176, i32 0) #11, !dbg !43
  %906 = bitcast <4 x i32> %905 to <8 x half>, !dbg !43
  %address.local4185 = or disjoint i32 %address.local3888, 608, !dbg !43
  %address.local4186 = zext nneg i32 %address.local4185 to i64, !dbg !43
  %address.local4187 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4186, !dbg !43
  %907 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4187, i32 0) #11, !dbg !43
  %908 = shufflevector <8 x half> %894, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %909 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %908, <4 x float> %886) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %910 = shufflevector <8 x half> %894, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %911 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %910, <4 x float> %887) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %908, <4 x float> %889) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %913 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %910, <4 x float> %890) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %914 = shufflevector <8 x half> %896, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %915 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %914, <4 x float> %909) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %916 = shufflevector <8 x half> %896, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %917 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %916, <4 x float> %911) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %914, <4 x float> %912) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %919 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %916, <4 x float> %913) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %920 = shufflevector <8 x half> %898, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %921 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %920, <4 x float> %915) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %922 = shufflevector <8 x half> %898, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %923 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %922, <4 x float> %917) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %920, <4 x float> %918) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %925 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %922, <4 x float> %919) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %926 = shufflevector <8 x half> %900, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %927 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %926, <4 x float> %921) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %928 = shufflevector <8 x half> %900, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %929 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %928, <4 x float> %923) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %926, <4 x float> %924) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %931 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %928, <4 x float> %925) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %932 = shufflevector <8 x half> %902, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %933 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %932, <4 x float> %927) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %934 = shufflevector <8 x half> %902, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %935 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %934, <4 x float> %929) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %936 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %932, <4 x float> %930) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %937 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %934, <4 x float> %931) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %938 = shufflevector <8 x half> %904, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %939 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %938, <4 x float> %933) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %940 = shufflevector <8 x half> %904, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %941 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %940, <4 x float> %935) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %942 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %938, <4 x float> %936) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %943 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %940, <4 x float> %937) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %944 = shufflevector <8 x half> %906, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %945 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %944, <4 x float> %939) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %946 = shufflevector <8 x half> %906, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %947 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %946, <4 x float> %941) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %948 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %944, <4 x float> %942) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %949 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %946, <4 x float> %943) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %950 = bitcast <4 x i32> %907 to <8 x half>, !dbg !43
  %951 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %952 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %951, <4 x float> %945) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %953 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %954 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %953, <4 x float> %947) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %955 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %951, <4 x float> %948) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %956 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %953, <4 x float> %949) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %957 = extractvalue [2 x <8 x float>] %lazy.state4995, 0, !dbg !42
  %958 = shufflevector <8 x float> %957, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  %959 = shufflevector <8 x float> %957, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %960 = extractvalue [2 x <8 x float>] %lazy.state4995, 1, !dbg !42
  %961 = shufflevector <8 x float> %960, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  %962 = shufflevector <8 x float> %960, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %963 = add nuw nsw i64 %122, 1, !dbg !13
  %k0.has.next.not = icmp eq i64 %122, 31, !dbg !43
  br i1 %k0.has.next.not, label %k0.join, label %k0.issue, !dbg !43

k0.issue:                                         ; preds = %lazy.join
  %k0.next. = shl nuw nsw i64 %963, 14, !dbg !12
  %k0.next.4871 = add nuw nsw i64 %k0.next., %k0.prime.4787, !dbg !12
  %k0.next..idx46434646 = add nsw i64 %k0.next.4871, %k0.prime.4810, !dbg !12
  %k0.next..idx46434646.tr = trunc nuw nsw i64 %k0.next..idx46434646 to i32, !dbg !12
  %k0.next.dma.buffer.offset4621 = shl nuw nsw i32 %k0.next..idx46434646.tr, 1, !dbg !12
  %k0.next.4872 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %139, i32 %k0.next.dma.buffer.offset4621, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.next..idx4647 = shl nuw nsw i64 %k0.next.4871, 1, !dbg !12
  %k0.next.4873 = add nsw i64 %k0.next..idx4647, %k0.prime.4833, !dbg !12
  %k0.next.dma.buffer.offset4619 = trunc nuw nsw i64 %k0.next.4873 to i32, !dbg !12
  %k0.next.4874 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %141, i32 %k0.next.dma.buffer.offset4619, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.next.4875 = add nsw i64 %k0.next..idx4647, %k0.prime.4852, !dbg !12
  %k0.next.dma.buffer.offset4617 = trunc nuw nsw i64 %k0.next.4875 to i32, !dbg !12
  %k0.next.4876 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %143, i32 %k0.next.dma.buffer.offset4617, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.next..idx46534656 = add nsw i64 %k0.next.4871, %k0.prime.4862, !dbg !12
  %k0.next..idx46534656.tr = trunc nuw nsw i64 %k0.next..idx46534656 to i32, !dbg !12
  %k0.next.4877 = shl nuw nsw i32 %k0.next..idx46534656.tr, 1, !dbg !12
  %k0.next.dma.buffer.offset4615 = add nuw nsw i32 %k0.prime.4864, %k0.next.4877, !dbg !12
  %k0.next.4879 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %145, i32 %k0.next.dma.buffer.offset4615, <4 x i32> %k0.prime.dma.buffer.rsrc4622) #9, !dbg !12, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %lazy.join
  %964 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4000, i32 0) #11, !dbg !43
  %965 = bitcast <4 x i32> %964 to <8 x half>, !dbg !43
  %address.local4189 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4120, !dbg !43
  %966 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4189, i32 0) #11, !dbg !43
  %967 = bitcast <4 x i32> %966 to <8 x half>, !dbg !43
  %address.local4190 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4131, !dbg !43
  %968 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4190, i32 0) #11, !dbg !43
  %969 = bitcast <4 x i32> %968 to <8 x half>, !dbg !43
  %address.local4191 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4142, !dbg !43
  %970 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4191, i32 0) #11, !dbg !43
  %971 = bitcast <4 x i32> %970 to <8 x half>, !dbg !43
  %address.local4192 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4153, !dbg !43
  %972 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4192, i32 0) #11, !dbg !43
  %973 = bitcast <4 x i32> %972 to <8 x half>, !dbg !43
  %address.local4193 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4164, !dbg !43
  %974 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4193, i32 0) #11, !dbg !43
  %975 = bitcast <4 x i32> %974 to <8 x half>, !dbg !43
  %address.local4194 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4175, !dbg !43
  %976 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4194, i32 0) #11, !dbg !43
  %977 = bitcast <4 x i32> %976 to <8 x half>, !dbg !43
  %address.local4195 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4186, !dbg !43
  %978 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4195, i32 0) #11, !dbg !43
  %979 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %980 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %979, <4 x float> %958) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %981 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %981, <4 x float> %959) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %983 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %979, <4 x float> %961) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %984 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed859, <4 x half> %981, <4 x float> %962) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %985 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %986 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %985, <4 x float> %980) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %987 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed845, <4 x half> %987, <4 x float> %982) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %989 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %985, <4 x float> %983) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %990 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed861, <4 x half> %987, <4 x float> %984) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %991 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %992 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %991, <4 x float> %986) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %993 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed847, <4 x half> %993, <4 x float> %988) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %995 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %991, <4 x float> %989) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %996 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed863, <4 x half> %993, <4 x float> %990) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %997 = shufflevector <8 x half> %971, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %997, <4 x float> %992) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %999 = shufflevector <8 x half> %971, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed849, <4 x half> %999, <4 x float> %994) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1001 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %997, <4 x float> %995) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed865, <4 x half> %999, <4 x float> %996) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1003 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %1003, <4 x float> %998) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1005 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed851, <4 x half> %1005, <4 x float> %1000) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1007 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %1003, <4 x float> %1001) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed867, <4 x half> %1005, <4 x float> %1002) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1009 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1010 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %1009, <4 x float> %1004) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1011 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed853, <4 x half> %1011, <4 x float> %1006) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1013 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %1009, <4 x float> %1007) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1014 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed869, <4 x half> %1011, <4 x float> %1008) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1015 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1016 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %1015, <4 x float> %1010) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1017 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1018 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed855, <4 x half> %1017, <4 x float> %1012) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1019 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %1015, <4 x float> %1013) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1020 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed871, <4 x half> %1017, <4 x float> %1014) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1021 = bitcast <4 x i32> %978 to <8 x half>, !dbg !43
  %1022 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1023 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %1022, <4 x float> %1016) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1024 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1025 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed857, <4 x half> %1024, <4 x float> %1018) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1026 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %1022, <4 x float> %1019) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1027 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed873, <4 x half> %1024, <4 x float> %1020) #8, !dbg !43, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13
  %1028 = shufflevector <4 x float> %789, <4 x float> %791, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1029 = insertvalue [2 x <8 x float>] poison, <8 x float> %1028, 0, !dbg !13
  %1030 = shufflevector <4 x float> %811, <4 x float> %812, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1031 = insertvalue [2 x <8 x float>] %1029, <8 x float> %1030, 1, !dbg !13
  %1032 = shufflevector <4 x float> %880, <4 x float> %882, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1033 = insertvalue [2 x <8 x float>] poison, <8 x float> %1032, 0, !dbg !13
  %1034 = shufflevector <4 x float> %883, <4 x float> %884, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1035 = insertvalue [2 x <8 x float>] %1033, <8 x float> %1034, 1, !dbg !13
  %1036 = shufflevector <4 x float> %952, <4 x float> %954, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1037 = insertvalue [2 x <8 x float>] poison, <8 x float> %1036, 0, !dbg !13
  %1038 = shufflevector <4 x float> %955, <4 x float> %956, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1039 = insertvalue [2 x <8 x float>] %1037, <8 x float> %1038, 1, !dbg !13
  %1040 = shufflevector <4 x float> %1023, <4 x float> %1025, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1041 = insertvalue [2 x <8 x float>] poison, <8 x float> %1040, 0, !dbg !13
  %1042 = shufflevector <4 x float> %1026, <4 x float> %1027, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1043 = insertvalue [2 x <8 x float>] %1041, <8 x float> %1042, 1, !dbg !13
  %1044 = icmp samesign ult i64 %122, 31, !dbg !13
  br i1 %1044, label %115, label %1045, !dbg !13

1045:                                             ; preds = %k0.join
  %1046 = bitcast float %k0.sum.anchor4785 to i32, !dbg !13
  %1047 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1046), !dbg !13
  %1048 = bitcast float %k0.sum.anchor4777 to i32, !dbg !13
  %1049 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1048), !dbg !13
  %1050 = and i32 %34, 16, !dbg !13
  %.not = icmp eq i32 %1050, 0, !dbg !13
  %1051 = and i32 %34, 32, !dbg !13
  %.not84 = icmp eq i32 %1051, 0, !dbg !13
  %1052 = extractelement <4 x float> %1023, i64 0, !dbg !13
  %1053 = extractelement <4 x float> %1023, i64 1, !dbg !13
  %1054 = extractelement <4 x float> %1023, i64 2, !dbg !13
  %1055 = extractelement <4 x float> %1023, i64 3, !dbg !13
  %1056 = select i1 %.not, float %1053, float %1052, !dbg !13
  %1057 = bitcast float %1056 to i32, !dbg !13
  %1058 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1057), !dbg !13
  %1059 = bitcast i32 %1058 to float, !dbg !13
  %1060 = select i1 %.not, float %1052, float %1059, !dbg !13
  %1061 = select i1 %.not, float %1059, float %1053, !dbg !13
  %1062 = select i1 %.not, float %1055, float %1054, !dbg !13
  %1063 = bitcast float %1062 to i32, !dbg !13
  %1064 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1063), !dbg !13
  %1065 = bitcast i32 %1064 to float, !dbg !13
  %1066 = select i1 %.not, float %1054, float %1065, !dbg !13
  %1067 = select i1 %.not, float %1065, float %1055, !dbg !13
  %1068 = select i1 %.not84, float %1066, float %1060, !dbg !13
  %1069 = bitcast float %1068 to i32, !dbg !13
  %1070 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1069), !dbg !13
  %1071 = bitcast i32 %1070 to float, !dbg !13
  %1072 = select i1 %.not84, float %1067, float %1061, !dbg !13
  %1073 = bitcast float %1072 to i32, !dbg !13
  %1074 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1073), !dbg !13
  %1075 = bitcast i32 %1074 to float, !dbg !13
  %1076 = insertelement <4 x i1> poison, i1 %.not84, i64 0, !dbg !13
  %1077 = shufflevector <4 x i1> %1076, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !13
  %1078 = insertelement <4 x float> poison, float %1060, i64 0, !dbg !13
  %1079 = insertelement <4 x float> %1078, float %1061, i64 1, !dbg !13
  %1080 = insertelement <4 x float> %1079, float %1071, i64 2, !dbg !13
  %1081 = insertelement <4 x float> %1080, float %1075, i64 3, !dbg !13
  %1082 = insertelement <4 x float> poison, float %1071, i64 0, !dbg !13
  %1083 = insertelement <4 x float> %1082, float %1075, i64 1, !dbg !13
  %1084 = insertelement <4 x float> %1083, float %1066, i64 2, !dbg !13
  %1085 = insertelement <4 x float> %1084, float %1067, i64 3, !dbg !13
  %1086 = select <4 x i1> %1077, <4 x float> %1081, <4 x float> %1085, !dbg !13
  %1087 = extractelement <4 x float> %1025, i64 0, !dbg !13
  %1088 = extractelement <4 x float> %1025, i64 1, !dbg !13
  %1089 = extractelement <4 x float> %1025, i64 2, !dbg !13
  %1090 = extractelement <4 x float> %1025, i64 3, !dbg !13
  %1091 = select i1 %.not, float %1088, float %1087, !dbg !13
  %1092 = bitcast float %1091 to i32, !dbg !13
  %1093 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1092), !dbg !13
  %1094 = bitcast i32 %1093 to float, !dbg !13
  %1095 = select i1 %.not, float %1087, float %1094, !dbg !13
  %1096 = select i1 %.not, float %1094, float %1088, !dbg !13
  %1097 = select i1 %.not, float %1090, float %1089, !dbg !13
  %1098 = bitcast float %1097 to i32, !dbg !13
  %1099 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1098), !dbg !13
  %1100 = bitcast i32 %1099 to float, !dbg !13
  %1101 = select i1 %.not, float %1089, float %1100, !dbg !13
  %1102 = select i1 %.not, float %1100, float %1090, !dbg !13
  %1103 = select i1 %.not84, float %1101, float %1095, !dbg !13
  %1104 = bitcast float %1103 to i32, !dbg !13
  %1105 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1104), !dbg !13
  %1106 = bitcast i32 %1105 to float, !dbg !13
  %1107 = select i1 %.not84, float %1102, float %1096, !dbg !13
  %1108 = bitcast float %1107 to i32, !dbg !13
  %1109 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1108), !dbg !13
  %1110 = bitcast i32 %1109 to float, !dbg !13
  %1111 = insertelement <4 x float> poison, float %1095, i64 0, !dbg !13
  %1112 = insertelement <4 x float> %1111, float %1096, i64 1, !dbg !13
  %1113 = insertelement <4 x float> %1112, float %1106, i64 2, !dbg !13
  %1114 = insertelement <4 x float> %1113, float %1110, i64 3, !dbg !13
  %1115 = insertelement <4 x float> poison, float %1106, i64 0, !dbg !13
  %1116 = insertelement <4 x float> %1115, float %1110, i64 1, !dbg !13
  %1117 = insertelement <4 x float> %1116, float %1101, i64 2, !dbg !13
  %1118 = insertelement <4 x float> %1117, float %1102, i64 3, !dbg !13
  %1119 = select <4 x i1> %1077, <4 x float> %1114, <4 x float> %1118, !dbg !13
  %1120 = extractelement <4 x float> %1026, i64 0, !dbg !13
  %1121 = extractelement <4 x float> %1026, i64 1, !dbg !13
  %1122 = extractelement <4 x float> %1026, i64 2, !dbg !13
  %1123 = extractelement <4 x float> %1026, i64 3, !dbg !13
  %1124 = select i1 %.not, float %1121, float %1120, !dbg !13
  %1125 = bitcast float %1124 to i32, !dbg !13
  %1126 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1125), !dbg !13
  %1127 = bitcast i32 %1126 to float, !dbg !13
  %1128 = select i1 %.not, float %1120, float %1127, !dbg !13
  %1129 = select i1 %.not, float %1127, float %1121, !dbg !13
  %1130 = select i1 %.not, float %1123, float %1122, !dbg !13
  %1131 = bitcast float %1130 to i32, !dbg !13
  %1132 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1131), !dbg !13
  %1133 = bitcast i32 %1132 to float, !dbg !13
  %1134 = select i1 %.not, float %1122, float %1133, !dbg !13
  %1135 = select i1 %.not, float %1133, float %1123, !dbg !13
  %1136 = select i1 %.not84, float %1134, float %1128, !dbg !13
  %1137 = bitcast float %1136 to i32, !dbg !13
  %1138 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1137), !dbg !13
  %1139 = bitcast i32 %1138 to float, !dbg !13
  %1140 = select i1 %.not84, float %1135, float %1129, !dbg !13
  %1141 = bitcast float %1140 to i32, !dbg !13
  %1142 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1141), !dbg !13
  %1143 = bitcast i32 %1142 to float, !dbg !13
  %1144 = insertelement <4 x float> poison, float %1128, i64 0, !dbg !13
  %1145 = insertelement <4 x float> %1144, float %1129, i64 1, !dbg !13
  %1146 = insertelement <4 x float> %1145, float %1139, i64 2, !dbg !13
  %1147 = insertelement <4 x float> %1146, float %1143, i64 3, !dbg !13
  %1148 = insertelement <4 x float> poison, float %1139, i64 0, !dbg !13
  %1149 = insertelement <4 x float> %1148, float %1143, i64 1, !dbg !13
  %1150 = insertelement <4 x float> %1149, float %1134, i64 2, !dbg !13
  %1151 = insertelement <4 x float> %1150, float %1135, i64 3, !dbg !13
  %1152 = select <4 x i1> %1077, <4 x float> %1147, <4 x float> %1151, !dbg !13
  %1153 = extractelement <4 x float> %1027, i64 0, !dbg !13
  %1154 = extractelement <4 x float> %1027, i64 1, !dbg !13
  %1155 = extractelement <4 x float> %1027, i64 2, !dbg !13
  %1156 = extractelement <4 x float> %1027, i64 3, !dbg !13
  %1157 = select i1 %.not, float %1154, float %1153, !dbg !13
  %1158 = bitcast float %1157 to i32, !dbg !13
  %1159 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1158), !dbg !13
  %1160 = bitcast i32 %1159 to float, !dbg !13
  %1161 = select i1 %.not, float %1153, float %1160, !dbg !13
  %1162 = select i1 %.not, float %1160, float %1154, !dbg !13
  %1163 = select i1 %.not, float %1156, float %1155, !dbg !13
  %1164 = bitcast float %1163 to i32, !dbg !13
  %1165 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1164), !dbg !13
  %1166 = bitcast i32 %1165 to float, !dbg !13
  %1167 = select i1 %.not, float %1155, float %1166, !dbg !13
  %1168 = select i1 %.not, float %1166, float %1156, !dbg !13
  %1169 = select i1 %.not84, float %1167, float %1161, !dbg !13
  %1170 = bitcast float %1169 to i32, !dbg !13
  %1171 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1170), !dbg !13
  %1172 = bitcast i32 %1171 to float, !dbg !13
  %1173 = select i1 %.not84, float %1168, float %1162, !dbg !13
  %1174 = bitcast float %1173 to i32, !dbg !13
  %1175 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1174), !dbg !13
  %1176 = bitcast i32 %1175 to float, !dbg !13
  %1177 = insertelement <4 x float> poison, float %1161, i64 0, !dbg !13
  %1178 = insertelement <4 x float> %1177, float %1162, i64 1, !dbg !13
  %1179 = insertelement <4 x float> %1178, float %1172, i64 2, !dbg !13
  %1180 = insertelement <4 x float> %1179, float %1176, i64 3, !dbg !13
  %1181 = insertelement <4 x float> poison, float %1172, i64 0, !dbg !13
  %1182 = insertelement <4 x float> %1181, float %1176, i64 1, !dbg !13
  %1183 = insertelement <4 x float> %1182, float %1167, i64 2, !dbg !13
  %1184 = insertelement <4 x float> %1183, float %1168, i64 3, !dbg !13
  %1185 = select <4 x i1> %1077, <4 x float> %1180, <4 x float> %1184, !dbg !13
  %1186 = extractelement <4 x float> %952, i64 0, !dbg !13
  %1187 = extractelement <4 x float> %952, i64 1, !dbg !13
  %1188 = extractelement <4 x float> %952, i64 2, !dbg !13
  %1189 = extractelement <4 x float> %952, i64 3, !dbg !13
  %1190 = select i1 %.not, float %1187, float %1186, !dbg !13
  %1191 = bitcast float %1190 to i32, !dbg !13
  %1192 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1191), !dbg !13
  %1193 = bitcast i32 %1192 to float, !dbg !13
  %1194 = select i1 %.not, float %1186, float %1193, !dbg !13
  %1195 = select i1 %.not, float %1193, float %1187, !dbg !13
  %1196 = select i1 %.not, float %1189, float %1188, !dbg !13
  %1197 = bitcast float %1196 to i32, !dbg !13
  %1198 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1197), !dbg !13
  %1199 = bitcast i32 %1198 to float, !dbg !13
  %1200 = select i1 %.not, float %1188, float %1199, !dbg !13
  %1201 = select i1 %.not, float %1199, float %1189, !dbg !13
  %1202 = select i1 %.not84, float %1200, float %1194, !dbg !13
  %1203 = bitcast float %1202 to i32, !dbg !13
  %1204 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1203), !dbg !13
  %1205 = bitcast i32 %1204 to float, !dbg !13
  %1206 = select i1 %.not84, float %1201, float %1195, !dbg !13
  %1207 = bitcast float %1206 to i32, !dbg !13
  %1208 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1207), !dbg !13
  %1209 = bitcast i32 %1208 to float, !dbg !13
  %1210 = insertelement <4 x float> poison, float %1194, i64 0, !dbg !13
  %1211 = insertelement <4 x float> %1210, float %1195, i64 1, !dbg !13
  %1212 = insertelement <4 x float> %1211, float %1205, i64 2, !dbg !13
  %1213 = insertelement <4 x float> %1212, float %1209, i64 3, !dbg !13
  %1214 = insertelement <4 x float> poison, float %1205, i64 0, !dbg !13
  %1215 = insertelement <4 x float> %1214, float %1209, i64 1, !dbg !13
  %1216 = insertelement <4 x float> %1215, float %1200, i64 2, !dbg !13
  %1217 = insertelement <4 x float> %1216, float %1201, i64 3, !dbg !13
  %1218 = select <4 x i1> %1077, <4 x float> %1213, <4 x float> %1217, !dbg !13
  %1219 = extractelement <4 x float> %954, i64 0, !dbg !13
  %1220 = extractelement <4 x float> %954, i64 1, !dbg !13
  %1221 = extractelement <4 x float> %954, i64 2, !dbg !13
  %1222 = extractelement <4 x float> %954, i64 3, !dbg !13
  %1223 = select i1 %.not, float %1220, float %1219, !dbg !13
  %1224 = bitcast float %1223 to i32, !dbg !13
  %1225 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1224), !dbg !13
  %1226 = bitcast i32 %1225 to float, !dbg !13
  %1227 = select i1 %.not, float %1219, float %1226, !dbg !13
  %1228 = select i1 %.not, float %1226, float %1220, !dbg !13
  %1229 = select i1 %.not, float %1222, float %1221, !dbg !13
  %1230 = bitcast float %1229 to i32, !dbg !13
  %1231 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1230), !dbg !13
  %1232 = bitcast i32 %1231 to float, !dbg !13
  %1233 = select i1 %.not, float %1221, float %1232, !dbg !13
  %1234 = select i1 %.not, float %1232, float %1222, !dbg !13
  %1235 = select i1 %.not84, float %1233, float %1227, !dbg !13
  %1236 = bitcast float %1235 to i32, !dbg !13
  %1237 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1236), !dbg !13
  %1238 = bitcast i32 %1237 to float, !dbg !13
  %1239 = select i1 %.not84, float %1234, float %1228, !dbg !13
  %1240 = bitcast float %1239 to i32, !dbg !13
  %1241 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1240), !dbg !13
  %1242 = bitcast i32 %1241 to float, !dbg !13
  %1243 = insertelement <4 x float> poison, float %1227, i64 0, !dbg !13
  %1244 = insertelement <4 x float> %1243, float %1228, i64 1, !dbg !13
  %1245 = insertelement <4 x float> %1244, float %1238, i64 2, !dbg !13
  %1246 = insertelement <4 x float> %1245, float %1242, i64 3, !dbg !13
  %1247 = insertelement <4 x float> poison, float %1238, i64 0, !dbg !13
  %1248 = insertelement <4 x float> %1247, float %1242, i64 1, !dbg !13
  %1249 = insertelement <4 x float> %1248, float %1233, i64 2, !dbg !13
  %1250 = insertelement <4 x float> %1249, float %1234, i64 3, !dbg !13
  %1251 = select <4 x i1> %1077, <4 x float> %1246, <4 x float> %1250, !dbg !13
  %1252 = extractelement <4 x float> %955, i64 0, !dbg !13
  %1253 = extractelement <4 x float> %955, i64 1, !dbg !13
  %1254 = extractelement <4 x float> %955, i64 2, !dbg !13
  %1255 = extractelement <4 x float> %955, i64 3, !dbg !13
  %1256 = select i1 %.not, float %1253, float %1252, !dbg !13
  %1257 = bitcast float %1256 to i32, !dbg !13
  %1258 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1257), !dbg !13
  %1259 = bitcast i32 %1258 to float, !dbg !13
  %1260 = select i1 %.not, float %1252, float %1259, !dbg !13
  %1261 = select i1 %.not, float %1259, float %1253, !dbg !13
  %1262 = select i1 %.not, float %1255, float %1254, !dbg !13
  %1263 = bitcast float %1262 to i32, !dbg !13
  %1264 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1263), !dbg !13
  %1265 = bitcast i32 %1264 to float, !dbg !13
  %1266 = select i1 %.not, float %1254, float %1265, !dbg !13
  %1267 = select i1 %.not, float %1265, float %1255, !dbg !13
  %1268 = select i1 %.not84, float %1266, float %1260, !dbg !13
  %1269 = bitcast float %1268 to i32, !dbg !13
  %1270 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1269), !dbg !13
  %1271 = bitcast i32 %1270 to float, !dbg !13
  %1272 = select i1 %.not84, float %1267, float %1261, !dbg !13
  %1273 = bitcast float %1272 to i32, !dbg !13
  %1274 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1273), !dbg !13
  %1275 = bitcast i32 %1274 to float, !dbg !13
  %1276 = insertelement <4 x float> poison, float %1260, i64 0, !dbg !13
  %1277 = insertelement <4 x float> %1276, float %1261, i64 1, !dbg !13
  %1278 = insertelement <4 x float> %1277, float %1271, i64 2, !dbg !13
  %1279 = insertelement <4 x float> %1278, float %1275, i64 3, !dbg !13
  %1280 = insertelement <4 x float> poison, float %1271, i64 0, !dbg !13
  %1281 = insertelement <4 x float> %1280, float %1275, i64 1, !dbg !13
  %1282 = insertelement <4 x float> %1281, float %1266, i64 2, !dbg !13
  %1283 = insertelement <4 x float> %1282, float %1267, i64 3, !dbg !13
  %1284 = select <4 x i1> %1077, <4 x float> %1279, <4 x float> %1283, !dbg !13
  %1285 = extractelement <4 x float> %956, i64 0, !dbg !13
  %1286 = extractelement <4 x float> %956, i64 1, !dbg !13
  %1287 = extractelement <4 x float> %956, i64 2, !dbg !13
  %1288 = extractelement <4 x float> %956, i64 3, !dbg !13
  %1289 = select i1 %.not, float %1286, float %1285, !dbg !13
  %1290 = bitcast float %1289 to i32, !dbg !13
  %1291 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1290), !dbg !13
  %1292 = bitcast i32 %1291 to float, !dbg !13
  %1293 = select i1 %.not, float %1285, float %1292, !dbg !13
  %1294 = select i1 %.not, float %1292, float %1286, !dbg !13
  %1295 = select i1 %.not, float %1288, float %1287, !dbg !13
  %1296 = bitcast float %1295 to i32, !dbg !13
  %1297 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1296), !dbg !13
  %1298 = bitcast i32 %1297 to float, !dbg !13
  %1299 = select i1 %.not, float %1287, float %1298, !dbg !13
  %1300 = select i1 %.not, float %1298, float %1288, !dbg !13
  %1301 = select i1 %.not84, float %1299, float %1293, !dbg !13
  %1302 = bitcast float %1301 to i32, !dbg !13
  %1303 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1302), !dbg !13
  %1304 = bitcast i32 %1303 to float, !dbg !13
  %1305 = select i1 %.not84, float %1300, float %1294, !dbg !13
  %1306 = bitcast float %1305 to i32, !dbg !13
  %1307 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1306), !dbg !13
  %1308 = bitcast i32 %1307 to float, !dbg !13
  %1309 = insertelement <4 x float> poison, float %1293, i64 0, !dbg !13
  %1310 = insertelement <4 x float> %1309, float %1294, i64 1, !dbg !13
  %1311 = insertelement <4 x float> %1310, float %1304, i64 2, !dbg !13
  %1312 = insertelement <4 x float> %1311, float %1308, i64 3, !dbg !13
  %1313 = insertelement <4 x float> poison, float %1304, i64 0, !dbg !13
  %1314 = insertelement <4 x float> %1313, float %1308, i64 1, !dbg !13
  %1315 = insertelement <4 x float> %1314, float %1299, i64 2, !dbg !13
  %1316 = insertelement <4 x float> %1315, float %1300, i64 3, !dbg !13
  %1317 = select <4 x i1> %1077, <4 x float> %1312, <4 x float> %1316, !dbg !13
  %1318 = extractelement <4 x float> %880, i64 0, !dbg !13
  %1319 = extractelement <4 x float> %880, i64 1, !dbg !13
  %1320 = extractelement <4 x float> %880, i64 2, !dbg !13
  %1321 = extractelement <4 x float> %880, i64 3, !dbg !13
  %1322 = select i1 %.not, float %1319, float %1318, !dbg !13
  %1323 = bitcast float %1322 to i32, !dbg !13
  %1324 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1323), !dbg !13
  %1325 = bitcast i32 %1324 to float, !dbg !13
  %1326 = select i1 %.not, float %1318, float %1325, !dbg !13
  %1327 = select i1 %.not, float %1325, float %1319, !dbg !13
  %1328 = select i1 %.not, float %1321, float %1320, !dbg !13
  %1329 = bitcast float %1328 to i32, !dbg !13
  %1330 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1329), !dbg !13
  %1331 = bitcast i32 %1330 to float, !dbg !13
  %1332 = select i1 %.not, float %1320, float %1331, !dbg !13
  %1333 = select i1 %.not, float %1331, float %1321, !dbg !13
  %1334 = select i1 %.not84, float %1332, float %1326, !dbg !13
  %1335 = bitcast float %1334 to i32, !dbg !13
  %1336 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1335), !dbg !13
  %1337 = bitcast i32 %1336 to float, !dbg !13
  %1338 = select i1 %.not84, float %1333, float %1327, !dbg !13
  %1339 = bitcast float %1338 to i32, !dbg !13
  %1340 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1339), !dbg !13
  %1341 = bitcast i32 %1340 to float, !dbg !13
  %1342 = insertelement <4 x float> poison, float %1326, i64 0, !dbg !13
  %1343 = insertelement <4 x float> %1342, float %1327, i64 1, !dbg !13
  %1344 = insertelement <4 x float> %1343, float %1337, i64 2, !dbg !13
  %1345 = insertelement <4 x float> %1344, float %1341, i64 3, !dbg !13
  %1346 = insertelement <4 x float> poison, float %1337, i64 0, !dbg !13
  %1347 = insertelement <4 x float> %1346, float %1341, i64 1, !dbg !13
  %1348 = insertelement <4 x float> %1347, float %1332, i64 2, !dbg !13
  %1349 = insertelement <4 x float> %1348, float %1333, i64 3, !dbg !13
  %1350 = select <4 x i1> %1077, <4 x float> %1345, <4 x float> %1349, !dbg !13
  %1351 = extractelement <4 x float> %882, i64 0, !dbg !13
  %1352 = extractelement <4 x float> %882, i64 1, !dbg !13
  %1353 = extractelement <4 x float> %882, i64 2, !dbg !13
  %1354 = extractelement <4 x float> %882, i64 3, !dbg !13
  %1355 = select i1 %.not, float %1352, float %1351, !dbg !13
  %1356 = bitcast float %1355 to i32, !dbg !13
  %1357 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1356), !dbg !13
  %1358 = bitcast i32 %1357 to float, !dbg !13
  %1359 = select i1 %.not, float %1351, float %1358, !dbg !13
  %1360 = select i1 %.not, float %1358, float %1352, !dbg !13
  %1361 = select i1 %.not, float %1354, float %1353, !dbg !13
  %1362 = bitcast float %1361 to i32, !dbg !13
  %1363 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1362), !dbg !13
  %1364 = bitcast i32 %1363 to float, !dbg !13
  %1365 = select i1 %.not, float %1353, float %1364, !dbg !13
  %1366 = select i1 %.not, float %1364, float %1354, !dbg !13
  %1367 = select i1 %.not84, float %1365, float %1359, !dbg !13
  %1368 = bitcast float %1367 to i32, !dbg !13
  %1369 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1368), !dbg !13
  %1370 = bitcast i32 %1369 to float, !dbg !13
  %1371 = select i1 %.not84, float %1366, float %1360, !dbg !13
  %1372 = bitcast float %1371 to i32, !dbg !13
  %1373 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1372), !dbg !13
  %1374 = bitcast i32 %1373 to float, !dbg !13
  %1375 = insertelement <4 x float> poison, float %1359, i64 0, !dbg !13
  %1376 = insertelement <4 x float> %1375, float %1360, i64 1, !dbg !13
  %1377 = insertelement <4 x float> %1376, float %1370, i64 2, !dbg !13
  %1378 = insertelement <4 x float> %1377, float %1374, i64 3, !dbg !13
  %1379 = insertelement <4 x float> poison, float %1370, i64 0, !dbg !13
  %1380 = insertelement <4 x float> %1379, float %1374, i64 1, !dbg !13
  %1381 = insertelement <4 x float> %1380, float %1365, i64 2, !dbg !13
  %1382 = insertelement <4 x float> %1381, float %1366, i64 3, !dbg !13
  %1383 = select <4 x i1> %1077, <4 x float> %1378, <4 x float> %1382, !dbg !13
  %1384 = extractelement <4 x float> %883, i64 0, !dbg !13
  %1385 = extractelement <4 x float> %883, i64 1, !dbg !13
  %1386 = extractelement <4 x float> %883, i64 2, !dbg !13
  %1387 = extractelement <4 x float> %883, i64 3, !dbg !13
  %1388 = select i1 %.not, float %1385, float %1384, !dbg !13
  %1389 = bitcast float %1388 to i32, !dbg !13
  %1390 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1389), !dbg !13
  %1391 = bitcast i32 %1390 to float, !dbg !13
  %1392 = select i1 %.not, float %1384, float %1391, !dbg !13
  %1393 = select i1 %.not, float %1391, float %1385, !dbg !13
  %1394 = select i1 %.not, float %1387, float %1386, !dbg !13
  %1395 = bitcast float %1394 to i32, !dbg !13
  %1396 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1395), !dbg !13
  %1397 = bitcast i32 %1396 to float, !dbg !13
  %1398 = select i1 %.not, float %1386, float %1397, !dbg !13
  %1399 = select i1 %.not, float %1397, float %1387, !dbg !13
  %1400 = select i1 %.not84, float %1398, float %1392, !dbg !13
  %1401 = bitcast float %1400 to i32, !dbg !13
  %1402 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1401), !dbg !13
  %1403 = bitcast i32 %1402 to float, !dbg !13
  %1404 = select i1 %.not84, float %1399, float %1393, !dbg !13
  %1405 = bitcast float %1404 to i32, !dbg !13
  %1406 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1405), !dbg !13
  %1407 = bitcast i32 %1406 to float, !dbg !13
  %1408 = insertelement <4 x float> poison, float %1392, i64 0, !dbg !13
  %1409 = insertelement <4 x float> %1408, float %1393, i64 1, !dbg !13
  %1410 = insertelement <4 x float> %1409, float %1403, i64 2, !dbg !13
  %1411 = insertelement <4 x float> %1410, float %1407, i64 3, !dbg !13
  %1412 = insertelement <4 x float> poison, float %1403, i64 0, !dbg !13
  %1413 = insertelement <4 x float> %1412, float %1407, i64 1, !dbg !13
  %1414 = insertelement <4 x float> %1413, float %1398, i64 2, !dbg !13
  %1415 = insertelement <4 x float> %1414, float %1399, i64 3, !dbg !13
  %1416 = select <4 x i1> %1077, <4 x float> %1411, <4 x float> %1415, !dbg !13
  %1417 = extractelement <4 x float> %884, i64 0, !dbg !13
  %1418 = extractelement <4 x float> %884, i64 1, !dbg !13
  %1419 = extractelement <4 x float> %884, i64 2, !dbg !13
  %1420 = extractelement <4 x float> %884, i64 3, !dbg !13
  %1421 = select i1 %.not, float %1418, float %1417, !dbg !13
  %1422 = bitcast float %1421 to i32, !dbg !13
  %1423 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1422), !dbg !13
  %1424 = bitcast i32 %1423 to float, !dbg !13
  %1425 = select i1 %.not, float %1417, float %1424, !dbg !13
  %1426 = select i1 %.not, float %1424, float %1418, !dbg !13
  %1427 = select i1 %.not, float %1420, float %1419, !dbg !13
  %1428 = bitcast float %1427 to i32, !dbg !13
  %1429 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1428), !dbg !13
  %1430 = bitcast i32 %1429 to float, !dbg !13
  %1431 = select i1 %.not, float %1419, float %1430, !dbg !13
  %1432 = select i1 %.not, float %1430, float %1420, !dbg !13
  %1433 = select i1 %.not84, float %1431, float %1425, !dbg !13
  %1434 = bitcast float %1433 to i32, !dbg !13
  %1435 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1434), !dbg !13
  %1436 = bitcast i32 %1435 to float, !dbg !13
  %1437 = select i1 %.not84, float %1432, float %1426, !dbg !13
  %1438 = bitcast float %1437 to i32, !dbg !13
  %1439 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1438), !dbg !13
  %1440 = bitcast i32 %1439 to float, !dbg !13
  %1441 = insertelement <4 x float> poison, float %1425, i64 0, !dbg !13
  %1442 = insertelement <4 x float> %1441, float %1426, i64 1, !dbg !13
  %1443 = insertelement <4 x float> %1442, float %1436, i64 2, !dbg !13
  %1444 = insertelement <4 x float> %1443, float %1440, i64 3, !dbg !13
  %1445 = insertelement <4 x float> poison, float %1436, i64 0, !dbg !13
  %1446 = insertelement <4 x float> %1445, float %1440, i64 1, !dbg !13
  %1447 = insertelement <4 x float> %1446, float %1431, i64 2, !dbg !13
  %1448 = insertelement <4 x float> %1447, float %1432, i64 3, !dbg !13
  %1449 = select <4 x i1> %1077, <4 x float> %1444, <4 x float> %1448, !dbg !13
  %1450 = extractelement <4 x float> %789, i64 0, !dbg !13
  %1451 = extractelement <4 x float> %789, i64 1, !dbg !13
  %1452 = extractelement <4 x float> %789, i64 2, !dbg !13
  %1453 = extractelement <4 x float> %789, i64 3, !dbg !13
  %1454 = select i1 %.not, float %1451, float %1450, !dbg !13
  %1455 = bitcast float %1454 to i32, !dbg !13
  %1456 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1455), !dbg !13
  %1457 = bitcast i32 %1456 to float, !dbg !13
  %1458 = select i1 %.not, float %1450, float %1457, !dbg !13
  %1459 = select i1 %.not, float %1457, float %1451, !dbg !13
  %1460 = select i1 %.not, float %1453, float %1452, !dbg !13
  %1461 = bitcast float %1460 to i32, !dbg !13
  %1462 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1461), !dbg !13
  %1463 = bitcast i32 %1462 to float, !dbg !13
  %1464 = select i1 %.not, float %1452, float %1463, !dbg !13
  %1465 = select i1 %.not, float %1463, float %1453, !dbg !13
  %1466 = select i1 %.not84, float %1464, float %1458, !dbg !13
  %1467 = bitcast float %1466 to i32, !dbg !13
  %1468 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1467), !dbg !13
  %1469 = select i1 %.not84, float %1465, float %1459, !dbg !13
  %1470 = bitcast float %1469 to i32, !dbg !13
  %1471 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1470), !dbg !13
  %1472 = extractelement <4 x float> %791, i64 0, !dbg !13
  %1473 = extractelement <4 x float> %791, i64 1, !dbg !13
  %1474 = extractelement <4 x float> %791, i64 2, !dbg !13
  %1475 = extractelement <4 x float> %791, i64 3, !dbg !13
  %1476 = select i1 %.not, float %1473, float %1472, !dbg !13
  %1477 = bitcast float %1476 to i32, !dbg !13
  %1478 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1477), !dbg !13
  %1479 = bitcast i32 %1478 to float, !dbg !13
  %1480 = select i1 %.not, float %1472, float %1479, !dbg !13
  %1481 = select i1 %.not, float %1479, float %1473, !dbg !13
  %1482 = select i1 %.not, float %1475, float %1474, !dbg !13
  %1483 = bitcast float %1482 to i32, !dbg !13
  %1484 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1483), !dbg !13
  %1485 = bitcast i32 %1484 to float, !dbg !13
  %1486 = select i1 %.not, float %1474, float %1485, !dbg !13
  %1487 = select i1 %.not, float %1485, float %1475, !dbg !13
  %1488 = select i1 %.not84, float %1486, float %1480, !dbg !13
  %1489 = bitcast float %1488 to i32, !dbg !13
  %1490 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1489), !dbg !13
  %1491 = select i1 %.not84, float %1487, float %1481, !dbg !13
  %1492 = bitcast float %1491 to i32, !dbg !13
  %1493 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1492), !dbg !13
  %1494 = extractelement <4 x float> %811, i64 0, !dbg !13
  %1495 = extractelement <4 x float> %811, i64 1, !dbg !13
  %1496 = extractelement <4 x float> %811, i64 2, !dbg !13
  %1497 = extractelement <4 x float> %811, i64 3, !dbg !13
  %1498 = select i1 %.not, float %1495, float %1494, !dbg !13
  %1499 = bitcast float %1498 to i32, !dbg !13
  %1500 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1499), !dbg !13
  %1501 = bitcast i32 %1500 to float, !dbg !13
  %1502 = select i1 %.not, float %1494, float %1501, !dbg !13
  %1503 = select i1 %.not, float %1501, float %1495, !dbg !13
  %1504 = select i1 %.not, float %1497, float %1496, !dbg !13
  %1505 = bitcast float %1504 to i32, !dbg !13
  %1506 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1505), !dbg !13
  %1507 = bitcast i32 %1506 to float, !dbg !13
  %1508 = select i1 %.not, float %1496, float %1507, !dbg !13
  %1509 = select i1 %.not, float %1507, float %1497, !dbg !13
  %1510 = select i1 %.not84, float %1508, float %1502, !dbg !13
  %1511 = bitcast float %1510 to i32, !dbg !13
  %1512 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1511), !dbg !13
  %1513 = select i1 %.not84, float %1509, float %1503, !dbg !13
  %1514 = bitcast float %1513 to i32, !dbg !13
  %1515 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1514), !dbg !13
  %1516 = extractelement <4 x float> %812, i64 0, !dbg !13
  %1517 = extractelement <4 x float> %812, i64 1, !dbg !13
  %1518 = extractelement <4 x float> %812, i64 2, !dbg !13
  %1519 = extractelement <4 x float> %812, i64 3, !dbg !13
  %1520 = select i1 %.not, float %1517, float %1516, !dbg !13
  %1521 = bitcast float %1520 to i32, !dbg !13
  %1522 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1521), !dbg !13
  %1523 = bitcast i32 %1522 to float, !dbg !13
  %1524 = select i1 %.not, float %1516, float %1523, !dbg !13
  %1525 = select i1 %.not, float %1523, float %1517, !dbg !13
  %1526 = select i1 %.not, float %1519, float %1518, !dbg !13
  %1527 = bitcast float %1526 to i32, !dbg !13
  %1528 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1527), !dbg !13
  %1529 = bitcast i32 %1528 to float, !dbg !13
  %1530 = select i1 %.not, float %1518, float %1529, !dbg !13
  %1531 = select i1 %.not, float %1529, float %1519, !dbg !13
  %1532 = select i1 %.not84, float %1530, float %1524, !dbg !13
  %1533 = bitcast float %1532 to i32, !dbg !13
  %1534 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1533), !dbg !13
  %1535 = select i1 %.not84, float %1531, float %1525, !dbg !13
  %1536 = bitcast float %1535 to i32, !dbg !13
  %1537 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1536), !dbg !13
  %1538 = insertelement <2 x i32> poison, i32 %1049, i64 0, !dbg !13
  %1539 = insertelement <2 x i32> %1538, i32 %1047, i64 1, !dbg !13
  %1540 = bitcast <2 x i32> %1539 to <2 x float>, !dbg !13
  %1541 = insertelement <2 x float> poison, float %k0.sum.anchor4777, i64 0, !dbg !13
  %1542 = insertelement <2 x float> %1541, float %k0.sum.anchor4785, i64 1, !dbg !13
  %1543 = fadd <2 x float> %1542, %1540, !dbg !13
  %bc = bitcast <2 x float> %1543 to <2 x i32>, !dbg !13
  %1544 = extractelement <2 x i32> %bc, i64 1, !dbg !13
  %1545 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1544), !dbg !13
  %bc186 = bitcast <2 x float> %1543 to <2 x i32>, !dbg !13
  %1546 = extractelement <2 x i32> %bc186, i64 0, !dbg !13
  %1547 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1546), !dbg !13
  %1548 = insertelement <2 x i32> poison, i32 %1547, i64 0, !dbg !13
  %1549 = insertelement <2 x i32> %1548, i32 %1545, i64 1, !dbg !13
  %1550 = bitcast <2 x i32> %1549 to <2 x float>, !dbg !13
  %1551 = fadd <2 x float> %1543, %1550, !dbg !13
  %1552 = fdiv arcp <2 x float> splat (float 1.000000e+00), %1551, !dbg !46
  %1553 = shufflevector <2 x float> %1552, <2 x float> poison, <4 x i32> zeroinitializer, !dbg !46
  %1554 = shufflevector <2 x float> %1552, <2 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, !dbg !46
  %1555 = bitcast i32 %1534 to float, !dbg !13
  %1556 = bitcast i32 %1537 to float, !dbg !13
  %1557 = insertelement <4 x float> poison, float %1524, i64 0, !dbg !13
  %1558 = insertelement <4 x float> %1557, float %1525, i64 1, !dbg !13
  %1559 = insertelement <4 x float> %1558, float %1555, i64 2, !dbg !13
  %1560 = insertelement <4 x float> %1559, float %1556, i64 3, !dbg !13
  %1561 = insertelement <4 x float> poison, float %1555, i64 0, !dbg !13
  %1562 = insertelement <4 x float> %1561, float %1556, i64 1, !dbg !13
  %1563 = insertelement <4 x float> %1562, float %1530, i64 2, !dbg !13
  %1564 = insertelement <4 x float> %1563, float %1531, i64 3, !dbg !13
  %1565 = select <4 x i1> %1077, <4 x float> %1560, <4 x float> %1564, !dbg !13
  %1566 = fmul <4 x float> %1554, %1565, !dbg !46
  %1567 = fptrunc <4 x float> %1566 to <4 x half>, !dbg !47
  %1568 = extractelement <4 x half> %1567, i64 3, !dbg !47
  %1569 = extractelement <4 x half> %1567, i64 2, !dbg !47
  %1570 = extractelement <4 x half> %1567, i64 1, !dbg !47
  %1571 = extractelement <4 x half> %1567, i64 0, !dbg !47
  %1572 = bitcast i32 %1512 to float, !dbg !13
  %1573 = bitcast i32 %1515 to float, !dbg !13
  %1574 = insertelement <4 x float> poison, float %1502, i64 0, !dbg !13
  %1575 = insertelement <4 x float> %1574, float %1503, i64 1, !dbg !13
  %1576 = insertelement <4 x float> %1575, float %1572, i64 2, !dbg !13
  %1577 = insertelement <4 x float> %1576, float %1573, i64 3, !dbg !13
  %1578 = insertelement <4 x float> poison, float %1572, i64 0, !dbg !13
  %1579 = insertelement <4 x float> %1578, float %1573, i64 1, !dbg !13
  %1580 = insertelement <4 x float> %1579, float %1508, i64 2, !dbg !13
  %1581 = insertelement <4 x float> %1580, float %1509, i64 3, !dbg !13
  %1582 = select <4 x i1> %1077, <4 x float> %1577, <4 x float> %1581, !dbg !13
  %1583 = fmul <4 x float> %1554, %1582, !dbg !46
  %1584 = fptrunc <4 x float> %1583 to <4 x half>, !dbg !47
  %1585 = extractelement <4 x half> %1584, i64 3, !dbg !47
  %1586 = extractelement <4 x half> %1584, i64 2, !dbg !47
  %1587 = extractelement <4 x half> %1584, i64 1, !dbg !47
  %1588 = extractelement <4 x half> %1584, i64 0, !dbg !47
  %1589 = bitcast i32 %1490 to float, !dbg !13
  %1590 = bitcast i32 %1493 to float, !dbg !13
  %1591 = insertelement <4 x float> poison, float %1480, i64 0, !dbg !13
  %1592 = insertelement <4 x float> %1591, float %1481, i64 1, !dbg !13
  %1593 = insertelement <4 x float> %1592, float %1589, i64 2, !dbg !13
  %1594 = insertelement <4 x float> %1593, float %1590, i64 3, !dbg !13
  %1595 = insertelement <4 x float> poison, float %1589, i64 0, !dbg !13
  %1596 = insertelement <4 x float> %1595, float %1590, i64 1, !dbg !13
  %1597 = insertelement <4 x float> %1596, float %1486, i64 2, !dbg !13
  %1598 = insertelement <4 x float> %1597, float %1487, i64 3, !dbg !13
  %1599 = select <4 x i1> %1077, <4 x float> %1594, <4 x float> %1598, !dbg !13
  %1600 = fmul <4 x float> %1553, %1599, !dbg !46
  %1601 = fptrunc <4 x float> %1600 to <4 x half>, !dbg !47
  %1602 = extractelement <4 x half> %1601, i64 3, !dbg !47
  %1603 = extractelement <4 x half> %1601, i64 2, !dbg !47
  %1604 = extractelement <4 x half> %1601, i64 1, !dbg !47
  %1605 = extractelement <4 x half> %1601, i64 0, !dbg !47
  %1606 = bitcast i32 %1468 to float, !dbg !13
  %1607 = bitcast i32 %1471 to float, !dbg !13
  %1608 = insertelement <4 x float> poison, float %1458, i64 0, !dbg !13
  %1609 = insertelement <4 x float> %1608, float %1459, i64 1, !dbg !13
  %1610 = insertelement <4 x float> %1609, float %1606, i64 2, !dbg !13
  %1611 = insertelement <4 x float> %1610, float %1607, i64 3, !dbg !13
  %1612 = insertelement <4 x float> poison, float %1606, i64 0, !dbg !13
  %1613 = insertelement <4 x float> %1612, float %1607, i64 1, !dbg !13
  %1614 = insertelement <4 x float> %1613, float %1464, i64 2, !dbg !13
  %1615 = insertelement <4 x float> %1614, float %1465, i64 3, !dbg !13
  %1616 = select <4 x i1> %1077, <4 x float> %1611, <4 x float> %1615, !dbg !13
  %1617 = fmul <4 x float> %1553, %1616, !dbg !46
  %1618 = fptrunc <4 x float> %1617 to <4 x half>, !dbg !47
  %1619 = extractelement <4 x half> %1618, i64 3, !dbg !47
  %1620 = extractelement <4 x half> %1618, i64 2, !dbg !47
  %1621 = extractelement <4 x half> %1618, i64 1, !dbg !47
  %1622 = extractelement <4 x half> %1618, i64 0, !dbg !47
  %address.thread4196 = tail call i32 asm sideeffect "", "=v,0"(i32 %34) #8, !dbg !48, !range !10, !frisk.register_lifetimes !5
  %address.local4197 = zext nneg i32 %address.thread4196 to i64, !dbg !9
  %1623 = lshr i64 %address.local4197, 2, !dbg !48
  %address.local4200 = and i64 %1623, 48, !dbg !48
  %address.local4202 = and i64 %address.local4197, 240, !dbg !48
  %address.local4201 = add i64 %32, %address.local4197, !dbg !48
  %address.local4203 = sub i64 %address.local4201, %address.local4202, !dbg !48
  %address.local4204 = add i64 %address.local4203, %address.local4200, !dbg !48
  %1624 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %42, !dbg !48
  %address.local4207.idx = shl nuw nsw i64 %address.local4204, 8, !dbg !48
  %address.local4207 = getelementptr inbounds nuw i8, ptr addrspace(1) %1624, i64 %address.local4207.idx, !dbg !48
  %address.local4210 = and i64 %1623, 12, !dbg !11
  %address.local4211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4210, !dbg !48
  store half %1622, ptr addrspace(1) %address.local4211, align 2, !dbg !48
  %address.local4213 = or disjoint i64 %address.local4210, 1, !dbg !48
  %address.local4214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4213, !dbg !48
  store half %1621, ptr addrspace(1) %address.local4214, align 2, !dbg !48
  %address.local4216 = or disjoint i64 %address.local4210, 2, !dbg !48
  %address.local4217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4216, !dbg !48
  store half %1620, ptr addrspace(1) %address.local4217, align 2, !dbg !48
  %address.local4219 = or disjoint i64 %address.local4210, 3, !dbg !48
  %address.local4220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4219, !dbg !48
  store half %1619, ptr addrspace(1) %address.local4220, align 2, !dbg !48
  %address.local4222 = or disjoint i64 %address.local4210, 16, !dbg !48
  %address.local4223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4222, !dbg !48
  store half %1605, ptr addrspace(1) %address.local4223, align 2, !dbg !48
  %address.local4225 = or disjoint i64 %address.local4210, 17, !dbg !48
  %address.local4226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4225, !dbg !48
  store half %1604, ptr addrspace(1) %address.local4226, align 2, !dbg !48
  %address.local4228 = or disjoint i64 %address.local4210, 18, !dbg !48
  %address.local4229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4228, !dbg !48
  store half %1603, ptr addrspace(1) %address.local4229, align 2, !dbg !48
  %address.local4231 = or disjoint i64 %address.local4210, 19, !dbg !48
  %address.local4232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4231, !dbg !48
  store half %1602, ptr addrspace(1) %address.local4232, align 2, !dbg !48
  %address.local4236 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local4207, i64 16384, !dbg !48
  %address.local4237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4210, !dbg !48
  store half %1588, ptr addrspace(1) %address.local4237, align 2, !dbg !48
  %address.local4239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4213, !dbg !48
  store half %1587, ptr addrspace(1) %address.local4239, align 2, !dbg !48
  %address.local4241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4216, !dbg !48
  store half %1586, ptr addrspace(1) %address.local4241, align 2, !dbg !48
  %address.local4243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4219, !dbg !48
  store half %1585, ptr addrspace(1) %address.local4243, align 2, !dbg !48
  %address.local4245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4222, !dbg !48
  store half %1571, ptr addrspace(1) %address.local4245, align 2, !dbg !48
  %address.local4247 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4225, !dbg !48
  store half %1570, ptr addrspace(1) %address.local4247, align 2, !dbg !48
  %address.local4249 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4228, !dbg !48
  store half %1569, ptr addrspace(1) %address.local4249, align 2, !dbg !48
  %address.local4251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4231, !dbg !48
  store half %1568, ptr addrspace(1) %address.local4251, align 2, !dbg !48
  %1625 = fmul <4 x float> %1554, %1449, !dbg !46
  %1626 = fptrunc <4 x float> %1625 to <4 x half>, !dbg !47
  %1627 = extractelement <4 x half> %1626, i64 3, !dbg !47
  %1628 = extractelement <4 x half> %1626, i64 2, !dbg !47
  %1629 = extractelement <4 x half> %1626, i64 1, !dbg !47
  %1630 = extractelement <4 x half> %1626, i64 0, !dbg !47
  %1631 = fmul <4 x float> %1554, %1416, !dbg !46
  %1632 = fptrunc <4 x float> %1631 to <4 x half>, !dbg !47
  %1633 = extractelement <4 x half> %1632, i64 3, !dbg !47
  %1634 = extractelement <4 x half> %1632, i64 2, !dbg !47
  %1635 = extractelement <4 x half> %1632, i64 1, !dbg !47
  %1636 = extractelement <4 x half> %1632, i64 0, !dbg !47
  %1637 = fmul <4 x float> %1553, %1383, !dbg !46
  %1638 = fptrunc <4 x float> %1637 to <4 x half>, !dbg !47
  %1639 = extractelement <4 x half> %1638, i64 3, !dbg !47
  %1640 = extractelement <4 x half> %1638, i64 2, !dbg !47
  %1641 = extractelement <4 x half> %1638, i64 1, !dbg !47
  %1642 = extractelement <4 x half> %1638, i64 0, !dbg !47
  %1643 = fmul <4 x float> %1553, %1350, !dbg !46
  %1644 = fptrunc <4 x float> %1643 to <4 x half>, !dbg !47
  %1645 = extractelement <4 x half> %1644, i64 3, !dbg !47
  %1646 = extractelement <4 x half> %1644, i64 2, !dbg !47
  %1647 = extractelement <4 x half> %1644, i64 1, !dbg !47
  %1648 = extractelement <4 x half> %1644, i64 0, !dbg !47
  %address.local4267 = or disjoint i64 %address.local4210, 32, !dbg !48
  %address.local4268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4267, !dbg !48
  store half %1648, ptr addrspace(1) %address.local4268, align 2, !dbg !48
  %address.local4270 = or disjoint i64 %address.local4210, 33, !dbg !48
  %address.local4271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4270, !dbg !48
  store half %1647, ptr addrspace(1) %address.local4271, align 2, !dbg !48
  %address.local4273 = or disjoint i64 %address.local4210, 34, !dbg !48
  %address.local4274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4273, !dbg !48
  store half %1646, ptr addrspace(1) %address.local4274, align 2, !dbg !48
  %address.local4276 = or disjoint i64 %address.local4210, 35, !dbg !48
  %address.local4277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4276, !dbg !48
  store half %1645, ptr addrspace(1) %address.local4277, align 2, !dbg !48
  %address.local4279 = or disjoint i64 %address.local4210, 48, !dbg !48
  %address.local4280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4279, !dbg !48
  store half %1642, ptr addrspace(1) %address.local4280, align 2, !dbg !48
  %address.local4282 = or disjoint i64 %address.local4210, 49, !dbg !48
  %address.local4283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4282, !dbg !48
  store half %1641, ptr addrspace(1) %address.local4283, align 2, !dbg !48
  %address.local4285 = or disjoint i64 %address.local4210, 50, !dbg !48
  %address.local4286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4285, !dbg !48
  store half %1640, ptr addrspace(1) %address.local4286, align 2, !dbg !48
  %address.local4288 = or i64 %1623, 51, !dbg !48
  %address.local4289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4288, !dbg !48
  store half %1639, ptr addrspace(1) %address.local4289, align 2, !dbg !48
  %address.local4294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4267, !dbg !48
  store half %1636, ptr addrspace(1) %address.local4294, align 2, !dbg !48
  %address.local4296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4270, !dbg !48
  store half %1635, ptr addrspace(1) %address.local4296, align 2, !dbg !48
  %address.local4298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4273, !dbg !48
  store half %1634, ptr addrspace(1) %address.local4298, align 2, !dbg !48
  %address.local4300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4276, !dbg !48
  store half %1633, ptr addrspace(1) %address.local4300, align 2, !dbg !48
  %address.local4302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4279, !dbg !48
  store half %1630, ptr addrspace(1) %address.local4302, align 2, !dbg !48
  %address.local4304 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4282, !dbg !48
  store half %1629, ptr addrspace(1) %address.local4304, align 2, !dbg !48
  %address.local4306 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4285, !dbg !48
  store half %1628, ptr addrspace(1) %address.local4306, align 2, !dbg !48
  %address.local4308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4288, !dbg !48
  store half %1627, ptr addrspace(1) %address.local4308, align 2, !dbg !48
  %1649 = fmul <4 x float> %1554, %1317, !dbg !46
  %1650 = fptrunc <4 x float> %1649 to <4 x half>, !dbg !47
  %1651 = extractelement <4 x half> %1650, i64 3, !dbg !47
  %1652 = extractelement <4 x half> %1650, i64 2, !dbg !47
  %1653 = extractelement <4 x half> %1650, i64 1, !dbg !47
  %1654 = extractelement <4 x half> %1650, i64 0, !dbg !47
  %1655 = fmul <4 x float> %1554, %1284, !dbg !46
  %1656 = fptrunc <4 x float> %1655 to <4 x half>, !dbg !47
  %1657 = extractelement <4 x half> %1656, i64 3, !dbg !47
  %1658 = extractelement <4 x half> %1656, i64 2, !dbg !47
  %1659 = extractelement <4 x half> %1656, i64 1, !dbg !47
  %1660 = extractelement <4 x half> %1656, i64 0, !dbg !47
  %1661 = fmul <4 x float> %1553, %1251, !dbg !46
  %1662 = fptrunc <4 x float> %1661 to <4 x half>, !dbg !47
  %1663 = extractelement <4 x half> %1662, i64 3, !dbg !47
  %1664 = extractelement <4 x half> %1662, i64 2, !dbg !47
  %1665 = extractelement <4 x half> %1662, i64 1, !dbg !47
  %1666 = extractelement <4 x half> %1662, i64 0, !dbg !47
  %1667 = fmul <4 x float> %1553, %1218, !dbg !46
  %1668 = fptrunc <4 x float> %1667 to <4 x half>, !dbg !47
  %1669 = extractelement <4 x half> %1668, i64 3, !dbg !47
  %1670 = extractelement <4 x half> %1668, i64 2, !dbg !47
  %1671 = extractelement <4 x half> %1668, i64 1, !dbg !47
  %1672 = extractelement <4 x half> %1668, i64 0, !dbg !47
  %address.local4324 = or disjoint i64 %address.local4210, 64, !dbg !48
  %address.local4325 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4324, !dbg !48
  store half %1672, ptr addrspace(1) %address.local4325, align 2, !dbg !48
  %address.local4327 = or disjoint i64 %address.local4210, 65, !dbg !48
  %address.local4328 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4327, !dbg !48
  store half %1671, ptr addrspace(1) %address.local4328, align 2, !dbg !48
  %address.local4330 = or disjoint i64 %address.local4210, 66, !dbg !48
  %address.local4331 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4330, !dbg !48
  store half %1670, ptr addrspace(1) %address.local4331, align 2, !dbg !48
  %address.local4333 = or disjoint i64 %address.local4210, 67, !dbg !48
  %address.local4334 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4333, !dbg !48
  store half %1669, ptr addrspace(1) %address.local4334, align 2, !dbg !48
  %address.local4336 = or disjoint i64 %address.local4210, 80, !dbg !48
  %address.local4337 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4336, !dbg !48
  store half %1666, ptr addrspace(1) %address.local4337, align 2, !dbg !48
  %address.local4339 = or disjoint i64 %address.local4210, 81, !dbg !48
  %address.local4340 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4339, !dbg !48
  store half %1665, ptr addrspace(1) %address.local4340, align 2, !dbg !48
  %address.local4342 = or disjoint i64 %address.local4210, 82, !dbg !48
  %address.local4343 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4342, !dbg !48
  store half %1664, ptr addrspace(1) %address.local4343, align 2, !dbg !48
  %address.local4345 = or disjoint i64 %address.local4210, 83, !dbg !48
  %address.local4346 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4345, !dbg !48
  store half %1663, ptr addrspace(1) %address.local4346, align 2, !dbg !48
  %address.local4351 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4324, !dbg !48
  store half %1660, ptr addrspace(1) %address.local4351, align 2, !dbg !48
  %address.local4353 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4327, !dbg !48
  store half %1659, ptr addrspace(1) %address.local4353, align 2, !dbg !48
  %address.local4355 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4330, !dbg !48
  store half %1658, ptr addrspace(1) %address.local4355, align 2, !dbg !48
  %address.local4357 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4333, !dbg !48
  store half %1657, ptr addrspace(1) %address.local4357, align 2, !dbg !48
  %address.local4359 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4336, !dbg !48
  store half %1654, ptr addrspace(1) %address.local4359, align 2, !dbg !48
  %address.local4361 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4339, !dbg !48
  store half %1653, ptr addrspace(1) %address.local4361, align 2, !dbg !48
  %address.local4363 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4342, !dbg !48
  store half %1652, ptr addrspace(1) %address.local4363, align 2, !dbg !48
  %address.local4365 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4345, !dbg !48
  store half %1651, ptr addrspace(1) %address.local4365, align 2, !dbg !48
  %1673 = fmul <4 x float> %1554, %1185, !dbg !46
  %1674 = fptrunc <4 x float> %1673 to <4 x half>, !dbg !47
  %1675 = extractelement <4 x half> %1674, i64 3, !dbg !47
  %1676 = extractelement <4 x half> %1674, i64 2, !dbg !47
  %1677 = extractelement <4 x half> %1674, i64 1, !dbg !47
  %1678 = extractelement <4 x half> %1674, i64 0, !dbg !47
  %1679 = fmul <4 x float> %1554, %1152, !dbg !46
  %1680 = fptrunc <4 x float> %1679 to <4 x half>, !dbg !47
  %1681 = extractelement <4 x half> %1680, i64 3, !dbg !47
  %1682 = extractelement <4 x half> %1680, i64 2, !dbg !47
  %1683 = extractelement <4 x half> %1680, i64 1, !dbg !47
  %1684 = extractelement <4 x half> %1680, i64 0, !dbg !47
  %1685 = fmul <4 x float> %1553, %1119, !dbg !46
  %1686 = fptrunc <4 x float> %1685 to <4 x half>, !dbg !47
  %1687 = extractelement <4 x half> %1686, i64 3, !dbg !47
  %1688 = extractelement <4 x half> %1686, i64 2, !dbg !47
  %1689 = extractelement <4 x half> %1686, i64 1, !dbg !47
  %1690 = extractelement <4 x half> %1686, i64 0, !dbg !47
  %1691 = fmul <4 x float> %1553, %1086, !dbg !46
  %1692 = fptrunc <4 x float> %1691 to <4 x half>, !dbg !47
  %1693 = extractelement <4 x half> %1692, i64 3, !dbg !47
  %1694 = extractelement <4 x half> %1692, i64 2, !dbg !47
  %1695 = extractelement <4 x half> %1692, i64 1, !dbg !47
  %1696 = extractelement <4 x half> %1692, i64 0, !dbg !47
  %address.local4381 = or disjoint i64 %address.local4210, 96, !dbg !48
  %address.local4382 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4381, !dbg !48
  store half %1696, ptr addrspace(1) %address.local4382, align 2, !dbg !48
  %address.local4384 = or disjoint i64 %address.local4210, 97, !dbg !48
  %address.local4385 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4384, !dbg !48
  store half %1695, ptr addrspace(1) %address.local4385, align 2, !dbg !48
  %address.local4387 = or disjoint i64 %address.local4210, 98, !dbg !48
  %address.local4388 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4387, !dbg !48
  store half %1694, ptr addrspace(1) %address.local4388, align 2, !dbg !48
  %address.local4390 = or disjoint i64 %address.local4210, 99, !dbg !48
  %address.local4391 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4390, !dbg !48
  store half %1693, ptr addrspace(1) %address.local4391, align 2, !dbg !48
  %address.local4393 = or disjoint i64 %address.local4210, 112, !dbg !48
  %address.local4394 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4393, !dbg !48
  store half %1690, ptr addrspace(1) %address.local4394, align 2, !dbg !48
  %address.local4396 = or disjoint i64 %address.local4210, 113, !dbg !48
  %address.local4397 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4396, !dbg !48
  store half %1689, ptr addrspace(1) %address.local4397, align 2, !dbg !48
  %address.local4399 = or disjoint i64 %address.local4210, 114, !dbg !48
  %address.local4400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4399, !dbg !48
  store half %1688, ptr addrspace(1) %address.local4400, align 2, !dbg !48
  %address.local4402 = or i64 %1623, 115, !dbg !48
  %address.local4403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4207, i64 %address.local4402, !dbg !48
  store half %1687, ptr addrspace(1) %address.local4403, align 2, !dbg !48
  %address.local4408 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4381, !dbg !48
  store half %1684, ptr addrspace(1) %address.local4408, align 2, !dbg !48
  %address.local4410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4384, !dbg !48
  store half %1683, ptr addrspace(1) %address.local4410, align 2, !dbg !48
  %address.local4412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4387, !dbg !48
  store half %1682, ptr addrspace(1) %address.local4412, align 2, !dbg !48
  %address.local4414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4390, !dbg !48
  store half %1681, ptr addrspace(1) %address.local4414, align 2, !dbg !48
  %address.local4416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4393, !dbg !48
  store half %1678, ptr addrspace(1) %address.local4416, align 2, !dbg !48
  %address.local4418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4396, !dbg !48
  store half %1677, ptr addrspace(1) %address.local4418, align 2, !dbg !48
  %address.local4420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4399, !dbg !48
  store half %1676, ptr addrspace(1) %address.local4420, align 2, !dbg !48
  %address.local4422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4236, i64 %address.local4402, !dbg !48
  store half %1675, ptr addrspace(1) %address.local4422, align 2, !dbg !48
  ret void, !dbg !8
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.ballot.i64(i1) #4

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.lazy_softmax_base"="v1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_cvt_pack"="rtz" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { memory(inaccessiblemem: readwrite) }
attributes #11 = { convergent willreturn memory(read) }
attributes #12 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/shared_memory_plan/noncausal")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 74, column: 20, scope: !3)
!8 = !DILocation(line: 23, column: 3, scope: !3)
!9 = !DILocation(line: 67, column: 22, scope: !3)
!10 = !{i32 0, i32 256}
!11 = !DILocation(line: 69, column: 24, scope: !3)
!12 = !DILocation(line: 68, column: 22, scope: !3)
!13 = !DILocation(line: 57, column: 24, scope: !3)
!14 = !DILocation(line: 71, column: 23, scope: !3)
!15 = !{i32 0}
!16 = !{i32 1}
!17 = !{i32 2}
!18 = !{i32 3}
!19 = !{i32 4}
!20 = !{i32 5}
!21 = !{i32 6}
!22 = !{i32 7}
!23 = !{i32 8}
!24 = !{i32 9}
!25 = !{i32 10}
!26 = !{i32 11}
!27 = !{i32 12}
!28 = !{i32 13}
!29 = !{i32 14}
!30 = !{i32 15}
!31 = !{i32 16}
!32 = !{i32 17}
!33 = !{i32 18}
!34 = !{i32 19}
!35 = !{i32 20}
!36 = !{i32 21}
!37 = !{i32 22}
!38 = !{i32 23}
!39 = !DILocation(line: 86, column: 20, scope: !3)
!40 = !DILocation(line: 78, column: 22, scope: !3)
!41 = !DILocation(line: 79, column: 22, scope: !3)
!42 = !DILocation(line: 88, column: 22, scope: !3)
!43 = !DILocation(line: 89, column: 20, scope: !3)
!44 = !DILocation(line: 81, column: 16, scope: !3)
!45 = !DILocation(line: 80, column: 20, scope: !3)
!46 = !DILocation(line: 92, column: 23, scope: !3)
!47 = !DILocation(line: 93, column: 19, scope: !3)
!48 = !DILocation(line: 94, column: 9, scope: !3)
