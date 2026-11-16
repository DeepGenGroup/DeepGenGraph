; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/shared_memory_plan/causal/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/shared_memory_plan/causal/input.mlir"

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
  %24 = and i64 %23, 576460752303423484, !dbg !8
  %25 = and i64 %18, 3, !dbg !8
  %26 = or disjoint i64 %24, %25, !dbg !8
  %27 = sub nsw i64 31, %26, !dbg !8
  %28 = lshr i64 %22, 2, !dbg !8
  %29 = and i64 %28, 31, !dbg !8
  %30 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !8
  %31 = sext i32 %30 to i64, !dbg !8
  %32 = icmp ugt i64 %22, 1023, !dbg !9
  %33 = add nsw i64 %26, -32, !dbg !9
  %34 = select i1 %32, i64 %33, i64 %27, !dbg !9
  %35 = sdiv i64 %34, 32, !dbg !9
  %36 = sext i1 %32 to i64, !dbg !9
  %37 = xor i64 %35, %36, !dbg !9
  %38 = add nsw i64 %37, %28, !dbg !9
  %39 = and i64 %38, 31, !dbg !9
  %40 = shl i64 %27, 7, !dbg !9
  %41 = shl i64 %37, 12, !dbg !9
  %42 = sub i64 %40, %41, !dbg !9
  %43 = shl nsw i64 %42, 7, !dbg !9
  %44 = tail call range(i32 0, 256) i32 @llvm.amdgcn.workitem.id.x(), !dbg !9
  %45 = zext nneg i32 %44 to i64, !dbg !9
  %.idx = shl nuw nsw i64 %39, 20, !dbg !9
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %44) #8, !dbg !9, !range !10, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !9
  %address.local941 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local942 = and i64 %address.local941, 1536, !dbg !9
  %address.local943 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local942, !dbg !9
  %address.local946 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local947 = and i64 %address.local946, 16256, !dbg !9
  %address.local949 = lshr i64 %address.local, 4, !dbg !9
  %address.local950 = and i64 %address.local949, 8, !dbg !9
  %46 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx4694 = shl i64 %42, 8, !dbg !9
  %47 = add i64 %.idx, %.idx4694, !dbg !9
  %address.local948.idx4695 = or disjoint i64 %address.local947, %address.local950, !dbg !9
  %48 = shl nuw nsw i64 %address.local948.idx4695, 1, !dbg !9
  %49 = or disjoint i64 %48, %47, !dbg !9
  %dma.buffer.offset4660 = trunc i64 %49 to i32, !dbg !9
  %50 = insertelement <2 x i64> poison, i64 %46, i64 0, !dbg !9
  %51 = bitcast <2 x i64> %50 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc4661 = shufflevector <4 x i32> %51, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local943, i64 4096, !dbg !9
  %dma.buffer.offset4658 = or disjoint i32 %dma.buffer.offset4660, 32, !dbg !9
  %52 = shl nuw nsw i64 %39, 19, !dbg !9
  %53 = add i64 %52, %43, !dbg !9
  %address.local965 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local942, !dbg !9
  %54 = or disjoint i64 %address.local948.idx4695, %53, !dbg !9
  %55 = trunc i64 %54 to i32, !dbg !9
  %.tr = shl i32 %55, 1, !dbg !9
  %dma.buffer.offset4656 = or disjoint i32 %.tr, 64, !dbg !9
  %address.local977 = getelementptr i8, ptr addrspace(3) %address.local965, i64 4096, !dbg !9
  %56 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local943, i64 0, !dbg !9
  %57 = insertelement <4 x ptr addrspace(3)> %56, ptr addrspace(3) %address.local955, i64 1, !dbg !9
  %58 = insertelement <4 x ptr addrspace(3)> %57, ptr addrspace(3) %address.local965, i64 2, !dbg !9
  %59 = insertelement <4 x ptr addrspace(3)> %58, ptr addrspace(3) %address.local977, i64 3, !dbg !9
  %60 = ptrtoint <4 x ptr addrspace(3)> %59 to <4 x i64>, !dbg !9
  %61 = trunc <4 x i64> %60 to <4 x i32>, !dbg !9
  %62 = extractelement <4 x i32> %61, i64 0, !dbg !9
  %63 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %62), !dbg !9
  %64 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %63, i32 %dma.buffer.offset4660, <4 x i32> %dma.buffer.rsrc4661) #9, !dbg !9
  %65 = extractelement <4 x i32> %61, i64 1, !dbg !9
  %66 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %65), !dbg !9
  %67 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset4658, <4 x i32> %dma.buffer.rsrc4661) #9, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %68 = extractelement <4 x i32> %61, i64 2, !dbg !9
  %69 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %68), !dbg !9
  %70 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %dma.buffer.offset4656, <4 x i32> %dma.buffer.rsrc4661) #9, !dbg !9
  %71 = extractelement <4 x i32> %61, i64 3, !dbg !9
  %72 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %71), !dbg !9
  %dma.buffer.offset4654 = or disjoint i32 %.tr, 96, !dbg !9
  %73 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %dma.buffer.offset4654, <4 x i32> %dma.buffer.rsrc4661) #9, !dbg !9
  %address.local984 = lshr i32 %address.thread, 2, !dbg !11
  %address.local985 = and i32 %address.local984, 4, !dbg !11
  %address.local986 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local987 = and i32 %address.local986, 384, !dbg !11
  %address.local988 = or disjoint i32 %address.local985, %address.local987, !dbg !11
  %address.local989 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local990 = and i32 %address.local989, 120, !dbg !11
  %address.local991 = or disjoint i32 %address.local988, %address.local990, !dbg !11
  %address.local992 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local993 = and i32 %address.local992, 1024, !dbg !11
  %address.local994 = or disjoint i32 %address.local991, %address.local993, !dbg !11
  %address.local995 = zext nneg i32 %address.local994 to i64, !dbg !11
  %address.local996 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local995, !dbg !11
  %74 = load <4 x half>, ptr addrspace(3) %address.local996, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local996, i64 4096, !dbg !11
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %76 = fmul <4 x half> %74, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep4494 = getelementptr i8, ptr addrspace(3) %address.local996, i64 1024, !dbg !11
  %77 = load <4 x half>, ptr addrspace(3) %lds.const.gep4494, align 8, !dbg !11
  %78 = fmul <4 x half> %75, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep4495 = getelementptr i8, ptr addrspace(3) %address.local996, i64 5120, !dbg !11
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep4495, align 8, !dbg !11
  %80 = fmul <4 x half> %77, splat (half 1.275630e-01), !dbg !11
  %81 = fmul <4 x half> %79, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset4652 = or disjoint i32 %.tr, 128, !dbg !9
  %82 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %63, i32 %dma.buffer.offset4652, <4 x i32> %dma.buffer.rsrc4661) #9, !dbg !9
  %dma.buffer.offset4650 = or disjoint i32 %.tr, 160, !dbg !9
  %83 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset4650, <4 x i32> %dma.buffer.rsrc4661) #9, !dbg !9
  %lds.const.gep4496 = getelementptr i8, ptr addrspace(3) %address.local996, i64 8192, !dbg !11
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep4496, align 8, !dbg !11
  %lds.const.gep4497 = getelementptr i8, ptr addrspace(3) %address.local996, i64 12288, !dbg !11
  %85 = load <4 x half>, ptr addrspace(3) %lds.const.gep4497, align 8, !dbg !11
  %lds.const.gep4498 = getelementptr i8, ptr addrspace(3) %address.local996, i64 9216, !dbg !11
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep4498, align 8, !dbg !11
  %87 = fmul <4 x half> %84, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep4499 = getelementptr i8, ptr addrspace(3) %address.local996, i64 13312, !dbg !11
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep4499, align 8, !dbg !11
  %89 = fmul <4 x half> %85, splat (half 1.275630e-01), !dbg !11
  %90 = fmul <4 x half> %86, splat (half 1.275630e-01), !dbg !11
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset4648 = or disjoint i32 %.tr, 192, !dbg !9
  %92 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %69, i32 %dma.buffer.offset4648, <4 x i32> %dma.buffer.rsrc4661) #9, !dbg !9
  %dma.buffer.offset4646 = or disjoint i32 %.tr, 224, !dbg !9
  %93 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %72, i32 %dma.buffer.offset4646, <4 x i32> %dma.buffer.rsrc4661) #9, !dbg !9
  %94 = load <4 x half>, ptr addrspace(3) %address.local996, align 8, !dbg !11
  %95 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep4494, align 8, !dbg !11
  %97 = fmul <4 x half> %94, splat (half 1.275630e-01), !dbg !11
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep4495, align 8, !dbg !11
  %99 = fmul <4 x half> %95, splat (half 1.275630e-01), !dbg !11
  %100 = fmul <4 x half> %96, splat (half 1.275630e-01), !dbg !11
  %101 = fmul <4 x half> %98, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %102 = load <4 x half>, ptr addrspace(3) %lds.const.gep4496, align 8, !dbg !11
  %103 = load <4 x half>, ptr addrspace(3) %lds.const.gep4497, align 8, !dbg !11
  %104 = load <4 x half>, ptr addrspace(3) %lds.const.gep4498, align 8, !dbg !11
  %105 = fmul <4 x half> %102, splat (half 1.275630e-01), !dbg !11
  %106 = load <4 x half>, ptr addrspace(3) %lds.const.gep4499, align 8, !dbg !11
  %107 = fmul <4 x half> %103, splat (half 1.275630e-01), !dbg !11
  %108 = fmul <4 x half> %104, splat (half 1.275630e-01), !dbg !11
  %109 = fmul <4 x half> %106, splat (half 1.275630e-01), !dbg !11
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %110 = sub nsw i64 32, %26, !dbg !13
  %111 = shl nuw nsw i64 %29, 7, !dbg !14
  %112 = shl nsw i64 %31, 12, !dbg !14
  %113 = or disjoint i64 %111, %112, !dbg !14
  %114 = and i64 %45, 48, !dbg !15
  %115 = icmp ult i64 %22, 1024, !dbg !12
  br i1 %115, label %.lr.ph, label %._crit_edge, !dbg !12

.lr.ph:                                           ; preds = %4
  %116 = and i64 %45, 15, !dbg !15
  %117 = lshr i64 %45, 6, !dbg !11
  %118 = shl nuw nsw i64 %117, 4, !dbg !15
  %119 = or disjoint i64 %116, %118, !dbg !15
  %120 = or disjoint i64 %119, %40, !dbg !15
  %121 = trunc i64 %120 to i32, !dbg !15
  %122 = add i32 %121, 65, !dbg !15
  %123 = add i32 %121, 1, !dbg !15
  %124 = shl nuw nsw i64 %45, 2, !dbg !16
  %125 = shl nsw i64 %31, 17, !dbg !14
  %126 = shl nuw nsw i64 %29, 12, !dbg !14
  %127 = shl nuw nsw i64 %117, 2
  %128 = add nuw nsw i64 %127, %124
  %129 = add nuw nsw i64 %117, %45
  %130 = shl nuw nsw i64 %129, 2
  %131 = and i64 %130, 2040
  %132 = sub nsw i64 %128, %131
  %133 = lshr i32 %44, 6
  %134 = and i32 %44, 40
  %135 = icmp eq i32 %134, 40
  %.neg19 = sext i1 %135 to i32
  %136 = zext i1 %135 to i32
  %narrow871 = add nuw nsw i32 %133, %136
  %137 = shl nuw nsw i32 %narrow871, 10
  %138 = zext nneg i32 %137 to i64
  %139 = shl nuw nsw i32 %44, 4
  %140 = and i32 %139, 3584
  %141 = shl nuw nsw i32 %44, 6
  %142 = and i32 %141, 16256
  %narrow872 = add nuw nsw i32 %140, %142
  %143 = zext nneg i32 %narrow872 to i64
  %144 = sub nsw i64 %143, %138
  %145 = ptrtoint ptr addrspace(1) %1 to i64
  %146 = and i64 %144, 16256
  %147 = add nsw i64 %132, %146
  %148 = insertelement <2 x i64> poison, i64 %145, i64 0
  %149 = bitcast <2 x i64> %148 to <4 x i32>
  %dma.buffer.rsrc4693 = shufflevector <4 x i32> %149, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %150 = add nuw nsw i64 %124, 1040
  %151 = add nuw nsw i64 %150, %127
  %152 = add nuw nsw i64 %130, 1040
  %153 = and i64 %152, 8184
  %154 = sub nsw i64 %151, %153
  %reass.sub = sub nsw i32 %.neg19, %133
  %155 = shl nuw nsw i32 %44, 5
  %156 = and i32 %155, 7168
  %157 = or disjoint i32 %156, 8192
  %158 = shl nuw nsw i32 %44, 7
  %159 = and i32 %158, 32512
  %narrow874 = add nuw nsw i32 %157, %159
  %.idx4724 = shl nsw i64 %154, 1
  %160 = add nsw i64 %.idx4724, 16
  %narrow873.neg = shl nsw i32 %reass.sub, 11
  %.neg20 = add nsw i32 %narrow873.neg, 24576
  %161 = add nuw nsw i32 %narrow874, %.neg20
  %162 = and i32 %161, 32512
  %.idx4725 = zext nneg i32 %162 to i64
  %163 = add nsw i64 %160, %.idx4725
  %164 = add nuw nsw i64 %124, 2080
  %165 = add nuw nsw i64 %164, %127
  %166 = add nuw nsw i64 %130, 2080
  %167 = and i64 %166, 8184
  %168 = sub nsw i64 %165, %167
  %169 = or disjoint i32 %133, 8
  %narrow876 = add nuw nsw i32 %169, %136
  %170 = shl nuw nsw i32 %narrow876, 10
  %171 = zext nneg i32 %170 to i64
  %172 = or disjoint i32 %140, 8192
  %narrow877 = add nuw nsw i32 %172, %142
  %173 = zext nneg i32 %narrow877 to i64
  %174 = sub nsw i64 %173, %171
  %.idx4730 = shl nsw i64 %168, 1
  %175 = add nsw i64 %.idx4730, 32
  %176 = shl nsw i64 %174, 1
  %.idx4731 = and i64 %176, 32512
  %177 = add nsw i64 %175, %.idx4731
  %178 = add nuw nsw i64 %124, 3120
  %179 = add nuw nsw i64 %178, %127
  %180 = add nuw nsw i64 %130, 3120
  %181 = and i64 %180, 16376
  %182 = sub nsw i64 %179, %181
  %183 = or disjoint i32 %156, 24576
  %narrow879 = add nuw nsw i32 %183, %159
  %.neg22 = add nsw i32 %narrow873.neg, 8192
  %184 = add nuw nsw i32 %narrow879, %.neg22
  %185 = and i32 %184, 32512
  %186 = or disjoint i32 %185, 48
  %187 = or disjoint i64 %126, %125
  %188 = or disjoint i32 %185, 112
  %189 = or disjoint i32 %185, 176
  %190 = or disjoint i32 %185, 240
  %191 = ptrtoint ptr addrspace(1) %2 to i64
  %192 = insertelement <2 x i64> poison, i64 %191, i64 0
  %193 = bitcast <2 x i64> %192 to <4 x i32>
  %dma.buffer.rsrc4645 = shufflevector <4 x i32> %193, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %194 = insertelement <2 x i64> poison, i64 %113, i64 0, !dbg !14
  %195 = insertelement <32 x i32> poison, i32 %122, i64 0, !dbg !15
  %196 = shufflevector <32 x i32> %195, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !15
  %197 = insertelement <32 x i32> poison, i32 %123, i64 0, !dbg !15
  %198 = shufflevector <32 x i32> %197, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !15
  %199 = insertelement <2 x i64> poison, i64 %187, i64 1
  br label %200, !dbg !12

200:                                              ; preds = %.lr.ph, %200
  %201 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1480, %200 ]
  %202 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1476, %200 ]
  %203 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1472, %200 ]
  %204 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %.lr.ph ], [ %1468, %200 ]
  %205 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %1232, %200 ]
  %206 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1466, %200 ]
  %207 = phi i64 [ 0, %.lr.ph ], [ %1481, %200 ]
  %208 = lshr i64 %207, 5, !dbg !14
  %209 = shl nuw nsw i64 %207, 7, !dbg !14
  %210 = shl i64 %208, 19, !dbg !14
  %211 = shl nuw nsw i64 %207, 14, !dbg !14
  %212 = sub i64 %211, %210, !dbg !14
  %213 = lshr i64 %207, 12, !dbg !14
  %214 = add nuw nsw i64 %213, %28, !dbg !14
  %215 = shl i64 %214, 19, !dbg !14
  %216 = and i64 %215, 16252928, !dbg !14
  %217 = and i64 %208, 127, !dbg !14
  %address.thread1199 = tail call i32 asm sideeffect "", "=v,0"(i32 %44) #8, !dbg !14, !range !10, !frisk.register_lifetimes !5
  %address.local1200 = zext nneg i32 %address.thread1199 to i64, !dbg !9
  %address.local1201 = shl nuw nsw i64 %address.local1200, 2, !dbg !14
  %address.local1202 = and i64 %address.local1201, 768, !dbg !14
  %address.local1203 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1202, !dbg !14
  %.idx47144719 = or disjoint i64 %216, %217, !dbg !14
  %218 = add i64 %.idx47144719, %212, !dbg !14
  %219 = add i64 %218, %147, !dbg !14
  %.tr4720 = trunc i64 %219 to i32, !dbg !14
  %dma.buffer.offset4692 = shl i32 %.tr4720, 1, !dbg !14
  %address.local1209 = getelementptr i8, ptr addrspace(3) %address.local1203, i64 2048, !dbg !14
  %220 = shl i64 %218, 1, !dbg !14
  %221 = add i64 %220, %163, !dbg !14
  %dma.buffer.offset4690 = trunc i64 %221 to i32, !dbg !14
  %address.local1215 = getelementptr i8, ptr addrspace(3) %address.local1203, i64 4096, !dbg !14
  %222 = add i64 %220, %177, !dbg !14
  %dma.buffer.offset4688 = trunc i64 %222 to i32, !dbg !14
  %address.local1221 = getelementptr i8, ptr addrspace(3) %address.local1203, i64 6144, !dbg !14
  %223 = add i64 %218, %182, !dbg !14
  %.tr4739 = trunc i64 %223 to i32, !dbg !14
  %224 = shl i32 %.tr4739, 1, !dbg !14
  %dma.buffer.offset4686 = add i32 %186, %224, !dbg !14
  %address.local1228 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1202, !dbg !14
  %.idx47404746 = add i64 %212, %208, !dbg !14
  %225 = add i64 %.idx47404746, %113, !dbg !14
  %226 = insertelement <2 x i64> %199, i64 %208, i64 0, !dbg !14
  %227 = insertelement <2 x i64> %194, i64 %207, i64 1, !dbg !14
  %228 = add nsw <2 x i64> %226, %227, !dbg !14
  %229 = extractelement <2 x i64> %228, i64 1, !dbg !14
  %230 = extractelement <2 x i64> %228, i64 0, !dbg !14
  %231 = add nuw nsw i64 %229, 1024, !dbg !14
  %232 = add nsw <2 x i64> %228, <i64 32, i64 1024>, !dbg !14
  %.lobit124 = ashr i64 %231, 63, !dbg !14
  %233 = icmp slt <2 x i64> %228, <i64 -32, i64 -1024>, !dbg !14
  %234 = icmp slt i64 %230, -32, !dbg !14
  %235 = sub nsw <2 x i64> <i64 -33, i64 -1025>, %228, !dbg !14
  %236 = select <2 x i1> %233, <2 x i64> %235, <2 x i64> %232, !dbg !14
  %237 = sext i1 %234 to i64, !dbg !14
  %238 = insertelement <2 x i64> poison, i64 %237, i64 0, !dbg !14
  %239 = insertelement <2 x i64> %238, i64 %.lobit124, i64 1, !dbg !14
  %address.local1234 = getelementptr i8, ptr addrspace(3) %address.local1228, i64 2048, !dbg !14
  %address.local1240 = getelementptr i8, ptr addrspace(3) %address.local1228, i64 4096, !dbg !14
  %address.local1246 = getelementptr i8, ptr addrspace(3) %address.local1228, i64 6144, !dbg !14
  %240 = insertelement <8 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local1203, i64 0, !dbg !14
  %241 = insertelement <8 x ptr addrspace(3)> %240, ptr addrspace(3) %address.local1209, i64 1, !dbg !14
  %242 = insertelement <8 x ptr addrspace(3)> %241, ptr addrspace(3) %address.local1215, i64 2, !dbg !14
  %243 = insertelement <8 x ptr addrspace(3)> %242, ptr addrspace(3) %address.local1221, i64 3, !dbg !14
  %244 = insertelement <8 x ptr addrspace(3)> %243, ptr addrspace(3) %address.local1228, i64 4, !dbg !14
  %245 = insertelement <8 x ptr addrspace(3)> %244, ptr addrspace(3) %address.local1234, i64 5, !dbg !14
  %246 = insertelement <8 x ptr addrspace(3)> %245, ptr addrspace(3) %address.local1240, i64 6, !dbg !14
  %247 = insertelement <8 x ptr addrspace(3)> %246, ptr addrspace(3) %address.local1246, i64 7, !dbg !14
  %248 = ptrtoint <8 x ptr addrspace(3)> %247 to <8 x i64>, !dbg !14
  %249 = trunc <8 x i64> %248 to <8 x i32>, !dbg !14
  %250 = extractelement <8 x i32> %249, i64 0, !dbg !14
  %251 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %250), !dbg !14
  %252 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %251, i32 %dma.buffer.offset4692, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %253 = extractelement <8 x i32> %249, i64 1, !dbg !14
  %254 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %253), !dbg !14
  %255 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %254, i32 %dma.buffer.offset4690, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %256 = extractelement <8 x i32> %249, i64 2, !dbg !14
  %257 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %256), !dbg !14
  %258 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %257, i32 %dma.buffer.offset4688, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %259 = extractelement <8 x i32> %249, i64 3, !dbg !14
  %260 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %259), !dbg !14
  %261 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %260, i32 %dma.buffer.offset4686, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %262 = extractelement <8 x i32> %249, i64 4, !dbg !14
  %263 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %262), !dbg !14
  %264 = sdiv <2 x i64> %236, <i64 128, i64 4096>, !dbg !14
  %265 = xor <2 x i64> %264, %239, !dbg !14
  %266 = shl <2 x i64> %265, <i64 7, i64 19>, !dbg !14
  %267 = extractelement <2 x i64> %266, i64 0, !dbg !14
  %268 = sub i64 %225, %267, !dbg !14
  %269 = extractelement <2 x i64> %266, i64 1, !dbg !14
  %270 = and i64 %269, 16252928, !dbg !14
  %271 = add i64 %268, %270, !dbg !14
  %272 = add i64 %271, %147, !dbg !14
  %.tr4747 = trunc i64 %272 to i32, !dbg !14
  %273 = shl i32 %.tr4747, 1, !dbg !14
  %dma.buffer.offset4684 = add i32 %273, 64, !dbg !14
  %274 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %263, i32 %dma.buffer.offset4684, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %275 = extractelement <8 x i32> %249, i64 5, !dbg !14
  %276 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %275), !dbg !14
  %277 = shl i64 %271, 1, !dbg !14
  %278 = add i64 %277, 64, !dbg !14
  %279 = add i64 %278, %163, !dbg !14
  %dma.buffer.offset4682 = trunc i64 %279 to i32, !dbg !14
  %280 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %276, i32 %dma.buffer.offset4682, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %281 = extractelement <8 x i32> %249, i64 6, !dbg !14
  %282 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %281), !dbg !14
  %283 = add i64 %278, %177, !dbg !14
  %dma.buffer.offset4680 = trunc i64 %283 to i32, !dbg !14
  %284 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %282, i32 %dma.buffer.offset4680, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %285 = extractelement <8 x i32> %249, i64 7, !dbg !14
  %286 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %285), !dbg !14
  %287 = add i64 %271, %182, !dbg !14
  %.tr4769 = trunc i64 %287 to i32, !dbg !14
  %288 = shl i32 %.tr4769, 1, !dbg !14
  %dma.buffer.offset4678 = add i32 %188, %288, !dbg !14
  %289 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %286, i32 %dma.buffer.offset4678, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %address.local1251 = mul nuw nsw i64 %address.local1200, 160, !dbg !16
  %address.local1252 = lshr i64 %address.local1200, 2, !dbg !16
  %address.local1253 = mul nsw i64 %address.local1252, -640, !dbg !16
  %address.local1255 = shl nuw nsw i64 %address.local1200, 1, !dbg !16
  %address.local1256 = and i64 %address.local1255, 24, !dbg !16
  %address.local1258 = shl nuw nsw i64 %address.local1200, 5, !dbg !16
  %address.local1259 = and i64 %address.local1258, 64, !dbg !16
  %address.local1261 = and i64 %address.local1200, 63, !dbg !11
  %address.local1262 = lshr i64 %address.local1261, 4, !dbg !11
  %address.local1263 = shl nuw nsw i64 %address.local1262, 2, !dbg !11
  %address.local1265 = lshr i64 %address.local1261, 5, !dbg !11
  %address.local1266 = mul nuw nsw i64 %address.local1265, 1016, !dbg !11
  %address.local1268 = lshr i32 %address.thread1199, 1, !dbg !16
  %address.local1269 = and i32 %address.local1268, 16, !dbg !16
  %address.local1270 = shl nuw nsw i32 %address.thread1199, 1, !dbg !16
  %address.local1271 = and i32 %address.local1270, 4, !dbg !16
  %address.local1272 = or disjoint i32 %address.local1269, %address.local1271, !dbg !16
  %address.local1273 = zext nneg i32 %address.local1272 to i64, !dbg !16
  %address.local1276 = and i32 %address.local1268, 1, !dbg !16
  %address.local1277 = zext nneg i32 %address.local1276 to i64, !dbg !16
  %address.local1278 = and i64 %address.local1262, %address.local1277, !dbg !16
  %address.local1279 = lshr i32 %address.thread1199, 4, !dbg !16
  %address.local1280 = and i32 %address.local1279, 2, !dbg !16
  %address.local1281 = zext nneg i32 %address.local1280 to i64, !dbg !16
  %address.local1282 = or disjoint i64 %address.local1278, %address.local1281, !dbg !16
  %address.local1283.neg24 = mul nsw i64 %address.local1282, -8, !dbg !16
  %address.local1257 = add nuw nsw i64 %address.local1266, %address.local1263, !dbg !16
  %address.local1260 = add nuw nsw i64 %address.local1257, %address.local1251, !dbg !16
  %address.local1264 = add nsw i64 %address.local1260, %address.local1253, !dbg !16
  %address.local1254 = add nsw i64 %address.local1264, %address.local1256, !dbg !16
  %address.local1267 = sub nsw i64 %address.local1254, %address.local1259, !dbg !16
  %290 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1267, !dbg !16
  %291 = getelementptr [2 x i8], ptr addrspace(3) %290, i64 %address.local1273, !dbg !16
  %address.local1285 = getelementptr [2 x i8], ptr addrspace(3) %291, i64 %address.local1283.neg24, !dbg !16
  %292 = load <4 x half>, ptr addrspace(3) %address.local1285, align 8, !dbg !16, !frisk.k_lds_pair !17
  %lds.const.gep4510 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 128, !dbg !16
  %293 = load <4 x half>, ptr addrspace(3) %lds.const.gep4510, align 8, !dbg !16, !frisk.k_lds_pair !17
  %lds.const.gep4508 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 4096, !dbg !16
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep4508, align 8, !dbg !16, !frisk.k_lds_pair !18
  %lds.const.gep4511 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 4224, !dbg !16
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep4511, align 8, !dbg !16, !frisk.k_lds_pair !18
  %address.local1314 = and i32 %address.local1270, 24, !dbg !16
  %address.local1315 = shl nuw nsw i32 %address.thread1199, 8, !dbg !16
  %address.local1316 = and i32 %address.local1315, 768, !dbg !16
  %address.local1318 = shl nuw nsw i32 %address.thread1199, 5, !dbg !16
  %address.local1319 = and i32 %address.local1318, 96, !dbg !16
  %296 = or disjoint i32 %address.local1316, %address.local1319, !dbg !16
  %address.local1320 = or disjoint i32 %296, %address.local1314, !dbg !16
  %address.local1321 = zext nneg i32 %address.local1320 to i64, !dbg !16
  %297 = shl nuw nsw i64 %address.local1252, 4, !dbg !16
  %address.local1324 = sub nsw i64 %address.local1201, %297, !dbg !16
  %address.local1325 = and i32 %address.thread1199, 1, !dbg !16
  %address.local1326 = zext nneg i32 %address.local1325 to i64, !dbg !16
  %address.local1327 = or disjoint i64 %address.local1324, %address.local1326, !dbg !16
  %298 = trunc nsw i64 %address.local1327 to i16, !dbg !16
  %address.local1329.lhs.trunc = add nuw nsw i16 %298, 1, !dbg !16
  %address.local13291 = sdiv i16 %address.local1329.lhs.trunc, 2, !dbg !16
  %address.local1329.sext = sext i16 %address.local13291 to i64, !dbg !16
  %address.local1333 = add nsw i64 %address.local1329.sext, %address.local1277, !dbg !16
  %address.local1336 = add nuw nsw i64 %address.local1257, %address.local1273, !dbg !16
  %address.local1337 = add nuw nsw i64 %address.local1336, %address.local1321, !dbg !16
  %address.local1344 = add nsw i64 %address.local1337, %address.local1283.neg24, !dbg !16
  %299 = shl nsw i64 %address.local1333, 6, !dbg !16
  %address.local1354 = sub nsw i64 %address.local1344, %299, !dbg !16
  %300 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1354, !dbg !16
  %address.local1356 = getelementptr i8, ptr addrspace(3) %300, i64 64, !dbg !16
  %301 = load <4 x half>, ptr addrspace(3) %address.local1356, align 8, !dbg !16, !frisk.k_lds_pair !19
  %lds.const.gep4512 = getelementptr i8, ptr addrspace(3) %300, i64 192, !dbg !16
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep4512, align 8, !dbg !16, !frisk.k_lds_pair !19
  %lds.const.gep4509 = getelementptr i8, ptr addrspace(3) %300, i64 4160, !dbg !16
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep4509, align 8, !dbg !16, !frisk.k_lds_pair !20
  %lds.const.gep4513 = getelementptr i8, ptr addrspace(3) %300, i64 4288, !dbg !16
  %304 = load <4 x half>, ptr addrspace(3) %lds.const.gep4513, align 8, !dbg !16, !frisk.k_lds_pair !20
  %lds.const.gep4514 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 1024, !dbg !16
  %305 = load <4 x half>, ptr addrspace(3) %lds.const.gep4514, align 8, !dbg !16, !frisk.k_lds_pair !21
  %lds.const.gep4518 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 1152, !dbg !16
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep4518, align 8, !dbg !16, !frisk.k_lds_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %292, <4 x float> zeroinitializer) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %lds.const.gep4515 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 5120, !dbg !16
  %308 = load <4 x half>, ptr addrspace(3) %lds.const.gep4515, align 8, !dbg !16, !frisk.k_lds_pair !22
  %lds.const.gep4519 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 5248, !dbg !16
  %309 = load <4 x half>, ptr addrspace(3) %lds.const.gep4519, align 8, !dbg !16, !frisk.k_lds_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %294, <4 x float> %307) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %lds.const.gep4516 = getelementptr i8, ptr addrspace(3) %300, i64 1088, !dbg !16
  %311 = load <4 x half>, ptr addrspace(3) %lds.const.gep4516, align 8, !dbg !16, !frisk.k_lds_pair !23
  %lds.const.gep4520 = getelementptr i8, ptr addrspace(3) %300, i64 1216, !dbg !16
  %312 = load <4 x half>, ptr addrspace(3) %lds.const.gep4520, align 8, !dbg !16, !frisk.k_lds_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %301, <4 x float> zeroinitializer) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %lds.const.gep4517 = getelementptr i8, ptr addrspace(3) %300, i64 5184, !dbg !16
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep4517, align 8, !dbg !16, !frisk.k_lds_pair !24
  %lds.const.gep4521 = getelementptr i8, ptr addrspace(3) %300, i64 5312, !dbg !16
  %315 = load <4 x half>, ptr addrspace(3) %lds.const.gep4521, align 8, !dbg !16, !frisk.k_lds_pair !24
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %303, <4 x float> %313) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %293, <4 x float> zeroinitializer) #8, !dbg !16
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %295, <4 x float> %317) #8, !dbg !16
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %302, <4 x float> zeroinitializer) #8, !dbg !16
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %304, <4 x float> %319) #8, !dbg !16
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %305, <4 x float> zeroinitializer) #8, !dbg !16
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %308, <4 x float> %321) #8, !dbg !16
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %311, <4 x float> zeroinitializer) #8, !dbg !16
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %314, <4 x float> %323) #8, !dbg !16
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %306, <4 x float> zeroinitializer) #8, !dbg !16
  %326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %309, <4 x float> %325) #8, !dbg !16
  %327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %312, <4 x float> zeroinitializer) #8, !dbg !16
  %328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %315, <4 x float> %327) #8, !dbg !16
  %329 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %292, <4 x float> zeroinitializer) #8, !dbg !16
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %294, <4 x float> %329) #8, !dbg !16
  %331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %301, <4 x float> zeroinitializer) #8, !dbg !16
  %332 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %303, <4 x float> %331) #8, !dbg !16
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %293, <4 x float> zeroinitializer) #8, !dbg !16
  %334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %295, <4 x float> %333) #8, !dbg !16
  %335 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %302, <4 x float> zeroinitializer) #8, !dbg !16
  %336 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %304, <4 x float> %335) #8, !dbg !16
  %337 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %305, <4 x float> zeroinitializer) #8, !dbg !16
  %338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %308, <4 x float> %337) #8, !dbg !16
  %339 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %311, <4 x float> zeroinitializer) #8, !dbg !16
  %340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %314, <4 x float> %339) #8, !dbg !16
  %341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %306, <4 x float> zeroinitializer) #8, !dbg !16
  %342 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %309, <4 x float> %341) #8, !dbg !16
  %343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %312, <4 x float> zeroinitializer) #8, !dbg !16
  %344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %315, <4 x float> %343) #8, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %345 = add nuw nsw i64 %229, 2048, !dbg !14
  %346 = add nsw <2 x i64> %228, <i64 64, i64 2048>, !dbg !14
  %.lobit165 = ashr i64 %345, 63, !dbg !14
  %347 = icmp slt <2 x i64> %228, <i64 -64, i64 -2048>, !dbg !14
  %348 = icmp slt i64 %230, -64, !dbg !14
  %349 = sub nsw <2 x i64> <i64 -65, i64 -2049>, %228, !dbg !14
  %350 = select <2 x i1> %347, <2 x i64> %349, <2 x i64> %346, !dbg !14
  %351 = sdiv <2 x i64> %350, <i64 128, i64 4096>, !dbg !14
  %352 = sext i1 %348 to i64, !dbg !14
  %353 = insertelement <2 x i64> poison, i64 %352, i64 0, !dbg !14
  %354 = insertelement <2 x i64> %353, i64 %.lobit165, i64 1, !dbg !14
  %355 = xor <2 x i64> %351, %354, !dbg !14
  %356 = shl <2 x i64> %355, <i64 7, i64 19>, !dbg !14
  %357 = extractelement <2 x i64> %356, i64 0, !dbg !14
  %358 = sub i64 %225, %357, !dbg !14
  %359 = extractelement <2 x i64> %356, i64 1, !dbg !14
  %360 = and i64 %359, 16252928, !dbg !14
  %361 = add i64 %358, %360, !dbg !14
  %362 = add i64 %361, %147, !dbg !14
  %.tr4777 = trunc i64 %362 to i32, !dbg !14
  %363 = shl i32 %.tr4777, 1, !dbg !14
  %dma.buffer.offset4676 = add i32 %363, 128, !dbg !14
  %364 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %251, i32 %dma.buffer.offset4676, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %365 = shl i64 %361, 1, !dbg !14
  %366 = add i64 %365, 128, !dbg !14
  %367 = add i64 %366, %163, !dbg !14
  %dma.buffer.offset4674 = trunc i64 %367 to i32, !dbg !14
  %368 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %254, i32 %dma.buffer.offset4674, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %369 = add i64 %366, %177, !dbg !14
  %dma.buffer.offset4672 = trunc i64 %369 to i32, !dbg !14
  %370 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %257, i32 %dma.buffer.offset4672, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %371 = add i64 %361, %182, !dbg !14
  %.tr4799 = trunc i64 %371 to i32, !dbg !14
  %372 = shl i32 %.tr4799, 1, !dbg !14
  %dma.buffer.offset4670 = add i32 %189, %372, !dbg !14
  %373 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %260, i32 %dma.buffer.offset4670, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %lds.const.gep4536 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 8192, !dbg !16
  %374 = load <4 x half>, ptr addrspace(3) %lds.const.gep4536, align 8, !dbg !16, !frisk.k_lds_pair !25
  %lds.const.gep4540 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 8320, !dbg !16
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep4540, align 8, !dbg !16, !frisk.k_lds_pair !25
  %lds.const.gep4537 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 12288, !dbg !16
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep4537, align 8, !dbg !16, !frisk.k_lds_pair !26
  %lds.const.gep4541 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 12416, !dbg !16
  %377 = load <4 x half>, ptr addrspace(3) %lds.const.gep4541, align 8, !dbg !16, !frisk.k_lds_pair !26
  %lds.const.gep4538 = getelementptr i8, ptr addrspace(3) %300, i64 8256, !dbg !16
  %378 = load <4 x half>, ptr addrspace(3) %lds.const.gep4538, align 8, !dbg !16, !frisk.k_lds_pair !27
  %lds.const.gep4542 = getelementptr i8, ptr addrspace(3) %300, i64 8384, !dbg !16
  %379 = load <4 x half>, ptr addrspace(3) %lds.const.gep4542, align 8, !dbg !16, !frisk.k_lds_pair !27
  %lds.const.gep4539 = getelementptr i8, ptr addrspace(3) %300, i64 12352, !dbg !16
  %380 = load <4 x half>, ptr addrspace(3) %lds.const.gep4539, align 8, !dbg !16, !frisk.k_lds_pair !28
  %lds.const.gep4543 = getelementptr i8, ptr addrspace(3) %300, i64 12480, !dbg !16
  %381 = load <4 x half>, ptr addrspace(3) %lds.const.gep4543, align 8, !dbg !16, !frisk.k_lds_pair !28
  %lds.const.gep4544 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 9216, !dbg !16
  %382 = load <4 x half>, ptr addrspace(3) %lds.const.gep4544, align 8, !dbg !16, !frisk.k_lds_pair !29
  %lds.const.gep4548 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 9344, !dbg !16
  %383 = load <4 x half>, ptr addrspace(3) %lds.const.gep4548, align 8, !dbg !16, !frisk.k_lds_pair !29
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %374, <4 x float> %310) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %lds.const.gep4545 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 13312, !dbg !16
  %385 = load <4 x half>, ptr addrspace(3) %lds.const.gep4545, align 8, !dbg !16, !frisk.k_lds_pair !30
  %lds.const.gep4549 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 13440, !dbg !16
  %386 = load <4 x half>, ptr addrspace(3) %lds.const.gep4549, align 8, !dbg !16, !frisk.k_lds_pair !30
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %376, <4 x float> %384) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %lds.const.gep4546 = getelementptr i8, ptr addrspace(3) %300, i64 9280, !dbg !16
  %388 = load <4 x half>, ptr addrspace(3) %lds.const.gep4546, align 8, !dbg !16, !frisk.k_lds_pair !31
  %lds.const.gep4550 = getelementptr i8, ptr addrspace(3) %300, i64 9408, !dbg !16
  %389 = load <4 x half>, ptr addrspace(3) %lds.const.gep4550, align 8, !dbg !16, !frisk.k_lds_pair !31
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %378, <4 x float> %316) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %lds.const.gep4547 = getelementptr i8, ptr addrspace(3) %300, i64 13376, !dbg !16
  %391 = load <4 x half>, ptr addrspace(3) %lds.const.gep4547, align 8, !dbg !16, !frisk.k_lds_pair !32
  %lds.const.gep4551 = getelementptr i8, ptr addrspace(3) %300, i64 13504, !dbg !16
  %392 = load <4 x half>, ptr addrspace(3) %lds.const.gep4551, align 8, !dbg !16, !frisk.k_lds_pair !32
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %380, <4 x float> %390) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %375, <4 x float> %318) #8, !dbg !16
  %395 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %377, <4 x float> %394) #8, !dbg !16
  %396 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %379, <4 x float> %320) #8, !dbg !16
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %381, <4 x float> %396) #8, !dbg !16
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %382, <4 x float> %322) #8, !dbg !16
  %399 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %385, <4 x float> %398) #8, !dbg !16
  %400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %388, <4 x float> %324) #8, !dbg !16
  %401 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %391, <4 x float> %400) #8, !dbg !16
  %402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %383, <4 x float> %326) #8, !dbg !16
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %386, <4 x float> %402) #8, !dbg !16
  %404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %389, <4 x float> %328) #8, !dbg !16
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %392, <4 x float> %404) #8, !dbg !16
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %374, <4 x float> %330) #8, !dbg !16
  %407 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %376, <4 x float> %406) #8, !dbg !16
  %408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %378, <4 x float> %332) #8, !dbg !16
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %380, <4 x float> %408) #8, !dbg !16
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %375, <4 x float> %334) #8, !dbg !16
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %377, <4 x float> %410) #8, !dbg !16
  %412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %379, <4 x float> %336) #8, !dbg !16
  %413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %381, <4 x float> %412) #8, !dbg !16
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %382, <4 x float> %338) #8, !dbg !16
  %415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %385, <4 x float> %414) #8, !dbg !16
  %416 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %388, <4 x float> %340) #8, !dbg !16
  %417 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %391, <4 x float> %416) #8, !dbg !16
  %418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %383, <4 x float> %342) #8, !dbg !16
  %419 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %386, <4 x float> %418) #8, !dbg !16
  %420 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %389, <4 x float> %344) #8, !dbg !16
  %421 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %392, <4 x float> %420) #8, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %422 = add nuw nsw i64 %229, 3072, !dbg !14
  %423 = add nsw <2 x i64> %228, <i64 96, i64 3072>, !dbg !14
  %.lobit206 = ashr i64 %422, 63, !dbg !14
  %424 = icmp slt <2 x i64> %228, <i64 -96, i64 -3072>, !dbg !14
  %425 = icmp slt i64 %230, -96, !dbg !14
  %426 = sub nsw <2 x i64> <i64 -97, i64 -3073>, %228, !dbg !14
  %427 = select <2 x i1> %424, <2 x i64> %426, <2 x i64> %423, !dbg !14
  %428 = sdiv <2 x i64> %427, <i64 128, i64 4096>, !dbg !14
  %429 = sext i1 %425 to i64, !dbg !14
  %430 = insertelement <2 x i64> poison, i64 %429, i64 0, !dbg !14
  %431 = insertelement <2 x i64> %430, i64 %.lobit206, i64 1, !dbg !14
  %432 = xor <2 x i64> %428, %431, !dbg !14
  %433 = shl <2 x i64> %432, <i64 7, i64 19>, !dbg !14
  %434 = extractelement <2 x i64> %433, i64 0, !dbg !14
  %435 = sub i64 %225, %434, !dbg !14
  %436 = extractelement <2 x i64> %433, i64 1, !dbg !14
  %437 = and i64 %436, 16252928, !dbg !14
  %438 = add i64 %435, %437, !dbg !14
  %439 = add i64 %438, %147, !dbg !14
  %.tr4807 = trunc i64 %439 to i32, !dbg !14
  %440 = shl i32 %.tr4807, 1, !dbg !14
  %dma.buffer.offset4668 = add i32 %440, 192, !dbg !14
  %441 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %263, i32 %dma.buffer.offset4668, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %442 = shl i64 %438, 1, !dbg !14
  %443 = add i64 %442, 192, !dbg !14
  %444 = add i64 %443, %163, !dbg !14
  %dma.buffer.offset4666 = trunc i64 %444 to i32, !dbg !14
  %445 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %276, i32 %dma.buffer.offset4666, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %446 = add i64 %443, %177, !dbg !14
  %dma.buffer.offset4664 = trunc i64 %446 to i32, !dbg !14
  %447 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %282, i32 %dma.buffer.offset4664, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %448 = add i64 %438, %182, !dbg !14
  %.tr4829 = trunc i64 %448 to i32, !dbg !14
  %449 = shl i32 %.tr4829, 1, !dbg !14
  %dma.buffer.offset4662 = add i32 %190, %449, !dbg !14
  %450 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %286, i32 %dma.buffer.offset4662, <4 x i32> %dma.buffer.rsrc4693) #9, !dbg !14
  %451 = load <4 x half>, ptr addrspace(3) %address.local1285, align 8, !dbg !16, !frisk.k_lds_pair !33
  %452 = load <4 x half>, ptr addrspace(3) %lds.const.gep4510, align 8, !dbg !16, !frisk.k_lds_pair !33
  %453 = load <4 x half>, ptr addrspace(3) %lds.const.gep4508, align 8, !dbg !16, !frisk.k_lds_pair !34
  %454 = load <4 x half>, ptr addrspace(3) %lds.const.gep4511, align 8, !dbg !16, !frisk.k_lds_pair !34
  %455 = load <4 x half>, ptr addrspace(3) %address.local1356, align 8, !dbg !16, !frisk.k_lds_pair !35
  %456 = load <4 x half>, ptr addrspace(3) %lds.const.gep4512, align 8, !dbg !16, !frisk.k_lds_pair !35
  %457 = load <4 x half>, ptr addrspace(3) %lds.const.gep4509, align 8, !dbg !16, !frisk.k_lds_pair !36
  %458 = load <4 x half>, ptr addrspace(3) %lds.const.gep4513, align 8, !dbg !16, !frisk.k_lds_pair !36
  %459 = load <4 x half>, ptr addrspace(3) %lds.const.gep4514, align 8, !dbg !16, !frisk.k_lds_pair !37
  %460 = load <4 x half>, ptr addrspace(3) %lds.const.gep4518, align 8, !dbg !16, !frisk.k_lds_pair !37
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %461 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %451, <4 x float> %387) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %462 = load <4 x half>, ptr addrspace(3) %lds.const.gep4515, align 8, !dbg !16, !frisk.k_lds_pair !38
  %463 = load <4 x half>, ptr addrspace(3) %lds.const.gep4519, align 8, !dbg !16, !frisk.k_lds_pair !38
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %464 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %453, <4 x float> %461) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep4516, align 8, !dbg !16, !frisk.k_lds_pair !39
  %466 = load <4 x half>, ptr addrspace(3) %lds.const.gep4520, align 8, !dbg !16, !frisk.k_lds_pair !39
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %467 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %455, <4 x float> %393) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %468 = load <4 x half>, ptr addrspace(3) %lds.const.gep4517, align 8, !dbg !16, !frisk.k_lds_pair !40
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep4521, align 8, !dbg !16, !frisk.k_lds_pair !40
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %470 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %457, <4 x float> %467) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_lds_prefetch !5
  %471 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %452, <4 x float> %395) #8, !dbg !16
  %472 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %454, <4 x float> %471) #8, !dbg !16
  %473 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %456, <4 x float> %397) #8, !dbg !16
  %474 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %458, <4 x float> %473) #8, !dbg !16
  %475 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %459, <4 x float> %399) #8, !dbg !16
  %476 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %462, <4 x float> %475) #8, !dbg !16
  %477 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %465, <4 x float> %401) #8, !dbg !16
  %478 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %468, <4 x float> %477) #8, !dbg !16
  %479 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %460, <4 x float> %403) #8, !dbg !16
  %480 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %463, <4 x float> %479) #8, !dbg !16
  %481 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %466, <4 x float> %405) #8, !dbg !16
  %482 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %469, <4 x float> %481) #8, !dbg !16
  %483 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %451, <4 x float> %407) #8, !dbg !16
  %484 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %453, <4 x float> %483) #8, !dbg !16
  %485 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %455, <4 x float> %409) #8, !dbg !16
  %486 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %457, <4 x float> %485) #8, !dbg !16
  %487 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %452, <4 x float> %411) #8, !dbg !16
  %488 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %454, <4 x float> %487) #8, !dbg !16
  %489 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %456, <4 x float> %413) #8, !dbg !16
  %490 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %458, <4 x float> %489) #8, !dbg !16
  %491 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %459, <4 x float> %415) #8, !dbg !16
  %492 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %462, <4 x float> %491) #8, !dbg !16
  %493 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %465, <4 x float> %417) #8, !dbg !16
  %494 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %468, <4 x float> %493) #8, !dbg !16
  %495 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %460, <4 x float> %419) #8, !dbg !16
  %496 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %463, <4 x float> %495) #8, !dbg !16
  %497 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %466, <4 x float> %421) #8, !dbg !16
  %498 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %469, <4 x float> %497) #8, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %499 = add nuw nsw i64 %208, %28, !dbg !41
  %500 = shl i64 %499, 19, !dbg !41
  %501 = and i64 %500, 16252928, !dbg !41
  %address.local3254 = shl nuw nsw i64 %address.local1200, 3, !dbg !41
  %address.local3255 = and i64 %address.local3254, 1536, !dbg !41
  %address.local3256 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local3255, !dbg !41
  %address.local3257 = ptrtoint ptr addrspace(3) %address.local3256 to i64, !dbg !41
  %address.local3258 = trunc i64 %address.local3257 to i32, !dbg !41
  %502 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3258), !dbg !41
  %address.local3259 = shl nuw nsw i64 %address.local1200, 7, !dbg !41
  %address.local3260 = and i64 %address.local3259, 16256, !dbg !41
  %address.local3262 = lshr i64 %address.local1200, 4, !dbg !41
  %address.local3263 = and i64 %address.local3262, 8, !dbg !41
  %.idx48304832 = add i64 %501, %212, !dbg !41
  %503 = or disjoint i64 %address.local3263, %.idx48304832, !dbg !41
  %504 = or disjoint i64 %503, %address.local3260, !dbg !41
  %.tr4834 = trunc i64 %504 to i32, !dbg !41
  %dma.buffer.offset4644 = shl i32 %.tr4834, 1, !dbg !41
  %505 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %502, i32 %dma.buffer.offset4644, <4 x i32> %dma.buffer.rsrc4645) #9, !dbg !41
  %address.local3268 = getelementptr i8, ptr addrspace(3) %address.local3256, i64 4096, !dbg !41
  %address.local3269 = ptrtoint ptr addrspace(3) %address.local3268 to i64, !dbg !41
  %address.local3270 = trunc i64 %address.local3269 to i32, !dbg !41
  %506 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3270), !dbg !41
  %507 = or disjoint i64 %address.local3260, %.idx48304832, !dbg !41
  %508 = or disjoint i64 %507, %address.local3263, !dbg !41
  %.tr4838 = trunc i64 %508 to i32, !dbg !41
  %509 = shl i32 %.tr4838, 1, !dbg !41
  %dma.buffer.offset4642 = or i32 %509, 32, !dbg !41
  %510 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %506, i32 %dma.buffer.offset4642, <4 x i32> %dma.buffer.rsrc4645) #9, !dbg !41
  %511 = load <4 x half>, ptr addrspace(3) %lds.const.gep4536, align 8, !dbg !16, !frisk.k_tail_pair !17
  %512 = load <4 x half>, ptr addrspace(3) %lds.const.gep4540, align 8, !dbg !16, !frisk.k_tail_pair !17
  %513 = load <4 x half>, ptr addrspace(3) %lds.const.gep4537, align 8, !dbg !16, !frisk.k_tail_pair !18
  %514 = load <4 x half>, ptr addrspace(3) %lds.const.gep4541, align 8, !dbg !16, !frisk.k_tail_pair !18
  %515 = load <4 x half>, ptr addrspace(3) %lds.const.gep4538, align 8, !dbg !16, !frisk.k_tail_pair !19
  %516 = load <4 x half>, ptr addrspace(3) %lds.const.gep4542, align 8, !dbg !16, !frisk.k_tail_pair !19
  %517 = load <4 x half>, ptr addrspace(3) %lds.const.gep4539, align 8, !dbg !16, !frisk.k_tail_pair !20
  %518 = load <4 x half>, ptr addrspace(3) %lds.const.gep4543, align 8, !dbg !16, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_tail_prefetch !5
  %519 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %511, <4 x float> %464) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !15, !frisk.k_tail_prefetch !5
  %520 = or disjoint i64 %209, %114, !dbg !15
  %521 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %513, <4 x float> %519) #8, !dbg !16
  %522 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %515, <4 x float> %470) #8, !dbg !16
  %523 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %517, <4 x float> %522) #8, !dbg !16
  %524 = load <4 x half>, ptr addrspace(3) %lds.const.gep4544, align 8, !dbg !16, !frisk.k_tail_pair !21
  %525 = load <4 x half>, ptr addrspace(3) %lds.const.gep4548, align 8, !dbg !16, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_tail_prefetch !5
  %526 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %512, <4 x float> %472) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_tail_prefetch !5
  %527 = load <4 x half>, ptr addrspace(3) %lds.const.gep4545, align 8, !dbg !16, !frisk.k_tail_pair !22
  %528 = load <4 x half>, ptr addrspace(3) %lds.const.gep4549, align 8, !dbg !16, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_tail_prefetch !5
  %529 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %514, <4 x float> %526) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_tail_prefetch !5
  %530 = load <4 x half>, ptr addrspace(3) %lds.const.gep4546, align 8, !dbg !16, !frisk.k_tail_pair !23
  %531 = load <4 x half>, ptr addrspace(3) %lds.const.gep4550, align 8, !dbg !16, !frisk.k_tail_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_tail_prefetch !5
  %532 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %516, <4 x float> %474) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_tail_prefetch !5
  %533 = load <4 x half>, ptr addrspace(3) %lds.const.gep4547, align 8, !dbg !16, !frisk.k_tail_pair !24
  %534 = load <4 x half>, ptr addrspace(3) %lds.const.gep4551, align 8, !dbg !16, !frisk.k_tail_pair !24
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_tail_prefetch !5
  %535 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %518, <4 x float> %532) #8, !dbg !16
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !16, !frisk.k_tail_prefetch !5
  %536 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %524, <4 x float> %476) #8, !dbg !16
  %537 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %527, <4 x float> %536) #8, !dbg !16
  %538 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %530, <4 x float> %478) #8, !dbg !16
  %539 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %533, <4 x float> %538) #8, !dbg !16
  %540 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %525, <4 x float> %480) #8, !dbg !16
  %541 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %528, <4 x float> %540) #8, !dbg !16
  %542 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %531, <4 x float> %482) #8, !dbg !16
  %543 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %534, <4 x float> %542) #8, !dbg !16
  %544 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %511, <4 x float> %484) #8, !dbg !16
  %545 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %513, <4 x float> %544) #8, !dbg !16
  %546 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %515, <4 x float> %486) #8, !dbg !16
  %547 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %517, <4 x float> %546) #8, !dbg !16
  %548 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %512, <4 x float> %488) #8, !dbg !16
  %549 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %514, <4 x float> %548) #8, !dbg !16
  %550 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %516, <4 x float> %490) #8, !dbg !16
  %551 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %518, <4 x float> %550) #8, !dbg !16
  %552 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %524, <4 x float> %492) #8, !dbg !16
  %553 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %527, <4 x float> %552) #8, !dbg !16
  %554 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %530, <4 x float> %494) #8, !dbg !16
  %555 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %533, <4 x float> %554) #8, !dbg !16
  %556 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %525, <4 x float> %496) #8, !dbg !16
  %557 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %528, <4 x float> %556) #8, !dbg !16
  %558 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %531, <4 x float> %498) #8, !dbg !16
  %559 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %534, <4 x float> %558) #8, !dbg !16
  %560 = trunc i64 %520 to i32, !dbg !15
  %561 = insertelement <32 x i32> poison, i32 %560, i64 0, !dbg !15
  %562 = shufflevector <32 x i32> %561, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !15
  %563 = or disjoint <32 x i32> %562, <i32 0, i32 4, i32 8, i32 12, i32 64, i32 68, i32 72, i32 76, i32 1, i32 5, i32 9, i32 13, i32 65, i32 69, i32 73, i32 77, i32 2, i32 6, i32 10, i32 14, i32 66, i32 70, i32 74, i32 78, i32 3, i32 7, i32 11, i32 15, i32 67, i32 71, i32 75, i32 79>, !dbg !15
  %564 = icmp ugt <32 x i32> %196, %563, !dbg !15
  %565 = insertelement <8 x i32> poison, i32 %560, i64 0, !dbg !15
  %566 = insertelement <4 x i32> poison, i32 %560, i64 0, !dbg !15
  %567 = or disjoint i32 %560, 12, !dbg !15
  %568 = or disjoint i32 %560, 8, !dbg !15
  %569 = or disjoint i32 %560, 4, !dbg !15
  %570 = insertelement <32 x i32> poison, i32 %568, i64 0, !dbg !15
  %571 = insertelement <32 x i32> %570, i32 %567, i64 1, !dbg !15
  %572 = insertelement <32 x i32> %561, i32 %569, i64 1, !dbg !15
  %573 = shufflevector <32 x i32> %572, <32 x i32> %563, <32 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !15
  %574 = shufflevector <8 x i32> %565, <8 x i32> poison, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %575 = or disjoint <32 x i32> %574, <i32 1, i32 5, i32 9, i32 13, i32 65, i32 69, i32 73, i32 77, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %576 = shufflevector <32 x i32> %573, <32 x i32> %575, <32 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !15
  %577 = shufflevector <4 x i32> %566, <4 x i32> poison, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %578 = or disjoint <32 x i32> %577, <i32 64, i32 68, i32 72, i32 76, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %579 = shufflevector <32 x i32> %576, <32 x i32> %578, <32 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 32, i32 33, i32 34, i32 35, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !15
  %580 = shufflevector <32 x i32> %579, <32 x i32> %571, <32 x i32> <i32 0, i32 1, i32 32, i32 33, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !15
  %581 = icmp ugt <32 x i32> %198, %580, !dbg !15
  %582 = shufflevector <4 x float> %521, <4 x float> %523, <32 x i32> <i32 0, i32 4, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 1, i32 5, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 2, i32 6, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 3, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %583 = shufflevector <4 x float> %529, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %584 = shufflevector <32 x float> %582, <32 x float> %583, <32 x i32> <i32 0, i32 1, i32 32, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %585 = shufflevector <4 x float> %535, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %586 = shufflevector <32 x float> %584, <32 x float> %585, <32 x i32> <i32 0, i32 1, i32 2, i32 32, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %587 = shufflevector <4 x float> %537, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %588 = shufflevector <32 x float> %586, <32 x float> %587, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 32, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %589 = shufflevector <4 x float> %539, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %590 = shufflevector <32 x float> %588, <32 x float> %589, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 32, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %591 = shufflevector <4 x float> %541, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %592 = shufflevector <32 x float> %590, <32 x float> %591, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 32, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %593 = shufflevector <4 x float> %543, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %594 = shufflevector <32 x float> %592, <32 x float> %593, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 32, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %595 = shufflevector <4 x float> %529, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %596 = shufflevector <32 x float> %594, <32 x float> %595, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 33, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %597 = shufflevector <4 x float> %535, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %598 = shufflevector <32 x float> %596, <32 x float> %597, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 33, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %599 = shufflevector <4 x float> %537, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %600 = shufflevector <32 x float> %598, <32 x float> %599, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 33, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %601 = shufflevector <4 x float> %539, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %602 = shufflevector <32 x float> %600, <32 x float> %601, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 33, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %603 = shufflevector <4 x float> %541, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %604 = shufflevector <32 x float> %602, <32 x float> %603, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 33, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %605 = shufflevector <4 x float> %543, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %606 = shufflevector <32 x float> %604, <32 x float> %605, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 33, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %607 = shufflevector <4 x float> %529, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %608 = shufflevector <32 x float> %606, <32 x float> %607, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 34, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %609 = shufflevector <4 x float> %535, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %610 = shufflevector <32 x float> %608, <32 x float> %609, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 34, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %611 = shufflevector <4 x float> %537, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %612 = shufflevector <32 x float> %610, <32 x float> %611, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 34, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %613 = shufflevector <4 x float> %539, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %614 = shufflevector <32 x float> %612, <32 x float> %613, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 34, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %615 = shufflevector <4 x float> %541, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %616 = shufflevector <32 x float> %614, <32 x float> %615, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 34, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %617 = shufflevector <4 x float> %543, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %618 = shufflevector <32 x float> %616, <32 x float> %617, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 34, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %619 = shufflevector <4 x float> %529, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %620 = shufflevector <32 x float> %618, <32 x float> %619, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %621 = shufflevector <4 x float> %535, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %622 = shufflevector <32 x float> %620, <32 x float> %621, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %623 = shufflevector <4 x float> %537, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %624 = shufflevector <32 x float> %622, <32 x float> %623, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 35, i32 poison, i32 poison, i32 poison>, !dbg !15
  %625 = shufflevector <4 x float> %539, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %626 = shufflevector <32 x float> %624, <32 x float> %625, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 35, i32 poison, i32 poison>, !dbg !15
  %627 = shufflevector <4 x float> %541, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %628 = shufflevector <32 x float> %626, <32 x float> %627, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 35, i32 poison>, !dbg !15
  %629 = shufflevector <4 x float> %543, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %630 = shufflevector <32 x float> %628, <32 x float> %629, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 35>, !dbg !15
  %631 = select <32 x i1> %581, <32 x float> %630, <32 x float> splat (float -inf), !dbg !15
  %632 = extractelement <32 x float> %631, i64 0, !dbg !7
  %633 = tail call float @llvm.maxnum.f32(float %632, float -inf), !dbg !7
  %634 = extractelement <32 x float> %631, i64 1, !dbg !7
  %635 = tail call float @llvm.maxnum.f32(float %633, float %634), !dbg !7
  %636 = extractelement <32 x float> %631, i64 8, !dbg !7
  %637 = extractelement <32 x float> %631, i64 9, !dbg !7
  %638 = tail call float @llvm.maxnum.f32(float %636, float %637), !dbg !7
  %639 = extractelement <32 x float> %631, i64 16, !dbg !7
  %640 = extractelement <32 x float> %631, i64 17, !dbg !7
  %641 = tail call float @llvm.maxnum.f32(float %639, float %640), !dbg !7
  %642 = extractelement <32 x float> %631, i64 24, !dbg !7
  %643 = extractelement <32 x float> %631, i64 25, !dbg !7
  %644 = tail call float @llvm.maxnum.f32(float %642, float %643), !dbg !7
  %645 = extractelement <32 x float> %631, i64 2, !dbg !7
  %646 = tail call float @llvm.maxnum.f32(float %635, float %645), !dbg !7
  %647 = extractelement <32 x float> %631, i64 10, !dbg !7
  %648 = tail call float @llvm.maxnum.f32(float %638, float %647), !dbg !7
  %649 = extractelement <32 x float> %631, i64 18, !dbg !7
  %650 = tail call float @llvm.maxnum.f32(float %641, float %649), !dbg !7
  %651 = extractelement <32 x float> %631, i64 26, !dbg !7
  %652 = tail call float @llvm.maxnum.f32(float %644, float %651), !dbg !7
  %653 = extractelement <32 x float> %631, i64 3, !dbg !7
  %654 = tail call float @llvm.maxnum.f32(float %646, float %653), !dbg !7
  %655 = extractelement <32 x float> %631, i64 11, !dbg !7
  %656 = tail call float @llvm.maxnum.f32(float %648, float %655), !dbg !7
  %657 = extractelement <32 x float> %631, i64 19, !dbg !7
  %658 = tail call float @llvm.maxnum.f32(float %650, float %657), !dbg !7
  %659 = extractelement <32 x float> %631, i64 27, !dbg !7
  %660 = tail call float @llvm.maxnum.f32(float %652, float %659), !dbg !7
  %661 = extractelement <32 x float> %631, i64 4, !dbg !7
  %662 = tail call float @llvm.maxnum.f32(float %654, float %661), !dbg !7
  %663 = extractelement <32 x float> %631, i64 12, !dbg !7
  %664 = tail call float @llvm.maxnum.f32(float %656, float %663), !dbg !7
  %665 = extractelement <32 x float> %631, i64 20, !dbg !7
  %666 = tail call float @llvm.maxnum.f32(float %658, float %665), !dbg !7
  %667 = extractelement <32 x float> %631, i64 28, !dbg !7
  %668 = tail call float @llvm.maxnum.f32(float %660, float %667), !dbg !7
  %669 = extractelement <32 x float> %631, i64 5, !dbg !7
  %670 = tail call float @llvm.maxnum.f32(float %662, float %669), !dbg !7
  %671 = extractelement <32 x float> %631, i64 13, !dbg !7
  %672 = tail call float @llvm.maxnum.f32(float %664, float %671), !dbg !7
  %673 = extractelement <32 x float> %631, i64 21, !dbg !7
  %674 = tail call float @llvm.maxnum.f32(float %666, float %673), !dbg !7
  %675 = extractelement <32 x float> %631, i64 29, !dbg !7
  %676 = tail call float @llvm.maxnum.f32(float %668, float %675), !dbg !7
  %677 = extractelement <32 x float> %631, i64 6, !dbg !7
  %678 = tail call float @llvm.maxnum.f32(float %670, float %677), !dbg !7
  %679 = extractelement <32 x float> %631, i64 14, !dbg !7
  %680 = tail call float @llvm.maxnum.f32(float %672, float %679), !dbg !7
  %681 = extractelement <32 x float> %631, i64 22, !dbg !7
  %682 = tail call float @llvm.maxnum.f32(float %674, float %681), !dbg !7
  %683 = extractelement <32 x float> %631, i64 30, !dbg !7
  %684 = tail call float @llvm.maxnum.f32(float %676, float %683), !dbg !7
  %685 = extractelement <32 x float> %631, i64 7, !dbg !7
  %686 = tail call float @llvm.maxnum.f32(float %678, float %685), !dbg !7
  %687 = extractelement <32 x float> %631, i64 15, !dbg !7
  %688 = tail call float @llvm.maxnum.f32(float %680, float %687), !dbg !7
  %689 = extractelement <32 x float> %631, i64 23, !dbg !7
  %690 = tail call float @llvm.maxnum.f32(float %682, float %689), !dbg !7
  %691 = extractelement <32 x float> %631, i64 31, !dbg !7
  %692 = tail call float @llvm.maxnum.f32(float %684, float %691), !dbg !7
  %693 = tail call float @llvm.maxnum.f32(float %686, float %688), !dbg !7
  %694 = tail call float @llvm.maxnum.f32(float %690, float %692), !dbg !7
  %695 = tail call float @llvm.maxnum.f32(float %693, float %694), !dbg !7
  %696 = shufflevector <4 x float> %545, <4 x float> %547, <32 x i32> <i32 0, i32 4, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 1, i32 5, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 2, i32 6, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 3, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %697 = shufflevector <4 x float> %549, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %698 = shufflevector <32 x float> %696, <32 x float> %697, <32 x i32> <i32 0, i32 1, i32 32, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %699 = shufflevector <4 x float> %551, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %700 = shufflevector <32 x float> %698, <32 x float> %699, <32 x i32> <i32 0, i32 1, i32 2, i32 32, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %701 = shufflevector <4 x float> %553, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %702 = shufflevector <32 x float> %700, <32 x float> %701, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 32, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %703 = shufflevector <4 x float> %555, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %704 = shufflevector <32 x float> %702, <32 x float> %703, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 32, i32 poison, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %705 = shufflevector <4 x float> %557, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %706 = shufflevector <32 x float> %704, <32 x float> %705, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 32, i32 poison, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %707 = shufflevector <4 x float> %559, <4 x float> poison, <32 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %708 = shufflevector <32 x float> %706, <32 x float> %707, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 32, i32 8, i32 9, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %709 = shufflevector <4 x float> %549, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %710 = shufflevector <32 x float> %708, <32 x float> %709, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 33, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %711 = shufflevector <4 x float> %551, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %712 = shufflevector <32 x float> %710, <32 x float> %711, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 33, i32 poison, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %713 = shufflevector <4 x float> %553, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %714 = shufflevector <32 x float> %712, <32 x float> %713, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 33, i32 poison, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %715 = shufflevector <4 x float> %555, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %716 = shufflevector <32 x float> %714, <32 x float> %715, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 33, i32 poison, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %717 = shufflevector <4 x float> %557, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %718 = shufflevector <32 x float> %716, <32 x float> %717, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 33, i32 poison, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %719 = shufflevector <4 x float> %559, <4 x float> poison, <32 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %720 = shufflevector <32 x float> %718, <32 x float> %719, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 33, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %721 = shufflevector <4 x float> %549, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %722 = shufflevector <32 x float> %720, <32 x float> %721, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 34, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %723 = shufflevector <4 x float> %551, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %724 = shufflevector <32 x float> %722, <32 x float> %723, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 34, i32 poison, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %725 = shufflevector <4 x float> %553, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %726 = shufflevector <32 x float> %724, <32 x float> %725, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 34, i32 poison, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %727 = shufflevector <4 x float> %555, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %728 = shufflevector <32 x float> %726, <32 x float> %727, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 34, i32 poison, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %729 = shufflevector <4 x float> %557, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %730 = shufflevector <32 x float> %728, <32 x float> %729, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 34, i32 poison, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %731 = shufflevector <4 x float> %559, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 2, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %732 = shufflevector <32 x float> %730, <32 x float> %731, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 34, i32 24, i32 25, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %733 = shufflevector <4 x float> %549, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %734 = shufflevector <32 x float> %732, <32 x float> %733, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %735 = shufflevector <4 x float> %551, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %736 = shufflevector <32 x float> %734, <32 x float> %735, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %737 = shufflevector <4 x float> %553, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %738 = shufflevector <32 x float> %736, <32 x float> %737, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 35, i32 poison, i32 poison, i32 poison>, !dbg !15
  %739 = shufflevector <4 x float> %555, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %740 = shufflevector <32 x float> %738, <32 x float> %739, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 35, i32 poison, i32 poison>, !dbg !15
  %741 = shufflevector <4 x float> %557, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %742 = shufflevector <32 x float> %740, <32 x float> %741, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 35, i32 poison>, !dbg !15
  %743 = shufflevector <4 x float> %559, <4 x float> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %744 = shufflevector <32 x float> %742, <32 x float> %743, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 35>, !dbg !15
  %745 = select <32 x i1> %564, <32 x float> %744, <32 x float> splat (float -inf), !dbg !15
  %746 = bitcast float %695 to i32, !dbg !7
  %747 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %746), !dbg !7
  %748 = bitcast i32 %747 to float, !dbg !7
  %749 = tail call float @llvm.maxnum.f32(float %695, float %748), !dbg !7
  %750 = bitcast float %749 to i32, !dbg !7
  %751 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %750), !dbg !7
  %752 = bitcast i32 %751 to float, !dbg !7
  %753 = tail call float @llvm.maxnum.f32(float %749, float %752), !dbg !7
  %754 = extractelement <32 x float> %745, i64 0, !dbg !7
  %755 = tail call float @llvm.maxnum.f32(float %754, float -inf), !dbg !7
  %756 = extractelement <32 x float> %745, i64 1, !dbg !7
  %757 = tail call float @llvm.maxnum.f32(float %755, float %756), !dbg !7
  %758 = extractelement <32 x float> %745, i64 8, !dbg !7
  %759 = extractelement <32 x float> %745, i64 9, !dbg !7
  %760 = tail call float @llvm.maxnum.f32(float %758, float %759), !dbg !7
  %761 = extractelement <32 x float> %745, i64 16, !dbg !7
  %762 = extractelement <32 x float> %745, i64 17, !dbg !7
  %763 = tail call float @llvm.maxnum.f32(float %761, float %762), !dbg !7
  %764 = extractelement <32 x float> %745, i64 24, !dbg !7
  %765 = extractelement <32 x float> %745, i64 25, !dbg !7
  %766 = tail call float @llvm.maxnum.f32(float %764, float %765), !dbg !7
  %767 = extractelement <32 x float> %745, i64 2, !dbg !7
  %768 = tail call float @llvm.maxnum.f32(float %757, float %767), !dbg !7
  %769 = extractelement <32 x float> %745, i64 10, !dbg !7
  %770 = tail call float @llvm.maxnum.f32(float %760, float %769), !dbg !7
  %771 = extractelement <32 x float> %745, i64 18, !dbg !7
  %772 = tail call float @llvm.maxnum.f32(float %763, float %771), !dbg !7
  %773 = extractelement <32 x float> %745, i64 26, !dbg !7
  %774 = tail call float @llvm.maxnum.f32(float %766, float %773), !dbg !7
  %775 = extractelement <32 x float> %745, i64 3, !dbg !7
  %776 = tail call float @llvm.maxnum.f32(float %768, float %775), !dbg !7
  %777 = extractelement <32 x float> %745, i64 11, !dbg !7
  %778 = tail call float @llvm.maxnum.f32(float %770, float %777), !dbg !7
  %779 = extractelement <32 x float> %745, i64 19, !dbg !7
  %780 = tail call float @llvm.maxnum.f32(float %772, float %779), !dbg !7
  %781 = extractelement <32 x float> %745, i64 27, !dbg !7
  %782 = tail call float @llvm.maxnum.f32(float %774, float %781), !dbg !7
  %783 = extractelement <32 x float> %745, i64 4, !dbg !7
  %784 = tail call float @llvm.maxnum.f32(float %776, float %783), !dbg !7
  %785 = extractelement <32 x float> %745, i64 12, !dbg !7
  %786 = tail call float @llvm.maxnum.f32(float %778, float %785), !dbg !7
  %787 = extractelement <32 x float> %745, i64 20, !dbg !7
  %788 = tail call float @llvm.maxnum.f32(float %780, float %787), !dbg !7
  %789 = extractelement <32 x float> %745, i64 28, !dbg !7
  %790 = tail call float @llvm.maxnum.f32(float %782, float %789), !dbg !7
  %791 = extractelement <32 x float> %745, i64 5, !dbg !7
  %792 = tail call float @llvm.maxnum.f32(float %784, float %791), !dbg !7
  %793 = extractelement <32 x float> %745, i64 13, !dbg !7
  %794 = tail call float @llvm.maxnum.f32(float %786, float %793), !dbg !7
  %795 = extractelement <32 x float> %745, i64 21, !dbg !7
  %796 = tail call float @llvm.maxnum.f32(float %788, float %795), !dbg !7
  %797 = extractelement <32 x float> %745, i64 29, !dbg !7
  %798 = tail call float @llvm.maxnum.f32(float %790, float %797), !dbg !7
  %799 = extractelement <32 x float> %745, i64 6, !dbg !7
  %800 = tail call float @llvm.maxnum.f32(float %792, float %799), !dbg !7
  %801 = extractelement <32 x float> %745, i64 14, !dbg !7
  %802 = tail call float @llvm.maxnum.f32(float %794, float %801), !dbg !7
  %803 = extractelement <32 x float> %745, i64 22, !dbg !7
  %804 = tail call float @llvm.maxnum.f32(float %796, float %803), !dbg !7
  %805 = extractelement <32 x float> %745, i64 30, !dbg !7
  %806 = tail call float @llvm.maxnum.f32(float %798, float %805), !dbg !7
  %807 = extractelement <32 x float> %745, i64 7, !dbg !7
  %808 = tail call float @llvm.maxnum.f32(float %800, float %807), !dbg !7
  %809 = extractelement <32 x float> %745, i64 15, !dbg !7
  %810 = tail call float @llvm.maxnum.f32(float %802, float %809), !dbg !7
  %811 = extractelement <32 x float> %745, i64 23, !dbg !7
  %812 = tail call float @llvm.maxnum.f32(float %804, float %811), !dbg !7
  %813 = extractelement <32 x float> %745, i64 31, !dbg !7
  %814 = tail call float @llvm.maxnum.f32(float %806, float %813), !dbg !7
  %815 = tail call float @llvm.maxnum.f32(float %808, float %810), !dbg !7
  %816 = tail call float @llvm.maxnum.f32(float %812, float %814), !dbg !7
  %817 = tail call float @llvm.maxnum.f32(float %815, float %816), !dbg !7
  %818 = bitcast float %817 to i32, !dbg !7
  %819 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %818), !dbg !7
  %820 = bitcast i32 %819 to float, !dbg !7
  %821 = tail call float @llvm.maxnum.f32(float %817, float %820), !dbg !7
  %822 = bitcast float %821 to i32, !dbg !7
  %823 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %822), !dbg !7
  %824 = bitcast i32 %823 to float, !dbg !7
  %825 = tail call float @llvm.maxnum.f32(float %821, float %824), !dbg !7
  %826 = extractvalue [2 x <1 x float>] %204, 0, !dbg !7
  %827 = insertelement <1 x float> poison, float %753, i64 0, !dbg !7
  %828 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %827, <1 x float> %826), !dbg !7
  %829 = extractvalue [2 x <1 x float>] %204, 1, !dbg !7
  %830 = insertelement <1 x float> poison, float %825, i64 0, !dbg !7
  %831 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %830, <1 x float> %829), !dbg !7
  %832 = shufflevector <32 x float> %631, <32 x float> poison, <4 x i32> <i32 0, i32 8, i32 16, i32 24>, !dbg !42
  %833 = shufflevector <1 x float> %828, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !42
  %834 = shufflevector <32 x float> %631, <32 x float> poison, <4 x i32> <i32 1, i32 9, i32 17, i32 25>, !dbg !42
  %835 = shufflevector <32 x float> %631, <32 x float> poison, <4 x i32> <i32 2, i32 10, i32 18, i32 26>, !dbg !42
  %836 = shufflevector <32 x float> %631, <32 x float> poison, <4 x i32> <i32 3, i32 11, i32 19, i32 27>, !dbg !42
  %837 = shufflevector <32 x float> %631, <32 x float> poison, <4 x i32> <i32 4, i32 12, i32 20, i32 28>, !dbg !42
  %838 = shufflevector <32 x float> %631, <32 x float> poison, <4 x i32> <i32 5, i32 13, i32 21, i32 29>, !dbg !42
  %839 = shufflevector <32 x float> %631, <32 x float> poison, <4 x i32> <i32 6, i32 14, i32 22, i32 30>, !dbg !42
  %840 = shufflevector <32 x float> %631, <32 x float> poison, <4 x i32> <i32 7, i32 15, i32 23, i32 31>, !dbg !42
  %841 = shufflevector <32 x float> %745, <32 x float> poison, <4 x i32> <i32 0, i32 8, i32 16, i32 24>, !dbg !42
  %842 = shufflevector <1 x float> %831, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !42
  %843 = shufflevector <32 x float> %745, <32 x float> poison, <4 x i32> <i32 1, i32 9, i32 17, i32 25>, !dbg !42
  %844 = shufflevector <32 x float> %745, <32 x float> poison, <4 x i32> <i32 2, i32 10, i32 18, i32 26>, !dbg !42
  %845 = shufflevector <32 x float> %745, <32 x float> poison, <4 x i32> <i32 3, i32 11, i32 19, i32 27>, !dbg !42
  %846 = shufflevector <32 x float> %745, <32 x float> poison, <4 x i32> <i32 4, i32 12, i32 20, i32 28>, !dbg !42
  %847 = shufflevector <32 x float> %745, <32 x float> poison, <4 x i32> <i32 5, i32 13, i32 21, i32 29>, !dbg !42
  %848 = shufflevector <32 x float> %745, <32 x float> poison, <4 x i32> <i32 6, i32 14, i32 22, i32 30>, !dbg !42
  %849 = shufflevector <32 x float> %745, <32 x float> poison, <4 x i32> <i32 7, i32 15, i32 23, i32 31>, !dbg !42
  %850 = fsub <1 x float> %826, %828, !dbg !43
  %851 = extractelement <1 x float> %850, i64 0, !dbg !44
  %852 = tail call float @llvm.amdgcn.exp2.f32(float %851), !dbg !44
  %853 = insertelement <4 x float> poison, float %852, i64 0, !dbg !45
  %854 = shufflevector <4 x float> %853, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !45
  %855 = extractvalue [2 x <1 x float>] %205, 0, !dbg !46
  %856 = extractelement <1 x float> %855, i64 0, !dbg !46
  %857 = fmul float %856, %852, !dbg !46
  %858 = fsub <1 x float> %829, %831, !dbg !43
  %859 = extractelement <1 x float> %858, i64 0, !dbg !44
  %860 = tail call float @llvm.amdgcn.exp2.f32(float %859), !dbg !44
  %861 = insertelement <4 x float> poison, float %860, i64 0, !dbg !45
  %862 = shufflevector <4 x float> %861, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !45
  %863 = extractvalue [2 x <1 x float>] %205, 1, !dbg !46
  %864 = extractelement <1 x float> %863, i64 0, !dbg !46
  %865 = fmul float %864, %860, !dbg !46
  %866 = extractvalue [2 x <8 x float>] %206, 0, !dbg !45
  %867 = shufflevector <8 x float> %866, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %868 = fmul <4 x float> %867, %854, !dbg !45
  %869 = shufflevector <8 x float> %866, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %870 = fmul <4 x float> %869, %854, !dbg !45
  %871 = extractvalue [2 x <8 x float>] %206, 1, !dbg !45
  %872 = shufflevector <8 x float> %871, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %873 = fmul <4 x float> %872, %862, !dbg !45
  %874 = shufflevector <8 x float> %871, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %875 = fmul <4 x float> %874, %862, !dbg !45
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  tail call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %address.local3929 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local3255, !dbg !41
  %address.local3930 = ptrtoint ptr addrspace(3) %address.local3929 to i64, !dbg !41
  %address.local3931 = trunc i64 %address.local3930 to i32, !dbg !41
  %876 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3931), !dbg !41
  %dma.buffer.offset4640 = or i32 %dma.buffer.offset4644, 64, !dbg !41
  %877 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %876, i32 %dma.buffer.offset4640, <4 x i32> %dma.buffer.rsrc4645) #9, !dbg !41
  %address.local3941 = getelementptr i8, ptr addrspace(3) %address.local3929, i64 4096, !dbg !41
  %address.local3942 = ptrtoint ptr addrspace(3) %address.local3941 to i64, !dbg !41
  %address.local3943 = trunc i64 %address.local3942 to i32, !dbg !41
  %878 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3943), !dbg !41
  %dma.buffer.offset4638 = or i32 %509, 96, !dbg !41
  %879 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %878, i32 %dma.buffer.offset4638, <4 x i32> %dma.buffer.rsrc4645) #9, !dbg !41
  %address.local3950 = shl nuw nsw i32 %address.thread1199, 3, !dbg !47
  %address.local3951 = and i32 %address.local3950, 384, !dbg !47
  %address.local3953 = shl nuw nsw i32 %address.thread1199, 10, !dbg !47
  %address.local3954 = and i32 %address.local3953, 3072, !dbg !47
  %880 = or disjoint i32 %address.local3951, %address.local3954, !dbg !47
  %address.local3955 = or disjoint i32 %880, %address.local1314, !dbg !47
  %address.local3956 = zext nneg i32 %address.local3955 to i64, !dbg !47
  %address.local3957 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local3956, !dbg !47
  %881 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3957, i32 0) #11, !dbg !47
  %882 = bitcast <4 x i32> %881 to <8 x half>, !dbg !47
  %address.local3968 = getelementptr i8, ptr addrspace(3) %address.local3957, i64 64, !dbg !47
  %883 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3968, i32 0) #11, !dbg !47
  %884 = bitcast <4 x i32> %883 to <8 x half>, !dbg !47
  %address.local3979 = getelementptr i8, ptr addrspace(3) %address.local3957, i64 128, !dbg !47
  %885 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3979, i32 0) #11, !dbg !47
  %886 = bitcast <4 x i32> %885 to <8 x half>, !dbg !47
  %address.local3990 = getelementptr i8, ptr addrspace(3) %address.local3957, i64 192, !dbg !47
  %887 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3990, i32 0) #11, !dbg !47
  %888 = bitcast <4 x i32> %887 to <8 x half>, !dbg !47
  %address.local4001 = getelementptr i8, ptr addrspace(3) %address.local3957, i64 1024, !dbg !47
  %889 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4001, i32 0) #11, !dbg !47
  %890 = bitcast <4 x i32> %889 to <8 x half>, !dbg !47
  %address.local4012 = getelementptr i8, ptr addrspace(3) %address.local3957, i64 1088, !dbg !47
  %891 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4012, i32 0) #11, !dbg !47
  %892 = bitcast <4 x i32> %891 to <8 x half>, !dbg !47
  %address.local4023 = getelementptr i8, ptr addrspace(3) %address.local3957, i64 1152, !dbg !47
  %893 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4023, i32 0) #11, !dbg !47
  %894 = bitcast <4 x i32> %893 to <8 x half>, !dbg !47
  %address.local4034 = getelementptr i8, ptr addrspace(3) %address.local3957, i64 1216, !dbg !47
  %895 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4034, i32 0) #11, !dbg !47
  %896 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %897 = fsub <4 x float> %832, %833, !dbg !42
  %898 = extractelement <4 x float> %897, i64 0, !dbg !48
  %899 = tail call float @llvm.amdgcn.exp2.f32(float %898), !dbg !48
  %900 = insertelement <4 x float> poison, float %899, i64 0, !dbg !49
  %901 = extractelement <4 x float> %897, i64 1, !dbg !48
  %902 = tail call float @llvm.amdgcn.exp2.f32(float %901), !dbg !48
  %903 = insertelement <4 x float> %900, float %902, i64 1, !dbg !49
  %904 = extractelement <4 x float> %897, i64 2, !dbg !48
  %905 = tail call float @llvm.amdgcn.exp2.f32(float %904), !dbg !48
  %906 = insertelement <4 x float> %903, float %905, i64 2, !dbg !49
  %907 = extractelement <4 x float> %897, i64 3, !dbg !48
  %908 = tail call float @llvm.amdgcn.exp2.f32(float %907), !dbg !48
  %909 = insertelement <4 x float> %906, float %908, i64 3, !dbg !49
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %909) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i64 0, !dbg !49
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i64 1, !dbg !49
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i64 2, !dbg !49
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i64 3, !dbg !49
  %.cvtpack.p0 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1) #12, !dbg !49
  %.cvtpack.p1 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3) #12, !dbg !49
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i64 0, !dbg !49
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i64 1, !dbg !49
  %910 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !49
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %910) #8, !dbg !48, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %911 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %896, <4 x float> %868) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %912 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %913 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %912, <4 x float> %870) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %914 = fsub <4 x float> %841, %842, !dbg !42
  %915 = extractelement <4 x float> %914, i64 0, !dbg !48
  %916 = tail call float @llvm.amdgcn.exp2.f32(float %915), !dbg !48
  %917 = insertelement <4 x float> poison, float %916, i64 0, !dbg !49
  %918 = extractelement <4 x float> %914, i64 1, !dbg !48
  %919 = tail call float @llvm.amdgcn.exp2.f32(float %918), !dbg !48
  %920 = insertelement <4 x float> %917, float %919, i64 1, !dbg !49
  %921 = extractelement <4 x float> %914, i64 2, !dbg !48
  %922 = tail call float @llvm.amdgcn.exp2.f32(float %921), !dbg !48
  %923 = insertelement <4 x float> %920, float %922, i64 2, !dbg !49
  %924 = extractelement <4 x float> %914, i64 3, !dbg !48
  %925 = tail call float @llvm.amdgcn.exp2.f32(float %924), !dbg !48
  %926 = insertelement <4 x float> %923, float %925, i64 3, !dbg !49
  %math.result925 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %926) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04857 = extractelement <4 x float> %math.result925, i64 0, !dbg !49
  %.cvtpack.f14858 = extractelement <4 x float> %math.result925, i64 1, !dbg !49
  %.cvtpack.f24859 = extractelement <4 x float> %math.result925, i64 2, !dbg !49
  %.cvtpack.f34860 = extractelement <4 x float> %math.result925, i64 3, !dbg !49
  %.cvtpack.p04861 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04857, float %.cvtpack.f14858) #12, !dbg !49
  %.cvtpack.p14862 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24859, float %.cvtpack.f34860) #12, !dbg !49
  %.cvtpack.v04863 = insertelement <2 x i32> poison, i32 %.cvtpack.p04861, i64 0, !dbg !49
  %.cvtpack.v14864 = insertelement <2 x i32> %.cvtpack.v04863, i32 %.cvtpack.p14862, i64 1, !dbg !49
  %927 = bitcast <2 x i32> %.cvtpack.v14864 to <4 x half>, !dbg !49
  %math.packed926 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %927) #8, !dbg !48, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %928 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %896, <4 x float> %873) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %929 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %912, <4 x float> %875) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %930 = shufflevector <8 x half> %884, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %931 = fsub <4 x float> %834, %833, !dbg !42
  %932 = extractelement <4 x float> %931, i64 0, !dbg !48
  %933 = tail call float @llvm.amdgcn.exp2.f32(float %932), !dbg !48
  %934 = fadd float %899, %933, !dbg !50
  %935 = insertelement <4 x float> poison, float %933, i64 0, !dbg !49
  %936 = extractelement <4 x float> %931, i64 1, !dbg !48
  %937 = tail call float @llvm.amdgcn.exp2.f32(float %936), !dbg !48
  %938 = fadd float %902, %937, !dbg !50
  %939 = insertelement <4 x float> %935, float %937, i64 1, !dbg !49
  %940 = extractelement <4 x float> %931, i64 2, !dbg !48
  %941 = tail call float @llvm.amdgcn.exp2.f32(float %940), !dbg !48
  %942 = fadd float %905, %941, !dbg !50
  %943 = insertelement <4 x float> %939, float %941, i64 2, !dbg !49
  %944 = extractelement <4 x float> %931, i64 3, !dbg !48
  %945 = tail call float @llvm.amdgcn.exp2.f32(float %944), !dbg !48
  %946 = fadd float %908, %945, !dbg !50
  %947 = insertelement <4 x float> %943, float %945, i64 3, !dbg !49
  %math.result911 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %947) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04865 = extractelement <4 x float> %math.result911, i64 0, !dbg !49
  %.cvtpack.f14866 = extractelement <4 x float> %math.result911, i64 1, !dbg !49
  %.cvtpack.f24867 = extractelement <4 x float> %math.result911, i64 2, !dbg !49
  %.cvtpack.f34868 = extractelement <4 x float> %math.result911, i64 3, !dbg !49
  %.cvtpack.p04869 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04865, float %.cvtpack.f14866) #12, !dbg !49
  %.cvtpack.p14870 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24867, float %.cvtpack.f34868) #12, !dbg !49
  %.cvtpack.v04871 = insertelement <2 x i32> poison, i32 %.cvtpack.p04869, i64 0, !dbg !49
  %.cvtpack.v14872 = insertelement <2 x i32> %.cvtpack.v04871, i32 %.cvtpack.p14870, i64 1, !dbg !49
  %948 = bitcast <2 x i32> %.cvtpack.v14872 to <4 x half>, !dbg !49
  %math.packed912 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %948) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %949 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %930, <4 x float> %911) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %950 = shufflevector <8 x half> %884, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %951 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %950, <4 x float> %913) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %952 = fsub <4 x float> %843, %842, !dbg !42
  %953 = extractelement <4 x float> %952, i64 0, !dbg !48
  %954 = tail call float @llvm.amdgcn.exp2.f32(float %953), !dbg !48
  %955 = fadd float %916, %954, !dbg !50
  %956 = insertelement <4 x float> poison, float %954, i64 0, !dbg !49
  %957 = extractelement <4 x float> %952, i64 1, !dbg !48
  %958 = tail call float @llvm.amdgcn.exp2.f32(float %957), !dbg !48
  %959 = fadd float %919, %958, !dbg !50
  %960 = insertelement <4 x float> %956, float %958, i64 1, !dbg !49
  %961 = extractelement <4 x float> %952, i64 2, !dbg !48
  %962 = tail call float @llvm.amdgcn.exp2.f32(float %961), !dbg !48
  %963 = fadd float %922, %962, !dbg !50
  %964 = insertelement <4 x float> %960, float %962, i64 2, !dbg !49
  %965 = extractelement <4 x float> %952, i64 3, !dbg !48
  %966 = tail call float @llvm.amdgcn.exp2.f32(float %965), !dbg !48
  %967 = fadd float %925, %966, !dbg !50
  %968 = insertelement <4 x float> %964, float %966, i64 3, !dbg !49
  %math.result927 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %968) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04873 = extractelement <4 x float> %math.result927, i64 0, !dbg !49
  %.cvtpack.f14874 = extractelement <4 x float> %math.result927, i64 1, !dbg !49
  %.cvtpack.f24875 = extractelement <4 x float> %math.result927, i64 2, !dbg !49
  %.cvtpack.f34876 = extractelement <4 x float> %math.result927, i64 3, !dbg !49
  %.cvtpack.p04877 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04873, float %.cvtpack.f14874) #12, !dbg !49
  %.cvtpack.p14878 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24875, float %.cvtpack.f34876) #12, !dbg !49
  %.cvtpack.v04879 = insertelement <2 x i32> poison, i32 %.cvtpack.p04877, i64 0, !dbg !49
  %.cvtpack.v14880 = insertelement <2 x i32> %.cvtpack.v04879, i32 %.cvtpack.p14878, i64 1, !dbg !49
  %969 = bitcast <2 x i32> %.cvtpack.v14880 to <4 x half>, !dbg !49
  %math.packed928 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %969) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %930, <4 x float> %928) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %971 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %950, <4 x float> %929) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %972 = shufflevector <8 x half> %886, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %973 = fsub <4 x float> %835, %833, !dbg !42
  %974 = extractelement <4 x float> %973, i64 0, !dbg !48
  %975 = tail call float @llvm.amdgcn.exp2.f32(float %974), !dbg !48
  %976 = fadd float %975, %934, !dbg !50
  %977 = insertelement <4 x float> poison, float %975, i64 0, !dbg !49
  %978 = extractelement <4 x float> %973, i64 1, !dbg !48
  %979 = tail call float @llvm.amdgcn.exp2.f32(float %978), !dbg !48
  %980 = fadd float %979, %938, !dbg !50
  %981 = insertelement <4 x float> %977, float %979, i64 1, !dbg !49
  %982 = extractelement <4 x float> %973, i64 2, !dbg !48
  %983 = tail call float @llvm.amdgcn.exp2.f32(float %982), !dbg !48
  %984 = fadd float %983, %942, !dbg !50
  %985 = insertelement <4 x float> %981, float %983, i64 2, !dbg !49
  %986 = extractelement <4 x float> %973, i64 3, !dbg !48
  %987 = tail call float @llvm.amdgcn.exp2.f32(float %986), !dbg !48
  %988 = fadd float %987, %946, !dbg !50
  %989 = insertelement <4 x float> %985, float %987, i64 3, !dbg !49
  %math.result913 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %989) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04881 = extractelement <4 x float> %math.result913, i64 0, !dbg !49
  %.cvtpack.f14882 = extractelement <4 x float> %math.result913, i64 1, !dbg !49
  %.cvtpack.f24883 = extractelement <4 x float> %math.result913, i64 2, !dbg !49
  %.cvtpack.f34884 = extractelement <4 x float> %math.result913, i64 3, !dbg !49
  %.cvtpack.p04885 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04881, float %.cvtpack.f14882) #12, !dbg !49
  %.cvtpack.p14886 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24883, float %.cvtpack.f34884) #12, !dbg !49
  %.cvtpack.v04887 = insertelement <2 x i32> poison, i32 %.cvtpack.p04885, i64 0, !dbg !49
  %.cvtpack.v14888 = insertelement <2 x i32> %.cvtpack.v04887, i32 %.cvtpack.p14886, i64 1, !dbg !49
  %990 = bitcast <2 x i32> %.cvtpack.v14888 to <4 x half>, !dbg !49
  %math.packed914 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %990) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %991 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %972, <4 x float> %949) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %992 = shufflevector <8 x half> %886, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %993 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %992, <4 x float> %951) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %994 = fsub <4 x float> %844, %842, !dbg !42
  %995 = extractelement <4 x float> %994, i64 0, !dbg !48
  %996 = tail call float @llvm.amdgcn.exp2.f32(float %995), !dbg !48
  %997 = fadd float %996, %955, !dbg !50
  %998 = insertelement <4 x float> poison, float %996, i64 0, !dbg !49
  %999 = extractelement <4 x float> %994, i64 1, !dbg !48
  %1000 = tail call float @llvm.amdgcn.exp2.f32(float %999), !dbg !48
  %1001 = fadd float %1000, %959, !dbg !50
  %1002 = insertelement <4 x float> %998, float %1000, i64 1, !dbg !49
  %1003 = extractelement <4 x float> %994, i64 2, !dbg !48
  %1004 = tail call float @llvm.amdgcn.exp2.f32(float %1003), !dbg !48
  %1005 = fadd float %1004, %963, !dbg !50
  %1006 = insertelement <4 x float> %1002, float %1004, i64 2, !dbg !49
  %1007 = extractelement <4 x float> %994, i64 3, !dbg !48
  %1008 = tail call float @llvm.amdgcn.exp2.f32(float %1007), !dbg !48
  %1009 = fadd float %1008, %967, !dbg !50
  %1010 = insertelement <4 x float> %1006, float %1008, i64 3, !dbg !49
  %math.result929 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1010) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04889 = extractelement <4 x float> %math.result929, i64 0, !dbg !49
  %.cvtpack.f14890 = extractelement <4 x float> %math.result929, i64 1, !dbg !49
  %.cvtpack.f24891 = extractelement <4 x float> %math.result929, i64 2, !dbg !49
  %.cvtpack.f34892 = extractelement <4 x float> %math.result929, i64 3, !dbg !49
  %.cvtpack.p04893 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04889, float %.cvtpack.f14890) #12, !dbg !49
  %.cvtpack.p14894 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24891, float %.cvtpack.f34892) #12, !dbg !49
  %.cvtpack.v04895 = insertelement <2 x i32> poison, i32 %.cvtpack.p04893, i64 0, !dbg !49
  %.cvtpack.v14896 = insertelement <2 x i32> %.cvtpack.v04895, i32 %.cvtpack.p14894, i64 1, !dbg !49
  %1011 = bitcast <2 x i32> %.cvtpack.v14896 to <4 x half>, !dbg !49
  %math.packed930 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1011) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %972, <4 x float> %970) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1013 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %992, <4 x float> %971) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1014 = shufflevector <8 x half> %888, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1015 = fsub <4 x float> %836, %833, !dbg !42
  %1016 = extractelement <4 x float> %1015, i64 0, !dbg !48
  %1017 = tail call float @llvm.amdgcn.exp2.f32(float %1016), !dbg !48
  %1018 = fadd float %1017, %976, !dbg !50
  %1019 = insertelement <4 x float> poison, float %1017, i64 0, !dbg !49
  %1020 = extractelement <4 x float> %1015, i64 1, !dbg !48
  %1021 = tail call float @llvm.amdgcn.exp2.f32(float %1020), !dbg !48
  %1022 = fadd float %1021, %980, !dbg !50
  %1023 = insertelement <4 x float> %1019, float %1021, i64 1, !dbg !49
  %1024 = extractelement <4 x float> %1015, i64 2, !dbg !48
  %1025 = tail call float @llvm.amdgcn.exp2.f32(float %1024), !dbg !48
  %1026 = fadd float %1025, %984, !dbg !50
  %1027 = insertelement <4 x float> %1023, float %1025, i64 2, !dbg !49
  %1028 = extractelement <4 x float> %1015, i64 3, !dbg !48
  %1029 = tail call float @llvm.amdgcn.exp2.f32(float %1028), !dbg !48
  %1030 = fadd float %1029, %988, !dbg !50
  %1031 = insertelement <4 x float> %1027, float %1029, i64 3, !dbg !49
  %math.result915 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1031) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04897 = extractelement <4 x float> %math.result915, i64 0, !dbg !49
  %.cvtpack.f14898 = extractelement <4 x float> %math.result915, i64 1, !dbg !49
  %.cvtpack.f24899 = extractelement <4 x float> %math.result915, i64 2, !dbg !49
  %.cvtpack.f34900 = extractelement <4 x float> %math.result915, i64 3, !dbg !49
  %.cvtpack.p04901 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04897, float %.cvtpack.f14898) #12, !dbg !49
  %.cvtpack.p14902 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24899, float %.cvtpack.f34900) #12, !dbg !49
  %.cvtpack.v04903 = insertelement <2 x i32> poison, i32 %.cvtpack.p04901, i64 0, !dbg !49
  %.cvtpack.v14904 = insertelement <2 x i32> %.cvtpack.v04903, i32 %.cvtpack.p14902, i64 1, !dbg !49
  %1032 = bitcast <2 x i32> %.cvtpack.v14904 to <4 x half>, !dbg !49
  %math.packed916 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1032) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %1033 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1014, <4 x float> %991) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1034 = shufflevector <8 x half> %888, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1035 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1034, <4 x float> %993) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1036 = fsub <4 x float> %845, %842, !dbg !42
  %1037 = extractelement <4 x float> %1036, i64 0, !dbg !48
  %1038 = tail call float @llvm.amdgcn.exp2.f32(float %1037), !dbg !48
  %1039 = fadd float %1038, %997, !dbg !50
  %1040 = insertelement <4 x float> poison, float %1038, i64 0, !dbg !49
  %1041 = extractelement <4 x float> %1036, i64 1, !dbg !48
  %1042 = tail call float @llvm.amdgcn.exp2.f32(float %1041), !dbg !48
  %1043 = fadd float %1042, %1001, !dbg !50
  %1044 = insertelement <4 x float> %1040, float %1042, i64 1, !dbg !49
  %1045 = extractelement <4 x float> %1036, i64 2, !dbg !48
  %1046 = tail call float @llvm.amdgcn.exp2.f32(float %1045), !dbg !48
  %1047 = fadd float %1046, %1005, !dbg !50
  %1048 = insertelement <4 x float> %1044, float %1046, i64 2, !dbg !49
  %1049 = extractelement <4 x float> %1036, i64 3, !dbg !48
  %1050 = tail call float @llvm.amdgcn.exp2.f32(float %1049), !dbg !48
  %1051 = fadd float %1050, %1009, !dbg !50
  %1052 = insertelement <4 x float> %1048, float %1050, i64 3, !dbg !49
  %math.result931 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1052) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04905 = extractelement <4 x float> %math.result931, i64 0, !dbg !49
  %.cvtpack.f14906 = extractelement <4 x float> %math.result931, i64 1, !dbg !49
  %.cvtpack.f24907 = extractelement <4 x float> %math.result931, i64 2, !dbg !49
  %.cvtpack.f34908 = extractelement <4 x float> %math.result931, i64 3, !dbg !49
  %.cvtpack.p04909 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04905, float %.cvtpack.f14906) #12, !dbg !49
  %.cvtpack.p14910 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24907, float %.cvtpack.f34908) #12, !dbg !49
  %.cvtpack.v04911 = insertelement <2 x i32> poison, i32 %.cvtpack.p04909, i64 0, !dbg !49
  %.cvtpack.v14912 = insertelement <2 x i32> %.cvtpack.v04911, i32 %.cvtpack.p14910, i64 1, !dbg !49
  %1053 = bitcast <2 x i32> %.cvtpack.v14912 to <4 x half>, !dbg !49
  %math.packed932 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1053) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %1054 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1014, <4 x float> %1012) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1055 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1034, <4 x float> %1013) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1056 = shufflevector <8 x half> %890, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1057 = fsub <4 x float> %837, %833, !dbg !42
  %1058 = extractelement <4 x float> %1057, i64 0, !dbg !48
  %1059 = tail call float @llvm.amdgcn.exp2.f32(float %1058), !dbg !48
  %1060 = fadd float %1059, %1018, !dbg !50
  %1061 = insertelement <4 x float> poison, float %1059, i64 0, !dbg !49
  %1062 = extractelement <4 x float> %1057, i64 1, !dbg !48
  %1063 = tail call float @llvm.amdgcn.exp2.f32(float %1062), !dbg !48
  %1064 = fadd float %1063, %1022, !dbg !50
  %1065 = insertelement <4 x float> %1061, float %1063, i64 1, !dbg !49
  %1066 = extractelement <4 x float> %1057, i64 2, !dbg !48
  %1067 = tail call float @llvm.amdgcn.exp2.f32(float %1066), !dbg !48
  %1068 = fadd float %1067, %1026, !dbg !50
  %1069 = insertelement <4 x float> %1065, float %1067, i64 2, !dbg !49
  %1070 = extractelement <4 x float> %1057, i64 3, !dbg !48
  %1071 = tail call float @llvm.amdgcn.exp2.f32(float %1070), !dbg !48
  %1072 = fadd float %1071, %1030, !dbg !50
  %1073 = insertelement <4 x float> %1069, float %1071, i64 3, !dbg !49
  %math.result917 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1073) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04913 = extractelement <4 x float> %math.result917, i64 0, !dbg !49
  %.cvtpack.f14914 = extractelement <4 x float> %math.result917, i64 1, !dbg !49
  %.cvtpack.f24915 = extractelement <4 x float> %math.result917, i64 2, !dbg !49
  %.cvtpack.f34916 = extractelement <4 x float> %math.result917, i64 3, !dbg !49
  %.cvtpack.p04917 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04913, float %.cvtpack.f14914) #12, !dbg !49
  %.cvtpack.p14918 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24915, float %.cvtpack.f34916) #12, !dbg !49
  %.cvtpack.v04919 = insertelement <2 x i32> poison, i32 %.cvtpack.p04917, i64 0, !dbg !49
  %.cvtpack.v14920 = insertelement <2 x i32> %.cvtpack.v04919, i32 %.cvtpack.p14918, i64 1, !dbg !49
  %1074 = bitcast <2 x i32> %.cvtpack.v14920 to <4 x half>, !dbg !49
  %math.packed918 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1074) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %1075 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1056, <4 x float> %1033) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1076 = shufflevector <8 x half> %890, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1077 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1076, <4 x float> %1035) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1078 = fsub <4 x float> %846, %842, !dbg !42
  %1079 = extractelement <4 x float> %1078, i64 0, !dbg !48
  %1080 = tail call float @llvm.amdgcn.exp2.f32(float %1079), !dbg !48
  %1081 = fadd float %1080, %1039, !dbg !50
  %1082 = insertelement <4 x float> poison, float %1080, i64 0, !dbg !49
  %1083 = extractelement <4 x float> %1078, i64 1, !dbg !48
  %1084 = tail call float @llvm.amdgcn.exp2.f32(float %1083), !dbg !48
  %1085 = fadd float %1084, %1043, !dbg !50
  %1086 = insertelement <4 x float> %1082, float %1084, i64 1, !dbg !49
  %1087 = extractelement <4 x float> %1078, i64 2, !dbg !48
  %1088 = tail call float @llvm.amdgcn.exp2.f32(float %1087), !dbg !48
  %1089 = fadd float %1088, %1047, !dbg !50
  %1090 = insertelement <4 x float> %1086, float %1088, i64 2, !dbg !49
  %1091 = extractelement <4 x float> %1078, i64 3, !dbg !48
  %1092 = tail call float @llvm.amdgcn.exp2.f32(float %1091), !dbg !48
  %1093 = fadd float %1092, %1051, !dbg !50
  %1094 = insertelement <4 x float> %1090, float %1092, i64 3, !dbg !49
  %math.result933 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1094) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04921 = extractelement <4 x float> %math.result933, i64 0, !dbg !49
  %.cvtpack.f14922 = extractelement <4 x float> %math.result933, i64 1, !dbg !49
  %.cvtpack.f24923 = extractelement <4 x float> %math.result933, i64 2, !dbg !49
  %.cvtpack.f34924 = extractelement <4 x float> %math.result933, i64 3, !dbg !49
  %.cvtpack.p04925 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04921, float %.cvtpack.f14922) #12, !dbg !49
  %.cvtpack.p14926 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24923, float %.cvtpack.f34924) #12, !dbg !49
  %.cvtpack.v04927 = insertelement <2 x i32> poison, i32 %.cvtpack.p04925, i64 0, !dbg !49
  %.cvtpack.v14928 = insertelement <2 x i32> %.cvtpack.v04927, i32 %.cvtpack.p14926, i64 1, !dbg !49
  %1095 = bitcast <2 x i32> %.cvtpack.v14928 to <4 x half>, !dbg !49
  %math.packed934 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1095) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %1096 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1056, <4 x float> %1054) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1097 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1076, <4 x float> %1055) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1098 = shufflevector <8 x half> %892, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1099 = fsub <4 x float> %838, %833, !dbg !42
  %1100 = extractelement <4 x float> %1099, i64 0, !dbg !48
  %1101 = tail call float @llvm.amdgcn.exp2.f32(float %1100), !dbg !48
  %1102 = fadd float %1101, %1060, !dbg !50
  %1103 = insertelement <4 x float> poison, float %1101, i64 0, !dbg !49
  %1104 = extractelement <4 x float> %1099, i64 1, !dbg !48
  %1105 = tail call float @llvm.amdgcn.exp2.f32(float %1104), !dbg !48
  %1106 = fadd float %1105, %1064, !dbg !50
  %1107 = insertelement <4 x float> %1103, float %1105, i64 1, !dbg !49
  %1108 = extractelement <4 x float> %1099, i64 2, !dbg !48
  %1109 = tail call float @llvm.amdgcn.exp2.f32(float %1108), !dbg !48
  %1110 = fadd float %1109, %1068, !dbg !50
  %1111 = insertelement <4 x float> %1107, float %1109, i64 2, !dbg !49
  %1112 = extractelement <4 x float> %1099, i64 3, !dbg !48
  %1113 = tail call float @llvm.amdgcn.exp2.f32(float %1112), !dbg !48
  %1114 = fadd float %1113, %1072, !dbg !50
  %1115 = insertelement <4 x float> %1111, float %1113, i64 3, !dbg !49
  %math.result919 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1115) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04929 = extractelement <4 x float> %math.result919, i64 0, !dbg !49
  %.cvtpack.f14930 = extractelement <4 x float> %math.result919, i64 1, !dbg !49
  %.cvtpack.f24931 = extractelement <4 x float> %math.result919, i64 2, !dbg !49
  %.cvtpack.f34932 = extractelement <4 x float> %math.result919, i64 3, !dbg !49
  %.cvtpack.p04933 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04929, float %.cvtpack.f14930) #12, !dbg !49
  %.cvtpack.p14934 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24931, float %.cvtpack.f34932) #12, !dbg !49
  %.cvtpack.v04935 = insertelement <2 x i32> poison, i32 %.cvtpack.p04933, i64 0, !dbg !49
  %.cvtpack.v14936 = insertelement <2 x i32> %.cvtpack.v04935, i32 %.cvtpack.p14934, i64 1, !dbg !49
  %1116 = bitcast <2 x i32> %.cvtpack.v14936 to <4 x half>, !dbg !49
  %math.packed920 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1116) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %1117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1098, <4 x float> %1075) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1118 = shufflevector <8 x half> %892, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1119 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1118, <4 x float> %1077) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1120 = fsub <4 x float> %847, %842, !dbg !42
  %1121 = extractelement <4 x float> %1120, i64 0, !dbg !48
  %1122 = tail call float @llvm.amdgcn.exp2.f32(float %1121), !dbg !48
  %1123 = fadd float %1122, %1081, !dbg !50
  %1124 = insertelement <4 x float> poison, float %1122, i64 0, !dbg !49
  %1125 = extractelement <4 x float> %1120, i64 1, !dbg !48
  %1126 = tail call float @llvm.amdgcn.exp2.f32(float %1125), !dbg !48
  %1127 = fadd float %1126, %1085, !dbg !50
  %1128 = insertelement <4 x float> %1124, float %1126, i64 1, !dbg !49
  %1129 = extractelement <4 x float> %1120, i64 2, !dbg !48
  %1130 = tail call float @llvm.amdgcn.exp2.f32(float %1129), !dbg !48
  %1131 = fadd float %1130, %1089, !dbg !50
  %1132 = insertelement <4 x float> %1128, float %1130, i64 2, !dbg !49
  %1133 = extractelement <4 x float> %1120, i64 3, !dbg !48
  %1134 = tail call float @llvm.amdgcn.exp2.f32(float %1133), !dbg !48
  %1135 = fadd float %1134, %1093, !dbg !50
  %1136 = insertelement <4 x float> %1132, float %1134, i64 3, !dbg !49
  %math.result935 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1136) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04937 = extractelement <4 x float> %math.result935, i64 0, !dbg !49
  %.cvtpack.f14938 = extractelement <4 x float> %math.result935, i64 1, !dbg !49
  %.cvtpack.f24939 = extractelement <4 x float> %math.result935, i64 2, !dbg !49
  %.cvtpack.f34940 = extractelement <4 x float> %math.result935, i64 3, !dbg !49
  %.cvtpack.p04941 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04937, float %.cvtpack.f14938) #12, !dbg !49
  %.cvtpack.p14942 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24939, float %.cvtpack.f34940) #12, !dbg !49
  %.cvtpack.v04943 = insertelement <2 x i32> poison, i32 %.cvtpack.p04941, i64 0, !dbg !49
  %.cvtpack.v14944 = insertelement <2 x i32> %.cvtpack.v04943, i32 %.cvtpack.p14942, i64 1, !dbg !49
  %1137 = bitcast <2 x i32> %.cvtpack.v14944 to <4 x half>, !dbg !49
  %math.packed936 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1137) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %1138 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1098, <4 x float> %1096) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1139 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1118, <4 x float> %1097) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1140 = shufflevector <8 x half> %894, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1141 = fsub <4 x float> %839, %833, !dbg !42
  %1142 = extractelement <4 x float> %1141, i64 0, !dbg !48
  %1143 = tail call float @llvm.amdgcn.exp2.f32(float %1142), !dbg !48
  %1144 = fadd float %1143, %1102, !dbg !50
  %1145 = insertelement <4 x float> poison, float %1143, i64 0, !dbg !49
  %1146 = extractelement <4 x float> %1141, i64 1, !dbg !48
  %1147 = tail call float @llvm.amdgcn.exp2.f32(float %1146), !dbg !48
  %1148 = fadd float %1147, %1106, !dbg !50
  %1149 = insertelement <4 x float> %1145, float %1147, i64 1, !dbg !49
  %1150 = extractelement <4 x float> %1141, i64 2, !dbg !48
  %1151 = tail call float @llvm.amdgcn.exp2.f32(float %1150), !dbg !48
  %1152 = fadd float %1151, %1110, !dbg !50
  %1153 = insertelement <4 x float> %1149, float %1151, i64 2, !dbg !49
  %1154 = extractelement <4 x float> %1141, i64 3, !dbg !48
  %1155 = tail call float @llvm.amdgcn.exp2.f32(float %1154), !dbg !48
  %1156 = fadd float %1155, %1114, !dbg !50
  %1157 = insertelement <4 x float> %1153, float %1155, i64 3, !dbg !49
  %math.result921 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1157) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04945 = extractelement <4 x float> %math.result921, i64 0, !dbg !49
  %.cvtpack.f14946 = extractelement <4 x float> %math.result921, i64 1, !dbg !49
  %.cvtpack.f24947 = extractelement <4 x float> %math.result921, i64 2, !dbg !49
  %.cvtpack.f34948 = extractelement <4 x float> %math.result921, i64 3, !dbg !49
  %.cvtpack.p04949 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04945, float %.cvtpack.f14946) #12, !dbg !49
  %.cvtpack.p14950 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24947, float %.cvtpack.f34948) #12, !dbg !49
  %.cvtpack.v04951 = insertelement <2 x i32> poison, i32 %.cvtpack.p04949, i64 0, !dbg !49
  %.cvtpack.v14952 = insertelement <2 x i32> %.cvtpack.v04951, i32 %.cvtpack.p14950, i64 1, !dbg !49
  %1158 = bitcast <2 x i32> %.cvtpack.v14952 to <4 x half>, !dbg !49
  %math.packed922 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1158) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %1159 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1140, <4 x float> %1117) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1160 = shufflevector <8 x half> %894, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1160, <4 x float> %1119) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1162 = fsub <4 x float> %848, %842, !dbg !42
  %1163 = extractelement <4 x float> %1162, i64 0, !dbg !48
  %1164 = tail call float @llvm.amdgcn.exp2.f32(float %1163), !dbg !48
  %1165 = fadd float %1164, %1123, !dbg !50
  %1166 = insertelement <4 x float> poison, float %1164, i64 0, !dbg !49
  %1167 = extractelement <4 x float> %1162, i64 1, !dbg !48
  %1168 = tail call float @llvm.amdgcn.exp2.f32(float %1167), !dbg !48
  %1169 = fadd float %1168, %1127, !dbg !50
  %1170 = insertelement <4 x float> %1166, float %1168, i64 1, !dbg !49
  %1171 = extractelement <4 x float> %1162, i64 2, !dbg !48
  %1172 = tail call float @llvm.amdgcn.exp2.f32(float %1171), !dbg !48
  %1173 = fadd float %1172, %1131, !dbg !50
  %1174 = insertelement <4 x float> %1170, float %1172, i64 2, !dbg !49
  %1175 = extractelement <4 x float> %1162, i64 3, !dbg !48
  %1176 = tail call float @llvm.amdgcn.exp2.f32(float %1175), !dbg !48
  %1177 = fadd float %1176, %1135, !dbg !50
  %1178 = insertelement <4 x float> %1174, float %1176, i64 3, !dbg !49
  %math.result937 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1178) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04953 = extractelement <4 x float> %math.result937, i64 0, !dbg !49
  %.cvtpack.f14954 = extractelement <4 x float> %math.result937, i64 1, !dbg !49
  %.cvtpack.f24955 = extractelement <4 x float> %math.result937, i64 2, !dbg !49
  %.cvtpack.f34956 = extractelement <4 x float> %math.result937, i64 3, !dbg !49
  %.cvtpack.p04957 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04953, float %.cvtpack.f14954) #12, !dbg !49
  %.cvtpack.p14958 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24955, float %.cvtpack.f34956) #12, !dbg !49
  %.cvtpack.v04959 = insertelement <2 x i32> poison, i32 %.cvtpack.p04957, i64 0, !dbg !49
  %.cvtpack.v14960 = insertelement <2 x i32> %.cvtpack.v04959, i32 %.cvtpack.p14958, i64 1, !dbg !49
  %1179 = bitcast <2 x i32> %.cvtpack.v14960 to <4 x half>, !dbg !49
  %math.packed938 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1179) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %1180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1140, <4 x float> %1138) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1160, <4 x float> %1139) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1182 = bitcast <4 x i32> %895 to <8 x half>, !dbg !47
  %1183 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1184 = fsub <4 x float> %840, %833, !dbg !42
  %1185 = extractelement <4 x float> %1184, i64 0, !dbg !48
  %1186 = tail call float @llvm.amdgcn.exp2.f32(float %1185), !dbg !48
  %1187 = fadd float %1186, %1144, !dbg !50
  %1188 = insertelement <4 x float> poison, float %1186, i64 0, !dbg !49
  %1189 = extractelement <4 x float> %1184, i64 1, !dbg !48
  %1190 = tail call float @llvm.amdgcn.exp2.f32(float %1189), !dbg !48
  %1191 = fadd float %1190, %1148, !dbg !50
  %1192 = fadd float %1187, %1191, !dbg !50
  %1193 = insertelement <4 x float> %1188, float %1190, i64 1, !dbg !49
  %1194 = extractelement <4 x float> %1184, i64 2, !dbg !48
  %1195 = tail call float @llvm.amdgcn.exp2.f32(float %1194), !dbg !48
  %1196 = fadd float %1195, %1152, !dbg !50
  %1197 = insertelement <4 x float> %1193, float %1195, i64 2, !dbg !49
  %1198 = extractelement <4 x float> %1184, i64 3, !dbg !48
  %1199 = tail call float @llvm.amdgcn.exp2.f32(float %1198), !dbg !48
  %1200 = fadd float %1199, %1156, !dbg !50
  %1201 = fadd float %1196, %1200, !dbg !50
  %1202 = fadd float %1192, %1201, !dbg !50
  %1203 = fadd float %857, %1202, !dbg !50
  %1204 = insertelement <1 x float> poison, float %1203, i64 0, !dbg !12
  %1205 = insertvalue [2 x <1 x float>] poison, <1 x float> %1204, 0, !dbg !12
  %1206 = insertelement <4 x float> %1197, float %1199, i64 3, !dbg !49
  %math.result923 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1206) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04961 = extractelement <4 x float> %math.result923, i64 0, !dbg !49
  %.cvtpack.f14962 = extractelement <4 x float> %math.result923, i64 1, !dbg !49
  %.cvtpack.f24963 = extractelement <4 x float> %math.result923, i64 2, !dbg !49
  %.cvtpack.f34964 = extractelement <4 x float> %math.result923, i64 3, !dbg !49
  %.cvtpack.p04965 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04961, float %.cvtpack.f14962) #12, !dbg !49
  %.cvtpack.p14966 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24963, float %.cvtpack.f34964) #12, !dbg !49
  %.cvtpack.v04967 = insertelement <2 x i32> poison, i32 %.cvtpack.p04965, i64 0, !dbg !49
  %.cvtpack.v14968 = insertelement <2 x i32> %.cvtpack.v04967, i32 %.cvtpack.p14966, i64 1, !dbg !49
  %1207 = bitcast <2 x i32> %.cvtpack.v14968 to <4 x half>, !dbg !49
  %math.packed924 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1207) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %1208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1183, <4 x float> %1159) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1209 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1209, <4 x float> %1161) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1211 = fsub <4 x float> %849, %842, !dbg !42
  %1212 = extractelement <4 x float> %1211, i64 0, !dbg !48
  %1213 = tail call float @llvm.amdgcn.exp2.f32(float %1212), !dbg !48
  %1214 = fadd float %1213, %1165, !dbg !50
  %1215 = insertelement <4 x float> poison, float %1213, i64 0, !dbg !49
  %1216 = extractelement <4 x float> %1211, i64 1, !dbg !48
  %1217 = tail call float @llvm.amdgcn.exp2.f32(float %1216), !dbg !48
  %1218 = fadd float %1217, %1169, !dbg !50
  %1219 = fadd float %1214, %1218, !dbg !50
  %1220 = insertelement <4 x float> %1215, float %1217, i64 1, !dbg !49
  %1221 = extractelement <4 x float> %1211, i64 2, !dbg !48
  %1222 = tail call float @llvm.amdgcn.exp2.f32(float %1221), !dbg !48
  %1223 = fadd float %1222, %1173, !dbg !50
  %1224 = insertelement <4 x float> %1220, float %1222, i64 2, !dbg !49
  %1225 = extractelement <4 x float> %1211, i64 3, !dbg !48
  %1226 = tail call float @llvm.amdgcn.exp2.f32(float %1225), !dbg !48
  %1227 = fadd float %1226, %1177, !dbg !50
  %1228 = fadd float %1223, %1227, !dbg !50
  %1229 = fadd float %1219, %1228, !dbg !50
  %1230 = fadd float %865, %1229, !dbg !50
  %1231 = insertelement <1 x float> poison, float %1230, i64 0, !dbg !12
  %1232 = insertvalue [2 x <1 x float>] %1205, <1 x float> %1231, 1, !dbg !12
  %1233 = insertelement <4 x float> %1224, float %1226, i64 3, !dbg !49
  %math.result939 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1233) #8, !dbg !49, !frisk.register_lifetimes !5
  %.cvtpack.f04969 = extractelement <4 x float> %math.result939, i64 0, !dbg !49
  %.cvtpack.f14970 = extractelement <4 x float> %math.result939, i64 1, !dbg !49
  %.cvtpack.f24971 = extractelement <4 x float> %math.result939, i64 2, !dbg !49
  %.cvtpack.f34972 = extractelement <4 x float> %math.result939, i64 3, !dbg !49
  %.cvtpack.p04973 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04969, float %.cvtpack.f14970) #12, !dbg !49
  %.cvtpack.p14974 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24971, float %.cvtpack.f34972) #12, !dbg !49
  %.cvtpack.v04975 = insertelement <2 x i32> poison, i32 %.cvtpack.p04973, i64 0, !dbg !49
  %.cvtpack.v14976 = insertelement <2 x i32> %.cvtpack.v04975, i32 %.cvtpack.p14974, i64 1, !dbg !49
  %1234 = bitcast <2 x i32> %.cvtpack.v14976 to <4 x half>, !dbg !49
  %math.packed940 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1234) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47, !frisk.softmax_fragments !5
  %1235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1183, <4 x float> %1180) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1209, <4 x float> %1181) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1237 = extractvalue [2 x <8 x float>] %203, 0, !dbg !45
  %1238 = shufflevector <8 x float> %1237, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1239 = fmul <4 x float> %1238, %854, !dbg !45
  %1240 = shufflevector <8 x float> %1237, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1241 = fmul <4 x float> %1240, %854, !dbg !45
  %1242 = extractvalue [2 x <8 x float>] %203, 1, !dbg !45
  %1243 = shufflevector <8 x float> %1242, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1244 = fmul <4 x float> %1243, %862, !dbg !45
  %1245 = shufflevector <8 x float> %1242, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1246 = fmul <4 x float> %1245, %862, !dbg !45
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  tail call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %dma.buffer.offset4636 = or i32 %dma.buffer.offset4644, 128, !dbg !41
  %1247 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %502, i32 %dma.buffer.offset4636, <4 x i32> %dma.buffer.rsrc4645) #9, !dbg !41
  %dma.buffer.offset4634 = or i32 %509, 160, !dbg !41
  %1248 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %506, i32 %dma.buffer.offset4634, <4 x i32> %dma.buffer.rsrc4645) #9, !dbg !41
  %address.local4067 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local3956, !dbg !47
  %1249 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4067, i32 0) #11, !dbg !47
  %1250 = bitcast <4 x i32> %1249 to <8 x half>, !dbg !47
  %address.local4078 = getelementptr i8, ptr addrspace(3) %address.local4067, i64 64, !dbg !47
  %1251 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4078, i32 0) #11, !dbg !47
  %1252 = bitcast <4 x i32> %1251 to <8 x half>, !dbg !47
  %address.local4089 = getelementptr i8, ptr addrspace(3) %address.local4067, i64 128, !dbg !47
  %1253 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4089, i32 0) #11, !dbg !47
  %1254 = bitcast <4 x i32> %1253 to <8 x half>, !dbg !47
  %address.local4100 = getelementptr i8, ptr addrspace(3) %address.local4067, i64 192, !dbg !47
  %1255 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4100, i32 0) #11, !dbg !47
  %1256 = bitcast <4 x i32> %1255 to <8 x half>, !dbg !47
  %address.local4111 = getelementptr i8, ptr addrspace(3) %address.local4067, i64 1024, !dbg !47
  %1257 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4111, i32 0) #11, !dbg !47
  %1258 = bitcast <4 x i32> %1257 to <8 x half>, !dbg !47
  %address.local4122 = getelementptr i8, ptr addrspace(3) %address.local4067, i64 1088, !dbg !47
  %1259 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4122, i32 0) #11, !dbg !47
  %1260 = bitcast <4 x i32> %1259 to <8 x half>, !dbg !47
  %address.local4133 = getelementptr i8, ptr addrspace(3) %address.local4067, i64 1152, !dbg !47
  %1261 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4133, i32 0) #11, !dbg !47
  %1262 = bitcast <4 x i32> %1261 to <8 x half>, !dbg !47
  %address.local4144 = getelementptr i8, ptr addrspace(3) %address.local4067, i64 1216, !dbg !47
  %1263 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4144, i32 0) #11, !dbg !47
  %1264 = shufflevector <8 x half> %1250, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1264, <4 x float> %1239) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1266 = shufflevector <8 x half> %1250, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1266, <4 x float> %1241) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %1264, <4 x float> %1244) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %1266, <4 x float> %1246) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1270 = shufflevector <8 x half> %1252, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %1270, <4 x float> %1265) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1272 = shufflevector <8 x half> %1252, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %1272, <4 x float> %1267) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %1270, <4 x float> %1268) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %1272, <4 x float> %1269) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1276 = shufflevector <8 x half> %1254, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %1276, <4 x float> %1271) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1278 = shufflevector <8 x half> %1254, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %1278, <4 x float> %1273) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %1276, <4 x float> %1274) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %1278, <4 x float> %1275) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1282 = shufflevector <8 x half> %1256, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1282, <4 x float> %1277) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1284 = shufflevector <8 x half> %1256, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1284, <4 x float> %1279) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1282, <4 x float> %1280) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1284, <4 x float> %1281) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1288 = shufflevector <8 x half> %1258, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1288, <4 x float> %1283) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1290 = shufflevector <8 x half> %1258, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1290, <4 x float> %1285) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1288, <4 x float> %1286) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1290, <4 x float> %1287) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1294 = shufflevector <8 x half> %1260, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1295 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1294, <4 x float> %1289) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1296 = shufflevector <8 x half> %1260, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1296, <4 x float> %1291) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1294, <4 x float> %1292) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1296, <4 x float> %1293) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1300 = shufflevector <8 x half> %1262, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1300, <4 x float> %1295) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1302 = shufflevector <8 x half> %1262, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1302, <4 x float> %1297) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1300, <4 x float> %1298) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1302, <4 x float> %1299) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1306 = bitcast <4 x i32> %1263 to <8 x half>, !dbg !47
  %1307 = shufflevector <8 x half> %1306, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1307, <4 x float> %1301) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1309 = shufflevector <8 x half> %1306, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1309, <4 x float> %1303) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1307, <4 x float> %1304) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1309, <4 x float> %1305) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1313 = extractvalue [2 x <8 x float>] %202, 0, !dbg !45
  %1314 = shufflevector <8 x float> %1313, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1315 = fmul <4 x float> %1314, %854, !dbg !45
  %1316 = shufflevector <8 x float> %1313, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1317 = fmul <4 x float> %1316, %854, !dbg !45
  %1318 = extractvalue [2 x <8 x float>] %202, 1, !dbg !45
  %1319 = shufflevector <8 x float> %1318, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1320 = fmul <4 x float> %1319, %862, !dbg !45
  %1321 = shufflevector <8 x float> %1318, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1322 = fmul <4 x float> %1321, %862, !dbg !45
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  tail call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %dma.buffer.offset4632 = or i32 %dma.buffer.offset4644, 192, !dbg !41
  %1323 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %876, i32 %dma.buffer.offset4632, <4 x i32> %dma.buffer.rsrc4645) #9, !dbg !41
  %dma.buffer.offset = or i32 %509, 224, !dbg !41
  %1324 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %878, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc4645) #9, !dbg !41
  %1325 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3957, i32 0) #11, !dbg !47
  %1326 = bitcast <4 x i32> %1325 to <8 x half>, !dbg !47
  %address.local4186 = or disjoint i32 %address.local3955, 32, !dbg !47
  %address.local4187 = zext nneg i32 %address.local4186 to i64, !dbg !47
  %address.local4188 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4187, !dbg !47
  %1327 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4188, i32 0) #11, !dbg !47
  %1328 = bitcast <4 x i32> %1327 to <8 x half>, !dbg !47
  %address.local4197 = or disjoint i32 %address.local3955, 64, !dbg !47
  %address.local4198 = zext nneg i32 %address.local4197 to i64, !dbg !47
  %address.local4199 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4198, !dbg !47
  %1329 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4199, i32 0) #11, !dbg !47
  %1330 = bitcast <4 x i32> %1329 to <8 x half>, !dbg !47
  %address.local4208 = or disjoint i32 %address.local3955, 96, !dbg !47
  %address.local4209 = zext nneg i32 %address.local4208 to i64, !dbg !47
  %address.local4210 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4209, !dbg !47
  %1331 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4210, i32 0) #11, !dbg !47
  %1332 = bitcast <4 x i32> %1331 to <8 x half>, !dbg !47
  %address.local4219 = or disjoint i32 %address.local3955, 512, !dbg !47
  %address.local4220 = zext nneg i32 %address.local4219 to i64, !dbg !47
  %address.local4221 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4220, !dbg !47
  %1333 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4221, i32 0) #11, !dbg !47
  %1334 = bitcast <4 x i32> %1333 to <8 x half>, !dbg !47
  %address.local4230 = or disjoint i32 %address.local3955, 544, !dbg !47
  %address.local4231 = zext nneg i32 %address.local4230 to i64, !dbg !47
  %address.local4232 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4231, !dbg !47
  %1335 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4232, i32 0) #11, !dbg !47
  %1336 = bitcast <4 x i32> %1335 to <8 x half>, !dbg !47
  %address.local4241 = or disjoint i32 %address.local3955, 576, !dbg !47
  %address.local4242 = zext nneg i32 %address.local4241 to i64, !dbg !47
  %address.local4243 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4242, !dbg !47
  %1337 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4243, i32 0) #11, !dbg !47
  %1338 = bitcast <4 x i32> %1337 to <8 x half>, !dbg !47
  %address.local4252 = or disjoint i32 %address.local3955, 608, !dbg !47
  %address.local4253 = zext nneg i32 %address.local4252 to i64, !dbg !47
  %address.local4254 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4253, !dbg !47
  %1339 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4254, i32 0) #11, !dbg !47
  %1340 = shufflevector <8 x half> %1326, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1340, <4 x float> %1315) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1342 = shufflevector <8 x half> %1326, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1342, <4 x float> %1317) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %1340, <4 x float> %1320) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %1342, <4 x float> %1322) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1346 = shufflevector <8 x half> %1328, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %1346, <4 x float> %1341) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1348 = shufflevector <8 x half> %1328, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %1348, <4 x float> %1343) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %1346, <4 x float> %1344) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %1348, <4 x float> %1345) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1352 = shufflevector <8 x half> %1330, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %1352, <4 x float> %1347) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1354 = shufflevector <8 x half> %1330, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %1354, <4 x float> %1349) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %1352, <4 x float> %1350) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %1354, <4 x float> %1351) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1358 = shufflevector <8 x half> %1332, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1358, <4 x float> %1353) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1360 = shufflevector <8 x half> %1332, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1360, <4 x float> %1355) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1358, <4 x float> %1356) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1360, <4 x float> %1357) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1364 = shufflevector <8 x half> %1334, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1364, <4 x float> %1359) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1366 = shufflevector <8 x half> %1334, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1366, <4 x float> %1361) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1364, <4 x float> %1362) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1366, <4 x float> %1363) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1370 = shufflevector <8 x half> %1336, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1371 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1370, <4 x float> %1365) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1372 = shufflevector <8 x half> %1336, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1372, <4 x float> %1367) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1370, <4 x float> %1368) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1375 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1372, <4 x float> %1369) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1376 = shufflevector <8 x half> %1338, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1376, <4 x float> %1371) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1378 = shufflevector <8 x half> %1338, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1378, <4 x float> %1373) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1376, <4 x float> %1374) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1378, <4 x float> %1375) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1382 = bitcast <4 x i32> %1339 to <8 x half>, !dbg !47
  %1383 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1383, <4 x float> %1377) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1385 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1385, <4 x float> %1379) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1383, <4 x float> %1380) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1385, <4 x float> %1381) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1389 = extractvalue [2 x <8 x float>] %201, 0, !dbg !45
  %1390 = shufflevector <8 x float> %1389, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1391 = fmul <4 x float> %1390, %854, !dbg !45
  %1392 = shufflevector <8 x float> %1389, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1393 = fmul <4 x float> %1392, %854, !dbg !45
  %1394 = extractvalue [2 x <8 x float>] %201, 1, !dbg !45
  %1395 = shufflevector <8 x float> %1394, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %1396 = fmul <4 x float> %1395, %862, !dbg !45
  %1397 = shufflevector <8 x float> %1394, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %1398 = fmul <4 x float> %1397, %862, !dbg !45
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  tail call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %1399 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4067, i32 0) #11, !dbg !47
  %1400 = bitcast <4 x i32> %1399 to <8 x half>, !dbg !47
  %address.local4256 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4187, !dbg !47
  %1401 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4256, i32 0) #11, !dbg !47
  %1402 = bitcast <4 x i32> %1401 to <8 x half>, !dbg !47
  %address.local4257 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4198, !dbg !47
  %1403 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4257, i32 0) #11, !dbg !47
  %1404 = bitcast <4 x i32> %1403 to <8 x half>, !dbg !47
  %address.local4258 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4209, !dbg !47
  %1405 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4258, i32 0) #11, !dbg !47
  %1406 = bitcast <4 x i32> %1405 to <8 x half>, !dbg !47
  %address.local4259 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4220, !dbg !47
  %1407 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4259, i32 0) #11, !dbg !47
  %1408 = bitcast <4 x i32> %1407 to <8 x half>, !dbg !47
  %address.local4260 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4231, !dbg !47
  %1409 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4260, i32 0) #11, !dbg !47
  %1410 = bitcast <4 x i32> %1409 to <8 x half>, !dbg !47
  %address.local4261 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4242, !dbg !47
  %1411 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4261, i32 0) #11, !dbg !47
  %1412 = bitcast <4 x i32> %1411 to <8 x half>, !dbg !47
  %address.local4262 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4253, !dbg !47
  %1413 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4262, i32 0) #11, !dbg !47
  %1414 = shufflevector <8 x half> %1400, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1414, <4 x float> %1391) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1416 = shufflevector <8 x half> %1400, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1417 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1416, <4 x float> %1393) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %1414, <4 x float> %1396) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1419 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %1416, <4 x float> %1398) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1420 = shufflevector <8 x half> %1402, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1421 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %1420, <4 x float> %1415) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1422 = shufflevector <8 x half> %1402, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %1422, <4 x float> %1417) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1424 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %1420, <4 x float> %1418) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1425 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %1422, <4 x float> %1419) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1426 = shufflevector <8 x half> %1404, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1427 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %1426, <4 x float> %1421) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1428 = shufflevector <8 x half> %1404, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1429 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %1428, <4 x float> %1423) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1430 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %1426, <4 x float> %1424) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1431 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %1428, <4 x float> %1425) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1432 = shufflevector <8 x half> %1406, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1433 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1432, <4 x float> %1427) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1434 = shufflevector <8 x half> %1406, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1435 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1434, <4 x float> %1429) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1436 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1432, <4 x float> %1430) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1437 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1434, <4 x float> %1431) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1438 = shufflevector <8 x half> %1408, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1439 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1438, <4 x float> %1433) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1440 = shufflevector <8 x half> %1408, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1440, <4 x float> %1435) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1438, <4 x float> %1436) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1443 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1440, <4 x float> %1437) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1444 = shufflevector <8 x half> %1410, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1445 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1444, <4 x float> %1439) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1446 = shufflevector <8 x half> %1410, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1446, <4 x float> %1441) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1444, <4 x float> %1442) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1449 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1446, <4 x float> %1443) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1450 = shufflevector <8 x half> %1412, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1451 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1450, <4 x float> %1445) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1452 = shufflevector <8 x half> %1412, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1452, <4 x float> %1447) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1454 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1450, <4 x float> %1448) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1455 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1452, <4 x float> %1449) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1456 = bitcast <4 x i32> %1413 to <8 x half>, !dbg !47
  %1457 = shufflevector <8 x half> %1456, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1458 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1457, <4 x float> %1451) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1459 = shufflevector <8 x half> %1456, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1460 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1459, <4 x float> %1453) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1461 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1457, <4 x float> %1454) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1462 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1459, <4 x float> %1455) #8, !dbg !47, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %1463 = shufflevector <4 x float> %1208, <4 x float> %1210, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1464 = insertvalue [2 x <8 x float>] poison, <8 x float> %1463, 0, !dbg !12
  %1465 = shufflevector <4 x float> %1235, <4 x float> %1236, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1466 = insertvalue [2 x <8 x float>] %1464, <8 x float> %1465, 1, !dbg !12
  %1467 = insertvalue [2 x <1 x float>] poison, <1 x float> %828, 0, !dbg !12
  %1468 = insertvalue [2 x <1 x float>] %1467, <1 x float> %831, 1, !dbg !12
  %1469 = shufflevector <4 x float> %1308, <4 x float> %1310, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1470 = insertvalue [2 x <8 x float>] poison, <8 x float> %1469, 0, !dbg !12
  %1471 = shufflevector <4 x float> %1311, <4 x float> %1312, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1472 = insertvalue [2 x <8 x float>] %1470, <8 x float> %1471, 1, !dbg !12
  %1473 = shufflevector <4 x float> %1384, <4 x float> %1386, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1474 = insertvalue [2 x <8 x float>] poison, <8 x float> %1473, 0, !dbg !12
  %1475 = shufflevector <4 x float> %1387, <4 x float> %1388, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1476 = insertvalue [2 x <8 x float>] %1474, <8 x float> %1475, 1, !dbg !12
  %1477 = shufflevector <4 x float> %1458, <4 x float> %1460, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1478 = insertvalue [2 x <8 x float>] poison, <8 x float> %1477, 0, !dbg !12
  %1479 = shufflevector <4 x float> %1461, <4 x float> %1462, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1480 = insertvalue [2 x <8 x float>] %1478, <8 x float> %1479, 1, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %1481 = add nuw nsw i64 %207, 1, !dbg !12
  %1482 = icmp slt i64 %1481, %110, !dbg !12
  br i1 %1482, label %200, label %._crit_edge, !dbg !12

._crit_edge:                                      ; preds = %200, %4
  %.lcssa14 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1466, %200 ], !dbg !12
  %.lcssa13 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %1232, %200 ], !dbg !12
  %.lcssa12 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1472, %200 ], !dbg !12
  %.lcssa11 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1476, %200 ], !dbg !12
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1480, %200 ], !dbg !12
  %1483 = extractvalue [2 x <1 x float>] %.lcssa13, 1, !dbg !12
  %bc = bitcast <1 x float> %1483 to <1 x i32>, !dbg !12
  %1484 = extractelement <1 x i32> %bc, i64 0, !dbg !12
  %1485 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1484), !dbg !12
  %1486 = extractvalue [2 x <1 x float>] %.lcssa13, 0, !dbg !12
  %bc42 = bitcast <1 x float> %1486 to <1 x i32>, !dbg !12
  %1487 = extractelement <1 x i32> %bc42, i64 0, !dbg !12
  %1488 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1487), !dbg !12
  %1489 = and i32 %44, 16, !dbg !12
  %.not = icmp eq i32 %1489, 0, !dbg !12
  %1490 = and i32 %44, 32, !dbg !12
  %.not84 = icmp eq i32 %1490, 0, !dbg !12
  %1491 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !12
  %1492 = extractelement <8 x float> %1491, i64 0, !dbg !12
  %1493 = extractelement <8 x float> %1491, i64 1, !dbg !12
  %1494 = extractelement <8 x float> %1491, i64 2, !dbg !12
  %1495 = extractelement <8 x float> %1491, i64 3, !dbg !12
  %1496 = select i1 %.not, float %1493, float %1492, !dbg !12
  %1497 = bitcast float %1496 to i32, !dbg !12
  %1498 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1497), !dbg !12
  %1499 = bitcast i32 %1498 to float, !dbg !12
  %1500 = select i1 %.not, float %1492, float %1499, !dbg !12
  %1501 = select i1 %.not, float %1499, float %1493, !dbg !12
  %1502 = select i1 %.not, float %1495, float %1494, !dbg !12
  %1503 = bitcast float %1502 to i32, !dbg !12
  %1504 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1503), !dbg !12
  %1505 = bitcast i32 %1504 to float, !dbg !12
  %1506 = select i1 %.not, float %1494, float %1505, !dbg !12
  %1507 = select i1 %.not, float %1505, float %1495, !dbg !12
  %1508 = select i1 %.not84, float %1506, float %1500, !dbg !12
  %1509 = bitcast float %1508 to i32, !dbg !12
  %1510 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1509), !dbg !12
  %1511 = bitcast i32 %1510 to float, !dbg !12
  %1512 = select i1 %.not84, float %1507, float %1501, !dbg !12
  %1513 = bitcast float %1512 to i32, !dbg !12
  %1514 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1513), !dbg !12
  %1515 = bitcast i32 %1514 to float, !dbg !12
  %1516 = insertelement <4 x i1> poison, i1 %.not84, i64 0, !dbg !12
  %1517 = shufflevector <4 x i1> %1516, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %1518 = insertelement <4 x float> poison, float %1500, i64 0, !dbg !12
  %1519 = insertelement <4 x float> %1518, float %1501, i64 1, !dbg !12
  %1520 = insertelement <4 x float> %1519, float %1511, i64 2, !dbg !12
  %1521 = insertelement <4 x float> %1520, float %1515, i64 3, !dbg !12
  %1522 = insertelement <4 x float> poison, float %1511, i64 0, !dbg !12
  %1523 = insertelement <4 x float> %1522, float %1515, i64 1, !dbg !12
  %1524 = insertelement <4 x float> %1523, float %1506, i64 2, !dbg !12
  %1525 = insertelement <4 x float> %1524, float %1507, i64 3, !dbg !12
  %1526 = select <4 x i1> %1517, <4 x float> %1521, <4 x float> %1525, !dbg !12
  %1527 = extractelement <8 x float> %1491, i64 4, !dbg !12
  %1528 = extractelement <8 x float> %1491, i64 5, !dbg !12
  %1529 = extractelement <8 x float> %1491, i64 6, !dbg !12
  %1530 = extractelement <8 x float> %1491, i64 7, !dbg !12
  %1531 = select i1 %.not, float %1528, float %1527, !dbg !12
  %1532 = bitcast float %1531 to i32, !dbg !12
  %1533 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1532), !dbg !12
  %1534 = bitcast i32 %1533 to float, !dbg !12
  %1535 = select i1 %.not, float %1527, float %1534, !dbg !12
  %1536 = select i1 %.not, float %1534, float %1528, !dbg !12
  %1537 = select i1 %.not, float %1530, float %1529, !dbg !12
  %1538 = bitcast float %1537 to i32, !dbg !12
  %1539 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1538), !dbg !12
  %1540 = bitcast i32 %1539 to float, !dbg !12
  %1541 = select i1 %.not, float %1529, float %1540, !dbg !12
  %1542 = select i1 %.not, float %1540, float %1530, !dbg !12
  %1543 = select i1 %.not84, float %1541, float %1535, !dbg !12
  %1544 = bitcast float %1543 to i32, !dbg !12
  %1545 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1544), !dbg !12
  %1546 = bitcast i32 %1545 to float, !dbg !12
  %1547 = select i1 %.not84, float %1542, float %1536, !dbg !12
  %1548 = bitcast float %1547 to i32, !dbg !12
  %1549 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1548), !dbg !12
  %1550 = bitcast i32 %1549 to float, !dbg !12
  %1551 = insertelement <4 x float> poison, float %1535, i64 0, !dbg !12
  %1552 = insertelement <4 x float> %1551, float %1536, i64 1, !dbg !12
  %1553 = insertelement <4 x float> %1552, float %1546, i64 2, !dbg !12
  %1554 = insertelement <4 x float> %1553, float %1550, i64 3, !dbg !12
  %1555 = insertelement <4 x float> poison, float %1546, i64 0, !dbg !12
  %1556 = insertelement <4 x float> %1555, float %1550, i64 1, !dbg !12
  %1557 = insertelement <4 x float> %1556, float %1541, i64 2, !dbg !12
  %1558 = insertelement <4 x float> %1557, float %1542, i64 3, !dbg !12
  %1559 = select <4 x i1> %1517, <4 x float> %1554, <4 x float> %1558, !dbg !12
  %1560 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !12
  %1561 = extractelement <8 x float> %1560, i64 0, !dbg !12
  %1562 = extractelement <8 x float> %1560, i64 1, !dbg !12
  %1563 = extractelement <8 x float> %1560, i64 2, !dbg !12
  %1564 = extractelement <8 x float> %1560, i64 3, !dbg !12
  %1565 = select i1 %.not, float %1562, float %1561, !dbg !12
  %1566 = bitcast float %1565 to i32, !dbg !12
  %1567 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1566), !dbg !12
  %1568 = bitcast i32 %1567 to float, !dbg !12
  %1569 = select i1 %.not, float %1561, float %1568, !dbg !12
  %1570 = select i1 %.not, float %1568, float %1562, !dbg !12
  %1571 = select i1 %.not, float %1564, float %1563, !dbg !12
  %1572 = bitcast float %1571 to i32, !dbg !12
  %1573 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1572), !dbg !12
  %1574 = bitcast i32 %1573 to float, !dbg !12
  %1575 = select i1 %.not, float %1563, float %1574, !dbg !12
  %1576 = select i1 %.not, float %1574, float %1564, !dbg !12
  %1577 = select i1 %.not84, float %1575, float %1569, !dbg !12
  %1578 = bitcast float %1577 to i32, !dbg !12
  %1579 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1578), !dbg !12
  %1580 = bitcast i32 %1579 to float, !dbg !12
  %1581 = select i1 %.not84, float %1576, float %1570, !dbg !12
  %1582 = bitcast float %1581 to i32, !dbg !12
  %1583 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1582), !dbg !12
  %1584 = bitcast i32 %1583 to float, !dbg !12
  %1585 = insertelement <4 x float> poison, float %1569, i64 0, !dbg !12
  %1586 = insertelement <4 x float> %1585, float %1570, i64 1, !dbg !12
  %1587 = insertelement <4 x float> %1586, float %1580, i64 2, !dbg !12
  %1588 = insertelement <4 x float> %1587, float %1584, i64 3, !dbg !12
  %1589 = insertelement <4 x float> poison, float %1580, i64 0, !dbg !12
  %1590 = insertelement <4 x float> %1589, float %1584, i64 1, !dbg !12
  %1591 = insertelement <4 x float> %1590, float %1575, i64 2, !dbg !12
  %1592 = insertelement <4 x float> %1591, float %1576, i64 3, !dbg !12
  %1593 = select <4 x i1> %1517, <4 x float> %1588, <4 x float> %1592, !dbg !12
  %1594 = extractelement <8 x float> %1560, i64 4, !dbg !12
  %1595 = extractelement <8 x float> %1560, i64 5, !dbg !12
  %1596 = extractelement <8 x float> %1560, i64 6, !dbg !12
  %1597 = extractelement <8 x float> %1560, i64 7, !dbg !12
  %1598 = select i1 %.not, float %1595, float %1594, !dbg !12
  %1599 = bitcast float %1598 to i32, !dbg !12
  %1600 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1599), !dbg !12
  %1601 = bitcast i32 %1600 to float, !dbg !12
  %1602 = select i1 %.not, float %1594, float %1601, !dbg !12
  %1603 = select i1 %.not, float %1601, float %1595, !dbg !12
  %1604 = select i1 %.not, float %1597, float %1596, !dbg !12
  %1605 = bitcast float %1604 to i32, !dbg !12
  %1606 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1605), !dbg !12
  %1607 = bitcast i32 %1606 to float, !dbg !12
  %1608 = select i1 %.not, float %1596, float %1607, !dbg !12
  %1609 = select i1 %.not, float %1607, float %1597, !dbg !12
  %1610 = select i1 %.not84, float %1608, float %1602, !dbg !12
  %1611 = bitcast float %1610 to i32, !dbg !12
  %1612 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1611), !dbg !12
  %1613 = bitcast i32 %1612 to float, !dbg !12
  %1614 = select i1 %.not84, float %1609, float %1603, !dbg !12
  %1615 = bitcast float %1614 to i32, !dbg !12
  %1616 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1615), !dbg !12
  %1617 = bitcast i32 %1616 to float, !dbg !12
  %1618 = insertelement <4 x float> poison, float %1602, i64 0, !dbg !12
  %1619 = insertelement <4 x float> %1618, float %1603, i64 1, !dbg !12
  %1620 = insertelement <4 x float> %1619, float %1613, i64 2, !dbg !12
  %1621 = insertelement <4 x float> %1620, float %1617, i64 3, !dbg !12
  %1622 = insertelement <4 x float> poison, float %1613, i64 0, !dbg !12
  %1623 = insertelement <4 x float> %1622, float %1617, i64 1, !dbg !12
  %1624 = insertelement <4 x float> %1623, float %1608, i64 2, !dbg !12
  %1625 = insertelement <4 x float> %1624, float %1609, i64 3, !dbg !12
  %1626 = select <4 x i1> %1517, <4 x float> %1621, <4 x float> %1625, !dbg !12
  %1627 = extractvalue [2 x <8 x float>] %.lcssa11, 0, !dbg !12
  %1628 = extractelement <8 x float> %1627, i64 0, !dbg !12
  %1629 = extractelement <8 x float> %1627, i64 1, !dbg !12
  %1630 = extractelement <8 x float> %1627, i64 2, !dbg !12
  %1631 = extractelement <8 x float> %1627, i64 3, !dbg !12
  %1632 = select i1 %.not, float %1629, float %1628, !dbg !12
  %1633 = bitcast float %1632 to i32, !dbg !12
  %1634 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1633), !dbg !12
  %1635 = bitcast i32 %1634 to float, !dbg !12
  %1636 = select i1 %.not, float %1628, float %1635, !dbg !12
  %1637 = select i1 %.not, float %1635, float %1629, !dbg !12
  %1638 = select i1 %.not, float %1631, float %1630, !dbg !12
  %1639 = bitcast float %1638 to i32, !dbg !12
  %1640 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1639), !dbg !12
  %1641 = bitcast i32 %1640 to float, !dbg !12
  %1642 = select i1 %.not, float %1630, float %1641, !dbg !12
  %1643 = select i1 %.not, float %1641, float %1631, !dbg !12
  %1644 = select i1 %.not84, float %1642, float %1636, !dbg !12
  %1645 = bitcast float %1644 to i32, !dbg !12
  %1646 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1645), !dbg !12
  %1647 = bitcast i32 %1646 to float, !dbg !12
  %1648 = select i1 %.not84, float %1643, float %1637, !dbg !12
  %1649 = bitcast float %1648 to i32, !dbg !12
  %1650 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1649), !dbg !12
  %1651 = bitcast i32 %1650 to float, !dbg !12
  %1652 = insertelement <4 x float> poison, float %1636, i64 0, !dbg !12
  %1653 = insertelement <4 x float> %1652, float %1637, i64 1, !dbg !12
  %1654 = insertelement <4 x float> %1653, float %1647, i64 2, !dbg !12
  %1655 = insertelement <4 x float> %1654, float %1651, i64 3, !dbg !12
  %1656 = insertelement <4 x float> poison, float %1647, i64 0, !dbg !12
  %1657 = insertelement <4 x float> %1656, float %1651, i64 1, !dbg !12
  %1658 = insertelement <4 x float> %1657, float %1642, i64 2, !dbg !12
  %1659 = insertelement <4 x float> %1658, float %1643, i64 3, !dbg !12
  %1660 = select <4 x i1> %1517, <4 x float> %1655, <4 x float> %1659, !dbg !12
  %1661 = extractelement <8 x float> %1627, i64 4, !dbg !12
  %1662 = extractelement <8 x float> %1627, i64 5, !dbg !12
  %1663 = extractelement <8 x float> %1627, i64 6, !dbg !12
  %1664 = extractelement <8 x float> %1627, i64 7, !dbg !12
  %1665 = select i1 %.not, float %1662, float %1661, !dbg !12
  %1666 = bitcast float %1665 to i32, !dbg !12
  %1667 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1666), !dbg !12
  %1668 = bitcast i32 %1667 to float, !dbg !12
  %1669 = select i1 %.not, float %1661, float %1668, !dbg !12
  %1670 = select i1 %.not, float %1668, float %1662, !dbg !12
  %1671 = select i1 %.not, float %1664, float %1663, !dbg !12
  %1672 = bitcast float %1671 to i32, !dbg !12
  %1673 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1672), !dbg !12
  %1674 = bitcast i32 %1673 to float, !dbg !12
  %1675 = select i1 %.not, float %1663, float %1674, !dbg !12
  %1676 = select i1 %.not, float %1674, float %1664, !dbg !12
  %1677 = select i1 %.not84, float %1675, float %1669, !dbg !12
  %1678 = bitcast float %1677 to i32, !dbg !12
  %1679 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1678), !dbg !12
  %1680 = bitcast i32 %1679 to float, !dbg !12
  %1681 = select i1 %.not84, float %1676, float %1670, !dbg !12
  %1682 = bitcast float %1681 to i32, !dbg !12
  %1683 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1682), !dbg !12
  %1684 = bitcast i32 %1683 to float, !dbg !12
  %1685 = insertelement <4 x float> poison, float %1669, i64 0, !dbg !12
  %1686 = insertelement <4 x float> %1685, float %1670, i64 1, !dbg !12
  %1687 = insertelement <4 x float> %1686, float %1680, i64 2, !dbg !12
  %1688 = insertelement <4 x float> %1687, float %1684, i64 3, !dbg !12
  %1689 = insertelement <4 x float> poison, float %1680, i64 0, !dbg !12
  %1690 = insertelement <4 x float> %1689, float %1684, i64 1, !dbg !12
  %1691 = insertelement <4 x float> %1690, float %1675, i64 2, !dbg !12
  %1692 = insertelement <4 x float> %1691, float %1676, i64 3, !dbg !12
  %1693 = select <4 x i1> %1517, <4 x float> %1688, <4 x float> %1692, !dbg !12
  %1694 = extractvalue [2 x <8 x float>] %.lcssa11, 1, !dbg !12
  %1695 = extractelement <8 x float> %1694, i64 0, !dbg !12
  %1696 = extractelement <8 x float> %1694, i64 1, !dbg !12
  %1697 = extractelement <8 x float> %1694, i64 2, !dbg !12
  %1698 = extractelement <8 x float> %1694, i64 3, !dbg !12
  %1699 = select i1 %.not, float %1696, float %1695, !dbg !12
  %1700 = bitcast float %1699 to i32, !dbg !12
  %1701 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1700), !dbg !12
  %1702 = bitcast i32 %1701 to float, !dbg !12
  %1703 = select i1 %.not, float %1695, float %1702, !dbg !12
  %1704 = select i1 %.not, float %1702, float %1696, !dbg !12
  %1705 = select i1 %.not, float %1698, float %1697, !dbg !12
  %1706 = bitcast float %1705 to i32, !dbg !12
  %1707 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1706), !dbg !12
  %1708 = bitcast i32 %1707 to float, !dbg !12
  %1709 = select i1 %.not, float %1697, float %1708, !dbg !12
  %1710 = select i1 %.not, float %1708, float %1698, !dbg !12
  %1711 = select i1 %.not84, float %1709, float %1703, !dbg !12
  %1712 = bitcast float %1711 to i32, !dbg !12
  %1713 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1712), !dbg !12
  %1714 = bitcast i32 %1713 to float, !dbg !12
  %1715 = select i1 %.not84, float %1710, float %1704, !dbg !12
  %1716 = bitcast float %1715 to i32, !dbg !12
  %1717 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1716), !dbg !12
  %1718 = bitcast i32 %1717 to float, !dbg !12
  %1719 = insertelement <4 x float> poison, float %1703, i64 0, !dbg !12
  %1720 = insertelement <4 x float> %1719, float %1704, i64 1, !dbg !12
  %1721 = insertelement <4 x float> %1720, float %1714, i64 2, !dbg !12
  %1722 = insertelement <4 x float> %1721, float %1718, i64 3, !dbg !12
  %1723 = insertelement <4 x float> poison, float %1714, i64 0, !dbg !12
  %1724 = insertelement <4 x float> %1723, float %1718, i64 1, !dbg !12
  %1725 = insertelement <4 x float> %1724, float %1709, i64 2, !dbg !12
  %1726 = insertelement <4 x float> %1725, float %1710, i64 3, !dbg !12
  %1727 = select <4 x i1> %1517, <4 x float> %1722, <4 x float> %1726, !dbg !12
  %1728 = extractelement <8 x float> %1694, i64 4, !dbg !12
  %1729 = extractelement <8 x float> %1694, i64 5, !dbg !12
  %1730 = extractelement <8 x float> %1694, i64 6, !dbg !12
  %1731 = extractelement <8 x float> %1694, i64 7, !dbg !12
  %1732 = select i1 %.not, float %1729, float %1728, !dbg !12
  %1733 = bitcast float %1732 to i32, !dbg !12
  %1734 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1733), !dbg !12
  %1735 = bitcast i32 %1734 to float, !dbg !12
  %1736 = select i1 %.not, float %1728, float %1735, !dbg !12
  %1737 = select i1 %.not, float %1735, float %1729, !dbg !12
  %1738 = select i1 %.not, float %1731, float %1730, !dbg !12
  %1739 = bitcast float %1738 to i32, !dbg !12
  %1740 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1739), !dbg !12
  %1741 = bitcast i32 %1740 to float, !dbg !12
  %1742 = select i1 %.not, float %1730, float %1741, !dbg !12
  %1743 = select i1 %.not, float %1741, float %1731, !dbg !12
  %1744 = select i1 %.not84, float %1742, float %1736, !dbg !12
  %1745 = bitcast float %1744 to i32, !dbg !12
  %1746 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1745), !dbg !12
  %1747 = bitcast i32 %1746 to float, !dbg !12
  %1748 = select i1 %.not84, float %1743, float %1737, !dbg !12
  %1749 = bitcast float %1748 to i32, !dbg !12
  %1750 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1749), !dbg !12
  %1751 = bitcast i32 %1750 to float, !dbg !12
  %1752 = insertelement <4 x float> poison, float %1736, i64 0, !dbg !12
  %1753 = insertelement <4 x float> %1752, float %1737, i64 1, !dbg !12
  %1754 = insertelement <4 x float> %1753, float %1747, i64 2, !dbg !12
  %1755 = insertelement <4 x float> %1754, float %1751, i64 3, !dbg !12
  %1756 = insertelement <4 x float> poison, float %1747, i64 0, !dbg !12
  %1757 = insertelement <4 x float> %1756, float %1751, i64 1, !dbg !12
  %1758 = insertelement <4 x float> %1757, float %1742, i64 2, !dbg !12
  %1759 = insertelement <4 x float> %1758, float %1743, i64 3, !dbg !12
  %1760 = select <4 x i1> %1517, <4 x float> %1755, <4 x float> %1759, !dbg !12
  %1761 = extractvalue [2 x <8 x float>] %.lcssa12, 0, !dbg !12
  %1762 = extractelement <8 x float> %1761, i64 0, !dbg !12
  %1763 = extractelement <8 x float> %1761, i64 1, !dbg !12
  %1764 = extractelement <8 x float> %1761, i64 2, !dbg !12
  %1765 = extractelement <8 x float> %1761, i64 3, !dbg !12
  %1766 = select i1 %.not, float %1763, float %1762, !dbg !12
  %1767 = bitcast float %1766 to i32, !dbg !12
  %1768 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1767), !dbg !12
  %1769 = bitcast i32 %1768 to float, !dbg !12
  %1770 = select i1 %.not, float %1762, float %1769, !dbg !12
  %1771 = select i1 %.not, float %1769, float %1763, !dbg !12
  %1772 = select i1 %.not, float %1765, float %1764, !dbg !12
  %1773 = bitcast float %1772 to i32, !dbg !12
  %1774 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1773), !dbg !12
  %1775 = bitcast i32 %1774 to float, !dbg !12
  %1776 = select i1 %.not, float %1764, float %1775, !dbg !12
  %1777 = select i1 %.not, float %1775, float %1765, !dbg !12
  %1778 = select i1 %.not84, float %1776, float %1770, !dbg !12
  %1779 = bitcast float %1778 to i32, !dbg !12
  %1780 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1779), !dbg !12
  %1781 = bitcast i32 %1780 to float, !dbg !12
  %1782 = select i1 %.not84, float %1777, float %1771, !dbg !12
  %1783 = bitcast float %1782 to i32, !dbg !12
  %1784 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1783), !dbg !12
  %1785 = bitcast i32 %1784 to float, !dbg !12
  %1786 = insertelement <4 x float> poison, float %1770, i64 0, !dbg !12
  %1787 = insertelement <4 x float> %1786, float %1771, i64 1, !dbg !12
  %1788 = insertelement <4 x float> %1787, float %1781, i64 2, !dbg !12
  %1789 = insertelement <4 x float> %1788, float %1785, i64 3, !dbg !12
  %1790 = insertelement <4 x float> poison, float %1781, i64 0, !dbg !12
  %1791 = insertelement <4 x float> %1790, float %1785, i64 1, !dbg !12
  %1792 = insertelement <4 x float> %1791, float %1776, i64 2, !dbg !12
  %1793 = insertelement <4 x float> %1792, float %1777, i64 3, !dbg !12
  %1794 = select <4 x i1> %1517, <4 x float> %1789, <4 x float> %1793, !dbg !12
  %1795 = extractelement <8 x float> %1761, i64 4, !dbg !12
  %1796 = extractelement <8 x float> %1761, i64 5, !dbg !12
  %1797 = extractelement <8 x float> %1761, i64 6, !dbg !12
  %1798 = extractelement <8 x float> %1761, i64 7, !dbg !12
  %1799 = select i1 %.not, float %1796, float %1795, !dbg !12
  %1800 = bitcast float %1799 to i32, !dbg !12
  %1801 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1800), !dbg !12
  %1802 = bitcast i32 %1801 to float, !dbg !12
  %1803 = select i1 %.not, float %1795, float %1802, !dbg !12
  %1804 = select i1 %.not, float %1802, float %1796, !dbg !12
  %1805 = select i1 %.not, float %1798, float %1797, !dbg !12
  %1806 = bitcast float %1805 to i32, !dbg !12
  %1807 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1806), !dbg !12
  %1808 = bitcast i32 %1807 to float, !dbg !12
  %1809 = select i1 %.not, float %1797, float %1808, !dbg !12
  %1810 = select i1 %.not, float %1808, float %1798, !dbg !12
  %1811 = select i1 %.not84, float %1809, float %1803, !dbg !12
  %1812 = bitcast float %1811 to i32, !dbg !12
  %1813 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1812), !dbg !12
  %1814 = bitcast i32 %1813 to float, !dbg !12
  %1815 = select i1 %.not84, float %1810, float %1804, !dbg !12
  %1816 = bitcast float %1815 to i32, !dbg !12
  %1817 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1816), !dbg !12
  %1818 = bitcast i32 %1817 to float, !dbg !12
  %1819 = insertelement <4 x float> poison, float %1803, i64 0, !dbg !12
  %1820 = insertelement <4 x float> %1819, float %1804, i64 1, !dbg !12
  %1821 = insertelement <4 x float> %1820, float %1814, i64 2, !dbg !12
  %1822 = insertelement <4 x float> %1821, float %1818, i64 3, !dbg !12
  %1823 = insertelement <4 x float> poison, float %1814, i64 0, !dbg !12
  %1824 = insertelement <4 x float> %1823, float %1818, i64 1, !dbg !12
  %1825 = insertelement <4 x float> %1824, float %1809, i64 2, !dbg !12
  %1826 = insertelement <4 x float> %1825, float %1810, i64 3, !dbg !12
  %1827 = select <4 x i1> %1517, <4 x float> %1822, <4 x float> %1826, !dbg !12
  %1828 = extractvalue [2 x <8 x float>] %.lcssa12, 1, !dbg !12
  %1829 = extractelement <8 x float> %1828, i64 0, !dbg !12
  %1830 = extractelement <8 x float> %1828, i64 1, !dbg !12
  %1831 = extractelement <8 x float> %1828, i64 2, !dbg !12
  %1832 = extractelement <8 x float> %1828, i64 3, !dbg !12
  %1833 = select i1 %.not, float %1830, float %1829, !dbg !12
  %1834 = bitcast float %1833 to i32, !dbg !12
  %1835 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1834), !dbg !12
  %1836 = bitcast i32 %1835 to float, !dbg !12
  %1837 = select i1 %.not, float %1829, float %1836, !dbg !12
  %1838 = select i1 %.not, float %1836, float %1830, !dbg !12
  %1839 = select i1 %.not, float %1832, float %1831, !dbg !12
  %1840 = bitcast float %1839 to i32, !dbg !12
  %1841 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1840), !dbg !12
  %1842 = bitcast i32 %1841 to float, !dbg !12
  %1843 = select i1 %.not, float %1831, float %1842, !dbg !12
  %1844 = select i1 %.not, float %1842, float %1832, !dbg !12
  %1845 = select i1 %.not84, float %1843, float %1837, !dbg !12
  %1846 = bitcast float %1845 to i32, !dbg !12
  %1847 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1846), !dbg !12
  %1848 = bitcast i32 %1847 to float, !dbg !12
  %1849 = select i1 %.not84, float %1844, float %1838, !dbg !12
  %1850 = bitcast float %1849 to i32, !dbg !12
  %1851 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1850), !dbg !12
  %1852 = bitcast i32 %1851 to float, !dbg !12
  %1853 = insertelement <4 x float> poison, float %1837, i64 0, !dbg !12
  %1854 = insertelement <4 x float> %1853, float %1838, i64 1, !dbg !12
  %1855 = insertelement <4 x float> %1854, float %1848, i64 2, !dbg !12
  %1856 = insertelement <4 x float> %1855, float %1852, i64 3, !dbg !12
  %1857 = insertelement <4 x float> poison, float %1848, i64 0, !dbg !12
  %1858 = insertelement <4 x float> %1857, float %1852, i64 1, !dbg !12
  %1859 = insertelement <4 x float> %1858, float %1843, i64 2, !dbg !12
  %1860 = insertelement <4 x float> %1859, float %1844, i64 3, !dbg !12
  %1861 = select <4 x i1> %1517, <4 x float> %1856, <4 x float> %1860, !dbg !12
  %1862 = extractelement <8 x float> %1828, i64 4, !dbg !12
  %1863 = extractelement <8 x float> %1828, i64 5, !dbg !12
  %1864 = extractelement <8 x float> %1828, i64 6, !dbg !12
  %1865 = extractelement <8 x float> %1828, i64 7, !dbg !12
  %1866 = select i1 %.not, float %1863, float %1862, !dbg !12
  %1867 = bitcast float %1866 to i32, !dbg !12
  %1868 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1867), !dbg !12
  %1869 = bitcast i32 %1868 to float, !dbg !12
  %1870 = select i1 %.not, float %1862, float %1869, !dbg !12
  %1871 = select i1 %.not, float %1869, float %1863, !dbg !12
  %1872 = select i1 %.not, float %1865, float %1864, !dbg !12
  %1873 = bitcast float %1872 to i32, !dbg !12
  %1874 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1873), !dbg !12
  %1875 = bitcast i32 %1874 to float, !dbg !12
  %1876 = select i1 %.not, float %1864, float %1875, !dbg !12
  %1877 = select i1 %.not, float %1875, float %1865, !dbg !12
  %1878 = select i1 %.not84, float %1876, float %1870, !dbg !12
  %1879 = bitcast float %1878 to i32, !dbg !12
  %1880 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1879), !dbg !12
  %1881 = bitcast i32 %1880 to float, !dbg !12
  %1882 = select i1 %.not84, float %1877, float %1871, !dbg !12
  %1883 = bitcast float %1882 to i32, !dbg !12
  %1884 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1883), !dbg !12
  %1885 = bitcast i32 %1884 to float, !dbg !12
  %1886 = insertelement <4 x float> poison, float %1870, i64 0, !dbg !12
  %1887 = insertelement <4 x float> %1886, float %1871, i64 1, !dbg !12
  %1888 = insertelement <4 x float> %1887, float %1881, i64 2, !dbg !12
  %1889 = insertelement <4 x float> %1888, float %1885, i64 3, !dbg !12
  %1890 = insertelement <4 x float> poison, float %1881, i64 0, !dbg !12
  %1891 = insertelement <4 x float> %1890, float %1885, i64 1, !dbg !12
  %1892 = insertelement <4 x float> %1891, float %1876, i64 2, !dbg !12
  %1893 = insertelement <4 x float> %1892, float %1877, i64 3, !dbg !12
  %1894 = select <4 x i1> %1517, <4 x float> %1889, <4 x float> %1893, !dbg !12
  %1895 = extractvalue [2 x <8 x float>] %.lcssa14, 0, !dbg !12
  %1896 = extractelement <8 x float> %1895, i64 0, !dbg !12
  %1897 = extractelement <8 x float> %1895, i64 1, !dbg !12
  %1898 = extractelement <8 x float> %1895, i64 2, !dbg !12
  %1899 = extractelement <8 x float> %1895, i64 3, !dbg !12
  %1900 = select i1 %.not, float %1897, float %1896, !dbg !12
  %1901 = bitcast float %1900 to i32, !dbg !12
  %1902 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1901), !dbg !12
  %1903 = bitcast i32 %1902 to float, !dbg !12
  %1904 = select i1 %.not, float %1896, float %1903, !dbg !12
  %1905 = select i1 %.not, float %1903, float %1897, !dbg !12
  %1906 = select i1 %.not, float %1899, float %1898, !dbg !12
  %1907 = bitcast float %1906 to i32, !dbg !12
  %1908 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1907), !dbg !12
  %1909 = bitcast i32 %1908 to float, !dbg !12
  %1910 = select i1 %.not, float %1898, float %1909, !dbg !12
  %1911 = select i1 %.not, float %1909, float %1899, !dbg !12
  %1912 = select i1 %.not84, float %1910, float %1904, !dbg !12
  %1913 = bitcast float %1912 to i32, !dbg !12
  %1914 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1913), !dbg !12
  %1915 = select i1 %.not84, float %1911, float %1905, !dbg !12
  %1916 = bitcast float %1915 to i32, !dbg !12
  %1917 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1916), !dbg !12
  %1918 = extractelement <8 x float> %1895, i64 4, !dbg !12
  %1919 = extractelement <8 x float> %1895, i64 5, !dbg !12
  %1920 = extractelement <8 x float> %1895, i64 6, !dbg !12
  %1921 = extractelement <8 x float> %1895, i64 7, !dbg !12
  %1922 = select i1 %.not, float %1919, float %1918, !dbg !12
  %1923 = bitcast float %1922 to i32, !dbg !12
  %1924 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1923), !dbg !12
  %1925 = bitcast i32 %1924 to float, !dbg !12
  %1926 = select i1 %.not, float %1918, float %1925, !dbg !12
  %1927 = select i1 %.not, float %1925, float %1919, !dbg !12
  %1928 = select i1 %.not, float %1921, float %1920, !dbg !12
  %1929 = bitcast float %1928 to i32, !dbg !12
  %1930 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1929), !dbg !12
  %1931 = bitcast i32 %1930 to float, !dbg !12
  %1932 = select i1 %.not, float %1920, float %1931, !dbg !12
  %1933 = select i1 %.not, float %1931, float %1921, !dbg !12
  %1934 = select i1 %.not84, float %1932, float %1926, !dbg !12
  %1935 = bitcast float %1934 to i32, !dbg !12
  %1936 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1935), !dbg !12
  %1937 = select i1 %.not84, float %1933, float %1927, !dbg !12
  %1938 = bitcast float %1937 to i32, !dbg !12
  %1939 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1938), !dbg !12
  %1940 = extractvalue [2 x <8 x float>] %.lcssa14, 1, !dbg !12
  %1941 = extractelement <8 x float> %1940, i64 0, !dbg !12
  %1942 = extractelement <8 x float> %1940, i64 1, !dbg !12
  %1943 = extractelement <8 x float> %1940, i64 2, !dbg !12
  %1944 = extractelement <8 x float> %1940, i64 3, !dbg !12
  %1945 = select i1 %.not, float %1942, float %1941, !dbg !12
  %1946 = bitcast float %1945 to i32, !dbg !12
  %1947 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1946), !dbg !12
  %1948 = bitcast i32 %1947 to float, !dbg !12
  %1949 = select i1 %.not, float %1941, float %1948, !dbg !12
  %1950 = select i1 %.not, float %1948, float %1942, !dbg !12
  %1951 = select i1 %.not, float %1944, float %1943, !dbg !12
  %1952 = bitcast float %1951 to i32, !dbg !12
  %1953 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1952), !dbg !12
  %1954 = bitcast i32 %1953 to float, !dbg !12
  %1955 = select i1 %.not, float %1943, float %1954, !dbg !12
  %1956 = select i1 %.not, float %1954, float %1944, !dbg !12
  %1957 = select i1 %.not84, float %1955, float %1949, !dbg !12
  %1958 = bitcast float %1957 to i32, !dbg !12
  %1959 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1958), !dbg !12
  %1960 = select i1 %.not84, float %1956, float %1950, !dbg !12
  %1961 = bitcast float %1960 to i32, !dbg !12
  %1962 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1961), !dbg !12
  %1963 = extractelement <8 x float> %1940, i64 4, !dbg !12
  %1964 = extractelement <8 x float> %1940, i64 5, !dbg !12
  %1965 = extractelement <8 x float> %1940, i64 6, !dbg !12
  %1966 = extractelement <8 x float> %1940, i64 7, !dbg !12
  %1967 = select i1 %.not, float %1964, float %1963, !dbg !12
  %1968 = bitcast float %1967 to i32, !dbg !12
  %1969 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1968), !dbg !12
  %1970 = bitcast i32 %1969 to float, !dbg !12
  %1971 = select i1 %.not, float %1963, float %1970, !dbg !12
  %1972 = select i1 %.not, float %1970, float %1964, !dbg !12
  %1973 = select i1 %.not, float %1966, float %1965, !dbg !12
  %1974 = bitcast float %1973 to i32, !dbg !12
  %1975 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1974), !dbg !12
  %1976 = bitcast i32 %1975 to float, !dbg !12
  %1977 = select i1 %.not, float %1965, float %1976, !dbg !12
  %1978 = select i1 %.not, float %1976, float %1966, !dbg !12
  %1979 = select i1 %.not84, float %1977, float %1971, !dbg !12
  %1980 = bitcast float %1979 to i32, !dbg !12
  %1981 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1980), !dbg !12
  %1982 = select i1 %.not84, float %1978, float %1972, !dbg !12
  %1983 = bitcast float %1982 to i32, !dbg !12
  %1984 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1983), !dbg !12
  %1985 = insertelement <2 x i32> poison, i32 %1488, i64 0, !dbg !12
  %1986 = insertelement <2 x i32> %1985, i32 %1485, i64 1, !dbg !12
  %1987 = bitcast <2 x i32> %1986 to <2 x float>, !dbg !12
  %1988 = shufflevector <1 x float> %1486, <1 x float> %1483, <2 x i32> <i32 0, i32 1>, !dbg !12
  %1989 = fadd <2 x float> %1988, %1987, !dbg !12
  %bc43 = bitcast <2 x float> %1989 to <2 x i32>, !dbg !12
  %1990 = extractelement <2 x i32> %bc43, i64 1, !dbg !12
  %1991 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1990), !dbg !12
  %bc44 = bitcast <2 x float> %1989 to <2 x i32>, !dbg !12
  %1992 = extractelement <2 x i32> %bc44, i64 0, !dbg !12
  %1993 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1992), !dbg !12
  %1994 = insertelement <2 x i32> poison, i32 %1993, i64 0, !dbg !12
  %1995 = insertelement <2 x i32> %1994, i32 %1991, i64 1, !dbg !12
  %1996 = bitcast <2 x i32> %1995 to <2 x float>, !dbg !12
  %1997 = fadd <2 x float> %1989, %1996, !dbg !12
  %1998 = fdiv arcp <2 x float> splat (float 1.000000e+00), %1997, !dbg !51
  %1999 = shufflevector <2 x float> %1998, <2 x float> poison, <4 x i32> zeroinitializer, !dbg !51
  %2000 = shufflevector <2 x float> %1998, <2 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, !dbg !51
  %2001 = bitcast i32 %1981 to float, !dbg !12
  %2002 = bitcast i32 %1984 to float, !dbg !12
  %2003 = insertelement <4 x float> poison, float %1971, i64 0, !dbg !12
  %2004 = insertelement <4 x float> %2003, float %1972, i64 1, !dbg !12
  %2005 = insertelement <4 x float> %2004, float %2001, i64 2, !dbg !12
  %2006 = insertelement <4 x float> %2005, float %2002, i64 3, !dbg !12
  %2007 = insertelement <4 x float> poison, float %2001, i64 0, !dbg !12
  %2008 = insertelement <4 x float> %2007, float %2002, i64 1, !dbg !12
  %2009 = insertelement <4 x float> %2008, float %1977, i64 2, !dbg !12
  %2010 = insertelement <4 x float> %2009, float %1978, i64 3, !dbg !12
  %2011 = select <4 x i1> %1517, <4 x float> %2006, <4 x float> %2010, !dbg !12
  %2012 = fmul <4 x float> %2000, %2011, !dbg !51
  %2013 = fptrunc <4 x float> %2012 to <4 x half>, !dbg !52
  %2014 = extractelement <4 x half> %2013, i64 3, !dbg !52
  %2015 = extractelement <4 x half> %2013, i64 2, !dbg !52
  %2016 = extractelement <4 x half> %2013, i64 1, !dbg !52
  %2017 = extractelement <4 x half> %2013, i64 0, !dbg !52
  %2018 = bitcast i32 %1959 to float, !dbg !12
  %2019 = bitcast i32 %1962 to float, !dbg !12
  %2020 = insertelement <4 x float> poison, float %1949, i64 0, !dbg !12
  %2021 = insertelement <4 x float> %2020, float %1950, i64 1, !dbg !12
  %2022 = insertelement <4 x float> %2021, float %2018, i64 2, !dbg !12
  %2023 = insertelement <4 x float> %2022, float %2019, i64 3, !dbg !12
  %2024 = insertelement <4 x float> poison, float %2018, i64 0, !dbg !12
  %2025 = insertelement <4 x float> %2024, float %2019, i64 1, !dbg !12
  %2026 = insertelement <4 x float> %2025, float %1955, i64 2, !dbg !12
  %2027 = insertelement <4 x float> %2026, float %1956, i64 3, !dbg !12
  %2028 = select <4 x i1> %1517, <4 x float> %2023, <4 x float> %2027, !dbg !12
  %2029 = fmul <4 x float> %2000, %2028, !dbg !51
  %2030 = fptrunc <4 x float> %2029 to <4 x half>, !dbg !52
  %2031 = extractelement <4 x half> %2030, i64 3, !dbg !52
  %2032 = extractelement <4 x half> %2030, i64 2, !dbg !52
  %2033 = extractelement <4 x half> %2030, i64 1, !dbg !52
  %2034 = extractelement <4 x half> %2030, i64 0, !dbg !52
  %2035 = bitcast i32 %1936 to float, !dbg !12
  %2036 = bitcast i32 %1939 to float, !dbg !12
  %2037 = insertelement <4 x float> poison, float %1926, i64 0, !dbg !12
  %2038 = insertelement <4 x float> %2037, float %1927, i64 1, !dbg !12
  %2039 = insertelement <4 x float> %2038, float %2035, i64 2, !dbg !12
  %2040 = insertelement <4 x float> %2039, float %2036, i64 3, !dbg !12
  %2041 = insertelement <4 x float> poison, float %2035, i64 0, !dbg !12
  %2042 = insertelement <4 x float> %2041, float %2036, i64 1, !dbg !12
  %2043 = insertelement <4 x float> %2042, float %1932, i64 2, !dbg !12
  %2044 = insertelement <4 x float> %2043, float %1933, i64 3, !dbg !12
  %2045 = select <4 x i1> %1517, <4 x float> %2040, <4 x float> %2044, !dbg !12
  %2046 = fmul <4 x float> %1999, %2045, !dbg !51
  %2047 = fptrunc <4 x float> %2046 to <4 x half>, !dbg !52
  %2048 = extractelement <4 x half> %2047, i64 3, !dbg !52
  %2049 = extractelement <4 x half> %2047, i64 2, !dbg !52
  %2050 = extractelement <4 x half> %2047, i64 1, !dbg !52
  %2051 = extractelement <4 x half> %2047, i64 0, !dbg !52
  %2052 = bitcast i32 %1914 to float, !dbg !12
  %2053 = bitcast i32 %1917 to float, !dbg !12
  %2054 = insertelement <4 x float> poison, float %1904, i64 0, !dbg !12
  %2055 = insertelement <4 x float> %2054, float %1905, i64 1, !dbg !12
  %2056 = insertelement <4 x float> %2055, float %2052, i64 2, !dbg !12
  %2057 = insertelement <4 x float> %2056, float %2053, i64 3, !dbg !12
  %2058 = insertelement <4 x float> poison, float %2052, i64 0, !dbg !12
  %2059 = insertelement <4 x float> %2058, float %2053, i64 1, !dbg !12
  %2060 = insertelement <4 x float> %2059, float %1910, i64 2, !dbg !12
  %2061 = insertelement <4 x float> %2060, float %1911, i64 3, !dbg !12
  %2062 = select <4 x i1> %1517, <4 x float> %2057, <4 x float> %2061, !dbg !12
  %2063 = fmul <4 x float> %1999, %2062, !dbg !51
  %2064 = fptrunc <4 x float> %2063 to <4 x half>, !dbg !52
  %2065 = extractelement <4 x half> %2064, i64 3, !dbg !52
  %2066 = extractelement <4 x half> %2064, i64 2, !dbg !52
  %2067 = extractelement <4 x half> %2064, i64 1, !dbg !52
  %2068 = extractelement <4 x half> %2064, i64 0, !dbg !52
  %address.thread4263 = tail call i32 asm sideeffect "", "=v,0"(i32 %44) #8, !dbg !53, !range !10, !frisk.register_lifetimes !5
  %address.local4264 = zext nneg i32 %address.thread4263 to i64, !dbg !9
  %2069 = lshr i64 %address.local4264, 2, !dbg !15
  %address.local4267 = and i64 %2069, 48, !dbg !15
  %address.local4269 = and i64 %address.local4264, 240, !dbg !53
  %address.local4268 = add i64 %42, %address.local4264, !dbg !53
  %address.local4271 = sub i64 %address.local4268, %address.local4269, !dbg !53
  %address.local4272 = add i64 %address.local4271, %address.local4267, !dbg !53
  %2070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %52, !dbg !53
  %address.local4275.idx = shl nuw nsw i64 %address.local4272, 8, !dbg !53
  %address.local4275 = getelementptr inbounds nuw i8, ptr addrspace(1) %2070, i64 %address.local4275.idx, !dbg !53
  %address.local4278 = and i64 %2069, 12, !dbg !11
  %address.local4279 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4278, !dbg !53
  store half %2068, ptr addrspace(1) %address.local4279, align 2, !dbg !53
  %address.local4281 = or disjoint i64 %address.local4278, 1, !dbg !53
  %address.local4282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4281, !dbg !53
  store half %2067, ptr addrspace(1) %address.local4282, align 2, !dbg !53
  %address.local4284 = or disjoint i64 %address.local4278, 2, !dbg !53
  %address.local4285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4284, !dbg !53
  store half %2066, ptr addrspace(1) %address.local4285, align 2, !dbg !53
  %address.local4287 = or disjoint i64 %address.local4278, 3, !dbg !53
  %address.local4288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4287, !dbg !53
  store half %2065, ptr addrspace(1) %address.local4288, align 2, !dbg !53
  %address.local4290 = or disjoint i64 %address.local4278, 16, !dbg !53
  %address.local4291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4290, !dbg !53
  store half %2051, ptr addrspace(1) %address.local4291, align 2, !dbg !53
  %address.local4293 = or disjoint i64 %address.local4278, 17, !dbg !53
  %address.local4294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4293, !dbg !53
  store half %2050, ptr addrspace(1) %address.local4294, align 2, !dbg !53
  %address.local4296 = or disjoint i64 %address.local4278, 18, !dbg !53
  %address.local4297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4296, !dbg !53
  store half %2049, ptr addrspace(1) %address.local4297, align 2, !dbg !53
  %address.local4299 = or disjoint i64 %address.local4278, 19, !dbg !53
  %address.local4300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4299, !dbg !53
  store half %2048, ptr addrspace(1) %address.local4300, align 2, !dbg !53
  %address.local4304 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local4275, i64 16384, !dbg !53
  %address.local4305 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4278, !dbg !53
  store half %2034, ptr addrspace(1) %address.local4305, align 2, !dbg !53
  %address.local4307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4281, !dbg !53
  store half %2033, ptr addrspace(1) %address.local4307, align 2, !dbg !53
  %address.local4309 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4284, !dbg !53
  store half %2032, ptr addrspace(1) %address.local4309, align 2, !dbg !53
  %address.local4311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4287, !dbg !53
  store half %2031, ptr addrspace(1) %address.local4311, align 2, !dbg !53
  %address.local4313 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4290, !dbg !53
  store half %2017, ptr addrspace(1) %address.local4313, align 2, !dbg !53
  %address.local4315 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4293, !dbg !53
  store half %2016, ptr addrspace(1) %address.local4315, align 2, !dbg !53
  %address.local4317 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4296, !dbg !53
  store half %2015, ptr addrspace(1) %address.local4317, align 2, !dbg !53
  %address.local4319 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4299, !dbg !53
  store half %2014, ptr addrspace(1) %address.local4319, align 2, !dbg !53
  %2071 = fmul <4 x float> %2000, %1894, !dbg !51
  %2072 = fptrunc <4 x float> %2071 to <4 x half>, !dbg !52
  %2073 = extractelement <4 x half> %2072, i64 3, !dbg !52
  %2074 = extractelement <4 x half> %2072, i64 2, !dbg !52
  %2075 = extractelement <4 x half> %2072, i64 1, !dbg !52
  %2076 = extractelement <4 x half> %2072, i64 0, !dbg !52
  %2077 = fmul <4 x float> %2000, %1861, !dbg !51
  %2078 = fptrunc <4 x float> %2077 to <4 x half>, !dbg !52
  %2079 = extractelement <4 x half> %2078, i64 3, !dbg !52
  %2080 = extractelement <4 x half> %2078, i64 2, !dbg !52
  %2081 = extractelement <4 x half> %2078, i64 1, !dbg !52
  %2082 = extractelement <4 x half> %2078, i64 0, !dbg !52
  %2083 = fmul <4 x float> %1999, %1827, !dbg !51
  %2084 = fptrunc <4 x float> %2083 to <4 x half>, !dbg !52
  %2085 = extractelement <4 x half> %2084, i64 3, !dbg !52
  %2086 = extractelement <4 x half> %2084, i64 2, !dbg !52
  %2087 = extractelement <4 x half> %2084, i64 1, !dbg !52
  %2088 = extractelement <4 x half> %2084, i64 0, !dbg !52
  %2089 = fmul <4 x float> %1999, %1794, !dbg !51
  %2090 = fptrunc <4 x float> %2089 to <4 x half>, !dbg !52
  %2091 = extractelement <4 x half> %2090, i64 3, !dbg !52
  %2092 = extractelement <4 x half> %2090, i64 2, !dbg !52
  %2093 = extractelement <4 x half> %2090, i64 1, !dbg !52
  %2094 = extractelement <4 x half> %2090, i64 0, !dbg !52
  %address.local4336 = or disjoint i64 %address.local4278, 32, !dbg !53
  %address.local4337 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4336, !dbg !53
  store half %2094, ptr addrspace(1) %address.local4337, align 2, !dbg !53
  %address.local4339 = or disjoint i64 %address.local4278, 33, !dbg !53
  %address.local4340 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4339, !dbg !53
  store half %2093, ptr addrspace(1) %address.local4340, align 2, !dbg !53
  %address.local4342 = or disjoint i64 %address.local4278, 34, !dbg !53
  %address.local4343 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4342, !dbg !53
  store half %2092, ptr addrspace(1) %address.local4343, align 2, !dbg !53
  %address.local4345 = or disjoint i64 %address.local4278, 35, !dbg !53
  %address.local4346 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4345, !dbg !53
  store half %2091, ptr addrspace(1) %address.local4346, align 2, !dbg !53
  %address.local4348 = or disjoint i64 %address.local4278, 48, !dbg !53
  %address.local4349 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4348, !dbg !53
  store half %2088, ptr addrspace(1) %address.local4349, align 2, !dbg !53
  %address.local4351 = or disjoint i64 %address.local4278, 49, !dbg !53
  %address.local4352 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4351, !dbg !53
  store half %2087, ptr addrspace(1) %address.local4352, align 2, !dbg !53
  %address.local4354 = or disjoint i64 %address.local4278, 50, !dbg !53
  %address.local4355 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4354, !dbg !53
  store half %2086, ptr addrspace(1) %address.local4355, align 2, !dbg !53
  %address.local4357 = or i64 %2069, 51, !dbg !53
  %address.local4358 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4357, !dbg !53
  store half %2085, ptr addrspace(1) %address.local4358, align 2, !dbg !53
  %address.local4363 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4336, !dbg !53
  store half %2082, ptr addrspace(1) %address.local4363, align 2, !dbg !53
  %address.local4365 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4339, !dbg !53
  store half %2081, ptr addrspace(1) %address.local4365, align 2, !dbg !53
  %address.local4367 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4342, !dbg !53
  store half %2080, ptr addrspace(1) %address.local4367, align 2, !dbg !53
  %address.local4369 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4345, !dbg !53
  store half %2079, ptr addrspace(1) %address.local4369, align 2, !dbg !53
  %address.local4371 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4348, !dbg !53
  store half %2076, ptr addrspace(1) %address.local4371, align 2, !dbg !53
  %address.local4373 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4351, !dbg !53
  store half %2075, ptr addrspace(1) %address.local4373, align 2, !dbg !53
  %address.local4375 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4354, !dbg !53
  store half %2074, ptr addrspace(1) %address.local4375, align 2, !dbg !53
  %address.local4377 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4357, !dbg !53
  store half %2073, ptr addrspace(1) %address.local4377, align 2, !dbg !53
  %2095 = fmul <4 x float> %2000, %1760, !dbg !51
  %2096 = fptrunc <4 x float> %2095 to <4 x half>, !dbg !52
  %2097 = extractelement <4 x half> %2096, i64 3, !dbg !52
  %2098 = extractelement <4 x half> %2096, i64 2, !dbg !52
  %2099 = extractelement <4 x half> %2096, i64 1, !dbg !52
  %2100 = extractelement <4 x half> %2096, i64 0, !dbg !52
  %2101 = fmul <4 x float> %2000, %1727, !dbg !51
  %2102 = fptrunc <4 x float> %2101 to <4 x half>, !dbg !52
  %2103 = extractelement <4 x half> %2102, i64 3, !dbg !52
  %2104 = extractelement <4 x half> %2102, i64 2, !dbg !52
  %2105 = extractelement <4 x half> %2102, i64 1, !dbg !52
  %2106 = extractelement <4 x half> %2102, i64 0, !dbg !52
  %2107 = fmul <4 x float> %1999, %1693, !dbg !51
  %2108 = fptrunc <4 x float> %2107 to <4 x half>, !dbg !52
  %2109 = extractelement <4 x half> %2108, i64 3, !dbg !52
  %2110 = extractelement <4 x half> %2108, i64 2, !dbg !52
  %2111 = extractelement <4 x half> %2108, i64 1, !dbg !52
  %2112 = extractelement <4 x half> %2108, i64 0, !dbg !52
  %2113 = fmul <4 x float> %1999, %1660, !dbg !51
  %2114 = fptrunc <4 x float> %2113 to <4 x half>, !dbg !52
  %2115 = extractelement <4 x half> %2114, i64 3, !dbg !52
  %2116 = extractelement <4 x half> %2114, i64 2, !dbg !52
  %2117 = extractelement <4 x half> %2114, i64 1, !dbg !52
  %2118 = extractelement <4 x half> %2114, i64 0, !dbg !52
  %address.local4394 = or disjoint i64 %address.local4278, 64, !dbg !53
  %address.local4395 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4394, !dbg !53
  store half %2118, ptr addrspace(1) %address.local4395, align 2, !dbg !53
  %address.local4397 = or disjoint i64 %address.local4278, 65, !dbg !53
  %address.local4398 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4397, !dbg !53
  store half %2117, ptr addrspace(1) %address.local4398, align 2, !dbg !53
  %address.local4400 = or disjoint i64 %address.local4278, 66, !dbg !53
  %address.local4401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4400, !dbg !53
  store half %2116, ptr addrspace(1) %address.local4401, align 2, !dbg !53
  %address.local4403 = or disjoint i64 %address.local4278, 67, !dbg !53
  %address.local4404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4403, !dbg !53
  store half %2115, ptr addrspace(1) %address.local4404, align 2, !dbg !53
  %address.local4406 = or disjoint i64 %address.local4278, 80, !dbg !53
  %address.local4407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4406, !dbg !53
  store half %2112, ptr addrspace(1) %address.local4407, align 2, !dbg !53
  %address.local4409 = or disjoint i64 %address.local4278, 81, !dbg !53
  %address.local4410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4409, !dbg !53
  store half %2111, ptr addrspace(1) %address.local4410, align 2, !dbg !53
  %address.local4412 = or disjoint i64 %address.local4278, 82, !dbg !53
  %address.local4413 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4412, !dbg !53
  store half %2110, ptr addrspace(1) %address.local4413, align 2, !dbg !53
  %address.local4415 = or disjoint i64 %address.local4278, 83, !dbg !53
  %address.local4416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4415, !dbg !53
  store half %2109, ptr addrspace(1) %address.local4416, align 2, !dbg !53
  %address.local4421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4394, !dbg !53
  store half %2106, ptr addrspace(1) %address.local4421, align 2, !dbg !53
  %address.local4423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4397, !dbg !53
  store half %2105, ptr addrspace(1) %address.local4423, align 2, !dbg !53
  %address.local4425 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4400, !dbg !53
  store half %2104, ptr addrspace(1) %address.local4425, align 2, !dbg !53
  %address.local4427 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4403, !dbg !53
  store half %2103, ptr addrspace(1) %address.local4427, align 2, !dbg !53
  %address.local4429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4406, !dbg !53
  store half %2100, ptr addrspace(1) %address.local4429, align 2, !dbg !53
  %address.local4431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4409, !dbg !53
  store half %2099, ptr addrspace(1) %address.local4431, align 2, !dbg !53
  %address.local4433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4412, !dbg !53
  store half %2098, ptr addrspace(1) %address.local4433, align 2, !dbg !53
  %address.local4435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4415, !dbg !53
  store half %2097, ptr addrspace(1) %address.local4435, align 2, !dbg !53
  %2119 = fmul <4 x float> %2000, %1626, !dbg !51
  %2120 = fptrunc <4 x float> %2119 to <4 x half>, !dbg !52
  %2121 = extractelement <4 x half> %2120, i64 3, !dbg !52
  %2122 = extractelement <4 x half> %2120, i64 2, !dbg !52
  %2123 = extractelement <4 x half> %2120, i64 1, !dbg !52
  %2124 = extractelement <4 x half> %2120, i64 0, !dbg !52
  %2125 = fmul <4 x float> %2000, %1593, !dbg !51
  %2126 = fptrunc <4 x float> %2125 to <4 x half>, !dbg !52
  %2127 = extractelement <4 x half> %2126, i64 3, !dbg !52
  %2128 = extractelement <4 x half> %2126, i64 2, !dbg !52
  %2129 = extractelement <4 x half> %2126, i64 1, !dbg !52
  %2130 = extractelement <4 x half> %2126, i64 0, !dbg !52
  %2131 = fmul <4 x float> %1999, %1559, !dbg !51
  %2132 = fptrunc <4 x float> %2131 to <4 x half>, !dbg !52
  %2133 = extractelement <4 x half> %2132, i64 3, !dbg !52
  %2134 = extractelement <4 x half> %2132, i64 2, !dbg !52
  %2135 = extractelement <4 x half> %2132, i64 1, !dbg !52
  %2136 = extractelement <4 x half> %2132, i64 0, !dbg !52
  %2137 = fmul <4 x float> %1999, %1526, !dbg !51
  %2138 = fptrunc <4 x float> %2137 to <4 x half>, !dbg !52
  %2139 = extractelement <4 x half> %2138, i64 3, !dbg !52
  %2140 = extractelement <4 x half> %2138, i64 2, !dbg !52
  %2141 = extractelement <4 x half> %2138, i64 1, !dbg !52
  %2142 = extractelement <4 x half> %2138, i64 0, !dbg !52
  %address.local4452 = or disjoint i64 %address.local4278, 96, !dbg !53
  %address.local4453 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4452, !dbg !53
  store half %2142, ptr addrspace(1) %address.local4453, align 2, !dbg !53
  %address.local4455 = or disjoint i64 %address.local4278, 97, !dbg !53
  %address.local4456 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4455, !dbg !53
  store half %2141, ptr addrspace(1) %address.local4456, align 2, !dbg !53
  %address.local4458 = or disjoint i64 %address.local4278, 98, !dbg !53
  %address.local4459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4458, !dbg !53
  store half %2140, ptr addrspace(1) %address.local4459, align 2, !dbg !53
  %address.local4461 = or disjoint i64 %address.local4278, 99, !dbg !53
  %address.local4462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4461, !dbg !53
  store half %2139, ptr addrspace(1) %address.local4462, align 2, !dbg !53
  %address.local4464 = or disjoint i64 %address.local4278, 112, !dbg !53
  %address.local4465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4464, !dbg !53
  store half %2136, ptr addrspace(1) %address.local4465, align 2, !dbg !53
  %address.local4467 = or disjoint i64 %address.local4278, 113, !dbg !53
  %address.local4468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4467, !dbg !53
  store half %2135, ptr addrspace(1) %address.local4468, align 2, !dbg !53
  %address.local4470 = or disjoint i64 %address.local4278, 114, !dbg !53
  %address.local4471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4470, !dbg !53
  store half %2134, ptr addrspace(1) %address.local4471, align 2, !dbg !53
  %address.local4473 = or i64 %2069, 115, !dbg !53
  %address.local4474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4473, !dbg !53
  store half %2133, ptr addrspace(1) %address.local4474, align 2, !dbg !53
  %address.local4479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4452, !dbg !53
  store half %2130, ptr addrspace(1) %address.local4479, align 2, !dbg !53
  %address.local4481 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4455, !dbg !53
  store half %2129, ptr addrspace(1) %address.local4481, align 2, !dbg !53
  %address.local4483 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4458, !dbg !53
  store half %2128, ptr addrspace(1) %address.local4483, align 2, !dbg !53
  %address.local4485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4461, !dbg !53
  store half %2127, ptr addrspace(1) %address.local4485, align 2, !dbg !53
  %address.local4487 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4464, !dbg !53
  store half %2124, ptr addrspace(1) %address.local4487, align 2, !dbg !53
  %address.local4489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4467, !dbg !53
  store half %2123, ptr addrspace(1) %address.local4489, align 2, !dbg !53
  %address.local4491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4470, !dbg !53
  store half %2122, ptr addrspace(1) %address.local4491, align 2, !dbg !53
  %address.local4493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4473, !dbg !53
  store half %2121, ptr addrspace(1) %address.local4493, align 2, !dbg !53
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
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #3

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
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_cvt_pack"="rtz" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
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
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/shared_memory_plan/causal")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 89, column: 20, scope: !3)
!8 = !DILocation(line: 23, column: 3, scope: !3)
!9 = !DILocation(line: 70, column: 22, scope: !3)
!10 = !{i32 0, i32 256}
!11 = !DILocation(line: 72, column: 24, scope: !3)
!12 = !DILocation(line: 60, column: 24, scope: !3)
!13 = !DILocation(line: 0, scope: !3)
!14 = !DILocation(line: 71, column: 22, scope: !3)
!15 = !DILocation(line: 77, column: 19, scope: !3)
!16 = !DILocation(line: 74, column: 23, scope: !3)
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
!40 = !{i32 23}
!41 = !DILocation(line: 101, column: 20, scope: !3)
!42 = !DILocation(line: 90, column: 22, scope: !3)
!43 = !DILocation(line: 91, column: 23, scope: !3)
!44 = !DILocation(line: 92, column: 20, scope: !3)
!45 = !DILocation(line: 103, column: 22, scope: !3)
!46 = !DILocation(line: 93, column: 22, scope: !3)
!47 = !DILocation(line: 104, column: 20, scope: !3)
!48 = !DILocation(line: 94, column: 22, scope: !3)
!49 = !DILocation(line: 96, column: 16, scope: !3)
!50 = !DILocation(line: 95, column: 20, scope: !3)
!51 = !DILocation(line: 107, column: 23, scope: !3)
!52 = !DILocation(line: 108, column: 19, scope: !3)
!53 = !DILocation(line: 109, column: 9, scope: !3)
