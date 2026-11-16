; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/shared_memory_plan/causal/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/shared_memory_plan/causal/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !31 {
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
  %24 = and i64 %23, 576460752303423484, !dbg !5
  %25 = and i64 %18, 3, !dbg !5
  %26 = or disjoint i64 %24, %25, !dbg !5
  %27 = sub nsw i64 31, %26, !dbg !5
  %28 = lshr i64 %22, 2, !dbg !5
  %29 = and i64 %28, 31, !dbg !5
  %30 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %31 = sext i32 %30 to i64, !dbg !5
  %32 = icmp ugt i64 %22, 1023, !dbg !9
  %33 = add nsw i64 %26, -32, !dbg !9
  %34 = select i1 %32, i64 %33, i64 %27, !dbg !9
  %35 = sdiv i64 %34, 32, !dbg !9
  %36 = sext i1 %32 to i64, !dbg !9
  %37 = xor i64 %35, %36, !dbg !9
  %38 = add nsw i64 %28, %37, !dbg !9
  %39 = and i64 %38, 31, !dbg !9
  %40 = shl i64 %27, 7, !dbg !9
  %41 = mul nsw i64 %37, -4096, !dbg !9
  %42 = add i64 %40, %41, !dbg !9
  %43 = shl nsw i64 %42, 7, !dbg !9
  %44 = call range(i32 0, 256) i32 @llvm.amdgcn.workitem.id.x(), !dbg !9
  %45 = zext nneg i32 %44 to i64, !dbg !9
  %.idx = shl nuw nsw i64 %39, 20, !dbg !9
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %44) #9, !dbg !9, !range !33, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !9
  %address.local941 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local942 = and i64 %address.local941, 1536, !dbg !9
  %address.local943 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local942, !dbg !9
  %address.local944 = ptrtoint ptr addrspace(3) %address.local943 to i64, !dbg !9
  %address.local945 = trunc i64 %address.local944 to i32, !dbg !9
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local945), !dbg !9
  %address.local946 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local947 = and i64 %address.local946, 16256, !dbg !9
  %address.local949 = lshr i64 %address.local, 4, !dbg !9
  %address.local950 = and i64 %address.local949, 8, !dbg !9
  %47 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx4694 = shl i64 %42, 8, !dbg !9
  %48 = add i64 %.idx, %.idx4694, !dbg !9
  %address.local948.idx4695 = or disjoint i64 %address.local947, %address.local950, !dbg !9
  %49 = shl nuw nsw i64 %address.local948.idx4695, 1, !dbg !9
  %50 = or disjoint i64 %48, %49, !dbg !9
  %dma.buffer.offset4660 = trunc i64 %50 to i32, !dbg !9
  %51 = insertelement <2 x i64> poison, i64 %47, i64 0, !dbg !9
  %52 = bitcast <2 x i64> %51 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc4661 = shufflevector <4 x i32> %52, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %53 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset4660, <4 x i32> %dma.buffer.rsrc4661) #10, !dbg !9
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local943, i64 4096, !dbg !9
  %address.local956 = ptrtoint ptr addrspace(3) %address.local955 to i64, !dbg !9
  %address.local957 = trunc i64 %address.local956 to i32, !dbg !9
  %54 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local957), !dbg !9
  %dma.buffer.offset4658 = or disjoint i32 %dma.buffer.offset4660, 32, !dbg !9
  %55 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %54, i32 %dma.buffer.offset4658, <4 x i32> %dma.buffer.rsrc4661) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %56 = shl nuw nsw i64 %39, 19, !dbg !9
  %57 = add i64 %56, %43, !dbg !9
  %address.local965 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local942, !dbg !9
  %address.local966 = ptrtoint ptr addrspace(3) %address.local965 to i64, !dbg !9
  %address.local967 = trunc i64 %address.local966 to i32, !dbg !9
  %58 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local967), !dbg !9
  %59 = or disjoint i64 %57, 32, !dbg !9
  %60 = or disjoint i64 %address.local950, %59, !dbg !9
  %61 = or i64 %60, %address.local947, !dbg !9
  %.tr = trunc i64 %61 to i32, !dbg !9
  %dma.buffer.offset4656 = shl i32 %.tr, 1, !dbg !9
  %62 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %58, i32 %dma.buffer.offset4656, <4 x i32> %dma.buffer.rsrc4661) #10, !dbg !9
  %address.local977 = getelementptr i8, ptr addrspace(3) %address.local965, i64 4096, !dbg !9
  %address.local978 = ptrtoint ptr addrspace(3) %address.local977 to i64, !dbg !9
  %address.local979 = trunc i64 %address.local978 to i32, !dbg !9
  %63 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local979), !dbg !9
  %.idx46994700 = or disjoint i64 %57, %address.local947, !dbg !9
  %64 = or disjoint i64 %.idx46994700, %address.local950, !dbg !9
  %.tr4701 = trunc i64 %64 to i32, !dbg !9
  %65 = shl i32 %.tr4701, 1, !dbg !9
  %dma.buffer.offset4654 = or i32 %65, 96, !dbg !9
  %66 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %63, i32 %dma.buffer.offset4654, <4 x i32> %dma.buffer.rsrc4661) #10, !dbg !9
  %67 = lshr i64 %45, 6, !dbg !13
  %68 = and i64 %45, 240, !dbg !18
  %address.local984 = lshr i32 %address.thread, 2, !dbg !13
  %address.local985 = and i32 %address.local984, 4, !dbg !13
  %address.local986 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local987 = and i32 %address.local986, 384, !dbg !13
  %address.local988 = or disjoint i32 %address.local985, %address.local987, !dbg !13
  %address.local989 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local990 = and i32 %address.local989, 120, !dbg !13
  %address.local991 = or disjoint i32 %address.local988, %address.local990, !dbg !13
  %address.local992 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local993 = and i32 %address.local992, 1024, !dbg !13
  %address.local994 = or disjoint i32 %address.local991, %address.local993, !dbg !13
  %address.local995 = zext nneg i32 %address.local994 to i64, !dbg !13
  %address.local996 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local995, !dbg !13
  %69 = load <4 x half>, ptr addrspace(3) %address.local996, align 8, !dbg !13
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local996, i64 4096, !dbg !13
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !13
  %71 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !13
  %lds.const.gep4494 = getelementptr i8, ptr addrspace(3) %address.local996, i64 1024, !dbg !13
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep4494, align 8, !dbg !13
  %73 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !13
  %lds.const.gep4495 = getelementptr i8, ptr addrspace(3) %address.local996, i64 5120, !dbg !13
  %74 = load <4 x half>, ptr addrspace(3) %lds.const.gep4495, align 8, !dbg !13
  %75 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !13
  %76 = fmul <4 x half> %74, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %77 = or disjoint i64 %57, 64, !dbg !9
  %78 = or disjoint i64 %address.local950, %77, !dbg !9
  %79 = or i64 %78, %address.local947, !dbg !9
  %.tr4704 = trunc i64 %79 to i32, !dbg !9
  %dma.buffer.offset4652 = shl i32 %.tr4704, 1, !dbg !9
  %80 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset4652, <4 x i32> %dma.buffer.rsrc4661) #10, !dbg !9
  %dma.buffer.offset4650 = or i32 %65, 160, !dbg !9
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %54, i32 %dma.buffer.offset4650, <4 x i32> %dma.buffer.rsrc4661) #10, !dbg !9
  %lds.const.gep4496 = getelementptr i8, ptr addrspace(3) %address.local996, i64 8192, !dbg !13
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep4496, align 8, !dbg !13
  %lds.const.gep4497 = getelementptr i8, ptr addrspace(3) %address.local996, i64 12288, !dbg !13
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep4497, align 8, !dbg !13
  %lds.const.gep4498 = getelementptr i8, ptr addrspace(3) %address.local996, i64 9216, !dbg !13
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep4498, align 8, !dbg !13
  %85 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !13
  %lds.const.gep4499 = getelementptr i8, ptr addrspace(3) %address.local996, i64 13312, !dbg !13
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep4499, align 8, !dbg !13
  %87 = fmul <4 x half> %83, splat (half 0xH3015), !dbg !13
  %88 = fmul <4 x half> %84, splat (half 0xH3015), !dbg !13
  %89 = fmul <4 x half> %86, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %90 = or disjoint i64 %57, 96, !dbg !9
  %91 = or disjoint i64 %address.local950, %90, !dbg !9
  %92 = or i64 %91, %address.local947, !dbg !9
  %.tr4710 = trunc i64 %92 to i32, !dbg !9
  %dma.buffer.offset4648 = shl i32 %.tr4710, 1, !dbg !9
  %93 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %58, i32 %dma.buffer.offset4648, <4 x i32> %dma.buffer.rsrc4661) #10, !dbg !9
  %dma.buffer.offset4646 = or i32 %65, 224, !dbg !9
  %94 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %63, i32 %dma.buffer.offset4646, <4 x i32> %dma.buffer.rsrc4661) #10, !dbg !9
  %95 = load <4 x half>, ptr addrspace(3) %address.local996, align 8, !dbg !13
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !13
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep4494, align 8, !dbg !13
  %98 = fmul <4 x half> %95, splat (half 0xH3015), !dbg !13
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep4495, align 8, !dbg !13
  %100 = fmul <4 x half> %96, splat (half 0xH3015), !dbg !13
  %101 = fmul <4 x half> %97, splat (half 0xH3015), !dbg !13
  %102 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %103 = load <4 x half>, ptr addrspace(3) %lds.const.gep4496, align 8, !dbg !13
  %104 = load <4 x half>, ptr addrspace(3) %lds.const.gep4497, align 8, !dbg !13
  %105 = load <4 x half>, ptr addrspace(3) %lds.const.gep4498, align 8, !dbg !13
  %106 = fmul <4 x half> %103, splat (half 0xH3015), !dbg !13
  %107 = load <4 x half>, ptr addrspace(3) %lds.const.gep4499, align 8, !dbg !13
  %108 = fmul <4 x half> %104, splat (half 0xH3015), !dbg !13
  %109 = fmul <4 x half> %105, splat (half 0xH3015), !dbg !13
  %110 = fmul <4 x half> %107, splat (half 0xH3015), !dbg !13
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  %111 = sub nsw i64 32, %26, !dbg !15
  %112 = shl nuw nsw i64 %29, 12, !dbg !16
  %113 = shl nsw i64 %31, 17, !dbg !16
  %114 = shl nuw nsw i64 %29, 7, !dbg !16
  %115 = shl nsw i64 %31, 12, !dbg !16
  %116 = or disjoint i64 %114, %115, !dbg !16
  %117 = shl nuw nsw i64 %45, 2, !dbg !17
  %118 = add i64 %40, %45, !dbg !18
  %119 = shl nuw nsw i64 %67, 4, !dbg !18
  %120 = add i64 %118, %119, !dbg !18
  %121 = sub i64 %120, %68, !dbg !18
  %122 = and i64 %45, 48, !dbg !18
  %123 = trunc i64 %121 to i32, !dbg !18
  %124 = add i32 %123, 1, !dbg !18
  %125 = add i32 %123, 65, !dbg !18
  br label %126, !dbg !6

126:                                              ; preds = %135, %4
  %127 = phi i64 [ %1498, %135 ], [ 0, %4 ], !dbg !6
  %128 = phi [2 x <8 x float>] [ %1483, %135 ], [ zeroinitializer, %4 ], !dbg !6
  %129 = phi [2 x <1 x float>] [ %1243, %135 ], [ zeroinitializer, %4 ], !dbg !6
  %130 = phi [2 x <1 x float>] [ %1485, %135 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !6
  %131 = phi [2 x <8 x float>] [ %1489, %135 ], [ zeroinitializer, %4 ], !dbg !6
  %132 = phi [2 x <8 x float>] [ %1493, %135 ], [ zeroinitializer, %4 ], !dbg !6
  %133 = phi [2 x <8 x float>] [ %1497, %135 ], [ zeroinitializer, %4 ], !dbg !6
  %134 = icmp slt i64 %127, %111, !dbg !6
  br i1 %134, label %135, label %1499, !dbg !6

135:                                              ; preds = %126
  %136 = lshr i64 %127, 5, !dbg !16
  %137 = shl nsw i64 %127, 7, !dbg !16
  %138 = shl i64 %136, 19, !dbg !16
  %139 = shl i64 %127, 14, !dbg !16
  %140 = sub i64 %139, %138, !dbg !16
  %141 = lshr i64 %127, 12, !dbg !16
  %142 = add nuw nsw i64 %141, %28, !dbg !16
  %143 = shl i64 %142, 19, !dbg !16
  %144 = and i64 %143, 16252928, !dbg !16
  %145 = and i64 %136, 127, !dbg !16
  %146 = shl nuw nsw i64 %67, 2, !dbg !16
  %147 = add nuw nsw i64 %117, %146, !dbg !16
  %148 = add nuw nsw i64 %67, %45, !dbg !16
  %149 = shl nuw nsw i64 %148, 2, !dbg !16
  %150 = and i64 %149, 2040, !dbg !16
  %151 = sub nsw i64 %147, %150, !dbg !16
  %152 = lshr i32 %44, 5, !dbg !16
  %153 = lshr i32 %44, 6, !dbg !16
  %154 = lshr i32 %44, 3, !dbg !16
  %155 = and i32 %154, 1, !dbg !16
  %156 = and i32 %152, %155, !dbg !16
  %narrow871 = add nuw nsw i32 %153, %156, !dbg !16
  %157 = shl nuw nsw i32 %narrow871, 10, !dbg !16
  %158 = zext nneg i32 %157 to i64, !dbg !16
  %159 = shl nuw nsw i32 %44, 4, !dbg !16
  %160 = and i32 %159, 3584, !dbg !16
  %161 = shl nuw nsw i32 %44, 6, !dbg !16
  %162 = and i32 %161, 16256, !dbg !16
  %narrow872 = add nuw nsw i32 %160, %162, !dbg !16
  %163 = zext nneg i32 %narrow872 to i64, !dbg !16
  %164 = sub nsw i64 %163, %158, !dbg !16
  %address.thread1199 = call i32 asm sideeffect "", "=v,0"(i32 %44) #9, !dbg !16, !range !33, !frisk.register_lifetimes !1
  %address.local1200 = zext nneg i32 %address.thread1199 to i64, !dbg !9
  %address.local1201 = shl nuw nsw i64 %address.local1200, 2, !dbg !16
  %address.local1202 = and i64 %address.local1201, 768, !dbg !16
  %address.local1203 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1202, !dbg !16
  %address.local1204 = ptrtoint ptr addrspace(3) %address.local1203 to i64, !dbg !16
  %address.local1205 = trunc i64 %address.local1204 to i32, !dbg !16
  %165 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1205), !dbg !16
  %166 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %.idx47144719 = or disjoint i64 %144, %145, !dbg !16
  %167 = add i64 %.idx47144719, %140, !dbg !16
  %168 = and i64 %164, 16256, !dbg !16
  %169 = add nsw i64 %151, %168, !dbg !16
  %170 = add i64 %167, %169, !dbg !16
  %.tr4720 = trunc i64 %170 to i32, !dbg !16
  %dma.buffer.offset4692 = shl i32 %.tr4720, 1, !dbg !16
  %171 = insertelement <2 x i64> poison, i64 %166, i64 0, !dbg !16
  %172 = bitcast <2 x i64> %171 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc4693 = shufflevector <4 x i32> %172, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %173 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %165, i32 %dma.buffer.offset4692, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %174 = add nuw nsw i64 %117, 1040, !dbg !16
  %175 = add nuw nsw i64 %174, %146, !dbg !16
  %176 = add nuw nsw i64 %45, 260, !dbg !16
  %177 = add nuw nsw i64 %176, %67, !dbg !16
  %178 = shl nuw nsw i64 %177, 2, !dbg !16
  %179 = and i64 %178, 8184, !dbg !16
  %180 = sub nsw i64 %175, %179, !dbg !16
  %181 = or disjoint i32 %153, 4, !dbg !16
  %narrow873 = add nuw nsw i32 %181, %156, !dbg !16
  %182 = shl nuw nsw i32 %44, 5, !dbg !16
  %183 = and i32 %182, 7168, !dbg !16
  %184 = or disjoint i32 %183, 8192, !dbg !16
  %185 = shl nuw nsw i32 %44, 7, !dbg !16
  %186 = and i32 %185, 32512, !dbg !16
  %narrow874 = add nuw nsw i32 %184, %186, !dbg !16
  %address.local1209 = getelementptr i8, ptr addrspace(3) %address.local1203, i64 2048, !dbg !16
  %address.local1210 = ptrtoint ptr addrspace(3) %address.local1209 to i64, !dbg !16
  %address.local1211 = trunc i64 %address.local1210 to i32, !dbg !16
  %187 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1211), !dbg !16
  %188 = shl i64 %167, 1, !dbg !16
  %.idx4724 = shl nsw i64 %180, 1, !dbg !16
  %189 = add nsw i64 %.idx4724, 16, !dbg !16
  %190 = shl nuw nsw i32 %narrow873, 11, !dbg !16
  %191 = sub nsw i32 %narrow874, %190, !dbg !16
  %192 = and i32 %191, 32512, !dbg !16
  %.idx4725 = zext nneg i32 %192 to i64, !dbg !16
  %193 = add i64 %189, %.idx4725, !dbg !16
  %194 = add i64 %188, %193, !dbg !16
  %dma.buffer.offset4690 = trunc i64 %194 to i32, !dbg !16
  %195 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %187, i32 %dma.buffer.offset4690, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %196 = add nuw nsw i64 %117, 2080, !dbg !16
  %197 = add nuw nsw i64 %196, %146, !dbg !16
  %198 = add nuw nsw i64 %45, 520, !dbg !16
  %199 = add nuw nsw i64 %198, %67, !dbg !16
  %200 = shl nuw nsw i64 %199, 2, !dbg !16
  %201 = and i64 %200, 8184, !dbg !16
  %202 = sub nsw i64 %197, %201, !dbg !16
  %203 = or disjoint i32 %153, 8, !dbg !16
  %narrow876 = add nuw nsw i32 %203, %156, !dbg !16
  %204 = shl nuw nsw i32 %narrow876, 10, !dbg !16
  %205 = zext nneg i32 %204 to i64, !dbg !16
  %206 = or disjoint i32 %160, 8192, !dbg !16
  %narrow877 = add nuw nsw i32 %206, %162, !dbg !16
  %207 = zext nneg i32 %narrow877 to i64, !dbg !16
  %208 = sub nsw i64 %207, %205, !dbg !16
  %address.local1215 = getelementptr i8, ptr addrspace(3) %address.local1203, i64 4096, !dbg !16
  %address.local1216 = ptrtoint ptr addrspace(3) %address.local1215 to i64, !dbg !16
  %address.local1217 = trunc i64 %address.local1216 to i32, !dbg !16
  %209 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1217), !dbg !16
  %.idx4730 = shl nsw i64 %202, 1, !dbg !16
  %210 = add nsw i64 %.idx4730, 32, !dbg !16
  %211 = shl nsw i64 %208, 1, !dbg !16
  %.idx4731 = and i64 %211, 32512, !dbg !16
  %212 = add i64 %210, %.idx4731, !dbg !16
  %213 = add i64 %188, %212, !dbg !16
  %dma.buffer.offset4688 = trunc i64 %213 to i32, !dbg !16
  %214 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %209, i32 %dma.buffer.offset4688, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %215 = add nuw nsw i64 %117, 3120, !dbg !16
  %216 = add nuw nsw i64 %215, %146, !dbg !16
  %217 = add nuw nsw i64 %45, 780, !dbg !16
  %218 = add nuw nsw i64 %217, %67, !dbg !16
  %219 = shl nuw nsw i64 %218, 2, !dbg !16
  %220 = and i64 %219, 16376, !dbg !16
  %221 = sub nsw i64 %216, %220, !dbg !16
  %222 = or disjoint i32 %153, 12, !dbg !16
  %narrow878 = add nuw nsw i32 %222, %156, !dbg !16
  %223 = or disjoint i32 %183, 24576, !dbg !16
  %narrow879 = add nuw nsw i32 %223, %186, !dbg !16
  %address.local1221 = getelementptr i8, ptr addrspace(3) %address.local1203, i64 6144, !dbg !16
  %address.local1222 = ptrtoint ptr addrspace(3) %address.local1221 to i64, !dbg !16
  %address.local1223 = trunc i64 %address.local1222 to i32, !dbg !16
  %224 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1223), !dbg !16
  %225 = add i64 %167, %221, !dbg !16
  %226 = shl nuw nsw i32 %narrow878, 11, !dbg !16
  %227 = sub nsw i32 %narrow879, %226, !dbg !16
  %228 = and i32 %227, 32512, !dbg !16
  %.tr4739 = trunc i64 %225 to i32, !dbg !16
  %229 = shl i32 %.tr4739, 1, !dbg !16
  %230 = add i32 %229, 48, !dbg !16
  %dma.buffer.offset4686 = add i32 %230, %228, !dbg !16
  %231 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %224, i32 %dma.buffer.offset4686, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %232 = add nuw i64 %127, %112, !dbg !16
  %233 = add i64 %232, %113, !dbg !16
  %234 = add nsw i64 %136, %116, !dbg !16
  %235 = add i64 %233, 1024, !dbg !16
  %236 = icmp slt i64 %235, 0, !dbg !16
  %237 = sub i64 -1025, %233, !dbg !16
  %238 = select i1 %236, i64 %237, i64 %235, !dbg !16
  %239 = sdiv i64 %238, 4096, !dbg !16
  %.lobit124 = ashr i64 %235, 63, !dbg !16
  %240 = xor i64 %239, %.lobit124, !dbg !16
  %241 = shl i64 %240, 19, !dbg !16
  %242 = and i64 %241, 16252928, !dbg !16
  %243 = icmp slt i64 %234, -32, !dbg !16
  %244 = sub nsw i64 -33, %234, !dbg !16
  %245 = add nsw i64 %234, 32, !dbg !16
  %246 = select i1 %243, i64 %244, i64 %245, !dbg !16
  %247 = sdiv i64 %246, 128, !dbg !16
  %248 = sext i1 %243 to i64, !dbg !16
  %249 = xor i64 %247, %248, !dbg !16
  %250 = shl nsw i64 %249, 7, !dbg !16
  %251 = sub i64 %116, %250, !dbg !16
  %address.local1228 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1202, !dbg !16
  %address.local1229 = ptrtoint ptr addrspace(3) %address.local1228 to i64, !dbg !16
  %address.local1230 = trunc i64 %address.local1229 to i32, !dbg !16
  %252 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1230), !dbg !16
  %.idx47404746 = add nuw nsw i64 %242, %136, !dbg !16
  %253 = add i64 %.idx47404746, %251, !dbg !16
  %254 = add i64 %253, %140, !dbg !16
  %255 = add i64 %254, %169, !dbg !16
  %.tr4747 = trunc i64 %255 to i32, !dbg !16
  %256 = shl i32 %.tr4747, 1, !dbg !16
  %dma.buffer.offset4684 = add i32 %256, 64, !dbg !16
  %257 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %252, i32 %dma.buffer.offset4684, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %address.local1234 = getelementptr i8, ptr addrspace(3) %address.local1228, i64 2048, !dbg !16
  %address.local1235 = ptrtoint ptr addrspace(3) %address.local1234 to i64, !dbg !16
  %address.local1236 = trunc i64 %address.local1235 to i32, !dbg !16
  %258 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1236), !dbg !16
  %259 = shl i64 %254, 1, !dbg !16
  %260 = add i64 %259, 64, !dbg !16
  %261 = add i64 %260, %193, !dbg !16
  %dma.buffer.offset4682 = trunc i64 %261 to i32, !dbg !16
  %262 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %258, i32 %dma.buffer.offset4682, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %address.local1240 = getelementptr i8, ptr addrspace(3) %address.local1228, i64 4096, !dbg !16
  %address.local1241 = ptrtoint ptr addrspace(3) %address.local1240 to i64, !dbg !16
  %address.local1242 = trunc i64 %address.local1241 to i32, !dbg !16
  %263 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1242), !dbg !16
  %264 = add i64 %260, %212, !dbg !16
  %dma.buffer.offset4680 = trunc i64 %264 to i32, !dbg !16
  %265 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %263, i32 %dma.buffer.offset4680, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %address.local1246 = getelementptr i8, ptr addrspace(3) %address.local1228, i64 6144, !dbg !16
  %address.local1247 = ptrtoint ptr addrspace(3) %address.local1246 to i64, !dbg !16
  %address.local1248 = trunc i64 %address.local1247 to i32, !dbg !16
  %266 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1248), !dbg !16
  %267 = add i64 %254, %221, !dbg !16
  %.tr4769 = trunc i64 %267 to i32, !dbg !16
  %268 = shl i32 %.tr4769, 1, !dbg !16
  %269 = add i32 %268, 112, !dbg !16
  %dma.buffer.offset4678 = add i32 %269, %228, !dbg !16
  %270 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %266, i32 %dma.buffer.offset4678, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %address.local1251 = mul nuw nsw i64 %address.local1200, 160, !dbg !17
  %address.local1252 = lshr i64 %address.local1200, 2, !dbg !17
  %address.local1253 = mul nsw i64 %address.local1252, -640, !dbg !17
  %address.local1254 = add nsw i64 %address.local1251, %address.local1253, !dbg !17
  %address.local1255 = shl nuw nsw i64 %address.local1200, 1, !dbg !17
  %address.local1256 = and i64 %address.local1255, 24, !dbg !17
  %address.local1257 = or disjoint i64 %address.local1254, %address.local1256, !dbg !17
  %address.local1258 = shl nuw nsw i64 %address.local1200, 5, !dbg !17
  %address.local1259 = and i64 %address.local1258, 64, !dbg !17
  %address.local1260 = sub nsw i64 %address.local1257, %address.local1259, !dbg !17
  %address.local1261 = and i64 %address.local1200, 63, !dbg !13
  %address.local1262 = lshr i64 %address.local1261, 4, !dbg !13
  %address.local1263 = shl nuw nsw i64 %address.local1262, 2, !dbg !13
  %address.local1264 = add nsw i64 %address.local1260, %address.local1263, !dbg !17
  %address.local1265 = lshr i64 %address.local1261, 5, !dbg !13
  %address.local1266 = mul nuw nsw i64 %address.local1265, 1016, !dbg !13
  %address.local1267 = add i64 %address.local1264, %address.local1266, !dbg !17
  %address.local1268 = lshr i32 %address.thread1199, 1, !dbg !17
  %address.local1269 = and i32 %address.local1268, 16, !dbg !17
  %address.local1270 = shl nuw nsw i32 %address.thread1199, 1, !dbg !17
  %address.local1271 = and i32 %address.local1270, 4, !dbg !17
  %address.local1272 = or disjoint i32 %address.local1269, %address.local1271, !dbg !17
  %address.local1273 = zext nneg i32 %address.local1272 to i64, !dbg !17
  %address.local1274 = add i64 %address.local1267, %address.local1273, !dbg !17
  %address.local1276 = and i32 %address.local1268, 1, !dbg !17
  %address.local1277 = zext nneg i32 %address.local1276 to i64, !dbg !17
  %address.local1278 = and i64 %address.local1262, %address.local1277, !dbg !17
  %address.local1279 = lshr i32 %address.thread1199, 4, !dbg !17
  %address.local1280 = and i32 %address.local1279, 2, !dbg !17
  %address.local1281 = zext nneg i32 %address.local1280 to i64, !dbg !17
  %address.local1282 = or disjoint i64 %address.local1278, %address.local1281, !dbg !17
  %address.local1283 = shl nuw nsw i64 %address.local1282, 3, !dbg !17
  %address.local1284 = sub i64 %address.local1274, %address.local1283, !dbg !17
  %address.local1285 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1284, !dbg !17
  %271 = load <4 x half>, ptr addrspace(3) %address.local1285, align 8, !dbg !17, !frisk.k_lds_pair !35
  %lds.const.gep4510 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 128, !dbg !17
  %272 = load <4 x half>, ptr addrspace(3) %lds.const.gep4510, align 8, !dbg !17, !frisk.k_lds_pair !35
  %lds.const.gep4508 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 4096, !dbg !17
  %273 = load <4 x half>, ptr addrspace(3) %lds.const.gep4508, align 8, !dbg !17, !frisk.k_lds_pair !36
  %lds.const.gep4511 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 4224, !dbg !17
  %274 = load <4 x half>, ptr addrspace(3) %lds.const.gep4511, align 8, !dbg !17, !frisk.k_lds_pair !36
  %address.local1314 = and i32 %address.local1270, 24, !dbg !17
  %address.local1315 = shl nuw nsw i32 %address.thread1199, 8, !dbg !17
  %address.local1316 = and i32 %address.local1315, 768, !dbg !17
  %address.local1318 = shl nuw nsw i32 %address.thread1199, 5, !dbg !17
  %address.local1319 = and i32 %address.local1318, 96, !dbg !17
  %275 = or disjoint i32 %address.local1316, %address.local1319, !dbg !17
  %address.local1320 = or disjoint i32 %275, %address.local1314, !dbg !17
  %address.local1321 = zext nneg i32 %address.local1320 to i64, !dbg !17
  %276 = shl nuw nsw i64 %address.local1252, 4, !dbg !17
  %address.local1324 = sub nsw i64 %address.local1201, %276, !dbg !17
  %address.local1325 = and i32 %address.thread1199, 1, !dbg !17
  %address.local1326 = zext nneg i32 %address.local1325 to i64, !dbg !17
  %address.local1327 = or disjoint i64 %address.local1324, %address.local1326, !dbg !17
  %address.local1328 = add nuw nsw i64 %address.local1327, 1, !dbg !17
  %address.local1329 = sdiv i64 %address.local1328, 2, !dbg !17
  %address.local1333 = add nsw i64 %address.local1329, %address.local1277, !dbg !17
  %address.local1334 = shl i64 %address.local1333, 6, !dbg !17
  %address.local1335 = sub i64 %address.local1321, %address.local1334, !dbg !17
  %address.local1336 = add i64 %address.local1335, %address.local1263, !dbg !17
  %address.local1337 = add i64 %address.local1336, %address.local1266, !dbg !17
  %address.local1344 = add i64 %address.local1337, %address.local1273, !dbg !17
  %address.local1354 = sub i64 %address.local1344, %address.local1283, !dbg !17
  %277 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1354, !dbg !17
  %address.local1356 = getelementptr i8, ptr addrspace(3) %277, i64 64, !dbg !17
  %278 = load <4 x half>, ptr addrspace(3) %address.local1356, align 8, !dbg !17, !frisk.k_lds_pair !37
  %lds.const.gep4512 = getelementptr i8, ptr addrspace(3) %277, i64 192, !dbg !17
  %279 = load <4 x half>, ptr addrspace(3) %lds.const.gep4512, align 8, !dbg !17, !frisk.k_lds_pair !37
  %lds.const.gep4509 = getelementptr i8, ptr addrspace(3) %277, i64 4160, !dbg !17
  %280 = load <4 x half>, ptr addrspace(3) %lds.const.gep4509, align 8, !dbg !17, !frisk.k_lds_pair !38
  %lds.const.gep4513 = getelementptr i8, ptr addrspace(3) %277, i64 4288, !dbg !17
  %281 = load <4 x half>, ptr addrspace(3) %lds.const.gep4513, align 8, !dbg !17, !frisk.k_lds_pair !38
  %lds.const.gep4514 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 1024, !dbg !17
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep4514, align 8, !dbg !17, !frisk.k_lds_pair !34
  %lds.const.gep4518 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 1152, !dbg !17
  %283 = load <4 x half>, ptr addrspace(3) %lds.const.gep4518, align 8, !dbg !17, !frisk.k_lds_pair !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %271, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %lds.const.gep4515 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 5120, !dbg !17
  %285 = load <4 x half>, ptr addrspace(3) %lds.const.gep4515, align 8, !dbg !17, !frisk.k_lds_pair !39
  %lds.const.gep4519 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 5248, !dbg !17
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep4519, align 8, !dbg !17, !frisk.k_lds_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %273, <4 x float> %284) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %lds.const.gep4516 = getelementptr i8, ptr addrspace(3) %277, i64 1088, !dbg !17
  %288 = load <4 x half>, ptr addrspace(3) %lds.const.gep4516, align 8, !dbg !17, !frisk.k_lds_pair !40
  %lds.const.gep4520 = getelementptr i8, ptr addrspace(3) %277, i64 1216, !dbg !17
  %289 = load <4 x half>, ptr addrspace(3) %lds.const.gep4520, align 8, !dbg !17, !frisk.k_lds_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %278, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %lds.const.gep4517 = getelementptr i8, ptr addrspace(3) %277, i64 5184, !dbg !17
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep4517, align 8, !dbg !17, !frisk.k_lds_pair !41
  %lds.const.gep4521 = getelementptr i8, ptr addrspace(3) %277, i64 5312, !dbg !17
  %292 = load <4 x half>, ptr addrspace(3) %lds.const.gep4521, align 8, !dbg !17, !frisk.k_lds_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %280, <4 x float> %290) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %272, <4 x float> zeroinitializer) #9, !dbg !17
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %274, <4 x float> %294) #9, !dbg !17
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %279, <4 x float> zeroinitializer) #9, !dbg !17
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %281, <4 x float> %296) #9, !dbg !17
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %282, <4 x float> zeroinitializer) #9, !dbg !17
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %285, <4 x float> %298) #9, !dbg !17
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %288, <4 x float> zeroinitializer) #9, !dbg !17
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %291, <4 x float> %300) #9, !dbg !17
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %283, <4 x float> zeroinitializer) #9, !dbg !17
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %286, <4 x float> %302) #9, !dbg !17
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %289, <4 x float> zeroinitializer) #9, !dbg !17
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %292, <4 x float> %304) #9, !dbg !17
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %271, <4 x float> zeroinitializer) #9, !dbg !17
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %273, <4 x float> %306) #9, !dbg !17
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %278, <4 x float> zeroinitializer) #9, !dbg !17
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %280, <4 x float> %308) #9, !dbg !17
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %272, <4 x float> zeroinitializer) #9, !dbg !17
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %274, <4 x float> %310) #9, !dbg !17
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %279, <4 x float> zeroinitializer) #9, !dbg !17
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %281, <4 x float> %312) #9, !dbg !17
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %282, <4 x float> zeroinitializer) #9, !dbg !17
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %285, <4 x float> %314) #9, !dbg !17
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %288, <4 x float> zeroinitializer) #9, !dbg !17
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %291, <4 x float> %316) #9, !dbg !17
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %283, <4 x float> zeroinitializer) #9, !dbg !17
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %286, <4 x float> %318) #9, !dbg !17
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %289, <4 x float> zeroinitializer) #9, !dbg !17
  %321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %292, <4 x float> %320) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %322 = add i64 %233, 2048, !dbg !16
  %323 = icmp slt i64 %322, 0, !dbg !16
  %324 = sub i64 -2049, %233, !dbg !16
  %325 = select i1 %323, i64 %324, i64 %322, !dbg !16
  %326 = sdiv i64 %325, 4096, !dbg !16
  %.lobit165 = ashr i64 %322, 63, !dbg !16
  %327 = xor i64 %326, %.lobit165, !dbg !16
  %328 = shl i64 %327, 19, !dbg !16
  %329 = and i64 %328, 16252928, !dbg !16
  %330 = icmp slt i64 %234, -64, !dbg !16
  %331 = sub nsw i64 -65, %234, !dbg !16
  %332 = add nsw i64 %234, 64, !dbg !16
  %333 = select i1 %330, i64 %331, i64 %332, !dbg !16
  %334 = sdiv i64 %333, 128, !dbg !16
  %335 = sext i1 %330 to i64, !dbg !16
  %336 = xor i64 %334, %335, !dbg !16
  %337 = shl nsw i64 %336, 7, !dbg !16
  %338 = sub i64 %116, %337, !dbg !16
  %.idx47704776 = add nuw nsw i64 %329, %136, !dbg !16
  %339 = add i64 %.idx47704776, %338, !dbg !16
  %340 = add i64 %339, %140, !dbg !16
  %341 = add i64 %340, %169, !dbg !16
  %.tr4777 = trunc i64 %341 to i32, !dbg !16
  %342 = shl i32 %.tr4777, 1, !dbg !16
  %dma.buffer.offset4676 = add i32 %342, 128, !dbg !16
  %343 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %165, i32 %dma.buffer.offset4676, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %344 = shl i64 %340, 1, !dbg !16
  %345 = add i64 %344, 128, !dbg !16
  %346 = add i64 %345, %193, !dbg !16
  %dma.buffer.offset4674 = trunc i64 %346 to i32, !dbg !16
  %347 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %187, i32 %dma.buffer.offset4674, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %348 = add i64 %345, %212, !dbg !16
  %dma.buffer.offset4672 = trunc i64 %348 to i32, !dbg !16
  %349 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %209, i32 %dma.buffer.offset4672, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %350 = add i64 %340, %221, !dbg !16
  %.tr4799 = trunc i64 %350 to i32, !dbg !16
  %351 = shl i32 %.tr4799, 1, !dbg !16
  %352 = add i32 %351, 176, !dbg !16
  %dma.buffer.offset4670 = add i32 %352, %228, !dbg !16
  %353 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %224, i32 %dma.buffer.offset4670, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %lds.const.gep4536 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 8192, !dbg !17
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep4536, align 8, !dbg !17, !frisk.k_lds_pair !42
  %lds.const.gep4540 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 8320, !dbg !17
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep4540, align 8, !dbg !17, !frisk.k_lds_pair !42
  %lds.const.gep4537 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 12288, !dbg !17
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep4537, align 8, !dbg !17, !frisk.k_lds_pair !43
  %lds.const.gep4541 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 12416, !dbg !17
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep4541, align 8, !dbg !17, !frisk.k_lds_pair !43
  %lds.const.gep4538 = getelementptr i8, ptr addrspace(3) %277, i64 8256, !dbg !17
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep4538, align 8, !dbg !17, !frisk.k_lds_pair !44
  %lds.const.gep4542 = getelementptr i8, ptr addrspace(3) %277, i64 8384, !dbg !17
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep4542, align 8, !dbg !17, !frisk.k_lds_pair !44
  %lds.const.gep4539 = getelementptr i8, ptr addrspace(3) %277, i64 12352, !dbg !17
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep4539, align 8, !dbg !17, !frisk.k_lds_pair !45
  %lds.const.gep4543 = getelementptr i8, ptr addrspace(3) %277, i64 12480, !dbg !17
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep4543, align 8, !dbg !17, !frisk.k_lds_pair !45
  %lds.const.gep4544 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 9216, !dbg !17
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep4544, align 8, !dbg !17, !frisk.k_lds_pair !46
  %lds.const.gep4548 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 9344, !dbg !17
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep4548, align 8, !dbg !17, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %354, <4 x float> %287) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %lds.const.gep4545 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 13312, !dbg !17
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep4545, align 8, !dbg !17, !frisk.k_lds_pair !47
  %lds.const.gep4549 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 13440, !dbg !17
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep4549, align 8, !dbg !17, !frisk.k_lds_pair !47
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %356, <4 x float> %364) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %lds.const.gep4546 = getelementptr i8, ptr addrspace(3) %277, i64 9280, !dbg !17
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep4546, align 8, !dbg !17, !frisk.k_lds_pair !48
  %lds.const.gep4550 = getelementptr i8, ptr addrspace(3) %277, i64 9408, !dbg !17
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep4550, align 8, !dbg !17, !frisk.k_lds_pair !48
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %358, <4 x float> %293) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %lds.const.gep4547 = getelementptr i8, ptr addrspace(3) %277, i64 13376, !dbg !17
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep4547, align 8, !dbg !17, !frisk.k_lds_pair !49
  %lds.const.gep4551 = getelementptr i8, ptr addrspace(3) %277, i64 13504, !dbg !17
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep4551, align 8, !dbg !17, !frisk.k_lds_pair !49
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %360, <4 x float> %370) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %355, <4 x float> %295) #9, !dbg !17
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %357, <4 x float> %374) #9, !dbg !17
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %359, <4 x float> %297) #9, !dbg !17
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %361, <4 x float> %376) #9, !dbg !17
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %362, <4 x float> %299) #9, !dbg !17
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %365, <4 x float> %378) #9, !dbg !17
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %368, <4 x float> %301) #9, !dbg !17
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %371, <4 x float> %380) #9, !dbg !17
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %363, <4 x float> %303) #9, !dbg !17
  %383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %366, <4 x float> %382) #9, !dbg !17
  %384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %369, <4 x float> %305) #9, !dbg !17
  %385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %372, <4 x float> %384) #9, !dbg !17
  %386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %354, <4 x float> %307) #9, !dbg !17
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %356, <4 x float> %386) #9, !dbg !17
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %358, <4 x float> %309) #9, !dbg !17
  %389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %360, <4 x float> %388) #9, !dbg !17
  %390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %355, <4 x float> %311) #9, !dbg !17
  %391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %357, <4 x float> %390) #9, !dbg !17
  %392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %359, <4 x float> %313) #9, !dbg !17
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %361, <4 x float> %392) #9, !dbg !17
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %362, <4 x float> %315) #9, !dbg !17
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %365, <4 x float> %394) #9, !dbg !17
  %396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %368, <4 x float> %317) #9, !dbg !17
  %397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %371, <4 x float> %396) #9, !dbg !17
  %398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %363, <4 x float> %319) #9, !dbg !17
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %366, <4 x float> %398) #9, !dbg !17
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %369, <4 x float> %321) #9, !dbg !17
  %401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %372, <4 x float> %400) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %402 = add i64 %233, 3072, !dbg !16
  %403 = icmp slt i64 %402, 0, !dbg !16
  %404 = sub i64 -3073, %233, !dbg !16
  %405 = select i1 %403, i64 %404, i64 %402, !dbg !16
  %406 = sdiv i64 %405, 4096, !dbg !16
  %.lobit206 = ashr i64 %402, 63, !dbg !16
  %407 = xor i64 %406, %.lobit206, !dbg !16
  %408 = shl i64 %407, 19, !dbg !16
  %409 = and i64 %408, 16252928, !dbg !16
  %410 = icmp slt i64 %234, -96, !dbg !16
  %411 = sub nsw i64 -97, %234, !dbg !16
  %412 = add nsw i64 %234, 96, !dbg !16
  %413 = select i1 %410, i64 %411, i64 %412, !dbg !16
  %414 = sdiv i64 %413, 128, !dbg !16
  %415 = sext i1 %410 to i64, !dbg !16
  %416 = xor i64 %414, %415, !dbg !16
  %417 = shl nsw i64 %416, 7, !dbg !16
  %418 = sub i64 %116, %417, !dbg !16
  %.idx48004806 = add nuw nsw i64 %409, %136, !dbg !16
  %419 = add i64 %.idx48004806, %418, !dbg !16
  %420 = add i64 %419, %140, !dbg !16
  %421 = add i64 %420, %169, !dbg !16
  %.tr4807 = trunc i64 %421 to i32, !dbg !16
  %422 = shl i32 %.tr4807, 1, !dbg !16
  %dma.buffer.offset4668 = add i32 %422, 192, !dbg !16
  %423 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %252, i32 %dma.buffer.offset4668, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %424 = shl i64 %420, 1, !dbg !16
  %425 = add i64 %424, 192, !dbg !16
  %426 = add i64 %425, %193, !dbg !16
  %dma.buffer.offset4666 = trunc i64 %426 to i32, !dbg !16
  %427 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %258, i32 %dma.buffer.offset4666, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %428 = add i64 %425, %212, !dbg !16
  %dma.buffer.offset4664 = trunc i64 %428 to i32, !dbg !16
  %429 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %263, i32 %dma.buffer.offset4664, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %430 = add i64 %420, %221, !dbg !16
  %.tr4829 = trunc i64 %430 to i32, !dbg !16
  %431 = shl i32 %.tr4829, 1, !dbg !16
  %432 = add i32 %431, 240, !dbg !16
  %dma.buffer.offset4662 = add i32 %432, %228, !dbg !16
  %433 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %266, i32 %dma.buffer.offset4662, <4 x i32> %dma.buffer.rsrc4693) #10, !dbg !16
  %434 = load <4 x half>, ptr addrspace(3) %address.local1285, align 8, !dbg !17, !frisk.k_lds_pair !50
  %435 = load <4 x half>, ptr addrspace(3) %lds.const.gep4510, align 8, !dbg !17, !frisk.k_lds_pair !50
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep4508, align 8, !dbg !17, !frisk.k_lds_pair !51
  %437 = load <4 x half>, ptr addrspace(3) %lds.const.gep4511, align 8, !dbg !17, !frisk.k_lds_pair !51
  %438 = load <4 x half>, ptr addrspace(3) %address.local1356, align 8, !dbg !17, !frisk.k_lds_pair !52
  %439 = load <4 x half>, ptr addrspace(3) %lds.const.gep4512, align 8, !dbg !17, !frisk.k_lds_pair !52
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep4509, align 8, !dbg !17, !frisk.k_lds_pair !53
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep4513, align 8, !dbg !17, !frisk.k_lds_pair !53
  %442 = load <4 x half>, ptr addrspace(3) %lds.const.gep4514, align 8, !dbg !17, !frisk.k_lds_pair !54
  %443 = load <4 x half>, ptr addrspace(3) %lds.const.gep4518, align 8, !dbg !17, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %434, <4 x float> %367) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep4515, align 8, !dbg !17, !frisk.k_lds_pair !55
  %446 = load <4 x half>, ptr addrspace(3) %lds.const.gep4519, align 8, !dbg !17, !frisk.k_lds_pair !55
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %436, <4 x float> %444) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %448 = load <4 x half>, ptr addrspace(3) %lds.const.gep4516, align 8, !dbg !17, !frisk.k_lds_pair !56
  %449 = load <4 x half>, ptr addrspace(3) %lds.const.gep4520, align 8, !dbg !17, !frisk.k_lds_pair !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %438, <4 x float> %373) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %451 = load <4 x half>, ptr addrspace(3) %lds.const.gep4517, align 8, !dbg !17, !frisk.k_lds_pair !57
  %452 = load <4 x half>, ptr addrspace(3) %lds.const.gep4521, align 8, !dbg !17, !frisk.k_lds_pair !57
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %440, <4 x float> %450) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %435, <4 x float> %375) #9, !dbg !17
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %437, <4 x float> %454) #9, !dbg !17
  %456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %439, <4 x float> %377) #9, !dbg !17
  %457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %441, <4 x float> %456) #9, !dbg !17
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %442, <4 x float> %379) #9, !dbg !17
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %445, <4 x float> %458) #9, !dbg !17
  %460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %448, <4 x float> %381) #9, !dbg !17
  %461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %451, <4 x float> %460) #9, !dbg !17
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %443, <4 x float> %383) #9, !dbg !17
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %446, <4 x float> %462) #9, !dbg !17
  %464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %449, <4 x float> %385) #9, !dbg !17
  %465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %452, <4 x float> %464) #9, !dbg !17
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %434, <4 x float> %387) #9, !dbg !17
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %436, <4 x float> %466) #9, !dbg !17
  %468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %438, <4 x float> %389) #9, !dbg !17
  %469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %440, <4 x float> %468) #9, !dbg !17
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %435, <4 x float> %391) #9, !dbg !17
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %437, <4 x float> %470) #9, !dbg !17
  %472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %439, <4 x float> %393) #9, !dbg !17
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %441, <4 x float> %472) #9, !dbg !17
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %442, <4 x float> %395) #9, !dbg !17
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %445, <4 x float> %474) #9, !dbg !17
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %448, <4 x float> %397) #9, !dbg !17
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %451, <4 x float> %476) #9, !dbg !17
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %443, <4 x float> %399) #9, !dbg !17
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %446, <4 x float> %478) #9, !dbg !17
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %449, <4 x float> %401) #9, !dbg !17
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %452, <4 x float> %480) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %482 = add nuw nsw i64 %136, %28, !dbg !22
  %483 = shl i64 %482, 19, !dbg !22
  %484 = and i64 %483, 16252928, !dbg !22
  %address.local3254 = shl nuw nsw i64 %address.local1200, 3, !dbg !22
  %address.local3255 = and i64 %address.local3254, 1536, !dbg !22
  %address.local3256 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local3255, !dbg !22
  %address.local3257 = ptrtoint ptr addrspace(3) %address.local3256 to i64, !dbg !22
  %address.local3258 = trunc i64 %address.local3257 to i32, !dbg !22
  %485 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3258), !dbg !22
  %address.local3259 = shl nuw nsw i64 %address.local1200, 7, !dbg !22
  %address.local3260 = and i64 %address.local3259, 16256, !dbg !22
  %address.local3262 = lshr i64 %address.local1200, 4, !dbg !22
  %address.local3263 = and i64 %address.local3262, 8, !dbg !22
  %486 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !22
  %.idx48304832 = add i64 %484, %140, !dbg !22
  %487 = or disjoint i64 %address.local3263, %.idx48304832, !dbg !22
  %488 = or disjoint i64 %487, %address.local3260, !dbg !22
  %.tr4834 = trunc i64 %488 to i32, !dbg !22
  %dma.buffer.offset4644 = shl i32 %.tr4834, 1, !dbg !22
  %489 = insertelement <2 x i64> poison, i64 %486, i64 0, !dbg !22
  %490 = bitcast <2 x i64> %489 to <4 x i32>, !dbg !22
  %dma.buffer.rsrc4645 = shufflevector <4 x i32> %490, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !22
  %491 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %485, i32 %dma.buffer.offset4644, <4 x i32> %dma.buffer.rsrc4645) #10, !dbg !22
  %address.local3268 = getelementptr i8, ptr addrspace(3) %address.local3256, i64 4096, !dbg !22
  %address.local3269 = ptrtoint ptr addrspace(3) %address.local3268 to i64, !dbg !22
  %address.local3270 = trunc i64 %address.local3269 to i32, !dbg !22
  %492 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3270), !dbg !22
  %493 = or disjoint i64 %.idx48304832, %address.local3260, !dbg !22
  %494 = or disjoint i64 %493, %address.local3263, !dbg !22
  %.tr4838 = trunc i64 %494 to i32, !dbg !22
  %495 = shl i32 %.tr4838, 1, !dbg !22
  %dma.buffer.offset4642 = or i32 %495, 32, !dbg !22
  %496 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %492, i32 %dma.buffer.offset4642, <4 x i32> %dma.buffer.rsrc4645) #10, !dbg !22
  %497 = load <4 x half>, ptr addrspace(3) %lds.const.gep4536, align 8, !dbg !17, !frisk.k_tail_pair !35
  %498 = load <4 x half>, ptr addrspace(3) %lds.const.gep4540, align 8, !dbg !17, !frisk.k_tail_pair !35
  %499 = load <4 x half>, ptr addrspace(3) %lds.const.gep4537, align 8, !dbg !17, !frisk.k_tail_pair !36
  %500 = load <4 x half>, ptr addrspace(3) %lds.const.gep4541, align 8, !dbg !17, !frisk.k_tail_pair !36
  %501 = load <4 x half>, ptr addrspace(3) %lds.const.gep4538, align 8, !dbg !17, !frisk.k_tail_pair !37
  %502 = load <4 x half>, ptr addrspace(3) %lds.const.gep4542, align 8, !dbg !17, !frisk.k_tail_pair !37
  %503 = load <4 x half>, ptr addrspace(3) %lds.const.gep4539, align 8, !dbg !17, !frisk.k_tail_pair !38
  %504 = load <4 x half>, ptr addrspace(3) %lds.const.gep4543, align 8, !dbg !17, !frisk.k_tail_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %505 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %497, <4 x float> %447) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18, !frisk.k_tail_prefetch !1
  %506 = or disjoint i64 %137, %122, !dbg !18
  %507 = trunc i64 %506 to i32, !dbg !18
  %.not247 = icmp ugt i32 %124, %507, !dbg !18
  %508 = or disjoint i32 %507, 1, !dbg !18
  %.not248 = icmp ugt i32 %124, %508, !dbg !18
  %509 = or disjoint i32 %507, 2, !dbg !18
  %.not249 = icmp ugt i32 %124, %509, !dbg !18
  %510 = or disjoint i32 %507, 3, !dbg !18
  %.not250 = icmp ugt i32 %124, %510, !dbg !18
  %511 = or disjoint i32 %507, 4, !dbg !18
  %.not251 = icmp ugt i32 %124, %511, !dbg !18
  %512 = or disjoint i32 %507, 5, !dbg !18
  %.not252 = icmp ugt i32 %124, %512, !dbg !18
  %513 = or disjoint i32 %507, 6, !dbg !18
  %.not253 = icmp ugt i32 %124, %513, !dbg !18
  %514 = or disjoint i32 %507, 7, !dbg !18
  %.not254 = icmp ugt i32 %124, %514, !dbg !18
  %515 = or disjoint i32 %507, 8, !dbg !18
  %.not255 = icmp ugt i32 %124, %515, !dbg !18
  %516 = or disjoint i32 %507, 9, !dbg !18
  %.not256 = icmp ugt i32 %124, %516, !dbg !18
  %517 = or disjoint i32 %507, 10, !dbg !18
  %.not257 = icmp ugt i32 %124, %517, !dbg !18
  %518 = or disjoint i32 %507, 11, !dbg !18
  %.not258 = icmp ugt i32 %124, %518, !dbg !18
  %519 = or disjoint i32 %507, 12, !dbg !18
  %.not259 = icmp ugt i32 %124, %519, !dbg !18
  %520 = or disjoint i32 %507, 13, !dbg !18
  %.not260 = icmp ugt i32 %124, %520, !dbg !18
  %521 = or disjoint i32 %507, 14, !dbg !18
  %.not261 = icmp ugt i32 %124, %521, !dbg !18
  %522 = or disjoint i32 %507, 15, !dbg !18
  %.not262 = icmp ugt i32 %124, %522, !dbg !18
  %523 = or disjoint i32 %507, 64, !dbg !18
  %.not263 = icmp ugt i32 %124, %523, !dbg !18
  %524 = or disjoint i32 %507, 65, !dbg !18
  %.not264 = icmp ugt i32 %124, %524, !dbg !18
  %525 = or disjoint i32 %507, 66, !dbg !18
  %.not265 = icmp ugt i32 %124, %525, !dbg !18
  %526 = or disjoint i32 %507, 67, !dbg !18
  %.not266 = icmp ugt i32 %124, %526, !dbg !18
  %527 = or disjoint i32 %507, 68, !dbg !18
  %.not267 = icmp ugt i32 %124, %527, !dbg !18
  %528 = or disjoint i32 %507, 69, !dbg !18
  %.not268 = icmp ugt i32 %124, %528, !dbg !18
  %529 = or disjoint i32 %507, 70, !dbg !18
  %.not269 = icmp ugt i32 %124, %529, !dbg !18
  %530 = or disjoint i32 %507, 71, !dbg !18
  %.not270 = icmp ugt i32 %124, %530, !dbg !18
  %531 = or disjoint i32 %507, 72, !dbg !18
  %.not271 = icmp ugt i32 %124, %531, !dbg !18
  %532 = or disjoint i32 %507, 73, !dbg !18
  %.not272 = icmp ugt i32 %124, %532, !dbg !18
  %533 = or disjoint i32 %507, 74, !dbg !18
  %.not273 = icmp ugt i32 %124, %533, !dbg !18
  %534 = or disjoint i32 %507, 75, !dbg !18
  %.not274 = icmp ugt i32 %124, %534, !dbg !18
  %535 = or disjoint i32 %507, 76, !dbg !18
  %.not275 = icmp ugt i32 %124, %535, !dbg !18
  %536 = or disjoint i32 %507, 77, !dbg !18
  %.not276 = icmp ugt i32 %124, %536, !dbg !18
  %537 = or disjoint i32 %507, 78, !dbg !18
  %.not277 = icmp ugt i32 %124, %537, !dbg !18
  %538 = or disjoint i32 %507, 79, !dbg !18
  %.not278 = icmp ugt i32 %124, %538, !dbg !18
  %.not279 = icmp ugt i32 %125, %507, !dbg !18
  %.not280 = icmp ugt i32 %125, %508, !dbg !18
  %.not281 = icmp ugt i32 %125, %509, !dbg !18
  %.not282 = icmp ugt i32 %125, %510, !dbg !18
  %.not283 = icmp ugt i32 %125, %511, !dbg !18
  %.not284 = icmp ugt i32 %125, %512, !dbg !18
  %.not285 = icmp ugt i32 %125, %513, !dbg !18
  %.not286 = icmp ugt i32 %125, %514, !dbg !18
  %.not287 = icmp ugt i32 %125, %515, !dbg !18
  %.not288 = icmp ugt i32 %125, %516, !dbg !18
  %.not289 = icmp ugt i32 %125, %517, !dbg !18
  %.not290 = icmp ugt i32 %125, %518, !dbg !18
  %.not291 = icmp ugt i32 %125, %519, !dbg !18
  %.not292 = icmp ugt i32 %125, %520, !dbg !18
  %.not293 = icmp ugt i32 %125, %521, !dbg !18
  %.not294 = icmp ugt i32 %125, %522, !dbg !18
  %.not295 = icmp ugt i32 %125, %523, !dbg !18
  %.not296 = icmp ugt i32 %125, %524, !dbg !18
  %.not297 = icmp ugt i32 %125, %525, !dbg !18
  %.not298 = icmp ugt i32 %125, %526, !dbg !18
  %.not299 = icmp ugt i32 %125, %527, !dbg !18
  %.not300 = icmp ugt i32 %125, %528, !dbg !18
  %.not301 = icmp ugt i32 %125, %529, !dbg !18
  %.not302 = icmp ugt i32 %125, %530, !dbg !18
  %.not303 = icmp ugt i32 %125, %531, !dbg !18
  %.not304 = icmp ugt i32 %125, %532, !dbg !18
  %.not305 = icmp ugt i32 %125, %533, !dbg !18
  %.not306 = icmp ugt i32 %125, %534, !dbg !18
  %.not307 = icmp ugt i32 %125, %535, !dbg !18
  %.not308 = icmp ugt i32 %125, %536, !dbg !18
  %.not309 = icmp ugt i32 %125, %537, !dbg !18
  %.not310 = icmp ugt i32 %125, %538, !dbg !18
  %539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %499, <4 x float> %505) #9, !dbg !17
  %540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %501, <4 x float> %453) #9, !dbg !17
  %541 = extractelement <4 x float> %539, i64 0, !dbg !17
  %542 = select i1 %.not247, float %541, float 0xFFF0000000000000, !dbg !18
  %543 = extractelement <4 x float> %539, i64 1, !dbg !17
  %544 = select i1 %.not248, float %543, float 0xFFF0000000000000, !dbg !18
  %545 = call float @llvm.maxnum.f32(float %542, float 0xFFF0000000000000), !dbg !8
  %546 = extractelement <4 x float> %539, i64 2, !dbg !17
  %547 = select i1 %.not249, float %546, float 0xFFF0000000000000, !dbg !18
  %548 = extractelement <4 x float> %539, i64 3, !dbg !17
  %549 = select i1 %.not250, float %548, float 0xFFF0000000000000, !dbg !18
  %550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %503, <4 x float> %540) #9, !dbg !17
  %551 = load <4 x half>, ptr addrspace(3) %lds.const.gep4544, align 8, !dbg !17, !frisk.k_tail_pair !34
  %552 = load <4 x half>, ptr addrspace(3) %lds.const.gep4548, align 8, !dbg !17, !frisk.k_tail_pair !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %498, <4 x float> %455) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %554 = extractelement <4 x float> %550, i64 0, !dbg !17
  %555 = select i1 %.not251, float %554, float 0xFFF0000000000000, !dbg !18
  %556 = extractelement <4 x float> %550, i64 1, !dbg !17
  %557 = select i1 %.not252, float %556, float 0xFFF0000000000000, !dbg !18
  %558 = call float @llvm.maxnum.f32(float %545, float %555), !dbg !8
  %559 = extractelement <4 x float> %550, i64 2, !dbg !17
  %560 = select i1 %.not253, float %559, float 0xFFF0000000000000, !dbg !18
  %561 = call float @llvm.maxnum.f32(float %544, float %557), !dbg !8
  %562 = extractelement <4 x float> %550, i64 3, !dbg !17
  %563 = select i1 %.not254, float %562, float 0xFFF0000000000000, !dbg !18
  %564 = call float @llvm.maxnum.f32(float %547, float %560), !dbg !8
  %565 = call float @llvm.maxnum.f32(float %549, float %563), !dbg !8
  %566 = load <4 x half>, ptr addrspace(3) %lds.const.gep4545, align 8, !dbg !17, !frisk.k_tail_pair !39
  %567 = load <4 x half>, ptr addrspace(3) %lds.const.gep4549, align 8, !dbg !17, !frisk.k_tail_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %500, <4 x float> %553) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %569 = load <4 x half>, ptr addrspace(3) %lds.const.gep4546, align 8, !dbg !17, !frisk.k_tail_pair !40
  %570 = load <4 x half>, ptr addrspace(3) %lds.const.gep4550, align 8, !dbg !17, !frisk.k_tail_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %502, <4 x float> %457) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %572 = extractelement <4 x float> %568, i64 0, !dbg !17
  %573 = select i1 %.not255, float %572, float 0xFFF0000000000000, !dbg !18
  %574 = extractelement <4 x float> %568, i64 1, !dbg !17
  %575 = select i1 %.not256, float %574, float 0xFFF0000000000000, !dbg !18
  %576 = call float @llvm.maxnum.f32(float %558, float %573), !dbg !8
  %577 = extractelement <4 x float> %568, i64 2, !dbg !17
  %578 = select i1 %.not257, float %577, float 0xFFF0000000000000, !dbg !18
  %579 = call float @llvm.maxnum.f32(float %561, float %575), !dbg !8
  %580 = extractelement <4 x float> %568, i64 3, !dbg !17
  %581 = select i1 %.not258, float %580, float 0xFFF0000000000000, !dbg !18
  %582 = call float @llvm.maxnum.f32(float %564, float %578), !dbg !8
  %583 = call float @llvm.maxnum.f32(float %565, float %581), !dbg !8
  %584 = load <4 x half>, ptr addrspace(3) %lds.const.gep4547, align 8, !dbg !17, !frisk.k_tail_pair !41
  %585 = load <4 x half>, ptr addrspace(3) %lds.const.gep4551, align 8, !dbg !17, !frisk.k_tail_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %504, <4 x float> %571) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %587 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %551, <4 x float> %459) #9, !dbg !17
  %588 = extractelement <4 x float> %586, i64 0, !dbg !17
  %589 = select i1 %.not259, float %588, float 0xFFF0000000000000, !dbg !18
  %590 = extractelement <4 x float> %586, i64 1, !dbg !17
  %591 = select i1 %.not260, float %590, float 0xFFF0000000000000, !dbg !18
  %592 = call float @llvm.maxnum.f32(float %576, float %589), !dbg !8
  %593 = extractelement <4 x float> %586, i64 2, !dbg !17
  %594 = select i1 %.not261, float %593, float 0xFFF0000000000000, !dbg !18
  %595 = call float @llvm.maxnum.f32(float %579, float %591), !dbg !8
  %596 = extractelement <4 x float> %586, i64 3, !dbg !17
  %597 = select i1 %.not262, float %596, float 0xFFF0000000000000, !dbg !18
  %598 = call float @llvm.maxnum.f32(float %582, float %594), !dbg !8
  %599 = call float @llvm.maxnum.f32(float %583, float %597), !dbg !8
  %600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %566, <4 x float> %587) #9, !dbg !17
  %601 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %569, <4 x float> %461) #9, !dbg !17
  %602 = extractelement <4 x float> %600, i64 0, !dbg !17
  %603 = select i1 %.not263, float %602, float 0xFFF0000000000000, !dbg !18
  %604 = extractelement <4 x float> %600, i64 1, !dbg !17
  %605 = select i1 %.not264, float %604, float 0xFFF0000000000000, !dbg !18
  %606 = call float @llvm.maxnum.f32(float %592, float %603), !dbg !8
  %607 = extractelement <4 x float> %600, i64 2, !dbg !17
  %608 = select i1 %.not265, float %607, float 0xFFF0000000000000, !dbg !18
  %609 = call float @llvm.maxnum.f32(float %595, float %605), !dbg !8
  %610 = extractelement <4 x float> %600, i64 3, !dbg !17
  %611 = select i1 %.not266, float %610, float 0xFFF0000000000000, !dbg !18
  %612 = call float @llvm.maxnum.f32(float %598, float %608), !dbg !8
  %613 = call float @llvm.maxnum.f32(float %599, float %611), !dbg !8
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %584, <4 x float> %601) #9, !dbg !17
  %615 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %552, <4 x float> %463) #9, !dbg !17
  %616 = extractelement <4 x float> %614, i64 0, !dbg !17
  %617 = select i1 %.not267, float %616, float 0xFFF0000000000000, !dbg !18
  %618 = extractelement <4 x float> %614, i64 1, !dbg !17
  %619 = select i1 %.not268, float %618, float 0xFFF0000000000000, !dbg !18
  %620 = call float @llvm.maxnum.f32(float %606, float %617), !dbg !8
  %621 = extractelement <4 x float> %614, i64 2, !dbg !17
  %622 = select i1 %.not269, float %621, float 0xFFF0000000000000, !dbg !18
  %623 = call float @llvm.maxnum.f32(float %609, float %619), !dbg !8
  %624 = extractelement <4 x float> %614, i64 3, !dbg !17
  %625 = select i1 %.not270, float %624, float 0xFFF0000000000000, !dbg !18
  %626 = call float @llvm.maxnum.f32(float %612, float %622), !dbg !8
  %627 = call float @llvm.maxnum.f32(float %613, float %625), !dbg !8
  %628 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %567, <4 x float> %615) #9, !dbg !17
  %629 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %570, <4 x float> %465) #9, !dbg !17
  %630 = extractelement <4 x float> %628, i64 0, !dbg !17
  %631 = select i1 %.not271, float %630, float 0xFFF0000000000000, !dbg !18
  %632 = extractelement <4 x float> %628, i64 1, !dbg !17
  %633 = select i1 %.not272, float %632, float 0xFFF0000000000000, !dbg !18
  %634 = call float @llvm.maxnum.f32(float %620, float %631), !dbg !8
  %635 = extractelement <4 x float> %628, i64 2, !dbg !17
  %636 = select i1 %.not273, float %635, float 0xFFF0000000000000, !dbg !18
  %637 = call float @llvm.maxnum.f32(float %623, float %633), !dbg !8
  %638 = extractelement <4 x float> %628, i64 3, !dbg !17
  %639 = select i1 %.not274, float %638, float 0xFFF0000000000000, !dbg !18
  %640 = call float @llvm.maxnum.f32(float %626, float %636), !dbg !8
  %641 = call float @llvm.maxnum.f32(float %627, float %639), !dbg !8
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %585, <4 x float> %629) #9, !dbg !17
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %497, <4 x float> %467) #9, !dbg !17
  %644 = extractelement <4 x float> %642, i64 0, !dbg !17
  %645 = select i1 %.not275, float %644, float 0xFFF0000000000000, !dbg !18
  %646 = extractelement <4 x float> %642, i64 1, !dbg !17
  %647 = select i1 %.not276, float %646, float 0xFFF0000000000000, !dbg !18
  %648 = call float @llvm.maxnum.f32(float %634, float %645), !dbg !8
  %649 = extractelement <4 x float> %642, i64 2, !dbg !17
  %650 = select i1 %.not277, float %649, float 0xFFF0000000000000, !dbg !18
  %651 = call float @llvm.maxnum.f32(float %637, float %647), !dbg !8
  %652 = extractelement <4 x float> %642, i64 3, !dbg !17
  %653 = select i1 %.not278, float %652, float 0xFFF0000000000000, !dbg !18
  %654 = call float @llvm.maxnum.f32(float %640, float %650), !dbg !8
  %655 = call float @llvm.maxnum.f32(float %641, float %653), !dbg !8
  %656 = call float @llvm.maxnum.f32(float %648, float %651), !dbg !8
  %657 = call float @llvm.maxnum.f32(float %654, float %655), !dbg !8
  %658 = call float @llvm.maxnum.f32(float %656, float %657), !dbg !8
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %499, <4 x float> %643) #9, !dbg !17
  %660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %501, <4 x float> %469) #9, !dbg !17
  %661 = extractelement <4 x float> %659, i64 0, !dbg !17
  %662 = select i1 %.not279, float %661, float 0xFFF0000000000000, !dbg !18
  %663 = extractelement <4 x float> %659, i64 1, !dbg !17
  %664 = select i1 %.not280, float %663, float 0xFFF0000000000000, !dbg !18
  %665 = extractelement <4 x float> %659, i64 2, !dbg !17
  %666 = select i1 %.not281, float %665, float 0xFFF0000000000000, !dbg !18
  %667 = extractelement <4 x float> %659, i64 3, !dbg !17
  %668 = select i1 %.not282, float %667, float 0xFFF0000000000000, !dbg !18
  %669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %503, <4 x float> %660) #9, !dbg !17
  %670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %498, <4 x float> %471) #9, !dbg !17
  %671 = extractelement <4 x float> %669, i64 0, !dbg !17
  %672 = select i1 %.not283, float %671, float 0xFFF0000000000000, !dbg !18
  %673 = extractelement <4 x float> %669, i64 1, !dbg !17
  %674 = select i1 %.not284, float %673, float 0xFFF0000000000000, !dbg !18
  %675 = extractelement <4 x float> %669, i64 2, !dbg !17
  %676 = select i1 %.not285, float %675, float 0xFFF0000000000000, !dbg !18
  %677 = extractelement <4 x float> %669, i64 3, !dbg !17
  %678 = select i1 %.not286, float %677, float 0xFFF0000000000000, !dbg !18
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %500, <4 x float> %670) #9, !dbg !17
  %680 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %502, <4 x float> %473) #9, !dbg !17
  %681 = extractelement <4 x float> %679, i64 0, !dbg !17
  %682 = select i1 %.not287, float %681, float 0xFFF0000000000000, !dbg !18
  %683 = extractelement <4 x float> %679, i64 1, !dbg !17
  %684 = select i1 %.not288, float %683, float 0xFFF0000000000000, !dbg !18
  %685 = extractelement <4 x float> %679, i64 2, !dbg !17
  %686 = select i1 %.not289, float %685, float 0xFFF0000000000000, !dbg !18
  %687 = extractelement <4 x float> %679, i64 3, !dbg !17
  %688 = select i1 %.not290, float %687, float 0xFFF0000000000000, !dbg !18
  %689 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %504, <4 x float> %680) #9, !dbg !17
  %690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %551, <4 x float> %475) #9, !dbg !17
  %691 = extractelement <4 x float> %689, i64 0, !dbg !17
  %692 = select i1 %.not291, float %691, float 0xFFF0000000000000, !dbg !18
  %693 = extractelement <4 x float> %689, i64 1, !dbg !17
  %694 = select i1 %.not292, float %693, float 0xFFF0000000000000, !dbg !18
  %695 = extractelement <4 x float> %689, i64 2, !dbg !17
  %696 = select i1 %.not293, float %695, float 0xFFF0000000000000, !dbg !18
  %697 = extractelement <4 x float> %689, i64 3, !dbg !17
  %698 = select i1 %.not294, float %697, float 0xFFF0000000000000, !dbg !18
  %699 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %566, <4 x float> %690) #9, !dbg !17
  %700 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %569, <4 x float> %477) #9, !dbg !17
  %701 = extractelement <4 x float> %699, i64 0, !dbg !17
  %702 = select i1 %.not295, float %701, float 0xFFF0000000000000, !dbg !18
  %703 = extractelement <4 x float> %699, i64 1, !dbg !17
  %704 = select i1 %.not296, float %703, float 0xFFF0000000000000, !dbg !18
  %705 = extractelement <4 x float> %699, i64 2, !dbg !17
  %706 = select i1 %.not297, float %705, float 0xFFF0000000000000, !dbg !18
  %707 = extractelement <4 x float> %699, i64 3, !dbg !17
  %708 = select i1 %.not298, float %707, float 0xFFF0000000000000, !dbg !18
  %709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %584, <4 x float> %700) #9, !dbg !17
  %710 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %552, <4 x float> %479) #9, !dbg !17
  %711 = extractelement <4 x float> %709, i64 0, !dbg !17
  %712 = select i1 %.not299, float %711, float 0xFFF0000000000000, !dbg !18
  %713 = extractelement <4 x float> %709, i64 1, !dbg !17
  %714 = select i1 %.not300, float %713, float 0xFFF0000000000000, !dbg !18
  %715 = extractelement <4 x float> %709, i64 2, !dbg !17
  %716 = select i1 %.not301, float %715, float 0xFFF0000000000000, !dbg !18
  %717 = extractelement <4 x float> %709, i64 3, !dbg !17
  %718 = select i1 %.not302, float %717, float 0xFFF0000000000000, !dbg !18
  %719 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %567, <4 x float> %710) #9, !dbg !17
  %720 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %570, <4 x float> %481) #9, !dbg !17
  %721 = extractelement <4 x float> %719, i64 0, !dbg !17
  %722 = select i1 %.not303, float %721, float 0xFFF0000000000000, !dbg !18
  %723 = extractelement <4 x float> %719, i64 1, !dbg !17
  %724 = select i1 %.not304, float %723, float 0xFFF0000000000000, !dbg !18
  %725 = extractelement <4 x float> %719, i64 2, !dbg !17
  %726 = select i1 %.not305, float %725, float 0xFFF0000000000000, !dbg !18
  %727 = extractelement <4 x float> %719, i64 3, !dbg !17
  %728 = select i1 %.not306, float %727, float 0xFFF0000000000000, !dbg !18
  %729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %585, <4 x float> %720) #9, !dbg !17
  %730 = extractelement <4 x float> %729, i64 0, !dbg !17
  %731 = select i1 %.not307, float %730, float 0xFFF0000000000000, !dbg !18
  %732 = extractelement <4 x float> %729, i64 1, !dbg !17
  %733 = select i1 %.not308, float %732, float 0xFFF0000000000000, !dbg !18
  %734 = extractelement <4 x float> %729, i64 2, !dbg !17
  %735 = select i1 %.not309, float %734, float 0xFFF0000000000000, !dbg !18
  %736 = extractelement <4 x float> %729, i64 3, !dbg !17
  %737 = select i1 %.not310, float %736, float 0xFFF0000000000000, !dbg !18
  %738 = bitcast float %658 to i32, !dbg !8
  %739 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %738), !dbg !8
  %740 = bitcast i32 %739 to float, !dbg !8
  %741 = call float @llvm.maxnum.f32(float %658, float %740), !dbg !8
  %742 = bitcast float %741 to i32, !dbg !8
  %743 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %742), !dbg !8
  %744 = bitcast i32 %743 to float, !dbg !8
  %745 = call float @llvm.maxnum.f32(float %741, float %744), !dbg !8
  %746 = call float @llvm.maxnum.f32(float %662, float 0xFFF0000000000000), !dbg !8
  %747 = call float @llvm.maxnum.f32(float %746, float %672), !dbg !8
  %748 = call float @llvm.maxnum.f32(float %664, float %674), !dbg !8
  %749 = call float @llvm.maxnum.f32(float %666, float %676), !dbg !8
  %750 = call float @llvm.maxnum.f32(float %668, float %678), !dbg !8
  %751 = call float @llvm.maxnum.f32(float %747, float %682), !dbg !8
  %752 = call float @llvm.maxnum.f32(float %748, float %684), !dbg !8
  %753 = call float @llvm.maxnum.f32(float %749, float %686), !dbg !8
  %754 = call float @llvm.maxnum.f32(float %750, float %688), !dbg !8
  %755 = call float @llvm.maxnum.f32(float %751, float %692), !dbg !8
  %756 = call float @llvm.maxnum.f32(float %752, float %694), !dbg !8
  %757 = call float @llvm.maxnum.f32(float %753, float %696), !dbg !8
  %758 = call float @llvm.maxnum.f32(float %754, float %698), !dbg !8
  %759 = call float @llvm.maxnum.f32(float %755, float %702), !dbg !8
  %760 = call float @llvm.maxnum.f32(float %756, float %704), !dbg !8
  %761 = call float @llvm.maxnum.f32(float %757, float %706), !dbg !8
  %762 = call float @llvm.maxnum.f32(float %758, float %708), !dbg !8
  %763 = call float @llvm.maxnum.f32(float %759, float %712), !dbg !8
  %764 = call float @llvm.maxnum.f32(float %760, float %714), !dbg !8
  %765 = call float @llvm.maxnum.f32(float %761, float %716), !dbg !8
  %766 = call float @llvm.maxnum.f32(float %762, float %718), !dbg !8
  %767 = call float @llvm.maxnum.f32(float %763, float %722), !dbg !8
  %768 = call float @llvm.maxnum.f32(float %764, float %724), !dbg !8
  %769 = call float @llvm.maxnum.f32(float %765, float %726), !dbg !8
  %770 = call float @llvm.maxnum.f32(float %766, float %728), !dbg !8
  %771 = call float @llvm.maxnum.f32(float %767, float %731), !dbg !8
  %772 = call float @llvm.maxnum.f32(float %768, float %733), !dbg !8
  %773 = call float @llvm.maxnum.f32(float %769, float %735), !dbg !8
  %774 = call float @llvm.maxnum.f32(float %770, float %737), !dbg !8
  %775 = call float @llvm.maxnum.f32(float %771, float %772), !dbg !8
  %776 = call float @llvm.maxnum.f32(float %773, float %774), !dbg !8
  %777 = call float @llvm.maxnum.f32(float %775, float %776), !dbg !8
  %778 = bitcast float %777 to i32, !dbg !8
  %779 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %778), !dbg !8
  %780 = bitcast i32 %779 to float, !dbg !8
  %781 = call float @llvm.maxnum.f32(float %777, float %780), !dbg !8
  %782 = bitcast float %781 to i32, !dbg !8
  %783 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %782), !dbg !8
  %784 = bitcast i32 %783 to float, !dbg !8
  %785 = call float @llvm.maxnum.f32(float %781, float %784), !dbg !8
  %786 = extractvalue [2 x <1 x float>] %130, 0, !dbg !8
  %787 = insertelement <1 x float> poison, float %745, i64 0, !dbg !8
  %788 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %787, <1 x float> %786), !dbg !8
  %789 = extractvalue [2 x <1 x float>] %130, 1, !dbg !8
  %790 = insertelement <1 x float> poison, float %785, i64 0, !dbg !8
  %791 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %790, <1 x float> %789), !dbg !8
  %792 = insertelement <4 x float> poison, float %542, i64 0, !dbg !23
  %793 = insertelement <4 x float> %792, float %544, i64 1, !dbg !23
  %794 = insertelement <4 x float> %793, float %547, i64 2, !dbg !23
  %795 = insertelement <4 x float> %794, float %549, i64 3, !dbg !23
  %796 = shufflevector <1 x float> %788, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %797 = insertelement <4 x float> poison, float %555, i64 0, !dbg !23
  %798 = insertelement <4 x float> %797, float %557, i64 1, !dbg !23
  %799 = insertelement <4 x float> %798, float %560, i64 2, !dbg !23
  %800 = insertelement <4 x float> %799, float %563, i64 3, !dbg !23
  %801 = insertelement <4 x float> poison, float %573, i64 0, !dbg !23
  %802 = insertelement <4 x float> %801, float %575, i64 1, !dbg !23
  %803 = insertelement <4 x float> %802, float %578, i64 2, !dbg !23
  %804 = insertelement <4 x float> %803, float %581, i64 3, !dbg !23
  %805 = insertelement <4 x float> poison, float %589, i64 0, !dbg !23
  %806 = insertelement <4 x float> %805, float %591, i64 1, !dbg !23
  %807 = insertelement <4 x float> %806, float %594, i64 2, !dbg !23
  %808 = insertelement <4 x float> %807, float %597, i64 3, !dbg !23
  %809 = insertelement <4 x float> poison, float %603, i64 0, !dbg !23
  %810 = insertelement <4 x float> %809, float %605, i64 1, !dbg !23
  %811 = insertelement <4 x float> %810, float %608, i64 2, !dbg !23
  %812 = insertelement <4 x float> %811, float %611, i64 3, !dbg !23
  %813 = insertelement <4 x float> poison, float %617, i64 0, !dbg !23
  %814 = insertelement <4 x float> %813, float %619, i64 1, !dbg !23
  %815 = insertelement <4 x float> %814, float %622, i64 2, !dbg !23
  %816 = insertelement <4 x float> %815, float %625, i64 3, !dbg !23
  %817 = insertelement <4 x float> poison, float %631, i64 0, !dbg !23
  %818 = insertelement <4 x float> %817, float %633, i64 1, !dbg !23
  %819 = insertelement <4 x float> %818, float %636, i64 2, !dbg !23
  %820 = insertelement <4 x float> %819, float %639, i64 3, !dbg !23
  %821 = insertelement <4 x float> poison, float %645, i64 0, !dbg !23
  %822 = insertelement <4 x float> %821, float %647, i64 1, !dbg !23
  %823 = insertelement <4 x float> %822, float %650, i64 2, !dbg !23
  %824 = insertelement <4 x float> %823, float %653, i64 3, !dbg !23
  %825 = insertelement <4 x float> poison, float %662, i64 0, !dbg !23
  %826 = insertelement <4 x float> %825, float %664, i64 1, !dbg !23
  %827 = insertelement <4 x float> %826, float %666, i64 2, !dbg !23
  %828 = insertelement <4 x float> %827, float %668, i64 3, !dbg !23
  %829 = shufflevector <1 x float> %791, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %830 = insertelement <4 x float> poison, float %672, i64 0, !dbg !23
  %831 = insertelement <4 x float> %830, float %674, i64 1, !dbg !23
  %832 = insertelement <4 x float> %831, float %676, i64 2, !dbg !23
  %833 = insertelement <4 x float> %832, float %678, i64 3, !dbg !23
  %834 = insertelement <4 x float> poison, float %682, i64 0, !dbg !23
  %835 = insertelement <4 x float> %834, float %684, i64 1, !dbg !23
  %836 = insertelement <4 x float> %835, float %686, i64 2, !dbg !23
  %837 = insertelement <4 x float> %836, float %688, i64 3, !dbg !23
  %838 = insertelement <4 x float> poison, float %692, i64 0, !dbg !23
  %839 = insertelement <4 x float> %838, float %694, i64 1, !dbg !23
  %840 = insertelement <4 x float> %839, float %696, i64 2, !dbg !23
  %841 = insertelement <4 x float> %840, float %698, i64 3, !dbg !23
  %842 = insertelement <4 x float> poison, float %702, i64 0, !dbg !23
  %843 = insertelement <4 x float> %842, float %704, i64 1, !dbg !23
  %844 = insertelement <4 x float> %843, float %706, i64 2, !dbg !23
  %845 = insertelement <4 x float> %844, float %708, i64 3, !dbg !23
  %846 = insertelement <4 x float> poison, float %712, i64 0, !dbg !23
  %847 = insertelement <4 x float> %846, float %714, i64 1, !dbg !23
  %848 = insertelement <4 x float> %847, float %716, i64 2, !dbg !23
  %849 = insertelement <4 x float> %848, float %718, i64 3, !dbg !23
  %850 = insertelement <4 x float> poison, float %722, i64 0, !dbg !23
  %851 = insertelement <4 x float> %850, float %724, i64 1, !dbg !23
  %852 = insertelement <4 x float> %851, float %726, i64 2, !dbg !23
  %853 = insertelement <4 x float> %852, float %728, i64 3, !dbg !23
  %854 = insertelement <4 x float> poison, float %731, i64 0, !dbg !23
  %855 = insertelement <4 x float> %854, float %733, i64 1, !dbg !23
  %856 = insertelement <4 x float> %855, float %735, i64 2, !dbg !23
  %857 = insertelement <4 x float> %856, float %737, i64 3, !dbg !23
  %858 = fsub <1 x float> %786, %788, !dbg !24
  %859 = extractelement <1 x float> %858, i64 0, !dbg !25
  %860 = call float @llvm.amdgcn.exp2.f32(float %859), !dbg !25
  %861 = insertelement <4 x float> poison, float %860, i64 0, !dbg !30
  %862 = shufflevector <4 x float> %861, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %863 = extractvalue [2 x <1 x float>] %129, 0, !dbg !26
  %864 = extractelement <1 x float> %863, i64 0, !dbg !26
  %865 = fmul float %864, %860, !dbg !26
  %866 = fsub <1 x float> %789, %791, !dbg !24
  %867 = extractelement <1 x float> %866, i64 0, !dbg !25
  %868 = call float @llvm.amdgcn.exp2.f32(float %867), !dbg !25
  %869 = insertelement <4 x float> poison, float %868, i64 0, !dbg !30
  %870 = shufflevector <4 x float> %869, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %871 = extractvalue [2 x <1 x float>] %129, 1, !dbg !26
  %872 = extractelement <1 x float> %871, i64 0, !dbg !26
  %873 = fmul float %872, %868, !dbg !26
  %874 = extractvalue [2 x <8 x float>] %128, 0, !dbg !30
  %875 = shufflevector <8 x float> %874, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %876 = fmul <4 x float> %875, %862, !dbg !30
  %877 = shufflevector <8 x float> %874, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %878 = fmul <4 x float> %877, %862, !dbg !30
  %879 = extractvalue [2 x <8 x float>] %128, 1, !dbg !30
  %880 = shufflevector <8 x float> %879, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %881 = fmul <4 x float> %880, %870, !dbg !30
  %882 = shufflevector <8 x float> %879, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %883 = fmul <4 x float> %882, %870, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %address.local3929 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local3255, !dbg !22
  %address.local3930 = ptrtoint ptr addrspace(3) %address.local3929 to i64, !dbg !22
  %address.local3931 = trunc i64 %address.local3930 to i32, !dbg !22
  %884 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3931), !dbg !22
  %885 = or disjoint i64 %.idx48304832, 32, !dbg !22
  %886 = or disjoint i64 %address.local3263, %885, !dbg !22
  %887 = or disjoint i64 %886, %address.local3260, !dbg !22
  %.tr4841 = trunc i64 %887 to i32, !dbg !22
  %dma.buffer.offset4640 = shl i32 %.tr4841, 1, !dbg !22
  %888 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %884, i32 %dma.buffer.offset4640, <4 x i32> %dma.buffer.rsrc4645) #10, !dbg !22
  %address.local3941 = getelementptr i8, ptr addrspace(3) %address.local3929, i64 4096, !dbg !22
  %address.local3942 = ptrtoint ptr addrspace(3) %address.local3941 to i64, !dbg !22
  %address.local3943 = trunc i64 %address.local3942 to i32, !dbg !22
  %889 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local3943), !dbg !22
  %dma.buffer.offset4638 = or i32 %495, 96, !dbg !22
  %890 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %889, i32 %dma.buffer.offset4638, <4 x i32> %dma.buffer.rsrc4645) #10, !dbg !22
  %address.local3950 = shl nuw nsw i32 %address.thread1199, 3, !dbg !14
  %address.local3951 = and i32 %address.local3950, 384, !dbg !14
  %address.local3953 = shl nuw nsw i32 %address.thread1199, 10, !dbg !14
  %address.local3954 = and i32 %address.local3953, 3072, !dbg !14
  %891 = or disjoint i32 %address.local3951, %address.local3954, !dbg !14
  %address.local3955 = or disjoint i32 %891, %address.local1314, !dbg !14
  %address.local3956 = zext nneg i32 %address.local3955 to i64, !dbg !14
  %address.local3957 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local3956, !dbg !14
  %892 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3957, i32 0) #0, !dbg !14
  %893 = bitcast <4 x i32> %892 to <8 x half>, !dbg !14
  %address.local3968 = getelementptr i8, ptr addrspace(3) %address.local3957, i64 64, !dbg !14
  %894 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3968, i32 0) #0, !dbg !14
  %895 = bitcast <4 x i32> %894 to <8 x half>, !dbg !14
  %address.local3979 = getelementptr i8, ptr addrspace(3) %address.local3957, i64 128, !dbg !14
  %896 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3979, i32 0) #0, !dbg !14
  %897 = bitcast <4 x i32> %896 to <8 x half>, !dbg !14
  %address.local3990 = getelementptr i8, ptr addrspace(3) %address.local3957, i64 192, !dbg !14
  %898 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3990, i32 0) #0, !dbg !14
  %899 = bitcast <4 x i32> %898 to <8 x half>, !dbg !14
  %address.local4001 = getelementptr i8, ptr addrspace(3) %address.local3957, i64 1024, !dbg !14
  %900 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4001, i32 0) #0, !dbg !14
  %901 = bitcast <4 x i32> %900 to <8 x half>, !dbg !14
  %address.local4012 = getelementptr i8, ptr addrspace(3) %address.local3957, i64 1088, !dbg !14
  %902 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4012, i32 0) #0, !dbg !14
  %903 = bitcast <4 x i32> %902 to <8 x half>, !dbg !14
  %address.local4023 = getelementptr i8, ptr addrspace(3) %address.local3957, i64 1152, !dbg !14
  %904 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4023, i32 0) #0, !dbg !14
  %905 = bitcast <4 x i32> %904 to <8 x half>, !dbg !14
  %address.local4034 = getelementptr i8, ptr addrspace(3) %address.local3957, i64 1216, !dbg !14
  %906 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4034, i32 0) #0, !dbg !14
  %907 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %908 = fsub <4 x float> %795, %796, !dbg !23
  %909 = extractelement <4 x float> %908, i64 0, !dbg !27
  %910 = call float @llvm.amdgcn.exp2.f32(float %909), !dbg !27
  %911 = insertelement <4 x float> poison, float %910, i64 0, !dbg !29
  %912 = extractelement <4 x float> %908, i64 1, !dbg !27
  %913 = call float @llvm.amdgcn.exp2.f32(float %912), !dbg !27
  %914 = insertelement <4 x float> %911, float %913, i64 1, !dbg !29
  %915 = extractelement <4 x float> %908, i64 2, !dbg !27
  %916 = call float @llvm.amdgcn.exp2.f32(float %915), !dbg !27
  %917 = insertelement <4 x float> %914, float %916, i64 2, !dbg !29
  %918 = extractelement <4 x float> %908, i64 3, !dbg !27
  %919 = call float @llvm.amdgcn.exp2.f32(float %918), !dbg !27
  %920 = insertelement <4 x float> %917, float %919, i64 3, !dbg !29
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %920) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i32 0, !dbg !29
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i32 1, !dbg !29
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i32 2, !dbg !29
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i32 3, !dbg !29
  %.cvtpack.p0 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1), !dbg !29
  %.cvtpack.p1 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3), !dbg !29
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i32 0, !dbg !29
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i32 1, !dbg !29
  %921 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %921) #9, !dbg !27, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %907, <4 x float> %876) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %923 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %923, <4 x float> %878) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %925 = fsub <4 x float> %828, %829, !dbg !23
  %926 = extractelement <4 x float> %925, i64 0, !dbg !27
  %927 = call float @llvm.amdgcn.exp2.f32(float %926), !dbg !27
  %928 = insertelement <4 x float> poison, float %927, i64 0, !dbg !29
  %929 = extractelement <4 x float> %925, i64 1, !dbg !27
  %930 = call float @llvm.amdgcn.exp2.f32(float %929), !dbg !27
  %931 = insertelement <4 x float> %928, float %930, i64 1, !dbg !29
  %932 = extractelement <4 x float> %925, i64 2, !dbg !27
  %933 = call float @llvm.amdgcn.exp2.f32(float %932), !dbg !27
  %934 = insertelement <4 x float> %931, float %933, i64 2, !dbg !29
  %935 = extractelement <4 x float> %925, i64 3, !dbg !27
  %936 = call float @llvm.amdgcn.exp2.f32(float %935), !dbg !27
  %937 = insertelement <4 x float> %934, float %936, i64 3, !dbg !29
  %math.result925 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %937) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04857 = extractelement <4 x float> %math.result925, i32 0, !dbg !29
  %.cvtpack.f14858 = extractelement <4 x float> %math.result925, i32 1, !dbg !29
  %.cvtpack.f24859 = extractelement <4 x float> %math.result925, i32 2, !dbg !29
  %.cvtpack.f34860 = extractelement <4 x float> %math.result925, i32 3, !dbg !29
  %.cvtpack.p04861 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04857, float %.cvtpack.f14858), !dbg !29
  %.cvtpack.p14862 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24859, float %.cvtpack.f34860), !dbg !29
  %.cvtpack.v04863 = insertelement <2 x i32> poison, i32 %.cvtpack.p04861, i32 0, !dbg !29
  %.cvtpack.v14864 = insertelement <2 x i32> %.cvtpack.v04863, i32 %.cvtpack.p14862, i32 1, !dbg !29
  %938 = bitcast <2 x i32> %.cvtpack.v14864 to <4 x half>, !dbg !29
  %math.packed926 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %938) #9, !dbg !27, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %907, <4 x float> %881) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %923, <4 x float> %883) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %941 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %942 = fsub <4 x float> %800, %796, !dbg !23
  %943 = extractelement <4 x float> %942, i64 0, !dbg !27
  %944 = call float @llvm.amdgcn.exp2.f32(float %943), !dbg !27
  %945 = fadd float %910, %944, !dbg !28
  %946 = insertelement <4 x float> poison, float %944, i64 0, !dbg !29
  %947 = extractelement <4 x float> %942, i64 1, !dbg !27
  %948 = call float @llvm.amdgcn.exp2.f32(float %947), !dbg !27
  %949 = fadd float %913, %948, !dbg !28
  %950 = insertelement <4 x float> %946, float %948, i64 1, !dbg !29
  %951 = extractelement <4 x float> %942, i64 2, !dbg !27
  %952 = call float @llvm.amdgcn.exp2.f32(float %951), !dbg !27
  %953 = fadd float %916, %952, !dbg !28
  %954 = insertelement <4 x float> %950, float %952, i64 2, !dbg !29
  %955 = extractelement <4 x float> %942, i64 3, !dbg !27
  %956 = call float @llvm.amdgcn.exp2.f32(float %955), !dbg !27
  %957 = fadd float %919, %956, !dbg !28
  %958 = insertelement <4 x float> %954, float %956, i64 3, !dbg !29
  %math.result911 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %958) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04865 = extractelement <4 x float> %math.result911, i32 0, !dbg !29
  %.cvtpack.f14866 = extractelement <4 x float> %math.result911, i32 1, !dbg !29
  %.cvtpack.f24867 = extractelement <4 x float> %math.result911, i32 2, !dbg !29
  %.cvtpack.f34868 = extractelement <4 x float> %math.result911, i32 3, !dbg !29
  %.cvtpack.p04869 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04865, float %.cvtpack.f14866), !dbg !29
  %.cvtpack.p14870 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24867, float %.cvtpack.f34868), !dbg !29
  %.cvtpack.v04871 = insertelement <2 x i32> poison, i32 %.cvtpack.p04869, i32 0, !dbg !29
  %.cvtpack.v14872 = insertelement <2 x i32> %.cvtpack.v04871, i32 %.cvtpack.p14870, i32 1, !dbg !29
  %959 = bitcast <2 x i32> %.cvtpack.v14872 to <4 x half>, !dbg !29
  %math.packed912 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %959) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %941, <4 x float> %922) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %961 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %961, <4 x float> %924) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %963 = fsub <4 x float> %833, %829, !dbg !23
  %964 = extractelement <4 x float> %963, i64 0, !dbg !27
  %965 = call float @llvm.amdgcn.exp2.f32(float %964), !dbg !27
  %966 = fadd float %927, %965, !dbg !28
  %967 = insertelement <4 x float> poison, float %965, i64 0, !dbg !29
  %968 = extractelement <4 x float> %963, i64 1, !dbg !27
  %969 = call float @llvm.amdgcn.exp2.f32(float %968), !dbg !27
  %970 = fadd float %930, %969, !dbg !28
  %971 = insertelement <4 x float> %967, float %969, i64 1, !dbg !29
  %972 = extractelement <4 x float> %963, i64 2, !dbg !27
  %973 = call float @llvm.amdgcn.exp2.f32(float %972), !dbg !27
  %974 = fadd float %933, %973, !dbg !28
  %975 = insertelement <4 x float> %971, float %973, i64 2, !dbg !29
  %976 = extractelement <4 x float> %963, i64 3, !dbg !27
  %977 = call float @llvm.amdgcn.exp2.f32(float %976), !dbg !27
  %978 = fadd float %936, %977, !dbg !28
  %979 = insertelement <4 x float> %975, float %977, i64 3, !dbg !29
  %math.result927 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %979) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04873 = extractelement <4 x float> %math.result927, i32 0, !dbg !29
  %.cvtpack.f14874 = extractelement <4 x float> %math.result927, i32 1, !dbg !29
  %.cvtpack.f24875 = extractelement <4 x float> %math.result927, i32 2, !dbg !29
  %.cvtpack.f34876 = extractelement <4 x float> %math.result927, i32 3, !dbg !29
  %.cvtpack.p04877 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04873, float %.cvtpack.f14874), !dbg !29
  %.cvtpack.p14878 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24875, float %.cvtpack.f34876), !dbg !29
  %.cvtpack.v04879 = insertelement <2 x i32> poison, i32 %.cvtpack.p04877, i32 0, !dbg !29
  %.cvtpack.v14880 = insertelement <2 x i32> %.cvtpack.v04879, i32 %.cvtpack.p14878, i32 1, !dbg !29
  %980 = bitcast <2 x i32> %.cvtpack.v14880 to <4 x half>, !dbg !29
  %math.packed928 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %980) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %941, <4 x float> %939) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %961, <4 x float> %940) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %983 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %984 = fsub <4 x float> %804, %796, !dbg !23
  %985 = extractelement <4 x float> %984, i64 0, !dbg !27
  %986 = call float @llvm.amdgcn.exp2.f32(float %985), !dbg !27
  %987 = fadd float %945, %986, !dbg !28
  %988 = insertelement <4 x float> poison, float %986, i64 0, !dbg !29
  %989 = extractelement <4 x float> %984, i64 1, !dbg !27
  %990 = call float @llvm.amdgcn.exp2.f32(float %989), !dbg !27
  %991 = fadd float %949, %990, !dbg !28
  %992 = insertelement <4 x float> %988, float %990, i64 1, !dbg !29
  %993 = extractelement <4 x float> %984, i64 2, !dbg !27
  %994 = call float @llvm.amdgcn.exp2.f32(float %993), !dbg !27
  %995 = fadd float %953, %994, !dbg !28
  %996 = insertelement <4 x float> %992, float %994, i64 2, !dbg !29
  %997 = extractelement <4 x float> %984, i64 3, !dbg !27
  %998 = call float @llvm.amdgcn.exp2.f32(float %997), !dbg !27
  %999 = fadd float %957, %998, !dbg !28
  %1000 = insertelement <4 x float> %996, float %998, i64 3, !dbg !29
  %math.result913 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1000) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04881 = extractelement <4 x float> %math.result913, i32 0, !dbg !29
  %.cvtpack.f14882 = extractelement <4 x float> %math.result913, i32 1, !dbg !29
  %.cvtpack.f24883 = extractelement <4 x float> %math.result913, i32 2, !dbg !29
  %.cvtpack.f34884 = extractelement <4 x float> %math.result913, i32 3, !dbg !29
  %.cvtpack.p04885 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04881, float %.cvtpack.f14882), !dbg !29
  %.cvtpack.p14886 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24883, float %.cvtpack.f34884), !dbg !29
  %.cvtpack.v04887 = insertelement <2 x i32> poison, i32 %.cvtpack.p04885, i32 0, !dbg !29
  %.cvtpack.v14888 = insertelement <2 x i32> %.cvtpack.v04887, i32 %.cvtpack.p14886, i32 1, !dbg !29
  %1001 = bitcast <2 x i32> %.cvtpack.v14888 to <4 x half>, !dbg !29
  %math.packed914 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1001) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %983, <4 x float> %960) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1003 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %1003, <4 x float> %962) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1005 = fsub <4 x float> %837, %829, !dbg !23
  %1006 = extractelement <4 x float> %1005, i64 0, !dbg !27
  %1007 = call float @llvm.amdgcn.exp2.f32(float %1006), !dbg !27
  %1008 = fadd float %966, %1007, !dbg !28
  %1009 = insertelement <4 x float> poison, float %1007, i64 0, !dbg !29
  %1010 = extractelement <4 x float> %1005, i64 1, !dbg !27
  %1011 = call float @llvm.amdgcn.exp2.f32(float %1010), !dbg !27
  %1012 = fadd float %970, %1011, !dbg !28
  %1013 = insertelement <4 x float> %1009, float %1011, i64 1, !dbg !29
  %1014 = extractelement <4 x float> %1005, i64 2, !dbg !27
  %1015 = call float @llvm.amdgcn.exp2.f32(float %1014), !dbg !27
  %1016 = fadd float %974, %1015, !dbg !28
  %1017 = insertelement <4 x float> %1013, float %1015, i64 2, !dbg !29
  %1018 = extractelement <4 x float> %1005, i64 3, !dbg !27
  %1019 = call float @llvm.amdgcn.exp2.f32(float %1018), !dbg !27
  %1020 = fadd float %978, %1019, !dbg !28
  %1021 = insertelement <4 x float> %1017, float %1019, i64 3, !dbg !29
  %math.result929 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1021) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04889 = extractelement <4 x float> %math.result929, i32 0, !dbg !29
  %.cvtpack.f14890 = extractelement <4 x float> %math.result929, i32 1, !dbg !29
  %.cvtpack.f24891 = extractelement <4 x float> %math.result929, i32 2, !dbg !29
  %.cvtpack.f34892 = extractelement <4 x float> %math.result929, i32 3, !dbg !29
  %.cvtpack.p04893 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04889, float %.cvtpack.f14890), !dbg !29
  %.cvtpack.p14894 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24891, float %.cvtpack.f34892), !dbg !29
  %.cvtpack.v04895 = insertelement <2 x i32> poison, i32 %.cvtpack.p04893, i32 0, !dbg !29
  %.cvtpack.v14896 = insertelement <2 x i32> %.cvtpack.v04895, i32 %.cvtpack.p14894, i32 1, !dbg !29
  %1022 = bitcast <2 x i32> %.cvtpack.v14896 to <4 x half>, !dbg !29
  %math.packed930 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1022) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %983, <4 x float> %981) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %1003, <4 x float> %982) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1025 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1026 = fsub <4 x float> %808, %796, !dbg !23
  %1027 = extractelement <4 x float> %1026, i64 0, !dbg !27
  %1028 = call float @llvm.amdgcn.exp2.f32(float %1027), !dbg !27
  %1029 = fadd float %987, %1028, !dbg !28
  %1030 = insertelement <4 x float> poison, float %1028, i64 0, !dbg !29
  %1031 = extractelement <4 x float> %1026, i64 1, !dbg !27
  %1032 = call float @llvm.amdgcn.exp2.f32(float %1031), !dbg !27
  %1033 = fadd float %991, %1032, !dbg !28
  %1034 = insertelement <4 x float> %1030, float %1032, i64 1, !dbg !29
  %1035 = extractelement <4 x float> %1026, i64 2, !dbg !27
  %1036 = call float @llvm.amdgcn.exp2.f32(float %1035), !dbg !27
  %1037 = fadd float %995, %1036, !dbg !28
  %1038 = insertelement <4 x float> %1034, float %1036, i64 2, !dbg !29
  %1039 = extractelement <4 x float> %1026, i64 3, !dbg !27
  %1040 = call float @llvm.amdgcn.exp2.f32(float %1039), !dbg !27
  %1041 = fadd float %999, %1040, !dbg !28
  %1042 = insertelement <4 x float> %1038, float %1040, i64 3, !dbg !29
  %math.result915 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1042) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04897 = extractelement <4 x float> %math.result915, i32 0, !dbg !29
  %.cvtpack.f14898 = extractelement <4 x float> %math.result915, i32 1, !dbg !29
  %.cvtpack.f24899 = extractelement <4 x float> %math.result915, i32 2, !dbg !29
  %.cvtpack.f34900 = extractelement <4 x float> %math.result915, i32 3, !dbg !29
  %.cvtpack.p04901 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04897, float %.cvtpack.f14898), !dbg !29
  %.cvtpack.p14902 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24899, float %.cvtpack.f34900), !dbg !29
  %.cvtpack.v04903 = insertelement <2 x i32> poison, i32 %.cvtpack.p04901, i32 0, !dbg !29
  %.cvtpack.v14904 = insertelement <2 x i32> %.cvtpack.v04903, i32 %.cvtpack.p14902, i32 1, !dbg !29
  %1043 = bitcast <2 x i32> %.cvtpack.v14904 to <4 x half>, !dbg !29
  %math.packed916 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1043) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1025, <4 x float> %1002) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1045 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1045, <4 x float> %1004) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1047 = fsub <4 x float> %841, %829, !dbg !23
  %1048 = extractelement <4 x float> %1047, i64 0, !dbg !27
  %1049 = call float @llvm.amdgcn.exp2.f32(float %1048), !dbg !27
  %1050 = fadd float %1008, %1049, !dbg !28
  %1051 = insertelement <4 x float> poison, float %1049, i64 0, !dbg !29
  %1052 = extractelement <4 x float> %1047, i64 1, !dbg !27
  %1053 = call float @llvm.amdgcn.exp2.f32(float %1052), !dbg !27
  %1054 = fadd float %1012, %1053, !dbg !28
  %1055 = insertelement <4 x float> %1051, float %1053, i64 1, !dbg !29
  %1056 = extractelement <4 x float> %1047, i64 2, !dbg !27
  %1057 = call float @llvm.amdgcn.exp2.f32(float %1056), !dbg !27
  %1058 = fadd float %1016, %1057, !dbg !28
  %1059 = insertelement <4 x float> %1055, float %1057, i64 2, !dbg !29
  %1060 = extractelement <4 x float> %1047, i64 3, !dbg !27
  %1061 = call float @llvm.amdgcn.exp2.f32(float %1060), !dbg !27
  %1062 = fadd float %1020, %1061, !dbg !28
  %1063 = insertelement <4 x float> %1059, float %1061, i64 3, !dbg !29
  %math.result931 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1063) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04905 = extractelement <4 x float> %math.result931, i32 0, !dbg !29
  %.cvtpack.f14906 = extractelement <4 x float> %math.result931, i32 1, !dbg !29
  %.cvtpack.f24907 = extractelement <4 x float> %math.result931, i32 2, !dbg !29
  %.cvtpack.f34908 = extractelement <4 x float> %math.result931, i32 3, !dbg !29
  %.cvtpack.p04909 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04905, float %.cvtpack.f14906), !dbg !29
  %.cvtpack.p14910 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24907, float %.cvtpack.f34908), !dbg !29
  %.cvtpack.v04911 = insertelement <2 x i32> poison, i32 %.cvtpack.p04909, i32 0, !dbg !29
  %.cvtpack.v14912 = insertelement <2 x i32> %.cvtpack.v04911, i32 %.cvtpack.p14910, i32 1, !dbg !29
  %1064 = bitcast <2 x i32> %.cvtpack.v14912 to <4 x half>, !dbg !29
  %math.packed932 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1064) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1025, <4 x float> %1023) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1045, <4 x float> %1024) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1067 = shufflevector <8 x half> %901, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1068 = fsub <4 x float> %812, %796, !dbg !23
  %1069 = extractelement <4 x float> %1068, i64 0, !dbg !27
  %1070 = call float @llvm.amdgcn.exp2.f32(float %1069), !dbg !27
  %1071 = fadd float %1029, %1070, !dbg !28
  %1072 = insertelement <4 x float> poison, float %1070, i64 0, !dbg !29
  %1073 = extractelement <4 x float> %1068, i64 1, !dbg !27
  %1074 = call float @llvm.amdgcn.exp2.f32(float %1073), !dbg !27
  %1075 = fadd float %1033, %1074, !dbg !28
  %1076 = insertelement <4 x float> %1072, float %1074, i64 1, !dbg !29
  %1077 = extractelement <4 x float> %1068, i64 2, !dbg !27
  %1078 = call float @llvm.amdgcn.exp2.f32(float %1077), !dbg !27
  %1079 = fadd float %1037, %1078, !dbg !28
  %1080 = insertelement <4 x float> %1076, float %1078, i64 2, !dbg !29
  %1081 = extractelement <4 x float> %1068, i64 3, !dbg !27
  %1082 = call float @llvm.amdgcn.exp2.f32(float %1081), !dbg !27
  %1083 = fadd float %1041, %1082, !dbg !28
  %1084 = insertelement <4 x float> %1080, float %1082, i64 3, !dbg !29
  %math.result917 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1084) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04913 = extractelement <4 x float> %math.result917, i32 0, !dbg !29
  %.cvtpack.f14914 = extractelement <4 x float> %math.result917, i32 1, !dbg !29
  %.cvtpack.f24915 = extractelement <4 x float> %math.result917, i32 2, !dbg !29
  %.cvtpack.f34916 = extractelement <4 x float> %math.result917, i32 3, !dbg !29
  %.cvtpack.p04917 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04913, float %.cvtpack.f14914), !dbg !29
  %.cvtpack.p14918 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24915, float %.cvtpack.f34916), !dbg !29
  %.cvtpack.v04919 = insertelement <2 x i32> poison, i32 %.cvtpack.p04917, i32 0, !dbg !29
  %.cvtpack.v14920 = insertelement <2 x i32> %.cvtpack.v04919, i32 %.cvtpack.p14918, i32 1, !dbg !29
  %1085 = bitcast <2 x i32> %.cvtpack.v14920 to <4 x half>, !dbg !29
  %math.packed918 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1085) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1067, <4 x float> %1044) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1087 = shufflevector <8 x half> %901, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1087, <4 x float> %1046) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1089 = fsub <4 x float> %845, %829, !dbg !23
  %1090 = extractelement <4 x float> %1089, i64 0, !dbg !27
  %1091 = call float @llvm.amdgcn.exp2.f32(float %1090), !dbg !27
  %1092 = fadd float %1050, %1091, !dbg !28
  %1093 = insertelement <4 x float> poison, float %1091, i64 0, !dbg !29
  %1094 = extractelement <4 x float> %1089, i64 1, !dbg !27
  %1095 = call float @llvm.amdgcn.exp2.f32(float %1094), !dbg !27
  %1096 = fadd float %1054, %1095, !dbg !28
  %1097 = insertelement <4 x float> %1093, float %1095, i64 1, !dbg !29
  %1098 = extractelement <4 x float> %1089, i64 2, !dbg !27
  %1099 = call float @llvm.amdgcn.exp2.f32(float %1098), !dbg !27
  %1100 = fadd float %1058, %1099, !dbg !28
  %1101 = insertelement <4 x float> %1097, float %1099, i64 2, !dbg !29
  %1102 = extractelement <4 x float> %1089, i64 3, !dbg !27
  %1103 = call float @llvm.amdgcn.exp2.f32(float %1102), !dbg !27
  %1104 = fadd float %1062, %1103, !dbg !28
  %1105 = insertelement <4 x float> %1101, float %1103, i64 3, !dbg !29
  %math.result933 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1105) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04921 = extractelement <4 x float> %math.result933, i32 0, !dbg !29
  %.cvtpack.f14922 = extractelement <4 x float> %math.result933, i32 1, !dbg !29
  %.cvtpack.f24923 = extractelement <4 x float> %math.result933, i32 2, !dbg !29
  %.cvtpack.f34924 = extractelement <4 x float> %math.result933, i32 3, !dbg !29
  %.cvtpack.p04925 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04921, float %.cvtpack.f14922), !dbg !29
  %.cvtpack.p14926 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24923, float %.cvtpack.f34924), !dbg !29
  %.cvtpack.v04927 = insertelement <2 x i32> poison, i32 %.cvtpack.p04925, i32 0, !dbg !29
  %.cvtpack.v14928 = insertelement <2 x i32> %.cvtpack.v04927, i32 %.cvtpack.p14926, i32 1, !dbg !29
  %1106 = bitcast <2 x i32> %.cvtpack.v14928 to <4 x half>, !dbg !29
  %math.packed934 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1106) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1107 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1067, <4 x float> %1065) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1087, <4 x float> %1066) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1109 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1110 = fsub <4 x float> %816, %796, !dbg !23
  %1111 = extractelement <4 x float> %1110, i64 0, !dbg !27
  %1112 = call float @llvm.amdgcn.exp2.f32(float %1111), !dbg !27
  %1113 = fadd float %1071, %1112, !dbg !28
  %1114 = insertelement <4 x float> poison, float %1112, i64 0, !dbg !29
  %1115 = extractelement <4 x float> %1110, i64 1, !dbg !27
  %1116 = call float @llvm.amdgcn.exp2.f32(float %1115), !dbg !27
  %1117 = fadd float %1075, %1116, !dbg !28
  %1118 = insertelement <4 x float> %1114, float %1116, i64 1, !dbg !29
  %1119 = extractelement <4 x float> %1110, i64 2, !dbg !27
  %1120 = call float @llvm.amdgcn.exp2.f32(float %1119), !dbg !27
  %1121 = fadd float %1079, %1120, !dbg !28
  %1122 = insertelement <4 x float> %1118, float %1120, i64 2, !dbg !29
  %1123 = extractelement <4 x float> %1110, i64 3, !dbg !27
  %1124 = call float @llvm.amdgcn.exp2.f32(float %1123), !dbg !27
  %1125 = fadd float %1083, %1124, !dbg !28
  %1126 = insertelement <4 x float> %1122, float %1124, i64 3, !dbg !29
  %math.result919 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1126) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04929 = extractelement <4 x float> %math.result919, i32 0, !dbg !29
  %.cvtpack.f14930 = extractelement <4 x float> %math.result919, i32 1, !dbg !29
  %.cvtpack.f24931 = extractelement <4 x float> %math.result919, i32 2, !dbg !29
  %.cvtpack.f34932 = extractelement <4 x float> %math.result919, i32 3, !dbg !29
  %.cvtpack.p04933 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04929, float %.cvtpack.f14930), !dbg !29
  %.cvtpack.p14934 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24931, float %.cvtpack.f34932), !dbg !29
  %.cvtpack.v04935 = insertelement <2 x i32> poison, i32 %.cvtpack.p04933, i32 0, !dbg !29
  %.cvtpack.v14936 = insertelement <2 x i32> %.cvtpack.v04935, i32 %.cvtpack.p14934, i32 1, !dbg !29
  %1127 = bitcast <2 x i32> %.cvtpack.v14936 to <4 x half>, !dbg !29
  %math.packed920 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1127) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1109, <4 x float> %1086) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1129 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1129, <4 x float> %1088) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1131 = fsub <4 x float> %849, %829, !dbg !23
  %1132 = extractelement <4 x float> %1131, i64 0, !dbg !27
  %1133 = call float @llvm.amdgcn.exp2.f32(float %1132), !dbg !27
  %1134 = fadd float %1092, %1133, !dbg !28
  %1135 = insertelement <4 x float> poison, float %1133, i64 0, !dbg !29
  %1136 = extractelement <4 x float> %1131, i64 1, !dbg !27
  %1137 = call float @llvm.amdgcn.exp2.f32(float %1136), !dbg !27
  %1138 = fadd float %1096, %1137, !dbg !28
  %1139 = insertelement <4 x float> %1135, float %1137, i64 1, !dbg !29
  %1140 = extractelement <4 x float> %1131, i64 2, !dbg !27
  %1141 = call float @llvm.amdgcn.exp2.f32(float %1140), !dbg !27
  %1142 = fadd float %1100, %1141, !dbg !28
  %1143 = insertelement <4 x float> %1139, float %1141, i64 2, !dbg !29
  %1144 = extractelement <4 x float> %1131, i64 3, !dbg !27
  %1145 = call float @llvm.amdgcn.exp2.f32(float %1144), !dbg !27
  %1146 = fadd float %1104, %1145, !dbg !28
  %1147 = insertelement <4 x float> %1143, float %1145, i64 3, !dbg !29
  %math.result935 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1147) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04937 = extractelement <4 x float> %math.result935, i32 0, !dbg !29
  %.cvtpack.f14938 = extractelement <4 x float> %math.result935, i32 1, !dbg !29
  %.cvtpack.f24939 = extractelement <4 x float> %math.result935, i32 2, !dbg !29
  %.cvtpack.f34940 = extractelement <4 x float> %math.result935, i32 3, !dbg !29
  %.cvtpack.p04941 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04937, float %.cvtpack.f14938), !dbg !29
  %.cvtpack.p14942 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24939, float %.cvtpack.f34940), !dbg !29
  %.cvtpack.v04943 = insertelement <2 x i32> poison, i32 %.cvtpack.p04941, i32 0, !dbg !29
  %.cvtpack.v14944 = insertelement <2 x i32> %.cvtpack.v04943, i32 %.cvtpack.p14942, i32 1, !dbg !29
  %1148 = bitcast <2 x i32> %.cvtpack.v14944 to <4 x half>, !dbg !29
  %math.packed936 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1148) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1109, <4 x float> %1107) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1129, <4 x float> %1108) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1151 = shufflevector <8 x half> %905, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1152 = fsub <4 x float> %820, %796, !dbg !23
  %1153 = extractelement <4 x float> %1152, i64 0, !dbg !27
  %1154 = call float @llvm.amdgcn.exp2.f32(float %1153), !dbg !27
  %1155 = fadd float %1113, %1154, !dbg !28
  %1156 = insertelement <4 x float> poison, float %1154, i64 0, !dbg !29
  %1157 = extractelement <4 x float> %1152, i64 1, !dbg !27
  %1158 = call float @llvm.amdgcn.exp2.f32(float %1157), !dbg !27
  %1159 = fadd float %1117, %1158, !dbg !28
  %1160 = insertelement <4 x float> %1156, float %1158, i64 1, !dbg !29
  %1161 = extractelement <4 x float> %1152, i64 2, !dbg !27
  %1162 = call float @llvm.amdgcn.exp2.f32(float %1161), !dbg !27
  %1163 = fadd float %1121, %1162, !dbg !28
  %1164 = insertelement <4 x float> %1160, float %1162, i64 2, !dbg !29
  %1165 = extractelement <4 x float> %1152, i64 3, !dbg !27
  %1166 = call float @llvm.amdgcn.exp2.f32(float %1165), !dbg !27
  %1167 = fadd float %1125, %1166, !dbg !28
  %1168 = insertelement <4 x float> %1164, float %1166, i64 3, !dbg !29
  %math.result921 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1168) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04945 = extractelement <4 x float> %math.result921, i32 0, !dbg !29
  %.cvtpack.f14946 = extractelement <4 x float> %math.result921, i32 1, !dbg !29
  %.cvtpack.f24947 = extractelement <4 x float> %math.result921, i32 2, !dbg !29
  %.cvtpack.f34948 = extractelement <4 x float> %math.result921, i32 3, !dbg !29
  %.cvtpack.p04949 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04945, float %.cvtpack.f14946), !dbg !29
  %.cvtpack.p14950 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24947, float %.cvtpack.f34948), !dbg !29
  %.cvtpack.v04951 = insertelement <2 x i32> poison, i32 %.cvtpack.p04949, i32 0, !dbg !29
  %.cvtpack.v14952 = insertelement <2 x i32> %.cvtpack.v04951, i32 %.cvtpack.p14950, i32 1, !dbg !29
  %1169 = bitcast <2 x i32> %.cvtpack.v14952 to <4 x half>, !dbg !29
  %math.packed922 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1169) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1151, <4 x float> %1128) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1171 = shufflevector <8 x half> %905, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1171, <4 x float> %1130) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1173 = fsub <4 x float> %853, %829, !dbg !23
  %1174 = extractelement <4 x float> %1173, i64 0, !dbg !27
  %1175 = call float @llvm.amdgcn.exp2.f32(float %1174), !dbg !27
  %1176 = fadd float %1134, %1175, !dbg !28
  %1177 = insertelement <4 x float> poison, float %1175, i64 0, !dbg !29
  %1178 = extractelement <4 x float> %1173, i64 1, !dbg !27
  %1179 = call float @llvm.amdgcn.exp2.f32(float %1178), !dbg !27
  %1180 = fadd float %1138, %1179, !dbg !28
  %1181 = insertelement <4 x float> %1177, float %1179, i64 1, !dbg !29
  %1182 = extractelement <4 x float> %1173, i64 2, !dbg !27
  %1183 = call float @llvm.amdgcn.exp2.f32(float %1182), !dbg !27
  %1184 = fadd float %1142, %1183, !dbg !28
  %1185 = insertelement <4 x float> %1181, float %1183, i64 2, !dbg !29
  %1186 = extractelement <4 x float> %1173, i64 3, !dbg !27
  %1187 = call float @llvm.amdgcn.exp2.f32(float %1186), !dbg !27
  %1188 = fadd float %1146, %1187, !dbg !28
  %1189 = insertelement <4 x float> %1185, float %1187, i64 3, !dbg !29
  %math.result937 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1189) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04953 = extractelement <4 x float> %math.result937, i32 0, !dbg !29
  %.cvtpack.f14954 = extractelement <4 x float> %math.result937, i32 1, !dbg !29
  %.cvtpack.f24955 = extractelement <4 x float> %math.result937, i32 2, !dbg !29
  %.cvtpack.f34956 = extractelement <4 x float> %math.result937, i32 3, !dbg !29
  %.cvtpack.p04957 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04953, float %.cvtpack.f14954), !dbg !29
  %.cvtpack.p14958 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24955, float %.cvtpack.f34956), !dbg !29
  %.cvtpack.v04959 = insertelement <2 x i32> poison, i32 %.cvtpack.p04957, i32 0, !dbg !29
  %.cvtpack.v14960 = insertelement <2 x i32> %.cvtpack.v04959, i32 %.cvtpack.p14958, i32 1, !dbg !29
  %1190 = bitcast <2 x i32> %.cvtpack.v14960 to <4 x half>, !dbg !29
  %math.packed938 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1190) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1151, <4 x float> %1149) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1171, <4 x float> %1150) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1193 = bitcast <4 x i32> %906 to <8 x half>, !dbg !14
  %1194 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1195 = fsub <4 x float> %824, %796, !dbg !23
  %1196 = extractelement <4 x float> %1195, i64 0, !dbg !27
  %1197 = call float @llvm.amdgcn.exp2.f32(float %1196), !dbg !27
  %1198 = fadd float %1155, %1197, !dbg !28
  %1199 = insertelement <4 x float> poison, float %1197, i64 0, !dbg !29
  %1200 = extractelement <4 x float> %1195, i64 1, !dbg !27
  %1201 = call float @llvm.amdgcn.exp2.f32(float %1200), !dbg !27
  %1202 = fadd float %1159, %1201, !dbg !28
  %1203 = fadd float %1198, %1202, !dbg !28
  %1204 = insertelement <4 x float> %1199, float %1201, i64 1, !dbg !29
  %1205 = extractelement <4 x float> %1195, i64 2, !dbg !27
  %1206 = call float @llvm.amdgcn.exp2.f32(float %1205), !dbg !27
  %1207 = fadd float %1163, %1206, !dbg !28
  %1208 = insertelement <4 x float> %1204, float %1206, i64 2, !dbg !29
  %1209 = extractelement <4 x float> %1195, i64 3, !dbg !27
  %1210 = call float @llvm.amdgcn.exp2.f32(float %1209), !dbg !27
  %1211 = fadd float %1167, %1210, !dbg !28
  %1212 = fadd float %1207, %1211, !dbg !28
  %1213 = fadd float %1203, %1212, !dbg !28
  %1214 = fadd float %1213, %865, !dbg !28
  %1215 = insertelement <1 x float> poison, float %1214, i64 0, !dbg !6
  %1216 = insertvalue [2 x <1 x float>] poison, <1 x float> %1215, 0, !dbg !6
  %1217 = insertelement <4 x float> %1208, float %1210, i64 3, !dbg !29
  %math.result923 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1217) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04961 = extractelement <4 x float> %math.result923, i32 0, !dbg !29
  %.cvtpack.f14962 = extractelement <4 x float> %math.result923, i32 1, !dbg !29
  %.cvtpack.f24963 = extractelement <4 x float> %math.result923, i32 2, !dbg !29
  %.cvtpack.f34964 = extractelement <4 x float> %math.result923, i32 3, !dbg !29
  %.cvtpack.p04965 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04961, float %.cvtpack.f14962), !dbg !29
  %.cvtpack.p14966 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24963, float %.cvtpack.f34964), !dbg !29
  %.cvtpack.v04967 = insertelement <2 x i32> poison, i32 %.cvtpack.p04965, i32 0, !dbg !29
  %.cvtpack.v14968 = insertelement <2 x i32> %.cvtpack.v04967, i32 %.cvtpack.p14966, i32 1, !dbg !29
  %1218 = bitcast <2 x i32> %.cvtpack.v14968 to <4 x half>, !dbg !29
  %math.packed924 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1218) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1194, <4 x float> %1170) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1220 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1220, <4 x float> %1172) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1222 = fsub <4 x float> %857, %829, !dbg !23
  %1223 = extractelement <4 x float> %1222, i64 0, !dbg !27
  %1224 = call float @llvm.amdgcn.exp2.f32(float %1223), !dbg !27
  %1225 = fadd float %1176, %1224, !dbg !28
  %1226 = insertelement <4 x float> poison, float %1224, i64 0, !dbg !29
  %1227 = extractelement <4 x float> %1222, i64 1, !dbg !27
  %1228 = call float @llvm.amdgcn.exp2.f32(float %1227), !dbg !27
  %1229 = fadd float %1180, %1228, !dbg !28
  %1230 = fadd float %1225, %1229, !dbg !28
  %1231 = insertelement <4 x float> %1226, float %1228, i64 1, !dbg !29
  %1232 = extractelement <4 x float> %1222, i64 2, !dbg !27
  %1233 = call float @llvm.amdgcn.exp2.f32(float %1232), !dbg !27
  %1234 = fadd float %1184, %1233, !dbg !28
  %1235 = insertelement <4 x float> %1231, float %1233, i64 2, !dbg !29
  %1236 = extractelement <4 x float> %1222, i64 3, !dbg !27
  %1237 = call float @llvm.amdgcn.exp2.f32(float %1236), !dbg !27
  %1238 = fadd float %1188, %1237, !dbg !28
  %1239 = fadd float %1234, %1238, !dbg !28
  %1240 = fadd float %1230, %1239, !dbg !28
  %1241 = fadd float %1240, %873, !dbg !28
  %1242 = insertelement <1 x float> poison, float %1241, i64 0, !dbg !6
  %1243 = insertvalue [2 x <1 x float>] %1216, <1 x float> %1242, 1, !dbg !6
  %1244 = insertelement <4 x float> %1235, float %1237, i64 3, !dbg !29
  %math.result939 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1244) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f04969 = extractelement <4 x float> %math.result939, i32 0, !dbg !29
  %.cvtpack.f14970 = extractelement <4 x float> %math.result939, i32 1, !dbg !29
  %.cvtpack.f24971 = extractelement <4 x float> %math.result939, i32 2, !dbg !29
  %.cvtpack.f34972 = extractelement <4 x float> %math.result939, i32 3, !dbg !29
  %.cvtpack.p04973 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f04969, float %.cvtpack.f14970), !dbg !29
  %.cvtpack.p14974 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f24971, float %.cvtpack.f34972), !dbg !29
  %.cvtpack.v04975 = insertelement <2 x i32> poison, i32 %.cvtpack.p04973, i32 0, !dbg !29
  %.cvtpack.v14976 = insertelement <2 x i32> %.cvtpack.v04975, i32 %.cvtpack.p14974, i32 1, !dbg !29
  %1245 = bitcast <2 x i32> %.cvtpack.v14976 to <4 x half>, !dbg !29
  %math.packed940 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1245) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1194, <4 x float> %1191) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1220, <4 x float> %1192) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !30
  %1248 = extractvalue [2 x <8 x float>] %131, 0, !dbg !30
  %1249 = shufflevector <8 x float> %1248, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1250 = fmul <4 x float> %1249, %862, !dbg !30
  %1251 = shufflevector <8 x float> %1248, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1252 = fmul <4 x float> %1251, %862, !dbg !30
  %1253 = extractvalue [2 x <8 x float>] %131, 1, !dbg !30
  %1254 = shufflevector <8 x float> %1253, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1255 = fmul <4 x float> %1254, %870, !dbg !30
  %1256 = shufflevector <8 x float> %1253, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1257 = fmul <4 x float> %1256, %870, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %1258 = or disjoint i64 %.idx48304832, 64, !dbg !22
  %1259 = or disjoint i64 %address.local3263, %1258, !dbg !22
  %1260 = or disjoint i64 %1259, %address.local3260, !dbg !22
  %.tr4847 = trunc i64 %1260 to i32, !dbg !22
  %dma.buffer.offset4636 = shl i32 %.tr4847, 1, !dbg !22
  %1261 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %485, i32 %dma.buffer.offset4636, <4 x i32> %dma.buffer.rsrc4645) #10, !dbg !22
  %dma.buffer.offset4634 = or i32 %495, 160, !dbg !22
  %1262 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %492, i32 %dma.buffer.offset4634, <4 x i32> %dma.buffer.rsrc4645) #10, !dbg !22
  %address.local4067 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local3956, !dbg !14
  %1263 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4067, i32 0) #0, !dbg !14
  %1264 = bitcast <4 x i32> %1263 to <8 x half>, !dbg !14
  %address.local4078 = getelementptr i8, ptr addrspace(3) %address.local4067, i64 64, !dbg !14
  %1265 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4078, i32 0) #0, !dbg !14
  %1266 = bitcast <4 x i32> %1265 to <8 x half>, !dbg !14
  %address.local4089 = getelementptr i8, ptr addrspace(3) %address.local4067, i64 128, !dbg !14
  %1267 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4089, i32 0) #0, !dbg !14
  %1268 = bitcast <4 x i32> %1267 to <8 x half>, !dbg !14
  %address.local4100 = getelementptr i8, ptr addrspace(3) %address.local4067, i64 192, !dbg !14
  %1269 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4100, i32 0) #0, !dbg !14
  %1270 = bitcast <4 x i32> %1269 to <8 x half>, !dbg !14
  %address.local4111 = getelementptr i8, ptr addrspace(3) %address.local4067, i64 1024, !dbg !14
  %1271 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4111, i32 0) #0, !dbg !14
  %1272 = bitcast <4 x i32> %1271 to <8 x half>, !dbg !14
  %address.local4122 = getelementptr i8, ptr addrspace(3) %address.local4067, i64 1088, !dbg !14
  %1273 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4122, i32 0) #0, !dbg !14
  %1274 = bitcast <4 x i32> %1273 to <8 x half>, !dbg !14
  %address.local4133 = getelementptr i8, ptr addrspace(3) %address.local4067, i64 1152, !dbg !14
  %1275 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4133, i32 0) #0, !dbg !14
  %1276 = bitcast <4 x i32> %1275 to <8 x half>, !dbg !14
  %address.local4144 = getelementptr i8, ptr addrspace(3) %address.local4067, i64 1216, !dbg !14
  %1277 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4144, i32 0) #0, !dbg !14
  %1278 = shufflevector <8 x half> %1264, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1278, <4 x float> %1250) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1280 = shufflevector <8 x half> %1264, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1280, <4 x float> %1252) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %1278, <4 x float> %1255) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %1280, <4 x float> %1257) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1284 = shufflevector <8 x half> %1266, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %1284, <4 x float> %1279) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1286 = shufflevector <8 x half> %1266, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %1286, <4 x float> %1281) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %1284, <4 x float> %1282) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %1286, <4 x float> %1283) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1290 = shufflevector <8 x half> %1268, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %1290, <4 x float> %1285) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1292 = shufflevector <8 x half> %1268, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %1292, <4 x float> %1287) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %1290, <4 x float> %1288) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %1292, <4 x float> %1289) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1296 = shufflevector <8 x half> %1270, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1296, <4 x float> %1291) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1298 = shufflevector <8 x half> %1270, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1298, <4 x float> %1293) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1296, <4 x float> %1294) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1298, <4 x float> %1295) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1302 = shufflevector <8 x half> %1272, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1302, <4 x float> %1297) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1304 = shufflevector <8 x half> %1272, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1304, <4 x float> %1299) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1302, <4 x float> %1300) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1304, <4 x float> %1301) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1308 = shufflevector <8 x half> %1274, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1308, <4 x float> %1303) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1310 = shufflevector <8 x half> %1274, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1310, <4 x float> %1305) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1308, <4 x float> %1306) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1310, <4 x float> %1307) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1314 = shufflevector <8 x half> %1276, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1314, <4 x float> %1309) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1316 = shufflevector <8 x half> %1276, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1316, <4 x float> %1311) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1314, <4 x float> %1312) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1316, <4 x float> %1313) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1320 = bitcast <4 x i32> %1277 to <8 x half>, !dbg !14
  %1321 = shufflevector <8 x half> %1320, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1321, <4 x float> %1315) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1323 = shufflevector <8 x half> %1320, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1323, <4 x float> %1317) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1321, <4 x float> %1318) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1323, <4 x float> %1319) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !30
  %1327 = extractvalue [2 x <8 x float>] %132, 0, !dbg !30
  %1328 = shufflevector <8 x float> %1327, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1329 = fmul <4 x float> %1328, %862, !dbg !30
  %1330 = shufflevector <8 x float> %1327, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1331 = fmul <4 x float> %1330, %862, !dbg !30
  %1332 = extractvalue [2 x <8 x float>] %132, 1, !dbg !30
  %1333 = shufflevector <8 x float> %1332, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1334 = fmul <4 x float> %1333, %870, !dbg !30
  %1335 = shufflevector <8 x float> %1332, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1336 = fmul <4 x float> %1335, %870, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %1337 = or disjoint i64 %.idx48304832, 96, !dbg !22
  %1338 = or disjoint i64 %address.local3263, %1337, !dbg !22
  %1339 = or disjoint i64 %1338, %address.local3260, !dbg !22
  %.tr4853 = trunc i64 %1339 to i32, !dbg !22
  %dma.buffer.offset4632 = shl i32 %.tr4853, 1, !dbg !22
  %1340 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %884, i32 %dma.buffer.offset4632, <4 x i32> %dma.buffer.rsrc4645) #10, !dbg !22
  %dma.buffer.offset = or i32 %495, 224, !dbg !22
  %1341 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %889, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc4645) #10, !dbg !22
  %1342 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local3957, i32 0) #0, !dbg !14
  %1343 = bitcast <4 x i32> %1342 to <8 x half>, !dbg !14
  %address.local4186 = or disjoint i32 %address.local3955, 32, !dbg !14
  %address.local4187 = zext nneg i32 %address.local4186 to i64, !dbg !14
  %address.local4188 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4187, !dbg !14
  %1344 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4188, i32 0) #0, !dbg !14
  %1345 = bitcast <4 x i32> %1344 to <8 x half>, !dbg !14
  %address.local4197 = or disjoint i32 %address.local3955, 64, !dbg !14
  %address.local4198 = zext nneg i32 %address.local4197 to i64, !dbg !14
  %address.local4199 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4198, !dbg !14
  %1346 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4199, i32 0) #0, !dbg !14
  %1347 = bitcast <4 x i32> %1346 to <8 x half>, !dbg !14
  %address.local4208 = or disjoint i32 %address.local3955, 96, !dbg !14
  %address.local4209 = zext nneg i32 %address.local4208 to i64, !dbg !14
  %address.local4210 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4209, !dbg !14
  %1348 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4210, i32 0) #0, !dbg !14
  %1349 = bitcast <4 x i32> %1348 to <8 x half>, !dbg !14
  %address.local4219 = or disjoint i32 %address.local3955, 512, !dbg !14
  %address.local4220 = zext nneg i32 %address.local4219 to i64, !dbg !14
  %address.local4221 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4220, !dbg !14
  %1350 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4221, i32 0) #0, !dbg !14
  %1351 = bitcast <4 x i32> %1350 to <8 x half>, !dbg !14
  %address.local4230 = or disjoint i32 %address.local3955, 544, !dbg !14
  %address.local4231 = zext nneg i32 %address.local4230 to i64, !dbg !14
  %address.local4232 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4231, !dbg !14
  %1352 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4232, i32 0) #0, !dbg !14
  %1353 = bitcast <4 x i32> %1352 to <8 x half>, !dbg !14
  %address.local4241 = or disjoint i32 %address.local3955, 576, !dbg !14
  %address.local4242 = zext nneg i32 %address.local4241 to i64, !dbg !14
  %address.local4243 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4242, !dbg !14
  %1354 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4243, i32 0) #0, !dbg !14
  %1355 = bitcast <4 x i32> %1354 to <8 x half>, !dbg !14
  %address.local4252 = or disjoint i32 %address.local3955, 608, !dbg !14
  %address.local4253 = zext nneg i32 %address.local4252 to i64, !dbg !14
  %address.local4254 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local4253, !dbg !14
  %1356 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4254, i32 0) #0, !dbg !14
  %1357 = shufflevector <8 x half> %1343, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1357, <4 x float> %1329) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1359 = shufflevector <8 x half> %1343, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1359, <4 x float> %1331) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %1357, <4 x float> %1334) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %1359, <4 x float> %1336) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1363 = shufflevector <8 x half> %1345, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %1363, <4 x float> %1358) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1365 = shufflevector <8 x half> %1345, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %1365, <4 x float> %1360) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %1363, <4 x float> %1361) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %1365, <4 x float> %1362) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1369 = shufflevector <8 x half> %1347, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %1369, <4 x float> %1364) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1371 = shufflevector <8 x half> %1347, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %1371, <4 x float> %1366) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %1369, <4 x float> %1367) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %1371, <4 x float> %1368) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1375 = shufflevector <8 x half> %1349, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1375, <4 x float> %1370) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1377 = shufflevector <8 x half> %1349, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1377, <4 x float> %1372) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1375, <4 x float> %1373) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1377, <4 x float> %1374) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1381 = shufflevector <8 x half> %1351, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1381, <4 x float> %1376) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1383 = shufflevector <8 x half> %1351, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1383, <4 x float> %1378) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1381, <4 x float> %1379) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1383, <4 x float> %1380) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1387 = shufflevector <8 x half> %1353, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1387, <4 x float> %1382) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1389 = shufflevector <8 x half> %1353, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1389, <4 x float> %1384) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1387, <4 x float> %1385) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1389, <4 x float> %1386) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1393 = shufflevector <8 x half> %1355, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1393, <4 x float> %1388) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1395 = shufflevector <8 x half> %1355, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1395, <4 x float> %1390) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1393, <4 x float> %1391) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1395, <4 x float> %1392) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1399 = bitcast <4 x i32> %1356 to <8 x half>, !dbg !14
  %1400 = shufflevector <8 x half> %1399, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1400, <4 x float> %1394) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1402 = shufflevector <8 x half> %1399, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1402, <4 x float> %1396) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1400, <4 x float> %1397) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1402, <4 x float> %1398) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !30
  %1406 = extractvalue [2 x <8 x float>] %133, 0, !dbg !30
  %1407 = shufflevector <8 x float> %1406, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1408 = fmul <4 x float> %1407, %862, !dbg !30
  %1409 = shufflevector <8 x float> %1406, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1410 = fmul <4 x float> %1409, %862, !dbg !30
  %1411 = extractvalue [2 x <8 x float>] %133, 1, !dbg !30
  %1412 = shufflevector <8 x float> %1411, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1413 = fmul <4 x float> %1412, %870, !dbg !30
  %1414 = shufflevector <8 x float> %1411, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1415 = fmul <4 x float> %1414, %870, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %1416 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4067, i32 0) #0, !dbg !14
  %1417 = bitcast <4 x i32> %1416 to <8 x half>, !dbg !14
  %address.local4256 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4187, !dbg !14
  %1418 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4256, i32 0) #0, !dbg !14
  %1419 = bitcast <4 x i32> %1418 to <8 x half>, !dbg !14
  %address.local4257 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4198, !dbg !14
  %1420 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4257, i32 0) #0, !dbg !14
  %1421 = bitcast <4 x i32> %1420 to <8 x half>, !dbg !14
  %address.local4258 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4209, !dbg !14
  %1422 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4258, i32 0) #0, !dbg !14
  %1423 = bitcast <4 x i32> %1422 to <8 x half>, !dbg !14
  %address.local4259 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4220, !dbg !14
  %1424 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4259, i32 0) #0, !dbg !14
  %1425 = bitcast <4 x i32> %1424 to <8 x half>, !dbg !14
  %address.local4260 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4231, !dbg !14
  %1426 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4260, i32 0) #0, !dbg !14
  %1427 = bitcast <4 x i32> %1426 to <8 x half>, !dbg !14
  %address.local4261 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4242, !dbg !14
  %1428 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4261, i32 0) #0, !dbg !14
  %1429 = bitcast <4 x i32> %1428 to <8 x half>, !dbg !14
  %address.local4262 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local4253, !dbg !14
  %1430 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local4262, i32 0) #0, !dbg !14
  %1431 = shufflevector <8 x half> %1417, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1431, <4 x float> %1408) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1433 = shufflevector <8 x half> %1417, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1433, <4 x float> %1410) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %1431, <4 x float> %1413) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed926, <4 x half> %1433, <4 x float> %1415) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1437 = shufflevector <8 x half> %1419, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %1437, <4 x float> %1432) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1439 = shufflevector <8 x half> %1419, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed912, <4 x half> %1439, <4 x float> %1434) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %1437, <4 x float> %1435) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed928, <4 x half> %1439, <4 x float> %1436) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1443 = shufflevector <8 x half> %1421, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %1443, <4 x float> %1438) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1445 = shufflevector <8 x half> %1421, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed914, <4 x half> %1445, <4 x float> %1440) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %1443, <4 x float> %1441) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed930, <4 x half> %1445, <4 x float> %1442) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1449 = shufflevector <8 x half> %1423, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1449, <4 x float> %1444) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1451 = shufflevector <8 x half> %1423, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1452 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed916, <4 x half> %1451, <4 x float> %1446) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1449, <4 x float> %1447) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed932, <4 x half> %1451, <4 x float> %1448) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1455 = shufflevector <8 x half> %1425, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1455, <4 x float> %1450) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1457 = shufflevector <8 x half> %1425, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed918, <4 x half> %1457, <4 x float> %1452) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1455, <4 x float> %1453) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed934, <4 x half> %1457, <4 x float> %1454) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1461 = shufflevector <8 x half> %1427, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1461, <4 x float> %1456) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1463 = shufflevector <8 x half> %1427, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed920, <4 x half> %1463, <4 x float> %1458) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1461, <4 x float> %1459) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed936, <4 x half> %1463, <4 x float> %1460) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1467 = shufflevector <8 x half> %1429, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1467, <4 x float> %1462) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1469 = shufflevector <8 x half> %1429, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed922, <4 x half> %1469, <4 x float> %1464) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1467, <4 x float> %1465) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed938, <4 x half> %1469, <4 x float> %1466) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1473 = bitcast <4 x i32> %1430 to <8 x half>, !dbg !14
  %1474 = shufflevector <8 x half> %1473, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1474, <4 x float> %1468) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1476 = shufflevector <8 x half> %1473, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed924, <4 x half> %1476, <4 x float> %1470) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1474, <4 x float> %1471) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed940, <4 x half> %1476, <4 x float> %1472) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !6
  %1480 = shufflevector <4 x float> %1219, <4 x float> %1221, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1481 = insertvalue [2 x <8 x float>] poison, <8 x float> %1480, 0, !dbg !6
  %1482 = shufflevector <4 x float> %1246, <4 x float> %1247, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1483 = insertvalue [2 x <8 x float>] %1481, <8 x float> %1482, 1, !dbg !6
  %1484 = insertvalue [2 x <1 x float>] poison, <1 x float> %788, 0, !dbg !6
  %1485 = insertvalue [2 x <1 x float>] %1484, <1 x float> %791, 1, !dbg !6
  %1486 = shufflevector <4 x float> %1322, <4 x float> %1324, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1487 = insertvalue [2 x <8 x float>] poison, <8 x float> %1486, 0, !dbg !6
  %1488 = shufflevector <4 x float> %1325, <4 x float> %1326, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1489 = insertvalue [2 x <8 x float>] %1487, <8 x float> %1488, 1, !dbg !6
  %1490 = shufflevector <4 x float> %1401, <4 x float> %1403, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1491 = insertvalue [2 x <8 x float>] poison, <8 x float> %1490, 0, !dbg !6
  %1492 = shufflevector <4 x float> %1404, <4 x float> %1405, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1493 = insertvalue [2 x <8 x float>] %1491, <8 x float> %1492, 1, !dbg !6
  %1494 = shufflevector <4 x float> %1475, <4 x float> %1477, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1495 = insertvalue [2 x <8 x float>] poison, <8 x float> %1494, 0, !dbg !6
  %1496 = shufflevector <4 x float> %1478, <4 x float> %1479, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1497 = insertvalue [2 x <8 x float>] %1495, <8 x float> %1496, 1, !dbg !6
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  %1498 = add nuw nsw i64 %127, 1, !dbg !6
  br label %126, !dbg !6

1499:                                             ; preds = %126
  %1500 = extractvalue [2 x <1 x float>] %129, 1, !dbg !6
  %1501 = extractelement <1 x float> %1500, i64 0, !dbg !6
  %1502 = bitcast float %1501 to i32, !dbg !6
  %1503 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1502), !dbg !6
  %1504 = bitcast i32 %1503 to float, !dbg !6
  %1505 = fadd float %1501, %1504, !dbg !6
  %1506 = bitcast float %1505 to i32, !dbg !6
  %1507 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1506), !dbg !6
  %1508 = bitcast i32 %1507 to float, !dbg !6
  %1509 = fadd float %1505, %1508, !dbg !6
  %1510 = extractvalue [2 x <1 x float>] %129, 0, !dbg !6
  %1511 = extractelement <1 x float> %1510, i64 0, !dbg !6
  %1512 = bitcast float %1511 to i32, !dbg !6
  %1513 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1512), !dbg !6
  %1514 = bitcast i32 %1513 to float, !dbg !6
  %1515 = fadd float %1511, %1514, !dbg !6
  %1516 = bitcast float %1515 to i32, !dbg !6
  %1517 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1516), !dbg !6
  %1518 = bitcast i32 %1517 to float, !dbg !6
  %1519 = fadd float %1515, %1518, !dbg !6
  %1520 = and i32 %44, 16, !dbg !6
  %.not = icmp eq i32 %1520, 0, !dbg !6
  %1521 = and i32 %44, 32, !dbg !6
  %.not84 = icmp eq i32 %1521, 0, !dbg !6
  %1522 = extractvalue [2 x <8 x float>] %133, 0, !dbg !6
  %1523 = extractelement <8 x float> %1522, i64 0, !dbg !6
  %1524 = extractelement <8 x float> %1522, i64 1, !dbg !6
  %1525 = extractelement <8 x float> %1522, i64 2, !dbg !6
  %1526 = extractelement <8 x float> %1522, i64 3, !dbg !6
  %1527 = select i1 %.not, float %1524, float %1523, !dbg !6
  %1528 = bitcast float %1527 to i32, !dbg !6
  %1529 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1528), !dbg !6
  %1530 = bitcast i32 %1529 to float, !dbg !6
  %1531 = select i1 %.not, float %1523, float %1530, !dbg !6
  %1532 = select i1 %.not, float %1530, float %1524, !dbg !6
  %1533 = select i1 %.not, float %1526, float %1525, !dbg !6
  %1534 = bitcast float %1533 to i32, !dbg !6
  %1535 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1534), !dbg !6
  %1536 = bitcast i32 %1535 to float, !dbg !6
  %1537 = select i1 %.not, float %1525, float %1536, !dbg !6
  %1538 = select i1 %.not, float %1536, float %1526, !dbg !6
  %1539 = select i1 %.not84, float %1537, float %1531, !dbg !6
  %1540 = bitcast float %1539 to i32, !dbg !6
  %1541 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1540), !dbg !6
  %1542 = bitcast i32 %1541 to float, !dbg !6
  %1543 = select i1 %.not84, float %1531, float %1542, !dbg !6
  %1544 = select i1 %.not84, float %1542, float %1537, !dbg !6
  %1545 = select i1 %.not84, float %1538, float %1532, !dbg !6
  %1546 = bitcast float %1545 to i32, !dbg !6
  %1547 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1546), !dbg !6
  %1548 = bitcast i32 %1547 to float, !dbg !6
  %1549 = select i1 %.not84, float %1532, float %1548, !dbg !6
  %1550 = select i1 %.not84, float %1548, float %1538, !dbg !6
  %1551 = extractelement <8 x float> %1522, i64 4, !dbg !6
  %1552 = extractelement <8 x float> %1522, i64 5, !dbg !6
  %1553 = extractelement <8 x float> %1522, i64 6, !dbg !6
  %1554 = extractelement <8 x float> %1522, i64 7, !dbg !6
  %1555 = select i1 %.not, float %1552, float %1551, !dbg !6
  %1556 = bitcast float %1555 to i32, !dbg !6
  %1557 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1556), !dbg !6
  %1558 = bitcast i32 %1557 to float, !dbg !6
  %1559 = select i1 %.not, float %1551, float %1558, !dbg !6
  %1560 = select i1 %.not, float %1558, float %1552, !dbg !6
  %1561 = select i1 %.not, float %1554, float %1553, !dbg !6
  %1562 = bitcast float %1561 to i32, !dbg !6
  %1563 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1562), !dbg !6
  %1564 = bitcast i32 %1563 to float, !dbg !6
  %1565 = select i1 %.not, float %1553, float %1564, !dbg !6
  %1566 = select i1 %.not, float %1564, float %1554, !dbg !6
  %1567 = select i1 %.not84, float %1565, float %1559, !dbg !6
  %1568 = bitcast float %1567 to i32, !dbg !6
  %1569 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1568), !dbg !6
  %1570 = bitcast i32 %1569 to float, !dbg !6
  %1571 = select i1 %.not84, float %1559, float %1570, !dbg !6
  %1572 = select i1 %.not84, float %1570, float %1565, !dbg !6
  %1573 = select i1 %.not84, float %1566, float %1560, !dbg !6
  %1574 = bitcast float %1573 to i32, !dbg !6
  %1575 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1574), !dbg !6
  %1576 = bitcast i32 %1575 to float, !dbg !6
  %1577 = select i1 %.not84, float %1560, float %1576, !dbg !6
  %1578 = select i1 %.not84, float %1576, float %1566, !dbg !6
  %1579 = extractvalue [2 x <8 x float>] %133, 1, !dbg !6
  %1580 = extractelement <8 x float> %1579, i64 0, !dbg !6
  %1581 = extractelement <8 x float> %1579, i64 1, !dbg !6
  %1582 = extractelement <8 x float> %1579, i64 2, !dbg !6
  %1583 = extractelement <8 x float> %1579, i64 3, !dbg !6
  %1584 = select i1 %.not, float %1581, float %1580, !dbg !6
  %1585 = bitcast float %1584 to i32, !dbg !6
  %1586 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1585), !dbg !6
  %1587 = bitcast i32 %1586 to float, !dbg !6
  %1588 = select i1 %.not, float %1580, float %1587, !dbg !6
  %1589 = select i1 %.not, float %1587, float %1581, !dbg !6
  %1590 = select i1 %.not, float %1583, float %1582, !dbg !6
  %1591 = bitcast float %1590 to i32, !dbg !6
  %1592 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1591), !dbg !6
  %1593 = bitcast i32 %1592 to float, !dbg !6
  %1594 = select i1 %.not, float %1582, float %1593, !dbg !6
  %1595 = select i1 %.not, float %1593, float %1583, !dbg !6
  %1596 = select i1 %.not84, float %1594, float %1588, !dbg !6
  %1597 = bitcast float %1596 to i32, !dbg !6
  %1598 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1597), !dbg !6
  %1599 = bitcast i32 %1598 to float, !dbg !6
  %1600 = select i1 %.not84, float %1588, float %1599, !dbg !6
  %1601 = select i1 %.not84, float %1599, float %1594, !dbg !6
  %1602 = select i1 %.not84, float %1595, float %1589, !dbg !6
  %1603 = bitcast float %1602 to i32, !dbg !6
  %1604 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1603), !dbg !6
  %1605 = bitcast i32 %1604 to float, !dbg !6
  %1606 = select i1 %.not84, float %1589, float %1605, !dbg !6
  %1607 = select i1 %.not84, float %1605, float %1595, !dbg !6
  %1608 = extractelement <8 x float> %1579, i64 4, !dbg !6
  %1609 = extractelement <8 x float> %1579, i64 5, !dbg !6
  %1610 = extractelement <8 x float> %1579, i64 6, !dbg !6
  %1611 = extractelement <8 x float> %1579, i64 7, !dbg !6
  %1612 = select i1 %.not, float %1609, float %1608, !dbg !6
  %1613 = bitcast float %1612 to i32, !dbg !6
  %1614 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1613), !dbg !6
  %1615 = bitcast i32 %1614 to float, !dbg !6
  %1616 = select i1 %.not, float %1608, float %1615, !dbg !6
  %1617 = select i1 %.not, float %1615, float %1609, !dbg !6
  %1618 = select i1 %.not, float %1611, float %1610, !dbg !6
  %1619 = bitcast float %1618 to i32, !dbg !6
  %1620 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1619), !dbg !6
  %1621 = bitcast i32 %1620 to float, !dbg !6
  %1622 = select i1 %.not, float %1610, float %1621, !dbg !6
  %1623 = select i1 %.not, float %1621, float %1611, !dbg !6
  %1624 = select i1 %.not84, float %1622, float %1616, !dbg !6
  %1625 = bitcast float %1624 to i32, !dbg !6
  %1626 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1625), !dbg !6
  %1627 = bitcast i32 %1626 to float, !dbg !6
  %1628 = select i1 %.not84, float %1616, float %1627, !dbg !6
  %1629 = select i1 %.not84, float %1627, float %1622, !dbg !6
  %1630 = select i1 %.not84, float %1623, float %1617, !dbg !6
  %1631 = bitcast float %1630 to i32, !dbg !6
  %1632 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1631), !dbg !6
  %1633 = bitcast i32 %1632 to float, !dbg !6
  %1634 = select i1 %.not84, float %1617, float %1633, !dbg !6
  %1635 = select i1 %.not84, float %1633, float %1623, !dbg !6
  %1636 = extractvalue [2 x <8 x float>] %132, 0, !dbg !6
  %1637 = extractelement <8 x float> %1636, i64 0, !dbg !6
  %1638 = extractelement <8 x float> %1636, i64 1, !dbg !6
  %1639 = extractelement <8 x float> %1636, i64 2, !dbg !6
  %1640 = extractelement <8 x float> %1636, i64 3, !dbg !6
  %1641 = select i1 %.not, float %1638, float %1637, !dbg !6
  %1642 = bitcast float %1641 to i32, !dbg !6
  %1643 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1642), !dbg !6
  %1644 = bitcast i32 %1643 to float, !dbg !6
  %1645 = select i1 %.not, float %1637, float %1644, !dbg !6
  %1646 = select i1 %.not, float %1644, float %1638, !dbg !6
  %1647 = select i1 %.not, float %1640, float %1639, !dbg !6
  %1648 = bitcast float %1647 to i32, !dbg !6
  %1649 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1648), !dbg !6
  %1650 = bitcast i32 %1649 to float, !dbg !6
  %1651 = select i1 %.not, float %1639, float %1650, !dbg !6
  %1652 = select i1 %.not, float %1650, float %1640, !dbg !6
  %1653 = select i1 %.not84, float %1651, float %1645, !dbg !6
  %1654 = bitcast float %1653 to i32, !dbg !6
  %1655 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1654), !dbg !6
  %1656 = bitcast i32 %1655 to float, !dbg !6
  %1657 = select i1 %.not84, float %1645, float %1656, !dbg !6
  %1658 = select i1 %.not84, float %1656, float %1651, !dbg !6
  %1659 = select i1 %.not84, float %1652, float %1646, !dbg !6
  %1660 = bitcast float %1659 to i32, !dbg !6
  %1661 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1660), !dbg !6
  %1662 = bitcast i32 %1661 to float, !dbg !6
  %1663 = select i1 %.not84, float %1646, float %1662, !dbg !6
  %1664 = select i1 %.not84, float %1662, float %1652, !dbg !6
  %1665 = extractelement <8 x float> %1636, i64 4, !dbg !6
  %1666 = extractelement <8 x float> %1636, i64 5, !dbg !6
  %1667 = extractelement <8 x float> %1636, i64 6, !dbg !6
  %1668 = extractelement <8 x float> %1636, i64 7, !dbg !6
  %1669 = select i1 %.not, float %1666, float %1665, !dbg !6
  %1670 = bitcast float %1669 to i32, !dbg !6
  %1671 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1670), !dbg !6
  %1672 = bitcast i32 %1671 to float, !dbg !6
  %1673 = select i1 %.not, float %1665, float %1672, !dbg !6
  %1674 = select i1 %.not, float %1672, float %1666, !dbg !6
  %1675 = select i1 %.not, float %1668, float %1667, !dbg !6
  %1676 = bitcast float %1675 to i32, !dbg !6
  %1677 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1676), !dbg !6
  %1678 = bitcast i32 %1677 to float, !dbg !6
  %1679 = select i1 %.not, float %1667, float %1678, !dbg !6
  %1680 = select i1 %.not, float %1678, float %1668, !dbg !6
  %1681 = select i1 %.not84, float %1679, float %1673, !dbg !6
  %1682 = bitcast float %1681 to i32, !dbg !6
  %1683 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1682), !dbg !6
  %1684 = bitcast i32 %1683 to float, !dbg !6
  %1685 = select i1 %.not84, float %1673, float %1684, !dbg !6
  %1686 = select i1 %.not84, float %1684, float %1679, !dbg !6
  %1687 = select i1 %.not84, float %1680, float %1674, !dbg !6
  %1688 = bitcast float %1687 to i32, !dbg !6
  %1689 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1688), !dbg !6
  %1690 = bitcast i32 %1689 to float, !dbg !6
  %1691 = select i1 %.not84, float %1674, float %1690, !dbg !6
  %1692 = select i1 %.not84, float %1690, float %1680, !dbg !6
  %1693 = extractvalue [2 x <8 x float>] %132, 1, !dbg !6
  %1694 = extractelement <8 x float> %1693, i64 0, !dbg !6
  %1695 = extractelement <8 x float> %1693, i64 1, !dbg !6
  %1696 = extractelement <8 x float> %1693, i64 2, !dbg !6
  %1697 = extractelement <8 x float> %1693, i64 3, !dbg !6
  %1698 = select i1 %.not, float %1695, float %1694, !dbg !6
  %1699 = bitcast float %1698 to i32, !dbg !6
  %1700 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1699), !dbg !6
  %1701 = bitcast i32 %1700 to float, !dbg !6
  %1702 = select i1 %.not, float %1694, float %1701, !dbg !6
  %1703 = select i1 %.not, float %1701, float %1695, !dbg !6
  %1704 = select i1 %.not, float %1697, float %1696, !dbg !6
  %1705 = bitcast float %1704 to i32, !dbg !6
  %1706 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1705), !dbg !6
  %1707 = bitcast i32 %1706 to float, !dbg !6
  %1708 = select i1 %.not, float %1696, float %1707, !dbg !6
  %1709 = select i1 %.not, float %1707, float %1697, !dbg !6
  %1710 = select i1 %.not84, float %1708, float %1702, !dbg !6
  %1711 = bitcast float %1710 to i32, !dbg !6
  %1712 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1711), !dbg !6
  %1713 = bitcast i32 %1712 to float, !dbg !6
  %1714 = select i1 %.not84, float %1702, float %1713, !dbg !6
  %1715 = select i1 %.not84, float %1713, float %1708, !dbg !6
  %1716 = select i1 %.not84, float %1709, float %1703, !dbg !6
  %1717 = bitcast float %1716 to i32, !dbg !6
  %1718 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1717), !dbg !6
  %1719 = bitcast i32 %1718 to float, !dbg !6
  %1720 = select i1 %.not84, float %1703, float %1719, !dbg !6
  %1721 = select i1 %.not84, float %1719, float %1709, !dbg !6
  %1722 = extractelement <8 x float> %1693, i64 4, !dbg !6
  %1723 = extractelement <8 x float> %1693, i64 5, !dbg !6
  %1724 = extractelement <8 x float> %1693, i64 6, !dbg !6
  %1725 = extractelement <8 x float> %1693, i64 7, !dbg !6
  %1726 = select i1 %.not, float %1723, float %1722, !dbg !6
  %1727 = bitcast float %1726 to i32, !dbg !6
  %1728 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1727), !dbg !6
  %1729 = bitcast i32 %1728 to float, !dbg !6
  %1730 = select i1 %.not, float %1722, float %1729, !dbg !6
  %1731 = select i1 %.not, float %1729, float %1723, !dbg !6
  %1732 = select i1 %.not, float %1725, float %1724, !dbg !6
  %1733 = bitcast float %1732 to i32, !dbg !6
  %1734 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1733), !dbg !6
  %1735 = bitcast i32 %1734 to float, !dbg !6
  %1736 = select i1 %.not, float %1724, float %1735, !dbg !6
  %1737 = select i1 %.not, float %1735, float %1725, !dbg !6
  %1738 = select i1 %.not84, float %1736, float %1730, !dbg !6
  %1739 = bitcast float %1738 to i32, !dbg !6
  %1740 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1739), !dbg !6
  %1741 = bitcast i32 %1740 to float, !dbg !6
  %1742 = select i1 %.not84, float %1730, float %1741, !dbg !6
  %1743 = select i1 %.not84, float %1741, float %1736, !dbg !6
  %1744 = select i1 %.not84, float %1737, float %1731, !dbg !6
  %1745 = bitcast float %1744 to i32, !dbg !6
  %1746 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1745), !dbg !6
  %1747 = bitcast i32 %1746 to float, !dbg !6
  %1748 = select i1 %.not84, float %1731, float %1747, !dbg !6
  %1749 = select i1 %.not84, float %1747, float %1737, !dbg !6
  %1750 = extractvalue [2 x <8 x float>] %131, 0, !dbg !6
  %1751 = extractelement <8 x float> %1750, i64 0, !dbg !6
  %1752 = extractelement <8 x float> %1750, i64 1, !dbg !6
  %1753 = extractelement <8 x float> %1750, i64 2, !dbg !6
  %1754 = extractelement <8 x float> %1750, i64 3, !dbg !6
  %1755 = select i1 %.not, float %1752, float %1751, !dbg !6
  %1756 = bitcast float %1755 to i32, !dbg !6
  %1757 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1756), !dbg !6
  %1758 = bitcast i32 %1757 to float, !dbg !6
  %1759 = select i1 %.not, float %1751, float %1758, !dbg !6
  %1760 = select i1 %.not, float %1758, float %1752, !dbg !6
  %1761 = select i1 %.not, float %1754, float %1753, !dbg !6
  %1762 = bitcast float %1761 to i32, !dbg !6
  %1763 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1762), !dbg !6
  %1764 = bitcast i32 %1763 to float, !dbg !6
  %1765 = select i1 %.not, float %1753, float %1764, !dbg !6
  %1766 = select i1 %.not, float %1764, float %1754, !dbg !6
  %1767 = select i1 %.not84, float %1765, float %1759, !dbg !6
  %1768 = bitcast float %1767 to i32, !dbg !6
  %1769 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1768), !dbg !6
  %1770 = bitcast i32 %1769 to float, !dbg !6
  %1771 = select i1 %.not84, float %1759, float %1770, !dbg !6
  %1772 = select i1 %.not84, float %1770, float %1765, !dbg !6
  %1773 = select i1 %.not84, float %1766, float %1760, !dbg !6
  %1774 = bitcast float %1773 to i32, !dbg !6
  %1775 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1774), !dbg !6
  %1776 = bitcast i32 %1775 to float, !dbg !6
  %1777 = select i1 %.not84, float %1760, float %1776, !dbg !6
  %1778 = select i1 %.not84, float %1776, float %1766, !dbg !6
  %1779 = extractelement <8 x float> %1750, i64 4, !dbg !6
  %1780 = extractelement <8 x float> %1750, i64 5, !dbg !6
  %1781 = extractelement <8 x float> %1750, i64 6, !dbg !6
  %1782 = extractelement <8 x float> %1750, i64 7, !dbg !6
  %1783 = select i1 %.not, float %1780, float %1779, !dbg !6
  %1784 = bitcast float %1783 to i32, !dbg !6
  %1785 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1784), !dbg !6
  %1786 = bitcast i32 %1785 to float, !dbg !6
  %1787 = select i1 %.not, float %1779, float %1786, !dbg !6
  %1788 = select i1 %.not, float %1786, float %1780, !dbg !6
  %1789 = select i1 %.not, float %1782, float %1781, !dbg !6
  %1790 = bitcast float %1789 to i32, !dbg !6
  %1791 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1790), !dbg !6
  %1792 = bitcast i32 %1791 to float, !dbg !6
  %1793 = select i1 %.not, float %1781, float %1792, !dbg !6
  %1794 = select i1 %.not, float %1792, float %1782, !dbg !6
  %1795 = select i1 %.not84, float %1793, float %1787, !dbg !6
  %1796 = bitcast float %1795 to i32, !dbg !6
  %1797 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1796), !dbg !6
  %1798 = bitcast i32 %1797 to float, !dbg !6
  %1799 = select i1 %.not84, float %1787, float %1798, !dbg !6
  %1800 = select i1 %.not84, float %1798, float %1793, !dbg !6
  %1801 = select i1 %.not84, float %1794, float %1788, !dbg !6
  %1802 = bitcast float %1801 to i32, !dbg !6
  %1803 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1802), !dbg !6
  %1804 = bitcast i32 %1803 to float, !dbg !6
  %1805 = select i1 %.not84, float %1788, float %1804, !dbg !6
  %1806 = select i1 %.not84, float %1804, float %1794, !dbg !6
  %1807 = extractvalue [2 x <8 x float>] %131, 1, !dbg !6
  %1808 = extractelement <8 x float> %1807, i64 0, !dbg !6
  %1809 = extractelement <8 x float> %1807, i64 1, !dbg !6
  %1810 = extractelement <8 x float> %1807, i64 2, !dbg !6
  %1811 = extractelement <8 x float> %1807, i64 3, !dbg !6
  %1812 = select i1 %.not, float %1809, float %1808, !dbg !6
  %1813 = bitcast float %1812 to i32, !dbg !6
  %1814 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1813), !dbg !6
  %1815 = bitcast i32 %1814 to float, !dbg !6
  %1816 = select i1 %.not, float %1808, float %1815, !dbg !6
  %1817 = select i1 %.not, float %1815, float %1809, !dbg !6
  %1818 = select i1 %.not, float %1811, float %1810, !dbg !6
  %1819 = bitcast float %1818 to i32, !dbg !6
  %1820 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1819), !dbg !6
  %1821 = bitcast i32 %1820 to float, !dbg !6
  %1822 = select i1 %.not, float %1810, float %1821, !dbg !6
  %1823 = select i1 %.not, float %1821, float %1811, !dbg !6
  %1824 = select i1 %.not84, float %1822, float %1816, !dbg !6
  %1825 = bitcast float %1824 to i32, !dbg !6
  %1826 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1825), !dbg !6
  %1827 = bitcast i32 %1826 to float, !dbg !6
  %1828 = select i1 %.not84, float %1816, float %1827, !dbg !6
  %1829 = select i1 %.not84, float %1827, float %1822, !dbg !6
  %1830 = select i1 %.not84, float %1823, float %1817, !dbg !6
  %1831 = bitcast float %1830 to i32, !dbg !6
  %1832 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1831), !dbg !6
  %1833 = bitcast i32 %1832 to float, !dbg !6
  %1834 = select i1 %.not84, float %1817, float %1833, !dbg !6
  %1835 = select i1 %.not84, float %1833, float %1823, !dbg !6
  %1836 = extractelement <8 x float> %1807, i64 4, !dbg !6
  %1837 = extractelement <8 x float> %1807, i64 5, !dbg !6
  %1838 = extractelement <8 x float> %1807, i64 6, !dbg !6
  %1839 = extractelement <8 x float> %1807, i64 7, !dbg !6
  %1840 = select i1 %.not, float %1837, float %1836, !dbg !6
  %1841 = bitcast float %1840 to i32, !dbg !6
  %1842 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1841), !dbg !6
  %1843 = bitcast i32 %1842 to float, !dbg !6
  %1844 = select i1 %.not, float %1836, float %1843, !dbg !6
  %1845 = select i1 %.not, float %1843, float %1837, !dbg !6
  %1846 = select i1 %.not, float %1839, float %1838, !dbg !6
  %1847 = bitcast float %1846 to i32, !dbg !6
  %1848 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1847), !dbg !6
  %1849 = bitcast i32 %1848 to float, !dbg !6
  %1850 = select i1 %.not, float %1838, float %1849, !dbg !6
  %1851 = select i1 %.not, float %1849, float %1839, !dbg !6
  %1852 = select i1 %.not84, float %1850, float %1844, !dbg !6
  %1853 = bitcast float %1852 to i32, !dbg !6
  %1854 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1853), !dbg !6
  %1855 = bitcast i32 %1854 to float, !dbg !6
  %1856 = select i1 %.not84, float %1844, float %1855, !dbg !6
  %1857 = select i1 %.not84, float %1855, float %1850, !dbg !6
  %1858 = select i1 %.not84, float %1851, float %1845, !dbg !6
  %1859 = bitcast float %1858 to i32, !dbg !6
  %1860 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1859), !dbg !6
  %1861 = bitcast i32 %1860 to float, !dbg !6
  %1862 = select i1 %.not84, float %1845, float %1861, !dbg !6
  %1863 = select i1 %.not84, float %1861, float %1851, !dbg !6
  %1864 = extractvalue [2 x <8 x float>] %128, 0, !dbg !6
  %1865 = extractelement <8 x float> %1864, i64 0, !dbg !6
  %1866 = extractelement <8 x float> %1864, i64 1, !dbg !6
  %1867 = extractelement <8 x float> %1864, i64 2, !dbg !6
  %1868 = extractelement <8 x float> %1864, i64 3, !dbg !6
  %1869 = select i1 %.not, float %1866, float %1865, !dbg !6
  %1870 = bitcast float %1869 to i32, !dbg !6
  %1871 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1870), !dbg !6
  %1872 = bitcast i32 %1871 to float, !dbg !6
  %1873 = select i1 %.not, float %1865, float %1872, !dbg !6
  %1874 = select i1 %.not, float %1872, float %1866, !dbg !6
  %1875 = select i1 %.not, float %1868, float %1867, !dbg !6
  %1876 = bitcast float %1875 to i32, !dbg !6
  %1877 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1876), !dbg !6
  %1878 = bitcast i32 %1877 to float, !dbg !6
  %1879 = select i1 %.not, float %1867, float %1878, !dbg !6
  %1880 = select i1 %.not, float %1878, float %1868, !dbg !6
  %1881 = select i1 %.not84, float %1879, float %1873, !dbg !6
  %1882 = bitcast float %1881 to i32, !dbg !6
  %1883 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1882), !dbg !6
  %1884 = select i1 %.not84, float %1880, float %1874, !dbg !6
  %1885 = bitcast float %1884 to i32, !dbg !6
  %1886 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1885), !dbg !6
  %1887 = extractelement <8 x float> %1864, i64 4, !dbg !6
  %1888 = extractelement <8 x float> %1864, i64 5, !dbg !6
  %1889 = extractelement <8 x float> %1864, i64 6, !dbg !6
  %1890 = extractelement <8 x float> %1864, i64 7, !dbg !6
  %1891 = select i1 %.not, float %1888, float %1887, !dbg !6
  %1892 = bitcast float %1891 to i32, !dbg !6
  %1893 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1892), !dbg !6
  %1894 = bitcast i32 %1893 to float, !dbg !6
  %1895 = select i1 %.not, float %1887, float %1894, !dbg !6
  %1896 = select i1 %.not, float %1894, float %1888, !dbg !6
  %1897 = select i1 %.not, float %1890, float %1889, !dbg !6
  %1898 = bitcast float %1897 to i32, !dbg !6
  %1899 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1898), !dbg !6
  %1900 = bitcast i32 %1899 to float, !dbg !6
  %1901 = select i1 %.not, float %1889, float %1900, !dbg !6
  %1902 = select i1 %.not, float %1900, float %1890, !dbg !6
  %1903 = select i1 %.not84, float %1901, float %1895, !dbg !6
  %1904 = bitcast float %1903 to i32, !dbg !6
  %1905 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1904), !dbg !6
  %1906 = select i1 %.not84, float %1902, float %1896, !dbg !6
  %1907 = bitcast float %1906 to i32, !dbg !6
  %1908 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1907), !dbg !6
  %1909 = extractvalue [2 x <8 x float>] %128, 1, !dbg !6
  %1910 = extractelement <8 x float> %1909, i64 0, !dbg !6
  %1911 = extractelement <8 x float> %1909, i64 1, !dbg !6
  %1912 = extractelement <8 x float> %1909, i64 2, !dbg !6
  %1913 = extractelement <8 x float> %1909, i64 3, !dbg !6
  %1914 = select i1 %.not, float %1911, float %1910, !dbg !6
  %1915 = bitcast float %1914 to i32, !dbg !6
  %1916 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1915), !dbg !6
  %1917 = bitcast i32 %1916 to float, !dbg !6
  %1918 = select i1 %.not, float %1910, float %1917, !dbg !6
  %1919 = select i1 %.not, float %1917, float %1911, !dbg !6
  %1920 = select i1 %.not, float %1913, float %1912, !dbg !6
  %1921 = bitcast float %1920 to i32, !dbg !6
  %1922 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1921), !dbg !6
  %1923 = bitcast i32 %1922 to float, !dbg !6
  %1924 = select i1 %.not, float %1912, float %1923, !dbg !6
  %1925 = select i1 %.not, float %1923, float %1913, !dbg !6
  %1926 = select i1 %.not84, float %1924, float %1918, !dbg !6
  %1927 = bitcast float %1926 to i32, !dbg !6
  %1928 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1927), !dbg !6
  %1929 = select i1 %.not84, float %1925, float %1919, !dbg !6
  %1930 = bitcast float %1929 to i32, !dbg !6
  %1931 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1930), !dbg !6
  %1932 = extractelement <8 x float> %1909, i64 4, !dbg !6
  %1933 = extractelement <8 x float> %1909, i64 5, !dbg !6
  %1934 = extractelement <8 x float> %1909, i64 6, !dbg !6
  %1935 = extractelement <8 x float> %1909, i64 7, !dbg !6
  %1936 = select i1 %.not, float %1933, float %1932, !dbg !6
  %1937 = bitcast float %1936 to i32, !dbg !6
  %1938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1937), !dbg !6
  %1939 = bitcast i32 %1938 to float, !dbg !6
  %1940 = select i1 %.not, float %1932, float %1939, !dbg !6
  %1941 = select i1 %.not, float %1939, float %1933, !dbg !6
  %1942 = select i1 %.not, float %1935, float %1934, !dbg !6
  %1943 = bitcast float %1942 to i32, !dbg !6
  %1944 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1943), !dbg !6
  %1945 = bitcast i32 %1944 to float, !dbg !6
  %1946 = select i1 %.not, float %1934, float %1945, !dbg !6
  %1947 = select i1 %.not, float %1945, float %1935, !dbg !6
  %1948 = select i1 %.not84, float %1946, float %1940, !dbg !6
  %1949 = bitcast float %1948 to i32, !dbg !6
  %1950 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1949), !dbg !6
  %1951 = select i1 %.not84, float %1947, float %1941, !dbg !6
  %1952 = bitcast float %1951 to i32, !dbg !6
  %1953 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1952), !dbg !6
  %1954 = insertelement <4 x float> poison, float %1519, i64 0, !dbg !19
  %1955 = fdiv arcp <4 x float> <float 0x3FF0000000000000, float poison, float poison, float poison>, %1954, !dbg !19
  %1956 = shufflevector <4 x float> %1955, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1957 = insertelement <4 x float> poison, float %1509, i64 0, !dbg !19
  %1958 = fdiv arcp <4 x float> <float 0x3FF0000000000000, float poison, float poison, float poison>, %1957, !dbg !19
  %1959 = shufflevector <4 x float> %1958, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1960 = bitcast i32 %1950 to float, !dbg !6
  %1961 = select i1 %.not84, float %1940, float %1960, !dbg !6
  %1962 = insertelement <4 x float> poison, float %1961, i64 0, !dbg !19
  %1963 = bitcast i32 %1953 to float, !dbg !6
  %1964 = select i1 %.not84, float %1941, float %1963, !dbg !6
  %1965 = insertelement <4 x float> %1962, float %1964, i64 1, !dbg !19
  %1966 = select i1 %.not84, float %1960, float %1946, !dbg !6
  %1967 = insertelement <4 x float> %1965, float %1966, i64 2, !dbg !19
  %1968 = select i1 %.not84, float %1963, float %1947, !dbg !6
  %1969 = insertelement <4 x float> %1967, float %1968, i64 3, !dbg !19
  %1970 = fmul <4 x float> %1969, %1959, !dbg !19
  %1971 = fptrunc <4 x float> %1970 to <4 x half>, !dbg !20
  %1972 = extractelement <4 x half> %1971, i64 3, !dbg !20
  %1973 = extractelement <4 x half> %1971, i64 2, !dbg !20
  %1974 = extractelement <4 x half> %1971, i64 1, !dbg !20
  %1975 = extractelement <4 x half> %1971, i64 0, !dbg !20
  %1976 = bitcast i32 %1928 to float, !dbg !6
  %1977 = select i1 %.not84, float %1918, float %1976, !dbg !6
  %1978 = insertelement <4 x float> poison, float %1977, i64 0, !dbg !19
  %1979 = bitcast i32 %1931 to float, !dbg !6
  %1980 = select i1 %.not84, float %1919, float %1979, !dbg !6
  %1981 = insertelement <4 x float> %1978, float %1980, i64 1, !dbg !19
  %1982 = select i1 %.not84, float %1976, float %1924, !dbg !6
  %1983 = insertelement <4 x float> %1981, float %1982, i64 2, !dbg !19
  %1984 = select i1 %.not84, float %1979, float %1925, !dbg !6
  %1985 = insertelement <4 x float> %1983, float %1984, i64 3, !dbg !19
  %1986 = fmul <4 x float> %1985, %1959, !dbg !19
  %1987 = fptrunc <4 x float> %1986 to <4 x half>, !dbg !20
  %1988 = extractelement <4 x half> %1987, i64 3, !dbg !20
  %1989 = extractelement <4 x half> %1987, i64 2, !dbg !20
  %1990 = extractelement <4 x half> %1987, i64 1, !dbg !20
  %1991 = extractelement <4 x half> %1987, i64 0, !dbg !20
  %1992 = bitcast i32 %1905 to float, !dbg !6
  %1993 = select i1 %.not84, float %1895, float %1992, !dbg !6
  %1994 = insertelement <4 x float> poison, float %1993, i64 0, !dbg !19
  %1995 = bitcast i32 %1908 to float, !dbg !6
  %1996 = select i1 %.not84, float %1896, float %1995, !dbg !6
  %1997 = insertelement <4 x float> %1994, float %1996, i64 1, !dbg !19
  %1998 = select i1 %.not84, float %1992, float %1901, !dbg !6
  %1999 = insertelement <4 x float> %1997, float %1998, i64 2, !dbg !19
  %2000 = select i1 %.not84, float %1995, float %1902, !dbg !6
  %2001 = insertelement <4 x float> %1999, float %2000, i64 3, !dbg !19
  %2002 = fmul <4 x float> %2001, %1956, !dbg !19
  %2003 = fptrunc <4 x float> %2002 to <4 x half>, !dbg !20
  %2004 = extractelement <4 x half> %2003, i64 3, !dbg !20
  %2005 = extractelement <4 x half> %2003, i64 2, !dbg !20
  %2006 = extractelement <4 x half> %2003, i64 1, !dbg !20
  %2007 = extractelement <4 x half> %2003, i64 0, !dbg !20
  %2008 = bitcast i32 %1883 to float, !dbg !6
  %2009 = select i1 %.not84, float %1873, float %2008, !dbg !6
  %2010 = insertelement <4 x float> poison, float %2009, i64 0, !dbg !19
  %2011 = bitcast i32 %1886 to float, !dbg !6
  %2012 = select i1 %.not84, float %1874, float %2011, !dbg !6
  %2013 = insertelement <4 x float> %2010, float %2012, i64 1, !dbg !19
  %2014 = select i1 %.not84, float %2008, float %1879, !dbg !6
  %2015 = insertelement <4 x float> %2013, float %2014, i64 2, !dbg !19
  %2016 = select i1 %.not84, float %2011, float %1880, !dbg !6
  %2017 = insertelement <4 x float> %2015, float %2016, i64 3, !dbg !19
  %2018 = fmul <4 x float> %2017, %1956, !dbg !19
  %2019 = fptrunc <4 x float> %2018 to <4 x half>, !dbg !20
  %2020 = extractelement <4 x half> %2019, i64 3, !dbg !20
  %2021 = extractelement <4 x half> %2019, i64 2, !dbg !20
  %2022 = extractelement <4 x half> %2019, i64 1, !dbg !20
  %2023 = extractelement <4 x half> %2019, i64 0, !dbg !20
  %address.thread4263 = call i32 asm sideeffect "", "=v,0"(i32 %44) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local4264 = zext nneg i32 %address.thread4263 to i64, !dbg !9
  %address.local4265 = add i64 %40, %address.local4264, !dbg !21
  %2024 = lshr i64 %address.local4264, 2, !dbg !18
  %address.local4267 = and i64 %2024, 48, !dbg !18
  %address.local4268 = add i64 %address.local4265, %address.local4267, !dbg !21
  %address.local4269 = and i64 %address.local4264, 240, !dbg !21
  %address.local4271 = sub i64 %address.local4268, %address.local4269, !dbg !21
  %address.local4272 = add i64 %address.local4271, %41, !dbg !21
  %address.local4273 = shl nuw nsw i64 %address.local4272, 7, !dbg !21
  %2025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %56, !dbg !21
  %address.local4275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %2025, i64 %address.local4273, !dbg !21
  %address.local4278 = and i64 %2024, 12, !dbg !13
  %address.local4279 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4278, !dbg !21
  store half %2023, ptr addrspace(1) %address.local4279, align 2, !dbg !21
  %address.local4281 = or disjoint i64 %address.local4278, 1, !dbg !21
  %address.local4282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4281, !dbg !21
  store half %2022, ptr addrspace(1) %address.local4282, align 2, !dbg !21
  %address.local4284 = or disjoint i64 %address.local4278, 2, !dbg !21
  %address.local4285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4284, !dbg !21
  store half %2021, ptr addrspace(1) %address.local4285, align 2, !dbg !21
  %address.local4287 = or disjoint i64 %address.local4278, 3, !dbg !21
  %address.local4288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4287, !dbg !21
  store half %2020, ptr addrspace(1) %address.local4288, align 2, !dbg !21
  %address.local4290 = or disjoint i64 %address.local4278, 16, !dbg !21
  %address.local4291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4290, !dbg !21
  store half %2007, ptr addrspace(1) %address.local4291, align 2, !dbg !21
  %address.local4293 = or disjoint i64 %address.local4278, 17, !dbg !21
  %address.local4294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4293, !dbg !21
  store half %2006, ptr addrspace(1) %address.local4294, align 2, !dbg !21
  %address.local4296 = or disjoint i64 %address.local4278, 18, !dbg !21
  %address.local4297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4296, !dbg !21
  store half %2005, ptr addrspace(1) %address.local4297, align 2, !dbg !21
  %address.local4299 = or disjoint i64 %address.local4278, 19, !dbg !21
  %address.local4300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4299, !dbg !21
  store half %2004, ptr addrspace(1) %address.local4300, align 2, !dbg !21
  %address.local4304 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local4275, i64 16384, !dbg !21
  %address.local4305 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4278, !dbg !21
  store half %1991, ptr addrspace(1) %address.local4305, align 2, !dbg !21
  %address.local4307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4281, !dbg !21
  store half %1990, ptr addrspace(1) %address.local4307, align 2, !dbg !21
  %address.local4309 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4284, !dbg !21
  store half %1989, ptr addrspace(1) %address.local4309, align 2, !dbg !21
  %address.local4311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4287, !dbg !21
  store half %1988, ptr addrspace(1) %address.local4311, align 2, !dbg !21
  %address.local4313 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4290, !dbg !21
  store half %1975, ptr addrspace(1) %address.local4313, align 2, !dbg !21
  %address.local4315 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4293, !dbg !21
  store half %1974, ptr addrspace(1) %address.local4315, align 2, !dbg !21
  %address.local4317 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4296, !dbg !21
  store half %1973, ptr addrspace(1) %address.local4317, align 2, !dbg !21
  %address.local4319 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4299, !dbg !21
  store half %1972, ptr addrspace(1) %address.local4319, align 2, !dbg !21
  %2026 = insertelement <4 x float> poison, float %1856, i64 0, !dbg !19
  %2027 = insertelement <4 x float> %2026, float %1862, i64 1, !dbg !19
  %2028 = insertelement <4 x float> %2027, float %1857, i64 2, !dbg !19
  %2029 = insertelement <4 x float> %2028, float %1863, i64 3, !dbg !19
  %2030 = fmul <4 x float> %2029, %1959, !dbg !19
  %2031 = fptrunc <4 x float> %2030 to <4 x half>, !dbg !20
  %2032 = extractelement <4 x half> %2031, i64 3, !dbg !20
  %2033 = extractelement <4 x half> %2031, i64 2, !dbg !20
  %2034 = extractelement <4 x half> %2031, i64 1, !dbg !20
  %2035 = extractelement <4 x half> %2031, i64 0, !dbg !20
  %2036 = insertelement <4 x float> poison, float %1828, i64 0, !dbg !19
  %2037 = insertelement <4 x float> %2036, float %1834, i64 1, !dbg !19
  %2038 = insertelement <4 x float> %2037, float %1829, i64 2, !dbg !19
  %2039 = insertelement <4 x float> %2038, float %1835, i64 3, !dbg !19
  %2040 = fmul <4 x float> %2039, %1959, !dbg !19
  %2041 = fptrunc <4 x float> %2040 to <4 x half>, !dbg !20
  %2042 = extractelement <4 x half> %2041, i64 3, !dbg !20
  %2043 = extractelement <4 x half> %2041, i64 2, !dbg !20
  %2044 = extractelement <4 x half> %2041, i64 1, !dbg !20
  %2045 = extractelement <4 x half> %2041, i64 0, !dbg !20
  %2046 = insertelement <4 x float> poison, float %1799, i64 0, !dbg !19
  %2047 = insertelement <4 x float> %2046, float %1805, i64 1, !dbg !19
  %2048 = insertelement <4 x float> %2047, float %1800, i64 2, !dbg !19
  %2049 = insertelement <4 x float> %2048, float %1806, i64 3, !dbg !19
  %2050 = fmul <4 x float> %2049, %1956, !dbg !19
  %2051 = fptrunc <4 x float> %2050 to <4 x half>, !dbg !20
  %2052 = extractelement <4 x half> %2051, i64 3, !dbg !20
  %2053 = extractelement <4 x half> %2051, i64 2, !dbg !20
  %2054 = extractelement <4 x half> %2051, i64 1, !dbg !20
  %2055 = extractelement <4 x half> %2051, i64 0, !dbg !20
  %2056 = insertelement <4 x float> poison, float %1771, i64 0, !dbg !19
  %2057 = insertelement <4 x float> %2056, float %1777, i64 1, !dbg !19
  %2058 = insertelement <4 x float> %2057, float %1772, i64 2, !dbg !19
  %2059 = insertelement <4 x float> %2058, float %1778, i64 3, !dbg !19
  %2060 = fmul <4 x float> %2059, %1956, !dbg !19
  %2061 = fptrunc <4 x float> %2060 to <4 x half>, !dbg !20
  %2062 = extractelement <4 x half> %2061, i64 3, !dbg !20
  %2063 = extractelement <4 x half> %2061, i64 2, !dbg !20
  %2064 = extractelement <4 x half> %2061, i64 1, !dbg !20
  %2065 = extractelement <4 x half> %2061, i64 0, !dbg !20
  %address.local4336 = or disjoint i64 %address.local4278, 32, !dbg !21
  %address.local4337 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4336, !dbg !21
  store half %2065, ptr addrspace(1) %address.local4337, align 2, !dbg !21
  %address.local4339 = or disjoint i64 %address.local4278, 33, !dbg !21
  %address.local4340 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4339, !dbg !21
  store half %2064, ptr addrspace(1) %address.local4340, align 2, !dbg !21
  %address.local4342 = or disjoint i64 %address.local4278, 34, !dbg !21
  %address.local4343 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4342, !dbg !21
  store half %2063, ptr addrspace(1) %address.local4343, align 2, !dbg !21
  %address.local4345 = or disjoint i64 %address.local4278, 35, !dbg !21
  %address.local4346 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4345, !dbg !21
  store half %2062, ptr addrspace(1) %address.local4346, align 2, !dbg !21
  %address.local4348 = or disjoint i64 %address.local4278, 48, !dbg !21
  %address.local4349 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4348, !dbg !21
  store half %2055, ptr addrspace(1) %address.local4349, align 2, !dbg !21
  %address.local4351 = or disjoint i64 %address.local4278, 49, !dbg !21
  %address.local4352 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4351, !dbg !21
  store half %2054, ptr addrspace(1) %address.local4352, align 2, !dbg !21
  %address.local4354 = or disjoint i64 %address.local4278, 50, !dbg !21
  %address.local4355 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4354, !dbg !21
  store half %2053, ptr addrspace(1) %address.local4355, align 2, !dbg !21
  %address.local4357 = or i64 %2024, 51, !dbg !21
  %address.local4358 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4357, !dbg !21
  store half %2052, ptr addrspace(1) %address.local4358, align 2, !dbg !21
  %address.local4363 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4336, !dbg !21
  store half %2045, ptr addrspace(1) %address.local4363, align 2, !dbg !21
  %address.local4365 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4339, !dbg !21
  store half %2044, ptr addrspace(1) %address.local4365, align 2, !dbg !21
  %address.local4367 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4342, !dbg !21
  store half %2043, ptr addrspace(1) %address.local4367, align 2, !dbg !21
  %address.local4369 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4345, !dbg !21
  store half %2042, ptr addrspace(1) %address.local4369, align 2, !dbg !21
  %address.local4371 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4348, !dbg !21
  store half %2035, ptr addrspace(1) %address.local4371, align 2, !dbg !21
  %address.local4373 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4351, !dbg !21
  store half %2034, ptr addrspace(1) %address.local4373, align 2, !dbg !21
  %address.local4375 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4354, !dbg !21
  store half %2033, ptr addrspace(1) %address.local4375, align 2, !dbg !21
  %address.local4377 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4357, !dbg !21
  store half %2032, ptr addrspace(1) %address.local4377, align 2, !dbg !21
  %2066 = insertelement <4 x float> poison, float %1742, i64 0, !dbg !19
  %2067 = insertelement <4 x float> %2066, float %1748, i64 1, !dbg !19
  %2068 = insertelement <4 x float> %2067, float %1743, i64 2, !dbg !19
  %2069 = insertelement <4 x float> %2068, float %1749, i64 3, !dbg !19
  %2070 = fmul <4 x float> %2069, %1959, !dbg !19
  %2071 = fptrunc <4 x float> %2070 to <4 x half>, !dbg !20
  %2072 = extractelement <4 x half> %2071, i64 3, !dbg !20
  %2073 = extractelement <4 x half> %2071, i64 2, !dbg !20
  %2074 = extractelement <4 x half> %2071, i64 1, !dbg !20
  %2075 = extractelement <4 x half> %2071, i64 0, !dbg !20
  %2076 = insertelement <4 x float> poison, float %1714, i64 0, !dbg !19
  %2077 = insertelement <4 x float> %2076, float %1720, i64 1, !dbg !19
  %2078 = insertelement <4 x float> %2077, float %1715, i64 2, !dbg !19
  %2079 = insertelement <4 x float> %2078, float %1721, i64 3, !dbg !19
  %2080 = fmul <4 x float> %2079, %1959, !dbg !19
  %2081 = fptrunc <4 x float> %2080 to <4 x half>, !dbg !20
  %2082 = extractelement <4 x half> %2081, i64 3, !dbg !20
  %2083 = extractelement <4 x half> %2081, i64 2, !dbg !20
  %2084 = extractelement <4 x half> %2081, i64 1, !dbg !20
  %2085 = extractelement <4 x half> %2081, i64 0, !dbg !20
  %2086 = insertelement <4 x float> poison, float %1685, i64 0, !dbg !19
  %2087 = insertelement <4 x float> %2086, float %1691, i64 1, !dbg !19
  %2088 = insertelement <4 x float> %2087, float %1686, i64 2, !dbg !19
  %2089 = insertelement <4 x float> %2088, float %1692, i64 3, !dbg !19
  %2090 = fmul <4 x float> %2089, %1956, !dbg !19
  %2091 = fptrunc <4 x float> %2090 to <4 x half>, !dbg !20
  %2092 = extractelement <4 x half> %2091, i64 3, !dbg !20
  %2093 = extractelement <4 x half> %2091, i64 2, !dbg !20
  %2094 = extractelement <4 x half> %2091, i64 1, !dbg !20
  %2095 = extractelement <4 x half> %2091, i64 0, !dbg !20
  %2096 = insertelement <4 x float> poison, float %1657, i64 0, !dbg !19
  %2097 = insertelement <4 x float> %2096, float %1663, i64 1, !dbg !19
  %2098 = insertelement <4 x float> %2097, float %1658, i64 2, !dbg !19
  %2099 = insertelement <4 x float> %2098, float %1664, i64 3, !dbg !19
  %2100 = fmul <4 x float> %2099, %1956, !dbg !19
  %2101 = fptrunc <4 x float> %2100 to <4 x half>, !dbg !20
  %2102 = extractelement <4 x half> %2101, i64 3, !dbg !20
  %2103 = extractelement <4 x half> %2101, i64 2, !dbg !20
  %2104 = extractelement <4 x half> %2101, i64 1, !dbg !20
  %2105 = extractelement <4 x half> %2101, i64 0, !dbg !20
  %address.local4394 = or disjoint i64 %address.local4278, 64, !dbg !21
  %address.local4395 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4394, !dbg !21
  store half %2105, ptr addrspace(1) %address.local4395, align 2, !dbg !21
  %address.local4397 = or disjoint i64 %address.local4278, 65, !dbg !21
  %address.local4398 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4397, !dbg !21
  store half %2104, ptr addrspace(1) %address.local4398, align 2, !dbg !21
  %address.local4400 = or disjoint i64 %address.local4278, 66, !dbg !21
  %address.local4401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4400, !dbg !21
  store half %2103, ptr addrspace(1) %address.local4401, align 2, !dbg !21
  %address.local4403 = or disjoint i64 %address.local4278, 67, !dbg !21
  %address.local4404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4403, !dbg !21
  store half %2102, ptr addrspace(1) %address.local4404, align 2, !dbg !21
  %address.local4406 = or disjoint i64 %address.local4278, 80, !dbg !21
  %address.local4407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4406, !dbg !21
  store half %2095, ptr addrspace(1) %address.local4407, align 2, !dbg !21
  %address.local4409 = or disjoint i64 %address.local4278, 81, !dbg !21
  %address.local4410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4409, !dbg !21
  store half %2094, ptr addrspace(1) %address.local4410, align 2, !dbg !21
  %address.local4412 = or disjoint i64 %address.local4278, 82, !dbg !21
  %address.local4413 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4412, !dbg !21
  store half %2093, ptr addrspace(1) %address.local4413, align 2, !dbg !21
  %address.local4415 = or disjoint i64 %address.local4278, 83, !dbg !21
  %address.local4416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4415, !dbg !21
  store half %2092, ptr addrspace(1) %address.local4416, align 2, !dbg !21
  %address.local4421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4394, !dbg !21
  store half %2085, ptr addrspace(1) %address.local4421, align 2, !dbg !21
  %address.local4423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4397, !dbg !21
  store half %2084, ptr addrspace(1) %address.local4423, align 2, !dbg !21
  %address.local4425 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4400, !dbg !21
  store half %2083, ptr addrspace(1) %address.local4425, align 2, !dbg !21
  %address.local4427 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4403, !dbg !21
  store half %2082, ptr addrspace(1) %address.local4427, align 2, !dbg !21
  %address.local4429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4406, !dbg !21
  store half %2075, ptr addrspace(1) %address.local4429, align 2, !dbg !21
  %address.local4431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4409, !dbg !21
  store half %2074, ptr addrspace(1) %address.local4431, align 2, !dbg !21
  %address.local4433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4412, !dbg !21
  store half %2073, ptr addrspace(1) %address.local4433, align 2, !dbg !21
  %address.local4435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4415, !dbg !21
  store half %2072, ptr addrspace(1) %address.local4435, align 2, !dbg !21
  %2106 = insertelement <4 x float> poison, float %1628, i64 0, !dbg !19
  %2107 = insertelement <4 x float> %2106, float %1634, i64 1, !dbg !19
  %2108 = insertelement <4 x float> %2107, float %1629, i64 2, !dbg !19
  %2109 = insertelement <4 x float> %2108, float %1635, i64 3, !dbg !19
  %2110 = fmul <4 x float> %2109, %1959, !dbg !19
  %2111 = fptrunc <4 x float> %2110 to <4 x half>, !dbg !20
  %2112 = extractelement <4 x half> %2111, i64 3, !dbg !20
  %2113 = extractelement <4 x half> %2111, i64 2, !dbg !20
  %2114 = extractelement <4 x half> %2111, i64 1, !dbg !20
  %2115 = extractelement <4 x half> %2111, i64 0, !dbg !20
  %2116 = insertelement <4 x float> poison, float %1600, i64 0, !dbg !19
  %2117 = insertelement <4 x float> %2116, float %1606, i64 1, !dbg !19
  %2118 = insertelement <4 x float> %2117, float %1601, i64 2, !dbg !19
  %2119 = insertelement <4 x float> %2118, float %1607, i64 3, !dbg !19
  %2120 = fmul <4 x float> %2119, %1959, !dbg !19
  %2121 = fptrunc <4 x float> %2120 to <4 x half>, !dbg !20
  %2122 = extractelement <4 x half> %2121, i64 3, !dbg !20
  %2123 = extractelement <4 x half> %2121, i64 2, !dbg !20
  %2124 = extractelement <4 x half> %2121, i64 1, !dbg !20
  %2125 = extractelement <4 x half> %2121, i64 0, !dbg !20
  %2126 = insertelement <4 x float> poison, float %1571, i64 0, !dbg !19
  %2127 = insertelement <4 x float> %2126, float %1577, i64 1, !dbg !19
  %2128 = insertelement <4 x float> %2127, float %1572, i64 2, !dbg !19
  %2129 = insertelement <4 x float> %2128, float %1578, i64 3, !dbg !19
  %2130 = fmul <4 x float> %2129, %1956, !dbg !19
  %2131 = fptrunc <4 x float> %2130 to <4 x half>, !dbg !20
  %2132 = extractelement <4 x half> %2131, i64 3, !dbg !20
  %2133 = extractelement <4 x half> %2131, i64 2, !dbg !20
  %2134 = extractelement <4 x half> %2131, i64 1, !dbg !20
  %2135 = extractelement <4 x half> %2131, i64 0, !dbg !20
  %2136 = insertelement <4 x float> poison, float %1543, i64 0, !dbg !19
  %2137 = insertelement <4 x float> %2136, float %1549, i64 1, !dbg !19
  %2138 = insertelement <4 x float> %2137, float %1544, i64 2, !dbg !19
  %2139 = insertelement <4 x float> %2138, float %1550, i64 3, !dbg !19
  %2140 = fmul <4 x float> %2139, %1956, !dbg !19
  %2141 = fptrunc <4 x float> %2140 to <4 x half>, !dbg !20
  %2142 = extractelement <4 x half> %2141, i64 3, !dbg !20
  %2143 = extractelement <4 x half> %2141, i64 2, !dbg !20
  %2144 = extractelement <4 x half> %2141, i64 1, !dbg !20
  %2145 = extractelement <4 x half> %2141, i64 0, !dbg !20
  %address.local4452 = or disjoint i64 %address.local4278, 96, !dbg !21
  %address.local4453 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4452, !dbg !21
  store half %2145, ptr addrspace(1) %address.local4453, align 2, !dbg !21
  %address.local4455 = or disjoint i64 %address.local4278, 97, !dbg !21
  %address.local4456 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4455, !dbg !21
  store half %2144, ptr addrspace(1) %address.local4456, align 2, !dbg !21
  %address.local4458 = or disjoint i64 %address.local4278, 98, !dbg !21
  %address.local4459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4458, !dbg !21
  store half %2143, ptr addrspace(1) %address.local4459, align 2, !dbg !21
  %address.local4461 = or disjoint i64 %address.local4278, 99, !dbg !21
  %address.local4462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4461, !dbg !21
  store half %2142, ptr addrspace(1) %address.local4462, align 2, !dbg !21
  %address.local4464 = or disjoint i64 %address.local4278, 112, !dbg !21
  %address.local4465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4464, !dbg !21
  store half %2135, ptr addrspace(1) %address.local4465, align 2, !dbg !21
  %address.local4467 = or disjoint i64 %address.local4278, 113, !dbg !21
  %address.local4468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4467, !dbg !21
  store half %2134, ptr addrspace(1) %address.local4468, align 2, !dbg !21
  %address.local4470 = or disjoint i64 %address.local4278, 114, !dbg !21
  %address.local4471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4470, !dbg !21
  store half %2133, ptr addrspace(1) %address.local4471, align 2, !dbg !21
  %address.local4473 = or i64 %2024, 115, !dbg !21
  %address.local4474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4275, i64 %address.local4473, !dbg !21
  store half %2132, ptr addrspace(1) %address.local4474, align 2, !dbg !21
  %address.local4479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4452, !dbg !21
  store half %2125, ptr addrspace(1) %address.local4479, align 2, !dbg !21
  %address.local4481 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4455, !dbg !21
  store half %2124, ptr addrspace(1) %address.local4481, align 2, !dbg !21
  %address.local4483 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4458, !dbg !21
  store half %2123, ptr addrspace(1) %address.local4483, align 2, !dbg !21
  %address.local4485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4461, !dbg !21
  store half %2122, ptr addrspace(1) %address.local4485, align 2, !dbg !21
  %address.local4487 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4464, !dbg !21
  store half %2115, ptr addrspace(1) %address.local4487, align 2, !dbg !21
  %address.local4489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4467, !dbg !21
  store half %2114, ptr addrspace(1) %address.local4489, align 2, !dbg !21
  %address.local4491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4470, !dbg !21
  store half %2113, ptr addrspace(1) %address.local4491, align 2, !dbg !21
  %address.local4493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local4304, i64 %address.local4473, !dbg !21
  store half %2112, ptr addrspace(1) %address.local4493, align 2, !dbg !21
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_cvt_pack"="rtz" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/shared_memory_plan/causal")
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
!18 = !DILocation(line: 77, column: 19, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
!22 = !DILocation(line: 101, column: 20, scope: !4)
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
!57 = !{i32 23}
