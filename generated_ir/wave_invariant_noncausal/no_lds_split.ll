; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/async_task_noncausal/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/async_task_noncausal/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !30 {
  %5 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !6
  %6 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %5), !dbg !6
  %7 = add nuw nsw i32 %6, 64, !dbg !6
  %8 = xor i32 %6, 16, !dbg !6
  %9 = and i32 %7, 192, !dbg !6
  %10 = icmp samesign ult i32 %8, %9, !dbg !6
  %11 = select i1 %10, i32 %8, i32 %6, !dbg !6
  %12 = shl nuw nsw i32 %11, 2, !dbg !6
  %13 = xor i32 %6, 32, !dbg !6
  %14 = icmp samesign ult i32 %13, %9, !dbg !6
  %15 = select i1 %14, i32 %13, i32 %6, !dbg !6
  %16 = shl nuw nsw i32 %15, 2, !dbg !6
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
  %27 = lshr i64 %22, 5, !dbg !5
  %28 = lshr i64 %22, 10, !dbg !9
  %29 = add nuw nsw i64 %27, %28, !dbg !9
  %30 = and i64 %29, 31, !dbg !9
  %31 = shl nsw i64 %26, 7, !dbg !9
  %32 = mul nsw i64 %28, -4096, !dbg !9
  %33 = add i64 %31, %32, !dbg !9
  %34 = shl nsw i64 %33, 7, !dbg !9
  %35 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !9, !range !32
  %.idx = shl nuw nsw i64 %30, 20, !dbg !9
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %35) #9, !dbg !9, !range !32, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !9
  %address.local237 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local238 = and i64 %address.local237, 1536, !dbg !9
  %address.local239 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local238, !dbg !9
  %address.local240 = ptrtoint ptr addrspace(3) %address.local239 to i64, !dbg !9
  %address.local241 = trunc i64 %address.local240 to i32, !dbg !9
  %36 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local241), !dbg !9
  %address.local242 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local243 = and i64 %address.local242, 16256, !dbg !9
  %address.local244 = lshr i64 %address.local, 4, !dbg !9
  %address.local245 = and i64 %address.local244, 8, !dbg !9
  %address.local246 = or disjoint i64 %address.local243, %address.local245, !dbg !9
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1056 = shl i64 %33, 8, !dbg !9
  %38 = add i64 %.idx, %.idx1056, !dbg !9
  %address.local247.idx = shl nuw nsw i64 %address.local246, 1, !dbg !9
  %39 = or disjoint i64 %38, %address.local247.idx, !dbg !9
  %dma.buffer.offset1054 = trunc i64 %39 to i32, !dbg !9
  %40 = insertelement <2 x i64> poison, i64 %37, i64 0, !dbg !9
  %41 = bitcast <2 x i64> %40 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1055 = shufflevector <4 x i32> %41, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %42 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1054, <4 x i32> %dma.buffer.rsrc1055) #10, !dbg !9
  %address.local248 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local249 = shl nuw nsw i64 %address.local248, 3, !dbg !9
  %address.local250 = and i64 %address.local249, 3584, !dbg !9
  %address.local251 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local250, !dbg !9
  %address.local252 = ptrtoint ptr addrspace(3) %address.local251 to i64, !dbg !9
  %address.local253 = trunc i64 %address.local252 to i32, !dbg !9
  %43 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local253), !dbg !9
  %address.local254 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local255 = and i64 %address.local254, 16256, !dbg !9
  %address.local256 = lshr i64 %address.local248, 4, !dbg !9
  %address.local257 = and i64 %address.local256, 24, !dbg !9
  %address.local258 = or disjoint i64 %address.local255, %address.local257, !dbg !9
  %44 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1057 = shl i64 %33, 8, !dbg !9
  %45 = add i64 %.idx, %.idx1057, !dbg !9
  %address.local259.idx = shl nuw nsw i64 %address.local258, 1, !dbg !9
  %46 = or disjoint i64 %45, %address.local259.idx, !dbg !9
  %dma.buffer.offset1052 = trunc i64 %46 to i32, !dbg !9
  %47 = insertelement <2 x i64> poison, i64 %44, i64 0, !dbg !9
  %48 = bitcast <2 x i64> %47 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1053 = shufflevector <4 x i32> %48, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %49 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset1052, <4 x i32> %dma.buffer.rsrc1053) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %50 = add nuw nsw i64 %27, %28, !dbg !9
  %51 = and i64 %50, 31, !dbg !9
  %52 = shl nuw nsw i64 %51, 19, !dbg !9
  %53 = add i64 %52, %34, !dbg !9
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local238, !dbg !9
  %address.local261 = ptrtoint ptr addrspace(3) %address.local260 to i64, !dbg !9
  %address.local262 = trunc i64 %address.local261 to i32, !dbg !9
  %54 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local262), !dbg !9
  %55 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %56 = or disjoint i64 %53, 32, !dbg !9
  %57 = or disjoint i64 %address.local246, %56, !dbg !9
  %.tr = trunc i64 %57 to i32, !dbg !9
  %dma.buffer.offset1050 = shl i32 %.tr, 1, !dbg !9
  %58 = insertelement <2 x i64> poison, i64 %55, i64 0, !dbg !9
  %59 = bitcast <2 x i64> %58 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1051 = shufflevector <4 x i32> %59, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %60 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %54, i32 %dma.buffer.offset1050, <4 x i32> %dma.buffer.rsrc1051) #10, !dbg !9
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local250, !dbg !9
  %address.local265 = ptrtoint ptr addrspace(3) %address.local264 to i64, !dbg !9
  %address.local266 = trunc i64 %address.local265 to i32, !dbg !9
  %61 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local266), !dbg !9
  %62 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %63 = or disjoint i64 %53, 32, !dbg !9
  %64 = or disjoint i64 %address.local258, %63, !dbg !9
  %.tr1060 = trunc i64 %64 to i32, !dbg !9
  %dma.buffer.offset1048 = shl i32 %.tr1060, 1, !dbg !9
  %65 = insertelement <2 x i64> poison, i64 %62, i64 0, !dbg !9
  %66 = bitcast <2 x i64> %65 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1049 = shufflevector <4 x i32> %66, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %61, i32 %dma.buffer.offset1048, <4 x i32> %dma.buffer.rsrc1049) #10, !dbg !9
  %address.local268 = lshr i32 %address.thread, 2, !dbg !12
  %address.local269 = and i32 %address.local268, 4, !dbg !12
  %address.local270 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local271 = and i32 %address.local270, 384, !dbg !12
  %address.local272 = or disjoint i32 %address.local269, %address.local271, !dbg !12
  %address.local273 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local274 = and i32 %address.local273, 120, !dbg !12
  %address.local275 = or disjoint i32 %address.local272, %address.local274, !dbg !12
  %address.local276 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local277 = and i32 %address.local276, 1024, !dbg !12
  %address.local278 = or disjoint i32 %address.local275, %address.local277, !dbg !12
  %address.local279 = zext nneg i32 %address.local278 to i64, !dbg !12
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local279, !dbg !12
  %68 = load <4 x half>, ptr addrspace(3) %address.local280, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local280, i64 4096, !dbg !12
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %70 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !12
  %lds.const.gep873 = getelementptr i8, ptr addrspace(3) %address.local280, i64 1024, !dbg !12
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep873, align 8, !dbg !12
  %72 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !12
  %lds.const.gep874 = getelementptr i8, ptr addrspace(3) %address.local280, i64 5120, !dbg !12
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep874, align 8, !dbg !12
  %74 = fmul <4 x half> %71, splat (half 0xH3015), !dbg !12
  %75 = fmul <4 x half> %73, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %76 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %77 = or disjoint i64 %53, 64, !dbg !9
  %78 = or disjoint i64 %address.local246, %77, !dbg !9
  %.tr1062 = trunc i64 %78 to i32, !dbg !9
  %dma.buffer.offset1046 = shl i32 %.tr1062, 1, !dbg !9
  %79 = insertelement <2 x i64> poison, i64 %76, i64 0, !dbg !9
  %80 = bitcast <2 x i64> %79 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1047 = shufflevector <4 x i32> %80, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1046, <4 x i32> %dma.buffer.rsrc1047) #10, !dbg !9
  %82 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %83 = or disjoint i64 %53, 64, !dbg !9
  %84 = or disjoint i64 %address.local258, %83, !dbg !9
  %.tr1064 = trunc i64 %84 to i32, !dbg !9
  %dma.buffer.offset1044 = shl i32 %.tr1064, 1, !dbg !9
  %85 = insertelement <2 x i64> poison, i64 %82, i64 0, !dbg !9
  %86 = bitcast <2 x i64> %85 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1045 = shufflevector <4 x i32> %86, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %87 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset1044, <4 x i32> %dma.buffer.rsrc1045) #10, !dbg !9
  %lds.const.gep875 = getelementptr i8, ptr addrspace(3) %address.local280, i64 8192, !dbg !12
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep875, align 8, !dbg !12
  %lds.const.gep876 = getelementptr i8, ptr addrspace(3) %address.local280, i64 12288, !dbg !12
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep876, align 8, !dbg !12
  %lds.const.gep877 = getelementptr i8, ptr addrspace(3) %address.local280, i64 9216, !dbg !12
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep877, align 8, !dbg !12
  %91 = fmul <4 x half> %88, splat (half 0xH3015), !dbg !12
  %lds.const.gep878 = getelementptr i8, ptr addrspace(3) %address.local280, i64 13312, !dbg !12
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep878, align 8, !dbg !12
  %93 = fmul <4 x half> %89, splat (half 0xH3015), !dbg !12
  %94 = fmul <4 x half> %90, splat (half 0xH3015), !dbg !12
  %95 = fmul <4 x half> %92, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %96 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %97 = or disjoint i64 %53, 96, !dbg !9
  %98 = or disjoint i64 %address.local246, %97, !dbg !9
  %.tr1066 = trunc i64 %98 to i32, !dbg !9
  %dma.buffer.offset1042 = shl i32 %.tr1066, 1, !dbg !9
  %99 = insertelement <2 x i64> poison, i64 %96, i64 0, !dbg !9
  %100 = bitcast <2 x i64> %99 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1043 = shufflevector <4 x i32> %100, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %101 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %54, i32 %dma.buffer.offset1042, <4 x i32> %dma.buffer.rsrc1043) #10, !dbg !9
  %102 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %103 = or disjoint i64 %53, 96, !dbg !9
  %104 = or disjoint i64 %address.local258, %103, !dbg !9
  %.tr1068 = trunc i64 %104 to i32, !dbg !9
  %dma.buffer.offset1040 = shl i32 %.tr1068, 1, !dbg !9
  %105 = insertelement <2 x i64> poison, i64 %102, i64 0, !dbg !9
  %106 = bitcast <2 x i64> %105 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1041 = shufflevector <4 x i32> %106, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %107 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %61, i32 %dma.buffer.offset1040, <4 x i32> %dma.buffer.rsrc1041) #10, !dbg !9
  %108 = load <4 x half>, ptr addrspace(3) %address.local280, align 8, !dbg !12
  %lds.const.gep879 = getelementptr i8, ptr addrspace(3) %address.local280, i64 4096, !dbg !12
  %109 = load <4 x half>, ptr addrspace(3) %lds.const.gep879, align 8, !dbg !12
  %lds.const.gep880 = getelementptr i8, ptr addrspace(3) %address.local280, i64 1024, !dbg !12
  %110 = load <4 x half>, ptr addrspace(3) %lds.const.gep880, align 8, !dbg !12
  %111 = fmul <4 x half> %108, splat (half 0xH3015), !dbg !12
  %lds.const.gep881 = getelementptr i8, ptr addrspace(3) %address.local280, i64 5120, !dbg !12
  %112 = load <4 x half>, ptr addrspace(3) %lds.const.gep881, align 8, !dbg !12
  %113 = fmul <4 x half> %109, splat (half 0xH3015), !dbg !12
  %114 = fmul <4 x half> %110, splat (half 0xH3015), !dbg !12
  %115 = fmul <4 x half> %112, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %lds.const.gep882 = getelementptr i8, ptr addrspace(3) %address.local280, i64 8192, !dbg !12
  %116 = load <4 x half>, ptr addrspace(3) %lds.const.gep882, align 8, !dbg !12
  %lds.const.gep883 = getelementptr i8, ptr addrspace(3) %address.local280, i64 12288, !dbg !12
  %117 = load <4 x half>, ptr addrspace(3) %lds.const.gep883, align 8, !dbg !12
  %lds.const.gep884 = getelementptr i8, ptr addrspace(3) %address.local280, i64 9216, !dbg !12
  %118 = load <4 x half>, ptr addrspace(3) %lds.const.gep884, align 8, !dbg !12
  %119 = fmul <4 x half> %116, splat (half 0xH3015), !dbg !12
  %lds.const.gep885 = getelementptr i8, ptr addrspace(3) %address.local280, i64 13312, !dbg !12
  %120 = load <4 x half>, ptr addrspace(3) %lds.const.gep885, align 8, !dbg !12
  %121 = fmul <4 x half> %117, splat (half 0xH3015), !dbg !12
  %122 = fmul <4 x half> %118, splat (half 0xH3015), !dbg !12
  %123 = fmul <4 x half> %120, splat (half 0xH3015), !dbg !12
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  br label %124, !dbg !8

124:                                              ; preds = %133, %4
  %125 = phi i64 [ %1115, %133 ], [ 0, %4 ], !dbg !8
  %126 = phi [2 x <8 x float>] [ %1100, %133 ], [ zeroinitializer, %4 ], !dbg !8
  %127 = phi [2 x <1 x float>] [ %854, %133 ], [ zeroinitializer, %4 ], !dbg !8
  %128 = phi [2 x <1 x float>] [ %1102, %133 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !8
  %129 = phi [2 x <8 x float>] [ %1106, %133 ], [ zeroinitializer, %4 ], !dbg !8
  %130 = phi [2 x <8 x float>] [ %1110, %133 ], [ zeroinitializer, %4 ], !dbg !8
  %131 = phi [2 x <8 x float>] [ %1114, %133 ], [ zeroinitializer, %4 ], !dbg !8
  %132 = icmp samesign ult i64 %125, 32, !dbg !8
  br i1 %132, label %133, label %1116, !dbg !8

133:                                              ; preds = %124
  %134 = shl i64 %27, 19, !dbg !13
  %135 = and i64 %134, 16252928, !dbg !13
  %136 = shl nuw nsw i64 %125, 14, !dbg !13
  %137 = or disjoint i64 %135, %136, !dbg !13
  %address.thread331 = call i32 asm sideeffect "", "=v,0"(i32 %35) #9, !dbg !13, !range !32, !frisk.register_lifetimes !1
  %address.local332 = zext nneg i32 %address.thread331 to i64, !dbg !9
  %address.local333 = shl nuw nsw i64 %address.local332, 7, !dbg !9
  %address.local334 = and i64 %address.local333, 16256, !dbg !9
  %address.local335 = lshr i64 %address.local332, 4, !dbg !9
  %address.local336 = and i64 %address.local335, 8, !dbg !9
  %address.local337 = or disjoint i64 %address.local334, %address.local336, !dbg !9
  %138 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %.idx10691070 = or disjoint i64 %137, %address.local337, !dbg !13
  %.idx10691070.tr = trunc nuw i64 %.idx10691070 to i32, !dbg !13
  %dma.buffer.offset1038 = shl nuw nsw i32 %.idx10691070.tr, 1, !dbg !13
  %139 = insertelement <2 x i64> poison, i64 %138, i64 0, !dbg !13
  %140 = bitcast <2 x i64> %139 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1039 = shufflevector <4 x i32> %140, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %141 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1038, <4 x i32> %dma.buffer.rsrc1039) #10, !dbg !13
  %address.local339 = shl nuw nsw i64 %address.local332, 7, !dbg !9
  %address.local340 = and i64 %address.local339, 16256, !dbg !9
  %address.local341 = or disjoint i64 %address.local332, 256, !dbg !9
  %address.local342 = lshr i64 %address.local341, 4, !dbg !9
  %address.local343 = and i64 %address.local342, 24, !dbg !9
  %address.local344 = or disjoint i64 %address.local340, %address.local343, !dbg !9
  %142 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %.idx10711072 = or disjoint i64 %137, %address.local344, !dbg !13
  %.idx10711072.tr = trunc nuw i64 %.idx10711072 to i32, !dbg !13
  %dma.buffer.offset1036 = shl nuw nsw i32 %.idx10711072.tr, 1, !dbg !13
  %143 = insertelement <2 x i64> poison, i64 %142, i64 0, !dbg !13
  %144 = bitcast <2 x i64> %143 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1037 = shufflevector <4 x i32> %144, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %145 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset1036, <4 x i32> %dma.buffer.rsrc1037) #10, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %146 = or disjoint i64 %137, 32, !dbg !13
  %147 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %.idx10731074 = or disjoint i64 %146, %address.local337, !dbg !13
  %.idx10731074.tr = trunc nuw i64 %.idx10731074 to i32, !dbg !13
  %dma.buffer.offset1034 = shl nuw nsw i32 %.idx10731074.tr, 1, !dbg !13
  %148 = insertelement <2 x i64> poison, i64 %147, i64 0, !dbg !13
  %149 = bitcast <2 x i64> %148 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1035 = shufflevector <4 x i32> %149, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %150 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %54, i32 %dma.buffer.offset1034, <4 x i32> %dma.buffer.rsrc1035) #10, !dbg !13
  %151 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %.idx10751076 = or disjoint i64 %146, %address.local344, !dbg !13
  %.idx10751076.tr = trunc nuw i64 %.idx10751076 to i32, !dbg !13
  %dma.buffer.offset1032 = shl nuw nsw i32 %.idx10751076.tr, 1, !dbg !13
  %152 = insertelement <2 x i64> poison, i64 %151, i64 0, !dbg !13
  %153 = bitcast <2 x i64> %152 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1033 = shufflevector <4 x i32> %153, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %154 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %61, i32 %dma.buffer.offset1032, <4 x i32> %dma.buffer.rsrc1033) #10, !dbg !13
  %address.local348 = lshr i32 %address.thread331, 2, !dbg !15
  %address.local349 = and i32 %address.local348, 4, !dbg !15
  %address.local350 = shl nuw nsw i32 %address.thread331, 1, !dbg !15
  %address.local351 = and i32 %address.local350, 24, !dbg !15
  %address.local352 = or disjoint i32 %address.local349, %address.local351, !dbg !15
  %address.local353 = shl nuw nsw i32 %address.thread331, 5, !dbg !15
  %address.local354 = and i32 %address.local353, 1024, !dbg !15
  %address.local355 = or disjoint i32 %address.local352, %address.local354, !dbg !15
  %address.local356 = shl nuw nsw i32 %address.thread331, 7, !dbg !15
  %address.local357 = and i32 %address.local356, 384, !dbg !15
  %address.local358 = or disjoint i32 %address.local355, %address.local357, !dbg !15
  %address.local359 = zext nneg i32 %address.local358 to i64, !dbg !15
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local359, !dbg !15
  %155 = load <4 x half>, ptr addrspace(3) %address.local360, align 8, !dbg !15
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %155, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep886 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4096, !dbg !15
  %157 = load <4 x half>, ptr addrspace(3) %lds.const.gep886, align 8, !dbg !15
  %lds.const.gep887 = getelementptr i8, ptr addrspace(3) %address.local360, i64 64, !dbg !15
  %158 = load <4 x half>, ptr addrspace(3) %lds.const.gep887, align 8, !dbg !15
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %157, <4 x float> %156) #9, !dbg !15
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %158, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep888 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4160, !dbg !15
  %161 = load <4 x half>, ptr addrspace(3) %lds.const.gep888, align 8, !dbg !15
  %lds.const.gep889 = getelementptr i8, ptr addrspace(3) %address.local360, i64 128, !dbg !15
  %162 = load <4 x half>, ptr addrspace(3) %lds.const.gep889, align 8, !dbg !15
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %161, <4 x float> %160) #9, !dbg !15
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %162, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep890 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4224, !dbg !15
  %165 = load <4 x half>, ptr addrspace(3) %lds.const.gep890, align 8, !dbg !15
  %lds.const.gep891 = getelementptr i8, ptr addrspace(3) %address.local360, i64 192, !dbg !15
  %166 = load <4 x half>, ptr addrspace(3) %lds.const.gep891, align 8, !dbg !15
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %165, <4 x float> %164) #9, !dbg !15
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %166, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep892 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4288, !dbg !15
  %169 = load <4 x half>, ptr addrspace(3) %lds.const.gep892, align 8, !dbg !15
  %lds.const.gep893 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1024, !dbg !15
  %170 = load <4 x half>, ptr addrspace(3) %lds.const.gep893, align 8, !dbg !15
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %169, <4 x float> %168) #9, !dbg !15
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %170, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep894 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5120, !dbg !15
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep894, align 8, !dbg !15
  %lds.const.gep895 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1088, !dbg !15
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep895, align 8, !dbg !15
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %173, <4 x float> %172) #9, !dbg !15
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %174, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep896 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5184, !dbg !15
  %177 = load <4 x half>, ptr addrspace(3) %lds.const.gep896, align 8, !dbg !15
  %lds.const.gep897 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1152, !dbg !15
  %178 = load <4 x half>, ptr addrspace(3) %lds.const.gep897, align 8, !dbg !15
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %177, <4 x float> %176) #9, !dbg !15
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %178, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep898 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5248, !dbg !15
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep898, align 8, !dbg !15
  %lds.const.gep899 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1216, !dbg !15
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep899, align 8, !dbg !15
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %181, <4 x float> %180) #9, !dbg !15
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %182, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep900 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5312, !dbg !15
  %185 = load <4 x half>, ptr addrspace(3) %lds.const.gep900, align 8, !dbg !15
  %186 = load <4 x half>, ptr addrspace(3) %address.local360, align 8, !dbg !15
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %185, <4 x float> %184) #9, !dbg !15
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %186, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep901 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4096, !dbg !15
  %189 = load <4 x half>, ptr addrspace(3) %lds.const.gep901, align 8, !dbg !15
  %lds.const.gep902 = getelementptr i8, ptr addrspace(3) %address.local360, i64 64, !dbg !15
  %190 = load <4 x half>, ptr addrspace(3) %lds.const.gep902, align 8, !dbg !15
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %189, <4 x float> %188) #9, !dbg !15
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %190, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep903 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4160, !dbg !15
  %193 = load <4 x half>, ptr addrspace(3) %lds.const.gep903, align 8, !dbg !15
  %lds.const.gep904 = getelementptr i8, ptr addrspace(3) %address.local360, i64 128, !dbg !15
  %194 = load <4 x half>, ptr addrspace(3) %lds.const.gep904, align 8, !dbg !15
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %193, <4 x float> %192) #9, !dbg !15
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %194, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep905 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4224, !dbg !15
  %197 = load <4 x half>, ptr addrspace(3) %lds.const.gep905, align 8, !dbg !15
  %lds.const.gep906 = getelementptr i8, ptr addrspace(3) %address.local360, i64 192, !dbg !15
  %198 = load <4 x half>, ptr addrspace(3) %lds.const.gep906, align 8, !dbg !15
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %197, <4 x float> %196) #9, !dbg !15
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %198, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep907 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4288, !dbg !15
  %201 = load <4 x half>, ptr addrspace(3) %lds.const.gep907, align 8, !dbg !15
  %lds.const.gep908 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1024, !dbg !15
  %202 = load <4 x half>, ptr addrspace(3) %lds.const.gep908, align 8, !dbg !15
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %201, <4 x float> %200) #9, !dbg !15
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %202, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep909 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5120, !dbg !15
  %205 = load <4 x half>, ptr addrspace(3) %lds.const.gep909, align 8, !dbg !15
  %lds.const.gep910 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1088, !dbg !15
  %206 = load <4 x half>, ptr addrspace(3) %lds.const.gep910, align 8, !dbg !15
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %205, <4 x float> %204) #9, !dbg !15
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %206, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep911 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5184, !dbg !15
  %209 = load <4 x half>, ptr addrspace(3) %lds.const.gep911, align 8, !dbg !15
  %lds.const.gep912 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1152, !dbg !15
  %210 = load <4 x half>, ptr addrspace(3) %lds.const.gep912, align 8, !dbg !15
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %209, <4 x float> %208) #9, !dbg !15
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %210, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep913 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5248, !dbg !15
  %213 = load <4 x half>, ptr addrspace(3) %lds.const.gep913, align 8, !dbg !15
  %lds.const.gep914 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1216, !dbg !15
  %214 = load <4 x half>, ptr addrspace(3) %lds.const.gep914, align 8, !dbg !15
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %213, <4 x float> %212) #9, !dbg !15
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %214, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep915 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5312, !dbg !15
  %217 = load <4 x half>, ptr addrspace(3) %lds.const.gep915, align 8, !dbg !15
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %217, <4 x float> %216) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %219 = or disjoint i64 %137, 64, !dbg !13
  %220 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %.idx10771078 = or disjoint i64 %219, %address.local337, !dbg !13
  %.idx10771078.tr = trunc nuw i64 %.idx10771078 to i32, !dbg !13
  %dma.buffer.offset1030 = shl nuw nsw i32 %.idx10771078.tr, 1, !dbg !13
  %221 = insertelement <2 x i64> poison, i64 %220, i64 0, !dbg !13
  %222 = bitcast <2 x i64> %221 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1031 = shufflevector <4 x i32> %222, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %223 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1030, <4 x i32> %dma.buffer.rsrc1031) #10, !dbg !13
  %224 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %.idx10791080 = or disjoint i64 %219, %address.local344, !dbg !13
  %.idx10791080.tr = trunc nuw i64 %.idx10791080 to i32, !dbg !13
  %dma.buffer.offset1028 = shl nuw nsw i32 %.idx10791080.tr, 1, !dbg !13
  %225 = insertelement <2 x i64> poison, i64 %224, i64 0, !dbg !13
  %226 = bitcast <2 x i64> %225 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1029 = shufflevector <4 x i32> %226, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %227 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset1028, <4 x i32> %dma.buffer.rsrc1029) #10, !dbg !13
  %lds.const.gep916 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8192, !dbg !15
  %228 = load <4 x half>, ptr addrspace(3) %lds.const.gep916, align 8, !dbg !15
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %228, <4 x float> %159) #9, !dbg !15
  %lds.const.gep917 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12288, !dbg !15
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep917, align 8, !dbg !15
  %lds.const.gep918 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8256, !dbg !15
  %231 = load <4 x half>, ptr addrspace(3) %lds.const.gep918, align 8, !dbg !15
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %230, <4 x float> %229) #9, !dbg !15
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %231, <4 x float> %163) #9, !dbg !15
  %lds.const.gep919 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12352, !dbg !15
  %234 = load <4 x half>, ptr addrspace(3) %lds.const.gep919, align 8, !dbg !15
  %lds.const.gep920 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8320, !dbg !15
  %235 = load <4 x half>, ptr addrspace(3) %lds.const.gep920, align 8, !dbg !15
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %234, <4 x float> %233) #9, !dbg !15
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %235, <4 x float> %167) #9, !dbg !15
  %lds.const.gep921 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12416, !dbg !15
  %238 = load <4 x half>, ptr addrspace(3) %lds.const.gep921, align 8, !dbg !15
  %lds.const.gep922 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8384, !dbg !15
  %239 = load <4 x half>, ptr addrspace(3) %lds.const.gep922, align 8, !dbg !15
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %238, <4 x float> %237) #9, !dbg !15
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %239, <4 x float> %171) #9, !dbg !15
  %lds.const.gep923 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12480, !dbg !15
  %242 = load <4 x half>, ptr addrspace(3) %lds.const.gep923, align 8, !dbg !15
  %lds.const.gep924 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9216, !dbg !15
  %243 = load <4 x half>, ptr addrspace(3) %lds.const.gep924, align 8, !dbg !15
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %242, <4 x float> %241) #9, !dbg !15
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %243, <4 x float> %175) #9, !dbg !15
  %lds.const.gep925 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13312, !dbg !15
  %246 = load <4 x half>, ptr addrspace(3) %lds.const.gep925, align 8, !dbg !15
  %lds.const.gep926 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9280, !dbg !15
  %247 = load <4 x half>, ptr addrspace(3) %lds.const.gep926, align 8, !dbg !15
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %246, <4 x float> %245) #9, !dbg !15
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %247, <4 x float> %179) #9, !dbg !15
  %lds.const.gep927 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13376, !dbg !15
  %250 = load <4 x half>, ptr addrspace(3) %lds.const.gep927, align 8, !dbg !15
  %lds.const.gep928 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9344, !dbg !15
  %251 = load <4 x half>, ptr addrspace(3) %lds.const.gep928, align 8, !dbg !15
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %250, <4 x float> %249) #9, !dbg !15
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %251, <4 x float> %183) #9, !dbg !15
  %lds.const.gep929 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13440, !dbg !15
  %254 = load <4 x half>, ptr addrspace(3) %lds.const.gep929, align 8, !dbg !15
  %lds.const.gep930 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9408, !dbg !15
  %255 = load <4 x half>, ptr addrspace(3) %lds.const.gep930, align 8, !dbg !15
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %254, <4 x float> %253) #9, !dbg !15
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %255, <4 x float> %187) #9, !dbg !15
  %lds.const.gep931 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13504, !dbg !15
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep931, align 8, !dbg !15
  %lds.const.gep932 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8192, !dbg !15
  %259 = load <4 x half>, ptr addrspace(3) %lds.const.gep932, align 8, !dbg !15
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %258, <4 x float> %257) #9, !dbg !15
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %259, <4 x float> %191) #9, !dbg !15
  %lds.const.gep933 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12288, !dbg !15
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep933, align 8, !dbg !15
  %lds.const.gep934 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8256, !dbg !15
  %263 = load <4 x half>, ptr addrspace(3) %lds.const.gep934, align 8, !dbg !15
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %262, <4 x float> %261) #9, !dbg !15
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %263, <4 x float> %195) #9, !dbg !15
  %lds.const.gep935 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12352, !dbg !15
  %266 = load <4 x half>, ptr addrspace(3) %lds.const.gep935, align 8, !dbg !15
  %lds.const.gep936 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8320, !dbg !15
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep936, align 8, !dbg !15
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %266, <4 x float> %265) #9, !dbg !15
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %267, <4 x float> %199) #9, !dbg !15
  %lds.const.gep937 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12416, !dbg !15
  %270 = load <4 x half>, ptr addrspace(3) %lds.const.gep937, align 8, !dbg !15
  %lds.const.gep938 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8384, !dbg !15
  %271 = load <4 x half>, ptr addrspace(3) %lds.const.gep938, align 8, !dbg !15
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %270, <4 x float> %269) #9, !dbg !15
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %271, <4 x float> %203) #9, !dbg !15
  %lds.const.gep939 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12480, !dbg !15
  %274 = load <4 x half>, ptr addrspace(3) %lds.const.gep939, align 8, !dbg !15
  %lds.const.gep940 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9216, !dbg !15
  %275 = load <4 x half>, ptr addrspace(3) %lds.const.gep940, align 8, !dbg !15
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %274, <4 x float> %273) #9, !dbg !15
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %275, <4 x float> %207) #9, !dbg !15
  %lds.const.gep941 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13312, !dbg !15
  %278 = load <4 x half>, ptr addrspace(3) %lds.const.gep941, align 8, !dbg !15
  %lds.const.gep942 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9280, !dbg !15
  %279 = load <4 x half>, ptr addrspace(3) %lds.const.gep942, align 8, !dbg !15
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %278, <4 x float> %277) #9, !dbg !15
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %279, <4 x float> %211) #9, !dbg !15
  %lds.const.gep943 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13376, !dbg !15
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep943, align 8, !dbg !15
  %lds.const.gep944 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9344, !dbg !15
  %283 = load <4 x half>, ptr addrspace(3) %lds.const.gep944, align 8, !dbg !15
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %282, <4 x float> %281) #9, !dbg !15
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %283, <4 x float> %215) #9, !dbg !15
  %lds.const.gep945 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13440, !dbg !15
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep945, align 8, !dbg !15
  %lds.const.gep946 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9408, !dbg !15
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep946, align 8, !dbg !15
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %286, <4 x float> %285) #9, !dbg !15
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %287, <4 x float> %218) #9, !dbg !15
  %lds.const.gep947 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13504, !dbg !15
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep947, align 8, !dbg !15
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %290, <4 x float> %289) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %292 = or disjoint i64 %137, 96, !dbg !13
  %293 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %.idx10811082 = or disjoint i64 %292, %address.local337, !dbg !13
  %.idx10811082.tr = trunc nuw i64 %.idx10811082 to i32, !dbg !13
  %dma.buffer.offset1026 = shl nuw nsw i32 %.idx10811082.tr, 1, !dbg !13
  %294 = insertelement <2 x i64> poison, i64 %293, i64 0, !dbg !13
  %295 = bitcast <2 x i64> %294 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1027 = shufflevector <4 x i32> %295, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %296 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %54, i32 %dma.buffer.offset1026, <4 x i32> %dma.buffer.rsrc1027) #10, !dbg !13
  %297 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %.idx10831084 = or disjoint i64 %292, %address.local344, !dbg !13
  %.idx10831084.tr = trunc nuw i64 %.idx10831084 to i32, !dbg !13
  %dma.buffer.offset1024 = shl nuw nsw i32 %.idx10831084.tr, 1, !dbg !13
  %298 = insertelement <2 x i64> poison, i64 %297, i64 0, !dbg !13
  %299 = bitcast <2 x i64> %298 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1025 = shufflevector <4 x i32> %299, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %300 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %61, i32 %dma.buffer.offset1024, <4 x i32> %dma.buffer.rsrc1025) #10, !dbg !13
  %301 = load <4 x half>, ptr addrspace(3) %address.local360, align 8, !dbg !15
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %301, <4 x float> %232) #9, !dbg !15
  %lds.const.gep948 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4096, !dbg !15
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep948, align 8, !dbg !15
  %lds.const.gep949 = getelementptr i8, ptr addrspace(3) %address.local360, i64 64, !dbg !15
  %304 = load <4 x half>, ptr addrspace(3) %lds.const.gep949, align 8, !dbg !15
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %303, <4 x float> %302) #9, !dbg !15
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %304, <4 x float> %236) #9, !dbg !15
  %lds.const.gep950 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4160, !dbg !15
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep950, align 8, !dbg !15
  %lds.const.gep951 = getelementptr i8, ptr addrspace(3) %address.local360, i64 128, !dbg !15
  %308 = load <4 x half>, ptr addrspace(3) %lds.const.gep951, align 8, !dbg !15
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %307, <4 x float> %306) #9, !dbg !15
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %308, <4 x float> %240) #9, !dbg !15
  %lds.const.gep952 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4224, !dbg !15
  %311 = load <4 x half>, ptr addrspace(3) %lds.const.gep952, align 8, !dbg !15
  %lds.const.gep953 = getelementptr i8, ptr addrspace(3) %address.local360, i64 192, !dbg !15
  %312 = load <4 x half>, ptr addrspace(3) %lds.const.gep953, align 8, !dbg !15
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %311, <4 x float> %310) #9, !dbg !15
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %312, <4 x float> %244) #9, !dbg !15
  %lds.const.gep954 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4288, !dbg !15
  %315 = load <4 x half>, ptr addrspace(3) %lds.const.gep954, align 8, !dbg !15
  %lds.const.gep955 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1024, !dbg !15
  %316 = load <4 x half>, ptr addrspace(3) %lds.const.gep955, align 8, !dbg !15
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %315, <4 x float> %314) #9, !dbg !15
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %316, <4 x float> %248) #9, !dbg !15
  %lds.const.gep956 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5120, !dbg !15
  %319 = load <4 x half>, ptr addrspace(3) %lds.const.gep956, align 8, !dbg !15
  %lds.const.gep957 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1088, !dbg !15
  %320 = load <4 x half>, ptr addrspace(3) %lds.const.gep957, align 8, !dbg !15
  %321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %319, <4 x float> %318) #9, !dbg !15
  %322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %320, <4 x float> %252) #9, !dbg !15
  %lds.const.gep958 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5184, !dbg !15
  %323 = load <4 x half>, ptr addrspace(3) %lds.const.gep958, align 8, !dbg !15
  %lds.const.gep959 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1152, !dbg !15
  %324 = load <4 x half>, ptr addrspace(3) %lds.const.gep959, align 8, !dbg !15
  %325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %323, <4 x float> %322) #9, !dbg !15
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %324, <4 x float> %256) #9, !dbg !15
  %lds.const.gep960 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5248, !dbg !15
  %327 = load <4 x half>, ptr addrspace(3) %lds.const.gep960, align 8, !dbg !15
  %lds.const.gep961 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1216, !dbg !15
  %328 = load <4 x half>, ptr addrspace(3) %lds.const.gep961, align 8, !dbg !15
  %329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %327, <4 x float> %326) #9, !dbg !15
  %330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %328, <4 x float> %260) #9, !dbg !15
  %lds.const.gep962 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5312, !dbg !15
  %331 = load <4 x half>, ptr addrspace(3) %lds.const.gep962, align 8, !dbg !15
  %332 = load <4 x half>, ptr addrspace(3) %address.local360, align 8, !dbg !15
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %331, <4 x float> %330) #9, !dbg !15
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %332, <4 x float> %264) #9, !dbg !15
  %lds.const.gep963 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4096, !dbg !15
  %335 = load <4 x half>, ptr addrspace(3) %lds.const.gep963, align 8, !dbg !15
  %lds.const.gep964 = getelementptr i8, ptr addrspace(3) %address.local360, i64 64, !dbg !15
  %336 = load <4 x half>, ptr addrspace(3) %lds.const.gep964, align 8, !dbg !15
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %335, <4 x float> %334) #9, !dbg !15
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %336, <4 x float> %268) #9, !dbg !15
  %lds.const.gep965 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4160, !dbg !15
  %339 = load <4 x half>, ptr addrspace(3) %lds.const.gep965, align 8, !dbg !15
  %lds.const.gep966 = getelementptr i8, ptr addrspace(3) %address.local360, i64 128, !dbg !15
  %340 = load <4 x half>, ptr addrspace(3) %lds.const.gep966, align 8, !dbg !15
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %339, <4 x float> %338) #9, !dbg !15
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %340, <4 x float> %272) #9, !dbg !15
  %lds.const.gep967 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4224, !dbg !15
  %343 = load <4 x half>, ptr addrspace(3) %lds.const.gep967, align 8, !dbg !15
  %lds.const.gep968 = getelementptr i8, ptr addrspace(3) %address.local360, i64 192, !dbg !15
  %344 = load <4 x half>, ptr addrspace(3) %lds.const.gep968, align 8, !dbg !15
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %343, <4 x float> %342) #9, !dbg !15
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %344, <4 x float> %276) #9, !dbg !15
  %lds.const.gep969 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4288, !dbg !15
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep969, align 8, !dbg !15
  %lds.const.gep970 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1024, !dbg !15
  %348 = load <4 x half>, ptr addrspace(3) %lds.const.gep970, align 8, !dbg !15
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %347, <4 x float> %346) #9, !dbg !15
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %348, <4 x float> %280) #9, !dbg !15
  %lds.const.gep971 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5120, !dbg !15
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep971, align 8, !dbg !15
  %lds.const.gep972 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1088, !dbg !15
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep972, align 8, !dbg !15
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %351, <4 x float> %350) #9, !dbg !15
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %352, <4 x float> %284) #9, !dbg !15
  %lds.const.gep973 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5184, !dbg !15
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep973, align 8, !dbg !15
  %lds.const.gep974 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1152, !dbg !15
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep974, align 8, !dbg !15
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %355, <4 x float> %354) #9, !dbg !15
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %356, <4 x float> %288) #9, !dbg !15
  %lds.const.gep975 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5248, !dbg !15
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep975, align 8, !dbg !15
  %lds.const.gep976 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1216, !dbg !15
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep976, align 8, !dbg !15
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %359, <4 x float> %358) #9, !dbg !15
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %360, <4 x float> %291) #9, !dbg !15
  %lds.const.gep977 = getelementptr i8, ptr addrspace(3) %address.local360, i64 5312, !dbg !15
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep977, align 8, !dbg !15
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %363, <4 x float> %362) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local591 = shl nuw nsw i64 %address.local332, 5, !dbg !16
  %address.local592 = and i64 %address.local591, 8064, !dbg !16
  %address.local593 = shl nuw nsw i64 %address.local332, 3, !dbg !16
  %address.local594 = and i64 %address.local593, 24, !dbg !16
  %address.local595 = or disjoint i64 %address.local592, %address.local594, !dbg !16
  %365 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx10851086 = or disjoint i64 %137, %address.local595, !dbg !16
  %.idx10851086.tr = trunc nuw i64 %.idx10851086 to i32, !dbg !16
  %dma.buffer.offset1022 = shl nuw nsw i32 %.idx10851086.tr, 1, !dbg !16
  %366 = insertelement <2 x i64> poison, i64 %365, i64 0, !dbg !16
  %367 = bitcast <2 x i64> %366 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1023 = shufflevector <4 x i32> %367, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %368 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1022, <4 x i32> %dma.buffer.rsrc1023) #10, !dbg !16
  %address.local597 = shl nuw nsw i64 %address.local341, 5, !dbg !16
  %address.local598 = and i64 %address.local597, 16256, !dbg !16
  %address.local599 = shl nuw nsw i64 %address.local332, 3, !dbg !16
  %address.local600 = and i64 %address.local599, 24, !dbg !16
  %address.local601 = or disjoint i64 %address.local598, %address.local600, !dbg !16
  %369 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx10871088 = or disjoint i64 %137, %address.local601, !dbg !16
  %.idx10871088.tr = trunc nuw i64 %.idx10871088 to i32, !dbg !16
  %dma.buffer.offset1020 = shl nuw nsw i32 %.idx10871088.tr, 1, !dbg !16
  %370 = insertelement <2 x i64> poison, i64 %369, i64 0, !dbg !16
  %371 = bitcast <2 x i64> %370 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1021 = shufflevector <4 x i32> %371, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %372 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset1020, <4 x i32> %dma.buffer.rsrc1021) #10, !dbg !16
  %lds.const.gep978 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8192, !dbg !15
  %373 = load <4 x half>, ptr addrspace(3) %lds.const.gep978, align 8, !dbg !15
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %373, <4 x float> %305) #9, !dbg !15
  %lds.const.gep979 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12288, !dbg !15
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep979, align 8, !dbg !15
  %lds.const.gep980 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8256, !dbg !15
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep980, align 8, !dbg !15
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %375, <4 x float> %374) #9, !dbg !15
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %376, <4 x float> %309) #9, !dbg !15
  %379 = extractelement <4 x float> %377, i64 0, !dbg !20
  %380 = call float @llvm.maxnum.f32(float %379, float 0xFFF0000000000000), !dbg !6, !frisk.local_reduction !1
  %381 = extractelement <4 x float> %377, i64 1, !dbg !20
  %382 = extractelement <4 x float> %377, i64 2, !dbg !20
  %383 = extractelement <4 x float> %377, i64 3, !dbg !20
  %lds.const.gep981 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12352, !dbg !15
  %384 = load <4 x half>, ptr addrspace(3) %lds.const.gep981, align 8, !dbg !15
  %lds.const.gep982 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8320, !dbg !15
  %385 = load <4 x half>, ptr addrspace(3) %lds.const.gep982, align 8, !dbg !15
  %386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %384, <4 x float> %378) #9, !dbg !15
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %385, <4 x float> %313) #9, !dbg !15
  %388 = extractelement <4 x float> %386, i64 0, !dbg !20
  %local.reduce116 = call float @llvm.maxnum.f32(float %380, float %388), !dbg !6, !frisk.local_reduction !1
  %389 = extractelement <4 x float> %386, i64 1, !dbg !20
  %local.reduce117 = call float @llvm.maxnum.f32(float %381, float %389), !dbg !6, !frisk.local_reduction !1
  %390 = extractelement <4 x float> %386, i64 2, !dbg !20
  %local.reduce118 = call float @llvm.maxnum.f32(float %382, float %390), !dbg !6, !frisk.local_reduction !1
  %391 = extractelement <4 x float> %386, i64 3, !dbg !20
  %local.reduce119 = call float @llvm.maxnum.f32(float %383, float %391), !dbg !6, !frisk.local_reduction !1
  %lds.const.gep983 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12416, !dbg !15
  %392 = load <4 x half>, ptr addrspace(3) %lds.const.gep983, align 8, !dbg !15
  %lds.const.gep984 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8384, !dbg !15
  %393 = load <4 x half>, ptr addrspace(3) %lds.const.gep984, align 8, !dbg !15
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %392, <4 x float> %387) #9, !dbg !15
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %393, <4 x float> %317) #9, !dbg !15
  %396 = extractelement <4 x float> %394, i64 0, !dbg !20
  %local.reduce120 = call float @llvm.maxnum.f32(float %local.reduce116, float %396), !dbg !6, !frisk.local_reduction !1
  %397 = extractelement <4 x float> %394, i64 1, !dbg !20
  %local.reduce121 = call float @llvm.maxnum.f32(float %local.reduce117, float %397), !dbg !6, !frisk.local_reduction !1
  %398 = extractelement <4 x float> %394, i64 2, !dbg !20
  %local.reduce122 = call float @llvm.maxnum.f32(float %local.reduce118, float %398), !dbg !6, !frisk.local_reduction !1
  %399 = extractelement <4 x float> %394, i64 3, !dbg !20
  %local.reduce123 = call float @llvm.maxnum.f32(float %local.reduce119, float %399), !dbg !6, !frisk.local_reduction !1
  %lds.const.gep985 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12480, !dbg !15
  %400 = load <4 x half>, ptr addrspace(3) %lds.const.gep985, align 8, !dbg !15
  %lds.const.gep986 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9216, !dbg !15
  %401 = load <4 x half>, ptr addrspace(3) %lds.const.gep986, align 8, !dbg !15
  %402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %400, <4 x float> %395) #9, !dbg !15
  %403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %401, <4 x float> %321) #9, !dbg !15
  %404 = extractelement <4 x float> %402, i64 0, !dbg !20
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %404), !dbg !6, !frisk.local_reduction !1
  %405 = extractelement <4 x float> %402, i64 1, !dbg !20
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %405), !dbg !6, !frisk.local_reduction !1
  %406 = extractelement <4 x float> %402, i64 2, !dbg !20
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %406), !dbg !6, !frisk.local_reduction !1
  %407 = extractelement <4 x float> %402, i64 3, !dbg !20
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %407), !dbg !6, !frisk.local_reduction !1
  %lds.const.gep987 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13312, !dbg !15
  %408 = load <4 x half>, ptr addrspace(3) %lds.const.gep987, align 8, !dbg !15
  %lds.const.gep988 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9280, !dbg !15
  %409 = load <4 x half>, ptr addrspace(3) %lds.const.gep988, align 8, !dbg !15
  %410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %408, <4 x float> %403) #9, !dbg !15
  %411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %409, <4 x float> %325) #9, !dbg !15
  %412 = extractelement <4 x float> %410, i64 0, !dbg !20
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %412), !dbg !6, !frisk.local_reduction !1
  %413 = extractelement <4 x float> %410, i64 1, !dbg !20
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %413), !dbg !6, !frisk.local_reduction !1
  %414 = extractelement <4 x float> %410, i64 2, !dbg !20
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %414), !dbg !6, !frisk.local_reduction !1
  %415 = extractelement <4 x float> %410, i64 3, !dbg !20
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %415), !dbg !6, !frisk.local_reduction !1
  %lds.const.gep989 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13376, !dbg !15
  %416 = load <4 x half>, ptr addrspace(3) %lds.const.gep989, align 8, !dbg !15
  %lds.const.gep990 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9344, !dbg !15
  %417 = load <4 x half>, ptr addrspace(3) %lds.const.gep990, align 8, !dbg !15
  %418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %416, <4 x float> %411) #9, !dbg !15
  %419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %417, <4 x float> %329) #9, !dbg !15
  %420 = extractelement <4 x float> %418, i64 0, !dbg !20
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %420), !dbg !6, !frisk.local_reduction !1
  %421 = extractelement <4 x float> %418, i64 1, !dbg !20
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %421), !dbg !6, !frisk.local_reduction !1
  %422 = extractelement <4 x float> %418, i64 2, !dbg !20
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %422), !dbg !6, !frisk.local_reduction !1
  %423 = extractelement <4 x float> %418, i64 3, !dbg !20
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %423), !dbg !6, !frisk.local_reduction !1
  %lds.const.gep991 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13440, !dbg !15
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep991, align 8, !dbg !15
  %lds.const.gep992 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9408, !dbg !15
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep992, align 8, !dbg !15
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %424, <4 x float> %419) #9, !dbg !15
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %425, <4 x float> %333) #9, !dbg !15
  %428 = extractelement <4 x float> %426, i64 0, !dbg !20
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %428), !dbg !6, !frisk.local_reduction !1
  %429 = extractelement <4 x float> %426, i64 1, !dbg !20
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %429), !dbg !6, !frisk.local_reduction !1
  %430 = extractelement <4 x float> %426, i64 2, !dbg !20
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %430), !dbg !6, !frisk.local_reduction !1
  %431 = extractelement <4 x float> %426, i64 3, !dbg !20
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %431), !dbg !6, !frisk.local_reduction !1
  %lds.const.gep993 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13504, !dbg !15
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep993, align 8, !dbg !15
  %lds.const.gep994 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8192, !dbg !15
  %433 = load <4 x half>, ptr addrspace(3) %lds.const.gep994, align 8, !dbg !15
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %432, <4 x float> %427) #9, !dbg !15
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %433, <4 x float> %337) #9, !dbg !15
  %436 = extractelement <4 x float> %434, i64 0, !dbg !20
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %436), !dbg !6, !frisk.local_reduction !1
  %437 = extractelement <4 x float> %434, i64 1, !dbg !20
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %437), !dbg !6, !frisk.local_reduction !1
  %438 = extractelement <4 x float> %434, i64 2, !dbg !20
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %438), !dbg !6, !frisk.local_reduction !1
  %439 = extractelement <4 x float> %434, i64 3, !dbg !20
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %439), !dbg !6, !frisk.local_reduction !1
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %local.reduce141), !dbg !6, !frisk.local_reduction !1
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce142, float %local.reduce143), !dbg !6, !frisk.local_reduction !1
  %440 = call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !6, !frisk.local_reduction !1
  %lds.const.gep995 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12288, !dbg !15
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep995, align 8, !dbg !15
  %lds.const.gep996 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8256, !dbg !15
  %442 = load <4 x half>, ptr addrspace(3) %lds.const.gep996, align 8, !dbg !15
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %441, <4 x float> %435) #9, !dbg !15
  %444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %442, <4 x float> %341) #9, !dbg !15
  %445 = extractelement <4 x float> %443, i64 0, !dbg !20
  %446 = extractelement <4 x float> %443, i64 1, !dbg !20
  %447 = extractelement <4 x float> %443, i64 2, !dbg !20
  %448 = extractelement <4 x float> %443, i64 3, !dbg !20
  %lds.const.gep997 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12352, !dbg !15
  %449 = load <4 x half>, ptr addrspace(3) %lds.const.gep997, align 8, !dbg !15
  %lds.const.gep998 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8320, !dbg !15
  %450 = load <4 x half>, ptr addrspace(3) %lds.const.gep998, align 8, !dbg !15
  %451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %449, <4 x float> %444) #9, !dbg !15
  %452 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %450, <4 x float> %345) #9, !dbg !15
  %453 = extractelement <4 x float> %451, i64 0, !dbg !20
  %454 = extractelement <4 x float> %451, i64 1, !dbg !20
  %455 = extractelement <4 x float> %451, i64 2, !dbg !20
  %456 = extractelement <4 x float> %451, i64 3, !dbg !20
  %lds.const.gep999 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12416, !dbg !15
  %457 = load <4 x half>, ptr addrspace(3) %lds.const.gep999, align 8, !dbg !15
  %lds.const.gep1000 = getelementptr i8, ptr addrspace(3) %address.local360, i64 8384, !dbg !15
  %458 = load <4 x half>, ptr addrspace(3) %lds.const.gep1000, align 8, !dbg !15
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %457, <4 x float> %452) #9, !dbg !15
  %460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %458, <4 x float> %349) #9, !dbg !15
  %461 = extractelement <4 x float> %459, i64 0, !dbg !20
  %462 = extractelement <4 x float> %459, i64 1, !dbg !20
  %463 = extractelement <4 x float> %459, i64 2, !dbg !20
  %464 = extractelement <4 x float> %459, i64 3, !dbg !20
  %lds.const.gep1001 = getelementptr i8, ptr addrspace(3) %address.local360, i64 12480, !dbg !15
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1001, align 8, !dbg !15
  %lds.const.gep1002 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9216, !dbg !15
  %466 = load <4 x half>, ptr addrspace(3) %lds.const.gep1002, align 8, !dbg !15
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %465, <4 x float> %460) #9, !dbg !15
  %468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %466, <4 x float> %353) #9, !dbg !15
  %469 = extractelement <4 x float> %467, i64 0, !dbg !20
  %470 = extractelement <4 x float> %467, i64 1, !dbg !20
  %471 = extractelement <4 x float> %467, i64 2, !dbg !20
  %472 = extractelement <4 x float> %467, i64 3, !dbg !20
  %lds.const.gep1003 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13312, !dbg !15
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1003, align 8, !dbg !15
  %lds.const.gep1004 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9280, !dbg !15
  %474 = load <4 x half>, ptr addrspace(3) %lds.const.gep1004, align 8, !dbg !15
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %473, <4 x float> %468) #9, !dbg !15
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %474, <4 x float> %357) #9, !dbg !15
  %477 = extractelement <4 x float> %475, i64 0, !dbg !20
  %478 = extractelement <4 x float> %475, i64 1, !dbg !20
  %479 = extractelement <4 x float> %475, i64 2, !dbg !20
  %480 = extractelement <4 x float> %475, i64 3, !dbg !20
  %lds.const.gep1005 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13376, !dbg !15
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1005, align 8, !dbg !15
  %lds.const.gep1006 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9344, !dbg !15
  %482 = load <4 x half>, ptr addrspace(3) %lds.const.gep1006, align 8, !dbg !15
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %481, <4 x float> %476) #9, !dbg !15
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %482, <4 x float> %361) #9, !dbg !15
  %485 = extractelement <4 x float> %483, i64 0, !dbg !20
  %486 = extractelement <4 x float> %483, i64 1, !dbg !20
  %487 = extractelement <4 x float> %483, i64 2, !dbg !20
  %488 = extractelement <4 x float> %483, i64 3, !dbg !20
  %lds.const.gep1007 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13440, !dbg !15
  %489 = load <4 x half>, ptr addrspace(3) %lds.const.gep1007, align 8, !dbg !15
  %lds.const.gep1008 = getelementptr i8, ptr addrspace(3) %address.local360, i64 9408, !dbg !15
  %490 = load <4 x half>, ptr addrspace(3) %lds.const.gep1008, align 8, !dbg !15
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %489, <4 x float> %484) #9, !dbg !15
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %490, <4 x float> %364) #9, !dbg !15
  %493 = extractelement <4 x float> %491, i64 0, !dbg !20
  %494 = extractelement <4 x float> %491, i64 1, !dbg !20
  %495 = extractelement <4 x float> %491, i64 2, !dbg !20
  %496 = extractelement <4 x float> %491, i64 3, !dbg !20
  %lds.const.gep1009 = getelementptr i8, ptr addrspace(3) %address.local360, i64 13504, !dbg !15
  %497 = load <4 x half>, ptr addrspace(3) %lds.const.gep1009, align 8, !dbg !15
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %497, <4 x float> %492) #9, !dbg !15
  %499 = extractelement <4 x float> %498, i64 0, !dbg !20
  %500 = extractelement <4 x float> %498, i64 1, !dbg !20
  %501 = extractelement <4 x float> %498, i64 2, !dbg !20
  %502 = extractelement <4 x float> %498, i64 3, !dbg !20
  %503 = bitcast float %440 to i32, !dbg !6
  %504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %503), !dbg !6
  %505 = bitcast i32 %504 to float, !dbg !6
  %506 = call float @llvm.maxnum.f32(float %440, float %505), !dbg !6
  %507 = bitcast float %506 to i32, !dbg !6
  %508 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %507), !dbg !6
  %509 = bitcast i32 %508 to float, !dbg !6
  %510 = call float @llvm.maxnum.f32(float %506, float %509), !dbg !6
  %511 = call float @llvm.maxnum.f32(float %445, float 0xFFF0000000000000), !dbg !6, !frisk.local_reduction !1
  %local.reduce85 = call float @llvm.maxnum.f32(float %511, float %453), !dbg !6, !frisk.local_reduction !1
  %local.reduce86 = call float @llvm.maxnum.f32(float %446, float %454), !dbg !6, !frisk.local_reduction !1
  %local.reduce87 = call float @llvm.maxnum.f32(float %447, float %455), !dbg !6, !frisk.local_reduction !1
  %local.reduce88 = call float @llvm.maxnum.f32(float %448, float %456), !dbg !6, !frisk.local_reduction !1
  %local.reduce89 = call float @llvm.maxnum.f32(float %local.reduce85, float %461), !dbg !6, !frisk.local_reduction !1
  %local.reduce90 = call float @llvm.maxnum.f32(float %local.reduce86, float %462), !dbg !6, !frisk.local_reduction !1
  %local.reduce91 = call float @llvm.maxnum.f32(float %local.reduce87, float %463), !dbg !6, !frisk.local_reduction !1
  %local.reduce92 = call float @llvm.maxnum.f32(float %local.reduce88, float %464), !dbg !6, !frisk.local_reduction !1
  %local.reduce93 = call float @llvm.maxnum.f32(float %local.reduce89, float %469), !dbg !6, !frisk.local_reduction !1
  %local.reduce94 = call float @llvm.maxnum.f32(float %local.reduce90, float %470), !dbg !6, !frisk.local_reduction !1
  %local.reduce95 = call float @llvm.maxnum.f32(float %local.reduce91, float %471), !dbg !6, !frisk.local_reduction !1
  %local.reduce96 = call float @llvm.maxnum.f32(float %local.reduce92, float %472), !dbg !6, !frisk.local_reduction !1
  %local.reduce97 = call float @llvm.maxnum.f32(float %local.reduce93, float %477), !dbg !6, !frisk.local_reduction !1
  %local.reduce98 = call float @llvm.maxnum.f32(float %local.reduce94, float %478), !dbg !6, !frisk.local_reduction !1
  %local.reduce99 = call float @llvm.maxnum.f32(float %local.reduce95, float %479), !dbg !6, !frisk.local_reduction !1
  %local.reduce100 = call float @llvm.maxnum.f32(float %local.reduce96, float %480), !dbg !6, !frisk.local_reduction !1
  %local.reduce101 = call float @llvm.maxnum.f32(float %local.reduce97, float %485), !dbg !6, !frisk.local_reduction !1
  %local.reduce102 = call float @llvm.maxnum.f32(float %local.reduce98, float %486), !dbg !6, !frisk.local_reduction !1
  %local.reduce103 = call float @llvm.maxnum.f32(float %local.reduce99, float %487), !dbg !6, !frisk.local_reduction !1
  %local.reduce104 = call float @llvm.maxnum.f32(float %local.reduce100, float %488), !dbg !6, !frisk.local_reduction !1
  %local.reduce105 = call float @llvm.maxnum.f32(float %local.reduce101, float %493), !dbg !6, !frisk.local_reduction !1
  %local.reduce106 = call float @llvm.maxnum.f32(float %local.reduce102, float %494), !dbg !6, !frisk.local_reduction !1
  %local.reduce107 = call float @llvm.maxnum.f32(float %local.reduce103, float %495), !dbg !6, !frisk.local_reduction !1
  %local.reduce108 = call float @llvm.maxnum.f32(float %local.reduce104, float %496), !dbg !6, !frisk.local_reduction !1
  %local.reduce109 = call float @llvm.maxnum.f32(float %local.reduce105, float %499), !dbg !6, !frisk.local_reduction !1
  %local.reduce110 = call float @llvm.maxnum.f32(float %local.reduce106, float %500), !dbg !6, !frisk.local_reduction !1
  %local.reduce111 = call float @llvm.maxnum.f32(float %local.reduce107, float %501), !dbg !6, !frisk.local_reduction !1
  %local.reduce112 = call float @llvm.maxnum.f32(float %local.reduce108, float %502), !dbg !6, !frisk.local_reduction !1
  %local.reduce113 = call float @llvm.maxnum.f32(float %local.reduce109, float %local.reduce110), !dbg !6, !frisk.local_reduction !1
  %local.reduce114 = call float @llvm.maxnum.f32(float %local.reduce111, float %local.reduce112), !dbg !6, !frisk.local_reduction !1
  %512 = call float @llvm.maxnum.f32(float %local.reduce113, float %local.reduce114), !dbg !6, !frisk.local_reduction !1
  %513 = bitcast float %512 to i32, !dbg !6
  %514 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %513), !dbg !6
  %515 = bitcast i32 %514 to float, !dbg !6
  %516 = call float @llvm.maxnum.f32(float %512, float %515), !dbg !6
  %517 = bitcast float %516 to i32, !dbg !6
  %518 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %517), !dbg !6
  %519 = bitcast i32 %518 to float, !dbg !6
  %520 = call float @llvm.maxnum.f32(float %516, float %519), !dbg !6
  %521 = extractvalue [2 x <1 x float>] %128, 0, !dbg !6
  %522 = insertelement <1 x float> poison, float %510, i64 0, !dbg !6
  %523 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %522, <1 x float> %521), !dbg !6
  %524 = extractvalue [2 x <1 x float>] %128, 1, !dbg !6
  %525 = insertelement <1 x float> poison, float %520, i64 0, !dbg !6
  %526 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %525, <1 x float> %524), !dbg !6
  %527 = shufflevector <1 x float> %523, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %528 = shufflevector <1 x float> %526, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %529 = fsub <1 x float> %521, %523, !dbg !22
  %530 = extractelement <1 x float> %529, i64 0, !dbg !23
  %531 = call float @llvm.amdgcn.exp2.f32(float %530), !dbg !23
  %532 = insertelement <4 x float> poison, float %531, i64 0, !dbg !28
  %533 = shufflevector <4 x float> %532, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %534 = extractvalue [2 x <1 x float>] %127, 0, !dbg !24
  %535 = extractelement <1 x float> %534, i64 0, !dbg !24
  %536 = fmul float %535, %531, !dbg !24
  %537 = fsub <1 x float> %524, %526, !dbg !22
  %538 = extractelement <1 x float> %537, i64 0, !dbg !23
  %539 = call float @llvm.amdgcn.exp2.f32(float %538), !dbg !23
  %540 = insertelement <4 x float> poison, float %539, i64 0, !dbg !28
  %541 = shufflevector <4 x float> %540, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %542 = extractvalue [2 x <1 x float>] %127, 1, !dbg !24
  %543 = extractelement <1 x float> %542, i64 0, !dbg !24
  %544 = fmul float %543, %539, !dbg !24
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local603 = shl nuw nsw i32 %address.thread331, 3, !dbg !14
  %address.local604 = and i32 %address.local603, 120, !dbg !14
  %address.local605 = shl nuw nsw i32 %address.thread331, 5, !dbg !14
  %address.local606 = and i32 %address.local605, 1536, !dbg !14
  %address.local607 = or disjoint i32 %address.local604, %address.local606, !dbg !14
  %address.local608 = zext nneg i32 %address.local607 to i64, !dbg !14
  %address.local609 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local608, !dbg !14
  %545 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local609, i32 0) #0, !dbg !14
  %546 = bitcast <4 x i32> %545 to <8 x half>, !dbg !14
  %address.local610 = and i64 %address.local332, 60, !dbg !14
  %address.local611 = add nuw nsw i64 %address.local610, %address.local332, !dbg !14
  %address.local612 = shl nuw nsw i64 %address.local611, 3, !dbg !14
  %address.local613 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local612, !dbg !14
  %address.local614 = lshr i64 %address.local332, 4, !dbg !14
  %address.local615 = and i64 %address.local614, 3, !dbg !14
  %address.local616 = mul nuw nsw i64 %address.local615, 384, !dbg !14
  %address.local617 = getelementptr [2 x i8], ptr addrspace(3) %address.local613, i64 %address.local616, !dbg !14
  %address.local618 = lshr i64 %address.local332, 2, !dbg !15
  %address.local619 = mul nsw i64 %address.local618, -32, !dbg !14
  %address.local620 = getelementptr [2 x i8], ptr addrspace(3) %address.local617, i64 %address.local619, !dbg !14
  %address.local621 = getelementptr i8, ptr addrspace(3) %address.local620, i64 256, !dbg !14
  %547 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local621, i32 0) #0, !dbg !14
  %548 = bitcast <4 x i32> %547 to <8 x half>, !dbg !14
  %address.local622 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local612, !dbg !14
  %address.local623 = getelementptr [2 x i8], ptr addrspace(3) %address.local622, i64 %address.local616, !dbg !14
  %address.local624 = getelementptr [2 x i8], ptr addrspace(3) %address.local623, i64 %address.local619, !dbg !14
  %address.local625 = getelementptr i8, ptr addrspace(3) %address.local624, i64 512, !dbg !14
  %549 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local625, i32 0) #0, !dbg !14
  %550 = bitcast <4 x i32> %549 to <8 x half>, !dbg !14
  %address.local626 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local612, !dbg !14
  %address.local627 = getelementptr [2 x i8], ptr addrspace(3) %address.local626, i64 %address.local616, !dbg !14
  %address.local628 = getelementptr [2 x i8], ptr addrspace(3) %address.local627, i64 %address.local619, !dbg !14
  %address.local629 = getelementptr i8, ptr addrspace(3) %address.local628, i64 768, !dbg !14
  %551 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local629, i32 0) #0, !dbg !14
  %552 = bitcast <4 x i32> %551 to <8 x half>, !dbg !14
  %address.local630 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local608, !dbg !14
  %address.local631 = getelementptr i8, ptr addrspace(3) %address.local630, i64 4096, !dbg !14
  %553 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local631, i32 0) #0, !dbg !14
  %554 = bitcast <4 x i32> %553 to <8 x half>, !dbg !14
  %address.local632 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local612, !dbg !14
  %address.local633 = getelementptr [2 x i8], ptr addrspace(3) %address.local632, i64 %address.local616, !dbg !14
  %address.local634 = getelementptr [2 x i8], ptr addrspace(3) %address.local633, i64 %address.local619, !dbg !14
  %address.local635 = getelementptr i8, ptr addrspace(3) %address.local634, i64 4352, !dbg !14
  %555 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local635, i32 0) #0, !dbg !14
  %556 = bitcast <4 x i32> %555 to <8 x half>, !dbg !14
  %address.local636 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local612, !dbg !14
  %address.local637 = getelementptr [2 x i8], ptr addrspace(3) %address.local636, i64 %address.local616, !dbg !14
  %address.local638 = getelementptr [2 x i8], ptr addrspace(3) %address.local637, i64 %address.local619, !dbg !14
  %address.local639 = getelementptr i8, ptr addrspace(3) %address.local638, i64 4608, !dbg !14
  %557 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local639, i32 0) #0, !dbg !14
  %558 = bitcast <4 x i32> %557 to <8 x half>, !dbg !14
  %address.local640 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local612, !dbg !14
  %address.local641 = getelementptr [2 x i8], ptr addrspace(3) %address.local640, i64 %address.local616, !dbg !14
  %address.local642 = getelementptr [2 x i8], ptr addrspace(3) %address.local641, i64 %address.local619, !dbg !14
  %address.local643 = getelementptr i8, ptr addrspace(3) %address.local642, i64 4864, !dbg !14
  %559 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local643, i32 0) #0, !dbg !14
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %.idx10891090 = or disjoint i64 %146, %address.local595, !dbg !16
  %.idx10891090.tr = trunc nuw i64 %.idx10891090 to i32, !dbg !16
  %dma.buffer.offset1018 = shl nuw nsw i32 %.idx10891090.tr, 1, !dbg !16
  %560 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %561 = insertelement <2 x i64> poison, i64 %560, i64 0, !dbg !16
  %562 = bitcast <2 x i64> %561 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1019 = shufflevector <4 x i32> %562, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %563 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1018, <4 x i32> %dma.buffer.rsrc1019) #10, !dbg !16
  %.idx10911092 = or disjoint i64 %146, %address.local601, !dbg !16
  %.idx10911092.tr = trunc nuw i64 %.idx10911092 to i32, !dbg !16
  %dma.buffer.offset1016 = shl nuw nsw i32 %.idx10911092.tr, 1, !dbg !16
  %564 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %565 = insertelement <2 x i64> poison, i64 %564, i64 0, !dbg !16
  %566 = bitcast <2 x i64> %565 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1017 = shufflevector <4 x i32> %566, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %567 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset1016, <4 x i32> %dma.buffer.rsrc1017) #10, !dbg !16
  %568 = shufflevector <8 x half> %546, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %569 = extractvalue [2 x <8 x float>] %126, 0, !dbg !28
  %570 = shufflevector <8 x float> %569, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %571 = fmul <4 x float> %570, %533, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %572 = fsub <4 x float> %377, %527, !dbg !21
  %573 = extractelement <4 x float> %572, i64 0, !dbg !25
  %574 = call float @llvm.amdgcn.exp2.f32(float %573), !dbg !25
  %575 = insertelement <4 x float> poison, float %574, i64 0, !dbg !27
  %576 = extractelement <4 x float> %572, i64 1, !dbg !25
  %577 = call float @llvm.amdgcn.exp2.f32(float %576), !dbg !25
  %578 = insertelement <4 x float> %575, float %577, i64 1, !dbg !27
  %579 = extractelement <4 x float> %572, i64 2, !dbg !25
  %580 = call float @llvm.amdgcn.exp2.f32(float %579), !dbg !25
  %581 = insertelement <4 x float> %578, float %580, i64 2, !dbg !27
  %582 = extractelement <4 x float> %572, i64 3, !dbg !25
  %583 = call float @llvm.amdgcn.exp2.f32(float %582), !dbg !25
  %584 = insertelement <4 x float> %581, float %583, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %584) #9, !dbg !27, !frisk.register_lifetimes !1
  %585 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %585) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %568, <4 x float> %571) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %587 = shufflevector <8 x half> %546, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %588 = shufflevector <8 x float> %569, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %589 = fmul <4 x float> %588, %533, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %587, <4 x float> %589) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %591 = extractvalue [2 x <8 x float>] %126, 1, !dbg !28
  %592 = shufflevector <8 x float> %591, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %593 = fmul <4 x float> %592, %541, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %594 = fsub <4 x float> %443, %528, !dbg !21
  %595 = extractelement <4 x float> %594, i64 0, !dbg !25
  %596 = call float @llvm.amdgcn.exp2.f32(float %595), !dbg !25
  %597 = insertelement <4 x float> poison, float %596, i64 0, !dbg !27
  %598 = extractelement <4 x float> %594, i64 1, !dbg !25
  %599 = call float @llvm.amdgcn.exp2.f32(float %598), !dbg !25
  %600 = insertelement <4 x float> %597, float %599, i64 1, !dbg !27
  %601 = extractelement <4 x float> %594, i64 2, !dbg !25
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !25
  %603 = insertelement <4 x float> %600, float %602, i64 2, !dbg !27
  %604 = extractelement <4 x float> %594, i64 3, !dbg !25
  %605 = call float @llvm.amdgcn.exp2.f32(float %604), !dbg !25
  %606 = insertelement <4 x float> %603, float %605, i64 3, !dbg !27
  %math.result221 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %606) #9, !dbg !27, !frisk.register_lifetimes !1
  %607 = fptrunc <4 x float> %math.result221 to <4 x half>, !dbg !27
  %math.packed222 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %607) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %568, <4 x float> %593) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %609 = shufflevector <8 x float> %591, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %610 = fmul <4 x float> %609, %541, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %611 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %587, <4 x float> %610) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %612 = shufflevector <8 x half> %548, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %613 = fsub <4 x float> %386, %527, !dbg !21
  %614 = extractelement <4 x float> %613, i64 0, !dbg !25
  %615 = call float @llvm.amdgcn.exp2.f32(float %614), !dbg !25
  %local.reduce54 = fadd float %574, %615, !dbg !26, !frisk.local_reduction !1
  %616 = insertelement <4 x float> poison, float %615, i64 0, !dbg !27
  %617 = extractelement <4 x float> %613, i64 1, !dbg !25
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !25
  %local.reduce55 = fadd float %577, %618, !dbg !26, !frisk.local_reduction !1
  %619 = insertelement <4 x float> %616, float %618, i64 1, !dbg !27
  %620 = extractelement <4 x float> %613, i64 2, !dbg !25
  %621 = call float @llvm.amdgcn.exp2.f32(float %620), !dbg !25
  %local.reduce56 = fadd float %580, %621, !dbg !26, !frisk.local_reduction !1
  %622 = insertelement <4 x float> %619, float %621, i64 2, !dbg !27
  %623 = extractelement <4 x float> %613, i64 3, !dbg !25
  %624 = call float @llvm.amdgcn.exp2.f32(float %623), !dbg !25
  %local.reduce57 = fadd float %583, %624, !dbg !26, !frisk.local_reduction !1
  %625 = insertelement <4 x float> %622, float %624, i64 3, !dbg !27
  %math.result207 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %625) #9, !dbg !27, !frisk.register_lifetimes !1
  %626 = fptrunc <4 x float> %math.result207 to <4 x half>, !dbg !27
  %math.packed208 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %626) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %627 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed208, <4 x half> %612, <4 x float> %586) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %628 = shufflevector <8 x half> %548, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %629 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed208, <4 x half> %628, <4 x float> %590) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %630 = fsub <4 x float> %451, %528, !dbg !21
  %631 = extractelement <4 x float> %630, i64 0, !dbg !25
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !25
  %local.reduce = fadd float %596, %632, !dbg !26, !frisk.local_reduction !1
  %633 = insertelement <4 x float> poison, float %632, i64 0, !dbg !27
  %634 = extractelement <4 x float> %630, i64 1, !dbg !25
  %635 = call float @llvm.amdgcn.exp2.f32(float %634), !dbg !25
  %local.reduce24 = fadd float %599, %635, !dbg !26, !frisk.local_reduction !1
  %636 = insertelement <4 x float> %633, float %635, i64 1, !dbg !27
  %637 = extractelement <4 x float> %630, i64 2, !dbg !25
  %638 = call float @llvm.amdgcn.exp2.f32(float %637), !dbg !25
  %local.reduce25 = fadd float %602, %638, !dbg !26, !frisk.local_reduction !1
  %639 = insertelement <4 x float> %636, float %638, i64 2, !dbg !27
  %640 = extractelement <4 x float> %630, i64 3, !dbg !25
  %641 = call float @llvm.amdgcn.exp2.f32(float %640), !dbg !25
  %local.reduce26 = fadd float %605, %641, !dbg !26, !frisk.local_reduction !1
  %642 = insertelement <4 x float> %639, float %641, i64 3, !dbg !27
  %math.result223 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %642) #9, !dbg !27, !frisk.register_lifetimes !1
  %643 = fptrunc <4 x float> %math.result223 to <4 x half>, !dbg !27
  %math.packed224 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %643) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %612, <4 x float> %608) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %628, <4 x float> %611) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %646 = shufflevector <8 x half> %550, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %647 = fsub <4 x float> %394, %527, !dbg !21
  %648 = extractelement <4 x float> %647, i64 0, !dbg !25
  %649 = call float @llvm.amdgcn.exp2.f32(float %648), !dbg !25
  %local.reduce58 = fadd float %local.reduce54, %649, !dbg !26, !frisk.local_reduction !1
  %650 = insertelement <4 x float> poison, float %649, i64 0, !dbg !27
  %651 = extractelement <4 x float> %647, i64 1, !dbg !25
  %652 = call float @llvm.amdgcn.exp2.f32(float %651), !dbg !25
  %local.reduce59 = fadd float %local.reduce55, %652, !dbg !26, !frisk.local_reduction !1
  %653 = insertelement <4 x float> %650, float %652, i64 1, !dbg !27
  %654 = extractelement <4 x float> %647, i64 2, !dbg !25
  %655 = call float @llvm.amdgcn.exp2.f32(float %654), !dbg !25
  %local.reduce60 = fadd float %local.reduce56, %655, !dbg !26, !frisk.local_reduction !1
  %656 = insertelement <4 x float> %653, float %655, i64 2, !dbg !27
  %657 = extractelement <4 x float> %647, i64 3, !dbg !25
  %658 = call float @llvm.amdgcn.exp2.f32(float %657), !dbg !25
  %local.reduce61 = fadd float %local.reduce57, %658, !dbg !26, !frisk.local_reduction !1
  %659 = insertelement <4 x float> %656, float %658, i64 3, !dbg !27
  %math.result209 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %659) #9, !dbg !27, !frisk.register_lifetimes !1
  %660 = fptrunc <4 x float> %math.result209 to <4 x half>, !dbg !27
  %math.packed210 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %660) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed210, <4 x half> %646, <4 x float> %627) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %662 = shufflevector <8 x half> %550, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed210, <4 x half> %662, <4 x float> %629) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %664 = fsub <4 x float> %459, %528, !dbg !21
  %665 = extractelement <4 x float> %664, i64 0, !dbg !25
  %666 = call float @llvm.amdgcn.exp2.f32(float %665), !dbg !25
  %local.reduce27 = fadd float %local.reduce, %666, !dbg !26, !frisk.local_reduction !1
  %667 = insertelement <4 x float> poison, float %666, i64 0, !dbg !27
  %668 = extractelement <4 x float> %664, i64 1, !dbg !25
  %669 = call float @llvm.amdgcn.exp2.f32(float %668), !dbg !25
  %local.reduce28 = fadd float %local.reduce24, %669, !dbg !26, !frisk.local_reduction !1
  %670 = insertelement <4 x float> %667, float %669, i64 1, !dbg !27
  %671 = extractelement <4 x float> %664, i64 2, !dbg !25
  %672 = call float @llvm.amdgcn.exp2.f32(float %671), !dbg !25
  %local.reduce29 = fadd float %local.reduce25, %672, !dbg !26, !frisk.local_reduction !1
  %673 = insertelement <4 x float> %670, float %672, i64 2, !dbg !27
  %674 = extractelement <4 x float> %664, i64 3, !dbg !25
  %675 = call float @llvm.amdgcn.exp2.f32(float %674), !dbg !25
  %local.reduce30 = fadd float %local.reduce26, %675, !dbg !26, !frisk.local_reduction !1
  %676 = insertelement <4 x float> %673, float %675, i64 3, !dbg !27
  %math.result225 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %676) #9, !dbg !27, !frisk.register_lifetimes !1
  %677 = fptrunc <4 x float> %math.result225 to <4 x half>, !dbg !27
  %math.packed226 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %677) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %646, <4 x float> %644) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %662, <4 x float> %645) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %680 = shufflevector <8 x half> %552, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %681 = fsub <4 x float> %402, %527, !dbg !21
  %682 = extractelement <4 x float> %681, i64 0, !dbg !25
  %683 = call float @llvm.amdgcn.exp2.f32(float %682), !dbg !25
  %local.reduce62 = fadd float %local.reduce58, %683, !dbg !26, !frisk.local_reduction !1
  %684 = insertelement <4 x float> poison, float %683, i64 0, !dbg !27
  %685 = extractelement <4 x float> %681, i64 1, !dbg !25
  %686 = call float @llvm.amdgcn.exp2.f32(float %685), !dbg !25
  %local.reduce63 = fadd float %local.reduce59, %686, !dbg !26, !frisk.local_reduction !1
  %687 = insertelement <4 x float> %684, float %686, i64 1, !dbg !27
  %688 = extractelement <4 x float> %681, i64 2, !dbg !25
  %689 = call float @llvm.amdgcn.exp2.f32(float %688), !dbg !25
  %local.reduce64 = fadd float %local.reduce60, %689, !dbg !26, !frisk.local_reduction !1
  %690 = insertelement <4 x float> %687, float %689, i64 2, !dbg !27
  %691 = extractelement <4 x float> %681, i64 3, !dbg !25
  %692 = call float @llvm.amdgcn.exp2.f32(float %691), !dbg !25
  %local.reduce65 = fadd float %local.reduce61, %692, !dbg !26, !frisk.local_reduction !1
  %693 = insertelement <4 x float> %690, float %692, i64 3, !dbg !27
  %math.result211 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %693) #9, !dbg !27, !frisk.register_lifetimes !1
  %694 = fptrunc <4 x float> %math.result211 to <4 x half>, !dbg !27
  %math.packed212 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %694) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed212, <4 x half> %680, <4 x float> %661) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %696 = shufflevector <8 x half> %552, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %697 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed212, <4 x half> %696, <4 x float> %663) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %698 = fsub <4 x float> %467, %528, !dbg !21
  %699 = extractelement <4 x float> %698, i64 0, !dbg !25
  %700 = call float @llvm.amdgcn.exp2.f32(float %699), !dbg !25
  %local.reduce31 = fadd float %local.reduce27, %700, !dbg !26, !frisk.local_reduction !1
  %701 = insertelement <4 x float> poison, float %700, i64 0, !dbg !27
  %702 = extractelement <4 x float> %698, i64 1, !dbg !25
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !25
  %local.reduce32 = fadd float %local.reduce28, %703, !dbg !26, !frisk.local_reduction !1
  %704 = insertelement <4 x float> %701, float %703, i64 1, !dbg !27
  %705 = extractelement <4 x float> %698, i64 2, !dbg !25
  %706 = call float @llvm.amdgcn.exp2.f32(float %705), !dbg !25
  %local.reduce33 = fadd float %local.reduce29, %706, !dbg !26, !frisk.local_reduction !1
  %707 = insertelement <4 x float> %704, float %706, i64 2, !dbg !27
  %708 = extractelement <4 x float> %698, i64 3, !dbg !25
  %709 = call float @llvm.amdgcn.exp2.f32(float %708), !dbg !25
  %local.reduce34 = fadd float %local.reduce30, %709, !dbg !26, !frisk.local_reduction !1
  %710 = insertelement <4 x float> %707, float %709, i64 3, !dbg !27
  %math.result227 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %710) #9, !dbg !27, !frisk.register_lifetimes !1
  %711 = fptrunc <4 x float> %math.result227 to <4 x half>, !dbg !27
  %math.packed228 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %711) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %712 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %680, <4 x float> %678) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %713 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %696, <4 x float> %679) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %714 = shufflevector <8 x half> %554, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %715 = fsub <4 x float> %410, %527, !dbg !21
  %716 = extractelement <4 x float> %715, i64 0, !dbg !25
  %717 = call float @llvm.amdgcn.exp2.f32(float %716), !dbg !25
  %local.reduce66 = fadd float %local.reduce62, %717, !dbg !26, !frisk.local_reduction !1
  %718 = insertelement <4 x float> poison, float %717, i64 0, !dbg !27
  %719 = extractelement <4 x float> %715, i64 1, !dbg !25
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !25
  %local.reduce67 = fadd float %local.reduce63, %720, !dbg !26, !frisk.local_reduction !1
  %721 = insertelement <4 x float> %718, float %720, i64 1, !dbg !27
  %722 = extractelement <4 x float> %715, i64 2, !dbg !25
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !25
  %local.reduce68 = fadd float %local.reduce64, %723, !dbg !26, !frisk.local_reduction !1
  %724 = insertelement <4 x float> %721, float %723, i64 2, !dbg !27
  %725 = extractelement <4 x float> %715, i64 3, !dbg !25
  %726 = call float @llvm.amdgcn.exp2.f32(float %725), !dbg !25
  %local.reduce69 = fadd float %local.reduce65, %726, !dbg !26, !frisk.local_reduction !1
  %727 = insertelement <4 x float> %724, float %726, i64 3, !dbg !27
  %math.result213 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %727) #9, !dbg !27, !frisk.register_lifetimes !1
  %728 = fptrunc <4 x float> %math.result213 to <4 x half>, !dbg !27
  %math.packed214 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %728) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed214, <4 x half> %714, <4 x float> %695) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %730 = shufflevector <8 x half> %554, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %731 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed214, <4 x half> %730, <4 x float> %697) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %732 = fsub <4 x float> %475, %528, !dbg !21
  %733 = extractelement <4 x float> %732, i64 0, !dbg !25
  %734 = call float @llvm.amdgcn.exp2.f32(float %733), !dbg !25
  %local.reduce35 = fadd float %local.reduce31, %734, !dbg !26, !frisk.local_reduction !1
  %735 = insertelement <4 x float> poison, float %734, i64 0, !dbg !27
  %736 = extractelement <4 x float> %732, i64 1, !dbg !25
  %737 = call float @llvm.amdgcn.exp2.f32(float %736), !dbg !25
  %local.reduce36 = fadd float %local.reduce32, %737, !dbg !26, !frisk.local_reduction !1
  %738 = insertelement <4 x float> %735, float %737, i64 1, !dbg !27
  %739 = extractelement <4 x float> %732, i64 2, !dbg !25
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !25
  %local.reduce37 = fadd float %local.reduce33, %740, !dbg !26, !frisk.local_reduction !1
  %741 = insertelement <4 x float> %738, float %740, i64 2, !dbg !27
  %742 = extractelement <4 x float> %732, i64 3, !dbg !25
  %743 = call float @llvm.amdgcn.exp2.f32(float %742), !dbg !25
  %local.reduce38 = fadd float %local.reduce34, %743, !dbg !26, !frisk.local_reduction !1
  %744 = insertelement <4 x float> %741, float %743, i64 3, !dbg !27
  %math.result229 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %744) #9, !dbg !27, !frisk.register_lifetimes !1
  %745 = fptrunc <4 x float> %math.result229 to <4 x half>, !dbg !27
  %math.packed230 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %745) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %746 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %714, <4 x float> %712) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %747 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %730, <4 x float> %713) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %748 = shufflevector <8 x half> %556, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %749 = fsub <4 x float> %418, %527, !dbg !21
  %750 = extractelement <4 x float> %749, i64 0, !dbg !25
  %751 = call float @llvm.amdgcn.exp2.f32(float %750), !dbg !25
  %local.reduce70 = fadd float %local.reduce66, %751, !dbg !26, !frisk.local_reduction !1
  %752 = insertelement <4 x float> poison, float %751, i64 0, !dbg !27
  %753 = extractelement <4 x float> %749, i64 1, !dbg !25
  %754 = call float @llvm.amdgcn.exp2.f32(float %753), !dbg !25
  %local.reduce71 = fadd float %local.reduce67, %754, !dbg !26, !frisk.local_reduction !1
  %755 = insertelement <4 x float> %752, float %754, i64 1, !dbg !27
  %756 = extractelement <4 x float> %749, i64 2, !dbg !25
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !25
  %local.reduce72 = fadd float %local.reduce68, %757, !dbg !26, !frisk.local_reduction !1
  %758 = insertelement <4 x float> %755, float %757, i64 2, !dbg !27
  %759 = extractelement <4 x float> %749, i64 3, !dbg !25
  %760 = call float @llvm.amdgcn.exp2.f32(float %759), !dbg !25
  %local.reduce73 = fadd float %local.reduce69, %760, !dbg !26, !frisk.local_reduction !1
  %761 = insertelement <4 x float> %758, float %760, i64 3, !dbg !27
  %math.result215 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %761) #9, !dbg !27, !frisk.register_lifetimes !1
  %762 = fptrunc <4 x float> %math.result215 to <4 x half>, !dbg !27
  %math.packed216 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %762) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %763 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed216, <4 x half> %748, <4 x float> %729) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %764 = shufflevector <8 x half> %556, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %765 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed216, <4 x half> %764, <4 x float> %731) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %766 = fsub <4 x float> %483, %528, !dbg !21
  %767 = extractelement <4 x float> %766, i64 0, !dbg !25
  %768 = call float @llvm.amdgcn.exp2.f32(float %767), !dbg !25
  %local.reduce39 = fadd float %local.reduce35, %768, !dbg !26, !frisk.local_reduction !1
  %769 = insertelement <4 x float> poison, float %768, i64 0, !dbg !27
  %770 = extractelement <4 x float> %766, i64 1, !dbg !25
  %771 = call float @llvm.amdgcn.exp2.f32(float %770), !dbg !25
  %local.reduce40 = fadd float %local.reduce36, %771, !dbg !26, !frisk.local_reduction !1
  %772 = insertelement <4 x float> %769, float %771, i64 1, !dbg !27
  %773 = extractelement <4 x float> %766, i64 2, !dbg !25
  %774 = call float @llvm.amdgcn.exp2.f32(float %773), !dbg !25
  %local.reduce41 = fadd float %local.reduce37, %774, !dbg !26, !frisk.local_reduction !1
  %775 = insertelement <4 x float> %772, float %774, i64 2, !dbg !27
  %776 = extractelement <4 x float> %766, i64 3, !dbg !25
  %777 = call float @llvm.amdgcn.exp2.f32(float %776), !dbg !25
  %local.reduce42 = fadd float %local.reduce38, %777, !dbg !26, !frisk.local_reduction !1
  %778 = insertelement <4 x float> %775, float %777, i64 3, !dbg !27
  %math.result231 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %778) #9, !dbg !27, !frisk.register_lifetimes !1
  %779 = fptrunc <4 x float> %math.result231 to <4 x half>, !dbg !27
  %math.packed232 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %779) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %780 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %748, <4 x float> %746) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %781 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %764, <4 x float> %747) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %782 = shufflevector <8 x half> %558, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %783 = fsub <4 x float> %426, %527, !dbg !21
  %784 = extractelement <4 x float> %783, i64 0, !dbg !25
  %785 = call float @llvm.amdgcn.exp2.f32(float %784), !dbg !25
  %local.reduce74 = fadd float %local.reduce70, %785, !dbg !26, !frisk.local_reduction !1
  %786 = insertelement <4 x float> poison, float %785, i64 0, !dbg !27
  %787 = extractelement <4 x float> %783, i64 1, !dbg !25
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !25
  %local.reduce75 = fadd float %local.reduce71, %788, !dbg !26, !frisk.local_reduction !1
  %789 = insertelement <4 x float> %786, float %788, i64 1, !dbg !27
  %790 = extractelement <4 x float> %783, i64 2, !dbg !25
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !25
  %local.reduce76 = fadd float %local.reduce72, %791, !dbg !26, !frisk.local_reduction !1
  %792 = insertelement <4 x float> %789, float %791, i64 2, !dbg !27
  %793 = extractelement <4 x float> %783, i64 3, !dbg !25
  %794 = call float @llvm.amdgcn.exp2.f32(float %793), !dbg !25
  %local.reduce77 = fadd float %local.reduce73, %794, !dbg !26, !frisk.local_reduction !1
  %795 = insertelement <4 x float> %792, float %794, i64 3, !dbg !27
  %math.result217 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %795) #9, !dbg !27, !frisk.register_lifetimes !1
  %796 = fptrunc <4 x float> %math.result217 to <4 x half>, !dbg !27
  %math.packed218 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %796) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %797 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed218, <4 x half> %782, <4 x float> %763) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %798 = shufflevector <8 x half> %558, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %799 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed218, <4 x half> %798, <4 x float> %765) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %800 = fsub <4 x float> %491, %528, !dbg !21
  %801 = extractelement <4 x float> %800, i64 0, !dbg !25
  %802 = call float @llvm.amdgcn.exp2.f32(float %801), !dbg !25
  %local.reduce43 = fadd float %local.reduce39, %802, !dbg !26, !frisk.local_reduction !1
  %803 = insertelement <4 x float> poison, float %802, i64 0, !dbg !27
  %804 = extractelement <4 x float> %800, i64 1, !dbg !25
  %805 = call float @llvm.amdgcn.exp2.f32(float %804), !dbg !25
  %local.reduce44 = fadd float %local.reduce40, %805, !dbg !26, !frisk.local_reduction !1
  %806 = insertelement <4 x float> %803, float %805, i64 1, !dbg !27
  %807 = extractelement <4 x float> %800, i64 2, !dbg !25
  %808 = call float @llvm.amdgcn.exp2.f32(float %807), !dbg !25
  %local.reduce45 = fadd float %local.reduce41, %808, !dbg !26, !frisk.local_reduction !1
  %809 = insertelement <4 x float> %806, float %808, i64 2, !dbg !27
  %810 = extractelement <4 x float> %800, i64 3, !dbg !25
  %811 = call float @llvm.amdgcn.exp2.f32(float %810), !dbg !25
  %local.reduce46 = fadd float %local.reduce42, %811, !dbg !26, !frisk.local_reduction !1
  %812 = insertelement <4 x float> %809, float %811, i64 3, !dbg !27
  %math.result233 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %812) #9, !dbg !27, !frisk.register_lifetimes !1
  %813 = fptrunc <4 x float> %math.result233 to <4 x half>, !dbg !27
  %math.packed234 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %813) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %782, <4 x float> %780) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %815 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %798, <4 x float> %781) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %816 = bitcast <4 x i32> %559 to <8 x half>, !dbg !14
  %817 = shufflevector <8 x half> %816, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %818 = fsub <4 x float> %434, %527, !dbg !21
  %819 = extractelement <4 x float> %818, i64 0, !dbg !25
  %820 = call float @llvm.amdgcn.exp2.f32(float %819), !dbg !25
  %local.reduce78 = fadd float %local.reduce74, %820, !dbg !26, !frisk.local_reduction !1
  %821 = insertelement <4 x float> poison, float %820, i64 0, !dbg !27
  %822 = extractelement <4 x float> %818, i64 1, !dbg !25
  %823 = call float @llvm.amdgcn.exp2.f32(float %822), !dbg !25
  %local.reduce79 = fadd float %local.reduce75, %823, !dbg !26, !frisk.local_reduction !1
  %local.reduce82 = fadd float %local.reduce78, %local.reduce79, !dbg !26, !frisk.local_reduction !1
  %824 = insertelement <4 x float> %821, float %823, i64 1, !dbg !27
  %825 = extractelement <4 x float> %818, i64 2, !dbg !25
  %826 = call float @llvm.amdgcn.exp2.f32(float %825), !dbg !25
  %local.reduce80 = fadd float %local.reduce76, %826, !dbg !26, !frisk.local_reduction !1
  %827 = insertelement <4 x float> %824, float %826, i64 2, !dbg !27
  %828 = extractelement <4 x float> %818, i64 3, !dbg !25
  %829 = call float @llvm.amdgcn.exp2.f32(float %828), !dbg !25
  %local.reduce81 = fadd float %local.reduce77, %829, !dbg !26, !frisk.local_reduction !1
  %local.reduce83 = fadd float %local.reduce80, %local.reduce81, !dbg !26, !frisk.local_reduction !1
  %830 = fadd float %local.reduce82, %local.reduce83, !dbg !26, !frisk.local_reduction !1
  %831 = fadd float %830, %536, !dbg !26
  %832 = insertelement <1 x float> poison, float %831, i64 0, !dbg !8
  %833 = insertvalue [2 x <1 x float>] poison, <1 x float> %832, 0, !dbg !8
  %834 = insertelement <4 x float> %827, float %829, i64 3, !dbg !27
  %math.result219 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %834) #9, !dbg !27, !frisk.register_lifetimes !1
  %835 = fptrunc <4 x float> %math.result219 to <4 x half>, !dbg !27
  %math.packed220 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %835) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %836 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %817, <4 x float> %797) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %837 = shufflevector <8 x half> %816, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %838 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %837, <4 x float> %799) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %839 = fsub <4 x float> %498, %528, !dbg !21
  %840 = extractelement <4 x float> %839, i64 0, !dbg !25
  %841 = call float @llvm.amdgcn.exp2.f32(float %840), !dbg !25
  %local.reduce47 = fadd float %local.reduce43, %841, !dbg !26, !frisk.local_reduction !1
  %842 = insertelement <4 x float> poison, float %841, i64 0, !dbg !27
  %843 = extractelement <4 x float> %839, i64 1, !dbg !25
  %844 = call float @llvm.amdgcn.exp2.f32(float %843), !dbg !25
  %local.reduce48 = fadd float %local.reduce44, %844, !dbg !26, !frisk.local_reduction !1
  %local.reduce51 = fadd float %local.reduce47, %local.reduce48, !dbg !26, !frisk.local_reduction !1
  %845 = insertelement <4 x float> %842, float %844, i64 1, !dbg !27
  %846 = extractelement <4 x float> %839, i64 2, !dbg !25
  %847 = call float @llvm.amdgcn.exp2.f32(float %846), !dbg !25
  %local.reduce49 = fadd float %local.reduce45, %847, !dbg !26, !frisk.local_reduction !1
  %848 = insertelement <4 x float> %845, float %847, i64 2, !dbg !27
  %849 = extractelement <4 x float> %839, i64 3, !dbg !25
  %850 = call float @llvm.amdgcn.exp2.f32(float %849), !dbg !25
  %local.reduce50 = fadd float %local.reduce46, %850, !dbg !26, !frisk.local_reduction !1
  %local.reduce52 = fadd float %local.reduce49, %local.reduce50, !dbg !26, !frisk.local_reduction !1
  %851 = fadd float %local.reduce51, %local.reduce52, !dbg !26, !frisk.local_reduction !1
  %852 = fadd float %851, %544, !dbg !26
  %853 = insertelement <1 x float> poison, float %852, i64 0, !dbg !8
  %854 = insertvalue [2 x <1 x float>] %833, <1 x float> %853, 1, !dbg !8
  %855 = insertelement <4 x float> %848, float %850, i64 3, !dbg !27
  %math.result235 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %855) #9, !dbg !27, !frisk.register_lifetimes !1
  %856 = fptrunc <4 x float> %math.result235 to <4 x half>, !dbg !27
  %math.packed236 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %856) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %857 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %817, <4 x float> %814) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %837, <4 x float> %815) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %859 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local609, i32 0) #0, !dbg !14
  %860 = bitcast <4 x i32> %859 to <8 x half>, !dbg !14
  %861 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local621, i32 0) #0, !dbg !14
  %862 = bitcast <4 x i32> %861 to <8 x half>, !dbg !14
  %863 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local625, i32 0) #0, !dbg !14
  %864 = bitcast <4 x i32> %863 to <8 x half>, !dbg !14
  %865 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local629, i32 0) #0, !dbg !14
  %866 = bitcast <4 x i32> %865 to <8 x half>, !dbg !14
  %867 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local631, i32 0) #0, !dbg !14
  %868 = bitcast <4 x i32> %867 to <8 x half>, !dbg !14
  %869 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local635, i32 0) #0, !dbg !14
  %870 = bitcast <4 x i32> %869 to <8 x half>, !dbg !14
  %871 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local639, i32 0) #0, !dbg !14
  %872 = bitcast <4 x i32> %871 to <8 x half>, !dbg !14
  %873 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local643, i32 0) #0, !dbg !14
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %.idx10931094 = or disjoint i64 %219, %address.local595, !dbg !16
  %.idx10931094.tr = trunc nuw i64 %.idx10931094 to i32, !dbg !16
  %dma.buffer.offset1014 = shl nuw nsw i32 %.idx10931094.tr, 1, !dbg !16
  %874 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %875 = insertelement <2 x i64> poison, i64 %874, i64 0, !dbg !16
  %876 = bitcast <2 x i64> %875 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1015 = shufflevector <4 x i32> %876, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %877 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1014, <4 x i32> %dma.buffer.rsrc1015) #10, !dbg !16
  %.idx10951096 = or disjoint i64 %219, %address.local601, !dbg !16
  %.idx10951096.tr = trunc nuw i64 %.idx10951096 to i32, !dbg !16
  %dma.buffer.offset1012 = shl nuw nsw i32 %.idx10951096.tr, 1, !dbg !16
  %878 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %879 = insertelement <2 x i64> poison, i64 %878, i64 0, !dbg !16
  %880 = bitcast <2 x i64> %879 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1013 = shufflevector <4 x i32> %880, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %881 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset1012, <4 x i32> %dma.buffer.rsrc1013) #10, !dbg !16
  %882 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %883 = extractvalue [2 x <8 x float>] %129, 0, !dbg !28
  %884 = shufflevector <8 x float> %883, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %885 = fmul <4 x float> %884, %533, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %882, <4 x float> %885) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %887 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %888 = shufflevector <8 x float> %883, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %889 = fmul <4 x float> %888, %533, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %887, <4 x float> %889) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %891 = extractvalue [2 x <8 x float>] %129, 1, !dbg !28
  %892 = shufflevector <8 x float> %891, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %893 = fmul <4 x float> %892, %541, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %894 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %882, <4 x float> %893) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %895 = shufflevector <8 x float> %891, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %896 = fmul <4 x float> %895, %541, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %887, <4 x float> %896) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %898 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed208, <4 x half> %898, <4 x float> %886) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %900 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed208, <4 x half> %900, <4 x float> %890) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %898, <4 x float> %894) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %900, <4 x float> %897) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %904 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed210, <4 x half> %904, <4 x float> %899) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %906 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed210, <4 x half> %906, <4 x float> %901) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %904, <4 x float> %902) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %906, <4 x float> %903) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %910 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed212, <4 x half> %910, <4 x float> %905) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %912 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed212, <4 x half> %912, <4 x float> %907) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %910, <4 x float> %908) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %912, <4 x float> %909) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %916 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed214, <4 x half> %916, <4 x float> %911) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %918 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed214, <4 x half> %918, <4 x float> %913) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %916, <4 x float> %914) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %918, <4 x float> %915) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %922 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed216, <4 x half> %922, <4 x float> %917) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %924 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed216, <4 x half> %924, <4 x float> %919) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %922, <4 x float> %920) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %924, <4 x float> %921) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %928 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed218, <4 x half> %928, <4 x float> %923) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %930 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed218, <4 x half> %930, <4 x float> %925) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %928, <4 x float> %926) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %930, <4 x float> %927) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %934 = bitcast <4 x i32> %873 to <8 x half>, !dbg !14
  %935 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %935, <4 x float> %929) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %937 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %937, <4 x float> %931) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %935, <4 x float> %932) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %937, <4 x float> %933) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %941 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local609, i32 0) #0, !dbg !14
  %942 = bitcast <4 x i32> %941 to <8 x half>, !dbg !14
  %943 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local621, i32 0) #0, !dbg !14
  %944 = bitcast <4 x i32> %943 to <8 x half>, !dbg !14
  %945 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local625, i32 0) #0, !dbg !14
  %946 = bitcast <4 x i32> %945 to <8 x half>, !dbg !14
  %947 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local629, i32 0) #0, !dbg !14
  %948 = bitcast <4 x i32> %947 to <8 x half>, !dbg !14
  %949 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local631, i32 0) #0, !dbg !14
  %950 = bitcast <4 x i32> %949 to <8 x half>, !dbg !14
  %951 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local635, i32 0) #0, !dbg !14
  %952 = bitcast <4 x i32> %951 to <8 x half>, !dbg !14
  %953 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local639, i32 0) #0, !dbg !14
  %954 = bitcast <4 x i32> %953 to <8 x half>, !dbg !14
  %955 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local643, i32 0) #0, !dbg !14
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %.idx10971098 = or disjoint i64 %292, %address.local595, !dbg !16
  %.idx10971098.tr = trunc nuw i64 %.idx10971098 to i32, !dbg !16
  %dma.buffer.offset1010 = shl nuw nsw i32 %.idx10971098.tr, 1, !dbg !16
  %956 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %957 = insertelement <2 x i64> poison, i64 %956, i64 0, !dbg !16
  %958 = bitcast <2 x i64> %957 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1011 = shufflevector <4 x i32> %958, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %959 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1010, <4 x i32> %dma.buffer.rsrc1011) #10, !dbg !16
  %.idx10991100 = or disjoint i64 %292, %address.local601, !dbg !16
  %.idx10991100.tr = trunc nuw i64 %.idx10991100 to i32, !dbg !16
  %dma.buffer.offset = shl nuw nsw i32 %.idx10991100.tr, 1, !dbg !16
  %960 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %961 = insertelement <2 x i64> poison, i64 %960, i64 0, !dbg !16
  %962 = bitcast <2 x i64> %961 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc = shufflevector <4 x i32> %962, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %963 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !16
  %964 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %965 = extractvalue [2 x <8 x float>] %130, 0, !dbg !28
  %966 = shufflevector <8 x float> %965, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %967 = fmul <4 x float> %966, %533, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %964, <4 x float> %967) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %969 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %970 = shufflevector <8 x float> %965, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %971 = fmul <4 x float> %970, %533, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %969, <4 x float> %971) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %973 = extractvalue [2 x <8 x float>] %130, 1, !dbg !28
  %974 = shufflevector <8 x float> %973, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %975 = fmul <4 x float> %974, %541, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %964, <4 x float> %975) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %977 = shufflevector <8 x float> %973, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %978 = fmul <4 x float> %977, %541, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %969, <4 x float> %978) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %980 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed208, <4 x half> %980, <4 x float> %968) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %982 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed208, <4 x half> %982, <4 x float> %972) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %980, <4 x float> %976) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %982, <4 x float> %979) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %986 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed210, <4 x half> %986, <4 x float> %981) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %988 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed210, <4 x half> %988, <4 x float> %983) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %986, <4 x float> %984) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %988, <4 x float> %985) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %992 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed212, <4 x half> %992, <4 x float> %987) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %994 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed212, <4 x half> %994, <4 x float> %989) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %992, <4 x float> %990) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %994, <4 x float> %991) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %998 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed214, <4 x half> %998, <4 x float> %993) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1000 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed214, <4 x half> %1000, <4 x float> %995) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %998, <4 x float> %996) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1000, <4 x float> %997) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1004 = shufflevector <8 x half> %952, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1005 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed216, <4 x half> %1004, <4 x float> %999) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1006 = shufflevector <8 x half> %952, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1007 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed216, <4 x half> %1006, <4 x float> %1001) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1004, <4 x float> %1002) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1009 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1006, <4 x float> %1003) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1010 = shufflevector <8 x half> %954, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1011 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed218, <4 x half> %1010, <4 x float> %1005) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1012 = shufflevector <8 x half> %954, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1013 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed218, <4 x half> %1012, <4 x float> %1007) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1010, <4 x float> %1008) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1012, <4 x float> %1009) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1016 = bitcast <4 x i32> %955 to <8 x half>, !dbg !14
  %1017 = shufflevector <8 x half> %1016, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %1017, <4 x float> %1011) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1019 = shufflevector <8 x half> %1016, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %1019, <4 x float> %1013) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1017, <4 x float> %1014) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1019, <4 x float> %1015) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1023 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local609, i32 0) #0, !dbg !14
  %1024 = bitcast <4 x i32> %1023 to <8 x half>, !dbg !14
  %1025 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local621, i32 0) #0, !dbg !14
  %1026 = bitcast <4 x i32> %1025 to <8 x half>, !dbg !14
  %1027 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local625, i32 0) #0, !dbg !14
  %1028 = bitcast <4 x i32> %1027 to <8 x half>, !dbg !14
  %1029 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local629, i32 0) #0, !dbg !14
  %1030 = bitcast <4 x i32> %1029 to <8 x half>, !dbg !14
  %1031 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local631, i32 0) #0, !dbg !14
  %1032 = bitcast <4 x i32> %1031 to <8 x half>, !dbg !14
  %1033 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local635, i32 0) #0, !dbg !14
  %1034 = bitcast <4 x i32> %1033 to <8 x half>, !dbg !14
  %1035 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local639, i32 0) #0, !dbg !14
  %1036 = bitcast <4 x i32> %1035 to <8 x half>, !dbg !14
  %1037 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local643, i32 0) #0, !dbg !14
  %1038 = shufflevector <8 x half> %1024, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1039 = extractvalue [2 x <8 x float>] %131, 0, !dbg !28
  %1040 = shufflevector <8 x float> %1039, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1041 = fmul <4 x float> %1040, %533, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1038, <4 x float> %1041) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1043 = shufflevector <8 x half> %1024, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1044 = shufflevector <8 x float> %1039, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1045 = fmul <4 x float> %1044, %533, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1043, <4 x float> %1045) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1047 = extractvalue [2 x <8 x float>] %131, 1, !dbg !28
  %1048 = shufflevector <8 x float> %1047, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1049 = fmul <4 x float> %1048, %541, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %1038, <4 x float> %1049) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1051 = shufflevector <8 x float> %1047, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1052 = fmul <4 x float> %1051, %541, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %1043, <4 x float> %1052) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1054 = shufflevector <8 x half> %1026, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed208, <4 x half> %1054, <4 x float> %1042) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1056 = shufflevector <8 x half> %1026, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed208, <4 x half> %1056, <4 x float> %1046) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %1054, <4 x float> %1050) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %1056, <4 x float> %1053) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1060 = shufflevector <8 x half> %1028, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed210, <4 x half> %1060, <4 x float> %1055) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1062 = shufflevector <8 x half> %1028, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed210, <4 x half> %1062, <4 x float> %1057) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %1060, <4 x float> %1058) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %1062, <4 x float> %1059) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1066 = shufflevector <8 x half> %1030, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed212, <4 x half> %1066, <4 x float> %1061) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1068 = shufflevector <8 x half> %1030, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed212, <4 x half> %1068, <4 x float> %1063) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1066, <4 x float> %1064) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1071 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1068, <4 x float> %1065) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1072 = shufflevector <8 x half> %1032, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed214, <4 x half> %1072, <4 x float> %1067) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1074 = shufflevector <8 x half> %1032, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed214, <4 x half> %1074, <4 x float> %1069) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1072, <4 x float> %1070) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1074, <4 x float> %1071) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1078 = shufflevector <8 x half> %1034, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed216, <4 x half> %1078, <4 x float> %1073) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1080 = shufflevector <8 x half> %1034, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed216, <4 x half> %1080, <4 x float> %1075) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1078, <4 x float> %1076) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1080, <4 x float> %1077) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1084 = shufflevector <8 x half> %1036, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1085 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed218, <4 x half> %1084, <4 x float> %1079) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1086 = shufflevector <8 x half> %1036, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed218, <4 x half> %1086, <4 x float> %1081) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1084, <4 x float> %1082) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1089 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1086, <4 x float> %1083) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1090 = bitcast <4 x i32> %1037 to <8 x half>, !dbg !14
  %1091 = shufflevector <8 x half> %1090, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %1091, <4 x float> %1085) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1093 = shufflevector <8 x half> %1090, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %1093, <4 x float> %1087) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1095 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1091, <4 x float> %1088) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1093, <4 x float> %1089) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %1097 = shufflevector <4 x float> %836, <4 x float> %838, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1098 = insertvalue [2 x <8 x float>] poison, <8 x float> %1097, 0, !dbg !8
  %1099 = shufflevector <4 x float> %857, <4 x float> %858, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1100 = insertvalue [2 x <8 x float>] %1098, <8 x float> %1099, 1, !dbg !8
  %1101 = insertvalue [2 x <1 x float>] poison, <1 x float> %523, 0, !dbg !8
  %1102 = insertvalue [2 x <1 x float>] %1101, <1 x float> %526, 1, !dbg !8
  %1103 = shufflevector <4 x float> %936, <4 x float> %938, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1104 = insertvalue [2 x <8 x float>] poison, <8 x float> %1103, 0, !dbg !8
  %1105 = shufflevector <4 x float> %939, <4 x float> %940, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1106 = insertvalue [2 x <8 x float>] %1104, <8 x float> %1105, 1, !dbg !8
  %1107 = shufflevector <4 x float> %1018, <4 x float> %1020, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1108 = insertvalue [2 x <8 x float>] poison, <8 x float> %1107, 0, !dbg !8
  %1109 = shufflevector <4 x float> %1021, <4 x float> %1022, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1110 = insertvalue [2 x <8 x float>] %1108, <8 x float> %1109, 1, !dbg !8
  %1111 = shufflevector <4 x float> %1092, <4 x float> %1094, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1112 = insertvalue [2 x <8 x float>] poison, <8 x float> %1111, 0, !dbg !8
  %1113 = shufflevector <4 x float> %1095, <4 x float> %1096, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1114 = insertvalue [2 x <8 x float>] %1112, <8 x float> %1113, 1, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %1115 = add nuw nsw i64 %125, 1, !dbg !8
  br label %124, !dbg !8

1116:                                             ; preds = %124
  %lane.sum.field149 = extractvalue [2 x <1 x float>] %127, 1, !dbg !8
  %lane.sum.local150 = extractelement <1 x float> %lane.sum.field149, i64 0, !dbg !8
  %lane.sum.lane.lo151 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %lane.sum.lane152 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo151), !dbg !8
  %1117 = shl nuw nsw i32 %lane.sum.lane152, 2, !dbg !8
  %1118 = xor i32 %1117, 64, !dbg !8
  %1119 = bitcast float %lane.sum.local150 to i32, !dbg !8
  %lane.sum.shuffle153 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1118, i32 %1119), !dbg !8
  %1120 = bitcast i32 %lane.sum.shuffle153 to float, !dbg !8
  %lane.sum.final154 = fadd float %lane.sum.local150, %1120, !dbg !8
  %1121 = shl nuw nsw i32 %lane.sum.lane152, 2, !dbg !8
  %1122 = xor i32 %1121, 128, !dbg !8
  %1123 = bitcast float %lane.sum.final154 to i32, !dbg !8
  %lane.sum.shuffle155 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1122, i32 %1123), !dbg !8
  %1124 = bitcast i32 %lane.sum.shuffle155 to float, !dbg !8
  %lane.sum.final156 = fadd float %lane.sum.final154, %1124, !dbg !8
  %1125 = insertelement <1 x float> poison, float %lane.sum.final156, i64 0, !dbg !8
  %lane.sum.field = extractvalue [2 x <1 x float>] %127, 0, !dbg !8
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !8
  %lane.sum.lane.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %lane.sum.lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo), !dbg !8
  %1126 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !8
  %1127 = xor i32 %1126, 64, !dbg !8
  %1128 = bitcast float %lane.sum.local to i32, !dbg !8
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1127, i32 %1128), !dbg !8
  %1129 = bitcast i32 %lane.sum.shuffle to float, !dbg !8
  %lane.sum.final = fadd float %lane.sum.local, %1129, !dbg !8
  %1130 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !8
  %1131 = xor i32 %1130, 128, !dbg !8
  %1132 = bitcast float %lane.sum.final to i32, !dbg !8
  %lane.sum.shuffle147 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1131, i32 %1132), !dbg !8
  %1133 = bitcast i32 %lane.sum.shuffle147 to float, !dbg !8
  %lane.sum.final148 = fadd float %lane.sum.final, %1133, !dbg !8
  %1134 = insertelement <1 x float> poison, float %lane.sum.final148, i64 0, !dbg !8
  %1135 = and i32 %35, 16, !dbg !8
  %.not = icmp eq i32 %1135, 0, !dbg !8
  %1136 = and i32 %35, 32, !dbg !8
  %.not1 = icmp eq i32 %1136, 0, !dbg !8
  %1137 = extractvalue [2 x <8 x float>] %131, 0, !dbg !8
  %1138 = extractelement <8 x float> %1137, i64 0, !dbg !8
  %1139 = extractelement <8 x float> %1137, i64 1, !dbg !8
  %1140 = extractelement <8 x float> %1137, i64 2, !dbg !8
  %1141 = extractelement <8 x float> %1137, i64 3, !dbg !8
  %1142 = select i1 %.not, float %1139, float %1138, !dbg !8
  %1143 = bitcast float %1142 to i32, !dbg !8
  %1144 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1143), !dbg !8
  %1145 = bitcast i32 %1144 to float, !dbg !8
  %1146 = select i1 %.not, float %1138, float %1145, !dbg !8
  %1147 = select i1 %.not, float %1145, float %1139, !dbg !8
  %1148 = select i1 %.not, float %1141, float %1140, !dbg !8
  %1149 = bitcast float %1148 to i32, !dbg !8
  %1150 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1149), !dbg !8
  %1151 = bitcast i32 %1150 to float, !dbg !8
  %1152 = select i1 %.not, float %1140, float %1151, !dbg !8
  %1153 = select i1 %.not, float %1151, float %1141, !dbg !8
  %1154 = select i1 %.not1, float %1152, float %1146, !dbg !8
  %1155 = bitcast float %1154 to i32, !dbg !8
  %1156 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1155), !dbg !8
  %1157 = bitcast i32 %1156 to float, !dbg !8
  %1158 = select i1 %.not1, float %1146, float %1157, !dbg !8
  %1159 = select i1 %.not1, float %1157, float %1152, !dbg !8
  %1160 = select i1 %.not1, float %1153, float %1147, !dbg !8
  %1161 = bitcast float %1160 to i32, !dbg !8
  %1162 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1161), !dbg !8
  %1163 = bitcast i32 %1162 to float, !dbg !8
  %1164 = select i1 %.not1, float %1147, float %1163, !dbg !8
  %1165 = select i1 %.not1, float %1163, float %1153, !dbg !8
  %1166 = extractelement <8 x float> %1137, i64 4, !dbg !8
  %1167 = extractelement <8 x float> %1137, i64 5, !dbg !8
  %1168 = extractelement <8 x float> %1137, i64 6, !dbg !8
  %1169 = extractelement <8 x float> %1137, i64 7, !dbg !8
  %1170 = select i1 %.not, float %1167, float %1166, !dbg !8
  %1171 = bitcast float %1170 to i32, !dbg !8
  %1172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1171), !dbg !8
  %1173 = bitcast i32 %1172 to float, !dbg !8
  %1174 = select i1 %.not, float %1166, float %1173, !dbg !8
  %1175 = select i1 %.not, float %1173, float %1167, !dbg !8
  %1176 = select i1 %.not, float %1169, float %1168, !dbg !8
  %1177 = bitcast float %1176 to i32, !dbg !8
  %1178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1177), !dbg !8
  %1179 = bitcast i32 %1178 to float, !dbg !8
  %1180 = select i1 %.not, float %1168, float %1179, !dbg !8
  %1181 = select i1 %.not, float %1179, float %1169, !dbg !8
  %1182 = select i1 %.not1, float %1180, float %1174, !dbg !8
  %1183 = bitcast float %1182 to i32, !dbg !8
  %1184 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1183), !dbg !8
  %1185 = bitcast i32 %1184 to float, !dbg !8
  %1186 = select i1 %.not1, float %1174, float %1185, !dbg !8
  %1187 = select i1 %.not1, float %1185, float %1180, !dbg !8
  %1188 = select i1 %.not1, float %1181, float %1175, !dbg !8
  %1189 = bitcast float %1188 to i32, !dbg !8
  %1190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1189), !dbg !8
  %1191 = bitcast i32 %1190 to float, !dbg !8
  %1192 = select i1 %.not1, float %1175, float %1191, !dbg !8
  %1193 = select i1 %.not1, float %1191, float %1181, !dbg !8
  %1194 = extractvalue [2 x <8 x float>] %131, 1, !dbg !8
  %1195 = extractelement <8 x float> %1194, i64 0, !dbg !8
  %1196 = extractelement <8 x float> %1194, i64 1, !dbg !8
  %1197 = extractelement <8 x float> %1194, i64 2, !dbg !8
  %1198 = extractelement <8 x float> %1194, i64 3, !dbg !8
  %1199 = select i1 %.not, float %1196, float %1195, !dbg !8
  %1200 = bitcast float %1199 to i32, !dbg !8
  %1201 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1200), !dbg !8
  %1202 = bitcast i32 %1201 to float, !dbg !8
  %1203 = select i1 %.not, float %1195, float %1202, !dbg !8
  %1204 = select i1 %.not, float %1202, float %1196, !dbg !8
  %1205 = select i1 %.not, float %1198, float %1197, !dbg !8
  %1206 = bitcast float %1205 to i32, !dbg !8
  %1207 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1206), !dbg !8
  %1208 = bitcast i32 %1207 to float, !dbg !8
  %1209 = select i1 %.not, float %1197, float %1208, !dbg !8
  %1210 = select i1 %.not, float %1208, float %1198, !dbg !8
  %1211 = select i1 %.not1, float %1209, float %1203, !dbg !8
  %1212 = bitcast float %1211 to i32, !dbg !8
  %1213 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1212), !dbg !8
  %1214 = bitcast i32 %1213 to float, !dbg !8
  %1215 = select i1 %.not1, float %1203, float %1214, !dbg !8
  %1216 = select i1 %.not1, float %1214, float %1209, !dbg !8
  %1217 = select i1 %.not1, float %1210, float %1204, !dbg !8
  %1218 = bitcast float %1217 to i32, !dbg !8
  %1219 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1218), !dbg !8
  %1220 = bitcast i32 %1219 to float, !dbg !8
  %1221 = select i1 %.not1, float %1204, float %1220, !dbg !8
  %1222 = select i1 %.not1, float %1220, float %1210, !dbg !8
  %1223 = extractelement <8 x float> %1194, i64 4, !dbg !8
  %1224 = extractelement <8 x float> %1194, i64 5, !dbg !8
  %1225 = extractelement <8 x float> %1194, i64 6, !dbg !8
  %1226 = extractelement <8 x float> %1194, i64 7, !dbg !8
  %1227 = select i1 %.not, float %1224, float %1223, !dbg !8
  %1228 = bitcast float %1227 to i32, !dbg !8
  %1229 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1228), !dbg !8
  %1230 = bitcast i32 %1229 to float, !dbg !8
  %1231 = select i1 %.not, float %1223, float %1230, !dbg !8
  %1232 = select i1 %.not, float %1230, float %1224, !dbg !8
  %1233 = select i1 %.not, float %1226, float %1225, !dbg !8
  %1234 = bitcast float %1233 to i32, !dbg !8
  %1235 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1234), !dbg !8
  %1236 = bitcast i32 %1235 to float, !dbg !8
  %1237 = select i1 %.not, float %1225, float %1236, !dbg !8
  %1238 = select i1 %.not, float %1236, float %1226, !dbg !8
  %1239 = select i1 %.not1, float %1237, float %1231, !dbg !8
  %1240 = bitcast float %1239 to i32, !dbg !8
  %1241 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1240), !dbg !8
  %1242 = bitcast i32 %1241 to float, !dbg !8
  %1243 = select i1 %.not1, float %1231, float %1242, !dbg !8
  %1244 = select i1 %.not1, float %1242, float %1237, !dbg !8
  %1245 = select i1 %.not1, float %1238, float %1232, !dbg !8
  %1246 = bitcast float %1245 to i32, !dbg !8
  %1247 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1246), !dbg !8
  %1248 = bitcast i32 %1247 to float, !dbg !8
  %1249 = select i1 %.not1, float %1232, float %1248, !dbg !8
  %1250 = select i1 %.not1, float %1248, float %1238, !dbg !8
  %1251 = extractvalue [2 x <8 x float>] %130, 0, !dbg !8
  %1252 = extractelement <8 x float> %1251, i64 0, !dbg !8
  %1253 = extractelement <8 x float> %1251, i64 1, !dbg !8
  %1254 = extractelement <8 x float> %1251, i64 2, !dbg !8
  %1255 = extractelement <8 x float> %1251, i64 3, !dbg !8
  %1256 = select i1 %.not, float %1253, float %1252, !dbg !8
  %1257 = bitcast float %1256 to i32, !dbg !8
  %1258 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1257), !dbg !8
  %1259 = bitcast i32 %1258 to float, !dbg !8
  %1260 = select i1 %.not, float %1252, float %1259, !dbg !8
  %1261 = select i1 %.not, float %1259, float %1253, !dbg !8
  %1262 = select i1 %.not, float %1255, float %1254, !dbg !8
  %1263 = bitcast float %1262 to i32, !dbg !8
  %1264 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1263), !dbg !8
  %1265 = bitcast i32 %1264 to float, !dbg !8
  %1266 = select i1 %.not, float %1254, float %1265, !dbg !8
  %1267 = select i1 %.not, float %1265, float %1255, !dbg !8
  %1268 = select i1 %.not1, float %1266, float %1260, !dbg !8
  %1269 = bitcast float %1268 to i32, !dbg !8
  %1270 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1269), !dbg !8
  %1271 = bitcast i32 %1270 to float, !dbg !8
  %1272 = select i1 %.not1, float %1260, float %1271, !dbg !8
  %1273 = select i1 %.not1, float %1271, float %1266, !dbg !8
  %1274 = select i1 %.not1, float %1267, float %1261, !dbg !8
  %1275 = bitcast float %1274 to i32, !dbg !8
  %1276 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1275), !dbg !8
  %1277 = bitcast i32 %1276 to float, !dbg !8
  %1278 = select i1 %.not1, float %1261, float %1277, !dbg !8
  %1279 = select i1 %.not1, float %1277, float %1267, !dbg !8
  %1280 = extractelement <8 x float> %1251, i64 4, !dbg !8
  %1281 = extractelement <8 x float> %1251, i64 5, !dbg !8
  %1282 = extractelement <8 x float> %1251, i64 6, !dbg !8
  %1283 = extractelement <8 x float> %1251, i64 7, !dbg !8
  %1284 = select i1 %.not, float %1281, float %1280, !dbg !8
  %1285 = bitcast float %1284 to i32, !dbg !8
  %1286 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1285), !dbg !8
  %1287 = bitcast i32 %1286 to float, !dbg !8
  %1288 = select i1 %.not, float %1280, float %1287, !dbg !8
  %1289 = select i1 %.not, float %1287, float %1281, !dbg !8
  %1290 = select i1 %.not, float %1283, float %1282, !dbg !8
  %1291 = bitcast float %1290 to i32, !dbg !8
  %1292 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1291), !dbg !8
  %1293 = bitcast i32 %1292 to float, !dbg !8
  %1294 = select i1 %.not, float %1282, float %1293, !dbg !8
  %1295 = select i1 %.not, float %1293, float %1283, !dbg !8
  %1296 = select i1 %.not1, float %1294, float %1288, !dbg !8
  %1297 = bitcast float %1296 to i32, !dbg !8
  %1298 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1297), !dbg !8
  %1299 = bitcast i32 %1298 to float, !dbg !8
  %1300 = select i1 %.not1, float %1288, float %1299, !dbg !8
  %1301 = select i1 %.not1, float %1299, float %1294, !dbg !8
  %1302 = select i1 %.not1, float %1295, float %1289, !dbg !8
  %1303 = bitcast float %1302 to i32, !dbg !8
  %1304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1303), !dbg !8
  %1305 = bitcast i32 %1304 to float, !dbg !8
  %1306 = select i1 %.not1, float %1289, float %1305, !dbg !8
  %1307 = select i1 %.not1, float %1305, float %1295, !dbg !8
  %1308 = extractvalue [2 x <8 x float>] %130, 1, !dbg !8
  %1309 = extractelement <8 x float> %1308, i64 0, !dbg !8
  %1310 = extractelement <8 x float> %1308, i64 1, !dbg !8
  %1311 = extractelement <8 x float> %1308, i64 2, !dbg !8
  %1312 = extractelement <8 x float> %1308, i64 3, !dbg !8
  %1313 = select i1 %.not, float %1310, float %1309, !dbg !8
  %1314 = bitcast float %1313 to i32, !dbg !8
  %1315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1314), !dbg !8
  %1316 = bitcast i32 %1315 to float, !dbg !8
  %1317 = select i1 %.not, float %1309, float %1316, !dbg !8
  %1318 = select i1 %.not, float %1316, float %1310, !dbg !8
  %1319 = select i1 %.not, float %1312, float %1311, !dbg !8
  %1320 = bitcast float %1319 to i32, !dbg !8
  %1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1320), !dbg !8
  %1322 = bitcast i32 %1321 to float, !dbg !8
  %1323 = select i1 %.not, float %1311, float %1322, !dbg !8
  %1324 = select i1 %.not, float %1322, float %1312, !dbg !8
  %1325 = select i1 %.not1, float %1323, float %1317, !dbg !8
  %1326 = bitcast float %1325 to i32, !dbg !8
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1326), !dbg !8
  %1328 = bitcast i32 %1327 to float, !dbg !8
  %1329 = select i1 %.not1, float %1317, float %1328, !dbg !8
  %1330 = select i1 %.not1, float %1328, float %1323, !dbg !8
  %1331 = select i1 %.not1, float %1324, float %1318, !dbg !8
  %1332 = bitcast float %1331 to i32, !dbg !8
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1332), !dbg !8
  %1334 = bitcast i32 %1333 to float, !dbg !8
  %1335 = select i1 %.not1, float %1318, float %1334, !dbg !8
  %1336 = select i1 %.not1, float %1334, float %1324, !dbg !8
  %1337 = extractelement <8 x float> %1308, i64 4, !dbg !8
  %1338 = extractelement <8 x float> %1308, i64 5, !dbg !8
  %1339 = extractelement <8 x float> %1308, i64 6, !dbg !8
  %1340 = extractelement <8 x float> %1308, i64 7, !dbg !8
  %1341 = select i1 %.not, float %1338, float %1337, !dbg !8
  %1342 = bitcast float %1341 to i32, !dbg !8
  %1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1342), !dbg !8
  %1344 = bitcast i32 %1343 to float, !dbg !8
  %1345 = select i1 %.not, float %1337, float %1344, !dbg !8
  %1346 = select i1 %.not, float %1344, float %1338, !dbg !8
  %1347 = select i1 %.not, float %1340, float %1339, !dbg !8
  %1348 = bitcast float %1347 to i32, !dbg !8
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1348), !dbg !8
  %1350 = bitcast i32 %1349 to float, !dbg !8
  %1351 = select i1 %.not, float %1339, float %1350, !dbg !8
  %1352 = select i1 %.not, float %1350, float %1340, !dbg !8
  %1353 = select i1 %.not1, float %1351, float %1345, !dbg !8
  %1354 = bitcast float %1353 to i32, !dbg !8
  %1355 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1354), !dbg !8
  %1356 = bitcast i32 %1355 to float, !dbg !8
  %1357 = select i1 %.not1, float %1345, float %1356, !dbg !8
  %1358 = select i1 %.not1, float %1356, float %1351, !dbg !8
  %1359 = select i1 %.not1, float %1352, float %1346, !dbg !8
  %1360 = bitcast float %1359 to i32, !dbg !8
  %1361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1360), !dbg !8
  %1362 = bitcast i32 %1361 to float, !dbg !8
  %1363 = select i1 %.not1, float %1346, float %1362, !dbg !8
  %1364 = select i1 %.not1, float %1362, float %1352, !dbg !8
  %1365 = extractvalue [2 x <8 x float>] %129, 0, !dbg !8
  %1366 = extractelement <8 x float> %1365, i64 0, !dbg !8
  %1367 = extractelement <8 x float> %1365, i64 1, !dbg !8
  %1368 = extractelement <8 x float> %1365, i64 2, !dbg !8
  %1369 = extractelement <8 x float> %1365, i64 3, !dbg !8
  %1370 = select i1 %.not, float %1367, float %1366, !dbg !8
  %1371 = bitcast float %1370 to i32, !dbg !8
  %1372 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1371), !dbg !8
  %1373 = bitcast i32 %1372 to float, !dbg !8
  %1374 = select i1 %.not, float %1366, float %1373, !dbg !8
  %1375 = select i1 %.not, float %1373, float %1367, !dbg !8
  %1376 = select i1 %.not, float %1369, float %1368, !dbg !8
  %1377 = bitcast float %1376 to i32, !dbg !8
  %1378 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1377), !dbg !8
  %1379 = bitcast i32 %1378 to float, !dbg !8
  %1380 = select i1 %.not, float %1368, float %1379, !dbg !8
  %1381 = select i1 %.not, float %1379, float %1369, !dbg !8
  %1382 = select i1 %.not1, float %1380, float %1374, !dbg !8
  %1383 = bitcast float %1382 to i32, !dbg !8
  %1384 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1383), !dbg !8
  %1385 = bitcast i32 %1384 to float, !dbg !8
  %1386 = select i1 %.not1, float %1374, float %1385, !dbg !8
  %1387 = select i1 %.not1, float %1385, float %1380, !dbg !8
  %1388 = select i1 %.not1, float %1381, float %1375, !dbg !8
  %1389 = bitcast float %1388 to i32, !dbg !8
  %1390 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1389), !dbg !8
  %1391 = bitcast i32 %1390 to float, !dbg !8
  %1392 = select i1 %.not1, float %1375, float %1391, !dbg !8
  %1393 = select i1 %.not1, float %1391, float %1381, !dbg !8
  %1394 = extractelement <8 x float> %1365, i64 4, !dbg !8
  %1395 = extractelement <8 x float> %1365, i64 5, !dbg !8
  %1396 = extractelement <8 x float> %1365, i64 6, !dbg !8
  %1397 = extractelement <8 x float> %1365, i64 7, !dbg !8
  %1398 = select i1 %.not, float %1395, float %1394, !dbg !8
  %1399 = bitcast float %1398 to i32, !dbg !8
  %1400 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1399), !dbg !8
  %1401 = bitcast i32 %1400 to float, !dbg !8
  %1402 = select i1 %.not, float %1394, float %1401, !dbg !8
  %1403 = select i1 %.not, float %1401, float %1395, !dbg !8
  %1404 = select i1 %.not, float %1397, float %1396, !dbg !8
  %1405 = bitcast float %1404 to i32, !dbg !8
  %1406 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1405), !dbg !8
  %1407 = bitcast i32 %1406 to float, !dbg !8
  %1408 = select i1 %.not, float %1396, float %1407, !dbg !8
  %1409 = select i1 %.not, float %1407, float %1397, !dbg !8
  %1410 = select i1 %.not1, float %1408, float %1402, !dbg !8
  %1411 = bitcast float %1410 to i32, !dbg !8
  %1412 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1411), !dbg !8
  %1413 = bitcast i32 %1412 to float, !dbg !8
  %1414 = select i1 %.not1, float %1402, float %1413, !dbg !8
  %1415 = select i1 %.not1, float %1413, float %1408, !dbg !8
  %1416 = select i1 %.not1, float %1409, float %1403, !dbg !8
  %1417 = bitcast float %1416 to i32, !dbg !8
  %1418 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1417), !dbg !8
  %1419 = bitcast i32 %1418 to float, !dbg !8
  %1420 = select i1 %.not1, float %1403, float %1419, !dbg !8
  %1421 = select i1 %.not1, float %1419, float %1409, !dbg !8
  %1422 = extractvalue [2 x <8 x float>] %129, 1, !dbg !8
  %1423 = extractelement <8 x float> %1422, i64 0, !dbg !8
  %1424 = extractelement <8 x float> %1422, i64 1, !dbg !8
  %1425 = extractelement <8 x float> %1422, i64 2, !dbg !8
  %1426 = extractelement <8 x float> %1422, i64 3, !dbg !8
  %1427 = select i1 %.not, float %1424, float %1423, !dbg !8
  %1428 = bitcast float %1427 to i32, !dbg !8
  %1429 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1428), !dbg !8
  %1430 = bitcast i32 %1429 to float, !dbg !8
  %1431 = select i1 %.not, float %1423, float %1430, !dbg !8
  %1432 = select i1 %.not, float %1430, float %1424, !dbg !8
  %1433 = select i1 %.not, float %1426, float %1425, !dbg !8
  %1434 = bitcast float %1433 to i32, !dbg !8
  %1435 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1434), !dbg !8
  %1436 = bitcast i32 %1435 to float, !dbg !8
  %1437 = select i1 %.not, float %1425, float %1436, !dbg !8
  %1438 = select i1 %.not, float %1436, float %1426, !dbg !8
  %1439 = select i1 %.not1, float %1437, float %1431, !dbg !8
  %1440 = bitcast float %1439 to i32, !dbg !8
  %1441 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1440), !dbg !8
  %1442 = bitcast i32 %1441 to float, !dbg !8
  %1443 = select i1 %.not1, float %1431, float %1442, !dbg !8
  %1444 = select i1 %.not1, float %1442, float %1437, !dbg !8
  %1445 = select i1 %.not1, float %1438, float %1432, !dbg !8
  %1446 = bitcast float %1445 to i32, !dbg !8
  %1447 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1446), !dbg !8
  %1448 = bitcast i32 %1447 to float, !dbg !8
  %1449 = select i1 %.not1, float %1432, float %1448, !dbg !8
  %1450 = select i1 %.not1, float %1448, float %1438, !dbg !8
  %1451 = extractelement <8 x float> %1422, i64 4, !dbg !8
  %1452 = extractelement <8 x float> %1422, i64 5, !dbg !8
  %1453 = extractelement <8 x float> %1422, i64 6, !dbg !8
  %1454 = extractelement <8 x float> %1422, i64 7, !dbg !8
  %1455 = select i1 %.not, float %1452, float %1451, !dbg !8
  %1456 = bitcast float %1455 to i32, !dbg !8
  %1457 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1456), !dbg !8
  %1458 = bitcast i32 %1457 to float, !dbg !8
  %1459 = select i1 %.not, float %1451, float %1458, !dbg !8
  %1460 = select i1 %.not, float %1458, float %1452, !dbg !8
  %1461 = select i1 %.not, float %1454, float %1453, !dbg !8
  %1462 = bitcast float %1461 to i32, !dbg !8
  %1463 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1462), !dbg !8
  %1464 = bitcast i32 %1463 to float, !dbg !8
  %1465 = select i1 %.not, float %1453, float %1464, !dbg !8
  %1466 = select i1 %.not, float %1464, float %1454, !dbg !8
  %1467 = select i1 %.not1, float %1465, float %1459, !dbg !8
  %1468 = bitcast float %1467 to i32, !dbg !8
  %1469 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1468), !dbg !8
  %1470 = bitcast i32 %1469 to float, !dbg !8
  %1471 = select i1 %.not1, float %1459, float %1470, !dbg !8
  %1472 = select i1 %.not1, float %1470, float %1465, !dbg !8
  %1473 = select i1 %.not1, float %1466, float %1460, !dbg !8
  %1474 = bitcast float %1473 to i32, !dbg !8
  %1475 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1474), !dbg !8
  %1476 = bitcast i32 %1475 to float, !dbg !8
  %1477 = select i1 %.not1, float %1460, float %1476, !dbg !8
  %1478 = select i1 %.not1, float %1476, float %1466, !dbg !8
  %1479 = extractvalue [2 x <8 x float>] %126, 0, !dbg !8
  %1480 = extractelement <8 x float> %1479, i64 0, !dbg !8
  %1481 = extractelement <8 x float> %1479, i64 1, !dbg !8
  %1482 = extractelement <8 x float> %1479, i64 2, !dbg !8
  %1483 = extractelement <8 x float> %1479, i64 3, !dbg !8
  %1484 = select i1 %.not, float %1481, float %1480, !dbg !8
  %1485 = bitcast float %1484 to i32, !dbg !8
  %1486 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1485), !dbg !8
  %1487 = bitcast i32 %1486 to float, !dbg !8
  %1488 = select i1 %.not, float %1480, float %1487, !dbg !8
  %1489 = select i1 %.not, float %1487, float %1481, !dbg !8
  %1490 = select i1 %.not, float %1483, float %1482, !dbg !8
  %1491 = bitcast float %1490 to i32, !dbg !8
  %1492 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1491), !dbg !8
  %1493 = bitcast i32 %1492 to float, !dbg !8
  %1494 = select i1 %.not, float %1482, float %1493, !dbg !8
  %1495 = select i1 %.not, float %1493, float %1483, !dbg !8
  %1496 = select i1 %.not1, float %1494, float %1488, !dbg !8
  %1497 = bitcast float %1496 to i32, !dbg !8
  %1498 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1497), !dbg !8
  %1499 = select i1 %.not1, float %1495, float %1489, !dbg !8
  %1500 = bitcast float %1499 to i32, !dbg !8
  %1501 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1500), !dbg !8
  %1502 = extractelement <8 x float> %1479, i64 4, !dbg !8
  %1503 = extractelement <8 x float> %1479, i64 5, !dbg !8
  %1504 = extractelement <8 x float> %1479, i64 6, !dbg !8
  %1505 = extractelement <8 x float> %1479, i64 7, !dbg !8
  %1506 = select i1 %.not, float %1503, float %1502, !dbg !8
  %1507 = bitcast float %1506 to i32, !dbg !8
  %1508 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1507), !dbg !8
  %1509 = bitcast i32 %1508 to float, !dbg !8
  %1510 = select i1 %.not, float %1502, float %1509, !dbg !8
  %1511 = select i1 %.not, float %1509, float %1503, !dbg !8
  %1512 = select i1 %.not, float %1505, float %1504, !dbg !8
  %1513 = bitcast float %1512 to i32, !dbg !8
  %1514 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1513), !dbg !8
  %1515 = bitcast i32 %1514 to float, !dbg !8
  %1516 = select i1 %.not, float %1504, float %1515, !dbg !8
  %1517 = select i1 %.not, float %1515, float %1505, !dbg !8
  %1518 = select i1 %.not1, float %1516, float %1510, !dbg !8
  %1519 = bitcast float %1518 to i32, !dbg !8
  %1520 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1519), !dbg !8
  %1521 = select i1 %.not1, float %1517, float %1511, !dbg !8
  %1522 = bitcast float %1521 to i32, !dbg !8
  %1523 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1522), !dbg !8
  %1524 = extractvalue [2 x <8 x float>] %126, 1, !dbg !8
  %1525 = extractelement <8 x float> %1524, i64 0, !dbg !8
  %1526 = extractelement <8 x float> %1524, i64 1, !dbg !8
  %1527 = extractelement <8 x float> %1524, i64 2, !dbg !8
  %1528 = extractelement <8 x float> %1524, i64 3, !dbg !8
  %1529 = select i1 %.not, float %1526, float %1525, !dbg !8
  %1530 = bitcast float %1529 to i32, !dbg !8
  %1531 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1530), !dbg !8
  %1532 = bitcast i32 %1531 to float, !dbg !8
  %1533 = select i1 %.not, float %1525, float %1532, !dbg !8
  %1534 = select i1 %.not, float %1532, float %1526, !dbg !8
  %1535 = select i1 %.not, float %1528, float %1527, !dbg !8
  %1536 = bitcast float %1535 to i32, !dbg !8
  %1537 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1536), !dbg !8
  %1538 = bitcast i32 %1537 to float, !dbg !8
  %1539 = select i1 %.not, float %1527, float %1538, !dbg !8
  %1540 = select i1 %.not, float %1538, float %1528, !dbg !8
  %1541 = select i1 %.not1, float %1539, float %1533, !dbg !8
  %1542 = bitcast float %1541 to i32, !dbg !8
  %1543 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1542), !dbg !8
  %1544 = select i1 %.not1, float %1540, float %1534, !dbg !8
  %1545 = bitcast float %1544 to i32, !dbg !8
  %1546 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1545), !dbg !8
  %1547 = extractelement <8 x float> %1524, i64 4, !dbg !8
  %1548 = extractelement <8 x float> %1524, i64 5, !dbg !8
  %1549 = extractelement <8 x float> %1524, i64 6, !dbg !8
  %1550 = extractelement <8 x float> %1524, i64 7, !dbg !8
  %1551 = select i1 %.not, float %1548, float %1547, !dbg !8
  %1552 = bitcast float %1551 to i32, !dbg !8
  %1553 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1552), !dbg !8
  %1554 = bitcast i32 %1553 to float, !dbg !8
  %1555 = select i1 %.not, float %1547, float %1554, !dbg !8
  %1556 = select i1 %.not, float %1554, float %1548, !dbg !8
  %1557 = select i1 %.not, float %1550, float %1549, !dbg !8
  %1558 = bitcast float %1557 to i32, !dbg !8
  %1559 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1558), !dbg !8
  %1560 = bitcast i32 %1559 to float, !dbg !8
  %1561 = select i1 %.not, float %1549, float %1560, !dbg !8
  %1562 = select i1 %.not, float %1560, float %1550, !dbg !8
  %1563 = select i1 %.not1, float %1561, float %1555, !dbg !8
  %1564 = bitcast float %1563 to i32, !dbg !8
  %1565 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1564), !dbg !8
  %1566 = select i1 %.not1, float %1562, float %1556, !dbg !8
  %1567 = bitcast float %1566 to i32, !dbg !8
  %1568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1567), !dbg !8
  %1569 = shufflevector <1 x float> %1134, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1570 = shufflevector <1 x float> %1125, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1571 = bitcast i32 %1565 to float, !dbg !8
  %1572 = select i1 %.not1, float %1555, float %1571, !dbg !8
  %1573 = insertelement <4 x float> poison, float %1572, i64 0, !dbg !17
  %1574 = bitcast i32 %1568 to float, !dbg !8
  %1575 = select i1 %.not1, float %1556, float %1574, !dbg !8
  %1576 = insertelement <4 x float> %1573, float %1575, i64 1, !dbg !17
  %1577 = select i1 %.not1, float %1571, float %1561, !dbg !8
  %1578 = insertelement <4 x float> %1576, float %1577, i64 2, !dbg !17
  %1579 = select i1 %.not1, float %1574, float %1562, !dbg !8
  %1580 = insertelement <4 x float> %1578, float %1579, i64 3, !dbg !17
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1570, !dbg !17
  %normalized = fmul <4 x float> %1580, %row.reciprocal, !dbg !17
  %1581 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !18
  %1582 = extractelement <4 x half> %1581, i64 3, !dbg !18
  %1583 = extractelement <4 x half> %1581, i64 2, !dbg !18
  %1584 = extractelement <4 x half> %1581, i64 1, !dbg !18
  %1585 = extractelement <4 x half> %1581, i64 0, !dbg !18
  %1586 = bitcast i32 %1543 to float, !dbg !8
  %1587 = select i1 %.not1, float %1533, float %1586, !dbg !8
  %1588 = insertelement <4 x float> poison, float %1587, i64 0, !dbg !17
  %1589 = bitcast i32 %1546 to float, !dbg !8
  %1590 = select i1 %.not1, float %1534, float %1589, !dbg !8
  %1591 = insertelement <4 x float> %1588, float %1590, i64 1, !dbg !17
  %1592 = select i1 %.not1, float %1586, float %1539, !dbg !8
  %1593 = insertelement <4 x float> %1591, float %1592, i64 2, !dbg !17
  %1594 = select i1 %.not1, float %1589, float %1540, !dbg !8
  %1595 = insertelement <4 x float> %1593, float %1594, i64 3, !dbg !17
  %normalized2 = fmul <4 x float> %1595, %row.reciprocal, !dbg !17
  %1596 = fptrunc <4 x float> %normalized2 to <4 x half>, !dbg !18
  %1597 = extractelement <4 x half> %1596, i64 3, !dbg !18
  %1598 = extractelement <4 x half> %1596, i64 2, !dbg !18
  %1599 = extractelement <4 x half> %1596, i64 1, !dbg !18
  %1600 = extractelement <4 x half> %1596, i64 0, !dbg !18
  %1601 = bitcast i32 %1520 to float, !dbg !8
  %1602 = select i1 %.not1, float %1510, float %1601, !dbg !8
  %1603 = insertelement <4 x float> poison, float %1602, i64 0, !dbg !17
  %1604 = bitcast i32 %1523 to float, !dbg !8
  %1605 = select i1 %.not1, float %1511, float %1604, !dbg !8
  %1606 = insertelement <4 x float> %1603, float %1605, i64 1, !dbg !17
  %1607 = select i1 %.not1, float %1601, float %1516, !dbg !8
  %1608 = insertelement <4 x float> %1606, float %1607, i64 2, !dbg !17
  %1609 = select i1 %.not1, float %1604, float %1517, !dbg !8
  %1610 = insertelement <4 x float> %1608, float %1609, i64 3, !dbg !17
  %row.reciprocal3 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1569, !dbg !17
  %normalized4 = fmul <4 x float> %1610, %row.reciprocal3, !dbg !17
  %1611 = fptrunc <4 x float> %normalized4 to <4 x half>, !dbg !18
  %1612 = extractelement <4 x half> %1611, i64 3, !dbg !18
  %1613 = extractelement <4 x half> %1611, i64 2, !dbg !18
  %1614 = extractelement <4 x half> %1611, i64 1, !dbg !18
  %1615 = extractelement <4 x half> %1611, i64 0, !dbg !18
  %1616 = bitcast i32 %1498 to float, !dbg !8
  %1617 = select i1 %.not1, float %1488, float %1616, !dbg !8
  %1618 = insertelement <4 x float> poison, float %1617, i64 0, !dbg !17
  %1619 = bitcast i32 %1501 to float, !dbg !8
  %1620 = select i1 %.not1, float %1489, float %1619, !dbg !8
  %1621 = insertelement <4 x float> %1618, float %1620, i64 1, !dbg !17
  %1622 = select i1 %.not1, float %1616, float %1494, !dbg !8
  %1623 = insertelement <4 x float> %1621, float %1622, i64 2, !dbg !17
  %1624 = select i1 %.not1, float %1619, float %1495, !dbg !8
  %1625 = insertelement <4 x float> %1623, float %1624, i64 3, !dbg !17
  %normalized5 = fmul <4 x float> %1625, %row.reciprocal3, !dbg !17
  %1626 = fptrunc <4 x float> %normalized5 to <4 x half>, !dbg !18
  %1627 = extractelement <4 x half> %1626, i64 3, !dbg !18
  %1628 = extractelement <4 x half> %1626, i64 2, !dbg !18
  %1629 = extractelement <4 x half> %1626, i64 1, !dbg !18
  %1630 = extractelement <4 x half> %1626, i64 0, !dbg !18
  %1631 = shl nuw nsw i64 %30, 19, !dbg !19
  %address.thread650 = call i32 asm sideeffect "", "=v,0"(i32 %35) #9, !dbg !19, !range !32, !frisk.register_lifetimes !1
  %address.local651 = zext nneg i32 %address.thread650 to i64, !dbg !9
  %address.local652 = add nuw i64 %31, %address.local651, !dbg !19
  %1632 = lshr i64 %address.local651, 2, !dbg !19
  %address.local654 = and i64 %1632, 48, !dbg !19
  %address.local655 = add i64 %address.local652, %address.local654, !dbg !19
  %address.local656 = and i64 %address.local651, 240, !dbg !19
  %address.local657 = sub i64 %address.local655, %address.local656, !dbg !19
  %address.local658 = add i64 %address.local657, %32, !dbg !19
  %address.local659 = shl nuw nsw i64 %address.local658, 7, !dbg !19
  %address.local660 = add nuw nsw i64 %1631, %address.local659, !dbg !19
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !19
  %address.local662 = lshr i64 %address.local651, 2, !dbg !12
  %address.local663 = and i64 %address.local662, 12, !dbg !12
  %address.local664 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local661, i64 %address.local663, !dbg !19
  store half %1630, ptr addrspace(1) %address.local664, align 2, !dbg !19
  %address.local665 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !19
  %address.local666 = or disjoint i64 %address.local663, 1, !dbg !19
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local665, i64 %address.local666, !dbg !19
  store half %1629, ptr addrspace(1) %address.local667, align 2, !dbg !19
  %address.local668 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !19
  %address.local669 = or disjoint i64 %address.local663, 2, !dbg !19
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local668, i64 %address.local669, !dbg !19
  store half %1628, ptr addrspace(1) %address.local670, align 2, !dbg !19
  %address.local671 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !19
  %address.local672 = or disjoint i64 %address.local663, 3, !dbg !19
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local671, i64 %address.local672, !dbg !19
  store half %1627, ptr addrspace(1) %address.local673, align 2, !dbg !19
  %address.local674 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !19
  %address.local675 = or disjoint i64 %address.local663, 16, !dbg !19
  %address.local676 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local674, i64 %address.local675, !dbg !19
  store half %1615, ptr addrspace(1) %address.local676, align 2, !dbg !19
  %address.local677 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !19
  %address.local678 = or disjoint i64 %address.local663, 17, !dbg !19
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local677, i64 %address.local678, !dbg !19
  store half %1614, ptr addrspace(1) %address.local679, align 2, !dbg !19
  %address.local680 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !19
  %address.local681 = or disjoint i64 %address.local663, 18, !dbg !19
  %address.local682 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local680, i64 %address.local681, !dbg !19
  store half %1613, ptr addrspace(1) %address.local682, align 2, !dbg !19
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !19
  %address.local684 = or disjoint i64 %address.local663, 19, !dbg !19
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local683, i64 %address.local684, !dbg !19
  store half %1612, ptr addrspace(1) %address.local685, align 2, !dbg !19
  %address.local686 = shl i64 %address.local658, 7, !dbg !19
  %address.local687 = add i64 %address.local686, 8192, !dbg !19
  %address.local688 = add nuw nsw i64 %1631, %address.local687, !dbg !19
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local688, !dbg !19
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local689, i64 %address.local663, !dbg !19
  store half %1600, ptr addrspace(1) %address.local690, align 2, !dbg !19
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local688, !dbg !19
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local691, i64 %address.local666, !dbg !19
  store half %1599, ptr addrspace(1) %address.local692, align 2, !dbg !19
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local688, !dbg !19
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local693, i64 %address.local669, !dbg !19
  store half %1598, ptr addrspace(1) %address.local694, align 2, !dbg !19
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local688, !dbg !19
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local695, i64 %address.local672, !dbg !19
  store half %1597, ptr addrspace(1) %address.local696, align 2, !dbg !19
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local688, !dbg !19
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local697, i64 %address.local675, !dbg !19
  store half %1585, ptr addrspace(1) %address.local698, align 2, !dbg !19
  %address.local699 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local688, !dbg !19
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local699, i64 %address.local678, !dbg !19
  store half %1584, ptr addrspace(1) %address.local700, align 2, !dbg !19
  %address.local701 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local688, !dbg !19
  %address.local702 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local701, i64 %address.local681, !dbg !19
  store half %1583, ptr addrspace(1) %address.local702, align 2, !dbg !19
  %address.local703 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local688, !dbg !19
  %address.local704 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local703, i64 %address.local684, !dbg !19
  store half %1582, ptr addrspace(1) %address.local704, align 2, !dbg !19
  %1633 = insertelement <4 x float> poison, float %1471, i64 0, !dbg !17
  %1634 = insertelement <4 x float> %1633, float %1477, i64 1, !dbg !17
  %1635 = insertelement <4 x float> %1634, float %1472, i64 2, !dbg !17
  %1636 = insertelement <4 x float> %1635, float %1478, i64 3, !dbg !17
  %row.reciprocal6 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1570, !dbg !17
  %normalized7 = fmul <4 x float> %1636, %row.reciprocal6, !dbg !17
  %1637 = fptrunc <4 x float> %normalized7 to <4 x half>, !dbg !18
  %1638 = extractelement <4 x half> %1637, i64 3, !dbg !18
  %1639 = extractelement <4 x half> %1637, i64 2, !dbg !18
  %1640 = extractelement <4 x half> %1637, i64 1, !dbg !18
  %1641 = extractelement <4 x half> %1637, i64 0, !dbg !18
  %1642 = insertelement <4 x float> poison, float %1443, i64 0, !dbg !17
  %1643 = insertelement <4 x float> %1642, float %1449, i64 1, !dbg !17
  %1644 = insertelement <4 x float> %1643, float %1444, i64 2, !dbg !17
  %1645 = insertelement <4 x float> %1644, float %1450, i64 3, !dbg !17
  %normalized8 = fmul <4 x float> %1645, %row.reciprocal6, !dbg !17
  %1646 = fptrunc <4 x float> %normalized8 to <4 x half>, !dbg !18
  %1647 = extractelement <4 x half> %1646, i64 3, !dbg !18
  %1648 = extractelement <4 x half> %1646, i64 2, !dbg !18
  %1649 = extractelement <4 x half> %1646, i64 1, !dbg !18
  %1650 = extractelement <4 x half> %1646, i64 0, !dbg !18
  %1651 = insertelement <4 x float> poison, float %1414, i64 0, !dbg !17
  %1652 = insertelement <4 x float> %1651, float %1420, i64 1, !dbg !17
  %1653 = insertelement <4 x float> %1652, float %1415, i64 2, !dbg !17
  %1654 = insertelement <4 x float> %1653, float %1421, i64 3, !dbg !17
  %row.reciprocal9 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1569, !dbg !17
  %normalized10 = fmul <4 x float> %1654, %row.reciprocal9, !dbg !17
  %1655 = fptrunc <4 x float> %normalized10 to <4 x half>, !dbg !18
  %1656 = extractelement <4 x half> %1655, i64 3, !dbg !18
  %1657 = extractelement <4 x half> %1655, i64 2, !dbg !18
  %1658 = extractelement <4 x half> %1655, i64 1, !dbg !18
  %1659 = extractelement <4 x half> %1655, i64 0, !dbg !18
  %1660 = insertelement <4 x float> poison, float %1386, i64 0, !dbg !17
  %1661 = insertelement <4 x float> %1660, float %1392, i64 1, !dbg !17
  %1662 = insertelement <4 x float> %1661, float %1387, i64 2, !dbg !17
  %1663 = insertelement <4 x float> %1662, float %1393, i64 3, !dbg !17
  %normalized11 = fmul <4 x float> %1663, %row.reciprocal9, !dbg !17
  %1664 = fptrunc <4 x float> %normalized11 to <4 x half>, !dbg !18
  %1665 = extractelement <4 x half> %1664, i64 3, !dbg !18
  %1666 = extractelement <4 x half> %1664, i64 2, !dbg !18
  %1667 = extractelement <4 x half> %1664, i64 1, !dbg !18
  %1668 = extractelement <4 x half> %1664, i64 0, !dbg !18
  %1669 = shl nuw nsw i64 %51, 19, !dbg !19
  %address.thread705 = call i32 asm sideeffect "", "=v,0"(i32 %35) #9, !dbg !19, !range !32, !frisk.register_lifetimes !1
  %address.local706 = zext nneg i32 %address.thread705 to i64, !dbg !9
  %address.local707 = add nuw i64 %31, %address.local706, !dbg !19
  %1670 = lshr i64 %address.local706, 2, !dbg !19
  %address.local709 = and i64 %1670, 48, !dbg !19
  %address.local710 = add i64 %address.local707, %address.local709, !dbg !19
  %address.local711 = and i64 %address.local706, 240, !dbg !19
  %address.local712 = sub i64 %address.local710, %address.local711, !dbg !19
  %address.local713 = add i64 %address.local712, %32, !dbg !19
  %address.local714 = shl nuw nsw i64 %address.local713, 7, !dbg !19
  %address.local715 = add nuw nsw i64 %1669, %address.local714, !dbg !19
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local715, !dbg !19
  %address.local717 = lshr i64 %address.local706, 2, !dbg !12
  %address.local718 = and i64 %address.local717, 12, !dbg !12
  %address.local719 = or disjoint i64 %address.local718, 32, !dbg !19
  %address.local720 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local716, i64 %address.local719, !dbg !19
  store half %1668, ptr addrspace(1) %address.local720, align 2, !dbg !19
  %address.local721 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local715, !dbg !19
  %address.local722 = or disjoint i64 %address.local718, 33, !dbg !19
  %address.local723 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local721, i64 %address.local722, !dbg !19
  store half %1667, ptr addrspace(1) %address.local723, align 2, !dbg !19
  %address.local724 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local715, !dbg !19
  %address.local725 = or disjoint i64 %address.local718, 34, !dbg !19
  %address.local726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local724, i64 %address.local725, !dbg !19
  store half %1666, ptr addrspace(1) %address.local726, align 2, !dbg !19
  %address.local727 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local715, !dbg !19
  %address.local728 = or disjoint i64 %address.local718, 35, !dbg !19
  %address.local729 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local727, i64 %address.local728, !dbg !19
  store half %1665, ptr addrspace(1) %address.local729, align 2, !dbg !19
  %address.local730 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local715, !dbg !19
  %address.local731 = or disjoint i64 %address.local718, 48, !dbg !19
  %address.local732 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local730, i64 %address.local731, !dbg !19
  store half %1659, ptr addrspace(1) %address.local732, align 2, !dbg !19
  %address.local733 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local715, !dbg !19
  %address.local734 = or disjoint i64 %address.local718, 49, !dbg !19
  %address.local735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local733, i64 %address.local734, !dbg !19
  store half %1658, ptr addrspace(1) %address.local735, align 2, !dbg !19
  %address.local736 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local715, !dbg !19
  %address.local737 = or disjoint i64 %address.local718, 50, !dbg !19
  %address.local738 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local736, i64 %address.local737, !dbg !19
  store half %1657, ptr addrspace(1) %address.local738, align 2, !dbg !19
  %address.local739 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local715, !dbg !19
  %address.local740 = or i64 %address.local717, 51, !dbg !19
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local739, i64 %address.local740, !dbg !19
  store half %1656, ptr addrspace(1) %address.local741, align 2, !dbg !19
  %address.local742 = shl i64 %address.local713, 7, !dbg !19
  %address.local743 = add i64 %address.local742, 8192, !dbg !19
  %address.local744 = add nuw nsw i64 %1669, %address.local743, !dbg !19
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local744, !dbg !19
  %address.local746 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local745, i64 %address.local719, !dbg !19
  store half %1650, ptr addrspace(1) %address.local746, align 2, !dbg !19
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local744, !dbg !19
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local747, i64 %address.local722, !dbg !19
  store half %1649, ptr addrspace(1) %address.local748, align 2, !dbg !19
  %address.local749 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local744, !dbg !19
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local749, i64 %address.local725, !dbg !19
  store half %1648, ptr addrspace(1) %address.local750, align 2, !dbg !19
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local744, !dbg !19
  %address.local752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local751, i64 %address.local728, !dbg !19
  store half %1647, ptr addrspace(1) %address.local752, align 2, !dbg !19
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local744, !dbg !19
  %address.local754 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local753, i64 %address.local731, !dbg !19
  store half %1641, ptr addrspace(1) %address.local754, align 2, !dbg !19
  %address.local755 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local744, !dbg !19
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local755, i64 %address.local734, !dbg !19
  store half %1640, ptr addrspace(1) %address.local756, align 2, !dbg !19
  %address.local757 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local744, !dbg !19
  %address.local758 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local757, i64 %address.local737, !dbg !19
  store half %1639, ptr addrspace(1) %address.local758, align 2, !dbg !19
  %address.local759 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local744, !dbg !19
  %address.local760 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local759, i64 %address.local740, !dbg !19
  store half %1638, ptr addrspace(1) %address.local760, align 2, !dbg !19
  %1671 = insertelement <4 x float> poison, float %1357, i64 0, !dbg !17
  %1672 = insertelement <4 x float> %1671, float %1363, i64 1, !dbg !17
  %1673 = insertelement <4 x float> %1672, float %1358, i64 2, !dbg !17
  %1674 = insertelement <4 x float> %1673, float %1364, i64 3, !dbg !17
  %row.reciprocal12 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1570, !dbg !17
  %normalized13 = fmul <4 x float> %1674, %row.reciprocal12, !dbg !17
  %1675 = fptrunc <4 x float> %normalized13 to <4 x half>, !dbg !18
  %1676 = extractelement <4 x half> %1675, i64 3, !dbg !18
  %1677 = extractelement <4 x half> %1675, i64 2, !dbg !18
  %1678 = extractelement <4 x half> %1675, i64 1, !dbg !18
  %1679 = extractelement <4 x half> %1675, i64 0, !dbg !18
  %1680 = insertelement <4 x float> poison, float %1329, i64 0, !dbg !17
  %1681 = insertelement <4 x float> %1680, float %1335, i64 1, !dbg !17
  %1682 = insertelement <4 x float> %1681, float %1330, i64 2, !dbg !17
  %1683 = insertelement <4 x float> %1682, float %1336, i64 3, !dbg !17
  %normalized14 = fmul <4 x float> %1683, %row.reciprocal12, !dbg !17
  %1684 = fptrunc <4 x float> %normalized14 to <4 x half>, !dbg !18
  %1685 = extractelement <4 x half> %1684, i64 3, !dbg !18
  %1686 = extractelement <4 x half> %1684, i64 2, !dbg !18
  %1687 = extractelement <4 x half> %1684, i64 1, !dbg !18
  %1688 = extractelement <4 x half> %1684, i64 0, !dbg !18
  %1689 = insertelement <4 x float> poison, float %1300, i64 0, !dbg !17
  %1690 = insertelement <4 x float> %1689, float %1306, i64 1, !dbg !17
  %1691 = insertelement <4 x float> %1690, float %1301, i64 2, !dbg !17
  %1692 = insertelement <4 x float> %1691, float %1307, i64 3, !dbg !17
  %row.reciprocal15 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1569, !dbg !17
  %normalized16 = fmul <4 x float> %1692, %row.reciprocal15, !dbg !17
  %1693 = fptrunc <4 x float> %normalized16 to <4 x half>, !dbg !18
  %1694 = extractelement <4 x half> %1693, i64 3, !dbg !18
  %1695 = extractelement <4 x half> %1693, i64 2, !dbg !18
  %1696 = extractelement <4 x half> %1693, i64 1, !dbg !18
  %1697 = extractelement <4 x half> %1693, i64 0, !dbg !18
  %1698 = insertelement <4 x float> poison, float %1272, i64 0, !dbg !17
  %1699 = insertelement <4 x float> %1698, float %1278, i64 1, !dbg !17
  %1700 = insertelement <4 x float> %1699, float %1273, i64 2, !dbg !17
  %1701 = insertelement <4 x float> %1700, float %1279, i64 3, !dbg !17
  %normalized17 = fmul <4 x float> %1701, %row.reciprocal15, !dbg !17
  %1702 = fptrunc <4 x float> %normalized17 to <4 x half>, !dbg !18
  %1703 = extractelement <4 x half> %1702, i64 3, !dbg !18
  %1704 = extractelement <4 x half> %1702, i64 2, !dbg !18
  %1705 = extractelement <4 x half> %1702, i64 1, !dbg !18
  %1706 = extractelement <4 x half> %1702, i64 0, !dbg !18
  %1707 = shl nuw nsw i64 %51, 19, !dbg !19
  %address.thread761 = call i32 asm sideeffect "", "=v,0"(i32 %35) #9, !dbg !19, !range !32, !frisk.register_lifetimes !1
  %address.local762 = zext nneg i32 %address.thread761 to i64, !dbg !9
  %address.local763 = add nuw i64 %31, %address.local762, !dbg !19
  %1708 = lshr i64 %address.local762, 2, !dbg !19
  %address.local765 = and i64 %1708, 48, !dbg !19
  %address.local766 = add i64 %address.local763, %address.local765, !dbg !19
  %address.local767 = and i64 %address.local762, 240, !dbg !19
  %address.local768 = sub i64 %address.local766, %address.local767, !dbg !19
  %address.local769 = add i64 %address.local768, %32, !dbg !19
  %address.local770 = shl nuw nsw i64 %address.local769, 7, !dbg !19
  %address.local771 = add nuw nsw i64 %1707, %address.local770, !dbg !19
  %address.local772 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local771, !dbg !19
  %address.local773 = lshr i64 %address.local762, 2, !dbg !12
  %address.local774 = and i64 %address.local773, 12, !dbg !12
  %address.local775 = or disjoint i64 %address.local774, 64, !dbg !19
  %address.local776 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local772, i64 %address.local775, !dbg !19
  store half %1706, ptr addrspace(1) %address.local776, align 2, !dbg !19
  %address.local777 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local771, !dbg !19
  %address.local778 = or disjoint i64 %address.local774, 65, !dbg !19
  %address.local779 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local777, i64 %address.local778, !dbg !19
  store half %1705, ptr addrspace(1) %address.local779, align 2, !dbg !19
  %address.local780 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local771, !dbg !19
  %address.local781 = or disjoint i64 %address.local774, 66, !dbg !19
  %address.local782 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local780, i64 %address.local781, !dbg !19
  store half %1704, ptr addrspace(1) %address.local782, align 2, !dbg !19
  %address.local783 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local771, !dbg !19
  %address.local784 = or disjoint i64 %address.local774, 67, !dbg !19
  %address.local785 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local783, i64 %address.local784, !dbg !19
  store half %1703, ptr addrspace(1) %address.local785, align 2, !dbg !19
  %address.local786 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local771, !dbg !19
  %address.local787 = or disjoint i64 %address.local774, 80, !dbg !19
  %address.local788 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local786, i64 %address.local787, !dbg !19
  store half %1697, ptr addrspace(1) %address.local788, align 2, !dbg !19
  %address.local789 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local771, !dbg !19
  %address.local790 = or disjoint i64 %address.local774, 81, !dbg !19
  %address.local791 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local789, i64 %address.local790, !dbg !19
  store half %1696, ptr addrspace(1) %address.local791, align 2, !dbg !19
  %address.local792 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local771, !dbg !19
  %address.local793 = or disjoint i64 %address.local774, 82, !dbg !19
  %address.local794 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local792, i64 %address.local793, !dbg !19
  store half %1695, ptr addrspace(1) %address.local794, align 2, !dbg !19
  %address.local795 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local771, !dbg !19
  %address.local796 = or disjoint i64 %address.local774, 83, !dbg !19
  %address.local797 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local795, i64 %address.local796, !dbg !19
  store half %1694, ptr addrspace(1) %address.local797, align 2, !dbg !19
  %address.local798 = shl i64 %address.local769, 7, !dbg !19
  %address.local799 = add i64 %address.local798, 8192, !dbg !19
  %address.local800 = add nuw nsw i64 %1707, %address.local799, !dbg !19
  %address.local801 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local800, !dbg !19
  %address.local802 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local801, i64 %address.local775, !dbg !19
  store half %1688, ptr addrspace(1) %address.local802, align 2, !dbg !19
  %address.local803 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local800, !dbg !19
  %address.local804 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local803, i64 %address.local778, !dbg !19
  store half %1687, ptr addrspace(1) %address.local804, align 2, !dbg !19
  %address.local805 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local800, !dbg !19
  %address.local806 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local805, i64 %address.local781, !dbg !19
  store half %1686, ptr addrspace(1) %address.local806, align 2, !dbg !19
  %address.local807 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local800, !dbg !19
  %address.local808 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local807, i64 %address.local784, !dbg !19
  store half %1685, ptr addrspace(1) %address.local808, align 2, !dbg !19
  %address.local809 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local800, !dbg !19
  %address.local810 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local809, i64 %address.local787, !dbg !19
  store half %1679, ptr addrspace(1) %address.local810, align 2, !dbg !19
  %address.local811 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local800, !dbg !19
  %address.local812 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local811, i64 %address.local790, !dbg !19
  store half %1678, ptr addrspace(1) %address.local812, align 2, !dbg !19
  %address.local813 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local800, !dbg !19
  %address.local814 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local813, i64 %address.local793, !dbg !19
  store half %1677, ptr addrspace(1) %address.local814, align 2, !dbg !19
  %address.local815 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local800, !dbg !19
  %address.local816 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local815, i64 %address.local796, !dbg !19
  store half %1676, ptr addrspace(1) %address.local816, align 2, !dbg !19
  %1709 = insertelement <4 x float> poison, float %1243, i64 0, !dbg !17
  %1710 = insertelement <4 x float> %1709, float %1249, i64 1, !dbg !17
  %1711 = insertelement <4 x float> %1710, float %1244, i64 2, !dbg !17
  %1712 = insertelement <4 x float> %1711, float %1250, i64 3, !dbg !17
  %row.reciprocal18 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1570, !dbg !17
  %normalized19 = fmul <4 x float> %1712, %row.reciprocal18, !dbg !17
  %1713 = fptrunc <4 x float> %normalized19 to <4 x half>, !dbg !18
  %1714 = extractelement <4 x half> %1713, i64 3, !dbg !18
  %1715 = extractelement <4 x half> %1713, i64 2, !dbg !18
  %1716 = extractelement <4 x half> %1713, i64 1, !dbg !18
  %1717 = extractelement <4 x half> %1713, i64 0, !dbg !18
  %1718 = insertelement <4 x float> poison, float %1215, i64 0, !dbg !17
  %1719 = insertelement <4 x float> %1718, float %1221, i64 1, !dbg !17
  %1720 = insertelement <4 x float> %1719, float %1216, i64 2, !dbg !17
  %1721 = insertelement <4 x float> %1720, float %1222, i64 3, !dbg !17
  %normalized20 = fmul <4 x float> %1721, %row.reciprocal18, !dbg !17
  %1722 = fptrunc <4 x float> %normalized20 to <4 x half>, !dbg !18
  %1723 = extractelement <4 x half> %1722, i64 3, !dbg !18
  %1724 = extractelement <4 x half> %1722, i64 2, !dbg !18
  %1725 = extractelement <4 x half> %1722, i64 1, !dbg !18
  %1726 = extractelement <4 x half> %1722, i64 0, !dbg !18
  %1727 = insertelement <4 x float> poison, float %1186, i64 0, !dbg !17
  %1728 = insertelement <4 x float> %1727, float %1192, i64 1, !dbg !17
  %1729 = insertelement <4 x float> %1728, float %1187, i64 2, !dbg !17
  %1730 = insertelement <4 x float> %1729, float %1193, i64 3, !dbg !17
  %row.reciprocal21 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1569, !dbg !17
  %normalized22 = fmul <4 x float> %1730, %row.reciprocal21, !dbg !17
  %1731 = fptrunc <4 x float> %normalized22 to <4 x half>, !dbg !18
  %1732 = extractelement <4 x half> %1731, i64 3, !dbg !18
  %1733 = extractelement <4 x half> %1731, i64 2, !dbg !18
  %1734 = extractelement <4 x half> %1731, i64 1, !dbg !18
  %1735 = extractelement <4 x half> %1731, i64 0, !dbg !18
  %1736 = insertelement <4 x float> poison, float %1158, i64 0, !dbg !17
  %1737 = insertelement <4 x float> %1736, float %1164, i64 1, !dbg !17
  %1738 = insertelement <4 x float> %1737, float %1159, i64 2, !dbg !17
  %1739 = insertelement <4 x float> %1738, float %1165, i64 3, !dbg !17
  %normalized23 = fmul <4 x float> %1739, %row.reciprocal21, !dbg !17
  %1740 = fptrunc <4 x float> %normalized23 to <4 x half>, !dbg !18
  %1741 = extractelement <4 x half> %1740, i64 3, !dbg !18
  %1742 = extractelement <4 x half> %1740, i64 2, !dbg !18
  %1743 = extractelement <4 x half> %1740, i64 1, !dbg !18
  %1744 = extractelement <4 x half> %1740, i64 0, !dbg !18
  %1745 = shl nuw nsw i64 %51, 19, !dbg !19
  %address.thread817 = call i32 asm sideeffect "", "=v,0"(i32 %35) #9, !dbg !19, !range !32, !frisk.register_lifetimes !1
  %address.local818 = zext nneg i32 %address.thread817 to i64, !dbg !9
  %address.local819 = add nuw i64 %31, %address.local818, !dbg !19
  %1746 = lshr i64 %address.local818, 2, !dbg !19
  %address.local821 = and i64 %1746, 48, !dbg !19
  %address.local822 = add i64 %address.local819, %address.local821, !dbg !19
  %address.local823 = and i64 %address.local818, 240, !dbg !19
  %address.local824 = sub i64 %address.local822, %address.local823, !dbg !19
  %address.local825 = add i64 %address.local824, %32, !dbg !19
  %address.local826 = shl nuw nsw i64 %address.local825, 7, !dbg !19
  %address.local827 = add nuw nsw i64 %1745, %address.local826, !dbg !19
  %address.local828 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local827, !dbg !19
  %address.local829 = lshr i64 %address.local818, 2, !dbg !12
  %address.local830 = and i64 %address.local829, 12, !dbg !12
  %address.local831 = or disjoint i64 %address.local830, 96, !dbg !19
  %address.local832 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local828, i64 %address.local831, !dbg !19
  store half %1744, ptr addrspace(1) %address.local832, align 2, !dbg !19
  %address.local833 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local827, !dbg !19
  %address.local834 = or disjoint i64 %address.local830, 97, !dbg !19
  %address.local835 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local833, i64 %address.local834, !dbg !19
  store half %1743, ptr addrspace(1) %address.local835, align 2, !dbg !19
  %address.local836 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local827, !dbg !19
  %address.local837 = or disjoint i64 %address.local830, 98, !dbg !19
  %address.local838 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local836, i64 %address.local837, !dbg !19
  store half %1742, ptr addrspace(1) %address.local838, align 2, !dbg !19
  %address.local839 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local827, !dbg !19
  %address.local840 = or disjoint i64 %address.local830, 99, !dbg !19
  %address.local841 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local839, i64 %address.local840, !dbg !19
  store half %1741, ptr addrspace(1) %address.local841, align 2, !dbg !19
  %address.local842 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local827, !dbg !19
  %address.local843 = or disjoint i64 %address.local830, 112, !dbg !19
  %address.local844 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local842, i64 %address.local843, !dbg !19
  store half %1735, ptr addrspace(1) %address.local844, align 2, !dbg !19
  %address.local845 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local827, !dbg !19
  %address.local846 = or disjoint i64 %address.local830, 113, !dbg !19
  %address.local847 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local845, i64 %address.local846, !dbg !19
  store half %1734, ptr addrspace(1) %address.local847, align 2, !dbg !19
  %address.local848 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local827, !dbg !19
  %address.local849 = or disjoint i64 %address.local830, 114, !dbg !19
  %address.local850 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local849, !dbg !19
  store half %1733, ptr addrspace(1) %address.local850, align 2, !dbg !19
  %address.local851 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local827, !dbg !19
  %address.local852 = or i64 %address.local829, 115, !dbg !19
  %address.local853 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local851, i64 %address.local852, !dbg !19
  store half %1732, ptr addrspace(1) %address.local853, align 2, !dbg !19
  %address.local854 = shl i64 %address.local825, 7, !dbg !19
  %address.local855 = add i64 %address.local854, 8192, !dbg !19
  %address.local856 = add nuw nsw i64 %1745, %address.local855, !dbg !19
  %address.local857 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local856, !dbg !19
  %address.local858 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local857, i64 %address.local831, !dbg !19
  store half %1726, ptr addrspace(1) %address.local858, align 2, !dbg !19
  %address.local859 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local856, !dbg !19
  %address.local860 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local834, !dbg !19
  store half %1725, ptr addrspace(1) %address.local860, align 2, !dbg !19
  %address.local861 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local856, !dbg !19
  %address.local862 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local861, i64 %address.local837, !dbg !19
  store half %1724, ptr addrspace(1) %address.local862, align 2, !dbg !19
  %address.local863 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local856, !dbg !19
  %address.local864 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local863, i64 %address.local840, !dbg !19
  store half %1723, ptr addrspace(1) %address.local864, align 2, !dbg !19
  %address.local865 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local856, !dbg !19
  %address.local866 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local865, i64 %address.local843, !dbg !19
  store half %1717, ptr addrspace(1) %address.local866, align 2, !dbg !19
  %address.local867 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local856, !dbg !19
  %address.local868 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local867, i64 %address.local846, !dbg !19
  store half %1716, ptr addrspace(1) %address.local868, align 2, !dbg !19
  %address.local869 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local856, !dbg !19
  %address.local870 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local869, i64 %address.local849, !dbg !19
  store half %1715, ptr addrspace(1) %address.local870, align 2, !dbg !19
  %address.local871 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local856, !dbg !19
  %address.local872 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local871, i64 %address.local852, !dbg !19
  store half %1714, ptr addrspace(1) %address.local872, align 2, !dbg !19
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
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
!llvm.module.flags = !{!31}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/async_task_noncausal")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!6 = !DILocation(line: 74, column: 20, scope: !4)
!8 = !DILocation(line: 57, column: 24, scope: !4)
!9 = !DILocation(line: 67, column: 22, scope: !4)
!12 = !DILocation(line: 69, column: 24, scope: !4)
!13 = !DILocation(line: 68, column: 22, scope: !4)
!14 = !DILocation(line: 87, column: 23, scope: !4)
!15 = !DILocation(line: 70, column: 26, scope: !4)
!16 = !DILocation(line: 86, column: 20, scope: !4)
!17 = !DILocation(line: 92, column: 23, scope: !4)
!18 = !DILocation(line: 93, column: 19, scope: !4)
!19 = !DILocation(line: 94, column: 9, scope: !4)
!20 = !DILocation(line: 71, column: 23, scope: !4)
!21 = !DILocation(line: 75, column: 22, scope: !4)
!22 = !DILocation(line: 76, column: 23, scope: !4)
!23 = !DILocation(line: 77, column: 20, scope: !4)
!24 = !DILocation(line: 78, column: 22, scope: !4)
!25 = !DILocation(line: 79, column: 22, scope: !4)
!26 = !DILocation(line: 80, column: 20, scope: !4)
!27 = !DILocation(line: 81, column: 16, scope: !4)
!28 = !DILocation(line: 88, column: 22, scope: !4)
!30 = !{i32 256, i32 1, i32 1}
!31 = !{i32 2, !"Debug Info Version", i32 3}
!32 = !{i32 0, i32 256}
!33 = !{i32 4}
