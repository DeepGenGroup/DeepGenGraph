; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-162611-z5os6mty/grid-000001/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-162611-z5os6mty/grid-000001/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !30 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 5, !dbg !5
  %10 = add nsw i64 %9, %6, !dbg !5
  %11 = lshr i64 %10, 5, !dbg !5
  %12 = and i64 %11, 576460752303423456, !dbg !5
  %13 = and i64 %6, 31, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = sub nsw i64 31, %14, !dbg !5
  %16 = lshr i64 %10, 5, !dbg !5
  %17 = icmp ugt i64 %10, 1023, !dbg !8
  %18 = add nsw i64 %14, -32, !dbg !8
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !8
  %20 = sdiv i64 %19, 32, !dbg !8
  %21 = sext i1 %17 to i64, !dbg !8
  %22 = xor i64 %20, %21, !dbg !8
  %23 = add nsw i64 %16, %22, !dbg !8
  %24 = and i64 %23, 31, !dbg !8
  %25 = shl nsw i64 %15, 7, !dbg !8
  %26 = mul nsw i64 %22, -4096, !dbg !8
  %27 = add i64 %25, %26, !dbg !8
  %28 = shl nuw nsw i64 %24, 19, !dbg !8
  %29 = shl nsw i64 %27, 7, !dbg !8
  %30 = add i64 %28, %29, !dbg !8
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !32
  %32 = zext nneg i32 %31 to i64, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !8, !range !32, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local271 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local272 = and i64 %address.local271, 1536, !dbg !8
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local272, !dbg !8
  %address.local274 = ptrtoint ptr addrspace(3) %address.local273 to i64, !dbg !8
  %address.local275 = trunc i64 %address.local274 to i32, !dbg !8
  %33 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local275), !dbg !8
  %address.local276 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local277 = and i64 %address.local276, 16256, !dbg !8
  %address.local278 = lshr i64 %address.local, 4, !dbg !8
  %address.local279 = and i64 %address.local278, 8, !dbg !8
  %address.local280 = or disjoint i64 %address.local277, %address.local279, !dbg !8
  %34 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1266 = or disjoint i64 %30, %address.local280, !dbg !8
  %.idx1266.tr = trunc i64 %.idx1266 to i32, !dbg !8
  %dma.buffer.offset1263 = shl i32 %.idx1266.tr, 1, !dbg !8
  %35 = insertelement <2 x i64> poison, i64 %34, i64 0, !dbg !8
  %36 = bitcast <2 x i64> %35 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1264 = shufflevector <4 x i32> %36, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1265 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %33), !dbg !8
  %37 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1265, i32 %dma.buffer.offset1263, <4 x i32> %dma.buffer.rsrc1264) #10, !dbg !8
  %address.local282 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local283 = shl nuw nsw i64 %address.local282, 3, !dbg !8
  %address.local284 = and i64 %address.local283, 3584, !dbg !8
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local284, !dbg !8
  %address.local286 = ptrtoint ptr addrspace(3) %address.local285 to i64, !dbg !8
  %address.local287 = trunc i64 %address.local286 to i32, !dbg !8
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local287), !dbg !8
  %address.local288 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local289 = and i64 %address.local288, 16256, !dbg !8
  %address.local290 = lshr i64 %address.local282, 4, !dbg !8
  %address.local291 = and i64 %address.local290, 24, !dbg !8
  %address.local292 = or disjoint i64 %address.local289, %address.local291, !dbg !8
  %39 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1267 = or disjoint i64 %30, %address.local292, !dbg !8
  %.idx1267.tr = trunc i64 %.idx1267 to i32, !dbg !8
  %dma.buffer.offset1260 = shl i32 %.idx1267.tr, 1, !dbg !8
  %40 = insertelement <2 x i64> poison, i64 %39, i64 0, !dbg !8
  %41 = bitcast <2 x i64> %40 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1261 = shufflevector <4 x i32> %41, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1262 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %38), !dbg !8
  %42 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1262, i32 %dma.buffer.offset1260, <4 x i32> %dma.buffer.rsrc1261) #10, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local272, !dbg !8
  %address.local295 = ptrtoint ptr addrspace(3) %address.local294 to i64, !dbg !8
  %address.local296 = trunc i64 %address.local295 to i32, !dbg !8
  %43 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local296), !dbg !8
  %44 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %45 = or disjoint i64 %30, 32, !dbg !8
  %46 = or disjoint i64 %address.local280, %45, !dbg !8
  %.tr = trunc i64 %46 to i32, !dbg !8
  %dma.buffer.offset1257 = shl i32 %.tr, 1, !dbg !8
  %47 = insertelement <2 x i64> poison, i64 %44, i64 0, !dbg !8
  %48 = bitcast <2 x i64> %47 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1258 = shufflevector <4 x i32> %48, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1259 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %43), !dbg !8
  %49 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1259, i32 %dma.buffer.offset1257, <4 x i32> %dma.buffer.rsrc1258) #10, !dbg !8
  %address.local298 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local284, !dbg !8
  %address.local299 = ptrtoint ptr addrspace(3) %address.local298 to i64, !dbg !8
  %address.local300 = trunc i64 %address.local299 to i32, !dbg !8
  %50 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local300), !dbg !8
  %51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %52 = or disjoint i64 %30, 32, !dbg !8
  %53 = or disjoint i64 %address.local292, %52, !dbg !8
  %.tr1268 = trunc i64 %53 to i32, !dbg !8
  %dma.buffer.offset1254 = shl i32 %.tr1268, 1, !dbg !8
  %54 = insertelement <2 x i64> poison, i64 %51, i64 0, !dbg !8
  %55 = bitcast <2 x i64> %54 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1255 = shufflevector <4 x i32> %55, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1256 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %56 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1256, i32 %dma.buffer.offset1254, <4 x i32> %dma.buffer.rsrc1255) #10, !dbg !8
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
  %57 = load <4 x half>, ptr addrspace(3) %address.local314, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local314, i64 4096, !dbg !10
  %58 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %59 = fmul <4 x half> %57, splat (half 0xH3015), !dbg !10
  %lds.const.gep1064 = getelementptr i8, ptr addrspace(3) %address.local314, i64 1024, !dbg !10
  %60 = load <4 x half>, ptr addrspace(3) %lds.const.gep1064, align 8, !dbg !10
  %61 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !10
  %lds.const.gep1065 = getelementptr i8, ptr addrspace(3) %address.local314, i64 5120, !dbg !10
  %62 = load <4 x half>, ptr addrspace(3) %lds.const.gep1065, align 8, !dbg !10
  %63 = fmul <4 x half> %60, splat (half 0xH3015), !dbg !10
  %64 = fmul <4 x half> %62, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %65 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local275), !dbg !8
  %66 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %67 = or disjoint i64 %30, 64, !dbg !8
  %68 = or disjoint i64 %address.local280, %67, !dbg !8
  %.tr1269 = trunc i64 %68 to i32, !dbg !8
  %dma.buffer.offset1251 = shl i32 %.tr1269, 1, !dbg !8
  %69 = insertelement <2 x i64> poison, i64 %66, i64 0, !dbg !8
  %70 = bitcast <2 x i64> %69 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1252 = shufflevector <4 x i32> %70, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1253 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %65), !dbg !8
  %71 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1253, i32 %dma.buffer.offset1251, <4 x i32> %dma.buffer.rsrc1252) #10, !dbg !8
  %72 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local287), !dbg !8
  %73 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %74 = or disjoint i64 %30, 64, !dbg !8
  %75 = or disjoint i64 %address.local292, %74, !dbg !8
  %.tr1270 = trunc i64 %75 to i32, !dbg !8
  %dma.buffer.offset1248 = shl i32 %.tr1270, 1, !dbg !8
  %76 = insertelement <2 x i64> poison, i64 %73, i64 0, !dbg !8
  %77 = bitcast <2 x i64> %76 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1249 = shufflevector <4 x i32> %77, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1250 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %72), !dbg !8
  %78 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1250, i32 %dma.buffer.offset1248, <4 x i32> %dma.buffer.rsrc1249) #10, !dbg !8
  %lds.const.gep1066 = getelementptr i8, ptr addrspace(3) %address.local314, i64 8192, !dbg !10
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1066, align 8, !dbg !10
  %lds.const.gep1067 = getelementptr i8, ptr addrspace(3) %address.local314, i64 12288, !dbg !10
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1067, align 8, !dbg !10
  %lds.const.gep1068 = getelementptr i8, ptr addrspace(3) %address.local314, i64 9216, !dbg !10
  %81 = load <4 x half>, ptr addrspace(3) %lds.const.gep1068, align 8, !dbg !10
  %82 = fmul <4 x half> %79, splat (half 0xH3015), !dbg !10
  %lds.const.gep1069 = getelementptr i8, ptr addrspace(3) %address.local314, i64 13312, !dbg !10
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1069, align 8, !dbg !10
  %84 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !10
  %85 = fmul <4 x half> %81, splat (half 0xH3015), !dbg !10
  %86 = fmul <4 x half> %83, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %87 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local296), !dbg !8
  %88 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %89 = or disjoint i64 %30, 96, !dbg !8
  %90 = or disjoint i64 %address.local280, %89, !dbg !8
  %.tr1271 = trunc i64 %90 to i32, !dbg !8
  %dma.buffer.offset1245 = shl i32 %.tr1271, 1, !dbg !8
  %91 = insertelement <2 x i64> poison, i64 %88, i64 0, !dbg !8
  %92 = bitcast <2 x i64> %91 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1246 = shufflevector <4 x i32> %92, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1247 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %87), !dbg !8
  %93 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1247, i32 %dma.buffer.offset1245, <4 x i32> %dma.buffer.rsrc1246) #10, !dbg !8
  %94 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local300), !dbg !8
  %95 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %96 = or disjoint i64 %30, 96, !dbg !8
  %97 = or disjoint i64 %address.local292, %96, !dbg !8
  %.tr1272 = trunc i64 %97 to i32, !dbg !8
  %dma.buffer.offset1242 = shl i32 %.tr1272, 1, !dbg !8
  %98 = insertelement <2 x i64> poison, i64 %95, i64 0, !dbg !8
  %99 = bitcast <2 x i64> %98 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1243 = shufflevector <4 x i32> %99, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1244 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !8
  %100 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1244, i32 %dma.buffer.offset1242, <4 x i32> %dma.buffer.rsrc1243) #10, !dbg !8
  %101 = load <4 x half>, ptr addrspace(3) %address.local314, align 8, !dbg !10
  %lds.const.gep1070 = getelementptr i8, ptr addrspace(3) %address.local314, i64 4096, !dbg !10
  %102 = load <4 x half>, ptr addrspace(3) %lds.const.gep1070, align 8, !dbg !10
  %lds.const.gep1071 = getelementptr i8, ptr addrspace(3) %address.local314, i64 1024, !dbg !10
  %103 = load <4 x half>, ptr addrspace(3) %lds.const.gep1071, align 8, !dbg !10
  %104 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !10
  %lds.const.gep1072 = getelementptr i8, ptr addrspace(3) %address.local314, i64 5120, !dbg !10
  %105 = load <4 x half>, ptr addrspace(3) %lds.const.gep1072, align 8, !dbg !10
  %106 = fmul <4 x half> %102, splat (half 0xH3015), !dbg !10
  %107 = fmul <4 x half> %103, splat (half 0xH3015), !dbg !10
  %108 = fmul <4 x half> %105, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1073 = getelementptr i8, ptr addrspace(3) %address.local314, i64 8192, !dbg !10
  %109 = load <4 x half>, ptr addrspace(3) %lds.const.gep1073, align 8, !dbg !10
  %lds.const.gep1074 = getelementptr i8, ptr addrspace(3) %address.local314, i64 12288, !dbg !10
  %110 = load <4 x half>, ptr addrspace(3) %lds.const.gep1074, align 8, !dbg !10
  %lds.const.gep1075 = getelementptr i8, ptr addrspace(3) %address.local314, i64 9216, !dbg !10
  %111 = load <4 x half>, ptr addrspace(3) %lds.const.gep1075, align 8, !dbg !10
  %112 = fmul <4 x half> %109, splat (half 0xH3015), !dbg !10
  %lds.const.gep1076 = getelementptr i8, ptr addrspace(3) %address.local314, i64 13312, !dbg !10
  %113 = load <4 x half>, ptr addrspace(3) %lds.const.gep1076, align 8, !dbg !10
  %114 = fmul <4 x half> %110, splat (half 0xH3015), !dbg !10
  %115 = fmul <4 x half> %111, splat (half 0xH3015), !dbg !10
  %116 = fmul <4 x half> %113, splat (half 0xH3015), !dbg !10
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %117 = shl nuw nsw i64 %32, 2, !dbg !14
  %118 = shl nuw nsw i64 %32, 2, !dbg !14
  %119 = and i64 %118, 1008, !dbg !14
  %120 = sub nsw i64 %117, %119, !dbg !14
  %121 = and i32 %31, 1, !dbg !14
  %index.compact211 = zext nneg i32 %121 to i64, !dbg !14
  %122 = or disjoint i64 %120, %index.compact211, !dbg !14
  %123 = add nuw nsw i64 %122, 1, !dbg !14
  %124 = sdiv i64 %123, 2, !dbg !14
  br label %125, !dbg !7

125:                                              ; preds = %134, %4
  %126 = phi i64 [ %1288, %134 ], [ 0, %4 ], !dbg !7
  %127 = phi [2 x <8 x float>] [ %1273, %134 ], [ zeroinitializer, %4 ], !dbg !7
  %128 = phi [2 x <1 x float>] [ %943, %134 ], [ zeroinitializer, %4 ], !dbg !7
  %129 = phi [2 x <1 x float>] [ %1275, %134 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %130 = phi [2 x <8 x float>] [ %1279, %134 ], [ zeroinitializer, %4 ], !dbg !7
  %131 = phi [2 x <8 x float>] [ %1283, %134 ], [ zeroinitializer, %4 ], !dbg !7
  %132 = phi [2 x <8 x float>] [ %1287, %134 ], [ zeroinitializer, %4 ], !dbg !7
  %133 = icmp samesign ult i64 %126, 32, !dbg !7
  br i1 %133, label %134, label %1289, !dbg !7

134:                                              ; preds = %125
  %135 = shl i64 %16, 19, !dbg !12
  %136 = and i64 %135, 16252928, !dbg !12
  %137 = shl nuw nsw i64 %126, 14, !dbg !12
  %138 = or disjoint i64 %136, %137, !dbg !12
  %address.thread365 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !12, !range !32, !frisk.register_lifetimes !1
  %address.local366 = zext nneg i32 %address.thread365 to i64, !dbg !8
  %address.local367 = shl nuw nsw i64 %address.local366, 3, !dbg !8
  %address.local368 = and i64 %address.local367, 1536, !dbg !8
  %address.local369 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local368, !dbg !8
  %address.local370 = ptrtoint ptr addrspace(3) %address.local369 to i64, !dbg !8
  %address.local371 = trunc i64 %address.local370 to i32, !dbg !8
  %139 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !12
  %address.local372 = lshr i64 %address.local366, 4, !dbg !12
  %address.local373 = and i64 %address.local372, 8, !dbg !12
  %address.local374 = and i64 %address.local366, 123, !dbg !12
  %address.local375 = lshr i64 %address.local366, 2, !dbg !12
  %address.local376 = and i64 %address.local375, 4, !dbg !12
  %address.local377 = add nuw nsw i64 %address.local376, %address.local366, !dbg !12
  %address.local378 = and i64 %address.local377, 4, !dbg !12
  %address.local379 = or disjoint i64 %address.local374, %address.local378, !dbg !12
  %address.local380 = shl nuw nsw i64 %address.local379, 7, !dbg !12
  %address.local381 = or disjoint i64 %address.local373, %address.local380, !dbg !12
  %140 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1273 = or disjoint i64 %138, %address.local381, !dbg !12
  %.idx1273.tr = trunc nuw i64 %.idx1273 to i32, !dbg !12
  %dma.buffer.offset1239 = shl nuw nsw i32 %.idx1273.tr, 1, !dbg !12
  %141 = insertelement <2 x i64> poison, i64 %140, i64 0, !dbg !12
  %142 = bitcast <2 x i64> %141 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1240 = shufflevector <4 x i32> %142, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %139), !dbg !12
  %143 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1241, i32 %dma.buffer.offset1239, <4 x i32> %dma.buffer.rsrc1240) #10, !dbg !12
  %address.local383 = or disjoint i64 %address.local366, 256, !dbg !8
  %address.local384 = shl nuw nsw i64 %address.local383, 3, !dbg !8
  %address.local385 = and i64 %address.local384, 3584, !dbg !8
  %address.local386 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local385, !dbg !8
  %address.local387 = ptrtoint ptr addrspace(3) %address.local386 to i64, !dbg !8
  %address.local388 = trunc i64 %address.local387 to i32, !dbg !8
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !12
  %address.local389 = lshr i64 %address.local383, 4, !dbg !12
  %address.local390 = and i64 %address.local389, 24, !dbg !12
  %address.local391 = and i64 %address.local366, 123, !dbg !12
  %address.local392 = lshr i64 %address.local366, 2, !dbg !12
  %address.local393 = and i64 %address.local392, 4, !dbg !12
  %address.local394 = add nuw nsw i64 %address.local393, %address.local366, !dbg !12
  %address.local395 = and i64 %address.local394, 4, !dbg !12
  %address.local396 = or disjoint i64 %address.local391, %address.local395, !dbg !12
  %address.local397 = shl nuw nsw i64 %address.local396, 7, !dbg !12
  %address.local398 = or disjoint i64 %address.local390, %address.local397, !dbg !12
  %145 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1274 = or disjoint i64 %138, %address.local398, !dbg !12
  %.idx1274.tr = trunc nuw i64 %.idx1274 to i32, !dbg !12
  %dma.buffer.offset1236 = shl nuw nsw i32 %.idx1274.tr, 1, !dbg !12
  %146 = insertelement <2 x i64> poison, i64 %145, i64 0, !dbg !12
  %147 = bitcast <2 x i64> %146 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1237 = shufflevector <4 x i32> %147, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1238 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %144), !dbg !12
  %148 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1238, i32 %dma.buffer.offset1236, <4 x i32> %dma.buffer.rsrc1237) #10, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %149 = or disjoint i64 %138, 32, !dbg !12
  %address.local400 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local368, !dbg !8
  %address.local401 = ptrtoint ptr addrspace(3) %address.local400 to i64, !dbg !8
  %address.local402 = trunc i64 %address.local401 to i32, !dbg !8
  %150 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local402), !dbg !12
  %151 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1275 = or disjoint i64 %149, %address.local381, !dbg !12
  %.idx1275.tr = trunc nuw i64 %.idx1275 to i32, !dbg !12
  %dma.buffer.offset1233 = shl nuw nsw i32 %.idx1275.tr, 1, !dbg !12
  %152 = insertelement <2 x i64> poison, i64 %151, i64 0, !dbg !12
  %153 = bitcast <2 x i64> %152 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1234 = shufflevector <4 x i32> %153, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1235 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %150), !dbg !12
  %154 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1235, i32 %dma.buffer.offset1233, <4 x i32> %dma.buffer.rsrc1234) #10, !dbg !12
  %address.local404 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local385, !dbg !8
  %address.local405 = ptrtoint ptr addrspace(3) %address.local404 to i64, !dbg !8
  %address.local406 = trunc i64 %address.local405 to i32, !dbg !8
  %155 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local406), !dbg !12
  %156 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1276 = or disjoint i64 %149, %address.local398, !dbg !12
  %.idx1276.tr = trunc nuw i64 %.idx1276 to i32, !dbg !12
  %dma.buffer.offset1230 = shl nuw nsw i32 %.idx1276.tr, 1, !dbg !12
  %157 = insertelement <2 x i64> poison, i64 %156, i64 0, !dbg !12
  %158 = bitcast <2 x i64> %157 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1231 = shufflevector <4 x i32> %158, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1232 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %155), !dbg !12
  %159 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1232, i32 %dma.buffer.offset1230, <4 x i32> %dma.buffer.rsrc1231) #10, !dbg !12
  %address.local409 = lshr i64 %address.local366, 2, !dbg !14
  %address.local412 = shl nuw nsw i64 %address.local366, 1, !dbg !14
  %address.local413 = and i64 %address.local412, 24, !dbg !14
  %address.local415 = shl i64 %address.local366, 62, !dbg !14
  %address.local416 = ashr exact i64 %address.local415, 57, !dbg !14
  %address.local417 = and i64 %address.local416, -64, !dbg !14
  %address.local419 = lshr i64 %address.local366, 2, !dbg !10
  %address.local420 = and i64 %address.local419, 12, !dbg !10
  %address.local422 = and i64 %address.local366, 63, !dbg !10
  %address.local423 = lshr i64 %address.local422, 5, !dbg !10
  %address.local424 = mul nuw nsw i64 %address.local423, 1016, !dbg !10
  %.idx = mul nuw nsw i64 %address.local366, 320, !dbg !14
  %160 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !14
  %.idx1277 = mul nsw i64 %address.local409, -1280, !dbg !14
  %161 = getelementptr i8, ptr addrspace(3) %160, i64 %.idx1277, !dbg !14
  %162 = getelementptr [2 x i8], ptr addrspace(3) %161, i64 %address.local413, !dbg !14
  %163 = getelementptr [2 x i8], ptr addrspace(3) %162, i64 %address.local417, !dbg !14
  %164 = getelementptr [2 x i8], ptr addrspace(3) %163, i64 %address.local420, !dbg !14
  %address.local426 = getelementptr [2 x i8], ptr addrspace(3) %164, i64 %address.local424, !dbg !14
  %165 = load <4 x half>, ptr addrspace(3) %address.local426, align 8, !dbg !14
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %165, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1077 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4096, !dbg !14
  %167 = load <4 x half>, ptr addrspace(3) %lds.const.gep1077, align 8, !dbg !14
  %address.local437 = shl nuw nsw i32 %address.thread365, 1, !dbg !14
  %address.local438 = and i32 %address.local437, 24, !dbg !14
  %address.local439 = shl nuw nsw i32 %address.thread365, 8, !dbg !14
  %address.local440 = and i32 %address.local439, 768, !dbg !14
  %address.local441 = or disjoint i32 %address.local438, %address.local440, !dbg !14
  %address.local442 = shl nuw nsw i32 %address.thread365, 5, !dbg !14
  %address.local443 = and i32 %address.local442, 96, !dbg !14
  %address.local444 = or disjoint i32 %address.local441, %address.local443, !dbg !14
  %address.local445 = zext nneg i32 %address.local444 to i64, !dbg !14
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %124) #9, !dbg !14, !frisk.register_lifetimes !1
  %address.local446 = lshr i32 %address.thread365, 1, !dbg !14
  %address.local447 = and i32 %address.local446, 1, !dbg !14
  %address.local448 = zext nneg i32 %address.local447 to i64, !dbg !14
  %address.local449 = add nsw i64 %address.base, %address.local448, !dbg !14
  %address.local450 = shl i64 %address.local449, 6, !dbg !14
  %address.local451 = sub i64 %address.local445, %address.local450, !dbg !14
  %168 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local451, !dbg !14
  %169 = getelementptr [2 x i8], ptr addrspace(3) %168, i64 %address.local420, !dbg !14
  %170 = getelementptr [2 x i8], ptr addrspace(3) %169, i64 %address.local424, !dbg !14
  %address.local455 = getelementptr i8, ptr addrspace(3) %170, i64 64, !dbg !14
  %171 = load <4 x half>, ptr addrspace(3) %address.local455, align 8, !dbg !14
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %167, <4 x float> %166) #9, !dbg !14
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %171, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1078 = getelementptr i8, ptr addrspace(3) %170, i64 4160, !dbg !14
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1078, align 8, !dbg !14
  %lds.const.gep1079 = getelementptr i8, ptr addrspace(3) %address.local426, i64 128, !dbg !14
  %175 = load <4 x half>, ptr addrspace(3) %lds.const.gep1079, align 8, !dbg !14
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %174, <4 x float> %173) #9, !dbg !14
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %175, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1080 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4224, !dbg !14
  %178 = load <4 x half>, ptr addrspace(3) %lds.const.gep1080, align 8, !dbg !14
  %lds.const.gep1081 = getelementptr i8, ptr addrspace(3) %170, i64 192, !dbg !14
  %179 = load <4 x half>, ptr addrspace(3) %lds.const.gep1081, align 8, !dbg !14
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %178, <4 x float> %177) #9, !dbg !14
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %179, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1082 = getelementptr i8, ptr addrspace(3) %170, i64 4288, !dbg !14
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1082, align 8, !dbg !14
  %lds.const.gep1083 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1024, !dbg !14
  %183 = load <4 x half>, ptr addrspace(3) %lds.const.gep1083, align 8, !dbg !14
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %182, <4 x float> %181) #9, !dbg !14
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %183, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1084 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5120, !dbg !14
  %186 = load <4 x half>, ptr addrspace(3) %lds.const.gep1084, align 8, !dbg !14
  %lds.const.gep1085 = getelementptr i8, ptr addrspace(3) %170, i64 1088, !dbg !14
  %187 = load <4 x half>, ptr addrspace(3) %lds.const.gep1085, align 8, !dbg !14
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %186, <4 x float> %185) #9, !dbg !14
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %187, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1086 = getelementptr i8, ptr addrspace(3) %170, i64 5184, !dbg !14
  %190 = load <4 x half>, ptr addrspace(3) %lds.const.gep1086, align 8, !dbg !14
  %lds.const.gep1087 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1152, !dbg !14
  %191 = load <4 x half>, ptr addrspace(3) %lds.const.gep1087, align 8, !dbg !14
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %190, <4 x float> %189) #9, !dbg !14
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %191, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1088 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5248, !dbg !14
  %194 = load <4 x half>, ptr addrspace(3) %lds.const.gep1088, align 8, !dbg !14
  %lds.const.gep1089 = getelementptr i8, ptr addrspace(3) %170, i64 1216, !dbg !14
  %195 = load <4 x half>, ptr addrspace(3) %lds.const.gep1089, align 8, !dbg !14
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %194, <4 x float> %193) #9, !dbg !14
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %195, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1090 = getelementptr i8, ptr addrspace(3) %170, i64 5312, !dbg !14
  %198 = load <4 x half>, ptr addrspace(3) %lds.const.gep1090, align 8, !dbg !14
  %199 = load <4 x half>, ptr addrspace(3) %address.local426, align 8, !dbg !14
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %198, <4 x float> %197) #9, !dbg !14
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %199, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1091 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4096, !dbg !14
  %202 = load <4 x half>, ptr addrspace(3) %lds.const.gep1091, align 8, !dbg !14
  %203 = load <4 x half>, ptr addrspace(3) %address.local455, align 8, !dbg !14
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %202, <4 x float> %201) #9, !dbg !14
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %203, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1092 = getelementptr i8, ptr addrspace(3) %170, i64 4160, !dbg !14
  %206 = load <4 x half>, ptr addrspace(3) %lds.const.gep1092, align 8, !dbg !14
  %lds.const.gep1093 = getelementptr i8, ptr addrspace(3) %address.local426, i64 128, !dbg !14
  %207 = load <4 x half>, ptr addrspace(3) %lds.const.gep1093, align 8, !dbg !14
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %206, <4 x float> %205) #9, !dbg !14
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %207, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1094 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4224, !dbg !14
  %210 = load <4 x half>, ptr addrspace(3) %lds.const.gep1094, align 8, !dbg !14
  %lds.const.gep1095 = getelementptr i8, ptr addrspace(3) %170, i64 192, !dbg !14
  %211 = load <4 x half>, ptr addrspace(3) %lds.const.gep1095, align 8, !dbg !14
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %210, <4 x float> %209) #9, !dbg !14
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %211, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1096 = getelementptr i8, ptr addrspace(3) %170, i64 4288, !dbg !14
  %214 = load <4 x half>, ptr addrspace(3) %lds.const.gep1096, align 8, !dbg !14
  %lds.const.gep1097 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1024, !dbg !14
  %215 = load <4 x half>, ptr addrspace(3) %lds.const.gep1097, align 8, !dbg !14
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %214, <4 x float> %213) #9, !dbg !14
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %215, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1098 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5120, !dbg !14
  %218 = load <4 x half>, ptr addrspace(3) %lds.const.gep1098, align 8, !dbg !14
  %lds.const.gep1099 = getelementptr i8, ptr addrspace(3) %170, i64 1088, !dbg !14
  %219 = load <4 x half>, ptr addrspace(3) %lds.const.gep1099, align 8, !dbg !14
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %218, <4 x float> %217) #9, !dbg !14
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %219, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1100 = getelementptr i8, ptr addrspace(3) %170, i64 5184, !dbg !14
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1100, align 8, !dbg !14
  %lds.const.gep1101 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1152, !dbg !14
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1101, align 8, !dbg !14
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %222, <4 x float> %221) #9, !dbg !14
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %223, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1102 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5248, !dbg !14
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep1102, align 8, !dbg !14
  %lds.const.gep1103 = getelementptr i8, ptr addrspace(3) %170, i64 1216, !dbg !14
  %227 = load <4 x half>, ptr addrspace(3) %lds.const.gep1103, align 8, !dbg !14
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %226, <4 x float> %225) #9, !dbg !14
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %227, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1104 = getelementptr i8, ptr addrspace(3) %170, i64 5312, !dbg !14
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep1104, align 8, !dbg !14
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %230, <4 x float> %229) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %232 = or disjoint i64 %138, 64, !dbg !12
  %233 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !12
  %234 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx12781279 = or disjoint i64 %232, %address.local381, !dbg !12
  %.idx12781279.tr = trunc nuw i64 %.idx12781279 to i32, !dbg !12
  %dma.buffer.offset1227 = shl nuw nsw i32 %.idx12781279.tr, 1, !dbg !12
  %235 = insertelement <2 x i64> poison, i64 %234, i64 0, !dbg !12
  %236 = bitcast <2 x i64> %235 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1228 = shufflevector <4 x i32> %236, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1229 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %233), !dbg !12
  %237 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1229, i32 %dma.buffer.offset1227, <4 x i32> %dma.buffer.rsrc1228) #10, !dbg !12
  %238 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !12
  %239 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx12801281 = or disjoint i64 %232, %address.local398, !dbg !12
  %.idx12801281.tr = trunc nuw i64 %.idx12801281 to i32, !dbg !12
  %dma.buffer.offset1224 = shl nuw nsw i32 %.idx12801281.tr, 1, !dbg !12
  %240 = insertelement <2 x i64> poison, i64 %239, i64 0, !dbg !12
  %241 = bitcast <2 x i64> %240 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1225 = shufflevector <4 x i32> %241, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %238), !dbg !12
  %242 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1226, i32 %dma.buffer.offset1224, <4 x i32> %dma.buffer.rsrc1225) #10, !dbg !12
  %lds.const.gep1105 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8192, !dbg !14
  %243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1105, align 8, !dbg !14
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %243, <4 x float> %172) #9, !dbg !14
  %lds.const.gep1106 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12288, !dbg !14
  %245 = load <4 x half>, ptr addrspace(3) %lds.const.gep1106, align 8, !dbg !14
  %lds.const.gep1107 = getelementptr i8, ptr addrspace(3) %170, i64 8256, !dbg !14
  %246 = load <4 x half>, ptr addrspace(3) %lds.const.gep1107, align 8, !dbg !14
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %245, <4 x float> %244) #9, !dbg !14
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %246, <4 x float> %176) #9, !dbg !14
  %lds.const.gep1108 = getelementptr i8, ptr addrspace(3) %170, i64 12352, !dbg !14
  %249 = load <4 x half>, ptr addrspace(3) %lds.const.gep1108, align 8, !dbg !14
  %lds.const.gep1109 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8320, !dbg !14
  %250 = load <4 x half>, ptr addrspace(3) %lds.const.gep1109, align 8, !dbg !14
  %251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %249, <4 x float> %248) #9, !dbg !14
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %250, <4 x float> %180) #9, !dbg !14
  %lds.const.gep1110 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12416, !dbg !14
  %253 = load <4 x half>, ptr addrspace(3) %lds.const.gep1110, align 8, !dbg !14
  %lds.const.gep1111 = getelementptr i8, ptr addrspace(3) %170, i64 8384, !dbg !14
  %254 = load <4 x half>, ptr addrspace(3) %lds.const.gep1111, align 8, !dbg !14
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %253, <4 x float> %252) #9, !dbg !14
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %254, <4 x float> %184) #9, !dbg !14
  %lds.const.gep1112 = getelementptr i8, ptr addrspace(3) %170, i64 12480, !dbg !14
  %257 = load <4 x half>, ptr addrspace(3) %lds.const.gep1112, align 8, !dbg !14
  %lds.const.gep1113 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9216, !dbg !14
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep1113, align 8, !dbg !14
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %257, <4 x float> %256) #9, !dbg !14
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %258, <4 x float> %188) #9, !dbg !14
  %lds.const.gep1114 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13312, !dbg !14
  %261 = load <4 x half>, ptr addrspace(3) %lds.const.gep1114, align 8, !dbg !14
  %lds.const.gep1115 = getelementptr i8, ptr addrspace(3) %170, i64 9280, !dbg !14
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep1115, align 8, !dbg !14
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %261, <4 x float> %260) #9, !dbg !14
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %262, <4 x float> %192) #9, !dbg !14
  %lds.const.gep1116 = getelementptr i8, ptr addrspace(3) %170, i64 13376, !dbg !14
  %265 = load <4 x half>, ptr addrspace(3) %lds.const.gep1116, align 8, !dbg !14
  %lds.const.gep1117 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9344, !dbg !14
  %266 = load <4 x half>, ptr addrspace(3) %lds.const.gep1117, align 8, !dbg !14
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %265, <4 x float> %264) #9, !dbg !14
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %266, <4 x float> %196) #9, !dbg !14
  %lds.const.gep1118 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13440, !dbg !14
  %269 = load <4 x half>, ptr addrspace(3) %lds.const.gep1118, align 8, !dbg !14
  %lds.const.gep1119 = getelementptr i8, ptr addrspace(3) %170, i64 9408, !dbg !14
  %270 = load <4 x half>, ptr addrspace(3) %lds.const.gep1119, align 8, !dbg !14
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %269, <4 x float> %268) #9, !dbg !14
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %270, <4 x float> %200) #9, !dbg !14
  %lds.const.gep1120 = getelementptr i8, ptr addrspace(3) %170, i64 13504, !dbg !14
  %273 = load <4 x half>, ptr addrspace(3) %lds.const.gep1120, align 8, !dbg !14
  %lds.const.gep1121 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8192, !dbg !14
  %274 = load <4 x half>, ptr addrspace(3) %lds.const.gep1121, align 8, !dbg !14
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %273, <4 x float> %272) #9, !dbg !14
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %274, <4 x float> %204) #9, !dbg !14
  %lds.const.gep1122 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12288, !dbg !14
  %277 = load <4 x half>, ptr addrspace(3) %lds.const.gep1122, align 8, !dbg !14
  %lds.const.gep1123 = getelementptr i8, ptr addrspace(3) %170, i64 8256, !dbg !14
  %278 = load <4 x half>, ptr addrspace(3) %lds.const.gep1123, align 8, !dbg !14
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %277, <4 x float> %276) #9, !dbg !14
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %278, <4 x float> %208) #9, !dbg !14
  %lds.const.gep1124 = getelementptr i8, ptr addrspace(3) %170, i64 12352, !dbg !14
  %281 = load <4 x half>, ptr addrspace(3) %lds.const.gep1124, align 8, !dbg !14
  %lds.const.gep1125 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8320, !dbg !14
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep1125, align 8, !dbg !14
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %281, <4 x float> %280) #9, !dbg !14
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %282, <4 x float> %212) #9, !dbg !14
  %lds.const.gep1126 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12416, !dbg !14
  %285 = load <4 x half>, ptr addrspace(3) %lds.const.gep1126, align 8, !dbg !14
  %lds.const.gep1127 = getelementptr i8, ptr addrspace(3) %170, i64 8384, !dbg !14
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep1127, align 8, !dbg !14
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %285, <4 x float> %284) #9, !dbg !14
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %286, <4 x float> %216) #9, !dbg !14
  %lds.const.gep1128 = getelementptr i8, ptr addrspace(3) %170, i64 12480, !dbg !14
  %289 = load <4 x half>, ptr addrspace(3) %lds.const.gep1128, align 8, !dbg !14
  %lds.const.gep1129 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9216, !dbg !14
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep1129, align 8, !dbg !14
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %289, <4 x float> %288) #9, !dbg !14
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %290, <4 x float> %220) #9, !dbg !14
  %lds.const.gep1130 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13312, !dbg !14
  %293 = load <4 x half>, ptr addrspace(3) %lds.const.gep1130, align 8, !dbg !14
  %lds.const.gep1131 = getelementptr i8, ptr addrspace(3) %170, i64 9280, !dbg !14
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1131, align 8, !dbg !14
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %293, <4 x float> %292) #9, !dbg !14
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %294, <4 x float> %224) #9, !dbg !14
  %lds.const.gep1132 = getelementptr i8, ptr addrspace(3) %170, i64 13376, !dbg !14
  %297 = load <4 x half>, ptr addrspace(3) %lds.const.gep1132, align 8, !dbg !14
  %lds.const.gep1133 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9344, !dbg !14
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep1133, align 8, !dbg !14
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %297, <4 x float> %296) #9, !dbg !14
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %298, <4 x float> %228) #9, !dbg !14
  %lds.const.gep1134 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13440, !dbg !14
  %301 = load <4 x half>, ptr addrspace(3) %lds.const.gep1134, align 8, !dbg !14
  %lds.const.gep1135 = getelementptr i8, ptr addrspace(3) %170, i64 9408, !dbg !14
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep1135, align 8, !dbg !14
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %301, <4 x float> %300) #9, !dbg !14
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %302, <4 x float> %231) #9, !dbg !14
  %lds.const.gep1136 = getelementptr i8, ptr addrspace(3) %170, i64 13504, !dbg !14
  %305 = load <4 x half>, ptr addrspace(3) %lds.const.gep1136, align 8, !dbg !14
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %305, <4 x float> %304) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %307 = or disjoint i64 %138, 96, !dbg !12
  %308 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local402), !dbg !12
  %309 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx12821283 = or disjoint i64 %307, %address.local381, !dbg !12
  %.idx12821283.tr = trunc nuw i64 %.idx12821283 to i32, !dbg !12
  %dma.buffer.offset1221 = shl nuw nsw i32 %.idx12821283.tr, 1, !dbg !12
  %310 = insertelement <2 x i64> poison, i64 %309, i64 0, !dbg !12
  %311 = bitcast <2 x i64> %310 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1222 = shufflevector <4 x i32> %311, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1223 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %308), !dbg !12
  %312 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1223, i32 %dma.buffer.offset1221, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !12
  %313 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local406), !dbg !12
  %314 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx12841285 = or disjoint i64 %307, %address.local398, !dbg !12
  %.idx12841285.tr = trunc nuw i64 %.idx12841285 to i32, !dbg !12
  %dma.buffer.offset1218 = shl nuw nsw i32 %.idx12841285.tr, 1, !dbg !12
  %315 = insertelement <2 x i64> poison, i64 %314, i64 0, !dbg !12
  %316 = bitcast <2 x i64> %315 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1219 = shufflevector <4 x i32> %316, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1220 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %313), !dbg !12
  %317 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1220, i32 %dma.buffer.offset1218, <4 x i32> %dma.buffer.rsrc1219) #10, !dbg !12
  %318 = load <4 x half>, ptr addrspace(3) %address.local426, align 8, !dbg !14
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %318, <4 x float> %247) #9, !dbg !14
  %lds.const.gep1137 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4096, !dbg !14
  %320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1137, align 8, !dbg !14
  %321 = load <4 x half>, ptr addrspace(3) %address.local455, align 8, !dbg !14
  %322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %320, <4 x float> %319) #9, !dbg !14
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %321, <4 x float> %251) #9, !dbg !14
  %lds.const.gep1138 = getelementptr i8, ptr addrspace(3) %170, i64 4160, !dbg !14
  %324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1138, align 8, !dbg !14
  %lds.const.gep1139 = getelementptr i8, ptr addrspace(3) %address.local426, i64 128, !dbg !14
  %325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1139, align 8, !dbg !14
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %324, <4 x float> %323) #9, !dbg !14
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %325, <4 x float> %255) #9, !dbg !14
  %lds.const.gep1140 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4224, !dbg !14
  %328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1140, align 8, !dbg !14
  %lds.const.gep1141 = getelementptr i8, ptr addrspace(3) %170, i64 192, !dbg !14
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1141, align 8, !dbg !14
  %330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %328, <4 x float> %327) #9, !dbg !14
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %329, <4 x float> %259) #9, !dbg !14
  %lds.const.gep1142 = getelementptr i8, ptr addrspace(3) %170, i64 4288, !dbg !14
  %332 = load <4 x half>, ptr addrspace(3) %lds.const.gep1142, align 8, !dbg !14
  %lds.const.gep1143 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1024, !dbg !14
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1143, align 8, !dbg !14
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %332, <4 x float> %331) #9, !dbg !14
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %333, <4 x float> %263) #9, !dbg !14
  %lds.const.gep1144 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5120, !dbg !14
  %336 = load <4 x half>, ptr addrspace(3) %lds.const.gep1144, align 8, !dbg !14
  %lds.const.gep1145 = getelementptr i8, ptr addrspace(3) %170, i64 1088, !dbg !14
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1145, align 8, !dbg !14
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %336, <4 x float> %335) #9, !dbg !14
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %337, <4 x float> %267) #9, !dbg !14
  %lds.const.gep1146 = getelementptr i8, ptr addrspace(3) %170, i64 5184, !dbg !14
  %340 = load <4 x half>, ptr addrspace(3) %lds.const.gep1146, align 8, !dbg !14
  %lds.const.gep1147 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1152, !dbg !14
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1147, align 8, !dbg !14
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %340, <4 x float> %339) #9, !dbg !14
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %341, <4 x float> %271) #9, !dbg !14
  %lds.const.gep1148 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5248, !dbg !14
  %344 = load <4 x half>, ptr addrspace(3) %lds.const.gep1148, align 8, !dbg !14
  %lds.const.gep1149 = getelementptr i8, ptr addrspace(3) %170, i64 1216, !dbg !14
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1149, align 8, !dbg !14
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %344, <4 x float> %343) #9, !dbg !14
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %345, <4 x float> %275) #9, !dbg !14
  %lds.const.gep1150 = getelementptr i8, ptr addrspace(3) %170, i64 5312, !dbg !14
  %348 = load <4 x half>, ptr addrspace(3) %lds.const.gep1150, align 8, !dbg !14
  %349 = load <4 x half>, ptr addrspace(3) %address.local426, align 8, !dbg !14
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %348, <4 x float> %347) #9, !dbg !14
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %349, <4 x float> %279) #9, !dbg !14
  %lds.const.gep1151 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4096, !dbg !14
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep1151, align 8, !dbg !14
  %353 = load <4 x half>, ptr addrspace(3) %address.local455, align 8, !dbg !14
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %352, <4 x float> %351) #9, !dbg !14
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %353, <4 x float> %283) #9, !dbg !14
  %lds.const.gep1152 = getelementptr i8, ptr addrspace(3) %170, i64 4160, !dbg !14
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1152, align 8, !dbg !14
  %lds.const.gep1153 = getelementptr i8, ptr addrspace(3) %address.local426, i64 128, !dbg !14
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1153, align 8, !dbg !14
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %356, <4 x float> %355) #9, !dbg !14
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %357, <4 x float> %287) #9, !dbg !14
  %lds.const.gep1154 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4224, !dbg !14
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1154, align 8, !dbg !14
  %lds.const.gep1155 = getelementptr i8, ptr addrspace(3) %170, i64 192, !dbg !14
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1155, align 8, !dbg !14
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %360, <4 x float> %359) #9, !dbg !14
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %361, <4 x float> %291) #9, !dbg !14
  %lds.const.gep1156 = getelementptr i8, ptr addrspace(3) %170, i64 4288, !dbg !14
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep1156, align 8, !dbg !14
  %lds.const.gep1157 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1024, !dbg !14
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1157, align 8, !dbg !14
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %364, <4 x float> %363) #9, !dbg !14
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %365, <4 x float> %295) #9, !dbg !14
  %lds.const.gep1158 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5120, !dbg !14
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep1158, align 8, !dbg !14
  %lds.const.gep1159 = getelementptr i8, ptr addrspace(3) %170, i64 1088, !dbg !14
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1159, align 8, !dbg !14
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %368, <4 x float> %367) #9, !dbg !14
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %369, <4 x float> %299) #9, !dbg !14
  %lds.const.gep1160 = getelementptr i8, ptr addrspace(3) %170, i64 5184, !dbg !14
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep1160, align 8, !dbg !14
  %lds.const.gep1161 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1152, !dbg !14
  %373 = load <4 x half>, ptr addrspace(3) %lds.const.gep1161, align 8, !dbg !14
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %372, <4 x float> %371) #9, !dbg !14
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %373, <4 x float> %303) #9, !dbg !14
  %lds.const.gep1162 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5248, !dbg !14
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep1162, align 8, !dbg !14
  %lds.const.gep1163 = getelementptr i8, ptr addrspace(3) %170, i64 1216, !dbg !14
  %377 = load <4 x half>, ptr addrspace(3) %lds.const.gep1163, align 8, !dbg !14
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %376, <4 x float> %375) #9, !dbg !14
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %377, <4 x float> %306) #9, !dbg !14
  %lds.const.gep1164 = getelementptr i8, ptr addrspace(3) %170, i64 5312, !dbg !14
  %380 = load <4 x half>, ptr addrspace(3) %lds.const.gep1164, align 8, !dbg !14
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %380, <4 x float> %379) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %382 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !15
  %address.local784 = shl nuw nsw i64 %address.local366, 5, !dbg !15
  %address.local785 = and i64 %address.local784, 8064, !dbg !15
  %address.local786 = shl nuw nsw i64 %address.local366, 3, !dbg !15
  %address.local787 = and i64 %address.local786, 24, !dbg !15
  %address.local788 = or disjoint i64 %address.local785, %address.local787, !dbg !15
  %383 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx12861287 = or disjoint i64 %138, %address.local788, !dbg !15
  %.idx12861287.tr = trunc nuw i64 %.idx12861287 to i32, !dbg !15
  %dma.buffer.offset1215 = shl nuw nsw i32 %.idx12861287.tr, 1, !dbg !15
  %384 = insertelement <2 x i64> poison, i64 %383, i64 0, !dbg !15
  %385 = bitcast <2 x i64> %384 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1216 = shufflevector <4 x i32> %385, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1217 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %382), !dbg !15
  %386 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1217, i32 %dma.buffer.offset1215, <4 x i32> %dma.buffer.rsrc1216) #10, !dbg !15
  %387 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !15
  %address.local790 = shl nuw nsw i64 %address.local383, 5, !dbg !15
  %address.local791 = and i64 %address.local790, 16256, !dbg !15
  %address.local792 = shl nuw nsw i64 %address.local366, 3, !dbg !15
  %address.local793 = and i64 %address.local792, 24, !dbg !15
  %address.local794 = or disjoint i64 %address.local791, %address.local793, !dbg !15
  %388 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx12881289 = or disjoint i64 %138, %address.local794, !dbg !15
  %.idx12881289.tr = trunc nuw i64 %.idx12881289 to i32, !dbg !15
  %dma.buffer.offset1212 = shl nuw nsw i32 %.idx12881289.tr, 1, !dbg !15
  %389 = insertelement <2 x i64> poison, i64 %388, i64 0, !dbg !15
  %390 = bitcast <2 x i64> %389 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1213 = shufflevector <4 x i32> %390, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1214 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %387), !dbg !15
  %391 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1214, i32 %dma.buffer.offset1212, <4 x i32> %dma.buffer.rsrc1213) #10, !dbg !15
  %lds.const.gep1165 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8192, !dbg !14
  %392 = load <4 x half>, ptr addrspace(3) %lds.const.gep1165, align 8, !dbg !14
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %392, <4 x float> %322) #9, !dbg !14
  %lds.const.gep1166 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12288, !dbg !14
  %394 = load <4 x half>, ptr addrspace(3) %lds.const.gep1166, align 8, !dbg !14
  %lds.const.gep1167 = getelementptr i8, ptr addrspace(3) %170, i64 8256, !dbg !14
  %395 = load <4 x half>, ptr addrspace(3) %lds.const.gep1167, align 8, !dbg !14
  %396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %394, <4 x float> %393) #9, !dbg !14
  %397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %395, <4 x float> %326) #9, !dbg !14
  %398 = extractelement <4 x float> %396, i64 0, !dbg !19
  %399 = call float @llvm.maxnum.f32(float %398, float 0xFFF0000000000000), !dbg !20
  %400 = extractelement <4 x float> %396, i64 1, !dbg !19
  %401 = call float @llvm.maxnum.f32(float %399, float %400), !dbg !20
  %402 = extractelement <4 x float> %396, i64 2, !dbg !19
  %403 = call float @llvm.maxnum.f32(float %401, float %402), !dbg !20
  %404 = extractelement <4 x float> %396, i64 3, !dbg !19
  %405 = call float @llvm.maxnum.f32(float %403, float %404), !dbg !20
  %lds.const.gep1168 = getelementptr i8, ptr addrspace(3) %170, i64 12352, !dbg !14
  %406 = load <4 x half>, ptr addrspace(3) %lds.const.gep1168, align 8, !dbg !14
  %lds.const.gep1169 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8320, !dbg !14
  %407 = load <4 x half>, ptr addrspace(3) %lds.const.gep1169, align 8, !dbg !14
  %408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %406, <4 x float> %397) #9, !dbg !14
  %409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %407, <4 x float> %330) #9, !dbg !14
  %410 = extractelement <4 x float> %408, i64 0, !dbg !19
  %411 = call float @llvm.maxnum.f32(float %405, float %410), !dbg !20
  %412 = extractelement <4 x float> %408, i64 1, !dbg !19
  %413 = call float @llvm.maxnum.f32(float %411, float %412), !dbg !20
  %414 = extractelement <4 x float> %408, i64 2, !dbg !19
  %415 = call float @llvm.maxnum.f32(float %413, float %414), !dbg !20
  %416 = extractelement <4 x float> %408, i64 3, !dbg !19
  %417 = call float @llvm.maxnum.f32(float %415, float %416), !dbg !20
  %lds.const.gep1170 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12416, !dbg !14
  %418 = load <4 x half>, ptr addrspace(3) %lds.const.gep1170, align 8, !dbg !14
  %lds.const.gep1171 = getelementptr i8, ptr addrspace(3) %170, i64 8384, !dbg !14
  %419 = load <4 x half>, ptr addrspace(3) %lds.const.gep1171, align 8, !dbg !14
  %420 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %418, <4 x float> %409) #9, !dbg !14
  %421 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %419, <4 x float> %334) #9, !dbg !14
  %422 = extractelement <4 x float> %420, i64 0, !dbg !19
  %423 = call float @llvm.maxnum.f32(float %417, float %422), !dbg !20
  %424 = extractelement <4 x float> %420, i64 1, !dbg !19
  %425 = call float @llvm.maxnum.f32(float %423, float %424), !dbg !20
  %426 = extractelement <4 x float> %420, i64 2, !dbg !19
  %427 = call float @llvm.maxnum.f32(float %425, float %426), !dbg !20
  %428 = extractelement <4 x float> %420, i64 3, !dbg !19
  %429 = call float @llvm.maxnum.f32(float %427, float %428), !dbg !20
  %lds.const.gep1172 = getelementptr i8, ptr addrspace(3) %170, i64 12480, !dbg !14
  %430 = load <4 x half>, ptr addrspace(3) %lds.const.gep1172, align 8, !dbg !14
  %lds.const.gep1173 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9216, !dbg !14
  %431 = load <4 x half>, ptr addrspace(3) %lds.const.gep1173, align 8, !dbg !14
  %432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %430, <4 x float> %421) #9, !dbg !14
  %433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %431, <4 x float> %338) #9, !dbg !14
  %434 = extractelement <4 x float> %432, i64 0, !dbg !19
  %435 = call float @llvm.maxnum.f32(float %429, float %434), !dbg !20
  %436 = extractelement <4 x float> %432, i64 1, !dbg !19
  %437 = call float @llvm.maxnum.f32(float %435, float %436), !dbg !20
  %438 = extractelement <4 x float> %432, i64 2, !dbg !19
  %439 = call float @llvm.maxnum.f32(float %437, float %438), !dbg !20
  %440 = extractelement <4 x float> %432, i64 3, !dbg !19
  %441 = call float @llvm.maxnum.f32(float %439, float %440), !dbg !20
  %lds.const.gep1174 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13312, !dbg !14
  %442 = load <4 x half>, ptr addrspace(3) %lds.const.gep1174, align 8, !dbg !14
  %lds.const.gep1175 = getelementptr i8, ptr addrspace(3) %170, i64 9280, !dbg !14
  %443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1175, align 8, !dbg !14
  %444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %442, <4 x float> %433) #9, !dbg !14
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %443, <4 x float> %342) #9, !dbg !14
  %446 = extractelement <4 x float> %444, i64 0, !dbg !19
  %447 = call float @llvm.maxnum.f32(float %441, float %446), !dbg !20
  %448 = extractelement <4 x float> %444, i64 1, !dbg !19
  %449 = call float @llvm.maxnum.f32(float %447, float %448), !dbg !20
  %450 = extractelement <4 x float> %444, i64 2, !dbg !19
  %451 = call float @llvm.maxnum.f32(float %449, float %450), !dbg !20
  %452 = extractelement <4 x float> %444, i64 3, !dbg !19
  %453 = call float @llvm.maxnum.f32(float %451, float %452), !dbg !20
  %lds.const.gep1176 = getelementptr i8, ptr addrspace(3) %170, i64 13376, !dbg !14
  %454 = load <4 x half>, ptr addrspace(3) %lds.const.gep1176, align 8, !dbg !14
  %lds.const.gep1177 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9344, !dbg !14
  %455 = load <4 x half>, ptr addrspace(3) %lds.const.gep1177, align 8, !dbg !14
  %456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %454, <4 x float> %445) #9, !dbg !14
  %457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %455, <4 x float> %346) #9, !dbg !14
  %458 = extractelement <4 x float> %456, i64 0, !dbg !19
  %459 = call float @llvm.maxnum.f32(float %453, float %458), !dbg !20
  %460 = extractelement <4 x float> %456, i64 1, !dbg !19
  %461 = call float @llvm.maxnum.f32(float %459, float %460), !dbg !20
  %462 = extractelement <4 x float> %456, i64 2, !dbg !19
  %463 = call float @llvm.maxnum.f32(float %461, float %462), !dbg !20
  %464 = extractelement <4 x float> %456, i64 3, !dbg !19
  %465 = call float @llvm.maxnum.f32(float %463, float %464), !dbg !20
  %lds.const.gep1178 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13440, !dbg !14
  %466 = load <4 x half>, ptr addrspace(3) %lds.const.gep1178, align 8, !dbg !14
  %lds.const.gep1179 = getelementptr i8, ptr addrspace(3) %170, i64 9408, !dbg !14
  %467 = load <4 x half>, ptr addrspace(3) %lds.const.gep1179, align 8, !dbg !14
  %468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %466, <4 x float> %457) #9, !dbg !14
  %469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %467, <4 x float> %350) #9, !dbg !14
  %470 = extractelement <4 x float> %468, i64 0, !dbg !19
  %471 = call float @llvm.maxnum.f32(float %465, float %470), !dbg !20
  %472 = extractelement <4 x float> %468, i64 1, !dbg !19
  %473 = call float @llvm.maxnum.f32(float %471, float %472), !dbg !20
  %474 = extractelement <4 x float> %468, i64 2, !dbg !19
  %475 = call float @llvm.maxnum.f32(float %473, float %474), !dbg !20
  %476 = extractelement <4 x float> %468, i64 3, !dbg !19
  %477 = call float @llvm.maxnum.f32(float %475, float %476), !dbg !20
  %lds.const.gep1180 = getelementptr i8, ptr addrspace(3) %170, i64 13504, !dbg !14
  %478 = load <4 x half>, ptr addrspace(3) %lds.const.gep1180, align 8, !dbg !14
  %lds.const.gep1181 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8192, !dbg !14
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep1181, align 8, !dbg !14
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %478, <4 x float> %469) #9, !dbg !14
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %479, <4 x float> %354) #9, !dbg !14
  %482 = extractelement <4 x float> %480, i64 0, !dbg !19
  %483 = call float @llvm.maxnum.f32(float %477, float %482), !dbg !20
  %484 = extractelement <4 x float> %480, i64 1, !dbg !19
  %485 = call float @llvm.maxnum.f32(float %483, float %484), !dbg !20
  %486 = extractelement <4 x float> %480, i64 2, !dbg !19
  %487 = call float @llvm.maxnum.f32(float %485, float %486), !dbg !20
  %488 = extractelement <4 x float> %480, i64 3, !dbg !19
  %489 = call float @llvm.maxnum.f32(float %487, float %488), !dbg !20
  %lds.const.gep1182 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12288, !dbg !14
  %490 = load <4 x half>, ptr addrspace(3) %lds.const.gep1182, align 8, !dbg !14
  %lds.const.gep1183 = getelementptr i8, ptr addrspace(3) %170, i64 8256, !dbg !14
  %491 = load <4 x half>, ptr addrspace(3) %lds.const.gep1183, align 8, !dbg !14
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %490, <4 x float> %481) #9, !dbg !14
  %493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %491, <4 x float> %358) #9, !dbg !14
  %494 = extractelement <4 x float> %492, i64 0, !dbg !19
  %495 = extractelement <4 x float> %492, i64 1, !dbg !19
  %496 = extractelement <4 x float> %492, i64 2, !dbg !19
  %497 = extractelement <4 x float> %492, i64 3, !dbg !19
  %lds.const.gep1184 = getelementptr i8, ptr addrspace(3) %170, i64 12352, !dbg !14
  %498 = load <4 x half>, ptr addrspace(3) %lds.const.gep1184, align 8, !dbg !14
  %lds.const.gep1185 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8320, !dbg !14
  %499 = load <4 x half>, ptr addrspace(3) %lds.const.gep1185, align 8, !dbg !14
  %500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %498, <4 x float> %493) #9, !dbg !14
  %501 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %499, <4 x float> %362) #9, !dbg !14
  %502 = extractelement <4 x float> %500, i64 0, !dbg !19
  %503 = extractelement <4 x float> %500, i64 1, !dbg !19
  %504 = extractelement <4 x float> %500, i64 2, !dbg !19
  %505 = extractelement <4 x float> %500, i64 3, !dbg !19
  %lds.const.gep1186 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12416, !dbg !14
  %506 = load <4 x half>, ptr addrspace(3) %lds.const.gep1186, align 8, !dbg !14
  %lds.const.gep1187 = getelementptr i8, ptr addrspace(3) %170, i64 8384, !dbg !14
  %507 = load <4 x half>, ptr addrspace(3) %lds.const.gep1187, align 8, !dbg !14
  %508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %506, <4 x float> %501) #9, !dbg !14
  %509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %507, <4 x float> %366) #9, !dbg !14
  %510 = extractelement <4 x float> %508, i64 0, !dbg !19
  %511 = extractelement <4 x float> %508, i64 1, !dbg !19
  %512 = extractelement <4 x float> %508, i64 2, !dbg !19
  %513 = extractelement <4 x float> %508, i64 3, !dbg !19
  %lds.const.gep1188 = getelementptr i8, ptr addrspace(3) %170, i64 12480, !dbg !14
  %514 = load <4 x half>, ptr addrspace(3) %lds.const.gep1188, align 8, !dbg !14
  %lds.const.gep1189 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9216, !dbg !14
  %515 = load <4 x half>, ptr addrspace(3) %lds.const.gep1189, align 8, !dbg !14
  %516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %514, <4 x float> %509) #9, !dbg !14
  %517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %515, <4 x float> %370) #9, !dbg !14
  %518 = extractelement <4 x float> %516, i64 0, !dbg !19
  %519 = extractelement <4 x float> %516, i64 1, !dbg !19
  %520 = extractelement <4 x float> %516, i64 2, !dbg !19
  %521 = extractelement <4 x float> %516, i64 3, !dbg !19
  %lds.const.gep1190 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13312, !dbg !14
  %522 = load <4 x half>, ptr addrspace(3) %lds.const.gep1190, align 8, !dbg !14
  %lds.const.gep1191 = getelementptr i8, ptr addrspace(3) %170, i64 9280, !dbg !14
  %523 = load <4 x half>, ptr addrspace(3) %lds.const.gep1191, align 8, !dbg !14
  %524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %522, <4 x float> %517) #9, !dbg !14
  %525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %523, <4 x float> %374) #9, !dbg !14
  %526 = extractelement <4 x float> %524, i64 0, !dbg !19
  %527 = extractelement <4 x float> %524, i64 1, !dbg !19
  %528 = extractelement <4 x float> %524, i64 2, !dbg !19
  %529 = extractelement <4 x float> %524, i64 3, !dbg !19
  %lds.const.gep1192 = getelementptr i8, ptr addrspace(3) %170, i64 13376, !dbg !14
  %530 = load <4 x half>, ptr addrspace(3) %lds.const.gep1192, align 8, !dbg !14
  %lds.const.gep1193 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9344, !dbg !14
  %531 = load <4 x half>, ptr addrspace(3) %lds.const.gep1193, align 8, !dbg !14
  %532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %530, <4 x float> %525) #9, !dbg !14
  %533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %531, <4 x float> %378) #9, !dbg !14
  %534 = extractelement <4 x float> %532, i64 0, !dbg !19
  %535 = extractelement <4 x float> %532, i64 1, !dbg !19
  %536 = extractelement <4 x float> %532, i64 2, !dbg !19
  %537 = extractelement <4 x float> %532, i64 3, !dbg !19
  %lds.const.gep1194 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13440, !dbg !14
  %538 = load <4 x half>, ptr addrspace(3) %lds.const.gep1194, align 8, !dbg !14
  %lds.const.gep1195 = getelementptr i8, ptr addrspace(3) %170, i64 9408, !dbg !14
  %539 = load <4 x half>, ptr addrspace(3) %lds.const.gep1195, align 8, !dbg !14
  %540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %538, <4 x float> %533) #9, !dbg !14
  %541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %539, <4 x float> %381) #9, !dbg !14
  %542 = extractelement <4 x float> %540, i64 0, !dbg !19
  %543 = extractelement <4 x float> %540, i64 1, !dbg !19
  %544 = extractelement <4 x float> %540, i64 2, !dbg !19
  %545 = extractelement <4 x float> %540, i64 3, !dbg !19
  %lds.const.gep1196 = getelementptr i8, ptr addrspace(3) %170, i64 13504, !dbg !14
  %546 = load <4 x half>, ptr addrspace(3) %lds.const.gep1196, align 8, !dbg !14
  %547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %546, <4 x float> %541) #9, !dbg !14
  %548 = extractelement <4 x float> %547, i64 0, !dbg !19
  %549 = extractelement <4 x float> %547, i64 1, !dbg !19
  %550 = extractelement <4 x float> %547, i64 2, !dbg !19
  %551 = extractelement <4 x float> %547, i64 3, !dbg !19
  %552 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %553 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %552), !dbg !20
  %554 = add nuw nsw i32 %553, 64, !dbg !20
  %555 = xor i32 %553, 16, !dbg !20
  %556 = and i32 %554, 192, !dbg !20
  %557 = icmp samesign ult i32 %555, %556, !dbg !20
  %558 = select i1 %557, i32 %555, i32 %553, !dbg !20
  %559 = shl nuw nsw i32 %558, 2, !dbg !20
  %560 = bitcast float %489 to i32, !dbg !20
  %561 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %560), !dbg !20
  %562 = bitcast i32 %561 to float, !dbg !20
  %563 = call float @llvm.maxnum.f32(float %489, float %562), !dbg !20
  %564 = xor i32 %553, 32, !dbg !20
  %565 = icmp samesign ult i32 %564, %556, !dbg !20
  %566 = select i1 %565, i32 %564, i32 %553, !dbg !20
  %567 = shl nuw nsw i32 %566, 2, !dbg !20
  %568 = bitcast float %563 to i32, !dbg !20
  %569 = call i32 @llvm.amdgcn.ds.bpermute(i32 %567, i32 %568), !dbg !20
  %570 = bitcast i32 %569 to float, !dbg !20
  %571 = call float @llvm.maxnum.f32(float %563, float %570), !dbg !20
  %572 = extractvalue [2 x <1 x float>] %129, 0, !dbg !20
  %573 = insertelement <1 x float> poison, float %571, i64 0, !dbg !20
  %574 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %573, <1 x float> %572), !dbg !20
  %575 = call float @llvm.maxnum.f32(float %494, float 0xFFF0000000000000), !dbg !20
  %576 = call float @llvm.maxnum.f32(float %575, float %495), !dbg !20
  %577 = call float @llvm.maxnum.f32(float %576, float %496), !dbg !20
  %578 = call float @llvm.maxnum.f32(float %577, float %497), !dbg !20
  %579 = call float @llvm.maxnum.f32(float %578, float %502), !dbg !20
  %580 = call float @llvm.maxnum.f32(float %579, float %503), !dbg !20
  %581 = call float @llvm.maxnum.f32(float %580, float %504), !dbg !20
  %582 = call float @llvm.maxnum.f32(float %581, float %505), !dbg !20
  %583 = call float @llvm.maxnum.f32(float %582, float %510), !dbg !20
  %584 = call float @llvm.maxnum.f32(float %583, float %511), !dbg !20
  %585 = call float @llvm.maxnum.f32(float %584, float %512), !dbg !20
  %586 = call float @llvm.maxnum.f32(float %585, float %513), !dbg !20
  %587 = call float @llvm.maxnum.f32(float %586, float %518), !dbg !20
  %588 = call float @llvm.maxnum.f32(float %587, float %519), !dbg !20
  %589 = call float @llvm.maxnum.f32(float %588, float %520), !dbg !20
  %590 = call float @llvm.maxnum.f32(float %589, float %521), !dbg !20
  %591 = call float @llvm.maxnum.f32(float %590, float %526), !dbg !20
  %592 = call float @llvm.maxnum.f32(float %591, float %527), !dbg !20
  %593 = call float @llvm.maxnum.f32(float %592, float %528), !dbg !20
  %594 = call float @llvm.maxnum.f32(float %593, float %529), !dbg !20
  %595 = call float @llvm.maxnum.f32(float %594, float %534), !dbg !20
  %596 = call float @llvm.maxnum.f32(float %595, float %535), !dbg !20
  %597 = call float @llvm.maxnum.f32(float %596, float %536), !dbg !20
  %598 = call float @llvm.maxnum.f32(float %597, float %537), !dbg !20
  %599 = call float @llvm.maxnum.f32(float %598, float %542), !dbg !20
  %600 = call float @llvm.maxnum.f32(float %599, float %543), !dbg !20
  %601 = call float @llvm.maxnum.f32(float %600, float %544), !dbg !20
  %602 = call float @llvm.maxnum.f32(float %601, float %545), !dbg !20
  %603 = call float @llvm.maxnum.f32(float %602, float %548), !dbg !20
  %604 = call float @llvm.maxnum.f32(float %603, float %549), !dbg !20
  %605 = call float @llvm.maxnum.f32(float %604, float %550), !dbg !20
  %606 = call float @llvm.maxnum.f32(float %605, float %551), !dbg !20
  %607 = bitcast float %606 to i32, !dbg !20
  %608 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %607), !dbg !20
  %609 = bitcast i32 %608 to float, !dbg !20
  %610 = call float @llvm.maxnum.f32(float %606, float %609), !dbg !20
  %611 = bitcast float %610 to i32, !dbg !20
  %612 = call i32 @llvm.amdgcn.ds.bpermute(i32 %567, i32 %611), !dbg !20
  %613 = bitcast i32 %612 to float, !dbg !20
  %614 = call float @llvm.maxnum.f32(float %610, float %613), !dbg !20
  %615 = extractvalue [2 x <1 x float>] %129, 1, !dbg !20
  %616 = insertelement <1 x float> poison, float %614, i64 0, !dbg !20
  %617 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %616, <1 x float> %615), !dbg !20
  %618 = shufflevector <1 x float> %574, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %619 = fsub <4 x float> %396, %618, !dbg !21
  %620 = fsub <4 x float> %408, %618, !dbg !21
  %621 = fsub <4 x float> %420, %618, !dbg !21
  %622 = fsub <4 x float> %432, %618, !dbg !21
  %623 = fsub <4 x float> %444, %618, !dbg !21
  %624 = fsub <4 x float> %456, %618, !dbg !21
  %625 = fsub <4 x float> %468, %618, !dbg !21
  %626 = fsub <4 x float> %480, %618, !dbg !21
  %627 = shufflevector <1 x float> %617, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %628 = fsub <4 x float> %492, %627, !dbg !21
  %629 = fsub <4 x float> %500, %627, !dbg !21
  %630 = fsub <4 x float> %508, %627, !dbg !21
  %631 = fsub <4 x float> %516, %627, !dbg !21
  %632 = fsub <4 x float> %524, %627, !dbg !21
  %633 = fsub <4 x float> %532, %627, !dbg !21
  %634 = fsub <4 x float> %540, %627, !dbg !21
  %635 = fsub <4 x float> %547, %627, !dbg !21
  %636 = fsub <1 x float> %572, %574, !dbg !22
  %637 = extractelement <1 x float> %636, i64 0, !dbg !23
  %638 = call float @llvm.amdgcn.exp2.f32(float %637), !dbg !23
  %639 = insertelement <4 x float> poison, float %638, i64 0, !dbg !28
  %640 = shufflevector <4 x float> %639, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %641 = extractvalue [2 x <1 x float>] %128, 0, !dbg !24
  %642 = extractelement <1 x float> %641, i64 0, !dbg !24
  %643 = fmul float %642, %638, !dbg !24
  %644 = fsub <1 x float> %615, %617, !dbg !22
  %645 = extractelement <1 x float> %644, i64 0, !dbg !23
  %646 = call float @llvm.amdgcn.exp2.f32(float %645), !dbg !23
  %647 = insertelement <4 x float> poison, float %646, i64 0, !dbg !28
  %648 = shufflevector <4 x float> %647, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %649 = extractvalue [2 x <1 x float>] %128, 1, !dbg !24
  %650 = extractelement <1 x float> %649, i64 0, !dbg !24
  %651 = fmul float %650, %646, !dbg !24
  %652 = extractelement <4 x float> %619, i64 0, !dbg !25
  %653 = call float @llvm.amdgcn.exp2.f32(float %652), !dbg !25
  %654 = insertelement <4 x float> poison, float %653, i64 0, !dbg !27
  %655 = extractelement <4 x float> %619, i64 1, !dbg !25
  %656 = call float @llvm.amdgcn.exp2.f32(float %655), !dbg !25
  %657 = insertelement <4 x float> %654, float %656, i64 1, !dbg !27
  %658 = fadd float %653, %656, !dbg !26
  %659 = extractelement <4 x float> %619, i64 2, !dbg !25
  %660 = call float @llvm.amdgcn.exp2.f32(float %659), !dbg !25
  %661 = insertelement <4 x float> %657, float %660, i64 2, !dbg !27
  %662 = fadd float %658, %660, !dbg !26
  %663 = extractelement <4 x float> %619, i64 3, !dbg !25
  %664 = call float @llvm.amdgcn.exp2.f32(float %663), !dbg !25
  %665 = insertelement <4 x float> %661, float %664, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %665) #9, !dbg !27, !frisk.register_lifetimes !1
  %666 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %666) #9, !dbg !26, !frisk.register_lifetimes !1
  %667 = fadd float %662, %664, !dbg !26
  %668 = extractelement <4 x float> %620, i64 0, !dbg !25
  %669 = call float @llvm.amdgcn.exp2.f32(float %668), !dbg !25
  %670 = insertelement <4 x float> poison, float %669, i64 0, !dbg !27
  %671 = fadd float %667, %669, !dbg !26
  %672 = extractelement <4 x float> %620, i64 1, !dbg !25
  %673 = call float @llvm.amdgcn.exp2.f32(float %672), !dbg !25
  %674 = insertelement <4 x float> %670, float %673, i64 1, !dbg !27
  %675 = fadd float %671, %673, !dbg !26
  %676 = extractelement <4 x float> %620, i64 2, !dbg !25
  %677 = call float @llvm.amdgcn.exp2.f32(float %676), !dbg !25
  %678 = insertelement <4 x float> %674, float %677, i64 2, !dbg !27
  %679 = fadd float %675, %677, !dbg !26
  %680 = extractelement <4 x float> %620, i64 3, !dbg !25
  %681 = call float @llvm.amdgcn.exp2.f32(float %680), !dbg !25
  %682 = insertelement <4 x float> %678, float %681, i64 3, !dbg !27
  %math.result241 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %682) #9, !dbg !27, !frisk.register_lifetimes !1
  %683 = fptrunc <4 x float> %math.result241 to <4 x half>, !dbg !27
  %math.packed242 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %683) #9, !dbg !26, !frisk.register_lifetimes !1
  %684 = fadd float %679, %681, !dbg !26
  %685 = extractelement <4 x float> %621, i64 0, !dbg !25
  %686 = call float @llvm.amdgcn.exp2.f32(float %685), !dbg !25
  %687 = insertelement <4 x float> poison, float %686, i64 0, !dbg !27
  %688 = fadd float %684, %686, !dbg !26
  %689 = extractelement <4 x float> %621, i64 1, !dbg !25
  %690 = call float @llvm.amdgcn.exp2.f32(float %689), !dbg !25
  %691 = insertelement <4 x float> %687, float %690, i64 1, !dbg !27
  %692 = fadd float %688, %690, !dbg !26
  %693 = extractelement <4 x float> %621, i64 2, !dbg !25
  %694 = call float @llvm.amdgcn.exp2.f32(float %693), !dbg !25
  %695 = insertelement <4 x float> %691, float %694, i64 2, !dbg !27
  %696 = fadd float %692, %694, !dbg !26
  %697 = extractelement <4 x float> %621, i64 3, !dbg !25
  %698 = call float @llvm.amdgcn.exp2.f32(float %697), !dbg !25
  %699 = insertelement <4 x float> %695, float %698, i64 3, !dbg !27
  %math.result243 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %699) #9, !dbg !27, !frisk.register_lifetimes !1
  %700 = fptrunc <4 x float> %math.result243 to <4 x half>, !dbg !27
  %math.packed244 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %700) #9, !dbg !26, !frisk.register_lifetimes !1
  %701 = fadd float %696, %698, !dbg !26
  %702 = extractelement <4 x float> %622, i64 0, !dbg !25
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !25
  %704 = insertelement <4 x float> poison, float %703, i64 0, !dbg !27
  %705 = fadd float %701, %703, !dbg !26
  %706 = extractelement <4 x float> %622, i64 1, !dbg !25
  %707 = call float @llvm.amdgcn.exp2.f32(float %706), !dbg !25
  %708 = insertelement <4 x float> %704, float %707, i64 1, !dbg !27
  %709 = fadd float %705, %707, !dbg !26
  %710 = extractelement <4 x float> %622, i64 2, !dbg !25
  %711 = call float @llvm.amdgcn.exp2.f32(float %710), !dbg !25
  %712 = insertelement <4 x float> %708, float %711, i64 2, !dbg !27
  %713 = fadd float %709, %711, !dbg !26
  %714 = extractelement <4 x float> %622, i64 3, !dbg !25
  %715 = call float @llvm.amdgcn.exp2.f32(float %714), !dbg !25
  %716 = insertelement <4 x float> %712, float %715, i64 3, !dbg !27
  %math.result245 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %716) #9, !dbg !27, !frisk.register_lifetimes !1
  %717 = fptrunc <4 x float> %math.result245 to <4 x half>, !dbg !27
  %math.packed246 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %717) #9, !dbg !26, !frisk.register_lifetimes !1
  %718 = fadd float %713, %715, !dbg !26
  %719 = extractelement <4 x float> %623, i64 0, !dbg !25
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !25
  %721 = insertelement <4 x float> poison, float %720, i64 0, !dbg !27
  %722 = fadd float %718, %720, !dbg !26
  %723 = extractelement <4 x float> %623, i64 1, !dbg !25
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !25
  %725 = insertelement <4 x float> %721, float %724, i64 1, !dbg !27
  %726 = fadd float %722, %724, !dbg !26
  %727 = extractelement <4 x float> %623, i64 2, !dbg !25
  %728 = call float @llvm.amdgcn.exp2.f32(float %727), !dbg !25
  %729 = insertelement <4 x float> %725, float %728, i64 2, !dbg !27
  %730 = fadd float %726, %728, !dbg !26
  %731 = extractelement <4 x float> %623, i64 3, !dbg !25
  %732 = call float @llvm.amdgcn.exp2.f32(float %731), !dbg !25
  %733 = insertelement <4 x float> %729, float %732, i64 3, !dbg !27
  %math.result247 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %733) #9, !dbg !27, !frisk.register_lifetimes !1
  %734 = fptrunc <4 x float> %math.result247 to <4 x half>, !dbg !27
  %math.packed248 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %734) #9, !dbg !26, !frisk.register_lifetimes !1
  %735 = fadd float %730, %732, !dbg !26
  %736 = extractelement <4 x float> %624, i64 0, !dbg !25
  %737 = call float @llvm.amdgcn.exp2.f32(float %736), !dbg !25
  %738 = insertelement <4 x float> poison, float %737, i64 0, !dbg !27
  %739 = fadd float %735, %737, !dbg !26
  %740 = extractelement <4 x float> %624, i64 1, !dbg !25
  %741 = call float @llvm.amdgcn.exp2.f32(float %740), !dbg !25
  %742 = insertelement <4 x float> %738, float %741, i64 1, !dbg !27
  %743 = fadd float %739, %741, !dbg !26
  %744 = extractelement <4 x float> %624, i64 2, !dbg !25
  %745 = call float @llvm.amdgcn.exp2.f32(float %744), !dbg !25
  %746 = insertelement <4 x float> %742, float %745, i64 2, !dbg !27
  %747 = fadd float %743, %745, !dbg !26
  %748 = extractelement <4 x float> %624, i64 3, !dbg !25
  %749 = call float @llvm.amdgcn.exp2.f32(float %748), !dbg !25
  %750 = insertelement <4 x float> %746, float %749, i64 3, !dbg !27
  %math.result249 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %750) #9, !dbg !27, !frisk.register_lifetimes !1
  %751 = fptrunc <4 x float> %math.result249 to <4 x half>, !dbg !27
  %math.packed250 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %751) #9, !dbg !26, !frisk.register_lifetimes !1
  %752 = fadd float %747, %749, !dbg !26
  %753 = extractelement <4 x float> %625, i64 0, !dbg !25
  %754 = call float @llvm.amdgcn.exp2.f32(float %753), !dbg !25
  %755 = insertelement <4 x float> poison, float %754, i64 0, !dbg !27
  %756 = fadd float %752, %754, !dbg !26
  %757 = extractelement <4 x float> %625, i64 1, !dbg !25
  %758 = call float @llvm.amdgcn.exp2.f32(float %757), !dbg !25
  %759 = insertelement <4 x float> %755, float %758, i64 1, !dbg !27
  %760 = fadd float %756, %758, !dbg !26
  %761 = extractelement <4 x float> %625, i64 2, !dbg !25
  %762 = call float @llvm.amdgcn.exp2.f32(float %761), !dbg !25
  %763 = insertelement <4 x float> %759, float %762, i64 2, !dbg !27
  %764 = fadd float %760, %762, !dbg !26
  %765 = extractelement <4 x float> %625, i64 3, !dbg !25
  %766 = call float @llvm.amdgcn.exp2.f32(float %765), !dbg !25
  %767 = insertelement <4 x float> %763, float %766, i64 3, !dbg !27
  %math.result251 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %767) #9, !dbg !27, !frisk.register_lifetimes !1
  %768 = fptrunc <4 x float> %math.result251 to <4 x half>, !dbg !27
  %math.packed252 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %768) #9, !dbg !26, !frisk.register_lifetimes !1
  %769 = fadd float %764, %766, !dbg !26
  %770 = extractelement <4 x float> %626, i64 0, !dbg !25
  %771 = call float @llvm.amdgcn.exp2.f32(float %770), !dbg !25
  %772 = insertelement <4 x float> poison, float %771, i64 0, !dbg !27
  %773 = fadd float %769, %771, !dbg !26
  %774 = extractelement <4 x float> %626, i64 1, !dbg !25
  %775 = call float @llvm.amdgcn.exp2.f32(float %774), !dbg !25
  %776 = insertelement <4 x float> %772, float %775, i64 1, !dbg !27
  %777 = fadd float %773, %775, !dbg !26
  %778 = extractelement <4 x float> %626, i64 2, !dbg !25
  %779 = call float @llvm.amdgcn.exp2.f32(float %778), !dbg !25
  %780 = insertelement <4 x float> %776, float %779, i64 2, !dbg !27
  %781 = fadd float %777, %779, !dbg !26
  %782 = extractelement <4 x float> %626, i64 3, !dbg !25
  %783 = call float @llvm.amdgcn.exp2.f32(float %782), !dbg !25
  %784 = insertelement <4 x float> %780, float %783, i64 3, !dbg !27
  %math.result253 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %784) #9, !dbg !27, !frisk.register_lifetimes !1
  %785 = fptrunc <4 x float> %math.result253 to <4 x half>, !dbg !27
  %math.packed254 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %785) #9, !dbg !26, !frisk.register_lifetimes !1
  %786 = fadd float %781, %783, !dbg !26
  %787 = extractelement <4 x float> %628, i64 0, !dbg !25
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !25
  %789 = insertelement <4 x float> poison, float %788, i64 0, !dbg !27
  %790 = extractelement <4 x float> %628, i64 1, !dbg !25
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !25
  %792 = insertelement <4 x float> %789, float %791, i64 1, !dbg !27
  %793 = fadd float %788, %791, !dbg !26
  %794 = extractelement <4 x float> %628, i64 2, !dbg !25
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !25
  %796 = insertelement <4 x float> %792, float %795, i64 2, !dbg !27
  %797 = fadd float %793, %795, !dbg !26
  %798 = extractelement <4 x float> %628, i64 3, !dbg !25
  %799 = call float @llvm.amdgcn.exp2.f32(float %798), !dbg !25
  %800 = insertelement <4 x float> %796, float %799, i64 3, !dbg !27
  %math.result255 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %800) #9, !dbg !27, !frisk.register_lifetimes !1
  %801 = fptrunc <4 x float> %math.result255 to <4 x half>, !dbg !27
  %math.packed256 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %801) #9, !dbg !26, !frisk.register_lifetimes !1
  %802 = fadd float %797, %799, !dbg !26
  %803 = extractelement <4 x float> %629, i64 0, !dbg !25
  %804 = call float @llvm.amdgcn.exp2.f32(float %803), !dbg !25
  %805 = insertelement <4 x float> poison, float %804, i64 0, !dbg !27
  %806 = fadd float %802, %804, !dbg !26
  %807 = extractelement <4 x float> %629, i64 1, !dbg !25
  %808 = call float @llvm.amdgcn.exp2.f32(float %807), !dbg !25
  %809 = insertelement <4 x float> %805, float %808, i64 1, !dbg !27
  %810 = fadd float %806, %808, !dbg !26
  %811 = extractelement <4 x float> %629, i64 2, !dbg !25
  %812 = call float @llvm.amdgcn.exp2.f32(float %811), !dbg !25
  %813 = insertelement <4 x float> %809, float %812, i64 2, !dbg !27
  %814 = fadd float %810, %812, !dbg !26
  %815 = extractelement <4 x float> %629, i64 3, !dbg !25
  %816 = call float @llvm.amdgcn.exp2.f32(float %815), !dbg !25
  %817 = insertelement <4 x float> %813, float %816, i64 3, !dbg !27
  %math.result257 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %817) #9, !dbg !27, !frisk.register_lifetimes !1
  %818 = fptrunc <4 x float> %math.result257 to <4 x half>, !dbg !27
  %math.packed258 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %818) #9, !dbg !26, !frisk.register_lifetimes !1
  %819 = fadd float %814, %816, !dbg !26
  %820 = extractelement <4 x float> %630, i64 0, !dbg !25
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !25
  %822 = insertelement <4 x float> poison, float %821, i64 0, !dbg !27
  %823 = fadd float %819, %821, !dbg !26
  %824 = extractelement <4 x float> %630, i64 1, !dbg !25
  %825 = call float @llvm.amdgcn.exp2.f32(float %824), !dbg !25
  %826 = insertelement <4 x float> %822, float %825, i64 1, !dbg !27
  %827 = fadd float %823, %825, !dbg !26
  %828 = extractelement <4 x float> %630, i64 2, !dbg !25
  %829 = call float @llvm.amdgcn.exp2.f32(float %828), !dbg !25
  %830 = insertelement <4 x float> %826, float %829, i64 2, !dbg !27
  %831 = fadd float %827, %829, !dbg !26
  %832 = extractelement <4 x float> %630, i64 3, !dbg !25
  %833 = call float @llvm.amdgcn.exp2.f32(float %832), !dbg !25
  %834 = insertelement <4 x float> %830, float %833, i64 3, !dbg !27
  %math.result259 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %834) #9, !dbg !27, !frisk.register_lifetimes !1
  %835 = fptrunc <4 x float> %math.result259 to <4 x half>, !dbg !27
  %math.packed260 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %835) #9, !dbg !26, !frisk.register_lifetimes !1
  %836 = fadd float %831, %833, !dbg !26
  %837 = extractelement <4 x float> %631, i64 0, !dbg !25
  %838 = call float @llvm.amdgcn.exp2.f32(float %837), !dbg !25
  %839 = insertelement <4 x float> poison, float %838, i64 0, !dbg !27
  %840 = fadd float %836, %838, !dbg !26
  %841 = extractelement <4 x float> %631, i64 1, !dbg !25
  %842 = call float @llvm.amdgcn.exp2.f32(float %841), !dbg !25
  %843 = insertelement <4 x float> %839, float %842, i64 1, !dbg !27
  %844 = fadd float %840, %842, !dbg !26
  %845 = extractelement <4 x float> %631, i64 2, !dbg !25
  %846 = call float @llvm.amdgcn.exp2.f32(float %845), !dbg !25
  %847 = insertelement <4 x float> %843, float %846, i64 2, !dbg !27
  %848 = fadd float %844, %846, !dbg !26
  %849 = extractelement <4 x float> %631, i64 3, !dbg !25
  %850 = call float @llvm.amdgcn.exp2.f32(float %849), !dbg !25
  %851 = insertelement <4 x float> %847, float %850, i64 3, !dbg !27
  %math.result261 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %851) #9, !dbg !27, !frisk.register_lifetimes !1
  %852 = fptrunc <4 x float> %math.result261 to <4 x half>, !dbg !27
  %math.packed262 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %852) #9, !dbg !26, !frisk.register_lifetimes !1
  %853 = fadd float %848, %850, !dbg !26
  %854 = extractelement <4 x float> %632, i64 0, !dbg !25
  %855 = call float @llvm.amdgcn.exp2.f32(float %854), !dbg !25
  %856 = insertelement <4 x float> poison, float %855, i64 0, !dbg !27
  %857 = fadd float %853, %855, !dbg !26
  %858 = extractelement <4 x float> %632, i64 1, !dbg !25
  %859 = call float @llvm.amdgcn.exp2.f32(float %858), !dbg !25
  %860 = insertelement <4 x float> %856, float %859, i64 1, !dbg !27
  %861 = fadd float %857, %859, !dbg !26
  %862 = extractelement <4 x float> %632, i64 2, !dbg !25
  %863 = call float @llvm.amdgcn.exp2.f32(float %862), !dbg !25
  %864 = insertelement <4 x float> %860, float %863, i64 2, !dbg !27
  %865 = fadd float %861, %863, !dbg !26
  %866 = extractelement <4 x float> %632, i64 3, !dbg !25
  %867 = call float @llvm.amdgcn.exp2.f32(float %866), !dbg !25
  %868 = insertelement <4 x float> %864, float %867, i64 3, !dbg !27
  %math.result263 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %868) #9, !dbg !27, !frisk.register_lifetimes !1
  %869 = fptrunc <4 x float> %math.result263 to <4 x half>, !dbg !27
  %math.packed264 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %869) #9, !dbg !26, !frisk.register_lifetimes !1
  %870 = fadd float %865, %867, !dbg !26
  %871 = extractelement <4 x float> %633, i64 0, !dbg !25
  %872 = call float @llvm.amdgcn.exp2.f32(float %871), !dbg !25
  %873 = insertelement <4 x float> poison, float %872, i64 0, !dbg !27
  %874 = fadd float %870, %872, !dbg !26
  %875 = extractelement <4 x float> %633, i64 1, !dbg !25
  %876 = call float @llvm.amdgcn.exp2.f32(float %875), !dbg !25
  %877 = insertelement <4 x float> %873, float %876, i64 1, !dbg !27
  %878 = fadd float %874, %876, !dbg !26
  %879 = extractelement <4 x float> %633, i64 2, !dbg !25
  %880 = call float @llvm.amdgcn.exp2.f32(float %879), !dbg !25
  %881 = insertelement <4 x float> %877, float %880, i64 2, !dbg !27
  %882 = fadd float %878, %880, !dbg !26
  %883 = extractelement <4 x float> %633, i64 3, !dbg !25
  %884 = call float @llvm.amdgcn.exp2.f32(float %883), !dbg !25
  %885 = insertelement <4 x float> %881, float %884, i64 3, !dbg !27
  %math.result265 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %885) #9, !dbg !27, !frisk.register_lifetimes !1
  %886 = fptrunc <4 x float> %math.result265 to <4 x half>, !dbg !27
  %math.packed266 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %886) #9, !dbg !26, !frisk.register_lifetimes !1
  %887 = fadd float %882, %884, !dbg !26
  %888 = extractelement <4 x float> %634, i64 0, !dbg !25
  %889 = call float @llvm.amdgcn.exp2.f32(float %888), !dbg !25
  %890 = insertelement <4 x float> poison, float %889, i64 0, !dbg !27
  %891 = fadd float %887, %889, !dbg !26
  %892 = extractelement <4 x float> %634, i64 1, !dbg !25
  %893 = call float @llvm.amdgcn.exp2.f32(float %892), !dbg !25
  %894 = insertelement <4 x float> %890, float %893, i64 1, !dbg !27
  %895 = fadd float %891, %893, !dbg !26
  %896 = extractelement <4 x float> %634, i64 2, !dbg !25
  %897 = call float @llvm.amdgcn.exp2.f32(float %896), !dbg !25
  %898 = insertelement <4 x float> %894, float %897, i64 2, !dbg !27
  %899 = fadd float %895, %897, !dbg !26
  %900 = extractelement <4 x float> %634, i64 3, !dbg !25
  %901 = call float @llvm.amdgcn.exp2.f32(float %900), !dbg !25
  %902 = insertelement <4 x float> %898, float %901, i64 3, !dbg !27
  %math.result267 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %902) #9, !dbg !27, !frisk.register_lifetimes !1
  %903 = fptrunc <4 x float> %math.result267 to <4 x half>, !dbg !27
  %math.packed268 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %903) #9, !dbg !26, !frisk.register_lifetimes !1
  %904 = fadd float %899, %901, !dbg !26
  %905 = extractelement <4 x float> %635, i64 0, !dbg !25
  %906 = call float @llvm.amdgcn.exp2.f32(float %905), !dbg !25
  %907 = insertelement <4 x float> poison, float %906, i64 0, !dbg !27
  %908 = fadd float %904, %906, !dbg !26
  %909 = extractelement <4 x float> %635, i64 1, !dbg !25
  %910 = call float @llvm.amdgcn.exp2.f32(float %909), !dbg !25
  %911 = insertelement <4 x float> %907, float %910, i64 1, !dbg !27
  %912 = fadd float %908, %910, !dbg !26
  %913 = extractelement <4 x float> %635, i64 2, !dbg !25
  %914 = call float @llvm.amdgcn.exp2.f32(float %913), !dbg !25
  %915 = insertelement <4 x float> %911, float %914, i64 2, !dbg !27
  %916 = fadd float %912, %914, !dbg !26
  %917 = extractelement <4 x float> %635, i64 3, !dbg !25
  %918 = call float @llvm.amdgcn.exp2.f32(float %917), !dbg !25
  %919 = insertelement <4 x float> %915, float %918, i64 3, !dbg !27
  %math.result269 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %919) #9, !dbg !27, !frisk.register_lifetimes !1
  %920 = fptrunc <4 x float> %math.result269 to <4 x half>, !dbg !27
  %math.packed270 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %920) #9, !dbg !26, !frisk.register_lifetimes !1
  %921 = fadd float %916, %918, !dbg !26
  %922 = bitcast float %786 to i32, !dbg !26
  %923 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %922), !dbg !26
  %924 = bitcast i32 %923 to float, !dbg !26
  %925 = fadd float %786, %924, !dbg !26
  %926 = bitcast float %925 to i32, !dbg !26
  %927 = call i32 @llvm.amdgcn.ds.bpermute(i32 %567, i32 %926), !dbg !26
  %928 = bitcast i32 %927 to float, !dbg !26
  %929 = fadd float %925, %928, !dbg !26
  %930 = fadd float %929, %643, !dbg !26
  %931 = insertelement <1 x float> poison, float %930, i64 0, !dbg !7
  %932 = insertvalue [2 x <1 x float>] poison, <1 x float> %931, 0, !dbg !7
  %933 = bitcast float %921 to i32, !dbg !26
  %934 = call i32 @llvm.amdgcn.ds.bpermute(i32 %559, i32 %933), !dbg !26
  %935 = bitcast i32 %934 to float, !dbg !26
  %936 = fadd float %921, %935, !dbg !26
  %937 = bitcast float %936 to i32, !dbg !26
  %938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %567, i32 %937), !dbg !26
  %939 = bitcast i32 %938 to float, !dbg !26
  %940 = fadd float %936, %939, !dbg !26
  %941 = fadd float %940, %651, !dbg !26
  %942 = insertelement <1 x float> poison, float %941, i64 0, !dbg !7
  %943 = insertvalue [2 x <1 x float>] %932, <1 x float> %942, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !15
  %vdb.address1302 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address), !dbg !15
  %vdb.address1303 = or disjoint i64 %149, %address.local788, !dbg !15
  %vdb.address1304 = trunc nuw i64 %vdb.address1303 to i32, !dbg !15
  %vdb.address1305 = shl nuw nsw i32 %vdb.address1304, 1, !dbg !15
  %vdb.address1306 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1307 = insertelement <2 x i64> poison, i64 %vdb.address1306, i64 0, !dbg !15
  %vdb.address1308 = bitcast <2 x i64> %vdb.address1307 to <4 x i32>, !dbg !15
  %vdb.address1309 = shufflevector <4 x i32> %vdb.address1308, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1310 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !15
  %vdb.address1311 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1310), !dbg !15
  %vdb.address1312 = or disjoint i64 %149, %address.local794, !dbg !15
  %vdb.address1313 = trunc nuw i64 %vdb.address1312 to i32, !dbg !15
  %vdb.address1314 = shl nuw nsw i32 %vdb.address1313, 1, !dbg !15
  %vdb.address1315 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1316 = insertelement <2 x i64> poison, i64 %vdb.address1315, i64 0, !dbg !15
  %vdb.address1317 = bitcast <2 x i64> %vdb.address1316 to <4 x i32>, !dbg !15
  %vdb.address1318 = shufflevector <4 x i32> %vdb.address1317, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.dma.base = add i32 %vdb.address1302, 8192, !dbg !13
  %944 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1305, <4 x i32> %vdb.address1309) #10, !dbg !15
  %vdb.dma.base1319 = add i32 %vdb.address1311, 8192, !dbg !13
  %945 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1319, i32 %vdb.address1314, <4 x i32> %vdb.address1318) #10, !dbg !15
  %address.local796 = shl nuw nsw i32 %address.thread365, 3, !dbg !13
  %address.local797 = and i32 %address.local796, 120, !dbg !13
  %address.local798 = shl nuw nsw i32 %address.thread365, 5, !dbg !13
  %address.local799 = and i32 %address.local798, 1536, !dbg !13
  %address.local800 = or disjoint i32 %address.local797, %address.local799, !dbg !13
  %address.local801 = zext nneg i32 %address.local800 to i64, !dbg !13
  %address.local802 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local801, !dbg !13
  %946 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local802, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %947 = bitcast <4 x i32> %946 to <8 x half>, !dbg !13
  %address.local803 = and i64 %address.local366, 60, !dbg !13
  %address.local804 = add nuw nsw i64 %address.local803, %address.local366, !dbg !13
  %address.local805 = shl nuw nsw i64 %address.local804, 3, !dbg !13
  %address.local806 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local807 = lshr i64 %address.local422, 4, !dbg !13
  %address.local808 = mul nuw nsw i64 %address.local807, 384, !dbg !13
  %address.local809 = getelementptr [2 x i8], ptr addrspace(3) %address.local806, i64 %address.local808, !dbg !13
  %address.local810 = mul nsw i64 %address.local409, -32, !dbg !13
  %address.local811 = getelementptr [2 x i8], ptr addrspace(3) %address.local809, i64 %address.local810, !dbg !13
  %address.local812 = getelementptr i8, ptr addrspace(3) %address.local811, i64 256, !dbg !13
  %948 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local812, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %949 = bitcast <4 x i32> %948 to <8 x half>, !dbg !13
  %address.local813 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local814 = getelementptr [2 x i8], ptr addrspace(3) %address.local813, i64 %address.local808, !dbg !13
  %address.local815 = getelementptr [2 x i8], ptr addrspace(3) %address.local814, i64 %address.local810, !dbg !13
  %address.local816 = getelementptr i8, ptr addrspace(3) %address.local815, i64 512, !dbg !13
  %950 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %951 = bitcast <4 x i32> %950 to <8 x half>, !dbg !13
  %address.local817 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local818 = getelementptr [2 x i8], ptr addrspace(3) %address.local817, i64 %address.local808, !dbg !13
  %address.local819 = getelementptr [2 x i8], ptr addrspace(3) %address.local818, i64 %address.local810, !dbg !13
  %address.local820 = getelementptr i8, ptr addrspace(3) %address.local819, i64 768, !dbg !13
  %952 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %953 = bitcast <4 x i32> %952 to <8 x half>, !dbg !13
  %address.local821 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local801, !dbg !13
  %address.local822 = getelementptr i8, ptr addrspace(3) %address.local821, i64 4096, !dbg !13
  %954 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %955 = bitcast <4 x i32> %954 to <8 x half>, !dbg !13
  %address.local823 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local824 = getelementptr [2 x i8], ptr addrspace(3) %address.local823, i64 %address.local808, !dbg !13
  %address.local825 = getelementptr [2 x i8], ptr addrspace(3) %address.local824, i64 %address.local810, !dbg !13
  %address.local826 = getelementptr i8, ptr addrspace(3) %address.local825, i64 4352, !dbg !13
  %956 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %957 = bitcast <4 x i32> %956 to <8 x half>, !dbg !13
  %address.local827 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local828 = getelementptr [2 x i8], ptr addrspace(3) %address.local827, i64 %address.local808, !dbg !13
  %address.local829 = getelementptr [2 x i8], ptr addrspace(3) %address.local828, i64 %address.local810, !dbg !13
  %address.local830 = getelementptr i8, ptr addrspace(3) %address.local829, i64 4608, !dbg !13
  %958 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %959 = bitcast <4 x i32> %958 to <8 x half>, !dbg !13
  %address.local831 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local832 = getelementptr [2 x i8], ptr addrspace(3) %address.local831, i64 %address.local808, !dbg !13
  %address.local833 = getelementptr [2 x i8], ptr addrspace(3) %address.local832, i64 %address.local810, !dbg !13
  %address.local834 = getelementptr i8, ptr addrspace(3) %address.local833, i64 4864, !dbg !13
  %960 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local834, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %961 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !15
  %dma.buffer.lds1211 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %961), !dbg !15
  %.idx12901291 = or disjoint i64 %149, %address.local788, !dbg !15
  %.idx12901291.tr = trunc nuw i64 %.idx12901291 to i32, !dbg !15
  %dma.buffer.offset1209 = shl nuw nsw i32 %.idx12901291.tr, 1, !dbg !15
  %962 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %963 = insertelement <2 x i64> poison, i64 %962, i64 0, !dbg !15
  %964 = bitcast <2 x i64> %963 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1210 = shufflevector <4 x i32> %964, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %965 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !15
  %dma.buffer.lds1208 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %965), !dbg !15
  %.idx12921293 = or disjoint i64 %149, %address.local794, !dbg !15
  %.idx12921293.tr = trunc nuw i64 %.idx12921293 to i32, !dbg !15
  %dma.buffer.offset1206 = shl nuw nsw i32 %.idx12921293.tr, 1, !dbg !15
  %966 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %967 = insertelement <2 x i64> poison, i64 %966, i64 0, !dbg !15
  %968 = bitcast <2 x i64> %967 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1207 = shufflevector <4 x i32> %968, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %969 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %970 = extractvalue [2 x <8 x float>] %127, 0, !dbg !28
  %971 = shufflevector <8 x float> %970, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %972 = fmul <4 x float> %971, %640, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %969, <4 x float> %972) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %974 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %975 = shufflevector <8 x float> %970, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %976 = fmul <4 x float> %975, %640, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %974, <4 x float> %976) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %978 = extractvalue [2 x <8 x float>] %127, 1, !dbg !28
  %979 = shufflevector <8 x float> %978, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %980 = fmul <4 x float> %979, %648, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %969, <4 x float> %980) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %982 = shufflevector <8 x float> %978, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %983 = fmul <4 x float> %982, %648, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %974, <4 x float> %983) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %985 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %985, <4 x float> %973) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %987 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %987, <4 x float> %977) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %985, <4 x float> %981) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %987, <4 x float> %984) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %991 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %991, <4 x float> %986) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %993 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %993, <4 x float> %988) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %991, <4 x float> %989) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %993, <4 x float> %990) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %997 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %997, <4 x float> %992) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %999 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1000 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %999, <4 x float> %994) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %997, <4 x float> %995) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %999, <4 x float> %996) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1003 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1003, <4 x float> %998) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1005 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1006 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1005, <4 x float> %1000) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1007 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1003, <4 x float> %1001) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1005, <4 x float> %1002) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1009 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1009, <4 x float> %1004) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1011 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1011, <4 x float> %1006) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1013 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1009, <4 x float> %1007) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1011, <4 x float> %1008) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1015 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1015, <4 x float> %1010) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1017 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1017, <4 x float> %1012) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1019 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1015, <4 x float> %1013) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1017, <4 x float> %1014) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1021 = bitcast <4 x i32> %960 to <8 x half>, !dbg !13
  %1022 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1022, <4 x float> %1016) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1024 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1024, <4 x float> %1018) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1022, <4 x float> %1019) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1024, <4 x float> %1020) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1327 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !15
  %vdb.address1328 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1327), !dbg !15
  %vdb.address1329 = or disjoint i64 %232, %address.local788, !dbg !15
  %vdb.address1330 = trunc nuw i64 %vdb.address1329 to i32, !dbg !15
  %vdb.address1331 = shl nuw nsw i32 %vdb.address1330, 1, !dbg !15
  %vdb.address1332 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1333 = insertelement <2 x i64> poison, i64 %vdb.address1332, i64 0, !dbg !15
  %vdb.address1334 = bitcast <2 x i64> %vdb.address1333 to <4 x i32>, !dbg !15
  %vdb.address1335 = shufflevector <4 x i32> %vdb.address1334, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1336 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !15
  %vdb.address1337 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1336), !dbg !15
  %vdb.address1338 = or disjoint i64 %232, %address.local794, !dbg !15
  %vdb.address1339 = trunc nuw i64 %vdb.address1338 to i32, !dbg !15
  %vdb.address1340 = shl nuw nsw i32 %vdb.address1339, 1, !dbg !15
  %vdb.address1341 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1342 = insertelement <2 x i64> poison, i64 %vdb.address1341, i64 0, !dbg !15
  %vdb.address1343 = bitcast <2 x i64> %vdb.address1342 to <4 x i32>, !dbg !15
  %vdb.address1344 = shufflevector <4 x i32> %vdb.address1343, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1028 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.address1328, i32 %vdb.address1331, <4 x i32> %vdb.address1335) #10, !dbg !15
  %1029 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.address1337, i32 %vdb.address1340, <4 x i32> %vdb.address1344) #10, !dbg !15
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local802, i64 8192, !dbg !13
  %1030 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1031 = bitcast <4 x i32> %1030 to <8 x half>, !dbg !13
  %vdb.read1320 = getelementptr i8, ptr addrspace(3) %address.local812, i64 8192, !dbg !13
  %1032 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1320, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1033 = bitcast <4 x i32> %1032 to <8 x half>, !dbg !13
  %vdb.read1321 = getelementptr i8, ptr addrspace(3) %address.local816, i64 8192, !dbg !13
  %1034 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1321, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1035 = bitcast <4 x i32> %1034 to <8 x half>, !dbg !13
  %vdb.read1322 = getelementptr i8, ptr addrspace(3) %address.local820, i64 8192, !dbg !13
  %1036 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1322, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1037 = bitcast <4 x i32> %1036 to <8 x half>, !dbg !13
  %vdb.read1323 = getelementptr i8, ptr addrspace(3) %address.local822, i64 8192, !dbg !13
  %1038 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1323, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1039 = bitcast <4 x i32> %1038 to <8 x half>, !dbg !13
  %vdb.read1324 = getelementptr i8, ptr addrspace(3) %address.local826, i64 8192, !dbg !13
  %1040 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1324, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1041 = bitcast <4 x i32> %1040 to <8 x half>, !dbg !13
  %vdb.read1325 = getelementptr i8, ptr addrspace(3) %address.local830, i64 8192, !dbg !13
  %1042 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1325, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1043 = bitcast <4 x i32> %1042 to <8 x half>, !dbg !13
  %vdb.read1326 = getelementptr i8, ptr addrspace(3) %address.local834, i64 8192, !dbg !13
  %1044 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1326, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1045 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !15
  %dma.buffer.lds1205 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1045), !dbg !15
  %.idx12941295 = or disjoint i64 %232, %address.local788, !dbg !15
  %.idx12941295.tr = trunc nuw i64 %.idx12941295 to i32, !dbg !15
  %dma.buffer.offset1203 = shl nuw nsw i32 %.idx12941295.tr, 1, !dbg !15
  %1046 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1047 = insertelement <2 x i64> poison, i64 %1046, i64 0, !dbg !15
  %1048 = bitcast <2 x i64> %1047 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1204 = shufflevector <4 x i32> %1048, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1049 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !15
  %dma.buffer.lds1202 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1049), !dbg !15
  %.idx12961297 = or disjoint i64 %232, %address.local794, !dbg !15
  %.idx12961297.tr = trunc nuw i64 %.idx12961297 to i32, !dbg !15
  %dma.buffer.offset1200 = shl nuw nsw i32 %.idx12961297.tr, 1, !dbg !15
  %1050 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1051 = insertelement <2 x i64> poison, i64 %1050, i64 0, !dbg !15
  %1052 = bitcast <2 x i64> %1051 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1201 = shufflevector <4 x i32> %1052, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1053 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1054 = extractvalue [2 x <8 x float>] %130, 0, !dbg !28
  %1055 = shufflevector <8 x float> %1054, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1056 = fmul <4 x float> %1055, %640, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1053, <4 x float> %1056) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1058 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1059 = shufflevector <8 x float> %1054, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1060 = fmul <4 x float> %1059, %640, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1058, <4 x float> %1060) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1062 = extractvalue [2 x <8 x float>] %130, 1, !dbg !28
  %1063 = shufflevector <8 x float> %1062, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1064 = fmul <4 x float> %1063, %648, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1053, <4 x float> %1064) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1066 = shufflevector <8 x float> %1062, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1067 = fmul <4 x float> %1066, %648, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1058, <4 x float> %1067) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1069 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1069, <4 x float> %1057) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1071 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1071, <4 x float> %1061) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1069, <4 x float> %1065) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1071, <4 x float> %1068) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1075 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1075, <4 x float> %1070) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1077 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1077, <4 x float> %1072) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1075, <4 x float> %1073) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1077, <4 x float> %1074) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1081 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1081, <4 x float> %1076) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1083 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1083, <4 x float> %1078) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1085 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1081, <4 x float> %1079) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1083, <4 x float> %1080) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1087 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1087, <4 x float> %1082) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1089 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1089, <4 x float> %1084) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1091 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1087, <4 x float> %1085) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1089, <4 x float> %1086) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1093 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1093, <4 x float> %1088) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1095 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1095, <4 x float> %1090) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1097 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1093, <4 x float> %1091) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1095, <4 x float> %1092) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1099 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1099, <4 x float> %1094) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1101 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1101, <4 x float> %1096) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1099, <4 x float> %1097) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1101, <4 x float> %1098) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1105 = bitcast <4 x i32> %1044 to <8 x half>, !dbg !13
  %1106 = shufflevector <8 x half> %1105, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1107 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1106, <4 x float> %1100) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1108 = shufflevector <8 x half> %1105, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1109 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1108, <4 x float> %1102) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1106, <4 x float> %1103) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1111 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1108, <4 x float> %1104) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1345 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !15
  %vdb.address1346 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1345), !dbg !15
  %vdb.address1347 = or disjoint i64 %307, %address.local788, !dbg !15
  %vdb.address1348 = trunc nuw i64 %vdb.address1347 to i32, !dbg !15
  %vdb.address1349 = shl nuw nsw i32 %vdb.address1348, 1, !dbg !15
  %vdb.address1350 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1351 = insertelement <2 x i64> poison, i64 %vdb.address1350, i64 0, !dbg !15
  %vdb.address1352 = bitcast <2 x i64> %vdb.address1351 to <4 x i32>, !dbg !15
  %vdb.address1353 = shufflevector <4 x i32> %vdb.address1352, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1354 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !15
  %vdb.address1355 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1354), !dbg !15
  %vdb.address1356 = or disjoint i64 %307, %address.local794, !dbg !15
  %vdb.address1357 = trunc nuw i64 %vdb.address1356 to i32, !dbg !15
  %vdb.address1358 = shl nuw nsw i32 %vdb.address1357, 1, !dbg !15
  %vdb.address1359 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1360 = insertelement <2 x i64> poison, i64 %vdb.address1359, i64 0, !dbg !15
  %vdb.address1361 = bitcast <2 x i64> %vdb.address1360 to <4 x i32>, !dbg !15
  %vdb.address1362 = shufflevector <4 x i32> %vdb.address1361, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.dma.base1363 = add i32 %vdb.address1346, 8192, !dbg !13
  %1112 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1363, i32 %vdb.address1349, <4 x i32> %vdb.address1353) #10, !dbg !15
  %vdb.dma.base1364 = add i32 %vdb.address1355, 8192, !dbg !13
  %1113 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1364, i32 %vdb.address1358, <4 x i32> %vdb.address1362) #10, !dbg !15
  %1114 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local802, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1115 = bitcast <4 x i32> %1114 to <8 x half>, !dbg !13
  %1116 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local812, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1117 = bitcast <4 x i32> %1116 to <8 x half>, !dbg !13
  %1118 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1119 = bitcast <4 x i32> %1118 to <8 x half>, !dbg !13
  %1120 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1121 = bitcast <4 x i32> %1120 to <8 x half>, !dbg !13
  %1122 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1123 = bitcast <4 x i32> %1122 to <8 x half>, !dbg !13
  %1124 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1125 = bitcast <4 x i32> %1124 to <8 x half>, !dbg !13
  %1126 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1127 = bitcast <4 x i32> %1126 to <8 x half>, !dbg !13
  %1128 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local834, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1129 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !15
  %dma.buffer.lds1199 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1129), !dbg !15
  %.idx12981299 = or disjoint i64 %307, %address.local788, !dbg !15
  %.idx12981299.tr = trunc nuw i64 %.idx12981299 to i32, !dbg !15
  %dma.buffer.offset1197 = shl nuw nsw i32 %.idx12981299.tr, 1, !dbg !15
  %1130 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1131 = insertelement <2 x i64> poison, i64 %1130, i64 0, !dbg !15
  %1132 = bitcast <2 x i64> %1131 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1198 = shufflevector <4 x i32> %1132, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1133 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !15
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1133), !dbg !15
  %.idx13001301 = or disjoint i64 %307, %address.local794, !dbg !15
  %.idx13001301.tr = trunc nuw i64 %.idx13001301 to i32, !dbg !15
  %dma.buffer.offset = shl nuw nsw i32 %.idx13001301.tr, 1, !dbg !15
  %1134 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1135 = insertelement <2 x i64> poison, i64 %1134, i64 0, !dbg !15
  %1136 = bitcast <2 x i64> %1135 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc = shufflevector <4 x i32> %1136, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1137 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1138 = extractvalue [2 x <8 x float>] %131, 0, !dbg !28
  %1139 = shufflevector <8 x float> %1138, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1140 = fmul <4 x float> %1139, %640, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1137, <4 x float> %1140) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1142 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1143 = shufflevector <8 x float> %1138, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1144 = fmul <4 x float> %1143, %640, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1142, <4 x float> %1144) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1146 = extractvalue [2 x <8 x float>] %131, 1, !dbg !28
  %1147 = shufflevector <8 x float> %1146, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1148 = fmul <4 x float> %1147, %648, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1137, <4 x float> %1148) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1150 = shufflevector <8 x float> %1146, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1151 = fmul <4 x float> %1150, %648, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1142, <4 x float> %1151) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1153 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1153, <4 x float> %1141) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1155 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1155, <4 x float> %1145) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1153, <4 x float> %1149) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1155, <4 x float> %1152) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1159 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1159, <4 x float> %1154) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1161 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1161, <4 x float> %1156) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1159, <4 x float> %1157) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1161, <4 x float> %1158) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1165 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1165, <4 x float> %1160) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1167 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1167, <4 x float> %1162) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1165, <4 x float> %1163) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1167, <4 x float> %1164) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1171 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1171, <4 x float> %1166) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1173 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1173, <4 x float> %1168) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1171, <4 x float> %1169) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1173, <4 x float> %1170) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1177 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1177, <4 x float> %1172) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1179 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1179, <4 x float> %1174) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1177, <4 x float> %1175) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1179, <4 x float> %1176) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1183 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1183, <4 x float> %1178) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1185 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1185, <4 x float> %1180) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1183, <4 x float> %1181) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1185, <4 x float> %1182) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1189 = bitcast <4 x i32> %1128 to <8 x half>, !dbg !13
  %1190 = shufflevector <8 x half> %1189, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1190, <4 x float> %1184) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1192 = shufflevector <8 x half> %1189, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1192, <4 x float> %1186) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1190, <4 x float> %1187) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1192, <4 x float> %1188) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.read1365 = getelementptr i8, ptr addrspace(3) %address.local802, i64 8192, !dbg !13
  %1196 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1365, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1197 = bitcast <4 x i32> %1196 to <8 x half>, !dbg !13
  %vdb.read1366 = getelementptr i8, ptr addrspace(3) %address.local812, i64 8192, !dbg !13
  %1198 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1366, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1199 = bitcast <4 x i32> %1198 to <8 x half>, !dbg !13
  %vdb.read1367 = getelementptr i8, ptr addrspace(3) %address.local816, i64 8192, !dbg !13
  %1200 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1367, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1201 = bitcast <4 x i32> %1200 to <8 x half>, !dbg !13
  %vdb.read1368 = getelementptr i8, ptr addrspace(3) %address.local820, i64 8192, !dbg !13
  %1202 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1368, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1203 = bitcast <4 x i32> %1202 to <8 x half>, !dbg !13
  %vdb.read1369 = getelementptr i8, ptr addrspace(3) %address.local822, i64 8192, !dbg !13
  %1204 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1369, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1205 = bitcast <4 x i32> %1204 to <8 x half>, !dbg !13
  %vdb.read1370 = getelementptr i8, ptr addrspace(3) %address.local826, i64 8192, !dbg !13
  %1206 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1370, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1207 = bitcast <4 x i32> %1206 to <8 x half>, !dbg !13
  %vdb.read1371 = getelementptr i8, ptr addrspace(3) %address.local830, i64 8192, !dbg !13
  %1208 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1371, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1209 = bitcast <4 x i32> %1208 to <8 x half>, !dbg !13
  %vdb.read1372 = getelementptr i8, ptr addrspace(3) %address.local834, i64 8192, !dbg !13
  %1210 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1372, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1211 = shufflevector <8 x half> %1197, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1212 = extractvalue [2 x <8 x float>] %132, 0, !dbg !28
  %1213 = shufflevector <8 x float> %1212, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1214 = fmul <4 x float> %1213, %640, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1211, <4 x float> %1214) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1216 = shufflevector <8 x half> %1197, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1217 = shufflevector <8 x float> %1212, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1218 = fmul <4 x float> %1217, %640, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1216, <4 x float> %1218) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1220 = extractvalue [2 x <8 x float>] %132, 1, !dbg !28
  %1221 = shufflevector <8 x float> %1220, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1222 = fmul <4 x float> %1221, %648, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1211, <4 x float> %1222) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1224 = shufflevector <8 x float> %1220, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1225 = fmul <4 x float> %1224, %648, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1216, <4 x float> %1225) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1227 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1227, <4 x float> %1215) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1229 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1229, <4 x float> %1219) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1227, <4 x float> %1223) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1229, <4 x float> %1226) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1233 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1233, <4 x float> %1228) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1235 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1235, <4 x float> %1230) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1233, <4 x float> %1231) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1235, <4 x float> %1232) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1239 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1239, <4 x float> %1234) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1241 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1241, <4 x float> %1236) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1239, <4 x float> %1237) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1241, <4 x float> %1238) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1245 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1245, <4 x float> %1240) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1247 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1247, <4 x float> %1242) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1245, <4 x float> %1243) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1247, <4 x float> %1244) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1251 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1251, <4 x float> %1246) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1253 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1253, <4 x float> %1248) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1251, <4 x float> %1249) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1253, <4 x float> %1250) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1257 = shufflevector <8 x half> %1209, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1257, <4 x float> %1252) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1259 = shufflevector <8 x half> %1209, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1259, <4 x float> %1254) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1257, <4 x float> %1255) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1259, <4 x float> %1256) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1263 = bitcast <4 x i32> %1210 to <8 x half>, !dbg !13
  %1264 = shufflevector <8 x half> %1263, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1264, <4 x float> %1258) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1266 = shufflevector <8 x half> %1263, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1266, <4 x float> %1260) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1264, <4 x float> %1261) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1266, <4 x float> %1262) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %1270 = shufflevector <4 x float> %1023, <4 x float> %1025, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1271 = insertvalue [2 x <8 x float>] poison, <8 x float> %1270, 0, !dbg !7
  %1272 = shufflevector <4 x float> %1026, <4 x float> %1027, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1273 = insertvalue [2 x <8 x float>] %1271, <8 x float> %1272, 1, !dbg !7
  %1274 = insertvalue [2 x <1 x float>] poison, <1 x float> %574, 0, !dbg !7
  %1275 = insertvalue [2 x <1 x float>] %1274, <1 x float> %617, 1, !dbg !7
  %1276 = shufflevector <4 x float> %1107, <4 x float> %1109, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1277 = insertvalue [2 x <8 x float>] poison, <8 x float> %1276, 0, !dbg !7
  %1278 = shufflevector <4 x float> %1110, <4 x float> %1111, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1279 = insertvalue [2 x <8 x float>] %1277, <8 x float> %1278, 1, !dbg !7
  %1280 = shufflevector <4 x float> %1191, <4 x float> %1193, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1281 = insertvalue [2 x <8 x float>] poison, <8 x float> %1280, 0, !dbg !7
  %1282 = shufflevector <4 x float> %1194, <4 x float> %1195, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1283 = insertvalue [2 x <8 x float>] %1281, <8 x float> %1282, 1, !dbg !7
  %1284 = shufflevector <4 x float> %1265, <4 x float> %1267, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1285 = insertvalue [2 x <8 x float>] poison, <8 x float> %1284, 0, !dbg !7
  %1286 = shufflevector <4 x float> %1268, <4 x float> %1269, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1287 = insertvalue [2 x <8 x float>] %1285, <8 x float> %1286, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1288 = add nuw nsw i64 %126, 1, !dbg !7
  br label %125, !dbg !7

1289:                                             ; preds = %125
  %1290 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %1290, 0, !dbg !7
  %1291 = and i32 %31, 32, !dbg !7
  %.not36 = icmp eq i32 %1291, 0, !dbg !7
  %1292 = extractvalue [2 x <8 x float>] %132, 0, !dbg !7
  %1293 = extractelement <8 x float> %1292, i64 0, !dbg !7
  %1294 = extractelement <8 x float> %1292, i64 1, !dbg !7
  %1295 = extractelement <8 x float> %1292, i64 2, !dbg !7
  %1296 = extractelement <8 x float> %1292, i64 3, !dbg !7
  %1297 = select i1 %.not, float %1294, float %1293, !dbg !7
  %1298 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1299 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1298), !dbg !7
  %1300 = add nuw nsw i32 %1299, 64, !dbg !7
  %1301 = xor i32 %1299, 16, !dbg !7
  %1302 = and i32 %1300, 192, !dbg !7
  %1303 = icmp samesign ult i32 %1301, %1302, !dbg !7
  %1304 = select i1 %1303, i32 %1301, i32 %1299, !dbg !7
  %1305 = shl nuw nsw i32 %1304, 2, !dbg !7
  %1306 = bitcast float %1297 to i32, !dbg !7
  %1307 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1306), !dbg !7
  %1308 = bitcast i32 %1307 to float, !dbg !7
  %1309 = select i1 %.not, float %1293, float %1308, !dbg !7
  %1310 = select i1 %.not, float %1308, float %1294, !dbg !7
  %1311 = select i1 %.not, float %1296, float %1295, !dbg !7
  %1312 = bitcast float %1311 to i32, !dbg !7
  %1313 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1312), !dbg !7
  %1314 = bitcast i32 %1313 to float, !dbg !7
  %1315 = select i1 %.not, float %1295, float %1314, !dbg !7
  %1316 = select i1 %.not, float %1314, float %1296, !dbg !7
  %1317 = select i1 %.not36, float %1315, float %1309, !dbg !7
  %1318 = xor i32 %1299, 32, !dbg !7
  %1319 = icmp samesign ult i32 %1318, %1302, !dbg !7
  %1320 = select i1 %1319, i32 %1318, i32 %1299, !dbg !7
  %1321 = shl nuw nsw i32 %1320, 2, !dbg !7
  %1322 = bitcast float %1317 to i32, !dbg !7
  %1323 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1322), !dbg !7
  %1324 = bitcast i32 %1323 to float, !dbg !7
  %1325 = select i1 %.not36, float %1309, float %1324, !dbg !7
  %1326 = select i1 %.not36, float %1324, float %1315, !dbg !7
  %1327 = select i1 %.not36, float %1316, float %1310, !dbg !7
  %1328 = bitcast float %1327 to i32, !dbg !7
  %1329 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1328), !dbg !7
  %1330 = bitcast i32 %1329 to float, !dbg !7
  %1331 = select i1 %.not36, float %1310, float %1330, !dbg !7
  %1332 = select i1 %.not36, float %1330, float %1316, !dbg !7
  %1333 = extractvalue [2 x <8 x float>] %131, 0, !dbg !7
  %1334 = extractelement <8 x float> %1333, i64 0, !dbg !7
  %1335 = extractelement <8 x float> %1333, i64 1, !dbg !7
  %1336 = extractelement <8 x float> %1333, i64 2, !dbg !7
  %1337 = extractelement <8 x float> %1333, i64 3, !dbg !7
  %1338 = select i1 %.not, float %1335, float %1334, !dbg !7
  %1339 = bitcast float %1338 to i32, !dbg !7
  %1340 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1339), !dbg !7
  %1341 = bitcast i32 %1340 to float, !dbg !7
  %1342 = select i1 %.not, float %1334, float %1341, !dbg !7
  %1343 = select i1 %.not, float %1341, float %1335, !dbg !7
  %1344 = select i1 %.not, float %1337, float %1336, !dbg !7
  %1345 = bitcast float %1344 to i32, !dbg !7
  %1346 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1345), !dbg !7
  %1347 = bitcast i32 %1346 to float, !dbg !7
  %1348 = select i1 %.not, float %1336, float %1347, !dbg !7
  %1349 = select i1 %.not, float %1347, float %1337, !dbg !7
  %1350 = select i1 %.not36, float %1348, float %1342, !dbg !7
  %1351 = bitcast float %1350 to i32, !dbg !7
  %1352 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1351), !dbg !7
  %1353 = bitcast i32 %1352 to float, !dbg !7
  %1354 = select i1 %.not36, float %1342, float %1353, !dbg !7
  %1355 = select i1 %.not36, float %1353, float %1348, !dbg !7
  %1356 = select i1 %.not36, float %1349, float %1343, !dbg !7
  %1357 = bitcast float %1356 to i32, !dbg !7
  %1358 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1357), !dbg !7
  %1359 = bitcast i32 %1358 to float, !dbg !7
  %1360 = select i1 %.not36, float %1343, float %1359, !dbg !7
  %1361 = select i1 %.not36, float %1359, float %1349, !dbg !7
  %1362 = extractvalue [2 x <8 x float>] %130, 0, !dbg !7
  %1363 = extractelement <8 x float> %1362, i64 0, !dbg !7
  %1364 = extractelement <8 x float> %1362, i64 1, !dbg !7
  %1365 = extractelement <8 x float> %1362, i64 2, !dbg !7
  %1366 = extractelement <8 x float> %1362, i64 3, !dbg !7
  %1367 = select i1 %.not, float %1364, float %1363, !dbg !7
  %1368 = bitcast float %1367 to i32, !dbg !7
  %1369 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1368), !dbg !7
  %1370 = bitcast i32 %1369 to float, !dbg !7
  %1371 = select i1 %.not, float %1363, float %1370, !dbg !7
  %1372 = select i1 %.not, float %1370, float %1364, !dbg !7
  %1373 = select i1 %.not, float %1366, float %1365, !dbg !7
  %1374 = bitcast float %1373 to i32, !dbg !7
  %1375 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1374), !dbg !7
  %1376 = bitcast i32 %1375 to float, !dbg !7
  %1377 = select i1 %.not, float %1365, float %1376, !dbg !7
  %1378 = select i1 %.not, float %1376, float %1366, !dbg !7
  %1379 = select i1 %.not36, float %1377, float %1371, !dbg !7
  %1380 = bitcast float %1379 to i32, !dbg !7
  %1381 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1380), !dbg !7
  %1382 = bitcast i32 %1381 to float, !dbg !7
  %1383 = select i1 %.not36, float %1371, float %1382, !dbg !7
  %1384 = select i1 %.not36, float %1382, float %1377, !dbg !7
  %1385 = select i1 %.not36, float %1378, float %1372, !dbg !7
  %1386 = bitcast float %1385 to i32, !dbg !7
  %1387 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1386), !dbg !7
  %1388 = bitcast i32 %1387 to float, !dbg !7
  %1389 = select i1 %.not36, float %1372, float %1388, !dbg !7
  %1390 = select i1 %.not36, float %1388, float %1378, !dbg !7
  %1391 = extractvalue [2 x <8 x float>] %127, 0, !dbg !7
  %1392 = extractelement <8 x float> %1391, i64 0, !dbg !7
  %1393 = extractelement <8 x float> %1391, i64 1, !dbg !7
  %1394 = extractelement <8 x float> %1391, i64 2, !dbg !7
  %1395 = extractelement <8 x float> %1391, i64 3, !dbg !7
  %1396 = select i1 %.not, float %1393, float %1392, !dbg !7
  %1397 = bitcast float %1396 to i32, !dbg !7
  %1398 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1397), !dbg !7
  %1399 = bitcast i32 %1398 to float, !dbg !7
  %1400 = select i1 %.not, float %1392, float %1399, !dbg !7
  %1401 = select i1 %.not, float %1399, float %1393, !dbg !7
  %1402 = select i1 %.not, float %1395, float %1394, !dbg !7
  %1403 = bitcast float %1402 to i32, !dbg !7
  %1404 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1403), !dbg !7
  %1405 = bitcast i32 %1404 to float, !dbg !7
  %1406 = select i1 %.not, float %1394, float %1405, !dbg !7
  %1407 = select i1 %.not, float %1405, float %1395, !dbg !7
  %1408 = select i1 %.not36, float %1406, float %1400, !dbg !7
  %1409 = bitcast float %1408 to i32, !dbg !7
  %1410 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1409), !dbg !7
  %1411 = select i1 %.not36, float %1407, float %1401, !dbg !7
  %1412 = bitcast float %1411 to i32, !dbg !7
  %1413 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1412), !dbg !7
  %1414 = extractvalue [2 x <1 x float>] %128, 0, !dbg !16
  %1415 = shufflevector <1 x float> %1414, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1416 = extractelement <8 x float> %1292, i64 4, !dbg !7
  %1417 = extractelement <8 x float> %1292, i64 5, !dbg !7
  %1418 = extractelement <8 x float> %1292, i64 6, !dbg !7
  %1419 = extractelement <8 x float> %1292, i64 7, !dbg !7
  %1420 = select i1 %.not, float %1417, float %1416, !dbg !7
  %1421 = bitcast float %1420 to i32, !dbg !7
  %1422 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1421), !dbg !7
  %1423 = bitcast i32 %1422 to float, !dbg !7
  %1424 = select i1 %.not, float %1416, float %1423, !dbg !7
  %1425 = select i1 %.not, float %1423, float %1417, !dbg !7
  %1426 = select i1 %.not, float %1419, float %1418, !dbg !7
  %1427 = bitcast float %1426 to i32, !dbg !7
  %1428 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1427), !dbg !7
  %1429 = bitcast i32 %1428 to float, !dbg !7
  %1430 = select i1 %.not, float %1418, float %1429, !dbg !7
  %1431 = select i1 %.not, float %1429, float %1419, !dbg !7
  %1432 = select i1 %.not36, float %1430, float %1424, !dbg !7
  %1433 = bitcast float %1432 to i32, !dbg !7
  %1434 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1433), !dbg !7
  %1435 = bitcast i32 %1434 to float, !dbg !7
  %1436 = select i1 %.not36, float %1424, float %1435, !dbg !7
  %1437 = select i1 %.not36, float %1435, float %1430, !dbg !7
  %1438 = select i1 %.not36, float %1431, float %1425, !dbg !7
  %1439 = bitcast float %1438 to i32, !dbg !7
  %1440 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1439), !dbg !7
  %1441 = bitcast i32 %1440 to float, !dbg !7
  %1442 = select i1 %.not36, float %1425, float %1441, !dbg !7
  %1443 = select i1 %.not36, float %1441, float %1431, !dbg !7
  %1444 = extractelement <8 x float> %1333, i64 4, !dbg !7
  %1445 = extractelement <8 x float> %1333, i64 5, !dbg !7
  %1446 = extractelement <8 x float> %1333, i64 6, !dbg !7
  %1447 = extractelement <8 x float> %1333, i64 7, !dbg !7
  %1448 = select i1 %.not, float %1445, float %1444, !dbg !7
  %1449 = bitcast float %1448 to i32, !dbg !7
  %1450 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1449), !dbg !7
  %1451 = bitcast i32 %1450 to float, !dbg !7
  %1452 = select i1 %.not, float %1444, float %1451, !dbg !7
  %1453 = select i1 %.not, float %1451, float %1445, !dbg !7
  %1454 = select i1 %.not, float %1447, float %1446, !dbg !7
  %1455 = bitcast float %1454 to i32, !dbg !7
  %1456 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1455), !dbg !7
  %1457 = bitcast i32 %1456 to float, !dbg !7
  %1458 = select i1 %.not, float %1446, float %1457, !dbg !7
  %1459 = select i1 %.not, float %1457, float %1447, !dbg !7
  %1460 = select i1 %.not36, float %1458, float %1452, !dbg !7
  %1461 = bitcast float %1460 to i32, !dbg !7
  %1462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1461), !dbg !7
  %1463 = bitcast i32 %1462 to float, !dbg !7
  %1464 = select i1 %.not36, float %1452, float %1463, !dbg !7
  %1465 = select i1 %.not36, float %1463, float %1458, !dbg !7
  %1466 = select i1 %.not36, float %1459, float %1453, !dbg !7
  %1467 = bitcast float %1466 to i32, !dbg !7
  %1468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1467), !dbg !7
  %1469 = bitcast i32 %1468 to float, !dbg !7
  %1470 = select i1 %.not36, float %1453, float %1469, !dbg !7
  %1471 = select i1 %.not36, float %1469, float %1459, !dbg !7
  %1472 = extractelement <8 x float> %1362, i64 4, !dbg !7
  %1473 = extractelement <8 x float> %1362, i64 5, !dbg !7
  %1474 = extractelement <8 x float> %1362, i64 6, !dbg !7
  %1475 = extractelement <8 x float> %1362, i64 7, !dbg !7
  %1476 = select i1 %.not, float %1473, float %1472, !dbg !7
  %1477 = bitcast float %1476 to i32, !dbg !7
  %1478 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1477), !dbg !7
  %1479 = bitcast i32 %1478 to float, !dbg !7
  %1480 = select i1 %.not, float %1472, float %1479, !dbg !7
  %1481 = select i1 %.not, float %1479, float %1473, !dbg !7
  %1482 = select i1 %.not, float %1475, float %1474, !dbg !7
  %1483 = bitcast float %1482 to i32, !dbg !7
  %1484 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1483), !dbg !7
  %1485 = bitcast i32 %1484 to float, !dbg !7
  %1486 = select i1 %.not, float %1474, float %1485, !dbg !7
  %1487 = select i1 %.not, float %1485, float %1475, !dbg !7
  %1488 = select i1 %.not36, float %1486, float %1480, !dbg !7
  %1489 = bitcast float %1488 to i32, !dbg !7
  %1490 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1489), !dbg !7
  %1491 = bitcast i32 %1490 to float, !dbg !7
  %1492 = select i1 %.not36, float %1480, float %1491, !dbg !7
  %1493 = select i1 %.not36, float %1491, float %1486, !dbg !7
  %1494 = select i1 %.not36, float %1487, float %1481, !dbg !7
  %1495 = bitcast float %1494 to i32, !dbg !7
  %1496 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1495), !dbg !7
  %1497 = bitcast i32 %1496 to float, !dbg !7
  %1498 = select i1 %.not36, float %1481, float %1497, !dbg !7
  %1499 = select i1 %.not36, float %1497, float %1487, !dbg !7
  %1500 = extractelement <8 x float> %1391, i64 4, !dbg !7
  %1501 = extractelement <8 x float> %1391, i64 5, !dbg !7
  %1502 = extractelement <8 x float> %1391, i64 6, !dbg !7
  %1503 = extractelement <8 x float> %1391, i64 7, !dbg !7
  %1504 = select i1 %.not, float %1501, float %1500, !dbg !7
  %1505 = bitcast float %1504 to i32, !dbg !7
  %1506 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1505), !dbg !7
  %1507 = bitcast i32 %1506 to float, !dbg !7
  %1508 = select i1 %.not, float %1500, float %1507, !dbg !7
  %1509 = select i1 %.not, float %1507, float %1501, !dbg !7
  %1510 = select i1 %.not, float %1503, float %1502, !dbg !7
  %1511 = bitcast float %1510 to i32, !dbg !7
  %1512 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1511), !dbg !7
  %1513 = bitcast i32 %1512 to float, !dbg !7
  %1514 = select i1 %.not, float %1502, float %1513, !dbg !7
  %1515 = select i1 %.not, float %1513, float %1503, !dbg !7
  %1516 = select i1 %.not36, float %1514, float %1508, !dbg !7
  %1517 = bitcast float %1516 to i32, !dbg !7
  %1518 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1517), !dbg !7
  %1519 = select i1 %.not36, float %1515, float %1509, !dbg !7
  %1520 = bitcast float %1519 to i32, !dbg !7
  %1521 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1520), !dbg !7
  %1522 = extractvalue [2 x <8 x float>] %132, 1, !dbg !7
  %1523 = extractelement <8 x float> %1522, i64 0, !dbg !7
  %1524 = extractelement <8 x float> %1522, i64 1, !dbg !7
  %1525 = extractelement <8 x float> %1522, i64 2, !dbg !7
  %1526 = extractelement <8 x float> %1522, i64 3, !dbg !7
  %1527 = select i1 %.not, float %1524, float %1523, !dbg !7
  %1528 = bitcast float %1527 to i32, !dbg !7
  %1529 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1528), !dbg !7
  %1530 = bitcast i32 %1529 to float, !dbg !7
  %1531 = select i1 %.not, float %1523, float %1530, !dbg !7
  %1532 = select i1 %.not, float %1530, float %1524, !dbg !7
  %1533 = select i1 %.not, float %1526, float %1525, !dbg !7
  %1534 = bitcast float %1533 to i32, !dbg !7
  %1535 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1534), !dbg !7
  %1536 = bitcast i32 %1535 to float, !dbg !7
  %1537 = select i1 %.not, float %1525, float %1536, !dbg !7
  %1538 = select i1 %.not, float %1536, float %1526, !dbg !7
  %1539 = select i1 %.not36, float %1537, float %1531, !dbg !7
  %1540 = bitcast float %1539 to i32, !dbg !7
  %1541 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1540), !dbg !7
  %1542 = bitcast i32 %1541 to float, !dbg !7
  %1543 = select i1 %.not36, float %1531, float %1542, !dbg !7
  %1544 = select i1 %.not36, float %1542, float %1537, !dbg !7
  %1545 = select i1 %.not36, float %1538, float %1532, !dbg !7
  %1546 = bitcast float %1545 to i32, !dbg !7
  %1547 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1546), !dbg !7
  %1548 = bitcast i32 %1547 to float, !dbg !7
  %1549 = select i1 %.not36, float %1532, float %1548, !dbg !7
  %1550 = select i1 %.not36, float %1548, float %1538, !dbg !7
  %1551 = extractvalue [2 x <8 x float>] %131, 1, !dbg !7
  %1552 = extractelement <8 x float> %1551, i64 0, !dbg !7
  %1553 = extractelement <8 x float> %1551, i64 1, !dbg !7
  %1554 = extractelement <8 x float> %1551, i64 2, !dbg !7
  %1555 = extractelement <8 x float> %1551, i64 3, !dbg !7
  %1556 = select i1 %.not, float %1553, float %1552, !dbg !7
  %1557 = bitcast float %1556 to i32, !dbg !7
  %1558 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1557), !dbg !7
  %1559 = bitcast i32 %1558 to float, !dbg !7
  %1560 = select i1 %.not, float %1552, float %1559, !dbg !7
  %1561 = select i1 %.not, float %1559, float %1553, !dbg !7
  %1562 = select i1 %.not, float %1555, float %1554, !dbg !7
  %1563 = bitcast float %1562 to i32, !dbg !7
  %1564 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1563), !dbg !7
  %1565 = bitcast i32 %1564 to float, !dbg !7
  %1566 = select i1 %.not, float %1554, float %1565, !dbg !7
  %1567 = select i1 %.not, float %1565, float %1555, !dbg !7
  %1568 = select i1 %.not36, float %1566, float %1560, !dbg !7
  %1569 = bitcast float %1568 to i32, !dbg !7
  %1570 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1569), !dbg !7
  %1571 = bitcast i32 %1570 to float, !dbg !7
  %1572 = select i1 %.not36, float %1560, float %1571, !dbg !7
  %1573 = select i1 %.not36, float %1571, float %1566, !dbg !7
  %1574 = select i1 %.not36, float %1567, float %1561, !dbg !7
  %1575 = bitcast float %1574 to i32, !dbg !7
  %1576 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1575), !dbg !7
  %1577 = bitcast i32 %1576 to float, !dbg !7
  %1578 = select i1 %.not36, float %1561, float %1577, !dbg !7
  %1579 = select i1 %.not36, float %1577, float %1567, !dbg !7
  %1580 = extractvalue [2 x <8 x float>] %130, 1, !dbg !7
  %1581 = extractelement <8 x float> %1580, i64 0, !dbg !7
  %1582 = extractelement <8 x float> %1580, i64 1, !dbg !7
  %1583 = extractelement <8 x float> %1580, i64 2, !dbg !7
  %1584 = extractelement <8 x float> %1580, i64 3, !dbg !7
  %1585 = select i1 %.not, float %1582, float %1581, !dbg !7
  %1586 = bitcast float %1585 to i32, !dbg !7
  %1587 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1586), !dbg !7
  %1588 = bitcast i32 %1587 to float, !dbg !7
  %1589 = select i1 %.not, float %1581, float %1588, !dbg !7
  %1590 = select i1 %.not, float %1588, float %1582, !dbg !7
  %1591 = select i1 %.not, float %1584, float %1583, !dbg !7
  %1592 = bitcast float %1591 to i32, !dbg !7
  %1593 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1592), !dbg !7
  %1594 = bitcast i32 %1593 to float, !dbg !7
  %1595 = select i1 %.not, float %1583, float %1594, !dbg !7
  %1596 = select i1 %.not, float %1594, float %1584, !dbg !7
  %1597 = select i1 %.not36, float %1595, float %1589, !dbg !7
  %1598 = bitcast float %1597 to i32, !dbg !7
  %1599 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1598), !dbg !7
  %1600 = bitcast i32 %1599 to float, !dbg !7
  %1601 = select i1 %.not36, float %1589, float %1600, !dbg !7
  %1602 = select i1 %.not36, float %1600, float %1595, !dbg !7
  %1603 = select i1 %.not36, float %1596, float %1590, !dbg !7
  %1604 = bitcast float %1603 to i32, !dbg !7
  %1605 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1604), !dbg !7
  %1606 = bitcast i32 %1605 to float, !dbg !7
  %1607 = select i1 %.not36, float %1590, float %1606, !dbg !7
  %1608 = select i1 %.not36, float %1606, float %1596, !dbg !7
  %1609 = extractvalue [2 x <8 x float>] %127, 1, !dbg !7
  %1610 = extractelement <8 x float> %1609, i64 0, !dbg !7
  %1611 = extractelement <8 x float> %1609, i64 1, !dbg !7
  %1612 = extractelement <8 x float> %1609, i64 2, !dbg !7
  %1613 = extractelement <8 x float> %1609, i64 3, !dbg !7
  %1614 = select i1 %.not, float %1611, float %1610, !dbg !7
  %1615 = bitcast float %1614 to i32, !dbg !7
  %1616 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1615), !dbg !7
  %1617 = bitcast i32 %1616 to float, !dbg !7
  %1618 = select i1 %.not, float %1610, float %1617, !dbg !7
  %1619 = select i1 %.not, float %1617, float %1611, !dbg !7
  %1620 = select i1 %.not, float %1613, float %1612, !dbg !7
  %1621 = bitcast float %1620 to i32, !dbg !7
  %1622 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1621), !dbg !7
  %1623 = bitcast i32 %1622 to float, !dbg !7
  %1624 = select i1 %.not, float %1612, float %1623, !dbg !7
  %1625 = select i1 %.not, float %1623, float %1613, !dbg !7
  %1626 = select i1 %.not36, float %1624, float %1618, !dbg !7
  %1627 = bitcast float %1626 to i32, !dbg !7
  %1628 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1627), !dbg !7
  %1629 = select i1 %.not36, float %1625, float %1619, !dbg !7
  %1630 = bitcast float %1629 to i32, !dbg !7
  %1631 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1630), !dbg !7
  %1632 = extractvalue [2 x <1 x float>] %128, 1, !dbg !16
  %1633 = shufflevector <1 x float> %1632, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1634 = extractelement <8 x float> %1522, i64 4, !dbg !7
  %1635 = extractelement <8 x float> %1522, i64 5, !dbg !7
  %1636 = extractelement <8 x float> %1522, i64 6, !dbg !7
  %1637 = extractelement <8 x float> %1522, i64 7, !dbg !7
  %1638 = select i1 %.not, float %1635, float %1634, !dbg !7
  %1639 = bitcast float %1638 to i32, !dbg !7
  %1640 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1639), !dbg !7
  %1641 = bitcast i32 %1640 to float, !dbg !7
  %1642 = select i1 %.not, float %1634, float %1641, !dbg !7
  %1643 = select i1 %.not, float %1641, float %1635, !dbg !7
  %1644 = select i1 %.not, float %1637, float %1636, !dbg !7
  %1645 = bitcast float %1644 to i32, !dbg !7
  %1646 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1645), !dbg !7
  %1647 = bitcast i32 %1646 to float, !dbg !7
  %1648 = select i1 %.not, float %1636, float %1647, !dbg !7
  %1649 = select i1 %.not, float %1647, float %1637, !dbg !7
  %1650 = select i1 %.not36, float %1648, float %1642, !dbg !7
  %1651 = bitcast float %1650 to i32, !dbg !7
  %1652 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1651), !dbg !7
  %1653 = bitcast i32 %1652 to float, !dbg !7
  %1654 = select i1 %.not36, float %1642, float %1653, !dbg !7
  %1655 = select i1 %.not36, float %1653, float %1648, !dbg !7
  %1656 = select i1 %.not36, float %1649, float %1643, !dbg !7
  %1657 = bitcast float %1656 to i32, !dbg !7
  %1658 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1657), !dbg !7
  %1659 = bitcast i32 %1658 to float, !dbg !7
  %1660 = select i1 %.not36, float %1643, float %1659, !dbg !7
  %1661 = select i1 %.not36, float %1659, float %1649, !dbg !7
  %1662 = extractelement <8 x float> %1551, i64 4, !dbg !7
  %1663 = extractelement <8 x float> %1551, i64 5, !dbg !7
  %1664 = extractelement <8 x float> %1551, i64 6, !dbg !7
  %1665 = extractelement <8 x float> %1551, i64 7, !dbg !7
  %1666 = select i1 %.not, float %1663, float %1662, !dbg !7
  %1667 = bitcast float %1666 to i32, !dbg !7
  %1668 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1667), !dbg !7
  %1669 = bitcast i32 %1668 to float, !dbg !7
  %1670 = select i1 %.not, float %1662, float %1669, !dbg !7
  %1671 = select i1 %.not, float %1669, float %1663, !dbg !7
  %1672 = select i1 %.not, float %1665, float %1664, !dbg !7
  %1673 = bitcast float %1672 to i32, !dbg !7
  %1674 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1673), !dbg !7
  %1675 = bitcast i32 %1674 to float, !dbg !7
  %1676 = select i1 %.not, float %1664, float %1675, !dbg !7
  %1677 = select i1 %.not, float %1675, float %1665, !dbg !7
  %1678 = select i1 %.not36, float %1676, float %1670, !dbg !7
  %1679 = bitcast float %1678 to i32, !dbg !7
  %1680 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1679), !dbg !7
  %1681 = bitcast i32 %1680 to float, !dbg !7
  %1682 = select i1 %.not36, float %1670, float %1681, !dbg !7
  %1683 = select i1 %.not36, float %1681, float %1676, !dbg !7
  %1684 = select i1 %.not36, float %1677, float %1671, !dbg !7
  %1685 = bitcast float %1684 to i32, !dbg !7
  %1686 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1685), !dbg !7
  %1687 = bitcast i32 %1686 to float, !dbg !7
  %1688 = select i1 %.not36, float %1671, float %1687, !dbg !7
  %1689 = select i1 %.not36, float %1687, float %1677, !dbg !7
  %1690 = extractelement <8 x float> %1580, i64 4, !dbg !7
  %1691 = extractelement <8 x float> %1580, i64 5, !dbg !7
  %1692 = extractelement <8 x float> %1580, i64 6, !dbg !7
  %1693 = extractelement <8 x float> %1580, i64 7, !dbg !7
  %1694 = select i1 %.not, float %1691, float %1690, !dbg !7
  %1695 = bitcast float %1694 to i32, !dbg !7
  %1696 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1695), !dbg !7
  %1697 = bitcast i32 %1696 to float, !dbg !7
  %1698 = select i1 %.not, float %1690, float %1697, !dbg !7
  %1699 = select i1 %.not, float %1697, float %1691, !dbg !7
  %1700 = select i1 %.not, float %1693, float %1692, !dbg !7
  %1701 = bitcast float %1700 to i32, !dbg !7
  %1702 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1701), !dbg !7
  %1703 = bitcast i32 %1702 to float, !dbg !7
  %1704 = select i1 %.not, float %1692, float %1703, !dbg !7
  %1705 = select i1 %.not, float %1703, float %1693, !dbg !7
  %1706 = select i1 %.not36, float %1704, float %1698, !dbg !7
  %1707 = bitcast float %1706 to i32, !dbg !7
  %1708 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1707), !dbg !7
  %1709 = bitcast i32 %1708 to float, !dbg !7
  %1710 = select i1 %.not36, float %1698, float %1709, !dbg !7
  %1711 = select i1 %.not36, float %1709, float %1704, !dbg !7
  %1712 = select i1 %.not36, float %1705, float %1699, !dbg !7
  %1713 = bitcast float %1712 to i32, !dbg !7
  %1714 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1713), !dbg !7
  %1715 = bitcast i32 %1714 to float, !dbg !7
  %1716 = select i1 %.not36, float %1699, float %1715, !dbg !7
  %1717 = select i1 %.not36, float %1715, float %1705, !dbg !7
  %1718 = extractelement <8 x float> %1609, i64 4, !dbg !7
  %1719 = extractelement <8 x float> %1609, i64 5, !dbg !7
  %1720 = extractelement <8 x float> %1609, i64 6, !dbg !7
  %1721 = extractelement <8 x float> %1609, i64 7, !dbg !7
  %1722 = select i1 %.not, float %1719, float %1718, !dbg !7
  %1723 = bitcast float %1722 to i32, !dbg !7
  %1724 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1723), !dbg !7
  %1725 = bitcast i32 %1724 to float, !dbg !7
  %1726 = select i1 %.not, float %1718, float %1725, !dbg !7
  %1727 = select i1 %.not, float %1725, float %1719, !dbg !7
  %1728 = select i1 %.not, float %1721, float %1720, !dbg !7
  %1729 = bitcast float %1728 to i32, !dbg !7
  %1730 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1305, i32 %1729), !dbg !7
  %1731 = bitcast i32 %1730 to float, !dbg !7
  %1732 = select i1 %.not, float %1720, float %1731, !dbg !7
  %1733 = select i1 %.not, float %1731, float %1721, !dbg !7
  %1734 = select i1 %.not36, float %1732, float %1726, !dbg !7
  %1735 = bitcast float %1734 to i32, !dbg !7
  %1736 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1735), !dbg !7
  %1737 = select i1 %.not36, float %1733, float %1727, !dbg !7
  %1738 = bitcast float %1737 to i32, !dbg !7
  %1739 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1321, i32 %1738), !dbg !7
  %1740 = bitcast i32 %1736 to float, !dbg !7
  %1741 = select i1 %.not36, float %1726, float %1740, !dbg !7
  %1742 = insertelement <4 x float> poison, float %1741, i64 0, !dbg !16
  %1743 = bitcast i32 %1739 to float, !dbg !7
  %1744 = select i1 %.not36, float %1727, float %1743, !dbg !7
  %1745 = insertelement <4 x float> %1742, float %1744, i64 1, !dbg !16
  %1746 = select i1 %.not36, float %1740, float %1732, !dbg !7
  %1747 = insertelement <4 x float> %1745, float %1746, i64 2, !dbg !16
  %1748 = select i1 %.not36, float %1743, float %1733, !dbg !7
  %1749 = insertelement <4 x float> %1747, float %1748, i64 3, !dbg !16
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1633, !dbg !16
  %normalized = fmul <4 x float> %1749, %row.reciprocal, !dbg !16
  %1750 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !17
  %1751 = extractelement <4 x half> %1750, i64 3, !dbg !17
  %1752 = extractelement <4 x half> %1750, i64 2, !dbg !17
  %1753 = extractelement <4 x half> %1750, i64 1, !dbg !17
  %1754 = extractelement <4 x half> %1750, i64 0, !dbg !17
  %1755 = bitcast i32 %1628 to float, !dbg !7
  %1756 = select i1 %.not36, float %1618, float %1755, !dbg !7
  %1757 = insertelement <4 x float> poison, float %1756, i64 0, !dbg !16
  %1758 = bitcast i32 %1631 to float, !dbg !7
  %1759 = select i1 %.not36, float %1619, float %1758, !dbg !7
  %1760 = insertelement <4 x float> %1757, float %1759, i64 1, !dbg !16
  %1761 = select i1 %.not36, float %1755, float %1624, !dbg !7
  %1762 = insertelement <4 x float> %1760, float %1761, i64 2, !dbg !16
  %1763 = select i1 %.not36, float %1758, float %1625, !dbg !7
  %1764 = insertelement <4 x float> %1762, float %1763, i64 3, !dbg !16
  %normalized37 = fmul <4 x float> %1764, %row.reciprocal, !dbg !16
  %1765 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !17
  %1766 = extractelement <4 x half> %1765, i64 3, !dbg !17
  %1767 = extractelement <4 x half> %1765, i64 2, !dbg !17
  %1768 = extractelement <4 x half> %1765, i64 1, !dbg !17
  %1769 = extractelement <4 x half> %1765, i64 0, !dbg !17
  %1770 = bitcast i32 %1518 to float, !dbg !7
  %1771 = select i1 %.not36, float %1508, float %1770, !dbg !7
  %1772 = insertelement <4 x float> poison, float %1771, i64 0, !dbg !16
  %1773 = bitcast i32 %1521 to float, !dbg !7
  %1774 = select i1 %.not36, float %1509, float %1773, !dbg !7
  %1775 = insertelement <4 x float> %1772, float %1774, i64 1, !dbg !16
  %1776 = select i1 %.not36, float %1770, float %1514, !dbg !7
  %1777 = insertelement <4 x float> %1775, float %1776, i64 2, !dbg !16
  %1778 = select i1 %.not36, float %1773, float %1515, !dbg !7
  %1779 = insertelement <4 x float> %1777, float %1778, i64 3, !dbg !16
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1415, !dbg !16
  %normalized39 = fmul <4 x float> %1779, %row.reciprocal38, !dbg !16
  %1780 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !17
  %1781 = extractelement <4 x half> %1780, i64 3, !dbg !17
  %1782 = extractelement <4 x half> %1780, i64 2, !dbg !17
  %1783 = extractelement <4 x half> %1780, i64 1, !dbg !17
  %1784 = extractelement <4 x half> %1780, i64 0, !dbg !17
  %1785 = bitcast i32 %1410 to float, !dbg !7
  %1786 = select i1 %.not36, float %1400, float %1785, !dbg !7
  %1787 = insertelement <4 x float> poison, float %1786, i64 0, !dbg !16
  %1788 = bitcast i32 %1413 to float, !dbg !7
  %1789 = select i1 %.not36, float %1401, float %1788, !dbg !7
  %1790 = insertelement <4 x float> %1787, float %1789, i64 1, !dbg !16
  %1791 = select i1 %.not36, float %1785, float %1406, !dbg !7
  %1792 = insertelement <4 x float> %1790, float %1791, i64 2, !dbg !16
  %1793 = select i1 %.not36, float %1788, float %1407, !dbg !7
  %1794 = insertelement <4 x float> %1792, float %1793, i64 3, !dbg !16
  %normalized40 = fmul <4 x float> %1794, %row.reciprocal38, !dbg !16
  %1795 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !17
  %1796 = extractelement <4 x half> %1795, i64 3, !dbg !17
  %1797 = extractelement <4 x half> %1795, i64 2, !dbg !17
  %1798 = extractelement <4 x half> %1795, i64 1, !dbg !17
  %1799 = extractelement <4 x half> %1795, i64 0, !dbg !17
  %1800 = add nsw i64 %16, %22, !dbg !18
  %1801 = shl i64 %1800, 19, !dbg !18
  %1802 = and i64 %1801, 16252928, !dbg !18
  %address.thread841 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local842 = zext nneg i32 %address.thread841 to i64, !dbg !8
  %address.local843 = add i64 %25, %address.local842, !dbg !18
  %1803 = lshr i64 %address.local842, 2, !dbg !18
  %address.local845 = and i64 %1803, 48, !dbg !18
  %address.local846 = add i64 %address.local843, %address.local845, !dbg !18
  %address.local847 = and i64 %address.local842, 240, !dbg !18
  %address.local848 = sub i64 %address.local846, %address.local847, !dbg !18
  %address.local849 = add i64 %address.local848, %26, !dbg !18
  %address.local850 = shl nuw nsw i64 %address.local849, 7, !dbg !18
  %address.local851 = add nuw nsw i64 %1802, %address.local850, !dbg !18
  %address.local852 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local853 = lshr i64 %address.local842, 2, !dbg !10
  %address.local854 = and i64 %address.local853, 12, !dbg !10
  %address.local855 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local854, !dbg !18
  store half %1799, ptr addrspace(1) %address.local855, align 2, !dbg !18
  %address.local856 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local857 = or disjoint i64 %address.local854, 1, !dbg !18
  %address.local858 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local856, i64 %address.local857, !dbg !18
  store half %1798, ptr addrspace(1) %address.local858, align 2, !dbg !18
  %address.local859 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local860 = or disjoint i64 %address.local854, 2, !dbg !18
  %address.local861 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local860, !dbg !18
  store half %1797, ptr addrspace(1) %address.local861, align 2, !dbg !18
  %address.local862 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local863 = or disjoint i64 %address.local854, 3, !dbg !18
  %address.local864 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local862, i64 %address.local863, !dbg !18
  store half %1796, ptr addrspace(1) %address.local864, align 2, !dbg !18
  %address.local865 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local866 = or disjoint i64 %address.local854, 16, !dbg !18
  %address.local867 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local865, i64 %address.local866, !dbg !18
  store half %1784, ptr addrspace(1) %address.local867, align 2, !dbg !18
  %address.local868 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local869 = or disjoint i64 %address.local854, 17, !dbg !18
  %address.local870 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local868, i64 %address.local869, !dbg !18
  store half %1783, ptr addrspace(1) %address.local870, align 2, !dbg !18
  %address.local871 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local872 = or disjoint i64 %address.local854, 18, !dbg !18
  %address.local873 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local871, i64 %address.local872, !dbg !18
  store half %1782, ptr addrspace(1) %address.local873, align 2, !dbg !18
  %address.local874 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local875 = or disjoint i64 %address.local854, 19, !dbg !18
  %address.local876 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local874, i64 %address.local875, !dbg !18
  store half %1781, ptr addrspace(1) %address.local876, align 2, !dbg !18
  %address.local877 = shl i64 %address.local849, 7, !dbg !18
  %address.local878 = add i64 %address.local877, 8192, !dbg !18
  %address.local879 = add nuw nsw i64 %1802, %address.local878, !dbg !18
  %address.local880 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local881 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local854, !dbg !18
  store half %1769, ptr addrspace(1) %address.local881, align 2, !dbg !18
  %address.local882 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local882, i64 %address.local857, !dbg !18
  store half %1768, ptr addrspace(1) %address.local883, align 2, !dbg !18
  %address.local884 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local885 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local884, i64 %address.local860, !dbg !18
  store half %1767, ptr addrspace(1) %address.local885, align 2, !dbg !18
  %address.local886 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local887 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local886, i64 %address.local863, !dbg !18
  store half %1766, ptr addrspace(1) %address.local887, align 2, !dbg !18
  %address.local888 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local888, i64 %address.local866, !dbg !18
  store half %1754, ptr addrspace(1) %address.local889, align 2, !dbg !18
  %address.local890 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local891 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local890, i64 %address.local869, !dbg !18
  store half %1753, ptr addrspace(1) %address.local891, align 2, !dbg !18
  %address.local892 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local893 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local892, i64 %address.local872, !dbg !18
  store half %1752, ptr addrspace(1) %address.local893, align 2, !dbg !18
  %address.local894 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local895 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local875, !dbg !18
  store half %1751, ptr addrspace(1) %address.local895, align 2, !dbg !18
  %1804 = insertelement <4 x float> poison, float %1710, i64 0, !dbg !16
  %1805 = insertelement <4 x float> %1804, float %1716, i64 1, !dbg !16
  %1806 = insertelement <4 x float> %1805, float %1711, i64 2, !dbg !16
  %1807 = insertelement <4 x float> %1806, float %1717, i64 3, !dbg !16
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1633, !dbg !16
  %normalized42 = fmul <4 x float> %1807, %row.reciprocal41, !dbg !16
  %1808 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !17
  %1809 = extractelement <4 x half> %1808, i64 3, !dbg !17
  %1810 = extractelement <4 x half> %1808, i64 2, !dbg !17
  %1811 = extractelement <4 x half> %1808, i64 1, !dbg !17
  %1812 = extractelement <4 x half> %1808, i64 0, !dbg !17
  %1813 = insertelement <4 x float> poison, float %1601, i64 0, !dbg !16
  %1814 = insertelement <4 x float> %1813, float %1607, i64 1, !dbg !16
  %1815 = insertelement <4 x float> %1814, float %1602, i64 2, !dbg !16
  %1816 = insertelement <4 x float> %1815, float %1608, i64 3, !dbg !16
  %normalized43 = fmul <4 x float> %1816, %row.reciprocal41, !dbg !16
  %1817 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !17
  %1818 = extractelement <4 x half> %1817, i64 3, !dbg !17
  %1819 = extractelement <4 x half> %1817, i64 2, !dbg !17
  %1820 = extractelement <4 x half> %1817, i64 1, !dbg !17
  %1821 = extractelement <4 x half> %1817, i64 0, !dbg !17
  %1822 = insertelement <4 x float> poison, float %1492, i64 0, !dbg !16
  %1823 = insertelement <4 x float> %1822, float %1498, i64 1, !dbg !16
  %1824 = insertelement <4 x float> %1823, float %1493, i64 2, !dbg !16
  %1825 = insertelement <4 x float> %1824, float %1499, i64 3, !dbg !16
  %row.reciprocal44 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1415, !dbg !16
  %normalized45 = fmul <4 x float> %1825, %row.reciprocal44, !dbg !16
  %1826 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !17
  %1827 = extractelement <4 x half> %1826, i64 3, !dbg !17
  %1828 = extractelement <4 x half> %1826, i64 2, !dbg !17
  %1829 = extractelement <4 x half> %1826, i64 1, !dbg !17
  %1830 = extractelement <4 x half> %1826, i64 0, !dbg !17
  %1831 = insertelement <4 x float> poison, float %1383, i64 0, !dbg !16
  %1832 = insertelement <4 x float> %1831, float %1389, i64 1, !dbg !16
  %1833 = insertelement <4 x float> %1832, float %1384, i64 2, !dbg !16
  %1834 = insertelement <4 x float> %1833, float %1390, i64 3, !dbg !16
  %normalized46 = fmul <4 x float> %1834, %row.reciprocal44, !dbg !16
  %1835 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !17
  %1836 = extractelement <4 x half> %1835, i64 3, !dbg !17
  %1837 = extractelement <4 x half> %1835, i64 2, !dbg !17
  %1838 = extractelement <4 x half> %1835, i64 1, !dbg !17
  %1839 = extractelement <4 x half> %1835, i64 0, !dbg !17
  %1840 = shl nuw nsw i64 %24, 19, !dbg !18
  %address.thread896 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local897 = zext nneg i32 %address.thread896 to i64, !dbg !8
  %address.local898 = add i64 %25, %address.local897, !dbg !18
  %1841 = lshr i64 %address.local897, 2, !dbg !18
  %address.local900 = and i64 %1841, 48, !dbg !18
  %address.local901 = add i64 %address.local898, %address.local900, !dbg !18
  %address.local902 = and i64 %address.local897, 240, !dbg !18
  %address.local903 = sub i64 %address.local901, %address.local902, !dbg !18
  %address.local904 = add i64 %address.local903, %26, !dbg !18
  %address.local905 = shl nuw nsw i64 %address.local904, 7, !dbg !18
  %address.local906 = add nuw nsw i64 %1840, %address.local905, !dbg !18
  %address.local907 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local908 = lshr i64 %address.local897, 2, !dbg !10
  %address.local909 = and i64 %address.local908, 12, !dbg !10
  %address.local910 = or disjoint i64 %address.local909, 32, !dbg !18
  %address.local911 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local907, i64 %address.local910, !dbg !18
  store half %1839, ptr addrspace(1) %address.local911, align 2, !dbg !18
  %address.local912 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local913 = or disjoint i64 %address.local909, 33, !dbg !18
  %address.local914 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local912, i64 %address.local913, !dbg !18
  store half %1838, ptr addrspace(1) %address.local914, align 2, !dbg !18
  %address.local915 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local916 = or disjoint i64 %address.local909, 34, !dbg !18
  %address.local917 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local915, i64 %address.local916, !dbg !18
  store half %1837, ptr addrspace(1) %address.local917, align 2, !dbg !18
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local919 = or disjoint i64 %address.local909, 35, !dbg !18
  %address.local920 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local918, i64 %address.local919, !dbg !18
  store half %1836, ptr addrspace(1) %address.local920, align 2, !dbg !18
  %address.local921 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local922 = or disjoint i64 %address.local909, 48, !dbg !18
  %address.local923 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local921, i64 %address.local922, !dbg !18
  store half %1830, ptr addrspace(1) %address.local923, align 2, !dbg !18
  %address.local924 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local925 = or disjoint i64 %address.local909, 49, !dbg !18
  %address.local926 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local924, i64 %address.local925, !dbg !18
  store half %1829, ptr addrspace(1) %address.local926, align 2, !dbg !18
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local928 = or disjoint i64 %address.local909, 50, !dbg !18
  %address.local929 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local927, i64 %address.local928, !dbg !18
  store half %1828, ptr addrspace(1) %address.local929, align 2, !dbg !18
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local931 = or i64 %address.local908, 51, !dbg !18
  %address.local932 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local930, i64 %address.local931, !dbg !18
  store half %1827, ptr addrspace(1) %address.local932, align 2, !dbg !18
  %address.local933 = shl i64 %address.local904, 7, !dbg !18
  %address.local934 = add i64 %address.local933, 8192, !dbg !18
  %address.local935 = add nuw nsw i64 %1840, %address.local934, !dbg !18
  %address.local936 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local936, i64 %address.local910, !dbg !18
  store half %1821, ptr addrspace(1) %address.local937, align 2, !dbg !18
  %address.local938 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local938, i64 %address.local913, !dbg !18
  store half %1820, ptr addrspace(1) %address.local939, align 2, !dbg !18
  %address.local940 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local941 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local940, i64 %address.local916, !dbg !18
  store half %1819, ptr addrspace(1) %address.local941, align 2, !dbg !18
  %address.local942 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local943 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local942, i64 %address.local919, !dbg !18
  store half %1818, ptr addrspace(1) %address.local943, align 2, !dbg !18
  %address.local944 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local922, !dbg !18
  store half %1812, ptr addrspace(1) %address.local945, align 2, !dbg !18
  %address.local946 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local946, i64 %address.local925, !dbg !18
  store half %1811, ptr addrspace(1) %address.local947, align 2, !dbg !18
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local948, i64 %address.local928, !dbg !18
  store half %1810, ptr addrspace(1) %address.local949, align 2, !dbg !18
  %address.local950 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local950, i64 %address.local931, !dbg !18
  store half %1809, ptr addrspace(1) %address.local951, align 2, !dbg !18
  %1842 = insertelement <4 x float> poison, float %1682, i64 0, !dbg !16
  %1843 = insertelement <4 x float> %1842, float %1688, i64 1, !dbg !16
  %1844 = insertelement <4 x float> %1843, float %1683, i64 2, !dbg !16
  %1845 = insertelement <4 x float> %1844, float %1689, i64 3, !dbg !16
  %row.reciprocal47 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1633, !dbg !16
  %normalized48 = fmul <4 x float> %1845, %row.reciprocal47, !dbg !16
  %1846 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !17
  %1847 = extractelement <4 x half> %1846, i64 3, !dbg !17
  %1848 = extractelement <4 x half> %1846, i64 2, !dbg !17
  %1849 = extractelement <4 x half> %1846, i64 1, !dbg !17
  %1850 = extractelement <4 x half> %1846, i64 0, !dbg !17
  %1851 = insertelement <4 x float> poison, float %1572, i64 0, !dbg !16
  %1852 = insertelement <4 x float> %1851, float %1578, i64 1, !dbg !16
  %1853 = insertelement <4 x float> %1852, float %1573, i64 2, !dbg !16
  %1854 = insertelement <4 x float> %1853, float %1579, i64 3, !dbg !16
  %normalized49 = fmul <4 x float> %1854, %row.reciprocal47, !dbg !16
  %1855 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !17
  %1856 = extractelement <4 x half> %1855, i64 3, !dbg !17
  %1857 = extractelement <4 x half> %1855, i64 2, !dbg !17
  %1858 = extractelement <4 x half> %1855, i64 1, !dbg !17
  %1859 = extractelement <4 x half> %1855, i64 0, !dbg !17
  %1860 = insertelement <4 x float> poison, float %1464, i64 0, !dbg !16
  %1861 = insertelement <4 x float> %1860, float %1470, i64 1, !dbg !16
  %1862 = insertelement <4 x float> %1861, float %1465, i64 2, !dbg !16
  %1863 = insertelement <4 x float> %1862, float %1471, i64 3, !dbg !16
  %row.reciprocal50 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1415, !dbg !16
  %normalized51 = fmul <4 x float> %1863, %row.reciprocal50, !dbg !16
  %1864 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !17
  %1865 = extractelement <4 x half> %1864, i64 3, !dbg !17
  %1866 = extractelement <4 x half> %1864, i64 2, !dbg !17
  %1867 = extractelement <4 x half> %1864, i64 1, !dbg !17
  %1868 = extractelement <4 x half> %1864, i64 0, !dbg !17
  %1869 = insertelement <4 x float> poison, float %1354, i64 0, !dbg !16
  %1870 = insertelement <4 x float> %1869, float %1360, i64 1, !dbg !16
  %1871 = insertelement <4 x float> %1870, float %1355, i64 2, !dbg !16
  %1872 = insertelement <4 x float> %1871, float %1361, i64 3, !dbg !16
  %normalized52 = fmul <4 x float> %1872, %row.reciprocal50, !dbg !16
  %1873 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !17
  %1874 = extractelement <4 x half> %1873, i64 3, !dbg !17
  %1875 = extractelement <4 x half> %1873, i64 2, !dbg !17
  %1876 = extractelement <4 x half> %1873, i64 1, !dbg !17
  %1877 = extractelement <4 x half> %1873, i64 0, !dbg !17
  %1878 = shl nuw nsw i64 %24, 19, !dbg !18
  %address.thread952 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local953 = zext nneg i32 %address.thread952 to i64, !dbg !8
  %address.local954 = add i64 %25, %address.local953, !dbg !18
  %1879 = lshr i64 %address.local953, 2, !dbg !18
  %address.local956 = and i64 %1879, 48, !dbg !18
  %address.local957 = add i64 %address.local954, %address.local956, !dbg !18
  %address.local958 = and i64 %address.local953, 240, !dbg !18
  %address.local959 = sub i64 %address.local957, %address.local958, !dbg !18
  %address.local960 = add i64 %address.local959, %26, !dbg !18
  %address.local961 = shl nuw nsw i64 %address.local960, 7, !dbg !18
  %address.local962 = add nuw nsw i64 %1878, %address.local961, !dbg !18
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local964 = lshr i64 %address.local953, 2, !dbg !10
  %address.local965 = and i64 %address.local964, 12, !dbg !10
  %address.local966 = or disjoint i64 %address.local965, 64, !dbg !18
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local963, i64 %address.local966, !dbg !18
  store half %1877, ptr addrspace(1) %address.local967, align 2, !dbg !18
  %address.local968 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local969 = or disjoint i64 %address.local965, 65, !dbg !18
  %address.local970 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local968, i64 %address.local969, !dbg !18
  store half %1876, ptr addrspace(1) %address.local970, align 2, !dbg !18
  %address.local971 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local972 = or disjoint i64 %address.local965, 66, !dbg !18
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local971, i64 %address.local972, !dbg !18
  store half %1875, ptr addrspace(1) %address.local973, align 2, !dbg !18
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local975 = or disjoint i64 %address.local965, 67, !dbg !18
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local974, i64 %address.local975, !dbg !18
  store half %1874, ptr addrspace(1) %address.local976, align 2, !dbg !18
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local978 = or disjoint i64 %address.local965, 80, !dbg !18
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local977, i64 %address.local978, !dbg !18
  store half %1868, ptr addrspace(1) %address.local979, align 2, !dbg !18
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local981 = or disjoint i64 %address.local965, 81, !dbg !18
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local980, i64 %address.local981, !dbg !18
  store half %1867, ptr addrspace(1) %address.local982, align 2, !dbg !18
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local984 = or disjoint i64 %address.local965, 82, !dbg !18
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local983, i64 %address.local984, !dbg !18
  store half %1866, ptr addrspace(1) %address.local985, align 2, !dbg !18
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local987 = or disjoint i64 %address.local965, 83, !dbg !18
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local986, i64 %address.local987, !dbg !18
  store half %1865, ptr addrspace(1) %address.local988, align 2, !dbg !18
  %address.local989 = shl i64 %address.local960, 7, !dbg !18
  %address.local990 = add i64 %address.local989, 8192, !dbg !18
  %address.local991 = add nuw nsw i64 %1878, %address.local990, !dbg !18
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local992, i64 %address.local966, !dbg !18
  store half %1859, ptr addrspace(1) %address.local993, align 2, !dbg !18
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local994, i64 %address.local969, !dbg !18
  store half %1858, ptr addrspace(1) %address.local995, align 2, !dbg !18
  %address.local996 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local996, i64 %address.local972, !dbg !18
  store half %1857, ptr addrspace(1) %address.local997, align 2, !dbg !18
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local999 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local998, i64 %address.local975, !dbg !18
  store half %1856, ptr addrspace(1) %address.local999, align 2, !dbg !18
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1000, i64 %address.local978, !dbg !18
  store half %1850, ptr addrspace(1) %address.local1001, align 2, !dbg !18
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1002, i64 %address.local981, !dbg !18
  store half %1849, ptr addrspace(1) %address.local1003, align 2, !dbg !18
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local984, !dbg !18
  store half %1848, ptr addrspace(1) %address.local1005, align 2, !dbg !18
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1006, i64 %address.local987, !dbg !18
  store half %1847, ptr addrspace(1) %address.local1007, align 2, !dbg !18
  %1880 = insertelement <4 x float> poison, float %1654, i64 0, !dbg !16
  %1881 = insertelement <4 x float> %1880, float %1660, i64 1, !dbg !16
  %1882 = insertelement <4 x float> %1881, float %1655, i64 2, !dbg !16
  %1883 = insertelement <4 x float> %1882, float %1661, i64 3, !dbg !16
  %row.reciprocal53 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1633, !dbg !16
  %normalized54 = fmul <4 x float> %1883, %row.reciprocal53, !dbg !16
  %1884 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !17
  %1885 = extractelement <4 x half> %1884, i64 3, !dbg !17
  %1886 = extractelement <4 x half> %1884, i64 2, !dbg !17
  %1887 = extractelement <4 x half> %1884, i64 1, !dbg !17
  %1888 = extractelement <4 x half> %1884, i64 0, !dbg !17
  %1889 = insertelement <4 x float> poison, float %1543, i64 0, !dbg !16
  %1890 = insertelement <4 x float> %1889, float %1549, i64 1, !dbg !16
  %1891 = insertelement <4 x float> %1890, float %1544, i64 2, !dbg !16
  %1892 = insertelement <4 x float> %1891, float %1550, i64 3, !dbg !16
  %normalized55 = fmul <4 x float> %1892, %row.reciprocal53, !dbg !16
  %1893 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !17
  %1894 = extractelement <4 x half> %1893, i64 3, !dbg !17
  %1895 = extractelement <4 x half> %1893, i64 2, !dbg !17
  %1896 = extractelement <4 x half> %1893, i64 1, !dbg !17
  %1897 = extractelement <4 x half> %1893, i64 0, !dbg !17
  %1898 = insertelement <4 x float> poison, float %1436, i64 0, !dbg !16
  %1899 = insertelement <4 x float> %1898, float %1442, i64 1, !dbg !16
  %1900 = insertelement <4 x float> %1899, float %1437, i64 2, !dbg !16
  %1901 = insertelement <4 x float> %1900, float %1443, i64 3, !dbg !16
  %row.reciprocal56 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1415, !dbg !16
  %normalized57 = fmul <4 x float> %1901, %row.reciprocal56, !dbg !16
  %1902 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !17
  %1903 = extractelement <4 x half> %1902, i64 3, !dbg !17
  %1904 = extractelement <4 x half> %1902, i64 2, !dbg !17
  %1905 = extractelement <4 x half> %1902, i64 1, !dbg !17
  %1906 = extractelement <4 x half> %1902, i64 0, !dbg !17
  %1907 = insertelement <4 x float> poison, float %1325, i64 0, !dbg !16
  %1908 = insertelement <4 x float> %1907, float %1331, i64 1, !dbg !16
  %1909 = insertelement <4 x float> %1908, float %1326, i64 2, !dbg !16
  %1910 = insertelement <4 x float> %1909, float %1332, i64 3, !dbg !16
  %normalized58 = fmul <4 x float> %1910, %row.reciprocal56, !dbg !16
  %1911 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !17
  %1912 = extractelement <4 x half> %1911, i64 3, !dbg !17
  %1913 = extractelement <4 x half> %1911, i64 2, !dbg !17
  %1914 = extractelement <4 x half> %1911, i64 1, !dbg !17
  %1915 = extractelement <4 x half> %1911, i64 0, !dbg !17
  %1916 = shl nuw nsw i64 %24, 19, !dbg !18
  %address.thread1008 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1009 = zext nneg i32 %address.thread1008 to i64, !dbg !8
  %address.local1010 = add i64 %25, %address.local1009, !dbg !18
  %1917 = lshr i64 %address.local1009, 2, !dbg !18
  %address.local1012 = and i64 %1917, 48, !dbg !18
  %address.local1013 = add i64 %address.local1010, %address.local1012, !dbg !18
  %address.local1014 = and i64 %address.local1009, 240, !dbg !18
  %address.local1015 = sub i64 %address.local1013, %address.local1014, !dbg !18
  %address.local1016 = add i64 %address.local1015, %26, !dbg !18
  %address.local1017 = shl nuw nsw i64 %address.local1016, 7, !dbg !18
  %address.local1018 = add nuw nsw i64 %1916, %address.local1017, !dbg !18
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1020 = lshr i64 %address.local1009, 2, !dbg !10
  %address.local1021 = and i64 %address.local1020, 12, !dbg !10
  %address.local1022 = or disjoint i64 %address.local1021, 96, !dbg !18
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1019, i64 %address.local1022, !dbg !18
  store half %1915, ptr addrspace(1) %address.local1023, align 2, !dbg !18
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1025 = or disjoint i64 %address.local1021, 97, !dbg !18
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1024, i64 %address.local1025, !dbg !18
  store half %1914, ptr addrspace(1) %address.local1026, align 2, !dbg !18
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1028 = or disjoint i64 %address.local1021, 98, !dbg !18
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1027, i64 %address.local1028, !dbg !18
  store half %1913, ptr addrspace(1) %address.local1029, align 2, !dbg !18
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1031 = or disjoint i64 %address.local1021, 99, !dbg !18
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1030, i64 %address.local1031, !dbg !18
  store half %1912, ptr addrspace(1) %address.local1032, align 2, !dbg !18
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1034 = or disjoint i64 %address.local1021, 112, !dbg !18
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local1034, !dbg !18
  store half %1906, ptr addrspace(1) %address.local1035, align 2, !dbg !18
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1037 = or disjoint i64 %address.local1021, 113, !dbg !18
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1036, i64 %address.local1037, !dbg !18
  store half %1905, ptr addrspace(1) %address.local1038, align 2, !dbg !18
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1040 = or disjoint i64 %address.local1021, 114, !dbg !18
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1040, !dbg !18
  store half %1904, ptr addrspace(1) %address.local1041, align 2, !dbg !18
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1043 = or i64 %address.local1020, 115, !dbg !18
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1042, i64 %address.local1043, !dbg !18
  store half %1903, ptr addrspace(1) %address.local1044, align 2, !dbg !18
  %address.local1045 = shl i64 %address.local1016, 7, !dbg !18
  %address.local1046 = add i64 %address.local1045, 8192, !dbg !18
  %address.local1047 = add nuw nsw i64 %1916, %address.local1046, !dbg !18
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1048, i64 %address.local1022, !dbg !18
  store half %1897, ptr addrspace(1) %address.local1049, align 2, !dbg !18
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1025, !dbg !18
  store half %1896, ptr addrspace(1) %address.local1051, align 2, !dbg !18
  %address.local1052 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1052, i64 %address.local1028, !dbg !18
  store half %1895, ptr addrspace(1) %address.local1053, align 2, !dbg !18
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1055 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1031, !dbg !18
  store half %1894, ptr addrspace(1) %address.local1055, align 2, !dbg !18
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1034, !dbg !18
  store half %1888, ptr addrspace(1) %address.local1057, align 2, !dbg !18
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1058, i64 %address.local1037, !dbg !18
  store half %1887, ptr addrspace(1) %address.local1059, align 2, !dbg !18
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1060, i64 %address.local1040, !dbg !18
  store half %1886, ptr addrspace(1) %address.local1061, align 2, !dbg !18
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1062, i64 %address.local1043, !dbg !18
  store half %1885, ptr addrspace(1) %address.local1063, align 2, !dbg !18
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!31}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-162611-z5os6mty/grid-000001")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 57, column: 24, scope: !4)
!8 = !DILocation(line: 67, column: 22, scope: !4)
!10 = !DILocation(line: 69, column: 24, scope: !4)
!12 = !DILocation(line: 68, column: 22, scope: !4)
!13 = !DILocation(line: 87, column: 23, scope: !4)
!14 = !DILocation(line: 70, column: 26, scope: !4)
!15 = !DILocation(line: 86, column: 20, scope: !4)
!16 = !DILocation(line: 92, column: 23, scope: !4)
!17 = !DILocation(line: 93, column: 19, scope: !4)
!18 = !DILocation(line: 94, column: 9, scope: !4)
!19 = !DILocation(line: 71, column: 23, scope: !4)
!20 = !DILocation(line: 74, column: 20, scope: !4)
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
!34 = !{i64 8192}
