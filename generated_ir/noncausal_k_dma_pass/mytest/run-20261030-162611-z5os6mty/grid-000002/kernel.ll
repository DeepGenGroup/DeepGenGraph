; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-162611-z5os6mty/grid-000002/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-162611-z5os6mty/grid-000002/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !30 {
  %kdma.tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !5
  %kdma.lane = and i32 %kdma.tid, 63, !dbg !5
  %kdma.wave = and i32 %kdma.tid, -64, !dbg !5
  %5 = lshr i32 %kdma.lane, 1, !dbg !5
  %kdma.owner = or i32 %kdma.wave, %5, !dbg !5
  %kdma.owner.hi = add i32 %kdma.owner, 32, !dbg !5
  %6 = and i32 %kdma.lane, 1, !dbg !5
  %kdma.intra = shl i32 %6, 3, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %10 = sext i32 %9 to i64, !dbg !5
  %11 = shl nsw i64 %10, 5, !dbg !5
  %12 = add nsw i64 %11, %8, !dbg !5
  %13 = lshr i64 %12, 5, !dbg !5
  %14 = and i64 %13, 576460752303423456, !dbg !5
  %15 = and i64 %8, 31, !dbg !5
  %16 = or disjoint i64 %14, %15, !dbg !5
  %17 = sub nsw i64 31, %16, !dbg !5
  %18 = lshr i64 %12, 5, !dbg !5
  %19 = icmp ugt i64 %12, 1023, !dbg !8
  %20 = add nsw i64 %16, -32, !dbg !8
  %21 = select i1 %19, i64 %20, i64 %17, !dbg !8
  %22 = sdiv i64 %21, 32, !dbg !8
  %23 = sext i1 %19 to i64, !dbg !8
  %24 = xor i64 %22, %23, !dbg !8
  %25 = add nsw i64 %18, %24, !dbg !8
  %26 = and i64 %25, 31, !dbg !8
  %27 = shl nsw i64 %17, 7, !dbg !8
  %28 = mul nsw i64 %24, -4096, !dbg !8
  %29 = add i64 %27, %28, !dbg !8
  %30 = shl nuw nsw i64 %26, 19, !dbg !8
  %31 = shl nsw i64 %29, 7, !dbg !8
  %32 = add i64 %30, %31, !dbg !8
  %33 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !32
  %34 = zext nneg i32 %33 to i64, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %33) #9, !dbg !8, !range !32, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local271 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local272 = and i64 %address.local271, 1536, !dbg !8
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local272, !dbg !8
  %address.local274 = ptrtoint ptr addrspace(3) %address.local273 to i64, !dbg !8
  %address.local275 = trunc i64 %address.local274 to i32, !dbg !8
  %35 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local275), !dbg !8
  %address.local276 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local277 = and i64 %address.local276, 16256, !dbg !8
  %address.local278 = lshr i64 %address.local, 4, !dbg !8
  %address.local279 = and i64 %address.local278, 8, !dbg !8
  %address.local280 = or disjoint i64 %address.local277, %address.local279, !dbg !8
  %36 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1266 = or disjoint i64 %32, %address.local280, !dbg !8
  %.idx1266.tr = trunc i64 %.idx1266 to i32, !dbg !8
  %dma.buffer.offset1263 = shl i32 %.idx1266.tr, 1, !dbg !8
  %37 = insertelement <2 x i64> poison, i64 %36, i64 0, !dbg !8
  %38 = bitcast <2 x i64> %37 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1264 = shufflevector <4 x i32> %38, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1265 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %35), !dbg !8
  %39 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1265, i32 %dma.buffer.offset1263, <4 x i32> %dma.buffer.rsrc1264) #10, !dbg !8
  %address.local282 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local283 = shl nuw nsw i64 %address.local282, 3, !dbg !8
  %address.local284 = and i64 %address.local283, 3584, !dbg !8
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local284, !dbg !8
  %address.local286 = ptrtoint ptr addrspace(3) %address.local285 to i64, !dbg !8
  %address.local287 = trunc i64 %address.local286 to i32, !dbg !8
  %40 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local287), !dbg !8
  %address.local288 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local289 = and i64 %address.local288, 16256, !dbg !8
  %address.local290 = lshr i64 %address.local282, 4, !dbg !8
  %address.local291 = and i64 %address.local290, 24, !dbg !8
  %address.local292 = or disjoint i64 %address.local289, %address.local291, !dbg !8
  %41 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1267 = or disjoint i64 %32, %address.local292, !dbg !8
  %.idx1267.tr = trunc i64 %.idx1267 to i32, !dbg !8
  %dma.buffer.offset1260 = shl i32 %.idx1267.tr, 1, !dbg !8
  %42 = insertelement <2 x i64> poison, i64 %41, i64 0, !dbg !8
  %43 = bitcast <2 x i64> %42 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1261 = shufflevector <4 x i32> %43, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1262 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %40), !dbg !8
  %44 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1262, i32 %dma.buffer.offset1260, <4 x i32> %dma.buffer.rsrc1261) #10, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local272, !dbg !8
  %address.local295 = ptrtoint ptr addrspace(3) %address.local294 to i64, !dbg !8
  %address.local296 = trunc i64 %address.local295 to i32, !dbg !8
  %45 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local296), !dbg !8
  %46 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %47 = or disjoint i64 %32, 32, !dbg !8
  %48 = or disjoint i64 %address.local280, %47, !dbg !8
  %.tr = trunc i64 %48 to i32, !dbg !8
  %dma.buffer.offset1257 = shl i32 %.tr, 1, !dbg !8
  %49 = insertelement <2 x i64> poison, i64 %46, i64 0, !dbg !8
  %50 = bitcast <2 x i64> %49 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1258 = shufflevector <4 x i32> %50, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1259 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %45), !dbg !8
  %51 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1259, i32 %dma.buffer.offset1257, <4 x i32> %dma.buffer.rsrc1258) #10, !dbg !8
  %address.local298 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local284, !dbg !8
  %address.local299 = ptrtoint ptr addrspace(3) %address.local298 to i64, !dbg !8
  %address.local300 = trunc i64 %address.local299 to i32, !dbg !8
  %52 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local300), !dbg !8
  %53 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %54 = or disjoint i64 %32, 32, !dbg !8
  %55 = or disjoint i64 %address.local292, %54, !dbg !8
  %.tr1268 = trunc i64 %55 to i32, !dbg !8
  %dma.buffer.offset1254 = shl i32 %.tr1268, 1, !dbg !8
  %56 = insertelement <2 x i64> poison, i64 %53, i64 0, !dbg !8
  %57 = bitcast <2 x i64> %56 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1255 = shufflevector <4 x i32> %57, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1256 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %52), !dbg !8
  %58 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1256, i32 %dma.buffer.offset1254, <4 x i32> %dma.buffer.rsrc1255) #10, !dbg !8
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
  %59 = load <4 x half>, ptr addrspace(3) %address.local314, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local314, i64 4096, !dbg !10
  %60 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %61 = fmul <4 x half> %59, splat (half 0xH3015), !dbg !10
  %lds.const.gep1064 = getelementptr i8, ptr addrspace(3) %address.local314, i64 1024, !dbg !10
  %62 = load <4 x half>, ptr addrspace(3) %lds.const.gep1064, align 8, !dbg !10
  %63 = fmul <4 x half> %60, splat (half 0xH3015), !dbg !10
  %lds.const.gep1065 = getelementptr i8, ptr addrspace(3) %address.local314, i64 5120, !dbg !10
  %64 = load <4 x half>, ptr addrspace(3) %lds.const.gep1065, align 8, !dbg !10
  %65 = fmul <4 x half> %62, splat (half 0xH3015), !dbg !10
  %66 = fmul <4 x half> %64, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %67 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local275), !dbg !8
  %68 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %69 = or disjoint i64 %32, 64, !dbg !8
  %70 = or disjoint i64 %address.local280, %69, !dbg !8
  %.tr1269 = trunc i64 %70 to i32, !dbg !8
  %dma.buffer.offset1251 = shl i32 %.tr1269, 1, !dbg !8
  %71 = insertelement <2 x i64> poison, i64 %68, i64 0, !dbg !8
  %72 = bitcast <2 x i64> %71 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1252 = shufflevector <4 x i32> %72, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1253 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %67), !dbg !8
  %73 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1253, i32 %dma.buffer.offset1251, <4 x i32> %dma.buffer.rsrc1252) #10, !dbg !8
  %74 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local287), !dbg !8
  %75 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %76 = or disjoint i64 %32, 64, !dbg !8
  %77 = or disjoint i64 %address.local292, %76, !dbg !8
  %.tr1270 = trunc i64 %77 to i32, !dbg !8
  %dma.buffer.offset1248 = shl i32 %.tr1270, 1, !dbg !8
  %78 = insertelement <2 x i64> poison, i64 %75, i64 0, !dbg !8
  %79 = bitcast <2 x i64> %78 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1249 = shufflevector <4 x i32> %79, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1250 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %74), !dbg !8
  %80 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1250, i32 %dma.buffer.offset1248, <4 x i32> %dma.buffer.rsrc1249) #10, !dbg !8
  %lds.const.gep1066 = getelementptr i8, ptr addrspace(3) %address.local314, i64 8192, !dbg !10
  %81 = load <4 x half>, ptr addrspace(3) %lds.const.gep1066, align 8, !dbg !10
  %lds.const.gep1067 = getelementptr i8, ptr addrspace(3) %address.local314, i64 12288, !dbg !10
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1067, align 8, !dbg !10
  %lds.const.gep1068 = getelementptr i8, ptr addrspace(3) %address.local314, i64 9216, !dbg !10
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1068, align 8, !dbg !10
  %84 = fmul <4 x half> %81, splat (half 0xH3015), !dbg !10
  %lds.const.gep1069 = getelementptr i8, ptr addrspace(3) %address.local314, i64 13312, !dbg !10
  %85 = load <4 x half>, ptr addrspace(3) %lds.const.gep1069, align 8, !dbg !10
  %86 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !10
  %87 = fmul <4 x half> %83, splat (half 0xH3015), !dbg !10
  %88 = fmul <4 x half> %85, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %89 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local296), !dbg !8
  %90 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %91 = or disjoint i64 %32, 96, !dbg !8
  %92 = or disjoint i64 %address.local280, %91, !dbg !8
  %.tr1271 = trunc i64 %92 to i32, !dbg !8
  %dma.buffer.offset1245 = shl i32 %.tr1271, 1, !dbg !8
  %93 = insertelement <2 x i64> poison, i64 %90, i64 0, !dbg !8
  %94 = bitcast <2 x i64> %93 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1246 = shufflevector <4 x i32> %94, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1247 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %89), !dbg !8
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1247, i32 %dma.buffer.offset1245, <4 x i32> %dma.buffer.rsrc1246) #10, !dbg !8
  %96 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local300), !dbg !8
  %97 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %98 = or disjoint i64 %32, 96, !dbg !8
  %99 = or disjoint i64 %address.local292, %98, !dbg !8
  %.tr1272 = trunc i64 %99 to i32, !dbg !8
  %dma.buffer.offset1242 = shl i32 %.tr1272, 1, !dbg !8
  %100 = insertelement <2 x i64> poison, i64 %97, i64 0, !dbg !8
  %101 = bitcast <2 x i64> %100 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1243 = shufflevector <4 x i32> %101, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1244 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %96), !dbg !8
  %102 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1244, i32 %dma.buffer.offset1242, <4 x i32> %dma.buffer.rsrc1243) #10, !dbg !8
  %103 = load <4 x half>, ptr addrspace(3) %address.local314, align 8, !dbg !10
  %lds.const.gep1070 = getelementptr i8, ptr addrspace(3) %address.local314, i64 4096, !dbg !10
  %104 = load <4 x half>, ptr addrspace(3) %lds.const.gep1070, align 8, !dbg !10
  %lds.const.gep1071 = getelementptr i8, ptr addrspace(3) %address.local314, i64 1024, !dbg !10
  %105 = load <4 x half>, ptr addrspace(3) %lds.const.gep1071, align 8, !dbg !10
  %106 = fmul <4 x half> %103, splat (half 0xH3015), !dbg !10
  %lds.const.gep1072 = getelementptr i8, ptr addrspace(3) %address.local314, i64 5120, !dbg !10
  %107 = load <4 x half>, ptr addrspace(3) %lds.const.gep1072, align 8, !dbg !10
  %108 = fmul <4 x half> %104, splat (half 0xH3015), !dbg !10
  %109 = fmul <4 x half> %105, splat (half 0xH3015), !dbg !10
  %110 = fmul <4 x half> %107, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1073 = getelementptr i8, ptr addrspace(3) %address.local314, i64 8192, !dbg !10
  %111 = load <4 x half>, ptr addrspace(3) %lds.const.gep1073, align 8, !dbg !10
  %lds.const.gep1074 = getelementptr i8, ptr addrspace(3) %address.local314, i64 12288, !dbg !10
  %112 = load <4 x half>, ptr addrspace(3) %lds.const.gep1074, align 8, !dbg !10
  %lds.const.gep1075 = getelementptr i8, ptr addrspace(3) %address.local314, i64 9216, !dbg !10
  %113 = load <4 x half>, ptr addrspace(3) %lds.const.gep1075, align 8, !dbg !10
  %114 = fmul <4 x half> %111, splat (half 0xH3015), !dbg !10
  %lds.const.gep1076 = getelementptr i8, ptr addrspace(3) %address.local314, i64 13312, !dbg !10
  %115 = load <4 x half>, ptr addrspace(3) %lds.const.gep1076, align 8, !dbg !10
  %116 = fmul <4 x half> %112, splat (half 0xH3015), !dbg !10
  %117 = fmul <4 x half> %113, splat (half 0xH3015), !dbg !10
  %118 = fmul <4 x half> %115, splat (half 0xH3015), !dbg !10
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %119 = shl nuw nsw i64 %34, 2, !dbg !14
  %120 = shl nuw nsw i64 %34, 2, !dbg !14
  %121 = and i64 %120, 1008, !dbg !14
  %122 = sub nsw i64 %119, %121, !dbg !14
  %123 = and i32 %33, 1, !dbg !14
  %index.compact211 = zext nneg i32 %123 to i64, !dbg !14
  %124 = or disjoint i64 %122, %index.compact211, !dbg !14
  %125 = add nuw nsw i64 %124, 1, !dbg !14
  %126 = sdiv i64 %125, 2, !dbg !14
  br label %127, !dbg !7

127:                                              ; preds = %136, %4
  %128 = phi i64 [ %1274, %136 ], [ 0, %4 ], !dbg !7
  %129 = phi [2 x <8 x float>] [ %1259, %136 ], [ zeroinitializer, %4 ], !dbg !7
  %130 = phi [2 x <1 x float>] [ %953, %136 ], [ zeroinitializer, %4 ], !dbg !7
  %131 = phi [2 x <1 x float>] [ %1261, %136 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %132 = phi [2 x <8 x float>] [ %1265, %136 ], [ zeroinitializer, %4 ], !dbg !7
  %133 = phi [2 x <8 x float>] [ %1269, %136 ], [ zeroinitializer, %4 ], !dbg !7
  %134 = phi [2 x <8 x float>] [ %1273, %136 ], [ zeroinitializer, %4 ], !dbg !7
  %135 = icmp samesign ult i64 %128, 32, !dbg !7
  br i1 %135, label %136, label %1275, !dbg !7

136:                                              ; preds = %127
  %137 = shl i64 %18, 19, !dbg !12
  %138 = and i64 %137, 16252928, !dbg !12
  %139 = shl nuw nsw i64 %128, 14, !dbg !12
  %140 = or disjoint i64 %138, %139, !dbg !12
  %address.thread365 = call i32 asm sideeffect "", "=v,0"(i32 %33) #9, !dbg !12, !range !32, !frisk.register_lifetimes !1
  %address.local366 = zext nneg i32 %address.thread365 to i64, !dbg !8
  %address.local367 = shl nuw nsw i64 %address.local366, 3, !dbg !8
  %address.local368 = and i64 %address.local367, 1536, !dbg !8
  %address.local369 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local368, !dbg !8
  %address.local370 = ptrtoint ptr addrspace(3) %address.local369 to i64, !dbg !8
  %address.local371 = trunc i64 %address.local370 to i32, !dbg !8
  %141 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !12
  %142 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %143 = insertelement <2 x i64> poison, i64 %142, i64 0, !dbg !12
  %144 = bitcast <2 x i64> %143 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1240 = shufflevector <4 x i32> %144, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %141), !dbg !12
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1373 = lshr i64 %kdma.source, 4, !dbg !12
  %kdma.source1374 = and i64 %kdma.source1373, 8, !dbg !12
  %kdma.source1375 = and i64 %kdma.source, 123, !dbg !12
  %kdma.source1376 = lshr i64 %kdma.source, 2, !dbg !12
  %kdma.source1377 = and i64 %kdma.source1376, 4, !dbg !12
  %kdma.source1378 = add nuw nsw i64 %kdma.source1377, %kdma.source, !dbg !12
  %kdma.source1379 = and i64 %kdma.source1378, 4, !dbg !12
  %kdma.source1380 = or disjoint i64 %kdma.source1375, %kdma.source1379, !dbg !12
  %kdma.source1381 = shl nuw nsw i64 %kdma.source1380, 7, !dbg !12
  %kdma.source1382 = or disjoint i64 %kdma.source1374, %kdma.source1381, !dbg !12
  %kdma.source1383 = or disjoint i64 %140, %kdma.source1382, !dbg !12
  %kdma.source1384 = trunc nuw i64 %kdma.source1383 to i32, !dbg !12
  %kdma.source1385 = shl nuw nsw i32 %kdma.source1384, 1, !dbg !12
  %kdma.offset = add i32 %kdma.source1385, %kdma.intra, !dbg !12
  %kdma.lds = add i32 %dma.buffer.lds1241, 0, !dbg !12
  %145 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1240) #10, !dbg !12
  %kdma.source1386 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1387 = lshr i64 %kdma.source1386, 4, !dbg !12
  %kdma.source1388 = and i64 %kdma.source1387, 8, !dbg !12
  %kdma.source1389 = and i64 %kdma.source1386, 123, !dbg !12
  %kdma.source1390 = lshr i64 %kdma.source1386, 2, !dbg !12
  %kdma.source1391 = and i64 %kdma.source1390, 4, !dbg !12
  %kdma.source1392 = add nuw nsw i64 %kdma.source1391, %kdma.source1386, !dbg !12
  %kdma.source1393 = and i64 %kdma.source1392, 4, !dbg !12
  %kdma.source1394 = or disjoint i64 %kdma.source1389, %kdma.source1393, !dbg !12
  %kdma.source1395 = shl nuw nsw i64 %kdma.source1394, 7, !dbg !12
  %kdma.source1396 = or disjoint i64 %kdma.source1388, %kdma.source1395, !dbg !12
  %kdma.source1397 = or disjoint i64 %140, %kdma.source1396, !dbg !12
  %kdma.source1398 = trunc nuw i64 %kdma.source1397 to i32, !dbg !12
  %kdma.source1399 = shl nuw nsw i32 %kdma.source1398, 1, !dbg !12
  %kdma.offset1400 = add i32 %kdma.source1399, %kdma.intra, !dbg !12
  %kdma.lds1401 = add i32 %dma.buffer.lds1241, 512, !dbg !12
  %146 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1401, i32 %kdma.offset1400, <4 x i32> %dma.buffer.rsrc1240) #10, !dbg !12
  %address.local383 = or disjoint i64 %address.local366, 256, !dbg !8
  %address.local384 = shl nuw nsw i64 %address.local383, 3, !dbg !8
  %address.local385 = and i64 %address.local384, 3584, !dbg !8
  %address.local386 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local385, !dbg !8
  %address.local387 = ptrtoint ptr addrspace(3) %address.local386 to i64, !dbg !8
  %address.local388 = trunc i64 %address.local387 to i32, !dbg !8
  %147 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !12
  %148 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %149 = insertelement <2 x i64> poison, i64 %148, i64 0, !dbg !12
  %150 = bitcast <2 x i64> %149 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1237 = shufflevector <4 x i32> %150, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1238 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %147), !dbg !12
  %kdma.source1402 = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1403 = or disjoint i64 %kdma.source1402, 256, !dbg !12
  %kdma.source1404 = lshr i64 %kdma.source1403, 4, !dbg !12
  %kdma.source1405 = and i64 %kdma.source1404, 24, !dbg !12
  %kdma.source1406 = and i64 %kdma.source1402, 123, !dbg !12
  %kdma.source1407 = lshr i64 %kdma.source1402, 2, !dbg !12
  %kdma.source1408 = and i64 %kdma.source1407, 4, !dbg !12
  %kdma.source1409 = add nuw nsw i64 %kdma.source1408, %kdma.source1402, !dbg !12
  %kdma.source1410 = and i64 %kdma.source1409, 4, !dbg !12
  %kdma.source1411 = or disjoint i64 %kdma.source1406, %kdma.source1410, !dbg !12
  %kdma.source1412 = shl nuw nsw i64 %kdma.source1411, 7, !dbg !12
  %kdma.source1413 = or disjoint i64 %kdma.source1405, %kdma.source1412, !dbg !12
  %kdma.source1414 = or disjoint i64 %140, %kdma.source1413, !dbg !12
  %kdma.source1415 = trunc nuw i64 %kdma.source1414 to i32, !dbg !12
  %kdma.source1416 = shl nuw nsw i32 %kdma.source1415, 1, !dbg !12
  %kdma.offset1417 = add i32 %kdma.source1416, %kdma.intra, !dbg !12
  %kdma.lds1418 = add i32 %dma.buffer.lds1238, 0, !dbg !12
  %151 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1418, i32 %kdma.offset1417, <4 x i32> %dma.buffer.rsrc1237) #10, !dbg !12
  %kdma.source1419 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1420 = or disjoint i64 %kdma.source1419, 256, !dbg !12
  %kdma.source1421 = lshr i64 %kdma.source1420, 4, !dbg !12
  %kdma.source1422 = and i64 %kdma.source1421, 24, !dbg !12
  %kdma.source1423 = and i64 %kdma.source1419, 123, !dbg !12
  %kdma.source1424 = lshr i64 %kdma.source1419, 2, !dbg !12
  %kdma.source1425 = and i64 %kdma.source1424, 4, !dbg !12
  %kdma.source1426 = add nuw nsw i64 %kdma.source1425, %kdma.source1419, !dbg !12
  %kdma.source1427 = and i64 %kdma.source1426, 4, !dbg !12
  %kdma.source1428 = or disjoint i64 %kdma.source1423, %kdma.source1427, !dbg !12
  %kdma.source1429 = shl nuw nsw i64 %kdma.source1428, 7, !dbg !12
  %kdma.source1430 = or disjoint i64 %kdma.source1422, %kdma.source1429, !dbg !12
  %kdma.source1431 = or disjoint i64 %140, %kdma.source1430, !dbg !12
  %kdma.source1432 = trunc nuw i64 %kdma.source1431 to i32, !dbg !12
  %kdma.source1433 = shl nuw nsw i32 %kdma.source1432, 1, !dbg !12
  %kdma.offset1434 = add i32 %kdma.source1433, %kdma.intra, !dbg !12
  %kdma.lds1435 = add i32 %dma.buffer.lds1238, 512, !dbg !12
  %152 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1435, i32 %kdma.offset1434, <4 x i32> %dma.buffer.rsrc1237) #10, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %153 = or disjoint i64 %140, 32, !dbg !12
  %address.local400 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local368, !dbg !8
  %address.local401 = ptrtoint ptr addrspace(3) %address.local400 to i64, !dbg !8
  %address.local402 = trunc i64 %address.local401 to i32, !dbg !8
  %154 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local402), !dbg !12
  %155 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %156 = insertelement <2 x i64> poison, i64 %155, i64 0, !dbg !12
  %157 = bitcast <2 x i64> %156 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1234 = shufflevector <4 x i32> %157, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1235 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %154), !dbg !12
  %kdma.source1436 = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1437 = lshr i64 %kdma.source1436, 4, !dbg !12
  %kdma.source1438 = and i64 %kdma.source1437, 8, !dbg !12
  %kdma.source1439 = and i64 %kdma.source1436, 123, !dbg !12
  %kdma.source1440 = lshr i64 %kdma.source1436, 2, !dbg !12
  %kdma.source1441 = and i64 %kdma.source1440, 4, !dbg !12
  %kdma.source1442 = add nuw nsw i64 %kdma.source1441, %kdma.source1436, !dbg !12
  %kdma.source1443 = and i64 %kdma.source1442, 4, !dbg !12
  %kdma.source1444 = or disjoint i64 %kdma.source1439, %kdma.source1443, !dbg !12
  %kdma.source1445 = shl nuw nsw i64 %kdma.source1444, 7, !dbg !12
  %kdma.source1446 = or disjoint i64 %kdma.source1438, %kdma.source1445, !dbg !12
  %kdma.source1447 = or disjoint i64 %153, %kdma.source1446, !dbg !12
  %kdma.source1448 = trunc nuw i64 %kdma.source1447 to i32, !dbg !12
  %kdma.source1449 = shl nuw nsw i32 %kdma.source1448, 1, !dbg !12
  %kdma.offset1450 = add i32 %kdma.source1449, %kdma.intra, !dbg !12
  %kdma.lds1451 = add i32 %dma.buffer.lds1235, 0, !dbg !12
  %158 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1451, i32 %kdma.offset1450, <4 x i32> %dma.buffer.rsrc1234) #10, !dbg !12
  %kdma.source1452 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1453 = lshr i64 %kdma.source1452, 4, !dbg !12
  %kdma.source1454 = and i64 %kdma.source1453, 8, !dbg !12
  %kdma.source1455 = and i64 %kdma.source1452, 123, !dbg !12
  %kdma.source1456 = lshr i64 %kdma.source1452, 2, !dbg !12
  %kdma.source1457 = and i64 %kdma.source1456, 4, !dbg !12
  %kdma.source1458 = add nuw nsw i64 %kdma.source1457, %kdma.source1452, !dbg !12
  %kdma.source1459 = and i64 %kdma.source1458, 4, !dbg !12
  %kdma.source1460 = or disjoint i64 %kdma.source1455, %kdma.source1459, !dbg !12
  %kdma.source1461 = shl nuw nsw i64 %kdma.source1460, 7, !dbg !12
  %kdma.source1462 = or disjoint i64 %kdma.source1454, %kdma.source1461, !dbg !12
  %kdma.source1463 = or disjoint i64 %153, %kdma.source1462, !dbg !12
  %kdma.source1464 = trunc nuw i64 %kdma.source1463 to i32, !dbg !12
  %kdma.source1465 = shl nuw nsw i32 %kdma.source1464, 1, !dbg !12
  %kdma.offset1466 = add i32 %kdma.source1465, %kdma.intra, !dbg !12
  %kdma.lds1467 = add i32 %dma.buffer.lds1235, 512, !dbg !12
  %159 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1467, i32 %kdma.offset1466, <4 x i32> %dma.buffer.rsrc1234) #10, !dbg !12
  %address.local404 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local385, !dbg !8
  %address.local405 = ptrtoint ptr addrspace(3) %address.local404 to i64, !dbg !8
  %address.local406 = trunc i64 %address.local405 to i32, !dbg !8
  %160 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local406), !dbg !12
  %161 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %162 = insertelement <2 x i64> poison, i64 %161, i64 0, !dbg !12
  %163 = bitcast <2 x i64> %162 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1231 = shufflevector <4 x i32> %163, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1232 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %160), !dbg !12
  %kdma.source1468 = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1469 = or disjoint i64 %kdma.source1468, 256, !dbg !12
  %kdma.source1470 = lshr i64 %kdma.source1469, 4, !dbg !12
  %kdma.source1471 = and i64 %kdma.source1470, 24, !dbg !12
  %kdma.source1472 = and i64 %kdma.source1468, 123, !dbg !12
  %kdma.source1473 = lshr i64 %kdma.source1468, 2, !dbg !12
  %kdma.source1474 = and i64 %kdma.source1473, 4, !dbg !12
  %kdma.source1475 = add nuw nsw i64 %kdma.source1474, %kdma.source1468, !dbg !12
  %kdma.source1476 = and i64 %kdma.source1475, 4, !dbg !12
  %kdma.source1477 = or disjoint i64 %kdma.source1472, %kdma.source1476, !dbg !12
  %kdma.source1478 = shl nuw nsw i64 %kdma.source1477, 7, !dbg !12
  %kdma.source1479 = or disjoint i64 %kdma.source1471, %kdma.source1478, !dbg !12
  %kdma.source1480 = or disjoint i64 %153, %kdma.source1479, !dbg !12
  %kdma.source1481 = trunc nuw i64 %kdma.source1480 to i32, !dbg !12
  %kdma.source1482 = shl nuw nsw i32 %kdma.source1481, 1, !dbg !12
  %kdma.offset1483 = add i32 %kdma.source1482, %kdma.intra, !dbg !12
  %kdma.lds1484 = add i32 %dma.buffer.lds1232, 0, !dbg !12
  %164 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1484, i32 %kdma.offset1483, <4 x i32> %dma.buffer.rsrc1231) #10, !dbg !12
  %kdma.source1485 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1486 = or disjoint i64 %kdma.source1485, 256, !dbg !12
  %kdma.source1487 = lshr i64 %kdma.source1486, 4, !dbg !12
  %kdma.source1488 = and i64 %kdma.source1487, 24, !dbg !12
  %kdma.source1489 = and i64 %kdma.source1485, 123, !dbg !12
  %kdma.source1490 = lshr i64 %kdma.source1485, 2, !dbg !12
  %kdma.source1491 = and i64 %kdma.source1490, 4, !dbg !12
  %kdma.source1492 = add nuw nsw i64 %kdma.source1491, %kdma.source1485, !dbg !12
  %kdma.source1493 = and i64 %kdma.source1492, 4, !dbg !12
  %kdma.source1494 = or disjoint i64 %kdma.source1489, %kdma.source1493, !dbg !12
  %kdma.source1495 = shl nuw nsw i64 %kdma.source1494, 7, !dbg !12
  %kdma.source1496 = or disjoint i64 %kdma.source1488, %kdma.source1495, !dbg !12
  %kdma.source1497 = or disjoint i64 %153, %kdma.source1496, !dbg !12
  %kdma.source1498 = trunc nuw i64 %kdma.source1497 to i32, !dbg !12
  %kdma.source1499 = shl nuw nsw i32 %kdma.source1498, 1, !dbg !12
  %kdma.offset1500 = add i32 %kdma.source1499, %kdma.intra, !dbg !12
  %kdma.lds1501 = add i32 %dma.buffer.lds1232, 512, !dbg !12
  %165 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1501, i32 %kdma.offset1500, <4 x i32> %dma.buffer.rsrc1231) #10, !dbg !12
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
  %166 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !14
  %.idx1277 = mul nsw i64 %address.local409, -1280, !dbg !14
  %167 = getelementptr i8, ptr addrspace(3) %166, i64 %.idx1277, !dbg !14
  %168 = getelementptr [2 x i8], ptr addrspace(3) %167, i64 %address.local413, !dbg !14
  %169 = getelementptr [2 x i8], ptr addrspace(3) %168, i64 %address.local417, !dbg !14
  %170 = getelementptr [2 x i8], ptr addrspace(3) %169, i64 %address.local420, !dbg !14
  %address.local426 = getelementptr [2 x i8], ptr addrspace(3) %170, i64 %address.local424, !dbg !14
  %171 = load <4 x half>, ptr addrspace(3) %address.local426, align 8, !dbg !14
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %171, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1077 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4096, !dbg !14
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1077, align 8, !dbg !14
  %address.local437 = shl nuw nsw i32 %address.thread365, 1, !dbg !14
  %address.local438 = and i32 %address.local437, 24, !dbg !14
  %address.local439 = shl nuw nsw i32 %address.thread365, 8, !dbg !14
  %address.local440 = and i32 %address.local439, 768, !dbg !14
  %address.local441 = or disjoint i32 %address.local438, %address.local440, !dbg !14
  %address.local442 = shl nuw nsw i32 %address.thread365, 5, !dbg !14
  %address.local443 = and i32 %address.local442, 96, !dbg !14
  %address.local444 = or disjoint i32 %address.local441, %address.local443, !dbg !14
  %address.local445 = zext nneg i32 %address.local444 to i64, !dbg !14
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %126) #9, !dbg !14, !frisk.register_lifetimes !1
  %address.local446 = lshr i32 %address.thread365, 1, !dbg !14
  %address.local447 = and i32 %address.local446, 1, !dbg !14
  %address.local448 = zext nneg i32 %address.local447 to i64, !dbg !14
  %address.local449 = add nsw i64 %address.base, %address.local448, !dbg !14
  %address.local450 = shl i64 %address.local449, 6, !dbg !14
  %address.local451 = sub i64 %address.local445, %address.local450, !dbg !14
  %174 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local451, !dbg !14
  %175 = getelementptr [2 x i8], ptr addrspace(3) %174, i64 %address.local420, !dbg !14
  %176 = getelementptr [2 x i8], ptr addrspace(3) %175, i64 %address.local424, !dbg !14
  %address.local455 = getelementptr i8, ptr addrspace(3) %176, i64 64, !dbg !14
  %177 = load <4 x half>, ptr addrspace(3) %address.local455, align 8, !dbg !14
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %173, <4 x float> %172) #9, !dbg !14
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %177, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1078 = getelementptr i8, ptr addrspace(3) %176, i64 4160, !dbg !14
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1078, align 8, !dbg !14
  %lds.const.gep1079 = getelementptr i8, ptr addrspace(3) %address.local426, i64 128, !dbg !14
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1079, align 8, !dbg !14
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %180, <4 x float> %179) #9, !dbg !14
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %181, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1080 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4224, !dbg !14
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1080, align 8, !dbg !14
  %lds.const.gep1081 = getelementptr i8, ptr addrspace(3) %176, i64 192, !dbg !14
  %185 = load <4 x half>, ptr addrspace(3) %lds.const.gep1081, align 8, !dbg !14
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %184, <4 x float> %183) #9, !dbg !14
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %185, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1082 = getelementptr i8, ptr addrspace(3) %176, i64 4288, !dbg !14
  %188 = load <4 x half>, ptr addrspace(3) %lds.const.gep1082, align 8, !dbg !14
  %lds.const.gep1083 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1024, !dbg !14
  %189 = load <4 x half>, ptr addrspace(3) %lds.const.gep1083, align 8, !dbg !14
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %188, <4 x float> %187) #9, !dbg !14
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %189, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1084 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5120, !dbg !14
  %192 = load <4 x half>, ptr addrspace(3) %lds.const.gep1084, align 8, !dbg !14
  %lds.const.gep1085 = getelementptr i8, ptr addrspace(3) %176, i64 1088, !dbg !14
  %193 = load <4 x half>, ptr addrspace(3) %lds.const.gep1085, align 8, !dbg !14
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %192, <4 x float> %191) #9, !dbg !14
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %193, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1086 = getelementptr i8, ptr addrspace(3) %176, i64 5184, !dbg !14
  %196 = load <4 x half>, ptr addrspace(3) %lds.const.gep1086, align 8, !dbg !14
  %lds.const.gep1087 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1152, !dbg !14
  %197 = load <4 x half>, ptr addrspace(3) %lds.const.gep1087, align 8, !dbg !14
  %198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %196, <4 x float> %195) #9, !dbg !14
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %197, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1088 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5248, !dbg !14
  %200 = load <4 x half>, ptr addrspace(3) %lds.const.gep1088, align 8, !dbg !14
  %lds.const.gep1089 = getelementptr i8, ptr addrspace(3) %176, i64 1216, !dbg !14
  %201 = load <4 x half>, ptr addrspace(3) %lds.const.gep1089, align 8, !dbg !14
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %200, <4 x float> %199) #9, !dbg !14
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %201, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1090 = getelementptr i8, ptr addrspace(3) %176, i64 5312, !dbg !14
  %204 = load <4 x half>, ptr addrspace(3) %lds.const.gep1090, align 8, !dbg !14
  %205 = load <4 x half>, ptr addrspace(3) %address.local426, align 8, !dbg !14
  %206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %204, <4 x float> %203) #9, !dbg !14
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %205, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1091 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4096, !dbg !14
  %208 = load <4 x half>, ptr addrspace(3) %lds.const.gep1091, align 8, !dbg !14
  %209 = load <4 x half>, ptr addrspace(3) %address.local455, align 8, !dbg !14
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %208, <4 x float> %207) #9, !dbg !14
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %209, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1092 = getelementptr i8, ptr addrspace(3) %176, i64 4160, !dbg !14
  %212 = load <4 x half>, ptr addrspace(3) %lds.const.gep1092, align 8, !dbg !14
  %lds.const.gep1093 = getelementptr i8, ptr addrspace(3) %address.local426, i64 128, !dbg !14
  %213 = load <4 x half>, ptr addrspace(3) %lds.const.gep1093, align 8, !dbg !14
  %214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %212, <4 x float> %211) #9, !dbg !14
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %213, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1094 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4224, !dbg !14
  %216 = load <4 x half>, ptr addrspace(3) %lds.const.gep1094, align 8, !dbg !14
  %lds.const.gep1095 = getelementptr i8, ptr addrspace(3) %176, i64 192, !dbg !14
  %217 = load <4 x half>, ptr addrspace(3) %lds.const.gep1095, align 8, !dbg !14
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %216, <4 x float> %215) #9, !dbg !14
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %217, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1096 = getelementptr i8, ptr addrspace(3) %176, i64 4288, !dbg !14
  %220 = load <4 x half>, ptr addrspace(3) %lds.const.gep1096, align 8, !dbg !14
  %lds.const.gep1097 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1024, !dbg !14
  %221 = load <4 x half>, ptr addrspace(3) %lds.const.gep1097, align 8, !dbg !14
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %220, <4 x float> %219) #9, !dbg !14
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %221, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1098 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5120, !dbg !14
  %224 = load <4 x half>, ptr addrspace(3) %lds.const.gep1098, align 8, !dbg !14
  %lds.const.gep1099 = getelementptr i8, ptr addrspace(3) %176, i64 1088, !dbg !14
  %225 = load <4 x half>, ptr addrspace(3) %lds.const.gep1099, align 8, !dbg !14
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %224, <4 x float> %223) #9, !dbg !14
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %225, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1100 = getelementptr i8, ptr addrspace(3) %176, i64 5184, !dbg !14
  %228 = load <4 x half>, ptr addrspace(3) %lds.const.gep1100, align 8, !dbg !14
  %lds.const.gep1101 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1152, !dbg !14
  %229 = load <4 x half>, ptr addrspace(3) %lds.const.gep1101, align 8, !dbg !14
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %228, <4 x float> %227) #9, !dbg !14
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %229, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1102 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5248, !dbg !14
  %232 = load <4 x half>, ptr addrspace(3) %lds.const.gep1102, align 8, !dbg !14
  %lds.const.gep1103 = getelementptr i8, ptr addrspace(3) %176, i64 1216, !dbg !14
  %233 = load <4 x half>, ptr addrspace(3) %lds.const.gep1103, align 8, !dbg !14
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %232, <4 x float> %231) #9, !dbg !14
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %233, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1104 = getelementptr i8, ptr addrspace(3) %176, i64 5312, !dbg !14
  %236 = load <4 x half>, ptr addrspace(3) %lds.const.gep1104, align 8, !dbg !14
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %236, <4 x float> %235) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %238 = or disjoint i64 %140, 64, !dbg !12
  %239 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !12
  %240 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %241 = insertelement <2 x i64> poison, i64 %240, i64 0, !dbg !12
  %242 = bitcast <2 x i64> %241 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1228 = shufflevector <4 x i32> %242, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1229 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %239), !dbg !12
  %kdma.source1502 = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1503 = lshr i64 %kdma.source1502, 4, !dbg !12
  %kdma.source1504 = and i64 %kdma.source1503, 8, !dbg !12
  %kdma.source1505 = and i64 %kdma.source1502, 123, !dbg !12
  %kdma.source1506 = lshr i64 %kdma.source1502, 2, !dbg !12
  %kdma.source1507 = and i64 %kdma.source1506, 4, !dbg !12
  %kdma.source1508 = add nuw nsw i64 %kdma.source1507, %kdma.source1502, !dbg !12
  %kdma.source1509 = and i64 %kdma.source1508, 4, !dbg !12
  %kdma.source1510 = or disjoint i64 %kdma.source1505, %kdma.source1509, !dbg !12
  %kdma.source1511 = shl nuw nsw i64 %kdma.source1510, 7, !dbg !12
  %kdma.source1512 = or disjoint i64 %kdma.source1504, %kdma.source1511, !dbg !12
  %kdma.source1513 = or disjoint i64 %238, %kdma.source1512, !dbg !12
  %kdma.source1514 = trunc nuw i64 %kdma.source1513 to i32, !dbg !12
  %kdma.source1515 = shl nuw nsw i32 %kdma.source1514, 1, !dbg !12
  %kdma.offset1516 = add i32 %kdma.source1515, %kdma.intra, !dbg !12
  %kdma.lds1517 = add i32 %dma.buffer.lds1229, 0, !dbg !12
  %243 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1517, i32 %kdma.offset1516, <4 x i32> %dma.buffer.rsrc1228) #10, !dbg !12
  %kdma.source1518 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1519 = lshr i64 %kdma.source1518, 4, !dbg !12
  %kdma.source1520 = and i64 %kdma.source1519, 8, !dbg !12
  %kdma.source1521 = and i64 %kdma.source1518, 123, !dbg !12
  %kdma.source1522 = lshr i64 %kdma.source1518, 2, !dbg !12
  %kdma.source1523 = and i64 %kdma.source1522, 4, !dbg !12
  %kdma.source1524 = add nuw nsw i64 %kdma.source1523, %kdma.source1518, !dbg !12
  %kdma.source1525 = and i64 %kdma.source1524, 4, !dbg !12
  %kdma.source1526 = or disjoint i64 %kdma.source1521, %kdma.source1525, !dbg !12
  %kdma.source1527 = shl nuw nsw i64 %kdma.source1526, 7, !dbg !12
  %kdma.source1528 = or disjoint i64 %kdma.source1520, %kdma.source1527, !dbg !12
  %kdma.source1529 = or disjoint i64 %238, %kdma.source1528, !dbg !12
  %kdma.source1530 = trunc nuw i64 %kdma.source1529 to i32, !dbg !12
  %kdma.source1531 = shl nuw nsw i32 %kdma.source1530, 1, !dbg !12
  %kdma.offset1532 = add i32 %kdma.source1531, %kdma.intra, !dbg !12
  %kdma.lds1533 = add i32 %dma.buffer.lds1229, 512, !dbg !12
  %244 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1533, i32 %kdma.offset1532, <4 x i32> %dma.buffer.rsrc1228) #10, !dbg !12
  %245 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !12
  %246 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %247 = insertelement <2 x i64> poison, i64 %246, i64 0, !dbg !12
  %248 = bitcast <2 x i64> %247 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1225 = shufflevector <4 x i32> %248, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %245), !dbg !12
  %kdma.source1534 = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1535 = or disjoint i64 %kdma.source1534, 256, !dbg !12
  %kdma.source1536 = lshr i64 %kdma.source1535, 4, !dbg !12
  %kdma.source1537 = and i64 %kdma.source1536, 24, !dbg !12
  %kdma.source1538 = and i64 %kdma.source1534, 123, !dbg !12
  %kdma.source1539 = lshr i64 %kdma.source1534, 2, !dbg !12
  %kdma.source1540 = and i64 %kdma.source1539, 4, !dbg !12
  %kdma.source1541 = add nuw nsw i64 %kdma.source1540, %kdma.source1534, !dbg !12
  %kdma.source1542 = and i64 %kdma.source1541, 4, !dbg !12
  %kdma.source1543 = or disjoint i64 %kdma.source1538, %kdma.source1542, !dbg !12
  %kdma.source1544 = shl nuw nsw i64 %kdma.source1543, 7, !dbg !12
  %kdma.source1545 = or disjoint i64 %kdma.source1537, %kdma.source1544, !dbg !12
  %kdma.source1546 = or disjoint i64 %238, %kdma.source1545, !dbg !12
  %kdma.source1547 = trunc nuw i64 %kdma.source1546 to i32, !dbg !12
  %kdma.source1548 = shl nuw nsw i32 %kdma.source1547, 1, !dbg !12
  %kdma.offset1549 = add i32 %kdma.source1548, %kdma.intra, !dbg !12
  %kdma.lds1550 = add i32 %dma.buffer.lds1226, 0, !dbg !12
  %249 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1550, i32 %kdma.offset1549, <4 x i32> %dma.buffer.rsrc1225) #10, !dbg !12
  %kdma.source1551 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1552 = or disjoint i64 %kdma.source1551, 256, !dbg !12
  %kdma.source1553 = lshr i64 %kdma.source1552, 4, !dbg !12
  %kdma.source1554 = and i64 %kdma.source1553, 24, !dbg !12
  %kdma.source1555 = and i64 %kdma.source1551, 123, !dbg !12
  %kdma.source1556 = lshr i64 %kdma.source1551, 2, !dbg !12
  %kdma.source1557 = and i64 %kdma.source1556, 4, !dbg !12
  %kdma.source1558 = add nuw nsw i64 %kdma.source1557, %kdma.source1551, !dbg !12
  %kdma.source1559 = and i64 %kdma.source1558, 4, !dbg !12
  %kdma.source1560 = or disjoint i64 %kdma.source1555, %kdma.source1559, !dbg !12
  %kdma.source1561 = shl nuw nsw i64 %kdma.source1560, 7, !dbg !12
  %kdma.source1562 = or disjoint i64 %kdma.source1554, %kdma.source1561, !dbg !12
  %kdma.source1563 = or disjoint i64 %238, %kdma.source1562, !dbg !12
  %kdma.source1564 = trunc nuw i64 %kdma.source1563 to i32, !dbg !12
  %kdma.source1565 = shl nuw nsw i32 %kdma.source1564, 1, !dbg !12
  %kdma.offset1566 = add i32 %kdma.source1565, %kdma.intra, !dbg !12
  %kdma.lds1567 = add i32 %dma.buffer.lds1226, 512, !dbg !12
  %250 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1567, i32 %kdma.offset1566, <4 x i32> %dma.buffer.rsrc1225) #10, !dbg !12
  %lds.const.gep1105 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8192, !dbg !14
  %251 = load <4 x half>, ptr addrspace(3) %lds.const.gep1105, align 8, !dbg !14
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %251, <4 x float> %178) #9, !dbg !14
  %lds.const.gep1106 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12288, !dbg !14
  %253 = load <4 x half>, ptr addrspace(3) %lds.const.gep1106, align 8, !dbg !14
  %lds.const.gep1107 = getelementptr i8, ptr addrspace(3) %176, i64 8256, !dbg !14
  %254 = load <4 x half>, ptr addrspace(3) %lds.const.gep1107, align 8, !dbg !14
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %253, <4 x float> %252) #9, !dbg !14
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %254, <4 x float> %182) #9, !dbg !14
  %lds.const.gep1108 = getelementptr i8, ptr addrspace(3) %176, i64 12352, !dbg !14
  %257 = load <4 x half>, ptr addrspace(3) %lds.const.gep1108, align 8, !dbg !14
  %lds.const.gep1109 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8320, !dbg !14
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep1109, align 8, !dbg !14
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %257, <4 x float> %256) #9, !dbg !14
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %258, <4 x float> %186) #9, !dbg !14
  %lds.const.gep1110 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12416, !dbg !14
  %261 = load <4 x half>, ptr addrspace(3) %lds.const.gep1110, align 8, !dbg !14
  %lds.const.gep1111 = getelementptr i8, ptr addrspace(3) %176, i64 8384, !dbg !14
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep1111, align 8, !dbg !14
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %261, <4 x float> %260) #9, !dbg !14
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %262, <4 x float> %190) #9, !dbg !14
  %lds.const.gep1112 = getelementptr i8, ptr addrspace(3) %176, i64 12480, !dbg !14
  %265 = load <4 x half>, ptr addrspace(3) %lds.const.gep1112, align 8, !dbg !14
  %lds.const.gep1113 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9216, !dbg !14
  %266 = load <4 x half>, ptr addrspace(3) %lds.const.gep1113, align 8, !dbg !14
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %265, <4 x float> %264) #9, !dbg !14
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %266, <4 x float> %194) #9, !dbg !14
  %lds.const.gep1114 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13312, !dbg !14
  %269 = load <4 x half>, ptr addrspace(3) %lds.const.gep1114, align 8, !dbg !14
  %lds.const.gep1115 = getelementptr i8, ptr addrspace(3) %176, i64 9280, !dbg !14
  %270 = load <4 x half>, ptr addrspace(3) %lds.const.gep1115, align 8, !dbg !14
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %269, <4 x float> %268) #9, !dbg !14
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %270, <4 x float> %198) #9, !dbg !14
  %lds.const.gep1116 = getelementptr i8, ptr addrspace(3) %176, i64 13376, !dbg !14
  %273 = load <4 x half>, ptr addrspace(3) %lds.const.gep1116, align 8, !dbg !14
  %lds.const.gep1117 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9344, !dbg !14
  %274 = load <4 x half>, ptr addrspace(3) %lds.const.gep1117, align 8, !dbg !14
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %273, <4 x float> %272) #9, !dbg !14
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %274, <4 x float> %202) #9, !dbg !14
  %lds.const.gep1118 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13440, !dbg !14
  %277 = load <4 x half>, ptr addrspace(3) %lds.const.gep1118, align 8, !dbg !14
  %lds.const.gep1119 = getelementptr i8, ptr addrspace(3) %176, i64 9408, !dbg !14
  %278 = load <4 x half>, ptr addrspace(3) %lds.const.gep1119, align 8, !dbg !14
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %277, <4 x float> %276) #9, !dbg !14
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %278, <4 x float> %206) #9, !dbg !14
  %lds.const.gep1120 = getelementptr i8, ptr addrspace(3) %176, i64 13504, !dbg !14
  %281 = load <4 x half>, ptr addrspace(3) %lds.const.gep1120, align 8, !dbg !14
  %lds.const.gep1121 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8192, !dbg !14
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep1121, align 8, !dbg !14
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %281, <4 x float> %280) #9, !dbg !14
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %282, <4 x float> %210) #9, !dbg !14
  %lds.const.gep1122 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12288, !dbg !14
  %285 = load <4 x half>, ptr addrspace(3) %lds.const.gep1122, align 8, !dbg !14
  %lds.const.gep1123 = getelementptr i8, ptr addrspace(3) %176, i64 8256, !dbg !14
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep1123, align 8, !dbg !14
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %285, <4 x float> %284) #9, !dbg !14
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %286, <4 x float> %214) #9, !dbg !14
  %lds.const.gep1124 = getelementptr i8, ptr addrspace(3) %176, i64 12352, !dbg !14
  %289 = load <4 x half>, ptr addrspace(3) %lds.const.gep1124, align 8, !dbg !14
  %lds.const.gep1125 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8320, !dbg !14
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep1125, align 8, !dbg !14
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %289, <4 x float> %288) #9, !dbg !14
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %290, <4 x float> %218) #9, !dbg !14
  %lds.const.gep1126 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12416, !dbg !14
  %293 = load <4 x half>, ptr addrspace(3) %lds.const.gep1126, align 8, !dbg !14
  %lds.const.gep1127 = getelementptr i8, ptr addrspace(3) %176, i64 8384, !dbg !14
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1127, align 8, !dbg !14
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %293, <4 x float> %292) #9, !dbg !14
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %294, <4 x float> %222) #9, !dbg !14
  %lds.const.gep1128 = getelementptr i8, ptr addrspace(3) %176, i64 12480, !dbg !14
  %297 = load <4 x half>, ptr addrspace(3) %lds.const.gep1128, align 8, !dbg !14
  %lds.const.gep1129 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9216, !dbg !14
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep1129, align 8, !dbg !14
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %297, <4 x float> %296) #9, !dbg !14
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %298, <4 x float> %226) #9, !dbg !14
  %lds.const.gep1130 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13312, !dbg !14
  %301 = load <4 x half>, ptr addrspace(3) %lds.const.gep1130, align 8, !dbg !14
  %lds.const.gep1131 = getelementptr i8, ptr addrspace(3) %176, i64 9280, !dbg !14
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep1131, align 8, !dbg !14
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %301, <4 x float> %300) #9, !dbg !14
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %302, <4 x float> %230) #9, !dbg !14
  %lds.const.gep1132 = getelementptr i8, ptr addrspace(3) %176, i64 13376, !dbg !14
  %305 = load <4 x half>, ptr addrspace(3) %lds.const.gep1132, align 8, !dbg !14
  %lds.const.gep1133 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9344, !dbg !14
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep1133, align 8, !dbg !14
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %305, <4 x float> %304) #9, !dbg !14
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %306, <4 x float> %234) #9, !dbg !14
  %lds.const.gep1134 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13440, !dbg !14
  %309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1134, align 8, !dbg !14
  %lds.const.gep1135 = getelementptr i8, ptr addrspace(3) %176, i64 9408, !dbg !14
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep1135, align 8, !dbg !14
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %309, <4 x float> %308) #9, !dbg !14
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %310, <4 x float> %237) #9, !dbg !14
  %lds.const.gep1136 = getelementptr i8, ptr addrspace(3) %176, i64 13504, !dbg !14
  %313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1136, align 8, !dbg !14
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %313, <4 x float> %312) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %315 = or disjoint i64 %140, 96, !dbg !12
  %316 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local402), !dbg !12
  %317 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %318 = insertelement <2 x i64> poison, i64 %317, i64 0, !dbg !12
  %319 = bitcast <2 x i64> %318 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1222 = shufflevector <4 x i32> %319, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1223 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %316), !dbg !12
  %kdma.source1568 = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1569 = lshr i64 %kdma.source1568, 4, !dbg !12
  %kdma.source1570 = and i64 %kdma.source1569, 8, !dbg !12
  %kdma.source1571 = and i64 %kdma.source1568, 123, !dbg !12
  %kdma.source1572 = lshr i64 %kdma.source1568, 2, !dbg !12
  %kdma.source1573 = and i64 %kdma.source1572, 4, !dbg !12
  %kdma.source1574 = add nuw nsw i64 %kdma.source1573, %kdma.source1568, !dbg !12
  %kdma.source1575 = and i64 %kdma.source1574, 4, !dbg !12
  %kdma.source1576 = or disjoint i64 %kdma.source1571, %kdma.source1575, !dbg !12
  %kdma.source1577 = shl nuw nsw i64 %kdma.source1576, 7, !dbg !12
  %kdma.source1578 = or disjoint i64 %kdma.source1570, %kdma.source1577, !dbg !12
  %kdma.source1579 = or disjoint i64 %315, %kdma.source1578, !dbg !12
  %kdma.source1580 = trunc nuw i64 %kdma.source1579 to i32, !dbg !12
  %kdma.source1581 = shl nuw nsw i32 %kdma.source1580, 1, !dbg !12
  %kdma.offset1582 = add i32 %kdma.source1581, %kdma.intra, !dbg !12
  %kdma.lds1583 = add i32 %dma.buffer.lds1223, 0, !dbg !12
  %320 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1583, i32 %kdma.offset1582, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !12
  %kdma.source1584 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1585 = lshr i64 %kdma.source1584, 4, !dbg !12
  %kdma.source1586 = and i64 %kdma.source1585, 8, !dbg !12
  %kdma.source1587 = and i64 %kdma.source1584, 123, !dbg !12
  %kdma.source1588 = lshr i64 %kdma.source1584, 2, !dbg !12
  %kdma.source1589 = and i64 %kdma.source1588, 4, !dbg !12
  %kdma.source1590 = add nuw nsw i64 %kdma.source1589, %kdma.source1584, !dbg !12
  %kdma.source1591 = and i64 %kdma.source1590, 4, !dbg !12
  %kdma.source1592 = or disjoint i64 %kdma.source1587, %kdma.source1591, !dbg !12
  %kdma.source1593 = shl nuw nsw i64 %kdma.source1592, 7, !dbg !12
  %kdma.source1594 = or disjoint i64 %kdma.source1586, %kdma.source1593, !dbg !12
  %kdma.source1595 = or disjoint i64 %315, %kdma.source1594, !dbg !12
  %kdma.source1596 = trunc nuw i64 %kdma.source1595 to i32, !dbg !12
  %kdma.source1597 = shl nuw nsw i32 %kdma.source1596, 1, !dbg !12
  %kdma.offset1598 = add i32 %kdma.source1597, %kdma.intra, !dbg !12
  %kdma.lds1599 = add i32 %dma.buffer.lds1223, 512, !dbg !12
  %321 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1599, i32 %kdma.offset1598, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !12
  %322 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local406), !dbg !12
  %323 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %324 = insertelement <2 x i64> poison, i64 %323, i64 0, !dbg !12
  %325 = bitcast <2 x i64> %324 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1219 = shufflevector <4 x i32> %325, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1220 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %322), !dbg !12
  %kdma.source1600 = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1601 = or disjoint i64 %kdma.source1600, 256, !dbg !12
  %kdma.source1602 = lshr i64 %kdma.source1601, 4, !dbg !12
  %kdma.source1603 = and i64 %kdma.source1602, 24, !dbg !12
  %kdma.source1604 = and i64 %kdma.source1600, 123, !dbg !12
  %kdma.source1605 = lshr i64 %kdma.source1600, 2, !dbg !12
  %kdma.source1606 = and i64 %kdma.source1605, 4, !dbg !12
  %kdma.source1607 = add nuw nsw i64 %kdma.source1606, %kdma.source1600, !dbg !12
  %kdma.source1608 = and i64 %kdma.source1607, 4, !dbg !12
  %kdma.source1609 = or disjoint i64 %kdma.source1604, %kdma.source1608, !dbg !12
  %kdma.source1610 = shl nuw nsw i64 %kdma.source1609, 7, !dbg !12
  %kdma.source1611 = or disjoint i64 %kdma.source1603, %kdma.source1610, !dbg !12
  %kdma.source1612 = or disjoint i64 %315, %kdma.source1611, !dbg !12
  %kdma.source1613 = trunc nuw i64 %kdma.source1612 to i32, !dbg !12
  %kdma.source1614 = shl nuw nsw i32 %kdma.source1613, 1, !dbg !12
  %kdma.offset1615 = add i32 %kdma.source1614, %kdma.intra, !dbg !12
  %kdma.lds1616 = add i32 %dma.buffer.lds1220, 0, !dbg !12
  %326 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1616, i32 %kdma.offset1615, <4 x i32> %dma.buffer.rsrc1219) #10, !dbg !12
  %kdma.source1617 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1618 = or disjoint i64 %kdma.source1617, 256, !dbg !12
  %kdma.source1619 = lshr i64 %kdma.source1618, 4, !dbg !12
  %kdma.source1620 = and i64 %kdma.source1619, 24, !dbg !12
  %kdma.source1621 = and i64 %kdma.source1617, 123, !dbg !12
  %kdma.source1622 = lshr i64 %kdma.source1617, 2, !dbg !12
  %kdma.source1623 = and i64 %kdma.source1622, 4, !dbg !12
  %kdma.source1624 = add nuw nsw i64 %kdma.source1623, %kdma.source1617, !dbg !12
  %kdma.source1625 = and i64 %kdma.source1624, 4, !dbg !12
  %kdma.source1626 = or disjoint i64 %kdma.source1621, %kdma.source1625, !dbg !12
  %kdma.source1627 = shl nuw nsw i64 %kdma.source1626, 7, !dbg !12
  %kdma.source1628 = or disjoint i64 %kdma.source1620, %kdma.source1627, !dbg !12
  %kdma.source1629 = or disjoint i64 %315, %kdma.source1628, !dbg !12
  %kdma.source1630 = trunc nuw i64 %kdma.source1629 to i32, !dbg !12
  %kdma.source1631 = shl nuw nsw i32 %kdma.source1630, 1, !dbg !12
  %kdma.offset1632 = add i32 %kdma.source1631, %kdma.intra, !dbg !12
  %kdma.lds1633 = add i32 %dma.buffer.lds1220, 512, !dbg !12
  %327 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1633, i32 %kdma.offset1632, <4 x i32> %dma.buffer.rsrc1219) #10, !dbg !12
  %328 = load <4 x half>, ptr addrspace(3) %address.local426, align 8, !dbg !14
  %329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %328, <4 x float> %255) #9, !dbg !14
  %lds.const.gep1137 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4096, !dbg !14
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1137, align 8, !dbg !14
  %331 = load <4 x half>, ptr addrspace(3) %address.local455, align 8, !dbg !14
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %330, <4 x float> %329) #9, !dbg !14
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %331, <4 x float> %259) #9, !dbg !14
  %lds.const.gep1138 = getelementptr i8, ptr addrspace(3) %176, i64 4160, !dbg !14
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1138, align 8, !dbg !14
  %lds.const.gep1139 = getelementptr i8, ptr addrspace(3) %address.local426, i64 128, !dbg !14
  %335 = load <4 x half>, ptr addrspace(3) %lds.const.gep1139, align 8, !dbg !14
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %334, <4 x float> %333) #9, !dbg !14
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %335, <4 x float> %263) #9, !dbg !14
  %lds.const.gep1140 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4224, !dbg !14
  %338 = load <4 x half>, ptr addrspace(3) %lds.const.gep1140, align 8, !dbg !14
  %lds.const.gep1141 = getelementptr i8, ptr addrspace(3) %176, i64 192, !dbg !14
  %339 = load <4 x half>, ptr addrspace(3) %lds.const.gep1141, align 8, !dbg !14
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %338, <4 x float> %337) #9, !dbg !14
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %339, <4 x float> %267) #9, !dbg !14
  %lds.const.gep1142 = getelementptr i8, ptr addrspace(3) %176, i64 4288, !dbg !14
  %342 = load <4 x half>, ptr addrspace(3) %lds.const.gep1142, align 8, !dbg !14
  %lds.const.gep1143 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1024, !dbg !14
  %343 = load <4 x half>, ptr addrspace(3) %lds.const.gep1143, align 8, !dbg !14
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %342, <4 x float> %341) #9, !dbg !14
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %343, <4 x float> %271) #9, !dbg !14
  %lds.const.gep1144 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5120, !dbg !14
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1144, align 8, !dbg !14
  %lds.const.gep1145 = getelementptr i8, ptr addrspace(3) %176, i64 1088, !dbg !14
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep1145, align 8, !dbg !14
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %346, <4 x float> %345) #9, !dbg !14
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %347, <4 x float> %275) #9, !dbg !14
  %lds.const.gep1146 = getelementptr i8, ptr addrspace(3) %176, i64 5184, !dbg !14
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1146, align 8, !dbg !14
  %lds.const.gep1147 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1152, !dbg !14
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep1147, align 8, !dbg !14
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %350, <4 x float> %349) #9, !dbg !14
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %351, <4 x float> %279) #9, !dbg !14
  %lds.const.gep1148 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5248, !dbg !14
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1148, align 8, !dbg !14
  %lds.const.gep1149 = getelementptr i8, ptr addrspace(3) %176, i64 1216, !dbg !14
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep1149, align 8, !dbg !14
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %354, <4 x float> %353) #9, !dbg !14
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %355, <4 x float> %283) #9, !dbg !14
  %lds.const.gep1150 = getelementptr i8, ptr addrspace(3) %176, i64 5312, !dbg !14
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1150, align 8, !dbg !14
  %359 = load <4 x half>, ptr addrspace(3) %address.local426, align 8, !dbg !14
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %358, <4 x float> %357) #9, !dbg !14
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %359, <4 x float> %287) #9, !dbg !14
  %lds.const.gep1151 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4096, !dbg !14
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1151, align 8, !dbg !14
  %363 = load <4 x half>, ptr addrspace(3) %address.local455, align 8, !dbg !14
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %362, <4 x float> %361) #9, !dbg !14
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %363, <4 x float> %291) #9, !dbg !14
  %lds.const.gep1152 = getelementptr i8, ptr addrspace(3) %176, i64 4160, !dbg !14
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1152, align 8, !dbg !14
  %lds.const.gep1153 = getelementptr i8, ptr addrspace(3) %address.local426, i64 128, !dbg !14
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1153, align 8, !dbg !14
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %366, <4 x float> %365) #9, !dbg !14
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %367, <4 x float> %295) #9, !dbg !14
  %lds.const.gep1154 = getelementptr i8, ptr addrspace(3) %address.local426, i64 4224, !dbg !14
  %370 = load <4 x half>, ptr addrspace(3) %lds.const.gep1154, align 8, !dbg !14
  %lds.const.gep1155 = getelementptr i8, ptr addrspace(3) %176, i64 192, !dbg !14
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep1155, align 8, !dbg !14
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %370, <4 x float> %369) #9, !dbg !14
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %371, <4 x float> %299) #9, !dbg !14
  %lds.const.gep1156 = getelementptr i8, ptr addrspace(3) %176, i64 4288, !dbg !14
  %374 = load <4 x half>, ptr addrspace(3) %lds.const.gep1156, align 8, !dbg !14
  %lds.const.gep1157 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1024, !dbg !14
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep1157, align 8, !dbg !14
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %374, <4 x float> %373) #9, !dbg !14
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %375, <4 x float> %303) #9, !dbg !14
  %lds.const.gep1158 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5120, !dbg !14
  %378 = load <4 x half>, ptr addrspace(3) %lds.const.gep1158, align 8, !dbg !14
  %lds.const.gep1159 = getelementptr i8, ptr addrspace(3) %176, i64 1088, !dbg !14
  %379 = load <4 x half>, ptr addrspace(3) %lds.const.gep1159, align 8, !dbg !14
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %378, <4 x float> %377) #9, !dbg !14
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %379, <4 x float> %307) #9, !dbg !14
  %lds.const.gep1160 = getelementptr i8, ptr addrspace(3) %176, i64 5184, !dbg !14
  %382 = load <4 x half>, ptr addrspace(3) %lds.const.gep1160, align 8, !dbg !14
  %lds.const.gep1161 = getelementptr i8, ptr addrspace(3) %address.local426, i64 1152, !dbg !14
  %383 = load <4 x half>, ptr addrspace(3) %lds.const.gep1161, align 8, !dbg !14
  %384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %382, <4 x float> %381) #9, !dbg !14
  %385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %383, <4 x float> %311) #9, !dbg !14
  %lds.const.gep1162 = getelementptr i8, ptr addrspace(3) %address.local426, i64 5248, !dbg !14
  %386 = load <4 x half>, ptr addrspace(3) %lds.const.gep1162, align 8, !dbg !14
  %lds.const.gep1163 = getelementptr i8, ptr addrspace(3) %176, i64 1216, !dbg !14
  %387 = load <4 x half>, ptr addrspace(3) %lds.const.gep1163, align 8, !dbg !14
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %386, <4 x float> %385) #9, !dbg !14
  %389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %387, <4 x float> %314) #9, !dbg !14
  %lds.const.gep1164 = getelementptr i8, ptr addrspace(3) %176, i64 5312, !dbg !14
  %390 = load <4 x half>, ptr addrspace(3) %lds.const.gep1164, align 8, !dbg !14
  %391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %390, <4 x float> %389) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !15
  %address.local784 = shl nuw nsw i64 %address.local366, 5, !dbg !15
  %address.local785 = and i64 %address.local784, 8064, !dbg !15
  %address.local786 = shl nuw nsw i64 %address.local366, 3, !dbg !15
  %address.local787 = and i64 %address.local786, 24, !dbg !15
  %address.local788 = or disjoint i64 %address.local785, %address.local787, !dbg !15
  %393 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx12861287 = or disjoint i64 %140, %address.local788, !dbg !15
  %.idx12861287.tr = trunc nuw i64 %.idx12861287 to i32, !dbg !15
  %dma.buffer.offset1215 = shl nuw nsw i32 %.idx12861287.tr, 1, !dbg !15
  %394 = insertelement <2 x i64> poison, i64 %393, i64 0, !dbg !15
  %395 = bitcast <2 x i64> %394 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1216 = shufflevector <4 x i32> %395, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1217 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %392), !dbg !15
  %396 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1217, i32 %dma.buffer.offset1215, <4 x i32> %dma.buffer.rsrc1216) #10, !dbg !15
  %397 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !15
  %address.local790 = shl nuw nsw i64 %address.local383, 5, !dbg !15
  %address.local791 = and i64 %address.local790, 16256, !dbg !15
  %address.local792 = shl nuw nsw i64 %address.local366, 3, !dbg !15
  %address.local793 = and i64 %address.local792, 24, !dbg !15
  %address.local794 = or disjoint i64 %address.local791, %address.local793, !dbg !15
  %398 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx12881289 = or disjoint i64 %140, %address.local794, !dbg !15
  %.idx12881289.tr = trunc nuw i64 %.idx12881289 to i32, !dbg !15
  %dma.buffer.offset1212 = shl nuw nsw i32 %.idx12881289.tr, 1, !dbg !15
  %399 = insertelement <2 x i64> poison, i64 %398, i64 0, !dbg !15
  %400 = bitcast <2 x i64> %399 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1213 = shufflevector <4 x i32> %400, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1214 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %397), !dbg !15
  %401 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1214, i32 %dma.buffer.offset1212, <4 x i32> %dma.buffer.rsrc1213) #10, !dbg !15
  %lds.const.gep1165 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8192, !dbg !14
  %402 = load <4 x half>, ptr addrspace(3) %lds.const.gep1165, align 8, !dbg !14
  %403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %402, <4 x float> %332) #9, !dbg !14
  %lds.const.gep1166 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12288, !dbg !14
  %404 = load <4 x half>, ptr addrspace(3) %lds.const.gep1166, align 8, !dbg !14
  %lds.const.gep1167 = getelementptr i8, ptr addrspace(3) %176, i64 8256, !dbg !14
  %405 = load <4 x half>, ptr addrspace(3) %lds.const.gep1167, align 8, !dbg !14
  %406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %404, <4 x float> %403) #9, !dbg !14
  %407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %405, <4 x float> %336) #9, !dbg !14
  %408 = extractelement <4 x float> %406, i64 0, !dbg !19
  %409 = call float @llvm.maxnum.f32(float %408, float 0xFFF0000000000000), !dbg !20
  %410 = extractelement <4 x float> %406, i64 1, !dbg !19
  %411 = call float @llvm.maxnum.f32(float %409, float %410), !dbg !20
  %412 = extractelement <4 x float> %406, i64 2, !dbg !19
  %413 = call float @llvm.maxnum.f32(float %411, float %412), !dbg !20
  %414 = extractelement <4 x float> %406, i64 3, !dbg !19
  %415 = call float @llvm.maxnum.f32(float %413, float %414), !dbg !20
  %lds.const.gep1168 = getelementptr i8, ptr addrspace(3) %176, i64 12352, !dbg !14
  %416 = load <4 x half>, ptr addrspace(3) %lds.const.gep1168, align 8, !dbg !14
  %lds.const.gep1169 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8320, !dbg !14
  %417 = load <4 x half>, ptr addrspace(3) %lds.const.gep1169, align 8, !dbg !14
  %418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %416, <4 x float> %407) #9, !dbg !14
  %419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %417, <4 x float> %340) #9, !dbg !14
  %420 = extractelement <4 x float> %418, i64 0, !dbg !19
  %421 = call float @llvm.maxnum.f32(float %415, float %420), !dbg !20
  %422 = extractelement <4 x float> %418, i64 1, !dbg !19
  %423 = call float @llvm.maxnum.f32(float %421, float %422), !dbg !20
  %424 = extractelement <4 x float> %418, i64 2, !dbg !19
  %425 = call float @llvm.maxnum.f32(float %423, float %424), !dbg !20
  %426 = extractelement <4 x float> %418, i64 3, !dbg !19
  %427 = call float @llvm.maxnum.f32(float %425, float %426), !dbg !20
  %lds.const.gep1170 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12416, !dbg !14
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1170, align 8, !dbg !14
  %lds.const.gep1171 = getelementptr i8, ptr addrspace(3) %176, i64 8384, !dbg !14
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1171, align 8, !dbg !14
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %428, <4 x float> %419) #9, !dbg !14
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %429, <4 x float> %344) #9, !dbg !14
  %432 = extractelement <4 x float> %430, i64 0, !dbg !19
  %433 = call float @llvm.maxnum.f32(float %427, float %432), !dbg !20
  %434 = extractelement <4 x float> %430, i64 1, !dbg !19
  %435 = call float @llvm.maxnum.f32(float %433, float %434), !dbg !20
  %436 = extractelement <4 x float> %430, i64 2, !dbg !19
  %437 = call float @llvm.maxnum.f32(float %435, float %436), !dbg !20
  %438 = extractelement <4 x float> %430, i64 3, !dbg !19
  %439 = call float @llvm.maxnum.f32(float %437, float %438), !dbg !20
  %lds.const.gep1172 = getelementptr i8, ptr addrspace(3) %176, i64 12480, !dbg !14
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1172, align 8, !dbg !14
  %lds.const.gep1173 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9216, !dbg !14
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1173, align 8, !dbg !14
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %440, <4 x float> %431) #9, !dbg !14
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %441, <4 x float> %348) #9, !dbg !14
  %444 = extractelement <4 x float> %442, i64 0, !dbg !19
  %445 = call float @llvm.maxnum.f32(float %439, float %444), !dbg !20
  %446 = extractelement <4 x float> %442, i64 1, !dbg !19
  %447 = call float @llvm.maxnum.f32(float %445, float %446), !dbg !20
  %448 = extractelement <4 x float> %442, i64 2, !dbg !19
  %449 = call float @llvm.maxnum.f32(float %447, float %448), !dbg !20
  %450 = extractelement <4 x float> %442, i64 3, !dbg !19
  %451 = call float @llvm.maxnum.f32(float %449, float %450), !dbg !20
  %lds.const.gep1174 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13312, !dbg !14
  %452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1174, align 8, !dbg !14
  %lds.const.gep1175 = getelementptr i8, ptr addrspace(3) %176, i64 9280, !dbg !14
  %453 = load <4 x half>, ptr addrspace(3) %lds.const.gep1175, align 8, !dbg !14
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %452, <4 x float> %443) #9, !dbg !14
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %453, <4 x float> %352) #9, !dbg !14
  %456 = extractelement <4 x float> %454, i64 0, !dbg !19
  %457 = call float @llvm.maxnum.f32(float %451, float %456), !dbg !20
  %458 = extractelement <4 x float> %454, i64 1, !dbg !19
  %459 = call float @llvm.maxnum.f32(float %457, float %458), !dbg !20
  %460 = extractelement <4 x float> %454, i64 2, !dbg !19
  %461 = call float @llvm.maxnum.f32(float %459, float %460), !dbg !20
  %462 = extractelement <4 x float> %454, i64 3, !dbg !19
  %463 = call float @llvm.maxnum.f32(float %461, float %462), !dbg !20
  %lds.const.gep1176 = getelementptr i8, ptr addrspace(3) %176, i64 13376, !dbg !14
  %464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1176, align 8, !dbg !14
  %lds.const.gep1177 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9344, !dbg !14
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1177, align 8, !dbg !14
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %464, <4 x float> %455) #9, !dbg !14
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %465, <4 x float> %356) #9, !dbg !14
  %468 = extractelement <4 x float> %466, i64 0, !dbg !19
  %469 = call float @llvm.maxnum.f32(float %463, float %468), !dbg !20
  %470 = extractelement <4 x float> %466, i64 1, !dbg !19
  %471 = call float @llvm.maxnum.f32(float %469, float %470), !dbg !20
  %472 = extractelement <4 x float> %466, i64 2, !dbg !19
  %473 = call float @llvm.maxnum.f32(float %471, float %472), !dbg !20
  %474 = extractelement <4 x float> %466, i64 3, !dbg !19
  %475 = call float @llvm.maxnum.f32(float %473, float %474), !dbg !20
  %lds.const.gep1178 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13440, !dbg !14
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1178, align 8, !dbg !14
  %lds.const.gep1179 = getelementptr i8, ptr addrspace(3) %176, i64 9408, !dbg !14
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1179, align 8, !dbg !14
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %476, <4 x float> %467) #9, !dbg !14
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %477, <4 x float> %360) #9, !dbg !14
  %480 = extractelement <4 x float> %478, i64 0, !dbg !19
  %481 = call float @llvm.maxnum.f32(float %475, float %480), !dbg !20
  %482 = extractelement <4 x float> %478, i64 1, !dbg !19
  %483 = call float @llvm.maxnum.f32(float %481, float %482), !dbg !20
  %484 = extractelement <4 x float> %478, i64 2, !dbg !19
  %485 = call float @llvm.maxnum.f32(float %483, float %484), !dbg !20
  %486 = extractelement <4 x float> %478, i64 3, !dbg !19
  %487 = call float @llvm.maxnum.f32(float %485, float %486), !dbg !20
  %lds.const.gep1180 = getelementptr i8, ptr addrspace(3) %176, i64 13504, !dbg !14
  %488 = load <4 x half>, ptr addrspace(3) %lds.const.gep1180, align 8, !dbg !14
  %lds.const.gep1181 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8192, !dbg !14
  %489 = load <4 x half>, ptr addrspace(3) %lds.const.gep1181, align 8, !dbg !14
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %488, <4 x float> %479) #9, !dbg !14
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %489, <4 x float> %364) #9, !dbg !14
  %492 = extractelement <4 x float> %490, i64 0, !dbg !19
  %493 = call float @llvm.maxnum.f32(float %487, float %492), !dbg !20
  %494 = extractelement <4 x float> %490, i64 1, !dbg !19
  %495 = call float @llvm.maxnum.f32(float %493, float %494), !dbg !20
  %496 = extractelement <4 x float> %490, i64 2, !dbg !19
  %497 = call float @llvm.maxnum.f32(float %495, float %496), !dbg !20
  %498 = extractelement <4 x float> %490, i64 3, !dbg !19
  %499 = call float @llvm.maxnum.f32(float %497, float %498), !dbg !20
  %lds.const.gep1182 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12288, !dbg !14
  %500 = load <4 x half>, ptr addrspace(3) %lds.const.gep1182, align 8, !dbg !14
  %lds.const.gep1183 = getelementptr i8, ptr addrspace(3) %176, i64 8256, !dbg !14
  %501 = load <4 x half>, ptr addrspace(3) %lds.const.gep1183, align 8, !dbg !14
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %500, <4 x float> %491) #9, !dbg !14
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %501, <4 x float> %368) #9, !dbg !14
  %504 = extractelement <4 x float> %502, i64 0, !dbg !19
  %505 = extractelement <4 x float> %502, i64 1, !dbg !19
  %506 = extractelement <4 x float> %502, i64 2, !dbg !19
  %507 = extractelement <4 x float> %502, i64 3, !dbg !19
  %lds.const.gep1184 = getelementptr i8, ptr addrspace(3) %176, i64 12352, !dbg !14
  %508 = load <4 x half>, ptr addrspace(3) %lds.const.gep1184, align 8, !dbg !14
  %lds.const.gep1185 = getelementptr i8, ptr addrspace(3) %address.local426, i64 8320, !dbg !14
  %509 = load <4 x half>, ptr addrspace(3) %lds.const.gep1185, align 8, !dbg !14
  %510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %508, <4 x float> %503) #9, !dbg !14
  %511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %509, <4 x float> %372) #9, !dbg !14
  %512 = extractelement <4 x float> %510, i64 0, !dbg !19
  %513 = extractelement <4 x float> %510, i64 1, !dbg !19
  %514 = extractelement <4 x float> %510, i64 2, !dbg !19
  %515 = extractelement <4 x float> %510, i64 3, !dbg !19
  %lds.const.gep1186 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12416, !dbg !14
  %516 = load <4 x half>, ptr addrspace(3) %lds.const.gep1186, align 8, !dbg !14
  %lds.const.gep1187 = getelementptr i8, ptr addrspace(3) %176, i64 8384, !dbg !14
  %517 = load <4 x half>, ptr addrspace(3) %lds.const.gep1187, align 8, !dbg !14
  %518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %516, <4 x float> %511) #9, !dbg !14
  %519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %517, <4 x float> %376) #9, !dbg !14
  %520 = extractelement <4 x float> %518, i64 0, !dbg !19
  %521 = extractelement <4 x float> %518, i64 1, !dbg !19
  %522 = extractelement <4 x float> %518, i64 2, !dbg !19
  %523 = extractelement <4 x float> %518, i64 3, !dbg !19
  %lds.const.gep1188 = getelementptr i8, ptr addrspace(3) %176, i64 12480, !dbg !14
  %524 = load <4 x half>, ptr addrspace(3) %lds.const.gep1188, align 8, !dbg !14
  %lds.const.gep1189 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9216, !dbg !14
  %525 = load <4 x half>, ptr addrspace(3) %lds.const.gep1189, align 8, !dbg !14
  %526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %524, <4 x float> %519) #9, !dbg !14
  %527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %525, <4 x float> %380) #9, !dbg !14
  %528 = extractelement <4 x float> %526, i64 0, !dbg !19
  %529 = extractelement <4 x float> %526, i64 1, !dbg !19
  %530 = extractelement <4 x float> %526, i64 2, !dbg !19
  %531 = extractelement <4 x float> %526, i64 3, !dbg !19
  %lds.const.gep1190 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13312, !dbg !14
  %532 = load <4 x half>, ptr addrspace(3) %lds.const.gep1190, align 8, !dbg !14
  %lds.const.gep1191 = getelementptr i8, ptr addrspace(3) %176, i64 9280, !dbg !14
  %533 = load <4 x half>, ptr addrspace(3) %lds.const.gep1191, align 8, !dbg !14
  %534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %532, <4 x float> %527) #9, !dbg !14
  %535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %533, <4 x float> %384) #9, !dbg !14
  %536 = extractelement <4 x float> %534, i64 0, !dbg !19
  %537 = extractelement <4 x float> %534, i64 1, !dbg !19
  %538 = extractelement <4 x float> %534, i64 2, !dbg !19
  %539 = extractelement <4 x float> %534, i64 3, !dbg !19
  %lds.const.gep1192 = getelementptr i8, ptr addrspace(3) %176, i64 13376, !dbg !14
  %540 = load <4 x half>, ptr addrspace(3) %lds.const.gep1192, align 8, !dbg !14
  %lds.const.gep1193 = getelementptr i8, ptr addrspace(3) %address.local426, i64 9344, !dbg !14
  %541 = load <4 x half>, ptr addrspace(3) %lds.const.gep1193, align 8, !dbg !14
  %542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %540, <4 x float> %535) #9, !dbg !14
  %543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %541, <4 x float> %388) #9, !dbg !14
  %544 = extractelement <4 x float> %542, i64 0, !dbg !19
  %545 = extractelement <4 x float> %542, i64 1, !dbg !19
  %546 = extractelement <4 x float> %542, i64 2, !dbg !19
  %547 = extractelement <4 x float> %542, i64 3, !dbg !19
  %lds.const.gep1194 = getelementptr i8, ptr addrspace(3) %address.local426, i64 13440, !dbg !14
  %548 = load <4 x half>, ptr addrspace(3) %lds.const.gep1194, align 8, !dbg !14
  %lds.const.gep1195 = getelementptr i8, ptr addrspace(3) %176, i64 9408, !dbg !14
  %549 = load <4 x half>, ptr addrspace(3) %lds.const.gep1195, align 8, !dbg !14
  %550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %548, <4 x float> %543) #9, !dbg !14
  %551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %549, <4 x float> %391) #9, !dbg !14
  %552 = extractelement <4 x float> %550, i64 0, !dbg !19
  %553 = extractelement <4 x float> %550, i64 1, !dbg !19
  %554 = extractelement <4 x float> %550, i64 2, !dbg !19
  %555 = extractelement <4 x float> %550, i64 3, !dbg !19
  %lds.const.gep1196 = getelementptr i8, ptr addrspace(3) %176, i64 13504, !dbg !14
  %556 = load <4 x half>, ptr addrspace(3) %lds.const.gep1196, align 8, !dbg !14
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %556, <4 x float> %551) #9, !dbg !14
  %558 = extractelement <4 x float> %557, i64 0, !dbg !19
  %559 = extractelement <4 x float> %557, i64 1, !dbg !19
  %560 = extractelement <4 x float> %557, i64 2, !dbg !19
  %561 = extractelement <4 x float> %557, i64 3, !dbg !19
  %562 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %563 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %562), !dbg !20
  %564 = add nuw nsw i32 %563, 64, !dbg !20
  %565 = xor i32 %563, 16, !dbg !20
  %566 = and i32 %564, 192, !dbg !20
  %567 = icmp samesign ult i32 %565, %566, !dbg !20
  %568 = select i1 %567, i32 %565, i32 %563, !dbg !20
  %569 = shl nuw nsw i32 %568, 2, !dbg !20
  %570 = bitcast float %499 to i32, !dbg !20
  %571 = call i32 @llvm.amdgcn.ds.bpermute(i32 %569, i32 %570), !dbg !20
  %572 = bitcast i32 %571 to float, !dbg !20
  %573 = call float @llvm.maxnum.f32(float %499, float %572), !dbg !20
  %574 = xor i32 %563, 32, !dbg !20
  %575 = icmp samesign ult i32 %574, %566, !dbg !20
  %576 = select i1 %575, i32 %574, i32 %563, !dbg !20
  %577 = shl nuw nsw i32 %576, 2, !dbg !20
  %578 = bitcast float %573 to i32, !dbg !20
  %579 = call i32 @llvm.amdgcn.ds.bpermute(i32 %577, i32 %578), !dbg !20
  %580 = bitcast i32 %579 to float, !dbg !20
  %581 = call float @llvm.maxnum.f32(float %573, float %580), !dbg !20
  %582 = extractvalue [2 x <1 x float>] %131, 0, !dbg !20
  %583 = insertelement <1 x float> poison, float %581, i64 0, !dbg !20
  %584 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %583, <1 x float> %582), !dbg !20
  %585 = call float @llvm.maxnum.f32(float %504, float 0xFFF0000000000000), !dbg !20
  %586 = call float @llvm.maxnum.f32(float %585, float %505), !dbg !20
  %587 = call float @llvm.maxnum.f32(float %586, float %506), !dbg !20
  %588 = call float @llvm.maxnum.f32(float %587, float %507), !dbg !20
  %589 = call float @llvm.maxnum.f32(float %588, float %512), !dbg !20
  %590 = call float @llvm.maxnum.f32(float %589, float %513), !dbg !20
  %591 = call float @llvm.maxnum.f32(float %590, float %514), !dbg !20
  %592 = call float @llvm.maxnum.f32(float %591, float %515), !dbg !20
  %593 = call float @llvm.maxnum.f32(float %592, float %520), !dbg !20
  %594 = call float @llvm.maxnum.f32(float %593, float %521), !dbg !20
  %595 = call float @llvm.maxnum.f32(float %594, float %522), !dbg !20
  %596 = call float @llvm.maxnum.f32(float %595, float %523), !dbg !20
  %597 = call float @llvm.maxnum.f32(float %596, float %528), !dbg !20
  %598 = call float @llvm.maxnum.f32(float %597, float %529), !dbg !20
  %599 = call float @llvm.maxnum.f32(float %598, float %530), !dbg !20
  %600 = call float @llvm.maxnum.f32(float %599, float %531), !dbg !20
  %601 = call float @llvm.maxnum.f32(float %600, float %536), !dbg !20
  %602 = call float @llvm.maxnum.f32(float %601, float %537), !dbg !20
  %603 = call float @llvm.maxnum.f32(float %602, float %538), !dbg !20
  %604 = call float @llvm.maxnum.f32(float %603, float %539), !dbg !20
  %605 = call float @llvm.maxnum.f32(float %604, float %544), !dbg !20
  %606 = call float @llvm.maxnum.f32(float %605, float %545), !dbg !20
  %607 = call float @llvm.maxnum.f32(float %606, float %546), !dbg !20
  %608 = call float @llvm.maxnum.f32(float %607, float %547), !dbg !20
  %609 = call float @llvm.maxnum.f32(float %608, float %552), !dbg !20
  %610 = call float @llvm.maxnum.f32(float %609, float %553), !dbg !20
  %611 = call float @llvm.maxnum.f32(float %610, float %554), !dbg !20
  %612 = call float @llvm.maxnum.f32(float %611, float %555), !dbg !20
  %613 = call float @llvm.maxnum.f32(float %612, float %558), !dbg !20
  %614 = call float @llvm.maxnum.f32(float %613, float %559), !dbg !20
  %615 = call float @llvm.maxnum.f32(float %614, float %560), !dbg !20
  %616 = call float @llvm.maxnum.f32(float %615, float %561), !dbg !20
  %617 = bitcast float %616 to i32, !dbg !20
  %618 = call i32 @llvm.amdgcn.ds.bpermute(i32 %569, i32 %617), !dbg !20
  %619 = bitcast i32 %618 to float, !dbg !20
  %620 = call float @llvm.maxnum.f32(float %616, float %619), !dbg !20
  %621 = bitcast float %620 to i32, !dbg !20
  %622 = call i32 @llvm.amdgcn.ds.bpermute(i32 %577, i32 %621), !dbg !20
  %623 = bitcast i32 %622 to float, !dbg !20
  %624 = call float @llvm.maxnum.f32(float %620, float %623), !dbg !20
  %625 = extractvalue [2 x <1 x float>] %131, 1, !dbg !20
  %626 = insertelement <1 x float> poison, float %624, i64 0, !dbg !20
  %627 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %626, <1 x float> %625), !dbg !20
  %628 = shufflevector <1 x float> %584, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %629 = fsub <4 x float> %406, %628, !dbg !21
  %630 = fsub <4 x float> %418, %628, !dbg !21
  %631 = fsub <4 x float> %430, %628, !dbg !21
  %632 = fsub <4 x float> %442, %628, !dbg !21
  %633 = fsub <4 x float> %454, %628, !dbg !21
  %634 = fsub <4 x float> %466, %628, !dbg !21
  %635 = fsub <4 x float> %478, %628, !dbg !21
  %636 = fsub <4 x float> %490, %628, !dbg !21
  %637 = shufflevector <1 x float> %627, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %638 = fsub <4 x float> %502, %637, !dbg !21
  %639 = fsub <4 x float> %510, %637, !dbg !21
  %640 = fsub <4 x float> %518, %637, !dbg !21
  %641 = fsub <4 x float> %526, %637, !dbg !21
  %642 = fsub <4 x float> %534, %637, !dbg !21
  %643 = fsub <4 x float> %542, %637, !dbg !21
  %644 = fsub <4 x float> %550, %637, !dbg !21
  %645 = fsub <4 x float> %557, %637, !dbg !21
  %646 = fsub <1 x float> %582, %584, !dbg !22
  %647 = extractelement <1 x float> %646, i64 0, !dbg !23
  %648 = call float @llvm.amdgcn.exp2.f32(float %647), !dbg !23
  %649 = insertelement <4 x float> poison, float %648, i64 0, !dbg !28
  %650 = shufflevector <4 x float> %649, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %651 = extractvalue [2 x <1 x float>] %130, 0, !dbg !24
  %652 = extractelement <1 x float> %651, i64 0, !dbg !24
  %653 = fmul float %652, %648, !dbg !24
  %654 = fsub <1 x float> %625, %627, !dbg !22
  %655 = extractelement <1 x float> %654, i64 0, !dbg !23
  %656 = call float @llvm.amdgcn.exp2.f32(float %655), !dbg !23
  %657 = insertelement <4 x float> poison, float %656, i64 0, !dbg !28
  %658 = shufflevector <4 x float> %657, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %659 = extractvalue [2 x <1 x float>] %130, 1, !dbg !24
  %660 = extractelement <1 x float> %659, i64 0, !dbg !24
  %661 = fmul float %660, %656, !dbg !24
  %662 = extractelement <4 x float> %629, i64 0, !dbg !25
  %663 = call float @llvm.amdgcn.exp2.f32(float %662), !dbg !25
  %664 = insertelement <4 x float> poison, float %663, i64 0, !dbg !27
  %665 = extractelement <4 x float> %629, i64 1, !dbg !25
  %666 = call float @llvm.amdgcn.exp2.f32(float %665), !dbg !25
  %667 = insertelement <4 x float> %664, float %666, i64 1, !dbg !27
  %668 = fadd float %663, %666, !dbg !26
  %669 = extractelement <4 x float> %629, i64 2, !dbg !25
  %670 = call float @llvm.amdgcn.exp2.f32(float %669), !dbg !25
  %671 = insertelement <4 x float> %667, float %670, i64 2, !dbg !27
  %672 = fadd float %668, %670, !dbg !26
  %673 = extractelement <4 x float> %629, i64 3, !dbg !25
  %674 = call float @llvm.amdgcn.exp2.f32(float %673), !dbg !25
  %675 = insertelement <4 x float> %671, float %674, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %675) #9, !dbg !27, !frisk.register_lifetimes !1
  %676 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %676) #9, !dbg !26, !frisk.register_lifetimes !1
  %677 = fadd float %672, %674, !dbg !26
  %678 = extractelement <4 x float> %630, i64 0, !dbg !25
  %679 = call float @llvm.amdgcn.exp2.f32(float %678), !dbg !25
  %680 = insertelement <4 x float> poison, float %679, i64 0, !dbg !27
  %681 = fadd float %677, %679, !dbg !26
  %682 = extractelement <4 x float> %630, i64 1, !dbg !25
  %683 = call float @llvm.amdgcn.exp2.f32(float %682), !dbg !25
  %684 = insertelement <4 x float> %680, float %683, i64 1, !dbg !27
  %685 = fadd float %681, %683, !dbg !26
  %686 = extractelement <4 x float> %630, i64 2, !dbg !25
  %687 = call float @llvm.amdgcn.exp2.f32(float %686), !dbg !25
  %688 = insertelement <4 x float> %684, float %687, i64 2, !dbg !27
  %689 = fadd float %685, %687, !dbg !26
  %690 = extractelement <4 x float> %630, i64 3, !dbg !25
  %691 = call float @llvm.amdgcn.exp2.f32(float %690), !dbg !25
  %692 = insertelement <4 x float> %688, float %691, i64 3, !dbg !27
  %math.result241 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %692) #9, !dbg !27, !frisk.register_lifetimes !1
  %693 = fptrunc <4 x float> %math.result241 to <4 x half>, !dbg !27
  %math.packed242 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %693) #9, !dbg !26, !frisk.register_lifetimes !1
  %694 = fadd float %689, %691, !dbg !26
  %695 = extractelement <4 x float> %631, i64 0, !dbg !25
  %696 = call float @llvm.amdgcn.exp2.f32(float %695), !dbg !25
  %697 = insertelement <4 x float> poison, float %696, i64 0, !dbg !27
  %698 = fadd float %694, %696, !dbg !26
  %699 = extractelement <4 x float> %631, i64 1, !dbg !25
  %700 = call float @llvm.amdgcn.exp2.f32(float %699), !dbg !25
  %701 = insertelement <4 x float> %697, float %700, i64 1, !dbg !27
  %702 = fadd float %698, %700, !dbg !26
  %703 = extractelement <4 x float> %631, i64 2, !dbg !25
  %704 = call float @llvm.amdgcn.exp2.f32(float %703), !dbg !25
  %705 = insertelement <4 x float> %701, float %704, i64 2, !dbg !27
  %706 = fadd float %702, %704, !dbg !26
  %707 = extractelement <4 x float> %631, i64 3, !dbg !25
  %708 = call float @llvm.amdgcn.exp2.f32(float %707), !dbg !25
  %709 = insertelement <4 x float> %705, float %708, i64 3, !dbg !27
  %math.result243 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %709) #9, !dbg !27, !frisk.register_lifetimes !1
  %710 = fptrunc <4 x float> %math.result243 to <4 x half>, !dbg !27
  %math.packed244 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %710) #9, !dbg !26, !frisk.register_lifetimes !1
  %711 = fadd float %706, %708, !dbg !26
  %712 = extractelement <4 x float> %632, i64 0, !dbg !25
  %713 = call float @llvm.amdgcn.exp2.f32(float %712), !dbg !25
  %714 = insertelement <4 x float> poison, float %713, i64 0, !dbg !27
  %715 = fadd float %711, %713, !dbg !26
  %716 = extractelement <4 x float> %632, i64 1, !dbg !25
  %717 = call float @llvm.amdgcn.exp2.f32(float %716), !dbg !25
  %718 = insertelement <4 x float> %714, float %717, i64 1, !dbg !27
  %719 = fadd float %715, %717, !dbg !26
  %720 = extractelement <4 x float> %632, i64 2, !dbg !25
  %721 = call float @llvm.amdgcn.exp2.f32(float %720), !dbg !25
  %722 = insertelement <4 x float> %718, float %721, i64 2, !dbg !27
  %723 = fadd float %719, %721, !dbg !26
  %724 = extractelement <4 x float> %632, i64 3, !dbg !25
  %725 = call float @llvm.amdgcn.exp2.f32(float %724), !dbg !25
  %726 = insertelement <4 x float> %722, float %725, i64 3, !dbg !27
  %math.result245 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %726) #9, !dbg !27, !frisk.register_lifetimes !1
  %727 = fptrunc <4 x float> %math.result245 to <4 x half>, !dbg !27
  %math.packed246 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %727) #9, !dbg !26, !frisk.register_lifetimes !1
  %728 = fadd float %723, %725, !dbg !26
  %729 = extractelement <4 x float> %633, i64 0, !dbg !25
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !25
  %731 = insertelement <4 x float> poison, float %730, i64 0, !dbg !27
  %732 = fadd float %728, %730, !dbg !26
  %733 = extractelement <4 x float> %633, i64 1, !dbg !25
  %734 = call float @llvm.amdgcn.exp2.f32(float %733), !dbg !25
  %735 = insertelement <4 x float> %731, float %734, i64 1, !dbg !27
  %736 = fadd float %732, %734, !dbg !26
  %737 = extractelement <4 x float> %633, i64 2, !dbg !25
  %738 = call float @llvm.amdgcn.exp2.f32(float %737), !dbg !25
  %739 = insertelement <4 x float> %735, float %738, i64 2, !dbg !27
  %740 = fadd float %736, %738, !dbg !26
  %741 = extractelement <4 x float> %633, i64 3, !dbg !25
  %742 = call float @llvm.amdgcn.exp2.f32(float %741), !dbg !25
  %743 = insertelement <4 x float> %739, float %742, i64 3, !dbg !27
  %math.result247 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %743) #9, !dbg !27, !frisk.register_lifetimes !1
  %744 = fptrunc <4 x float> %math.result247 to <4 x half>, !dbg !27
  %math.packed248 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %744) #9, !dbg !26, !frisk.register_lifetimes !1
  %745 = fadd float %740, %742, !dbg !26
  %746 = extractelement <4 x float> %634, i64 0, !dbg !25
  %747 = call float @llvm.amdgcn.exp2.f32(float %746), !dbg !25
  %748 = insertelement <4 x float> poison, float %747, i64 0, !dbg !27
  %749 = fadd float %745, %747, !dbg !26
  %750 = extractelement <4 x float> %634, i64 1, !dbg !25
  %751 = call float @llvm.amdgcn.exp2.f32(float %750), !dbg !25
  %752 = insertelement <4 x float> %748, float %751, i64 1, !dbg !27
  %753 = fadd float %749, %751, !dbg !26
  %754 = extractelement <4 x float> %634, i64 2, !dbg !25
  %755 = call float @llvm.amdgcn.exp2.f32(float %754), !dbg !25
  %756 = insertelement <4 x float> %752, float %755, i64 2, !dbg !27
  %757 = fadd float %753, %755, !dbg !26
  %758 = extractelement <4 x float> %634, i64 3, !dbg !25
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !25
  %760 = insertelement <4 x float> %756, float %759, i64 3, !dbg !27
  %math.result249 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %760) #9, !dbg !27, !frisk.register_lifetimes !1
  %761 = fptrunc <4 x float> %math.result249 to <4 x half>, !dbg !27
  %math.packed250 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %761) #9, !dbg !26, !frisk.register_lifetimes !1
  %762 = fadd float %757, %759, !dbg !26
  %763 = extractelement <4 x float> %635, i64 0, !dbg !25
  %764 = call float @llvm.amdgcn.exp2.f32(float %763), !dbg !25
  %765 = insertelement <4 x float> poison, float %764, i64 0, !dbg !27
  %766 = fadd float %762, %764, !dbg !26
  %767 = extractelement <4 x float> %635, i64 1, !dbg !25
  %768 = call float @llvm.amdgcn.exp2.f32(float %767), !dbg !25
  %769 = insertelement <4 x float> %765, float %768, i64 1, !dbg !27
  %770 = fadd float %766, %768, !dbg !26
  %771 = extractelement <4 x float> %635, i64 2, !dbg !25
  %772 = call float @llvm.amdgcn.exp2.f32(float %771), !dbg !25
  %773 = insertelement <4 x float> %769, float %772, i64 2, !dbg !27
  %774 = fadd float %770, %772, !dbg !26
  %775 = extractelement <4 x float> %635, i64 3, !dbg !25
  %776 = call float @llvm.amdgcn.exp2.f32(float %775), !dbg !25
  %777 = insertelement <4 x float> %773, float %776, i64 3, !dbg !27
  %math.result251 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %777) #9, !dbg !27, !frisk.register_lifetimes !1
  %778 = fptrunc <4 x float> %math.result251 to <4 x half>, !dbg !27
  %math.packed252 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %778) #9, !dbg !26, !frisk.register_lifetimes !1
  %779 = fadd float %774, %776, !dbg !26
  %780 = extractelement <4 x float> %636, i64 0, !dbg !25
  %781 = call float @llvm.amdgcn.exp2.f32(float %780), !dbg !25
  %782 = insertelement <4 x float> poison, float %781, i64 0, !dbg !27
  %783 = fadd float %779, %781, !dbg !26
  %784 = extractelement <4 x float> %636, i64 1, !dbg !25
  %785 = call float @llvm.amdgcn.exp2.f32(float %784), !dbg !25
  %786 = insertelement <4 x float> %782, float %785, i64 1, !dbg !27
  %787 = fadd float %783, %785, !dbg !26
  %788 = extractelement <4 x float> %636, i64 2, !dbg !25
  %789 = call float @llvm.amdgcn.exp2.f32(float %788), !dbg !25
  %790 = insertelement <4 x float> %786, float %789, i64 2, !dbg !27
  %791 = fadd float %787, %789, !dbg !26
  %792 = extractelement <4 x float> %636, i64 3, !dbg !25
  %793 = call float @llvm.amdgcn.exp2.f32(float %792), !dbg !25
  %794 = insertelement <4 x float> %790, float %793, i64 3, !dbg !27
  %math.result253 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %794) #9, !dbg !27, !frisk.register_lifetimes !1
  %795 = fptrunc <4 x float> %math.result253 to <4 x half>, !dbg !27
  %math.packed254 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %795) #9, !dbg !26, !frisk.register_lifetimes !1
  %796 = fadd float %791, %793, !dbg !26
  %797 = extractelement <4 x float> %638, i64 0, !dbg !25
  %798 = call float @llvm.amdgcn.exp2.f32(float %797), !dbg !25
  %799 = insertelement <4 x float> poison, float %798, i64 0, !dbg !27
  %800 = extractelement <4 x float> %638, i64 1, !dbg !25
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !25
  %802 = insertelement <4 x float> %799, float %801, i64 1, !dbg !27
  %803 = fadd float %798, %801, !dbg !26
  %804 = extractelement <4 x float> %638, i64 2, !dbg !25
  %805 = call float @llvm.amdgcn.exp2.f32(float %804), !dbg !25
  %806 = insertelement <4 x float> %802, float %805, i64 2, !dbg !27
  %807 = fadd float %803, %805, !dbg !26
  %808 = extractelement <4 x float> %638, i64 3, !dbg !25
  %809 = call float @llvm.amdgcn.exp2.f32(float %808), !dbg !25
  %810 = insertelement <4 x float> %806, float %809, i64 3, !dbg !27
  %math.result255 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %810) #9, !dbg !27, !frisk.register_lifetimes !1
  %811 = fptrunc <4 x float> %math.result255 to <4 x half>, !dbg !27
  %math.packed256 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %811) #9, !dbg !26, !frisk.register_lifetimes !1
  %812 = fadd float %807, %809, !dbg !26
  %813 = extractelement <4 x float> %639, i64 0, !dbg !25
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !25
  %815 = insertelement <4 x float> poison, float %814, i64 0, !dbg !27
  %816 = fadd float %812, %814, !dbg !26
  %817 = extractelement <4 x float> %639, i64 1, !dbg !25
  %818 = call float @llvm.amdgcn.exp2.f32(float %817), !dbg !25
  %819 = insertelement <4 x float> %815, float %818, i64 1, !dbg !27
  %820 = fadd float %816, %818, !dbg !26
  %821 = extractelement <4 x float> %639, i64 2, !dbg !25
  %822 = call float @llvm.amdgcn.exp2.f32(float %821), !dbg !25
  %823 = insertelement <4 x float> %819, float %822, i64 2, !dbg !27
  %824 = fadd float %820, %822, !dbg !26
  %825 = extractelement <4 x float> %639, i64 3, !dbg !25
  %826 = call float @llvm.amdgcn.exp2.f32(float %825), !dbg !25
  %827 = insertelement <4 x float> %823, float %826, i64 3, !dbg !27
  %math.result257 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %827) #9, !dbg !27, !frisk.register_lifetimes !1
  %828 = fptrunc <4 x float> %math.result257 to <4 x half>, !dbg !27
  %math.packed258 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %828) #9, !dbg !26, !frisk.register_lifetimes !1
  %829 = fadd float %824, %826, !dbg !26
  %830 = extractelement <4 x float> %640, i64 0, !dbg !25
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !25
  %832 = insertelement <4 x float> poison, float %831, i64 0, !dbg !27
  %833 = fadd float %829, %831, !dbg !26
  %834 = extractelement <4 x float> %640, i64 1, !dbg !25
  %835 = call float @llvm.amdgcn.exp2.f32(float %834), !dbg !25
  %836 = insertelement <4 x float> %832, float %835, i64 1, !dbg !27
  %837 = fadd float %833, %835, !dbg !26
  %838 = extractelement <4 x float> %640, i64 2, !dbg !25
  %839 = call float @llvm.amdgcn.exp2.f32(float %838), !dbg !25
  %840 = insertelement <4 x float> %836, float %839, i64 2, !dbg !27
  %841 = fadd float %837, %839, !dbg !26
  %842 = extractelement <4 x float> %640, i64 3, !dbg !25
  %843 = call float @llvm.amdgcn.exp2.f32(float %842), !dbg !25
  %844 = insertelement <4 x float> %840, float %843, i64 3, !dbg !27
  %math.result259 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %844) #9, !dbg !27, !frisk.register_lifetimes !1
  %845 = fptrunc <4 x float> %math.result259 to <4 x half>, !dbg !27
  %math.packed260 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %845) #9, !dbg !26, !frisk.register_lifetimes !1
  %846 = fadd float %841, %843, !dbg !26
  %847 = extractelement <4 x float> %641, i64 0, !dbg !25
  %848 = call float @llvm.amdgcn.exp2.f32(float %847), !dbg !25
  %849 = insertelement <4 x float> poison, float %848, i64 0, !dbg !27
  %850 = fadd float %846, %848, !dbg !26
  %851 = extractelement <4 x float> %641, i64 1, !dbg !25
  %852 = call float @llvm.amdgcn.exp2.f32(float %851), !dbg !25
  %853 = insertelement <4 x float> %849, float %852, i64 1, !dbg !27
  %854 = fadd float %850, %852, !dbg !26
  %855 = extractelement <4 x float> %641, i64 2, !dbg !25
  %856 = call float @llvm.amdgcn.exp2.f32(float %855), !dbg !25
  %857 = insertelement <4 x float> %853, float %856, i64 2, !dbg !27
  %858 = fadd float %854, %856, !dbg !26
  %859 = extractelement <4 x float> %641, i64 3, !dbg !25
  %860 = call float @llvm.amdgcn.exp2.f32(float %859), !dbg !25
  %861 = insertelement <4 x float> %857, float %860, i64 3, !dbg !27
  %math.result261 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %861) #9, !dbg !27, !frisk.register_lifetimes !1
  %862 = fptrunc <4 x float> %math.result261 to <4 x half>, !dbg !27
  %math.packed262 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %862) #9, !dbg !26, !frisk.register_lifetimes !1
  %863 = fadd float %858, %860, !dbg !26
  %864 = extractelement <4 x float> %642, i64 0, !dbg !25
  %865 = call float @llvm.amdgcn.exp2.f32(float %864), !dbg !25
  %866 = insertelement <4 x float> poison, float %865, i64 0, !dbg !27
  %867 = fadd float %863, %865, !dbg !26
  %868 = extractelement <4 x float> %642, i64 1, !dbg !25
  %869 = call float @llvm.amdgcn.exp2.f32(float %868), !dbg !25
  %870 = insertelement <4 x float> %866, float %869, i64 1, !dbg !27
  %871 = fadd float %867, %869, !dbg !26
  %872 = extractelement <4 x float> %642, i64 2, !dbg !25
  %873 = call float @llvm.amdgcn.exp2.f32(float %872), !dbg !25
  %874 = insertelement <4 x float> %870, float %873, i64 2, !dbg !27
  %875 = fadd float %871, %873, !dbg !26
  %876 = extractelement <4 x float> %642, i64 3, !dbg !25
  %877 = call float @llvm.amdgcn.exp2.f32(float %876), !dbg !25
  %878 = insertelement <4 x float> %874, float %877, i64 3, !dbg !27
  %math.result263 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %878) #9, !dbg !27, !frisk.register_lifetimes !1
  %879 = fptrunc <4 x float> %math.result263 to <4 x half>, !dbg !27
  %math.packed264 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %879) #9, !dbg !26, !frisk.register_lifetimes !1
  %880 = fadd float %875, %877, !dbg !26
  %881 = extractelement <4 x float> %643, i64 0, !dbg !25
  %882 = call float @llvm.amdgcn.exp2.f32(float %881), !dbg !25
  %883 = insertelement <4 x float> poison, float %882, i64 0, !dbg !27
  %884 = fadd float %880, %882, !dbg !26
  %885 = extractelement <4 x float> %643, i64 1, !dbg !25
  %886 = call float @llvm.amdgcn.exp2.f32(float %885), !dbg !25
  %887 = insertelement <4 x float> %883, float %886, i64 1, !dbg !27
  %888 = fadd float %884, %886, !dbg !26
  %889 = extractelement <4 x float> %643, i64 2, !dbg !25
  %890 = call float @llvm.amdgcn.exp2.f32(float %889), !dbg !25
  %891 = insertelement <4 x float> %887, float %890, i64 2, !dbg !27
  %892 = fadd float %888, %890, !dbg !26
  %893 = extractelement <4 x float> %643, i64 3, !dbg !25
  %894 = call float @llvm.amdgcn.exp2.f32(float %893), !dbg !25
  %895 = insertelement <4 x float> %891, float %894, i64 3, !dbg !27
  %math.result265 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %895) #9, !dbg !27, !frisk.register_lifetimes !1
  %896 = fptrunc <4 x float> %math.result265 to <4 x half>, !dbg !27
  %math.packed266 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %896) #9, !dbg !26, !frisk.register_lifetimes !1
  %897 = fadd float %892, %894, !dbg !26
  %898 = extractelement <4 x float> %644, i64 0, !dbg !25
  %899 = call float @llvm.amdgcn.exp2.f32(float %898), !dbg !25
  %900 = insertelement <4 x float> poison, float %899, i64 0, !dbg !27
  %901 = fadd float %897, %899, !dbg !26
  %902 = extractelement <4 x float> %644, i64 1, !dbg !25
  %903 = call float @llvm.amdgcn.exp2.f32(float %902), !dbg !25
  %904 = insertelement <4 x float> %900, float %903, i64 1, !dbg !27
  %905 = fadd float %901, %903, !dbg !26
  %906 = extractelement <4 x float> %644, i64 2, !dbg !25
  %907 = call float @llvm.amdgcn.exp2.f32(float %906), !dbg !25
  %908 = insertelement <4 x float> %904, float %907, i64 2, !dbg !27
  %909 = fadd float %905, %907, !dbg !26
  %910 = extractelement <4 x float> %644, i64 3, !dbg !25
  %911 = call float @llvm.amdgcn.exp2.f32(float %910), !dbg !25
  %912 = insertelement <4 x float> %908, float %911, i64 3, !dbg !27
  %math.result267 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %912) #9, !dbg !27, !frisk.register_lifetimes !1
  %913 = fptrunc <4 x float> %math.result267 to <4 x half>, !dbg !27
  %math.packed268 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %913) #9, !dbg !26, !frisk.register_lifetimes !1
  %914 = fadd float %909, %911, !dbg !26
  %915 = extractelement <4 x float> %645, i64 0, !dbg !25
  %916 = call float @llvm.amdgcn.exp2.f32(float %915), !dbg !25
  %917 = insertelement <4 x float> poison, float %916, i64 0, !dbg !27
  %918 = fadd float %914, %916, !dbg !26
  %919 = extractelement <4 x float> %645, i64 1, !dbg !25
  %920 = call float @llvm.amdgcn.exp2.f32(float %919), !dbg !25
  %921 = insertelement <4 x float> %917, float %920, i64 1, !dbg !27
  %922 = fadd float %918, %920, !dbg !26
  %923 = extractelement <4 x float> %645, i64 2, !dbg !25
  %924 = call float @llvm.amdgcn.exp2.f32(float %923), !dbg !25
  %925 = insertelement <4 x float> %921, float %924, i64 2, !dbg !27
  %926 = fadd float %922, %924, !dbg !26
  %927 = extractelement <4 x float> %645, i64 3, !dbg !25
  %928 = call float @llvm.amdgcn.exp2.f32(float %927), !dbg !25
  %929 = insertelement <4 x float> %925, float %928, i64 3, !dbg !27
  %math.result269 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %929) #9, !dbg !27, !frisk.register_lifetimes !1
  %930 = fptrunc <4 x float> %math.result269 to <4 x half>, !dbg !27
  %math.packed270 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %930) #9, !dbg !26, !frisk.register_lifetimes !1
  %931 = fadd float %926, %928, !dbg !26
  %932 = bitcast float %796 to i32, !dbg !26
  %933 = call i32 @llvm.amdgcn.ds.bpermute(i32 %569, i32 %932), !dbg !26
  %934 = bitcast i32 %933 to float, !dbg !26
  %935 = fadd float %796, %934, !dbg !26
  %936 = bitcast float %935 to i32, !dbg !26
  %937 = call i32 @llvm.amdgcn.ds.bpermute(i32 %577, i32 %936), !dbg !26
  %938 = bitcast i32 %937 to float, !dbg !26
  %939 = fadd float %935, %938, !dbg !26
  %940 = fadd float %939, %653, !dbg !26
  %941 = insertelement <1 x float> poison, float %940, i64 0, !dbg !7
  %942 = insertvalue [2 x <1 x float>] poison, <1 x float> %941, 0, !dbg !7
  %943 = bitcast float %931 to i32, !dbg !26
  %944 = call i32 @llvm.amdgcn.ds.bpermute(i32 %569, i32 %943), !dbg !26
  %945 = bitcast i32 %944 to float, !dbg !26
  %946 = fadd float %931, %945, !dbg !26
  %947 = bitcast float %946 to i32, !dbg !26
  %948 = call i32 @llvm.amdgcn.ds.bpermute(i32 %577, i32 %947), !dbg !26
  %949 = bitcast i32 %948 to float, !dbg !26
  %950 = fadd float %946, %949, !dbg !26
  %951 = fadd float %950, %661, !dbg !26
  %952 = insertelement <1 x float> poison, float %951, i64 0, !dbg !7
  %953 = insertvalue [2 x <1 x float>] %942, <1 x float> %952, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !15
  %vdb.address1302 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address), !dbg !15
  %vdb.address1303 = or disjoint i64 %153, %address.local788, !dbg !15
  %vdb.address1304 = trunc nuw i64 %vdb.address1303 to i32, !dbg !15
  %vdb.address1305 = shl nuw nsw i32 %vdb.address1304, 1, !dbg !15
  %vdb.address1306 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1307 = insertelement <2 x i64> poison, i64 %vdb.address1306, i64 0, !dbg !15
  %vdb.address1308 = bitcast <2 x i64> %vdb.address1307 to <4 x i32>, !dbg !15
  %vdb.address1309 = shufflevector <4 x i32> %vdb.address1308, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1310 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !15
  %vdb.address1311 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1310), !dbg !15
  %vdb.address1312 = or disjoint i64 %153, %address.local794, !dbg !15
  %vdb.address1313 = trunc nuw i64 %vdb.address1312 to i32, !dbg !15
  %vdb.address1314 = shl nuw nsw i32 %vdb.address1313, 1, !dbg !15
  %vdb.address1315 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1316 = insertelement <2 x i64> poison, i64 %vdb.address1315, i64 0, !dbg !15
  %vdb.address1317 = bitcast <2 x i64> %vdb.address1316 to <4 x i32>, !dbg !15
  %vdb.address1318 = shufflevector <4 x i32> %vdb.address1317, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.dma.base = add i32 %vdb.address1302, 8192, !dbg !13
  %954 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1305, <4 x i32> %vdb.address1309) #10, !dbg !15
  %vdb.dma.base1319 = add i32 %vdb.address1311, 8192, !dbg !13
  %955 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1319, i32 %vdb.address1314, <4 x i32> %vdb.address1318) #10, !dbg !15
  %address.local796 = shl nuw nsw i32 %address.thread365, 3, !dbg !13
  %address.local797 = and i32 %address.local796, 120, !dbg !13
  %address.local798 = shl nuw nsw i32 %address.thread365, 5, !dbg !13
  %address.local799 = and i32 %address.local798, 1536, !dbg !13
  %address.local800 = or disjoint i32 %address.local797, %address.local799, !dbg !13
  %address.local801 = zext nneg i32 %address.local800 to i64, !dbg !13
  %address.local802 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local801, !dbg !13
  %956 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local802, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %957 = bitcast <4 x i32> %956 to <8 x half>, !dbg !13
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
  %958 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local812, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %959 = bitcast <4 x i32> %958 to <8 x half>, !dbg !13
  %address.local813 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local814 = getelementptr [2 x i8], ptr addrspace(3) %address.local813, i64 %address.local808, !dbg !13
  %address.local815 = getelementptr [2 x i8], ptr addrspace(3) %address.local814, i64 %address.local810, !dbg !13
  %address.local816 = getelementptr i8, ptr addrspace(3) %address.local815, i64 512, !dbg !13
  %960 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %961 = bitcast <4 x i32> %960 to <8 x half>, !dbg !13
  %address.local817 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local818 = getelementptr [2 x i8], ptr addrspace(3) %address.local817, i64 %address.local808, !dbg !13
  %address.local819 = getelementptr [2 x i8], ptr addrspace(3) %address.local818, i64 %address.local810, !dbg !13
  %address.local820 = getelementptr i8, ptr addrspace(3) %address.local819, i64 768, !dbg !13
  %962 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %963 = bitcast <4 x i32> %962 to <8 x half>, !dbg !13
  %address.local821 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local801, !dbg !13
  %address.local822 = getelementptr i8, ptr addrspace(3) %address.local821, i64 4096, !dbg !13
  %964 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %965 = bitcast <4 x i32> %964 to <8 x half>, !dbg !13
  %address.local823 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local824 = getelementptr [2 x i8], ptr addrspace(3) %address.local823, i64 %address.local808, !dbg !13
  %address.local825 = getelementptr [2 x i8], ptr addrspace(3) %address.local824, i64 %address.local810, !dbg !13
  %address.local826 = getelementptr i8, ptr addrspace(3) %address.local825, i64 4352, !dbg !13
  %966 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %967 = bitcast <4 x i32> %966 to <8 x half>, !dbg !13
  %address.local827 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local828 = getelementptr [2 x i8], ptr addrspace(3) %address.local827, i64 %address.local808, !dbg !13
  %address.local829 = getelementptr [2 x i8], ptr addrspace(3) %address.local828, i64 %address.local810, !dbg !13
  %address.local830 = getelementptr i8, ptr addrspace(3) %address.local829, i64 4608, !dbg !13
  %968 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %969 = bitcast <4 x i32> %968 to <8 x half>, !dbg !13
  %address.local831 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local832 = getelementptr [2 x i8], ptr addrspace(3) %address.local831, i64 %address.local808, !dbg !13
  %address.local833 = getelementptr [2 x i8], ptr addrspace(3) %address.local832, i64 %address.local810, !dbg !13
  %address.local834 = getelementptr i8, ptr addrspace(3) %address.local833, i64 4864, !dbg !13
  %970 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local834, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %971 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %972 = extractvalue [2 x <8 x float>] %129, 0, !dbg !28
  %973 = shufflevector <8 x float> %972, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %974 = fmul <4 x float> %973, %650, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %971, <4 x float> %974) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %976 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %977 = shufflevector <8 x float> %972, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %978 = fmul <4 x float> %977, %650, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %976, <4 x float> %978) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %980 = extractvalue [2 x <8 x float>] %129, 1, !dbg !28
  %981 = shufflevector <8 x float> %980, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %982 = fmul <4 x float> %981, %658, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %971, <4 x float> %982) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %984 = shufflevector <8 x float> %980, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %985 = fmul <4 x float> %984, %658, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %976, <4 x float> %985) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %987 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %987, <4 x float> %975) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %989 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %989, <4 x float> %979) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %987, <4 x float> %983) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %989, <4 x float> %986) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %993 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %993, <4 x float> %988) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %995 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %995, <4 x float> %990) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %993, <4 x float> %991) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %995, <4 x float> %992) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %999 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1000 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %999, <4 x float> %994) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1001 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1001, <4 x float> %996) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %999, <4 x float> %997) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1001, <4 x float> %998) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1005 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1006 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1005, <4 x float> %1000) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1007 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1007, <4 x float> %1002) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1009 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1005, <4 x float> %1003) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1007, <4 x float> %1004) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1011 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1011, <4 x float> %1006) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1013 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1013, <4 x float> %1008) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1011, <4 x float> %1009) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1013, <4 x float> %1010) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1017 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1017, <4 x float> %1012) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1019 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1019, <4 x float> %1014) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1017, <4 x float> %1015) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1019, <4 x float> %1016) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1023 = bitcast <4 x i32> %970 to <8 x half>, !dbg !13
  %1024 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1024, <4 x float> %1018) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1026 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1026, <4 x float> %1020) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1028 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1024, <4 x float> %1021) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1026, <4 x float> %1022) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1327 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !15
  %vdb.address1328 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1327), !dbg !15
  %vdb.address1329 = or disjoint i64 %238, %address.local788, !dbg !15
  %vdb.address1330 = trunc nuw i64 %vdb.address1329 to i32, !dbg !15
  %vdb.address1331 = shl nuw nsw i32 %vdb.address1330, 1, !dbg !15
  %vdb.address1332 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1333 = insertelement <2 x i64> poison, i64 %vdb.address1332, i64 0, !dbg !15
  %vdb.address1334 = bitcast <2 x i64> %vdb.address1333 to <4 x i32>, !dbg !15
  %vdb.address1335 = shufflevector <4 x i32> %vdb.address1334, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1336 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !15
  %vdb.address1337 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1336), !dbg !15
  %vdb.address1338 = or disjoint i64 %238, %address.local794, !dbg !15
  %vdb.address1339 = trunc nuw i64 %vdb.address1338 to i32, !dbg !15
  %vdb.address1340 = shl nuw nsw i32 %vdb.address1339, 1, !dbg !15
  %vdb.address1341 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1342 = insertelement <2 x i64> poison, i64 %vdb.address1341, i64 0, !dbg !15
  %vdb.address1343 = bitcast <2 x i64> %vdb.address1342 to <4 x i32>, !dbg !15
  %vdb.address1344 = shufflevector <4 x i32> %vdb.address1343, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1030 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.address1328, i32 %vdb.address1331, <4 x i32> %vdb.address1335) #10, !dbg !15
  %1031 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.address1337, i32 %vdb.address1340, <4 x i32> %vdb.address1344) #10, !dbg !15
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local802, i64 8192, !dbg !13
  %1032 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1033 = bitcast <4 x i32> %1032 to <8 x half>, !dbg !13
  %vdb.read1320 = getelementptr i8, ptr addrspace(3) %address.local812, i64 8192, !dbg !13
  %1034 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1320, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1035 = bitcast <4 x i32> %1034 to <8 x half>, !dbg !13
  %vdb.read1321 = getelementptr i8, ptr addrspace(3) %address.local816, i64 8192, !dbg !13
  %1036 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1321, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1037 = bitcast <4 x i32> %1036 to <8 x half>, !dbg !13
  %vdb.read1322 = getelementptr i8, ptr addrspace(3) %address.local820, i64 8192, !dbg !13
  %1038 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1322, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1039 = bitcast <4 x i32> %1038 to <8 x half>, !dbg !13
  %vdb.read1323 = getelementptr i8, ptr addrspace(3) %address.local822, i64 8192, !dbg !13
  %1040 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1323, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1041 = bitcast <4 x i32> %1040 to <8 x half>, !dbg !13
  %vdb.read1324 = getelementptr i8, ptr addrspace(3) %address.local826, i64 8192, !dbg !13
  %1042 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1324, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1043 = bitcast <4 x i32> %1042 to <8 x half>, !dbg !13
  %vdb.read1325 = getelementptr i8, ptr addrspace(3) %address.local830, i64 8192, !dbg !13
  %1044 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1325, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1045 = bitcast <4 x i32> %1044 to <8 x half>, !dbg !13
  %vdb.read1326 = getelementptr i8, ptr addrspace(3) %address.local834, i64 8192, !dbg !13
  %1046 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1326, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1047 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1048 = extractvalue [2 x <8 x float>] %132, 0, !dbg !28
  %1049 = shufflevector <8 x float> %1048, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1050 = fmul <4 x float> %1049, %650, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1047, <4 x float> %1050) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1052 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1053 = shufflevector <8 x float> %1048, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1054 = fmul <4 x float> %1053, %650, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1052, <4 x float> %1054) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1056 = extractvalue [2 x <8 x float>] %132, 1, !dbg !28
  %1057 = shufflevector <8 x float> %1056, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1058 = fmul <4 x float> %1057, %658, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1047, <4 x float> %1058) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1060 = shufflevector <8 x float> %1056, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1061 = fmul <4 x float> %1060, %658, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1052, <4 x float> %1061) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1063 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1063, <4 x float> %1051) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1065 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1065, <4 x float> %1055) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1063, <4 x float> %1059) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1065, <4 x float> %1062) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1069 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1069, <4 x float> %1064) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1071 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1071, <4 x float> %1066) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1069, <4 x float> %1067) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1071, <4 x float> %1068) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1075 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1075, <4 x float> %1070) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1077 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1077, <4 x float> %1072) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1075, <4 x float> %1073) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1077, <4 x float> %1074) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1081 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1081, <4 x float> %1076) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1083 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1083, <4 x float> %1078) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1085 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1081, <4 x float> %1079) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1083, <4 x float> %1080) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1087 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1087, <4 x float> %1082) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1089 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1089, <4 x float> %1084) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1091 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1087, <4 x float> %1085) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1089, <4 x float> %1086) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1093 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1093, <4 x float> %1088) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1095 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1095, <4 x float> %1090) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1097 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1093, <4 x float> %1091) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1095, <4 x float> %1092) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1099 = bitcast <4 x i32> %1046 to <8 x half>, !dbg !13
  %1100 = shufflevector <8 x half> %1099, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1101 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1100, <4 x float> %1094) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1102 = shufflevector <8 x half> %1099, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1102, <4 x float> %1096) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1100, <4 x float> %1097) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1102, <4 x float> %1098) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1345 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !15
  %vdb.address1346 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1345), !dbg !15
  %vdb.address1347 = or disjoint i64 %315, %address.local788, !dbg !15
  %vdb.address1348 = trunc nuw i64 %vdb.address1347 to i32, !dbg !15
  %vdb.address1349 = shl nuw nsw i32 %vdb.address1348, 1, !dbg !15
  %vdb.address1350 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1351 = insertelement <2 x i64> poison, i64 %vdb.address1350, i64 0, !dbg !15
  %vdb.address1352 = bitcast <2 x i64> %vdb.address1351 to <4 x i32>, !dbg !15
  %vdb.address1353 = shufflevector <4 x i32> %vdb.address1352, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1354 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !15
  %vdb.address1355 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1354), !dbg !15
  %vdb.address1356 = or disjoint i64 %315, %address.local794, !dbg !15
  %vdb.address1357 = trunc nuw i64 %vdb.address1356 to i32, !dbg !15
  %vdb.address1358 = shl nuw nsw i32 %vdb.address1357, 1, !dbg !15
  %vdb.address1359 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1360 = insertelement <2 x i64> poison, i64 %vdb.address1359, i64 0, !dbg !15
  %vdb.address1361 = bitcast <2 x i64> %vdb.address1360 to <4 x i32>, !dbg !15
  %vdb.address1362 = shufflevector <4 x i32> %vdb.address1361, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.dma.base1363 = add i32 %vdb.address1346, 8192, !dbg !13
  %1106 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1363, i32 %vdb.address1349, <4 x i32> %vdb.address1353) #10, !dbg !15
  %vdb.dma.base1364 = add i32 %vdb.address1355, 8192, !dbg !13
  %1107 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1364, i32 %vdb.address1358, <4 x i32> %vdb.address1362) #10, !dbg !15
  %1108 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local802, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1109 = bitcast <4 x i32> %1108 to <8 x half>, !dbg !13
  %1110 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local812, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1111 = bitcast <4 x i32> %1110 to <8 x half>, !dbg !13
  %1112 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1113 = bitcast <4 x i32> %1112 to <8 x half>, !dbg !13
  %1114 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1115 = bitcast <4 x i32> %1114 to <8 x half>, !dbg !13
  %1116 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1117 = bitcast <4 x i32> %1116 to <8 x half>, !dbg !13
  %1118 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1119 = bitcast <4 x i32> %1118 to <8 x half>, !dbg !13
  %1120 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1121 = bitcast <4 x i32> %1120 to <8 x half>, !dbg !13
  %1122 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local834, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1123 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1124 = extractvalue [2 x <8 x float>] %133, 0, !dbg !28
  %1125 = shufflevector <8 x float> %1124, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1126 = fmul <4 x float> %1125, %650, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1123, <4 x float> %1126) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1128 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1129 = shufflevector <8 x float> %1124, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1130 = fmul <4 x float> %1129, %650, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1128, <4 x float> %1130) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1132 = extractvalue [2 x <8 x float>] %133, 1, !dbg !28
  %1133 = shufflevector <8 x float> %1132, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1134 = fmul <4 x float> %1133, %658, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1123, <4 x float> %1134) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1136 = shufflevector <8 x float> %1132, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1137 = fmul <4 x float> %1136, %658, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1128, <4 x float> %1137) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1139 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1139, <4 x float> %1127) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1141 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1141, <4 x float> %1131) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1139, <4 x float> %1135) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1141, <4 x float> %1138) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1145 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1145, <4 x float> %1140) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1147 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1147, <4 x float> %1142) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1145, <4 x float> %1143) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1147, <4 x float> %1144) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1151 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1151, <4 x float> %1146) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1153 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1153, <4 x float> %1148) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1151, <4 x float> %1149) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1153, <4 x float> %1150) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1157 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1157, <4 x float> %1152) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1159 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1159, <4 x float> %1154) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1157, <4 x float> %1155) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1159, <4 x float> %1156) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1163 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1163, <4 x float> %1158) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1165 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1165, <4 x float> %1160) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1163, <4 x float> %1161) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1165, <4 x float> %1162) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1169 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1169, <4 x float> %1164) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1171 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1171, <4 x float> %1166) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1169, <4 x float> %1167) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1171, <4 x float> %1168) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1175 = bitcast <4 x i32> %1122 to <8 x half>, !dbg !13
  %1176 = shufflevector <8 x half> %1175, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1176, <4 x float> %1170) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1178 = shufflevector <8 x half> %1175, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1178, <4 x float> %1172) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1176, <4 x float> %1173) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1178, <4 x float> %1174) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.read1365 = getelementptr i8, ptr addrspace(3) %address.local802, i64 8192, !dbg !13
  %1182 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1365, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1183 = bitcast <4 x i32> %1182 to <8 x half>, !dbg !13
  %vdb.read1366 = getelementptr i8, ptr addrspace(3) %address.local812, i64 8192, !dbg !13
  %1184 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1366, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1185 = bitcast <4 x i32> %1184 to <8 x half>, !dbg !13
  %vdb.read1367 = getelementptr i8, ptr addrspace(3) %address.local816, i64 8192, !dbg !13
  %1186 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1367, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1187 = bitcast <4 x i32> %1186 to <8 x half>, !dbg !13
  %vdb.read1368 = getelementptr i8, ptr addrspace(3) %address.local820, i64 8192, !dbg !13
  %1188 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1368, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1189 = bitcast <4 x i32> %1188 to <8 x half>, !dbg !13
  %vdb.read1369 = getelementptr i8, ptr addrspace(3) %address.local822, i64 8192, !dbg !13
  %1190 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1369, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1191 = bitcast <4 x i32> %1190 to <8 x half>, !dbg !13
  %vdb.read1370 = getelementptr i8, ptr addrspace(3) %address.local826, i64 8192, !dbg !13
  %1192 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1370, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1193 = bitcast <4 x i32> %1192 to <8 x half>, !dbg !13
  %vdb.read1371 = getelementptr i8, ptr addrspace(3) %address.local830, i64 8192, !dbg !13
  %1194 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1371, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1195 = bitcast <4 x i32> %1194 to <8 x half>, !dbg !13
  %vdb.read1372 = getelementptr i8, ptr addrspace(3) %address.local834, i64 8192, !dbg !13
  %1196 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1372, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1197 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1198 = extractvalue [2 x <8 x float>] %134, 0, !dbg !28
  %1199 = shufflevector <8 x float> %1198, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1200 = fmul <4 x float> %1199, %650, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1197, <4 x float> %1200) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1202 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1203 = shufflevector <8 x float> %1198, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1204 = fmul <4 x float> %1203, %650, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1202, <4 x float> %1204) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1206 = extractvalue [2 x <8 x float>] %134, 1, !dbg !28
  %1207 = shufflevector <8 x float> %1206, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1208 = fmul <4 x float> %1207, %658, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1197, <4 x float> %1208) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1210 = shufflevector <8 x float> %1206, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1211 = fmul <4 x float> %1210, %658, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1202, <4 x float> %1211) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1213 = shufflevector <8 x half> %1185, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1213, <4 x float> %1201) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1215 = shufflevector <8 x half> %1185, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1215, <4 x float> %1205) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1213, <4 x float> %1209) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1215, <4 x float> %1212) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1219 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1219, <4 x float> %1214) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1221 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1221, <4 x float> %1216) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1219, <4 x float> %1217) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1221, <4 x float> %1218) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1225 = shufflevector <8 x half> %1189, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1225, <4 x float> %1220) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1227 = shufflevector <8 x half> %1189, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1227, <4 x float> %1222) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1225, <4 x float> %1223) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1227, <4 x float> %1224) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1231 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1231, <4 x float> %1226) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1233 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1233, <4 x float> %1228) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1231, <4 x float> %1229) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1233, <4 x float> %1230) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1237 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1237, <4 x float> %1232) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1239 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1239, <4 x float> %1234) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1237, <4 x float> %1235) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1239, <4 x float> %1236) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1243 = shufflevector <8 x half> %1195, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1243, <4 x float> %1238) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1245 = shufflevector <8 x half> %1195, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1245, <4 x float> %1240) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1243, <4 x float> %1241) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1245, <4 x float> %1242) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1249 = bitcast <4 x i32> %1196 to <8 x half>, !dbg !13
  %1250 = shufflevector <8 x half> %1249, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1250, <4 x float> %1244) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1252 = shufflevector <8 x half> %1249, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1252, <4 x float> %1246) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1250, <4 x float> %1247) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1252, <4 x float> %1248) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %1256 = shufflevector <4 x float> %1025, <4 x float> %1027, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1257 = insertvalue [2 x <8 x float>] poison, <8 x float> %1256, 0, !dbg !7
  %1258 = shufflevector <4 x float> %1028, <4 x float> %1029, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1259 = insertvalue [2 x <8 x float>] %1257, <8 x float> %1258, 1, !dbg !7
  %1260 = insertvalue [2 x <1 x float>] poison, <1 x float> %584, 0, !dbg !7
  %1261 = insertvalue [2 x <1 x float>] %1260, <1 x float> %627, 1, !dbg !7
  %1262 = shufflevector <4 x float> %1101, <4 x float> %1103, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1263 = insertvalue [2 x <8 x float>] poison, <8 x float> %1262, 0, !dbg !7
  %1264 = shufflevector <4 x float> %1104, <4 x float> %1105, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1265 = insertvalue [2 x <8 x float>] %1263, <8 x float> %1264, 1, !dbg !7
  %1266 = shufflevector <4 x float> %1177, <4 x float> %1179, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1267 = insertvalue [2 x <8 x float>] poison, <8 x float> %1266, 0, !dbg !7
  %1268 = shufflevector <4 x float> %1180, <4 x float> %1181, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1269 = insertvalue [2 x <8 x float>] %1267, <8 x float> %1268, 1, !dbg !7
  %1270 = shufflevector <4 x float> %1251, <4 x float> %1253, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1271 = insertvalue [2 x <8 x float>] poison, <8 x float> %1270, 0, !dbg !7
  %1272 = shufflevector <4 x float> %1254, <4 x float> %1255, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1273 = insertvalue [2 x <8 x float>] %1271, <8 x float> %1272, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1274 = add nuw nsw i64 %128, 1, !dbg !7
  br label %127, !dbg !7

1275:                                             ; preds = %127
  %1276 = and i32 %33, 16, !dbg !7
  %.not = icmp eq i32 %1276, 0, !dbg !7
  %1277 = and i32 %33, 32, !dbg !7
  %.not36 = icmp eq i32 %1277, 0, !dbg !7
  %1278 = extractvalue [2 x <8 x float>] %134, 0, !dbg !7
  %1279 = extractelement <8 x float> %1278, i64 0, !dbg !7
  %1280 = extractelement <8 x float> %1278, i64 1, !dbg !7
  %1281 = extractelement <8 x float> %1278, i64 2, !dbg !7
  %1282 = extractelement <8 x float> %1278, i64 3, !dbg !7
  %1283 = select i1 %.not, float %1280, float %1279, !dbg !7
  %1284 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1285 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1284), !dbg !7
  %1286 = add nuw nsw i32 %1285, 64, !dbg !7
  %1287 = xor i32 %1285, 16, !dbg !7
  %1288 = and i32 %1286, 192, !dbg !7
  %1289 = icmp samesign ult i32 %1287, %1288, !dbg !7
  %1290 = select i1 %1289, i32 %1287, i32 %1285, !dbg !7
  %1291 = shl nuw nsw i32 %1290, 2, !dbg !7
  %1292 = bitcast float %1283 to i32, !dbg !7
  %1293 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1292), !dbg !7
  %1294 = bitcast i32 %1293 to float, !dbg !7
  %1295 = select i1 %.not, float %1279, float %1294, !dbg !7
  %1296 = select i1 %.not, float %1294, float %1280, !dbg !7
  %1297 = select i1 %.not, float %1282, float %1281, !dbg !7
  %1298 = bitcast float %1297 to i32, !dbg !7
  %1299 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1298), !dbg !7
  %1300 = bitcast i32 %1299 to float, !dbg !7
  %1301 = select i1 %.not, float %1281, float %1300, !dbg !7
  %1302 = select i1 %.not, float %1300, float %1282, !dbg !7
  %1303 = select i1 %.not36, float %1301, float %1295, !dbg !7
  %1304 = xor i32 %1285, 32, !dbg !7
  %1305 = icmp samesign ult i32 %1304, %1288, !dbg !7
  %1306 = select i1 %1305, i32 %1304, i32 %1285, !dbg !7
  %1307 = shl nuw nsw i32 %1306, 2, !dbg !7
  %1308 = bitcast float %1303 to i32, !dbg !7
  %1309 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1308), !dbg !7
  %1310 = bitcast i32 %1309 to float, !dbg !7
  %1311 = select i1 %.not36, float %1295, float %1310, !dbg !7
  %1312 = select i1 %.not36, float %1310, float %1301, !dbg !7
  %1313 = select i1 %.not36, float %1302, float %1296, !dbg !7
  %1314 = bitcast float %1313 to i32, !dbg !7
  %1315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1314), !dbg !7
  %1316 = bitcast i32 %1315 to float, !dbg !7
  %1317 = select i1 %.not36, float %1296, float %1316, !dbg !7
  %1318 = select i1 %.not36, float %1316, float %1302, !dbg !7
  %1319 = extractvalue [2 x <8 x float>] %133, 0, !dbg !7
  %1320 = extractelement <8 x float> %1319, i64 0, !dbg !7
  %1321 = extractelement <8 x float> %1319, i64 1, !dbg !7
  %1322 = extractelement <8 x float> %1319, i64 2, !dbg !7
  %1323 = extractelement <8 x float> %1319, i64 3, !dbg !7
  %1324 = select i1 %.not, float %1321, float %1320, !dbg !7
  %1325 = bitcast float %1324 to i32, !dbg !7
  %1326 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1325), !dbg !7
  %1327 = bitcast i32 %1326 to float, !dbg !7
  %1328 = select i1 %.not, float %1320, float %1327, !dbg !7
  %1329 = select i1 %.not, float %1327, float %1321, !dbg !7
  %1330 = select i1 %.not, float %1323, float %1322, !dbg !7
  %1331 = bitcast float %1330 to i32, !dbg !7
  %1332 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1331), !dbg !7
  %1333 = bitcast i32 %1332 to float, !dbg !7
  %1334 = select i1 %.not, float %1322, float %1333, !dbg !7
  %1335 = select i1 %.not, float %1333, float %1323, !dbg !7
  %1336 = select i1 %.not36, float %1334, float %1328, !dbg !7
  %1337 = bitcast float %1336 to i32, !dbg !7
  %1338 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1337), !dbg !7
  %1339 = bitcast i32 %1338 to float, !dbg !7
  %1340 = select i1 %.not36, float %1328, float %1339, !dbg !7
  %1341 = select i1 %.not36, float %1339, float %1334, !dbg !7
  %1342 = select i1 %.not36, float %1335, float %1329, !dbg !7
  %1343 = bitcast float %1342 to i32, !dbg !7
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1343), !dbg !7
  %1345 = bitcast i32 %1344 to float, !dbg !7
  %1346 = select i1 %.not36, float %1329, float %1345, !dbg !7
  %1347 = select i1 %.not36, float %1345, float %1335, !dbg !7
  %1348 = extractvalue [2 x <8 x float>] %132, 0, !dbg !7
  %1349 = extractelement <8 x float> %1348, i64 0, !dbg !7
  %1350 = extractelement <8 x float> %1348, i64 1, !dbg !7
  %1351 = extractelement <8 x float> %1348, i64 2, !dbg !7
  %1352 = extractelement <8 x float> %1348, i64 3, !dbg !7
  %1353 = select i1 %.not, float %1350, float %1349, !dbg !7
  %1354 = bitcast float %1353 to i32, !dbg !7
  %1355 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1354), !dbg !7
  %1356 = bitcast i32 %1355 to float, !dbg !7
  %1357 = select i1 %.not, float %1349, float %1356, !dbg !7
  %1358 = select i1 %.not, float %1356, float %1350, !dbg !7
  %1359 = select i1 %.not, float %1352, float %1351, !dbg !7
  %1360 = bitcast float %1359 to i32, !dbg !7
  %1361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1360), !dbg !7
  %1362 = bitcast i32 %1361 to float, !dbg !7
  %1363 = select i1 %.not, float %1351, float %1362, !dbg !7
  %1364 = select i1 %.not, float %1362, float %1352, !dbg !7
  %1365 = select i1 %.not36, float %1363, float %1357, !dbg !7
  %1366 = bitcast float %1365 to i32, !dbg !7
  %1367 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1366), !dbg !7
  %1368 = bitcast i32 %1367 to float, !dbg !7
  %1369 = select i1 %.not36, float %1357, float %1368, !dbg !7
  %1370 = select i1 %.not36, float %1368, float %1363, !dbg !7
  %1371 = select i1 %.not36, float %1364, float %1358, !dbg !7
  %1372 = bitcast float %1371 to i32, !dbg !7
  %1373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1372), !dbg !7
  %1374 = bitcast i32 %1373 to float, !dbg !7
  %1375 = select i1 %.not36, float %1358, float %1374, !dbg !7
  %1376 = select i1 %.not36, float %1374, float %1364, !dbg !7
  %1377 = extractvalue [2 x <8 x float>] %129, 0, !dbg !7
  %1378 = extractelement <8 x float> %1377, i64 0, !dbg !7
  %1379 = extractelement <8 x float> %1377, i64 1, !dbg !7
  %1380 = extractelement <8 x float> %1377, i64 2, !dbg !7
  %1381 = extractelement <8 x float> %1377, i64 3, !dbg !7
  %1382 = select i1 %.not, float %1379, float %1378, !dbg !7
  %1383 = bitcast float %1382 to i32, !dbg !7
  %1384 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1383), !dbg !7
  %1385 = bitcast i32 %1384 to float, !dbg !7
  %1386 = select i1 %.not, float %1378, float %1385, !dbg !7
  %1387 = select i1 %.not, float %1385, float %1379, !dbg !7
  %1388 = select i1 %.not, float %1381, float %1380, !dbg !7
  %1389 = bitcast float %1388 to i32, !dbg !7
  %1390 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1389), !dbg !7
  %1391 = bitcast i32 %1390 to float, !dbg !7
  %1392 = select i1 %.not, float %1380, float %1391, !dbg !7
  %1393 = select i1 %.not, float %1391, float %1381, !dbg !7
  %1394 = select i1 %.not36, float %1392, float %1386, !dbg !7
  %1395 = bitcast float %1394 to i32, !dbg !7
  %1396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1395), !dbg !7
  %1397 = select i1 %.not36, float %1393, float %1387, !dbg !7
  %1398 = bitcast float %1397 to i32, !dbg !7
  %1399 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1398), !dbg !7
  %1400 = extractvalue [2 x <1 x float>] %130, 0, !dbg !16
  %1401 = shufflevector <1 x float> %1400, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1402 = extractelement <8 x float> %1278, i64 4, !dbg !7
  %1403 = extractelement <8 x float> %1278, i64 5, !dbg !7
  %1404 = extractelement <8 x float> %1278, i64 6, !dbg !7
  %1405 = extractelement <8 x float> %1278, i64 7, !dbg !7
  %1406 = select i1 %.not, float %1403, float %1402, !dbg !7
  %1407 = bitcast float %1406 to i32, !dbg !7
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1407), !dbg !7
  %1409 = bitcast i32 %1408 to float, !dbg !7
  %1410 = select i1 %.not, float %1402, float %1409, !dbg !7
  %1411 = select i1 %.not, float %1409, float %1403, !dbg !7
  %1412 = select i1 %.not, float %1405, float %1404, !dbg !7
  %1413 = bitcast float %1412 to i32, !dbg !7
  %1414 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1413), !dbg !7
  %1415 = bitcast i32 %1414 to float, !dbg !7
  %1416 = select i1 %.not, float %1404, float %1415, !dbg !7
  %1417 = select i1 %.not, float %1415, float %1405, !dbg !7
  %1418 = select i1 %.not36, float %1416, float %1410, !dbg !7
  %1419 = bitcast float %1418 to i32, !dbg !7
  %1420 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1419), !dbg !7
  %1421 = bitcast i32 %1420 to float, !dbg !7
  %1422 = select i1 %.not36, float %1410, float %1421, !dbg !7
  %1423 = select i1 %.not36, float %1421, float %1416, !dbg !7
  %1424 = select i1 %.not36, float %1417, float %1411, !dbg !7
  %1425 = bitcast float %1424 to i32, !dbg !7
  %1426 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1425), !dbg !7
  %1427 = bitcast i32 %1426 to float, !dbg !7
  %1428 = select i1 %.not36, float %1411, float %1427, !dbg !7
  %1429 = select i1 %.not36, float %1427, float %1417, !dbg !7
  %1430 = extractelement <8 x float> %1319, i64 4, !dbg !7
  %1431 = extractelement <8 x float> %1319, i64 5, !dbg !7
  %1432 = extractelement <8 x float> %1319, i64 6, !dbg !7
  %1433 = extractelement <8 x float> %1319, i64 7, !dbg !7
  %1434 = select i1 %.not, float %1431, float %1430, !dbg !7
  %1435 = bitcast float %1434 to i32, !dbg !7
  %1436 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1435), !dbg !7
  %1437 = bitcast i32 %1436 to float, !dbg !7
  %1438 = select i1 %.not, float %1430, float %1437, !dbg !7
  %1439 = select i1 %.not, float %1437, float %1431, !dbg !7
  %1440 = select i1 %.not, float %1433, float %1432, !dbg !7
  %1441 = bitcast float %1440 to i32, !dbg !7
  %1442 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1441), !dbg !7
  %1443 = bitcast i32 %1442 to float, !dbg !7
  %1444 = select i1 %.not, float %1432, float %1443, !dbg !7
  %1445 = select i1 %.not, float %1443, float %1433, !dbg !7
  %1446 = select i1 %.not36, float %1444, float %1438, !dbg !7
  %1447 = bitcast float %1446 to i32, !dbg !7
  %1448 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1447), !dbg !7
  %1449 = bitcast i32 %1448 to float, !dbg !7
  %1450 = select i1 %.not36, float %1438, float %1449, !dbg !7
  %1451 = select i1 %.not36, float %1449, float %1444, !dbg !7
  %1452 = select i1 %.not36, float %1445, float %1439, !dbg !7
  %1453 = bitcast float %1452 to i32, !dbg !7
  %1454 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1453), !dbg !7
  %1455 = bitcast i32 %1454 to float, !dbg !7
  %1456 = select i1 %.not36, float %1439, float %1455, !dbg !7
  %1457 = select i1 %.not36, float %1455, float %1445, !dbg !7
  %1458 = extractelement <8 x float> %1348, i64 4, !dbg !7
  %1459 = extractelement <8 x float> %1348, i64 5, !dbg !7
  %1460 = extractelement <8 x float> %1348, i64 6, !dbg !7
  %1461 = extractelement <8 x float> %1348, i64 7, !dbg !7
  %1462 = select i1 %.not, float %1459, float %1458, !dbg !7
  %1463 = bitcast float %1462 to i32, !dbg !7
  %1464 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1463), !dbg !7
  %1465 = bitcast i32 %1464 to float, !dbg !7
  %1466 = select i1 %.not, float %1458, float %1465, !dbg !7
  %1467 = select i1 %.not, float %1465, float %1459, !dbg !7
  %1468 = select i1 %.not, float %1461, float %1460, !dbg !7
  %1469 = bitcast float %1468 to i32, !dbg !7
  %1470 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1469), !dbg !7
  %1471 = bitcast i32 %1470 to float, !dbg !7
  %1472 = select i1 %.not, float %1460, float %1471, !dbg !7
  %1473 = select i1 %.not, float %1471, float %1461, !dbg !7
  %1474 = select i1 %.not36, float %1472, float %1466, !dbg !7
  %1475 = bitcast float %1474 to i32, !dbg !7
  %1476 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1475), !dbg !7
  %1477 = bitcast i32 %1476 to float, !dbg !7
  %1478 = select i1 %.not36, float %1466, float %1477, !dbg !7
  %1479 = select i1 %.not36, float %1477, float %1472, !dbg !7
  %1480 = select i1 %.not36, float %1473, float %1467, !dbg !7
  %1481 = bitcast float %1480 to i32, !dbg !7
  %1482 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1481), !dbg !7
  %1483 = bitcast i32 %1482 to float, !dbg !7
  %1484 = select i1 %.not36, float %1467, float %1483, !dbg !7
  %1485 = select i1 %.not36, float %1483, float %1473, !dbg !7
  %1486 = extractelement <8 x float> %1377, i64 4, !dbg !7
  %1487 = extractelement <8 x float> %1377, i64 5, !dbg !7
  %1488 = extractelement <8 x float> %1377, i64 6, !dbg !7
  %1489 = extractelement <8 x float> %1377, i64 7, !dbg !7
  %1490 = select i1 %.not, float %1487, float %1486, !dbg !7
  %1491 = bitcast float %1490 to i32, !dbg !7
  %1492 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1491), !dbg !7
  %1493 = bitcast i32 %1492 to float, !dbg !7
  %1494 = select i1 %.not, float %1486, float %1493, !dbg !7
  %1495 = select i1 %.not, float %1493, float %1487, !dbg !7
  %1496 = select i1 %.not, float %1489, float %1488, !dbg !7
  %1497 = bitcast float %1496 to i32, !dbg !7
  %1498 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1497), !dbg !7
  %1499 = bitcast i32 %1498 to float, !dbg !7
  %1500 = select i1 %.not, float %1488, float %1499, !dbg !7
  %1501 = select i1 %.not, float %1499, float %1489, !dbg !7
  %1502 = select i1 %.not36, float %1500, float %1494, !dbg !7
  %1503 = bitcast float %1502 to i32, !dbg !7
  %1504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1503), !dbg !7
  %1505 = select i1 %.not36, float %1501, float %1495, !dbg !7
  %1506 = bitcast float %1505 to i32, !dbg !7
  %1507 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1506), !dbg !7
  %1508 = extractvalue [2 x <8 x float>] %134, 1, !dbg !7
  %1509 = extractelement <8 x float> %1508, i64 0, !dbg !7
  %1510 = extractelement <8 x float> %1508, i64 1, !dbg !7
  %1511 = extractelement <8 x float> %1508, i64 2, !dbg !7
  %1512 = extractelement <8 x float> %1508, i64 3, !dbg !7
  %1513 = select i1 %.not, float %1510, float %1509, !dbg !7
  %1514 = bitcast float %1513 to i32, !dbg !7
  %1515 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1514), !dbg !7
  %1516 = bitcast i32 %1515 to float, !dbg !7
  %1517 = select i1 %.not, float %1509, float %1516, !dbg !7
  %1518 = select i1 %.not, float %1516, float %1510, !dbg !7
  %1519 = select i1 %.not, float %1512, float %1511, !dbg !7
  %1520 = bitcast float %1519 to i32, !dbg !7
  %1521 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1520), !dbg !7
  %1522 = bitcast i32 %1521 to float, !dbg !7
  %1523 = select i1 %.not, float %1511, float %1522, !dbg !7
  %1524 = select i1 %.not, float %1522, float %1512, !dbg !7
  %1525 = select i1 %.not36, float %1523, float %1517, !dbg !7
  %1526 = bitcast float %1525 to i32, !dbg !7
  %1527 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1526), !dbg !7
  %1528 = bitcast i32 %1527 to float, !dbg !7
  %1529 = select i1 %.not36, float %1517, float %1528, !dbg !7
  %1530 = select i1 %.not36, float %1528, float %1523, !dbg !7
  %1531 = select i1 %.not36, float %1524, float %1518, !dbg !7
  %1532 = bitcast float %1531 to i32, !dbg !7
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1532), !dbg !7
  %1534 = bitcast i32 %1533 to float, !dbg !7
  %1535 = select i1 %.not36, float %1518, float %1534, !dbg !7
  %1536 = select i1 %.not36, float %1534, float %1524, !dbg !7
  %1537 = extractvalue [2 x <8 x float>] %133, 1, !dbg !7
  %1538 = extractelement <8 x float> %1537, i64 0, !dbg !7
  %1539 = extractelement <8 x float> %1537, i64 1, !dbg !7
  %1540 = extractelement <8 x float> %1537, i64 2, !dbg !7
  %1541 = extractelement <8 x float> %1537, i64 3, !dbg !7
  %1542 = select i1 %.not, float %1539, float %1538, !dbg !7
  %1543 = bitcast float %1542 to i32, !dbg !7
  %1544 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1543), !dbg !7
  %1545 = bitcast i32 %1544 to float, !dbg !7
  %1546 = select i1 %.not, float %1538, float %1545, !dbg !7
  %1547 = select i1 %.not, float %1545, float %1539, !dbg !7
  %1548 = select i1 %.not, float %1541, float %1540, !dbg !7
  %1549 = bitcast float %1548 to i32, !dbg !7
  %1550 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1549), !dbg !7
  %1551 = bitcast i32 %1550 to float, !dbg !7
  %1552 = select i1 %.not, float %1540, float %1551, !dbg !7
  %1553 = select i1 %.not, float %1551, float %1541, !dbg !7
  %1554 = select i1 %.not36, float %1552, float %1546, !dbg !7
  %1555 = bitcast float %1554 to i32, !dbg !7
  %1556 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1555), !dbg !7
  %1557 = bitcast i32 %1556 to float, !dbg !7
  %1558 = select i1 %.not36, float %1546, float %1557, !dbg !7
  %1559 = select i1 %.not36, float %1557, float %1552, !dbg !7
  %1560 = select i1 %.not36, float %1553, float %1547, !dbg !7
  %1561 = bitcast float %1560 to i32, !dbg !7
  %1562 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1561), !dbg !7
  %1563 = bitcast i32 %1562 to float, !dbg !7
  %1564 = select i1 %.not36, float %1547, float %1563, !dbg !7
  %1565 = select i1 %.not36, float %1563, float %1553, !dbg !7
  %1566 = extractvalue [2 x <8 x float>] %132, 1, !dbg !7
  %1567 = extractelement <8 x float> %1566, i64 0, !dbg !7
  %1568 = extractelement <8 x float> %1566, i64 1, !dbg !7
  %1569 = extractelement <8 x float> %1566, i64 2, !dbg !7
  %1570 = extractelement <8 x float> %1566, i64 3, !dbg !7
  %1571 = select i1 %.not, float %1568, float %1567, !dbg !7
  %1572 = bitcast float %1571 to i32, !dbg !7
  %1573 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1572), !dbg !7
  %1574 = bitcast i32 %1573 to float, !dbg !7
  %1575 = select i1 %.not, float %1567, float %1574, !dbg !7
  %1576 = select i1 %.not, float %1574, float %1568, !dbg !7
  %1577 = select i1 %.not, float %1570, float %1569, !dbg !7
  %1578 = bitcast float %1577 to i32, !dbg !7
  %1579 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1578), !dbg !7
  %1580 = bitcast i32 %1579 to float, !dbg !7
  %1581 = select i1 %.not, float %1569, float %1580, !dbg !7
  %1582 = select i1 %.not, float %1580, float %1570, !dbg !7
  %1583 = select i1 %.not36, float %1581, float %1575, !dbg !7
  %1584 = bitcast float %1583 to i32, !dbg !7
  %1585 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1584), !dbg !7
  %1586 = bitcast i32 %1585 to float, !dbg !7
  %1587 = select i1 %.not36, float %1575, float %1586, !dbg !7
  %1588 = select i1 %.not36, float %1586, float %1581, !dbg !7
  %1589 = select i1 %.not36, float %1582, float %1576, !dbg !7
  %1590 = bitcast float %1589 to i32, !dbg !7
  %1591 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1590), !dbg !7
  %1592 = bitcast i32 %1591 to float, !dbg !7
  %1593 = select i1 %.not36, float %1576, float %1592, !dbg !7
  %1594 = select i1 %.not36, float %1592, float %1582, !dbg !7
  %1595 = extractvalue [2 x <8 x float>] %129, 1, !dbg !7
  %1596 = extractelement <8 x float> %1595, i64 0, !dbg !7
  %1597 = extractelement <8 x float> %1595, i64 1, !dbg !7
  %1598 = extractelement <8 x float> %1595, i64 2, !dbg !7
  %1599 = extractelement <8 x float> %1595, i64 3, !dbg !7
  %1600 = select i1 %.not, float %1597, float %1596, !dbg !7
  %1601 = bitcast float %1600 to i32, !dbg !7
  %1602 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1601), !dbg !7
  %1603 = bitcast i32 %1602 to float, !dbg !7
  %1604 = select i1 %.not, float %1596, float %1603, !dbg !7
  %1605 = select i1 %.not, float %1603, float %1597, !dbg !7
  %1606 = select i1 %.not, float %1599, float %1598, !dbg !7
  %1607 = bitcast float %1606 to i32, !dbg !7
  %1608 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1607), !dbg !7
  %1609 = bitcast i32 %1608 to float, !dbg !7
  %1610 = select i1 %.not, float %1598, float %1609, !dbg !7
  %1611 = select i1 %.not, float %1609, float %1599, !dbg !7
  %1612 = select i1 %.not36, float %1610, float %1604, !dbg !7
  %1613 = bitcast float %1612 to i32, !dbg !7
  %1614 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1613), !dbg !7
  %1615 = select i1 %.not36, float %1611, float %1605, !dbg !7
  %1616 = bitcast float %1615 to i32, !dbg !7
  %1617 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1616), !dbg !7
  %1618 = extractvalue [2 x <1 x float>] %130, 1, !dbg !16
  %1619 = shufflevector <1 x float> %1618, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1620 = extractelement <8 x float> %1508, i64 4, !dbg !7
  %1621 = extractelement <8 x float> %1508, i64 5, !dbg !7
  %1622 = extractelement <8 x float> %1508, i64 6, !dbg !7
  %1623 = extractelement <8 x float> %1508, i64 7, !dbg !7
  %1624 = select i1 %.not, float %1621, float %1620, !dbg !7
  %1625 = bitcast float %1624 to i32, !dbg !7
  %1626 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1625), !dbg !7
  %1627 = bitcast i32 %1626 to float, !dbg !7
  %1628 = select i1 %.not, float %1620, float %1627, !dbg !7
  %1629 = select i1 %.not, float %1627, float %1621, !dbg !7
  %1630 = select i1 %.not, float %1623, float %1622, !dbg !7
  %1631 = bitcast float %1630 to i32, !dbg !7
  %1632 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1631), !dbg !7
  %1633 = bitcast i32 %1632 to float, !dbg !7
  %1634 = select i1 %.not, float %1622, float %1633, !dbg !7
  %1635 = select i1 %.not, float %1633, float %1623, !dbg !7
  %1636 = select i1 %.not36, float %1634, float %1628, !dbg !7
  %1637 = bitcast float %1636 to i32, !dbg !7
  %1638 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1637), !dbg !7
  %1639 = bitcast i32 %1638 to float, !dbg !7
  %1640 = select i1 %.not36, float %1628, float %1639, !dbg !7
  %1641 = select i1 %.not36, float %1639, float %1634, !dbg !7
  %1642 = select i1 %.not36, float %1635, float %1629, !dbg !7
  %1643 = bitcast float %1642 to i32, !dbg !7
  %1644 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1643), !dbg !7
  %1645 = bitcast i32 %1644 to float, !dbg !7
  %1646 = select i1 %.not36, float %1629, float %1645, !dbg !7
  %1647 = select i1 %.not36, float %1645, float %1635, !dbg !7
  %1648 = extractelement <8 x float> %1537, i64 4, !dbg !7
  %1649 = extractelement <8 x float> %1537, i64 5, !dbg !7
  %1650 = extractelement <8 x float> %1537, i64 6, !dbg !7
  %1651 = extractelement <8 x float> %1537, i64 7, !dbg !7
  %1652 = select i1 %.not, float %1649, float %1648, !dbg !7
  %1653 = bitcast float %1652 to i32, !dbg !7
  %1654 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1653), !dbg !7
  %1655 = bitcast i32 %1654 to float, !dbg !7
  %1656 = select i1 %.not, float %1648, float %1655, !dbg !7
  %1657 = select i1 %.not, float %1655, float %1649, !dbg !7
  %1658 = select i1 %.not, float %1651, float %1650, !dbg !7
  %1659 = bitcast float %1658 to i32, !dbg !7
  %1660 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1659), !dbg !7
  %1661 = bitcast i32 %1660 to float, !dbg !7
  %1662 = select i1 %.not, float %1650, float %1661, !dbg !7
  %1663 = select i1 %.not, float %1661, float %1651, !dbg !7
  %1664 = select i1 %.not36, float %1662, float %1656, !dbg !7
  %1665 = bitcast float %1664 to i32, !dbg !7
  %1666 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1665), !dbg !7
  %1667 = bitcast i32 %1666 to float, !dbg !7
  %1668 = select i1 %.not36, float %1656, float %1667, !dbg !7
  %1669 = select i1 %.not36, float %1667, float %1662, !dbg !7
  %1670 = select i1 %.not36, float %1663, float %1657, !dbg !7
  %1671 = bitcast float %1670 to i32, !dbg !7
  %1672 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1671), !dbg !7
  %1673 = bitcast i32 %1672 to float, !dbg !7
  %1674 = select i1 %.not36, float %1657, float %1673, !dbg !7
  %1675 = select i1 %.not36, float %1673, float %1663, !dbg !7
  %1676 = extractelement <8 x float> %1566, i64 4, !dbg !7
  %1677 = extractelement <8 x float> %1566, i64 5, !dbg !7
  %1678 = extractelement <8 x float> %1566, i64 6, !dbg !7
  %1679 = extractelement <8 x float> %1566, i64 7, !dbg !7
  %1680 = select i1 %.not, float %1677, float %1676, !dbg !7
  %1681 = bitcast float %1680 to i32, !dbg !7
  %1682 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1681), !dbg !7
  %1683 = bitcast i32 %1682 to float, !dbg !7
  %1684 = select i1 %.not, float %1676, float %1683, !dbg !7
  %1685 = select i1 %.not, float %1683, float %1677, !dbg !7
  %1686 = select i1 %.not, float %1679, float %1678, !dbg !7
  %1687 = bitcast float %1686 to i32, !dbg !7
  %1688 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1687), !dbg !7
  %1689 = bitcast i32 %1688 to float, !dbg !7
  %1690 = select i1 %.not, float %1678, float %1689, !dbg !7
  %1691 = select i1 %.not, float %1689, float %1679, !dbg !7
  %1692 = select i1 %.not36, float %1690, float %1684, !dbg !7
  %1693 = bitcast float %1692 to i32, !dbg !7
  %1694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1693), !dbg !7
  %1695 = bitcast i32 %1694 to float, !dbg !7
  %1696 = select i1 %.not36, float %1684, float %1695, !dbg !7
  %1697 = select i1 %.not36, float %1695, float %1690, !dbg !7
  %1698 = select i1 %.not36, float %1691, float %1685, !dbg !7
  %1699 = bitcast float %1698 to i32, !dbg !7
  %1700 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1699), !dbg !7
  %1701 = bitcast i32 %1700 to float, !dbg !7
  %1702 = select i1 %.not36, float %1685, float %1701, !dbg !7
  %1703 = select i1 %.not36, float %1701, float %1691, !dbg !7
  %1704 = extractelement <8 x float> %1595, i64 4, !dbg !7
  %1705 = extractelement <8 x float> %1595, i64 5, !dbg !7
  %1706 = extractelement <8 x float> %1595, i64 6, !dbg !7
  %1707 = extractelement <8 x float> %1595, i64 7, !dbg !7
  %1708 = select i1 %.not, float %1705, float %1704, !dbg !7
  %1709 = bitcast float %1708 to i32, !dbg !7
  %1710 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1709), !dbg !7
  %1711 = bitcast i32 %1710 to float, !dbg !7
  %1712 = select i1 %.not, float %1704, float %1711, !dbg !7
  %1713 = select i1 %.not, float %1711, float %1705, !dbg !7
  %1714 = select i1 %.not, float %1707, float %1706, !dbg !7
  %1715 = bitcast float %1714 to i32, !dbg !7
  %1716 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1291, i32 %1715), !dbg !7
  %1717 = bitcast i32 %1716 to float, !dbg !7
  %1718 = select i1 %.not, float %1706, float %1717, !dbg !7
  %1719 = select i1 %.not, float %1717, float %1707, !dbg !7
  %1720 = select i1 %.not36, float %1718, float %1712, !dbg !7
  %1721 = bitcast float %1720 to i32, !dbg !7
  %1722 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1721), !dbg !7
  %1723 = select i1 %.not36, float %1719, float %1713, !dbg !7
  %1724 = bitcast float %1723 to i32, !dbg !7
  %1725 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1307, i32 %1724), !dbg !7
  %1726 = bitcast i32 %1722 to float, !dbg !7
  %1727 = select i1 %.not36, float %1712, float %1726, !dbg !7
  %1728 = insertelement <4 x float> poison, float %1727, i64 0, !dbg !16
  %1729 = bitcast i32 %1725 to float, !dbg !7
  %1730 = select i1 %.not36, float %1713, float %1729, !dbg !7
  %1731 = insertelement <4 x float> %1728, float %1730, i64 1, !dbg !16
  %1732 = select i1 %.not36, float %1726, float %1718, !dbg !7
  %1733 = insertelement <4 x float> %1731, float %1732, i64 2, !dbg !16
  %1734 = select i1 %.not36, float %1729, float %1719, !dbg !7
  %1735 = insertelement <4 x float> %1733, float %1734, i64 3, !dbg !16
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1619, !dbg !16
  %normalized = fmul <4 x float> %1735, %row.reciprocal, !dbg !16
  %1736 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !17
  %1737 = extractelement <4 x half> %1736, i64 3, !dbg !17
  %1738 = extractelement <4 x half> %1736, i64 2, !dbg !17
  %1739 = extractelement <4 x half> %1736, i64 1, !dbg !17
  %1740 = extractelement <4 x half> %1736, i64 0, !dbg !17
  %1741 = bitcast i32 %1614 to float, !dbg !7
  %1742 = select i1 %.not36, float %1604, float %1741, !dbg !7
  %1743 = insertelement <4 x float> poison, float %1742, i64 0, !dbg !16
  %1744 = bitcast i32 %1617 to float, !dbg !7
  %1745 = select i1 %.not36, float %1605, float %1744, !dbg !7
  %1746 = insertelement <4 x float> %1743, float %1745, i64 1, !dbg !16
  %1747 = select i1 %.not36, float %1741, float %1610, !dbg !7
  %1748 = insertelement <4 x float> %1746, float %1747, i64 2, !dbg !16
  %1749 = select i1 %.not36, float %1744, float %1611, !dbg !7
  %1750 = insertelement <4 x float> %1748, float %1749, i64 3, !dbg !16
  %normalized37 = fmul <4 x float> %1750, %row.reciprocal, !dbg !16
  %1751 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !17
  %1752 = extractelement <4 x half> %1751, i64 3, !dbg !17
  %1753 = extractelement <4 x half> %1751, i64 2, !dbg !17
  %1754 = extractelement <4 x half> %1751, i64 1, !dbg !17
  %1755 = extractelement <4 x half> %1751, i64 0, !dbg !17
  %1756 = bitcast i32 %1504 to float, !dbg !7
  %1757 = select i1 %.not36, float %1494, float %1756, !dbg !7
  %1758 = insertelement <4 x float> poison, float %1757, i64 0, !dbg !16
  %1759 = bitcast i32 %1507 to float, !dbg !7
  %1760 = select i1 %.not36, float %1495, float %1759, !dbg !7
  %1761 = insertelement <4 x float> %1758, float %1760, i64 1, !dbg !16
  %1762 = select i1 %.not36, float %1756, float %1500, !dbg !7
  %1763 = insertelement <4 x float> %1761, float %1762, i64 2, !dbg !16
  %1764 = select i1 %.not36, float %1759, float %1501, !dbg !7
  %1765 = insertelement <4 x float> %1763, float %1764, i64 3, !dbg !16
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1401, !dbg !16
  %normalized39 = fmul <4 x float> %1765, %row.reciprocal38, !dbg !16
  %1766 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !17
  %1767 = extractelement <4 x half> %1766, i64 3, !dbg !17
  %1768 = extractelement <4 x half> %1766, i64 2, !dbg !17
  %1769 = extractelement <4 x half> %1766, i64 1, !dbg !17
  %1770 = extractelement <4 x half> %1766, i64 0, !dbg !17
  %1771 = bitcast i32 %1396 to float, !dbg !7
  %1772 = select i1 %.not36, float %1386, float %1771, !dbg !7
  %1773 = insertelement <4 x float> poison, float %1772, i64 0, !dbg !16
  %1774 = bitcast i32 %1399 to float, !dbg !7
  %1775 = select i1 %.not36, float %1387, float %1774, !dbg !7
  %1776 = insertelement <4 x float> %1773, float %1775, i64 1, !dbg !16
  %1777 = select i1 %.not36, float %1771, float %1392, !dbg !7
  %1778 = insertelement <4 x float> %1776, float %1777, i64 2, !dbg !16
  %1779 = select i1 %.not36, float %1774, float %1393, !dbg !7
  %1780 = insertelement <4 x float> %1778, float %1779, i64 3, !dbg !16
  %normalized40 = fmul <4 x float> %1780, %row.reciprocal38, !dbg !16
  %1781 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !17
  %1782 = extractelement <4 x half> %1781, i64 3, !dbg !17
  %1783 = extractelement <4 x half> %1781, i64 2, !dbg !17
  %1784 = extractelement <4 x half> %1781, i64 1, !dbg !17
  %1785 = extractelement <4 x half> %1781, i64 0, !dbg !17
  %1786 = add nsw i64 %18, %24, !dbg !18
  %1787 = shl i64 %1786, 19, !dbg !18
  %1788 = and i64 %1787, 16252928, !dbg !18
  %address.thread841 = call i32 asm sideeffect "", "=v,0"(i32 %33) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local842 = zext nneg i32 %address.thread841 to i64, !dbg !8
  %address.local843 = add i64 %27, %address.local842, !dbg !18
  %1789 = lshr i64 %address.local842, 2, !dbg !18
  %address.local845 = and i64 %1789, 48, !dbg !18
  %address.local846 = add i64 %address.local843, %address.local845, !dbg !18
  %address.local847 = and i64 %address.local842, 240, !dbg !18
  %address.local848 = sub i64 %address.local846, %address.local847, !dbg !18
  %address.local849 = add i64 %address.local848, %28, !dbg !18
  %address.local850 = shl nuw nsw i64 %address.local849, 7, !dbg !18
  %address.local851 = add nuw nsw i64 %1788, %address.local850, !dbg !18
  %address.local852 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local853 = lshr i64 %address.local842, 2, !dbg !10
  %address.local854 = and i64 %address.local853, 12, !dbg !10
  %address.local855 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local854, !dbg !18
  store half %1785, ptr addrspace(1) %address.local855, align 2, !dbg !18
  %address.local856 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local857 = or disjoint i64 %address.local854, 1, !dbg !18
  %address.local858 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local856, i64 %address.local857, !dbg !18
  store half %1784, ptr addrspace(1) %address.local858, align 2, !dbg !18
  %address.local859 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local860 = or disjoint i64 %address.local854, 2, !dbg !18
  %address.local861 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local860, !dbg !18
  store half %1783, ptr addrspace(1) %address.local861, align 2, !dbg !18
  %address.local862 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local863 = or disjoint i64 %address.local854, 3, !dbg !18
  %address.local864 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local862, i64 %address.local863, !dbg !18
  store half %1782, ptr addrspace(1) %address.local864, align 2, !dbg !18
  %address.local865 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local866 = or disjoint i64 %address.local854, 16, !dbg !18
  %address.local867 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local865, i64 %address.local866, !dbg !18
  store half %1770, ptr addrspace(1) %address.local867, align 2, !dbg !18
  %address.local868 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local869 = or disjoint i64 %address.local854, 17, !dbg !18
  %address.local870 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local868, i64 %address.local869, !dbg !18
  store half %1769, ptr addrspace(1) %address.local870, align 2, !dbg !18
  %address.local871 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local872 = or disjoint i64 %address.local854, 18, !dbg !18
  %address.local873 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local871, i64 %address.local872, !dbg !18
  store half %1768, ptr addrspace(1) %address.local873, align 2, !dbg !18
  %address.local874 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local875 = or disjoint i64 %address.local854, 19, !dbg !18
  %address.local876 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local874, i64 %address.local875, !dbg !18
  store half %1767, ptr addrspace(1) %address.local876, align 2, !dbg !18
  %address.local877 = shl i64 %address.local849, 7, !dbg !18
  %address.local878 = add i64 %address.local877, 8192, !dbg !18
  %address.local879 = add nuw nsw i64 %1788, %address.local878, !dbg !18
  %address.local880 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local881 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local854, !dbg !18
  store half %1755, ptr addrspace(1) %address.local881, align 2, !dbg !18
  %address.local882 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local882, i64 %address.local857, !dbg !18
  store half %1754, ptr addrspace(1) %address.local883, align 2, !dbg !18
  %address.local884 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local885 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local884, i64 %address.local860, !dbg !18
  store half %1753, ptr addrspace(1) %address.local885, align 2, !dbg !18
  %address.local886 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local887 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local886, i64 %address.local863, !dbg !18
  store half %1752, ptr addrspace(1) %address.local887, align 2, !dbg !18
  %address.local888 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local888, i64 %address.local866, !dbg !18
  store half %1740, ptr addrspace(1) %address.local889, align 2, !dbg !18
  %address.local890 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local891 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local890, i64 %address.local869, !dbg !18
  store half %1739, ptr addrspace(1) %address.local891, align 2, !dbg !18
  %address.local892 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local893 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local892, i64 %address.local872, !dbg !18
  store half %1738, ptr addrspace(1) %address.local893, align 2, !dbg !18
  %address.local894 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local895 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local875, !dbg !18
  store half %1737, ptr addrspace(1) %address.local895, align 2, !dbg !18
  %1790 = insertelement <4 x float> poison, float %1696, i64 0, !dbg !16
  %1791 = insertelement <4 x float> %1790, float %1702, i64 1, !dbg !16
  %1792 = insertelement <4 x float> %1791, float %1697, i64 2, !dbg !16
  %1793 = insertelement <4 x float> %1792, float %1703, i64 3, !dbg !16
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1619, !dbg !16
  %normalized42 = fmul <4 x float> %1793, %row.reciprocal41, !dbg !16
  %1794 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !17
  %1795 = extractelement <4 x half> %1794, i64 3, !dbg !17
  %1796 = extractelement <4 x half> %1794, i64 2, !dbg !17
  %1797 = extractelement <4 x half> %1794, i64 1, !dbg !17
  %1798 = extractelement <4 x half> %1794, i64 0, !dbg !17
  %1799 = insertelement <4 x float> poison, float %1587, i64 0, !dbg !16
  %1800 = insertelement <4 x float> %1799, float %1593, i64 1, !dbg !16
  %1801 = insertelement <4 x float> %1800, float %1588, i64 2, !dbg !16
  %1802 = insertelement <4 x float> %1801, float %1594, i64 3, !dbg !16
  %normalized43 = fmul <4 x float> %1802, %row.reciprocal41, !dbg !16
  %1803 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !17
  %1804 = extractelement <4 x half> %1803, i64 3, !dbg !17
  %1805 = extractelement <4 x half> %1803, i64 2, !dbg !17
  %1806 = extractelement <4 x half> %1803, i64 1, !dbg !17
  %1807 = extractelement <4 x half> %1803, i64 0, !dbg !17
  %1808 = insertelement <4 x float> poison, float %1478, i64 0, !dbg !16
  %1809 = insertelement <4 x float> %1808, float %1484, i64 1, !dbg !16
  %1810 = insertelement <4 x float> %1809, float %1479, i64 2, !dbg !16
  %1811 = insertelement <4 x float> %1810, float %1485, i64 3, !dbg !16
  %row.reciprocal44 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1401, !dbg !16
  %normalized45 = fmul <4 x float> %1811, %row.reciprocal44, !dbg !16
  %1812 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !17
  %1813 = extractelement <4 x half> %1812, i64 3, !dbg !17
  %1814 = extractelement <4 x half> %1812, i64 2, !dbg !17
  %1815 = extractelement <4 x half> %1812, i64 1, !dbg !17
  %1816 = extractelement <4 x half> %1812, i64 0, !dbg !17
  %1817 = insertelement <4 x float> poison, float %1369, i64 0, !dbg !16
  %1818 = insertelement <4 x float> %1817, float %1375, i64 1, !dbg !16
  %1819 = insertelement <4 x float> %1818, float %1370, i64 2, !dbg !16
  %1820 = insertelement <4 x float> %1819, float %1376, i64 3, !dbg !16
  %normalized46 = fmul <4 x float> %1820, %row.reciprocal44, !dbg !16
  %1821 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !17
  %1822 = extractelement <4 x half> %1821, i64 3, !dbg !17
  %1823 = extractelement <4 x half> %1821, i64 2, !dbg !17
  %1824 = extractelement <4 x half> %1821, i64 1, !dbg !17
  %1825 = extractelement <4 x half> %1821, i64 0, !dbg !17
  %1826 = shl nuw nsw i64 %26, 19, !dbg !18
  %address.thread896 = call i32 asm sideeffect "", "=v,0"(i32 %33) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local897 = zext nneg i32 %address.thread896 to i64, !dbg !8
  %address.local898 = add i64 %27, %address.local897, !dbg !18
  %1827 = lshr i64 %address.local897, 2, !dbg !18
  %address.local900 = and i64 %1827, 48, !dbg !18
  %address.local901 = add i64 %address.local898, %address.local900, !dbg !18
  %address.local902 = and i64 %address.local897, 240, !dbg !18
  %address.local903 = sub i64 %address.local901, %address.local902, !dbg !18
  %address.local904 = add i64 %address.local903, %28, !dbg !18
  %address.local905 = shl nuw nsw i64 %address.local904, 7, !dbg !18
  %address.local906 = add nuw nsw i64 %1826, %address.local905, !dbg !18
  %address.local907 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local908 = lshr i64 %address.local897, 2, !dbg !10
  %address.local909 = and i64 %address.local908, 12, !dbg !10
  %address.local910 = or disjoint i64 %address.local909, 32, !dbg !18
  %address.local911 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local907, i64 %address.local910, !dbg !18
  store half %1825, ptr addrspace(1) %address.local911, align 2, !dbg !18
  %address.local912 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local913 = or disjoint i64 %address.local909, 33, !dbg !18
  %address.local914 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local912, i64 %address.local913, !dbg !18
  store half %1824, ptr addrspace(1) %address.local914, align 2, !dbg !18
  %address.local915 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local916 = or disjoint i64 %address.local909, 34, !dbg !18
  %address.local917 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local915, i64 %address.local916, !dbg !18
  store half %1823, ptr addrspace(1) %address.local917, align 2, !dbg !18
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local919 = or disjoint i64 %address.local909, 35, !dbg !18
  %address.local920 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local918, i64 %address.local919, !dbg !18
  store half %1822, ptr addrspace(1) %address.local920, align 2, !dbg !18
  %address.local921 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local922 = or disjoint i64 %address.local909, 48, !dbg !18
  %address.local923 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local921, i64 %address.local922, !dbg !18
  store half %1816, ptr addrspace(1) %address.local923, align 2, !dbg !18
  %address.local924 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local925 = or disjoint i64 %address.local909, 49, !dbg !18
  %address.local926 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local924, i64 %address.local925, !dbg !18
  store half %1815, ptr addrspace(1) %address.local926, align 2, !dbg !18
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local928 = or disjoint i64 %address.local909, 50, !dbg !18
  %address.local929 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local927, i64 %address.local928, !dbg !18
  store half %1814, ptr addrspace(1) %address.local929, align 2, !dbg !18
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local931 = or i64 %address.local908, 51, !dbg !18
  %address.local932 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local930, i64 %address.local931, !dbg !18
  store half %1813, ptr addrspace(1) %address.local932, align 2, !dbg !18
  %address.local933 = shl i64 %address.local904, 7, !dbg !18
  %address.local934 = add i64 %address.local933, 8192, !dbg !18
  %address.local935 = add nuw nsw i64 %1826, %address.local934, !dbg !18
  %address.local936 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local936, i64 %address.local910, !dbg !18
  store half %1807, ptr addrspace(1) %address.local937, align 2, !dbg !18
  %address.local938 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local938, i64 %address.local913, !dbg !18
  store half %1806, ptr addrspace(1) %address.local939, align 2, !dbg !18
  %address.local940 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local941 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local940, i64 %address.local916, !dbg !18
  store half %1805, ptr addrspace(1) %address.local941, align 2, !dbg !18
  %address.local942 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local943 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local942, i64 %address.local919, !dbg !18
  store half %1804, ptr addrspace(1) %address.local943, align 2, !dbg !18
  %address.local944 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local922, !dbg !18
  store half %1798, ptr addrspace(1) %address.local945, align 2, !dbg !18
  %address.local946 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local946, i64 %address.local925, !dbg !18
  store half %1797, ptr addrspace(1) %address.local947, align 2, !dbg !18
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local948, i64 %address.local928, !dbg !18
  store half %1796, ptr addrspace(1) %address.local949, align 2, !dbg !18
  %address.local950 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local950, i64 %address.local931, !dbg !18
  store half %1795, ptr addrspace(1) %address.local951, align 2, !dbg !18
  %1828 = insertelement <4 x float> poison, float %1668, i64 0, !dbg !16
  %1829 = insertelement <4 x float> %1828, float %1674, i64 1, !dbg !16
  %1830 = insertelement <4 x float> %1829, float %1669, i64 2, !dbg !16
  %1831 = insertelement <4 x float> %1830, float %1675, i64 3, !dbg !16
  %row.reciprocal47 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1619, !dbg !16
  %normalized48 = fmul <4 x float> %1831, %row.reciprocal47, !dbg !16
  %1832 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !17
  %1833 = extractelement <4 x half> %1832, i64 3, !dbg !17
  %1834 = extractelement <4 x half> %1832, i64 2, !dbg !17
  %1835 = extractelement <4 x half> %1832, i64 1, !dbg !17
  %1836 = extractelement <4 x half> %1832, i64 0, !dbg !17
  %1837 = insertelement <4 x float> poison, float %1558, i64 0, !dbg !16
  %1838 = insertelement <4 x float> %1837, float %1564, i64 1, !dbg !16
  %1839 = insertelement <4 x float> %1838, float %1559, i64 2, !dbg !16
  %1840 = insertelement <4 x float> %1839, float %1565, i64 3, !dbg !16
  %normalized49 = fmul <4 x float> %1840, %row.reciprocal47, !dbg !16
  %1841 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !17
  %1842 = extractelement <4 x half> %1841, i64 3, !dbg !17
  %1843 = extractelement <4 x half> %1841, i64 2, !dbg !17
  %1844 = extractelement <4 x half> %1841, i64 1, !dbg !17
  %1845 = extractelement <4 x half> %1841, i64 0, !dbg !17
  %1846 = insertelement <4 x float> poison, float %1450, i64 0, !dbg !16
  %1847 = insertelement <4 x float> %1846, float %1456, i64 1, !dbg !16
  %1848 = insertelement <4 x float> %1847, float %1451, i64 2, !dbg !16
  %1849 = insertelement <4 x float> %1848, float %1457, i64 3, !dbg !16
  %row.reciprocal50 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1401, !dbg !16
  %normalized51 = fmul <4 x float> %1849, %row.reciprocal50, !dbg !16
  %1850 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !17
  %1851 = extractelement <4 x half> %1850, i64 3, !dbg !17
  %1852 = extractelement <4 x half> %1850, i64 2, !dbg !17
  %1853 = extractelement <4 x half> %1850, i64 1, !dbg !17
  %1854 = extractelement <4 x half> %1850, i64 0, !dbg !17
  %1855 = insertelement <4 x float> poison, float %1340, i64 0, !dbg !16
  %1856 = insertelement <4 x float> %1855, float %1346, i64 1, !dbg !16
  %1857 = insertelement <4 x float> %1856, float %1341, i64 2, !dbg !16
  %1858 = insertelement <4 x float> %1857, float %1347, i64 3, !dbg !16
  %normalized52 = fmul <4 x float> %1858, %row.reciprocal50, !dbg !16
  %1859 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !17
  %1860 = extractelement <4 x half> %1859, i64 3, !dbg !17
  %1861 = extractelement <4 x half> %1859, i64 2, !dbg !17
  %1862 = extractelement <4 x half> %1859, i64 1, !dbg !17
  %1863 = extractelement <4 x half> %1859, i64 0, !dbg !17
  %1864 = shl nuw nsw i64 %26, 19, !dbg !18
  %address.thread952 = call i32 asm sideeffect "", "=v,0"(i32 %33) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local953 = zext nneg i32 %address.thread952 to i64, !dbg !8
  %address.local954 = add i64 %27, %address.local953, !dbg !18
  %1865 = lshr i64 %address.local953, 2, !dbg !18
  %address.local956 = and i64 %1865, 48, !dbg !18
  %address.local957 = add i64 %address.local954, %address.local956, !dbg !18
  %address.local958 = and i64 %address.local953, 240, !dbg !18
  %address.local959 = sub i64 %address.local957, %address.local958, !dbg !18
  %address.local960 = add i64 %address.local959, %28, !dbg !18
  %address.local961 = shl nuw nsw i64 %address.local960, 7, !dbg !18
  %address.local962 = add nuw nsw i64 %1864, %address.local961, !dbg !18
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local964 = lshr i64 %address.local953, 2, !dbg !10
  %address.local965 = and i64 %address.local964, 12, !dbg !10
  %address.local966 = or disjoint i64 %address.local965, 64, !dbg !18
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local963, i64 %address.local966, !dbg !18
  store half %1863, ptr addrspace(1) %address.local967, align 2, !dbg !18
  %address.local968 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local969 = or disjoint i64 %address.local965, 65, !dbg !18
  %address.local970 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local968, i64 %address.local969, !dbg !18
  store half %1862, ptr addrspace(1) %address.local970, align 2, !dbg !18
  %address.local971 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local972 = or disjoint i64 %address.local965, 66, !dbg !18
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local971, i64 %address.local972, !dbg !18
  store half %1861, ptr addrspace(1) %address.local973, align 2, !dbg !18
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local975 = or disjoint i64 %address.local965, 67, !dbg !18
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local974, i64 %address.local975, !dbg !18
  store half %1860, ptr addrspace(1) %address.local976, align 2, !dbg !18
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local978 = or disjoint i64 %address.local965, 80, !dbg !18
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local977, i64 %address.local978, !dbg !18
  store half %1854, ptr addrspace(1) %address.local979, align 2, !dbg !18
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local981 = or disjoint i64 %address.local965, 81, !dbg !18
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local980, i64 %address.local981, !dbg !18
  store half %1853, ptr addrspace(1) %address.local982, align 2, !dbg !18
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local984 = or disjoint i64 %address.local965, 82, !dbg !18
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local983, i64 %address.local984, !dbg !18
  store half %1852, ptr addrspace(1) %address.local985, align 2, !dbg !18
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local987 = or disjoint i64 %address.local965, 83, !dbg !18
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local986, i64 %address.local987, !dbg !18
  store half %1851, ptr addrspace(1) %address.local988, align 2, !dbg !18
  %address.local989 = shl i64 %address.local960, 7, !dbg !18
  %address.local990 = add i64 %address.local989, 8192, !dbg !18
  %address.local991 = add nuw nsw i64 %1864, %address.local990, !dbg !18
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local992, i64 %address.local966, !dbg !18
  store half %1845, ptr addrspace(1) %address.local993, align 2, !dbg !18
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local994, i64 %address.local969, !dbg !18
  store half %1844, ptr addrspace(1) %address.local995, align 2, !dbg !18
  %address.local996 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local996, i64 %address.local972, !dbg !18
  store half %1843, ptr addrspace(1) %address.local997, align 2, !dbg !18
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local999 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local998, i64 %address.local975, !dbg !18
  store half %1842, ptr addrspace(1) %address.local999, align 2, !dbg !18
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1000, i64 %address.local978, !dbg !18
  store half %1836, ptr addrspace(1) %address.local1001, align 2, !dbg !18
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1002, i64 %address.local981, !dbg !18
  store half %1835, ptr addrspace(1) %address.local1003, align 2, !dbg !18
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local984, !dbg !18
  store half %1834, ptr addrspace(1) %address.local1005, align 2, !dbg !18
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1006, i64 %address.local987, !dbg !18
  store half %1833, ptr addrspace(1) %address.local1007, align 2, !dbg !18
  %1866 = insertelement <4 x float> poison, float %1640, i64 0, !dbg !16
  %1867 = insertelement <4 x float> %1866, float %1646, i64 1, !dbg !16
  %1868 = insertelement <4 x float> %1867, float %1641, i64 2, !dbg !16
  %1869 = insertelement <4 x float> %1868, float %1647, i64 3, !dbg !16
  %row.reciprocal53 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1619, !dbg !16
  %normalized54 = fmul <4 x float> %1869, %row.reciprocal53, !dbg !16
  %1870 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !17
  %1871 = extractelement <4 x half> %1870, i64 3, !dbg !17
  %1872 = extractelement <4 x half> %1870, i64 2, !dbg !17
  %1873 = extractelement <4 x half> %1870, i64 1, !dbg !17
  %1874 = extractelement <4 x half> %1870, i64 0, !dbg !17
  %1875 = insertelement <4 x float> poison, float %1529, i64 0, !dbg !16
  %1876 = insertelement <4 x float> %1875, float %1535, i64 1, !dbg !16
  %1877 = insertelement <4 x float> %1876, float %1530, i64 2, !dbg !16
  %1878 = insertelement <4 x float> %1877, float %1536, i64 3, !dbg !16
  %normalized55 = fmul <4 x float> %1878, %row.reciprocal53, !dbg !16
  %1879 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !17
  %1880 = extractelement <4 x half> %1879, i64 3, !dbg !17
  %1881 = extractelement <4 x half> %1879, i64 2, !dbg !17
  %1882 = extractelement <4 x half> %1879, i64 1, !dbg !17
  %1883 = extractelement <4 x half> %1879, i64 0, !dbg !17
  %1884 = insertelement <4 x float> poison, float %1422, i64 0, !dbg !16
  %1885 = insertelement <4 x float> %1884, float %1428, i64 1, !dbg !16
  %1886 = insertelement <4 x float> %1885, float %1423, i64 2, !dbg !16
  %1887 = insertelement <4 x float> %1886, float %1429, i64 3, !dbg !16
  %row.reciprocal56 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1401, !dbg !16
  %normalized57 = fmul <4 x float> %1887, %row.reciprocal56, !dbg !16
  %1888 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !17
  %1889 = extractelement <4 x half> %1888, i64 3, !dbg !17
  %1890 = extractelement <4 x half> %1888, i64 2, !dbg !17
  %1891 = extractelement <4 x half> %1888, i64 1, !dbg !17
  %1892 = extractelement <4 x half> %1888, i64 0, !dbg !17
  %1893 = insertelement <4 x float> poison, float %1311, i64 0, !dbg !16
  %1894 = insertelement <4 x float> %1893, float %1317, i64 1, !dbg !16
  %1895 = insertelement <4 x float> %1894, float %1312, i64 2, !dbg !16
  %1896 = insertelement <4 x float> %1895, float %1318, i64 3, !dbg !16
  %normalized58 = fmul <4 x float> %1896, %row.reciprocal56, !dbg !16
  %1897 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !17
  %1898 = extractelement <4 x half> %1897, i64 3, !dbg !17
  %1899 = extractelement <4 x half> %1897, i64 2, !dbg !17
  %1900 = extractelement <4 x half> %1897, i64 1, !dbg !17
  %1901 = extractelement <4 x half> %1897, i64 0, !dbg !17
  %1902 = shl nuw nsw i64 %26, 19, !dbg !18
  %address.thread1008 = call i32 asm sideeffect "", "=v,0"(i32 %33) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1009 = zext nneg i32 %address.thread1008 to i64, !dbg !8
  %address.local1010 = add i64 %27, %address.local1009, !dbg !18
  %1903 = lshr i64 %address.local1009, 2, !dbg !18
  %address.local1012 = and i64 %1903, 48, !dbg !18
  %address.local1013 = add i64 %address.local1010, %address.local1012, !dbg !18
  %address.local1014 = and i64 %address.local1009, 240, !dbg !18
  %address.local1015 = sub i64 %address.local1013, %address.local1014, !dbg !18
  %address.local1016 = add i64 %address.local1015, %28, !dbg !18
  %address.local1017 = shl nuw nsw i64 %address.local1016, 7, !dbg !18
  %address.local1018 = add nuw nsw i64 %1902, %address.local1017, !dbg !18
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1020 = lshr i64 %address.local1009, 2, !dbg !10
  %address.local1021 = and i64 %address.local1020, 12, !dbg !10
  %address.local1022 = or disjoint i64 %address.local1021, 96, !dbg !18
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1019, i64 %address.local1022, !dbg !18
  store half %1901, ptr addrspace(1) %address.local1023, align 2, !dbg !18
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1025 = or disjoint i64 %address.local1021, 97, !dbg !18
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1024, i64 %address.local1025, !dbg !18
  store half %1900, ptr addrspace(1) %address.local1026, align 2, !dbg !18
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1028 = or disjoint i64 %address.local1021, 98, !dbg !18
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1027, i64 %address.local1028, !dbg !18
  store half %1899, ptr addrspace(1) %address.local1029, align 2, !dbg !18
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1031 = or disjoint i64 %address.local1021, 99, !dbg !18
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1030, i64 %address.local1031, !dbg !18
  store half %1898, ptr addrspace(1) %address.local1032, align 2, !dbg !18
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1034 = or disjoint i64 %address.local1021, 112, !dbg !18
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local1034, !dbg !18
  store half %1892, ptr addrspace(1) %address.local1035, align 2, !dbg !18
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1037 = or disjoint i64 %address.local1021, 113, !dbg !18
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1036, i64 %address.local1037, !dbg !18
  store half %1891, ptr addrspace(1) %address.local1038, align 2, !dbg !18
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1040 = or disjoint i64 %address.local1021, 114, !dbg !18
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1040, !dbg !18
  store half %1890, ptr addrspace(1) %address.local1041, align 2, !dbg !18
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1043 = or i64 %address.local1020, 115, !dbg !18
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1042, i64 %address.local1043, !dbg !18
  store half %1889, ptr addrspace(1) %address.local1044, align 2, !dbg !18
  %address.local1045 = shl i64 %address.local1016, 7, !dbg !18
  %address.local1046 = add i64 %address.local1045, 8192, !dbg !18
  %address.local1047 = add nuw nsw i64 %1902, %address.local1046, !dbg !18
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1048, i64 %address.local1022, !dbg !18
  store half %1883, ptr addrspace(1) %address.local1049, align 2, !dbg !18
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1025, !dbg !18
  store half %1882, ptr addrspace(1) %address.local1051, align 2, !dbg !18
  %address.local1052 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1052, i64 %address.local1028, !dbg !18
  store half %1881, ptr addrspace(1) %address.local1053, align 2, !dbg !18
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1055 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1031, !dbg !18
  store half %1880, ptr addrspace(1) %address.local1055, align 2, !dbg !18
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1034, !dbg !18
  store half %1874, ptr addrspace(1) %address.local1057, align 2, !dbg !18
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1058, i64 %address.local1037, !dbg !18
  store half %1873, ptr addrspace(1) %address.local1059, align 2, !dbg !18
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1060, i64 %address.local1040, !dbg !18
  store half %1872, ptr addrspace(1) %address.local1061, align 2, !dbg !18
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1062, i64 %address.local1043, !dbg !18
  store half %1871, ptr addrspace(1) %address.local1063, align 2, !dbg !18
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64:1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-162611-z5os6mty/grid-000002")
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
