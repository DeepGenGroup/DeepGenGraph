; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-162611-z5os6mty/grid-000003/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-162611-z5os6mty/grid-000003/input.mlir"

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
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !5
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
  %128 = phi i64 [ %1286, %136 ], [ 0, %4 ], !dbg !7
  %129 = phi [2 x <8 x float>] [ %1271, %136 ], [ zeroinitializer, %4 ], !dbg !7
  %130 = phi [2 x <1 x float>] [ %965, %136 ], [ zeroinitializer, %4 ], !dbg !7
  %131 = phi [2 x <1 x float>] [ %1273, %136 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %132 = phi [2 x <8 x float>] [ %1277, %136 ], [ zeroinitializer, %4 ], !dbg !7
  %133 = phi [2 x <8 x float>] [ %1281, %136 ], [ zeroinitializer, %4 ], !dbg !7
  %134 = phi [2 x <8 x float>] [ %1285, %136 ], [ zeroinitializer, %4 ], !dbg !7
  %135 = icmp samesign ult i64 %128, 32, !dbg !7
  br i1 %135, label %136, label %1287, !dbg !7

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
  %kdma.offset1400 = add i32 %kdma.source1399, %kdma.intra.hi, !dbg !12
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
  %kdma.offset1434 = add i32 %kdma.source1433, %kdma.intra.hi, !dbg !12
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
  %kdma.offset1466 = add i32 %kdma.source1465, %kdma.intra.hi, !dbg !12
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
  %kdma.offset1500 = add i32 %kdma.source1499, %kdma.intra.hi, !dbg !12
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
  %171 = ptrtoint ptr addrspace(3) %address.local426 to i64, !dbg !14
  %172 = sub i64 %171, ptrtoint (ptr addrspace(3) @shm_0 to i64), !dbg !14
  %173 = lshr i64 %172, 6, !dbg !14
  %174 = and i64 %173, 8, !dbg !14
  %175 = xor i64 %172, %174, !dbg !14
  %176 = sub i64 %175, %172, !dbg !14
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local426, i64 %176, !dbg !14
  %177 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !14
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %177, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !14
  %179 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14
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
  %180 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local451, !dbg !14
  %181 = getelementptr [2 x i8], ptr addrspace(3) %180, i64 %address.local420, !dbg !14
  %182 = getelementptr [2 x i8], ptr addrspace(3) %181, i64 %address.local424, !dbg !14
  %address.local455 = getelementptr i8, ptr addrspace(3) %182, i64 64, !dbg !14
  %183 = ptrtoint ptr addrspace(3) %address.local455 to i64, !dbg !14
  %184 = sub i64 %183, ptrtoint (ptr addrspace(3) @shm_0 to i64), !dbg !14
  %185 = lshr i64 %184, 6, !dbg !14
  %186 = and i64 %185, 8, !dbg !14
  %187 = xor i64 %184, %186, !dbg !14
  %188 = sub i64 %187, %184, !dbg !14
  %kdma.swizzled1634 = getelementptr i8, ptr addrspace(3) %address.local455, i64 %188, !dbg !14
  %189 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1634, align 8, !dbg !14
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %179, <4 x float> %178) #9, !dbg !14
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %189, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1635 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 4096, !dbg !14
  %192 = load <4 x half>, ptr addrspace(3) %kdma.gep1635, align 8, !dbg !14
  %kdma.gep1636 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !14
  %193 = load <4 x half>, ptr addrspace(3) %kdma.gep1636, align 8, !dbg !14
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %192, <4 x float> %191) #9, !dbg !14
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %193, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1637 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !14
  %196 = load <4 x half>, ptr addrspace(3) %kdma.gep1637, align 8, !dbg !14
  %kdma.gep1638 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 128, !dbg !14
  %197 = load <4 x half>, ptr addrspace(3) %kdma.gep1638, align 8, !dbg !14
  %198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %196, <4 x float> %195) #9, !dbg !14
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %197, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1639 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 4224, !dbg !14
  %200 = load <4 x half>, ptr addrspace(3) %kdma.gep1639, align 8, !dbg !14
  %kdma.gep1640 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !14
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep1640, align 8, !dbg !14
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %200, <4 x float> %199) #9, !dbg !14
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %201, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1641 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !14
  %204 = load <4 x half>, ptr addrspace(3) %kdma.gep1641, align 8, !dbg !14
  %kdma.gep1642 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 1024, !dbg !14
  %205 = load <4 x half>, ptr addrspace(3) %kdma.gep1642, align 8, !dbg !14
  %206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %204, <4 x float> %203) #9, !dbg !14
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %205, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1643 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 5120, !dbg !14
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep1643, align 8, !dbg !14
  %kdma.gep1644 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !14
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1644, align 8, !dbg !14
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %208, <4 x float> %207) #9, !dbg !14
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %209, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1645 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !14
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1645, align 8, !dbg !14
  %kdma.gep1646 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 1152, !dbg !14
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1646, align 8, !dbg !14
  %214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %212, <4 x float> %211) #9, !dbg !14
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %213, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1647 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 5248, !dbg !14
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1647, align 8, !dbg !14
  %kdma.gep1648 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 0, !dbg !14
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1648, align 8, !dbg !14
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %216, <4 x float> %215) #9, !dbg !14
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %217, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1649 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !14
  %220 = load <4 x half>, ptr addrspace(3) %kdma.gep1649, align 8, !dbg !14
  %kdma.gep1650 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 0, !dbg !14
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep1650, align 8, !dbg !14
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %220, <4 x float> %219) #9, !dbg !14
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %221, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1651 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 4096, !dbg !14
  %224 = load <4 x half>, ptr addrspace(3) %kdma.gep1651, align 8, !dbg !14
  %kdma.gep1652 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !14
  %225 = load <4 x half>, ptr addrspace(3) %kdma.gep1652, align 8, !dbg !14
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %224, <4 x float> %223) #9, !dbg !14
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %225, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1653 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !14
  %228 = load <4 x half>, ptr addrspace(3) %kdma.gep1653, align 8, !dbg !14
  %kdma.gep1654 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 128, !dbg !14
  %229 = load <4 x half>, ptr addrspace(3) %kdma.gep1654, align 8, !dbg !14
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %228, <4 x float> %227) #9, !dbg !14
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %229, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1655 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 4224, !dbg !14
  %232 = load <4 x half>, ptr addrspace(3) %kdma.gep1655, align 8, !dbg !14
  %kdma.gep1656 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !14
  %233 = load <4 x half>, ptr addrspace(3) %kdma.gep1656, align 8, !dbg !14
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %232, <4 x float> %231) #9, !dbg !14
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %233, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1657 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !14
  %236 = load <4 x half>, ptr addrspace(3) %kdma.gep1657, align 8, !dbg !14
  %kdma.gep1658 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 1024, !dbg !14
  %237 = load <4 x half>, ptr addrspace(3) %kdma.gep1658, align 8, !dbg !14
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %236, <4 x float> %235) #9, !dbg !14
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %237, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1659 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 5120, !dbg !14
  %240 = load <4 x half>, ptr addrspace(3) %kdma.gep1659, align 8, !dbg !14
  %kdma.gep1660 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !14
  %241 = load <4 x half>, ptr addrspace(3) %kdma.gep1660, align 8, !dbg !14
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %240, <4 x float> %239) #9, !dbg !14
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %241, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1661 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !14
  %244 = load <4 x half>, ptr addrspace(3) %kdma.gep1661, align 8, !dbg !14
  %kdma.gep1662 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 1152, !dbg !14
  %245 = load <4 x half>, ptr addrspace(3) %kdma.gep1662, align 8, !dbg !14
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %244, <4 x float> %243) #9, !dbg !14
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %245, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1663 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 5248, !dbg !14
  %248 = load <4 x half>, ptr addrspace(3) %kdma.gep1663, align 8, !dbg !14
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %248, <4 x float> %247) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %250 = or disjoint i64 %140, 64, !dbg !12
  %251 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !12
  %252 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %253 = insertelement <2 x i64> poison, i64 %252, i64 0, !dbg !12
  %254 = bitcast <2 x i64> %253 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1228 = shufflevector <4 x i32> %254, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1229 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %251), !dbg !12
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
  %kdma.source1513 = or disjoint i64 %250, %kdma.source1512, !dbg !12
  %kdma.source1514 = trunc nuw i64 %kdma.source1513 to i32, !dbg !12
  %kdma.source1515 = shl nuw nsw i32 %kdma.source1514, 1, !dbg !12
  %kdma.offset1516 = add i32 %kdma.source1515, %kdma.intra, !dbg !12
  %kdma.lds1517 = add i32 %dma.buffer.lds1229, 0, !dbg !12
  %255 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1517, i32 %kdma.offset1516, <4 x i32> %dma.buffer.rsrc1228) #10, !dbg !12
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
  %kdma.source1529 = or disjoint i64 %250, %kdma.source1528, !dbg !12
  %kdma.source1530 = trunc nuw i64 %kdma.source1529 to i32, !dbg !12
  %kdma.source1531 = shl nuw nsw i32 %kdma.source1530, 1, !dbg !12
  %kdma.offset1532 = add i32 %kdma.source1531, %kdma.intra.hi, !dbg !12
  %kdma.lds1533 = add i32 %dma.buffer.lds1229, 512, !dbg !12
  %256 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1533, i32 %kdma.offset1532, <4 x i32> %dma.buffer.rsrc1228) #10, !dbg !12
  %257 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !12
  %258 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %259 = insertelement <2 x i64> poison, i64 %258, i64 0, !dbg !12
  %260 = bitcast <2 x i64> %259 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1225 = shufflevector <4 x i32> %260, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %257), !dbg !12
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
  %kdma.source1546 = or disjoint i64 %250, %kdma.source1545, !dbg !12
  %kdma.source1547 = trunc nuw i64 %kdma.source1546 to i32, !dbg !12
  %kdma.source1548 = shl nuw nsw i32 %kdma.source1547, 1, !dbg !12
  %kdma.offset1549 = add i32 %kdma.source1548, %kdma.intra, !dbg !12
  %kdma.lds1550 = add i32 %dma.buffer.lds1226, 0, !dbg !12
  %261 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1550, i32 %kdma.offset1549, <4 x i32> %dma.buffer.rsrc1225) #10, !dbg !12
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
  %kdma.source1563 = or disjoint i64 %250, %kdma.source1562, !dbg !12
  %kdma.source1564 = trunc nuw i64 %kdma.source1563 to i32, !dbg !12
  %kdma.source1565 = shl nuw nsw i32 %kdma.source1564, 1, !dbg !12
  %kdma.offset1566 = add i32 %kdma.source1565, %kdma.intra.hi, !dbg !12
  %kdma.lds1567 = add i32 %dma.buffer.lds1226, 512, !dbg !12
  %262 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1567, i32 %kdma.offset1566, <4 x i32> %dma.buffer.rsrc1225) #10, !dbg !12
  %kdma.gep1664 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !14
  %263 = load <4 x half>, ptr addrspace(3) %kdma.gep1664, align 8, !dbg !14
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %263, <4 x float> %190) #9, !dbg !14
  %kdma.gep1665 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !14
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep1665, align 8, !dbg !14
  %kdma.gep1666 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 8192, !dbg !14
  %266 = load <4 x half>, ptr addrspace(3) %kdma.gep1666, align 8, !dbg !14
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %265, <4 x float> %264) #9, !dbg !14
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %266, <4 x float> %194) #9, !dbg !14
  %kdma.gep1667 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 12288, !dbg !14
  %269 = load <4 x half>, ptr addrspace(3) %kdma.gep1667, align 8, !dbg !14
  %kdma.gep1668 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !14
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1668, align 8, !dbg !14
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %269, <4 x float> %268) #9, !dbg !14
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %270, <4 x float> %198) #9, !dbg !14
  %kdma.gep1669 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !14
  %273 = load <4 x half>, ptr addrspace(3) %kdma.gep1669, align 8, !dbg !14
  %kdma.gep1670 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 8320, !dbg !14
  %274 = load <4 x half>, ptr addrspace(3) %kdma.gep1670, align 8, !dbg !14
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %273, <4 x float> %272) #9, !dbg !14
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %274, <4 x float> %202) #9, !dbg !14
  %kdma.gep1671 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 12416, !dbg !14
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep1671, align 8, !dbg !14
  %kdma.gep1672 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !14
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep1672, align 8, !dbg !14
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %277, <4 x float> %276) #9, !dbg !14
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %278, <4 x float> %206) #9, !dbg !14
  %kdma.gep1673 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !14
  %281 = load <4 x half>, ptr addrspace(3) %kdma.gep1673, align 8, !dbg !14
  %kdma.gep1674 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 9216, !dbg !14
  %282 = load <4 x half>, ptr addrspace(3) %kdma.gep1674, align 8, !dbg !14
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %281, <4 x float> %280) #9, !dbg !14
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %282, <4 x float> %210) #9, !dbg !14
  %kdma.gep1675 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 13312, !dbg !14
  %285 = load <4 x half>, ptr addrspace(3) %kdma.gep1675, align 8, !dbg !14
  %kdma.gep1676 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !14
  %286 = load <4 x half>, ptr addrspace(3) %kdma.gep1676, align 8, !dbg !14
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %285, <4 x float> %284) #9, !dbg !14
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %286, <4 x float> %214) #9, !dbg !14
  %kdma.gep1677 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !14
  %289 = load <4 x half>, ptr addrspace(3) %kdma.gep1677, align 8, !dbg !14
  %kdma.gep1678 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 9344, !dbg !14
  %290 = load <4 x half>, ptr addrspace(3) %kdma.gep1678, align 8, !dbg !14
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %289, <4 x float> %288) #9, !dbg !14
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %290, <4 x float> %218) #9, !dbg !14
  %kdma.gep1679 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 13440, !dbg !14
  %293 = load <4 x half>, ptr addrspace(3) %kdma.gep1679, align 8, !dbg !14
  %kdma.gep1680 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !14
  %294 = load <4 x half>, ptr addrspace(3) %kdma.gep1680, align 8, !dbg !14
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %293, <4 x float> %292) #9, !dbg !14
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %294, <4 x float> %222) #9, !dbg !14
  %kdma.gep1681 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !14
  %297 = load <4 x half>, ptr addrspace(3) %kdma.gep1681, align 8, !dbg !14
  %kdma.gep1682 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 8192, !dbg !14
  %298 = load <4 x half>, ptr addrspace(3) %kdma.gep1682, align 8, !dbg !14
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %297, <4 x float> %296) #9, !dbg !14
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %298, <4 x float> %226) #9, !dbg !14
  %kdma.gep1683 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 12288, !dbg !14
  %301 = load <4 x half>, ptr addrspace(3) %kdma.gep1683, align 8, !dbg !14
  %kdma.gep1684 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !14
  %302 = load <4 x half>, ptr addrspace(3) %kdma.gep1684, align 8, !dbg !14
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %301, <4 x float> %300) #9, !dbg !14
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %302, <4 x float> %230) #9, !dbg !14
  %kdma.gep1685 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !14
  %305 = load <4 x half>, ptr addrspace(3) %kdma.gep1685, align 8, !dbg !14
  %kdma.gep1686 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 8320, !dbg !14
  %306 = load <4 x half>, ptr addrspace(3) %kdma.gep1686, align 8, !dbg !14
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %305, <4 x float> %304) #9, !dbg !14
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %306, <4 x float> %234) #9, !dbg !14
  %kdma.gep1687 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 12416, !dbg !14
  %309 = load <4 x half>, ptr addrspace(3) %kdma.gep1687, align 8, !dbg !14
  %kdma.gep1688 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !14
  %310 = load <4 x half>, ptr addrspace(3) %kdma.gep1688, align 8, !dbg !14
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %309, <4 x float> %308) #9, !dbg !14
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %310, <4 x float> %238) #9, !dbg !14
  %kdma.gep1689 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !14
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1689, align 8, !dbg !14
  %kdma.gep1690 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 9216, !dbg !14
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1690, align 8, !dbg !14
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %313, <4 x float> %312) #9, !dbg !14
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %314, <4 x float> %242) #9, !dbg !14
  %kdma.gep1691 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 13312, !dbg !14
  %317 = load <4 x half>, ptr addrspace(3) %kdma.gep1691, align 8, !dbg !14
  %kdma.gep1692 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !14
  %318 = load <4 x half>, ptr addrspace(3) %kdma.gep1692, align 8, !dbg !14
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %317, <4 x float> %316) #9, !dbg !14
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %318, <4 x float> %246) #9, !dbg !14
  %kdma.gep1693 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !14
  %321 = load <4 x half>, ptr addrspace(3) %kdma.gep1693, align 8, !dbg !14
  %kdma.gep1694 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 9344, !dbg !14
  %322 = load <4 x half>, ptr addrspace(3) %kdma.gep1694, align 8, !dbg !14
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %321, <4 x float> %320) #9, !dbg !14
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %322, <4 x float> %249) #9, !dbg !14
  %kdma.gep1695 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 13440, !dbg !14
  %325 = load <4 x half>, ptr addrspace(3) %kdma.gep1695, align 8, !dbg !14
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %325, <4 x float> %324) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %327 = or disjoint i64 %140, 96, !dbg !12
  %328 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local402), !dbg !12
  %329 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %330 = insertelement <2 x i64> poison, i64 %329, i64 0, !dbg !12
  %331 = bitcast <2 x i64> %330 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1222 = shufflevector <4 x i32> %331, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1223 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %328), !dbg !12
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
  %kdma.source1579 = or disjoint i64 %327, %kdma.source1578, !dbg !12
  %kdma.source1580 = trunc nuw i64 %kdma.source1579 to i32, !dbg !12
  %kdma.source1581 = shl nuw nsw i32 %kdma.source1580, 1, !dbg !12
  %kdma.offset1582 = add i32 %kdma.source1581, %kdma.intra, !dbg !12
  %kdma.lds1583 = add i32 %dma.buffer.lds1223, 0, !dbg !12
  %332 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1583, i32 %kdma.offset1582, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !12
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
  %kdma.source1595 = or disjoint i64 %327, %kdma.source1594, !dbg !12
  %kdma.source1596 = trunc nuw i64 %kdma.source1595 to i32, !dbg !12
  %kdma.source1597 = shl nuw nsw i32 %kdma.source1596, 1, !dbg !12
  %kdma.offset1598 = add i32 %kdma.source1597, %kdma.intra.hi, !dbg !12
  %kdma.lds1599 = add i32 %dma.buffer.lds1223, 512, !dbg !12
  %333 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1599, i32 %kdma.offset1598, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !12
  %334 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local406), !dbg !12
  %335 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %336 = insertelement <2 x i64> poison, i64 %335, i64 0, !dbg !12
  %337 = bitcast <2 x i64> %336 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1219 = shufflevector <4 x i32> %337, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1220 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %334), !dbg !12
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
  %kdma.source1612 = or disjoint i64 %327, %kdma.source1611, !dbg !12
  %kdma.source1613 = trunc nuw i64 %kdma.source1612 to i32, !dbg !12
  %kdma.source1614 = shl nuw nsw i32 %kdma.source1613, 1, !dbg !12
  %kdma.offset1615 = add i32 %kdma.source1614, %kdma.intra, !dbg !12
  %kdma.lds1616 = add i32 %dma.buffer.lds1220, 0, !dbg !12
  %338 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1616, i32 %kdma.offset1615, <4 x i32> %dma.buffer.rsrc1219) #10, !dbg !12
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
  %kdma.source1629 = or disjoint i64 %327, %kdma.source1628, !dbg !12
  %kdma.source1630 = trunc nuw i64 %kdma.source1629 to i32, !dbg !12
  %kdma.source1631 = shl nuw nsw i32 %kdma.source1630, 1, !dbg !12
  %kdma.offset1632 = add i32 %kdma.source1631, %kdma.intra.hi, !dbg !12
  %kdma.lds1633 = add i32 %dma.buffer.lds1220, 512, !dbg !12
  %339 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1633, i32 %kdma.offset1632, <4 x i32> %dma.buffer.rsrc1219) #10, !dbg !12
  %kdma.gep1696 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 0, !dbg !14
  %340 = load <4 x half>, ptr addrspace(3) %kdma.gep1696, align 8, !dbg !14
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %340, <4 x float> %267) #9, !dbg !14
  %kdma.gep1697 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !14
  %342 = load <4 x half>, ptr addrspace(3) %kdma.gep1697, align 8, !dbg !14
  %kdma.gep1698 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 0, !dbg !14
  %343 = load <4 x half>, ptr addrspace(3) %kdma.gep1698, align 8, !dbg !14
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %342, <4 x float> %341) #9, !dbg !14
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %343, <4 x float> %271) #9, !dbg !14
  %kdma.gep1699 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 4096, !dbg !14
  %346 = load <4 x half>, ptr addrspace(3) %kdma.gep1699, align 8, !dbg !14
  %kdma.gep1700 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !14
  %347 = load <4 x half>, ptr addrspace(3) %kdma.gep1700, align 8, !dbg !14
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %346, <4 x float> %345) #9, !dbg !14
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %347, <4 x float> %275) #9, !dbg !14
  %kdma.gep1701 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !14
  %350 = load <4 x half>, ptr addrspace(3) %kdma.gep1701, align 8, !dbg !14
  %kdma.gep1702 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 128, !dbg !14
  %351 = load <4 x half>, ptr addrspace(3) %kdma.gep1702, align 8, !dbg !14
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %350, <4 x float> %349) #9, !dbg !14
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %351, <4 x float> %279) #9, !dbg !14
  %kdma.gep1703 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 4224, !dbg !14
  %354 = load <4 x half>, ptr addrspace(3) %kdma.gep1703, align 8, !dbg !14
  %kdma.gep1704 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !14
  %355 = load <4 x half>, ptr addrspace(3) %kdma.gep1704, align 8, !dbg !14
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %354, <4 x float> %353) #9, !dbg !14
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %355, <4 x float> %283) #9, !dbg !14
  %kdma.gep1705 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !14
  %358 = load <4 x half>, ptr addrspace(3) %kdma.gep1705, align 8, !dbg !14
  %kdma.gep1706 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 1024, !dbg !14
  %359 = load <4 x half>, ptr addrspace(3) %kdma.gep1706, align 8, !dbg !14
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %358, <4 x float> %357) #9, !dbg !14
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %359, <4 x float> %287) #9, !dbg !14
  %kdma.gep1707 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 5120, !dbg !14
  %362 = load <4 x half>, ptr addrspace(3) %kdma.gep1707, align 8, !dbg !14
  %kdma.gep1708 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !14
  %363 = load <4 x half>, ptr addrspace(3) %kdma.gep1708, align 8, !dbg !14
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %362, <4 x float> %361) #9, !dbg !14
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %363, <4 x float> %291) #9, !dbg !14
  %kdma.gep1709 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !14
  %366 = load <4 x half>, ptr addrspace(3) %kdma.gep1709, align 8, !dbg !14
  %kdma.gep1710 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 1152, !dbg !14
  %367 = load <4 x half>, ptr addrspace(3) %kdma.gep1710, align 8, !dbg !14
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %366, <4 x float> %365) #9, !dbg !14
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %367, <4 x float> %295) #9, !dbg !14
  %kdma.gep1711 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 5248, !dbg !14
  %370 = load <4 x half>, ptr addrspace(3) %kdma.gep1711, align 8, !dbg !14
  %kdma.gep1712 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 0, !dbg !14
  %371 = load <4 x half>, ptr addrspace(3) %kdma.gep1712, align 8, !dbg !14
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %370, <4 x float> %369) #9, !dbg !14
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %371, <4 x float> %299) #9, !dbg !14
  %kdma.gep1713 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !14
  %374 = load <4 x half>, ptr addrspace(3) %kdma.gep1713, align 8, !dbg !14
  %kdma.gep1714 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 0, !dbg !14
  %375 = load <4 x half>, ptr addrspace(3) %kdma.gep1714, align 8, !dbg !14
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %374, <4 x float> %373) #9, !dbg !14
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %375, <4 x float> %303) #9, !dbg !14
  %kdma.gep1715 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 4096, !dbg !14
  %378 = load <4 x half>, ptr addrspace(3) %kdma.gep1715, align 8, !dbg !14
  %kdma.gep1716 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !14
  %379 = load <4 x half>, ptr addrspace(3) %kdma.gep1716, align 8, !dbg !14
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %378, <4 x float> %377) #9, !dbg !14
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %379, <4 x float> %307) #9, !dbg !14
  %kdma.gep1717 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !14
  %382 = load <4 x half>, ptr addrspace(3) %kdma.gep1717, align 8, !dbg !14
  %kdma.gep1718 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 128, !dbg !14
  %383 = load <4 x half>, ptr addrspace(3) %kdma.gep1718, align 8, !dbg !14
  %384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %382, <4 x float> %381) #9, !dbg !14
  %385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %383, <4 x float> %311) #9, !dbg !14
  %kdma.gep1719 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 4224, !dbg !14
  %386 = load <4 x half>, ptr addrspace(3) %kdma.gep1719, align 8, !dbg !14
  %kdma.gep1720 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !14
  %387 = load <4 x half>, ptr addrspace(3) %kdma.gep1720, align 8, !dbg !14
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %386, <4 x float> %385) #9, !dbg !14
  %389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %387, <4 x float> %315) #9, !dbg !14
  %kdma.gep1721 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !14
  %390 = load <4 x half>, ptr addrspace(3) %kdma.gep1721, align 8, !dbg !14
  %kdma.gep1722 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 1024, !dbg !14
  %391 = load <4 x half>, ptr addrspace(3) %kdma.gep1722, align 8, !dbg !14
  %392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %390, <4 x float> %389) #9, !dbg !14
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %391, <4 x float> %319) #9, !dbg !14
  %kdma.gep1723 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 5120, !dbg !14
  %394 = load <4 x half>, ptr addrspace(3) %kdma.gep1723, align 8, !dbg !14
  %kdma.gep1724 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !14
  %395 = load <4 x half>, ptr addrspace(3) %kdma.gep1724, align 8, !dbg !14
  %396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %394, <4 x float> %393) #9, !dbg !14
  %397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %395, <4 x float> %323) #9, !dbg !14
  %kdma.gep1725 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !14
  %398 = load <4 x half>, ptr addrspace(3) %kdma.gep1725, align 8, !dbg !14
  %kdma.gep1726 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 1152, !dbg !14
  %399 = load <4 x half>, ptr addrspace(3) %kdma.gep1726, align 8, !dbg !14
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %398, <4 x float> %397) #9, !dbg !14
  %401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %399, <4 x float> %326) #9, !dbg !14
  %kdma.gep1727 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 5248, !dbg !14
  %402 = load <4 x half>, ptr addrspace(3) %kdma.gep1727, align 8, !dbg !14
  %403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %402, <4 x float> %401) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %404 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !15
  %address.local784 = shl nuw nsw i64 %address.local366, 5, !dbg !15
  %address.local785 = and i64 %address.local784, 8064, !dbg !15
  %address.local786 = shl nuw nsw i64 %address.local366, 3, !dbg !15
  %address.local787 = and i64 %address.local786, 24, !dbg !15
  %address.local788 = or disjoint i64 %address.local785, %address.local787, !dbg !15
  %405 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx12861287 = or disjoint i64 %140, %address.local788, !dbg !15
  %.idx12861287.tr = trunc nuw i64 %.idx12861287 to i32, !dbg !15
  %dma.buffer.offset1215 = shl nuw nsw i32 %.idx12861287.tr, 1, !dbg !15
  %406 = insertelement <2 x i64> poison, i64 %405, i64 0, !dbg !15
  %407 = bitcast <2 x i64> %406 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1216 = shufflevector <4 x i32> %407, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1217 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %404), !dbg !15
  %408 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1217, i32 %dma.buffer.offset1215, <4 x i32> %dma.buffer.rsrc1216) #10, !dbg !15
  %409 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !15
  %address.local790 = shl nuw nsw i64 %address.local383, 5, !dbg !15
  %address.local791 = and i64 %address.local790, 16256, !dbg !15
  %address.local792 = shl nuw nsw i64 %address.local366, 3, !dbg !15
  %address.local793 = and i64 %address.local792, 24, !dbg !15
  %address.local794 = or disjoint i64 %address.local791, %address.local793, !dbg !15
  %410 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx12881289 = or disjoint i64 %140, %address.local794, !dbg !15
  %.idx12881289.tr = trunc nuw i64 %.idx12881289 to i32, !dbg !15
  %dma.buffer.offset1212 = shl nuw nsw i32 %.idx12881289.tr, 1, !dbg !15
  %411 = insertelement <2 x i64> poison, i64 %410, i64 0, !dbg !15
  %412 = bitcast <2 x i64> %411 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1213 = shufflevector <4 x i32> %412, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1214 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %409), !dbg !15
  %413 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1214, i32 %dma.buffer.offset1212, <4 x i32> %dma.buffer.rsrc1213) #10, !dbg !15
  %kdma.gep1728 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !14
  %414 = load <4 x half>, ptr addrspace(3) %kdma.gep1728, align 8, !dbg !14
  %415 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %414, <4 x float> %344) #9, !dbg !14
  %kdma.gep1729 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !14
  %416 = load <4 x half>, ptr addrspace(3) %kdma.gep1729, align 8, !dbg !14
  %kdma.gep1730 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 8192, !dbg !14
  %417 = load <4 x half>, ptr addrspace(3) %kdma.gep1730, align 8, !dbg !14
  %418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %416, <4 x float> %415) #9, !dbg !14
  %419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %417, <4 x float> %348) #9, !dbg !14
  %420 = extractelement <4 x float> %418, i64 0, !dbg !19
  %421 = call float @llvm.maxnum.f32(float %420, float 0xFFF0000000000000), !dbg !20
  %422 = extractelement <4 x float> %418, i64 1, !dbg !19
  %423 = call float @llvm.maxnum.f32(float %421, float %422), !dbg !20
  %424 = extractelement <4 x float> %418, i64 2, !dbg !19
  %425 = call float @llvm.maxnum.f32(float %423, float %424), !dbg !20
  %426 = extractelement <4 x float> %418, i64 3, !dbg !19
  %427 = call float @llvm.maxnum.f32(float %425, float %426), !dbg !20
  %kdma.gep1731 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 12288, !dbg !14
  %428 = load <4 x half>, ptr addrspace(3) %kdma.gep1731, align 8, !dbg !14
  %kdma.gep1732 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !14
  %429 = load <4 x half>, ptr addrspace(3) %kdma.gep1732, align 8, !dbg !14
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %428, <4 x float> %419) #9, !dbg !14
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %429, <4 x float> %352) #9, !dbg !14
  %432 = extractelement <4 x float> %430, i64 0, !dbg !19
  %433 = call float @llvm.maxnum.f32(float %427, float %432), !dbg !20
  %434 = extractelement <4 x float> %430, i64 1, !dbg !19
  %435 = call float @llvm.maxnum.f32(float %433, float %434), !dbg !20
  %436 = extractelement <4 x float> %430, i64 2, !dbg !19
  %437 = call float @llvm.maxnum.f32(float %435, float %436), !dbg !20
  %438 = extractelement <4 x float> %430, i64 3, !dbg !19
  %439 = call float @llvm.maxnum.f32(float %437, float %438), !dbg !20
  %kdma.gep1733 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !14
  %440 = load <4 x half>, ptr addrspace(3) %kdma.gep1733, align 8, !dbg !14
  %kdma.gep1734 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 8320, !dbg !14
  %441 = load <4 x half>, ptr addrspace(3) %kdma.gep1734, align 8, !dbg !14
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %440, <4 x float> %431) #9, !dbg !14
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %441, <4 x float> %356) #9, !dbg !14
  %444 = extractelement <4 x float> %442, i64 0, !dbg !19
  %445 = call float @llvm.maxnum.f32(float %439, float %444), !dbg !20
  %446 = extractelement <4 x float> %442, i64 1, !dbg !19
  %447 = call float @llvm.maxnum.f32(float %445, float %446), !dbg !20
  %448 = extractelement <4 x float> %442, i64 2, !dbg !19
  %449 = call float @llvm.maxnum.f32(float %447, float %448), !dbg !20
  %450 = extractelement <4 x float> %442, i64 3, !dbg !19
  %451 = call float @llvm.maxnum.f32(float %449, float %450), !dbg !20
  %kdma.gep1735 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 12416, !dbg !14
  %452 = load <4 x half>, ptr addrspace(3) %kdma.gep1735, align 8, !dbg !14
  %kdma.gep1736 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !14
  %453 = load <4 x half>, ptr addrspace(3) %kdma.gep1736, align 8, !dbg !14
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %452, <4 x float> %443) #9, !dbg !14
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %453, <4 x float> %360) #9, !dbg !14
  %456 = extractelement <4 x float> %454, i64 0, !dbg !19
  %457 = call float @llvm.maxnum.f32(float %451, float %456), !dbg !20
  %458 = extractelement <4 x float> %454, i64 1, !dbg !19
  %459 = call float @llvm.maxnum.f32(float %457, float %458), !dbg !20
  %460 = extractelement <4 x float> %454, i64 2, !dbg !19
  %461 = call float @llvm.maxnum.f32(float %459, float %460), !dbg !20
  %462 = extractelement <4 x float> %454, i64 3, !dbg !19
  %463 = call float @llvm.maxnum.f32(float %461, float %462), !dbg !20
  %kdma.gep1737 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !14
  %464 = load <4 x half>, ptr addrspace(3) %kdma.gep1737, align 8, !dbg !14
  %kdma.gep1738 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 9216, !dbg !14
  %465 = load <4 x half>, ptr addrspace(3) %kdma.gep1738, align 8, !dbg !14
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %464, <4 x float> %455) #9, !dbg !14
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %465, <4 x float> %364) #9, !dbg !14
  %468 = extractelement <4 x float> %466, i64 0, !dbg !19
  %469 = call float @llvm.maxnum.f32(float %463, float %468), !dbg !20
  %470 = extractelement <4 x float> %466, i64 1, !dbg !19
  %471 = call float @llvm.maxnum.f32(float %469, float %470), !dbg !20
  %472 = extractelement <4 x float> %466, i64 2, !dbg !19
  %473 = call float @llvm.maxnum.f32(float %471, float %472), !dbg !20
  %474 = extractelement <4 x float> %466, i64 3, !dbg !19
  %475 = call float @llvm.maxnum.f32(float %473, float %474), !dbg !20
  %kdma.gep1739 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 13312, !dbg !14
  %476 = load <4 x half>, ptr addrspace(3) %kdma.gep1739, align 8, !dbg !14
  %kdma.gep1740 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !14
  %477 = load <4 x half>, ptr addrspace(3) %kdma.gep1740, align 8, !dbg !14
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %476, <4 x float> %467) #9, !dbg !14
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %477, <4 x float> %368) #9, !dbg !14
  %480 = extractelement <4 x float> %478, i64 0, !dbg !19
  %481 = call float @llvm.maxnum.f32(float %475, float %480), !dbg !20
  %482 = extractelement <4 x float> %478, i64 1, !dbg !19
  %483 = call float @llvm.maxnum.f32(float %481, float %482), !dbg !20
  %484 = extractelement <4 x float> %478, i64 2, !dbg !19
  %485 = call float @llvm.maxnum.f32(float %483, float %484), !dbg !20
  %486 = extractelement <4 x float> %478, i64 3, !dbg !19
  %487 = call float @llvm.maxnum.f32(float %485, float %486), !dbg !20
  %kdma.gep1741 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !14
  %488 = load <4 x half>, ptr addrspace(3) %kdma.gep1741, align 8, !dbg !14
  %kdma.gep1742 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 9344, !dbg !14
  %489 = load <4 x half>, ptr addrspace(3) %kdma.gep1742, align 8, !dbg !14
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %488, <4 x float> %479) #9, !dbg !14
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %489, <4 x float> %372) #9, !dbg !14
  %492 = extractelement <4 x float> %490, i64 0, !dbg !19
  %493 = call float @llvm.maxnum.f32(float %487, float %492), !dbg !20
  %494 = extractelement <4 x float> %490, i64 1, !dbg !19
  %495 = call float @llvm.maxnum.f32(float %493, float %494), !dbg !20
  %496 = extractelement <4 x float> %490, i64 2, !dbg !19
  %497 = call float @llvm.maxnum.f32(float %495, float %496), !dbg !20
  %498 = extractelement <4 x float> %490, i64 3, !dbg !19
  %499 = call float @llvm.maxnum.f32(float %497, float %498), !dbg !20
  %kdma.gep1743 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 13440, !dbg !14
  %500 = load <4 x half>, ptr addrspace(3) %kdma.gep1743, align 8, !dbg !14
  %kdma.gep1744 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !14
  %501 = load <4 x half>, ptr addrspace(3) %kdma.gep1744, align 8, !dbg !14
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %500, <4 x float> %491) #9, !dbg !14
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %501, <4 x float> %376) #9, !dbg !14
  %504 = extractelement <4 x float> %502, i64 0, !dbg !19
  %505 = call float @llvm.maxnum.f32(float %499, float %504), !dbg !20
  %506 = extractelement <4 x float> %502, i64 1, !dbg !19
  %507 = call float @llvm.maxnum.f32(float %505, float %506), !dbg !20
  %508 = extractelement <4 x float> %502, i64 2, !dbg !19
  %509 = call float @llvm.maxnum.f32(float %507, float %508), !dbg !20
  %510 = extractelement <4 x float> %502, i64 3, !dbg !19
  %511 = call float @llvm.maxnum.f32(float %509, float %510), !dbg !20
  %kdma.gep1745 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !14
  %512 = load <4 x half>, ptr addrspace(3) %kdma.gep1745, align 8, !dbg !14
  %kdma.gep1746 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 8192, !dbg !14
  %513 = load <4 x half>, ptr addrspace(3) %kdma.gep1746, align 8, !dbg !14
  %514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %512, <4 x float> %503) #9, !dbg !14
  %515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %513, <4 x float> %380) #9, !dbg !14
  %516 = extractelement <4 x float> %514, i64 0, !dbg !19
  %517 = extractelement <4 x float> %514, i64 1, !dbg !19
  %518 = extractelement <4 x float> %514, i64 2, !dbg !19
  %519 = extractelement <4 x float> %514, i64 3, !dbg !19
  %kdma.gep1747 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 12288, !dbg !14
  %520 = load <4 x half>, ptr addrspace(3) %kdma.gep1747, align 8, !dbg !14
  %kdma.gep1748 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !14
  %521 = load <4 x half>, ptr addrspace(3) %kdma.gep1748, align 8, !dbg !14
  %522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %520, <4 x float> %515) #9, !dbg !14
  %523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %521, <4 x float> %384) #9, !dbg !14
  %524 = extractelement <4 x float> %522, i64 0, !dbg !19
  %525 = extractelement <4 x float> %522, i64 1, !dbg !19
  %526 = extractelement <4 x float> %522, i64 2, !dbg !19
  %527 = extractelement <4 x float> %522, i64 3, !dbg !19
  %kdma.gep1749 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !14
  %528 = load <4 x half>, ptr addrspace(3) %kdma.gep1749, align 8, !dbg !14
  %kdma.gep1750 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 8320, !dbg !14
  %529 = load <4 x half>, ptr addrspace(3) %kdma.gep1750, align 8, !dbg !14
  %530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %528, <4 x float> %523) #9, !dbg !14
  %531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %529, <4 x float> %388) #9, !dbg !14
  %532 = extractelement <4 x float> %530, i64 0, !dbg !19
  %533 = extractelement <4 x float> %530, i64 1, !dbg !19
  %534 = extractelement <4 x float> %530, i64 2, !dbg !19
  %535 = extractelement <4 x float> %530, i64 3, !dbg !19
  %kdma.gep1751 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 12416, !dbg !14
  %536 = load <4 x half>, ptr addrspace(3) %kdma.gep1751, align 8, !dbg !14
  %kdma.gep1752 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !14
  %537 = load <4 x half>, ptr addrspace(3) %kdma.gep1752, align 8, !dbg !14
  %538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %536, <4 x float> %531) #9, !dbg !14
  %539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %537, <4 x float> %392) #9, !dbg !14
  %540 = extractelement <4 x float> %538, i64 0, !dbg !19
  %541 = extractelement <4 x float> %538, i64 1, !dbg !19
  %542 = extractelement <4 x float> %538, i64 2, !dbg !19
  %543 = extractelement <4 x float> %538, i64 3, !dbg !19
  %kdma.gep1753 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !14
  %544 = load <4 x half>, ptr addrspace(3) %kdma.gep1753, align 8, !dbg !14
  %kdma.gep1754 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 9216, !dbg !14
  %545 = load <4 x half>, ptr addrspace(3) %kdma.gep1754, align 8, !dbg !14
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %544, <4 x float> %539) #9, !dbg !14
  %547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %545, <4 x float> %396) #9, !dbg !14
  %548 = extractelement <4 x float> %546, i64 0, !dbg !19
  %549 = extractelement <4 x float> %546, i64 1, !dbg !19
  %550 = extractelement <4 x float> %546, i64 2, !dbg !19
  %551 = extractelement <4 x float> %546, i64 3, !dbg !19
  %kdma.gep1755 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 13312, !dbg !14
  %552 = load <4 x half>, ptr addrspace(3) %kdma.gep1755, align 8, !dbg !14
  %kdma.gep1756 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !14
  %553 = load <4 x half>, ptr addrspace(3) %kdma.gep1756, align 8, !dbg !14
  %554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %552, <4 x float> %547) #9, !dbg !14
  %555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %553, <4 x float> %400) #9, !dbg !14
  %556 = extractelement <4 x float> %554, i64 0, !dbg !19
  %557 = extractelement <4 x float> %554, i64 1, !dbg !19
  %558 = extractelement <4 x float> %554, i64 2, !dbg !19
  %559 = extractelement <4 x float> %554, i64 3, !dbg !19
  %kdma.gep1757 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !14
  %560 = load <4 x half>, ptr addrspace(3) %kdma.gep1757, align 8, !dbg !14
  %kdma.gep1758 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 9344, !dbg !14
  %561 = load <4 x half>, ptr addrspace(3) %kdma.gep1758, align 8, !dbg !14
  %562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %560, <4 x float> %555) #9, !dbg !14
  %563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %561, <4 x float> %403) #9, !dbg !14
  %564 = extractelement <4 x float> %562, i64 0, !dbg !19
  %565 = extractelement <4 x float> %562, i64 1, !dbg !19
  %566 = extractelement <4 x float> %562, i64 2, !dbg !19
  %567 = extractelement <4 x float> %562, i64 3, !dbg !19
  %kdma.gep1759 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 13440, !dbg !14
  %568 = load <4 x half>, ptr addrspace(3) %kdma.gep1759, align 8, !dbg !14
  %569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %568, <4 x float> %563) #9, !dbg !14
  %570 = extractelement <4 x float> %569, i64 0, !dbg !19
  %571 = extractelement <4 x float> %569, i64 1, !dbg !19
  %572 = extractelement <4 x float> %569, i64 2, !dbg !19
  %573 = extractelement <4 x float> %569, i64 3, !dbg !19
  %574 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %575 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %574), !dbg !20
  %576 = add nuw nsw i32 %575, 64, !dbg !20
  %577 = xor i32 %575, 16, !dbg !20
  %578 = and i32 %576, 192, !dbg !20
  %579 = icmp samesign ult i32 %577, %578, !dbg !20
  %580 = select i1 %579, i32 %577, i32 %575, !dbg !20
  %581 = shl nuw nsw i32 %580, 2, !dbg !20
  %582 = bitcast float %511 to i32, !dbg !20
  %583 = call i32 @llvm.amdgcn.ds.bpermute(i32 %581, i32 %582), !dbg !20
  %584 = bitcast i32 %583 to float, !dbg !20
  %585 = call float @llvm.maxnum.f32(float %511, float %584), !dbg !20
  %586 = xor i32 %575, 32, !dbg !20
  %587 = icmp samesign ult i32 %586, %578, !dbg !20
  %588 = select i1 %587, i32 %586, i32 %575, !dbg !20
  %589 = shl nuw nsw i32 %588, 2, !dbg !20
  %590 = bitcast float %585 to i32, !dbg !20
  %591 = call i32 @llvm.amdgcn.ds.bpermute(i32 %589, i32 %590), !dbg !20
  %592 = bitcast i32 %591 to float, !dbg !20
  %593 = call float @llvm.maxnum.f32(float %585, float %592), !dbg !20
  %594 = extractvalue [2 x <1 x float>] %131, 0, !dbg !20
  %595 = insertelement <1 x float> poison, float %593, i64 0, !dbg !20
  %596 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %595, <1 x float> %594), !dbg !20
  %597 = call float @llvm.maxnum.f32(float %516, float 0xFFF0000000000000), !dbg !20
  %598 = call float @llvm.maxnum.f32(float %597, float %517), !dbg !20
  %599 = call float @llvm.maxnum.f32(float %598, float %518), !dbg !20
  %600 = call float @llvm.maxnum.f32(float %599, float %519), !dbg !20
  %601 = call float @llvm.maxnum.f32(float %600, float %524), !dbg !20
  %602 = call float @llvm.maxnum.f32(float %601, float %525), !dbg !20
  %603 = call float @llvm.maxnum.f32(float %602, float %526), !dbg !20
  %604 = call float @llvm.maxnum.f32(float %603, float %527), !dbg !20
  %605 = call float @llvm.maxnum.f32(float %604, float %532), !dbg !20
  %606 = call float @llvm.maxnum.f32(float %605, float %533), !dbg !20
  %607 = call float @llvm.maxnum.f32(float %606, float %534), !dbg !20
  %608 = call float @llvm.maxnum.f32(float %607, float %535), !dbg !20
  %609 = call float @llvm.maxnum.f32(float %608, float %540), !dbg !20
  %610 = call float @llvm.maxnum.f32(float %609, float %541), !dbg !20
  %611 = call float @llvm.maxnum.f32(float %610, float %542), !dbg !20
  %612 = call float @llvm.maxnum.f32(float %611, float %543), !dbg !20
  %613 = call float @llvm.maxnum.f32(float %612, float %548), !dbg !20
  %614 = call float @llvm.maxnum.f32(float %613, float %549), !dbg !20
  %615 = call float @llvm.maxnum.f32(float %614, float %550), !dbg !20
  %616 = call float @llvm.maxnum.f32(float %615, float %551), !dbg !20
  %617 = call float @llvm.maxnum.f32(float %616, float %556), !dbg !20
  %618 = call float @llvm.maxnum.f32(float %617, float %557), !dbg !20
  %619 = call float @llvm.maxnum.f32(float %618, float %558), !dbg !20
  %620 = call float @llvm.maxnum.f32(float %619, float %559), !dbg !20
  %621 = call float @llvm.maxnum.f32(float %620, float %564), !dbg !20
  %622 = call float @llvm.maxnum.f32(float %621, float %565), !dbg !20
  %623 = call float @llvm.maxnum.f32(float %622, float %566), !dbg !20
  %624 = call float @llvm.maxnum.f32(float %623, float %567), !dbg !20
  %625 = call float @llvm.maxnum.f32(float %624, float %570), !dbg !20
  %626 = call float @llvm.maxnum.f32(float %625, float %571), !dbg !20
  %627 = call float @llvm.maxnum.f32(float %626, float %572), !dbg !20
  %628 = call float @llvm.maxnum.f32(float %627, float %573), !dbg !20
  %629 = bitcast float %628 to i32, !dbg !20
  %630 = call i32 @llvm.amdgcn.ds.bpermute(i32 %581, i32 %629), !dbg !20
  %631 = bitcast i32 %630 to float, !dbg !20
  %632 = call float @llvm.maxnum.f32(float %628, float %631), !dbg !20
  %633 = bitcast float %632 to i32, !dbg !20
  %634 = call i32 @llvm.amdgcn.ds.bpermute(i32 %589, i32 %633), !dbg !20
  %635 = bitcast i32 %634 to float, !dbg !20
  %636 = call float @llvm.maxnum.f32(float %632, float %635), !dbg !20
  %637 = extractvalue [2 x <1 x float>] %131, 1, !dbg !20
  %638 = insertelement <1 x float> poison, float %636, i64 0, !dbg !20
  %639 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %638, <1 x float> %637), !dbg !20
  %640 = shufflevector <1 x float> %596, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %641 = fsub <4 x float> %418, %640, !dbg !21
  %642 = fsub <4 x float> %430, %640, !dbg !21
  %643 = fsub <4 x float> %442, %640, !dbg !21
  %644 = fsub <4 x float> %454, %640, !dbg !21
  %645 = fsub <4 x float> %466, %640, !dbg !21
  %646 = fsub <4 x float> %478, %640, !dbg !21
  %647 = fsub <4 x float> %490, %640, !dbg !21
  %648 = fsub <4 x float> %502, %640, !dbg !21
  %649 = shufflevector <1 x float> %639, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %650 = fsub <4 x float> %514, %649, !dbg !21
  %651 = fsub <4 x float> %522, %649, !dbg !21
  %652 = fsub <4 x float> %530, %649, !dbg !21
  %653 = fsub <4 x float> %538, %649, !dbg !21
  %654 = fsub <4 x float> %546, %649, !dbg !21
  %655 = fsub <4 x float> %554, %649, !dbg !21
  %656 = fsub <4 x float> %562, %649, !dbg !21
  %657 = fsub <4 x float> %569, %649, !dbg !21
  %658 = fsub <1 x float> %594, %596, !dbg !22
  %659 = extractelement <1 x float> %658, i64 0, !dbg !23
  %660 = call float @llvm.amdgcn.exp2.f32(float %659), !dbg !23
  %661 = insertelement <4 x float> poison, float %660, i64 0, !dbg !28
  %662 = shufflevector <4 x float> %661, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %663 = extractvalue [2 x <1 x float>] %130, 0, !dbg !24
  %664 = extractelement <1 x float> %663, i64 0, !dbg !24
  %665 = fmul float %664, %660, !dbg !24
  %666 = fsub <1 x float> %637, %639, !dbg !22
  %667 = extractelement <1 x float> %666, i64 0, !dbg !23
  %668 = call float @llvm.amdgcn.exp2.f32(float %667), !dbg !23
  %669 = insertelement <4 x float> poison, float %668, i64 0, !dbg !28
  %670 = shufflevector <4 x float> %669, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %671 = extractvalue [2 x <1 x float>] %130, 1, !dbg !24
  %672 = extractelement <1 x float> %671, i64 0, !dbg !24
  %673 = fmul float %672, %668, !dbg !24
  %674 = extractelement <4 x float> %641, i64 0, !dbg !25
  %675 = call float @llvm.amdgcn.exp2.f32(float %674), !dbg !25
  %676 = insertelement <4 x float> poison, float %675, i64 0, !dbg !27
  %677 = extractelement <4 x float> %641, i64 1, !dbg !25
  %678 = call float @llvm.amdgcn.exp2.f32(float %677), !dbg !25
  %679 = insertelement <4 x float> %676, float %678, i64 1, !dbg !27
  %680 = fadd float %675, %678, !dbg !26
  %681 = extractelement <4 x float> %641, i64 2, !dbg !25
  %682 = call float @llvm.amdgcn.exp2.f32(float %681), !dbg !25
  %683 = insertelement <4 x float> %679, float %682, i64 2, !dbg !27
  %684 = fadd float %680, %682, !dbg !26
  %685 = extractelement <4 x float> %641, i64 3, !dbg !25
  %686 = call float @llvm.amdgcn.exp2.f32(float %685), !dbg !25
  %687 = insertelement <4 x float> %683, float %686, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %687) #9, !dbg !27, !frisk.register_lifetimes !1
  %688 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %688) #9, !dbg !26, !frisk.register_lifetimes !1
  %689 = fadd float %684, %686, !dbg !26
  %690 = extractelement <4 x float> %642, i64 0, !dbg !25
  %691 = call float @llvm.amdgcn.exp2.f32(float %690), !dbg !25
  %692 = insertelement <4 x float> poison, float %691, i64 0, !dbg !27
  %693 = fadd float %689, %691, !dbg !26
  %694 = extractelement <4 x float> %642, i64 1, !dbg !25
  %695 = call float @llvm.amdgcn.exp2.f32(float %694), !dbg !25
  %696 = insertelement <4 x float> %692, float %695, i64 1, !dbg !27
  %697 = fadd float %693, %695, !dbg !26
  %698 = extractelement <4 x float> %642, i64 2, !dbg !25
  %699 = call float @llvm.amdgcn.exp2.f32(float %698), !dbg !25
  %700 = insertelement <4 x float> %696, float %699, i64 2, !dbg !27
  %701 = fadd float %697, %699, !dbg !26
  %702 = extractelement <4 x float> %642, i64 3, !dbg !25
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !25
  %704 = insertelement <4 x float> %700, float %703, i64 3, !dbg !27
  %math.result241 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %704) #9, !dbg !27, !frisk.register_lifetimes !1
  %705 = fptrunc <4 x float> %math.result241 to <4 x half>, !dbg !27
  %math.packed242 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %705) #9, !dbg !26, !frisk.register_lifetimes !1
  %706 = fadd float %701, %703, !dbg !26
  %707 = extractelement <4 x float> %643, i64 0, !dbg !25
  %708 = call float @llvm.amdgcn.exp2.f32(float %707), !dbg !25
  %709 = insertelement <4 x float> poison, float %708, i64 0, !dbg !27
  %710 = fadd float %706, %708, !dbg !26
  %711 = extractelement <4 x float> %643, i64 1, !dbg !25
  %712 = call float @llvm.amdgcn.exp2.f32(float %711), !dbg !25
  %713 = insertelement <4 x float> %709, float %712, i64 1, !dbg !27
  %714 = fadd float %710, %712, !dbg !26
  %715 = extractelement <4 x float> %643, i64 2, !dbg !25
  %716 = call float @llvm.amdgcn.exp2.f32(float %715), !dbg !25
  %717 = insertelement <4 x float> %713, float %716, i64 2, !dbg !27
  %718 = fadd float %714, %716, !dbg !26
  %719 = extractelement <4 x float> %643, i64 3, !dbg !25
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !25
  %721 = insertelement <4 x float> %717, float %720, i64 3, !dbg !27
  %math.result243 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %721) #9, !dbg !27, !frisk.register_lifetimes !1
  %722 = fptrunc <4 x float> %math.result243 to <4 x half>, !dbg !27
  %math.packed244 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %722) #9, !dbg !26, !frisk.register_lifetimes !1
  %723 = fadd float %718, %720, !dbg !26
  %724 = extractelement <4 x float> %644, i64 0, !dbg !25
  %725 = call float @llvm.amdgcn.exp2.f32(float %724), !dbg !25
  %726 = insertelement <4 x float> poison, float %725, i64 0, !dbg !27
  %727 = fadd float %723, %725, !dbg !26
  %728 = extractelement <4 x float> %644, i64 1, !dbg !25
  %729 = call float @llvm.amdgcn.exp2.f32(float %728), !dbg !25
  %730 = insertelement <4 x float> %726, float %729, i64 1, !dbg !27
  %731 = fadd float %727, %729, !dbg !26
  %732 = extractelement <4 x float> %644, i64 2, !dbg !25
  %733 = call float @llvm.amdgcn.exp2.f32(float %732), !dbg !25
  %734 = insertelement <4 x float> %730, float %733, i64 2, !dbg !27
  %735 = fadd float %731, %733, !dbg !26
  %736 = extractelement <4 x float> %644, i64 3, !dbg !25
  %737 = call float @llvm.amdgcn.exp2.f32(float %736), !dbg !25
  %738 = insertelement <4 x float> %734, float %737, i64 3, !dbg !27
  %math.result245 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %738) #9, !dbg !27, !frisk.register_lifetimes !1
  %739 = fptrunc <4 x float> %math.result245 to <4 x half>, !dbg !27
  %math.packed246 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %739) #9, !dbg !26, !frisk.register_lifetimes !1
  %740 = fadd float %735, %737, !dbg !26
  %741 = extractelement <4 x float> %645, i64 0, !dbg !25
  %742 = call float @llvm.amdgcn.exp2.f32(float %741), !dbg !25
  %743 = insertelement <4 x float> poison, float %742, i64 0, !dbg !27
  %744 = fadd float %740, %742, !dbg !26
  %745 = extractelement <4 x float> %645, i64 1, !dbg !25
  %746 = call float @llvm.amdgcn.exp2.f32(float %745), !dbg !25
  %747 = insertelement <4 x float> %743, float %746, i64 1, !dbg !27
  %748 = fadd float %744, %746, !dbg !26
  %749 = extractelement <4 x float> %645, i64 2, !dbg !25
  %750 = call float @llvm.amdgcn.exp2.f32(float %749), !dbg !25
  %751 = insertelement <4 x float> %747, float %750, i64 2, !dbg !27
  %752 = fadd float %748, %750, !dbg !26
  %753 = extractelement <4 x float> %645, i64 3, !dbg !25
  %754 = call float @llvm.amdgcn.exp2.f32(float %753), !dbg !25
  %755 = insertelement <4 x float> %751, float %754, i64 3, !dbg !27
  %math.result247 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %755) #9, !dbg !27, !frisk.register_lifetimes !1
  %756 = fptrunc <4 x float> %math.result247 to <4 x half>, !dbg !27
  %math.packed248 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %756) #9, !dbg !26, !frisk.register_lifetimes !1
  %757 = fadd float %752, %754, !dbg !26
  %758 = extractelement <4 x float> %646, i64 0, !dbg !25
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !25
  %760 = insertelement <4 x float> poison, float %759, i64 0, !dbg !27
  %761 = fadd float %757, %759, !dbg !26
  %762 = extractelement <4 x float> %646, i64 1, !dbg !25
  %763 = call float @llvm.amdgcn.exp2.f32(float %762), !dbg !25
  %764 = insertelement <4 x float> %760, float %763, i64 1, !dbg !27
  %765 = fadd float %761, %763, !dbg !26
  %766 = extractelement <4 x float> %646, i64 2, !dbg !25
  %767 = call float @llvm.amdgcn.exp2.f32(float %766), !dbg !25
  %768 = insertelement <4 x float> %764, float %767, i64 2, !dbg !27
  %769 = fadd float %765, %767, !dbg !26
  %770 = extractelement <4 x float> %646, i64 3, !dbg !25
  %771 = call float @llvm.amdgcn.exp2.f32(float %770), !dbg !25
  %772 = insertelement <4 x float> %768, float %771, i64 3, !dbg !27
  %math.result249 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %772) #9, !dbg !27, !frisk.register_lifetimes !1
  %773 = fptrunc <4 x float> %math.result249 to <4 x half>, !dbg !27
  %math.packed250 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %773) #9, !dbg !26, !frisk.register_lifetimes !1
  %774 = fadd float %769, %771, !dbg !26
  %775 = extractelement <4 x float> %647, i64 0, !dbg !25
  %776 = call float @llvm.amdgcn.exp2.f32(float %775), !dbg !25
  %777 = insertelement <4 x float> poison, float %776, i64 0, !dbg !27
  %778 = fadd float %774, %776, !dbg !26
  %779 = extractelement <4 x float> %647, i64 1, !dbg !25
  %780 = call float @llvm.amdgcn.exp2.f32(float %779), !dbg !25
  %781 = insertelement <4 x float> %777, float %780, i64 1, !dbg !27
  %782 = fadd float %778, %780, !dbg !26
  %783 = extractelement <4 x float> %647, i64 2, !dbg !25
  %784 = call float @llvm.amdgcn.exp2.f32(float %783), !dbg !25
  %785 = insertelement <4 x float> %781, float %784, i64 2, !dbg !27
  %786 = fadd float %782, %784, !dbg !26
  %787 = extractelement <4 x float> %647, i64 3, !dbg !25
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !25
  %789 = insertelement <4 x float> %785, float %788, i64 3, !dbg !27
  %math.result251 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %789) #9, !dbg !27, !frisk.register_lifetimes !1
  %790 = fptrunc <4 x float> %math.result251 to <4 x half>, !dbg !27
  %math.packed252 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %790) #9, !dbg !26, !frisk.register_lifetimes !1
  %791 = fadd float %786, %788, !dbg !26
  %792 = extractelement <4 x float> %648, i64 0, !dbg !25
  %793 = call float @llvm.amdgcn.exp2.f32(float %792), !dbg !25
  %794 = insertelement <4 x float> poison, float %793, i64 0, !dbg !27
  %795 = fadd float %791, %793, !dbg !26
  %796 = extractelement <4 x float> %648, i64 1, !dbg !25
  %797 = call float @llvm.amdgcn.exp2.f32(float %796), !dbg !25
  %798 = insertelement <4 x float> %794, float %797, i64 1, !dbg !27
  %799 = fadd float %795, %797, !dbg !26
  %800 = extractelement <4 x float> %648, i64 2, !dbg !25
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !25
  %802 = insertelement <4 x float> %798, float %801, i64 2, !dbg !27
  %803 = fadd float %799, %801, !dbg !26
  %804 = extractelement <4 x float> %648, i64 3, !dbg !25
  %805 = call float @llvm.amdgcn.exp2.f32(float %804), !dbg !25
  %806 = insertelement <4 x float> %802, float %805, i64 3, !dbg !27
  %math.result253 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %806) #9, !dbg !27, !frisk.register_lifetimes !1
  %807 = fptrunc <4 x float> %math.result253 to <4 x half>, !dbg !27
  %math.packed254 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %807) #9, !dbg !26, !frisk.register_lifetimes !1
  %808 = fadd float %803, %805, !dbg !26
  %809 = extractelement <4 x float> %650, i64 0, !dbg !25
  %810 = call float @llvm.amdgcn.exp2.f32(float %809), !dbg !25
  %811 = insertelement <4 x float> poison, float %810, i64 0, !dbg !27
  %812 = extractelement <4 x float> %650, i64 1, !dbg !25
  %813 = call float @llvm.amdgcn.exp2.f32(float %812), !dbg !25
  %814 = insertelement <4 x float> %811, float %813, i64 1, !dbg !27
  %815 = fadd float %810, %813, !dbg !26
  %816 = extractelement <4 x float> %650, i64 2, !dbg !25
  %817 = call float @llvm.amdgcn.exp2.f32(float %816), !dbg !25
  %818 = insertelement <4 x float> %814, float %817, i64 2, !dbg !27
  %819 = fadd float %815, %817, !dbg !26
  %820 = extractelement <4 x float> %650, i64 3, !dbg !25
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !25
  %822 = insertelement <4 x float> %818, float %821, i64 3, !dbg !27
  %math.result255 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %822) #9, !dbg !27, !frisk.register_lifetimes !1
  %823 = fptrunc <4 x float> %math.result255 to <4 x half>, !dbg !27
  %math.packed256 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %823) #9, !dbg !26, !frisk.register_lifetimes !1
  %824 = fadd float %819, %821, !dbg !26
  %825 = extractelement <4 x float> %651, i64 0, !dbg !25
  %826 = call float @llvm.amdgcn.exp2.f32(float %825), !dbg !25
  %827 = insertelement <4 x float> poison, float %826, i64 0, !dbg !27
  %828 = fadd float %824, %826, !dbg !26
  %829 = extractelement <4 x float> %651, i64 1, !dbg !25
  %830 = call float @llvm.amdgcn.exp2.f32(float %829), !dbg !25
  %831 = insertelement <4 x float> %827, float %830, i64 1, !dbg !27
  %832 = fadd float %828, %830, !dbg !26
  %833 = extractelement <4 x float> %651, i64 2, !dbg !25
  %834 = call float @llvm.amdgcn.exp2.f32(float %833), !dbg !25
  %835 = insertelement <4 x float> %831, float %834, i64 2, !dbg !27
  %836 = fadd float %832, %834, !dbg !26
  %837 = extractelement <4 x float> %651, i64 3, !dbg !25
  %838 = call float @llvm.amdgcn.exp2.f32(float %837), !dbg !25
  %839 = insertelement <4 x float> %835, float %838, i64 3, !dbg !27
  %math.result257 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %839) #9, !dbg !27, !frisk.register_lifetimes !1
  %840 = fptrunc <4 x float> %math.result257 to <4 x half>, !dbg !27
  %math.packed258 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %840) #9, !dbg !26, !frisk.register_lifetimes !1
  %841 = fadd float %836, %838, !dbg !26
  %842 = extractelement <4 x float> %652, i64 0, !dbg !25
  %843 = call float @llvm.amdgcn.exp2.f32(float %842), !dbg !25
  %844 = insertelement <4 x float> poison, float %843, i64 0, !dbg !27
  %845 = fadd float %841, %843, !dbg !26
  %846 = extractelement <4 x float> %652, i64 1, !dbg !25
  %847 = call float @llvm.amdgcn.exp2.f32(float %846), !dbg !25
  %848 = insertelement <4 x float> %844, float %847, i64 1, !dbg !27
  %849 = fadd float %845, %847, !dbg !26
  %850 = extractelement <4 x float> %652, i64 2, !dbg !25
  %851 = call float @llvm.amdgcn.exp2.f32(float %850), !dbg !25
  %852 = insertelement <4 x float> %848, float %851, i64 2, !dbg !27
  %853 = fadd float %849, %851, !dbg !26
  %854 = extractelement <4 x float> %652, i64 3, !dbg !25
  %855 = call float @llvm.amdgcn.exp2.f32(float %854), !dbg !25
  %856 = insertelement <4 x float> %852, float %855, i64 3, !dbg !27
  %math.result259 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %856) #9, !dbg !27, !frisk.register_lifetimes !1
  %857 = fptrunc <4 x float> %math.result259 to <4 x half>, !dbg !27
  %math.packed260 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %857) #9, !dbg !26, !frisk.register_lifetimes !1
  %858 = fadd float %853, %855, !dbg !26
  %859 = extractelement <4 x float> %653, i64 0, !dbg !25
  %860 = call float @llvm.amdgcn.exp2.f32(float %859), !dbg !25
  %861 = insertelement <4 x float> poison, float %860, i64 0, !dbg !27
  %862 = fadd float %858, %860, !dbg !26
  %863 = extractelement <4 x float> %653, i64 1, !dbg !25
  %864 = call float @llvm.amdgcn.exp2.f32(float %863), !dbg !25
  %865 = insertelement <4 x float> %861, float %864, i64 1, !dbg !27
  %866 = fadd float %862, %864, !dbg !26
  %867 = extractelement <4 x float> %653, i64 2, !dbg !25
  %868 = call float @llvm.amdgcn.exp2.f32(float %867), !dbg !25
  %869 = insertelement <4 x float> %865, float %868, i64 2, !dbg !27
  %870 = fadd float %866, %868, !dbg !26
  %871 = extractelement <4 x float> %653, i64 3, !dbg !25
  %872 = call float @llvm.amdgcn.exp2.f32(float %871), !dbg !25
  %873 = insertelement <4 x float> %869, float %872, i64 3, !dbg !27
  %math.result261 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %873) #9, !dbg !27, !frisk.register_lifetimes !1
  %874 = fptrunc <4 x float> %math.result261 to <4 x half>, !dbg !27
  %math.packed262 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %874) #9, !dbg !26, !frisk.register_lifetimes !1
  %875 = fadd float %870, %872, !dbg !26
  %876 = extractelement <4 x float> %654, i64 0, !dbg !25
  %877 = call float @llvm.amdgcn.exp2.f32(float %876), !dbg !25
  %878 = insertelement <4 x float> poison, float %877, i64 0, !dbg !27
  %879 = fadd float %875, %877, !dbg !26
  %880 = extractelement <4 x float> %654, i64 1, !dbg !25
  %881 = call float @llvm.amdgcn.exp2.f32(float %880), !dbg !25
  %882 = insertelement <4 x float> %878, float %881, i64 1, !dbg !27
  %883 = fadd float %879, %881, !dbg !26
  %884 = extractelement <4 x float> %654, i64 2, !dbg !25
  %885 = call float @llvm.amdgcn.exp2.f32(float %884), !dbg !25
  %886 = insertelement <4 x float> %882, float %885, i64 2, !dbg !27
  %887 = fadd float %883, %885, !dbg !26
  %888 = extractelement <4 x float> %654, i64 3, !dbg !25
  %889 = call float @llvm.amdgcn.exp2.f32(float %888), !dbg !25
  %890 = insertelement <4 x float> %886, float %889, i64 3, !dbg !27
  %math.result263 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %890) #9, !dbg !27, !frisk.register_lifetimes !1
  %891 = fptrunc <4 x float> %math.result263 to <4 x half>, !dbg !27
  %math.packed264 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %891) #9, !dbg !26, !frisk.register_lifetimes !1
  %892 = fadd float %887, %889, !dbg !26
  %893 = extractelement <4 x float> %655, i64 0, !dbg !25
  %894 = call float @llvm.amdgcn.exp2.f32(float %893), !dbg !25
  %895 = insertelement <4 x float> poison, float %894, i64 0, !dbg !27
  %896 = fadd float %892, %894, !dbg !26
  %897 = extractelement <4 x float> %655, i64 1, !dbg !25
  %898 = call float @llvm.amdgcn.exp2.f32(float %897), !dbg !25
  %899 = insertelement <4 x float> %895, float %898, i64 1, !dbg !27
  %900 = fadd float %896, %898, !dbg !26
  %901 = extractelement <4 x float> %655, i64 2, !dbg !25
  %902 = call float @llvm.amdgcn.exp2.f32(float %901), !dbg !25
  %903 = insertelement <4 x float> %899, float %902, i64 2, !dbg !27
  %904 = fadd float %900, %902, !dbg !26
  %905 = extractelement <4 x float> %655, i64 3, !dbg !25
  %906 = call float @llvm.amdgcn.exp2.f32(float %905), !dbg !25
  %907 = insertelement <4 x float> %903, float %906, i64 3, !dbg !27
  %math.result265 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %907) #9, !dbg !27, !frisk.register_lifetimes !1
  %908 = fptrunc <4 x float> %math.result265 to <4 x half>, !dbg !27
  %math.packed266 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %908) #9, !dbg !26, !frisk.register_lifetimes !1
  %909 = fadd float %904, %906, !dbg !26
  %910 = extractelement <4 x float> %656, i64 0, !dbg !25
  %911 = call float @llvm.amdgcn.exp2.f32(float %910), !dbg !25
  %912 = insertelement <4 x float> poison, float %911, i64 0, !dbg !27
  %913 = fadd float %909, %911, !dbg !26
  %914 = extractelement <4 x float> %656, i64 1, !dbg !25
  %915 = call float @llvm.amdgcn.exp2.f32(float %914), !dbg !25
  %916 = insertelement <4 x float> %912, float %915, i64 1, !dbg !27
  %917 = fadd float %913, %915, !dbg !26
  %918 = extractelement <4 x float> %656, i64 2, !dbg !25
  %919 = call float @llvm.amdgcn.exp2.f32(float %918), !dbg !25
  %920 = insertelement <4 x float> %916, float %919, i64 2, !dbg !27
  %921 = fadd float %917, %919, !dbg !26
  %922 = extractelement <4 x float> %656, i64 3, !dbg !25
  %923 = call float @llvm.amdgcn.exp2.f32(float %922), !dbg !25
  %924 = insertelement <4 x float> %920, float %923, i64 3, !dbg !27
  %math.result267 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %924) #9, !dbg !27, !frisk.register_lifetimes !1
  %925 = fptrunc <4 x float> %math.result267 to <4 x half>, !dbg !27
  %math.packed268 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %925) #9, !dbg !26, !frisk.register_lifetimes !1
  %926 = fadd float %921, %923, !dbg !26
  %927 = extractelement <4 x float> %657, i64 0, !dbg !25
  %928 = call float @llvm.amdgcn.exp2.f32(float %927), !dbg !25
  %929 = insertelement <4 x float> poison, float %928, i64 0, !dbg !27
  %930 = fadd float %926, %928, !dbg !26
  %931 = extractelement <4 x float> %657, i64 1, !dbg !25
  %932 = call float @llvm.amdgcn.exp2.f32(float %931), !dbg !25
  %933 = insertelement <4 x float> %929, float %932, i64 1, !dbg !27
  %934 = fadd float %930, %932, !dbg !26
  %935 = extractelement <4 x float> %657, i64 2, !dbg !25
  %936 = call float @llvm.amdgcn.exp2.f32(float %935), !dbg !25
  %937 = insertelement <4 x float> %933, float %936, i64 2, !dbg !27
  %938 = fadd float %934, %936, !dbg !26
  %939 = extractelement <4 x float> %657, i64 3, !dbg !25
  %940 = call float @llvm.amdgcn.exp2.f32(float %939), !dbg !25
  %941 = insertelement <4 x float> %937, float %940, i64 3, !dbg !27
  %math.result269 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %941) #9, !dbg !27, !frisk.register_lifetimes !1
  %942 = fptrunc <4 x float> %math.result269 to <4 x half>, !dbg !27
  %math.packed270 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %942) #9, !dbg !26, !frisk.register_lifetimes !1
  %943 = fadd float %938, %940, !dbg !26
  %944 = bitcast float %808 to i32, !dbg !26
  %945 = call i32 @llvm.amdgcn.ds.bpermute(i32 %581, i32 %944), !dbg !26
  %946 = bitcast i32 %945 to float, !dbg !26
  %947 = fadd float %808, %946, !dbg !26
  %948 = bitcast float %947 to i32, !dbg !26
  %949 = call i32 @llvm.amdgcn.ds.bpermute(i32 %589, i32 %948), !dbg !26
  %950 = bitcast i32 %949 to float, !dbg !26
  %951 = fadd float %947, %950, !dbg !26
  %952 = fadd float %951, %665, !dbg !26
  %953 = insertelement <1 x float> poison, float %952, i64 0, !dbg !7
  %954 = insertvalue [2 x <1 x float>] poison, <1 x float> %953, 0, !dbg !7
  %955 = bitcast float %943 to i32, !dbg !26
  %956 = call i32 @llvm.amdgcn.ds.bpermute(i32 %581, i32 %955), !dbg !26
  %957 = bitcast i32 %956 to float, !dbg !26
  %958 = fadd float %943, %957, !dbg !26
  %959 = bitcast float %958 to i32, !dbg !26
  %960 = call i32 @llvm.amdgcn.ds.bpermute(i32 %589, i32 %959), !dbg !26
  %961 = bitcast i32 %960 to float, !dbg !26
  %962 = fadd float %958, %961, !dbg !26
  %963 = fadd float %962, %673, !dbg !26
  %964 = insertelement <1 x float> poison, float %963, i64 0, !dbg !7
  %965 = insertvalue [2 x <1 x float>] %954, <1 x float> %964, 1, !dbg !7
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
  %966 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1305, <4 x i32> %vdb.address1309) #10, !dbg !15
  %vdb.dma.base1319 = add i32 %vdb.address1311, 8192, !dbg !13
  %967 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1319, i32 %vdb.address1314, <4 x i32> %vdb.address1318) #10, !dbg !15
  %address.local796 = shl nuw nsw i32 %address.thread365, 3, !dbg !13
  %address.local797 = and i32 %address.local796, 120, !dbg !13
  %address.local798 = shl nuw nsw i32 %address.thread365, 5, !dbg !13
  %address.local799 = and i32 %address.local798, 1536, !dbg !13
  %address.local800 = or disjoint i32 %address.local797, %address.local799, !dbg !13
  %address.local801 = zext nneg i32 %address.local800 to i64, !dbg !13
  %address.local802 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local801, !dbg !13
  %968 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local802, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %969 = bitcast <4 x i32> %968 to <8 x half>, !dbg !13
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
  %970 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local812, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %971 = bitcast <4 x i32> %970 to <8 x half>, !dbg !13
  %address.local813 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local814 = getelementptr [2 x i8], ptr addrspace(3) %address.local813, i64 %address.local808, !dbg !13
  %address.local815 = getelementptr [2 x i8], ptr addrspace(3) %address.local814, i64 %address.local810, !dbg !13
  %address.local816 = getelementptr i8, ptr addrspace(3) %address.local815, i64 512, !dbg !13
  %972 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %973 = bitcast <4 x i32> %972 to <8 x half>, !dbg !13
  %address.local817 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local818 = getelementptr [2 x i8], ptr addrspace(3) %address.local817, i64 %address.local808, !dbg !13
  %address.local819 = getelementptr [2 x i8], ptr addrspace(3) %address.local818, i64 %address.local810, !dbg !13
  %address.local820 = getelementptr i8, ptr addrspace(3) %address.local819, i64 768, !dbg !13
  %974 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %975 = bitcast <4 x i32> %974 to <8 x half>, !dbg !13
  %address.local821 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local801, !dbg !13
  %address.local822 = getelementptr i8, ptr addrspace(3) %address.local821, i64 4096, !dbg !13
  %976 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %977 = bitcast <4 x i32> %976 to <8 x half>, !dbg !13
  %address.local823 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local824 = getelementptr [2 x i8], ptr addrspace(3) %address.local823, i64 %address.local808, !dbg !13
  %address.local825 = getelementptr [2 x i8], ptr addrspace(3) %address.local824, i64 %address.local810, !dbg !13
  %address.local826 = getelementptr i8, ptr addrspace(3) %address.local825, i64 4352, !dbg !13
  %978 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %979 = bitcast <4 x i32> %978 to <8 x half>, !dbg !13
  %address.local827 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local828 = getelementptr [2 x i8], ptr addrspace(3) %address.local827, i64 %address.local808, !dbg !13
  %address.local829 = getelementptr [2 x i8], ptr addrspace(3) %address.local828, i64 %address.local810, !dbg !13
  %address.local830 = getelementptr i8, ptr addrspace(3) %address.local829, i64 4608, !dbg !13
  %980 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %981 = bitcast <4 x i32> %980 to <8 x half>, !dbg !13
  %address.local831 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local805, !dbg !13
  %address.local832 = getelementptr [2 x i8], ptr addrspace(3) %address.local831, i64 %address.local808, !dbg !13
  %address.local833 = getelementptr [2 x i8], ptr addrspace(3) %address.local832, i64 %address.local810, !dbg !13
  %address.local834 = getelementptr i8, ptr addrspace(3) %address.local833, i64 4864, !dbg !13
  %982 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local834, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %983 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %984 = extractvalue [2 x <8 x float>] %129, 0, !dbg !28
  %985 = shufflevector <8 x float> %984, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %986 = fmul <4 x float> %985, %662, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %983, <4 x float> %986) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %988 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %989 = shufflevector <8 x float> %984, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %990 = fmul <4 x float> %989, %662, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %988, <4 x float> %990) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %992 = extractvalue [2 x <8 x float>] %129, 1, !dbg !28
  %993 = shufflevector <8 x float> %992, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %994 = fmul <4 x float> %993, %670, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %983, <4 x float> %994) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %996 = shufflevector <8 x float> %992, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %997 = fmul <4 x float> %996, %670, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %988, <4 x float> %997) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %999 = shufflevector <8 x half> %971, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1000 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %999, <4 x float> %987) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1001 = shufflevector <8 x half> %971, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1001, <4 x float> %991) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %999, <4 x float> %995) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1001, <4 x float> %998) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1005 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1006 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1005, <4 x float> %1000) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1007 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1007, <4 x float> %1002) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1009 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1005, <4 x float> %1003) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1007, <4 x float> %1004) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1011 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1011, <4 x float> %1006) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1013 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1013, <4 x float> %1008) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1011, <4 x float> %1009) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1013, <4 x float> %1010) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1017 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1017, <4 x float> %1012) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1019 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1019, <4 x float> %1014) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1017, <4 x float> %1015) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1019, <4 x float> %1016) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1023 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1023, <4 x float> %1018) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1025 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1025, <4 x float> %1020) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1023, <4 x float> %1021) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1028 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1025, <4 x float> %1022) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1029 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1029, <4 x float> %1024) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1031 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1032 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1031, <4 x float> %1026) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1029, <4 x float> %1027) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1034 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1031, <4 x float> %1028) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1035 = bitcast <4 x i32> %982 to <8 x half>, !dbg !13
  %1036 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1036, <4 x float> %1030) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1038 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1038, <4 x float> %1032) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1036, <4 x float> %1033) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1041 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1038, <4 x float> %1034) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1327 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !15
  %vdb.address1328 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1327), !dbg !15
  %vdb.address1329 = or disjoint i64 %250, %address.local788, !dbg !15
  %vdb.address1330 = trunc nuw i64 %vdb.address1329 to i32, !dbg !15
  %vdb.address1331 = shl nuw nsw i32 %vdb.address1330, 1, !dbg !15
  %vdb.address1332 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1333 = insertelement <2 x i64> poison, i64 %vdb.address1332, i64 0, !dbg !15
  %vdb.address1334 = bitcast <2 x i64> %vdb.address1333 to <4 x i32>, !dbg !15
  %vdb.address1335 = shufflevector <4 x i32> %vdb.address1334, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1336 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !15
  %vdb.address1337 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1336), !dbg !15
  %vdb.address1338 = or disjoint i64 %250, %address.local794, !dbg !15
  %vdb.address1339 = trunc nuw i64 %vdb.address1338 to i32, !dbg !15
  %vdb.address1340 = shl nuw nsw i32 %vdb.address1339, 1, !dbg !15
  %vdb.address1341 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1342 = insertelement <2 x i64> poison, i64 %vdb.address1341, i64 0, !dbg !15
  %vdb.address1343 = bitcast <2 x i64> %vdb.address1342 to <4 x i32>, !dbg !15
  %vdb.address1344 = shufflevector <4 x i32> %vdb.address1343, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1042 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.address1328, i32 %vdb.address1331, <4 x i32> %vdb.address1335) #10, !dbg !15
  %1043 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.address1337, i32 %vdb.address1340, <4 x i32> %vdb.address1344) #10, !dbg !15
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local802, i64 8192, !dbg !13
  %1044 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1045 = bitcast <4 x i32> %1044 to <8 x half>, !dbg !13
  %vdb.read1320 = getelementptr i8, ptr addrspace(3) %address.local812, i64 8192, !dbg !13
  %1046 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1320, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1047 = bitcast <4 x i32> %1046 to <8 x half>, !dbg !13
  %vdb.read1321 = getelementptr i8, ptr addrspace(3) %address.local816, i64 8192, !dbg !13
  %1048 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1321, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1049 = bitcast <4 x i32> %1048 to <8 x half>, !dbg !13
  %vdb.read1322 = getelementptr i8, ptr addrspace(3) %address.local820, i64 8192, !dbg !13
  %1050 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1322, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1051 = bitcast <4 x i32> %1050 to <8 x half>, !dbg !13
  %vdb.read1323 = getelementptr i8, ptr addrspace(3) %address.local822, i64 8192, !dbg !13
  %1052 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1323, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1053 = bitcast <4 x i32> %1052 to <8 x half>, !dbg !13
  %vdb.read1324 = getelementptr i8, ptr addrspace(3) %address.local826, i64 8192, !dbg !13
  %1054 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1324, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1055 = bitcast <4 x i32> %1054 to <8 x half>, !dbg !13
  %vdb.read1325 = getelementptr i8, ptr addrspace(3) %address.local830, i64 8192, !dbg !13
  %1056 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1325, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1057 = bitcast <4 x i32> %1056 to <8 x half>, !dbg !13
  %vdb.read1326 = getelementptr i8, ptr addrspace(3) %address.local834, i64 8192, !dbg !13
  %1058 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1326, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1059 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1060 = extractvalue [2 x <8 x float>] %132, 0, !dbg !28
  %1061 = shufflevector <8 x float> %1060, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1062 = fmul <4 x float> %1061, %662, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1059, <4 x float> %1062) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1064 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1065 = shufflevector <8 x float> %1060, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1066 = fmul <4 x float> %1065, %662, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1064, <4 x float> %1066) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1068 = extractvalue [2 x <8 x float>] %132, 1, !dbg !28
  %1069 = shufflevector <8 x float> %1068, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1070 = fmul <4 x float> %1069, %670, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1071 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1059, <4 x float> %1070) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1072 = shufflevector <8 x float> %1068, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1073 = fmul <4 x float> %1072, %670, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1064, <4 x float> %1073) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1075 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1075, <4 x float> %1063) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1077 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1077, <4 x float> %1067) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1075, <4 x float> %1071) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1077, <4 x float> %1074) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1081 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1081, <4 x float> %1076) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1083 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1083, <4 x float> %1078) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1085 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1081, <4 x float> %1079) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1083, <4 x float> %1080) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1087 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1087, <4 x float> %1082) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1089 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1089, <4 x float> %1084) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1091 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1087, <4 x float> %1085) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1089, <4 x float> %1086) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1093 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1093, <4 x float> %1088) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1095 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1095, <4 x float> %1090) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1097 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1093, <4 x float> %1091) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1095, <4 x float> %1092) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1099 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1099, <4 x float> %1094) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1101 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1101, <4 x float> %1096) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1099, <4 x float> %1097) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1101, <4 x float> %1098) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1105 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1105, <4 x float> %1100) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1107 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1107, <4 x float> %1102) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1109 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1105, <4 x float> %1103) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1107, <4 x float> %1104) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1111 = bitcast <4 x i32> %1058 to <8 x half>, !dbg !13
  %1112 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1113 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1112, <4 x float> %1106) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1114 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1115 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1114, <4 x float> %1108) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1112, <4 x float> %1109) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1114, <4 x float> %1110) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1345 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !15
  %vdb.address1346 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1345), !dbg !15
  %vdb.address1347 = or disjoint i64 %327, %address.local788, !dbg !15
  %vdb.address1348 = trunc nuw i64 %vdb.address1347 to i32, !dbg !15
  %vdb.address1349 = shl nuw nsw i32 %vdb.address1348, 1, !dbg !15
  %vdb.address1350 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1351 = insertelement <2 x i64> poison, i64 %vdb.address1350, i64 0, !dbg !15
  %vdb.address1352 = bitcast <2 x i64> %vdb.address1351 to <4 x i32>, !dbg !15
  %vdb.address1353 = shufflevector <4 x i32> %vdb.address1352, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1354 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !15
  %vdb.address1355 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1354), !dbg !15
  %vdb.address1356 = or disjoint i64 %327, %address.local794, !dbg !15
  %vdb.address1357 = trunc nuw i64 %vdb.address1356 to i32, !dbg !15
  %vdb.address1358 = shl nuw nsw i32 %vdb.address1357, 1, !dbg !15
  %vdb.address1359 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1360 = insertelement <2 x i64> poison, i64 %vdb.address1359, i64 0, !dbg !15
  %vdb.address1361 = bitcast <2 x i64> %vdb.address1360 to <4 x i32>, !dbg !15
  %vdb.address1362 = shufflevector <4 x i32> %vdb.address1361, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.dma.base1363 = add i32 %vdb.address1346, 8192, !dbg !13
  %1118 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1363, i32 %vdb.address1349, <4 x i32> %vdb.address1353) #10, !dbg !15
  %vdb.dma.base1364 = add i32 %vdb.address1355, 8192, !dbg !13
  %1119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1364, i32 %vdb.address1358, <4 x i32> %vdb.address1362) #10, !dbg !15
  %1120 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local802, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1121 = bitcast <4 x i32> %1120 to <8 x half>, !dbg !13
  %1122 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local812, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1123 = bitcast <4 x i32> %1122 to <8 x half>, !dbg !13
  %1124 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1125 = bitcast <4 x i32> %1124 to <8 x half>, !dbg !13
  %1126 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1127 = bitcast <4 x i32> %1126 to <8 x half>, !dbg !13
  %1128 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1129 = bitcast <4 x i32> %1128 to <8 x half>, !dbg !13
  %1130 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1131 = bitcast <4 x i32> %1130 to <8 x half>, !dbg !13
  %1132 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1133 = bitcast <4 x i32> %1132 to <8 x half>, !dbg !13
  %1134 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local834, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1135 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1136 = extractvalue [2 x <8 x float>] %133, 0, !dbg !28
  %1137 = shufflevector <8 x float> %1136, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1138 = fmul <4 x float> %1137, %662, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1135, <4 x float> %1138) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1140 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1141 = shufflevector <8 x float> %1136, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1142 = fmul <4 x float> %1141, %662, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1140, <4 x float> %1142) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1144 = extractvalue [2 x <8 x float>] %133, 1, !dbg !28
  %1145 = shufflevector <8 x float> %1144, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1146 = fmul <4 x float> %1145, %670, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1135, <4 x float> %1146) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1148 = shufflevector <8 x float> %1144, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1149 = fmul <4 x float> %1148, %670, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1140, <4 x float> %1149) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1151 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1151, <4 x float> %1139) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1153 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1153, <4 x float> %1143) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1151, <4 x float> %1147) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1153, <4 x float> %1150) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1157 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1157, <4 x float> %1152) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1159 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1159, <4 x float> %1154) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1157, <4 x float> %1155) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1159, <4 x float> %1156) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1163 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1163, <4 x float> %1158) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1165 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1165, <4 x float> %1160) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1163, <4 x float> %1161) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1165, <4 x float> %1162) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1169 = shufflevector <8 x half> %1129, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1169, <4 x float> %1164) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1171 = shufflevector <8 x half> %1129, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1171, <4 x float> %1166) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1169, <4 x float> %1167) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1171, <4 x float> %1168) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1175 = shufflevector <8 x half> %1131, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1175, <4 x float> %1170) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1177 = shufflevector <8 x half> %1131, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1177, <4 x float> %1172) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1175, <4 x float> %1173) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1177, <4 x float> %1174) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1181 = shufflevector <8 x half> %1133, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1181, <4 x float> %1176) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1183 = shufflevector <8 x half> %1133, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1183, <4 x float> %1178) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1181, <4 x float> %1179) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1183, <4 x float> %1180) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1187 = bitcast <4 x i32> %1134 to <8 x half>, !dbg !13
  %1188 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1188, <4 x float> %1182) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1190 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1190, <4 x float> %1184) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1188, <4 x float> %1185) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1190, <4 x float> %1186) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.read1365 = getelementptr i8, ptr addrspace(3) %address.local802, i64 8192, !dbg !13
  %1194 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1365, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1195 = bitcast <4 x i32> %1194 to <8 x half>, !dbg !13
  %vdb.read1366 = getelementptr i8, ptr addrspace(3) %address.local812, i64 8192, !dbg !13
  %1196 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1366, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1197 = bitcast <4 x i32> %1196 to <8 x half>, !dbg !13
  %vdb.read1367 = getelementptr i8, ptr addrspace(3) %address.local816, i64 8192, !dbg !13
  %1198 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1367, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1199 = bitcast <4 x i32> %1198 to <8 x half>, !dbg !13
  %vdb.read1368 = getelementptr i8, ptr addrspace(3) %address.local820, i64 8192, !dbg !13
  %1200 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1368, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1201 = bitcast <4 x i32> %1200 to <8 x half>, !dbg !13
  %vdb.read1369 = getelementptr i8, ptr addrspace(3) %address.local822, i64 8192, !dbg !13
  %1202 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1369, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1203 = bitcast <4 x i32> %1202 to <8 x half>, !dbg !13
  %vdb.read1370 = getelementptr i8, ptr addrspace(3) %address.local826, i64 8192, !dbg !13
  %1204 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1370, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1205 = bitcast <4 x i32> %1204 to <8 x half>, !dbg !13
  %vdb.read1371 = getelementptr i8, ptr addrspace(3) %address.local830, i64 8192, !dbg !13
  %1206 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1371, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1207 = bitcast <4 x i32> %1206 to <8 x half>, !dbg !13
  %vdb.read1372 = getelementptr i8, ptr addrspace(3) %address.local834, i64 8192, !dbg !13
  %1208 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1372, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1209 = shufflevector <8 x half> %1195, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1210 = extractvalue [2 x <8 x float>] %134, 0, !dbg !28
  %1211 = shufflevector <8 x float> %1210, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1212 = fmul <4 x float> %1211, %662, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1209, <4 x float> %1212) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1214 = shufflevector <8 x half> %1195, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1215 = shufflevector <8 x float> %1210, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1216 = fmul <4 x float> %1215, %662, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1214, <4 x float> %1216) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1218 = extractvalue [2 x <8 x float>] %134, 1, !dbg !28
  %1219 = shufflevector <8 x float> %1218, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1220 = fmul <4 x float> %1219, %670, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1209, <4 x float> %1220) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1222 = shufflevector <8 x float> %1218, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1223 = fmul <4 x float> %1222, %670, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1214, <4 x float> %1223) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1225 = shufflevector <8 x half> %1197, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1225, <4 x float> %1213) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1227 = shufflevector <8 x half> %1197, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1227, <4 x float> %1217) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1225, <4 x float> %1221) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1227, <4 x float> %1224) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1231 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1231, <4 x float> %1226) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1233 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1233, <4 x float> %1228) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1231, <4 x float> %1229) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1233, <4 x float> %1230) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1237 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1237, <4 x float> %1232) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1239 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1239, <4 x float> %1234) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1237, <4 x float> %1235) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1239, <4 x float> %1236) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1243 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1243, <4 x float> %1238) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1245 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1245, <4 x float> %1240) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1243, <4 x float> %1241) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1245, <4 x float> %1242) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1249 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1249, <4 x float> %1244) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1251 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1251, <4 x float> %1246) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1249, <4 x float> %1247) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1251, <4 x float> %1248) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1255 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1255, <4 x float> %1250) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1257 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1257, <4 x float> %1252) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1255, <4 x float> %1253) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1257, <4 x float> %1254) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1261 = bitcast <4 x i32> %1208 to <8 x half>, !dbg !13
  %1262 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1262, <4 x float> %1256) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1264 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1264, <4 x float> %1258) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1262, <4 x float> %1259) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1264, <4 x float> %1260) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %1268 = shufflevector <4 x float> %1037, <4 x float> %1039, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1269 = insertvalue [2 x <8 x float>] poison, <8 x float> %1268, 0, !dbg !7
  %1270 = shufflevector <4 x float> %1040, <4 x float> %1041, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1271 = insertvalue [2 x <8 x float>] %1269, <8 x float> %1270, 1, !dbg !7
  %1272 = insertvalue [2 x <1 x float>] poison, <1 x float> %596, 0, !dbg !7
  %1273 = insertvalue [2 x <1 x float>] %1272, <1 x float> %639, 1, !dbg !7
  %1274 = shufflevector <4 x float> %1113, <4 x float> %1115, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1275 = insertvalue [2 x <8 x float>] poison, <8 x float> %1274, 0, !dbg !7
  %1276 = shufflevector <4 x float> %1116, <4 x float> %1117, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1277 = insertvalue [2 x <8 x float>] %1275, <8 x float> %1276, 1, !dbg !7
  %1278 = shufflevector <4 x float> %1189, <4 x float> %1191, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1279 = insertvalue [2 x <8 x float>] poison, <8 x float> %1278, 0, !dbg !7
  %1280 = shufflevector <4 x float> %1192, <4 x float> %1193, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1281 = insertvalue [2 x <8 x float>] %1279, <8 x float> %1280, 1, !dbg !7
  %1282 = shufflevector <4 x float> %1263, <4 x float> %1265, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1283 = insertvalue [2 x <8 x float>] poison, <8 x float> %1282, 0, !dbg !7
  %1284 = shufflevector <4 x float> %1266, <4 x float> %1267, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1285 = insertvalue [2 x <8 x float>] %1283, <8 x float> %1284, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1286 = add nuw nsw i64 %128, 1, !dbg !7
  br label %127, !dbg !7

1287:                                             ; preds = %127
  %1288 = and i32 %33, 16, !dbg !7
  %.not = icmp eq i32 %1288, 0, !dbg !7
  %1289 = and i32 %33, 32, !dbg !7
  %.not36 = icmp eq i32 %1289, 0, !dbg !7
  %1290 = extractvalue [2 x <8 x float>] %134, 0, !dbg !7
  %1291 = extractelement <8 x float> %1290, i64 0, !dbg !7
  %1292 = extractelement <8 x float> %1290, i64 1, !dbg !7
  %1293 = extractelement <8 x float> %1290, i64 2, !dbg !7
  %1294 = extractelement <8 x float> %1290, i64 3, !dbg !7
  %1295 = select i1 %.not, float %1292, float %1291, !dbg !7
  %1296 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1297 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1296), !dbg !7
  %1298 = add nuw nsw i32 %1297, 64, !dbg !7
  %1299 = xor i32 %1297, 16, !dbg !7
  %1300 = and i32 %1298, 192, !dbg !7
  %1301 = icmp samesign ult i32 %1299, %1300, !dbg !7
  %1302 = select i1 %1301, i32 %1299, i32 %1297, !dbg !7
  %1303 = shl nuw nsw i32 %1302, 2, !dbg !7
  %1304 = bitcast float %1295 to i32, !dbg !7
  %1305 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1304), !dbg !7
  %1306 = bitcast i32 %1305 to float, !dbg !7
  %1307 = select i1 %.not, float %1291, float %1306, !dbg !7
  %1308 = select i1 %.not, float %1306, float %1292, !dbg !7
  %1309 = select i1 %.not, float %1294, float %1293, !dbg !7
  %1310 = bitcast float %1309 to i32, !dbg !7
  %1311 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1310), !dbg !7
  %1312 = bitcast i32 %1311 to float, !dbg !7
  %1313 = select i1 %.not, float %1293, float %1312, !dbg !7
  %1314 = select i1 %.not, float %1312, float %1294, !dbg !7
  %1315 = select i1 %.not36, float %1313, float %1307, !dbg !7
  %1316 = xor i32 %1297, 32, !dbg !7
  %1317 = icmp samesign ult i32 %1316, %1300, !dbg !7
  %1318 = select i1 %1317, i32 %1316, i32 %1297, !dbg !7
  %1319 = shl nuw nsw i32 %1318, 2, !dbg !7
  %1320 = bitcast float %1315 to i32, !dbg !7
  %1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1320), !dbg !7
  %1322 = bitcast i32 %1321 to float, !dbg !7
  %1323 = select i1 %.not36, float %1307, float %1322, !dbg !7
  %1324 = select i1 %.not36, float %1322, float %1313, !dbg !7
  %1325 = select i1 %.not36, float %1314, float %1308, !dbg !7
  %1326 = bitcast float %1325 to i32, !dbg !7
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1326), !dbg !7
  %1328 = bitcast i32 %1327 to float, !dbg !7
  %1329 = select i1 %.not36, float %1308, float %1328, !dbg !7
  %1330 = select i1 %.not36, float %1328, float %1314, !dbg !7
  %1331 = extractvalue [2 x <8 x float>] %133, 0, !dbg !7
  %1332 = extractelement <8 x float> %1331, i64 0, !dbg !7
  %1333 = extractelement <8 x float> %1331, i64 1, !dbg !7
  %1334 = extractelement <8 x float> %1331, i64 2, !dbg !7
  %1335 = extractelement <8 x float> %1331, i64 3, !dbg !7
  %1336 = select i1 %.not, float %1333, float %1332, !dbg !7
  %1337 = bitcast float %1336 to i32, !dbg !7
  %1338 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1337), !dbg !7
  %1339 = bitcast i32 %1338 to float, !dbg !7
  %1340 = select i1 %.not, float %1332, float %1339, !dbg !7
  %1341 = select i1 %.not, float %1339, float %1333, !dbg !7
  %1342 = select i1 %.not, float %1335, float %1334, !dbg !7
  %1343 = bitcast float %1342 to i32, !dbg !7
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1343), !dbg !7
  %1345 = bitcast i32 %1344 to float, !dbg !7
  %1346 = select i1 %.not, float %1334, float %1345, !dbg !7
  %1347 = select i1 %.not, float %1345, float %1335, !dbg !7
  %1348 = select i1 %.not36, float %1346, float %1340, !dbg !7
  %1349 = bitcast float %1348 to i32, !dbg !7
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1349), !dbg !7
  %1351 = bitcast i32 %1350 to float, !dbg !7
  %1352 = select i1 %.not36, float %1340, float %1351, !dbg !7
  %1353 = select i1 %.not36, float %1351, float %1346, !dbg !7
  %1354 = select i1 %.not36, float %1347, float %1341, !dbg !7
  %1355 = bitcast float %1354 to i32, !dbg !7
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1355), !dbg !7
  %1357 = bitcast i32 %1356 to float, !dbg !7
  %1358 = select i1 %.not36, float %1341, float %1357, !dbg !7
  %1359 = select i1 %.not36, float %1357, float %1347, !dbg !7
  %1360 = extractvalue [2 x <8 x float>] %132, 0, !dbg !7
  %1361 = extractelement <8 x float> %1360, i64 0, !dbg !7
  %1362 = extractelement <8 x float> %1360, i64 1, !dbg !7
  %1363 = extractelement <8 x float> %1360, i64 2, !dbg !7
  %1364 = extractelement <8 x float> %1360, i64 3, !dbg !7
  %1365 = select i1 %.not, float %1362, float %1361, !dbg !7
  %1366 = bitcast float %1365 to i32, !dbg !7
  %1367 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1366), !dbg !7
  %1368 = bitcast i32 %1367 to float, !dbg !7
  %1369 = select i1 %.not, float %1361, float %1368, !dbg !7
  %1370 = select i1 %.not, float %1368, float %1362, !dbg !7
  %1371 = select i1 %.not, float %1364, float %1363, !dbg !7
  %1372 = bitcast float %1371 to i32, !dbg !7
  %1373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1372), !dbg !7
  %1374 = bitcast i32 %1373 to float, !dbg !7
  %1375 = select i1 %.not, float %1363, float %1374, !dbg !7
  %1376 = select i1 %.not, float %1374, float %1364, !dbg !7
  %1377 = select i1 %.not36, float %1375, float %1369, !dbg !7
  %1378 = bitcast float %1377 to i32, !dbg !7
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1378), !dbg !7
  %1380 = bitcast i32 %1379 to float, !dbg !7
  %1381 = select i1 %.not36, float %1369, float %1380, !dbg !7
  %1382 = select i1 %.not36, float %1380, float %1375, !dbg !7
  %1383 = select i1 %.not36, float %1376, float %1370, !dbg !7
  %1384 = bitcast float %1383 to i32, !dbg !7
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1384), !dbg !7
  %1386 = bitcast i32 %1385 to float, !dbg !7
  %1387 = select i1 %.not36, float %1370, float %1386, !dbg !7
  %1388 = select i1 %.not36, float %1386, float %1376, !dbg !7
  %1389 = extractvalue [2 x <8 x float>] %129, 0, !dbg !7
  %1390 = extractelement <8 x float> %1389, i64 0, !dbg !7
  %1391 = extractelement <8 x float> %1389, i64 1, !dbg !7
  %1392 = extractelement <8 x float> %1389, i64 2, !dbg !7
  %1393 = extractelement <8 x float> %1389, i64 3, !dbg !7
  %1394 = select i1 %.not, float %1391, float %1390, !dbg !7
  %1395 = bitcast float %1394 to i32, !dbg !7
  %1396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1395), !dbg !7
  %1397 = bitcast i32 %1396 to float, !dbg !7
  %1398 = select i1 %.not, float %1390, float %1397, !dbg !7
  %1399 = select i1 %.not, float %1397, float %1391, !dbg !7
  %1400 = select i1 %.not, float %1393, float %1392, !dbg !7
  %1401 = bitcast float %1400 to i32, !dbg !7
  %1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1401), !dbg !7
  %1403 = bitcast i32 %1402 to float, !dbg !7
  %1404 = select i1 %.not, float %1392, float %1403, !dbg !7
  %1405 = select i1 %.not, float %1403, float %1393, !dbg !7
  %1406 = select i1 %.not36, float %1404, float %1398, !dbg !7
  %1407 = bitcast float %1406 to i32, !dbg !7
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1407), !dbg !7
  %1409 = select i1 %.not36, float %1405, float %1399, !dbg !7
  %1410 = bitcast float %1409 to i32, !dbg !7
  %1411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1410), !dbg !7
  %1412 = extractvalue [2 x <1 x float>] %130, 0, !dbg !16
  %1413 = shufflevector <1 x float> %1412, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1414 = extractelement <8 x float> %1290, i64 4, !dbg !7
  %1415 = extractelement <8 x float> %1290, i64 5, !dbg !7
  %1416 = extractelement <8 x float> %1290, i64 6, !dbg !7
  %1417 = extractelement <8 x float> %1290, i64 7, !dbg !7
  %1418 = select i1 %.not, float %1415, float %1414, !dbg !7
  %1419 = bitcast float %1418 to i32, !dbg !7
  %1420 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1419), !dbg !7
  %1421 = bitcast i32 %1420 to float, !dbg !7
  %1422 = select i1 %.not, float %1414, float %1421, !dbg !7
  %1423 = select i1 %.not, float %1421, float %1415, !dbg !7
  %1424 = select i1 %.not, float %1417, float %1416, !dbg !7
  %1425 = bitcast float %1424 to i32, !dbg !7
  %1426 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1425), !dbg !7
  %1427 = bitcast i32 %1426 to float, !dbg !7
  %1428 = select i1 %.not, float %1416, float %1427, !dbg !7
  %1429 = select i1 %.not, float %1427, float %1417, !dbg !7
  %1430 = select i1 %.not36, float %1428, float %1422, !dbg !7
  %1431 = bitcast float %1430 to i32, !dbg !7
  %1432 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1431), !dbg !7
  %1433 = bitcast i32 %1432 to float, !dbg !7
  %1434 = select i1 %.not36, float %1422, float %1433, !dbg !7
  %1435 = select i1 %.not36, float %1433, float %1428, !dbg !7
  %1436 = select i1 %.not36, float %1429, float %1423, !dbg !7
  %1437 = bitcast float %1436 to i32, !dbg !7
  %1438 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1437), !dbg !7
  %1439 = bitcast i32 %1438 to float, !dbg !7
  %1440 = select i1 %.not36, float %1423, float %1439, !dbg !7
  %1441 = select i1 %.not36, float %1439, float %1429, !dbg !7
  %1442 = extractelement <8 x float> %1331, i64 4, !dbg !7
  %1443 = extractelement <8 x float> %1331, i64 5, !dbg !7
  %1444 = extractelement <8 x float> %1331, i64 6, !dbg !7
  %1445 = extractelement <8 x float> %1331, i64 7, !dbg !7
  %1446 = select i1 %.not, float %1443, float %1442, !dbg !7
  %1447 = bitcast float %1446 to i32, !dbg !7
  %1448 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1447), !dbg !7
  %1449 = bitcast i32 %1448 to float, !dbg !7
  %1450 = select i1 %.not, float %1442, float %1449, !dbg !7
  %1451 = select i1 %.not, float %1449, float %1443, !dbg !7
  %1452 = select i1 %.not, float %1445, float %1444, !dbg !7
  %1453 = bitcast float %1452 to i32, !dbg !7
  %1454 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1453), !dbg !7
  %1455 = bitcast i32 %1454 to float, !dbg !7
  %1456 = select i1 %.not, float %1444, float %1455, !dbg !7
  %1457 = select i1 %.not, float %1455, float %1445, !dbg !7
  %1458 = select i1 %.not36, float %1456, float %1450, !dbg !7
  %1459 = bitcast float %1458 to i32, !dbg !7
  %1460 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1459), !dbg !7
  %1461 = bitcast i32 %1460 to float, !dbg !7
  %1462 = select i1 %.not36, float %1450, float %1461, !dbg !7
  %1463 = select i1 %.not36, float %1461, float %1456, !dbg !7
  %1464 = select i1 %.not36, float %1457, float %1451, !dbg !7
  %1465 = bitcast float %1464 to i32, !dbg !7
  %1466 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1465), !dbg !7
  %1467 = bitcast i32 %1466 to float, !dbg !7
  %1468 = select i1 %.not36, float %1451, float %1467, !dbg !7
  %1469 = select i1 %.not36, float %1467, float %1457, !dbg !7
  %1470 = extractelement <8 x float> %1360, i64 4, !dbg !7
  %1471 = extractelement <8 x float> %1360, i64 5, !dbg !7
  %1472 = extractelement <8 x float> %1360, i64 6, !dbg !7
  %1473 = extractelement <8 x float> %1360, i64 7, !dbg !7
  %1474 = select i1 %.not, float %1471, float %1470, !dbg !7
  %1475 = bitcast float %1474 to i32, !dbg !7
  %1476 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1475), !dbg !7
  %1477 = bitcast i32 %1476 to float, !dbg !7
  %1478 = select i1 %.not, float %1470, float %1477, !dbg !7
  %1479 = select i1 %.not, float %1477, float %1471, !dbg !7
  %1480 = select i1 %.not, float %1473, float %1472, !dbg !7
  %1481 = bitcast float %1480 to i32, !dbg !7
  %1482 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1481), !dbg !7
  %1483 = bitcast i32 %1482 to float, !dbg !7
  %1484 = select i1 %.not, float %1472, float %1483, !dbg !7
  %1485 = select i1 %.not, float %1483, float %1473, !dbg !7
  %1486 = select i1 %.not36, float %1484, float %1478, !dbg !7
  %1487 = bitcast float %1486 to i32, !dbg !7
  %1488 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1487), !dbg !7
  %1489 = bitcast i32 %1488 to float, !dbg !7
  %1490 = select i1 %.not36, float %1478, float %1489, !dbg !7
  %1491 = select i1 %.not36, float %1489, float %1484, !dbg !7
  %1492 = select i1 %.not36, float %1485, float %1479, !dbg !7
  %1493 = bitcast float %1492 to i32, !dbg !7
  %1494 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1493), !dbg !7
  %1495 = bitcast i32 %1494 to float, !dbg !7
  %1496 = select i1 %.not36, float %1479, float %1495, !dbg !7
  %1497 = select i1 %.not36, float %1495, float %1485, !dbg !7
  %1498 = extractelement <8 x float> %1389, i64 4, !dbg !7
  %1499 = extractelement <8 x float> %1389, i64 5, !dbg !7
  %1500 = extractelement <8 x float> %1389, i64 6, !dbg !7
  %1501 = extractelement <8 x float> %1389, i64 7, !dbg !7
  %1502 = select i1 %.not, float %1499, float %1498, !dbg !7
  %1503 = bitcast float %1502 to i32, !dbg !7
  %1504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1503), !dbg !7
  %1505 = bitcast i32 %1504 to float, !dbg !7
  %1506 = select i1 %.not, float %1498, float %1505, !dbg !7
  %1507 = select i1 %.not, float %1505, float %1499, !dbg !7
  %1508 = select i1 %.not, float %1501, float %1500, !dbg !7
  %1509 = bitcast float %1508 to i32, !dbg !7
  %1510 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1509), !dbg !7
  %1511 = bitcast i32 %1510 to float, !dbg !7
  %1512 = select i1 %.not, float %1500, float %1511, !dbg !7
  %1513 = select i1 %.not, float %1511, float %1501, !dbg !7
  %1514 = select i1 %.not36, float %1512, float %1506, !dbg !7
  %1515 = bitcast float %1514 to i32, !dbg !7
  %1516 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1515), !dbg !7
  %1517 = select i1 %.not36, float %1513, float %1507, !dbg !7
  %1518 = bitcast float %1517 to i32, !dbg !7
  %1519 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1518), !dbg !7
  %1520 = extractvalue [2 x <8 x float>] %134, 1, !dbg !7
  %1521 = extractelement <8 x float> %1520, i64 0, !dbg !7
  %1522 = extractelement <8 x float> %1520, i64 1, !dbg !7
  %1523 = extractelement <8 x float> %1520, i64 2, !dbg !7
  %1524 = extractelement <8 x float> %1520, i64 3, !dbg !7
  %1525 = select i1 %.not, float %1522, float %1521, !dbg !7
  %1526 = bitcast float %1525 to i32, !dbg !7
  %1527 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1526), !dbg !7
  %1528 = bitcast i32 %1527 to float, !dbg !7
  %1529 = select i1 %.not, float %1521, float %1528, !dbg !7
  %1530 = select i1 %.not, float %1528, float %1522, !dbg !7
  %1531 = select i1 %.not, float %1524, float %1523, !dbg !7
  %1532 = bitcast float %1531 to i32, !dbg !7
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1532), !dbg !7
  %1534 = bitcast i32 %1533 to float, !dbg !7
  %1535 = select i1 %.not, float %1523, float %1534, !dbg !7
  %1536 = select i1 %.not, float %1534, float %1524, !dbg !7
  %1537 = select i1 %.not36, float %1535, float %1529, !dbg !7
  %1538 = bitcast float %1537 to i32, !dbg !7
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1538), !dbg !7
  %1540 = bitcast i32 %1539 to float, !dbg !7
  %1541 = select i1 %.not36, float %1529, float %1540, !dbg !7
  %1542 = select i1 %.not36, float %1540, float %1535, !dbg !7
  %1543 = select i1 %.not36, float %1536, float %1530, !dbg !7
  %1544 = bitcast float %1543 to i32, !dbg !7
  %1545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1544), !dbg !7
  %1546 = bitcast i32 %1545 to float, !dbg !7
  %1547 = select i1 %.not36, float %1530, float %1546, !dbg !7
  %1548 = select i1 %.not36, float %1546, float %1536, !dbg !7
  %1549 = extractvalue [2 x <8 x float>] %133, 1, !dbg !7
  %1550 = extractelement <8 x float> %1549, i64 0, !dbg !7
  %1551 = extractelement <8 x float> %1549, i64 1, !dbg !7
  %1552 = extractelement <8 x float> %1549, i64 2, !dbg !7
  %1553 = extractelement <8 x float> %1549, i64 3, !dbg !7
  %1554 = select i1 %.not, float %1551, float %1550, !dbg !7
  %1555 = bitcast float %1554 to i32, !dbg !7
  %1556 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1555), !dbg !7
  %1557 = bitcast i32 %1556 to float, !dbg !7
  %1558 = select i1 %.not, float %1550, float %1557, !dbg !7
  %1559 = select i1 %.not, float %1557, float %1551, !dbg !7
  %1560 = select i1 %.not, float %1553, float %1552, !dbg !7
  %1561 = bitcast float %1560 to i32, !dbg !7
  %1562 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1561), !dbg !7
  %1563 = bitcast i32 %1562 to float, !dbg !7
  %1564 = select i1 %.not, float %1552, float %1563, !dbg !7
  %1565 = select i1 %.not, float %1563, float %1553, !dbg !7
  %1566 = select i1 %.not36, float %1564, float %1558, !dbg !7
  %1567 = bitcast float %1566 to i32, !dbg !7
  %1568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1567), !dbg !7
  %1569 = bitcast i32 %1568 to float, !dbg !7
  %1570 = select i1 %.not36, float %1558, float %1569, !dbg !7
  %1571 = select i1 %.not36, float %1569, float %1564, !dbg !7
  %1572 = select i1 %.not36, float %1565, float %1559, !dbg !7
  %1573 = bitcast float %1572 to i32, !dbg !7
  %1574 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1573), !dbg !7
  %1575 = bitcast i32 %1574 to float, !dbg !7
  %1576 = select i1 %.not36, float %1559, float %1575, !dbg !7
  %1577 = select i1 %.not36, float %1575, float %1565, !dbg !7
  %1578 = extractvalue [2 x <8 x float>] %132, 1, !dbg !7
  %1579 = extractelement <8 x float> %1578, i64 0, !dbg !7
  %1580 = extractelement <8 x float> %1578, i64 1, !dbg !7
  %1581 = extractelement <8 x float> %1578, i64 2, !dbg !7
  %1582 = extractelement <8 x float> %1578, i64 3, !dbg !7
  %1583 = select i1 %.not, float %1580, float %1579, !dbg !7
  %1584 = bitcast float %1583 to i32, !dbg !7
  %1585 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1584), !dbg !7
  %1586 = bitcast i32 %1585 to float, !dbg !7
  %1587 = select i1 %.not, float %1579, float %1586, !dbg !7
  %1588 = select i1 %.not, float %1586, float %1580, !dbg !7
  %1589 = select i1 %.not, float %1582, float %1581, !dbg !7
  %1590 = bitcast float %1589 to i32, !dbg !7
  %1591 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1590), !dbg !7
  %1592 = bitcast i32 %1591 to float, !dbg !7
  %1593 = select i1 %.not, float %1581, float %1592, !dbg !7
  %1594 = select i1 %.not, float %1592, float %1582, !dbg !7
  %1595 = select i1 %.not36, float %1593, float %1587, !dbg !7
  %1596 = bitcast float %1595 to i32, !dbg !7
  %1597 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1596), !dbg !7
  %1598 = bitcast i32 %1597 to float, !dbg !7
  %1599 = select i1 %.not36, float %1587, float %1598, !dbg !7
  %1600 = select i1 %.not36, float %1598, float %1593, !dbg !7
  %1601 = select i1 %.not36, float %1594, float %1588, !dbg !7
  %1602 = bitcast float %1601 to i32, !dbg !7
  %1603 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1602), !dbg !7
  %1604 = bitcast i32 %1603 to float, !dbg !7
  %1605 = select i1 %.not36, float %1588, float %1604, !dbg !7
  %1606 = select i1 %.not36, float %1604, float %1594, !dbg !7
  %1607 = extractvalue [2 x <8 x float>] %129, 1, !dbg !7
  %1608 = extractelement <8 x float> %1607, i64 0, !dbg !7
  %1609 = extractelement <8 x float> %1607, i64 1, !dbg !7
  %1610 = extractelement <8 x float> %1607, i64 2, !dbg !7
  %1611 = extractelement <8 x float> %1607, i64 3, !dbg !7
  %1612 = select i1 %.not, float %1609, float %1608, !dbg !7
  %1613 = bitcast float %1612 to i32, !dbg !7
  %1614 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1613), !dbg !7
  %1615 = bitcast i32 %1614 to float, !dbg !7
  %1616 = select i1 %.not, float %1608, float %1615, !dbg !7
  %1617 = select i1 %.not, float %1615, float %1609, !dbg !7
  %1618 = select i1 %.not, float %1611, float %1610, !dbg !7
  %1619 = bitcast float %1618 to i32, !dbg !7
  %1620 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1619), !dbg !7
  %1621 = bitcast i32 %1620 to float, !dbg !7
  %1622 = select i1 %.not, float %1610, float %1621, !dbg !7
  %1623 = select i1 %.not, float %1621, float %1611, !dbg !7
  %1624 = select i1 %.not36, float %1622, float %1616, !dbg !7
  %1625 = bitcast float %1624 to i32, !dbg !7
  %1626 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1625), !dbg !7
  %1627 = select i1 %.not36, float %1623, float %1617, !dbg !7
  %1628 = bitcast float %1627 to i32, !dbg !7
  %1629 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1628), !dbg !7
  %1630 = extractvalue [2 x <1 x float>] %130, 1, !dbg !16
  %1631 = shufflevector <1 x float> %1630, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1632 = extractelement <8 x float> %1520, i64 4, !dbg !7
  %1633 = extractelement <8 x float> %1520, i64 5, !dbg !7
  %1634 = extractelement <8 x float> %1520, i64 6, !dbg !7
  %1635 = extractelement <8 x float> %1520, i64 7, !dbg !7
  %1636 = select i1 %.not, float %1633, float %1632, !dbg !7
  %1637 = bitcast float %1636 to i32, !dbg !7
  %1638 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1637), !dbg !7
  %1639 = bitcast i32 %1638 to float, !dbg !7
  %1640 = select i1 %.not, float %1632, float %1639, !dbg !7
  %1641 = select i1 %.not, float %1639, float %1633, !dbg !7
  %1642 = select i1 %.not, float %1635, float %1634, !dbg !7
  %1643 = bitcast float %1642 to i32, !dbg !7
  %1644 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1643), !dbg !7
  %1645 = bitcast i32 %1644 to float, !dbg !7
  %1646 = select i1 %.not, float %1634, float %1645, !dbg !7
  %1647 = select i1 %.not, float %1645, float %1635, !dbg !7
  %1648 = select i1 %.not36, float %1646, float %1640, !dbg !7
  %1649 = bitcast float %1648 to i32, !dbg !7
  %1650 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1649), !dbg !7
  %1651 = bitcast i32 %1650 to float, !dbg !7
  %1652 = select i1 %.not36, float %1640, float %1651, !dbg !7
  %1653 = select i1 %.not36, float %1651, float %1646, !dbg !7
  %1654 = select i1 %.not36, float %1647, float %1641, !dbg !7
  %1655 = bitcast float %1654 to i32, !dbg !7
  %1656 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1655), !dbg !7
  %1657 = bitcast i32 %1656 to float, !dbg !7
  %1658 = select i1 %.not36, float %1641, float %1657, !dbg !7
  %1659 = select i1 %.not36, float %1657, float %1647, !dbg !7
  %1660 = extractelement <8 x float> %1549, i64 4, !dbg !7
  %1661 = extractelement <8 x float> %1549, i64 5, !dbg !7
  %1662 = extractelement <8 x float> %1549, i64 6, !dbg !7
  %1663 = extractelement <8 x float> %1549, i64 7, !dbg !7
  %1664 = select i1 %.not, float %1661, float %1660, !dbg !7
  %1665 = bitcast float %1664 to i32, !dbg !7
  %1666 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1665), !dbg !7
  %1667 = bitcast i32 %1666 to float, !dbg !7
  %1668 = select i1 %.not, float %1660, float %1667, !dbg !7
  %1669 = select i1 %.not, float %1667, float %1661, !dbg !7
  %1670 = select i1 %.not, float %1663, float %1662, !dbg !7
  %1671 = bitcast float %1670 to i32, !dbg !7
  %1672 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1671), !dbg !7
  %1673 = bitcast i32 %1672 to float, !dbg !7
  %1674 = select i1 %.not, float %1662, float %1673, !dbg !7
  %1675 = select i1 %.not, float %1673, float %1663, !dbg !7
  %1676 = select i1 %.not36, float %1674, float %1668, !dbg !7
  %1677 = bitcast float %1676 to i32, !dbg !7
  %1678 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1677), !dbg !7
  %1679 = bitcast i32 %1678 to float, !dbg !7
  %1680 = select i1 %.not36, float %1668, float %1679, !dbg !7
  %1681 = select i1 %.not36, float %1679, float %1674, !dbg !7
  %1682 = select i1 %.not36, float %1675, float %1669, !dbg !7
  %1683 = bitcast float %1682 to i32, !dbg !7
  %1684 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1683), !dbg !7
  %1685 = bitcast i32 %1684 to float, !dbg !7
  %1686 = select i1 %.not36, float %1669, float %1685, !dbg !7
  %1687 = select i1 %.not36, float %1685, float %1675, !dbg !7
  %1688 = extractelement <8 x float> %1578, i64 4, !dbg !7
  %1689 = extractelement <8 x float> %1578, i64 5, !dbg !7
  %1690 = extractelement <8 x float> %1578, i64 6, !dbg !7
  %1691 = extractelement <8 x float> %1578, i64 7, !dbg !7
  %1692 = select i1 %.not, float %1689, float %1688, !dbg !7
  %1693 = bitcast float %1692 to i32, !dbg !7
  %1694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1693), !dbg !7
  %1695 = bitcast i32 %1694 to float, !dbg !7
  %1696 = select i1 %.not, float %1688, float %1695, !dbg !7
  %1697 = select i1 %.not, float %1695, float %1689, !dbg !7
  %1698 = select i1 %.not, float %1691, float %1690, !dbg !7
  %1699 = bitcast float %1698 to i32, !dbg !7
  %1700 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1699), !dbg !7
  %1701 = bitcast i32 %1700 to float, !dbg !7
  %1702 = select i1 %.not, float %1690, float %1701, !dbg !7
  %1703 = select i1 %.not, float %1701, float %1691, !dbg !7
  %1704 = select i1 %.not36, float %1702, float %1696, !dbg !7
  %1705 = bitcast float %1704 to i32, !dbg !7
  %1706 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1705), !dbg !7
  %1707 = bitcast i32 %1706 to float, !dbg !7
  %1708 = select i1 %.not36, float %1696, float %1707, !dbg !7
  %1709 = select i1 %.not36, float %1707, float %1702, !dbg !7
  %1710 = select i1 %.not36, float %1703, float %1697, !dbg !7
  %1711 = bitcast float %1710 to i32, !dbg !7
  %1712 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1711), !dbg !7
  %1713 = bitcast i32 %1712 to float, !dbg !7
  %1714 = select i1 %.not36, float %1697, float %1713, !dbg !7
  %1715 = select i1 %.not36, float %1713, float %1703, !dbg !7
  %1716 = extractelement <8 x float> %1607, i64 4, !dbg !7
  %1717 = extractelement <8 x float> %1607, i64 5, !dbg !7
  %1718 = extractelement <8 x float> %1607, i64 6, !dbg !7
  %1719 = extractelement <8 x float> %1607, i64 7, !dbg !7
  %1720 = select i1 %.not, float %1717, float %1716, !dbg !7
  %1721 = bitcast float %1720 to i32, !dbg !7
  %1722 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1721), !dbg !7
  %1723 = bitcast i32 %1722 to float, !dbg !7
  %1724 = select i1 %.not, float %1716, float %1723, !dbg !7
  %1725 = select i1 %.not, float %1723, float %1717, !dbg !7
  %1726 = select i1 %.not, float %1719, float %1718, !dbg !7
  %1727 = bitcast float %1726 to i32, !dbg !7
  %1728 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1727), !dbg !7
  %1729 = bitcast i32 %1728 to float, !dbg !7
  %1730 = select i1 %.not, float %1718, float %1729, !dbg !7
  %1731 = select i1 %.not, float %1729, float %1719, !dbg !7
  %1732 = select i1 %.not36, float %1730, float %1724, !dbg !7
  %1733 = bitcast float %1732 to i32, !dbg !7
  %1734 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1733), !dbg !7
  %1735 = select i1 %.not36, float %1731, float %1725, !dbg !7
  %1736 = bitcast float %1735 to i32, !dbg !7
  %1737 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1736), !dbg !7
  %1738 = bitcast i32 %1734 to float, !dbg !7
  %1739 = select i1 %.not36, float %1724, float %1738, !dbg !7
  %1740 = insertelement <4 x float> poison, float %1739, i64 0, !dbg !16
  %1741 = bitcast i32 %1737 to float, !dbg !7
  %1742 = select i1 %.not36, float %1725, float %1741, !dbg !7
  %1743 = insertelement <4 x float> %1740, float %1742, i64 1, !dbg !16
  %1744 = select i1 %.not36, float %1738, float %1730, !dbg !7
  %1745 = insertelement <4 x float> %1743, float %1744, i64 2, !dbg !16
  %1746 = select i1 %.not36, float %1741, float %1731, !dbg !7
  %1747 = insertelement <4 x float> %1745, float %1746, i64 3, !dbg !16
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1631, !dbg !16
  %normalized = fmul <4 x float> %1747, %row.reciprocal, !dbg !16
  %1748 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !17
  %1749 = extractelement <4 x half> %1748, i64 3, !dbg !17
  %1750 = extractelement <4 x half> %1748, i64 2, !dbg !17
  %1751 = extractelement <4 x half> %1748, i64 1, !dbg !17
  %1752 = extractelement <4 x half> %1748, i64 0, !dbg !17
  %1753 = bitcast i32 %1626 to float, !dbg !7
  %1754 = select i1 %.not36, float %1616, float %1753, !dbg !7
  %1755 = insertelement <4 x float> poison, float %1754, i64 0, !dbg !16
  %1756 = bitcast i32 %1629 to float, !dbg !7
  %1757 = select i1 %.not36, float %1617, float %1756, !dbg !7
  %1758 = insertelement <4 x float> %1755, float %1757, i64 1, !dbg !16
  %1759 = select i1 %.not36, float %1753, float %1622, !dbg !7
  %1760 = insertelement <4 x float> %1758, float %1759, i64 2, !dbg !16
  %1761 = select i1 %.not36, float %1756, float %1623, !dbg !7
  %1762 = insertelement <4 x float> %1760, float %1761, i64 3, !dbg !16
  %normalized37 = fmul <4 x float> %1762, %row.reciprocal, !dbg !16
  %1763 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !17
  %1764 = extractelement <4 x half> %1763, i64 3, !dbg !17
  %1765 = extractelement <4 x half> %1763, i64 2, !dbg !17
  %1766 = extractelement <4 x half> %1763, i64 1, !dbg !17
  %1767 = extractelement <4 x half> %1763, i64 0, !dbg !17
  %1768 = bitcast i32 %1516 to float, !dbg !7
  %1769 = select i1 %.not36, float %1506, float %1768, !dbg !7
  %1770 = insertelement <4 x float> poison, float %1769, i64 0, !dbg !16
  %1771 = bitcast i32 %1519 to float, !dbg !7
  %1772 = select i1 %.not36, float %1507, float %1771, !dbg !7
  %1773 = insertelement <4 x float> %1770, float %1772, i64 1, !dbg !16
  %1774 = select i1 %.not36, float %1768, float %1512, !dbg !7
  %1775 = insertelement <4 x float> %1773, float %1774, i64 2, !dbg !16
  %1776 = select i1 %.not36, float %1771, float %1513, !dbg !7
  %1777 = insertelement <4 x float> %1775, float %1776, i64 3, !dbg !16
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1413, !dbg !16
  %normalized39 = fmul <4 x float> %1777, %row.reciprocal38, !dbg !16
  %1778 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !17
  %1779 = extractelement <4 x half> %1778, i64 3, !dbg !17
  %1780 = extractelement <4 x half> %1778, i64 2, !dbg !17
  %1781 = extractelement <4 x half> %1778, i64 1, !dbg !17
  %1782 = extractelement <4 x half> %1778, i64 0, !dbg !17
  %1783 = bitcast i32 %1408 to float, !dbg !7
  %1784 = select i1 %.not36, float %1398, float %1783, !dbg !7
  %1785 = insertelement <4 x float> poison, float %1784, i64 0, !dbg !16
  %1786 = bitcast i32 %1411 to float, !dbg !7
  %1787 = select i1 %.not36, float %1399, float %1786, !dbg !7
  %1788 = insertelement <4 x float> %1785, float %1787, i64 1, !dbg !16
  %1789 = select i1 %.not36, float %1783, float %1404, !dbg !7
  %1790 = insertelement <4 x float> %1788, float %1789, i64 2, !dbg !16
  %1791 = select i1 %.not36, float %1786, float %1405, !dbg !7
  %1792 = insertelement <4 x float> %1790, float %1791, i64 3, !dbg !16
  %normalized40 = fmul <4 x float> %1792, %row.reciprocal38, !dbg !16
  %1793 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !17
  %1794 = extractelement <4 x half> %1793, i64 3, !dbg !17
  %1795 = extractelement <4 x half> %1793, i64 2, !dbg !17
  %1796 = extractelement <4 x half> %1793, i64 1, !dbg !17
  %1797 = extractelement <4 x half> %1793, i64 0, !dbg !17
  %1798 = add nsw i64 %18, %24, !dbg !18
  %1799 = shl i64 %1798, 19, !dbg !18
  %1800 = and i64 %1799, 16252928, !dbg !18
  %address.thread841 = call i32 asm sideeffect "", "=v,0"(i32 %33) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local842 = zext nneg i32 %address.thread841 to i64, !dbg !8
  %address.local843 = add i64 %27, %address.local842, !dbg !18
  %1801 = lshr i64 %address.local842, 2, !dbg !18
  %address.local845 = and i64 %1801, 48, !dbg !18
  %address.local846 = add i64 %address.local843, %address.local845, !dbg !18
  %address.local847 = and i64 %address.local842, 240, !dbg !18
  %address.local848 = sub i64 %address.local846, %address.local847, !dbg !18
  %address.local849 = add i64 %address.local848, %28, !dbg !18
  %address.local850 = shl nuw nsw i64 %address.local849, 7, !dbg !18
  %address.local851 = add nuw nsw i64 %1800, %address.local850, !dbg !18
  %address.local852 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local853 = lshr i64 %address.local842, 2, !dbg !10
  %address.local854 = and i64 %address.local853, 12, !dbg !10
  %address.local855 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local854, !dbg !18
  store half %1797, ptr addrspace(1) %address.local855, align 2, !dbg !18
  %address.local856 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local857 = or disjoint i64 %address.local854, 1, !dbg !18
  %address.local858 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local856, i64 %address.local857, !dbg !18
  store half %1796, ptr addrspace(1) %address.local858, align 2, !dbg !18
  %address.local859 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local860 = or disjoint i64 %address.local854, 2, !dbg !18
  %address.local861 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local860, !dbg !18
  store half %1795, ptr addrspace(1) %address.local861, align 2, !dbg !18
  %address.local862 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local863 = or disjoint i64 %address.local854, 3, !dbg !18
  %address.local864 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local862, i64 %address.local863, !dbg !18
  store half %1794, ptr addrspace(1) %address.local864, align 2, !dbg !18
  %address.local865 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local866 = or disjoint i64 %address.local854, 16, !dbg !18
  %address.local867 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local865, i64 %address.local866, !dbg !18
  store half %1782, ptr addrspace(1) %address.local867, align 2, !dbg !18
  %address.local868 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local869 = or disjoint i64 %address.local854, 17, !dbg !18
  %address.local870 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local868, i64 %address.local869, !dbg !18
  store half %1781, ptr addrspace(1) %address.local870, align 2, !dbg !18
  %address.local871 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local872 = or disjoint i64 %address.local854, 18, !dbg !18
  %address.local873 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local871, i64 %address.local872, !dbg !18
  store half %1780, ptr addrspace(1) %address.local873, align 2, !dbg !18
  %address.local874 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local851, !dbg !18
  %address.local875 = or disjoint i64 %address.local854, 19, !dbg !18
  %address.local876 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local874, i64 %address.local875, !dbg !18
  store half %1779, ptr addrspace(1) %address.local876, align 2, !dbg !18
  %address.local877 = shl i64 %address.local849, 7, !dbg !18
  %address.local878 = add i64 %address.local877, 8192, !dbg !18
  %address.local879 = add nuw nsw i64 %1800, %address.local878, !dbg !18
  %address.local880 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local881 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local854, !dbg !18
  store half %1767, ptr addrspace(1) %address.local881, align 2, !dbg !18
  %address.local882 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local882, i64 %address.local857, !dbg !18
  store half %1766, ptr addrspace(1) %address.local883, align 2, !dbg !18
  %address.local884 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local885 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local884, i64 %address.local860, !dbg !18
  store half %1765, ptr addrspace(1) %address.local885, align 2, !dbg !18
  %address.local886 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local887 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local886, i64 %address.local863, !dbg !18
  store half %1764, ptr addrspace(1) %address.local887, align 2, !dbg !18
  %address.local888 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local888, i64 %address.local866, !dbg !18
  store half %1752, ptr addrspace(1) %address.local889, align 2, !dbg !18
  %address.local890 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local891 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local890, i64 %address.local869, !dbg !18
  store half %1751, ptr addrspace(1) %address.local891, align 2, !dbg !18
  %address.local892 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local893 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local892, i64 %address.local872, !dbg !18
  store half %1750, ptr addrspace(1) %address.local893, align 2, !dbg !18
  %address.local894 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local879, !dbg !18
  %address.local895 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local875, !dbg !18
  store half %1749, ptr addrspace(1) %address.local895, align 2, !dbg !18
  %1802 = insertelement <4 x float> poison, float %1708, i64 0, !dbg !16
  %1803 = insertelement <4 x float> %1802, float %1714, i64 1, !dbg !16
  %1804 = insertelement <4 x float> %1803, float %1709, i64 2, !dbg !16
  %1805 = insertelement <4 x float> %1804, float %1715, i64 3, !dbg !16
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1631, !dbg !16
  %normalized42 = fmul <4 x float> %1805, %row.reciprocal41, !dbg !16
  %1806 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !17
  %1807 = extractelement <4 x half> %1806, i64 3, !dbg !17
  %1808 = extractelement <4 x half> %1806, i64 2, !dbg !17
  %1809 = extractelement <4 x half> %1806, i64 1, !dbg !17
  %1810 = extractelement <4 x half> %1806, i64 0, !dbg !17
  %1811 = insertelement <4 x float> poison, float %1599, i64 0, !dbg !16
  %1812 = insertelement <4 x float> %1811, float %1605, i64 1, !dbg !16
  %1813 = insertelement <4 x float> %1812, float %1600, i64 2, !dbg !16
  %1814 = insertelement <4 x float> %1813, float %1606, i64 3, !dbg !16
  %normalized43 = fmul <4 x float> %1814, %row.reciprocal41, !dbg !16
  %1815 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !17
  %1816 = extractelement <4 x half> %1815, i64 3, !dbg !17
  %1817 = extractelement <4 x half> %1815, i64 2, !dbg !17
  %1818 = extractelement <4 x half> %1815, i64 1, !dbg !17
  %1819 = extractelement <4 x half> %1815, i64 0, !dbg !17
  %1820 = insertelement <4 x float> poison, float %1490, i64 0, !dbg !16
  %1821 = insertelement <4 x float> %1820, float %1496, i64 1, !dbg !16
  %1822 = insertelement <4 x float> %1821, float %1491, i64 2, !dbg !16
  %1823 = insertelement <4 x float> %1822, float %1497, i64 3, !dbg !16
  %row.reciprocal44 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1413, !dbg !16
  %normalized45 = fmul <4 x float> %1823, %row.reciprocal44, !dbg !16
  %1824 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !17
  %1825 = extractelement <4 x half> %1824, i64 3, !dbg !17
  %1826 = extractelement <4 x half> %1824, i64 2, !dbg !17
  %1827 = extractelement <4 x half> %1824, i64 1, !dbg !17
  %1828 = extractelement <4 x half> %1824, i64 0, !dbg !17
  %1829 = insertelement <4 x float> poison, float %1381, i64 0, !dbg !16
  %1830 = insertelement <4 x float> %1829, float %1387, i64 1, !dbg !16
  %1831 = insertelement <4 x float> %1830, float %1382, i64 2, !dbg !16
  %1832 = insertelement <4 x float> %1831, float %1388, i64 3, !dbg !16
  %normalized46 = fmul <4 x float> %1832, %row.reciprocal44, !dbg !16
  %1833 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !17
  %1834 = extractelement <4 x half> %1833, i64 3, !dbg !17
  %1835 = extractelement <4 x half> %1833, i64 2, !dbg !17
  %1836 = extractelement <4 x half> %1833, i64 1, !dbg !17
  %1837 = extractelement <4 x half> %1833, i64 0, !dbg !17
  %1838 = shl nuw nsw i64 %26, 19, !dbg !18
  %address.thread896 = call i32 asm sideeffect "", "=v,0"(i32 %33) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local897 = zext nneg i32 %address.thread896 to i64, !dbg !8
  %address.local898 = add i64 %27, %address.local897, !dbg !18
  %1839 = lshr i64 %address.local897, 2, !dbg !18
  %address.local900 = and i64 %1839, 48, !dbg !18
  %address.local901 = add i64 %address.local898, %address.local900, !dbg !18
  %address.local902 = and i64 %address.local897, 240, !dbg !18
  %address.local903 = sub i64 %address.local901, %address.local902, !dbg !18
  %address.local904 = add i64 %address.local903, %28, !dbg !18
  %address.local905 = shl nuw nsw i64 %address.local904, 7, !dbg !18
  %address.local906 = add nuw nsw i64 %1838, %address.local905, !dbg !18
  %address.local907 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local908 = lshr i64 %address.local897, 2, !dbg !10
  %address.local909 = and i64 %address.local908, 12, !dbg !10
  %address.local910 = or disjoint i64 %address.local909, 32, !dbg !18
  %address.local911 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local907, i64 %address.local910, !dbg !18
  store half %1837, ptr addrspace(1) %address.local911, align 2, !dbg !18
  %address.local912 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local913 = or disjoint i64 %address.local909, 33, !dbg !18
  %address.local914 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local912, i64 %address.local913, !dbg !18
  store half %1836, ptr addrspace(1) %address.local914, align 2, !dbg !18
  %address.local915 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local916 = or disjoint i64 %address.local909, 34, !dbg !18
  %address.local917 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local915, i64 %address.local916, !dbg !18
  store half %1835, ptr addrspace(1) %address.local917, align 2, !dbg !18
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local919 = or disjoint i64 %address.local909, 35, !dbg !18
  %address.local920 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local918, i64 %address.local919, !dbg !18
  store half %1834, ptr addrspace(1) %address.local920, align 2, !dbg !18
  %address.local921 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local922 = or disjoint i64 %address.local909, 48, !dbg !18
  %address.local923 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local921, i64 %address.local922, !dbg !18
  store half %1828, ptr addrspace(1) %address.local923, align 2, !dbg !18
  %address.local924 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local925 = or disjoint i64 %address.local909, 49, !dbg !18
  %address.local926 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local924, i64 %address.local925, !dbg !18
  store half %1827, ptr addrspace(1) %address.local926, align 2, !dbg !18
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local928 = or disjoint i64 %address.local909, 50, !dbg !18
  %address.local929 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local927, i64 %address.local928, !dbg !18
  store half %1826, ptr addrspace(1) %address.local929, align 2, !dbg !18
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local906, !dbg !18
  %address.local931 = or i64 %address.local908, 51, !dbg !18
  %address.local932 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local930, i64 %address.local931, !dbg !18
  store half %1825, ptr addrspace(1) %address.local932, align 2, !dbg !18
  %address.local933 = shl i64 %address.local904, 7, !dbg !18
  %address.local934 = add i64 %address.local933, 8192, !dbg !18
  %address.local935 = add nuw nsw i64 %1838, %address.local934, !dbg !18
  %address.local936 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local936, i64 %address.local910, !dbg !18
  store half %1819, ptr addrspace(1) %address.local937, align 2, !dbg !18
  %address.local938 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local938, i64 %address.local913, !dbg !18
  store half %1818, ptr addrspace(1) %address.local939, align 2, !dbg !18
  %address.local940 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local941 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local940, i64 %address.local916, !dbg !18
  store half %1817, ptr addrspace(1) %address.local941, align 2, !dbg !18
  %address.local942 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local943 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local942, i64 %address.local919, !dbg !18
  store half %1816, ptr addrspace(1) %address.local943, align 2, !dbg !18
  %address.local944 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local922, !dbg !18
  store half %1810, ptr addrspace(1) %address.local945, align 2, !dbg !18
  %address.local946 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local946, i64 %address.local925, !dbg !18
  store half %1809, ptr addrspace(1) %address.local947, align 2, !dbg !18
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local948, i64 %address.local928, !dbg !18
  store half %1808, ptr addrspace(1) %address.local949, align 2, !dbg !18
  %address.local950 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local935, !dbg !18
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local950, i64 %address.local931, !dbg !18
  store half %1807, ptr addrspace(1) %address.local951, align 2, !dbg !18
  %1840 = insertelement <4 x float> poison, float %1680, i64 0, !dbg !16
  %1841 = insertelement <4 x float> %1840, float %1686, i64 1, !dbg !16
  %1842 = insertelement <4 x float> %1841, float %1681, i64 2, !dbg !16
  %1843 = insertelement <4 x float> %1842, float %1687, i64 3, !dbg !16
  %row.reciprocal47 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1631, !dbg !16
  %normalized48 = fmul <4 x float> %1843, %row.reciprocal47, !dbg !16
  %1844 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !17
  %1845 = extractelement <4 x half> %1844, i64 3, !dbg !17
  %1846 = extractelement <4 x half> %1844, i64 2, !dbg !17
  %1847 = extractelement <4 x half> %1844, i64 1, !dbg !17
  %1848 = extractelement <4 x half> %1844, i64 0, !dbg !17
  %1849 = insertelement <4 x float> poison, float %1570, i64 0, !dbg !16
  %1850 = insertelement <4 x float> %1849, float %1576, i64 1, !dbg !16
  %1851 = insertelement <4 x float> %1850, float %1571, i64 2, !dbg !16
  %1852 = insertelement <4 x float> %1851, float %1577, i64 3, !dbg !16
  %normalized49 = fmul <4 x float> %1852, %row.reciprocal47, !dbg !16
  %1853 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !17
  %1854 = extractelement <4 x half> %1853, i64 3, !dbg !17
  %1855 = extractelement <4 x half> %1853, i64 2, !dbg !17
  %1856 = extractelement <4 x half> %1853, i64 1, !dbg !17
  %1857 = extractelement <4 x half> %1853, i64 0, !dbg !17
  %1858 = insertelement <4 x float> poison, float %1462, i64 0, !dbg !16
  %1859 = insertelement <4 x float> %1858, float %1468, i64 1, !dbg !16
  %1860 = insertelement <4 x float> %1859, float %1463, i64 2, !dbg !16
  %1861 = insertelement <4 x float> %1860, float %1469, i64 3, !dbg !16
  %row.reciprocal50 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1413, !dbg !16
  %normalized51 = fmul <4 x float> %1861, %row.reciprocal50, !dbg !16
  %1862 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !17
  %1863 = extractelement <4 x half> %1862, i64 3, !dbg !17
  %1864 = extractelement <4 x half> %1862, i64 2, !dbg !17
  %1865 = extractelement <4 x half> %1862, i64 1, !dbg !17
  %1866 = extractelement <4 x half> %1862, i64 0, !dbg !17
  %1867 = insertelement <4 x float> poison, float %1352, i64 0, !dbg !16
  %1868 = insertelement <4 x float> %1867, float %1358, i64 1, !dbg !16
  %1869 = insertelement <4 x float> %1868, float %1353, i64 2, !dbg !16
  %1870 = insertelement <4 x float> %1869, float %1359, i64 3, !dbg !16
  %normalized52 = fmul <4 x float> %1870, %row.reciprocal50, !dbg !16
  %1871 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !17
  %1872 = extractelement <4 x half> %1871, i64 3, !dbg !17
  %1873 = extractelement <4 x half> %1871, i64 2, !dbg !17
  %1874 = extractelement <4 x half> %1871, i64 1, !dbg !17
  %1875 = extractelement <4 x half> %1871, i64 0, !dbg !17
  %1876 = shl nuw nsw i64 %26, 19, !dbg !18
  %address.thread952 = call i32 asm sideeffect "", "=v,0"(i32 %33) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local953 = zext nneg i32 %address.thread952 to i64, !dbg !8
  %address.local954 = add i64 %27, %address.local953, !dbg !18
  %1877 = lshr i64 %address.local953, 2, !dbg !18
  %address.local956 = and i64 %1877, 48, !dbg !18
  %address.local957 = add i64 %address.local954, %address.local956, !dbg !18
  %address.local958 = and i64 %address.local953, 240, !dbg !18
  %address.local959 = sub i64 %address.local957, %address.local958, !dbg !18
  %address.local960 = add i64 %address.local959, %28, !dbg !18
  %address.local961 = shl nuw nsw i64 %address.local960, 7, !dbg !18
  %address.local962 = add nuw nsw i64 %1876, %address.local961, !dbg !18
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local964 = lshr i64 %address.local953, 2, !dbg !10
  %address.local965 = and i64 %address.local964, 12, !dbg !10
  %address.local966 = or disjoint i64 %address.local965, 64, !dbg !18
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local963, i64 %address.local966, !dbg !18
  store half %1875, ptr addrspace(1) %address.local967, align 2, !dbg !18
  %address.local968 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local969 = or disjoint i64 %address.local965, 65, !dbg !18
  %address.local970 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local968, i64 %address.local969, !dbg !18
  store half %1874, ptr addrspace(1) %address.local970, align 2, !dbg !18
  %address.local971 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local972 = or disjoint i64 %address.local965, 66, !dbg !18
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local971, i64 %address.local972, !dbg !18
  store half %1873, ptr addrspace(1) %address.local973, align 2, !dbg !18
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local975 = or disjoint i64 %address.local965, 67, !dbg !18
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local974, i64 %address.local975, !dbg !18
  store half %1872, ptr addrspace(1) %address.local976, align 2, !dbg !18
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local978 = or disjoint i64 %address.local965, 80, !dbg !18
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local977, i64 %address.local978, !dbg !18
  store half %1866, ptr addrspace(1) %address.local979, align 2, !dbg !18
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local981 = or disjoint i64 %address.local965, 81, !dbg !18
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local980, i64 %address.local981, !dbg !18
  store half %1865, ptr addrspace(1) %address.local982, align 2, !dbg !18
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local984 = or disjoint i64 %address.local965, 82, !dbg !18
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local983, i64 %address.local984, !dbg !18
  store half %1864, ptr addrspace(1) %address.local985, align 2, !dbg !18
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local962, !dbg !18
  %address.local987 = or disjoint i64 %address.local965, 83, !dbg !18
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local986, i64 %address.local987, !dbg !18
  store half %1863, ptr addrspace(1) %address.local988, align 2, !dbg !18
  %address.local989 = shl i64 %address.local960, 7, !dbg !18
  %address.local990 = add i64 %address.local989, 8192, !dbg !18
  %address.local991 = add nuw nsw i64 %1876, %address.local990, !dbg !18
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local992, i64 %address.local966, !dbg !18
  store half %1857, ptr addrspace(1) %address.local993, align 2, !dbg !18
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local994, i64 %address.local969, !dbg !18
  store half %1856, ptr addrspace(1) %address.local995, align 2, !dbg !18
  %address.local996 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local996, i64 %address.local972, !dbg !18
  store half %1855, ptr addrspace(1) %address.local997, align 2, !dbg !18
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local999 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local998, i64 %address.local975, !dbg !18
  store half %1854, ptr addrspace(1) %address.local999, align 2, !dbg !18
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1000, i64 %address.local978, !dbg !18
  store half %1848, ptr addrspace(1) %address.local1001, align 2, !dbg !18
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1002, i64 %address.local981, !dbg !18
  store half %1847, ptr addrspace(1) %address.local1003, align 2, !dbg !18
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local984, !dbg !18
  store half %1846, ptr addrspace(1) %address.local1005, align 2, !dbg !18
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local991, !dbg !18
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1006, i64 %address.local987, !dbg !18
  store half %1845, ptr addrspace(1) %address.local1007, align 2, !dbg !18
  %1878 = insertelement <4 x float> poison, float %1652, i64 0, !dbg !16
  %1879 = insertelement <4 x float> %1878, float %1658, i64 1, !dbg !16
  %1880 = insertelement <4 x float> %1879, float %1653, i64 2, !dbg !16
  %1881 = insertelement <4 x float> %1880, float %1659, i64 3, !dbg !16
  %row.reciprocal53 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1631, !dbg !16
  %normalized54 = fmul <4 x float> %1881, %row.reciprocal53, !dbg !16
  %1882 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !17
  %1883 = extractelement <4 x half> %1882, i64 3, !dbg !17
  %1884 = extractelement <4 x half> %1882, i64 2, !dbg !17
  %1885 = extractelement <4 x half> %1882, i64 1, !dbg !17
  %1886 = extractelement <4 x half> %1882, i64 0, !dbg !17
  %1887 = insertelement <4 x float> poison, float %1541, i64 0, !dbg !16
  %1888 = insertelement <4 x float> %1887, float %1547, i64 1, !dbg !16
  %1889 = insertelement <4 x float> %1888, float %1542, i64 2, !dbg !16
  %1890 = insertelement <4 x float> %1889, float %1548, i64 3, !dbg !16
  %normalized55 = fmul <4 x float> %1890, %row.reciprocal53, !dbg !16
  %1891 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !17
  %1892 = extractelement <4 x half> %1891, i64 3, !dbg !17
  %1893 = extractelement <4 x half> %1891, i64 2, !dbg !17
  %1894 = extractelement <4 x half> %1891, i64 1, !dbg !17
  %1895 = extractelement <4 x half> %1891, i64 0, !dbg !17
  %1896 = insertelement <4 x float> poison, float %1434, i64 0, !dbg !16
  %1897 = insertelement <4 x float> %1896, float %1440, i64 1, !dbg !16
  %1898 = insertelement <4 x float> %1897, float %1435, i64 2, !dbg !16
  %1899 = insertelement <4 x float> %1898, float %1441, i64 3, !dbg !16
  %row.reciprocal56 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1413, !dbg !16
  %normalized57 = fmul <4 x float> %1899, %row.reciprocal56, !dbg !16
  %1900 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !17
  %1901 = extractelement <4 x half> %1900, i64 3, !dbg !17
  %1902 = extractelement <4 x half> %1900, i64 2, !dbg !17
  %1903 = extractelement <4 x half> %1900, i64 1, !dbg !17
  %1904 = extractelement <4 x half> %1900, i64 0, !dbg !17
  %1905 = insertelement <4 x float> poison, float %1323, i64 0, !dbg !16
  %1906 = insertelement <4 x float> %1905, float %1329, i64 1, !dbg !16
  %1907 = insertelement <4 x float> %1906, float %1324, i64 2, !dbg !16
  %1908 = insertelement <4 x float> %1907, float %1330, i64 3, !dbg !16
  %normalized58 = fmul <4 x float> %1908, %row.reciprocal56, !dbg !16
  %1909 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !17
  %1910 = extractelement <4 x half> %1909, i64 3, !dbg !17
  %1911 = extractelement <4 x half> %1909, i64 2, !dbg !17
  %1912 = extractelement <4 x half> %1909, i64 1, !dbg !17
  %1913 = extractelement <4 x half> %1909, i64 0, !dbg !17
  %1914 = shl nuw nsw i64 %26, 19, !dbg !18
  %address.thread1008 = call i32 asm sideeffect "", "=v,0"(i32 %33) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1009 = zext nneg i32 %address.thread1008 to i64, !dbg !8
  %address.local1010 = add i64 %27, %address.local1009, !dbg !18
  %1915 = lshr i64 %address.local1009, 2, !dbg !18
  %address.local1012 = and i64 %1915, 48, !dbg !18
  %address.local1013 = add i64 %address.local1010, %address.local1012, !dbg !18
  %address.local1014 = and i64 %address.local1009, 240, !dbg !18
  %address.local1015 = sub i64 %address.local1013, %address.local1014, !dbg !18
  %address.local1016 = add i64 %address.local1015, %28, !dbg !18
  %address.local1017 = shl nuw nsw i64 %address.local1016, 7, !dbg !18
  %address.local1018 = add nuw nsw i64 %1914, %address.local1017, !dbg !18
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1020 = lshr i64 %address.local1009, 2, !dbg !10
  %address.local1021 = and i64 %address.local1020, 12, !dbg !10
  %address.local1022 = or disjoint i64 %address.local1021, 96, !dbg !18
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1019, i64 %address.local1022, !dbg !18
  store half %1913, ptr addrspace(1) %address.local1023, align 2, !dbg !18
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1025 = or disjoint i64 %address.local1021, 97, !dbg !18
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1024, i64 %address.local1025, !dbg !18
  store half %1912, ptr addrspace(1) %address.local1026, align 2, !dbg !18
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1028 = or disjoint i64 %address.local1021, 98, !dbg !18
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1027, i64 %address.local1028, !dbg !18
  store half %1911, ptr addrspace(1) %address.local1029, align 2, !dbg !18
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1031 = or disjoint i64 %address.local1021, 99, !dbg !18
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1030, i64 %address.local1031, !dbg !18
  store half %1910, ptr addrspace(1) %address.local1032, align 2, !dbg !18
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1034 = or disjoint i64 %address.local1021, 112, !dbg !18
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local1034, !dbg !18
  store half %1904, ptr addrspace(1) %address.local1035, align 2, !dbg !18
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1037 = or disjoint i64 %address.local1021, 113, !dbg !18
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1036, i64 %address.local1037, !dbg !18
  store half %1903, ptr addrspace(1) %address.local1038, align 2, !dbg !18
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1040 = or disjoint i64 %address.local1021, 114, !dbg !18
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1040, !dbg !18
  store half %1902, ptr addrspace(1) %address.local1041, align 2, !dbg !18
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1018, !dbg !18
  %address.local1043 = or i64 %address.local1020, 115, !dbg !18
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1042, i64 %address.local1043, !dbg !18
  store half %1901, ptr addrspace(1) %address.local1044, align 2, !dbg !18
  %address.local1045 = shl i64 %address.local1016, 7, !dbg !18
  %address.local1046 = add i64 %address.local1045, 8192, !dbg !18
  %address.local1047 = add nuw nsw i64 %1914, %address.local1046, !dbg !18
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1048, i64 %address.local1022, !dbg !18
  store half %1895, ptr addrspace(1) %address.local1049, align 2, !dbg !18
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1025, !dbg !18
  store half %1894, ptr addrspace(1) %address.local1051, align 2, !dbg !18
  %address.local1052 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1052, i64 %address.local1028, !dbg !18
  store half %1893, ptr addrspace(1) %address.local1053, align 2, !dbg !18
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1055 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1031, !dbg !18
  store half %1892, ptr addrspace(1) %address.local1055, align 2, !dbg !18
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1034, !dbg !18
  store half %1886, ptr addrspace(1) %address.local1057, align 2, !dbg !18
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1058, i64 %address.local1037, !dbg !18
  store half %1885, ptr addrspace(1) %address.local1059, align 2, !dbg !18
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1060, i64 %address.local1040, !dbg !18
  store half %1884, ptr addrspace(1) %address.local1061, align 2, !dbg !18
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1047, !dbg !18
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1062, i64 %address.local1043, !dbg !18
  store half %1883, ptr addrspace(1) %address.local1063, align 2, !dbg !18
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-162611-z5os6mty/grid-000003")
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
