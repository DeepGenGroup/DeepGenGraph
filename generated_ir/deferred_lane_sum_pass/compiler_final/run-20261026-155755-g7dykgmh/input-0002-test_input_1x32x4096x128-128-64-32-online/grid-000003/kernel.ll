; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/deferred_lane_sum_pass/compiler_final/run-20261026-155755-g7dykgmh/input-0002-test_input_1x32x4096x128-128-64-32-online/grid-000003/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/deferred_lane_sum_pass/compiler_final/run-20261026-155755-g7dykgmh/input-0002-test_input_1x32x4096x128-128-64-32-online/grid-000003/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 5, !dbg !5
  %10 = add nsw i64 %9, %6, !dbg !5
  %11 = lshr i64 %10, 5, !dbg !5
  %12 = and i64 %11, 576460752303423484, !dbg !5
  %13 = and i64 %6, 3, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = sub nsw i64 31, %14, !dbg !5
  %16 = lshr i64 %10, 2, !dbg !5
  %17 = icmp ugt i64 %10, 1023, !dbg !9
  %18 = add nsw i64 %14, -32, !dbg !9
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !9
  %20 = sdiv i64 %19, 32, !dbg !9
  %21 = sext i1 %17 to i64, !dbg !9
  %22 = xor i64 %20, %21, !dbg !9
  %23 = add nsw i64 %16, %22, !dbg !9
  %24 = and i64 %23, 31, !dbg !9
  %25 = shl nsw i64 %15, 7, !dbg !9
  %26 = mul nsw i64 %22, -4096, !dbg !9
  %27 = add i64 %25, %26, !dbg !9
  %28 = shl nuw nsw i64 %24, 19, !dbg !9
  %29 = shl nsw i64 %27, 7, !dbg !9
  %30 = add i64 %28, %29, !dbg !9
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !9, !range !39
  %32 = zext nneg i32 %31 to i64, !dbg !9
  %33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !9
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !9, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !9
  %address.local262 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local263 = and i64 %address.local262, 1536, !dbg !9
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local263, !dbg !9
  %address.local265 = ptrtoint ptr addrspace(3) %address.local264 to i64, !dbg !9
  %address.local266 = trunc i64 %address.local265 to i32, !dbg !9
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local266), !dbg !9
  %35 = zext i32 %34 to i64, !dbg !9
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !9
  %address.local267 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local268 = and i64 %address.local267, 16256, !dbg !9
  %address.local269 = lshr i64 %address.local, 4, !dbg !9
  %address.local270 = and i64 %address.local269, 8, !dbg !9
  %address.local271 = or disjoint i64 %address.local268, %address.local270, !dbg !9
  %address.local272 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local271, !dbg !9
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %38 = ptrtoint ptr addrspace(1) %address.local272 to i64, !dbg !9
  %39 = sub i64 %38, %37, !dbg !9
  %dma.buffer.offset1082 = trunc i64 %39 to i32, !dbg !9
  %40 = trunc i64 %37 to i32, !dbg !9
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !9
  %42 = lshr i64 %37, 32, !dbg !9
  %43 = trunc i64 %42 to i32, !dbg !9
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !9
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1083 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !9
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !9
  %dma.buffer.lds1084 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !9
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1084, i32 %dma.buffer.offset1082, <4 x i32> %dma.buffer.rsrc1083) #10, !dbg !9
  %address.local273 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local274 = shl nuw nsw i64 %address.local273, 3, !dbg !9
  %address.local275 = and i64 %address.local274, 3584, !dbg !9
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local275, !dbg !9
  %address.local277 = ptrtoint ptr addrspace(3) %address.local276 to i64, !dbg !9
  %address.local278 = trunc i64 %address.local277 to i32, !dbg !9
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local278), !dbg !9
  %49 = zext i32 %48 to i64, !dbg !9
  %50 = inttoptr i64 %49 to ptr addrspace(3), !dbg !9
  %address.local279 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local280 = and i64 %address.local279, 16256, !dbg !9
  %address.local281 = lshr i64 %address.local273, 4, !dbg !9
  %address.local282 = and i64 %address.local281, 24, !dbg !9
  %address.local283 = or disjoint i64 %address.local280, %address.local282, !dbg !9
  %address.local284 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local283, !dbg !9
  %51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %52 = ptrtoint ptr addrspace(1) %address.local284 to i64, !dbg !9
  %53 = sub i64 %52, %51, !dbg !9
  %dma.buffer.offset1079 = trunc i64 %53 to i32, !dbg !9
  %54 = trunc i64 %51 to i32, !dbg !9
  %55 = insertelement <4 x i32> poison, i32 %54, i64 0, !dbg !9
  %56 = lshr i64 %51, 32, !dbg !9
  %57 = trunc i64 %56 to i32, !dbg !9
  %58 = insertelement <4 x i32> %55, i32 %57, i64 1, !dbg !9
  %59 = insertelement <4 x i32> %58, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1080 = insertelement <4 x i32> %59, i32 131072, i64 3, !dbg !9
  %60 = ptrtoint ptr addrspace(3) %50 to i32, !dbg !9
  %dma.buffer.lds1081 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !9
  %61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1081, i32 %dma.buffer.offset1079, <4 x i32> %dma.buffer.rsrc1080) #10, !dbg !9
  %62 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %63 = sext i32 %62 to i64, !dbg !5
  %64 = shl i64 %15, 7, !dbg !11
  %65 = lshr i64 %32, 6, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %66 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !9
  %67 = getelementptr i8, ptr addrspace(1) %66, i64 64, !dbg !9
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local263, !dbg !9
  %address.local286 = ptrtoint ptr addrspace(3) %address.local285 to i64, !dbg !9
  %address.local287 = trunc i64 %address.local286 to i32, !dbg !9
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local287), !dbg !9
  %69 = zext i32 %68 to i64, !dbg !9
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !9
  %address.local288 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local271, !dbg !9
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %72 = ptrtoint ptr addrspace(1) %address.local288 to i64, !dbg !9
  %73 = sub i64 %72, %71, !dbg !9
  %dma.buffer.offset1076 = trunc i64 %73 to i32, !dbg !9
  %74 = trunc i64 %71 to i32, !dbg !9
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !9
  %76 = lshr i64 %71, 32, !dbg !9
  %77 = trunc i64 %76 to i32, !dbg !9
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !9
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1077 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !9
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !9
  %dma.buffer.lds1078 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !9
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1078, i32 %dma.buffer.offset1076, <4 x i32> %dma.buffer.rsrc1077) #10, !dbg !9
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local275, !dbg !9
  %address.local290 = ptrtoint ptr addrspace(3) %address.local289 to i64, !dbg !9
  %address.local291 = trunc i64 %address.local290 to i32, !dbg !9
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local291), !dbg !9
  %83 = zext i32 %82 to i64, !dbg !9
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !9
  %address.local292 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local283, !dbg !9
  %85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %86 = ptrtoint ptr addrspace(1) %address.local292 to i64, !dbg !9
  %87 = sub i64 %86, %85, !dbg !9
  %dma.buffer.offset1073 = trunc i64 %87 to i32, !dbg !9
  %88 = trunc i64 %85 to i32, !dbg !9
  %89 = insertelement <4 x i32> poison, i32 %88, i64 0, !dbg !9
  %90 = lshr i64 %85, 32, !dbg !9
  %91 = trunc i64 %90 to i32, !dbg !9
  %92 = insertelement <4 x i32> %89, i32 %91, i64 1, !dbg !9
  %93 = insertelement <4 x i32> %92, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1074 = insertelement <4 x i32> %93, i32 131072, i64 3, !dbg !9
  %94 = ptrtoint ptr addrspace(3) %84 to i32, !dbg !9
  %dma.buffer.lds1075 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !9
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1075, i32 %dma.buffer.offset1073, <4 x i32> %dma.buffer.rsrc1074) #10, !dbg !9
  %address.local293 = lshr i32 %address.thread, 2, !dbg !8
  %address.local294 = and i32 %address.local293, 4, !dbg !8
  %address.local295 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local296 = and i32 %address.local295, 384, !dbg !8
  %address.local297 = or disjoint i32 %address.local294, %address.local296, !dbg !8
  %address.local298 = shl nuw nsw i32 %address.thread, 3, !dbg !8
  %address.local299 = and i32 %address.local298, 120, !dbg !8
  %address.local300 = or disjoint i32 %address.local297, %address.local299, !dbg !8
  %address.local301 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local302 = and i32 %address.local301, 1024, !dbg !8
  %address.local303 = or disjoint i32 %address.local300, %address.local302, !dbg !8
  %address.local304 = zext nneg i32 %address.local303 to i64, !dbg !8
  %address.local305 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local304, !dbg !8
  %96 = load <4 x half>, ptr addrspace(3) %address.local305, align 8, !dbg !8
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local305, i64 4096, !dbg !8
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !8
  %98 = fmul <4 x half> %96, splat (half 0xH3015), !dbg !8
  %lds.const.gep971 = getelementptr i8, ptr addrspace(3) %address.local305, i64 1024, !dbg !8
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep971, align 8, !dbg !8
  %100 = fmul <4 x half> %97, splat (half 0xH3015), !dbg !8
  %lds.const.gep972 = getelementptr i8, ptr addrspace(3) %address.local305, i64 5120, !dbg !8
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep972, align 8, !dbg !8
  %102 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !8
  %103 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !9
  %105 = getelementptr i8, ptr addrspace(1) %104, i64 128, !dbg !9
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local266), !dbg !9
  %107 = zext i32 %106 to i64, !dbg !9
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !9
  %address.local348 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local271, !dbg !9
  %109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %110 = ptrtoint ptr addrspace(1) %address.local348 to i64, !dbg !9
  %111 = sub i64 %110, %109, !dbg !9
  %dma.buffer.offset1070 = trunc i64 %111 to i32, !dbg !9
  %112 = trunc i64 %109 to i32, !dbg !9
  %113 = insertelement <4 x i32> poison, i32 %112, i64 0, !dbg !9
  %114 = lshr i64 %109, 32, !dbg !9
  %115 = trunc i64 %114 to i32, !dbg !9
  %116 = insertelement <4 x i32> %113, i32 %115, i64 1, !dbg !9
  %117 = insertelement <4 x i32> %116, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1071 = insertelement <4 x i32> %117, i32 131072, i64 3, !dbg !9
  %118 = ptrtoint ptr addrspace(3) %108 to i32, !dbg !9
  %dma.buffer.lds1072 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !9
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1072, i32 %dma.buffer.offset1070, <4 x i32> %dma.buffer.rsrc1071) #10, !dbg !9
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local278), !dbg !9
  %121 = zext i32 %120 to i64, !dbg !9
  %122 = inttoptr i64 %121 to ptr addrspace(3), !dbg !9
  %address.local349 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local283, !dbg !9
  %123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %124 = ptrtoint ptr addrspace(1) %address.local349 to i64, !dbg !9
  %125 = sub i64 %124, %123, !dbg !9
  %dma.buffer.offset1067 = trunc i64 %125 to i32, !dbg !9
  %126 = trunc i64 %123 to i32, !dbg !9
  %127 = insertelement <4 x i32> poison, i32 %126, i64 0, !dbg !9
  %128 = lshr i64 %123, 32, !dbg !9
  %129 = trunc i64 %128 to i32, !dbg !9
  %130 = insertelement <4 x i32> %127, i32 %129, i64 1, !dbg !9
  %131 = insertelement <4 x i32> %130, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1068 = insertelement <4 x i32> %131, i32 131072, i64 3, !dbg !9
  %132 = ptrtoint ptr addrspace(3) %122 to i32, !dbg !9
  %dma.buffer.lds1069 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !9
  %133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1069, i32 %dma.buffer.offset1067, <4 x i32> %dma.buffer.rsrc1068) #10, !dbg !9
  %lds.const.gep973 = getelementptr i8, ptr addrspace(3) %address.local305, i64 8192, !dbg !8
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep973, align 8, !dbg !8
  %lds.const.gep974 = getelementptr i8, ptr addrspace(3) %address.local305, i64 12288, !dbg !8
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep974, align 8, !dbg !8
  %lds.const.gep975 = getelementptr i8, ptr addrspace(3) %address.local305, i64 9216, !dbg !8
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep975, align 8, !dbg !8
  %137 = fmul <4 x half> %134, splat (half 0xH3015), !dbg !8
  %lds.const.gep976 = getelementptr i8, ptr addrspace(3) %address.local305, i64 13312, !dbg !8
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep976, align 8, !dbg !8
  %139 = fmul <4 x half> %135, splat (half 0xH3015), !dbg !8
  %140 = fmul <4 x half> %136, splat (half 0xH3015), !dbg !8
  %141 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !9
  %143 = getelementptr i8, ptr addrspace(1) %142, i64 192, !dbg !9
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local287), !dbg !9
  %145 = zext i32 %144 to i64, !dbg !9
  %146 = inttoptr i64 %145 to ptr addrspace(3), !dbg !9
  %address.local354 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local271, !dbg !9
  %147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %148 = ptrtoint ptr addrspace(1) %address.local354 to i64, !dbg !9
  %149 = sub i64 %148, %147, !dbg !9
  %dma.buffer.offset1064 = trunc i64 %149 to i32, !dbg !9
  %150 = trunc i64 %147 to i32, !dbg !9
  %151 = insertelement <4 x i32> poison, i32 %150, i64 0, !dbg !9
  %152 = lshr i64 %147, 32, !dbg !9
  %153 = trunc i64 %152 to i32, !dbg !9
  %154 = insertelement <4 x i32> %151, i32 %153, i64 1, !dbg !9
  %155 = insertelement <4 x i32> %154, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1065 = insertelement <4 x i32> %155, i32 131072, i64 3, !dbg !9
  %156 = ptrtoint ptr addrspace(3) %146 to i32, !dbg !9
  %dma.buffer.lds1066 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %156), !dbg !9
  %157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1066, i32 %dma.buffer.offset1064, <4 x i32> %dma.buffer.rsrc1065) #10, !dbg !9
  %158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local291), !dbg !9
  %159 = zext i32 %158 to i64, !dbg !9
  %160 = inttoptr i64 %159 to ptr addrspace(3), !dbg !9
  %address.local355 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local283, !dbg !9
  %161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %162 = ptrtoint ptr addrspace(1) %address.local355 to i64, !dbg !9
  %163 = sub i64 %162, %161, !dbg !9
  %dma.buffer.offset1061 = trunc i64 %163 to i32, !dbg !9
  %164 = trunc i64 %161 to i32, !dbg !9
  %165 = insertelement <4 x i32> poison, i32 %164, i64 0, !dbg !9
  %166 = lshr i64 %161, 32, !dbg !9
  %167 = trunc i64 %166 to i32, !dbg !9
  %168 = insertelement <4 x i32> %165, i32 %167, i64 1, !dbg !9
  %169 = insertelement <4 x i32> %168, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1062 = insertelement <4 x i32> %169, i32 131072, i64 3, !dbg !9
  %170 = ptrtoint ptr addrspace(3) %160 to i32, !dbg !9
  %dma.buffer.lds1063 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !9
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1063, i32 %dma.buffer.offset1061, <4 x i32> %dma.buffer.rsrc1062) #10, !dbg !9
  %172 = load <4 x half>, ptr addrspace(3) %address.local305, align 8, !dbg !8
  %lds.const.gep977 = getelementptr i8, ptr addrspace(3) %address.local305, i64 4096, !dbg !8
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep977, align 8, !dbg !8
  %lds.const.gep978 = getelementptr i8, ptr addrspace(3) %address.local305, i64 1024, !dbg !8
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep978, align 8, !dbg !8
  %175 = fmul <4 x half> %172, splat (half 0xH3015), !dbg !8
  %lds.const.gep979 = getelementptr i8, ptr addrspace(3) %address.local305, i64 5120, !dbg !8
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep979, align 8, !dbg !8
  %177 = fmul <4 x half> %173, splat (half 0xH3015), !dbg !8
  %178 = fmul <4 x half> %174, splat (half 0xH3015), !dbg !8
  %179 = fmul <4 x half> %176, splat (half 0xH3015), !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %lds.const.gep980 = getelementptr i8, ptr addrspace(3) %address.local305, i64 8192, !dbg !8
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep980, align 8, !dbg !8
  %lds.const.gep981 = getelementptr i8, ptr addrspace(3) %address.local305, i64 12288, !dbg !8
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep981, align 8, !dbg !8
  %lds.const.gep982 = getelementptr i8, ptr addrspace(3) %address.local305, i64 9216, !dbg !8
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep982, align 8, !dbg !8
  %183 = fmul <4 x half> %180, splat (half 0xH3015), !dbg !8
  %lds.const.gep983 = getelementptr i8, ptr addrspace(3) %address.local305, i64 13312, !dbg !8
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep983, align 8, !dbg !8
  %185 = fmul <4 x half> %181, splat (half 0xH3015), !dbg !8
  %186 = fmul <4 x half> %182, splat (half 0xH3015), !dbg !8
  %187 = fmul <4 x half> %184, splat (half 0xH3015), !dbg !8
  %188 = or disjoint i64 %64, 127, !dbg !14
  %189 = add i64 %64, 128, !dbg !14
  %190 = icmp uge i64 %189, %188, !dbg !14
  %191 = shl nsw i64 %15, 1, !dbg !7
  %192 = add nsw i64 %191, 2, !dbg !7
  %193 = shl i64 %16, 7, !dbg !16
  %194 = and i64 %193, 3968, !dbg !16
  %195 = shl nsw i64 %63, 12, !dbg !16
  %196 = or disjoint i64 %194, %195, !dbg !16
  %197 = lshr i64 %32, 2, !dbg !17
  %198 = shl nuw nsw i64 %32, 2, !dbg !17
  %199 = mul nsw i64 %197, -16, !dbg !17
  %200 = add nsw i64 %198, %199, !dbg !17
  %201 = and i32 %31, 1, !dbg !17
  %index.compact228 = zext nneg i32 %201 to i64, !dbg !17
  %202 = or disjoint i64 %200, %index.compact228, !dbg !17
  %203 = add nuw nsw i64 %202, 1, !dbg !17
  %204 = sdiv i64 %203, 2, !dbg !17
  br label %205, !dbg !7

205:                                              ; preds = %584, %4
  %206 = phi i64 [ %1031, %584 ], [ 0, %4 ], !dbg !7
  %207 = phi [2 x <8 x float>] [ %1016, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %208 = phi [2 x <1 x float>] [ %792, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %209 = phi [2 x <1 x float>] [ %1018, %584 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %210 = phi [2 x <8 x float>] [ %1022, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %211 = phi [2 x <8 x float>] [ %1026, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %212 = phi [2 x <8 x float>] [ %1030, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %213 = icmp slt i64 %206, %192, !dbg !7
  br i1 %213, label %214, label %1032, !dbg !7

214:                                              ; preds = %205
  %215 = shl nsw i64 %206, 6, !dbg !7
  %216 = lshr i64 %206, 13, !dbg !16
  %217 = add nuw nsw i64 %216, %16, !dbg !16
  %218 = lshr i64 %206, 6, !dbg !16
  %219 = and i64 %218, 127, !dbg !16
  %220 = shl i64 %217, 19, !dbg !16
  %221 = and i64 %220, 16252928, !dbg !16
  %222 = shl i64 %218, 19, !dbg !16
  %223 = shl i64 %206, 13, !dbg !16
  %224 = sub i64 %223, %222, !dbg !16
  %225 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %221, !dbg !16
  %226 = getelementptr [2 x i8], ptr addrspace(1) %225, i64 %219, !dbg !16
  %227 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %224, !dbg !16
  %address.thread356 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local357 = zext nneg i32 %address.thread356 to i64, !dbg !9
  %address.local358 = shl nuw nsw i64 %address.local357, 3, !dbg !9
  %address.local359 = and i64 %address.local358, 1536, !dbg !16
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local359, !dbg !16
  %address.local361 = ptrtoint ptr addrspace(3) %address.local360 to i64, !dbg !16
  %address.local362 = trunc i64 %address.local361 to i32, !dbg !16
  %228 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local362), !dbg !16
  %229 = zext i32 %228 to i64, !dbg !16
  %230 = inttoptr i64 %229 to ptr addrspace(3), !dbg !16
  %address.local363 = lshr i64 %address.local357, 3, !dbg !16
  %address.local364 = and i64 %address.local363, 24, !dbg !16
  %address.local365 = and i64 %address.local357, 59, !dbg !16
  %address.local366 = lshr i64 %address.local357, 2, !dbg !16
  %address.local367 = and i64 %address.local366, 4, !dbg !16
  %address.local368 = add nuw nsw i64 %address.local367, %address.local357, !dbg !16
  %address.local369 = and i64 %address.local368, 4, !dbg !16
  %address.local370 = or disjoint i64 %address.local365, %address.local369, !dbg !16
  %address.local371 = shl nuw nsw i64 %address.local370, 7, !dbg !16
  %address.local372 = or disjoint i64 %address.local364, %address.local371, !dbg !16
  %address.local373 = getelementptr [2 x i8], ptr addrspace(1) %227, i64 %address.local372, !dbg !16
  %231 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %232 = ptrtoint ptr addrspace(1) %address.local373 to i64, !dbg !16
  %233 = sub i64 %232, %231, !dbg !16
  %dma.buffer.offset1058 = trunc i64 %233 to i32, !dbg !16
  %234 = trunc i64 %231 to i32, !dbg !16
  %235 = insertelement <4 x i32> poison, i32 %234, i64 0, !dbg !16
  %236 = lshr i64 %231, 32, !dbg !16
  %237 = trunc i64 %236 to i32, !dbg !16
  %238 = insertelement <4 x i32> %235, i32 %237, i64 1, !dbg !16
  %239 = insertelement <4 x i32> %238, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1059 = insertelement <4 x i32> %239, i32 131072, i64 3, !dbg !16
  %240 = ptrtoint ptr addrspace(3) %230 to i32, !dbg !16
  %dma.buffer.lds1060 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %240), !dbg !16
  %241 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1060, i32 %dma.buffer.offset1058, <4 x i32> %dma.buffer.rsrc1059) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %242 = add nuw i64 %206, 2048, !dbg !16
  %243 = icmp slt i64 %242, 0, !dbg !16
  %244 = sub nuw i64 -2049, %206, !dbg !16
  %245 = select i1 %243, i64 %244, i64 %242, !dbg !16
  %246 = sdiv i64 %245, 8192, !dbg !16
  %.lobit19 = ashr i64 %242, 63, !dbg !16
  %247 = xor i64 %246, %.lobit19, !dbg !16
  %248 = add nsw i64 %247, %16, !dbg !16
  %249 = add nsw i64 %218, %196, !dbg !16
  %250 = add nsw i64 %249, 32, !dbg !16
  %251 = icmp slt i64 %249, -32, !dbg !16
  %252 = sub nsw i64 -33, %249, !dbg !16
  %253 = select i1 %251, i64 %252, i64 %250, !dbg !16
  %254 = sdiv i64 %253, 128, !dbg !16
  %255 = sext i1 %251 to i64, !dbg !16
  %256 = xor i64 %254, %255, !dbg !16
  %257 = shl nsw i64 %256, 7, !dbg !16
  %258 = sub i64 %196, %257, !dbg !16
  %259 = shl i64 %248, 19, !dbg !16
  %260 = and i64 %259, 16252928, !dbg !16
  %261 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %260, !dbg !16
  %262 = getelementptr [2 x i8], ptr addrspace(1) %261, i64 %218, !dbg !16
  %263 = getelementptr [2 x i8], ptr addrspace(1) %262, i64 %258, !dbg !16
  %264 = getelementptr i8, ptr addrspace(1) %263, i64 64, !dbg !16
  %265 = getelementptr [2 x i8], ptr addrspace(1) %264, i64 %224, !dbg !16
  %address.local374 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local359, !dbg !16
  %address.local375 = ptrtoint ptr addrspace(3) %address.local374 to i64, !dbg !16
  %address.local376 = trunc i64 %address.local375 to i32, !dbg !16
  %266 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local376), !dbg !16
  %267 = zext i32 %266 to i64, !dbg !16
  %268 = inttoptr i64 %267 to ptr addrspace(3), !dbg !16
  %address.local377 = getelementptr [2 x i8], ptr addrspace(1) %265, i64 %address.local372, !dbg !16
  %269 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %270 = ptrtoint ptr addrspace(1) %address.local377 to i64, !dbg !16
  %271 = sub i64 %270, %269, !dbg !16
  %dma.buffer.offset1055 = trunc i64 %271 to i32, !dbg !16
  %272 = trunc i64 %269 to i32, !dbg !16
  %273 = insertelement <4 x i32> poison, i32 %272, i64 0, !dbg !16
  %274 = lshr i64 %269, 32, !dbg !16
  %275 = trunc i64 %274 to i32, !dbg !16
  %276 = insertelement <4 x i32> %273, i32 %275, i64 1, !dbg !16
  %277 = insertelement <4 x i32> %276, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1056 = insertelement <4 x i32> %277, i32 131072, i64 3, !dbg !16
  %278 = ptrtoint ptr addrspace(3) %268 to i32, !dbg !16
  %dma.buffer.lds1057 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %278), !dbg !16
  %279 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1057, i32 %dma.buffer.offset1055, <4 x i32> %dma.buffer.rsrc1056) #10, !dbg !16
  %address.local378 = mul nuw nsw i64 %address.local357, 160, !dbg !17
  %address.local379 = lshr i64 %address.local357, 2, !dbg !17
  %address.local380 = mul nsw i64 %address.local379, -640, !dbg !17
  %address.local381 = add nsw i64 %address.local378, %address.local380, !dbg !17
  %address.local382 = shl nuw nsw i64 %address.local357, 1, !dbg !17
  %address.local383 = and i64 %address.local382, 24, !dbg !17
  %address.local384 = or disjoint i64 %address.local381, %address.local383, !dbg !17
  %address.local385 = shl i64 %address.local357, 62, !dbg !17
  %address.local386 = ashr exact i64 %address.local385, 57, !dbg !17
  %address.local387 = and i64 %address.local386, -64, !dbg !17
  %address.local388 = add nsw i64 %address.local384, %address.local387, !dbg !17
  %address.local389 = lshr i64 %address.local357, 2, !dbg !8
  %address.local390 = and i64 %address.local389, 12, !dbg !8
  %address.local391 = add nsw i64 %address.local388, %address.local390, !dbg !17
  %address.local392 = and i64 %address.local357, 63, !dbg !8
  %address.local393 = lshr i64 %address.local392, 5, !dbg !8
  %address.local394 = mul nuw nsw i64 %address.local393, 504, !dbg !17
  %address.local395 = add i64 %address.local391, %address.local394, !dbg !17
  %address.local396 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local395, !dbg !17
  %280 = load <4 x half>, ptr addrspace(3) %address.local396, align 8, !dbg !17
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %280, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep984 = getelementptr i8, ptr addrspace(3) %address.local396, i64 2048, !dbg !17
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep984, align 8, !dbg !17
  %address.local407 = shl nuw nsw i32 %address.thread356, 1, !dbg !17
  %address.local408 = and i32 %address.local407, 24, !dbg !17
  %address.local409 = shl nuw nsw i32 %address.thread356, 8, !dbg !17
  %address.local410 = and i32 %address.local409, 768, !dbg !17
  %address.local411 = or disjoint i32 %address.local408, %address.local410, !dbg !17
  %address.local412 = shl nuw nsw i32 %address.thread356, 5, !dbg !17
  %address.local413 = and i32 %address.local412, 96, !dbg !17
  %address.local414 = or disjoint i32 %address.local411, %address.local413, !dbg !17
  %address.local415 = zext nneg i32 %address.local414 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %204) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local416 = lshr i32 %address.thread356, 1, !dbg !17
  %address.local417 = and i32 %address.local416, 1, !dbg !17
  %address.local418 = zext nneg i32 %address.local417 to i64, !dbg !17
  %address.local419 = add nsw i64 %address.base, %address.local418, !dbg !17
  %address.local420 = shl i64 %address.local419, 6, !dbg !17
  %address.local421 = sub i64 %address.local415, %address.local420, !dbg !17
  %address.local422 = add i64 %address.local421, %address.local390, !dbg !17
  %address.local423 = add i64 %address.local422, %address.local394, !dbg !17
  %address.local424 = add i64 %address.local423, 32, !dbg !17
  %address.local425 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local424, !dbg !17
  %283 = load <4 x half>, ptr addrspace(3) %address.local425, align 8, !dbg !17
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %282, <4 x float> %281) #9, !dbg !17
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %283, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep985 = getelementptr i8, ptr addrspace(3) %address.local425, i64 2048, !dbg !17
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep985, align 8, !dbg !17
  %lds.const.gep986 = getelementptr i8, ptr addrspace(3) %address.local396, i64 128, !dbg !17
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep986, align 8, !dbg !17
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %286, <4 x float> %285) #9, !dbg !17
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %287, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep987 = getelementptr i8, ptr addrspace(3) %address.local396, i64 2176, !dbg !17
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep987, align 8, !dbg !17
  %lds.const.gep988 = getelementptr i8, ptr addrspace(3) %address.local425, i64 128, !dbg !17
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep988, align 8, !dbg !17
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %290, <4 x float> %289) #9, !dbg !17
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %291, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep989 = getelementptr i8, ptr addrspace(3) %address.local425, i64 2176, !dbg !17
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep989, align 8, !dbg !17
  %295 = load <4 x half>, ptr addrspace(3) %address.local396, align 8, !dbg !17
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %294, <4 x float> %293) #9, !dbg !17
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %295, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep990 = getelementptr i8, ptr addrspace(3) %address.local396, i64 2048, !dbg !17
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep990, align 8, !dbg !17
  %299 = load <4 x half>, ptr addrspace(3) %address.local425, align 8, !dbg !17
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %298, <4 x float> %297) #9, !dbg !17
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %299, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep991 = getelementptr i8, ptr addrspace(3) %address.local425, i64 2048, !dbg !17
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep991, align 8, !dbg !17
  %lds.const.gep992 = getelementptr i8, ptr addrspace(3) %address.local396, i64 128, !dbg !17
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep992, align 8, !dbg !17
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %302, <4 x float> %301) #9, !dbg !17
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %303, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep993 = getelementptr i8, ptr addrspace(3) %address.local396, i64 2176, !dbg !17
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep993, align 8, !dbg !17
  %lds.const.gep994 = getelementptr i8, ptr addrspace(3) %address.local425, i64 128, !dbg !17
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep994, align 8, !dbg !17
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %306, <4 x float> %305) #9, !dbg !17
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %307, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep995 = getelementptr i8, ptr addrspace(3) %address.local425, i64 2176, !dbg !17
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep995, align 8, !dbg !17
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %310, <4 x float> %309) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %312 = add nuw i64 %206, 4096, !dbg !16
  %313 = icmp slt i64 %312, 0, !dbg !16
  %314 = sub nuw i64 -4097, %206, !dbg !16
  %315 = select i1 %313, i64 %314, i64 %312, !dbg !16
  %316 = sdiv i64 %315, 8192, !dbg !16
  %.lobit20 = ashr i64 %312, 63, !dbg !16
  %317 = xor i64 %316, %.lobit20, !dbg !16
  %318 = add nsw i64 %317, %16, !dbg !16
  %319 = add nsw i64 %249, 64, !dbg !16
  %320 = icmp slt i64 %249, -64, !dbg !16
  %321 = sub nsw i64 -65, %249, !dbg !16
  %322 = select i1 %320, i64 %321, i64 %319, !dbg !16
  %323 = sdiv i64 %322, 128, !dbg !16
  %324 = sext i1 %320 to i64, !dbg !16
  %325 = xor i64 %323, %324, !dbg !16
  %326 = shl nsw i64 %325, 7, !dbg !16
  %327 = sub i64 %196, %326, !dbg !16
  %328 = shl i64 %318, 19, !dbg !16
  %329 = and i64 %328, 16252928, !dbg !16
  %330 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %329, !dbg !16
  %331 = getelementptr [2 x i8], ptr addrspace(1) %330, i64 %218, !dbg !16
  %332 = getelementptr [2 x i8], ptr addrspace(1) %331, i64 %327, !dbg !16
  %333 = getelementptr i8, ptr addrspace(1) %332, i64 128, !dbg !16
  %334 = getelementptr [2 x i8], ptr addrspace(1) %333, i64 %224, !dbg !16
  %335 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local362), !dbg !16
  %336 = zext i32 %335 to i64, !dbg !16
  %337 = inttoptr i64 %336 to ptr addrspace(3), !dbg !16
  %address.local566 = getelementptr [2 x i8], ptr addrspace(1) %334, i64 %address.local372, !dbg !16
  %338 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %339 = ptrtoint ptr addrspace(1) %address.local566 to i64, !dbg !16
  %340 = sub i64 %339, %338, !dbg !16
  %dma.buffer.offset1052 = trunc i64 %340 to i32, !dbg !16
  %341 = trunc i64 %338 to i32, !dbg !16
  %342 = insertelement <4 x i32> poison, i32 %341, i64 0, !dbg !16
  %343 = lshr i64 %338, 32, !dbg !16
  %344 = trunc i64 %343 to i32, !dbg !16
  %345 = insertelement <4 x i32> %342, i32 %344, i64 1, !dbg !16
  %346 = insertelement <4 x i32> %345, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1053 = insertelement <4 x i32> %346, i32 131072, i64 3, !dbg !16
  %347 = ptrtoint ptr addrspace(3) %337 to i32, !dbg !16
  %dma.buffer.lds1054 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %347), !dbg !16
  %348 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1054, i32 %dma.buffer.offset1052, <4 x i32> %dma.buffer.rsrc1053) #10, !dbg !16
  %lds.const.gep996 = getelementptr i8, ptr addrspace(3) %address.local396, i64 4096, !dbg !17
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep996, align 8, !dbg !17
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %349, <4 x float> %284) #9, !dbg !17
  %lds.const.gep997 = getelementptr i8, ptr addrspace(3) %address.local396, i64 6144, !dbg !17
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep997, align 8, !dbg !17
  %lds.const.gep998 = getelementptr i8, ptr addrspace(3) %address.local425, i64 4096, !dbg !17
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep998, align 8, !dbg !17
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %351, <4 x float> %350) #9, !dbg !17
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %352, <4 x float> %288) #9, !dbg !17
  %lds.const.gep999 = getelementptr i8, ptr addrspace(3) %address.local425, i64 6144, !dbg !17
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep999, align 8, !dbg !17
  %lds.const.gep1000 = getelementptr i8, ptr addrspace(3) %address.local396, i64 4224, !dbg !17
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1000, align 8, !dbg !17
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %355, <4 x float> %354) #9, !dbg !17
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %356, <4 x float> %292) #9, !dbg !17
  %lds.const.gep1001 = getelementptr i8, ptr addrspace(3) %address.local396, i64 6272, !dbg !17
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1001, align 8, !dbg !17
  %lds.const.gep1002 = getelementptr i8, ptr addrspace(3) %address.local425, i64 4224, !dbg !17
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1002, align 8, !dbg !17
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %359, <4 x float> %358) #9, !dbg !17
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %360, <4 x float> %296) #9, !dbg !17
  %lds.const.gep1003 = getelementptr i8, ptr addrspace(3) %address.local425, i64 6272, !dbg !17
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep1003, align 8, !dbg !17
  %lds.const.gep1004 = getelementptr i8, ptr addrspace(3) %address.local396, i64 4096, !dbg !17
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep1004, align 8, !dbg !17
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %363, <4 x float> %362) #9, !dbg !17
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %364, <4 x float> %300) #9, !dbg !17
  %lds.const.gep1005 = getelementptr i8, ptr addrspace(3) %address.local396, i64 6144, !dbg !17
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1005, align 8, !dbg !17
  %lds.const.gep1006 = getelementptr i8, ptr addrspace(3) %address.local425, i64 4096, !dbg !17
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep1006, align 8, !dbg !17
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %367, <4 x float> %366) #9, !dbg !17
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %368, <4 x float> %304) #9, !dbg !17
  %lds.const.gep1007 = getelementptr i8, ptr addrspace(3) %address.local425, i64 6144, !dbg !17
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep1007, align 8, !dbg !17
  %lds.const.gep1008 = getelementptr i8, ptr addrspace(3) %address.local396, i64 4224, !dbg !17
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep1008, align 8, !dbg !17
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %371, <4 x float> %370) #9, !dbg !17
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %372, <4 x float> %308) #9, !dbg !17
  %lds.const.gep1009 = getelementptr i8, ptr addrspace(3) %address.local396, i64 6272, !dbg !17
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep1009, align 8, !dbg !17
  %lds.const.gep1010 = getelementptr i8, ptr addrspace(3) %address.local425, i64 4224, !dbg !17
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep1010, align 8, !dbg !17
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %375, <4 x float> %374) #9, !dbg !17
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %376, <4 x float> %311) #9, !dbg !17
  %lds.const.gep1011 = getelementptr i8, ptr addrspace(3) %address.local425, i64 6272, !dbg !17
  %379 = load <4 x half>, ptr addrspace(3) %lds.const.gep1011, align 8, !dbg !17
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %379, <4 x float> %378) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %381 = add nuw i64 %206, 6144, !dbg !16
  %382 = icmp slt i64 %381, 0, !dbg !16
  %383 = sub nuw i64 -6145, %206, !dbg !16
  %384 = select i1 %382, i64 %383, i64 %381, !dbg !16
  %385 = sdiv i64 %384, 8192, !dbg !16
  %.lobit21 = ashr i64 %381, 63, !dbg !16
  %386 = xor i64 %385, %.lobit21, !dbg !16
  %387 = add nsw i64 %386, %16, !dbg !16
  %388 = add nsw i64 %249, 96, !dbg !16
  %389 = icmp slt i64 %249, -96, !dbg !16
  %390 = sub nsw i64 -97, %249, !dbg !16
  %391 = select i1 %389, i64 %390, i64 %388, !dbg !16
  %392 = sdiv i64 %391, 128, !dbg !16
  %393 = sext i1 %389 to i64, !dbg !16
  %394 = xor i64 %392, %393, !dbg !16
  %395 = shl nsw i64 %394, 7, !dbg !16
  %396 = sub i64 %196, %395, !dbg !16
  %397 = shl i64 %387, 19, !dbg !16
  %398 = and i64 %397, 16252928, !dbg !16
  %399 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %398, !dbg !16
  %400 = getelementptr [2 x i8], ptr addrspace(1) %399, i64 %218, !dbg !16
  %401 = getelementptr [2 x i8], ptr addrspace(1) %400, i64 %396, !dbg !16
  %402 = getelementptr i8, ptr addrspace(1) %401, i64 192, !dbg !16
  %403 = getelementptr [2 x i8], ptr addrspace(1) %402, i64 %224, !dbg !16
  %404 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local376), !dbg !16
  %405 = zext i32 %404 to i64, !dbg !16
  %406 = inttoptr i64 %405 to ptr addrspace(3), !dbg !16
  %address.local575 = getelementptr [2 x i8], ptr addrspace(1) %403, i64 %address.local372, !dbg !16
  %407 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %408 = ptrtoint ptr addrspace(1) %address.local575 to i64, !dbg !16
  %409 = sub i64 %408, %407, !dbg !16
  %dma.buffer.offset1049 = trunc i64 %409 to i32, !dbg !16
  %410 = trunc i64 %407 to i32, !dbg !16
  %411 = insertelement <4 x i32> poison, i32 %410, i64 0, !dbg !16
  %412 = lshr i64 %407, 32, !dbg !16
  %413 = trunc i64 %412 to i32, !dbg !16
  %414 = insertelement <4 x i32> %411, i32 %413, i64 1, !dbg !16
  %415 = insertelement <4 x i32> %414, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1050 = insertelement <4 x i32> %415, i32 131072, i64 3, !dbg !16
  %416 = ptrtoint ptr addrspace(3) %406 to i32, !dbg !16
  %dma.buffer.lds1051 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %416), !dbg !16
  %417 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1051, i32 %dma.buffer.offset1049, <4 x i32> %dma.buffer.rsrc1050) #10, !dbg !16
  %418 = load <4 x half>, ptr addrspace(3) %address.local396, align 8, !dbg !17
  %419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %418, <4 x float> %353) #9, !dbg !17
  %lds.const.gep1012 = getelementptr i8, ptr addrspace(3) %address.local396, i64 2048, !dbg !17
  %420 = load <4 x half>, ptr addrspace(3) %lds.const.gep1012, align 8, !dbg !17
  %421 = load <4 x half>, ptr addrspace(3) %address.local425, align 8, !dbg !17
  %422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %420, <4 x float> %419) #9, !dbg !17
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %421, <4 x float> %357) #9, !dbg !17
  %lds.const.gep1013 = getelementptr i8, ptr addrspace(3) %address.local425, i64 2048, !dbg !17
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1013, align 8, !dbg !17
  %lds.const.gep1014 = getelementptr i8, ptr addrspace(3) %address.local396, i64 128, !dbg !17
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1014, align 8, !dbg !17
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %424, <4 x float> %423) #9, !dbg !17
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %425, <4 x float> %361) #9, !dbg !17
  %lds.const.gep1015 = getelementptr i8, ptr addrspace(3) %address.local396, i64 2176, !dbg !17
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1015, align 8, !dbg !17
  %lds.const.gep1016 = getelementptr i8, ptr addrspace(3) %address.local425, i64 128, !dbg !17
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1016, align 8, !dbg !17
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %428, <4 x float> %427) #9, !dbg !17
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %429, <4 x float> %365) #9, !dbg !17
  %lds.const.gep1017 = getelementptr i8, ptr addrspace(3) %address.local425, i64 2176, !dbg !17
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1017, align 8, !dbg !17
  %433 = load <4 x half>, ptr addrspace(3) %address.local396, align 8, !dbg !17
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %432, <4 x float> %431) #9, !dbg !17
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %433, <4 x float> %369) #9, !dbg !17
  %lds.const.gep1018 = getelementptr i8, ptr addrspace(3) %address.local396, i64 2048, !dbg !17
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1018, align 8, !dbg !17
  %437 = load <4 x half>, ptr addrspace(3) %address.local425, align 8, !dbg !17
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %436, <4 x float> %435) #9, !dbg !17
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %437, <4 x float> %373) #9, !dbg !17
  %lds.const.gep1019 = getelementptr i8, ptr addrspace(3) %address.local425, i64 2048, !dbg !17
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1019, align 8, !dbg !17
  %lds.const.gep1020 = getelementptr i8, ptr addrspace(3) %address.local396, i64 128, !dbg !17
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1020, align 8, !dbg !17
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %440, <4 x float> %439) #9, !dbg !17
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %441, <4 x float> %377) #9, !dbg !17
  %lds.const.gep1021 = getelementptr i8, ptr addrspace(3) %address.local396, i64 2176, !dbg !17
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1021, align 8, !dbg !17
  %lds.const.gep1022 = getelementptr i8, ptr addrspace(3) %address.local425, i64 128, !dbg !17
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1022, align 8, !dbg !17
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %444, <4 x float> %443) #9, !dbg !17
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %445, <4 x float> %380) #9, !dbg !17
  %lds.const.gep1023 = getelementptr i8, ptr addrspace(3) %address.local425, i64 2176, !dbg !17
  %448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1023, align 8, !dbg !17
  %449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %448, <4 x float> %447) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %450 = add nuw nsw i64 %218, %16, !dbg !18
  %451 = shl i64 %450, 19, !dbg !18
  %452 = and i64 %451, 16252928, !dbg !18
  %453 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %452, !dbg !18
  %454 = getelementptr [2 x i8], ptr addrspace(1) %453, i64 %224, !dbg !18
  %455 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local362), !dbg !18
  %456 = zext i32 %455 to i64, !dbg !18
  %457 = inttoptr i64 %456 to ptr addrspace(3), !dbg !18
  %address.local576 = shl nuw nsw i64 %address.local357, 5, !dbg !18
  %address.local577 = and i64 %address.local576, 8064, !dbg !18
  %address.local578 = shl nuw nsw i64 %address.local357, 3, !dbg !18
  %address.local579 = and i64 %address.local578, 24, !dbg !18
  %address.local580 = or disjoint i64 %address.local577, %address.local579, !dbg !18
  %address.local581 = getelementptr [2 x i8], ptr addrspace(1) %454, i64 %address.local580, !dbg !18
  %458 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %459 = ptrtoint ptr addrspace(1) %address.local581 to i64, !dbg !18
  %460 = sub i64 %459, %458, !dbg !18
  %dma.buffer.offset1046 = trunc i64 %460 to i32, !dbg !18
  %461 = trunc i64 %458 to i32, !dbg !18
  %462 = insertelement <4 x i32> poison, i32 %461, i64 0, !dbg !18
  %463 = lshr i64 %458, 32, !dbg !18
  %464 = trunc i64 %463 to i32, !dbg !18
  %465 = insertelement <4 x i32> %462, i32 %464, i64 1, !dbg !18
  %466 = insertelement <4 x i32> %465, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1047 = insertelement <4 x i32> %466, i32 131072, i64 3, !dbg !18
  %467 = ptrtoint ptr addrspace(3) %457 to i32, !dbg !18
  %dma.buffer.lds1048 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %467), !dbg !18
  %468 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1048, i32 %dma.buffer.offset1046, <4 x i32> %dma.buffer.rsrc1047) #10, !dbg !18
  %lds.const.gep1024 = getelementptr i8, ptr addrspace(3) %address.local396, i64 4096, !dbg !17
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1024, align 8, !dbg !17
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %469, <4 x float> %422) #9, !dbg !17
  %lds.const.gep1025 = getelementptr i8, ptr addrspace(3) %address.local396, i64 6144, !dbg !17
  %471 = load <4 x half>, ptr addrspace(3) %lds.const.gep1025, align 8, !dbg !17
  %lds.const.gep1026 = getelementptr i8, ptr addrspace(3) %address.local425, i64 4096, !dbg !17
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1026, align 8, !dbg !17
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %471, <4 x float> %470) #9, !dbg !17
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %472, <4 x float> %426) #9, !dbg !17
  %lds.const.gep1027 = getelementptr i8, ptr addrspace(3) %address.local425, i64 6144, !dbg !17
  %475 = load <4 x half>, ptr addrspace(3) %lds.const.gep1027, align 8, !dbg !17
  %lds.const.gep1028 = getelementptr i8, ptr addrspace(3) %address.local396, i64 4224, !dbg !17
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1028, align 8, !dbg !17
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %475, <4 x float> %474) #9, !dbg !17
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %476, <4 x float> %430) #9, !dbg !17
  %lds.const.gep1029 = getelementptr i8, ptr addrspace(3) %address.local396, i64 6272, !dbg !17
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep1029, align 8, !dbg !17
  %lds.const.gep1030 = getelementptr i8, ptr addrspace(3) %address.local425, i64 4224, !dbg !17
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1030, align 8, !dbg !17
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %479, <4 x float> %478) #9, !dbg !17
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %480, <4 x float> %434) #9, !dbg !17
  %lds.const.gep1031 = getelementptr i8, ptr addrspace(3) %address.local425, i64 6272, !dbg !17
  %483 = load <4 x half>, ptr addrspace(3) %lds.const.gep1031, align 8, !dbg !17
  %lds.const.gep1032 = getelementptr i8, ptr addrspace(3) %address.local396, i64 4096, !dbg !17
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1032, align 8, !dbg !17
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %483, <4 x float> %482) #9, !dbg !17
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %484, <4 x float> %438) #9, !dbg !17
  %lds.const.gep1033 = getelementptr i8, ptr addrspace(3) %address.local396, i64 6144, !dbg !17
  %487 = load <4 x half>, ptr addrspace(3) %lds.const.gep1033, align 8, !dbg !17
  %lds.const.gep1034 = getelementptr i8, ptr addrspace(3) %address.local425, i64 4096, !dbg !17
  %488 = load <4 x half>, ptr addrspace(3) %lds.const.gep1034, align 8, !dbg !17
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %487, <4 x float> %486) #9, !dbg !17
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %488, <4 x float> %442) #9, !dbg !17
  %lds.const.gep1035 = getelementptr i8, ptr addrspace(3) %address.local425, i64 6144, !dbg !17
  %491 = load <4 x half>, ptr addrspace(3) %lds.const.gep1035, align 8, !dbg !17
  %lds.const.gep1036 = getelementptr i8, ptr addrspace(3) %address.local396, i64 4224, !dbg !17
  %492 = load <4 x half>, ptr addrspace(3) %lds.const.gep1036, align 8, !dbg !17
  %493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %491, <4 x float> %490) #9, !dbg !17
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %492, <4 x float> %446) #9, !dbg !17
  %lds.const.gep1037 = getelementptr i8, ptr addrspace(3) %address.local396, i64 6272, !dbg !17
  %495 = load <4 x half>, ptr addrspace(3) %lds.const.gep1037, align 8, !dbg !17
  %lds.const.gep1038 = getelementptr i8, ptr addrspace(3) %address.local425, i64 4224, !dbg !17
  %496 = load <4 x half>, ptr addrspace(3) %lds.const.gep1038, align 8, !dbg !17
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %495, <4 x float> %494) #9, !dbg !17
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %496, <4 x float> %449) #9, !dbg !17
  %lds.const.gep1039 = getelementptr i8, ptr addrspace(3) %address.local425, i64 6272, !dbg !17
  %499 = load <4 x half>, ptr addrspace(3) %lds.const.gep1039, align 8, !dbg !17
  %500 = or disjoint i64 %215, 63, !dbg !14
  %501 = icmp uge i64 %64, %500, !dbg !14
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %499, <4 x float> %498) #9, !dbg !17
  %503 = and i1 %190, %501, !dbg !14
  br i1 %503, label %504, label %505, !dbg !14

504:                                              ; preds = %214
  br label %575, !dbg !14

505:                                              ; preds = %214
  %506 = add i64 %64, %32, !dbg !14
  %507 = shl nuw nsw i64 %65, 4, !dbg !14
  %508 = add i64 %506, %507, !dbg !14
  %509 = and i32 %31, 240, !dbg !14
  %index.compact = zext nneg i32 %509 to i64, !dbg !14
  %510 = sub i64 %508, %index.compact, !dbg !14
  %511 = and i64 %32, 48, !dbg !14
  %512 = or disjoint i64 %215, %511, !dbg !14
  %513 = add i64 %510, 1, !dbg !23
  %514 = lshr i64 %513, 32, !dbg !24
  %compare.high = trunc i64 %514 to i32, !dbg !24
  %515 = lshr i64 %512, 32, !dbg !24
  %compare.high843 = trunc i64 %515 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high843, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high843, !dbg !24
  %compare.low.base = trunc i64 %513 to i32, !dbg !24
  %compare.low.base844 = trunc i64 %512 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base844, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %516 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low845 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result846 = select i1 %compare.high.equal, i1 %compare.low845, i1 %compare.high.order, !dbg !24
  %517 = select i1 %compare.result846, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base847 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local848 = or i32 %compare.local.base847, 2, !dbg !24
  %compare.low849 = icmp ugt i32 %compare.low.base, %compare.local848, !dbg !24
  %compare.result850 = select i1 %compare.high.equal, i1 %compare.low849, i1 %compare.high.order, !dbg !24
  %518 = select i1 %compare.result850, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base851 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local852 = or i32 %compare.local.base851, 3, !dbg !24
  %compare.low853 = icmp ugt i32 %compare.low.base, %compare.local852, !dbg !24
  %compare.result854 = select i1 %compare.high.equal, i1 %compare.low853, i1 %compare.high.order, !dbg !24
  %519 = select i1 %compare.result854, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base855 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local856 = or i32 %compare.local.base855, 4, !dbg !24
  %compare.low857 = icmp ugt i32 %compare.low.base, %compare.local856, !dbg !24
  %compare.result858 = select i1 %compare.high.equal, i1 %compare.low857, i1 %compare.high.order, !dbg !24
  %520 = select i1 %compare.result858, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base859 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local860 = or i32 %compare.local.base859, 5, !dbg !24
  %compare.low861 = icmp ugt i32 %compare.low.base, %compare.local860, !dbg !24
  %compare.result862 = select i1 %compare.high.equal, i1 %compare.low861, i1 %compare.high.order, !dbg !24
  %521 = select i1 %compare.result862, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base863 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local864 = or i32 %compare.local.base863, 6, !dbg !24
  %compare.low865 = icmp ugt i32 %compare.low.base, %compare.local864, !dbg !24
  %compare.result866 = select i1 %compare.high.equal, i1 %compare.low865, i1 %compare.high.order, !dbg !24
  %522 = select i1 %compare.result866, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base867 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local868 = or i32 %compare.local.base867, 7, !dbg !24
  %compare.low869 = icmp ugt i32 %compare.low.base, %compare.local868, !dbg !24
  %compare.result870 = select i1 %compare.high.equal, i1 %compare.low869, i1 %compare.high.order, !dbg !24
  %523 = select i1 %compare.result870, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base871 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local872 = or i32 %compare.local.base871, 8, !dbg !24
  %compare.low873 = icmp ugt i32 %compare.low.base, %compare.local872, !dbg !24
  %compare.result874 = select i1 %compare.high.equal, i1 %compare.low873, i1 %compare.high.order, !dbg !24
  %524 = select i1 %compare.result874, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base875 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local876 = or i32 %compare.local.base875, 9, !dbg !24
  %compare.low877 = icmp ugt i32 %compare.low.base, %compare.local876, !dbg !24
  %compare.result878 = select i1 %compare.high.equal, i1 %compare.low877, i1 %compare.high.order, !dbg !24
  %525 = select i1 %compare.result878, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base879 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local880 = or i32 %compare.local.base879, 10, !dbg !24
  %compare.low881 = icmp ugt i32 %compare.low.base, %compare.local880, !dbg !24
  %compare.result882 = select i1 %compare.high.equal, i1 %compare.low881, i1 %compare.high.order, !dbg !24
  %526 = select i1 %compare.result882, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base883 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local884 = or i32 %compare.local.base883, 11, !dbg !24
  %compare.low885 = icmp ugt i32 %compare.low.base, %compare.local884, !dbg !24
  %compare.result886 = select i1 %compare.high.equal, i1 %compare.low885, i1 %compare.high.order, !dbg !24
  %527 = select i1 %compare.result886, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base887 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local888 = or i32 %compare.local.base887, 12, !dbg !24
  %compare.low889 = icmp ugt i32 %compare.low.base, %compare.local888, !dbg !24
  %compare.result890 = select i1 %compare.high.equal, i1 %compare.low889, i1 %compare.high.order, !dbg !24
  %528 = select i1 %compare.result890, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base891 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local892 = or i32 %compare.local.base891, 13, !dbg !24
  %compare.low893 = icmp ugt i32 %compare.low.base, %compare.local892, !dbg !24
  %compare.result894 = select i1 %compare.high.equal, i1 %compare.low893, i1 %compare.high.order, !dbg !24
  %529 = select i1 %compare.result894, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base895 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local896 = or i32 %compare.local.base895, 14, !dbg !24
  %compare.low897 = icmp ugt i32 %compare.low.base, %compare.local896, !dbg !24
  %compare.result898 = select i1 %compare.high.equal, i1 %compare.low897, i1 %compare.high.order, !dbg !24
  %530 = select i1 %compare.result898, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base899 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local900 = or i32 %compare.local.base899, 15, !dbg !24
  %compare.low901 = icmp ugt i32 %compare.low.base, %compare.local900, !dbg !24
  %compare.result902 = select i1 %compare.high.equal, i1 %compare.low901, i1 %compare.high.order, !dbg !24
  %531 = select i1 %compare.result902, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %532 = add i64 %510, 65, !dbg !23
  %533 = lshr i64 %532, 32, !dbg !24
  %compare.high903 = trunc i64 %533 to i32, !dbg !24
  %534 = lshr i64 %512, 32, !dbg !24
  %compare.high904 = trunc i64 %534 to i32, !dbg !24
  %compare.high.equal905 = icmp eq i32 %compare.high903, %compare.high904, !dbg !24
  %compare.high.order906 = icmp ugt i32 %compare.high903, %compare.high904, !dbg !24
  %compare.low.base907 = trunc i64 %532 to i32, !dbg !24
  %compare.low.base908 = trunc i64 %512 to i32, !dbg !24
  %compare.low909 = icmp ugt i32 %compare.low.base907, %compare.low.base908, !dbg !24
  %compare.result910 = select i1 %compare.high.equal905, i1 %compare.low909, i1 %compare.high.order906, !dbg !24
  %535 = select i1 %compare.result910, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base911 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local912 = or i32 %compare.local.base911, 1, !dbg !24
  %compare.low913 = icmp ugt i32 %compare.low.base907, %compare.local912, !dbg !24
  %compare.result914 = select i1 %compare.high.equal905, i1 %compare.low913, i1 %compare.high.order906, !dbg !24
  %536 = select i1 %compare.result914, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base915 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local916 = or i32 %compare.local.base915, 2, !dbg !24
  %compare.low917 = icmp ugt i32 %compare.low.base907, %compare.local916, !dbg !24
  %compare.result918 = select i1 %compare.high.equal905, i1 %compare.low917, i1 %compare.high.order906, !dbg !24
  %537 = select i1 %compare.result918, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base919 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local920 = or i32 %compare.local.base919, 3, !dbg !24
  %compare.low921 = icmp ugt i32 %compare.low.base907, %compare.local920, !dbg !24
  %compare.result922 = select i1 %compare.high.equal905, i1 %compare.low921, i1 %compare.high.order906, !dbg !24
  %538 = select i1 %compare.result922, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base923 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local924 = or i32 %compare.local.base923, 4, !dbg !24
  %compare.low925 = icmp ugt i32 %compare.low.base907, %compare.local924, !dbg !24
  %compare.result926 = select i1 %compare.high.equal905, i1 %compare.low925, i1 %compare.high.order906, !dbg !24
  %539 = select i1 %compare.result926, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base927 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local928 = or i32 %compare.local.base927, 5, !dbg !24
  %compare.low929 = icmp ugt i32 %compare.low.base907, %compare.local928, !dbg !24
  %compare.result930 = select i1 %compare.high.equal905, i1 %compare.low929, i1 %compare.high.order906, !dbg !24
  %540 = select i1 %compare.result930, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base931 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local932 = or i32 %compare.local.base931, 6, !dbg !24
  %compare.low933 = icmp ugt i32 %compare.low.base907, %compare.local932, !dbg !24
  %compare.result934 = select i1 %compare.high.equal905, i1 %compare.low933, i1 %compare.high.order906, !dbg !24
  %541 = select i1 %compare.result934, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base935 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local936 = or i32 %compare.local.base935, 7, !dbg !24
  %compare.low937 = icmp ugt i32 %compare.low.base907, %compare.local936, !dbg !24
  %compare.result938 = select i1 %compare.high.equal905, i1 %compare.low937, i1 %compare.high.order906, !dbg !24
  %542 = select i1 %compare.result938, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base939 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local940 = or i32 %compare.local.base939, 8, !dbg !24
  %compare.low941 = icmp ugt i32 %compare.low.base907, %compare.local940, !dbg !24
  %compare.result942 = select i1 %compare.high.equal905, i1 %compare.low941, i1 %compare.high.order906, !dbg !24
  %543 = select i1 %compare.result942, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base943 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local944 = or i32 %compare.local.base943, 9, !dbg !24
  %compare.low945 = icmp ugt i32 %compare.low.base907, %compare.local944, !dbg !24
  %compare.result946 = select i1 %compare.high.equal905, i1 %compare.low945, i1 %compare.high.order906, !dbg !24
  %544 = select i1 %compare.result946, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base947 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local948 = or i32 %compare.local.base947, 10, !dbg !24
  %compare.low949 = icmp ugt i32 %compare.low.base907, %compare.local948, !dbg !24
  %compare.result950 = select i1 %compare.high.equal905, i1 %compare.low949, i1 %compare.high.order906, !dbg !24
  %545 = select i1 %compare.result950, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base951 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local952 = or i32 %compare.local.base951, 11, !dbg !24
  %compare.low953 = icmp ugt i32 %compare.low.base907, %compare.local952, !dbg !24
  %compare.result954 = select i1 %compare.high.equal905, i1 %compare.low953, i1 %compare.high.order906, !dbg !24
  %546 = select i1 %compare.result954, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base955 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local956 = or i32 %compare.local.base955, 12, !dbg !24
  %compare.low957 = icmp ugt i32 %compare.low.base907, %compare.local956, !dbg !24
  %compare.result958 = select i1 %compare.high.equal905, i1 %compare.low957, i1 %compare.high.order906, !dbg !24
  %547 = select i1 %compare.result958, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base959 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local960 = or i32 %compare.local.base959, 13, !dbg !24
  %compare.low961 = icmp ugt i32 %compare.low.base907, %compare.local960, !dbg !24
  %compare.result962 = select i1 %compare.high.equal905, i1 %compare.low961, i1 %compare.high.order906, !dbg !24
  %548 = select i1 %compare.result962, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base963 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local964 = or i32 %compare.local.base963, 14, !dbg !24
  %compare.low965 = icmp ugt i32 %compare.low.base907, %compare.local964, !dbg !24
  %compare.result966 = select i1 %compare.high.equal905, i1 %compare.low965, i1 %compare.high.order906, !dbg !24
  %549 = select i1 %compare.result966, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base967 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base908, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local968 = or i32 %compare.local.base967, 15, !dbg !24
  %compare.low969 = icmp ugt i32 %compare.low.base907, %compare.local968, !dbg !24
  %compare.result970 = select i1 %compare.high.equal905, i1 %compare.low969, i1 %compare.high.order906, !dbg !24
  %550 = select i1 %compare.result970, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %551 = insertelement <4 x float> poison, float %516, i64 0, !dbg !14
  %552 = insertelement <4 x float> %551, float %517, i64 1, !dbg !14
  %553 = insertelement <4 x float> %552, float %518, i64 2, !dbg !14
  %.masked76 = insertelement <4 x float> %553, float %519, i64 3, !dbg !14
  %554 = insertelement <4 x float> poison, float %520, i64 0, !dbg !14
  %555 = insertelement <4 x float> %554, float %521, i64 1, !dbg !14
  %556 = insertelement <4 x float> %555, float %522, i64 2, !dbg !14
  %.masked78 = insertelement <4 x float> %556, float %523, i64 3, !dbg !14
  %557 = insertelement <4 x float> poison, float %524, i64 0, !dbg !14
  %558 = insertelement <4 x float> %557, float %525, i64 1, !dbg !14
  %559 = insertelement <4 x float> %558, float %526, i64 2, !dbg !14
  %.masked80 = insertelement <4 x float> %559, float %527, i64 3, !dbg !14
  %560 = insertelement <4 x float> poison, float %528, i64 0, !dbg !14
  %561 = insertelement <4 x float> %560, float %529, i64 1, !dbg !14
  %562 = insertelement <4 x float> %561, float %530, i64 2, !dbg !14
  %.masked82 = insertelement <4 x float> %562, float %531, i64 3, !dbg !14
  %563 = insertelement <4 x float> poison, float %535, i64 0, !dbg !14
  %564 = insertelement <4 x float> %563, float %536, i64 1, !dbg !14
  %565 = insertelement <4 x float> %564, float %537, i64 2, !dbg !14
  %.masked85 = insertelement <4 x float> %565, float %538, i64 3, !dbg !14
  %566 = insertelement <4 x float> poison, float %539, i64 0, !dbg !14
  %567 = insertelement <4 x float> %566, float %540, i64 1, !dbg !14
  %568 = insertelement <4 x float> %567, float %541, i64 2, !dbg !14
  %.masked87 = insertelement <4 x float> %568, float %542, i64 3, !dbg !14
  %569 = insertelement <4 x float> poison, float %543, i64 0, !dbg !14
  %570 = insertelement <4 x float> %569, float %544, i64 1, !dbg !14
  %571 = insertelement <4 x float> %570, float %545, i64 2, !dbg !14
  %.masked89 = insertelement <4 x float> %571, float %546, i64 3, !dbg !14
  %572 = insertelement <4 x float> poison, float %547, i64 0, !dbg !14
  %573 = insertelement <4 x float> %572, float %548, i64 1, !dbg !14
  %574 = insertelement <4 x float> %573, float %549, i64 2, !dbg !14
  %.masked91 = insertelement <4 x float> %574, float %550, i64 3, !dbg !14
  %.masked77 = fadd <4 x float> %473, %.masked76, !dbg !26
  %.masked79 = fadd <4 x float> %477, %.masked78, !dbg !26
  %.masked81 = fadd <4 x float> %481, %.masked80, !dbg !26
  %.masked83 = fadd <4 x float> %485, %.masked82, !dbg !26
  %.masked86 = fadd <4 x float> %489, %.masked85, !dbg !26
  %.masked88 = fadd <4 x float> %493, %.masked87, !dbg !26
  %.masked90 = fadd <4 x float> %497, %.masked89, !dbg !26
  %.masked92 = fadd <4 x float> %502, %.masked91, !dbg !26
  br label %575, !dbg !14

575:                                              ; preds = %504, %505
  %576 = phi <4 x float> [ %.masked77, %505 ], [ %473, %504 ], !dbg !26
  %577 = phi <4 x float> [ %.masked79, %505 ], [ %477, %504 ], !dbg !26
  %578 = phi <4 x float> [ %.masked81, %505 ], [ %481, %504 ], !dbg !26
  %579 = phi <4 x float> [ %.masked83, %505 ], [ %485, %504 ], !dbg !26
  %580 = phi <4 x float> [ %.masked86, %505 ], [ %489, %504 ], !dbg !26
  %581 = phi <4 x float> [ %.masked88, %505 ], [ %493, %504 ], !dbg !26
  %582 = phi <4 x float> [ %.masked90, %505 ], [ %497, %504 ], !dbg !26
  %583 = phi <4 x float> [ %.masked92, %505 ], [ %502, %504 ], !dbg !26
  br label %584, !dbg !14

584:                                              ; preds = %575
  %585 = extractelement <4 x float> %576, i64 0, !dbg !26
  %586 = extractelement <4 x float> %576, i64 1, !dbg !26
  %587 = extractelement <4 x float> %576, i64 2, !dbg !26
  %588 = extractelement <4 x float> %576, i64 3, !dbg !26
  %589 = extractelement <4 x float> %577, i64 0, !dbg !26
  %590 = extractelement <4 x float> %577, i64 1, !dbg !26
  %591 = extractelement <4 x float> %577, i64 2, !dbg !26
  %592 = extractelement <4 x float> %577, i64 3, !dbg !26
  %593 = extractelement <4 x float> %578, i64 0, !dbg !26
  %594 = extractelement <4 x float> %578, i64 1, !dbg !26
  %595 = extractelement <4 x float> %578, i64 2, !dbg !26
  %596 = extractelement <4 x float> %578, i64 3, !dbg !26
  %597 = extractelement <4 x float> %579, i64 0, !dbg !26
  %598 = extractelement <4 x float> %579, i64 1, !dbg !26
  %599 = extractelement <4 x float> %579, i64 2, !dbg !26
  %600 = extractelement <4 x float> %579, i64 3, !dbg !26
  %601 = extractelement <4 x float> %580, i64 0, !dbg !26
  %602 = extractelement <4 x float> %580, i64 1, !dbg !26
  %603 = extractelement <4 x float> %580, i64 2, !dbg !26
  %604 = extractelement <4 x float> %580, i64 3, !dbg !26
  %605 = extractelement <4 x float> %581, i64 0, !dbg !26
  %606 = extractelement <4 x float> %581, i64 1, !dbg !26
  %607 = extractelement <4 x float> %581, i64 2, !dbg !26
  %608 = extractelement <4 x float> %581, i64 3, !dbg !26
  %609 = extractelement <4 x float> %582, i64 0, !dbg !26
  %610 = extractelement <4 x float> %582, i64 1, !dbg !26
  %611 = extractelement <4 x float> %582, i64 2, !dbg !26
  %612 = extractelement <4 x float> %582, i64 3, !dbg !26
  %613 = extractelement <4 x float> %583, i64 0, !dbg !26
  %614 = extractelement <4 x float> %583, i64 1, !dbg !26
  %615 = extractelement <4 x float> %583, i64 2, !dbg !26
  %616 = extractelement <4 x float> %583, i64 3, !dbg !26
  %617 = call float @llvm.maxnum.f32(float %585, float 0xFFF0000000000000), !dbg !27, !frisk.local_reduction !1
  %local.reduce137 = call float @llvm.maxnum.f32(float %617, float %589), !dbg !27, !frisk.local_reduction !1
  %local.reduce138 = call float @llvm.maxnum.f32(float %586, float %590), !dbg !27, !frisk.local_reduction !1
  %local.reduce139 = call float @llvm.maxnum.f32(float %587, float %591), !dbg !27, !frisk.local_reduction !1
  %local.reduce140 = call float @llvm.maxnum.f32(float %588, float %592), !dbg !27, !frisk.local_reduction !1
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %593), !dbg !27, !frisk.local_reduction !1
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %594), !dbg !27, !frisk.local_reduction !1
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %595), !dbg !27, !frisk.local_reduction !1
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %596), !dbg !27, !frisk.local_reduction !1
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %597), !dbg !27, !frisk.local_reduction !1
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %598), !dbg !27, !frisk.local_reduction !1
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %599), !dbg !27, !frisk.local_reduction !1
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %600), !dbg !27, !frisk.local_reduction !1
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce145, float %local.reduce146), !dbg !27, !frisk.local_reduction !1
  %local.reduce150 = call float @llvm.maxnum.f32(float %local.reduce147, float %local.reduce148), !dbg !27, !frisk.local_reduction !1
  %618 = call float @llvm.maxnum.f32(float %local.reduce149, float %local.reduce150), !dbg !27, !frisk.local_reduction !1
  %619 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %620 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %619), !dbg !27
  %621 = add nuw nsw i32 %620, 64, !dbg !27
  %622 = xor i32 %620, 16, !dbg !27
  %623 = and i32 %621, 192, !dbg !27
  %624 = icmp samesign ult i32 %622, %623, !dbg !27
  %625 = select i1 %624, i32 %622, i32 %620, !dbg !27
  %626 = shl nuw nsw i32 %625, 2, !dbg !27
  %627 = bitcast float %618 to i32, !dbg !27
  %628 = call i32 @llvm.amdgcn.ds.bpermute(i32 %626, i32 %627), !dbg !27
  %629 = bitcast i32 %628 to float, !dbg !27
  %630 = call float @llvm.maxnum.f32(float %618, float %629), !dbg !27
  %631 = xor i32 %620, 32, !dbg !27
  %632 = icmp samesign ult i32 %631, %623, !dbg !27
  %633 = select i1 %632, i32 %631, i32 %620, !dbg !27
  %634 = shl nuw nsw i32 %633, 2, !dbg !27
  %635 = bitcast float %630 to i32, !dbg !27
  %636 = call i32 @llvm.amdgcn.ds.bpermute(i32 %634, i32 %635), !dbg !27
  %637 = bitcast i32 %636 to float, !dbg !27
  %638 = call float @llvm.maxnum.f32(float %630, float %637), !dbg !27
  %639 = extractvalue [2 x <1 x float>] %209, 0, !dbg !27
  %640 = insertelement <1 x float> poison, float %638, i64 0, !dbg !27
  %641 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %640, <1 x float> %639), !dbg !27
  %642 = call float @llvm.maxnum.f32(float %601, float 0xFFF0000000000000), !dbg !27, !frisk.local_reduction !1
  %local.reduce122 = call float @llvm.maxnum.f32(float %642, float %605), !dbg !27, !frisk.local_reduction !1
  %local.reduce123 = call float @llvm.maxnum.f32(float %602, float %606), !dbg !27, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %603, float %607), !dbg !27, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %604, float %608), !dbg !27, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %609), !dbg !27, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %610), !dbg !27, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %611), !dbg !27, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %612), !dbg !27, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %613), !dbg !27, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %614), !dbg !27, !frisk.local_reduction !1
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %615), !dbg !27, !frisk.local_reduction !1
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %616), !dbg !27, !frisk.local_reduction !1
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %local.reduce131), !dbg !27, !frisk.local_reduction !1
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce132, float %local.reduce133), !dbg !27, !frisk.local_reduction !1
  %643 = call float @llvm.maxnum.f32(float %local.reduce134, float %local.reduce135), !dbg !27, !frisk.local_reduction !1
  %644 = bitcast float %643 to i32, !dbg !27
  %645 = call i32 @llvm.amdgcn.ds.bpermute(i32 %626, i32 %644), !dbg !27
  %646 = bitcast i32 %645 to float, !dbg !27
  %647 = call float @llvm.maxnum.f32(float %643, float %646), !dbg !27
  %648 = bitcast float %647 to i32, !dbg !27
  %649 = call i32 @llvm.amdgcn.ds.bpermute(i32 %634, i32 %648), !dbg !27
  %650 = bitcast i32 %649 to float, !dbg !27
  %651 = call float @llvm.maxnum.f32(float %647, float %650), !dbg !27
  %652 = extractvalue [2 x <1 x float>] %209, 1, !dbg !27
  %653 = insertelement <1 x float> poison, float %651, i64 0, !dbg !27
  %654 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %653, <1 x float> %652), !dbg !27
  %655 = shufflevector <1 x float> %641, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %656 = fsub <4 x float> %576, %655, !dbg !28
  %657 = fsub <4 x float> %577, %655, !dbg !28
  %658 = fsub <4 x float> %578, %655, !dbg !28
  %659 = fsub <4 x float> %579, %655, !dbg !28
  %660 = shufflevector <1 x float> %654, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %661 = fsub <4 x float> %580, %660, !dbg !28
  %662 = fsub <4 x float> %581, %660, !dbg !28
  %663 = fsub <4 x float> %582, %660, !dbg !28
  %664 = fsub <4 x float> %583, %660, !dbg !28
  %665 = fsub <1 x float> %639, %641, !dbg !29
  %666 = extractelement <1 x float> %665, i64 0, !dbg !30
  %667 = call float @llvm.amdgcn.exp2.f32(float %666), !dbg !30
  %668 = insertelement <4 x float> poison, float %667, i64 0, !dbg !35
  %669 = shufflevector <4 x float> %668, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %670 = extractvalue [2 x <1 x float>] %208, 0, !dbg !31
  %671 = extractelement <1 x float> %670, i64 0, !dbg !31
  %672 = fmul float %671, %667, !dbg !31
  %673 = fsub <1 x float> %652, %654, !dbg !29
  %674 = extractelement <1 x float> %673, i64 0, !dbg !30
  %675 = call float @llvm.amdgcn.exp2.f32(float %674), !dbg !30
  %676 = insertelement <4 x float> poison, float %675, i64 0, !dbg !35
  %677 = shufflevector <4 x float> %676, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %678 = extractvalue [2 x <1 x float>] %208, 1, !dbg !31
  %679 = extractelement <1 x float> %678, i64 0, !dbg !31
  %680 = fmul float %679, %675, !dbg !31
  %681 = extractelement <4 x float> %656, i64 0, !dbg !32
  %682 = call float @llvm.amdgcn.exp2.f32(float %681), !dbg !32
  %683 = insertelement <4 x float> poison, float %682, i64 0, !dbg !34
  %684 = extractelement <4 x float> %656, i64 1, !dbg !32
  %685 = call float @llvm.amdgcn.exp2.f32(float %684), !dbg !32
  %686 = insertelement <4 x float> %683, float %685, i64 1, !dbg !34
  %687 = extractelement <4 x float> %656, i64 2, !dbg !32
  %688 = call float @llvm.amdgcn.exp2.f32(float %687), !dbg !32
  %689 = insertelement <4 x float> %686, float %688, i64 2, !dbg !34
  %690 = extractelement <4 x float> %656, i64 3, !dbg !32
  %691 = call float @llvm.amdgcn.exp2.f32(float %690), !dbg !32
  %692 = insertelement <4 x float> %689, float %691, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %692) #9, !dbg !34, !frisk.register_lifetimes !1
  %693 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %693) #9, !dbg !32, !frisk.register_lifetimes !1
  %694 = extractelement <4 x float> %657, i64 0, !dbg !32
  %695 = call float @llvm.amdgcn.exp2.f32(float %694), !dbg !32
  %696 = insertelement <4 x float> poison, float %695, i64 0, !dbg !34
  %local.reduce107 = fadd float %682, %695, !dbg !33, !frisk.local_reduction !1
  %697 = extractelement <4 x float> %657, i64 1, !dbg !32
  %698 = call float @llvm.amdgcn.exp2.f32(float %697), !dbg !32
  %699 = insertelement <4 x float> %696, float %698, i64 1, !dbg !34
  %local.reduce108 = fadd float %685, %698, !dbg !33, !frisk.local_reduction !1
  %700 = extractelement <4 x float> %657, i64 2, !dbg !32
  %701 = call float @llvm.amdgcn.exp2.f32(float %700), !dbg !32
  %702 = insertelement <4 x float> %699, float %701, i64 2, !dbg !34
  %local.reduce109 = fadd float %688, %701, !dbg !33, !frisk.local_reduction !1
  %703 = extractelement <4 x float> %657, i64 3, !dbg !32
  %704 = call float @llvm.amdgcn.exp2.f32(float %703), !dbg !32
  %705 = insertelement <4 x float> %702, float %704, i64 3, !dbg !34
  %math.result248 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %705) #9, !dbg !34, !frisk.register_lifetimes !1
  %706 = fptrunc <4 x float> %math.result248 to <4 x half>, !dbg !34
  %math.packed249 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %706) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce110 = fadd float %691, %704, !dbg !33, !frisk.local_reduction !1
  %707 = extractelement <4 x float> %658, i64 0, !dbg !32
  %708 = call float @llvm.amdgcn.exp2.f32(float %707), !dbg !32
  %709 = insertelement <4 x float> poison, float %708, i64 0, !dbg !34
  %local.reduce111 = fadd float %local.reduce107, %708, !dbg !33, !frisk.local_reduction !1
  %710 = extractelement <4 x float> %658, i64 1, !dbg !32
  %711 = call float @llvm.amdgcn.exp2.f32(float %710), !dbg !32
  %712 = insertelement <4 x float> %709, float %711, i64 1, !dbg !34
  %local.reduce112 = fadd float %local.reduce108, %711, !dbg !33, !frisk.local_reduction !1
  %713 = extractelement <4 x float> %658, i64 2, !dbg !32
  %714 = call float @llvm.amdgcn.exp2.f32(float %713), !dbg !32
  %715 = insertelement <4 x float> %712, float %714, i64 2, !dbg !34
  %local.reduce113 = fadd float %local.reduce109, %714, !dbg !33, !frisk.local_reduction !1
  %716 = extractelement <4 x float> %658, i64 3, !dbg !32
  %717 = call float @llvm.amdgcn.exp2.f32(float %716), !dbg !32
  %718 = insertelement <4 x float> %715, float %717, i64 3, !dbg !34
  %math.result250 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %718) #9, !dbg !34, !frisk.register_lifetimes !1
  %719 = fptrunc <4 x float> %math.result250 to <4 x half>, !dbg !34
  %math.packed251 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %719) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce114 = fadd float %local.reduce110, %717, !dbg !33, !frisk.local_reduction !1
  %720 = extractelement <4 x float> %659, i64 0, !dbg !32
  %721 = call float @llvm.amdgcn.exp2.f32(float %720), !dbg !32
  %722 = insertelement <4 x float> poison, float %721, i64 0, !dbg !34
  %local.reduce115 = fadd float %local.reduce111, %721, !dbg !33, !frisk.local_reduction !1
  %723 = extractelement <4 x float> %659, i64 1, !dbg !32
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !32
  %725 = insertelement <4 x float> %722, float %724, i64 1, !dbg !34
  %local.reduce116 = fadd float %local.reduce112, %724, !dbg !33, !frisk.local_reduction !1
  %local.reduce119 = fadd float %local.reduce115, %local.reduce116, !dbg !33, !frisk.local_reduction !1
  %726 = extractelement <4 x float> %659, i64 2, !dbg !32
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !32
  %728 = insertelement <4 x float> %725, float %727, i64 2, !dbg !34
  %local.reduce117 = fadd float %local.reduce113, %727, !dbg !33, !frisk.local_reduction !1
  %729 = extractelement <4 x float> %659, i64 3, !dbg !32
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !32
  %731 = insertelement <4 x float> %728, float %730, i64 3, !dbg !34
  %math.result252 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %731) #9, !dbg !34, !frisk.register_lifetimes !1
  %732 = fptrunc <4 x float> %math.result252 to <4 x half>, !dbg !34
  %math.packed253 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %732) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce118 = fadd float %local.reduce114, %730, !dbg !33, !frisk.local_reduction !1
  %local.reduce120 = fadd float %local.reduce117, %local.reduce118, !dbg !33, !frisk.local_reduction !1
  %733 = fadd float %local.reduce119, %local.reduce120, !dbg !33, !frisk.local_reduction !1
  %734 = fadd float %733, %672, !dbg !33
  %735 = insertelement <1 x float> poison, float %734, i64 0, !dbg !7
  %736 = insertvalue [2 x <1 x float>] poison, <1 x float> %735, 0, !dbg !7
  %737 = extractelement <4 x float> %661, i64 0, !dbg !32
  %738 = call float @llvm.amdgcn.exp2.f32(float %737), !dbg !32
  %739 = insertelement <4 x float> poison, float %738, i64 0, !dbg !34
  %740 = extractelement <4 x float> %661, i64 1, !dbg !32
  %741 = call float @llvm.amdgcn.exp2.f32(float %740), !dbg !32
  %742 = insertelement <4 x float> %739, float %741, i64 1, !dbg !34
  %743 = extractelement <4 x float> %661, i64 2, !dbg !32
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !32
  %745 = insertelement <4 x float> %742, float %744, i64 2, !dbg !34
  %746 = extractelement <4 x float> %661, i64 3, !dbg !32
  %747 = call float @llvm.amdgcn.exp2.f32(float %746), !dbg !32
  %748 = insertelement <4 x float> %745, float %747, i64 3, !dbg !34
  %math.result254 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %748) #9, !dbg !34, !frisk.register_lifetimes !1
  %749 = fptrunc <4 x float> %math.result254 to <4 x half>, !dbg !34
  %math.packed255 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %749) #9, !dbg !32, !frisk.register_lifetimes !1
  %750 = extractelement <4 x float> %662, i64 0, !dbg !32
  %751 = call float @llvm.amdgcn.exp2.f32(float %750), !dbg !32
  %752 = insertelement <4 x float> poison, float %751, i64 0, !dbg !34
  %local.reduce = fadd float %738, %751, !dbg !33, !frisk.local_reduction !1
  %753 = extractelement <4 x float> %662, i64 1, !dbg !32
  %754 = call float @llvm.amdgcn.exp2.f32(float %753), !dbg !32
  %755 = insertelement <4 x float> %752, float %754, i64 1, !dbg !34
  %local.reduce93 = fadd float %741, %754, !dbg !33, !frisk.local_reduction !1
  %756 = extractelement <4 x float> %662, i64 2, !dbg !32
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !32
  %758 = insertelement <4 x float> %755, float %757, i64 2, !dbg !34
  %local.reduce94 = fadd float %744, %757, !dbg !33, !frisk.local_reduction !1
  %759 = extractelement <4 x float> %662, i64 3, !dbg !32
  %760 = call float @llvm.amdgcn.exp2.f32(float %759), !dbg !32
  %761 = insertelement <4 x float> %758, float %760, i64 3, !dbg !34
  %math.result256 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %761) #9, !dbg !34, !frisk.register_lifetimes !1
  %762 = fptrunc <4 x float> %math.result256 to <4 x half>, !dbg !34
  %math.packed257 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %762) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce95 = fadd float %747, %760, !dbg !33, !frisk.local_reduction !1
  %763 = extractelement <4 x float> %663, i64 0, !dbg !32
  %764 = call float @llvm.amdgcn.exp2.f32(float %763), !dbg !32
  %765 = insertelement <4 x float> poison, float %764, i64 0, !dbg !34
  %local.reduce96 = fadd float %local.reduce, %764, !dbg !33, !frisk.local_reduction !1
  %766 = extractelement <4 x float> %663, i64 1, !dbg !32
  %767 = call float @llvm.amdgcn.exp2.f32(float %766), !dbg !32
  %768 = insertelement <4 x float> %765, float %767, i64 1, !dbg !34
  %local.reduce97 = fadd float %local.reduce93, %767, !dbg !33, !frisk.local_reduction !1
  %769 = extractelement <4 x float> %663, i64 2, !dbg !32
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !32
  %771 = insertelement <4 x float> %768, float %770, i64 2, !dbg !34
  %local.reduce98 = fadd float %local.reduce94, %770, !dbg !33, !frisk.local_reduction !1
  %772 = extractelement <4 x float> %663, i64 3, !dbg !32
  %773 = call float @llvm.amdgcn.exp2.f32(float %772), !dbg !32
  %774 = insertelement <4 x float> %771, float %773, i64 3, !dbg !34
  %math.result258 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %774) #9, !dbg !34, !frisk.register_lifetimes !1
  %775 = fptrunc <4 x float> %math.result258 to <4 x half>, !dbg !34
  %math.packed259 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %775) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce99 = fadd float %local.reduce95, %773, !dbg !33, !frisk.local_reduction !1
  %776 = extractelement <4 x float> %664, i64 0, !dbg !32
  %777 = call float @llvm.amdgcn.exp2.f32(float %776), !dbg !32
  %778 = insertelement <4 x float> poison, float %777, i64 0, !dbg !34
  %local.reduce100 = fadd float %local.reduce96, %777, !dbg !33, !frisk.local_reduction !1
  %779 = extractelement <4 x float> %664, i64 1, !dbg !32
  %780 = call float @llvm.amdgcn.exp2.f32(float %779), !dbg !32
  %781 = insertelement <4 x float> %778, float %780, i64 1, !dbg !34
  %local.reduce101 = fadd float %local.reduce97, %780, !dbg !33, !frisk.local_reduction !1
  %local.reduce104 = fadd float %local.reduce100, %local.reduce101, !dbg !33, !frisk.local_reduction !1
  %782 = extractelement <4 x float> %664, i64 2, !dbg !32
  %783 = call float @llvm.amdgcn.exp2.f32(float %782), !dbg !32
  %784 = insertelement <4 x float> %781, float %783, i64 2, !dbg !34
  %local.reduce102 = fadd float %local.reduce98, %783, !dbg !33, !frisk.local_reduction !1
  %785 = extractelement <4 x float> %664, i64 3, !dbg !32
  %786 = call float @llvm.amdgcn.exp2.f32(float %785), !dbg !32
  %787 = insertelement <4 x float> %784, float %786, i64 3, !dbg !34
  %math.result260 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %787) #9, !dbg !34, !frisk.register_lifetimes !1
  %788 = fptrunc <4 x float> %math.result260 to <4 x half>, !dbg !34
  %math.packed261 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %788) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce103 = fadd float %local.reduce99, %786, !dbg !33, !frisk.local_reduction !1
  %local.reduce105 = fadd float %local.reduce102, %local.reduce103, !dbg !33, !frisk.local_reduction !1
  %789 = fadd float %local.reduce104, %local.reduce105, !dbg !33, !frisk.local_reduction !1
  %790 = fadd float %789, %680, !dbg !33
  %791 = insertelement <1 x float> poison, float %790, i64 0, !dbg !7
  %792 = insertvalue [2 x <1 x float>] %736, <1 x float> %791, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread582 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local583 = zext nneg i32 %address.thread582 to i64, !dbg !9
  %address.local584 = and i64 %address.local583, 60, !dbg !15
  %address.local585 = add nuw nsw i64 %address.local584, %address.local583, !dbg !15
  %address.local586 = shl nuw nsw i64 %address.local585, 3, !dbg !15
  %address.local587 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local586, !dbg !15
  %address.local588 = and i64 %address.local583, 63, !dbg !8
  %address.local589 = lshr i64 %address.local588, 4, !dbg !15
  %address.local590 = mul nuw nsw i64 %address.local589, 384, !dbg !15
  %address.local591 = getelementptr [2 x i8], ptr addrspace(3) %address.local587, i64 %address.local590, !dbg !15
  %address.local592 = lshr i64 %address.local583, 2, !dbg !17
  %address.local593 = mul nsw i64 %address.local592, -32, !dbg !15
  %address.local594 = getelementptr [2 x i8], ptr addrspace(3) %address.local591, i64 %address.local593, !dbg !15
  %793 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local594, i32 0) #0, !dbg !15
  %794 = bitcast <4 x i32> %793 to <8 x half>, !dbg !15
  %address.local595 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local586, !dbg !15
  %address.local596 = getelementptr [2 x i8], ptr addrspace(3) %address.local595, i64 %address.local590, !dbg !15
  %address.local597 = getelementptr [2 x i8], ptr addrspace(3) %address.local596, i64 %address.local593, !dbg !15
  %address.local598 = getelementptr i8, ptr addrspace(3) %address.local597, i64 256, !dbg !15
  %795 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local598, i32 0) #0, !dbg !15
  %796 = bitcast <4 x i32> %795 to <8 x half>, !dbg !15
  %address.local599 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local586, !dbg !15
  %address.local600 = getelementptr [2 x i8], ptr addrspace(3) %address.local599, i64 %address.local590, !dbg !15
  %address.local601 = getelementptr [2 x i8], ptr addrspace(3) %address.local600, i64 %address.local593, !dbg !15
  %address.local602 = getelementptr i8, ptr addrspace(3) %address.local601, i64 512, !dbg !15
  %797 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local602, i32 0) #0, !dbg !15
  %798 = bitcast <4 x i32> %797 to <8 x half>, !dbg !15
  %address.local603 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local586, !dbg !15
  %address.local604 = getelementptr [2 x i8], ptr addrspace(3) %address.local603, i64 %address.local590, !dbg !15
  %address.local605 = getelementptr [2 x i8], ptr addrspace(3) %address.local604, i64 %address.local593, !dbg !15
  %address.local606 = getelementptr i8, ptr addrspace(3) %address.local605, i64 768, !dbg !15
  %799 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local606, i32 0) #0, !dbg !15
  %800 = bitcast <4 x i32> %799 to <8 x half>, !dbg !15
  %801 = add nuw nsw i64 %218, %16, !dbg !18
  %802 = shl i64 %801, 19, !dbg !18
  %803 = and i64 %802, 16252928, !dbg !18
  %804 = add i64 %803, %224, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %805 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %804, !dbg !18
  %806 = getelementptr i8, ptr addrspace(1) %805, i64 64, !dbg !18
  %address.local607 = shl nuw nsw i64 %address.local583, 3, !dbg !9
  %address.local608 = and i64 %address.local607, 1536, !dbg !16
  %address.local609 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local608, !dbg !16
  %address.local610 = ptrtoint ptr addrspace(3) %address.local609 to i64, !dbg !16
  %address.local611 = trunc i64 %address.local610 to i32, !dbg !16
  %807 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local611), !dbg !18
  %808 = zext i32 %807 to i64, !dbg !18
  %809 = inttoptr i64 %808 to ptr addrspace(3), !dbg !18
  %address.local612 = shl nuw nsw i64 %address.local583, 5, !dbg !18
  %address.local613 = and i64 %address.local612, 8064, !dbg !18
  %address.local614 = shl nuw nsw i64 %address.local583, 3, !dbg !18
  %address.local615 = and i64 %address.local614, 24, !dbg !18
  %address.local616 = or disjoint i64 %address.local613, %address.local615, !dbg !18
  %address.local617 = getelementptr [2 x i8], ptr addrspace(1) %806, i64 %address.local616, !dbg !18
  %810 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %811 = ptrtoint ptr addrspace(1) %address.local617 to i64, !dbg !18
  %812 = sub i64 %811, %810, !dbg !18
  %dma.buffer.offset1043 = trunc i64 %812 to i32, !dbg !18
  %813 = trunc i64 %810 to i32, !dbg !18
  %814 = insertelement <4 x i32> poison, i32 %813, i64 0, !dbg !18
  %815 = lshr i64 %810, 32, !dbg !18
  %816 = trunc i64 %815 to i32, !dbg !18
  %817 = insertelement <4 x i32> %814, i32 %816, i64 1, !dbg !18
  %818 = insertelement <4 x i32> %817, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1044 = insertelement <4 x i32> %818, i32 131072, i64 3, !dbg !18
  %819 = ptrtoint ptr addrspace(3) %809 to i32, !dbg !18
  %dma.buffer.lds1045 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %819), !dbg !18
  %820 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1045, i32 %dma.buffer.offset1043, <4 x i32> %dma.buffer.rsrc1044) #10, !dbg !18
  %821 = shufflevector <8 x half> %794, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %822 = extractvalue [2 x <8 x float>] %207, 0, !dbg !35
  %823 = shufflevector <8 x float> %822, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %824 = fmul <4 x float> %823, %669, !dbg !35
  %825 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %821, <4 x float> %824) #9, !dbg !15
  %826 = shufflevector <8 x half> %796, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %826, <4 x float> %825) #9, !dbg !15
  %828 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %828, <4 x float> %827) #9, !dbg !15
  %830 = shufflevector <8 x half> %800, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %830, <4 x float> %829) #9, !dbg !15
  %832 = shufflevector <8 x half> %794, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %833 = shufflevector <8 x float> %822, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %834 = fmul <4 x float> %833, %669, !dbg !35
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %832, <4 x float> %834) #9, !dbg !15
  %836 = shufflevector <8 x half> %796, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %836, <4 x float> %835) #9, !dbg !15
  %838 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %838, <4 x float> %837) #9, !dbg !15
  %840 = shufflevector <8 x half> %800, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %841 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %840, <4 x float> %839) #9, !dbg !15
  %842 = extractvalue [2 x <8 x float>] %207, 1, !dbg !35
  %843 = shufflevector <8 x float> %842, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %844 = fmul <4 x float> %843, %677, !dbg !35
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %821, <4 x float> %844) #9, !dbg !15
  %846 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %826, <4 x float> %845) #9, !dbg !15
  %847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %828, <4 x float> %846) #9, !dbg !15
  %848 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %830, <4 x float> %847) #9, !dbg !15
  %849 = shufflevector <8 x float> %842, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %850 = fmul <4 x float> %849, %677, !dbg !35
  %851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %832, <4 x float> %850) #9, !dbg !15
  %852 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %836, <4 x float> %851) #9, !dbg !15
  %853 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %838, <4 x float> %852) #9, !dbg !15
  %854 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %840, <4 x float> %853) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %855 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local594, i32 0) #0, !dbg !15
  %856 = bitcast <4 x i32> %855 to <8 x half>, !dbg !15
  %857 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local598, i32 0) #0, !dbg !15
  %858 = bitcast <4 x i32> %857 to <8 x half>, !dbg !15
  %859 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local602, i32 0) #0, !dbg !15
  %860 = bitcast <4 x i32> %859 to <8 x half>, !dbg !15
  %861 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local606, i32 0) #0, !dbg !15
  %862 = bitcast <4 x i32> %861 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %863 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %804, !dbg !18
  %864 = getelementptr i8, ptr addrspace(1) %863, i64 128, !dbg !18
  %865 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local611), !dbg !18
  %866 = zext i32 %865 to i64, !dbg !18
  %867 = inttoptr i64 %866 to ptr addrspace(3), !dbg !18
  %address.local618 = getelementptr [2 x i8], ptr addrspace(1) %864, i64 %address.local616, !dbg !18
  %868 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %869 = ptrtoint ptr addrspace(1) %address.local618 to i64, !dbg !18
  %870 = sub i64 %869, %868, !dbg !18
  %dma.buffer.offset1040 = trunc i64 %870 to i32, !dbg !18
  %871 = trunc i64 %868 to i32, !dbg !18
  %872 = insertelement <4 x i32> poison, i32 %871, i64 0, !dbg !18
  %873 = lshr i64 %868, 32, !dbg !18
  %874 = trunc i64 %873 to i32, !dbg !18
  %875 = insertelement <4 x i32> %872, i32 %874, i64 1, !dbg !18
  %876 = insertelement <4 x i32> %875, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1041 = insertelement <4 x i32> %876, i32 131072, i64 3, !dbg !18
  %877 = ptrtoint ptr addrspace(3) %867 to i32, !dbg !18
  %dma.buffer.lds1042 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %877), !dbg !18
  %878 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1042, i32 %dma.buffer.offset1040, <4 x i32> %dma.buffer.rsrc1041) #10, !dbg !18
  %879 = shufflevector <8 x half> %856, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %880 = extractvalue [2 x <8 x float>] %210, 0, !dbg !35
  %881 = shufflevector <8 x float> %880, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %882 = fmul <4 x float> %881, %669, !dbg !35
  %883 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %879, <4 x float> %882) #9, !dbg !15
  %884 = shufflevector <8 x half> %858, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %884, <4 x float> %883) #9, !dbg !15
  %886 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %887 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %886, <4 x float> %885) #9, !dbg !15
  %888 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %889 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %888, <4 x float> %887) #9, !dbg !15
  %890 = shufflevector <8 x half> %856, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %891 = shufflevector <8 x float> %880, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %892 = fmul <4 x float> %891, %669, !dbg !35
  %893 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %890, <4 x float> %892) #9, !dbg !15
  %894 = shufflevector <8 x half> %858, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %894, <4 x float> %893) #9, !dbg !15
  %896 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %896, <4 x float> %895) #9, !dbg !15
  %898 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %898, <4 x float> %897) #9, !dbg !15
  %900 = extractvalue [2 x <8 x float>] %210, 1, !dbg !35
  %901 = shufflevector <8 x float> %900, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %902 = fmul <4 x float> %901, %677, !dbg !35
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %879, <4 x float> %902) #9, !dbg !15
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %884, <4 x float> %903) #9, !dbg !15
  %905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %886, <4 x float> %904) #9, !dbg !15
  %906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %888, <4 x float> %905) #9, !dbg !15
  %907 = shufflevector <8 x float> %900, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %908 = fmul <4 x float> %907, %677, !dbg !35
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %890, <4 x float> %908) #9, !dbg !15
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %894, <4 x float> %909) #9, !dbg !15
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %896, <4 x float> %910) #9, !dbg !15
  %912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %898, <4 x float> %911) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %913 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local594, i32 0) #0, !dbg !15
  %914 = bitcast <4 x i32> %913 to <8 x half>, !dbg !15
  %915 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local598, i32 0) #0, !dbg !15
  %916 = bitcast <4 x i32> %915 to <8 x half>, !dbg !15
  %917 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local602, i32 0) #0, !dbg !15
  %918 = bitcast <4 x i32> %917 to <8 x half>, !dbg !15
  %919 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local606, i32 0) #0, !dbg !15
  %920 = bitcast <4 x i32> %919 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %921 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %804, !dbg !18
  %922 = getelementptr i8, ptr addrspace(1) %921, i64 192, !dbg !18
  %923 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local611), !dbg !18
  %924 = zext i32 %923 to i64, !dbg !18
  %925 = inttoptr i64 %924 to ptr addrspace(3), !dbg !18
  %address.local619 = getelementptr [2 x i8], ptr addrspace(1) %922, i64 %address.local616, !dbg !18
  %926 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %927 = ptrtoint ptr addrspace(1) %address.local619 to i64, !dbg !18
  %928 = sub i64 %927, %926, !dbg !18
  %dma.buffer.offset = trunc i64 %928 to i32, !dbg !18
  %929 = trunc i64 %926 to i32, !dbg !18
  %930 = insertelement <4 x i32> poison, i32 %929, i64 0, !dbg !18
  %931 = lshr i64 %926, 32, !dbg !18
  %932 = trunc i64 %931 to i32, !dbg !18
  %933 = insertelement <4 x i32> %930, i32 %932, i64 1, !dbg !18
  %934 = insertelement <4 x i32> %933, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %934, i32 131072, i64 3, !dbg !18
  %935 = ptrtoint ptr addrspace(3) %925 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %935), !dbg !18
  %936 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %937 = shufflevector <8 x half> %914, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %938 = extractvalue [2 x <8 x float>] %211, 0, !dbg !35
  %939 = shufflevector <8 x float> %938, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %940 = fmul <4 x float> %939, %669, !dbg !35
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %937, <4 x float> %940) #9, !dbg !15
  %942 = shufflevector <8 x half> %916, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %942, <4 x float> %941) #9, !dbg !15
  %944 = shufflevector <8 x half> %918, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %944, <4 x float> %943) #9, !dbg !15
  %946 = shufflevector <8 x half> %920, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %946, <4 x float> %945) #9, !dbg !15
  %948 = shufflevector <8 x half> %914, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %949 = shufflevector <8 x float> %938, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %950 = fmul <4 x float> %949, %669, !dbg !35
  %951 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %948, <4 x float> %950) #9, !dbg !15
  %952 = shufflevector <8 x half> %916, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %953 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %952, <4 x float> %951) #9, !dbg !15
  %954 = shufflevector <8 x half> %918, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %954, <4 x float> %953) #9, !dbg !15
  %956 = shufflevector <8 x half> %920, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %957 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %956, <4 x float> %955) #9, !dbg !15
  %958 = extractvalue [2 x <8 x float>] %211, 1, !dbg !35
  %959 = shufflevector <8 x float> %958, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %960 = fmul <4 x float> %959, %677, !dbg !35
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %937, <4 x float> %960) #9, !dbg !15
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %942, <4 x float> %961) #9, !dbg !15
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %944, <4 x float> %962) #9, !dbg !15
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %946, <4 x float> %963) #9, !dbg !15
  %965 = shufflevector <8 x float> %958, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %966 = fmul <4 x float> %965, %677, !dbg !35
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %948, <4 x float> %966) #9, !dbg !15
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %952, <4 x float> %967) #9, !dbg !15
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %954, <4 x float> %968) #9, !dbg !15
  %970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %956, <4 x float> %969) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %971 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local594, i32 0) #0, !dbg !15
  %972 = bitcast <4 x i32> %971 to <8 x half>, !dbg !15
  %973 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local598, i32 0) #0, !dbg !15
  %974 = bitcast <4 x i32> %973 to <8 x half>, !dbg !15
  %975 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local602, i32 0) #0, !dbg !15
  %976 = bitcast <4 x i32> %975 to <8 x half>, !dbg !15
  %977 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local606, i32 0) #0, !dbg !15
  %978 = bitcast <4 x i32> %977 to <8 x half>, !dbg !15
  %979 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %980 = extractvalue [2 x <8 x float>] %212, 0, !dbg !35
  %981 = shufflevector <8 x float> %980, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %982 = fmul <4 x float> %981, %669, !dbg !35
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %979, <4 x float> %982) #9, !dbg !15
  %984 = shufflevector <8 x half> %974, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %984, <4 x float> %983) #9, !dbg !15
  %986 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %986, <4 x float> %985) #9, !dbg !15
  %988 = shufflevector <8 x half> %978, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %988, <4 x float> %987) #9, !dbg !15
  %990 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %991 = shufflevector <8 x float> %980, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %992 = fmul <4 x float> %991, %669, !dbg !35
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %990, <4 x float> %992) #9, !dbg !15
  %994 = shufflevector <8 x half> %974, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %994, <4 x float> %993) #9, !dbg !15
  %996 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %996, <4 x float> %995) #9, !dbg !15
  %998 = shufflevector <8 x half> %978, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %998, <4 x float> %997) #9, !dbg !15
  %1000 = extractvalue [2 x <8 x float>] %212, 1, !dbg !35
  %1001 = shufflevector <8 x float> %1000, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1002 = fmul <4 x float> %1001, %677, !dbg !35
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %979, <4 x float> %1002) #9, !dbg !15
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %984, <4 x float> %1003) #9, !dbg !15
  %1005 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %986, <4 x float> %1004) #9, !dbg !15
  %1006 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %988, <4 x float> %1005) #9, !dbg !15
  %1007 = shufflevector <8 x float> %1000, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1008 = fmul <4 x float> %1007, %677, !dbg !35
  %1009 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %990, <4 x float> %1008) #9, !dbg !15
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %994, <4 x float> %1009) #9, !dbg !15
  %1011 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %996, <4 x float> %1010) #9, !dbg !15
  %1012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %998, <4 x float> %1011) #9, !dbg !15
  %1013 = shufflevector <4 x float> %831, <4 x float> %841, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1014 = insertvalue [2 x <8 x float>] poison, <8 x float> %1013, 0, !dbg !7
  %1015 = shufflevector <4 x float> %848, <4 x float> %854, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1016 = insertvalue [2 x <8 x float>] %1014, <8 x float> %1015, 1, !dbg !7
  %1017 = insertvalue [2 x <1 x float>] poison, <1 x float> %641, 0, !dbg !7
  %1018 = insertvalue [2 x <1 x float>] %1017, <1 x float> %654, 1, !dbg !7
  %1019 = shufflevector <4 x float> %889, <4 x float> %899, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1020 = insertvalue [2 x <8 x float>] poison, <8 x float> %1019, 0, !dbg !7
  %1021 = shufflevector <4 x float> %906, <4 x float> %912, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1022 = insertvalue [2 x <8 x float>] %1020, <8 x float> %1021, 1, !dbg !7
  %1023 = shufflevector <4 x float> %947, <4 x float> %957, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1024 = insertvalue [2 x <8 x float>] poison, <8 x float> %1023, 0, !dbg !7
  %1025 = shufflevector <4 x float> %964, <4 x float> %970, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1026 = insertvalue [2 x <8 x float>] %1024, <8 x float> %1025, 1, !dbg !7
  %1027 = shufflevector <4 x float> %989, <4 x float> %999, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1028 = insertvalue [2 x <8 x float>] poison, <8 x float> %1027, 0, !dbg !7
  %1029 = shufflevector <4 x float> %1006, <4 x float> %1012, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1030 = insertvalue [2 x <8 x float>] %1028, <8 x float> %1029, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1031 = add nuw nsw i64 %206, 1, !dbg !7
  br label %205, !dbg !7

1032:                                             ; preds = %205
  %lane.sum.field154 = extractvalue [2 x <1 x float>] %208, 1, !dbg !7
  %lane.sum.local155 = extractelement <1 x float> %lane.sum.field154, i64 0, !dbg !7
  %lane.sum.lane.lo156 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane157 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo156), !dbg !7
  %1033 = shl nuw nsw i32 %lane.sum.lane157, 2, !dbg !7
  %1034 = xor i32 %1033, 64, !dbg !7
  %1035 = bitcast float %lane.sum.local155 to i32, !dbg !7
  %lane.sum.shuffle158 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1034, i32 %1035), !dbg !7
  %1036 = bitcast i32 %lane.sum.shuffle158 to float, !dbg !7
  %lane.sum.final159 = fadd float %lane.sum.local155, %1036, !dbg !7
  %1037 = shl nuw nsw i32 %lane.sum.lane157, 2, !dbg !7
  %1038 = xor i32 %1037, 128, !dbg !7
  %1039 = bitcast float %lane.sum.final159 to i32, !dbg !7
  %lane.sum.shuffle160 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1038, i32 %1039), !dbg !7
  %1040 = bitcast i32 %lane.sum.shuffle160 to float, !dbg !7
  %lane.sum.final161 = fadd float %lane.sum.final159, %1040, !dbg !7
  %1041 = insertelement <1 x float> poison, float %lane.sum.final161, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %208, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %lane.sum.lane.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo), !dbg !7
  %1042 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1043 = xor i32 %1042, 64, !dbg !7
  %1044 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1043, i32 %1044), !dbg !7
  %1045 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1045, !dbg !7
  %1046 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1047 = xor i32 %1046, 128, !dbg !7
  %1048 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle152 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1047, i32 %1048), !dbg !7
  %1049 = bitcast i32 %lane.sum.shuffle152 to float, !dbg !7
  %lane.sum.final153 = fadd float %lane.sum.final, %1049, !dbg !7
  %1050 = insertelement <1 x float> poison, float %lane.sum.final153, i64 0, !dbg !7
  %1051 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %1051, 0, !dbg !7
  %1052 = and i32 %31, 32, !dbg !7
  %.not18 = icmp eq i32 %1052, 0, !dbg !7
  %1053 = extractvalue [2 x <8 x float>] %212, 0, !dbg !7
  %1054 = extractelement <8 x float> %1053, i64 0, !dbg !7
  %1055 = extractelement <8 x float> %1053, i64 1, !dbg !7
  %1056 = extractelement <8 x float> %1053, i64 2, !dbg !7
  %1057 = extractelement <8 x float> %1053, i64 3, !dbg !7
  %1058 = select i1 %.not, float %1055, float %1054, !dbg !7
  %1059 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1060 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1059), !dbg !7
  %1061 = add nuw nsw i32 %1060, 64, !dbg !7
  %1062 = xor i32 %1060, 16, !dbg !7
  %1063 = and i32 %1061, 192, !dbg !7
  %1064 = icmp samesign ult i32 %1062, %1063, !dbg !7
  %1065 = select i1 %1064, i32 %1062, i32 %1060, !dbg !7
  %1066 = shl nuw nsw i32 %1065, 2, !dbg !7
  %1067 = bitcast float %1058 to i32, !dbg !7
  %1068 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1067), !dbg !7
  %1069 = bitcast i32 %1068 to float, !dbg !7
  %1070 = select i1 %.not, float %1054, float %1069, !dbg !7
  %1071 = select i1 %.not, float %1069, float %1055, !dbg !7
  %1072 = select i1 %.not, float %1057, float %1056, !dbg !7
  %1073 = bitcast float %1072 to i32, !dbg !7
  %1074 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1073), !dbg !7
  %1075 = bitcast i32 %1074 to float, !dbg !7
  %1076 = select i1 %.not, float %1056, float %1075, !dbg !7
  %1077 = select i1 %.not, float %1075, float %1057, !dbg !7
  %1078 = select i1 %.not18, float %1076, float %1070, !dbg !7
  %1079 = xor i32 %1060, 32, !dbg !7
  %1080 = icmp samesign ult i32 %1079, %1063, !dbg !7
  %1081 = select i1 %1080, i32 %1079, i32 %1060, !dbg !7
  %1082 = shl nuw nsw i32 %1081, 2, !dbg !7
  %1083 = bitcast float %1078 to i32, !dbg !7
  %1084 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1083), !dbg !7
  %1085 = bitcast i32 %1084 to float, !dbg !7
  %1086 = select i1 %.not18, float %1070, float %1085, !dbg !7
  %1087 = select i1 %.not18, float %1085, float %1076, !dbg !7
  %1088 = select i1 %.not18, float %1077, float %1071, !dbg !7
  %1089 = bitcast float %1088 to i32, !dbg !7
  %1090 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1089), !dbg !7
  %1091 = bitcast i32 %1090 to float, !dbg !7
  %1092 = select i1 %.not18, float %1071, float %1091, !dbg !7
  %1093 = select i1 %.not18, float %1091, float %1077, !dbg !7
  %1094 = extractvalue [2 x <8 x float>] %211, 0, !dbg !7
  %1095 = extractelement <8 x float> %1094, i64 0, !dbg !7
  %1096 = extractelement <8 x float> %1094, i64 1, !dbg !7
  %1097 = extractelement <8 x float> %1094, i64 2, !dbg !7
  %1098 = extractelement <8 x float> %1094, i64 3, !dbg !7
  %1099 = select i1 %.not, float %1096, float %1095, !dbg !7
  %1100 = bitcast float %1099 to i32, !dbg !7
  %1101 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1100), !dbg !7
  %1102 = bitcast i32 %1101 to float, !dbg !7
  %1103 = select i1 %.not, float %1095, float %1102, !dbg !7
  %1104 = select i1 %.not, float %1102, float %1096, !dbg !7
  %1105 = select i1 %.not, float %1098, float %1097, !dbg !7
  %1106 = bitcast float %1105 to i32, !dbg !7
  %1107 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1106), !dbg !7
  %1108 = bitcast i32 %1107 to float, !dbg !7
  %1109 = select i1 %.not, float %1097, float %1108, !dbg !7
  %1110 = select i1 %.not, float %1108, float %1098, !dbg !7
  %1111 = select i1 %.not18, float %1109, float %1103, !dbg !7
  %1112 = bitcast float %1111 to i32, !dbg !7
  %1113 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1112), !dbg !7
  %1114 = bitcast i32 %1113 to float, !dbg !7
  %1115 = select i1 %.not18, float %1103, float %1114, !dbg !7
  %1116 = select i1 %.not18, float %1114, float %1109, !dbg !7
  %1117 = select i1 %.not18, float %1110, float %1104, !dbg !7
  %1118 = bitcast float %1117 to i32, !dbg !7
  %1119 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1118), !dbg !7
  %1120 = bitcast i32 %1119 to float, !dbg !7
  %1121 = select i1 %.not18, float %1104, float %1120, !dbg !7
  %1122 = select i1 %.not18, float %1120, float %1110, !dbg !7
  %1123 = extractvalue [2 x <8 x float>] %210, 0, !dbg !7
  %1124 = extractelement <8 x float> %1123, i64 0, !dbg !7
  %1125 = extractelement <8 x float> %1123, i64 1, !dbg !7
  %1126 = extractelement <8 x float> %1123, i64 2, !dbg !7
  %1127 = extractelement <8 x float> %1123, i64 3, !dbg !7
  %1128 = select i1 %.not, float %1125, float %1124, !dbg !7
  %1129 = bitcast float %1128 to i32, !dbg !7
  %1130 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1129), !dbg !7
  %1131 = bitcast i32 %1130 to float, !dbg !7
  %1132 = select i1 %.not, float %1124, float %1131, !dbg !7
  %1133 = select i1 %.not, float %1131, float %1125, !dbg !7
  %1134 = select i1 %.not, float %1127, float %1126, !dbg !7
  %1135 = bitcast float %1134 to i32, !dbg !7
  %1136 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1135), !dbg !7
  %1137 = bitcast i32 %1136 to float, !dbg !7
  %1138 = select i1 %.not, float %1126, float %1137, !dbg !7
  %1139 = select i1 %.not, float %1137, float %1127, !dbg !7
  %1140 = select i1 %.not18, float %1138, float %1132, !dbg !7
  %1141 = bitcast float %1140 to i32, !dbg !7
  %1142 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1141), !dbg !7
  %1143 = bitcast i32 %1142 to float, !dbg !7
  %1144 = select i1 %.not18, float %1132, float %1143, !dbg !7
  %1145 = select i1 %.not18, float %1143, float %1138, !dbg !7
  %1146 = select i1 %.not18, float %1139, float %1133, !dbg !7
  %1147 = bitcast float %1146 to i32, !dbg !7
  %1148 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1147), !dbg !7
  %1149 = bitcast i32 %1148 to float, !dbg !7
  %1150 = select i1 %.not18, float %1133, float %1149, !dbg !7
  %1151 = select i1 %.not18, float %1149, float %1139, !dbg !7
  %1152 = extractvalue [2 x <8 x float>] %207, 0, !dbg !7
  %1153 = extractelement <8 x float> %1152, i64 0, !dbg !7
  %1154 = extractelement <8 x float> %1152, i64 1, !dbg !7
  %1155 = extractelement <8 x float> %1152, i64 2, !dbg !7
  %1156 = extractelement <8 x float> %1152, i64 3, !dbg !7
  %1157 = select i1 %.not, float %1154, float %1153, !dbg !7
  %1158 = bitcast float %1157 to i32, !dbg !7
  %1159 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1158), !dbg !7
  %1160 = bitcast i32 %1159 to float, !dbg !7
  %1161 = select i1 %.not, float %1153, float %1160, !dbg !7
  %1162 = select i1 %.not, float %1160, float %1154, !dbg !7
  %1163 = select i1 %.not, float %1156, float %1155, !dbg !7
  %1164 = bitcast float %1163 to i32, !dbg !7
  %1165 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1164), !dbg !7
  %1166 = bitcast i32 %1165 to float, !dbg !7
  %1167 = select i1 %.not, float %1155, float %1166, !dbg !7
  %1168 = select i1 %.not, float %1166, float %1156, !dbg !7
  %1169 = select i1 %.not18, float %1167, float %1161, !dbg !7
  %1170 = bitcast float %1169 to i32, !dbg !7
  %1171 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1170), !dbg !7
  %1172 = select i1 %.not18, float %1168, float %1162, !dbg !7
  %1173 = bitcast float %1172 to i32, !dbg !7
  %1174 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1173), !dbg !7
  %1175 = shufflevector <1 x float> %1050, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1176 = extractelement <8 x float> %1053, i64 4, !dbg !7
  %1177 = extractelement <8 x float> %1053, i64 5, !dbg !7
  %1178 = extractelement <8 x float> %1053, i64 6, !dbg !7
  %1179 = extractelement <8 x float> %1053, i64 7, !dbg !7
  %1180 = select i1 %.not, float %1177, float %1176, !dbg !7
  %1181 = bitcast float %1180 to i32, !dbg !7
  %1182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1181), !dbg !7
  %1183 = bitcast i32 %1182 to float, !dbg !7
  %1184 = select i1 %.not, float %1176, float %1183, !dbg !7
  %1185 = select i1 %.not, float %1183, float %1177, !dbg !7
  %1186 = select i1 %.not, float %1179, float %1178, !dbg !7
  %1187 = bitcast float %1186 to i32, !dbg !7
  %1188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1187), !dbg !7
  %1189 = bitcast i32 %1188 to float, !dbg !7
  %1190 = select i1 %.not, float %1178, float %1189, !dbg !7
  %1191 = select i1 %.not, float %1189, float %1179, !dbg !7
  %1192 = select i1 %.not18, float %1190, float %1184, !dbg !7
  %1193 = bitcast float %1192 to i32, !dbg !7
  %1194 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1193), !dbg !7
  %1195 = bitcast i32 %1194 to float, !dbg !7
  %1196 = select i1 %.not18, float %1184, float %1195, !dbg !7
  %1197 = select i1 %.not18, float %1195, float %1190, !dbg !7
  %1198 = select i1 %.not18, float %1191, float %1185, !dbg !7
  %1199 = bitcast float %1198 to i32, !dbg !7
  %1200 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1199), !dbg !7
  %1201 = bitcast i32 %1200 to float, !dbg !7
  %1202 = select i1 %.not18, float %1185, float %1201, !dbg !7
  %1203 = select i1 %.not18, float %1201, float %1191, !dbg !7
  %1204 = extractelement <8 x float> %1094, i64 4, !dbg !7
  %1205 = extractelement <8 x float> %1094, i64 5, !dbg !7
  %1206 = extractelement <8 x float> %1094, i64 6, !dbg !7
  %1207 = extractelement <8 x float> %1094, i64 7, !dbg !7
  %1208 = select i1 %.not, float %1205, float %1204, !dbg !7
  %1209 = bitcast float %1208 to i32, !dbg !7
  %1210 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1209), !dbg !7
  %1211 = bitcast i32 %1210 to float, !dbg !7
  %1212 = select i1 %.not, float %1204, float %1211, !dbg !7
  %1213 = select i1 %.not, float %1211, float %1205, !dbg !7
  %1214 = select i1 %.not, float %1207, float %1206, !dbg !7
  %1215 = bitcast float %1214 to i32, !dbg !7
  %1216 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1215), !dbg !7
  %1217 = bitcast i32 %1216 to float, !dbg !7
  %1218 = select i1 %.not, float %1206, float %1217, !dbg !7
  %1219 = select i1 %.not, float %1217, float %1207, !dbg !7
  %1220 = select i1 %.not18, float %1218, float %1212, !dbg !7
  %1221 = bitcast float %1220 to i32, !dbg !7
  %1222 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1221), !dbg !7
  %1223 = bitcast i32 %1222 to float, !dbg !7
  %1224 = select i1 %.not18, float %1212, float %1223, !dbg !7
  %1225 = select i1 %.not18, float %1223, float %1218, !dbg !7
  %1226 = select i1 %.not18, float %1219, float %1213, !dbg !7
  %1227 = bitcast float %1226 to i32, !dbg !7
  %1228 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1227), !dbg !7
  %1229 = bitcast i32 %1228 to float, !dbg !7
  %1230 = select i1 %.not18, float %1213, float %1229, !dbg !7
  %1231 = select i1 %.not18, float %1229, float %1219, !dbg !7
  %1232 = extractelement <8 x float> %1123, i64 4, !dbg !7
  %1233 = extractelement <8 x float> %1123, i64 5, !dbg !7
  %1234 = extractelement <8 x float> %1123, i64 6, !dbg !7
  %1235 = extractelement <8 x float> %1123, i64 7, !dbg !7
  %1236 = select i1 %.not, float %1233, float %1232, !dbg !7
  %1237 = bitcast float %1236 to i32, !dbg !7
  %1238 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1237), !dbg !7
  %1239 = bitcast i32 %1238 to float, !dbg !7
  %1240 = select i1 %.not, float %1232, float %1239, !dbg !7
  %1241 = select i1 %.not, float %1239, float %1233, !dbg !7
  %1242 = select i1 %.not, float %1235, float %1234, !dbg !7
  %1243 = bitcast float %1242 to i32, !dbg !7
  %1244 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1243), !dbg !7
  %1245 = bitcast i32 %1244 to float, !dbg !7
  %1246 = select i1 %.not, float %1234, float %1245, !dbg !7
  %1247 = select i1 %.not, float %1245, float %1235, !dbg !7
  %1248 = select i1 %.not18, float %1246, float %1240, !dbg !7
  %1249 = bitcast float %1248 to i32, !dbg !7
  %1250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1249), !dbg !7
  %1251 = bitcast i32 %1250 to float, !dbg !7
  %1252 = select i1 %.not18, float %1240, float %1251, !dbg !7
  %1253 = select i1 %.not18, float %1251, float %1246, !dbg !7
  %1254 = select i1 %.not18, float %1247, float %1241, !dbg !7
  %1255 = bitcast float %1254 to i32, !dbg !7
  %1256 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1255), !dbg !7
  %1257 = bitcast i32 %1256 to float, !dbg !7
  %1258 = select i1 %.not18, float %1241, float %1257, !dbg !7
  %1259 = select i1 %.not18, float %1257, float %1247, !dbg !7
  %1260 = extractelement <8 x float> %1152, i64 4, !dbg !7
  %1261 = extractelement <8 x float> %1152, i64 5, !dbg !7
  %1262 = extractelement <8 x float> %1152, i64 6, !dbg !7
  %1263 = extractelement <8 x float> %1152, i64 7, !dbg !7
  %1264 = select i1 %.not, float %1261, float %1260, !dbg !7
  %1265 = bitcast float %1264 to i32, !dbg !7
  %1266 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1265), !dbg !7
  %1267 = bitcast i32 %1266 to float, !dbg !7
  %1268 = select i1 %.not, float %1260, float %1267, !dbg !7
  %1269 = select i1 %.not, float %1267, float %1261, !dbg !7
  %1270 = select i1 %.not, float %1263, float %1262, !dbg !7
  %1271 = bitcast float %1270 to i32, !dbg !7
  %1272 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1271), !dbg !7
  %1273 = bitcast i32 %1272 to float, !dbg !7
  %1274 = select i1 %.not, float %1262, float %1273, !dbg !7
  %1275 = select i1 %.not, float %1273, float %1263, !dbg !7
  %1276 = select i1 %.not18, float %1274, float %1268, !dbg !7
  %1277 = bitcast float %1276 to i32, !dbg !7
  %1278 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1277), !dbg !7
  %1279 = select i1 %.not18, float %1275, float %1269, !dbg !7
  %1280 = bitcast float %1279 to i32, !dbg !7
  %1281 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1280), !dbg !7
  %1282 = extractvalue [2 x <8 x float>] %212, 1, !dbg !7
  %1283 = extractelement <8 x float> %1282, i64 0, !dbg !7
  %1284 = extractelement <8 x float> %1282, i64 1, !dbg !7
  %1285 = extractelement <8 x float> %1282, i64 2, !dbg !7
  %1286 = extractelement <8 x float> %1282, i64 3, !dbg !7
  %1287 = select i1 %.not, float %1284, float %1283, !dbg !7
  %1288 = bitcast float %1287 to i32, !dbg !7
  %1289 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1288), !dbg !7
  %1290 = bitcast i32 %1289 to float, !dbg !7
  %1291 = select i1 %.not, float %1283, float %1290, !dbg !7
  %1292 = select i1 %.not, float %1290, float %1284, !dbg !7
  %1293 = select i1 %.not, float %1286, float %1285, !dbg !7
  %1294 = bitcast float %1293 to i32, !dbg !7
  %1295 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1294), !dbg !7
  %1296 = bitcast i32 %1295 to float, !dbg !7
  %1297 = select i1 %.not, float %1285, float %1296, !dbg !7
  %1298 = select i1 %.not, float %1296, float %1286, !dbg !7
  %1299 = select i1 %.not18, float %1297, float %1291, !dbg !7
  %1300 = bitcast float %1299 to i32, !dbg !7
  %1301 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1300), !dbg !7
  %1302 = bitcast i32 %1301 to float, !dbg !7
  %1303 = select i1 %.not18, float %1291, float %1302, !dbg !7
  %1304 = select i1 %.not18, float %1302, float %1297, !dbg !7
  %1305 = select i1 %.not18, float %1298, float %1292, !dbg !7
  %1306 = bitcast float %1305 to i32, !dbg !7
  %1307 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1306), !dbg !7
  %1308 = bitcast i32 %1307 to float, !dbg !7
  %1309 = select i1 %.not18, float %1292, float %1308, !dbg !7
  %1310 = select i1 %.not18, float %1308, float %1298, !dbg !7
  %1311 = extractvalue [2 x <8 x float>] %211, 1, !dbg !7
  %1312 = extractelement <8 x float> %1311, i64 0, !dbg !7
  %1313 = extractelement <8 x float> %1311, i64 1, !dbg !7
  %1314 = extractelement <8 x float> %1311, i64 2, !dbg !7
  %1315 = extractelement <8 x float> %1311, i64 3, !dbg !7
  %1316 = select i1 %.not, float %1313, float %1312, !dbg !7
  %1317 = bitcast float %1316 to i32, !dbg !7
  %1318 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1317), !dbg !7
  %1319 = bitcast i32 %1318 to float, !dbg !7
  %1320 = select i1 %.not, float %1312, float %1319, !dbg !7
  %1321 = select i1 %.not, float %1319, float %1313, !dbg !7
  %1322 = select i1 %.not, float %1315, float %1314, !dbg !7
  %1323 = bitcast float %1322 to i32, !dbg !7
  %1324 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1323), !dbg !7
  %1325 = bitcast i32 %1324 to float, !dbg !7
  %1326 = select i1 %.not, float %1314, float %1325, !dbg !7
  %1327 = select i1 %.not, float %1325, float %1315, !dbg !7
  %1328 = select i1 %.not18, float %1326, float %1320, !dbg !7
  %1329 = bitcast float %1328 to i32, !dbg !7
  %1330 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1329), !dbg !7
  %1331 = bitcast i32 %1330 to float, !dbg !7
  %1332 = select i1 %.not18, float %1320, float %1331, !dbg !7
  %1333 = select i1 %.not18, float %1331, float %1326, !dbg !7
  %1334 = select i1 %.not18, float %1327, float %1321, !dbg !7
  %1335 = bitcast float %1334 to i32, !dbg !7
  %1336 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1335), !dbg !7
  %1337 = bitcast i32 %1336 to float, !dbg !7
  %1338 = select i1 %.not18, float %1321, float %1337, !dbg !7
  %1339 = select i1 %.not18, float %1337, float %1327, !dbg !7
  %1340 = extractvalue [2 x <8 x float>] %210, 1, !dbg !7
  %1341 = extractelement <8 x float> %1340, i64 0, !dbg !7
  %1342 = extractelement <8 x float> %1340, i64 1, !dbg !7
  %1343 = extractelement <8 x float> %1340, i64 2, !dbg !7
  %1344 = extractelement <8 x float> %1340, i64 3, !dbg !7
  %1345 = select i1 %.not, float %1342, float %1341, !dbg !7
  %1346 = bitcast float %1345 to i32, !dbg !7
  %1347 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1346), !dbg !7
  %1348 = bitcast i32 %1347 to float, !dbg !7
  %1349 = select i1 %.not, float %1341, float %1348, !dbg !7
  %1350 = select i1 %.not, float %1348, float %1342, !dbg !7
  %1351 = select i1 %.not, float %1344, float %1343, !dbg !7
  %1352 = bitcast float %1351 to i32, !dbg !7
  %1353 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1352), !dbg !7
  %1354 = bitcast i32 %1353 to float, !dbg !7
  %1355 = select i1 %.not, float %1343, float %1354, !dbg !7
  %1356 = select i1 %.not, float %1354, float %1344, !dbg !7
  %1357 = select i1 %.not18, float %1355, float %1349, !dbg !7
  %1358 = bitcast float %1357 to i32, !dbg !7
  %1359 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1358), !dbg !7
  %1360 = bitcast i32 %1359 to float, !dbg !7
  %1361 = select i1 %.not18, float %1349, float %1360, !dbg !7
  %1362 = select i1 %.not18, float %1360, float %1355, !dbg !7
  %1363 = select i1 %.not18, float %1356, float %1350, !dbg !7
  %1364 = bitcast float %1363 to i32, !dbg !7
  %1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1364), !dbg !7
  %1366 = bitcast i32 %1365 to float, !dbg !7
  %1367 = select i1 %.not18, float %1350, float %1366, !dbg !7
  %1368 = select i1 %.not18, float %1366, float %1356, !dbg !7
  %1369 = extractvalue [2 x <8 x float>] %207, 1, !dbg !7
  %1370 = extractelement <8 x float> %1369, i64 0, !dbg !7
  %1371 = extractelement <8 x float> %1369, i64 1, !dbg !7
  %1372 = extractelement <8 x float> %1369, i64 2, !dbg !7
  %1373 = extractelement <8 x float> %1369, i64 3, !dbg !7
  %1374 = select i1 %.not, float %1371, float %1370, !dbg !7
  %1375 = bitcast float %1374 to i32, !dbg !7
  %1376 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1375), !dbg !7
  %1377 = bitcast i32 %1376 to float, !dbg !7
  %1378 = select i1 %.not, float %1370, float %1377, !dbg !7
  %1379 = select i1 %.not, float %1377, float %1371, !dbg !7
  %1380 = select i1 %.not, float %1373, float %1372, !dbg !7
  %1381 = bitcast float %1380 to i32, !dbg !7
  %1382 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1381), !dbg !7
  %1383 = bitcast i32 %1382 to float, !dbg !7
  %1384 = select i1 %.not, float %1372, float %1383, !dbg !7
  %1385 = select i1 %.not, float %1383, float %1373, !dbg !7
  %1386 = select i1 %.not18, float %1384, float %1378, !dbg !7
  %1387 = bitcast float %1386 to i32, !dbg !7
  %1388 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1387), !dbg !7
  %1389 = select i1 %.not18, float %1385, float %1379, !dbg !7
  %1390 = bitcast float %1389 to i32, !dbg !7
  %1391 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1390), !dbg !7
  %1392 = shufflevector <1 x float> %1041, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1393 = extractelement <8 x float> %1282, i64 4, !dbg !7
  %1394 = extractelement <8 x float> %1282, i64 5, !dbg !7
  %1395 = extractelement <8 x float> %1282, i64 6, !dbg !7
  %1396 = extractelement <8 x float> %1282, i64 7, !dbg !7
  %1397 = select i1 %.not, float %1394, float %1393, !dbg !7
  %1398 = bitcast float %1397 to i32, !dbg !7
  %1399 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1398), !dbg !7
  %1400 = bitcast i32 %1399 to float, !dbg !7
  %1401 = select i1 %.not, float %1393, float %1400, !dbg !7
  %1402 = select i1 %.not, float %1400, float %1394, !dbg !7
  %1403 = select i1 %.not, float %1396, float %1395, !dbg !7
  %1404 = bitcast float %1403 to i32, !dbg !7
  %1405 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1404), !dbg !7
  %1406 = bitcast i32 %1405 to float, !dbg !7
  %1407 = select i1 %.not, float %1395, float %1406, !dbg !7
  %1408 = select i1 %.not, float %1406, float %1396, !dbg !7
  %1409 = select i1 %.not18, float %1407, float %1401, !dbg !7
  %1410 = bitcast float %1409 to i32, !dbg !7
  %1411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1410), !dbg !7
  %1412 = bitcast i32 %1411 to float, !dbg !7
  %1413 = select i1 %.not18, float %1401, float %1412, !dbg !7
  %1414 = select i1 %.not18, float %1412, float %1407, !dbg !7
  %1415 = select i1 %.not18, float %1408, float %1402, !dbg !7
  %1416 = bitcast float %1415 to i32, !dbg !7
  %1417 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1416), !dbg !7
  %1418 = bitcast i32 %1417 to float, !dbg !7
  %1419 = select i1 %.not18, float %1402, float %1418, !dbg !7
  %1420 = select i1 %.not18, float %1418, float %1408, !dbg !7
  %1421 = extractelement <8 x float> %1311, i64 4, !dbg !7
  %1422 = extractelement <8 x float> %1311, i64 5, !dbg !7
  %1423 = extractelement <8 x float> %1311, i64 6, !dbg !7
  %1424 = extractelement <8 x float> %1311, i64 7, !dbg !7
  %1425 = select i1 %.not, float %1422, float %1421, !dbg !7
  %1426 = bitcast float %1425 to i32, !dbg !7
  %1427 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1426), !dbg !7
  %1428 = bitcast i32 %1427 to float, !dbg !7
  %1429 = select i1 %.not, float %1421, float %1428, !dbg !7
  %1430 = select i1 %.not, float %1428, float %1422, !dbg !7
  %1431 = select i1 %.not, float %1424, float %1423, !dbg !7
  %1432 = bitcast float %1431 to i32, !dbg !7
  %1433 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1432), !dbg !7
  %1434 = bitcast i32 %1433 to float, !dbg !7
  %1435 = select i1 %.not, float %1423, float %1434, !dbg !7
  %1436 = select i1 %.not, float %1434, float %1424, !dbg !7
  %1437 = select i1 %.not18, float %1435, float %1429, !dbg !7
  %1438 = bitcast float %1437 to i32, !dbg !7
  %1439 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1438), !dbg !7
  %1440 = bitcast i32 %1439 to float, !dbg !7
  %1441 = select i1 %.not18, float %1429, float %1440, !dbg !7
  %1442 = select i1 %.not18, float %1440, float %1435, !dbg !7
  %1443 = select i1 %.not18, float %1436, float %1430, !dbg !7
  %1444 = bitcast float %1443 to i32, !dbg !7
  %1445 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1444), !dbg !7
  %1446 = bitcast i32 %1445 to float, !dbg !7
  %1447 = select i1 %.not18, float %1430, float %1446, !dbg !7
  %1448 = select i1 %.not18, float %1446, float %1436, !dbg !7
  %1449 = extractelement <8 x float> %1340, i64 4, !dbg !7
  %1450 = extractelement <8 x float> %1340, i64 5, !dbg !7
  %1451 = extractelement <8 x float> %1340, i64 6, !dbg !7
  %1452 = extractelement <8 x float> %1340, i64 7, !dbg !7
  %1453 = select i1 %.not, float %1450, float %1449, !dbg !7
  %1454 = bitcast float %1453 to i32, !dbg !7
  %1455 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1454), !dbg !7
  %1456 = bitcast i32 %1455 to float, !dbg !7
  %1457 = select i1 %.not, float %1449, float %1456, !dbg !7
  %1458 = select i1 %.not, float %1456, float %1450, !dbg !7
  %1459 = select i1 %.not, float %1452, float %1451, !dbg !7
  %1460 = bitcast float %1459 to i32, !dbg !7
  %1461 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1460), !dbg !7
  %1462 = bitcast i32 %1461 to float, !dbg !7
  %1463 = select i1 %.not, float %1451, float %1462, !dbg !7
  %1464 = select i1 %.not, float %1462, float %1452, !dbg !7
  %1465 = select i1 %.not18, float %1463, float %1457, !dbg !7
  %1466 = bitcast float %1465 to i32, !dbg !7
  %1467 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1466), !dbg !7
  %1468 = bitcast i32 %1467 to float, !dbg !7
  %1469 = select i1 %.not18, float %1457, float %1468, !dbg !7
  %1470 = select i1 %.not18, float %1468, float %1463, !dbg !7
  %1471 = select i1 %.not18, float %1464, float %1458, !dbg !7
  %1472 = bitcast float %1471 to i32, !dbg !7
  %1473 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1472), !dbg !7
  %1474 = bitcast i32 %1473 to float, !dbg !7
  %1475 = select i1 %.not18, float %1458, float %1474, !dbg !7
  %1476 = select i1 %.not18, float %1474, float %1464, !dbg !7
  %1477 = extractelement <8 x float> %1369, i64 4, !dbg !7
  %1478 = extractelement <8 x float> %1369, i64 5, !dbg !7
  %1479 = extractelement <8 x float> %1369, i64 6, !dbg !7
  %1480 = extractelement <8 x float> %1369, i64 7, !dbg !7
  %1481 = select i1 %.not, float %1478, float %1477, !dbg !7
  %1482 = bitcast float %1481 to i32, !dbg !7
  %1483 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1482), !dbg !7
  %1484 = bitcast i32 %1483 to float, !dbg !7
  %1485 = select i1 %.not, float %1477, float %1484, !dbg !7
  %1486 = select i1 %.not, float %1484, float %1478, !dbg !7
  %1487 = select i1 %.not, float %1480, float %1479, !dbg !7
  %1488 = bitcast float %1487 to i32, !dbg !7
  %1489 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1488), !dbg !7
  %1490 = bitcast i32 %1489 to float, !dbg !7
  %1491 = select i1 %.not, float %1479, float %1490, !dbg !7
  %1492 = select i1 %.not, float %1490, float %1480, !dbg !7
  %1493 = select i1 %.not18, float %1491, float %1485, !dbg !7
  %1494 = bitcast float %1493 to i32, !dbg !7
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1494), !dbg !7
  %1496 = select i1 %.not18, float %1492, float %1486, !dbg !7
  %1497 = bitcast float %1496 to i32, !dbg !7
  %1498 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1082, i32 %1497), !dbg !7
  br i1 true, label %1499, label %1563, !dbg !21

1499:                                             ; preds = %1032
  %1500 = bitcast i32 %1495 to float, !dbg !7
  %1501 = select i1 %.not18, float %1485, float %1500, !dbg !7
  %1502 = insertelement <4 x float> poison, float %1501, i64 0, !dbg !19
  %1503 = bitcast i32 %1498 to float, !dbg !7
  %1504 = select i1 %.not18, float %1486, float %1503, !dbg !7
  %1505 = insertelement <4 x float> %1502, float %1504, i64 1, !dbg !19
  %1506 = select i1 %.not18, float %1500, float %1491, !dbg !7
  %1507 = insertelement <4 x float> %1505, float %1506, i64 2, !dbg !19
  %1508 = select i1 %.not18, float %1503, float %1492, !dbg !7
  %1509 = insertelement <4 x float> %1507, float %1508, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1392, !dbg !19
  %normalized = fmul <4 x float> %1509, %row.reciprocal, !dbg !19
  %1510 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1511 = extractelement <4 x half> %1510, i64 3, !dbg !20
  %1512 = extractelement <4 x half> %1510, i64 2, !dbg !20
  %1513 = extractelement <4 x half> %1510, i64 1, !dbg !20
  %1514 = extractelement <4 x half> %1510, i64 0, !dbg !20
  %1515 = bitcast i32 %1388 to float, !dbg !7
  %1516 = select i1 %.not18, float %1378, float %1515, !dbg !7
  %1517 = insertelement <4 x float> poison, float %1516, i64 0, !dbg !19
  %1518 = bitcast i32 %1391 to float, !dbg !7
  %1519 = select i1 %.not18, float %1379, float %1518, !dbg !7
  %1520 = insertelement <4 x float> %1517, float %1519, i64 1, !dbg !19
  %1521 = select i1 %.not18, float %1515, float %1384, !dbg !7
  %1522 = insertelement <4 x float> %1520, float %1521, i64 2, !dbg !19
  %1523 = select i1 %.not18, float %1518, float %1385, !dbg !7
  %1524 = insertelement <4 x float> %1522, float %1523, i64 3, !dbg !19
  %normalized54 = fmul <4 x float> %1524, %row.reciprocal, !dbg !19
  %1525 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !20
  %1526 = extractelement <4 x half> %1525, i64 3, !dbg !20
  %1527 = extractelement <4 x half> %1525, i64 2, !dbg !20
  %1528 = extractelement <4 x half> %1525, i64 1, !dbg !20
  %1529 = extractelement <4 x half> %1525, i64 0, !dbg !20
  %1530 = bitcast i32 %1278 to float, !dbg !7
  %1531 = select i1 %.not18, float %1268, float %1530, !dbg !7
  %1532 = insertelement <4 x float> poison, float %1531, i64 0, !dbg !19
  %1533 = bitcast i32 %1281 to float, !dbg !7
  %1534 = select i1 %.not18, float %1269, float %1533, !dbg !7
  %1535 = insertelement <4 x float> %1532, float %1534, i64 1, !dbg !19
  %1536 = select i1 %.not18, float %1530, float %1274, !dbg !7
  %1537 = insertelement <4 x float> %1535, float %1536, i64 2, !dbg !19
  %1538 = select i1 %.not18, float %1533, float %1275, !dbg !7
  %1539 = insertelement <4 x float> %1537, float %1538, i64 3, !dbg !19
  %row.reciprocal55 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1175, !dbg !19
  %normalized56 = fmul <4 x float> %1539, %row.reciprocal55, !dbg !19
  %1540 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !20
  %1541 = extractelement <4 x half> %1540, i64 3, !dbg !20
  %1542 = extractelement <4 x half> %1540, i64 2, !dbg !20
  %1543 = extractelement <4 x half> %1540, i64 1, !dbg !20
  %1544 = extractelement <4 x half> %1540, i64 0, !dbg !20
  %1545 = bitcast i32 %1171 to float, !dbg !7
  %1546 = select i1 %.not18, float %1161, float %1545, !dbg !7
  %1547 = insertelement <4 x float> poison, float %1546, i64 0, !dbg !19
  %1548 = bitcast i32 %1174 to float, !dbg !7
  %1549 = select i1 %.not18, float %1162, float %1548, !dbg !7
  %1550 = insertelement <4 x float> %1547, float %1549, i64 1, !dbg !19
  %1551 = select i1 %.not18, float %1545, float %1167, !dbg !7
  %1552 = insertelement <4 x float> %1550, float %1551, i64 2, !dbg !19
  %1553 = select i1 %.not18, float %1548, float %1168, !dbg !7
  %1554 = insertelement <4 x float> %1552, float %1553, i64 3, !dbg !19
  %normalized57 = fmul <4 x float> %1554, %row.reciprocal55, !dbg !19
  %1555 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !20
  %1556 = extractelement <4 x half> %1555, i64 3, !dbg !20
  %1557 = extractelement <4 x half> %1555, i64 2, !dbg !20
  %1558 = extractelement <4 x half> %1555, i64 1, !dbg !20
  %1559 = extractelement <4 x half> %1555, i64 0, !dbg !20
  %1560 = add nsw i64 %16, %22, !dbg !21
  %1561 = shl i64 %1560, 19, !dbg !21
  %1562 = and i64 %1561, 16252928, !dbg !21
  %address.thread620 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local621 = zext nneg i32 %address.thread620 to i64, !dbg !9
  %address.local622 = add i64 %25, %address.local621, !dbg !21
  %address.local623 = lshr i64 %address.local621, 6, !dbg !8
  %address.local624 = shl nuw nsw i64 %address.local623, 4, !dbg !21
  %address.local625 = add i64 %address.local622, %address.local624, !dbg !21
  %address.local626 = and i64 %address.local621, 240, !dbg !21
  %address.local627 = sub i64 %address.local625, %address.local626, !dbg !21
  %address.local628 = add i64 %address.local627, %26, !dbg !21
  %address.local629 = shl nuw nsw i64 %address.local628, 7, !dbg !21
  %address.local630 = add nuw nsw i64 %1562, %address.local629, !dbg !21
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local630, !dbg !21
  %address.local632 = lshr i64 %address.local621, 2, !dbg !8
  %address.local633 = and i64 %address.local632, 12, !dbg !8
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local633, !dbg !21
  store half %1559, ptr addrspace(1) %address.local634, align 2, !dbg !21
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local630, !dbg !21
  %address.local636 = or disjoint i64 %address.local633, 1, !dbg !21
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local635, i64 %address.local636, !dbg !21
  store half %1558, ptr addrspace(1) %address.local637, align 2, !dbg !21
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local630, !dbg !21
  %address.local639 = or disjoint i64 %address.local633, 2, !dbg !21
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local638, i64 %address.local639, !dbg !21
  store half %1557, ptr addrspace(1) %address.local640, align 2, !dbg !21
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local630, !dbg !21
  %address.local642 = or disjoint i64 %address.local633, 3, !dbg !21
  %address.local643 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local641, i64 %address.local642, !dbg !21
  store half %1556, ptr addrspace(1) %address.local643, align 2, !dbg !21
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local630, !dbg !21
  %address.local645 = or disjoint i64 %address.local633, 16, !dbg !21
  %address.local646 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local644, i64 %address.local645, !dbg !21
  store half %1544, ptr addrspace(1) %address.local646, align 2, !dbg !21
  %address.local647 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local630, !dbg !21
  %address.local648 = or disjoint i64 %address.local633, 17, !dbg !21
  %address.local649 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local647, i64 %address.local648, !dbg !21
  store half %1543, ptr addrspace(1) %address.local649, align 2, !dbg !21
  %address.local650 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local630, !dbg !21
  %address.local651 = or disjoint i64 %address.local633, 18, !dbg !21
  %address.local652 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local650, i64 %address.local651, !dbg !21
  store half %1542, ptr addrspace(1) %address.local652, align 2, !dbg !21
  %address.local653 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local630, !dbg !21
  %address.local654 = or disjoint i64 %address.local633, 19, !dbg !21
  %address.local655 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local653, i64 %address.local654, !dbg !21
  store half %1541, ptr addrspace(1) %address.local655, align 2, !dbg !21
  %address.local656 = shl i64 %address.local628, 7, !dbg !21
  %address.local657 = add i64 %address.local656, 8192, !dbg !21
  %address.local658 = add nuw nsw i64 %1562, %address.local657, !dbg !21
  %address.local659 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local658, !dbg !21
  %address.local660 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local633, !dbg !21
  store half %1529, ptr addrspace(1) %address.local660, align 2, !dbg !21
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local658, !dbg !21
  %address.local662 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local661, i64 %address.local636, !dbg !21
  store half %1528, ptr addrspace(1) %address.local662, align 2, !dbg !21
  %address.local663 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local658, !dbg !21
  %address.local664 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local663, i64 %address.local639, !dbg !21
  store half %1527, ptr addrspace(1) %address.local664, align 2, !dbg !21
  %address.local665 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local658, !dbg !21
  %address.local666 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local665, i64 %address.local642, !dbg !21
  store half %1526, ptr addrspace(1) %address.local666, align 2, !dbg !21
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local658, !dbg !21
  %address.local668 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local667, i64 %address.local645, !dbg !21
  store half %1514, ptr addrspace(1) %address.local668, align 2, !dbg !21
  %address.local669 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local658, !dbg !21
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local669, i64 %address.local648, !dbg !21
  store half %1513, ptr addrspace(1) %address.local670, align 2, !dbg !21
  %address.local671 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local658, !dbg !21
  %address.local672 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local671, i64 %address.local651, !dbg !21
  store half %1512, ptr addrspace(1) %address.local672, align 2, !dbg !21
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local658, !dbg !21
  %address.local674 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local673, i64 %address.local654, !dbg !21
  store half %1511, ptr addrspace(1) %address.local674, align 2, !dbg !21
  br label %1563, !dbg !21

1563:                                             ; preds = %1499, %1032
  br i1 true, label %1564, label %1602, !dbg !21

1564:                                             ; preds = %1563
  %1565 = insertelement <4 x float> poison, float %1469, i64 0, !dbg !19
  %1566 = insertelement <4 x float> %1565, float %1475, i64 1, !dbg !19
  %1567 = insertelement <4 x float> %1566, float %1470, i64 2, !dbg !19
  %1568 = insertelement <4 x float> %1567, float %1476, i64 3, !dbg !19
  %row.reciprocal58 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1392, !dbg !19
  %normalized59 = fmul <4 x float> %1568, %row.reciprocal58, !dbg !19
  %1569 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !20
  %1570 = extractelement <4 x half> %1569, i64 3, !dbg !20
  %1571 = extractelement <4 x half> %1569, i64 2, !dbg !20
  %1572 = extractelement <4 x half> %1569, i64 1, !dbg !20
  %1573 = extractelement <4 x half> %1569, i64 0, !dbg !20
  %1574 = insertelement <4 x float> poison, float %1361, i64 0, !dbg !19
  %1575 = insertelement <4 x float> %1574, float %1367, i64 1, !dbg !19
  %1576 = insertelement <4 x float> %1575, float %1362, i64 2, !dbg !19
  %1577 = insertelement <4 x float> %1576, float %1368, i64 3, !dbg !19
  %normalized60 = fmul <4 x float> %1577, %row.reciprocal58, !dbg !19
  %1578 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !20
  %1579 = extractelement <4 x half> %1578, i64 3, !dbg !20
  %1580 = extractelement <4 x half> %1578, i64 2, !dbg !20
  %1581 = extractelement <4 x half> %1578, i64 1, !dbg !20
  %1582 = extractelement <4 x half> %1578, i64 0, !dbg !20
  %1583 = insertelement <4 x float> poison, float %1252, i64 0, !dbg !19
  %1584 = insertelement <4 x float> %1583, float %1258, i64 1, !dbg !19
  %1585 = insertelement <4 x float> %1584, float %1253, i64 2, !dbg !19
  %1586 = insertelement <4 x float> %1585, float %1259, i64 3, !dbg !19
  %row.reciprocal61 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1175, !dbg !19
  %normalized62 = fmul <4 x float> %1586, %row.reciprocal61, !dbg !19
  %1587 = fptrunc <4 x float> %normalized62 to <4 x half>, !dbg !20
  %1588 = extractelement <4 x half> %1587, i64 3, !dbg !20
  %1589 = extractelement <4 x half> %1587, i64 2, !dbg !20
  %1590 = extractelement <4 x half> %1587, i64 1, !dbg !20
  %1591 = extractelement <4 x half> %1587, i64 0, !dbg !20
  %1592 = insertelement <4 x float> poison, float %1144, i64 0, !dbg !19
  %1593 = insertelement <4 x float> %1592, float %1150, i64 1, !dbg !19
  %1594 = insertelement <4 x float> %1593, float %1145, i64 2, !dbg !19
  %1595 = insertelement <4 x float> %1594, float %1151, i64 3, !dbg !19
  %normalized63 = fmul <4 x float> %1595, %row.reciprocal61, !dbg !19
  %1596 = fptrunc <4 x float> %normalized63 to <4 x half>, !dbg !20
  %1597 = extractelement <4 x half> %1596, i64 3, !dbg !20
  %1598 = extractelement <4 x half> %1596, i64 2, !dbg !20
  %1599 = extractelement <4 x half> %1596, i64 1, !dbg !20
  %1600 = extractelement <4 x half> %1596, i64 0, !dbg !20
  %1601 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread675 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local676 = zext nneg i32 %address.thread675 to i64, !dbg !9
  %address.local677 = add i64 %25, %address.local676, !dbg !21
  %address.local678 = lshr i64 %address.local676, 6, !dbg !8
  %address.local679 = shl nuw nsw i64 %address.local678, 4, !dbg !21
  %address.local680 = add i64 %address.local677, %address.local679, !dbg !21
  %address.local681 = and i64 %address.local676, 240, !dbg !21
  %address.local682 = sub i64 %address.local680, %address.local681, !dbg !21
  %address.local683 = add i64 %address.local682, %26, !dbg !21
  %address.local684 = shl nuw nsw i64 %address.local683, 7, !dbg !21
  %address.local685 = add nuw nsw i64 %1601, %address.local684, !dbg !21
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local685, !dbg !21
  %address.local687 = lshr i64 %address.local676, 2, !dbg !8
  %address.local688 = and i64 %address.local687, 12, !dbg !8
  %address.local689 = or disjoint i64 %address.local688, 32, !dbg !21
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local686, i64 %address.local689, !dbg !21
  store half %1600, ptr addrspace(1) %address.local690, align 2, !dbg !21
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local685, !dbg !21
  %address.local692 = or disjoint i64 %address.local688, 33, !dbg !21
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local691, i64 %address.local692, !dbg !21
  store half %1599, ptr addrspace(1) %address.local693, align 2, !dbg !21
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local685, !dbg !21
  %address.local695 = or disjoint i64 %address.local688, 34, !dbg !21
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local694, i64 %address.local695, !dbg !21
  store half %1598, ptr addrspace(1) %address.local696, align 2, !dbg !21
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local685, !dbg !21
  %address.local698 = or disjoint i64 %address.local688, 35, !dbg !21
  %address.local699 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local697, i64 %address.local698, !dbg !21
  store half %1597, ptr addrspace(1) %address.local699, align 2, !dbg !21
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local685, !dbg !21
  %address.local701 = or disjoint i64 %address.local688, 48, !dbg !21
  %address.local702 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local700, i64 %address.local701, !dbg !21
  store half %1591, ptr addrspace(1) %address.local702, align 2, !dbg !21
  %address.local703 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local685, !dbg !21
  %address.local704 = or disjoint i64 %address.local688, 49, !dbg !21
  %address.local705 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local703, i64 %address.local704, !dbg !21
  store half %1590, ptr addrspace(1) %address.local705, align 2, !dbg !21
  %address.local706 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local685, !dbg !21
  %address.local707 = or disjoint i64 %address.local688, 50, !dbg !21
  %address.local708 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local706, i64 %address.local707, !dbg !21
  store half %1589, ptr addrspace(1) %address.local708, align 2, !dbg !21
  %address.local709 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local685, !dbg !21
  %address.local710 = or i64 %address.local687, 51, !dbg !21
  %address.local711 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local709, i64 %address.local710, !dbg !21
  store half %1588, ptr addrspace(1) %address.local711, align 2, !dbg !21
  %address.local712 = shl i64 %address.local683, 7, !dbg !21
  %address.local713 = add i64 %address.local712, 8192, !dbg !21
  %address.local714 = add nuw nsw i64 %1601, %address.local713, !dbg !21
  %address.local715 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local714, !dbg !21
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local715, i64 %address.local689, !dbg !21
  store half %1582, ptr addrspace(1) %address.local716, align 2, !dbg !21
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local714, !dbg !21
  %address.local718 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local717, i64 %address.local692, !dbg !21
  store half %1581, ptr addrspace(1) %address.local718, align 2, !dbg !21
  %address.local719 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local714, !dbg !21
  %address.local720 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local719, i64 %address.local695, !dbg !21
  store half %1580, ptr addrspace(1) %address.local720, align 2, !dbg !21
  %address.local721 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local714, !dbg !21
  %address.local722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local721, i64 %address.local698, !dbg !21
  store half %1579, ptr addrspace(1) %address.local722, align 2, !dbg !21
  %address.local723 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local714, !dbg !21
  %address.local724 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local723, i64 %address.local701, !dbg !21
  store half %1573, ptr addrspace(1) %address.local724, align 2, !dbg !21
  %address.local725 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local714, !dbg !21
  %address.local726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local725, i64 %address.local704, !dbg !21
  store half %1572, ptr addrspace(1) %address.local726, align 2, !dbg !21
  %address.local727 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local714, !dbg !21
  %address.local728 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local727, i64 %address.local707, !dbg !21
  store half %1571, ptr addrspace(1) %address.local728, align 2, !dbg !21
  %address.local729 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local714, !dbg !21
  %address.local730 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local729, i64 %address.local710, !dbg !21
  store half %1570, ptr addrspace(1) %address.local730, align 2, !dbg !21
  br label %1602, !dbg !21

1602:                                             ; preds = %1564, %1563
  br i1 true, label %1603, label %1641, !dbg !21

1603:                                             ; preds = %1602
  %1604 = insertelement <4 x float> poison, float %1441, i64 0, !dbg !19
  %1605 = insertelement <4 x float> %1604, float %1447, i64 1, !dbg !19
  %1606 = insertelement <4 x float> %1605, float %1442, i64 2, !dbg !19
  %1607 = insertelement <4 x float> %1606, float %1448, i64 3, !dbg !19
  %row.reciprocal64 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1392, !dbg !19
  %normalized65 = fmul <4 x float> %1607, %row.reciprocal64, !dbg !19
  %1608 = fptrunc <4 x float> %normalized65 to <4 x half>, !dbg !20
  %1609 = extractelement <4 x half> %1608, i64 3, !dbg !20
  %1610 = extractelement <4 x half> %1608, i64 2, !dbg !20
  %1611 = extractelement <4 x half> %1608, i64 1, !dbg !20
  %1612 = extractelement <4 x half> %1608, i64 0, !dbg !20
  %1613 = insertelement <4 x float> poison, float %1332, i64 0, !dbg !19
  %1614 = insertelement <4 x float> %1613, float %1338, i64 1, !dbg !19
  %1615 = insertelement <4 x float> %1614, float %1333, i64 2, !dbg !19
  %1616 = insertelement <4 x float> %1615, float %1339, i64 3, !dbg !19
  %normalized66 = fmul <4 x float> %1616, %row.reciprocal64, !dbg !19
  %1617 = fptrunc <4 x float> %normalized66 to <4 x half>, !dbg !20
  %1618 = extractelement <4 x half> %1617, i64 3, !dbg !20
  %1619 = extractelement <4 x half> %1617, i64 2, !dbg !20
  %1620 = extractelement <4 x half> %1617, i64 1, !dbg !20
  %1621 = extractelement <4 x half> %1617, i64 0, !dbg !20
  %1622 = insertelement <4 x float> poison, float %1224, i64 0, !dbg !19
  %1623 = insertelement <4 x float> %1622, float %1230, i64 1, !dbg !19
  %1624 = insertelement <4 x float> %1623, float %1225, i64 2, !dbg !19
  %1625 = insertelement <4 x float> %1624, float %1231, i64 3, !dbg !19
  %row.reciprocal67 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1175, !dbg !19
  %normalized68 = fmul <4 x float> %1625, %row.reciprocal67, !dbg !19
  %1626 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !20
  %1627 = extractelement <4 x half> %1626, i64 3, !dbg !20
  %1628 = extractelement <4 x half> %1626, i64 2, !dbg !20
  %1629 = extractelement <4 x half> %1626, i64 1, !dbg !20
  %1630 = extractelement <4 x half> %1626, i64 0, !dbg !20
  %1631 = insertelement <4 x float> poison, float %1115, i64 0, !dbg !19
  %1632 = insertelement <4 x float> %1631, float %1121, i64 1, !dbg !19
  %1633 = insertelement <4 x float> %1632, float %1116, i64 2, !dbg !19
  %1634 = insertelement <4 x float> %1633, float %1122, i64 3, !dbg !19
  %normalized69 = fmul <4 x float> %1634, %row.reciprocal67, !dbg !19
  %1635 = fptrunc <4 x float> %normalized69 to <4 x half>, !dbg !20
  %1636 = extractelement <4 x half> %1635, i64 3, !dbg !20
  %1637 = extractelement <4 x half> %1635, i64 2, !dbg !20
  %1638 = extractelement <4 x half> %1635, i64 1, !dbg !20
  %1639 = extractelement <4 x half> %1635, i64 0, !dbg !20
  %1640 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread731 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local732 = zext nneg i32 %address.thread731 to i64, !dbg !9
  %address.local733 = add i64 %25, %address.local732, !dbg !21
  %address.local734 = lshr i64 %address.local732, 6, !dbg !8
  %address.local735 = shl nuw nsw i64 %address.local734, 4, !dbg !21
  %address.local736 = add i64 %address.local733, %address.local735, !dbg !21
  %address.local737 = and i64 %address.local732, 240, !dbg !21
  %address.local738 = sub i64 %address.local736, %address.local737, !dbg !21
  %address.local739 = add i64 %address.local738, %26, !dbg !21
  %address.local740 = shl nuw nsw i64 %address.local739, 7, !dbg !21
  %address.local741 = add nuw nsw i64 %1640, %address.local740, !dbg !21
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local741, !dbg !21
  %address.local743 = lshr i64 %address.local732, 2, !dbg !8
  %address.local744 = and i64 %address.local743, 12, !dbg !8
  %address.local745 = or disjoint i64 %address.local744, 64, !dbg !21
  %address.local746 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local742, i64 %address.local745, !dbg !21
  store half %1639, ptr addrspace(1) %address.local746, align 2, !dbg !21
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local741, !dbg !21
  %address.local748 = or disjoint i64 %address.local744, 65, !dbg !21
  %address.local749 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local747, i64 %address.local748, !dbg !21
  store half %1638, ptr addrspace(1) %address.local749, align 2, !dbg !21
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local741, !dbg !21
  %address.local751 = or disjoint i64 %address.local744, 66, !dbg !21
  %address.local752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local750, i64 %address.local751, !dbg !21
  store half %1637, ptr addrspace(1) %address.local752, align 2, !dbg !21
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local741, !dbg !21
  %address.local754 = or disjoint i64 %address.local744, 67, !dbg !21
  %address.local755 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local753, i64 %address.local754, !dbg !21
  store half %1636, ptr addrspace(1) %address.local755, align 2, !dbg !21
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local741, !dbg !21
  %address.local757 = or disjoint i64 %address.local744, 80, !dbg !21
  %address.local758 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local756, i64 %address.local757, !dbg !21
  store half %1630, ptr addrspace(1) %address.local758, align 2, !dbg !21
  %address.local759 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local741, !dbg !21
  %address.local760 = or disjoint i64 %address.local744, 81, !dbg !21
  %address.local761 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local759, i64 %address.local760, !dbg !21
  store half %1629, ptr addrspace(1) %address.local761, align 2, !dbg !21
  %address.local762 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local741, !dbg !21
  %address.local763 = or disjoint i64 %address.local744, 82, !dbg !21
  %address.local764 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local762, i64 %address.local763, !dbg !21
  store half %1628, ptr addrspace(1) %address.local764, align 2, !dbg !21
  %address.local765 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local741, !dbg !21
  %address.local766 = or disjoint i64 %address.local744, 83, !dbg !21
  %address.local767 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local765, i64 %address.local766, !dbg !21
  store half %1627, ptr addrspace(1) %address.local767, align 2, !dbg !21
  %address.local768 = shl i64 %address.local739, 7, !dbg !21
  %address.local769 = add i64 %address.local768, 8192, !dbg !21
  %address.local770 = add nuw nsw i64 %1640, %address.local769, !dbg !21
  %address.local771 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local770, !dbg !21
  %address.local772 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local771, i64 %address.local745, !dbg !21
  store half %1621, ptr addrspace(1) %address.local772, align 2, !dbg !21
  %address.local773 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local770, !dbg !21
  %address.local774 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local773, i64 %address.local748, !dbg !21
  store half %1620, ptr addrspace(1) %address.local774, align 2, !dbg !21
  %address.local775 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local770, !dbg !21
  %address.local776 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local775, i64 %address.local751, !dbg !21
  store half %1619, ptr addrspace(1) %address.local776, align 2, !dbg !21
  %address.local777 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local770, !dbg !21
  %address.local778 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local777, i64 %address.local754, !dbg !21
  store half %1618, ptr addrspace(1) %address.local778, align 2, !dbg !21
  %address.local779 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local770, !dbg !21
  %address.local780 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local779, i64 %address.local757, !dbg !21
  store half %1612, ptr addrspace(1) %address.local780, align 2, !dbg !21
  %address.local781 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local770, !dbg !21
  %address.local782 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local781, i64 %address.local760, !dbg !21
  store half %1611, ptr addrspace(1) %address.local782, align 2, !dbg !21
  %address.local783 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local770, !dbg !21
  %address.local784 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local783, i64 %address.local763, !dbg !21
  store half %1610, ptr addrspace(1) %address.local784, align 2, !dbg !21
  %address.local785 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local770, !dbg !21
  %address.local786 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local785, i64 %address.local766, !dbg !21
  store half %1609, ptr addrspace(1) %address.local786, align 2, !dbg !21
  br label %1641, !dbg !21

1641:                                             ; preds = %1603, %1602
  br i1 true, label %1642, label %1680, !dbg !21

1642:                                             ; preds = %1641
  %1643 = insertelement <4 x float> poison, float %1413, i64 0, !dbg !19
  %1644 = insertelement <4 x float> %1643, float %1419, i64 1, !dbg !19
  %1645 = insertelement <4 x float> %1644, float %1414, i64 2, !dbg !19
  %1646 = insertelement <4 x float> %1645, float %1420, i64 3, !dbg !19
  %row.reciprocal70 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1392, !dbg !19
  %normalized71 = fmul <4 x float> %1646, %row.reciprocal70, !dbg !19
  %1647 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !20
  %1648 = extractelement <4 x half> %1647, i64 3, !dbg !20
  %1649 = extractelement <4 x half> %1647, i64 2, !dbg !20
  %1650 = extractelement <4 x half> %1647, i64 1, !dbg !20
  %1651 = extractelement <4 x half> %1647, i64 0, !dbg !20
  %1652 = insertelement <4 x float> poison, float %1303, i64 0, !dbg !19
  %1653 = insertelement <4 x float> %1652, float %1309, i64 1, !dbg !19
  %1654 = insertelement <4 x float> %1653, float %1304, i64 2, !dbg !19
  %1655 = insertelement <4 x float> %1654, float %1310, i64 3, !dbg !19
  %normalized72 = fmul <4 x float> %1655, %row.reciprocal70, !dbg !19
  %1656 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !20
  %1657 = extractelement <4 x half> %1656, i64 3, !dbg !20
  %1658 = extractelement <4 x half> %1656, i64 2, !dbg !20
  %1659 = extractelement <4 x half> %1656, i64 1, !dbg !20
  %1660 = extractelement <4 x half> %1656, i64 0, !dbg !20
  %1661 = insertelement <4 x float> poison, float %1196, i64 0, !dbg !19
  %1662 = insertelement <4 x float> %1661, float %1202, i64 1, !dbg !19
  %1663 = insertelement <4 x float> %1662, float %1197, i64 2, !dbg !19
  %1664 = insertelement <4 x float> %1663, float %1203, i64 3, !dbg !19
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1175, !dbg !19
  %normalized74 = fmul <4 x float> %1664, %row.reciprocal73, !dbg !19
  %1665 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %1666 = extractelement <4 x half> %1665, i64 3, !dbg !20
  %1667 = extractelement <4 x half> %1665, i64 2, !dbg !20
  %1668 = extractelement <4 x half> %1665, i64 1, !dbg !20
  %1669 = extractelement <4 x half> %1665, i64 0, !dbg !20
  %1670 = insertelement <4 x float> poison, float %1086, i64 0, !dbg !19
  %1671 = insertelement <4 x float> %1670, float %1092, i64 1, !dbg !19
  %1672 = insertelement <4 x float> %1671, float %1087, i64 2, !dbg !19
  %1673 = insertelement <4 x float> %1672, float %1093, i64 3, !dbg !19
  %normalized75 = fmul <4 x float> %1673, %row.reciprocal73, !dbg !19
  %1674 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !20
  %1675 = extractelement <4 x half> %1674, i64 3, !dbg !20
  %1676 = extractelement <4 x half> %1674, i64 2, !dbg !20
  %1677 = extractelement <4 x half> %1674, i64 1, !dbg !20
  %1678 = extractelement <4 x half> %1674, i64 0, !dbg !20
  %1679 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread787 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local788 = zext nneg i32 %address.thread787 to i64, !dbg !9
  %address.local789 = add i64 %25, %address.local788, !dbg !21
  %address.local790 = lshr i64 %address.local788, 6, !dbg !8
  %address.local791 = shl nuw nsw i64 %address.local790, 4, !dbg !21
  %address.local792 = add i64 %address.local789, %address.local791, !dbg !21
  %address.local793 = and i64 %address.local788, 240, !dbg !21
  %address.local794 = sub i64 %address.local792, %address.local793, !dbg !21
  %address.local795 = add i64 %address.local794, %26, !dbg !21
  %address.local796 = shl nuw nsw i64 %address.local795, 7, !dbg !21
  %address.local797 = add nuw nsw i64 %1679, %address.local796, !dbg !21
  %address.local798 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local797, !dbg !21
  %address.local799 = lshr i64 %address.local788, 2, !dbg !8
  %address.local800 = and i64 %address.local799, 12, !dbg !8
  %address.local801 = or disjoint i64 %address.local800, 96, !dbg !21
  %address.local802 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local798, i64 %address.local801, !dbg !21
  store half %1678, ptr addrspace(1) %address.local802, align 2, !dbg !21
  %address.local803 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local797, !dbg !21
  %address.local804 = or disjoint i64 %address.local800, 97, !dbg !21
  %address.local805 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local803, i64 %address.local804, !dbg !21
  store half %1677, ptr addrspace(1) %address.local805, align 2, !dbg !21
  %address.local806 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local797, !dbg !21
  %address.local807 = or disjoint i64 %address.local800, 98, !dbg !21
  %address.local808 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local806, i64 %address.local807, !dbg !21
  store half %1676, ptr addrspace(1) %address.local808, align 2, !dbg !21
  %address.local809 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local797, !dbg !21
  %address.local810 = or disjoint i64 %address.local800, 99, !dbg !21
  %address.local811 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local809, i64 %address.local810, !dbg !21
  store half %1675, ptr addrspace(1) %address.local811, align 2, !dbg !21
  %address.local812 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local797, !dbg !21
  %address.local813 = or disjoint i64 %address.local800, 112, !dbg !21
  %address.local814 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local812, i64 %address.local813, !dbg !21
  store half %1669, ptr addrspace(1) %address.local814, align 2, !dbg !21
  %address.local815 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local797, !dbg !21
  %address.local816 = or disjoint i64 %address.local800, 113, !dbg !21
  %address.local817 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local815, i64 %address.local816, !dbg !21
  store half %1668, ptr addrspace(1) %address.local817, align 2, !dbg !21
  %address.local818 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local797, !dbg !21
  %address.local819 = or disjoint i64 %address.local800, 114, !dbg !21
  %address.local820 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local818, i64 %address.local819, !dbg !21
  store half %1667, ptr addrspace(1) %address.local820, align 2, !dbg !21
  %address.local821 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local797, !dbg !21
  %address.local822 = or i64 %address.local799, 115, !dbg !21
  %address.local823 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local821, i64 %address.local822, !dbg !21
  store half %1666, ptr addrspace(1) %address.local823, align 2, !dbg !21
  %address.local824 = shl i64 %address.local795, 7, !dbg !21
  %address.local825 = add i64 %address.local824, 8192, !dbg !21
  %address.local826 = add nuw nsw i64 %1679, %address.local825, !dbg !21
  %address.local827 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local826, !dbg !21
  %address.local828 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local827, i64 %address.local801, !dbg !21
  store half %1660, ptr addrspace(1) %address.local828, align 2, !dbg !21
  %address.local829 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local826, !dbg !21
  %address.local830 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local829, i64 %address.local804, !dbg !21
  store half %1659, ptr addrspace(1) %address.local830, align 2, !dbg !21
  %address.local831 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local826, !dbg !21
  %address.local832 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local831, i64 %address.local807, !dbg !21
  store half %1658, ptr addrspace(1) %address.local832, align 2, !dbg !21
  %address.local833 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local826, !dbg !21
  %address.local834 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local833, i64 %address.local810, !dbg !21
  store half %1657, ptr addrspace(1) %address.local834, align 2, !dbg !21
  %address.local835 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local826, !dbg !21
  %address.local836 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local835, i64 %address.local813, !dbg !21
  store half %1651, ptr addrspace(1) %address.local836, align 2, !dbg !21
  %address.local837 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local826, !dbg !21
  %address.local838 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local837, i64 %address.local816, !dbg !21
  store half %1650, ptr addrspace(1) %address.local838, align 2, !dbg !21
  %address.local839 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local826, !dbg !21
  %address.local840 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local839, i64 %address.local819, !dbg !21
  store half %1649, ptr addrspace(1) %address.local840, align 2, !dbg !21
  %address.local841 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local826, !dbg !21
  %address.local842 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local841, i64 %address.local822, !dbg !21
  store half %1648, ptr addrspace(1) %address.local842, align 2, !dbg !21
  br label %1680, !dbg !21

1680:                                             ; preds = %1642, %1641
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

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/deferred_lane_sum_pass/compiler_final/run-20261026-155755-g7dykgmh/input-0002-test_input_1x32x4096x128-128-64-32-online/grid-000003")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 72, column: 24, scope: !4)
!9 = !DILocation(line: 70, column: 22, scope: !4)
!11 = !DILocation(line: 48, column: 17, scope: !4)
!14 = !DILocation(line: 77, column: 19, scope: !4)
!15 = !DILocation(line: 102, column: 23, scope: !4)
!16 = !DILocation(line: 71, column: 22, scope: !4)
!17 = !DILocation(line: 73, column: 26, scope: !4)
!18 = !DILocation(line: 101, column: 20, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
!23 = !DILocation(line: 79, column: 24, scope: !4)
!24 = !DILocation(line: 80, column: 23, scope: !4)
!25 = !DILocation(line: 81, column: 26, scope: !4)
!26 = !DILocation(line: 88, column: 21, scope: !4)
!27 = !DILocation(line: 89, column: 20, scope: !4)
!28 = !DILocation(line: 90, column: 22, scope: !4)
!29 = !DILocation(line: 91, column: 23, scope: !4)
!30 = !DILocation(line: 92, column: 20, scope: !4)
!31 = !DILocation(line: 93, column: 22, scope: !4)
!32 = !DILocation(line: 94, column: 22, scope: !4)
!33 = !DILocation(line: 95, column: 20, scope: !4)
!34 = !DILocation(line: 96, column: 16, scope: !4)
!35 = !DILocation(line: 103, column: 22, scope: !4)
!37 = !{i32 256, i32 1, i32 1}
!38 = !{i32 2, !"Debug Info Version", i32 3}
!39 = !{i32 0, i32 256}
