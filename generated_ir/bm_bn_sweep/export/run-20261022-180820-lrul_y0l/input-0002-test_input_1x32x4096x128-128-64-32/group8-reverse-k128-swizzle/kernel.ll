; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/bm_bn_sweep/export/run-20261022-180820-lrul_y0l/input-0002-test_input_1x32x4096x128-128-64-32/group8-reverse-k128-swizzle/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/bm_bn_sweep/export/run-20261022-180820-lrul_y0l/input-0002-test_input_1x32x4096x128-128-64-32/group8-reverse-k128-swizzle/input.mlir"

@shm_0 = addrspace(3) global [12288 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !31 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 5, !dbg !5
  %10 = add nsw i64 %9, %6, !dbg !5
  %11 = lshr i64 %10, 5, !dbg !5
  %12 = and i64 %11, 576460752303423480, !dbg !5
  %13 = and i64 %6, 7, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = lshr i64 %10, 3, !dbg !5
  %16 = sub nsw i64 31, %14, !dbg !5
  %17 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %18 = sext i32 %17 to i64, !dbg !5
  %19 = shl i64 %16, 7, !dbg !9
  %20 = icmp ugt i64 %10, 1023, !dbg !11
  %21 = add nsw i64 %14, -32, !dbg !11
  %22 = select i1 %20, i64 %21, i64 %16, !dbg !11
  %23 = sdiv i64 %22, 32, !dbg !11
  %24 = sext i1 %20 to i64, !dbg !11
  %25 = xor i64 %23, %24, !dbg !11
  %26 = add nsw i64 %15, %25, !dbg !11
  %27 = and i64 %26, 31, !dbg !11
  %28 = shl nsw i64 %16, 7, !dbg !11
  %29 = mul nsw i64 %25, -4096, !dbg !11
  %30 = add i64 %28, %29, !dbg !11
  %31 = shl nuw nsw i64 %27, 19, !dbg !11
  %32 = shl nsw i64 %30, 7, !dbg !11
  %33 = add i64 %31, %32, !dbg !11
  %34 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !33
  %35 = zext nneg i32 %34 to i64, !dbg !12
  %36 = lshr i64 %35, 6, !dbg !12
  %37 = or disjoint i64 %19, 127, !dbg !13
  %38 = add i64 %19, 128, !dbg !13
  %39 = icmp uge i64 %38, %37, !dbg !13
  %40 = shl nsw i64 %16, 1, !dbg !15
  %41 = add nsw i64 %40, 2, !dbg !15
  %42 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %43 = lshr i64 %35, 2, !dbg !17
  %44 = shl nuw nsw i64 %35, 2, !dbg !17
  %45 = mul nsw i64 %43, -16, !dbg !17
  %46 = add nsw i64 %44, %45, !dbg !17
  %47 = and i32 %34, 1, !dbg !17
  %index.compact142 = zext nneg i32 %47 to i64, !dbg !17
  %48 = or disjoint i64 %46, %index.compact142, !dbg !17
  %49 = add nuw nsw i64 %48, 1, !dbg !17
  %50 = sdiv i64 %49, 2, !dbg !17
  %51 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %52 = getelementptr i8, ptr addrspace(1) %51, i64 64, !dbg !11
  %53 = shl i64 %15, 7, !dbg !16
  %54 = and i64 %53, 3968, !dbg !16
  %55 = shl nsw i64 %18, 12, !dbg !16
  %56 = or disjoint i64 %54, %55, !dbg !16
  %57 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %58 = getelementptr i8, ptr addrspace(1) %57, i64 128, !dbg !11
  %59 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %60 = getelementptr i8, ptr addrspace(1) %59, i64 192, !dbg !11
  br label %61, !dbg !15

61:                                               ; preds = %586, %4
  %62 = phi i64 [ %997, %586 ], [ 0, %4 ], !dbg !15
  %63 = phi [2 x <8 x float>] [ %984, %586 ], [ zeroinitializer, %4 ], !dbg !15
  %64 = phi [2 x <1 x float>] [ %776, %586 ], [ zeroinitializer, %4 ], !dbg !15
  %65 = phi [2 x <8 x float>] [ %988, %586 ], [ zeroinitializer, %4 ], !dbg !15
  %66 = phi [2 x <8 x float>] [ %992, %586 ], [ zeroinitializer, %4 ], !dbg !15
  %67 = phi [2 x <8 x float>] [ %996, %586 ], [ zeroinitializer, %4 ], !dbg !15
  %68 = icmp slt i64 %62, %41, !dbg !15
  br i1 %68, label %69, label %998, !dbg !15

69:                                               ; preds = %61
  %70 = shl nsw i64 %62, 6, !dbg !15
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !33, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local173 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local174 = and i64 %address.local173, 1536, !dbg !11
  %address.local175 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local174, !dbg !11
  %address.local176 = ptrtoint ptr addrspace(3) %address.local175 to i64, !dbg !11
  %address.local177 = trunc i64 %address.local176 to i32, !dbg !11
  %71 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local177), !dbg !11
  %72 = zext i32 %71 to i64, !dbg !11
  %73 = inttoptr i64 %72 to ptr addrspace(3), !dbg !11
  %address.local178 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local179 = and i64 %address.local178, 16256, !dbg !11
  %address.local180 = lshr i64 %address.local, 4, !dbg !11
  %address.local181 = and i64 %address.local180, 8, !dbg !11
  %address.local182 = or disjoint i64 %address.local179, %address.local181, !dbg !11
  %address.local183 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %address.local182, !dbg !11
  %74 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %75 = ptrtoint ptr addrspace(1) %address.local183 to i64, !dbg !11
  %76 = sub i64 %75, %74, !dbg !11
  %dma.buffer.offset926 = trunc i64 %76 to i32, !dbg !11
  %77 = trunc i64 %74 to i32, !dbg !11
  %78 = insertelement <4 x i32> poison, i32 %77, i64 0, !dbg !11
  %79 = lshr i64 %74, 32, !dbg !11
  %80 = trunc i64 %79 to i32, !dbg !11
  %81 = insertelement <4 x i32> %78, i32 %80, i64 1, !dbg !11
  %82 = insertelement <4 x i32> %81, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc927 = insertelement <4 x i32> %82, i32 131072, i64 3, !dbg !11
  %83 = ptrtoint ptr addrspace(3) %73 to i32, !dbg !11
  %dma.buffer.lds928 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %83), !dbg !11
  %84 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds928, i32 %dma.buffer.offset926, <4 x i32> %dma.buffer.rsrc927) #10, !dbg !11
  %address.local184 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local185 = shl nuw nsw i64 %address.local184, 3, !dbg !11
  %address.local186 = and i64 %address.local185, 3584, !dbg !11
  %address.local187 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local186, !dbg !11
  %address.local188 = ptrtoint ptr addrspace(3) %address.local187 to i64, !dbg !11
  %address.local189 = trunc i64 %address.local188 to i32, !dbg !11
  %85 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local189), !dbg !11
  %86 = zext i32 %85 to i64, !dbg !11
  %87 = inttoptr i64 %86 to ptr addrspace(3), !dbg !11
  %address.local190 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local191 = and i64 %address.local190, 16256, !dbg !11
  %address.local192 = lshr i64 %address.local184, 4, !dbg !11
  %address.local193 = and i64 %address.local192, 24, !dbg !11
  %address.local194 = or disjoint i64 %address.local191, %address.local193, !dbg !11
  %address.local195 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %address.local194, !dbg !11
  %88 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %89 = ptrtoint ptr addrspace(1) %address.local195 to i64, !dbg !11
  %90 = sub i64 %89, %88, !dbg !11
  %dma.buffer.offset923 = trunc i64 %90 to i32, !dbg !11
  %91 = trunc i64 %88 to i32, !dbg !11
  %92 = insertelement <4 x i32> poison, i32 %91, i64 0, !dbg !11
  %93 = lshr i64 %88, 32, !dbg !11
  %94 = trunc i64 %93 to i32, !dbg !11
  %95 = insertelement <4 x i32> %92, i32 %94, i64 1, !dbg !11
  %96 = insertelement <4 x i32> %95, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc924 = insertelement <4 x i32> %96, i32 131072, i64 3, !dbg !11
  %97 = ptrtoint ptr addrspace(3) %87 to i32, !dbg !11
  %dma.buffer.lds925 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %97), !dbg !11
  %98 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds925, i32 %dma.buffer.offset923, <4 x i32> %dma.buffer.rsrc924) #10, !dbg !11
  %99 = lshr i64 %62, 13, !dbg !16
  %100 = add nuw nsw i64 %99, %15, !dbg !16
  %101 = lshr i64 %62, 6, !dbg !16
  %102 = and i64 %101, 127, !dbg !16
  %103 = shl i64 %100, 19, !dbg !16
  %104 = and i64 %103, 16252928, !dbg !16
  %105 = shl i64 %101, 19, !dbg !16
  %106 = shl i64 %62, 13, !dbg !16
  %107 = sub i64 %106, %105, !dbg !16
  %108 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %104, !dbg !16
  %109 = getelementptr [2 x i8], ptr addrspace(1) %108, i64 %102, !dbg !16
  %110 = getelementptr [2 x i8], ptr addrspace(1) %109, i64 %107, !dbg !16
  %address.local196 = and i64 %address.local173, 1536, !dbg !16
  %address.local197 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local196, !dbg !16
  %address.local198 = ptrtoint ptr addrspace(3) %address.local197 to i64, !dbg !16
  %address.local199 = trunc i64 %address.local198 to i32, !dbg !16
  %111 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local199), !dbg !16
  %112 = zext i32 %111 to i64, !dbg !16
  %113 = inttoptr i64 %112 to ptr addrspace(3), !dbg !16
  %address.local200 = lshr i64 %address.local, 3, !dbg !16
  %address.local201 = and i64 %address.local200, 24, !dbg !16
  %address.local202 = and i64 %address.local, 59, !dbg !16
  %address.local203 = lshr i64 %address.local, 2, !dbg !16
  %address.local204 = and i64 %address.local203, 4, !dbg !16
  %address.local205 = add nuw nsw i64 %address.local204, %address.local, !dbg !16
  %address.local206 = and i64 %address.local205, 4, !dbg !16
  %address.local207 = or disjoint i64 %address.local202, %address.local206, !dbg !16
  %address.local208 = shl nuw nsw i64 %address.local207, 7, !dbg !16
  %address.local209 = or disjoint i64 %address.local201, %address.local208, !dbg !16
  %address.local210 = getelementptr [2 x i8], ptr addrspace(1) %110, i64 %address.local209, !dbg !16
  %114 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %115 = ptrtoint ptr addrspace(1) %address.local210 to i64, !dbg !16
  %116 = sub i64 %115, %114, !dbg !16
  %dma.buffer.offset920 = trunc i64 %116 to i32, !dbg !16
  %117 = trunc i64 %114 to i32, !dbg !16
  %118 = insertelement <4 x i32> poison, i32 %117, i64 0, !dbg !16
  %119 = lshr i64 %114, 32, !dbg !16
  %120 = trunc i64 %119 to i32, !dbg !16
  %121 = insertelement <4 x i32> %118, i32 %120, i64 1, !dbg !16
  %122 = insertelement <4 x i32> %121, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc921 = insertelement <4 x i32> %122, i32 131072, i64 3, !dbg !16
  %123 = ptrtoint ptr addrspace(3) %113 to i32, !dbg !16
  %dma.buffer.lds922 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %123), !dbg !16
  %124 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds922, i32 %dma.buffer.offset920, <4 x i32> %dma.buffer.rsrc921) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local211 = lshr i32 %address.thread, 2, !dbg !12
  %address.local212 = and i32 %address.local211, 4, !dbg !12
  %address.local213 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local214 = and i32 %address.local213, 384, !dbg !12
  %address.local215 = or disjoint i32 %address.local212, %address.local214, !dbg !12
  %address.local216 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local217 = and i32 %address.local216, 120, !dbg !12
  %address.local218 = or disjoint i32 %address.local215, %address.local217, !dbg !12
  %address.local219 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local220 = and i32 %address.local219, 1024, !dbg !12
  %address.local221 = or disjoint i32 %address.local218, %address.local220, !dbg !12
  %address.local222 = zext nneg i32 %address.local221 to i64, !dbg !12
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local222, !dbg !12
  %125 = load <4 x half>, ptr addrspace(3) %address.local223, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local223, i64 4096, !dbg !12
  %126 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %lds.const.gep825 = getelementptr i8, ptr addrspace(3) %address.local223, i64 1024, !dbg !12
  %127 = load <4 x half>, ptr addrspace(3) %lds.const.gep825, align 8, !dbg !12
  %128 = fmul <4 x half> %125, splat (half 0xH3015), !dbg !12
  %lds.const.gep826 = getelementptr i8, ptr addrspace(3) %address.local223, i64 5120, !dbg !12
  %129 = load <4 x half>, ptr addrspace(3) %lds.const.gep826, align 8, !dbg !12
  %130 = fmul <4 x half> %126, splat (half 0xH3015), !dbg !12
  %131 = fmul <4 x half> %127, splat (half 0xH3015), !dbg !12
  %132 = fmul <4 x half> %129, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local230 = mul nuw nsw i64 %address.local, 160, !dbg !17
  %address.local231 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local230, !dbg !17
  %address.local232 = lshr i64 %address.local, 2, !dbg !17
  %address.local233 = mul nsw i64 %address.local232, -1280, !dbg !17
  %address.local234 = getelementptr i8, ptr addrspace(3) %address.local231, i64 %address.local233, !dbg !17
  %address.local235 = shl nuw nsw i64 %address.local, 1, !dbg !17
  %address.local236 = and i64 %address.local235, 24, !dbg !17
  %address.local237 = getelementptr [2 x i8], ptr addrspace(3) %address.local234, i64 %address.local236, !dbg !17
  %address.local238 = shl i64 %address.local, 62, !dbg !17
  %address.local239 = ashr exact i64 %address.local238, 57, !dbg !17
  %address.local240 = and i64 %address.local239, -64, !dbg !17
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local237, i64 %address.local240, !dbg !17
  %address.local242 = lshr i64 %address.local, 2, !dbg !12
  %address.local243 = and i64 %address.local242, 12, !dbg !12
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local243, !dbg !17
  %address.local245 = and i64 %address.local, 63, !dbg !12
  %address.local246 = lshr i64 %address.local245, 5, !dbg !12
  %address.local247 = mul nuw nsw i64 %address.local246, 504, !dbg !17
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local244, i64 %address.local247, !dbg !17
  %133 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !17
  %134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %128, <4 x half> %133, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep827 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !17
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep827, align 8, !dbg !17
  %address.local259 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local260 = and i32 %address.local259, 24, !dbg !17
  %address.local261 = shl nuw nsw i32 %address.thread, 8, !dbg !17
  %address.local262 = and i32 %address.local261, 768, !dbg !17
  %address.local263 = or disjoint i32 %address.local260, %address.local262, !dbg !17
  %address.local264 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local265 = and i32 %address.local264, 96, !dbg !17
  %address.local266 = or disjoint i32 %address.local263, %address.local265, !dbg !17
  %address.local267 = zext nneg i32 %address.local266 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %50) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local268 = lshr i32 %address.thread, 1, !dbg !17
  %address.local269 = and i32 %address.local268, 1, !dbg !17
  %address.local270 = zext nneg i32 %address.local269 to i64, !dbg !17
  %address.local271 = add nsw i64 %address.base, %address.local270, !dbg !17
  %address.local272 = shl i64 %address.local271, 6, !dbg !17
  %address.local273 = sub i64 %address.local267, %address.local272, !dbg !17
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local273, !dbg !17
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) %address.local274, i64 %address.local243, !dbg !17
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) %address.local275, i64 %address.local247, !dbg !17
  %address.local277 = getelementptr i8, ptr addrspace(3) %address.local276, i64 64, !dbg !17
  %136 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !17
  %137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %135, <4 x float> %134) #9, !dbg !17
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %128, <4 x half> %136, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep828 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !17
  %139 = load <4 x half>, ptr addrspace(3) %lds.const.gep828, align 8, !dbg !17
  %lds.const.gep829 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !17
  %140 = load <4 x half>, ptr addrspace(3) %lds.const.gep829, align 8, !dbg !17
  %141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %139, <4 x float> %138) #9, !dbg !17
  %142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %128, <4 x half> %140, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep830 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !17
  %143 = load <4 x half>, ptr addrspace(3) %lds.const.gep830, align 8, !dbg !17
  %lds.const.gep831 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !17
  %144 = load <4 x half>, ptr addrspace(3) %lds.const.gep831, align 8, !dbg !17
  %145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %143, <4 x float> %142) #9, !dbg !17
  %146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %128, <4 x half> %144, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep832 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !17
  %147 = load <4 x half>, ptr addrspace(3) %lds.const.gep832, align 8, !dbg !17
  %148 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !17
  %149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %130, <4 x half> %147, <4 x float> %146) #9, !dbg !17
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %148, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep833 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !17
  %151 = load <4 x half>, ptr addrspace(3) %lds.const.gep833, align 8, !dbg !17
  %152 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !17
  %153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %151, <4 x float> %150) #9, !dbg !17
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %152, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep834 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !17
  %155 = load <4 x half>, ptr addrspace(3) %lds.const.gep834, align 8, !dbg !17
  %lds.const.gep835 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !17
  %156 = load <4 x half>, ptr addrspace(3) %lds.const.gep835, align 8, !dbg !17
  %157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %155, <4 x float> %154) #9, !dbg !17
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %156, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep836 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !17
  %159 = load <4 x half>, ptr addrspace(3) %lds.const.gep836, align 8, !dbg !17
  %lds.const.gep837 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !17
  %160 = load <4 x half>, ptr addrspace(3) %lds.const.gep837, align 8, !dbg !17
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %159, <4 x float> %158) #9, !dbg !17
  %lds.const.gep838 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !17
  %162 = load <4 x half>, ptr addrspace(3) %lds.const.gep838, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %160, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %162, <4 x float> %163) #9, !dbg !17
  %165 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local177), !dbg !11
  %166 = zext i32 %165 to i64, !dbg !11
  %167 = inttoptr i64 %166 to ptr addrspace(3), !dbg !11
  %address.local418 = getelementptr [2 x i8], ptr addrspace(1) %52, i64 %address.local182, !dbg !11
  %168 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %169 = ptrtoint ptr addrspace(1) %address.local418 to i64, !dbg !11
  %170 = sub i64 %169, %168, !dbg !11
  %dma.buffer.offset917 = trunc i64 %170 to i32, !dbg !11
  %171 = trunc i64 %168 to i32, !dbg !11
  %172 = insertelement <4 x i32> poison, i32 %171, i64 0, !dbg !11
  %173 = lshr i64 %168, 32, !dbg !11
  %174 = trunc i64 %173 to i32, !dbg !11
  %175 = insertelement <4 x i32> %172, i32 %174, i64 1, !dbg !11
  %176 = insertelement <4 x i32> %175, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc918 = insertelement <4 x i32> %176, i32 131072, i64 3, !dbg !11
  %177 = ptrtoint ptr addrspace(3) %167 to i32, !dbg !11
  %dma.buffer.lds919 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %177), !dbg !11
  %178 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds919, i32 %dma.buffer.offset917, <4 x i32> %dma.buffer.rsrc918) #10, !dbg !11
  %179 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local189), !dbg !11
  %180 = zext i32 %179 to i64, !dbg !11
  %181 = inttoptr i64 %180 to ptr addrspace(3), !dbg !11
  %address.local419 = getelementptr [2 x i8], ptr addrspace(1) %52, i64 %address.local194, !dbg !11
  %182 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %183 = ptrtoint ptr addrspace(1) %address.local419 to i64, !dbg !11
  %184 = sub i64 %183, %182, !dbg !11
  %dma.buffer.offset914 = trunc i64 %184 to i32, !dbg !11
  %185 = trunc i64 %182 to i32, !dbg !11
  %186 = insertelement <4 x i32> poison, i32 %185, i64 0, !dbg !11
  %187 = lshr i64 %182, 32, !dbg !11
  %188 = trunc i64 %187 to i32, !dbg !11
  %189 = insertelement <4 x i32> %186, i32 %188, i64 1, !dbg !11
  %190 = insertelement <4 x i32> %189, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc915 = insertelement <4 x i32> %190, i32 131072, i64 3, !dbg !11
  %191 = ptrtoint ptr addrspace(3) %181 to i32, !dbg !11
  %dma.buffer.lds916 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %191), !dbg !11
  %192 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds916, i32 %dma.buffer.offset914, <4 x i32> %dma.buffer.rsrc915) #10, !dbg !11
  %193 = add nuw i64 %62, 2048, !dbg !16
  %194 = icmp slt i64 %193, 0, !dbg !16
  %195 = sub nuw i64 -2049, %62, !dbg !16
  %196 = select i1 %194, i64 %195, i64 %193, !dbg !16
  %197 = sdiv i64 %196, 8192, !dbg !16
  %.lobit19 = ashr i64 %193, 63, !dbg !16
  %198 = xor i64 %197, %.lobit19, !dbg !16
  %199 = add nsw i64 %198, %15, !dbg !16
  %200 = add nsw i64 %101, %56, !dbg !16
  %201 = add nsw i64 %200, 32, !dbg !16
  %202 = icmp slt i64 %200, -32, !dbg !16
  %203 = sub nsw i64 -33, %200, !dbg !16
  %204 = select i1 %202, i64 %203, i64 %201, !dbg !16
  %205 = sdiv i64 %204, 128, !dbg !16
  %206 = sext i1 %202 to i64, !dbg !16
  %207 = xor i64 %205, %206, !dbg !16
  %208 = shl nsw i64 %207, 7, !dbg !16
  %209 = sub i64 %56, %208, !dbg !16
  %210 = shl i64 %199, 19, !dbg !16
  %211 = and i64 %210, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %212 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %211, !dbg !16
  %213 = getelementptr [2 x i8], ptr addrspace(1) %212, i64 %101, !dbg !16
  %214 = getelementptr [2 x i8], ptr addrspace(1) %213, i64 %209, !dbg !16
  %215 = getelementptr i8, ptr addrspace(1) %214, i64 64, !dbg !16
  %216 = getelementptr [2 x i8], ptr addrspace(1) %215, i64 %107, !dbg !16
  %217 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local199), !dbg !16
  %218 = zext i32 %217 to i64, !dbg !16
  %219 = inttoptr i64 %218 to ptr addrspace(3), !dbg !16
  %address.local420 = getelementptr [2 x i8], ptr addrspace(1) %216, i64 %address.local209, !dbg !16
  %220 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %221 = ptrtoint ptr addrspace(1) %address.local420 to i64, !dbg !16
  %222 = sub i64 %221, %220, !dbg !16
  %dma.buffer.offset911 = trunc i64 %222 to i32, !dbg !16
  %223 = trunc i64 %220 to i32, !dbg !16
  %224 = insertelement <4 x i32> poison, i32 %223, i64 0, !dbg !16
  %225 = lshr i64 %220, 32, !dbg !16
  %226 = trunc i64 %225 to i32, !dbg !16
  %227 = insertelement <4 x i32> %224, i32 %226, i64 1, !dbg !16
  %228 = insertelement <4 x i32> %227, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc912 = insertelement <4 x i32> %228, i32 131072, i64 3, !dbg !16
  %229 = ptrtoint ptr addrspace(3) %219 to i32, !dbg !16
  %dma.buffer.lds913 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %229), !dbg !16
  %230 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds913, i32 %dma.buffer.offset911, <4 x i32> %dma.buffer.rsrc912) #10, !dbg !16
  %231 = load <4 x half>, ptr addrspace(3) %address.local223, align 8, !dbg !12
  %lds.const.gep839 = getelementptr i8, ptr addrspace(3) %address.local223, i64 4096, !dbg !12
  %232 = load <4 x half>, ptr addrspace(3) %lds.const.gep839, align 8, !dbg !12
  %lds.const.gep840 = getelementptr i8, ptr addrspace(3) %address.local223, i64 1024, !dbg !12
  %233 = load <4 x half>, ptr addrspace(3) %lds.const.gep840, align 8, !dbg !12
  %234 = fmul <4 x half> %231, splat (half 0xH3015), !dbg !12
  %lds.const.gep841 = getelementptr i8, ptr addrspace(3) %address.local223, i64 5120, !dbg !12
  %235 = load <4 x half>, ptr addrspace(3) %lds.const.gep841, align 8, !dbg !12
  %236 = fmul <4 x half> %232, splat (half 0xH3015), !dbg !12
  %237 = fmul <4 x half> %233, splat (half 0xH3015), !dbg !12
  %238 = fmul <4 x half> %235, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %239 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !17
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %239, <4 x float> %137) #9, !dbg !17
  %lds.const.gep842 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !17
  %241 = load <4 x half>, ptr addrspace(3) %lds.const.gep842, align 8, !dbg !17
  %242 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !17
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %236, <4 x half> %241, <4 x float> %240) #9, !dbg !17
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %242, <4 x float> %141) #9, !dbg !17
  %lds.const.gep843 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !17
  %245 = load <4 x half>, ptr addrspace(3) %lds.const.gep843, align 8, !dbg !17
  %lds.const.gep844 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !17
  %246 = load <4 x half>, ptr addrspace(3) %lds.const.gep844, align 8, !dbg !17
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %236, <4 x half> %245, <4 x float> %244) #9, !dbg !17
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %246, <4 x float> %145) #9, !dbg !17
  %lds.const.gep845 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !17
  %249 = load <4 x half>, ptr addrspace(3) %lds.const.gep845, align 8, !dbg !17
  %lds.const.gep846 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !17
  %250 = load <4 x half>, ptr addrspace(3) %lds.const.gep846, align 8, !dbg !17
  %251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %236, <4 x half> %249, <4 x float> %248) #9, !dbg !17
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %250, <4 x float> %149) #9, !dbg !17
  %lds.const.gep847 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !17
  %253 = load <4 x half>, ptr addrspace(3) %lds.const.gep847, align 8, !dbg !17
  %254 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !17
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %236, <4 x half> %253, <4 x float> %252) #9, !dbg !17
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %254, <4 x float> %153) #9, !dbg !17
  %lds.const.gep848 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !17
  %257 = load <4 x half>, ptr addrspace(3) %lds.const.gep848, align 8, !dbg !17
  %258 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !17
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %257, <4 x float> %256) #9, !dbg !17
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %258, <4 x float> %157) #9, !dbg !17
  %lds.const.gep849 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !17
  %261 = load <4 x half>, ptr addrspace(3) %lds.const.gep849, align 8, !dbg !17
  %lds.const.gep850 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !17
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep850, align 8, !dbg !17
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %261, <4 x float> %260) #9, !dbg !17
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %262, <4 x float> %161) #9, !dbg !17
  %lds.const.gep851 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !17
  %265 = load <4 x half>, ptr addrspace(3) %lds.const.gep851, align 8, !dbg !17
  %lds.const.gep852 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !17
  %266 = load <4 x half>, ptr addrspace(3) %lds.const.gep852, align 8, !dbg !17
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %265, <4 x float> %264) #9, !dbg !17
  %lds.const.gep853 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !17
  %268 = load <4 x half>, ptr addrspace(3) %lds.const.gep853, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %266, <4 x float> %164) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %268, <4 x float> %269) #9, !dbg !17
  %271 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local177), !dbg !11
  %272 = zext i32 %271 to i64, !dbg !11
  %273 = inttoptr i64 %272 to ptr addrspace(3), !dbg !11
  %address.local421 = getelementptr [2 x i8], ptr addrspace(1) %58, i64 %address.local182, !dbg !11
  %274 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %275 = ptrtoint ptr addrspace(1) %address.local421 to i64, !dbg !11
  %276 = sub i64 %275, %274, !dbg !11
  %dma.buffer.offset908 = trunc i64 %276 to i32, !dbg !11
  %277 = trunc i64 %274 to i32, !dbg !11
  %278 = insertelement <4 x i32> poison, i32 %277, i64 0, !dbg !11
  %279 = lshr i64 %274, 32, !dbg !11
  %280 = trunc i64 %279 to i32, !dbg !11
  %281 = insertelement <4 x i32> %278, i32 %280, i64 1, !dbg !11
  %282 = insertelement <4 x i32> %281, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc909 = insertelement <4 x i32> %282, i32 131072, i64 3, !dbg !11
  %283 = ptrtoint ptr addrspace(3) %273 to i32, !dbg !11
  %dma.buffer.lds910 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %283), !dbg !11
  %284 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds910, i32 %dma.buffer.offset908, <4 x i32> %dma.buffer.rsrc909) #10, !dbg !11
  %285 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local189), !dbg !11
  %286 = zext i32 %285 to i64, !dbg !11
  %287 = inttoptr i64 %286 to ptr addrspace(3), !dbg !11
  %address.local422 = getelementptr [2 x i8], ptr addrspace(1) %58, i64 %address.local194, !dbg !11
  %288 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %289 = ptrtoint ptr addrspace(1) %address.local422 to i64, !dbg !11
  %290 = sub i64 %289, %288, !dbg !11
  %dma.buffer.offset905 = trunc i64 %290 to i32, !dbg !11
  %291 = trunc i64 %288 to i32, !dbg !11
  %292 = insertelement <4 x i32> poison, i32 %291, i64 0, !dbg !11
  %293 = lshr i64 %288, 32, !dbg !11
  %294 = trunc i64 %293 to i32, !dbg !11
  %295 = insertelement <4 x i32> %292, i32 %294, i64 1, !dbg !11
  %296 = insertelement <4 x i32> %295, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc906 = insertelement <4 x i32> %296, i32 131072, i64 3, !dbg !11
  %297 = ptrtoint ptr addrspace(3) %287 to i32, !dbg !11
  %dma.buffer.lds907 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %297), !dbg !11
  %298 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds907, i32 %dma.buffer.offset905, <4 x i32> %dma.buffer.rsrc906) #10, !dbg !11
  %299 = add nuw i64 %62, 4096, !dbg !16
  %300 = icmp slt i64 %299, 0, !dbg !16
  %301 = sub nuw i64 -4097, %62, !dbg !16
  %302 = select i1 %300, i64 %301, i64 %299, !dbg !16
  %303 = sdiv i64 %302, 8192, !dbg !16
  %.lobit20 = ashr i64 %299, 63, !dbg !16
  %304 = xor i64 %303, %.lobit20, !dbg !16
  %305 = add nsw i64 %304, %15, !dbg !16
  %306 = add nsw i64 %200, 64, !dbg !16
  %307 = icmp slt i64 %200, -64, !dbg !16
  %308 = sub nsw i64 -65, %200, !dbg !16
  %309 = select i1 %307, i64 %308, i64 %306, !dbg !16
  %310 = sdiv i64 %309, 128, !dbg !16
  %311 = sext i1 %307 to i64, !dbg !16
  %312 = xor i64 %310, %311, !dbg !16
  %313 = shl nsw i64 %312, 7, !dbg !16
  %314 = sub i64 %56, %313, !dbg !16
  %315 = shl i64 %305, 19, !dbg !16
  %316 = and i64 %315, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %317 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %316, !dbg !16
  %318 = getelementptr [2 x i8], ptr addrspace(1) %317, i64 %101, !dbg !16
  %319 = getelementptr [2 x i8], ptr addrspace(1) %318, i64 %314, !dbg !16
  %320 = getelementptr i8, ptr addrspace(1) %319, i64 128, !dbg !16
  %321 = getelementptr [2 x i8], ptr addrspace(1) %320, i64 %107, !dbg !16
  %322 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local199), !dbg !16
  %323 = zext i32 %322 to i64, !dbg !16
  %324 = inttoptr i64 %323 to ptr addrspace(3), !dbg !16
  %address.local423 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local209, !dbg !16
  %325 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %326 = ptrtoint ptr addrspace(1) %address.local423 to i64, !dbg !16
  %327 = sub i64 %326, %325, !dbg !16
  %dma.buffer.offset902 = trunc i64 %327 to i32, !dbg !16
  %328 = trunc i64 %325 to i32, !dbg !16
  %329 = insertelement <4 x i32> poison, i32 %328, i64 0, !dbg !16
  %330 = lshr i64 %325, 32, !dbg !16
  %331 = trunc i64 %330 to i32, !dbg !16
  %332 = insertelement <4 x i32> %329, i32 %331, i64 1, !dbg !16
  %333 = insertelement <4 x i32> %332, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc903 = insertelement <4 x i32> %333, i32 131072, i64 3, !dbg !16
  %334 = ptrtoint ptr addrspace(3) %324 to i32, !dbg !16
  %dma.buffer.lds904 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %334), !dbg !16
  %335 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds904, i32 %dma.buffer.offset902, <4 x i32> %dma.buffer.rsrc903) #10, !dbg !16
  %336 = load <4 x half>, ptr addrspace(3) %address.local223, align 8, !dbg !12
  %lds.const.gep854 = getelementptr i8, ptr addrspace(3) %address.local223, i64 4096, !dbg !12
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep854, align 8, !dbg !12
  %lds.const.gep855 = getelementptr i8, ptr addrspace(3) %address.local223, i64 1024, !dbg !12
  %338 = load <4 x half>, ptr addrspace(3) %lds.const.gep855, align 8, !dbg !12
  %339 = fmul <4 x half> %336, splat (half 0xH3015), !dbg !12
  %lds.const.gep856 = getelementptr i8, ptr addrspace(3) %address.local223, i64 5120, !dbg !12
  %340 = load <4 x half>, ptr addrspace(3) %lds.const.gep856, align 8, !dbg !12
  %341 = fmul <4 x half> %337, splat (half 0xH3015), !dbg !12
  %342 = fmul <4 x half> %338, splat (half 0xH3015), !dbg !12
  %343 = fmul <4 x half> %340, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %344 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !17
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %339, <4 x half> %344, <4 x float> %243) #9, !dbg !17
  %lds.const.gep857 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !17
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep857, align 8, !dbg !17
  %347 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !17
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %341, <4 x half> %346, <4 x float> %345) #9, !dbg !17
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %339, <4 x half> %347, <4 x float> %247) #9, !dbg !17
  %lds.const.gep858 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !17
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep858, align 8, !dbg !17
  %lds.const.gep859 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !17
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep859, align 8, !dbg !17
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %341, <4 x half> %350, <4 x float> %349) #9, !dbg !17
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %339, <4 x half> %351, <4 x float> %251) #9, !dbg !17
  %lds.const.gep860 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !17
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep860, align 8, !dbg !17
  %lds.const.gep861 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !17
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep861, align 8, !dbg !17
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %341, <4 x half> %354, <4 x float> %353) #9, !dbg !17
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %339, <4 x half> %355, <4 x float> %255) #9, !dbg !17
  %lds.const.gep862 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !17
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep862, align 8, !dbg !17
  %359 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !17
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %341, <4 x half> %358, <4 x float> %357) #9, !dbg !17
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %359, <4 x float> %259) #9, !dbg !17
  %lds.const.gep863 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !17
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep863, align 8, !dbg !17
  %363 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !17
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %362, <4 x float> %361) #9, !dbg !17
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %363, <4 x float> %263) #9, !dbg !17
  %lds.const.gep864 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !17
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep864, align 8, !dbg !17
  %lds.const.gep865 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !17
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep865, align 8, !dbg !17
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %366, <4 x float> %365) #9, !dbg !17
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %367, <4 x float> %267) #9, !dbg !17
  %lds.const.gep866 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !17
  %370 = load <4 x half>, ptr addrspace(3) %lds.const.gep866, align 8, !dbg !17
  %lds.const.gep867 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !17
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep867, align 8, !dbg !17
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %370, <4 x float> %369) #9, !dbg !17
  %lds.const.gep868 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !17
  %373 = load <4 x half>, ptr addrspace(3) %lds.const.gep868, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %371, <4 x float> %270) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %373, <4 x float> %374) #9, !dbg !17
  %376 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local177), !dbg !11
  %377 = zext i32 %376 to i64, !dbg !11
  %378 = inttoptr i64 %377 to ptr addrspace(3), !dbg !11
  %address.local424 = getelementptr [2 x i8], ptr addrspace(1) %60, i64 %address.local182, !dbg !11
  %379 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %380 = ptrtoint ptr addrspace(1) %address.local424 to i64, !dbg !11
  %381 = sub i64 %380, %379, !dbg !11
  %dma.buffer.offset899 = trunc i64 %381 to i32, !dbg !11
  %382 = trunc i64 %379 to i32, !dbg !11
  %383 = insertelement <4 x i32> poison, i32 %382, i64 0, !dbg !11
  %384 = lshr i64 %379, 32, !dbg !11
  %385 = trunc i64 %384 to i32, !dbg !11
  %386 = insertelement <4 x i32> %383, i32 %385, i64 1, !dbg !11
  %387 = insertelement <4 x i32> %386, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc900 = insertelement <4 x i32> %387, i32 131072, i64 3, !dbg !11
  %388 = ptrtoint ptr addrspace(3) %378 to i32, !dbg !11
  %dma.buffer.lds901 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %388), !dbg !11
  %389 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds901, i32 %dma.buffer.offset899, <4 x i32> %dma.buffer.rsrc900) #10, !dbg !11
  %390 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local189), !dbg !11
  %391 = zext i32 %390 to i64, !dbg !11
  %392 = inttoptr i64 %391 to ptr addrspace(3), !dbg !11
  %address.local425 = getelementptr [2 x i8], ptr addrspace(1) %60, i64 %address.local194, !dbg !11
  %393 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %394 = ptrtoint ptr addrspace(1) %address.local425 to i64, !dbg !11
  %395 = sub i64 %394, %393, !dbg !11
  %dma.buffer.offset896 = trunc i64 %395 to i32, !dbg !11
  %396 = trunc i64 %393 to i32, !dbg !11
  %397 = insertelement <4 x i32> poison, i32 %396, i64 0, !dbg !11
  %398 = lshr i64 %393, 32, !dbg !11
  %399 = trunc i64 %398 to i32, !dbg !11
  %400 = insertelement <4 x i32> %397, i32 %399, i64 1, !dbg !11
  %401 = insertelement <4 x i32> %400, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc897 = insertelement <4 x i32> %401, i32 131072, i64 3, !dbg !11
  %402 = ptrtoint ptr addrspace(3) %392 to i32, !dbg !11
  %dma.buffer.lds898 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %402), !dbg !11
  %403 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds898, i32 %dma.buffer.offset896, <4 x i32> %dma.buffer.rsrc897) #10, !dbg !11
  %404 = add nuw i64 %62, 6144, !dbg !16
  %405 = icmp slt i64 %404, 0, !dbg !16
  %406 = sub nuw i64 -6145, %62, !dbg !16
  %407 = select i1 %405, i64 %406, i64 %404, !dbg !16
  %408 = sdiv i64 %407, 8192, !dbg !16
  %.lobit21 = ashr i64 %404, 63, !dbg !16
  %409 = xor i64 %408, %.lobit21, !dbg !16
  %410 = add nsw i64 %409, %15, !dbg !16
  %411 = add nsw i64 %200, 96, !dbg !16
  %412 = icmp slt i64 %200, -96, !dbg !16
  %413 = sub nsw i64 -97, %200, !dbg !16
  %414 = select i1 %412, i64 %413, i64 %411, !dbg !16
  %415 = sdiv i64 %414, 128, !dbg !16
  %416 = sext i1 %412 to i64, !dbg !16
  %417 = xor i64 %415, %416, !dbg !16
  %418 = shl nsw i64 %417, 7, !dbg !16
  %419 = sub i64 %56, %418, !dbg !16
  %420 = shl i64 %410, 19, !dbg !16
  %421 = and i64 %420, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %422 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %421, !dbg !16
  %423 = getelementptr [2 x i8], ptr addrspace(1) %422, i64 %101, !dbg !16
  %424 = getelementptr [2 x i8], ptr addrspace(1) %423, i64 %419, !dbg !16
  %425 = getelementptr i8, ptr addrspace(1) %424, i64 192, !dbg !16
  %426 = getelementptr [2 x i8], ptr addrspace(1) %425, i64 %107, !dbg !16
  %427 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local199), !dbg !16
  %428 = zext i32 %427 to i64, !dbg !16
  %429 = inttoptr i64 %428 to ptr addrspace(3), !dbg !16
  %address.local426 = getelementptr [2 x i8], ptr addrspace(1) %426, i64 %address.local209, !dbg !16
  %430 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %431 = ptrtoint ptr addrspace(1) %address.local426 to i64, !dbg !16
  %432 = sub i64 %431, %430, !dbg !16
  %dma.buffer.offset893 = trunc i64 %432 to i32, !dbg !16
  %433 = trunc i64 %430 to i32, !dbg !16
  %434 = insertelement <4 x i32> poison, i32 %433, i64 0, !dbg !16
  %435 = lshr i64 %430, 32, !dbg !16
  %436 = trunc i64 %435 to i32, !dbg !16
  %437 = insertelement <4 x i32> %434, i32 %436, i64 1, !dbg !16
  %438 = insertelement <4 x i32> %437, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc894 = insertelement <4 x i32> %438, i32 131072, i64 3, !dbg !16
  %439 = ptrtoint ptr addrspace(3) %429 to i32, !dbg !16
  %dma.buffer.lds895 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %439), !dbg !16
  %440 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds895, i32 %dma.buffer.offset893, <4 x i32> %dma.buffer.rsrc894) #10, !dbg !16
  %441 = load <4 x half>, ptr addrspace(3) %address.local223, align 8, !dbg !12
  %lds.const.gep869 = getelementptr i8, ptr addrspace(3) %address.local223, i64 4096, !dbg !12
  %442 = load <4 x half>, ptr addrspace(3) %lds.const.gep869, align 8, !dbg !12
  %lds.const.gep870 = getelementptr i8, ptr addrspace(3) %address.local223, i64 1024, !dbg !12
  %443 = load <4 x half>, ptr addrspace(3) %lds.const.gep870, align 8, !dbg !12
  %444 = fmul <4 x half> %441, splat (half 0xH3015), !dbg !12
  %lds.const.gep871 = getelementptr i8, ptr addrspace(3) %address.local223, i64 5120, !dbg !12
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep871, align 8, !dbg !12
  %446 = fmul <4 x half> %442, splat (half 0xH3015), !dbg !12
  %447 = fmul <4 x half> %443, splat (half 0xH3015), !dbg !12
  %448 = fmul <4 x half> %445, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %449 = add nuw nsw i64 %101, %15, !dbg !18
  %450 = shl i64 %449, 19, !dbg !18
  %451 = and i64 %450, 16252928, !dbg !18
  %452 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %451, !dbg !18
  %453 = getelementptr [2 x i8], ptr addrspace(1) %452, i64 %107, !dbg !18
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local196, !dbg !18
  %address.local428 = ptrtoint ptr addrspace(3) %address.local427 to i64, !dbg !18
  %address.local429 = trunc i64 %address.local428 to i32, !dbg !18
  %454 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !18
  %455 = zext i32 %454 to i64, !dbg !18
  %456 = inttoptr i64 %455 to ptr addrspace(3), !dbg !18
  %address.local430 = shl nuw nsw i64 %address.local, 5, !dbg !18
  %address.local431 = and i64 %address.local430, 8064, !dbg !18
  %address.local432 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local433 = and i64 %address.local432, 24, !dbg !18
  %address.local434 = or disjoint i64 %address.local431, %address.local433, !dbg !18
  %address.local435 = getelementptr [2 x i8], ptr addrspace(1) %453, i64 %address.local434, !dbg !18
  %457 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %458 = ptrtoint ptr addrspace(1) %address.local435 to i64, !dbg !18
  %459 = sub i64 %458, %457, !dbg !18
  %dma.buffer.offset890 = trunc i64 %459 to i32, !dbg !18
  %460 = trunc i64 %457 to i32, !dbg !18
  %461 = insertelement <4 x i32> poison, i32 %460, i64 0, !dbg !18
  %462 = lshr i64 %457, 32, !dbg !18
  %463 = trunc i64 %462 to i32, !dbg !18
  %464 = insertelement <4 x i32> %461, i32 %463, i64 1, !dbg !18
  %465 = insertelement <4 x i32> %464, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc891 = insertelement <4 x i32> %465, i32 131072, i64 3, !dbg !18
  %466 = ptrtoint ptr addrspace(3) %456 to i32, !dbg !18
  %dma.buffer.lds892 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %466), !dbg !18
  %467 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds892, i32 %dma.buffer.offset890, <4 x i32> %dma.buffer.rsrc891) #10, !dbg !18
  %468 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !17
  %469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %468, <4 x float> %348) #9, !dbg !17
  %lds.const.gep872 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !17
  %470 = load <4 x half>, ptr addrspace(3) %lds.const.gep872, align 8, !dbg !17
  %471 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !17
  %472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %470, <4 x float> %469) #9, !dbg !17
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %471, <4 x float> %352) #9, !dbg !17
  %lds.const.gep873 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !17
  %474 = load <4 x half>, ptr addrspace(3) %lds.const.gep873, align 8, !dbg !17
  %lds.const.gep874 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !17
  %475 = load <4 x half>, ptr addrspace(3) %lds.const.gep874, align 8, !dbg !17
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %474, <4 x float> %473) #9, !dbg !17
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %475, <4 x float> %356) #9, !dbg !17
  %lds.const.gep875 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !17
  %478 = load <4 x half>, ptr addrspace(3) %lds.const.gep875, align 8, !dbg !17
  %lds.const.gep876 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !17
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep876, align 8, !dbg !17
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %478, <4 x float> %477) #9, !dbg !17
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %479, <4 x float> %360) #9, !dbg !17
  %lds.const.gep877 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !17
  %482 = load <4 x half>, ptr addrspace(3) %lds.const.gep877, align 8, !dbg !17
  %483 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !17
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %482, <4 x float> %481) #9, !dbg !17
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %483, <4 x float> %364) #9, !dbg !17
  %lds.const.gep878 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !17
  %486 = load <4 x half>, ptr addrspace(3) %lds.const.gep878, align 8, !dbg !17
  %487 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !17
  %488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %486, <4 x float> %485) #9, !dbg !17
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %487, <4 x float> %368) #9, !dbg !17
  %lds.const.gep879 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !17
  %490 = load <4 x half>, ptr addrspace(3) %lds.const.gep879, align 8, !dbg !17
  %lds.const.gep880 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !17
  %491 = load <4 x half>, ptr addrspace(3) %lds.const.gep880, align 8, !dbg !17
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %490, <4 x float> %489) #9, !dbg !17
  %493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %491, <4 x float> %372) #9, !dbg !17
  %lds.const.gep881 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !17
  %494 = load <4 x half>, ptr addrspace(3) %lds.const.gep881, align 8, !dbg !17
  %lds.const.gep882 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !17
  %495 = load <4 x half>, ptr addrspace(3) %lds.const.gep882, align 8, !dbg !17
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %494, <4 x float> %493) #9, !dbg !17
  %lds.const.gep883 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !17
  %497 = load <4 x half>, ptr addrspace(3) %lds.const.gep883, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %495, <4 x float> %375) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %499 = or disjoint i64 %70, 63, !dbg !13
  %500 = icmp uge i64 %19, %499, !dbg !13
  %501 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %497, <4 x float> %498) #9, !dbg !17
  %502 = and i1 %39, %500, !dbg !13
  br i1 %502, label %503, label %504, !dbg !13

503:                                              ; preds = %69
  br label %584, !dbg !13

504:                                              ; preds = %69
  %505 = add i64 %19, %35, !dbg !13
  %506 = shl nuw nsw i64 %36, 4, !dbg !13
  %507 = add i64 %505, %506, !dbg !13
  %508 = and i32 %34, 240, !dbg !13
  %index.compact = zext nneg i32 %508 to i64, !dbg !13
  %509 = sub i64 %507, %index.compact, !dbg !13
  %510 = and i64 %35, 48, !dbg !13
  %511 = or disjoint i64 %70, %510, !dbg !13
  %512 = add i64 %509, 1, !dbg !23
  %513 = lshr i64 %512, 32, !dbg !24
  %compare.high = trunc i64 %513 to i32, !dbg !24
  %514 = lshr i64 %511, 32, !dbg !24
  %compare.high697 = trunc i64 %514 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high697, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high697, !dbg !24
  %compare.low.base = trunc i64 %512 to i32, !dbg !24
  %compare.low.base698 = trunc i64 %511 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base698, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %515 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low699 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result700 = select i1 %compare.high.equal, i1 %compare.low699, i1 %compare.high.order, !dbg !24
  %516 = select i1 %compare.result700, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base701 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local702 = or i32 %compare.local.base701, 2, !dbg !24
  %compare.low703 = icmp ugt i32 %compare.low.base, %compare.local702, !dbg !24
  %compare.result704 = select i1 %compare.high.equal, i1 %compare.low703, i1 %compare.high.order, !dbg !24
  %517 = select i1 %compare.result704, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base705 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local706 = or i32 %compare.local.base705, 3, !dbg !24
  %compare.low707 = icmp ugt i32 %compare.low.base, %compare.local706, !dbg !24
  %compare.result708 = select i1 %compare.high.equal, i1 %compare.low707, i1 %compare.high.order, !dbg !24
  %518 = select i1 %compare.result708, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base709 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local710 = or i32 %compare.local.base709, 4, !dbg !24
  %compare.low711 = icmp ugt i32 %compare.low.base, %compare.local710, !dbg !24
  %compare.result712 = select i1 %compare.high.equal, i1 %compare.low711, i1 %compare.high.order, !dbg !24
  %519 = select i1 %compare.result712, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base713 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local714 = or i32 %compare.local.base713, 5, !dbg !24
  %compare.low715 = icmp ugt i32 %compare.low.base, %compare.local714, !dbg !24
  %compare.result716 = select i1 %compare.high.equal, i1 %compare.low715, i1 %compare.high.order, !dbg !24
  %520 = select i1 %compare.result716, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base717 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local718 = or i32 %compare.local.base717, 6, !dbg !24
  %compare.low719 = icmp ugt i32 %compare.low.base, %compare.local718, !dbg !24
  %compare.result720 = select i1 %compare.high.equal, i1 %compare.low719, i1 %compare.high.order, !dbg !24
  %521 = select i1 %compare.result720, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base721 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local722 = or i32 %compare.local.base721, 7, !dbg !24
  %compare.low723 = icmp ugt i32 %compare.low.base, %compare.local722, !dbg !24
  %compare.result724 = select i1 %compare.high.equal, i1 %compare.low723, i1 %compare.high.order, !dbg !24
  %522 = select i1 %compare.result724, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base725 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local726 = or i32 %compare.local.base725, 8, !dbg !24
  %compare.low727 = icmp ugt i32 %compare.low.base, %compare.local726, !dbg !24
  %compare.result728 = select i1 %compare.high.equal, i1 %compare.low727, i1 %compare.high.order, !dbg !24
  %523 = select i1 %compare.result728, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base729 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local730 = or i32 %compare.local.base729, 9, !dbg !24
  %compare.low731 = icmp ugt i32 %compare.low.base, %compare.local730, !dbg !24
  %compare.result732 = select i1 %compare.high.equal, i1 %compare.low731, i1 %compare.high.order, !dbg !24
  %524 = select i1 %compare.result732, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base733 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local734 = or i32 %compare.local.base733, 10, !dbg !24
  %compare.low735 = icmp ugt i32 %compare.low.base, %compare.local734, !dbg !24
  %compare.result736 = select i1 %compare.high.equal, i1 %compare.low735, i1 %compare.high.order, !dbg !24
  %525 = select i1 %compare.result736, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base737 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local738 = or i32 %compare.local.base737, 11, !dbg !24
  %compare.low739 = icmp ugt i32 %compare.low.base, %compare.local738, !dbg !24
  %compare.result740 = select i1 %compare.high.equal, i1 %compare.low739, i1 %compare.high.order, !dbg !24
  %526 = select i1 %compare.result740, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base741 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local742 = or i32 %compare.local.base741, 12, !dbg !24
  %compare.low743 = icmp ugt i32 %compare.low.base, %compare.local742, !dbg !24
  %compare.result744 = select i1 %compare.high.equal, i1 %compare.low743, i1 %compare.high.order, !dbg !24
  %527 = select i1 %compare.result744, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base745 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local746 = or i32 %compare.local.base745, 13, !dbg !24
  %compare.low747 = icmp ugt i32 %compare.low.base, %compare.local746, !dbg !24
  %compare.result748 = select i1 %compare.high.equal, i1 %compare.low747, i1 %compare.high.order, !dbg !24
  %528 = select i1 %compare.result748, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base749 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local750 = or i32 %compare.local.base749, 14, !dbg !24
  %compare.low751 = icmp ugt i32 %compare.low.base, %compare.local750, !dbg !24
  %compare.result752 = select i1 %compare.high.equal, i1 %compare.low751, i1 %compare.high.order, !dbg !24
  %529 = select i1 %compare.result752, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base753 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local754 = or i32 %compare.local.base753, 15, !dbg !24
  %compare.low755 = icmp ugt i32 %compare.low.base, %compare.local754, !dbg !24
  %compare.result756 = select i1 %compare.high.equal, i1 %compare.low755, i1 %compare.high.order, !dbg !24
  %530 = select i1 %compare.result756, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %531 = add i64 %509, 65, !dbg !23
  %532 = lshr i64 %531, 32, !dbg !24
  %compare.high757 = trunc i64 %532 to i32, !dbg !24
  %533 = lshr i64 %511, 32, !dbg !24
  %compare.high758 = trunc i64 %533 to i32, !dbg !24
  %compare.high.equal759 = icmp eq i32 %compare.high757, %compare.high758, !dbg !24
  %compare.high.order760 = icmp ugt i32 %compare.high757, %compare.high758, !dbg !24
  %compare.low.base761 = trunc i64 %531 to i32, !dbg !24
  %compare.low.base762 = trunc i64 %511 to i32, !dbg !24
  %compare.low763 = icmp ugt i32 %compare.low.base761, %compare.low.base762, !dbg !24
  %compare.result764 = select i1 %compare.high.equal759, i1 %compare.low763, i1 %compare.high.order760, !dbg !24
  %534 = select i1 %compare.result764, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base765 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local766 = or i32 %compare.local.base765, 1, !dbg !24
  %compare.low767 = icmp ugt i32 %compare.low.base761, %compare.local766, !dbg !24
  %compare.result768 = select i1 %compare.high.equal759, i1 %compare.low767, i1 %compare.high.order760, !dbg !24
  %535 = select i1 %compare.result768, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base769 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local770 = or i32 %compare.local.base769, 2, !dbg !24
  %compare.low771 = icmp ugt i32 %compare.low.base761, %compare.local770, !dbg !24
  %compare.result772 = select i1 %compare.high.equal759, i1 %compare.low771, i1 %compare.high.order760, !dbg !24
  %536 = select i1 %compare.result772, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base773 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local774 = or i32 %compare.local.base773, 3, !dbg !24
  %compare.low775 = icmp ugt i32 %compare.low.base761, %compare.local774, !dbg !24
  %compare.result776 = select i1 %compare.high.equal759, i1 %compare.low775, i1 %compare.high.order760, !dbg !24
  %537 = select i1 %compare.result776, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base777 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local778 = or i32 %compare.local.base777, 4, !dbg !24
  %compare.low779 = icmp ugt i32 %compare.low.base761, %compare.local778, !dbg !24
  %compare.result780 = select i1 %compare.high.equal759, i1 %compare.low779, i1 %compare.high.order760, !dbg !24
  %538 = select i1 %compare.result780, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base781 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local782 = or i32 %compare.local.base781, 5, !dbg !24
  %compare.low783 = icmp ugt i32 %compare.low.base761, %compare.local782, !dbg !24
  %compare.result784 = select i1 %compare.high.equal759, i1 %compare.low783, i1 %compare.high.order760, !dbg !24
  %539 = select i1 %compare.result784, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base785 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local786 = or i32 %compare.local.base785, 6, !dbg !24
  %compare.low787 = icmp ugt i32 %compare.low.base761, %compare.local786, !dbg !24
  %compare.result788 = select i1 %compare.high.equal759, i1 %compare.low787, i1 %compare.high.order760, !dbg !24
  %540 = select i1 %compare.result788, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base789 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local790 = or i32 %compare.local.base789, 7, !dbg !24
  %compare.low791 = icmp ugt i32 %compare.low.base761, %compare.local790, !dbg !24
  %compare.result792 = select i1 %compare.high.equal759, i1 %compare.low791, i1 %compare.high.order760, !dbg !24
  %541 = select i1 %compare.result792, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base793 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local794 = or i32 %compare.local.base793, 8, !dbg !24
  %compare.low795 = icmp ugt i32 %compare.low.base761, %compare.local794, !dbg !24
  %compare.result796 = select i1 %compare.high.equal759, i1 %compare.low795, i1 %compare.high.order760, !dbg !24
  %542 = select i1 %compare.result796, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base797 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local798 = or i32 %compare.local.base797, 9, !dbg !24
  %compare.low799 = icmp ugt i32 %compare.low.base761, %compare.local798, !dbg !24
  %compare.result800 = select i1 %compare.high.equal759, i1 %compare.low799, i1 %compare.high.order760, !dbg !24
  %543 = select i1 %compare.result800, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base801 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local802 = or i32 %compare.local.base801, 10, !dbg !24
  %compare.low803 = icmp ugt i32 %compare.low.base761, %compare.local802, !dbg !24
  %compare.result804 = select i1 %compare.high.equal759, i1 %compare.low803, i1 %compare.high.order760, !dbg !24
  %544 = select i1 %compare.result804, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base805 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local806 = or i32 %compare.local.base805, 11, !dbg !24
  %compare.low807 = icmp ugt i32 %compare.low.base761, %compare.local806, !dbg !24
  %compare.result808 = select i1 %compare.high.equal759, i1 %compare.low807, i1 %compare.high.order760, !dbg !24
  %545 = select i1 %compare.result808, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base809 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local810 = or i32 %compare.local.base809, 12, !dbg !24
  %compare.low811 = icmp ugt i32 %compare.low.base761, %compare.local810, !dbg !24
  %compare.result812 = select i1 %compare.high.equal759, i1 %compare.low811, i1 %compare.high.order760, !dbg !24
  %546 = select i1 %compare.result812, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base813 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local814 = or i32 %compare.local.base813, 13, !dbg !24
  %compare.low815 = icmp ugt i32 %compare.low.base761, %compare.local814, !dbg !24
  %compare.result816 = select i1 %compare.high.equal759, i1 %compare.low815, i1 %compare.high.order760, !dbg !24
  %547 = select i1 %compare.result816, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base817 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local818 = or i32 %compare.local.base817, 14, !dbg !24
  %compare.low819 = icmp ugt i32 %compare.low.base761, %compare.local818, !dbg !24
  %compare.result820 = select i1 %compare.high.equal759, i1 %compare.low819, i1 %compare.high.order760, !dbg !24
  %548 = select i1 %compare.result820, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base821 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local822 = or i32 %compare.local.base821, 15, !dbg !24
  %compare.low823 = icmp ugt i32 %compare.low.base761, %compare.local822, !dbg !24
  %compare.result824 = select i1 %compare.high.equal759, i1 %compare.low823, i1 %compare.high.order760, !dbg !24
  %549 = select i1 %compare.result824, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %550 = insertelement <16 x float> poison, float %515, i64 0, !dbg !13
  %551 = insertelement <16 x float> %550, float %516, i64 1, !dbg !13
  %552 = insertelement <16 x float> %551, float %517, i64 2, !dbg !13
  %553 = insertelement <16 x float> %552, float %518, i64 3, !dbg !13
  %554 = insertelement <16 x float> %553, float %519, i64 4, !dbg !13
  %555 = insertelement <16 x float> %554, float %520, i64 5, !dbg !13
  %556 = insertelement <16 x float> %555, float %521, i64 6, !dbg !13
  %557 = insertelement <16 x float> %556, float %522, i64 7, !dbg !13
  %558 = insertelement <16 x float> %557, float %523, i64 8, !dbg !13
  %559 = insertelement <16 x float> %558, float %524, i64 9, !dbg !13
  %560 = insertelement <16 x float> %559, float %525, i64 10, !dbg !13
  %561 = insertelement <16 x float> %560, float %526, i64 11, !dbg !13
  %562 = insertelement <16 x float> %561, float %527, i64 12, !dbg !13
  %563 = insertelement <16 x float> %562, float %528, i64 13, !dbg !13
  %564 = insertelement <16 x float> %563, float %529, i64 14, !dbg !13
  %565 = insertelement <16 x float> %564, float %530, i64 15, !dbg !13
  %566 = insertvalue [2 x <16 x float>] poison, <16 x float> %565, 0, !dbg !13
  %567 = insertelement <16 x float> poison, float %534, i64 0, !dbg !13
  %568 = insertelement <16 x float> %567, float %535, i64 1, !dbg !13
  %569 = insertelement <16 x float> %568, float %536, i64 2, !dbg !13
  %570 = insertelement <16 x float> %569, float %537, i64 3, !dbg !13
  %571 = insertelement <16 x float> %570, float %538, i64 4, !dbg !13
  %572 = insertelement <16 x float> %571, float %539, i64 5, !dbg !13
  %573 = insertelement <16 x float> %572, float %540, i64 6, !dbg !13
  %574 = insertelement <16 x float> %573, float %541, i64 7, !dbg !13
  %575 = insertelement <16 x float> %574, float %542, i64 8, !dbg !13
  %576 = insertelement <16 x float> %575, float %543, i64 9, !dbg !13
  %577 = insertelement <16 x float> %576, float %544, i64 10, !dbg !13
  %578 = insertelement <16 x float> %577, float %545, i64 11, !dbg !13
  %579 = insertelement <16 x float> %578, float %546, i64 12, !dbg !13
  %580 = insertelement <16 x float> %579, float %547, i64 13, !dbg !13
  %581 = insertelement <16 x float> %580, float %548, i64 14, !dbg !13
  %582 = insertelement <16 x float> %581, float %549, i64 15, !dbg !13
  %583 = insertvalue [2 x <16 x float>] %566, <16 x float> %582, 1, !dbg !13
  br label %584, !dbg !13

584:                                              ; preds = %503, %504
  %585 = phi [2 x <16 x float>] [ %583, %504 ], [ zeroinitializer, %503 ], !dbg !13
  br label %586, !dbg !13

586:                                              ; preds = %584
  %587 = extractvalue [2 x <16 x float>] %585, 0, !dbg !26
  %588 = shufflevector <16 x float> %587, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %589 = fadd <4 x float> %472, %588, !dbg !26
  %590 = extractelement <4 x float> %589, i64 0, !dbg !27
  %591 = call float @llvm.amdgcn.exp2.f32(float %590), !dbg !27
  %592 = insertelement <4 x float> poison, float %591, i64 0, !dbg !29
  %593 = extractelement <4 x float> %589, i64 1, !dbg !27
  %594 = call float @llvm.amdgcn.exp2.f32(float %593), !dbg !27
  %595 = insertelement <4 x float> %592, float %594, i64 1, !dbg !29
  %596 = fadd float %591, %594, !dbg !28
  %597 = extractelement <4 x float> %589, i64 2, !dbg !27
  %598 = call float @llvm.amdgcn.exp2.f32(float %597), !dbg !27
  %599 = insertelement <4 x float> %595, float %598, i64 2, !dbg !29
  %600 = fadd float %596, %598, !dbg !28
  %601 = extractelement <4 x float> %589, i64 3, !dbg !27
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !27
  %603 = insertelement <4 x float> %599, float %602, i64 3, !dbg !29
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %603) #9, !dbg !29, !frisk.register_lifetimes !1
  %604 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %604) #9, !dbg !28, !frisk.register_lifetimes !1
  %605 = fadd float %600, %602, !dbg !28
  %606 = shufflevector <16 x float> %587, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %607 = fadd <4 x float> %476, %606, !dbg !26
  %608 = extractelement <4 x float> %607, i64 0, !dbg !27
  %609 = call float @llvm.amdgcn.exp2.f32(float %608), !dbg !27
  %610 = insertelement <4 x float> poison, float %609, i64 0, !dbg !29
  %611 = fadd float %605, %609, !dbg !28
  %612 = extractelement <4 x float> %607, i64 1, !dbg !27
  %613 = call float @llvm.amdgcn.exp2.f32(float %612), !dbg !27
  %614 = insertelement <4 x float> %610, float %613, i64 1, !dbg !29
  %615 = fadd float %611, %613, !dbg !28
  %616 = extractelement <4 x float> %607, i64 2, !dbg !27
  %617 = call float @llvm.amdgcn.exp2.f32(float %616), !dbg !27
  %618 = insertelement <4 x float> %614, float %617, i64 2, !dbg !29
  %619 = fadd float %615, %617, !dbg !28
  %620 = extractelement <4 x float> %607, i64 3, !dbg !27
  %621 = call float @llvm.amdgcn.exp2.f32(float %620), !dbg !27
  %622 = insertelement <4 x float> %618, float %621, i64 3, !dbg !29
  %math.result159 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %622) #9, !dbg !29, !frisk.register_lifetimes !1
  %623 = fptrunc <4 x float> %math.result159 to <4 x half>, !dbg !29
  %math.packed160 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %623) #9, !dbg !28, !frisk.register_lifetimes !1
  %624 = fadd float %619, %621, !dbg !28
  %625 = shufflevector <16 x float> %587, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %626 = fadd <4 x float> %480, %625, !dbg !26
  %627 = extractelement <4 x float> %626, i64 0, !dbg !27
  %628 = call float @llvm.amdgcn.exp2.f32(float %627), !dbg !27
  %629 = insertelement <4 x float> poison, float %628, i64 0, !dbg !29
  %630 = fadd float %624, %628, !dbg !28
  %631 = extractelement <4 x float> %626, i64 1, !dbg !27
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !27
  %633 = insertelement <4 x float> %629, float %632, i64 1, !dbg !29
  %634 = fadd float %630, %632, !dbg !28
  %635 = extractelement <4 x float> %626, i64 2, !dbg !27
  %636 = call float @llvm.amdgcn.exp2.f32(float %635), !dbg !27
  %637 = insertelement <4 x float> %633, float %636, i64 2, !dbg !29
  %638 = fadd float %634, %636, !dbg !28
  %639 = extractelement <4 x float> %626, i64 3, !dbg !27
  %640 = call float @llvm.amdgcn.exp2.f32(float %639), !dbg !27
  %641 = insertelement <4 x float> %637, float %640, i64 3, !dbg !29
  %math.result161 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %641) #9, !dbg !29, !frisk.register_lifetimes !1
  %642 = fptrunc <4 x float> %math.result161 to <4 x half>, !dbg !29
  %math.packed162 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %642) #9, !dbg !28, !frisk.register_lifetimes !1
  %643 = fadd float %638, %640, !dbg !28
  %644 = shufflevector <16 x float> %587, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %645 = fadd <4 x float> %484, %644, !dbg !26
  %646 = extractelement <4 x float> %645, i64 0, !dbg !27
  %647 = call float @llvm.amdgcn.exp2.f32(float %646), !dbg !27
  %648 = insertelement <4 x float> poison, float %647, i64 0, !dbg !29
  %649 = fadd float %643, %647, !dbg !28
  %650 = extractelement <4 x float> %645, i64 1, !dbg !27
  %651 = call float @llvm.amdgcn.exp2.f32(float %650), !dbg !27
  %652 = insertelement <4 x float> %648, float %651, i64 1, !dbg !29
  %653 = fadd float %649, %651, !dbg !28
  %654 = extractelement <4 x float> %645, i64 2, !dbg !27
  %655 = call float @llvm.amdgcn.exp2.f32(float %654), !dbg !27
  %656 = insertelement <4 x float> %652, float %655, i64 2, !dbg !29
  %657 = fadd float %653, %655, !dbg !28
  %658 = extractelement <4 x float> %645, i64 3, !dbg !27
  %659 = call float @llvm.amdgcn.exp2.f32(float %658), !dbg !27
  %660 = insertelement <4 x float> %656, float %659, i64 3, !dbg !29
  %math.result163 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %660) #9, !dbg !29, !frisk.register_lifetimes !1
  %661 = fptrunc <4 x float> %math.result163 to <4 x half>, !dbg !29
  %math.packed164 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %661) #9, !dbg !28, !frisk.register_lifetimes !1
  %662 = fadd float %657, %659, !dbg !28
  %663 = extractvalue [2 x <16 x float>] %585, 1, !dbg !26
  %664 = shufflevector <16 x float> %663, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %665 = fadd <4 x float> %488, %664, !dbg !26
  %666 = extractelement <4 x float> %665, i64 0, !dbg !27
  %667 = call float @llvm.amdgcn.exp2.f32(float %666), !dbg !27
  %668 = insertelement <4 x float> poison, float %667, i64 0, !dbg !29
  %669 = extractelement <4 x float> %665, i64 1, !dbg !27
  %670 = call float @llvm.amdgcn.exp2.f32(float %669), !dbg !27
  %671 = insertelement <4 x float> %668, float %670, i64 1, !dbg !29
  %672 = fadd float %667, %670, !dbg !28
  %673 = extractelement <4 x float> %665, i64 2, !dbg !27
  %674 = call float @llvm.amdgcn.exp2.f32(float %673), !dbg !27
  %675 = insertelement <4 x float> %671, float %674, i64 2, !dbg !29
  %676 = fadd float %672, %674, !dbg !28
  %677 = extractelement <4 x float> %665, i64 3, !dbg !27
  %678 = call float @llvm.amdgcn.exp2.f32(float %677), !dbg !27
  %679 = insertelement <4 x float> %675, float %678, i64 3, !dbg !29
  %math.result165 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %679) #9, !dbg !29, !frisk.register_lifetimes !1
  %680 = fptrunc <4 x float> %math.result165 to <4 x half>, !dbg !29
  %math.packed166 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %680) #9, !dbg !28, !frisk.register_lifetimes !1
  %681 = fadd float %676, %678, !dbg !28
  %682 = shufflevector <16 x float> %663, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %683 = fadd <4 x float> %492, %682, !dbg !26
  %684 = extractelement <4 x float> %683, i64 0, !dbg !27
  %685 = call float @llvm.amdgcn.exp2.f32(float %684), !dbg !27
  %686 = insertelement <4 x float> poison, float %685, i64 0, !dbg !29
  %687 = fadd float %681, %685, !dbg !28
  %688 = extractelement <4 x float> %683, i64 1, !dbg !27
  %689 = call float @llvm.amdgcn.exp2.f32(float %688), !dbg !27
  %690 = insertelement <4 x float> %686, float %689, i64 1, !dbg !29
  %691 = fadd float %687, %689, !dbg !28
  %692 = extractelement <4 x float> %683, i64 2, !dbg !27
  %693 = call float @llvm.amdgcn.exp2.f32(float %692), !dbg !27
  %694 = insertelement <4 x float> %690, float %693, i64 2, !dbg !29
  %695 = fadd float %691, %693, !dbg !28
  %696 = extractelement <4 x float> %683, i64 3, !dbg !27
  %697 = call float @llvm.amdgcn.exp2.f32(float %696), !dbg !27
  %698 = insertelement <4 x float> %694, float %697, i64 3, !dbg !29
  %math.result167 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %698) #9, !dbg !29, !frisk.register_lifetimes !1
  %699 = fptrunc <4 x float> %math.result167 to <4 x half>, !dbg !29
  %math.packed168 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %699) #9, !dbg !28, !frisk.register_lifetimes !1
  %700 = fadd float %695, %697, !dbg !28
  %701 = shufflevector <16 x float> %663, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %702 = fadd <4 x float> %496, %701, !dbg !26
  %703 = extractelement <4 x float> %702, i64 0, !dbg !27
  %704 = call float @llvm.amdgcn.exp2.f32(float %703), !dbg !27
  %705 = insertelement <4 x float> poison, float %704, i64 0, !dbg !29
  %706 = fadd float %700, %704, !dbg !28
  %707 = extractelement <4 x float> %702, i64 1, !dbg !27
  %708 = call float @llvm.amdgcn.exp2.f32(float %707), !dbg !27
  %709 = insertelement <4 x float> %705, float %708, i64 1, !dbg !29
  %710 = fadd float %706, %708, !dbg !28
  %711 = extractelement <4 x float> %702, i64 2, !dbg !27
  %712 = call float @llvm.amdgcn.exp2.f32(float %711), !dbg !27
  %713 = insertelement <4 x float> %709, float %712, i64 2, !dbg !29
  %714 = fadd float %710, %712, !dbg !28
  %715 = extractelement <4 x float> %702, i64 3, !dbg !27
  %716 = call float @llvm.amdgcn.exp2.f32(float %715), !dbg !27
  %717 = insertelement <4 x float> %713, float %716, i64 3, !dbg !29
  %math.result169 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %717) #9, !dbg !29, !frisk.register_lifetimes !1
  %718 = fptrunc <4 x float> %math.result169 to <4 x half>, !dbg !29
  %math.packed170 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %718) #9, !dbg !28, !frisk.register_lifetimes !1
  %719 = fadd float %714, %716, !dbg !28
  %720 = shufflevector <16 x float> %663, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %721 = fadd <4 x float> %501, %720, !dbg !26
  %722 = extractelement <4 x float> %721, i64 0, !dbg !27
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !27
  %724 = insertelement <4 x float> poison, float %723, i64 0, !dbg !29
  %725 = fadd float %719, %723, !dbg !28
  %726 = extractelement <4 x float> %721, i64 1, !dbg !27
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !27
  %728 = insertelement <4 x float> %724, float %727, i64 1, !dbg !29
  %729 = fadd float %725, %727, !dbg !28
  %730 = extractelement <4 x float> %721, i64 2, !dbg !27
  %731 = call float @llvm.amdgcn.exp2.f32(float %730), !dbg !27
  %732 = insertelement <4 x float> %728, float %731, i64 2, !dbg !29
  %733 = fadd float %729, %731, !dbg !28
  %734 = extractelement <4 x float> %721, i64 3, !dbg !27
  %735 = call float @llvm.amdgcn.exp2.f32(float %734), !dbg !27
  %736 = insertelement <4 x float> %732, float %735, i64 3, !dbg !29
  %math.result171 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %736) #9, !dbg !29, !frisk.register_lifetimes !1
  %737 = fptrunc <4 x float> %math.result171 to <4 x half>, !dbg !29
  %math.packed172 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %737) #9, !dbg !28, !frisk.register_lifetimes !1
  %738 = fadd float %733, %735, !dbg !28
  %739 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %740 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %739), !dbg !28
  %741 = add nuw nsw i32 %740, 64, !dbg !28
  %742 = xor i32 %740, 16, !dbg !28
  %743 = and i32 %741, 192, !dbg !28
  %744 = icmp samesign ult i32 %742, %743, !dbg !28
  %745 = select i1 %744, i32 %742, i32 %740, !dbg !28
  %746 = shl nuw nsw i32 %745, 2, !dbg !28
  %747 = bitcast float %662 to i32, !dbg !28
  %748 = call i32 @llvm.amdgcn.ds.bpermute(i32 %746, i32 %747), !dbg !28
  %749 = bitcast i32 %748 to float, !dbg !28
  %750 = fadd float %662, %749, !dbg !28
  %751 = xor i32 %740, 32, !dbg !28
  %752 = icmp samesign ult i32 %751, %743, !dbg !28
  %753 = select i1 %752, i32 %751, i32 %740, !dbg !28
  %754 = shl nuw nsw i32 %753, 2, !dbg !28
  %755 = bitcast float %750 to i32, !dbg !28
  %756 = call i32 @llvm.amdgcn.ds.bpermute(i32 %754, i32 %755), !dbg !28
  %757 = bitcast i32 %756 to float, !dbg !28
  %758 = fadd float %750, %757, !dbg !28
  %759 = extractvalue [2 x <1 x float>] %64, 0, !dbg !28
  %760 = extractelement <1 x float> %759, i64 0, !dbg !28
  %761 = fadd float %758, %760, !dbg !28
  %762 = insertelement <1 x float> poison, float %761, i64 0, !dbg !15
  %763 = insertvalue [2 x <1 x float>] poison, <1 x float> %762, 0, !dbg !15
  %764 = bitcast float %738 to i32, !dbg !28
  %765 = call i32 @llvm.amdgcn.ds.bpermute(i32 %746, i32 %764), !dbg !28
  %766 = bitcast i32 %765 to float, !dbg !28
  %767 = fadd float %738, %766, !dbg !28
  %768 = bitcast float %767 to i32, !dbg !28
  %769 = call i32 @llvm.amdgcn.ds.bpermute(i32 %754, i32 %768), !dbg !28
  %770 = bitcast i32 %769 to float, !dbg !28
  %771 = fadd float %767, %770, !dbg !28
  %772 = extractvalue [2 x <1 x float>] %64, 1, !dbg !28
  %773 = extractelement <1 x float> %772, i64 0, !dbg !28
  %774 = fadd float %771, %773, !dbg !28
  %775 = insertelement <1 x float> poison, float %774, i64 0, !dbg !15
  %776 = insertvalue [2 x <1 x float>] %763, <1 x float> %775, 1, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread436 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !14, !range !33, !frisk.register_lifetimes !1
  %address.local437 = zext nneg i32 %address.thread436 to i64, !dbg !12
  %address.local438 = and i64 %address.local437, 60, !dbg !14
  %address.local439 = add nuw nsw i64 %address.local438, %address.local437, !dbg !14
  %address.local440 = shl nuw nsw i64 %address.local439, 3, !dbg !14
  %address.local441 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local440, !dbg !14
  %address.local442 = and i64 %address.local437, 63, !dbg !12
  %address.local443 = lshr i64 %address.local442, 4, !dbg !14
  %address.local444 = mul nuw nsw i64 %address.local443, 384, !dbg !14
  %address.local445 = getelementptr [2 x i8], ptr addrspace(3) %address.local441, i64 %address.local444, !dbg !14
  %address.local446 = lshr i64 %address.local437, 2, !dbg !17
  %address.local447 = mul nsw i64 %address.local446, -32, !dbg !14
  %address.local448 = getelementptr [2 x i8], ptr addrspace(3) %address.local445, i64 %address.local447, !dbg !14
  %777 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local448, i32 0) #0, !dbg !14
  %778 = bitcast <4 x i32> %777 to <8 x half>, !dbg !14
  %address.local449 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local440, !dbg !14
  %address.local450 = getelementptr [2 x i8], ptr addrspace(3) %address.local449, i64 %address.local444, !dbg !14
  %address.local451 = getelementptr [2 x i8], ptr addrspace(3) %address.local450, i64 %address.local447, !dbg !14
  %address.local452 = getelementptr i8, ptr addrspace(3) %address.local451, i64 256, !dbg !14
  %779 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local452, i32 0) #0, !dbg !14
  %780 = bitcast <4 x i32> %779 to <8 x half>, !dbg !14
  %address.local453 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local440, !dbg !14
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) %address.local453, i64 %address.local444, !dbg !14
  %address.local455 = getelementptr [2 x i8], ptr addrspace(3) %address.local454, i64 %address.local447, !dbg !14
  %address.local456 = getelementptr i8, ptr addrspace(3) %address.local455, i64 512, !dbg !14
  %781 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local456, i32 0) #0, !dbg !14
  %782 = bitcast <4 x i32> %781 to <8 x half>, !dbg !14
  %address.local457 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local440, !dbg !14
  %address.local458 = getelementptr [2 x i8], ptr addrspace(3) %address.local457, i64 %address.local444, !dbg !14
  %address.local459 = getelementptr [2 x i8], ptr addrspace(3) %address.local458, i64 %address.local447, !dbg !14
  %address.local460 = getelementptr i8, ptr addrspace(3) %address.local459, i64 768, !dbg !14
  %783 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local460, i32 0) #0, !dbg !14
  %784 = bitcast <4 x i32> %783 to <8 x half>, !dbg !14
  %785 = add nuw nsw i64 %101, %15, !dbg !18
  %786 = shl i64 %785, 19, !dbg !18
  %787 = and i64 %786, 16252928, !dbg !18
  %788 = add i64 %787, %107, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %789 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %788, !dbg !18
  %790 = getelementptr i8, ptr addrspace(1) %789, i64 64, !dbg !18
  %address.local461 = shl nuw nsw i64 %address.local437, 3, !dbg !11
  %address.local462 = and i64 %address.local461, 1536, !dbg !16
  %address.local463 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local462, !dbg !18
  %address.local464 = ptrtoint ptr addrspace(3) %address.local463 to i64, !dbg !18
  %address.local465 = trunc i64 %address.local464 to i32, !dbg !18
  %791 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local465), !dbg !18
  %792 = zext i32 %791 to i64, !dbg !18
  %793 = inttoptr i64 %792 to ptr addrspace(3), !dbg !18
  %address.local466 = shl nuw nsw i64 %address.local437, 5, !dbg !18
  %address.local467 = and i64 %address.local466, 8064, !dbg !18
  %address.local468 = shl nuw nsw i64 %address.local437, 3, !dbg !18
  %address.local469 = and i64 %address.local468, 24, !dbg !18
  %address.local470 = or disjoint i64 %address.local467, %address.local469, !dbg !18
  %address.local471 = getelementptr [2 x i8], ptr addrspace(1) %790, i64 %address.local470, !dbg !18
  %794 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %795 = ptrtoint ptr addrspace(1) %address.local471 to i64, !dbg !18
  %796 = sub i64 %795, %794, !dbg !18
  %dma.buffer.offset887 = trunc i64 %796 to i32, !dbg !18
  %797 = trunc i64 %794 to i32, !dbg !18
  %798 = insertelement <4 x i32> poison, i32 %797, i64 0, !dbg !18
  %799 = lshr i64 %794, 32, !dbg !18
  %800 = trunc i64 %799 to i32, !dbg !18
  %801 = insertelement <4 x i32> %798, i32 %800, i64 1, !dbg !18
  %802 = insertelement <4 x i32> %801, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc888 = insertelement <4 x i32> %802, i32 131072, i64 3, !dbg !18
  %803 = ptrtoint ptr addrspace(3) %793 to i32, !dbg !18
  %dma.buffer.lds889 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %803), !dbg !18
  %804 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds889, i32 %dma.buffer.offset887, <4 x i32> %dma.buffer.rsrc888) #10, !dbg !18
  %805 = shufflevector <8 x half> %778, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %806 = extractvalue [2 x <8 x float>] %63, 0, !dbg !30
  %807 = shufflevector <8 x float> %806, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %808 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %805, <4 x float> %807) #9, !dbg !14
  %809 = shufflevector <8 x half> %780, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %810 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %809, <4 x float> %808) #9, !dbg !14
  %811 = shufflevector <8 x half> %782, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %812 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %811, <4 x float> %810) #9, !dbg !14
  %813 = shufflevector <8 x half> %784, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %813, <4 x float> %812) #9, !dbg !14
  %815 = shufflevector <8 x half> %778, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %816 = shufflevector <8 x float> %806, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %817 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %815, <4 x float> %816) #9, !dbg !14
  %818 = shufflevector <8 x half> %780, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %819 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %818, <4 x float> %817) #9, !dbg !14
  %820 = shufflevector <8 x half> %782, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %821 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %820, <4 x float> %819) #9, !dbg !14
  %822 = shufflevector <8 x half> %784, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %823 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %822, <4 x float> %821) #9, !dbg !14
  %824 = extractvalue [2 x <8 x float>] %63, 1, !dbg !30
  %825 = shufflevector <8 x float> %824, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %805, <4 x float> %825) #9, !dbg !14
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %809, <4 x float> %826) #9, !dbg !14
  %828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %811, <4 x float> %827) #9, !dbg !14
  %829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %813, <4 x float> %828) #9, !dbg !14
  %830 = shufflevector <8 x float> %824, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %815, <4 x float> %830) #9, !dbg !14
  %832 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %818, <4 x float> %831) #9, !dbg !14
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %820, <4 x float> %832) #9, !dbg !14
  %834 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %822, <4 x float> %833) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %835 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local448, i32 0) #0, !dbg !14
  %836 = bitcast <4 x i32> %835 to <8 x half>, !dbg !14
  %837 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local452, i32 0) #0, !dbg !14
  %838 = bitcast <4 x i32> %837 to <8 x half>, !dbg !14
  %839 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local456, i32 0) #0, !dbg !14
  %840 = bitcast <4 x i32> %839 to <8 x half>, !dbg !14
  %841 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local460, i32 0) #0, !dbg !14
  %842 = bitcast <4 x i32> %841 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %843 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %788, !dbg !18
  %844 = getelementptr i8, ptr addrspace(1) %843, i64 128, !dbg !18
  %845 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local465), !dbg !18
  %846 = zext i32 %845 to i64, !dbg !18
  %847 = inttoptr i64 %846 to ptr addrspace(3), !dbg !18
  %address.local472 = getelementptr [2 x i8], ptr addrspace(1) %844, i64 %address.local470, !dbg !18
  %848 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %849 = ptrtoint ptr addrspace(1) %address.local472 to i64, !dbg !18
  %850 = sub i64 %849, %848, !dbg !18
  %dma.buffer.offset884 = trunc i64 %850 to i32, !dbg !18
  %851 = trunc i64 %848 to i32, !dbg !18
  %852 = insertelement <4 x i32> poison, i32 %851, i64 0, !dbg !18
  %853 = lshr i64 %848, 32, !dbg !18
  %854 = trunc i64 %853 to i32, !dbg !18
  %855 = insertelement <4 x i32> %852, i32 %854, i64 1, !dbg !18
  %856 = insertelement <4 x i32> %855, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc885 = insertelement <4 x i32> %856, i32 131072, i64 3, !dbg !18
  %857 = ptrtoint ptr addrspace(3) %847 to i32, !dbg !18
  %dma.buffer.lds886 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %857), !dbg !18
  %858 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds886, i32 %dma.buffer.offset884, <4 x i32> %dma.buffer.rsrc885) #10, !dbg !18
  %859 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %860 = extractvalue [2 x <8 x float>] %65, 0, !dbg !30
  %861 = shufflevector <8 x float> %860, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %859, <4 x float> %861) #9, !dbg !14
  %863 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %864 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %863, <4 x float> %862) #9, !dbg !14
  %865 = shufflevector <8 x half> %840, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %866 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %865, <4 x float> %864) #9, !dbg !14
  %867 = shufflevector <8 x half> %842, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %868 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %867, <4 x float> %866) #9, !dbg !14
  %869 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %870 = shufflevector <8 x float> %860, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %871 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %869, <4 x float> %870) #9, !dbg !14
  %872 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %872, <4 x float> %871) #9, !dbg !14
  %874 = shufflevector <8 x half> %840, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %875 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %874, <4 x float> %873) #9, !dbg !14
  %876 = shufflevector <8 x half> %842, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %877 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %876, <4 x float> %875) #9, !dbg !14
  %878 = extractvalue [2 x <8 x float>] %65, 1, !dbg !30
  %879 = shufflevector <8 x float> %878, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %880 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %859, <4 x float> %879) #9, !dbg !14
  %881 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %863, <4 x float> %880) #9, !dbg !14
  %882 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %865, <4 x float> %881) #9, !dbg !14
  %883 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %867, <4 x float> %882) #9, !dbg !14
  %884 = shufflevector <8 x float> %878, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %869, <4 x float> %884) #9, !dbg !14
  %886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %872, <4 x float> %885) #9, !dbg !14
  %887 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %874, <4 x float> %886) #9, !dbg !14
  %888 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %876, <4 x float> %887) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %889 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local448, i32 0) #0, !dbg !14
  %890 = bitcast <4 x i32> %889 to <8 x half>, !dbg !14
  %891 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local452, i32 0) #0, !dbg !14
  %892 = bitcast <4 x i32> %891 to <8 x half>, !dbg !14
  %893 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local456, i32 0) #0, !dbg !14
  %894 = bitcast <4 x i32> %893 to <8 x half>, !dbg !14
  %895 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local460, i32 0) #0, !dbg !14
  %896 = bitcast <4 x i32> %895 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %897 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %788, !dbg !18
  %898 = getelementptr i8, ptr addrspace(1) %897, i64 192, !dbg !18
  %899 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local465), !dbg !18
  %900 = zext i32 %899 to i64, !dbg !18
  %901 = inttoptr i64 %900 to ptr addrspace(3), !dbg !18
  %address.local473 = getelementptr [2 x i8], ptr addrspace(1) %898, i64 %address.local470, !dbg !18
  %902 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %903 = ptrtoint ptr addrspace(1) %address.local473 to i64, !dbg !18
  %904 = sub i64 %903, %902, !dbg !18
  %dma.buffer.offset = trunc i64 %904 to i32, !dbg !18
  %905 = trunc i64 %902 to i32, !dbg !18
  %906 = insertelement <4 x i32> poison, i32 %905, i64 0, !dbg !18
  %907 = lshr i64 %902, 32, !dbg !18
  %908 = trunc i64 %907 to i32, !dbg !18
  %909 = insertelement <4 x i32> %906, i32 %908, i64 1, !dbg !18
  %910 = insertelement <4 x i32> %909, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %910, i32 131072, i64 3, !dbg !18
  %911 = ptrtoint ptr addrspace(3) %901 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %911), !dbg !18
  %912 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %913 = shufflevector <8 x half> %890, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %914 = extractvalue [2 x <8 x float>] %66, 0, !dbg !30
  %915 = shufflevector <8 x float> %914, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %913, <4 x float> %915) #9, !dbg !14
  %917 = shufflevector <8 x half> %892, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %917, <4 x float> %916) #9, !dbg !14
  %919 = shufflevector <8 x half> %894, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %919, <4 x float> %918) #9, !dbg !14
  %921 = shufflevector <8 x half> %896, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %921, <4 x float> %920) #9, !dbg !14
  %923 = shufflevector <8 x half> %890, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %924 = shufflevector <8 x float> %914, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %923, <4 x float> %924) #9, !dbg !14
  %926 = shufflevector <8 x half> %892, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %926, <4 x float> %925) #9, !dbg !14
  %928 = shufflevector <8 x half> %894, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %928, <4 x float> %927) #9, !dbg !14
  %930 = shufflevector <8 x half> %896, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %930, <4 x float> %929) #9, !dbg !14
  %932 = extractvalue [2 x <8 x float>] %66, 1, !dbg !30
  %933 = shufflevector <8 x float> %932, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %913, <4 x float> %933) #9, !dbg !14
  %935 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %917, <4 x float> %934) #9, !dbg !14
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %919, <4 x float> %935) #9, !dbg !14
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %921, <4 x float> %936) #9, !dbg !14
  %938 = shufflevector <8 x float> %932, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %923, <4 x float> %938) #9, !dbg !14
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %926, <4 x float> %939) #9, !dbg !14
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %928, <4 x float> %940) #9, !dbg !14
  %942 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %930, <4 x float> %941) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %943 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local448, i32 0) #0, !dbg !14
  %944 = bitcast <4 x i32> %943 to <8 x half>, !dbg !14
  %945 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local452, i32 0) #0, !dbg !14
  %946 = bitcast <4 x i32> %945 to <8 x half>, !dbg !14
  %947 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local456, i32 0) #0, !dbg !14
  %948 = bitcast <4 x i32> %947 to <8 x half>, !dbg !14
  %949 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local460, i32 0) #0, !dbg !14
  %950 = bitcast <4 x i32> %949 to <8 x half>, !dbg !14
  %951 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %952 = extractvalue [2 x <8 x float>] %67, 0, !dbg !30
  %953 = shufflevector <8 x float> %952, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %951, <4 x float> %953) #9, !dbg !14
  %955 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %955, <4 x float> %954) #9, !dbg !14
  %957 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %957, <4 x float> %956) #9, !dbg !14
  %959 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %959, <4 x float> %958) #9, !dbg !14
  %961 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %962 = shufflevector <8 x float> %952, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %961, <4 x float> %962) #9, !dbg !14
  %964 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %964, <4 x float> %963) #9, !dbg !14
  %966 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %966, <4 x float> %965) #9, !dbg !14
  %968 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %968, <4 x float> %967) #9, !dbg !14
  %970 = extractvalue [2 x <8 x float>] %67, 1, !dbg !30
  %971 = shufflevector <8 x float> %970, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %951, <4 x float> %971) #9, !dbg !14
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %955, <4 x float> %972) #9, !dbg !14
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %957, <4 x float> %973) #9, !dbg !14
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %959, <4 x float> %974) #9, !dbg !14
  %976 = shufflevector <8 x float> %970, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %961, <4 x float> %976) #9, !dbg !14
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %964, <4 x float> %977) #9, !dbg !14
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %966, <4 x float> %978) #9, !dbg !14
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %968, <4 x float> %979) #9, !dbg !14
  %981 = shufflevector <4 x float> %814, <4 x float> %823, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %982 = insertvalue [2 x <8 x float>] poison, <8 x float> %981, 0, !dbg !15
  %983 = shufflevector <4 x float> %829, <4 x float> %834, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %984 = insertvalue [2 x <8 x float>] %982, <8 x float> %983, 1, !dbg !15
  %985 = shufflevector <4 x float> %868, <4 x float> %877, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %986 = insertvalue [2 x <8 x float>] poison, <8 x float> %985, 0, !dbg !15
  %987 = shufflevector <4 x float> %883, <4 x float> %888, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %988 = insertvalue [2 x <8 x float>] %986, <8 x float> %987, 1, !dbg !15
  %989 = shufflevector <4 x float> %922, <4 x float> %931, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %990 = insertvalue [2 x <8 x float>] poison, <8 x float> %989, 0, !dbg !15
  %991 = shufflevector <4 x float> %937, <4 x float> %942, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %992 = insertvalue [2 x <8 x float>] %990, <8 x float> %991, 1, !dbg !15
  %993 = shufflevector <4 x float> %960, <4 x float> %969, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %994 = insertvalue [2 x <8 x float>] poison, <8 x float> %993, 0, !dbg !15
  %995 = shufflevector <4 x float> %975, <4 x float> %980, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %996 = insertvalue [2 x <8 x float>] %994, <8 x float> %995, 1, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %997 = add nuw nsw i64 %62, 1, !dbg !15
  br label %61, !dbg !15

998:                                              ; preds = %61
  %999 = and i32 %34, 16, !dbg !15
  %.not = icmp eq i32 %999, 0, !dbg !15
  %1000 = and i32 %34, 32, !dbg !15
  %.not18 = icmp eq i32 %1000, 0, !dbg !15
  %1001 = extractvalue [2 x <8 x float>] %67, 0, !dbg !15
  %1002 = extractelement <8 x float> %1001, i64 0, !dbg !15
  %1003 = extractelement <8 x float> %1001, i64 1, !dbg !15
  %1004 = extractelement <8 x float> %1001, i64 2, !dbg !15
  %1005 = extractelement <8 x float> %1001, i64 3, !dbg !15
  %1006 = select i1 %.not, float %1003, float %1002, !dbg !15
  %1007 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !15
  %1008 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1007), !dbg !15
  %1009 = add nuw nsw i32 %1008, 64, !dbg !15
  %1010 = xor i32 %1008, 16, !dbg !15
  %1011 = and i32 %1009, 192, !dbg !15
  %1012 = icmp samesign ult i32 %1010, %1011, !dbg !15
  %1013 = select i1 %1012, i32 %1010, i32 %1008, !dbg !15
  %1014 = shl nuw nsw i32 %1013, 2, !dbg !15
  %1015 = bitcast float %1006 to i32, !dbg !15
  %1016 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1015), !dbg !15
  %1017 = bitcast i32 %1016 to float, !dbg !15
  %1018 = select i1 %.not, float %1002, float %1017, !dbg !15
  %1019 = select i1 %.not, float %1017, float %1003, !dbg !15
  %1020 = select i1 %.not, float %1005, float %1004, !dbg !15
  %1021 = bitcast float %1020 to i32, !dbg !15
  %1022 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1021), !dbg !15
  %1023 = bitcast i32 %1022 to float, !dbg !15
  %1024 = select i1 %.not, float %1004, float %1023, !dbg !15
  %1025 = select i1 %.not, float %1023, float %1005, !dbg !15
  %1026 = select i1 %.not18, float %1024, float %1018, !dbg !15
  %1027 = xor i32 %1008, 32, !dbg !15
  %1028 = icmp samesign ult i32 %1027, %1011, !dbg !15
  %1029 = select i1 %1028, i32 %1027, i32 %1008, !dbg !15
  %1030 = shl nuw nsw i32 %1029, 2, !dbg !15
  %1031 = bitcast float %1026 to i32, !dbg !15
  %1032 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1031), !dbg !15
  %1033 = bitcast i32 %1032 to float, !dbg !15
  %1034 = select i1 %.not18, float %1018, float %1033, !dbg !15
  %1035 = select i1 %.not18, float %1033, float %1024, !dbg !15
  %1036 = select i1 %.not18, float %1025, float %1019, !dbg !15
  %1037 = bitcast float %1036 to i32, !dbg !15
  %1038 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1037), !dbg !15
  %1039 = bitcast i32 %1038 to float, !dbg !15
  %1040 = select i1 %.not18, float %1019, float %1039, !dbg !15
  %1041 = select i1 %.not18, float %1039, float %1025, !dbg !15
  %1042 = extractvalue [2 x <8 x float>] %66, 0, !dbg !15
  %1043 = extractelement <8 x float> %1042, i64 0, !dbg !15
  %1044 = extractelement <8 x float> %1042, i64 1, !dbg !15
  %1045 = extractelement <8 x float> %1042, i64 2, !dbg !15
  %1046 = extractelement <8 x float> %1042, i64 3, !dbg !15
  %1047 = select i1 %.not, float %1044, float %1043, !dbg !15
  %1048 = bitcast float %1047 to i32, !dbg !15
  %1049 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1048), !dbg !15
  %1050 = bitcast i32 %1049 to float, !dbg !15
  %1051 = select i1 %.not, float %1043, float %1050, !dbg !15
  %1052 = select i1 %.not, float %1050, float %1044, !dbg !15
  %1053 = select i1 %.not, float %1046, float %1045, !dbg !15
  %1054 = bitcast float %1053 to i32, !dbg !15
  %1055 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1054), !dbg !15
  %1056 = bitcast i32 %1055 to float, !dbg !15
  %1057 = select i1 %.not, float %1045, float %1056, !dbg !15
  %1058 = select i1 %.not, float %1056, float %1046, !dbg !15
  %1059 = select i1 %.not18, float %1057, float %1051, !dbg !15
  %1060 = bitcast float %1059 to i32, !dbg !15
  %1061 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1060), !dbg !15
  %1062 = bitcast i32 %1061 to float, !dbg !15
  %1063 = select i1 %.not18, float %1051, float %1062, !dbg !15
  %1064 = select i1 %.not18, float %1062, float %1057, !dbg !15
  %1065 = select i1 %.not18, float %1058, float %1052, !dbg !15
  %1066 = bitcast float %1065 to i32, !dbg !15
  %1067 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1066), !dbg !15
  %1068 = bitcast i32 %1067 to float, !dbg !15
  %1069 = select i1 %.not18, float %1052, float %1068, !dbg !15
  %1070 = select i1 %.not18, float %1068, float %1058, !dbg !15
  %1071 = extractvalue [2 x <8 x float>] %65, 0, !dbg !15
  %1072 = extractelement <8 x float> %1071, i64 0, !dbg !15
  %1073 = extractelement <8 x float> %1071, i64 1, !dbg !15
  %1074 = extractelement <8 x float> %1071, i64 2, !dbg !15
  %1075 = extractelement <8 x float> %1071, i64 3, !dbg !15
  %1076 = select i1 %.not, float %1073, float %1072, !dbg !15
  %1077 = bitcast float %1076 to i32, !dbg !15
  %1078 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1077), !dbg !15
  %1079 = bitcast i32 %1078 to float, !dbg !15
  %1080 = select i1 %.not, float %1072, float %1079, !dbg !15
  %1081 = select i1 %.not, float %1079, float %1073, !dbg !15
  %1082 = select i1 %.not, float %1075, float %1074, !dbg !15
  %1083 = bitcast float %1082 to i32, !dbg !15
  %1084 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1083), !dbg !15
  %1085 = bitcast i32 %1084 to float, !dbg !15
  %1086 = select i1 %.not, float %1074, float %1085, !dbg !15
  %1087 = select i1 %.not, float %1085, float %1075, !dbg !15
  %1088 = select i1 %.not18, float %1086, float %1080, !dbg !15
  %1089 = bitcast float %1088 to i32, !dbg !15
  %1090 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1089), !dbg !15
  %1091 = bitcast i32 %1090 to float, !dbg !15
  %1092 = select i1 %.not18, float %1080, float %1091, !dbg !15
  %1093 = select i1 %.not18, float %1091, float %1086, !dbg !15
  %1094 = select i1 %.not18, float %1087, float %1081, !dbg !15
  %1095 = bitcast float %1094 to i32, !dbg !15
  %1096 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1095), !dbg !15
  %1097 = bitcast i32 %1096 to float, !dbg !15
  %1098 = select i1 %.not18, float %1081, float %1097, !dbg !15
  %1099 = select i1 %.not18, float %1097, float %1087, !dbg !15
  %1100 = extractvalue [2 x <8 x float>] %63, 0, !dbg !15
  %1101 = extractelement <8 x float> %1100, i64 0, !dbg !15
  %1102 = extractelement <8 x float> %1100, i64 1, !dbg !15
  %1103 = extractelement <8 x float> %1100, i64 2, !dbg !15
  %1104 = extractelement <8 x float> %1100, i64 3, !dbg !15
  %1105 = select i1 %.not, float %1102, float %1101, !dbg !15
  %1106 = bitcast float %1105 to i32, !dbg !15
  %1107 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1106), !dbg !15
  %1108 = bitcast i32 %1107 to float, !dbg !15
  %1109 = select i1 %.not, float %1101, float %1108, !dbg !15
  %1110 = select i1 %.not, float %1108, float %1102, !dbg !15
  %1111 = select i1 %.not, float %1104, float %1103, !dbg !15
  %1112 = bitcast float %1111 to i32, !dbg !15
  %1113 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1112), !dbg !15
  %1114 = bitcast i32 %1113 to float, !dbg !15
  %1115 = select i1 %.not, float %1103, float %1114, !dbg !15
  %1116 = select i1 %.not, float %1114, float %1104, !dbg !15
  %1117 = select i1 %.not18, float %1115, float %1109, !dbg !15
  %1118 = bitcast float %1117 to i32, !dbg !15
  %1119 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1118), !dbg !15
  %1120 = select i1 %.not18, float %1116, float %1110, !dbg !15
  %1121 = bitcast float %1120 to i32, !dbg !15
  %1122 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1121), !dbg !15
  %1123 = extractvalue [2 x <1 x float>] %64, 0, !dbg !19
  %1124 = shufflevector <1 x float> %1123, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1125 = extractelement <8 x float> %1001, i64 4, !dbg !15
  %1126 = extractelement <8 x float> %1001, i64 5, !dbg !15
  %1127 = extractelement <8 x float> %1001, i64 6, !dbg !15
  %1128 = extractelement <8 x float> %1001, i64 7, !dbg !15
  %1129 = select i1 %.not, float %1126, float %1125, !dbg !15
  %1130 = bitcast float %1129 to i32, !dbg !15
  %1131 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1130), !dbg !15
  %1132 = bitcast i32 %1131 to float, !dbg !15
  %1133 = select i1 %.not, float %1125, float %1132, !dbg !15
  %1134 = select i1 %.not, float %1132, float %1126, !dbg !15
  %1135 = select i1 %.not, float %1128, float %1127, !dbg !15
  %1136 = bitcast float %1135 to i32, !dbg !15
  %1137 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1136), !dbg !15
  %1138 = bitcast i32 %1137 to float, !dbg !15
  %1139 = select i1 %.not, float %1127, float %1138, !dbg !15
  %1140 = select i1 %.not, float %1138, float %1128, !dbg !15
  %1141 = select i1 %.not18, float %1139, float %1133, !dbg !15
  %1142 = bitcast float %1141 to i32, !dbg !15
  %1143 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1142), !dbg !15
  %1144 = bitcast i32 %1143 to float, !dbg !15
  %1145 = select i1 %.not18, float %1133, float %1144, !dbg !15
  %1146 = select i1 %.not18, float %1144, float %1139, !dbg !15
  %1147 = select i1 %.not18, float %1140, float %1134, !dbg !15
  %1148 = bitcast float %1147 to i32, !dbg !15
  %1149 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1148), !dbg !15
  %1150 = bitcast i32 %1149 to float, !dbg !15
  %1151 = select i1 %.not18, float %1134, float %1150, !dbg !15
  %1152 = select i1 %.not18, float %1150, float %1140, !dbg !15
  %1153 = extractelement <8 x float> %1042, i64 4, !dbg !15
  %1154 = extractelement <8 x float> %1042, i64 5, !dbg !15
  %1155 = extractelement <8 x float> %1042, i64 6, !dbg !15
  %1156 = extractelement <8 x float> %1042, i64 7, !dbg !15
  %1157 = select i1 %.not, float %1154, float %1153, !dbg !15
  %1158 = bitcast float %1157 to i32, !dbg !15
  %1159 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1158), !dbg !15
  %1160 = bitcast i32 %1159 to float, !dbg !15
  %1161 = select i1 %.not, float %1153, float %1160, !dbg !15
  %1162 = select i1 %.not, float %1160, float %1154, !dbg !15
  %1163 = select i1 %.not, float %1156, float %1155, !dbg !15
  %1164 = bitcast float %1163 to i32, !dbg !15
  %1165 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1164), !dbg !15
  %1166 = bitcast i32 %1165 to float, !dbg !15
  %1167 = select i1 %.not, float %1155, float %1166, !dbg !15
  %1168 = select i1 %.not, float %1166, float %1156, !dbg !15
  %1169 = select i1 %.not18, float %1167, float %1161, !dbg !15
  %1170 = bitcast float %1169 to i32, !dbg !15
  %1171 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1170), !dbg !15
  %1172 = bitcast i32 %1171 to float, !dbg !15
  %1173 = select i1 %.not18, float %1161, float %1172, !dbg !15
  %1174 = select i1 %.not18, float %1172, float %1167, !dbg !15
  %1175 = select i1 %.not18, float %1168, float %1162, !dbg !15
  %1176 = bitcast float %1175 to i32, !dbg !15
  %1177 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1176), !dbg !15
  %1178 = bitcast i32 %1177 to float, !dbg !15
  %1179 = select i1 %.not18, float %1162, float %1178, !dbg !15
  %1180 = select i1 %.not18, float %1178, float %1168, !dbg !15
  %1181 = extractelement <8 x float> %1071, i64 4, !dbg !15
  %1182 = extractelement <8 x float> %1071, i64 5, !dbg !15
  %1183 = extractelement <8 x float> %1071, i64 6, !dbg !15
  %1184 = extractelement <8 x float> %1071, i64 7, !dbg !15
  %1185 = select i1 %.not, float %1182, float %1181, !dbg !15
  %1186 = bitcast float %1185 to i32, !dbg !15
  %1187 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1186), !dbg !15
  %1188 = bitcast i32 %1187 to float, !dbg !15
  %1189 = select i1 %.not, float %1181, float %1188, !dbg !15
  %1190 = select i1 %.not, float %1188, float %1182, !dbg !15
  %1191 = select i1 %.not, float %1184, float %1183, !dbg !15
  %1192 = bitcast float %1191 to i32, !dbg !15
  %1193 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1192), !dbg !15
  %1194 = bitcast i32 %1193 to float, !dbg !15
  %1195 = select i1 %.not, float %1183, float %1194, !dbg !15
  %1196 = select i1 %.not, float %1194, float %1184, !dbg !15
  %1197 = select i1 %.not18, float %1195, float %1189, !dbg !15
  %1198 = bitcast float %1197 to i32, !dbg !15
  %1199 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1198), !dbg !15
  %1200 = bitcast i32 %1199 to float, !dbg !15
  %1201 = select i1 %.not18, float %1189, float %1200, !dbg !15
  %1202 = select i1 %.not18, float %1200, float %1195, !dbg !15
  %1203 = select i1 %.not18, float %1196, float %1190, !dbg !15
  %1204 = bitcast float %1203 to i32, !dbg !15
  %1205 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1204), !dbg !15
  %1206 = bitcast i32 %1205 to float, !dbg !15
  %1207 = select i1 %.not18, float %1190, float %1206, !dbg !15
  %1208 = select i1 %.not18, float %1206, float %1196, !dbg !15
  %1209 = extractelement <8 x float> %1100, i64 4, !dbg !15
  %1210 = extractelement <8 x float> %1100, i64 5, !dbg !15
  %1211 = extractelement <8 x float> %1100, i64 6, !dbg !15
  %1212 = extractelement <8 x float> %1100, i64 7, !dbg !15
  %1213 = select i1 %.not, float %1210, float %1209, !dbg !15
  %1214 = bitcast float %1213 to i32, !dbg !15
  %1215 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1214), !dbg !15
  %1216 = bitcast i32 %1215 to float, !dbg !15
  %1217 = select i1 %.not, float %1209, float %1216, !dbg !15
  %1218 = select i1 %.not, float %1216, float %1210, !dbg !15
  %1219 = select i1 %.not, float %1212, float %1211, !dbg !15
  %1220 = bitcast float %1219 to i32, !dbg !15
  %1221 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1220), !dbg !15
  %1222 = bitcast i32 %1221 to float, !dbg !15
  %1223 = select i1 %.not, float %1211, float %1222, !dbg !15
  %1224 = select i1 %.not, float %1222, float %1212, !dbg !15
  %1225 = select i1 %.not18, float %1223, float %1217, !dbg !15
  %1226 = bitcast float %1225 to i32, !dbg !15
  %1227 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1226), !dbg !15
  %1228 = select i1 %.not18, float %1224, float %1218, !dbg !15
  %1229 = bitcast float %1228 to i32, !dbg !15
  %1230 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1229), !dbg !15
  %1231 = extractvalue [2 x <8 x float>] %67, 1, !dbg !15
  %1232 = extractelement <8 x float> %1231, i64 0, !dbg !15
  %1233 = extractelement <8 x float> %1231, i64 1, !dbg !15
  %1234 = extractelement <8 x float> %1231, i64 2, !dbg !15
  %1235 = extractelement <8 x float> %1231, i64 3, !dbg !15
  %1236 = select i1 %.not, float %1233, float %1232, !dbg !15
  %1237 = bitcast float %1236 to i32, !dbg !15
  %1238 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1237), !dbg !15
  %1239 = bitcast i32 %1238 to float, !dbg !15
  %1240 = select i1 %.not, float %1232, float %1239, !dbg !15
  %1241 = select i1 %.not, float %1239, float %1233, !dbg !15
  %1242 = select i1 %.not, float %1235, float %1234, !dbg !15
  %1243 = bitcast float %1242 to i32, !dbg !15
  %1244 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1243), !dbg !15
  %1245 = bitcast i32 %1244 to float, !dbg !15
  %1246 = select i1 %.not, float %1234, float %1245, !dbg !15
  %1247 = select i1 %.not, float %1245, float %1235, !dbg !15
  %1248 = select i1 %.not18, float %1246, float %1240, !dbg !15
  %1249 = bitcast float %1248 to i32, !dbg !15
  %1250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1249), !dbg !15
  %1251 = bitcast i32 %1250 to float, !dbg !15
  %1252 = select i1 %.not18, float %1240, float %1251, !dbg !15
  %1253 = select i1 %.not18, float %1251, float %1246, !dbg !15
  %1254 = select i1 %.not18, float %1247, float %1241, !dbg !15
  %1255 = bitcast float %1254 to i32, !dbg !15
  %1256 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1255), !dbg !15
  %1257 = bitcast i32 %1256 to float, !dbg !15
  %1258 = select i1 %.not18, float %1241, float %1257, !dbg !15
  %1259 = select i1 %.not18, float %1257, float %1247, !dbg !15
  %1260 = extractvalue [2 x <8 x float>] %66, 1, !dbg !15
  %1261 = extractelement <8 x float> %1260, i64 0, !dbg !15
  %1262 = extractelement <8 x float> %1260, i64 1, !dbg !15
  %1263 = extractelement <8 x float> %1260, i64 2, !dbg !15
  %1264 = extractelement <8 x float> %1260, i64 3, !dbg !15
  %1265 = select i1 %.not, float %1262, float %1261, !dbg !15
  %1266 = bitcast float %1265 to i32, !dbg !15
  %1267 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1266), !dbg !15
  %1268 = bitcast i32 %1267 to float, !dbg !15
  %1269 = select i1 %.not, float %1261, float %1268, !dbg !15
  %1270 = select i1 %.not, float %1268, float %1262, !dbg !15
  %1271 = select i1 %.not, float %1264, float %1263, !dbg !15
  %1272 = bitcast float %1271 to i32, !dbg !15
  %1273 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1272), !dbg !15
  %1274 = bitcast i32 %1273 to float, !dbg !15
  %1275 = select i1 %.not, float %1263, float %1274, !dbg !15
  %1276 = select i1 %.not, float %1274, float %1264, !dbg !15
  %1277 = select i1 %.not18, float %1275, float %1269, !dbg !15
  %1278 = bitcast float %1277 to i32, !dbg !15
  %1279 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1278), !dbg !15
  %1280 = bitcast i32 %1279 to float, !dbg !15
  %1281 = select i1 %.not18, float %1269, float %1280, !dbg !15
  %1282 = select i1 %.not18, float %1280, float %1275, !dbg !15
  %1283 = select i1 %.not18, float %1276, float %1270, !dbg !15
  %1284 = bitcast float %1283 to i32, !dbg !15
  %1285 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1284), !dbg !15
  %1286 = bitcast i32 %1285 to float, !dbg !15
  %1287 = select i1 %.not18, float %1270, float %1286, !dbg !15
  %1288 = select i1 %.not18, float %1286, float %1276, !dbg !15
  %1289 = extractvalue [2 x <8 x float>] %65, 1, !dbg !15
  %1290 = extractelement <8 x float> %1289, i64 0, !dbg !15
  %1291 = extractelement <8 x float> %1289, i64 1, !dbg !15
  %1292 = extractelement <8 x float> %1289, i64 2, !dbg !15
  %1293 = extractelement <8 x float> %1289, i64 3, !dbg !15
  %1294 = select i1 %.not, float %1291, float %1290, !dbg !15
  %1295 = bitcast float %1294 to i32, !dbg !15
  %1296 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1295), !dbg !15
  %1297 = bitcast i32 %1296 to float, !dbg !15
  %1298 = select i1 %.not, float %1290, float %1297, !dbg !15
  %1299 = select i1 %.not, float %1297, float %1291, !dbg !15
  %1300 = select i1 %.not, float %1293, float %1292, !dbg !15
  %1301 = bitcast float %1300 to i32, !dbg !15
  %1302 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1301), !dbg !15
  %1303 = bitcast i32 %1302 to float, !dbg !15
  %1304 = select i1 %.not, float %1292, float %1303, !dbg !15
  %1305 = select i1 %.not, float %1303, float %1293, !dbg !15
  %1306 = select i1 %.not18, float %1304, float %1298, !dbg !15
  %1307 = bitcast float %1306 to i32, !dbg !15
  %1308 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1307), !dbg !15
  %1309 = bitcast i32 %1308 to float, !dbg !15
  %1310 = select i1 %.not18, float %1298, float %1309, !dbg !15
  %1311 = select i1 %.not18, float %1309, float %1304, !dbg !15
  %1312 = select i1 %.not18, float %1305, float %1299, !dbg !15
  %1313 = bitcast float %1312 to i32, !dbg !15
  %1314 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1313), !dbg !15
  %1315 = bitcast i32 %1314 to float, !dbg !15
  %1316 = select i1 %.not18, float %1299, float %1315, !dbg !15
  %1317 = select i1 %.not18, float %1315, float %1305, !dbg !15
  %1318 = extractvalue [2 x <8 x float>] %63, 1, !dbg !15
  %1319 = extractelement <8 x float> %1318, i64 0, !dbg !15
  %1320 = extractelement <8 x float> %1318, i64 1, !dbg !15
  %1321 = extractelement <8 x float> %1318, i64 2, !dbg !15
  %1322 = extractelement <8 x float> %1318, i64 3, !dbg !15
  %1323 = select i1 %.not, float %1320, float %1319, !dbg !15
  %1324 = bitcast float %1323 to i32, !dbg !15
  %1325 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1324), !dbg !15
  %1326 = bitcast i32 %1325 to float, !dbg !15
  %1327 = select i1 %.not, float %1319, float %1326, !dbg !15
  %1328 = select i1 %.not, float %1326, float %1320, !dbg !15
  %1329 = select i1 %.not, float %1322, float %1321, !dbg !15
  %1330 = bitcast float %1329 to i32, !dbg !15
  %1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1330), !dbg !15
  %1332 = bitcast i32 %1331 to float, !dbg !15
  %1333 = select i1 %.not, float %1321, float %1332, !dbg !15
  %1334 = select i1 %.not, float %1332, float %1322, !dbg !15
  %1335 = select i1 %.not18, float %1333, float %1327, !dbg !15
  %1336 = bitcast float %1335 to i32, !dbg !15
  %1337 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1336), !dbg !15
  %1338 = select i1 %.not18, float %1334, float %1328, !dbg !15
  %1339 = bitcast float %1338 to i32, !dbg !15
  %1340 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1339), !dbg !15
  %1341 = extractvalue [2 x <1 x float>] %64, 1, !dbg !19
  %1342 = shufflevector <1 x float> %1341, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1343 = extractelement <8 x float> %1231, i64 4, !dbg !15
  %1344 = extractelement <8 x float> %1231, i64 5, !dbg !15
  %1345 = extractelement <8 x float> %1231, i64 6, !dbg !15
  %1346 = extractelement <8 x float> %1231, i64 7, !dbg !15
  %1347 = select i1 %.not, float %1344, float %1343, !dbg !15
  %1348 = bitcast float %1347 to i32, !dbg !15
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1348), !dbg !15
  %1350 = bitcast i32 %1349 to float, !dbg !15
  %1351 = select i1 %.not, float %1343, float %1350, !dbg !15
  %1352 = select i1 %.not, float %1350, float %1344, !dbg !15
  %1353 = select i1 %.not, float %1346, float %1345, !dbg !15
  %1354 = bitcast float %1353 to i32, !dbg !15
  %1355 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1354), !dbg !15
  %1356 = bitcast i32 %1355 to float, !dbg !15
  %1357 = select i1 %.not, float %1345, float %1356, !dbg !15
  %1358 = select i1 %.not, float %1356, float %1346, !dbg !15
  %1359 = select i1 %.not18, float %1357, float %1351, !dbg !15
  %1360 = bitcast float %1359 to i32, !dbg !15
  %1361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1360), !dbg !15
  %1362 = bitcast i32 %1361 to float, !dbg !15
  %1363 = select i1 %.not18, float %1351, float %1362, !dbg !15
  %1364 = select i1 %.not18, float %1362, float %1357, !dbg !15
  %1365 = select i1 %.not18, float %1358, float %1352, !dbg !15
  %1366 = bitcast float %1365 to i32, !dbg !15
  %1367 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1366), !dbg !15
  %1368 = bitcast i32 %1367 to float, !dbg !15
  %1369 = select i1 %.not18, float %1352, float %1368, !dbg !15
  %1370 = select i1 %.not18, float %1368, float %1358, !dbg !15
  %1371 = extractelement <8 x float> %1260, i64 4, !dbg !15
  %1372 = extractelement <8 x float> %1260, i64 5, !dbg !15
  %1373 = extractelement <8 x float> %1260, i64 6, !dbg !15
  %1374 = extractelement <8 x float> %1260, i64 7, !dbg !15
  %1375 = select i1 %.not, float %1372, float %1371, !dbg !15
  %1376 = bitcast float %1375 to i32, !dbg !15
  %1377 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1376), !dbg !15
  %1378 = bitcast i32 %1377 to float, !dbg !15
  %1379 = select i1 %.not, float %1371, float %1378, !dbg !15
  %1380 = select i1 %.not, float %1378, float %1372, !dbg !15
  %1381 = select i1 %.not, float %1374, float %1373, !dbg !15
  %1382 = bitcast float %1381 to i32, !dbg !15
  %1383 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1382), !dbg !15
  %1384 = bitcast i32 %1383 to float, !dbg !15
  %1385 = select i1 %.not, float %1373, float %1384, !dbg !15
  %1386 = select i1 %.not, float %1384, float %1374, !dbg !15
  %1387 = select i1 %.not18, float %1385, float %1379, !dbg !15
  %1388 = bitcast float %1387 to i32, !dbg !15
  %1389 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1388), !dbg !15
  %1390 = bitcast i32 %1389 to float, !dbg !15
  %1391 = select i1 %.not18, float %1379, float %1390, !dbg !15
  %1392 = select i1 %.not18, float %1390, float %1385, !dbg !15
  %1393 = select i1 %.not18, float %1386, float %1380, !dbg !15
  %1394 = bitcast float %1393 to i32, !dbg !15
  %1395 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1394), !dbg !15
  %1396 = bitcast i32 %1395 to float, !dbg !15
  %1397 = select i1 %.not18, float %1380, float %1396, !dbg !15
  %1398 = select i1 %.not18, float %1396, float %1386, !dbg !15
  %1399 = extractelement <8 x float> %1289, i64 4, !dbg !15
  %1400 = extractelement <8 x float> %1289, i64 5, !dbg !15
  %1401 = extractelement <8 x float> %1289, i64 6, !dbg !15
  %1402 = extractelement <8 x float> %1289, i64 7, !dbg !15
  %1403 = select i1 %.not, float %1400, float %1399, !dbg !15
  %1404 = bitcast float %1403 to i32, !dbg !15
  %1405 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1404), !dbg !15
  %1406 = bitcast i32 %1405 to float, !dbg !15
  %1407 = select i1 %.not, float %1399, float %1406, !dbg !15
  %1408 = select i1 %.not, float %1406, float %1400, !dbg !15
  %1409 = select i1 %.not, float %1402, float %1401, !dbg !15
  %1410 = bitcast float %1409 to i32, !dbg !15
  %1411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1410), !dbg !15
  %1412 = bitcast i32 %1411 to float, !dbg !15
  %1413 = select i1 %.not, float %1401, float %1412, !dbg !15
  %1414 = select i1 %.not, float %1412, float %1402, !dbg !15
  %1415 = select i1 %.not18, float %1413, float %1407, !dbg !15
  %1416 = bitcast float %1415 to i32, !dbg !15
  %1417 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1416), !dbg !15
  %1418 = bitcast i32 %1417 to float, !dbg !15
  %1419 = select i1 %.not18, float %1407, float %1418, !dbg !15
  %1420 = select i1 %.not18, float %1418, float %1413, !dbg !15
  %1421 = select i1 %.not18, float %1414, float %1408, !dbg !15
  %1422 = bitcast float %1421 to i32, !dbg !15
  %1423 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1422), !dbg !15
  %1424 = bitcast i32 %1423 to float, !dbg !15
  %1425 = select i1 %.not18, float %1408, float %1424, !dbg !15
  %1426 = select i1 %.not18, float %1424, float %1414, !dbg !15
  %1427 = extractelement <8 x float> %1318, i64 4, !dbg !15
  %1428 = extractelement <8 x float> %1318, i64 5, !dbg !15
  %1429 = extractelement <8 x float> %1318, i64 6, !dbg !15
  %1430 = extractelement <8 x float> %1318, i64 7, !dbg !15
  %1431 = select i1 %.not, float %1428, float %1427, !dbg !15
  %1432 = bitcast float %1431 to i32, !dbg !15
  %1433 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1432), !dbg !15
  %1434 = bitcast i32 %1433 to float, !dbg !15
  %1435 = select i1 %.not, float %1427, float %1434, !dbg !15
  %1436 = select i1 %.not, float %1434, float %1428, !dbg !15
  %1437 = select i1 %.not, float %1430, float %1429, !dbg !15
  %1438 = bitcast float %1437 to i32, !dbg !15
  %1439 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1438), !dbg !15
  %1440 = bitcast i32 %1439 to float, !dbg !15
  %1441 = select i1 %.not, float %1429, float %1440, !dbg !15
  %1442 = select i1 %.not, float %1440, float %1430, !dbg !15
  %1443 = select i1 %.not18, float %1441, float %1435, !dbg !15
  %1444 = bitcast float %1443 to i32, !dbg !15
  %1445 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1444), !dbg !15
  %1446 = select i1 %.not18, float %1442, float %1436, !dbg !15
  %1447 = bitcast float %1446 to i32, !dbg !15
  %1448 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1030, i32 %1447), !dbg !15
  br i1 true, label %1449, label %1513, !dbg !21

1449:                                             ; preds = %998
  %1450 = bitcast i32 %1445 to float, !dbg !15
  %1451 = select i1 %.not18, float %1435, float %1450, !dbg !15
  %1452 = insertelement <4 x float> poison, float %1451, i64 0, !dbg !19
  %1453 = bitcast i32 %1448 to float, !dbg !15
  %1454 = select i1 %.not18, float %1436, float %1453, !dbg !15
  %1455 = insertelement <4 x float> %1452, float %1454, i64 1, !dbg !19
  %1456 = select i1 %.not18, float %1450, float %1441, !dbg !15
  %1457 = insertelement <4 x float> %1455, float %1456, i64 2, !dbg !19
  %1458 = select i1 %.not18, float %1453, float %1442, !dbg !15
  %1459 = insertelement <4 x float> %1457, float %1458, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1342, !dbg !19
  %normalized = fmul <4 x float> %1459, %row.reciprocal, !dbg !19
  %1460 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1461 = extractelement <4 x half> %1460, i64 3, !dbg !20
  %1462 = extractelement <4 x half> %1460, i64 2, !dbg !20
  %1463 = extractelement <4 x half> %1460, i64 1, !dbg !20
  %1464 = extractelement <4 x half> %1460, i64 0, !dbg !20
  %1465 = bitcast i32 %1337 to float, !dbg !15
  %1466 = select i1 %.not18, float %1327, float %1465, !dbg !15
  %1467 = insertelement <4 x float> poison, float %1466, i64 0, !dbg !19
  %1468 = bitcast i32 %1340 to float, !dbg !15
  %1469 = select i1 %.not18, float %1328, float %1468, !dbg !15
  %1470 = insertelement <4 x float> %1467, float %1469, i64 1, !dbg !19
  %1471 = select i1 %.not18, float %1465, float %1333, !dbg !15
  %1472 = insertelement <4 x float> %1470, float %1471, i64 2, !dbg !19
  %1473 = select i1 %.not18, float %1468, float %1334, !dbg !15
  %1474 = insertelement <4 x float> %1472, float %1473, i64 3, !dbg !19
  %normalized54 = fmul <4 x float> %1474, %row.reciprocal, !dbg !19
  %1475 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !20
  %1476 = extractelement <4 x half> %1475, i64 3, !dbg !20
  %1477 = extractelement <4 x half> %1475, i64 2, !dbg !20
  %1478 = extractelement <4 x half> %1475, i64 1, !dbg !20
  %1479 = extractelement <4 x half> %1475, i64 0, !dbg !20
  %1480 = bitcast i32 %1227 to float, !dbg !15
  %1481 = select i1 %.not18, float %1217, float %1480, !dbg !15
  %1482 = insertelement <4 x float> poison, float %1481, i64 0, !dbg !19
  %1483 = bitcast i32 %1230 to float, !dbg !15
  %1484 = select i1 %.not18, float %1218, float %1483, !dbg !15
  %1485 = insertelement <4 x float> %1482, float %1484, i64 1, !dbg !19
  %1486 = select i1 %.not18, float %1480, float %1223, !dbg !15
  %1487 = insertelement <4 x float> %1485, float %1486, i64 2, !dbg !19
  %1488 = select i1 %.not18, float %1483, float %1224, !dbg !15
  %1489 = insertelement <4 x float> %1487, float %1488, i64 3, !dbg !19
  %row.reciprocal55 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1124, !dbg !19
  %normalized56 = fmul <4 x float> %1489, %row.reciprocal55, !dbg !19
  %1490 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !20
  %1491 = extractelement <4 x half> %1490, i64 3, !dbg !20
  %1492 = extractelement <4 x half> %1490, i64 2, !dbg !20
  %1493 = extractelement <4 x half> %1490, i64 1, !dbg !20
  %1494 = extractelement <4 x half> %1490, i64 0, !dbg !20
  %1495 = bitcast i32 %1119 to float, !dbg !15
  %1496 = select i1 %.not18, float %1109, float %1495, !dbg !15
  %1497 = insertelement <4 x float> poison, float %1496, i64 0, !dbg !19
  %1498 = bitcast i32 %1122 to float, !dbg !15
  %1499 = select i1 %.not18, float %1110, float %1498, !dbg !15
  %1500 = insertelement <4 x float> %1497, float %1499, i64 1, !dbg !19
  %1501 = select i1 %.not18, float %1495, float %1115, !dbg !15
  %1502 = insertelement <4 x float> %1500, float %1501, i64 2, !dbg !19
  %1503 = select i1 %.not18, float %1498, float %1116, !dbg !15
  %1504 = insertelement <4 x float> %1502, float %1503, i64 3, !dbg !19
  %normalized57 = fmul <4 x float> %1504, %row.reciprocal55, !dbg !19
  %1505 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !20
  %1506 = extractelement <4 x half> %1505, i64 3, !dbg !20
  %1507 = extractelement <4 x half> %1505, i64 2, !dbg !20
  %1508 = extractelement <4 x half> %1505, i64 1, !dbg !20
  %1509 = extractelement <4 x half> %1505, i64 0, !dbg !20
  %1510 = add nsw i64 %15, %25, !dbg !21
  %1511 = shl i64 %1510, 19, !dbg !21
  %1512 = and i64 %1511, 16252928, !dbg !21
  %address.thread474 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local475 = zext nneg i32 %address.thread474 to i64, !dbg !12
  %address.local476 = add i64 %28, %address.local475, !dbg !21
  %address.local477 = lshr i64 %address.local475, 6, !dbg !12
  %address.local478 = shl nuw nsw i64 %address.local477, 4, !dbg !21
  %address.local479 = add i64 %address.local476, %address.local478, !dbg !21
  %address.local480 = and i64 %address.local475, 240, !dbg !21
  %address.local481 = sub i64 %address.local479, %address.local480, !dbg !21
  %address.local482 = add i64 %address.local481, %29, !dbg !21
  %address.local483 = shl nuw nsw i64 %address.local482, 7, !dbg !21
  %address.local484 = add nuw nsw i64 %1512, %address.local483, !dbg !21
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local486 = lshr i64 %address.local475, 2, !dbg !12
  %address.local487 = and i64 %address.local486, 12, !dbg !12
  %address.local488 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local485, i64 %address.local487, !dbg !21
  store half %1509, ptr addrspace(1) %address.local488, align 2, !dbg !21
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local490 = or disjoint i64 %address.local487, 1, !dbg !21
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local490, !dbg !21
  store half %1508, ptr addrspace(1) %address.local491, align 2, !dbg !21
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local493 = or disjoint i64 %address.local487, 2, !dbg !21
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local492, i64 %address.local493, !dbg !21
  store half %1507, ptr addrspace(1) %address.local494, align 2, !dbg !21
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local496 = or disjoint i64 %address.local487, 3, !dbg !21
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local495, i64 %address.local496, !dbg !21
  store half %1506, ptr addrspace(1) %address.local497, align 2, !dbg !21
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local499 = or disjoint i64 %address.local487, 16, !dbg !21
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local498, i64 %address.local499, !dbg !21
  store half %1494, ptr addrspace(1) %address.local500, align 2, !dbg !21
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local502 = or disjoint i64 %address.local487, 17, !dbg !21
  %address.local503 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local501, i64 %address.local502, !dbg !21
  store half %1493, ptr addrspace(1) %address.local503, align 2, !dbg !21
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local505 = or disjoint i64 %address.local487, 18, !dbg !21
  %address.local506 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local504, i64 %address.local505, !dbg !21
  store half %1492, ptr addrspace(1) %address.local506, align 2, !dbg !21
  %address.local507 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !21
  %address.local508 = or disjoint i64 %address.local487, 19, !dbg !21
  %address.local509 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local507, i64 %address.local508, !dbg !21
  store half %1491, ptr addrspace(1) %address.local509, align 2, !dbg !21
  %address.local510 = shl i64 %address.local482, 7, !dbg !21
  %address.local511 = add i64 %address.local510, 8192, !dbg !21
  %address.local512 = add nuw nsw i64 %1512, %address.local511, !dbg !21
  %address.local513 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local514 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local513, i64 %address.local487, !dbg !21
  store half %1479, ptr addrspace(1) %address.local514, align 2, !dbg !21
  %address.local515 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local515, i64 %address.local490, !dbg !21
  store half %1478, ptr addrspace(1) %address.local516, align 2, !dbg !21
  %address.local517 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local517, i64 %address.local493, !dbg !21
  store half %1477, ptr addrspace(1) %address.local518, align 2, !dbg !21
  %address.local519 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local520 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local519, i64 %address.local496, !dbg !21
  store half %1476, ptr addrspace(1) %address.local520, align 2, !dbg !21
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local522 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local521, i64 %address.local499, !dbg !21
  store half %1464, ptr addrspace(1) %address.local522, align 2, !dbg !21
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local523, i64 %address.local502, !dbg !21
  store half %1463, ptr addrspace(1) %address.local524, align 2, !dbg !21
  %address.local525 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local525, i64 %address.local505, !dbg !21
  store half %1462, ptr addrspace(1) %address.local526, align 2, !dbg !21
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local528 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local527, i64 %address.local508, !dbg !21
  store half %1461, ptr addrspace(1) %address.local528, align 2, !dbg !21
  br label %1513, !dbg !21

1513:                                             ; preds = %1449, %998
  br i1 true, label %1514, label %1552, !dbg !21

1514:                                             ; preds = %1513
  %1515 = insertelement <4 x float> poison, float %1419, i64 0, !dbg !19
  %1516 = insertelement <4 x float> %1515, float %1425, i64 1, !dbg !19
  %1517 = insertelement <4 x float> %1516, float %1420, i64 2, !dbg !19
  %1518 = insertelement <4 x float> %1517, float %1426, i64 3, !dbg !19
  %row.reciprocal58 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1342, !dbg !19
  %normalized59 = fmul <4 x float> %1518, %row.reciprocal58, !dbg !19
  %1519 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !20
  %1520 = extractelement <4 x half> %1519, i64 3, !dbg !20
  %1521 = extractelement <4 x half> %1519, i64 2, !dbg !20
  %1522 = extractelement <4 x half> %1519, i64 1, !dbg !20
  %1523 = extractelement <4 x half> %1519, i64 0, !dbg !20
  %1524 = insertelement <4 x float> poison, float %1310, i64 0, !dbg !19
  %1525 = insertelement <4 x float> %1524, float %1316, i64 1, !dbg !19
  %1526 = insertelement <4 x float> %1525, float %1311, i64 2, !dbg !19
  %1527 = insertelement <4 x float> %1526, float %1317, i64 3, !dbg !19
  %normalized60 = fmul <4 x float> %1527, %row.reciprocal58, !dbg !19
  %1528 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !20
  %1529 = extractelement <4 x half> %1528, i64 3, !dbg !20
  %1530 = extractelement <4 x half> %1528, i64 2, !dbg !20
  %1531 = extractelement <4 x half> %1528, i64 1, !dbg !20
  %1532 = extractelement <4 x half> %1528, i64 0, !dbg !20
  %1533 = insertelement <4 x float> poison, float %1201, i64 0, !dbg !19
  %1534 = insertelement <4 x float> %1533, float %1207, i64 1, !dbg !19
  %1535 = insertelement <4 x float> %1534, float %1202, i64 2, !dbg !19
  %1536 = insertelement <4 x float> %1535, float %1208, i64 3, !dbg !19
  %row.reciprocal61 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1124, !dbg !19
  %normalized62 = fmul <4 x float> %1536, %row.reciprocal61, !dbg !19
  %1537 = fptrunc <4 x float> %normalized62 to <4 x half>, !dbg !20
  %1538 = extractelement <4 x half> %1537, i64 3, !dbg !20
  %1539 = extractelement <4 x half> %1537, i64 2, !dbg !20
  %1540 = extractelement <4 x half> %1537, i64 1, !dbg !20
  %1541 = extractelement <4 x half> %1537, i64 0, !dbg !20
  %1542 = insertelement <4 x float> poison, float %1092, i64 0, !dbg !19
  %1543 = insertelement <4 x float> %1542, float %1098, i64 1, !dbg !19
  %1544 = insertelement <4 x float> %1543, float %1093, i64 2, !dbg !19
  %1545 = insertelement <4 x float> %1544, float %1099, i64 3, !dbg !19
  %normalized63 = fmul <4 x float> %1545, %row.reciprocal61, !dbg !19
  %1546 = fptrunc <4 x float> %normalized63 to <4 x half>, !dbg !20
  %1547 = extractelement <4 x half> %1546, i64 3, !dbg !20
  %1548 = extractelement <4 x half> %1546, i64 2, !dbg !20
  %1549 = extractelement <4 x half> %1546, i64 1, !dbg !20
  %1550 = extractelement <4 x half> %1546, i64 0, !dbg !20
  %1551 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread529 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local530 = zext nneg i32 %address.thread529 to i64, !dbg !12
  %address.local531 = add i64 %28, %address.local530, !dbg !21
  %address.local532 = lshr i64 %address.local530, 6, !dbg !12
  %address.local533 = shl nuw nsw i64 %address.local532, 4, !dbg !21
  %address.local534 = add i64 %address.local531, %address.local533, !dbg !21
  %address.local535 = and i64 %address.local530, 240, !dbg !21
  %address.local536 = sub i64 %address.local534, %address.local535, !dbg !21
  %address.local537 = add i64 %address.local536, %29, !dbg !21
  %address.local538 = shl nuw nsw i64 %address.local537, 7, !dbg !21
  %address.local539 = add nuw nsw i64 %1551, %address.local538, !dbg !21
  %address.local540 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local541 = lshr i64 %address.local530, 2, !dbg !12
  %address.local542 = and i64 %address.local541, 12, !dbg !12
  %address.local543 = or disjoint i64 %address.local542, 32, !dbg !21
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local540, i64 %address.local543, !dbg !21
  store half %1550, ptr addrspace(1) %address.local544, align 2, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local546 = or disjoint i64 %address.local542, 33, !dbg !21
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local546, !dbg !21
  store half %1549, ptr addrspace(1) %address.local547, align 2, !dbg !21
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local549 = or disjoint i64 %address.local542, 34, !dbg !21
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local548, i64 %address.local549, !dbg !21
  store half %1548, ptr addrspace(1) %address.local550, align 2, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local552 = or disjoint i64 %address.local542, 35, !dbg !21
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local551, i64 %address.local552, !dbg !21
  store half %1547, ptr addrspace(1) %address.local553, align 2, !dbg !21
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local555 = or disjoint i64 %address.local542, 48, !dbg !21
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local554, i64 %address.local555, !dbg !21
  store half %1541, ptr addrspace(1) %address.local556, align 2, !dbg !21
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local558 = or disjoint i64 %address.local542, 49, !dbg !21
  %address.local559 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local557, i64 %address.local558, !dbg !21
  store half %1540, ptr addrspace(1) %address.local559, align 2, !dbg !21
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local561 = or disjoint i64 %address.local542, 50, !dbg !21
  %address.local562 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local560, i64 %address.local561, !dbg !21
  store half %1539, ptr addrspace(1) %address.local562, align 2, !dbg !21
  %address.local563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local564 = or i64 %address.local541, 51, !dbg !21
  %address.local565 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local563, i64 %address.local564, !dbg !21
  store half %1538, ptr addrspace(1) %address.local565, align 2, !dbg !21
  %address.local566 = shl i64 %address.local537, 7, !dbg !21
  %address.local567 = add i64 %address.local566, 8192, !dbg !21
  %address.local568 = add nuw nsw i64 %1551, %address.local567, !dbg !21
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local570 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local569, i64 %address.local543, !dbg !21
  store half %1532, ptr addrspace(1) %address.local570, align 2, !dbg !21
  %address.local571 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local571, i64 %address.local546, !dbg !21
  store half %1531, ptr addrspace(1) %address.local572, align 2, !dbg !21
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local573, i64 %address.local549, !dbg !21
  store half %1530, ptr addrspace(1) %address.local574, align 2, !dbg !21
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local552, !dbg !21
  store half %1529, ptr addrspace(1) %address.local576, align 2, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local555, !dbg !21
  store half %1523, ptr addrspace(1) %address.local578, align 2, !dbg !21
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local579, i64 %address.local558, !dbg !21
  store half %1522, ptr addrspace(1) %address.local580, align 2, !dbg !21
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local561, !dbg !21
  store half %1521, ptr addrspace(1) %address.local582, align 2, !dbg !21
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local564, !dbg !21
  store half %1520, ptr addrspace(1) %address.local584, align 2, !dbg !21
  br label %1552, !dbg !21

1552:                                             ; preds = %1514, %1513
  br i1 true, label %1553, label %1591, !dbg !21

1553:                                             ; preds = %1552
  %1554 = insertelement <4 x float> poison, float %1391, i64 0, !dbg !19
  %1555 = insertelement <4 x float> %1554, float %1397, i64 1, !dbg !19
  %1556 = insertelement <4 x float> %1555, float %1392, i64 2, !dbg !19
  %1557 = insertelement <4 x float> %1556, float %1398, i64 3, !dbg !19
  %row.reciprocal64 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1342, !dbg !19
  %normalized65 = fmul <4 x float> %1557, %row.reciprocal64, !dbg !19
  %1558 = fptrunc <4 x float> %normalized65 to <4 x half>, !dbg !20
  %1559 = extractelement <4 x half> %1558, i64 3, !dbg !20
  %1560 = extractelement <4 x half> %1558, i64 2, !dbg !20
  %1561 = extractelement <4 x half> %1558, i64 1, !dbg !20
  %1562 = extractelement <4 x half> %1558, i64 0, !dbg !20
  %1563 = insertelement <4 x float> poison, float %1281, i64 0, !dbg !19
  %1564 = insertelement <4 x float> %1563, float %1287, i64 1, !dbg !19
  %1565 = insertelement <4 x float> %1564, float %1282, i64 2, !dbg !19
  %1566 = insertelement <4 x float> %1565, float %1288, i64 3, !dbg !19
  %normalized66 = fmul <4 x float> %1566, %row.reciprocal64, !dbg !19
  %1567 = fptrunc <4 x float> %normalized66 to <4 x half>, !dbg !20
  %1568 = extractelement <4 x half> %1567, i64 3, !dbg !20
  %1569 = extractelement <4 x half> %1567, i64 2, !dbg !20
  %1570 = extractelement <4 x half> %1567, i64 1, !dbg !20
  %1571 = extractelement <4 x half> %1567, i64 0, !dbg !20
  %1572 = insertelement <4 x float> poison, float %1173, i64 0, !dbg !19
  %1573 = insertelement <4 x float> %1572, float %1179, i64 1, !dbg !19
  %1574 = insertelement <4 x float> %1573, float %1174, i64 2, !dbg !19
  %1575 = insertelement <4 x float> %1574, float %1180, i64 3, !dbg !19
  %row.reciprocal67 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1124, !dbg !19
  %normalized68 = fmul <4 x float> %1575, %row.reciprocal67, !dbg !19
  %1576 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !20
  %1577 = extractelement <4 x half> %1576, i64 3, !dbg !20
  %1578 = extractelement <4 x half> %1576, i64 2, !dbg !20
  %1579 = extractelement <4 x half> %1576, i64 1, !dbg !20
  %1580 = extractelement <4 x half> %1576, i64 0, !dbg !20
  %1581 = insertelement <4 x float> poison, float %1063, i64 0, !dbg !19
  %1582 = insertelement <4 x float> %1581, float %1069, i64 1, !dbg !19
  %1583 = insertelement <4 x float> %1582, float %1064, i64 2, !dbg !19
  %1584 = insertelement <4 x float> %1583, float %1070, i64 3, !dbg !19
  %normalized69 = fmul <4 x float> %1584, %row.reciprocal67, !dbg !19
  %1585 = fptrunc <4 x float> %normalized69 to <4 x half>, !dbg !20
  %1586 = extractelement <4 x half> %1585, i64 3, !dbg !20
  %1587 = extractelement <4 x half> %1585, i64 2, !dbg !20
  %1588 = extractelement <4 x half> %1585, i64 1, !dbg !20
  %1589 = extractelement <4 x half> %1585, i64 0, !dbg !20
  %1590 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread585 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local586 = zext nneg i32 %address.thread585 to i64, !dbg !12
  %address.local587 = add i64 %28, %address.local586, !dbg !21
  %address.local588 = lshr i64 %address.local586, 6, !dbg !12
  %address.local589 = shl nuw nsw i64 %address.local588, 4, !dbg !21
  %address.local590 = add i64 %address.local587, %address.local589, !dbg !21
  %address.local591 = and i64 %address.local586, 240, !dbg !21
  %address.local592 = sub i64 %address.local590, %address.local591, !dbg !21
  %address.local593 = add i64 %address.local592, %29, !dbg !21
  %address.local594 = shl nuw nsw i64 %address.local593, 7, !dbg !21
  %address.local595 = add nuw nsw i64 %1590, %address.local594, !dbg !21
  %address.local596 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local597 = lshr i64 %address.local586, 2, !dbg !12
  %address.local598 = and i64 %address.local597, 12, !dbg !12
  %address.local599 = or disjoint i64 %address.local598, 64, !dbg !21
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local596, i64 %address.local599, !dbg !21
  store half %1589, ptr addrspace(1) %address.local600, align 2, !dbg !21
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local602 = or disjoint i64 %address.local598, 65, !dbg !21
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local601, i64 %address.local602, !dbg !21
  store half %1588, ptr addrspace(1) %address.local603, align 2, !dbg !21
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local605 = or disjoint i64 %address.local598, 66, !dbg !21
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local604, i64 %address.local605, !dbg !21
  store half %1587, ptr addrspace(1) %address.local606, align 2, !dbg !21
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local608 = or disjoint i64 %address.local598, 67, !dbg !21
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local607, i64 %address.local608, !dbg !21
  store half %1586, ptr addrspace(1) %address.local609, align 2, !dbg !21
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local611 = or disjoint i64 %address.local598, 80, !dbg !21
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local610, i64 %address.local611, !dbg !21
  store half %1580, ptr addrspace(1) %address.local612, align 2, !dbg !21
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local614 = or disjoint i64 %address.local598, 81, !dbg !21
  %address.local615 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local613, i64 %address.local614, !dbg !21
  store half %1579, ptr addrspace(1) %address.local615, align 2, !dbg !21
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local617 = or disjoint i64 %address.local598, 82, !dbg !21
  %address.local618 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local616, i64 %address.local617, !dbg !21
  store half %1578, ptr addrspace(1) %address.local618, align 2, !dbg !21
  %address.local619 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local620 = or disjoint i64 %address.local598, 83, !dbg !21
  %address.local621 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local619, i64 %address.local620, !dbg !21
  store half %1577, ptr addrspace(1) %address.local621, align 2, !dbg !21
  %address.local622 = shl i64 %address.local593, 7, !dbg !21
  %address.local623 = add i64 %address.local622, 8192, !dbg !21
  %address.local624 = add nuw nsw i64 %1590, %address.local623, !dbg !21
  %address.local625 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !21
  %address.local626 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local625, i64 %address.local599, !dbg !21
  store half %1571, ptr addrspace(1) %address.local626, align 2, !dbg !21
  %address.local627 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !21
  %address.local628 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local627, i64 %address.local602, !dbg !21
  store half %1570, ptr addrspace(1) %address.local628, align 2, !dbg !21
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !21
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local629, i64 %address.local605, !dbg !21
  store half %1569, ptr addrspace(1) %address.local630, align 2, !dbg !21
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !21
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local608, !dbg !21
  store half %1568, ptr addrspace(1) %address.local632, align 2, !dbg !21
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !21
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local633, i64 %address.local611, !dbg !21
  store half %1562, ptr addrspace(1) %address.local634, align 2, !dbg !21
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !21
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local635, i64 %address.local614, !dbg !21
  store half %1561, ptr addrspace(1) %address.local636, align 2, !dbg !21
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !21
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local637, i64 %address.local617, !dbg !21
  store half %1560, ptr addrspace(1) %address.local638, align 2, !dbg !21
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !21
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local639, i64 %address.local620, !dbg !21
  store half %1559, ptr addrspace(1) %address.local640, align 2, !dbg !21
  br label %1591, !dbg !21

1591:                                             ; preds = %1553, %1552
  br i1 true, label %1592, label %1630, !dbg !21

1592:                                             ; preds = %1591
  %1593 = insertelement <4 x float> poison, float %1363, i64 0, !dbg !19
  %1594 = insertelement <4 x float> %1593, float %1369, i64 1, !dbg !19
  %1595 = insertelement <4 x float> %1594, float %1364, i64 2, !dbg !19
  %1596 = insertelement <4 x float> %1595, float %1370, i64 3, !dbg !19
  %row.reciprocal70 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1342, !dbg !19
  %normalized71 = fmul <4 x float> %1596, %row.reciprocal70, !dbg !19
  %1597 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !20
  %1598 = extractelement <4 x half> %1597, i64 3, !dbg !20
  %1599 = extractelement <4 x half> %1597, i64 2, !dbg !20
  %1600 = extractelement <4 x half> %1597, i64 1, !dbg !20
  %1601 = extractelement <4 x half> %1597, i64 0, !dbg !20
  %1602 = insertelement <4 x float> poison, float %1252, i64 0, !dbg !19
  %1603 = insertelement <4 x float> %1602, float %1258, i64 1, !dbg !19
  %1604 = insertelement <4 x float> %1603, float %1253, i64 2, !dbg !19
  %1605 = insertelement <4 x float> %1604, float %1259, i64 3, !dbg !19
  %normalized72 = fmul <4 x float> %1605, %row.reciprocal70, !dbg !19
  %1606 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !20
  %1607 = extractelement <4 x half> %1606, i64 3, !dbg !20
  %1608 = extractelement <4 x half> %1606, i64 2, !dbg !20
  %1609 = extractelement <4 x half> %1606, i64 1, !dbg !20
  %1610 = extractelement <4 x half> %1606, i64 0, !dbg !20
  %1611 = insertelement <4 x float> poison, float %1145, i64 0, !dbg !19
  %1612 = insertelement <4 x float> %1611, float %1151, i64 1, !dbg !19
  %1613 = insertelement <4 x float> %1612, float %1146, i64 2, !dbg !19
  %1614 = insertelement <4 x float> %1613, float %1152, i64 3, !dbg !19
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1124, !dbg !19
  %normalized74 = fmul <4 x float> %1614, %row.reciprocal73, !dbg !19
  %1615 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %1616 = extractelement <4 x half> %1615, i64 3, !dbg !20
  %1617 = extractelement <4 x half> %1615, i64 2, !dbg !20
  %1618 = extractelement <4 x half> %1615, i64 1, !dbg !20
  %1619 = extractelement <4 x half> %1615, i64 0, !dbg !20
  %1620 = insertelement <4 x float> poison, float %1034, i64 0, !dbg !19
  %1621 = insertelement <4 x float> %1620, float %1040, i64 1, !dbg !19
  %1622 = insertelement <4 x float> %1621, float %1035, i64 2, !dbg !19
  %1623 = insertelement <4 x float> %1622, float %1041, i64 3, !dbg !19
  %normalized75 = fmul <4 x float> %1623, %row.reciprocal73, !dbg !19
  %1624 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !20
  %1625 = extractelement <4 x half> %1624, i64 3, !dbg !20
  %1626 = extractelement <4 x half> %1624, i64 2, !dbg !20
  %1627 = extractelement <4 x half> %1624, i64 1, !dbg !20
  %1628 = extractelement <4 x half> %1624, i64 0, !dbg !20
  %1629 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread641 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local642 = zext nneg i32 %address.thread641 to i64, !dbg !12
  %address.local643 = add i64 %28, %address.local642, !dbg !21
  %address.local644 = lshr i64 %address.local642, 6, !dbg !12
  %address.local645 = shl nuw nsw i64 %address.local644, 4, !dbg !21
  %address.local646 = add i64 %address.local643, %address.local645, !dbg !21
  %address.local647 = and i64 %address.local642, 240, !dbg !21
  %address.local648 = sub i64 %address.local646, %address.local647, !dbg !21
  %address.local649 = add i64 %address.local648, %29, !dbg !21
  %address.local650 = shl nuw nsw i64 %address.local649, 7, !dbg !21
  %address.local651 = add nuw nsw i64 %1629, %address.local650, !dbg !21
  %address.local652 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !21
  %address.local653 = lshr i64 %address.local642, 2, !dbg !12
  %address.local654 = and i64 %address.local653, 12, !dbg !12
  %address.local655 = or disjoint i64 %address.local654, 96, !dbg !21
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local652, i64 %address.local655, !dbg !21
  store half %1628, ptr addrspace(1) %address.local656, align 2, !dbg !21
  %address.local657 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !21
  %address.local658 = or disjoint i64 %address.local654, 97, !dbg !21
  %address.local659 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local657, i64 %address.local658, !dbg !21
  store half %1627, ptr addrspace(1) %address.local659, align 2, !dbg !21
  %address.local660 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !21
  %address.local661 = or disjoint i64 %address.local654, 98, !dbg !21
  %address.local662 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local660, i64 %address.local661, !dbg !21
  store half %1626, ptr addrspace(1) %address.local662, align 2, !dbg !21
  %address.local663 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !21
  %address.local664 = or disjoint i64 %address.local654, 99, !dbg !21
  %address.local665 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local663, i64 %address.local664, !dbg !21
  store half %1625, ptr addrspace(1) %address.local665, align 2, !dbg !21
  %address.local666 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !21
  %address.local667 = or disjoint i64 %address.local654, 112, !dbg !21
  %address.local668 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local666, i64 %address.local667, !dbg !21
  store half %1619, ptr addrspace(1) %address.local668, align 2, !dbg !21
  %address.local669 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !21
  %address.local670 = or disjoint i64 %address.local654, 113, !dbg !21
  %address.local671 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local669, i64 %address.local670, !dbg !21
  store half %1618, ptr addrspace(1) %address.local671, align 2, !dbg !21
  %address.local672 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !21
  %address.local673 = or disjoint i64 %address.local654, 114, !dbg !21
  %address.local674 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local672, i64 %address.local673, !dbg !21
  store half %1617, ptr addrspace(1) %address.local674, align 2, !dbg !21
  %address.local675 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !21
  %address.local676 = or i64 %address.local653, 115, !dbg !21
  %address.local677 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local675, i64 %address.local676, !dbg !21
  store half %1616, ptr addrspace(1) %address.local677, align 2, !dbg !21
  %address.local678 = shl i64 %address.local649, 7, !dbg !21
  %address.local679 = add i64 %address.local678, 8192, !dbg !21
  %address.local680 = add nuw nsw i64 %1629, %address.local679, !dbg !21
  %address.local681 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !21
  %address.local682 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local681, i64 %address.local655, !dbg !21
  store half %1610, ptr addrspace(1) %address.local682, align 2, !dbg !21
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !21
  %address.local684 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local683, i64 %address.local658, !dbg !21
  store half %1609, ptr addrspace(1) %address.local684, align 2, !dbg !21
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !21
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local685, i64 %address.local661, !dbg !21
  store half %1608, ptr addrspace(1) %address.local686, align 2, !dbg !21
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !21
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local687, i64 %address.local664, !dbg !21
  store half %1607, ptr addrspace(1) %address.local688, align 2, !dbg !21
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !21
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local689, i64 %address.local667, !dbg !21
  store half %1601, ptr addrspace(1) %address.local690, align 2, !dbg !21
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !21
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local691, i64 %address.local670, !dbg !21
  store half %1600, ptr addrspace(1) %address.local692, align 2, !dbg !21
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !21
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local693, i64 %address.local673, !dbg !21
  store half %1599, ptr addrspace(1) %address.local694, align 2, !dbg !21
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !21
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local695, i64 %address.local676, !dbg !21
  store half %1598, ptr addrspace(1) %address.local696, align 2, !dbg !21
  br label %1630, !dbg !21

1630:                                             ; preds = %1592, %1591
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #6

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #7

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #6 = { nocallback nofree nounwind willreturn }
attributes #7 = { convergent nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!32}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/bm_bn_sweep/export/run-20261022-180820-lrul_y0l/input-0002-test_input_1x32x4096x128-128-64-32/group8-reverse-k128-swizzle")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!9 = !DILocation(line: 47, column: 17, scope: !4)
!11 = !DILocation(line: 67, column: 22, scope: !4)
!12 = !DILocation(line: 69, column: 24, scope: !4)
!13 = !DILocation(line: 74, column: 19, scope: !4)
!14 = !DILocation(line: 94, column: 23, scope: !4)
!15 = !DILocation(line: 57, column: 24, scope: !4)
!16 = !DILocation(line: 68, column: 22, scope: !4)
!17 = !DILocation(line: 70, column: 26, scope: !4)
!18 = !DILocation(line: 93, column: 20, scope: !4)
!19 = !DILocation(line: 98, column: 23, scope: !4)
!20 = !DILocation(line: 99, column: 19, scope: !4)
!21 = !DILocation(line: 100, column: 9, scope: !4)
!23 = !DILocation(line: 76, column: 24, scope: !4)
!24 = !DILocation(line: 77, column: 23, scope: !4)
!25 = !DILocation(line: 78, column: 26, scope: !4)
!26 = !DILocation(line: 85, column: 21, scope: !4)
!27 = !DILocation(line: 86, column: 22, scope: !4)
!28 = !DILocation(line: 87, column: 20, scope: !4)
!29 = !DILocation(line: 88, column: 16, scope: !4)
!30 = !DILocation(line: 95, column: 20, scope: !4)
!31 = !{i32 256, i32 1, i32 1}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{i32 0, i32 256}
