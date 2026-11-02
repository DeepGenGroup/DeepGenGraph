; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/online/session-20261026-101601-tehle7yk/tuning/run-20261026-101601-inpooz4_/input-0001-test_input_1x32x4096x128-128-64-32-online/grid-000007/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/online/session-20261026-101601-tehle7yk/tuning/run-20261026-101601-inpooz4_/input-0001-test_input_1x32x4096x128-128-64-32-online/grid-000007/input.mlir"

@shm_0 = addrspace(3) global [12288 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !36 {
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
  %34 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !38
  %35 = zext nneg i32 %34 to i64, !dbg !12
  %36 = lshr i64 %35, 6, !dbg !12
  %37 = or disjoint i64 %19, 127, !dbg !13
  %38 = add i64 %19, 128, !dbg !13
  %39 = icmp uge i64 %38, %37, !dbg !13
  %40 = shl nsw i64 %16, 1, !dbg !7
  %41 = add nsw i64 %40, 2, !dbg !7
  %42 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %43 = lshr i64 %35, 2, !dbg !16
  %44 = shl nuw nsw i64 %35, 2, !dbg !16
  %45 = mul nsw i64 %43, -16, !dbg !16
  %46 = add nsw i64 %44, %45, !dbg !16
  %47 = and i32 %34, 1, !dbg !16
  %index.compact142 = zext nneg i32 %47 to i64, !dbg !16
  %48 = or disjoint i64 %46, %index.compact142, !dbg !16
  %49 = add nuw nsw i64 %48, 1, !dbg !16
  %50 = sdiv i64 %49, 2, !dbg !16
  %51 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %52 = getelementptr i8, ptr addrspace(1) %51, i64 64, !dbg !11
  %53 = shl i64 %15, 7, !dbg !15
  %54 = and i64 %53, 3968, !dbg !15
  %55 = shl nsw i64 %18, 12, !dbg !15
  %56 = or disjoint i64 %54, %55, !dbg !15
  %57 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %58 = getelementptr i8, ptr addrspace(1) %57, i64 128, !dbg !11
  %59 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %60 = getelementptr i8, ptr addrspace(1) %59, i64 192, !dbg !11
  br label %61, !dbg !7

61:                                               ; preds = %587, %4
  %62 = phi i64 [ %1124, %587 ], [ 0, %4 ], !dbg !7
  %63 = phi [2 x <8 x float>] [ %1109, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %64 = phi [2 x <1 x float>] [ %885, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %65 = phi [2 x <1 x float>] [ %1111, %587 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %66 = phi [2 x <8 x float>] [ %1115, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %67 = phi [2 x <8 x float>] [ %1119, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %68 = phi [2 x <8 x float>] [ %1123, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %69 = icmp slt i64 %62, %41, !dbg !7
  br i1 %69, label %70, label %1125, !dbg !7

70:                                               ; preds = %61
  %71 = shl nsw i64 %62, 6, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !38, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local173 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local174 = and i64 %address.local173, 1536, !dbg !11
  %address.local175 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local174, !dbg !11
  %address.local176 = ptrtoint ptr addrspace(3) %address.local175 to i64, !dbg !11
  %address.local177 = trunc i64 %address.local176 to i32, !dbg !11
  %72 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local177), !dbg !11
  %73 = zext i32 %72 to i64, !dbg !11
  %74 = inttoptr i64 %73 to ptr addrspace(3), !dbg !11
  %address.local178 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local179 = and i64 %address.local178, 16256, !dbg !11
  %address.local180 = lshr i64 %address.local, 4, !dbg !11
  %address.local181 = and i64 %address.local180, 8, !dbg !11
  %address.local182 = or disjoint i64 %address.local179, %address.local181, !dbg !11
  %address.local183 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %address.local182, !dbg !11
  %75 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %76 = ptrtoint ptr addrspace(1) %address.local183 to i64, !dbg !11
  %77 = sub i64 %76, %75, !dbg !11
  %dma.buffer.offset926 = trunc i64 %77 to i32, !dbg !11
  %78 = trunc i64 %75 to i32, !dbg !11
  %79 = insertelement <4 x i32> poison, i32 %78, i64 0, !dbg !11
  %80 = lshr i64 %75, 32, !dbg !11
  %81 = trunc i64 %80 to i32, !dbg !11
  %82 = insertelement <4 x i32> %79, i32 %81, i64 1, !dbg !11
  %83 = insertelement <4 x i32> %82, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc927 = insertelement <4 x i32> %83, i32 131072, i64 3, !dbg !11
  %84 = ptrtoint ptr addrspace(3) %74 to i32, !dbg !11
  %dma.buffer.lds928 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %84), !dbg !11
  %85 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds928, i32 %dma.buffer.offset926, <4 x i32> %dma.buffer.rsrc927) #10, !dbg !11
  %address.local184 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local185 = shl nuw nsw i64 %address.local184, 3, !dbg !11
  %address.local186 = and i64 %address.local185, 3584, !dbg !11
  %address.local187 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local186, !dbg !11
  %address.local188 = ptrtoint ptr addrspace(3) %address.local187 to i64, !dbg !11
  %address.local189 = trunc i64 %address.local188 to i32, !dbg !11
  %86 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local189), !dbg !11
  %87 = zext i32 %86 to i64, !dbg !11
  %88 = inttoptr i64 %87 to ptr addrspace(3), !dbg !11
  %address.local190 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local191 = and i64 %address.local190, 16256, !dbg !11
  %address.local192 = lshr i64 %address.local184, 4, !dbg !11
  %address.local193 = and i64 %address.local192, 24, !dbg !11
  %address.local194 = or disjoint i64 %address.local191, %address.local193, !dbg !11
  %address.local195 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %address.local194, !dbg !11
  %89 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %90 = ptrtoint ptr addrspace(1) %address.local195 to i64, !dbg !11
  %91 = sub i64 %90, %89, !dbg !11
  %dma.buffer.offset923 = trunc i64 %91 to i32, !dbg !11
  %92 = trunc i64 %89 to i32, !dbg !11
  %93 = insertelement <4 x i32> poison, i32 %92, i64 0, !dbg !11
  %94 = lshr i64 %89, 32, !dbg !11
  %95 = trunc i64 %94 to i32, !dbg !11
  %96 = insertelement <4 x i32> %93, i32 %95, i64 1, !dbg !11
  %97 = insertelement <4 x i32> %96, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc924 = insertelement <4 x i32> %97, i32 131072, i64 3, !dbg !11
  %98 = ptrtoint ptr addrspace(3) %88 to i32, !dbg !11
  %dma.buffer.lds925 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %98), !dbg !11
  %99 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds925, i32 %dma.buffer.offset923, <4 x i32> %dma.buffer.rsrc924) #10, !dbg !11
  %100 = lshr i64 %62, 13, !dbg !15
  %101 = add nuw nsw i64 %100, %15, !dbg !15
  %102 = lshr i64 %62, 6, !dbg !15
  %103 = and i64 %102, 127, !dbg !15
  %104 = shl i64 %101, 19, !dbg !15
  %105 = and i64 %104, 16252928, !dbg !15
  %106 = shl i64 %102, 19, !dbg !15
  %107 = shl i64 %62, 13, !dbg !15
  %108 = sub i64 %107, %106, !dbg !15
  %109 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %105, !dbg !15
  %110 = getelementptr [2 x i8], ptr addrspace(1) %109, i64 %103, !dbg !15
  %111 = getelementptr [2 x i8], ptr addrspace(1) %110, i64 %108, !dbg !15
  %address.local196 = and i64 %address.local173, 1536, !dbg !15
  %address.local197 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local196, !dbg !15
  %address.local198 = ptrtoint ptr addrspace(3) %address.local197 to i64, !dbg !15
  %address.local199 = trunc i64 %address.local198 to i32, !dbg !15
  %112 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local199), !dbg !15
  %113 = zext i32 %112 to i64, !dbg !15
  %114 = inttoptr i64 %113 to ptr addrspace(3), !dbg !15
  %address.local200 = lshr i64 %address.local, 3, !dbg !15
  %address.local201 = and i64 %address.local200, 24, !dbg !15
  %address.local202 = and i64 %address.local, 59, !dbg !15
  %address.local203 = lshr i64 %address.local, 2, !dbg !15
  %address.local204 = and i64 %address.local203, 4, !dbg !15
  %address.local205 = add nuw nsw i64 %address.local204, %address.local, !dbg !15
  %address.local206 = and i64 %address.local205, 4, !dbg !15
  %address.local207 = or disjoint i64 %address.local202, %address.local206, !dbg !15
  %address.local208 = shl nuw nsw i64 %address.local207, 7, !dbg !15
  %address.local209 = or disjoint i64 %address.local201, %address.local208, !dbg !15
  %address.local210 = getelementptr [2 x i8], ptr addrspace(1) %111, i64 %address.local209, !dbg !15
  %115 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %116 = ptrtoint ptr addrspace(1) %address.local210 to i64, !dbg !15
  %117 = sub i64 %116, %115, !dbg !15
  %dma.buffer.offset920 = trunc i64 %117 to i32, !dbg !15
  %118 = trunc i64 %115 to i32, !dbg !15
  %119 = insertelement <4 x i32> poison, i32 %118, i64 0, !dbg !15
  %120 = lshr i64 %115, 32, !dbg !15
  %121 = trunc i64 %120 to i32, !dbg !15
  %122 = insertelement <4 x i32> %119, i32 %121, i64 1, !dbg !15
  %123 = insertelement <4 x i32> %122, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc921 = insertelement <4 x i32> %123, i32 131072, i64 3, !dbg !15
  %124 = ptrtoint ptr addrspace(3) %114 to i32, !dbg !15
  %dma.buffer.lds922 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %124), !dbg !15
  %125 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds922, i32 %dma.buffer.offset920, <4 x i32> %dma.buffer.rsrc921) #10, !dbg !15
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
  %126 = load <4 x half>, ptr addrspace(3) %address.local223, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local223, i64 4096, !dbg !12
  %127 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %lds.const.gep825 = getelementptr i8, ptr addrspace(3) %address.local223, i64 1024, !dbg !12
  %128 = load <4 x half>, ptr addrspace(3) %lds.const.gep825, align 8, !dbg !12
  %129 = fmul <4 x half> %126, splat (half 0xH3015), !dbg !12
  %lds.const.gep826 = getelementptr i8, ptr addrspace(3) %address.local223, i64 5120, !dbg !12
  %130 = load <4 x half>, ptr addrspace(3) %lds.const.gep826, align 8, !dbg !12
  %131 = fmul <4 x half> %127, splat (half 0xH3015), !dbg !12
  %132 = fmul <4 x half> %128, splat (half 0xH3015), !dbg !12
  %133 = fmul <4 x half> %130, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local230 = mul nuw nsw i64 %address.local, 160, !dbg !16
  %address.local231 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local230, !dbg !16
  %address.local232 = lshr i64 %address.local, 2, !dbg !16
  %address.local233 = mul nsw i64 %address.local232, -1280, !dbg !16
  %address.local234 = getelementptr i8, ptr addrspace(3) %address.local231, i64 %address.local233, !dbg !16
  %address.local235 = shl nuw nsw i64 %address.local, 1, !dbg !16
  %address.local236 = and i64 %address.local235, 24, !dbg !16
  %address.local237 = getelementptr [2 x i8], ptr addrspace(3) %address.local234, i64 %address.local236, !dbg !16
  %address.local238 = shl i64 %address.local, 62, !dbg !16
  %address.local239 = ashr exact i64 %address.local238, 57, !dbg !16
  %address.local240 = and i64 %address.local239, -64, !dbg !16
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local237, i64 %address.local240, !dbg !16
  %address.local242 = lshr i64 %address.local, 2, !dbg !12
  %address.local243 = and i64 %address.local242, 12, !dbg !12
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local243, !dbg !16
  %address.local245 = and i64 %address.local, 63, !dbg !12
  %address.local246 = lshr i64 %address.local245, 5, !dbg !12
  %address.local247 = mul nuw nsw i64 %address.local246, 504, !dbg !16
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local244, i64 %address.local247, !dbg !16
  %134 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !16
  %135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %129, <4 x half> %134, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep827 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !16
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep827, align 8, !dbg !16
  %address.local259 = shl nuw nsw i32 %address.thread, 1, !dbg !16
  %address.local260 = and i32 %address.local259, 24, !dbg !16
  %address.local261 = shl nuw nsw i32 %address.thread, 8, !dbg !16
  %address.local262 = and i32 %address.local261, 768, !dbg !16
  %address.local263 = or disjoint i32 %address.local260, %address.local262, !dbg !16
  %address.local264 = shl nuw nsw i32 %address.thread, 5, !dbg !16
  %address.local265 = and i32 %address.local264, 96, !dbg !16
  %address.local266 = or disjoint i32 %address.local263, %address.local265, !dbg !16
  %address.local267 = zext nneg i32 %address.local266 to i64, !dbg !16
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %50) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local268 = lshr i32 %address.thread, 1, !dbg !16
  %address.local269 = and i32 %address.local268, 1, !dbg !16
  %address.local270 = zext nneg i32 %address.local269 to i64, !dbg !16
  %address.local271 = add nsw i64 %address.base, %address.local270, !dbg !16
  %address.local272 = shl i64 %address.local271, 6, !dbg !16
  %address.local273 = sub i64 %address.local267, %address.local272, !dbg !16
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local273, !dbg !16
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) %address.local274, i64 %address.local243, !dbg !16
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) %address.local275, i64 %address.local247, !dbg !16
  %address.local277 = getelementptr i8, ptr addrspace(3) %address.local276, i64 64, !dbg !16
  %137 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !16
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %136, <4 x float> %135) #9, !dbg !16
  %139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %129, <4 x half> %137, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep828 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !16
  %140 = load <4 x half>, ptr addrspace(3) %lds.const.gep828, align 8, !dbg !16
  %lds.const.gep829 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !16
  %141 = load <4 x half>, ptr addrspace(3) %lds.const.gep829, align 8, !dbg !16
  %142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %140, <4 x float> %139) #9, !dbg !16
  %143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %129, <4 x half> %141, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep830 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !16
  %144 = load <4 x half>, ptr addrspace(3) %lds.const.gep830, align 8, !dbg !16
  %lds.const.gep831 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !16
  %145 = load <4 x half>, ptr addrspace(3) %lds.const.gep831, align 8, !dbg !16
  %146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %144, <4 x float> %143) #9, !dbg !16
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %129, <4 x half> %145, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep832 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !16
  %148 = load <4 x half>, ptr addrspace(3) %lds.const.gep832, align 8, !dbg !16
  %149 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !16
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %131, <4 x half> %148, <4 x float> %147) #9, !dbg !16
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %149, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep833 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !16
  %152 = load <4 x half>, ptr addrspace(3) %lds.const.gep833, align 8, !dbg !16
  %153 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !16
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %152, <4 x float> %151) #9, !dbg !16
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %153, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep834 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !16
  %156 = load <4 x half>, ptr addrspace(3) %lds.const.gep834, align 8, !dbg !16
  %lds.const.gep835 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !16
  %157 = load <4 x half>, ptr addrspace(3) %lds.const.gep835, align 8, !dbg !16
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %156, <4 x float> %155) #9, !dbg !16
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %157, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep836 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !16
  %160 = load <4 x half>, ptr addrspace(3) %lds.const.gep836, align 8, !dbg !16
  %lds.const.gep837 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !16
  %161 = load <4 x half>, ptr addrspace(3) %lds.const.gep837, align 8, !dbg !16
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %160, <4 x float> %159) #9, !dbg !16
  %lds.const.gep838 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !16
  %163 = load <4 x half>, ptr addrspace(3) %lds.const.gep838, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %161, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %163, <4 x float> %164) #9, !dbg !16
  %166 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local177), !dbg !11
  %167 = zext i32 %166 to i64, !dbg !11
  %168 = inttoptr i64 %167 to ptr addrspace(3), !dbg !11
  %address.local418 = getelementptr [2 x i8], ptr addrspace(1) %52, i64 %address.local182, !dbg !11
  %169 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %170 = ptrtoint ptr addrspace(1) %address.local418 to i64, !dbg !11
  %171 = sub i64 %170, %169, !dbg !11
  %dma.buffer.offset917 = trunc i64 %171 to i32, !dbg !11
  %172 = trunc i64 %169 to i32, !dbg !11
  %173 = insertelement <4 x i32> poison, i32 %172, i64 0, !dbg !11
  %174 = lshr i64 %169, 32, !dbg !11
  %175 = trunc i64 %174 to i32, !dbg !11
  %176 = insertelement <4 x i32> %173, i32 %175, i64 1, !dbg !11
  %177 = insertelement <4 x i32> %176, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc918 = insertelement <4 x i32> %177, i32 131072, i64 3, !dbg !11
  %178 = ptrtoint ptr addrspace(3) %168 to i32, !dbg !11
  %dma.buffer.lds919 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %178), !dbg !11
  %179 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds919, i32 %dma.buffer.offset917, <4 x i32> %dma.buffer.rsrc918) #10, !dbg !11
  %180 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local189), !dbg !11
  %181 = zext i32 %180 to i64, !dbg !11
  %182 = inttoptr i64 %181 to ptr addrspace(3), !dbg !11
  %address.local419 = getelementptr [2 x i8], ptr addrspace(1) %52, i64 %address.local194, !dbg !11
  %183 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %184 = ptrtoint ptr addrspace(1) %address.local419 to i64, !dbg !11
  %185 = sub i64 %184, %183, !dbg !11
  %dma.buffer.offset914 = trunc i64 %185 to i32, !dbg !11
  %186 = trunc i64 %183 to i32, !dbg !11
  %187 = insertelement <4 x i32> poison, i32 %186, i64 0, !dbg !11
  %188 = lshr i64 %183, 32, !dbg !11
  %189 = trunc i64 %188 to i32, !dbg !11
  %190 = insertelement <4 x i32> %187, i32 %189, i64 1, !dbg !11
  %191 = insertelement <4 x i32> %190, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc915 = insertelement <4 x i32> %191, i32 131072, i64 3, !dbg !11
  %192 = ptrtoint ptr addrspace(3) %182 to i32, !dbg !11
  %dma.buffer.lds916 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %192), !dbg !11
  %193 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds916, i32 %dma.buffer.offset914, <4 x i32> %dma.buffer.rsrc915) #10, !dbg !11
  %194 = add nuw i64 %62, 2048, !dbg !15
  %195 = icmp slt i64 %194, 0, !dbg !15
  %196 = sub nuw i64 -2049, %62, !dbg !15
  %197 = select i1 %195, i64 %196, i64 %194, !dbg !15
  %198 = sdiv i64 %197, 8192, !dbg !15
  %.lobit19 = ashr i64 %194, 63, !dbg !15
  %199 = xor i64 %198, %.lobit19, !dbg !15
  %200 = add nsw i64 %199, %15, !dbg !15
  %201 = add nsw i64 %102, %56, !dbg !15
  %202 = add nsw i64 %201, 32, !dbg !15
  %203 = icmp slt i64 %201, -32, !dbg !15
  %204 = sub nsw i64 -33, %201, !dbg !15
  %205 = select i1 %203, i64 %204, i64 %202, !dbg !15
  %206 = sdiv i64 %205, 128, !dbg !15
  %207 = sext i1 %203 to i64, !dbg !15
  %208 = xor i64 %206, %207, !dbg !15
  %209 = shl nsw i64 %208, 7, !dbg !15
  %210 = sub i64 %56, %209, !dbg !15
  %211 = shl i64 %200, 19, !dbg !15
  %212 = and i64 %211, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %213 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %212, !dbg !15
  %214 = getelementptr [2 x i8], ptr addrspace(1) %213, i64 %102, !dbg !15
  %215 = getelementptr [2 x i8], ptr addrspace(1) %214, i64 %210, !dbg !15
  %216 = getelementptr i8, ptr addrspace(1) %215, i64 64, !dbg !15
  %217 = getelementptr [2 x i8], ptr addrspace(1) %216, i64 %108, !dbg !15
  %218 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local199), !dbg !15
  %219 = zext i32 %218 to i64, !dbg !15
  %220 = inttoptr i64 %219 to ptr addrspace(3), !dbg !15
  %address.local420 = getelementptr [2 x i8], ptr addrspace(1) %217, i64 %address.local209, !dbg !15
  %221 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %222 = ptrtoint ptr addrspace(1) %address.local420 to i64, !dbg !15
  %223 = sub i64 %222, %221, !dbg !15
  %dma.buffer.offset911 = trunc i64 %223 to i32, !dbg !15
  %224 = trunc i64 %221 to i32, !dbg !15
  %225 = insertelement <4 x i32> poison, i32 %224, i64 0, !dbg !15
  %226 = lshr i64 %221, 32, !dbg !15
  %227 = trunc i64 %226 to i32, !dbg !15
  %228 = insertelement <4 x i32> %225, i32 %227, i64 1, !dbg !15
  %229 = insertelement <4 x i32> %228, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc912 = insertelement <4 x i32> %229, i32 131072, i64 3, !dbg !15
  %230 = ptrtoint ptr addrspace(3) %220 to i32, !dbg !15
  %dma.buffer.lds913 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %230), !dbg !15
  %231 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds913, i32 %dma.buffer.offset911, <4 x i32> %dma.buffer.rsrc912) #10, !dbg !15
  %232 = load <4 x half>, ptr addrspace(3) %address.local223, align 8, !dbg !12
  %lds.const.gep839 = getelementptr i8, ptr addrspace(3) %address.local223, i64 4096, !dbg !12
  %233 = load <4 x half>, ptr addrspace(3) %lds.const.gep839, align 8, !dbg !12
  %lds.const.gep840 = getelementptr i8, ptr addrspace(3) %address.local223, i64 1024, !dbg !12
  %234 = load <4 x half>, ptr addrspace(3) %lds.const.gep840, align 8, !dbg !12
  %235 = fmul <4 x half> %232, splat (half 0xH3015), !dbg !12
  %lds.const.gep841 = getelementptr i8, ptr addrspace(3) %address.local223, i64 5120, !dbg !12
  %236 = load <4 x half>, ptr addrspace(3) %lds.const.gep841, align 8, !dbg !12
  %237 = fmul <4 x half> %233, splat (half 0xH3015), !dbg !12
  %238 = fmul <4 x half> %234, splat (half 0xH3015), !dbg !12
  %239 = fmul <4 x half> %236, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %240 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !16
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %240, <4 x float> %138) #9, !dbg !16
  %lds.const.gep842 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !16
  %242 = load <4 x half>, ptr addrspace(3) %lds.const.gep842, align 8, !dbg !16
  %243 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !16
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %242, <4 x float> %241) #9, !dbg !16
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %243, <4 x float> %142) #9, !dbg !16
  %lds.const.gep843 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !16
  %246 = load <4 x half>, ptr addrspace(3) %lds.const.gep843, align 8, !dbg !16
  %lds.const.gep844 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !16
  %247 = load <4 x half>, ptr addrspace(3) %lds.const.gep844, align 8, !dbg !16
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %246, <4 x float> %245) #9, !dbg !16
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %247, <4 x float> %146) #9, !dbg !16
  %lds.const.gep845 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !16
  %250 = load <4 x half>, ptr addrspace(3) %lds.const.gep845, align 8, !dbg !16
  %lds.const.gep846 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !16
  %251 = load <4 x half>, ptr addrspace(3) %lds.const.gep846, align 8, !dbg !16
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %250, <4 x float> %249) #9, !dbg !16
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %251, <4 x float> %150) #9, !dbg !16
  %lds.const.gep847 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !16
  %254 = load <4 x half>, ptr addrspace(3) %lds.const.gep847, align 8, !dbg !16
  %255 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !16
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %254, <4 x float> %253) #9, !dbg !16
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %255, <4 x float> %154) #9, !dbg !16
  %lds.const.gep848 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !16
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep848, align 8, !dbg !16
  %259 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !16
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %258, <4 x float> %257) #9, !dbg !16
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %259, <4 x float> %158) #9, !dbg !16
  %lds.const.gep849 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !16
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep849, align 8, !dbg !16
  %lds.const.gep850 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !16
  %263 = load <4 x half>, ptr addrspace(3) %lds.const.gep850, align 8, !dbg !16
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %262, <4 x float> %261) #9, !dbg !16
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %263, <4 x float> %162) #9, !dbg !16
  %lds.const.gep851 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !16
  %266 = load <4 x half>, ptr addrspace(3) %lds.const.gep851, align 8, !dbg !16
  %lds.const.gep852 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !16
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep852, align 8, !dbg !16
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %266, <4 x float> %265) #9, !dbg !16
  %lds.const.gep853 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !16
  %269 = load <4 x half>, ptr addrspace(3) %lds.const.gep853, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %267, <4 x float> %165) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %269, <4 x float> %270) #9, !dbg !16
  %272 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local177), !dbg !11
  %273 = zext i32 %272 to i64, !dbg !11
  %274 = inttoptr i64 %273 to ptr addrspace(3), !dbg !11
  %address.local421 = getelementptr [2 x i8], ptr addrspace(1) %58, i64 %address.local182, !dbg !11
  %275 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %276 = ptrtoint ptr addrspace(1) %address.local421 to i64, !dbg !11
  %277 = sub i64 %276, %275, !dbg !11
  %dma.buffer.offset908 = trunc i64 %277 to i32, !dbg !11
  %278 = trunc i64 %275 to i32, !dbg !11
  %279 = insertelement <4 x i32> poison, i32 %278, i64 0, !dbg !11
  %280 = lshr i64 %275, 32, !dbg !11
  %281 = trunc i64 %280 to i32, !dbg !11
  %282 = insertelement <4 x i32> %279, i32 %281, i64 1, !dbg !11
  %283 = insertelement <4 x i32> %282, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc909 = insertelement <4 x i32> %283, i32 131072, i64 3, !dbg !11
  %284 = ptrtoint ptr addrspace(3) %274 to i32, !dbg !11
  %dma.buffer.lds910 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %284), !dbg !11
  %285 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds910, i32 %dma.buffer.offset908, <4 x i32> %dma.buffer.rsrc909) #10, !dbg !11
  %286 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local189), !dbg !11
  %287 = zext i32 %286 to i64, !dbg !11
  %288 = inttoptr i64 %287 to ptr addrspace(3), !dbg !11
  %address.local422 = getelementptr [2 x i8], ptr addrspace(1) %58, i64 %address.local194, !dbg !11
  %289 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %290 = ptrtoint ptr addrspace(1) %address.local422 to i64, !dbg !11
  %291 = sub i64 %290, %289, !dbg !11
  %dma.buffer.offset905 = trunc i64 %291 to i32, !dbg !11
  %292 = trunc i64 %289 to i32, !dbg !11
  %293 = insertelement <4 x i32> poison, i32 %292, i64 0, !dbg !11
  %294 = lshr i64 %289, 32, !dbg !11
  %295 = trunc i64 %294 to i32, !dbg !11
  %296 = insertelement <4 x i32> %293, i32 %295, i64 1, !dbg !11
  %297 = insertelement <4 x i32> %296, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc906 = insertelement <4 x i32> %297, i32 131072, i64 3, !dbg !11
  %298 = ptrtoint ptr addrspace(3) %288 to i32, !dbg !11
  %dma.buffer.lds907 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %298), !dbg !11
  %299 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds907, i32 %dma.buffer.offset905, <4 x i32> %dma.buffer.rsrc906) #10, !dbg !11
  %300 = add nuw i64 %62, 4096, !dbg !15
  %301 = icmp slt i64 %300, 0, !dbg !15
  %302 = sub nuw i64 -4097, %62, !dbg !15
  %303 = select i1 %301, i64 %302, i64 %300, !dbg !15
  %304 = sdiv i64 %303, 8192, !dbg !15
  %.lobit20 = ashr i64 %300, 63, !dbg !15
  %305 = xor i64 %304, %.lobit20, !dbg !15
  %306 = add nsw i64 %305, %15, !dbg !15
  %307 = add nsw i64 %201, 64, !dbg !15
  %308 = icmp slt i64 %201, -64, !dbg !15
  %309 = sub nsw i64 -65, %201, !dbg !15
  %310 = select i1 %308, i64 %309, i64 %307, !dbg !15
  %311 = sdiv i64 %310, 128, !dbg !15
  %312 = sext i1 %308 to i64, !dbg !15
  %313 = xor i64 %311, %312, !dbg !15
  %314 = shl nsw i64 %313, 7, !dbg !15
  %315 = sub i64 %56, %314, !dbg !15
  %316 = shl i64 %306, 19, !dbg !15
  %317 = and i64 %316, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %318 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %317, !dbg !15
  %319 = getelementptr [2 x i8], ptr addrspace(1) %318, i64 %102, !dbg !15
  %320 = getelementptr [2 x i8], ptr addrspace(1) %319, i64 %315, !dbg !15
  %321 = getelementptr i8, ptr addrspace(1) %320, i64 128, !dbg !15
  %322 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %108, !dbg !15
  %323 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local199), !dbg !15
  %324 = zext i32 %323 to i64, !dbg !15
  %325 = inttoptr i64 %324 to ptr addrspace(3), !dbg !15
  %address.local423 = getelementptr [2 x i8], ptr addrspace(1) %322, i64 %address.local209, !dbg !15
  %326 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %327 = ptrtoint ptr addrspace(1) %address.local423 to i64, !dbg !15
  %328 = sub i64 %327, %326, !dbg !15
  %dma.buffer.offset902 = trunc i64 %328 to i32, !dbg !15
  %329 = trunc i64 %326 to i32, !dbg !15
  %330 = insertelement <4 x i32> poison, i32 %329, i64 0, !dbg !15
  %331 = lshr i64 %326, 32, !dbg !15
  %332 = trunc i64 %331 to i32, !dbg !15
  %333 = insertelement <4 x i32> %330, i32 %332, i64 1, !dbg !15
  %334 = insertelement <4 x i32> %333, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc903 = insertelement <4 x i32> %334, i32 131072, i64 3, !dbg !15
  %335 = ptrtoint ptr addrspace(3) %325 to i32, !dbg !15
  %dma.buffer.lds904 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %335), !dbg !15
  %336 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds904, i32 %dma.buffer.offset902, <4 x i32> %dma.buffer.rsrc903) #10, !dbg !15
  %337 = load <4 x half>, ptr addrspace(3) %address.local223, align 8, !dbg !12
  %lds.const.gep854 = getelementptr i8, ptr addrspace(3) %address.local223, i64 4096, !dbg !12
  %338 = load <4 x half>, ptr addrspace(3) %lds.const.gep854, align 8, !dbg !12
  %lds.const.gep855 = getelementptr i8, ptr addrspace(3) %address.local223, i64 1024, !dbg !12
  %339 = load <4 x half>, ptr addrspace(3) %lds.const.gep855, align 8, !dbg !12
  %340 = fmul <4 x half> %337, splat (half 0xH3015), !dbg !12
  %lds.const.gep856 = getelementptr i8, ptr addrspace(3) %address.local223, i64 5120, !dbg !12
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep856, align 8, !dbg !12
  %342 = fmul <4 x half> %338, splat (half 0xH3015), !dbg !12
  %343 = fmul <4 x half> %339, splat (half 0xH3015), !dbg !12
  %344 = fmul <4 x half> %341, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %345 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !16
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %340, <4 x half> %345, <4 x float> %244) #9, !dbg !16
  %lds.const.gep857 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !16
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep857, align 8, !dbg !16
  %348 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !16
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %347, <4 x float> %346) #9, !dbg !16
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %340, <4 x half> %348, <4 x float> %248) #9, !dbg !16
  %lds.const.gep858 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !16
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep858, align 8, !dbg !16
  %lds.const.gep859 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !16
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep859, align 8, !dbg !16
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %351, <4 x float> %350) #9, !dbg !16
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %340, <4 x half> %352, <4 x float> %252) #9, !dbg !16
  %lds.const.gep860 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !16
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep860, align 8, !dbg !16
  %lds.const.gep861 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !16
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep861, align 8, !dbg !16
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %355, <4 x float> %354) #9, !dbg !16
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %340, <4 x half> %356, <4 x float> %256) #9, !dbg !16
  %lds.const.gep862 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !16
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep862, align 8, !dbg !16
  %360 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !16
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %342, <4 x half> %359, <4 x float> %358) #9, !dbg !16
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %360, <4 x float> %260) #9, !dbg !16
  %lds.const.gep863 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !16
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep863, align 8, !dbg !16
  %364 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !16
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %344, <4 x half> %363, <4 x float> %362) #9, !dbg !16
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %364, <4 x float> %264) #9, !dbg !16
  %lds.const.gep864 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !16
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep864, align 8, !dbg !16
  %lds.const.gep865 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !16
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep865, align 8, !dbg !16
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %344, <4 x half> %367, <4 x float> %366) #9, !dbg !16
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %368, <4 x float> %268) #9, !dbg !16
  %lds.const.gep866 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !16
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep866, align 8, !dbg !16
  %lds.const.gep867 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !16
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep867, align 8, !dbg !16
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %344, <4 x half> %371, <4 x float> %370) #9, !dbg !16
  %lds.const.gep868 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !16
  %374 = load <4 x half>, ptr addrspace(3) %lds.const.gep868, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %343, <4 x half> %372, <4 x float> %271) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %344, <4 x half> %374, <4 x float> %375) #9, !dbg !16
  %377 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local177), !dbg !11
  %378 = zext i32 %377 to i64, !dbg !11
  %379 = inttoptr i64 %378 to ptr addrspace(3), !dbg !11
  %address.local424 = getelementptr [2 x i8], ptr addrspace(1) %60, i64 %address.local182, !dbg !11
  %380 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %381 = ptrtoint ptr addrspace(1) %address.local424 to i64, !dbg !11
  %382 = sub i64 %381, %380, !dbg !11
  %dma.buffer.offset899 = trunc i64 %382 to i32, !dbg !11
  %383 = trunc i64 %380 to i32, !dbg !11
  %384 = insertelement <4 x i32> poison, i32 %383, i64 0, !dbg !11
  %385 = lshr i64 %380, 32, !dbg !11
  %386 = trunc i64 %385 to i32, !dbg !11
  %387 = insertelement <4 x i32> %384, i32 %386, i64 1, !dbg !11
  %388 = insertelement <4 x i32> %387, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc900 = insertelement <4 x i32> %388, i32 131072, i64 3, !dbg !11
  %389 = ptrtoint ptr addrspace(3) %379 to i32, !dbg !11
  %dma.buffer.lds901 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %389), !dbg !11
  %390 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds901, i32 %dma.buffer.offset899, <4 x i32> %dma.buffer.rsrc900) #10, !dbg !11
  %391 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local189), !dbg !11
  %392 = zext i32 %391 to i64, !dbg !11
  %393 = inttoptr i64 %392 to ptr addrspace(3), !dbg !11
  %address.local425 = getelementptr [2 x i8], ptr addrspace(1) %60, i64 %address.local194, !dbg !11
  %394 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %395 = ptrtoint ptr addrspace(1) %address.local425 to i64, !dbg !11
  %396 = sub i64 %395, %394, !dbg !11
  %dma.buffer.offset896 = trunc i64 %396 to i32, !dbg !11
  %397 = trunc i64 %394 to i32, !dbg !11
  %398 = insertelement <4 x i32> poison, i32 %397, i64 0, !dbg !11
  %399 = lshr i64 %394, 32, !dbg !11
  %400 = trunc i64 %399 to i32, !dbg !11
  %401 = insertelement <4 x i32> %398, i32 %400, i64 1, !dbg !11
  %402 = insertelement <4 x i32> %401, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc897 = insertelement <4 x i32> %402, i32 131072, i64 3, !dbg !11
  %403 = ptrtoint ptr addrspace(3) %393 to i32, !dbg !11
  %dma.buffer.lds898 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %403), !dbg !11
  %404 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds898, i32 %dma.buffer.offset896, <4 x i32> %dma.buffer.rsrc897) #10, !dbg !11
  %405 = add nuw i64 %62, 6144, !dbg !15
  %406 = icmp slt i64 %405, 0, !dbg !15
  %407 = sub nuw i64 -6145, %62, !dbg !15
  %408 = select i1 %406, i64 %407, i64 %405, !dbg !15
  %409 = sdiv i64 %408, 8192, !dbg !15
  %.lobit21 = ashr i64 %405, 63, !dbg !15
  %410 = xor i64 %409, %.lobit21, !dbg !15
  %411 = add nsw i64 %410, %15, !dbg !15
  %412 = add nsw i64 %201, 96, !dbg !15
  %413 = icmp slt i64 %201, -96, !dbg !15
  %414 = sub nsw i64 -97, %201, !dbg !15
  %415 = select i1 %413, i64 %414, i64 %412, !dbg !15
  %416 = sdiv i64 %415, 128, !dbg !15
  %417 = sext i1 %413 to i64, !dbg !15
  %418 = xor i64 %416, %417, !dbg !15
  %419 = shl nsw i64 %418, 7, !dbg !15
  %420 = sub i64 %56, %419, !dbg !15
  %421 = shl i64 %411, 19, !dbg !15
  %422 = and i64 %421, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %423 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %422, !dbg !15
  %424 = getelementptr [2 x i8], ptr addrspace(1) %423, i64 %102, !dbg !15
  %425 = getelementptr [2 x i8], ptr addrspace(1) %424, i64 %420, !dbg !15
  %426 = getelementptr i8, ptr addrspace(1) %425, i64 192, !dbg !15
  %427 = getelementptr [2 x i8], ptr addrspace(1) %426, i64 %108, !dbg !15
  %428 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local199), !dbg !15
  %429 = zext i32 %428 to i64, !dbg !15
  %430 = inttoptr i64 %429 to ptr addrspace(3), !dbg !15
  %address.local426 = getelementptr [2 x i8], ptr addrspace(1) %427, i64 %address.local209, !dbg !15
  %431 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %432 = ptrtoint ptr addrspace(1) %address.local426 to i64, !dbg !15
  %433 = sub i64 %432, %431, !dbg !15
  %dma.buffer.offset893 = trunc i64 %433 to i32, !dbg !15
  %434 = trunc i64 %431 to i32, !dbg !15
  %435 = insertelement <4 x i32> poison, i32 %434, i64 0, !dbg !15
  %436 = lshr i64 %431, 32, !dbg !15
  %437 = trunc i64 %436 to i32, !dbg !15
  %438 = insertelement <4 x i32> %435, i32 %437, i64 1, !dbg !15
  %439 = insertelement <4 x i32> %438, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc894 = insertelement <4 x i32> %439, i32 131072, i64 3, !dbg !15
  %440 = ptrtoint ptr addrspace(3) %430 to i32, !dbg !15
  %dma.buffer.lds895 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %440), !dbg !15
  %441 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds895, i32 %dma.buffer.offset893, <4 x i32> %dma.buffer.rsrc894) #10, !dbg !15
  %442 = load <4 x half>, ptr addrspace(3) %address.local223, align 8, !dbg !12
  %lds.const.gep869 = getelementptr i8, ptr addrspace(3) %address.local223, i64 4096, !dbg !12
  %443 = load <4 x half>, ptr addrspace(3) %lds.const.gep869, align 8, !dbg !12
  %lds.const.gep870 = getelementptr i8, ptr addrspace(3) %address.local223, i64 1024, !dbg !12
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep870, align 8, !dbg !12
  %445 = fmul <4 x half> %442, splat (half 0xH3015), !dbg !12
  %lds.const.gep871 = getelementptr i8, ptr addrspace(3) %address.local223, i64 5120, !dbg !12
  %446 = load <4 x half>, ptr addrspace(3) %lds.const.gep871, align 8, !dbg !12
  %447 = fmul <4 x half> %443, splat (half 0xH3015), !dbg !12
  %448 = fmul <4 x half> %444, splat (half 0xH3015), !dbg !12
  %449 = fmul <4 x half> %446, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %450 = add nuw nsw i64 %102, %15, !dbg !17
  %451 = shl i64 %450, 19, !dbg !17
  %452 = and i64 %451, 16252928, !dbg !17
  %453 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %452, !dbg !17
  %454 = getelementptr [2 x i8], ptr addrspace(1) %453, i64 %108, !dbg !17
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local196, !dbg !17
  %address.local428 = ptrtoint ptr addrspace(3) %address.local427 to i64, !dbg !17
  %address.local429 = trunc i64 %address.local428 to i32, !dbg !17
  %455 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !17
  %456 = zext i32 %455 to i64, !dbg !17
  %457 = inttoptr i64 %456 to ptr addrspace(3), !dbg !17
  %address.local430 = shl nuw nsw i64 %address.local, 5, !dbg !17
  %address.local431 = and i64 %address.local430, 8064, !dbg !17
  %address.local432 = shl nuw nsw i64 %address.local, 3, !dbg !17
  %address.local433 = and i64 %address.local432, 24, !dbg !17
  %address.local434 = or disjoint i64 %address.local431, %address.local433, !dbg !17
  %address.local435 = getelementptr [2 x i8], ptr addrspace(1) %454, i64 %address.local434, !dbg !17
  %458 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %459 = ptrtoint ptr addrspace(1) %address.local435 to i64, !dbg !17
  %460 = sub i64 %459, %458, !dbg !17
  %dma.buffer.offset890 = trunc i64 %460 to i32, !dbg !17
  %461 = trunc i64 %458 to i32, !dbg !17
  %462 = insertelement <4 x i32> poison, i32 %461, i64 0, !dbg !17
  %463 = lshr i64 %458, 32, !dbg !17
  %464 = trunc i64 %463 to i32, !dbg !17
  %465 = insertelement <4 x i32> %462, i32 %464, i64 1, !dbg !17
  %466 = insertelement <4 x i32> %465, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc891 = insertelement <4 x i32> %466, i32 131072, i64 3, !dbg !17
  %467 = ptrtoint ptr addrspace(3) %457 to i32, !dbg !17
  %dma.buffer.lds892 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %467), !dbg !17
  %468 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds892, i32 %dma.buffer.offset890, <4 x i32> %dma.buffer.rsrc891) #10, !dbg !17
  %469 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !16
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %469, <4 x float> %349) #9, !dbg !16
  %lds.const.gep872 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !16
  %471 = load <4 x half>, ptr addrspace(3) %lds.const.gep872, align 8, !dbg !16
  %472 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !16
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %471, <4 x float> %470) #9, !dbg !16
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %472, <4 x float> %353) #9, !dbg !16
  %lds.const.gep873 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !16
  %475 = load <4 x half>, ptr addrspace(3) %lds.const.gep873, align 8, !dbg !16
  %lds.const.gep874 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !16
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep874, align 8, !dbg !16
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %475, <4 x float> %474) #9, !dbg !16
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %476, <4 x float> %357) #9, !dbg !16
  %lds.const.gep875 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !16
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep875, align 8, !dbg !16
  %lds.const.gep876 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !16
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep876, align 8, !dbg !16
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %479, <4 x float> %478) #9, !dbg !16
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %445, <4 x half> %480, <4 x float> %361) #9, !dbg !16
  %lds.const.gep877 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !16
  %483 = load <4 x half>, ptr addrspace(3) %lds.const.gep877, align 8, !dbg !16
  %484 = load <4 x half>, ptr addrspace(3) %address.local248, align 8, !dbg !16
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %483, <4 x float> %482) #9, !dbg !16
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %484, <4 x float> %365) #9, !dbg !16
  %lds.const.gep878 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2048, !dbg !16
  %487 = load <4 x half>, ptr addrspace(3) %lds.const.gep878, align 8, !dbg !16
  %488 = load <4 x half>, ptr addrspace(3) %address.local277, align 8, !dbg !16
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %487, <4 x float> %486) #9, !dbg !16
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %488, <4 x float> %369) #9, !dbg !16
  %lds.const.gep879 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2048, !dbg !16
  %491 = load <4 x half>, ptr addrspace(3) %lds.const.gep879, align 8, !dbg !16
  %lds.const.gep880 = getelementptr i8, ptr addrspace(3) %address.local248, i64 128, !dbg !16
  %492 = load <4 x half>, ptr addrspace(3) %lds.const.gep880, align 8, !dbg !16
  %493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %491, <4 x float> %490) #9, !dbg !16
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %492, <4 x float> %373) #9, !dbg !16
  %lds.const.gep881 = getelementptr i8, ptr addrspace(3) %address.local248, i64 2176, !dbg !16
  %495 = load <4 x half>, ptr addrspace(3) %lds.const.gep881, align 8, !dbg !16
  %lds.const.gep882 = getelementptr i8, ptr addrspace(3) %address.local277, i64 128, !dbg !16
  %496 = load <4 x half>, ptr addrspace(3) %lds.const.gep882, align 8, !dbg !16
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %495, <4 x float> %494) #9, !dbg !16
  %lds.const.gep883 = getelementptr i8, ptr addrspace(3) %address.local277, i64 2176, !dbg !16
  %498 = load <4 x half>, ptr addrspace(3) %lds.const.gep883, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %496, <4 x float> %376) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %500 = or disjoint i64 %71, 63, !dbg !13
  %501 = icmp uge i64 %19, %500, !dbg !13
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %449, <4 x half> %498, <4 x float> %499) #9, !dbg !16
  %503 = and i1 %39, %501, !dbg !13
  br i1 %503, label %504, label %505, !dbg !13

504:                                              ; preds = %70
  br label %585, !dbg !13

505:                                              ; preds = %70
  %506 = add i64 %19, %35, !dbg !13
  %507 = shl nuw nsw i64 %36, 4, !dbg !13
  %508 = add i64 %506, %507, !dbg !13
  %509 = and i32 %34, 240, !dbg !13
  %index.compact = zext nneg i32 %509 to i64, !dbg !13
  %510 = sub i64 %508, %index.compact, !dbg !13
  %511 = and i64 %35, 48, !dbg !13
  %512 = or disjoint i64 %71, %511, !dbg !13
  %513 = add i64 %510, 1, !dbg !22
  %514 = lshr i64 %513, 32, !dbg !23
  %compare.high = trunc i64 %514 to i32, !dbg !23
  %515 = lshr i64 %512, 32, !dbg !23
  %compare.high697 = trunc i64 %515 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high697, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high697, !dbg !23
  %compare.low.base = trunc i64 %513 to i32, !dbg !23
  %compare.low.base698 = trunc i64 %512 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base698, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %516 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low699 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result700 = select i1 %compare.high.equal, i1 %compare.low699, i1 %compare.high.order, !dbg !23
  %517 = select i1 %compare.result700, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base701 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local702 = or i32 %compare.local.base701, 2, !dbg !23
  %compare.low703 = icmp ugt i32 %compare.low.base, %compare.local702, !dbg !23
  %compare.result704 = select i1 %compare.high.equal, i1 %compare.low703, i1 %compare.high.order, !dbg !23
  %518 = select i1 %compare.result704, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base705 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local706 = or i32 %compare.local.base705, 3, !dbg !23
  %compare.low707 = icmp ugt i32 %compare.low.base, %compare.local706, !dbg !23
  %compare.result708 = select i1 %compare.high.equal, i1 %compare.low707, i1 %compare.high.order, !dbg !23
  %519 = select i1 %compare.result708, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base709 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local710 = or i32 %compare.local.base709, 4, !dbg !23
  %compare.low711 = icmp ugt i32 %compare.low.base, %compare.local710, !dbg !23
  %compare.result712 = select i1 %compare.high.equal, i1 %compare.low711, i1 %compare.high.order, !dbg !23
  %520 = select i1 %compare.result712, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base713 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local714 = or i32 %compare.local.base713, 5, !dbg !23
  %compare.low715 = icmp ugt i32 %compare.low.base, %compare.local714, !dbg !23
  %compare.result716 = select i1 %compare.high.equal, i1 %compare.low715, i1 %compare.high.order, !dbg !23
  %521 = select i1 %compare.result716, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base717 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local718 = or i32 %compare.local.base717, 6, !dbg !23
  %compare.low719 = icmp ugt i32 %compare.low.base, %compare.local718, !dbg !23
  %compare.result720 = select i1 %compare.high.equal, i1 %compare.low719, i1 %compare.high.order, !dbg !23
  %522 = select i1 %compare.result720, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base721 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local722 = or i32 %compare.local.base721, 7, !dbg !23
  %compare.low723 = icmp ugt i32 %compare.low.base, %compare.local722, !dbg !23
  %compare.result724 = select i1 %compare.high.equal, i1 %compare.low723, i1 %compare.high.order, !dbg !23
  %523 = select i1 %compare.result724, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base725 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local726 = or i32 %compare.local.base725, 8, !dbg !23
  %compare.low727 = icmp ugt i32 %compare.low.base, %compare.local726, !dbg !23
  %compare.result728 = select i1 %compare.high.equal, i1 %compare.low727, i1 %compare.high.order, !dbg !23
  %524 = select i1 %compare.result728, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base729 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local730 = or i32 %compare.local.base729, 9, !dbg !23
  %compare.low731 = icmp ugt i32 %compare.low.base, %compare.local730, !dbg !23
  %compare.result732 = select i1 %compare.high.equal, i1 %compare.low731, i1 %compare.high.order, !dbg !23
  %525 = select i1 %compare.result732, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base733 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local734 = or i32 %compare.local.base733, 10, !dbg !23
  %compare.low735 = icmp ugt i32 %compare.low.base, %compare.local734, !dbg !23
  %compare.result736 = select i1 %compare.high.equal, i1 %compare.low735, i1 %compare.high.order, !dbg !23
  %526 = select i1 %compare.result736, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base737 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local738 = or i32 %compare.local.base737, 11, !dbg !23
  %compare.low739 = icmp ugt i32 %compare.low.base, %compare.local738, !dbg !23
  %compare.result740 = select i1 %compare.high.equal, i1 %compare.low739, i1 %compare.high.order, !dbg !23
  %527 = select i1 %compare.result740, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base741 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local742 = or i32 %compare.local.base741, 12, !dbg !23
  %compare.low743 = icmp ugt i32 %compare.low.base, %compare.local742, !dbg !23
  %compare.result744 = select i1 %compare.high.equal, i1 %compare.low743, i1 %compare.high.order, !dbg !23
  %528 = select i1 %compare.result744, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base745 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local746 = or i32 %compare.local.base745, 13, !dbg !23
  %compare.low747 = icmp ugt i32 %compare.low.base, %compare.local746, !dbg !23
  %compare.result748 = select i1 %compare.high.equal, i1 %compare.low747, i1 %compare.high.order, !dbg !23
  %529 = select i1 %compare.result748, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base749 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local750 = or i32 %compare.local.base749, 14, !dbg !23
  %compare.low751 = icmp ugt i32 %compare.low.base, %compare.local750, !dbg !23
  %compare.result752 = select i1 %compare.high.equal, i1 %compare.low751, i1 %compare.high.order, !dbg !23
  %530 = select i1 %compare.result752, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base753 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base698, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local754 = or i32 %compare.local.base753, 15, !dbg !23
  %compare.low755 = icmp ugt i32 %compare.low.base, %compare.local754, !dbg !23
  %compare.result756 = select i1 %compare.high.equal, i1 %compare.low755, i1 %compare.high.order, !dbg !23
  %531 = select i1 %compare.result756, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %532 = add i64 %510, 65, !dbg !22
  %533 = lshr i64 %532, 32, !dbg !23
  %compare.high757 = trunc i64 %533 to i32, !dbg !23
  %534 = lshr i64 %512, 32, !dbg !23
  %compare.high758 = trunc i64 %534 to i32, !dbg !23
  %compare.high.equal759 = icmp eq i32 %compare.high757, %compare.high758, !dbg !23
  %compare.high.order760 = icmp ugt i32 %compare.high757, %compare.high758, !dbg !23
  %compare.low.base761 = trunc i64 %532 to i32, !dbg !23
  %compare.low.base762 = trunc i64 %512 to i32, !dbg !23
  %compare.low763 = icmp ugt i32 %compare.low.base761, %compare.low.base762, !dbg !23
  %compare.result764 = select i1 %compare.high.equal759, i1 %compare.low763, i1 %compare.high.order760, !dbg !23
  %535 = select i1 %compare.result764, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base765 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 16) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local766 = or i32 %compare.local.base765, 1, !dbg !23
  %compare.low767 = icmp ugt i32 %compare.low.base761, %compare.local766, !dbg !23
  %compare.result768 = select i1 %compare.high.equal759, i1 %compare.low767, i1 %compare.high.order760, !dbg !23
  %536 = select i1 %compare.result768, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base769 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 17) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local770 = or i32 %compare.local.base769, 2, !dbg !23
  %compare.low771 = icmp ugt i32 %compare.low.base761, %compare.local770, !dbg !23
  %compare.result772 = select i1 %compare.high.equal759, i1 %compare.low771, i1 %compare.high.order760, !dbg !23
  %537 = select i1 %compare.result772, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base773 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 18) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local774 = or i32 %compare.local.base773, 3, !dbg !23
  %compare.low775 = icmp ugt i32 %compare.low.base761, %compare.local774, !dbg !23
  %compare.result776 = select i1 %compare.high.equal759, i1 %compare.low775, i1 %compare.high.order760, !dbg !23
  %538 = select i1 %compare.result776, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base777 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 19) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local778 = or i32 %compare.local.base777, 4, !dbg !23
  %compare.low779 = icmp ugt i32 %compare.low.base761, %compare.local778, !dbg !23
  %compare.result780 = select i1 %compare.high.equal759, i1 %compare.low779, i1 %compare.high.order760, !dbg !23
  %539 = select i1 %compare.result780, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base781 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 20) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local782 = or i32 %compare.local.base781, 5, !dbg !23
  %compare.low783 = icmp ugt i32 %compare.low.base761, %compare.local782, !dbg !23
  %compare.result784 = select i1 %compare.high.equal759, i1 %compare.low783, i1 %compare.high.order760, !dbg !23
  %540 = select i1 %compare.result784, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base785 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 21) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local786 = or i32 %compare.local.base785, 6, !dbg !23
  %compare.low787 = icmp ugt i32 %compare.low.base761, %compare.local786, !dbg !23
  %compare.result788 = select i1 %compare.high.equal759, i1 %compare.low787, i1 %compare.high.order760, !dbg !23
  %541 = select i1 %compare.result788, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base789 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 22) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local790 = or i32 %compare.local.base789, 7, !dbg !23
  %compare.low791 = icmp ugt i32 %compare.low.base761, %compare.local790, !dbg !23
  %compare.result792 = select i1 %compare.high.equal759, i1 %compare.low791, i1 %compare.high.order760, !dbg !23
  %542 = select i1 %compare.result792, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base793 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 23) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local794 = or i32 %compare.local.base793, 8, !dbg !23
  %compare.low795 = icmp ugt i32 %compare.low.base761, %compare.local794, !dbg !23
  %compare.result796 = select i1 %compare.high.equal759, i1 %compare.low795, i1 %compare.high.order760, !dbg !23
  %543 = select i1 %compare.result796, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base797 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 24) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local798 = or i32 %compare.local.base797, 9, !dbg !23
  %compare.low799 = icmp ugt i32 %compare.low.base761, %compare.local798, !dbg !23
  %compare.result800 = select i1 %compare.high.equal759, i1 %compare.low799, i1 %compare.high.order760, !dbg !23
  %544 = select i1 %compare.result800, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base801 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 25) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local802 = or i32 %compare.local.base801, 10, !dbg !23
  %compare.low803 = icmp ugt i32 %compare.low.base761, %compare.local802, !dbg !23
  %compare.result804 = select i1 %compare.high.equal759, i1 %compare.low803, i1 %compare.high.order760, !dbg !23
  %545 = select i1 %compare.result804, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base805 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 26) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local806 = or i32 %compare.local.base805, 11, !dbg !23
  %compare.low807 = icmp ugt i32 %compare.low.base761, %compare.local806, !dbg !23
  %compare.result808 = select i1 %compare.high.equal759, i1 %compare.low807, i1 %compare.high.order760, !dbg !23
  %546 = select i1 %compare.result808, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base809 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 27) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local810 = or i32 %compare.local.base809, 12, !dbg !23
  %compare.low811 = icmp ugt i32 %compare.low.base761, %compare.local810, !dbg !23
  %compare.result812 = select i1 %compare.high.equal759, i1 %compare.low811, i1 %compare.high.order760, !dbg !23
  %547 = select i1 %compare.result812, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base813 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 28) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local814 = or i32 %compare.local.base813, 13, !dbg !23
  %compare.low815 = icmp ugt i32 %compare.low.base761, %compare.local814, !dbg !23
  %compare.result816 = select i1 %compare.high.equal759, i1 %compare.low815, i1 %compare.high.order760, !dbg !23
  %548 = select i1 %compare.result816, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base817 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 29) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local818 = or i32 %compare.local.base817, 14, !dbg !23
  %compare.low819 = icmp ugt i32 %compare.low.base761, %compare.local818, !dbg !23
  %compare.result820 = select i1 %compare.high.equal759, i1 %compare.low819, i1 %compare.high.order760, !dbg !23
  %549 = select i1 %compare.result820, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base821 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base762, i32 30) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local822 = or i32 %compare.local.base821, 15, !dbg !23
  %compare.low823 = icmp ugt i32 %compare.low.base761, %compare.local822, !dbg !23
  %compare.result824 = select i1 %compare.high.equal759, i1 %compare.low823, i1 %compare.high.order760, !dbg !23
  %550 = select i1 %compare.result824, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %551 = insertelement <16 x float> poison, float %516, i64 0, !dbg !13
  %552 = insertelement <16 x float> %551, float %517, i64 1, !dbg !13
  %553 = insertelement <16 x float> %552, float %518, i64 2, !dbg !13
  %554 = insertelement <16 x float> %553, float %519, i64 3, !dbg !13
  %555 = insertelement <16 x float> %554, float %520, i64 4, !dbg !13
  %556 = insertelement <16 x float> %555, float %521, i64 5, !dbg !13
  %557 = insertelement <16 x float> %556, float %522, i64 6, !dbg !13
  %558 = insertelement <16 x float> %557, float %523, i64 7, !dbg !13
  %559 = insertelement <16 x float> %558, float %524, i64 8, !dbg !13
  %560 = insertelement <16 x float> %559, float %525, i64 9, !dbg !13
  %561 = insertelement <16 x float> %560, float %526, i64 10, !dbg !13
  %562 = insertelement <16 x float> %561, float %527, i64 11, !dbg !13
  %563 = insertelement <16 x float> %562, float %528, i64 12, !dbg !13
  %564 = insertelement <16 x float> %563, float %529, i64 13, !dbg !13
  %565 = insertelement <16 x float> %564, float %530, i64 14, !dbg !13
  %566 = insertelement <16 x float> %565, float %531, i64 15, !dbg !13
  %567 = insertvalue [2 x <16 x float>] poison, <16 x float> %566, 0, !dbg !13
  %568 = insertelement <16 x float> poison, float %535, i64 0, !dbg !13
  %569 = insertelement <16 x float> %568, float %536, i64 1, !dbg !13
  %570 = insertelement <16 x float> %569, float %537, i64 2, !dbg !13
  %571 = insertelement <16 x float> %570, float %538, i64 3, !dbg !13
  %572 = insertelement <16 x float> %571, float %539, i64 4, !dbg !13
  %573 = insertelement <16 x float> %572, float %540, i64 5, !dbg !13
  %574 = insertelement <16 x float> %573, float %541, i64 6, !dbg !13
  %575 = insertelement <16 x float> %574, float %542, i64 7, !dbg !13
  %576 = insertelement <16 x float> %575, float %543, i64 8, !dbg !13
  %577 = insertelement <16 x float> %576, float %544, i64 9, !dbg !13
  %578 = insertelement <16 x float> %577, float %545, i64 10, !dbg !13
  %579 = insertelement <16 x float> %578, float %546, i64 11, !dbg !13
  %580 = insertelement <16 x float> %579, float %547, i64 12, !dbg !13
  %581 = insertelement <16 x float> %580, float %548, i64 13, !dbg !13
  %582 = insertelement <16 x float> %581, float %549, i64 14, !dbg !13
  %583 = insertelement <16 x float> %582, float %550, i64 15, !dbg !13
  %584 = insertvalue [2 x <16 x float>] %567, <16 x float> %583, 1, !dbg !13
  br label %585, !dbg !13

585:                                              ; preds = %504, %505
  %586 = phi [2 x <16 x float>] [ %584, %505 ], [ zeroinitializer, %504 ], !dbg !13
  br label %587, !dbg !13

587:                                              ; preds = %585
  %588 = extractvalue [2 x <16 x float>] %586, 0, !dbg !25
  %589 = shufflevector <16 x float> %588, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %590 = fadd <4 x float> %473, %589, !dbg !25
  %591 = extractelement <4 x float> %590, i64 0, !dbg !25
  %592 = extractelement <4 x float> %590, i64 1, !dbg !25
  %593 = extractelement <4 x float> %590, i64 2, !dbg !25
  %594 = extractelement <4 x float> %590, i64 3, !dbg !25
  %595 = shufflevector <16 x float> %588, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %596 = fadd <4 x float> %477, %595, !dbg !25
  %597 = extractelement <4 x float> %596, i64 0, !dbg !25
  %598 = extractelement <4 x float> %596, i64 1, !dbg !25
  %599 = extractelement <4 x float> %596, i64 2, !dbg !25
  %600 = extractelement <4 x float> %596, i64 3, !dbg !25
  %601 = shufflevector <16 x float> %588, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %602 = fadd <4 x float> %481, %601, !dbg !25
  %603 = extractelement <4 x float> %602, i64 0, !dbg !25
  %604 = extractelement <4 x float> %602, i64 1, !dbg !25
  %605 = extractelement <4 x float> %602, i64 2, !dbg !25
  %606 = extractelement <4 x float> %602, i64 3, !dbg !25
  %607 = shufflevector <16 x float> %588, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %608 = fadd <4 x float> %485, %607, !dbg !25
  %609 = extractelement <4 x float> %608, i64 0, !dbg !25
  %610 = extractelement <4 x float> %608, i64 1, !dbg !25
  %611 = extractelement <4 x float> %608, i64 2, !dbg !25
  %612 = extractelement <4 x float> %608, i64 3, !dbg !25
  %613 = extractvalue [2 x <16 x float>] %586, 1, !dbg !25
  %614 = shufflevector <16 x float> %613, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %615 = fadd <4 x float> %489, %614, !dbg !25
  %616 = extractelement <4 x float> %615, i64 0, !dbg !25
  %617 = extractelement <4 x float> %615, i64 1, !dbg !25
  %618 = extractelement <4 x float> %615, i64 2, !dbg !25
  %619 = extractelement <4 x float> %615, i64 3, !dbg !25
  %620 = shufflevector <16 x float> %613, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %621 = fadd <4 x float> %493, %620, !dbg !25
  %622 = extractelement <4 x float> %621, i64 0, !dbg !25
  %623 = extractelement <4 x float> %621, i64 1, !dbg !25
  %624 = extractelement <4 x float> %621, i64 2, !dbg !25
  %625 = extractelement <4 x float> %621, i64 3, !dbg !25
  %626 = shufflevector <16 x float> %613, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %627 = fadd <4 x float> %497, %626, !dbg !25
  %628 = extractelement <4 x float> %627, i64 0, !dbg !25
  %629 = extractelement <4 x float> %627, i64 1, !dbg !25
  %630 = extractelement <4 x float> %627, i64 2, !dbg !25
  %631 = extractelement <4 x float> %627, i64 3, !dbg !25
  %632 = shufflevector <16 x float> %613, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %633 = fadd <4 x float> %502, %632, !dbg !25
  %634 = extractelement <4 x float> %633, i64 0, !dbg !25
  %635 = extractelement <4 x float> %633, i64 1, !dbg !25
  %636 = extractelement <4 x float> %633, i64 2, !dbg !25
  %637 = extractelement <4 x float> %633, i64 3, !dbg !25
  %638 = call float @llvm.maxnum.f32(float %591, float 0xFFF0000000000000), !dbg !26
  %639 = call float @llvm.maxnum.f32(float %638, float %592), !dbg !26
  %640 = call float @llvm.maxnum.f32(float %639, float %593), !dbg !26
  %641 = call float @llvm.maxnum.f32(float %640, float %594), !dbg !26
  %642 = call float @llvm.maxnum.f32(float %641, float %597), !dbg !26
  %643 = call float @llvm.maxnum.f32(float %642, float %598), !dbg !26
  %644 = call float @llvm.maxnum.f32(float %643, float %599), !dbg !26
  %645 = call float @llvm.maxnum.f32(float %644, float %600), !dbg !26
  %646 = call float @llvm.maxnum.f32(float %645, float %603), !dbg !26
  %647 = call float @llvm.maxnum.f32(float %646, float %604), !dbg !26
  %648 = call float @llvm.maxnum.f32(float %647, float %605), !dbg !26
  %649 = call float @llvm.maxnum.f32(float %648, float %606), !dbg !26
  %650 = call float @llvm.maxnum.f32(float %649, float %609), !dbg !26
  %651 = call float @llvm.maxnum.f32(float %650, float %610), !dbg !26
  %652 = call float @llvm.maxnum.f32(float %651, float %611), !dbg !26
  %653 = call float @llvm.maxnum.f32(float %652, float %612), !dbg !26
  %654 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %655 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %654), !dbg !26
  %656 = add nuw nsw i32 %655, 64, !dbg !26
  %657 = xor i32 %655, 16, !dbg !26
  %658 = and i32 %656, 192, !dbg !26
  %659 = icmp samesign ult i32 %657, %658, !dbg !26
  %660 = select i1 %659, i32 %657, i32 %655, !dbg !26
  %661 = shl nuw nsw i32 %660, 2, !dbg !26
  %662 = bitcast float %653 to i32, !dbg !26
  %663 = call i32 @llvm.amdgcn.ds.bpermute(i32 %661, i32 %662), !dbg !26
  %664 = bitcast i32 %663 to float, !dbg !26
  %665 = call float @llvm.maxnum.f32(float %653, float %664), !dbg !26
  %666 = xor i32 %655, 32, !dbg !26
  %667 = icmp samesign ult i32 %666, %658, !dbg !26
  %668 = select i1 %667, i32 %666, i32 %655, !dbg !26
  %669 = shl nuw nsw i32 %668, 2, !dbg !26
  %670 = bitcast float %665 to i32, !dbg !26
  %671 = call i32 @llvm.amdgcn.ds.bpermute(i32 %669, i32 %670), !dbg !26
  %672 = bitcast i32 %671 to float, !dbg !26
  %673 = call float @llvm.maxnum.f32(float %665, float %672), !dbg !26
  %674 = extractvalue [2 x <1 x float>] %65, 0, !dbg !26
  %675 = insertelement <1 x float> poison, float %673, i64 0, !dbg !26
  %676 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %675, <1 x float> %674), !dbg !26
  %677 = call float @llvm.maxnum.f32(float %616, float 0xFFF0000000000000), !dbg !26
  %678 = call float @llvm.maxnum.f32(float %677, float %617), !dbg !26
  %679 = call float @llvm.maxnum.f32(float %678, float %618), !dbg !26
  %680 = call float @llvm.maxnum.f32(float %679, float %619), !dbg !26
  %681 = call float @llvm.maxnum.f32(float %680, float %622), !dbg !26
  %682 = call float @llvm.maxnum.f32(float %681, float %623), !dbg !26
  %683 = call float @llvm.maxnum.f32(float %682, float %624), !dbg !26
  %684 = call float @llvm.maxnum.f32(float %683, float %625), !dbg !26
  %685 = call float @llvm.maxnum.f32(float %684, float %628), !dbg !26
  %686 = call float @llvm.maxnum.f32(float %685, float %629), !dbg !26
  %687 = call float @llvm.maxnum.f32(float %686, float %630), !dbg !26
  %688 = call float @llvm.maxnum.f32(float %687, float %631), !dbg !26
  %689 = call float @llvm.maxnum.f32(float %688, float %634), !dbg !26
  %690 = call float @llvm.maxnum.f32(float %689, float %635), !dbg !26
  %691 = call float @llvm.maxnum.f32(float %690, float %636), !dbg !26
  %692 = call float @llvm.maxnum.f32(float %691, float %637), !dbg !26
  %693 = bitcast float %692 to i32, !dbg !26
  %694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %661, i32 %693), !dbg !26
  %695 = bitcast i32 %694 to float, !dbg !26
  %696 = call float @llvm.maxnum.f32(float %692, float %695), !dbg !26
  %697 = bitcast float %696 to i32, !dbg !26
  %698 = call i32 @llvm.amdgcn.ds.bpermute(i32 %669, i32 %697), !dbg !26
  %699 = bitcast i32 %698 to float, !dbg !26
  %700 = call float @llvm.maxnum.f32(float %696, float %699), !dbg !26
  %701 = extractvalue [2 x <1 x float>] %65, 1, !dbg !26
  %702 = insertelement <1 x float> poison, float %700, i64 0, !dbg !26
  %703 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %702, <1 x float> %701), !dbg !26
  %704 = shufflevector <1 x float> %676, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %705 = fsub <4 x float> %590, %704, !dbg !27
  %706 = fsub <4 x float> %596, %704, !dbg !27
  %707 = fsub <4 x float> %602, %704, !dbg !27
  %708 = fsub <4 x float> %608, %704, !dbg !27
  %709 = shufflevector <1 x float> %703, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %710 = fsub <4 x float> %615, %709, !dbg !27
  %711 = fsub <4 x float> %621, %709, !dbg !27
  %712 = fsub <4 x float> %627, %709, !dbg !27
  %713 = fsub <4 x float> %633, %709, !dbg !27
  %714 = fsub <1 x float> %674, %676, !dbg !28
  %715 = extractelement <1 x float> %714, i64 0, !dbg !29
  %716 = call float @llvm.amdgcn.exp2.f32(float %715), !dbg !29
  %717 = insertelement <4 x float> poison, float %716, i64 0, !dbg !34
  %718 = shufflevector <4 x float> %717, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %719 = extractvalue [2 x <1 x float>] %64, 0, !dbg !30
  %720 = extractelement <1 x float> %719, i64 0, !dbg !30
  %721 = fmul float %720, %716, !dbg !30
  %722 = fsub <1 x float> %701, %703, !dbg !28
  %723 = extractelement <1 x float> %722, i64 0, !dbg !29
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !29
  %725 = insertelement <4 x float> poison, float %724, i64 0, !dbg !34
  %726 = shufflevector <4 x float> %725, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %727 = extractvalue [2 x <1 x float>] %64, 1, !dbg !30
  %728 = extractelement <1 x float> %727, i64 0, !dbg !30
  %729 = fmul float %728, %724, !dbg !30
  %730 = extractelement <4 x float> %705, i64 0, !dbg !31
  %731 = call float @llvm.amdgcn.exp2.f32(float %730), !dbg !31
  %732 = insertelement <4 x float> poison, float %731, i64 0, !dbg !33
  %733 = extractelement <4 x float> %705, i64 1, !dbg !31
  %734 = call float @llvm.amdgcn.exp2.f32(float %733), !dbg !31
  %735 = insertelement <4 x float> %732, float %734, i64 1, !dbg !33
  %736 = fadd float %731, %734, !dbg !32
  %737 = extractelement <4 x float> %705, i64 2, !dbg !31
  %738 = call float @llvm.amdgcn.exp2.f32(float %737), !dbg !31
  %739 = insertelement <4 x float> %735, float %738, i64 2, !dbg !33
  %740 = fadd float %736, %738, !dbg !32
  %741 = extractelement <4 x float> %705, i64 3, !dbg !31
  %742 = call float @llvm.amdgcn.exp2.f32(float %741), !dbg !31
  %743 = insertelement <4 x float> %739, float %742, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %743) #9, !dbg !33, !frisk.register_lifetimes !1
  %744 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %744) #9, !dbg !32, !frisk.register_lifetimes !1
  %745 = fadd float %740, %742, !dbg !32
  %746 = extractelement <4 x float> %706, i64 0, !dbg !31
  %747 = call float @llvm.amdgcn.exp2.f32(float %746), !dbg !31
  %748 = insertelement <4 x float> poison, float %747, i64 0, !dbg !33
  %749 = fadd float %745, %747, !dbg !32
  %750 = extractelement <4 x float> %706, i64 1, !dbg !31
  %751 = call float @llvm.amdgcn.exp2.f32(float %750), !dbg !31
  %752 = insertelement <4 x float> %748, float %751, i64 1, !dbg !33
  %753 = fadd float %749, %751, !dbg !32
  %754 = extractelement <4 x float> %706, i64 2, !dbg !31
  %755 = call float @llvm.amdgcn.exp2.f32(float %754), !dbg !31
  %756 = insertelement <4 x float> %752, float %755, i64 2, !dbg !33
  %757 = fadd float %753, %755, !dbg !32
  %758 = extractelement <4 x float> %706, i64 3, !dbg !31
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !31
  %760 = insertelement <4 x float> %756, float %759, i64 3, !dbg !33
  %math.result159 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %760) #9, !dbg !33, !frisk.register_lifetimes !1
  %761 = fptrunc <4 x float> %math.result159 to <4 x half>, !dbg !33
  %math.packed160 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %761) #9, !dbg !32, !frisk.register_lifetimes !1
  %762 = fadd float %757, %759, !dbg !32
  %763 = extractelement <4 x float> %707, i64 0, !dbg !31
  %764 = call float @llvm.amdgcn.exp2.f32(float %763), !dbg !31
  %765 = insertelement <4 x float> poison, float %764, i64 0, !dbg !33
  %766 = fadd float %762, %764, !dbg !32
  %767 = extractelement <4 x float> %707, i64 1, !dbg !31
  %768 = call float @llvm.amdgcn.exp2.f32(float %767), !dbg !31
  %769 = insertelement <4 x float> %765, float %768, i64 1, !dbg !33
  %770 = fadd float %766, %768, !dbg !32
  %771 = extractelement <4 x float> %707, i64 2, !dbg !31
  %772 = call float @llvm.amdgcn.exp2.f32(float %771), !dbg !31
  %773 = insertelement <4 x float> %769, float %772, i64 2, !dbg !33
  %774 = fadd float %770, %772, !dbg !32
  %775 = extractelement <4 x float> %707, i64 3, !dbg !31
  %776 = call float @llvm.amdgcn.exp2.f32(float %775), !dbg !31
  %777 = insertelement <4 x float> %773, float %776, i64 3, !dbg !33
  %math.result161 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %777) #9, !dbg !33, !frisk.register_lifetimes !1
  %778 = fptrunc <4 x float> %math.result161 to <4 x half>, !dbg !33
  %math.packed162 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %778) #9, !dbg !32, !frisk.register_lifetimes !1
  %779 = fadd float %774, %776, !dbg !32
  %780 = extractelement <4 x float> %708, i64 0, !dbg !31
  %781 = call float @llvm.amdgcn.exp2.f32(float %780), !dbg !31
  %782 = insertelement <4 x float> poison, float %781, i64 0, !dbg !33
  %783 = fadd float %779, %781, !dbg !32
  %784 = extractelement <4 x float> %708, i64 1, !dbg !31
  %785 = call float @llvm.amdgcn.exp2.f32(float %784), !dbg !31
  %786 = insertelement <4 x float> %782, float %785, i64 1, !dbg !33
  %787 = fadd float %783, %785, !dbg !32
  %788 = extractelement <4 x float> %708, i64 2, !dbg !31
  %789 = call float @llvm.amdgcn.exp2.f32(float %788), !dbg !31
  %790 = insertelement <4 x float> %786, float %789, i64 2, !dbg !33
  %791 = fadd float %787, %789, !dbg !32
  %792 = extractelement <4 x float> %708, i64 3, !dbg !31
  %793 = call float @llvm.amdgcn.exp2.f32(float %792), !dbg !31
  %794 = insertelement <4 x float> %790, float %793, i64 3, !dbg !33
  %math.result163 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %794) #9, !dbg !33, !frisk.register_lifetimes !1
  %795 = fptrunc <4 x float> %math.result163 to <4 x half>, !dbg !33
  %math.packed164 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %795) #9, !dbg !32, !frisk.register_lifetimes !1
  %796 = fadd float %791, %793, !dbg !32
  %797 = extractelement <4 x float> %710, i64 0, !dbg !31
  %798 = call float @llvm.amdgcn.exp2.f32(float %797), !dbg !31
  %799 = insertelement <4 x float> poison, float %798, i64 0, !dbg !33
  %800 = extractelement <4 x float> %710, i64 1, !dbg !31
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !31
  %802 = insertelement <4 x float> %799, float %801, i64 1, !dbg !33
  %803 = fadd float %798, %801, !dbg !32
  %804 = extractelement <4 x float> %710, i64 2, !dbg !31
  %805 = call float @llvm.amdgcn.exp2.f32(float %804), !dbg !31
  %806 = insertelement <4 x float> %802, float %805, i64 2, !dbg !33
  %807 = fadd float %803, %805, !dbg !32
  %808 = extractelement <4 x float> %710, i64 3, !dbg !31
  %809 = call float @llvm.amdgcn.exp2.f32(float %808), !dbg !31
  %810 = insertelement <4 x float> %806, float %809, i64 3, !dbg !33
  %math.result165 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %810) #9, !dbg !33, !frisk.register_lifetimes !1
  %811 = fptrunc <4 x float> %math.result165 to <4 x half>, !dbg !33
  %math.packed166 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %811) #9, !dbg !32, !frisk.register_lifetimes !1
  %812 = fadd float %807, %809, !dbg !32
  %813 = extractelement <4 x float> %711, i64 0, !dbg !31
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !31
  %815 = insertelement <4 x float> poison, float %814, i64 0, !dbg !33
  %816 = fadd float %812, %814, !dbg !32
  %817 = extractelement <4 x float> %711, i64 1, !dbg !31
  %818 = call float @llvm.amdgcn.exp2.f32(float %817), !dbg !31
  %819 = insertelement <4 x float> %815, float %818, i64 1, !dbg !33
  %820 = fadd float %816, %818, !dbg !32
  %821 = extractelement <4 x float> %711, i64 2, !dbg !31
  %822 = call float @llvm.amdgcn.exp2.f32(float %821), !dbg !31
  %823 = insertelement <4 x float> %819, float %822, i64 2, !dbg !33
  %824 = fadd float %820, %822, !dbg !32
  %825 = extractelement <4 x float> %711, i64 3, !dbg !31
  %826 = call float @llvm.amdgcn.exp2.f32(float %825), !dbg !31
  %827 = insertelement <4 x float> %823, float %826, i64 3, !dbg !33
  %math.result167 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %827) #9, !dbg !33, !frisk.register_lifetimes !1
  %828 = fptrunc <4 x float> %math.result167 to <4 x half>, !dbg !33
  %math.packed168 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %828) #9, !dbg !32, !frisk.register_lifetimes !1
  %829 = fadd float %824, %826, !dbg !32
  %830 = extractelement <4 x float> %712, i64 0, !dbg !31
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !31
  %832 = insertelement <4 x float> poison, float %831, i64 0, !dbg !33
  %833 = fadd float %829, %831, !dbg !32
  %834 = extractelement <4 x float> %712, i64 1, !dbg !31
  %835 = call float @llvm.amdgcn.exp2.f32(float %834), !dbg !31
  %836 = insertelement <4 x float> %832, float %835, i64 1, !dbg !33
  %837 = fadd float %833, %835, !dbg !32
  %838 = extractelement <4 x float> %712, i64 2, !dbg !31
  %839 = call float @llvm.amdgcn.exp2.f32(float %838), !dbg !31
  %840 = insertelement <4 x float> %836, float %839, i64 2, !dbg !33
  %841 = fadd float %837, %839, !dbg !32
  %842 = extractelement <4 x float> %712, i64 3, !dbg !31
  %843 = call float @llvm.amdgcn.exp2.f32(float %842), !dbg !31
  %844 = insertelement <4 x float> %840, float %843, i64 3, !dbg !33
  %math.result169 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %844) #9, !dbg !33, !frisk.register_lifetimes !1
  %845 = fptrunc <4 x float> %math.result169 to <4 x half>, !dbg !33
  %math.packed170 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %845) #9, !dbg !32, !frisk.register_lifetimes !1
  %846 = fadd float %841, %843, !dbg !32
  %847 = extractelement <4 x float> %713, i64 0, !dbg !31
  %848 = call float @llvm.amdgcn.exp2.f32(float %847), !dbg !31
  %849 = insertelement <4 x float> poison, float %848, i64 0, !dbg !33
  %850 = fadd float %846, %848, !dbg !32
  %851 = extractelement <4 x float> %713, i64 1, !dbg !31
  %852 = call float @llvm.amdgcn.exp2.f32(float %851), !dbg !31
  %853 = insertelement <4 x float> %849, float %852, i64 1, !dbg !33
  %854 = fadd float %850, %852, !dbg !32
  %855 = extractelement <4 x float> %713, i64 2, !dbg !31
  %856 = call float @llvm.amdgcn.exp2.f32(float %855), !dbg !31
  %857 = insertelement <4 x float> %853, float %856, i64 2, !dbg !33
  %858 = fadd float %854, %856, !dbg !32
  %859 = extractelement <4 x float> %713, i64 3, !dbg !31
  %860 = call float @llvm.amdgcn.exp2.f32(float %859), !dbg !31
  %861 = insertelement <4 x float> %857, float %860, i64 3, !dbg !33
  %math.result171 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %861) #9, !dbg !33, !frisk.register_lifetimes !1
  %862 = fptrunc <4 x float> %math.result171 to <4 x half>, !dbg !33
  %math.packed172 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %862) #9, !dbg !32, !frisk.register_lifetimes !1
  %863 = fadd float %858, %860, !dbg !32
  %864 = bitcast float %796 to i32, !dbg !32
  %865 = call i32 @llvm.amdgcn.ds.bpermute(i32 %661, i32 %864), !dbg !32
  %866 = bitcast i32 %865 to float, !dbg !32
  %867 = fadd float %796, %866, !dbg !32
  %868 = bitcast float %867 to i32, !dbg !32
  %869 = call i32 @llvm.amdgcn.ds.bpermute(i32 %669, i32 %868), !dbg !32
  %870 = bitcast i32 %869 to float, !dbg !32
  %871 = fadd float %867, %870, !dbg !32
  %872 = fadd float %871, %721, !dbg !32
  %873 = insertelement <1 x float> poison, float %872, i64 0, !dbg !7
  %874 = insertvalue [2 x <1 x float>] poison, <1 x float> %873, 0, !dbg !7
  %875 = bitcast float %863 to i32, !dbg !32
  %876 = call i32 @llvm.amdgcn.ds.bpermute(i32 %661, i32 %875), !dbg !32
  %877 = bitcast i32 %876 to float, !dbg !32
  %878 = fadd float %863, %877, !dbg !32
  %879 = bitcast float %878 to i32, !dbg !32
  %880 = call i32 @llvm.amdgcn.ds.bpermute(i32 %669, i32 %879), !dbg !32
  %881 = bitcast i32 %880 to float, !dbg !32
  %882 = fadd float %878, %881, !dbg !32
  %883 = fadd float %882, %729, !dbg !32
  %884 = insertelement <1 x float> poison, float %883, i64 0, !dbg !7
  %885 = insertvalue [2 x <1 x float>] %874, <1 x float> %884, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread436 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !14, !range !38, !frisk.register_lifetimes !1
  %address.local437 = zext nneg i32 %address.thread436 to i64, !dbg !12
  %address.local438 = and i64 %address.local437, 60, !dbg !14
  %address.local439 = add nuw nsw i64 %address.local438, %address.local437, !dbg !14
  %address.local440 = shl nuw nsw i64 %address.local439, 3, !dbg !14
  %address.local441 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local440, !dbg !14
  %address.local442 = and i64 %address.local437, 63, !dbg !12
  %address.local443 = lshr i64 %address.local442, 4, !dbg !14
  %address.local444 = mul nuw nsw i64 %address.local443, 384, !dbg !14
  %address.local445 = getelementptr [2 x i8], ptr addrspace(3) %address.local441, i64 %address.local444, !dbg !14
  %address.local446 = lshr i64 %address.local437, 2, !dbg !16
  %address.local447 = mul nsw i64 %address.local446, -32, !dbg !14
  %address.local448 = getelementptr [2 x i8], ptr addrspace(3) %address.local445, i64 %address.local447, !dbg !14
  %886 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local448, i32 0) #0, !dbg !14
  %887 = bitcast <4 x i32> %886 to <8 x half>, !dbg !14
  %address.local449 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local440, !dbg !14
  %address.local450 = getelementptr [2 x i8], ptr addrspace(3) %address.local449, i64 %address.local444, !dbg !14
  %address.local451 = getelementptr [2 x i8], ptr addrspace(3) %address.local450, i64 %address.local447, !dbg !14
  %address.local452 = getelementptr i8, ptr addrspace(3) %address.local451, i64 256, !dbg !14
  %888 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local452, i32 0) #0, !dbg !14
  %889 = bitcast <4 x i32> %888 to <8 x half>, !dbg !14
  %address.local453 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local440, !dbg !14
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) %address.local453, i64 %address.local444, !dbg !14
  %address.local455 = getelementptr [2 x i8], ptr addrspace(3) %address.local454, i64 %address.local447, !dbg !14
  %address.local456 = getelementptr i8, ptr addrspace(3) %address.local455, i64 512, !dbg !14
  %890 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local456, i32 0) #0, !dbg !14
  %891 = bitcast <4 x i32> %890 to <8 x half>, !dbg !14
  %address.local457 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local440, !dbg !14
  %address.local458 = getelementptr [2 x i8], ptr addrspace(3) %address.local457, i64 %address.local444, !dbg !14
  %address.local459 = getelementptr [2 x i8], ptr addrspace(3) %address.local458, i64 %address.local447, !dbg !14
  %address.local460 = getelementptr i8, ptr addrspace(3) %address.local459, i64 768, !dbg !14
  %892 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local460, i32 0) #0, !dbg !14
  %893 = bitcast <4 x i32> %892 to <8 x half>, !dbg !14
  %894 = add nuw nsw i64 %102, %15, !dbg !17
  %895 = shl i64 %894, 19, !dbg !17
  %896 = and i64 %895, 16252928, !dbg !17
  %897 = add i64 %896, %108, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %898 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %897, !dbg !17
  %899 = getelementptr i8, ptr addrspace(1) %898, i64 64, !dbg !17
  %address.local461 = shl nuw nsw i64 %address.local437, 3, !dbg !11
  %address.local462 = and i64 %address.local461, 1536, !dbg !15
  %address.local463 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local462, !dbg !17
  %address.local464 = ptrtoint ptr addrspace(3) %address.local463 to i64, !dbg !17
  %address.local465 = trunc i64 %address.local464 to i32, !dbg !17
  %900 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local465), !dbg !17
  %901 = zext i32 %900 to i64, !dbg !17
  %902 = inttoptr i64 %901 to ptr addrspace(3), !dbg !17
  %address.local466 = shl nuw nsw i64 %address.local437, 5, !dbg !17
  %address.local467 = and i64 %address.local466, 8064, !dbg !17
  %address.local468 = shl nuw nsw i64 %address.local437, 3, !dbg !17
  %address.local469 = and i64 %address.local468, 24, !dbg !17
  %address.local470 = or disjoint i64 %address.local467, %address.local469, !dbg !17
  %address.local471 = getelementptr [2 x i8], ptr addrspace(1) %899, i64 %address.local470, !dbg !17
  %903 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %904 = ptrtoint ptr addrspace(1) %address.local471 to i64, !dbg !17
  %905 = sub i64 %904, %903, !dbg !17
  %dma.buffer.offset887 = trunc i64 %905 to i32, !dbg !17
  %906 = trunc i64 %903 to i32, !dbg !17
  %907 = insertelement <4 x i32> poison, i32 %906, i64 0, !dbg !17
  %908 = lshr i64 %903, 32, !dbg !17
  %909 = trunc i64 %908 to i32, !dbg !17
  %910 = insertelement <4 x i32> %907, i32 %909, i64 1, !dbg !17
  %911 = insertelement <4 x i32> %910, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc888 = insertelement <4 x i32> %911, i32 131072, i64 3, !dbg !17
  %912 = ptrtoint ptr addrspace(3) %902 to i32, !dbg !17
  %dma.buffer.lds889 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %912), !dbg !17
  %913 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds889, i32 %dma.buffer.offset887, <4 x i32> %dma.buffer.rsrc888) #10, !dbg !17
  %914 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %915 = extractvalue [2 x <8 x float>] %63, 0, !dbg !34
  %916 = shufflevector <8 x float> %915, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %917 = fmul <4 x float> %916, %718, !dbg !34
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %914, <4 x float> %917) #9, !dbg !14
  %919 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %919, <4 x float> %918) #9, !dbg !14
  %921 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %921, <4 x float> %920) #9, !dbg !14
  %923 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %923, <4 x float> %922) #9, !dbg !14
  %925 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %926 = shufflevector <8 x float> %915, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %927 = fmul <4 x float> %926, %718, !dbg !34
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %925, <4 x float> %927) #9, !dbg !14
  %929 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %929, <4 x float> %928) #9, !dbg !14
  %931 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %931, <4 x float> %930) #9, !dbg !14
  %933 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %933, <4 x float> %932) #9, !dbg !14
  %935 = extractvalue [2 x <8 x float>] %63, 1, !dbg !34
  %936 = shufflevector <8 x float> %935, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %937 = fmul <4 x float> %936, %726, !dbg !34
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %914, <4 x float> %937) #9, !dbg !14
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %919, <4 x float> %938) #9, !dbg !14
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %921, <4 x float> %939) #9, !dbg !14
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %923, <4 x float> %940) #9, !dbg !14
  %942 = shufflevector <8 x float> %935, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %943 = fmul <4 x float> %942, %726, !dbg !34
  %944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %925, <4 x float> %943) #9, !dbg !14
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %929, <4 x float> %944) #9, !dbg !14
  %946 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %931, <4 x float> %945) #9, !dbg !14
  %947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %933, <4 x float> %946) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %948 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local448, i32 0) #0, !dbg !14
  %949 = bitcast <4 x i32> %948 to <8 x half>, !dbg !14
  %950 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local452, i32 0) #0, !dbg !14
  %951 = bitcast <4 x i32> %950 to <8 x half>, !dbg !14
  %952 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local456, i32 0) #0, !dbg !14
  %953 = bitcast <4 x i32> %952 to <8 x half>, !dbg !14
  %954 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local460, i32 0) #0, !dbg !14
  %955 = bitcast <4 x i32> %954 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %956 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %897, !dbg !17
  %957 = getelementptr i8, ptr addrspace(1) %956, i64 128, !dbg !17
  %958 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local465), !dbg !17
  %959 = zext i32 %958 to i64, !dbg !17
  %960 = inttoptr i64 %959 to ptr addrspace(3), !dbg !17
  %address.local472 = getelementptr [2 x i8], ptr addrspace(1) %957, i64 %address.local470, !dbg !17
  %961 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %962 = ptrtoint ptr addrspace(1) %address.local472 to i64, !dbg !17
  %963 = sub i64 %962, %961, !dbg !17
  %dma.buffer.offset884 = trunc i64 %963 to i32, !dbg !17
  %964 = trunc i64 %961 to i32, !dbg !17
  %965 = insertelement <4 x i32> poison, i32 %964, i64 0, !dbg !17
  %966 = lshr i64 %961, 32, !dbg !17
  %967 = trunc i64 %966 to i32, !dbg !17
  %968 = insertelement <4 x i32> %965, i32 %967, i64 1, !dbg !17
  %969 = insertelement <4 x i32> %968, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc885 = insertelement <4 x i32> %969, i32 131072, i64 3, !dbg !17
  %970 = ptrtoint ptr addrspace(3) %960 to i32, !dbg !17
  %dma.buffer.lds886 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %970), !dbg !17
  %971 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds886, i32 %dma.buffer.offset884, <4 x i32> %dma.buffer.rsrc885) #10, !dbg !17
  %972 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %973 = extractvalue [2 x <8 x float>] %66, 0, !dbg !34
  %974 = shufflevector <8 x float> %973, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %975 = fmul <4 x float> %974, %718, !dbg !34
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %972, <4 x float> %975) #9, !dbg !14
  %977 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %977, <4 x float> %976) #9, !dbg !14
  %979 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %979, <4 x float> %978) #9, !dbg !14
  %981 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %981, <4 x float> %980) #9, !dbg !14
  %983 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %984 = shufflevector <8 x float> %973, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %985 = fmul <4 x float> %984, %718, !dbg !34
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %983, <4 x float> %985) #9, !dbg !14
  %987 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %987, <4 x float> %986) #9, !dbg !14
  %989 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %989, <4 x float> %988) #9, !dbg !14
  %991 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %991, <4 x float> %990) #9, !dbg !14
  %993 = extractvalue [2 x <8 x float>] %66, 1, !dbg !34
  %994 = shufflevector <8 x float> %993, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %995 = fmul <4 x float> %994, %726, !dbg !34
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %972, <4 x float> %995) #9, !dbg !14
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %977, <4 x float> %996) #9, !dbg !14
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %979, <4 x float> %997) #9, !dbg !14
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %981, <4 x float> %998) #9, !dbg !14
  %1000 = shufflevector <8 x float> %993, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1001 = fmul <4 x float> %1000, %726, !dbg !34
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %983, <4 x float> %1001) #9, !dbg !14
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %987, <4 x float> %1002) #9, !dbg !14
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %989, <4 x float> %1003) #9, !dbg !14
  %1005 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %991, <4 x float> %1004) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1006 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local448, i32 0) #0, !dbg !14
  %1007 = bitcast <4 x i32> %1006 to <8 x half>, !dbg !14
  %1008 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local452, i32 0) #0, !dbg !14
  %1009 = bitcast <4 x i32> %1008 to <8 x half>, !dbg !14
  %1010 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local456, i32 0) #0, !dbg !14
  %1011 = bitcast <4 x i32> %1010 to <8 x half>, !dbg !14
  %1012 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local460, i32 0) #0, !dbg !14
  %1013 = bitcast <4 x i32> %1012 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1014 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %897, !dbg !17
  %1015 = getelementptr i8, ptr addrspace(1) %1014, i64 192, !dbg !17
  %1016 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local465), !dbg !17
  %1017 = zext i32 %1016 to i64, !dbg !17
  %1018 = inttoptr i64 %1017 to ptr addrspace(3), !dbg !17
  %address.local473 = getelementptr [2 x i8], ptr addrspace(1) %1015, i64 %address.local470, !dbg !17
  %1019 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1020 = ptrtoint ptr addrspace(1) %address.local473 to i64, !dbg !17
  %1021 = sub i64 %1020, %1019, !dbg !17
  %dma.buffer.offset = trunc i64 %1021 to i32, !dbg !17
  %1022 = trunc i64 %1019 to i32, !dbg !17
  %1023 = insertelement <4 x i32> poison, i32 %1022, i64 0, !dbg !17
  %1024 = lshr i64 %1019, 32, !dbg !17
  %1025 = trunc i64 %1024 to i32, !dbg !17
  %1026 = insertelement <4 x i32> %1023, i32 %1025, i64 1, !dbg !17
  %1027 = insertelement <4 x i32> %1026, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %1027, i32 131072, i64 3, !dbg !17
  %1028 = ptrtoint ptr addrspace(3) %1018 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1028), !dbg !17
  %1029 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %1030 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1031 = extractvalue [2 x <8 x float>] %67, 0, !dbg !34
  %1032 = shufflevector <8 x float> %1031, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1033 = fmul <4 x float> %1032, %718, !dbg !34
  %1034 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1030, <4 x float> %1033) #9, !dbg !14
  %1035 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %1035, <4 x float> %1034) #9, !dbg !14
  %1037 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %1037, <4 x float> %1036) #9, !dbg !14
  %1039 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %1039, <4 x float> %1038) #9, !dbg !14
  %1041 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1042 = shufflevector <8 x float> %1031, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1043 = fmul <4 x float> %1042, %718, !dbg !34
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1041, <4 x float> %1043) #9, !dbg !14
  %1045 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %1045, <4 x float> %1044) #9, !dbg !14
  %1047 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1048 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %1047, <4 x float> %1046) #9, !dbg !14
  %1049 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %1049, <4 x float> %1048) #9, !dbg !14
  %1051 = extractvalue [2 x <8 x float>] %67, 1, !dbg !34
  %1052 = shufflevector <8 x float> %1051, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1053 = fmul <4 x float> %1052, %726, !dbg !34
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %1030, <4 x float> %1053) #9, !dbg !14
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %1035, <4 x float> %1054) #9, !dbg !14
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %1037, <4 x float> %1055) #9, !dbg !14
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %1039, <4 x float> %1056) #9, !dbg !14
  %1058 = shufflevector <8 x float> %1051, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1059 = fmul <4 x float> %1058, %726, !dbg !34
  %1060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %1041, <4 x float> %1059) #9, !dbg !14
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %1045, <4 x float> %1060) #9, !dbg !14
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %1047, <4 x float> %1061) #9, !dbg !14
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %1049, <4 x float> %1062) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1064 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local448, i32 0) #0, !dbg !14
  %1065 = bitcast <4 x i32> %1064 to <8 x half>, !dbg !14
  %1066 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local452, i32 0) #0, !dbg !14
  %1067 = bitcast <4 x i32> %1066 to <8 x half>, !dbg !14
  %1068 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local456, i32 0) #0, !dbg !14
  %1069 = bitcast <4 x i32> %1068 to <8 x half>, !dbg !14
  %1070 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local460, i32 0) #0, !dbg !14
  %1071 = bitcast <4 x i32> %1070 to <8 x half>, !dbg !14
  %1072 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1073 = extractvalue [2 x <8 x float>] %68, 0, !dbg !34
  %1074 = shufflevector <8 x float> %1073, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1075 = fmul <4 x float> %1074, %718, !dbg !34
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1072, <4 x float> %1075) #9, !dbg !14
  %1077 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %1077, <4 x float> %1076) #9, !dbg !14
  %1079 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %1079, <4 x float> %1078) #9, !dbg !14
  %1081 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %1081, <4 x float> %1080) #9, !dbg !14
  %1083 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1084 = shufflevector <8 x float> %1073, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1085 = fmul <4 x float> %1084, %718, !dbg !34
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1083, <4 x float> %1085) #9, !dbg !14
  %1087 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed160, <4 x half> %1087, <4 x float> %1086) #9, !dbg !14
  %1089 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed162, <4 x half> %1089, <4 x float> %1088) #9, !dbg !14
  %1091 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed164, <4 x half> %1091, <4 x float> %1090) #9, !dbg !14
  %1093 = extractvalue [2 x <8 x float>] %68, 1, !dbg !34
  %1094 = shufflevector <8 x float> %1093, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1095 = fmul <4 x float> %1094, %726, !dbg !34
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %1072, <4 x float> %1095) #9, !dbg !14
  %1097 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %1077, <4 x float> %1096) #9, !dbg !14
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %1079, <4 x float> %1097) #9, !dbg !14
  %1099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %1081, <4 x float> %1098) #9, !dbg !14
  %1100 = shufflevector <8 x float> %1093, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1101 = fmul <4 x float> %1100, %726, !dbg !34
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed166, <4 x half> %1083, <4 x float> %1101) #9, !dbg !14
  %1103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed168, <4 x half> %1087, <4 x float> %1102) #9, !dbg !14
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed170, <4 x half> %1089, <4 x float> %1103) #9, !dbg !14
  %1105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed172, <4 x half> %1091, <4 x float> %1104) #9, !dbg !14
  %1106 = shufflevector <4 x float> %924, <4 x float> %934, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1107 = insertvalue [2 x <8 x float>] poison, <8 x float> %1106, 0, !dbg !7
  %1108 = shufflevector <4 x float> %941, <4 x float> %947, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1109 = insertvalue [2 x <8 x float>] %1107, <8 x float> %1108, 1, !dbg !7
  %1110 = insertvalue [2 x <1 x float>] poison, <1 x float> %676, 0, !dbg !7
  %1111 = insertvalue [2 x <1 x float>] %1110, <1 x float> %703, 1, !dbg !7
  %1112 = shufflevector <4 x float> %982, <4 x float> %992, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1113 = insertvalue [2 x <8 x float>] poison, <8 x float> %1112, 0, !dbg !7
  %1114 = shufflevector <4 x float> %999, <4 x float> %1005, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1115 = insertvalue [2 x <8 x float>] %1113, <8 x float> %1114, 1, !dbg !7
  %1116 = shufflevector <4 x float> %1040, <4 x float> %1050, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1117 = insertvalue [2 x <8 x float>] poison, <8 x float> %1116, 0, !dbg !7
  %1118 = shufflevector <4 x float> %1057, <4 x float> %1063, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1119 = insertvalue [2 x <8 x float>] %1117, <8 x float> %1118, 1, !dbg !7
  %1120 = shufflevector <4 x float> %1082, <4 x float> %1092, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1121 = insertvalue [2 x <8 x float>] poison, <8 x float> %1120, 0, !dbg !7
  %1122 = shufflevector <4 x float> %1099, <4 x float> %1105, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1123 = insertvalue [2 x <8 x float>] %1121, <8 x float> %1122, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1124 = add nuw nsw i64 %62, 1, !dbg !7
  br label %61, !dbg !7

1125:                                             ; preds = %61
  %1126 = and i32 %34, 16, !dbg !7
  %.not = icmp eq i32 %1126, 0, !dbg !7
  %1127 = and i32 %34, 32, !dbg !7
  %.not18 = icmp eq i32 %1127, 0, !dbg !7
  %1128 = extractvalue [2 x <8 x float>] %68, 0, !dbg !7
  %1129 = extractelement <8 x float> %1128, i64 0, !dbg !7
  %1130 = extractelement <8 x float> %1128, i64 1, !dbg !7
  %1131 = extractelement <8 x float> %1128, i64 2, !dbg !7
  %1132 = extractelement <8 x float> %1128, i64 3, !dbg !7
  %1133 = select i1 %.not, float %1130, float %1129, !dbg !7
  %1134 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1135 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1134), !dbg !7
  %1136 = add nuw nsw i32 %1135, 64, !dbg !7
  %1137 = xor i32 %1135, 16, !dbg !7
  %1138 = and i32 %1136, 192, !dbg !7
  %1139 = icmp samesign ult i32 %1137, %1138, !dbg !7
  %1140 = select i1 %1139, i32 %1137, i32 %1135, !dbg !7
  %1141 = shl nuw nsw i32 %1140, 2, !dbg !7
  %1142 = bitcast float %1133 to i32, !dbg !7
  %1143 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1142), !dbg !7
  %1144 = bitcast i32 %1143 to float, !dbg !7
  %1145 = select i1 %.not, float %1129, float %1144, !dbg !7
  %1146 = select i1 %.not, float %1144, float %1130, !dbg !7
  %1147 = select i1 %.not, float %1132, float %1131, !dbg !7
  %1148 = bitcast float %1147 to i32, !dbg !7
  %1149 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1148), !dbg !7
  %1150 = bitcast i32 %1149 to float, !dbg !7
  %1151 = select i1 %.not, float %1131, float %1150, !dbg !7
  %1152 = select i1 %.not, float %1150, float %1132, !dbg !7
  %1153 = select i1 %.not18, float %1151, float %1145, !dbg !7
  %1154 = xor i32 %1135, 32, !dbg !7
  %1155 = icmp samesign ult i32 %1154, %1138, !dbg !7
  %1156 = select i1 %1155, i32 %1154, i32 %1135, !dbg !7
  %1157 = shl nuw nsw i32 %1156, 2, !dbg !7
  %1158 = bitcast float %1153 to i32, !dbg !7
  %1159 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1158), !dbg !7
  %1160 = bitcast i32 %1159 to float, !dbg !7
  %1161 = select i1 %.not18, float %1145, float %1160, !dbg !7
  %1162 = select i1 %.not18, float %1160, float %1151, !dbg !7
  %1163 = select i1 %.not18, float %1152, float %1146, !dbg !7
  %1164 = bitcast float %1163 to i32, !dbg !7
  %1165 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1164), !dbg !7
  %1166 = bitcast i32 %1165 to float, !dbg !7
  %1167 = select i1 %.not18, float %1146, float %1166, !dbg !7
  %1168 = select i1 %.not18, float %1166, float %1152, !dbg !7
  %1169 = extractvalue [2 x <8 x float>] %67, 0, !dbg !7
  %1170 = extractelement <8 x float> %1169, i64 0, !dbg !7
  %1171 = extractelement <8 x float> %1169, i64 1, !dbg !7
  %1172 = extractelement <8 x float> %1169, i64 2, !dbg !7
  %1173 = extractelement <8 x float> %1169, i64 3, !dbg !7
  %1174 = select i1 %.not, float %1171, float %1170, !dbg !7
  %1175 = bitcast float %1174 to i32, !dbg !7
  %1176 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1175), !dbg !7
  %1177 = bitcast i32 %1176 to float, !dbg !7
  %1178 = select i1 %.not, float %1170, float %1177, !dbg !7
  %1179 = select i1 %.not, float %1177, float %1171, !dbg !7
  %1180 = select i1 %.not, float %1173, float %1172, !dbg !7
  %1181 = bitcast float %1180 to i32, !dbg !7
  %1182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1181), !dbg !7
  %1183 = bitcast i32 %1182 to float, !dbg !7
  %1184 = select i1 %.not, float %1172, float %1183, !dbg !7
  %1185 = select i1 %.not, float %1183, float %1173, !dbg !7
  %1186 = select i1 %.not18, float %1184, float %1178, !dbg !7
  %1187 = bitcast float %1186 to i32, !dbg !7
  %1188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1187), !dbg !7
  %1189 = bitcast i32 %1188 to float, !dbg !7
  %1190 = select i1 %.not18, float %1178, float %1189, !dbg !7
  %1191 = select i1 %.not18, float %1189, float %1184, !dbg !7
  %1192 = select i1 %.not18, float %1185, float %1179, !dbg !7
  %1193 = bitcast float %1192 to i32, !dbg !7
  %1194 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1193), !dbg !7
  %1195 = bitcast i32 %1194 to float, !dbg !7
  %1196 = select i1 %.not18, float %1179, float %1195, !dbg !7
  %1197 = select i1 %.not18, float %1195, float %1185, !dbg !7
  %1198 = extractvalue [2 x <8 x float>] %66, 0, !dbg !7
  %1199 = extractelement <8 x float> %1198, i64 0, !dbg !7
  %1200 = extractelement <8 x float> %1198, i64 1, !dbg !7
  %1201 = extractelement <8 x float> %1198, i64 2, !dbg !7
  %1202 = extractelement <8 x float> %1198, i64 3, !dbg !7
  %1203 = select i1 %.not, float %1200, float %1199, !dbg !7
  %1204 = bitcast float %1203 to i32, !dbg !7
  %1205 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1204), !dbg !7
  %1206 = bitcast i32 %1205 to float, !dbg !7
  %1207 = select i1 %.not, float %1199, float %1206, !dbg !7
  %1208 = select i1 %.not, float %1206, float %1200, !dbg !7
  %1209 = select i1 %.not, float %1202, float %1201, !dbg !7
  %1210 = bitcast float %1209 to i32, !dbg !7
  %1211 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1210), !dbg !7
  %1212 = bitcast i32 %1211 to float, !dbg !7
  %1213 = select i1 %.not, float %1201, float %1212, !dbg !7
  %1214 = select i1 %.not, float %1212, float %1202, !dbg !7
  %1215 = select i1 %.not18, float %1213, float %1207, !dbg !7
  %1216 = bitcast float %1215 to i32, !dbg !7
  %1217 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1216), !dbg !7
  %1218 = bitcast i32 %1217 to float, !dbg !7
  %1219 = select i1 %.not18, float %1207, float %1218, !dbg !7
  %1220 = select i1 %.not18, float %1218, float %1213, !dbg !7
  %1221 = select i1 %.not18, float %1214, float %1208, !dbg !7
  %1222 = bitcast float %1221 to i32, !dbg !7
  %1223 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1222), !dbg !7
  %1224 = bitcast i32 %1223 to float, !dbg !7
  %1225 = select i1 %.not18, float %1208, float %1224, !dbg !7
  %1226 = select i1 %.not18, float %1224, float %1214, !dbg !7
  %1227 = extractvalue [2 x <8 x float>] %63, 0, !dbg !7
  %1228 = extractelement <8 x float> %1227, i64 0, !dbg !7
  %1229 = extractelement <8 x float> %1227, i64 1, !dbg !7
  %1230 = extractelement <8 x float> %1227, i64 2, !dbg !7
  %1231 = extractelement <8 x float> %1227, i64 3, !dbg !7
  %1232 = select i1 %.not, float %1229, float %1228, !dbg !7
  %1233 = bitcast float %1232 to i32, !dbg !7
  %1234 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1233), !dbg !7
  %1235 = bitcast i32 %1234 to float, !dbg !7
  %1236 = select i1 %.not, float %1228, float %1235, !dbg !7
  %1237 = select i1 %.not, float %1235, float %1229, !dbg !7
  %1238 = select i1 %.not, float %1231, float %1230, !dbg !7
  %1239 = bitcast float %1238 to i32, !dbg !7
  %1240 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1239), !dbg !7
  %1241 = bitcast i32 %1240 to float, !dbg !7
  %1242 = select i1 %.not, float %1230, float %1241, !dbg !7
  %1243 = select i1 %.not, float %1241, float %1231, !dbg !7
  %1244 = select i1 %.not18, float %1242, float %1236, !dbg !7
  %1245 = bitcast float %1244 to i32, !dbg !7
  %1246 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1245), !dbg !7
  %1247 = select i1 %.not18, float %1243, float %1237, !dbg !7
  %1248 = bitcast float %1247 to i32, !dbg !7
  %1249 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1248), !dbg !7
  %1250 = extractvalue [2 x <1 x float>] %64, 0, !dbg !18
  %1251 = shufflevector <1 x float> %1250, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %1252 = extractelement <8 x float> %1128, i64 4, !dbg !7
  %1253 = extractelement <8 x float> %1128, i64 5, !dbg !7
  %1254 = extractelement <8 x float> %1128, i64 6, !dbg !7
  %1255 = extractelement <8 x float> %1128, i64 7, !dbg !7
  %1256 = select i1 %.not, float %1253, float %1252, !dbg !7
  %1257 = bitcast float %1256 to i32, !dbg !7
  %1258 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1257), !dbg !7
  %1259 = bitcast i32 %1258 to float, !dbg !7
  %1260 = select i1 %.not, float %1252, float %1259, !dbg !7
  %1261 = select i1 %.not, float %1259, float %1253, !dbg !7
  %1262 = select i1 %.not, float %1255, float %1254, !dbg !7
  %1263 = bitcast float %1262 to i32, !dbg !7
  %1264 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1263), !dbg !7
  %1265 = bitcast i32 %1264 to float, !dbg !7
  %1266 = select i1 %.not, float %1254, float %1265, !dbg !7
  %1267 = select i1 %.not, float %1265, float %1255, !dbg !7
  %1268 = select i1 %.not18, float %1266, float %1260, !dbg !7
  %1269 = bitcast float %1268 to i32, !dbg !7
  %1270 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1269), !dbg !7
  %1271 = bitcast i32 %1270 to float, !dbg !7
  %1272 = select i1 %.not18, float %1260, float %1271, !dbg !7
  %1273 = select i1 %.not18, float %1271, float %1266, !dbg !7
  %1274 = select i1 %.not18, float %1267, float %1261, !dbg !7
  %1275 = bitcast float %1274 to i32, !dbg !7
  %1276 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1275), !dbg !7
  %1277 = bitcast i32 %1276 to float, !dbg !7
  %1278 = select i1 %.not18, float %1261, float %1277, !dbg !7
  %1279 = select i1 %.not18, float %1277, float %1267, !dbg !7
  %1280 = extractelement <8 x float> %1169, i64 4, !dbg !7
  %1281 = extractelement <8 x float> %1169, i64 5, !dbg !7
  %1282 = extractelement <8 x float> %1169, i64 6, !dbg !7
  %1283 = extractelement <8 x float> %1169, i64 7, !dbg !7
  %1284 = select i1 %.not, float %1281, float %1280, !dbg !7
  %1285 = bitcast float %1284 to i32, !dbg !7
  %1286 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1285), !dbg !7
  %1287 = bitcast i32 %1286 to float, !dbg !7
  %1288 = select i1 %.not, float %1280, float %1287, !dbg !7
  %1289 = select i1 %.not, float %1287, float %1281, !dbg !7
  %1290 = select i1 %.not, float %1283, float %1282, !dbg !7
  %1291 = bitcast float %1290 to i32, !dbg !7
  %1292 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1291), !dbg !7
  %1293 = bitcast i32 %1292 to float, !dbg !7
  %1294 = select i1 %.not, float %1282, float %1293, !dbg !7
  %1295 = select i1 %.not, float %1293, float %1283, !dbg !7
  %1296 = select i1 %.not18, float %1294, float %1288, !dbg !7
  %1297 = bitcast float %1296 to i32, !dbg !7
  %1298 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1297), !dbg !7
  %1299 = bitcast i32 %1298 to float, !dbg !7
  %1300 = select i1 %.not18, float %1288, float %1299, !dbg !7
  %1301 = select i1 %.not18, float %1299, float %1294, !dbg !7
  %1302 = select i1 %.not18, float %1295, float %1289, !dbg !7
  %1303 = bitcast float %1302 to i32, !dbg !7
  %1304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1303), !dbg !7
  %1305 = bitcast i32 %1304 to float, !dbg !7
  %1306 = select i1 %.not18, float %1289, float %1305, !dbg !7
  %1307 = select i1 %.not18, float %1305, float %1295, !dbg !7
  %1308 = extractelement <8 x float> %1198, i64 4, !dbg !7
  %1309 = extractelement <8 x float> %1198, i64 5, !dbg !7
  %1310 = extractelement <8 x float> %1198, i64 6, !dbg !7
  %1311 = extractelement <8 x float> %1198, i64 7, !dbg !7
  %1312 = select i1 %.not, float %1309, float %1308, !dbg !7
  %1313 = bitcast float %1312 to i32, !dbg !7
  %1314 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1313), !dbg !7
  %1315 = bitcast i32 %1314 to float, !dbg !7
  %1316 = select i1 %.not, float %1308, float %1315, !dbg !7
  %1317 = select i1 %.not, float %1315, float %1309, !dbg !7
  %1318 = select i1 %.not, float %1311, float %1310, !dbg !7
  %1319 = bitcast float %1318 to i32, !dbg !7
  %1320 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1319), !dbg !7
  %1321 = bitcast i32 %1320 to float, !dbg !7
  %1322 = select i1 %.not, float %1310, float %1321, !dbg !7
  %1323 = select i1 %.not, float %1321, float %1311, !dbg !7
  %1324 = select i1 %.not18, float %1322, float %1316, !dbg !7
  %1325 = bitcast float %1324 to i32, !dbg !7
  %1326 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1325), !dbg !7
  %1327 = bitcast i32 %1326 to float, !dbg !7
  %1328 = select i1 %.not18, float %1316, float %1327, !dbg !7
  %1329 = select i1 %.not18, float %1327, float %1322, !dbg !7
  %1330 = select i1 %.not18, float %1323, float %1317, !dbg !7
  %1331 = bitcast float %1330 to i32, !dbg !7
  %1332 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1331), !dbg !7
  %1333 = bitcast i32 %1332 to float, !dbg !7
  %1334 = select i1 %.not18, float %1317, float %1333, !dbg !7
  %1335 = select i1 %.not18, float %1333, float %1323, !dbg !7
  %1336 = extractelement <8 x float> %1227, i64 4, !dbg !7
  %1337 = extractelement <8 x float> %1227, i64 5, !dbg !7
  %1338 = extractelement <8 x float> %1227, i64 6, !dbg !7
  %1339 = extractelement <8 x float> %1227, i64 7, !dbg !7
  %1340 = select i1 %.not, float %1337, float %1336, !dbg !7
  %1341 = bitcast float %1340 to i32, !dbg !7
  %1342 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1341), !dbg !7
  %1343 = bitcast i32 %1342 to float, !dbg !7
  %1344 = select i1 %.not, float %1336, float %1343, !dbg !7
  %1345 = select i1 %.not, float %1343, float %1337, !dbg !7
  %1346 = select i1 %.not, float %1339, float %1338, !dbg !7
  %1347 = bitcast float %1346 to i32, !dbg !7
  %1348 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1347), !dbg !7
  %1349 = bitcast i32 %1348 to float, !dbg !7
  %1350 = select i1 %.not, float %1338, float %1349, !dbg !7
  %1351 = select i1 %.not, float %1349, float %1339, !dbg !7
  %1352 = select i1 %.not18, float %1350, float %1344, !dbg !7
  %1353 = bitcast float %1352 to i32, !dbg !7
  %1354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1353), !dbg !7
  %1355 = select i1 %.not18, float %1351, float %1345, !dbg !7
  %1356 = bitcast float %1355 to i32, !dbg !7
  %1357 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1356), !dbg !7
  %1358 = extractvalue [2 x <8 x float>] %68, 1, !dbg !7
  %1359 = extractelement <8 x float> %1358, i64 0, !dbg !7
  %1360 = extractelement <8 x float> %1358, i64 1, !dbg !7
  %1361 = extractelement <8 x float> %1358, i64 2, !dbg !7
  %1362 = extractelement <8 x float> %1358, i64 3, !dbg !7
  %1363 = select i1 %.not, float %1360, float %1359, !dbg !7
  %1364 = bitcast float %1363 to i32, !dbg !7
  %1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1364), !dbg !7
  %1366 = bitcast i32 %1365 to float, !dbg !7
  %1367 = select i1 %.not, float %1359, float %1366, !dbg !7
  %1368 = select i1 %.not, float %1366, float %1360, !dbg !7
  %1369 = select i1 %.not, float %1362, float %1361, !dbg !7
  %1370 = bitcast float %1369 to i32, !dbg !7
  %1371 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1370), !dbg !7
  %1372 = bitcast i32 %1371 to float, !dbg !7
  %1373 = select i1 %.not, float %1361, float %1372, !dbg !7
  %1374 = select i1 %.not, float %1372, float %1362, !dbg !7
  %1375 = select i1 %.not18, float %1373, float %1367, !dbg !7
  %1376 = bitcast float %1375 to i32, !dbg !7
  %1377 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1376), !dbg !7
  %1378 = bitcast i32 %1377 to float, !dbg !7
  %1379 = select i1 %.not18, float %1367, float %1378, !dbg !7
  %1380 = select i1 %.not18, float %1378, float %1373, !dbg !7
  %1381 = select i1 %.not18, float %1374, float %1368, !dbg !7
  %1382 = bitcast float %1381 to i32, !dbg !7
  %1383 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1382), !dbg !7
  %1384 = bitcast i32 %1383 to float, !dbg !7
  %1385 = select i1 %.not18, float %1368, float %1384, !dbg !7
  %1386 = select i1 %.not18, float %1384, float %1374, !dbg !7
  %1387 = extractvalue [2 x <8 x float>] %67, 1, !dbg !7
  %1388 = extractelement <8 x float> %1387, i64 0, !dbg !7
  %1389 = extractelement <8 x float> %1387, i64 1, !dbg !7
  %1390 = extractelement <8 x float> %1387, i64 2, !dbg !7
  %1391 = extractelement <8 x float> %1387, i64 3, !dbg !7
  %1392 = select i1 %.not, float %1389, float %1388, !dbg !7
  %1393 = bitcast float %1392 to i32, !dbg !7
  %1394 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1393), !dbg !7
  %1395 = bitcast i32 %1394 to float, !dbg !7
  %1396 = select i1 %.not, float %1388, float %1395, !dbg !7
  %1397 = select i1 %.not, float %1395, float %1389, !dbg !7
  %1398 = select i1 %.not, float %1391, float %1390, !dbg !7
  %1399 = bitcast float %1398 to i32, !dbg !7
  %1400 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1399), !dbg !7
  %1401 = bitcast i32 %1400 to float, !dbg !7
  %1402 = select i1 %.not, float %1390, float %1401, !dbg !7
  %1403 = select i1 %.not, float %1401, float %1391, !dbg !7
  %1404 = select i1 %.not18, float %1402, float %1396, !dbg !7
  %1405 = bitcast float %1404 to i32, !dbg !7
  %1406 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1405), !dbg !7
  %1407 = bitcast i32 %1406 to float, !dbg !7
  %1408 = select i1 %.not18, float %1396, float %1407, !dbg !7
  %1409 = select i1 %.not18, float %1407, float %1402, !dbg !7
  %1410 = select i1 %.not18, float %1403, float %1397, !dbg !7
  %1411 = bitcast float %1410 to i32, !dbg !7
  %1412 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1411), !dbg !7
  %1413 = bitcast i32 %1412 to float, !dbg !7
  %1414 = select i1 %.not18, float %1397, float %1413, !dbg !7
  %1415 = select i1 %.not18, float %1413, float %1403, !dbg !7
  %1416 = extractvalue [2 x <8 x float>] %66, 1, !dbg !7
  %1417 = extractelement <8 x float> %1416, i64 0, !dbg !7
  %1418 = extractelement <8 x float> %1416, i64 1, !dbg !7
  %1419 = extractelement <8 x float> %1416, i64 2, !dbg !7
  %1420 = extractelement <8 x float> %1416, i64 3, !dbg !7
  %1421 = select i1 %.not, float %1418, float %1417, !dbg !7
  %1422 = bitcast float %1421 to i32, !dbg !7
  %1423 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1422), !dbg !7
  %1424 = bitcast i32 %1423 to float, !dbg !7
  %1425 = select i1 %.not, float %1417, float %1424, !dbg !7
  %1426 = select i1 %.not, float %1424, float %1418, !dbg !7
  %1427 = select i1 %.not, float %1420, float %1419, !dbg !7
  %1428 = bitcast float %1427 to i32, !dbg !7
  %1429 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1428), !dbg !7
  %1430 = bitcast i32 %1429 to float, !dbg !7
  %1431 = select i1 %.not, float %1419, float %1430, !dbg !7
  %1432 = select i1 %.not, float %1430, float %1420, !dbg !7
  %1433 = select i1 %.not18, float %1431, float %1425, !dbg !7
  %1434 = bitcast float %1433 to i32, !dbg !7
  %1435 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1434), !dbg !7
  %1436 = bitcast i32 %1435 to float, !dbg !7
  %1437 = select i1 %.not18, float %1425, float %1436, !dbg !7
  %1438 = select i1 %.not18, float %1436, float %1431, !dbg !7
  %1439 = select i1 %.not18, float %1432, float %1426, !dbg !7
  %1440 = bitcast float %1439 to i32, !dbg !7
  %1441 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1440), !dbg !7
  %1442 = bitcast i32 %1441 to float, !dbg !7
  %1443 = select i1 %.not18, float %1426, float %1442, !dbg !7
  %1444 = select i1 %.not18, float %1442, float %1432, !dbg !7
  %1445 = extractvalue [2 x <8 x float>] %63, 1, !dbg !7
  %1446 = extractelement <8 x float> %1445, i64 0, !dbg !7
  %1447 = extractelement <8 x float> %1445, i64 1, !dbg !7
  %1448 = extractelement <8 x float> %1445, i64 2, !dbg !7
  %1449 = extractelement <8 x float> %1445, i64 3, !dbg !7
  %1450 = select i1 %.not, float %1447, float %1446, !dbg !7
  %1451 = bitcast float %1450 to i32, !dbg !7
  %1452 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1451), !dbg !7
  %1453 = bitcast i32 %1452 to float, !dbg !7
  %1454 = select i1 %.not, float %1446, float %1453, !dbg !7
  %1455 = select i1 %.not, float %1453, float %1447, !dbg !7
  %1456 = select i1 %.not, float %1449, float %1448, !dbg !7
  %1457 = bitcast float %1456 to i32, !dbg !7
  %1458 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1457), !dbg !7
  %1459 = bitcast i32 %1458 to float, !dbg !7
  %1460 = select i1 %.not, float %1448, float %1459, !dbg !7
  %1461 = select i1 %.not, float %1459, float %1449, !dbg !7
  %1462 = select i1 %.not18, float %1460, float %1454, !dbg !7
  %1463 = bitcast float %1462 to i32, !dbg !7
  %1464 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1463), !dbg !7
  %1465 = select i1 %.not18, float %1461, float %1455, !dbg !7
  %1466 = bitcast float %1465 to i32, !dbg !7
  %1467 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1466), !dbg !7
  %1468 = extractvalue [2 x <1 x float>] %64, 1, !dbg !18
  %1469 = shufflevector <1 x float> %1468, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %1470 = extractelement <8 x float> %1358, i64 4, !dbg !7
  %1471 = extractelement <8 x float> %1358, i64 5, !dbg !7
  %1472 = extractelement <8 x float> %1358, i64 6, !dbg !7
  %1473 = extractelement <8 x float> %1358, i64 7, !dbg !7
  %1474 = select i1 %.not, float %1471, float %1470, !dbg !7
  %1475 = bitcast float %1474 to i32, !dbg !7
  %1476 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1475), !dbg !7
  %1477 = bitcast i32 %1476 to float, !dbg !7
  %1478 = select i1 %.not, float %1470, float %1477, !dbg !7
  %1479 = select i1 %.not, float %1477, float %1471, !dbg !7
  %1480 = select i1 %.not, float %1473, float %1472, !dbg !7
  %1481 = bitcast float %1480 to i32, !dbg !7
  %1482 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1481), !dbg !7
  %1483 = bitcast i32 %1482 to float, !dbg !7
  %1484 = select i1 %.not, float %1472, float %1483, !dbg !7
  %1485 = select i1 %.not, float %1483, float %1473, !dbg !7
  %1486 = select i1 %.not18, float %1484, float %1478, !dbg !7
  %1487 = bitcast float %1486 to i32, !dbg !7
  %1488 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1487), !dbg !7
  %1489 = bitcast i32 %1488 to float, !dbg !7
  %1490 = select i1 %.not18, float %1478, float %1489, !dbg !7
  %1491 = select i1 %.not18, float %1489, float %1484, !dbg !7
  %1492 = select i1 %.not18, float %1485, float %1479, !dbg !7
  %1493 = bitcast float %1492 to i32, !dbg !7
  %1494 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1493), !dbg !7
  %1495 = bitcast i32 %1494 to float, !dbg !7
  %1496 = select i1 %.not18, float %1479, float %1495, !dbg !7
  %1497 = select i1 %.not18, float %1495, float %1485, !dbg !7
  %1498 = extractelement <8 x float> %1387, i64 4, !dbg !7
  %1499 = extractelement <8 x float> %1387, i64 5, !dbg !7
  %1500 = extractelement <8 x float> %1387, i64 6, !dbg !7
  %1501 = extractelement <8 x float> %1387, i64 7, !dbg !7
  %1502 = select i1 %.not, float %1499, float %1498, !dbg !7
  %1503 = bitcast float %1502 to i32, !dbg !7
  %1504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1503), !dbg !7
  %1505 = bitcast i32 %1504 to float, !dbg !7
  %1506 = select i1 %.not, float %1498, float %1505, !dbg !7
  %1507 = select i1 %.not, float %1505, float %1499, !dbg !7
  %1508 = select i1 %.not, float %1501, float %1500, !dbg !7
  %1509 = bitcast float %1508 to i32, !dbg !7
  %1510 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1509), !dbg !7
  %1511 = bitcast i32 %1510 to float, !dbg !7
  %1512 = select i1 %.not, float %1500, float %1511, !dbg !7
  %1513 = select i1 %.not, float %1511, float %1501, !dbg !7
  %1514 = select i1 %.not18, float %1512, float %1506, !dbg !7
  %1515 = bitcast float %1514 to i32, !dbg !7
  %1516 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1515), !dbg !7
  %1517 = bitcast i32 %1516 to float, !dbg !7
  %1518 = select i1 %.not18, float %1506, float %1517, !dbg !7
  %1519 = select i1 %.not18, float %1517, float %1512, !dbg !7
  %1520 = select i1 %.not18, float %1513, float %1507, !dbg !7
  %1521 = bitcast float %1520 to i32, !dbg !7
  %1522 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1521), !dbg !7
  %1523 = bitcast i32 %1522 to float, !dbg !7
  %1524 = select i1 %.not18, float %1507, float %1523, !dbg !7
  %1525 = select i1 %.not18, float %1523, float %1513, !dbg !7
  %1526 = extractelement <8 x float> %1416, i64 4, !dbg !7
  %1527 = extractelement <8 x float> %1416, i64 5, !dbg !7
  %1528 = extractelement <8 x float> %1416, i64 6, !dbg !7
  %1529 = extractelement <8 x float> %1416, i64 7, !dbg !7
  %1530 = select i1 %.not, float %1527, float %1526, !dbg !7
  %1531 = bitcast float %1530 to i32, !dbg !7
  %1532 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1531), !dbg !7
  %1533 = bitcast i32 %1532 to float, !dbg !7
  %1534 = select i1 %.not, float %1526, float %1533, !dbg !7
  %1535 = select i1 %.not, float %1533, float %1527, !dbg !7
  %1536 = select i1 %.not, float %1529, float %1528, !dbg !7
  %1537 = bitcast float %1536 to i32, !dbg !7
  %1538 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1537), !dbg !7
  %1539 = bitcast i32 %1538 to float, !dbg !7
  %1540 = select i1 %.not, float %1528, float %1539, !dbg !7
  %1541 = select i1 %.not, float %1539, float %1529, !dbg !7
  %1542 = select i1 %.not18, float %1540, float %1534, !dbg !7
  %1543 = bitcast float %1542 to i32, !dbg !7
  %1544 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1543), !dbg !7
  %1545 = bitcast i32 %1544 to float, !dbg !7
  %1546 = select i1 %.not18, float %1534, float %1545, !dbg !7
  %1547 = select i1 %.not18, float %1545, float %1540, !dbg !7
  %1548 = select i1 %.not18, float %1541, float %1535, !dbg !7
  %1549 = bitcast float %1548 to i32, !dbg !7
  %1550 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1549), !dbg !7
  %1551 = bitcast i32 %1550 to float, !dbg !7
  %1552 = select i1 %.not18, float %1535, float %1551, !dbg !7
  %1553 = select i1 %.not18, float %1551, float %1541, !dbg !7
  %1554 = extractelement <8 x float> %1445, i64 4, !dbg !7
  %1555 = extractelement <8 x float> %1445, i64 5, !dbg !7
  %1556 = extractelement <8 x float> %1445, i64 6, !dbg !7
  %1557 = extractelement <8 x float> %1445, i64 7, !dbg !7
  %1558 = select i1 %.not, float %1555, float %1554, !dbg !7
  %1559 = bitcast float %1558 to i32, !dbg !7
  %1560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1559), !dbg !7
  %1561 = bitcast i32 %1560 to float, !dbg !7
  %1562 = select i1 %.not, float %1554, float %1561, !dbg !7
  %1563 = select i1 %.not, float %1561, float %1555, !dbg !7
  %1564 = select i1 %.not, float %1557, float %1556, !dbg !7
  %1565 = bitcast float %1564 to i32, !dbg !7
  %1566 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1141, i32 %1565), !dbg !7
  %1567 = bitcast i32 %1566 to float, !dbg !7
  %1568 = select i1 %.not, float %1556, float %1567, !dbg !7
  %1569 = select i1 %.not, float %1567, float %1557, !dbg !7
  %1570 = select i1 %.not18, float %1568, float %1562, !dbg !7
  %1571 = bitcast float %1570 to i32, !dbg !7
  %1572 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1571), !dbg !7
  %1573 = select i1 %.not18, float %1569, float %1563, !dbg !7
  %1574 = bitcast float %1573 to i32, !dbg !7
  %1575 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1574), !dbg !7
  br i1 true, label %1576, label %1640, !dbg !20

1576:                                             ; preds = %1125
  %1577 = bitcast i32 %1572 to float, !dbg !7
  %1578 = select i1 %.not18, float %1562, float %1577, !dbg !7
  %1579 = insertelement <4 x float> poison, float %1578, i64 0, !dbg !18
  %1580 = bitcast i32 %1575 to float, !dbg !7
  %1581 = select i1 %.not18, float %1563, float %1580, !dbg !7
  %1582 = insertelement <4 x float> %1579, float %1581, i64 1, !dbg !18
  %1583 = select i1 %.not18, float %1577, float %1568, !dbg !7
  %1584 = insertelement <4 x float> %1582, float %1583, i64 2, !dbg !18
  %1585 = select i1 %.not18, float %1580, float %1569, !dbg !7
  %1586 = insertelement <4 x float> %1584, float %1585, i64 3, !dbg !18
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1469, !dbg !18
  %normalized = fmul <4 x float> %1586, %row.reciprocal, !dbg !18
  %1587 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !19
  %1588 = extractelement <4 x half> %1587, i64 3, !dbg !19
  %1589 = extractelement <4 x half> %1587, i64 2, !dbg !19
  %1590 = extractelement <4 x half> %1587, i64 1, !dbg !19
  %1591 = extractelement <4 x half> %1587, i64 0, !dbg !19
  %1592 = bitcast i32 %1464 to float, !dbg !7
  %1593 = select i1 %.not18, float %1454, float %1592, !dbg !7
  %1594 = insertelement <4 x float> poison, float %1593, i64 0, !dbg !18
  %1595 = bitcast i32 %1467 to float, !dbg !7
  %1596 = select i1 %.not18, float %1455, float %1595, !dbg !7
  %1597 = insertelement <4 x float> %1594, float %1596, i64 1, !dbg !18
  %1598 = select i1 %.not18, float %1592, float %1460, !dbg !7
  %1599 = insertelement <4 x float> %1597, float %1598, i64 2, !dbg !18
  %1600 = select i1 %.not18, float %1595, float %1461, !dbg !7
  %1601 = insertelement <4 x float> %1599, float %1600, i64 3, !dbg !18
  %normalized54 = fmul <4 x float> %1601, %row.reciprocal, !dbg !18
  %1602 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !19
  %1603 = extractelement <4 x half> %1602, i64 3, !dbg !19
  %1604 = extractelement <4 x half> %1602, i64 2, !dbg !19
  %1605 = extractelement <4 x half> %1602, i64 1, !dbg !19
  %1606 = extractelement <4 x half> %1602, i64 0, !dbg !19
  %1607 = bitcast i32 %1354 to float, !dbg !7
  %1608 = select i1 %.not18, float %1344, float %1607, !dbg !7
  %1609 = insertelement <4 x float> poison, float %1608, i64 0, !dbg !18
  %1610 = bitcast i32 %1357 to float, !dbg !7
  %1611 = select i1 %.not18, float %1345, float %1610, !dbg !7
  %1612 = insertelement <4 x float> %1609, float %1611, i64 1, !dbg !18
  %1613 = select i1 %.not18, float %1607, float %1350, !dbg !7
  %1614 = insertelement <4 x float> %1612, float %1613, i64 2, !dbg !18
  %1615 = select i1 %.not18, float %1610, float %1351, !dbg !7
  %1616 = insertelement <4 x float> %1614, float %1615, i64 3, !dbg !18
  %row.reciprocal55 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1251, !dbg !18
  %normalized56 = fmul <4 x float> %1616, %row.reciprocal55, !dbg !18
  %1617 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !19
  %1618 = extractelement <4 x half> %1617, i64 3, !dbg !19
  %1619 = extractelement <4 x half> %1617, i64 2, !dbg !19
  %1620 = extractelement <4 x half> %1617, i64 1, !dbg !19
  %1621 = extractelement <4 x half> %1617, i64 0, !dbg !19
  %1622 = bitcast i32 %1246 to float, !dbg !7
  %1623 = select i1 %.not18, float %1236, float %1622, !dbg !7
  %1624 = insertelement <4 x float> poison, float %1623, i64 0, !dbg !18
  %1625 = bitcast i32 %1249 to float, !dbg !7
  %1626 = select i1 %.not18, float %1237, float %1625, !dbg !7
  %1627 = insertelement <4 x float> %1624, float %1626, i64 1, !dbg !18
  %1628 = select i1 %.not18, float %1622, float %1242, !dbg !7
  %1629 = insertelement <4 x float> %1627, float %1628, i64 2, !dbg !18
  %1630 = select i1 %.not18, float %1625, float %1243, !dbg !7
  %1631 = insertelement <4 x float> %1629, float %1630, i64 3, !dbg !18
  %normalized57 = fmul <4 x float> %1631, %row.reciprocal55, !dbg !18
  %1632 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !19
  %1633 = extractelement <4 x half> %1632, i64 3, !dbg !19
  %1634 = extractelement <4 x half> %1632, i64 2, !dbg !19
  %1635 = extractelement <4 x half> %1632, i64 1, !dbg !19
  %1636 = extractelement <4 x half> %1632, i64 0, !dbg !19
  %1637 = add nsw i64 %15, %25, !dbg !20
  %1638 = shl i64 %1637, 19, !dbg !20
  %1639 = and i64 %1638, 16252928, !dbg !20
  %address.thread474 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local475 = zext nneg i32 %address.thread474 to i64, !dbg !12
  %address.local476 = add i64 %28, %address.local475, !dbg !20
  %address.local477 = lshr i64 %address.local475, 6, !dbg !12
  %address.local478 = shl nuw nsw i64 %address.local477, 4, !dbg !20
  %address.local479 = add i64 %address.local476, %address.local478, !dbg !20
  %address.local480 = and i64 %address.local475, 240, !dbg !20
  %address.local481 = sub i64 %address.local479, %address.local480, !dbg !20
  %address.local482 = add i64 %address.local481, %29, !dbg !20
  %address.local483 = shl nuw nsw i64 %address.local482, 7, !dbg !20
  %address.local484 = add nuw nsw i64 %1639, %address.local483, !dbg !20
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !20
  %address.local486 = lshr i64 %address.local475, 2, !dbg !12
  %address.local487 = and i64 %address.local486, 12, !dbg !12
  %address.local488 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local485, i64 %address.local487, !dbg !20
  store half %1636, ptr addrspace(1) %address.local488, align 2, !dbg !20
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !20
  %address.local490 = or disjoint i64 %address.local487, 1, !dbg !20
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local490, !dbg !20
  store half %1635, ptr addrspace(1) %address.local491, align 2, !dbg !20
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !20
  %address.local493 = or disjoint i64 %address.local487, 2, !dbg !20
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local492, i64 %address.local493, !dbg !20
  store half %1634, ptr addrspace(1) %address.local494, align 2, !dbg !20
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !20
  %address.local496 = or disjoint i64 %address.local487, 3, !dbg !20
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local495, i64 %address.local496, !dbg !20
  store half %1633, ptr addrspace(1) %address.local497, align 2, !dbg !20
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !20
  %address.local499 = or disjoint i64 %address.local487, 16, !dbg !20
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local498, i64 %address.local499, !dbg !20
  store half %1621, ptr addrspace(1) %address.local500, align 2, !dbg !20
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !20
  %address.local502 = or disjoint i64 %address.local487, 17, !dbg !20
  %address.local503 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local501, i64 %address.local502, !dbg !20
  store half %1620, ptr addrspace(1) %address.local503, align 2, !dbg !20
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !20
  %address.local505 = or disjoint i64 %address.local487, 18, !dbg !20
  %address.local506 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local504, i64 %address.local505, !dbg !20
  store half %1619, ptr addrspace(1) %address.local506, align 2, !dbg !20
  %address.local507 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local484, !dbg !20
  %address.local508 = or disjoint i64 %address.local487, 19, !dbg !20
  %address.local509 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local507, i64 %address.local508, !dbg !20
  store half %1618, ptr addrspace(1) %address.local509, align 2, !dbg !20
  %address.local510 = shl i64 %address.local482, 7, !dbg !20
  %address.local511 = add i64 %address.local510, 8192, !dbg !20
  %address.local512 = add nuw nsw i64 %1639, %address.local511, !dbg !20
  %address.local513 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local514 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local513, i64 %address.local487, !dbg !20
  store half %1606, ptr addrspace(1) %address.local514, align 2, !dbg !20
  %address.local515 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local515, i64 %address.local490, !dbg !20
  store half %1605, ptr addrspace(1) %address.local516, align 2, !dbg !20
  %address.local517 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local517, i64 %address.local493, !dbg !20
  store half %1604, ptr addrspace(1) %address.local518, align 2, !dbg !20
  %address.local519 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local520 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local519, i64 %address.local496, !dbg !20
  store half %1603, ptr addrspace(1) %address.local520, align 2, !dbg !20
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local522 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local521, i64 %address.local499, !dbg !20
  store half %1591, ptr addrspace(1) %address.local522, align 2, !dbg !20
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local523, i64 %address.local502, !dbg !20
  store half %1590, ptr addrspace(1) %address.local524, align 2, !dbg !20
  %address.local525 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local525, i64 %address.local505, !dbg !20
  store half %1589, ptr addrspace(1) %address.local526, align 2, !dbg !20
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local528 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local527, i64 %address.local508, !dbg !20
  store half %1588, ptr addrspace(1) %address.local528, align 2, !dbg !20
  br label %1640, !dbg !20

1640:                                             ; preds = %1576, %1125
  br i1 true, label %1641, label %1679, !dbg !20

1641:                                             ; preds = %1640
  %1642 = insertelement <4 x float> poison, float %1546, i64 0, !dbg !18
  %1643 = insertelement <4 x float> %1642, float %1552, i64 1, !dbg !18
  %1644 = insertelement <4 x float> %1643, float %1547, i64 2, !dbg !18
  %1645 = insertelement <4 x float> %1644, float %1553, i64 3, !dbg !18
  %row.reciprocal58 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1469, !dbg !18
  %normalized59 = fmul <4 x float> %1645, %row.reciprocal58, !dbg !18
  %1646 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !19
  %1647 = extractelement <4 x half> %1646, i64 3, !dbg !19
  %1648 = extractelement <4 x half> %1646, i64 2, !dbg !19
  %1649 = extractelement <4 x half> %1646, i64 1, !dbg !19
  %1650 = extractelement <4 x half> %1646, i64 0, !dbg !19
  %1651 = insertelement <4 x float> poison, float %1437, i64 0, !dbg !18
  %1652 = insertelement <4 x float> %1651, float %1443, i64 1, !dbg !18
  %1653 = insertelement <4 x float> %1652, float %1438, i64 2, !dbg !18
  %1654 = insertelement <4 x float> %1653, float %1444, i64 3, !dbg !18
  %normalized60 = fmul <4 x float> %1654, %row.reciprocal58, !dbg !18
  %1655 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !19
  %1656 = extractelement <4 x half> %1655, i64 3, !dbg !19
  %1657 = extractelement <4 x half> %1655, i64 2, !dbg !19
  %1658 = extractelement <4 x half> %1655, i64 1, !dbg !19
  %1659 = extractelement <4 x half> %1655, i64 0, !dbg !19
  %1660 = insertelement <4 x float> poison, float %1328, i64 0, !dbg !18
  %1661 = insertelement <4 x float> %1660, float %1334, i64 1, !dbg !18
  %1662 = insertelement <4 x float> %1661, float %1329, i64 2, !dbg !18
  %1663 = insertelement <4 x float> %1662, float %1335, i64 3, !dbg !18
  %row.reciprocal61 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1251, !dbg !18
  %normalized62 = fmul <4 x float> %1663, %row.reciprocal61, !dbg !18
  %1664 = fptrunc <4 x float> %normalized62 to <4 x half>, !dbg !19
  %1665 = extractelement <4 x half> %1664, i64 3, !dbg !19
  %1666 = extractelement <4 x half> %1664, i64 2, !dbg !19
  %1667 = extractelement <4 x half> %1664, i64 1, !dbg !19
  %1668 = extractelement <4 x half> %1664, i64 0, !dbg !19
  %1669 = insertelement <4 x float> poison, float %1219, i64 0, !dbg !18
  %1670 = insertelement <4 x float> %1669, float %1225, i64 1, !dbg !18
  %1671 = insertelement <4 x float> %1670, float %1220, i64 2, !dbg !18
  %1672 = insertelement <4 x float> %1671, float %1226, i64 3, !dbg !18
  %normalized63 = fmul <4 x float> %1672, %row.reciprocal61, !dbg !18
  %1673 = fptrunc <4 x float> %normalized63 to <4 x half>, !dbg !19
  %1674 = extractelement <4 x half> %1673, i64 3, !dbg !19
  %1675 = extractelement <4 x half> %1673, i64 2, !dbg !19
  %1676 = extractelement <4 x half> %1673, i64 1, !dbg !19
  %1677 = extractelement <4 x half> %1673, i64 0, !dbg !19
  %1678 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread529 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local530 = zext nneg i32 %address.thread529 to i64, !dbg !12
  %address.local531 = add i64 %28, %address.local530, !dbg !20
  %address.local532 = lshr i64 %address.local530, 6, !dbg !12
  %address.local533 = shl nuw nsw i64 %address.local532, 4, !dbg !20
  %address.local534 = add i64 %address.local531, %address.local533, !dbg !20
  %address.local535 = and i64 %address.local530, 240, !dbg !20
  %address.local536 = sub i64 %address.local534, %address.local535, !dbg !20
  %address.local537 = add i64 %address.local536, %29, !dbg !20
  %address.local538 = shl nuw nsw i64 %address.local537, 7, !dbg !20
  %address.local539 = add nuw nsw i64 %1678, %address.local538, !dbg !20
  %address.local540 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local541 = lshr i64 %address.local530, 2, !dbg !12
  %address.local542 = and i64 %address.local541, 12, !dbg !12
  %address.local543 = or disjoint i64 %address.local542, 32, !dbg !20
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local540, i64 %address.local543, !dbg !20
  store half %1677, ptr addrspace(1) %address.local544, align 2, !dbg !20
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local546 = or disjoint i64 %address.local542, 33, !dbg !20
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local546, !dbg !20
  store half %1676, ptr addrspace(1) %address.local547, align 2, !dbg !20
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local549 = or disjoint i64 %address.local542, 34, !dbg !20
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local548, i64 %address.local549, !dbg !20
  store half %1675, ptr addrspace(1) %address.local550, align 2, !dbg !20
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local552 = or disjoint i64 %address.local542, 35, !dbg !20
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local551, i64 %address.local552, !dbg !20
  store half %1674, ptr addrspace(1) %address.local553, align 2, !dbg !20
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local555 = or disjoint i64 %address.local542, 48, !dbg !20
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local554, i64 %address.local555, !dbg !20
  store half %1668, ptr addrspace(1) %address.local556, align 2, !dbg !20
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local558 = or disjoint i64 %address.local542, 49, !dbg !20
  %address.local559 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local557, i64 %address.local558, !dbg !20
  store half %1667, ptr addrspace(1) %address.local559, align 2, !dbg !20
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local561 = or disjoint i64 %address.local542, 50, !dbg !20
  %address.local562 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local560, i64 %address.local561, !dbg !20
  store half %1666, ptr addrspace(1) %address.local562, align 2, !dbg !20
  %address.local563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !20
  %address.local564 = or i64 %address.local541, 51, !dbg !20
  %address.local565 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local563, i64 %address.local564, !dbg !20
  store half %1665, ptr addrspace(1) %address.local565, align 2, !dbg !20
  %address.local566 = shl i64 %address.local537, 7, !dbg !20
  %address.local567 = add i64 %address.local566, 8192, !dbg !20
  %address.local568 = add nuw nsw i64 %1678, %address.local567, !dbg !20
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local570 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local569, i64 %address.local543, !dbg !20
  store half %1659, ptr addrspace(1) %address.local570, align 2, !dbg !20
  %address.local571 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local571, i64 %address.local546, !dbg !20
  store half %1658, ptr addrspace(1) %address.local572, align 2, !dbg !20
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local573, i64 %address.local549, !dbg !20
  store half %1657, ptr addrspace(1) %address.local574, align 2, !dbg !20
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local552, !dbg !20
  store half %1656, ptr addrspace(1) %address.local576, align 2, !dbg !20
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local555, !dbg !20
  store half %1650, ptr addrspace(1) %address.local578, align 2, !dbg !20
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local579, i64 %address.local558, !dbg !20
  store half %1649, ptr addrspace(1) %address.local580, align 2, !dbg !20
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local561, !dbg !20
  store half %1648, ptr addrspace(1) %address.local582, align 2, !dbg !20
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local564, !dbg !20
  store half %1647, ptr addrspace(1) %address.local584, align 2, !dbg !20
  br label %1679, !dbg !20

1679:                                             ; preds = %1641, %1640
  br i1 true, label %1680, label %1718, !dbg !20

1680:                                             ; preds = %1679
  %1681 = insertelement <4 x float> poison, float %1518, i64 0, !dbg !18
  %1682 = insertelement <4 x float> %1681, float %1524, i64 1, !dbg !18
  %1683 = insertelement <4 x float> %1682, float %1519, i64 2, !dbg !18
  %1684 = insertelement <4 x float> %1683, float %1525, i64 3, !dbg !18
  %row.reciprocal64 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1469, !dbg !18
  %normalized65 = fmul <4 x float> %1684, %row.reciprocal64, !dbg !18
  %1685 = fptrunc <4 x float> %normalized65 to <4 x half>, !dbg !19
  %1686 = extractelement <4 x half> %1685, i64 3, !dbg !19
  %1687 = extractelement <4 x half> %1685, i64 2, !dbg !19
  %1688 = extractelement <4 x half> %1685, i64 1, !dbg !19
  %1689 = extractelement <4 x half> %1685, i64 0, !dbg !19
  %1690 = insertelement <4 x float> poison, float %1408, i64 0, !dbg !18
  %1691 = insertelement <4 x float> %1690, float %1414, i64 1, !dbg !18
  %1692 = insertelement <4 x float> %1691, float %1409, i64 2, !dbg !18
  %1693 = insertelement <4 x float> %1692, float %1415, i64 3, !dbg !18
  %normalized66 = fmul <4 x float> %1693, %row.reciprocal64, !dbg !18
  %1694 = fptrunc <4 x float> %normalized66 to <4 x half>, !dbg !19
  %1695 = extractelement <4 x half> %1694, i64 3, !dbg !19
  %1696 = extractelement <4 x half> %1694, i64 2, !dbg !19
  %1697 = extractelement <4 x half> %1694, i64 1, !dbg !19
  %1698 = extractelement <4 x half> %1694, i64 0, !dbg !19
  %1699 = insertelement <4 x float> poison, float %1300, i64 0, !dbg !18
  %1700 = insertelement <4 x float> %1699, float %1306, i64 1, !dbg !18
  %1701 = insertelement <4 x float> %1700, float %1301, i64 2, !dbg !18
  %1702 = insertelement <4 x float> %1701, float %1307, i64 3, !dbg !18
  %row.reciprocal67 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1251, !dbg !18
  %normalized68 = fmul <4 x float> %1702, %row.reciprocal67, !dbg !18
  %1703 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !19
  %1704 = extractelement <4 x half> %1703, i64 3, !dbg !19
  %1705 = extractelement <4 x half> %1703, i64 2, !dbg !19
  %1706 = extractelement <4 x half> %1703, i64 1, !dbg !19
  %1707 = extractelement <4 x half> %1703, i64 0, !dbg !19
  %1708 = insertelement <4 x float> poison, float %1190, i64 0, !dbg !18
  %1709 = insertelement <4 x float> %1708, float %1196, i64 1, !dbg !18
  %1710 = insertelement <4 x float> %1709, float %1191, i64 2, !dbg !18
  %1711 = insertelement <4 x float> %1710, float %1197, i64 3, !dbg !18
  %normalized69 = fmul <4 x float> %1711, %row.reciprocal67, !dbg !18
  %1712 = fptrunc <4 x float> %normalized69 to <4 x half>, !dbg !19
  %1713 = extractelement <4 x half> %1712, i64 3, !dbg !19
  %1714 = extractelement <4 x half> %1712, i64 2, !dbg !19
  %1715 = extractelement <4 x half> %1712, i64 1, !dbg !19
  %1716 = extractelement <4 x half> %1712, i64 0, !dbg !19
  %1717 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread585 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local586 = zext nneg i32 %address.thread585 to i64, !dbg !12
  %address.local587 = add i64 %28, %address.local586, !dbg !20
  %address.local588 = lshr i64 %address.local586, 6, !dbg !12
  %address.local589 = shl nuw nsw i64 %address.local588, 4, !dbg !20
  %address.local590 = add i64 %address.local587, %address.local589, !dbg !20
  %address.local591 = and i64 %address.local586, 240, !dbg !20
  %address.local592 = sub i64 %address.local590, %address.local591, !dbg !20
  %address.local593 = add i64 %address.local592, %29, !dbg !20
  %address.local594 = shl nuw nsw i64 %address.local593, 7, !dbg !20
  %address.local595 = add nuw nsw i64 %1717, %address.local594, !dbg !20
  %address.local596 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local597 = lshr i64 %address.local586, 2, !dbg !12
  %address.local598 = and i64 %address.local597, 12, !dbg !12
  %address.local599 = or disjoint i64 %address.local598, 64, !dbg !20
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local596, i64 %address.local599, !dbg !20
  store half %1716, ptr addrspace(1) %address.local600, align 2, !dbg !20
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local602 = or disjoint i64 %address.local598, 65, !dbg !20
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local601, i64 %address.local602, !dbg !20
  store half %1715, ptr addrspace(1) %address.local603, align 2, !dbg !20
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local605 = or disjoint i64 %address.local598, 66, !dbg !20
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local604, i64 %address.local605, !dbg !20
  store half %1714, ptr addrspace(1) %address.local606, align 2, !dbg !20
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local608 = or disjoint i64 %address.local598, 67, !dbg !20
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local607, i64 %address.local608, !dbg !20
  store half %1713, ptr addrspace(1) %address.local609, align 2, !dbg !20
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local611 = or disjoint i64 %address.local598, 80, !dbg !20
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local610, i64 %address.local611, !dbg !20
  store half %1707, ptr addrspace(1) %address.local612, align 2, !dbg !20
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local614 = or disjoint i64 %address.local598, 81, !dbg !20
  %address.local615 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local613, i64 %address.local614, !dbg !20
  store half %1706, ptr addrspace(1) %address.local615, align 2, !dbg !20
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local617 = or disjoint i64 %address.local598, 82, !dbg !20
  %address.local618 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local616, i64 %address.local617, !dbg !20
  store half %1705, ptr addrspace(1) %address.local618, align 2, !dbg !20
  %address.local619 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !20
  %address.local620 = or disjoint i64 %address.local598, 83, !dbg !20
  %address.local621 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local619, i64 %address.local620, !dbg !20
  store half %1704, ptr addrspace(1) %address.local621, align 2, !dbg !20
  %address.local622 = shl i64 %address.local593, 7, !dbg !20
  %address.local623 = add i64 %address.local622, 8192, !dbg !20
  %address.local624 = add nuw nsw i64 %1717, %address.local623, !dbg !20
  %address.local625 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !20
  %address.local626 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local625, i64 %address.local599, !dbg !20
  store half %1698, ptr addrspace(1) %address.local626, align 2, !dbg !20
  %address.local627 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !20
  %address.local628 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local627, i64 %address.local602, !dbg !20
  store half %1697, ptr addrspace(1) %address.local628, align 2, !dbg !20
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !20
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local629, i64 %address.local605, !dbg !20
  store half %1696, ptr addrspace(1) %address.local630, align 2, !dbg !20
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !20
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local608, !dbg !20
  store half %1695, ptr addrspace(1) %address.local632, align 2, !dbg !20
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !20
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local633, i64 %address.local611, !dbg !20
  store half %1689, ptr addrspace(1) %address.local634, align 2, !dbg !20
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !20
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local635, i64 %address.local614, !dbg !20
  store half %1688, ptr addrspace(1) %address.local636, align 2, !dbg !20
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !20
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local637, i64 %address.local617, !dbg !20
  store half %1687, ptr addrspace(1) %address.local638, align 2, !dbg !20
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local624, !dbg !20
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local639, i64 %address.local620, !dbg !20
  store half %1686, ptr addrspace(1) %address.local640, align 2, !dbg !20
  br label %1718, !dbg !20

1718:                                             ; preds = %1680, %1679
  br i1 true, label %1719, label %1757, !dbg !20

1719:                                             ; preds = %1718
  %1720 = insertelement <4 x float> poison, float %1490, i64 0, !dbg !18
  %1721 = insertelement <4 x float> %1720, float %1496, i64 1, !dbg !18
  %1722 = insertelement <4 x float> %1721, float %1491, i64 2, !dbg !18
  %1723 = insertelement <4 x float> %1722, float %1497, i64 3, !dbg !18
  %row.reciprocal70 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1469, !dbg !18
  %normalized71 = fmul <4 x float> %1723, %row.reciprocal70, !dbg !18
  %1724 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !19
  %1725 = extractelement <4 x half> %1724, i64 3, !dbg !19
  %1726 = extractelement <4 x half> %1724, i64 2, !dbg !19
  %1727 = extractelement <4 x half> %1724, i64 1, !dbg !19
  %1728 = extractelement <4 x half> %1724, i64 0, !dbg !19
  %1729 = insertelement <4 x float> poison, float %1379, i64 0, !dbg !18
  %1730 = insertelement <4 x float> %1729, float %1385, i64 1, !dbg !18
  %1731 = insertelement <4 x float> %1730, float %1380, i64 2, !dbg !18
  %1732 = insertelement <4 x float> %1731, float %1386, i64 3, !dbg !18
  %normalized72 = fmul <4 x float> %1732, %row.reciprocal70, !dbg !18
  %1733 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !19
  %1734 = extractelement <4 x half> %1733, i64 3, !dbg !19
  %1735 = extractelement <4 x half> %1733, i64 2, !dbg !19
  %1736 = extractelement <4 x half> %1733, i64 1, !dbg !19
  %1737 = extractelement <4 x half> %1733, i64 0, !dbg !19
  %1738 = insertelement <4 x float> poison, float %1272, i64 0, !dbg !18
  %1739 = insertelement <4 x float> %1738, float %1278, i64 1, !dbg !18
  %1740 = insertelement <4 x float> %1739, float %1273, i64 2, !dbg !18
  %1741 = insertelement <4 x float> %1740, float %1279, i64 3, !dbg !18
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1251, !dbg !18
  %normalized74 = fmul <4 x float> %1741, %row.reciprocal73, !dbg !18
  %1742 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !19
  %1743 = extractelement <4 x half> %1742, i64 3, !dbg !19
  %1744 = extractelement <4 x half> %1742, i64 2, !dbg !19
  %1745 = extractelement <4 x half> %1742, i64 1, !dbg !19
  %1746 = extractelement <4 x half> %1742, i64 0, !dbg !19
  %1747 = insertelement <4 x float> poison, float %1161, i64 0, !dbg !18
  %1748 = insertelement <4 x float> %1747, float %1167, i64 1, !dbg !18
  %1749 = insertelement <4 x float> %1748, float %1162, i64 2, !dbg !18
  %1750 = insertelement <4 x float> %1749, float %1168, i64 3, !dbg !18
  %normalized75 = fmul <4 x float> %1750, %row.reciprocal73, !dbg !18
  %1751 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !19
  %1752 = extractelement <4 x half> %1751, i64 3, !dbg !19
  %1753 = extractelement <4 x half> %1751, i64 2, !dbg !19
  %1754 = extractelement <4 x half> %1751, i64 1, !dbg !19
  %1755 = extractelement <4 x half> %1751, i64 0, !dbg !19
  %1756 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread641 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local642 = zext nneg i32 %address.thread641 to i64, !dbg !12
  %address.local643 = add i64 %28, %address.local642, !dbg !20
  %address.local644 = lshr i64 %address.local642, 6, !dbg !12
  %address.local645 = shl nuw nsw i64 %address.local644, 4, !dbg !20
  %address.local646 = add i64 %address.local643, %address.local645, !dbg !20
  %address.local647 = and i64 %address.local642, 240, !dbg !20
  %address.local648 = sub i64 %address.local646, %address.local647, !dbg !20
  %address.local649 = add i64 %address.local648, %29, !dbg !20
  %address.local650 = shl nuw nsw i64 %address.local649, 7, !dbg !20
  %address.local651 = add nuw nsw i64 %1756, %address.local650, !dbg !20
  %address.local652 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !20
  %address.local653 = lshr i64 %address.local642, 2, !dbg !12
  %address.local654 = and i64 %address.local653, 12, !dbg !12
  %address.local655 = or disjoint i64 %address.local654, 96, !dbg !20
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local652, i64 %address.local655, !dbg !20
  store half %1755, ptr addrspace(1) %address.local656, align 2, !dbg !20
  %address.local657 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !20
  %address.local658 = or disjoint i64 %address.local654, 97, !dbg !20
  %address.local659 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local657, i64 %address.local658, !dbg !20
  store half %1754, ptr addrspace(1) %address.local659, align 2, !dbg !20
  %address.local660 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !20
  %address.local661 = or disjoint i64 %address.local654, 98, !dbg !20
  %address.local662 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local660, i64 %address.local661, !dbg !20
  store half %1753, ptr addrspace(1) %address.local662, align 2, !dbg !20
  %address.local663 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !20
  %address.local664 = or disjoint i64 %address.local654, 99, !dbg !20
  %address.local665 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local663, i64 %address.local664, !dbg !20
  store half %1752, ptr addrspace(1) %address.local665, align 2, !dbg !20
  %address.local666 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !20
  %address.local667 = or disjoint i64 %address.local654, 112, !dbg !20
  %address.local668 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local666, i64 %address.local667, !dbg !20
  store half %1746, ptr addrspace(1) %address.local668, align 2, !dbg !20
  %address.local669 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !20
  %address.local670 = or disjoint i64 %address.local654, 113, !dbg !20
  %address.local671 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local669, i64 %address.local670, !dbg !20
  store half %1745, ptr addrspace(1) %address.local671, align 2, !dbg !20
  %address.local672 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !20
  %address.local673 = or disjoint i64 %address.local654, 114, !dbg !20
  %address.local674 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local672, i64 %address.local673, !dbg !20
  store half %1744, ptr addrspace(1) %address.local674, align 2, !dbg !20
  %address.local675 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local651, !dbg !20
  %address.local676 = or i64 %address.local653, 115, !dbg !20
  %address.local677 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local675, i64 %address.local676, !dbg !20
  store half %1743, ptr addrspace(1) %address.local677, align 2, !dbg !20
  %address.local678 = shl i64 %address.local649, 7, !dbg !20
  %address.local679 = add i64 %address.local678, 8192, !dbg !20
  %address.local680 = add nuw nsw i64 %1756, %address.local679, !dbg !20
  %address.local681 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !20
  %address.local682 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local681, i64 %address.local655, !dbg !20
  store half %1737, ptr addrspace(1) %address.local682, align 2, !dbg !20
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !20
  %address.local684 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local683, i64 %address.local658, !dbg !20
  store half %1736, ptr addrspace(1) %address.local684, align 2, !dbg !20
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !20
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local685, i64 %address.local661, !dbg !20
  store half %1735, ptr addrspace(1) %address.local686, align 2, !dbg !20
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !20
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local687, i64 %address.local664, !dbg !20
  store half %1734, ptr addrspace(1) %address.local688, align 2, !dbg !20
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !20
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local689, i64 %address.local667, !dbg !20
  store half %1728, ptr addrspace(1) %address.local690, align 2, !dbg !20
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !20
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local691, i64 %address.local670, !dbg !20
  store half %1727, ptr addrspace(1) %address.local692, align 2, !dbg !20
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !20
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local693, i64 %address.local673, !dbg !20
  store half %1726, ptr addrspace(1) %address.local694, align 2, !dbg !20
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local680, !dbg !20
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local695, i64 %address.local676, !dbg !20
  store half %1725, ptr addrspace(1) %address.local696, align 2, !dbg !20
  br label %1757, !dbg !20

1757:                                             ; preds = %1719, %1718
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
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

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
!llvm.module.flags = !{!37}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/online/session-20261026-101601-tehle7yk/tuning/run-20261026-101601-inpooz4_/input-0001-test_input_1x32x4096x128-128-64-32-online/grid-000007")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!9 = !DILocation(line: 48, column: 17, scope: !4)
!11 = !DILocation(line: 70, column: 22, scope: !4)
!12 = !DILocation(line: 72, column: 24, scope: !4)
!13 = !DILocation(line: 77, column: 19, scope: !4)
!14 = !DILocation(line: 102, column: 23, scope: !4)
!15 = !DILocation(line: 71, column: 22, scope: !4)
!16 = !DILocation(line: 73, column: 26, scope: !4)
!17 = !DILocation(line: 101, column: 20, scope: !4)
!18 = !DILocation(line: 107, column: 23, scope: !4)
!19 = !DILocation(line: 108, column: 19, scope: !4)
!20 = !DILocation(line: 109, column: 9, scope: !4)
!22 = !DILocation(line: 79, column: 24, scope: !4)
!23 = !DILocation(line: 80, column: 23, scope: !4)
!24 = !DILocation(line: 81, column: 26, scope: !4)
!25 = !DILocation(line: 88, column: 21, scope: !4)
!26 = !DILocation(line: 89, column: 20, scope: !4)
!27 = !DILocation(line: 90, column: 22, scope: !4)
!28 = !DILocation(line: 91, column: 23, scope: !4)
!29 = !DILocation(line: 92, column: 20, scope: !4)
!30 = !DILocation(line: 93, column: 22, scope: !4)
!31 = !DILocation(line: 94, column: 22, scope: !4)
!32 = !DILocation(line: 95, column: 20, scope: !4)
!33 = !DILocation(line: 96, column: 16, scope: !4)
!34 = !DILocation(line: 103, column: 22, scope: !4)
!36 = !{i32 256, i32 1, i32 1}
!37 = !{i32 2, !"Debug Info Version", i32 3}
!38 = !{i32 0, i32 256}
