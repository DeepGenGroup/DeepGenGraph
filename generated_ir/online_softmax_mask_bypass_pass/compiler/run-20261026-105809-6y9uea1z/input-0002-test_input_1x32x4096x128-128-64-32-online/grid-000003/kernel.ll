; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/online_softmax_mask_bypass_pass/compiler/run-20261026-105809-6y9uea1z/input-0002-test_input_1x32x4096x128-128-64-32-online/grid-000003/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_mask_bypass_pass/compiler/run-20261026-105809-6y9uea1z/input-0002-test_input_1x32x4096x128-128-64-32-online/grid-000003/input.mlir"

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
  %address.local193 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local194 = and i64 %address.local193, 1536, !dbg !9
  %address.local195 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local194, !dbg !9
  %address.local196 = ptrtoint ptr addrspace(3) %address.local195 to i64, !dbg !9
  %address.local197 = trunc i64 %address.local196 to i32, !dbg !9
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local197), !dbg !9
  %35 = zext i32 %34 to i64, !dbg !9
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !9
  %address.local198 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local199 = and i64 %address.local198, 16256, !dbg !9
  %address.local200 = lshr i64 %address.local, 4, !dbg !9
  %address.local201 = and i64 %address.local200, 8, !dbg !9
  %address.local202 = or disjoint i64 %address.local199, %address.local201, !dbg !9
  %address.local203 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local202, !dbg !9
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %38 = ptrtoint ptr addrspace(1) %address.local203 to i64, !dbg !9
  %39 = sub i64 %38, %37, !dbg !9
  %dma.buffer.offset1013 = trunc i64 %39 to i32, !dbg !9
  %40 = trunc i64 %37 to i32, !dbg !9
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !9
  %42 = lshr i64 %37, 32, !dbg !9
  %43 = trunc i64 %42 to i32, !dbg !9
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !9
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1014 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !9
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !9
  %dma.buffer.lds1015 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !9
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1015, i32 %dma.buffer.offset1013, <4 x i32> %dma.buffer.rsrc1014) #10, !dbg !9
  %address.local204 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local205 = shl nuw nsw i64 %address.local204, 3, !dbg !9
  %address.local206 = and i64 %address.local205, 3584, !dbg !9
  %address.local207 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local206, !dbg !9
  %address.local208 = ptrtoint ptr addrspace(3) %address.local207 to i64, !dbg !9
  %address.local209 = trunc i64 %address.local208 to i32, !dbg !9
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local209), !dbg !9
  %49 = zext i32 %48 to i64, !dbg !9
  %50 = inttoptr i64 %49 to ptr addrspace(3), !dbg !9
  %address.local210 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local211 = and i64 %address.local210, 16256, !dbg !9
  %address.local212 = lshr i64 %address.local204, 4, !dbg !9
  %address.local213 = and i64 %address.local212, 24, !dbg !9
  %address.local214 = or disjoint i64 %address.local211, %address.local213, !dbg !9
  %address.local215 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local214, !dbg !9
  %51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %52 = ptrtoint ptr addrspace(1) %address.local215 to i64, !dbg !9
  %53 = sub i64 %52, %51, !dbg !9
  %dma.buffer.offset1010 = trunc i64 %53 to i32, !dbg !9
  %54 = trunc i64 %51 to i32, !dbg !9
  %55 = insertelement <4 x i32> poison, i32 %54, i64 0, !dbg !9
  %56 = lshr i64 %51, 32, !dbg !9
  %57 = trunc i64 %56 to i32, !dbg !9
  %58 = insertelement <4 x i32> %55, i32 %57, i64 1, !dbg !9
  %59 = insertelement <4 x i32> %58, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1011 = insertelement <4 x i32> %59, i32 131072, i64 3, !dbg !9
  %60 = ptrtoint ptr addrspace(3) %50 to i32, !dbg !9
  %dma.buffer.lds1012 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !9
  %61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1012, i32 %dma.buffer.offset1010, <4 x i32> %dma.buffer.rsrc1011) #10, !dbg !9
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
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local194, !dbg !9
  %address.local217 = ptrtoint ptr addrspace(3) %address.local216 to i64, !dbg !9
  %address.local218 = trunc i64 %address.local217 to i32, !dbg !9
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local218), !dbg !9
  %69 = zext i32 %68 to i64, !dbg !9
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !9
  %address.local219 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local202, !dbg !9
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %72 = ptrtoint ptr addrspace(1) %address.local219 to i64, !dbg !9
  %73 = sub i64 %72, %71, !dbg !9
  %dma.buffer.offset1007 = trunc i64 %73 to i32, !dbg !9
  %74 = trunc i64 %71 to i32, !dbg !9
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !9
  %76 = lshr i64 %71, 32, !dbg !9
  %77 = trunc i64 %76 to i32, !dbg !9
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !9
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1008 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !9
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !9
  %dma.buffer.lds1009 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !9
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1009, i32 %dma.buffer.offset1007, <4 x i32> %dma.buffer.rsrc1008) #10, !dbg !9
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !9
  %address.local221 = ptrtoint ptr addrspace(3) %address.local220 to i64, !dbg !9
  %address.local222 = trunc i64 %address.local221 to i32, !dbg !9
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local222), !dbg !9
  %83 = zext i32 %82 to i64, !dbg !9
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !9
  %address.local223 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local214, !dbg !9
  %85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %86 = ptrtoint ptr addrspace(1) %address.local223 to i64, !dbg !9
  %87 = sub i64 %86, %85, !dbg !9
  %dma.buffer.offset1004 = trunc i64 %87 to i32, !dbg !9
  %88 = trunc i64 %85 to i32, !dbg !9
  %89 = insertelement <4 x i32> poison, i32 %88, i64 0, !dbg !9
  %90 = lshr i64 %85, 32, !dbg !9
  %91 = trunc i64 %90 to i32, !dbg !9
  %92 = insertelement <4 x i32> %89, i32 %91, i64 1, !dbg !9
  %93 = insertelement <4 x i32> %92, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1005 = insertelement <4 x i32> %93, i32 131072, i64 3, !dbg !9
  %94 = ptrtoint ptr addrspace(3) %84 to i32, !dbg !9
  %dma.buffer.lds1006 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !9
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1006, i32 %dma.buffer.offset1004, <4 x i32> %dma.buffer.rsrc1005) #10, !dbg !9
  %address.local224 = lshr i32 %address.thread, 2, !dbg !8
  %address.local225 = and i32 %address.local224, 4, !dbg !8
  %address.local226 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local227 = and i32 %address.local226, 384, !dbg !8
  %address.local228 = or disjoint i32 %address.local225, %address.local227, !dbg !8
  %address.local229 = shl nuw nsw i32 %address.thread, 3, !dbg !8
  %address.local230 = and i32 %address.local229, 120, !dbg !8
  %address.local231 = or disjoint i32 %address.local228, %address.local230, !dbg !8
  %address.local232 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local233 = and i32 %address.local232, 1024, !dbg !8
  %address.local234 = or disjoint i32 %address.local231, %address.local233, !dbg !8
  %address.local235 = zext nneg i32 %address.local234 to i64, !dbg !8
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local235, !dbg !8
  %96 = load <4 x half>, ptr addrspace(3) %address.local236, align 8, !dbg !8
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local236, i64 4096, !dbg !8
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !8
  %98 = fmul <4 x half> %96, splat (half 0xH3015), !dbg !8
  %lds.const.gep902 = getelementptr i8, ptr addrspace(3) %address.local236, i64 1024, !dbg !8
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep902, align 8, !dbg !8
  %100 = fmul <4 x half> %97, splat (half 0xH3015), !dbg !8
  %lds.const.gep903 = getelementptr i8, ptr addrspace(3) %address.local236, i64 5120, !dbg !8
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep903, align 8, !dbg !8
  %102 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !8
  %103 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !9
  %105 = getelementptr i8, ptr addrspace(1) %104, i64 128, !dbg !9
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local197), !dbg !9
  %107 = zext i32 %106 to i64, !dbg !9
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !9
  %address.local279 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local202, !dbg !9
  %109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %110 = ptrtoint ptr addrspace(1) %address.local279 to i64, !dbg !9
  %111 = sub i64 %110, %109, !dbg !9
  %dma.buffer.offset1001 = trunc i64 %111 to i32, !dbg !9
  %112 = trunc i64 %109 to i32, !dbg !9
  %113 = insertelement <4 x i32> poison, i32 %112, i64 0, !dbg !9
  %114 = lshr i64 %109, 32, !dbg !9
  %115 = trunc i64 %114 to i32, !dbg !9
  %116 = insertelement <4 x i32> %113, i32 %115, i64 1, !dbg !9
  %117 = insertelement <4 x i32> %116, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc1002 = insertelement <4 x i32> %117, i32 131072, i64 3, !dbg !9
  %118 = ptrtoint ptr addrspace(3) %108 to i32, !dbg !9
  %dma.buffer.lds1003 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !9
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1003, i32 %dma.buffer.offset1001, <4 x i32> %dma.buffer.rsrc1002) #10, !dbg !9
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local209), !dbg !9
  %121 = zext i32 %120 to i64, !dbg !9
  %122 = inttoptr i64 %121 to ptr addrspace(3), !dbg !9
  %address.local280 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local214, !dbg !9
  %123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %124 = ptrtoint ptr addrspace(1) %address.local280 to i64, !dbg !9
  %125 = sub i64 %124, %123, !dbg !9
  %dma.buffer.offset998 = trunc i64 %125 to i32, !dbg !9
  %126 = trunc i64 %123 to i32, !dbg !9
  %127 = insertelement <4 x i32> poison, i32 %126, i64 0, !dbg !9
  %128 = lshr i64 %123, 32, !dbg !9
  %129 = trunc i64 %128 to i32, !dbg !9
  %130 = insertelement <4 x i32> %127, i32 %129, i64 1, !dbg !9
  %131 = insertelement <4 x i32> %130, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc999 = insertelement <4 x i32> %131, i32 131072, i64 3, !dbg !9
  %132 = ptrtoint ptr addrspace(3) %122 to i32, !dbg !9
  %dma.buffer.lds1000 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !9
  %133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1000, i32 %dma.buffer.offset998, <4 x i32> %dma.buffer.rsrc999) #10, !dbg !9
  %lds.const.gep904 = getelementptr i8, ptr addrspace(3) %address.local236, i64 8192, !dbg !8
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep904, align 8, !dbg !8
  %lds.const.gep905 = getelementptr i8, ptr addrspace(3) %address.local236, i64 12288, !dbg !8
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep905, align 8, !dbg !8
  %lds.const.gep906 = getelementptr i8, ptr addrspace(3) %address.local236, i64 9216, !dbg !8
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep906, align 8, !dbg !8
  %137 = fmul <4 x half> %134, splat (half 0xH3015), !dbg !8
  %lds.const.gep907 = getelementptr i8, ptr addrspace(3) %address.local236, i64 13312, !dbg !8
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep907, align 8, !dbg !8
  %139 = fmul <4 x half> %135, splat (half 0xH3015), !dbg !8
  %140 = fmul <4 x half> %136, splat (half 0xH3015), !dbg !8
  %141 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !9
  %143 = getelementptr i8, ptr addrspace(1) %142, i64 192, !dbg !9
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local218), !dbg !9
  %145 = zext i32 %144 to i64, !dbg !9
  %146 = inttoptr i64 %145 to ptr addrspace(3), !dbg !9
  %address.local285 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local202, !dbg !9
  %147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %148 = ptrtoint ptr addrspace(1) %address.local285 to i64, !dbg !9
  %149 = sub i64 %148, %147, !dbg !9
  %dma.buffer.offset995 = trunc i64 %149 to i32, !dbg !9
  %150 = trunc i64 %147 to i32, !dbg !9
  %151 = insertelement <4 x i32> poison, i32 %150, i64 0, !dbg !9
  %152 = lshr i64 %147, 32, !dbg !9
  %153 = trunc i64 %152 to i32, !dbg !9
  %154 = insertelement <4 x i32> %151, i32 %153, i64 1, !dbg !9
  %155 = insertelement <4 x i32> %154, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc996 = insertelement <4 x i32> %155, i32 131072, i64 3, !dbg !9
  %156 = ptrtoint ptr addrspace(3) %146 to i32, !dbg !9
  %dma.buffer.lds997 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %156), !dbg !9
  %157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds997, i32 %dma.buffer.offset995, <4 x i32> %dma.buffer.rsrc996) #10, !dbg !9
  %158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local222), !dbg !9
  %159 = zext i32 %158 to i64, !dbg !9
  %160 = inttoptr i64 %159 to ptr addrspace(3), !dbg !9
  %address.local286 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local214, !dbg !9
  %161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %162 = ptrtoint ptr addrspace(1) %address.local286 to i64, !dbg !9
  %163 = sub i64 %162, %161, !dbg !9
  %dma.buffer.offset992 = trunc i64 %163 to i32, !dbg !9
  %164 = trunc i64 %161 to i32, !dbg !9
  %165 = insertelement <4 x i32> poison, i32 %164, i64 0, !dbg !9
  %166 = lshr i64 %161, 32, !dbg !9
  %167 = trunc i64 %166 to i32, !dbg !9
  %168 = insertelement <4 x i32> %165, i32 %167, i64 1, !dbg !9
  %169 = insertelement <4 x i32> %168, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc993 = insertelement <4 x i32> %169, i32 131072, i64 3, !dbg !9
  %170 = ptrtoint ptr addrspace(3) %160 to i32, !dbg !9
  %dma.buffer.lds994 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !9
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds994, i32 %dma.buffer.offset992, <4 x i32> %dma.buffer.rsrc993) #10, !dbg !9
  %172 = load <4 x half>, ptr addrspace(3) %address.local236, align 8, !dbg !8
  %lds.const.gep908 = getelementptr i8, ptr addrspace(3) %address.local236, i64 4096, !dbg !8
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep908, align 8, !dbg !8
  %lds.const.gep909 = getelementptr i8, ptr addrspace(3) %address.local236, i64 1024, !dbg !8
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep909, align 8, !dbg !8
  %175 = fmul <4 x half> %172, splat (half 0xH3015), !dbg !8
  %lds.const.gep910 = getelementptr i8, ptr addrspace(3) %address.local236, i64 5120, !dbg !8
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep910, align 8, !dbg !8
  %177 = fmul <4 x half> %173, splat (half 0xH3015), !dbg !8
  %178 = fmul <4 x half> %174, splat (half 0xH3015), !dbg !8
  %179 = fmul <4 x half> %176, splat (half 0xH3015), !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %lds.const.gep911 = getelementptr i8, ptr addrspace(3) %address.local236, i64 8192, !dbg !8
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep911, align 8, !dbg !8
  %lds.const.gep912 = getelementptr i8, ptr addrspace(3) %address.local236, i64 12288, !dbg !8
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep912, align 8, !dbg !8
  %lds.const.gep913 = getelementptr i8, ptr addrspace(3) %address.local236, i64 9216, !dbg !8
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep913, align 8, !dbg !8
  %183 = fmul <4 x half> %180, splat (half 0xH3015), !dbg !8
  %lds.const.gep914 = getelementptr i8, ptr addrspace(3) %address.local236, i64 13312, !dbg !8
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep914, align 8, !dbg !8
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
  %index.compact159 = zext nneg i32 %201 to i64, !dbg !17
  %202 = or disjoint i64 %200, %index.compact159, !dbg !17
  %203 = add nuw nsw i64 %202, 1, !dbg !17
  %204 = sdiv i64 %203, 2, !dbg !17
  br label %205, !dbg !7

205:                                              ; preds = %584, %4
  %206 = phi i64 [ %1103, %584 ], [ 0, %4 ], !dbg !7
  %207 = phi [2 x <8 x float>] [ %1088, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %208 = phi [2 x <1 x float>] [ %864, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %209 = phi [2 x <1 x float>] [ %1090, %584 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %210 = phi [2 x <8 x float>] [ %1094, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %211 = phi [2 x <8 x float>] [ %1098, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %212 = phi [2 x <8 x float>] [ %1102, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %213 = icmp slt i64 %206, %192, !dbg !7
  br i1 %213, label %214, label %1104, !dbg !7

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
  %address.thread287 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local288 = zext nneg i32 %address.thread287 to i64, !dbg !9
  %address.local289 = shl nuw nsw i64 %address.local288, 3, !dbg !9
  %address.local290 = and i64 %address.local289, 1536, !dbg !16
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local290, !dbg !16
  %address.local292 = ptrtoint ptr addrspace(3) %address.local291 to i64, !dbg !16
  %address.local293 = trunc i64 %address.local292 to i32, !dbg !16
  %228 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local293), !dbg !16
  %229 = zext i32 %228 to i64, !dbg !16
  %230 = inttoptr i64 %229 to ptr addrspace(3), !dbg !16
  %address.local294 = lshr i64 %address.local288, 3, !dbg !16
  %address.local295 = and i64 %address.local294, 24, !dbg !16
  %address.local296 = and i64 %address.local288, 59, !dbg !16
  %address.local297 = lshr i64 %address.local288, 2, !dbg !16
  %address.local298 = and i64 %address.local297, 4, !dbg !16
  %address.local299 = add nuw nsw i64 %address.local298, %address.local288, !dbg !16
  %address.local300 = and i64 %address.local299, 4, !dbg !16
  %address.local301 = or disjoint i64 %address.local296, %address.local300, !dbg !16
  %address.local302 = shl nuw nsw i64 %address.local301, 7, !dbg !16
  %address.local303 = or disjoint i64 %address.local295, %address.local302, !dbg !16
  %address.local304 = getelementptr [2 x i8], ptr addrspace(1) %227, i64 %address.local303, !dbg !16
  %231 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %232 = ptrtoint ptr addrspace(1) %address.local304 to i64, !dbg !16
  %233 = sub i64 %232, %231, !dbg !16
  %dma.buffer.offset989 = trunc i64 %233 to i32, !dbg !16
  %234 = trunc i64 %231 to i32, !dbg !16
  %235 = insertelement <4 x i32> poison, i32 %234, i64 0, !dbg !16
  %236 = lshr i64 %231, 32, !dbg !16
  %237 = trunc i64 %236 to i32, !dbg !16
  %238 = insertelement <4 x i32> %235, i32 %237, i64 1, !dbg !16
  %239 = insertelement <4 x i32> %238, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc990 = insertelement <4 x i32> %239, i32 131072, i64 3, !dbg !16
  %240 = ptrtoint ptr addrspace(3) %230 to i32, !dbg !16
  %dma.buffer.lds991 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %240), !dbg !16
  %241 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds991, i32 %dma.buffer.offset989, <4 x i32> %dma.buffer.rsrc990) #10, !dbg !16
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
  %address.local305 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local290, !dbg !16
  %address.local306 = ptrtoint ptr addrspace(3) %address.local305 to i64, !dbg !16
  %address.local307 = trunc i64 %address.local306 to i32, !dbg !16
  %266 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local307), !dbg !16
  %267 = zext i32 %266 to i64, !dbg !16
  %268 = inttoptr i64 %267 to ptr addrspace(3), !dbg !16
  %address.local308 = getelementptr [2 x i8], ptr addrspace(1) %265, i64 %address.local303, !dbg !16
  %269 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %270 = ptrtoint ptr addrspace(1) %address.local308 to i64, !dbg !16
  %271 = sub i64 %270, %269, !dbg !16
  %dma.buffer.offset986 = trunc i64 %271 to i32, !dbg !16
  %272 = trunc i64 %269 to i32, !dbg !16
  %273 = insertelement <4 x i32> poison, i32 %272, i64 0, !dbg !16
  %274 = lshr i64 %269, 32, !dbg !16
  %275 = trunc i64 %274 to i32, !dbg !16
  %276 = insertelement <4 x i32> %273, i32 %275, i64 1, !dbg !16
  %277 = insertelement <4 x i32> %276, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc987 = insertelement <4 x i32> %277, i32 131072, i64 3, !dbg !16
  %278 = ptrtoint ptr addrspace(3) %268 to i32, !dbg !16
  %dma.buffer.lds988 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %278), !dbg !16
  %279 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds988, i32 %dma.buffer.offset986, <4 x i32> %dma.buffer.rsrc987) #10, !dbg !16
  %address.local309 = mul nuw nsw i64 %address.local288, 160, !dbg !17
  %address.local310 = lshr i64 %address.local288, 2, !dbg !17
  %address.local311 = mul nsw i64 %address.local310, -640, !dbg !17
  %address.local312 = add nsw i64 %address.local309, %address.local311, !dbg !17
  %address.local313 = shl nuw nsw i64 %address.local288, 1, !dbg !17
  %address.local314 = and i64 %address.local313, 24, !dbg !17
  %address.local315 = or disjoint i64 %address.local312, %address.local314, !dbg !17
  %address.local316 = shl i64 %address.local288, 62, !dbg !17
  %address.local317 = ashr exact i64 %address.local316, 57, !dbg !17
  %address.local318 = and i64 %address.local317, -64, !dbg !17
  %address.local319 = add nsw i64 %address.local315, %address.local318, !dbg !17
  %address.local320 = lshr i64 %address.local288, 2, !dbg !8
  %address.local321 = and i64 %address.local320, 12, !dbg !8
  %address.local322 = add nsw i64 %address.local319, %address.local321, !dbg !17
  %address.local323 = and i64 %address.local288, 63, !dbg !8
  %address.local324 = lshr i64 %address.local323, 5, !dbg !8
  %address.local325 = mul nuw nsw i64 %address.local324, 504, !dbg !17
  %address.local326 = add i64 %address.local322, %address.local325, !dbg !17
  %address.local327 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local326, !dbg !17
  %280 = load <4 x half>, ptr addrspace(3) %address.local327, align 8, !dbg !17
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %280, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep915 = getelementptr i8, ptr addrspace(3) %address.local327, i64 2048, !dbg !17
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep915, align 8, !dbg !17
  %address.local338 = shl nuw nsw i32 %address.thread287, 1, !dbg !17
  %address.local339 = and i32 %address.local338, 24, !dbg !17
  %address.local340 = shl nuw nsw i32 %address.thread287, 8, !dbg !17
  %address.local341 = and i32 %address.local340, 768, !dbg !17
  %address.local342 = or disjoint i32 %address.local339, %address.local341, !dbg !17
  %address.local343 = shl nuw nsw i32 %address.thread287, 5, !dbg !17
  %address.local344 = and i32 %address.local343, 96, !dbg !17
  %address.local345 = or disjoint i32 %address.local342, %address.local344, !dbg !17
  %address.local346 = zext nneg i32 %address.local345 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %204) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local347 = lshr i32 %address.thread287, 1, !dbg !17
  %address.local348 = and i32 %address.local347, 1, !dbg !17
  %address.local349 = zext nneg i32 %address.local348 to i64, !dbg !17
  %address.local350 = add nsw i64 %address.base, %address.local349, !dbg !17
  %address.local351 = shl i64 %address.local350, 6, !dbg !17
  %address.local352 = sub i64 %address.local346, %address.local351, !dbg !17
  %address.local353 = add i64 %address.local352, %address.local321, !dbg !17
  %address.local354 = add i64 %address.local353, %address.local325, !dbg !17
  %address.local355 = add i64 %address.local354, 32, !dbg !17
  %address.local356 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local355, !dbg !17
  %283 = load <4 x half>, ptr addrspace(3) %address.local356, align 8, !dbg !17
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %282, <4 x float> %281) #9, !dbg !17
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %283, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep916 = getelementptr i8, ptr addrspace(3) %address.local356, i64 2048, !dbg !17
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep916, align 8, !dbg !17
  %lds.const.gep917 = getelementptr i8, ptr addrspace(3) %address.local327, i64 128, !dbg !17
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep917, align 8, !dbg !17
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %286, <4 x float> %285) #9, !dbg !17
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %287, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep918 = getelementptr i8, ptr addrspace(3) %address.local327, i64 2176, !dbg !17
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep918, align 8, !dbg !17
  %lds.const.gep919 = getelementptr i8, ptr addrspace(3) %address.local356, i64 128, !dbg !17
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep919, align 8, !dbg !17
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %290, <4 x float> %289) #9, !dbg !17
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %291, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep920 = getelementptr i8, ptr addrspace(3) %address.local356, i64 2176, !dbg !17
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep920, align 8, !dbg !17
  %295 = load <4 x half>, ptr addrspace(3) %address.local327, align 8, !dbg !17
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %294, <4 x float> %293) #9, !dbg !17
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %295, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep921 = getelementptr i8, ptr addrspace(3) %address.local327, i64 2048, !dbg !17
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep921, align 8, !dbg !17
  %299 = load <4 x half>, ptr addrspace(3) %address.local356, align 8, !dbg !17
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %298, <4 x float> %297) #9, !dbg !17
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %299, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep922 = getelementptr i8, ptr addrspace(3) %address.local356, i64 2048, !dbg !17
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep922, align 8, !dbg !17
  %lds.const.gep923 = getelementptr i8, ptr addrspace(3) %address.local327, i64 128, !dbg !17
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep923, align 8, !dbg !17
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %302, <4 x float> %301) #9, !dbg !17
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %303, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep924 = getelementptr i8, ptr addrspace(3) %address.local327, i64 2176, !dbg !17
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep924, align 8, !dbg !17
  %lds.const.gep925 = getelementptr i8, ptr addrspace(3) %address.local356, i64 128, !dbg !17
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep925, align 8, !dbg !17
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %306, <4 x float> %305) #9, !dbg !17
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %307, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep926 = getelementptr i8, ptr addrspace(3) %address.local356, i64 2176, !dbg !17
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep926, align 8, !dbg !17
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
  %335 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local293), !dbg !16
  %336 = zext i32 %335 to i64, !dbg !16
  %337 = inttoptr i64 %336 to ptr addrspace(3), !dbg !16
  %address.local497 = getelementptr [2 x i8], ptr addrspace(1) %334, i64 %address.local303, !dbg !16
  %338 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %339 = ptrtoint ptr addrspace(1) %address.local497 to i64, !dbg !16
  %340 = sub i64 %339, %338, !dbg !16
  %dma.buffer.offset983 = trunc i64 %340 to i32, !dbg !16
  %341 = trunc i64 %338 to i32, !dbg !16
  %342 = insertelement <4 x i32> poison, i32 %341, i64 0, !dbg !16
  %343 = lshr i64 %338, 32, !dbg !16
  %344 = trunc i64 %343 to i32, !dbg !16
  %345 = insertelement <4 x i32> %342, i32 %344, i64 1, !dbg !16
  %346 = insertelement <4 x i32> %345, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc984 = insertelement <4 x i32> %346, i32 131072, i64 3, !dbg !16
  %347 = ptrtoint ptr addrspace(3) %337 to i32, !dbg !16
  %dma.buffer.lds985 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %347), !dbg !16
  %348 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds985, i32 %dma.buffer.offset983, <4 x i32> %dma.buffer.rsrc984) #10, !dbg !16
  %lds.const.gep927 = getelementptr i8, ptr addrspace(3) %address.local327, i64 4096, !dbg !17
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep927, align 8, !dbg !17
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %349, <4 x float> %284) #9, !dbg !17
  %lds.const.gep928 = getelementptr i8, ptr addrspace(3) %address.local327, i64 6144, !dbg !17
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep928, align 8, !dbg !17
  %lds.const.gep929 = getelementptr i8, ptr addrspace(3) %address.local356, i64 4096, !dbg !17
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep929, align 8, !dbg !17
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %351, <4 x float> %350) #9, !dbg !17
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %352, <4 x float> %288) #9, !dbg !17
  %lds.const.gep930 = getelementptr i8, ptr addrspace(3) %address.local356, i64 6144, !dbg !17
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep930, align 8, !dbg !17
  %lds.const.gep931 = getelementptr i8, ptr addrspace(3) %address.local327, i64 4224, !dbg !17
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep931, align 8, !dbg !17
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %355, <4 x float> %354) #9, !dbg !17
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %356, <4 x float> %292) #9, !dbg !17
  %lds.const.gep932 = getelementptr i8, ptr addrspace(3) %address.local327, i64 6272, !dbg !17
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep932, align 8, !dbg !17
  %lds.const.gep933 = getelementptr i8, ptr addrspace(3) %address.local356, i64 4224, !dbg !17
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep933, align 8, !dbg !17
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %359, <4 x float> %358) #9, !dbg !17
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %360, <4 x float> %296) #9, !dbg !17
  %lds.const.gep934 = getelementptr i8, ptr addrspace(3) %address.local356, i64 6272, !dbg !17
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep934, align 8, !dbg !17
  %lds.const.gep935 = getelementptr i8, ptr addrspace(3) %address.local327, i64 4096, !dbg !17
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep935, align 8, !dbg !17
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %363, <4 x float> %362) #9, !dbg !17
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %364, <4 x float> %300) #9, !dbg !17
  %lds.const.gep936 = getelementptr i8, ptr addrspace(3) %address.local327, i64 6144, !dbg !17
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep936, align 8, !dbg !17
  %lds.const.gep937 = getelementptr i8, ptr addrspace(3) %address.local356, i64 4096, !dbg !17
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep937, align 8, !dbg !17
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %367, <4 x float> %366) #9, !dbg !17
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %368, <4 x float> %304) #9, !dbg !17
  %lds.const.gep938 = getelementptr i8, ptr addrspace(3) %address.local356, i64 6144, !dbg !17
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep938, align 8, !dbg !17
  %lds.const.gep939 = getelementptr i8, ptr addrspace(3) %address.local327, i64 4224, !dbg !17
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep939, align 8, !dbg !17
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %371, <4 x float> %370) #9, !dbg !17
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %372, <4 x float> %308) #9, !dbg !17
  %lds.const.gep940 = getelementptr i8, ptr addrspace(3) %address.local327, i64 6272, !dbg !17
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep940, align 8, !dbg !17
  %lds.const.gep941 = getelementptr i8, ptr addrspace(3) %address.local356, i64 4224, !dbg !17
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep941, align 8, !dbg !17
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %375, <4 x float> %374) #9, !dbg !17
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %376, <4 x float> %311) #9, !dbg !17
  %lds.const.gep942 = getelementptr i8, ptr addrspace(3) %address.local356, i64 6272, !dbg !17
  %379 = load <4 x half>, ptr addrspace(3) %lds.const.gep942, align 8, !dbg !17
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
  %404 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local307), !dbg !16
  %405 = zext i32 %404 to i64, !dbg !16
  %406 = inttoptr i64 %405 to ptr addrspace(3), !dbg !16
  %address.local506 = getelementptr [2 x i8], ptr addrspace(1) %403, i64 %address.local303, !dbg !16
  %407 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %408 = ptrtoint ptr addrspace(1) %address.local506 to i64, !dbg !16
  %409 = sub i64 %408, %407, !dbg !16
  %dma.buffer.offset980 = trunc i64 %409 to i32, !dbg !16
  %410 = trunc i64 %407 to i32, !dbg !16
  %411 = insertelement <4 x i32> poison, i32 %410, i64 0, !dbg !16
  %412 = lshr i64 %407, 32, !dbg !16
  %413 = trunc i64 %412 to i32, !dbg !16
  %414 = insertelement <4 x i32> %411, i32 %413, i64 1, !dbg !16
  %415 = insertelement <4 x i32> %414, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc981 = insertelement <4 x i32> %415, i32 131072, i64 3, !dbg !16
  %416 = ptrtoint ptr addrspace(3) %406 to i32, !dbg !16
  %dma.buffer.lds982 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %416), !dbg !16
  %417 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds982, i32 %dma.buffer.offset980, <4 x i32> %dma.buffer.rsrc981) #10, !dbg !16
  %418 = load <4 x half>, ptr addrspace(3) %address.local327, align 8, !dbg !17
  %419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %418, <4 x float> %353) #9, !dbg !17
  %lds.const.gep943 = getelementptr i8, ptr addrspace(3) %address.local327, i64 2048, !dbg !17
  %420 = load <4 x half>, ptr addrspace(3) %lds.const.gep943, align 8, !dbg !17
  %421 = load <4 x half>, ptr addrspace(3) %address.local356, align 8, !dbg !17
  %422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %420, <4 x float> %419) #9, !dbg !17
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %421, <4 x float> %357) #9, !dbg !17
  %lds.const.gep944 = getelementptr i8, ptr addrspace(3) %address.local356, i64 2048, !dbg !17
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep944, align 8, !dbg !17
  %lds.const.gep945 = getelementptr i8, ptr addrspace(3) %address.local327, i64 128, !dbg !17
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep945, align 8, !dbg !17
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %424, <4 x float> %423) #9, !dbg !17
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %425, <4 x float> %361) #9, !dbg !17
  %lds.const.gep946 = getelementptr i8, ptr addrspace(3) %address.local327, i64 2176, !dbg !17
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep946, align 8, !dbg !17
  %lds.const.gep947 = getelementptr i8, ptr addrspace(3) %address.local356, i64 128, !dbg !17
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep947, align 8, !dbg !17
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %428, <4 x float> %427) #9, !dbg !17
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %429, <4 x float> %365) #9, !dbg !17
  %lds.const.gep948 = getelementptr i8, ptr addrspace(3) %address.local356, i64 2176, !dbg !17
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep948, align 8, !dbg !17
  %433 = load <4 x half>, ptr addrspace(3) %address.local327, align 8, !dbg !17
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %432, <4 x float> %431) #9, !dbg !17
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %433, <4 x float> %369) #9, !dbg !17
  %lds.const.gep949 = getelementptr i8, ptr addrspace(3) %address.local327, i64 2048, !dbg !17
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep949, align 8, !dbg !17
  %437 = load <4 x half>, ptr addrspace(3) %address.local356, align 8, !dbg !17
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %436, <4 x float> %435) #9, !dbg !17
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %437, <4 x float> %373) #9, !dbg !17
  %lds.const.gep950 = getelementptr i8, ptr addrspace(3) %address.local356, i64 2048, !dbg !17
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep950, align 8, !dbg !17
  %lds.const.gep951 = getelementptr i8, ptr addrspace(3) %address.local327, i64 128, !dbg !17
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep951, align 8, !dbg !17
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %440, <4 x float> %439) #9, !dbg !17
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %441, <4 x float> %377) #9, !dbg !17
  %lds.const.gep952 = getelementptr i8, ptr addrspace(3) %address.local327, i64 2176, !dbg !17
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep952, align 8, !dbg !17
  %lds.const.gep953 = getelementptr i8, ptr addrspace(3) %address.local356, i64 128, !dbg !17
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep953, align 8, !dbg !17
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %444, <4 x float> %443) #9, !dbg !17
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %445, <4 x float> %380) #9, !dbg !17
  %lds.const.gep954 = getelementptr i8, ptr addrspace(3) %address.local356, i64 2176, !dbg !17
  %448 = load <4 x half>, ptr addrspace(3) %lds.const.gep954, align 8, !dbg !17
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
  %455 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local293), !dbg !18
  %456 = zext i32 %455 to i64, !dbg !18
  %457 = inttoptr i64 %456 to ptr addrspace(3), !dbg !18
  %address.local507 = shl nuw nsw i64 %address.local288, 5, !dbg !18
  %address.local508 = and i64 %address.local507, 8064, !dbg !18
  %address.local509 = shl nuw nsw i64 %address.local288, 3, !dbg !18
  %address.local510 = and i64 %address.local509, 24, !dbg !18
  %address.local511 = or disjoint i64 %address.local508, %address.local510, !dbg !18
  %address.local512 = getelementptr [2 x i8], ptr addrspace(1) %454, i64 %address.local511, !dbg !18
  %458 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %459 = ptrtoint ptr addrspace(1) %address.local512 to i64, !dbg !18
  %460 = sub i64 %459, %458, !dbg !18
  %dma.buffer.offset977 = trunc i64 %460 to i32, !dbg !18
  %461 = trunc i64 %458 to i32, !dbg !18
  %462 = insertelement <4 x i32> poison, i32 %461, i64 0, !dbg !18
  %463 = lshr i64 %458, 32, !dbg !18
  %464 = trunc i64 %463 to i32, !dbg !18
  %465 = insertelement <4 x i32> %462, i32 %464, i64 1, !dbg !18
  %466 = insertelement <4 x i32> %465, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc978 = insertelement <4 x i32> %466, i32 131072, i64 3, !dbg !18
  %467 = ptrtoint ptr addrspace(3) %457 to i32, !dbg !18
  %dma.buffer.lds979 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %467), !dbg !18
  %468 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds979, i32 %dma.buffer.offset977, <4 x i32> %dma.buffer.rsrc978) #10, !dbg !18
  %lds.const.gep955 = getelementptr i8, ptr addrspace(3) %address.local327, i64 4096, !dbg !17
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep955, align 8, !dbg !17
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %469, <4 x float> %422) #9, !dbg !17
  %lds.const.gep956 = getelementptr i8, ptr addrspace(3) %address.local327, i64 6144, !dbg !17
  %471 = load <4 x half>, ptr addrspace(3) %lds.const.gep956, align 8, !dbg !17
  %lds.const.gep957 = getelementptr i8, ptr addrspace(3) %address.local356, i64 4096, !dbg !17
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep957, align 8, !dbg !17
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %471, <4 x float> %470) #9, !dbg !17
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %472, <4 x float> %426) #9, !dbg !17
  %lds.const.gep958 = getelementptr i8, ptr addrspace(3) %address.local356, i64 6144, !dbg !17
  %475 = load <4 x half>, ptr addrspace(3) %lds.const.gep958, align 8, !dbg !17
  %lds.const.gep959 = getelementptr i8, ptr addrspace(3) %address.local327, i64 4224, !dbg !17
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep959, align 8, !dbg !17
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %475, <4 x float> %474) #9, !dbg !17
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %476, <4 x float> %430) #9, !dbg !17
  %lds.const.gep960 = getelementptr i8, ptr addrspace(3) %address.local327, i64 6272, !dbg !17
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep960, align 8, !dbg !17
  %lds.const.gep961 = getelementptr i8, ptr addrspace(3) %address.local356, i64 4224, !dbg !17
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep961, align 8, !dbg !17
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %479, <4 x float> %478) #9, !dbg !17
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %480, <4 x float> %434) #9, !dbg !17
  %lds.const.gep962 = getelementptr i8, ptr addrspace(3) %address.local356, i64 6272, !dbg !17
  %483 = load <4 x half>, ptr addrspace(3) %lds.const.gep962, align 8, !dbg !17
  %lds.const.gep963 = getelementptr i8, ptr addrspace(3) %address.local327, i64 4096, !dbg !17
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep963, align 8, !dbg !17
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %483, <4 x float> %482) #9, !dbg !17
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %484, <4 x float> %438) #9, !dbg !17
  %lds.const.gep964 = getelementptr i8, ptr addrspace(3) %address.local327, i64 6144, !dbg !17
  %487 = load <4 x half>, ptr addrspace(3) %lds.const.gep964, align 8, !dbg !17
  %lds.const.gep965 = getelementptr i8, ptr addrspace(3) %address.local356, i64 4096, !dbg !17
  %488 = load <4 x half>, ptr addrspace(3) %lds.const.gep965, align 8, !dbg !17
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %487, <4 x float> %486) #9, !dbg !17
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %488, <4 x float> %442) #9, !dbg !17
  %lds.const.gep966 = getelementptr i8, ptr addrspace(3) %address.local356, i64 6144, !dbg !17
  %491 = load <4 x half>, ptr addrspace(3) %lds.const.gep966, align 8, !dbg !17
  %lds.const.gep967 = getelementptr i8, ptr addrspace(3) %address.local327, i64 4224, !dbg !17
  %492 = load <4 x half>, ptr addrspace(3) %lds.const.gep967, align 8, !dbg !17
  %493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %491, <4 x float> %490) #9, !dbg !17
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %492, <4 x float> %446) #9, !dbg !17
  %lds.const.gep968 = getelementptr i8, ptr addrspace(3) %address.local327, i64 6272, !dbg !17
  %495 = load <4 x half>, ptr addrspace(3) %lds.const.gep968, align 8, !dbg !17
  %lds.const.gep969 = getelementptr i8, ptr addrspace(3) %address.local356, i64 4224, !dbg !17
  %496 = load <4 x half>, ptr addrspace(3) %lds.const.gep969, align 8, !dbg !17
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %495, <4 x float> %494) #9, !dbg !17
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %496, <4 x float> %449) #9, !dbg !17
  %lds.const.gep970 = getelementptr i8, ptr addrspace(3) %address.local356, i64 6272, !dbg !17
  %499 = load <4 x half>, ptr addrspace(3) %lds.const.gep970, align 8, !dbg !17
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
  %compare.high774 = trunc i64 %515 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high774, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high774, !dbg !24
  %compare.low.base = trunc i64 %513 to i32, !dbg !24
  %compare.low.base775 = trunc i64 %512 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base775, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %516 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low776 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result777 = select i1 %compare.high.equal, i1 %compare.low776, i1 %compare.high.order, !dbg !24
  %517 = select i1 %compare.result777, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base778 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local779 = or i32 %compare.local.base778, 2, !dbg !24
  %compare.low780 = icmp ugt i32 %compare.low.base, %compare.local779, !dbg !24
  %compare.result781 = select i1 %compare.high.equal, i1 %compare.low780, i1 %compare.high.order, !dbg !24
  %518 = select i1 %compare.result781, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base782 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local783 = or i32 %compare.local.base782, 3, !dbg !24
  %compare.low784 = icmp ugt i32 %compare.low.base, %compare.local783, !dbg !24
  %compare.result785 = select i1 %compare.high.equal, i1 %compare.low784, i1 %compare.high.order, !dbg !24
  %519 = select i1 %compare.result785, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base786 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local787 = or i32 %compare.local.base786, 4, !dbg !24
  %compare.low788 = icmp ugt i32 %compare.low.base, %compare.local787, !dbg !24
  %compare.result789 = select i1 %compare.high.equal, i1 %compare.low788, i1 %compare.high.order, !dbg !24
  %520 = select i1 %compare.result789, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base790 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local791 = or i32 %compare.local.base790, 5, !dbg !24
  %compare.low792 = icmp ugt i32 %compare.low.base, %compare.local791, !dbg !24
  %compare.result793 = select i1 %compare.high.equal, i1 %compare.low792, i1 %compare.high.order, !dbg !24
  %521 = select i1 %compare.result793, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base794 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local795 = or i32 %compare.local.base794, 6, !dbg !24
  %compare.low796 = icmp ugt i32 %compare.low.base, %compare.local795, !dbg !24
  %compare.result797 = select i1 %compare.high.equal, i1 %compare.low796, i1 %compare.high.order, !dbg !24
  %522 = select i1 %compare.result797, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base798 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local799 = or i32 %compare.local.base798, 7, !dbg !24
  %compare.low800 = icmp ugt i32 %compare.low.base, %compare.local799, !dbg !24
  %compare.result801 = select i1 %compare.high.equal, i1 %compare.low800, i1 %compare.high.order, !dbg !24
  %523 = select i1 %compare.result801, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base802 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local803 = or i32 %compare.local.base802, 8, !dbg !24
  %compare.low804 = icmp ugt i32 %compare.low.base, %compare.local803, !dbg !24
  %compare.result805 = select i1 %compare.high.equal, i1 %compare.low804, i1 %compare.high.order, !dbg !24
  %524 = select i1 %compare.result805, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base806 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local807 = or i32 %compare.local.base806, 9, !dbg !24
  %compare.low808 = icmp ugt i32 %compare.low.base, %compare.local807, !dbg !24
  %compare.result809 = select i1 %compare.high.equal, i1 %compare.low808, i1 %compare.high.order, !dbg !24
  %525 = select i1 %compare.result809, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base810 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local811 = or i32 %compare.local.base810, 10, !dbg !24
  %compare.low812 = icmp ugt i32 %compare.low.base, %compare.local811, !dbg !24
  %compare.result813 = select i1 %compare.high.equal, i1 %compare.low812, i1 %compare.high.order, !dbg !24
  %526 = select i1 %compare.result813, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base814 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local815 = or i32 %compare.local.base814, 11, !dbg !24
  %compare.low816 = icmp ugt i32 %compare.low.base, %compare.local815, !dbg !24
  %compare.result817 = select i1 %compare.high.equal, i1 %compare.low816, i1 %compare.high.order, !dbg !24
  %527 = select i1 %compare.result817, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base818 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local819 = or i32 %compare.local.base818, 12, !dbg !24
  %compare.low820 = icmp ugt i32 %compare.low.base, %compare.local819, !dbg !24
  %compare.result821 = select i1 %compare.high.equal, i1 %compare.low820, i1 %compare.high.order, !dbg !24
  %528 = select i1 %compare.result821, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base822 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local823 = or i32 %compare.local.base822, 13, !dbg !24
  %compare.low824 = icmp ugt i32 %compare.low.base, %compare.local823, !dbg !24
  %compare.result825 = select i1 %compare.high.equal, i1 %compare.low824, i1 %compare.high.order, !dbg !24
  %529 = select i1 %compare.result825, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base826 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local827 = or i32 %compare.local.base826, 14, !dbg !24
  %compare.low828 = icmp ugt i32 %compare.low.base, %compare.local827, !dbg !24
  %compare.result829 = select i1 %compare.high.equal, i1 %compare.low828, i1 %compare.high.order, !dbg !24
  %530 = select i1 %compare.result829, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base830 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base775, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local831 = or i32 %compare.local.base830, 15, !dbg !24
  %compare.low832 = icmp ugt i32 %compare.low.base, %compare.local831, !dbg !24
  %compare.result833 = select i1 %compare.high.equal, i1 %compare.low832, i1 %compare.high.order, !dbg !24
  %531 = select i1 %compare.result833, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %532 = add i64 %510, 65, !dbg !23
  %533 = lshr i64 %532, 32, !dbg !24
  %compare.high834 = trunc i64 %533 to i32, !dbg !24
  %534 = lshr i64 %512, 32, !dbg !24
  %compare.high835 = trunc i64 %534 to i32, !dbg !24
  %compare.high.equal836 = icmp eq i32 %compare.high834, %compare.high835, !dbg !24
  %compare.high.order837 = icmp ugt i32 %compare.high834, %compare.high835, !dbg !24
  %compare.low.base838 = trunc i64 %532 to i32, !dbg !24
  %compare.low.base839 = trunc i64 %512 to i32, !dbg !24
  %compare.low840 = icmp ugt i32 %compare.low.base838, %compare.low.base839, !dbg !24
  %compare.result841 = select i1 %compare.high.equal836, i1 %compare.low840, i1 %compare.high.order837, !dbg !24
  %535 = select i1 %compare.result841, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base842 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local843 = or i32 %compare.local.base842, 1, !dbg !24
  %compare.low844 = icmp ugt i32 %compare.low.base838, %compare.local843, !dbg !24
  %compare.result845 = select i1 %compare.high.equal836, i1 %compare.low844, i1 %compare.high.order837, !dbg !24
  %536 = select i1 %compare.result845, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base846 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local847 = or i32 %compare.local.base846, 2, !dbg !24
  %compare.low848 = icmp ugt i32 %compare.low.base838, %compare.local847, !dbg !24
  %compare.result849 = select i1 %compare.high.equal836, i1 %compare.low848, i1 %compare.high.order837, !dbg !24
  %537 = select i1 %compare.result849, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base850 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local851 = or i32 %compare.local.base850, 3, !dbg !24
  %compare.low852 = icmp ugt i32 %compare.low.base838, %compare.local851, !dbg !24
  %compare.result853 = select i1 %compare.high.equal836, i1 %compare.low852, i1 %compare.high.order837, !dbg !24
  %538 = select i1 %compare.result853, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base854 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local855 = or i32 %compare.local.base854, 4, !dbg !24
  %compare.low856 = icmp ugt i32 %compare.low.base838, %compare.local855, !dbg !24
  %compare.result857 = select i1 %compare.high.equal836, i1 %compare.low856, i1 %compare.high.order837, !dbg !24
  %539 = select i1 %compare.result857, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base858 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local859 = or i32 %compare.local.base858, 5, !dbg !24
  %compare.low860 = icmp ugt i32 %compare.low.base838, %compare.local859, !dbg !24
  %compare.result861 = select i1 %compare.high.equal836, i1 %compare.low860, i1 %compare.high.order837, !dbg !24
  %540 = select i1 %compare.result861, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base862 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local863 = or i32 %compare.local.base862, 6, !dbg !24
  %compare.low864 = icmp ugt i32 %compare.low.base838, %compare.local863, !dbg !24
  %compare.result865 = select i1 %compare.high.equal836, i1 %compare.low864, i1 %compare.high.order837, !dbg !24
  %541 = select i1 %compare.result865, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base866 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local867 = or i32 %compare.local.base866, 7, !dbg !24
  %compare.low868 = icmp ugt i32 %compare.low.base838, %compare.local867, !dbg !24
  %compare.result869 = select i1 %compare.high.equal836, i1 %compare.low868, i1 %compare.high.order837, !dbg !24
  %542 = select i1 %compare.result869, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base870 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local871 = or i32 %compare.local.base870, 8, !dbg !24
  %compare.low872 = icmp ugt i32 %compare.low.base838, %compare.local871, !dbg !24
  %compare.result873 = select i1 %compare.high.equal836, i1 %compare.low872, i1 %compare.high.order837, !dbg !24
  %543 = select i1 %compare.result873, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base874 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local875 = or i32 %compare.local.base874, 9, !dbg !24
  %compare.low876 = icmp ugt i32 %compare.low.base838, %compare.local875, !dbg !24
  %compare.result877 = select i1 %compare.high.equal836, i1 %compare.low876, i1 %compare.high.order837, !dbg !24
  %544 = select i1 %compare.result877, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base878 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local879 = or i32 %compare.local.base878, 10, !dbg !24
  %compare.low880 = icmp ugt i32 %compare.low.base838, %compare.local879, !dbg !24
  %compare.result881 = select i1 %compare.high.equal836, i1 %compare.low880, i1 %compare.high.order837, !dbg !24
  %545 = select i1 %compare.result881, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base882 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local883 = or i32 %compare.local.base882, 11, !dbg !24
  %compare.low884 = icmp ugt i32 %compare.low.base838, %compare.local883, !dbg !24
  %compare.result885 = select i1 %compare.high.equal836, i1 %compare.low884, i1 %compare.high.order837, !dbg !24
  %546 = select i1 %compare.result885, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base886 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local887 = or i32 %compare.local.base886, 12, !dbg !24
  %compare.low888 = icmp ugt i32 %compare.low.base838, %compare.local887, !dbg !24
  %compare.result889 = select i1 %compare.high.equal836, i1 %compare.low888, i1 %compare.high.order837, !dbg !24
  %547 = select i1 %compare.result889, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base890 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local891 = or i32 %compare.local.base890, 13, !dbg !24
  %compare.low892 = icmp ugt i32 %compare.low.base838, %compare.local891, !dbg !24
  %compare.result893 = select i1 %compare.high.equal836, i1 %compare.low892, i1 %compare.high.order837, !dbg !24
  %548 = select i1 %compare.result893, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base894 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local895 = or i32 %compare.local.base894, 14, !dbg !24
  %compare.low896 = icmp ugt i32 %compare.low.base838, %compare.local895, !dbg !24
  %compare.result897 = select i1 %compare.high.equal836, i1 %compare.low896, i1 %compare.high.order837, !dbg !24
  %549 = select i1 %compare.result897, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base898 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base839, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local899 = or i32 %compare.local.base898, 15, !dbg !24
  %compare.low900 = icmp ugt i32 %compare.low.base838, %compare.local899, !dbg !24
  %compare.result901 = select i1 %compare.high.equal836, i1 %compare.low900, i1 %compare.high.order837, !dbg !24
  %550 = select i1 %compare.result901, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
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
  %617 = call float @llvm.maxnum.f32(float %585, float 0xFFF0000000000000), !dbg !27
  %618 = call float @llvm.maxnum.f32(float %617, float %586), !dbg !27
  %619 = call float @llvm.maxnum.f32(float %618, float %587), !dbg !27
  %620 = call float @llvm.maxnum.f32(float %619, float %588), !dbg !27
  %621 = call float @llvm.maxnum.f32(float %620, float %589), !dbg !27
  %622 = call float @llvm.maxnum.f32(float %621, float %590), !dbg !27
  %623 = call float @llvm.maxnum.f32(float %622, float %591), !dbg !27
  %624 = call float @llvm.maxnum.f32(float %623, float %592), !dbg !27
  %625 = call float @llvm.maxnum.f32(float %624, float %593), !dbg !27
  %626 = call float @llvm.maxnum.f32(float %625, float %594), !dbg !27
  %627 = call float @llvm.maxnum.f32(float %626, float %595), !dbg !27
  %628 = call float @llvm.maxnum.f32(float %627, float %596), !dbg !27
  %629 = call float @llvm.maxnum.f32(float %628, float %597), !dbg !27
  %630 = call float @llvm.maxnum.f32(float %629, float %598), !dbg !27
  %631 = call float @llvm.maxnum.f32(float %630, float %599), !dbg !27
  %632 = call float @llvm.maxnum.f32(float %631, float %600), !dbg !27
  %633 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %634 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %633), !dbg !27
  %635 = add nuw nsw i32 %634, 64, !dbg !27
  %636 = xor i32 %634, 16, !dbg !27
  %637 = and i32 %635, 192, !dbg !27
  %638 = icmp samesign ult i32 %636, %637, !dbg !27
  %639 = select i1 %638, i32 %636, i32 %634, !dbg !27
  %640 = shl nuw nsw i32 %639, 2, !dbg !27
  %641 = bitcast float %632 to i32, !dbg !27
  %642 = call i32 @llvm.amdgcn.ds.bpermute(i32 %640, i32 %641), !dbg !27
  %643 = bitcast i32 %642 to float, !dbg !27
  %644 = call float @llvm.maxnum.f32(float %632, float %643), !dbg !27
  %645 = xor i32 %634, 32, !dbg !27
  %646 = icmp samesign ult i32 %645, %637, !dbg !27
  %647 = select i1 %646, i32 %645, i32 %634, !dbg !27
  %648 = shl nuw nsw i32 %647, 2, !dbg !27
  %649 = bitcast float %644 to i32, !dbg !27
  %650 = call i32 @llvm.amdgcn.ds.bpermute(i32 %648, i32 %649), !dbg !27
  %651 = bitcast i32 %650 to float, !dbg !27
  %652 = call float @llvm.maxnum.f32(float %644, float %651), !dbg !27
  %653 = extractvalue [2 x <1 x float>] %209, 0, !dbg !27
  %654 = insertelement <1 x float> poison, float %652, i64 0, !dbg !27
  %655 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %654, <1 x float> %653), !dbg !27
  %656 = call float @llvm.maxnum.f32(float %601, float 0xFFF0000000000000), !dbg !27
  %657 = call float @llvm.maxnum.f32(float %656, float %602), !dbg !27
  %658 = call float @llvm.maxnum.f32(float %657, float %603), !dbg !27
  %659 = call float @llvm.maxnum.f32(float %658, float %604), !dbg !27
  %660 = call float @llvm.maxnum.f32(float %659, float %605), !dbg !27
  %661 = call float @llvm.maxnum.f32(float %660, float %606), !dbg !27
  %662 = call float @llvm.maxnum.f32(float %661, float %607), !dbg !27
  %663 = call float @llvm.maxnum.f32(float %662, float %608), !dbg !27
  %664 = call float @llvm.maxnum.f32(float %663, float %609), !dbg !27
  %665 = call float @llvm.maxnum.f32(float %664, float %610), !dbg !27
  %666 = call float @llvm.maxnum.f32(float %665, float %611), !dbg !27
  %667 = call float @llvm.maxnum.f32(float %666, float %612), !dbg !27
  %668 = call float @llvm.maxnum.f32(float %667, float %613), !dbg !27
  %669 = call float @llvm.maxnum.f32(float %668, float %614), !dbg !27
  %670 = call float @llvm.maxnum.f32(float %669, float %615), !dbg !27
  %671 = call float @llvm.maxnum.f32(float %670, float %616), !dbg !27
  %672 = bitcast float %671 to i32, !dbg !27
  %673 = call i32 @llvm.amdgcn.ds.bpermute(i32 %640, i32 %672), !dbg !27
  %674 = bitcast i32 %673 to float, !dbg !27
  %675 = call float @llvm.maxnum.f32(float %671, float %674), !dbg !27
  %676 = bitcast float %675 to i32, !dbg !27
  %677 = call i32 @llvm.amdgcn.ds.bpermute(i32 %648, i32 %676), !dbg !27
  %678 = bitcast i32 %677 to float, !dbg !27
  %679 = call float @llvm.maxnum.f32(float %675, float %678), !dbg !27
  %680 = extractvalue [2 x <1 x float>] %209, 1, !dbg !27
  %681 = insertelement <1 x float> poison, float %679, i64 0, !dbg !27
  %682 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %681, <1 x float> %680), !dbg !27
  %683 = shufflevector <1 x float> %655, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %684 = fsub <4 x float> %576, %683, !dbg !28
  %685 = fsub <4 x float> %577, %683, !dbg !28
  %686 = fsub <4 x float> %578, %683, !dbg !28
  %687 = fsub <4 x float> %579, %683, !dbg !28
  %688 = shufflevector <1 x float> %682, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %689 = fsub <4 x float> %580, %688, !dbg !28
  %690 = fsub <4 x float> %581, %688, !dbg !28
  %691 = fsub <4 x float> %582, %688, !dbg !28
  %692 = fsub <4 x float> %583, %688, !dbg !28
  %693 = fsub <1 x float> %653, %655, !dbg !29
  %694 = extractelement <1 x float> %693, i64 0, !dbg !30
  %695 = call float @llvm.amdgcn.exp2.f32(float %694), !dbg !30
  %696 = insertelement <4 x float> poison, float %695, i64 0, !dbg !35
  %697 = shufflevector <4 x float> %696, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %698 = extractvalue [2 x <1 x float>] %208, 0, !dbg !31
  %699 = extractelement <1 x float> %698, i64 0, !dbg !31
  %700 = fmul float %699, %695, !dbg !31
  %701 = fsub <1 x float> %680, %682, !dbg !29
  %702 = extractelement <1 x float> %701, i64 0, !dbg !30
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !30
  %704 = insertelement <4 x float> poison, float %703, i64 0, !dbg !35
  %705 = shufflevector <4 x float> %704, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %706 = extractvalue [2 x <1 x float>] %208, 1, !dbg !31
  %707 = extractelement <1 x float> %706, i64 0, !dbg !31
  %708 = fmul float %707, %703, !dbg !31
  %709 = extractelement <4 x float> %684, i64 0, !dbg !32
  %710 = call float @llvm.amdgcn.exp2.f32(float %709), !dbg !32
  %711 = insertelement <4 x float> poison, float %710, i64 0, !dbg !34
  %712 = extractelement <4 x float> %684, i64 1, !dbg !32
  %713 = call float @llvm.amdgcn.exp2.f32(float %712), !dbg !32
  %714 = insertelement <4 x float> %711, float %713, i64 1, !dbg !34
  %715 = fadd float %710, %713, !dbg !33
  %716 = extractelement <4 x float> %684, i64 2, !dbg !32
  %717 = call float @llvm.amdgcn.exp2.f32(float %716), !dbg !32
  %718 = insertelement <4 x float> %714, float %717, i64 2, !dbg !34
  %719 = fadd float %715, %717, !dbg !33
  %720 = extractelement <4 x float> %684, i64 3, !dbg !32
  %721 = call float @llvm.amdgcn.exp2.f32(float %720), !dbg !32
  %722 = insertelement <4 x float> %718, float %721, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %722) #9, !dbg !34, !frisk.register_lifetimes !1
  %723 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %723) #9, !dbg !33, !frisk.register_lifetimes !1
  %724 = fadd float %719, %721, !dbg !33
  %725 = extractelement <4 x float> %685, i64 0, !dbg !32
  %726 = call float @llvm.amdgcn.exp2.f32(float %725), !dbg !32
  %727 = insertelement <4 x float> poison, float %726, i64 0, !dbg !34
  %728 = fadd float %724, %726, !dbg !33
  %729 = extractelement <4 x float> %685, i64 1, !dbg !32
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !32
  %731 = insertelement <4 x float> %727, float %730, i64 1, !dbg !34
  %732 = fadd float %728, %730, !dbg !33
  %733 = extractelement <4 x float> %685, i64 2, !dbg !32
  %734 = call float @llvm.amdgcn.exp2.f32(float %733), !dbg !32
  %735 = insertelement <4 x float> %731, float %734, i64 2, !dbg !34
  %736 = fadd float %732, %734, !dbg !33
  %737 = extractelement <4 x float> %685, i64 3, !dbg !32
  %738 = call float @llvm.amdgcn.exp2.f32(float %737), !dbg !32
  %739 = insertelement <4 x float> %735, float %738, i64 3, !dbg !34
  %math.result179 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %739) #9, !dbg !34, !frisk.register_lifetimes !1
  %740 = fptrunc <4 x float> %math.result179 to <4 x half>, !dbg !34
  %math.packed180 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %740) #9, !dbg !33, !frisk.register_lifetimes !1
  %741 = fadd float %736, %738, !dbg !33
  %742 = extractelement <4 x float> %686, i64 0, !dbg !32
  %743 = call float @llvm.amdgcn.exp2.f32(float %742), !dbg !32
  %744 = insertelement <4 x float> poison, float %743, i64 0, !dbg !34
  %745 = fadd float %741, %743, !dbg !33
  %746 = extractelement <4 x float> %686, i64 1, !dbg !32
  %747 = call float @llvm.amdgcn.exp2.f32(float %746), !dbg !32
  %748 = insertelement <4 x float> %744, float %747, i64 1, !dbg !34
  %749 = fadd float %745, %747, !dbg !33
  %750 = extractelement <4 x float> %686, i64 2, !dbg !32
  %751 = call float @llvm.amdgcn.exp2.f32(float %750), !dbg !32
  %752 = insertelement <4 x float> %748, float %751, i64 2, !dbg !34
  %753 = fadd float %749, %751, !dbg !33
  %754 = extractelement <4 x float> %686, i64 3, !dbg !32
  %755 = call float @llvm.amdgcn.exp2.f32(float %754), !dbg !32
  %756 = insertelement <4 x float> %752, float %755, i64 3, !dbg !34
  %math.result181 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %756) #9, !dbg !34, !frisk.register_lifetimes !1
  %757 = fptrunc <4 x float> %math.result181 to <4 x half>, !dbg !34
  %math.packed182 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %757) #9, !dbg !33, !frisk.register_lifetimes !1
  %758 = fadd float %753, %755, !dbg !33
  %759 = extractelement <4 x float> %687, i64 0, !dbg !32
  %760 = call float @llvm.amdgcn.exp2.f32(float %759), !dbg !32
  %761 = insertelement <4 x float> poison, float %760, i64 0, !dbg !34
  %762 = fadd float %758, %760, !dbg !33
  %763 = extractelement <4 x float> %687, i64 1, !dbg !32
  %764 = call float @llvm.amdgcn.exp2.f32(float %763), !dbg !32
  %765 = insertelement <4 x float> %761, float %764, i64 1, !dbg !34
  %766 = fadd float %762, %764, !dbg !33
  %767 = extractelement <4 x float> %687, i64 2, !dbg !32
  %768 = call float @llvm.amdgcn.exp2.f32(float %767), !dbg !32
  %769 = insertelement <4 x float> %765, float %768, i64 2, !dbg !34
  %770 = fadd float %766, %768, !dbg !33
  %771 = extractelement <4 x float> %687, i64 3, !dbg !32
  %772 = call float @llvm.amdgcn.exp2.f32(float %771), !dbg !32
  %773 = insertelement <4 x float> %769, float %772, i64 3, !dbg !34
  %math.result183 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %773) #9, !dbg !34, !frisk.register_lifetimes !1
  %774 = fptrunc <4 x float> %math.result183 to <4 x half>, !dbg !34
  %math.packed184 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %774) #9, !dbg !33, !frisk.register_lifetimes !1
  %775 = fadd float %770, %772, !dbg !33
  %776 = extractelement <4 x float> %689, i64 0, !dbg !32
  %777 = call float @llvm.amdgcn.exp2.f32(float %776), !dbg !32
  %778 = insertelement <4 x float> poison, float %777, i64 0, !dbg !34
  %779 = extractelement <4 x float> %689, i64 1, !dbg !32
  %780 = call float @llvm.amdgcn.exp2.f32(float %779), !dbg !32
  %781 = insertelement <4 x float> %778, float %780, i64 1, !dbg !34
  %782 = fadd float %777, %780, !dbg !33
  %783 = extractelement <4 x float> %689, i64 2, !dbg !32
  %784 = call float @llvm.amdgcn.exp2.f32(float %783), !dbg !32
  %785 = insertelement <4 x float> %781, float %784, i64 2, !dbg !34
  %786 = fadd float %782, %784, !dbg !33
  %787 = extractelement <4 x float> %689, i64 3, !dbg !32
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !32
  %789 = insertelement <4 x float> %785, float %788, i64 3, !dbg !34
  %math.result185 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %789) #9, !dbg !34, !frisk.register_lifetimes !1
  %790 = fptrunc <4 x float> %math.result185 to <4 x half>, !dbg !34
  %math.packed186 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %790) #9, !dbg !33, !frisk.register_lifetimes !1
  %791 = fadd float %786, %788, !dbg !33
  %792 = extractelement <4 x float> %690, i64 0, !dbg !32
  %793 = call float @llvm.amdgcn.exp2.f32(float %792), !dbg !32
  %794 = insertelement <4 x float> poison, float %793, i64 0, !dbg !34
  %795 = fadd float %791, %793, !dbg !33
  %796 = extractelement <4 x float> %690, i64 1, !dbg !32
  %797 = call float @llvm.amdgcn.exp2.f32(float %796), !dbg !32
  %798 = insertelement <4 x float> %794, float %797, i64 1, !dbg !34
  %799 = fadd float %795, %797, !dbg !33
  %800 = extractelement <4 x float> %690, i64 2, !dbg !32
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !32
  %802 = insertelement <4 x float> %798, float %801, i64 2, !dbg !34
  %803 = fadd float %799, %801, !dbg !33
  %804 = extractelement <4 x float> %690, i64 3, !dbg !32
  %805 = call float @llvm.amdgcn.exp2.f32(float %804), !dbg !32
  %806 = insertelement <4 x float> %802, float %805, i64 3, !dbg !34
  %math.result187 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %806) #9, !dbg !34, !frisk.register_lifetimes !1
  %807 = fptrunc <4 x float> %math.result187 to <4 x half>, !dbg !34
  %math.packed188 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %807) #9, !dbg !33, !frisk.register_lifetimes !1
  %808 = fadd float %803, %805, !dbg !33
  %809 = extractelement <4 x float> %691, i64 0, !dbg !32
  %810 = call float @llvm.amdgcn.exp2.f32(float %809), !dbg !32
  %811 = insertelement <4 x float> poison, float %810, i64 0, !dbg !34
  %812 = fadd float %808, %810, !dbg !33
  %813 = extractelement <4 x float> %691, i64 1, !dbg !32
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !32
  %815 = insertelement <4 x float> %811, float %814, i64 1, !dbg !34
  %816 = fadd float %812, %814, !dbg !33
  %817 = extractelement <4 x float> %691, i64 2, !dbg !32
  %818 = call float @llvm.amdgcn.exp2.f32(float %817), !dbg !32
  %819 = insertelement <4 x float> %815, float %818, i64 2, !dbg !34
  %820 = fadd float %816, %818, !dbg !33
  %821 = extractelement <4 x float> %691, i64 3, !dbg !32
  %822 = call float @llvm.amdgcn.exp2.f32(float %821), !dbg !32
  %823 = insertelement <4 x float> %819, float %822, i64 3, !dbg !34
  %math.result189 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %823) #9, !dbg !34, !frisk.register_lifetimes !1
  %824 = fptrunc <4 x float> %math.result189 to <4 x half>, !dbg !34
  %math.packed190 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %824) #9, !dbg !33, !frisk.register_lifetimes !1
  %825 = fadd float %820, %822, !dbg !33
  %826 = extractelement <4 x float> %692, i64 0, !dbg !32
  %827 = call float @llvm.amdgcn.exp2.f32(float %826), !dbg !32
  %828 = insertelement <4 x float> poison, float %827, i64 0, !dbg !34
  %829 = fadd float %825, %827, !dbg !33
  %830 = extractelement <4 x float> %692, i64 1, !dbg !32
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !32
  %832 = insertelement <4 x float> %828, float %831, i64 1, !dbg !34
  %833 = fadd float %829, %831, !dbg !33
  %834 = extractelement <4 x float> %692, i64 2, !dbg !32
  %835 = call float @llvm.amdgcn.exp2.f32(float %834), !dbg !32
  %836 = insertelement <4 x float> %832, float %835, i64 2, !dbg !34
  %837 = fadd float %833, %835, !dbg !33
  %838 = extractelement <4 x float> %692, i64 3, !dbg !32
  %839 = call float @llvm.amdgcn.exp2.f32(float %838), !dbg !32
  %840 = insertelement <4 x float> %836, float %839, i64 3, !dbg !34
  %math.result191 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %840) #9, !dbg !34, !frisk.register_lifetimes !1
  %841 = fptrunc <4 x float> %math.result191 to <4 x half>, !dbg !34
  %math.packed192 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %841) #9, !dbg !33, !frisk.register_lifetimes !1
  %842 = fadd float %837, %839, !dbg !33
  %843 = bitcast float %775 to i32, !dbg !33
  %844 = call i32 @llvm.amdgcn.ds.bpermute(i32 %640, i32 %843), !dbg !33
  %845 = bitcast i32 %844 to float, !dbg !33
  %846 = fadd float %775, %845, !dbg !33
  %847 = bitcast float %846 to i32, !dbg !33
  %848 = call i32 @llvm.amdgcn.ds.bpermute(i32 %648, i32 %847), !dbg !33
  %849 = bitcast i32 %848 to float, !dbg !33
  %850 = fadd float %846, %849, !dbg !33
  %851 = fadd float %850, %700, !dbg !33
  %852 = insertelement <1 x float> poison, float %851, i64 0, !dbg !7
  %853 = insertvalue [2 x <1 x float>] poison, <1 x float> %852, 0, !dbg !7
  %854 = bitcast float %842 to i32, !dbg !33
  %855 = call i32 @llvm.amdgcn.ds.bpermute(i32 %640, i32 %854), !dbg !33
  %856 = bitcast i32 %855 to float, !dbg !33
  %857 = fadd float %842, %856, !dbg !33
  %858 = bitcast float %857 to i32, !dbg !33
  %859 = call i32 @llvm.amdgcn.ds.bpermute(i32 %648, i32 %858), !dbg !33
  %860 = bitcast i32 %859 to float, !dbg !33
  %861 = fadd float %857, %860, !dbg !33
  %862 = fadd float %861, %708, !dbg !33
  %863 = insertelement <1 x float> poison, float %862, i64 0, !dbg !7
  %864 = insertvalue [2 x <1 x float>] %853, <1 x float> %863, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread513 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local514 = zext nneg i32 %address.thread513 to i64, !dbg !9
  %address.local515 = and i64 %address.local514, 60, !dbg !15
  %address.local516 = add nuw nsw i64 %address.local515, %address.local514, !dbg !15
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !15
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local517, !dbg !15
  %address.local519 = and i64 %address.local514, 63, !dbg !8
  %address.local520 = lshr i64 %address.local519, 4, !dbg !15
  %address.local521 = mul nuw nsw i64 %address.local520, 384, !dbg !15
  %address.local522 = getelementptr [2 x i8], ptr addrspace(3) %address.local518, i64 %address.local521, !dbg !15
  %address.local523 = lshr i64 %address.local514, 2, !dbg !17
  %address.local524 = mul nsw i64 %address.local523, -32, !dbg !15
  %address.local525 = getelementptr [2 x i8], ptr addrspace(3) %address.local522, i64 %address.local524, !dbg !15
  %865 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local525, i32 0) #0, !dbg !15
  %866 = bitcast <4 x i32> %865 to <8 x half>, !dbg !15
  %address.local526 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local517, !dbg !15
  %address.local527 = getelementptr [2 x i8], ptr addrspace(3) %address.local526, i64 %address.local521, !dbg !15
  %address.local528 = getelementptr [2 x i8], ptr addrspace(3) %address.local527, i64 %address.local524, !dbg !15
  %address.local529 = getelementptr i8, ptr addrspace(3) %address.local528, i64 256, !dbg !15
  %867 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local529, i32 0) #0, !dbg !15
  %868 = bitcast <4 x i32> %867 to <8 x half>, !dbg !15
  %address.local530 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local517, !dbg !15
  %address.local531 = getelementptr [2 x i8], ptr addrspace(3) %address.local530, i64 %address.local521, !dbg !15
  %address.local532 = getelementptr [2 x i8], ptr addrspace(3) %address.local531, i64 %address.local524, !dbg !15
  %address.local533 = getelementptr i8, ptr addrspace(3) %address.local532, i64 512, !dbg !15
  %869 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local533, i32 0) #0, !dbg !15
  %870 = bitcast <4 x i32> %869 to <8 x half>, !dbg !15
  %address.local534 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local517, !dbg !15
  %address.local535 = getelementptr [2 x i8], ptr addrspace(3) %address.local534, i64 %address.local521, !dbg !15
  %address.local536 = getelementptr [2 x i8], ptr addrspace(3) %address.local535, i64 %address.local524, !dbg !15
  %address.local537 = getelementptr i8, ptr addrspace(3) %address.local536, i64 768, !dbg !15
  %871 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local537, i32 0) #0, !dbg !15
  %872 = bitcast <4 x i32> %871 to <8 x half>, !dbg !15
  %873 = add nuw nsw i64 %218, %16, !dbg !18
  %874 = shl i64 %873, 19, !dbg !18
  %875 = and i64 %874, 16252928, !dbg !18
  %876 = add i64 %875, %224, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %877 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %876, !dbg !18
  %878 = getelementptr i8, ptr addrspace(1) %877, i64 64, !dbg !18
  %address.local538 = shl nuw nsw i64 %address.local514, 3, !dbg !9
  %address.local539 = and i64 %address.local538, 1536, !dbg !16
  %address.local540 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local539, !dbg !16
  %address.local541 = ptrtoint ptr addrspace(3) %address.local540 to i64, !dbg !16
  %address.local542 = trunc i64 %address.local541 to i32, !dbg !16
  %879 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local542), !dbg !18
  %880 = zext i32 %879 to i64, !dbg !18
  %881 = inttoptr i64 %880 to ptr addrspace(3), !dbg !18
  %address.local543 = shl nuw nsw i64 %address.local514, 5, !dbg !18
  %address.local544 = and i64 %address.local543, 8064, !dbg !18
  %address.local545 = shl nuw nsw i64 %address.local514, 3, !dbg !18
  %address.local546 = and i64 %address.local545, 24, !dbg !18
  %address.local547 = or disjoint i64 %address.local544, %address.local546, !dbg !18
  %address.local548 = getelementptr [2 x i8], ptr addrspace(1) %878, i64 %address.local547, !dbg !18
  %882 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %883 = ptrtoint ptr addrspace(1) %address.local548 to i64, !dbg !18
  %884 = sub i64 %883, %882, !dbg !18
  %dma.buffer.offset974 = trunc i64 %884 to i32, !dbg !18
  %885 = trunc i64 %882 to i32, !dbg !18
  %886 = insertelement <4 x i32> poison, i32 %885, i64 0, !dbg !18
  %887 = lshr i64 %882, 32, !dbg !18
  %888 = trunc i64 %887 to i32, !dbg !18
  %889 = insertelement <4 x i32> %886, i32 %888, i64 1, !dbg !18
  %890 = insertelement <4 x i32> %889, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc975 = insertelement <4 x i32> %890, i32 131072, i64 3, !dbg !18
  %891 = ptrtoint ptr addrspace(3) %881 to i32, !dbg !18
  %dma.buffer.lds976 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %891), !dbg !18
  %892 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds976, i32 %dma.buffer.offset974, <4 x i32> %dma.buffer.rsrc975) #10, !dbg !18
  %893 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %894 = extractvalue [2 x <8 x float>] %207, 0, !dbg !35
  %895 = shufflevector <8 x float> %894, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %896 = fmul <4 x float> %895, %697, !dbg !35
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %893, <4 x float> %896) #9, !dbg !15
  %898 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed180, <4 x half> %898, <4 x float> %897) #9, !dbg !15
  %900 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed182, <4 x half> %900, <4 x float> %899) #9, !dbg !15
  %902 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed184, <4 x half> %902, <4 x float> %901) #9, !dbg !15
  %904 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %905 = shufflevector <8 x float> %894, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %906 = fmul <4 x float> %905, %697, !dbg !35
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %904, <4 x float> %906) #9, !dbg !15
  %908 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed180, <4 x half> %908, <4 x float> %907) #9, !dbg !15
  %910 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed182, <4 x half> %910, <4 x float> %909) #9, !dbg !15
  %912 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed184, <4 x half> %912, <4 x float> %911) #9, !dbg !15
  %914 = extractvalue [2 x <8 x float>] %207, 1, !dbg !35
  %915 = shufflevector <8 x float> %914, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %916 = fmul <4 x float> %915, %705, !dbg !35
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed186, <4 x half> %893, <4 x float> %916) #9, !dbg !15
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed188, <4 x half> %898, <4 x float> %917) #9, !dbg !15
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed190, <4 x half> %900, <4 x float> %918) #9, !dbg !15
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed192, <4 x half> %902, <4 x float> %919) #9, !dbg !15
  %921 = shufflevector <8 x float> %914, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %922 = fmul <4 x float> %921, %705, !dbg !35
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed186, <4 x half> %904, <4 x float> %922) #9, !dbg !15
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed188, <4 x half> %908, <4 x float> %923) #9, !dbg !15
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed190, <4 x half> %910, <4 x float> %924) #9, !dbg !15
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed192, <4 x half> %912, <4 x float> %925) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %927 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local525, i32 0) #0, !dbg !15
  %928 = bitcast <4 x i32> %927 to <8 x half>, !dbg !15
  %929 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local529, i32 0) #0, !dbg !15
  %930 = bitcast <4 x i32> %929 to <8 x half>, !dbg !15
  %931 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local533, i32 0) #0, !dbg !15
  %932 = bitcast <4 x i32> %931 to <8 x half>, !dbg !15
  %933 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local537, i32 0) #0, !dbg !15
  %934 = bitcast <4 x i32> %933 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %935 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %876, !dbg !18
  %936 = getelementptr i8, ptr addrspace(1) %935, i64 128, !dbg !18
  %937 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local542), !dbg !18
  %938 = zext i32 %937 to i64, !dbg !18
  %939 = inttoptr i64 %938 to ptr addrspace(3), !dbg !18
  %address.local549 = getelementptr [2 x i8], ptr addrspace(1) %936, i64 %address.local547, !dbg !18
  %940 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %941 = ptrtoint ptr addrspace(1) %address.local549 to i64, !dbg !18
  %942 = sub i64 %941, %940, !dbg !18
  %dma.buffer.offset971 = trunc i64 %942 to i32, !dbg !18
  %943 = trunc i64 %940 to i32, !dbg !18
  %944 = insertelement <4 x i32> poison, i32 %943, i64 0, !dbg !18
  %945 = lshr i64 %940, 32, !dbg !18
  %946 = trunc i64 %945 to i32, !dbg !18
  %947 = insertelement <4 x i32> %944, i32 %946, i64 1, !dbg !18
  %948 = insertelement <4 x i32> %947, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc972 = insertelement <4 x i32> %948, i32 131072, i64 3, !dbg !18
  %949 = ptrtoint ptr addrspace(3) %939 to i32, !dbg !18
  %dma.buffer.lds973 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %949), !dbg !18
  %950 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds973, i32 %dma.buffer.offset971, <4 x i32> %dma.buffer.rsrc972) #10, !dbg !18
  %951 = shufflevector <8 x half> %928, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %952 = extractvalue [2 x <8 x float>] %210, 0, !dbg !35
  %953 = shufflevector <8 x float> %952, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %954 = fmul <4 x float> %953, %697, !dbg !35
  %955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %951, <4 x float> %954) #9, !dbg !15
  %956 = shufflevector <8 x half> %930, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %957 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed180, <4 x half> %956, <4 x float> %955) #9, !dbg !15
  %958 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed182, <4 x half> %958, <4 x float> %957) #9, !dbg !15
  %960 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed184, <4 x half> %960, <4 x float> %959) #9, !dbg !15
  %962 = shufflevector <8 x half> %928, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %963 = shufflevector <8 x float> %952, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %964 = fmul <4 x float> %963, %697, !dbg !35
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %962, <4 x float> %964) #9, !dbg !15
  %966 = shufflevector <8 x half> %930, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed180, <4 x half> %966, <4 x float> %965) #9, !dbg !15
  %968 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed182, <4 x half> %968, <4 x float> %967) #9, !dbg !15
  %970 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed184, <4 x half> %970, <4 x float> %969) #9, !dbg !15
  %972 = extractvalue [2 x <8 x float>] %210, 1, !dbg !35
  %973 = shufflevector <8 x float> %972, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %974 = fmul <4 x float> %973, %705, !dbg !35
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed186, <4 x half> %951, <4 x float> %974) #9, !dbg !15
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed188, <4 x half> %956, <4 x float> %975) #9, !dbg !15
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed190, <4 x half> %958, <4 x float> %976) #9, !dbg !15
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed192, <4 x half> %960, <4 x float> %977) #9, !dbg !15
  %979 = shufflevector <8 x float> %972, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %980 = fmul <4 x float> %979, %705, !dbg !35
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed186, <4 x half> %962, <4 x float> %980) #9, !dbg !15
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed188, <4 x half> %966, <4 x float> %981) #9, !dbg !15
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed190, <4 x half> %968, <4 x float> %982) #9, !dbg !15
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed192, <4 x half> %970, <4 x float> %983) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %985 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local525, i32 0) #0, !dbg !15
  %986 = bitcast <4 x i32> %985 to <8 x half>, !dbg !15
  %987 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local529, i32 0) #0, !dbg !15
  %988 = bitcast <4 x i32> %987 to <8 x half>, !dbg !15
  %989 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local533, i32 0) #0, !dbg !15
  %990 = bitcast <4 x i32> %989 to <8 x half>, !dbg !15
  %991 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local537, i32 0) #0, !dbg !15
  %992 = bitcast <4 x i32> %991 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %993 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %876, !dbg !18
  %994 = getelementptr i8, ptr addrspace(1) %993, i64 192, !dbg !18
  %995 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local542), !dbg !18
  %996 = zext i32 %995 to i64, !dbg !18
  %997 = inttoptr i64 %996 to ptr addrspace(3), !dbg !18
  %address.local550 = getelementptr [2 x i8], ptr addrspace(1) %994, i64 %address.local547, !dbg !18
  %998 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %999 = ptrtoint ptr addrspace(1) %address.local550 to i64, !dbg !18
  %1000 = sub i64 %999, %998, !dbg !18
  %dma.buffer.offset = trunc i64 %1000 to i32, !dbg !18
  %1001 = trunc i64 %998 to i32, !dbg !18
  %1002 = insertelement <4 x i32> poison, i32 %1001, i64 0, !dbg !18
  %1003 = lshr i64 %998, 32, !dbg !18
  %1004 = trunc i64 %1003 to i32, !dbg !18
  %1005 = insertelement <4 x i32> %1002, i32 %1004, i64 1, !dbg !18
  %1006 = insertelement <4 x i32> %1005, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1006, i32 131072, i64 3, !dbg !18
  %1007 = ptrtoint ptr addrspace(3) %997 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1007), !dbg !18
  %1008 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1009 = shufflevector <8 x half> %986, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1010 = extractvalue [2 x <8 x float>] %211, 0, !dbg !35
  %1011 = shufflevector <8 x float> %1010, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1012 = fmul <4 x float> %1011, %697, !dbg !35
  %1013 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1009, <4 x float> %1012) #9, !dbg !15
  %1014 = shufflevector <8 x half> %988, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed180, <4 x half> %1014, <4 x float> %1013) #9, !dbg !15
  %1016 = shufflevector <8 x half> %990, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1017 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed182, <4 x half> %1016, <4 x float> %1015) #9, !dbg !15
  %1018 = shufflevector <8 x half> %992, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1019 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed184, <4 x half> %1018, <4 x float> %1017) #9, !dbg !15
  %1020 = shufflevector <8 x half> %986, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1021 = shufflevector <8 x float> %1010, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1022 = fmul <4 x float> %1021, %697, !dbg !35
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1020, <4 x float> %1022) #9, !dbg !15
  %1024 = shufflevector <8 x half> %988, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed180, <4 x half> %1024, <4 x float> %1023) #9, !dbg !15
  %1026 = shufflevector <8 x half> %990, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed182, <4 x half> %1026, <4 x float> %1025) #9, !dbg !15
  %1028 = shufflevector <8 x half> %992, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed184, <4 x half> %1028, <4 x float> %1027) #9, !dbg !15
  %1030 = extractvalue [2 x <8 x float>] %211, 1, !dbg !35
  %1031 = shufflevector <8 x float> %1030, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1032 = fmul <4 x float> %1031, %705, !dbg !35
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed186, <4 x half> %1009, <4 x float> %1032) #9, !dbg !15
  %1034 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed188, <4 x half> %1014, <4 x float> %1033) #9, !dbg !15
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed190, <4 x half> %1016, <4 x float> %1034) #9, !dbg !15
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed192, <4 x half> %1018, <4 x float> %1035) #9, !dbg !15
  %1037 = shufflevector <8 x float> %1030, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1038 = fmul <4 x float> %1037, %705, !dbg !35
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed186, <4 x half> %1020, <4 x float> %1038) #9, !dbg !15
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed188, <4 x half> %1024, <4 x float> %1039) #9, !dbg !15
  %1041 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed190, <4 x half> %1026, <4 x float> %1040) #9, !dbg !15
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed192, <4 x half> %1028, <4 x float> %1041) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1043 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local525, i32 0) #0, !dbg !15
  %1044 = bitcast <4 x i32> %1043 to <8 x half>, !dbg !15
  %1045 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local529, i32 0) #0, !dbg !15
  %1046 = bitcast <4 x i32> %1045 to <8 x half>, !dbg !15
  %1047 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local533, i32 0) #0, !dbg !15
  %1048 = bitcast <4 x i32> %1047 to <8 x half>, !dbg !15
  %1049 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local537, i32 0) #0, !dbg !15
  %1050 = bitcast <4 x i32> %1049 to <8 x half>, !dbg !15
  %1051 = shufflevector <8 x half> %1044, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1052 = extractvalue [2 x <8 x float>] %212, 0, !dbg !35
  %1053 = shufflevector <8 x float> %1052, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1054 = fmul <4 x float> %1053, %697, !dbg !35
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1051, <4 x float> %1054) #9, !dbg !15
  %1056 = shufflevector <8 x half> %1046, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed180, <4 x half> %1056, <4 x float> %1055) #9, !dbg !15
  %1058 = shufflevector <8 x half> %1048, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed182, <4 x half> %1058, <4 x float> %1057) #9, !dbg !15
  %1060 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed184, <4 x half> %1060, <4 x float> %1059) #9, !dbg !15
  %1062 = shufflevector <8 x half> %1044, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1063 = shufflevector <8 x float> %1052, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1064 = fmul <4 x float> %1063, %697, !dbg !35
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1062, <4 x float> %1064) #9, !dbg !15
  %1066 = shufflevector <8 x half> %1046, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed180, <4 x half> %1066, <4 x float> %1065) #9, !dbg !15
  %1068 = shufflevector <8 x half> %1048, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed182, <4 x half> %1068, <4 x float> %1067) #9, !dbg !15
  %1070 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1071 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed184, <4 x half> %1070, <4 x float> %1069) #9, !dbg !15
  %1072 = extractvalue [2 x <8 x float>] %212, 1, !dbg !35
  %1073 = shufflevector <8 x float> %1072, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1074 = fmul <4 x float> %1073, %705, !dbg !35
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed186, <4 x half> %1051, <4 x float> %1074) #9, !dbg !15
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed188, <4 x half> %1056, <4 x float> %1075) #9, !dbg !15
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed190, <4 x half> %1058, <4 x float> %1076) #9, !dbg !15
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed192, <4 x half> %1060, <4 x float> %1077) #9, !dbg !15
  %1079 = shufflevector <8 x float> %1072, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1080 = fmul <4 x float> %1079, %705, !dbg !35
  %1081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed186, <4 x half> %1062, <4 x float> %1080) #9, !dbg !15
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed188, <4 x half> %1066, <4 x float> %1081) #9, !dbg !15
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed190, <4 x half> %1068, <4 x float> %1082) #9, !dbg !15
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed192, <4 x half> %1070, <4 x float> %1083) #9, !dbg !15
  %1085 = shufflevector <4 x float> %903, <4 x float> %913, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1086 = insertvalue [2 x <8 x float>] poison, <8 x float> %1085, 0, !dbg !7
  %1087 = shufflevector <4 x float> %920, <4 x float> %926, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1088 = insertvalue [2 x <8 x float>] %1086, <8 x float> %1087, 1, !dbg !7
  %1089 = insertvalue [2 x <1 x float>] poison, <1 x float> %655, 0, !dbg !7
  %1090 = insertvalue [2 x <1 x float>] %1089, <1 x float> %682, 1, !dbg !7
  %1091 = shufflevector <4 x float> %961, <4 x float> %971, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1092 = insertvalue [2 x <8 x float>] poison, <8 x float> %1091, 0, !dbg !7
  %1093 = shufflevector <4 x float> %978, <4 x float> %984, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1094 = insertvalue [2 x <8 x float>] %1092, <8 x float> %1093, 1, !dbg !7
  %1095 = shufflevector <4 x float> %1019, <4 x float> %1029, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1096 = insertvalue [2 x <8 x float>] poison, <8 x float> %1095, 0, !dbg !7
  %1097 = shufflevector <4 x float> %1036, <4 x float> %1042, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1098 = insertvalue [2 x <8 x float>] %1096, <8 x float> %1097, 1, !dbg !7
  %1099 = shufflevector <4 x float> %1061, <4 x float> %1071, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1100 = insertvalue [2 x <8 x float>] poison, <8 x float> %1099, 0, !dbg !7
  %1101 = shufflevector <4 x float> %1078, <4 x float> %1084, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1102 = insertvalue [2 x <8 x float>] %1100, <8 x float> %1101, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1103 = add nuw nsw i64 %206, 1, !dbg !7
  br label %205, !dbg !7

1104:                                             ; preds = %205
  %1105 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %1105, 0, !dbg !7
  %1106 = and i32 %31, 32, !dbg !7
  %.not18 = icmp eq i32 %1106, 0, !dbg !7
  %1107 = extractvalue [2 x <8 x float>] %212, 0, !dbg !7
  %1108 = extractelement <8 x float> %1107, i64 0, !dbg !7
  %1109 = extractelement <8 x float> %1107, i64 1, !dbg !7
  %1110 = extractelement <8 x float> %1107, i64 2, !dbg !7
  %1111 = extractelement <8 x float> %1107, i64 3, !dbg !7
  %1112 = select i1 %.not, float %1109, float %1108, !dbg !7
  %1113 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1114 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1113), !dbg !7
  %1115 = add nuw nsw i32 %1114, 64, !dbg !7
  %1116 = xor i32 %1114, 16, !dbg !7
  %1117 = and i32 %1115, 192, !dbg !7
  %1118 = icmp samesign ult i32 %1116, %1117, !dbg !7
  %1119 = select i1 %1118, i32 %1116, i32 %1114, !dbg !7
  %1120 = shl nuw nsw i32 %1119, 2, !dbg !7
  %1121 = bitcast float %1112 to i32, !dbg !7
  %1122 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1121), !dbg !7
  %1123 = bitcast i32 %1122 to float, !dbg !7
  %1124 = select i1 %.not, float %1108, float %1123, !dbg !7
  %1125 = select i1 %.not, float %1123, float %1109, !dbg !7
  %1126 = select i1 %.not, float %1111, float %1110, !dbg !7
  %1127 = bitcast float %1126 to i32, !dbg !7
  %1128 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1127), !dbg !7
  %1129 = bitcast i32 %1128 to float, !dbg !7
  %1130 = select i1 %.not, float %1110, float %1129, !dbg !7
  %1131 = select i1 %.not, float %1129, float %1111, !dbg !7
  %1132 = select i1 %.not18, float %1130, float %1124, !dbg !7
  %1133 = xor i32 %1114, 32, !dbg !7
  %1134 = icmp samesign ult i32 %1133, %1117, !dbg !7
  %1135 = select i1 %1134, i32 %1133, i32 %1114, !dbg !7
  %1136 = shl nuw nsw i32 %1135, 2, !dbg !7
  %1137 = bitcast float %1132 to i32, !dbg !7
  %1138 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1137), !dbg !7
  %1139 = bitcast i32 %1138 to float, !dbg !7
  %1140 = select i1 %.not18, float %1124, float %1139, !dbg !7
  %1141 = select i1 %.not18, float %1139, float %1130, !dbg !7
  %1142 = select i1 %.not18, float %1131, float %1125, !dbg !7
  %1143 = bitcast float %1142 to i32, !dbg !7
  %1144 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1143), !dbg !7
  %1145 = bitcast i32 %1144 to float, !dbg !7
  %1146 = select i1 %.not18, float %1125, float %1145, !dbg !7
  %1147 = select i1 %.not18, float %1145, float %1131, !dbg !7
  %1148 = extractvalue [2 x <8 x float>] %211, 0, !dbg !7
  %1149 = extractelement <8 x float> %1148, i64 0, !dbg !7
  %1150 = extractelement <8 x float> %1148, i64 1, !dbg !7
  %1151 = extractelement <8 x float> %1148, i64 2, !dbg !7
  %1152 = extractelement <8 x float> %1148, i64 3, !dbg !7
  %1153 = select i1 %.not, float %1150, float %1149, !dbg !7
  %1154 = bitcast float %1153 to i32, !dbg !7
  %1155 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1154), !dbg !7
  %1156 = bitcast i32 %1155 to float, !dbg !7
  %1157 = select i1 %.not, float %1149, float %1156, !dbg !7
  %1158 = select i1 %.not, float %1156, float %1150, !dbg !7
  %1159 = select i1 %.not, float %1152, float %1151, !dbg !7
  %1160 = bitcast float %1159 to i32, !dbg !7
  %1161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1160), !dbg !7
  %1162 = bitcast i32 %1161 to float, !dbg !7
  %1163 = select i1 %.not, float %1151, float %1162, !dbg !7
  %1164 = select i1 %.not, float %1162, float %1152, !dbg !7
  %1165 = select i1 %.not18, float %1163, float %1157, !dbg !7
  %1166 = bitcast float %1165 to i32, !dbg !7
  %1167 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1166), !dbg !7
  %1168 = bitcast i32 %1167 to float, !dbg !7
  %1169 = select i1 %.not18, float %1157, float %1168, !dbg !7
  %1170 = select i1 %.not18, float %1168, float %1163, !dbg !7
  %1171 = select i1 %.not18, float %1164, float %1158, !dbg !7
  %1172 = bitcast float %1171 to i32, !dbg !7
  %1173 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1172), !dbg !7
  %1174 = bitcast i32 %1173 to float, !dbg !7
  %1175 = select i1 %.not18, float %1158, float %1174, !dbg !7
  %1176 = select i1 %.not18, float %1174, float %1164, !dbg !7
  %1177 = extractvalue [2 x <8 x float>] %210, 0, !dbg !7
  %1178 = extractelement <8 x float> %1177, i64 0, !dbg !7
  %1179 = extractelement <8 x float> %1177, i64 1, !dbg !7
  %1180 = extractelement <8 x float> %1177, i64 2, !dbg !7
  %1181 = extractelement <8 x float> %1177, i64 3, !dbg !7
  %1182 = select i1 %.not, float %1179, float %1178, !dbg !7
  %1183 = bitcast float %1182 to i32, !dbg !7
  %1184 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1183), !dbg !7
  %1185 = bitcast i32 %1184 to float, !dbg !7
  %1186 = select i1 %.not, float %1178, float %1185, !dbg !7
  %1187 = select i1 %.not, float %1185, float %1179, !dbg !7
  %1188 = select i1 %.not, float %1181, float %1180, !dbg !7
  %1189 = bitcast float %1188 to i32, !dbg !7
  %1190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1189), !dbg !7
  %1191 = bitcast i32 %1190 to float, !dbg !7
  %1192 = select i1 %.not, float %1180, float %1191, !dbg !7
  %1193 = select i1 %.not, float %1191, float %1181, !dbg !7
  %1194 = select i1 %.not18, float %1192, float %1186, !dbg !7
  %1195 = bitcast float %1194 to i32, !dbg !7
  %1196 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1195), !dbg !7
  %1197 = bitcast i32 %1196 to float, !dbg !7
  %1198 = select i1 %.not18, float %1186, float %1197, !dbg !7
  %1199 = select i1 %.not18, float %1197, float %1192, !dbg !7
  %1200 = select i1 %.not18, float %1193, float %1187, !dbg !7
  %1201 = bitcast float %1200 to i32, !dbg !7
  %1202 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1201), !dbg !7
  %1203 = bitcast i32 %1202 to float, !dbg !7
  %1204 = select i1 %.not18, float %1187, float %1203, !dbg !7
  %1205 = select i1 %.not18, float %1203, float %1193, !dbg !7
  %1206 = extractvalue [2 x <8 x float>] %207, 0, !dbg !7
  %1207 = extractelement <8 x float> %1206, i64 0, !dbg !7
  %1208 = extractelement <8 x float> %1206, i64 1, !dbg !7
  %1209 = extractelement <8 x float> %1206, i64 2, !dbg !7
  %1210 = extractelement <8 x float> %1206, i64 3, !dbg !7
  %1211 = select i1 %.not, float %1208, float %1207, !dbg !7
  %1212 = bitcast float %1211 to i32, !dbg !7
  %1213 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1212), !dbg !7
  %1214 = bitcast i32 %1213 to float, !dbg !7
  %1215 = select i1 %.not, float %1207, float %1214, !dbg !7
  %1216 = select i1 %.not, float %1214, float %1208, !dbg !7
  %1217 = select i1 %.not, float %1210, float %1209, !dbg !7
  %1218 = bitcast float %1217 to i32, !dbg !7
  %1219 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1218), !dbg !7
  %1220 = bitcast i32 %1219 to float, !dbg !7
  %1221 = select i1 %.not, float %1209, float %1220, !dbg !7
  %1222 = select i1 %.not, float %1220, float %1210, !dbg !7
  %1223 = select i1 %.not18, float %1221, float %1215, !dbg !7
  %1224 = bitcast float %1223 to i32, !dbg !7
  %1225 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1224), !dbg !7
  %1226 = select i1 %.not18, float %1222, float %1216, !dbg !7
  %1227 = bitcast float %1226 to i32, !dbg !7
  %1228 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1227), !dbg !7
  %1229 = extractvalue [2 x <1 x float>] %208, 0, !dbg !19
  %1230 = shufflevector <1 x float> %1229, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1231 = extractelement <8 x float> %1107, i64 4, !dbg !7
  %1232 = extractelement <8 x float> %1107, i64 5, !dbg !7
  %1233 = extractelement <8 x float> %1107, i64 6, !dbg !7
  %1234 = extractelement <8 x float> %1107, i64 7, !dbg !7
  %1235 = select i1 %.not, float %1232, float %1231, !dbg !7
  %1236 = bitcast float %1235 to i32, !dbg !7
  %1237 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1236), !dbg !7
  %1238 = bitcast i32 %1237 to float, !dbg !7
  %1239 = select i1 %.not, float %1231, float %1238, !dbg !7
  %1240 = select i1 %.not, float %1238, float %1232, !dbg !7
  %1241 = select i1 %.not, float %1234, float %1233, !dbg !7
  %1242 = bitcast float %1241 to i32, !dbg !7
  %1243 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1242), !dbg !7
  %1244 = bitcast i32 %1243 to float, !dbg !7
  %1245 = select i1 %.not, float %1233, float %1244, !dbg !7
  %1246 = select i1 %.not, float %1244, float %1234, !dbg !7
  %1247 = select i1 %.not18, float %1245, float %1239, !dbg !7
  %1248 = bitcast float %1247 to i32, !dbg !7
  %1249 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1248), !dbg !7
  %1250 = bitcast i32 %1249 to float, !dbg !7
  %1251 = select i1 %.not18, float %1239, float %1250, !dbg !7
  %1252 = select i1 %.not18, float %1250, float %1245, !dbg !7
  %1253 = select i1 %.not18, float %1246, float %1240, !dbg !7
  %1254 = bitcast float %1253 to i32, !dbg !7
  %1255 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1254), !dbg !7
  %1256 = bitcast i32 %1255 to float, !dbg !7
  %1257 = select i1 %.not18, float %1240, float %1256, !dbg !7
  %1258 = select i1 %.not18, float %1256, float %1246, !dbg !7
  %1259 = extractelement <8 x float> %1148, i64 4, !dbg !7
  %1260 = extractelement <8 x float> %1148, i64 5, !dbg !7
  %1261 = extractelement <8 x float> %1148, i64 6, !dbg !7
  %1262 = extractelement <8 x float> %1148, i64 7, !dbg !7
  %1263 = select i1 %.not, float %1260, float %1259, !dbg !7
  %1264 = bitcast float %1263 to i32, !dbg !7
  %1265 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1264), !dbg !7
  %1266 = bitcast i32 %1265 to float, !dbg !7
  %1267 = select i1 %.not, float %1259, float %1266, !dbg !7
  %1268 = select i1 %.not, float %1266, float %1260, !dbg !7
  %1269 = select i1 %.not, float %1262, float %1261, !dbg !7
  %1270 = bitcast float %1269 to i32, !dbg !7
  %1271 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1270), !dbg !7
  %1272 = bitcast i32 %1271 to float, !dbg !7
  %1273 = select i1 %.not, float %1261, float %1272, !dbg !7
  %1274 = select i1 %.not, float %1272, float %1262, !dbg !7
  %1275 = select i1 %.not18, float %1273, float %1267, !dbg !7
  %1276 = bitcast float %1275 to i32, !dbg !7
  %1277 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1276), !dbg !7
  %1278 = bitcast i32 %1277 to float, !dbg !7
  %1279 = select i1 %.not18, float %1267, float %1278, !dbg !7
  %1280 = select i1 %.not18, float %1278, float %1273, !dbg !7
  %1281 = select i1 %.not18, float %1274, float %1268, !dbg !7
  %1282 = bitcast float %1281 to i32, !dbg !7
  %1283 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1282), !dbg !7
  %1284 = bitcast i32 %1283 to float, !dbg !7
  %1285 = select i1 %.not18, float %1268, float %1284, !dbg !7
  %1286 = select i1 %.not18, float %1284, float %1274, !dbg !7
  %1287 = extractelement <8 x float> %1177, i64 4, !dbg !7
  %1288 = extractelement <8 x float> %1177, i64 5, !dbg !7
  %1289 = extractelement <8 x float> %1177, i64 6, !dbg !7
  %1290 = extractelement <8 x float> %1177, i64 7, !dbg !7
  %1291 = select i1 %.not, float %1288, float %1287, !dbg !7
  %1292 = bitcast float %1291 to i32, !dbg !7
  %1293 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1292), !dbg !7
  %1294 = bitcast i32 %1293 to float, !dbg !7
  %1295 = select i1 %.not, float %1287, float %1294, !dbg !7
  %1296 = select i1 %.not, float %1294, float %1288, !dbg !7
  %1297 = select i1 %.not, float %1290, float %1289, !dbg !7
  %1298 = bitcast float %1297 to i32, !dbg !7
  %1299 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1298), !dbg !7
  %1300 = bitcast i32 %1299 to float, !dbg !7
  %1301 = select i1 %.not, float %1289, float %1300, !dbg !7
  %1302 = select i1 %.not, float %1300, float %1290, !dbg !7
  %1303 = select i1 %.not18, float %1301, float %1295, !dbg !7
  %1304 = bitcast float %1303 to i32, !dbg !7
  %1305 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1304), !dbg !7
  %1306 = bitcast i32 %1305 to float, !dbg !7
  %1307 = select i1 %.not18, float %1295, float %1306, !dbg !7
  %1308 = select i1 %.not18, float %1306, float %1301, !dbg !7
  %1309 = select i1 %.not18, float %1302, float %1296, !dbg !7
  %1310 = bitcast float %1309 to i32, !dbg !7
  %1311 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1310), !dbg !7
  %1312 = bitcast i32 %1311 to float, !dbg !7
  %1313 = select i1 %.not18, float %1296, float %1312, !dbg !7
  %1314 = select i1 %.not18, float %1312, float %1302, !dbg !7
  %1315 = extractelement <8 x float> %1206, i64 4, !dbg !7
  %1316 = extractelement <8 x float> %1206, i64 5, !dbg !7
  %1317 = extractelement <8 x float> %1206, i64 6, !dbg !7
  %1318 = extractelement <8 x float> %1206, i64 7, !dbg !7
  %1319 = select i1 %.not, float %1316, float %1315, !dbg !7
  %1320 = bitcast float %1319 to i32, !dbg !7
  %1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1320), !dbg !7
  %1322 = bitcast i32 %1321 to float, !dbg !7
  %1323 = select i1 %.not, float %1315, float %1322, !dbg !7
  %1324 = select i1 %.not, float %1322, float %1316, !dbg !7
  %1325 = select i1 %.not, float %1318, float %1317, !dbg !7
  %1326 = bitcast float %1325 to i32, !dbg !7
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1326), !dbg !7
  %1328 = bitcast i32 %1327 to float, !dbg !7
  %1329 = select i1 %.not, float %1317, float %1328, !dbg !7
  %1330 = select i1 %.not, float %1328, float %1318, !dbg !7
  %1331 = select i1 %.not18, float %1329, float %1323, !dbg !7
  %1332 = bitcast float %1331 to i32, !dbg !7
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1332), !dbg !7
  %1334 = select i1 %.not18, float %1330, float %1324, !dbg !7
  %1335 = bitcast float %1334 to i32, !dbg !7
  %1336 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1335), !dbg !7
  %1337 = extractvalue [2 x <8 x float>] %212, 1, !dbg !7
  %1338 = extractelement <8 x float> %1337, i64 0, !dbg !7
  %1339 = extractelement <8 x float> %1337, i64 1, !dbg !7
  %1340 = extractelement <8 x float> %1337, i64 2, !dbg !7
  %1341 = extractelement <8 x float> %1337, i64 3, !dbg !7
  %1342 = select i1 %.not, float %1339, float %1338, !dbg !7
  %1343 = bitcast float %1342 to i32, !dbg !7
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1343), !dbg !7
  %1345 = bitcast i32 %1344 to float, !dbg !7
  %1346 = select i1 %.not, float %1338, float %1345, !dbg !7
  %1347 = select i1 %.not, float %1345, float %1339, !dbg !7
  %1348 = select i1 %.not, float %1341, float %1340, !dbg !7
  %1349 = bitcast float %1348 to i32, !dbg !7
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1349), !dbg !7
  %1351 = bitcast i32 %1350 to float, !dbg !7
  %1352 = select i1 %.not, float %1340, float %1351, !dbg !7
  %1353 = select i1 %.not, float %1351, float %1341, !dbg !7
  %1354 = select i1 %.not18, float %1352, float %1346, !dbg !7
  %1355 = bitcast float %1354 to i32, !dbg !7
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1355), !dbg !7
  %1357 = bitcast i32 %1356 to float, !dbg !7
  %1358 = select i1 %.not18, float %1346, float %1357, !dbg !7
  %1359 = select i1 %.not18, float %1357, float %1352, !dbg !7
  %1360 = select i1 %.not18, float %1353, float %1347, !dbg !7
  %1361 = bitcast float %1360 to i32, !dbg !7
  %1362 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1361), !dbg !7
  %1363 = bitcast i32 %1362 to float, !dbg !7
  %1364 = select i1 %.not18, float %1347, float %1363, !dbg !7
  %1365 = select i1 %.not18, float %1363, float %1353, !dbg !7
  %1366 = extractvalue [2 x <8 x float>] %211, 1, !dbg !7
  %1367 = extractelement <8 x float> %1366, i64 0, !dbg !7
  %1368 = extractelement <8 x float> %1366, i64 1, !dbg !7
  %1369 = extractelement <8 x float> %1366, i64 2, !dbg !7
  %1370 = extractelement <8 x float> %1366, i64 3, !dbg !7
  %1371 = select i1 %.not, float %1368, float %1367, !dbg !7
  %1372 = bitcast float %1371 to i32, !dbg !7
  %1373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1372), !dbg !7
  %1374 = bitcast i32 %1373 to float, !dbg !7
  %1375 = select i1 %.not, float %1367, float %1374, !dbg !7
  %1376 = select i1 %.not, float %1374, float %1368, !dbg !7
  %1377 = select i1 %.not, float %1370, float %1369, !dbg !7
  %1378 = bitcast float %1377 to i32, !dbg !7
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1378), !dbg !7
  %1380 = bitcast i32 %1379 to float, !dbg !7
  %1381 = select i1 %.not, float %1369, float %1380, !dbg !7
  %1382 = select i1 %.not, float %1380, float %1370, !dbg !7
  %1383 = select i1 %.not18, float %1381, float %1375, !dbg !7
  %1384 = bitcast float %1383 to i32, !dbg !7
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1384), !dbg !7
  %1386 = bitcast i32 %1385 to float, !dbg !7
  %1387 = select i1 %.not18, float %1375, float %1386, !dbg !7
  %1388 = select i1 %.not18, float %1386, float %1381, !dbg !7
  %1389 = select i1 %.not18, float %1382, float %1376, !dbg !7
  %1390 = bitcast float %1389 to i32, !dbg !7
  %1391 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1390), !dbg !7
  %1392 = bitcast i32 %1391 to float, !dbg !7
  %1393 = select i1 %.not18, float %1376, float %1392, !dbg !7
  %1394 = select i1 %.not18, float %1392, float %1382, !dbg !7
  %1395 = extractvalue [2 x <8 x float>] %210, 1, !dbg !7
  %1396 = extractelement <8 x float> %1395, i64 0, !dbg !7
  %1397 = extractelement <8 x float> %1395, i64 1, !dbg !7
  %1398 = extractelement <8 x float> %1395, i64 2, !dbg !7
  %1399 = extractelement <8 x float> %1395, i64 3, !dbg !7
  %1400 = select i1 %.not, float %1397, float %1396, !dbg !7
  %1401 = bitcast float %1400 to i32, !dbg !7
  %1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1401), !dbg !7
  %1403 = bitcast i32 %1402 to float, !dbg !7
  %1404 = select i1 %.not, float %1396, float %1403, !dbg !7
  %1405 = select i1 %.not, float %1403, float %1397, !dbg !7
  %1406 = select i1 %.not, float %1399, float %1398, !dbg !7
  %1407 = bitcast float %1406 to i32, !dbg !7
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1407), !dbg !7
  %1409 = bitcast i32 %1408 to float, !dbg !7
  %1410 = select i1 %.not, float %1398, float %1409, !dbg !7
  %1411 = select i1 %.not, float %1409, float %1399, !dbg !7
  %1412 = select i1 %.not18, float %1410, float %1404, !dbg !7
  %1413 = bitcast float %1412 to i32, !dbg !7
  %1414 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1413), !dbg !7
  %1415 = bitcast i32 %1414 to float, !dbg !7
  %1416 = select i1 %.not18, float %1404, float %1415, !dbg !7
  %1417 = select i1 %.not18, float %1415, float %1410, !dbg !7
  %1418 = select i1 %.not18, float %1411, float %1405, !dbg !7
  %1419 = bitcast float %1418 to i32, !dbg !7
  %1420 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1419), !dbg !7
  %1421 = bitcast i32 %1420 to float, !dbg !7
  %1422 = select i1 %.not18, float %1405, float %1421, !dbg !7
  %1423 = select i1 %.not18, float %1421, float %1411, !dbg !7
  %1424 = extractvalue [2 x <8 x float>] %207, 1, !dbg !7
  %1425 = extractelement <8 x float> %1424, i64 0, !dbg !7
  %1426 = extractelement <8 x float> %1424, i64 1, !dbg !7
  %1427 = extractelement <8 x float> %1424, i64 2, !dbg !7
  %1428 = extractelement <8 x float> %1424, i64 3, !dbg !7
  %1429 = select i1 %.not, float %1426, float %1425, !dbg !7
  %1430 = bitcast float %1429 to i32, !dbg !7
  %1431 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1430), !dbg !7
  %1432 = bitcast i32 %1431 to float, !dbg !7
  %1433 = select i1 %.not, float %1425, float %1432, !dbg !7
  %1434 = select i1 %.not, float %1432, float %1426, !dbg !7
  %1435 = select i1 %.not, float %1428, float %1427, !dbg !7
  %1436 = bitcast float %1435 to i32, !dbg !7
  %1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1436), !dbg !7
  %1438 = bitcast i32 %1437 to float, !dbg !7
  %1439 = select i1 %.not, float %1427, float %1438, !dbg !7
  %1440 = select i1 %.not, float %1438, float %1428, !dbg !7
  %1441 = select i1 %.not18, float %1439, float %1433, !dbg !7
  %1442 = bitcast float %1441 to i32, !dbg !7
  %1443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1442), !dbg !7
  %1444 = select i1 %.not18, float %1440, float %1434, !dbg !7
  %1445 = bitcast float %1444 to i32, !dbg !7
  %1446 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1445), !dbg !7
  %1447 = extractvalue [2 x <1 x float>] %208, 1, !dbg !19
  %1448 = shufflevector <1 x float> %1447, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1449 = extractelement <8 x float> %1337, i64 4, !dbg !7
  %1450 = extractelement <8 x float> %1337, i64 5, !dbg !7
  %1451 = extractelement <8 x float> %1337, i64 6, !dbg !7
  %1452 = extractelement <8 x float> %1337, i64 7, !dbg !7
  %1453 = select i1 %.not, float %1450, float %1449, !dbg !7
  %1454 = bitcast float %1453 to i32, !dbg !7
  %1455 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1454), !dbg !7
  %1456 = bitcast i32 %1455 to float, !dbg !7
  %1457 = select i1 %.not, float %1449, float %1456, !dbg !7
  %1458 = select i1 %.not, float %1456, float %1450, !dbg !7
  %1459 = select i1 %.not, float %1452, float %1451, !dbg !7
  %1460 = bitcast float %1459 to i32, !dbg !7
  %1461 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1460), !dbg !7
  %1462 = bitcast i32 %1461 to float, !dbg !7
  %1463 = select i1 %.not, float %1451, float %1462, !dbg !7
  %1464 = select i1 %.not, float %1462, float %1452, !dbg !7
  %1465 = select i1 %.not18, float %1463, float %1457, !dbg !7
  %1466 = bitcast float %1465 to i32, !dbg !7
  %1467 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1466), !dbg !7
  %1468 = bitcast i32 %1467 to float, !dbg !7
  %1469 = select i1 %.not18, float %1457, float %1468, !dbg !7
  %1470 = select i1 %.not18, float %1468, float %1463, !dbg !7
  %1471 = select i1 %.not18, float %1464, float %1458, !dbg !7
  %1472 = bitcast float %1471 to i32, !dbg !7
  %1473 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1472), !dbg !7
  %1474 = bitcast i32 %1473 to float, !dbg !7
  %1475 = select i1 %.not18, float %1458, float %1474, !dbg !7
  %1476 = select i1 %.not18, float %1474, float %1464, !dbg !7
  %1477 = extractelement <8 x float> %1366, i64 4, !dbg !7
  %1478 = extractelement <8 x float> %1366, i64 5, !dbg !7
  %1479 = extractelement <8 x float> %1366, i64 6, !dbg !7
  %1480 = extractelement <8 x float> %1366, i64 7, !dbg !7
  %1481 = select i1 %.not, float %1478, float %1477, !dbg !7
  %1482 = bitcast float %1481 to i32, !dbg !7
  %1483 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1482), !dbg !7
  %1484 = bitcast i32 %1483 to float, !dbg !7
  %1485 = select i1 %.not, float %1477, float %1484, !dbg !7
  %1486 = select i1 %.not, float %1484, float %1478, !dbg !7
  %1487 = select i1 %.not, float %1480, float %1479, !dbg !7
  %1488 = bitcast float %1487 to i32, !dbg !7
  %1489 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1488), !dbg !7
  %1490 = bitcast i32 %1489 to float, !dbg !7
  %1491 = select i1 %.not, float %1479, float %1490, !dbg !7
  %1492 = select i1 %.not, float %1490, float %1480, !dbg !7
  %1493 = select i1 %.not18, float %1491, float %1485, !dbg !7
  %1494 = bitcast float %1493 to i32, !dbg !7
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1494), !dbg !7
  %1496 = bitcast i32 %1495 to float, !dbg !7
  %1497 = select i1 %.not18, float %1485, float %1496, !dbg !7
  %1498 = select i1 %.not18, float %1496, float %1491, !dbg !7
  %1499 = select i1 %.not18, float %1492, float %1486, !dbg !7
  %1500 = bitcast float %1499 to i32, !dbg !7
  %1501 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1500), !dbg !7
  %1502 = bitcast i32 %1501 to float, !dbg !7
  %1503 = select i1 %.not18, float %1486, float %1502, !dbg !7
  %1504 = select i1 %.not18, float %1502, float %1492, !dbg !7
  %1505 = extractelement <8 x float> %1395, i64 4, !dbg !7
  %1506 = extractelement <8 x float> %1395, i64 5, !dbg !7
  %1507 = extractelement <8 x float> %1395, i64 6, !dbg !7
  %1508 = extractelement <8 x float> %1395, i64 7, !dbg !7
  %1509 = select i1 %.not, float %1506, float %1505, !dbg !7
  %1510 = bitcast float %1509 to i32, !dbg !7
  %1511 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1510), !dbg !7
  %1512 = bitcast i32 %1511 to float, !dbg !7
  %1513 = select i1 %.not, float %1505, float %1512, !dbg !7
  %1514 = select i1 %.not, float %1512, float %1506, !dbg !7
  %1515 = select i1 %.not, float %1508, float %1507, !dbg !7
  %1516 = bitcast float %1515 to i32, !dbg !7
  %1517 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1516), !dbg !7
  %1518 = bitcast i32 %1517 to float, !dbg !7
  %1519 = select i1 %.not, float %1507, float %1518, !dbg !7
  %1520 = select i1 %.not, float %1518, float %1508, !dbg !7
  %1521 = select i1 %.not18, float %1519, float %1513, !dbg !7
  %1522 = bitcast float %1521 to i32, !dbg !7
  %1523 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1522), !dbg !7
  %1524 = bitcast i32 %1523 to float, !dbg !7
  %1525 = select i1 %.not18, float %1513, float %1524, !dbg !7
  %1526 = select i1 %.not18, float %1524, float %1519, !dbg !7
  %1527 = select i1 %.not18, float %1520, float %1514, !dbg !7
  %1528 = bitcast float %1527 to i32, !dbg !7
  %1529 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1528), !dbg !7
  %1530 = bitcast i32 %1529 to float, !dbg !7
  %1531 = select i1 %.not18, float %1514, float %1530, !dbg !7
  %1532 = select i1 %.not18, float %1530, float %1520, !dbg !7
  %1533 = extractelement <8 x float> %1424, i64 4, !dbg !7
  %1534 = extractelement <8 x float> %1424, i64 5, !dbg !7
  %1535 = extractelement <8 x float> %1424, i64 6, !dbg !7
  %1536 = extractelement <8 x float> %1424, i64 7, !dbg !7
  %1537 = select i1 %.not, float %1534, float %1533, !dbg !7
  %1538 = bitcast float %1537 to i32, !dbg !7
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1538), !dbg !7
  %1540 = bitcast i32 %1539 to float, !dbg !7
  %1541 = select i1 %.not, float %1533, float %1540, !dbg !7
  %1542 = select i1 %.not, float %1540, float %1534, !dbg !7
  %1543 = select i1 %.not, float %1536, float %1535, !dbg !7
  %1544 = bitcast float %1543 to i32, !dbg !7
  %1545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1120, i32 %1544), !dbg !7
  %1546 = bitcast i32 %1545 to float, !dbg !7
  %1547 = select i1 %.not, float %1535, float %1546, !dbg !7
  %1548 = select i1 %.not, float %1546, float %1536, !dbg !7
  %1549 = select i1 %.not18, float %1547, float %1541, !dbg !7
  %1550 = bitcast float %1549 to i32, !dbg !7
  %1551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1550), !dbg !7
  %1552 = select i1 %.not18, float %1548, float %1542, !dbg !7
  %1553 = bitcast float %1552 to i32, !dbg !7
  %1554 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1136, i32 %1553), !dbg !7
  br i1 true, label %1555, label %1619, !dbg !21

1555:                                             ; preds = %1104
  %1556 = bitcast i32 %1551 to float, !dbg !7
  %1557 = select i1 %.not18, float %1541, float %1556, !dbg !7
  %1558 = insertelement <4 x float> poison, float %1557, i64 0, !dbg !19
  %1559 = bitcast i32 %1554 to float, !dbg !7
  %1560 = select i1 %.not18, float %1542, float %1559, !dbg !7
  %1561 = insertelement <4 x float> %1558, float %1560, i64 1, !dbg !19
  %1562 = select i1 %.not18, float %1556, float %1547, !dbg !7
  %1563 = insertelement <4 x float> %1561, float %1562, i64 2, !dbg !19
  %1564 = select i1 %.not18, float %1559, float %1548, !dbg !7
  %1565 = insertelement <4 x float> %1563, float %1564, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1448, !dbg !19
  %normalized = fmul <4 x float> %1565, %row.reciprocal, !dbg !19
  %1566 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1567 = extractelement <4 x half> %1566, i64 3, !dbg !20
  %1568 = extractelement <4 x half> %1566, i64 2, !dbg !20
  %1569 = extractelement <4 x half> %1566, i64 1, !dbg !20
  %1570 = extractelement <4 x half> %1566, i64 0, !dbg !20
  %1571 = bitcast i32 %1443 to float, !dbg !7
  %1572 = select i1 %.not18, float %1433, float %1571, !dbg !7
  %1573 = insertelement <4 x float> poison, float %1572, i64 0, !dbg !19
  %1574 = bitcast i32 %1446 to float, !dbg !7
  %1575 = select i1 %.not18, float %1434, float %1574, !dbg !7
  %1576 = insertelement <4 x float> %1573, float %1575, i64 1, !dbg !19
  %1577 = select i1 %.not18, float %1571, float %1439, !dbg !7
  %1578 = insertelement <4 x float> %1576, float %1577, i64 2, !dbg !19
  %1579 = select i1 %.not18, float %1574, float %1440, !dbg !7
  %1580 = insertelement <4 x float> %1578, float %1579, i64 3, !dbg !19
  %normalized54 = fmul <4 x float> %1580, %row.reciprocal, !dbg !19
  %1581 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !20
  %1582 = extractelement <4 x half> %1581, i64 3, !dbg !20
  %1583 = extractelement <4 x half> %1581, i64 2, !dbg !20
  %1584 = extractelement <4 x half> %1581, i64 1, !dbg !20
  %1585 = extractelement <4 x half> %1581, i64 0, !dbg !20
  %1586 = bitcast i32 %1333 to float, !dbg !7
  %1587 = select i1 %.not18, float %1323, float %1586, !dbg !7
  %1588 = insertelement <4 x float> poison, float %1587, i64 0, !dbg !19
  %1589 = bitcast i32 %1336 to float, !dbg !7
  %1590 = select i1 %.not18, float %1324, float %1589, !dbg !7
  %1591 = insertelement <4 x float> %1588, float %1590, i64 1, !dbg !19
  %1592 = select i1 %.not18, float %1586, float %1329, !dbg !7
  %1593 = insertelement <4 x float> %1591, float %1592, i64 2, !dbg !19
  %1594 = select i1 %.not18, float %1589, float %1330, !dbg !7
  %1595 = insertelement <4 x float> %1593, float %1594, i64 3, !dbg !19
  %row.reciprocal55 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1230, !dbg !19
  %normalized56 = fmul <4 x float> %1595, %row.reciprocal55, !dbg !19
  %1596 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !20
  %1597 = extractelement <4 x half> %1596, i64 3, !dbg !20
  %1598 = extractelement <4 x half> %1596, i64 2, !dbg !20
  %1599 = extractelement <4 x half> %1596, i64 1, !dbg !20
  %1600 = extractelement <4 x half> %1596, i64 0, !dbg !20
  %1601 = bitcast i32 %1225 to float, !dbg !7
  %1602 = select i1 %.not18, float %1215, float %1601, !dbg !7
  %1603 = insertelement <4 x float> poison, float %1602, i64 0, !dbg !19
  %1604 = bitcast i32 %1228 to float, !dbg !7
  %1605 = select i1 %.not18, float %1216, float %1604, !dbg !7
  %1606 = insertelement <4 x float> %1603, float %1605, i64 1, !dbg !19
  %1607 = select i1 %.not18, float %1601, float %1221, !dbg !7
  %1608 = insertelement <4 x float> %1606, float %1607, i64 2, !dbg !19
  %1609 = select i1 %.not18, float %1604, float %1222, !dbg !7
  %1610 = insertelement <4 x float> %1608, float %1609, i64 3, !dbg !19
  %normalized57 = fmul <4 x float> %1610, %row.reciprocal55, !dbg !19
  %1611 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !20
  %1612 = extractelement <4 x half> %1611, i64 3, !dbg !20
  %1613 = extractelement <4 x half> %1611, i64 2, !dbg !20
  %1614 = extractelement <4 x half> %1611, i64 1, !dbg !20
  %1615 = extractelement <4 x half> %1611, i64 0, !dbg !20
  %1616 = add nsw i64 %16, %22, !dbg !21
  %1617 = shl i64 %1616, 19, !dbg !21
  %1618 = and i64 %1617, 16252928, !dbg !21
  %address.thread551 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local552 = zext nneg i32 %address.thread551 to i64, !dbg !9
  %address.local553 = add i64 %25, %address.local552, !dbg !21
  %address.local554 = lshr i64 %address.local552, 6, !dbg !8
  %address.local555 = shl nuw nsw i64 %address.local554, 4, !dbg !21
  %address.local556 = add i64 %address.local553, %address.local555, !dbg !21
  %address.local557 = and i64 %address.local552, 240, !dbg !21
  %address.local558 = sub i64 %address.local556, %address.local557, !dbg !21
  %address.local559 = add i64 %address.local558, %26, !dbg !21
  %address.local560 = shl nuw nsw i64 %address.local559, 7, !dbg !21
  %address.local561 = add nuw nsw i64 %1618, %address.local560, !dbg !21
  %address.local562 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local563 = lshr i64 %address.local552, 2, !dbg !8
  %address.local564 = and i64 %address.local563, 12, !dbg !8
  %address.local565 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local562, i64 %address.local564, !dbg !21
  store half %1615, ptr addrspace(1) %address.local565, align 2, !dbg !21
  %address.local566 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local567 = or disjoint i64 %address.local564, 1, !dbg !21
  %address.local568 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local566, i64 %address.local567, !dbg !21
  store half %1614, ptr addrspace(1) %address.local568, align 2, !dbg !21
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local570 = or disjoint i64 %address.local564, 2, !dbg !21
  %address.local571 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local569, i64 %address.local570, !dbg !21
  store half %1613, ptr addrspace(1) %address.local571, align 2, !dbg !21
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local573 = or disjoint i64 %address.local564, 3, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local572, i64 %address.local573, !dbg !21
  store half %1612, ptr addrspace(1) %address.local574, align 2, !dbg !21
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local576 = or disjoint i64 %address.local564, 16, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local576, !dbg !21
  store half %1600, ptr addrspace(1) %address.local577, align 2, !dbg !21
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local579 = or disjoint i64 %address.local564, 17, !dbg !21
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local578, i64 %address.local579, !dbg !21
  store half %1599, ptr addrspace(1) %address.local580, align 2, !dbg !21
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local582 = or disjoint i64 %address.local564, 18, !dbg !21
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local582, !dbg !21
  store half %1598, ptr addrspace(1) %address.local583, align 2, !dbg !21
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local585 = or disjoint i64 %address.local564, 19, !dbg !21
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local584, i64 %address.local585, !dbg !21
  store half %1597, ptr addrspace(1) %address.local586, align 2, !dbg !21
  %address.local587 = shl i64 %address.local559, 7, !dbg !21
  %address.local588 = add i64 %address.local587, 8192, !dbg !21
  %address.local589 = add nuw nsw i64 %1618, %address.local588, !dbg !21
  %address.local590 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local589, !dbg !21
  %address.local591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local590, i64 %address.local564, !dbg !21
  store half %1585, ptr addrspace(1) %address.local591, align 2, !dbg !21
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local589, !dbg !21
  %address.local593 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local592, i64 %address.local567, !dbg !21
  store half %1584, ptr addrspace(1) %address.local593, align 2, !dbg !21
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local589, !dbg !21
  %address.local595 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local594, i64 %address.local570, !dbg !21
  store half %1583, ptr addrspace(1) %address.local595, align 2, !dbg !21
  %address.local596 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local589, !dbg !21
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local596, i64 %address.local573, !dbg !21
  store half %1582, ptr addrspace(1) %address.local597, align 2, !dbg !21
  %address.local598 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local589, !dbg !21
  %address.local599 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local598, i64 %address.local576, !dbg !21
  store half %1570, ptr addrspace(1) %address.local599, align 2, !dbg !21
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local589, !dbg !21
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local579, !dbg !21
  store half %1569, ptr addrspace(1) %address.local601, align 2, !dbg !21
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local589, !dbg !21
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local602, i64 %address.local582, !dbg !21
  store half %1568, ptr addrspace(1) %address.local603, align 2, !dbg !21
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local589, !dbg !21
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local604, i64 %address.local585, !dbg !21
  store half %1567, ptr addrspace(1) %address.local605, align 2, !dbg !21
  br label %1619, !dbg !21

1619:                                             ; preds = %1555, %1104
  br i1 true, label %1620, label %1658, !dbg !21

1620:                                             ; preds = %1619
  %1621 = insertelement <4 x float> poison, float %1525, i64 0, !dbg !19
  %1622 = insertelement <4 x float> %1621, float %1531, i64 1, !dbg !19
  %1623 = insertelement <4 x float> %1622, float %1526, i64 2, !dbg !19
  %1624 = insertelement <4 x float> %1623, float %1532, i64 3, !dbg !19
  %row.reciprocal58 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1448, !dbg !19
  %normalized59 = fmul <4 x float> %1624, %row.reciprocal58, !dbg !19
  %1625 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !20
  %1626 = extractelement <4 x half> %1625, i64 3, !dbg !20
  %1627 = extractelement <4 x half> %1625, i64 2, !dbg !20
  %1628 = extractelement <4 x half> %1625, i64 1, !dbg !20
  %1629 = extractelement <4 x half> %1625, i64 0, !dbg !20
  %1630 = insertelement <4 x float> poison, float %1416, i64 0, !dbg !19
  %1631 = insertelement <4 x float> %1630, float %1422, i64 1, !dbg !19
  %1632 = insertelement <4 x float> %1631, float %1417, i64 2, !dbg !19
  %1633 = insertelement <4 x float> %1632, float %1423, i64 3, !dbg !19
  %normalized60 = fmul <4 x float> %1633, %row.reciprocal58, !dbg !19
  %1634 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !20
  %1635 = extractelement <4 x half> %1634, i64 3, !dbg !20
  %1636 = extractelement <4 x half> %1634, i64 2, !dbg !20
  %1637 = extractelement <4 x half> %1634, i64 1, !dbg !20
  %1638 = extractelement <4 x half> %1634, i64 0, !dbg !20
  %1639 = insertelement <4 x float> poison, float %1307, i64 0, !dbg !19
  %1640 = insertelement <4 x float> %1639, float %1313, i64 1, !dbg !19
  %1641 = insertelement <4 x float> %1640, float %1308, i64 2, !dbg !19
  %1642 = insertelement <4 x float> %1641, float %1314, i64 3, !dbg !19
  %row.reciprocal61 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1230, !dbg !19
  %normalized62 = fmul <4 x float> %1642, %row.reciprocal61, !dbg !19
  %1643 = fptrunc <4 x float> %normalized62 to <4 x half>, !dbg !20
  %1644 = extractelement <4 x half> %1643, i64 3, !dbg !20
  %1645 = extractelement <4 x half> %1643, i64 2, !dbg !20
  %1646 = extractelement <4 x half> %1643, i64 1, !dbg !20
  %1647 = extractelement <4 x half> %1643, i64 0, !dbg !20
  %1648 = insertelement <4 x float> poison, float %1198, i64 0, !dbg !19
  %1649 = insertelement <4 x float> %1648, float %1204, i64 1, !dbg !19
  %1650 = insertelement <4 x float> %1649, float %1199, i64 2, !dbg !19
  %1651 = insertelement <4 x float> %1650, float %1205, i64 3, !dbg !19
  %normalized63 = fmul <4 x float> %1651, %row.reciprocal61, !dbg !19
  %1652 = fptrunc <4 x float> %normalized63 to <4 x half>, !dbg !20
  %1653 = extractelement <4 x half> %1652, i64 3, !dbg !20
  %1654 = extractelement <4 x half> %1652, i64 2, !dbg !20
  %1655 = extractelement <4 x half> %1652, i64 1, !dbg !20
  %1656 = extractelement <4 x half> %1652, i64 0, !dbg !20
  %1657 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread606 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local607 = zext nneg i32 %address.thread606 to i64, !dbg !9
  %address.local608 = add i64 %25, %address.local607, !dbg !21
  %address.local609 = lshr i64 %address.local607, 6, !dbg !8
  %address.local610 = shl nuw nsw i64 %address.local609, 4, !dbg !21
  %address.local611 = add i64 %address.local608, %address.local610, !dbg !21
  %address.local612 = and i64 %address.local607, 240, !dbg !21
  %address.local613 = sub i64 %address.local611, %address.local612, !dbg !21
  %address.local614 = add i64 %address.local613, %26, !dbg !21
  %address.local615 = shl nuw nsw i64 %address.local614, 7, !dbg !21
  %address.local616 = add nuw nsw i64 %1657, %address.local615, !dbg !21
  %address.local617 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local616, !dbg !21
  %address.local618 = lshr i64 %address.local607, 2, !dbg !8
  %address.local619 = and i64 %address.local618, 12, !dbg !8
  %address.local620 = or disjoint i64 %address.local619, 32, !dbg !21
  %address.local621 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local617, i64 %address.local620, !dbg !21
  store half %1656, ptr addrspace(1) %address.local621, align 2, !dbg !21
  %address.local622 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local616, !dbg !21
  %address.local623 = or disjoint i64 %address.local619, 33, !dbg !21
  %address.local624 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local622, i64 %address.local623, !dbg !21
  store half %1655, ptr addrspace(1) %address.local624, align 2, !dbg !21
  %address.local625 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local616, !dbg !21
  %address.local626 = or disjoint i64 %address.local619, 34, !dbg !21
  %address.local627 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local625, i64 %address.local626, !dbg !21
  store half %1654, ptr addrspace(1) %address.local627, align 2, !dbg !21
  %address.local628 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local616, !dbg !21
  %address.local629 = or disjoint i64 %address.local619, 35, !dbg !21
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local628, i64 %address.local629, !dbg !21
  store half %1653, ptr addrspace(1) %address.local630, align 2, !dbg !21
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local616, !dbg !21
  %address.local632 = or disjoint i64 %address.local619, 48, !dbg !21
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local632, !dbg !21
  store half %1647, ptr addrspace(1) %address.local633, align 2, !dbg !21
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local616, !dbg !21
  %address.local635 = or disjoint i64 %address.local619, 49, !dbg !21
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local634, i64 %address.local635, !dbg !21
  store half %1646, ptr addrspace(1) %address.local636, align 2, !dbg !21
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local616, !dbg !21
  %address.local638 = or disjoint i64 %address.local619, 50, !dbg !21
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local637, i64 %address.local638, !dbg !21
  store half %1645, ptr addrspace(1) %address.local639, align 2, !dbg !21
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local616, !dbg !21
  %address.local641 = or i64 %address.local618, 51, !dbg !21
  %address.local642 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local640, i64 %address.local641, !dbg !21
  store half %1644, ptr addrspace(1) %address.local642, align 2, !dbg !21
  %address.local643 = shl i64 %address.local614, 7, !dbg !21
  %address.local644 = add i64 %address.local643, 8192, !dbg !21
  %address.local645 = add nuw nsw i64 %1657, %address.local644, !dbg !21
  %address.local646 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local645, !dbg !21
  %address.local647 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local646, i64 %address.local620, !dbg !21
  store half %1638, ptr addrspace(1) %address.local647, align 2, !dbg !21
  %address.local648 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local645, !dbg !21
  %address.local649 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local623, !dbg !21
  store half %1637, ptr addrspace(1) %address.local649, align 2, !dbg !21
  %address.local650 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local645, !dbg !21
  %address.local651 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local650, i64 %address.local626, !dbg !21
  store half %1636, ptr addrspace(1) %address.local651, align 2, !dbg !21
  %address.local652 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local645, !dbg !21
  %address.local653 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local652, i64 %address.local629, !dbg !21
  store half %1635, ptr addrspace(1) %address.local653, align 2, !dbg !21
  %address.local654 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local645, !dbg !21
  %address.local655 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local654, i64 %address.local632, !dbg !21
  store half %1629, ptr addrspace(1) %address.local655, align 2, !dbg !21
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local645, !dbg !21
  %address.local657 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local656, i64 %address.local635, !dbg !21
  store half %1628, ptr addrspace(1) %address.local657, align 2, !dbg !21
  %address.local658 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local645, !dbg !21
  %address.local659 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local658, i64 %address.local638, !dbg !21
  store half %1627, ptr addrspace(1) %address.local659, align 2, !dbg !21
  %address.local660 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local645, !dbg !21
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local660, i64 %address.local641, !dbg !21
  store half %1626, ptr addrspace(1) %address.local661, align 2, !dbg !21
  br label %1658, !dbg !21

1658:                                             ; preds = %1620, %1619
  br i1 true, label %1659, label %1697, !dbg !21

1659:                                             ; preds = %1658
  %1660 = insertelement <4 x float> poison, float %1497, i64 0, !dbg !19
  %1661 = insertelement <4 x float> %1660, float %1503, i64 1, !dbg !19
  %1662 = insertelement <4 x float> %1661, float %1498, i64 2, !dbg !19
  %1663 = insertelement <4 x float> %1662, float %1504, i64 3, !dbg !19
  %row.reciprocal64 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1448, !dbg !19
  %normalized65 = fmul <4 x float> %1663, %row.reciprocal64, !dbg !19
  %1664 = fptrunc <4 x float> %normalized65 to <4 x half>, !dbg !20
  %1665 = extractelement <4 x half> %1664, i64 3, !dbg !20
  %1666 = extractelement <4 x half> %1664, i64 2, !dbg !20
  %1667 = extractelement <4 x half> %1664, i64 1, !dbg !20
  %1668 = extractelement <4 x half> %1664, i64 0, !dbg !20
  %1669 = insertelement <4 x float> poison, float %1387, i64 0, !dbg !19
  %1670 = insertelement <4 x float> %1669, float %1393, i64 1, !dbg !19
  %1671 = insertelement <4 x float> %1670, float %1388, i64 2, !dbg !19
  %1672 = insertelement <4 x float> %1671, float %1394, i64 3, !dbg !19
  %normalized66 = fmul <4 x float> %1672, %row.reciprocal64, !dbg !19
  %1673 = fptrunc <4 x float> %normalized66 to <4 x half>, !dbg !20
  %1674 = extractelement <4 x half> %1673, i64 3, !dbg !20
  %1675 = extractelement <4 x half> %1673, i64 2, !dbg !20
  %1676 = extractelement <4 x half> %1673, i64 1, !dbg !20
  %1677 = extractelement <4 x half> %1673, i64 0, !dbg !20
  %1678 = insertelement <4 x float> poison, float %1279, i64 0, !dbg !19
  %1679 = insertelement <4 x float> %1678, float %1285, i64 1, !dbg !19
  %1680 = insertelement <4 x float> %1679, float %1280, i64 2, !dbg !19
  %1681 = insertelement <4 x float> %1680, float %1286, i64 3, !dbg !19
  %row.reciprocal67 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1230, !dbg !19
  %normalized68 = fmul <4 x float> %1681, %row.reciprocal67, !dbg !19
  %1682 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !20
  %1683 = extractelement <4 x half> %1682, i64 3, !dbg !20
  %1684 = extractelement <4 x half> %1682, i64 2, !dbg !20
  %1685 = extractelement <4 x half> %1682, i64 1, !dbg !20
  %1686 = extractelement <4 x half> %1682, i64 0, !dbg !20
  %1687 = insertelement <4 x float> poison, float %1169, i64 0, !dbg !19
  %1688 = insertelement <4 x float> %1687, float %1175, i64 1, !dbg !19
  %1689 = insertelement <4 x float> %1688, float %1170, i64 2, !dbg !19
  %1690 = insertelement <4 x float> %1689, float %1176, i64 3, !dbg !19
  %normalized69 = fmul <4 x float> %1690, %row.reciprocal67, !dbg !19
  %1691 = fptrunc <4 x float> %normalized69 to <4 x half>, !dbg !20
  %1692 = extractelement <4 x half> %1691, i64 3, !dbg !20
  %1693 = extractelement <4 x half> %1691, i64 2, !dbg !20
  %1694 = extractelement <4 x half> %1691, i64 1, !dbg !20
  %1695 = extractelement <4 x half> %1691, i64 0, !dbg !20
  %1696 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread662 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local663 = zext nneg i32 %address.thread662 to i64, !dbg !9
  %address.local664 = add i64 %25, %address.local663, !dbg !21
  %address.local665 = lshr i64 %address.local663, 6, !dbg !8
  %address.local666 = shl nuw nsw i64 %address.local665, 4, !dbg !21
  %address.local667 = add i64 %address.local664, %address.local666, !dbg !21
  %address.local668 = and i64 %address.local663, 240, !dbg !21
  %address.local669 = sub i64 %address.local667, %address.local668, !dbg !21
  %address.local670 = add i64 %address.local669, %26, !dbg !21
  %address.local671 = shl nuw nsw i64 %address.local670, 7, !dbg !21
  %address.local672 = add nuw nsw i64 %1696, %address.local671, !dbg !21
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local672, !dbg !21
  %address.local674 = lshr i64 %address.local663, 2, !dbg !8
  %address.local675 = and i64 %address.local674, 12, !dbg !8
  %address.local676 = or disjoint i64 %address.local675, 64, !dbg !21
  %address.local677 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local673, i64 %address.local676, !dbg !21
  store half %1695, ptr addrspace(1) %address.local677, align 2, !dbg !21
  %address.local678 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local672, !dbg !21
  %address.local679 = or disjoint i64 %address.local675, 65, !dbg !21
  %address.local680 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local678, i64 %address.local679, !dbg !21
  store half %1694, ptr addrspace(1) %address.local680, align 2, !dbg !21
  %address.local681 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local672, !dbg !21
  %address.local682 = or disjoint i64 %address.local675, 66, !dbg !21
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local681, i64 %address.local682, !dbg !21
  store half %1693, ptr addrspace(1) %address.local683, align 2, !dbg !21
  %address.local684 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local672, !dbg !21
  %address.local685 = or disjoint i64 %address.local675, 67, !dbg !21
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local684, i64 %address.local685, !dbg !21
  store half %1692, ptr addrspace(1) %address.local686, align 2, !dbg !21
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local672, !dbg !21
  %address.local688 = or disjoint i64 %address.local675, 80, !dbg !21
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local687, i64 %address.local688, !dbg !21
  store half %1686, ptr addrspace(1) %address.local689, align 2, !dbg !21
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local672, !dbg !21
  %address.local691 = or disjoint i64 %address.local675, 81, !dbg !21
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local690, i64 %address.local691, !dbg !21
  store half %1685, ptr addrspace(1) %address.local692, align 2, !dbg !21
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local672, !dbg !21
  %address.local694 = or disjoint i64 %address.local675, 82, !dbg !21
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local693, i64 %address.local694, !dbg !21
  store half %1684, ptr addrspace(1) %address.local695, align 2, !dbg !21
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local672, !dbg !21
  %address.local697 = or disjoint i64 %address.local675, 83, !dbg !21
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local696, i64 %address.local697, !dbg !21
  store half %1683, ptr addrspace(1) %address.local698, align 2, !dbg !21
  %address.local699 = shl i64 %address.local670, 7, !dbg !21
  %address.local700 = add i64 %address.local699, 8192, !dbg !21
  %address.local701 = add nuw nsw i64 %1696, %address.local700, !dbg !21
  %address.local702 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local701, !dbg !21
  %address.local703 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local702, i64 %address.local676, !dbg !21
  store half %1677, ptr addrspace(1) %address.local703, align 2, !dbg !21
  %address.local704 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local701, !dbg !21
  %address.local705 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local704, i64 %address.local679, !dbg !21
  store half %1676, ptr addrspace(1) %address.local705, align 2, !dbg !21
  %address.local706 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local701, !dbg !21
  %address.local707 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local706, i64 %address.local682, !dbg !21
  store half %1675, ptr addrspace(1) %address.local707, align 2, !dbg !21
  %address.local708 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local701, !dbg !21
  %address.local709 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local708, i64 %address.local685, !dbg !21
  store half %1674, ptr addrspace(1) %address.local709, align 2, !dbg !21
  %address.local710 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local701, !dbg !21
  %address.local711 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local710, i64 %address.local688, !dbg !21
  store half %1668, ptr addrspace(1) %address.local711, align 2, !dbg !21
  %address.local712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local701, !dbg !21
  %address.local713 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local712, i64 %address.local691, !dbg !21
  store half %1667, ptr addrspace(1) %address.local713, align 2, !dbg !21
  %address.local714 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local701, !dbg !21
  %address.local715 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local714, i64 %address.local694, !dbg !21
  store half %1666, ptr addrspace(1) %address.local715, align 2, !dbg !21
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local701, !dbg !21
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local716, i64 %address.local697, !dbg !21
  store half %1665, ptr addrspace(1) %address.local717, align 2, !dbg !21
  br label %1697, !dbg !21

1697:                                             ; preds = %1659, %1658
  br i1 true, label %1698, label %1736, !dbg !21

1698:                                             ; preds = %1697
  %1699 = insertelement <4 x float> poison, float %1469, i64 0, !dbg !19
  %1700 = insertelement <4 x float> %1699, float %1475, i64 1, !dbg !19
  %1701 = insertelement <4 x float> %1700, float %1470, i64 2, !dbg !19
  %1702 = insertelement <4 x float> %1701, float %1476, i64 3, !dbg !19
  %row.reciprocal70 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1448, !dbg !19
  %normalized71 = fmul <4 x float> %1702, %row.reciprocal70, !dbg !19
  %1703 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !20
  %1704 = extractelement <4 x half> %1703, i64 3, !dbg !20
  %1705 = extractelement <4 x half> %1703, i64 2, !dbg !20
  %1706 = extractelement <4 x half> %1703, i64 1, !dbg !20
  %1707 = extractelement <4 x half> %1703, i64 0, !dbg !20
  %1708 = insertelement <4 x float> poison, float %1358, i64 0, !dbg !19
  %1709 = insertelement <4 x float> %1708, float %1364, i64 1, !dbg !19
  %1710 = insertelement <4 x float> %1709, float %1359, i64 2, !dbg !19
  %1711 = insertelement <4 x float> %1710, float %1365, i64 3, !dbg !19
  %normalized72 = fmul <4 x float> %1711, %row.reciprocal70, !dbg !19
  %1712 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !20
  %1713 = extractelement <4 x half> %1712, i64 3, !dbg !20
  %1714 = extractelement <4 x half> %1712, i64 2, !dbg !20
  %1715 = extractelement <4 x half> %1712, i64 1, !dbg !20
  %1716 = extractelement <4 x half> %1712, i64 0, !dbg !20
  %1717 = insertelement <4 x float> poison, float %1251, i64 0, !dbg !19
  %1718 = insertelement <4 x float> %1717, float %1257, i64 1, !dbg !19
  %1719 = insertelement <4 x float> %1718, float %1252, i64 2, !dbg !19
  %1720 = insertelement <4 x float> %1719, float %1258, i64 3, !dbg !19
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1230, !dbg !19
  %normalized74 = fmul <4 x float> %1720, %row.reciprocal73, !dbg !19
  %1721 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %1722 = extractelement <4 x half> %1721, i64 3, !dbg !20
  %1723 = extractelement <4 x half> %1721, i64 2, !dbg !20
  %1724 = extractelement <4 x half> %1721, i64 1, !dbg !20
  %1725 = extractelement <4 x half> %1721, i64 0, !dbg !20
  %1726 = insertelement <4 x float> poison, float %1140, i64 0, !dbg !19
  %1727 = insertelement <4 x float> %1726, float %1146, i64 1, !dbg !19
  %1728 = insertelement <4 x float> %1727, float %1141, i64 2, !dbg !19
  %1729 = insertelement <4 x float> %1728, float %1147, i64 3, !dbg !19
  %normalized75 = fmul <4 x float> %1729, %row.reciprocal73, !dbg !19
  %1730 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !20
  %1731 = extractelement <4 x half> %1730, i64 3, !dbg !20
  %1732 = extractelement <4 x half> %1730, i64 2, !dbg !20
  %1733 = extractelement <4 x half> %1730, i64 1, !dbg !20
  %1734 = extractelement <4 x half> %1730, i64 0, !dbg !20
  %1735 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread718 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local719 = zext nneg i32 %address.thread718 to i64, !dbg !9
  %address.local720 = add i64 %25, %address.local719, !dbg !21
  %address.local721 = lshr i64 %address.local719, 6, !dbg !8
  %address.local722 = shl nuw nsw i64 %address.local721, 4, !dbg !21
  %address.local723 = add i64 %address.local720, %address.local722, !dbg !21
  %address.local724 = and i64 %address.local719, 240, !dbg !21
  %address.local725 = sub i64 %address.local723, %address.local724, !dbg !21
  %address.local726 = add i64 %address.local725, %26, !dbg !21
  %address.local727 = shl nuw nsw i64 %address.local726, 7, !dbg !21
  %address.local728 = add nuw nsw i64 %1735, %address.local727, !dbg !21
  %address.local729 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local728, !dbg !21
  %address.local730 = lshr i64 %address.local719, 2, !dbg !8
  %address.local731 = and i64 %address.local730, 12, !dbg !8
  %address.local732 = or disjoint i64 %address.local731, 96, !dbg !21
  %address.local733 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local729, i64 %address.local732, !dbg !21
  store half %1734, ptr addrspace(1) %address.local733, align 2, !dbg !21
  %address.local734 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local728, !dbg !21
  %address.local735 = or disjoint i64 %address.local731, 97, !dbg !21
  %address.local736 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local734, i64 %address.local735, !dbg !21
  store half %1733, ptr addrspace(1) %address.local736, align 2, !dbg !21
  %address.local737 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local728, !dbg !21
  %address.local738 = or disjoint i64 %address.local731, 98, !dbg !21
  %address.local739 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local737, i64 %address.local738, !dbg !21
  store half %1732, ptr addrspace(1) %address.local739, align 2, !dbg !21
  %address.local740 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local728, !dbg !21
  %address.local741 = or disjoint i64 %address.local731, 99, !dbg !21
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local740, i64 %address.local741, !dbg !21
  store half %1731, ptr addrspace(1) %address.local742, align 2, !dbg !21
  %address.local743 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local728, !dbg !21
  %address.local744 = or disjoint i64 %address.local731, 112, !dbg !21
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local743, i64 %address.local744, !dbg !21
  store half %1725, ptr addrspace(1) %address.local745, align 2, !dbg !21
  %address.local746 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local728, !dbg !21
  %address.local747 = or disjoint i64 %address.local731, 113, !dbg !21
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local746, i64 %address.local747, !dbg !21
  store half %1724, ptr addrspace(1) %address.local748, align 2, !dbg !21
  %address.local749 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local728, !dbg !21
  %address.local750 = or disjoint i64 %address.local731, 114, !dbg !21
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local749, i64 %address.local750, !dbg !21
  store half %1723, ptr addrspace(1) %address.local751, align 2, !dbg !21
  %address.local752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local728, !dbg !21
  %address.local753 = or i64 %address.local730, 115, !dbg !21
  %address.local754 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local752, i64 %address.local753, !dbg !21
  store half %1722, ptr addrspace(1) %address.local754, align 2, !dbg !21
  %address.local755 = shl i64 %address.local726, 7, !dbg !21
  %address.local756 = add i64 %address.local755, 8192, !dbg !21
  %address.local757 = add nuw nsw i64 %1735, %address.local756, !dbg !21
  %address.local758 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local757, !dbg !21
  %address.local759 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local758, i64 %address.local732, !dbg !21
  store half %1716, ptr addrspace(1) %address.local759, align 2, !dbg !21
  %address.local760 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local757, !dbg !21
  %address.local761 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local760, i64 %address.local735, !dbg !21
  store half %1715, ptr addrspace(1) %address.local761, align 2, !dbg !21
  %address.local762 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local757, !dbg !21
  %address.local763 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local762, i64 %address.local738, !dbg !21
  store half %1714, ptr addrspace(1) %address.local763, align 2, !dbg !21
  %address.local764 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local757, !dbg !21
  %address.local765 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local764, i64 %address.local741, !dbg !21
  store half %1713, ptr addrspace(1) %address.local765, align 2, !dbg !21
  %address.local766 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local757, !dbg !21
  %address.local767 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local744, !dbg !21
  store half %1707, ptr addrspace(1) %address.local767, align 2, !dbg !21
  %address.local768 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local757, !dbg !21
  %address.local769 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local768, i64 %address.local747, !dbg !21
  store half %1706, ptr addrspace(1) %address.local769, align 2, !dbg !21
  %address.local770 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local757, !dbg !21
  %address.local771 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local770, i64 %address.local750, !dbg !21
  store half %1705, ptr addrspace(1) %address.local771, align 2, !dbg !21
  %address.local772 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local757, !dbg !21
  %address.local773 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local772, i64 %address.local753, !dbg !21
  store half %1704, ptr addrspace(1) %address.local773, align 2, !dbg !21
  br label %1736, !dbg !21

1736:                                             ; preds = %1698, %1697
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_mask_bypass_pass/compiler/run-20261026-105809-6y9uea1z/input-0002-test_input_1x32x4096x128-128-64-32-online/grid-000003")
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
