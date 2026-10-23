; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/export/run-20261023-182033-aeo9_47b/input-0002-test_input_1x32x4096x128-128-64-32-online/retained32-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/export/run-20261023-182033-aeo9_47b/input-0002-test_input_1x32x4096x128-128-64-32-online/retained32-baseline/input.mlir"

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
  %address.local176 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local177 = and i64 %address.local176, 1536, !dbg !9
  %address.local178 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local177, !dbg !9
  %address.local179 = ptrtoint ptr addrspace(3) %address.local178 to i64, !dbg !9
  %address.local180 = trunc i64 %address.local179 to i32, !dbg !9
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local180), !dbg !9
  %35 = zext i32 %34 to i64, !dbg !9
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !9
  %address.local181 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local182 = and i64 %address.local181, 16256, !dbg !9
  %address.local183 = lshr i64 %address.local, 4, !dbg !9
  %address.local184 = and i64 %address.local183, 8, !dbg !9
  %address.local185 = or disjoint i64 %address.local182, %address.local184, !dbg !9
  %address.local186 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local185, !dbg !9
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %38 = ptrtoint ptr addrspace(1) %address.local186 to i64, !dbg !9
  %39 = sub i64 %38, %37, !dbg !9
  %dma.buffer.offset996 = trunc i64 %39 to i32, !dbg !9
  %40 = trunc i64 %37 to i32, !dbg !9
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !9
  %42 = lshr i64 %37, 32, !dbg !9
  %43 = trunc i64 %42 to i32, !dbg !9
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !9
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc997 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !9
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !9
  %dma.buffer.lds998 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !9
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds998, i32 %dma.buffer.offset996, <4 x i32> %dma.buffer.rsrc997) #10, !dbg !9
  %address.local187 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local188 = shl nuw nsw i64 %address.local187, 3, !dbg !9
  %address.local189 = and i64 %address.local188, 3584, !dbg !9
  %address.local190 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local189, !dbg !9
  %address.local191 = ptrtoint ptr addrspace(3) %address.local190 to i64, !dbg !9
  %address.local192 = trunc i64 %address.local191 to i32, !dbg !9
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local192), !dbg !9
  %49 = zext i32 %48 to i64, !dbg !9
  %50 = inttoptr i64 %49 to ptr addrspace(3), !dbg !9
  %address.local193 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local194 = and i64 %address.local193, 16256, !dbg !9
  %address.local195 = lshr i64 %address.local187, 4, !dbg !9
  %address.local196 = and i64 %address.local195, 24, !dbg !9
  %address.local197 = or disjoint i64 %address.local194, %address.local196, !dbg !9
  %address.local198 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local197, !dbg !9
  %51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %52 = ptrtoint ptr addrspace(1) %address.local198 to i64, !dbg !9
  %53 = sub i64 %52, %51, !dbg !9
  %dma.buffer.offset993 = trunc i64 %53 to i32, !dbg !9
  %54 = trunc i64 %51 to i32, !dbg !9
  %55 = insertelement <4 x i32> poison, i32 %54, i64 0, !dbg !9
  %56 = lshr i64 %51, 32, !dbg !9
  %57 = trunc i64 %56 to i32, !dbg !9
  %58 = insertelement <4 x i32> %55, i32 %57, i64 1, !dbg !9
  %59 = insertelement <4 x i32> %58, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc994 = insertelement <4 x i32> %59, i32 131072, i64 3, !dbg !9
  %60 = ptrtoint ptr addrspace(3) %50 to i32, !dbg !9
  %dma.buffer.lds995 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !9
  %61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds995, i32 %dma.buffer.offset993, <4 x i32> %dma.buffer.rsrc994) #10, !dbg !9
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
  %address.local199 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local177, !dbg !9
  %address.local200 = ptrtoint ptr addrspace(3) %address.local199 to i64, !dbg !9
  %address.local201 = trunc i64 %address.local200 to i32, !dbg !9
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local201), !dbg !9
  %69 = zext i32 %68 to i64, !dbg !9
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !9
  %address.local202 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local185, !dbg !9
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %72 = ptrtoint ptr addrspace(1) %address.local202 to i64, !dbg !9
  %73 = sub i64 %72, %71, !dbg !9
  %dma.buffer.offset990 = trunc i64 %73 to i32, !dbg !9
  %74 = trunc i64 %71 to i32, !dbg !9
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !9
  %76 = lshr i64 %71, 32, !dbg !9
  %77 = trunc i64 %76 to i32, !dbg !9
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !9
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc991 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !9
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !9
  %dma.buffer.lds992 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !9
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds992, i32 %dma.buffer.offset990, <4 x i32> %dma.buffer.rsrc991) #10, !dbg !9
  %address.local203 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local189, !dbg !9
  %address.local204 = ptrtoint ptr addrspace(3) %address.local203 to i64, !dbg !9
  %address.local205 = trunc i64 %address.local204 to i32, !dbg !9
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local205), !dbg !9
  %83 = zext i32 %82 to i64, !dbg !9
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !9
  %address.local206 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local197, !dbg !9
  %85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %86 = ptrtoint ptr addrspace(1) %address.local206 to i64, !dbg !9
  %87 = sub i64 %86, %85, !dbg !9
  %dma.buffer.offset987 = trunc i64 %87 to i32, !dbg !9
  %88 = trunc i64 %85 to i32, !dbg !9
  %89 = insertelement <4 x i32> poison, i32 %88, i64 0, !dbg !9
  %90 = lshr i64 %85, 32, !dbg !9
  %91 = trunc i64 %90 to i32, !dbg !9
  %92 = insertelement <4 x i32> %89, i32 %91, i64 1, !dbg !9
  %93 = insertelement <4 x i32> %92, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc988 = insertelement <4 x i32> %93, i32 131072, i64 3, !dbg !9
  %94 = ptrtoint ptr addrspace(3) %84 to i32, !dbg !9
  %dma.buffer.lds989 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !9
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds989, i32 %dma.buffer.offset987, <4 x i32> %dma.buffer.rsrc988) #10, !dbg !9
  %address.local207 = lshr i32 %address.thread, 2, !dbg !8
  %address.local208 = and i32 %address.local207, 4, !dbg !8
  %address.local209 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local210 = and i32 %address.local209, 384, !dbg !8
  %address.local211 = or disjoint i32 %address.local208, %address.local210, !dbg !8
  %address.local212 = shl nuw nsw i32 %address.thread, 3, !dbg !8
  %address.local213 = and i32 %address.local212, 120, !dbg !8
  %address.local214 = or disjoint i32 %address.local211, %address.local213, !dbg !8
  %address.local215 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local216 = and i32 %address.local215, 1024, !dbg !8
  %address.local217 = or disjoint i32 %address.local214, %address.local216, !dbg !8
  %address.local218 = zext nneg i32 %address.local217 to i64, !dbg !8
  %address.local219 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local218, !dbg !8
  %96 = load <4 x half>, ptr addrspace(3) %address.local219, align 8, !dbg !8
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local219, i64 4096, !dbg !8
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !8
  %98 = fmul <4 x half> %96, splat (half 0xH3015), !dbg !8
  %lds.const.gep885 = getelementptr i8, ptr addrspace(3) %address.local219, i64 1024, !dbg !8
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep885, align 8, !dbg !8
  %100 = fmul <4 x half> %97, splat (half 0xH3015), !dbg !8
  %lds.const.gep886 = getelementptr i8, ptr addrspace(3) %address.local219, i64 5120, !dbg !8
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep886, align 8, !dbg !8
  %102 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !8
  %103 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !9
  %105 = getelementptr i8, ptr addrspace(1) %104, i64 128, !dbg !9
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local180), !dbg !9
  %107 = zext i32 %106 to i64, !dbg !9
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !9
  %address.local262 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local185, !dbg !9
  %109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %110 = ptrtoint ptr addrspace(1) %address.local262 to i64, !dbg !9
  %111 = sub i64 %110, %109, !dbg !9
  %dma.buffer.offset984 = trunc i64 %111 to i32, !dbg !9
  %112 = trunc i64 %109 to i32, !dbg !9
  %113 = insertelement <4 x i32> poison, i32 %112, i64 0, !dbg !9
  %114 = lshr i64 %109, 32, !dbg !9
  %115 = trunc i64 %114 to i32, !dbg !9
  %116 = insertelement <4 x i32> %113, i32 %115, i64 1, !dbg !9
  %117 = insertelement <4 x i32> %116, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc985 = insertelement <4 x i32> %117, i32 131072, i64 3, !dbg !9
  %118 = ptrtoint ptr addrspace(3) %108 to i32, !dbg !9
  %dma.buffer.lds986 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !9
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds986, i32 %dma.buffer.offset984, <4 x i32> %dma.buffer.rsrc985) #10, !dbg !9
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local192), !dbg !9
  %121 = zext i32 %120 to i64, !dbg !9
  %122 = inttoptr i64 %121 to ptr addrspace(3), !dbg !9
  %address.local263 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local197, !dbg !9
  %123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %124 = ptrtoint ptr addrspace(1) %address.local263 to i64, !dbg !9
  %125 = sub i64 %124, %123, !dbg !9
  %dma.buffer.offset981 = trunc i64 %125 to i32, !dbg !9
  %126 = trunc i64 %123 to i32, !dbg !9
  %127 = insertelement <4 x i32> poison, i32 %126, i64 0, !dbg !9
  %128 = lshr i64 %123, 32, !dbg !9
  %129 = trunc i64 %128 to i32, !dbg !9
  %130 = insertelement <4 x i32> %127, i32 %129, i64 1, !dbg !9
  %131 = insertelement <4 x i32> %130, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc982 = insertelement <4 x i32> %131, i32 131072, i64 3, !dbg !9
  %132 = ptrtoint ptr addrspace(3) %122 to i32, !dbg !9
  %dma.buffer.lds983 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !9
  %133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds983, i32 %dma.buffer.offset981, <4 x i32> %dma.buffer.rsrc982) #10, !dbg !9
  %lds.const.gep887 = getelementptr i8, ptr addrspace(3) %address.local219, i64 8192, !dbg !8
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep887, align 8, !dbg !8
  %lds.const.gep888 = getelementptr i8, ptr addrspace(3) %address.local219, i64 12288, !dbg !8
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep888, align 8, !dbg !8
  %lds.const.gep889 = getelementptr i8, ptr addrspace(3) %address.local219, i64 9216, !dbg !8
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep889, align 8, !dbg !8
  %137 = fmul <4 x half> %134, splat (half 0xH3015), !dbg !8
  %lds.const.gep890 = getelementptr i8, ptr addrspace(3) %address.local219, i64 13312, !dbg !8
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep890, align 8, !dbg !8
  %139 = fmul <4 x half> %135, splat (half 0xH3015), !dbg !8
  %140 = fmul <4 x half> %136, splat (half 0xH3015), !dbg !8
  %141 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !9
  %143 = getelementptr i8, ptr addrspace(1) %142, i64 192, !dbg !9
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local201), !dbg !9
  %145 = zext i32 %144 to i64, !dbg !9
  %146 = inttoptr i64 %145 to ptr addrspace(3), !dbg !9
  %address.local268 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local185, !dbg !9
  %147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %148 = ptrtoint ptr addrspace(1) %address.local268 to i64, !dbg !9
  %149 = sub i64 %148, %147, !dbg !9
  %dma.buffer.offset978 = trunc i64 %149 to i32, !dbg !9
  %150 = trunc i64 %147 to i32, !dbg !9
  %151 = insertelement <4 x i32> poison, i32 %150, i64 0, !dbg !9
  %152 = lshr i64 %147, 32, !dbg !9
  %153 = trunc i64 %152 to i32, !dbg !9
  %154 = insertelement <4 x i32> %151, i32 %153, i64 1, !dbg !9
  %155 = insertelement <4 x i32> %154, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc979 = insertelement <4 x i32> %155, i32 131072, i64 3, !dbg !9
  %156 = ptrtoint ptr addrspace(3) %146 to i32, !dbg !9
  %dma.buffer.lds980 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %156), !dbg !9
  %157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds980, i32 %dma.buffer.offset978, <4 x i32> %dma.buffer.rsrc979) #10, !dbg !9
  %158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local205), !dbg !9
  %159 = zext i32 %158 to i64, !dbg !9
  %160 = inttoptr i64 %159 to ptr addrspace(3), !dbg !9
  %address.local269 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local197, !dbg !9
  %161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %162 = ptrtoint ptr addrspace(1) %address.local269 to i64, !dbg !9
  %163 = sub i64 %162, %161, !dbg !9
  %dma.buffer.offset975 = trunc i64 %163 to i32, !dbg !9
  %164 = trunc i64 %161 to i32, !dbg !9
  %165 = insertelement <4 x i32> poison, i32 %164, i64 0, !dbg !9
  %166 = lshr i64 %161, 32, !dbg !9
  %167 = trunc i64 %166 to i32, !dbg !9
  %168 = insertelement <4 x i32> %165, i32 %167, i64 1, !dbg !9
  %169 = insertelement <4 x i32> %168, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc976 = insertelement <4 x i32> %169, i32 131072, i64 3, !dbg !9
  %170 = ptrtoint ptr addrspace(3) %160 to i32, !dbg !9
  %dma.buffer.lds977 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !9
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds977, i32 %dma.buffer.offset975, <4 x i32> %dma.buffer.rsrc976) #10, !dbg !9
  %172 = load <4 x half>, ptr addrspace(3) %address.local219, align 8, !dbg !8
  %lds.const.gep891 = getelementptr i8, ptr addrspace(3) %address.local219, i64 4096, !dbg !8
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep891, align 8, !dbg !8
  %lds.const.gep892 = getelementptr i8, ptr addrspace(3) %address.local219, i64 1024, !dbg !8
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep892, align 8, !dbg !8
  %175 = fmul <4 x half> %172, splat (half 0xH3015), !dbg !8
  %lds.const.gep893 = getelementptr i8, ptr addrspace(3) %address.local219, i64 5120, !dbg !8
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep893, align 8, !dbg !8
  %177 = fmul <4 x half> %173, splat (half 0xH3015), !dbg !8
  %178 = fmul <4 x half> %174, splat (half 0xH3015), !dbg !8
  %179 = fmul <4 x half> %176, splat (half 0xH3015), !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %lds.const.gep894 = getelementptr i8, ptr addrspace(3) %address.local219, i64 8192, !dbg !8
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep894, align 8, !dbg !8
  %lds.const.gep895 = getelementptr i8, ptr addrspace(3) %address.local219, i64 12288, !dbg !8
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep895, align 8, !dbg !8
  %lds.const.gep896 = getelementptr i8, ptr addrspace(3) %address.local219, i64 9216, !dbg !8
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep896, align 8, !dbg !8
  %183 = fmul <4 x half> %180, splat (half 0xH3015), !dbg !8
  %lds.const.gep897 = getelementptr i8, ptr addrspace(3) %address.local219, i64 13312, !dbg !8
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep897, align 8, !dbg !8
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
  %index.compact142 = zext nneg i32 %201 to i64, !dbg !17
  %202 = or disjoint i64 %200, %index.compact142, !dbg !17
  %203 = add nuw nsw i64 %202, 1, !dbg !17
  %204 = sdiv i64 %203, 2, !dbg !17
  br label %205, !dbg !7

205:                                              ; preds = %587, %4
  %206 = phi i64 [ %1126, %587 ], [ 0, %4 ], !dbg !7
  %207 = phi [2 x <8 x float>] [ %953, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %208 = phi [2 x <1 x float>] [ %885, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %209 = phi [2 x <1 x float>] [ %1113, %587 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %210 = phi [2 x <8 x float>] [ %1117, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %211 = phi [2 x <8 x float>] [ %1121, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %212 = phi [2 x <8 x float>] [ %1125, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %213 = icmp slt i64 %206, %192, !dbg !7
  br i1 %213, label %214, label %1127, !dbg !7

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
  %address.thread270 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local271 = zext nneg i32 %address.thread270 to i64, !dbg !9
  %address.local272 = shl nuw nsw i64 %address.local271, 3, !dbg !9
  %address.local273 = and i64 %address.local272, 1536, !dbg !16
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local273, !dbg !16
  %address.local275 = ptrtoint ptr addrspace(3) %address.local274 to i64, !dbg !16
  %address.local276 = trunc i64 %address.local275 to i32, !dbg !16
  %228 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local276), !dbg !16
  %229 = zext i32 %228 to i64, !dbg !16
  %230 = inttoptr i64 %229 to ptr addrspace(3), !dbg !16
  %address.local277 = lshr i64 %address.local271, 3, !dbg !16
  %address.local278 = and i64 %address.local277, 24, !dbg !16
  %address.local279 = and i64 %address.local271, 59, !dbg !16
  %address.local280 = lshr i64 %address.local271, 2, !dbg !16
  %address.local281 = and i64 %address.local280, 4, !dbg !16
  %address.local282 = add nuw nsw i64 %address.local281, %address.local271, !dbg !16
  %address.local283 = and i64 %address.local282, 4, !dbg !16
  %address.local284 = or disjoint i64 %address.local279, %address.local283, !dbg !16
  %address.local285 = shl nuw nsw i64 %address.local284, 7, !dbg !16
  %address.local286 = or disjoint i64 %address.local278, %address.local285, !dbg !16
  %address.local287 = getelementptr [2 x i8], ptr addrspace(1) %227, i64 %address.local286, !dbg !16
  %231 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %232 = ptrtoint ptr addrspace(1) %address.local287 to i64, !dbg !16
  %233 = sub i64 %232, %231, !dbg !16
  %dma.buffer.offset972 = trunc i64 %233 to i32, !dbg !16
  %234 = trunc i64 %231 to i32, !dbg !16
  %235 = insertelement <4 x i32> poison, i32 %234, i64 0, !dbg !16
  %236 = lshr i64 %231, 32, !dbg !16
  %237 = trunc i64 %236 to i32, !dbg !16
  %238 = insertelement <4 x i32> %235, i32 %237, i64 1, !dbg !16
  %239 = insertelement <4 x i32> %238, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc973 = insertelement <4 x i32> %239, i32 131072, i64 3, !dbg !16
  %240 = ptrtoint ptr addrspace(3) %230 to i32, !dbg !16
  %dma.buffer.lds974 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %240), !dbg !16
  %241 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds974, i32 %dma.buffer.offset972, <4 x i32> %dma.buffer.rsrc973) #10, !dbg !16
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
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local273, !dbg !16
  %address.local289 = ptrtoint ptr addrspace(3) %address.local288 to i64, !dbg !16
  %address.local290 = trunc i64 %address.local289 to i32, !dbg !16
  %266 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local290), !dbg !16
  %267 = zext i32 %266 to i64, !dbg !16
  %268 = inttoptr i64 %267 to ptr addrspace(3), !dbg !16
  %address.local291 = getelementptr [2 x i8], ptr addrspace(1) %265, i64 %address.local286, !dbg !16
  %269 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %270 = ptrtoint ptr addrspace(1) %address.local291 to i64, !dbg !16
  %271 = sub i64 %270, %269, !dbg !16
  %dma.buffer.offset969 = trunc i64 %271 to i32, !dbg !16
  %272 = trunc i64 %269 to i32, !dbg !16
  %273 = insertelement <4 x i32> poison, i32 %272, i64 0, !dbg !16
  %274 = lshr i64 %269, 32, !dbg !16
  %275 = trunc i64 %274 to i32, !dbg !16
  %276 = insertelement <4 x i32> %273, i32 %275, i64 1, !dbg !16
  %277 = insertelement <4 x i32> %276, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc970 = insertelement <4 x i32> %277, i32 131072, i64 3, !dbg !16
  %278 = ptrtoint ptr addrspace(3) %268 to i32, !dbg !16
  %dma.buffer.lds971 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %278), !dbg !16
  %279 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds971, i32 %dma.buffer.offset969, <4 x i32> %dma.buffer.rsrc970) #10, !dbg !16
  %address.local292 = mul nuw nsw i64 %address.local271, 160, !dbg !17
  %address.local293 = lshr i64 %address.local271, 2, !dbg !17
  %address.local294 = mul nsw i64 %address.local293, -640, !dbg !17
  %address.local295 = add nsw i64 %address.local292, %address.local294, !dbg !17
  %address.local296 = shl nuw nsw i64 %address.local271, 1, !dbg !17
  %address.local297 = and i64 %address.local296, 24, !dbg !17
  %address.local298 = or disjoint i64 %address.local295, %address.local297, !dbg !17
  %address.local299 = shl i64 %address.local271, 62, !dbg !17
  %address.local300 = ashr exact i64 %address.local299, 57, !dbg !17
  %address.local301 = and i64 %address.local300, -64, !dbg !17
  %address.local302 = add nsw i64 %address.local298, %address.local301, !dbg !17
  %address.local303 = lshr i64 %address.local271, 2, !dbg !8
  %address.local304 = and i64 %address.local303, 12, !dbg !8
  %address.local305 = add nsw i64 %address.local302, %address.local304, !dbg !17
  %address.local306 = and i64 %address.local271, 63, !dbg !8
  %address.local307 = lshr i64 %address.local306, 5, !dbg !8
  %address.local308 = mul nuw nsw i64 %address.local307, 504, !dbg !17
  %address.local309 = add i64 %address.local305, %address.local308, !dbg !17
  %address.local310 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local309, !dbg !17
  %280 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !17
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %280, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep898 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !17
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep898, align 8, !dbg !17
  %address.local321 = shl nuw nsw i32 %address.thread270, 1, !dbg !17
  %address.local322 = and i32 %address.local321, 24, !dbg !17
  %address.local323 = shl nuw nsw i32 %address.thread270, 8, !dbg !17
  %address.local324 = and i32 %address.local323, 768, !dbg !17
  %address.local325 = or disjoint i32 %address.local322, %address.local324, !dbg !17
  %address.local326 = shl nuw nsw i32 %address.thread270, 5, !dbg !17
  %address.local327 = and i32 %address.local326, 96, !dbg !17
  %address.local328 = or disjoint i32 %address.local325, %address.local327, !dbg !17
  %address.local329 = zext nneg i32 %address.local328 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %204) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local330 = lshr i32 %address.thread270, 1, !dbg !17
  %address.local331 = and i32 %address.local330, 1, !dbg !17
  %address.local332 = zext nneg i32 %address.local331 to i64, !dbg !17
  %address.local333 = add nsw i64 %address.base, %address.local332, !dbg !17
  %address.local334 = shl i64 %address.local333, 6, !dbg !17
  %address.local335 = sub i64 %address.local329, %address.local334, !dbg !17
  %address.local336 = add i64 %address.local335, %address.local304, !dbg !17
  %address.local337 = add i64 %address.local336, %address.local308, !dbg !17
  %address.local338 = add i64 %address.local337, 32, !dbg !17
  %address.local339 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local338, !dbg !17
  %283 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !17
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %282, <4 x float> %281) #9, !dbg !17
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %283, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep899 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !17
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep899, align 8, !dbg !17
  %lds.const.gep900 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !17
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep900, align 8, !dbg !17
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %286, <4 x float> %285) #9, !dbg !17
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %287, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep901 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !17
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep901, align 8, !dbg !17
  %lds.const.gep902 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !17
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep902, align 8, !dbg !17
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %290, <4 x float> %289) #9, !dbg !17
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %291, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep903 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !17
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep903, align 8, !dbg !17
  %295 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !17
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %294, <4 x float> %293) #9, !dbg !17
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %295, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep904 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !17
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep904, align 8, !dbg !17
  %299 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !17
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %298, <4 x float> %297) #9, !dbg !17
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %299, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep905 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !17
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep905, align 8, !dbg !17
  %lds.const.gep906 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !17
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep906, align 8, !dbg !17
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %302, <4 x float> %301) #9, !dbg !17
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %303, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep907 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !17
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep907, align 8, !dbg !17
  %lds.const.gep908 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !17
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep908, align 8, !dbg !17
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %306, <4 x float> %305) #9, !dbg !17
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %307, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep909 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !17
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep909, align 8, !dbg !17
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
  %335 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local276), !dbg !16
  %336 = zext i32 %335 to i64, !dbg !16
  %337 = inttoptr i64 %336 to ptr addrspace(3), !dbg !16
  %address.local480 = getelementptr [2 x i8], ptr addrspace(1) %334, i64 %address.local286, !dbg !16
  %338 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %339 = ptrtoint ptr addrspace(1) %address.local480 to i64, !dbg !16
  %340 = sub i64 %339, %338, !dbg !16
  %dma.buffer.offset966 = trunc i64 %340 to i32, !dbg !16
  %341 = trunc i64 %338 to i32, !dbg !16
  %342 = insertelement <4 x i32> poison, i32 %341, i64 0, !dbg !16
  %343 = lshr i64 %338, 32, !dbg !16
  %344 = trunc i64 %343 to i32, !dbg !16
  %345 = insertelement <4 x i32> %342, i32 %344, i64 1, !dbg !16
  %346 = insertelement <4 x i32> %345, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc967 = insertelement <4 x i32> %346, i32 131072, i64 3, !dbg !16
  %347 = ptrtoint ptr addrspace(3) %337 to i32, !dbg !16
  %dma.buffer.lds968 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %347), !dbg !16
  %348 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds968, i32 %dma.buffer.offset966, <4 x i32> %dma.buffer.rsrc967) #10, !dbg !16
  %lds.const.gep910 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !17
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep910, align 8, !dbg !17
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %349, <4 x float> %284) #9, !dbg !17
  %lds.const.gep911 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !17
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep911, align 8, !dbg !17
  %lds.const.gep912 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !17
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep912, align 8, !dbg !17
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %351, <4 x float> %350) #9, !dbg !17
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %352, <4 x float> %288) #9, !dbg !17
  %lds.const.gep913 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !17
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep913, align 8, !dbg !17
  %lds.const.gep914 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !17
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep914, align 8, !dbg !17
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %355, <4 x float> %354) #9, !dbg !17
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %356, <4 x float> %292) #9, !dbg !17
  %lds.const.gep915 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !17
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep915, align 8, !dbg !17
  %lds.const.gep916 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !17
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep916, align 8, !dbg !17
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %359, <4 x float> %358) #9, !dbg !17
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %360, <4 x float> %296) #9, !dbg !17
  %lds.const.gep917 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !17
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep917, align 8, !dbg !17
  %lds.const.gep918 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !17
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep918, align 8, !dbg !17
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %363, <4 x float> %362) #9, !dbg !17
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %364, <4 x float> %300) #9, !dbg !17
  %lds.const.gep919 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !17
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep919, align 8, !dbg !17
  %lds.const.gep920 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !17
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep920, align 8, !dbg !17
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %367, <4 x float> %366) #9, !dbg !17
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %368, <4 x float> %304) #9, !dbg !17
  %lds.const.gep921 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !17
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep921, align 8, !dbg !17
  %lds.const.gep922 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !17
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep922, align 8, !dbg !17
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %371, <4 x float> %370) #9, !dbg !17
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %372, <4 x float> %308) #9, !dbg !17
  %lds.const.gep923 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !17
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep923, align 8, !dbg !17
  %lds.const.gep924 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !17
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep924, align 8, !dbg !17
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %375, <4 x float> %374) #9, !dbg !17
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %376, <4 x float> %311) #9, !dbg !17
  %lds.const.gep925 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !17
  %379 = load <4 x half>, ptr addrspace(3) %lds.const.gep925, align 8, !dbg !17
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
  %404 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local290), !dbg !16
  %405 = zext i32 %404 to i64, !dbg !16
  %406 = inttoptr i64 %405 to ptr addrspace(3), !dbg !16
  %address.local489 = getelementptr [2 x i8], ptr addrspace(1) %403, i64 %address.local286, !dbg !16
  %407 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %408 = ptrtoint ptr addrspace(1) %address.local489 to i64, !dbg !16
  %409 = sub i64 %408, %407, !dbg !16
  %dma.buffer.offset963 = trunc i64 %409 to i32, !dbg !16
  %410 = trunc i64 %407 to i32, !dbg !16
  %411 = insertelement <4 x i32> poison, i32 %410, i64 0, !dbg !16
  %412 = lshr i64 %407, 32, !dbg !16
  %413 = trunc i64 %412 to i32, !dbg !16
  %414 = insertelement <4 x i32> %411, i32 %413, i64 1, !dbg !16
  %415 = insertelement <4 x i32> %414, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc964 = insertelement <4 x i32> %415, i32 131072, i64 3, !dbg !16
  %416 = ptrtoint ptr addrspace(3) %406 to i32, !dbg !16
  %dma.buffer.lds965 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %416), !dbg !16
  %417 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds965, i32 %dma.buffer.offset963, <4 x i32> %dma.buffer.rsrc964) #10, !dbg !16
  %418 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !17
  %419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %418, <4 x float> %353) #9, !dbg !17
  %lds.const.gep926 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !17
  %420 = load <4 x half>, ptr addrspace(3) %lds.const.gep926, align 8, !dbg !17
  %421 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !17
  %422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %420, <4 x float> %419) #9, !dbg !17
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %421, <4 x float> %357) #9, !dbg !17
  %lds.const.gep927 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !17
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep927, align 8, !dbg !17
  %lds.const.gep928 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !17
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep928, align 8, !dbg !17
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %424, <4 x float> %423) #9, !dbg !17
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %425, <4 x float> %361) #9, !dbg !17
  %lds.const.gep929 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !17
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep929, align 8, !dbg !17
  %lds.const.gep930 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !17
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep930, align 8, !dbg !17
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %428, <4 x float> %427) #9, !dbg !17
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %429, <4 x float> %365) #9, !dbg !17
  %lds.const.gep931 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !17
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep931, align 8, !dbg !17
  %433 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !17
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %432, <4 x float> %431) #9, !dbg !17
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %433, <4 x float> %369) #9, !dbg !17
  %lds.const.gep932 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !17
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep932, align 8, !dbg !17
  %437 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !17
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %436, <4 x float> %435) #9, !dbg !17
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %437, <4 x float> %373) #9, !dbg !17
  %lds.const.gep933 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !17
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep933, align 8, !dbg !17
  %lds.const.gep934 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !17
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep934, align 8, !dbg !17
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %440, <4 x float> %439) #9, !dbg !17
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %441, <4 x float> %377) #9, !dbg !17
  %lds.const.gep935 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !17
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep935, align 8, !dbg !17
  %lds.const.gep936 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !17
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep936, align 8, !dbg !17
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %444, <4 x float> %443) #9, !dbg !17
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %445, <4 x float> %380) #9, !dbg !17
  %lds.const.gep937 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !17
  %448 = load <4 x half>, ptr addrspace(3) %lds.const.gep937, align 8, !dbg !17
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
  %455 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local276), !dbg !18
  %456 = zext i32 %455 to i64, !dbg !18
  %457 = inttoptr i64 %456 to ptr addrspace(3), !dbg !18
  %address.local490 = shl nuw nsw i64 %address.local271, 5, !dbg !18
  %address.local491 = and i64 %address.local490, 8064, !dbg !18
  %address.local492 = shl nuw nsw i64 %address.local271, 3, !dbg !18
  %address.local493 = and i64 %address.local492, 24, !dbg !18
  %address.local494 = or disjoint i64 %address.local491, %address.local493, !dbg !18
  %address.local495 = getelementptr [2 x i8], ptr addrspace(1) %454, i64 %address.local494, !dbg !18
  %458 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %459 = ptrtoint ptr addrspace(1) %address.local495 to i64, !dbg !18
  %460 = sub i64 %459, %458, !dbg !18
  %dma.buffer.offset960 = trunc i64 %460 to i32, !dbg !18
  %461 = trunc i64 %458 to i32, !dbg !18
  %462 = insertelement <4 x i32> poison, i32 %461, i64 0, !dbg !18
  %463 = lshr i64 %458, 32, !dbg !18
  %464 = trunc i64 %463 to i32, !dbg !18
  %465 = insertelement <4 x i32> %462, i32 %464, i64 1, !dbg !18
  %466 = insertelement <4 x i32> %465, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc961 = insertelement <4 x i32> %466, i32 131072, i64 3, !dbg !18
  %467 = ptrtoint ptr addrspace(3) %457 to i32, !dbg !18
  %dma.buffer.lds962 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %467), !dbg !18
  %468 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds962, i32 %dma.buffer.offset960, <4 x i32> %dma.buffer.rsrc961) #10, !dbg !18
  %lds.const.gep938 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !17
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep938, align 8, !dbg !17
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %469, <4 x float> %422) #9, !dbg !17
  %lds.const.gep939 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !17
  %471 = load <4 x half>, ptr addrspace(3) %lds.const.gep939, align 8, !dbg !17
  %lds.const.gep940 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !17
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep940, align 8, !dbg !17
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %471, <4 x float> %470) #9, !dbg !17
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %472, <4 x float> %426) #9, !dbg !17
  %lds.const.gep941 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !17
  %475 = load <4 x half>, ptr addrspace(3) %lds.const.gep941, align 8, !dbg !17
  %lds.const.gep942 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !17
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep942, align 8, !dbg !17
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %475, <4 x float> %474) #9, !dbg !17
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %476, <4 x float> %430) #9, !dbg !17
  %lds.const.gep943 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !17
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep943, align 8, !dbg !17
  %lds.const.gep944 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !17
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep944, align 8, !dbg !17
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %479, <4 x float> %478) #9, !dbg !17
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %480, <4 x float> %434) #9, !dbg !17
  %lds.const.gep945 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !17
  %483 = load <4 x half>, ptr addrspace(3) %lds.const.gep945, align 8, !dbg !17
  %lds.const.gep946 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !17
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep946, align 8, !dbg !17
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %483, <4 x float> %482) #9, !dbg !17
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %484, <4 x float> %438) #9, !dbg !17
  %lds.const.gep947 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !17
  %487 = load <4 x half>, ptr addrspace(3) %lds.const.gep947, align 8, !dbg !17
  %lds.const.gep948 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !17
  %488 = load <4 x half>, ptr addrspace(3) %lds.const.gep948, align 8, !dbg !17
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %487, <4 x float> %486) #9, !dbg !17
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %488, <4 x float> %442) #9, !dbg !17
  %lds.const.gep949 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !17
  %491 = load <4 x half>, ptr addrspace(3) %lds.const.gep949, align 8, !dbg !17
  %lds.const.gep950 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !17
  %492 = load <4 x half>, ptr addrspace(3) %lds.const.gep950, align 8, !dbg !17
  %493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %491, <4 x float> %490) #9, !dbg !17
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %492, <4 x float> %446) #9, !dbg !17
  %lds.const.gep951 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !17
  %495 = load <4 x half>, ptr addrspace(3) %lds.const.gep951, align 8, !dbg !17
  %lds.const.gep952 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !17
  %496 = load <4 x half>, ptr addrspace(3) %lds.const.gep952, align 8, !dbg !17
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %495, <4 x float> %494) #9, !dbg !17
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %496, <4 x float> %449) #9, !dbg !17
  %lds.const.gep953 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !17
  %499 = load <4 x half>, ptr addrspace(3) %lds.const.gep953, align 8, !dbg !17
  %500 = or disjoint i64 %215, 63, !dbg !14
  %501 = icmp uge i64 %64, %500, !dbg !14
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %499, <4 x float> %498) #9, !dbg !17
  %503 = and i1 %190, %501, !dbg !14
  br i1 %503, label %504, label %505, !dbg !14

504:                                              ; preds = %214
  br label %585, !dbg !14

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
  %compare.high757 = trunc i64 %515 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high757, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high757, !dbg !24
  %compare.low.base = trunc i64 %513 to i32, !dbg !24
  %compare.low.base758 = trunc i64 %512 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base758, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %516 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low759 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result760 = select i1 %compare.high.equal, i1 %compare.low759, i1 %compare.high.order, !dbg !24
  %517 = select i1 %compare.result760, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base761 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local762 = or i32 %compare.local.base761, 2, !dbg !24
  %compare.low763 = icmp ugt i32 %compare.low.base, %compare.local762, !dbg !24
  %compare.result764 = select i1 %compare.high.equal, i1 %compare.low763, i1 %compare.high.order, !dbg !24
  %518 = select i1 %compare.result764, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base765 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local766 = or i32 %compare.local.base765, 3, !dbg !24
  %compare.low767 = icmp ugt i32 %compare.low.base, %compare.local766, !dbg !24
  %compare.result768 = select i1 %compare.high.equal, i1 %compare.low767, i1 %compare.high.order, !dbg !24
  %519 = select i1 %compare.result768, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base769 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local770 = or i32 %compare.local.base769, 4, !dbg !24
  %compare.low771 = icmp ugt i32 %compare.low.base, %compare.local770, !dbg !24
  %compare.result772 = select i1 %compare.high.equal, i1 %compare.low771, i1 %compare.high.order, !dbg !24
  %520 = select i1 %compare.result772, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base773 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local774 = or i32 %compare.local.base773, 5, !dbg !24
  %compare.low775 = icmp ugt i32 %compare.low.base, %compare.local774, !dbg !24
  %compare.result776 = select i1 %compare.high.equal, i1 %compare.low775, i1 %compare.high.order, !dbg !24
  %521 = select i1 %compare.result776, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base777 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local778 = or i32 %compare.local.base777, 6, !dbg !24
  %compare.low779 = icmp ugt i32 %compare.low.base, %compare.local778, !dbg !24
  %compare.result780 = select i1 %compare.high.equal, i1 %compare.low779, i1 %compare.high.order, !dbg !24
  %522 = select i1 %compare.result780, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base781 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local782 = or i32 %compare.local.base781, 7, !dbg !24
  %compare.low783 = icmp ugt i32 %compare.low.base, %compare.local782, !dbg !24
  %compare.result784 = select i1 %compare.high.equal, i1 %compare.low783, i1 %compare.high.order, !dbg !24
  %523 = select i1 %compare.result784, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base785 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local786 = or i32 %compare.local.base785, 8, !dbg !24
  %compare.low787 = icmp ugt i32 %compare.low.base, %compare.local786, !dbg !24
  %compare.result788 = select i1 %compare.high.equal, i1 %compare.low787, i1 %compare.high.order, !dbg !24
  %524 = select i1 %compare.result788, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base789 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local790 = or i32 %compare.local.base789, 9, !dbg !24
  %compare.low791 = icmp ugt i32 %compare.low.base, %compare.local790, !dbg !24
  %compare.result792 = select i1 %compare.high.equal, i1 %compare.low791, i1 %compare.high.order, !dbg !24
  %525 = select i1 %compare.result792, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base793 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local794 = or i32 %compare.local.base793, 10, !dbg !24
  %compare.low795 = icmp ugt i32 %compare.low.base, %compare.local794, !dbg !24
  %compare.result796 = select i1 %compare.high.equal, i1 %compare.low795, i1 %compare.high.order, !dbg !24
  %526 = select i1 %compare.result796, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base797 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local798 = or i32 %compare.local.base797, 11, !dbg !24
  %compare.low799 = icmp ugt i32 %compare.low.base, %compare.local798, !dbg !24
  %compare.result800 = select i1 %compare.high.equal, i1 %compare.low799, i1 %compare.high.order, !dbg !24
  %527 = select i1 %compare.result800, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base801 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local802 = or i32 %compare.local.base801, 12, !dbg !24
  %compare.low803 = icmp ugt i32 %compare.low.base, %compare.local802, !dbg !24
  %compare.result804 = select i1 %compare.high.equal, i1 %compare.low803, i1 %compare.high.order, !dbg !24
  %528 = select i1 %compare.result804, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base805 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local806 = or i32 %compare.local.base805, 13, !dbg !24
  %compare.low807 = icmp ugt i32 %compare.low.base, %compare.local806, !dbg !24
  %compare.result808 = select i1 %compare.high.equal, i1 %compare.low807, i1 %compare.high.order, !dbg !24
  %529 = select i1 %compare.result808, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base809 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local810 = or i32 %compare.local.base809, 14, !dbg !24
  %compare.low811 = icmp ugt i32 %compare.low.base, %compare.local810, !dbg !24
  %compare.result812 = select i1 %compare.high.equal, i1 %compare.low811, i1 %compare.high.order, !dbg !24
  %530 = select i1 %compare.result812, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base813 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local814 = or i32 %compare.local.base813, 15, !dbg !24
  %compare.low815 = icmp ugt i32 %compare.low.base, %compare.local814, !dbg !24
  %compare.result816 = select i1 %compare.high.equal, i1 %compare.low815, i1 %compare.high.order, !dbg !24
  %531 = select i1 %compare.result816, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %532 = add i64 %510, 65, !dbg !23
  %533 = lshr i64 %532, 32, !dbg !24
  %compare.high817 = trunc i64 %533 to i32, !dbg !24
  %534 = lshr i64 %512, 32, !dbg !24
  %compare.high818 = trunc i64 %534 to i32, !dbg !24
  %compare.high.equal819 = icmp eq i32 %compare.high817, %compare.high818, !dbg !24
  %compare.high.order820 = icmp ugt i32 %compare.high817, %compare.high818, !dbg !24
  %compare.low.base821 = trunc i64 %532 to i32, !dbg !24
  %compare.low.base822 = trunc i64 %512 to i32, !dbg !24
  %compare.low823 = icmp ugt i32 %compare.low.base821, %compare.low.base822, !dbg !24
  %compare.result824 = select i1 %compare.high.equal819, i1 %compare.low823, i1 %compare.high.order820, !dbg !24
  %535 = select i1 %compare.result824, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base825 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local826 = or i32 %compare.local.base825, 1, !dbg !24
  %compare.low827 = icmp ugt i32 %compare.low.base821, %compare.local826, !dbg !24
  %compare.result828 = select i1 %compare.high.equal819, i1 %compare.low827, i1 %compare.high.order820, !dbg !24
  %536 = select i1 %compare.result828, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base829 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local830 = or i32 %compare.local.base829, 2, !dbg !24
  %compare.low831 = icmp ugt i32 %compare.low.base821, %compare.local830, !dbg !24
  %compare.result832 = select i1 %compare.high.equal819, i1 %compare.low831, i1 %compare.high.order820, !dbg !24
  %537 = select i1 %compare.result832, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base833 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local834 = or i32 %compare.local.base833, 3, !dbg !24
  %compare.low835 = icmp ugt i32 %compare.low.base821, %compare.local834, !dbg !24
  %compare.result836 = select i1 %compare.high.equal819, i1 %compare.low835, i1 %compare.high.order820, !dbg !24
  %538 = select i1 %compare.result836, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base837 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local838 = or i32 %compare.local.base837, 4, !dbg !24
  %compare.low839 = icmp ugt i32 %compare.low.base821, %compare.local838, !dbg !24
  %compare.result840 = select i1 %compare.high.equal819, i1 %compare.low839, i1 %compare.high.order820, !dbg !24
  %539 = select i1 %compare.result840, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base841 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local842 = or i32 %compare.local.base841, 5, !dbg !24
  %compare.low843 = icmp ugt i32 %compare.low.base821, %compare.local842, !dbg !24
  %compare.result844 = select i1 %compare.high.equal819, i1 %compare.low843, i1 %compare.high.order820, !dbg !24
  %540 = select i1 %compare.result844, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base845 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local846 = or i32 %compare.local.base845, 6, !dbg !24
  %compare.low847 = icmp ugt i32 %compare.low.base821, %compare.local846, !dbg !24
  %compare.result848 = select i1 %compare.high.equal819, i1 %compare.low847, i1 %compare.high.order820, !dbg !24
  %541 = select i1 %compare.result848, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base849 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local850 = or i32 %compare.local.base849, 7, !dbg !24
  %compare.low851 = icmp ugt i32 %compare.low.base821, %compare.local850, !dbg !24
  %compare.result852 = select i1 %compare.high.equal819, i1 %compare.low851, i1 %compare.high.order820, !dbg !24
  %542 = select i1 %compare.result852, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base853 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local854 = or i32 %compare.local.base853, 8, !dbg !24
  %compare.low855 = icmp ugt i32 %compare.low.base821, %compare.local854, !dbg !24
  %compare.result856 = select i1 %compare.high.equal819, i1 %compare.low855, i1 %compare.high.order820, !dbg !24
  %543 = select i1 %compare.result856, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base857 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local858 = or i32 %compare.local.base857, 9, !dbg !24
  %compare.low859 = icmp ugt i32 %compare.low.base821, %compare.local858, !dbg !24
  %compare.result860 = select i1 %compare.high.equal819, i1 %compare.low859, i1 %compare.high.order820, !dbg !24
  %544 = select i1 %compare.result860, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base861 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local862 = or i32 %compare.local.base861, 10, !dbg !24
  %compare.low863 = icmp ugt i32 %compare.low.base821, %compare.local862, !dbg !24
  %compare.result864 = select i1 %compare.high.equal819, i1 %compare.low863, i1 %compare.high.order820, !dbg !24
  %545 = select i1 %compare.result864, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base865 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local866 = or i32 %compare.local.base865, 11, !dbg !24
  %compare.low867 = icmp ugt i32 %compare.low.base821, %compare.local866, !dbg !24
  %compare.result868 = select i1 %compare.high.equal819, i1 %compare.low867, i1 %compare.high.order820, !dbg !24
  %546 = select i1 %compare.result868, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base869 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local870 = or i32 %compare.local.base869, 12, !dbg !24
  %compare.low871 = icmp ugt i32 %compare.low.base821, %compare.local870, !dbg !24
  %compare.result872 = select i1 %compare.high.equal819, i1 %compare.low871, i1 %compare.high.order820, !dbg !24
  %547 = select i1 %compare.result872, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base873 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local874 = or i32 %compare.local.base873, 13, !dbg !24
  %compare.low875 = icmp ugt i32 %compare.low.base821, %compare.local874, !dbg !24
  %compare.result876 = select i1 %compare.high.equal819, i1 %compare.low875, i1 %compare.high.order820, !dbg !24
  %548 = select i1 %compare.result876, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base877 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local878 = or i32 %compare.local.base877, 14, !dbg !24
  %compare.low879 = icmp ugt i32 %compare.low.base821, %compare.local878, !dbg !24
  %compare.result880 = select i1 %compare.high.equal819, i1 %compare.low879, i1 %compare.high.order820, !dbg !24
  %549 = select i1 %compare.result880, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base881 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local882 = or i32 %compare.local.base881, 15, !dbg !24
  %compare.low883 = icmp ugt i32 %compare.low.base821, %compare.local882, !dbg !24
  %compare.result884 = select i1 %compare.high.equal819, i1 %compare.low883, i1 %compare.high.order820, !dbg !24
  %550 = select i1 %compare.result884, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %551 = insertelement <16 x float> poison, float %516, i64 0, !dbg !14
  %552 = insertelement <16 x float> %551, float %517, i64 1, !dbg !14
  %553 = insertelement <16 x float> %552, float %518, i64 2, !dbg !14
  %554 = insertelement <16 x float> %553, float %519, i64 3, !dbg !14
  %555 = insertelement <16 x float> %554, float %520, i64 4, !dbg !14
  %556 = insertelement <16 x float> %555, float %521, i64 5, !dbg !14
  %557 = insertelement <16 x float> %556, float %522, i64 6, !dbg !14
  %558 = insertelement <16 x float> %557, float %523, i64 7, !dbg !14
  %559 = insertelement <16 x float> %558, float %524, i64 8, !dbg !14
  %560 = insertelement <16 x float> %559, float %525, i64 9, !dbg !14
  %561 = insertelement <16 x float> %560, float %526, i64 10, !dbg !14
  %562 = insertelement <16 x float> %561, float %527, i64 11, !dbg !14
  %563 = insertelement <16 x float> %562, float %528, i64 12, !dbg !14
  %564 = insertelement <16 x float> %563, float %529, i64 13, !dbg !14
  %565 = insertelement <16 x float> %564, float %530, i64 14, !dbg !14
  %566 = insertelement <16 x float> %565, float %531, i64 15, !dbg !14
  %567 = insertvalue [2 x <16 x float>] poison, <16 x float> %566, 0, !dbg !14
  %568 = insertelement <16 x float> poison, float %535, i64 0, !dbg !14
  %569 = insertelement <16 x float> %568, float %536, i64 1, !dbg !14
  %570 = insertelement <16 x float> %569, float %537, i64 2, !dbg !14
  %571 = insertelement <16 x float> %570, float %538, i64 3, !dbg !14
  %572 = insertelement <16 x float> %571, float %539, i64 4, !dbg !14
  %573 = insertelement <16 x float> %572, float %540, i64 5, !dbg !14
  %574 = insertelement <16 x float> %573, float %541, i64 6, !dbg !14
  %575 = insertelement <16 x float> %574, float %542, i64 7, !dbg !14
  %576 = insertelement <16 x float> %575, float %543, i64 8, !dbg !14
  %577 = insertelement <16 x float> %576, float %544, i64 9, !dbg !14
  %578 = insertelement <16 x float> %577, float %545, i64 10, !dbg !14
  %579 = insertelement <16 x float> %578, float %546, i64 11, !dbg !14
  %580 = insertelement <16 x float> %579, float %547, i64 12, !dbg !14
  %581 = insertelement <16 x float> %580, float %548, i64 13, !dbg !14
  %582 = insertelement <16 x float> %581, float %549, i64 14, !dbg !14
  %583 = insertelement <16 x float> %582, float %550, i64 15, !dbg !14
  %584 = insertvalue [2 x <16 x float>] %567, <16 x float> %583, 1, !dbg !14
  br label %585, !dbg !14

585:                                              ; preds = %504, %505
  %586 = phi [2 x <16 x float>] [ %584, %505 ], [ zeroinitializer, %504 ], !dbg !14
  br label %587, !dbg !14

587:                                              ; preds = %585
  %588 = extractvalue [2 x <16 x float>] %586, 0, !dbg !26
  %589 = shufflevector <16 x float> %588, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %590 = fadd <4 x float> %473, %589, !dbg !26
  %591 = extractelement <4 x float> %590, i64 0, !dbg !26
  %592 = extractelement <4 x float> %590, i64 1, !dbg !26
  %593 = extractelement <4 x float> %590, i64 2, !dbg !26
  %594 = extractelement <4 x float> %590, i64 3, !dbg !26
  %595 = shufflevector <16 x float> %588, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %596 = fadd <4 x float> %477, %595, !dbg !26
  %597 = extractelement <4 x float> %596, i64 0, !dbg !26
  %598 = extractelement <4 x float> %596, i64 1, !dbg !26
  %599 = extractelement <4 x float> %596, i64 2, !dbg !26
  %600 = extractelement <4 x float> %596, i64 3, !dbg !26
  %601 = shufflevector <16 x float> %588, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %602 = fadd <4 x float> %481, %601, !dbg !26
  %603 = extractelement <4 x float> %602, i64 0, !dbg !26
  %604 = extractelement <4 x float> %602, i64 1, !dbg !26
  %605 = extractelement <4 x float> %602, i64 2, !dbg !26
  %606 = extractelement <4 x float> %602, i64 3, !dbg !26
  %607 = shufflevector <16 x float> %588, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %608 = fadd <4 x float> %485, %607, !dbg !26
  %609 = extractelement <4 x float> %608, i64 0, !dbg !26
  %610 = extractelement <4 x float> %608, i64 1, !dbg !26
  %611 = extractelement <4 x float> %608, i64 2, !dbg !26
  %612 = extractelement <4 x float> %608, i64 3, !dbg !26
  %613 = extractvalue [2 x <16 x float>] %586, 1, !dbg !26
  %614 = shufflevector <16 x float> %613, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %615 = fadd <4 x float> %489, %614, !dbg !26
  %616 = extractelement <4 x float> %615, i64 0, !dbg !26
  %617 = extractelement <4 x float> %615, i64 1, !dbg !26
  %618 = extractelement <4 x float> %615, i64 2, !dbg !26
  %619 = extractelement <4 x float> %615, i64 3, !dbg !26
  %620 = shufflevector <16 x float> %613, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %621 = fadd <4 x float> %493, %620, !dbg !26
  %622 = extractelement <4 x float> %621, i64 0, !dbg !26
  %623 = extractelement <4 x float> %621, i64 1, !dbg !26
  %624 = extractelement <4 x float> %621, i64 2, !dbg !26
  %625 = extractelement <4 x float> %621, i64 3, !dbg !26
  %626 = shufflevector <16 x float> %613, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %627 = fadd <4 x float> %497, %626, !dbg !26
  %628 = extractelement <4 x float> %627, i64 0, !dbg !26
  %629 = extractelement <4 x float> %627, i64 1, !dbg !26
  %630 = extractelement <4 x float> %627, i64 2, !dbg !26
  %631 = extractelement <4 x float> %627, i64 3, !dbg !26
  %632 = shufflevector <16 x float> %613, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %633 = fadd <4 x float> %502, %632, !dbg !26
  %634 = extractelement <4 x float> %633, i64 0, !dbg !26
  %635 = extractelement <4 x float> %633, i64 1, !dbg !26
  %636 = extractelement <4 x float> %633, i64 2, !dbg !26
  %637 = extractelement <4 x float> %633, i64 3, !dbg !26
  %638 = call float @llvm.maxnum.f32(float %591, float 0xFFF0000000000000), !dbg !27
  %639 = call float @llvm.maxnum.f32(float %638, float %592), !dbg !27
  %640 = call float @llvm.maxnum.f32(float %639, float %593), !dbg !27
  %641 = call float @llvm.maxnum.f32(float %640, float %594), !dbg !27
  %642 = call float @llvm.maxnum.f32(float %641, float %597), !dbg !27
  %643 = call float @llvm.maxnum.f32(float %642, float %598), !dbg !27
  %644 = call float @llvm.maxnum.f32(float %643, float %599), !dbg !27
  %645 = call float @llvm.maxnum.f32(float %644, float %600), !dbg !27
  %646 = call float @llvm.maxnum.f32(float %645, float %603), !dbg !27
  %647 = call float @llvm.maxnum.f32(float %646, float %604), !dbg !27
  %648 = call float @llvm.maxnum.f32(float %647, float %605), !dbg !27
  %649 = call float @llvm.maxnum.f32(float %648, float %606), !dbg !27
  %650 = call float @llvm.maxnum.f32(float %649, float %609), !dbg !27
  %651 = call float @llvm.maxnum.f32(float %650, float %610), !dbg !27
  %652 = call float @llvm.maxnum.f32(float %651, float %611), !dbg !27
  %653 = call float @llvm.maxnum.f32(float %652, float %612), !dbg !27
  %654 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %655 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %654), !dbg !27
  %656 = add nuw nsw i32 %655, 64, !dbg !27
  %657 = xor i32 %655, 16, !dbg !27
  %658 = and i32 %656, 192, !dbg !27
  %659 = icmp samesign ult i32 %657, %658, !dbg !27
  %660 = select i1 %659, i32 %657, i32 %655, !dbg !27
  %661 = shl nuw nsw i32 %660, 2, !dbg !27
  %662 = bitcast float %653 to i32, !dbg !27
  %663 = call i32 @llvm.amdgcn.ds.bpermute(i32 %661, i32 %662), !dbg !27
  %664 = bitcast i32 %663 to float, !dbg !27
  %665 = call float @llvm.maxnum.f32(float %653, float %664), !dbg !27
  %666 = xor i32 %655, 32, !dbg !27
  %667 = icmp samesign ult i32 %666, %658, !dbg !27
  %668 = select i1 %667, i32 %666, i32 %655, !dbg !27
  %669 = shl nuw nsw i32 %668, 2, !dbg !27
  %670 = bitcast float %665 to i32, !dbg !27
  %671 = call i32 @llvm.amdgcn.ds.bpermute(i32 %669, i32 %670), !dbg !27
  %672 = bitcast i32 %671 to float, !dbg !27
  %673 = call float @llvm.maxnum.f32(float %665, float %672), !dbg !27
  %674 = extractvalue [2 x <1 x float>] %209, 0, !dbg !27
  %675 = insertelement <1 x float> poison, float %673, i64 0, !dbg !27
  %676 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %675, <1 x float> %674), !dbg !27
  %677 = call float @llvm.maxnum.f32(float %616, float 0xFFF0000000000000), !dbg !27
  %678 = call float @llvm.maxnum.f32(float %677, float %617), !dbg !27
  %679 = call float @llvm.maxnum.f32(float %678, float %618), !dbg !27
  %680 = call float @llvm.maxnum.f32(float %679, float %619), !dbg !27
  %681 = call float @llvm.maxnum.f32(float %680, float %622), !dbg !27
  %682 = call float @llvm.maxnum.f32(float %681, float %623), !dbg !27
  %683 = call float @llvm.maxnum.f32(float %682, float %624), !dbg !27
  %684 = call float @llvm.maxnum.f32(float %683, float %625), !dbg !27
  %685 = call float @llvm.maxnum.f32(float %684, float %628), !dbg !27
  %686 = call float @llvm.maxnum.f32(float %685, float %629), !dbg !27
  %687 = call float @llvm.maxnum.f32(float %686, float %630), !dbg !27
  %688 = call float @llvm.maxnum.f32(float %687, float %631), !dbg !27
  %689 = call float @llvm.maxnum.f32(float %688, float %634), !dbg !27
  %690 = call float @llvm.maxnum.f32(float %689, float %635), !dbg !27
  %691 = call float @llvm.maxnum.f32(float %690, float %636), !dbg !27
  %692 = call float @llvm.maxnum.f32(float %691, float %637), !dbg !27
  %693 = bitcast float %692 to i32, !dbg !27
  %694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %661, i32 %693), !dbg !27
  %695 = bitcast i32 %694 to float, !dbg !27
  %696 = call float @llvm.maxnum.f32(float %692, float %695), !dbg !27
  %697 = bitcast float %696 to i32, !dbg !27
  %698 = call i32 @llvm.amdgcn.ds.bpermute(i32 %669, i32 %697), !dbg !27
  %699 = bitcast i32 %698 to float, !dbg !27
  %700 = call float @llvm.maxnum.f32(float %696, float %699), !dbg !27
  %701 = extractvalue [2 x <1 x float>] %209, 1, !dbg !27
  %702 = insertelement <1 x float> poison, float %700, i64 0, !dbg !27
  %703 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %702, <1 x float> %701), !dbg !27
  %704 = shufflevector <1 x float> %676, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %705 = fsub <4 x float> %590, %704, !dbg !28
  %706 = fsub <4 x float> %596, %704, !dbg !28
  %707 = fsub <4 x float> %602, %704, !dbg !28
  %708 = fsub <4 x float> %608, %704, !dbg !28
  %709 = shufflevector <1 x float> %703, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %710 = fsub <4 x float> %615, %709, !dbg !28
  %711 = fsub <4 x float> %621, %709, !dbg !28
  %712 = fsub <4 x float> %627, %709, !dbg !28
  %713 = fsub <4 x float> %633, %709, !dbg !28
  %714 = fsub <1 x float> %674, %676, !dbg !29
  %715 = extractelement <1 x float> %714, i64 0, !dbg !30
  %716 = call float @llvm.amdgcn.exp2.f32(float %715), !dbg !30
  %717 = insertelement <4 x float> poison, float %716, i64 0, !dbg !35
  %718 = shufflevector <4 x float> %717, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %719 = extractvalue [2 x <1 x float>] %208, 0, !dbg !31
  %720 = extractelement <1 x float> %719, i64 0, !dbg !31
  %721 = fmul float %720, %716, !dbg !31
  %722 = fsub <1 x float> %701, %703, !dbg !29
  %723 = extractelement <1 x float> %722, i64 0, !dbg !30
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !30
  %725 = insertelement <4 x float> poison, float %724, i64 0, !dbg !35
  %726 = shufflevector <4 x float> %725, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %727 = extractvalue [2 x <1 x float>] %208, 1, !dbg !31
  %728 = extractelement <1 x float> %727, i64 0, !dbg !31
  %729 = fmul float %728, %724, !dbg !31
  %730 = extractelement <4 x float> %705, i64 0, !dbg !32
  %731 = call float @llvm.amdgcn.exp2.f32(float %730), !dbg !32
  %732 = insertelement <4 x float> poison, float %731, i64 0, !dbg !34
  %733 = extractelement <4 x float> %705, i64 1, !dbg !32
  %734 = call float @llvm.amdgcn.exp2.f32(float %733), !dbg !32
  %735 = insertelement <4 x float> %732, float %734, i64 1, !dbg !34
  %736 = fadd float %731, %734, !dbg !33
  %737 = extractelement <4 x float> %705, i64 2, !dbg !32
  %738 = call float @llvm.amdgcn.exp2.f32(float %737), !dbg !32
  %739 = insertelement <4 x float> %735, float %738, i64 2, !dbg !34
  %740 = fadd float %736, %738, !dbg !33
  %741 = extractelement <4 x float> %705, i64 3, !dbg !32
  %742 = call float @llvm.amdgcn.exp2.f32(float %741), !dbg !32
  %743 = insertelement <4 x float> %739, float %742, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %743) #9, !dbg !34, !frisk.register_lifetimes !1
  %744 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %744) #9, !dbg !33, !frisk.register_lifetimes !1
  %745 = fadd float %740, %742, !dbg !33
  %746 = extractelement <4 x float> %706, i64 0, !dbg !32
  %747 = call float @llvm.amdgcn.exp2.f32(float %746), !dbg !32
  %748 = insertelement <4 x float> poison, float %747, i64 0, !dbg !34
  %749 = fadd float %745, %747, !dbg !33
  %750 = extractelement <4 x float> %706, i64 1, !dbg !32
  %751 = call float @llvm.amdgcn.exp2.f32(float %750), !dbg !32
  %752 = insertelement <4 x float> %748, float %751, i64 1, !dbg !34
  %753 = fadd float %749, %751, !dbg !33
  %754 = extractelement <4 x float> %706, i64 2, !dbg !32
  %755 = call float @llvm.amdgcn.exp2.f32(float %754), !dbg !32
  %756 = insertelement <4 x float> %752, float %755, i64 2, !dbg !34
  %757 = fadd float %753, %755, !dbg !33
  %758 = extractelement <4 x float> %706, i64 3, !dbg !32
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !32
  %760 = insertelement <4 x float> %756, float %759, i64 3, !dbg !34
  %math.result162 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %760) #9, !dbg !34, !frisk.register_lifetimes !1
  %761 = fptrunc <4 x float> %math.result162 to <4 x half>, !dbg !34
  %math.packed163 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %761) #9, !dbg !33, !frisk.register_lifetimes !1
  %762 = fadd float %757, %759, !dbg !33
  %763 = extractelement <4 x float> %707, i64 0, !dbg !32
  %764 = call float @llvm.amdgcn.exp2.f32(float %763), !dbg !32
  %765 = insertelement <4 x float> poison, float %764, i64 0, !dbg !34
  %766 = fadd float %762, %764, !dbg !33
  %767 = extractelement <4 x float> %707, i64 1, !dbg !32
  %768 = call float @llvm.amdgcn.exp2.f32(float %767), !dbg !32
  %769 = insertelement <4 x float> %765, float %768, i64 1, !dbg !34
  %770 = fadd float %766, %768, !dbg !33
  %771 = extractelement <4 x float> %707, i64 2, !dbg !32
  %772 = call float @llvm.amdgcn.exp2.f32(float %771), !dbg !32
  %773 = insertelement <4 x float> %769, float %772, i64 2, !dbg !34
  %774 = fadd float %770, %772, !dbg !33
  %775 = extractelement <4 x float> %707, i64 3, !dbg !32
  %776 = call float @llvm.amdgcn.exp2.f32(float %775), !dbg !32
  %777 = insertelement <4 x float> %773, float %776, i64 3, !dbg !34
  %math.result164 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %777) #9, !dbg !34, !frisk.register_lifetimes !1
  %778 = fptrunc <4 x float> %math.result164 to <4 x half>, !dbg !34
  %math.packed165 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %778) #9, !dbg !33, !frisk.register_lifetimes !1
  %779 = fadd float %774, %776, !dbg !33
  %780 = extractelement <4 x float> %708, i64 0, !dbg !32
  %781 = call float @llvm.amdgcn.exp2.f32(float %780), !dbg !32
  %782 = insertelement <4 x float> poison, float %781, i64 0, !dbg !34
  %783 = fadd float %779, %781, !dbg !33
  %784 = extractelement <4 x float> %708, i64 1, !dbg !32
  %785 = call float @llvm.amdgcn.exp2.f32(float %784), !dbg !32
  %786 = insertelement <4 x float> %782, float %785, i64 1, !dbg !34
  %787 = fadd float %783, %785, !dbg !33
  %788 = extractelement <4 x float> %708, i64 2, !dbg !32
  %789 = call float @llvm.amdgcn.exp2.f32(float %788), !dbg !32
  %790 = insertelement <4 x float> %786, float %789, i64 2, !dbg !34
  %791 = fadd float %787, %789, !dbg !33
  %792 = extractelement <4 x float> %708, i64 3, !dbg !32
  %793 = call float @llvm.amdgcn.exp2.f32(float %792), !dbg !32
  %794 = insertelement <4 x float> %790, float %793, i64 3, !dbg !34
  %math.result166 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %794) #9, !dbg !34, !frisk.register_lifetimes !1
  %795 = fptrunc <4 x float> %math.result166 to <4 x half>, !dbg !34
  %math.packed167 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %795) #9, !dbg !33, !frisk.register_lifetimes !1
  %796 = fadd float %791, %793, !dbg !33
  %797 = extractelement <4 x float> %710, i64 0, !dbg !32
  %798 = call float @llvm.amdgcn.exp2.f32(float %797), !dbg !32
  %799 = insertelement <4 x float> poison, float %798, i64 0, !dbg !34
  %800 = extractelement <4 x float> %710, i64 1, !dbg !32
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !32
  %802 = insertelement <4 x float> %799, float %801, i64 1, !dbg !34
  %803 = fadd float %798, %801, !dbg !33
  %804 = extractelement <4 x float> %710, i64 2, !dbg !32
  %805 = call float @llvm.amdgcn.exp2.f32(float %804), !dbg !32
  %806 = insertelement <4 x float> %802, float %805, i64 2, !dbg !34
  %807 = fadd float %803, %805, !dbg !33
  %808 = extractelement <4 x float> %710, i64 3, !dbg !32
  %809 = call float @llvm.amdgcn.exp2.f32(float %808), !dbg !32
  %810 = insertelement <4 x float> %806, float %809, i64 3, !dbg !34
  %math.result168 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %810) #9, !dbg !34, !frisk.register_lifetimes !1
  %811 = fptrunc <4 x float> %math.result168 to <4 x half>, !dbg !34
  %math.packed169 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %811) #9, !dbg !33, !frisk.register_lifetimes !1
  %812 = fadd float %807, %809, !dbg !33
  %813 = extractelement <4 x float> %711, i64 0, !dbg !32
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !32
  %815 = insertelement <4 x float> poison, float %814, i64 0, !dbg !34
  %816 = fadd float %812, %814, !dbg !33
  %817 = extractelement <4 x float> %711, i64 1, !dbg !32
  %818 = call float @llvm.amdgcn.exp2.f32(float %817), !dbg !32
  %819 = insertelement <4 x float> %815, float %818, i64 1, !dbg !34
  %820 = fadd float %816, %818, !dbg !33
  %821 = extractelement <4 x float> %711, i64 2, !dbg !32
  %822 = call float @llvm.amdgcn.exp2.f32(float %821), !dbg !32
  %823 = insertelement <4 x float> %819, float %822, i64 2, !dbg !34
  %824 = fadd float %820, %822, !dbg !33
  %825 = extractelement <4 x float> %711, i64 3, !dbg !32
  %826 = call float @llvm.amdgcn.exp2.f32(float %825), !dbg !32
  %827 = insertelement <4 x float> %823, float %826, i64 3, !dbg !34
  %math.result170 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %827) #9, !dbg !34, !frisk.register_lifetimes !1
  %828 = fptrunc <4 x float> %math.result170 to <4 x half>, !dbg !34
  %math.packed171 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %828) #9, !dbg !33, !frisk.register_lifetimes !1
  %829 = fadd float %824, %826, !dbg !33
  %830 = extractelement <4 x float> %712, i64 0, !dbg !32
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !32
  %832 = insertelement <4 x float> poison, float %831, i64 0, !dbg !34
  %833 = fadd float %829, %831, !dbg !33
  %834 = extractelement <4 x float> %712, i64 1, !dbg !32
  %835 = call float @llvm.amdgcn.exp2.f32(float %834), !dbg !32
  %836 = insertelement <4 x float> %832, float %835, i64 1, !dbg !34
  %837 = fadd float %833, %835, !dbg !33
  %838 = extractelement <4 x float> %712, i64 2, !dbg !32
  %839 = call float @llvm.amdgcn.exp2.f32(float %838), !dbg !32
  %840 = insertelement <4 x float> %836, float %839, i64 2, !dbg !34
  %841 = fadd float %837, %839, !dbg !33
  %842 = extractelement <4 x float> %712, i64 3, !dbg !32
  %843 = call float @llvm.amdgcn.exp2.f32(float %842), !dbg !32
  %844 = insertelement <4 x float> %840, float %843, i64 3, !dbg !34
  %math.result172 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %844) #9, !dbg !34, !frisk.register_lifetimes !1
  %845 = fptrunc <4 x float> %math.result172 to <4 x half>, !dbg !34
  %math.packed173 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %845) #9, !dbg !33, !frisk.register_lifetimes !1
  %846 = fadd float %841, %843, !dbg !33
  %847 = extractelement <4 x float> %713, i64 0, !dbg !32
  %848 = call float @llvm.amdgcn.exp2.f32(float %847), !dbg !32
  %849 = insertelement <4 x float> poison, float %848, i64 0, !dbg !34
  %850 = fadd float %846, %848, !dbg !33
  %851 = extractelement <4 x float> %713, i64 1, !dbg !32
  %852 = call float @llvm.amdgcn.exp2.f32(float %851), !dbg !32
  %853 = insertelement <4 x float> %849, float %852, i64 1, !dbg !34
  %854 = fadd float %850, %852, !dbg !33
  %855 = extractelement <4 x float> %713, i64 2, !dbg !32
  %856 = call float @llvm.amdgcn.exp2.f32(float %855), !dbg !32
  %857 = insertelement <4 x float> %853, float %856, i64 2, !dbg !34
  %858 = fadd float %854, %856, !dbg !33
  %859 = extractelement <4 x float> %713, i64 3, !dbg !32
  %860 = call float @llvm.amdgcn.exp2.f32(float %859), !dbg !32
  %861 = insertelement <4 x float> %857, float %860, i64 3, !dbg !34
  %math.result174 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %861) #9, !dbg !34, !frisk.register_lifetimes !1
  %862 = fptrunc <4 x float> %math.result174 to <4 x half>, !dbg !34
  %math.packed175 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %862) #9, !dbg !33, !frisk.register_lifetimes !1
  %863 = fadd float %858, %860, !dbg !33
  %864 = bitcast float %796 to i32, !dbg !33
  %865 = call i32 @llvm.amdgcn.ds.bpermute(i32 %661, i32 %864), !dbg !33
  %866 = bitcast i32 %865 to float, !dbg !33
  %867 = fadd float %796, %866, !dbg !33
  %868 = bitcast float %867 to i32, !dbg !33
  %869 = call i32 @llvm.amdgcn.ds.bpermute(i32 %669, i32 %868), !dbg !33
  %870 = bitcast i32 %869 to float, !dbg !33
  %871 = fadd float %867, %870, !dbg !33
  %872 = fadd float %871, %721, !dbg !33
  %873 = insertelement <1 x float> poison, float %872, i64 0, !dbg !7
  %874 = insertvalue [2 x <1 x float>] poison, <1 x float> %873, 0, !dbg !7
  %875 = bitcast float %863 to i32, !dbg !33
  %876 = call i32 @llvm.amdgcn.ds.bpermute(i32 %661, i32 %875), !dbg !33
  %877 = bitcast i32 %876 to float, !dbg !33
  %878 = fadd float %863, %877, !dbg !33
  %879 = bitcast float %878 to i32, !dbg !33
  %880 = call i32 @llvm.amdgcn.ds.bpermute(i32 %669, i32 %879), !dbg !33
  %881 = bitcast i32 %880 to float, !dbg !33
  %882 = fadd float %878, %881, !dbg !33
  %883 = fadd float %882, %729, !dbg !33
  %884 = insertelement <1 x float> poison, float %883, i64 0, !dbg !7
  %885 = insertvalue [2 x <1 x float>] %874, <1 x float> %884, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread496 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local497 = zext nneg i32 %address.thread496 to i64, !dbg !9
  %address.local498 = and i64 %address.local497, 60, !dbg !15
  %address.local499 = add nuw nsw i64 %address.local498, %address.local497, !dbg !15
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !15
  %address.local501 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !15
  %address.local502 = and i64 %address.local497, 63, !dbg !8
  %address.local503 = lshr i64 %address.local502, 4, !dbg !15
  %address.local504 = mul nuw nsw i64 %address.local503, 384, !dbg !15
  %address.local505 = getelementptr [2 x i8], ptr addrspace(3) %address.local501, i64 %address.local504, !dbg !15
  %address.local506 = lshr i64 %address.local497, 2, !dbg !17
  %address.local507 = mul nsw i64 %address.local506, -32, !dbg !15
  %address.local508 = getelementptr [2 x i8], ptr addrspace(3) %address.local505, i64 %address.local507, !dbg !15
  %886 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !15
  %887 = bitcast <4 x i32> %886 to <8 x half>, !dbg !15
  %address.local509 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !15
  %address.local510 = getelementptr [2 x i8], ptr addrspace(3) %address.local509, i64 %address.local504, !dbg !15
  %address.local511 = getelementptr [2 x i8], ptr addrspace(3) %address.local510, i64 %address.local507, !dbg !15
  %address.local512 = getelementptr i8, ptr addrspace(3) %address.local511, i64 256, !dbg !15
  %888 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !15
  %889 = bitcast <4 x i32> %888 to <8 x half>, !dbg !15
  %address.local513 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !15
  %address.local514 = getelementptr [2 x i8], ptr addrspace(3) %address.local513, i64 %address.local504, !dbg !15
  %address.local515 = getelementptr [2 x i8], ptr addrspace(3) %address.local514, i64 %address.local507, !dbg !15
  %address.local516 = getelementptr i8, ptr addrspace(3) %address.local515, i64 512, !dbg !15
  %890 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !15
  %891 = bitcast <4 x i32> %890 to <8 x half>, !dbg !15
  %address.local517 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !15
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) %address.local517, i64 %address.local504, !dbg !15
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) %address.local518, i64 %address.local507, !dbg !15
  %address.local520 = getelementptr i8, ptr addrspace(3) %address.local519, i64 768, !dbg !15
  %892 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !15
  %893 = bitcast <4 x i32> %892 to <8 x half>, !dbg !15
  %894 = add nuw nsw i64 %218, %16, !dbg !18
  %895 = shl i64 %894, 19, !dbg !18
  %896 = and i64 %895, 16252928, !dbg !18
  %897 = add i64 %896, %224, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %898 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %897, !dbg !18
  %899 = getelementptr i8, ptr addrspace(1) %898, i64 64, !dbg !18
  %address.local521 = shl nuw nsw i64 %address.local497, 3, !dbg !9
  %address.local522 = and i64 %address.local521, 1536, !dbg !16
  %address.local523 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local522, !dbg !16
  %address.local524 = ptrtoint ptr addrspace(3) %address.local523 to i64, !dbg !16
  %address.local525 = trunc i64 %address.local524 to i32, !dbg !16
  %900 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !18
  %901 = zext i32 %900 to i64, !dbg !18
  %902 = inttoptr i64 %901 to ptr addrspace(3), !dbg !18
  %address.local526 = shl nuw nsw i64 %address.local497, 5, !dbg !18
  %address.local527 = and i64 %address.local526, 8064, !dbg !18
  %address.local528 = shl nuw nsw i64 %address.local497, 3, !dbg !18
  %address.local529 = and i64 %address.local528, 24, !dbg !18
  %address.local530 = or disjoint i64 %address.local527, %address.local529, !dbg !18
  %address.local531 = getelementptr [2 x i8], ptr addrspace(1) %899, i64 %address.local530, !dbg !18
  %903 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %904 = ptrtoint ptr addrspace(1) %address.local531 to i64, !dbg !18
  %905 = sub i64 %904, %903, !dbg !18
  %dma.buffer.offset957 = trunc i64 %905 to i32, !dbg !18
  %906 = trunc i64 %903 to i32, !dbg !18
  %907 = insertelement <4 x i32> poison, i32 %906, i64 0, !dbg !18
  %908 = lshr i64 %903, 32, !dbg !18
  %909 = trunc i64 %908 to i32, !dbg !18
  %910 = insertelement <4 x i32> %907, i32 %909, i64 1, !dbg !18
  %911 = insertelement <4 x i32> %910, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc958 = insertelement <4 x i32> %911, i32 131072, i64 3, !dbg !18
  %912 = ptrtoint ptr addrspace(3) %902 to i32, !dbg !18
  %dma.buffer.lds959 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %912), !dbg !18
  %913 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds959, i32 %dma.buffer.offset957, <4 x i32> %dma.buffer.rsrc958) #10, !dbg !18
  %914 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %914, <4 x float> zeroinitializer) #9, !dbg !15
  %916 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %916, <4 x float> %915) #9, !dbg !15
  %918 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %918, <4 x float> %917) #9, !dbg !15
  %920 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %920, <4 x float> %919) #9, !dbg !15
  %922 = extractvalue [2 x <8 x float>] %207, 0, !dbg !35
  %923 = shufflevector <8 x float> %922, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %924 = fmul <4 x float> %923, %718, !dbg !35
  %925 = fadd <4 x float> %924, %921, !dbg !36
  %926 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %927 = shufflevector <8 x float> %922, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %928 = fmul <4 x float> %927, %718, !dbg !35
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %926, <4 x float> %928) #9, !dbg !15
  %930 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %930, <4 x float> %929) #9, !dbg !15
  %932 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %932, <4 x float> %931) #9, !dbg !15
  %934 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %935 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %934, <4 x float> %933) #9, !dbg !15
  %936 = shufflevector <4 x float> %925, <4 x float> %935, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %937 = insertvalue [2 x <8 x float>] poison, <8 x float> %936, 0, !dbg !7
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %914, <4 x float> zeroinitializer) #9, !dbg !15
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %916, <4 x float> %938) #9, !dbg !15
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %918, <4 x float> %939) #9, !dbg !15
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %920, <4 x float> %940) #9, !dbg !15
  %942 = extractvalue [2 x <8 x float>] %207, 1, !dbg !35
  %943 = shufflevector <8 x float> %942, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %944 = fmul <4 x float> %943, %726, !dbg !35
  %945 = fadd <4 x float> %944, %941, !dbg !36
  %946 = shufflevector <8 x float> %942, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %947 = fmul <4 x float> %946, %726, !dbg !35
  %948 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %926, <4 x float> %947) #9, !dbg !15
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %930, <4 x float> %948) #9, !dbg !15
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %932, <4 x float> %949) #9, !dbg !15
  %951 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %934, <4 x float> %950) #9, !dbg !15
  %952 = shufflevector <4 x float> %945, <4 x float> %951, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %953 = insertvalue [2 x <8 x float>] %937, <8 x float> %952, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %954 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !15
  %955 = bitcast <4 x i32> %954 to <8 x half>, !dbg !15
  %956 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !15
  %957 = bitcast <4 x i32> %956 to <8 x half>, !dbg !15
  %958 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !15
  %959 = bitcast <4 x i32> %958 to <8 x half>, !dbg !15
  %960 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !15
  %961 = bitcast <4 x i32> %960 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %962 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %897, !dbg !18
  %963 = getelementptr i8, ptr addrspace(1) %962, i64 128, !dbg !18
  %964 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !18
  %965 = zext i32 %964 to i64, !dbg !18
  %966 = inttoptr i64 %965 to ptr addrspace(3), !dbg !18
  %address.local532 = getelementptr [2 x i8], ptr addrspace(1) %963, i64 %address.local530, !dbg !18
  %967 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %968 = ptrtoint ptr addrspace(1) %address.local532 to i64, !dbg !18
  %969 = sub i64 %968, %967, !dbg !18
  %dma.buffer.offset954 = trunc i64 %969 to i32, !dbg !18
  %970 = trunc i64 %967 to i32, !dbg !18
  %971 = insertelement <4 x i32> poison, i32 %970, i64 0, !dbg !18
  %972 = lshr i64 %967, 32, !dbg !18
  %973 = trunc i64 %972 to i32, !dbg !18
  %974 = insertelement <4 x i32> %971, i32 %973, i64 1, !dbg !18
  %975 = insertelement <4 x i32> %974, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc955 = insertelement <4 x i32> %975, i32 131072, i64 3, !dbg !18
  %976 = ptrtoint ptr addrspace(3) %966 to i32, !dbg !18
  %dma.buffer.lds956 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %976), !dbg !18
  %977 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds956, i32 %dma.buffer.offset954, <4 x i32> %dma.buffer.rsrc955) #10, !dbg !18
  %978 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %979 = extractvalue [2 x <8 x float>] %210, 0, !dbg !35
  %980 = shufflevector <8 x float> %979, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %981 = fmul <4 x float> %980, %718, !dbg !35
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %978, <4 x float> %981) #9, !dbg !15
  %983 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %983, <4 x float> %982) #9, !dbg !15
  %985 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %985, <4 x float> %984) #9, !dbg !15
  %987 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %987, <4 x float> %986) #9, !dbg !15
  %989 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %990 = shufflevector <8 x float> %979, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %991 = fmul <4 x float> %990, %718, !dbg !35
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %989, <4 x float> %991) #9, !dbg !15
  %993 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %993, <4 x float> %992) #9, !dbg !15
  %995 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %995, <4 x float> %994) #9, !dbg !15
  %997 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %997, <4 x float> %996) #9, !dbg !15
  %999 = extractvalue [2 x <8 x float>] %210, 1, !dbg !35
  %1000 = shufflevector <8 x float> %999, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1001 = fmul <4 x float> %1000, %726, !dbg !35
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %978, <4 x float> %1001) #9, !dbg !15
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %983, <4 x float> %1002) #9, !dbg !15
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %985, <4 x float> %1003) #9, !dbg !15
  %1005 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %987, <4 x float> %1004) #9, !dbg !15
  %1006 = shufflevector <8 x float> %999, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1007 = fmul <4 x float> %1006, %726, !dbg !35
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %989, <4 x float> %1007) #9, !dbg !15
  %1009 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %993, <4 x float> %1008) #9, !dbg !15
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %995, <4 x float> %1009) #9, !dbg !15
  %1011 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %997, <4 x float> %1010) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1012 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !15
  %1013 = bitcast <4 x i32> %1012 to <8 x half>, !dbg !15
  %1014 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !15
  %1015 = bitcast <4 x i32> %1014 to <8 x half>, !dbg !15
  %1016 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !15
  %1017 = bitcast <4 x i32> %1016 to <8 x half>, !dbg !15
  %1018 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !15
  %1019 = bitcast <4 x i32> %1018 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1020 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %897, !dbg !18
  %1021 = getelementptr i8, ptr addrspace(1) %1020, i64 192, !dbg !18
  %1022 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !18
  %1023 = zext i32 %1022 to i64, !dbg !18
  %1024 = inttoptr i64 %1023 to ptr addrspace(3), !dbg !18
  %address.local533 = getelementptr [2 x i8], ptr addrspace(1) %1021, i64 %address.local530, !dbg !18
  %1025 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1026 = ptrtoint ptr addrspace(1) %address.local533 to i64, !dbg !18
  %1027 = sub i64 %1026, %1025, !dbg !18
  %dma.buffer.offset = trunc i64 %1027 to i32, !dbg !18
  %1028 = trunc i64 %1025 to i32, !dbg !18
  %1029 = insertelement <4 x i32> poison, i32 %1028, i64 0, !dbg !18
  %1030 = lshr i64 %1025, 32, !dbg !18
  %1031 = trunc i64 %1030 to i32, !dbg !18
  %1032 = insertelement <4 x i32> %1029, i32 %1031, i64 1, !dbg !18
  %1033 = insertelement <4 x i32> %1032, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1033, i32 131072, i64 3, !dbg !18
  %1034 = ptrtoint ptr addrspace(3) %1024 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1034), !dbg !18
  %1035 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1036 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1037 = extractvalue [2 x <8 x float>] %211, 0, !dbg !35
  %1038 = shufflevector <8 x float> %1037, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1039 = fmul <4 x float> %1038, %718, !dbg !35
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1036, <4 x float> %1039) #9, !dbg !15
  %1041 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1041, <4 x float> %1040) #9, !dbg !15
  %1043 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1043, <4 x float> %1042) #9, !dbg !15
  %1045 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1045, <4 x float> %1044) #9, !dbg !15
  %1047 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1048 = shufflevector <8 x float> %1037, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1049 = fmul <4 x float> %1048, %718, !dbg !35
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1047, <4 x float> %1049) #9, !dbg !15
  %1051 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1051, <4 x float> %1050) #9, !dbg !15
  %1053 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1053, <4 x float> %1052) #9, !dbg !15
  %1055 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1055, <4 x float> %1054) #9, !dbg !15
  %1057 = extractvalue [2 x <8 x float>] %211, 1, !dbg !35
  %1058 = shufflevector <8 x float> %1057, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1059 = fmul <4 x float> %1058, %726, !dbg !35
  %1060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1036, <4 x float> %1059) #9, !dbg !15
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1041, <4 x float> %1060) #9, !dbg !15
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1043, <4 x float> %1061) #9, !dbg !15
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1045, <4 x float> %1062) #9, !dbg !15
  %1064 = shufflevector <8 x float> %1057, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1065 = fmul <4 x float> %1064, %726, !dbg !35
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1047, <4 x float> %1065) #9, !dbg !15
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1051, <4 x float> %1066) #9, !dbg !15
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1053, <4 x float> %1067) #9, !dbg !15
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1055, <4 x float> %1068) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1070 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !15
  %1071 = bitcast <4 x i32> %1070 to <8 x half>, !dbg !15
  %1072 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !15
  %1073 = bitcast <4 x i32> %1072 to <8 x half>, !dbg !15
  %1074 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !15
  %1075 = bitcast <4 x i32> %1074 to <8 x half>, !dbg !15
  %1076 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !15
  %1077 = bitcast <4 x i32> %1076 to <8 x half>, !dbg !15
  %1078 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1079 = extractvalue [2 x <8 x float>] %212, 0, !dbg !35
  %1080 = shufflevector <8 x float> %1079, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1081 = fmul <4 x float> %1080, %718, !dbg !35
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1078, <4 x float> %1081) #9, !dbg !15
  %1083 = shufflevector <8 x half> %1073, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1083, <4 x float> %1082) #9, !dbg !15
  %1085 = shufflevector <8 x half> %1075, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1085, <4 x float> %1084) #9, !dbg !15
  %1087 = shufflevector <8 x half> %1077, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1087, <4 x float> %1086) #9, !dbg !15
  %1089 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1090 = shufflevector <8 x float> %1079, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1091 = fmul <4 x float> %1090, %718, !dbg !35
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1089, <4 x float> %1091) #9, !dbg !15
  %1093 = shufflevector <8 x half> %1073, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1093, <4 x float> %1092) #9, !dbg !15
  %1095 = shufflevector <8 x half> %1075, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1095, <4 x float> %1094) #9, !dbg !15
  %1097 = shufflevector <8 x half> %1077, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1097, <4 x float> %1096) #9, !dbg !15
  %1099 = extractvalue [2 x <8 x float>] %212, 1, !dbg !35
  %1100 = shufflevector <8 x float> %1099, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1101 = fmul <4 x float> %1100, %726, !dbg !35
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1078, <4 x float> %1101) #9, !dbg !15
  %1103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1083, <4 x float> %1102) #9, !dbg !15
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1085, <4 x float> %1103) #9, !dbg !15
  %1105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1087, <4 x float> %1104) #9, !dbg !15
  %1106 = shufflevector <8 x float> %1099, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1107 = fmul <4 x float> %1106, %726, !dbg !35
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1089, <4 x float> %1107) #9, !dbg !15
  %1109 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1093, <4 x float> %1108) #9, !dbg !15
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1095, <4 x float> %1109) #9, !dbg !15
  %1111 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1097, <4 x float> %1110) #9, !dbg !15
  %1112 = insertvalue [2 x <1 x float>] poison, <1 x float> %676, 0, !dbg !7
  %1113 = insertvalue [2 x <1 x float>] %1112, <1 x float> %703, 1, !dbg !7
  %1114 = shufflevector <4 x float> %988, <4 x float> %998, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1115 = insertvalue [2 x <8 x float>] poison, <8 x float> %1114, 0, !dbg !7
  %1116 = shufflevector <4 x float> %1005, <4 x float> %1011, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1117 = insertvalue [2 x <8 x float>] %1115, <8 x float> %1116, 1, !dbg !7
  %1118 = shufflevector <4 x float> %1046, <4 x float> %1056, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1119 = insertvalue [2 x <8 x float>] poison, <8 x float> %1118, 0, !dbg !7
  %1120 = shufflevector <4 x float> %1063, <4 x float> %1069, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1121 = insertvalue [2 x <8 x float>] %1119, <8 x float> %1120, 1, !dbg !7
  %1122 = shufflevector <4 x float> %1088, <4 x float> %1098, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1123 = insertvalue [2 x <8 x float>] poison, <8 x float> %1122, 0, !dbg !7
  %1124 = shufflevector <4 x float> %1105, <4 x float> %1111, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1125 = insertvalue [2 x <8 x float>] %1123, <8 x float> %1124, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1126 = add nuw nsw i64 %206, 1, !dbg !7
  br label %205, !dbg !7

1127:                                             ; preds = %205
  %1128 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %1128, 0, !dbg !7
  %1129 = and i32 %31, 32, !dbg !7
  %.not18 = icmp eq i32 %1129, 0, !dbg !7
  %1130 = extractvalue [2 x <8 x float>] %212, 0, !dbg !7
  %1131 = extractelement <8 x float> %1130, i64 0, !dbg !7
  %1132 = extractelement <8 x float> %1130, i64 1, !dbg !7
  %1133 = extractelement <8 x float> %1130, i64 2, !dbg !7
  %1134 = extractelement <8 x float> %1130, i64 3, !dbg !7
  %1135 = select i1 %.not, float %1132, float %1131, !dbg !7
  %1136 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1137 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1136), !dbg !7
  %1138 = add nuw nsw i32 %1137, 64, !dbg !7
  %1139 = xor i32 %1137, 16, !dbg !7
  %1140 = and i32 %1138, 192, !dbg !7
  %1141 = icmp samesign ult i32 %1139, %1140, !dbg !7
  %1142 = select i1 %1141, i32 %1139, i32 %1137, !dbg !7
  %1143 = shl nuw nsw i32 %1142, 2, !dbg !7
  %1144 = bitcast float %1135 to i32, !dbg !7
  %1145 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1144), !dbg !7
  %1146 = bitcast i32 %1145 to float, !dbg !7
  %1147 = select i1 %.not, float %1131, float %1146, !dbg !7
  %1148 = select i1 %.not, float %1146, float %1132, !dbg !7
  %1149 = select i1 %.not, float %1134, float %1133, !dbg !7
  %1150 = bitcast float %1149 to i32, !dbg !7
  %1151 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1150), !dbg !7
  %1152 = bitcast i32 %1151 to float, !dbg !7
  %1153 = select i1 %.not, float %1133, float %1152, !dbg !7
  %1154 = select i1 %.not, float %1152, float %1134, !dbg !7
  %1155 = select i1 %.not18, float %1153, float %1147, !dbg !7
  %1156 = xor i32 %1137, 32, !dbg !7
  %1157 = icmp samesign ult i32 %1156, %1140, !dbg !7
  %1158 = select i1 %1157, i32 %1156, i32 %1137, !dbg !7
  %1159 = shl nuw nsw i32 %1158, 2, !dbg !7
  %1160 = bitcast float %1155 to i32, !dbg !7
  %1161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1160), !dbg !7
  %1162 = bitcast i32 %1161 to float, !dbg !7
  %1163 = select i1 %.not18, float %1147, float %1162, !dbg !7
  %1164 = select i1 %.not18, float %1162, float %1153, !dbg !7
  %1165 = select i1 %.not18, float %1154, float %1148, !dbg !7
  %1166 = bitcast float %1165 to i32, !dbg !7
  %1167 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1166), !dbg !7
  %1168 = bitcast i32 %1167 to float, !dbg !7
  %1169 = select i1 %.not18, float %1148, float %1168, !dbg !7
  %1170 = select i1 %.not18, float %1168, float %1154, !dbg !7
  %1171 = extractvalue [2 x <8 x float>] %211, 0, !dbg !7
  %1172 = extractelement <8 x float> %1171, i64 0, !dbg !7
  %1173 = extractelement <8 x float> %1171, i64 1, !dbg !7
  %1174 = extractelement <8 x float> %1171, i64 2, !dbg !7
  %1175 = extractelement <8 x float> %1171, i64 3, !dbg !7
  %1176 = select i1 %.not, float %1173, float %1172, !dbg !7
  %1177 = bitcast float %1176 to i32, !dbg !7
  %1178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1177), !dbg !7
  %1179 = bitcast i32 %1178 to float, !dbg !7
  %1180 = select i1 %.not, float %1172, float %1179, !dbg !7
  %1181 = select i1 %.not, float %1179, float %1173, !dbg !7
  %1182 = select i1 %.not, float %1175, float %1174, !dbg !7
  %1183 = bitcast float %1182 to i32, !dbg !7
  %1184 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1183), !dbg !7
  %1185 = bitcast i32 %1184 to float, !dbg !7
  %1186 = select i1 %.not, float %1174, float %1185, !dbg !7
  %1187 = select i1 %.not, float %1185, float %1175, !dbg !7
  %1188 = select i1 %.not18, float %1186, float %1180, !dbg !7
  %1189 = bitcast float %1188 to i32, !dbg !7
  %1190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1189), !dbg !7
  %1191 = bitcast i32 %1190 to float, !dbg !7
  %1192 = select i1 %.not18, float %1180, float %1191, !dbg !7
  %1193 = select i1 %.not18, float %1191, float %1186, !dbg !7
  %1194 = select i1 %.not18, float %1187, float %1181, !dbg !7
  %1195 = bitcast float %1194 to i32, !dbg !7
  %1196 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1195), !dbg !7
  %1197 = bitcast i32 %1196 to float, !dbg !7
  %1198 = select i1 %.not18, float %1181, float %1197, !dbg !7
  %1199 = select i1 %.not18, float %1197, float %1187, !dbg !7
  %1200 = extractvalue [2 x <8 x float>] %210, 0, !dbg !7
  %1201 = extractelement <8 x float> %1200, i64 0, !dbg !7
  %1202 = extractelement <8 x float> %1200, i64 1, !dbg !7
  %1203 = extractelement <8 x float> %1200, i64 2, !dbg !7
  %1204 = extractelement <8 x float> %1200, i64 3, !dbg !7
  %1205 = select i1 %.not, float %1202, float %1201, !dbg !7
  %1206 = bitcast float %1205 to i32, !dbg !7
  %1207 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1206), !dbg !7
  %1208 = bitcast i32 %1207 to float, !dbg !7
  %1209 = select i1 %.not, float %1201, float %1208, !dbg !7
  %1210 = select i1 %.not, float %1208, float %1202, !dbg !7
  %1211 = select i1 %.not, float %1204, float %1203, !dbg !7
  %1212 = bitcast float %1211 to i32, !dbg !7
  %1213 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1212), !dbg !7
  %1214 = bitcast i32 %1213 to float, !dbg !7
  %1215 = select i1 %.not, float %1203, float %1214, !dbg !7
  %1216 = select i1 %.not, float %1214, float %1204, !dbg !7
  %1217 = select i1 %.not18, float %1215, float %1209, !dbg !7
  %1218 = bitcast float %1217 to i32, !dbg !7
  %1219 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1218), !dbg !7
  %1220 = bitcast i32 %1219 to float, !dbg !7
  %1221 = select i1 %.not18, float %1209, float %1220, !dbg !7
  %1222 = select i1 %.not18, float %1220, float %1215, !dbg !7
  %1223 = select i1 %.not18, float %1216, float %1210, !dbg !7
  %1224 = bitcast float %1223 to i32, !dbg !7
  %1225 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1224), !dbg !7
  %1226 = bitcast i32 %1225 to float, !dbg !7
  %1227 = select i1 %.not18, float %1210, float %1226, !dbg !7
  %1228 = select i1 %.not18, float %1226, float %1216, !dbg !7
  %1229 = extractvalue [2 x <8 x float>] %207, 0, !dbg !7
  %1230 = extractelement <8 x float> %1229, i64 0, !dbg !7
  %1231 = extractelement <8 x float> %1229, i64 1, !dbg !7
  %1232 = extractelement <8 x float> %1229, i64 2, !dbg !7
  %1233 = extractelement <8 x float> %1229, i64 3, !dbg !7
  %1234 = select i1 %.not, float %1231, float %1230, !dbg !7
  %1235 = bitcast float %1234 to i32, !dbg !7
  %1236 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1235), !dbg !7
  %1237 = bitcast i32 %1236 to float, !dbg !7
  %1238 = select i1 %.not, float %1230, float %1237, !dbg !7
  %1239 = select i1 %.not, float %1237, float %1231, !dbg !7
  %1240 = select i1 %.not, float %1233, float %1232, !dbg !7
  %1241 = bitcast float %1240 to i32, !dbg !7
  %1242 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1241), !dbg !7
  %1243 = bitcast i32 %1242 to float, !dbg !7
  %1244 = select i1 %.not, float %1232, float %1243, !dbg !7
  %1245 = select i1 %.not, float %1243, float %1233, !dbg !7
  %1246 = select i1 %.not18, float %1244, float %1238, !dbg !7
  %1247 = bitcast float %1246 to i32, !dbg !7
  %1248 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1247), !dbg !7
  %1249 = select i1 %.not18, float %1245, float %1239, !dbg !7
  %1250 = bitcast float %1249 to i32, !dbg !7
  %1251 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1250), !dbg !7
  %1252 = extractvalue [2 x <1 x float>] %208, 0, !dbg !19
  %1253 = shufflevector <1 x float> %1252, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1254 = extractelement <8 x float> %1130, i64 4, !dbg !7
  %1255 = extractelement <8 x float> %1130, i64 5, !dbg !7
  %1256 = extractelement <8 x float> %1130, i64 6, !dbg !7
  %1257 = extractelement <8 x float> %1130, i64 7, !dbg !7
  %1258 = select i1 %.not, float %1255, float %1254, !dbg !7
  %1259 = bitcast float %1258 to i32, !dbg !7
  %1260 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1259), !dbg !7
  %1261 = bitcast i32 %1260 to float, !dbg !7
  %1262 = select i1 %.not, float %1254, float %1261, !dbg !7
  %1263 = select i1 %.not, float %1261, float %1255, !dbg !7
  %1264 = select i1 %.not, float %1257, float %1256, !dbg !7
  %1265 = bitcast float %1264 to i32, !dbg !7
  %1266 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1265), !dbg !7
  %1267 = bitcast i32 %1266 to float, !dbg !7
  %1268 = select i1 %.not, float %1256, float %1267, !dbg !7
  %1269 = select i1 %.not, float %1267, float %1257, !dbg !7
  %1270 = select i1 %.not18, float %1268, float %1262, !dbg !7
  %1271 = bitcast float %1270 to i32, !dbg !7
  %1272 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1271), !dbg !7
  %1273 = bitcast i32 %1272 to float, !dbg !7
  %1274 = select i1 %.not18, float %1262, float %1273, !dbg !7
  %1275 = select i1 %.not18, float %1273, float %1268, !dbg !7
  %1276 = select i1 %.not18, float %1269, float %1263, !dbg !7
  %1277 = bitcast float %1276 to i32, !dbg !7
  %1278 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1277), !dbg !7
  %1279 = bitcast i32 %1278 to float, !dbg !7
  %1280 = select i1 %.not18, float %1263, float %1279, !dbg !7
  %1281 = select i1 %.not18, float %1279, float %1269, !dbg !7
  %1282 = extractelement <8 x float> %1171, i64 4, !dbg !7
  %1283 = extractelement <8 x float> %1171, i64 5, !dbg !7
  %1284 = extractelement <8 x float> %1171, i64 6, !dbg !7
  %1285 = extractelement <8 x float> %1171, i64 7, !dbg !7
  %1286 = select i1 %.not, float %1283, float %1282, !dbg !7
  %1287 = bitcast float %1286 to i32, !dbg !7
  %1288 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1287), !dbg !7
  %1289 = bitcast i32 %1288 to float, !dbg !7
  %1290 = select i1 %.not, float %1282, float %1289, !dbg !7
  %1291 = select i1 %.not, float %1289, float %1283, !dbg !7
  %1292 = select i1 %.not, float %1285, float %1284, !dbg !7
  %1293 = bitcast float %1292 to i32, !dbg !7
  %1294 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1293), !dbg !7
  %1295 = bitcast i32 %1294 to float, !dbg !7
  %1296 = select i1 %.not, float %1284, float %1295, !dbg !7
  %1297 = select i1 %.not, float %1295, float %1285, !dbg !7
  %1298 = select i1 %.not18, float %1296, float %1290, !dbg !7
  %1299 = bitcast float %1298 to i32, !dbg !7
  %1300 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1299), !dbg !7
  %1301 = bitcast i32 %1300 to float, !dbg !7
  %1302 = select i1 %.not18, float %1290, float %1301, !dbg !7
  %1303 = select i1 %.not18, float %1301, float %1296, !dbg !7
  %1304 = select i1 %.not18, float %1297, float %1291, !dbg !7
  %1305 = bitcast float %1304 to i32, !dbg !7
  %1306 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1305), !dbg !7
  %1307 = bitcast i32 %1306 to float, !dbg !7
  %1308 = select i1 %.not18, float %1291, float %1307, !dbg !7
  %1309 = select i1 %.not18, float %1307, float %1297, !dbg !7
  %1310 = extractelement <8 x float> %1200, i64 4, !dbg !7
  %1311 = extractelement <8 x float> %1200, i64 5, !dbg !7
  %1312 = extractelement <8 x float> %1200, i64 6, !dbg !7
  %1313 = extractelement <8 x float> %1200, i64 7, !dbg !7
  %1314 = select i1 %.not, float %1311, float %1310, !dbg !7
  %1315 = bitcast float %1314 to i32, !dbg !7
  %1316 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1315), !dbg !7
  %1317 = bitcast i32 %1316 to float, !dbg !7
  %1318 = select i1 %.not, float %1310, float %1317, !dbg !7
  %1319 = select i1 %.not, float %1317, float %1311, !dbg !7
  %1320 = select i1 %.not, float %1313, float %1312, !dbg !7
  %1321 = bitcast float %1320 to i32, !dbg !7
  %1322 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1321), !dbg !7
  %1323 = bitcast i32 %1322 to float, !dbg !7
  %1324 = select i1 %.not, float %1312, float %1323, !dbg !7
  %1325 = select i1 %.not, float %1323, float %1313, !dbg !7
  %1326 = select i1 %.not18, float %1324, float %1318, !dbg !7
  %1327 = bitcast float %1326 to i32, !dbg !7
  %1328 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1327), !dbg !7
  %1329 = bitcast i32 %1328 to float, !dbg !7
  %1330 = select i1 %.not18, float %1318, float %1329, !dbg !7
  %1331 = select i1 %.not18, float %1329, float %1324, !dbg !7
  %1332 = select i1 %.not18, float %1325, float %1319, !dbg !7
  %1333 = bitcast float %1332 to i32, !dbg !7
  %1334 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1333), !dbg !7
  %1335 = bitcast i32 %1334 to float, !dbg !7
  %1336 = select i1 %.not18, float %1319, float %1335, !dbg !7
  %1337 = select i1 %.not18, float %1335, float %1325, !dbg !7
  %1338 = extractelement <8 x float> %1229, i64 4, !dbg !7
  %1339 = extractelement <8 x float> %1229, i64 5, !dbg !7
  %1340 = extractelement <8 x float> %1229, i64 6, !dbg !7
  %1341 = extractelement <8 x float> %1229, i64 7, !dbg !7
  %1342 = select i1 %.not, float %1339, float %1338, !dbg !7
  %1343 = bitcast float %1342 to i32, !dbg !7
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1343), !dbg !7
  %1345 = bitcast i32 %1344 to float, !dbg !7
  %1346 = select i1 %.not, float %1338, float %1345, !dbg !7
  %1347 = select i1 %.not, float %1345, float %1339, !dbg !7
  %1348 = select i1 %.not, float %1341, float %1340, !dbg !7
  %1349 = bitcast float %1348 to i32, !dbg !7
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1349), !dbg !7
  %1351 = bitcast i32 %1350 to float, !dbg !7
  %1352 = select i1 %.not, float %1340, float %1351, !dbg !7
  %1353 = select i1 %.not, float %1351, float %1341, !dbg !7
  %1354 = select i1 %.not18, float %1352, float %1346, !dbg !7
  %1355 = bitcast float %1354 to i32, !dbg !7
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1355), !dbg !7
  %1357 = select i1 %.not18, float %1353, float %1347, !dbg !7
  %1358 = bitcast float %1357 to i32, !dbg !7
  %1359 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1358), !dbg !7
  %1360 = extractvalue [2 x <8 x float>] %212, 1, !dbg !7
  %1361 = extractelement <8 x float> %1360, i64 0, !dbg !7
  %1362 = extractelement <8 x float> %1360, i64 1, !dbg !7
  %1363 = extractelement <8 x float> %1360, i64 2, !dbg !7
  %1364 = extractelement <8 x float> %1360, i64 3, !dbg !7
  %1365 = select i1 %.not, float %1362, float %1361, !dbg !7
  %1366 = bitcast float %1365 to i32, !dbg !7
  %1367 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1366), !dbg !7
  %1368 = bitcast i32 %1367 to float, !dbg !7
  %1369 = select i1 %.not, float %1361, float %1368, !dbg !7
  %1370 = select i1 %.not, float %1368, float %1362, !dbg !7
  %1371 = select i1 %.not, float %1364, float %1363, !dbg !7
  %1372 = bitcast float %1371 to i32, !dbg !7
  %1373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1372), !dbg !7
  %1374 = bitcast i32 %1373 to float, !dbg !7
  %1375 = select i1 %.not, float %1363, float %1374, !dbg !7
  %1376 = select i1 %.not, float %1374, float %1364, !dbg !7
  %1377 = select i1 %.not18, float %1375, float %1369, !dbg !7
  %1378 = bitcast float %1377 to i32, !dbg !7
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1378), !dbg !7
  %1380 = bitcast i32 %1379 to float, !dbg !7
  %1381 = select i1 %.not18, float %1369, float %1380, !dbg !7
  %1382 = select i1 %.not18, float %1380, float %1375, !dbg !7
  %1383 = select i1 %.not18, float %1376, float %1370, !dbg !7
  %1384 = bitcast float %1383 to i32, !dbg !7
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1384), !dbg !7
  %1386 = bitcast i32 %1385 to float, !dbg !7
  %1387 = select i1 %.not18, float %1370, float %1386, !dbg !7
  %1388 = select i1 %.not18, float %1386, float %1376, !dbg !7
  %1389 = extractvalue [2 x <8 x float>] %211, 1, !dbg !7
  %1390 = extractelement <8 x float> %1389, i64 0, !dbg !7
  %1391 = extractelement <8 x float> %1389, i64 1, !dbg !7
  %1392 = extractelement <8 x float> %1389, i64 2, !dbg !7
  %1393 = extractelement <8 x float> %1389, i64 3, !dbg !7
  %1394 = select i1 %.not, float %1391, float %1390, !dbg !7
  %1395 = bitcast float %1394 to i32, !dbg !7
  %1396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1395), !dbg !7
  %1397 = bitcast i32 %1396 to float, !dbg !7
  %1398 = select i1 %.not, float %1390, float %1397, !dbg !7
  %1399 = select i1 %.not, float %1397, float %1391, !dbg !7
  %1400 = select i1 %.not, float %1393, float %1392, !dbg !7
  %1401 = bitcast float %1400 to i32, !dbg !7
  %1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1401), !dbg !7
  %1403 = bitcast i32 %1402 to float, !dbg !7
  %1404 = select i1 %.not, float %1392, float %1403, !dbg !7
  %1405 = select i1 %.not, float %1403, float %1393, !dbg !7
  %1406 = select i1 %.not18, float %1404, float %1398, !dbg !7
  %1407 = bitcast float %1406 to i32, !dbg !7
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1407), !dbg !7
  %1409 = bitcast i32 %1408 to float, !dbg !7
  %1410 = select i1 %.not18, float %1398, float %1409, !dbg !7
  %1411 = select i1 %.not18, float %1409, float %1404, !dbg !7
  %1412 = select i1 %.not18, float %1405, float %1399, !dbg !7
  %1413 = bitcast float %1412 to i32, !dbg !7
  %1414 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1413), !dbg !7
  %1415 = bitcast i32 %1414 to float, !dbg !7
  %1416 = select i1 %.not18, float %1399, float %1415, !dbg !7
  %1417 = select i1 %.not18, float %1415, float %1405, !dbg !7
  %1418 = extractvalue [2 x <8 x float>] %210, 1, !dbg !7
  %1419 = extractelement <8 x float> %1418, i64 0, !dbg !7
  %1420 = extractelement <8 x float> %1418, i64 1, !dbg !7
  %1421 = extractelement <8 x float> %1418, i64 2, !dbg !7
  %1422 = extractelement <8 x float> %1418, i64 3, !dbg !7
  %1423 = select i1 %.not, float %1420, float %1419, !dbg !7
  %1424 = bitcast float %1423 to i32, !dbg !7
  %1425 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1424), !dbg !7
  %1426 = bitcast i32 %1425 to float, !dbg !7
  %1427 = select i1 %.not, float %1419, float %1426, !dbg !7
  %1428 = select i1 %.not, float %1426, float %1420, !dbg !7
  %1429 = select i1 %.not, float %1422, float %1421, !dbg !7
  %1430 = bitcast float %1429 to i32, !dbg !7
  %1431 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1430), !dbg !7
  %1432 = bitcast i32 %1431 to float, !dbg !7
  %1433 = select i1 %.not, float %1421, float %1432, !dbg !7
  %1434 = select i1 %.not, float %1432, float %1422, !dbg !7
  %1435 = select i1 %.not18, float %1433, float %1427, !dbg !7
  %1436 = bitcast float %1435 to i32, !dbg !7
  %1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1436), !dbg !7
  %1438 = bitcast i32 %1437 to float, !dbg !7
  %1439 = select i1 %.not18, float %1427, float %1438, !dbg !7
  %1440 = select i1 %.not18, float %1438, float %1433, !dbg !7
  %1441 = select i1 %.not18, float %1434, float %1428, !dbg !7
  %1442 = bitcast float %1441 to i32, !dbg !7
  %1443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1442), !dbg !7
  %1444 = bitcast i32 %1443 to float, !dbg !7
  %1445 = select i1 %.not18, float %1428, float %1444, !dbg !7
  %1446 = select i1 %.not18, float %1444, float %1434, !dbg !7
  %1447 = extractvalue [2 x <8 x float>] %207, 1, !dbg !7
  %1448 = extractelement <8 x float> %1447, i64 0, !dbg !7
  %1449 = extractelement <8 x float> %1447, i64 1, !dbg !7
  %1450 = extractelement <8 x float> %1447, i64 2, !dbg !7
  %1451 = extractelement <8 x float> %1447, i64 3, !dbg !7
  %1452 = select i1 %.not, float %1449, float %1448, !dbg !7
  %1453 = bitcast float %1452 to i32, !dbg !7
  %1454 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1453), !dbg !7
  %1455 = bitcast i32 %1454 to float, !dbg !7
  %1456 = select i1 %.not, float %1448, float %1455, !dbg !7
  %1457 = select i1 %.not, float %1455, float %1449, !dbg !7
  %1458 = select i1 %.not, float %1451, float %1450, !dbg !7
  %1459 = bitcast float %1458 to i32, !dbg !7
  %1460 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1459), !dbg !7
  %1461 = bitcast i32 %1460 to float, !dbg !7
  %1462 = select i1 %.not, float %1450, float %1461, !dbg !7
  %1463 = select i1 %.not, float %1461, float %1451, !dbg !7
  %1464 = select i1 %.not18, float %1462, float %1456, !dbg !7
  %1465 = bitcast float %1464 to i32, !dbg !7
  %1466 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1465), !dbg !7
  %1467 = select i1 %.not18, float %1463, float %1457, !dbg !7
  %1468 = bitcast float %1467 to i32, !dbg !7
  %1469 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1468), !dbg !7
  %1470 = extractvalue [2 x <1 x float>] %208, 1, !dbg !19
  %1471 = shufflevector <1 x float> %1470, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1472 = extractelement <8 x float> %1360, i64 4, !dbg !7
  %1473 = extractelement <8 x float> %1360, i64 5, !dbg !7
  %1474 = extractelement <8 x float> %1360, i64 6, !dbg !7
  %1475 = extractelement <8 x float> %1360, i64 7, !dbg !7
  %1476 = select i1 %.not, float %1473, float %1472, !dbg !7
  %1477 = bitcast float %1476 to i32, !dbg !7
  %1478 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1477), !dbg !7
  %1479 = bitcast i32 %1478 to float, !dbg !7
  %1480 = select i1 %.not, float %1472, float %1479, !dbg !7
  %1481 = select i1 %.not, float %1479, float %1473, !dbg !7
  %1482 = select i1 %.not, float %1475, float %1474, !dbg !7
  %1483 = bitcast float %1482 to i32, !dbg !7
  %1484 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1483), !dbg !7
  %1485 = bitcast i32 %1484 to float, !dbg !7
  %1486 = select i1 %.not, float %1474, float %1485, !dbg !7
  %1487 = select i1 %.not, float %1485, float %1475, !dbg !7
  %1488 = select i1 %.not18, float %1486, float %1480, !dbg !7
  %1489 = bitcast float %1488 to i32, !dbg !7
  %1490 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1489), !dbg !7
  %1491 = bitcast i32 %1490 to float, !dbg !7
  %1492 = select i1 %.not18, float %1480, float %1491, !dbg !7
  %1493 = select i1 %.not18, float %1491, float %1486, !dbg !7
  %1494 = select i1 %.not18, float %1487, float %1481, !dbg !7
  %1495 = bitcast float %1494 to i32, !dbg !7
  %1496 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1495), !dbg !7
  %1497 = bitcast i32 %1496 to float, !dbg !7
  %1498 = select i1 %.not18, float %1481, float %1497, !dbg !7
  %1499 = select i1 %.not18, float %1497, float %1487, !dbg !7
  %1500 = extractelement <8 x float> %1389, i64 4, !dbg !7
  %1501 = extractelement <8 x float> %1389, i64 5, !dbg !7
  %1502 = extractelement <8 x float> %1389, i64 6, !dbg !7
  %1503 = extractelement <8 x float> %1389, i64 7, !dbg !7
  %1504 = select i1 %.not, float %1501, float %1500, !dbg !7
  %1505 = bitcast float %1504 to i32, !dbg !7
  %1506 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1505), !dbg !7
  %1507 = bitcast i32 %1506 to float, !dbg !7
  %1508 = select i1 %.not, float %1500, float %1507, !dbg !7
  %1509 = select i1 %.not, float %1507, float %1501, !dbg !7
  %1510 = select i1 %.not, float %1503, float %1502, !dbg !7
  %1511 = bitcast float %1510 to i32, !dbg !7
  %1512 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1511), !dbg !7
  %1513 = bitcast i32 %1512 to float, !dbg !7
  %1514 = select i1 %.not, float %1502, float %1513, !dbg !7
  %1515 = select i1 %.not, float %1513, float %1503, !dbg !7
  %1516 = select i1 %.not18, float %1514, float %1508, !dbg !7
  %1517 = bitcast float %1516 to i32, !dbg !7
  %1518 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1517), !dbg !7
  %1519 = bitcast i32 %1518 to float, !dbg !7
  %1520 = select i1 %.not18, float %1508, float %1519, !dbg !7
  %1521 = select i1 %.not18, float %1519, float %1514, !dbg !7
  %1522 = select i1 %.not18, float %1515, float %1509, !dbg !7
  %1523 = bitcast float %1522 to i32, !dbg !7
  %1524 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1523), !dbg !7
  %1525 = bitcast i32 %1524 to float, !dbg !7
  %1526 = select i1 %.not18, float %1509, float %1525, !dbg !7
  %1527 = select i1 %.not18, float %1525, float %1515, !dbg !7
  %1528 = extractelement <8 x float> %1418, i64 4, !dbg !7
  %1529 = extractelement <8 x float> %1418, i64 5, !dbg !7
  %1530 = extractelement <8 x float> %1418, i64 6, !dbg !7
  %1531 = extractelement <8 x float> %1418, i64 7, !dbg !7
  %1532 = select i1 %.not, float %1529, float %1528, !dbg !7
  %1533 = bitcast float %1532 to i32, !dbg !7
  %1534 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1533), !dbg !7
  %1535 = bitcast i32 %1534 to float, !dbg !7
  %1536 = select i1 %.not, float %1528, float %1535, !dbg !7
  %1537 = select i1 %.not, float %1535, float %1529, !dbg !7
  %1538 = select i1 %.not, float %1531, float %1530, !dbg !7
  %1539 = bitcast float %1538 to i32, !dbg !7
  %1540 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1539), !dbg !7
  %1541 = bitcast i32 %1540 to float, !dbg !7
  %1542 = select i1 %.not, float %1530, float %1541, !dbg !7
  %1543 = select i1 %.not, float %1541, float %1531, !dbg !7
  %1544 = select i1 %.not18, float %1542, float %1536, !dbg !7
  %1545 = bitcast float %1544 to i32, !dbg !7
  %1546 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1545), !dbg !7
  %1547 = bitcast i32 %1546 to float, !dbg !7
  %1548 = select i1 %.not18, float %1536, float %1547, !dbg !7
  %1549 = select i1 %.not18, float %1547, float %1542, !dbg !7
  %1550 = select i1 %.not18, float %1543, float %1537, !dbg !7
  %1551 = bitcast float %1550 to i32, !dbg !7
  %1552 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1551), !dbg !7
  %1553 = bitcast i32 %1552 to float, !dbg !7
  %1554 = select i1 %.not18, float %1537, float %1553, !dbg !7
  %1555 = select i1 %.not18, float %1553, float %1543, !dbg !7
  %1556 = extractelement <8 x float> %1447, i64 4, !dbg !7
  %1557 = extractelement <8 x float> %1447, i64 5, !dbg !7
  %1558 = extractelement <8 x float> %1447, i64 6, !dbg !7
  %1559 = extractelement <8 x float> %1447, i64 7, !dbg !7
  %1560 = select i1 %.not, float %1557, float %1556, !dbg !7
  %1561 = bitcast float %1560 to i32, !dbg !7
  %1562 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1561), !dbg !7
  %1563 = bitcast i32 %1562 to float, !dbg !7
  %1564 = select i1 %.not, float %1556, float %1563, !dbg !7
  %1565 = select i1 %.not, float %1563, float %1557, !dbg !7
  %1566 = select i1 %.not, float %1559, float %1558, !dbg !7
  %1567 = bitcast float %1566 to i32, !dbg !7
  %1568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1143, i32 %1567), !dbg !7
  %1569 = bitcast i32 %1568 to float, !dbg !7
  %1570 = select i1 %.not, float %1558, float %1569, !dbg !7
  %1571 = select i1 %.not, float %1569, float %1559, !dbg !7
  %1572 = select i1 %.not18, float %1570, float %1564, !dbg !7
  %1573 = bitcast float %1572 to i32, !dbg !7
  %1574 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1573), !dbg !7
  %1575 = select i1 %.not18, float %1571, float %1565, !dbg !7
  %1576 = bitcast float %1575 to i32, !dbg !7
  %1577 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1576), !dbg !7
  br i1 true, label %1578, label %1642, !dbg !21

1578:                                             ; preds = %1127
  %1579 = bitcast i32 %1574 to float, !dbg !7
  %1580 = select i1 %.not18, float %1564, float %1579, !dbg !7
  %1581 = insertelement <4 x float> poison, float %1580, i64 0, !dbg !19
  %1582 = bitcast i32 %1577 to float, !dbg !7
  %1583 = select i1 %.not18, float %1565, float %1582, !dbg !7
  %1584 = insertelement <4 x float> %1581, float %1583, i64 1, !dbg !19
  %1585 = select i1 %.not18, float %1579, float %1570, !dbg !7
  %1586 = insertelement <4 x float> %1584, float %1585, i64 2, !dbg !19
  %1587 = select i1 %.not18, float %1582, float %1571, !dbg !7
  %1588 = insertelement <4 x float> %1586, float %1587, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1471, !dbg !19
  %normalized = fmul <4 x float> %1588, %row.reciprocal, !dbg !19
  %1589 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1590 = extractelement <4 x half> %1589, i64 3, !dbg !20
  %1591 = extractelement <4 x half> %1589, i64 2, !dbg !20
  %1592 = extractelement <4 x half> %1589, i64 1, !dbg !20
  %1593 = extractelement <4 x half> %1589, i64 0, !dbg !20
  %1594 = bitcast i32 %1466 to float, !dbg !7
  %1595 = select i1 %.not18, float %1456, float %1594, !dbg !7
  %1596 = insertelement <4 x float> poison, float %1595, i64 0, !dbg !19
  %1597 = bitcast i32 %1469 to float, !dbg !7
  %1598 = select i1 %.not18, float %1457, float %1597, !dbg !7
  %1599 = insertelement <4 x float> %1596, float %1598, i64 1, !dbg !19
  %1600 = select i1 %.not18, float %1594, float %1462, !dbg !7
  %1601 = insertelement <4 x float> %1599, float %1600, i64 2, !dbg !19
  %1602 = select i1 %.not18, float %1597, float %1463, !dbg !7
  %1603 = insertelement <4 x float> %1601, float %1602, i64 3, !dbg !19
  %normalized54 = fmul <4 x float> %1603, %row.reciprocal, !dbg !19
  %1604 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !20
  %1605 = extractelement <4 x half> %1604, i64 3, !dbg !20
  %1606 = extractelement <4 x half> %1604, i64 2, !dbg !20
  %1607 = extractelement <4 x half> %1604, i64 1, !dbg !20
  %1608 = extractelement <4 x half> %1604, i64 0, !dbg !20
  %1609 = bitcast i32 %1356 to float, !dbg !7
  %1610 = select i1 %.not18, float %1346, float %1609, !dbg !7
  %1611 = insertelement <4 x float> poison, float %1610, i64 0, !dbg !19
  %1612 = bitcast i32 %1359 to float, !dbg !7
  %1613 = select i1 %.not18, float %1347, float %1612, !dbg !7
  %1614 = insertelement <4 x float> %1611, float %1613, i64 1, !dbg !19
  %1615 = select i1 %.not18, float %1609, float %1352, !dbg !7
  %1616 = insertelement <4 x float> %1614, float %1615, i64 2, !dbg !19
  %1617 = select i1 %.not18, float %1612, float %1353, !dbg !7
  %1618 = insertelement <4 x float> %1616, float %1617, i64 3, !dbg !19
  %row.reciprocal55 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1253, !dbg !19
  %normalized56 = fmul <4 x float> %1618, %row.reciprocal55, !dbg !19
  %1619 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !20
  %1620 = extractelement <4 x half> %1619, i64 3, !dbg !20
  %1621 = extractelement <4 x half> %1619, i64 2, !dbg !20
  %1622 = extractelement <4 x half> %1619, i64 1, !dbg !20
  %1623 = extractelement <4 x half> %1619, i64 0, !dbg !20
  %1624 = bitcast i32 %1248 to float, !dbg !7
  %1625 = select i1 %.not18, float %1238, float %1624, !dbg !7
  %1626 = insertelement <4 x float> poison, float %1625, i64 0, !dbg !19
  %1627 = bitcast i32 %1251 to float, !dbg !7
  %1628 = select i1 %.not18, float %1239, float %1627, !dbg !7
  %1629 = insertelement <4 x float> %1626, float %1628, i64 1, !dbg !19
  %1630 = select i1 %.not18, float %1624, float %1244, !dbg !7
  %1631 = insertelement <4 x float> %1629, float %1630, i64 2, !dbg !19
  %1632 = select i1 %.not18, float %1627, float %1245, !dbg !7
  %1633 = insertelement <4 x float> %1631, float %1632, i64 3, !dbg !19
  %normalized57 = fmul <4 x float> %1633, %row.reciprocal55, !dbg !19
  %1634 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !20
  %1635 = extractelement <4 x half> %1634, i64 3, !dbg !20
  %1636 = extractelement <4 x half> %1634, i64 2, !dbg !20
  %1637 = extractelement <4 x half> %1634, i64 1, !dbg !20
  %1638 = extractelement <4 x half> %1634, i64 0, !dbg !20
  %1639 = add nsw i64 %16, %22, !dbg !21
  %1640 = shl i64 %1639, 19, !dbg !21
  %1641 = and i64 %1640, 16252928, !dbg !21
  %address.thread534 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local535 = zext nneg i32 %address.thread534 to i64, !dbg !9
  %address.local536 = add i64 %25, %address.local535, !dbg !21
  %address.local537 = lshr i64 %address.local535, 6, !dbg !8
  %address.local538 = shl nuw nsw i64 %address.local537, 4, !dbg !21
  %address.local539 = add i64 %address.local536, %address.local538, !dbg !21
  %address.local540 = and i64 %address.local535, 240, !dbg !21
  %address.local541 = sub i64 %address.local539, %address.local540, !dbg !21
  %address.local542 = add i64 %address.local541, %26, !dbg !21
  %address.local543 = shl nuw nsw i64 %address.local542, 7, !dbg !21
  %address.local544 = add nuw nsw i64 %1641, %address.local543, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local546 = lshr i64 %address.local535, 2, !dbg !8
  %address.local547 = and i64 %address.local546, 12, !dbg !8
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local547, !dbg !21
  store half %1638, ptr addrspace(1) %address.local548, align 2, !dbg !21
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local550 = or disjoint i64 %address.local547, 1, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local549, i64 %address.local550, !dbg !21
  store half %1637, ptr addrspace(1) %address.local551, align 2, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local553 = or disjoint i64 %address.local547, 2, !dbg !21
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local552, i64 %address.local553, !dbg !21
  store half %1636, ptr addrspace(1) %address.local554, align 2, !dbg !21
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local556 = or disjoint i64 %address.local547, 3, !dbg !21
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local555, i64 %address.local556, !dbg !21
  store half %1635, ptr addrspace(1) %address.local557, align 2, !dbg !21
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local559 = or disjoint i64 %address.local547, 16, !dbg !21
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local558, i64 %address.local559, !dbg !21
  store half %1623, ptr addrspace(1) %address.local560, align 2, !dbg !21
  %address.local561 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local562 = or disjoint i64 %address.local547, 17, !dbg !21
  %address.local563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local561, i64 %address.local562, !dbg !21
  store half %1622, ptr addrspace(1) %address.local563, align 2, !dbg !21
  %address.local564 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local565 = or disjoint i64 %address.local547, 18, !dbg !21
  %address.local566 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local564, i64 %address.local565, !dbg !21
  store half %1621, ptr addrspace(1) %address.local566, align 2, !dbg !21
  %address.local567 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local568 = or disjoint i64 %address.local547, 19, !dbg !21
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local567, i64 %address.local568, !dbg !21
  store half %1620, ptr addrspace(1) %address.local569, align 2, !dbg !21
  %address.local570 = shl i64 %address.local542, 7, !dbg !21
  %address.local571 = add i64 %address.local570, 8192, !dbg !21
  %address.local572 = add nuw nsw i64 %1641, %address.local571, !dbg !21
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local573, i64 %address.local547, !dbg !21
  store half %1608, ptr addrspace(1) %address.local574, align 2, !dbg !21
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local550, !dbg !21
  store half %1607, ptr addrspace(1) %address.local576, align 2, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local553, !dbg !21
  store half %1606, ptr addrspace(1) %address.local578, align 2, !dbg !21
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local579, i64 %address.local556, !dbg !21
  store half %1605, ptr addrspace(1) %address.local580, align 2, !dbg !21
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local559, !dbg !21
  store half %1593, ptr addrspace(1) %address.local582, align 2, !dbg !21
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local562, !dbg !21
  store half %1592, ptr addrspace(1) %address.local584, align 2, !dbg !21
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local585, i64 %address.local565, !dbg !21
  store half %1591, ptr addrspace(1) %address.local586, align 2, !dbg !21
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local587, i64 %address.local568, !dbg !21
  store half %1590, ptr addrspace(1) %address.local588, align 2, !dbg !21
  br label %1642, !dbg !21

1642:                                             ; preds = %1578, %1127
  br i1 true, label %1643, label %1681, !dbg !21

1643:                                             ; preds = %1642
  %1644 = insertelement <4 x float> poison, float %1548, i64 0, !dbg !19
  %1645 = insertelement <4 x float> %1644, float %1554, i64 1, !dbg !19
  %1646 = insertelement <4 x float> %1645, float %1549, i64 2, !dbg !19
  %1647 = insertelement <4 x float> %1646, float %1555, i64 3, !dbg !19
  %row.reciprocal58 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1471, !dbg !19
  %normalized59 = fmul <4 x float> %1647, %row.reciprocal58, !dbg !19
  %1648 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !20
  %1649 = extractelement <4 x half> %1648, i64 3, !dbg !20
  %1650 = extractelement <4 x half> %1648, i64 2, !dbg !20
  %1651 = extractelement <4 x half> %1648, i64 1, !dbg !20
  %1652 = extractelement <4 x half> %1648, i64 0, !dbg !20
  %1653 = insertelement <4 x float> poison, float %1439, i64 0, !dbg !19
  %1654 = insertelement <4 x float> %1653, float %1445, i64 1, !dbg !19
  %1655 = insertelement <4 x float> %1654, float %1440, i64 2, !dbg !19
  %1656 = insertelement <4 x float> %1655, float %1446, i64 3, !dbg !19
  %normalized60 = fmul <4 x float> %1656, %row.reciprocal58, !dbg !19
  %1657 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !20
  %1658 = extractelement <4 x half> %1657, i64 3, !dbg !20
  %1659 = extractelement <4 x half> %1657, i64 2, !dbg !20
  %1660 = extractelement <4 x half> %1657, i64 1, !dbg !20
  %1661 = extractelement <4 x half> %1657, i64 0, !dbg !20
  %1662 = insertelement <4 x float> poison, float %1330, i64 0, !dbg !19
  %1663 = insertelement <4 x float> %1662, float %1336, i64 1, !dbg !19
  %1664 = insertelement <4 x float> %1663, float %1331, i64 2, !dbg !19
  %1665 = insertelement <4 x float> %1664, float %1337, i64 3, !dbg !19
  %row.reciprocal61 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1253, !dbg !19
  %normalized62 = fmul <4 x float> %1665, %row.reciprocal61, !dbg !19
  %1666 = fptrunc <4 x float> %normalized62 to <4 x half>, !dbg !20
  %1667 = extractelement <4 x half> %1666, i64 3, !dbg !20
  %1668 = extractelement <4 x half> %1666, i64 2, !dbg !20
  %1669 = extractelement <4 x half> %1666, i64 1, !dbg !20
  %1670 = extractelement <4 x half> %1666, i64 0, !dbg !20
  %1671 = insertelement <4 x float> poison, float %1221, i64 0, !dbg !19
  %1672 = insertelement <4 x float> %1671, float %1227, i64 1, !dbg !19
  %1673 = insertelement <4 x float> %1672, float %1222, i64 2, !dbg !19
  %1674 = insertelement <4 x float> %1673, float %1228, i64 3, !dbg !19
  %normalized63 = fmul <4 x float> %1674, %row.reciprocal61, !dbg !19
  %1675 = fptrunc <4 x float> %normalized63 to <4 x half>, !dbg !20
  %1676 = extractelement <4 x half> %1675, i64 3, !dbg !20
  %1677 = extractelement <4 x half> %1675, i64 2, !dbg !20
  %1678 = extractelement <4 x half> %1675, i64 1, !dbg !20
  %1679 = extractelement <4 x half> %1675, i64 0, !dbg !20
  %1680 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread589 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local590 = zext nneg i32 %address.thread589 to i64, !dbg !9
  %address.local591 = add i64 %25, %address.local590, !dbg !21
  %address.local592 = lshr i64 %address.local590, 6, !dbg !8
  %address.local593 = shl nuw nsw i64 %address.local592, 4, !dbg !21
  %address.local594 = add i64 %address.local591, %address.local593, !dbg !21
  %address.local595 = and i64 %address.local590, 240, !dbg !21
  %address.local596 = sub i64 %address.local594, %address.local595, !dbg !21
  %address.local597 = add i64 %address.local596, %26, !dbg !21
  %address.local598 = shl nuw nsw i64 %address.local597, 7, !dbg !21
  %address.local599 = add nuw nsw i64 %1680, %address.local598, !dbg !21
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local601 = lshr i64 %address.local590, 2, !dbg !8
  %address.local602 = and i64 %address.local601, 12, !dbg !8
  %address.local603 = or disjoint i64 %address.local602, 32, !dbg !21
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local603, !dbg !21
  store half %1679, ptr addrspace(1) %address.local604, align 2, !dbg !21
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local606 = or disjoint i64 %address.local602, 33, !dbg !21
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local605, i64 %address.local606, !dbg !21
  store half %1678, ptr addrspace(1) %address.local607, align 2, !dbg !21
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local609 = or disjoint i64 %address.local602, 34, !dbg !21
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local608, i64 %address.local609, !dbg !21
  store half %1677, ptr addrspace(1) %address.local610, align 2, !dbg !21
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local612 = or disjoint i64 %address.local602, 35, !dbg !21
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local611, i64 %address.local612, !dbg !21
  store half %1676, ptr addrspace(1) %address.local613, align 2, !dbg !21
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local615 = or disjoint i64 %address.local602, 48, !dbg !21
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local614, i64 %address.local615, !dbg !21
  store half %1670, ptr addrspace(1) %address.local616, align 2, !dbg !21
  %address.local617 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local618 = or disjoint i64 %address.local602, 49, !dbg !21
  %address.local619 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local617, i64 %address.local618, !dbg !21
  store half %1669, ptr addrspace(1) %address.local619, align 2, !dbg !21
  %address.local620 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local621 = or disjoint i64 %address.local602, 50, !dbg !21
  %address.local622 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local621, !dbg !21
  store half %1668, ptr addrspace(1) %address.local622, align 2, !dbg !21
  %address.local623 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local624 = or i64 %address.local601, 51, !dbg !21
  %address.local625 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local623, i64 %address.local624, !dbg !21
  store half %1667, ptr addrspace(1) %address.local625, align 2, !dbg !21
  %address.local626 = shl i64 %address.local597, 7, !dbg !21
  %address.local627 = add i64 %address.local626, 8192, !dbg !21
  %address.local628 = add nuw nsw i64 %1680, %address.local627, !dbg !21
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local629, i64 %address.local603, !dbg !21
  store half %1661, ptr addrspace(1) %address.local630, align 2, !dbg !21
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local606, !dbg !21
  store half %1660, ptr addrspace(1) %address.local632, align 2, !dbg !21
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local633, i64 %address.local609, !dbg !21
  store half %1659, ptr addrspace(1) %address.local634, align 2, !dbg !21
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local635, i64 %address.local612, !dbg !21
  store half %1658, ptr addrspace(1) %address.local636, align 2, !dbg !21
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local637, i64 %address.local615, !dbg !21
  store half %1652, ptr addrspace(1) %address.local638, align 2, !dbg !21
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local639, i64 %address.local618, !dbg !21
  store half %1651, ptr addrspace(1) %address.local640, align 2, !dbg !21
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local642 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local641, i64 %address.local621, !dbg !21
  store half %1650, ptr addrspace(1) %address.local642, align 2, !dbg !21
  %address.local643 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local624, !dbg !21
  store half %1649, ptr addrspace(1) %address.local644, align 2, !dbg !21
  br label %1681, !dbg !21

1681:                                             ; preds = %1643, %1642
  br i1 true, label %1682, label %1720, !dbg !21

1682:                                             ; preds = %1681
  %1683 = insertelement <4 x float> poison, float %1520, i64 0, !dbg !19
  %1684 = insertelement <4 x float> %1683, float %1526, i64 1, !dbg !19
  %1685 = insertelement <4 x float> %1684, float %1521, i64 2, !dbg !19
  %1686 = insertelement <4 x float> %1685, float %1527, i64 3, !dbg !19
  %row.reciprocal64 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1471, !dbg !19
  %normalized65 = fmul <4 x float> %1686, %row.reciprocal64, !dbg !19
  %1687 = fptrunc <4 x float> %normalized65 to <4 x half>, !dbg !20
  %1688 = extractelement <4 x half> %1687, i64 3, !dbg !20
  %1689 = extractelement <4 x half> %1687, i64 2, !dbg !20
  %1690 = extractelement <4 x half> %1687, i64 1, !dbg !20
  %1691 = extractelement <4 x half> %1687, i64 0, !dbg !20
  %1692 = insertelement <4 x float> poison, float %1410, i64 0, !dbg !19
  %1693 = insertelement <4 x float> %1692, float %1416, i64 1, !dbg !19
  %1694 = insertelement <4 x float> %1693, float %1411, i64 2, !dbg !19
  %1695 = insertelement <4 x float> %1694, float %1417, i64 3, !dbg !19
  %normalized66 = fmul <4 x float> %1695, %row.reciprocal64, !dbg !19
  %1696 = fptrunc <4 x float> %normalized66 to <4 x half>, !dbg !20
  %1697 = extractelement <4 x half> %1696, i64 3, !dbg !20
  %1698 = extractelement <4 x half> %1696, i64 2, !dbg !20
  %1699 = extractelement <4 x half> %1696, i64 1, !dbg !20
  %1700 = extractelement <4 x half> %1696, i64 0, !dbg !20
  %1701 = insertelement <4 x float> poison, float %1302, i64 0, !dbg !19
  %1702 = insertelement <4 x float> %1701, float %1308, i64 1, !dbg !19
  %1703 = insertelement <4 x float> %1702, float %1303, i64 2, !dbg !19
  %1704 = insertelement <4 x float> %1703, float %1309, i64 3, !dbg !19
  %row.reciprocal67 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1253, !dbg !19
  %normalized68 = fmul <4 x float> %1704, %row.reciprocal67, !dbg !19
  %1705 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !20
  %1706 = extractelement <4 x half> %1705, i64 3, !dbg !20
  %1707 = extractelement <4 x half> %1705, i64 2, !dbg !20
  %1708 = extractelement <4 x half> %1705, i64 1, !dbg !20
  %1709 = extractelement <4 x half> %1705, i64 0, !dbg !20
  %1710 = insertelement <4 x float> poison, float %1192, i64 0, !dbg !19
  %1711 = insertelement <4 x float> %1710, float %1198, i64 1, !dbg !19
  %1712 = insertelement <4 x float> %1711, float %1193, i64 2, !dbg !19
  %1713 = insertelement <4 x float> %1712, float %1199, i64 3, !dbg !19
  %normalized69 = fmul <4 x float> %1713, %row.reciprocal67, !dbg !19
  %1714 = fptrunc <4 x float> %normalized69 to <4 x half>, !dbg !20
  %1715 = extractelement <4 x half> %1714, i64 3, !dbg !20
  %1716 = extractelement <4 x half> %1714, i64 2, !dbg !20
  %1717 = extractelement <4 x half> %1714, i64 1, !dbg !20
  %1718 = extractelement <4 x half> %1714, i64 0, !dbg !20
  %1719 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread645 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local646 = zext nneg i32 %address.thread645 to i64, !dbg !9
  %address.local647 = add i64 %25, %address.local646, !dbg !21
  %address.local648 = lshr i64 %address.local646, 6, !dbg !8
  %address.local649 = shl nuw nsw i64 %address.local648, 4, !dbg !21
  %address.local650 = add i64 %address.local647, %address.local649, !dbg !21
  %address.local651 = and i64 %address.local646, 240, !dbg !21
  %address.local652 = sub i64 %address.local650, %address.local651, !dbg !21
  %address.local653 = add i64 %address.local652, %26, !dbg !21
  %address.local654 = shl nuw nsw i64 %address.local653, 7, !dbg !21
  %address.local655 = add nuw nsw i64 %1719, %address.local654, !dbg !21
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local657 = lshr i64 %address.local646, 2, !dbg !8
  %address.local658 = and i64 %address.local657, 12, !dbg !8
  %address.local659 = or disjoint i64 %address.local658, 64, !dbg !21
  %address.local660 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local656, i64 %address.local659, !dbg !21
  store half %1718, ptr addrspace(1) %address.local660, align 2, !dbg !21
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local662 = or disjoint i64 %address.local658, 65, !dbg !21
  %address.local663 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local661, i64 %address.local662, !dbg !21
  store half %1717, ptr addrspace(1) %address.local663, align 2, !dbg !21
  %address.local664 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local665 = or disjoint i64 %address.local658, 66, !dbg !21
  %address.local666 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local664, i64 %address.local665, !dbg !21
  store half %1716, ptr addrspace(1) %address.local666, align 2, !dbg !21
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local668 = or disjoint i64 %address.local658, 67, !dbg !21
  %address.local669 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local667, i64 %address.local668, !dbg !21
  store half %1715, ptr addrspace(1) %address.local669, align 2, !dbg !21
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local671 = or disjoint i64 %address.local658, 80, !dbg !21
  %address.local672 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local670, i64 %address.local671, !dbg !21
  store half %1709, ptr addrspace(1) %address.local672, align 2, !dbg !21
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local674 = or disjoint i64 %address.local658, 81, !dbg !21
  %address.local675 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local673, i64 %address.local674, !dbg !21
  store half %1708, ptr addrspace(1) %address.local675, align 2, !dbg !21
  %address.local676 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local677 = or disjoint i64 %address.local658, 82, !dbg !21
  %address.local678 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local676, i64 %address.local677, !dbg !21
  store half %1707, ptr addrspace(1) %address.local678, align 2, !dbg !21
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local680 = or disjoint i64 %address.local658, 83, !dbg !21
  %address.local681 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local679, i64 %address.local680, !dbg !21
  store half %1706, ptr addrspace(1) %address.local681, align 2, !dbg !21
  %address.local682 = shl i64 %address.local653, 7, !dbg !21
  %address.local683 = add i64 %address.local682, 8192, !dbg !21
  %address.local684 = add nuw nsw i64 %1719, %address.local683, !dbg !21
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local685, i64 %address.local659, !dbg !21
  store half %1700, ptr addrspace(1) %address.local686, align 2, !dbg !21
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local687, i64 %address.local662, !dbg !21
  store half %1699, ptr addrspace(1) %address.local688, align 2, !dbg !21
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local689, i64 %address.local665, !dbg !21
  store half %1698, ptr addrspace(1) %address.local690, align 2, !dbg !21
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local691, i64 %address.local668, !dbg !21
  store half %1697, ptr addrspace(1) %address.local692, align 2, !dbg !21
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local693, i64 %address.local671, !dbg !21
  store half %1691, ptr addrspace(1) %address.local694, align 2, !dbg !21
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local695, i64 %address.local674, !dbg !21
  store half %1690, ptr addrspace(1) %address.local696, align 2, !dbg !21
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local697, i64 %address.local677, !dbg !21
  store half %1689, ptr addrspace(1) %address.local698, align 2, !dbg !21
  %address.local699 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local699, i64 %address.local680, !dbg !21
  store half %1688, ptr addrspace(1) %address.local700, align 2, !dbg !21
  br label %1720, !dbg !21

1720:                                             ; preds = %1682, %1681
  br i1 true, label %1721, label %1759, !dbg !21

1721:                                             ; preds = %1720
  %1722 = insertelement <4 x float> poison, float %1492, i64 0, !dbg !19
  %1723 = insertelement <4 x float> %1722, float %1498, i64 1, !dbg !19
  %1724 = insertelement <4 x float> %1723, float %1493, i64 2, !dbg !19
  %1725 = insertelement <4 x float> %1724, float %1499, i64 3, !dbg !19
  %row.reciprocal70 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1471, !dbg !19
  %normalized71 = fmul <4 x float> %1725, %row.reciprocal70, !dbg !19
  %1726 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !20
  %1727 = extractelement <4 x half> %1726, i64 3, !dbg !20
  %1728 = extractelement <4 x half> %1726, i64 2, !dbg !20
  %1729 = extractelement <4 x half> %1726, i64 1, !dbg !20
  %1730 = extractelement <4 x half> %1726, i64 0, !dbg !20
  %1731 = insertelement <4 x float> poison, float %1381, i64 0, !dbg !19
  %1732 = insertelement <4 x float> %1731, float %1387, i64 1, !dbg !19
  %1733 = insertelement <4 x float> %1732, float %1382, i64 2, !dbg !19
  %1734 = insertelement <4 x float> %1733, float %1388, i64 3, !dbg !19
  %normalized72 = fmul <4 x float> %1734, %row.reciprocal70, !dbg !19
  %1735 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !20
  %1736 = extractelement <4 x half> %1735, i64 3, !dbg !20
  %1737 = extractelement <4 x half> %1735, i64 2, !dbg !20
  %1738 = extractelement <4 x half> %1735, i64 1, !dbg !20
  %1739 = extractelement <4 x half> %1735, i64 0, !dbg !20
  %1740 = insertelement <4 x float> poison, float %1274, i64 0, !dbg !19
  %1741 = insertelement <4 x float> %1740, float %1280, i64 1, !dbg !19
  %1742 = insertelement <4 x float> %1741, float %1275, i64 2, !dbg !19
  %1743 = insertelement <4 x float> %1742, float %1281, i64 3, !dbg !19
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1253, !dbg !19
  %normalized74 = fmul <4 x float> %1743, %row.reciprocal73, !dbg !19
  %1744 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %1745 = extractelement <4 x half> %1744, i64 3, !dbg !20
  %1746 = extractelement <4 x half> %1744, i64 2, !dbg !20
  %1747 = extractelement <4 x half> %1744, i64 1, !dbg !20
  %1748 = extractelement <4 x half> %1744, i64 0, !dbg !20
  %1749 = insertelement <4 x float> poison, float %1163, i64 0, !dbg !19
  %1750 = insertelement <4 x float> %1749, float %1169, i64 1, !dbg !19
  %1751 = insertelement <4 x float> %1750, float %1164, i64 2, !dbg !19
  %1752 = insertelement <4 x float> %1751, float %1170, i64 3, !dbg !19
  %normalized75 = fmul <4 x float> %1752, %row.reciprocal73, !dbg !19
  %1753 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !20
  %1754 = extractelement <4 x half> %1753, i64 3, !dbg !20
  %1755 = extractelement <4 x half> %1753, i64 2, !dbg !20
  %1756 = extractelement <4 x half> %1753, i64 1, !dbg !20
  %1757 = extractelement <4 x half> %1753, i64 0, !dbg !20
  %1758 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread701 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local702 = zext nneg i32 %address.thread701 to i64, !dbg !9
  %address.local703 = add i64 %25, %address.local702, !dbg !21
  %address.local704 = lshr i64 %address.local702, 6, !dbg !8
  %address.local705 = shl nuw nsw i64 %address.local704, 4, !dbg !21
  %address.local706 = add i64 %address.local703, %address.local705, !dbg !21
  %address.local707 = and i64 %address.local702, 240, !dbg !21
  %address.local708 = sub i64 %address.local706, %address.local707, !dbg !21
  %address.local709 = add i64 %address.local708, %26, !dbg !21
  %address.local710 = shl nuw nsw i64 %address.local709, 7, !dbg !21
  %address.local711 = add nuw nsw i64 %1758, %address.local710, !dbg !21
  %address.local712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local713 = lshr i64 %address.local702, 2, !dbg !8
  %address.local714 = and i64 %address.local713, 12, !dbg !8
  %address.local715 = or disjoint i64 %address.local714, 96, !dbg !21
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local712, i64 %address.local715, !dbg !21
  store half %1757, ptr addrspace(1) %address.local716, align 2, !dbg !21
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local718 = or disjoint i64 %address.local714, 97, !dbg !21
  %address.local719 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local717, i64 %address.local718, !dbg !21
  store half %1756, ptr addrspace(1) %address.local719, align 2, !dbg !21
  %address.local720 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local721 = or disjoint i64 %address.local714, 98, !dbg !21
  %address.local722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local720, i64 %address.local721, !dbg !21
  store half %1755, ptr addrspace(1) %address.local722, align 2, !dbg !21
  %address.local723 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local724 = or disjoint i64 %address.local714, 99, !dbg !21
  %address.local725 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local723, i64 %address.local724, !dbg !21
  store half %1754, ptr addrspace(1) %address.local725, align 2, !dbg !21
  %address.local726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local727 = or disjoint i64 %address.local714, 112, !dbg !21
  %address.local728 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local726, i64 %address.local727, !dbg !21
  store half %1748, ptr addrspace(1) %address.local728, align 2, !dbg !21
  %address.local729 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local730 = or disjoint i64 %address.local714, 113, !dbg !21
  %address.local731 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local729, i64 %address.local730, !dbg !21
  store half %1747, ptr addrspace(1) %address.local731, align 2, !dbg !21
  %address.local732 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local733 = or disjoint i64 %address.local714, 114, !dbg !21
  %address.local734 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local732, i64 %address.local733, !dbg !21
  store half %1746, ptr addrspace(1) %address.local734, align 2, !dbg !21
  %address.local735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local736 = or i64 %address.local713, 115, !dbg !21
  %address.local737 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local735, i64 %address.local736, !dbg !21
  store half %1745, ptr addrspace(1) %address.local737, align 2, !dbg !21
  %address.local738 = shl i64 %address.local709, 7, !dbg !21
  %address.local739 = add i64 %address.local738, 8192, !dbg !21
  %address.local740 = add nuw nsw i64 %1758, %address.local739, !dbg !21
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local741, i64 %address.local715, !dbg !21
  store half %1739, ptr addrspace(1) %address.local742, align 2, !dbg !21
  %address.local743 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local743, i64 %address.local718, !dbg !21
  store half %1738, ptr addrspace(1) %address.local744, align 2, !dbg !21
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local746 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local745, i64 %address.local721, !dbg !21
  store half %1737, ptr addrspace(1) %address.local746, align 2, !dbg !21
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local747, i64 %address.local724, !dbg !21
  store half %1736, ptr addrspace(1) %address.local748, align 2, !dbg !21
  %address.local749 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local749, i64 %address.local727, !dbg !21
  store half %1730, ptr addrspace(1) %address.local750, align 2, !dbg !21
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local751, i64 %address.local730, !dbg !21
  store half %1729, ptr addrspace(1) %address.local752, align 2, !dbg !21
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local754 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local753, i64 %address.local733, !dbg !21
  store half %1728, ptr addrspace(1) %address.local754, align 2, !dbg !21
  %address.local755 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local755, i64 %address.local736, !dbg !21
  store half %1727, ptr addrspace(1) %address.local756, align 2, !dbg !21
  br label %1759, !dbg !21

1759:                                             ; preds = %1721, %1720
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/export/run-20261023-182033-aeo9_47b/input-0002-test_input_1x32x4096x128-128-64-32-online/retained32-baseline")
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
!36 = !DILocation(line: 104, column: 20, scope: !4)
!37 = !{i32 256, i32 1, i32 1}
!38 = !{i32 2, !"Debug Info Version", i32 3}
!39 = !{i32 0, i32 256}
