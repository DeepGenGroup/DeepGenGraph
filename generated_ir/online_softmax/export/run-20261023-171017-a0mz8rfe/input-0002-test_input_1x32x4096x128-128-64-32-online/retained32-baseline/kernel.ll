; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/online_softmax/export/run-20261023-171017-a0mz8rfe/input-0002-test_input_1x32x4096x128-128-64-32-online/retained32-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax/export/run-20261023-171017-a0mz8rfe/input-0002-test_input_1x32x4096x128-128-64-32-online/retained32-baseline/input.mlir"

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
  %12 = and i64 %11, 576460752303423480, !dbg !5
  %13 = and i64 %6, 7, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = sub nsw i64 31, %14, !dbg !5
  %16 = lshr i64 %10, 3, !dbg !5
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
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !39
  %32 = zext nneg i32 %31 to i64, !dbg !8
  %33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !8, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local176 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local177 = and i64 %address.local176, 1536, !dbg !8
  %address.local178 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local177, !dbg !8
  %address.local179 = ptrtoint ptr addrspace(3) %address.local178 to i64, !dbg !8
  %address.local180 = trunc i64 %address.local179 to i32, !dbg !8
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local180), !dbg !8
  %35 = zext i32 %34 to i64, !dbg !8
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !8
  %address.local181 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local182 = and i64 %address.local181, 16256, !dbg !8
  %address.local183 = lshr i64 %address.local, 4, !dbg !8
  %address.local184 = and i64 %address.local183, 8, !dbg !8
  %address.local185 = or disjoint i64 %address.local182, %address.local184, !dbg !8
  %address.local186 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local185, !dbg !8
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %38 = ptrtoint ptr addrspace(1) %address.local186 to i64, !dbg !8
  %39 = sub i64 %38, %37, !dbg !8
  %dma.buffer.offset996 = trunc i64 %39 to i32, !dbg !8
  %40 = trunc i64 %37 to i32, !dbg !8
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !8
  %42 = lshr i64 %37, 32, !dbg !8
  %43 = trunc i64 %42 to i32, !dbg !8
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !8
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc997 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !8
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !8
  %dma.buffer.lds998 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds998, i32 %dma.buffer.offset996, <4 x i32> %dma.buffer.rsrc997) #10, !dbg !8
  %address.local187 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local188 = shl nuw nsw i64 %address.local187, 3, !dbg !8
  %address.local189 = and i64 %address.local188, 3584, !dbg !8
  %address.local190 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local189, !dbg !8
  %address.local191 = ptrtoint ptr addrspace(3) %address.local190 to i64, !dbg !8
  %address.local192 = trunc i64 %address.local191 to i32, !dbg !8
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local192), !dbg !8
  %49 = zext i32 %48 to i64, !dbg !8
  %50 = inttoptr i64 %49 to ptr addrspace(3), !dbg !8
  %address.local193 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local194 = and i64 %address.local193, 16256, !dbg !8
  %address.local195 = lshr i64 %address.local187, 4, !dbg !8
  %address.local196 = and i64 %address.local195, 24, !dbg !8
  %address.local197 = or disjoint i64 %address.local194, %address.local196, !dbg !8
  %address.local198 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local197, !dbg !8
  %51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %52 = ptrtoint ptr addrspace(1) %address.local198 to i64, !dbg !8
  %53 = sub i64 %52, %51, !dbg !8
  %dma.buffer.offset993 = trunc i64 %53 to i32, !dbg !8
  %54 = trunc i64 %51 to i32, !dbg !8
  %55 = insertelement <4 x i32> poison, i32 %54, i64 0, !dbg !8
  %56 = lshr i64 %51, 32, !dbg !8
  %57 = trunc i64 %56 to i32, !dbg !8
  %58 = insertelement <4 x i32> %55, i32 %57, i64 1, !dbg !8
  %59 = insertelement <4 x i32> %58, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc994 = insertelement <4 x i32> %59, i32 131072, i64 3, !dbg !8
  %60 = ptrtoint ptr addrspace(3) %50 to i32, !dbg !8
  %dma.buffer.lds995 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !8
  %61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds995, i32 %dma.buffer.offset993, <4 x i32> %dma.buffer.rsrc994) #10, !dbg !8
  %62 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %63 = sext i32 %62 to i64, !dbg !5
  %64 = shl i64 %15, 7, !dbg !10
  %65 = lshr i64 %32, 6, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %66 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %67 = getelementptr i8, ptr addrspace(1) %66, i64 64, !dbg !8
  %address.local199 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local177, !dbg !8
  %address.local200 = ptrtoint ptr addrspace(3) %address.local199 to i64, !dbg !8
  %address.local201 = trunc i64 %address.local200 to i32, !dbg !8
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local201), !dbg !8
  %69 = zext i32 %68 to i64, !dbg !8
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !8
  %address.local202 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local185, !dbg !8
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %72 = ptrtoint ptr addrspace(1) %address.local202 to i64, !dbg !8
  %73 = sub i64 %72, %71, !dbg !8
  %dma.buffer.offset990 = trunc i64 %73 to i32, !dbg !8
  %74 = trunc i64 %71 to i32, !dbg !8
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !8
  %76 = lshr i64 %71, 32, !dbg !8
  %77 = trunc i64 %76 to i32, !dbg !8
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !8
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc991 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !8
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !8
  %dma.buffer.lds992 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !8
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds992, i32 %dma.buffer.offset990, <4 x i32> %dma.buffer.rsrc991) #10, !dbg !8
  %address.local203 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local189, !dbg !8
  %address.local204 = ptrtoint ptr addrspace(3) %address.local203 to i64, !dbg !8
  %address.local205 = trunc i64 %address.local204 to i32, !dbg !8
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local205), !dbg !8
  %83 = zext i32 %82 to i64, !dbg !8
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !8
  %address.local206 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local197, !dbg !8
  %85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %86 = ptrtoint ptr addrspace(1) %address.local206 to i64, !dbg !8
  %87 = sub i64 %86, %85, !dbg !8
  %dma.buffer.offset987 = trunc i64 %87 to i32, !dbg !8
  %88 = trunc i64 %85 to i32, !dbg !8
  %89 = insertelement <4 x i32> poison, i32 %88, i64 0, !dbg !8
  %90 = lshr i64 %85, 32, !dbg !8
  %91 = trunc i64 %90 to i32, !dbg !8
  %92 = insertelement <4 x i32> %89, i32 %91, i64 1, !dbg !8
  %93 = insertelement <4 x i32> %92, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc988 = insertelement <4 x i32> %93, i32 131072, i64 3, !dbg !8
  %94 = ptrtoint ptr addrspace(3) %84 to i32, !dbg !8
  %dma.buffer.lds989 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !8
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds989, i32 %dma.buffer.offset987, <4 x i32> %dma.buffer.rsrc988) #10, !dbg !8
  %address.local207 = lshr i32 %address.thread, 2, !dbg !7
  %address.local208 = and i32 %address.local207, 4, !dbg !7
  %address.local209 = shl nuw nsw i32 %address.thread, 1, !dbg !7
  %address.local210 = and i32 %address.local209, 384, !dbg !7
  %address.local211 = or disjoint i32 %address.local208, %address.local210, !dbg !7
  %address.local212 = shl nuw nsw i32 %address.thread, 3, !dbg !7
  %address.local213 = and i32 %address.local212, 120, !dbg !7
  %address.local214 = or disjoint i32 %address.local211, %address.local213, !dbg !7
  %address.local215 = shl nuw nsw i32 %address.thread, 5, !dbg !7
  %address.local216 = and i32 %address.local215, 1024, !dbg !7
  %address.local217 = or disjoint i32 %address.local214, %address.local216, !dbg !7
  %address.local218 = zext nneg i32 %address.local217 to i64, !dbg !7
  %address.local219 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local218, !dbg !7
  %96 = load <4 x half>, ptr addrspace(3) %address.local219, align 8, !dbg !7
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local219, i64 4096, !dbg !7
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !7
  %98 = fmul <4 x half> %96, splat (half 0xH3015), !dbg !7
  %lds.const.gep885 = getelementptr i8, ptr addrspace(3) %address.local219, i64 1024, !dbg !7
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep885, align 8, !dbg !7
  %100 = fmul <4 x half> %97, splat (half 0xH3015), !dbg !7
  %lds.const.gep886 = getelementptr i8, ptr addrspace(3) %address.local219, i64 5120, !dbg !7
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep886, align 8, !dbg !7
  %102 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !7
  %103 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %105 = getelementptr i8, ptr addrspace(1) %104, i64 128, !dbg !8
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local180), !dbg !8
  %107 = zext i32 %106 to i64, !dbg !8
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !8
  %address.local262 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local185, !dbg !8
  %109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %110 = ptrtoint ptr addrspace(1) %address.local262 to i64, !dbg !8
  %111 = sub i64 %110, %109, !dbg !8
  %dma.buffer.offset984 = trunc i64 %111 to i32, !dbg !8
  %112 = trunc i64 %109 to i32, !dbg !8
  %113 = insertelement <4 x i32> poison, i32 %112, i64 0, !dbg !8
  %114 = lshr i64 %109, 32, !dbg !8
  %115 = trunc i64 %114 to i32, !dbg !8
  %116 = insertelement <4 x i32> %113, i32 %115, i64 1, !dbg !8
  %117 = insertelement <4 x i32> %116, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc985 = insertelement <4 x i32> %117, i32 131072, i64 3, !dbg !8
  %118 = ptrtoint ptr addrspace(3) %108 to i32, !dbg !8
  %dma.buffer.lds986 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !8
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds986, i32 %dma.buffer.offset984, <4 x i32> %dma.buffer.rsrc985) #10, !dbg !8
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local192), !dbg !8
  %121 = zext i32 %120 to i64, !dbg !8
  %122 = inttoptr i64 %121 to ptr addrspace(3), !dbg !8
  %address.local263 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local197, !dbg !8
  %123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %124 = ptrtoint ptr addrspace(1) %address.local263 to i64, !dbg !8
  %125 = sub i64 %124, %123, !dbg !8
  %dma.buffer.offset981 = trunc i64 %125 to i32, !dbg !8
  %126 = trunc i64 %123 to i32, !dbg !8
  %127 = insertelement <4 x i32> poison, i32 %126, i64 0, !dbg !8
  %128 = lshr i64 %123, 32, !dbg !8
  %129 = trunc i64 %128 to i32, !dbg !8
  %130 = insertelement <4 x i32> %127, i32 %129, i64 1, !dbg !8
  %131 = insertelement <4 x i32> %130, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc982 = insertelement <4 x i32> %131, i32 131072, i64 3, !dbg !8
  %132 = ptrtoint ptr addrspace(3) %122 to i32, !dbg !8
  %dma.buffer.lds983 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !8
  %133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds983, i32 %dma.buffer.offset981, <4 x i32> %dma.buffer.rsrc982) #10, !dbg !8
  %lds.const.gep887 = getelementptr i8, ptr addrspace(3) %address.local219, i64 8192, !dbg !7
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep887, align 8, !dbg !7
  %lds.const.gep888 = getelementptr i8, ptr addrspace(3) %address.local219, i64 12288, !dbg !7
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep888, align 8, !dbg !7
  %lds.const.gep889 = getelementptr i8, ptr addrspace(3) %address.local219, i64 9216, !dbg !7
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep889, align 8, !dbg !7
  %137 = fmul <4 x half> %134, splat (half 0xH3015), !dbg !7
  %lds.const.gep890 = getelementptr i8, ptr addrspace(3) %address.local219, i64 13312, !dbg !7
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep890, align 8, !dbg !7
  %139 = fmul <4 x half> %135, splat (half 0xH3015), !dbg !7
  %140 = fmul <4 x half> %136, splat (half 0xH3015), !dbg !7
  %141 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %143 = getelementptr i8, ptr addrspace(1) %142, i64 192, !dbg !8
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local201), !dbg !8
  %145 = zext i32 %144 to i64, !dbg !8
  %146 = inttoptr i64 %145 to ptr addrspace(3), !dbg !8
  %address.local268 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local185, !dbg !8
  %147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %148 = ptrtoint ptr addrspace(1) %address.local268 to i64, !dbg !8
  %149 = sub i64 %148, %147, !dbg !8
  %dma.buffer.offset978 = trunc i64 %149 to i32, !dbg !8
  %150 = trunc i64 %147 to i32, !dbg !8
  %151 = insertelement <4 x i32> poison, i32 %150, i64 0, !dbg !8
  %152 = lshr i64 %147, 32, !dbg !8
  %153 = trunc i64 %152 to i32, !dbg !8
  %154 = insertelement <4 x i32> %151, i32 %153, i64 1, !dbg !8
  %155 = insertelement <4 x i32> %154, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc979 = insertelement <4 x i32> %155, i32 131072, i64 3, !dbg !8
  %156 = ptrtoint ptr addrspace(3) %146 to i32, !dbg !8
  %dma.buffer.lds980 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %156), !dbg !8
  %157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds980, i32 %dma.buffer.offset978, <4 x i32> %dma.buffer.rsrc979) #10, !dbg !8
  %158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local205), !dbg !8
  %159 = zext i32 %158 to i64, !dbg !8
  %160 = inttoptr i64 %159 to ptr addrspace(3), !dbg !8
  %address.local269 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local197, !dbg !8
  %161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %162 = ptrtoint ptr addrspace(1) %address.local269 to i64, !dbg !8
  %163 = sub i64 %162, %161, !dbg !8
  %dma.buffer.offset975 = trunc i64 %163 to i32, !dbg !8
  %164 = trunc i64 %161 to i32, !dbg !8
  %165 = insertelement <4 x i32> poison, i32 %164, i64 0, !dbg !8
  %166 = lshr i64 %161, 32, !dbg !8
  %167 = trunc i64 %166 to i32, !dbg !8
  %168 = insertelement <4 x i32> %165, i32 %167, i64 1, !dbg !8
  %169 = insertelement <4 x i32> %168, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc976 = insertelement <4 x i32> %169, i32 131072, i64 3, !dbg !8
  %170 = ptrtoint ptr addrspace(3) %160 to i32, !dbg !8
  %dma.buffer.lds977 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !8
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds977, i32 %dma.buffer.offset975, <4 x i32> %dma.buffer.rsrc976) #10, !dbg !8
  %172 = load <4 x half>, ptr addrspace(3) %address.local219, align 8, !dbg !7
  %lds.const.gep891 = getelementptr i8, ptr addrspace(3) %address.local219, i64 4096, !dbg !7
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep891, align 8, !dbg !7
  %lds.const.gep892 = getelementptr i8, ptr addrspace(3) %address.local219, i64 1024, !dbg !7
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep892, align 8, !dbg !7
  %175 = fmul <4 x half> %172, splat (half 0xH3015), !dbg !7
  %lds.const.gep893 = getelementptr i8, ptr addrspace(3) %address.local219, i64 5120, !dbg !7
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep893, align 8, !dbg !7
  %177 = fmul <4 x half> %173, splat (half 0xH3015), !dbg !7
  %178 = fmul <4 x half> %174, splat (half 0xH3015), !dbg !7
  %179 = fmul <4 x half> %176, splat (half 0xH3015), !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep894 = getelementptr i8, ptr addrspace(3) %address.local219, i64 8192, !dbg !7
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep894, align 8, !dbg !7
  %lds.const.gep895 = getelementptr i8, ptr addrspace(3) %address.local219, i64 12288, !dbg !7
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep895, align 8, !dbg !7
  %lds.const.gep896 = getelementptr i8, ptr addrspace(3) %address.local219, i64 9216, !dbg !7
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep896, align 8, !dbg !7
  %183 = fmul <4 x half> %180, splat (half 0xH3015), !dbg !7
  %lds.const.gep897 = getelementptr i8, ptr addrspace(3) %address.local219, i64 13312, !dbg !7
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep897, align 8, !dbg !7
  %185 = fmul <4 x half> %181, splat (half 0xH3015), !dbg !7
  %186 = fmul <4 x half> %182, splat (half 0xH3015), !dbg !7
  %187 = fmul <4 x half> %184, splat (half 0xH3015), !dbg !7
  %188 = or disjoint i64 %64, 127, !dbg !13
  %189 = add i64 %64, 128, !dbg !13
  %190 = icmp uge i64 %189, %188, !dbg !13
  %191 = and i32 %31, 16, !dbg !14
  %.not = icmp eq i32 %191, 0, !dbg !14
  %192 = and i32 %31, 32, !dbg !14
  %.not1 = icmp eq i32 %192, 0, !dbg !14
  %193 = shl nsw i64 %15, 1, !dbg !15
  %194 = add nsw i64 %193, 2, !dbg !15
  %195 = shl i64 %16, 7, !dbg !16
  %196 = and i64 %195, 3968, !dbg !16
  %197 = shl nsw i64 %63, 12, !dbg !16
  %198 = or disjoint i64 %196, %197, !dbg !16
  %199 = lshr i64 %32, 2, !dbg !17
  %200 = shl nuw nsw i64 %32, 2, !dbg !17
  %201 = mul nsw i64 %199, -16, !dbg !17
  %202 = add nsw i64 %200, %201, !dbg !17
  %203 = and i32 %31, 1, !dbg !17
  %index.compact142 = zext nneg i32 %203 to i64, !dbg !17
  %204 = or disjoint i64 %202, %index.compact142, !dbg !17
  %205 = add nuw nsw i64 %204, 1, !dbg !17
  %206 = sdiv i64 %205, 2, !dbg !17
  br label %207, !dbg !15

207:                                              ; preds = %588, %4
  %208 = phi i64 [ %1649, %588 ], [ 0, %4 ], !dbg !15
  %209 = phi [2 x <8 x float>] [ %1082, %588 ], [ zeroinitializer, %4 ], !dbg !15
  %210 = phi [2 x <1 x float>] [ %884, %588 ], [ zeroinitializer, %4 ], !dbg !15
  %211 = phi [2 x <8 x float>] [ %1276, %588 ], [ zeroinitializer, %4 ], !dbg !15
  %212 = phi [2 x <8 x float>] [ %1470, %588 ], [ zeroinitializer, %4 ], !dbg !15
  %213 = phi [2 x <8 x float>] [ %1648, %588 ], [ zeroinitializer, %4 ], !dbg !15
  %214 = icmp slt i64 %208, %194, !dbg !15
  br i1 %214, label %215, label %1650, !dbg !15

215:                                              ; preds = %207
  %216 = shl nsw i64 %208, 6, !dbg !15
  %217 = lshr i64 %208, 13, !dbg !16
  %218 = add nuw nsw i64 %217, %16, !dbg !16
  %219 = lshr i64 %208, 6, !dbg !16
  %220 = and i64 %219, 127, !dbg !16
  %221 = shl i64 %218, 19, !dbg !16
  %222 = and i64 %221, 16252928, !dbg !16
  %223 = shl i64 %219, 19, !dbg !16
  %224 = shl i64 %208, 13, !dbg !16
  %225 = sub i64 %224, %223, !dbg !16
  %226 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %222, !dbg !16
  %227 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %220, !dbg !16
  %228 = getelementptr [2 x i8], ptr addrspace(1) %227, i64 %225, !dbg !16
  %address.thread270 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local271 = zext nneg i32 %address.thread270 to i64, !dbg !8
  %address.local272 = shl nuw nsw i64 %address.local271, 3, !dbg !8
  %address.local273 = and i64 %address.local272, 1536, !dbg !16
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local273, !dbg !16
  %address.local275 = ptrtoint ptr addrspace(3) %address.local274 to i64, !dbg !16
  %address.local276 = trunc i64 %address.local275 to i32, !dbg !16
  %229 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local276), !dbg !16
  %230 = zext i32 %229 to i64, !dbg !16
  %231 = inttoptr i64 %230 to ptr addrspace(3), !dbg !16
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
  %address.local287 = getelementptr [2 x i8], ptr addrspace(1) %228, i64 %address.local286, !dbg !16
  %232 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %233 = ptrtoint ptr addrspace(1) %address.local287 to i64, !dbg !16
  %234 = sub i64 %233, %232, !dbg !16
  %dma.buffer.offset972 = trunc i64 %234 to i32, !dbg !16
  %235 = trunc i64 %232 to i32, !dbg !16
  %236 = insertelement <4 x i32> poison, i32 %235, i64 0, !dbg !16
  %237 = lshr i64 %232, 32, !dbg !16
  %238 = trunc i64 %237 to i32, !dbg !16
  %239 = insertelement <4 x i32> %236, i32 %238, i64 1, !dbg !16
  %240 = insertelement <4 x i32> %239, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc973 = insertelement <4 x i32> %240, i32 131072, i64 3, !dbg !16
  %241 = ptrtoint ptr addrspace(3) %231 to i32, !dbg !16
  %dma.buffer.lds974 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %241), !dbg !16
  %242 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds974, i32 %dma.buffer.offset972, <4 x i32> %dma.buffer.rsrc973) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %243 = add nuw i64 %208, 2048, !dbg !16
  %244 = icmp slt i64 %243, 0, !dbg !16
  %245 = sub nuw i64 -2049, %208, !dbg !16
  %246 = select i1 %244, i64 %245, i64 %243, !dbg !16
  %247 = sdiv i64 %246, 8192, !dbg !16
  %.lobit19 = ashr i64 %243, 63, !dbg !16
  %248 = xor i64 %247, %.lobit19, !dbg !16
  %249 = add nsw i64 %248, %16, !dbg !16
  %250 = add nsw i64 %219, %198, !dbg !16
  %251 = add nsw i64 %250, 32, !dbg !16
  %252 = icmp slt i64 %250, -32, !dbg !16
  %253 = sub nsw i64 -33, %250, !dbg !16
  %254 = select i1 %252, i64 %253, i64 %251, !dbg !16
  %255 = sdiv i64 %254, 128, !dbg !16
  %256 = sext i1 %252 to i64, !dbg !16
  %257 = xor i64 %255, %256, !dbg !16
  %258 = shl nsw i64 %257, 7, !dbg !16
  %259 = sub i64 %198, %258, !dbg !16
  %260 = shl i64 %249, 19, !dbg !16
  %261 = and i64 %260, 16252928, !dbg !16
  %262 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %261, !dbg !16
  %263 = getelementptr [2 x i8], ptr addrspace(1) %262, i64 %219, !dbg !16
  %264 = getelementptr [2 x i8], ptr addrspace(1) %263, i64 %259, !dbg !16
  %265 = getelementptr i8, ptr addrspace(1) %264, i64 64, !dbg !16
  %266 = getelementptr [2 x i8], ptr addrspace(1) %265, i64 %225, !dbg !16
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local273, !dbg !16
  %address.local289 = ptrtoint ptr addrspace(3) %address.local288 to i64, !dbg !16
  %address.local290 = trunc i64 %address.local289 to i32, !dbg !16
  %267 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local290), !dbg !16
  %268 = zext i32 %267 to i64, !dbg !16
  %269 = inttoptr i64 %268 to ptr addrspace(3), !dbg !16
  %address.local291 = getelementptr [2 x i8], ptr addrspace(1) %266, i64 %address.local286, !dbg !16
  %270 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %271 = ptrtoint ptr addrspace(1) %address.local291 to i64, !dbg !16
  %272 = sub i64 %271, %270, !dbg !16
  %dma.buffer.offset969 = trunc i64 %272 to i32, !dbg !16
  %273 = trunc i64 %270 to i32, !dbg !16
  %274 = insertelement <4 x i32> poison, i32 %273, i64 0, !dbg !16
  %275 = lshr i64 %270, 32, !dbg !16
  %276 = trunc i64 %275 to i32, !dbg !16
  %277 = insertelement <4 x i32> %274, i32 %276, i64 1, !dbg !16
  %278 = insertelement <4 x i32> %277, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc970 = insertelement <4 x i32> %278, i32 131072, i64 3, !dbg !16
  %279 = ptrtoint ptr addrspace(3) %269 to i32, !dbg !16
  %dma.buffer.lds971 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %279), !dbg !16
  %280 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds971, i32 %dma.buffer.offset969, <4 x i32> %dma.buffer.rsrc970) #10, !dbg !16
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
  %address.local303 = lshr i64 %address.local271, 2, !dbg !7
  %address.local304 = and i64 %address.local303, 12, !dbg !7
  %address.local305 = add nsw i64 %address.local302, %address.local304, !dbg !17
  %address.local306 = and i64 %address.local271, 63, !dbg !7
  %address.local307 = lshr i64 %address.local306, 5, !dbg !7
  %address.local308 = mul nuw nsw i64 %address.local307, 504, !dbg !17
  %address.local309 = add i64 %address.local305, %address.local308, !dbg !17
  %address.local310 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local309, !dbg !17
  %281 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !17
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %281, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep898 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !17
  %283 = load <4 x half>, ptr addrspace(3) %lds.const.gep898, align 8, !dbg !17
  %address.local321 = shl nuw nsw i32 %address.thread270, 1, !dbg !17
  %address.local322 = and i32 %address.local321, 24, !dbg !17
  %address.local323 = shl nuw nsw i32 %address.thread270, 8, !dbg !17
  %address.local324 = and i32 %address.local323, 768, !dbg !17
  %address.local325 = or disjoint i32 %address.local322, %address.local324, !dbg !17
  %address.local326 = shl nuw nsw i32 %address.thread270, 5, !dbg !17
  %address.local327 = and i32 %address.local326, 96, !dbg !17
  %address.local328 = or disjoint i32 %address.local325, %address.local327, !dbg !17
  %address.local329 = zext nneg i32 %address.local328 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %206) #9, !dbg !17, !frisk.register_lifetimes !1
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
  %284 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !17
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %283, <4 x float> %282) #9, !dbg !17
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %284, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep899 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !17
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep899, align 8, !dbg !17
  %lds.const.gep900 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !17
  %288 = load <4 x half>, ptr addrspace(3) %lds.const.gep900, align 8, !dbg !17
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %287, <4 x float> %286) #9, !dbg !17
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %288, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep901 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !17
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep901, align 8, !dbg !17
  %lds.const.gep902 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !17
  %292 = load <4 x half>, ptr addrspace(3) %lds.const.gep902, align 8, !dbg !17
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %291, <4 x float> %290) #9, !dbg !17
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %292, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep903 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !17
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep903, align 8, !dbg !17
  %296 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !17
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %295, <4 x float> %294) #9, !dbg !17
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %296, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep904 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !17
  %299 = load <4 x half>, ptr addrspace(3) %lds.const.gep904, align 8, !dbg !17
  %300 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !17
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %299, <4 x float> %298) #9, !dbg !17
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %300, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep905 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !17
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep905, align 8, !dbg !17
  %lds.const.gep906 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !17
  %304 = load <4 x half>, ptr addrspace(3) %lds.const.gep906, align 8, !dbg !17
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %303, <4 x float> %302) #9, !dbg !17
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %304, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep907 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !17
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep907, align 8, !dbg !17
  %lds.const.gep908 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !17
  %308 = load <4 x half>, ptr addrspace(3) %lds.const.gep908, align 8, !dbg !17
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %307, <4 x float> %306) #9, !dbg !17
  %lds.const.gep909 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !17
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep909, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %308, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %310, <4 x float> %311) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %313 = add nuw i64 %208, 4096, !dbg !16
  %314 = icmp slt i64 %313, 0, !dbg !16
  %315 = sub nuw i64 -4097, %208, !dbg !16
  %316 = select i1 %314, i64 %315, i64 %313, !dbg !16
  %317 = sdiv i64 %316, 8192, !dbg !16
  %.lobit20 = ashr i64 %313, 63, !dbg !16
  %318 = xor i64 %317, %.lobit20, !dbg !16
  %319 = add nsw i64 %318, %16, !dbg !16
  %320 = add nsw i64 %250, 64, !dbg !16
  %321 = icmp slt i64 %250, -64, !dbg !16
  %322 = sub nsw i64 -65, %250, !dbg !16
  %323 = select i1 %321, i64 %322, i64 %320, !dbg !16
  %324 = sdiv i64 %323, 128, !dbg !16
  %325 = sext i1 %321 to i64, !dbg !16
  %326 = xor i64 %324, %325, !dbg !16
  %327 = shl nsw i64 %326, 7, !dbg !16
  %328 = sub i64 %198, %327, !dbg !16
  %329 = shl i64 %319, 19, !dbg !16
  %330 = and i64 %329, 16252928, !dbg !16
  %331 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %330, !dbg !16
  %332 = getelementptr [2 x i8], ptr addrspace(1) %331, i64 %219, !dbg !16
  %333 = getelementptr [2 x i8], ptr addrspace(1) %332, i64 %328, !dbg !16
  %334 = getelementptr i8, ptr addrspace(1) %333, i64 128, !dbg !16
  %335 = getelementptr [2 x i8], ptr addrspace(1) %334, i64 %225, !dbg !16
  %336 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local276), !dbg !16
  %337 = zext i32 %336 to i64, !dbg !16
  %338 = inttoptr i64 %337 to ptr addrspace(3), !dbg !16
  %address.local480 = getelementptr [2 x i8], ptr addrspace(1) %335, i64 %address.local286, !dbg !16
  %339 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %340 = ptrtoint ptr addrspace(1) %address.local480 to i64, !dbg !16
  %341 = sub i64 %340, %339, !dbg !16
  %dma.buffer.offset966 = trunc i64 %341 to i32, !dbg !16
  %342 = trunc i64 %339 to i32, !dbg !16
  %343 = insertelement <4 x i32> poison, i32 %342, i64 0, !dbg !16
  %344 = lshr i64 %339, 32, !dbg !16
  %345 = trunc i64 %344 to i32, !dbg !16
  %346 = insertelement <4 x i32> %343, i32 %345, i64 1, !dbg !16
  %347 = insertelement <4 x i32> %346, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc967 = insertelement <4 x i32> %347, i32 131072, i64 3, !dbg !16
  %348 = ptrtoint ptr addrspace(3) %338 to i32, !dbg !16
  %dma.buffer.lds968 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %348), !dbg !16
  %349 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds968, i32 %dma.buffer.offset966, <4 x i32> %dma.buffer.rsrc967) #10, !dbg !16
  %lds.const.gep910 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !17
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep910, align 8, !dbg !17
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %350, <4 x float> %285) #9, !dbg !17
  %lds.const.gep911 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !17
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep911, align 8, !dbg !17
  %lds.const.gep912 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !17
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep912, align 8, !dbg !17
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %352, <4 x float> %351) #9, !dbg !17
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %353, <4 x float> %289) #9, !dbg !17
  %lds.const.gep913 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !17
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep913, align 8, !dbg !17
  %lds.const.gep914 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !17
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep914, align 8, !dbg !17
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %356, <4 x float> %355) #9, !dbg !17
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %357, <4 x float> %293) #9, !dbg !17
  %lds.const.gep915 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !17
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep915, align 8, !dbg !17
  %lds.const.gep916 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !17
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep916, align 8, !dbg !17
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %360, <4 x float> %359) #9, !dbg !17
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %361, <4 x float> %297) #9, !dbg !17
  %lds.const.gep917 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !17
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep917, align 8, !dbg !17
  %lds.const.gep918 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !17
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep918, align 8, !dbg !17
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %364, <4 x float> %363) #9, !dbg !17
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %365, <4 x float> %301) #9, !dbg !17
  %lds.const.gep919 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !17
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep919, align 8, !dbg !17
  %lds.const.gep920 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !17
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep920, align 8, !dbg !17
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %368, <4 x float> %367) #9, !dbg !17
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %369, <4 x float> %305) #9, !dbg !17
  %lds.const.gep921 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !17
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep921, align 8, !dbg !17
  %lds.const.gep922 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !17
  %373 = load <4 x half>, ptr addrspace(3) %lds.const.gep922, align 8, !dbg !17
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %372, <4 x float> %371) #9, !dbg !17
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %373, <4 x float> %309) #9, !dbg !17
  %lds.const.gep923 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !17
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep923, align 8, !dbg !17
  %lds.const.gep924 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !17
  %377 = load <4 x half>, ptr addrspace(3) %lds.const.gep924, align 8, !dbg !17
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %376, <4 x float> %375) #9, !dbg !17
  %lds.const.gep925 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !17
  %379 = load <4 x half>, ptr addrspace(3) %lds.const.gep925, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %377, <4 x float> %312) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %379, <4 x float> %380) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %382 = add nuw i64 %208, 6144, !dbg !16
  %383 = icmp slt i64 %382, 0, !dbg !16
  %384 = sub nuw i64 -6145, %208, !dbg !16
  %385 = select i1 %383, i64 %384, i64 %382, !dbg !16
  %386 = sdiv i64 %385, 8192, !dbg !16
  %.lobit21 = ashr i64 %382, 63, !dbg !16
  %387 = xor i64 %386, %.lobit21, !dbg !16
  %388 = add nsw i64 %387, %16, !dbg !16
  %389 = add nsw i64 %250, 96, !dbg !16
  %390 = icmp slt i64 %250, -96, !dbg !16
  %391 = sub nsw i64 -97, %250, !dbg !16
  %392 = select i1 %390, i64 %391, i64 %389, !dbg !16
  %393 = sdiv i64 %392, 128, !dbg !16
  %394 = sext i1 %390 to i64, !dbg !16
  %395 = xor i64 %393, %394, !dbg !16
  %396 = shl nsw i64 %395, 7, !dbg !16
  %397 = sub i64 %198, %396, !dbg !16
  %398 = shl i64 %388, 19, !dbg !16
  %399 = and i64 %398, 16252928, !dbg !16
  %400 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %399, !dbg !16
  %401 = getelementptr [2 x i8], ptr addrspace(1) %400, i64 %219, !dbg !16
  %402 = getelementptr [2 x i8], ptr addrspace(1) %401, i64 %397, !dbg !16
  %403 = getelementptr i8, ptr addrspace(1) %402, i64 192, !dbg !16
  %404 = getelementptr [2 x i8], ptr addrspace(1) %403, i64 %225, !dbg !16
  %405 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local290), !dbg !16
  %406 = zext i32 %405 to i64, !dbg !16
  %407 = inttoptr i64 %406 to ptr addrspace(3), !dbg !16
  %address.local489 = getelementptr [2 x i8], ptr addrspace(1) %404, i64 %address.local286, !dbg !16
  %408 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %409 = ptrtoint ptr addrspace(1) %address.local489 to i64, !dbg !16
  %410 = sub i64 %409, %408, !dbg !16
  %dma.buffer.offset963 = trunc i64 %410 to i32, !dbg !16
  %411 = trunc i64 %408 to i32, !dbg !16
  %412 = insertelement <4 x i32> poison, i32 %411, i64 0, !dbg !16
  %413 = lshr i64 %408, 32, !dbg !16
  %414 = trunc i64 %413 to i32, !dbg !16
  %415 = insertelement <4 x i32> %412, i32 %414, i64 1, !dbg !16
  %416 = insertelement <4 x i32> %415, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc964 = insertelement <4 x i32> %416, i32 131072, i64 3, !dbg !16
  %417 = ptrtoint ptr addrspace(3) %407 to i32, !dbg !16
  %dma.buffer.lds965 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %417), !dbg !16
  %418 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds965, i32 %dma.buffer.offset963, <4 x i32> %dma.buffer.rsrc964) #10, !dbg !16
  %419 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !17
  %420 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %419, <4 x float> %354) #9, !dbg !17
  %lds.const.gep926 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !17
  %421 = load <4 x half>, ptr addrspace(3) %lds.const.gep926, align 8, !dbg !17
  %422 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !17
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %421, <4 x float> %420) #9, !dbg !17
  %424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %422, <4 x float> %358) #9, !dbg !17
  %lds.const.gep927 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !17
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep927, align 8, !dbg !17
  %lds.const.gep928 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !17
  %426 = load <4 x half>, ptr addrspace(3) %lds.const.gep928, align 8, !dbg !17
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %425, <4 x float> %424) #9, !dbg !17
  %428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %426, <4 x float> %362) #9, !dbg !17
  %lds.const.gep929 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !17
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep929, align 8, !dbg !17
  %lds.const.gep930 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !17
  %430 = load <4 x half>, ptr addrspace(3) %lds.const.gep930, align 8, !dbg !17
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %429, <4 x float> %428) #9, !dbg !17
  %432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %430, <4 x float> %366) #9, !dbg !17
  %lds.const.gep931 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !17
  %433 = load <4 x half>, ptr addrspace(3) %lds.const.gep931, align 8, !dbg !17
  %434 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !17
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %433, <4 x float> %432) #9, !dbg !17
  %436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %434, <4 x float> %370) #9, !dbg !17
  %lds.const.gep932 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !17
  %437 = load <4 x half>, ptr addrspace(3) %lds.const.gep932, align 8, !dbg !17
  %438 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !17
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %437, <4 x float> %436) #9, !dbg !17
  %440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %438, <4 x float> %374) #9, !dbg !17
  %lds.const.gep933 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !17
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep933, align 8, !dbg !17
  %lds.const.gep934 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !17
  %442 = load <4 x half>, ptr addrspace(3) %lds.const.gep934, align 8, !dbg !17
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %441, <4 x float> %440) #9, !dbg !17
  %444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %442, <4 x float> %378) #9, !dbg !17
  %lds.const.gep935 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !17
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep935, align 8, !dbg !17
  %lds.const.gep936 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !17
  %446 = load <4 x half>, ptr addrspace(3) %lds.const.gep936, align 8, !dbg !17
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %445, <4 x float> %444) #9, !dbg !17
  %lds.const.gep937 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !17
  %448 = load <4 x half>, ptr addrspace(3) %lds.const.gep937, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %446, <4 x float> %381) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %448, <4 x float> %449) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %451 = add nuw nsw i64 %219, %16, !dbg !18
  %452 = shl i64 %451, 19, !dbg !18
  %453 = and i64 %452, 16252928, !dbg !18
  %454 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %453, !dbg !18
  %455 = getelementptr [2 x i8], ptr addrspace(1) %454, i64 %225, !dbg !18
  %456 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local276), !dbg !18
  %457 = zext i32 %456 to i64, !dbg !18
  %458 = inttoptr i64 %457 to ptr addrspace(3), !dbg !18
  %address.local490 = shl nuw nsw i64 %address.local271, 5, !dbg !18
  %address.local491 = and i64 %address.local490, 8064, !dbg !18
  %address.local492 = shl nuw nsw i64 %address.local271, 3, !dbg !18
  %address.local493 = and i64 %address.local492, 24, !dbg !18
  %address.local494 = or disjoint i64 %address.local491, %address.local493, !dbg !18
  %address.local495 = getelementptr [2 x i8], ptr addrspace(1) %455, i64 %address.local494, !dbg !18
  %459 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %460 = ptrtoint ptr addrspace(1) %address.local495 to i64, !dbg !18
  %461 = sub i64 %460, %459, !dbg !18
  %dma.buffer.offset960 = trunc i64 %461 to i32, !dbg !18
  %462 = trunc i64 %459 to i32, !dbg !18
  %463 = insertelement <4 x i32> poison, i32 %462, i64 0, !dbg !18
  %464 = lshr i64 %459, 32, !dbg !18
  %465 = trunc i64 %464 to i32, !dbg !18
  %466 = insertelement <4 x i32> %463, i32 %465, i64 1, !dbg !18
  %467 = insertelement <4 x i32> %466, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc961 = insertelement <4 x i32> %467, i32 131072, i64 3, !dbg !18
  %468 = ptrtoint ptr addrspace(3) %458 to i32, !dbg !18
  %dma.buffer.lds962 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %468), !dbg !18
  %469 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds962, i32 %dma.buffer.offset960, <4 x i32> %dma.buffer.rsrc961) #10, !dbg !18
  %lds.const.gep938 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !17
  %470 = load <4 x half>, ptr addrspace(3) %lds.const.gep938, align 8, !dbg !17
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %470, <4 x float> %423) #9, !dbg !17
  %lds.const.gep939 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !17
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep939, align 8, !dbg !17
  %lds.const.gep940 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !17
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep940, align 8, !dbg !17
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %472, <4 x float> %471) #9, !dbg !17
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %473, <4 x float> %427) #9, !dbg !17
  %lds.const.gep941 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !17
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep941, align 8, !dbg !17
  %lds.const.gep942 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !17
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep942, align 8, !dbg !17
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %476, <4 x float> %475) #9, !dbg !17
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %477, <4 x float> %431) #9, !dbg !17
  %lds.const.gep943 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !17
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep943, align 8, !dbg !17
  %lds.const.gep944 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !17
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep944, align 8, !dbg !17
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %480, <4 x float> %479) #9, !dbg !17
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %481, <4 x float> %435) #9, !dbg !17
  %lds.const.gep945 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !17
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep945, align 8, !dbg !17
  %lds.const.gep946 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !17
  %485 = load <4 x half>, ptr addrspace(3) %lds.const.gep946, align 8, !dbg !17
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %484, <4 x float> %483) #9, !dbg !17
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %485, <4 x float> %439) #9, !dbg !17
  %lds.const.gep947 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !17
  %488 = load <4 x half>, ptr addrspace(3) %lds.const.gep947, align 8, !dbg !17
  %lds.const.gep948 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !17
  %489 = load <4 x half>, ptr addrspace(3) %lds.const.gep948, align 8, !dbg !17
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %488, <4 x float> %487) #9, !dbg !17
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %489, <4 x float> %443) #9, !dbg !17
  %lds.const.gep949 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !17
  %492 = load <4 x half>, ptr addrspace(3) %lds.const.gep949, align 8, !dbg !17
  %lds.const.gep950 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !17
  %493 = load <4 x half>, ptr addrspace(3) %lds.const.gep950, align 8, !dbg !17
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %492, <4 x float> %491) #9, !dbg !17
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %493, <4 x float> %447) #9, !dbg !17
  %lds.const.gep951 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !17
  %496 = load <4 x half>, ptr addrspace(3) %lds.const.gep951, align 8, !dbg !17
  %lds.const.gep952 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !17
  %497 = load <4 x half>, ptr addrspace(3) %lds.const.gep952, align 8, !dbg !17
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %496, <4 x float> %495) #9, !dbg !17
  %lds.const.gep953 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !17
  %499 = load <4 x half>, ptr addrspace(3) %lds.const.gep953, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %497, <4 x float> %450) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %501 = or disjoint i64 %216, 63, !dbg !13
  %502 = icmp uge i64 %64, %501, !dbg !13
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %499, <4 x float> %500) #9, !dbg !17
  %504 = and i1 %190, %502, !dbg !13
  br i1 %504, label %505, label %506, !dbg !13

505:                                              ; preds = %215
  br label %586, !dbg !13

506:                                              ; preds = %215
  %507 = add i64 %64, %32, !dbg !13
  %508 = shl nuw nsw i64 %65, 4, !dbg !13
  %509 = add i64 %507, %508, !dbg !13
  %510 = and i32 %31, 240, !dbg !13
  %index.compact = zext nneg i32 %510 to i64, !dbg !13
  %511 = sub i64 %509, %index.compact, !dbg !13
  %512 = and i64 %32, 48, !dbg !13
  %513 = or disjoint i64 %216, %512, !dbg !13
  %514 = add i64 %511, 1, !dbg !23
  %515 = lshr i64 %514, 32, !dbg !24
  %compare.high = trunc i64 %515 to i32, !dbg !24
  %516 = lshr i64 %513, 32, !dbg !24
  %compare.high757 = trunc i64 %516 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high757, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high757, !dbg !24
  %compare.low.base = trunc i64 %514 to i32, !dbg !24
  %compare.low.base758 = trunc i64 %513 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base758, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %517 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low759 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result760 = select i1 %compare.high.equal, i1 %compare.low759, i1 %compare.high.order, !dbg !24
  %518 = select i1 %compare.result760, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base761 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local762 = or i32 %compare.local.base761, 2, !dbg !24
  %compare.low763 = icmp ugt i32 %compare.low.base, %compare.local762, !dbg !24
  %compare.result764 = select i1 %compare.high.equal, i1 %compare.low763, i1 %compare.high.order, !dbg !24
  %519 = select i1 %compare.result764, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base765 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local766 = or i32 %compare.local.base765, 3, !dbg !24
  %compare.low767 = icmp ugt i32 %compare.low.base, %compare.local766, !dbg !24
  %compare.result768 = select i1 %compare.high.equal, i1 %compare.low767, i1 %compare.high.order, !dbg !24
  %520 = select i1 %compare.result768, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base769 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local770 = or i32 %compare.local.base769, 4, !dbg !24
  %compare.low771 = icmp ugt i32 %compare.low.base, %compare.local770, !dbg !24
  %compare.result772 = select i1 %compare.high.equal, i1 %compare.low771, i1 %compare.high.order, !dbg !24
  %521 = select i1 %compare.result772, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base773 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local774 = or i32 %compare.local.base773, 5, !dbg !24
  %compare.low775 = icmp ugt i32 %compare.low.base, %compare.local774, !dbg !24
  %compare.result776 = select i1 %compare.high.equal, i1 %compare.low775, i1 %compare.high.order, !dbg !24
  %522 = select i1 %compare.result776, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base777 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local778 = or i32 %compare.local.base777, 6, !dbg !24
  %compare.low779 = icmp ugt i32 %compare.low.base, %compare.local778, !dbg !24
  %compare.result780 = select i1 %compare.high.equal, i1 %compare.low779, i1 %compare.high.order, !dbg !24
  %523 = select i1 %compare.result780, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base781 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local782 = or i32 %compare.local.base781, 7, !dbg !24
  %compare.low783 = icmp ugt i32 %compare.low.base, %compare.local782, !dbg !24
  %compare.result784 = select i1 %compare.high.equal, i1 %compare.low783, i1 %compare.high.order, !dbg !24
  %524 = select i1 %compare.result784, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base785 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local786 = or i32 %compare.local.base785, 8, !dbg !24
  %compare.low787 = icmp ugt i32 %compare.low.base, %compare.local786, !dbg !24
  %compare.result788 = select i1 %compare.high.equal, i1 %compare.low787, i1 %compare.high.order, !dbg !24
  %525 = select i1 %compare.result788, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base789 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local790 = or i32 %compare.local.base789, 9, !dbg !24
  %compare.low791 = icmp ugt i32 %compare.low.base, %compare.local790, !dbg !24
  %compare.result792 = select i1 %compare.high.equal, i1 %compare.low791, i1 %compare.high.order, !dbg !24
  %526 = select i1 %compare.result792, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base793 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local794 = or i32 %compare.local.base793, 10, !dbg !24
  %compare.low795 = icmp ugt i32 %compare.low.base, %compare.local794, !dbg !24
  %compare.result796 = select i1 %compare.high.equal, i1 %compare.low795, i1 %compare.high.order, !dbg !24
  %527 = select i1 %compare.result796, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base797 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local798 = or i32 %compare.local.base797, 11, !dbg !24
  %compare.low799 = icmp ugt i32 %compare.low.base, %compare.local798, !dbg !24
  %compare.result800 = select i1 %compare.high.equal, i1 %compare.low799, i1 %compare.high.order, !dbg !24
  %528 = select i1 %compare.result800, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base801 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local802 = or i32 %compare.local.base801, 12, !dbg !24
  %compare.low803 = icmp ugt i32 %compare.low.base, %compare.local802, !dbg !24
  %compare.result804 = select i1 %compare.high.equal, i1 %compare.low803, i1 %compare.high.order, !dbg !24
  %529 = select i1 %compare.result804, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base805 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local806 = or i32 %compare.local.base805, 13, !dbg !24
  %compare.low807 = icmp ugt i32 %compare.low.base, %compare.local806, !dbg !24
  %compare.result808 = select i1 %compare.high.equal, i1 %compare.low807, i1 %compare.high.order, !dbg !24
  %530 = select i1 %compare.result808, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base809 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local810 = or i32 %compare.local.base809, 14, !dbg !24
  %compare.low811 = icmp ugt i32 %compare.low.base, %compare.local810, !dbg !24
  %compare.result812 = select i1 %compare.high.equal, i1 %compare.low811, i1 %compare.high.order, !dbg !24
  %531 = select i1 %compare.result812, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base813 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local814 = or i32 %compare.local.base813, 15, !dbg !24
  %compare.low815 = icmp ugt i32 %compare.low.base, %compare.local814, !dbg !24
  %compare.result816 = select i1 %compare.high.equal, i1 %compare.low815, i1 %compare.high.order, !dbg !24
  %532 = select i1 %compare.result816, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %533 = add i64 %511, 65, !dbg !23
  %534 = lshr i64 %533, 32, !dbg !24
  %compare.high817 = trunc i64 %534 to i32, !dbg !24
  %535 = lshr i64 %513, 32, !dbg !24
  %compare.high818 = trunc i64 %535 to i32, !dbg !24
  %compare.high.equal819 = icmp eq i32 %compare.high817, %compare.high818, !dbg !24
  %compare.high.order820 = icmp ugt i32 %compare.high817, %compare.high818, !dbg !24
  %compare.low.base821 = trunc i64 %533 to i32, !dbg !24
  %compare.low.base822 = trunc i64 %513 to i32, !dbg !24
  %compare.low823 = icmp ugt i32 %compare.low.base821, %compare.low.base822, !dbg !24
  %compare.result824 = select i1 %compare.high.equal819, i1 %compare.low823, i1 %compare.high.order820, !dbg !24
  %536 = select i1 %compare.result824, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base825 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local826 = or i32 %compare.local.base825, 1, !dbg !24
  %compare.low827 = icmp ugt i32 %compare.low.base821, %compare.local826, !dbg !24
  %compare.result828 = select i1 %compare.high.equal819, i1 %compare.low827, i1 %compare.high.order820, !dbg !24
  %537 = select i1 %compare.result828, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base829 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local830 = or i32 %compare.local.base829, 2, !dbg !24
  %compare.low831 = icmp ugt i32 %compare.low.base821, %compare.local830, !dbg !24
  %compare.result832 = select i1 %compare.high.equal819, i1 %compare.low831, i1 %compare.high.order820, !dbg !24
  %538 = select i1 %compare.result832, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base833 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local834 = or i32 %compare.local.base833, 3, !dbg !24
  %compare.low835 = icmp ugt i32 %compare.low.base821, %compare.local834, !dbg !24
  %compare.result836 = select i1 %compare.high.equal819, i1 %compare.low835, i1 %compare.high.order820, !dbg !24
  %539 = select i1 %compare.result836, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base837 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local838 = or i32 %compare.local.base837, 4, !dbg !24
  %compare.low839 = icmp ugt i32 %compare.low.base821, %compare.local838, !dbg !24
  %compare.result840 = select i1 %compare.high.equal819, i1 %compare.low839, i1 %compare.high.order820, !dbg !24
  %540 = select i1 %compare.result840, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base841 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local842 = or i32 %compare.local.base841, 5, !dbg !24
  %compare.low843 = icmp ugt i32 %compare.low.base821, %compare.local842, !dbg !24
  %compare.result844 = select i1 %compare.high.equal819, i1 %compare.low843, i1 %compare.high.order820, !dbg !24
  %541 = select i1 %compare.result844, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base845 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local846 = or i32 %compare.local.base845, 6, !dbg !24
  %compare.low847 = icmp ugt i32 %compare.low.base821, %compare.local846, !dbg !24
  %compare.result848 = select i1 %compare.high.equal819, i1 %compare.low847, i1 %compare.high.order820, !dbg !24
  %542 = select i1 %compare.result848, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base849 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local850 = or i32 %compare.local.base849, 7, !dbg !24
  %compare.low851 = icmp ugt i32 %compare.low.base821, %compare.local850, !dbg !24
  %compare.result852 = select i1 %compare.high.equal819, i1 %compare.low851, i1 %compare.high.order820, !dbg !24
  %543 = select i1 %compare.result852, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base853 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local854 = or i32 %compare.local.base853, 8, !dbg !24
  %compare.low855 = icmp ugt i32 %compare.low.base821, %compare.local854, !dbg !24
  %compare.result856 = select i1 %compare.high.equal819, i1 %compare.low855, i1 %compare.high.order820, !dbg !24
  %544 = select i1 %compare.result856, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base857 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local858 = or i32 %compare.local.base857, 9, !dbg !24
  %compare.low859 = icmp ugt i32 %compare.low.base821, %compare.local858, !dbg !24
  %compare.result860 = select i1 %compare.high.equal819, i1 %compare.low859, i1 %compare.high.order820, !dbg !24
  %545 = select i1 %compare.result860, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base861 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local862 = or i32 %compare.local.base861, 10, !dbg !24
  %compare.low863 = icmp ugt i32 %compare.low.base821, %compare.local862, !dbg !24
  %compare.result864 = select i1 %compare.high.equal819, i1 %compare.low863, i1 %compare.high.order820, !dbg !24
  %546 = select i1 %compare.result864, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base865 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local866 = or i32 %compare.local.base865, 11, !dbg !24
  %compare.low867 = icmp ugt i32 %compare.low.base821, %compare.local866, !dbg !24
  %compare.result868 = select i1 %compare.high.equal819, i1 %compare.low867, i1 %compare.high.order820, !dbg !24
  %547 = select i1 %compare.result868, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base869 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local870 = or i32 %compare.local.base869, 12, !dbg !24
  %compare.low871 = icmp ugt i32 %compare.low.base821, %compare.local870, !dbg !24
  %compare.result872 = select i1 %compare.high.equal819, i1 %compare.low871, i1 %compare.high.order820, !dbg !24
  %548 = select i1 %compare.result872, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base873 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local874 = or i32 %compare.local.base873, 13, !dbg !24
  %compare.low875 = icmp ugt i32 %compare.low.base821, %compare.local874, !dbg !24
  %compare.result876 = select i1 %compare.high.equal819, i1 %compare.low875, i1 %compare.high.order820, !dbg !24
  %549 = select i1 %compare.result876, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base877 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local878 = or i32 %compare.local.base877, 14, !dbg !24
  %compare.low879 = icmp ugt i32 %compare.low.base821, %compare.local878, !dbg !24
  %compare.result880 = select i1 %compare.high.equal819, i1 %compare.low879, i1 %compare.high.order820, !dbg !24
  %550 = select i1 %compare.result880, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base881 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local882 = or i32 %compare.local.base881, 15, !dbg !24
  %compare.low883 = icmp ugt i32 %compare.low.base821, %compare.local882, !dbg !24
  %compare.result884 = select i1 %compare.high.equal819, i1 %compare.low883, i1 %compare.high.order820, !dbg !24
  %551 = select i1 %compare.result884, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %552 = insertelement <16 x float> poison, float %517, i64 0, !dbg !13
  %553 = insertelement <16 x float> %552, float %518, i64 1, !dbg !13
  %554 = insertelement <16 x float> %553, float %519, i64 2, !dbg !13
  %555 = insertelement <16 x float> %554, float %520, i64 3, !dbg !13
  %556 = insertelement <16 x float> %555, float %521, i64 4, !dbg !13
  %557 = insertelement <16 x float> %556, float %522, i64 5, !dbg !13
  %558 = insertelement <16 x float> %557, float %523, i64 6, !dbg !13
  %559 = insertelement <16 x float> %558, float %524, i64 7, !dbg !13
  %560 = insertelement <16 x float> %559, float %525, i64 8, !dbg !13
  %561 = insertelement <16 x float> %560, float %526, i64 9, !dbg !13
  %562 = insertelement <16 x float> %561, float %527, i64 10, !dbg !13
  %563 = insertelement <16 x float> %562, float %528, i64 11, !dbg !13
  %564 = insertelement <16 x float> %563, float %529, i64 12, !dbg !13
  %565 = insertelement <16 x float> %564, float %530, i64 13, !dbg !13
  %566 = insertelement <16 x float> %565, float %531, i64 14, !dbg !13
  %567 = insertelement <16 x float> %566, float %532, i64 15, !dbg !13
  %568 = insertvalue [2 x <16 x float>] poison, <16 x float> %567, 0, !dbg !13
  %569 = insertelement <16 x float> poison, float %536, i64 0, !dbg !13
  %570 = insertelement <16 x float> %569, float %537, i64 1, !dbg !13
  %571 = insertelement <16 x float> %570, float %538, i64 2, !dbg !13
  %572 = insertelement <16 x float> %571, float %539, i64 3, !dbg !13
  %573 = insertelement <16 x float> %572, float %540, i64 4, !dbg !13
  %574 = insertelement <16 x float> %573, float %541, i64 5, !dbg !13
  %575 = insertelement <16 x float> %574, float %542, i64 6, !dbg !13
  %576 = insertelement <16 x float> %575, float %543, i64 7, !dbg !13
  %577 = insertelement <16 x float> %576, float %544, i64 8, !dbg !13
  %578 = insertelement <16 x float> %577, float %545, i64 9, !dbg !13
  %579 = insertelement <16 x float> %578, float %546, i64 10, !dbg !13
  %580 = insertelement <16 x float> %579, float %547, i64 11, !dbg !13
  %581 = insertelement <16 x float> %580, float %548, i64 12, !dbg !13
  %582 = insertelement <16 x float> %581, float %549, i64 13, !dbg !13
  %583 = insertelement <16 x float> %582, float %550, i64 14, !dbg !13
  %584 = insertelement <16 x float> %583, float %551, i64 15, !dbg !13
  %585 = insertvalue [2 x <16 x float>] %568, <16 x float> %584, 1, !dbg !13
  br label %586, !dbg !13

586:                                              ; preds = %505, %506
  %587 = phi [2 x <16 x float>] [ %585, %506 ], [ zeroinitializer, %505 ], !dbg !13
  br label %588, !dbg !13

588:                                              ; preds = %586
  %589 = extractvalue [2 x <16 x float>] %587, 0, !dbg !26
  %590 = shufflevector <16 x float> %589, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %591 = fadd <4 x float> %474, %590, !dbg !26
  %592 = extractelement <4 x float> %591, i64 0, !dbg !26
  %593 = extractelement <4 x float> %591, i64 1, !dbg !26
  %594 = extractelement <4 x float> %591, i64 2, !dbg !26
  %595 = extractelement <4 x float> %591, i64 3, !dbg !26
  %596 = shufflevector <16 x float> %589, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %597 = fadd <4 x float> %478, %596, !dbg !26
  %598 = extractelement <4 x float> %597, i64 0, !dbg !26
  %599 = extractelement <4 x float> %597, i64 1, !dbg !26
  %600 = extractelement <4 x float> %597, i64 2, !dbg !26
  %601 = extractelement <4 x float> %597, i64 3, !dbg !26
  %602 = shufflevector <16 x float> %589, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %603 = fadd <4 x float> %482, %602, !dbg !26
  %604 = extractelement <4 x float> %603, i64 0, !dbg !26
  %605 = extractelement <4 x float> %603, i64 1, !dbg !26
  %606 = extractelement <4 x float> %603, i64 2, !dbg !26
  %607 = extractelement <4 x float> %603, i64 3, !dbg !26
  %608 = shufflevector <16 x float> %589, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %609 = fadd <4 x float> %486, %608, !dbg !26
  %610 = extractelement <4 x float> %609, i64 0, !dbg !26
  %611 = extractelement <4 x float> %609, i64 1, !dbg !26
  %612 = extractelement <4 x float> %609, i64 2, !dbg !26
  %613 = extractelement <4 x float> %609, i64 3, !dbg !26
  %614 = extractvalue [2 x <16 x float>] %587, 1, !dbg !26
  %615 = shufflevector <16 x float> %614, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %616 = fadd <4 x float> %490, %615, !dbg !26
  %617 = extractelement <4 x float> %616, i64 0, !dbg !26
  %618 = extractelement <4 x float> %616, i64 1, !dbg !26
  %619 = extractelement <4 x float> %616, i64 2, !dbg !26
  %620 = extractelement <4 x float> %616, i64 3, !dbg !26
  %621 = shufflevector <16 x float> %614, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %622 = fadd <4 x float> %494, %621, !dbg !26
  %623 = extractelement <4 x float> %622, i64 0, !dbg !26
  %624 = extractelement <4 x float> %622, i64 1, !dbg !26
  %625 = extractelement <4 x float> %622, i64 2, !dbg !26
  %626 = extractelement <4 x float> %622, i64 3, !dbg !26
  %627 = shufflevector <16 x float> %614, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %628 = fadd <4 x float> %498, %627, !dbg !26
  %629 = extractelement <4 x float> %628, i64 0, !dbg !26
  %630 = extractelement <4 x float> %628, i64 1, !dbg !26
  %631 = extractelement <4 x float> %628, i64 2, !dbg !26
  %632 = extractelement <4 x float> %628, i64 3, !dbg !26
  %633 = shufflevector <16 x float> %614, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %634 = fadd <4 x float> %503, %633, !dbg !26
  %635 = extractelement <4 x float> %634, i64 0, !dbg !26
  %636 = extractelement <4 x float> %634, i64 1, !dbg !26
  %637 = extractelement <4 x float> %634, i64 2, !dbg !26
  %638 = extractelement <4 x float> %634, i64 3, !dbg !26
  %639 = call float @llvm.maxnum.f32(float %592, float 0xFFF0000000000000), !dbg !27
  %640 = call float @llvm.maxnum.f32(float %639, float %593), !dbg !27
  %641 = call float @llvm.maxnum.f32(float %640, float %594), !dbg !27
  %642 = call float @llvm.maxnum.f32(float %641, float %595), !dbg !27
  %643 = call float @llvm.maxnum.f32(float %642, float %598), !dbg !27
  %644 = call float @llvm.maxnum.f32(float %643, float %599), !dbg !27
  %645 = call float @llvm.maxnum.f32(float %644, float %600), !dbg !27
  %646 = call float @llvm.maxnum.f32(float %645, float %601), !dbg !27
  %647 = call float @llvm.maxnum.f32(float %646, float %604), !dbg !27
  %648 = call float @llvm.maxnum.f32(float %647, float %605), !dbg !27
  %649 = call float @llvm.maxnum.f32(float %648, float %606), !dbg !27
  %650 = call float @llvm.maxnum.f32(float %649, float %607), !dbg !27
  %651 = call float @llvm.maxnum.f32(float %650, float %610), !dbg !27
  %652 = call float @llvm.maxnum.f32(float %651, float %611), !dbg !27
  %653 = call float @llvm.maxnum.f32(float %652, float %612), !dbg !27
  %654 = call float @llvm.maxnum.f32(float %653, float %613), !dbg !27
  %655 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %656 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %655), !dbg !27
  %657 = add nuw nsw i32 %656, 64, !dbg !27
  %658 = xor i32 %656, 16, !dbg !27
  %659 = and i32 %657, 192, !dbg !27
  %660 = icmp samesign ult i32 %658, %659, !dbg !27
  %661 = select i1 %660, i32 %658, i32 %656, !dbg !27
  %662 = shl nuw nsw i32 %661, 2, !dbg !27
  %663 = bitcast float %654 to i32, !dbg !27
  %664 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %663), !dbg !27
  %665 = bitcast i32 %664 to float, !dbg !27
  %666 = call float @llvm.maxnum.f32(float %654, float %665), !dbg !27
  %667 = xor i32 %656, 32, !dbg !27
  %668 = icmp samesign ult i32 %667, %659, !dbg !27
  %669 = select i1 %668, i32 %667, i32 %656, !dbg !27
  %670 = shl nuw nsw i32 %669, 2, !dbg !27
  %671 = bitcast float %666 to i32, !dbg !27
  %672 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %671), !dbg !27
  %673 = bitcast i32 %672 to float, !dbg !27
  %674 = call float @llvm.maxnum.f32(float %666, float %673), !dbg !27
  %675 = insertelement <1 x float> poison, float %674, i64 0, !dbg !27
  %676 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %675, <1 x float> splat (float 0xFFF0000000000000)), !dbg !27
  %677 = call float @llvm.maxnum.f32(float %617, float 0xFFF0000000000000), !dbg !27
  %678 = call float @llvm.maxnum.f32(float %677, float %618), !dbg !27
  %679 = call float @llvm.maxnum.f32(float %678, float %619), !dbg !27
  %680 = call float @llvm.maxnum.f32(float %679, float %620), !dbg !27
  %681 = call float @llvm.maxnum.f32(float %680, float %623), !dbg !27
  %682 = call float @llvm.maxnum.f32(float %681, float %624), !dbg !27
  %683 = call float @llvm.maxnum.f32(float %682, float %625), !dbg !27
  %684 = call float @llvm.maxnum.f32(float %683, float %626), !dbg !27
  %685 = call float @llvm.maxnum.f32(float %684, float %629), !dbg !27
  %686 = call float @llvm.maxnum.f32(float %685, float %630), !dbg !27
  %687 = call float @llvm.maxnum.f32(float %686, float %631), !dbg !27
  %688 = call float @llvm.maxnum.f32(float %687, float %632), !dbg !27
  %689 = call float @llvm.maxnum.f32(float %688, float %635), !dbg !27
  %690 = call float @llvm.maxnum.f32(float %689, float %636), !dbg !27
  %691 = call float @llvm.maxnum.f32(float %690, float %637), !dbg !27
  %692 = call float @llvm.maxnum.f32(float %691, float %638), !dbg !27
  %693 = bitcast float %692 to i32, !dbg !27
  %694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %693), !dbg !27
  %695 = bitcast i32 %694 to float, !dbg !27
  %696 = call float @llvm.maxnum.f32(float %692, float %695), !dbg !27
  %697 = bitcast float %696 to i32, !dbg !27
  %698 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %697), !dbg !27
  %699 = bitcast i32 %698 to float, !dbg !27
  %700 = call float @llvm.maxnum.f32(float %696, float %699), !dbg !27
  %701 = insertelement <1 x float> poison, float %700, i64 0, !dbg !27
  %702 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %701, <1 x float> splat (float 0xFFF0000000000000)), !dbg !27
  %703 = shufflevector <1 x float> %676, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %704 = fsub <4 x float> %591, %703, !dbg !28
  %705 = fsub <4 x float> %597, %703, !dbg !28
  %706 = fsub <4 x float> %603, %703, !dbg !28
  %707 = fsub <4 x float> %609, %703, !dbg !28
  %708 = shufflevector <1 x float> %702, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %709 = fsub <4 x float> %616, %708, !dbg !28
  %710 = fsub <4 x float> %622, %708, !dbg !28
  %711 = fsub <4 x float> %628, %708, !dbg !28
  %712 = fsub <4 x float> %634, %708, !dbg !28
  %713 = extractelement <1 x float> %676, i64 0, !dbg !29
  %714 = fsub float 0xFFF0000000000000, %713, !dbg !29
  %715 = call float @llvm.amdgcn.exp2.f32(float %714), !dbg !30
  %716 = insertelement <4 x float> poison, float %715, i64 0, !dbg !35
  %717 = shufflevector <4 x float> %716, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %718 = extractvalue [2 x <1 x float>] %210, 0, !dbg !31
  %719 = extractelement <1 x float> %718, i64 0, !dbg !31
  %720 = fmul float %719, %715, !dbg !31
  %721 = extractelement <1 x float> %702, i64 0, !dbg !29
  %722 = fsub float 0xFFF0000000000000, %721, !dbg !29
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !30
  %724 = insertelement <4 x float> poison, float %723, i64 0, !dbg !35
  %725 = shufflevector <4 x float> %724, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %726 = extractvalue [2 x <1 x float>] %210, 1, !dbg !31
  %727 = extractelement <1 x float> %726, i64 0, !dbg !31
  %728 = fmul float %727, %723, !dbg !31
  %729 = extractelement <4 x float> %704, i64 0, !dbg !32
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !32
  %731 = insertelement <4 x float> poison, float %730, i64 0, !dbg !34
  %732 = extractelement <4 x float> %704, i64 1, !dbg !32
  %733 = call float @llvm.amdgcn.exp2.f32(float %732), !dbg !32
  %734 = insertelement <4 x float> %731, float %733, i64 1, !dbg !34
  %735 = fadd float %730, %733, !dbg !33
  %736 = extractelement <4 x float> %704, i64 2, !dbg !32
  %737 = call float @llvm.amdgcn.exp2.f32(float %736), !dbg !32
  %738 = insertelement <4 x float> %734, float %737, i64 2, !dbg !34
  %739 = fadd float %735, %737, !dbg !33
  %740 = extractelement <4 x float> %704, i64 3, !dbg !32
  %741 = call float @llvm.amdgcn.exp2.f32(float %740), !dbg !32
  %742 = insertelement <4 x float> %738, float %741, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %742) #9, !dbg !34, !frisk.register_lifetimes !1
  %743 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %743) #9, !dbg !33, !frisk.register_lifetimes !1
  %744 = fadd float %739, %741, !dbg !33
  %745 = extractelement <4 x float> %705, i64 0, !dbg !32
  %746 = call float @llvm.amdgcn.exp2.f32(float %745), !dbg !32
  %747 = insertelement <4 x float> poison, float %746, i64 0, !dbg !34
  %748 = fadd float %744, %746, !dbg !33
  %749 = extractelement <4 x float> %705, i64 1, !dbg !32
  %750 = call float @llvm.amdgcn.exp2.f32(float %749), !dbg !32
  %751 = insertelement <4 x float> %747, float %750, i64 1, !dbg !34
  %752 = fadd float %748, %750, !dbg !33
  %753 = extractelement <4 x float> %705, i64 2, !dbg !32
  %754 = call float @llvm.amdgcn.exp2.f32(float %753), !dbg !32
  %755 = insertelement <4 x float> %751, float %754, i64 2, !dbg !34
  %756 = fadd float %752, %754, !dbg !33
  %757 = extractelement <4 x float> %705, i64 3, !dbg !32
  %758 = call float @llvm.amdgcn.exp2.f32(float %757), !dbg !32
  %759 = insertelement <4 x float> %755, float %758, i64 3, !dbg !34
  %math.result162 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %759) #9, !dbg !34, !frisk.register_lifetimes !1
  %760 = fptrunc <4 x float> %math.result162 to <4 x half>, !dbg !34
  %math.packed163 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %760) #9, !dbg !33, !frisk.register_lifetimes !1
  %761 = fadd float %756, %758, !dbg !33
  %762 = extractelement <4 x float> %706, i64 0, !dbg !32
  %763 = call float @llvm.amdgcn.exp2.f32(float %762), !dbg !32
  %764 = insertelement <4 x float> poison, float %763, i64 0, !dbg !34
  %765 = fadd float %761, %763, !dbg !33
  %766 = extractelement <4 x float> %706, i64 1, !dbg !32
  %767 = call float @llvm.amdgcn.exp2.f32(float %766), !dbg !32
  %768 = insertelement <4 x float> %764, float %767, i64 1, !dbg !34
  %769 = fadd float %765, %767, !dbg !33
  %770 = extractelement <4 x float> %706, i64 2, !dbg !32
  %771 = call float @llvm.amdgcn.exp2.f32(float %770), !dbg !32
  %772 = insertelement <4 x float> %768, float %771, i64 2, !dbg !34
  %773 = fadd float %769, %771, !dbg !33
  %774 = extractelement <4 x float> %706, i64 3, !dbg !32
  %775 = call float @llvm.amdgcn.exp2.f32(float %774), !dbg !32
  %776 = insertelement <4 x float> %772, float %775, i64 3, !dbg !34
  %math.result164 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %776) #9, !dbg !34, !frisk.register_lifetimes !1
  %777 = fptrunc <4 x float> %math.result164 to <4 x half>, !dbg !34
  %math.packed165 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %777) #9, !dbg !33, !frisk.register_lifetimes !1
  %778 = fadd float %773, %775, !dbg !33
  %779 = extractelement <4 x float> %707, i64 0, !dbg !32
  %780 = call float @llvm.amdgcn.exp2.f32(float %779), !dbg !32
  %781 = insertelement <4 x float> poison, float %780, i64 0, !dbg !34
  %782 = fadd float %778, %780, !dbg !33
  %783 = extractelement <4 x float> %707, i64 1, !dbg !32
  %784 = call float @llvm.amdgcn.exp2.f32(float %783), !dbg !32
  %785 = insertelement <4 x float> %781, float %784, i64 1, !dbg !34
  %786 = fadd float %782, %784, !dbg !33
  %787 = extractelement <4 x float> %707, i64 2, !dbg !32
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !32
  %789 = insertelement <4 x float> %785, float %788, i64 2, !dbg !34
  %790 = fadd float %786, %788, !dbg !33
  %791 = extractelement <4 x float> %707, i64 3, !dbg !32
  %792 = call float @llvm.amdgcn.exp2.f32(float %791), !dbg !32
  %793 = insertelement <4 x float> %789, float %792, i64 3, !dbg !34
  %math.result166 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %793) #9, !dbg !34, !frisk.register_lifetimes !1
  %794 = fptrunc <4 x float> %math.result166 to <4 x half>, !dbg !34
  %math.packed167 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %794) #9, !dbg !33, !frisk.register_lifetimes !1
  %795 = fadd float %790, %792, !dbg !33
  %796 = extractelement <4 x float> %709, i64 0, !dbg !32
  %797 = call float @llvm.amdgcn.exp2.f32(float %796), !dbg !32
  %798 = insertelement <4 x float> poison, float %797, i64 0, !dbg !34
  %799 = extractelement <4 x float> %709, i64 1, !dbg !32
  %800 = call float @llvm.amdgcn.exp2.f32(float %799), !dbg !32
  %801 = insertelement <4 x float> %798, float %800, i64 1, !dbg !34
  %802 = fadd float %797, %800, !dbg !33
  %803 = extractelement <4 x float> %709, i64 2, !dbg !32
  %804 = call float @llvm.amdgcn.exp2.f32(float %803), !dbg !32
  %805 = insertelement <4 x float> %801, float %804, i64 2, !dbg !34
  %806 = fadd float %802, %804, !dbg !33
  %807 = extractelement <4 x float> %709, i64 3, !dbg !32
  %808 = call float @llvm.amdgcn.exp2.f32(float %807), !dbg !32
  %809 = insertelement <4 x float> %805, float %808, i64 3, !dbg !34
  %math.result168 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %809) #9, !dbg !34, !frisk.register_lifetimes !1
  %810 = fptrunc <4 x float> %math.result168 to <4 x half>, !dbg !34
  %math.packed169 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %810) #9, !dbg !33, !frisk.register_lifetimes !1
  %811 = fadd float %806, %808, !dbg !33
  %812 = extractelement <4 x float> %710, i64 0, !dbg !32
  %813 = call float @llvm.amdgcn.exp2.f32(float %812), !dbg !32
  %814 = insertelement <4 x float> poison, float %813, i64 0, !dbg !34
  %815 = fadd float %811, %813, !dbg !33
  %816 = extractelement <4 x float> %710, i64 1, !dbg !32
  %817 = call float @llvm.amdgcn.exp2.f32(float %816), !dbg !32
  %818 = insertelement <4 x float> %814, float %817, i64 1, !dbg !34
  %819 = fadd float %815, %817, !dbg !33
  %820 = extractelement <4 x float> %710, i64 2, !dbg !32
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !32
  %822 = insertelement <4 x float> %818, float %821, i64 2, !dbg !34
  %823 = fadd float %819, %821, !dbg !33
  %824 = extractelement <4 x float> %710, i64 3, !dbg !32
  %825 = call float @llvm.amdgcn.exp2.f32(float %824), !dbg !32
  %826 = insertelement <4 x float> %822, float %825, i64 3, !dbg !34
  %math.result170 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %826) #9, !dbg !34, !frisk.register_lifetimes !1
  %827 = fptrunc <4 x float> %math.result170 to <4 x half>, !dbg !34
  %math.packed171 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %827) #9, !dbg !33, !frisk.register_lifetimes !1
  %828 = fadd float %823, %825, !dbg !33
  %829 = extractelement <4 x float> %711, i64 0, !dbg !32
  %830 = call float @llvm.amdgcn.exp2.f32(float %829), !dbg !32
  %831 = insertelement <4 x float> poison, float %830, i64 0, !dbg !34
  %832 = fadd float %828, %830, !dbg !33
  %833 = extractelement <4 x float> %711, i64 1, !dbg !32
  %834 = call float @llvm.amdgcn.exp2.f32(float %833), !dbg !32
  %835 = insertelement <4 x float> %831, float %834, i64 1, !dbg !34
  %836 = fadd float %832, %834, !dbg !33
  %837 = extractelement <4 x float> %711, i64 2, !dbg !32
  %838 = call float @llvm.amdgcn.exp2.f32(float %837), !dbg !32
  %839 = insertelement <4 x float> %835, float %838, i64 2, !dbg !34
  %840 = fadd float %836, %838, !dbg !33
  %841 = extractelement <4 x float> %711, i64 3, !dbg !32
  %842 = call float @llvm.amdgcn.exp2.f32(float %841), !dbg !32
  %843 = insertelement <4 x float> %839, float %842, i64 3, !dbg !34
  %math.result172 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %843) #9, !dbg !34, !frisk.register_lifetimes !1
  %844 = fptrunc <4 x float> %math.result172 to <4 x half>, !dbg !34
  %math.packed173 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %844) #9, !dbg !33, !frisk.register_lifetimes !1
  %845 = fadd float %840, %842, !dbg !33
  %846 = extractelement <4 x float> %712, i64 0, !dbg !32
  %847 = call float @llvm.amdgcn.exp2.f32(float %846), !dbg !32
  %848 = insertelement <4 x float> poison, float %847, i64 0, !dbg !34
  %849 = fadd float %845, %847, !dbg !33
  %850 = extractelement <4 x float> %712, i64 1, !dbg !32
  %851 = call float @llvm.amdgcn.exp2.f32(float %850), !dbg !32
  %852 = insertelement <4 x float> %848, float %851, i64 1, !dbg !34
  %853 = fadd float %849, %851, !dbg !33
  %854 = extractelement <4 x float> %712, i64 2, !dbg !32
  %855 = call float @llvm.amdgcn.exp2.f32(float %854), !dbg !32
  %856 = insertelement <4 x float> %852, float %855, i64 2, !dbg !34
  %857 = fadd float %853, %855, !dbg !33
  %858 = extractelement <4 x float> %712, i64 3, !dbg !32
  %859 = call float @llvm.amdgcn.exp2.f32(float %858), !dbg !32
  %860 = insertelement <4 x float> %856, float %859, i64 3, !dbg !34
  %math.result174 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %860) #9, !dbg !34, !frisk.register_lifetimes !1
  %861 = fptrunc <4 x float> %math.result174 to <4 x half>, !dbg !34
  %math.packed175 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %861) #9, !dbg !33, !frisk.register_lifetimes !1
  %862 = fadd float %857, %859, !dbg !33
  %863 = bitcast float %795 to i32, !dbg !33
  %864 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %863), !dbg !33
  %865 = bitcast i32 %864 to float, !dbg !33
  %866 = fadd float %795, %865, !dbg !33
  %867 = bitcast float %866 to i32, !dbg !33
  %868 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %867), !dbg !33
  %869 = bitcast i32 %868 to float, !dbg !33
  %870 = fadd float %866, %869, !dbg !33
  %871 = fadd float %870, %720, !dbg !33
  %872 = insertelement <1 x float> poison, float %871, i64 0, !dbg !15
  %873 = insertvalue [2 x <1 x float>] poison, <1 x float> %872, 0, !dbg !15
  %874 = bitcast float %862 to i32, !dbg !33
  %875 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %874), !dbg !33
  %876 = bitcast i32 %875 to float, !dbg !33
  %877 = fadd float %862, %876, !dbg !33
  %878 = bitcast float %877 to i32, !dbg !33
  %879 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %878), !dbg !33
  %880 = bitcast i32 %879 to float, !dbg !33
  %881 = fadd float %877, %880, !dbg !33
  %882 = fadd float %881, %728, !dbg !33
  %883 = insertelement <1 x float> poison, float %882, i64 0, !dbg !15
  %884 = insertvalue [2 x <1 x float>] %873, <1 x float> %883, 1, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread496 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !14, !range !39, !frisk.register_lifetimes !1
  %address.local497 = zext nneg i32 %address.thread496 to i64, !dbg !8
  %address.local498 = and i64 %address.local497, 60, !dbg !14
  %address.local499 = add nuw nsw i64 %address.local498, %address.local497, !dbg !14
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !14
  %address.local501 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !14
  %address.local502 = and i64 %address.local497, 63, !dbg !7
  %address.local503 = lshr i64 %address.local502, 4, !dbg !14
  %address.local504 = mul nuw nsw i64 %address.local503, 384, !dbg !14
  %address.local505 = getelementptr [2 x i8], ptr addrspace(3) %address.local501, i64 %address.local504, !dbg !14
  %address.local506 = lshr i64 %address.local497, 2, !dbg !17
  %address.local507 = mul nsw i64 %address.local506, -32, !dbg !14
  %address.local508 = getelementptr [2 x i8], ptr addrspace(3) %address.local505, i64 %address.local507, !dbg !14
  %885 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !14
  %886 = bitcast <4 x i32> %885 to <8 x half>, !dbg !14
  %address.local509 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !14
  %address.local510 = getelementptr [2 x i8], ptr addrspace(3) %address.local509, i64 %address.local504, !dbg !14
  %address.local511 = getelementptr [2 x i8], ptr addrspace(3) %address.local510, i64 %address.local507, !dbg !14
  %address.local512 = getelementptr i8, ptr addrspace(3) %address.local511, i64 256, !dbg !14
  %887 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !14
  %888 = bitcast <4 x i32> %887 to <8 x half>, !dbg !14
  %address.local513 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !14
  %address.local514 = getelementptr [2 x i8], ptr addrspace(3) %address.local513, i64 %address.local504, !dbg !14
  %address.local515 = getelementptr [2 x i8], ptr addrspace(3) %address.local514, i64 %address.local507, !dbg !14
  %address.local516 = getelementptr i8, ptr addrspace(3) %address.local515, i64 512, !dbg !14
  %889 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !14
  %890 = bitcast <4 x i32> %889 to <8 x half>, !dbg !14
  %address.local517 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !14
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) %address.local517, i64 %address.local504, !dbg !14
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) %address.local518, i64 %address.local507, !dbg !14
  %address.local520 = getelementptr i8, ptr addrspace(3) %address.local519, i64 768, !dbg !14
  %891 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !14
  %892 = bitcast <4 x i32> %891 to <8 x half>, !dbg !14
  %893 = add nuw nsw i64 %219, %16, !dbg !18
  %894 = shl i64 %893, 19, !dbg !18
  %895 = and i64 %894, 16252928, !dbg !18
  %896 = add i64 %895, %225, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %897 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %896, !dbg !18
  %898 = getelementptr i8, ptr addrspace(1) %897, i64 64, !dbg !18
  %address.local521 = shl nuw nsw i64 %address.local497, 3, !dbg !8
  %address.local522 = and i64 %address.local521, 1536, !dbg !16
  %address.local523 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local522, !dbg !16
  %address.local524 = ptrtoint ptr addrspace(3) %address.local523 to i64, !dbg !16
  %address.local525 = trunc i64 %address.local524 to i32, !dbg !16
  %899 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !18
  %900 = zext i32 %899 to i64, !dbg !18
  %901 = inttoptr i64 %900 to ptr addrspace(3), !dbg !18
  %address.local526 = shl nuw nsw i64 %address.local497, 5, !dbg !18
  %address.local527 = and i64 %address.local526, 8064, !dbg !18
  %address.local528 = shl nuw nsw i64 %address.local497, 3, !dbg !18
  %address.local529 = and i64 %address.local528, 24, !dbg !18
  %address.local530 = or disjoint i64 %address.local527, %address.local529, !dbg !18
  %address.local531 = getelementptr [2 x i8], ptr addrspace(1) %898, i64 %address.local530, !dbg !18
  %902 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %903 = ptrtoint ptr addrspace(1) %address.local531 to i64, !dbg !18
  %904 = sub i64 %903, %902, !dbg !18
  %dma.buffer.offset957 = trunc i64 %904 to i32, !dbg !18
  %905 = trunc i64 %902 to i32, !dbg !18
  %906 = insertelement <4 x i32> poison, i32 %905, i64 0, !dbg !18
  %907 = lshr i64 %902, 32, !dbg !18
  %908 = trunc i64 %907 to i32, !dbg !18
  %909 = insertelement <4 x i32> %906, i32 %908, i64 1, !dbg !18
  %910 = insertelement <4 x i32> %909, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc958 = insertelement <4 x i32> %910, i32 131072, i64 3, !dbg !18
  %911 = ptrtoint ptr addrspace(3) %901 to i32, !dbg !18
  %dma.buffer.lds959 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %911), !dbg !18
  %912 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds959, i32 %dma.buffer.offset957, <4 x i32> %dma.buffer.rsrc958) #10, !dbg !18
  %913 = shufflevector <8 x half> %886, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %913, <4 x float> zeroinitializer) #9, !dbg !14
  %915 = shufflevector <8 x half> %888, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %915, <4 x float> %914) #9, !dbg !14
  %917 = shufflevector <8 x half> %890, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %917, <4 x float> %916) #9, !dbg !14
  %919 = shufflevector <8 x half> %892, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %919, <4 x float> %918) #9, !dbg !14
  %921 = extractelement <4 x float> %920, i64 0, !dbg !14
  %922 = extractelement <4 x float> %920, i64 1, !dbg !14
  %923 = extractelement <4 x float> %920, i64 2, !dbg !14
  %924 = extractelement <4 x float> %920, i64 3, !dbg !14
  %925 = select i1 %.not, float %922, float %921, !dbg !14
  %926 = bitcast float %925 to i32, !dbg !14
  %927 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %926), !dbg !14
  %928 = bitcast i32 %927 to float, !dbg !14
  %929 = select i1 %.not, float %921, float %928, !dbg !14
  %930 = select i1 %.not, float %928, float %922, !dbg !14
  %931 = select i1 %.not, float %924, float %923, !dbg !14
  %932 = bitcast float %931 to i32, !dbg !14
  %933 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %932), !dbg !14
  %934 = bitcast i32 %933 to float, !dbg !14
  %935 = select i1 %.not, float %923, float %934, !dbg !14
  %936 = select i1 %.not, float %934, float %924, !dbg !14
  %937 = select i1 %.not1, float %935, float %929, !dbg !14
  %938 = bitcast float %937 to i32, !dbg !14
  %939 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %938), !dbg !14
  %940 = bitcast i32 %939 to float, !dbg !14
  %941 = select i1 %.not1, float %929, float %940, !dbg !14
  %942 = select i1 %.not1, float %940, float %935, !dbg !14
  %943 = select i1 %.not1, float %936, float %930, !dbg !14
  %944 = bitcast float %943 to i32, !dbg !14
  %945 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %944), !dbg !14
  %946 = bitcast i32 %945 to float, !dbg !14
  %947 = select i1 %.not1, float %930, float %946, !dbg !14
  %948 = select i1 %.not1, float %946, float %936, !dbg !14
  %949 = extractvalue [2 x <8 x float>] %209, 0, !dbg !35
  %950 = shufflevector <8 x float> %949, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %951 = fmul <4 x float> %950, %717, !dbg !35
  %952 = insertelement <4 x float> poison, float %941, i64 0, !dbg !36
  %953 = insertelement <4 x float> %952, float %947, i64 1, !dbg !36
  %954 = insertelement <4 x float> %953, float %942, i64 2, !dbg !36
  %955 = insertelement <4 x float> %954, float %948, i64 3, !dbg !36
  %956 = fadd <4 x float> %951, %955, !dbg !36
  %957 = shufflevector <8 x half> %886, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %957, <4 x float> zeroinitializer) #9, !dbg !14
  %959 = shufflevector <8 x half> %888, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %959, <4 x float> %958) #9, !dbg !14
  %961 = shufflevector <8 x half> %890, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %961, <4 x float> %960) #9, !dbg !14
  %963 = shufflevector <8 x half> %892, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %963, <4 x float> %962) #9, !dbg !14
  %965 = extractelement <4 x float> %964, i64 0, !dbg !14
  %966 = extractelement <4 x float> %964, i64 1, !dbg !14
  %967 = extractelement <4 x float> %964, i64 2, !dbg !14
  %968 = extractelement <4 x float> %964, i64 3, !dbg !14
  %969 = select i1 %.not, float %966, float %965, !dbg !14
  %970 = bitcast float %969 to i32, !dbg !14
  %971 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %970), !dbg !14
  %972 = bitcast i32 %971 to float, !dbg !14
  %973 = select i1 %.not, float %965, float %972, !dbg !14
  %974 = select i1 %.not, float %972, float %966, !dbg !14
  %975 = select i1 %.not, float %968, float %967, !dbg !14
  %976 = bitcast float %975 to i32, !dbg !14
  %977 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %976), !dbg !14
  %978 = bitcast i32 %977 to float, !dbg !14
  %979 = select i1 %.not, float %967, float %978, !dbg !14
  %980 = select i1 %.not, float %978, float %968, !dbg !14
  %981 = select i1 %.not1, float %979, float %973, !dbg !14
  %982 = bitcast float %981 to i32, !dbg !14
  %983 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %982), !dbg !14
  %984 = bitcast i32 %983 to float, !dbg !14
  %985 = select i1 %.not1, float %973, float %984, !dbg !14
  %986 = select i1 %.not1, float %984, float %979, !dbg !14
  %987 = select i1 %.not1, float %980, float %974, !dbg !14
  %988 = bitcast float %987 to i32, !dbg !14
  %989 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %988), !dbg !14
  %990 = bitcast i32 %989 to float, !dbg !14
  %991 = select i1 %.not1, float %974, float %990, !dbg !14
  %992 = select i1 %.not1, float %990, float %980, !dbg !14
  %993 = shufflevector <8 x float> %949, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %994 = fmul <4 x float> %993, %717, !dbg !35
  %995 = insertelement <4 x float> poison, float %985, i64 0, !dbg !36
  %996 = insertelement <4 x float> %995, float %991, i64 1, !dbg !36
  %997 = insertelement <4 x float> %996, float %986, i64 2, !dbg !36
  %998 = insertelement <4 x float> %997, float %992, i64 3, !dbg !36
  %999 = fadd <4 x float> %994, %998, !dbg !36
  %1000 = shufflevector <4 x float> %956, <4 x float> %999, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1001 = insertvalue [2 x <8 x float>] poison, <8 x float> %1000, 0, !dbg !15
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %913, <4 x float> zeroinitializer) #9, !dbg !14
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %915, <4 x float> %1002) #9, !dbg !14
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %917, <4 x float> %1003) #9, !dbg !14
  %1005 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %919, <4 x float> %1004) #9, !dbg !14
  %1006 = extractelement <4 x float> %1005, i64 0, !dbg !14
  %1007 = extractelement <4 x float> %1005, i64 1, !dbg !14
  %1008 = extractelement <4 x float> %1005, i64 2, !dbg !14
  %1009 = extractelement <4 x float> %1005, i64 3, !dbg !14
  %1010 = select i1 %.not, float %1007, float %1006, !dbg !14
  %1011 = bitcast float %1010 to i32, !dbg !14
  %1012 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1011), !dbg !14
  %1013 = bitcast i32 %1012 to float, !dbg !14
  %1014 = select i1 %.not, float %1006, float %1013, !dbg !14
  %1015 = select i1 %.not, float %1013, float %1007, !dbg !14
  %1016 = select i1 %.not, float %1009, float %1008, !dbg !14
  %1017 = bitcast float %1016 to i32, !dbg !14
  %1018 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1017), !dbg !14
  %1019 = bitcast i32 %1018 to float, !dbg !14
  %1020 = select i1 %.not, float %1008, float %1019, !dbg !14
  %1021 = select i1 %.not, float %1019, float %1009, !dbg !14
  %1022 = select i1 %.not1, float %1020, float %1014, !dbg !14
  %1023 = bitcast float %1022 to i32, !dbg !14
  %1024 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1023), !dbg !14
  %1025 = bitcast i32 %1024 to float, !dbg !14
  %1026 = select i1 %.not1, float %1014, float %1025, !dbg !14
  %1027 = select i1 %.not1, float %1025, float %1020, !dbg !14
  %1028 = select i1 %.not1, float %1021, float %1015, !dbg !14
  %1029 = bitcast float %1028 to i32, !dbg !14
  %1030 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1029), !dbg !14
  %1031 = bitcast i32 %1030 to float, !dbg !14
  %1032 = select i1 %.not1, float %1015, float %1031, !dbg !14
  %1033 = select i1 %.not1, float %1031, float %1021, !dbg !14
  %1034 = extractvalue [2 x <8 x float>] %209, 1, !dbg !35
  %1035 = shufflevector <8 x float> %1034, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1036 = fmul <4 x float> %1035, %725, !dbg !35
  %1037 = insertelement <4 x float> poison, float %1026, i64 0, !dbg !36
  %1038 = insertelement <4 x float> %1037, float %1032, i64 1, !dbg !36
  %1039 = insertelement <4 x float> %1038, float %1027, i64 2, !dbg !36
  %1040 = insertelement <4 x float> %1039, float %1033, i64 3, !dbg !36
  %1041 = fadd <4 x float> %1036, %1040, !dbg !36
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %957, <4 x float> zeroinitializer) #9, !dbg !14
  %1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %959, <4 x float> %1042) #9, !dbg !14
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %961, <4 x float> %1043) #9, !dbg !14
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %963, <4 x float> %1044) #9, !dbg !14
  %1046 = extractelement <4 x float> %1045, i64 0, !dbg !14
  %1047 = extractelement <4 x float> %1045, i64 1, !dbg !14
  %1048 = extractelement <4 x float> %1045, i64 2, !dbg !14
  %1049 = extractelement <4 x float> %1045, i64 3, !dbg !14
  %1050 = select i1 %.not, float %1047, float %1046, !dbg !14
  %1051 = bitcast float %1050 to i32, !dbg !14
  %1052 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1051), !dbg !14
  %1053 = bitcast i32 %1052 to float, !dbg !14
  %1054 = select i1 %.not, float %1046, float %1053, !dbg !14
  %1055 = select i1 %.not, float %1053, float %1047, !dbg !14
  %1056 = select i1 %.not, float %1049, float %1048, !dbg !14
  %1057 = bitcast float %1056 to i32, !dbg !14
  %1058 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1057), !dbg !14
  %1059 = bitcast i32 %1058 to float, !dbg !14
  %1060 = select i1 %.not, float %1048, float %1059, !dbg !14
  %1061 = select i1 %.not, float %1059, float %1049, !dbg !14
  %1062 = select i1 %.not1, float %1060, float %1054, !dbg !14
  %1063 = bitcast float %1062 to i32, !dbg !14
  %1064 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1063), !dbg !14
  %1065 = bitcast i32 %1064 to float, !dbg !14
  %1066 = select i1 %.not1, float %1054, float %1065, !dbg !14
  %1067 = select i1 %.not1, float %1065, float %1060, !dbg !14
  %1068 = select i1 %.not1, float %1061, float %1055, !dbg !14
  %1069 = bitcast float %1068 to i32, !dbg !14
  %1070 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1069), !dbg !14
  %1071 = bitcast i32 %1070 to float, !dbg !14
  %1072 = select i1 %.not1, float %1055, float %1071, !dbg !14
  %1073 = select i1 %.not1, float %1071, float %1061, !dbg !14
  %1074 = shufflevector <8 x float> %1034, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1075 = fmul <4 x float> %1074, %725, !dbg !35
  %1076 = insertelement <4 x float> poison, float %1066, i64 0, !dbg !36
  %1077 = insertelement <4 x float> %1076, float %1072, i64 1, !dbg !36
  %1078 = insertelement <4 x float> %1077, float %1067, i64 2, !dbg !36
  %1079 = insertelement <4 x float> %1078, float %1073, i64 3, !dbg !36
  %1080 = fadd <4 x float> %1075, %1079, !dbg !36
  %1081 = shufflevector <4 x float> %1041, <4 x float> %1080, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1082 = insertvalue [2 x <8 x float>] %1001, <8 x float> %1081, 1, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1083 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !14
  %1084 = bitcast <4 x i32> %1083 to <8 x half>, !dbg !14
  %1085 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !14
  %1086 = bitcast <4 x i32> %1085 to <8 x half>, !dbg !14
  %1087 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !14
  %1088 = bitcast <4 x i32> %1087 to <8 x half>, !dbg !14
  %1089 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !14
  %1090 = bitcast <4 x i32> %1089 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1091 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %896, !dbg !18
  %1092 = getelementptr i8, ptr addrspace(1) %1091, i64 128, !dbg !18
  %1093 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !18
  %1094 = zext i32 %1093 to i64, !dbg !18
  %1095 = inttoptr i64 %1094 to ptr addrspace(3), !dbg !18
  %address.local532 = getelementptr [2 x i8], ptr addrspace(1) %1092, i64 %address.local530, !dbg !18
  %1096 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1097 = ptrtoint ptr addrspace(1) %address.local532 to i64, !dbg !18
  %1098 = sub i64 %1097, %1096, !dbg !18
  %dma.buffer.offset954 = trunc i64 %1098 to i32, !dbg !18
  %1099 = trunc i64 %1096 to i32, !dbg !18
  %1100 = insertelement <4 x i32> poison, i32 %1099, i64 0, !dbg !18
  %1101 = lshr i64 %1096, 32, !dbg !18
  %1102 = trunc i64 %1101 to i32, !dbg !18
  %1103 = insertelement <4 x i32> %1100, i32 %1102, i64 1, !dbg !18
  %1104 = insertelement <4 x i32> %1103, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc955 = insertelement <4 x i32> %1104, i32 131072, i64 3, !dbg !18
  %1105 = ptrtoint ptr addrspace(3) %1095 to i32, !dbg !18
  %dma.buffer.lds956 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1105), !dbg !18
  %1106 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds956, i32 %dma.buffer.offset954, <4 x i32> %dma.buffer.rsrc955) #10, !dbg !18
  %1107 = shufflevector <8 x half> %1084, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1107, <4 x float> zeroinitializer) #9, !dbg !14
  %1109 = shufflevector <8 x half> %1086, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1109, <4 x float> %1108) #9, !dbg !14
  %1111 = shufflevector <8 x half> %1088, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1111, <4 x float> %1110) #9, !dbg !14
  %1113 = shufflevector <8 x half> %1090, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1113, <4 x float> %1112) #9, !dbg !14
  %1115 = extractelement <4 x float> %1114, i64 0, !dbg !14
  %1116 = extractelement <4 x float> %1114, i64 1, !dbg !14
  %1117 = extractelement <4 x float> %1114, i64 2, !dbg !14
  %1118 = extractelement <4 x float> %1114, i64 3, !dbg !14
  %1119 = select i1 %.not, float %1116, float %1115, !dbg !14
  %1120 = bitcast float %1119 to i32, !dbg !14
  %1121 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1120), !dbg !14
  %1122 = bitcast i32 %1121 to float, !dbg !14
  %1123 = select i1 %.not, float %1115, float %1122, !dbg !14
  %1124 = select i1 %.not, float %1122, float %1116, !dbg !14
  %1125 = select i1 %.not, float %1118, float %1117, !dbg !14
  %1126 = bitcast float %1125 to i32, !dbg !14
  %1127 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1126), !dbg !14
  %1128 = bitcast i32 %1127 to float, !dbg !14
  %1129 = select i1 %.not, float %1117, float %1128, !dbg !14
  %1130 = select i1 %.not, float %1128, float %1118, !dbg !14
  %1131 = select i1 %.not1, float %1129, float %1123, !dbg !14
  %1132 = bitcast float %1131 to i32, !dbg !14
  %1133 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1132), !dbg !14
  %1134 = bitcast i32 %1133 to float, !dbg !14
  %1135 = select i1 %.not1, float %1123, float %1134, !dbg !14
  %1136 = select i1 %.not1, float %1134, float %1129, !dbg !14
  %1137 = select i1 %.not1, float %1130, float %1124, !dbg !14
  %1138 = bitcast float %1137 to i32, !dbg !14
  %1139 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1138), !dbg !14
  %1140 = bitcast i32 %1139 to float, !dbg !14
  %1141 = select i1 %.not1, float %1124, float %1140, !dbg !14
  %1142 = select i1 %.not1, float %1140, float %1130, !dbg !14
  %1143 = extractvalue [2 x <8 x float>] %211, 0, !dbg !35
  %1144 = shufflevector <8 x float> %1143, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1145 = fmul <4 x float> %1144, %717, !dbg !35
  %1146 = insertelement <4 x float> poison, float %1135, i64 0, !dbg !36
  %1147 = insertelement <4 x float> %1146, float %1141, i64 1, !dbg !36
  %1148 = insertelement <4 x float> %1147, float %1136, i64 2, !dbg !36
  %1149 = insertelement <4 x float> %1148, float %1142, i64 3, !dbg !36
  %1150 = fadd <4 x float> %1145, %1149, !dbg !36
  %1151 = shufflevector <8 x half> %1084, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1151, <4 x float> zeroinitializer) #9, !dbg !14
  %1153 = shufflevector <8 x half> %1086, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1153, <4 x float> %1152) #9, !dbg !14
  %1155 = shufflevector <8 x half> %1088, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1155, <4 x float> %1154) #9, !dbg !14
  %1157 = shufflevector <8 x half> %1090, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1157, <4 x float> %1156) #9, !dbg !14
  %1159 = extractelement <4 x float> %1158, i64 0, !dbg !14
  %1160 = extractelement <4 x float> %1158, i64 1, !dbg !14
  %1161 = extractelement <4 x float> %1158, i64 2, !dbg !14
  %1162 = extractelement <4 x float> %1158, i64 3, !dbg !14
  %1163 = select i1 %.not, float %1160, float %1159, !dbg !14
  %1164 = bitcast float %1163 to i32, !dbg !14
  %1165 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1164), !dbg !14
  %1166 = bitcast i32 %1165 to float, !dbg !14
  %1167 = select i1 %.not, float %1159, float %1166, !dbg !14
  %1168 = select i1 %.not, float %1166, float %1160, !dbg !14
  %1169 = select i1 %.not, float %1162, float %1161, !dbg !14
  %1170 = bitcast float %1169 to i32, !dbg !14
  %1171 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1170), !dbg !14
  %1172 = bitcast i32 %1171 to float, !dbg !14
  %1173 = select i1 %.not, float %1161, float %1172, !dbg !14
  %1174 = select i1 %.not, float %1172, float %1162, !dbg !14
  %1175 = select i1 %.not1, float %1173, float %1167, !dbg !14
  %1176 = bitcast float %1175 to i32, !dbg !14
  %1177 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1176), !dbg !14
  %1178 = bitcast i32 %1177 to float, !dbg !14
  %1179 = select i1 %.not1, float %1167, float %1178, !dbg !14
  %1180 = select i1 %.not1, float %1178, float %1173, !dbg !14
  %1181 = select i1 %.not1, float %1174, float %1168, !dbg !14
  %1182 = bitcast float %1181 to i32, !dbg !14
  %1183 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1182), !dbg !14
  %1184 = bitcast i32 %1183 to float, !dbg !14
  %1185 = select i1 %.not1, float %1168, float %1184, !dbg !14
  %1186 = select i1 %.not1, float %1184, float %1174, !dbg !14
  %1187 = shufflevector <8 x float> %1143, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1188 = fmul <4 x float> %1187, %717, !dbg !35
  %1189 = insertelement <4 x float> poison, float %1179, i64 0, !dbg !36
  %1190 = insertelement <4 x float> %1189, float %1185, i64 1, !dbg !36
  %1191 = insertelement <4 x float> %1190, float %1180, i64 2, !dbg !36
  %1192 = insertelement <4 x float> %1191, float %1186, i64 3, !dbg !36
  %1193 = fadd <4 x float> %1188, %1192, !dbg !36
  %1194 = shufflevector <4 x float> %1150, <4 x float> %1193, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1195 = insertvalue [2 x <8 x float>] poison, <8 x float> %1194, 0, !dbg !15
  %1196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1107, <4 x float> zeroinitializer) #9, !dbg !14
  %1197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1109, <4 x float> %1196) #9, !dbg !14
  %1198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1111, <4 x float> %1197) #9, !dbg !14
  %1199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1113, <4 x float> %1198) #9, !dbg !14
  %1200 = extractelement <4 x float> %1199, i64 0, !dbg !14
  %1201 = extractelement <4 x float> %1199, i64 1, !dbg !14
  %1202 = extractelement <4 x float> %1199, i64 2, !dbg !14
  %1203 = extractelement <4 x float> %1199, i64 3, !dbg !14
  %1204 = select i1 %.not, float %1201, float %1200, !dbg !14
  %1205 = bitcast float %1204 to i32, !dbg !14
  %1206 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1205), !dbg !14
  %1207 = bitcast i32 %1206 to float, !dbg !14
  %1208 = select i1 %.not, float %1200, float %1207, !dbg !14
  %1209 = select i1 %.not, float %1207, float %1201, !dbg !14
  %1210 = select i1 %.not, float %1203, float %1202, !dbg !14
  %1211 = bitcast float %1210 to i32, !dbg !14
  %1212 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1211), !dbg !14
  %1213 = bitcast i32 %1212 to float, !dbg !14
  %1214 = select i1 %.not, float %1202, float %1213, !dbg !14
  %1215 = select i1 %.not, float %1213, float %1203, !dbg !14
  %1216 = select i1 %.not1, float %1214, float %1208, !dbg !14
  %1217 = bitcast float %1216 to i32, !dbg !14
  %1218 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1217), !dbg !14
  %1219 = bitcast i32 %1218 to float, !dbg !14
  %1220 = select i1 %.not1, float %1208, float %1219, !dbg !14
  %1221 = select i1 %.not1, float %1219, float %1214, !dbg !14
  %1222 = select i1 %.not1, float %1215, float %1209, !dbg !14
  %1223 = bitcast float %1222 to i32, !dbg !14
  %1224 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1223), !dbg !14
  %1225 = bitcast i32 %1224 to float, !dbg !14
  %1226 = select i1 %.not1, float %1209, float %1225, !dbg !14
  %1227 = select i1 %.not1, float %1225, float %1215, !dbg !14
  %1228 = extractvalue [2 x <8 x float>] %211, 1, !dbg !35
  %1229 = shufflevector <8 x float> %1228, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1230 = fmul <4 x float> %1229, %725, !dbg !35
  %1231 = insertelement <4 x float> poison, float %1220, i64 0, !dbg !36
  %1232 = insertelement <4 x float> %1231, float %1226, i64 1, !dbg !36
  %1233 = insertelement <4 x float> %1232, float %1221, i64 2, !dbg !36
  %1234 = insertelement <4 x float> %1233, float %1227, i64 3, !dbg !36
  %1235 = fadd <4 x float> %1230, %1234, !dbg !36
  %1236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1151, <4 x float> zeroinitializer) #9, !dbg !14
  %1237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1153, <4 x float> %1236) #9, !dbg !14
  %1238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1155, <4 x float> %1237) #9, !dbg !14
  %1239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1157, <4 x float> %1238) #9, !dbg !14
  %1240 = extractelement <4 x float> %1239, i64 0, !dbg !14
  %1241 = extractelement <4 x float> %1239, i64 1, !dbg !14
  %1242 = extractelement <4 x float> %1239, i64 2, !dbg !14
  %1243 = extractelement <4 x float> %1239, i64 3, !dbg !14
  %1244 = select i1 %.not, float %1241, float %1240, !dbg !14
  %1245 = bitcast float %1244 to i32, !dbg !14
  %1246 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1245), !dbg !14
  %1247 = bitcast i32 %1246 to float, !dbg !14
  %1248 = select i1 %.not, float %1240, float %1247, !dbg !14
  %1249 = select i1 %.not, float %1247, float %1241, !dbg !14
  %1250 = select i1 %.not, float %1243, float %1242, !dbg !14
  %1251 = bitcast float %1250 to i32, !dbg !14
  %1252 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1251), !dbg !14
  %1253 = bitcast i32 %1252 to float, !dbg !14
  %1254 = select i1 %.not, float %1242, float %1253, !dbg !14
  %1255 = select i1 %.not, float %1253, float %1243, !dbg !14
  %1256 = select i1 %.not1, float %1254, float %1248, !dbg !14
  %1257 = bitcast float %1256 to i32, !dbg !14
  %1258 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1257), !dbg !14
  %1259 = bitcast i32 %1258 to float, !dbg !14
  %1260 = select i1 %.not1, float %1248, float %1259, !dbg !14
  %1261 = select i1 %.not1, float %1259, float %1254, !dbg !14
  %1262 = select i1 %.not1, float %1255, float %1249, !dbg !14
  %1263 = bitcast float %1262 to i32, !dbg !14
  %1264 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1263), !dbg !14
  %1265 = bitcast i32 %1264 to float, !dbg !14
  %1266 = select i1 %.not1, float %1249, float %1265, !dbg !14
  %1267 = select i1 %.not1, float %1265, float %1255, !dbg !14
  %1268 = shufflevector <8 x float> %1228, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1269 = fmul <4 x float> %1268, %725, !dbg !35
  %1270 = insertelement <4 x float> poison, float %1260, i64 0, !dbg !36
  %1271 = insertelement <4 x float> %1270, float %1266, i64 1, !dbg !36
  %1272 = insertelement <4 x float> %1271, float %1261, i64 2, !dbg !36
  %1273 = insertelement <4 x float> %1272, float %1267, i64 3, !dbg !36
  %1274 = fadd <4 x float> %1269, %1273, !dbg !36
  %1275 = shufflevector <4 x float> %1235, <4 x float> %1274, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1276 = insertvalue [2 x <8 x float>] %1195, <8 x float> %1275, 1, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1277 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !14
  %1278 = bitcast <4 x i32> %1277 to <8 x half>, !dbg !14
  %1279 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !14
  %1280 = bitcast <4 x i32> %1279 to <8 x half>, !dbg !14
  %1281 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !14
  %1282 = bitcast <4 x i32> %1281 to <8 x half>, !dbg !14
  %1283 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !14
  %1284 = bitcast <4 x i32> %1283 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1285 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %896, !dbg !18
  %1286 = getelementptr i8, ptr addrspace(1) %1285, i64 192, !dbg !18
  %1287 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !18
  %1288 = zext i32 %1287 to i64, !dbg !18
  %1289 = inttoptr i64 %1288 to ptr addrspace(3), !dbg !18
  %address.local533 = getelementptr [2 x i8], ptr addrspace(1) %1286, i64 %address.local530, !dbg !18
  %1290 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1291 = ptrtoint ptr addrspace(1) %address.local533 to i64, !dbg !18
  %1292 = sub i64 %1291, %1290, !dbg !18
  %dma.buffer.offset = trunc i64 %1292 to i32, !dbg !18
  %1293 = trunc i64 %1290 to i32, !dbg !18
  %1294 = insertelement <4 x i32> poison, i32 %1293, i64 0, !dbg !18
  %1295 = lshr i64 %1290, 32, !dbg !18
  %1296 = trunc i64 %1295 to i32, !dbg !18
  %1297 = insertelement <4 x i32> %1294, i32 %1296, i64 1, !dbg !18
  %1298 = insertelement <4 x i32> %1297, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1298, i32 131072, i64 3, !dbg !18
  %1299 = ptrtoint ptr addrspace(3) %1289 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1299), !dbg !18
  %1300 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1301 = shufflevector <8 x half> %1278, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1301, <4 x float> zeroinitializer) #9, !dbg !14
  %1303 = shufflevector <8 x half> %1280, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1303, <4 x float> %1302) #9, !dbg !14
  %1305 = shufflevector <8 x half> %1282, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1305, <4 x float> %1304) #9, !dbg !14
  %1307 = shufflevector <8 x half> %1284, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1307, <4 x float> %1306) #9, !dbg !14
  %1309 = extractelement <4 x float> %1308, i64 0, !dbg !14
  %1310 = extractelement <4 x float> %1308, i64 1, !dbg !14
  %1311 = extractelement <4 x float> %1308, i64 2, !dbg !14
  %1312 = extractelement <4 x float> %1308, i64 3, !dbg !14
  %1313 = select i1 %.not, float %1310, float %1309, !dbg !14
  %1314 = bitcast float %1313 to i32, !dbg !14
  %1315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1314), !dbg !14
  %1316 = bitcast i32 %1315 to float, !dbg !14
  %1317 = select i1 %.not, float %1309, float %1316, !dbg !14
  %1318 = select i1 %.not, float %1316, float %1310, !dbg !14
  %1319 = select i1 %.not, float %1312, float %1311, !dbg !14
  %1320 = bitcast float %1319 to i32, !dbg !14
  %1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1320), !dbg !14
  %1322 = bitcast i32 %1321 to float, !dbg !14
  %1323 = select i1 %.not, float %1311, float %1322, !dbg !14
  %1324 = select i1 %.not, float %1322, float %1312, !dbg !14
  %1325 = select i1 %.not1, float %1323, float %1317, !dbg !14
  %1326 = bitcast float %1325 to i32, !dbg !14
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1326), !dbg !14
  %1328 = bitcast i32 %1327 to float, !dbg !14
  %1329 = select i1 %.not1, float %1317, float %1328, !dbg !14
  %1330 = select i1 %.not1, float %1328, float %1323, !dbg !14
  %1331 = select i1 %.not1, float %1324, float %1318, !dbg !14
  %1332 = bitcast float %1331 to i32, !dbg !14
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1332), !dbg !14
  %1334 = bitcast i32 %1333 to float, !dbg !14
  %1335 = select i1 %.not1, float %1318, float %1334, !dbg !14
  %1336 = select i1 %.not1, float %1334, float %1324, !dbg !14
  %1337 = extractvalue [2 x <8 x float>] %212, 0, !dbg !35
  %1338 = shufflevector <8 x float> %1337, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1339 = fmul <4 x float> %1338, %717, !dbg !35
  %1340 = insertelement <4 x float> poison, float %1329, i64 0, !dbg !36
  %1341 = insertelement <4 x float> %1340, float %1335, i64 1, !dbg !36
  %1342 = insertelement <4 x float> %1341, float %1330, i64 2, !dbg !36
  %1343 = insertelement <4 x float> %1342, float %1336, i64 3, !dbg !36
  %1344 = fadd <4 x float> %1339, %1343, !dbg !36
  %1345 = shufflevector <8 x half> %1278, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1345, <4 x float> zeroinitializer) #9, !dbg !14
  %1347 = shufflevector <8 x half> %1280, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1347, <4 x float> %1346) #9, !dbg !14
  %1349 = shufflevector <8 x half> %1282, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1349, <4 x float> %1348) #9, !dbg !14
  %1351 = shufflevector <8 x half> %1284, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1351, <4 x float> %1350) #9, !dbg !14
  %1353 = extractelement <4 x float> %1352, i64 0, !dbg !14
  %1354 = extractelement <4 x float> %1352, i64 1, !dbg !14
  %1355 = extractelement <4 x float> %1352, i64 2, !dbg !14
  %1356 = extractelement <4 x float> %1352, i64 3, !dbg !14
  %1357 = select i1 %.not, float %1354, float %1353, !dbg !14
  %1358 = bitcast float %1357 to i32, !dbg !14
  %1359 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1358), !dbg !14
  %1360 = bitcast i32 %1359 to float, !dbg !14
  %1361 = select i1 %.not, float %1353, float %1360, !dbg !14
  %1362 = select i1 %.not, float %1360, float %1354, !dbg !14
  %1363 = select i1 %.not, float %1356, float %1355, !dbg !14
  %1364 = bitcast float %1363 to i32, !dbg !14
  %1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1364), !dbg !14
  %1366 = bitcast i32 %1365 to float, !dbg !14
  %1367 = select i1 %.not, float %1355, float %1366, !dbg !14
  %1368 = select i1 %.not, float %1366, float %1356, !dbg !14
  %1369 = select i1 %.not1, float %1367, float %1361, !dbg !14
  %1370 = bitcast float %1369 to i32, !dbg !14
  %1371 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1370), !dbg !14
  %1372 = bitcast i32 %1371 to float, !dbg !14
  %1373 = select i1 %.not1, float %1361, float %1372, !dbg !14
  %1374 = select i1 %.not1, float %1372, float %1367, !dbg !14
  %1375 = select i1 %.not1, float %1368, float %1362, !dbg !14
  %1376 = bitcast float %1375 to i32, !dbg !14
  %1377 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1376), !dbg !14
  %1378 = bitcast i32 %1377 to float, !dbg !14
  %1379 = select i1 %.not1, float %1362, float %1378, !dbg !14
  %1380 = select i1 %.not1, float %1378, float %1368, !dbg !14
  %1381 = shufflevector <8 x float> %1337, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1382 = fmul <4 x float> %1381, %717, !dbg !35
  %1383 = insertelement <4 x float> poison, float %1373, i64 0, !dbg !36
  %1384 = insertelement <4 x float> %1383, float %1379, i64 1, !dbg !36
  %1385 = insertelement <4 x float> %1384, float %1374, i64 2, !dbg !36
  %1386 = insertelement <4 x float> %1385, float %1380, i64 3, !dbg !36
  %1387 = fadd <4 x float> %1382, %1386, !dbg !36
  %1388 = shufflevector <4 x float> %1344, <4 x float> %1387, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1389 = insertvalue [2 x <8 x float>] poison, <8 x float> %1388, 0, !dbg !15
  %1390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1301, <4 x float> zeroinitializer) #9, !dbg !14
  %1391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1303, <4 x float> %1390) #9, !dbg !14
  %1392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1305, <4 x float> %1391) #9, !dbg !14
  %1393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1307, <4 x float> %1392) #9, !dbg !14
  %1394 = extractelement <4 x float> %1393, i64 0, !dbg !14
  %1395 = extractelement <4 x float> %1393, i64 1, !dbg !14
  %1396 = extractelement <4 x float> %1393, i64 2, !dbg !14
  %1397 = extractelement <4 x float> %1393, i64 3, !dbg !14
  %1398 = select i1 %.not, float %1395, float %1394, !dbg !14
  %1399 = bitcast float %1398 to i32, !dbg !14
  %1400 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1399), !dbg !14
  %1401 = bitcast i32 %1400 to float, !dbg !14
  %1402 = select i1 %.not, float %1394, float %1401, !dbg !14
  %1403 = select i1 %.not, float %1401, float %1395, !dbg !14
  %1404 = select i1 %.not, float %1397, float %1396, !dbg !14
  %1405 = bitcast float %1404 to i32, !dbg !14
  %1406 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1405), !dbg !14
  %1407 = bitcast i32 %1406 to float, !dbg !14
  %1408 = select i1 %.not, float %1396, float %1407, !dbg !14
  %1409 = select i1 %.not, float %1407, float %1397, !dbg !14
  %1410 = select i1 %.not1, float %1408, float %1402, !dbg !14
  %1411 = bitcast float %1410 to i32, !dbg !14
  %1412 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1411), !dbg !14
  %1413 = bitcast i32 %1412 to float, !dbg !14
  %1414 = select i1 %.not1, float %1402, float %1413, !dbg !14
  %1415 = select i1 %.not1, float %1413, float %1408, !dbg !14
  %1416 = select i1 %.not1, float %1409, float %1403, !dbg !14
  %1417 = bitcast float %1416 to i32, !dbg !14
  %1418 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1417), !dbg !14
  %1419 = bitcast i32 %1418 to float, !dbg !14
  %1420 = select i1 %.not1, float %1403, float %1419, !dbg !14
  %1421 = select i1 %.not1, float %1419, float %1409, !dbg !14
  %1422 = extractvalue [2 x <8 x float>] %212, 1, !dbg !35
  %1423 = shufflevector <8 x float> %1422, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1424 = fmul <4 x float> %1423, %725, !dbg !35
  %1425 = insertelement <4 x float> poison, float %1414, i64 0, !dbg !36
  %1426 = insertelement <4 x float> %1425, float %1420, i64 1, !dbg !36
  %1427 = insertelement <4 x float> %1426, float %1415, i64 2, !dbg !36
  %1428 = insertelement <4 x float> %1427, float %1421, i64 3, !dbg !36
  %1429 = fadd <4 x float> %1424, %1428, !dbg !36
  %1430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1345, <4 x float> zeroinitializer) #9, !dbg !14
  %1431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1347, <4 x float> %1430) #9, !dbg !14
  %1432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1349, <4 x float> %1431) #9, !dbg !14
  %1433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1351, <4 x float> %1432) #9, !dbg !14
  %1434 = extractelement <4 x float> %1433, i64 0, !dbg !14
  %1435 = extractelement <4 x float> %1433, i64 1, !dbg !14
  %1436 = extractelement <4 x float> %1433, i64 2, !dbg !14
  %1437 = extractelement <4 x float> %1433, i64 3, !dbg !14
  %1438 = select i1 %.not, float %1435, float %1434, !dbg !14
  %1439 = bitcast float %1438 to i32, !dbg !14
  %1440 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1439), !dbg !14
  %1441 = bitcast i32 %1440 to float, !dbg !14
  %1442 = select i1 %.not, float %1434, float %1441, !dbg !14
  %1443 = select i1 %.not, float %1441, float %1435, !dbg !14
  %1444 = select i1 %.not, float %1437, float %1436, !dbg !14
  %1445 = bitcast float %1444 to i32, !dbg !14
  %1446 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1445), !dbg !14
  %1447 = bitcast i32 %1446 to float, !dbg !14
  %1448 = select i1 %.not, float %1436, float %1447, !dbg !14
  %1449 = select i1 %.not, float %1447, float %1437, !dbg !14
  %1450 = select i1 %.not1, float %1448, float %1442, !dbg !14
  %1451 = bitcast float %1450 to i32, !dbg !14
  %1452 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1451), !dbg !14
  %1453 = bitcast i32 %1452 to float, !dbg !14
  %1454 = select i1 %.not1, float %1442, float %1453, !dbg !14
  %1455 = select i1 %.not1, float %1453, float %1448, !dbg !14
  %1456 = select i1 %.not1, float %1449, float %1443, !dbg !14
  %1457 = bitcast float %1456 to i32, !dbg !14
  %1458 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1457), !dbg !14
  %1459 = bitcast i32 %1458 to float, !dbg !14
  %1460 = select i1 %.not1, float %1443, float %1459, !dbg !14
  %1461 = select i1 %.not1, float %1459, float %1449, !dbg !14
  %1462 = shufflevector <8 x float> %1422, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1463 = fmul <4 x float> %1462, %725, !dbg !35
  %1464 = insertelement <4 x float> poison, float %1454, i64 0, !dbg !36
  %1465 = insertelement <4 x float> %1464, float %1460, i64 1, !dbg !36
  %1466 = insertelement <4 x float> %1465, float %1455, i64 2, !dbg !36
  %1467 = insertelement <4 x float> %1466, float %1461, i64 3, !dbg !36
  %1468 = fadd <4 x float> %1463, %1467, !dbg !36
  %1469 = shufflevector <4 x float> %1429, <4 x float> %1468, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1470 = insertvalue [2 x <8 x float>] %1389, <8 x float> %1469, 1, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1471 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !14
  %1472 = bitcast <4 x i32> %1471 to <8 x half>, !dbg !14
  %1473 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !14
  %1474 = bitcast <4 x i32> %1473 to <8 x half>, !dbg !14
  %1475 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !14
  %1476 = bitcast <4 x i32> %1475 to <8 x half>, !dbg !14
  %1477 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !14
  %1478 = bitcast <4 x i32> %1477 to <8 x half>, !dbg !14
  %1479 = shufflevector <8 x half> %1472, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1479, <4 x float> zeroinitializer) #9, !dbg !14
  %1481 = shufflevector <8 x half> %1474, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1481, <4 x float> %1480) #9, !dbg !14
  %1483 = shufflevector <8 x half> %1476, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1483, <4 x float> %1482) #9, !dbg !14
  %1485 = shufflevector <8 x half> %1478, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1485, <4 x float> %1484) #9, !dbg !14
  %1487 = extractelement <4 x float> %1486, i64 0, !dbg !14
  %1488 = extractelement <4 x float> %1486, i64 1, !dbg !14
  %1489 = extractelement <4 x float> %1486, i64 2, !dbg !14
  %1490 = extractelement <4 x float> %1486, i64 3, !dbg !14
  %1491 = select i1 %.not, float %1488, float %1487, !dbg !14
  %1492 = bitcast float %1491 to i32, !dbg !14
  %1493 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1492), !dbg !14
  %1494 = bitcast i32 %1493 to float, !dbg !14
  %1495 = select i1 %.not, float %1487, float %1494, !dbg !14
  %1496 = select i1 %.not, float %1494, float %1488, !dbg !14
  %1497 = select i1 %.not, float %1490, float %1489, !dbg !14
  %1498 = bitcast float %1497 to i32, !dbg !14
  %1499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1498), !dbg !14
  %1500 = bitcast i32 %1499 to float, !dbg !14
  %1501 = select i1 %.not, float %1489, float %1500, !dbg !14
  %1502 = select i1 %.not, float %1500, float %1490, !dbg !14
  %1503 = select i1 %.not1, float %1501, float %1495, !dbg !14
  %1504 = bitcast float %1503 to i32, !dbg !14
  %1505 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1504), !dbg !14
  %1506 = bitcast i32 %1505 to float, !dbg !14
  %1507 = select i1 %.not1, float %1495, float %1506, !dbg !14
  %1508 = select i1 %.not1, float %1506, float %1501, !dbg !14
  %1509 = select i1 %.not1, float %1502, float %1496, !dbg !14
  %1510 = bitcast float %1509 to i32, !dbg !14
  %1511 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1510), !dbg !14
  %1512 = bitcast i32 %1511 to float, !dbg !14
  %1513 = select i1 %.not1, float %1496, float %1512, !dbg !14
  %1514 = select i1 %.not1, float %1512, float %1502, !dbg !14
  %1515 = extractvalue [2 x <8 x float>] %213, 0, !dbg !35
  %1516 = shufflevector <8 x float> %1515, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1517 = fmul <4 x float> %1516, %717, !dbg !35
  %1518 = insertelement <4 x float> poison, float %1507, i64 0, !dbg !36
  %1519 = insertelement <4 x float> %1518, float %1513, i64 1, !dbg !36
  %1520 = insertelement <4 x float> %1519, float %1508, i64 2, !dbg !36
  %1521 = insertelement <4 x float> %1520, float %1514, i64 3, !dbg !36
  %1522 = fadd <4 x float> %1517, %1521, !dbg !36
  %1523 = shufflevector <8 x half> %1472, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1523, <4 x float> zeroinitializer) #9, !dbg !14
  %1525 = shufflevector <8 x half> %1474, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1525, <4 x float> %1524) #9, !dbg !14
  %1527 = shufflevector <8 x half> %1476, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1527, <4 x float> %1526) #9, !dbg !14
  %1529 = shufflevector <8 x half> %1478, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1529, <4 x float> %1528) #9, !dbg !14
  %1531 = extractelement <4 x float> %1530, i64 0, !dbg !14
  %1532 = extractelement <4 x float> %1530, i64 1, !dbg !14
  %1533 = extractelement <4 x float> %1530, i64 2, !dbg !14
  %1534 = extractelement <4 x float> %1530, i64 3, !dbg !14
  %1535 = select i1 %.not, float %1532, float %1531, !dbg !14
  %1536 = bitcast float %1535 to i32, !dbg !14
  %1537 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1536), !dbg !14
  %1538 = bitcast i32 %1537 to float, !dbg !14
  %1539 = select i1 %.not, float %1531, float %1538, !dbg !14
  %1540 = select i1 %.not, float %1538, float %1532, !dbg !14
  %1541 = select i1 %.not, float %1534, float %1533, !dbg !14
  %1542 = bitcast float %1541 to i32, !dbg !14
  %1543 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1542), !dbg !14
  %1544 = bitcast i32 %1543 to float, !dbg !14
  %1545 = select i1 %.not, float %1533, float %1544, !dbg !14
  %1546 = select i1 %.not, float %1544, float %1534, !dbg !14
  %1547 = select i1 %.not1, float %1545, float %1539, !dbg !14
  %1548 = bitcast float %1547 to i32, !dbg !14
  %1549 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1548), !dbg !14
  %1550 = bitcast i32 %1549 to float, !dbg !14
  %1551 = select i1 %.not1, float %1539, float %1550, !dbg !14
  %1552 = select i1 %.not1, float %1550, float %1545, !dbg !14
  %1553 = select i1 %.not1, float %1546, float %1540, !dbg !14
  %1554 = bitcast float %1553 to i32, !dbg !14
  %1555 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1554), !dbg !14
  %1556 = bitcast i32 %1555 to float, !dbg !14
  %1557 = select i1 %.not1, float %1540, float %1556, !dbg !14
  %1558 = select i1 %.not1, float %1556, float %1546, !dbg !14
  %1559 = shufflevector <8 x float> %1515, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1560 = fmul <4 x float> %1559, %717, !dbg !35
  %1561 = insertelement <4 x float> poison, float %1551, i64 0, !dbg !36
  %1562 = insertelement <4 x float> %1561, float %1557, i64 1, !dbg !36
  %1563 = insertelement <4 x float> %1562, float %1552, i64 2, !dbg !36
  %1564 = insertelement <4 x float> %1563, float %1558, i64 3, !dbg !36
  %1565 = fadd <4 x float> %1560, %1564, !dbg !36
  %1566 = shufflevector <4 x float> %1522, <4 x float> %1565, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1567 = insertvalue [2 x <8 x float>] poison, <8 x float> %1566, 0, !dbg !15
  %1568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1479, <4 x float> zeroinitializer) #9, !dbg !14
  %1569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1481, <4 x float> %1568) #9, !dbg !14
  %1570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1483, <4 x float> %1569) #9, !dbg !14
  %1571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1485, <4 x float> %1570) #9, !dbg !14
  %1572 = extractelement <4 x float> %1571, i64 0, !dbg !14
  %1573 = extractelement <4 x float> %1571, i64 1, !dbg !14
  %1574 = extractelement <4 x float> %1571, i64 2, !dbg !14
  %1575 = extractelement <4 x float> %1571, i64 3, !dbg !14
  %1576 = select i1 %.not, float %1573, float %1572, !dbg !14
  %1577 = bitcast float %1576 to i32, !dbg !14
  %1578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1577), !dbg !14
  %1579 = bitcast i32 %1578 to float, !dbg !14
  %1580 = select i1 %.not, float %1572, float %1579, !dbg !14
  %1581 = select i1 %.not, float %1579, float %1573, !dbg !14
  %1582 = select i1 %.not, float %1575, float %1574, !dbg !14
  %1583 = bitcast float %1582 to i32, !dbg !14
  %1584 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1583), !dbg !14
  %1585 = bitcast i32 %1584 to float, !dbg !14
  %1586 = select i1 %.not, float %1574, float %1585, !dbg !14
  %1587 = select i1 %.not, float %1585, float %1575, !dbg !14
  %1588 = select i1 %.not1, float %1586, float %1580, !dbg !14
  %1589 = bitcast float %1588 to i32, !dbg !14
  %1590 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1589), !dbg !14
  %1591 = bitcast i32 %1590 to float, !dbg !14
  %1592 = select i1 %.not1, float %1580, float %1591, !dbg !14
  %1593 = select i1 %.not1, float %1591, float %1586, !dbg !14
  %1594 = select i1 %.not1, float %1587, float %1581, !dbg !14
  %1595 = bitcast float %1594 to i32, !dbg !14
  %1596 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1595), !dbg !14
  %1597 = bitcast i32 %1596 to float, !dbg !14
  %1598 = select i1 %.not1, float %1581, float %1597, !dbg !14
  %1599 = select i1 %.not1, float %1597, float %1587, !dbg !14
  %1600 = extractvalue [2 x <8 x float>] %213, 1, !dbg !35
  %1601 = shufflevector <8 x float> %1600, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1602 = fmul <4 x float> %1601, %725, !dbg !35
  %1603 = insertelement <4 x float> poison, float %1592, i64 0, !dbg !36
  %1604 = insertelement <4 x float> %1603, float %1598, i64 1, !dbg !36
  %1605 = insertelement <4 x float> %1604, float %1593, i64 2, !dbg !36
  %1606 = insertelement <4 x float> %1605, float %1599, i64 3, !dbg !36
  %1607 = fadd <4 x float> %1602, %1606, !dbg !36
  %1608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1523, <4 x float> zeroinitializer) #9, !dbg !14
  %1609 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1525, <4 x float> %1608) #9, !dbg !14
  %1610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1527, <4 x float> %1609) #9, !dbg !14
  %1611 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1529, <4 x float> %1610) #9, !dbg !14
  %1612 = extractelement <4 x float> %1611, i64 0, !dbg !14
  %1613 = extractelement <4 x float> %1611, i64 1, !dbg !14
  %1614 = extractelement <4 x float> %1611, i64 2, !dbg !14
  %1615 = extractelement <4 x float> %1611, i64 3, !dbg !14
  %1616 = select i1 %.not, float %1613, float %1612, !dbg !14
  %1617 = bitcast float %1616 to i32, !dbg !14
  %1618 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1617), !dbg !14
  %1619 = bitcast i32 %1618 to float, !dbg !14
  %1620 = select i1 %.not, float %1612, float %1619, !dbg !14
  %1621 = select i1 %.not, float %1619, float %1613, !dbg !14
  %1622 = select i1 %.not, float %1615, float %1614, !dbg !14
  %1623 = bitcast float %1622 to i32, !dbg !14
  %1624 = call i32 @llvm.amdgcn.ds.bpermute(i32 %662, i32 %1623), !dbg !14
  %1625 = bitcast i32 %1624 to float, !dbg !14
  %1626 = select i1 %.not, float %1614, float %1625, !dbg !14
  %1627 = select i1 %.not, float %1625, float %1615, !dbg !14
  %1628 = select i1 %.not1, float %1626, float %1620, !dbg !14
  %1629 = bitcast float %1628 to i32, !dbg !14
  %1630 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1629), !dbg !14
  %1631 = bitcast i32 %1630 to float, !dbg !14
  %1632 = select i1 %.not1, float %1620, float %1631, !dbg !14
  %1633 = select i1 %.not1, float %1631, float %1626, !dbg !14
  %1634 = select i1 %.not1, float %1627, float %1621, !dbg !14
  %1635 = bitcast float %1634 to i32, !dbg !14
  %1636 = call i32 @llvm.amdgcn.ds.bpermute(i32 %670, i32 %1635), !dbg !14
  %1637 = bitcast i32 %1636 to float, !dbg !14
  %1638 = select i1 %.not1, float %1621, float %1637, !dbg !14
  %1639 = select i1 %.not1, float %1637, float %1627, !dbg !14
  %1640 = shufflevector <8 x float> %1600, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1641 = fmul <4 x float> %1640, %725, !dbg !35
  %1642 = insertelement <4 x float> poison, float %1632, i64 0, !dbg !36
  %1643 = insertelement <4 x float> %1642, float %1638, i64 1, !dbg !36
  %1644 = insertelement <4 x float> %1643, float %1633, i64 2, !dbg !36
  %1645 = insertelement <4 x float> %1644, float %1639, i64 3, !dbg !36
  %1646 = fadd <4 x float> %1641, %1645, !dbg !36
  %1647 = shufflevector <4 x float> %1607, <4 x float> %1646, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1648 = insertvalue [2 x <8 x float>] %1567, <8 x float> %1647, 1, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1649 = add nuw nsw i64 %208, 1, !dbg !15
  br label %207, !dbg !15

1650:                                             ; preds = %207
  %1651 = extractvalue [2 x <1 x float>] %210, 0, !dbg !19
  %1652 = shufflevector <1 x float> %1651, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1653 = extractvalue [2 x <1 x float>] %210, 1, !dbg !19
  %1654 = shufflevector <1 x float> %1653, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  br i1 true, label %1655, label %1685, !dbg !21

1655:                                             ; preds = %1650
  %1656 = extractvalue [2 x <8 x float>] %209, 1, !dbg !19
  %1657 = shufflevector <8 x float> %1656, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1654, !dbg !19
  %normalized = fmul <4 x float> %1657, %row.reciprocal, !dbg !19
  %1658 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1659 = extractelement <4 x half> %1658, i64 3, !dbg !20
  %1660 = extractelement <4 x half> %1658, i64 2, !dbg !20
  %1661 = extractelement <4 x half> %1658, i64 1, !dbg !20
  %1662 = extractelement <4 x half> %1658, i64 0, !dbg !20
  %1663 = shufflevector <8 x float> %1656, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized54 = fmul <4 x float> %1663, %row.reciprocal, !dbg !19
  %1664 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !20
  %1665 = extractelement <4 x half> %1664, i64 3, !dbg !20
  %1666 = extractelement <4 x half> %1664, i64 2, !dbg !20
  %1667 = extractelement <4 x half> %1664, i64 1, !dbg !20
  %1668 = extractelement <4 x half> %1664, i64 0, !dbg !20
  %1669 = extractvalue [2 x <8 x float>] %209, 0, !dbg !19
  %1670 = shufflevector <8 x float> %1669, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal55 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1652, !dbg !19
  %normalized56 = fmul <4 x float> %1670, %row.reciprocal55, !dbg !19
  %1671 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !20
  %1672 = extractelement <4 x half> %1671, i64 3, !dbg !20
  %1673 = extractelement <4 x half> %1671, i64 2, !dbg !20
  %1674 = extractelement <4 x half> %1671, i64 1, !dbg !20
  %1675 = extractelement <4 x half> %1671, i64 0, !dbg !20
  %1676 = shufflevector <8 x float> %1669, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized57 = fmul <4 x float> %1676, %row.reciprocal55, !dbg !19
  %1677 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !20
  %1678 = extractelement <4 x half> %1677, i64 3, !dbg !20
  %1679 = extractelement <4 x half> %1677, i64 2, !dbg !20
  %1680 = extractelement <4 x half> %1677, i64 1, !dbg !20
  %1681 = extractelement <4 x half> %1677, i64 0, !dbg !20
  %1682 = add nsw i64 %16, %22, !dbg !21
  %1683 = shl i64 %1682, 19, !dbg !21
  %1684 = and i64 %1683, 16252928, !dbg !21
  %address.thread534 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local535 = zext nneg i32 %address.thread534 to i64, !dbg !8
  %address.local536 = add i64 %25, %address.local535, !dbg !21
  %address.local537 = lshr i64 %address.local535, 6, !dbg !7
  %address.local538 = shl nuw nsw i64 %address.local537, 4, !dbg !21
  %address.local539 = add i64 %address.local536, %address.local538, !dbg !21
  %address.local540 = and i64 %address.local535, 240, !dbg !21
  %address.local541 = sub i64 %address.local539, %address.local540, !dbg !21
  %address.local542 = add i64 %address.local541, %26, !dbg !21
  %address.local543 = shl nuw nsw i64 %address.local542, 7, !dbg !21
  %address.local544 = add nuw nsw i64 %1684, %address.local543, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local546 = lshr i64 %address.local535, 2, !dbg !7
  %address.local547 = and i64 %address.local546, 12, !dbg !7
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local547, !dbg !21
  store half %1681, ptr addrspace(1) %address.local548, align 2, !dbg !21
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local550 = or disjoint i64 %address.local547, 1, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local549, i64 %address.local550, !dbg !21
  store half %1680, ptr addrspace(1) %address.local551, align 2, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local553 = or disjoint i64 %address.local547, 2, !dbg !21
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local552, i64 %address.local553, !dbg !21
  store half %1679, ptr addrspace(1) %address.local554, align 2, !dbg !21
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local556 = or disjoint i64 %address.local547, 3, !dbg !21
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local555, i64 %address.local556, !dbg !21
  store half %1678, ptr addrspace(1) %address.local557, align 2, !dbg !21
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local559 = or disjoint i64 %address.local547, 16, !dbg !21
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local558, i64 %address.local559, !dbg !21
  store half %1675, ptr addrspace(1) %address.local560, align 2, !dbg !21
  %address.local561 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local562 = or disjoint i64 %address.local547, 17, !dbg !21
  %address.local563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local561, i64 %address.local562, !dbg !21
  store half %1674, ptr addrspace(1) %address.local563, align 2, !dbg !21
  %address.local564 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local565 = or disjoint i64 %address.local547, 18, !dbg !21
  %address.local566 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local564, i64 %address.local565, !dbg !21
  store half %1673, ptr addrspace(1) %address.local566, align 2, !dbg !21
  %address.local567 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local568 = or disjoint i64 %address.local547, 19, !dbg !21
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local567, i64 %address.local568, !dbg !21
  store half %1672, ptr addrspace(1) %address.local569, align 2, !dbg !21
  %address.local570 = shl i64 %address.local542, 7, !dbg !21
  %address.local571 = add i64 %address.local570, 8192, !dbg !21
  %address.local572 = add nuw nsw i64 %1684, %address.local571, !dbg !21
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local573, i64 %address.local547, !dbg !21
  store half %1668, ptr addrspace(1) %address.local574, align 2, !dbg !21
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local550, !dbg !21
  store half %1667, ptr addrspace(1) %address.local576, align 2, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local553, !dbg !21
  store half %1666, ptr addrspace(1) %address.local578, align 2, !dbg !21
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local579, i64 %address.local556, !dbg !21
  store half %1665, ptr addrspace(1) %address.local580, align 2, !dbg !21
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local559, !dbg !21
  store half %1662, ptr addrspace(1) %address.local582, align 2, !dbg !21
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local562, !dbg !21
  store half %1661, ptr addrspace(1) %address.local584, align 2, !dbg !21
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local585, i64 %address.local565, !dbg !21
  store half %1660, ptr addrspace(1) %address.local586, align 2, !dbg !21
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local587, i64 %address.local568, !dbg !21
  store half %1659, ptr addrspace(1) %address.local588, align 2, !dbg !21
  br label %1685, !dbg !21

1685:                                             ; preds = %1655, %1650
  br i1 true, label %1686, label %1714, !dbg !21

1686:                                             ; preds = %1685
  %1687 = extractvalue [2 x <8 x float>] %211, 1, !dbg !19
  %1688 = shufflevector <8 x float> %1687, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal58 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1654, !dbg !19
  %normalized59 = fmul <4 x float> %1688, %row.reciprocal58, !dbg !19
  %1689 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !20
  %1690 = extractelement <4 x half> %1689, i64 3, !dbg !20
  %1691 = extractelement <4 x half> %1689, i64 2, !dbg !20
  %1692 = extractelement <4 x half> %1689, i64 1, !dbg !20
  %1693 = extractelement <4 x half> %1689, i64 0, !dbg !20
  %1694 = shufflevector <8 x float> %1687, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized60 = fmul <4 x float> %1694, %row.reciprocal58, !dbg !19
  %1695 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !20
  %1696 = extractelement <4 x half> %1695, i64 3, !dbg !20
  %1697 = extractelement <4 x half> %1695, i64 2, !dbg !20
  %1698 = extractelement <4 x half> %1695, i64 1, !dbg !20
  %1699 = extractelement <4 x half> %1695, i64 0, !dbg !20
  %1700 = extractvalue [2 x <8 x float>] %211, 0, !dbg !19
  %1701 = shufflevector <8 x float> %1700, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal61 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1652, !dbg !19
  %normalized62 = fmul <4 x float> %1701, %row.reciprocal61, !dbg !19
  %1702 = fptrunc <4 x float> %normalized62 to <4 x half>, !dbg !20
  %1703 = extractelement <4 x half> %1702, i64 3, !dbg !20
  %1704 = extractelement <4 x half> %1702, i64 2, !dbg !20
  %1705 = extractelement <4 x half> %1702, i64 1, !dbg !20
  %1706 = extractelement <4 x half> %1702, i64 0, !dbg !20
  %1707 = shufflevector <8 x float> %1700, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized63 = fmul <4 x float> %1707, %row.reciprocal61, !dbg !19
  %1708 = fptrunc <4 x float> %normalized63 to <4 x half>, !dbg !20
  %1709 = extractelement <4 x half> %1708, i64 3, !dbg !20
  %1710 = extractelement <4 x half> %1708, i64 2, !dbg !20
  %1711 = extractelement <4 x half> %1708, i64 1, !dbg !20
  %1712 = extractelement <4 x half> %1708, i64 0, !dbg !20
  %1713 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread589 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local590 = zext nneg i32 %address.thread589 to i64, !dbg !8
  %address.local591 = add i64 %25, %address.local590, !dbg !21
  %address.local592 = lshr i64 %address.local590, 6, !dbg !7
  %address.local593 = shl nuw nsw i64 %address.local592, 4, !dbg !21
  %address.local594 = add i64 %address.local591, %address.local593, !dbg !21
  %address.local595 = and i64 %address.local590, 240, !dbg !21
  %address.local596 = sub i64 %address.local594, %address.local595, !dbg !21
  %address.local597 = add i64 %address.local596, %26, !dbg !21
  %address.local598 = shl nuw nsw i64 %address.local597, 7, !dbg !21
  %address.local599 = add nuw nsw i64 %1713, %address.local598, !dbg !21
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local601 = lshr i64 %address.local590, 2, !dbg !7
  %address.local602 = and i64 %address.local601, 12, !dbg !7
  %address.local603 = or disjoint i64 %address.local602, 32, !dbg !21
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local603, !dbg !21
  store half %1712, ptr addrspace(1) %address.local604, align 2, !dbg !21
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local606 = or disjoint i64 %address.local602, 33, !dbg !21
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local605, i64 %address.local606, !dbg !21
  store half %1711, ptr addrspace(1) %address.local607, align 2, !dbg !21
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local609 = or disjoint i64 %address.local602, 34, !dbg !21
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local608, i64 %address.local609, !dbg !21
  store half %1710, ptr addrspace(1) %address.local610, align 2, !dbg !21
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local612 = or disjoint i64 %address.local602, 35, !dbg !21
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local611, i64 %address.local612, !dbg !21
  store half %1709, ptr addrspace(1) %address.local613, align 2, !dbg !21
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local615 = or disjoint i64 %address.local602, 48, !dbg !21
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local614, i64 %address.local615, !dbg !21
  store half %1706, ptr addrspace(1) %address.local616, align 2, !dbg !21
  %address.local617 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local618 = or disjoint i64 %address.local602, 49, !dbg !21
  %address.local619 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local617, i64 %address.local618, !dbg !21
  store half %1705, ptr addrspace(1) %address.local619, align 2, !dbg !21
  %address.local620 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local621 = or disjoint i64 %address.local602, 50, !dbg !21
  %address.local622 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local621, !dbg !21
  store half %1704, ptr addrspace(1) %address.local622, align 2, !dbg !21
  %address.local623 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local624 = or i64 %address.local601, 51, !dbg !21
  %address.local625 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local623, i64 %address.local624, !dbg !21
  store half %1703, ptr addrspace(1) %address.local625, align 2, !dbg !21
  %address.local626 = shl i64 %address.local597, 7, !dbg !21
  %address.local627 = add i64 %address.local626, 8192, !dbg !21
  %address.local628 = add nuw nsw i64 %1713, %address.local627, !dbg !21
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local629, i64 %address.local603, !dbg !21
  store half %1699, ptr addrspace(1) %address.local630, align 2, !dbg !21
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local606, !dbg !21
  store half %1698, ptr addrspace(1) %address.local632, align 2, !dbg !21
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local633, i64 %address.local609, !dbg !21
  store half %1697, ptr addrspace(1) %address.local634, align 2, !dbg !21
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local635, i64 %address.local612, !dbg !21
  store half %1696, ptr addrspace(1) %address.local636, align 2, !dbg !21
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local637, i64 %address.local615, !dbg !21
  store half %1693, ptr addrspace(1) %address.local638, align 2, !dbg !21
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local639, i64 %address.local618, !dbg !21
  store half %1692, ptr addrspace(1) %address.local640, align 2, !dbg !21
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local642 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local641, i64 %address.local621, !dbg !21
  store half %1691, ptr addrspace(1) %address.local642, align 2, !dbg !21
  %address.local643 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local624, !dbg !21
  store half %1690, ptr addrspace(1) %address.local644, align 2, !dbg !21
  br label %1714, !dbg !21

1714:                                             ; preds = %1686, %1685
  br i1 true, label %1715, label %1743, !dbg !21

1715:                                             ; preds = %1714
  %1716 = extractvalue [2 x <8 x float>] %212, 1, !dbg !19
  %1717 = shufflevector <8 x float> %1716, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal64 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1654, !dbg !19
  %normalized65 = fmul <4 x float> %1717, %row.reciprocal64, !dbg !19
  %1718 = fptrunc <4 x float> %normalized65 to <4 x half>, !dbg !20
  %1719 = extractelement <4 x half> %1718, i64 3, !dbg !20
  %1720 = extractelement <4 x half> %1718, i64 2, !dbg !20
  %1721 = extractelement <4 x half> %1718, i64 1, !dbg !20
  %1722 = extractelement <4 x half> %1718, i64 0, !dbg !20
  %1723 = shufflevector <8 x float> %1716, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized66 = fmul <4 x float> %1723, %row.reciprocal64, !dbg !19
  %1724 = fptrunc <4 x float> %normalized66 to <4 x half>, !dbg !20
  %1725 = extractelement <4 x half> %1724, i64 3, !dbg !20
  %1726 = extractelement <4 x half> %1724, i64 2, !dbg !20
  %1727 = extractelement <4 x half> %1724, i64 1, !dbg !20
  %1728 = extractelement <4 x half> %1724, i64 0, !dbg !20
  %1729 = extractvalue [2 x <8 x float>] %212, 0, !dbg !19
  %1730 = shufflevector <8 x float> %1729, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal67 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1652, !dbg !19
  %normalized68 = fmul <4 x float> %1730, %row.reciprocal67, !dbg !19
  %1731 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !20
  %1732 = extractelement <4 x half> %1731, i64 3, !dbg !20
  %1733 = extractelement <4 x half> %1731, i64 2, !dbg !20
  %1734 = extractelement <4 x half> %1731, i64 1, !dbg !20
  %1735 = extractelement <4 x half> %1731, i64 0, !dbg !20
  %1736 = shufflevector <8 x float> %1729, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized69 = fmul <4 x float> %1736, %row.reciprocal67, !dbg !19
  %1737 = fptrunc <4 x float> %normalized69 to <4 x half>, !dbg !20
  %1738 = extractelement <4 x half> %1737, i64 3, !dbg !20
  %1739 = extractelement <4 x half> %1737, i64 2, !dbg !20
  %1740 = extractelement <4 x half> %1737, i64 1, !dbg !20
  %1741 = extractelement <4 x half> %1737, i64 0, !dbg !20
  %1742 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread645 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local646 = zext nneg i32 %address.thread645 to i64, !dbg !8
  %address.local647 = add i64 %25, %address.local646, !dbg !21
  %address.local648 = lshr i64 %address.local646, 6, !dbg !7
  %address.local649 = shl nuw nsw i64 %address.local648, 4, !dbg !21
  %address.local650 = add i64 %address.local647, %address.local649, !dbg !21
  %address.local651 = and i64 %address.local646, 240, !dbg !21
  %address.local652 = sub i64 %address.local650, %address.local651, !dbg !21
  %address.local653 = add i64 %address.local652, %26, !dbg !21
  %address.local654 = shl nuw nsw i64 %address.local653, 7, !dbg !21
  %address.local655 = add nuw nsw i64 %1742, %address.local654, !dbg !21
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local657 = lshr i64 %address.local646, 2, !dbg !7
  %address.local658 = and i64 %address.local657, 12, !dbg !7
  %address.local659 = or disjoint i64 %address.local658, 64, !dbg !21
  %address.local660 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local656, i64 %address.local659, !dbg !21
  store half %1741, ptr addrspace(1) %address.local660, align 2, !dbg !21
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local662 = or disjoint i64 %address.local658, 65, !dbg !21
  %address.local663 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local661, i64 %address.local662, !dbg !21
  store half %1740, ptr addrspace(1) %address.local663, align 2, !dbg !21
  %address.local664 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local665 = or disjoint i64 %address.local658, 66, !dbg !21
  %address.local666 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local664, i64 %address.local665, !dbg !21
  store half %1739, ptr addrspace(1) %address.local666, align 2, !dbg !21
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local668 = or disjoint i64 %address.local658, 67, !dbg !21
  %address.local669 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local667, i64 %address.local668, !dbg !21
  store half %1738, ptr addrspace(1) %address.local669, align 2, !dbg !21
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local671 = or disjoint i64 %address.local658, 80, !dbg !21
  %address.local672 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local670, i64 %address.local671, !dbg !21
  store half %1735, ptr addrspace(1) %address.local672, align 2, !dbg !21
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local674 = or disjoint i64 %address.local658, 81, !dbg !21
  %address.local675 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local673, i64 %address.local674, !dbg !21
  store half %1734, ptr addrspace(1) %address.local675, align 2, !dbg !21
  %address.local676 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local677 = or disjoint i64 %address.local658, 82, !dbg !21
  %address.local678 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local676, i64 %address.local677, !dbg !21
  store half %1733, ptr addrspace(1) %address.local678, align 2, !dbg !21
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local680 = or disjoint i64 %address.local658, 83, !dbg !21
  %address.local681 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local679, i64 %address.local680, !dbg !21
  store half %1732, ptr addrspace(1) %address.local681, align 2, !dbg !21
  %address.local682 = shl i64 %address.local653, 7, !dbg !21
  %address.local683 = add i64 %address.local682, 8192, !dbg !21
  %address.local684 = add nuw nsw i64 %1742, %address.local683, !dbg !21
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local685, i64 %address.local659, !dbg !21
  store half %1728, ptr addrspace(1) %address.local686, align 2, !dbg !21
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local687, i64 %address.local662, !dbg !21
  store half %1727, ptr addrspace(1) %address.local688, align 2, !dbg !21
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local689, i64 %address.local665, !dbg !21
  store half %1726, ptr addrspace(1) %address.local690, align 2, !dbg !21
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local691, i64 %address.local668, !dbg !21
  store half %1725, ptr addrspace(1) %address.local692, align 2, !dbg !21
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local693, i64 %address.local671, !dbg !21
  store half %1722, ptr addrspace(1) %address.local694, align 2, !dbg !21
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local695, i64 %address.local674, !dbg !21
  store half %1721, ptr addrspace(1) %address.local696, align 2, !dbg !21
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local697, i64 %address.local677, !dbg !21
  store half %1720, ptr addrspace(1) %address.local698, align 2, !dbg !21
  %address.local699 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local699, i64 %address.local680, !dbg !21
  store half %1719, ptr addrspace(1) %address.local700, align 2, !dbg !21
  br label %1743, !dbg !21

1743:                                             ; preds = %1715, %1714
  br i1 true, label %1744, label %1772, !dbg !21

1744:                                             ; preds = %1743
  %1745 = extractvalue [2 x <8 x float>] %213, 1, !dbg !19
  %1746 = shufflevector <8 x float> %1745, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal70 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1654, !dbg !19
  %normalized71 = fmul <4 x float> %1746, %row.reciprocal70, !dbg !19
  %1747 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !20
  %1748 = extractelement <4 x half> %1747, i64 3, !dbg !20
  %1749 = extractelement <4 x half> %1747, i64 2, !dbg !20
  %1750 = extractelement <4 x half> %1747, i64 1, !dbg !20
  %1751 = extractelement <4 x half> %1747, i64 0, !dbg !20
  %1752 = shufflevector <8 x float> %1745, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized72 = fmul <4 x float> %1752, %row.reciprocal70, !dbg !19
  %1753 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !20
  %1754 = extractelement <4 x half> %1753, i64 3, !dbg !20
  %1755 = extractelement <4 x half> %1753, i64 2, !dbg !20
  %1756 = extractelement <4 x half> %1753, i64 1, !dbg !20
  %1757 = extractelement <4 x half> %1753, i64 0, !dbg !20
  %1758 = extractvalue [2 x <8 x float>] %213, 0, !dbg !19
  %1759 = shufflevector <8 x float> %1758, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1652, !dbg !19
  %normalized74 = fmul <4 x float> %1759, %row.reciprocal73, !dbg !19
  %1760 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %1761 = extractelement <4 x half> %1760, i64 3, !dbg !20
  %1762 = extractelement <4 x half> %1760, i64 2, !dbg !20
  %1763 = extractelement <4 x half> %1760, i64 1, !dbg !20
  %1764 = extractelement <4 x half> %1760, i64 0, !dbg !20
  %1765 = shufflevector <8 x float> %1758, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized75 = fmul <4 x float> %1765, %row.reciprocal73, !dbg !19
  %1766 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !20
  %1767 = extractelement <4 x half> %1766, i64 3, !dbg !20
  %1768 = extractelement <4 x half> %1766, i64 2, !dbg !20
  %1769 = extractelement <4 x half> %1766, i64 1, !dbg !20
  %1770 = extractelement <4 x half> %1766, i64 0, !dbg !20
  %1771 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread701 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local702 = zext nneg i32 %address.thread701 to i64, !dbg !8
  %address.local703 = add i64 %25, %address.local702, !dbg !21
  %address.local704 = lshr i64 %address.local702, 6, !dbg !7
  %address.local705 = shl nuw nsw i64 %address.local704, 4, !dbg !21
  %address.local706 = add i64 %address.local703, %address.local705, !dbg !21
  %address.local707 = and i64 %address.local702, 240, !dbg !21
  %address.local708 = sub i64 %address.local706, %address.local707, !dbg !21
  %address.local709 = add i64 %address.local708, %26, !dbg !21
  %address.local710 = shl nuw nsw i64 %address.local709, 7, !dbg !21
  %address.local711 = add nuw nsw i64 %1771, %address.local710, !dbg !21
  %address.local712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local713 = lshr i64 %address.local702, 2, !dbg !7
  %address.local714 = and i64 %address.local713, 12, !dbg !7
  %address.local715 = or disjoint i64 %address.local714, 96, !dbg !21
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local712, i64 %address.local715, !dbg !21
  store half %1770, ptr addrspace(1) %address.local716, align 2, !dbg !21
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local718 = or disjoint i64 %address.local714, 97, !dbg !21
  %address.local719 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local717, i64 %address.local718, !dbg !21
  store half %1769, ptr addrspace(1) %address.local719, align 2, !dbg !21
  %address.local720 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local721 = or disjoint i64 %address.local714, 98, !dbg !21
  %address.local722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local720, i64 %address.local721, !dbg !21
  store half %1768, ptr addrspace(1) %address.local722, align 2, !dbg !21
  %address.local723 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local724 = or disjoint i64 %address.local714, 99, !dbg !21
  %address.local725 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local723, i64 %address.local724, !dbg !21
  store half %1767, ptr addrspace(1) %address.local725, align 2, !dbg !21
  %address.local726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local727 = or disjoint i64 %address.local714, 112, !dbg !21
  %address.local728 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local726, i64 %address.local727, !dbg !21
  store half %1764, ptr addrspace(1) %address.local728, align 2, !dbg !21
  %address.local729 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local730 = or disjoint i64 %address.local714, 113, !dbg !21
  %address.local731 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local729, i64 %address.local730, !dbg !21
  store half %1763, ptr addrspace(1) %address.local731, align 2, !dbg !21
  %address.local732 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local733 = or disjoint i64 %address.local714, 114, !dbg !21
  %address.local734 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local732, i64 %address.local733, !dbg !21
  store half %1762, ptr addrspace(1) %address.local734, align 2, !dbg !21
  %address.local735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local736 = or i64 %address.local713, 115, !dbg !21
  %address.local737 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local735, i64 %address.local736, !dbg !21
  store half %1761, ptr addrspace(1) %address.local737, align 2, !dbg !21
  %address.local738 = shl i64 %address.local709, 7, !dbg !21
  %address.local739 = add i64 %address.local738, 8192, !dbg !21
  %address.local740 = add nuw nsw i64 %1771, %address.local739, !dbg !21
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local741, i64 %address.local715, !dbg !21
  store half %1757, ptr addrspace(1) %address.local742, align 2, !dbg !21
  %address.local743 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local743, i64 %address.local718, !dbg !21
  store half %1756, ptr addrspace(1) %address.local744, align 2, !dbg !21
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local746 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local745, i64 %address.local721, !dbg !21
  store half %1755, ptr addrspace(1) %address.local746, align 2, !dbg !21
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local747, i64 %address.local724, !dbg !21
  store half %1754, ptr addrspace(1) %address.local748, align 2, !dbg !21
  %address.local749 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local749, i64 %address.local727, !dbg !21
  store half %1751, ptr addrspace(1) %address.local750, align 2, !dbg !21
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local751, i64 %address.local730, !dbg !21
  store half %1750, ptr addrspace(1) %address.local752, align 2, !dbg !21
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local754 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local753, i64 %address.local733, !dbg !21
  store half %1749, ptr addrspace(1) %address.local754, align 2, !dbg !21
  %address.local755 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local755, i64 %address.local736, !dbg !21
  store half %1748, ptr addrspace(1) %address.local756, align 2, !dbg !21
  br label %1772, !dbg !21

1772:                                             ; preds = %1744, %1743
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

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #8 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax/export/run-20261023-171017-a0mz8rfe/input-0002-test_input_1x32x4096x128-128-64-32-online/retained32-baseline")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 72, column: 24, scope: !4)
!8 = !DILocation(line: 70, column: 22, scope: !4)
!10 = !DILocation(line: 48, column: 17, scope: !4)
!13 = !DILocation(line: 77, column: 19, scope: !4)
!14 = !DILocation(line: 102, column: 23, scope: !4)
!15 = !DILocation(line: 60, column: 24, scope: !4)
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
