; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/online_softmax/final/run-20261023-171702-30cjis9r/retained32-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax/final/run-20261023-171702-30cjis9r/retained32-baseline/input.mlir"

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
  %191 = and i32 %31, 16, !dbg !15
  %.not = icmp eq i32 %191, 0, !dbg !15
  %192 = and i32 %31, 32, !dbg !15
  %.not1 = icmp eq i32 %192, 0, !dbg !15
  %193 = shl nsw i64 %15, 1, !dbg !7
  %194 = add nsw i64 %193, 2, !dbg !7
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
  br label %207, !dbg !7

207:                                              ; preds = %589, %4
  %208 = phi i64 [ %1654, %589 ], [ 0, %4 ], !dbg !7
  %209 = phi [2 x <8 x float>] [ %1085, %589 ], [ zeroinitializer, %4 ], !dbg !7
  %210 = phi [2 x <1 x float>] [ %887, %589 ], [ zeroinitializer, %4 ], !dbg !7
  %211 = phi [2 x <1 x float>] [ %1653, %589 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %212 = phi [2 x <8 x float>] [ %1279, %589 ], [ zeroinitializer, %4 ], !dbg !7
  %213 = phi [2 x <8 x float>] [ %1473, %589 ], [ zeroinitializer, %4 ], !dbg !7
  %214 = phi [2 x <8 x float>] [ %1651, %589 ], [ zeroinitializer, %4 ], !dbg !7
  %215 = icmp slt i64 %208, %194, !dbg !7
  br i1 %215, label %216, label %1655, !dbg !7

216:                                              ; preds = %207
  %217 = shl nsw i64 %208, 6, !dbg !7
  %218 = lshr i64 %208, 13, !dbg !16
  %219 = add nuw nsw i64 %218, %16, !dbg !16
  %220 = lshr i64 %208, 6, !dbg !16
  %221 = and i64 %220, 127, !dbg !16
  %222 = shl i64 %219, 19, !dbg !16
  %223 = and i64 %222, 16252928, !dbg !16
  %224 = shl i64 %220, 19, !dbg !16
  %225 = shl i64 %208, 13, !dbg !16
  %226 = sub i64 %225, %224, !dbg !16
  %227 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %223, !dbg !16
  %228 = getelementptr [2 x i8], ptr addrspace(1) %227, i64 %221, !dbg !16
  %229 = getelementptr [2 x i8], ptr addrspace(1) %228, i64 %226, !dbg !16
  %address.thread270 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local271 = zext nneg i32 %address.thread270 to i64, !dbg !9
  %address.local272 = shl nuw nsw i64 %address.local271, 3, !dbg !9
  %address.local273 = and i64 %address.local272, 1536, !dbg !16
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local273, !dbg !16
  %address.local275 = ptrtoint ptr addrspace(3) %address.local274 to i64, !dbg !16
  %address.local276 = trunc i64 %address.local275 to i32, !dbg !16
  %230 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local276), !dbg !16
  %231 = zext i32 %230 to i64, !dbg !16
  %232 = inttoptr i64 %231 to ptr addrspace(3), !dbg !16
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
  %address.local287 = getelementptr [2 x i8], ptr addrspace(1) %229, i64 %address.local286, !dbg !16
  %233 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %234 = ptrtoint ptr addrspace(1) %address.local287 to i64, !dbg !16
  %235 = sub i64 %234, %233, !dbg !16
  %dma.buffer.offset972 = trunc i64 %235 to i32, !dbg !16
  %236 = trunc i64 %233 to i32, !dbg !16
  %237 = insertelement <4 x i32> poison, i32 %236, i64 0, !dbg !16
  %238 = lshr i64 %233, 32, !dbg !16
  %239 = trunc i64 %238 to i32, !dbg !16
  %240 = insertelement <4 x i32> %237, i32 %239, i64 1, !dbg !16
  %241 = insertelement <4 x i32> %240, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc973 = insertelement <4 x i32> %241, i32 131072, i64 3, !dbg !16
  %242 = ptrtoint ptr addrspace(3) %232 to i32, !dbg !16
  %dma.buffer.lds974 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %242), !dbg !16
  %243 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds974, i32 %dma.buffer.offset972, <4 x i32> %dma.buffer.rsrc973) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %244 = add nuw i64 %208, 2048, !dbg !16
  %245 = icmp slt i64 %244, 0, !dbg !16
  %246 = sub nuw i64 -2049, %208, !dbg !16
  %247 = select i1 %245, i64 %246, i64 %244, !dbg !16
  %248 = sdiv i64 %247, 8192, !dbg !16
  %.lobit19 = ashr i64 %244, 63, !dbg !16
  %249 = xor i64 %248, %.lobit19, !dbg !16
  %250 = add nsw i64 %249, %16, !dbg !16
  %251 = add nsw i64 %220, %198, !dbg !16
  %252 = add nsw i64 %251, 32, !dbg !16
  %253 = icmp slt i64 %251, -32, !dbg !16
  %254 = sub nsw i64 -33, %251, !dbg !16
  %255 = select i1 %253, i64 %254, i64 %252, !dbg !16
  %256 = sdiv i64 %255, 128, !dbg !16
  %257 = sext i1 %253 to i64, !dbg !16
  %258 = xor i64 %256, %257, !dbg !16
  %259 = shl nsw i64 %258, 7, !dbg !16
  %260 = sub i64 %198, %259, !dbg !16
  %261 = shl i64 %250, 19, !dbg !16
  %262 = and i64 %261, 16252928, !dbg !16
  %263 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %262, !dbg !16
  %264 = getelementptr [2 x i8], ptr addrspace(1) %263, i64 %220, !dbg !16
  %265 = getelementptr [2 x i8], ptr addrspace(1) %264, i64 %260, !dbg !16
  %266 = getelementptr i8, ptr addrspace(1) %265, i64 64, !dbg !16
  %267 = getelementptr [2 x i8], ptr addrspace(1) %266, i64 %226, !dbg !16
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local273, !dbg !16
  %address.local289 = ptrtoint ptr addrspace(3) %address.local288 to i64, !dbg !16
  %address.local290 = trunc i64 %address.local289 to i32, !dbg !16
  %268 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local290), !dbg !16
  %269 = zext i32 %268 to i64, !dbg !16
  %270 = inttoptr i64 %269 to ptr addrspace(3), !dbg !16
  %address.local291 = getelementptr [2 x i8], ptr addrspace(1) %267, i64 %address.local286, !dbg !16
  %271 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %272 = ptrtoint ptr addrspace(1) %address.local291 to i64, !dbg !16
  %273 = sub i64 %272, %271, !dbg !16
  %dma.buffer.offset969 = trunc i64 %273 to i32, !dbg !16
  %274 = trunc i64 %271 to i32, !dbg !16
  %275 = insertelement <4 x i32> poison, i32 %274, i64 0, !dbg !16
  %276 = lshr i64 %271, 32, !dbg !16
  %277 = trunc i64 %276 to i32, !dbg !16
  %278 = insertelement <4 x i32> %275, i32 %277, i64 1, !dbg !16
  %279 = insertelement <4 x i32> %278, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc970 = insertelement <4 x i32> %279, i32 131072, i64 3, !dbg !16
  %280 = ptrtoint ptr addrspace(3) %270 to i32, !dbg !16
  %dma.buffer.lds971 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %280), !dbg !16
  %281 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds971, i32 %dma.buffer.offset969, <4 x i32> %dma.buffer.rsrc970) #10, !dbg !16
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
  %282 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !17
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %282, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep898 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !17
  %284 = load <4 x half>, ptr addrspace(3) %lds.const.gep898, align 8, !dbg !17
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
  %285 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !17
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %284, <4 x float> %283) #9, !dbg !17
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %285, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep899 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !17
  %288 = load <4 x half>, ptr addrspace(3) %lds.const.gep899, align 8, !dbg !17
  %lds.const.gep900 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !17
  %289 = load <4 x half>, ptr addrspace(3) %lds.const.gep900, align 8, !dbg !17
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %288, <4 x float> %287) #9, !dbg !17
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %289, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep901 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !17
  %292 = load <4 x half>, ptr addrspace(3) %lds.const.gep901, align 8, !dbg !17
  %lds.const.gep902 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !17
  %293 = load <4 x half>, ptr addrspace(3) %lds.const.gep902, align 8, !dbg !17
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %292, <4 x float> %291) #9, !dbg !17
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %293, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep903 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !17
  %296 = load <4 x half>, ptr addrspace(3) %lds.const.gep903, align 8, !dbg !17
  %297 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !17
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %296, <4 x float> %295) #9, !dbg !17
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %297, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep904 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !17
  %300 = load <4 x half>, ptr addrspace(3) %lds.const.gep904, align 8, !dbg !17
  %301 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !17
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %300, <4 x float> %299) #9, !dbg !17
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %301, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep905 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !17
  %304 = load <4 x half>, ptr addrspace(3) %lds.const.gep905, align 8, !dbg !17
  %lds.const.gep906 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !17
  %305 = load <4 x half>, ptr addrspace(3) %lds.const.gep906, align 8, !dbg !17
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %304, <4 x float> %303) #9, !dbg !17
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %305, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep907 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !17
  %308 = load <4 x half>, ptr addrspace(3) %lds.const.gep907, align 8, !dbg !17
  %lds.const.gep908 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !17
  %309 = load <4 x half>, ptr addrspace(3) %lds.const.gep908, align 8, !dbg !17
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %308, <4 x float> %307) #9, !dbg !17
  %lds.const.gep909 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !17
  %311 = load <4 x half>, ptr addrspace(3) %lds.const.gep909, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %309, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %311, <4 x float> %312) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %314 = add nuw i64 %208, 4096, !dbg !16
  %315 = icmp slt i64 %314, 0, !dbg !16
  %316 = sub nuw i64 -4097, %208, !dbg !16
  %317 = select i1 %315, i64 %316, i64 %314, !dbg !16
  %318 = sdiv i64 %317, 8192, !dbg !16
  %.lobit20 = ashr i64 %314, 63, !dbg !16
  %319 = xor i64 %318, %.lobit20, !dbg !16
  %320 = add nsw i64 %319, %16, !dbg !16
  %321 = add nsw i64 %251, 64, !dbg !16
  %322 = icmp slt i64 %251, -64, !dbg !16
  %323 = sub nsw i64 -65, %251, !dbg !16
  %324 = select i1 %322, i64 %323, i64 %321, !dbg !16
  %325 = sdiv i64 %324, 128, !dbg !16
  %326 = sext i1 %322 to i64, !dbg !16
  %327 = xor i64 %325, %326, !dbg !16
  %328 = shl nsw i64 %327, 7, !dbg !16
  %329 = sub i64 %198, %328, !dbg !16
  %330 = shl i64 %320, 19, !dbg !16
  %331 = and i64 %330, 16252928, !dbg !16
  %332 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %331, !dbg !16
  %333 = getelementptr [2 x i8], ptr addrspace(1) %332, i64 %220, !dbg !16
  %334 = getelementptr [2 x i8], ptr addrspace(1) %333, i64 %329, !dbg !16
  %335 = getelementptr i8, ptr addrspace(1) %334, i64 128, !dbg !16
  %336 = getelementptr [2 x i8], ptr addrspace(1) %335, i64 %226, !dbg !16
  %337 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local276), !dbg !16
  %338 = zext i32 %337 to i64, !dbg !16
  %339 = inttoptr i64 %338 to ptr addrspace(3), !dbg !16
  %address.local480 = getelementptr [2 x i8], ptr addrspace(1) %336, i64 %address.local286, !dbg !16
  %340 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %341 = ptrtoint ptr addrspace(1) %address.local480 to i64, !dbg !16
  %342 = sub i64 %341, %340, !dbg !16
  %dma.buffer.offset966 = trunc i64 %342 to i32, !dbg !16
  %343 = trunc i64 %340 to i32, !dbg !16
  %344 = insertelement <4 x i32> poison, i32 %343, i64 0, !dbg !16
  %345 = lshr i64 %340, 32, !dbg !16
  %346 = trunc i64 %345 to i32, !dbg !16
  %347 = insertelement <4 x i32> %344, i32 %346, i64 1, !dbg !16
  %348 = insertelement <4 x i32> %347, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc967 = insertelement <4 x i32> %348, i32 131072, i64 3, !dbg !16
  %349 = ptrtoint ptr addrspace(3) %339 to i32, !dbg !16
  %dma.buffer.lds968 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %349), !dbg !16
  %350 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds968, i32 %dma.buffer.offset966, <4 x i32> %dma.buffer.rsrc967) #10, !dbg !16
  %lds.const.gep910 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !17
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep910, align 8, !dbg !17
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %351, <4 x float> %286) #9, !dbg !17
  %lds.const.gep911 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !17
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep911, align 8, !dbg !17
  %lds.const.gep912 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !17
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep912, align 8, !dbg !17
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %353, <4 x float> %352) #9, !dbg !17
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %354, <4 x float> %290) #9, !dbg !17
  %lds.const.gep913 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !17
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep913, align 8, !dbg !17
  %lds.const.gep914 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !17
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep914, align 8, !dbg !17
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %357, <4 x float> %356) #9, !dbg !17
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %358, <4 x float> %294) #9, !dbg !17
  %lds.const.gep915 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !17
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep915, align 8, !dbg !17
  %lds.const.gep916 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !17
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep916, align 8, !dbg !17
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %361, <4 x float> %360) #9, !dbg !17
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %362, <4 x float> %298) #9, !dbg !17
  %lds.const.gep917 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !17
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep917, align 8, !dbg !17
  %lds.const.gep918 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !17
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep918, align 8, !dbg !17
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %365, <4 x float> %364) #9, !dbg !17
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %366, <4 x float> %302) #9, !dbg !17
  %lds.const.gep919 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !17
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep919, align 8, !dbg !17
  %lds.const.gep920 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !17
  %370 = load <4 x half>, ptr addrspace(3) %lds.const.gep920, align 8, !dbg !17
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %369, <4 x float> %368) #9, !dbg !17
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %370, <4 x float> %306) #9, !dbg !17
  %lds.const.gep921 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !17
  %373 = load <4 x half>, ptr addrspace(3) %lds.const.gep921, align 8, !dbg !17
  %lds.const.gep922 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !17
  %374 = load <4 x half>, ptr addrspace(3) %lds.const.gep922, align 8, !dbg !17
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %373, <4 x float> %372) #9, !dbg !17
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %374, <4 x float> %310) #9, !dbg !17
  %lds.const.gep923 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !17
  %377 = load <4 x half>, ptr addrspace(3) %lds.const.gep923, align 8, !dbg !17
  %lds.const.gep924 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !17
  %378 = load <4 x half>, ptr addrspace(3) %lds.const.gep924, align 8, !dbg !17
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %377, <4 x float> %376) #9, !dbg !17
  %lds.const.gep925 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !17
  %380 = load <4 x half>, ptr addrspace(3) %lds.const.gep925, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %378, <4 x float> %313) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %380, <4 x float> %381) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %383 = add nuw i64 %208, 6144, !dbg !16
  %384 = icmp slt i64 %383, 0, !dbg !16
  %385 = sub nuw i64 -6145, %208, !dbg !16
  %386 = select i1 %384, i64 %385, i64 %383, !dbg !16
  %387 = sdiv i64 %386, 8192, !dbg !16
  %.lobit21 = ashr i64 %383, 63, !dbg !16
  %388 = xor i64 %387, %.lobit21, !dbg !16
  %389 = add nsw i64 %388, %16, !dbg !16
  %390 = add nsw i64 %251, 96, !dbg !16
  %391 = icmp slt i64 %251, -96, !dbg !16
  %392 = sub nsw i64 -97, %251, !dbg !16
  %393 = select i1 %391, i64 %392, i64 %390, !dbg !16
  %394 = sdiv i64 %393, 128, !dbg !16
  %395 = sext i1 %391 to i64, !dbg !16
  %396 = xor i64 %394, %395, !dbg !16
  %397 = shl nsw i64 %396, 7, !dbg !16
  %398 = sub i64 %198, %397, !dbg !16
  %399 = shl i64 %389, 19, !dbg !16
  %400 = and i64 %399, 16252928, !dbg !16
  %401 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %400, !dbg !16
  %402 = getelementptr [2 x i8], ptr addrspace(1) %401, i64 %220, !dbg !16
  %403 = getelementptr [2 x i8], ptr addrspace(1) %402, i64 %398, !dbg !16
  %404 = getelementptr i8, ptr addrspace(1) %403, i64 192, !dbg !16
  %405 = getelementptr [2 x i8], ptr addrspace(1) %404, i64 %226, !dbg !16
  %406 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local290), !dbg !16
  %407 = zext i32 %406 to i64, !dbg !16
  %408 = inttoptr i64 %407 to ptr addrspace(3), !dbg !16
  %address.local489 = getelementptr [2 x i8], ptr addrspace(1) %405, i64 %address.local286, !dbg !16
  %409 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %410 = ptrtoint ptr addrspace(1) %address.local489 to i64, !dbg !16
  %411 = sub i64 %410, %409, !dbg !16
  %dma.buffer.offset963 = trunc i64 %411 to i32, !dbg !16
  %412 = trunc i64 %409 to i32, !dbg !16
  %413 = insertelement <4 x i32> poison, i32 %412, i64 0, !dbg !16
  %414 = lshr i64 %409, 32, !dbg !16
  %415 = trunc i64 %414 to i32, !dbg !16
  %416 = insertelement <4 x i32> %413, i32 %415, i64 1, !dbg !16
  %417 = insertelement <4 x i32> %416, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc964 = insertelement <4 x i32> %417, i32 131072, i64 3, !dbg !16
  %418 = ptrtoint ptr addrspace(3) %408 to i32, !dbg !16
  %dma.buffer.lds965 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %418), !dbg !16
  %419 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds965, i32 %dma.buffer.offset963, <4 x i32> %dma.buffer.rsrc964) #10, !dbg !16
  %420 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !17
  %421 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %420, <4 x float> %355) #9, !dbg !17
  %lds.const.gep926 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !17
  %422 = load <4 x half>, ptr addrspace(3) %lds.const.gep926, align 8, !dbg !17
  %423 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !17
  %424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %422, <4 x float> %421) #9, !dbg !17
  %425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %423, <4 x float> %359) #9, !dbg !17
  %lds.const.gep927 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !17
  %426 = load <4 x half>, ptr addrspace(3) %lds.const.gep927, align 8, !dbg !17
  %lds.const.gep928 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !17
  %427 = load <4 x half>, ptr addrspace(3) %lds.const.gep928, align 8, !dbg !17
  %428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %426, <4 x float> %425) #9, !dbg !17
  %429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %427, <4 x float> %363) #9, !dbg !17
  %lds.const.gep929 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !17
  %430 = load <4 x half>, ptr addrspace(3) %lds.const.gep929, align 8, !dbg !17
  %lds.const.gep930 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !17
  %431 = load <4 x half>, ptr addrspace(3) %lds.const.gep930, align 8, !dbg !17
  %432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %430, <4 x float> %429) #9, !dbg !17
  %433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %431, <4 x float> %367) #9, !dbg !17
  %lds.const.gep931 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !17
  %434 = load <4 x half>, ptr addrspace(3) %lds.const.gep931, align 8, !dbg !17
  %435 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !17
  %436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %434, <4 x float> %433) #9, !dbg !17
  %437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %435, <4 x float> %371) #9, !dbg !17
  %lds.const.gep932 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !17
  %438 = load <4 x half>, ptr addrspace(3) %lds.const.gep932, align 8, !dbg !17
  %439 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !17
  %440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %438, <4 x float> %437) #9, !dbg !17
  %441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %439, <4 x float> %375) #9, !dbg !17
  %lds.const.gep933 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !17
  %442 = load <4 x half>, ptr addrspace(3) %lds.const.gep933, align 8, !dbg !17
  %lds.const.gep934 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !17
  %443 = load <4 x half>, ptr addrspace(3) %lds.const.gep934, align 8, !dbg !17
  %444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %442, <4 x float> %441) #9, !dbg !17
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %443, <4 x float> %379) #9, !dbg !17
  %lds.const.gep935 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !17
  %446 = load <4 x half>, ptr addrspace(3) %lds.const.gep935, align 8, !dbg !17
  %lds.const.gep936 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !17
  %447 = load <4 x half>, ptr addrspace(3) %lds.const.gep936, align 8, !dbg !17
  %448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %446, <4 x float> %445) #9, !dbg !17
  %lds.const.gep937 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !17
  %449 = load <4 x half>, ptr addrspace(3) %lds.const.gep937, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %447, <4 x float> %382) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %449, <4 x float> %450) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %452 = add nuw nsw i64 %220, %16, !dbg !18
  %453 = shl i64 %452, 19, !dbg !18
  %454 = and i64 %453, 16252928, !dbg !18
  %455 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %454, !dbg !18
  %456 = getelementptr [2 x i8], ptr addrspace(1) %455, i64 %226, !dbg !18
  %457 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local276), !dbg !18
  %458 = zext i32 %457 to i64, !dbg !18
  %459 = inttoptr i64 %458 to ptr addrspace(3), !dbg !18
  %address.local490 = shl nuw nsw i64 %address.local271, 5, !dbg !18
  %address.local491 = and i64 %address.local490, 8064, !dbg !18
  %address.local492 = shl nuw nsw i64 %address.local271, 3, !dbg !18
  %address.local493 = and i64 %address.local492, 24, !dbg !18
  %address.local494 = or disjoint i64 %address.local491, %address.local493, !dbg !18
  %address.local495 = getelementptr [2 x i8], ptr addrspace(1) %456, i64 %address.local494, !dbg !18
  %460 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %461 = ptrtoint ptr addrspace(1) %address.local495 to i64, !dbg !18
  %462 = sub i64 %461, %460, !dbg !18
  %dma.buffer.offset960 = trunc i64 %462 to i32, !dbg !18
  %463 = trunc i64 %460 to i32, !dbg !18
  %464 = insertelement <4 x i32> poison, i32 %463, i64 0, !dbg !18
  %465 = lshr i64 %460, 32, !dbg !18
  %466 = trunc i64 %465 to i32, !dbg !18
  %467 = insertelement <4 x i32> %464, i32 %466, i64 1, !dbg !18
  %468 = insertelement <4 x i32> %467, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc961 = insertelement <4 x i32> %468, i32 131072, i64 3, !dbg !18
  %469 = ptrtoint ptr addrspace(3) %459 to i32, !dbg !18
  %dma.buffer.lds962 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %469), !dbg !18
  %470 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds962, i32 %dma.buffer.offset960, <4 x i32> %dma.buffer.rsrc961) #10, !dbg !18
  %lds.const.gep938 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !17
  %471 = load <4 x half>, ptr addrspace(3) %lds.const.gep938, align 8, !dbg !17
  %472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %471, <4 x float> %424) #9, !dbg !17
  %lds.const.gep939 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !17
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep939, align 8, !dbg !17
  %lds.const.gep940 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !17
  %474 = load <4 x half>, ptr addrspace(3) %lds.const.gep940, align 8, !dbg !17
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %473, <4 x float> %472) #9, !dbg !17
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %474, <4 x float> %428) #9, !dbg !17
  %lds.const.gep941 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !17
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep941, align 8, !dbg !17
  %lds.const.gep942 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !17
  %478 = load <4 x half>, ptr addrspace(3) %lds.const.gep942, align 8, !dbg !17
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %477, <4 x float> %476) #9, !dbg !17
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %478, <4 x float> %432) #9, !dbg !17
  %lds.const.gep943 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !17
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep943, align 8, !dbg !17
  %lds.const.gep944 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !17
  %482 = load <4 x half>, ptr addrspace(3) %lds.const.gep944, align 8, !dbg !17
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %481, <4 x float> %480) #9, !dbg !17
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %482, <4 x float> %436) #9, !dbg !17
  %lds.const.gep945 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !17
  %485 = load <4 x half>, ptr addrspace(3) %lds.const.gep945, align 8, !dbg !17
  %lds.const.gep946 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !17
  %486 = load <4 x half>, ptr addrspace(3) %lds.const.gep946, align 8, !dbg !17
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %485, <4 x float> %484) #9, !dbg !17
  %488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %486, <4 x float> %440) #9, !dbg !17
  %lds.const.gep947 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !17
  %489 = load <4 x half>, ptr addrspace(3) %lds.const.gep947, align 8, !dbg !17
  %lds.const.gep948 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !17
  %490 = load <4 x half>, ptr addrspace(3) %lds.const.gep948, align 8, !dbg !17
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %489, <4 x float> %488) #9, !dbg !17
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %490, <4 x float> %444) #9, !dbg !17
  %lds.const.gep949 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !17
  %493 = load <4 x half>, ptr addrspace(3) %lds.const.gep949, align 8, !dbg !17
  %lds.const.gep950 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !17
  %494 = load <4 x half>, ptr addrspace(3) %lds.const.gep950, align 8, !dbg !17
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %493, <4 x float> %492) #9, !dbg !17
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %494, <4 x float> %448) #9, !dbg !17
  %lds.const.gep951 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !17
  %497 = load <4 x half>, ptr addrspace(3) %lds.const.gep951, align 8, !dbg !17
  %lds.const.gep952 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !17
  %498 = load <4 x half>, ptr addrspace(3) %lds.const.gep952, align 8, !dbg !17
  %499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %497, <4 x float> %496) #9, !dbg !17
  %lds.const.gep953 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !17
  %500 = load <4 x half>, ptr addrspace(3) %lds.const.gep953, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %501 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %498, <4 x float> %451) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %502 = or disjoint i64 %217, 63, !dbg !14
  %503 = icmp uge i64 %64, %502, !dbg !14
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %500, <4 x float> %501) #9, !dbg !17
  %505 = and i1 %190, %503, !dbg !14
  br i1 %505, label %506, label %507, !dbg !14

506:                                              ; preds = %216
  br label %587, !dbg !14

507:                                              ; preds = %216
  %508 = add i64 %64, %32, !dbg !14
  %509 = shl nuw nsw i64 %65, 4, !dbg !14
  %510 = add i64 %508, %509, !dbg !14
  %511 = and i32 %31, 240, !dbg !14
  %index.compact = zext nneg i32 %511 to i64, !dbg !14
  %512 = sub i64 %510, %index.compact, !dbg !14
  %513 = and i64 %32, 48, !dbg !14
  %514 = or disjoint i64 %217, %513, !dbg !14
  %515 = add i64 %512, 1, !dbg !23
  %516 = lshr i64 %515, 32, !dbg !24
  %compare.high = trunc i64 %516 to i32, !dbg !24
  %517 = lshr i64 %514, 32, !dbg !24
  %compare.high757 = trunc i64 %517 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high757, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high757, !dbg !24
  %compare.low.base = trunc i64 %515 to i32, !dbg !24
  %compare.low.base758 = trunc i64 %514 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base758, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %518 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low759 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result760 = select i1 %compare.high.equal, i1 %compare.low759, i1 %compare.high.order, !dbg !24
  %519 = select i1 %compare.result760, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base761 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local762 = or i32 %compare.local.base761, 2, !dbg !24
  %compare.low763 = icmp ugt i32 %compare.low.base, %compare.local762, !dbg !24
  %compare.result764 = select i1 %compare.high.equal, i1 %compare.low763, i1 %compare.high.order, !dbg !24
  %520 = select i1 %compare.result764, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base765 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local766 = or i32 %compare.local.base765, 3, !dbg !24
  %compare.low767 = icmp ugt i32 %compare.low.base, %compare.local766, !dbg !24
  %compare.result768 = select i1 %compare.high.equal, i1 %compare.low767, i1 %compare.high.order, !dbg !24
  %521 = select i1 %compare.result768, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base769 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local770 = or i32 %compare.local.base769, 4, !dbg !24
  %compare.low771 = icmp ugt i32 %compare.low.base, %compare.local770, !dbg !24
  %compare.result772 = select i1 %compare.high.equal, i1 %compare.low771, i1 %compare.high.order, !dbg !24
  %522 = select i1 %compare.result772, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base773 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local774 = or i32 %compare.local.base773, 5, !dbg !24
  %compare.low775 = icmp ugt i32 %compare.low.base, %compare.local774, !dbg !24
  %compare.result776 = select i1 %compare.high.equal, i1 %compare.low775, i1 %compare.high.order, !dbg !24
  %523 = select i1 %compare.result776, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base777 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local778 = or i32 %compare.local.base777, 6, !dbg !24
  %compare.low779 = icmp ugt i32 %compare.low.base, %compare.local778, !dbg !24
  %compare.result780 = select i1 %compare.high.equal, i1 %compare.low779, i1 %compare.high.order, !dbg !24
  %524 = select i1 %compare.result780, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base781 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local782 = or i32 %compare.local.base781, 7, !dbg !24
  %compare.low783 = icmp ugt i32 %compare.low.base, %compare.local782, !dbg !24
  %compare.result784 = select i1 %compare.high.equal, i1 %compare.low783, i1 %compare.high.order, !dbg !24
  %525 = select i1 %compare.result784, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base785 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local786 = or i32 %compare.local.base785, 8, !dbg !24
  %compare.low787 = icmp ugt i32 %compare.low.base, %compare.local786, !dbg !24
  %compare.result788 = select i1 %compare.high.equal, i1 %compare.low787, i1 %compare.high.order, !dbg !24
  %526 = select i1 %compare.result788, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base789 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local790 = or i32 %compare.local.base789, 9, !dbg !24
  %compare.low791 = icmp ugt i32 %compare.low.base, %compare.local790, !dbg !24
  %compare.result792 = select i1 %compare.high.equal, i1 %compare.low791, i1 %compare.high.order, !dbg !24
  %527 = select i1 %compare.result792, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base793 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local794 = or i32 %compare.local.base793, 10, !dbg !24
  %compare.low795 = icmp ugt i32 %compare.low.base, %compare.local794, !dbg !24
  %compare.result796 = select i1 %compare.high.equal, i1 %compare.low795, i1 %compare.high.order, !dbg !24
  %528 = select i1 %compare.result796, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base797 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local798 = or i32 %compare.local.base797, 11, !dbg !24
  %compare.low799 = icmp ugt i32 %compare.low.base, %compare.local798, !dbg !24
  %compare.result800 = select i1 %compare.high.equal, i1 %compare.low799, i1 %compare.high.order, !dbg !24
  %529 = select i1 %compare.result800, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base801 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local802 = or i32 %compare.local.base801, 12, !dbg !24
  %compare.low803 = icmp ugt i32 %compare.low.base, %compare.local802, !dbg !24
  %compare.result804 = select i1 %compare.high.equal, i1 %compare.low803, i1 %compare.high.order, !dbg !24
  %530 = select i1 %compare.result804, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base805 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local806 = or i32 %compare.local.base805, 13, !dbg !24
  %compare.low807 = icmp ugt i32 %compare.low.base, %compare.local806, !dbg !24
  %compare.result808 = select i1 %compare.high.equal, i1 %compare.low807, i1 %compare.high.order, !dbg !24
  %531 = select i1 %compare.result808, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base809 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local810 = or i32 %compare.local.base809, 14, !dbg !24
  %compare.low811 = icmp ugt i32 %compare.low.base, %compare.local810, !dbg !24
  %compare.result812 = select i1 %compare.high.equal, i1 %compare.low811, i1 %compare.high.order, !dbg !24
  %532 = select i1 %compare.result812, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base813 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local814 = or i32 %compare.local.base813, 15, !dbg !24
  %compare.low815 = icmp ugt i32 %compare.low.base, %compare.local814, !dbg !24
  %compare.result816 = select i1 %compare.high.equal, i1 %compare.low815, i1 %compare.high.order, !dbg !24
  %533 = select i1 %compare.result816, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %534 = add i64 %512, 65, !dbg !23
  %535 = lshr i64 %534, 32, !dbg !24
  %compare.high817 = trunc i64 %535 to i32, !dbg !24
  %536 = lshr i64 %514, 32, !dbg !24
  %compare.high818 = trunc i64 %536 to i32, !dbg !24
  %compare.high.equal819 = icmp eq i32 %compare.high817, %compare.high818, !dbg !24
  %compare.high.order820 = icmp ugt i32 %compare.high817, %compare.high818, !dbg !24
  %compare.low.base821 = trunc i64 %534 to i32, !dbg !24
  %compare.low.base822 = trunc i64 %514 to i32, !dbg !24
  %compare.low823 = icmp ugt i32 %compare.low.base821, %compare.low.base822, !dbg !24
  %compare.result824 = select i1 %compare.high.equal819, i1 %compare.low823, i1 %compare.high.order820, !dbg !24
  %537 = select i1 %compare.result824, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base825 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local826 = or i32 %compare.local.base825, 1, !dbg !24
  %compare.low827 = icmp ugt i32 %compare.low.base821, %compare.local826, !dbg !24
  %compare.result828 = select i1 %compare.high.equal819, i1 %compare.low827, i1 %compare.high.order820, !dbg !24
  %538 = select i1 %compare.result828, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base829 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local830 = or i32 %compare.local.base829, 2, !dbg !24
  %compare.low831 = icmp ugt i32 %compare.low.base821, %compare.local830, !dbg !24
  %compare.result832 = select i1 %compare.high.equal819, i1 %compare.low831, i1 %compare.high.order820, !dbg !24
  %539 = select i1 %compare.result832, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base833 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local834 = or i32 %compare.local.base833, 3, !dbg !24
  %compare.low835 = icmp ugt i32 %compare.low.base821, %compare.local834, !dbg !24
  %compare.result836 = select i1 %compare.high.equal819, i1 %compare.low835, i1 %compare.high.order820, !dbg !24
  %540 = select i1 %compare.result836, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base837 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local838 = or i32 %compare.local.base837, 4, !dbg !24
  %compare.low839 = icmp ugt i32 %compare.low.base821, %compare.local838, !dbg !24
  %compare.result840 = select i1 %compare.high.equal819, i1 %compare.low839, i1 %compare.high.order820, !dbg !24
  %541 = select i1 %compare.result840, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base841 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local842 = or i32 %compare.local.base841, 5, !dbg !24
  %compare.low843 = icmp ugt i32 %compare.low.base821, %compare.local842, !dbg !24
  %compare.result844 = select i1 %compare.high.equal819, i1 %compare.low843, i1 %compare.high.order820, !dbg !24
  %542 = select i1 %compare.result844, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base845 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local846 = or i32 %compare.local.base845, 6, !dbg !24
  %compare.low847 = icmp ugt i32 %compare.low.base821, %compare.local846, !dbg !24
  %compare.result848 = select i1 %compare.high.equal819, i1 %compare.low847, i1 %compare.high.order820, !dbg !24
  %543 = select i1 %compare.result848, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base849 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local850 = or i32 %compare.local.base849, 7, !dbg !24
  %compare.low851 = icmp ugt i32 %compare.low.base821, %compare.local850, !dbg !24
  %compare.result852 = select i1 %compare.high.equal819, i1 %compare.low851, i1 %compare.high.order820, !dbg !24
  %544 = select i1 %compare.result852, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base853 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local854 = or i32 %compare.local.base853, 8, !dbg !24
  %compare.low855 = icmp ugt i32 %compare.low.base821, %compare.local854, !dbg !24
  %compare.result856 = select i1 %compare.high.equal819, i1 %compare.low855, i1 %compare.high.order820, !dbg !24
  %545 = select i1 %compare.result856, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base857 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local858 = or i32 %compare.local.base857, 9, !dbg !24
  %compare.low859 = icmp ugt i32 %compare.low.base821, %compare.local858, !dbg !24
  %compare.result860 = select i1 %compare.high.equal819, i1 %compare.low859, i1 %compare.high.order820, !dbg !24
  %546 = select i1 %compare.result860, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base861 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local862 = or i32 %compare.local.base861, 10, !dbg !24
  %compare.low863 = icmp ugt i32 %compare.low.base821, %compare.local862, !dbg !24
  %compare.result864 = select i1 %compare.high.equal819, i1 %compare.low863, i1 %compare.high.order820, !dbg !24
  %547 = select i1 %compare.result864, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base865 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local866 = or i32 %compare.local.base865, 11, !dbg !24
  %compare.low867 = icmp ugt i32 %compare.low.base821, %compare.local866, !dbg !24
  %compare.result868 = select i1 %compare.high.equal819, i1 %compare.low867, i1 %compare.high.order820, !dbg !24
  %548 = select i1 %compare.result868, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base869 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local870 = or i32 %compare.local.base869, 12, !dbg !24
  %compare.low871 = icmp ugt i32 %compare.low.base821, %compare.local870, !dbg !24
  %compare.result872 = select i1 %compare.high.equal819, i1 %compare.low871, i1 %compare.high.order820, !dbg !24
  %549 = select i1 %compare.result872, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base873 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local874 = or i32 %compare.local.base873, 13, !dbg !24
  %compare.low875 = icmp ugt i32 %compare.low.base821, %compare.local874, !dbg !24
  %compare.result876 = select i1 %compare.high.equal819, i1 %compare.low875, i1 %compare.high.order820, !dbg !24
  %550 = select i1 %compare.result876, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base877 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local878 = or i32 %compare.local.base877, 14, !dbg !24
  %compare.low879 = icmp ugt i32 %compare.low.base821, %compare.local878, !dbg !24
  %compare.result880 = select i1 %compare.high.equal819, i1 %compare.low879, i1 %compare.high.order820, !dbg !24
  %551 = select i1 %compare.result880, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base881 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local882 = or i32 %compare.local.base881, 15, !dbg !24
  %compare.low883 = icmp ugt i32 %compare.low.base821, %compare.local882, !dbg !24
  %compare.result884 = select i1 %compare.high.equal819, i1 %compare.low883, i1 %compare.high.order820, !dbg !24
  %552 = select i1 %compare.result884, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %553 = insertelement <16 x float> poison, float %518, i64 0, !dbg !14
  %554 = insertelement <16 x float> %553, float %519, i64 1, !dbg !14
  %555 = insertelement <16 x float> %554, float %520, i64 2, !dbg !14
  %556 = insertelement <16 x float> %555, float %521, i64 3, !dbg !14
  %557 = insertelement <16 x float> %556, float %522, i64 4, !dbg !14
  %558 = insertelement <16 x float> %557, float %523, i64 5, !dbg !14
  %559 = insertelement <16 x float> %558, float %524, i64 6, !dbg !14
  %560 = insertelement <16 x float> %559, float %525, i64 7, !dbg !14
  %561 = insertelement <16 x float> %560, float %526, i64 8, !dbg !14
  %562 = insertelement <16 x float> %561, float %527, i64 9, !dbg !14
  %563 = insertelement <16 x float> %562, float %528, i64 10, !dbg !14
  %564 = insertelement <16 x float> %563, float %529, i64 11, !dbg !14
  %565 = insertelement <16 x float> %564, float %530, i64 12, !dbg !14
  %566 = insertelement <16 x float> %565, float %531, i64 13, !dbg !14
  %567 = insertelement <16 x float> %566, float %532, i64 14, !dbg !14
  %568 = insertelement <16 x float> %567, float %533, i64 15, !dbg !14
  %569 = insertvalue [2 x <16 x float>] poison, <16 x float> %568, 0, !dbg !14
  %570 = insertelement <16 x float> poison, float %537, i64 0, !dbg !14
  %571 = insertelement <16 x float> %570, float %538, i64 1, !dbg !14
  %572 = insertelement <16 x float> %571, float %539, i64 2, !dbg !14
  %573 = insertelement <16 x float> %572, float %540, i64 3, !dbg !14
  %574 = insertelement <16 x float> %573, float %541, i64 4, !dbg !14
  %575 = insertelement <16 x float> %574, float %542, i64 5, !dbg !14
  %576 = insertelement <16 x float> %575, float %543, i64 6, !dbg !14
  %577 = insertelement <16 x float> %576, float %544, i64 7, !dbg !14
  %578 = insertelement <16 x float> %577, float %545, i64 8, !dbg !14
  %579 = insertelement <16 x float> %578, float %546, i64 9, !dbg !14
  %580 = insertelement <16 x float> %579, float %547, i64 10, !dbg !14
  %581 = insertelement <16 x float> %580, float %548, i64 11, !dbg !14
  %582 = insertelement <16 x float> %581, float %549, i64 12, !dbg !14
  %583 = insertelement <16 x float> %582, float %550, i64 13, !dbg !14
  %584 = insertelement <16 x float> %583, float %551, i64 14, !dbg !14
  %585 = insertelement <16 x float> %584, float %552, i64 15, !dbg !14
  %586 = insertvalue [2 x <16 x float>] %569, <16 x float> %585, 1, !dbg !14
  br label %587, !dbg !14

587:                                              ; preds = %506, %507
  %588 = phi [2 x <16 x float>] [ %586, %507 ], [ zeroinitializer, %506 ], !dbg !14
  br label %589, !dbg !14

589:                                              ; preds = %587
  %590 = extractvalue [2 x <16 x float>] %588, 0, !dbg !26
  %591 = shufflevector <16 x float> %590, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %592 = fadd <4 x float> %475, %591, !dbg !26
  %593 = extractelement <4 x float> %592, i64 0, !dbg !26
  %594 = extractelement <4 x float> %592, i64 1, !dbg !26
  %595 = extractelement <4 x float> %592, i64 2, !dbg !26
  %596 = extractelement <4 x float> %592, i64 3, !dbg !26
  %597 = shufflevector <16 x float> %590, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %598 = fadd <4 x float> %479, %597, !dbg !26
  %599 = extractelement <4 x float> %598, i64 0, !dbg !26
  %600 = extractelement <4 x float> %598, i64 1, !dbg !26
  %601 = extractelement <4 x float> %598, i64 2, !dbg !26
  %602 = extractelement <4 x float> %598, i64 3, !dbg !26
  %603 = shufflevector <16 x float> %590, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %604 = fadd <4 x float> %483, %603, !dbg !26
  %605 = extractelement <4 x float> %604, i64 0, !dbg !26
  %606 = extractelement <4 x float> %604, i64 1, !dbg !26
  %607 = extractelement <4 x float> %604, i64 2, !dbg !26
  %608 = extractelement <4 x float> %604, i64 3, !dbg !26
  %609 = shufflevector <16 x float> %590, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %610 = fadd <4 x float> %487, %609, !dbg !26
  %611 = extractelement <4 x float> %610, i64 0, !dbg !26
  %612 = extractelement <4 x float> %610, i64 1, !dbg !26
  %613 = extractelement <4 x float> %610, i64 2, !dbg !26
  %614 = extractelement <4 x float> %610, i64 3, !dbg !26
  %615 = extractvalue [2 x <16 x float>] %588, 1, !dbg !26
  %616 = shufflevector <16 x float> %615, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %617 = fadd <4 x float> %491, %616, !dbg !26
  %618 = extractelement <4 x float> %617, i64 0, !dbg !26
  %619 = extractelement <4 x float> %617, i64 1, !dbg !26
  %620 = extractelement <4 x float> %617, i64 2, !dbg !26
  %621 = extractelement <4 x float> %617, i64 3, !dbg !26
  %622 = shufflevector <16 x float> %615, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %623 = fadd <4 x float> %495, %622, !dbg !26
  %624 = extractelement <4 x float> %623, i64 0, !dbg !26
  %625 = extractelement <4 x float> %623, i64 1, !dbg !26
  %626 = extractelement <4 x float> %623, i64 2, !dbg !26
  %627 = extractelement <4 x float> %623, i64 3, !dbg !26
  %628 = shufflevector <16 x float> %615, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %629 = fadd <4 x float> %499, %628, !dbg !26
  %630 = extractelement <4 x float> %629, i64 0, !dbg !26
  %631 = extractelement <4 x float> %629, i64 1, !dbg !26
  %632 = extractelement <4 x float> %629, i64 2, !dbg !26
  %633 = extractelement <4 x float> %629, i64 3, !dbg !26
  %634 = shufflevector <16 x float> %615, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %635 = fadd <4 x float> %504, %634, !dbg !26
  %636 = extractelement <4 x float> %635, i64 0, !dbg !26
  %637 = extractelement <4 x float> %635, i64 1, !dbg !26
  %638 = extractelement <4 x float> %635, i64 2, !dbg !26
  %639 = extractelement <4 x float> %635, i64 3, !dbg !26
  %640 = call float @llvm.maxnum.f32(float %593, float 0xFFF0000000000000), !dbg !27
  %641 = call float @llvm.maxnum.f32(float %640, float %594), !dbg !27
  %642 = call float @llvm.maxnum.f32(float %641, float %595), !dbg !27
  %643 = call float @llvm.maxnum.f32(float %642, float %596), !dbg !27
  %644 = call float @llvm.maxnum.f32(float %643, float %599), !dbg !27
  %645 = call float @llvm.maxnum.f32(float %644, float %600), !dbg !27
  %646 = call float @llvm.maxnum.f32(float %645, float %601), !dbg !27
  %647 = call float @llvm.maxnum.f32(float %646, float %602), !dbg !27
  %648 = call float @llvm.maxnum.f32(float %647, float %605), !dbg !27
  %649 = call float @llvm.maxnum.f32(float %648, float %606), !dbg !27
  %650 = call float @llvm.maxnum.f32(float %649, float %607), !dbg !27
  %651 = call float @llvm.maxnum.f32(float %650, float %608), !dbg !27
  %652 = call float @llvm.maxnum.f32(float %651, float %611), !dbg !27
  %653 = call float @llvm.maxnum.f32(float %652, float %612), !dbg !27
  %654 = call float @llvm.maxnum.f32(float %653, float %613), !dbg !27
  %655 = call float @llvm.maxnum.f32(float %654, float %614), !dbg !27
  %656 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %657 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %656), !dbg !27
  %658 = add nuw nsw i32 %657, 64, !dbg !27
  %659 = xor i32 %657, 16, !dbg !27
  %660 = and i32 %658, 192, !dbg !27
  %661 = icmp samesign ult i32 %659, %660, !dbg !27
  %662 = select i1 %661, i32 %659, i32 %657, !dbg !27
  %663 = shl nuw nsw i32 %662, 2, !dbg !27
  %664 = bitcast float %655 to i32, !dbg !27
  %665 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %664), !dbg !27
  %666 = bitcast i32 %665 to float, !dbg !27
  %667 = call float @llvm.maxnum.f32(float %655, float %666), !dbg !27
  %668 = xor i32 %657, 32, !dbg !27
  %669 = icmp samesign ult i32 %668, %660, !dbg !27
  %670 = select i1 %669, i32 %668, i32 %657, !dbg !27
  %671 = shl nuw nsw i32 %670, 2, !dbg !27
  %672 = bitcast float %667 to i32, !dbg !27
  %673 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %672), !dbg !27
  %674 = bitcast i32 %673 to float, !dbg !27
  %675 = call float @llvm.maxnum.f32(float %667, float %674), !dbg !27
  %676 = extractvalue [2 x <1 x float>] %211, 0, !dbg !27
  %677 = insertelement <1 x float> poison, float %675, i64 0, !dbg !27
  %678 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %677, <1 x float> %676), !dbg !27
  %679 = call float @llvm.maxnum.f32(float %618, float 0xFFF0000000000000), !dbg !27
  %680 = call float @llvm.maxnum.f32(float %679, float %619), !dbg !27
  %681 = call float @llvm.maxnum.f32(float %680, float %620), !dbg !27
  %682 = call float @llvm.maxnum.f32(float %681, float %621), !dbg !27
  %683 = call float @llvm.maxnum.f32(float %682, float %624), !dbg !27
  %684 = call float @llvm.maxnum.f32(float %683, float %625), !dbg !27
  %685 = call float @llvm.maxnum.f32(float %684, float %626), !dbg !27
  %686 = call float @llvm.maxnum.f32(float %685, float %627), !dbg !27
  %687 = call float @llvm.maxnum.f32(float %686, float %630), !dbg !27
  %688 = call float @llvm.maxnum.f32(float %687, float %631), !dbg !27
  %689 = call float @llvm.maxnum.f32(float %688, float %632), !dbg !27
  %690 = call float @llvm.maxnum.f32(float %689, float %633), !dbg !27
  %691 = call float @llvm.maxnum.f32(float %690, float %636), !dbg !27
  %692 = call float @llvm.maxnum.f32(float %691, float %637), !dbg !27
  %693 = call float @llvm.maxnum.f32(float %692, float %638), !dbg !27
  %694 = call float @llvm.maxnum.f32(float %693, float %639), !dbg !27
  %695 = bitcast float %694 to i32, !dbg !27
  %696 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %695), !dbg !27
  %697 = bitcast i32 %696 to float, !dbg !27
  %698 = call float @llvm.maxnum.f32(float %694, float %697), !dbg !27
  %699 = bitcast float %698 to i32, !dbg !27
  %700 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %699), !dbg !27
  %701 = bitcast i32 %700 to float, !dbg !27
  %702 = call float @llvm.maxnum.f32(float %698, float %701), !dbg !27
  %703 = extractvalue [2 x <1 x float>] %211, 1, !dbg !27
  %704 = insertelement <1 x float> poison, float %702, i64 0, !dbg !27
  %705 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %704, <1 x float> %703), !dbg !27
  %706 = shufflevector <1 x float> %678, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %707 = fsub <4 x float> %592, %706, !dbg !28
  %708 = fsub <4 x float> %598, %706, !dbg !28
  %709 = fsub <4 x float> %604, %706, !dbg !28
  %710 = fsub <4 x float> %610, %706, !dbg !28
  %711 = shufflevector <1 x float> %705, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %712 = fsub <4 x float> %617, %711, !dbg !28
  %713 = fsub <4 x float> %623, %711, !dbg !28
  %714 = fsub <4 x float> %629, %711, !dbg !28
  %715 = fsub <4 x float> %635, %711, !dbg !28
  %716 = fsub <1 x float> %676, %678, !dbg !29
  %717 = extractelement <1 x float> %716, i64 0, !dbg !30
  %718 = call float @llvm.amdgcn.exp2.f32(float %717), !dbg !30
  %719 = insertelement <4 x float> poison, float %718, i64 0, !dbg !35
  %720 = shufflevector <4 x float> %719, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %721 = extractvalue [2 x <1 x float>] %210, 0, !dbg !31
  %722 = extractelement <1 x float> %721, i64 0, !dbg !31
  %723 = fmul float %722, %718, !dbg !31
  %724 = fsub <1 x float> %703, %705, !dbg !29
  %725 = extractelement <1 x float> %724, i64 0, !dbg !30
  %726 = call float @llvm.amdgcn.exp2.f32(float %725), !dbg !30
  %727 = insertelement <4 x float> poison, float %726, i64 0, !dbg !35
  %728 = shufflevector <4 x float> %727, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %729 = extractvalue [2 x <1 x float>] %210, 1, !dbg !31
  %730 = extractelement <1 x float> %729, i64 0, !dbg !31
  %731 = fmul float %730, %726, !dbg !31
  %732 = extractelement <4 x float> %707, i64 0, !dbg !32
  %733 = call float @llvm.amdgcn.exp2.f32(float %732), !dbg !32
  %734 = insertelement <4 x float> poison, float %733, i64 0, !dbg !34
  %735 = extractelement <4 x float> %707, i64 1, !dbg !32
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !32
  %737 = insertelement <4 x float> %734, float %736, i64 1, !dbg !34
  %738 = fadd float %733, %736, !dbg !33
  %739 = extractelement <4 x float> %707, i64 2, !dbg !32
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !32
  %741 = insertelement <4 x float> %737, float %740, i64 2, !dbg !34
  %742 = fadd float %738, %740, !dbg !33
  %743 = extractelement <4 x float> %707, i64 3, !dbg !32
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !32
  %745 = insertelement <4 x float> %741, float %744, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %745) #9, !dbg !34, !frisk.register_lifetimes !1
  %746 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %746) #9, !dbg !33, !frisk.register_lifetimes !1
  %747 = fadd float %742, %744, !dbg !33
  %748 = extractelement <4 x float> %708, i64 0, !dbg !32
  %749 = call float @llvm.amdgcn.exp2.f32(float %748), !dbg !32
  %750 = insertelement <4 x float> poison, float %749, i64 0, !dbg !34
  %751 = fadd float %747, %749, !dbg !33
  %752 = extractelement <4 x float> %708, i64 1, !dbg !32
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !32
  %754 = insertelement <4 x float> %750, float %753, i64 1, !dbg !34
  %755 = fadd float %751, %753, !dbg !33
  %756 = extractelement <4 x float> %708, i64 2, !dbg !32
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !32
  %758 = insertelement <4 x float> %754, float %757, i64 2, !dbg !34
  %759 = fadd float %755, %757, !dbg !33
  %760 = extractelement <4 x float> %708, i64 3, !dbg !32
  %761 = call float @llvm.amdgcn.exp2.f32(float %760), !dbg !32
  %762 = insertelement <4 x float> %758, float %761, i64 3, !dbg !34
  %math.result162 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %762) #9, !dbg !34, !frisk.register_lifetimes !1
  %763 = fptrunc <4 x float> %math.result162 to <4 x half>, !dbg !34
  %math.packed163 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %763) #9, !dbg !33, !frisk.register_lifetimes !1
  %764 = fadd float %759, %761, !dbg !33
  %765 = extractelement <4 x float> %709, i64 0, !dbg !32
  %766 = call float @llvm.amdgcn.exp2.f32(float %765), !dbg !32
  %767 = insertelement <4 x float> poison, float %766, i64 0, !dbg !34
  %768 = fadd float %764, %766, !dbg !33
  %769 = extractelement <4 x float> %709, i64 1, !dbg !32
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !32
  %771 = insertelement <4 x float> %767, float %770, i64 1, !dbg !34
  %772 = fadd float %768, %770, !dbg !33
  %773 = extractelement <4 x float> %709, i64 2, !dbg !32
  %774 = call float @llvm.amdgcn.exp2.f32(float %773), !dbg !32
  %775 = insertelement <4 x float> %771, float %774, i64 2, !dbg !34
  %776 = fadd float %772, %774, !dbg !33
  %777 = extractelement <4 x float> %709, i64 3, !dbg !32
  %778 = call float @llvm.amdgcn.exp2.f32(float %777), !dbg !32
  %779 = insertelement <4 x float> %775, float %778, i64 3, !dbg !34
  %math.result164 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %779) #9, !dbg !34, !frisk.register_lifetimes !1
  %780 = fptrunc <4 x float> %math.result164 to <4 x half>, !dbg !34
  %math.packed165 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %780) #9, !dbg !33, !frisk.register_lifetimes !1
  %781 = fadd float %776, %778, !dbg !33
  %782 = extractelement <4 x float> %710, i64 0, !dbg !32
  %783 = call float @llvm.amdgcn.exp2.f32(float %782), !dbg !32
  %784 = insertelement <4 x float> poison, float %783, i64 0, !dbg !34
  %785 = fadd float %781, %783, !dbg !33
  %786 = extractelement <4 x float> %710, i64 1, !dbg !32
  %787 = call float @llvm.amdgcn.exp2.f32(float %786), !dbg !32
  %788 = insertelement <4 x float> %784, float %787, i64 1, !dbg !34
  %789 = fadd float %785, %787, !dbg !33
  %790 = extractelement <4 x float> %710, i64 2, !dbg !32
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !32
  %792 = insertelement <4 x float> %788, float %791, i64 2, !dbg !34
  %793 = fadd float %789, %791, !dbg !33
  %794 = extractelement <4 x float> %710, i64 3, !dbg !32
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !32
  %796 = insertelement <4 x float> %792, float %795, i64 3, !dbg !34
  %math.result166 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %796) #9, !dbg !34, !frisk.register_lifetimes !1
  %797 = fptrunc <4 x float> %math.result166 to <4 x half>, !dbg !34
  %math.packed167 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %797) #9, !dbg !33, !frisk.register_lifetimes !1
  %798 = fadd float %793, %795, !dbg !33
  %799 = extractelement <4 x float> %712, i64 0, !dbg !32
  %800 = call float @llvm.amdgcn.exp2.f32(float %799), !dbg !32
  %801 = insertelement <4 x float> poison, float %800, i64 0, !dbg !34
  %802 = extractelement <4 x float> %712, i64 1, !dbg !32
  %803 = call float @llvm.amdgcn.exp2.f32(float %802), !dbg !32
  %804 = insertelement <4 x float> %801, float %803, i64 1, !dbg !34
  %805 = fadd float %800, %803, !dbg !33
  %806 = extractelement <4 x float> %712, i64 2, !dbg !32
  %807 = call float @llvm.amdgcn.exp2.f32(float %806), !dbg !32
  %808 = insertelement <4 x float> %804, float %807, i64 2, !dbg !34
  %809 = fadd float %805, %807, !dbg !33
  %810 = extractelement <4 x float> %712, i64 3, !dbg !32
  %811 = call float @llvm.amdgcn.exp2.f32(float %810), !dbg !32
  %812 = insertelement <4 x float> %808, float %811, i64 3, !dbg !34
  %math.result168 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %812) #9, !dbg !34, !frisk.register_lifetimes !1
  %813 = fptrunc <4 x float> %math.result168 to <4 x half>, !dbg !34
  %math.packed169 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %813) #9, !dbg !33, !frisk.register_lifetimes !1
  %814 = fadd float %809, %811, !dbg !33
  %815 = extractelement <4 x float> %713, i64 0, !dbg !32
  %816 = call float @llvm.amdgcn.exp2.f32(float %815), !dbg !32
  %817 = insertelement <4 x float> poison, float %816, i64 0, !dbg !34
  %818 = fadd float %814, %816, !dbg !33
  %819 = extractelement <4 x float> %713, i64 1, !dbg !32
  %820 = call float @llvm.amdgcn.exp2.f32(float %819), !dbg !32
  %821 = insertelement <4 x float> %817, float %820, i64 1, !dbg !34
  %822 = fadd float %818, %820, !dbg !33
  %823 = extractelement <4 x float> %713, i64 2, !dbg !32
  %824 = call float @llvm.amdgcn.exp2.f32(float %823), !dbg !32
  %825 = insertelement <4 x float> %821, float %824, i64 2, !dbg !34
  %826 = fadd float %822, %824, !dbg !33
  %827 = extractelement <4 x float> %713, i64 3, !dbg !32
  %828 = call float @llvm.amdgcn.exp2.f32(float %827), !dbg !32
  %829 = insertelement <4 x float> %825, float %828, i64 3, !dbg !34
  %math.result170 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %829) #9, !dbg !34, !frisk.register_lifetimes !1
  %830 = fptrunc <4 x float> %math.result170 to <4 x half>, !dbg !34
  %math.packed171 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %830) #9, !dbg !33, !frisk.register_lifetimes !1
  %831 = fadd float %826, %828, !dbg !33
  %832 = extractelement <4 x float> %714, i64 0, !dbg !32
  %833 = call float @llvm.amdgcn.exp2.f32(float %832), !dbg !32
  %834 = insertelement <4 x float> poison, float %833, i64 0, !dbg !34
  %835 = fadd float %831, %833, !dbg !33
  %836 = extractelement <4 x float> %714, i64 1, !dbg !32
  %837 = call float @llvm.amdgcn.exp2.f32(float %836), !dbg !32
  %838 = insertelement <4 x float> %834, float %837, i64 1, !dbg !34
  %839 = fadd float %835, %837, !dbg !33
  %840 = extractelement <4 x float> %714, i64 2, !dbg !32
  %841 = call float @llvm.amdgcn.exp2.f32(float %840), !dbg !32
  %842 = insertelement <4 x float> %838, float %841, i64 2, !dbg !34
  %843 = fadd float %839, %841, !dbg !33
  %844 = extractelement <4 x float> %714, i64 3, !dbg !32
  %845 = call float @llvm.amdgcn.exp2.f32(float %844), !dbg !32
  %846 = insertelement <4 x float> %842, float %845, i64 3, !dbg !34
  %math.result172 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %846) #9, !dbg !34, !frisk.register_lifetimes !1
  %847 = fptrunc <4 x float> %math.result172 to <4 x half>, !dbg !34
  %math.packed173 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %847) #9, !dbg !33, !frisk.register_lifetimes !1
  %848 = fadd float %843, %845, !dbg !33
  %849 = extractelement <4 x float> %715, i64 0, !dbg !32
  %850 = call float @llvm.amdgcn.exp2.f32(float %849), !dbg !32
  %851 = insertelement <4 x float> poison, float %850, i64 0, !dbg !34
  %852 = fadd float %848, %850, !dbg !33
  %853 = extractelement <4 x float> %715, i64 1, !dbg !32
  %854 = call float @llvm.amdgcn.exp2.f32(float %853), !dbg !32
  %855 = insertelement <4 x float> %851, float %854, i64 1, !dbg !34
  %856 = fadd float %852, %854, !dbg !33
  %857 = extractelement <4 x float> %715, i64 2, !dbg !32
  %858 = call float @llvm.amdgcn.exp2.f32(float %857), !dbg !32
  %859 = insertelement <4 x float> %855, float %858, i64 2, !dbg !34
  %860 = fadd float %856, %858, !dbg !33
  %861 = extractelement <4 x float> %715, i64 3, !dbg !32
  %862 = call float @llvm.amdgcn.exp2.f32(float %861), !dbg !32
  %863 = insertelement <4 x float> %859, float %862, i64 3, !dbg !34
  %math.result174 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %863) #9, !dbg !34, !frisk.register_lifetimes !1
  %864 = fptrunc <4 x float> %math.result174 to <4 x half>, !dbg !34
  %math.packed175 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %864) #9, !dbg !33, !frisk.register_lifetimes !1
  %865 = fadd float %860, %862, !dbg !33
  %866 = bitcast float %798 to i32, !dbg !33
  %867 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %866), !dbg !33
  %868 = bitcast i32 %867 to float, !dbg !33
  %869 = fadd float %798, %868, !dbg !33
  %870 = bitcast float %869 to i32, !dbg !33
  %871 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %870), !dbg !33
  %872 = bitcast i32 %871 to float, !dbg !33
  %873 = fadd float %869, %872, !dbg !33
  %874 = fadd float %873, %723, !dbg !33
  %875 = insertelement <1 x float> poison, float %874, i64 0, !dbg !7
  %876 = insertvalue [2 x <1 x float>] poison, <1 x float> %875, 0, !dbg !7
  %877 = bitcast float %865 to i32, !dbg !33
  %878 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %877), !dbg !33
  %879 = bitcast i32 %878 to float, !dbg !33
  %880 = fadd float %865, %879, !dbg !33
  %881 = bitcast float %880 to i32, !dbg !33
  %882 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %881), !dbg !33
  %883 = bitcast i32 %882 to float, !dbg !33
  %884 = fadd float %880, %883, !dbg !33
  %885 = fadd float %884, %731, !dbg !33
  %886 = insertelement <1 x float> poison, float %885, i64 0, !dbg !7
  %887 = insertvalue [2 x <1 x float>] %876, <1 x float> %886, 1, !dbg !7
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
  %888 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !15
  %889 = bitcast <4 x i32> %888 to <8 x half>, !dbg !15
  %address.local509 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !15
  %address.local510 = getelementptr [2 x i8], ptr addrspace(3) %address.local509, i64 %address.local504, !dbg !15
  %address.local511 = getelementptr [2 x i8], ptr addrspace(3) %address.local510, i64 %address.local507, !dbg !15
  %address.local512 = getelementptr i8, ptr addrspace(3) %address.local511, i64 256, !dbg !15
  %890 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !15
  %891 = bitcast <4 x i32> %890 to <8 x half>, !dbg !15
  %address.local513 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !15
  %address.local514 = getelementptr [2 x i8], ptr addrspace(3) %address.local513, i64 %address.local504, !dbg !15
  %address.local515 = getelementptr [2 x i8], ptr addrspace(3) %address.local514, i64 %address.local507, !dbg !15
  %address.local516 = getelementptr i8, ptr addrspace(3) %address.local515, i64 512, !dbg !15
  %892 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !15
  %893 = bitcast <4 x i32> %892 to <8 x half>, !dbg !15
  %address.local517 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !15
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) %address.local517, i64 %address.local504, !dbg !15
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) %address.local518, i64 %address.local507, !dbg !15
  %address.local520 = getelementptr i8, ptr addrspace(3) %address.local519, i64 768, !dbg !15
  %894 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !15
  %895 = bitcast <4 x i32> %894 to <8 x half>, !dbg !15
  %896 = add nuw nsw i64 %220, %16, !dbg !18
  %897 = shl i64 %896, 19, !dbg !18
  %898 = and i64 %897, 16252928, !dbg !18
  %899 = add i64 %898, %226, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %900 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %899, !dbg !18
  %901 = getelementptr i8, ptr addrspace(1) %900, i64 64, !dbg !18
  %address.local521 = shl nuw nsw i64 %address.local497, 3, !dbg !9
  %address.local522 = and i64 %address.local521, 1536, !dbg !16
  %address.local523 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local522, !dbg !16
  %address.local524 = ptrtoint ptr addrspace(3) %address.local523 to i64, !dbg !16
  %address.local525 = trunc i64 %address.local524 to i32, !dbg !16
  %902 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !18
  %903 = zext i32 %902 to i64, !dbg !18
  %904 = inttoptr i64 %903 to ptr addrspace(3), !dbg !18
  %address.local526 = shl nuw nsw i64 %address.local497, 5, !dbg !18
  %address.local527 = and i64 %address.local526, 8064, !dbg !18
  %address.local528 = shl nuw nsw i64 %address.local497, 3, !dbg !18
  %address.local529 = and i64 %address.local528, 24, !dbg !18
  %address.local530 = or disjoint i64 %address.local527, %address.local529, !dbg !18
  %address.local531 = getelementptr [2 x i8], ptr addrspace(1) %901, i64 %address.local530, !dbg !18
  %905 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %906 = ptrtoint ptr addrspace(1) %address.local531 to i64, !dbg !18
  %907 = sub i64 %906, %905, !dbg !18
  %dma.buffer.offset957 = trunc i64 %907 to i32, !dbg !18
  %908 = trunc i64 %905 to i32, !dbg !18
  %909 = insertelement <4 x i32> poison, i32 %908, i64 0, !dbg !18
  %910 = lshr i64 %905, 32, !dbg !18
  %911 = trunc i64 %910 to i32, !dbg !18
  %912 = insertelement <4 x i32> %909, i32 %911, i64 1, !dbg !18
  %913 = insertelement <4 x i32> %912, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc958 = insertelement <4 x i32> %913, i32 131072, i64 3, !dbg !18
  %914 = ptrtoint ptr addrspace(3) %904 to i32, !dbg !18
  %dma.buffer.lds959 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %914), !dbg !18
  %915 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds959, i32 %dma.buffer.offset957, <4 x i32> %dma.buffer.rsrc958) #10, !dbg !18
  %916 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %916, <4 x float> zeroinitializer) #9, !dbg !15
  %918 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %918, <4 x float> %917) #9, !dbg !15
  %920 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %920, <4 x float> %919) #9, !dbg !15
  %922 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %922, <4 x float> %921) #9, !dbg !15
  %924 = extractelement <4 x float> %923, i64 0, !dbg !15
  %925 = extractelement <4 x float> %923, i64 1, !dbg !15
  %926 = extractelement <4 x float> %923, i64 2, !dbg !15
  %927 = extractelement <4 x float> %923, i64 3, !dbg !15
  %928 = select i1 %.not, float %925, float %924, !dbg !15
  %929 = bitcast float %928 to i32, !dbg !15
  %930 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %929), !dbg !15
  %931 = bitcast i32 %930 to float, !dbg !15
  %932 = select i1 %.not, float %924, float %931, !dbg !15
  %933 = select i1 %.not, float %931, float %925, !dbg !15
  %934 = select i1 %.not, float %927, float %926, !dbg !15
  %935 = bitcast float %934 to i32, !dbg !15
  %936 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %935), !dbg !15
  %937 = bitcast i32 %936 to float, !dbg !15
  %938 = select i1 %.not, float %926, float %937, !dbg !15
  %939 = select i1 %.not, float %937, float %927, !dbg !15
  %940 = select i1 %.not1, float %938, float %932, !dbg !15
  %941 = bitcast float %940 to i32, !dbg !15
  %942 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %941), !dbg !15
  %943 = bitcast i32 %942 to float, !dbg !15
  %944 = select i1 %.not1, float %932, float %943, !dbg !15
  %945 = select i1 %.not1, float %943, float %938, !dbg !15
  %946 = select i1 %.not1, float %939, float %933, !dbg !15
  %947 = bitcast float %946 to i32, !dbg !15
  %948 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %947), !dbg !15
  %949 = bitcast i32 %948 to float, !dbg !15
  %950 = select i1 %.not1, float %933, float %949, !dbg !15
  %951 = select i1 %.not1, float %949, float %939, !dbg !15
  %952 = extractvalue [2 x <8 x float>] %209, 0, !dbg !35
  %953 = shufflevector <8 x float> %952, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %954 = fmul <4 x float> %953, %720, !dbg !35
  %955 = insertelement <4 x float> poison, float %944, i64 0, !dbg !36
  %956 = insertelement <4 x float> %955, float %950, i64 1, !dbg !36
  %957 = insertelement <4 x float> %956, float %945, i64 2, !dbg !36
  %958 = insertelement <4 x float> %957, float %951, i64 3, !dbg !36
  %959 = fadd <4 x float> %954, %958, !dbg !36
  %960 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %960, <4 x float> zeroinitializer) #9, !dbg !15
  %962 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %962, <4 x float> %961) #9, !dbg !15
  %964 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %964, <4 x float> %963) #9, !dbg !15
  %966 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %966, <4 x float> %965) #9, !dbg !15
  %968 = extractelement <4 x float> %967, i64 0, !dbg !15
  %969 = extractelement <4 x float> %967, i64 1, !dbg !15
  %970 = extractelement <4 x float> %967, i64 2, !dbg !15
  %971 = extractelement <4 x float> %967, i64 3, !dbg !15
  %972 = select i1 %.not, float %969, float %968, !dbg !15
  %973 = bitcast float %972 to i32, !dbg !15
  %974 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %973), !dbg !15
  %975 = bitcast i32 %974 to float, !dbg !15
  %976 = select i1 %.not, float %968, float %975, !dbg !15
  %977 = select i1 %.not, float %975, float %969, !dbg !15
  %978 = select i1 %.not, float %971, float %970, !dbg !15
  %979 = bitcast float %978 to i32, !dbg !15
  %980 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %979), !dbg !15
  %981 = bitcast i32 %980 to float, !dbg !15
  %982 = select i1 %.not, float %970, float %981, !dbg !15
  %983 = select i1 %.not, float %981, float %971, !dbg !15
  %984 = select i1 %.not1, float %982, float %976, !dbg !15
  %985 = bitcast float %984 to i32, !dbg !15
  %986 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %985), !dbg !15
  %987 = bitcast i32 %986 to float, !dbg !15
  %988 = select i1 %.not1, float %976, float %987, !dbg !15
  %989 = select i1 %.not1, float %987, float %982, !dbg !15
  %990 = select i1 %.not1, float %983, float %977, !dbg !15
  %991 = bitcast float %990 to i32, !dbg !15
  %992 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %991), !dbg !15
  %993 = bitcast i32 %992 to float, !dbg !15
  %994 = select i1 %.not1, float %977, float %993, !dbg !15
  %995 = select i1 %.not1, float %993, float %983, !dbg !15
  %996 = shufflevector <8 x float> %952, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %997 = fmul <4 x float> %996, %720, !dbg !35
  %998 = insertelement <4 x float> poison, float %988, i64 0, !dbg !36
  %999 = insertelement <4 x float> %998, float %994, i64 1, !dbg !36
  %1000 = insertelement <4 x float> %999, float %989, i64 2, !dbg !36
  %1001 = insertelement <4 x float> %1000, float %995, i64 3, !dbg !36
  %1002 = fadd <4 x float> %997, %1001, !dbg !36
  %1003 = shufflevector <4 x float> %959, <4 x float> %1002, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1004 = insertvalue [2 x <8 x float>] poison, <8 x float> %1003, 0, !dbg !7
  %1005 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %916, <4 x float> zeroinitializer) #9, !dbg !15
  %1006 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %918, <4 x float> %1005) #9, !dbg !15
  %1007 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %920, <4 x float> %1006) #9, !dbg !15
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %922, <4 x float> %1007) #9, !dbg !15
  %1009 = extractelement <4 x float> %1008, i64 0, !dbg !15
  %1010 = extractelement <4 x float> %1008, i64 1, !dbg !15
  %1011 = extractelement <4 x float> %1008, i64 2, !dbg !15
  %1012 = extractelement <4 x float> %1008, i64 3, !dbg !15
  %1013 = select i1 %.not, float %1010, float %1009, !dbg !15
  %1014 = bitcast float %1013 to i32, !dbg !15
  %1015 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1014), !dbg !15
  %1016 = bitcast i32 %1015 to float, !dbg !15
  %1017 = select i1 %.not, float %1009, float %1016, !dbg !15
  %1018 = select i1 %.not, float %1016, float %1010, !dbg !15
  %1019 = select i1 %.not, float %1012, float %1011, !dbg !15
  %1020 = bitcast float %1019 to i32, !dbg !15
  %1021 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1020), !dbg !15
  %1022 = bitcast i32 %1021 to float, !dbg !15
  %1023 = select i1 %.not, float %1011, float %1022, !dbg !15
  %1024 = select i1 %.not, float %1022, float %1012, !dbg !15
  %1025 = select i1 %.not1, float %1023, float %1017, !dbg !15
  %1026 = bitcast float %1025 to i32, !dbg !15
  %1027 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1026), !dbg !15
  %1028 = bitcast i32 %1027 to float, !dbg !15
  %1029 = select i1 %.not1, float %1017, float %1028, !dbg !15
  %1030 = select i1 %.not1, float %1028, float %1023, !dbg !15
  %1031 = select i1 %.not1, float %1024, float %1018, !dbg !15
  %1032 = bitcast float %1031 to i32, !dbg !15
  %1033 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1032), !dbg !15
  %1034 = bitcast i32 %1033 to float, !dbg !15
  %1035 = select i1 %.not1, float %1018, float %1034, !dbg !15
  %1036 = select i1 %.not1, float %1034, float %1024, !dbg !15
  %1037 = extractvalue [2 x <8 x float>] %209, 1, !dbg !35
  %1038 = shufflevector <8 x float> %1037, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1039 = fmul <4 x float> %1038, %728, !dbg !35
  %1040 = insertelement <4 x float> poison, float %1029, i64 0, !dbg !36
  %1041 = insertelement <4 x float> %1040, float %1035, i64 1, !dbg !36
  %1042 = insertelement <4 x float> %1041, float %1030, i64 2, !dbg !36
  %1043 = insertelement <4 x float> %1042, float %1036, i64 3, !dbg !36
  %1044 = fadd <4 x float> %1039, %1043, !dbg !36
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %960, <4 x float> zeroinitializer) #9, !dbg !15
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %962, <4 x float> %1045) #9, !dbg !15
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %964, <4 x float> %1046) #9, !dbg !15
  %1048 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %966, <4 x float> %1047) #9, !dbg !15
  %1049 = extractelement <4 x float> %1048, i64 0, !dbg !15
  %1050 = extractelement <4 x float> %1048, i64 1, !dbg !15
  %1051 = extractelement <4 x float> %1048, i64 2, !dbg !15
  %1052 = extractelement <4 x float> %1048, i64 3, !dbg !15
  %1053 = select i1 %.not, float %1050, float %1049, !dbg !15
  %1054 = bitcast float %1053 to i32, !dbg !15
  %1055 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1054), !dbg !15
  %1056 = bitcast i32 %1055 to float, !dbg !15
  %1057 = select i1 %.not, float %1049, float %1056, !dbg !15
  %1058 = select i1 %.not, float %1056, float %1050, !dbg !15
  %1059 = select i1 %.not, float %1052, float %1051, !dbg !15
  %1060 = bitcast float %1059 to i32, !dbg !15
  %1061 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1060), !dbg !15
  %1062 = bitcast i32 %1061 to float, !dbg !15
  %1063 = select i1 %.not, float %1051, float %1062, !dbg !15
  %1064 = select i1 %.not, float %1062, float %1052, !dbg !15
  %1065 = select i1 %.not1, float %1063, float %1057, !dbg !15
  %1066 = bitcast float %1065 to i32, !dbg !15
  %1067 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1066), !dbg !15
  %1068 = bitcast i32 %1067 to float, !dbg !15
  %1069 = select i1 %.not1, float %1057, float %1068, !dbg !15
  %1070 = select i1 %.not1, float %1068, float %1063, !dbg !15
  %1071 = select i1 %.not1, float %1064, float %1058, !dbg !15
  %1072 = bitcast float %1071 to i32, !dbg !15
  %1073 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1072), !dbg !15
  %1074 = bitcast i32 %1073 to float, !dbg !15
  %1075 = select i1 %.not1, float %1058, float %1074, !dbg !15
  %1076 = select i1 %.not1, float %1074, float %1064, !dbg !15
  %1077 = shufflevector <8 x float> %1037, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1078 = fmul <4 x float> %1077, %728, !dbg !35
  %1079 = insertelement <4 x float> poison, float %1069, i64 0, !dbg !36
  %1080 = insertelement <4 x float> %1079, float %1075, i64 1, !dbg !36
  %1081 = insertelement <4 x float> %1080, float %1070, i64 2, !dbg !36
  %1082 = insertelement <4 x float> %1081, float %1076, i64 3, !dbg !36
  %1083 = fadd <4 x float> %1078, %1082, !dbg !36
  %1084 = shufflevector <4 x float> %1044, <4 x float> %1083, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1085 = insertvalue [2 x <8 x float>] %1004, <8 x float> %1084, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1086 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !15
  %1087 = bitcast <4 x i32> %1086 to <8 x half>, !dbg !15
  %1088 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !15
  %1089 = bitcast <4 x i32> %1088 to <8 x half>, !dbg !15
  %1090 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !15
  %1091 = bitcast <4 x i32> %1090 to <8 x half>, !dbg !15
  %1092 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !15
  %1093 = bitcast <4 x i32> %1092 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1094 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %899, !dbg !18
  %1095 = getelementptr i8, ptr addrspace(1) %1094, i64 128, !dbg !18
  %1096 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !18
  %1097 = zext i32 %1096 to i64, !dbg !18
  %1098 = inttoptr i64 %1097 to ptr addrspace(3), !dbg !18
  %address.local532 = getelementptr [2 x i8], ptr addrspace(1) %1095, i64 %address.local530, !dbg !18
  %1099 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1100 = ptrtoint ptr addrspace(1) %address.local532 to i64, !dbg !18
  %1101 = sub i64 %1100, %1099, !dbg !18
  %dma.buffer.offset954 = trunc i64 %1101 to i32, !dbg !18
  %1102 = trunc i64 %1099 to i32, !dbg !18
  %1103 = insertelement <4 x i32> poison, i32 %1102, i64 0, !dbg !18
  %1104 = lshr i64 %1099, 32, !dbg !18
  %1105 = trunc i64 %1104 to i32, !dbg !18
  %1106 = insertelement <4 x i32> %1103, i32 %1105, i64 1, !dbg !18
  %1107 = insertelement <4 x i32> %1106, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc955 = insertelement <4 x i32> %1107, i32 131072, i64 3, !dbg !18
  %1108 = ptrtoint ptr addrspace(3) %1098 to i32, !dbg !18
  %dma.buffer.lds956 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1108), !dbg !18
  %1109 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds956, i32 %dma.buffer.offset954, <4 x i32> %dma.buffer.rsrc955) #10, !dbg !18
  %1110 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1111 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1110, <4 x float> zeroinitializer) #9, !dbg !15
  %1112 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1113 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1112, <4 x float> %1111) #9, !dbg !15
  %1114 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1115 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1114, <4 x float> %1113) #9, !dbg !15
  %1116 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1116, <4 x float> %1115) #9, !dbg !15
  %1118 = extractelement <4 x float> %1117, i64 0, !dbg !15
  %1119 = extractelement <4 x float> %1117, i64 1, !dbg !15
  %1120 = extractelement <4 x float> %1117, i64 2, !dbg !15
  %1121 = extractelement <4 x float> %1117, i64 3, !dbg !15
  %1122 = select i1 %.not, float %1119, float %1118, !dbg !15
  %1123 = bitcast float %1122 to i32, !dbg !15
  %1124 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1123), !dbg !15
  %1125 = bitcast i32 %1124 to float, !dbg !15
  %1126 = select i1 %.not, float %1118, float %1125, !dbg !15
  %1127 = select i1 %.not, float %1125, float %1119, !dbg !15
  %1128 = select i1 %.not, float %1121, float %1120, !dbg !15
  %1129 = bitcast float %1128 to i32, !dbg !15
  %1130 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1129), !dbg !15
  %1131 = bitcast i32 %1130 to float, !dbg !15
  %1132 = select i1 %.not, float %1120, float %1131, !dbg !15
  %1133 = select i1 %.not, float %1131, float %1121, !dbg !15
  %1134 = select i1 %.not1, float %1132, float %1126, !dbg !15
  %1135 = bitcast float %1134 to i32, !dbg !15
  %1136 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1135), !dbg !15
  %1137 = bitcast i32 %1136 to float, !dbg !15
  %1138 = select i1 %.not1, float %1126, float %1137, !dbg !15
  %1139 = select i1 %.not1, float %1137, float %1132, !dbg !15
  %1140 = select i1 %.not1, float %1133, float %1127, !dbg !15
  %1141 = bitcast float %1140 to i32, !dbg !15
  %1142 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1141), !dbg !15
  %1143 = bitcast i32 %1142 to float, !dbg !15
  %1144 = select i1 %.not1, float %1127, float %1143, !dbg !15
  %1145 = select i1 %.not1, float %1143, float %1133, !dbg !15
  %1146 = extractvalue [2 x <8 x float>] %212, 0, !dbg !35
  %1147 = shufflevector <8 x float> %1146, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1148 = fmul <4 x float> %1147, %720, !dbg !35
  %1149 = insertelement <4 x float> poison, float %1138, i64 0, !dbg !36
  %1150 = insertelement <4 x float> %1149, float %1144, i64 1, !dbg !36
  %1151 = insertelement <4 x float> %1150, float %1139, i64 2, !dbg !36
  %1152 = insertelement <4 x float> %1151, float %1145, i64 3, !dbg !36
  %1153 = fadd <4 x float> %1148, %1152, !dbg !36
  %1154 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1154, <4 x float> zeroinitializer) #9, !dbg !15
  %1156 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1156, <4 x float> %1155) #9, !dbg !15
  %1158 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1158, <4 x float> %1157) #9, !dbg !15
  %1160 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1160, <4 x float> %1159) #9, !dbg !15
  %1162 = extractelement <4 x float> %1161, i64 0, !dbg !15
  %1163 = extractelement <4 x float> %1161, i64 1, !dbg !15
  %1164 = extractelement <4 x float> %1161, i64 2, !dbg !15
  %1165 = extractelement <4 x float> %1161, i64 3, !dbg !15
  %1166 = select i1 %.not, float %1163, float %1162, !dbg !15
  %1167 = bitcast float %1166 to i32, !dbg !15
  %1168 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1167), !dbg !15
  %1169 = bitcast i32 %1168 to float, !dbg !15
  %1170 = select i1 %.not, float %1162, float %1169, !dbg !15
  %1171 = select i1 %.not, float %1169, float %1163, !dbg !15
  %1172 = select i1 %.not, float %1165, float %1164, !dbg !15
  %1173 = bitcast float %1172 to i32, !dbg !15
  %1174 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1173), !dbg !15
  %1175 = bitcast i32 %1174 to float, !dbg !15
  %1176 = select i1 %.not, float %1164, float %1175, !dbg !15
  %1177 = select i1 %.not, float %1175, float %1165, !dbg !15
  %1178 = select i1 %.not1, float %1176, float %1170, !dbg !15
  %1179 = bitcast float %1178 to i32, !dbg !15
  %1180 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1179), !dbg !15
  %1181 = bitcast i32 %1180 to float, !dbg !15
  %1182 = select i1 %.not1, float %1170, float %1181, !dbg !15
  %1183 = select i1 %.not1, float %1181, float %1176, !dbg !15
  %1184 = select i1 %.not1, float %1177, float %1171, !dbg !15
  %1185 = bitcast float %1184 to i32, !dbg !15
  %1186 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1185), !dbg !15
  %1187 = bitcast i32 %1186 to float, !dbg !15
  %1188 = select i1 %.not1, float %1171, float %1187, !dbg !15
  %1189 = select i1 %.not1, float %1187, float %1177, !dbg !15
  %1190 = shufflevector <8 x float> %1146, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1191 = fmul <4 x float> %1190, %720, !dbg !35
  %1192 = insertelement <4 x float> poison, float %1182, i64 0, !dbg !36
  %1193 = insertelement <4 x float> %1192, float %1188, i64 1, !dbg !36
  %1194 = insertelement <4 x float> %1193, float %1183, i64 2, !dbg !36
  %1195 = insertelement <4 x float> %1194, float %1189, i64 3, !dbg !36
  %1196 = fadd <4 x float> %1191, %1195, !dbg !36
  %1197 = shufflevector <4 x float> %1153, <4 x float> %1196, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1198 = insertvalue [2 x <8 x float>] poison, <8 x float> %1197, 0, !dbg !7
  %1199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1110, <4 x float> zeroinitializer) #9, !dbg !15
  %1200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1112, <4 x float> %1199) #9, !dbg !15
  %1201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1114, <4 x float> %1200) #9, !dbg !15
  %1202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1116, <4 x float> %1201) #9, !dbg !15
  %1203 = extractelement <4 x float> %1202, i64 0, !dbg !15
  %1204 = extractelement <4 x float> %1202, i64 1, !dbg !15
  %1205 = extractelement <4 x float> %1202, i64 2, !dbg !15
  %1206 = extractelement <4 x float> %1202, i64 3, !dbg !15
  %1207 = select i1 %.not, float %1204, float %1203, !dbg !15
  %1208 = bitcast float %1207 to i32, !dbg !15
  %1209 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1208), !dbg !15
  %1210 = bitcast i32 %1209 to float, !dbg !15
  %1211 = select i1 %.not, float %1203, float %1210, !dbg !15
  %1212 = select i1 %.not, float %1210, float %1204, !dbg !15
  %1213 = select i1 %.not, float %1206, float %1205, !dbg !15
  %1214 = bitcast float %1213 to i32, !dbg !15
  %1215 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1214), !dbg !15
  %1216 = bitcast i32 %1215 to float, !dbg !15
  %1217 = select i1 %.not, float %1205, float %1216, !dbg !15
  %1218 = select i1 %.not, float %1216, float %1206, !dbg !15
  %1219 = select i1 %.not1, float %1217, float %1211, !dbg !15
  %1220 = bitcast float %1219 to i32, !dbg !15
  %1221 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1220), !dbg !15
  %1222 = bitcast i32 %1221 to float, !dbg !15
  %1223 = select i1 %.not1, float %1211, float %1222, !dbg !15
  %1224 = select i1 %.not1, float %1222, float %1217, !dbg !15
  %1225 = select i1 %.not1, float %1218, float %1212, !dbg !15
  %1226 = bitcast float %1225 to i32, !dbg !15
  %1227 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1226), !dbg !15
  %1228 = bitcast i32 %1227 to float, !dbg !15
  %1229 = select i1 %.not1, float %1212, float %1228, !dbg !15
  %1230 = select i1 %.not1, float %1228, float %1218, !dbg !15
  %1231 = extractvalue [2 x <8 x float>] %212, 1, !dbg !35
  %1232 = shufflevector <8 x float> %1231, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1233 = fmul <4 x float> %1232, %728, !dbg !35
  %1234 = insertelement <4 x float> poison, float %1223, i64 0, !dbg !36
  %1235 = insertelement <4 x float> %1234, float %1229, i64 1, !dbg !36
  %1236 = insertelement <4 x float> %1235, float %1224, i64 2, !dbg !36
  %1237 = insertelement <4 x float> %1236, float %1230, i64 3, !dbg !36
  %1238 = fadd <4 x float> %1233, %1237, !dbg !36
  %1239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1154, <4 x float> zeroinitializer) #9, !dbg !15
  %1240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1156, <4 x float> %1239) #9, !dbg !15
  %1241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1158, <4 x float> %1240) #9, !dbg !15
  %1242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1160, <4 x float> %1241) #9, !dbg !15
  %1243 = extractelement <4 x float> %1242, i64 0, !dbg !15
  %1244 = extractelement <4 x float> %1242, i64 1, !dbg !15
  %1245 = extractelement <4 x float> %1242, i64 2, !dbg !15
  %1246 = extractelement <4 x float> %1242, i64 3, !dbg !15
  %1247 = select i1 %.not, float %1244, float %1243, !dbg !15
  %1248 = bitcast float %1247 to i32, !dbg !15
  %1249 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1248), !dbg !15
  %1250 = bitcast i32 %1249 to float, !dbg !15
  %1251 = select i1 %.not, float %1243, float %1250, !dbg !15
  %1252 = select i1 %.not, float %1250, float %1244, !dbg !15
  %1253 = select i1 %.not, float %1246, float %1245, !dbg !15
  %1254 = bitcast float %1253 to i32, !dbg !15
  %1255 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1254), !dbg !15
  %1256 = bitcast i32 %1255 to float, !dbg !15
  %1257 = select i1 %.not, float %1245, float %1256, !dbg !15
  %1258 = select i1 %.not, float %1256, float %1246, !dbg !15
  %1259 = select i1 %.not1, float %1257, float %1251, !dbg !15
  %1260 = bitcast float %1259 to i32, !dbg !15
  %1261 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1260), !dbg !15
  %1262 = bitcast i32 %1261 to float, !dbg !15
  %1263 = select i1 %.not1, float %1251, float %1262, !dbg !15
  %1264 = select i1 %.not1, float %1262, float %1257, !dbg !15
  %1265 = select i1 %.not1, float %1258, float %1252, !dbg !15
  %1266 = bitcast float %1265 to i32, !dbg !15
  %1267 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1266), !dbg !15
  %1268 = bitcast i32 %1267 to float, !dbg !15
  %1269 = select i1 %.not1, float %1252, float %1268, !dbg !15
  %1270 = select i1 %.not1, float %1268, float %1258, !dbg !15
  %1271 = shufflevector <8 x float> %1231, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1272 = fmul <4 x float> %1271, %728, !dbg !35
  %1273 = insertelement <4 x float> poison, float %1263, i64 0, !dbg !36
  %1274 = insertelement <4 x float> %1273, float %1269, i64 1, !dbg !36
  %1275 = insertelement <4 x float> %1274, float %1264, i64 2, !dbg !36
  %1276 = insertelement <4 x float> %1275, float %1270, i64 3, !dbg !36
  %1277 = fadd <4 x float> %1272, %1276, !dbg !36
  %1278 = shufflevector <4 x float> %1238, <4 x float> %1277, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1279 = insertvalue [2 x <8 x float>] %1198, <8 x float> %1278, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1280 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !15
  %1281 = bitcast <4 x i32> %1280 to <8 x half>, !dbg !15
  %1282 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !15
  %1283 = bitcast <4 x i32> %1282 to <8 x half>, !dbg !15
  %1284 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !15
  %1285 = bitcast <4 x i32> %1284 to <8 x half>, !dbg !15
  %1286 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !15
  %1287 = bitcast <4 x i32> %1286 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1288 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %899, !dbg !18
  %1289 = getelementptr i8, ptr addrspace(1) %1288, i64 192, !dbg !18
  %1290 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !18
  %1291 = zext i32 %1290 to i64, !dbg !18
  %1292 = inttoptr i64 %1291 to ptr addrspace(3), !dbg !18
  %address.local533 = getelementptr [2 x i8], ptr addrspace(1) %1289, i64 %address.local530, !dbg !18
  %1293 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1294 = ptrtoint ptr addrspace(1) %address.local533 to i64, !dbg !18
  %1295 = sub i64 %1294, %1293, !dbg !18
  %dma.buffer.offset = trunc i64 %1295 to i32, !dbg !18
  %1296 = trunc i64 %1293 to i32, !dbg !18
  %1297 = insertelement <4 x i32> poison, i32 %1296, i64 0, !dbg !18
  %1298 = lshr i64 %1293, 32, !dbg !18
  %1299 = trunc i64 %1298 to i32, !dbg !18
  %1300 = insertelement <4 x i32> %1297, i32 %1299, i64 1, !dbg !18
  %1301 = insertelement <4 x i32> %1300, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1301, i32 131072, i64 3, !dbg !18
  %1302 = ptrtoint ptr addrspace(3) %1292 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1302), !dbg !18
  %1303 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1304 = shufflevector <8 x half> %1281, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1304, <4 x float> zeroinitializer) #9, !dbg !15
  %1306 = shufflevector <8 x half> %1283, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1306, <4 x float> %1305) #9, !dbg !15
  %1308 = shufflevector <8 x half> %1285, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1308, <4 x float> %1307) #9, !dbg !15
  %1310 = shufflevector <8 x half> %1287, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1310, <4 x float> %1309) #9, !dbg !15
  %1312 = extractelement <4 x float> %1311, i64 0, !dbg !15
  %1313 = extractelement <4 x float> %1311, i64 1, !dbg !15
  %1314 = extractelement <4 x float> %1311, i64 2, !dbg !15
  %1315 = extractelement <4 x float> %1311, i64 3, !dbg !15
  %1316 = select i1 %.not, float %1313, float %1312, !dbg !15
  %1317 = bitcast float %1316 to i32, !dbg !15
  %1318 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1317), !dbg !15
  %1319 = bitcast i32 %1318 to float, !dbg !15
  %1320 = select i1 %.not, float %1312, float %1319, !dbg !15
  %1321 = select i1 %.not, float %1319, float %1313, !dbg !15
  %1322 = select i1 %.not, float %1315, float %1314, !dbg !15
  %1323 = bitcast float %1322 to i32, !dbg !15
  %1324 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1323), !dbg !15
  %1325 = bitcast i32 %1324 to float, !dbg !15
  %1326 = select i1 %.not, float %1314, float %1325, !dbg !15
  %1327 = select i1 %.not, float %1325, float %1315, !dbg !15
  %1328 = select i1 %.not1, float %1326, float %1320, !dbg !15
  %1329 = bitcast float %1328 to i32, !dbg !15
  %1330 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1329), !dbg !15
  %1331 = bitcast i32 %1330 to float, !dbg !15
  %1332 = select i1 %.not1, float %1320, float %1331, !dbg !15
  %1333 = select i1 %.not1, float %1331, float %1326, !dbg !15
  %1334 = select i1 %.not1, float %1327, float %1321, !dbg !15
  %1335 = bitcast float %1334 to i32, !dbg !15
  %1336 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1335), !dbg !15
  %1337 = bitcast i32 %1336 to float, !dbg !15
  %1338 = select i1 %.not1, float %1321, float %1337, !dbg !15
  %1339 = select i1 %.not1, float %1337, float %1327, !dbg !15
  %1340 = extractvalue [2 x <8 x float>] %213, 0, !dbg !35
  %1341 = shufflevector <8 x float> %1340, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1342 = fmul <4 x float> %1341, %720, !dbg !35
  %1343 = insertelement <4 x float> poison, float %1332, i64 0, !dbg !36
  %1344 = insertelement <4 x float> %1343, float %1338, i64 1, !dbg !36
  %1345 = insertelement <4 x float> %1344, float %1333, i64 2, !dbg !36
  %1346 = insertelement <4 x float> %1345, float %1339, i64 3, !dbg !36
  %1347 = fadd <4 x float> %1342, %1346, !dbg !36
  %1348 = shufflevector <8 x half> %1281, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1348, <4 x float> zeroinitializer) #9, !dbg !15
  %1350 = shufflevector <8 x half> %1283, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1350, <4 x float> %1349) #9, !dbg !15
  %1352 = shufflevector <8 x half> %1285, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1352, <4 x float> %1351) #9, !dbg !15
  %1354 = shufflevector <8 x half> %1287, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1354, <4 x float> %1353) #9, !dbg !15
  %1356 = extractelement <4 x float> %1355, i64 0, !dbg !15
  %1357 = extractelement <4 x float> %1355, i64 1, !dbg !15
  %1358 = extractelement <4 x float> %1355, i64 2, !dbg !15
  %1359 = extractelement <4 x float> %1355, i64 3, !dbg !15
  %1360 = select i1 %.not, float %1357, float %1356, !dbg !15
  %1361 = bitcast float %1360 to i32, !dbg !15
  %1362 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1361), !dbg !15
  %1363 = bitcast i32 %1362 to float, !dbg !15
  %1364 = select i1 %.not, float %1356, float %1363, !dbg !15
  %1365 = select i1 %.not, float %1363, float %1357, !dbg !15
  %1366 = select i1 %.not, float %1359, float %1358, !dbg !15
  %1367 = bitcast float %1366 to i32, !dbg !15
  %1368 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1367), !dbg !15
  %1369 = bitcast i32 %1368 to float, !dbg !15
  %1370 = select i1 %.not, float %1358, float %1369, !dbg !15
  %1371 = select i1 %.not, float %1369, float %1359, !dbg !15
  %1372 = select i1 %.not1, float %1370, float %1364, !dbg !15
  %1373 = bitcast float %1372 to i32, !dbg !15
  %1374 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1373), !dbg !15
  %1375 = bitcast i32 %1374 to float, !dbg !15
  %1376 = select i1 %.not1, float %1364, float %1375, !dbg !15
  %1377 = select i1 %.not1, float %1375, float %1370, !dbg !15
  %1378 = select i1 %.not1, float %1371, float %1365, !dbg !15
  %1379 = bitcast float %1378 to i32, !dbg !15
  %1380 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1379), !dbg !15
  %1381 = bitcast i32 %1380 to float, !dbg !15
  %1382 = select i1 %.not1, float %1365, float %1381, !dbg !15
  %1383 = select i1 %.not1, float %1381, float %1371, !dbg !15
  %1384 = shufflevector <8 x float> %1340, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1385 = fmul <4 x float> %1384, %720, !dbg !35
  %1386 = insertelement <4 x float> poison, float %1376, i64 0, !dbg !36
  %1387 = insertelement <4 x float> %1386, float %1382, i64 1, !dbg !36
  %1388 = insertelement <4 x float> %1387, float %1377, i64 2, !dbg !36
  %1389 = insertelement <4 x float> %1388, float %1383, i64 3, !dbg !36
  %1390 = fadd <4 x float> %1385, %1389, !dbg !36
  %1391 = shufflevector <4 x float> %1347, <4 x float> %1390, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1392 = insertvalue [2 x <8 x float>] poison, <8 x float> %1391, 0, !dbg !7
  %1393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1304, <4 x float> zeroinitializer) #9, !dbg !15
  %1394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1306, <4 x float> %1393) #9, !dbg !15
  %1395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1308, <4 x float> %1394) #9, !dbg !15
  %1396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1310, <4 x float> %1395) #9, !dbg !15
  %1397 = extractelement <4 x float> %1396, i64 0, !dbg !15
  %1398 = extractelement <4 x float> %1396, i64 1, !dbg !15
  %1399 = extractelement <4 x float> %1396, i64 2, !dbg !15
  %1400 = extractelement <4 x float> %1396, i64 3, !dbg !15
  %1401 = select i1 %.not, float %1398, float %1397, !dbg !15
  %1402 = bitcast float %1401 to i32, !dbg !15
  %1403 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1402), !dbg !15
  %1404 = bitcast i32 %1403 to float, !dbg !15
  %1405 = select i1 %.not, float %1397, float %1404, !dbg !15
  %1406 = select i1 %.not, float %1404, float %1398, !dbg !15
  %1407 = select i1 %.not, float %1400, float %1399, !dbg !15
  %1408 = bitcast float %1407 to i32, !dbg !15
  %1409 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1408), !dbg !15
  %1410 = bitcast i32 %1409 to float, !dbg !15
  %1411 = select i1 %.not, float %1399, float %1410, !dbg !15
  %1412 = select i1 %.not, float %1410, float %1400, !dbg !15
  %1413 = select i1 %.not1, float %1411, float %1405, !dbg !15
  %1414 = bitcast float %1413 to i32, !dbg !15
  %1415 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1414), !dbg !15
  %1416 = bitcast i32 %1415 to float, !dbg !15
  %1417 = select i1 %.not1, float %1405, float %1416, !dbg !15
  %1418 = select i1 %.not1, float %1416, float %1411, !dbg !15
  %1419 = select i1 %.not1, float %1412, float %1406, !dbg !15
  %1420 = bitcast float %1419 to i32, !dbg !15
  %1421 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1420), !dbg !15
  %1422 = bitcast i32 %1421 to float, !dbg !15
  %1423 = select i1 %.not1, float %1406, float %1422, !dbg !15
  %1424 = select i1 %.not1, float %1422, float %1412, !dbg !15
  %1425 = extractvalue [2 x <8 x float>] %213, 1, !dbg !35
  %1426 = shufflevector <8 x float> %1425, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1427 = fmul <4 x float> %1426, %728, !dbg !35
  %1428 = insertelement <4 x float> poison, float %1417, i64 0, !dbg !36
  %1429 = insertelement <4 x float> %1428, float %1423, i64 1, !dbg !36
  %1430 = insertelement <4 x float> %1429, float %1418, i64 2, !dbg !36
  %1431 = insertelement <4 x float> %1430, float %1424, i64 3, !dbg !36
  %1432 = fadd <4 x float> %1427, %1431, !dbg !36
  %1433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1348, <4 x float> zeroinitializer) #9, !dbg !15
  %1434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1350, <4 x float> %1433) #9, !dbg !15
  %1435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1352, <4 x float> %1434) #9, !dbg !15
  %1436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1354, <4 x float> %1435) #9, !dbg !15
  %1437 = extractelement <4 x float> %1436, i64 0, !dbg !15
  %1438 = extractelement <4 x float> %1436, i64 1, !dbg !15
  %1439 = extractelement <4 x float> %1436, i64 2, !dbg !15
  %1440 = extractelement <4 x float> %1436, i64 3, !dbg !15
  %1441 = select i1 %.not, float %1438, float %1437, !dbg !15
  %1442 = bitcast float %1441 to i32, !dbg !15
  %1443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1442), !dbg !15
  %1444 = bitcast i32 %1443 to float, !dbg !15
  %1445 = select i1 %.not, float %1437, float %1444, !dbg !15
  %1446 = select i1 %.not, float %1444, float %1438, !dbg !15
  %1447 = select i1 %.not, float %1440, float %1439, !dbg !15
  %1448 = bitcast float %1447 to i32, !dbg !15
  %1449 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1448), !dbg !15
  %1450 = bitcast i32 %1449 to float, !dbg !15
  %1451 = select i1 %.not, float %1439, float %1450, !dbg !15
  %1452 = select i1 %.not, float %1450, float %1440, !dbg !15
  %1453 = select i1 %.not1, float %1451, float %1445, !dbg !15
  %1454 = bitcast float %1453 to i32, !dbg !15
  %1455 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1454), !dbg !15
  %1456 = bitcast i32 %1455 to float, !dbg !15
  %1457 = select i1 %.not1, float %1445, float %1456, !dbg !15
  %1458 = select i1 %.not1, float %1456, float %1451, !dbg !15
  %1459 = select i1 %.not1, float %1452, float %1446, !dbg !15
  %1460 = bitcast float %1459 to i32, !dbg !15
  %1461 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1460), !dbg !15
  %1462 = bitcast i32 %1461 to float, !dbg !15
  %1463 = select i1 %.not1, float %1446, float %1462, !dbg !15
  %1464 = select i1 %.not1, float %1462, float %1452, !dbg !15
  %1465 = shufflevector <8 x float> %1425, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1466 = fmul <4 x float> %1465, %728, !dbg !35
  %1467 = insertelement <4 x float> poison, float %1457, i64 0, !dbg !36
  %1468 = insertelement <4 x float> %1467, float %1463, i64 1, !dbg !36
  %1469 = insertelement <4 x float> %1468, float %1458, i64 2, !dbg !36
  %1470 = insertelement <4 x float> %1469, float %1464, i64 3, !dbg !36
  %1471 = fadd <4 x float> %1466, %1470, !dbg !36
  %1472 = shufflevector <4 x float> %1432, <4 x float> %1471, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1473 = insertvalue [2 x <8 x float>] %1392, <8 x float> %1472, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1474 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !15
  %1475 = bitcast <4 x i32> %1474 to <8 x half>, !dbg !15
  %1476 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !15
  %1477 = bitcast <4 x i32> %1476 to <8 x half>, !dbg !15
  %1478 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !15
  %1479 = bitcast <4 x i32> %1478 to <8 x half>, !dbg !15
  %1480 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !15
  %1481 = bitcast <4 x i32> %1480 to <8 x half>, !dbg !15
  %1482 = shufflevector <8 x half> %1475, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1482, <4 x float> zeroinitializer) #9, !dbg !15
  %1484 = shufflevector <8 x half> %1477, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1484, <4 x float> %1483) #9, !dbg !15
  %1486 = shufflevector <8 x half> %1479, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1486, <4 x float> %1485) #9, !dbg !15
  %1488 = shufflevector <8 x half> %1481, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1488, <4 x float> %1487) #9, !dbg !15
  %1490 = extractelement <4 x float> %1489, i64 0, !dbg !15
  %1491 = extractelement <4 x float> %1489, i64 1, !dbg !15
  %1492 = extractelement <4 x float> %1489, i64 2, !dbg !15
  %1493 = extractelement <4 x float> %1489, i64 3, !dbg !15
  %1494 = select i1 %.not, float %1491, float %1490, !dbg !15
  %1495 = bitcast float %1494 to i32, !dbg !15
  %1496 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1495), !dbg !15
  %1497 = bitcast i32 %1496 to float, !dbg !15
  %1498 = select i1 %.not, float %1490, float %1497, !dbg !15
  %1499 = select i1 %.not, float %1497, float %1491, !dbg !15
  %1500 = select i1 %.not, float %1493, float %1492, !dbg !15
  %1501 = bitcast float %1500 to i32, !dbg !15
  %1502 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1501), !dbg !15
  %1503 = bitcast i32 %1502 to float, !dbg !15
  %1504 = select i1 %.not, float %1492, float %1503, !dbg !15
  %1505 = select i1 %.not, float %1503, float %1493, !dbg !15
  %1506 = select i1 %.not1, float %1504, float %1498, !dbg !15
  %1507 = bitcast float %1506 to i32, !dbg !15
  %1508 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1507), !dbg !15
  %1509 = bitcast i32 %1508 to float, !dbg !15
  %1510 = select i1 %.not1, float %1498, float %1509, !dbg !15
  %1511 = select i1 %.not1, float %1509, float %1504, !dbg !15
  %1512 = select i1 %.not1, float %1505, float %1499, !dbg !15
  %1513 = bitcast float %1512 to i32, !dbg !15
  %1514 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1513), !dbg !15
  %1515 = bitcast i32 %1514 to float, !dbg !15
  %1516 = select i1 %.not1, float %1499, float %1515, !dbg !15
  %1517 = select i1 %.not1, float %1515, float %1505, !dbg !15
  %1518 = extractvalue [2 x <8 x float>] %214, 0, !dbg !35
  %1519 = shufflevector <8 x float> %1518, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1520 = fmul <4 x float> %1519, %720, !dbg !35
  %1521 = insertelement <4 x float> poison, float %1510, i64 0, !dbg !36
  %1522 = insertelement <4 x float> %1521, float %1516, i64 1, !dbg !36
  %1523 = insertelement <4 x float> %1522, float %1511, i64 2, !dbg !36
  %1524 = insertelement <4 x float> %1523, float %1517, i64 3, !dbg !36
  %1525 = fadd <4 x float> %1520, %1524, !dbg !36
  %1526 = shufflevector <8 x half> %1475, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1526, <4 x float> zeroinitializer) #9, !dbg !15
  %1528 = shufflevector <8 x half> %1477, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1528, <4 x float> %1527) #9, !dbg !15
  %1530 = shufflevector <8 x half> %1479, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1530, <4 x float> %1529) #9, !dbg !15
  %1532 = shufflevector <8 x half> %1481, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1532, <4 x float> %1531) #9, !dbg !15
  %1534 = extractelement <4 x float> %1533, i64 0, !dbg !15
  %1535 = extractelement <4 x float> %1533, i64 1, !dbg !15
  %1536 = extractelement <4 x float> %1533, i64 2, !dbg !15
  %1537 = extractelement <4 x float> %1533, i64 3, !dbg !15
  %1538 = select i1 %.not, float %1535, float %1534, !dbg !15
  %1539 = bitcast float %1538 to i32, !dbg !15
  %1540 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1539), !dbg !15
  %1541 = bitcast i32 %1540 to float, !dbg !15
  %1542 = select i1 %.not, float %1534, float %1541, !dbg !15
  %1543 = select i1 %.not, float %1541, float %1535, !dbg !15
  %1544 = select i1 %.not, float %1537, float %1536, !dbg !15
  %1545 = bitcast float %1544 to i32, !dbg !15
  %1546 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1545), !dbg !15
  %1547 = bitcast i32 %1546 to float, !dbg !15
  %1548 = select i1 %.not, float %1536, float %1547, !dbg !15
  %1549 = select i1 %.not, float %1547, float %1537, !dbg !15
  %1550 = select i1 %.not1, float %1548, float %1542, !dbg !15
  %1551 = bitcast float %1550 to i32, !dbg !15
  %1552 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1551), !dbg !15
  %1553 = bitcast i32 %1552 to float, !dbg !15
  %1554 = select i1 %.not1, float %1542, float %1553, !dbg !15
  %1555 = select i1 %.not1, float %1553, float %1548, !dbg !15
  %1556 = select i1 %.not1, float %1549, float %1543, !dbg !15
  %1557 = bitcast float %1556 to i32, !dbg !15
  %1558 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1557), !dbg !15
  %1559 = bitcast i32 %1558 to float, !dbg !15
  %1560 = select i1 %.not1, float %1543, float %1559, !dbg !15
  %1561 = select i1 %.not1, float %1559, float %1549, !dbg !15
  %1562 = shufflevector <8 x float> %1518, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1563 = fmul <4 x float> %1562, %720, !dbg !35
  %1564 = insertelement <4 x float> poison, float %1554, i64 0, !dbg !36
  %1565 = insertelement <4 x float> %1564, float %1560, i64 1, !dbg !36
  %1566 = insertelement <4 x float> %1565, float %1555, i64 2, !dbg !36
  %1567 = insertelement <4 x float> %1566, float %1561, i64 3, !dbg !36
  %1568 = fadd <4 x float> %1563, %1567, !dbg !36
  %1569 = shufflevector <4 x float> %1525, <4 x float> %1568, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1570 = insertvalue [2 x <8 x float>] poison, <8 x float> %1569, 0, !dbg !7
  %1571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1482, <4 x float> zeroinitializer) #9, !dbg !15
  %1572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1484, <4 x float> %1571) #9, !dbg !15
  %1573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1486, <4 x float> %1572) #9, !dbg !15
  %1574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1488, <4 x float> %1573) #9, !dbg !15
  %1575 = extractelement <4 x float> %1574, i64 0, !dbg !15
  %1576 = extractelement <4 x float> %1574, i64 1, !dbg !15
  %1577 = extractelement <4 x float> %1574, i64 2, !dbg !15
  %1578 = extractelement <4 x float> %1574, i64 3, !dbg !15
  %1579 = select i1 %.not, float %1576, float %1575, !dbg !15
  %1580 = bitcast float %1579 to i32, !dbg !15
  %1581 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1580), !dbg !15
  %1582 = bitcast i32 %1581 to float, !dbg !15
  %1583 = select i1 %.not, float %1575, float %1582, !dbg !15
  %1584 = select i1 %.not, float %1582, float %1576, !dbg !15
  %1585 = select i1 %.not, float %1578, float %1577, !dbg !15
  %1586 = bitcast float %1585 to i32, !dbg !15
  %1587 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1586), !dbg !15
  %1588 = bitcast i32 %1587 to float, !dbg !15
  %1589 = select i1 %.not, float %1577, float %1588, !dbg !15
  %1590 = select i1 %.not, float %1588, float %1578, !dbg !15
  %1591 = select i1 %.not1, float %1589, float %1583, !dbg !15
  %1592 = bitcast float %1591 to i32, !dbg !15
  %1593 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1592), !dbg !15
  %1594 = bitcast i32 %1593 to float, !dbg !15
  %1595 = select i1 %.not1, float %1583, float %1594, !dbg !15
  %1596 = select i1 %.not1, float %1594, float %1589, !dbg !15
  %1597 = select i1 %.not1, float %1590, float %1584, !dbg !15
  %1598 = bitcast float %1597 to i32, !dbg !15
  %1599 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1598), !dbg !15
  %1600 = bitcast i32 %1599 to float, !dbg !15
  %1601 = select i1 %.not1, float %1584, float %1600, !dbg !15
  %1602 = select i1 %.not1, float %1600, float %1590, !dbg !15
  %1603 = extractvalue [2 x <8 x float>] %214, 1, !dbg !35
  %1604 = shufflevector <8 x float> %1603, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1605 = fmul <4 x float> %1604, %728, !dbg !35
  %1606 = insertelement <4 x float> poison, float %1595, i64 0, !dbg !36
  %1607 = insertelement <4 x float> %1606, float %1601, i64 1, !dbg !36
  %1608 = insertelement <4 x float> %1607, float %1596, i64 2, !dbg !36
  %1609 = insertelement <4 x float> %1608, float %1602, i64 3, !dbg !36
  %1610 = fadd <4 x float> %1605, %1609, !dbg !36
  %1611 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1526, <4 x float> zeroinitializer) #9, !dbg !15
  %1612 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1528, <4 x float> %1611) #9, !dbg !15
  %1613 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1530, <4 x float> %1612) #9, !dbg !15
  %1614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1532, <4 x float> %1613) #9, !dbg !15
  %1615 = extractelement <4 x float> %1614, i64 0, !dbg !15
  %1616 = extractelement <4 x float> %1614, i64 1, !dbg !15
  %1617 = extractelement <4 x float> %1614, i64 2, !dbg !15
  %1618 = extractelement <4 x float> %1614, i64 3, !dbg !15
  %1619 = select i1 %.not, float %1616, float %1615, !dbg !15
  %1620 = bitcast float %1619 to i32, !dbg !15
  %1621 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1620), !dbg !15
  %1622 = bitcast i32 %1621 to float, !dbg !15
  %1623 = select i1 %.not, float %1615, float %1622, !dbg !15
  %1624 = select i1 %.not, float %1622, float %1616, !dbg !15
  %1625 = select i1 %.not, float %1618, float %1617, !dbg !15
  %1626 = bitcast float %1625 to i32, !dbg !15
  %1627 = call i32 @llvm.amdgcn.ds.bpermute(i32 %663, i32 %1626), !dbg !15
  %1628 = bitcast i32 %1627 to float, !dbg !15
  %1629 = select i1 %.not, float %1617, float %1628, !dbg !15
  %1630 = select i1 %.not, float %1628, float %1618, !dbg !15
  %1631 = select i1 %.not1, float %1629, float %1623, !dbg !15
  %1632 = bitcast float %1631 to i32, !dbg !15
  %1633 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1632), !dbg !15
  %1634 = bitcast i32 %1633 to float, !dbg !15
  %1635 = select i1 %.not1, float %1623, float %1634, !dbg !15
  %1636 = select i1 %.not1, float %1634, float %1629, !dbg !15
  %1637 = select i1 %.not1, float %1630, float %1624, !dbg !15
  %1638 = bitcast float %1637 to i32, !dbg !15
  %1639 = call i32 @llvm.amdgcn.ds.bpermute(i32 %671, i32 %1638), !dbg !15
  %1640 = bitcast i32 %1639 to float, !dbg !15
  %1641 = select i1 %.not1, float %1624, float %1640, !dbg !15
  %1642 = select i1 %.not1, float %1640, float %1630, !dbg !15
  %1643 = shufflevector <8 x float> %1603, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1644 = fmul <4 x float> %1643, %728, !dbg !35
  %1645 = insertelement <4 x float> poison, float %1635, i64 0, !dbg !36
  %1646 = insertelement <4 x float> %1645, float %1641, i64 1, !dbg !36
  %1647 = insertelement <4 x float> %1646, float %1636, i64 2, !dbg !36
  %1648 = insertelement <4 x float> %1647, float %1642, i64 3, !dbg !36
  %1649 = fadd <4 x float> %1644, %1648, !dbg !36
  %1650 = shufflevector <4 x float> %1610, <4 x float> %1649, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1651 = insertvalue [2 x <8 x float>] %1570, <8 x float> %1650, 1, !dbg !7
  %1652 = insertvalue [2 x <1 x float>] poison, <1 x float> %678, 0, !dbg !7
  %1653 = insertvalue [2 x <1 x float>] %1652, <1 x float> %705, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1654 = add nuw nsw i64 %208, 1, !dbg !7
  br label %207, !dbg !7

1655:                                             ; preds = %207
  %1656 = extractvalue [2 x <1 x float>] %210, 0, !dbg !19
  %1657 = shufflevector <1 x float> %1656, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1658 = extractvalue [2 x <1 x float>] %210, 1, !dbg !19
  %1659 = shufflevector <1 x float> %1658, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  br i1 true, label %1660, label %1690, !dbg !21

1660:                                             ; preds = %1655
  %1661 = extractvalue [2 x <8 x float>] %209, 1, !dbg !19
  %1662 = shufflevector <8 x float> %1661, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1659, !dbg !19
  %normalized = fmul <4 x float> %1662, %row.reciprocal, !dbg !19
  %1663 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1664 = extractelement <4 x half> %1663, i64 3, !dbg !20
  %1665 = extractelement <4 x half> %1663, i64 2, !dbg !20
  %1666 = extractelement <4 x half> %1663, i64 1, !dbg !20
  %1667 = extractelement <4 x half> %1663, i64 0, !dbg !20
  %1668 = shufflevector <8 x float> %1661, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized54 = fmul <4 x float> %1668, %row.reciprocal, !dbg !19
  %1669 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !20
  %1670 = extractelement <4 x half> %1669, i64 3, !dbg !20
  %1671 = extractelement <4 x half> %1669, i64 2, !dbg !20
  %1672 = extractelement <4 x half> %1669, i64 1, !dbg !20
  %1673 = extractelement <4 x half> %1669, i64 0, !dbg !20
  %1674 = extractvalue [2 x <8 x float>] %209, 0, !dbg !19
  %1675 = shufflevector <8 x float> %1674, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal55 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1657, !dbg !19
  %normalized56 = fmul <4 x float> %1675, %row.reciprocal55, !dbg !19
  %1676 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !20
  %1677 = extractelement <4 x half> %1676, i64 3, !dbg !20
  %1678 = extractelement <4 x half> %1676, i64 2, !dbg !20
  %1679 = extractelement <4 x half> %1676, i64 1, !dbg !20
  %1680 = extractelement <4 x half> %1676, i64 0, !dbg !20
  %1681 = shufflevector <8 x float> %1674, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized57 = fmul <4 x float> %1681, %row.reciprocal55, !dbg !19
  %1682 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !20
  %1683 = extractelement <4 x half> %1682, i64 3, !dbg !20
  %1684 = extractelement <4 x half> %1682, i64 2, !dbg !20
  %1685 = extractelement <4 x half> %1682, i64 1, !dbg !20
  %1686 = extractelement <4 x half> %1682, i64 0, !dbg !20
  %1687 = add nsw i64 %16, %22, !dbg !21
  %1688 = shl i64 %1687, 19, !dbg !21
  %1689 = and i64 %1688, 16252928, !dbg !21
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
  %address.local544 = add nuw nsw i64 %1689, %address.local543, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local546 = lshr i64 %address.local535, 2, !dbg !8
  %address.local547 = and i64 %address.local546, 12, !dbg !8
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local547, !dbg !21
  store half %1686, ptr addrspace(1) %address.local548, align 2, !dbg !21
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local550 = or disjoint i64 %address.local547, 1, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local549, i64 %address.local550, !dbg !21
  store half %1685, ptr addrspace(1) %address.local551, align 2, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local553 = or disjoint i64 %address.local547, 2, !dbg !21
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local552, i64 %address.local553, !dbg !21
  store half %1684, ptr addrspace(1) %address.local554, align 2, !dbg !21
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local556 = or disjoint i64 %address.local547, 3, !dbg !21
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local555, i64 %address.local556, !dbg !21
  store half %1683, ptr addrspace(1) %address.local557, align 2, !dbg !21
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local559 = or disjoint i64 %address.local547, 16, !dbg !21
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local558, i64 %address.local559, !dbg !21
  store half %1680, ptr addrspace(1) %address.local560, align 2, !dbg !21
  %address.local561 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local562 = or disjoint i64 %address.local547, 17, !dbg !21
  %address.local563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local561, i64 %address.local562, !dbg !21
  store half %1679, ptr addrspace(1) %address.local563, align 2, !dbg !21
  %address.local564 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local565 = or disjoint i64 %address.local547, 18, !dbg !21
  %address.local566 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local564, i64 %address.local565, !dbg !21
  store half %1678, ptr addrspace(1) %address.local566, align 2, !dbg !21
  %address.local567 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local568 = or disjoint i64 %address.local547, 19, !dbg !21
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local567, i64 %address.local568, !dbg !21
  store half %1677, ptr addrspace(1) %address.local569, align 2, !dbg !21
  %address.local570 = shl i64 %address.local542, 7, !dbg !21
  %address.local571 = add i64 %address.local570, 8192, !dbg !21
  %address.local572 = add nuw nsw i64 %1689, %address.local571, !dbg !21
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local573, i64 %address.local547, !dbg !21
  store half %1673, ptr addrspace(1) %address.local574, align 2, !dbg !21
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local550, !dbg !21
  store half %1672, ptr addrspace(1) %address.local576, align 2, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local553, !dbg !21
  store half %1671, ptr addrspace(1) %address.local578, align 2, !dbg !21
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local579, i64 %address.local556, !dbg !21
  store half %1670, ptr addrspace(1) %address.local580, align 2, !dbg !21
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local559, !dbg !21
  store half %1667, ptr addrspace(1) %address.local582, align 2, !dbg !21
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local562, !dbg !21
  store half %1666, ptr addrspace(1) %address.local584, align 2, !dbg !21
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local585, i64 %address.local565, !dbg !21
  store half %1665, ptr addrspace(1) %address.local586, align 2, !dbg !21
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local587, i64 %address.local568, !dbg !21
  store half %1664, ptr addrspace(1) %address.local588, align 2, !dbg !21
  br label %1690, !dbg !21

1690:                                             ; preds = %1660, %1655
  br i1 true, label %1691, label %1719, !dbg !21

1691:                                             ; preds = %1690
  %1692 = extractvalue [2 x <8 x float>] %212, 1, !dbg !19
  %1693 = shufflevector <8 x float> %1692, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal58 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1659, !dbg !19
  %normalized59 = fmul <4 x float> %1693, %row.reciprocal58, !dbg !19
  %1694 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !20
  %1695 = extractelement <4 x half> %1694, i64 3, !dbg !20
  %1696 = extractelement <4 x half> %1694, i64 2, !dbg !20
  %1697 = extractelement <4 x half> %1694, i64 1, !dbg !20
  %1698 = extractelement <4 x half> %1694, i64 0, !dbg !20
  %1699 = shufflevector <8 x float> %1692, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized60 = fmul <4 x float> %1699, %row.reciprocal58, !dbg !19
  %1700 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !20
  %1701 = extractelement <4 x half> %1700, i64 3, !dbg !20
  %1702 = extractelement <4 x half> %1700, i64 2, !dbg !20
  %1703 = extractelement <4 x half> %1700, i64 1, !dbg !20
  %1704 = extractelement <4 x half> %1700, i64 0, !dbg !20
  %1705 = extractvalue [2 x <8 x float>] %212, 0, !dbg !19
  %1706 = shufflevector <8 x float> %1705, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal61 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1657, !dbg !19
  %normalized62 = fmul <4 x float> %1706, %row.reciprocal61, !dbg !19
  %1707 = fptrunc <4 x float> %normalized62 to <4 x half>, !dbg !20
  %1708 = extractelement <4 x half> %1707, i64 3, !dbg !20
  %1709 = extractelement <4 x half> %1707, i64 2, !dbg !20
  %1710 = extractelement <4 x half> %1707, i64 1, !dbg !20
  %1711 = extractelement <4 x half> %1707, i64 0, !dbg !20
  %1712 = shufflevector <8 x float> %1705, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized63 = fmul <4 x float> %1712, %row.reciprocal61, !dbg !19
  %1713 = fptrunc <4 x float> %normalized63 to <4 x half>, !dbg !20
  %1714 = extractelement <4 x half> %1713, i64 3, !dbg !20
  %1715 = extractelement <4 x half> %1713, i64 2, !dbg !20
  %1716 = extractelement <4 x half> %1713, i64 1, !dbg !20
  %1717 = extractelement <4 x half> %1713, i64 0, !dbg !20
  %1718 = shl nuw nsw i64 %24, 19, !dbg !21
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
  %address.local599 = add nuw nsw i64 %1718, %address.local598, !dbg !21
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local601 = lshr i64 %address.local590, 2, !dbg !8
  %address.local602 = and i64 %address.local601, 12, !dbg !8
  %address.local603 = or disjoint i64 %address.local602, 32, !dbg !21
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local603, !dbg !21
  store half %1717, ptr addrspace(1) %address.local604, align 2, !dbg !21
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local606 = or disjoint i64 %address.local602, 33, !dbg !21
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local605, i64 %address.local606, !dbg !21
  store half %1716, ptr addrspace(1) %address.local607, align 2, !dbg !21
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local609 = or disjoint i64 %address.local602, 34, !dbg !21
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local608, i64 %address.local609, !dbg !21
  store half %1715, ptr addrspace(1) %address.local610, align 2, !dbg !21
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local612 = or disjoint i64 %address.local602, 35, !dbg !21
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local611, i64 %address.local612, !dbg !21
  store half %1714, ptr addrspace(1) %address.local613, align 2, !dbg !21
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local615 = or disjoint i64 %address.local602, 48, !dbg !21
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local614, i64 %address.local615, !dbg !21
  store half %1711, ptr addrspace(1) %address.local616, align 2, !dbg !21
  %address.local617 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local618 = or disjoint i64 %address.local602, 49, !dbg !21
  %address.local619 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local617, i64 %address.local618, !dbg !21
  store half %1710, ptr addrspace(1) %address.local619, align 2, !dbg !21
  %address.local620 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local621 = or disjoint i64 %address.local602, 50, !dbg !21
  %address.local622 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local621, !dbg !21
  store half %1709, ptr addrspace(1) %address.local622, align 2, !dbg !21
  %address.local623 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local624 = or i64 %address.local601, 51, !dbg !21
  %address.local625 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local623, i64 %address.local624, !dbg !21
  store half %1708, ptr addrspace(1) %address.local625, align 2, !dbg !21
  %address.local626 = shl i64 %address.local597, 7, !dbg !21
  %address.local627 = add i64 %address.local626, 8192, !dbg !21
  %address.local628 = add nuw nsw i64 %1718, %address.local627, !dbg !21
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local629, i64 %address.local603, !dbg !21
  store half %1704, ptr addrspace(1) %address.local630, align 2, !dbg !21
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local606, !dbg !21
  store half %1703, ptr addrspace(1) %address.local632, align 2, !dbg !21
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local633, i64 %address.local609, !dbg !21
  store half %1702, ptr addrspace(1) %address.local634, align 2, !dbg !21
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local635, i64 %address.local612, !dbg !21
  store half %1701, ptr addrspace(1) %address.local636, align 2, !dbg !21
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local637, i64 %address.local615, !dbg !21
  store half %1698, ptr addrspace(1) %address.local638, align 2, !dbg !21
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local639, i64 %address.local618, !dbg !21
  store half %1697, ptr addrspace(1) %address.local640, align 2, !dbg !21
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local642 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local641, i64 %address.local621, !dbg !21
  store half %1696, ptr addrspace(1) %address.local642, align 2, !dbg !21
  %address.local643 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local624, !dbg !21
  store half %1695, ptr addrspace(1) %address.local644, align 2, !dbg !21
  br label %1719, !dbg !21

1719:                                             ; preds = %1691, %1690
  br i1 true, label %1720, label %1748, !dbg !21

1720:                                             ; preds = %1719
  %1721 = extractvalue [2 x <8 x float>] %213, 1, !dbg !19
  %1722 = shufflevector <8 x float> %1721, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal64 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1659, !dbg !19
  %normalized65 = fmul <4 x float> %1722, %row.reciprocal64, !dbg !19
  %1723 = fptrunc <4 x float> %normalized65 to <4 x half>, !dbg !20
  %1724 = extractelement <4 x half> %1723, i64 3, !dbg !20
  %1725 = extractelement <4 x half> %1723, i64 2, !dbg !20
  %1726 = extractelement <4 x half> %1723, i64 1, !dbg !20
  %1727 = extractelement <4 x half> %1723, i64 0, !dbg !20
  %1728 = shufflevector <8 x float> %1721, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized66 = fmul <4 x float> %1728, %row.reciprocal64, !dbg !19
  %1729 = fptrunc <4 x float> %normalized66 to <4 x half>, !dbg !20
  %1730 = extractelement <4 x half> %1729, i64 3, !dbg !20
  %1731 = extractelement <4 x half> %1729, i64 2, !dbg !20
  %1732 = extractelement <4 x half> %1729, i64 1, !dbg !20
  %1733 = extractelement <4 x half> %1729, i64 0, !dbg !20
  %1734 = extractvalue [2 x <8 x float>] %213, 0, !dbg !19
  %1735 = shufflevector <8 x float> %1734, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal67 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1657, !dbg !19
  %normalized68 = fmul <4 x float> %1735, %row.reciprocal67, !dbg !19
  %1736 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !20
  %1737 = extractelement <4 x half> %1736, i64 3, !dbg !20
  %1738 = extractelement <4 x half> %1736, i64 2, !dbg !20
  %1739 = extractelement <4 x half> %1736, i64 1, !dbg !20
  %1740 = extractelement <4 x half> %1736, i64 0, !dbg !20
  %1741 = shufflevector <8 x float> %1734, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized69 = fmul <4 x float> %1741, %row.reciprocal67, !dbg !19
  %1742 = fptrunc <4 x float> %normalized69 to <4 x half>, !dbg !20
  %1743 = extractelement <4 x half> %1742, i64 3, !dbg !20
  %1744 = extractelement <4 x half> %1742, i64 2, !dbg !20
  %1745 = extractelement <4 x half> %1742, i64 1, !dbg !20
  %1746 = extractelement <4 x half> %1742, i64 0, !dbg !20
  %1747 = shl nuw nsw i64 %24, 19, !dbg !21
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
  %address.local655 = add nuw nsw i64 %1747, %address.local654, !dbg !21
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local657 = lshr i64 %address.local646, 2, !dbg !8
  %address.local658 = and i64 %address.local657, 12, !dbg !8
  %address.local659 = or disjoint i64 %address.local658, 64, !dbg !21
  %address.local660 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local656, i64 %address.local659, !dbg !21
  store half %1746, ptr addrspace(1) %address.local660, align 2, !dbg !21
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local662 = or disjoint i64 %address.local658, 65, !dbg !21
  %address.local663 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local661, i64 %address.local662, !dbg !21
  store half %1745, ptr addrspace(1) %address.local663, align 2, !dbg !21
  %address.local664 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local665 = or disjoint i64 %address.local658, 66, !dbg !21
  %address.local666 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local664, i64 %address.local665, !dbg !21
  store half %1744, ptr addrspace(1) %address.local666, align 2, !dbg !21
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local668 = or disjoint i64 %address.local658, 67, !dbg !21
  %address.local669 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local667, i64 %address.local668, !dbg !21
  store half %1743, ptr addrspace(1) %address.local669, align 2, !dbg !21
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local671 = or disjoint i64 %address.local658, 80, !dbg !21
  %address.local672 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local670, i64 %address.local671, !dbg !21
  store half %1740, ptr addrspace(1) %address.local672, align 2, !dbg !21
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local674 = or disjoint i64 %address.local658, 81, !dbg !21
  %address.local675 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local673, i64 %address.local674, !dbg !21
  store half %1739, ptr addrspace(1) %address.local675, align 2, !dbg !21
  %address.local676 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local677 = or disjoint i64 %address.local658, 82, !dbg !21
  %address.local678 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local676, i64 %address.local677, !dbg !21
  store half %1738, ptr addrspace(1) %address.local678, align 2, !dbg !21
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local680 = or disjoint i64 %address.local658, 83, !dbg !21
  %address.local681 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local679, i64 %address.local680, !dbg !21
  store half %1737, ptr addrspace(1) %address.local681, align 2, !dbg !21
  %address.local682 = shl i64 %address.local653, 7, !dbg !21
  %address.local683 = add i64 %address.local682, 8192, !dbg !21
  %address.local684 = add nuw nsw i64 %1747, %address.local683, !dbg !21
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local685, i64 %address.local659, !dbg !21
  store half %1733, ptr addrspace(1) %address.local686, align 2, !dbg !21
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local687, i64 %address.local662, !dbg !21
  store half %1732, ptr addrspace(1) %address.local688, align 2, !dbg !21
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local689, i64 %address.local665, !dbg !21
  store half %1731, ptr addrspace(1) %address.local690, align 2, !dbg !21
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local691, i64 %address.local668, !dbg !21
  store half %1730, ptr addrspace(1) %address.local692, align 2, !dbg !21
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local693, i64 %address.local671, !dbg !21
  store half %1727, ptr addrspace(1) %address.local694, align 2, !dbg !21
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local695, i64 %address.local674, !dbg !21
  store half %1726, ptr addrspace(1) %address.local696, align 2, !dbg !21
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local697, i64 %address.local677, !dbg !21
  store half %1725, ptr addrspace(1) %address.local698, align 2, !dbg !21
  %address.local699 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local699, i64 %address.local680, !dbg !21
  store half %1724, ptr addrspace(1) %address.local700, align 2, !dbg !21
  br label %1748, !dbg !21

1748:                                             ; preds = %1720, %1719
  br i1 true, label %1749, label %1777, !dbg !21

1749:                                             ; preds = %1748
  %1750 = extractvalue [2 x <8 x float>] %214, 1, !dbg !19
  %1751 = shufflevector <8 x float> %1750, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal70 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1659, !dbg !19
  %normalized71 = fmul <4 x float> %1751, %row.reciprocal70, !dbg !19
  %1752 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !20
  %1753 = extractelement <4 x half> %1752, i64 3, !dbg !20
  %1754 = extractelement <4 x half> %1752, i64 2, !dbg !20
  %1755 = extractelement <4 x half> %1752, i64 1, !dbg !20
  %1756 = extractelement <4 x half> %1752, i64 0, !dbg !20
  %1757 = shufflevector <8 x float> %1750, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized72 = fmul <4 x float> %1757, %row.reciprocal70, !dbg !19
  %1758 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !20
  %1759 = extractelement <4 x half> %1758, i64 3, !dbg !20
  %1760 = extractelement <4 x half> %1758, i64 2, !dbg !20
  %1761 = extractelement <4 x half> %1758, i64 1, !dbg !20
  %1762 = extractelement <4 x half> %1758, i64 0, !dbg !20
  %1763 = extractvalue [2 x <8 x float>] %214, 0, !dbg !19
  %1764 = shufflevector <8 x float> %1763, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1657, !dbg !19
  %normalized74 = fmul <4 x float> %1764, %row.reciprocal73, !dbg !19
  %1765 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %1766 = extractelement <4 x half> %1765, i64 3, !dbg !20
  %1767 = extractelement <4 x half> %1765, i64 2, !dbg !20
  %1768 = extractelement <4 x half> %1765, i64 1, !dbg !20
  %1769 = extractelement <4 x half> %1765, i64 0, !dbg !20
  %1770 = shufflevector <8 x float> %1763, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized75 = fmul <4 x float> %1770, %row.reciprocal73, !dbg !19
  %1771 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !20
  %1772 = extractelement <4 x half> %1771, i64 3, !dbg !20
  %1773 = extractelement <4 x half> %1771, i64 2, !dbg !20
  %1774 = extractelement <4 x half> %1771, i64 1, !dbg !20
  %1775 = extractelement <4 x half> %1771, i64 0, !dbg !20
  %1776 = shl nuw nsw i64 %24, 19, !dbg !21
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
  %address.local711 = add nuw nsw i64 %1776, %address.local710, !dbg !21
  %address.local712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local713 = lshr i64 %address.local702, 2, !dbg !8
  %address.local714 = and i64 %address.local713, 12, !dbg !8
  %address.local715 = or disjoint i64 %address.local714, 96, !dbg !21
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local712, i64 %address.local715, !dbg !21
  store half %1775, ptr addrspace(1) %address.local716, align 2, !dbg !21
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local718 = or disjoint i64 %address.local714, 97, !dbg !21
  %address.local719 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local717, i64 %address.local718, !dbg !21
  store half %1774, ptr addrspace(1) %address.local719, align 2, !dbg !21
  %address.local720 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local721 = or disjoint i64 %address.local714, 98, !dbg !21
  %address.local722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local720, i64 %address.local721, !dbg !21
  store half %1773, ptr addrspace(1) %address.local722, align 2, !dbg !21
  %address.local723 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local724 = or disjoint i64 %address.local714, 99, !dbg !21
  %address.local725 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local723, i64 %address.local724, !dbg !21
  store half %1772, ptr addrspace(1) %address.local725, align 2, !dbg !21
  %address.local726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local727 = or disjoint i64 %address.local714, 112, !dbg !21
  %address.local728 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local726, i64 %address.local727, !dbg !21
  store half %1769, ptr addrspace(1) %address.local728, align 2, !dbg !21
  %address.local729 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local730 = or disjoint i64 %address.local714, 113, !dbg !21
  %address.local731 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local729, i64 %address.local730, !dbg !21
  store half %1768, ptr addrspace(1) %address.local731, align 2, !dbg !21
  %address.local732 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local733 = or disjoint i64 %address.local714, 114, !dbg !21
  %address.local734 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local732, i64 %address.local733, !dbg !21
  store half %1767, ptr addrspace(1) %address.local734, align 2, !dbg !21
  %address.local735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local736 = or i64 %address.local713, 115, !dbg !21
  %address.local737 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local735, i64 %address.local736, !dbg !21
  store half %1766, ptr addrspace(1) %address.local737, align 2, !dbg !21
  %address.local738 = shl i64 %address.local709, 7, !dbg !21
  %address.local739 = add i64 %address.local738, 8192, !dbg !21
  %address.local740 = add nuw nsw i64 %1776, %address.local739, !dbg !21
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local741, i64 %address.local715, !dbg !21
  store half %1762, ptr addrspace(1) %address.local742, align 2, !dbg !21
  %address.local743 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local743, i64 %address.local718, !dbg !21
  store half %1761, ptr addrspace(1) %address.local744, align 2, !dbg !21
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local746 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local745, i64 %address.local721, !dbg !21
  store half %1760, ptr addrspace(1) %address.local746, align 2, !dbg !21
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local747, i64 %address.local724, !dbg !21
  store half %1759, ptr addrspace(1) %address.local748, align 2, !dbg !21
  %address.local749 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local749, i64 %address.local727, !dbg !21
  store half %1756, ptr addrspace(1) %address.local750, align 2, !dbg !21
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local751, i64 %address.local730, !dbg !21
  store half %1755, ptr addrspace(1) %address.local752, align 2, !dbg !21
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local754 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local753, i64 %address.local733, !dbg !21
  store half %1754, ptr addrspace(1) %address.local754, align 2, !dbg !21
  %address.local755 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local755, i64 %address.local736, !dbg !21
  store half %1753, ptr addrspace(1) %address.local756, align 2, !dbg !21
  br label %1777, !dbg !21

1777:                                             ; preds = %1749, %1748
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax/final/run-20261023-171702-30cjis9r/retained32-baseline")
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
