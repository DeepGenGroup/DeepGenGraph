; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0002-test_input_1x32x4096x128-128-64-32-online/retained32-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0002-test_input_1x32x4096x128-128-64-32-online/retained32-baseline/input.mlir"

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
  %206 = phi i64 [ %1124, %587 ], [ 0, %4 ], !dbg !7
  %207 = phi [2 x <8 x float>] [ %1109, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %208 = phi [2 x <1 x float>] [ %885, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %209 = phi [2 x <1 x float>] [ %1111, %587 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %210 = phi [2 x <8 x float>] [ %1115, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %211 = phi [2 x <8 x float>] [ %1119, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %212 = phi [2 x <8 x float>] [ %1123, %587 ], [ zeroinitializer, %4 ], !dbg !7
  %213 = icmp slt i64 %206, %192, !dbg !7
  br i1 %213, label %214, label %1125, !dbg !7

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
  %915 = extractvalue [2 x <8 x float>] %207, 0, !dbg !35
  %916 = shufflevector <8 x float> %915, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %917 = fmul <4 x float> %916, %718, !dbg !35
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %914, <4 x float> %917) #9, !dbg !15
  %919 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %919, <4 x float> %918) #9, !dbg !15
  %921 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %921, <4 x float> %920) #9, !dbg !15
  %923 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %923, <4 x float> %922) #9, !dbg !15
  %925 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %926 = shufflevector <8 x float> %915, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %927 = fmul <4 x float> %926, %718, !dbg !35
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %925, <4 x float> %927) #9, !dbg !15
  %929 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %929, <4 x float> %928) #9, !dbg !15
  %931 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %931, <4 x float> %930) #9, !dbg !15
  %933 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %933, <4 x float> %932) #9, !dbg !15
  %935 = extractvalue [2 x <8 x float>] %207, 1, !dbg !35
  %936 = shufflevector <8 x float> %935, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %937 = fmul <4 x float> %936, %726, !dbg !35
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %914, <4 x float> %937) #9, !dbg !15
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %919, <4 x float> %938) #9, !dbg !15
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %921, <4 x float> %939) #9, !dbg !15
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %923, <4 x float> %940) #9, !dbg !15
  %942 = shufflevector <8 x float> %935, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %943 = fmul <4 x float> %942, %726, !dbg !35
  %944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %925, <4 x float> %943) #9, !dbg !15
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %929, <4 x float> %944) #9, !dbg !15
  %946 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %931, <4 x float> %945) #9, !dbg !15
  %947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %933, <4 x float> %946) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %948 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !15
  %949 = bitcast <4 x i32> %948 to <8 x half>, !dbg !15
  %950 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !15
  %951 = bitcast <4 x i32> %950 to <8 x half>, !dbg !15
  %952 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !15
  %953 = bitcast <4 x i32> %952 to <8 x half>, !dbg !15
  %954 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !15
  %955 = bitcast <4 x i32> %954 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %956 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %897, !dbg !18
  %957 = getelementptr i8, ptr addrspace(1) %956, i64 128, !dbg !18
  %958 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !18
  %959 = zext i32 %958 to i64, !dbg !18
  %960 = inttoptr i64 %959 to ptr addrspace(3), !dbg !18
  %address.local532 = getelementptr [2 x i8], ptr addrspace(1) %957, i64 %address.local530, !dbg !18
  %961 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %962 = ptrtoint ptr addrspace(1) %address.local532 to i64, !dbg !18
  %963 = sub i64 %962, %961, !dbg !18
  %dma.buffer.offset954 = trunc i64 %963 to i32, !dbg !18
  %964 = trunc i64 %961 to i32, !dbg !18
  %965 = insertelement <4 x i32> poison, i32 %964, i64 0, !dbg !18
  %966 = lshr i64 %961, 32, !dbg !18
  %967 = trunc i64 %966 to i32, !dbg !18
  %968 = insertelement <4 x i32> %965, i32 %967, i64 1, !dbg !18
  %969 = insertelement <4 x i32> %968, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc955 = insertelement <4 x i32> %969, i32 131072, i64 3, !dbg !18
  %970 = ptrtoint ptr addrspace(3) %960 to i32, !dbg !18
  %dma.buffer.lds956 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %970), !dbg !18
  %971 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds956, i32 %dma.buffer.offset954, <4 x i32> %dma.buffer.rsrc955) #10, !dbg !18
  %972 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %973 = extractvalue [2 x <8 x float>] %210, 0, !dbg !35
  %974 = shufflevector <8 x float> %973, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %975 = fmul <4 x float> %974, %718, !dbg !35
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %972, <4 x float> %975) #9, !dbg !15
  %977 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %977, <4 x float> %976) #9, !dbg !15
  %979 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %979, <4 x float> %978) #9, !dbg !15
  %981 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %981, <4 x float> %980) #9, !dbg !15
  %983 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %984 = shufflevector <8 x float> %973, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %985 = fmul <4 x float> %984, %718, !dbg !35
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %983, <4 x float> %985) #9, !dbg !15
  %987 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %987, <4 x float> %986) #9, !dbg !15
  %989 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %989, <4 x float> %988) #9, !dbg !15
  %991 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %991, <4 x float> %990) #9, !dbg !15
  %993 = extractvalue [2 x <8 x float>] %210, 1, !dbg !35
  %994 = shufflevector <8 x float> %993, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %995 = fmul <4 x float> %994, %726, !dbg !35
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %972, <4 x float> %995) #9, !dbg !15
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %977, <4 x float> %996) #9, !dbg !15
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %979, <4 x float> %997) #9, !dbg !15
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %981, <4 x float> %998) #9, !dbg !15
  %1000 = shufflevector <8 x float> %993, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1001 = fmul <4 x float> %1000, %726, !dbg !35
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %983, <4 x float> %1001) #9, !dbg !15
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %987, <4 x float> %1002) #9, !dbg !15
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %989, <4 x float> %1003) #9, !dbg !15
  %1005 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %991, <4 x float> %1004) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1006 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !15
  %1007 = bitcast <4 x i32> %1006 to <8 x half>, !dbg !15
  %1008 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !15
  %1009 = bitcast <4 x i32> %1008 to <8 x half>, !dbg !15
  %1010 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !15
  %1011 = bitcast <4 x i32> %1010 to <8 x half>, !dbg !15
  %1012 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !15
  %1013 = bitcast <4 x i32> %1012 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1014 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %897, !dbg !18
  %1015 = getelementptr i8, ptr addrspace(1) %1014, i64 192, !dbg !18
  %1016 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !18
  %1017 = zext i32 %1016 to i64, !dbg !18
  %1018 = inttoptr i64 %1017 to ptr addrspace(3), !dbg !18
  %address.local533 = getelementptr [2 x i8], ptr addrspace(1) %1015, i64 %address.local530, !dbg !18
  %1019 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1020 = ptrtoint ptr addrspace(1) %address.local533 to i64, !dbg !18
  %1021 = sub i64 %1020, %1019, !dbg !18
  %dma.buffer.offset = trunc i64 %1021 to i32, !dbg !18
  %1022 = trunc i64 %1019 to i32, !dbg !18
  %1023 = insertelement <4 x i32> poison, i32 %1022, i64 0, !dbg !18
  %1024 = lshr i64 %1019, 32, !dbg !18
  %1025 = trunc i64 %1024 to i32, !dbg !18
  %1026 = insertelement <4 x i32> %1023, i32 %1025, i64 1, !dbg !18
  %1027 = insertelement <4 x i32> %1026, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1027, i32 131072, i64 3, !dbg !18
  %1028 = ptrtoint ptr addrspace(3) %1018 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1028), !dbg !18
  %1029 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1030 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1031 = extractvalue [2 x <8 x float>] %211, 0, !dbg !35
  %1032 = shufflevector <8 x float> %1031, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1033 = fmul <4 x float> %1032, %718, !dbg !35
  %1034 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1030, <4 x float> %1033) #9, !dbg !15
  %1035 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1035, <4 x float> %1034) #9, !dbg !15
  %1037 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1037, <4 x float> %1036) #9, !dbg !15
  %1039 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1039, <4 x float> %1038) #9, !dbg !15
  %1041 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1042 = shufflevector <8 x float> %1031, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1043 = fmul <4 x float> %1042, %718, !dbg !35
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1041, <4 x float> %1043) #9, !dbg !15
  %1045 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1045, <4 x float> %1044) #9, !dbg !15
  %1047 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1048 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1047, <4 x float> %1046) #9, !dbg !15
  %1049 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1049, <4 x float> %1048) #9, !dbg !15
  %1051 = extractvalue [2 x <8 x float>] %211, 1, !dbg !35
  %1052 = shufflevector <8 x float> %1051, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1053 = fmul <4 x float> %1052, %726, !dbg !35
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1030, <4 x float> %1053) #9, !dbg !15
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1035, <4 x float> %1054) #9, !dbg !15
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1037, <4 x float> %1055) #9, !dbg !15
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1039, <4 x float> %1056) #9, !dbg !15
  %1058 = shufflevector <8 x float> %1051, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1059 = fmul <4 x float> %1058, %726, !dbg !35
  %1060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1041, <4 x float> %1059) #9, !dbg !15
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1045, <4 x float> %1060) #9, !dbg !15
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1047, <4 x float> %1061) #9, !dbg !15
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1049, <4 x float> %1062) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1064 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !15
  %1065 = bitcast <4 x i32> %1064 to <8 x half>, !dbg !15
  %1066 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !15
  %1067 = bitcast <4 x i32> %1066 to <8 x half>, !dbg !15
  %1068 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !15
  %1069 = bitcast <4 x i32> %1068 to <8 x half>, !dbg !15
  %1070 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !15
  %1071 = bitcast <4 x i32> %1070 to <8 x half>, !dbg !15
  %1072 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1073 = extractvalue [2 x <8 x float>] %212, 0, !dbg !35
  %1074 = shufflevector <8 x float> %1073, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1075 = fmul <4 x float> %1074, %718, !dbg !35
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1072, <4 x float> %1075) #9, !dbg !15
  %1077 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1077, <4 x float> %1076) #9, !dbg !15
  %1079 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1079, <4 x float> %1078) #9, !dbg !15
  %1081 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1081, <4 x float> %1080) #9, !dbg !15
  %1083 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1084 = shufflevector <8 x float> %1073, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1085 = fmul <4 x float> %1084, %718, !dbg !35
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1083, <4 x float> %1085) #9, !dbg !15
  %1087 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1087, <4 x float> %1086) #9, !dbg !15
  %1089 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1089, <4 x float> %1088) #9, !dbg !15
  %1091 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1091, <4 x float> %1090) #9, !dbg !15
  %1093 = extractvalue [2 x <8 x float>] %212, 1, !dbg !35
  %1094 = shufflevector <8 x float> %1093, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1095 = fmul <4 x float> %1094, %726, !dbg !35
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1072, <4 x float> %1095) #9, !dbg !15
  %1097 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1077, <4 x float> %1096) #9, !dbg !15
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1079, <4 x float> %1097) #9, !dbg !15
  %1099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1081, <4 x float> %1098) #9, !dbg !15
  %1100 = shufflevector <8 x float> %1093, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1101 = fmul <4 x float> %1100, %726, !dbg !35
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1083, <4 x float> %1101) #9, !dbg !15
  %1103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1087, <4 x float> %1102) #9, !dbg !15
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1089, <4 x float> %1103) #9, !dbg !15
  %1105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1091, <4 x float> %1104) #9, !dbg !15
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
  %1124 = add nuw nsw i64 %206, 1, !dbg !7
  br label %205, !dbg !7

1125:                                             ; preds = %205
  %1126 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %1126, 0, !dbg !7
  %1127 = and i32 %31, 32, !dbg !7
  %.not18 = icmp eq i32 %1127, 0, !dbg !7
  %1128 = extractvalue [2 x <8 x float>] %212, 0, !dbg !7
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
  %1169 = extractvalue [2 x <8 x float>] %211, 0, !dbg !7
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
  %1198 = extractvalue [2 x <8 x float>] %210, 0, !dbg !7
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
  %1227 = extractvalue [2 x <8 x float>] %207, 0, !dbg !7
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
  %1250 = extractvalue [2 x <1 x float>] %208, 0, !dbg !19
  %1251 = shufflevector <1 x float> %1250, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
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
  %1358 = extractvalue [2 x <8 x float>] %212, 1, !dbg !7
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
  %1387 = extractvalue [2 x <8 x float>] %211, 1, !dbg !7
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
  %1416 = extractvalue [2 x <8 x float>] %210, 1, !dbg !7
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
  %1445 = extractvalue [2 x <8 x float>] %207, 1, !dbg !7
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
  %1468 = extractvalue [2 x <1 x float>] %208, 1, !dbg !19
  %1469 = shufflevector <1 x float> %1468, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
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
  br i1 true, label %1576, label %1640, !dbg !21

1576:                                             ; preds = %1125
  %1577 = bitcast i32 %1572 to float, !dbg !7
  %1578 = select i1 %.not18, float %1562, float %1577, !dbg !7
  %1579 = insertelement <4 x float> poison, float %1578, i64 0, !dbg !19
  %1580 = bitcast i32 %1575 to float, !dbg !7
  %1581 = select i1 %.not18, float %1563, float %1580, !dbg !7
  %1582 = insertelement <4 x float> %1579, float %1581, i64 1, !dbg !19
  %1583 = select i1 %.not18, float %1577, float %1568, !dbg !7
  %1584 = insertelement <4 x float> %1582, float %1583, i64 2, !dbg !19
  %1585 = select i1 %.not18, float %1580, float %1569, !dbg !7
  %1586 = insertelement <4 x float> %1584, float %1585, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1469, !dbg !19
  %normalized = fmul <4 x float> %1586, %row.reciprocal, !dbg !19
  %1587 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1588 = extractelement <4 x half> %1587, i64 3, !dbg !20
  %1589 = extractelement <4 x half> %1587, i64 2, !dbg !20
  %1590 = extractelement <4 x half> %1587, i64 1, !dbg !20
  %1591 = extractelement <4 x half> %1587, i64 0, !dbg !20
  %1592 = bitcast i32 %1464 to float, !dbg !7
  %1593 = select i1 %.not18, float %1454, float %1592, !dbg !7
  %1594 = insertelement <4 x float> poison, float %1593, i64 0, !dbg !19
  %1595 = bitcast i32 %1467 to float, !dbg !7
  %1596 = select i1 %.not18, float %1455, float %1595, !dbg !7
  %1597 = insertelement <4 x float> %1594, float %1596, i64 1, !dbg !19
  %1598 = select i1 %.not18, float %1592, float %1460, !dbg !7
  %1599 = insertelement <4 x float> %1597, float %1598, i64 2, !dbg !19
  %1600 = select i1 %.not18, float %1595, float %1461, !dbg !7
  %1601 = insertelement <4 x float> %1599, float %1600, i64 3, !dbg !19
  %normalized54 = fmul <4 x float> %1601, %row.reciprocal, !dbg !19
  %1602 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !20
  %1603 = extractelement <4 x half> %1602, i64 3, !dbg !20
  %1604 = extractelement <4 x half> %1602, i64 2, !dbg !20
  %1605 = extractelement <4 x half> %1602, i64 1, !dbg !20
  %1606 = extractelement <4 x half> %1602, i64 0, !dbg !20
  %1607 = bitcast i32 %1354 to float, !dbg !7
  %1608 = select i1 %.not18, float %1344, float %1607, !dbg !7
  %1609 = insertelement <4 x float> poison, float %1608, i64 0, !dbg !19
  %1610 = bitcast i32 %1357 to float, !dbg !7
  %1611 = select i1 %.not18, float %1345, float %1610, !dbg !7
  %1612 = insertelement <4 x float> %1609, float %1611, i64 1, !dbg !19
  %1613 = select i1 %.not18, float %1607, float %1350, !dbg !7
  %1614 = insertelement <4 x float> %1612, float %1613, i64 2, !dbg !19
  %1615 = select i1 %.not18, float %1610, float %1351, !dbg !7
  %1616 = insertelement <4 x float> %1614, float %1615, i64 3, !dbg !19
  %row.reciprocal55 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1251, !dbg !19
  %normalized56 = fmul <4 x float> %1616, %row.reciprocal55, !dbg !19
  %1617 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !20
  %1618 = extractelement <4 x half> %1617, i64 3, !dbg !20
  %1619 = extractelement <4 x half> %1617, i64 2, !dbg !20
  %1620 = extractelement <4 x half> %1617, i64 1, !dbg !20
  %1621 = extractelement <4 x half> %1617, i64 0, !dbg !20
  %1622 = bitcast i32 %1246 to float, !dbg !7
  %1623 = select i1 %.not18, float %1236, float %1622, !dbg !7
  %1624 = insertelement <4 x float> poison, float %1623, i64 0, !dbg !19
  %1625 = bitcast i32 %1249 to float, !dbg !7
  %1626 = select i1 %.not18, float %1237, float %1625, !dbg !7
  %1627 = insertelement <4 x float> %1624, float %1626, i64 1, !dbg !19
  %1628 = select i1 %.not18, float %1622, float %1242, !dbg !7
  %1629 = insertelement <4 x float> %1627, float %1628, i64 2, !dbg !19
  %1630 = select i1 %.not18, float %1625, float %1243, !dbg !7
  %1631 = insertelement <4 x float> %1629, float %1630, i64 3, !dbg !19
  %normalized57 = fmul <4 x float> %1631, %row.reciprocal55, !dbg !19
  %1632 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !20
  %1633 = extractelement <4 x half> %1632, i64 3, !dbg !20
  %1634 = extractelement <4 x half> %1632, i64 2, !dbg !20
  %1635 = extractelement <4 x half> %1632, i64 1, !dbg !20
  %1636 = extractelement <4 x half> %1632, i64 0, !dbg !20
  %1637 = add nsw i64 %16, %22, !dbg !21
  %1638 = shl i64 %1637, 19, !dbg !21
  %1639 = and i64 %1638, 16252928, !dbg !21
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
  %address.local544 = add nuw nsw i64 %1639, %address.local543, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local546 = lshr i64 %address.local535, 2, !dbg !8
  %address.local547 = and i64 %address.local546, 12, !dbg !8
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local547, !dbg !21
  store half %1636, ptr addrspace(1) %address.local548, align 2, !dbg !21
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local550 = or disjoint i64 %address.local547, 1, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local549, i64 %address.local550, !dbg !21
  store half %1635, ptr addrspace(1) %address.local551, align 2, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local553 = or disjoint i64 %address.local547, 2, !dbg !21
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local552, i64 %address.local553, !dbg !21
  store half %1634, ptr addrspace(1) %address.local554, align 2, !dbg !21
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local556 = or disjoint i64 %address.local547, 3, !dbg !21
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local555, i64 %address.local556, !dbg !21
  store half %1633, ptr addrspace(1) %address.local557, align 2, !dbg !21
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local559 = or disjoint i64 %address.local547, 16, !dbg !21
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local558, i64 %address.local559, !dbg !21
  store half %1621, ptr addrspace(1) %address.local560, align 2, !dbg !21
  %address.local561 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local562 = or disjoint i64 %address.local547, 17, !dbg !21
  %address.local563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local561, i64 %address.local562, !dbg !21
  store half %1620, ptr addrspace(1) %address.local563, align 2, !dbg !21
  %address.local564 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local565 = or disjoint i64 %address.local547, 18, !dbg !21
  %address.local566 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local564, i64 %address.local565, !dbg !21
  store half %1619, ptr addrspace(1) %address.local566, align 2, !dbg !21
  %address.local567 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local568 = or disjoint i64 %address.local547, 19, !dbg !21
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local567, i64 %address.local568, !dbg !21
  store half %1618, ptr addrspace(1) %address.local569, align 2, !dbg !21
  %address.local570 = shl i64 %address.local542, 7, !dbg !21
  %address.local571 = add i64 %address.local570, 8192, !dbg !21
  %address.local572 = add nuw nsw i64 %1639, %address.local571, !dbg !21
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local573, i64 %address.local547, !dbg !21
  store half %1606, ptr addrspace(1) %address.local574, align 2, !dbg !21
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local550, !dbg !21
  store half %1605, ptr addrspace(1) %address.local576, align 2, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local553, !dbg !21
  store half %1604, ptr addrspace(1) %address.local578, align 2, !dbg !21
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local579, i64 %address.local556, !dbg !21
  store half %1603, ptr addrspace(1) %address.local580, align 2, !dbg !21
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local559, !dbg !21
  store half %1591, ptr addrspace(1) %address.local582, align 2, !dbg !21
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local562, !dbg !21
  store half %1590, ptr addrspace(1) %address.local584, align 2, !dbg !21
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local585, i64 %address.local565, !dbg !21
  store half %1589, ptr addrspace(1) %address.local586, align 2, !dbg !21
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !21
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local587, i64 %address.local568, !dbg !21
  store half %1588, ptr addrspace(1) %address.local588, align 2, !dbg !21
  br label %1640, !dbg !21

1640:                                             ; preds = %1576, %1125
  br i1 true, label %1641, label %1679, !dbg !21

1641:                                             ; preds = %1640
  %1642 = insertelement <4 x float> poison, float %1546, i64 0, !dbg !19
  %1643 = insertelement <4 x float> %1642, float %1552, i64 1, !dbg !19
  %1644 = insertelement <4 x float> %1643, float %1547, i64 2, !dbg !19
  %1645 = insertelement <4 x float> %1644, float %1553, i64 3, !dbg !19
  %row.reciprocal58 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1469, !dbg !19
  %normalized59 = fmul <4 x float> %1645, %row.reciprocal58, !dbg !19
  %1646 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !20
  %1647 = extractelement <4 x half> %1646, i64 3, !dbg !20
  %1648 = extractelement <4 x half> %1646, i64 2, !dbg !20
  %1649 = extractelement <4 x half> %1646, i64 1, !dbg !20
  %1650 = extractelement <4 x half> %1646, i64 0, !dbg !20
  %1651 = insertelement <4 x float> poison, float %1437, i64 0, !dbg !19
  %1652 = insertelement <4 x float> %1651, float %1443, i64 1, !dbg !19
  %1653 = insertelement <4 x float> %1652, float %1438, i64 2, !dbg !19
  %1654 = insertelement <4 x float> %1653, float %1444, i64 3, !dbg !19
  %normalized60 = fmul <4 x float> %1654, %row.reciprocal58, !dbg !19
  %1655 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !20
  %1656 = extractelement <4 x half> %1655, i64 3, !dbg !20
  %1657 = extractelement <4 x half> %1655, i64 2, !dbg !20
  %1658 = extractelement <4 x half> %1655, i64 1, !dbg !20
  %1659 = extractelement <4 x half> %1655, i64 0, !dbg !20
  %1660 = insertelement <4 x float> poison, float %1328, i64 0, !dbg !19
  %1661 = insertelement <4 x float> %1660, float %1334, i64 1, !dbg !19
  %1662 = insertelement <4 x float> %1661, float %1329, i64 2, !dbg !19
  %1663 = insertelement <4 x float> %1662, float %1335, i64 3, !dbg !19
  %row.reciprocal61 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1251, !dbg !19
  %normalized62 = fmul <4 x float> %1663, %row.reciprocal61, !dbg !19
  %1664 = fptrunc <4 x float> %normalized62 to <4 x half>, !dbg !20
  %1665 = extractelement <4 x half> %1664, i64 3, !dbg !20
  %1666 = extractelement <4 x half> %1664, i64 2, !dbg !20
  %1667 = extractelement <4 x half> %1664, i64 1, !dbg !20
  %1668 = extractelement <4 x half> %1664, i64 0, !dbg !20
  %1669 = insertelement <4 x float> poison, float %1219, i64 0, !dbg !19
  %1670 = insertelement <4 x float> %1669, float %1225, i64 1, !dbg !19
  %1671 = insertelement <4 x float> %1670, float %1220, i64 2, !dbg !19
  %1672 = insertelement <4 x float> %1671, float %1226, i64 3, !dbg !19
  %normalized63 = fmul <4 x float> %1672, %row.reciprocal61, !dbg !19
  %1673 = fptrunc <4 x float> %normalized63 to <4 x half>, !dbg !20
  %1674 = extractelement <4 x half> %1673, i64 3, !dbg !20
  %1675 = extractelement <4 x half> %1673, i64 2, !dbg !20
  %1676 = extractelement <4 x half> %1673, i64 1, !dbg !20
  %1677 = extractelement <4 x half> %1673, i64 0, !dbg !20
  %1678 = shl nuw nsw i64 %24, 19, !dbg !21
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
  %address.local599 = add nuw nsw i64 %1678, %address.local598, !dbg !21
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local601 = lshr i64 %address.local590, 2, !dbg !8
  %address.local602 = and i64 %address.local601, 12, !dbg !8
  %address.local603 = or disjoint i64 %address.local602, 32, !dbg !21
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local603, !dbg !21
  store half %1677, ptr addrspace(1) %address.local604, align 2, !dbg !21
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local606 = or disjoint i64 %address.local602, 33, !dbg !21
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local605, i64 %address.local606, !dbg !21
  store half %1676, ptr addrspace(1) %address.local607, align 2, !dbg !21
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local609 = or disjoint i64 %address.local602, 34, !dbg !21
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local608, i64 %address.local609, !dbg !21
  store half %1675, ptr addrspace(1) %address.local610, align 2, !dbg !21
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local612 = or disjoint i64 %address.local602, 35, !dbg !21
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local611, i64 %address.local612, !dbg !21
  store half %1674, ptr addrspace(1) %address.local613, align 2, !dbg !21
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local615 = or disjoint i64 %address.local602, 48, !dbg !21
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local614, i64 %address.local615, !dbg !21
  store half %1668, ptr addrspace(1) %address.local616, align 2, !dbg !21
  %address.local617 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local618 = or disjoint i64 %address.local602, 49, !dbg !21
  %address.local619 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local617, i64 %address.local618, !dbg !21
  store half %1667, ptr addrspace(1) %address.local619, align 2, !dbg !21
  %address.local620 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local621 = or disjoint i64 %address.local602, 50, !dbg !21
  %address.local622 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local621, !dbg !21
  store half %1666, ptr addrspace(1) %address.local622, align 2, !dbg !21
  %address.local623 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !21
  %address.local624 = or i64 %address.local601, 51, !dbg !21
  %address.local625 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local623, i64 %address.local624, !dbg !21
  store half %1665, ptr addrspace(1) %address.local625, align 2, !dbg !21
  %address.local626 = shl i64 %address.local597, 7, !dbg !21
  %address.local627 = add i64 %address.local626, 8192, !dbg !21
  %address.local628 = add nuw nsw i64 %1678, %address.local627, !dbg !21
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local629, i64 %address.local603, !dbg !21
  store half %1659, ptr addrspace(1) %address.local630, align 2, !dbg !21
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local606, !dbg !21
  store half %1658, ptr addrspace(1) %address.local632, align 2, !dbg !21
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local633, i64 %address.local609, !dbg !21
  store half %1657, ptr addrspace(1) %address.local634, align 2, !dbg !21
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local635, i64 %address.local612, !dbg !21
  store half %1656, ptr addrspace(1) %address.local636, align 2, !dbg !21
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local637, i64 %address.local615, !dbg !21
  store half %1650, ptr addrspace(1) %address.local638, align 2, !dbg !21
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local639, i64 %address.local618, !dbg !21
  store half %1649, ptr addrspace(1) %address.local640, align 2, !dbg !21
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local642 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local641, i64 %address.local621, !dbg !21
  store half %1648, ptr addrspace(1) %address.local642, align 2, !dbg !21
  %address.local643 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !21
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local624, !dbg !21
  store half %1647, ptr addrspace(1) %address.local644, align 2, !dbg !21
  br label %1679, !dbg !21

1679:                                             ; preds = %1641, %1640
  br i1 true, label %1680, label %1718, !dbg !21

1680:                                             ; preds = %1679
  %1681 = insertelement <4 x float> poison, float %1518, i64 0, !dbg !19
  %1682 = insertelement <4 x float> %1681, float %1524, i64 1, !dbg !19
  %1683 = insertelement <4 x float> %1682, float %1519, i64 2, !dbg !19
  %1684 = insertelement <4 x float> %1683, float %1525, i64 3, !dbg !19
  %row.reciprocal64 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1469, !dbg !19
  %normalized65 = fmul <4 x float> %1684, %row.reciprocal64, !dbg !19
  %1685 = fptrunc <4 x float> %normalized65 to <4 x half>, !dbg !20
  %1686 = extractelement <4 x half> %1685, i64 3, !dbg !20
  %1687 = extractelement <4 x half> %1685, i64 2, !dbg !20
  %1688 = extractelement <4 x half> %1685, i64 1, !dbg !20
  %1689 = extractelement <4 x half> %1685, i64 0, !dbg !20
  %1690 = insertelement <4 x float> poison, float %1408, i64 0, !dbg !19
  %1691 = insertelement <4 x float> %1690, float %1414, i64 1, !dbg !19
  %1692 = insertelement <4 x float> %1691, float %1409, i64 2, !dbg !19
  %1693 = insertelement <4 x float> %1692, float %1415, i64 3, !dbg !19
  %normalized66 = fmul <4 x float> %1693, %row.reciprocal64, !dbg !19
  %1694 = fptrunc <4 x float> %normalized66 to <4 x half>, !dbg !20
  %1695 = extractelement <4 x half> %1694, i64 3, !dbg !20
  %1696 = extractelement <4 x half> %1694, i64 2, !dbg !20
  %1697 = extractelement <4 x half> %1694, i64 1, !dbg !20
  %1698 = extractelement <4 x half> %1694, i64 0, !dbg !20
  %1699 = insertelement <4 x float> poison, float %1300, i64 0, !dbg !19
  %1700 = insertelement <4 x float> %1699, float %1306, i64 1, !dbg !19
  %1701 = insertelement <4 x float> %1700, float %1301, i64 2, !dbg !19
  %1702 = insertelement <4 x float> %1701, float %1307, i64 3, !dbg !19
  %row.reciprocal67 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1251, !dbg !19
  %normalized68 = fmul <4 x float> %1702, %row.reciprocal67, !dbg !19
  %1703 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !20
  %1704 = extractelement <4 x half> %1703, i64 3, !dbg !20
  %1705 = extractelement <4 x half> %1703, i64 2, !dbg !20
  %1706 = extractelement <4 x half> %1703, i64 1, !dbg !20
  %1707 = extractelement <4 x half> %1703, i64 0, !dbg !20
  %1708 = insertelement <4 x float> poison, float %1190, i64 0, !dbg !19
  %1709 = insertelement <4 x float> %1708, float %1196, i64 1, !dbg !19
  %1710 = insertelement <4 x float> %1709, float %1191, i64 2, !dbg !19
  %1711 = insertelement <4 x float> %1710, float %1197, i64 3, !dbg !19
  %normalized69 = fmul <4 x float> %1711, %row.reciprocal67, !dbg !19
  %1712 = fptrunc <4 x float> %normalized69 to <4 x half>, !dbg !20
  %1713 = extractelement <4 x half> %1712, i64 3, !dbg !20
  %1714 = extractelement <4 x half> %1712, i64 2, !dbg !20
  %1715 = extractelement <4 x half> %1712, i64 1, !dbg !20
  %1716 = extractelement <4 x half> %1712, i64 0, !dbg !20
  %1717 = shl nuw nsw i64 %24, 19, !dbg !21
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
  %address.local655 = add nuw nsw i64 %1717, %address.local654, !dbg !21
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local657 = lshr i64 %address.local646, 2, !dbg !8
  %address.local658 = and i64 %address.local657, 12, !dbg !8
  %address.local659 = or disjoint i64 %address.local658, 64, !dbg !21
  %address.local660 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local656, i64 %address.local659, !dbg !21
  store half %1716, ptr addrspace(1) %address.local660, align 2, !dbg !21
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local662 = or disjoint i64 %address.local658, 65, !dbg !21
  %address.local663 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local661, i64 %address.local662, !dbg !21
  store half %1715, ptr addrspace(1) %address.local663, align 2, !dbg !21
  %address.local664 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local665 = or disjoint i64 %address.local658, 66, !dbg !21
  %address.local666 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local664, i64 %address.local665, !dbg !21
  store half %1714, ptr addrspace(1) %address.local666, align 2, !dbg !21
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local668 = or disjoint i64 %address.local658, 67, !dbg !21
  %address.local669 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local667, i64 %address.local668, !dbg !21
  store half %1713, ptr addrspace(1) %address.local669, align 2, !dbg !21
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local671 = or disjoint i64 %address.local658, 80, !dbg !21
  %address.local672 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local670, i64 %address.local671, !dbg !21
  store half %1707, ptr addrspace(1) %address.local672, align 2, !dbg !21
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local674 = or disjoint i64 %address.local658, 81, !dbg !21
  %address.local675 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local673, i64 %address.local674, !dbg !21
  store half %1706, ptr addrspace(1) %address.local675, align 2, !dbg !21
  %address.local676 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local677 = or disjoint i64 %address.local658, 82, !dbg !21
  %address.local678 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local676, i64 %address.local677, !dbg !21
  store half %1705, ptr addrspace(1) %address.local678, align 2, !dbg !21
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !21
  %address.local680 = or disjoint i64 %address.local658, 83, !dbg !21
  %address.local681 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local679, i64 %address.local680, !dbg !21
  store half %1704, ptr addrspace(1) %address.local681, align 2, !dbg !21
  %address.local682 = shl i64 %address.local653, 7, !dbg !21
  %address.local683 = add i64 %address.local682, 8192, !dbg !21
  %address.local684 = add nuw nsw i64 %1717, %address.local683, !dbg !21
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local685, i64 %address.local659, !dbg !21
  store half %1698, ptr addrspace(1) %address.local686, align 2, !dbg !21
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local687, i64 %address.local662, !dbg !21
  store half %1697, ptr addrspace(1) %address.local688, align 2, !dbg !21
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local689, i64 %address.local665, !dbg !21
  store half %1696, ptr addrspace(1) %address.local690, align 2, !dbg !21
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local691, i64 %address.local668, !dbg !21
  store half %1695, ptr addrspace(1) %address.local692, align 2, !dbg !21
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local693, i64 %address.local671, !dbg !21
  store half %1689, ptr addrspace(1) %address.local694, align 2, !dbg !21
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local695, i64 %address.local674, !dbg !21
  store half %1688, ptr addrspace(1) %address.local696, align 2, !dbg !21
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local697, i64 %address.local677, !dbg !21
  store half %1687, ptr addrspace(1) %address.local698, align 2, !dbg !21
  %address.local699 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !21
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local699, i64 %address.local680, !dbg !21
  store half %1686, ptr addrspace(1) %address.local700, align 2, !dbg !21
  br label %1718, !dbg !21

1718:                                             ; preds = %1680, %1679
  br i1 true, label %1719, label %1757, !dbg !21

1719:                                             ; preds = %1718
  %1720 = insertelement <4 x float> poison, float %1490, i64 0, !dbg !19
  %1721 = insertelement <4 x float> %1720, float %1496, i64 1, !dbg !19
  %1722 = insertelement <4 x float> %1721, float %1491, i64 2, !dbg !19
  %1723 = insertelement <4 x float> %1722, float %1497, i64 3, !dbg !19
  %row.reciprocal70 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1469, !dbg !19
  %normalized71 = fmul <4 x float> %1723, %row.reciprocal70, !dbg !19
  %1724 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !20
  %1725 = extractelement <4 x half> %1724, i64 3, !dbg !20
  %1726 = extractelement <4 x half> %1724, i64 2, !dbg !20
  %1727 = extractelement <4 x half> %1724, i64 1, !dbg !20
  %1728 = extractelement <4 x half> %1724, i64 0, !dbg !20
  %1729 = insertelement <4 x float> poison, float %1379, i64 0, !dbg !19
  %1730 = insertelement <4 x float> %1729, float %1385, i64 1, !dbg !19
  %1731 = insertelement <4 x float> %1730, float %1380, i64 2, !dbg !19
  %1732 = insertelement <4 x float> %1731, float %1386, i64 3, !dbg !19
  %normalized72 = fmul <4 x float> %1732, %row.reciprocal70, !dbg !19
  %1733 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !20
  %1734 = extractelement <4 x half> %1733, i64 3, !dbg !20
  %1735 = extractelement <4 x half> %1733, i64 2, !dbg !20
  %1736 = extractelement <4 x half> %1733, i64 1, !dbg !20
  %1737 = extractelement <4 x half> %1733, i64 0, !dbg !20
  %1738 = insertelement <4 x float> poison, float %1272, i64 0, !dbg !19
  %1739 = insertelement <4 x float> %1738, float %1278, i64 1, !dbg !19
  %1740 = insertelement <4 x float> %1739, float %1273, i64 2, !dbg !19
  %1741 = insertelement <4 x float> %1740, float %1279, i64 3, !dbg !19
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1251, !dbg !19
  %normalized74 = fmul <4 x float> %1741, %row.reciprocal73, !dbg !19
  %1742 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %1743 = extractelement <4 x half> %1742, i64 3, !dbg !20
  %1744 = extractelement <4 x half> %1742, i64 2, !dbg !20
  %1745 = extractelement <4 x half> %1742, i64 1, !dbg !20
  %1746 = extractelement <4 x half> %1742, i64 0, !dbg !20
  %1747 = insertelement <4 x float> poison, float %1161, i64 0, !dbg !19
  %1748 = insertelement <4 x float> %1747, float %1167, i64 1, !dbg !19
  %1749 = insertelement <4 x float> %1748, float %1162, i64 2, !dbg !19
  %1750 = insertelement <4 x float> %1749, float %1168, i64 3, !dbg !19
  %normalized75 = fmul <4 x float> %1750, %row.reciprocal73, !dbg !19
  %1751 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !20
  %1752 = extractelement <4 x half> %1751, i64 3, !dbg !20
  %1753 = extractelement <4 x half> %1751, i64 2, !dbg !20
  %1754 = extractelement <4 x half> %1751, i64 1, !dbg !20
  %1755 = extractelement <4 x half> %1751, i64 0, !dbg !20
  %1756 = shl nuw nsw i64 %24, 19, !dbg !21
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
  %address.local711 = add nuw nsw i64 %1756, %address.local710, !dbg !21
  %address.local712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local713 = lshr i64 %address.local702, 2, !dbg !8
  %address.local714 = and i64 %address.local713, 12, !dbg !8
  %address.local715 = or disjoint i64 %address.local714, 96, !dbg !21
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local712, i64 %address.local715, !dbg !21
  store half %1755, ptr addrspace(1) %address.local716, align 2, !dbg !21
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local718 = or disjoint i64 %address.local714, 97, !dbg !21
  %address.local719 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local717, i64 %address.local718, !dbg !21
  store half %1754, ptr addrspace(1) %address.local719, align 2, !dbg !21
  %address.local720 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local721 = or disjoint i64 %address.local714, 98, !dbg !21
  %address.local722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local720, i64 %address.local721, !dbg !21
  store half %1753, ptr addrspace(1) %address.local722, align 2, !dbg !21
  %address.local723 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local724 = or disjoint i64 %address.local714, 99, !dbg !21
  %address.local725 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local723, i64 %address.local724, !dbg !21
  store half %1752, ptr addrspace(1) %address.local725, align 2, !dbg !21
  %address.local726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local727 = or disjoint i64 %address.local714, 112, !dbg !21
  %address.local728 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local726, i64 %address.local727, !dbg !21
  store half %1746, ptr addrspace(1) %address.local728, align 2, !dbg !21
  %address.local729 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local730 = or disjoint i64 %address.local714, 113, !dbg !21
  %address.local731 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local729, i64 %address.local730, !dbg !21
  store half %1745, ptr addrspace(1) %address.local731, align 2, !dbg !21
  %address.local732 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local733 = or disjoint i64 %address.local714, 114, !dbg !21
  %address.local734 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local732, i64 %address.local733, !dbg !21
  store half %1744, ptr addrspace(1) %address.local734, align 2, !dbg !21
  %address.local735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !21
  %address.local736 = or i64 %address.local713, 115, !dbg !21
  %address.local737 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local735, i64 %address.local736, !dbg !21
  store half %1743, ptr addrspace(1) %address.local737, align 2, !dbg !21
  %address.local738 = shl i64 %address.local709, 7, !dbg !21
  %address.local739 = add i64 %address.local738, 8192, !dbg !21
  %address.local740 = add nuw nsw i64 %1756, %address.local739, !dbg !21
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local741, i64 %address.local715, !dbg !21
  store half %1737, ptr addrspace(1) %address.local742, align 2, !dbg !21
  %address.local743 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local743, i64 %address.local718, !dbg !21
  store half %1736, ptr addrspace(1) %address.local744, align 2, !dbg !21
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local746 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local745, i64 %address.local721, !dbg !21
  store half %1735, ptr addrspace(1) %address.local746, align 2, !dbg !21
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local747, i64 %address.local724, !dbg !21
  store half %1734, ptr addrspace(1) %address.local748, align 2, !dbg !21
  %address.local749 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local749, i64 %address.local727, !dbg !21
  store half %1728, ptr addrspace(1) %address.local750, align 2, !dbg !21
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local751, i64 %address.local730, !dbg !21
  store half %1727, ptr addrspace(1) %address.local752, align 2, !dbg !21
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local754 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local753, i64 %address.local733, !dbg !21
  store half %1726, ptr addrspace(1) %address.local754, align 2, !dbg !21
  %address.local755 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !21
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local755, i64 %address.local736, !dbg !21
  store half %1725, ptr addrspace(1) %address.local756, align 2, !dbg !21
  br label %1757, !dbg !21

1757:                                             ; preds = %1719, %1718
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0002-test_input_1x32x4096x128-128-64-32-online/retained32-baseline")
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
