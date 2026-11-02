; ModuleID = 'generated_ir/online_softmax_native_acc/kernel_bn64.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0002-test_input_1x32x4096x128-128-64-32-online/retained32-baseline/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !36 !reqd_work_group_size !63 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !37
  %6 = sext i32 %5 to i64, !dbg !37
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !37
  %8 = sext i32 %7 to i64, !dbg !37
  %9 = shl nsw i64 %8, 5, !dbg !37
  %10 = add nsw i64 %9, %6, !dbg !37
  %11 = lshr i64 %10, 5, !dbg !37
  %12 = and i64 %11, 576460752303423484, !dbg !37
  %13 = and i64 %6, 3, !dbg !37
  %14 = or disjoint i64 %12, %13, !dbg !37
  %15 = sub nsw i64 31, %14, !dbg !37
  %16 = lshr i64 %10, 2, !dbg !37
  %17 = icmp ugt i64 %10, 1023, !dbg !40
  %18 = add nsw i64 %14, -32, !dbg !40
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !40
  %20 = sdiv i64 %19, 32, !dbg !40
  %21 = sext i1 %17 to i64, !dbg !40
  %22 = xor i64 %20, %21, !dbg !40
  %23 = add nsw i64 %16, %22, !dbg !40
  %24 = and i64 %23, 31, !dbg !40
  %25 = shl nsw i64 %15, 7, !dbg !40
  %26 = mul nsw i64 %22, -4096, !dbg !40
  %27 = add i64 %25, %26, !dbg !40
  %28 = shl nuw nsw i64 %24, 19, !dbg !40
  %29 = shl nsw i64 %27, 7, !dbg !40
  %30 = add i64 %28, %29, !dbg !40
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !40, !range !65
  %32 = zext nneg i32 %31 to i64, !dbg !40
  %33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !40
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !40, !range !65, !frisk.register_lifetimes !33
  %address.local = zext nneg i32 %address.thread to i64, !dbg !40
  %address.local176 = shl nuw nsw i64 %address.local, 3, !dbg !40
  %address.local177 = and i64 %address.local176, 1536, !dbg !40
  %address.local178 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local177, !dbg !40
  %address.local179 = ptrtoint ptr addrspace(3) %address.local178 to i64, !dbg !40
  %address.local180 = trunc i64 %address.local179 to i32, !dbg !40
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local180), !dbg !40
  %35 = zext i32 %34 to i64, !dbg !40
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !40
  %address.local181 = shl nuw nsw i64 %address.local, 7, !dbg !40
  %address.local182 = and i64 %address.local181, 16256, !dbg !40
  %address.local183 = lshr i64 %address.local, 4, !dbg !40
  %address.local184 = and i64 %address.local183, 8, !dbg !40
  %address.local185 = or disjoint i64 %address.local182, %address.local184, !dbg !40
  %address.local186 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local185, !dbg !40
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %38 = ptrtoint ptr addrspace(1) %address.local186 to i64, !dbg !40
  %39 = sub i64 %38, %37, !dbg !40
  %dma.buffer.offset996 = trunc i64 %39 to i32, !dbg !40
  %40 = trunc i64 %37 to i32, !dbg !40
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !40
  %42 = lshr i64 %37, 32, !dbg !40
  %43 = trunc i64 %42 to i32, !dbg !40
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !40
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc997 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !40
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !40
  %dma.buffer.lds998 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !40
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds998, i32 %dma.buffer.offset996, <4 x i32> %dma.buffer.rsrc997) #10, !dbg !40
  %address.local187 = or disjoint i64 %address.local, 256, !dbg !40
  %address.local188 = shl nuw nsw i64 %address.local187, 3, !dbg !40
  %address.local189 = and i64 %address.local188, 3584, !dbg !40
  %address.local190 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local189, !dbg !40
  %address.local191 = ptrtoint ptr addrspace(3) %address.local190 to i64, !dbg !40
  %address.local192 = trunc i64 %address.local191 to i32, !dbg !40
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local192), !dbg !40
  %49 = zext i32 %48 to i64, !dbg !40
  %50 = inttoptr i64 %49 to ptr addrspace(3), !dbg !40
  %address.local193 = shl nuw nsw i64 %address.local, 7, !dbg !40
  %address.local194 = and i64 %address.local193, 16256, !dbg !40
  %address.local195 = lshr i64 %address.local187, 4, !dbg !40
  %address.local196 = and i64 %address.local195, 24, !dbg !40
  %address.local197 = or disjoint i64 %address.local194, %address.local196, !dbg !40
  %address.local198 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local197, !dbg !40
  %51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %52 = ptrtoint ptr addrspace(1) %address.local198 to i64, !dbg !40
  %53 = sub i64 %52, %51, !dbg !40
  %dma.buffer.offset993 = trunc i64 %53 to i32, !dbg !40
  %54 = trunc i64 %51 to i32, !dbg !40
  %55 = insertelement <4 x i32> poison, i32 %54, i64 0, !dbg !40
  %56 = lshr i64 %51, 32, !dbg !40
  %57 = trunc i64 %56 to i32, !dbg !40
  %58 = insertelement <4 x i32> %55, i32 %57, i64 1, !dbg !40
  %59 = insertelement <4 x i32> %58, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc994 = insertelement <4 x i32> %59, i32 131072, i64 3, !dbg !40
  %60 = ptrtoint ptr addrspace(3) %50 to i32, !dbg !40
  %dma.buffer.lds995 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !40
  %61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds995, i32 %dma.buffer.offset993, <4 x i32> %dma.buffer.rsrc994) #10, !dbg !40
  %62 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !37
  %63 = sext i32 %62 to i64, !dbg !37
  %64 = shl i64 %15, 7, !dbg !41
  %65 = lshr i64 %32, 6, !dbg !39
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %66 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !40
  %67 = getelementptr i8, ptr addrspace(1) %66, i64 64, !dbg !40
  %address.local199 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local177, !dbg !40
  %address.local200 = ptrtoint ptr addrspace(3) %address.local199 to i64, !dbg !40
  %address.local201 = trunc i64 %address.local200 to i32, !dbg !40
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local201), !dbg !40
  %69 = zext i32 %68 to i64, !dbg !40
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !40
  %address.local202 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local185, !dbg !40
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %72 = ptrtoint ptr addrspace(1) %address.local202 to i64, !dbg !40
  %73 = sub i64 %72, %71, !dbg !40
  %dma.buffer.offset990 = trunc i64 %73 to i32, !dbg !40
  %74 = trunc i64 %71 to i32, !dbg !40
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !40
  %76 = lshr i64 %71, 32, !dbg !40
  %77 = trunc i64 %76 to i32, !dbg !40
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !40
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc991 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !40
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !40
  %dma.buffer.lds992 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !40
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds992, i32 %dma.buffer.offset990, <4 x i32> %dma.buffer.rsrc991) #10, !dbg !40
  %address.local203 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local189, !dbg !40
  %address.local204 = ptrtoint ptr addrspace(3) %address.local203 to i64, !dbg !40
  %address.local205 = trunc i64 %address.local204 to i32, !dbg !40
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local205), !dbg !40
  %83 = zext i32 %82 to i64, !dbg !40
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !40
  %address.local206 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local197, !dbg !40
  %85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %86 = ptrtoint ptr addrspace(1) %address.local206 to i64, !dbg !40
  %87 = sub i64 %86, %85, !dbg !40
  %dma.buffer.offset987 = trunc i64 %87 to i32, !dbg !40
  %88 = trunc i64 %85 to i32, !dbg !40
  %89 = insertelement <4 x i32> poison, i32 %88, i64 0, !dbg !40
  %90 = lshr i64 %85, 32, !dbg !40
  %91 = trunc i64 %90 to i32, !dbg !40
  %92 = insertelement <4 x i32> %89, i32 %91, i64 1, !dbg !40
  %93 = insertelement <4 x i32> %92, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc988 = insertelement <4 x i32> %93, i32 131072, i64 3, !dbg !40
  %94 = ptrtoint ptr addrspace(3) %84 to i32, !dbg !40
  %dma.buffer.lds989 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !40
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds989, i32 %dma.buffer.offset987, <4 x i32> %dma.buffer.rsrc988) #10, !dbg !40
  %address.local207 = lshr i32 %address.thread, 2, !dbg !39
  %address.local208 = and i32 %address.local207, 4, !dbg !39
  %address.local209 = shl nuw nsw i32 %address.thread, 1, !dbg !39
  %address.local210 = and i32 %address.local209, 384, !dbg !39
  %address.local211 = or disjoint i32 %address.local208, %address.local210, !dbg !39
  %address.local212 = shl nuw nsw i32 %address.thread, 3, !dbg !39
  %address.local213 = and i32 %address.local212, 120, !dbg !39
  %address.local214 = or disjoint i32 %address.local211, %address.local213, !dbg !39
  %address.local215 = shl nuw nsw i32 %address.thread, 5, !dbg !39
  %address.local216 = and i32 %address.local215, 1024, !dbg !39
  %address.local217 = or disjoint i32 %address.local214, %address.local216, !dbg !39
  %address.local218 = zext nneg i32 %address.local217 to i64, !dbg !39
  %address.local219 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local218, !dbg !39
  %96 = load <4 x half>, ptr addrspace(3) %address.local219, align 8, !dbg !39
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local219, i64 4096, !dbg !39
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !39
  %98 = fmul <4 x half> %96, splat (half 1.275630e-01), !dbg !39
  %lds.const.gep885 = getelementptr i8, ptr addrspace(3) %address.local219, i64 1024, !dbg !39
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep885, align 8, !dbg !39
  %100 = fmul <4 x half> %97, splat (half 1.275630e-01), !dbg !39
  %lds.const.gep886 = getelementptr i8, ptr addrspace(3) %address.local219, i64 5120, !dbg !39
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep886, align 8, !dbg !39
  %102 = fmul <4 x half> %99, splat (half 1.275630e-01), !dbg !39
  %103 = fmul <4 x half> %101, splat (half 1.275630e-01), !dbg !39
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !40
  %105 = getelementptr i8, ptr addrspace(1) %104, i64 128, !dbg !40
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local180), !dbg !40
  %107 = zext i32 %106 to i64, !dbg !40
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !40
  %address.local262 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local185, !dbg !40
  %109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %110 = ptrtoint ptr addrspace(1) %address.local262 to i64, !dbg !40
  %111 = sub i64 %110, %109, !dbg !40
  %dma.buffer.offset984 = trunc i64 %111 to i32, !dbg !40
  %112 = trunc i64 %109 to i32, !dbg !40
  %113 = insertelement <4 x i32> poison, i32 %112, i64 0, !dbg !40
  %114 = lshr i64 %109, 32, !dbg !40
  %115 = trunc i64 %114 to i32, !dbg !40
  %116 = insertelement <4 x i32> %113, i32 %115, i64 1, !dbg !40
  %117 = insertelement <4 x i32> %116, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc985 = insertelement <4 x i32> %117, i32 131072, i64 3, !dbg !40
  %118 = ptrtoint ptr addrspace(3) %108 to i32, !dbg !40
  %dma.buffer.lds986 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !40
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds986, i32 %dma.buffer.offset984, <4 x i32> %dma.buffer.rsrc985) #10, !dbg !40
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local192), !dbg !40
  %121 = zext i32 %120 to i64, !dbg !40
  %122 = inttoptr i64 %121 to ptr addrspace(3), !dbg !40
  %address.local263 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local197, !dbg !40
  %123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %124 = ptrtoint ptr addrspace(1) %address.local263 to i64, !dbg !40
  %125 = sub i64 %124, %123, !dbg !40
  %dma.buffer.offset981 = trunc i64 %125 to i32, !dbg !40
  %126 = trunc i64 %123 to i32, !dbg !40
  %127 = insertelement <4 x i32> poison, i32 %126, i64 0, !dbg !40
  %128 = lshr i64 %123, 32, !dbg !40
  %129 = trunc i64 %128 to i32, !dbg !40
  %130 = insertelement <4 x i32> %127, i32 %129, i64 1, !dbg !40
  %131 = insertelement <4 x i32> %130, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc982 = insertelement <4 x i32> %131, i32 131072, i64 3, !dbg !40
  %132 = ptrtoint ptr addrspace(3) %122 to i32, !dbg !40
  %dma.buffer.lds983 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !40
  %133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds983, i32 %dma.buffer.offset981, <4 x i32> %dma.buffer.rsrc982) #10, !dbg !40
  %lds.const.gep887 = getelementptr i8, ptr addrspace(3) %address.local219, i64 8192, !dbg !39
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep887, align 8, !dbg !39
  %lds.const.gep888 = getelementptr i8, ptr addrspace(3) %address.local219, i64 12288, !dbg !39
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep888, align 8, !dbg !39
  %lds.const.gep889 = getelementptr i8, ptr addrspace(3) %address.local219, i64 9216, !dbg !39
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep889, align 8, !dbg !39
  %137 = fmul <4 x half> %134, splat (half 1.275630e-01), !dbg !39
  %lds.const.gep890 = getelementptr i8, ptr addrspace(3) %address.local219, i64 13312, !dbg !39
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep890, align 8, !dbg !39
  %139 = fmul <4 x half> %135, splat (half 1.275630e-01), !dbg !39
  %140 = fmul <4 x half> %136, splat (half 1.275630e-01), !dbg !39
  %141 = fmul <4 x half> %138, splat (half 1.275630e-01), !dbg !39
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !40
  %143 = getelementptr i8, ptr addrspace(1) %142, i64 192, !dbg !40
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local201), !dbg !40
  %145 = zext i32 %144 to i64, !dbg !40
  %146 = inttoptr i64 %145 to ptr addrspace(3), !dbg !40
  %address.local268 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local185, !dbg !40
  %147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %148 = ptrtoint ptr addrspace(1) %address.local268 to i64, !dbg !40
  %149 = sub i64 %148, %147, !dbg !40
  %dma.buffer.offset978 = trunc i64 %149 to i32, !dbg !40
  %150 = trunc i64 %147 to i32, !dbg !40
  %151 = insertelement <4 x i32> poison, i32 %150, i64 0, !dbg !40
  %152 = lshr i64 %147, 32, !dbg !40
  %153 = trunc i64 %152 to i32, !dbg !40
  %154 = insertelement <4 x i32> %151, i32 %153, i64 1, !dbg !40
  %155 = insertelement <4 x i32> %154, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc979 = insertelement <4 x i32> %155, i32 131072, i64 3, !dbg !40
  %156 = ptrtoint ptr addrspace(3) %146 to i32, !dbg !40
  %dma.buffer.lds980 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %156), !dbg !40
  %157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds980, i32 %dma.buffer.offset978, <4 x i32> %dma.buffer.rsrc979) #10, !dbg !40
  %158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local205), !dbg !40
  %159 = zext i32 %158 to i64, !dbg !40
  %160 = inttoptr i64 %159 to ptr addrspace(3), !dbg !40
  %address.local269 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local197, !dbg !40
  %161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %162 = ptrtoint ptr addrspace(1) %address.local269 to i64, !dbg !40
  %163 = sub i64 %162, %161, !dbg !40
  %dma.buffer.offset975 = trunc i64 %163 to i32, !dbg !40
  %164 = trunc i64 %161 to i32, !dbg !40
  %165 = insertelement <4 x i32> poison, i32 %164, i64 0, !dbg !40
  %166 = lshr i64 %161, 32, !dbg !40
  %167 = trunc i64 %166 to i32, !dbg !40
  %168 = insertelement <4 x i32> %165, i32 %167, i64 1, !dbg !40
  %169 = insertelement <4 x i32> %168, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc976 = insertelement <4 x i32> %169, i32 131072, i64 3, !dbg !40
  %170 = ptrtoint ptr addrspace(3) %160 to i32, !dbg !40
  %dma.buffer.lds977 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !40
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds977, i32 %dma.buffer.offset975, <4 x i32> %dma.buffer.rsrc976) #10, !dbg !40
  %172 = load <4 x half>, ptr addrspace(3) %address.local219, align 8, !dbg !39
  %lds.const.gep891 = getelementptr i8, ptr addrspace(3) %address.local219, i64 4096, !dbg !39
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep891, align 8, !dbg !39
  %lds.const.gep892 = getelementptr i8, ptr addrspace(3) %address.local219, i64 1024, !dbg !39
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep892, align 8, !dbg !39
  %175 = fmul <4 x half> %172, splat (half 1.275630e-01), !dbg !39
  %lds.const.gep893 = getelementptr i8, ptr addrspace(3) %address.local219, i64 5120, !dbg !39
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep893, align 8, !dbg !39
  %177 = fmul <4 x half> %173, splat (half 1.275630e-01), !dbg !39
  %178 = fmul <4 x half> %174, splat (half 1.275630e-01), !dbg !39
  %179 = fmul <4 x half> %176, splat (half 1.275630e-01), !dbg !39
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %lds.const.gep894 = getelementptr i8, ptr addrspace(3) %address.local219, i64 8192, !dbg !39
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep894, align 8, !dbg !39
  %lds.const.gep895 = getelementptr i8, ptr addrspace(3) %address.local219, i64 12288, !dbg !39
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep895, align 8, !dbg !39
  %lds.const.gep896 = getelementptr i8, ptr addrspace(3) %address.local219, i64 9216, !dbg !39
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep896, align 8, !dbg !39
  %183 = fmul <4 x half> %180, splat (half 1.275630e-01), !dbg !39
  %lds.const.gep897 = getelementptr i8, ptr addrspace(3) %address.local219, i64 13312, !dbg !39
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep897, align 8, !dbg !39
  %185 = fmul <4 x half> %181, splat (half 1.275630e-01), !dbg !39
  %186 = fmul <4 x half> %182, splat (half 1.275630e-01), !dbg !39
  %187 = fmul <4 x half> %184, splat (half 1.275630e-01), !dbg !39
  %188 = or disjoint i64 %64, 127, !dbg !42
  %189 = add i64 %64, 128, !dbg !42
  %190 = icmp uge i64 %189, %188, !dbg !42
  %191 = shl nsw i64 %15, 1, !dbg !38
  %192 = add nsw i64 %191, 2, !dbg !38
  %193 = shl i64 %16, 7, !dbg !44
  %194 = and i64 %193, 3968, !dbg !44
  %195 = shl nsw i64 %63, 12, !dbg !44
  %196 = or disjoint i64 %194, %195, !dbg !44
  %197 = lshr i64 %32, 2, !dbg !45
  %198 = shl nuw nsw i64 %32, 2, !dbg !45
  %199 = mul nsw i64 %197, -16, !dbg !45
  %200 = add nsw i64 %198, %199, !dbg !45
  %201 = and i32 %31, 1, !dbg !45
  %index.compact142 = zext nneg i32 %201 to i64, !dbg !45
  %202 = or disjoint i64 %200, %index.compact142, !dbg !45
  %203 = add nuw nsw i64 %202, 1, !dbg !45
  %204 = sdiv i64 %203, 2, !dbg !45
  br label %205, !dbg !38

205:                                              ; preds = %594, %4
  %206 = phi i64 [ %1113, %594 ], [ 0, %4 ], !dbg !38
  %207 = phi [2 x <8 x float>] [ %1098, %594 ], [ zeroinitializer, %4 ], !dbg !38
  %208 = phi [2 x <1 x float>] [ %874, %594 ], [ zeroinitializer, %4 ], !dbg !38
  %209 = phi [2 x <1 x float>] [ %1100, %594 ], [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], !dbg !38
  %210 = phi [2 x <8 x float>] [ %1104, %594 ], [ zeroinitializer, %4 ], !dbg !38
  %211 = phi [2 x <8 x float>] [ %1108, %594 ], [ zeroinitializer, %4 ], !dbg !38
  %212 = phi [2 x <8 x float>] [ %1112, %594 ], [ zeroinitializer, %4 ], !dbg !38
  %213 = icmp slt i64 %206, %192, !dbg !38
  br i1 %213, label %214, label %1114, !dbg !38

214:                                              ; preds = %205
  %215 = shl nsw i64 %206, 6, !dbg !38
  %216 = lshr i64 %206, 13, !dbg !44
  %217 = add nuw nsw i64 %216, %16, !dbg !44
  %218 = lshr i64 %206, 6, !dbg !44
  %219 = and i64 %218, 127, !dbg !44
  %220 = shl i64 %217, 19, !dbg !44
  %221 = and i64 %220, 16252928, !dbg !44
  %222 = shl i64 %218, 19, !dbg !44
  %223 = shl i64 %206, 13, !dbg !44
  %224 = sub i64 %223, %222, !dbg !44
  %225 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %221, !dbg !44
  %226 = getelementptr [2 x i8], ptr addrspace(1) %225, i64 %219, !dbg !44
  %227 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %224, !dbg !44
  %address.thread270 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !44, !range !65, !frisk.register_lifetimes !33
  %address.local271 = zext nneg i32 %address.thread270 to i64, !dbg !40
  %address.local272 = shl nuw nsw i64 %address.local271, 3, !dbg !40
  %address.local273 = and i64 %address.local272, 1536, !dbg !44
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local273, !dbg !44
  %address.local275 = ptrtoint ptr addrspace(3) %address.local274 to i64, !dbg !44
  %address.local276 = trunc i64 %address.local275 to i32, !dbg !44
  %228 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local276), !dbg !44
  %229 = zext i32 %228 to i64, !dbg !44
  %230 = inttoptr i64 %229 to ptr addrspace(3), !dbg !44
  %address.local277 = lshr i64 %address.local271, 3, !dbg !44
  %address.local278 = and i64 %address.local277, 24, !dbg !44
  %address.local279 = and i64 %address.local271, 59, !dbg !44
  %address.local280 = lshr i64 %address.local271, 2, !dbg !44
  %address.local281 = and i64 %address.local280, 4, !dbg !44
  %address.local282 = add nuw nsw i64 %address.local281, %address.local271, !dbg !44
  %address.local283 = and i64 %address.local282, 4, !dbg !44
  %address.local284 = or disjoint i64 %address.local279, %address.local283, !dbg !44
  %address.local285 = shl nuw nsw i64 %address.local284, 7, !dbg !44
  %address.local286 = or disjoint i64 %address.local278, %address.local285, !dbg !44
  %address.local287 = getelementptr [2 x i8], ptr addrspace(1) %227, i64 %address.local286, !dbg !44
  %231 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !44
  %232 = ptrtoint ptr addrspace(1) %address.local287 to i64, !dbg !44
  %233 = sub i64 %232, %231, !dbg !44
  %dma.buffer.offset972 = trunc i64 %233 to i32, !dbg !44
  %234 = trunc i64 %231 to i32, !dbg !44
  %235 = insertelement <4 x i32> poison, i32 %234, i64 0, !dbg !44
  %236 = lshr i64 %231, 32, !dbg !44
  %237 = trunc i64 %236 to i32, !dbg !44
  %238 = insertelement <4 x i32> %235, i32 %237, i64 1, !dbg !44
  %239 = insertelement <4 x i32> %238, i32 33554432, i64 2, !dbg !44
  %dma.buffer.rsrc973 = insertelement <4 x i32> %239, i32 131072, i64 3, !dbg !44
  %240 = ptrtoint ptr addrspace(3) %230 to i32, !dbg !44
  %dma.buffer.lds974 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %240), !dbg !44
  %241 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds974, i32 %dma.buffer.offset972, <4 x i32> %dma.buffer.rsrc973) #10, !dbg !44
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !44
  fence syncscope("workgroup") release, !dbg !44
  call void @llvm.amdgcn.s.barrier(), !dbg !44
  fence syncscope("workgroup") acquire, !dbg !44
  %242 = add nuw i64 %206, 2048, !dbg !44
  %243 = icmp slt i64 %242, 0, !dbg !44
  %244 = sub nuw i64 -2049, %206, !dbg !44
  %245 = select i1 %243, i64 %244, i64 %242, !dbg !44
  %246 = sdiv i64 %245, 8192, !dbg !44
  %.lobit19 = ashr i64 %242, 63, !dbg !44
  %247 = xor i64 %246, %.lobit19, !dbg !44
  %248 = add nsw i64 %247, %16, !dbg !44
  %249 = add nsw i64 %218, %196, !dbg !44
  %250 = add nsw i64 %249, 32, !dbg !44
  %251 = icmp slt i64 %249, -32, !dbg !44
  %252 = sub nsw i64 -33, %249, !dbg !44
  %253 = select i1 %251, i64 %252, i64 %250, !dbg !44
  %254 = sdiv i64 %253, 128, !dbg !44
  %255 = sext i1 %251 to i64, !dbg !44
  %256 = xor i64 %254, %255, !dbg !44
  %257 = shl nsw i64 %256, 7, !dbg !44
  %258 = sub i64 %196, %257, !dbg !44
  %259 = shl i64 %248, 19, !dbg !44
  %260 = and i64 %259, 16252928, !dbg !44
  %261 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %260, !dbg !44
  %262 = getelementptr [2 x i8], ptr addrspace(1) %261, i64 %218, !dbg !44
  %263 = getelementptr [2 x i8], ptr addrspace(1) %262, i64 %258, !dbg !44
  %264 = getelementptr i8, ptr addrspace(1) %263, i64 64, !dbg !44
  %265 = getelementptr [2 x i8], ptr addrspace(1) %264, i64 %224, !dbg !44
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local273, !dbg !44
  %address.local289 = ptrtoint ptr addrspace(3) %address.local288 to i64, !dbg !44
  %address.local290 = trunc i64 %address.local289 to i32, !dbg !44
  %266 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local290), !dbg !44
  %267 = zext i32 %266 to i64, !dbg !44
  %268 = inttoptr i64 %267 to ptr addrspace(3), !dbg !44
  %address.local291 = getelementptr [2 x i8], ptr addrspace(1) %265, i64 %address.local286, !dbg !44
  %269 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !44
  %270 = ptrtoint ptr addrspace(1) %address.local291 to i64, !dbg !44
  %271 = sub i64 %270, %269, !dbg !44
  %dma.buffer.offset969 = trunc i64 %271 to i32, !dbg !44
  %272 = trunc i64 %269 to i32, !dbg !44
  %273 = insertelement <4 x i32> poison, i32 %272, i64 0, !dbg !44
  %274 = lshr i64 %269, 32, !dbg !44
  %275 = trunc i64 %274 to i32, !dbg !44
  %276 = insertelement <4 x i32> %273, i32 %275, i64 1, !dbg !44
  %277 = insertelement <4 x i32> %276, i32 33554432, i64 2, !dbg !44
  %dma.buffer.rsrc970 = insertelement <4 x i32> %277, i32 131072, i64 3, !dbg !44
  %278 = ptrtoint ptr addrspace(3) %268 to i32, !dbg !44
  %dma.buffer.lds971 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %278), !dbg !44
  %279 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds971, i32 %dma.buffer.offset969, <4 x i32> %dma.buffer.rsrc970) #10, !dbg !44
  %address.local292 = mul nuw nsw i64 %address.local271, 160, !dbg !45
  %address.local293 = lshr i64 %address.local271, 2, !dbg !45
  %address.local294 = mul nsw i64 %address.local293, -640, !dbg !45
  %address.local295 = add nsw i64 %address.local292, %address.local294, !dbg !45
  %address.local296 = shl nuw nsw i64 %address.local271, 1, !dbg !45
  %address.local297 = and i64 %address.local296, 24, !dbg !45
  %address.local298 = or disjoint i64 %address.local295, %address.local297, !dbg !45
  %address.local299 = shl i64 %address.local271, 62, !dbg !45
  %address.local300 = ashr exact i64 %address.local299, 57, !dbg !45
  %address.local301 = and i64 %address.local300, -64, !dbg !45
  %address.local302 = add nsw i64 %address.local298, %address.local301, !dbg !45
  %address.local303 = lshr i64 %address.local271, 2, !dbg !39
  %address.local304 = and i64 %address.local303, 12, !dbg !39
  %address.local305 = add nsw i64 %address.local302, %address.local304, !dbg !45
  %address.local306 = and i64 %address.local271, 63, !dbg !39
  %address.local307 = lshr i64 %address.local306, 5, !dbg !39
  %address.local308 = mul nuw nsw i64 %address.local307, 504, !dbg !45
  %address.local309 = add i64 %address.local305, %address.local308, !dbg !45
  %address.local310 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local309, !dbg !45
  %280 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !45
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %280, <4 x float> zeroinitializer) #9, !dbg !45
  %lds.const.gep898 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !45
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep898, align 8, !dbg !45
  %address.local321 = shl nuw nsw i32 %address.thread270, 1, !dbg !45
  %address.local322 = and i32 %address.local321, 24, !dbg !45
  %address.local323 = shl nuw nsw i32 %address.thread270, 8, !dbg !45
  %address.local324 = and i32 %address.local323, 768, !dbg !45
  %address.local325 = or disjoint i32 %address.local322, %address.local324, !dbg !45
  %address.local326 = shl nuw nsw i32 %address.thread270, 5, !dbg !45
  %address.local327 = and i32 %address.local326, 96, !dbg !45
  %address.local328 = or disjoint i32 %address.local325, %address.local327, !dbg !45
  %address.local329 = zext nneg i32 %address.local328 to i64, !dbg !45
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %204) #9, !dbg !45, !frisk.register_lifetimes !33
  %address.local330 = lshr i32 %address.thread270, 1, !dbg !45
  %address.local331 = and i32 %address.local330, 1, !dbg !45
  %address.local332 = zext nneg i32 %address.local331 to i64, !dbg !45
  %address.local333 = add nsw i64 %address.base, %address.local332, !dbg !45
  %address.local334 = shl i64 %address.local333, 6, !dbg !45
  %address.local335 = sub i64 %address.local329, %address.local334, !dbg !45
  %address.local336 = add i64 %address.local335, %address.local304, !dbg !45
  %address.local337 = add i64 %address.local336, %address.local308, !dbg !45
  %address.local338 = add i64 %address.local337, 32, !dbg !45
  %address.local339 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local338, !dbg !45
  %283 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !45
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %282, <4 x float> %281) #9, !dbg !45
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %283, <4 x float> zeroinitializer) #9, !dbg !45
  %lds.const.gep899 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !45
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep899, align 8, !dbg !45
  %lds.const.gep900 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !45
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep900, align 8, !dbg !45
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %286, <4 x float> %285) #9, !dbg !45
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %287, <4 x float> zeroinitializer) #9, !dbg !45
  %lds.const.gep901 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !45
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep901, align 8, !dbg !45
  %lds.const.gep902 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !45
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep902, align 8, !dbg !45
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %290, <4 x float> %289) #9, !dbg !45
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %291, <4 x float> zeroinitializer) #9, !dbg !45
  %lds.const.gep903 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !45
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep903, align 8, !dbg !45
  %295 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !45
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %294, <4 x float> %293) #9, !dbg !45
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %295, <4 x float> zeroinitializer) #9, !dbg !45
  %lds.const.gep904 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !45
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep904, align 8, !dbg !45
  %299 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !45
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %298, <4 x float> %297) #9, !dbg !45
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %299, <4 x float> zeroinitializer) #9, !dbg !45
  %lds.const.gep905 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !45
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep905, align 8, !dbg !45
  %lds.const.gep906 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !45
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep906, align 8, !dbg !45
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %302, <4 x float> %301) #9, !dbg !45
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %303, <4 x float> zeroinitializer) #9, !dbg !45
  %lds.const.gep907 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !45
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep907, align 8, !dbg !45
  %lds.const.gep908 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !45
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep908, align 8, !dbg !45
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %306, <4 x float> %305) #9, !dbg !45
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %307, <4 x float> zeroinitializer) #9, !dbg !45
  %lds.const.gep909 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !45
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep909, align 8, !dbg !45
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %310, <4 x float> %309) #9, !dbg !45
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !44
  fence syncscope("workgroup") release, !dbg !44
  call void @llvm.amdgcn.s.barrier(), !dbg !44
  fence syncscope("workgroup") acquire, !dbg !44
  %312 = add nuw i64 %206, 4096, !dbg !44
  %313 = icmp slt i64 %312, 0, !dbg !44
  %314 = sub nuw i64 -4097, %206, !dbg !44
  %315 = select i1 %313, i64 %314, i64 %312, !dbg !44
  %316 = sdiv i64 %315, 8192, !dbg !44
  %.lobit20 = ashr i64 %312, 63, !dbg !44
  %317 = xor i64 %316, %.lobit20, !dbg !44
  %318 = add nsw i64 %317, %16, !dbg !44
  %319 = add nsw i64 %249, 64, !dbg !44
  %320 = icmp slt i64 %249, -64, !dbg !44
  %321 = sub nsw i64 -65, %249, !dbg !44
  %322 = select i1 %320, i64 %321, i64 %319, !dbg !44
  %323 = sdiv i64 %322, 128, !dbg !44
  %324 = sext i1 %320 to i64, !dbg !44
  %325 = xor i64 %323, %324, !dbg !44
  %326 = shl nsw i64 %325, 7, !dbg !44
  %327 = sub i64 %196, %326, !dbg !44
  %328 = shl i64 %318, 19, !dbg !44
  %329 = and i64 %328, 16252928, !dbg !44
  %330 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %329, !dbg !44
  %331 = getelementptr [2 x i8], ptr addrspace(1) %330, i64 %218, !dbg !44
  %332 = getelementptr [2 x i8], ptr addrspace(1) %331, i64 %327, !dbg !44
  %333 = getelementptr i8, ptr addrspace(1) %332, i64 128, !dbg !44
  %334 = getelementptr [2 x i8], ptr addrspace(1) %333, i64 %224, !dbg !44
  %335 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local276), !dbg !44
  %336 = zext i32 %335 to i64, !dbg !44
  %337 = inttoptr i64 %336 to ptr addrspace(3), !dbg !44
  %address.local480 = getelementptr [2 x i8], ptr addrspace(1) %334, i64 %address.local286, !dbg !44
  %338 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !44
  %339 = ptrtoint ptr addrspace(1) %address.local480 to i64, !dbg !44
  %340 = sub i64 %339, %338, !dbg !44
  %dma.buffer.offset966 = trunc i64 %340 to i32, !dbg !44
  %341 = trunc i64 %338 to i32, !dbg !44
  %342 = insertelement <4 x i32> poison, i32 %341, i64 0, !dbg !44
  %343 = lshr i64 %338, 32, !dbg !44
  %344 = trunc i64 %343 to i32, !dbg !44
  %345 = insertelement <4 x i32> %342, i32 %344, i64 1, !dbg !44
  %346 = insertelement <4 x i32> %345, i32 33554432, i64 2, !dbg !44
  %dma.buffer.rsrc967 = insertelement <4 x i32> %346, i32 131072, i64 3, !dbg !44
  %347 = ptrtoint ptr addrspace(3) %337 to i32, !dbg !44
  %dma.buffer.lds968 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %347), !dbg !44
  %348 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds968, i32 %dma.buffer.offset966, <4 x i32> %dma.buffer.rsrc967) #10, !dbg !44
  %lds.const.gep910 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !45
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep910, align 8, !dbg !45
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %349, <4 x float> %284) #9, !dbg !45
  %lds.const.gep911 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !45
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep911, align 8, !dbg !45
  %lds.const.gep912 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !45
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep912, align 8, !dbg !45
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %351, <4 x float> %350) #9, !dbg !45
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %352, <4 x float> %288) #9, !dbg !45
  %lds.const.gep913 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !45
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep913, align 8, !dbg !45
  %lds.const.gep914 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !45
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep914, align 8, !dbg !45
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %355, <4 x float> %354) #9, !dbg !45
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %356, <4 x float> %292) #9, !dbg !45
  %lds.const.gep915 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !45
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep915, align 8, !dbg !45
  %lds.const.gep916 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !45
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep916, align 8, !dbg !45
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %359, <4 x float> %358) #9, !dbg !45
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %360, <4 x float> %296) #9, !dbg !45
  %lds.const.gep917 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !45
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep917, align 8, !dbg !45
  %lds.const.gep918 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !45
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep918, align 8, !dbg !45
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %363, <4 x float> %362) #9, !dbg !45
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %364, <4 x float> %300) #9, !dbg !45
  %lds.const.gep919 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !45
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep919, align 8, !dbg !45
  %lds.const.gep920 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !45
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep920, align 8, !dbg !45
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %367, <4 x float> %366) #9, !dbg !45
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %368, <4 x float> %304) #9, !dbg !45
  %lds.const.gep921 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !45
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep921, align 8, !dbg !45
  %lds.const.gep922 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !45
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep922, align 8, !dbg !45
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %371, <4 x float> %370) #9, !dbg !45
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %372, <4 x float> %308) #9, !dbg !45
  %lds.const.gep923 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !45
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep923, align 8, !dbg !45
  %lds.const.gep924 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !45
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep924, align 8, !dbg !45
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %375, <4 x float> %374) #9, !dbg !45
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %376, <4 x float> %311) #9, !dbg !45
  %lds.const.gep925 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !45
  %379 = load <4 x half>, ptr addrspace(3) %lds.const.gep925, align 8, !dbg !45
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %379, <4 x float> %378) #9, !dbg !45
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !44
  fence syncscope("workgroup") release, !dbg !44
  call void @llvm.amdgcn.s.barrier(), !dbg !44
  fence syncscope("workgroup") acquire, !dbg !44
  %381 = add nuw i64 %206, 6144, !dbg !44
  %382 = icmp slt i64 %381, 0, !dbg !44
  %383 = sub nuw i64 -6145, %206, !dbg !44
  %384 = select i1 %382, i64 %383, i64 %381, !dbg !44
  %385 = sdiv i64 %384, 8192, !dbg !44
  %.lobit21 = ashr i64 %381, 63, !dbg !44
  %386 = xor i64 %385, %.lobit21, !dbg !44
  %387 = add nsw i64 %386, %16, !dbg !44
  %388 = add nsw i64 %249, 96, !dbg !44
  %389 = icmp slt i64 %249, -96, !dbg !44
  %390 = sub nsw i64 -97, %249, !dbg !44
  %391 = select i1 %389, i64 %390, i64 %388, !dbg !44
  %392 = sdiv i64 %391, 128, !dbg !44
  %393 = sext i1 %389 to i64, !dbg !44
  %394 = xor i64 %392, %393, !dbg !44
  %395 = shl nsw i64 %394, 7, !dbg !44
  %396 = sub i64 %196, %395, !dbg !44
  %397 = shl i64 %387, 19, !dbg !44
  %398 = and i64 %397, 16252928, !dbg !44
  %399 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %398, !dbg !44
  %400 = getelementptr [2 x i8], ptr addrspace(1) %399, i64 %218, !dbg !44
  %401 = getelementptr [2 x i8], ptr addrspace(1) %400, i64 %396, !dbg !44
  %402 = getelementptr i8, ptr addrspace(1) %401, i64 192, !dbg !44
  %403 = getelementptr [2 x i8], ptr addrspace(1) %402, i64 %224, !dbg !44
  %404 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local290), !dbg !44
  %405 = zext i32 %404 to i64, !dbg !44
  %406 = inttoptr i64 %405 to ptr addrspace(3), !dbg !44
  %address.local489 = getelementptr [2 x i8], ptr addrspace(1) %403, i64 %address.local286, !dbg !44
  %407 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !44
  %408 = ptrtoint ptr addrspace(1) %address.local489 to i64, !dbg !44
  %409 = sub i64 %408, %407, !dbg !44
  %dma.buffer.offset963 = trunc i64 %409 to i32, !dbg !44
  %410 = trunc i64 %407 to i32, !dbg !44
  %411 = insertelement <4 x i32> poison, i32 %410, i64 0, !dbg !44
  %412 = lshr i64 %407, 32, !dbg !44
  %413 = trunc i64 %412 to i32, !dbg !44
  %414 = insertelement <4 x i32> %411, i32 %413, i64 1, !dbg !44
  %415 = insertelement <4 x i32> %414, i32 33554432, i64 2, !dbg !44
  %dma.buffer.rsrc964 = insertelement <4 x i32> %415, i32 131072, i64 3, !dbg !44
  %416 = ptrtoint ptr addrspace(3) %406 to i32, !dbg !44
  %dma.buffer.lds965 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %416), !dbg !44
  %417 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds965, i32 %dma.buffer.offset963, <4 x i32> %dma.buffer.rsrc964) #10, !dbg !44
  %418 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !45
  %419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %418, <4 x float> %353) #9, !dbg !45
  %lds.const.gep926 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !45
  %420 = load <4 x half>, ptr addrspace(3) %lds.const.gep926, align 8, !dbg !45
  %421 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !45
  %422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %420, <4 x float> %419) #9, !dbg !45
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %421, <4 x float> %357) #9, !dbg !45
  %lds.const.gep927 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !45
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep927, align 8, !dbg !45
  %lds.const.gep928 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !45
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep928, align 8, !dbg !45
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %424, <4 x float> %423) #9, !dbg !45
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %425, <4 x float> %361) #9, !dbg !45
  %lds.const.gep929 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !45
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep929, align 8, !dbg !45
  %lds.const.gep930 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !45
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep930, align 8, !dbg !45
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %428, <4 x float> %427) #9, !dbg !45
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %429, <4 x float> %365) #9, !dbg !45
  %lds.const.gep931 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !45
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep931, align 8, !dbg !45
  %433 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !45
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %432, <4 x float> %431) #9, !dbg !45
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %433, <4 x float> %369) #9, !dbg !45
  %lds.const.gep932 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2048, !dbg !45
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep932, align 8, !dbg !45
  %437 = load <4 x half>, ptr addrspace(3) %address.local339, align 8, !dbg !45
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %436, <4 x float> %435) #9, !dbg !45
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %437, <4 x float> %373) #9, !dbg !45
  %lds.const.gep933 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2048, !dbg !45
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep933, align 8, !dbg !45
  %lds.const.gep934 = getelementptr i8, ptr addrspace(3) %address.local310, i64 128, !dbg !45
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep934, align 8, !dbg !45
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %440, <4 x float> %439) #9, !dbg !45
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %441, <4 x float> %377) #9, !dbg !45
  %lds.const.gep935 = getelementptr i8, ptr addrspace(3) %address.local310, i64 2176, !dbg !45
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep935, align 8, !dbg !45
  %lds.const.gep936 = getelementptr i8, ptr addrspace(3) %address.local339, i64 128, !dbg !45
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep936, align 8, !dbg !45
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %444, <4 x float> %443) #9, !dbg !45
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %445, <4 x float> %380) #9, !dbg !45
  %lds.const.gep937 = getelementptr i8, ptr addrspace(3) %address.local339, i64 2176, !dbg !45
  %448 = load <4 x half>, ptr addrspace(3) %lds.const.gep937, align 8, !dbg !45
  %449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %448, <4 x float> %447) #9, !dbg !45
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !44
  fence syncscope("workgroup") release, !dbg !44
  call void @llvm.amdgcn.s.barrier(), !dbg !44
  fence syncscope("workgroup") acquire, !dbg !44
  %450 = add nuw nsw i64 %218, %16, !dbg !46
  %451 = shl i64 %450, 19, !dbg !46
  %452 = and i64 %451, 16252928, !dbg !46
  %453 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %452, !dbg !46
  %454 = getelementptr [2 x i8], ptr addrspace(1) %453, i64 %224, !dbg !46
  %455 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local276), !dbg !46
  %456 = zext i32 %455 to i64, !dbg !46
  %457 = inttoptr i64 %456 to ptr addrspace(3), !dbg !46
  %address.local490 = shl nuw nsw i64 %address.local271, 5, !dbg !46
  %address.local491 = and i64 %address.local490, 8064, !dbg !46
  %address.local492 = shl nuw nsw i64 %address.local271, 3, !dbg !46
  %address.local493 = and i64 %address.local492, 24, !dbg !46
  %address.local494 = or disjoint i64 %address.local491, %address.local493, !dbg !46
  %address.local495 = getelementptr [2 x i8], ptr addrspace(1) %454, i64 %address.local494, !dbg !46
  %458 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !46
  %459 = ptrtoint ptr addrspace(1) %address.local495 to i64, !dbg !46
  %460 = sub i64 %459, %458, !dbg !46
  %dma.buffer.offset960 = trunc i64 %460 to i32, !dbg !46
  %461 = trunc i64 %458 to i32, !dbg !46
  %462 = insertelement <4 x i32> poison, i32 %461, i64 0, !dbg !46
  %463 = lshr i64 %458, 32, !dbg !46
  %464 = trunc i64 %463 to i32, !dbg !46
  %465 = insertelement <4 x i32> %462, i32 %464, i64 1, !dbg !46
  %466 = insertelement <4 x i32> %465, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc961 = insertelement <4 x i32> %466, i32 131072, i64 3, !dbg !46
  %467 = ptrtoint ptr addrspace(3) %457 to i32, !dbg !46
  %dma.buffer.lds962 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %467), !dbg !46
  %468 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds962, i32 %dma.buffer.offset960, <4 x i32> %dma.buffer.rsrc961) #10, !dbg !46
  %lds.const.gep938 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !45
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep938, align 8, !dbg !45
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %469, <4 x float> %422) #9, !dbg !45
  %lds.const.gep939 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !45
  %471 = load <4 x half>, ptr addrspace(3) %lds.const.gep939, align 8, !dbg !45
  %lds.const.gep940 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !45
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep940, align 8, !dbg !45
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %471, <4 x float> %470) #9, !dbg !45
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %472, <4 x float> %426) #9, !dbg !45
  %lds.const.gep941 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !45
  %475 = load <4 x half>, ptr addrspace(3) %lds.const.gep941, align 8, !dbg !45
  %lds.const.gep942 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !45
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep942, align 8, !dbg !45
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %475, <4 x float> %474) #9, !dbg !45
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %476, <4 x float> %430) #9, !dbg !45
  %lds.const.gep943 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !45
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep943, align 8, !dbg !45
  %lds.const.gep944 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !45
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep944, align 8, !dbg !45
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %479, <4 x float> %478) #9, !dbg !45
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %480, <4 x float> %434) #9, !dbg !45
  %lds.const.gep945 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !45
  %483 = load <4 x half>, ptr addrspace(3) %lds.const.gep945, align 8, !dbg !45
  %lds.const.gep946 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !45
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep946, align 8, !dbg !45
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %483, <4 x float> %482) #9, !dbg !45
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %484, <4 x float> %438) #9, !dbg !45
  %lds.const.gep947 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6144, !dbg !45
  %487 = load <4 x half>, ptr addrspace(3) %lds.const.gep947, align 8, !dbg !45
  %lds.const.gep948 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4096, !dbg !45
  %488 = load <4 x half>, ptr addrspace(3) %lds.const.gep948, align 8, !dbg !45
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %487, <4 x float> %486) #9, !dbg !45
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %488, <4 x float> %442) #9, !dbg !45
  %lds.const.gep949 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6144, !dbg !45
  %491 = load <4 x half>, ptr addrspace(3) %lds.const.gep949, align 8, !dbg !45
  %lds.const.gep950 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4224, !dbg !45
  %492 = load <4 x half>, ptr addrspace(3) %lds.const.gep950, align 8, !dbg !45
  %493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %491, <4 x float> %490) #9, !dbg !45
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %492, <4 x float> %446) #9, !dbg !45
  %lds.const.gep951 = getelementptr i8, ptr addrspace(3) %address.local310, i64 6272, !dbg !45
  %495 = load <4 x half>, ptr addrspace(3) %lds.const.gep951, align 8, !dbg !45
  %lds.const.gep952 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4224, !dbg !45
  %496 = load <4 x half>, ptr addrspace(3) %lds.const.gep952, align 8, !dbg !45
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %495, <4 x float> %494) #9, !dbg !45
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %496, <4 x float> %449) #9, !dbg !45
  %lds.const.gep953 = getelementptr i8, ptr addrspace(3) %address.local339, i64 6272, !dbg !45
  %499 = load <4 x half>, ptr addrspace(3) %lds.const.gep953, align 8, !dbg !45
  %500 = or disjoint i64 %215, 63, !dbg !42
  %501 = icmp uge i64 %64, %500, !dbg !42
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %499, <4 x float> %498) #9, !dbg !45
  %503 = and i1 %190, %501, !dbg !42
  br i1 %503, label %504, label %505, !dbg !42

504:                                              ; preds = %214
  br label %585, !dbg !42

505:                                              ; preds = %214
  %506 = add i64 %64, %32, !dbg !42
  %507 = shl nuw nsw i64 %65, 4, !dbg !42
  %508 = add i64 %506, %507, !dbg !42
  %509 = and i32 %31, 240, !dbg !42
  %index.compact = zext nneg i32 %509 to i64, !dbg !42
  %510 = sub i64 %508, %index.compact, !dbg !42
  %511 = and i64 %32, 48, !dbg !42
  %512 = or disjoint i64 %215, %511, !dbg !42
  %513 = add i64 %510, 1, !dbg !50
  %514 = lshr i64 %513, 32, !dbg !51
  %compare.high = trunc i64 %514 to i32, !dbg !51
  %515 = lshr i64 %512, 32, !dbg !51
  %compare.high757 = trunc i64 %515 to i32, !dbg !51
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high757, !dbg !51
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high757, !dbg !51
  %compare.low.base = trunc i64 %513 to i32, !dbg !51
  %compare.low.base758 = trunc i64 %512 to i32, !dbg !51
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base758, !dbg !51
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !51
  %516 = select i1 %compare.result, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 1) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local = or i32 %compare.local.base, 1, !dbg !51
  %compare.low759 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !51
  %compare.result760 = select i1 %compare.high.equal, i1 %compare.low759, i1 %compare.high.order, !dbg !51
  %517 = select i1 %compare.result760, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base761 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 2) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local762 = or i32 %compare.local.base761, 2, !dbg !51
  %compare.low763 = icmp ugt i32 %compare.low.base, %compare.local762, !dbg !51
  %compare.result764 = select i1 %compare.high.equal, i1 %compare.low763, i1 %compare.high.order, !dbg !51
  %518 = select i1 %compare.result764, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base765 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 3) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local766 = or i32 %compare.local.base765, 3, !dbg !51
  %compare.low767 = icmp ugt i32 %compare.low.base, %compare.local766, !dbg !51
  %compare.result768 = select i1 %compare.high.equal, i1 %compare.low767, i1 %compare.high.order, !dbg !51
  %519 = select i1 %compare.result768, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base769 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 4) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local770 = or i32 %compare.local.base769, 4, !dbg !51
  %compare.low771 = icmp ugt i32 %compare.low.base, %compare.local770, !dbg !51
  %compare.result772 = select i1 %compare.high.equal, i1 %compare.low771, i1 %compare.high.order, !dbg !51
  %520 = select i1 %compare.result772, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base773 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 5) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local774 = or i32 %compare.local.base773, 5, !dbg !51
  %compare.low775 = icmp ugt i32 %compare.low.base, %compare.local774, !dbg !51
  %compare.result776 = select i1 %compare.high.equal, i1 %compare.low775, i1 %compare.high.order, !dbg !51
  %521 = select i1 %compare.result776, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base777 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 6) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local778 = or i32 %compare.local.base777, 6, !dbg !51
  %compare.low779 = icmp ugt i32 %compare.low.base, %compare.local778, !dbg !51
  %compare.result780 = select i1 %compare.high.equal, i1 %compare.low779, i1 %compare.high.order, !dbg !51
  %522 = select i1 %compare.result780, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base781 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 7) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local782 = or i32 %compare.local.base781, 7, !dbg !51
  %compare.low783 = icmp ugt i32 %compare.low.base, %compare.local782, !dbg !51
  %compare.result784 = select i1 %compare.high.equal, i1 %compare.low783, i1 %compare.high.order, !dbg !51
  %523 = select i1 %compare.result784, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base785 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 8) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local786 = or i32 %compare.local.base785, 8, !dbg !51
  %compare.low787 = icmp ugt i32 %compare.low.base, %compare.local786, !dbg !51
  %compare.result788 = select i1 %compare.high.equal, i1 %compare.low787, i1 %compare.high.order, !dbg !51
  %524 = select i1 %compare.result788, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base789 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 9) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local790 = or i32 %compare.local.base789, 9, !dbg !51
  %compare.low791 = icmp ugt i32 %compare.low.base, %compare.local790, !dbg !51
  %compare.result792 = select i1 %compare.high.equal, i1 %compare.low791, i1 %compare.high.order, !dbg !51
  %525 = select i1 %compare.result792, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base793 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 10) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local794 = or i32 %compare.local.base793, 10, !dbg !51
  %compare.low795 = icmp ugt i32 %compare.low.base, %compare.local794, !dbg !51
  %compare.result796 = select i1 %compare.high.equal, i1 %compare.low795, i1 %compare.high.order, !dbg !51
  %526 = select i1 %compare.result796, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base797 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 11) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local798 = or i32 %compare.local.base797, 11, !dbg !51
  %compare.low799 = icmp ugt i32 %compare.low.base, %compare.local798, !dbg !51
  %compare.result800 = select i1 %compare.high.equal, i1 %compare.low799, i1 %compare.high.order, !dbg !51
  %527 = select i1 %compare.result800, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base801 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 12) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local802 = or i32 %compare.local.base801, 12, !dbg !51
  %compare.low803 = icmp ugt i32 %compare.low.base, %compare.local802, !dbg !51
  %compare.result804 = select i1 %compare.high.equal, i1 %compare.low803, i1 %compare.high.order, !dbg !51
  %528 = select i1 %compare.result804, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base805 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 13) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local806 = or i32 %compare.local.base805, 13, !dbg !51
  %compare.low807 = icmp ugt i32 %compare.low.base, %compare.local806, !dbg !51
  %compare.result808 = select i1 %compare.high.equal, i1 %compare.low807, i1 %compare.high.order, !dbg !51
  %529 = select i1 %compare.result808, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base809 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 14) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local810 = or i32 %compare.local.base809, 14, !dbg !51
  %compare.low811 = icmp ugt i32 %compare.low.base, %compare.local810, !dbg !51
  %compare.result812 = select i1 %compare.high.equal, i1 %compare.low811, i1 %compare.high.order, !dbg !51
  %530 = select i1 %compare.result812, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base813 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base758, i32 15) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local814 = or i32 %compare.local.base813, 15, !dbg !51
  %compare.low815 = icmp ugt i32 %compare.low.base, %compare.local814, !dbg !51
  %compare.result816 = select i1 %compare.high.equal, i1 %compare.low815, i1 %compare.high.order, !dbg !51
  %531 = select i1 %compare.result816, float 0.000000e+00, float -inf, !dbg !52
  %532 = add i64 %510, 65, !dbg !50
  %533 = lshr i64 %532, 32, !dbg !51
  %compare.high817 = trunc i64 %533 to i32, !dbg !51
  %534 = lshr i64 %512, 32, !dbg !51
  %compare.high818 = trunc i64 %534 to i32, !dbg !51
  %compare.high.equal819 = icmp eq i32 %compare.high817, %compare.high818, !dbg !51
  %compare.high.order820 = icmp ugt i32 %compare.high817, %compare.high818, !dbg !51
  %compare.low.base821 = trunc i64 %532 to i32, !dbg !51
  %compare.low.base822 = trunc i64 %512 to i32, !dbg !51
  %compare.low823 = icmp ugt i32 %compare.low.base821, %compare.low.base822, !dbg !51
  %compare.result824 = select i1 %compare.high.equal819, i1 %compare.low823, i1 %compare.high.order820, !dbg !51
  %535 = select i1 %compare.result824, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base825 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 16) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local826 = or i32 %compare.local.base825, 1, !dbg !51
  %compare.low827 = icmp ugt i32 %compare.low.base821, %compare.local826, !dbg !51
  %compare.result828 = select i1 %compare.high.equal819, i1 %compare.low827, i1 %compare.high.order820, !dbg !51
  %536 = select i1 %compare.result828, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base829 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 17) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local830 = or i32 %compare.local.base829, 2, !dbg !51
  %compare.low831 = icmp ugt i32 %compare.low.base821, %compare.local830, !dbg !51
  %compare.result832 = select i1 %compare.high.equal819, i1 %compare.low831, i1 %compare.high.order820, !dbg !51
  %537 = select i1 %compare.result832, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base833 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 18) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local834 = or i32 %compare.local.base833, 3, !dbg !51
  %compare.low835 = icmp ugt i32 %compare.low.base821, %compare.local834, !dbg !51
  %compare.result836 = select i1 %compare.high.equal819, i1 %compare.low835, i1 %compare.high.order820, !dbg !51
  %538 = select i1 %compare.result836, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base837 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 19) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local838 = or i32 %compare.local.base837, 4, !dbg !51
  %compare.low839 = icmp ugt i32 %compare.low.base821, %compare.local838, !dbg !51
  %compare.result840 = select i1 %compare.high.equal819, i1 %compare.low839, i1 %compare.high.order820, !dbg !51
  %539 = select i1 %compare.result840, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base841 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 20) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local842 = or i32 %compare.local.base841, 5, !dbg !51
  %compare.low843 = icmp ugt i32 %compare.low.base821, %compare.local842, !dbg !51
  %compare.result844 = select i1 %compare.high.equal819, i1 %compare.low843, i1 %compare.high.order820, !dbg !51
  %540 = select i1 %compare.result844, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base845 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 21) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local846 = or i32 %compare.local.base845, 6, !dbg !51
  %compare.low847 = icmp ugt i32 %compare.low.base821, %compare.local846, !dbg !51
  %compare.result848 = select i1 %compare.high.equal819, i1 %compare.low847, i1 %compare.high.order820, !dbg !51
  %541 = select i1 %compare.result848, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base849 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 22) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local850 = or i32 %compare.local.base849, 7, !dbg !51
  %compare.low851 = icmp ugt i32 %compare.low.base821, %compare.local850, !dbg !51
  %compare.result852 = select i1 %compare.high.equal819, i1 %compare.low851, i1 %compare.high.order820, !dbg !51
  %542 = select i1 %compare.result852, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base853 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 23) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local854 = or i32 %compare.local.base853, 8, !dbg !51
  %compare.low855 = icmp ugt i32 %compare.low.base821, %compare.local854, !dbg !51
  %compare.result856 = select i1 %compare.high.equal819, i1 %compare.low855, i1 %compare.high.order820, !dbg !51
  %543 = select i1 %compare.result856, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base857 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 24) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local858 = or i32 %compare.local.base857, 9, !dbg !51
  %compare.low859 = icmp ugt i32 %compare.low.base821, %compare.local858, !dbg !51
  %compare.result860 = select i1 %compare.high.equal819, i1 %compare.low859, i1 %compare.high.order820, !dbg !51
  %544 = select i1 %compare.result860, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base861 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 25) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local862 = or i32 %compare.local.base861, 10, !dbg !51
  %compare.low863 = icmp ugt i32 %compare.low.base821, %compare.local862, !dbg !51
  %compare.result864 = select i1 %compare.high.equal819, i1 %compare.low863, i1 %compare.high.order820, !dbg !51
  %545 = select i1 %compare.result864, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base865 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 26) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local866 = or i32 %compare.local.base865, 11, !dbg !51
  %compare.low867 = icmp ugt i32 %compare.low.base821, %compare.local866, !dbg !51
  %compare.result868 = select i1 %compare.high.equal819, i1 %compare.low867, i1 %compare.high.order820, !dbg !51
  %546 = select i1 %compare.result868, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base869 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 27) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local870 = or i32 %compare.local.base869, 12, !dbg !51
  %compare.low871 = icmp ugt i32 %compare.low.base821, %compare.local870, !dbg !51
  %compare.result872 = select i1 %compare.high.equal819, i1 %compare.low871, i1 %compare.high.order820, !dbg !51
  %547 = select i1 %compare.result872, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base873 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 28) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local874 = or i32 %compare.local.base873, 13, !dbg !51
  %compare.low875 = icmp ugt i32 %compare.low.base821, %compare.local874, !dbg !51
  %compare.result876 = select i1 %compare.high.equal819, i1 %compare.low875, i1 %compare.high.order820, !dbg !51
  %548 = select i1 %compare.result876, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base877 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 29) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local878 = or i32 %compare.local.base877, 14, !dbg !51
  %compare.low879 = icmp ugt i32 %compare.low.base821, %compare.local878, !dbg !51
  %compare.result880 = select i1 %compare.high.equal819, i1 %compare.low879, i1 %compare.high.order820, !dbg !51
  %549 = select i1 %compare.result880, float 0.000000e+00, float -inf, !dbg !52
  %compare.local.base881 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base822, i32 30) #9, !dbg !51, !frisk.register_lifetimes !33
  %compare.local882 = or i32 %compare.local.base881, 15, !dbg !51
  %compare.low883 = icmp ugt i32 %compare.low.base821, %compare.local882, !dbg !51
  %compare.result884 = select i1 %compare.high.equal819, i1 %compare.low883, i1 %compare.high.order820, !dbg !51
  %550 = select i1 %compare.result884, float 0.000000e+00, float -inf, !dbg !52
  %551 = insertelement <16 x float> poison, float %516, i64 0, !dbg !42
  %552 = insertelement <16 x float> %551, float %517, i64 1, !dbg !42
  %553 = insertelement <16 x float> %552, float %518, i64 2, !dbg !42
  %554 = insertelement <16 x float> %553, float %519, i64 3, !dbg !42
  %555 = insertelement <16 x float> %554, float %520, i64 4, !dbg !42
  %556 = insertelement <16 x float> %555, float %521, i64 5, !dbg !42
  %557 = insertelement <16 x float> %556, float %522, i64 6, !dbg !42
  %558 = insertelement <16 x float> %557, float %523, i64 7, !dbg !42
  %559 = insertelement <16 x float> %558, float %524, i64 8, !dbg !42
  %560 = insertelement <16 x float> %559, float %525, i64 9, !dbg !42
  %561 = insertelement <16 x float> %560, float %526, i64 10, !dbg !42
  %562 = insertelement <16 x float> %561, float %527, i64 11, !dbg !42
  %563 = insertelement <16 x float> %562, float %528, i64 12, !dbg !42
  %564 = insertelement <16 x float> %563, float %529, i64 13, !dbg !42
  %565 = insertelement <16 x float> %564, float %530, i64 14, !dbg !42
  %566 = insertelement <16 x float> %565, float %531, i64 15, !dbg !42
  %567 = insertvalue [2 x <16 x float>] poison, <16 x float> %566, 0, !dbg !42
  %568 = insertelement <16 x float> poison, float %535, i64 0, !dbg !42
  %569 = insertelement <16 x float> %568, float %536, i64 1, !dbg !42
  %570 = insertelement <16 x float> %569, float %537, i64 2, !dbg !42
  %571 = insertelement <16 x float> %570, float %538, i64 3, !dbg !42
  %572 = insertelement <16 x float> %571, float %539, i64 4, !dbg !42
  %573 = insertelement <16 x float> %572, float %540, i64 5, !dbg !42
  %574 = insertelement <16 x float> %573, float %541, i64 6, !dbg !42
  %575 = insertelement <16 x float> %574, float %542, i64 7, !dbg !42
  %576 = insertelement <16 x float> %575, float %543, i64 8, !dbg !42
  %577 = insertelement <16 x float> %576, float %544, i64 9, !dbg !42
  %578 = insertelement <16 x float> %577, float %545, i64 10, !dbg !42
  %579 = insertelement <16 x float> %578, float %546, i64 11, !dbg !42
  %580 = insertelement <16 x float> %579, float %547, i64 12, !dbg !42
  %581 = insertelement <16 x float> %580, float %548, i64 13, !dbg !42
  %582 = insertelement <16 x float> %581, float %549, i64 14, !dbg !42
  %583 = insertelement <16 x float> %582, float %550, i64 15, !dbg !42
  %584 = insertvalue [2 x <16 x float>] %567, <16 x float> %583, 1, !dbg !42
  %.masked = extractvalue [2 x <16 x float>] %584, 0, !dbg !53
  %.masked1 = shufflevector <16 x float> %.masked, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %.masked2 = fadd <4 x float> %473, %.masked1, !dbg !53
  %.masked3 = shufflevector <16 x float> %.masked, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %.masked4 = fadd <4 x float> %477, %.masked3, !dbg !53
  %.masked5 = shufflevector <16 x float> %.masked, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !53
  %.masked6 = fadd <4 x float> %481, %.masked5, !dbg !53
  %.masked7 = shufflevector <16 x float> %.masked, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !53
  %.masked8 = fadd <4 x float> %485, %.masked7, !dbg !53
  %.masked9 = extractvalue [2 x <16 x float>] %584, 1, !dbg !53
  %.masked10 = shufflevector <16 x float> %.masked9, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %.masked11 = fadd <4 x float> %489, %.masked10, !dbg !53
  %.masked12 = shufflevector <16 x float> %.masked9, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %.masked13 = fadd <4 x float> %493, %.masked12, !dbg !53
  %.masked14 = shufflevector <16 x float> %.masked9, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !53
  %.masked15 = fadd <4 x float> %497, %.masked14, !dbg !53
  %.masked16 = shufflevector <16 x float> %.masked9, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !53
  %.masked17 = fadd <4 x float> %502, %.masked16, !dbg !53
  br label %585, !dbg !42

585:                                              ; preds = %505, %504
  %586 = phi <4 x float> [ %.masked2, %505 ], [ %473, %504 ], !dbg !53
  %587 = phi <4 x float> [ %.masked4, %505 ], [ %477, %504 ], !dbg !53
  %588 = phi <4 x float> [ %.masked6, %505 ], [ %481, %504 ], !dbg !53
  %589 = phi <4 x float> [ %.masked8, %505 ], [ %485, %504 ], !dbg !53
  %590 = phi <4 x float> [ %.masked11, %505 ], [ %489, %504 ], !dbg !53
  %591 = phi <4 x float> [ %.masked13, %505 ], [ %493, %504 ], !dbg !53
  %592 = phi <4 x float> [ %.masked15, %505 ], [ %497, %504 ], !dbg !53
  %593 = phi <4 x float> [ %.masked17, %505 ], [ %502, %504 ], !dbg !53
  br label %594, !dbg !42

594:                                              ; preds = %585
  %595 = extractelement <4 x float> %586, i64 0, !dbg !53
  %596 = extractelement <4 x float> %586, i64 1, !dbg !53
  %597 = extractelement <4 x float> %586, i64 2, !dbg !53
  %598 = extractelement <4 x float> %586, i64 3, !dbg !53
  %599 = extractelement <4 x float> %587, i64 0, !dbg !53
  %600 = extractelement <4 x float> %587, i64 1, !dbg !53
  %601 = extractelement <4 x float> %587, i64 2, !dbg !53
  %602 = extractelement <4 x float> %587, i64 3, !dbg !53
  %603 = extractelement <4 x float> %588, i64 0, !dbg !53
  %604 = extractelement <4 x float> %588, i64 1, !dbg !53
  %605 = extractelement <4 x float> %588, i64 2, !dbg !53
  %606 = extractelement <4 x float> %588, i64 3, !dbg !53
  %607 = extractelement <4 x float> %589, i64 0, !dbg !53
  %608 = extractelement <4 x float> %589, i64 1, !dbg !53
  %609 = extractelement <4 x float> %589, i64 2, !dbg !53
  %610 = extractelement <4 x float> %589, i64 3, !dbg !53
  %611 = extractelement <4 x float> %590, i64 0, !dbg !53
  %612 = extractelement <4 x float> %590, i64 1, !dbg !53
  %613 = extractelement <4 x float> %590, i64 2, !dbg !53
  %614 = extractelement <4 x float> %590, i64 3, !dbg !53
  %615 = extractelement <4 x float> %591, i64 0, !dbg !53
  %616 = extractelement <4 x float> %591, i64 1, !dbg !53
  %617 = extractelement <4 x float> %591, i64 2, !dbg !53
  %618 = extractelement <4 x float> %591, i64 3, !dbg !53
  %619 = extractelement <4 x float> %592, i64 0, !dbg !53
  %620 = extractelement <4 x float> %592, i64 1, !dbg !53
  %621 = extractelement <4 x float> %592, i64 2, !dbg !53
  %622 = extractelement <4 x float> %592, i64 3, !dbg !53
  %623 = extractelement <4 x float> %593, i64 0, !dbg !53
  %624 = extractelement <4 x float> %593, i64 1, !dbg !53
  %625 = extractelement <4 x float> %593, i64 2, !dbg !53
  %626 = extractelement <4 x float> %593, i64 3, !dbg !53
  %627 = call float @llvm.maxnum.f32(float %595, float -inf), !dbg !54
  %628 = call float @llvm.maxnum.f32(float %627, float %596), !dbg !54
  %629 = call float @llvm.maxnum.f32(float %628, float %597), !dbg !54
  %630 = call float @llvm.maxnum.f32(float %629, float %598), !dbg !54
  %631 = call float @llvm.maxnum.f32(float %630, float %599), !dbg !54
  %632 = call float @llvm.maxnum.f32(float %631, float %600), !dbg !54
  %633 = call float @llvm.maxnum.f32(float %632, float %601), !dbg !54
  %634 = call float @llvm.maxnum.f32(float %633, float %602), !dbg !54
  %635 = call float @llvm.maxnum.f32(float %634, float %603), !dbg !54
  %636 = call float @llvm.maxnum.f32(float %635, float %604), !dbg !54
  %637 = call float @llvm.maxnum.f32(float %636, float %605), !dbg !54
  %638 = call float @llvm.maxnum.f32(float %637, float %606), !dbg !54
  %639 = call float @llvm.maxnum.f32(float %638, float %607), !dbg !54
  %640 = call float @llvm.maxnum.f32(float %639, float %608), !dbg !54
  %641 = call float @llvm.maxnum.f32(float %640, float %609), !dbg !54
  %642 = call float @llvm.maxnum.f32(float %641, float %610), !dbg !54
  %643 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !54
  %644 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %643), !dbg !54
  %645 = add nuw nsw i32 %644, 64, !dbg !54
  %646 = xor i32 %644, 16, !dbg !54
  %647 = and i32 %645, 192, !dbg !54
  %648 = icmp samesign ult i32 %646, %647, !dbg !54
  %649 = select i1 %648, i32 %646, i32 %644, !dbg !54
  %650 = shl nuw nsw i32 %649, 2, !dbg !54
  %651 = bitcast float %642 to i32, !dbg !54
  %652 = call i32 @llvm.amdgcn.ds.bpermute(i32 %650, i32 %651), !dbg !54
  %653 = bitcast i32 %652 to float, !dbg !54
  %654 = call float @llvm.maxnum.f32(float %642, float %653), !dbg !54
  %655 = xor i32 %644, 32, !dbg !54
  %656 = icmp samesign ult i32 %655, %647, !dbg !54
  %657 = select i1 %656, i32 %655, i32 %644, !dbg !54
  %658 = shl nuw nsw i32 %657, 2, !dbg !54
  %659 = bitcast float %654 to i32, !dbg !54
  %660 = call i32 @llvm.amdgcn.ds.bpermute(i32 %658, i32 %659), !dbg !54
  %661 = bitcast i32 %660 to float, !dbg !54
  %662 = call float @llvm.maxnum.f32(float %654, float %661), !dbg !54
  %663 = extractvalue [2 x <1 x float>] %209, 0, !dbg !54
  %664 = insertelement <1 x float> poison, float %662, i64 0, !dbg !54
  %665 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %664, <1 x float> %663), !dbg !54
  %666 = call float @llvm.maxnum.f32(float %611, float -inf), !dbg !54
  %667 = call float @llvm.maxnum.f32(float %666, float %612), !dbg !54
  %668 = call float @llvm.maxnum.f32(float %667, float %613), !dbg !54
  %669 = call float @llvm.maxnum.f32(float %668, float %614), !dbg !54
  %670 = call float @llvm.maxnum.f32(float %669, float %615), !dbg !54
  %671 = call float @llvm.maxnum.f32(float %670, float %616), !dbg !54
  %672 = call float @llvm.maxnum.f32(float %671, float %617), !dbg !54
  %673 = call float @llvm.maxnum.f32(float %672, float %618), !dbg !54
  %674 = call float @llvm.maxnum.f32(float %673, float %619), !dbg !54
  %675 = call float @llvm.maxnum.f32(float %674, float %620), !dbg !54
  %676 = call float @llvm.maxnum.f32(float %675, float %621), !dbg !54
  %677 = call float @llvm.maxnum.f32(float %676, float %622), !dbg !54
  %678 = call float @llvm.maxnum.f32(float %677, float %623), !dbg !54
  %679 = call float @llvm.maxnum.f32(float %678, float %624), !dbg !54
  %680 = call float @llvm.maxnum.f32(float %679, float %625), !dbg !54
  %681 = call float @llvm.maxnum.f32(float %680, float %626), !dbg !54
  %682 = bitcast float %681 to i32, !dbg !54
  %683 = call i32 @llvm.amdgcn.ds.bpermute(i32 %650, i32 %682), !dbg !54
  %684 = bitcast i32 %683 to float, !dbg !54
  %685 = call float @llvm.maxnum.f32(float %681, float %684), !dbg !54
  %686 = bitcast float %685 to i32, !dbg !54
  %687 = call i32 @llvm.amdgcn.ds.bpermute(i32 %658, i32 %686), !dbg !54
  %688 = bitcast i32 %687 to float, !dbg !54
  %689 = call float @llvm.maxnum.f32(float %685, float %688), !dbg !54
  %690 = extractvalue [2 x <1 x float>] %209, 1, !dbg !54
  %691 = insertelement <1 x float> poison, float %689, i64 0, !dbg !54
  %692 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %691, <1 x float> %690), !dbg !54
  %693 = shufflevector <1 x float> %665, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !55
  %694 = fsub <4 x float> %586, %693, !dbg !55
  %695 = fsub <4 x float> %587, %693, !dbg !55
  %696 = fsub <4 x float> %588, %693, !dbg !55
  %697 = fsub <4 x float> %589, %693, !dbg !55
  %698 = shufflevector <1 x float> %692, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !55
  %699 = fsub <4 x float> %590, %698, !dbg !55
  %700 = fsub <4 x float> %591, %698, !dbg !55
  %701 = fsub <4 x float> %592, %698, !dbg !55
  %702 = fsub <4 x float> %593, %698, !dbg !55
  %703 = fsub <1 x float> %663, %665, !dbg !56
  %704 = extractelement <1 x float> %703, i64 0, !dbg !57
  %705 = call float @llvm.amdgcn.exp2.f32(float %704), !dbg !57
  %706 = insertelement <4 x float> poison, float %705, i64 0, !dbg !62
  %707 = shufflevector <4 x float> %706, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !62
  %708 = extractvalue [2 x <1 x float>] %208, 0, !dbg !58
  %709 = extractelement <1 x float> %708, i64 0, !dbg !58
  %710 = fmul float %709, %705, !dbg !58
  %711 = fsub <1 x float> %690, %692, !dbg !56
  %712 = extractelement <1 x float> %711, i64 0, !dbg !57
  %713 = call float @llvm.amdgcn.exp2.f32(float %712), !dbg !57
  %714 = insertelement <4 x float> poison, float %713, i64 0, !dbg !62
  %715 = shufflevector <4 x float> %714, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !62
  %716 = extractvalue [2 x <1 x float>] %208, 1, !dbg !58
  %717 = extractelement <1 x float> %716, i64 0, !dbg !58
  %718 = fmul float %717, %713, !dbg !58
  %719 = extractelement <4 x float> %694, i64 0, !dbg !59
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !59
  %721 = insertelement <4 x float> poison, float %720, i64 0, !dbg !61
  %722 = extractelement <4 x float> %694, i64 1, !dbg !59
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !59
  %724 = insertelement <4 x float> %721, float %723, i64 1, !dbg !61
  %725 = fadd float %720, %723, !dbg !60
  %726 = extractelement <4 x float> %694, i64 2, !dbg !59
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !59
  %728 = insertelement <4 x float> %724, float %727, i64 2, !dbg !61
  %729 = fadd float %725, %727, !dbg !60
  %730 = extractelement <4 x float> %694, i64 3, !dbg !59
  %731 = call float @llvm.amdgcn.exp2.f32(float %730), !dbg !59
  %732 = insertelement <4 x float> %728, float %731, i64 3, !dbg !61
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %732) #9, !dbg !61, !frisk.register_lifetimes !33
  %733 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !61
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %733) #9, !dbg !60, !frisk.register_lifetimes !33
  %734 = fadd float %729, %731, !dbg !60
  %735 = extractelement <4 x float> %695, i64 0, !dbg !59
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !59
  %737 = insertelement <4 x float> poison, float %736, i64 0, !dbg !61
  %738 = fadd float %734, %736, !dbg !60
  %739 = extractelement <4 x float> %695, i64 1, !dbg !59
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !59
  %741 = insertelement <4 x float> %737, float %740, i64 1, !dbg !61
  %742 = fadd float %738, %740, !dbg !60
  %743 = extractelement <4 x float> %695, i64 2, !dbg !59
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !59
  %745 = insertelement <4 x float> %741, float %744, i64 2, !dbg !61
  %746 = fadd float %742, %744, !dbg !60
  %747 = extractelement <4 x float> %695, i64 3, !dbg !59
  %748 = call float @llvm.amdgcn.exp2.f32(float %747), !dbg !59
  %749 = insertelement <4 x float> %745, float %748, i64 3, !dbg !61
  %math.result162 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %749) #9, !dbg !61, !frisk.register_lifetimes !33
  %750 = fptrunc <4 x float> %math.result162 to <4 x half>, !dbg !61
  %math.packed163 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %750) #9, !dbg !60, !frisk.register_lifetimes !33
  %751 = fadd float %746, %748, !dbg !60
  %752 = extractelement <4 x float> %696, i64 0, !dbg !59
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !59
  %754 = insertelement <4 x float> poison, float %753, i64 0, !dbg !61
  %755 = fadd float %751, %753, !dbg !60
  %756 = extractelement <4 x float> %696, i64 1, !dbg !59
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !59
  %758 = insertelement <4 x float> %754, float %757, i64 1, !dbg !61
  %759 = fadd float %755, %757, !dbg !60
  %760 = extractelement <4 x float> %696, i64 2, !dbg !59
  %761 = call float @llvm.amdgcn.exp2.f32(float %760), !dbg !59
  %762 = insertelement <4 x float> %758, float %761, i64 2, !dbg !61
  %763 = fadd float %759, %761, !dbg !60
  %764 = extractelement <4 x float> %696, i64 3, !dbg !59
  %765 = call float @llvm.amdgcn.exp2.f32(float %764), !dbg !59
  %766 = insertelement <4 x float> %762, float %765, i64 3, !dbg !61
  %math.result164 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %766) #9, !dbg !61, !frisk.register_lifetimes !33
  %767 = fptrunc <4 x float> %math.result164 to <4 x half>, !dbg !61
  %math.packed165 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %767) #9, !dbg !60, !frisk.register_lifetimes !33
  %768 = fadd float %763, %765, !dbg !60
  %769 = extractelement <4 x float> %697, i64 0, !dbg !59
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !59
  %771 = insertelement <4 x float> poison, float %770, i64 0, !dbg !61
  %772 = fadd float %768, %770, !dbg !60
  %773 = extractelement <4 x float> %697, i64 1, !dbg !59
  %774 = call float @llvm.amdgcn.exp2.f32(float %773), !dbg !59
  %775 = insertelement <4 x float> %771, float %774, i64 1, !dbg !61
  %776 = fadd float %772, %774, !dbg !60
  %777 = extractelement <4 x float> %697, i64 2, !dbg !59
  %778 = call float @llvm.amdgcn.exp2.f32(float %777), !dbg !59
  %779 = insertelement <4 x float> %775, float %778, i64 2, !dbg !61
  %780 = fadd float %776, %778, !dbg !60
  %781 = extractelement <4 x float> %697, i64 3, !dbg !59
  %782 = call float @llvm.amdgcn.exp2.f32(float %781), !dbg !59
  %783 = insertelement <4 x float> %779, float %782, i64 3, !dbg !61
  %math.result166 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %783) #9, !dbg !61, !frisk.register_lifetimes !33
  %784 = fptrunc <4 x float> %math.result166 to <4 x half>, !dbg !61
  %math.packed167 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %784) #9, !dbg !60, !frisk.register_lifetimes !33
  %785 = fadd float %780, %782, !dbg !60
  %786 = extractelement <4 x float> %699, i64 0, !dbg !59
  %787 = call float @llvm.amdgcn.exp2.f32(float %786), !dbg !59
  %788 = insertelement <4 x float> poison, float %787, i64 0, !dbg !61
  %789 = extractelement <4 x float> %699, i64 1, !dbg !59
  %790 = call float @llvm.amdgcn.exp2.f32(float %789), !dbg !59
  %791 = insertelement <4 x float> %788, float %790, i64 1, !dbg !61
  %792 = fadd float %787, %790, !dbg !60
  %793 = extractelement <4 x float> %699, i64 2, !dbg !59
  %794 = call float @llvm.amdgcn.exp2.f32(float %793), !dbg !59
  %795 = insertelement <4 x float> %791, float %794, i64 2, !dbg !61
  %796 = fadd float %792, %794, !dbg !60
  %797 = extractelement <4 x float> %699, i64 3, !dbg !59
  %798 = call float @llvm.amdgcn.exp2.f32(float %797), !dbg !59
  %799 = insertelement <4 x float> %795, float %798, i64 3, !dbg !61
  %math.result168 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %799) #9, !dbg !61, !frisk.register_lifetimes !33
  %800 = fptrunc <4 x float> %math.result168 to <4 x half>, !dbg !61
  %math.packed169 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %800) #9, !dbg !60, !frisk.register_lifetimes !33
  %801 = fadd float %796, %798, !dbg !60
  %802 = extractelement <4 x float> %700, i64 0, !dbg !59
  %803 = call float @llvm.amdgcn.exp2.f32(float %802), !dbg !59
  %804 = insertelement <4 x float> poison, float %803, i64 0, !dbg !61
  %805 = fadd float %801, %803, !dbg !60
  %806 = extractelement <4 x float> %700, i64 1, !dbg !59
  %807 = call float @llvm.amdgcn.exp2.f32(float %806), !dbg !59
  %808 = insertelement <4 x float> %804, float %807, i64 1, !dbg !61
  %809 = fadd float %805, %807, !dbg !60
  %810 = extractelement <4 x float> %700, i64 2, !dbg !59
  %811 = call float @llvm.amdgcn.exp2.f32(float %810), !dbg !59
  %812 = insertelement <4 x float> %808, float %811, i64 2, !dbg !61
  %813 = fadd float %809, %811, !dbg !60
  %814 = extractelement <4 x float> %700, i64 3, !dbg !59
  %815 = call float @llvm.amdgcn.exp2.f32(float %814), !dbg !59
  %816 = insertelement <4 x float> %812, float %815, i64 3, !dbg !61
  %math.result170 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %816) #9, !dbg !61, !frisk.register_lifetimes !33
  %817 = fptrunc <4 x float> %math.result170 to <4 x half>, !dbg !61
  %math.packed171 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %817) #9, !dbg !60, !frisk.register_lifetimes !33
  %818 = fadd float %813, %815, !dbg !60
  %819 = extractelement <4 x float> %701, i64 0, !dbg !59
  %820 = call float @llvm.amdgcn.exp2.f32(float %819), !dbg !59
  %821 = insertelement <4 x float> poison, float %820, i64 0, !dbg !61
  %822 = fadd float %818, %820, !dbg !60
  %823 = extractelement <4 x float> %701, i64 1, !dbg !59
  %824 = call float @llvm.amdgcn.exp2.f32(float %823), !dbg !59
  %825 = insertelement <4 x float> %821, float %824, i64 1, !dbg !61
  %826 = fadd float %822, %824, !dbg !60
  %827 = extractelement <4 x float> %701, i64 2, !dbg !59
  %828 = call float @llvm.amdgcn.exp2.f32(float %827), !dbg !59
  %829 = insertelement <4 x float> %825, float %828, i64 2, !dbg !61
  %830 = fadd float %826, %828, !dbg !60
  %831 = extractelement <4 x float> %701, i64 3, !dbg !59
  %832 = call float @llvm.amdgcn.exp2.f32(float %831), !dbg !59
  %833 = insertelement <4 x float> %829, float %832, i64 3, !dbg !61
  %math.result172 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %833) #9, !dbg !61, !frisk.register_lifetimes !33
  %834 = fptrunc <4 x float> %math.result172 to <4 x half>, !dbg !61
  %math.packed173 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %834) #9, !dbg !60, !frisk.register_lifetimes !33
  %835 = fadd float %830, %832, !dbg !60
  %836 = extractelement <4 x float> %702, i64 0, !dbg !59
  %837 = call float @llvm.amdgcn.exp2.f32(float %836), !dbg !59
  %838 = insertelement <4 x float> poison, float %837, i64 0, !dbg !61
  %839 = fadd float %835, %837, !dbg !60
  %840 = extractelement <4 x float> %702, i64 1, !dbg !59
  %841 = call float @llvm.amdgcn.exp2.f32(float %840), !dbg !59
  %842 = insertelement <4 x float> %838, float %841, i64 1, !dbg !61
  %843 = fadd float %839, %841, !dbg !60
  %844 = extractelement <4 x float> %702, i64 2, !dbg !59
  %845 = call float @llvm.amdgcn.exp2.f32(float %844), !dbg !59
  %846 = insertelement <4 x float> %842, float %845, i64 2, !dbg !61
  %847 = fadd float %843, %845, !dbg !60
  %848 = extractelement <4 x float> %702, i64 3, !dbg !59
  %849 = call float @llvm.amdgcn.exp2.f32(float %848), !dbg !59
  %850 = insertelement <4 x float> %846, float %849, i64 3, !dbg !61
  %math.result174 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %850) #9, !dbg !61, !frisk.register_lifetimes !33
  %851 = fptrunc <4 x float> %math.result174 to <4 x half>, !dbg !61
  %math.packed175 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %851) #9, !dbg !60, !frisk.register_lifetimes !33
  %852 = fadd float %847, %849, !dbg !60
  %853 = bitcast float %785 to i32, !dbg !60
  %854 = call i32 @llvm.amdgcn.ds.bpermute(i32 %650, i32 %853), !dbg !60
  %855 = bitcast i32 %854 to float, !dbg !60
  %856 = fadd float %785, %855, !dbg !60
  %857 = bitcast float %856 to i32, !dbg !60
  %858 = call i32 @llvm.amdgcn.ds.bpermute(i32 %658, i32 %857), !dbg !60
  %859 = bitcast i32 %858 to float, !dbg !60
  %860 = fadd float %856, %859, !dbg !60
  %861 = fadd float %860, %710, !dbg !60
  %862 = insertelement <1 x float> poison, float %861, i64 0, !dbg !38
  %863 = insertvalue [2 x <1 x float>] poison, <1 x float> %862, 0, !dbg !38
  %864 = bitcast float %852 to i32, !dbg !60
  %865 = call i32 @llvm.amdgcn.ds.bpermute(i32 %650, i32 %864), !dbg !60
  %866 = bitcast i32 %865 to float, !dbg !60
  %867 = fadd float %852, %866, !dbg !60
  %868 = bitcast float %867 to i32, !dbg !60
  %869 = call i32 @llvm.amdgcn.ds.bpermute(i32 %658, i32 %868), !dbg !60
  %870 = bitcast i32 %869 to float, !dbg !60
  %871 = fadd float %867, %870, !dbg !60
  %872 = fadd float %871, %718, !dbg !60
  %873 = insertelement <1 x float> poison, float %872, i64 0, !dbg !38
  %874 = insertvalue [2 x <1 x float>] %863, <1 x float> %873, 1, !dbg !38
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %address.thread496 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !43, !range !65, !frisk.register_lifetimes !33
  %address.local497 = zext nneg i32 %address.thread496 to i64, !dbg !40
  %address.local498 = and i64 %address.local497, 60, !dbg !43
  %address.local499 = add nuw nsw i64 %address.local498, %address.local497, !dbg !43
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !43
  %address.local501 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !43
  %address.local502 = and i64 %address.local497, 63, !dbg !39
  %address.local503 = lshr i64 %address.local502, 4, !dbg !43
  %address.local504 = mul nuw nsw i64 %address.local503, 384, !dbg !43
  %address.local505 = getelementptr [2 x i8], ptr addrspace(3) %address.local501, i64 %address.local504, !dbg !43
  %address.local506 = lshr i64 %address.local497, 2, !dbg !45
  %address.local507 = mul nsw i64 %address.local506, -32, !dbg !43
  %address.local508 = getelementptr [2 x i8], ptr addrspace(3) %address.local505, i64 %address.local507, !dbg !43
  %875 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !43
  %876 = bitcast <4 x i32> %875 to <8 x half>, !dbg !43
  %address.local509 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !43
  %address.local510 = getelementptr [2 x i8], ptr addrspace(3) %address.local509, i64 %address.local504, !dbg !43
  %address.local511 = getelementptr [2 x i8], ptr addrspace(3) %address.local510, i64 %address.local507, !dbg !43
  %address.local512 = getelementptr i8, ptr addrspace(3) %address.local511, i64 256, !dbg !43
  %877 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !43
  %878 = bitcast <4 x i32> %877 to <8 x half>, !dbg !43
  %address.local513 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !43
  %address.local514 = getelementptr [2 x i8], ptr addrspace(3) %address.local513, i64 %address.local504, !dbg !43
  %address.local515 = getelementptr [2 x i8], ptr addrspace(3) %address.local514, i64 %address.local507, !dbg !43
  %address.local516 = getelementptr i8, ptr addrspace(3) %address.local515, i64 512, !dbg !43
  %879 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !43
  %880 = bitcast <4 x i32> %879 to <8 x half>, !dbg !43
  %address.local517 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !43
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) %address.local517, i64 %address.local504, !dbg !43
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) %address.local518, i64 %address.local507, !dbg !43
  %address.local520 = getelementptr i8, ptr addrspace(3) %address.local519, i64 768, !dbg !43
  %881 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !43
  %882 = bitcast <4 x i32> %881 to <8 x half>, !dbg !43
  %883 = add nuw nsw i64 %218, %16, !dbg !46
  %884 = shl i64 %883, 19, !dbg !46
  %885 = and i64 %884, 16252928, !dbg !46
  %886 = add i64 %885, %224, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %887 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %886, !dbg !46
  %888 = getelementptr i8, ptr addrspace(1) %887, i64 64, !dbg !46
  %address.local521 = shl nuw nsw i64 %address.local497, 3, !dbg !40
  %address.local522 = and i64 %address.local521, 1536, !dbg !44
  %address.local523 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local522, !dbg !44
  %address.local524 = ptrtoint ptr addrspace(3) %address.local523 to i64, !dbg !44
  %address.local525 = trunc i64 %address.local524 to i32, !dbg !44
  %889 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !46
  %890 = zext i32 %889 to i64, !dbg !46
  %891 = inttoptr i64 %890 to ptr addrspace(3), !dbg !46
  %address.local526 = shl nuw nsw i64 %address.local497, 5, !dbg !46
  %address.local527 = and i64 %address.local526, 8064, !dbg !46
  %address.local528 = shl nuw nsw i64 %address.local497, 3, !dbg !46
  %address.local529 = and i64 %address.local528, 24, !dbg !46
  %address.local530 = or disjoint i64 %address.local527, %address.local529, !dbg !46
  %address.local531 = getelementptr [2 x i8], ptr addrspace(1) %888, i64 %address.local530, !dbg !46
  %892 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !46
  %893 = ptrtoint ptr addrspace(1) %address.local531 to i64, !dbg !46
  %894 = sub i64 %893, %892, !dbg !46
  %dma.buffer.offset957 = trunc i64 %894 to i32, !dbg !46
  %895 = trunc i64 %892 to i32, !dbg !46
  %896 = insertelement <4 x i32> poison, i32 %895, i64 0, !dbg !46
  %897 = lshr i64 %892, 32, !dbg !46
  %898 = trunc i64 %897 to i32, !dbg !46
  %899 = insertelement <4 x i32> %896, i32 %898, i64 1, !dbg !46
  %900 = insertelement <4 x i32> %899, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc958 = insertelement <4 x i32> %900, i32 131072, i64 3, !dbg !46
  %901 = ptrtoint ptr addrspace(3) %891 to i32, !dbg !46
  %dma.buffer.lds959 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %901), !dbg !46
  %902 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds959, i32 %dma.buffer.offset957, <4 x i32> %dma.buffer.rsrc958) #10, !dbg !46
  %903 = shufflevector <8 x half> %876, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %904 = extractvalue [2 x <8 x float>] %207, 0, !dbg !62
  %905 = shufflevector <8 x float> %904, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %906 = fmul <4 x float> %905, %707, !dbg !62
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %903, <4 x float> %906) #9, !dbg !43
  %908 = shufflevector <8 x half> %878, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %908, <4 x float> %907) #9, !dbg !43
  %910 = shufflevector <8 x half> %880, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %910, <4 x float> %909) #9, !dbg !43
  %912 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %912, <4 x float> %911) #9, !dbg !43
  %914 = shufflevector <8 x half> %876, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %915 = shufflevector <8 x float> %904, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %916 = fmul <4 x float> %915, %707, !dbg !62
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %914, <4 x float> %916) #9, !dbg !43
  %918 = shufflevector <8 x half> %878, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %918, <4 x float> %917) #9, !dbg !43
  %920 = shufflevector <8 x half> %880, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %920, <4 x float> %919) #9, !dbg !43
  %922 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %922, <4 x float> %921) #9, !dbg !43
  %924 = extractvalue [2 x <8 x float>] %207, 1, !dbg !62
  %925 = shufflevector <8 x float> %924, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %926 = fmul <4 x float> %925, %715, !dbg !62
  %927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %903, <4 x float> %926) #9, !dbg !43
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %908, <4 x float> %927) #9, !dbg !43
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %910, <4 x float> %928) #9, !dbg !43
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %912, <4 x float> %929) #9, !dbg !43
  %931 = shufflevector <8 x float> %924, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %932 = fmul <4 x float> %931, %715, !dbg !62
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %914, <4 x float> %932) #9, !dbg !43
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %918, <4 x float> %933) #9, !dbg !43
  %935 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %920, <4 x float> %934) #9, !dbg !43
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %922, <4 x float> %935) #9, !dbg !43
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %937 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !43
  %938 = bitcast <4 x i32> %937 to <8 x half>, !dbg !43
  %939 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !43
  %940 = bitcast <4 x i32> %939 to <8 x half>, !dbg !43
  %941 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !43
  %942 = bitcast <4 x i32> %941 to <8 x half>, !dbg !43
  %943 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !43
  %944 = bitcast <4 x i32> %943 to <8 x half>, !dbg !43
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %945 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %886, !dbg !46
  %946 = getelementptr i8, ptr addrspace(1) %945, i64 128, !dbg !46
  %947 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !46
  %948 = zext i32 %947 to i64, !dbg !46
  %949 = inttoptr i64 %948 to ptr addrspace(3), !dbg !46
  %address.local532 = getelementptr [2 x i8], ptr addrspace(1) %946, i64 %address.local530, !dbg !46
  %950 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !46
  %951 = ptrtoint ptr addrspace(1) %address.local532 to i64, !dbg !46
  %952 = sub i64 %951, %950, !dbg !46
  %dma.buffer.offset954 = trunc i64 %952 to i32, !dbg !46
  %953 = trunc i64 %950 to i32, !dbg !46
  %954 = insertelement <4 x i32> poison, i32 %953, i64 0, !dbg !46
  %955 = lshr i64 %950, 32, !dbg !46
  %956 = trunc i64 %955 to i32, !dbg !46
  %957 = insertelement <4 x i32> %954, i32 %956, i64 1, !dbg !46
  %958 = insertelement <4 x i32> %957, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc955 = insertelement <4 x i32> %958, i32 131072, i64 3, !dbg !46
  %959 = ptrtoint ptr addrspace(3) %949 to i32, !dbg !46
  %dma.buffer.lds956 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %959), !dbg !46
  %960 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds956, i32 %dma.buffer.offset954, <4 x i32> %dma.buffer.rsrc955) #10, !dbg !46
  %961 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %962 = extractvalue [2 x <8 x float>] %210, 0, !dbg !62
  %963 = shufflevector <8 x float> %962, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %964 = fmul <4 x float> %963, %707, !dbg !62
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %961, <4 x float> %964) #9, !dbg !43
  %966 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %966, <4 x float> %965) #9, !dbg !43
  %968 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %968, <4 x float> %967) #9, !dbg !43
  %970 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %970, <4 x float> %969) #9, !dbg !43
  %972 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %973 = shufflevector <8 x float> %962, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %974 = fmul <4 x float> %973, %707, !dbg !62
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %972, <4 x float> %974) #9, !dbg !43
  %976 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %976, <4 x float> %975) #9, !dbg !43
  %978 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %978, <4 x float> %977) #9, !dbg !43
  %980 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %980, <4 x float> %979) #9, !dbg !43
  %982 = extractvalue [2 x <8 x float>] %210, 1, !dbg !62
  %983 = shufflevector <8 x float> %982, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %984 = fmul <4 x float> %983, %715, !dbg !62
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %961, <4 x float> %984) #9, !dbg !43
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %966, <4 x float> %985) #9, !dbg !43
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %968, <4 x float> %986) #9, !dbg !43
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %970, <4 x float> %987) #9, !dbg !43
  %989 = shufflevector <8 x float> %982, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %990 = fmul <4 x float> %989, %715, !dbg !62
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %972, <4 x float> %990) #9, !dbg !43
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %976, <4 x float> %991) #9, !dbg !43
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %978, <4 x float> %992) #9, !dbg !43
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %980, <4 x float> %993) #9, !dbg !43
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %995 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !43
  %996 = bitcast <4 x i32> %995 to <8 x half>, !dbg !43
  %997 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !43
  %998 = bitcast <4 x i32> %997 to <8 x half>, !dbg !43
  %999 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !43
  %1000 = bitcast <4 x i32> %999 to <8 x half>, !dbg !43
  %1001 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !43
  %1002 = bitcast <4 x i32> %1001 to <8 x half>, !dbg !43
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %1003 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %886, !dbg !46
  %1004 = getelementptr i8, ptr addrspace(1) %1003, i64 192, !dbg !46
  %1005 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !46
  %1006 = zext i32 %1005 to i64, !dbg !46
  %1007 = inttoptr i64 %1006 to ptr addrspace(3), !dbg !46
  %address.local533 = getelementptr [2 x i8], ptr addrspace(1) %1004, i64 %address.local530, !dbg !46
  %1008 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !46
  %1009 = ptrtoint ptr addrspace(1) %address.local533 to i64, !dbg !46
  %1010 = sub i64 %1009, %1008, !dbg !46
  %dma.buffer.offset = trunc i64 %1010 to i32, !dbg !46
  %1011 = trunc i64 %1008 to i32, !dbg !46
  %1012 = insertelement <4 x i32> poison, i32 %1011, i64 0, !dbg !46
  %1013 = lshr i64 %1008, 32, !dbg !46
  %1014 = trunc i64 %1013 to i32, !dbg !46
  %1015 = insertelement <4 x i32> %1012, i32 %1014, i64 1, !dbg !46
  %1016 = insertelement <4 x i32> %1015, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc = insertelement <4 x i32> %1016, i32 131072, i64 3, !dbg !46
  %1017 = ptrtoint ptr addrspace(3) %1007 to i32, !dbg !46
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1017), !dbg !46
  %1018 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !46
  %1019 = shufflevector <8 x half> %996, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %1020 = extractvalue [2 x <8 x float>] %211, 0, !dbg !62
  %1021 = shufflevector <8 x float> %1020, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %1022 = fmul <4 x float> %1021, %707, !dbg !62
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1019, <4 x float> %1022) #9, !dbg !43
  %1024 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1024, <4 x float> %1023) #9, !dbg !43
  %1026 = shufflevector <8 x half> %1000, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1026, <4 x float> %1025) #9, !dbg !43
  %1028 = shufflevector <8 x half> %1002, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1028, <4 x float> %1027) #9, !dbg !43
  %1030 = shufflevector <8 x half> %996, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %1031 = shufflevector <8 x float> %1020, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %1032 = fmul <4 x float> %1031, %707, !dbg !62
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1030, <4 x float> %1032) #9, !dbg !43
  %1034 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1034, <4 x float> %1033) #9, !dbg !43
  %1036 = shufflevector <8 x half> %1000, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1036, <4 x float> %1035) #9, !dbg !43
  %1038 = shufflevector <8 x half> %1002, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1038, <4 x float> %1037) #9, !dbg !43
  %1040 = extractvalue [2 x <8 x float>] %211, 1, !dbg !62
  %1041 = shufflevector <8 x float> %1040, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %1042 = fmul <4 x float> %1041, %715, !dbg !62
  %1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1019, <4 x float> %1042) #9, !dbg !43
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1024, <4 x float> %1043) #9, !dbg !43
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1026, <4 x float> %1044) #9, !dbg !43
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1028, <4 x float> %1045) #9, !dbg !43
  %1047 = shufflevector <8 x float> %1040, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %1048 = fmul <4 x float> %1047, %715, !dbg !62
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1030, <4 x float> %1048) #9, !dbg !43
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1034, <4 x float> %1049) #9, !dbg !43
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1036, <4 x float> %1050) #9, !dbg !43
  %1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1038, <4 x float> %1051) #9, !dbg !43
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %1053 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !43
  %1054 = bitcast <4 x i32> %1053 to <8 x half>, !dbg !43
  %1055 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !43
  %1056 = bitcast <4 x i32> %1055 to <8 x half>, !dbg !43
  %1057 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !43
  %1058 = bitcast <4 x i32> %1057 to <8 x half>, !dbg !43
  %1059 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !43
  %1060 = bitcast <4 x i32> %1059 to <8 x half>, !dbg !43
  %1061 = shufflevector <8 x half> %1054, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %1062 = extractvalue [2 x <8 x float>] %212, 0, !dbg !62
  %1063 = shufflevector <8 x float> %1062, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %1064 = fmul <4 x float> %1063, %707, !dbg !62
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1061, <4 x float> %1064) #9, !dbg !43
  %1066 = shufflevector <8 x half> %1056, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1066, <4 x float> %1065) #9, !dbg !43
  %1068 = shufflevector <8 x half> %1058, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1068, <4 x float> %1067) #9, !dbg !43
  %1070 = shufflevector <8 x half> %1060, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %1071 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1070, <4 x float> %1069) #9, !dbg !43
  %1072 = shufflevector <8 x half> %1054, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %1073 = shufflevector <8 x float> %1062, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %1074 = fmul <4 x float> %1073, %707, !dbg !62
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1072, <4 x float> %1074) #9, !dbg !43
  %1076 = shufflevector <8 x half> %1056, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1076, <4 x float> %1075) #9, !dbg !43
  %1078 = shufflevector <8 x half> %1058, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1078, <4 x float> %1077) #9, !dbg !43
  %1080 = shufflevector <8 x half> %1060, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %1081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1080, <4 x float> %1079) #9, !dbg !43
  %1082 = extractvalue [2 x <8 x float>] %212, 1, !dbg !62
  %1083 = shufflevector <8 x float> %1082, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %1084 = fmul <4 x float> %1083, %715, !dbg !62
  %1085 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1061, <4 x float> %1084) #9, !dbg !43
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1066, <4 x float> %1085) #9, !dbg !43
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1068, <4 x float> %1086) #9, !dbg !43
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1070, <4 x float> %1087) #9, !dbg !43
  %1089 = shufflevector <8 x float> %1082, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %1090 = fmul <4 x float> %1089, %715, !dbg !62
  %1091 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1072, <4 x float> %1090) #9, !dbg !43
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1076, <4 x float> %1091) #9, !dbg !43
  %1093 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1078, <4 x float> %1092) #9, !dbg !43
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1080, <4 x float> %1093) #9, !dbg !43
  %1095 = shufflevector <4 x float> %913, <4 x float> %923, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1096 = insertvalue [2 x <8 x float>] poison, <8 x float> %1095, 0, !dbg !38
  %1097 = shufflevector <4 x float> %930, <4 x float> %936, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1098 = insertvalue [2 x <8 x float>] %1096, <8 x float> %1097, 1, !dbg !38
  %1099 = insertvalue [2 x <1 x float>] poison, <1 x float> %665, 0, !dbg !38
  %1100 = insertvalue [2 x <1 x float>] %1099, <1 x float> %692, 1, !dbg !38
  %1101 = shufflevector <4 x float> %971, <4 x float> %981, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1102 = insertvalue [2 x <8 x float>] poison, <8 x float> %1101, 0, !dbg !38
  %1103 = shufflevector <4 x float> %988, <4 x float> %994, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1104 = insertvalue [2 x <8 x float>] %1102, <8 x float> %1103, 1, !dbg !38
  %1105 = shufflevector <4 x float> %1029, <4 x float> %1039, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1106 = insertvalue [2 x <8 x float>] poison, <8 x float> %1105, 0, !dbg !38
  %1107 = shufflevector <4 x float> %1046, <4 x float> %1052, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1108 = insertvalue [2 x <8 x float>] %1106, <8 x float> %1107, 1, !dbg !38
  %1109 = shufflevector <4 x float> %1071, <4 x float> %1081, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1110 = insertvalue [2 x <8 x float>] poison, <8 x float> %1109, 0, !dbg !38
  %1111 = shufflevector <4 x float> %1088, <4 x float> %1094, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1112 = insertvalue [2 x <8 x float>] %1110, <8 x float> %1111, 1, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %1113 = add nuw nsw i64 %206, 1, !dbg !38
  br label %205, !dbg !38

1114:                                             ; preds = %205
  %1115 = and i32 %31, 16, !dbg !38
  %.not = icmp eq i32 %1115, 0, !dbg !38
  %1116 = and i32 %31, 32, !dbg !38
  %.not18 = icmp eq i32 %1116, 0, !dbg !38
  %1117 = extractvalue [2 x <8 x float>] %212, 0, !dbg !38
  %1118 = extractelement <8 x float> %1117, i64 0, !dbg !38
  %1119 = extractelement <8 x float> %1117, i64 1, !dbg !38
  %1120 = extractelement <8 x float> %1117, i64 2, !dbg !38
  %1121 = extractelement <8 x float> %1117, i64 3, !dbg !38
  %1122 = select i1 %.not, float %1119, float %1118, !dbg !38
  %1123 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !38
  %1124 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1123), !dbg !38
  %1125 = add nuw nsw i32 %1124, 64, !dbg !38
  %1126 = xor i32 %1124, 16, !dbg !38
  %1127 = and i32 %1125, 192, !dbg !38
  %1128 = icmp samesign ult i32 %1126, %1127, !dbg !38
  %1129 = select i1 %1128, i32 %1126, i32 %1124, !dbg !38
  %1130 = shl nuw nsw i32 %1129, 2, !dbg !38
  %1131 = bitcast float %1122 to i32, !dbg !38
  %1132 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1131), !dbg !38
  %1133 = bitcast i32 %1132 to float, !dbg !38
  %1134 = select i1 %.not, float %1118, float %1133, !dbg !38
  %1135 = select i1 %.not, float %1133, float %1119, !dbg !38
  %1136 = select i1 %.not, float %1121, float %1120, !dbg !38
  %1137 = bitcast float %1136 to i32, !dbg !38
  %1138 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1137), !dbg !38
  %1139 = bitcast i32 %1138 to float, !dbg !38
  %1140 = select i1 %.not, float %1120, float %1139, !dbg !38
  %1141 = select i1 %.not, float %1139, float %1121, !dbg !38
  %1142 = select i1 %.not18, float %1140, float %1134, !dbg !38
  %1143 = xor i32 %1124, 32, !dbg !38
  %1144 = icmp samesign ult i32 %1143, %1127, !dbg !38
  %1145 = select i1 %1144, i32 %1143, i32 %1124, !dbg !38
  %1146 = shl nuw nsw i32 %1145, 2, !dbg !38
  %1147 = bitcast float %1142 to i32, !dbg !38
  %1148 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1147), !dbg !38
  %1149 = bitcast i32 %1148 to float, !dbg !38
  %1150 = select i1 %.not18, float %1134, float %1149, !dbg !38
  %1151 = select i1 %.not18, float %1149, float %1140, !dbg !38
  %1152 = select i1 %.not18, float %1141, float %1135, !dbg !38
  %1153 = bitcast float %1152 to i32, !dbg !38
  %1154 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1153), !dbg !38
  %1155 = bitcast i32 %1154 to float, !dbg !38
  %1156 = select i1 %.not18, float %1135, float %1155, !dbg !38
  %1157 = select i1 %.not18, float %1155, float %1141, !dbg !38
  %1158 = extractvalue [2 x <8 x float>] %211, 0, !dbg !38
  %1159 = extractelement <8 x float> %1158, i64 0, !dbg !38
  %1160 = extractelement <8 x float> %1158, i64 1, !dbg !38
  %1161 = extractelement <8 x float> %1158, i64 2, !dbg !38
  %1162 = extractelement <8 x float> %1158, i64 3, !dbg !38
  %1163 = select i1 %.not, float %1160, float %1159, !dbg !38
  %1164 = bitcast float %1163 to i32, !dbg !38
  %1165 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1164), !dbg !38
  %1166 = bitcast i32 %1165 to float, !dbg !38
  %1167 = select i1 %.not, float %1159, float %1166, !dbg !38
  %1168 = select i1 %.not, float %1166, float %1160, !dbg !38
  %1169 = select i1 %.not, float %1162, float %1161, !dbg !38
  %1170 = bitcast float %1169 to i32, !dbg !38
  %1171 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1170), !dbg !38
  %1172 = bitcast i32 %1171 to float, !dbg !38
  %1173 = select i1 %.not, float %1161, float %1172, !dbg !38
  %1174 = select i1 %.not, float %1172, float %1162, !dbg !38
  %1175 = select i1 %.not18, float %1173, float %1167, !dbg !38
  %1176 = bitcast float %1175 to i32, !dbg !38
  %1177 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1176), !dbg !38
  %1178 = bitcast i32 %1177 to float, !dbg !38
  %1179 = select i1 %.not18, float %1167, float %1178, !dbg !38
  %1180 = select i1 %.not18, float %1178, float %1173, !dbg !38
  %1181 = select i1 %.not18, float %1174, float %1168, !dbg !38
  %1182 = bitcast float %1181 to i32, !dbg !38
  %1183 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1182), !dbg !38
  %1184 = bitcast i32 %1183 to float, !dbg !38
  %1185 = select i1 %.not18, float %1168, float %1184, !dbg !38
  %1186 = select i1 %.not18, float %1184, float %1174, !dbg !38
  %1187 = extractvalue [2 x <8 x float>] %210, 0, !dbg !38
  %1188 = extractelement <8 x float> %1187, i64 0, !dbg !38
  %1189 = extractelement <8 x float> %1187, i64 1, !dbg !38
  %1190 = extractelement <8 x float> %1187, i64 2, !dbg !38
  %1191 = extractelement <8 x float> %1187, i64 3, !dbg !38
  %1192 = select i1 %.not, float %1189, float %1188, !dbg !38
  %1193 = bitcast float %1192 to i32, !dbg !38
  %1194 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1193), !dbg !38
  %1195 = bitcast i32 %1194 to float, !dbg !38
  %1196 = select i1 %.not, float %1188, float %1195, !dbg !38
  %1197 = select i1 %.not, float %1195, float %1189, !dbg !38
  %1198 = select i1 %.not, float %1191, float %1190, !dbg !38
  %1199 = bitcast float %1198 to i32, !dbg !38
  %1200 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1199), !dbg !38
  %1201 = bitcast i32 %1200 to float, !dbg !38
  %1202 = select i1 %.not, float %1190, float %1201, !dbg !38
  %1203 = select i1 %.not, float %1201, float %1191, !dbg !38
  %1204 = select i1 %.not18, float %1202, float %1196, !dbg !38
  %1205 = bitcast float %1204 to i32, !dbg !38
  %1206 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1205), !dbg !38
  %1207 = bitcast i32 %1206 to float, !dbg !38
  %1208 = select i1 %.not18, float %1196, float %1207, !dbg !38
  %1209 = select i1 %.not18, float %1207, float %1202, !dbg !38
  %1210 = select i1 %.not18, float %1203, float %1197, !dbg !38
  %1211 = bitcast float %1210 to i32, !dbg !38
  %1212 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1211), !dbg !38
  %1213 = bitcast i32 %1212 to float, !dbg !38
  %1214 = select i1 %.not18, float %1197, float %1213, !dbg !38
  %1215 = select i1 %.not18, float %1213, float %1203, !dbg !38
  %1216 = extractvalue [2 x <8 x float>] %207, 0, !dbg !38
  %1217 = extractelement <8 x float> %1216, i64 0, !dbg !38
  %1218 = extractelement <8 x float> %1216, i64 1, !dbg !38
  %1219 = extractelement <8 x float> %1216, i64 2, !dbg !38
  %1220 = extractelement <8 x float> %1216, i64 3, !dbg !38
  %1221 = select i1 %.not, float %1218, float %1217, !dbg !38
  %1222 = bitcast float %1221 to i32, !dbg !38
  %1223 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1222), !dbg !38
  %1224 = bitcast i32 %1223 to float, !dbg !38
  %1225 = select i1 %.not, float %1217, float %1224, !dbg !38
  %1226 = select i1 %.not, float %1224, float %1218, !dbg !38
  %1227 = select i1 %.not, float %1220, float %1219, !dbg !38
  %1228 = bitcast float %1227 to i32, !dbg !38
  %1229 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1228), !dbg !38
  %1230 = bitcast i32 %1229 to float, !dbg !38
  %1231 = select i1 %.not, float %1219, float %1230, !dbg !38
  %1232 = select i1 %.not, float %1230, float %1220, !dbg !38
  %1233 = select i1 %.not18, float %1231, float %1225, !dbg !38
  %1234 = bitcast float %1233 to i32, !dbg !38
  %1235 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1234), !dbg !38
  %1236 = select i1 %.not18, float %1232, float %1226, !dbg !38
  %1237 = bitcast float %1236 to i32, !dbg !38
  %1238 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1237), !dbg !38
  %1239 = extractvalue [2 x <1 x float>] %208, 0, !dbg !47
  %1240 = shufflevector <1 x float> %1239, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !47
  %1241 = extractelement <8 x float> %1117, i64 4, !dbg !38
  %1242 = extractelement <8 x float> %1117, i64 5, !dbg !38
  %1243 = extractelement <8 x float> %1117, i64 6, !dbg !38
  %1244 = extractelement <8 x float> %1117, i64 7, !dbg !38
  %1245 = select i1 %.not, float %1242, float %1241, !dbg !38
  %1246 = bitcast float %1245 to i32, !dbg !38
  %1247 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1246), !dbg !38
  %1248 = bitcast i32 %1247 to float, !dbg !38
  %1249 = select i1 %.not, float %1241, float %1248, !dbg !38
  %1250 = select i1 %.not, float %1248, float %1242, !dbg !38
  %1251 = select i1 %.not, float %1244, float %1243, !dbg !38
  %1252 = bitcast float %1251 to i32, !dbg !38
  %1253 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1252), !dbg !38
  %1254 = bitcast i32 %1253 to float, !dbg !38
  %1255 = select i1 %.not, float %1243, float %1254, !dbg !38
  %1256 = select i1 %.not, float %1254, float %1244, !dbg !38
  %1257 = select i1 %.not18, float %1255, float %1249, !dbg !38
  %1258 = bitcast float %1257 to i32, !dbg !38
  %1259 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1258), !dbg !38
  %1260 = bitcast i32 %1259 to float, !dbg !38
  %1261 = select i1 %.not18, float %1249, float %1260, !dbg !38
  %1262 = select i1 %.not18, float %1260, float %1255, !dbg !38
  %1263 = select i1 %.not18, float %1256, float %1250, !dbg !38
  %1264 = bitcast float %1263 to i32, !dbg !38
  %1265 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1264), !dbg !38
  %1266 = bitcast i32 %1265 to float, !dbg !38
  %1267 = select i1 %.not18, float %1250, float %1266, !dbg !38
  %1268 = select i1 %.not18, float %1266, float %1256, !dbg !38
  %1269 = extractelement <8 x float> %1158, i64 4, !dbg !38
  %1270 = extractelement <8 x float> %1158, i64 5, !dbg !38
  %1271 = extractelement <8 x float> %1158, i64 6, !dbg !38
  %1272 = extractelement <8 x float> %1158, i64 7, !dbg !38
  %1273 = select i1 %.not, float %1270, float %1269, !dbg !38
  %1274 = bitcast float %1273 to i32, !dbg !38
  %1275 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1274), !dbg !38
  %1276 = bitcast i32 %1275 to float, !dbg !38
  %1277 = select i1 %.not, float %1269, float %1276, !dbg !38
  %1278 = select i1 %.not, float %1276, float %1270, !dbg !38
  %1279 = select i1 %.not, float %1272, float %1271, !dbg !38
  %1280 = bitcast float %1279 to i32, !dbg !38
  %1281 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1280), !dbg !38
  %1282 = bitcast i32 %1281 to float, !dbg !38
  %1283 = select i1 %.not, float %1271, float %1282, !dbg !38
  %1284 = select i1 %.not, float %1282, float %1272, !dbg !38
  %1285 = select i1 %.not18, float %1283, float %1277, !dbg !38
  %1286 = bitcast float %1285 to i32, !dbg !38
  %1287 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1286), !dbg !38
  %1288 = bitcast i32 %1287 to float, !dbg !38
  %1289 = select i1 %.not18, float %1277, float %1288, !dbg !38
  %1290 = select i1 %.not18, float %1288, float %1283, !dbg !38
  %1291 = select i1 %.not18, float %1284, float %1278, !dbg !38
  %1292 = bitcast float %1291 to i32, !dbg !38
  %1293 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1292), !dbg !38
  %1294 = bitcast i32 %1293 to float, !dbg !38
  %1295 = select i1 %.not18, float %1278, float %1294, !dbg !38
  %1296 = select i1 %.not18, float %1294, float %1284, !dbg !38
  %1297 = extractelement <8 x float> %1187, i64 4, !dbg !38
  %1298 = extractelement <8 x float> %1187, i64 5, !dbg !38
  %1299 = extractelement <8 x float> %1187, i64 6, !dbg !38
  %1300 = extractelement <8 x float> %1187, i64 7, !dbg !38
  %1301 = select i1 %.not, float %1298, float %1297, !dbg !38
  %1302 = bitcast float %1301 to i32, !dbg !38
  %1303 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1302), !dbg !38
  %1304 = bitcast i32 %1303 to float, !dbg !38
  %1305 = select i1 %.not, float %1297, float %1304, !dbg !38
  %1306 = select i1 %.not, float %1304, float %1298, !dbg !38
  %1307 = select i1 %.not, float %1300, float %1299, !dbg !38
  %1308 = bitcast float %1307 to i32, !dbg !38
  %1309 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1308), !dbg !38
  %1310 = bitcast i32 %1309 to float, !dbg !38
  %1311 = select i1 %.not, float %1299, float %1310, !dbg !38
  %1312 = select i1 %.not, float %1310, float %1300, !dbg !38
  %1313 = select i1 %.not18, float %1311, float %1305, !dbg !38
  %1314 = bitcast float %1313 to i32, !dbg !38
  %1315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1314), !dbg !38
  %1316 = bitcast i32 %1315 to float, !dbg !38
  %1317 = select i1 %.not18, float %1305, float %1316, !dbg !38
  %1318 = select i1 %.not18, float %1316, float %1311, !dbg !38
  %1319 = select i1 %.not18, float %1312, float %1306, !dbg !38
  %1320 = bitcast float %1319 to i32, !dbg !38
  %1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1320), !dbg !38
  %1322 = bitcast i32 %1321 to float, !dbg !38
  %1323 = select i1 %.not18, float %1306, float %1322, !dbg !38
  %1324 = select i1 %.not18, float %1322, float %1312, !dbg !38
  %1325 = extractelement <8 x float> %1216, i64 4, !dbg !38
  %1326 = extractelement <8 x float> %1216, i64 5, !dbg !38
  %1327 = extractelement <8 x float> %1216, i64 6, !dbg !38
  %1328 = extractelement <8 x float> %1216, i64 7, !dbg !38
  %1329 = select i1 %.not, float %1326, float %1325, !dbg !38
  %1330 = bitcast float %1329 to i32, !dbg !38
  %1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1330), !dbg !38
  %1332 = bitcast i32 %1331 to float, !dbg !38
  %1333 = select i1 %.not, float %1325, float %1332, !dbg !38
  %1334 = select i1 %.not, float %1332, float %1326, !dbg !38
  %1335 = select i1 %.not, float %1328, float %1327, !dbg !38
  %1336 = bitcast float %1335 to i32, !dbg !38
  %1337 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1336), !dbg !38
  %1338 = bitcast i32 %1337 to float, !dbg !38
  %1339 = select i1 %.not, float %1327, float %1338, !dbg !38
  %1340 = select i1 %.not, float %1338, float %1328, !dbg !38
  %1341 = select i1 %.not18, float %1339, float %1333, !dbg !38
  %1342 = bitcast float %1341 to i32, !dbg !38
  %1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1342), !dbg !38
  %1344 = select i1 %.not18, float %1340, float %1334, !dbg !38
  %1345 = bitcast float %1344 to i32, !dbg !38
  %1346 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1345), !dbg !38
  %1347 = extractvalue [2 x <8 x float>] %212, 1, !dbg !38
  %1348 = extractelement <8 x float> %1347, i64 0, !dbg !38
  %1349 = extractelement <8 x float> %1347, i64 1, !dbg !38
  %1350 = extractelement <8 x float> %1347, i64 2, !dbg !38
  %1351 = extractelement <8 x float> %1347, i64 3, !dbg !38
  %1352 = select i1 %.not, float %1349, float %1348, !dbg !38
  %1353 = bitcast float %1352 to i32, !dbg !38
  %1354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1353), !dbg !38
  %1355 = bitcast i32 %1354 to float, !dbg !38
  %1356 = select i1 %.not, float %1348, float %1355, !dbg !38
  %1357 = select i1 %.not, float %1355, float %1349, !dbg !38
  %1358 = select i1 %.not, float %1351, float %1350, !dbg !38
  %1359 = bitcast float %1358 to i32, !dbg !38
  %1360 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1359), !dbg !38
  %1361 = bitcast i32 %1360 to float, !dbg !38
  %1362 = select i1 %.not, float %1350, float %1361, !dbg !38
  %1363 = select i1 %.not, float %1361, float %1351, !dbg !38
  %1364 = select i1 %.not18, float %1362, float %1356, !dbg !38
  %1365 = bitcast float %1364 to i32, !dbg !38
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1365), !dbg !38
  %1367 = bitcast i32 %1366 to float, !dbg !38
  %1368 = select i1 %.not18, float %1356, float %1367, !dbg !38
  %1369 = select i1 %.not18, float %1367, float %1362, !dbg !38
  %1370 = select i1 %.not18, float %1363, float %1357, !dbg !38
  %1371 = bitcast float %1370 to i32, !dbg !38
  %1372 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1371), !dbg !38
  %1373 = bitcast i32 %1372 to float, !dbg !38
  %1374 = select i1 %.not18, float %1357, float %1373, !dbg !38
  %1375 = select i1 %.not18, float %1373, float %1363, !dbg !38
  %1376 = extractvalue [2 x <8 x float>] %211, 1, !dbg !38
  %1377 = extractelement <8 x float> %1376, i64 0, !dbg !38
  %1378 = extractelement <8 x float> %1376, i64 1, !dbg !38
  %1379 = extractelement <8 x float> %1376, i64 2, !dbg !38
  %1380 = extractelement <8 x float> %1376, i64 3, !dbg !38
  %1381 = select i1 %.not, float %1378, float %1377, !dbg !38
  %1382 = bitcast float %1381 to i32, !dbg !38
  %1383 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1382), !dbg !38
  %1384 = bitcast i32 %1383 to float, !dbg !38
  %1385 = select i1 %.not, float %1377, float %1384, !dbg !38
  %1386 = select i1 %.not, float %1384, float %1378, !dbg !38
  %1387 = select i1 %.not, float %1380, float %1379, !dbg !38
  %1388 = bitcast float %1387 to i32, !dbg !38
  %1389 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1388), !dbg !38
  %1390 = bitcast i32 %1389 to float, !dbg !38
  %1391 = select i1 %.not, float %1379, float %1390, !dbg !38
  %1392 = select i1 %.not, float %1390, float %1380, !dbg !38
  %1393 = select i1 %.not18, float %1391, float %1385, !dbg !38
  %1394 = bitcast float %1393 to i32, !dbg !38
  %1395 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1394), !dbg !38
  %1396 = bitcast i32 %1395 to float, !dbg !38
  %1397 = select i1 %.not18, float %1385, float %1396, !dbg !38
  %1398 = select i1 %.not18, float %1396, float %1391, !dbg !38
  %1399 = select i1 %.not18, float %1392, float %1386, !dbg !38
  %1400 = bitcast float %1399 to i32, !dbg !38
  %1401 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1400), !dbg !38
  %1402 = bitcast i32 %1401 to float, !dbg !38
  %1403 = select i1 %.not18, float %1386, float %1402, !dbg !38
  %1404 = select i1 %.not18, float %1402, float %1392, !dbg !38
  %1405 = extractvalue [2 x <8 x float>] %210, 1, !dbg !38
  %1406 = extractelement <8 x float> %1405, i64 0, !dbg !38
  %1407 = extractelement <8 x float> %1405, i64 1, !dbg !38
  %1408 = extractelement <8 x float> %1405, i64 2, !dbg !38
  %1409 = extractelement <8 x float> %1405, i64 3, !dbg !38
  %1410 = select i1 %.not, float %1407, float %1406, !dbg !38
  %1411 = bitcast float %1410 to i32, !dbg !38
  %1412 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1411), !dbg !38
  %1413 = bitcast i32 %1412 to float, !dbg !38
  %1414 = select i1 %.not, float %1406, float %1413, !dbg !38
  %1415 = select i1 %.not, float %1413, float %1407, !dbg !38
  %1416 = select i1 %.not, float %1409, float %1408, !dbg !38
  %1417 = bitcast float %1416 to i32, !dbg !38
  %1418 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1417), !dbg !38
  %1419 = bitcast i32 %1418 to float, !dbg !38
  %1420 = select i1 %.not, float %1408, float %1419, !dbg !38
  %1421 = select i1 %.not, float %1419, float %1409, !dbg !38
  %1422 = select i1 %.not18, float %1420, float %1414, !dbg !38
  %1423 = bitcast float %1422 to i32, !dbg !38
  %1424 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1423), !dbg !38
  %1425 = bitcast i32 %1424 to float, !dbg !38
  %1426 = select i1 %.not18, float %1414, float %1425, !dbg !38
  %1427 = select i1 %.not18, float %1425, float %1420, !dbg !38
  %1428 = select i1 %.not18, float %1421, float %1415, !dbg !38
  %1429 = bitcast float %1428 to i32, !dbg !38
  %1430 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1429), !dbg !38
  %1431 = bitcast i32 %1430 to float, !dbg !38
  %1432 = select i1 %.not18, float %1415, float %1431, !dbg !38
  %1433 = select i1 %.not18, float %1431, float %1421, !dbg !38
  %1434 = extractvalue [2 x <8 x float>] %207, 1, !dbg !38
  %1435 = extractelement <8 x float> %1434, i64 0, !dbg !38
  %1436 = extractelement <8 x float> %1434, i64 1, !dbg !38
  %1437 = extractelement <8 x float> %1434, i64 2, !dbg !38
  %1438 = extractelement <8 x float> %1434, i64 3, !dbg !38
  %1439 = select i1 %.not, float %1436, float %1435, !dbg !38
  %1440 = bitcast float %1439 to i32, !dbg !38
  %1441 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1440), !dbg !38
  %1442 = bitcast i32 %1441 to float, !dbg !38
  %1443 = select i1 %.not, float %1435, float %1442, !dbg !38
  %1444 = select i1 %.not, float %1442, float %1436, !dbg !38
  %1445 = select i1 %.not, float %1438, float %1437, !dbg !38
  %1446 = bitcast float %1445 to i32, !dbg !38
  %1447 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1446), !dbg !38
  %1448 = bitcast i32 %1447 to float, !dbg !38
  %1449 = select i1 %.not, float %1437, float %1448, !dbg !38
  %1450 = select i1 %.not, float %1448, float %1438, !dbg !38
  %1451 = select i1 %.not18, float %1449, float %1443, !dbg !38
  %1452 = bitcast float %1451 to i32, !dbg !38
  %1453 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1452), !dbg !38
  %1454 = select i1 %.not18, float %1450, float %1444, !dbg !38
  %1455 = bitcast float %1454 to i32, !dbg !38
  %1456 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1455), !dbg !38
  %1457 = extractvalue [2 x <1 x float>] %208, 1, !dbg !47
  %1458 = shufflevector <1 x float> %1457, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !47
  %1459 = extractelement <8 x float> %1347, i64 4, !dbg !38
  %1460 = extractelement <8 x float> %1347, i64 5, !dbg !38
  %1461 = extractelement <8 x float> %1347, i64 6, !dbg !38
  %1462 = extractelement <8 x float> %1347, i64 7, !dbg !38
  %1463 = select i1 %.not, float %1460, float %1459, !dbg !38
  %1464 = bitcast float %1463 to i32, !dbg !38
  %1465 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1464), !dbg !38
  %1466 = bitcast i32 %1465 to float, !dbg !38
  %1467 = select i1 %.not, float %1459, float %1466, !dbg !38
  %1468 = select i1 %.not, float %1466, float %1460, !dbg !38
  %1469 = select i1 %.not, float %1462, float %1461, !dbg !38
  %1470 = bitcast float %1469 to i32, !dbg !38
  %1471 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1470), !dbg !38
  %1472 = bitcast i32 %1471 to float, !dbg !38
  %1473 = select i1 %.not, float %1461, float %1472, !dbg !38
  %1474 = select i1 %.not, float %1472, float %1462, !dbg !38
  %1475 = select i1 %.not18, float %1473, float %1467, !dbg !38
  %1476 = bitcast float %1475 to i32, !dbg !38
  %1477 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1476), !dbg !38
  %1478 = bitcast i32 %1477 to float, !dbg !38
  %1479 = select i1 %.not18, float %1467, float %1478, !dbg !38
  %1480 = select i1 %.not18, float %1478, float %1473, !dbg !38
  %1481 = select i1 %.not18, float %1474, float %1468, !dbg !38
  %1482 = bitcast float %1481 to i32, !dbg !38
  %1483 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1482), !dbg !38
  %1484 = bitcast i32 %1483 to float, !dbg !38
  %1485 = select i1 %.not18, float %1468, float %1484, !dbg !38
  %1486 = select i1 %.not18, float %1484, float %1474, !dbg !38
  %1487 = extractelement <8 x float> %1376, i64 4, !dbg !38
  %1488 = extractelement <8 x float> %1376, i64 5, !dbg !38
  %1489 = extractelement <8 x float> %1376, i64 6, !dbg !38
  %1490 = extractelement <8 x float> %1376, i64 7, !dbg !38
  %1491 = select i1 %.not, float %1488, float %1487, !dbg !38
  %1492 = bitcast float %1491 to i32, !dbg !38
  %1493 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1492), !dbg !38
  %1494 = bitcast i32 %1493 to float, !dbg !38
  %1495 = select i1 %.not, float %1487, float %1494, !dbg !38
  %1496 = select i1 %.not, float %1494, float %1488, !dbg !38
  %1497 = select i1 %.not, float %1490, float %1489, !dbg !38
  %1498 = bitcast float %1497 to i32, !dbg !38
  %1499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1498), !dbg !38
  %1500 = bitcast i32 %1499 to float, !dbg !38
  %1501 = select i1 %.not, float %1489, float %1500, !dbg !38
  %1502 = select i1 %.not, float %1500, float %1490, !dbg !38
  %1503 = select i1 %.not18, float %1501, float %1495, !dbg !38
  %1504 = bitcast float %1503 to i32, !dbg !38
  %1505 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1504), !dbg !38
  %1506 = bitcast i32 %1505 to float, !dbg !38
  %1507 = select i1 %.not18, float %1495, float %1506, !dbg !38
  %1508 = select i1 %.not18, float %1506, float %1501, !dbg !38
  %1509 = select i1 %.not18, float %1502, float %1496, !dbg !38
  %1510 = bitcast float %1509 to i32, !dbg !38
  %1511 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1510), !dbg !38
  %1512 = bitcast i32 %1511 to float, !dbg !38
  %1513 = select i1 %.not18, float %1496, float %1512, !dbg !38
  %1514 = select i1 %.not18, float %1512, float %1502, !dbg !38
  %1515 = extractelement <8 x float> %1405, i64 4, !dbg !38
  %1516 = extractelement <8 x float> %1405, i64 5, !dbg !38
  %1517 = extractelement <8 x float> %1405, i64 6, !dbg !38
  %1518 = extractelement <8 x float> %1405, i64 7, !dbg !38
  %1519 = select i1 %.not, float %1516, float %1515, !dbg !38
  %1520 = bitcast float %1519 to i32, !dbg !38
  %1521 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1520), !dbg !38
  %1522 = bitcast i32 %1521 to float, !dbg !38
  %1523 = select i1 %.not, float %1515, float %1522, !dbg !38
  %1524 = select i1 %.not, float %1522, float %1516, !dbg !38
  %1525 = select i1 %.not, float %1518, float %1517, !dbg !38
  %1526 = bitcast float %1525 to i32, !dbg !38
  %1527 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1526), !dbg !38
  %1528 = bitcast i32 %1527 to float, !dbg !38
  %1529 = select i1 %.not, float %1517, float %1528, !dbg !38
  %1530 = select i1 %.not, float %1528, float %1518, !dbg !38
  %1531 = select i1 %.not18, float %1529, float %1523, !dbg !38
  %1532 = bitcast float %1531 to i32, !dbg !38
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1532), !dbg !38
  %1534 = bitcast i32 %1533 to float, !dbg !38
  %1535 = select i1 %.not18, float %1523, float %1534, !dbg !38
  %1536 = select i1 %.not18, float %1534, float %1529, !dbg !38
  %1537 = select i1 %.not18, float %1530, float %1524, !dbg !38
  %1538 = bitcast float %1537 to i32, !dbg !38
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1538), !dbg !38
  %1540 = bitcast i32 %1539 to float, !dbg !38
  %1541 = select i1 %.not18, float %1524, float %1540, !dbg !38
  %1542 = select i1 %.not18, float %1540, float %1530, !dbg !38
  %1543 = extractelement <8 x float> %1434, i64 4, !dbg !38
  %1544 = extractelement <8 x float> %1434, i64 5, !dbg !38
  %1545 = extractelement <8 x float> %1434, i64 6, !dbg !38
  %1546 = extractelement <8 x float> %1434, i64 7, !dbg !38
  %1547 = select i1 %.not, float %1544, float %1543, !dbg !38
  %1548 = bitcast float %1547 to i32, !dbg !38
  %1549 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1548), !dbg !38
  %1550 = bitcast i32 %1549 to float, !dbg !38
  %1551 = select i1 %.not, float %1543, float %1550, !dbg !38
  %1552 = select i1 %.not, float %1550, float %1544, !dbg !38
  %1553 = select i1 %.not, float %1546, float %1545, !dbg !38
  %1554 = bitcast float %1553 to i32, !dbg !38
  %1555 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1130, i32 %1554), !dbg !38
  %1556 = bitcast i32 %1555 to float, !dbg !38
  %1557 = select i1 %.not, float %1545, float %1556, !dbg !38
  %1558 = select i1 %.not, float %1556, float %1546, !dbg !38
  %1559 = select i1 %.not18, float %1557, float %1551, !dbg !38
  %1560 = bitcast float %1559 to i32, !dbg !38
  %1561 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1560), !dbg !38
  %1562 = select i1 %.not18, float %1558, float %1552, !dbg !38
  %1563 = bitcast float %1562 to i32, !dbg !38
  %1564 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1146, i32 %1563), !dbg !38
  br i1 true, label %1565, label %1629, !dbg !49

1565:                                             ; preds = %1114
  %1566 = bitcast i32 %1561 to float, !dbg !38
  %1567 = select i1 %.not18, float %1551, float %1566, !dbg !38
  %1568 = insertelement <4 x float> poison, float %1567, i64 0, !dbg !47
  %1569 = bitcast i32 %1564 to float, !dbg !38
  %1570 = select i1 %.not18, float %1552, float %1569, !dbg !38
  %1571 = insertelement <4 x float> %1568, float %1570, i64 1, !dbg !47
  %1572 = select i1 %.not18, float %1566, float %1557, !dbg !38
  %1573 = insertelement <4 x float> %1571, float %1572, i64 2, !dbg !47
  %1574 = select i1 %.not18, float %1569, float %1558, !dbg !38
  %1575 = insertelement <4 x float> %1573, float %1574, i64 3, !dbg !47
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1458, !dbg !47
  %normalized = fmul <4 x float> %1575, %row.reciprocal, !dbg !47
  %1576 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !48
  %1577 = extractelement <4 x half> %1576, i64 3, !dbg !48
  %1578 = extractelement <4 x half> %1576, i64 2, !dbg !48
  %1579 = extractelement <4 x half> %1576, i64 1, !dbg !48
  %1580 = extractelement <4 x half> %1576, i64 0, !dbg !48
  %1581 = bitcast i32 %1453 to float, !dbg !38
  %1582 = select i1 %.not18, float %1443, float %1581, !dbg !38
  %1583 = insertelement <4 x float> poison, float %1582, i64 0, !dbg !47
  %1584 = bitcast i32 %1456 to float, !dbg !38
  %1585 = select i1 %.not18, float %1444, float %1584, !dbg !38
  %1586 = insertelement <4 x float> %1583, float %1585, i64 1, !dbg !47
  %1587 = select i1 %.not18, float %1581, float %1449, !dbg !38
  %1588 = insertelement <4 x float> %1586, float %1587, i64 2, !dbg !47
  %1589 = select i1 %.not18, float %1584, float %1450, !dbg !38
  %1590 = insertelement <4 x float> %1588, float %1589, i64 3, !dbg !47
  %normalized54 = fmul <4 x float> %1590, %row.reciprocal, !dbg !47
  %1591 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !48
  %1592 = extractelement <4 x half> %1591, i64 3, !dbg !48
  %1593 = extractelement <4 x half> %1591, i64 2, !dbg !48
  %1594 = extractelement <4 x half> %1591, i64 1, !dbg !48
  %1595 = extractelement <4 x half> %1591, i64 0, !dbg !48
  %1596 = bitcast i32 %1343 to float, !dbg !38
  %1597 = select i1 %.not18, float %1333, float %1596, !dbg !38
  %1598 = insertelement <4 x float> poison, float %1597, i64 0, !dbg !47
  %1599 = bitcast i32 %1346 to float, !dbg !38
  %1600 = select i1 %.not18, float %1334, float %1599, !dbg !38
  %1601 = insertelement <4 x float> %1598, float %1600, i64 1, !dbg !47
  %1602 = select i1 %.not18, float %1596, float %1339, !dbg !38
  %1603 = insertelement <4 x float> %1601, float %1602, i64 2, !dbg !47
  %1604 = select i1 %.not18, float %1599, float %1340, !dbg !38
  %1605 = insertelement <4 x float> %1603, float %1604, i64 3, !dbg !47
  %row.reciprocal55 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1240, !dbg !47
  %normalized56 = fmul <4 x float> %1605, %row.reciprocal55, !dbg !47
  %1606 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !48
  %1607 = extractelement <4 x half> %1606, i64 3, !dbg !48
  %1608 = extractelement <4 x half> %1606, i64 2, !dbg !48
  %1609 = extractelement <4 x half> %1606, i64 1, !dbg !48
  %1610 = extractelement <4 x half> %1606, i64 0, !dbg !48
  %1611 = bitcast i32 %1235 to float, !dbg !38
  %1612 = select i1 %.not18, float %1225, float %1611, !dbg !38
  %1613 = insertelement <4 x float> poison, float %1612, i64 0, !dbg !47
  %1614 = bitcast i32 %1238 to float, !dbg !38
  %1615 = select i1 %.not18, float %1226, float %1614, !dbg !38
  %1616 = insertelement <4 x float> %1613, float %1615, i64 1, !dbg !47
  %1617 = select i1 %.not18, float %1611, float %1231, !dbg !38
  %1618 = insertelement <4 x float> %1616, float %1617, i64 2, !dbg !47
  %1619 = select i1 %.not18, float %1614, float %1232, !dbg !38
  %1620 = insertelement <4 x float> %1618, float %1619, i64 3, !dbg !47
  %normalized57 = fmul <4 x float> %1620, %row.reciprocal55, !dbg !47
  %1621 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !48
  %1622 = extractelement <4 x half> %1621, i64 3, !dbg !48
  %1623 = extractelement <4 x half> %1621, i64 2, !dbg !48
  %1624 = extractelement <4 x half> %1621, i64 1, !dbg !48
  %1625 = extractelement <4 x half> %1621, i64 0, !dbg !48
  %1626 = add nsw i64 %16, %22, !dbg !49
  %1627 = shl i64 %1626, 19, !dbg !49
  %1628 = and i64 %1627, 16252928, !dbg !49
  %address.thread534 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !49, !range !65, !frisk.register_lifetimes !33
  %address.local535 = zext nneg i32 %address.thread534 to i64, !dbg !40
  %address.local536 = add i64 %25, %address.local535, !dbg !49
  %address.local537 = lshr i64 %address.local535, 6, !dbg !39
  %address.local538 = shl nuw nsw i64 %address.local537, 4, !dbg !49
  %address.local539 = add i64 %address.local536, %address.local538, !dbg !49
  %address.local540 = and i64 %address.local535, 240, !dbg !49
  %address.local541 = sub i64 %address.local539, %address.local540, !dbg !49
  %address.local542 = add i64 %address.local541, %26, !dbg !49
  %address.local543 = shl nuw nsw i64 %address.local542, 7, !dbg !49
  %address.local544 = add nuw nsw i64 %1628, %address.local543, !dbg !49
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local546 = lshr i64 %address.local535, 2, !dbg !39
  %address.local547 = and i64 %address.local546, 12, !dbg !39
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local547, !dbg !49
  store half %1625, ptr addrspace(1) %address.local548, align 2, !dbg !49
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local550 = or disjoint i64 %address.local547, 1, !dbg !49
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local549, i64 %address.local550, !dbg !49
  store half %1624, ptr addrspace(1) %address.local551, align 2, !dbg !49
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local553 = or disjoint i64 %address.local547, 2, !dbg !49
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local552, i64 %address.local553, !dbg !49
  store half %1623, ptr addrspace(1) %address.local554, align 2, !dbg !49
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local556 = or disjoint i64 %address.local547, 3, !dbg !49
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local555, i64 %address.local556, !dbg !49
  store half %1622, ptr addrspace(1) %address.local557, align 2, !dbg !49
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local559 = or disjoint i64 %address.local547, 16, !dbg !49
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local558, i64 %address.local559, !dbg !49
  store half %1610, ptr addrspace(1) %address.local560, align 2, !dbg !49
  %address.local561 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local562 = or disjoint i64 %address.local547, 17, !dbg !49
  %address.local563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local561, i64 %address.local562, !dbg !49
  store half %1609, ptr addrspace(1) %address.local563, align 2, !dbg !49
  %address.local564 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local565 = or disjoint i64 %address.local547, 18, !dbg !49
  %address.local566 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local564, i64 %address.local565, !dbg !49
  store half %1608, ptr addrspace(1) %address.local566, align 2, !dbg !49
  %address.local567 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local568 = or disjoint i64 %address.local547, 19, !dbg !49
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local567, i64 %address.local568, !dbg !49
  store half %1607, ptr addrspace(1) %address.local569, align 2, !dbg !49
  %address.local570 = shl i64 %address.local542, 7, !dbg !49
  %address.local571 = add i64 %address.local570, 8192, !dbg !49
  %address.local572 = add nuw nsw i64 %1628, %address.local571, !dbg !49
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local573, i64 %address.local547, !dbg !49
  store half %1595, ptr addrspace(1) %address.local574, align 2, !dbg !49
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local550, !dbg !49
  store half %1594, ptr addrspace(1) %address.local576, align 2, !dbg !49
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local553, !dbg !49
  store half %1593, ptr addrspace(1) %address.local578, align 2, !dbg !49
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local579, i64 %address.local556, !dbg !49
  store half %1592, ptr addrspace(1) %address.local580, align 2, !dbg !49
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local559, !dbg !49
  store half %1580, ptr addrspace(1) %address.local582, align 2, !dbg !49
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local562, !dbg !49
  store half %1579, ptr addrspace(1) %address.local584, align 2, !dbg !49
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local585, i64 %address.local565, !dbg !49
  store half %1578, ptr addrspace(1) %address.local586, align 2, !dbg !49
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local587, i64 %address.local568, !dbg !49
  store half %1577, ptr addrspace(1) %address.local588, align 2, !dbg !49
  br label %1629, !dbg !49

1629:                                             ; preds = %1565, %1114
  br i1 true, label %1630, label %1668, !dbg !49

1630:                                             ; preds = %1629
  %1631 = insertelement <4 x float> poison, float %1535, i64 0, !dbg !47
  %1632 = insertelement <4 x float> %1631, float %1541, i64 1, !dbg !47
  %1633 = insertelement <4 x float> %1632, float %1536, i64 2, !dbg !47
  %1634 = insertelement <4 x float> %1633, float %1542, i64 3, !dbg !47
  %row.reciprocal58 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1458, !dbg !47
  %normalized59 = fmul <4 x float> %1634, %row.reciprocal58, !dbg !47
  %1635 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !48
  %1636 = extractelement <4 x half> %1635, i64 3, !dbg !48
  %1637 = extractelement <4 x half> %1635, i64 2, !dbg !48
  %1638 = extractelement <4 x half> %1635, i64 1, !dbg !48
  %1639 = extractelement <4 x half> %1635, i64 0, !dbg !48
  %1640 = insertelement <4 x float> poison, float %1426, i64 0, !dbg !47
  %1641 = insertelement <4 x float> %1640, float %1432, i64 1, !dbg !47
  %1642 = insertelement <4 x float> %1641, float %1427, i64 2, !dbg !47
  %1643 = insertelement <4 x float> %1642, float %1433, i64 3, !dbg !47
  %normalized60 = fmul <4 x float> %1643, %row.reciprocal58, !dbg !47
  %1644 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !48
  %1645 = extractelement <4 x half> %1644, i64 3, !dbg !48
  %1646 = extractelement <4 x half> %1644, i64 2, !dbg !48
  %1647 = extractelement <4 x half> %1644, i64 1, !dbg !48
  %1648 = extractelement <4 x half> %1644, i64 0, !dbg !48
  %1649 = insertelement <4 x float> poison, float %1317, i64 0, !dbg !47
  %1650 = insertelement <4 x float> %1649, float %1323, i64 1, !dbg !47
  %1651 = insertelement <4 x float> %1650, float %1318, i64 2, !dbg !47
  %1652 = insertelement <4 x float> %1651, float %1324, i64 3, !dbg !47
  %row.reciprocal61 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1240, !dbg !47
  %normalized62 = fmul <4 x float> %1652, %row.reciprocal61, !dbg !47
  %1653 = fptrunc <4 x float> %normalized62 to <4 x half>, !dbg !48
  %1654 = extractelement <4 x half> %1653, i64 3, !dbg !48
  %1655 = extractelement <4 x half> %1653, i64 2, !dbg !48
  %1656 = extractelement <4 x half> %1653, i64 1, !dbg !48
  %1657 = extractelement <4 x half> %1653, i64 0, !dbg !48
  %1658 = insertelement <4 x float> poison, float %1208, i64 0, !dbg !47
  %1659 = insertelement <4 x float> %1658, float %1214, i64 1, !dbg !47
  %1660 = insertelement <4 x float> %1659, float %1209, i64 2, !dbg !47
  %1661 = insertelement <4 x float> %1660, float %1215, i64 3, !dbg !47
  %normalized63 = fmul <4 x float> %1661, %row.reciprocal61, !dbg !47
  %1662 = fptrunc <4 x float> %normalized63 to <4 x half>, !dbg !48
  %1663 = extractelement <4 x half> %1662, i64 3, !dbg !48
  %1664 = extractelement <4 x half> %1662, i64 2, !dbg !48
  %1665 = extractelement <4 x half> %1662, i64 1, !dbg !48
  %1666 = extractelement <4 x half> %1662, i64 0, !dbg !48
  %1667 = shl nuw nsw i64 %24, 19, !dbg !49
  %address.thread589 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !49, !range !65, !frisk.register_lifetimes !33
  %address.local590 = zext nneg i32 %address.thread589 to i64, !dbg !40
  %address.local591 = add i64 %25, %address.local590, !dbg !49
  %address.local592 = lshr i64 %address.local590, 6, !dbg !39
  %address.local593 = shl nuw nsw i64 %address.local592, 4, !dbg !49
  %address.local594 = add i64 %address.local591, %address.local593, !dbg !49
  %address.local595 = and i64 %address.local590, 240, !dbg !49
  %address.local596 = sub i64 %address.local594, %address.local595, !dbg !49
  %address.local597 = add i64 %address.local596, %26, !dbg !49
  %address.local598 = shl nuw nsw i64 %address.local597, 7, !dbg !49
  %address.local599 = add nuw nsw i64 %1667, %address.local598, !dbg !49
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local601 = lshr i64 %address.local590, 2, !dbg !39
  %address.local602 = and i64 %address.local601, 12, !dbg !39
  %address.local603 = or disjoint i64 %address.local602, 32, !dbg !49
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local603, !dbg !49
  store half %1666, ptr addrspace(1) %address.local604, align 2, !dbg !49
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local606 = or disjoint i64 %address.local602, 33, !dbg !49
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local605, i64 %address.local606, !dbg !49
  store half %1665, ptr addrspace(1) %address.local607, align 2, !dbg !49
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local609 = or disjoint i64 %address.local602, 34, !dbg !49
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local608, i64 %address.local609, !dbg !49
  store half %1664, ptr addrspace(1) %address.local610, align 2, !dbg !49
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local612 = or disjoint i64 %address.local602, 35, !dbg !49
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local611, i64 %address.local612, !dbg !49
  store half %1663, ptr addrspace(1) %address.local613, align 2, !dbg !49
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local615 = or disjoint i64 %address.local602, 48, !dbg !49
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local614, i64 %address.local615, !dbg !49
  store half %1657, ptr addrspace(1) %address.local616, align 2, !dbg !49
  %address.local617 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local618 = or disjoint i64 %address.local602, 49, !dbg !49
  %address.local619 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local617, i64 %address.local618, !dbg !49
  store half %1656, ptr addrspace(1) %address.local619, align 2, !dbg !49
  %address.local620 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local621 = or disjoint i64 %address.local602, 50, !dbg !49
  %address.local622 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local621, !dbg !49
  store half %1655, ptr addrspace(1) %address.local622, align 2, !dbg !49
  %address.local623 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local624 = or i64 %address.local601, 51, !dbg !49
  %address.local625 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local623, i64 %address.local624, !dbg !49
  store half %1654, ptr addrspace(1) %address.local625, align 2, !dbg !49
  %address.local626 = shl i64 %address.local597, 7, !dbg !49
  %address.local627 = add i64 %address.local626, 8192, !dbg !49
  %address.local628 = add nuw nsw i64 %1667, %address.local627, !dbg !49
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local629, i64 %address.local603, !dbg !49
  store half %1648, ptr addrspace(1) %address.local630, align 2, !dbg !49
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local606, !dbg !49
  store half %1647, ptr addrspace(1) %address.local632, align 2, !dbg !49
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local633, i64 %address.local609, !dbg !49
  store half %1646, ptr addrspace(1) %address.local634, align 2, !dbg !49
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local635, i64 %address.local612, !dbg !49
  store half %1645, ptr addrspace(1) %address.local636, align 2, !dbg !49
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local637, i64 %address.local615, !dbg !49
  store half %1639, ptr addrspace(1) %address.local638, align 2, !dbg !49
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local639, i64 %address.local618, !dbg !49
  store half %1638, ptr addrspace(1) %address.local640, align 2, !dbg !49
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local642 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local641, i64 %address.local621, !dbg !49
  store half %1637, ptr addrspace(1) %address.local642, align 2, !dbg !49
  %address.local643 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local624, !dbg !49
  store half %1636, ptr addrspace(1) %address.local644, align 2, !dbg !49
  br label %1668, !dbg !49

1668:                                             ; preds = %1630, %1629
  br i1 true, label %1669, label %1707, !dbg !49

1669:                                             ; preds = %1668
  %1670 = insertelement <4 x float> poison, float %1507, i64 0, !dbg !47
  %1671 = insertelement <4 x float> %1670, float %1513, i64 1, !dbg !47
  %1672 = insertelement <4 x float> %1671, float %1508, i64 2, !dbg !47
  %1673 = insertelement <4 x float> %1672, float %1514, i64 3, !dbg !47
  %row.reciprocal64 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1458, !dbg !47
  %normalized65 = fmul <4 x float> %1673, %row.reciprocal64, !dbg !47
  %1674 = fptrunc <4 x float> %normalized65 to <4 x half>, !dbg !48
  %1675 = extractelement <4 x half> %1674, i64 3, !dbg !48
  %1676 = extractelement <4 x half> %1674, i64 2, !dbg !48
  %1677 = extractelement <4 x half> %1674, i64 1, !dbg !48
  %1678 = extractelement <4 x half> %1674, i64 0, !dbg !48
  %1679 = insertelement <4 x float> poison, float %1397, i64 0, !dbg !47
  %1680 = insertelement <4 x float> %1679, float %1403, i64 1, !dbg !47
  %1681 = insertelement <4 x float> %1680, float %1398, i64 2, !dbg !47
  %1682 = insertelement <4 x float> %1681, float %1404, i64 3, !dbg !47
  %normalized66 = fmul <4 x float> %1682, %row.reciprocal64, !dbg !47
  %1683 = fptrunc <4 x float> %normalized66 to <4 x half>, !dbg !48
  %1684 = extractelement <4 x half> %1683, i64 3, !dbg !48
  %1685 = extractelement <4 x half> %1683, i64 2, !dbg !48
  %1686 = extractelement <4 x half> %1683, i64 1, !dbg !48
  %1687 = extractelement <4 x half> %1683, i64 0, !dbg !48
  %1688 = insertelement <4 x float> poison, float %1289, i64 0, !dbg !47
  %1689 = insertelement <4 x float> %1688, float %1295, i64 1, !dbg !47
  %1690 = insertelement <4 x float> %1689, float %1290, i64 2, !dbg !47
  %1691 = insertelement <4 x float> %1690, float %1296, i64 3, !dbg !47
  %row.reciprocal67 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1240, !dbg !47
  %normalized68 = fmul <4 x float> %1691, %row.reciprocal67, !dbg !47
  %1692 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !48
  %1693 = extractelement <4 x half> %1692, i64 3, !dbg !48
  %1694 = extractelement <4 x half> %1692, i64 2, !dbg !48
  %1695 = extractelement <4 x half> %1692, i64 1, !dbg !48
  %1696 = extractelement <4 x half> %1692, i64 0, !dbg !48
  %1697 = insertelement <4 x float> poison, float %1179, i64 0, !dbg !47
  %1698 = insertelement <4 x float> %1697, float %1185, i64 1, !dbg !47
  %1699 = insertelement <4 x float> %1698, float %1180, i64 2, !dbg !47
  %1700 = insertelement <4 x float> %1699, float %1186, i64 3, !dbg !47
  %normalized69 = fmul <4 x float> %1700, %row.reciprocal67, !dbg !47
  %1701 = fptrunc <4 x float> %normalized69 to <4 x half>, !dbg !48
  %1702 = extractelement <4 x half> %1701, i64 3, !dbg !48
  %1703 = extractelement <4 x half> %1701, i64 2, !dbg !48
  %1704 = extractelement <4 x half> %1701, i64 1, !dbg !48
  %1705 = extractelement <4 x half> %1701, i64 0, !dbg !48
  %1706 = shl nuw nsw i64 %24, 19, !dbg !49
  %address.thread645 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !49, !range !65, !frisk.register_lifetimes !33
  %address.local646 = zext nneg i32 %address.thread645 to i64, !dbg !40
  %address.local647 = add i64 %25, %address.local646, !dbg !49
  %address.local648 = lshr i64 %address.local646, 6, !dbg !39
  %address.local649 = shl nuw nsw i64 %address.local648, 4, !dbg !49
  %address.local650 = add i64 %address.local647, %address.local649, !dbg !49
  %address.local651 = and i64 %address.local646, 240, !dbg !49
  %address.local652 = sub i64 %address.local650, %address.local651, !dbg !49
  %address.local653 = add i64 %address.local652, %26, !dbg !49
  %address.local654 = shl nuw nsw i64 %address.local653, 7, !dbg !49
  %address.local655 = add nuw nsw i64 %1706, %address.local654, !dbg !49
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local657 = lshr i64 %address.local646, 2, !dbg !39
  %address.local658 = and i64 %address.local657, 12, !dbg !39
  %address.local659 = or disjoint i64 %address.local658, 64, !dbg !49
  %address.local660 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local656, i64 %address.local659, !dbg !49
  store half %1705, ptr addrspace(1) %address.local660, align 2, !dbg !49
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local662 = or disjoint i64 %address.local658, 65, !dbg !49
  %address.local663 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local661, i64 %address.local662, !dbg !49
  store half %1704, ptr addrspace(1) %address.local663, align 2, !dbg !49
  %address.local664 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local665 = or disjoint i64 %address.local658, 66, !dbg !49
  %address.local666 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local664, i64 %address.local665, !dbg !49
  store half %1703, ptr addrspace(1) %address.local666, align 2, !dbg !49
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local668 = or disjoint i64 %address.local658, 67, !dbg !49
  %address.local669 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local667, i64 %address.local668, !dbg !49
  store half %1702, ptr addrspace(1) %address.local669, align 2, !dbg !49
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local671 = or disjoint i64 %address.local658, 80, !dbg !49
  %address.local672 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local670, i64 %address.local671, !dbg !49
  store half %1696, ptr addrspace(1) %address.local672, align 2, !dbg !49
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local674 = or disjoint i64 %address.local658, 81, !dbg !49
  %address.local675 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local673, i64 %address.local674, !dbg !49
  store half %1695, ptr addrspace(1) %address.local675, align 2, !dbg !49
  %address.local676 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local677 = or disjoint i64 %address.local658, 82, !dbg !49
  %address.local678 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local676, i64 %address.local677, !dbg !49
  store half %1694, ptr addrspace(1) %address.local678, align 2, !dbg !49
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local680 = or disjoint i64 %address.local658, 83, !dbg !49
  %address.local681 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local679, i64 %address.local680, !dbg !49
  store half %1693, ptr addrspace(1) %address.local681, align 2, !dbg !49
  %address.local682 = shl i64 %address.local653, 7, !dbg !49
  %address.local683 = add i64 %address.local682, 8192, !dbg !49
  %address.local684 = add nuw nsw i64 %1706, %address.local683, !dbg !49
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local685, i64 %address.local659, !dbg !49
  store half %1687, ptr addrspace(1) %address.local686, align 2, !dbg !49
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local687, i64 %address.local662, !dbg !49
  store half %1686, ptr addrspace(1) %address.local688, align 2, !dbg !49
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local689, i64 %address.local665, !dbg !49
  store half %1685, ptr addrspace(1) %address.local690, align 2, !dbg !49
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local691, i64 %address.local668, !dbg !49
  store half %1684, ptr addrspace(1) %address.local692, align 2, !dbg !49
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local693, i64 %address.local671, !dbg !49
  store half %1678, ptr addrspace(1) %address.local694, align 2, !dbg !49
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local695, i64 %address.local674, !dbg !49
  store half %1677, ptr addrspace(1) %address.local696, align 2, !dbg !49
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local697, i64 %address.local677, !dbg !49
  store half %1676, ptr addrspace(1) %address.local698, align 2, !dbg !49
  %address.local699 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local699, i64 %address.local680, !dbg !49
  store half %1675, ptr addrspace(1) %address.local700, align 2, !dbg !49
  br label %1707, !dbg !49

1707:                                             ; preds = %1669, %1668
  br i1 true, label %1708, label %1746, !dbg !49

1708:                                             ; preds = %1707
  %1709 = insertelement <4 x float> poison, float %1479, i64 0, !dbg !47
  %1710 = insertelement <4 x float> %1709, float %1485, i64 1, !dbg !47
  %1711 = insertelement <4 x float> %1710, float %1480, i64 2, !dbg !47
  %1712 = insertelement <4 x float> %1711, float %1486, i64 3, !dbg !47
  %row.reciprocal70 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1458, !dbg !47
  %normalized71 = fmul <4 x float> %1712, %row.reciprocal70, !dbg !47
  %1713 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !48
  %1714 = extractelement <4 x half> %1713, i64 3, !dbg !48
  %1715 = extractelement <4 x half> %1713, i64 2, !dbg !48
  %1716 = extractelement <4 x half> %1713, i64 1, !dbg !48
  %1717 = extractelement <4 x half> %1713, i64 0, !dbg !48
  %1718 = insertelement <4 x float> poison, float %1368, i64 0, !dbg !47
  %1719 = insertelement <4 x float> %1718, float %1374, i64 1, !dbg !47
  %1720 = insertelement <4 x float> %1719, float %1369, i64 2, !dbg !47
  %1721 = insertelement <4 x float> %1720, float %1375, i64 3, !dbg !47
  %normalized72 = fmul <4 x float> %1721, %row.reciprocal70, !dbg !47
  %1722 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !48
  %1723 = extractelement <4 x half> %1722, i64 3, !dbg !48
  %1724 = extractelement <4 x half> %1722, i64 2, !dbg !48
  %1725 = extractelement <4 x half> %1722, i64 1, !dbg !48
  %1726 = extractelement <4 x half> %1722, i64 0, !dbg !48
  %1727 = insertelement <4 x float> poison, float %1261, i64 0, !dbg !47
  %1728 = insertelement <4 x float> %1727, float %1267, i64 1, !dbg !47
  %1729 = insertelement <4 x float> %1728, float %1262, i64 2, !dbg !47
  %1730 = insertelement <4 x float> %1729, float %1268, i64 3, !dbg !47
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1240, !dbg !47
  %normalized74 = fmul <4 x float> %1730, %row.reciprocal73, !dbg !47
  %1731 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !48
  %1732 = extractelement <4 x half> %1731, i64 3, !dbg !48
  %1733 = extractelement <4 x half> %1731, i64 2, !dbg !48
  %1734 = extractelement <4 x half> %1731, i64 1, !dbg !48
  %1735 = extractelement <4 x half> %1731, i64 0, !dbg !48
  %1736 = insertelement <4 x float> poison, float %1150, i64 0, !dbg !47
  %1737 = insertelement <4 x float> %1736, float %1156, i64 1, !dbg !47
  %1738 = insertelement <4 x float> %1737, float %1151, i64 2, !dbg !47
  %1739 = insertelement <4 x float> %1738, float %1157, i64 3, !dbg !47
  %normalized75 = fmul <4 x float> %1739, %row.reciprocal73, !dbg !47
  %1740 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !48
  %1741 = extractelement <4 x half> %1740, i64 3, !dbg !48
  %1742 = extractelement <4 x half> %1740, i64 2, !dbg !48
  %1743 = extractelement <4 x half> %1740, i64 1, !dbg !48
  %1744 = extractelement <4 x half> %1740, i64 0, !dbg !48
  %1745 = shl nuw nsw i64 %24, 19, !dbg !49
  %address.thread701 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !49, !range !65, !frisk.register_lifetimes !33
  %address.local702 = zext nneg i32 %address.thread701 to i64, !dbg !40
  %address.local703 = add i64 %25, %address.local702, !dbg !49
  %address.local704 = lshr i64 %address.local702, 6, !dbg !39
  %address.local705 = shl nuw nsw i64 %address.local704, 4, !dbg !49
  %address.local706 = add i64 %address.local703, %address.local705, !dbg !49
  %address.local707 = and i64 %address.local702, 240, !dbg !49
  %address.local708 = sub i64 %address.local706, %address.local707, !dbg !49
  %address.local709 = add i64 %address.local708, %26, !dbg !49
  %address.local710 = shl nuw nsw i64 %address.local709, 7, !dbg !49
  %address.local711 = add nuw nsw i64 %1745, %address.local710, !dbg !49
  %address.local712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local713 = lshr i64 %address.local702, 2, !dbg !39
  %address.local714 = and i64 %address.local713, 12, !dbg !39
  %address.local715 = or disjoint i64 %address.local714, 96, !dbg !49
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local712, i64 %address.local715, !dbg !49
  store half %1744, ptr addrspace(1) %address.local716, align 2, !dbg !49
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local718 = or disjoint i64 %address.local714, 97, !dbg !49
  %address.local719 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local717, i64 %address.local718, !dbg !49
  store half %1743, ptr addrspace(1) %address.local719, align 2, !dbg !49
  %address.local720 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local721 = or disjoint i64 %address.local714, 98, !dbg !49
  %address.local722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local720, i64 %address.local721, !dbg !49
  store half %1742, ptr addrspace(1) %address.local722, align 2, !dbg !49
  %address.local723 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local724 = or disjoint i64 %address.local714, 99, !dbg !49
  %address.local725 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local723, i64 %address.local724, !dbg !49
  store half %1741, ptr addrspace(1) %address.local725, align 2, !dbg !49
  %address.local726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local727 = or disjoint i64 %address.local714, 112, !dbg !49
  %address.local728 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local726, i64 %address.local727, !dbg !49
  store half %1735, ptr addrspace(1) %address.local728, align 2, !dbg !49
  %address.local729 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local730 = or disjoint i64 %address.local714, 113, !dbg !49
  %address.local731 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local729, i64 %address.local730, !dbg !49
  store half %1734, ptr addrspace(1) %address.local731, align 2, !dbg !49
  %address.local732 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local733 = or disjoint i64 %address.local714, 114, !dbg !49
  %address.local734 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local732, i64 %address.local733, !dbg !49
  store half %1733, ptr addrspace(1) %address.local734, align 2, !dbg !49
  %address.local735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local736 = or i64 %address.local713, 115, !dbg !49
  %address.local737 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local735, i64 %address.local736, !dbg !49
  store half %1732, ptr addrspace(1) %address.local737, align 2, !dbg !49
  %address.local738 = shl i64 %address.local709, 7, !dbg !49
  %address.local739 = add i64 %address.local738, 8192, !dbg !49
  %address.local740 = add nuw nsw i64 %1745, %address.local739, !dbg !49
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local741, i64 %address.local715, !dbg !49
  store half %1726, ptr addrspace(1) %address.local742, align 2, !dbg !49
  %address.local743 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local743, i64 %address.local718, !dbg !49
  store half %1725, ptr addrspace(1) %address.local744, align 2, !dbg !49
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local746 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local745, i64 %address.local721, !dbg !49
  store half %1724, ptr addrspace(1) %address.local746, align 2, !dbg !49
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local747, i64 %address.local724, !dbg !49
  store half %1723, ptr addrspace(1) %address.local748, align 2, !dbg !49
  %address.local749 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local749, i64 %address.local727, !dbg !49
  store half %1717, ptr addrspace(1) %address.local750, align 2, !dbg !49
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local751, i64 %address.local730, !dbg !49
  store half %1716, ptr addrspace(1) %address.local752, align 2, !dbg !49
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local754 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local753, i64 %address.local733, !dbg !49
  store half %1715, ptr addrspace(1) %address.local754, align 2, !dbg !49
  %address.local755 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local755, i64 %address.local736, !dbg !49
  store half %1714, ptr addrspace(1) %address.local756, align 2, !dbg !49
  br label %1746, !dbg !49

1746:                                             ; preds = %1708, %1707
  ret void, !dbg !37
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

!llvm.dbg.cu = !{!35}
!llvm.module.flags = !{!64}

!32 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0002-test_input_1x32x4096x128-128-64-32-online/retained32-baseline")
!33 = !{}
!34 = !DISubroutineType(types: !33)
!35 = distinct !DICompileUnit(language: DW_LANG_C, file: !32, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!36 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !32, file: !32, line: 23, type: !34, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !35)
!37 = !DILocation(line: 23, column: 3, scope: !36)
!38 = !DILocation(line: 60, column: 24, scope: !36)
!39 = !DILocation(line: 72, column: 24, scope: !36)
!40 = !DILocation(line: 70, column: 22, scope: !36)
!41 = !DILocation(line: 48, column: 17, scope: !36)
!42 = !DILocation(line: 77, column: 19, scope: !36)
!43 = !DILocation(line: 102, column: 23, scope: !36)
!44 = !DILocation(line: 71, column: 22, scope: !36)
!45 = !DILocation(line: 73, column: 26, scope: !36)
!46 = !DILocation(line: 101, column: 20, scope: !36)
!47 = !DILocation(line: 107, column: 23, scope: !36)
!48 = !DILocation(line: 108, column: 19, scope: !36)
!49 = !DILocation(line: 109, column: 9, scope: !36)
!50 = !DILocation(line: 79, column: 24, scope: !36)
!51 = !DILocation(line: 80, column: 23, scope: !36)
!52 = !DILocation(line: 81, column: 26, scope: !36)
!53 = !DILocation(line: 88, column: 21, scope: !36)
!54 = !DILocation(line: 89, column: 20, scope: !36)
!55 = !DILocation(line: 90, column: 22, scope: !36)
!56 = !DILocation(line: 91, column: 23, scope: !36)
!57 = !DILocation(line: 92, column: 20, scope: !36)
!58 = !DILocation(line: 93, column: 22, scope: !36)
!59 = !DILocation(line: 94, column: 22, scope: !36)
!60 = !DILocation(line: 95, column: 20, scope: !36)
!61 = !DILocation(line: 96, column: 16, scope: !36)
!62 = !DILocation(line: 103, column: 22, scope: !36)
!63 = !{i32 256, i32 1, i32 1}
!64 = !{i32 2, !"Debug Info Version", i32 3}
!65 = !{i32 0, i32 256}
