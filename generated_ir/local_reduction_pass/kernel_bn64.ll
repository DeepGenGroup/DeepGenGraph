; ModuleID = 'generated_ir/online_softmax_mask_bypass_pass/kernel_bn64.ll'
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
  %206 = phi i64 [ %1057, %594 ], [ 0, %4 ], !dbg !38
  %207 = phi [2 x <8 x float>] [ %1042, %594 ], [ zeroinitializer, %4 ], !dbg !38
  %208 = phi [2 x <1 x float>] [ %818, %594 ], [ zeroinitializer, %4 ], !dbg !38
  %209 = phi [2 x <1 x float>] [ %1044, %594 ], [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], !dbg !38
  %210 = phi [2 x <8 x float>] [ %1048, %594 ], [ zeroinitializer, %4 ], !dbg !38
  %211 = phi [2 x <8 x float>] [ %1052, %594 ], [ zeroinitializer, %4 ], !dbg !38
  %212 = phi [2 x <8 x float>] [ %1056, %594 ], [ zeroinitializer, %4 ], !dbg !38
  %213 = icmp slt i64 %206, %192, !dbg !38
  br i1 %213, label %214, label %1058, !dbg !38

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
  %627 = call float @llvm.maxnum.f32(float %595, float -inf), !dbg !54, !frisk.local_reduction !33
  %local.reduce45 = call float @llvm.maxnum.f32(float %627, float %599), !dbg !54, !frisk.local_reduction !33
  %local.reduce46 = call float @llvm.maxnum.f32(float %596, float %600), !dbg !54, !frisk.local_reduction !33
  %local.reduce47 = call float @llvm.maxnum.f32(float %597, float %601), !dbg !54, !frisk.local_reduction !33
  %local.reduce48 = call float @llvm.maxnum.f32(float %598, float %602), !dbg !54, !frisk.local_reduction !33
  %local.reduce49 = call float @llvm.maxnum.f32(float %local.reduce45, float %603), !dbg !54, !frisk.local_reduction !33
  %local.reduce50 = call float @llvm.maxnum.f32(float %local.reduce46, float %604), !dbg !54, !frisk.local_reduction !33
  %local.reduce51 = call float @llvm.maxnum.f32(float %local.reduce47, float %605), !dbg !54, !frisk.local_reduction !33
  %local.reduce52 = call float @llvm.maxnum.f32(float %local.reduce48, float %606), !dbg !54, !frisk.local_reduction !33
  %local.reduce53 = call float @llvm.maxnum.f32(float %local.reduce49, float %607), !dbg !54, !frisk.local_reduction !33
  %local.reduce54 = call float @llvm.maxnum.f32(float %local.reduce50, float %608), !dbg !54, !frisk.local_reduction !33
  %local.reduce55 = call float @llvm.maxnum.f32(float %local.reduce51, float %609), !dbg !54, !frisk.local_reduction !33
  %local.reduce56 = call float @llvm.maxnum.f32(float %local.reduce52, float %610), !dbg !54, !frisk.local_reduction !33
  %local.reduce57 = call float @llvm.maxnum.f32(float %local.reduce53, float %local.reduce54), !dbg !54, !frisk.local_reduction !33
  %local.reduce58 = call float @llvm.maxnum.f32(float %local.reduce55, float %local.reduce56), !dbg !54, !frisk.local_reduction !33
  %628 = call float @llvm.maxnum.f32(float %local.reduce57, float %local.reduce58), !dbg !54, !frisk.local_reduction !33
  %629 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !54
  %630 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %629), !dbg !54
  %631 = add nuw nsw i32 %630, 64, !dbg !54
  %632 = xor i32 %630, 16, !dbg !54
  %633 = and i32 %631, 192, !dbg !54
  %634 = icmp samesign ult i32 %632, %633, !dbg !54
  %635 = select i1 %634, i32 %632, i32 %630, !dbg !54
  %636 = shl nuw nsw i32 %635, 2, !dbg !54
  %637 = bitcast float %628 to i32, !dbg !54
  %638 = call i32 @llvm.amdgcn.ds.bpermute(i32 %636, i32 %637), !dbg !54
  %639 = bitcast i32 %638 to float, !dbg !54
  %640 = call float @llvm.maxnum.f32(float %628, float %639), !dbg !54
  %641 = xor i32 %630, 32, !dbg !54
  %642 = icmp samesign ult i32 %641, %633, !dbg !54
  %643 = select i1 %642, i32 %641, i32 %630, !dbg !54
  %644 = shl nuw nsw i32 %643, 2, !dbg !54
  %645 = bitcast float %640 to i32, !dbg !54
  %646 = call i32 @llvm.amdgcn.ds.bpermute(i32 %644, i32 %645), !dbg !54
  %647 = bitcast i32 %646 to float, !dbg !54
  %648 = call float @llvm.maxnum.f32(float %640, float %647), !dbg !54
  %649 = extractvalue [2 x <1 x float>] %209, 0, !dbg !54
  %650 = insertelement <1 x float> poison, float %648, i64 0, !dbg !54
  %651 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %650, <1 x float> %649), !dbg !54
  %652 = call float @llvm.maxnum.f32(float %611, float -inf), !dbg !54, !frisk.local_reduction !33
  %local.reduce30 = call float @llvm.maxnum.f32(float %652, float %615), !dbg !54, !frisk.local_reduction !33
  %local.reduce31 = call float @llvm.maxnum.f32(float %612, float %616), !dbg !54, !frisk.local_reduction !33
  %local.reduce32 = call float @llvm.maxnum.f32(float %613, float %617), !dbg !54, !frisk.local_reduction !33
  %local.reduce33 = call float @llvm.maxnum.f32(float %614, float %618), !dbg !54, !frisk.local_reduction !33
  %local.reduce34 = call float @llvm.maxnum.f32(float %local.reduce30, float %619), !dbg !54, !frisk.local_reduction !33
  %local.reduce35 = call float @llvm.maxnum.f32(float %local.reduce31, float %620), !dbg !54, !frisk.local_reduction !33
  %local.reduce36 = call float @llvm.maxnum.f32(float %local.reduce32, float %621), !dbg !54, !frisk.local_reduction !33
  %local.reduce37 = call float @llvm.maxnum.f32(float %local.reduce33, float %622), !dbg !54, !frisk.local_reduction !33
  %local.reduce38 = call float @llvm.maxnum.f32(float %local.reduce34, float %623), !dbg !54, !frisk.local_reduction !33
  %local.reduce39 = call float @llvm.maxnum.f32(float %local.reduce35, float %624), !dbg !54, !frisk.local_reduction !33
  %local.reduce40 = call float @llvm.maxnum.f32(float %local.reduce36, float %625), !dbg !54, !frisk.local_reduction !33
  %local.reduce41 = call float @llvm.maxnum.f32(float %local.reduce37, float %626), !dbg !54, !frisk.local_reduction !33
  %local.reduce42 = call float @llvm.maxnum.f32(float %local.reduce38, float %local.reduce39), !dbg !54, !frisk.local_reduction !33
  %local.reduce43 = call float @llvm.maxnum.f32(float %local.reduce40, float %local.reduce41), !dbg !54, !frisk.local_reduction !33
  %653 = call float @llvm.maxnum.f32(float %local.reduce42, float %local.reduce43), !dbg !54, !frisk.local_reduction !33
  %654 = bitcast float %653 to i32, !dbg !54
  %655 = call i32 @llvm.amdgcn.ds.bpermute(i32 %636, i32 %654), !dbg !54
  %656 = bitcast i32 %655 to float, !dbg !54
  %657 = call float @llvm.maxnum.f32(float %653, float %656), !dbg !54
  %658 = bitcast float %657 to i32, !dbg !54
  %659 = call i32 @llvm.amdgcn.ds.bpermute(i32 %644, i32 %658), !dbg !54
  %660 = bitcast i32 %659 to float, !dbg !54
  %661 = call float @llvm.maxnum.f32(float %657, float %660), !dbg !54
  %662 = extractvalue [2 x <1 x float>] %209, 1, !dbg !54
  %663 = insertelement <1 x float> poison, float %661, i64 0, !dbg !54
  %664 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %663, <1 x float> %662), !dbg !54
  %665 = shufflevector <1 x float> %651, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !55
  %666 = fsub <4 x float> %586, %665, !dbg !55
  %667 = fsub <4 x float> %587, %665, !dbg !55
  %668 = fsub <4 x float> %588, %665, !dbg !55
  %669 = fsub <4 x float> %589, %665, !dbg !55
  %670 = shufflevector <1 x float> %664, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !55
  %671 = fsub <4 x float> %590, %670, !dbg !55
  %672 = fsub <4 x float> %591, %670, !dbg !55
  %673 = fsub <4 x float> %592, %670, !dbg !55
  %674 = fsub <4 x float> %593, %670, !dbg !55
  %675 = fsub <1 x float> %649, %651, !dbg !56
  %676 = extractelement <1 x float> %675, i64 0, !dbg !57
  %677 = call float @llvm.amdgcn.exp2.f32(float %676), !dbg !57
  %678 = insertelement <4 x float> poison, float %677, i64 0, !dbg !62
  %679 = shufflevector <4 x float> %678, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !62
  %680 = extractvalue [2 x <1 x float>] %208, 0, !dbg !58
  %681 = extractelement <1 x float> %680, i64 0, !dbg !58
  %682 = fmul float %681, %677, !dbg !58
  %683 = fsub <1 x float> %662, %664, !dbg !56
  %684 = extractelement <1 x float> %683, i64 0, !dbg !57
  %685 = call float @llvm.amdgcn.exp2.f32(float %684), !dbg !57
  %686 = insertelement <4 x float> poison, float %685, i64 0, !dbg !62
  %687 = shufflevector <4 x float> %686, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !62
  %688 = extractvalue [2 x <1 x float>] %208, 1, !dbg !58
  %689 = extractelement <1 x float> %688, i64 0, !dbg !58
  %690 = fmul float %689, %685, !dbg !58
  %691 = extractelement <4 x float> %666, i64 0, !dbg !59
  %692 = call float @llvm.amdgcn.exp2.f32(float %691), !dbg !59
  %693 = insertelement <4 x float> poison, float %692, i64 0, !dbg !61
  %694 = extractelement <4 x float> %666, i64 1, !dbg !59
  %695 = call float @llvm.amdgcn.exp2.f32(float %694), !dbg !59
  %696 = insertelement <4 x float> %693, float %695, i64 1, !dbg !61
  %697 = extractelement <4 x float> %666, i64 2, !dbg !59
  %698 = call float @llvm.amdgcn.exp2.f32(float %697), !dbg !59
  %699 = insertelement <4 x float> %696, float %698, i64 2, !dbg !61
  %700 = extractelement <4 x float> %666, i64 3, !dbg !59
  %701 = call float @llvm.amdgcn.exp2.f32(float %700), !dbg !59
  %702 = insertelement <4 x float> %699, float %701, i64 3, !dbg !61
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %702) #9, !dbg !61, !frisk.register_lifetimes !33
  %703 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !61
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %703) #9, !dbg !60, !frisk.register_lifetimes !33
  %704 = extractelement <4 x float> %667, i64 0, !dbg !59
  %705 = call float @llvm.amdgcn.exp2.f32(float %704), !dbg !59
  %706 = insertelement <4 x float> poison, float %705, i64 0, !dbg !61
  %local.reduce15 = fadd float %692, %705, !dbg !60, !frisk.local_reduction !33
  %707 = extractelement <4 x float> %667, i64 1, !dbg !59
  %708 = call float @llvm.amdgcn.exp2.f32(float %707), !dbg !59
  %709 = insertelement <4 x float> %706, float %708, i64 1, !dbg !61
  %local.reduce16 = fadd float %695, %708, !dbg !60, !frisk.local_reduction !33
  %710 = extractelement <4 x float> %667, i64 2, !dbg !59
  %711 = call float @llvm.amdgcn.exp2.f32(float %710), !dbg !59
  %712 = insertelement <4 x float> %709, float %711, i64 2, !dbg !61
  %local.reduce17 = fadd float %698, %711, !dbg !60, !frisk.local_reduction !33
  %713 = extractelement <4 x float> %667, i64 3, !dbg !59
  %714 = call float @llvm.amdgcn.exp2.f32(float %713), !dbg !59
  %715 = insertelement <4 x float> %712, float %714, i64 3, !dbg !61
  %math.result162 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %715) #9, !dbg !61, !frisk.register_lifetimes !33
  %716 = fptrunc <4 x float> %math.result162 to <4 x half>, !dbg !61
  %math.packed163 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %716) #9, !dbg !60, !frisk.register_lifetimes !33
  %local.reduce18 = fadd float %701, %714, !dbg !60, !frisk.local_reduction !33
  %717 = extractelement <4 x float> %668, i64 0, !dbg !59
  %718 = call float @llvm.amdgcn.exp2.f32(float %717), !dbg !59
  %719 = insertelement <4 x float> poison, float %718, i64 0, !dbg !61
  %local.reduce19 = fadd float %local.reduce15, %718, !dbg !60, !frisk.local_reduction !33
  %720 = extractelement <4 x float> %668, i64 1, !dbg !59
  %721 = call float @llvm.amdgcn.exp2.f32(float %720), !dbg !59
  %722 = insertelement <4 x float> %719, float %721, i64 1, !dbg !61
  %local.reduce20 = fadd float %local.reduce16, %721, !dbg !60, !frisk.local_reduction !33
  %723 = extractelement <4 x float> %668, i64 2, !dbg !59
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !59
  %725 = insertelement <4 x float> %722, float %724, i64 2, !dbg !61
  %local.reduce21 = fadd float %local.reduce17, %724, !dbg !60, !frisk.local_reduction !33
  %726 = extractelement <4 x float> %668, i64 3, !dbg !59
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !59
  %728 = insertelement <4 x float> %725, float %727, i64 3, !dbg !61
  %math.result164 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %728) #9, !dbg !61, !frisk.register_lifetimes !33
  %729 = fptrunc <4 x float> %math.result164 to <4 x half>, !dbg !61
  %math.packed165 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %729) #9, !dbg !60, !frisk.register_lifetimes !33
  %local.reduce22 = fadd float %local.reduce18, %727, !dbg !60, !frisk.local_reduction !33
  %730 = extractelement <4 x float> %669, i64 0, !dbg !59
  %731 = call float @llvm.amdgcn.exp2.f32(float %730), !dbg !59
  %732 = insertelement <4 x float> poison, float %731, i64 0, !dbg !61
  %local.reduce23 = fadd float %local.reduce19, %731, !dbg !60, !frisk.local_reduction !33
  %733 = extractelement <4 x float> %669, i64 1, !dbg !59
  %734 = call float @llvm.amdgcn.exp2.f32(float %733), !dbg !59
  %735 = insertelement <4 x float> %732, float %734, i64 1, !dbg !61
  %local.reduce24 = fadd float %local.reduce20, %734, !dbg !60, !frisk.local_reduction !33
  %736 = extractelement <4 x float> %669, i64 2, !dbg !59
  %737 = call float @llvm.amdgcn.exp2.f32(float %736), !dbg !59
  %738 = insertelement <4 x float> %735, float %737, i64 2, !dbg !61
  %local.reduce25 = fadd float %local.reduce21, %737, !dbg !60, !frisk.local_reduction !33
  %739 = extractelement <4 x float> %669, i64 3, !dbg !59
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !59
  %741 = insertelement <4 x float> %738, float %740, i64 3, !dbg !61
  %math.result166 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %741) #9, !dbg !61, !frisk.register_lifetimes !33
  %742 = fptrunc <4 x float> %math.result166 to <4 x half>, !dbg !61
  %math.packed167 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %742) #9, !dbg !60, !frisk.register_lifetimes !33
  %local.reduce26 = fadd float %local.reduce22, %740, !dbg !60, !frisk.local_reduction !33
  %local.reduce27 = fadd float %local.reduce23, %local.reduce24, !dbg !60, !frisk.local_reduction !33
  %local.reduce28 = fadd float %local.reduce25, %local.reduce26, !dbg !60, !frisk.local_reduction !33
  %743 = fadd float %local.reduce27, %local.reduce28, !dbg !60, !frisk.local_reduction !33
  %744 = extractelement <4 x float> %671, i64 0, !dbg !59
  %745 = call float @llvm.amdgcn.exp2.f32(float %744), !dbg !59
  %746 = insertelement <4 x float> poison, float %745, i64 0, !dbg !61
  %747 = extractelement <4 x float> %671, i64 1, !dbg !59
  %748 = call float @llvm.amdgcn.exp2.f32(float %747), !dbg !59
  %749 = insertelement <4 x float> %746, float %748, i64 1, !dbg !61
  %750 = extractelement <4 x float> %671, i64 2, !dbg !59
  %751 = call float @llvm.amdgcn.exp2.f32(float %750), !dbg !59
  %752 = insertelement <4 x float> %749, float %751, i64 2, !dbg !61
  %753 = extractelement <4 x float> %671, i64 3, !dbg !59
  %754 = call float @llvm.amdgcn.exp2.f32(float %753), !dbg !59
  %755 = insertelement <4 x float> %752, float %754, i64 3, !dbg !61
  %math.result168 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %755) #9, !dbg !61, !frisk.register_lifetimes !33
  %756 = fptrunc <4 x float> %math.result168 to <4 x half>, !dbg !61
  %math.packed169 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %756) #9, !dbg !60, !frisk.register_lifetimes !33
  %757 = extractelement <4 x float> %672, i64 0, !dbg !59
  %758 = call float @llvm.amdgcn.exp2.f32(float %757), !dbg !59
  %759 = insertelement <4 x float> poison, float %758, i64 0, !dbg !61
  %local.reduce = fadd float %745, %758, !dbg !60, !frisk.local_reduction !33
  %760 = extractelement <4 x float> %672, i64 1, !dbg !59
  %761 = call float @llvm.amdgcn.exp2.f32(float %760), !dbg !59
  %762 = insertelement <4 x float> %759, float %761, i64 1, !dbg !61
  %local.reduce1 = fadd float %748, %761, !dbg !60, !frisk.local_reduction !33
  %763 = extractelement <4 x float> %672, i64 2, !dbg !59
  %764 = call float @llvm.amdgcn.exp2.f32(float %763), !dbg !59
  %765 = insertelement <4 x float> %762, float %764, i64 2, !dbg !61
  %local.reduce2 = fadd float %751, %764, !dbg !60, !frisk.local_reduction !33
  %766 = extractelement <4 x float> %672, i64 3, !dbg !59
  %767 = call float @llvm.amdgcn.exp2.f32(float %766), !dbg !59
  %768 = insertelement <4 x float> %765, float %767, i64 3, !dbg !61
  %math.result170 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %768) #9, !dbg !61, !frisk.register_lifetimes !33
  %769 = fptrunc <4 x float> %math.result170 to <4 x half>, !dbg !61
  %math.packed171 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %769) #9, !dbg !60, !frisk.register_lifetimes !33
  %local.reduce3 = fadd float %754, %767, !dbg !60, !frisk.local_reduction !33
  %770 = extractelement <4 x float> %673, i64 0, !dbg !59
  %771 = call float @llvm.amdgcn.exp2.f32(float %770), !dbg !59
  %772 = insertelement <4 x float> poison, float %771, i64 0, !dbg !61
  %local.reduce4 = fadd float %local.reduce, %771, !dbg !60, !frisk.local_reduction !33
  %773 = extractelement <4 x float> %673, i64 1, !dbg !59
  %774 = call float @llvm.amdgcn.exp2.f32(float %773), !dbg !59
  %775 = insertelement <4 x float> %772, float %774, i64 1, !dbg !61
  %local.reduce5 = fadd float %local.reduce1, %774, !dbg !60, !frisk.local_reduction !33
  %776 = extractelement <4 x float> %673, i64 2, !dbg !59
  %777 = call float @llvm.amdgcn.exp2.f32(float %776), !dbg !59
  %778 = insertelement <4 x float> %775, float %777, i64 2, !dbg !61
  %local.reduce6 = fadd float %local.reduce2, %777, !dbg !60, !frisk.local_reduction !33
  %779 = extractelement <4 x float> %673, i64 3, !dbg !59
  %780 = call float @llvm.amdgcn.exp2.f32(float %779), !dbg !59
  %781 = insertelement <4 x float> %778, float %780, i64 3, !dbg !61
  %math.result172 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %781) #9, !dbg !61, !frisk.register_lifetimes !33
  %782 = fptrunc <4 x float> %math.result172 to <4 x half>, !dbg !61
  %math.packed173 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %782) #9, !dbg !60, !frisk.register_lifetimes !33
  %local.reduce7 = fadd float %local.reduce3, %780, !dbg !60, !frisk.local_reduction !33
  %783 = extractelement <4 x float> %674, i64 0, !dbg !59
  %784 = call float @llvm.amdgcn.exp2.f32(float %783), !dbg !59
  %785 = insertelement <4 x float> poison, float %784, i64 0, !dbg !61
  %local.reduce8 = fadd float %local.reduce4, %784, !dbg !60, !frisk.local_reduction !33
  %786 = extractelement <4 x float> %674, i64 1, !dbg !59
  %787 = call float @llvm.amdgcn.exp2.f32(float %786), !dbg !59
  %788 = insertelement <4 x float> %785, float %787, i64 1, !dbg !61
  %local.reduce9 = fadd float %local.reduce5, %787, !dbg !60, !frisk.local_reduction !33
  %789 = extractelement <4 x float> %674, i64 2, !dbg !59
  %790 = call float @llvm.amdgcn.exp2.f32(float %789), !dbg !59
  %791 = insertelement <4 x float> %788, float %790, i64 2, !dbg !61
  %local.reduce10 = fadd float %local.reduce6, %790, !dbg !60, !frisk.local_reduction !33
  %792 = extractelement <4 x float> %674, i64 3, !dbg !59
  %793 = call float @llvm.amdgcn.exp2.f32(float %792), !dbg !59
  %794 = insertelement <4 x float> %791, float %793, i64 3, !dbg !61
  %math.result174 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %794) #9, !dbg !61, !frisk.register_lifetimes !33
  %795 = fptrunc <4 x float> %math.result174 to <4 x half>, !dbg !61
  %math.packed175 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %795) #9, !dbg !60, !frisk.register_lifetimes !33
  %local.reduce11 = fadd float %local.reduce7, %793, !dbg !60, !frisk.local_reduction !33
  %local.reduce12 = fadd float %local.reduce8, %local.reduce9, !dbg !60, !frisk.local_reduction !33
  %local.reduce13 = fadd float %local.reduce10, %local.reduce11, !dbg !60, !frisk.local_reduction !33
  %796 = fadd float %local.reduce12, %local.reduce13, !dbg !60, !frisk.local_reduction !33
  %797 = bitcast float %743 to i32, !dbg !60
  %798 = call i32 @llvm.amdgcn.ds.bpermute(i32 %636, i32 %797), !dbg !60
  %799 = bitcast i32 %798 to float, !dbg !60
  %800 = fadd float %743, %799, !dbg !60
  %801 = bitcast float %800 to i32, !dbg !60
  %802 = call i32 @llvm.amdgcn.ds.bpermute(i32 %644, i32 %801), !dbg !60
  %803 = bitcast i32 %802 to float, !dbg !60
  %804 = fadd float %800, %803, !dbg !60
  %805 = fadd float %804, %682, !dbg !60
  %806 = insertelement <1 x float> poison, float %805, i64 0, !dbg !38
  %807 = insertvalue [2 x <1 x float>] poison, <1 x float> %806, 0, !dbg !38
  %808 = bitcast float %796 to i32, !dbg !60
  %809 = call i32 @llvm.amdgcn.ds.bpermute(i32 %636, i32 %808), !dbg !60
  %810 = bitcast i32 %809 to float, !dbg !60
  %811 = fadd float %796, %810, !dbg !60
  %812 = bitcast float %811 to i32, !dbg !60
  %813 = call i32 @llvm.amdgcn.ds.bpermute(i32 %644, i32 %812), !dbg !60
  %814 = bitcast i32 %813 to float, !dbg !60
  %815 = fadd float %811, %814, !dbg !60
  %816 = fadd float %815, %690, !dbg !60
  %817 = insertelement <1 x float> poison, float %816, i64 0, !dbg !38
  %818 = insertvalue [2 x <1 x float>] %807, <1 x float> %817, 1, !dbg !38
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
  %819 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !43
  %820 = bitcast <4 x i32> %819 to <8 x half>, !dbg !43
  %address.local509 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !43
  %address.local510 = getelementptr [2 x i8], ptr addrspace(3) %address.local509, i64 %address.local504, !dbg !43
  %address.local511 = getelementptr [2 x i8], ptr addrspace(3) %address.local510, i64 %address.local507, !dbg !43
  %address.local512 = getelementptr i8, ptr addrspace(3) %address.local511, i64 256, !dbg !43
  %821 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !43
  %822 = bitcast <4 x i32> %821 to <8 x half>, !dbg !43
  %address.local513 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !43
  %address.local514 = getelementptr [2 x i8], ptr addrspace(3) %address.local513, i64 %address.local504, !dbg !43
  %address.local515 = getelementptr [2 x i8], ptr addrspace(3) %address.local514, i64 %address.local507, !dbg !43
  %address.local516 = getelementptr i8, ptr addrspace(3) %address.local515, i64 512, !dbg !43
  %823 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !43
  %824 = bitcast <4 x i32> %823 to <8 x half>, !dbg !43
  %address.local517 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local500, !dbg !43
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) %address.local517, i64 %address.local504, !dbg !43
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) %address.local518, i64 %address.local507, !dbg !43
  %address.local520 = getelementptr i8, ptr addrspace(3) %address.local519, i64 768, !dbg !43
  %825 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !43
  %826 = bitcast <4 x i32> %825 to <8 x half>, !dbg !43
  %827 = add nuw nsw i64 %218, %16, !dbg !46
  %828 = shl i64 %827, 19, !dbg !46
  %829 = and i64 %828, 16252928, !dbg !46
  %830 = add i64 %829, %224, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %831 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %830, !dbg !46
  %832 = getelementptr i8, ptr addrspace(1) %831, i64 64, !dbg !46
  %address.local521 = shl nuw nsw i64 %address.local497, 3, !dbg !40
  %address.local522 = and i64 %address.local521, 1536, !dbg !44
  %address.local523 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local522, !dbg !44
  %address.local524 = ptrtoint ptr addrspace(3) %address.local523 to i64, !dbg !44
  %address.local525 = trunc i64 %address.local524 to i32, !dbg !44
  %833 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !46
  %834 = zext i32 %833 to i64, !dbg !46
  %835 = inttoptr i64 %834 to ptr addrspace(3), !dbg !46
  %address.local526 = shl nuw nsw i64 %address.local497, 5, !dbg !46
  %address.local527 = and i64 %address.local526, 8064, !dbg !46
  %address.local528 = shl nuw nsw i64 %address.local497, 3, !dbg !46
  %address.local529 = and i64 %address.local528, 24, !dbg !46
  %address.local530 = or disjoint i64 %address.local527, %address.local529, !dbg !46
  %address.local531 = getelementptr [2 x i8], ptr addrspace(1) %832, i64 %address.local530, !dbg !46
  %836 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !46
  %837 = ptrtoint ptr addrspace(1) %address.local531 to i64, !dbg !46
  %838 = sub i64 %837, %836, !dbg !46
  %dma.buffer.offset957 = trunc i64 %838 to i32, !dbg !46
  %839 = trunc i64 %836 to i32, !dbg !46
  %840 = insertelement <4 x i32> poison, i32 %839, i64 0, !dbg !46
  %841 = lshr i64 %836, 32, !dbg !46
  %842 = trunc i64 %841 to i32, !dbg !46
  %843 = insertelement <4 x i32> %840, i32 %842, i64 1, !dbg !46
  %844 = insertelement <4 x i32> %843, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc958 = insertelement <4 x i32> %844, i32 131072, i64 3, !dbg !46
  %845 = ptrtoint ptr addrspace(3) %835 to i32, !dbg !46
  %dma.buffer.lds959 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %845), !dbg !46
  %846 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds959, i32 %dma.buffer.offset957, <4 x i32> %dma.buffer.rsrc958) #10, !dbg !46
  %847 = shufflevector <8 x half> %820, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %848 = extractvalue [2 x <8 x float>] %207, 0, !dbg !62
  %849 = shufflevector <8 x float> %848, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %850 = fmul <4 x float> %849, %679, !dbg !62
  %851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %847, <4 x float> %850) #9, !dbg !43
  %852 = shufflevector <8 x half> %822, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %853 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %852, <4 x float> %851) #9, !dbg !43
  %854 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %855 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %854, <4 x float> %853) #9, !dbg !43
  %856 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %857 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %856, <4 x float> %855) #9, !dbg !43
  %858 = shufflevector <8 x half> %820, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %859 = shufflevector <8 x float> %848, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %860 = fmul <4 x float> %859, %679, !dbg !62
  %861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %858, <4 x float> %860) #9, !dbg !43
  %862 = shufflevector <8 x half> %822, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %863 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %862, <4 x float> %861) #9, !dbg !43
  %864 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %865 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %864, <4 x float> %863) #9, !dbg !43
  %866 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %867 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %866, <4 x float> %865) #9, !dbg !43
  %868 = extractvalue [2 x <8 x float>] %207, 1, !dbg !62
  %869 = shufflevector <8 x float> %868, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %870 = fmul <4 x float> %869, %687, !dbg !62
  %871 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %847, <4 x float> %870) #9, !dbg !43
  %872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %852, <4 x float> %871) #9, !dbg !43
  %873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %854, <4 x float> %872) #9, !dbg !43
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %856, <4 x float> %873) #9, !dbg !43
  %875 = shufflevector <8 x float> %868, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %876 = fmul <4 x float> %875, %687, !dbg !62
  %877 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %858, <4 x float> %876) #9, !dbg !43
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %862, <4 x float> %877) #9, !dbg !43
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %864, <4 x float> %878) #9, !dbg !43
  %880 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %866, <4 x float> %879) #9, !dbg !43
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %881 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !43
  %882 = bitcast <4 x i32> %881 to <8 x half>, !dbg !43
  %883 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !43
  %884 = bitcast <4 x i32> %883 to <8 x half>, !dbg !43
  %885 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !43
  %886 = bitcast <4 x i32> %885 to <8 x half>, !dbg !43
  %887 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !43
  %888 = bitcast <4 x i32> %887 to <8 x half>, !dbg !43
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %889 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %830, !dbg !46
  %890 = getelementptr i8, ptr addrspace(1) %889, i64 128, !dbg !46
  %891 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !46
  %892 = zext i32 %891 to i64, !dbg !46
  %893 = inttoptr i64 %892 to ptr addrspace(3), !dbg !46
  %address.local532 = getelementptr [2 x i8], ptr addrspace(1) %890, i64 %address.local530, !dbg !46
  %894 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !46
  %895 = ptrtoint ptr addrspace(1) %address.local532 to i64, !dbg !46
  %896 = sub i64 %895, %894, !dbg !46
  %dma.buffer.offset954 = trunc i64 %896 to i32, !dbg !46
  %897 = trunc i64 %894 to i32, !dbg !46
  %898 = insertelement <4 x i32> poison, i32 %897, i64 0, !dbg !46
  %899 = lshr i64 %894, 32, !dbg !46
  %900 = trunc i64 %899 to i32, !dbg !46
  %901 = insertelement <4 x i32> %898, i32 %900, i64 1, !dbg !46
  %902 = insertelement <4 x i32> %901, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc955 = insertelement <4 x i32> %902, i32 131072, i64 3, !dbg !46
  %903 = ptrtoint ptr addrspace(3) %893 to i32, !dbg !46
  %dma.buffer.lds956 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %903), !dbg !46
  %904 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds956, i32 %dma.buffer.offset954, <4 x i32> %dma.buffer.rsrc955) #10, !dbg !46
  %905 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %906 = extractvalue [2 x <8 x float>] %210, 0, !dbg !62
  %907 = shufflevector <8 x float> %906, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %908 = fmul <4 x float> %907, %679, !dbg !62
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %905, <4 x float> %908) #9, !dbg !43
  %910 = shufflevector <8 x half> %884, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %910, <4 x float> %909) #9, !dbg !43
  %912 = shufflevector <8 x half> %886, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %912, <4 x float> %911) #9, !dbg !43
  %914 = shufflevector <8 x half> %888, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %914, <4 x float> %913) #9, !dbg !43
  %916 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %917 = shufflevector <8 x float> %906, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %918 = fmul <4 x float> %917, %679, !dbg !62
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %916, <4 x float> %918) #9, !dbg !43
  %920 = shufflevector <8 x half> %884, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %920, <4 x float> %919) #9, !dbg !43
  %922 = shufflevector <8 x half> %886, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %922, <4 x float> %921) #9, !dbg !43
  %924 = shufflevector <8 x half> %888, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %924, <4 x float> %923) #9, !dbg !43
  %926 = extractvalue [2 x <8 x float>] %210, 1, !dbg !62
  %927 = shufflevector <8 x float> %926, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %928 = fmul <4 x float> %927, %687, !dbg !62
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %905, <4 x float> %928) #9, !dbg !43
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %910, <4 x float> %929) #9, !dbg !43
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %912, <4 x float> %930) #9, !dbg !43
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %914, <4 x float> %931) #9, !dbg !43
  %933 = shufflevector <8 x float> %926, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %934 = fmul <4 x float> %933, %687, !dbg !62
  %935 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %916, <4 x float> %934) #9, !dbg !43
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %920, <4 x float> %935) #9, !dbg !43
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %922, <4 x float> %936) #9, !dbg !43
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %924, <4 x float> %937) #9, !dbg !43
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %939 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !43
  %940 = bitcast <4 x i32> %939 to <8 x half>, !dbg !43
  %941 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !43
  %942 = bitcast <4 x i32> %941 to <8 x half>, !dbg !43
  %943 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !43
  %944 = bitcast <4 x i32> %943 to <8 x half>, !dbg !43
  %945 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !43
  %946 = bitcast <4 x i32> %945 to <8 x half>, !dbg !43
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %947 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %830, !dbg !46
  %948 = getelementptr i8, ptr addrspace(1) %947, i64 192, !dbg !46
  %949 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local525), !dbg !46
  %950 = zext i32 %949 to i64, !dbg !46
  %951 = inttoptr i64 %950 to ptr addrspace(3), !dbg !46
  %address.local533 = getelementptr [2 x i8], ptr addrspace(1) %948, i64 %address.local530, !dbg !46
  %952 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !46
  %953 = ptrtoint ptr addrspace(1) %address.local533 to i64, !dbg !46
  %954 = sub i64 %953, %952, !dbg !46
  %dma.buffer.offset = trunc i64 %954 to i32, !dbg !46
  %955 = trunc i64 %952 to i32, !dbg !46
  %956 = insertelement <4 x i32> poison, i32 %955, i64 0, !dbg !46
  %957 = lshr i64 %952, 32, !dbg !46
  %958 = trunc i64 %957 to i32, !dbg !46
  %959 = insertelement <4 x i32> %956, i32 %958, i64 1, !dbg !46
  %960 = insertelement <4 x i32> %959, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc = insertelement <4 x i32> %960, i32 131072, i64 3, !dbg !46
  %961 = ptrtoint ptr addrspace(3) %951 to i32, !dbg !46
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %961), !dbg !46
  %962 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !46
  %963 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %964 = extractvalue [2 x <8 x float>] %211, 0, !dbg !62
  %965 = shufflevector <8 x float> %964, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %966 = fmul <4 x float> %965, %679, !dbg !62
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %963, <4 x float> %966) #9, !dbg !43
  %968 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %968, <4 x float> %967) #9, !dbg !43
  %970 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %970, <4 x float> %969) #9, !dbg !43
  %972 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %972, <4 x float> %971) #9, !dbg !43
  %974 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %975 = shufflevector <8 x float> %964, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %976 = fmul <4 x float> %975, %679, !dbg !62
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %974, <4 x float> %976) #9, !dbg !43
  %978 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %978, <4 x float> %977) #9, !dbg !43
  %980 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %980, <4 x float> %979) #9, !dbg !43
  %982 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %982, <4 x float> %981) #9, !dbg !43
  %984 = extractvalue [2 x <8 x float>] %211, 1, !dbg !62
  %985 = shufflevector <8 x float> %984, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %986 = fmul <4 x float> %985, %687, !dbg !62
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %963, <4 x float> %986) #9, !dbg !43
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %968, <4 x float> %987) #9, !dbg !43
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %970, <4 x float> %988) #9, !dbg !43
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %972, <4 x float> %989) #9, !dbg !43
  %991 = shufflevector <8 x float> %984, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %992 = fmul <4 x float> %991, %687, !dbg !62
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %974, <4 x float> %992) #9, !dbg !43
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %978, <4 x float> %993) #9, !dbg !43
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %980, <4 x float> %994) #9, !dbg !43
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %982, <4 x float> %995) #9, !dbg !43
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %997 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !43
  %998 = bitcast <4 x i32> %997 to <8 x half>, !dbg !43
  %999 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local512, i32 0) #0, !dbg !43
  %1000 = bitcast <4 x i32> %999 to <8 x half>, !dbg !43
  %1001 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local516, i32 0) #0, !dbg !43
  %1002 = bitcast <4 x i32> %1001 to <8 x half>, !dbg !43
  %1003 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local520, i32 0) #0, !dbg !43
  %1004 = bitcast <4 x i32> %1003 to <8 x half>, !dbg !43
  %1005 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %1006 = extractvalue [2 x <8 x float>] %212, 0, !dbg !62
  %1007 = shufflevector <8 x float> %1006, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %1008 = fmul <4 x float> %1007, %679, !dbg !62
  %1009 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1005, <4 x float> %1008) #9, !dbg !43
  %1010 = shufflevector <8 x half> %1000, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %1011 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1010, <4 x float> %1009) #9, !dbg !43
  %1012 = shufflevector <8 x half> %1002, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %1013 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1012, <4 x float> %1011) #9, !dbg !43
  %1014 = shufflevector <8 x half> %1004, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1014, <4 x float> %1013) #9, !dbg !43
  %1016 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %1017 = shufflevector <8 x float> %1006, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %1018 = fmul <4 x float> %1017, %679, !dbg !62
  %1019 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1016, <4 x float> %1018) #9, !dbg !43
  %1020 = shufflevector <8 x half> %1000, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1020, <4 x float> %1019) #9, !dbg !43
  %1022 = shufflevector <8 x half> %1002, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1022, <4 x float> %1021) #9, !dbg !43
  %1024 = shufflevector <8 x half> %1004, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1024, <4 x float> %1023) #9, !dbg !43
  %1026 = extractvalue [2 x <8 x float>] %212, 1, !dbg !62
  %1027 = shufflevector <8 x float> %1026, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !62
  %1028 = fmul <4 x float> %1027, %687, !dbg !62
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1005, <4 x float> %1028) #9, !dbg !43
  %1030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1010, <4 x float> %1029) #9, !dbg !43
  %1031 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1012, <4 x float> %1030) #9, !dbg !43
  %1032 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1014, <4 x float> %1031) #9, !dbg !43
  %1033 = shufflevector <8 x float> %1026, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !62
  %1034 = fmul <4 x float> %1033, %687, !dbg !62
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1016, <4 x float> %1034) #9, !dbg !43
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1020, <4 x float> %1035) #9, !dbg !43
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1022, <4 x float> %1036) #9, !dbg !43
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed175, <4 x half> %1024, <4 x float> %1037) #9, !dbg !43
  %1039 = shufflevector <4 x float> %857, <4 x float> %867, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1040 = insertvalue [2 x <8 x float>] poison, <8 x float> %1039, 0, !dbg !38
  %1041 = shufflevector <4 x float> %874, <4 x float> %880, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1042 = insertvalue [2 x <8 x float>] %1040, <8 x float> %1041, 1, !dbg !38
  %1043 = insertvalue [2 x <1 x float>] poison, <1 x float> %651, 0, !dbg !38
  %1044 = insertvalue [2 x <1 x float>] %1043, <1 x float> %664, 1, !dbg !38
  %1045 = shufflevector <4 x float> %915, <4 x float> %925, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1046 = insertvalue [2 x <8 x float>] poison, <8 x float> %1045, 0, !dbg !38
  %1047 = shufflevector <4 x float> %932, <4 x float> %938, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1048 = insertvalue [2 x <8 x float>] %1046, <8 x float> %1047, 1, !dbg !38
  %1049 = shufflevector <4 x float> %973, <4 x float> %983, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1050 = insertvalue [2 x <8 x float>] poison, <8 x float> %1049, 0, !dbg !38
  %1051 = shufflevector <4 x float> %990, <4 x float> %996, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1052 = insertvalue [2 x <8 x float>] %1050, <8 x float> %1051, 1, !dbg !38
  %1053 = shufflevector <4 x float> %1015, <4 x float> %1025, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1054 = insertvalue [2 x <8 x float>] poison, <8 x float> %1053, 0, !dbg !38
  %1055 = shufflevector <4 x float> %1032, <4 x float> %1038, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !38
  %1056 = insertvalue [2 x <8 x float>] %1054, <8 x float> %1055, 1, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %1057 = add nuw nsw i64 %206, 1, !dbg !38
  br label %205, !dbg !38

1058:                                             ; preds = %205
  %1059 = and i32 %31, 16, !dbg !38
  %.not = icmp eq i32 %1059, 0, !dbg !38
  %1060 = and i32 %31, 32, !dbg !38
  %.not18 = icmp eq i32 %1060, 0, !dbg !38
  %1061 = extractvalue [2 x <8 x float>] %212, 0, !dbg !38
  %1062 = extractelement <8 x float> %1061, i64 0, !dbg !38
  %1063 = extractelement <8 x float> %1061, i64 1, !dbg !38
  %1064 = extractelement <8 x float> %1061, i64 2, !dbg !38
  %1065 = extractelement <8 x float> %1061, i64 3, !dbg !38
  %1066 = select i1 %.not, float %1063, float %1062, !dbg !38
  %1067 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !38
  %1068 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1067), !dbg !38
  %1069 = add nuw nsw i32 %1068, 64, !dbg !38
  %1070 = xor i32 %1068, 16, !dbg !38
  %1071 = and i32 %1069, 192, !dbg !38
  %1072 = icmp samesign ult i32 %1070, %1071, !dbg !38
  %1073 = select i1 %1072, i32 %1070, i32 %1068, !dbg !38
  %1074 = shl nuw nsw i32 %1073, 2, !dbg !38
  %1075 = bitcast float %1066 to i32, !dbg !38
  %1076 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1075), !dbg !38
  %1077 = bitcast i32 %1076 to float, !dbg !38
  %1078 = select i1 %.not, float %1062, float %1077, !dbg !38
  %1079 = select i1 %.not, float %1077, float %1063, !dbg !38
  %1080 = select i1 %.not, float %1065, float %1064, !dbg !38
  %1081 = bitcast float %1080 to i32, !dbg !38
  %1082 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1081), !dbg !38
  %1083 = bitcast i32 %1082 to float, !dbg !38
  %1084 = select i1 %.not, float %1064, float %1083, !dbg !38
  %1085 = select i1 %.not, float %1083, float %1065, !dbg !38
  %1086 = select i1 %.not18, float %1084, float %1078, !dbg !38
  %1087 = xor i32 %1068, 32, !dbg !38
  %1088 = icmp samesign ult i32 %1087, %1071, !dbg !38
  %1089 = select i1 %1088, i32 %1087, i32 %1068, !dbg !38
  %1090 = shl nuw nsw i32 %1089, 2, !dbg !38
  %1091 = bitcast float %1086 to i32, !dbg !38
  %1092 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1091), !dbg !38
  %1093 = bitcast i32 %1092 to float, !dbg !38
  %1094 = select i1 %.not18, float %1078, float %1093, !dbg !38
  %1095 = select i1 %.not18, float %1093, float %1084, !dbg !38
  %1096 = select i1 %.not18, float %1085, float %1079, !dbg !38
  %1097 = bitcast float %1096 to i32, !dbg !38
  %1098 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1097), !dbg !38
  %1099 = bitcast i32 %1098 to float, !dbg !38
  %1100 = select i1 %.not18, float %1079, float %1099, !dbg !38
  %1101 = select i1 %.not18, float %1099, float %1085, !dbg !38
  %1102 = extractvalue [2 x <8 x float>] %211, 0, !dbg !38
  %1103 = extractelement <8 x float> %1102, i64 0, !dbg !38
  %1104 = extractelement <8 x float> %1102, i64 1, !dbg !38
  %1105 = extractelement <8 x float> %1102, i64 2, !dbg !38
  %1106 = extractelement <8 x float> %1102, i64 3, !dbg !38
  %1107 = select i1 %.not, float %1104, float %1103, !dbg !38
  %1108 = bitcast float %1107 to i32, !dbg !38
  %1109 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1108), !dbg !38
  %1110 = bitcast i32 %1109 to float, !dbg !38
  %1111 = select i1 %.not, float %1103, float %1110, !dbg !38
  %1112 = select i1 %.not, float %1110, float %1104, !dbg !38
  %1113 = select i1 %.not, float %1106, float %1105, !dbg !38
  %1114 = bitcast float %1113 to i32, !dbg !38
  %1115 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1114), !dbg !38
  %1116 = bitcast i32 %1115 to float, !dbg !38
  %1117 = select i1 %.not, float %1105, float %1116, !dbg !38
  %1118 = select i1 %.not, float %1116, float %1106, !dbg !38
  %1119 = select i1 %.not18, float %1117, float %1111, !dbg !38
  %1120 = bitcast float %1119 to i32, !dbg !38
  %1121 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1120), !dbg !38
  %1122 = bitcast i32 %1121 to float, !dbg !38
  %1123 = select i1 %.not18, float %1111, float %1122, !dbg !38
  %1124 = select i1 %.not18, float %1122, float %1117, !dbg !38
  %1125 = select i1 %.not18, float %1118, float %1112, !dbg !38
  %1126 = bitcast float %1125 to i32, !dbg !38
  %1127 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1126), !dbg !38
  %1128 = bitcast i32 %1127 to float, !dbg !38
  %1129 = select i1 %.not18, float %1112, float %1128, !dbg !38
  %1130 = select i1 %.not18, float %1128, float %1118, !dbg !38
  %1131 = extractvalue [2 x <8 x float>] %210, 0, !dbg !38
  %1132 = extractelement <8 x float> %1131, i64 0, !dbg !38
  %1133 = extractelement <8 x float> %1131, i64 1, !dbg !38
  %1134 = extractelement <8 x float> %1131, i64 2, !dbg !38
  %1135 = extractelement <8 x float> %1131, i64 3, !dbg !38
  %1136 = select i1 %.not, float %1133, float %1132, !dbg !38
  %1137 = bitcast float %1136 to i32, !dbg !38
  %1138 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1137), !dbg !38
  %1139 = bitcast i32 %1138 to float, !dbg !38
  %1140 = select i1 %.not, float %1132, float %1139, !dbg !38
  %1141 = select i1 %.not, float %1139, float %1133, !dbg !38
  %1142 = select i1 %.not, float %1135, float %1134, !dbg !38
  %1143 = bitcast float %1142 to i32, !dbg !38
  %1144 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1143), !dbg !38
  %1145 = bitcast i32 %1144 to float, !dbg !38
  %1146 = select i1 %.not, float %1134, float %1145, !dbg !38
  %1147 = select i1 %.not, float %1145, float %1135, !dbg !38
  %1148 = select i1 %.not18, float %1146, float %1140, !dbg !38
  %1149 = bitcast float %1148 to i32, !dbg !38
  %1150 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1149), !dbg !38
  %1151 = bitcast i32 %1150 to float, !dbg !38
  %1152 = select i1 %.not18, float %1140, float %1151, !dbg !38
  %1153 = select i1 %.not18, float %1151, float %1146, !dbg !38
  %1154 = select i1 %.not18, float %1147, float %1141, !dbg !38
  %1155 = bitcast float %1154 to i32, !dbg !38
  %1156 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1155), !dbg !38
  %1157 = bitcast i32 %1156 to float, !dbg !38
  %1158 = select i1 %.not18, float %1141, float %1157, !dbg !38
  %1159 = select i1 %.not18, float %1157, float %1147, !dbg !38
  %1160 = extractvalue [2 x <8 x float>] %207, 0, !dbg !38
  %1161 = extractelement <8 x float> %1160, i64 0, !dbg !38
  %1162 = extractelement <8 x float> %1160, i64 1, !dbg !38
  %1163 = extractelement <8 x float> %1160, i64 2, !dbg !38
  %1164 = extractelement <8 x float> %1160, i64 3, !dbg !38
  %1165 = select i1 %.not, float %1162, float %1161, !dbg !38
  %1166 = bitcast float %1165 to i32, !dbg !38
  %1167 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1166), !dbg !38
  %1168 = bitcast i32 %1167 to float, !dbg !38
  %1169 = select i1 %.not, float %1161, float %1168, !dbg !38
  %1170 = select i1 %.not, float %1168, float %1162, !dbg !38
  %1171 = select i1 %.not, float %1164, float %1163, !dbg !38
  %1172 = bitcast float %1171 to i32, !dbg !38
  %1173 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1172), !dbg !38
  %1174 = bitcast i32 %1173 to float, !dbg !38
  %1175 = select i1 %.not, float %1163, float %1174, !dbg !38
  %1176 = select i1 %.not, float %1174, float %1164, !dbg !38
  %1177 = select i1 %.not18, float %1175, float %1169, !dbg !38
  %1178 = bitcast float %1177 to i32, !dbg !38
  %1179 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1178), !dbg !38
  %1180 = select i1 %.not18, float %1176, float %1170, !dbg !38
  %1181 = bitcast float %1180 to i32, !dbg !38
  %1182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1181), !dbg !38
  %1183 = extractvalue [2 x <1 x float>] %208, 0, !dbg !47
  %1184 = shufflevector <1 x float> %1183, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !47
  %1185 = extractelement <8 x float> %1061, i64 4, !dbg !38
  %1186 = extractelement <8 x float> %1061, i64 5, !dbg !38
  %1187 = extractelement <8 x float> %1061, i64 6, !dbg !38
  %1188 = extractelement <8 x float> %1061, i64 7, !dbg !38
  %1189 = select i1 %.not, float %1186, float %1185, !dbg !38
  %1190 = bitcast float %1189 to i32, !dbg !38
  %1191 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1190), !dbg !38
  %1192 = bitcast i32 %1191 to float, !dbg !38
  %1193 = select i1 %.not, float %1185, float %1192, !dbg !38
  %1194 = select i1 %.not, float %1192, float %1186, !dbg !38
  %1195 = select i1 %.not, float %1188, float %1187, !dbg !38
  %1196 = bitcast float %1195 to i32, !dbg !38
  %1197 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1196), !dbg !38
  %1198 = bitcast i32 %1197 to float, !dbg !38
  %1199 = select i1 %.not, float %1187, float %1198, !dbg !38
  %1200 = select i1 %.not, float %1198, float %1188, !dbg !38
  %1201 = select i1 %.not18, float %1199, float %1193, !dbg !38
  %1202 = bitcast float %1201 to i32, !dbg !38
  %1203 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1202), !dbg !38
  %1204 = bitcast i32 %1203 to float, !dbg !38
  %1205 = select i1 %.not18, float %1193, float %1204, !dbg !38
  %1206 = select i1 %.not18, float %1204, float %1199, !dbg !38
  %1207 = select i1 %.not18, float %1200, float %1194, !dbg !38
  %1208 = bitcast float %1207 to i32, !dbg !38
  %1209 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1208), !dbg !38
  %1210 = bitcast i32 %1209 to float, !dbg !38
  %1211 = select i1 %.not18, float %1194, float %1210, !dbg !38
  %1212 = select i1 %.not18, float %1210, float %1200, !dbg !38
  %1213 = extractelement <8 x float> %1102, i64 4, !dbg !38
  %1214 = extractelement <8 x float> %1102, i64 5, !dbg !38
  %1215 = extractelement <8 x float> %1102, i64 6, !dbg !38
  %1216 = extractelement <8 x float> %1102, i64 7, !dbg !38
  %1217 = select i1 %.not, float %1214, float %1213, !dbg !38
  %1218 = bitcast float %1217 to i32, !dbg !38
  %1219 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1218), !dbg !38
  %1220 = bitcast i32 %1219 to float, !dbg !38
  %1221 = select i1 %.not, float %1213, float %1220, !dbg !38
  %1222 = select i1 %.not, float %1220, float %1214, !dbg !38
  %1223 = select i1 %.not, float %1216, float %1215, !dbg !38
  %1224 = bitcast float %1223 to i32, !dbg !38
  %1225 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1224), !dbg !38
  %1226 = bitcast i32 %1225 to float, !dbg !38
  %1227 = select i1 %.not, float %1215, float %1226, !dbg !38
  %1228 = select i1 %.not, float %1226, float %1216, !dbg !38
  %1229 = select i1 %.not18, float %1227, float %1221, !dbg !38
  %1230 = bitcast float %1229 to i32, !dbg !38
  %1231 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1230), !dbg !38
  %1232 = bitcast i32 %1231 to float, !dbg !38
  %1233 = select i1 %.not18, float %1221, float %1232, !dbg !38
  %1234 = select i1 %.not18, float %1232, float %1227, !dbg !38
  %1235 = select i1 %.not18, float %1228, float %1222, !dbg !38
  %1236 = bitcast float %1235 to i32, !dbg !38
  %1237 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1236), !dbg !38
  %1238 = bitcast i32 %1237 to float, !dbg !38
  %1239 = select i1 %.not18, float %1222, float %1238, !dbg !38
  %1240 = select i1 %.not18, float %1238, float %1228, !dbg !38
  %1241 = extractelement <8 x float> %1131, i64 4, !dbg !38
  %1242 = extractelement <8 x float> %1131, i64 5, !dbg !38
  %1243 = extractelement <8 x float> %1131, i64 6, !dbg !38
  %1244 = extractelement <8 x float> %1131, i64 7, !dbg !38
  %1245 = select i1 %.not, float %1242, float %1241, !dbg !38
  %1246 = bitcast float %1245 to i32, !dbg !38
  %1247 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1246), !dbg !38
  %1248 = bitcast i32 %1247 to float, !dbg !38
  %1249 = select i1 %.not, float %1241, float %1248, !dbg !38
  %1250 = select i1 %.not, float %1248, float %1242, !dbg !38
  %1251 = select i1 %.not, float %1244, float %1243, !dbg !38
  %1252 = bitcast float %1251 to i32, !dbg !38
  %1253 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1252), !dbg !38
  %1254 = bitcast i32 %1253 to float, !dbg !38
  %1255 = select i1 %.not, float %1243, float %1254, !dbg !38
  %1256 = select i1 %.not, float %1254, float %1244, !dbg !38
  %1257 = select i1 %.not18, float %1255, float %1249, !dbg !38
  %1258 = bitcast float %1257 to i32, !dbg !38
  %1259 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1258), !dbg !38
  %1260 = bitcast i32 %1259 to float, !dbg !38
  %1261 = select i1 %.not18, float %1249, float %1260, !dbg !38
  %1262 = select i1 %.not18, float %1260, float %1255, !dbg !38
  %1263 = select i1 %.not18, float %1256, float %1250, !dbg !38
  %1264 = bitcast float %1263 to i32, !dbg !38
  %1265 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1264), !dbg !38
  %1266 = bitcast i32 %1265 to float, !dbg !38
  %1267 = select i1 %.not18, float %1250, float %1266, !dbg !38
  %1268 = select i1 %.not18, float %1266, float %1256, !dbg !38
  %1269 = extractelement <8 x float> %1160, i64 4, !dbg !38
  %1270 = extractelement <8 x float> %1160, i64 5, !dbg !38
  %1271 = extractelement <8 x float> %1160, i64 6, !dbg !38
  %1272 = extractelement <8 x float> %1160, i64 7, !dbg !38
  %1273 = select i1 %.not, float %1270, float %1269, !dbg !38
  %1274 = bitcast float %1273 to i32, !dbg !38
  %1275 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1274), !dbg !38
  %1276 = bitcast i32 %1275 to float, !dbg !38
  %1277 = select i1 %.not, float %1269, float %1276, !dbg !38
  %1278 = select i1 %.not, float %1276, float %1270, !dbg !38
  %1279 = select i1 %.not, float %1272, float %1271, !dbg !38
  %1280 = bitcast float %1279 to i32, !dbg !38
  %1281 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1280), !dbg !38
  %1282 = bitcast i32 %1281 to float, !dbg !38
  %1283 = select i1 %.not, float %1271, float %1282, !dbg !38
  %1284 = select i1 %.not, float %1282, float %1272, !dbg !38
  %1285 = select i1 %.not18, float %1283, float %1277, !dbg !38
  %1286 = bitcast float %1285 to i32, !dbg !38
  %1287 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1286), !dbg !38
  %1288 = select i1 %.not18, float %1284, float %1278, !dbg !38
  %1289 = bitcast float %1288 to i32, !dbg !38
  %1290 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1289), !dbg !38
  %1291 = extractvalue [2 x <8 x float>] %212, 1, !dbg !38
  %1292 = extractelement <8 x float> %1291, i64 0, !dbg !38
  %1293 = extractelement <8 x float> %1291, i64 1, !dbg !38
  %1294 = extractelement <8 x float> %1291, i64 2, !dbg !38
  %1295 = extractelement <8 x float> %1291, i64 3, !dbg !38
  %1296 = select i1 %.not, float %1293, float %1292, !dbg !38
  %1297 = bitcast float %1296 to i32, !dbg !38
  %1298 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1297), !dbg !38
  %1299 = bitcast i32 %1298 to float, !dbg !38
  %1300 = select i1 %.not, float %1292, float %1299, !dbg !38
  %1301 = select i1 %.not, float %1299, float %1293, !dbg !38
  %1302 = select i1 %.not, float %1295, float %1294, !dbg !38
  %1303 = bitcast float %1302 to i32, !dbg !38
  %1304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1303), !dbg !38
  %1305 = bitcast i32 %1304 to float, !dbg !38
  %1306 = select i1 %.not, float %1294, float %1305, !dbg !38
  %1307 = select i1 %.not, float %1305, float %1295, !dbg !38
  %1308 = select i1 %.not18, float %1306, float %1300, !dbg !38
  %1309 = bitcast float %1308 to i32, !dbg !38
  %1310 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1309), !dbg !38
  %1311 = bitcast i32 %1310 to float, !dbg !38
  %1312 = select i1 %.not18, float %1300, float %1311, !dbg !38
  %1313 = select i1 %.not18, float %1311, float %1306, !dbg !38
  %1314 = select i1 %.not18, float %1307, float %1301, !dbg !38
  %1315 = bitcast float %1314 to i32, !dbg !38
  %1316 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1315), !dbg !38
  %1317 = bitcast i32 %1316 to float, !dbg !38
  %1318 = select i1 %.not18, float %1301, float %1317, !dbg !38
  %1319 = select i1 %.not18, float %1317, float %1307, !dbg !38
  %1320 = extractvalue [2 x <8 x float>] %211, 1, !dbg !38
  %1321 = extractelement <8 x float> %1320, i64 0, !dbg !38
  %1322 = extractelement <8 x float> %1320, i64 1, !dbg !38
  %1323 = extractelement <8 x float> %1320, i64 2, !dbg !38
  %1324 = extractelement <8 x float> %1320, i64 3, !dbg !38
  %1325 = select i1 %.not, float %1322, float %1321, !dbg !38
  %1326 = bitcast float %1325 to i32, !dbg !38
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1326), !dbg !38
  %1328 = bitcast i32 %1327 to float, !dbg !38
  %1329 = select i1 %.not, float %1321, float %1328, !dbg !38
  %1330 = select i1 %.not, float %1328, float %1322, !dbg !38
  %1331 = select i1 %.not, float %1324, float %1323, !dbg !38
  %1332 = bitcast float %1331 to i32, !dbg !38
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1332), !dbg !38
  %1334 = bitcast i32 %1333 to float, !dbg !38
  %1335 = select i1 %.not, float %1323, float %1334, !dbg !38
  %1336 = select i1 %.not, float %1334, float %1324, !dbg !38
  %1337 = select i1 %.not18, float %1335, float %1329, !dbg !38
  %1338 = bitcast float %1337 to i32, !dbg !38
  %1339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1338), !dbg !38
  %1340 = bitcast i32 %1339 to float, !dbg !38
  %1341 = select i1 %.not18, float %1329, float %1340, !dbg !38
  %1342 = select i1 %.not18, float %1340, float %1335, !dbg !38
  %1343 = select i1 %.not18, float %1336, float %1330, !dbg !38
  %1344 = bitcast float %1343 to i32, !dbg !38
  %1345 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1344), !dbg !38
  %1346 = bitcast i32 %1345 to float, !dbg !38
  %1347 = select i1 %.not18, float %1330, float %1346, !dbg !38
  %1348 = select i1 %.not18, float %1346, float %1336, !dbg !38
  %1349 = extractvalue [2 x <8 x float>] %210, 1, !dbg !38
  %1350 = extractelement <8 x float> %1349, i64 0, !dbg !38
  %1351 = extractelement <8 x float> %1349, i64 1, !dbg !38
  %1352 = extractelement <8 x float> %1349, i64 2, !dbg !38
  %1353 = extractelement <8 x float> %1349, i64 3, !dbg !38
  %1354 = select i1 %.not, float %1351, float %1350, !dbg !38
  %1355 = bitcast float %1354 to i32, !dbg !38
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1355), !dbg !38
  %1357 = bitcast i32 %1356 to float, !dbg !38
  %1358 = select i1 %.not, float %1350, float %1357, !dbg !38
  %1359 = select i1 %.not, float %1357, float %1351, !dbg !38
  %1360 = select i1 %.not, float %1353, float %1352, !dbg !38
  %1361 = bitcast float %1360 to i32, !dbg !38
  %1362 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1361), !dbg !38
  %1363 = bitcast i32 %1362 to float, !dbg !38
  %1364 = select i1 %.not, float %1352, float %1363, !dbg !38
  %1365 = select i1 %.not, float %1363, float %1353, !dbg !38
  %1366 = select i1 %.not18, float %1364, float %1358, !dbg !38
  %1367 = bitcast float %1366 to i32, !dbg !38
  %1368 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1367), !dbg !38
  %1369 = bitcast i32 %1368 to float, !dbg !38
  %1370 = select i1 %.not18, float %1358, float %1369, !dbg !38
  %1371 = select i1 %.not18, float %1369, float %1364, !dbg !38
  %1372 = select i1 %.not18, float %1365, float %1359, !dbg !38
  %1373 = bitcast float %1372 to i32, !dbg !38
  %1374 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1373), !dbg !38
  %1375 = bitcast i32 %1374 to float, !dbg !38
  %1376 = select i1 %.not18, float %1359, float %1375, !dbg !38
  %1377 = select i1 %.not18, float %1375, float %1365, !dbg !38
  %1378 = extractvalue [2 x <8 x float>] %207, 1, !dbg !38
  %1379 = extractelement <8 x float> %1378, i64 0, !dbg !38
  %1380 = extractelement <8 x float> %1378, i64 1, !dbg !38
  %1381 = extractelement <8 x float> %1378, i64 2, !dbg !38
  %1382 = extractelement <8 x float> %1378, i64 3, !dbg !38
  %1383 = select i1 %.not, float %1380, float %1379, !dbg !38
  %1384 = bitcast float %1383 to i32, !dbg !38
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1384), !dbg !38
  %1386 = bitcast i32 %1385 to float, !dbg !38
  %1387 = select i1 %.not, float %1379, float %1386, !dbg !38
  %1388 = select i1 %.not, float %1386, float %1380, !dbg !38
  %1389 = select i1 %.not, float %1382, float %1381, !dbg !38
  %1390 = bitcast float %1389 to i32, !dbg !38
  %1391 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1390), !dbg !38
  %1392 = bitcast i32 %1391 to float, !dbg !38
  %1393 = select i1 %.not, float %1381, float %1392, !dbg !38
  %1394 = select i1 %.not, float %1392, float %1382, !dbg !38
  %1395 = select i1 %.not18, float %1393, float %1387, !dbg !38
  %1396 = bitcast float %1395 to i32, !dbg !38
  %1397 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1396), !dbg !38
  %1398 = select i1 %.not18, float %1394, float %1388, !dbg !38
  %1399 = bitcast float %1398 to i32, !dbg !38
  %1400 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1399), !dbg !38
  %1401 = extractvalue [2 x <1 x float>] %208, 1, !dbg !47
  %1402 = shufflevector <1 x float> %1401, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !47
  %1403 = extractelement <8 x float> %1291, i64 4, !dbg !38
  %1404 = extractelement <8 x float> %1291, i64 5, !dbg !38
  %1405 = extractelement <8 x float> %1291, i64 6, !dbg !38
  %1406 = extractelement <8 x float> %1291, i64 7, !dbg !38
  %1407 = select i1 %.not, float %1404, float %1403, !dbg !38
  %1408 = bitcast float %1407 to i32, !dbg !38
  %1409 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1408), !dbg !38
  %1410 = bitcast i32 %1409 to float, !dbg !38
  %1411 = select i1 %.not, float %1403, float %1410, !dbg !38
  %1412 = select i1 %.not, float %1410, float %1404, !dbg !38
  %1413 = select i1 %.not, float %1406, float %1405, !dbg !38
  %1414 = bitcast float %1413 to i32, !dbg !38
  %1415 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1414), !dbg !38
  %1416 = bitcast i32 %1415 to float, !dbg !38
  %1417 = select i1 %.not, float %1405, float %1416, !dbg !38
  %1418 = select i1 %.not, float %1416, float %1406, !dbg !38
  %1419 = select i1 %.not18, float %1417, float %1411, !dbg !38
  %1420 = bitcast float %1419 to i32, !dbg !38
  %1421 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1420), !dbg !38
  %1422 = bitcast i32 %1421 to float, !dbg !38
  %1423 = select i1 %.not18, float %1411, float %1422, !dbg !38
  %1424 = select i1 %.not18, float %1422, float %1417, !dbg !38
  %1425 = select i1 %.not18, float %1418, float %1412, !dbg !38
  %1426 = bitcast float %1425 to i32, !dbg !38
  %1427 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1426), !dbg !38
  %1428 = bitcast i32 %1427 to float, !dbg !38
  %1429 = select i1 %.not18, float %1412, float %1428, !dbg !38
  %1430 = select i1 %.not18, float %1428, float %1418, !dbg !38
  %1431 = extractelement <8 x float> %1320, i64 4, !dbg !38
  %1432 = extractelement <8 x float> %1320, i64 5, !dbg !38
  %1433 = extractelement <8 x float> %1320, i64 6, !dbg !38
  %1434 = extractelement <8 x float> %1320, i64 7, !dbg !38
  %1435 = select i1 %.not, float %1432, float %1431, !dbg !38
  %1436 = bitcast float %1435 to i32, !dbg !38
  %1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1436), !dbg !38
  %1438 = bitcast i32 %1437 to float, !dbg !38
  %1439 = select i1 %.not, float %1431, float %1438, !dbg !38
  %1440 = select i1 %.not, float %1438, float %1432, !dbg !38
  %1441 = select i1 %.not, float %1434, float %1433, !dbg !38
  %1442 = bitcast float %1441 to i32, !dbg !38
  %1443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1442), !dbg !38
  %1444 = bitcast i32 %1443 to float, !dbg !38
  %1445 = select i1 %.not, float %1433, float %1444, !dbg !38
  %1446 = select i1 %.not, float %1444, float %1434, !dbg !38
  %1447 = select i1 %.not18, float %1445, float %1439, !dbg !38
  %1448 = bitcast float %1447 to i32, !dbg !38
  %1449 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1448), !dbg !38
  %1450 = bitcast i32 %1449 to float, !dbg !38
  %1451 = select i1 %.not18, float %1439, float %1450, !dbg !38
  %1452 = select i1 %.not18, float %1450, float %1445, !dbg !38
  %1453 = select i1 %.not18, float %1446, float %1440, !dbg !38
  %1454 = bitcast float %1453 to i32, !dbg !38
  %1455 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1454), !dbg !38
  %1456 = bitcast i32 %1455 to float, !dbg !38
  %1457 = select i1 %.not18, float %1440, float %1456, !dbg !38
  %1458 = select i1 %.not18, float %1456, float %1446, !dbg !38
  %1459 = extractelement <8 x float> %1349, i64 4, !dbg !38
  %1460 = extractelement <8 x float> %1349, i64 5, !dbg !38
  %1461 = extractelement <8 x float> %1349, i64 6, !dbg !38
  %1462 = extractelement <8 x float> %1349, i64 7, !dbg !38
  %1463 = select i1 %.not, float %1460, float %1459, !dbg !38
  %1464 = bitcast float %1463 to i32, !dbg !38
  %1465 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1464), !dbg !38
  %1466 = bitcast i32 %1465 to float, !dbg !38
  %1467 = select i1 %.not, float %1459, float %1466, !dbg !38
  %1468 = select i1 %.not, float %1466, float %1460, !dbg !38
  %1469 = select i1 %.not, float %1462, float %1461, !dbg !38
  %1470 = bitcast float %1469 to i32, !dbg !38
  %1471 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1470), !dbg !38
  %1472 = bitcast i32 %1471 to float, !dbg !38
  %1473 = select i1 %.not, float %1461, float %1472, !dbg !38
  %1474 = select i1 %.not, float %1472, float %1462, !dbg !38
  %1475 = select i1 %.not18, float %1473, float %1467, !dbg !38
  %1476 = bitcast float %1475 to i32, !dbg !38
  %1477 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1476), !dbg !38
  %1478 = bitcast i32 %1477 to float, !dbg !38
  %1479 = select i1 %.not18, float %1467, float %1478, !dbg !38
  %1480 = select i1 %.not18, float %1478, float %1473, !dbg !38
  %1481 = select i1 %.not18, float %1474, float %1468, !dbg !38
  %1482 = bitcast float %1481 to i32, !dbg !38
  %1483 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1482), !dbg !38
  %1484 = bitcast i32 %1483 to float, !dbg !38
  %1485 = select i1 %.not18, float %1468, float %1484, !dbg !38
  %1486 = select i1 %.not18, float %1484, float %1474, !dbg !38
  %1487 = extractelement <8 x float> %1378, i64 4, !dbg !38
  %1488 = extractelement <8 x float> %1378, i64 5, !dbg !38
  %1489 = extractelement <8 x float> %1378, i64 6, !dbg !38
  %1490 = extractelement <8 x float> %1378, i64 7, !dbg !38
  %1491 = select i1 %.not, float %1488, float %1487, !dbg !38
  %1492 = bitcast float %1491 to i32, !dbg !38
  %1493 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1492), !dbg !38
  %1494 = bitcast i32 %1493 to float, !dbg !38
  %1495 = select i1 %.not, float %1487, float %1494, !dbg !38
  %1496 = select i1 %.not, float %1494, float %1488, !dbg !38
  %1497 = select i1 %.not, float %1490, float %1489, !dbg !38
  %1498 = bitcast float %1497 to i32, !dbg !38
  %1499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1074, i32 %1498), !dbg !38
  %1500 = bitcast i32 %1499 to float, !dbg !38
  %1501 = select i1 %.not, float %1489, float %1500, !dbg !38
  %1502 = select i1 %.not, float %1500, float %1490, !dbg !38
  %1503 = select i1 %.not18, float %1501, float %1495, !dbg !38
  %1504 = bitcast float %1503 to i32, !dbg !38
  %1505 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1504), !dbg !38
  %1506 = select i1 %.not18, float %1502, float %1496, !dbg !38
  %1507 = bitcast float %1506 to i32, !dbg !38
  %1508 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1090, i32 %1507), !dbg !38
  br i1 true, label %1509, label %1573, !dbg !49

1509:                                             ; preds = %1058
  %1510 = bitcast i32 %1505 to float, !dbg !38
  %1511 = select i1 %.not18, float %1495, float %1510, !dbg !38
  %1512 = insertelement <4 x float> poison, float %1511, i64 0, !dbg !47
  %1513 = bitcast i32 %1508 to float, !dbg !38
  %1514 = select i1 %.not18, float %1496, float %1513, !dbg !38
  %1515 = insertelement <4 x float> %1512, float %1514, i64 1, !dbg !47
  %1516 = select i1 %.not18, float %1510, float %1501, !dbg !38
  %1517 = insertelement <4 x float> %1515, float %1516, i64 2, !dbg !47
  %1518 = select i1 %.not18, float %1513, float %1502, !dbg !38
  %1519 = insertelement <4 x float> %1517, float %1518, i64 3, !dbg !47
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1402, !dbg !47
  %normalized = fmul <4 x float> %1519, %row.reciprocal, !dbg !47
  %1520 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !48
  %1521 = extractelement <4 x half> %1520, i64 3, !dbg !48
  %1522 = extractelement <4 x half> %1520, i64 2, !dbg !48
  %1523 = extractelement <4 x half> %1520, i64 1, !dbg !48
  %1524 = extractelement <4 x half> %1520, i64 0, !dbg !48
  %1525 = bitcast i32 %1397 to float, !dbg !38
  %1526 = select i1 %.not18, float %1387, float %1525, !dbg !38
  %1527 = insertelement <4 x float> poison, float %1526, i64 0, !dbg !47
  %1528 = bitcast i32 %1400 to float, !dbg !38
  %1529 = select i1 %.not18, float %1388, float %1528, !dbg !38
  %1530 = insertelement <4 x float> %1527, float %1529, i64 1, !dbg !47
  %1531 = select i1 %.not18, float %1525, float %1393, !dbg !38
  %1532 = insertelement <4 x float> %1530, float %1531, i64 2, !dbg !47
  %1533 = select i1 %.not18, float %1528, float %1394, !dbg !38
  %1534 = insertelement <4 x float> %1532, float %1533, i64 3, !dbg !47
  %normalized54 = fmul <4 x float> %1534, %row.reciprocal, !dbg !47
  %1535 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !48
  %1536 = extractelement <4 x half> %1535, i64 3, !dbg !48
  %1537 = extractelement <4 x half> %1535, i64 2, !dbg !48
  %1538 = extractelement <4 x half> %1535, i64 1, !dbg !48
  %1539 = extractelement <4 x half> %1535, i64 0, !dbg !48
  %1540 = bitcast i32 %1287 to float, !dbg !38
  %1541 = select i1 %.not18, float %1277, float %1540, !dbg !38
  %1542 = insertelement <4 x float> poison, float %1541, i64 0, !dbg !47
  %1543 = bitcast i32 %1290 to float, !dbg !38
  %1544 = select i1 %.not18, float %1278, float %1543, !dbg !38
  %1545 = insertelement <4 x float> %1542, float %1544, i64 1, !dbg !47
  %1546 = select i1 %.not18, float %1540, float %1283, !dbg !38
  %1547 = insertelement <4 x float> %1545, float %1546, i64 2, !dbg !47
  %1548 = select i1 %.not18, float %1543, float %1284, !dbg !38
  %1549 = insertelement <4 x float> %1547, float %1548, i64 3, !dbg !47
  %row.reciprocal55 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1184, !dbg !47
  %normalized56 = fmul <4 x float> %1549, %row.reciprocal55, !dbg !47
  %1550 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !48
  %1551 = extractelement <4 x half> %1550, i64 3, !dbg !48
  %1552 = extractelement <4 x half> %1550, i64 2, !dbg !48
  %1553 = extractelement <4 x half> %1550, i64 1, !dbg !48
  %1554 = extractelement <4 x half> %1550, i64 0, !dbg !48
  %1555 = bitcast i32 %1179 to float, !dbg !38
  %1556 = select i1 %.not18, float %1169, float %1555, !dbg !38
  %1557 = insertelement <4 x float> poison, float %1556, i64 0, !dbg !47
  %1558 = bitcast i32 %1182 to float, !dbg !38
  %1559 = select i1 %.not18, float %1170, float %1558, !dbg !38
  %1560 = insertelement <4 x float> %1557, float %1559, i64 1, !dbg !47
  %1561 = select i1 %.not18, float %1555, float %1175, !dbg !38
  %1562 = insertelement <4 x float> %1560, float %1561, i64 2, !dbg !47
  %1563 = select i1 %.not18, float %1558, float %1176, !dbg !38
  %1564 = insertelement <4 x float> %1562, float %1563, i64 3, !dbg !47
  %normalized57 = fmul <4 x float> %1564, %row.reciprocal55, !dbg !47
  %1565 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !48
  %1566 = extractelement <4 x half> %1565, i64 3, !dbg !48
  %1567 = extractelement <4 x half> %1565, i64 2, !dbg !48
  %1568 = extractelement <4 x half> %1565, i64 1, !dbg !48
  %1569 = extractelement <4 x half> %1565, i64 0, !dbg !48
  %1570 = add nsw i64 %16, %22, !dbg !49
  %1571 = shl i64 %1570, 19, !dbg !49
  %1572 = and i64 %1571, 16252928, !dbg !49
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
  %address.local544 = add nuw nsw i64 %1572, %address.local543, !dbg !49
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local546 = lshr i64 %address.local535, 2, !dbg !39
  %address.local547 = and i64 %address.local546, 12, !dbg !39
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local547, !dbg !49
  store half %1569, ptr addrspace(1) %address.local548, align 2, !dbg !49
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local550 = or disjoint i64 %address.local547, 1, !dbg !49
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local549, i64 %address.local550, !dbg !49
  store half %1568, ptr addrspace(1) %address.local551, align 2, !dbg !49
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local553 = or disjoint i64 %address.local547, 2, !dbg !49
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local552, i64 %address.local553, !dbg !49
  store half %1567, ptr addrspace(1) %address.local554, align 2, !dbg !49
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local556 = or disjoint i64 %address.local547, 3, !dbg !49
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local555, i64 %address.local556, !dbg !49
  store half %1566, ptr addrspace(1) %address.local557, align 2, !dbg !49
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local559 = or disjoint i64 %address.local547, 16, !dbg !49
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local558, i64 %address.local559, !dbg !49
  store half %1554, ptr addrspace(1) %address.local560, align 2, !dbg !49
  %address.local561 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local562 = or disjoint i64 %address.local547, 17, !dbg !49
  %address.local563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local561, i64 %address.local562, !dbg !49
  store half %1553, ptr addrspace(1) %address.local563, align 2, !dbg !49
  %address.local564 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local565 = or disjoint i64 %address.local547, 18, !dbg !49
  %address.local566 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local564, i64 %address.local565, !dbg !49
  store half %1552, ptr addrspace(1) %address.local566, align 2, !dbg !49
  %address.local567 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !49
  %address.local568 = or disjoint i64 %address.local547, 19, !dbg !49
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local567, i64 %address.local568, !dbg !49
  store half %1551, ptr addrspace(1) %address.local569, align 2, !dbg !49
  %address.local570 = shl i64 %address.local542, 7, !dbg !49
  %address.local571 = add i64 %address.local570, 8192, !dbg !49
  %address.local572 = add nuw nsw i64 %1572, %address.local571, !dbg !49
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local573, i64 %address.local547, !dbg !49
  store half %1539, ptr addrspace(1) %address.local574, align 2, !dbg !49
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local550, !dbg !49
  store half %1538, ptr addrspace(1) %address.local576, align 2, !dbg !49
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local553, !dbg !49
  store half %1537, ptr addrspace(1) %address.local578, align 2, !dbg !49
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local579, i64 %address.local556, !dbg !49
  store half %1536, ptr addrspace(1) %address.local580, align 2, !dbg !49
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local559, !dbg !49
  store half %1524, ptr addrspace(1) %address.local582, align 2, !dbg !49
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local562, !dbg !49
  store half %1523, ptr addrspace(1) %address.local584, align 2, !dbg !49
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local585, i64 %address.local565, !dbg !49
  store half %1522, ptr addrspace(1) %address.local586, align 2, !dbg !49
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local572, !dbg !49
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local587, i64 %address.local568, !dbg !49
  store half %1521, ptr addrspace(1) %address.local588, align 2, !dbg !49
  br label %1573, !dbg !49

1573:                                             ; preds = %1509, %1058
  br i1 true, label %1574, label %1612, !dbg !49

1574:                                             ; preds = %1573
  %1575 = insertelement <4 x float> poison, float %1479, i64 0, !dbg !47
  %1576 = insertelement <4 x float> %1575, float %1485, i64 1, !dbg !47
  %1577 = insertelement <4 x float> %1576, float %1480, i64 2, !dbg !47
  %1578 = insertelement <4 x float> %1577, float %1486, i64 3, !dbg !47
  %row.reciprocal58 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1402, !dbg !47
  %normalized59 = fmul <4 x float> %1578, %row.reciprocal58, !dbg !47
  %1579 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !48
  %1580 = extractelement <4 x half> %1579, i64 3, !dbg !48
  %1581 = extractelement <4 x half> %1579, i64 2, !dbg !48
  %1582 = extractelement <4 x half> %1579, i64 1, !dbg !48
  %1583 = extractelement <4 x half> %1579, i64 0, !dbg !48
  %1584 = insertelement <4 x float> poison, float %1370, i64 0, !dbg !47
  %1585 = insertelement <4 x float> %1584, float %1376, i64 1, !dbg !47
  %1586 = insertelement <4 x float> %1585, float %1371, i64 2, !dbg !47
  %1587 = insertelement <4 x float> %1586, float %1377, i64 3, !dbg !47
  %normalized60 = fmul <4 x float> %1587, %row.reciprocal58, !dbg !47
  %1588 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !48
  %1589 = extractelement <4 x half> %1588, i64 3, !dbg !48
  %1590 = extractelement <4 x half> %1588, i64 2, !dbg !48
  %1591 = extractelement <4 x half> %1588, i64 1, !dbg !48
  %1592 = extractelement <4 x half> %1588, i64 0, !dbg !48
  %1593 = insertelement <4 x float> poison, float %1261, i64 0, !dbg !47
  %1594 = insertelement <4 x float> %1593, float %1267, i64 1, !dbg !47
  %1595 = insertelement <4 x float> %1594, float %1262, i64 2, !dbg !47
  %1596 = insertelement <4 x float> %1595, float %1268, i64 3, !dbg !47
  %row.reciprocal61 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1184, !dbg !47
  %normalized62 = fmul <4 x float> %1596, %row.reciprocal61, !dbg !47
  %1597 = fptrunc <4 x float> %normalized62 to <4 x half>, !dbg !48
  %1598 = extractelement <4 x half> %1597, i64 3, !dbg !48
  %1599 = extractelement <4 x half> %1597, i64 2, !dbg !48
  %1600 = extractelement <4 x half> %1597, i64 1, !dbg !48
  %1601 = extractelement <4 x half> %1597, i64 0, !dbg !48
  %1602 = insertelement <4 x float> poison, float %1152, i64 0, !dbg !47
  %1603 = insertelement <4 x float> %1602, float %1158, i64 1, !dbg !47
  %1604 = insertelement <4 x float> %1603, float %1153, i64 2, !dbg !47
  %1605 = insertelement <4 x float> %1604, float %1159, i64 3, !dbg !47
  %normalized63 = fmul <4 x float> %1605, %row.reciprocal61, !dbg !47
  %1606 = fptrunc <4 x float> %normalized63 to <4 x half>, !dbg !48
  %1607 = extractelement <4 x half> %1606, i64 3, !dbg !48
  %1608 = extractelement <4 x half> %1606, i64 2, !dbg !48
  %1609 = extractelement <4 x half> %1606, i64 1, !dbg !48
  %1610 = extractelement <4 x half> %1606, i64 0, !dbg !48
  %1611 = shl nuw nsw i64 %24, 19, !dbg !49
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
  %address.local599 = add nuw nsw i64 %1611, %address.local598, !dbg !49
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local601 = lshr i64 %address.local590, 2, !dbg !39
  %address.local602 = and i64 %address.local601, 12, !dbg !39
  %address.local603 = or disjoint i64 %address.local602, 32, !dbg !49
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local603, !dbg !49
  store half %1610, ptr addrspace(1) %address.local604, align 2, !dbg !49
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local606 = or disjoint i64 %address.local602, 33, !dbg !49
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local605, i64 %address.local606, !dbg !49
  store half %1609, ptr addrspace(1) %address.local607, align 2, !dbg !49
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local609 = or disjoint i64 %address.local602, 34, !dbg !49
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local608, i64 %address.local609, !dbg !49
  store half %1608, ptr addrspace(1) %address.local610, align 2, !dbg !49
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local612 = or disjoint i64 %address.local602, 35, !dbg !49
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local611, i64 %address.local612, !dbg !49
  store half %1607, ptr addrspace(1) %address.local613, align 2, !dbg !49
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local615 = or disjoint i64 %address.local602, 48, !dbg !49
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local614, i64 %address.local615, !dbg !49
  store half %1601, ptr addrspace(1) %address.local616, align 2, !dbg !49
  %address.local617 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local618 = or disjoint i64 %address.local602, 49, !dbg !49
  %address.local619 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local617, i64 %address.local618, !dbg !49
  store half %1600, ptr addrspace(1) %address.local619, align 2, !dbg !49
  %address.local620 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local621 = or disjoint i64 %address.local602, 50, !dbg !49
  %address.local622 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local621, !dbg !49
  store half %1599, ptr addrspace(1) %address.local622, align 2, !dbg !49
  %address.local623 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !49
  %address.local624 = or i64 %address.local601, 51, !dbg !49
  %address.local625 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local623, i64 %address.local624, !dbg !49
  store half %1598, ptr addrspace(1) %address.local625, align 2, !dbg !49
  %address.local626 = shl i64 %address.local597, 7, !dbg !49
  %address.local627 = add i64 %address.local626, 8192, !dbg !49
  %address.local628 = add nuw nsw i64 %1611, %address.local627, !dbg !49
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local629, i64 %address.local603, !dbg !49
  store half %1592, ptr addrspace(1) %address.local630, align 2, !dbg !49
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local606, !dbg !49
  store half %1591, ptr addrspace(1) %address.local632, align 2, !dbg !49
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local633, i64 %address.local609, !dbg !49
  store half %1590, ptr addrspace(1) %address.local634, align 2, !dbg !49
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local635, i64 %address.local612, !dbg !49
  store half %1589, ptr addrspace(1) %address.local636, align 2, !dbg !49
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local637, i64 %address.local615, !dbg !49
  store half %1583, ptr addrspace(1) %address.local638, align 2, !dbg !49
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local639, i64 %address.local618, !dbg !49
  store half %1582, ptr addrspace(1) %address.local640, align 2, !dbg !49
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local642 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local641, i64 %address.local621, !dbg !49
  store half %1581, ptr addrspace(1) %address.local642, align 2, !dbg !49
  %address.local643 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local628, !dbg !49
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local624, !dbg !49
  store half %1580, ptr addrspace(1) %address.local644, align 2, !dbg !49
  br label %1612, !dbg !49

1612:                                             ; preds = %1574, %1573
  br i1 true, label %1613, label %1651, !dbg !49

1613:                                             ; preds = %1612
  %1614 = insertelement <4 x float> poison, float %1451, i64 0, !dbg !47
  %1615 = insertelement <4 x float> %1614, float %1457, i64 1, !dbg !47
  %1616 = insertelement <4 x float> %1615, float %1452, i64 2, !dbg !47
  %1617 = insertelement <4 x float> %1616, float %1458, i64 3, !dbg !47
  %row.reciprocal64 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1402, !dbg !47
  %normalized65 = fmul <4 x float> %1617, %row.reciprocal64, !dbg !47
  %1618 = fptrunc <4 x float> %normalized65 to <4 x half>, !dbg !48
  %1619 = extractelement <4 x half> %1618, i64 3, !dbg !48
  %1620 = extractelement <4 x half> %1618, i64 2, !dbg !48
  %1621 = extractelement <4 x half> %1618, i64 1, !dbg !48
  %1622 = extractelement <4 x half> %1618, i64 0, !dbg !48
  %1623 = insertelement <4 x float> poison, float %1341, i64 0, !dbg !47
  %1624 = insertelement <4 x float> %1623, float %1347, i64 1, !dbg !47
  %1625 = insertelement <4 x float> %1624, float %1342, i64 2, !dbg !47
  %1626 = insertelement <4 x float> %1625, float %1348, i64 3, !dbg !47
  %normalized66 = fmul <4 x float> %1626, %row.reciprocal64, !dbg !47
  %1627 = fptrunc <4 x float> %normalized66 to <4 x half>, !dbg !48
  %1628 = extractelement <4 x half> %1627, i64 3, !dbg !48
  %1629 = extractelement <4 x half> %1627, i64 2, !dbg !48
  %1630 = extractelement <4 x half> %1627, i64 1, !dbg !48
  %1631 = extractelement <4 x half> %1627, i64 0, !dbg !48
  %1632 = insertelement <4 x float> poison, float %1233, i64 0, !dbg !47
  %1633 = insertelement <4 x float> %1632, float %1239, i64 1, !dbg !47
  %1634 = insertelement <4 x float> %1633, float %1234, i64 2, !dbg !47
  %1635 = insertelement <4 x float> %1634, float %1240, i64 3, !dbg !47
  %row.reciprocal67 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1184, !dbg !47
  %normalized68 = fmul <4 x float> %1635, %row.reciprocal67, !dbg !47
  %1636 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !48
  %1637 = extractelement <4 x half> %1636, i64 3, !dbg !48
  %1638 = extractelement <4 x half> %1636, i64 2, !dbg !48
  %1639 = extractelement <4 x half> %1636, i64 1, !dbg !48
  %1640 = extractelement <4 x half> %1636, i64 0, !dbg !48
  %1641 = insertelement <4 x float> poison, float %1123, i64 0, !dbg !47
  %1642 = insertelement <4 x float> %1641, float %1129, i64 1, !dbg !47
  %1643 = insertelement <4 x float> %1642, float %1124, i64 2, !dbg !47
  %1644 = insertelement <4 x float> %1643, float %1130, i64 3, !dbg !47
  %normalized69 = fmul <4 x float> %1644, %row.reciprocal67, !dbg !47
  %1645 = fptrunc <4 x float> %normalized69 to <4 x half>, !dbg !48
  %1646 = extractelement <4 x half> %1645, i64 3, !dbg !48
  %1647 = extractelement <4 x half> %1645, i64 2, !dbg !48
  %1648 = extractelement <4 x half> %1645, i64 1, !dbg !48
  %1649 = extractelement <4 x half> %1645, i64 0, !dbg !48
  %1650 = shl nuw nsw i64 %24, 19, !dbg !49
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
  %address.local655 = add nuw nsw i64 %1650, %address.local654, !dbg !49
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local657 = lshr i64 %address.local646, 2, !dbg !39
  %address.local658 = and i64 %address.local657, 12, !dbg !39
  %address.local659 = or disjoint i64 %address.local658, 64, !dbg !49
  %address.local660 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local656, i64 %address.local659, !dbg !49
  store half %1649, ptr addrspace(1) %address.local660, align 2, !dbg !49
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local662 = or disjoint i64 %address.local658, 65, !dbg !49
  %address.local663 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local661, i64 %address.local662, !dbg !49
  store half %1648, ptr addrspace(1) %address.local663, align 2, !dbg !49
  %address.local664 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local665 = or disjoint i64 %address.local658, 66, !dbg !49
  %address.local666 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local664, i64 %address.local665, !dbg !49
  store half %1647, ptr addrspace(1) %address.local666, align 2, !dbg !49
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local668 = or disjoint i64 %address.local658, 67, !dbg !49
  %address.local669 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local667, i64 %address.local668, !dbg !49
  store half %1646, ptr addrspace(1) %address.local669, align 2, !dbg !49
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local671 = or disjoint i64 %address.local658, 80, !dbg !49
  %address.local672 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local670, i64 %address.local671, !dbg !49
  store half %1640, ptr addrspace(1) %address.local672, align 2, !dbg !49
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local674 = or disjoint i64 %address.local658, 81, !dbg !49
  %address.local675 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local673, i64 %address.local674, !dbg !49
  store half %1639, ptr addrspace(1) %address.local675, align 2, !dbg !49
  %address.local676 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local677 = or disjoint i64 %address.local658, 82, !dbg !49
  %address.local678 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local676, i64 %address.local677, !dbg !49
  store half %1638, ptr addrspace(1) %address.local678, align 2, !dbg !49
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local655, !dbg !49
  %address.local680 = or disjoint i64 %address.local658, 83, !dbg !49
  %address.local681 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local679, i64 %address.local680, !dbg !49
  store half %1637, ptr addrspace(1) %address.local681, align 2, !dbg !49
  %address.local682 = shl i64 %address.local653, 7, !dbg !49
  %address.local683 = add i64 %address.local682, 8192, !dbg !49
  %address.local684 = add nuw nsw i64 %1650, %address.local683, !dbg !49
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local685, i64 %address.local659, !dbg !49
  store half %1631, ptr addrspace(1) %address.local686, align 2, !dbg !49
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local687, i64 %address.local662, !dbg !49
  store half %1630, ptr addrspace(1) %address.local688, align 2, !dbg !49
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local689, i64 %address.local665, !dbg !49
  store half %1629, ptr addrspace(1) %address.local690, align 2, !dbg !49
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local691, i64 %address.local668, !dbg !49
  store half %1628, ptr addrspace(1) %address.local692, align 2, !dbg !49
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local693, i64 %address.local671, !dbg !49
  store half %1622, ptr addrspace(1) %address.local694, align 2, !dbg !49
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local695, i64 %address.local674, !dbg !49
  store half %1621, ptr addrspace(1) %address.local696, align 2, !dbg !49
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local697, i64 %address.local677, !dbg !49
  store half %1620, ptr addrspace(1) %address.local698, align 2, !dbg !49
  %address.local699 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local684, !dbg !49
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local699, i64 %address.local680, !dbg !49
  store half %1619, ptr addrspace(1) %address.local700, align 2, !dbg !49
  br label %1651, !dbg !49

1651:                                             ; preds = %1613, %1612
  br i1 true, label %1652, label %1690, !dbg !49

1652:                                             ; preds = %1651
  %1653 = insertelement <4 x float> poison, float %1423, i64 0, !dbg !47
  %1654 = insertelement <4 x float> %1653, float %1429, i64 1, !dbg !47
  %1655 = insertelement <4 x float> %1654, float %1424, i64 2, !dbg !47
  %1656 = insertelement <4 x float> %1655, float %1430, i64 3, !dbg !47
  %row.reciprocal70 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1402, !dbg !47
  %normalized71 = fmul <4 x float> %1656, %row.reciprocal70, !dbg !47
  %1657 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !48
  %1658 = extractelement <4 x half> %1657, i64 3, !dbg !48
  %1659 = extractelement <4 x half> %1657, i64 2, !dbg !48
  %1660 = extractelement <4 x half> %1657, i64 1, !dbg !48
  %1661 = extractelement <4 x half> %1657, i64 0, !dbg !48
  %1662 = insertelement <4 x float> poison, float %1312, i64 0, !dbg !47
  %1663 = insertelement <4 x float> %1662, float %1318, i64 1, !dbg !47
  %1664 = insertelement <4 x float> %1663, float %1313, i64 2, !dbg !47
  %1665 = insertelement <4 x float> %1664, float %1319, i64 3, !dbg !47
  %normalized72 = fmul <4 x float> %1665, %row.reciprocal70, !dbg !47
  %1666 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !48
  %1667 = extractelement <4 x half> %1666, i64 3, !dbg !48
  %1668 = extractelement <4 x half> %1666, i64 2, !dbg !48
  %1669 = extractelement <4 x half> %1666, i64 1, !dbg !48
  %1670 = extractelement <4 x half> %1666, i64 0, !dbg !48
  %1671 = insertelement <4 x float> poison, float %1205, i64 0, !dbg !47
  %1672 = insertelement <4 x float> %1671, float %1211, i64 1, !dbg !47
  %1673 = insertelement <4 x float> %1672, float %1206, i64 2, !dbg !47
  %1674 = insertelement <4 x float> %1673, float %1212, i64 3, !dbg !47
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1184, !dbg !47
  %normalized74 = fmul <4 x float> %1674, %row.reciprocal73, !dbg !47
  %1675 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !48
  %1676 = extractelement <4 x half> %1675, i64 3, !dbg !48
  %1677 = extractelement <4 x half> %1675, i64 2, !dbg !48
  %1678 = extractelement <4 x half> %1675, i64 1, !dbg !48
  %1679 = extractelement <4 x half> %1675, i64 0, !dbg !48
  %1680 = insertelement <4 x float> poison, float %1094, i64 0, !dbg !47
  %1681 = insertelement <4 x float> %1680, float %1100, i64 1, !dbg !47
  %1682 = insertelement <4 x float> %1681, float %1095, i64 2, !dbg !47
  %1683 = insertelement <4 x float> %1682, float %1101, i64 3, !dbg !47
  %normalized75 = fmul <4 x float> %1683, %row.reciprocal73, !dbg !47
  %1684 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !48
  %1685 = extractelement <4 x half> %1684, i64 3, !dbg !48
  %1686 = extractelement <4 x half> %1684, i64 2, !dbg !48
  %1687 = extractelement <4 x half> %1684, i64 1, !dbg !48
  %1688 = extractelement <4 x half> %1684, i64 0, !dbg !48
  %1689 = shl nuw nsw i64 %24, 19, !dbg !49
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
  %address.local711 = add nuw nsw i64 %1689, %address.local710, !dbg !49
  %address.local712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local713 = lshr i64 %address.local702, 2, !dbg !39
  %address.local714 = and i64 %address.local713, 12, !dbg !39
  %address.local715 = or disjoint i64 %address.local714, 96, !dbg !49
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local712, i64 %address.local715, !dbg !49
  store half %1688, ptr addrspace(1) %address.local716, align 2, !dbg !49
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local718 = or disjoint i64 %address.local714, 97, !dbg !49
  %address.local719 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local717, i64 %address.local718, !dbg !49
  store half %1687, ptr addrspace(1) %address.local719, align 2, !dbg !49
  %address.local720 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local721 = or disjoint i64 %address.local714, 98, !dbg !49
  %address.local722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local720, i64 %address.local721, !dbg !49
  store half %1686, ptr addrspace(1) %address.local722, align 2, !dbg !49
  %address.local723 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local724 = or disjoint i64 %address.local714, 99, !dbg !49
  %address.local725 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local723, i64 %address.local724, !dbg !49
  store half %1685, ptr addrspace(1) %address.local725, align 2, !dbg !49
  %address.local726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local727 = or disjoint i64 %address.local714, 112, !dbg !49
  %address.local728 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local726, i64 %address.local727, !dbg !49
  store half %1679, ptr addrspace(1) %address.local728, align 2, !dbg !49
  %address.local729 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local730 = or disjoint i64 %address.local714, 113, !dbg !49
  %address.local731 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local729, i64 %address.local730, !dbg !49
  store half %1678, ptr addrspace(1) %address.local731, align 2, !dbg !49
  %address.local732 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local733 = or disjoint i64 %address.local714, 114, !dbg !49
  %address.local734 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local732, i64 %address.local733, !dbg !49
  store half %1677, ptr addrspace(1) %address.local734, align 2, !dbg !49
  %address.local735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local711, !dbg !49
  %address.local736 = or i64 %address.local713, 115, !dbg !49
  %address.local737 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local735, i64 %address.local736, !dbg !49
  store half %1676, ptr addrspace(1) %address.local737, align 2, !dbg !49
  %address.local738 = shl i64 %address.local709, 7, !dbg !49
  %address.local739 = add i64 %address.local738, 8192, !dbg !49
  %address.local740 = add nuw nsw i64 %1689, %address.local739, !dbg !49
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local741, i64 %address.local715, !dbg !49
  store half %1670, ptr addrspace(1) %address.local742, align 2, !dbg !49
  %address.local743 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local743, i64 %address.local718, !dbg !49
  store half %1669, ptr addrspace(1) %address.local744, align 2, !dbg !49
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local746 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local745, i64 %address.local721, !dbg !49
  store half %1668, ptr addrspace(1) %address.local746, align 2, !dbg !49
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local747, i64 %address.local724, !dbg !49
  store half %1667, ptr addrspace(1) %address.local748, align 2, !dbg !49
  %address.local749 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local749, i64 %address.local727, !dbg !49
  store half %1661, ptr addrspace(1) %address.local750, align 2, !dbg !49
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local751, i64 %address.local730, !dbg !49
  store half %1660, ptr addrspace(1) %address.local752, align 2, !dbg !49
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local754 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local753, i64 %address.local733, !dbg !49
  store half %1659, ptr addrspace(1) %address.local754, align 2, !dbg !49
  %address.local755 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local740, !dbg !49
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local755, i64 %address.local736, !dbg !49
  store half %1658, ptr addrspace(1) %address.local756, align 2, !dbg !49
  br label %1690, !dbg !49

1690:                                             ; preds = %1652, %1651
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
