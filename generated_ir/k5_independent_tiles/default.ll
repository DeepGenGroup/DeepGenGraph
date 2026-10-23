; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/k5_independent_tiles/default.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/k5_independent_tiles/default.mlir"

@shm_0 = addrspace(3) global [32768 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !32 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %6, 7, !dbg !10
  %.lobit = ashr i32 %5, 31, !dbg !12
  %10 = xor i32 %5, %.lobit, !dbg !12
  %11 = sdiv i32 %10, 32, !dbg !12
  %.lobit1 = ashr i32 %5, 31, !dbg !12
  %12 = xor i32 %11, %.lobit1, !dbg !12
  %13 = sext i32 %12 to i64, !dbg !12
  %14 = add nsw i64 %8, %13, !dbg !12
  %15 = and i64 %14, 31, !dbg !12
  %16 = shl nsw i64 %6, 7, !dbg !12
  %17 = mul nsw i64 %13, -4096, !dbg !12
  %18 = add nsw i64 %16, %17, !dbg !12
  %19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !34
  %20 = zext nneg i32 %19 to i64, !dbg !13
  %21 = lshr i64 %20, 6, !dbg !13
  %22 = or disjoint i64 %9, 127, !dbg !14
  %23 = add nsw i64 %9, 128, !dbg !14
  %24 = icmp uge i64 %23, %22, !dbg !14
  br label %25, !dbg !15

25:                                               ; preds = %600, %4
  %26 = phi i64 [ %1323, %600 ], [ 0, %4 ], !dbg !15
  %27 = phi [2 x <16 x float>] [ %1314, %600 ], [ zeroinitializer, %4 ], !dbg !15
  %28 = phi [2 x <1 x float>] [ %960, %600 ], [ zeroinitializer, %4 ], !dbg !15
  %29 = phi [2 x <16 x float>] [ %1322, %600 ], [ zeroinitializer, %4 ], !dbg !15
  %.not = icmp sgt i64 %26, %6, !dbg !15
  br i1 %.not, label %1324, label %30, !dbg !15

30:                                               ; preds = %25
  %31 = shl nsw i64 %26, 7, !dbg !15
  %32 = lshr i64 %26, 5, !dbg !16
  %33 = shl i64 %32, 12, !dbg !16
  %34 = sub i64 %31, %33, !dbg !16
  br label %35, !dbg !22

35:                                               ; preds = %39, %30
  %36 = phi i64 [ %359, %39 ], [ 0, %30 ], !dbg !22
  %37 = phi [2 x <32 x float>] [ %358, %39 ], [ zeroinitializer, %30 ], !dbg !22
  %38 = icmp samesign ult i64 %36, 2, !dbg !22
  br i1 %38, label %39, label %360, !dbg !22

39:                                               ; preds = %35
  %40 = shl nuw nsw i64 %36, 6, !dbg !22
  %.idx = shl nuw nsw i64 %15, 20, !dbg !12
  %41 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %.idx68 = shl nsw i64 %18, 8, !dbg !12
  %42 = getelementptr i8, ptr addrspace(1) %41, i64 %.idx68, !dbg !12
  %43 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %40, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !12, !range !34, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local145 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local146 = and i64 %address.local145, 1536, !dbg !12
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local146, !dbg !12
  %address.local148 = ptrtoint ptr addrspace(3) %address.local147 to i64, !dbg !12
  %address.local149 = trunc i64 %address.local148 to i32, !dbg !12
  %44 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local149), !dbg !12
  %45 = zext i32 %44 to i64, !dbg !12
  %46 = inttoptr i64 %45 to ptr addrspace(3), !dbg !12
  %address.local150 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local151 = and i64 %address.local150, 16256, !dbg !12
  %address.local152 = lshr i64 %address.local, 4, !dbg !12
  %address.local153 = and i64 %address.local152, 8, !dbg !12
  %address.local154 = or disjoint i64 %address.local151, %address.local153, !dbg !12
  %address.local155 = getelementptr [2 x i8], ptr addrspace(1) %43, i64 %address.local154, !dbg !12
  %47 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %48 = ptrtoint ptr addrspace(1) %address.local155 to i64, !dbg !12
  %49 = sub i64 %48, %47, !dbg !12
  %dma.buffer.offset859 = trunc i64 %49 to i32, !dbg !12
  %50 = trunc i64 %47 to i32, !dbg !12
  %51 = insertelement <4 x i32> poison, i32 %50, i64 0, !dbg !12
  %52 = lshr i64 %47, 32, !dbg !12
  %53 = trunc i64 %52 to i32, !dbg !12
  %54 = insertelement <4 x i32> %51, i32 %53, i64 1, !dbg !12
  %55 = insertelement <4 x i32> %54, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc860 = insertelement <4 x i32> %55, i32 131072, i64 3, !dbg !12
  %56 = ptrtoint ptr addrspace(3) %46 to i32, !dbg !12
  %dma.buffer.lds861 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %56), !dbg !12
  %57 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds861, i32 %dma.buffer.offset859, <4 x i32> %dma.buffer.rsrc860) #10, !dbg !12
  %address.local156 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local157 = shl nuw nsw i64 %address.local156, 3, !dbg !12
  %address.local158 = and i64 %address.local157, 3584, !dbg !12
  %address.local159 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local158, !dbg !12
  %address.local160 = ptrtoint ptr addrspace(3) %address.local159 to i64, !dbg !12
  %address.local161 = trunc i64 %address.local160 to i32, !dbg !12
  %58 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local161), !dbg !12
  %59 = zext i32 %58 to i64, !dbg !12
  %60 = inttoptr i64 %59 to ptr addrspace(3), !dbg !12
  %address.local162 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local163 = and i64 %address.local162, 16256, !dbg !12
  %address.local164 = lshr i64 %address.local156, 4, !dbg !12
  %address.local165 = and i64 %address.local164, 24, !dbg !12
  %address.local166 = or disjoint i64 %address.local163, %address.local165, !dbg !12
  %address.local167 = getelementptr [2 x i8], ptr addrspace(1) %43, i64 %address.local166, !dbg !12
  %61 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %62 = ptrtoint ptr addrspace(1) %address.local167 to i64, !dbg !12
  %63 = sub i64 %62, %61, !dbg !12
  %dma.buffer.offset856 = trunc i64 %63 to i32, !dbg !12
  %64 = trunc i64 %61 to i32, !dbg !12
  %65 = insertelement <4 x i32> poison, i32 %64, i64 0, !dbg !12
  %66 = lshr i64 %61, 32, !dbg !12
  %67 = trunc i64 %66 to i32, !dbg !12
  %68 = insertelement <4 x i32> %65, i32 %67, i64 1, !dbg !12
  %69 = insertelement <4 x i32> %68, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc857 = insertelement <4 x i32> %69, i32 131072, i64 3, !dbg !12
  %70 = ptrtoint ptr addrspace(3) %60 to i32, !dbg !12
  %dma.buffer.lds858 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %70), !dbg !12
  %71 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds858, i32 %dma.buffer.offset856, <4 x i32> %dma.buffer.rsrc857) #10, !dbg !12
  %address.local168 = or disjoint i64 %address.local, 512, !dbg !12
  %address.local169 = shl nuw nsw i64 %address.local168, 3, !dbg !12
  %address.local170 = and i64 %address.local169, 5632, !dbg !12
  %address.local171 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local170, !dbg !12
  %address.local172 = ptrtoint ptr addrspace(3) %address.local171 to i64, !dbg !12
  %address.local173 = trunc i64 %address.local172 to i32, !dbg !12
  %72 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local173), !dbg !12
  %73 = zext i32 %72 to i64, !dbg !12
  %74 = inttoptr i64 %73 to ptr addrspace(3), !dbg !12
  %address.local174 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local175 = and i64 %address.local174, 16256, !dbg !12
  %address.local176 = lshr i64 %address.local168, 4, !dbg !12
  %address.local177 = and i64 %address.local176, 40, !dbg !12
  %address.local178 = or disjoint i64 %address.local175, %address.local177, !dbg !12
  %address.local179 = getelementptr [2 x i8], ptr addrspace(1) %43, i64 %address.local178, !dbg !12
  %75 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %76 = ptrtoint ptr addrspace(1) %address.local179 to i64, !dbg !12
  %77 = sub i64 %76, %75, !dbg !12
  %dma.buffer.offset853 = trunc i64 %77 to i32, !dbg !12
  %78 = trunc i64 %75 to i32, !dbg !12
  %79 = insertelement <4 x i32> poison, i32 %78, i64 0, !dbg !12
  %80 = lshr i64 %75, 32, !dbg !12
  %81 = trunc i64 %80 to i32, !dbg !12
  %82 = insertelement <4 x i32> %79, i32 %81, i64 1, !dbg !12
  %83 = insertelement <4 x i32> %82, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc854 = insertelement <4 x i32> %83, i32 131072, i64 3, !dbg !12
  %84 = ptrtoint ptr addrspace(3) %74 to i32, !dbg !12
  %dma.buffer.lds855 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %84), !dbg !12
  %85 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds855, i32 %dma.buffer.offset853, <4 x i32> %dma.buffer.rsrc854) #10, !dbg !12
  %address.local180 = or disjoint i64 %address.local, 768, !dbg !12
  %address.local181 = shl nuw nsw i64 %address.local180, 3, !dbg !12
  %address.local182 = and i64 %address.local181, 7680, !dbg !12
  %address.local183 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local182, !dbg !12
  %address.local184 = ptrtoint ptr addrspace(3) %address.local183 to i64, !dbg !12
  %address.local185 = trunc i64 %address.local184 to i32, !dbg !12
  %86 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local185), !dbg !12
  %87 = zext i32 %86 to i64, !dbg !12
  %88 = inttoptr i64 %87 to ptr addrspace(3), !dbg !12
  %address.local186 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local187 = and i64 %address.local186, 16256, !dbg !12
  %address.local188 = lshr i64 %address.local180, 4, !dbg !12
  %address.local189 = and i64 %address.local188, 56, !dbg !12
  %address.local190 = or disjoint i64 %address.local187, %address.local189, !dbg !12
  %address.local191 = getelementptr [2 x i8], ptr addrspace(1) %43, i64 %address.local190, !dbg !12
  %89 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %90 = ptrtoint ptr addrspace(1) %address.local191 to i64, !dbg !12
  %91 = sub i64 %90, %89, !dbg !12
  %dma.buffer.offset850 = trunc i64 %91 to i32, !dbg !12
  %92 = trunc i64 %89 to i32, !dbg !12
  %93 = insertelement <4 x i32> poison, i32 %92, i64 0, !dbg !12
  %94 = lshr i64 %89, 32, !dbg !12
  %95 = trunc i64 %94 to i32, !dbg !12
  %96 = insertelement <4 x i32> %93, i32 %95, i64 1, !dbg !12
  %97 = insertelement <4 x i32> %96, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc851 = insertelement <4 x i32> %97, i32 131072, i64 3, !dbg !12
  %98 = ptrtoint ptr addrspace(3) %88 to i32, !dbg !12
  %dma.buffer.lds852 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %98), !dbg !12
  %99 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds852, i32 %dma.buffer.offset850, <4 x i32> %dma.buffer.rsrc851) #10, !dbg !12
  %100 = shl nuw nsw i64 %36, 11, !dbg !16
  %101 = add nuw i64 %26, %100, !dbg !16
  %.lobit69 = ashr i64 %101, 63, !dbg !16
  %102 = xor i64 %101, %.lobit69, !dbg !16
  %103 = sdiv i64 %102, 4096, !dbg !16
  %.lobit70 = ashr i64 %101, 63, !dbg !16
  %104 = xor i64 %103, %.lobit70, !dbg !16
  %105 = add nsw i64 %104, %8, !dbg !16
  %106 = add nuw nsw i64 %40, %32, !dbg !16
  %107 = and i64 %106, 127, !dbg !16
  %108 = shl i64 %105, 19, !dbg !16
  %109 = and i64 %108, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %110 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %109, !dbg !16
  %111 = getelementptr [2 x i8], ptr addrspace(1) %110, i64 %107, !dbg !16
  %.idx71 = shl i64 %34, 8, !dbg !16
  %112 = getelementptr i8, ptr addrspace(1) %111, i64 %.idx71, !dbg !16
  %address.local192 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local146, !dbg !16
  %address.local193 = ptrtoint ptr addrspace(3) %address.local192 to i64, !dbg !16
  %address.local194 = trunc i64 %address.local193 to i32, !dbg !16
  %113 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local194), !dbg !16
  %114 = zext i32 %113 to i64, !dbg !16
  %115 = inttoptr i64 %114 to ptr addrspace(3), !dbg !16
  %address.local195 = getelementptr [2 x i8], ptr addrspace(1) %112, i64 %address.local154, !dbg !16
  %116 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %117 = ptrtoint ptr addrspace(1) %address.local195 to i64, !dbg !16
  %118 = sub i64 %117, %116, !dbg !16
  %dma.buffer.offset847 = trunc i64 %118 to i32, !dbg !16
  %119 = trunc i64 %116 to i32, !dbg !16
  %120 = insertelement <4 x i32> poison, i32 %119, i64 0, !dbg !16
  %121 = lshr i64 %116, 32, !dbg !16
  %122 = trunc i64 %121 to i32, !dbg !16
  %123 = insertelement <4 x i32> %120, i32 %122, i64 1, !dbg !16
  %124 = insertelement <4 x i32> %123, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc848 = insertelement <4 x i32> %124, i32 131072, i64 3, !dbg !16
  %125 = ptrtoint ptr addrspace(3) %115 to i32, !dbg !16
  %dma.buffer.lds849 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %125), !dbg !16
  %126 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds849, i32 %dma.buffer.offset847, <4 x i32> %dma.buffer.rsrc848) #10, !dbg !16
  %address.local196 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local158, !dbg !16
  %address.local197 = ptrtoint ptr addrspace(3) %address.local196 to i64, !dbg !16
  %address.local198 = trunc i64 %address.local197 to i32, !dbg !16
  %127 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local198), !dbg !16
  %128 = zext i32 %127 to i64, !dbg !16
  %129 = inttoptr i64 %128 to ptr addrspace(3), !dbg !16
  %address.local199 = getelementptr [2 x i8], ptr addrspace(1) %112, i64 %address.local166, !dbg !16
  %130 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %131 = ptrtoint ptr addrspace(1) %address.local199 to i64, !dbg !16
  %132 = sub i64 %131, %130, !dbg !16
  %dma.buffer.offset844 = trunc i64 %132 to i32, !dbg !16
  %133 = trunc i64 %130 to i32, !dbg !16
  %134 = insertelement <4 x i32> poison, i32 %133, i64 0, !dbg !16
  %135 = lshr i64 %130, 32, !dbg !16
  %136 = trunc i64 %135 to i32, !dbg !16
  %137 = insertelement <4 x i32> %134, i32 %136, i64 1, !dbg !16
  %138 = insertelement <4 x i32> %137, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc845 = insertelement <4 x i32> %138, i32 131072, i64 3, !dbg !16
  %139 = ptrtoint ptr addrspace(3) %129 to i32, !dbg !16
  %dma.buffer.lds846 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %139), !dbg !16
  %140 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds846, i32 %dma.buffer.offset844, <4 x i32> %dma.buffer.rsrc845) #10, !dbg !16
  %address.local200 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local170, !dbg !16
  %address.local201 = ptrtoint ptr addrspace(3) %address.local200 to i64, !dbg !16
  %address.local202 = trunc i64 %address.local201 to i32, !dbg !16
  %141 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local202), !dbg !16
  %142 = zext i32 %141 to i64, !dbg !16
  %143 = inttoptr i64 %142 to ptr addrspace(3), !dbg !16
  %address.local203 = getelementptr [2 x i8], ptr addrspace(1) %112, i64 %address.local178, !dbg !16
  %144 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %145 = ptrtoint ptr addrspace(1) %address.local203 to i64, !dbg !16
  %146 = sub i64 %145, %144, !dbg !16
  %dma.buffer.offset841 = trunc i64 %146 to i32, !dbg !16
  %147 = trunc i64 %144 to i32, !dbg !16
  %148 = insertelement <4 x i32> poison, i32 %147, i64 0, !dbg !16
  %149 = lshr i64 %144, 32, !dbg !16
  %150 = trunc i64 %149 to i32, !dbg !16
  %151 = insertelement <4 x i32> %148, i32 %150, i64 1, !dbg !16
  %152 = insertelement <4 x i32> %151, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc842 = insertelement <4 x i32> %152, i32 131072, i64 3, !dbg !16
  %153 = ptrtoint ptr addrspace(3) %143 to i32, !dbg !16
  %dma.buffer.lds843 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %153), !dbg !16
  %154 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds843, i32 %dma.buffer.offset841, <4 x i32> %dma.buffer.rsrc842) #10, !dbg !16
  %address.local204 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local182, !dbg !16
  %address.local205 = ptrtoint ptr addrspace(3) %address.local204 to i64, !dbg !16
  %address.local206 = trunc i64 %address.local205 to i32, !dbg !16
  %155 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local206), !dbg !16
  %156 = zext i32 %155 to i64, !dbg !16
  %157 = inttoptr i64 %156 to ptr addrspace(3), !dbg !16
  %address.local207 = getelementptr [2 x i8], ptr addrspace(1) %112, i64 %address.local190, !dbg !16
  %158 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %159 = ptrtoint ptr addrspace(1) %address.local207 to i64, !dbg !16
  %160 = sub i64 %159, %158, !dbg !16
  %dma.buffer.offset = trunc i64 %160 to i32, !dbg !16
  %161 = trunc i64 %158 to i32, !dbg !16
  %162 = insertelement <4 x i32> poison, i32 %161, i64 0, !dbg !16
  %163 = lshr i64 %158, 32, !dbg !16
  %164 = trunc i64 %163 to i32, !dbg !16
  %165 = insertelement <4 x i32> %162, i32 %164, i64 1, !dbg !16
  %166 = insertelement <4 x i32> %165, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc = insertelement <4 x i32> %166, i32 131072, i64 3, !dbg !16
  %167 = ptrtoint ptr addrspace(3) %157 to i32, !dbg !16
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %167), !dbg !16
  %168 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !16
  %address.local208 = lshr i32 %address.thread, 2, !dbg !13
  %address.local209 = and i32 %address.local208, 4, !dbg !13
  %address.local210 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local211 = and i32 %address.local210, 384, !dbg !13
  %address.local212 = or disjoint i32 %address.local209, %address.local211, !dbg !13
  %address.local213 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local214 = and i32 %address.local213, 120, !dbg !13
  %address.local215 = or disjoint i32 %address.local212, %address.local214, !dbg !13
  %address.local216 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local217 = and i32 %address.local216, 1024, !dbg !13
  %address.local218 = or disjoint i32 %address.local215, %address.local217, !dbg !13
  %address.local219 = zext nneg i32 %address.local218 to i64, !dbg !13
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local219, !dbg !13
  %169 = load <4 x half>, ptr addrspace(3) %address.local220, align 2, !dbg !13
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local219, !dbg !13
  %address.local222 = getelementptr i8, ptr addrspace(3) %address.local221, i64 4096, !dbg !13
  %170 = load <4 x half>, ptr addrspace(3) %address.local222, align 2, !dbg !13
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local219, !dbg !13
  %address.local224 = getelementptr i8, ptr addrspace(3) %address.local223, i64 8192, !dbg !13
  %171 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !13
  %172 = fmul <4 x half> %169, splat (half 0xH3015), !dbg !13
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local219, !dbg !13
  %address.local226 = getelementptr i8, ptr addrspace(3) %address.local225, i64 12288, !dbg !13
  %173 = load <4 x half>, ptr addrspace(3) %address.local226, align 2, !dbg !13
  %174 = fmul <4 x half> %170, splat (half 0xH3015), !dbg !13
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local219, !dbg !13
  %address.local228 = getelementptr i8, ptr addrspace(3) %address.local227, i64 1024, !dbg !13
  %175 = load <4 x half>, ptr addrspace(3) %address.local228, align 2, !dbg !13
  %176 = fmul <4 x half> %171, splat (half 0xH3015), !dbg !13
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local219, !dbg !13
  %address.local230 = getelementptr i8, ptr addrspace(3) %address.local229, i64 5120, !dbg !13
  %177 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !13
  %178 = fmul <4 x half> %173, splat (half 0xH3015), !dbg !13
  %address.local231 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local219, !dbg !13
  %address.local232 = getelementptr i8, ptr addrspace(3) %address.local231, i64 9216, !dbg !13
  %179 = load <4 x half>, ptr addrspace(3) %address.local232, align 2, !dbg !13
  %180 = fmul <4 x half> %175, splat (half 0xH3015), !dbg !13
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local219, !dbg !13
  %address.local234 = getelementptr i8, ptr addrspace(3) %address.local233, i64 13312, !dbg !13
  %181 = load <4 x half>, ptr addrspace(3) %address.local234, align 2, !dbg !13
  %182 = fmul <4 x half> %177, splat (half 0xH3015), !dbg !13
  %183 = fmul <4 x half> %179, splat (half 0xH3015), !dbg !13
  %184 = fmul <4 x half> %181, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local235 = lshr i32 %address.thread, 2, !dbg !8
  %address.local236 = and i32 %address.local235, 4, !dbg !8
  %address.local237 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local238 = and i32 %address.local237, 24, !dbg !8
  %address.local239 = or disjoint i32 %address.local236, %address.local238, !dbg !8
  %address.local240 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local241 = and i32 %address.local240, 1024, !dbg !8
  %address.local242 = or disjoint i32 %address.local239, %address.local241, !dbg !8
  %address.local243 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local244 = and i32 %address.local243, 384, !dbg !8
  %address.local245 = or disjoint i32 %address.local242, %address.local244, !dbg !8
  %address.local246 = zext nneg i32 %address.local245 to i64, !dbg !8
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local246, !dbg !8
  %185 = load <4 x half>, ptr addrspace(3) %address.local247, align 2, !dbg !8
  %186 = extractvalue [2 x <32 x float>] %37, 0, !dbg !31
  %187 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %188 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %185, <4 x float> %187) #11, !dbg !8
  %address.local248 = lshr i32 %address.thread, 2, !dbg !8
  %address.local249 = and i32 %address.local248, 4, !dbg !8
  %address.local250 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local251 = and i32 %address.local250, 24, !dbg !8
  %address.local252 = or disjoint i32 %address.local249, %address.local251, !dbg !8
  %address.local253 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local254 = and i32 %address.local253, 1024, !dbg !8
  %address.local255 = or disjoint i32 %address.local252, %address.local254, !dbg !8
  %address.local256 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local257 = and i32 %address.local256, 384, !dbg !8
  %address.local258 = or disjoint i32 %address.local255, %address.local257, !dbg !8
  %address.local259 = zext nneg i32 %address.local258 to i64, !dbg !8
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local259, !dbg !8
  %address.local261 = getelementptr i8, ptr addrspace(3) %address.local260, i64 4096, !dbg !8
  %189 = load <4 x half>, ptr addrspace(3) %address.local261, align 2, !dbg !8
  %190 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %191 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %189, <4 x float> %188) #11, !dbg !8
  %192 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %193 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %194 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %195 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %196 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %197 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %198 = extractvalue [2 x <32 x float>] %37, 1, !dbg !31
  %199 = shufflevector <32 x float> %198, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %200 = shufflevector <32 x float> %198, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %201 = shufflevector <32 x float> %198, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %202 = shufflevector <32 x float> %198, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %203 = shufflevector <32 x float> %198, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %204 = shufflevector <32 x float> %198, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %205 = shufflevector <32 x float> %198, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %206 = shufflevector <32 x float> %198, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local259, !dbg !8
  %address.local263 = getelementptr i8, ptr addrspace(3) %address.local262, i64 8192, !dbg !8
  %207 = load <4 x half>, ptr addrspace(3) %address.local263, align 2, !dbg !8
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local259, !dbg !8
  %address.local265 = getelementptr i8, ptr addrspace(3) %address.local264, i64 12288, !dbg !8
  %208 = load <4 x half>, ptr addrspace(3) %address.local265, align 2, !dbg !8
  %209 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %207, <4 x float> %191) #11, !dbg !8
  %address.local266 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local266, !dbg !8
  %address.local268 = lshr i64 %address.local, 2, !dbg !8
  %address.local269 = mul nsw i64 %address.local268, -512, !dbg !8
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) %address.local267, i64 %address.local269, !dbg !8
  %address.local271 = shl nuw nsw i64 %address.local, 1, !dbg !8
  %address.local272 = and i64 %address.local271, 24, !dbg !8
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local272, !dbg !8
  %address.local274 = lshr i64 %address.local, 2, !dbg !13
  %address.local275 = and i64 %address.local274, 12, !dbg !13
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) %address.local273, i64 %address.local275, !dbg !8
  %address.local277 = and i64 %address.local, 32, !dbg !13
  %address.local278 = icmp eq i64 %address.local277, 0, !dbg !13
  %address.local279 = select i1 %address.local278, i64 0, i64 1016, !dbg !13
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) %address.local276, i64 %address.local279, !dbg !8
  %address.local281 = getelementptr i8, ptr addrspace(3) %address.local280, i64 64, !dbg !8
  %210 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !8
  %211 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %208, <4 x float> %209) #11, !dbg !8
  %address.local282 = lshr i32 %address.thread, 2, !dbg !8
  %address.local283 = and i32 %address.local282, 4, !dbg !8
  %address.local284 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local285 = and i32 %address.local284, 24, !dbg !8
  %address.local286 = or disjoint i32 %address.local283, %address.local285, !dbg !8
  %address.local287 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local288 = and i32 %address.local287, 1024, !dbg !8
  %address.local289 = or disjoint i32 %address.local286, %address.local288, !dbg !8
  %address.local290 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local291 = and i32 %address.local290, 384, !dbg !8
  %address.local292 = or disjoint i32 %address.local289, %address.local291, !dbg !8
  %address.local293 = zext nneg i32 %address.local292 to i64, !dbg !8
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local293, !dbg !8
  %address.local295 = getelementptr i8, ptr addrspace(3) %address.local294, i64 4160, !dbg !8
  %212 = load <4 x half>, ptr addrspace(3) %address.local295, align 2, !dbg !8
  %213 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %210, <4 x float> %190) #11, !dbg !8
  %214 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %212, <4 x float> %213) #11, !dbg !8
  %address.local296 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local293, !dbg !8
  %address.local297 = getelementptr i8, ptr addrspace(3) %address.local296, i64 8256, !dbg !8
  %215 = load <4 x half>, ptr addrspace(3) %address.local297, align 2, !dbg !8
  %address.local298 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local293, !dbg !8
  %address.local299 = getelementptr i8, ptr addrspace(3) %address.local298, i64 12352, !dbg !8
  %216 = load <4 x half>, ptr addrspace(3) %address.local299, align 2, !dbg !8
  %217 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %215, <4 x float> %214) #11, !dbg !8
  %address.local300 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local266, !dbg !8
  %address.local301 = getelementptr [2 x i8], ptr addrspace(3) %address.local300, i64 %address.local269, !dbg !8
  %address.local302 = getelementptr [2 x i8], ptr addrspace(3) %address.local301, i64 %address.local272, !dbg !8
  %address.local303 = getelementptr [2 x i8], ptr addrspace(3) %address.local302, i64 %address.local275, !dbg !8
  %address.local304 = getelementptr [2 x i8], ptr addrspace(3) %address.local303, i64 %address.local279, !dbg !8
  %address.local305 = getelementptr i8, ptr addrspace(3) %address.local304, i64 128, !dbg !8
  %218 = load <4 x half>, ptr addrspace(3) %address.local305, align 2, !dbg !8
  %219 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %216, <4 x float> %217) #11, !dbg !8
  %address.local306 = lshr i32 %address.thread, 2, !dbg !8
  %address.local307 = and i32 %address.local306, 4, !dbg !8
  %address.local308 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local309 = and i32 %address.local308, 24, !dbg !8
  %address.local310 = or disjoint i32 %address.local307, %address.local309, !dbg !8
  %address.local311 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local312 = and i32 %address.local311, 1024, !dbg !8
  %address.local313 = or disjoint i32 %address.local310, %address.local312, !dbg !8
  %address.local314 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local315 = and i32 %address.local314, 384, !dbg !8
  %address.local316 = or disjoint i32 %address.local313, %address.local315, !dbg !8
  %address.local317 = zext nneg i32 %address.local316 to i64, !dbg !8
  %address.local318 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local317, !dbg !8
  %address.local319 = getelementptr i8, ptr addrspace(3) %address.local318, i64 4224, !dbg !8
  %220 = load <4 x half>, ptr addrspace(3) %address.local319, align 2, !dbg !8
  %221 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %218, <4 x float> %192) #11, !dbg !8
  %222 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %220, <4 x float> %221) #11, !dbg !8
  %address.local320 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local317, !dbg !8
  %address.local321 = getelementptr i8, ptr addrspace(3) %address.local320, i64 8320, !dbg !8
  %223 = load <4 x half>, ptr addrspace(3) %address.local321, align 2, !dbg !8
  %address.local322 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local317, !dbg !8
  %address.local323 = getelementptr i8, ptr addrspace(3) %address.local322, i64 12416, !dbg !8
  %224 = load <4 x half>, ptr addrspace(3) %address.local323, align 2, !dbg !8
  %225 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %223, <4 x float> %222) #11, !dbg !8
  %address.local324 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local266, !dbg !8
  %address.local325 = getelementptr [2 x i8], ptr addrspace(3) %address.local324, i64 %address.local269, !dbg !8
  %address.local326 = getelementptr [2 x i8], ptr addrspace(3) %address.local325, i64 %address.local272, !dbg !8
  %address.local327 = getelementptr [2 x i8], ptr addrspace(3) %address.local326, i64 %address.local275, !dbg !8
  %address.local328 = getelementptr [2 x i8], ptr addrspace(3) %address.local327, i64 %address.local279, !dbg !8
  %address.local329 = getelementptr i8, ptr addrspace(3) %address.local328, i64 192, !dbg !8
  %226 = load <4 x half>, ptr addrspace(3) %address.local329, align 2, !dbg !8
  %227 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %224, <4 x float> %225) #11, !dbg !8
  %address.local330 = lshr i32 %address.thread, 2, !dbg !8
  %address.local331 = and i32 %address.local330, 4, !dbg !8
  %address.local332 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local333 = and i32 %address.local332, 24, !dbg !8
  %address.local334 = or disjoint i32 %address.local331, %address.local333, !dbg !8
  %address.local335 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local336 = and i32 %address.local335, 1024, !dbg !8
  %address.local337 = or disjoint i32 %address.local334, %address.local336, !dbg !8
  %address.local338 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local339 = and i32 %address.local338, 384, !dbg !8
  %address.local340 = or disjoint i32 %address.local337, %address.local339, !dbg !8
  %address.local341 = zext nneg i32 %address.local340 to i64, !dbg !8
  %address.local342 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local341, !dbg !8
  %address.local343 = getelementptr i8, ptr addrspace(3) %address.local342, i64 4288, !dbg !8
  %228 = load <4 x half>, ptr addrspace(3) %address.local343, align 2, !dbg !8
  %229 = shufflevector <4 x float> %227, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %230 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %226, <4 x float> %193) #11, !dbg !8
  %231 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %228, <4 x float> %230) #11, !dbg !8
  %address.local344 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local341, !dbg !8
  %address.local345 = getelementptr i8, ptr addrspace(3) %address.local344, i64 8384, !dbg !8
  %232 = load <4 x half>, ptr addrspace(3) %address.local345, align 2, !dbg !8
  %address.local346 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local341, !dbg !8
  %address.local347 = getelementptr i8, ptr addrspace(3) %address.local346, i64 12480, !dbg !8
  %233 = load <4 x half>, ptr addrspace(3) %address.local347, align 2, !dbg !8
  %234 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %232, <4 x float> %231) #11, !dbg !8
  %address.local348 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local246, !dbg !8
  %address.local349 = getelementptr i8, ptr addrspace(3) %address.local348, i64 1024, !dbg !8
  %235 = load <4 x half>, ptr addrspace(3) %address.local349, align 2, !dbg !8
  %236 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %233, <4 x float> %234) #11, !dbg !8
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local259, !dbg !8
  %address.local351 = getelementptr i8, ptr addrspace(3) %address.local350, i64 5120, !dbg !8
  %237 = load <4 x half>, ptr addrspace(3) %address.local351, align 2, !dbg !8
  %238 = shufflevector <4 x float> %236, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %239 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %235, <4 x float> %194) #11, !dbg !8
  %240 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %237, <4 x float> %239) #11, !dbg !8
  %address.local352 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local259, !dbg !8
  %address.local353 = getelementptr i8, ptr addrspace(3) %address.local352, i64 9216, !dbg !8
  %241 = load <4 x half>, ptr addrspace(3) %address.local353, align 2, !dbg !8
  %address.local354 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local259, !dbg !8
  %address.local355 = getelementptr i8, ptr addrspace(3) %address.local354, i64 13312, !dbg !8
  %242 = load <4 x half>, ptr addrspace(3) %address.local355, align 2, !dbg !8
  %243 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %241, <4 x float> %240) #11, !dbg !8
  %address.local356 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local266, !dbg !8
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) %address.local356, i64 %address.local269, !dbg !8
  %address.local358 = getelementptr [2 x i8], ptr addrspace(3) %address.local357, i64 %address.local272, !dbg !8
  %address.local359 = getelementptr [2 x i8], ptr addrspace(3) %address.local358, i64 %address.local275, !dbg !8
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) %address.local359, i64 %address.local279, !dbg !8
  %address.local361 = getelementptr i8, ptr addrspace(3) %address.local360, i64 1088, !dbg !8
  %244 = load <4 x half>, ptr addrspace(3) %address.local361, align 2, !dbg !8
  %245 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %242, <4 x float> %243) #11, !dbg !8
  %address.local362 = lshr i32 %address.thread, 2, !dbg !8
  %address.local363 = and i32 %address.local362, 4, !dbg !8
  %address.local364 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local365 = and i32 %address.local364, 24, !dbg !8
  %address.local366 = or disjoint i32 %address.local363, %address.local365, !dbg !8
  %address.local367 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local368 = and i32 %address.local367, 1024, !dbg !8
  %address.local369 = or disjoint i32 %address.local366, %address.local368, !dbg !8
  %address.local370 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local371 = and i32 %address.local370, 384, !dbg !8
  %address.local372 = or disjoint i32 %address.local369, %address.local371, !dbg !8
  %address.local373 = or disjoint i32 %address.local372, 512, !dbg !8
  %address.local374 = zext nneg i32 %address.local373 to i64, !dbg !8
  %address.local375 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local374, !dbg !8
  %address.local376 = getelementptr i8, ptr addrspace(3) %address.local375, i64 4160, !dbg !8
  %246 = load <4 x half>, ptr addrspace(3) %address.local376, align 2, !dbg !8
  %247 = shufflevector <4 x float> %245, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %248 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %244, <4 x float> %195) #11, !dbg !8
  %249 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %246, <4 x float> %248) #11, !dbg !8
  %address.local377 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local374, !dbg !8
  %address.local378 = getelementptr i8, ptr addrspace(3) %address.local377, i64 8256, !dbg !8
  %250 = load <4 x half>, ptr addrspace(3) %address.local378, align 2, !dbg !8
  %address.local379 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local374, !dbg !8
  %address.local380 = getelementptr i8, ptr addrspace(3) %address.local379, i64 12352, !dbg !8
  %251 = load <4 x half>, ptr addrspace(3) %address.local380, align 2, !dbg !8
  %252 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %250, <4 x float> %249) #11, !dbg !8
  %address.local381 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local266, !dbg !8
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) %address.local381, i64 %address.local269, !dbg !8
  %address.local383 = getelementptr [2 x i8], ptr addrspace(3) %address.local382, i64 %address.local272, !dbg !8
  %address.local384 = getelementptr [2 x i8], ptr addrspace(3) %address.local383, i64 %address.local275, !dbg !8
  %address.local385 = getelementptr [2 x i8], ptr addrspace(3) %address.local384, i64 %address.local279, !dbg !8
  %address.local386 = getelementptr i8, ptr addrspace(3) %address.local385, i64 1152, !dbg !8
  %253 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !8
  %254 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %251, <4 x float> %252) #11, !dbg !8
  %address.local387 = lshr i32 %address.thread, 2, !dbg !8
  %address.local388 = and i32 %address.local387, 4, !dbg !8
  %address.local389 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local390 = and i32 %address.local389, 24, !dbg !8
  %address.local391 = or disjoint i32 %address.local388, %address.local390, !dbg !8
  %address.local392 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local393 = and i32 %address.local392, 1024, !dbg !8
  %address.local394 = or disjoint i32 %address.local391, %address.local393, !dbg !8
  %address.local395 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local396 = and i32 %address.local395, 384, !dbg !8
  %address.local397 = or disjoint i32 %address.local394, %address.local396, !dbg !8
  %address.local398 = or disjoint i32 %address.local397, 512, !dbg !8
  %address.local399 = zext nneg i32 %address.local398 to i64, !dbg !8
  %address.local400 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local399, !dbg !8
  %address.local401 = getelementptr i8, ptr addrspace(3) %address.local400, i64 4224, !dbg !8
  %255 = load <4 x half>, ptr addrspace(3) %address.local401, align 2, !dbg !8
  %256 = shufflevector <4 x float> %254, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %257 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %253, <4 x float> %196) #11, !dbg !8
  %258 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %255, <4 x float> %257) #11, !dbg !8
  %address.local402 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local399, !dbg !8
  %address.local403 = getelementptr i8, ptr addrspace(3) %address.local402, i64 8320, !dbg !8
  %259 = load <4 x half>, ptr addrspace(3) %address.local403, align 2, !dbg !8
  %address.local404 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local399, !dbg !8
  %address.local405 = getelementptr i8, ptr addrspace(3) %address.local404, i64 12416, !dbg !8
  %260 = load <4 x half>, ptr addrspace(3) %address.local405, align 2, !dbg !8
  %261 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %259, <4 x float> %258) #11, !dbg !8
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local266, !dbg !8
  %address.local407 = getelementptr [2 x i8], ptr addrspace(3) %address.local406, i64 %address.local269, !dbg !8
  %address.local408 = getelementptr [2 x i8], ptr addrspace(3) %address.local407, i64 %address.local272, !dbg !8
  %address.local409 = getelementptr [2 x i8], ptr addrspace(3) %address.local408, i64 %address.local275, !dbg !8
  %address.local410 = getelementptr [2 x i8], ptr addrspace(3) %address.local409, i64 %address.local279, !dbg !8
  %address.local411 = getelementptr i8, ptr addrspace(3) %address.local410, i64 1216, !dbg !8
  %262 = load <4 x half>, ptr addrspace(3) %address.local411, align 2, !dbg !8
  %263 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %260, <4 x float> %261) #11, !dbg !8
  %address.local412 = lshr i32 %address.thread, 2, !dbg !8
  %address.local413 = and i32 %address.local412, 4, !dbg !8
  %address.local414 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local415 = and i32 %address.local414, 24, !dbg !8
  %address.local416 = or disjoint i32 %address.local413, %address.local415, !dbg !8
  %address.local417 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local418 = and i32 %address.local417, 1024, !dbg !8
  %address.local419 = or disjoint i32 %address.local416, %address.local418, !dbg !8
  %address.local420 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local421 = and i32 %address.local420, 384, !dbg !8
  %address.local422 = or disjoint i32 %address.local419, %address.local421, !dbg !8
  %address.local423 = or disjoint i32 %address.local422, 512, !dbg !8
  %address.local424 = zext nneg i32 %address.local423 to i64, !dbg !8
  %address.local425 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local424, !dbg !8
  %address.local426 = getelementptr i8, ptr addrspace(3) %address.local425, i64 4288, !dbg !8
  %264 = load <4 x half>, ptr addrspace(3) %address.local426, align 2, !dbg !8
  %265 = shufflevector <4 x float> %263, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %266 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %262, <4 x float> %197) #11, !dbg !8
  %267 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %264, <4 x float> %266) #11, !dbg !8
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local424, !dbg !8
  %address.local428 = getelementptr i8, ptr addrspace(3) %address.local427, i64 8384, !dbg !8
  %268 = load <4 x half>, ptr addrspace(3) %address.local428, align 2, !dbg !8
  %address.local429 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local424, !dbg !8
  %address.local430 = getelementptr i8, ptr addrspace(3) %address.local429, i64 12480, !dbg !8
  %269 = load <4 x half>, ptr addrspace(3) %address.local430, align 2, !dbg !8
  %270 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %268, <4 x float> %267) #11, !dbg !8
  %271 = load <4 x half>, ptr addrspace(3) %address.local247, align 2, !dbg !8
  %272 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %269, <4 x float> %270) #11, !dbg !8
  %273 = load <4 x half>, ptr addrspace(3) %address.local261, align 2, !dbg !8
  %274 = shufflevector <4 x float> %272, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %275 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %271, <4 x float> %199) #11, !dbg !8
  %276 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %273, <4 x float> %275) #11, !dbg !8
  %277 = load <4 x half>, ptr addrspace(3) %address.local263, align 2, !dbg !8
  %278 = load <4 x half>, ptr addrspace(3) %address.local265, align 2, !dbg !8
  %279 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %277, <4 x float> %276) #11, !dbg !8
  %280 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !8
  %281 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %278, <4 x float> %279) #11, !dbg !8
  %282 = load <4 x half>, ptr addrspace(3) %address.local295, align 2, !dbg !8
  %283 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %280, <4 x float> %200) #11, !dbg !8
  %284 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %282, <4 x float> %283) #11, !dbg !8
  %285 = load <4 x half>, ptr addrspace(3) %address.local297, align 2, !dbg !8
  %286 = load <4 x half>, ptr addrspace(3) %address.local299, align 2, !dbg !8
  %287 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %285, <4 x float> %284) #11, !dbg !8
  %288 = load <4 x half>, ptr addrspace(3) %address.local305, align 2, !dbg !8
  %289 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %286, <4 x float> %287) #11, !dbg !8
  %290 = load <4 x half>, ptr addrspace(3) %address.local319, align 2, !dbg !8
  %291 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %288, <4 x float> %201) #11, !dbg !8
  %292 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %290, <4 x float> %291) #11, !dbg !8
  %293 = load <4 x half>, ptr addrspace(3) %address.local321, align 2, !dbg !8
  %294 = load <4 x half>, ptr addrspace(3) %address.local323, align 2, !dbg !8
  %295 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %293, <4 x float> %292) #11, !dbg !8
  %296 = load <4 x half>, ptr addrspace(3) %address.local329, align 2, !dbg !8
  %297 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %294, <4 x float> %295) #11, !dbg !8
  %298 = load <4 x half>, ptr addrspace(3) %address.local343, align 2, !dbg !8
  %299 = shufflevector <4 x float> %297, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %300 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %296, <4 x float> %202) #11, !dbg !8
  %301 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %298, <4 x float> %300) #11, !dbg !8
  %302 = load <4 x half>, ptr addrspace(3) %address.local345, align 2, !dbg !8
  %303 = load <4 x half>, ptr addrspace(3) %address.local347, align 2, !dbg !8
  %304 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %302, <4 x float> %301) #11, !dbg !8
  %305 = load <4 x half>, ptr addrspace(3) %address.local349, align 2, !dbg !8
  %306 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %303, <4 x float> %304) #11, !dbg !8
  %307 = load <4 x half>, ptr addrspace(3) %address.local351, align 2, !dbg !8
  %308 = shufflevector <4 x float> %306, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %309 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %305, <4 x float> %203) #11, !dbg !8
  %310 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %307, <4 x float> %309) #11, !dbg !8
  %311 = load <4 x half>, ptr addrspace(3) %address.local353, align 2, !dbg !8
  %312 = load <4 x half>, ptr addrspace(3) %address.local355, align 2, !dbg !8
  %313 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %311, <4 x float> %310) #11, !dbg !8
  %314 = load <4 x half>, ptr addrspace(3) %address.local361, align 2, !dbg !8
  %315 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %312, <4 x float> %313) #11, !dbg !8
  %316 = load <4 x half>, ptr addrspace(3) %address.local376, align 2, !dbg !8
  %317 = shufflevector <4 x float> %315, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %318 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %314, <4 x float> %204) #11, !dbg !8
  %319 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %316, <4 x float> %318) #11, !dbg !8
  %320 = load <4 x half>, ptr addrspace(3) %address.local378, align 2, !dbg !8
  %321 = load <4 x half>, ptr addrspace(3) %address.local380, align 2, !dbg !8
  %322 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %320, <4 x float> %319) #11, !dbg !8
  %323 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !8
  %324 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %321, <4 x float> %322) #11, !dbg !8
  %325 = load <4 x half>, ptr addrspace(3) %address.local401, align 2, !dbg !8
  %326 = shufflevector <4 x float> %324, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %327 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %323, <4 x float> %205) #11, !dbg !8
  %328 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %325, <4 x float> %327) #11, !dbg !8
  %329 = load <4 x half>, ptr addrspace(3) %address.local403, align 2, !dbg !8
  %330 = load <4 x half>, ptr addrspace(3) %address.local405, align 2, !dbg !8
  %331 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %329, <4 x float> %328) #11, !dbg !8
  %332 = load <4 x half>, ptr addrspace(3) %address.local411, align 2, !dbg !8
  %333 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %330, <4 x float> %331) #11, !dbg !8
  %334 = load <4 x half>, ptr addrspace(3) %address.local426, align 2, !dbg !8
  %335 = shufflevector <4 x float> %333, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %336 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %332, <4 x float> %206) #11, !dbg !8
  %337 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %334, <4 x float> %336) #11, !dbg !8
  %338 = load <4 x half>, ptr addrspace(3) %address.local428, align 2, !dbg !8
  %339 = load <4 x half>, ptr addrspace(3) %address.local430, align 2, !dbg !8
  %340 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %338, <4 x float> %337) #11, !dbg !8
  %341 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %339, <4 x float> %340) #11, !dbg !8
  %342 = shufflevector <4 x float> %341, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %343 = shufflevector <4 x float> %211, <4 x float> %219, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %344 = shufflevector <32 x float> %343, <32 x float> %229, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %345 = shufflevector <32 x float> %344, <32 x float> %238, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %346 = shufflevector <32 x float> %345, <32 x float> %247, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %347 = shufflevector <32 x float> %346, <32 x float> %256, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %348 = shufflevector <32 x float> %347, <32 x float> %265, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %349 = shufflevector <32 x float> %348, <32 x float> %274, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %350 = insertvalue [2 x <32 x float>] poison, <32 x float> %349, 0, !dbg !22
  %351 = shufflevector <4 x float> %281, <4 x float> %289, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %352 = shufflevector <32 x float> %351, <32 x float> %299, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %353 = shufflevector <32 x float> %352, <32 x float> %308, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %354 = shufflevector <32 x float> %353, <32 x float> %317, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %355 = shufflevector <32 x float> %354, <32 x float> %326, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %356 = shufflevector <32 x float> %355, <32 x float> %335, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %357 = shufflevector <32 x float> %356, <32 x float> %342, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %358 = insertvalue [2 x <32 x float>] %350, <32 x float> %357, 1, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %359 = add nuw nsw i64 %36, 1, !dbg !22
  br label %35, !dbg !22

360:                                              ; preds = %35
  %361 = add nsw i64 %32, %8, !dbg !17
  %362 = shl i64 %361, 19, !dbg !17
  %363 = and i64 %362, 16252928, !dbg !17
  %364 = shl nsw i64 %34, 7, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %365 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %363, !dbg !17
  %366 = getelementptr [2 x i8], ptr addrspace(1) %365, i64 %364, !dbg !17
  %address.thread431 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !17, !range !34, !frisk.register_lifetimes !1
  %address.local432 = zext nneg i32 %address.thread431 to i64, !dbg !13
  %address.local433 = shl nuw nsw i64 %address.local432, 3, !dbg !12
  %address.local434 = and i64 %address.local433, 1536, !dbg !12
  %address.local435 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local434, !dbg !12
  %address.local436 = ptrtoint ptr addrspace(3) %address.local435 to i64, !dbg !12
  %address.local437 = trunc i64 %address.local436 to i32, !dbg !12
  %367 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local437), !dbg !17
  %368 = zext i32 %367 to i64, !dbg !17
  %369 = inttoptr i64 %368 to ptr addrspace(3), !dbg !17
  %address.local438 = lshr i64 %address.local432, 3, !dbg !17
  %address.local439 = and i64 %address.local438, 30, !dbg !17
  %address.local440 = and i64 %address.local432, 1, !dbg !17
  %address.local441 = or disjoint i64 %address.local439, %address.local440, !dbg !17
  %address.local442 = shl nuw nsw i64 %address.local441, 7, !dbg !17
  %address.local443 = shl nuw nsw i64 %address.local432, 2, !dbg !17
  %address.local444 = and i64 %address.local443, 56, !dbg !17
  %address.local445 = or disjoint i64 %address.local442, %address.local444, !dbg !17
  %address.local446 = getelementptr [2 x i8], ptr addrspace(1) %366, i64 %address.local445, !dbg !17
  %370 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %371 = ptrtoint ptr addrspace(1) %address.local446 to i64, !dbg !17
  %372 = sub i64 %371, %370, !dbg !17
  %dma.buffer.offset883 = trunc i64 %372 to i32, !dbg !17
  %373 = trunc i64 %370 to i32, !dbg !17
  %374 = insertelement <4 x i32> poison, i32 %373, i64 0, !dbg !17
  %375 = lshr i64 %370, 32, !dbg !17
  %376 = trunc i64 %375 to i32, !dbg !17
  %377 = insertelement <4 x i32> %374, i32 %376, i64 1, !dbg !17
  %378 = insertelement <4 x i32> %377, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc884 = insertelement <4 x i32> %378, i32 131072, i64 3, !dbg !17
  %379 = ptrtoint ptr addrspace(3) %369 to i32, !dbg !17
  %dma.buffer.lds885 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %379), !dbg !17
  %380 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds885, i32 %dma.buffer.offset883, <4 x i32> %dma.buffer.rsrc884) #10, !dbg !17
  %address.local447 = or disjoint i64 %address.local432, 256, !dbg !12
  %address.local448 = shl nuw nsw i64 %address.local447, 3, !dbg !12
  %address.local449 = and i64 %address.local448, 3584, !dbg !12
  %address.local450 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local449, !dbg !12
  %address.local451 = ptrtoint ptr addrspace(3) %address.local450 to i64, !dbg !12
  %address.local452 = trunc i64 %address.local451 to i32, !dbg !12
  %381 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local452), !dbg !17
  %382 = zext i32 %381 to i64, !dbg !17
  %383 = inttoptr i64 %382 to ptr addrspace(3), !dbg !17
  %address.local453 = lshr i64 %address.local447, 3, !dbg !17
  %address.local454 = and i64 %address.local453, 62, !dbg !17
  %address.local455 = and i64 %address.local432, 1, !dbg !17
  %address.local456 = or disjoint i64 %address.local454, %address.local455, !dbg !17
  %address.local457 = shl nuw nsw i64 %address.local456, 7, !dbg !17
  %address.local458 = shl nuw nsw i64 %address.local432, 2, !dbg !17
  %address.local459 = and i64 %address.local458, 56, !dbg !17
  %address.local460 = or disjoint i64 %address.local457, %address.local459, !dbg !17
  %address.local461 = getelementptr [2 x i8], ptr addrspace(1) %366, i64 %address.local460, !dbg !17
  %384 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %385 = ptrtoint ptr addrspace(1) %address.local461 to i64, !dbg !17
  %386 = sub i64 %385, %384, !dbg !17
  %dma.buffer.offset880 = trunc i64 %386 to i32, !dbg !17
  %387 = trunc i64 %384 to i32, !dbg !17
  %388 = insertelement <4 x i32> poison, i32 %387, i64 0, !dbg !17
  %389 = lshr i64 %384, 32, !dbg !17
  %390 = trunc i64 %389 to i32, !dbg !17
  %391 = insertelement <4 x i32> %388, i32 %390, i64 1, !dbg !17
  %392 = insertelement <4 x i32> %391, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc881 = insertelement <4 x i32> %392, i32 131072, i64 3, !dbg !17
  %393 = ptrtoint ptr addrspace(3) %383 to i32, !dbg !17
  %dma.buffer.lds882 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %393), !dbg !17
  %394 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds882, i32 %dma.buffer.offset880, <4 x i32> %dma.buffer.rsrc881) #10, !dbg !17
  %address.local462 = or disjoint i64 %address.local432, 512, !dbg !12
  %address.local463 = shl nuw nsw i64 %address.local462, 3, !dbg !12
  %address.local464 = and i64 %address.local463, 5632, !dbg !12
  %address.local465 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local464, !dbg !12
  %address.local466 = ptrtoint ptr addrspace(3) %address.local465 to i64, !dbg !12
  %address.local467 = trunc i64 %address.local466 to i32, !dbg !12
  %395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local467), !dbg !17
  %396 = zext i32 %395 to i64, !dbg !17
  %397 = inttoptr i64 %396 to ptr addrspace(3), !dbg !17
  %address.local468 = lshr i64 %address.local462, 3, !dbg !17
  %address.local469 = and i64 %address.local468, 94, !dbg !17
  %address.local470 = and i64 %address.local432, 1, !dbg !17
  %address.local471 = or disjoint i64 %address.local469, %address.local470, !dbg !17
  %address.local472 = shl nuw nsw i64 %address.local471, 7, !dbg !17
  %address.local473 = shl nuw nsw i64 %address.local432, 2, !dbg !17
  %address.local474 = and i64 %address.local473, 56, !dbg !17
  %address.local475 = or disjoint i64 %address.local472, %address.local474, !dbg !17
  %address.local476 = getelementptr [2 x i8], ptr addrspace(1) %366, i64 %address.local475, !dbg !17
  %398 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %399 = ptrtoint ptr addrspace(1) %address.local476 to i64, !dbg !17
  %400 = sub i64 %399, %398, !dbg !17
  %dma.buffer.offset877 = trunc i64 %400 to i32, !dbg !17
  %401 = trunc i64 %398 to i32, !dbg !17
  %402 = insertelement <4 x i32> poison, i32 %401, i64 0, !dbg !17
  %403 = lshr i64 %398, 32, !dbg !17
  %404 = trunc i64 %403 to i32, !dbg !17
  %405 = insertelement <4 x i32> %402, i32 %404, i64 1, !dbg !17
  %406 = insertelement <4 x i32> %405, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc878 = insertelement <4 x i32> %406, i32 131072, i64 3, !dbg !17
  %407 = ptrtoint ptr addrspace(3) %397 to i32, !dbg !17
  %dma.buffer.lds879 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %407), !dbg !17
  %408 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds879, i32 %dma.buffer.offset877, <4 x i32> %dma.buffer.rsrc878) #10, !dbg !17
  %address.local477 = or disjoint i64 %address.local432, 768, !dbg !12
  %address.local478 = shl nuw nsw i64 %address.local477, 3, !dbg !12
  %address.local479 = and i64 %address.local478, 7680, !dbg !12
  %address.local480 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local479, !dbg !12
  %address.local481 = ptrtoint ptr addrspace(3) %address.local480 to i64, !dbg !12
  %address.local482 = trunc i64 %address.local481 to i32, !dbg !12
  %409 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local482), !dbg !17
  %410 = zext i32 %409 to i64, !dbg !17
  %411 = inttoptr i64 %410 to ptr addrspace(3), !dbg !17
  %address.local483 = lshr i64 %address.local477, 3, !dbg !17
  %address.local484 = and i64 %address.local483, 126, !dbg !17
  %address.local485 = and i64 %address.local432, 1, !dbg !17
  %address.local486 = or disjoint i64 %address.local484, %address.local485, !dbg !17
  %address.local487 = shl nuw nsw i64 %address.local486, 7, !dbg !17
  %address.local488 = shl nuw nsw i64 %address.local432, 2, !dbg !17
  %address.local489 = and i64 %address.local488, 56, !dbg !17
  %address.local490 = or disjoint i64 %address.local487, %address.local489, !dbg !17
  %address.local491 = getelementptr [2 x i8], ptr addrspace(1) %366, i64 %address.local490, !dbg !17
  %412 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %413 = ptrtoint ptr addrspace(1) %address.local491 to i64, !dbg !17
  %414 = sub i64 %413, %412, !dbg !17
  %dma.buffer.offset874 = trunc i64 %414 to i32, !dbg !17
  %415 = trunc i64 %412 to i32, !dbg !17
  %416 = insertelement <4 x i32> poison, i32 %415, i64 0, !dbg !17
  %417 = lshr i64 %412, 32, !dbg !17
  %418 = trunc i64 %417 to i32, !dbg !17
  %419 = insertelement <4 x i32> %416, i32 %418, i64 1, !dbg !17
  %420 = insertelement <4 x i32> %419, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc875 = insertelement <4 x i32> %420, i32 131072, i64 3, !dbg !17
  %421 = ptrtoint ptr addrspace(3) %411 to i32, !dbg !17
  %dma.buffer.lds876 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %421), !dbg !17
  %422 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds876, i32 %dma.buffer.offset874, <4 x i32> %dma.buffer.rsrc875) #10, !dbg !17
  %423 = or disjoint i64 %31, 127, !dbg !14
  %424 = icmp uge i64 %9, %423, !dbg !14
  %425 = and i1 %24, %424, !dbg !14
  br i1 %425, label %426, label %427, !dbg !14

426:                                              ; preds = %360
  br label %598, !dbg !14

427:                                              ; preds = %360
  %428 = add nsw i64 %9, %20, !dbg !14
  %429 = shl nuw nsw i64 %21, 4, !dbg !14
  %430 = add nsw i64 %428, %429, !dbg !14
  %431 = and i32 %19, 240, !dbg !14
  %index.compact = zext nneg i32 %431 to i64, !dbg !14
  %432 = sub nsw i64 %430, %index.compact, !dbg !14
  %433 = and i64 %20, 48, !dbg !14
  %434 = or disjoint i64 %31, %433, !dbg !14
  %435 = add nsw i64 %432, 1, !dbg !23
  %.not4 = icmp ugt i64 %435, %434, !dbg !24
  %436 = select i1 %.not4, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %437 = or disjoint i64 %434, 1, !dbg !14
  %.not5 = icmp ugt i64 %435, %437, !dbg !24
  %438 = select i1 %.not5, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %439 = or disjoint i64 %434, 2, !dbg !14
  %.not6 = icmp ugt i64 %435, %439, !dbg !24
  %440 = select i1 %.not6, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %441 = or disjoint i64 %434, 3, !dbg !14
  %.not7 = icmp ugt i64 %435, %441, !dbg !24
  %442 = select i1 %.not7, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %443 = or disjoint i64 %434, 4, !dbg !14
  %.not8 = icmp ugt i64 %435, %443, !dbg !24
  %444 = select i1 %.not8, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %445 = or disjoint i64 %434, 5, !dbg !14
  %.not9 = icmp ugt i64 %435, %445, !dbg !24
  %446 = select i1 %.not9, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %447 = or disjoint i64 %434, 6, !dbg !14
  %.not10 = icmp ugt i64 %435, %447, !dbg !24
  %448 = select i1 %.not10, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %449 = or disjoint i64 %434, 7, !dbg !14
  %.not11 = icmp ugt i64 %435, %449, !dbg !24
  %450 = select i1 %.not11, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %451 = or disjoint i64 %434, 8, !dbg !14
  %.not12 = icmp ugt i64 %435, %451, !dbg !24
  %452 = select i1 %.not12, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %453 = or disjoint i64 %434, 9, !dbg !14
  %.not13 = icmp ugt i64 %435, %453, !dbg !24
  %454 = select i1 %.not13, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %455 = or disjoint i64 %434, 10, !dbg !14
  %.not14 = icmp ugt i64 %435, %455, !dbg !24
  %456 = select i1 %.not14, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %457 = or disjoint i64 %434, 11, !dbg !14
  %.not15 = icmp ugt i64 %435, %457, !dbg !24
  %458 = select i1 %.not15, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %459 = or disjoint i64 %434, 12, !dbg !14
  %.not16 = icmp ugt i64 %435, %459, !dbg !24
  %460 = select i1 %.not16, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %461 = or disjoint i64 %434, 13, !dbg !14
  %.not17 = icmp ugt i64 %435, %461, !dbg !24
  %462 = select i1 %.not17, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %463 = or disjoint i64 %434, 14, !dbg !14
  %.not18 = icmp ugt i64 %435, %463, !dbg !24
  %464 = select i1 %.not18, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %465 = or disjoint i64 %434, 15, !dbg !14
  %.not19 = icmp ugt i64 %435, %465, !dbg !24
  %466 = select i1 %.not19, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %467 = or disjoint i64 %434, 64, !dbg !14
  %.not20 = icmp ugt i64 %435, %467, !dbg !24
  %468 = select i1 %.not20, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %469 = or disjoint i64 %434, 65, !dbg !14
  %.not21 = icmp ugt i64 %435, %469, !dbg !24
  %470 = select i1 %.not21, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %471 = or disjoint i64 %434, 66, !dbg !14
  %.not22 = icmp ugt i64 %435, %471, !dbg !24
  %472 = select i1 %.not22, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %473 = or disjoint i64 %434, 67, !dbg !14
  %.not23 = icmp ugt i64 %435, %473, !dbg !24
  %474 = select i1 %.not23, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %475 = or disjoint i64 %434, 68, !dbg !14
  %.not24 = icmp ugt i64 %435, %475, !dbg !24
  %476 = select i1 %.not24, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %477 = or disjoint i64 %434, 69, !dbg !14
  %.not25 = icmp ugt i64 %435, %477, !dbg !24
  %478 = select i1 %.not25, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %479 = or disjoint i64 %434, 70, !dbg !14
  %.not26 = icmp ugt i64 %435, %479, !dbg !24
  %480 = select i1 %.not26, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %481 = or disjoint i64 %434, 71, !dbg !14
  %.not27 = icmp ugt i64 %435, %481, !dbg !24
  %482 = select i1 %.not27, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %483 = or disjoint i64 %434, 72, !dbg !14
  %.not28 = icmp ugt i64 %435, %483, !dbg !24
  %484 = select i1 %.not28, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %485 = or disjoint i64 %434, 73, !dbg !14
  %.not29 = icmp ugt i64 %435, %485, !dbg !24
  %486 = select i1 %.not29, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %487 = or disjoint i64 %434, 74, !dbg !14
  %.not30 = icmp ugt i64 %435, %487, !dbg !24
  %488 = select i1 %.not30, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %489 = or disjoint i64 %434, 75, !dbg !14
  %.not31 = icmp ugt i64 %435, %489, !dbg !24
  %490 = select i1 %.not31, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %491 = or disjoint i64 %434, 76, !dbg !14
  %.not32 = icmp ugt i64 %435, %491, !dbg !24
  %492 = select i1 %.not32, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %493 = or disjoint i64 %434, 77, !dbg !14
  %.not33 = icmp ugt i64 %435, %493, !dbg !24
  %494 = select i1 %.not33, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %495 = or disjoint i64 %434, 78, !dbg !14
  %.not34 = icmp ugt i64 %435, %495, !dbg !24
  %496 = select i1 %.not34, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %497 = or disjoint i64 %434, 79, !dbg !14
  %.not35 = icmp ugt i64 %435, %497, !dbg !24
  %498 = select i1 %.not35, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %499 = add nsw i64 %432, 65, !dbg !23
  %.not36 = icmp ugt i64 %499, %434, !dbg !24
  %500 = select i1 %.not36, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not37 = icmp ugt i64 %499, %437, !dbg !24
  %501 = select i1 %.not37, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not38 = icmp ugt i64 %499, %439, !dbg !24
  %502 = select i1 %.not38, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not39 = icmp ugt i64 %499, %441, !dbg !24
  %503 = select i1 %.not39, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not40 = icmp ugt i64 %499, %443, !dbg !24
  %504 = select i1 %.not40, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not41 = icmp ugt i64 %499, %445, !dbg !24
  %505 = select i1 %.not41, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not42 = icmp ugt i64 %499, %447, !dbg !24
  %506 = select i1 %.not42, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not43 = icmp ugt i64 %499, %449, !dbg !24
  %507 = select i1 %.not43, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not44 = icmp ugt i64 %499, %451, !dbg !24
  %508 = select i1 %.not44, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not45 = icmp ugt i64 %499, %453, !dbg !24
  %509 = select i1 %.not45, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not46 = icmp ugt i64 %499, %455, !dbg !24
  %510 = select i1 %.not46, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not47 = icmp ugt i64 %499, %457, !dbg !24
  %511 = select i1 %.not47, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not48 = icmp ugt i64 %499, %459, !dbg !24
  %512 = select i1 %.not48, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not49 = icmp ugt i64 %499, %461, !dbg !24
  %513 = select i1 %.not49, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not50 = icmp ugt i64 %499, %463, !dbg !24
  %514 = select i1 %.not50, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not51 = icmp ugt i64 %499, %465, !dbg !24
  %515 = select i1 %.not51, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not52 = icmp ugt i64 %499, %467, !dbg !24
  %516 = select i1 %.not52, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not53 = icmp ugt i64 %499, %469, !dbg !24
  %517 = select i1 %.not53, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not54 = icmp ugt i64 %499, %471, !dbg !24
  %518 = select i1 %.not54, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not55 = icmp ugt i64 %499, %473, !dbg !24
  %519 = select i1 %.not55, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not56 = icmp ugt i64 %499, %475, !dbg !24
  %520 = select i1 %.not56, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not57 = icmp ugt i64 %499, %477, !dbg !24
  %521 = select i1 %.not57, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not58 = icmp ugt i64 %499, %479, !dbg !24
  %522 = select i1 %.not58, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not59 = icmp ugt i64 %499, %481, !dbg !24
  %523 = select i1 %.not59, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not60 = icmp ugt i64 %499, %483, !dbg !24
  %524 = select i1 %.not60, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not61 = icmp ugt i64 %499, %485, !dbg !24
  %525 = select i1 %.not61, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not62 = icmp ugt i64 %499, %487, !dbg !24
  %526 = select i1 %.not62, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not63 = icmp ugt i64 %499, %489, !dbg !24
  %527 = select i1 %.not63, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not64 = icmp ugt i64 %499, %491, !dbg !24
  %528 = select i1 %.not64, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not65 = icmp ugt i64 %499, %493, !dbg !24
  %529 = select i1 %.not65, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not66 = icmp ugt i64 %499, %495, !dbg !24
  %530 = select i1 %.not66, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not67 = icmp ugt i64 %499, %497, !dbg !24
  %531 = select i1 %.not67, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %532 = insertelement <32 x float> poison, float %436, i64 0, !dbg !14
  %533 = insertelement <32 x float> %532, float %438, i64 1, !dbg !14
  %534 = insertelement <32 x float> %533, float %440, i64 2, !dbg !14
  %535 = insertelement <32 x float> %534, float %442, i64 3, !dbg !14
  %536 = insertelement <32 x float> %535, float %444, i64 4, !dbg !14
  %537 = insertelement <32 x float> %536, float %446, i64 5, !dbg !14
  %538 = insertelement <32 x float> %537, float %448, i64 6, !dbg !14
  %539 = insertelement <32 x float> %538, float %450, i64 7, !dbg !14
  %540 = insertelement <32 x float> %539, float %452, i64 8, !dbg !14
  %541 = insertelement <32 x float> %540, float %454, i64 9, !dbg !14
  %542 = insertelement <32 x float> %541, float %456, i64 10, !dbg !14
  %543 = insertelement <32 x float> %542, float %458, i64 11, !dbg !14
  %544 = insertelement <32 x float> %543, float %460, i64 12, !dbg !14
  %545 = insertelement <32 x float> %544, float %462, i64 13, !dbg !14
  %546 = insertelement <32 x float> %545, float %464, i64 14, !dbg !14
  %547 = insertelement <32 x float> %546, float %466, i64 15, !dbg !14
  %548 = insertelement <32 x float> %547, float %468, i64 16, !dbg !14
  %549 = insertelement <32 x float> %548, float %470, i64 17, !dbg !14
  %550 = insertelement <32 x float> %549, float %472, i64 18, !dbg !14
  %551 = insertelement <32 x float> %550, float %474, i64 19, !dbg !14
  %552 = insertelement <32 x float> %551, float %476, i64 20, !dbg !14
  %553 = insertelement <32 x float> %552, float %478, i64 21, !dbg !14
  %554 = insertelement <32 x float> %553, float %480, i64 22, !dbg !14
  %555 = insertelement <32 x float> %554, float %482, i64 23, !dbg !14
  %556 = insertelement <32 x float> %555, float %484, i64 24, !dbg !14
  %557 = insertelement <32 x float> %556, float %486, i64 25, !dbg !14
  %558 = insertelement <32 x float> %557, float %488, i64 26, !dbg !14
  %559 = insertelement <32 x float> %558, float %490, i64 27, !dbg !14
  %560 = insertelement <32 x float> %559, float %492, i64 28, !dbg !14
  %561 = insertelement <32 x float> %560, float %494, i64 29, !dbg !14
  %562 = insertelement <32 x float> %561, float %496, i64 30, !dbg !14
  %563 = insertelement <32 x float> %562, float %498, i64 31, !dbg !14
  %564 = insertvalue [2 x <32 x float>] poison, <32 x float> %563, 0, !dbg !14
  %565 = insertelement <32 x float> poison, float %500, i64 0, !dbg !14
  %566 = insertelement <32 x float> %565, float %501, i64 1, !dbg !14
  %567 = insertelement <32 x float> %566, float %502, i64 2, !dbg !14
  %568 = insertelement <32 x float> %567, float %503, i64 3, !dbg !14
  %569 = insertelement <32 x float> %568, float %504, i64 4, !dbg !14
  %570 = insertelement <32 x float> %569, float %505, i64 5, !dbg !14
  %571 = insertelement <32 x float> %570, float %506, i64 6, !dbg !14
  %572 = insertelement <32 x float> %571, float %507, i64 7, !dbg !14
  %573 = insertelement <32 x float> %572, float %508, i64 8, !dbg !14
  %574 = insertelement <32 x float> %573, float %509, i64 9, !dbg !14
  %575 = insertelement <32 x float> %574, float %510, i64 10, !dbg !14
  %576 = insertelement <32 x float> %575, float %511, i64 11, !dbg !14
  %577 = insertelement <32 x float> %576, float %512, i64 12, !dbg !14
  %578 = insertelement <32 x float> %577, float %513, i64 13, !dbg !14
  %579 = insertelement <32 x float> %578, float %514, i64 14, !dbg !14
  %580 = insertelement <32 x float> %579, float %515, i64 15, !dbg !14
  %581 = insertelement <32 x float> %580, float %516, i64 16, !dbg !14
  %582 = insertelement <32 x float> %581, float %517, i64 17, !dbg !14
  %583 = insertelement <32 x float> %582, float %518, i64 18, !dbg !14
  %584 = insertelement <32 x float> %583, float %519, i64 19, !dbg !14
  %585 = insertelement <32 x float> %584, float %520, i64 20, !dbg !14
  %586 = insertelement <32 x float> %585, float %521, i64 21, !dbg !14
  %587 = insertelement <32 x float> %586, float %522, i64 22, !dbg !14
  %588 = insertelement <32 x float> %587, float %523, i64 23, !dbg !14
  %589 = insertelement <32 x float> %588, float %524, i64 24, !dbg !14
  %590 = insertelement <32 x float> %589, float %525, i64 25, !dbg !14
  %591 = insertelement <32 x float> %590, float %526, i64 26, !dbg !14
  %592 = insertelement <32 x float> %591, float %527, i64 27, !dbg !14
  %593 = insertelement <32 x float> %592, float %528, i64 28, !dbg !14
  %594 = insertelement <32 x float> %593, float %529, i64 29, !dbg !14
  %595 = insertelement <32 x float> %594, float %530, i64 30, !dbg !14
  %596 = insertelement <32 x float> %595, float %531, i64 31, !dbg !14
  %597 = insertvalue [2 x <32 x float>] %564, <32 x float> %596, 1, !dbg !14
  br label %598, !dbg !14

598:                                              ; preds = %426, %427
  %599 = phi [2 x <32 x float>] [ %597, %427 ], [ zeroinitializer, %426 ], !dbg !14
  br label %600, !dbg !14

600:                                              ; preds = %598
  %601 = extractvalue [2 x <32 x float>] %37, 0, !dbg !26
  %602 = extractvalue [2 x <32 x float>] %599, 0, !dbg !26
  %603 = shufflevector <32 x float> %601, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %604 = shufflevector <32 x float> %602, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %605 = fadd <4 x float> %603, %604, !dbg !26
  %606 = extractelement <4 x float> %605, i64 0, !dbg !27
  %607 = call float @llvm.amdgcn.exp2.f32(float %606), !dbg !27
  %608 = insertelement <4 x float> poison, float %607, i64 0, !dbg !27
  %609 = extractelement <4 x float> %605, i64 1, !dbg !27
  %610 = call float @llvm.amdgcn.exp2.f32(float %609), !dbg !27
  %611 = fadd float %607, %610, !dbg !28
  %612 = insertelement <4 x float> %608, float %610, i64 1, !dbg !27
  %613 = extractelement <4 x float> %605, i64 2, !dbg !27
  %614 = call float @llvm.amdgcn.exp2.f32(float %613), !dbg !27
  %615 = fadd float %611, %614, !dbg !28
  %616 = insertelement <4 x float> %612, float %614, i64 2, !dbg !27
  %617 = extractelement <4 x float> %605, i64 3, !dbg !27
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !27
  %619 = fadd float %615, %618, !dbg !28
  %620 = insertelement <4 x float> %616, float %618, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %620) #9, !dbg !29, !frisk.register_lifetimes !1
  %621 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %621) #9, !dbg !26, !frisk.register_lifetimes !1
  %622 = shufflevector <32 x float> %601, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %623 = shufflevector <32 x float> %602, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %624 = fadd <4 x float> %622, %623, !dbg !26
  %625 = extractelement <4 x float> %624, i64 0, !dbg !27
  %626 = call float @llvm.amdgcn.exp2.f32(float %625), !dbg !27
  %627 = fadd float %619, %626, !dbg !28
  %628 = insertelement <4 x float> poison, float %626, i64 0, !dbg !27
  %629 = extractelement <4 x float> %624, i64 1, !dbg !27
  %630 = call float @llvm.amdgcn.exp2.f32(float %629), !dbg !27
  %631 = fadd float %627, %630, !dbg !28
  %632 = insertelement <4 x float> %628, float %630, i64 1, !dbg !27
  %633 = extractelement <4 x float> %624, i64 2, !dbg !27
  %634 = call float @llvm.amdgcn.exp2.f32(float %633), !dbg !27
  %635 = fadd float %631, %634, !dbg !28
  %636 = insertelement <4 x float> %632, float %634, i64 2, !dbg !27
  %637 = extractelement <4 x float> %624, i64 3, !dbg !27
  %638 = call float @llvm.amdgcn.exp2.f32(float %637), !dbg !27
  %639 = fadd float %635, %638, !dbg !28
  %640 = insertelement <4 x float> %636, float %638, i64 3, !dbg !27
  %math.result115 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %640) #9, !dbg !29, !frisk.register_lifetimes !1
  %641 = fptrunc <4 x float> %math.result115 to <4 x half>, !dbg !29
  %math.packed116 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %641) #9, !dbg !26, !frisk.register_lifetimes !1
  %642 = shufflevector <32 x float> %601, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %643 = shufflevector <32 x float> %602, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %644 = fadd <4 x float> %642, %643, !dbg !26
  %645 = extractelement <4 x float> %644, i64 0, !dbg !27
  %646 = call float @llvm.amdgcn.exp2.f32(float %645), !dbg !27
  %647 = fadd float %639, %646, !dbg !28
  %648 = insertelement <4 x float> poison, float %646, i64 0, !dbg !27
  %649 = extractelement <4 x float> %644, i64 1, !dbg !27
  %650 = call float @llvm.amdgcn.exp2.f32(float %649), !dbg !27
  %651 = fadd float %647, %650, !dbg !28
  %652 = insertelement <4 x float> %648, float %650, i64 1, !dbg !27
  %653 = extractelement <4 x float> %644, i64 2, !dbg !27
  %654 = call float @llvm.amdgcn.exp2.f32(float %653), !dbg !27
  %655 = fadd float %651, %654, !dbg !28
  %656 = insertelement <4 x float> %652, float %654, i64 2, !dbg !27
  %657 = extractelement <4 x float> %644, i64 3, !dbg !27
  %658 = call float @llvm.amdgcn.exp2.f32(float %657), !dbg !27
  %659 = fadd float %655, %658, !dbg !28
  %660 = insertelement <4 x float> %656, float %658, i64 3, !dbg !27
  %math.result117 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %660) #9, !dbg !29, !frisk.register_lifetimes !1
  %661 = fptrunc <4 x float> %math.result117 to <4 x half>, !dbg !29
  %math.packed118 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %661) #9, !dbg !26, !frisk.register_lifetimes !1
  %662 = shufflevector <32 x float> %601, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %663 = shufflevector <32 x float> %602, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %664 = fadd <4 x float> %662, %663, !dbg !26
  %665 = extractelement <4 x float> %664, i64 0, !dbg !27
  %666 = call float @llvm.amdgcn.exp2.f32(float %665), !dbg !27
  %667 = fadd float %659, %666, !dbg !28
  %668 = insertelement <4 x float> poison, float %666, i64 0, !dbg !27
  %669 = extractelement <4 x float> %664, i64 1, !dbg !27
  %670 = call float @llvm.amdgcn.exp2.f32(float %669), !dbg !27
  %671 = fadd float %667, %670, !dbg !28
  %672 = insertelement <4 x float> %668, float %670, i64 1, !dbg !27
  %673 = extractelement <4 x float> %664, i64 2, !dbg !27
  %674 = call float @llvm.amdgcn.exp2.f32(float %673), !dbg !27
  %675 = fadd float %671, %674, !dbg !28
  %676 = insertelement <4 x float> %672, float %674, i64 2, !dbg !27
  %677 = extractelement <4 x float> %664, i64 3, !dbg !27
  %678 = call float @llvm.amdgcn.exp2.f32(float %677), !dbg !27
  %679 = fadd float %675, %678, !dbg !28
  %680 = insertelement <4 x float> %676, float %678, i64 3, !dbg !27
  %math.result119 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %680) #9, !dbg !29, !frisk.register_lifetimes !1
  %681 = fptrunc <4 x float> %math.result119 to <4 x half>, !dbg !29
  %math.packed120 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %681) #9, !dbg !26, !frisk.register_lifetimes !1
  %682 = shufflevector <32 x float> %601, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %683 = shufflevector <32 x float> %602, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %684 = fadd <4 x float> %682, %683, !dbg !26
  %685 = extractelement <4 x float> %684, i64 0, !dbg !27
  %686 = call float @llvm.amdgcn.exp2.f32(float %685), !dbg !27
  %687 = fadd float %679, %686, !dbg !28
  %688 = insertelement <4 x float> poison, float %686, i64 0, !dbg !27
  %689 = extractelement <4 x float> %684, i64 1, !dbg !27
  %690 = call float @llvm.amdgcn.exp2.f32(float %689), !dbg !27
  %691 = fadd float %687, %690, !dbg !28
  %692 = insertelement <4 x float> %688, float %690, i64 1, !dbg !27
  %693 = extractelement <4 x float> %684, i64 2, !dbg !27
  %694 = call float @llvm.amdgcn.exp2.f32(float %693), !dbg !27
  %695 = fadd float %691, %694, !dbg !28
  %696 = insertelement <4 x float> %692, float %694, i64 2, !dbg !27
  %697 = extractelement <4 x float> %684, i64 3, !dbg !27
  %698 = call float @llvm.amdgcn.exp2.f32(float %697), !dbg !27
  %699 = fadd float %695, %698, !dbg !28
  %700 = insertelement <4 x float> %696, float %698, i64 3, !dbg !27
  %math.result121 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %700) #9, !dbg !29, !frisk.register_lifetimes !1
  %701 = fptrunc <4 x float> %math.result121 to <4 x half>, !dbg !29
  %math.packed122 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %701) #9, !dbg !26, !frisk.register_lifetimes !1
  %702 = shufflevector <32 x float> %601, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %703 = shufflevector <32 x float> %602, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %704 = fadd <4 x float> %702, %703, !dbg !26
  %705 = extractelement <4 x float> %704, i64 0, !dbg !27
  %706 = call float @llvm.amdgcn.exp2.f32(float %705), !dbg !27
  %707 = fadd float %699, %706, !dbg !28
  %708 = insertelement <4 x float> poison, float %706, i64 0, !dbg !27
  %709 = extractelement <4 x float> %704, i64 1, !dbg !27
  %710 = call float @llvm.amdgcn.exp2.f32(float %709), !dbg !27
  %711 = fadd float %707, %710, !dbg !28
  %712 = insertelement <4 x float> %708, float %710, i64 1, !dbg !27
  %713 = extractelement <4 x float> %704, i64 2, !dbg !27
  %714 = call float @llvm.amdgcn.exp2.f32(float %713), !dbg !27
  %715 = fadd float %711, %714, !dbg !28
  %716 = insertelement <4 x float> %712, float %714, i64 2, !dbg !27
  %717 = extractelement <4 x float> %704, i64 3, !dbg !27
  %718 = call float @llvm.amdgcn.exp2.f32(float %717), !dbg !27
  %719 = fadd float %715, %718, !dbg !28
  %720 = insertelement <4 x float> %716, float %718, i64 3, !dbg !27
  %math.result123 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %720) #9, !dbg !29, !frisk.register_lifetimes !1
  %721 = fptrunc <4 x float> %math.result123 to <4 x half>, !dbg !29
  %math.packed124 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %721) #9, !dbg !26, !frisk.register_lifetimes !1
  %722 = shufflevector <32 x float> %601, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %723 = shufflevector <32 x float> %602, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %724 = fadd <4 x float> %722, %723, !dbg !26
  %725 = extractelement <4 x float> %724, i64 0, !dbg !27
  %726 = call float @llvm.amdgcn.exp2.f32(float %725), !dbg !27
  %727 = fadd float %719, %726, !dbg !28
  %728 = insertelement <4 x float> poison, float %726, i64 0, !dbg !27
  %729 = extractelement <4 x float> %724, i64 1, !dbg !27
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !27
  %731 = fadd float %727, %730, !dbg !28
  %732 = insertelement <4 x float> %728, float %730, i64 1, !dbg !27
  %733 = extractelement <4 x float> %724, i64 2, !dbg !27
  %734 = call float @llvm.amdgcn.exp2.f32(float %733), !dbg !27
  %735 = fadd float %731, %734, !dbg !28
  %736 = insertelement <4 x float> %732, float %734, i64 2, !dbg !27
  %737 = extractelement <4 x float> %724, i64 3, !dbg !27
  %738 = call float @llvm.amdgcn.exp2.f32(float %737), !dbg !27
  %739 = fadd float %735, %738, !dbg !28
  %740 = insertelement <4 x float> %736, float %738, i64 3, !dbg !27
  %math.result125 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %740) #9, !dbg !29, !frisk.register_lifetimes !1
  %741 = fptrunc <4 x float> %math.result125 to <4 x half>, !dbg !29
  %math.packed126 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %741) #9, !dbg !26, !frisk.register_lifetimes !1
  %742 = shufflevector <32 x float> %601, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %743 = shufflevector <32 x float> %602, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %744 = fadd <4 x float> %742, %743, !dbg !26
  %745 = extractelement <4 x float> %744, i64 0, !dbg !27
  %746 = call float @llvm.amdgcn.exp2.f32(float %745), !dbg !27
  %747 = fadd float %739, %746, !dbg !28
  %748 = insertelement <4 x float> poison, float %746, i64 0, !dbg !27
  %749 = extractelement <4 x float> %744, i64 1, !dbg !27
  %750 = call float @llvm.amdgcn.exp2.f32(float %749), !dbg !27
  %751 = fadd float %747, %750, !dbg !28
  %752 = insertelement <4 x float> %748, float %750, i64 1, !dbg !27
  %753 = extractelement <4 x float> %744, i64 2, !dbg !27
  %754 = call float @llvm.amdgcn.exp2.f32(float %753), !dbg !27
  %755 = fadd float %751, %754, !dbg !28
  %756 = insertelement <4 x float> %752, float %754, i64 2, !dbg !27
  %757 = extractelement <4 x float> %744, i64 3, !dbg !27
  %758 = call float @llvm.amdgcn.exp2.f32(float %757), !dbg !27
  %759 = fadd float %755, %758, !dbg !28
  %760 = insertelement <4 x float> %756, float %758, i64 3, !dbg !27
  %math.result127 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %760) #9, !dbg !29, !frisk.register_lifetimes !1
  %761 = fptrunc <4 x float> %math.result127 to <4 x half>, !dbg !29
  %math.packed128 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %761) #9, !dbg !26, !frisk.register_lifetimes !1
  %762 = extractvalue [2 x <32 x float>] %37, 1, !dbg !26
  %763 = extractvalue [2 x <32 x float>] %599, 1, !dbg !26
  %764 = shufflevector <32 x float> %762, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %765 = shufflevector <32 x float> %763, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %766 = fadd <4 x float> %764, %765, !dbg !26
  %767 = extractelement <4 x float> %766, i64 0, !dbg !27
  %768 = call float @llvm.amdgcn.exp2.f32(float %767), !dbg !27
  %769 = insertelement <4 x float> poison, float %768, i64 0, !dbg !27
  %770 = extractelement <4 x float> %766, i64 1, !dbg !27
  %771 = call float @llvm.amdgcn.exp2.f32(float %770), !dbg !27
  %772 = fadd float %768, %771, !dbg !28
  %773 = insertelement <4 x float> %769, float %771, i64 1, !dbg !27
  %774 = extractelement <4 x float> %766, i64 2, !dbg !27
  %775 = call float @llvm.amdgcn.exp2.f32(float %774), !dbg !27
  %776 = fadd float %772, %775, !dbg !28
  %777 = insertelement <4 x float> %773, float %775, i64 2, !dbg !27
  %778 = extractelement <4 x float> %766, i64 3, !dbg !27
  %779 = call float @llvm.amdgcn.exp2.f32(float %778), !dbg !27
  %780 = fadd float %776, %779, !dbg !28
  %781 = insertelement <4 x float> %777, float %779, i64 3, !dbg !27
  %math.result129 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %781) #9, !dbg !29, !frisk.register_lifetimes !1
  %782 = fptrunc <4 x float> %math.result129 to <4 x half>, !dbg !29
  %math.packed130 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %782) #9, !dbg !26, !frisk.register_lifetimes !1
  %783 = shufflevector <32 x float> %762, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %784 = shufflevector <32 x float> %763, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %785 = fadd <4 x float> %783, %784, !dbg !26
  %786 = extractelement <4 x float> %785, i64 0, !dbg !27
  %787 = call float @llvm.amdgcn.exp2.f32(float %786), !dbg !27
  %788 = fadd float %780, %787, !dbg !28
  %789 = insertelement <4 x float> poison, float %787, i64 0, !dbg !27
  %790 = extractelement <4 x float> %785, i64 1, !dbg !27
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !27
  %792 = fadd float %788, %791, !dbg !28
  %793 = insertelement <4 x float> %789, float %791, i64 1, !dbg !27
  %794 = extractelement <4 x float> %785, i64 2, !dbg !27
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !27
  %796 = fadd float %792, %795, !dbg !28
  %797 = insertelement <4 x float> %793, float %795, i64 2, !dbg !27
  %798 = extractelement <4 x float> %785, i64 3, !dbg !27
  %799 = call float @llvm.amdgcn.exp2.f32(float %798), !dbg !27
  %800 = fadd float %796, %799, !dbg !28
  %801 = insertelement <4 x float> %797, float %799, i64 3, !dbg !27
  %math.result131 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %801) #9, !dbg !29, !frisk.register_lifetimes !1
  %802 = fptrunc <4 x float> %math.result131 to <4 x half>, !dbg !29
  %math.packed132 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %802) #9, !dbg !26, !frisk.register_lifetimes !1
  %803 = shufflevector <32 x float> %762, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %804 = shufflevector <32 x float> %763, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %805 = fadd <4 x float> %803, %804, !dbg !26
  %806 = extractelement <4 x float> %805, i64 0, !dbg !27
  %807 = call float @llvm.amdgcn.exp2.f32(float %806), !dbg !27
  %808 = fadd float %800, %807, !dbg !28
  %809 = insertelement <4 x float> poison, float %807, i64 0, !dbg !27
  %810 = extractelement <4 x float> %805, i64 1, !dbg !27
  %811 = call float @llvm.amdgcn.exp2.f32(float %810), !dbg !27
  %812 = fadd float %808, %811, !dbg !28
  %813 = insertelement <4 x float> %809, float %811, i64 1, !dbg !27
  %814 = extractelement <4 x float> %805, i64 2, !dbg !27
  %815 = call float @llvm.amdgcn.exp2.f32(float %814), !dbg !27
  %816 = fadd float %812, %815, !dbg !28
  %817 = insertelement <4 x float> %813, float %815, i64 2, !dbg !27
  %818 = extractelement <4 x float> %805, i64 3, !dbg !27
  %819 = call float @llvm.amdgcn.exp2.f32(float %818), !dbg !27
  %820 = fadd float %816, %819, !dbg !28
  %821 = insertelement <4 x float> %817, float %819, i64 3, !dbg !27
  %math.result133 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %821) #9, !dbg !29, !frisk.register_lifetimes !1
  %822 = fptrunc <4 x float> %math.result133 to <4 x half>, !dbg !29
  %math.packed134 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %822) #9, !dbg !26, !frisk.register_lifetimes !1
  %823 = shufflevector <32 x float> %762, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %824 = shufflevector <32 x float> %763, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %825 = fadd <4 x float> %823, %824, !dbg !26
  %826 = extractelement <4 x float> %825, i64 0, !dbg !27
  %827 = call float @llvm.amdgcn.exp2.f32(float %826), !dbg !27
  %828 = fadd float %820, %827, !dbg !28
  %829 = insertelement <4 x float> poison, float %827, i64 0, !dbg !27
  %830 = extractelement <4 x float> %825, i64 1, !dbg !27
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !27
  %832 = fadd float %828, %831, !dbg !28
  %833 = insertelement <4 x float> %829, float %831, i64 1, !dbg !27
  %834 = extractelement <4 x float> %825, i64 2, !dbg !27
  %835 = call float @llvm.amdgcn.exp2.f32(float %834), !dbg !27
  %836 = fadd float %832, %835, !dbg !28
  %837 = insertelement <4 x float> %833, float %835, i64 2, !dbg !27
  %838 = extractelement <4 x float> %825, i64 3, !dbg !27
  %839 = call float @llvm.amdgcn.exp2.f32(float %838), !dbg !27
  %840 = fadd float %836, %839, !dbg !28
  %841 = insertelement <4 x float> %837, float %839, i64 3, !dbg !27
  %math.result135 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %841) #9, !dbg !29, !frisk.register_lifetimes !1
  %842 = fptrunc <4 x float> %math.result135 to <4 x half>, !dbg !29
  %math.packed136 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %842) #9, !dbg !26, !frisk.register_lifetimes !1
  %843 = shufflevector <32 x float> %762, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %844 = shufflevector <32 x float> %763, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %845 = fadd <4 x float> %843, %844, !dbg !26
  %846 = extractelement <4 x float> %845, i64 0, !dbg !27
  %847 = call float @llvm.amdgcn.exp2.f32(float %846), !dbg !27
  %848 = fadd float %840, %847, !dbg !28
  %849 = insertelement <4 x float> poison, float %847, i64 0, !dbg !27
  %850 = extractelement <4 x float> %845, i64 1, !dbg !27
  %851 = call float @llvm.amdgcn.exp2.f32(float %850), !dbg !27
  %852 = fadd float %848, %851, !dbg !28
  %853 = insertelement <4 x float> %849, float %851, i64 1, !dbg !27
  %854 = extractelement <4 x float> %845, i64 2, !dbg !27
  %855 = call float @llvm.amdgcn.exp2.f32(float %854), !dbg !27
  %856 = fadd float %852, %855, !dbg !28
  %857 = insertelement <4 x float> %853, float %855, i64 2, !dbg !27
  %858 = extractelement <4 x float> %845, i64 3, !dbg !27
  %859 = call float @llvm.amdgcn.exp2.f32(float %858), !dbg !27
  %860 = fadd float %856, %859, !dbg !28
  %861 = insertelement <4 x float> %857, float %859, i64 3, !dbg !27
  %math.result137 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %861) #9, !dbg !29, !frisk.register_lifetimes !1
  %862 = fptrunc <4 x float> %math.result137 to <4 x half>, !dbg !29
  %math.packed138 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %862) #9, !dbg !26, !frisk.register_lifetimes !1
  %863 = shufflevector <32 x float> %762, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %864 = shufflevector <32 x float> %763, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %865 = fadd <4 x float> %863, %864, !dbg !26
  %866 = extractelement <4 x float> %865, i64 0, !dbg !27
  %867 = call float @llvm.amdgcn.exp2.f32(float %866), !dbg !27
  %868 = fadd float %860, %867, !dbg !28
  %869 = insertelement <4 x float> poison, float %867, i64 0, !dbg !27
  %870 = extractelement <4 x float> %865, i64 1, !dbg !27
  %871 = call float @llvm.amdgcn.exp2.f32(float %870), !dbg !27
  %872 = fadd float %868, %871, !dbg !28
  %873 = insertelement <4 x float> %869, float %871, i64 1, !dbg !27
  %874 = extractelement <4 x float> %865, i64 2, !dbg !27
  %875 = call float @llvm.amdgcn.exp2.f32(float %874), !dbg !27
  %876 = fadd float %872, %875, !dbg !28
  %877 = insertelement <4 x float> %873, float %875, i64 2, !dbg !27
  %878 = extractelement <4 x float> %865, i64 3, !dbg !27
  %879 = call float @llvm.amdgcn.exp2.f32(float %878), !dbg !27
  %880 = fadd float %876, %879, !dbg !28
  %881 = insertelement <4 x float> %877, float %879, i64 3, !dbg !27
  %math.result139 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %881) #9, !dbg !29, !frisk.register_lifetimes !1
  %882 = fptrunc <4 x float> %math.result139 to <4 x half>, !dbg !29
  %math.packed140 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %882) #9, !dbg !26, !frisk.register_lifetimes !1
  %883 = shufflevector <32 x float> %762, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %884 = shufflevector <32 x float> %763, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %885 = fadd <4 x float> %883, %884, !dbg !26
  %886 = extractelement <4 x float> %885, i64 0, !dbg !27
  %887 = call float @llvm.amdgcn.exp2.f32(float %886), !dbg !27
  %888 = fadd float %880, %887, !dbg !28
  %889 = insertelement <4 x float> poison, float %887, i64 0, !dbg !27
  %890 = extractelement <4 x float> %885, i64 1, !dbg !27
  %891 = call float @llvm.amdgcn.exp2.f32(float %890), !dbg !27
  %892 = fadd float %888, %891, !dbg !28
  %893 = insertelement <4 x float> %889, float %891, i64 1, !dbg !27
  %894 = extractelement <4 x float> %885, i64 2, !dbg !27
  %895 = call float @llvm.amdgcn.exp2.f32(float %894), !dbg !27
  %896 = fadd float %892, %895, !dbg !28
  %897 = insertelement <4 x float> %893, float %895, i64 2, !dbg !27
  %898 = extractelement <4 x float> %885, i64 3, !dbg !27
  %899 = call float @llvm.amdgcn.exp2.f32(float %898), !dbg !27
  %900 = fadd float %896, %899, !dbg !28
  %901 = insertelement <4 x float> %897, float %899, i64 3, !dbg !27
  %math.result141 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %901) #9, !dbg !29, !frisk.register_lifetimes !1
  %902 = fptrunc <4 x float> %math.result141 to <4 x half>, !dbg !29
  %math.packed142 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %902) #9, !dbg !26, !frisk.register_lifetimes !1
  %903 = shufflevector <32 x float> %762, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %904 = shufflevector <32 x float> %763, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %905 = fadd <4 x float> %903, %904, !dbg !26
  %906 = extractelement <4 x float> %905, i64 0, !dbg !27
  %907 = call float @llvm.amdgcn.exp2.f32(float %906), !dbg !27
  %908 = fadd float %900, %907, !dbg !28
  %909 = insertelement <4 x float> poison, float %907, i64 0, !dbg !27
  %910 = extractelement <4 x float> %905, i64 1, !dbg !27
  %911 = call float @llvm.amdgcn.exp2.f32(float %910), !dbg !27
  %912 = fadd float %908, %911, !dbg !28
  %913 = insertelement <4 x float> %909, float %911, i64 1, !dbg !27
  %914 = extractelement <4 x float> %905, i64 2, !dbg !27
  %915 = call float @llvm.amdgcn.exp2.f32(float %914), !dbg !27
  %916 = fadd float %912, %915, !dbg !28
  %917 = insertelement <4 x float> %913, float %915, i64 2, !dbg !27
  %918 = extractelement <4 x float> %905, i64 3, !dbg !27
  %919 = call float @llvm.amdgcn.exp2.f32(float %918), !dbg !27
  %920 = fadd float %916, %919, !dbg !28
  %921 = insertelement <4 x float> %917, float %919, i64 3, !dbg !27
  %math.result143 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %921) #9, !dbg !29, !frisk.register_lifetimes !1
  %922 = fptrunc <4 x float> %math.result143 to <4 x half>, !dbg !29
  %math.packed144 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %922) #9, !dbg !28, !frisk.register_lifetimes !1
  %923 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %924 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %923), !dbg !28
  %925 = add nuw nsw i32 %924, 64, !dbg !28
  %926 = xor i32 %924, 16, !dbg !28
  %927 = and i32 %925, 192, !dbg !28
  %928 = icmp samesign ult i32 %926, %927, !dbg !28
  %929 = select i1 %928, i32 %926, i32 %924, !dbg !28
  %930 = shl nuw nsw i32 %929, 2, !dbg !28
  %931 = bitcast float %759 to i32, !dbg !28
  %932 = call i32 @llvm.amdgcn.ds.bpermute(i32 %930, i32 %931), !dbg !28
  %933 = bitcast i32 %932 to float, !dbg !28
  %934 = fadd float %759, %933, !dbg !28
  %935 = xor i32 %924, 32, !dbg !28
  %936 = icmp samesign ult i32 %935, %927, !dbg !28
  %937 = select i1 %936, i32 %935, i32 %924, !dbg !28
  %938 = shl nuw nsw i32 %937, 2, !dbg !28
  %939 = bitcast float %934 to i32, !dbg !28
  %940 = call i32 @llvm.amdgcn.ds.bpermute(i32 %938, i32 %939), !dbg !28
  %941 = bitcast i32 %940 to float, !dbg !28
  %942 = fadd float %934, %941, !dbg !28
  %943 = extractvalue [2 x <1 x float>] %28, 0, !dbg !28
  %944 = extractelement <1 x float> %943, i64 0, !dbg !28
  %945 = fadd float %942, %944, !dbg !28
  %946 = insertelement <1 x float> poison, float %945, i64 0, !dbg !15
  %947 = insertvalue [2 x <1 x float>] poison, <1 x float> %946, 0, !dbg !15
  %948 = bitcast float %920 to i32, !dbg !28
  %949 = call i32 @llvm.amdgcn.ds.bpermute(i32 %930, i32 %948), !dbg !28
  %950 = bitcast i32 %949 to float, !dbg !28
  %951 = fadd float %920, %950, !dbg !28
  %952 = bitcast float %951 to i32, !dbg !28
  %953 = call i32 @llvm.amdgcn.ds.bpermute(i32 %938, i32 %952), !dbg !28
  %954 = bitcast i32 %953 to float, !dbg !28
  %955 = fadd float %951, %954, !dbg !28
  %956 = extractvalue [2 x <1 x float>] %28, 1, !dbg !28
  %957 = extractelement <1 x float> %956, i64 0, !dbg !28
  %958 = fadd float %955, %957, !dbg !28
  %959 = insertelement <1 x float> poison, float %958, i64 0, !dbg !15
  %960 = insertvalue [2 x <1 x float>] %947, <1 x float> %959, 1, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread492 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !18, !range !34, !frisk.register_lifetimes !1
  %address.local493 = shl nuw nsw i32 %address.thread492, 1, !dbg !18
  %address.local494 = and i32 %address.local493, 8, !dbg !18
  %address.local495 = shl nuw nsw i32 %address.thread492, 4, !dbg !18
  %address.local496 = and i32 %address.local495, 176, !dbg !18
  %address.local497 = or disjoint i32 %address.local494, %address.local496, !dbg !18
  %address.local498 = shl nuw nsw i32 %address.thread492, 6, !dbg !18
  %address.local499 = and i32 %address.local498, 3072, !dbg !18
  %address.local500 = or disjoint i32 %address.local497, %address.local499, !dbg !18
  %address.local501 = zext nneg i32 %address.local500 to i64, !dbg !18
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !18
  %961 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local502, i32 0) #0, !dbg !18
  %962 = bitcast <4 x i32> %961 to <8 x half>, !dbg !18
  %address.local503 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !18
  %address.local504 = getelementptr i8, ptr addrspace(3) %address.local503, i64 128, !dbg !18
  %963 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local504, i32 0) #0, !dbg !18
  %964 = bitcast <4 x i32> %963 to <8 x half>, !dbg !18
  %address.local505 = shl nuw nsw i32 %address.thread492, 1, !dbg !18
  %address.local506 = and i32 %address.local505, 8, !dbg !18
  %address.local507 = shl nuw nsw i32 %address.thread492, 4, !dbg !18
  %address.local508 = and i32 %address.local507, 176, !dbg !18
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !18
  %address.local510 = shl nuw nsw i32 %address.thread492, 6, !dbg !18
  %address.local511 = and i32 %address.local510, 3072, !dbg !18
  %address.local512 = or disjoint i32 %address.local509, %address.local511, !dbg !18
  %address.local513 = zext nneg i32 %address.local512 to i64, !dbg !18
  %address.local514 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local513, !dbg !18
  %address.local515 = getelementptr i8, ptr addrspace(3) %address.local514, i64 512, !dbg !18
  %965 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local515, i32 0) #0, !dbg !18
  %966 = bitcast <4 x i32> %965 to <8 x half>, !dbg !18
  %address.local516 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local513, !dbg !18
  %address.local517 = getelementptr i8, ptr addrspace(3) %address.local516, i64 640, !dbg !18
  %967 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local517, i32 0) #0, !dbg !18
  %968 = bitcast <4 x i32> %967 to <8 x half>, !dbg !18
  %address.local518 = shl nuw nsw i32 %address.thread492, 1, !dbg !18
  %address.local519 = and i32 %address.local518, 8, !dbg !18
  %address.local520 = shl nuw nsw i32 %address.thread492, 4, !dbg !18
  %address.local521 = and i32 %address.local520, 176, !dbg !18
  %address.local522 = or disjoint i32 %address.local519, %address.local521, !dbg !18
  %address.local523 = shl nuw nsw i32 %address.thread492, 6, !dbg !18
  %address.local524 = and i32 %address.local523, 3072, !dbg !18
  %address.local525 = or disjoint i32 %address.local522, %address.local524, !dbg !18
  %address.local526 = zext nneg i32 %address.local525 to i64, !dbg !18
  %address.local527 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local526, !dbg !18
  %address.local528 = getelementptr i8, ptr addrspace(3) %address.local527, i64 1024, !dbg !18
  %969 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local528, i32 0) #0, !dbg !18
  %970 = bitcast <4 x i32> %969 to <8 x half>, !dbg !18
  %address.local529 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local526, !dbg !18
  %address.local530 = getelementptr i8, ptr addrspace(3) %address.local529, i64 1152, !dbg !18
  %971 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local530, i32 0) #0, !dbg !18
  %972 = bitcast <4 x i32> %971 to <8 x half>, !dbg !18
  %address.local531 = shl nuw nsw i32 %address.thread492, 1, !dbg !18
  %address.local532 = and i32 %address.local531, 8, !dbg !18
  %address.local533 = shl nuw nsw i32 %address.thread492, 4, !dbg !18
  %address.local534 = and i32 %address.local533, 176, !dbg !18
  %address.local535 = or disjoint i32 %address.local532, %address.local534, !dbg !18
  %address.local536 = shl nuw nsw i32 %address.thread492, 6, !dbg !18
  %address.local537 = and i32 %address.local536, 3072, !dbg !18
  %address.local538 = or disjoint i32 %address.local535, %address.local537, !dbg !18
  %address.local539 = zext nneg i32 %address.local538 to i64, !dbg !18
  %address.local540 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local539, !dbg !18
  %address.local541 = getelementptr i8, ptr addrspace(3) %address.local540, i64 1536, !dbg !18
  %973 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local541, i32 0) #0, !dbg !18
  %974 = bitcast <4 x i32> %973 to <8 x half>, !dbg !18
  %address.local542 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local539, !dbg !18
  %address.local543 = getelementptr i8, ptr addrspace(3) %address.local542, i64 1664, !dbg !18
  %975 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local543, i32 0) #0, !dbg !18
  %976 = bitcast <4 x i32> %975 to <8 x half>, !dbg !18
  %address.local544 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !18
  %address.local545 = getelementptr i8, ptr addrspace(3) %address.local544, i64 8192, !dbg !18
  %977 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local545, i32 0) #0, !dbg !18
  %978 = bitcast <4 x i32> %977 to <8 x half>, !dbg !18
  %address.local546 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !18
  %address.local547 = getelementptr i8, ptr addrspace(3) %address.local546, i64 8320, !dbg !18
  %979 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local547, i32 0) #0, !dbg !18
  %980 = bitcast <4 x i32> %979 to <8 x half>, !dbg !18
  %address.local548 = shl nuw nsw i32 %address.thread492, 1, !dbg !18
  %address.local549 = and i32 %address.local548, 8, !dbg !18
  %address.local550 = shl nuw nsw i32 %address.thread492, 4, !dbg !18
  %address.local551 = and i32 %address.local550, 176, !dbg !18
  %address.local552 = or disjoint i32 %address.local549, %address.local551, !dbg !18
  %address.local553 = shl nuw nsw i32 %address.thread492, 6, !dbg !18
  %address.local554 = and i32 %address.local553, 3072, !dbg !18
  %address.local555 = or disjoint i32 %address.local552, %address.local554, !dbg !18
  %address.local556 = or disjoint i32 %address.local555, 4096, !dbg !18
  %address.local557 = zext nneg i32 %address.local556 to i64, !dbg !18
  %address.local558 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local557, !dbg !18
  %address.local559 = getelementptr i8, ptr addrspace(3) %address.local558, i64 512, !dbg !18
  %981 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local559, i32 0) #0, !dbg !18
  %982 = bitcast <4 x i32> %981 to <8 x half>, !dbg !18
  %address.local560 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local557, !dbg !18
  %address.local561 = getelementptr i8, ptr addrspace(3) %address.local560, i64 640, !dbg !18
  %983 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local561, i32 0) #0, !dbg !18
  %984 = bitcast <4 x i32> %983 to <8 x half>, !dbg !18
  %address.local562 = shl nuw nsw i32 %address.thread492, 1, !dbg !18
  %address.local563 = and i32 %address.local562, 8, !dbg !18
  %address.local564 = shl nuw nsw i32 %address.thread492, 4, !dbg !18
  %address.local565 = and i32 %address.local564, 176, !dbg !18
  %address.local566 = or disjoint i32 %address.local563, %address.local565, !dbg !18
  %address.local567 = shl nuw nsw i32 %address.thread492, 6, !dbg !18
  %address.local568 = and i32 %address.local567, 3072, !dbg !18
  %address.local569 = or disjoint i32 %address.local566, %address.local568, !dbg !18
  %address.local570 = or disjoint i32 %address.local569, 4096, !dbg !18
  %address.local571 = zext nneg i32 %address.local570 to i64, !dbg !18
  %address.local572 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local571, !dbg !18
  %address.local573 = getelementptr i8, ptr addrspace(3) %address.local572, i64 1024, !dbg !18
  %985 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local573, i32 0) #0, !dbg !18
  %986 = bitcast <4 x i32> %985 to <8 x half>, !dbg !18
  %address.local574 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local571, !dbg !18
  %address.local575 = getelementptr i8, ptr addrspace(3) %address.local574, i64 1152, !dbg !18
  %987 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local575, i32 0) #0, !dbg !18
  %988 = bitcast <4 x i32> %987 to <8 x half>, !dbg !18
  %address.local576 = shl nuw nsw i32 %address.thread492, 1, !dbg !18
  %address.local577 = and i32 %address.local576, 8, !dbg !18
  %address.local578 = shl nuw nsw i32 %address.thread492, 4, !dbg !18
  %address.local579 = and i32 %address.local578, 176, !dbg !18
  %address.local580 = or disjoint i32 %address.local577, %address.local579, !dbg !18
  %address.local581 = shl nuw nsw i32 %address.thread492, 6, !dbg !18
  %address.local582 = and i32 %address.local581, 3072, !dbg !18
  %address.local583 = or disjoint i32 %address.local580, %address.local582, !dbg !18
  %address.local584 = or disjoint i32 %address.local583, 4096, !dbg !18
  %address.local585 = zext nneg i32 %address.local584 to i64, !dbg !18
  %address.local586 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local585, !dbg !18
  %address.local587 = getelementptr i8, ptr addrspace(3) %address.local586, i64 1536, !dbg !18
  %989 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local587, i32 0) #0, !dbg !18
  %990 = bitcast <4 x i32> %989 to <8 x half>, !dbg !18
  %address.local588 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local585, !dbg !18
  %address.local589 = getelementptr i8, ptr addrspace(3) %address.local588, i64 1664, !dbg !18
  %991 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local589, i32 0) #0, !dbg !18
  %992 = bitcast <4 x i32> %991 to <8 x half>, !dbg !18
  %993 = add nsw i64 %32, %8, !dbg !17
  %994 = shl i64 %993, 19, !dbg !17
  %995 = and i64 %994, 16252928, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %996 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %995, !dbg !17
  %997 = getelementptr [2 x i8], ptr addrspace(1) %996, i64 %364, !dbg !17
  %998 = getelementptr i8, ptr addrspace(1) %997, i64 128, !dbg !17
  %address.local590 = zext nneg i32 %address.thread492 to i64, !dbg !13
  %address.local591 = shl nuw nsw i64 %address.local590, 3, !dbg !12
  %address.local592 = and i64 %address.local591, 1536, !dbg !12
  %address.local593 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local592, !dbg !12
  %address.local594 = ptrtoint ptr addrspace(3) %address.local593 to i64, !dbg !12
  %address.local595 = trunc i64 %address.local594 to i32, !dbg !12
  %999 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local595), !dbg !17
  %1000 = zext i32 %999 to i64, !dbg !17
  %1001 = inttoptr i64 %1000 to ptr addrspace(3), !dbg !17
  %address.local596 = lshr i64 %address.local590, 3, !dbg !17
  %address.local597 = and i64 %address.local596, 30, !dbg !17
  %address.local598 = and i64 %address.local590, 1, !dbg !17
  %address.local599 = or disjoint i64 %address.local597, %address.local598, !dbg !17
  %address.local600 = shl nuw nsw i64 %address.local599, 7, !dbg !17
  %address.local601 = shl nuw nsw i64 %address.local590, 2, !dbg !17
  %address.local602 = and i64 %address.local601, 56, !dbg !17
  %address.local603 = or disjoint i64 %address.local600, %address.local602, !dbg !17
  %address.local604 = getelementptr [2 x i8], ptr addrspace(1) %998, i64 %address.local603, !dbg !17
  %1002 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1003 = ptrtoint ptr addrspace(1) %address.local604 to i64, !dbg !17
  %1004 = sub i64 %1003, %1002, !dbg !17
  %dma.buffer.offset871 = trunc i64 %1004 to i32, !dbg !17
  %1005 = trunc i64 %1002 to i32, !dbg !17
  %1006 = insertelement <4 x i32> poison, i32 %1005, i64 0, !dbg !17
  %1007 = lshr i64 %1002, 32, !dbg !17
  %1008 = trunc i64 %1007 to i32, !dbg !17
  %1009 = insertelement <4 x i32> %1006, i32 %1008, i64 1, !dbg !17
  %1010 = insertelement <4 x i32> %1009, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc872 = insertelement <4 x i32> %1010, i32 131072, i64 3, !dbg !17
  %1011 = ptrtoint ptr addrspace(3) %1001 to i32, !dbg !17
  %dma.buffer.lds873 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1011), !dbg !17
  %1012 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds873, i32 %dma.buffer.offset871, <4 x i32> %dma.buffer.rsrc872) #10, !dbg !17
  %address.local605 = or disjoint i64 %address.local590, 256, !dbg !12
  %address.local606 = shl nuw nsw i64 %address.local605, 3, !dbg !12
  %address.local607 = and i64 %address.local606, 3584, !dbg !12
  %address.local608 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local607, !dbg !12
  %address.local609 = ptrtoint ptr addrspace(3) %address.local608 to i64, !dbg !12
  %address.local610 = trunc i64 %address.local609 to i32, !dbg !12
  %1013 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local610), !dbg !17
  %1014 = zext i32 %1013 to i64, !dbg !17
  %1015 = inttoptr i64 %1014 to ptr addrspace(3), !dbg !17
  %address.local611 = lshr i64 %address.local605, 3, !dbg !17
  %address.local612 = and i64 %address.local611, 62, !dbg !17
  %address.local613 = and i64 %address.local590, 1, !dbg !17
  %address.local614 = or disjoint i64 %address.local612, %address.local613, !dbg !17
  %address.local615 = shl nuw nsw i64 %address.local614, 7, !dbg !17
  %address.local616 = shl nuw nsw i64 %address.local590, 2, !dbg !17
  %address.local617 = and i64 %address.local616, 56, !dbg !17
  %address.local618 = or disjoint i64 %address.local615, %address.local617, !dbg !17
  %address.local619 = getelementptr [2 x i8], ptr addrspace(1) %998, i64 %address.local618, !dbg !17
  %1016 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1017 = ptrtoint ptr addrspace(1) %address.local619 to i64, !dbg !17
  %1018 = sub i64 %1017, %1016, !dbg !17
  %dma.buffer.offset868 = trunc i64 %1018 to i32, !dbg !17
  %1019 = trunc i64 %1016 to i32, !dbg !17
  %1020 = insertelement <4 x i32> poison, i32 %1019, i64 0, !dbg !17
  %1021 = lshr i64 %1016, 32, !dbg !17
  %1022 = trunc i64 %1021 to i32, !dbg !17
  %1023 = insertelement <4 x i32> %1020, i32 %1022, i64 1, !dbg !17
  %1024 = insertelement <4 x i32> %1023, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc869 = insertelement <4 x i32> %1024, i32 131072, i64 3, !dbg !17
  %1025 = ptrtoint ptr addrspace(3) %1015 to i32, !dbg !17
  %dma.buffer.lds870 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1025), !dbg !17
  %1026 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds870, i32 %dma.buffer.offset868, <4 x i32> %dma.buffer.rsrc869) #10, !dbg !17
  %address.local620 = or disjoint i64 %address.local590, 512, !dbg !12
  %address.local621 = shl nuw nsw i64 %address.local620, 3, !dbg !12
  %address.local622 = and i64 %address.local621, 5632, !dbg !12
  %address.local623 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local622, !dbg !12
  %address.local624 = ptrtoint ptr addrspace(3) %address.local623 to i64, !dbg !12
  %address.local625 = trunc i64 %address.local624 to i32, !dbg !12
  %1027 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local625), !dbg !17
  %1028 = zext i32 %1027 to i64, !dbg !17
  %1029 = inttoptr i64 %1028 to ptr addrspace(3), !dbg !17
  %address.local626 = lshr i64 %address.local620, 3, !dbg !17
  %address.local627 = and i64 %address.local626, 94, !dbg !17
  %address.local628 = and i64 %address.local590, 1, !dbg !17
  %address.local629 = or disjoint i64 %address.local627, %address.local628, !dbg !17
  %address.local630 = shl nuw nsw i64 %address.local629, 7, !dbg !17
  %address.local631 = shl nuw nsw i64 %address.local590, 2, !dbg !17
  %address.local632 = and i64 %address.local631, 56, !dbg !17
  %address.local633 = or disjoint i64 %address.local630, %address.local632, !dbg !17
  %address.local634 = getelementptr [2 x i8], ptr addrspace(1) %998, i64 %address.local633, !dbg !17
  %1030 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1031 = ptrtoint ptr addrspace(1) %address.local634 to i64, !dbg !17
  %1032 = sub i64 %1031, %1030, !dbg !17
  %dma.buffer.offset865 = trunc i64 %1032 to i32, !dbg !17
  %1033 = trunc i64 %1030 to i32, !dbg !17
  %1034 = insertelement <4 x i32> poison, i32 %1033, i64 0, !dbg !17
  %1035 = lshr i64 %1030, 32, !dbg !17
  %1036 = trunc i64 %1035 to i32, !dbg !17
  %1037 = insertelement <4 x i32> %1034, i32 %1036, i64 1, !dbg !17
  %1038 = insertelement <4 x i32> %1037, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc866 = insertelement <4 x i32> %1038, i32 131072, i64 3, !dbg !17
  %1039 = ptrtoint ptr addrspace(3) %1029 to i32, !dbg !17
  %dma.buffer.lds867 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1039), !dbg !17
  %1040 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds867, i32 %dma.buffer.offset865, <4 x i32> %dma.buffer.rsrc866) #10, !dbg !17
  %address.local635 = or disjoint i64 %address.local590, 768, !dbg !12
  %address.local636 = shl nuw nsw i64 %address.local635, 3, !dbg !12
  %address.local637 = and i64 %address.local636, 7680, !dbg !12
  %address.local638 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local637, !dbg !12
  %address.local639 = ptrtoint ptr addrspace(3) %address.local638 to i64, !dbg !12
  %address.local640 = trunc i64 %address.local639 to i32, !dbg !12
  %1041 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local640), !dbg !17
  %1042 = zext i32 %1041 to i64, !dbg !17
  %1043 = inttoptr i64 %1042 to ptr addrspace(3), !dbg !17
  %address.local641 = lshr i64 %address.local635, 3, !dbg !17
  %address.local642 = and i64 %address.local641, 126, !dbg !17
  %address.local643 = and i64 %address.local590, 1, !dbg !17
  %address.local644 = or disjoint i64 %address.local642, %address.local643, !dbg !17
  %address.local645 = shl nuw nsw i64 %address.local644, 7, !dbg !17
  %address.local646 = shl nuw nsw i64 %address.local590, 2, !dbg !17
  %address.local647 = and i64 %address.local646, 56, !dbg !17
  %address.local648 = or disjoint i64 %address.local645, %address.local647, !dbg !17
  %address.local649 = getelementptr [2 x i8], ptr addrspace(1) %998, i64 %address.local648, !dbg !17
  %1044 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1045 = ptrtoint ptr addrspace(1) %address.local649 to i64, !dbg !17
  %1046 = sub i64 %1045, %1044, !dbg !17
  %dma.buffer.offset862 = trunc i64 %1046 to i32, !dbg !17
  %1047 = trunc i64 %1044 to i32, !dbg !17
  %1048 = insertelement <4 x i32> poison, i32 %1047, i64 0, !dbg !17
  %1049 = lshr i64 %1044, 32, !dbg !17
  %1050 = trunc i64 %1049 to i32, !dbg !17
  %1051 = insertelement <4 x i32> %1048, i32 %1050, i64 1, !dbg !17
  %1052 = insertelement <4 x i32> %1051, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc863 = insertelement <4 x i32> %1052, i32 131072, i64 3, !dbg !17
  %1053 = ptrtoint ptr addrspace(3) %1043 to i32, !dbg !17
  %dma.buffer.lds864 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1053), !dbg !17
  %1054 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds864, i32 %dma.buffer.offset862, <4 x i32> %dma.buffer.rsrc863) #10, !dbg !17
  %1055 = shufflevector <8 x half> %962, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1056 = extractvalue [2 x <16 x float>] %27, 0, !dbg !30
  %1057 = shufflevector <16 x float> %1056, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1058 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1055, <4 x float> %1057) #11, !dbg !18
  %1059 = shufflevector <8 x half> %966, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1060 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1059, <4 x float> %1058) #11, !dbg !18
  %1061 = shufflevector <8 x half> %970, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1062 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1061, <4 x float> %1060) #11, !dbg !18
  %1063 = shufflevector <8 x half> %974, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1064 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1063, <4 x float> %1062) #11, !dbg !18
  %1065 = shufflevector <8 x half> %978, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1066 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1065, <4 x float> %1064) #11, !dbg !18
  %1067 = shufflevector <8 x half> %982, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1068 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1067, <4 x float> %1066) #11, !dbg !18
  %1069 = shufflevector <8 x half> %986, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1070 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1069, <4 x float> %1068) #11, !dbg !18
  %1071 = shufflevector <8 x half> %990, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1072 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1071, <4 x float> %1070) #11, !dbg !18
  %1073 = shufflevector <8 x half> %962, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1074 = shufflevector <16 x float> %1056, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1075 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1073, <4 x float> %1074) #11, !dbg !18
  %1076 = shufflevector <8 x half> %966, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1077 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1076, <4 x float> %1075) #11, !dbg !18
  %1078 = shufflevector <8 x half> %970, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1079 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1078, <4 x float> %1077) #11, !dbg !18
  %1080 = shufflevector <8 x half> %974, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1081 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1080, <4 x float> %1079) #11, !dbg !18
  %1082 = shufflevector <8 x half> %978, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1083 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1082, <4 x float> %1081) #11, !dbg !18
  %1084 = shufflevector <8 x half> %982, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1085 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1084, <4 x float> %1083) #11, !dbg !18
  %1086 = shufflevector <8 x half> %986, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1087 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1086, <4 x float> %1085) #11, !dbg !18
  %1088 = shufflevector <8 x half> %990, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1089 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1088, <4 x float> %1087) #11, !dbg !18
  %1090 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1091 = shufflevector <16 x float> %1056, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %1092 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1090, <4 x float> %1091) #11, !dbg !18
  %1093 = shufflevector <8 x half> %968, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1094 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1093, <4 x float> %1092) #11, !dbg !18
  %1095 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1096 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1095, <4 x float> %1094) #11, !dbg !18
  %1097 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1098 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1097, <4 x float> %1096) #11, !dbg !18
  %1099 = shufflevector <8 x half> %980, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1100 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1099, <4 x float> %1098) #11, !dbg !18
  %1101 = shufflevector <8 x half> %984, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1102 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1101, <4 x float> %1100) #11, !dbg !18
  %1103 = shufflevector <8 x half> %988, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1104 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1103, <4 x float> %1102) #11, !dbg !18
  %1105 = shufflevector <8 x half> %992, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1106 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1105, <4 x float> %1104) #11, !dbg !18
  %1107 = shufflevector <4 x float> %1106, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1108 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1109 = shufflevector <16 x float> %1056, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %1110 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1108, <4 x float> %1109) #11, !dbg !18
  %1111 = shufflevector <8 x half> %968, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1112 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1111, <4 x float> %1110) #11, !dbg !18
  %1113 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1114 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1113, <4 x float> %1112) #11, !dbg !18
  %1115 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1116 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1115, <4 x float> %1114) #11, !dbg !18
  %1117 = shufflevector <8 x half> %980, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1118 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1117, <4 x float> %1116) #11, !dbg !18
  %1119 = shufflevector <8 x half> %984, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1120 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1119, <4 x float> %1118) #11, !dbg !18
  %1121 = shufflevector <8 x half> %988, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1122 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1121, <4 x float> %1120) #11, !dbg !18
  %1123 = shufflevector <8 x half> %992, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1124 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1123, <4 x float> %1122) #11, !dbg !18
  %1125 = shufflevector <4 x float> %1124, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1126 = extractvalue [2 x <16 x float>] %27, 1, !dbg !30
  %1127 = shufflevector <16 x float> %1126, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1128 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1055, <4 x float> %1127) #11, !dbg !18
  %1129 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1059, <4 x float> %1128) #11, !dbg !18
  %1130 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1061, <4 x float> %1129) #11, !dbg !18
  %1131 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1063, <4 x float> %1130) #11, !dbg !18
  %1132 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1065, <4 x float> %1131) #11, !dbg !18
  %1133 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1067, <4 x float> %1132) #11, !dbg !18
  %1134 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed142, <4 x half> %1069, <4 x float> %1133) #11, !dbg !18
  %1135 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed144, <4 x half> %1071, <4 x float> %1134) #11, !dbg !18
  %1136 = shufflevector <16 x float> %1126, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1137 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1073, <4 x float> %1136) #11, !dbg !18
  %1138 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1076, <4 x float> %1137) #11, !dbg !18
  %1139 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1078, <4 x float> %1138) #11, !dbg !18
  %1140 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1080, <4 x float> %1139) #11, !dbg !18
  %1141 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1082, <4 x float> %1140) #11, !dbg !18
  %1142 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1084, <4 x float> %1141) #11, !dbg !18
  %1143 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed142, <4 x half> %1086, <4 x float> %1142) #11, !dbg !18
  %1144 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed144, <4 x half> %1088, <4 x float> %1143) #11, !dbg !18
  %1145 = shufflevector <16 x float> %1126, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %1146 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1090, <4 x float> %1145) #11, !dbg !18
  %1147 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1093, <4 x float> %1146) #11, !dbg !18
  %1148 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1095, <4 x float> %1147) #11, !dbg !18
  %1149 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1097, <4 x float> %1148) #11, !dbg !18
  %1150 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1099, <4 x float> %1149) #11, !dbg !18
  %1151 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1101, <4 x float> %1150) #11, !dbg !18
  %1152 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed142, <4 x half> %1103, <4 x float> %1151) #11, !dbg !18
  %1153 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed144, <4 x half> %1105, <4 x float> %1152) #11, !dbg !18
  %1154 = shufflevector <4 x float> %1153, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1155 = shufflevector <16 x float> %1126, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %1156 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1108, <4 x float> %1155) #11, !dbg !18
  %1157 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1111, <4 x float> %1156) #11, !dbg !18
  %1158 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1113, <4 x float> %1157) #11, !dbg !18
  %1159 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1115, <4 x float> %1158) #11, !dbg !18
  %1160 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1117, <4 x float> %1159) #11, !dbg !18
  %1161 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1119, <4 x float> %1160) #11, !dbg !18
  %1162 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed142, <4 x half> %1121, <4 x float> %1161) #11, !dbg !18
  %1163 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed144, <4 x half> %1123, <4 x float> %1162) #11, !dbg !18
  %1164 = shufflevector <4 x float> %1163, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1165 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local502, i32 0) #0, !dbg !18
  %1166 = bitcast <4 x i32> %1165 to <8 x half>, !dbg !18
  %1167 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local504, i32 0) #0, !dbg !18
  %1168 = bitcast <4 x i32> %1167 to <8 x half>, !dbg !18
  %1169 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local515, i32 0) #0, !dbg !18
  %1170 = bitcast <4 x i32> %1169 to <8 x half>, !dbg !18
  %1171 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local517, i32 0) #0, !dbg !18
  %1172 = bitcast <4 x i32> %1171 to <8 x half>, !dbg !18
  %1173 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local528, i32 0) #0, !dbg !18
  %1174 = bitcast <4 x i32> %1173 to <8 x half>, !dbg !18
  %1175 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local530, i32 0) #0, !dbg !18
  %1176 = bitcast <4 x i32> %1175 to <8 x half>, !dbg !18
  %1177 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local541, i32 0) #0, !dbg !18
  %1178 = bitcast <4 x i32> %1177 to <8 x half>, !dbg !18
  %1179 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local543, i32 0) #0, !dbg !18
  %1180 = bitcast <4 x i32> %1179 to <8 x half>, !dbg !18
  %1181 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local545, i32 0) #0, !dbg !18
  %1182 = bitcast <4 x i32> %1181 to <8 x half>, !dbg !18
  %1183 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local547, i32 0) #0, !dbg !18
  %1184 = bitcast <4 x i32> %1183 to <8 x half>, !dbg !18
  %1185 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local559, i32 0) #0, !dbg !18
  %1186 = bitcast <4 x i32> %1185 to <8 x half>, !dbg !18
  %1187 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local561, i32 0) #0, !dbg !18
  %1188 = bitcast <4 x i32> %1187 to <8 x half>, !dbg !18
  %1189 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local573, i32 0) #0, !dbg !18
  %1190 = bitcast <4 x i32> %1189 to <8 x half>, !dbg !18
  %1191 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local575, i32 0) #0, !dbg !18
  %1192 = bitcast <4 x i32> %1191 to <8 x half>, !dbg !18
  %1193 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local587, i32 0) #0, !dbg !18
  %1194 = bitcast <4 x i32> %1193 to <8 x half>, !dbg !18
  %1195 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local589, i32 0) #0, !dbg !18
  %1196 = bitcast <4 x i32> %1195 to <8 x half>, !dbg !18
  %1197 = shufflevector <8 x half> %1166, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1198 = extractvalue [2 x <16 x float>] %29, 0, !dbg !30
  %1199 = shufflevector <16 x float> %1198, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1200 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1197, <4 x float> %1199) #11, !dbg !18
  %1201 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1202 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1201, <4 x float> %1200) #11, !dbg !18
  %1203 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1204 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1203, <4 x float> %1202) #11, !dbg !18
  %1205 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1206 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1205, <4 x float> %1204) #11, !dbg !18
  %1207 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1208 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1207, <4 x float> %1206) #11, !dbg !18
  %1209 = shufflevector <8 x half> %1186, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1210 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1209, <4 x float> %1208) #11, !dbg !18
  %1211 = shufflevector <8 x half> %1190, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1212 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1211, <4 x float> %1210) #11, !dbg !18
  %1213 = shufflevector <8 x half> %1194, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1214 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1213, <4 x float> %1212) #11, !dbg !18
  %1215 = shufflevector <8 x half> %1166, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1216 = shufflevector <16 x float> %1198, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1217 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1215, <4 x float> %1216) #11, !dbg !18
  %1218 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1219 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1218, <4 x float> %1217) #11, !dbg !18
  %1220 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1221 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1220, <4 x float> %1219) #11, !dbg !18
  %1222 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1223 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1222, <4 x float> %1221) #11, !dbg !18
  %1224 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1225 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1224, <4 x float> %1223) #11, !dbg !18
  %1226 = shufflevector <8 x half> %1186, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1227 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1226, <4 x float> %1225) #11, !dbg !18
  %1228 = shufflevector <8 x half> %1190, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1229 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1228, <4 x float> %1227) #11, !dbg !18
  %1230 = shufflevector <8 x half> %1194, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1231 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1230, <4 x float> %1229) #11, !dbg !18
  %1232 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1233 = shufflevector <16 x float> %1198, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %1234 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1232, <4 x float> %1233) #11, !dbg !18
  %1235 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1236 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1235, <4 x float> %1234) #11, !dbg !18
  %1237 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1238 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1237, <4 x float> %1236) #11, !dbg !18
  %1239 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1240 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1239, <4 x float> %1238) #11, !dbg !18
  %1241 = shufflevector <8 x half> %1184, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1242 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1241, <4 x float> %1240) #11, !dbg !18
  %1243 = shufflevector <8 x half> %1188, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1244 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1243, <4 x float> %1242) #11, !dbg !18
  %1245 = shufflevector <8 x half> %1192, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1246 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1245, <4 x float> %1244) #11, !dbg !18
  %1247 = shufflevector <8 x half> %1196, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1248 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1247, <4 x float> %1246) #11, !dbg !18
  %1249 = shufflevector <4 x float> %1248, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1250 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1251 = shufflevector <16 x float> %1198, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %1252 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1250, <4 x float> %1251) #11, !dbg !18
  %1253 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1254 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1253, <4 x float> %1252) #11, !dbg !18
  %1255 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1256 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1255, <4 x float> %1254) #11, !dbg !18
  %1257 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1258 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1257, <4 x float> %1256) #11, !dbg !18
  %1259 = shufflevector <8 x half> %1184, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1260 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1259, <4 x float> %1258) #11, !dbg !18
  %1261 = shufflevector <8 x half> %1188, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1262 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1261, <4 x float> %1260) #11, !dbg !18
  %1263 = shufflevector <8 x half> %1192, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1264 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1263, <4 x float> %1262) #11, !dbg !18
  %1265 = shufflevector <8 x half> %1196, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1266 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1265, <4 x float> %1264) #11, !dbg !18
  %1267 = shufflevector <4 x float> %1266, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1268 = extractvalue [2 x <16 x float>] %29, 1, !dbg !30
  %1269 = shufflevector <16 x float> %1268, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1270 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1197, <4 x float> %1269) #11, !dbg !18
  %1271 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1201, <4 x float> %1270) #11, !dbg !18
  %1272 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1203, <4 x float> %1271) #11, !dbg !18
  %1273 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1205, <4 x float> %1272) #11, !dbg !18
  %1274 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1207, <4 x float> %1273) #11, !dbg !18
  %1275 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1209, <4 x float> %1274) #11, !dbg !18
  %1276 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed142, <4 x half> %1211, <4 x float> %1275) #11, !dbg !18
  %1277 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed144, <4 x half> %1213, <4 x float> %1276) #11, !dbg !18
  %1278 = shufflevector <16 x float> %1268, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1279 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1215, <4 x float> %1278) #11, !dbg !18
  %1280 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1218, <4 x float> %1279) #11, !dbg !18
  %1281 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1220, <4 x float> %1280) #11, !dbg !18
  %1282 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1222, <4 x float> %1281) #11, !dbg !18
  %1283 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1224, <4 x float> %1282) #11, !dbg !18
  %1284 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1226, <4 x float> %1283) #11, !dbg !18
  %1285 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed142, <4 x half> %1228, <4 x float> %1284) #11, !dbg !18
  %1286 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed144, <4 x half> %1230, <4 x float> %1285) #11, !dbg !18
  %1287 = shufflevector <16 x float> %1268, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %1288 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1232, <4 x float> %1287) #11, !dbg !18
  %1289 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1235, <4 x float> %1288) #11, !dbg !18
  %1290 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1237, <4 x float> %1289) #11, !dbg !18
  %1291 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1239, <4 x float> %1290) #11, !dbg !18
  %1292 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1241, <4 x float> %1291) #11, !dbg !18
  %1293 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1243, <4 x float> %1292) #11, !dbg !18
  %1294 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed142, <4 x half> %1245, <4 x float> %1293) #11, !dbg !18
  %1295 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed144, <4 x half> %1247, <4 x float> %1294) #11, !dbg !18
  %1296 = shufflevector <4 x float> %1295, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1297 = shufflevector <16 x float> %1268, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %1298 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1250, <4 x float> %1297) #11, !dbg !18
  %1299 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1253, <4 x float> %1298) #11, !dbg !18
  %1300 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1255, <4 x float> %1299) #11, !dbg !18
  %1301 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1257, <4 x float> %1300) #11, !dbg !18
  %1302 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1259, <4 x float> %1301) #11, !dbg !18
  %1303 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1261, <4 x float> %1302) #11, !dbg !18
  %1304 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed142, <4 x half> %1263, <4 x float> %1303) #11, !dbg !18
  %1305 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed144, <4 x half> %1265, <4 x float> %1304) #11, !dbg !18
  %1306 = shufflevector <4 x float> %1305, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1307 = shufflevector <4 x float> %1072, <4 x float> %1089, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1308 = shufflevector <16 x float> %1307, <16 x float> %1107, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1309 = shufflevector <16 x float> %1308, <16 x float> %1125, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !15
  %1310 = insertvalue [2 x <16 x float>] poison, <16 x float> %1309, 0, !dbg !15
  %1311 = shufflevector <4 x float> %1135, <4 x float> %1144, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1312 = shufflevector <16 x float> %1311, <16 x float> %1154, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1313 = shufflevector <16 x float> %1312, <16 x float> %1164, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !15
  %1314 = insertvalue [2 x <16 x float>] %1310, <16 x float> %1313, 1, !dbg !15
  %1315 = shufflevector <4 x float> %1214, <4 x float> %1231, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1316 = shufflevector <16 x float> %1315, <16 x float> %1249, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1317 = shufflevector <16 x float> %1316, <16 x float> %1267, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !15
  %1318 = insertvalue [2 x <16 x float>] poison, <16 x float> %1317, 0, !dbg !15
  %1319 = shufflevector <4 x float> %1277, <4 x float> %1286, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1320 = shufflevector <16 x float> %1319, <16 x float> %1296, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !15
  %1321 = shufflevector <16 x float> %1320, <16 x float> %1306, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !15
  %1322 = insertvalue [2 x <16 x float>] %1318, <16 x float> %1321, 1, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1323 = add nuw nsw i64 %26, 1, !dbg !15
  br label %25, !dbg !15

1324:                                             ; preds = %25
  %1325 = and i32 %19, 16, !dbg !15
  %.not2 = icmp eq i32 %1325, 0, !dbg !15
  %1326 = and i32 %19, 32, !dbg !15
  %.not3 = icmp eq i32 %1326, 0, !dbg !15
  %1327 = extractvalue [2 x <16 x float>] %29, 0, !dbg !15
  %1328 = extractelement <16 x float> %1327, i64 0, !dbg !15
  %1329 = extractelement <16 x float> %1327, i64 1, !dbg !15
  %1330 = extractelement <16 x float> %1327, i64 2, !dbg !15
  %1331 = extractelement <16 x float> %1327, i64 3, !dbg !15
  %1332 = select i1 %.not2, float %1329, float %1328, !dbg !15
  %1333 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !15
  %1334 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1333), !dbg !15
  %1335 = add nuw nsw i32 %1334, 64, !dbg !15
  %1336 = xor i32 %1334, 16, !dbg !15
  %1337 = and i32 %1335, 192, !dbg !15
  %1338 = icmp samesign ult i32 %1336, %1337, !dbg !15
  %1339 = select i1 %1338, i32 %1336, i32 %1334, !dbg !15
  %1340 = shl nuw nsw i32 %1339, 2, !dbg !15
  %1341 = bitcast float %1332 to i32, !dbg !15
  %1342 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1341), !dbg !15
  %1343 = bitcast i32 %1342 to float, !dbg !15
  %1344 = select i1 %.not2, float %1328, float %1343, !dbg !15
  %1345 = select i1 %.not2, float %1343, float %1329, !dbg !15
  %1346 = select i1 %.not2, float %1331, float %1330, !dbg !15
  %1347 = bitcast float %1346 to i32, !dbg !15
  %1348 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1347), !dbg !15
  %1349 = bitcast i32 %1348 to float, !dbg !15
  %1350 = select i1 %.not2, float %1330, float %1349, !dbg !15
  %1351 = select i1 %.not2, float %1349, float %1331, !dbg !15
  %1352 = select i1 %.not3, float %1350, float %1344, !dbg !15
  %1353 = xor i32 %1334, 32, !dbg !15
  %1354 = icmp samesign ult i32 %1353, %1337, !dbg !15
  %1355 = select i1 %1354, i32 %1353, i32 %1334, !dbg !15
  %1356 = shl nuw nsw i32 %1355, 2, !dbg !15
  %1357 = bitcast float %1352 to i32, !dbg !15
  %1358 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1357), !dbg !15
  %1359 = bitcast i32 %1358 to float, !dbg !15
  %1360 = select i1 %.not3, float %1344, float %1359, !dbg !15
  %1361 = select i1 %.not3, float %1359, float %1350, !dbg !15
  %1362 = select i1 %.not3, float %1351, float %1345, !dbg !15
  %1363 = bitcast float %1362 to i32, !dbg !15
  %1364 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1363), !dbg !15
  %1365 = bitcast i32 %1364 to float, !dbg !15
  %1366 = select i1 %.not3, float %1345, float %1365, !dbg !15
  %1367 = select i1 %.not3, float %1365, float %1351, !dbg !15
  %1368 = extractvalue [2 x <16 x float>] %27, 0, !dbg !15
  %1369 = extractelement <16 x float> %1368, i64 0, !dbg !15
  %1370 = extractelement <16 x float> %1368, i64 1, !dbg !15
  %1371 = extractelement <16 x float> %1368, i64 2, !dbg !15
  %1372 = extractelement <16 x float> %1368, i64 3, !dbg !15
  %1373 = select i1 %.not2, float %1370, float %1369, !dbg !15
  %1374 = bitcast float %1373 to i32, !dbg !15
  %1375 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1374), !dbg !15
  %1376 = bitcast i32 %1375 to float, !dbg !15
  %1377 = select i1 %.not2, float %1369, float %1376, !dbg !15
  %1378 = select i1 %.not2, float %1376, float %1370, !dbg !15
  %1379 = select i1 %.not2, float %1372, float %1371, !dbg !15
  %1380 = bitcast float %1379 to i32, !dbg !15
  %1381 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1380), !dbg !15
  %1382 = bitcast i32 %1381 to float, !dbg !15
  %1383 = select i1 %.not2, float %1371, float %1382, !dbg !15
  %1384 = select i1 %.not2, float %1382, float %1372, !dbg !15
  %1385 = select i1 %.not3, float %1383, float %1377, !dbg !15
  %1386 = bitcast float %1385 to i32, !dbg !15
  %1387 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1386), !dbg !15
  %1388 = select i1 %.not3, float %1384, float %1378, !dbg !15
  %1389 = bitcast float %1388 to i32, !dbg !15
  %1390 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1389), !dbg !15
  %1391 = extractvalue [2 x <1 x float>] %28, 0, !dbg !19
  %1392 = shufflevector <1 x float> %1391, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1393 = extractelement <16 x float> %1327, i64 4, !dbg !15
  %1394 = extractelement <16 x float> %1327, i64 5, !dbg !15
  %1395 = extractelement <16 x float> %1327, i64 6, !dbg !15
  %1396 = extractelement <16 x float> %1327, i64 7, !dbg !15
  %1397 = select i1 %.not2, float %1394, float %1393, !dbg !15
  %1398 = bitcast float %1397 to i32, !dbg !15
  %1399 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1398), !dbg !15
  %1400 = bitcast i32 %1399 to float, !dbg !15
  %1401 = select i1 %.not2, float %1393, float %1400, !dbg !15
  %1402 = select i1 %.not2, float %1400, float %1394, !dbg !15
  %1403 = select i1 %.not2, float %1396, float %1395, !dbg !15
  %1404 = bitcast float %1403 to i32, !dbg !15
  %1405 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1404), !dbg !15
  %1406 = bitcast i32 %1405 to float, !dbg !15
  %1407 = select i1 %.not2, float %1395, float %1406, !dbg !15
  %1408 = select i1 %.not2, float %1406, float %1396, !dbg !15
  %1409 = select i1 %.not3, float %1407, float %1401, !dbg !15
  %1410 = bitcast float %1409 to i32, !dbg !15
  %1411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1410), !dbg !15
  %1412 = bitcast i32 %1411 to float, !dbg !15
  %1413 = select i1 %.not3, float %1401, float %1412, !dbg !15
  %1414 = select i1 %.not3, float %1412, float %1407, !dbg !15
  %1415 = select i1 %.not3, float %1408, float %1402, !dbg !15
  %1416 = bitcast float %1415 to i32, !dbg !15
  %1417 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1416), !dbg !15
  %1418 = bitcast i32 %1417 to float, !dbg !15
  %1419 = select i1 %.not3, float %1402, float %1418, !dbg !15
  %1420 = select i1 %.not3, float %1418, float %1408, !dbg !15
  %1421 = extractelement <16 x float> %1368, i64 4, !dbg !15
  %1422 = extractelement <16 x float> %1368, i64 5, !dbg !15
  %1423 = extractelement <16 x float> %1368, i64 6, !dbg !15
  %1424 = extractelement <16 x float> %1368, i64 7, !dbg !15
  %1425 = select i1 %.not2, float %1422, float %1421, !dbg !15
  %1426 = bitcast float %1425 to i32, !dbg !15
  %1427 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1426), !dbg !15
  %1428 = bitcast i32 %1427 to float, !dbg !15
  %1429 = select i1 %.not2, float %1421, float %1428, !dbg !15
  %1430 = select i1 %.not2, float %1428, float %1422, !dbg !15
  %1431 = select i1 %.not2, float %1424, float %1423, !dbg !15
  %1432 = bitcast float %1431 to i32, !dbg !15
  %1433 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1432), !dbg !15
  %1434 = bitcast i32 %1433 to float, !dbg !15
  %1435 = select i1 %.not2, float %1423, float %1434, !dbg !15
  %1436 = select i1 %.not2, float %1434, float %1424, !dbg !15
  %1437 = select i1 %.not3, float %1435, float %1429, !dbg !15
  %1438 = bitcast float %1437 to i32, !dbg !15
  %1439 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1438), !dbg !15
  %1440 = select i1 %.not3, float %1436, float %1430, !dbg !15
  %1441 = bitcast float %1440 to i32, !dbg !15
  %1442 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1441), !dbg !15
  %1443 = extractelement <16 x float> %1327, i64 8, !dbg !15
  %1444 = extractelement <16 x float> %1327, i64 9, !dbg !15
  %1445 = extractelement <16 x float> %1327, i64 10, !dbg !15
  %1446 = extractelement <16 x float> %1327, i64 11, !dbg !15
  %1447 = select i1 %.not2, float %1444, float %1443, !dbg !15
  %1448 = bitcast float %1447 to i32, !dbg !15
  %1449 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1448), !dbg !15
  %1450 = bitcast i32 %1449 to float, !dbg !15
  %1451 = select i1 %.not2, float %1443, float %1450, !dbg !15
  %1452 = select i1 %.not2, float %1450, float %1444, !dbg !15
  %1453 = select i1 %.not2, float %1446, float %1445, !dbg !15
  %1454 = bitcast float %1453 to i32, !dbg !15
  %1455 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1454), !dbg !15
  %1456 = bitcast i32 %1455 to float, !dbg !15
  %1457 = select i1 %.not2, float %1445, float %1456, !dbg !15
  %1458 = select i1 %.not2, float %1456, float %1446, !dbg !15
  %1459 = select i1 %.not3, float %1457, float %1451, !dbg !15
  %1460 = bitcast float %1459 to i32, !dbg !15
  %1461 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1460), !dbg !15
  %1462 = bitcast i32 %1461 to float, !dbg !15
  %1463 = select i1 %.not3, float %1451, float %1462, !dbg !15
  %1464 = select i1 %.not3, float %1462, float %1457, !dbg !15
  %1465 = select i1 %.not3, float %1458, float %1452, !dbg !15
  %1466 = bitcast float %1465 to i32, !dbg !15
  %1467 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1466), !dbg !15
  %1468 = bitcast i32 %1467 to float, !dbg !15
  %1469 = select i1 %.not3, float %1452, float %1468, !dbg !15
  %1470 = select i1 %.not3, float %1468, float %1458, !dbg !15
  %1471 = extractelement <16 x float> %1368, i64 8, !dbg !15
  %1472 = extractelement <16 x float> %1368, i64 9, !dbg !15
  %1473 = extractelement <16 x float> %1368, i64 10, !dbg !15
  %1474 = extractelement <16 x float> %1368, i64 11, !dbg !15
  %1475 = select i1 %.not2, float %1472, float %1471, !dbg !15
  %1476 = bitcast float %1475 to i32, !dbg !15
  %1477 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1476), !dbg !15
  %1478 = bitcast i32 %1477 to float, !dbg !15
  %1479 = select i1 %.not2, float %1471, float %1478, !dbg !15
  %1480 = select i1 %.not2, float %1478, float %1472, !dbg !15
  %1481 = select i1 %.not2, float %1474, float %1473, !dbg !15
  %1482 = bitcast float %1481 to i32, !dbg !15
  %1483 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1482), !dbg !15
  %1484 = bitcast i32 %1483 to float, !dbg !15
  %1485 = select i1 %.not2, float %1473, float %1484, !dbg !15
  %1486 = select i1 %.not2, float %1484, float %1474, !dbg !15
  %1487 = select i1 %.not3, float %1485, float %1479, !dbg !15
  %1488 = bitcast float %1487 to i32, !dbg !15
  %1489 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1488), !dbg !15
  %1490 = select i1 %.not3, float %1486, float %1480, !dbg !15
  %1491 = bitcast float %1490 to i32, !dbg !15
  %1492 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1491), !dbg !15
  %1493 = extractelement <16 x float> %1327, i64 12, !dbg !15
  %1494 = extractelement <16 x float> %1327, i64 13, !dbg !15
  %1495 = extractelement <16 x float> %1327, i64 14, !dbg !15
  %1496 = extractelement <16 x float> %1327, i64 15, !dbg !15
  %1497 = select i1 %.not2, float %1494, float %1493, !dbg !15
  %1498 = bitcast float %1497 to i32, !dbg !15
  %1499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1498), !dbg !15
  %1500 = bitcast i32 %1499 to float, !dbg !15
  %1501 = select i1 %.not2, float %1493, float %1500, !dbg !15
  %1502 = select i1 %.not2, float %1500, float %1494, !dbg !15
  %1503 = select i1 %.not2, float %1496, float %1495, !dbg !15
  %1504 = bitcast float %1503 to i32, !dbg !15
  %1505 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1504), !dbg !15
  %1506 = bitcast i32 %1505 to float, !dbg !15
  %1507 = select i1 %.not2, float %1495, float %1506, !dbg !15
  %1508 = select i1 %.not2, float %1506, float %1496, !dbg !15
  %1509 = select i1 %.not3, float %1507, float %1501, !dbg !15
  %1510 = bitcast float %1509 to i32, !dbg !15
  %1511 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1510), !dbg !15
  %1512 = bitcast i32 %1511 to float, !dbg !15
  %1513 = select i1 %.not3, float %1501, float %1512, !dbg !15
  %1514 = select i1 %.not3, float %1512, float %1507, !dbg !15
  %1515 = select i1 %.not3, float %1508, float %1502, !dbg !15
  %1516 = bitcast float %1515 to i32, !dbg !15
  %1517 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1516), !dbg !15
  %1518 = bitcast i32 %1517 to float, !dbg !15
  %1519 = select i1 %.not3, float %1502, float %1518, !dbg !15
  %1520 = select i1 %.not3, float %1518, float %1508, !dbg !15
  %1521 = extractelement <16 x float> %1368, i64 12, !dbg !15
  %1522 = extractelement <16 x float> %1368, i64 13, !dbg !15
  %1523 = extractelement <16 x float> %1368, i64 14, !dbg !15
  %1524 = extractelement <16 x float> %1368, i64 15, !dbg !15
  %1525 = select i1 %.not2, float %1522, float %1521, !dbg !15
  %1526 = bitcast float %1525 to i32, !dbg !15
  %1527 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1526), !dbg !15
  %1528 = bitcast i32 %1527 to float, !dbg !15
  %1529 = select i1 %.not2, float %1521, float %1528, !dbg !15
  %1530 = select i1 %.not2, float %1528, float %1522, !dbg !15
  %1531 = select i1 %.not2, float %1524, float %1523, !dbg !15
  %1532 = bitcast float %1531 to i32, !dbg !15
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1532), !dbg !15
  %1534 = bitcast i32 %1533 to float, !dbg !15
  %1535 = select i1 %.not2, float %1523, float %1534, !dbg !15
  %1536 = select i1 %.not2, float %1534, float %1524, !dbg !15
  %1537 = select i1 %.not3, float %1535, float %1529, !dbg !15
  %1538 = bitcast float %1537 to i32, !dbg !15
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1538), !dbg !15
  %1540 = select i1 %.not3, float %1536, float %1530, !dbg !15
  %1541 = bitcast float %1540 to i32, !dbg !15
  %1542 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1541), !dbg !15
  %1543 = extractvalue [2 x <16 x float>] %29, 1, !dbg !15
  %1544 = extractelement <16 x float> %1543, i64 0, !dbg !15
  %1545 = extractelement <16 x float> %1543, i64 1, !dbg !15
  %1546 = extractelement <16 x float> %1543, i64 2, !dbg !15
  %1547 = extractelement <16 x float> %1543, i64 3, !dbg !15
  %1548 = select i1 %.not2, float %1545, float %1544, !dbg !15
  %1549 = bitcast float %1548 to i32, !dbg !15
  %1550 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1549), !dbg !15
  %1551 = bitcast i32 %1550 to float, !dbg !15
  %1552 = select i1 %.not2, float %1544, float %1551, !dbg !15
  %1553 = select i1 %.not2, float %1551, float %1545, !dbg !15
  %1554 = select i1 %.not2, float %1547, float %1546, !dbg !15
  %1555 = bitcast float %1554 to i32, !dbg !15
  %1556 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1555), !dbg !15
  %1557 = bitcast i32 %1556 to float, !dbg !15
  %1558 = select i1 %.not2, float %1546, float %1557, !dbg !15
  %1559 = select i1 %.not2, float %1557, float %1547, !dbg !15
  %1560 = select i1 %.not3, float %1558, float %1552, !dbg !15
  %1561 = bitcast float %1560 to i32, !dbg !15
  %1562 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1561), !dbg !15
  %1563 = bitcast i32 %1562 to float, !dbg !15
  %1564 = select i1 %.not3, float %1552, float %1563, !dbg !15
  %1565 = select i1 %.not3, float %1563, float %1558, !dbg !15
  %1566 = select i1 %.not3, float %1559, float %1553, !dbg !15
  %1567 = bitcast float %1566 to i32, !dbg !15
  %1568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1567), !dbg !15
  %1569 = bitcast i32 %1568 to float, !dbg !15
  %1570 = select i1 %.not3, float %1553, float %1569, !dbg !15
  %1571 = select i1 %.not3, float %1569, float %1559, !dbg !15
  %1572 = extractvalue [2 x <16 x float>] %27, 1, !dbg !15
  %1573 = extractelement <16 x float> %1572, i64 0, !dbg !15
  %1574 = extractelement <16 x float> %1572, i64 1, !dbg !15
  %1575 = extractelement <16 x float> %1572, i64 2, !dbg !15
  %1576 = extractelement <16 x float> %1572, i64 3, !dbg !15
  %1577 = select i1 %.not2, float %1574, float %1573, !dbg !15
  %1578 = bitcast float %1577 to i32, !dbg !15
  %1579 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1578), !dbg !15
  %1580 = bitcast i32 %1579 to float, !dbg !15
  %1581 = select i1 %.not2, float %1573, float %1580, !dbg !15
  %1582 = select i1 %.not2, float %1580, float %1574, !dbg !15
  %1583 = select i1 %.not2, float %1576, float %1575, !dbg !15
  %1584 = bitcast float %1583 to i32, !dbg !15
  %1585 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1584), !dbg !15
  %1586 = bitcast i32 %1585 to float, !dbg !15
  %1587 = select i1 %.not2, float %1575, float %1586, !dbg !15
  %1588 = select i1 %.not2, float %1586, float %1576, !dbg !15
  %1589 = select i1 %.not3, float %1587, float %1581, !dbg !15
  %1590 = bitcast float %1589 to i32, !dbg !15
  %1591 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1590), !dbg !15
  %1592 = select i1 %.not3, float %1588, float %1582, !dbg !15
  %1593 = bitcast float %1592 to i32, !dbg !15
  %1594 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1593), !dbg !15
  %1595 = extractvalue [2 x <1 x float>] %28, 1, !dbg !19
  %1596 = shufflevector <1 x float> %1595, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1597 = extractelement <16 x float> %1543, i64 4, !dbg !15
  %1598 = extractelement <16 x float> %1543, i64 5, !dbg !15
  %1599 = extractelement <16 x float> %1543, i64 6, !dbg !15
  %1600 = extractelement <16 x float> %1543, i64 7, !dbg !15
  %1601 = select i1 %.not2, float %1598, float %1597, !dbg !15
  %1602 = bitcast float %1601 to i32, !dbg !15
  %1603 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1602), !dbg !15
  %1604 = bitcast i32 %1603 to float, !dbg !15
  %1605 = select i1 %.not2, float %1597, float %1604, !dbg !15
  %1606 = select i1 %.not2, float %1604, float %1598, !dbg !15
  %1607 = select i1 %.not2, float %1600, float %1599, !dbg !15
  %1608 = bitcast float %1607 to i32, !dbg !15
  %1609 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1608), !dbg !15
  %1610 = bitcast i32 %1609 to float, !dbg !15
  %1611 = select i1 %.not2, float %1599, float %1610, !dbg !15
  %1612 = select i1 %.not2, float %1610, float %1600, !dbg !15
  %1613 = select i1 %.not3, float %1611, float %1605, !dbg !15
  %1614 = bitcast float %1613 to i32, !dbg !15
  %1615 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1614), !dbg !15
  %1616 = bitcast i32 %1615 to float, !dbg !15
  %1617 = select i1 %.not3, float %1605, float %1616, !dbg !15
  %1618 = select i1 %.not3, float %1616, float %1611, !dbg !15
  %1619 = select i1 %.not3, float %1612, float %1606, !dbg !15
  %1620 = bitcast float %1619 to i32, !dbg !15
  %1621 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1620), !dbg !15
  %1622 = bitcast i32 %1621 to float, !dbg !15
  %1623 = select i1 %.not3, float %1606, float %1622, !dbg !15
  %1624 = select i1 %.not3, float %1622, float %1612, !dbg !15
  %1625 = extractelement <16 x float> %1572, i64 4, !dbg !15
  %1626 = extractelement <16 x float> %1572, i64 5, !dbg !15
  %1627 = extractelement <16 x float> %1572, i64 6, !dbg !15
  %1628 = extractelement <16 x float> %1572, i64 7, !dbg !15
  %1629 = select i1 %.not2, float %1626, float %1625, !dbg !15
  %1630 = bitcast float %1629 to i32, !dbg !15
  %1631 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1630), !dbg !15
  %1632 = bitcast i32 %1631 to float, !dbg !15
  %1633 = select i1 %.not2, float %1625, float %1632, !dbg !15
  %1634 = select i1 %.not2, float %1632, float %1626, !dbg !15
  %1635 = select i1 %.not2, float %1628, float %1627, !dbg !15
  %1636 = bitcast float %1635 to i32, !dbg !15
  %1637 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1636), !dbg !15
  %1638 = bitcast i32 %1637 to float, !dbg !15
  %1639 = select i1 %.not2, float %1627, float %1638, !dbg !15
  %1640 = select i1 %.not2, float %1638, float %1628, !dbg !15
  %1641 = select i1 %.not3, float %1639, float %1633, !dbg !15
  %1642 = bitcast float %1641 to i32, !dbg !15
  %1643 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1642), !dbg !15
  %1644 = select i1 %.not3, float %1640, float %1634, !dbg !15
  %1645 = bitcast float %1644 to i32, !dbg !15
  %1646 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1645), !dbg !15
  %1647 = extractelement <16 x float> %1543, i64 8, !dbg !15
  %1648 = extractelement <16 x float> %1543, i64 9, !dbg !15
  %1649 = extractelement <16 x float> %1543, i64 10, !dbg !15
  %1650 = extractelement <16 x float> %1543, i64 11, !dbg !15
  %1651 = select i1 %.not2, float %1648, float %1647, !dbg !15
  %1652 = bitcast float %1651 to i32, !dbg !15
  %1653 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1652), !dbg !15
  %1654 = bitcast i32 %1653 to float, !dbg !15
  %1655 = select i1 %.not2, float %1647, float %1654, !dbg !15
  %1656 = select i1 %.not2, float %1654, float %1648, !dbg !15
  %1657 = select i1 %.not2, float %1650, float %1649, !dbg !15
  %1658 = bitcast float %1657 to i32, !dbg !15
  %1659 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1658), !dbg !15
  %1660 = bitcast i32 %1659 to float, !dbg !15
  %1661 = select i1 %.not2, float %1649, float %1660, !dbg !15
  %1662 = select i1 %.not2, float %1660, float %1650, !dbg !15
  %1663 = select i1 %.not3, float %1661, float %1655, !dbg !15
  %1664 = bitcast float %1663 to i32, !dbg !15
  %1665 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1664), !dbg !15
  %1666 = bitcast i32 %1665 to float, !dbg !15
  %1667 = select i1 %.not3, float %1655, float %1666, !dbg !15
  %1668 = select i1 %.not3, float %1666, float %1661, !dbg !15
  %1669 = select i1 %.not3, float %1662, float %1656, !dbg !15
  %1670 = bitcast float %1669 to i32, !dbg !15
  %1671 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1670), !dbg !15
  %1672 = bitcast i32 %1671 to float, !dbg !15
  %1673 = select i1 %.not3, float %1656, float %1672, !dbg !15
  %1674 = select i1 %.not3, float %1672, float %1662, !dbg !15
  %1675 = extractelement <16 x float> %1572, i64 8, !dbg !15
  %1676 = extractelement <16 x float> %1572, i64 9, !dbg !15
  %1677 = extractelement <16 x float> %1572, i64 10, !dbg !15
  %1678 = extractelement <16 x float> %1572, i64 11, !dbg !15
  %1679 = select i1 %.not2, float %1676, float %1675, !dbg !15
  %1680 = bitcast float %1679 to i32, !dbg !15
  %1681 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1680), !dbg !15
  %1682 = bitcast i32 %1681 to float, !dbg !15
  %1683 = select i1 %.not2, float %1675, float %1682, !dbg !15
  %1684 = select i1 %.not2, float %1682, float %1676, !dbg !15
  %1685 = select i1 %.not2, float %1678, float %1677, !dbg !15
  %1686 = bitcast float %1685 to i32, !dbg !15
  %1687 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1686), !dbg !15
  %1688 = bitcast i32 %1687 to float, !dbg !15
  %1689 = select i1 %.not2, float %1677, float %1688, !dbg !15
  %1690 = select i1 %.not2, float %1688, float %1678, !dbg !15
  %1691 = select i1 %.not3, float %1689, float %1683, !dbg !15
  %1692 = bitcast float %1691 to i32, !dbg !15
  %1693 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1692), !dbg !15
  %1694 = select i1 %.not3, float %1690, float %1684, !dbg !15
  %1695 = bitcast float %1694 to i32, !dbg !15
  %1696 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1695), !dbg !15
  %1697 = extractelement <16 x float> %1543, i64 12, !dbg !15
  %1698 = extractelement <16 x float> %1543, i64 13, !dbg !15
  %1699 = extractelement <16 x float> %1543, i64 14, !dbg !15
  %1700 = extractelement <16 x float> %1543, i64 15, !dbg !15
  %1701 = select i1 %.not2, float %1698, float %1697, !dbg !15
  %1702 = bitcast float %1701 to i32, !dbg !15
  %1703 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1702), !dbg !15
  %1704 = bitcast i32 %1703 to float, !dbg !15
  %1705 = select i1 %.not2, float %1697, float %1704, !dbg !15
  %1706 = select i1 %.not2, float %1704, float %1698, !dbg !15
  %1707 = select i1 %.not2, float %1700, float %1699, !dbg !15
  %1708 = bitcast float %1707 to i32, !dbg !15
  %1709 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1708), !dbg !15
  %1710 = bitcast i32 %1709 to float, !dbg !15
  %1711 = select i1 %.not2, float %1699, float %1710, !dbg !15
  %1712 = select i1 %.not2, float %1710, float %1700, !dbg !15
  %1713 = select i1 %.not3, float %1711, float %1705, !dbg !15
  %1714 = bitcast float %1713 to i32, !dbg !15
  %1715 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1714), !dbg !15
  %1716 = bitcast i32 %1715 to float, !dbg !15
  %1717 = select i1 %.not3, float %1705, float %1716, !dbg !15
  %1718 = select i1 %.not3, float %1716, float %1711, !dbg !15
  %1719 = select i1 %.not3, float %1712, float %1706, !dbg !15
  %1720 = bitcast float %1719 to i32, !dbg !15
  %1721 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1720), !dbg !15
  %1722 = bitcast i32 %1721 to float, !dbg !15
  %1723 = select i1 %.not3, float %1706, float %1722, !dbg !15
  %1724 = select i1 %.not3, float %1722, float %1712, !dbg !15
  %1725 = extractelement <16 x float> %1572, i64 12, !dbg !15
  %1726 = extractelement <16 x float> %1572, i64 13, !dbg !15
  %1727 = extractelement <16 x float> %1572, i64 14, !dbg !15
  %1728 = extractelement <16 x float> %1572, i64 15, !dbg !15
  %1729 = select i1 %.not2, float %1726, float %1725, !dbg !15
  %1730 = bitcast float %1729 to i32, !dbg !15
  %1731 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1730), !dbg !15
  %1732 = bitcast i32 %1731 to float, !dbg !15
  %1733 = select i1 %.not2, float %1725, float %1732, !dbg !15
  %1734 = select i1 %.not2, float %1732, float %1726, !dbg !15
  %1735 = select i1 %.not2, float %1728, float %1727, !dbg !15
  %1736 = bitcast float %1735 to i32, !dbg !15
  %1737 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1340, i32 %1736), !dbg !15
  %1738 = bitcast i32 %1737 to float, !dbg !15
  %1739 = select i1 %.not2, float %1727, float %1738, !dbg !15
  %1740 = select i1 %.not2, float %1738, float %1728, !dbg !15
  %1741 = select i1 %.not3, float %1739, float %1733, !dbg !15
  %1742 = bitcast float %1741 to i32, !dbg !15
  %1743 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1742), !dbg !15
  %1744 = select i1 %.not3, float %1740, float %1734, !dbg !15
  %1745 = bitcast float %1744 to i32, !dbg !15
  %1746 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1356, i32 %1745), !dbg !15
  br i1 true, label %1747, label %1871, !dbg !21

1747:                                             ; preds = %1324
  %1748 = bitcast i32 %1743 to float, !dbg !15
  %1749 = select i1 %.not3, float %1733, float %1748, !dbg !15
  %1750 = insertelement <4 x float> poison, float %1749, i64 0, !dbg !19
  %1751 = bitcast i32 %1746 to float, !dbg !15
  %1752 = select i1 %.not3, float %1734, float %1751, !dbg !15
  %1753 = insertelement <4 x float> %1750, float %1752, i64 1, !dbg !19
  %1754 = select i1 %.not3, float %1748, float %1739, !dbg !15
  %1755 = insertelement <4 x float> %1753, float %1754, i64 2, !dbg !19
  %1756 = select i1 %.not3, float %1751, float %1740, !dbg !15
  %1757 = insertelement <4 x float> %1755, float %1756, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1596, !dbg !19
  %normalized = fmul <4 x float> %1757, %row.reciprocal, !dbg !19
  %1758 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1759 = extractelement <4 x half> %1758, i64 3, !dbg !20
  %1760 = extractelement <4 x half> %1758, i64 2, !dbg !20
  %1761 = extractelement <4 x half> %1758, i64 1, !dbg !20
  %1762 = extractelement <4 x half> %1758, i64 0, !dbg !20
  %1763 = bitcast i32 %1693 to float, !dbg !15
  %1764 = select i1 %.not3, float %1683, float %1763, !dbg !15
  %1765 = insertelement <4 x float> poison, float %1764, i64 0, !dbg !19
  %1766 = bitcast i32 %1696 to float, !dbg !15
  %1767 = select i1 %.not3, float %1684, float %1766, !dbg !15
  %1768 = insertelement <4 x float> %1765, float %1767, i64 1, !dbg !19
  %1769 = select i1 %.not3, float %1763, float %1689, !dbg !15
  %1770 = insertelement <4 x float> %1768, float %1769, i64 2, !dbg !19
  %1771 = select i1 %.not3, float %1766, float %1690, !dbg !15
  %1772 = insertelement <4 x float> %1770, float %1771, i64 3, !dbg !19
  %normalized72 = fmul <4 x float> %1772, %row.reciprocal, !dbg !19
  %1773 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !20
  %1774 = extractelement <4 x half> %1773, i64 3, !dbg !20
  %1775 = extractelement <4 x half> %1773, i64 2, !dbg !20
  %1776 = extractelement <4 x half> %1773, i64 1, !dbg !20
  %1777 = extractelement <4 x half> %1773, i64 0, !dbg !20
  %1778 = bitcast i32 %1643 to float, !dbg !15
  %1779 = select i1 %.not3, float %1633, float %1778, !dbg !15
  %1780 = insertelement <4 x float> poison, float %1779, i64 0, !dbg !19
  %1781 = bitcast i32 %1646 to float, !dbg !15
  %1782 = select i1 %.not3, float %1634, float %1781, !dbg !15
  %1783 = insertelement <4 x float> %1780, float %1782, i64 1, !dbg !19
  %1784 = select i1 %.not3, float %1778, float %1639, !dbg !15
  %1785 = insertelement <4 x float> %1783, float %1784, i64 2, !dbg !19
  %1786 = select i1 %.not3, float %1781, float %1640, !dbg !15
  %1787 = insertelement <4 x float> %1785, float %1786, i64 3, !dbg !19
  %normalized73 = fmul <4 x float> %1787, %row.reciprocal, !dbg !19
  %1788 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !20
  %1789 = extractelement <4 x half> %1788, i64 3, !dbg !20
  %1790 = extractelement <4 x half> %1788, i64 2, !dbg !20
  %1791 = extractelement <4 x half> %1788, i64 1, !dbg !20
  %1792 = extractelement <4 x half> %1788, i64 0, !dbg !20
  %1793 = bitcast i32 %1591 to float, !dbg !15
  %1794 = select i1 %.not3, float %1581, float %1793, !dbg !15
  %1795 = insertelement <4 x float> poison, float %1794, i64 0, !dbg !19
  %1796 = bitcast i32 %1594 to float, !dbg !15
  %1797 = select i1 %.not3, float %1582, float %1796, !dbg !15
  %1798 = insertelement <4 x float> %1795, float %1797, i64 1, !dbg !19
  %1799 = select i1 %.not3, float %1793, float %1587, !dbg !15
  %1800 = insertelement <4 x float> %1798, float %1799, i64 2, !dbg !19
  %1801 = select i1 %.not3, float %1796, float %1588, !dbg !15
  %1802 = insertelement <4 x float> %1800, float %1801, i64 3, !dbg !19
  %normalized74 = fmul <4 x float> %1802, %row.reciprocal, !dbg !19
  %1803 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %1804 = extractelement <4 x half> %1803, i64 3, !dbg !20
  %1805 = extractelement <4 x half> %1803, i64 2, !dbg !20
  %1806 = extractelement <4 x half> %1803, i64 1, !dbg !20
  %1807 = extractelement <4 x half> %1803, i64 0, !dbg !20
  %1808 = bitcast i32 %1539 to float, !dbg !15
  %1809 = select i1 %.not3, float %1529, float %1808, !dbg !15
  %1810 = insertelement <4 x float> poison, float %1809, i64 0, !dbg !19
  %1811 = bitcast i32 %1542 to float, !dbg !15
  %1812 = select i1 %.not3, float %1530, float %1811, !dbg !15
  %1813 = insertelement <4 x float> %1810, float %1812, i64 1, !dbg !19
  %1814 = select i1 %.not3, float %1808, float %1535, !dbg !15
  %1815 = insertelement <4 x float> %1813, float %1814, i64 2, !dbg !19
  %1816 = select i1 %.not3, float %1811, float %1536, !dbg !15
  %1817 = insertelement <4 x float> %1815, float %1816, i64 3, !dbg !19
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1392, !dbg !19
  %normalized76 = fmul <4 x float> %1817, %row.reciprocal75, !dbg !19
  %1818 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !20
  %1819 = extractelement <4 x half> %1818, i64 3, !dbg !20
  %1820 = extractelement <4 x half> %1818, i64 2, !dbg !20
  %1821 = extractelement <4 x half> %1818, i64 1, !dbg !20
  %1822 = extractelement <4 x half> %1818, i64 0, !dbg !20
  %1823 = bitcast i32 %1489 to float, !dbg !15
  %1824 = select i1 %.not3, float %1479, float %1823, !dbg !15
  %1825 = insertelement <4 x float> poison, float %1824, i64 0, !dbg !19
  %1826 = bitcast i32 %1492 to float, !dbg !15
  %1827 = select i1 %.not3, float %1480, float %1826, !dbg !15
  %1828 = insertelement <4 x float> %1825, float %1827, i64 1, !dbg !19
  %1829 = select i1 %.not3, float %1823, float %1485, !dbg !15
  %1830 = insertelement <4 x float> %1828, float %1829, i64 2, !dbg !19
  %1831 = select i1 %.not3, float %1826, float %1486, !dbg !15
  %1832 = insertelement <4 x float> %1830, float %1831, i64 3, !dbg !19
  %normalized77 = fmul <4 x float> %1832, %row.reciprocal75, !dbg !19
  %1833 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !20
  %1834 = extractelement <4 x half> %1833, i64 3, !dbg !20
  %1835 = extractelement <4 x half> %1833, i64 2, !dbg !20
  %1836 = extractelement <4 x half> %1833, i64 1, !dbg !20
  %1837 = extractelement <4 x half> %1833, i64 0, !dbg !20
  %1838 = bitcast i32 %1439 to float, !dbg !15
  %1839 = select i1 %.not3, float %1429, float %1838, !dbg !15
  %1840 = insertelement <4 x float> poison, float %1839, i64 0, !dbg !19
  %1841 = bitcast i32 %1442 to float, !dbg !15
  %1842 = select i1 %.not3, float %1430, float %1841, !dbg !15
  %1843 = insertelement <4 x float> %1840, float %1842, i64 1, !dbg !19
  %1844 = select i1 %.not3, float %1838, float %1435, !dbg !15
  %1845 = insertelement <4 x float> %1843, float %1844, i64 2, !dbg !19
  %1846 = select i1 %.not3, float %1841, float %1436, !dbg !15
  %1847 = insertelement <4 x float> %1845, float %1846, i64 3, !dbg !19
  %normalized78 = fmul <4 x float> %1847, %row.reciprocal75, !dbg !19
  %1848 = fptrunc <4 x float> %normalized78 to <4 x half>, !dbg !20
  %1849 = extractelement <4 x half> %1848, i64 3, !dbg !20
  %1850 = extractelement <4 x half> %1848, i64 2, !dbg !20
  %1851 = extractelement <4 x half> %1848, i64 1, !dbg !20
  %1852 = extractelement <4 x half> %1848, i64 0, !dbg !20
  %1853 = bitcast i32 %1387 to float, !dbg !15
  %1854 = select i1 %.not3, float %1377, float %1853, !dbg !15
  %1855 = insertelement <4 x float> poison, float %1854, i64 0, !dbg !19
  %1856 = bitcast i32 %1390 to float, !dbg !15
  %1857 = select i1 %.not3, float %1378, float %1856, !dbg !15
  %1858 = insertelement <4 x float> %1855, float %1857, i64 1, !dbg !19
  %1859 = select i1 %.not3, float %1853, float %1383, !dbg !15
  %1860 = insertelement <4 x float> %1858, float %1859, i64 2, !dbg !19
  %1861 = select i1 %.not3, float %1856, float %1384, !dbg !15
  %1862 = insertelement <4 x float> %1860, float %1861, i64 3, !dbg !19
  %normalized79 = fmul <4 x float> %1862, %row.reciprocal75, !dbg !19
  %1863 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !20
  %1864 = extractelement <4 x half> %1863, i64 3, !dbg !20
  %1865 = extractelement <4 x half> %1863, i64 2, !dbg !20
  %1866 = extractelement <4 x half> %1863, i64 1, !dbg !20
  %1867 = extractelement <4 x half> %1863, i64 0, !dbg !20
  %1868 = add nsw i64 %8, %13, !dbg !21
  %1869 = shl nsw i64 %1868, 19, !dbg !21
  %1870 = and i64 %1869, 16252928, !dbg !21
  %address.thread650 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local651 = zext nneg i32 %address.thread650 to i64, !dbg !13
  %address.local652 = add nsw i64 %16, %address.local651, !dbg !21
  %address.local653 = lshr i64 %address.local651, 6, !dbg !13
  %address.local654 = shl nuw nsw i64 %address.local653, 4, !dbg !21
  %address.local655 = add nsw i64 %address.local652, %address.local654, !dbg !21
  %address.local656 = and i64 %address.local651, 240, !dbg !21
  %address.local657 = sub nsw i64 %address.local655, %address.local656, !dbg !21
  %address.local658 = add nsw i64 %address.local657, %17, !dbg !21
  %address.local659 = shl nuw nsw i64 %address.local658, 7, !dbg !21
  %address.local660 = add nuw nsw i64 %1870, %address.local659, !dbg !21
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local662 = lshr i64 %address.local651, 2, !dbg !13
  %address.local663 = and i64 %address.local662, 12, !dbg !13
  %address.local664 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local661, i64 %address.local663, !dbg !21
  store half %1867, ptr addrspace(1) %address.local664, align 2, !dbg !21
  %address.local665 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local666 = or disjoint i64 %address.local663, 1, !dbg !21
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local665, i64 %address.local666, !dbg !21
  store half %1866, ptr addrspace(1) %address.local667, align 2, !dbg !21
  %address.local668 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local669 = or disjoint i64 %address.local663, 2, !dbg !21
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local668, i64 %address.local669, !dbg !21
  store half %1865, ptr addrspace(1) %address.local670, align 2, !dbg !21
  %address.local671 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local672 = or disjoint i64 %address.local663, 3, !dbg !21
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local671, i64 %address.local672, !dbg !21
  store half %1864, ptr addrspace(1) %address.local673, align 2, !dbg !21
  %address.local674 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local675 = or disjoint i64 %address.local663, 16, !dbg !21
  %address.local676 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local674, i64 %address.local675, !dbg !21
  store half %1852, ptr addrspace(1) %address.local676, align 2, !dbg !21
  %address.local677 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local678 = or disjoint i64 %address.local663, 17, !dbg !21
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local677, i64 %address.local678, !dbg !21
  store half %1851, ptr addrspace(1) %address.local679, align 2, !dbg !21
  %address.local680 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local681 = or disjoint i64 %address.local663, 18, !dbg !21
  %address.local682 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local680, i64 %address.local681, !dbg !21
  store half %1850, ptr addrspace(1) %address.local682, align 2, !dbg !21
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local684 = or disjoint i64 %address.local663, 19, !dbg !21
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local683, i64 %address.local684, !dbg !21
  store half %1849, ptr addrspace(1) %address.local685, align 2, !dbg !21
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local687 = or disjoint i64 %address.local663, 32, !dbg !21
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local686, i64 %address.local687, !dbg !21
  store half %1837, ptr addrspace(1) %address.local688, align 2, !dbg !21
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local690 = or disjoint i64 %address.local663, 33, !dbg !21
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local689, i64 %address.local690, !dbg !21
  store half %1836, ptr addrspace(1) %address.local691, align 2, !dbg !21
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local693 = or disjoint i64 %address.local663, 34, !dbg !21
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local692, i64 %address.local693, !dbg !21
  store half %1835, ptr addrspace(1) %address.local694, align 2, !dbg !21
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local696 = or disjoint i64 %address.local663, 35, !dbg !21
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local695, i64 %address.local696, !dbg !21
  store half %1834, ptr addrspace(1) %address.local697, align 2, !dbg !21
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local699 = or disjoint i64 %address.local663, 48, !dbg !21
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local698, i64 %address.local699, !dbg !21
  store half %1822, ptr addrspace(1) %address.local700, align 2, !dbg !21
  %address.local701 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local702 = or disjoint i64 %address.local663, 49, !dbg !21
  %address.local703 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local701, i64 %address.local702, !dbg !21
  store half %1821, ptr addrspace(1) %address.local703, align 2, !dbg !21
  %address.local704 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local705 = or disjoint i64 %address.local663, 50, !dbg !21
  %address.local706 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local704, i64 %address.local705, !dbg !21
  store half %1820, ptr addrspace(1) %address.local706, align 2, !dbg !21
  %address.local707 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local660, !dbg !21
  %address.local708 = or i64 %address.local662, 51, !dbg !21
  %address.local709 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local707, i64 %address.local708, !dbg !21
  store half %1819, ptr addrspace(1) %address.local709, align 2, !dbg !21
  %address.local710 = shl nsw i64 %address.local658, 7, !dbg !21
  %address.local711 = add i64 %address.local710, 8192, !dbg !21
  %address.local712 = add nuw nsw i64 %1870, %address.local711, !dbg !21
  %address.local713 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local714 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local713, i64 %address.local663, !dbg !21
  store half %1807, ptr addrspace(1) %address.local714, align 2, !dbg !21
  %address.local715 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local715, i64 %address.local666, !dbg !21
  store half %1806, ptr addrspace(1) %address.local716, align 2, !dbg !21
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local718 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local717, i64 %address.local669, !dbg !21
  store half %1805, ptr addrspace(1) %address.local718, align 2, !dbg !21
  %address.local719 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local720 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local719, i64 %address.local672, !dbg !21
  store half %1804, ptr addrspace(1) %address.local720, align 2, !dbg !21
  %address.local721 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local721, i64 %address.local675, !dbg !21
  store half %1792, ptr addrspace(1) %address.local722, align 2, !dbg !21
  %address.local723 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local724 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local723, i64 %address.local678, !dbg !21
  store half %1791, ptr addrspace(1) %address.local724, align 2, !dbg !21
  %address.local725 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local725, i64 %address.local681, !dbg !21
  store half %1790, ptr addrspace(1) %address.local726, align 2, !dbg !21
  %address.local727 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local728 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local727, i64 %address.local684, !dbg !21
  store half %1789, ptr addrspace(1) %address.local728, align 2, !dbg !21
  %address.local729 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local730 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local729, i64 %address.local687, !dbg !21
  store half %1777, ptr addrspace(1) %address.local730, align 2, !dbg !21
  %address.local731 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local732 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local731, i64 %address.local690, !dbg !21
  store half %1776, ptr addrspace(1) %address.local732, align 2, !dbg !21
  %address.local733 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local734 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local733, i64 %address.local693, !dbg !21
  store half %1775, ptr addrspace(1) %address.local734, align 2, !dbg !21
  %address.local735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local736 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local735, i64 %address.local696, !dbg !21
  store half %1774, ptr addrspace(1) %address.local736, align 2, !dbg !21
  %address.local737 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local738 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local737, i64 %address.local699, !dbg !21
  store half %1762, ptr addrspace(1) %address.local738, align 2, !dbg !21
  %address.local739 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local740 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local739, i64 %address.local702, !dbg !21
  store half %1761, ptr addrspace(1) %address.local740, align 2, !dbg !21
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local741, i64 %address.local705, !dbg !21
  store half %1760, ptr addrspace(1) %address.local742, align 2, !dbg !21
  %address.local743 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local712, !dbg !21
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local743, i64 %address.local708, !dbg !21
  store half %1759, ptr addrspace(1) %address.local744, align 2, !dbg !21
  br label %1871, !dbg !21

1871:                                             ; preds = %1747, %1324
  br i1 true, label %1872, label %1946, !dbg !21

1872:                                             ; preds = %1871
  %1873 = insertelement <4 x float> poison, float %1717, i64 0, !dbg !19
  %1874 = insertelement <4 x float> %1873, float %1723, i64 1, !dbg !19
  %1875 = insertelement <4 x float> %1874, float %1718, i64 2, !dbg !19
  %1876 = insertelement <4 x float> %1875, float %1724, i64 3, !dbg !19
  %row.reciprocal80 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1596, !dbg !19
  %normalized81 = fmul <4 x float> %1876, %row.reciprocal80, !dbg !19
  %1877 = fptrunc <4 x float> %normalized81 to <4 x half>, !dbg !20
  %1878 = extractelement <4 x half> %1877, i64 3, !dbg !20
  %1879 = extractelement <4 x half> %1877, i64 2, !dbg !20
  %1880 = extractelement <4 x half> %1877, i64 1, !dbg !20
  %1881 = extractelement <4 x half> %1877, i64 0, !dbg !20
  %1882 = insertelement <4 x float> poison, float %1667, i64 0, !dbg !19
  %1883 = insertelement <4 x float> %1882, float %1673, i64 1, !dbg !19
  %1884 = insertelement <4 x float> %1883, float %1668, i64 2, !dbg !19
  %1885 = insertelement <4 x float> %1884, float %1674, i64 3, !dbg !19
  %normalized82 = fmul <4 x float> %1885, %row.reciprocal80, !dbg !19
  %1886 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !20
  %1887 = extractelement <4 x half> %1886, i64 3, !dbg !20
  %1888 = extractelement <4 x half> %1886, i64 2, !dbg !20
  %1889 = extractelement <4 x half> %1886, i64 1, !dbg !20
  %1890 = extractelement <4 x half> %1886, i64 0, !dbg !20
  %1891 = insertelement <4 x float> poison, float %1617, i64 0, !dbg !19
  %1892 = insertelement <4 x float> %1891, float %1623, i64 1, !dbg !19
  %1893 = insertelement <4 x float> %1892, float %1618, i64 2, !dbg !19
  %1894 = insertelement <4 x float> %1893, float %1624, i64 3, !dbg !19
  %normalized83 = fmul <4 x float> %1894, %row.reciprocal80, !dbg !19
  %1895 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !20
  %1896 = extractelement <4 x half> %1895, i64 3, !dbg !20
  %1897 = extractelement <4 x half> %1895, i64 2, !dbg !20
  %1898 = extractelement <4 x half> %1895, i64 1, !dbg !20
  %1899 = extractelement <4 x half> %1895, i64 0, !dbg !20
  %1900 = insertelement <4 x float> poison, float %1564, i64 0, !dbg !19
  %1901 = insertelement <4 x float> %1900, float %1570, i64 1, !dbg !19
  %1902 = insertelement <4 x float> %1901, float %1565, i64 2, !dbg !19
  %1903 = insertelement <4 x float> %1902, float %1571, i64 3, !dbg !19
  %normalized84 = fmul <4 x float> %1903, %row.reciprocal80, !dbg !19
  %1904 = fptrunc <4 x float> %normalized84 to <4 x half>, !dbg !20
  %1905 = extractelement <4 x half> %1904, i64 3, !dbg !20
  %1906 = extractelement <4 x half> %1904, i64 2, !dbg !20
  %1907 = extractelement <4 x half> %1904, i64 1, !dbg !20
  %1908 = extractelement <4 x half> %1904, i64 0, !dbg !20
  %1909 = insertelement <4 x float> poison, float %1513, i64 0, !dbg !19
  %1910 = insertelement <4 x float> %1909, float %1519, i64 1, !dbg !19
  %1911 = insertelement <4 x float> %1910, float %1514, i64 2, !dbg !19
  %1912 = insertelement <4 x float> %1911, float %1520, i64 3, !dbg !19
  %row.reciprocal85 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1392, !dbg !19
  %normalized86 = fmul <4 x float> %1912, %row.reciprocal85, !dbg !19
  %1913 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !20
  %1914 = extractelement <4 x half> %1913, i64 3, !dbg !20
  %1915 = extractelement <4 x half> %1913, i64 2, !dbg !20
  %1916 = extractelement <4 x half> %1913, i64 1, !dbg !20
  %1917 = extractelement <4 x half> %1913, i64 0, !dbg !20
  %1918 = insertelement <4 x float> poison, float %1463, i64 0, !dbg !19
  %1919 = insertelement <4 x float> %1918, float %1469, i64 1, !dbg !19
  %1920 = insertelement <4 x float> %1919, float %1464, i64 2, !dbg !19
  %1921 = insertelement <4 x float> %1920, float %1470, i64 3, !dbg !19
  %normalized87 = fmul <4 x float> %1921, %row.reciprocal85, !dbg !19
  %1922 = fptrunc <4 x float> %normalized87 to <4 x half>, !dbg !20
  %1923 = extractelement <4 x half> %1922, i64 3, !dbg !20
  %1924 = extractelement <4 x half> %1922, i64 2, !dbg !20
  %1925 = extractelement <4 x half> %1922, i64 1, !dbg !20
  %1926 = extractelement <4 x half> %1922, i64 0, !dbg !20
  %1927 = insertelement <4 x float> poison, float %1413, i64 0, !dbg !19
  %1928 = insertelement <4 x float> %1927, float %1419, i64 1, !dbg !19
  %1929 = insertelement <4 x float> %1928, float %1414, i64 2, !dbg !19
  %1930 = insertelement <4 x float> %1929, float %1420, i64 3, !dbg !19
  %normalized88 = fmul <4 x float> %1930, %row.reciprocal85, !dbg !19
  %1931 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !20
  %1932 = extractelement <4 x half> %1931, i64 3, !dbg !20
  %1933 = extractelement <4 x half> %1931, i64 2, !dbg !20
  %1934 = extractelement <4 x half> %1931, i64 1, !dbg !20
  %1935 = extractelement <4 x half> %1931, i64 0, !dbg !20
  %1936 = insertelement <4 x float> poison, float %1360, i64 0, !dbg !19
  %1937 = insertelement <4 x float> %1936, float %1366, i64 1, !dbg !19
  %1938 = insertelement <4 x float> %1937, float %1361, i64 2, !dbg !19
  %1939 = insertelement <4 x float> %1938, float %1367, i64 3, !dbg !19
  %normalized89 = fmul <4 x float> %1939, %row.reciprocal85, !dbg !19
  %1940 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !20
  %1941 = extractelement <4 x half> %1940, i64 3, !dbg !20
  %1942 = extractelement <4 x half> %1940, i64 2, !dbg !20
  %1943 = extractelement <4 x half> %1940, i64 1, !dbg !20
  %1944 = extractelement <4 x half> %1940, i64 0, !dbg !20
  %1945 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread745 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local746 = zext nneg i32 %address.thread745 to i64, !dbg !13
  %address.local747 = add nsw i64 %16, %address.local746, !dbg !21
  %address.local748 = lshr i64 %address.local746, 6, !dbg !13
  %address.local749 = shl nuw nsw i64 %address.local748, 4, !dbg !21
  %address.local750 = add nsw i64 %address.local747, %address.local749, !dbg !21
  %address.local751 = and i64 %address.local746, 240, !dbg !21
  %address.local752 = sub nsw i64 %address.local750, %address.local751, !dbg !21
  %address.local753 = add nsw i64 %address.local752, %17, !dbg !21
  %address.local754 = shl nuw nsw i64 %address.local753, 7, !dbg !21
  %address.local755 = add nuw nsw i64 %1945, %address.local754, !dbg !21
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local757 = lshr i64 %address.local746, 2, !dbg !13
  %address.local758 = and i64 %address.local757, 12, !dbg !13
  %address.local759 = or disjoint i64 %address.local758, 64, !dbg !21
  %address.local760 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local756, i64 %address.local759, !dbg !21
  store half %1944, ptr addrspace(1) %address.local760, align 2, !dbg !21
  %address.local761 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local762 = or disjoint i64 %address.local758, 65, !dbg !21
  %address.local763 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local761, i64 %address.local762, !dbg !21
  store half %1943, ptr addrspace(1) %address.local763, align 2, !dbg !21
  %address.local764 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local765 = or disjoint i64 %address.local758, 66, !dbg !21
  %address.local766 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local764, i64 %address.local765, !dbg !21
  store half %1942, ptr addrspace(1) %address.local766, align 2, !dbg !21
  %address.local767 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local768 = or disjoint i64 %address.local758, 67, !dbg !21
  %address.local769 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local767, i64 %address.local768, !dbg !21
  store half %1941, ptr addrspace(1) %address.local769, align 2, !dbg !21
  %address.local770 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local771 = or disjoint i64 %address.local758, 80, !dbg !21
  %address.local772 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local770, i64 %address.local771, !dbg !21
  store half %1935, ptr addrspace(1) %address.local772, align 2, !dbg !21
  %address.local773 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local774 = or disjoint i64 %address.local758, 81, !dbg !21
  %address.local775 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local773, i64 %address.local774, !dbg !21
  store half %1934, ptr addrspace(1) %address.local775, align 2, !dbg !21
  %address.local776 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local777 = or disjoint i64 %address.local758, 82, !dbg !21
  %address.local778 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local776, i64 %address.local777, !dbg !21
  store half %1933, ptr addrspace(1) %address.local778, align 2, !dbg !21
  %address.local779 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local780 = or disjoint i64 %address.local758, 83, !dbg !21
  %address.local781 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local779, i64 %address.local780, !dbg !21
  store half %1932, ptr addrspace(1) %address.local781, align 2, !dbg !21
  %address.local782 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local783 = or disjoint i64 %address.local758, 96, !dbg !21
  %address.local784 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local782, i64 %address.local783, !dbg !21
  store half %1926, ptr addrspace(1) %address.local784, align 2, !dbg !21
  %address.local785 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local786 = or disjoint i64 %address.local758, 97, !dbg !21
  %address.local787 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local785, i64 %address.local786, !dbg !21
  store half %1925, ptr addrspace(1) %address.local787, align 2, !dbg !21
  %address.local788 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local789 = or disjoint i64 %address.local758, 98, !dbg !21
  %address.local790 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local788, i64 %address.local789, !dbg !21
  store half %1924, ptr addrspace(1) %address.local790, align 2, !dbg !21
  %address.local791 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local792 = or disjoint i64 %address.local758, 99, !dbg !21
  %address.local793 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local791, i64 %address.local792, !dbg !21
  store half %1923, ptr addrspace(1) %address.local793, align 2, !dbg !21
  %address.local794 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local795 = or disjoint i64 %address.local758, 112, !dbg !21
  %address.local796 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local794, i64 %address.local795, !dbg !21
  store half %1917, ptr addrspace(1) %address.local796, align 2, !dbg !21
  %address.local797 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local798 = or disjoint i64 %address.local758, 113, !dbg !21
  %address.local799 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local797, i64 %address.local798, !dbg !21
  store half %1916, ptr addrspace(1) %address.local799, align 2, !dbg !21
  %address.local800 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local801 = or disjoint i64 %address.local758, 114, !dbg !21
  %address.local802 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local800, i64 %address.local801, !dbg !21
  store half %1915, ptr addrspace(1) %address.local802, align 2, !dbg !21
  %address.local803 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local755, !dbg !21
  %address.local804 = or i64 %address.local757, 115, !dbg !21
  %address.local805 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local803, i64 %address.local804, !dbg !21
  store half %1914, ptr addrspace(1) %address.local805, align 2, !dbg !21
  %address.local806 = shl nsw i64 %address.local753, 7, !dbg !21
  %address.local807 = add i64 %address.local806, 8192, !dbg !21
  %address.local808 = add nuw nsw i64 %1945, %address.local807, !dbg !21
  %address.local809 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local810 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local809, i64 %address.local759, !dbg !21
  store half %1908, ptr addrspace(1) %address.local810, align 2, !dbg !21
  %address.local811 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local812 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local811, i64 %address.local762, !dbg !21
  store half %1907, ptr addrspace(1) %address.local812, align 2, !dbg !21
  %address.local813 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local814 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local813, i64 %address.local765, !dbg !21
  store half %1906, ptr addrspace(1) %address.local814, align 2, !dbg !21
  %address.local815 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local816 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local815, i64 %address.local768, !dbg !21
  store half %1905, ptr addrspace(1) %address.local816, align 2, !dbg !21
  %address.local817 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local818 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local817, i64 %address.local771, !dbg !21
  store half %1899, ptr addrspace(1) %address.local818, align 2, !dbg !21
  %address.local819 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local820 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local819, i64 %address.local774, !dbg !21
  store half %1898, ptr addrspace(1) %address.local820, align 2, !dbg !21
  %address.local821 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local822 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local821, i64 %address.local777, !dbg !21
  store half %1897, ptr addrspace(1) %address.local822, align 2, !dbg !21
  %address.local823 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local824 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local823, i64 %address.local780, !dbg !21
  store half %1896, ptr addrspace(1) %address.local824, align 2, !dbg !21
  %address.local825 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local826 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local825, i64 %address.local783, !dbg !21
  store half %1890, ptr addrspace(1) %address.local826, align 2, !dbg !21
  %address.local827 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local828 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local827, i64 %address.local786, !dbg !21
  store half %1889, ptr addrspace(1) %address.local828, align 2, !dbg !21
  %address.local829 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local830 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local829, i64 %address.local789, !dbg !21
  store half %1888, ptr addrspace(1) %address.local830, align 2, !dbg !21
  %address.local831 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local832 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local831, i64 %address.local792, !dbg !21
  store half %1887, ptr addrspace(1) %address.local832, align 2, !dbg !21
  %address.local833 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local834 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local833, i64 %address.local795, !dbg !21
  store half %1881, ptr addrspace(1) %address.local834, align 2, !dbg !21
  %address.local835 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local836 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local835, i64 %address.local798, !dbg !21
  store half %1880, ptr addrspace(1) %address.local836, align 2, !dbg !21
  %address.local837 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local838 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local837, i64 %address.local801, !dbg !21
  store half %1879, ptr addrspace(1) %address.local838, align 2, !dbg !21
  %address.local839 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local808, !dbg !21
  %address.local840 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local839, i64 %address.local804, !dbg !21
  store half %1878, ptr addrspace(1) %address.local840, align 2, !dbg !21
  br label %1946, !dbg !21

1946:                                             ; preds = %1872, %1871
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #6

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #7

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #7 = { nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { convergent nounwind memory(none) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!33}

!0 = !DIFile(filename: "default.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/k5_independent_tiles")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!8 = !DILocation(line: 69, column: 26, scope: !4)
!10 = !DILocation(line: 46, column: 17, scope: !4)
!12 = !DILocation(line: 66, column: 22, scope: !4)
!13 = !DILocation(line: 68, column: 24, scope: !4)
!14 = !DILocation(line: 73, column: 19, scope: !4)
!15 = !DILocation(line: 56, column: 24, scope: !4)
!16 = !DILocation(line: 67, column: 22, scope: !4)
!17 = !DILocation(line: 92, column: 20, scope: !4)
!18 = !DILocation(line: 93, column: 23, scope: !4)
!19 = !DILocation(line: 97, column: 23, scope: !4)
!20 = !DILocation(line: 98, column: 19, scope: !4)
!21 = !DILocation(line: 99, column: 9, scope: !4)
!22 = !DILocation(line: 59, column: 17, scope: !4)
!23 = !DILocation(line: 75, column: 24, scope: !4)
!24 = !DILocation(line: 76, column: 23, scope: !4)
!25 = !DILocation(line: 77, column: 26, scope: !4)
!26 = !DILocation(line: 84, column: 21, scope: !4)
!27 = !DILocation(line: 85, column: 22, scope: !4)
!28 = !DILocation(line: 86, column: 20, scope: !4)
!29 = !DILocation(line: 87, column: 16, scope: !4)
!30 = !DILocation(line: 94, column: 20, scope: !4)
!31 = !DILocation(line: 70, column: 23, scope: !4)
!32 = !{i32 256, i32 1, i32 1}
!33 = !{i32 2, !"Debug Info Version", i32 3}
!34 = !{i32 0, i32 256}
