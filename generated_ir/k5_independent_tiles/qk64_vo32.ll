; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/k5_independent_tiles/qk64_vo32.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/k5_independent_tiles/qk64_vo32.mlir"

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
  br label %25, !dbg !16

25:                                               ; preds = %574, %4
  %26 = phi i64 [ %1325, %574 ], [ 0, %4 ], !dbg !16
  %27 = phi [2 x <8 x float>] [ %1312, %574 ], [ zeroinitializer, %4 ], !dbg !16
  %28 = phi [2 x <1 x float>] [ %934, %574 ], [ zeroinitializer, %4 ], !dbg !16
  %29 = phi [2 x <8 x float>] [ %1316, %574 ], [ zeroinitializer, %4 ], !dbg !16
  %30 = phi [2 x <8 x float>] [ %1320, %574 ], [ zeroinitializer, %4 ], !dbg !16
  %31 = phi [2 x <8 x float>] [ %1324, %574 ], [ zeroinitializer, %4 ], !dbg !16
  %.not = icmp sgt i64 %26, %6, !dbg !16
  br i1 %.not, label %1326, label %32, !dbg !16

32:                                               ; preds = %25
  %33 = shl nsw i64 %26, 7, !dbg !16
  %34 = lshr i64 %26, 5, !dbg !17
  %35 = shl i64 %34, 12, !dbg !17
  %36 = sub i64 %33, %35, !dbg !17
  br label %37, !dbg !22

37:                                               ; preds = %41, %32
  %38 = phi i64 [ %361, %41 ], [ 0, %32 ], !dbg !22
  %39 = phi [2 x <32 x float>] [ %360, %41 ], [ zeroinitializer, %32 ], !dbg !22
  %40 = icmp samesign ult i64 %38, 2, !dbg !22
  br i1 %40, label %41, label %362, !dbg !22

41:                                               ; preds = %37
  %42 = shl nuw nsw i64 %38, 6, !dbg !22
  %.idx = shl nuw nsw i64 %15, 20, !dbg !12
  %43 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %.idx68 = shl nsw i64 %18, 8, !dbg !12
  %44 = getelementptr i8, ptr addrspace(1) %43, i64 %.idx68, !dbg !12
  %45 = getelementptr [2 x i8], ptr addrspace(1) %44, i64 %42, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !12, !range !34, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local142 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local143 = and i64 %address.local142, 1536, !dbg !12
  %address.local144 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local143, !dbg !12
  %address.local145 = ptrtoint ptr addrspace(3) %address.local144 to i64, !dbg !12
  %address.local146 = trunc i64 %address.local145 to i32, !dbg !12
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local146), !dbg !12
  %47 = zext i32 %46 to i64, !dbg !12
  %48 = inttoptr i64 %47 to ptr addrspace(3), !dbg !12
  %address.local147 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local148 = and i64 %address.local147, 16256, !dbg !12
  %address.local149 = lshr i64 %address.local, 4, !dbg !12
  %address.local150 = and i64 %address.local149, 8, !dbg !12
  %address.local151 = or disjoint i64 %address.local148, %address.local150, !dbg !12
  %address.local152 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %address.local151, !dbg !12
  %49 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %50 = ptrtoint ptr addrspace(1) %address.local152 to i64, !dbg !12
  %51 = sub i64 %50, %49, !dbg !12
  %dma.buffer.offset764 = trunc i64 %51 to i32, !dbg !12
  %52 = trunc i64 %49 to i32, !dbg !12
  %53 = insertelement <4 x i32> poison, i32 %52, i64 0, !dbg !12
  %54 = lshr i64 %49, 32, !dbg !12
  %55 = trunc i64 %54 to i32, !dbg !12
  %56 = insertelement <4 x i32> %53, i32 %55, i64 1, !dbg !12
  %57 = insertelement <4 x i32> %56, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc765 = insertelement <4 x i32> %57, i32 131072, i64 3, !dbg !12
  %58 = ptrtoint ptr addrspace(3) %48 to i32, !dbg !12
  %dma.buffer.lds766 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %58), !dbg !12
  %59 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds766, i32 %dma.buffer.offset764, <4 x i32> %dma.buffer.rsrc765) #10, !dbg !12
  %address.local153 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local154 = shl nuw nsw i64 %address.local153, 3, !dbg !12
  %address.local155 = and i64 %address.local154, 3584, !dbg !12
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local155, !dbg !12
  %address.local157 = ptrtoint ptr addrspace(3) %address.local156 to i64, !dbg !12
  %address.local158 = trunc i64 %address.local157 to i32, !dbg !12
  %60 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !12
  %61 = zext i32 %60 to i64, !dbg !12
  %62 = inttoptr i64 %61 to ptr addrspace(3), !dbg !12
  %address.local159 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local160 = and i64 %address.local159, 16256, !dbg !12
  %address.local161 = lshr i64 %address.local153, 4, !dbg !12
  %address.local162 = and i64 %address.local161, 24, !dbg !12
  %address.local163 = or disjoint i64 %address.local160, %address.local162, !dbg !12
  %address.local164 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %address.local163, !dbg !12
  %63 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %64 = ptrtoint ptr addrspace(1) %address.local164 to i64, !dbg !12
  %65 = sub i64 %64, %63, !dbg !12
  %dma.buffer.offset761 = trunc i64 %65 to i32, !dbg !12
  %66 = trunc i64 %63 to i32, !dbg !12
  %67 = insertelement <4 x i32> poison, i32 %66, i64 0, !dbg !12
  %68 = lshr i64 %63, 32, !dbg !12
  %69 = trunc i64 %68 to i32, !dbg !12
  %70 = insertelement <4 x i32> %67, i32 %69, i64 1, !dbg !12
  %71 = insertelement <4 x i32> %70, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc762 = insertelement <4 x i32> %71, i32 131072, i64 3, !dbg !12
  %72 = ptrtoint ptr addrspace(3) %62 to i32, !dbg !12
  %dma.buffer.lds763 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %72), !dbg !12
  %73 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds763, i32 %dma.buffer.offset761, <4 x i32> %dma.buffer.rsrc762) #10, !dbg !12
  %address.local165 = or disjoint i64 %address.local, 512, !dbg !12
  %address.local166 = shl nuw nsw i64 %address.local165, 3, !dbg !12
  %address.local167 = and i64 %address.local166, 5632, !dbg !12
  %address.local168 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local167, !dbg !12
  %address.local169 = ptrtoint ptr addrspace(3) %address.local168 to i64, !dbg !12
  %address.local170 = trunc i64 %address.local169 to i32, !dbg !12
  %74 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local170), !dbg !12
  %75 = zext i32 %74 to i64, !dbg !12
  %76 = inttoptr i64 %75 to ptr addrspace(3), !dbg !12
  %address.local171 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local172 = and i64 %address.local171, 16256, !dbg !12
  %address.local173 = lshr i64 %address.local165, 4, !dbg !12
  %address.local174 = and i64 %address.local173, 40, !dbg !12
  %address.local175 = or disjoint i64 %address.local172, %address.local174, !dbg !12
  %address.local176 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %address.local175, !dbg !12
  %77 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %78 = ptrtoint ptr addrspace(1) %address.local176 to i64, !dbg !12
  %79 = sub i64 %78, %77, !dbg !12
  %dma.buffer.offset758 = trunc i64 %79 to i32, !dbg !12
  %80 = trunc i64 %77 to i32, !dbg !12
  %81 = insertelement <4 x i32> poison, i32 %80, i64 0, !dbg !12
  %82 = lshr i64 %77, 32, !dbg !12
  %83 = trunc i64 %82 to i32, !dbg !12
  %84 = insertelement <4 x i32> %81, i32 %83, i64 1, !dbg !12
  %85 = insertelement <4 x i32> %84, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc759 = insertelement <4 x i32> %85, i32 131072, i64 3, !dbg !12
  %86 = ptrtoint ptr addrspace(3) %76 to i32, !dbg !12
  %dma.buffer.lds760 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %86), !dbg !12
  %87 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds760, i32 %dma.buffer.offset758, <4 x i32> %dma.buffer.rsrc759) #10, !dbg !12
  %address.local177 = or disjoint i64 %address.local, 768, !dbg !12
  %address.local178 = shl nuw nsw i64 %address.local177, 3, !dbg !12
  %address.local179 = and i64 %address.local178, 7680, !dbg !12
  %address.local180 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local179, !dbg !12
  %address.local181 = ptrtoint ptr addrspace(3) %address.local180 to i64, !dbg !12
  %address.local182 = trunc i64 %address.local181 to i32, !dbg !12
  %88 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local182), !dbg !12
  %89 = zext i32 %88 to i64, !dbg !12
  %90 = inttoptr i64 %89 to ptr addrspace(3), !dbg !12
  %address.local183 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local184 = and i64 %address.local183, 16256, !dbg !12
  %address.local185 = lshr i64 %address.local177, 4, !dbg !12
  %address.local186 = and i64 %address.local185, 56, !dbg !12
  %address.local187 = or disjoint i64 %address.local184, %address.local186, !dbg !12
  %address.local188 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %address.local187, !dbg !12
  %91 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %92 = ptrtoint ptr addrspace(1) %address.local188 to i64, !dbg !12
  %93 = sub i64 %92, %91, !dbg !12
  %dma.buffer.offset755 = trunc i64 %93 to i32, !dbg !12
  %94 = trunc i64 %91 to i32, !dbg !12
  %95 = insertelement <4 x i32> poison, i32 %94, i64 0, !dbg !12
  %96 = lshr i64 %91, 32, !dbg !12
  %97 = trunc i64 %96 to i32, !dbg !12
  %98 = insertelement <4 x i32> %95, i32 %97, i64 1, !dbg !12
  %99 = insertelement <4 x i32> %98, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc756 = insertelement <4 x i32> %99, i32 131072, i64 3, !dbg !12
  %100 = ptrtoint ptr addrspace(3) %90 to i32, !dbg !12
  %dma.buffer.lds757 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %100), !dbg !12
  %101 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds757, i32 %dma.buffer.offset755, <4 x i32> %dma.buffer.rsrc756) #10, !dbg !12
  %102 = shl nuw nsw i64 %38, 11, !dbg !17
  %103 = add nuw i64 %26, %102, !dbg !17
  %.lobit69 = ashr i64 %103, 63, !dbg !17
  %104 = xor i64 %103, %.lobit69, !dbg !17
  %105 = sdiv i64 %104, 4096, !dbg !17
  %.lobit70 = ashr i64 %103, 63, !dbg !17
  %106 = xor i64 %105, %.lobit70, !dbg !17
  %107 = add nsw i64 %106, %8, !dbg !17
  %108 = add nuw nsw i64 %42, %34, !dbg !17
  %109 = and i64 %108, 127, !dbg !17
  %110 = shl i64 %107, 19, !dbg !17
  %111 = and i64 %110, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %112 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %111, !dbg !17
  %113 = getelementptr [2 x i8], ptr addrspace(1) %112, i64 %109, !dbg !17
  %.idx71 = shl i64 %36, 8, !dbg !17
  %114 = getelementptr i8, ptr addrspace(1) %113, i64 %.idx71, !dbg !17
  %address.local189 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local143, !dbg !17
  %address.local190 = ptrtoint ptr addrspace(3) %address.local189 to i64, !dbg !17
  %address.local191 = trunc i64 %address.local190 to i32, !dbg !17
  %115 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local191), !dbg !17
  %116 = zext i32 %115 to i64, !dbg !17
  %117 = inttoptr i64 %116 to ptr addrspace(3), !dbg !17
  %address.local192 = getelementptr [2 x i8], ptr addrspace(1) %114, i64 %address.local151, !dbg !17
  %118 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %119 = ptrtoint ptr addrspace(1) %address.local192 to i64, !dbg !17
  %120 = sub i64 %119, %118, !dbg !17
  %dma.buffer.offset752 = trunc i64 %120 to i32, !dbg !17
  %121 = trunc i64 %118 to i32, !dbg !17
  %122 = insertelement <4 x i32> poison, i32 %121, i64 0, !dbg !17
  %123 = lshr i64 %118, 32, !dbg !17
  %124 = trunc i64 %123 to i32, !dbg !17
  %125 = insertelement <4 x i32> %122, i32 %124, i64 1, !dbg !17
  %126 = insertelement <4 x i32> %125, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc753 = insertelement <4 x i32> %126, i32 131072, i64 3, !dbg !17
  %127 = ptrtoint ptr addrspace(3) %117 to i32, !dbg !17
  %dma.buffer.lds754 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %127), !dbg !17
  %128 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds754, i32 %dma.buffer.offset752, <4 x i32> %dma.buffer.rsrc753) #10, !dbg !17
  %address.local193 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local155, !dbg !17
  %address.local194 = ptrtoint ptr addrspace(3) %address.local193 to i64, !dbg !17
  %address.local195 = trunc i64 %address.local194 to i32, !dbg !17
  %129 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local195), !dbg !17
  %130 = zext i32 %129 to i64, !dbg !17
  %131 = inttoptr i64 %130 to ptr addrspace(3), !dbg !17
  %address.local196 = getelementptr [2 x i8], ptr addrspace(1) %114, i64 %address.local163, !dbg !17
  %132 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %133 = ptrtoint ptr addrspace(1) %address.local196 to i64, !dbg !17
  %134 = sub i64 %133, %132, !dbg !17
  %dma.buffer.offset749 = trunc i64 %134 to i32, !dbg !17
  %135 = trunc i64 %132 to i32, !dbg !17
  %136 = insertelement <4 x i32> poison, i32 %135, i64 0, !dbg !17
  %137 = lshr i64 %132, 32, !dbg !17
  %138 = trunc i64 %137 to i32, !dbg !17
  %139 = insertelement <4 x i32> %136, i32 %138, i64 1, !dbg !17
  %140 = insertelement <4 x i32> %139, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc750 = insertelement <4 x i32> %140, i32 131072, i64 3, !dbg !17
  %141 = ptrtoint ptr addrspace(3) %131 to i32, !dbg !17
  %dma.buffer.lds751 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %141), !dbg !17
  %142 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds751, i32 %dma.buffer.offset749, <4 x i32> %dma.buffer.rsrc750) #10, !dbg !17
  %address.local197 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local167, !dbg !17
  %address.local198 = ptrtoint ptr addrspace(3) %address.local197 to i64, !dbg !17
  %address.local199 = trunc i64 %address.local198 to i32, !dbg !17
  %143 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local199), !dbg !17
  %144 = zext i32 %143 to i64, !dbg !17
  %145 = inttoptr i64 %144 to ptr addrspace(3), !dbg !17
  %address.local200 = getelementptr [2 x i8], ptr addrspace(1) %114, i64 %address.local175, !dbg !17
  %146 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %147 = ptrtoint ptr addrspace(1) %address.local200 to i64, !dbg !17
  %148 = sub i64 %147, %146, !dbg !17
  %dma.buffer.offset746 = trunc i64 %148 to i32, !dbg !17
  %149 = trunc i64 %146 to i32, !dbg !17
  %150 = insertelement <4 x i32> poison, i32 %149, i64 0, !dbg !17
  %151 = lshr i64 %146, 32, !dbg !17
  %152 = trunc i64 %151 to i32, !dbg !17
  %153 = insertelement <4 x i32> %150, i32 %152, i64 1, !dbg !17
  %154 = insertelement <4 x i32> %153, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc747 = insertelement <4 x i32> %154, i32 131072, i64 3, !dbg !17
  %155 = ptrtoint ptr addrspace(3) %145 to i32, !dbg !17
  %dma.buffer.lds748 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %155), !dbg !17
  %156 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds748, i32 %dma.buffer.offset746, <4 x i32> %dma.buffer.rsrc747) #10, !dbg !17
  %address.local201 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local179, !dbg !17
  %address.local202 = ptrtoint ptr addrspace(3) %address.local201 to i64, !dbg !17
  %address.local203 = trunc i64 %address.local202 to i32, !dbg !17
  %157 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local203), !dbg !17
  %158 = zext i32 %157 to i64, !dbg !17
  %159 = inttoptr i64 %158 to ptr addrspace(3), !dbg !17
  %address.local204 = getelementptr [2 x i8], ptr addrspace(1) %114, i64 %address.local187, !dbg !17
  %160 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %161 = ptrtoint ptr addrspace(1) %address.local204 to i64, !dbg !17
  %162 = sub i64 %161, %160, !dbg !17
  %dma.buffer.offset = trunc i64 %162 to i32, !dbg !17
  %163 = trunc i64 %160 to i32, !dbg !17
  %164 = insertelement <4 x i32> poison, i32 %163, i64 0, !dbg !17
  %165 = lshr i64 %160, 32, !dbg !17
  %166 = trunc i64 %165 to i32, !dbg !17
  %167 = insertelement <4 x i32> %164, i32 %166, i64 1, !dbg !17
  %168 = insertelement <4 x i32> %167, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %168, i32 131072, i64 3, !dbg !17
  %169 = ptrtoint ptr addrspace(3) %159 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %169), !dbg !17
  %170 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %address.local205 = lshr i32 %address.thread, 2, !dbg !13
  %address.local206 = and i32 %address.local205, 4, !dbg !13
  %address.local207 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local208 = and i32 %address.local207, 384, !dbg !13
  %address.local209 = or disjoint i32 %address.local206, %address.local208, !dbg !13
  %address.local210 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local211 = and i32 %address.local210, 120, !dbg !13
  %address.local212 = or disjoint i32 %address.local209, %address.local211, !dbg !13
  %address.local213 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local214 = and i32 %address.local213, 1024, !dbg !13
  %address.local215 = or disjoint i32 %address.local212, %address.local214, !dbg !13
  %address.local216 = zext nneg i32 %address.local215 to i64, !dbg !13
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local216, !dbg !13
  %171 = load <4 x half>, ptr addrspace(3) %address.local217, align 2, !dbg !13
  %address.local218 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local216, !dbg !13
  %address.local219 = getelementptr i8, ptr addrspace(3) %address.local218, i64 4096, !dbg !13
  %172 = load <4 x half>, ptr addrspace(3) %address.local219, align 2, !dbg !13
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local216, !dbg !13
  %address.local221 = getelementptr i8, ptr addrspace(3) %address.local220, i64 8192, !dbg !13
  %173 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !13
  %174 = fmul <4 x half> %171, splat (half 0xH3015), !dbg !13
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local216, !dbg !13
  %address.local223 = getelementptr i8, ptr addrspace(3) %address.local222, i64 12288, !dbg !13
  %175 = load <4 x half>, ptr addrspace(3) %address.local223, align 2, !dbg !13
  %176 = fmul <4 x half> %172, splat (half 0xH3015), !dbg !13
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local216, !dbg !13
  %address.local225 = getelementptr i8, ptr addrspace(3) %address.local224, i64 1024, !dbg !13
  %177 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !13
  %178 = fmul <4 x half> %173, splat (half 0xH3015), !dbg !13
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local216, !dbg !13
  %address.local227 = getelementptr i8, ptr addrspace(3) %address.local226, i64 5120, !dbg !13
  %179 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !13
  %180 = fmul <4 x half> %175, splat (half 0xH3015), !dbg !13
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local216, !dbg !13
  %address.local229 = getelementptr i8, ptr addrspace(3) %address.local228, i64 9216, !dbg !13
  %181 = load <4 x half>, ptr addrspace(3) %address.local229, align 2, !dbg !13
  %182 = fmul <4 x half> %177, splat (half 0xH3015), !dbg !13
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local216, !dbg !13
  %address.local231 = getelementptr i8, ptr addrspace(3) %address.local230, i64 13312, !dbg !13
  %183 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !13
  %184 = fmul <4 x half> %179, splat (half 0xH3015), !dbg !13
  %185 = fmul <4 x half> %181, splat (half 0xH3015), !dbg !13
  %186 = fmul <4 x half> %183, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local232 = lshr i32 %address.thread, 2, !dbg !8
  %address.local233 = and i32 %address.local232, 4, !dbg !8
  %address.local234 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local235 = and i32 %address.local234, 24, !dbg !8
  %address.local236 = or disjoint i32 %address.local233, %address.local235, !dbg !8
  %address.local237 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local238 = and i32 %address.local237, 1024, !dbg !8
  %address.local239 = or disjoint i32 %address.local236, %address.local238, !dbg !8
  %address.local240 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local241 = and i32 %address.local240, 384, !dbg !8
  %address.local242 = or disjoint i32 %address.local239, %address.local241, !dbg !8
  %address.local243 = zext nneg i32 %address.local242 to i64, !dbg !8
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local243, !dbg !8
  %187 = load <4 x half>, ptr addrspace(3) %address.local244, align 2, !dbg !8
  %188 = extractvalue [2 x <32 x float>] %39, 0, !dbg !31
  %189 = shufflevector <32 x float> %188, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %190 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %187, <4 x float> %189) #11, !dbg !8
  %address.local245 = lshr i32 %address.thread, 2, !dbg !8
  %address.local246 = and i32 %address.local245, 4, !dbg !8
  %address.local247 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local248 = and i32 %address.local247, 24, !dbg !8
  %address.local249 = or disjoint i32 %address.local246, %address.local248, !dbg !8
  %address.local250 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local251 = and i32 %address.local250, 1024, !dbg !8
  %address.local252 = or disjoint i32 %address.local249, %address.local251, !dbg !8
  %address.local253 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local254 = and i32 %address.local253, 384, !dbg !8
  %address.local255 = or disjoint i32 %address.local252, %address.local254, !dbg !8
  %address.local256 = zext nneg i32 %address.local255 to i64, !dbg !8
  %address.local257 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local256, !dbg !8
  %address.local258 = getelementptr i8, ptr addrspace(3) %address.local257, i64 4096, !dbg !8
  %191 = load <4 x half>, ptr addrspace(3) %address.local258, align 2, !dbg !8
  %192 = shufflevector <32 x float> %188, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %193 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %191, <4 x float> %190) #11, !dbg !8
  %194 = shufflevector <32 x float> %188, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %195 = shufflevector <32 x float> %188, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %196 = shufflevector <32 x float> %188, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %197 = shufflevector <32 x float> %188, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %198 = shufflevector <32 x float> %188, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %199 = shufflevector <32 x float> %188, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %200 = extractvalue [2 x <32 x float>] %39, 1, !dbg !31
  %201 = shufflevector <32 x float> %200, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %202 = shufflevector <32 x float> %200, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %203 = shufflevector <32 x float> %200, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %204 = shufflevector <32 x float> %200, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %205 = shufflevector <32 x float> %200, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %206 = shufflevector <32 x float> %200, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %207 = shufflevector <32 x float> %200, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %208 = shufflevector <32 x float> %200, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %address.local259 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local256, !dbg !8
  %address.local260 = getelementptr i8, ptr addrspace(3) %address.local259, i64 8192, !dbg !8
  %209 = load <4 x half>, ptr addrspace(3) %address.local260, align 2, !dbg !8
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local256, !dbg !8
  %address.local262 = getelementptr i8, ptr addrspace(3) %address.local261, i64 12288, !dbg !8
  %210 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !8
  %211 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %209, <4 x float> %193) #11, !dbg !8
  %address.local263 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local263, !dbg !8
  %address.local265 = lshr i64 %address.local, 2, !dbg !8
  %address.local266 = mul nsw i64 %address.local265, -512, !dbg !8
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local266, !dbg !8
  %address.local268 = shl nuw nsw i64 %address.local, 1, !dbg !8
  %address.local269 = and i64 %address.local268, 24, !dbg !8
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) %address.local267, i64 %address.local269, !dbg !8
  %address.local271 = lshr i64 %address.local, 2, !dbg !13
  %address.local272 = and i64 %address.local271, 12, !dbg !13
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local272, !dbg !8
  %address.local274 = and i64 %address.local, 63, !dbg !13
  %address.local275 = lshr i64 %address.local274, 5, !dbg !13
  %address.local276 = mul nuw nsw i64 %address.local275, 1016, !dbg !13
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) %address.local273, i64 %address.local276, !dbg !8
  %address.local278 = getelementptr i8, ptr addrspace(3) %address.local277, i64 64, !dbg !8
  %212 = load <4 x half>, ptr addrspace(3) %address.local278, align 2, !dbg !8
  %213 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %210, <4 x float> %211) #11, !dbg !8
  %address.local279 = lshr i32 %address.thread, 2, !dbg !8
  %address.local280 = and i32 %address.local279, 4, !dbg !8
  %address.local281 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local282 = and i32 %address.local281, 24, !dbg !8
  %address.local283 = or disjoint i32 %address.local280, %address.local282, !dbg !8
  %address.local284 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local285 = and i32 %address.local284, 1024, !dbg !8
  %address.local286 = or disjoint i32 %address.local283, %address.local285, !dbg !8
  %address.local287 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local288 = and i32 %address.local287, 384, !dbg !8
  %address.local289 = or disjoint i32 %address.local286, %address.local288, !dbg !8
  %address.local290 = zext nneg i32 %address.local289 to i64, !dbg !8
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local290, !dbg !8
  %address.local292 = getelementptr i8, ptr addrspace(3) %address.local291, i64 4160, !dbg !8
  %214 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !8
  %215 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %212, <4 x float> %192) #11, !dbg !8
  %216 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %214, <4 x float> %215) #11, !dbg !8
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local290, !dbg !8
  %address.local294 = getelementptr i8, ptr addrspace(3) %address.local293, i64 8256, !dbg !8
  %217 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !8
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local290, !dbg !8
  %address.local296 = getelementptr i8, ptr addrspace(3) %address.local295, i64 12352, !dbg !8
  %218 = load <4 x half>, ptr addrspace(3) %address.local296, align 2, !dbg !8
  %219 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %217, <4 x float> %216) #11, !dbg !8
  %address.local297 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local263, !dbg !8
  %address.local298 = getelementptr [2 x i8], ptr addrspace(3) %address.local297, i64 %address.local266, !dbg !8
  %address.local299 = getelementptr [2 x i8], ptr addrspace(3) %address.local298, i64 %address.local269, !dbg !8
  %address.local300 = getelementptr [2 x i8], ptr addrspace(3) %address.local299, i64 %address.local272, !dbg !8
  %address.local301 = getelementptr [2 x i8], ptr addrspace(3) %address.local300, i64 %address.local276, !dbg !8
  %address.local302 = getelementptr i8, ptr addrspace(3) %address.local301, i64 128, !dbg !8
  %220 = load <4 x half>, ptr addrspace(3) %address.local302, align 2, !dbg !8
  %221 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %218, <4 x float> %219) #11, !dbg !8
  %address.local303 = lshr i32 %address.thread, 2, !dbg !8
  %address.local304 = and i32 %address.local303, 4, !dbg !8
  %address.local305 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local306 = and i32 %address.local305, 24, !dbg !8
  %address.local307 = or disjoint i32 %address.local304, %address.local306, !dbg !8
  %address.local308 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local309 = and i32 %address.local308, 1024, !dbg !8
  %address.local310 = or disjoint i32 %address.local307, %address.local309, !dbg !8
  %address.local311 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local312 = and i32 %address.local311, 384, !dbg !8
  %address.local313 = or disjoint i32 %address.local310, %address.local312, !dbg !8
  %address.local314 = zext nneg i32 %address.local313 to i64, !dbg !8
  %address.local315 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local314, !dbg !8
  %address.local316 = getelementptr i8, ptr addrspace(3) %address.local315, i64 4224, !dbg !8
  %222 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !8
  %223 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %220, <4 x float> %194) #11, !dbg !8
  %224 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %222, <4 x float> %223) #11, !dbg !8
  %address.local317 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local314, !dbg !8
  %address.local318 = getelementptr i8, ptr addrspace(3) %address.local317, i64 8320, !dbg !8
  %225 = load <4 x half>, ptr addrspace(3) %address.local318, align 2, !dbg !8
  %address.local319 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local314, !dbg !8
  %address.local320 = getelementptr i8, ptr addrspace(3) %address.local319, i64 12416, !dbg !8
  %226 = load <4 x half>, ptr addrspace(3) %address.local320, align 2, !dbg !8
  %227 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %225, <4 x float> %224) #11, !dbg !8
  %address.local321 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local263, !dbg !8
  %address.local322 = getelementptr [2 x i8], ptr addrspace(3) %address.local321, i64 %address.local266, !dbg !8
  %address.local323 = getelementptr [2 x i8], ptr addrspace(3) %address.local322, i64 %address.local269, !dbg !8
  %address.local324 = getelementptr [2 x i8], ptr addrspace(3) %address.local323, i64 %address.local272, !dbg !8
  %address.local325 = getelementptr [2 x i8], ptr addrspace(3) %address.local324, i64 %address.local276, !dbg !8
  %address.local326 = getelementptr i8, ptr addrspace(3) %address.local325, i64 192, !dbg !8
  %228 = load <4 x half>, ptr addrspace(3) %address.local326, align 2, !dbg !8
  %229 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %226, <4 x float> %227) #11, !dbg !8
  %address.local327 = lshr i32 %address.thread, 2, !dbg !8
  %address.local328 = and i32 %address.local327, 4, !dbg !8
  %address.local329 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local330 = and i32 %address.local329, 24, !dbg !8
  %address.local331 = or disjoint i32 %address.local328, %address.local330, !dbg !8
  %address.local332 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local333 = and i32 %address.local332, 1024, !dbg !8
  %address.local334 = or disjoint i32 %address.local331, %address.local333, !dbg !8
  %address.local335 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local336 = and i32 %address.local335, 384, !dbg !8
  %address.local337 = or disjoint i32 %address.local334, %address.local336, !dbg !8
  %address.local338 = zext nneg i32 %address.local337 to i64, !dbg !8
  %address.local339 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local338, !dbg !8
  %address.local340 = getelementptr i8, ptr addrspace(3) %address.local339, i64 4288, !dbg !8
  %230 = load <4 x half>, ptr addrspace(3) %address.local340, align 2, !dbg !8
  %231 = shufflevector <4 x float> %229, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %232 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %228, <4 x float> %195) #11, !dbg !8
  %233 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %230, <4 x float> %232) #11, !dbg !8
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local338, !dbg !8
  %address.local342 = getelementptr i8, ptr addrspace(3) %address.local341, i64 8384, !dbg !8
  %234 = load <4 x half>, ptr addrspace(3) %address.local342, align 2, !dbg !8
  %address.local343 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local338, !dbg !8
  %address.local344 = getelementptr i8, ptr addrspace(3) %address.local343, i64 12480, !dbg !8
  %235 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !8
  %236 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %234, <4 x float> %233) #11, !dbg !8
  %address.local345 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local243, !dbg !8
  %address.local346 = getelementptr i8, ptr addrspace(3) %address.local345, i64 1024, !dbg !8
  %237 = load <4 x half>, ptr addrspace(3) %address.local346, align 2, !dbg !8
  %238 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %235, <4 x float> %236) #11, !dbg !8
  %address.local347 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local256, !dbg !8
  %address.local348 = getelementptr i8, ptr addrspace(3) %address.local347, i64 5120, !dbg !8
  %239 = load <4 x half>, ptr addrspace(3) %address.local348, align 2, !dbg !8
  %240 = shufflevector <4 x float> %238, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %241 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %237, <4 x float> %196) #11, !dbg !8
  %242 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %239, <4 x float> %241) #11, !dbg !8
  %address.local349 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local256, !dbg !8
  %address.local350 = getelementptr i8, ptr addrspace(3) %address.local349, i64 9216, !dbg !8
  %243 = load <4 x half>, ptr addrspace(3) %address.local350, align 2, !dbg !8
  %address.local351 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local256, !dbg !8
  %address.local352 = getelementptr i8, ptr addrspace(3) %address.local351, i64 13312, !dbg !8
  %244 = load <4 x half>, ptr addrspace(3) %address.local352, align 2, !dbg !8
  %245 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %243, <4 x float> %242) #11, !dbg !8
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local263, !dbg !8
  %address.local354 = getelementptr [2 x i8], ptr addrspace(3) %address.local353, i64 %address.local266, !dbg !8
  %address.local355 = getelementptr [2 x i8], ptr addrspace(3) %address.local354, i64 %address.local269, !dbg !8
  %address.local356 = getelementptr [2 x i8], ptr addrspace(3) %address.local355, i64 %address.local272, !dbg !8
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) %address.local356, i64 %address.local276, !dbg !8
  %address.local358 = getelementptr i8, ptr addrspace(3) %address.local357, i64 1088, !dbg !8
  %246 = load <4 x half>, ptr addrspace(3) %address.local358, align 2, !dbg !8
  %247 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %244, <4 x float> %245) #11, !dbg !8
  %address.local359 = lshr i32 %address.thread, 2, !dbg !8
  %address.local360 = and i32 %address.local359, 4, !dbg !8
  %address.local361 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local362 = and i32 %address.local361, 24, !dbg !8
  %address.local363 = or disjoint i32 %address.local360, %address.local362, !dbg !8
  %address.local364 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local365 = and i32 %address.local364, 1024, !dbg !8
  %address.local366 = or disjoint i32 %address.local363, %address.local365, !dbg !8
  %address.local367 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local368 = and i32 %address.local367, 384, !dbg !8
  %address.local369 = or disjoint i32 %address.local366, %address.local368, !dbg !8
  %address.local370 = or disjoint i32 %address.local369, 512, !dbg !8
  %address.local371 = zext nneg i32 %address.local370 to i64, !dbg !8
  %address.local372 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local371, !dbg !8
  %address.local373 = getelementptr i8, ptr addrspace(3) %address.local372, i64 4160, !dbg !8
  %248 = load <4 x half>, ptr addrspace(3) %address.local373, align 2, !dbg !8
  %249 = shufflevector <4 x float> %247, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %250 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %246, <4 x float> %197) #11, !dbg !8
  %251 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %248, <4 x float> %250) #11, !dbg !8
  %address.local374 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local371, !dbg !8
  %address.local375 = getelementptr i8, ptr addrspace(3) %address.local374, i64 8256, !dbg !8
  %252 = load <4 x half>, ptr addrspace(3) %address.local375, align 2, !dbg !8
  %address.local376 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local371, !dbg !8
  %address.local377 = getelementptr i8, ptr addrspace(3) %address.local376, i64 12352, !dbg !8
  %253 = load <4 x half>, ptr addrspace(3) %address.local377, align 2, !dbg !8
  %254 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %252, <4 x float> %251) #11, !dbg !8
  %address.local378 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local263, !dbg !8
  %address.local379 = getelementptr [2 x i8], ptr addrspace(3) %address.local378, i64 %address.local266, !dbg !8
  %address.local380 = getelementptr [2 x i8], ptr addrspace(3) %address.local379, i64 %address.local269, !dbg !8
  %address.local381 = getelementptr [2 x i8], ptr addrspace(3) %address.local380, i64 %address.local272, !dbg !8
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) %address.local381, i64 %address.local276, !dbg !8
  %address.local383 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1152, !dbg !8
  %255 = load <4 x half>, ptr addrspace(3) %address.local383, align 2, !dbg !8
  %256 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %253, <4 x float> %254) #11, !dbg !8
  %address.local384 = lshr i32 %address.thread, 2, !dbg !8
  %address.local385 = and i32 %address.local384, 4, !dbg !8
  %address.local386 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local387 = and i32 %address.local386, 24, !dbg !8
  %address.local388 = or disjoint i32 %address.local385, %address.local387, !dbg !8
  %address.local389 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local390 = and i32 %address.local389, 1024, !dbg !8
  %address.local391 = or disjoint i32 %address.local388, %address.local390, !dbg !8
  %address.local392 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local393 = and i32 %address.local392, 384, !dbg !8
  %address.local394 = or disjoint i32 %address.local391, %address.local393, !dbg !8
  %address.local395 = or disjoint i32 %address.local394, 512, !dbg !8
  %address.local396 = zext nneg i32 %address.local395 to i64, !dbg !8
  %address.local397 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local396, !dbg !8
  %address.local398 = getelementptr i8, ptr addrspace(3) %address.local397, i64 4224, !dbg !8
  %257 = load <4 x half>, ptr addrspace(3) %address.local398, align 2, !dbg !8
  %258 = shufflevector <4 x float> %256, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %259 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %255, <4 x float> %198) #11, !dbg !8
  %260 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %257, <4 x float> %259) #11, !dbg !8
  %address.local399 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local396, !dbg !8
  %address.local400 = getelementptr i8, ptr addrspace(3) %address.local399, i64 8320, !dbg !8
  %261 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !8
  %address.local401 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local396, !dbg !8
  %address.local402 = getelementptr i8, ptr addrspace(3) %address.local401, i64 12416, !dbg !8
  %262 = load <4 x half>, ptr addrspace(3) %address.local402, align 2, !dbg !8
  %263 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %261, <4 x float> %260) #11, !dbg !8
  %address.local403 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local263, !dbg !8
  %address.local404 = getelementptr [2 x i8], ptr addrspace(3) %address.local403, i64 %address.local266, !dbg !8
  %address.local405 = getelementptr [2 x i8], ptr addrspace(3) %address.local404, i64 %address.local269, !dbg !8
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) %address.local405, i64 %address.local272, !dbg !8
  %address.local407 = getelementptr [2 x i8], ptr addrspace(3) %address.local406, i64 %address.local276, !dbg !8
  %address.local408 = getelementptr i8, ptr addrspace(3) %address.local407, i64 1216, !dbg !8
  %264 = load <4 x half>, ptr addrspace(3) %address.local408, align 2, !dbg !8
  %265 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %262, <4 x float> %263) #11, !dbg !8
  %address.local409 = lshr i32 %address.thread, 2, !dbg !8
  %address.local410 = and i32 %address.local409, 4, !dbg !8
  %address.local411 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local412 = and i32 %address.local411, 24, !dbg !8
  %address.local413 = or disjoint i32 %address.local410, %address.local412, !dbg !8
  %address.local414 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local415 = and i32 %address.local414, 1024, !dbg !8
  %address.local416 = or disjoint i32 %address.local413, %address.local415, !dbg !8
  %address.local417 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local418 = and i32 %address.local417, 384, !dbg !8
  %address.local419 = or disjoint i32 %address.local416, %address.local418, !dbg !8
  %address.local420 = or disjoint i32 %address.local419, 512, !dbg !8
  %address.local421 = zext nneg i32 %address.local420 to i64, !dbg !8
  %address.local422 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local421, !dbg !8
  %address.local423 = getelementptr i8, ptr addrspace(3) %address.local422, i64 4288, !dbg !8
  %266 = load <4 x half>, ptr addrspace(3) %address.local423, align 2, !dbg !8
  %267 = shufflevector <4 x float> %265, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %268 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %264, <4 x float> %199) #11, !dbg !8
  %269 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %266, <4 x float> %268) #11, !dbg !8
  %address.local424 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local421, !dbg !8
  %address.local425 = getelementptr i8, ptr addrspace(3) %address.local424, i64 8384, !dbg !8
  %270 = load <4 x half>, ptr addrspace(3) %address.local425, align 2, !dbg !8
  %address.local426 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local421, !dbg !8
  %address.local427 = getelementptr i8, ptr addrspace(3) %address.local426, i64 12480, !dbg !8
  %271 = load <4 x half>, ptr addrspace(3) %address.local427, align 2, !dbg !8
  %272 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %270, <4 x float> %269) #11, !dbg !8
  %273 = load <4 x half>, ptr addrspace(3) %address.local244, align 2, !dbg !8
  %274 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %271, <4 x float> %272) #11, !dbg !8
  %275 = load <4 x half>, ptr addrspace(3) %address.local258, align 2, !dbg !8
  %276 = shufflevector <4 x float> %274, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %277 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %273, <4 x float> %201) #11, !dbg !8
  %278 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %275, <4 x float> %277) #11, !dbg !8
  %279 = load <4 x half>, ptr addrspace(3) %address.local260, align 2, !dbg !8
  %280 = load <4 x half>, ptr addrspace(3) %address.local262, align 2, !dbg !8
  %281 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %279, <4 x float> %278) #11, !dbg !8
  %282 = load <4 x half>, ptr addrspace(3) %address.local278, align 2, !dbg !8
  %283 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %280, <4 x float> %281) #11, !dbg !8
  %284 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !8
  %285 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %282, <4 x float> %202) #11, !dbg !8
  %286 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %284, <4 x float> %285) #11, !dbg !8
  %287 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !8
  %288 = load <4 x half>, ptr addrspace(3) %address.local296, align 2, !dbg !8
  %289 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %287, <4 x float> %286) #11, !dbg !8
  %290 = load <4 x half>, ptr addrspace(3) %address.local302, align 2, !dbg !8
  %291 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %288, <4 x float> %289) #11, !dbg !8
  %292 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !8
  %293 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %290, <4 x float> %203) #11, !dbg !8
  %294 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %292, <4 x float> %293) #11, !dbg !8
  %295 = load <4 x half>, ptr addrspace(3) %address.local318, align 2, !dbg !8
  %296 = load <4 x half>, ptr addrspace(3) %address.local320, align 2, !dbg !8
  %297 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %295, <4 x float> %294) #11, !dbg !8
  %298 = load <4 x half>, ptr addrspace(3) %address.local326, align 2, !dbg !8
  %299 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %296, <4 x float> %297) #11, !dbg !8
  %300 = load <4 x half>, ptr addrspace(3) %address.local340, align 2, !dbg !8
  %301 = shufflevector <4 x float> %299, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %302 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %298, <4 x float> %204) #11, !dbg !8
  %303 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %300, <4 x float> %302) #11, !dbg !8
  %304 = load <4 x half>, ptr addrspace(3) %address.local342, align 2, !dbg !8
  %305 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !8
  %306 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %304, <4 x float> %303) #11, !dbg !8
  %307 = load <4 x half>, ptr addrspace(3) %address.local346, align 2, !dbg !8
  %308 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %305, <4 x float> %306) #11, !dbg !8
  %309 = load <4 x half>, ptr addrspace(3) %address.local348, align 2, !dbg !8
  %310 = shufflevector <4 x float> %308, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %311 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %307, <4 x float> %205) #11, !dbg !8
  %312 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %309, <4 x float> %311) #11, !dbg !8
  %313 = load <4 x half>, ptr addrspace(3) %address.local350, align 2, !dbg !8
  %314 = load <4 x half>, ptr addrspace(3) %address.local352, align 2, !dbg !8
  %315 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %313, <4 x float> %312) #11, !dbg !8
  %316 = load <4 x half>, ptr addrspace(3) %address.local358, align 2, !dbg !8
  %317 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %314, <4 x float> %315) #11, !dbg !8
  %318 = load <4 x half>, ptr addrspace(3) %address.local373, align 2, !dbg !8
  %319 = shufflevector <4 x float> %317, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %320 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %316, <4 x float> %206) #11, !dbg !8
  %321 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %318, <4 x float> %320) #11, !dbg !8
  %322 = load <4 x half>, ptr addrspace(3) %address.local375, align 2, !dbg !8
  %323 = load <4 x half>, ptr addrspace(3) %address.local377, align 2, !dbg !8
  %324 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %322, <4 x float> %321) #11, !dbg !8
  %325 = load <4 x half>, ptr addrspace(3) %address.local383, align 2, !dbg !8
  %326 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %323, <4 x float> %324) #11, !dbg !8
  %327 = load <4 x half>, ptr addrspace(3) %address.local398, align 2, !dbg !8
  %328 = shufflevector <4 x float> %326, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %329 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %325, <4 x float> %207) #11, !dbg !8
  %330 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %327, <4 x float> %329) #11, !dbg !8
  %331 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !8
  %332 = load <4 x half>, ptr addrspace(3) %address.local402, align 2, !dbg !8
  %333 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %331, <4 x float> %330) #11, !dbg !8
  %334 = load <4 x half>, ptr addrspace(3) %address.local408, align 2, !dbg !8
  %335 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %332, <4 x float> %333) #11, !dbg !8
  %336 = load <4 x half>, ptr addrspace(3) %address.local423, align 2, !dbg !8
  %337 = shufflevector <4 x float> %335, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %338 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %334, <4 x float> %208) #11, !dbg !8
  %339 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %336, <4 x float> %338) #11, !dbg !8
  %340 = load <4 x half>, ptr addrspace(3) %address.local425, align 2, !dbg !8
  %341 = load <4 x half>, ptr addrspace(3) %address.local427, align 2, !dbg !8
  %342 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %340, <4 x float> %339) #11, !dbg !8
  %343 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %341, <4 x float> %342) #11, !dbg !8
  %344 = shufflevector <4 x float> %343, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %345 = shufflevector <4 x float> %213, <4 x float> %221, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %346 = shufflevector <32 x float> %345, <32 x float> %231, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %347 = shufflevector <32 x float> %346, <32 x float> %240, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %348 = shufflevector <32 x float> %347, <32 x float> %249, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %349 = shufflevector <32 x float> %348, <32 x float> %258, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %350 = shufflevector <32 x float> %349, <32 x float> %267, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %351 = shufflevector <32 x float> %350, <32 x float> %276, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %352 = insertvalue [2 x <32 x float>] poison, <32 x float> %351, 0, !dbg !22
  %353 = shufflevector <4 x float> %283, <4 x float> %291, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %354 = shufflevector <32 x float> %353, <32 x float> %301, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %355 = shufflevector <32 x float> %354, <32 x float> %310, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %356 = shufflevector <32 x float> %355, <32 x float> %319, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %357 = shufflevector <32 x float> %356, <32 x float> %328, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %358 = shufflevector <32 x float> %357, <32 x float> %337, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %359 = shufflevector <32 x float> %358, <32 x float> %344, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %360 = insertvalue [2 x <32 x float>] %352, <32 x float> %359, 1, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %361 = add nuw nsw i64 %38, 1, !dbg !22
  br label %37, !dbg !22

362:                                              ; preds = %37
  %363 = add nsw i64 %34, %8, !dbg !18
  %364 = shl i64 %363, 19, !dbg !18
  %365 = and i64 %364, 16252928, !dbg !18
  %366 = shl nsw i64 %36, 7, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %367 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %365, !dbg !18
  %368 = getelementptr [2 x i8], ptr addrspace(1) %367, i64 %366, !dbg !18
  %address.thread428 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !18, !range !34, !frisk.register_lifetimes !1
  %address.local429 = zext nneg i32 %address.thread428 to i64, !dbg !13
  %address.local430 = shl nuw nsw i64 %address.local429, 3, !dbg !12
  %address.local431 = and i64 %address.local430, 1536, !dbg !18
  %address.local432 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local431, !dbg !18
  %address.local433 = ptrtoint ptr addrspace(3) %address.local432 to i64, !dbg !18
  %address.local434 = trunc i64 %address.local433 to i32, !dbg !18
  %369 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local434), !dbg !18
  %370 = zext i32 %369 to i64, !dbg !18
  %371 = inttoptr i64 %370 to ptr addrspace(3), !dbg !18
  %address.local435 = shl nuw nsw i64 %address.local429, 5, !dbg !18
  %address.local436 = and i64 %address.local435, 8064, !dbg !18
  %address.local437 = shl nuw nsw i64 %address.local429, 3, !dbg !18
  %address.local438 = and i64 %address.local437, 24, !dbg !18
  %address.local439 = or disjoint i64 %address.local436, %address.local438, !dbg !18
  %address.local440 = getelementptr [2 x i8], ptr addrspace(1) %368, i64 %address.local439, !dbg !18
  %372 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %373 = ptrtoint ptr addrspace(1) %address.local440 to i64, !dbg !18
  %374 = sub i64 %373, %372, !dbg !18
  %dma.buffer.offset788 = trunc i64 %374 to i32, !dbg !18
  %375 = trunc i64 %372 to i32, !dbg !18
  %376 = insertelement <4 x i32> poison, i32 %375, i64 0, !dbg !18
  %377 = lshr i64 %372, 32, !dbg !18
  %378 = trunc i64 %377 to i32, !dbg !18
  %379 = insertelement <4 x i32> %376, i32 %378, i64 1, !dbg !18
  %380 = insertelement <4 x i32> %379, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc789 = insertelement <4 x i32> %380, i32 131072, i64 3, !dbg !18
  %381 = ptrtoint ptr addrspace(3) %371 to i32, !dbg !18
  %dma.buffer.lds790 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %381), !dbg !18
  %382 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds790, i32 %dma.buffer.offset788, <4 x i32> %dma.buffer.rsrc789) #10, !dbg !18
  %address.local441 = or disjoint i64 %address.local429, 256, !dbg !12
  %address.local442 = shl nuw nsw i64 %address.local441, 3, !dbg !12
  %address.local443 = and i64 %address.local442, 3584, !dbg !18
  %address.local444 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local443, !dbg !18
  %address.local445 = ptrtoint ptr addrspace(3) %address.local444 to i64, !dbg !18
  %address.local446 = trunc i64 %address.local445 to i32, !dbg !18
  %383 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local446), !dbg !18
  %384 = zext i32 %383 to i64, !dbg !18
  %385 = inttoptr i64 %384 to ptr addrspace(3), !dbg !18
  %address.local447 = shl nuw nsw i64 %address.local441, 5, !dbg !18
  %address.local448 = and i64 %address.local447, 16256, !dbg !18
  %address.local449 = shl nuw nsw i64 %address.local429, 3, !dbg !18
  %address.local450 = and i64 %address.local449, 24, !dbg !18
  %address.local451 = or disjoint i64 %address.local448, %address.local450, !dbg !18
  %address.local452 = getelementptr [2 x i8], ptr addrspace(1) %368, i64 %address.local451, !dbg !18
  %386 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %387 = ptrtoint ptr addrspace(1) %address.local452 to i64, !dbg !18
  %388 = sub i64 %387, %386, !dbg !18
  %dma.buffer.offset785 = trunc i64 %388 to i32, !dbg !18
  %389 = trunc i64 %386 to i32, !dbg !18
  %390 = insertelement <4 x i32> poison, i32 %389, i64 0, !dbg !18
  %391 = lshr i64 %386, 32, !dbg !18
  %392 = trunc i64 %391 to i32, !dbg !18
  %393 = insertelement <4 x i32> %390, i32 %392, i64 1, !dbg !18
  %394 = insertelement <4 x i32> %393, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc786 = insertelement <4 x i32> %394, i32 131072, i64 3, !dbg !18
  %395 = ptrtoint ptr addrspace(3) %385 to i32, !dbg !18
  %dma.buffer.lds787 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %395), !dbg !18
  %396 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds787, i32 %dma.buffer.offset785, <4 x i32> %dma.buffer.rsrc786) #10, !dbg !18
  %397 = or disjoint i64 %33, 127, !dbg !14
  %398 = icmp uge i64 %9, %397, !dbg !14
  %399 = and i1 %24, %398, !dbg !14
  br i1 %399, label %400, label %401, !dbg !14

400:                                              ; preds = %362
  br label %572, !dbg !14

401:                                              ; preds = %362
  %402 = add nsw i64 %9, %20, !dbg !14
  %403 = shl nuw nsw i64 %21, 4, !dbg !14
  %404 = add nsw i64 %402, %403, !dbg !14
  %405 = and i32 %19, 240, !dbg !14
  %index.compact = zext nneg i32 %405 to i64, !dbg !14
  %406 = sub nsw i64 %404, %index.compact, !dbg !14
  %407 = and i64 %20, 48, !dbg !14
  %408 = or disjoint i64 %33, %407, !dbg !14
  %409 = add nsw i64 %406, 1, !dbg !23
  %.not4 = icmp ugt i64 %409, %408, !dbg !24
  %410 = select i1 %.not4, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %411 = or disjoint i64 %408, 1, !dbg !14
  %.not5 = icmp ugt i64 %409, %411, !dbg !24
  %412 = select i1 %.not5, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %413 = or disjoint i64 %408, 2, !dbg !14
  %.not6 = icmp ugt i64 %409, %413, !dbg !24
  %414 = select i1 %.not6, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %415 = or disjoint i64 %408, 3, !dbg !14
  %.not7 = icmp ugt i64 %409, %415, !dbg !24
  %416 = select i1 %.not7, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %417 = or disjoint i64 %408, 4, !dbg !14
  %.not8 = icmp ugt i64 %409, %417, !dbg !24
  %418 = select i1 %.not8, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %419 = or disjoint i64 %408, 5, !dbg !14
  %.not9 = icmp ugt i64 %409, %419, !dbg !24
  %420 = select i1 %.not9, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %421 = or disjoint i64 %408, 6, !dbg !14
  %.not10 = icmp ugt i64 %409, %421, !dbg !24
  %422 = select i1 %.not10, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %423 = or disjoint i64 %408, 7, !dbg !14
  %.not11 = icmp ugt i64 %409, %423, !dbg !24
  %424 = select i1 %.not11, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %425 = or disjoint i64 %408, 8, !dbg !14
  %.not12 = icmp ugt i64 %409, %425, !dbg !24
  %426 = select i1 %.not12, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %427 = or disjoint i64 %408, 9, !dbg !14
  %.not13 = icmp ugt i64 %409, %427, !dbg !24
  %428 = select i1 %.not13, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %429 = or disjoint i64 %408, 10, !dbg !14
  %.not14 = icmp ugt i64 %409, %429, !dbg !24
  %430 = select i1 %.not14, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %431 = or disjoint i64 %408, 11, !dbg !14
  %.not15 = icmp ugt i64 %409, %431, !dbg !24
  %432 = select i1 %.not15, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %433 = or disjoint i64 %408, 12, !dbg !14
  %.not16 = icmp ugt i64 %409, %433, !dbg !24
  %434 = select i1 %.not16, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %435 = or disjoint i64 %408, 13, !dbg !14
  %.not17 = icmp ugt i64 %409, %435, !dbg !24
  %436 = select i1 %.not17, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %437 = or disjoint i64 %408, 14, !dbg !14
  %.not18 = icmp ugt i64 %409, %437, !dbg !24
  %438 = select i1 %.not18, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %439 = or disjoint i64 %408, 15, !dbg !14
  %.not19 = icmp ugt i64 %409, %439, !dbg !24
  %440 = select i1 %.not19, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %441 = or disjoint i64 %408, 64, !dbg !14
  %.not20 = icmp ugt i64 %409, %441, !dbg !24
  %442 = select i1 %.not20, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %443 = or disjoint i64 %408, 65, !dbg !14
  %.not21 = icmp ugt i64 %409, %443, !dbg !24
  %444 = select i1 %.not21, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %445 = or disjoint i64 %408, 66, !dbg !14
  %.not22 = icmp ugt i64 %409, %445, !dbg !24
  %446 = select i1 %.not22, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %447 = or disjoint i64 %408, 67, !dbg !14
  %.not23 = icmp ugt i64 %409, %447, !dbg !24
  %448 = select i1 %.not23, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %449 = or disjoint i64 %408, 68, !dbg !14
  %.not24 = icmp ugt i64 %409, %449, !dbg !24
  %450 = select i1 %.not24, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %451 = or disjoint i64 %408, 69, !dbg !14
  %.not25 = icmp ugt i64 %409, %451, !dbg !24
  %452 = select i1 %.not25, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %453 = or disjoint i64 %408, 70, !dbg !14
  %.not26 = icmp ugt i64 %409, %453, !dbg !24
  %454 = select i1 %.not26, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %455 = or disjoint i64 %408, 71, !dbg !14
  %.not27 = icmp ugt i64 %409, %455, !dbg !24
  %456 = select i1 %.not27, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %457 = or disjoint i64 %408, 72, !dbg !14
  %.not28 = icmp ugt i64 %409, %457, !dbg !24
  %458 = select i1 %.not28, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %459 = or disjoint i64 %408, 73, !dbg !14
  %.not29 = icmp ugt i64 %409, %459, !dbg !24
  %460 = select i1 %.not29, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %461 = or disjoint i64 %408, 74, !dbg !14
  %.not30 = icmp ugt i64 %409, %461, !dbg !24
  %462 = select i1 %.not30, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %463 = or disjoint i64 %408, 75, !dbg !14
  %.not31 = icmp ugt i64 %409, %463, !dbg !24
  %464 = select i1 %.not31, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %465 = or disjoint i64 %408, 76, !dbg !14
  %.not32 = icmp ugt i64 %409, %465, !dbg !24
  %466 = select i1 %.not32, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %467 = or disjoint i64 %408, 77, !dbg !14
  %.not33 = icmp ugt i64 %409, %467, !dbg !24
  %468 = select i1 %.not33, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %469 = or disjoint i64 %408, 78, !dbg !14
  %.not34 = icmp ugt i64 %409, %469, !dbg !24
  %470 = select i1 %.not34, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %471 = or disjoint i64 %408, 79, !dbg !14
  %.not35 = icmp ugt i64 %409, %471, !dbg !24
  %472 = select i1 %.not35, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %473 = add nsw i64 %406, 65, !dbg !23
  %.not36 = icmp ugt i64 %473, %408, !dbg !24
  %474 = select i1 %.not36, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not37 = icmp ugt i64 %473, %411, !dbg !24
  %475 = select i1 %.not37, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not38 = icmp ugt i64 %473, %413, !dbg !24
  %476 = select i1 %.not38, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not39 = icmp ugt i64 %473, %415, !dbg !24
  %477 = select i1 %.not39, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not40 = icmp ugt i64 %473, %417, !dbg !24
  %478 = select i1 %.not40, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not41 = icmp ugt i64 %473, %419, !dbg !24
  %479 = select i1 %.not41, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not42 = icmp ugt i64 %473, %421, !dbg !24
  %480 = select i1 %.not42, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not43 = icmp ugt i64 %473, %423, !dbg !24
  %481 = select i1 %.not43, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not44 = icmp ugt i64 %473, %425, !dbg !24
  %482 = select i1 %.not44, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not45 = icmp ugt i64 %473, %427, !dbg !24
  %483 = select i1 %.not45, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not46 = icmp ugt i64 %473, %429, !dbg !24
  %484 = select i1 %.not46, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not47 = icmp ugt i64 %473, %431, !dbg !24
  %485 = select i1 %.not47, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not48 = icmp ugt i64 %473, %433, !dbg !24
  %486 = select i1 %.not48, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not49 = icmp ugt i64 %473, %435, !dbg !24
  %487 = select i1 %.not49, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not50 = icmp ugt i64 %473, %437, !dbg !24
  %488 = select i1 %.not50, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not51 = icmp ugt i64 %473, %439, !dbg !24
  %489 = select i1 %.not51, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not52 = icmp ugt i64 %473, %441, !dbg !24
  %490 = select i1 %.not52, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not53 = icmp ugt i64 %473, %443, !dbg !24
  %491 = select i1 %.not53, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not54 = icmp ugt i64 %473, %445, !dbg !24
  %492 = select i1 %.not54, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not55 = icmp ugt i64 %473, %447, !dbg !24
  %493 = select i1 %.not55, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not56 = icmp ugt i64 %473, %449, !dbg !24
  %494 = select i1 %.not56, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not57 = icmp ugt i64 %473, %451, !dbg !24
  %495 = select i1 %.not57, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not58 = icmp ugt i64 %473, %453, !dbg !24
  %496 = select i1 %.not58, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not59 = icmp ugt i64 %473, %455, !dbg !24
  %497 = select i1 %.not59, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not60 = icmp ugt i64 %473, %457, !dbg !24
  %498 = select i1 %.not60, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not61 = icmp ugt i64 %473, %459, !dbg !24
  %499 = select i1 %.not61, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not62 = icmp ugt i64 %473, %461, !dbg !24
  %500 = select i1 %.not62, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not63 = icmp ugt i64 %473, %463, !dbg !24
  %501 = select i1 %.not63, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not64 = icmp ugt i64 %473, %465, !dbg !24
  %502 = select i1 %.not64, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not65 = icmp ugt i64 %473, %467, !dbg !24
  %503 = select i1 %.not65, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not66 = icmp ugt i64 %473, %469, !dbg !24
  %504 = select i1 %.not66, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not67 = icmp ugt i64 %473, %471, !dbg !24
  %505 = select i1 %.not67, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %506 = insertelement <32 x float> poison, float %410, i64 0, !dbg !14
  %507 = insertelement <32 x float> %506, float %412, i64 1, !dbg !14
  %508 = insertelement <32 x float> %507, float %414, i64 2, !dbg !14
  %509 = insertelement <32 x float> %508, float %416, i64 3, !dbg !14
  %510 = insertelement <32 x float> %509, float %418, i64 4, !dbg !14
  %511 = insertelement <32 x float> %510, float %420, i64 5, !dbg !14
  %512 = insertelement <32 x float> %511, float %422, i64 6, !dbg !14
  %513 = insertelement <32 x float> %512, float %424, i64 7, !dbg !14
  %514 = insertelement <32 x float> %513, float %426, i64 8, !dbg !14
  %515 = insertelement <32 x float> %514, float %428, i64 9, !dbg !14
  %516 = insertelement <32 x float> %515, float %430, i64 10, !dbg !14
  %517 = insertelement <32 x float> %516, float %432, i64 11, !dbg !14
  %518 = insertelement <32 x float> %517, float %434, i64 12, !dbg !14
  %519 = insertelement <32 x float> %518, float %436, i64 13, !dbg !14
  %520 = insertelement <32 x float> %519, float %438, i64 14, !dbg !14
  %521 = insertelement <32 x float> %520, float %440, i64 15, !dbg !14
  %522 = insertelement <32 x float> %521, float %442, i64 16, !dbg !14
  %523 = insertelement <32 x float> %522, float %444, i64 17, !dbg !14
  %524 = insertelement <32 x float> %523, float %446, i64 18, !dbg !14
  %525 = insertelement <32 x float> %524, float %448, i64 19, !dbg !14
  %526 = insertelement <32 x float> %525, float %450, i64 20, !dbg !14
  %527 = insertelement <32 x float> %526, float %452, i64 21, !dbg !14
  %528 = insertelement <32 x float> %527, float %454, i64 22, !dbg !14
  %529 = insertelement <32 x float> %528, float %456, i64 23, !dbg !14
  %530 = insertelement <32 x float> %529, float %458, i64 24, !dbg !14
  %531 = insertelement <32 x float> %530, float %460, i64 25, !dbg !14
  %532 = insertelement <32 x float> %531, float %462, i64 26, !dbg !14
  %533 = insertelement <32 x float> %532, float %464, i64 27, !dbg !14
  %534 = insertelement <32 x float> %533, float %466, i64 28, !dbg !14
  %535 = insertelement <32 x float> %534, float %468, i64 29, !dbg !14
  %536 = insertelement <32 x float> %535, float %470, i64 30, !dbg !14
  %537 = insertelement <32 x float> %536, float %472, i64 31, !dbg !14
  %538 = insertvalue [2 x <32 x float>] poison, <32 x float> %537, 0, !dbg !14
  %539 = insertelement <32 x float> poison, float %474, i64 0, !dbg !14
  %540 = insertelement <32 x float> %539, float %475, i64 1, !dbg !14
  %541 = insertelement <32 x float> %540, float %476, i64 2, !dbg !14
  %542 = insertelement <32 x float> %541, float %477, i64 3, !dbg !14
  %543 = insertelement <32 x float> %542, float %478, i64 4, !dbg !14
  %544 = insertelement <32 x float> %543, float %479, i64 5, !dbg !14
  %545 = insertelement <32 x float> %544, float %480, i64 6, !dbg !14
  %546 = insertelement <32 x float> %545, float %481, i64 7, !dbg !14
  %547 = insertelement <32 x float> %546, float %482, i64 8, !dbg !14
  %548 = insertelement <32 x float> %547, float %483, i64 9, !dbg !14
  %549 = insertelement <32 x float> %548, float %484, i64 10, !dbg !14
  %550 = insertelement <32 x float> %549, float %485, i64 11, !dbg !14
  %551 = insertelement <32 x float> %550, float %486, i64 12, !dbg !14
  %552 = insertelement <32 x float> %551, float %487, i64 13, !dbg !14
  %553 = insertelement <32 x float> %552, float %488, i64 14, !dbg !14
  %554 = insertelement <32 x float> %553, float %489, i64 15, !dbg !14
  %555 = insertelement <32 x float> %554, float %490, i64 16, !dbg !14
  %556 = insertelement <32 x float> %555, float %491, i64 17, !dbg !14
  %557 = insertelement <32 x float> %556, float %492, i64 18, !dbg !14
  %558 = insertelement <32 x float> %557, float %493, i64 19, !dbg !14
  %559 = insertelement <32 x float> %558, float %494, i64 20, !dbg !14
  %560 = insertelement <32 x float> %559, float %495, i64 21, !dbg !14
  %561 = insertelement <32 x float> %560, float %496, i64 22, !dbg !14
  %562 = insertelement <32 x float> %561, float %497, i64 23, !dbg !14
  %563 = insertelement <32 x float> %562, float %498, i64 24, !dbg !14
  %564 = insertelement <32 x float> %563, float %499, i64 25, !dbg !14
  %565 = insertelement <32 x float> %564, float %500, i64 26, !dbg !14
  %566 = insertelement <32 x float> %565, float %501, i64 27, !dbg !14
  %567 = insertelement <32 x float> %566, float %502, i64 28, !dbg !14
  %568 = insertelement <32 x float> %567, float %503, i64 29, !dbg !14
  %569 = insertelement <32 x float> %568, float %504, i64 30, !dbg !14
  %570 = insertelement <32 x float> %569, float %505, i64 31, !dbg !14
  %571 = insertvalue [2 x <32 x float>] %538, <32 x float> %570, 1, !dbg !14
  br label %572, !dbg !14

572:                                              ; preds = %400, %401
  %573 = phi [2 x <32 x float>] [ %571, %401 ], [ zeroinitializer, %400 ], !dbg !14
  br label %574, !dbg !14

574:                                              ; preds = %572
  %575 = extractvalue [2 x <32 x float>] %39, 0, !dbg !26
  %576 = extractvalue [2 x <32 x float>] %573, 0, !dbg !26
  %577 = shufflevector <32 x float> %575, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %578 = shufflevector <32 x float> %576, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %579 = fadd <4 x float> %577, %578, !dbg !26
  %580 = extractelement <4 x float> %579, i64 0, !dbg !27
  %581 = call float @llvm.amdgcn.exp2.f32(float %580), !dbg !27
  %582 = insertelement <4 x float> poison, float %581, i64 0, !dbg !27
  %583 = extractelement <4 x float> %579, i64 1, !dbg !27
  %584 = call float @llvm.amdgcn.exp2.f32(float %583), !dbg !27
  %585 = fadd float %581, %584, !dbg !28
  %586 = insertelement <4 x float> %582, float %584, i64 1, !dbg !27
  %587 = extractelement <4 x float> %579, i64 2, !dbg !27
  %588 = call float @llvm.amdgcn.exp2.f32(float %587), !dbg !27
  %589 = fadd float %585, %588, !dbg !28
  %590 = insertelement <4 x float> %586, float %588, i64 2, !dbg !27
  %591 = extractelement <4 x float> %579, i64 3, !dbg !27
  %592 = call float @llvm.amdgcn.exp2.f32(float %591), !dbg !27
  %593 = fadd float %589, %592, !dbg !28
  %594 = insertelement <4 x float> %590, float %592, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %594) #9, !dbg !29, !frisk.register_lifetimes !1
  %595 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %595) #9, !dbg !26, !frisk.register_lifetimes !1
  %596 = shufflevector <32 x float> %575, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %597 = shufflevector <32 x float> %576, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %598 = fadd <4 x float> %596, %597, !dbg !26
  %599 = extractelement <4 x float> %598, i64 0, !dbg !27
  %600 = call float @llvm.amdgcn.exp2.f32(float %599), !dbg !27
  %601 = fadd float %593, %600, !dbg !28
  %602 = insertelement <4 x float> poison, float %600, i64 0, !dbg !27
  %603 = extractelement <4 x float> %598, i64 1, !dbg !27
  %604 = call float @llvm.amdgcn.exp2.f32(float %603), !dbg !27
  %605 = fadd float %601, %604, !dbg !28
  %606 = insertelement <4 x float> %602, float %604, i64 1, !dbg !27
  %607 = extractelement <4 x float> %598, i64 2, !dbg !27
  %608 = call float @llvm.amdgcn.exp2.f32(float %607), !dbg !27
  %609 = fadd float %605, %608, !dbg !28
  %610 = insertelement <4 x float> %606, float %608, i64 2, !dbg !27
  %611 = extractelement <4 x float> %598, i64 3, !dbg !27
  %612 = call float @llvm.amdgcn.exp2.f32(float %611), !dbg !27
  %613 = fadd float %609, %612, !dbg !28
  %614 = insertelement <4 x float> %610, float %612, i64 3, !dbg !27
  %math.result112 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %614) #9, !dbg !29, !frisk.register_lifetimes !1
  %615 = fptrunc <4 x float> %math.result112 to <4 x half>, !dbg !29
  %math.packed113 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %615) #9, !dbg !26, !frisk.register_lifetimes !1
  %616 = shufflevector <32 x float> %575, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %617 = shufflevector <32 x float> %576, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %618 = fadd <4 x float> %616, %617, !dbg !26
  %619 = extractelement <4 x float> %618, i64 0, !dbg !27
  %620 = call float @llvm.amdgcn.exp2.f32(float %619), !dbg !27
  %621 = fadd float %613, %620, !dbg !28
  %622 = insertelement <4 x float> poison, float %620, i64 0, !dbg !27
  %623 = extractelement <4 x float> %618, i64 1, !dbg !27
  %624 = call float @llvm.amdgcn.exp2.f32(float %623), !dbg !27
  %625 = fadd float %621, %624, !dbg !28
  %626 = insertelement <4 x float> %622, float %624, i64 1, !dbg !27
  %627 = extractelement <4 x float> %618, i64 2, !dbg !27
  %628 = call float @llvm.amdgcn.exp2.f32(float %627), !dbg !27
  %629 = fadd float %625, %628, !dbg !28
  %630 = insertelement <4 x float> %626, float %628, i64 2, !dbg !27
  %631 = extractelement <4 x float> %618, i64 3, !dbg !27
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !27
  %633 = fadd float %629, %632, !dbg !28
  %634 = insertelement <4 x float> %630, float %632, i64 3, !dbg !27
  %math.result114 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %634) #9, !dbg !29, !frisk.register_lifetimes !1
  %635 = fptrunc <4 x float> %math.result114 to <4 x half>, !dbg !29
  %math.packed115 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %635) #9, !dbg !26, !frisk.register_lifetimes !1
  %636 = shufflevector <32 x float> %575, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %637 = shufflevector <32 x float> %576, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %638 = fadd <4 x float> %636, %637, !dbg !26
  %639 = extractelement <4 x float> %638, i64 0, !dbg !27
  %640 = call float @llvm.amdgcn.exp2.f32(float %639), !dbg !27
  %641 = fadd float %633, %640, !dbg !28
  %642 = insertelement <4 x float> poison, float %640, i64 0, !dbg !27
  %643 = extractelement <4 x float> %638, i64 1, !dbg !27
  %644 = call float @llvm.amdgcn.exp2.f32(float %643), !dbg !27
  %645 = fadd float %641, %644, !dbg !28
  %646 = insertelement <4 x float> %642, float %644, i64 1, !dbg !27
  %647 = extractelement <4 x float> %638, i64 2, !dbg !27
  %648 = call float @llvm.amdgcn.exp2.f32(float %647), !dbg !27
  %649 = fadd float %645, %648, !dbg !28
  %650 = insertelement <4 x float> %646, float %648, i64 2, !dbg !27
  %651 = extractelement <4 x float> %638, i64 3, !dbg !27
  %652 = call float @llvm.amdgcn.exp2.f32(float %651), !dbg !27
  %653 = fadd float %649, %652, !dbg !28
  %654 = insertelement <4 x float> %650, float %652, i64 3, !dbg !27
  %math.result116 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %654) #9, !dbg !29, !frisk.register_lifetimes !1
  %655 = fptrunc <4 x float> %math.result116 to <4 x half>, !dbg !29
  %math.packed117 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %655) #9, !dbg !26, !frisk.register_lifetimes !1
  %656 = shufflevector <32 x float> %575, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %657 = shufflevector <32 x float> %576, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %658 = fadd <4 x float> %656, %657, !dbg !26
  %659 = extractelement <4 x float> %658, i64 0, !dbg !27
  %660 = call float @llvm.amdgcn.exp2.f32(float %659), !dbg !27
  %661 = fadd float %653, %660, !dbg !28
  %662 = insertelement <4 x float> poison, float %660, i64 0, !dbg !27
  %663 = extractelement <4 x float> %658, i64 1, !dbg !27
  %664 = call float @llvm.amdgcn.exp2.f32(float %663), !dbg !27
  %665 = fadd float %661, %664, !dbg !28
  %666 = insertelement <4 x float> %662, float %664, i64 1, !dbg !27
  %667 = extractelement <4 x float> %658, i64 2, !dbg !27
  %668 = call float @llvm.amdgcn.exp2.f32(float %667), !dbg !27
  %669 = fadd float %665, %668, !dbg !28
  %670 = insertelement <4 x float> %666, float %668, i64 2, !dbg !27
  %671 = extractelement <4 x float> %658, i64 3, !dbg !27
  %672 = call float @llvm.amdgcn.exp2.f32(float %671), !dbg !27
  %673 = fadd float %669, %672, !dbg !28
  %674 = insertelement <4 x float> %670, float %672, i64 3, !dbg !27
  %math.result118 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %674) #9, !dbg !29, !frisk.register_lifetimes !1
  %675 = fptrunc <4 x float> %math.result118 to <4 x half>, !dbg !29
  %math.packed119 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %675) #9, !dbg !26, !frisk.register_lifetimes !1
  %676 = shufflevector <32 x float> %575, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %677 = shufflevector <32 x float> %576, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %678 = fadd <4 x float> %676, %677, !dbg !26
  %679 = extractelement <4 x float> %678, i64 0, !dbg !27
  %680 = call float @llvm.amdgcn.exp2.f32(float %679), !dbg !27
  %681 = fadd float %673, %680, !dbg !28
  %682 = insertelement <4 x float> poison, float %680, i64 0, !dbg !27
  %683 = extractelement <4 x float> %678, i64 1, !dbg !27
  %684 = call float @llvm.amdgcn.exp2.f32(float %683), !dbg !27
  %685 = fadd float %681, %684, !dbg !28
  %686 = insertelement <4 x float> %682, float %684, i64 1, !dbg !27
  %687 = extractelement <4 x float> %678, i64 2, !dbg !27
  %688 = call float @llvm.amdgcn.exp2.f32(float %687), !dbg !27
  %689 = fadd float %685, %688, !dbg !28
  %690 = insertelement <4 x float> %686, float %688, i64 2, !dbg !27
  %691 = extractelement <4 x float> %678, i64 3, !dbg !27
  %692 = call float @llvm.amdgcn.exp2.f32(float %691), !dbg !27
  %693 = fadd float %689, %692, !dbg !28
  %694 = insertelement <4 x float> %690, float %692, i64 3, !dbg !27
  %math.result120 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %694) #9, !dbg !29, !frisk.register_lifetimes !1
  %695 = fptrunc <4 x float> %math.result120 to <4 x half>, !dbg !29
  %math.packed121 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %695) #9, !dbg !26, !frisk.register_lifetimes !1
  %696 = shufflevector <32 x float> %575, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %697 = shufflevector <32 x float> %576, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %698 = fadd <4 x float> %696, %697, !dbg !26
  %699 = extractelement <4 x float> %698, i64 0, !dbg !27
  %700 = call float @llvm.amdgcn.exp2.f32(float %699), !dbg !27
  %701 = fadd float %693, %700, !dbg !28
  %702 = insertelement <4 x float> poison, float %700, i64 0, !dbg !27
  %703 = extractelement <4 x float> %698, i64 1, !dbg !27
  %704 = call float @llvm.amdgcn.exp2.f32(float %703), !dbg !27
  %705 = fadd float %701, %704, !dbg !28
  %706 = insertelement <4 x float> %702, float %704, i64 1, !dbg !27
  %707 = extractelement <4 x float> %698, i64 2, !dbg !27
  %708 = call float @llvm.amdgcn.exp2.f32(float %707), !dbg !27
  %709 = fadd float %705, %708, !dbg !28
  %710 = insertelement <4 x float> %706, float %708, i64 2, !dbg !27
  %711 = extractelement <4 x float> %698, i64 3, !dbg !27
  %712 = call float @llvm.amdgcn.exp2.f32(float %711), !dbg !27
  %713 = fadd float %709, %712, !dbg !28
  %714 = insertelement <4 x float> %710, float %712, i64 3, !dbg !27
  %math.result122 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %714) #9, !dbg !29, !frisk.register_lifetimes !1
  %715 = fptrunc <4 x float> %math.result122 to <4 x half>, !dbg !29
  %math.packed123 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %715) #9, !dbg !26, !frisk.register_lifetimes !1
  %716 = shufflevector <32 x float> %575, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %717 = shufflevector <32 x float> %576, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %718 = fadd <4 x float> %716, %717, !dbg !26
  %719 = extractelement <4 x float> %718, i64 0, !dbg !27
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !27
  %721 = fadd float %713, %720, !dbg !28
  %722 = insertelement <4 x float> poison, float %720, i64 0, !dbg !27
  %723 = extractelement <4 x float> %718, i64 1, !dbg !27
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !27
  %725 = fadd float %721, %724, !dbg !28
  %726 = insertelement <4 x float> %722, float %724, i64 1, !dbg !27
  %727 = extractelement <4 x float> %718, i64 2, !dbg !27
  %728 = call float @llvm.amdgcn.exp2.f32(float %727), !dbg !27
  %729 = fadd float %725, %728, !dbg !28
  %730 = insertelement <4 x float> %726, float %728, i64 2, !dbg !27
  %731 = extractelement <4 x float> %718, i64 3, !dbg !27
  %732 = call float @llvm.amdgcn.exp2.f32(float %731), !dbg !27
  %733 = fadd float %729, %732, !dbg !28
  %734 = insertelement <4 x float> %730, float %732, i64 3, !dbg !27
  %math.result124 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %734) #9, !dbg !29, !frisk.register_lifetimes !1
  %735 = fptrunc <4 x float> %math.result124 to <4 x half>, !dbg !29
  %math.packed125 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %735) #9, !dbg !26, !frisk.register_lifetimes !1
  %736 = extractvalue [2 x <32 x float>] %39, 1, !dbg !26
  %737 = extractvalue [2 x <32 x float>] %573, 1, !dbg !26
  %738 = shufflevector <32 x float> %736, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %739 = shufflevector <32 x float> %737, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %740 = fadd <4 x float> %738, %739, !dbg !26
  %741 = extractelement <4 x float> %740, i64 0, !dbg !27
  %742 = call float @llvm.amdgcn.exp2.f32(float %741), !dbg !27
  %743 = insertelement <4 x float> poison, float %742, i64 0, !dbg !27
  %744 = extractelement <4 x float> %740, i64 1, !dbg !27
  %745 = call float @llvm.amdgcn.exp2.f32(float %744), !dbg !27
  %746 = fadd float %742, %745, !dbg !28
  %747 = insertelement <4 x float> %743, float %745, i64 1, !dbg !27
  %748 = extractelement <4 x float> %740, i64 2, !dbg !27
  %749 = call float @llvm.amdgcn.exp2.f32(float %748), !dbg !27
  %750 = fadd float %746, %749, !dbg !28
  %751 = insertelement <4 x float> %747, float %749, i64 2, !dbg !27
  %752 = extractelement <4 x float> %740, i64 3, !dbg !27
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !27
  %754 = fadd float %750, %753, !dbg !28
  %755 = insertelement <4 x float> %751, float %753, i64 3, !dbg !27
  %math.result126 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %755) #9, !dbg !29, !frisk.register_lifetimes !1
  %756 = fptrunc <4 x float> %math.result126 to <4 x half>, !dbg !29
  %math.packed127 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %756) #9, !dbg !26, !frisk.register_lifetimes !1
  %757 = shufflevector <32 x float> %736, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %758 = shufflevector <32 x float> %737, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %759 = fadd <4 x float> %757, %758, !dbg !26
  %760 = extractelement <4 x float> %759, i64 0, !dbg !27
  %761 = call float @llvm.amdgcn.exp2.f32(float %760), !dbg !27
  %762 = fadd float %754, %761, !dbg !28
  %763 = insertelement <4 x float> poison, float %761, i64 0, !dbg !27
  %764 = extractelement <4 x float> %759, i64 1, !dbg !27
  %765 = call float @llvm.amdgcn.exp2.f32(float %764), !dbg !27
  %766 = fadd float %762, %765, !dbg !28
  %767 = insertelement <4 x float> %763, float %765, i64 1, !dbg !27
  %768 = extractelement <4 x float> %759, i64 2, !dbg !27
  %769 = call float @llvm.amdgcn.exp2.f32(float %768), !dbg !27
  %770 = fadd float %766, %769, !dbg !28
  %771 = insertelement <4 x float> %767, float %769, i64 2, !dbg !27
  %772 = extractelement <4 x float> %759, i64 3, !dbg !27
  %773 = call float @llvm.amdgcn.exp2.f32(float %772), !dbg !27
  %774 = fadd float %770, %773, !dbg !28
  %775 = insertelement <4 x float> %771, float %773, i64 3, !dbg !27
  %math.result128 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %775) #9, !dbg !29, !frisk.register_lifetimes !1
  %776 = fptrunc <4 x float> %math.result128 to <4 x half>, !dbg !29
  %math.packed129 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %776) #9, !dbg !26, !frisk.register_lifetimes !1
  %777 = shufflevector <32 x float> %736, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %778 = shufflevector <32 x float> %737, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %779 = fadd <4 x float> %777, %778, !dbg !26
  %780 = extractelement <4 x float> %779, i64 0, !dbg !27
  %781 = call float @llvm.amdgcn.exp2.f32(float %780), !dbg !27
  %782 = fadd float %774, %781, !dbg !28
  %783 = insertelement <4 x float> poison, float %781, i64 0, !dbg !27
  %784 = extractelement <4 x float> %779, i64 1, !dbg !27
  %785 = call float @llvm.amdgcn.exp2.f32(float %784), !dbg !27
  %786 = fadd float %782, %785, !dbg !28
  %787 = insertelement <4 x float> %783, float %785, i64 1, !dbg !27
  %788 = extractelement <4 x float> %779, i64 2, !dbg !27
  %789 = call float @llvm.amdgcn.exp2.f32(float %788), !dbg !27
  %790 = fadd float %786, %789, !dbg !28
  %791 = insertelement <4 x float> %787, float %789, i64 2, !dbg !27
  %792 = extractelement <4 x float> %779, i64 3, !dbg !27
  %793 = call float @llvm.amdgcn.exp2.f32(float %792), !dbg !27
  %794 = fadd float %790, %793, !dbg !28
  %795 = insertelement <4 x float> %791, float %793, i64 3, !dbg !27
  %math.result130 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %795) #9, !dbg !29, !frisk.register_lifetimes !1
  %796 = fptrunc <4 x float> %math.result130 to <4 x half>, !dbg !29
  %math.packed131 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %796) #9, !dbg !26, !frisk.register_lifetimes !1
  %797 = shufflevector <32 x float> %736, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %798 = shufflevector <32 x float> %737, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %799 = fadd <4 x float> %797, %798, !dbg !26
  %800 = extractelement <4 x float> %799, i64 0, !dbg !27
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !27
  %802 = fadd float %794, %801, !dbg !28
  %803 = insertelement <4 x float> poison, float %801, i64 0, !dbg !27
  %804 = extractelement <4 x float> %799, i64 1, !dbg !27
  %805 = call float @llvm.amdgcn.exp2.f32(float %804), !dbg !27
  %806 = fadd float %802, %805, !dbg !28
  %807 = insertelement <4 x float> %803, float %805, i64 1, !dbg !27
  %808 = extractelement <4 x float> %799, i64 2, !dbg !27
  %809 = call float @llvm.amdgcn.exp2.f32(float %808), !dbg !27
  %810 = fadd float %806, %809, !dbg !28
  %811 = insertelement <4 x float> %807, float %809, i64 2, !dbg !27
  %812 = extractelement <4 x float> %799, i64 3, !dbg !27
  %813 = call float @llvm.amdgcn.exp2.f32(float %812), !dbg !27
  %814 = fadd float %810, %813, !dbg !28
  %815 = insertelement <4 x float> %811, float %813, i64 3, !dbg !27
  %math.result132 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %815) #9, !dbg !29, !frisk.register_lifetimes !1
  %816 = fptrunc <4 x float> %math.result132 to <4 x half>, !dbg !29
  %math.packed133 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %816) #9, !dbg !26, !frisk.register_lifetimes !1
  %817 = shufflevector <32 x float> %736, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %818 = shufflevector <32 x float> %737, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %819 = fadd <4 x float> %817, %818, !dbg !26
  %820 = extractelement <4 x float> %819, i64 0, !dbg !27
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !27
  %822 = fadd float %814, %821, !dbg !28
  %823 = insertelement <4 x float> poison, float %821, i64 0, !dbg !27
  %824 = extractelement <4 x float> %819, i64 1, !dbg !27
  %825 = call float @llvm.amdgcn.exp2.f32(float %824), !dbg !27
  %826 = fadd float %822, %825, !dbg !28
  %827 = insertelement <4 x float> %823, float %825, i64 1, !dbg !27
  %828 = extractelement <4 x float> %819, i64 2, !dbg !27
  %829 = call float @llvm.amdgcn.exp2.f32(float %828), !dbg !27
  %830 = fadd float %826, %829, !dbg !28
  %831 = insertelement <4 x float> %827, float %829, i64 2, !dbg !27
  %832 = extractelement <4 x float> %819, i64 3, !dbg !27
  %833 = call float @llvm.amdgcn.exp2.f32(float %832), !dbg !27
  %834 = fadd float %830, %833, !dbg !28
  %835 = insertelement <4 x float> %831, float %833, i64 3, !dbg !27
  %math.result134 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %835) #9, !dbg !29, !frisk.register_lifetimes !1
  %836 = fptrunc <4 x float> %math.result134 to <4 x half>, !dbg !29
  %math.packed135 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %836) #9, !dbg !26, !frisk.register_lifetimes !1
  %837 = shufflevector <32 x float> %736, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %838 = shufflevector <32 x float> %737, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %839 = fadd <4 x float> %837, %838, !dbg !26
  %840 = extractelement <4 x float> %839, i64 0, !dbg !27
  %841 = call float @llvm.amdgcn.exp2.f32(float %840), !dbg !27
  %842 = fadd float %834, %841, !dbg !28
  %843 = insertelement <4 x float> poison, float %841, i64 0, !dbg !27
  %844 = extractelement <4 x float> %839, i64 1, !dbg !27
  %845 = call float @llvm.amdgcn.exp2.f32(float %844), !dbg !27
  %846 = fadd float %842, %845, !dbg !28
  %847 = insertelement <4 x float> %843, float %845, i64 1, !dbg !27
  %848 = extractelement <4 x float> %839, i64 2, !dbg !27
  %849 = call float @llvm.amdgcn.exp2.f32(float %848), !dbg !27
  %850 = fadd float %846, %849, !dbg !28
  %851 = insertelement <4 x float> %847, float %849, i64 2, !dbg !27
  %852 = extractelement <4 x float> %839, i64 3, !dbg !27
  %853 = call float @llvm.amdgcn.exp2.f32(float %852), !dbg !27
  %854 = fadd float %850, %853, !dbg !28
  %855 = insertelement <4 x float> %851, float %853, i64 3, !dbg !27
  %math.result136 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %855) #9, !dbg !29, !frisk.register_lifetimes !1
  %856 = fptrunc <4 x float> %math.result136 to <4 x half>, !dbg !29
  %math.packed137 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %856) #9, !dbg !26, !frisk.register_lifetimes !1
  %857 = shufflevector <32 x float> %736, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %858 = shufflevector <32 x float> %737, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %859 = fadd <4 x float> %857, %858, !dbg !26
  %860 = extractelement <4 x float> %859, i64 0, !dbg !27
  %861 = call float @llvm.amdgcn.exp2.f32(float %860), !dbg !27
  %862 = fadd float %854, %861, !dbg !28
  %863 = insertelement <4 x float> poison, float %861, i64 0, !dbg !27
  %864 = extractelement <4 x float> %859, i64 1, !dbg !27
  %865 = call float @llvm.amdgcn.exp2.f32(float %864), !dbg !27
  %866 = fadd float %862, %865, !dbg !28
  %867 = insertelement <4 x float> %863, float %865, i64 1, !dbg !27
  %868 = extractelement <4 x float> %859, i64 2, !dbg !27
  %869 = call float @llvm.amdgcn.exp2.f32(float %868), !dbg !27
  %870 = fadd float %866, %869, !dbg !28
  %871 = insertelement <4 x float> %867, float %869, i64 2, !dbg !27
  %872 = extractelement <4 x float> %859, i64 3, !dbg !27
  %873 = call float @llvm.amdgcn.exp2.f32(float %872), !dbg !27
  %874 = fadd float %870, %873, !dbg !28
  %875 = insertelement <4 x float> %871, float %873, i64 3, !dbg !27
  %math.result138 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %875) #9, !dbg !29, !frisk.register_lifetimes !1
  %876 = fptrunc <4 x float> %math.result138 to <4 x half>, !dbg !29
  %math.packed139 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %876) #9, !dbg !26, !frisk.register_lifetimes !1
  %877 = shufflevector <32 x float> %736, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %878 = shufflevector <32 x float> %737, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %879 = fadd <4 x float> %877, %878, !dbg !26
  %880 = extractelement <4 x float> %879, i64 0, !dbg !27
  %881 = call float @llvm.amdgcn.exp2.f32(float %880), !dbg !27
  %882 = fadd float %874, %881, !dbg !28
  %883 = insertelement <4 x float> poison, float %881, i64 0, !dbg !27
  %884 = extractelement <4 x float> %879, i64 1, !dbg !27
  %885 = call float @llvm.amdgcn.exp2.f32(float %884), !dbg !27
  %886 = fadd float %882, %885, !dbg !28
  %887 = insertelement <4 x float> %883, float %885, i64 1, !dbg !27
  %888 = extractelement <4 x float> %879, i64 2, !dbg !27
  %889 = call float @llvm.amdgcn.exp2.f32(float %888), !dbg !27
  %890 = fadd float %886, %889, !dbg !28
  %891 = insertelement <4 x float> %887, float %889, i64 2, !dbg !27
  %892 = extractelement <4 x float> %879, i64 3, !dbg !27
  %893 = call float @llvm.amdgcn.exp2.f32(float %892), !dbg !27
  %894 = fadd float %890, %893, !dbg !28
  %895 = insertelement <4 x float> %891, float %893, i64 3, !dbg !27
  %math.result140 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %895) #9, !dbg !29, !frisk.register_lifetimes !1
  %896 = fptrunc <4 x float> %math.result140 to <4 x half>, !dbg !29
  %math.packed141 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %896) #9, !dbg !28, !frisk.register_lifetimes !1
  %897 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %898 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %897), !dbg !28
  %899 = add nuw nsw i32 %898, 64, !dbg !28
  %900 = xor i32 %898, 16, !dbg !28
  %901 = and i32 %899, 192, !dbg !28
  %902 = icmp samesign ult i32 %900, %901, !dbg !28
  %903 = select i1 %902, i32 %900, i32 %898, !dbg !28
  %904 = shl nuw nsw i32 %903, 2, !dbg !28
  %905 = bitcast float %733 to i32, !dbg !28
  %906 = call i32 @llvm.amdgcn.ds.bpermute(i32 %904, i32 %905), !dbg !28
  %907 = bitcast i32 %906 to float, !dbg !28
  %908 = fadd float %733, %907, !dbg !28
  %909 = xor i32 %898, 32, !dbg !28
  %910 = icmp samesign ult i32 %909, %901, !dbg !28
  %911 = select i1 %910, i32 %909, i32 %898, !dbg !28
  %912 = shl nuw nsw i32 %911, 2, !dbg !28
  %913 = bitcast float %908 to i32, !dbg !28
  %914 = call i32 @llvm.amdgcn.ds.bpermute(i32 %912, i32 %913), !dbg !28
  %915 = bitcast i32 %914 to float, !dbg !28
  %916 = fadd float %908, %915, !dbg !28
  %917 = extractvalue [2 x <1 x float>] %28, 0, !dbg !28
  %918 = extractelement <1 x float> %917, i64 0, !dbg !28
  %919 = fadd float %916, %918, !dbg !28
  %920 = insertelement <1 x float> poison, float %919, i64 0, !dbg !16
  %921 = insertvalue [2 x <1 x float>] poison, <1 x float> %920, 0, !dbg !16
  %922 = bitcast float %894 to i32, !dbg !28
  %923 = call i32 @llvm.amdgcn.ds.bpermute(i32 %904, i32 %922), !dbg !28
  %924 = bitcast i32 %923 to float, !dbg !28
  %925 = fadd float %894, %924, !dbg !28
  %926 = bitcast float %925 to i32, !dbg !28
  %927 = call i32 @llvm.amdgcn.ds.bpermute(i32 %912, i32 %926), !dbg !28
  %928 = bitcast i32 %927 to float, !dbg !28
  %929 = fadd float %925, %928, !dbg !28
  %930 = extractvalue [2 x <1 x float>] %28, 1, !dbg !28
  %931 = extractelement <1 x float> %930, i64 0, !dbg !28
  %932 = fadd float %929, %931, !dbg !28
  %933 = insertelement <1 x float> poison, float %932, i64 0, !dbg !16
  %934 = insertvalue [2 x <1 x float>] %921, <1 x float> %933, 1, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread453 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !15, !range !34, !frisk.register_lifetimes !1
  %address.local454 = shl nuw nsw i32 %address.thread453, 3, !dbg !15
  %address.local455 = and i32 %address.local454, 120, !dbg !15
  %address.local456 = shl nuw nsw i32 %address.thread453, 5, !dbg !15
  %address.local457 = and i32 %address.local456, 1536, !dbg !15
  %address.local458 = or disjoint i32 %address.local455, %address.local457, !dbg !15
  %address.local459 = zext nneg i32 %address.local458 to i64, !dbg !15
  %address.local460 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local459, !dbg !15
  %935 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local460, i32 0) #0, !dbg !15
  %936 = bitcast <4 x i32> %935 to <8 x half>, !dbg !15
  %address.local461 = zext nneg i32 %address.thread453 to i64, !dbg !13
  %address.local462 = and i64 %address.local461, 60, !dbg !15
  %address.local463 = add nuw nsw i64 %address.local462, %address.local461, !dbg !15
  %address.local464 = shl nuw nsw i64 %address.local463, 3, !dbg !15
  %address.local465 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local464, !dbg !15
  %address.local466 = and i64 %address.local461, 63, !dbg !13
  %address.local467 = lshr i64 %address.local466, 4, !dbg !15
  %address.local468 = mul nuw nsw i64 %address.local467, 384, !dbg !15
  %address.local469 = getelementptr [2 x i8], ptr addrspace(3) %address.local465, i64 %address.local468, !dbg !15
  %address.local470 = lshr i64 %address.local461, 2, !dbg !8
  %address.local471 = mul nsw i64 %address.local470, -32, !dbg !15
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) %address.local469, i64 %address.local471, !dbg !15
  %address.local473 = getelementptr i8, ptr addrspace(3) %address.local472, i64 256, !dbg !15
  %937 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local473, i32 0) #0, !dbg !15
  %938 = bitcast <4 x i32> %937 to <8 x half>, !dbg !15
  %address.local474 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local464, !dbg !15
  %address.local475 = getelementptr [2 x i8], ptr addrspace(3) %address.local474, i64 %address.local468, !dbg !15
  %address.local476 = getelementptr [2 x i8], ptr addrspace(3) %address.local475, i64 %address.local471, !dbg !15
  %address.local477 = getelementptr i8, ptr addrspace(3) %address.local476, i64 512, !dbg !15
  %939 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local477, i32 0) #0, !dbg !15
  %940 = bitcast <4 x i32> %939 to <8 x half>, !dbg !15
  %address.local478 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local464, !dbg !15
  %address.local479 = getelementptr [2 x i8], ptr addrspace(3) %address.local478, i64 %address.local468, !dbg !15
  %address.local480 = getelementptr [2 x i8], ptr addrspace(3) %address.local479, i64 %address.local471, !dbg !15
  %address.local481 = getelementptr i8, ptr addrspace(3) %address.local480, i64 768, !dbg !15
  %941 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local481, i32 0) #0, !dbg !15
  %942 = bitcast <4 x i32> %941 to <8 x half>, !dbg !15
  %address.local482 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local459, !dbg !15
  %address.local483 = getelementptr i8, ptr addrspace(3) %address.local482, i64 4096, !dbg !15
  %943 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local483, i32 0) #0, !dbg !15
  %944 = bitcast <4 x i32> %943 to <8 x half>, !dbg !15
  %address.local484 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local464, !dbg !15
  %address.local485 = getelementptr [2 x i8], ptr addrspace(3) %address.local484, i64 %address.local468, !dbg !15
  %address.local486 = getelementptr [2 x i8], ptr addrspace(3) %address.local485, i64 %address.local471, !dbg !15
  %address.local487 = getelementptr i8, ptr addrspace(3) %address.local486, i64 4352, !dbg !15
  %945 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local487, i32 0) #0, !dbg !15
  %946 = bitcast <4 x i32> %945 to <8 x half>, !dbg !15
  %address.local488 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local464, !dbg !15
  %address.local489 = getelementptr [2 x i8], ptr addrspace(3) %address.local488, i64 %address.local468, !dbg !15
  %address.local490 = getelementptr [2 x i8], ptr addrspace(3) %address.local489, i64 %address.local471, !dbg !15
  %address.local491 = getelementptr i8, ptr addrspace(3) %address.local490, i64 4608, !dbg !15
  %947 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local491, i32 0) #0, !dbg !15
  %948 = bitcast <4 x i32> %947 to <8 x half>, !dbg !15
  %address.local492 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local464, !dbg !15
  %address.local493 = getelementptr [2 x i8], ptr addrspace(3) %address.local492, i64 %address.local468, !dbg !15
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) %address.local493, i64 %address.local471, !dbg !15
  %address.local495 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4864, !dbg !15
  %949 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local495, i32 0) #0, !dbg !15
  %950 = bitcast <4 x i32> %949 to <8 x half>, !dbg !15
  %951 = add nsw i64 %34, %8, !dbg !18
  %952 = shl i64 %951, 19, !dbg !18
  %953 = and i64 %952, 16252928, !dbg !18
  %954 = add i64 %953, %366, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %955 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %954, !dbg !18
  %956 = getelementptr i8, ptr addrspace(1) %955, i64 64, !dbg !18
  %address.local496 = shl nuw nsw i64 %address.local461, 3, !dbg !12
  %address.local497 = and i64 %address.local496, 1536, !dbg !18
  %address.local498 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local497, !dbg !18
  %address.local499 = ptrtoint ptr addrspace(3) %address.local498 to i64, !dbg !18
  %address.local500 = trunc i64 %address.local499 to i32, !dbg !18
  %957 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local500), !dbg !18
  %958 = zext i32 %957 to i64, !dbg !18
  %959 = inttoptr i64 %958 to ptr addrspace(3), !dbg !18
  %address.local501 = shl nuw nsw i64 %address.local461, 5, !dbg !18
  %address.local502 = and i64 %address.local501, 8064, !dbg !18
  %address.local503 = shl nuw nsw i64 %address.local461, 3, !dbg !18
  %address.local504 = and i64 %address.local503, 24, !dbg !18
  %address.local505 = or disjoint i64 %address.local502, %address.local504, !dbg !18
  %address.local506 = getelementptr [2 x i8], ptr addrspace(1) %956, i64 %address.local505, !dbg !18
  %960 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %961 = ptrtoint ptr addrspace(1) %address.local506 to i64, !dbg !18
  %962 = sub i64 %961, %960, !dbg !18
  %dma.buffer.offset782 = trunc i64 %962 to i32, !dbg !18
  %963 = trunc i64 %960 to i32, !dbg !18
  %964 = insertelement <4 x i32> poison, i32 %963, i64 0, !dbg !18
  %965 = lshr i64 %960, 32, !dbg !18
  %966 = trunc i64 %965 to i32, !dbg !18
  %967 = insertelement <4 x i32> %964, i32 %966, i64 1, !dbg !18
  %968 = insertelement <4 x i32> %967, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc783 = insertelement <4 x i32> %968, i32 131072, i64 3, !dbg !18
  %969 = ptrtoint ptr addrspace(3) %959 to i32, !dbg !18
  %dma.buffer.lds784 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %969), !dbg !18
  %970 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds784, i32 %dma.buffer.offset782, <4 x i32> %dma.buffer.rsrc783) #10, !dbg !18
  %address.local507 = or disjoint i64 %address.local461, 256, !dbg !12
  %address.local508 = shl nuw nsw i64 %address.local507, 3, !dbg !12
  %address.local509 = and i64 %address.local508, 3584, !dbg !18
  %address.local510 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local509, !dbg !18
  %address.local511 = ptrtoint ptr addrspace(3) %address.local510 to i64, !dbg !18
  %address.local512 = trunc i64 %address.local511 to i32, !dbg !18
  %971 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local512), !dbg !18
  %972 = zext i32 %971 to i64, !dbg !18
  %973 = inttoptr i64 %972 to ptr addrspace(3), !dbg !18
  %address.local513 = shl nuw nsw i64 %address.local507, 5, !dbg !18
  %address.local514 = and i64 %address.local513, 16256, !dbg !18
  %address.local515 = shl nuw nsw i64 %address.local461, 3, !dbg !18
  %address.local516 = and i64 %address.local515, 24, !dbg !18
  %address.local517 = or disjoint i64 %address.local514, %address.local516, !dbg !18
  %address.local518 = getelementptr [2 x i8], ptr addrspace(1) %956, i64 %address.local517, !dbg !18
  %974 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %975 = ptrtoint ptr addrspace(1) %address.local518 to i64, !dbg !18
  %976 = sub i64 %975, %974, !dbg !18
  %dma.buffer.offset779 = trunc i64 %976 to i32, !dbg !18
  %977 = trunc i64 %974 to i32, !dbg !18
  %978 = insertelement <4 x i32> poison, i32 %977, i64 0, !dbg !18
  %979 = lshr i64 %974, 32, !dbg !18
  %980 = trunc i64 %979 to i32, !dbg !18
  %981 = insertelement <4 x i32> %978, i32 %980, i64 1, !dbg !18
  %982 = insertelement <4 x i32> %981, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc780 = insertelement <4 x i32> %982, i32 131072, i64 3, !dbg !18
  %983 = ptrtoint ptr addrspace(3) %973 to i32, !dbg !18
  %dma.buffer.lds781 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %983), !dbg !18
  %984 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds781, i32 %dma.buffer.offset779, <4 x i32> %dma.buffer.rsrc780) #10, !dbg !18
  %985 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %986 = extractvalue [2 x <8 x float>] %27, 0, !dbg !30
  %987 = shufflevector <8 x float> %986, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %988 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %985, <4 x float> %987) #11, !dbg !15
  %989 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %990 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %989, <4 x float> %988) #11, !dbg !15
  %991 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %992 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %991, <4 x float> %990) #11, !dbg !15
  %993 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %994 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %993, <4 x float> %992) #11, !dbg !15
  %995 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %996 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %995, <4 x float> %994) #11, !dbg !15
  %997 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %998 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %997, <4 x float> %996) #11, !dbg !15
  %999 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1000 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %999, <4 x float> %998) #11, !dbg !15
  %1001 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1002 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1001, <4 x float> %1000) #11, !dbg !15
  %1003 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1004 = shufflevector <8 x float> %986, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1005 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1003, <4 x float> %1004) #11, !dbg !15
  %1006 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1007 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1006, <4 x float> %1005) #11, !dbg !15
  %1008 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1009 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1008, <4 x float> %1007) #11, !dbg !15
  %1010 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1011 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1010, <4 x float> %1009) #11, !dbg !15
  %1012 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1013 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1012, <4 x float> %1011) #11, !dbg !15
  %1014 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1015 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1014, <4 x float> %1013) #11, !dbg !15
  %1016 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1017 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1016, <4 x float> %1015) #11, !dbg !15
  %1018 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1019 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1018, <4 x float> %1017) #11, !dbg !15
  %1020 = extractvalue [2 x <8 x float>] %27, 1, !dbg !30
  %1021 = shufflevector <8 x float> %1020, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1022 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %985, <4 x float> %1021) #11, !dbg !15
  %1023 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %989, <4 x float> %1022) #11, !dbg !15
  %1024 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %991, <4 x float> %1023) #11, !dbg !15
  %1025 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %993, <4 x float> %1024) #11, !dbg !15
  %1026 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %995, <4 x float> %1025) #11, !dbg !15
  %1027 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %997, <4 x float> %1026) #11, !dbg !15
  %1028 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed139, <4 x half> %999, <4 x float> %1027) #11, !dbg !15
  %1029 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed141, <4 x half> %1001, <4 x float> %1028) #11, !dbg !15
  %1030 = shufflevector <8 x float> %1020, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1031 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1003, <4 x float> %1030) #11, !dbg !15
  %1032 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1006, <4 x float> %1031) #11, !dbg !15
  %1033 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %1008, <4 x float> %1032) #11, !dbg !15
  %1034 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %1010, <4 x float> %1033) #11, !dbg !15
  %1035 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %1012, <4 x float> %1034) #11, !dbg !15
  %1036 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %1014, <4 x float> %1035) #11, !dbg !15
  %1037 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed139, <4 x half> %1016, <4 x float> %1036) #11, !dbg !15
  %1038 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed141, <4 x half> %1018, <4 x float> %1037) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1039 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local460, i32 0) #0, !dbg !15
  %1040 = bitcast <4 x i32> %1039 to <8 x half>, !dbg !15
  %1041 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local473, i32 0) #0, !dbg !15
  %1042 = bitcast <4 x i32> %1041 to <8 x half>, !dbg !15
  %1043 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local477, i32 0) #0, !dbg !15
  %1044 = bitcast <4 x i32> %1043 to <8 x half>, !dbg !15
  %1045 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local481, i32 0) #0, !dbg !15
  %1046 = bitcast <4 x i32> %1045 to <8 x half>, !dbg !15
  %1047 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local483, i32 0) #0, !dbg !15
  %1048 = bitcast <4 x i32> %1047 to <8 x half>, !dbg !15
  %1049 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local487, i32 0) #0, !dbg !15
  %1050 = bitcast <4 x i32> %1049 to <8 x half>, !dbg !15
  %1051 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local491, i32 0) #0, !dbg !15
  %1052 = bitcast <4 x i32> %1051 to <8 x half>, !dbg !15
  %1053 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local495, i32 0) #0, !dbg !15
  %1054 = bitcast <4 x i32> %1053 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1055 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %954, !dbg !18
  %1056 = getelementptr i8, ptr addrspace(1) %1055, i64 128, !dbg !18
  %1057 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local500), !dbg !18
  %1058 = zext i32 %1057 to i64, !dbg !18
  %1059 = inttoptr i64 %1058 to ptr addrspace(3), !dbg !18
  %address.local519 = getelementptr [2 x i8], ptr addrspace(1) %1056, i64 %address.local505, !dbg !18
  %1060 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1061 = ptrtoint ptr addrspace(1) %address.local519 to i64, !dbg !18
  %1062 = sub i64 %1061, %1060, !dbg !18
  %dma.buffer.offset776 = trunc i64 %1062 to i32, !dbg !18
  %1063 = trunc i64 %1060 to i32, !dbg !18
  %1064 = insertelement <4 x i32> poison, i32 %1063, i64 0, !dbg !18
  %1065 = lshr i64 %1060, 32, !dbg !18
  %1066 = trunc i64 %1065 to i32, !dbg !18
  %1067 = insertelement <4 x i32> %1064, i32 %1066, i64 1, !dbg !18
  %1068 = insertelement <4 x i32> %1067, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc777 = insertelement <4 x i32> %1068, i32 131072, i64 3, !dbg !18
  %1069 = ptrtoint ptr addrspace(3) %1059 to i32, !dbg !18
  %dma.buffer.lds778 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1069), !dbg !18
  %1070 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds778, i32 %dma.buffer.offset776, <4 x i32> %dma.buffer.rsrc777) #10, !dbg !18
  %1071 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local512), !dbg !18
  %1072 = zext i32 %1071 to i64, !dbg !18
  %1073 = inttoptr i64 %1072 to ptr addrspace(3), !dbg !18
  %address.local520 = getelementptr [2 x i8], ptr addrspace(1) %1056, i64 %address.local517, !dbg !18
  %1074 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1075 = ptrtoint ptr addrspace(1) %address.local520 to i64, !dbg !18
  %1076 = sub i64 %1075, %1074, !dbg !18
  %dma.buffer.offset773 = trunc i64 %1076 to i32, !dbg !18
  %1077 = trunc i64 %1074 to i32, !dbg !18
  %1078 = insertelement <4 x i32> poison, i32 %1077, i64 0, !dbg !18
  %1079 = lshr i64 %1074, 32, !dbg !18
  %1080 = trunc i64 %1079 to i32, !dbg !18
  %1081 = insertelement <4 x i32> %1078, i32 %1080, i64 1, !dbg !18
  %1082 = insertelement <4 x i32> %1081, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc774 = insertelement <4 x i32> %1082, i32 131072, i64 3, !dbg !18
  %1083 = ptrtoint ptr addrspace(3) %1073 to i32, !dbg !18
  %dma.buffer.lds775 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1083), !dbg !18
  %1084 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds775, i32 %dma.buffer.offset773, <4 x i32> %dma.buffer.rsrc774) #10, !dbg !18
  %1085 = shufflevector <8 x half> %1040, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1086 = extractvalue [2 x <8 x float>] %29, 0, !dbg !30
  %1087 = shufflevector <8 x float> %1086, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1088 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1085, <4 x float> %1087) #11, !dbg !15
  %1089 = shufflevector <8 x half> %1042, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1090 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1089, <4 x float> %1088) #11, !dbg !15
  %1091 = shufflevector <8 x half> %1044, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1092 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1091, <4 x float> %1090) #11, !dbg !15
  %1093 = shufflevector <8 x half> %1046, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1094 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1093, <4 x float> %1092) #11, !dbg !15
  %1095 = shufflevector <8 x half> %1048, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1096 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1095, <4 x float> %1094) #11, !dbg !15
  %1097 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1098 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1097, <4 x float> %1096) #11, !dbg !15
  %1099 = shufflevector <8 x half> %1052, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1100 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1099, <4 x float> %1098) #11, !dbg !15
  %1101 = shufflevector <8 x half> %1054, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1102 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1101, <4 x float> %1100) #11, !dbg !15
  %1103 = shufflevector <8 x half> %1040, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1104 = shufflevector <8 x float> %1086, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1105 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1103, <4 x float> %1104) #11, !dbg !15
  %1106 = shufflevector <8 x half> %1042, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1107 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1106, <4 x float> %1105) #11, !dbg !15
  %1108 = shufflevector <8 x half> %1044, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1109 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1108, <4 x float> %1107) #11, !dbg !15
  %1110 = shufflevector <8 x half> %1046, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1111 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1110, <4 x float> %1109) #11, !dbg !15
  %1112 = shufflevector <8 x half> %1048, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1113 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1112, <4 x float> %1111) #11, !dbg !15
  %1114 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1115 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1114, <4 x float> %1113) #11, !dbg !15
  %1116 = shufflevector <8 x half> %1052, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1117 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1116, <4 x float> %1115) #11, !dbg !15
  %1118 = shufflevector <8 x half> %1054, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1119 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1118, <4 x float> %1117) #11, !dbg !15
  %1120 = extractvalue [2 x <8 x float>] %29, 1, !dbg !30
  %1121 = shufflevector <8 x float> %1120, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1122 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1085, <4 x float> %1121) #11, !dbg !15
  %1123 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1089, <4 x float> %1122) #11, !dbg !15
  %1124 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %1091, <4 x float> %1123) #11, !dbg !15
  %1125 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %1093, <4 x float> %1124) #11, !dbg !15
  %1126 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %1095, <4 x float> %1125) #11, !dbg !15
  %1127 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %1097, <4 x float> %1126) #11, !dbg !15
  %1128 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed139, <4 x half> %1099, <4 x float> %1127) #11, !dbg !15
  %1129 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed141, <4 x half> %1101, <4 x float> %1128) #11, !dbg !15
  %1130 = shufflevector <8 x float> %1120, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1131 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1103, <4 x float> %1130) #11, !dbg !15
  %1132 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1106, <4 x float> %1131) #11, !dbg !15
  %1133 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %1108, <4 x float> %1132) #11, !dbg !15
  %1134 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %1110, <4 x float> %1133) #11, !dbg !15
  %1135 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %1112, <4 x float> %1134) #11, !dbg !15
  %1136 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %1114, <4 x float> %1135) #11, !dbg !15
  %1137 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed139, <4 x half> %1116, <4 x float> %1136) #11, !dbg !15
  %1138 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed141, <4 x half> %1118, <4 x float> %1137) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1139 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local460, i32 0) #0, !dbg !15
  %1140 = bitcast <4 x i32> %1139 to <8 x half>, !dbg !15
  %1141 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local473, i32 0) #0, !dbg !15
  %1142 = bitcast <4 x i32> %1141 to <8 x half>, !dbg !15
  %1143 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local477, i32 0) #0, !dbg !15
  %1144 = bitcast <4 x i32> %1143 to <8 x half>, !dbg !15
  %1145 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local481, i32 0) #0, !dbg !15
  %1146 = bitcast <4 x i32> %1145 to <8 x half>, !dbg !15
  %1147 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local483, i32 0) #0, !dbg !15
  %1148 = bitcast <4 x i32> %1147 to <8 x half>, !dbg !15
  %1149 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local487, i32 0) #0, !dbg !15
  %1150 = bitcast <4 x i32> %1149 to <8 x half>, !dbg !15
  %1151 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local491, i32 0) #0, !dbg !15
  %1152 = bitcast <4 x i32> %1151 to <8 x half>, !dbg !15
  %1153 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local495, i32 0) #0, !dbg !15
  %1154 = bitcast <4 x i32> %1153 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1155 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %954, !dbg !18
  %1156 = getelementptr i8, ptr addrspace(1) %1155, i64 192, !dbg !18
  %1157 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local500), !dbg !18
  %1158 = zext i32 %1157 to i64, !dbg !18
  %1159 = inttoptr i64 %1158 to ptr addrspace(3), !dbg !18
  %address.local521 = getelementptr [2 x i8], ptr addrspace(1) %1156, i64 %address.local505, !dbg !18
  %1160 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1161 = ptrtoint ptr addrspace(1) %address.local521 to i64, !dbg !18
  %1162 = sub i64 %1161, %1160, !dbg !18
  %dma.buffer.offset770 = trunc i64 %1162 to i32, !dbg !18
  %1163 = trunc i64 %1160 to i32, !dbg !18
  %1164 = insertelement <4 x i32> poison, i32 %1163, i64 0, !dbg !18
  %1165 = lshr i64 %1160, 32, !dbg !18
  %1166 = trunc i64 %1165 to i32, !dbg !18
  %1167 = insertelement <4 x i32> %1164, i32 %1166, i64 1, !dbg !18
  %1168 = insertelement <4 x i32> %1167, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc771 = insertelement <4 x i32> %1168, i32 131072, i64 3, !dbg !18
  %1169 = ptrtoint ptr addrspace(3) %1159 to i32, !dbg !18
  %dma.buffer.lds772 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1169), !dbg !18
  %1170 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds772, i32 %dma.buffer.offset770, <4 x i32> %dma.buffer.rsrc771) #10, !dbg !18
  %1171 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local512), !dbg !18
  %1172 = zext i32 %1171 to i64, !dbg !18
  %1173 = inttoptr i64 %1172 to ptr addrspace(3), !dbg !18
  %address.local522 = getelementptr [2 x i8], ptr addrspace(1) %1156, i64 %address.local517, !dbg !18
  %1174 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1175 = ptrtoint ptr addrspace(1) %address.local522 to i64, !dbg !18
  %1176 = sub i64 %1175, %1174, !dbg !18
  %dma.buffer.offset767 = trunc i64 %1176 to i32, !dbg !18
  %1177 = trunc i64 %1174 to i32, !dbg !18
  %1178 = insertelement <4 x i32> poison, i32 %1177, i64 0, !dbg !18
  %1179 = lshr i64 %1174, 32, !dbg !18
  %1180 = trunc i64 %1179 to i32, !dbg !18
  %1181 = insertelement <4 x i32> %1178, i32 %1180, i64 1, !dbg !18
  %1182 = insertelement <4 x i32> %1181, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc768 = insertelement <4 x i32> %1182, i32 131072, i64 3, !dbg !18
  %1183 = ptrtoint ptr addrspace(3) %1173 to i32, !dbg !18
  %dma.buffer.lds769 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1183), !dbg !18
  %1184 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds769, i32 %dma.buffer.offset767, <4 x i32> %dma.buffer.rsrc768) #10, !dbg !18
  %1185 = shufflevector <8 x half> %1140, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1186 = extractvalue [2 x <8 x float>] %30, 0, !dbg !30
  %1187 = shufflevector <8 x float> %1186, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1188 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1185, <4 x float> %1187) #11, !dbg !15
  %1189 = shufflevector <8 x half> %1142, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1190 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1189, <4 x float> %1188) #11, !dbg !15
  %1191 = shufflevector <8 x half> %1144, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1192 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1191, <4 x float> %1190) #11, !dbg !15
  %1193 = shufflevector <8 x half> %1146, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1194 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1193, <4 x float> %1192) #11, !dbg !15
  %1195 = shufflevector <8 x half> %1148, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1196 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1195, <4 x float> %1194) #11, !dbg !15
  %1197 = shufflevector <8 x half> %1150, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1198 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1197, <4 x float> %1196) #11, !dbg !15
  %1199 = shufflevector <8 x half> %1152, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1200 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1199, <4 x float> %1198) #11, !dbg !15
  %1201 = shufflevector <8 x half> %1154, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1202 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1201, <4 x float> %1200) #11, !dbg !15
  %1203 = shufflevector <8 x half> %1140, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1204 = shufflevector <8 x float> %1186, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1205 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1203, <4 x float> %1204) #11, !dbg !15
  %1206 = shufflevector <8 x half> %1142, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1207 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1206, <4 x float> %1205) #11, !dbg !15
  %1208 = shufflevector <8 x half> %1144, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1209 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1208, <4 x float> %1207) #11, !dbg !15
  %1210 = shufflevector <8 x half> %1146, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1211 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1210, <4 x float> %1209) #11, !dbg !15
  %1212 = shufflevector <8 x half> %1148, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1213 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1212, <4 x float> %1211) #11, !dbg !15
  %1214 = shufflevector <8 x half> %1150, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1215 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1214, <4 x float> %1213) #11, !dbg !15
  %1216 = shufflevector <8 x half> %1152, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1217 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1216, <4 x float> %1215) #11, !dbg !15
  %1218 = shufflevector <8 x half> %1154, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1219 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1218, <4 x float> %1217) #11, !dbg !15
  %1220 = extractvalue [2 x <8 x float>] %30, 1, !dbg !30
  %1221 = shufflevector <8 x float> %1220, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1222 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1185, <4 x float> %1221) #11, !dbg !15
  %1223 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1189, <4 x float> %1222) #11, !dbg !15
  %1224 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %1191, <4 x float> %1223) #11, !dbg !15
  %1225 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %1193, <4 x float> %1224) #11, !dbg !15
  %1226 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %1195, <4 x float> %1225) #11, !dbg !15
  %1227 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %1197, <4 x float> %1226) #11, !dbg !15
  %1228 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed139, <4 x half> %1199, <4 x float> %1227) #11, !dbg !15
  %1229 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed141, <4 x half> %1201, <4 x float> %1228) #11, !dbg !15
  %1230 = shufflevector <8 x float> %1220, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1231 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1203, <4 x float> %1230) #11, !dbg !15
  %1232 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1206, <4 x float> %1231) #11, !dbg !15
  %1233 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %1208, <4 x float> %1232) #11, !dbg !15
  %1234 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %1210, <4 x float> %1233) #11, !dbg !15
  %1235 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %1212, <4 x float> %1234) #11, !dbg !15
  %1236 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %1214, <4 x float> %1235) #11, !dbg !15
  %1237 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed139, <4 x half> %1216, <4 x float> %1236) #11, !dbg !15
  %1238 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed141, <4 x half> %1218, <4 x float> %1237) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1239 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local460, i32 0) #0, !dbg !15
  %1240 = bitcast <4 x i32> %1239 to <8 x half>, !dbg !15
  %1241 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local473, i32 0) #0, !dbg !15
  %1242 = bitcast <4 x i32> %1241 to <8 x half>, !dbg !15
  %1243 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local477, i32 0) #0, !dbg !15
  %1244 = bitcast <4 x i32> %1243 to <8 x half>, !dbg !15
  %1245 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local481, i32 0) #0, !dbg !15
  %1246 = bitcast <4 x i32> %1245 to <8 x half>, !dbg !15
  %1247 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local483, i32 0) #0, !dbg !15
  %1248 = bitcast <4 x i32> %1247 to <8 x half>, !dbg !15
  %1249 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local487, i32 0) #0, !dbg !15
  %1250 = bitcast <4 x i32> %1249 to <8 x half>, !dbg !15
  %1251 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local491, i32 0) #0, !dbg !15
  %1252 = bitcast <4 x i32> %1251 to <8 x half>, !dbg !15
  %1253 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local495, i32 0) #0, !dbg !15
  %1254 = bitcast <4 x i32> %1253 to <8 x half>, !dbg !15
  %1255 = shufflevector <8 x half> %1240, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1256 = extractvalue [2 x <8 x float>] %31, 0, !dbg !30
  %1257 = shufflevector <8 x float> %1256, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1258 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1255, <4 x float> %1257) #11, !dbg !15
  %1259 = shufflevector <8 x half> %1242, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1260 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1259, <4 x float> %1258) #11, !dbg !15
  %1261 = shufflevector <8 x half> %1244, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1262 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1261, <4 x float> %1260) #11, !dbg !15
  %1263 = shufflevector <8 x half> %1246, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1264 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1263, <4 x float> %1262) #11, !dbg !15
  %1265 = shufflevector <8 x half> %1248, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1266 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1265, <4 x float> %1264) #11, !dbg !15
  %1267 = shufflevector <8 x half> %1250, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1268 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1267, <4 x float> %1266) #11, !dbg !15
  %1269 = shufflevector <8 x half> %1252, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1270 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1269, <4 x float> %1268) #11, !dbg !15
  %1271 = shufflevector <8 x half> %1254, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1272 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1271, <4 x float> %1270) #11, !dbg !15
  %1273 = shufflevector <8 x half> %1240, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1274 = shufflevector <8 x float> %1256, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1275 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1273, <4 x float> %1274) #11, !dbg !15
  %1276 = shufflevector <8 x half> %1242, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1277 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1276, <4 x float> %1275) #11, !dbg !15
  %1278 = shufflevector <8 x half> %1244, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1279 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1278, <4 x float> %1277) #11, !dbg !15
  %1280 = shufflevector <8 x half> %1246, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1281 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1280, <4 x float> %1279) #11, !dbg !15
  %1282 = shufflevector <8 x half> %1248, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1283 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1282, <4 x float> %1281) #11, !dbg !15
  %1284 = shufflevector <8 x half> %1250, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1285 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1284, <4 x float> %1283) #11, !dbg !15
  %1286 = shufflevector <8 x half> %1252, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1287 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1286, <4 x float> %1285) #11, !dbg !15
  %1288 = shufflevector <8 x half> %1254, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1289 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1288, <4 x float> %1287) #11, !dbg !15
  %1290 = extractvalue [2 x <8 x float>] %31, 1, !dbg !30
  %1291 = shufflevector <8 x float> %1290, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1292 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1255, <4 x float> %1291) #11, !dbg !15
  %1293 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1259, <4 x float> %1292) #11, !dbg !15
  %1294 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %1261, <4 x float> %1293) #11, !dbg !15
  %1295 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %1263, <4 x float> %1294) #11, !dbg !15
  %1296 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %1265, <4 x float> %1295) #11, !dbg !15
  %1297 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %1267, <4 x float> %1296) #11, !dbg !15
  %1298 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed139, <4 x half> %1269, <4 x float> %1297) #11, !dbg !15
  %1299 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed141, <4 x half> %1271, <4 x float> %1298) #11, !dbg !15
  %1300 = shufflevector <8 x float> %1290, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1301 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1273, <4 x float> %1300) #11, !dbg !15
  %1302 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1276, <4 x float> %1301) #11, !dbg !15
  %1303 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %1278, <4 x float> %1302) #11, !dbg !15
  %1304 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %1280, <4 x float> %1303) #11, !dbg !15
  %1305 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %1282, <4 x float> %1304) #11, !dbg !15
  %1306 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %1284, <4 x float> %1305) #11, !dbg !15
  %1307 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed139, <4 x half> %1286, <4 x float> %1306) #11, !dbg !15
  %1308 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed141, <4 x half> %1288, <4 x float> %1307) #11, !dbg !15
  %1309 = shufflevector <4 x float> %1002, <4 x float> %1019, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1310 = insertvalue [2 x <8 x float>] poison, <8 x float> %1309, 0, !dbg !16
  %1311 = shufflevector <4 x float> %1029, <4 x float> %1038, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1312 = insertvalue [2 x <8 x float>] %1310, <8 x float> %1311, 1, !dbg !16
  %1313 = shufflevector <4 x float> %1102, <4 x float> %1119, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1314 = insertvalue [2 x <8 x float>] poison, <8 x float> %1313, 0, !dbg !16
  %1315 = shufflevector <4 x float> %1129, <4 x float> %1138, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1316 = insertvalue [2 x <8 x float>] %1314, <8 x float> %1315, 1, !dbg !16
  %1317 = shufflevector <4 x float> %1202, <4 x float> %1219, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1318 = insertvalue [2 x <8 x float>] poison, <8 x float> %1317, 0, !dbg !16
  %1319 = shufflevector <4 x float> %1229, <4 x float> %1238, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1320 = insertvalue [2 x <8 x float>] %1318, <8 x float> %1319, 1, !dbg !16
  %1321 = shufflevector <4 x float> %1272, <4 x float> %1289, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1322 = insertvalue [2 x <8 x float>] poison, <8 x float> %1321, 0, !dbg !16
  %1323 = shufflevector <4 x float> %1299, <4 x float> %1308, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1324 = insertvalue [2 x <8 x float>] %1322, <8 x float> %1323, 1, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1325 = add nuw nsw i64 %26, 1, !dbg !16
  br label %25, !dbg !16

1326:                                             ; preds = %25
  %1327 = and i32 %19, 16, !dbg !16
  %.not2 = icmp eq i32 %1327, 0, !dbg !16
  %1328 = and i32 %19, 32, !dbg !16
  %.not3 = icmp eq i32 %1328, 0, !dbg !16
  %1329 = extractvalue [2 x <8 x float>] %31, 0, !dbg !16
  %1330 = extractelement <8 x float> %1329, i64 0, !dbg !16
  %1331 = extractelement <8 x float> %1329, i64 1, !dbg !16
  %1332 = extractelement <8 x float> %1329, i64 2, !dbg !16
  %1333 = extractelement <8 x float> %1329, i64 3, !dbg !16
  %1334 = select i1 %.not2, float %1331, float %1330, !dbg !16
  %1335 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %1336 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1335), !dbg !16
  %1337 = add nuw nsw i32 %1336, 64, !dbg !16
  %1338 = xor i32 %1336, 16, !dbg !16
  %1339 = and i32 %1337, 192, !dbg !16
  %1340 = icmp samesign ult i32 %1338, %1339, !dbg !16
  %1341 = select i1 %1340, i32 %1338, i32 %1336, !dbg !16
  %1342 = shl nuw nsw i32 %1341, 2, !dbg !16
  %1343 = bitcast float %1334 to i32, !dbg !16
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1343), !dbg !16
  %1345 = bitcast i32 %1344 to float, !dbg !16
  %1346 = select i1 %.not2, float %1330, float %1345, !dbg !16
  %1347 = select i1 %.not2, float %1345, float %1331, !dbg !16
  %1348 = select i1 %.not2, float %1333, float %1332, !dbg !16
  %1349 = bitcast float %1348 to i32, !dbg !16
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1349), !dbg !16
  %1351 = bitcast i32 %1350 to float, !dbg !16
  %1352 = select i1 %.not2, float %1332, float %1351, !dbg !16
  %1353 = select i1 %.not2, float %1351, float %1333, !dbg !16
  %1354 = select i1 %.not3, float %1352, float %1346, !dbg !16
  %1355 = xor i32 %1336, 32, !dbg !16
  %1356 = icmp samesign ult i32 %1355, %1339, !dbg !16
  %1357 = select i1 %1356, i32 %1355, i32 %1336, !dbg !16
  %1358 = shl nuw nsw i32 %1357, 2, !dbg !16
  %1359 = bitcast float %1354 to i32, !dbg !16
  %1360 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1359), !dbg !16
  %1361 = bitcast i32 %1360 to float, !dbg !16
  %1362 = select i1 %.not3, float %1346, float %1361, !dbg !16
  %1363 = select i1 %.not3, float %1361, float %1352, !dbg !16
  %1364 = select i1 %.not3, float %1353, float %1347, !dbg !16
  %1365 = bitcast float %1364 to i32, !dbg !16
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1365), !dbg !16
  %1367 = bitcast i32 %1366 to float, !dbg !16
  %1368 = select i1 %.not3, float %1347, float %1367, !dbg !16
  %1369 = select i1 %.not3, float %1367, float %1353, !dbg !16
  %1370 = extractvalue [2 x <8 x float>] %30, 0, !dbg !16
  %1371 = extractelement <8 x float> %1370, i64 0, !dbg !16
  %1372 = extractelement <8 x float> %1370, i64 1, !dbg !16
  %1373 = extractelement <8 x float> %1370, i64 2, !dbg !16
  %1374 = extractelement <8 x float> %1370, i64 3, !dbg !16
  %1375 = select i1 %.not2, float %1372, float %1371, !dbg !16
  %1376 = bitcast float %1375 to i32, !dbg !16
  %1377 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1376), !dbg !16
  %1378 = bitcast i32 %1377 to float, !dbg !16
  %1379 = select i1 %.not2, float %1371, float %1378, !dbg !16
  %1380 = select i1 %.not2, float %1378, float %1372, !dbg !16
  %1381 = select i1 %.not2, float %1374, float %1373, !dbg !16
  %1382 = bitcast float %1381 to i32, !dbg !16
  %1383 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1382), !dbg !16
  %1384 = bitcast i32 %1383 to float, !dbg !16
  %1385 = select i1 %.not2, float %1373, float %1384, !dbg !16
  %1386 = select i1 %.not2, float %1384, float %1374, !dbg !16
  %1387 = select i1 %.not3, float %1385, float %1379, !dbg !16
  %1388 = bitcast float %1387 to i32, !dbg !16
  %1389 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1388), !dbg !16
  %1390 = bitcast i32 %1389 to float, !dbg !16
  %1391 = select i1 %.not3, float %1379, float %1390, !dbg !16
  %1392 = select i1 %.not3, float %1390, float %1385, !dbg !16
  %1393 = select i1 %.not3, float %1386, float %1380, !dbg !16
  %1394 = bitcast float %1393 to i32, !dbg !16
  %1395 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1394), !dbg !16
  %1396 = bitcast i32 %1395 to float, !dbg !16
  %1397 = select i1 %.not3, float %1380, float %1396, !dbg !16
  %1398 = select i1 %.not3, float %1396, float %1386, !dbg !16
  %1399 = extractvalue [2 x <8 x float>] %29, 0, !dbg !16
  %1400 = extractelement <8 x float> %1399, i64 0, !dbg !16
  %1401 = extractelement <8 x float> %1399, i64 1, !dbg !16
  %1402 = extractelement <8 x float> %1399, i64 2, !dbg !16
  %1403 = extractelement <8 x float> %1399, i64 3, !dbg !16
  %1404 = select i1 %.not2, float %1401, float %1400, !dbg !16
  %1405 = bitcast float %1404 to i32, !dbg !16
  %1406 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1405), !dbg !16
  %1407 = bitcast i32 %1406 to float, !dbg !16
  %1408 = select i1 %.not2, float %1400, float %1407, !dbg !16
  %1409 = select i1 %.not2, float %1407, float %1401, !dbg !16
  %1410 = select i1 %.not2, float %1403, float %1402, !dbg !16
  %1411 = bitcast float %1410 to i32, !dbg !16
  %1412 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1411), !dbg !16
  %1413 = bitcast i32 %1412 to float, !dbg !16
  %1414 = select i1 %.not2, float %1402, float %1413, !dbg !16
  %1415 = select i1 %.not2, float %1413, float %1403, !dbg !16
  %1416 = select i1 %.not3, float %1414, float %1408, !dbg !16
  %1417 = bitcast float %1416 to i32, !dbg !16
  %1418 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1417), !dbg !16
  %1419 = bitcast i32 %1418 to float, !dbg !16
  %1420 = select i1 %.not3, float %1408, float %1419, !dbg !16
  %1421 = select i1 %.not3, float %1419, float %1414, !dbg !16
  %1422 = select i1 %.not3, float %1415, float %1409, !dbg !16
  %1423 = bitcast float %1422 to i32, !dbg !16
  %1424 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1423), !dbg !16
  %1425 = bitcast i32 %1424 to float, !dbg !16
  %1426 = select i1 %.not3, float %1409, float %1425, !dbg !16
  %1427 = select i1 %.not3, float %1425, float %1415, !dbg !16
  %1428 = extractvalue [2 x <8 x float>] %27, 0, !dbg !16
  %1429 = extractelement <8 x float> %1428, i64 0, !dbg !16
  %1430 = extractelement <8 x float> %1428, i64 1, !dbg !16
  %1431 = extractelement <8 x float> %1428, i64 2, !dbg !16
  %1432 = extractelement <8 x float> %1428, i64 3, !dbg !16
  %1433 = select i1 %.not2, float %1430, float %1429, !dbg !16
  %1434 = bitcast float %1433 to i32, !dbg !16
  %1435 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1434), !dbg !16
  %1436 = bitcast i32 %1435 to float, !dbg !16
  %1437 = select i1 %.not2, float %1429, float %1436, !dbg !16
  %1438 = select i1 %.not2, float %1436, float %1430, !dbg !16
  %1439 = select i1 %.not2, float %1432, float %1431, !dbg !16
  %1440 = bitcast float %1439 to i32, !dbg !16
  %1441 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1440), !dbg !16
  %1442 = bitcast i32 %1441 to float, !dbg !16
  %1443 = select i1 %.not2, float %1431, float %1442, !dbg !16
  %1444 = select i1 %.not2, float %1442, float %1432, !dbg !16
  %1445 = select i1 %.not3, float %1443, float %1437, !dbg !16
  %1446 = bitcast float %1445 to i32, !dbg !16
  %1447 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1446), !dbg !16
  %1448 = select i1 %.not3, float %1444, float %1438, !dbg !16
  %1449 = bitcast float %1448 to i32, !dbg !16
  %1450 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1449), !dbg !16
  %1451 = extractvalue [2 x <1 x float>] %28, 0, !dbg !19
  %1452 = shufflevector <1 x float> %1451, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1453 = extractelement <8 x float> %1329, i64 4, !dbg !16
  %1454 = extractelement <8 x float> %1329, i64 5, !dbg !16
  %1455 = extractelement <8 x float> %1329, i64 6, !dbg !16
  %1456 = extractelement <8 x float> %1329, i64 7, !dbg !16
  %1457 = select i1 %.not2, float %1454, float %1453, !dbg !16
  %1458 = bitcast float %1457 to i32, !dbg !16
  %1459 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1458), !dbg !16
  %1460 = bitcast i32 %1459 to float, !dbg !16
  %1461 = select i1 %.not2, float %1453, float %1460, !dbg !16
  %1462 = select i1 %.not2, float %1460, float %1454, !dbg !16
  %1463 = select i1 %.not2, float %1456, float %1455, !dbg !16
  %1464 = bitcast float %1463 to i32, !dbg !16
  %1465 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1464), !dbg !16
  %1466 = bitcast i32 %1465 to float, !dbg !16
  %1467 = select i1 %.not2, float %1455, float %1466, !dbg !16
  %1468 = select i1 %.not2, float %1466, float %1456, !dbg !16
  %1469 = select i1 %.not3, float %1467, float %1461, !dbg !16
  %1470 = bitcast float %1469 to i32, !dbg !16
  %1471 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1470), !dbg !16
  %1472 = bitcast i32 %1471 to float, !dbg !16
  %1473 = select i1 %.not3, float %1461, float %1472, !dbg !16
  %1474 = select i1 %.not3, float %1472, float %1467, !dbg !16
  %1475 = select i1 %.not3, float %1468, float %1462, !dbg !16
  %1476 = bitcast float %1475 to i32, !dbg !16
  %1477 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1476), !dbg !16
  %1478 = bitcast i32 %1477 to float, !dbg !16
  %1479 = select i1 %.not3, float %1462, float %1478, !dbg !16
  %1480 = select i1 %.not3, float %1478, float %1468, !dbg !16
  %1481 = extractelement <8 x float> %1370, i64 4, !dbg !16
  %1482 = extractelement <8 x float> %1370, i64 5, !dbg !16
  %1483 = extractelement <8 x float> %1370, i64 6, !dbg !16
  %1484 = extractelement <8 x float> %1370, i64 7, !dbg !16
  %1485 = select i1 %.not2, float %1482, float %1481, !dbg !16
  %1486 = bitcast float %1485 to i32, !dbg !16
  %1487 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1486), !dbg !16
  %1488 = bitcast i32 %1487 to float, !dbg !16
  %1489 = select i1 %.not2, float %1481, float %1488, !dbg !16
  %1490 = select i1 %.not2, float %1488, float %1482, !dbg !16
  %1491 = select i1 %.not2, float %1484, float %1483, !dbg !16
  %1492 = bitcast float %1491 to i32, !dbg !16
  %1493 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1492), !dbg !16
  %1494 = bitcast i32 %1493 to float, !dbg !16
  %1495 = select i1 %.not2, float %1483, float %1494, !dbg !16
  %1496 = select i1 %.not2, float %1494, float %1484, !dbg !16
  %1497 = select i1 %.not3, float %1495, float %1489, !dbg !16
  %1498 = bitcast float %1497 to i32, !dbg !16
  %1499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1498), !dbg !16
  %1500 = bitcast i32 %1499 to float, !dbg !16
  %1501 = select i1 %.not3, float %1489, float %1500, !dbg !16
  %1502 = select i1 %.not3, float %1500, float %1495, !dbg !16
  %1503 = select i1 %.not3, float %1496, float %1490, !dbg !16
  %1504 = bitcast float %1503 to i32, !dbg !16
  %1505 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1504), !dbg !16
  %1506 = bitcast i32 %1505 to float, !dbg !16
  %1507 = select i1 %.not3, float %1490, float %1506, !dbg !16
  %1508 = select i1 %.not3, float %1506, float %1496, !dbg !16
  %1509 = extractelement <8 x float> %1399, i64 4, !dbg !16
  %1510 = extractelement <8 x float> %1399, i64 5, !dbg !16
  %1511 = extractelement <8 x float> %1399, i64 6, !dbg !16
  %1512 = extractelement <8 x float> %1399, i64 7, !dbg !16
  %1513 = select i1 %.not2, float %1510, float %1509, !dbg !16
  %1514 = bitcast float %1513 to i32, !dbg !16
  %1515 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1514), !dbg !16
  %1516 = bitcast i32 %1515 to float, !dbg !16
  %1517 = select i1 %.not2, float %1509, float %1516, !dbg !16
  %1518 = select i1 %.not2, float %1516, float %1510, !dbg !16
  %1519 = select i1 %.not2, float %1512, float %1511, !dbg !16
  %1520 = bitcast float %1519 to i32, !dbg !16
  %1521 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1520), !dbg !16
  %1522 = bitcast i32 %1521 to float, !dbg !16
  %1523 = select i1 %.not2, float %1511, float %1522, !dbg !16
  %1524 = select i1 %.not2, float %1522, float %1512, !dbg !16
  %1525 = select i1 %.not3, float %1523, float %1517, !dbg !16
  %1526 = bitcast float %1525 to i32, !dbg !16
  %1527 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1526), !dbg !16
  %1528 = bitcast i32 %1527 to float, !dbg !16
  %1529 = select i1 %.not3, float %1517, float %1528, !dbg !16
  %1530 = select i1 %.not3, float %1528, float %1523, !dbg !16
  %1531 = select i1 %.not3, float %1524, float %1518, !dbg !16
  %1532 = bitcast float %1531 to i32, !dbg !16
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1532), !dbg !16
  %1534 = bitcast i32 %1533 to float, !dbg !16
  %1535 = select i1 %.not3, float %1518, float %1534, !dbg !16
  %1536 = select i1 %.not3, float %1534, float %1524, !dbg !16
  %1537 = extractelement <8 x float> %1428, i64 4, !dbg !16
  %1538 = extractelement <8 x float> %1428, i64 5, !dbg !16
  %1539 = extractelement <8 x float> %1428, i64 6, !dbg !16
  %1540 = extractelement <8 x float> %1428, i64 7, !dbg !16
  %1541 = select i1 %.not2, float %1538, float %1537, !dbg !16
  %1542 = bitcast float %1541 to i32, !dbg !16
  %1543 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1542), !dbg !16
  %1544 = bitcast i32 %1543 to float, !dbg !16
  %1545 = select i1 %.not2, float %1537, float %1544, !dbg !16
  %1546 = select i1 %.not2, float %1544, float %1538, !dbg !16
  %1547 = select i1 %.not2, float %1540, float %1539, !dbg !16
  %1548 = bitcast float %1547 to i32, !dbg !16
  %1549 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1548), !dbg !16
  %1550 = bitcast i32 %1549 to float, !dbg !16
  %1551 = select i1 %.not2, float %1539, float %1550, !dbg !16
  %1552 = select i1 %.not2, float %1550, float %1540, !dbg !16
  %1553 = select i1 %.not3, float %1551, float %1545, !dbg !16
  %1554 = bitcast float %1553 to i32, !dbg !16
  %1555 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1554), !dbg !16
  %1556 = select i1 %.not3, float %1552, float %1546, !dbg !16
  %1557 = bitcast float %1556 to i32, !dbg !16
  %1558 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1557), !dbg !16
  %1559 = extractvalue [2 x <8 x float>] %31, 1, !dbg !16
  %1560 = extractelement <8 x float> %1559, i64 0, !dbg !16
  %1561 = extractelement <8 x float> %1559, i64 1, !dbg !16
  %1562 = extractelement <8 x float> %1559, i64 2, !dbg !16
  %1563 = extractelement <8 x float> %1559, i64 3, !dbg !16
  %1564 = select i1 %.not2, float %1561, float %1560, !dbg !16
  %1565 = bitcast float %1564 to i32, !dbg !16
  %1566 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1565), !dbg !16
  %1567 = bitcast i32 %1566 to float, !dbg !16
  %1568 = select i1 %.not2, float %1560, float %1567, !dbg !16
  %1569 = select i1 %.not2, float %1567, float %1561, !dbg !16
  %1570 = select i1 %.not2, float %1563, float %1562, !dbg !16
  %1571 = bitcast float %1570 to i32, !dbg !16
  %1572 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1571), !dbg !16
  %1573 = bitcast i32 %1572 to float, !dbg !16
  %1574 = select i1 %.not2, float %1562, float %1573, !dbg !16
  %1575 = select i1 %.not2, float %1573, float %1563, !dbg !16
  %1576 = select i1 %.not3, float %1574, float %1568, !dbg !16
  %1577 = bitcast float %1576 to i32, !dbg !16
  %1578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1577), !dbg !16
  %1579 = bitcast i32 %1578 to float, !dbg !16
  %1580 = select i1 %.not3, float %1568, float %1579, !dbg !16
  %1581 = select i1 %.not3, float %1579, float %1574, !dbg !16
  %1582 = select i1 %.not3, float %1575, float %1569, !dbg !16
  %1583 = bitcast float %1582 to i32, !dbg !16
  %1584 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1583), !dbg !16
  %1585 = bitcast i32 %1584 to float, !dbg !16
  %1586 = select i1 %.not3, float %1569, float %1585, !dbg !16
  %1587 = select i1 %.not3, float %1585, float %1575, !dbg !16
  %1588 = extractvalue [2 x <8 x float>] %30, 1, !dbg !16
  %1589 = extractelement <8 x float> %1588, i64 0, !dbg !16
  %1590 = extractelement <8 x float> %1588, i64 1, !dbg !16
  %1591 = extractelement <8 x float> %1588, i64 2, !dbg !16
  %1592 = extractelement <8 x float> %1588, i64 3, !dbg !16
  %1593 = select i1 %.not2, float %1590, float %1589, !dbg !16
  %1594 = bitcast float %1593 to i32, !dbg !16
  %1595 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1594), !dbg !16
  %1596 = bitcast i32 %1595 to float, !dbg !16
  %1597 = select i1 %.not2, float %1589, float %1596, !dbg !16
  %1598 = select i1 %.not2, float %1596, float %1590, !dbg !16
  %1599 = select i1 %.not2, float %1592, float %1591, !dbg !16
  %1600 = bitcast float %1599 to i32, !dbg !16
  %1601 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1600), !dbg !16
  %1602 = bitcast i32 %1601 to float, !dbg !16
  %1603 = select i1 %.not2, float %1591, float %1602, !dbg !16
  %1604 = select i1 %.not2, float %1602, float %1592, !dbg !16
  %1605 = select i1 %.not3, float %1603, float %1597, !dbg !16
  %1606 = bitcast float %1605 to i32, !dbg !16
  %1607 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1606), !dbg !16
  %1608 = bitcast i32 %1607 to float, !dbg !16
  %1609 = select i1 %.not3, float %1597, float %1608, !dbg !16
  %1610 = select i1 %.not3, float %1608, float %1603, !dbg !16
  %1611 = select i1 %.not3, float %1604, float %1598, !dbg !16
  %1612 = bitcast float %1611 to i32, !dbg !16
  %1613 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1612), !dbg !16
  %1614 = bitcast i32 %1613 to float, !dbg !16
  %1615 = select i1 %.not3, float %1598, float %1614, !dbg !16
  %1616 = select i1 %.not3, float %1614, float %1604, !dbg !16
  %1617 = extractvalue [2 x <8 x float>] %29, 1, !dbg !16
  %1618 = extractelement <8 x float> %1617, i64 0, !dbg !16
  %1619 = extractelement <8 x float> %1617, i64 1, !dbg !16
  %1620 = extractelement <8 x float> %1617, i64 2, !dbg !16
  %1621 = extractelement <8 x float> %1617, i64 3, !dbg !16
  %1622 = select i1 %.not2, float %1619, float %1618, !dbg !16
  %1623 = bitcast float %1622 to i32, !dbg !16
  %1624 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1623), !dbg !16
  %1625 = bitcast i32 %1624 to float, !dbg !16
  %1626 = select i1 %.not2, float %1618, float %1625, !dbg !16
  %1627 = select i1 %.not2, float %1625, float %1619, !dbg !16
  %1628 = select i1 %.not2, float %1621, float %1620, !dbg !16
  %1629 = bitcast float %1628 to i32, !dbg !16
  %1630 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1629), !dbg !16
  %1631 = bitcast i32 %1630 to float, !dbg !16
  %1632 = select i1 %.not2, float %1620, float %1631, !dbg !16
  %1633 = select i1 %.not2, float %1631, float %1621, !dbg !16
  %1634 = select i1 %.not3, float %1632, float %1626, !dbg !16
  %1635 = bitcast float %1634 to i32, !dbg !16
  %1636 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1635), !dbg !16
  %1637 = bitcast i32 %1636 to float, !dbg !16
  %1638 = select i1 %.not3, float %1626, float %1637, !dbg !16
  %1639 = select i1 %.not3, float %1637, float %1632, !dbg !16
  %1640 = select i1 %.not3, float %1633, float %1627, !dbg !16
  %1641 = bitcast float %1640 to i32, !dbg !16
  %1642 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1641), !dbg !16
  %1643 = bitcast i32 %1642 to float, !dbg !16
  %1644 = select i1 %.not3, float %1627, float %1643, !dbg !16
  %1645 = select i1 %.not3, float %1643, float %1633, !dbg !16
  %1646 = extractvalue [2 x <8 x float>] %27, 1, !dbg !16
  %1647 = extractelement <8 x float> %1646, i64 0, !dbg !16
  %1648 = extractelement <8 x float> %1646, i64 1, !dbg !16
  %1649 = extractelement <8 x float> %1646, i64 2, !dbg !16
  %1650 = extractelement <8 x float> %1646, i64 3, !dbg !16
  %1651 = select i1 %.not2, float %1648, float %1647, !dbg !16
  %1652 = bitcast float %1651 to i32, !dbg !16
  %1653 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1652), !dbg !16
  %1654 = bitcast i32 %1653 to float, !dbg !16
  %1655 = select i1 %.not2, float %1647, float %1654, !dbg !16
  %1656 = select i1 %.not2, float %1654, float %1648, !dbg !16
  %1657 = select i1 %.not2, float %1650, float %1649, !dbg !16
  %1658 = bitcast float %1657 to i32, !dbg !16
  %1659 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1658), !dbg !16
  %1660 = bitcast i32 %1659 to float, !dbg !16
  %1661 = select i1 %.not2, float %1649, float %1660, !dbg !16
  %1662 = select i1 %.not2, float %1660, float %1650, !dbg !16
  %1663 = select i1 %.not3, float %1661, float %1655, !dbg !16
  %1664 = bitcast float %1663 to i32, !dbg !16
  %1665 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1664), !dbg !16
  %1666 = select i1 %.not3, float %1662, float %1656, !dbg !16
  %1667 = bitcast float %1666 to i32, !dbg !16
  %1668 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1667), !dbg !16
  %1669 = extractvalue [2 x <1 x float>] %28, 1, !dbg !19
  %1670 = shufflevector <1 x float> %1669, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1671 = extractelement <8 x float> %1559, i64 4, !dbg !16
  %1672 = extractelement <8 x float> %1559, i64 5, !dbg !16
  %1673 = extractelement <8 x float> %1559, i64 6, !dbg !16
  %1674 = extractelement <8 x float> %1559, i64 7, !dbg !16
  %1675 = select i1 %.not2, float %1672, float %1671, !dbg !16
  %1676 = bitcast float %1675 to i32, !dbg !16
  %1677 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1676), !dbg !16
  %1678 = bitcast i32 %1677 to float, !dbg !16
  %1679 = select i1 %.not2, float %1671, float %1678, !dbg !16
  %1680 = select i1 %.not2, float %1678, float %1672, !dbg !16
  %1681 = select i1 %.not2, float %1674, float %1673, !dbg !16
  %1682 = bitcast float %1681 to i32, !dbg !16
  %1683 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1682), !dbg !16
  %1684 = bitcast i32 %1683 to float, !dbg !16
  %1685 = select i1 %.not2, float %1673, float %1684, !dbg !16
  %1686 = select i1 %.not2, float %1684, float %1674, !dbg !16
  %1687 = select i1 %.not3, float %1685, float %1679, !dbg !16
  %1688 = bitcast float %1687 to i32, !dbg !16
  %1689 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1688), !dbg !16
  %1690 = bitcast i32 %1689 to float, !dbg !16
  %1691 = select i1 %.not3, float %1679, float %1690, !dbg !16
  %1692 = select i1 %.not3, float %1690, float %1685, !dbg !16
  %1693 = select i1 %.not3, float %1686, float %1680, !dbg !16
  %1694 = bitcast float %1693 to i32, !dbg !16
  %1695 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1694), !dbg !16
  %1696 = bitcast i32 %1695 to float, !dbg !16
  %1697 = select i1 %.not3, float %1680, float %1696, !dbg !16
  %1698 = select i1 %.not3, float %1696, float %1686, !dbg !16
  %1699 = extractelement <8 x float> %1588, i64 4, !dbg !16
  %1700 = extractelement <8 x float> %1588, i64 5, !dbg !16
  %1701 = extractelement <8 x float> %1588, i64 6, !dbg !16
  %1702 = extractelement <8 x float> %1588, i64 7, !dbg !16
  %1703 = select i1 %.not2, float %1700, float %1699, !dbg !16
  %1704 = bitcast float %1703 to i32, !dbg !16
  %1705 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1704), !dbg !16
  %1706 = bitcast i32 %1705 to float, !dbg !16
  %1707 = select i1 %.not2, float %1699, float %1706, !dbg !16
  %1708 = select i1 %.not2, float %1706, float %1700, !dbg !16
  %1709 = select i1 %.not2, float %1702, float %1701, !dbg !16
  %1710 = bitcast float %1709 to i32, !dbg !16
  %1711 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1710), !dbg !16
  %1712 = bitcast i32 %1711 to float, !dbg !16
  %1713 = select i1 %.not2, float %1701, float %1712, !dbg !16
  %1714 = select i1 %.not2, float %1712, float %1702, !dbg !16
  %1715 = select i1 %.not3, float %1713, float %1707, !dbg !16
  %1716 = bitcast float %1715 to i32, !dbg !16
  %1717 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1716), !dbg !16
  %1718 = bitcast i32 %1717 to float, !dbg !16
  %1719 = select i1 %.not3, float %1707, float %1718, !dbg !16
  %1720 = select i1 %.not3, float %1718, float %1713, !dbg !16
  %1721 = select i1 %.not3, float %1714, float %1708, !dbg !16
  %1722 = bitcast float %1721 to i32, !dbg !16
  %1723 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1722), !dbg !16
  %1724 = bitcast i32 %1723 to float, !dbg !16
  %1725 = select i1 %.not3, float %1708, float %1724, !dbg !16
  %1726 = select i1 %.not3, float %1724, float %1714, !dbg !16
  %1727 = extractelement <8 x float> %1617, i64 4, !dbg !16
  %1728 = extractelement <8 x float> %1617, i64 5, !dbg !16
  %1729 = extractelement <8 x float> %1617, i64 6, !dbg !16
  %1730 = extractelement <8 x float> %1617, i64 7, !dbg !16
  %1731 = select i1 %.not2, float %1728, float %1727, !dbg !16
  %1732 = bitcast float %1731 to i32, !dbg !16
  %1733 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1732), !dbg !16
  %1734 = bitcast i32 %1733 to float, !dbg !16
  %1735 = select i1 %.not2, float %1727, float %1734, !dbg !16
  %1736 = select i1 %.not2, float %1734, float %1728, !dbg !16
  %1737 = select i1 %.not2, float %1730, float %1729, !dbg !16
  %1738 = bitcast float %1737 to i32, !dbg !16
  %1739 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1738), !dbg !16
  %1740 = bitcast i32 %1739 to float, !dbg !16
  %1741 = select i1 %.not2, float %1729, float %1740, !dbg !16
  %1742 = select i1 %.not2, float %1740, float %1730, !dbg !16
  %1743 = select i1 %.not3, float %1741, float %1735, !dbg !16
  %1744 = bitcast float %1743 to i32, !dbg !16
  %1745 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1744), !dbg !16
  %1746 = bitcast i32 %1745 to float, !dbg !16
  %1747 = select i1 %.not3, float %1735, float %1746, !dbg !16
  %1748 = select i1 %.not3, float %1746, float %1741, !dbg !16
  %1749 = select i1 %.not3, float %1742, float %1736, !dbg !16
  %1750 = bitcast float %1749 to i32, !dbg !16
  %1751 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1750), !dbg !16
  %1752 = bitcast i32 %1751 to float, !dbg !16
  %1753 = select i1 %.not3, float %1736, float %1752, !dbg !16
  %1754 = select i1 %.not3, float %1752, float %1742, !dbg !16
  %1755 = extractelement <8 x float> %1646, i64 4, !dbg !16
  %1756 = extractelement <8 x float> %1646, i64 5, !dbg !16
  %1757 = extractelement <8 x float> %1646, i64 6, !dbg !16
  %1758 = extractelement <8 x float> %1646, i64 7, !dbg !16
  %1759 = select i1 %.not2, float %1756, float %1755, !dbg !16
  %1760 = bitcast float %1759 to i32, !dbg !16
  %1761 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1760), !dbg !16
  %1762 = bitcast i32 %1761 to float, !dbg !16
  %1763 = select i1 %.not2, float %1755, float %1762, !dbg !16
  %1764 = select i1 %.not2, float %1762, float %1756, !dbg !16
  %1765 = select i1 %.not2, float %1758, float %1757, !dbg !16
  %1766 = bitcast float %1765 to i32, !dbg !16
  %1767 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1766), !dbg !16
  %1768 = bitcast i32 %1767 to float, !dbg !16
  %1769 = select i1 %.not2, float %1757, float %1768, !dbg !16
  %1770 = select i1 %.not2, float %1768, float %1758, !dbg !16
  %1771 = select i1 %.not3, float %1769, float %1763, !dbg !16
  %1772 = bitcast float %1771 to i32, !dbg !16
  %1773 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1772), !dbg !16
  %1774 = select i1 %.not3, float %1770, float %1764, !dbg !16
  %1775 = bitcast float %1774 to i32, !dbg !16
  %1776 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1358, i32 %1775), !dbg !16
  br i1 true, label %1777, label %1841, !dbg !21

1777:                                             ; preds = %1326
  %1778 = bitcast i32 %1773 to float, !dbg !16
  %1779 = select i1 %.not3, float %1763, float %1778, !dbg !16
  %1780 = insertelement <4 x float> poison, float %1779, i64 0, !dbg !19
  %1781 = bitcast i32 %1776 to float, !dbg !16
  %1782 = select i1 %.not3, float %1764, float %1781, !dbg !16
  %1783 = insertelement <4 x float> %1780, float %1782, i64 1, !dbg !19
  %1784 = select i1 %.not3, float %1778, float %1769, !dbg !16
  %1785 = insertelement <4 x float> %1783, float %1784, i64 2, !dbg !19
  %1786 = select i1 %.not3, float %1781, float %1770, !dbg !16
  %1787 = insertelement <4 x float> %1785, float %1786, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1670, !dbg !19
  %normalized = fmul <4 x float> %1787, %row.reciprocal, !dbg !19
  %1788 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1789 = extractelement <4 x half> %1788, i64 3, !dbg !20
  %1790 = extractelement <4 x half> %1788, i64 2, !dbg !20
  %1791 = extractelement <4 x half> %1788, i64 1, !dbg !20
  %1792 = extractelement <4 x half> %1788, i64 0, !dbg !20
  %1793 = bitcast i32 %1665 to float, !dbg !16
  %1794 = select i1 %.not3, float %1655, float %1793, !dbg !16
  %1795 = insertelement <4 x float> poison, float %1794, i64 0, !dbg !19
  %1796 = bitcast i32 %1668 to float, !dbg !16
  %1797 = select i1 %.not3, float %1656, float %1796, !dbg !16
  %1798 = insertelement <4 x float> %1795, float %1797, i64 1, !dbg !19
  %1799 = select i1 %.not3, float %1793, float %1661, !dbg !16
  %1800 = insertelement <4 x float> %1798, float %1799, i64 2, !dbg !19
  %1801 = select i1 %.not3, float %1796, float %1662, !dbg !16
  %1802 = insertelement <4 x float> %1800, float %1801, i64 3, !dbg !19
  %normalized72 = fmul <4 x float> %1802, %row.reciprocal, !dbg !19
  %1803 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !20
  %1804 = extractelement <4 x half> %1803, i64 3, !dbg !20
  %1805 = extractelement <4 x half> %1803, i64 2, !dbg !20
  %1806 = extractelement <4 x half> %1803, i64 1, !dbg !20
  %1807 = extractelement <4 x half> %1803, i64 0, !dbg !20
  %1808 = bitcast i32 %1555 to float, !dbg !16
  %1809 = select i1 %.not3, float %1545, float %1808, !dbg !16
  %1810 = insertelement <4 x float> poison, float %1809, i64 0, !dbg !19
  %1811 = bitcast i32 %1558 to float, !dbg !16
  %1812 = select i1 %.not3, float %1546, float %1811, !dbg !16
  %1813 = insertelement <4 x float> %1810, float %1812, i64 1, !dbg !19
  %1814 = select i1 %.not3, float %1808, float %1551, !dbg !16
  %1815 = insertelement <4 x float> %1813, float %1814, i64 2, !dbg !19
  %1816 = select i1 %.not3, float %1811, float %1552, !dbg !16
  %1817 = insertelement <4 x float> %1815, float %1816, i64 3, !dbg !19
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1452, !dbg !19
  %normalized74 = fmul <4 x float> %1817, %row.reciprocal73, !dbg !19
  %1818 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %1819 = extractelement <4 x half> %1818, i64 3, !dbg !20
  %1820 = extractelement <4 x half> %1818, i64 2, !dbg !20
  %1821 = extractelement <4 x half> %1818, i64 1, !dbg !20
  %1822 = extractelement <4 x half> %1818, i64 0, !dbg !20
  %1823 = bitcast i32 %1447 to float, !dbg !16
  %1824 = select i1 %.not3, float %1437, float %1823, !dbg !16
  %1825 = insertelement <4 x float> poison, float %1824, i64 0, !dbg !19
  %1826 = bitcast i32 %1450 to float, !dbg !16
  %1827 = select i1 %.not3, float %1438, float %1826, !dbg !16
  %1828 = insertelement <4 x float> %1825, float %1827, i64 1, !dbg !19
  %1829 = select i1 %.not3, float %1823, float %1443, !dbg !16
  %1830 = insertelement <4 x float> %1828, float %1829, i64 2, !dbg !19
  %1831 = select i1 %.not3, float %1826, float %1444, !dbg !16
  %1832 = insertelement <4 x float> %1830, float %1831, i64 3, !dbg !19
  %normalized75 = fmul <4 x float> %1832, %row.reciprocal73, !dbg !19
  %1833 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !20
  %1834 = extractelement <4 x half> %1833, i64 3, !dbg !20
  %1835 = extractelement <4 x half> %1833, i64 2, !dbg !20
  %1836 = extractelement <4 x half> %1833, i64 1, !dbg !20
  %1837 = extractelement <4 x half> %1833, i64 0, !dbg !20
  %1838 = add nsw i64 %8, %13, !dbg !21
  %1839 = shl nsw i64 %1838, 19, !dbg !21
  %1840 = and i64 %1839, 16252928, !dbg !21
  %address.thread523 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local524 = zext nneg i32 %address.thread523 to i64, !dbg !13
  %address.local525 = add nsw i64 %16, %address.local524, !dbg !21
  %address.local526 = lshr i64 %address.local524, 6, !dbg !13
  %address.local527 = shl nuw nsw i64 %address.local526, 4, !dbg !21
  %address.local528 = add nsw i64 %address.local525, %address.local527, !dbg !21
  %address.local529 = and i64 %address.local524, 240, !dbg !21
  %address.local530 = sub nsw i64 %address.local528, %address.local529, !dbg !21
  %address.local531 = add nsw i64 %address.local530, %17, !dbg !21
  %address.local532 = shl nuw nsw i64 %address.local531, 7, !dbg !21
  %address.local533 = add nuw nsw i64 %1840, %address.local532, !dbg !21
  %address.local534 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local533, !dbg !21
  %address.local535 = lshr i64 %address.local524, 2, !dbg !13
  %address.local536 = and i64 %address.local535, 12, !dbg !13
  %address.local537 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local534, i64 %address.local536, !dbg !21
  store half %1837, ptr addrspace(1) %address.local537, align 2, !dbg !21
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local533, !dbg !21
  %address.local539 = or disjoint i64 %address.local536, 1, !dbg !21
  %address.local540 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local538, i64 %address.local539, !dbg !21
  store half %1836, ptr addrspace(1) %address.local540, align 2, !dbg !21
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local533, !dbg !21
  %address.local542 = or disjoint i64 %address.local536, 2, !dbg !21
  %address.local543 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local541, i64 %address.local542, !dbg !21
  store half %1835, ptr addrspace(1) %address.local543, align 2, !dbg !21
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local533, !dbg !21
  %address.local545 = or disjoint i64 %address.local536, 3, !dbg !21
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local544, i64 %address.local545, !dbg !21
  store half %1834, ptr addrspace(1) %address.local546, align 2, !dbg !21
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local533, !dbg !21
  %address.local548 = or disjoint i64 %address.local536, 16, !dbg !21
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local547, i64 %address.local548, !dbg !21
  store half %1822, ptr addrspace(1) %address.local549, align 2, !dbg !21
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local533, !dbg !21
  %address.local551 = or disjoint i64 %address.local536, 17, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local550, i64 %address.local551, !dbg !21
  store half %1821, ptr addrspace(1) %address.local552, align 2, !dbg !21
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local533, !dbg !21
  %address.local554 = or disjoint i64 %address.local536, 18, !dbg !21
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local553, i64 %address.local554, !dbg !21
  store half %1820, ptr addrspace(1) %address.local555, align 2, !dbg !21
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local533, !dbg !21
  %address.local557 = or disjoint i64 %address.local536, 19, !dbg !21
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local556, i64 %address.local557, !dbg !21
  store half %1819, ptr addrspace(1) %address.local558, align 2, !dbg !21
  %address.local559 = shl nsw i64 %address.local531, 7, !dbg !21
  %address.local560 = add i64 %address.local559, 8192, !dbg !21
  %address.local561 = add nuw nsw i64 %1840, %address.local560, !dbg !21
  %address.local562 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local562, i64 %address.local536, !dbg !21
  store half %1807, ptr addrspace(1) %address.local563, align 2, !dbg !21
  %address.local564 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local565 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local564, i64 %address.local539, !dbg !21
  store half %1806, ptr addrspace(1) %address.local565, align 2, !dbg !21
  %address.local566 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local567 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local566, i64 %address.local542, !dbg !21
  store half %1805, ptr addrspace(1) %address.local567, align 2, !dbg !21
  %address.local568 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local568, i64 %address.local545, !dbg !21
  store half %1804, ptr addrspace(1) %address.local569, align 2, !dbg !21
  %address.local570 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local571 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local570, i64 %address.local548, !dbg !21
  store half %1792, ptr addrspace(1) %address.local571, align 2, !dbg !21
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local572, i64 %address.local551, !dbg !21
  store half %1791, ptr addrspace(1) %address.local573, align 2, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local574, i64 %address.local554, !dbg !21
  store half %1790, ptr addrspace(1) %address.local575, align 2, !dbg !21
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local561, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local576, i64 %address.local557, !dbg !21
  store half %1789, ptr addrspace(1) %address.local577, align 2, !dbg !21
  br label %1841, !dbg !21

1841:                                             ; preds = %1777, %1326
  br i1 true, label %1842, label %1880, !dbg !21

1842:                                             ; preds = %1841
  %1843 = insertelement <4 x float> poison, float %1747, i64 0, !dbg !19
  %1844 = insertelement <4 x float> %1843, float %1753, i64 1, !dbg !19
  %1845 = insertelement <4 x float> %1844, float %1748, i64 2, !dbg !19
  %1846 = insertelement <4 x float> %1845, float %1754, i64 3, !dbg !19
  %row.reciprocal76 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1670, !dbg !19
  %normalized77 = fmul <4 x float> %1846, %row.reciprocal76, !dbg !19
  %1847 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !20
  %1848 = extractelement <4 x half> %1847, i64 3, !dbg !20
  %1849 = extractelement <4 x half> %1847, i64 2, !dbg !20
  %1850 = extractelement <4 x half> %1847, i64 1, !dbg !20
  %1851 = extractelement <4 x half> %1847, i64 0, !dbg !20
  %1852 = insertelement <4 x float> poison, float %1638, i64 0, !dbg !19
  %1853 = insertelement <4 x float> %1852, float %1644, i64 1, !dbg !19
  %1854 = insertelement <4 x float> %1853, float %1639, i64 2, !dbg !19
  %1855 = insertelement <4 x float> %1854, float %1645, i64 3, !dbg !19
  %normalized78 = fmul <4 x float> %1855, %row.reciprocal76, !dbg !19
  %1856 = fptrunc <4 x float> %normalized78 to <4 x half>, !dbg !20
  %1857 = extractelement <4 x half> %1856, i64 3, !dbg !20
  %1858 = extractelement <4 x half> %1856, i64 2, !dbg !20
  %1859 = extractelement <4 x half> %1856, i64 1, !dbg !20
  %1860 = extractelement <4 x half> %1856, i64 0, !dbg !20
  %1861 = insertelement <4 x float> poison, float %1529, i64 0, !dbg !19
  %1862 = insertelement <4 x float> %1861, float %1535, i64 1, !dbg !19
  %1863 = insertelement <4 x float> %1862, float %1530, i64 2, !dbg !19
  %1864 = insertelement <4 x float> %1863, float %1536, i64 3, !dbg !19
  %row.reciprocal79 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1452, !dbg !19
  %normalized80 = fmul <4 x float> %1864, %row.reciprocal79, !dbg !19
  %1865 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !20
  %1866 = extractelement <4 x half> %1865, i64 3, !dbg !20
  %1867 = extractelement <4 x half> %1865, i64 2, !dbg !20
  %1868 = extractelement <4 x half> %1865, i64 1, !dbg !20
  %1869 = extractelement <4 x half> %1865, i64 0, !dbg !20
  %1870 = insertelement <4 x float> poison, float %1420, i64 0, !dbg !19
  %1871 = insertelement <4 x float> %1870, float %1426, i64 1, !dbg !19
  %1872 = insertelement <4 x float> %1871, float %1421, i64 2, !dbg !19
  %1873 = insertelement <4 x float> %1872, float %1427, i64 3, !dbg !19
  %normalized81 = fmul <4 x float> %1873, %row.reciprocal79, !dbg !19
  %1874 = fptrunc <4 x float> %normalized81 to <4 x half>, !dbg !20
  %1875 = extractelement <4 x half> %1874, i64 3, !dbg !20
  %1876 = extractelement <4 x half> %1874, i64 2, !dbg !20
  %1877 = extractelement <4 x half> %1874, i64 1, !dbg !20
  %1878 = extractelement <4 x half> %1874, i64 0, !dbg !20
  %1879 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread578 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local579 = zext nneg i32 %address.thread578 to i64, !dbg !13
  %address.local580 = add nsw i64 %16, %address.local579, !dbg !21
  %address.local581 = lshr i64 %address.local579, 6, !dbg !13
  %address.local582 = shl nuw nsw i64 %address.local581, 4, !dbg !21
  %address.local583 = add nsw i64 %address.local580, %address.local582, !dbg !21
  %address.local584 = and i64 %address.local579, 240, !dbg !21
  %address.local585 = sub nsw i64 %address.local583, %address.local584, !dbg !21
  %address.local586 = add nsw i64 %address.local585, %17, !dbg !21
  %address.local587 = shl nuw nsw i64 %address.local586, 7, !dbg !21
  %address.local588 = add nuw nsw i64 %1879, %address.local587, !dbg !21
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local588, !dbg !21
  %address.local590 = lshr i64 %address.local579, 2, !dbg !13
  %address.local591 = and i64 %address.local590, 12, !dbg !13
  %address.local592 = or disjoint i64 %address.local591, 32, !dbg !21
  %address.local593 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local589, i64 %address.local592, !dbg !21
  store half %1878, ptr addrspace(1) %address.local593, align 2, !dbg !21
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local588, !dbg !21
  %address.local595 = or disjoint i64 %address.local591, 33, !dbg !21
  %address.local596 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local594, i64 %address.local595, !dbg !21
  store half %1877, ptr addrspace(1) %address.local596, align 2, !dbg !21
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local588, !dbg !21
  %address.local598 = or disjoint i64 %address.local591, 34, !dbg !21
  %address.local599 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local597, i64 %address.local598, !dbg !21
  store half %1876, ptr addrspace(1) %address.local599, align 2, !dbg !21
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local588, !dbg !21
  %address.local601 = or disjoint i64 %address.local591, 35, !dbg !21
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local601, !dbg !21
  store half %1875, ptr addrspace(1) %address.local602, align 2, !dbg !21
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local588, !dbg !21
  %address.local604 = or disjoint i64 %address.local591, 48, !dbg !21
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local603, i64 %address.local604, !dbg !21
  store half %1869, ptr addrspace(1) %address.local605, align 2, !dbg !21
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local588, !dbg !21
  %address.local607 = or disjoint i64 %address.local591, 49, !dbg !21
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local606, i64 %address.local607, !dbg !21
  store half %1868, ptr addrspace(1) %address.local608, align 2, !dbg !21
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local588, !dbg !21
  %address.local610 = or disjoint i64 %address.local591, 50, !dbg !21
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local609, i64 %address.local610, !dbg !21
  store half %1867, ptr addrspace(1) %address.local611, align 2, !dbg !21
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local588, !dbg !21
  %address.local613 = or i64 %address.local590, 51, !dbg !21
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local612, i64 %address.local613, !dbg !21
  store half %1866, ptr addrspace(1) %address.local614, align 2, !dbg !21
  %address.local615 = shl nsw i64 %address.local586, 7, !dbg !21
  %address.local616 = add i64 %address.local615, 8192, !dbg !21
  %address.local617 = add nuw nsw i64 %1879, %address.local616, !dbg !21
  %address.local618 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local617, !dbg !21
  %address.local619 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local618, i64 %address.local592, !dbg !21
  store half %1860, ptr addrspace(1) %address.local619, align 2, !dbg !21
  %address.local620 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local617, !dbg !21
  %address.local621 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local595, !dbg !21
  store half %1859, ptr addrspace(1) %address.local621, align 2, !dbg !21
  %address.local622 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local617, !dbg !21
  %address.local623 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local622, i64 %address.local598, !dbg !21
  store half %1858, ptr addrspace(1) %address.local623, align 2, !dbg !21
  %address.local624 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local617, !dbg !21
  %address.local625 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local624, i64 %address.local601, !dbg !21
  store half %1857, ptr addrspace(1) %address.local625, align 2, !dbg !21
  %address.local626 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local617, !dbg !21
  %address.local627 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local626, i64 %address.local604, !dbg !21
  store half %1851, ptr addrspace(1) %address.local627, align 2, !dbg !21
  %address.local628 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local617, !dbg !21
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local628, i64 %address.local607, !dbg !21
  store half %1850, ptr addrspace(1) %address.local629, align 2, !dbg !21
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local617, !dbg !21
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local630, i64 %address.local610, !dbg !21
  store half %1849, ptr addrspace(1) %address.local631, align 2, !dbg !21
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local617, !dbg !21
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local632, i64 %address.local613, !dbg !21
  store half %1848, ptr addrspace(1) %address.local633, align 2, !dbg !21
  br label %1880, !dbg !21

1880:                                             ; preds = %1842, %1841
  br i1 true, label %1881, label %1919, !dbg !21

1881:                                             ; preds = %1880
  %1882 = insertelement <4 x float> poison, float %1719, i64 0, !dbg !19
  %1883 = insertelement <4 x float> %1882, float %1725, i64 1, !dbg !19
  %1884 = insertelement <4 x float> %1883, float %1720, i64 2, !dbg !19
  %1885 = insertelement <4 x float> %1884, float %1726, i64 3, !dbg !19
  %row.reciprocal82 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1670, !dbg !19
  %normalized83 = fmul <4 x float> %1885, %row.reciprocal82, !dbg !19
  %1886 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !20
  %1887 = extractelement <4 x half> %1886, i64 3, !dbg !20
  %1888 = extractelement <4 x half> %1886, i64 2, !dbg !20
  %1889 = extractelement <4 x half> %1886, i64 1, !dbg !20
  %1890 = extractelement <4 x half> %1886, i64 0, !dbg !20
  %1891 = insertelement <4 x float> poison, float %1609, i64 0, !dbg !19
  %1892 = insertelement <4 x float> %1891, float %1615, i64 1, !dbg !19
  %1893 = insertelement <4 x float> %1892, float %1610, i64 2, !dbg !19
  %1894 = insertelement <4 x float> %1893, float %1616, i64 3, !dbg !19
  %normalized84 = fmul <4 x float> %1894, %row.reciprocal82, !dbg !19
  %1895 = fptrunc <4 x float> %normalized84 to <4 x half>, !dbg !20
  %1896 = extractelement <4 x half> %1895, i64 3, !dbg !20
  %1897 = extractelement <4 x half> %1895, i64 2, !dbg !20
  %1898 = extractelement <4 x half> %1895, i64 1, !dbg !20
  %1899 = extractelement <4 x half> %1895, i64 0, !dbg !20
  %1900 = insertelement <4 x float> poison, float %1501, i64 0, !dbg !19
  %1901 = insertelement <4 x float> %1900, float %1507, i64 1, !dbg !19
  %1902 = insertelement <4 x float> %1901, float %1502, i64 2, !dbg !19
  %1903 = insertelement <4 x float> %1902, float %1508, i64 3, !dbg !19
  %row.reciprocal85 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1452, !dbg !19
  %normalized86 = fmul <4 x float> %1903, %row.reciprocal85, !dbg !19
  %1904 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !20
  %1905 = extractelement <4 x half> %1904, i64 3, !dbg !20
  %1906 = extractelement <4 x half> %1904, i64 2, !dbg !20
  %1907 = extractelement <4 x half> %1904, i64 1, !dbg !20
  %1908 = extractelement <4 x half> %1904, i64 0, !dbg !20
  %1909 = insertelement <4 x float> poison, float %1391, i64 0, !dbg !19
  %1910 = insertelement <4 x float> %1909, float %1397, i64 1, !dbg !19
  %1911 = insertelement <4 x float> %1910, float %1392, i64 2, !dbg !19
  %1912 = insertelement <4 x float> %1911, float %1398, i64 3, !dbg !19
  %normalized87 = fmul <4 x float> %1912, %row.reciprocal85, !dbg !19
  %1913 = fptrunc <4 x float> %normalized87 to <4 x half>, !dbg !20
  %1914 = extractelement <4 x half> %1913, i64 3, !dbg !20
  %1915 = extractelement <4 x half> %1913, i64 2, !dbg !20
  %1916 = extractelement <4 x half> %1913, i64 1, !dbg !20
  %1917 = extractelement <4 x half> %1913, i64 0, !dbg !20
  %1918 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread634 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local635 = zext nneg i32 %address.thread634 to i64, !dbg !13
  %address.local636 = add nsw i64 %16, %address.local635, !dbg !21
  %address.local637 = lshr i64 %address.local635, 6, !dbg !13
  %address.local638 = shl nuw nsw i64 %address.local637, 4, !dbg !21
  %address.local639 = add nsw i64 %address.local636, %address.local638, !dbg !21
  %address.local640 = and i64 %address.local635, 240, !dbg !21
  %address.local641 = sub nsw i64 %address.local639, %address.local640, !dbg !21
  %address.local642 = add nsw i64 %address.local641, %17, !dbg !21
  %address.local643 = shl nuw nsw i64 %address.local642, 7, !dbg !21
  %address.local644 = add nuw nsw i64 %1918, %address.local643, !dbg !21
  %address.local645 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local644, !dbg !21
  %address.local646 = lshr i64 %address.local635, 2, !dbg !13
  %address.local647 = and i64 %address.local646, 12, !dbg !13
  %address.local648 = or disjoint i64 %address.local647, 64, !dbg !21
  %address.local649 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local645, i64 %address.local648, !dbg !21
  store half %1917, ptr addrspace(1) %address.local649, align 2, !dbg !21
  %address.local650 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local644, !dbg !21
  %address.local651 = or disjoint i64 %address.local647, 65, !dbg !21
  %address.local652 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local650, i64 %address.local651, !dbg !21
  store half %1916, ptr addrspace(1) %address.local652, align 2, !dbg !21
  %address.local653 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local644, !dbg !21
  %address.local654 = or disjoint i64 %address.local647, 66, !dbg !21
  %address.local655 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local653, i64 %address.local654, !dbg !21
  store half %1915, ptr addrspace(1) %address.local655, align 2, !dbg !21
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local644, !dbg !21
  %address.local657 = or disjoint i64 %address.local647, 67, !dbg !21
  %address.local658 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local656, i64 %address.local657, !dbg !21
  store half %1914, ptr addrspace(1) %address.local658, align 2, !dbg !21
  %address.local659 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local644, !dbg !21
  %address.local660 = or disjoint i64 %address.local647, 80, !dbg !21
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local660, !dbg !21
  store half %1908, ptr addrspace(1) %address.local661, align 2, !dbg !21
  %address.local662 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local644, !dbg !21
  %address.local663 = or disjoint i64 %address.local647, 81, !dbg !21
  %address.local664 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local662, i64 %address.local663, !dbg !21
  store half %1907, ptr addrspace(1) %address.local664, align 2, !dbg !21
  %address.local665 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local644, !dbg !21
  %address.local666 = or disjoint i64 %address.local647, 82, !dbg !21
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local665, i64 %address.local666, !dbg !21
  store half %1906, ptr addrspace(1) %address.local667, align 2, !dbg !21
  %address.local668 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local644, !dbg !21
  %address.local669 = or disjoint i64 %address.local647, 83, !dbg !21
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local668, i64 %address.local669, !dbg !21
  store half %1905, ptr addrspace(1) %address.local670, align 2, !dbg !21
  %address.local671 = shl nsw i64 %address.local642, 7, !dbg !21
  %address.local672 = add i64 %address.local671, 8192, !dbg !21
  %address.local673 = add nuw nsw i64 %1918, %address.local672, !dbg !21
  %address.local674 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local673, !dbg !21
  %address.local675 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local674, i64 %address.local648, !dbg !21
  store half %1899, ptr addrspace(1) %address.local675, align 2, !dbg !21
  %address.local676 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local673, !dbg !21
  %address.local677 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local676, i64 %address.local651, !dbg !21
  store half %1898, ptr addrspace(1) %address.local677, align 2, !dbg !21
  %address.local678 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local673, !dbg !21
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local678, i64 %address.local654, !dbg !21
  store half %1897, ptr addrspace(1) %address.local679, align 2, !dbg !21
  %address.local680 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local673, !dbg !21
  %address.local681 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local680, i64 %address.local657, !dbg !21
  store half %1896, ptr addrspace(1) %address.local681, align 2, !dbg !21
  %address.local682 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local673, !dbg !21
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local682, i64 %address.local660, !dbg !21
  store half %1890, ptr addrspace(1) %address.local683, align 2, !dbg !21
  %address.local684 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local673, !dbg !21
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local684, i64 %address.local663, !dbg !21
  store half %1889, ptr addrspace(1) %address.local685, align 2, !dbg !21
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local673, !dbg !21
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local686, i64 %address.local666, !dbg !21
  store half %1888, ptr addrspace(1) %address.local687, align 2, !dbg !21
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local673, !dbg !21
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local688, i64 %address.local669, !dbg !21
  store half %1887, ptr addrspace(1) %address.local689, align 2, !dbg !21
  br label %1919, !dbg !21

1919:                                             ; preds = %1881, %1880
  br i1 true, label %1920, label %1958, !dbg !21

1920:                                             ; preds = %1919
  %1921 = insertelement <4 x float> poison, float %1691, i64 0, !dbg !19
  %1922 = insertelement <4 x float> %1921, float %1697, i64 1, !dbg !19
  %1923 = insertelement <4 x float> %1922, float %1692, i64 2, !dbg !19
  %1924 = insertelement <4 x float> %1923, float %1698, i64 3, !dbg !19
  %row.reciprocal88 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1670, !dbg !19
  %normalized89 = fmul <4 x float> %1924, %row.reciprocal88, !dbg !19
  %1925 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !20
  %1926 = extractelement <4 x half> %1925, i64 3, !dbg !20
  %1927 = extractelement <4 x half> %1925, i64 2, !dbg !20
  %1928 = extractelement <4 x half> %1925, i64 1, !dbg !20
  %1929 = extractelement <4 x half> %1925, i64 0, !dbg !20
  %1930 = insertelement <4 x float> poison, float %1580, i64 0, !dbg !19
  %1931 = insertelement <4 x float> %1930, float %1586, i64 1, !dbg !19
  %1932 = insertelement <4 x float> %1931, float %1581, i64 2, !dbg !19
  %1933 = insertelement <4 x float> %1932, float %1587, i64 3, !dbg !19
  %normalized90 = fmul <4 x float> %1933, %row.reciprocal88, !dbg !19
  %1934 = fptrunc <4 x float> %normalized90 to <4 x half>, !dbg !20
  %1935 = extractelement <4 x half> %1934, i64 3, !dbg !20
  %1936 = extractelement <4 x half> %1934, i64 2, !dbg !20
  %1937 = extractelement <4 x half> %1934, i64 1, !dbg !20
  %1938 = extractelement <4 x half> %1934, i64 0, !dbg !20
  %1939 = insertelement <4 x float> poison, float %1473, i64 0, !dbg !19
  %1940 = insertelement <4 x float> %1939, float %1479, i64 1, !dbg !19
  %1941 = insertelement <4 x float> %1940, float %1474, i64 2, !dbg !19
  %1942 = insertelement <4 x float> %1941, float %1480, i64 3, !dbg !19
  %row.reciprocal91 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1452, !dbg !19
  %normalized92 = fmul <4 x float> %1942, %row.reciprocal91, !dbg !19
  %1943 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !20
  %1944 = extractelement <4 x half> %1943, i64 3, !dbg !20
  %1945 = extractelement <4 x half> %1943, i64 2, !dbg !20
  %1946 = extractelement <4 x half> %1943, i64 1, !dbg !20
  %1947 = extractelement <4 x half> %1943, i64 0, !dbg !20
  %1948 = insertelement <4 x float> poison, float %1362, i64 0, !dbg !19
  %1949 = insertelement <4 x float> %1948, float %1368, i64 1, !dbg !19
  %1950 = insertelement <4 x float> %1949, float %1363, i64 2, !dbg !19
  %1951 = insertelement <4 x float> %1950, float %1369, i64 3, !dbg !19
  %normalized93 = fmul <4 x float> %1951, %row.reciprocal91, !dbg !19
  %1952 = fptrunc <4 x float> %normalized93 to <4 x half>, !dbg !20
  %1953 = extractelement <4 x half> %1952, i64 3, !dbg !20
  %1954 = extractelement <4 x half> %1952, i64 2, !dbg !20
  %1955 = extractelement <4 x half> %1952, i64 1, !dbg !20
  %1956 = extractelement <4 x half> %1952, i64 0, !dbg !20
  %1957 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread690 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local691 = zext nneg i32 %address.thread690 to i64, !dbg !13
  %address.local692 = add nsw i64 %16, %address.local691, !dbg !21
  %address.local693 = lshr i64 %address.local691, 6, !dbg !13
  %address.local694 = shl nuw nsw i64 %address.local693, 4, !dbg !21
  %address.local695 = add nsw i64 %address.local692, %address.local694, !dbg !21
  %address.local696 = and i64 %address.local691, 240, !dbg !21
  %address.local697 = sub nsw i64 %address.local695, %address.local696, !dbg !21
  %address.local698 = add nsw i64 %address.local697, %17, !dbg !21
  %address.local699 = shl nuw nsw i64 %address.local698, 7, !dbg !21
  %address.local700 = add nuw nsw i64 %1957, %address.local699, !dbg !21
  %address.local701 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local700, !dbg !21
  %address.local702 = lshr i64 %address.local691, 2, !dbg !13
  %address.local703 = and i64 %address.local702, 12, !dbg !13
  %address.local704 = or disjoint i64 %address.local703, 96, !dbg !21
  %address.local705 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local701, i64 %address.local704, !dbg !21
  store half %1956, ptr addrspace(1) %address.local705, align 2, !dbg !21
  %address.local706 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local700, !dbg !21
  %address.local707 = or disjoint i64 %address.local703, 97, !dbg !21
  %address.local708 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local706, i64 %address.local707, !dbg !21
  store half %1955, ptr addrspace(1) %address.local708, align 2, !dbg !21
  %address.local709 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local700, !dbg !21
  %address.local710 = or disjoint i64 %address.local703, 98, !dbg !21
  %address.local711 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local709, i64 %address.local710, !dbg !21
  store half %1954, ptr addrspace(1) %address.local711, align 2, !dbg !21
  %address.local712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local700, !dbg !21
  %address.local713 = or disjoint i64 %address.local703, 99, !dbg !21
  %address.local714 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local712, i64 %address.local713, !dbg !21
  store half %1953, ptr addrspace(1) %address.local714, align 2, !dbg !21
  %address.local715 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local700, !dbg !21
  %address.local716 = or disjoint i64 %address.local703, 112, !dbg !21
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local715, i64 %address.local716, !dbg !21
  store half %1947, ptr addrspace(1) %address.local717, align 2, !dbg !21
  %address.local718 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local700, !dbg !21
  %address.local719 = or disjoint i64 %address.local703, 113, !dbg !21
  %address.local720 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local718, i64 %address.local719, !dbg !21
  store half %1946, ptr addrspace(1) %address.local720, align 2, !dbg !21
  %address.local721 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local700, !dbg !21
  %address.local722 = or disjoint i64 %address.local703, 114, !dbg !21
  %address.local723 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local721, i64 %address.local722, !dbg !21
  store half %1945, ptr addrspace(1) %address.local723, align 2, !dbg !21
  %address.local724 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local700, !dbg !21
  %address.local725 = or i64 %address.local702, 115, !dbg !21
  %address.local726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local724, i64 %address.local725, !dbg !21
  store half %1944, ptr addrspace(1) %address.local726, align 2, !dbg !21
  %address.local727 = shl nsw i64 %address.local698, 7, !dbg !21
  %address.local728 = add i64 %address.local727, 8192, !dbg !21
  %address.local729 = add nuw nsw i64 %1957, %address.local728, !dbg !21
  %address.local730 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local729, !dbg !21
  %address.local731 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local730, i64 %address.local704, !dbg !21
  store half %1938, ptr addrspace(1) %address.local731, align 2, !dbg !21
  %address.local732 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local729, !dbg !21
  %address.local733 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local732, i64 %address.local707, !dbg !21
  store half %1937, ptr addrspace(1) %address.local733, align 2, !dbg !21
  %address.local734 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local729, !dbg !21
  %address.local735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local734, i64 %address.local710, !dbg !21
  store half %1936, ptr addrspace(1) %address.local735, align 2, !dbg !21
  %address.local736 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local729, !dbg !21
  %address.local737 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local736, i64 %address.local713, !dbg !21
  store half %1935, ptr addrspace(1) %address.local737, align 2, !dbg !21
  %address.local738 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local729, !dbg !21
  %address.local739 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local716, !dbg !21
  store half %1929, ptr addrspace(1) %address.local739, align 2, !dbg !21
  %address.local740 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local729, !dbg !21
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local740, i64 %address.local719, !dbg !21
  store half %1928, ptr addrspace(1) %address.local741, align 2, !dbg !21
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local729, !dbg !21
  %address.local743 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local742, i64 %address.local722, !dbg !21
  store half %1927, ptr addrspace(1) %address.local743, align 2, !dbg !21
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local729, !dbg !21
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local744, i64 %address.local725, !dbg !21
  store half %1926, ptr addrspace(1) %address.local745, align 2, !dbg !21
  br label %1958, !dbg !21

1958:                                             ; preds = %1920, %1919
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

!0 = !DIFile(filename: "qk64_vo32.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/k5_independent_tiles")
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
!15 = !DILocation(line: 93, column: 23, scope: !4)
!16 = !DILocation(line: 56, column: 24, scope: !4)
!17 = !DILocation(line: 67, column: 22, scope: !4)
!18 = !DILocation(line: 92, column: 20, scope: !4)
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
