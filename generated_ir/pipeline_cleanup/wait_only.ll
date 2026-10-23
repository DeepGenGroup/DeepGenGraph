; ModuleID = '3rd/deepgengraph/test/test_input_spidD.mlir'
source_filename = "3rd/deepgengraph/test/test_input_spidD.mlir"

@shm_4 = addrspace(3) global [4096 x half] undef, align 16
@shm_3 = addrspace(3) global [4096 x half] undef, align 16
@shm_2 = addrspace(3) global [4096 x half] undef, align 16
@shm_1 = addrspace(3) global [4096 x half] undef, align 16
@shm_0 = addrspace(3) global [8192 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !33 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %10 = sext i32 %9 to i64, !dbg !5
  %11 = shl nsw i64 %8, 7, !dbg !12
  %.lobit = ashr i32 %7, 31, !dbg !13
  %12 = xor i32 %7, %.lobit, !dbg !13
  %13 = sdiv i32 %12, 32, !dbg !13
  %.lobit1 = ashr i32 %7, 31, !dbg !13
  %14 = xor i32 %13, %.lobit1, !dbg !13
  %15 = sext i32 %14 to i64, !dbg !13
  %16 = add nsw i64 %10, %15, !dbg !13
  %17 = and i64 %16, 31, !dbg !13
  %18 = shl nsw i64 %8, 7, !dbg !13
  %19 = mul nsw i64 %15, -4096, !dbg !13
  %20 = add nsw i64 %18, %19, !dbg !13
  %21 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !14, !range !35
  %22 = zext nneg i32 %21 to i64, !dbg !14
  %23 = lshr i64 %22, 6, !dbg !14
  %24 = or disjoint i64 %11, 127, !dbg !15
  %25 = add nsw i64 %11, 128, !dbg !15
  %26 = icmp uge i64 %25, %24, !dbg !15
  %.idx = shl nuw nsw i64 %17, 20, !dbg !13
  %27 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !13
  %.idx2 = shl nsw i64 %20, 8, !dbg !13
  %28 = getelementptr i8, ptr addrspace(1) %27, i64 %.idx2, !dbg !13
  br label %29, !dbg !17

29:                                               ; preds = %535, %4
  %30 = phi i64 [ %1286, %535 ], [ 0, %4 ], !dbg !17
  %31 = phi [2 x <8 x float>] [ %1273, %535 ], [ zeroinitializer, %4 ], !dbg !17
  %32 = phi [2 x <1 x float>] [ %895, %535 ], [ zeroinitializer, %4 ], !dbg !17
  %33 = phi [2 x <8 x float>] [ %1277, %535 ], [ zeroinitializer, %4 ], !dbg !17
  %34 = phi [2 x <8 x float>] [ %1281, %535 ], [ zeroinitializer, %4 ], !dbg !17
  %35 = phi [2 x <8 x float>] [ %1285, %535 ], [ zeroinitializer, %4 ], !dbg !17
  %.not = icmp sgt i64 %30, %8, !dbg !17
  br i1 %.not, label %1287, label %36, !dbg !17

36:                                               ; preds = %29
  %37 = shl nsw i64 %30, 7, !dbg !17
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !13, !range !35, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !14
  %address.local132 = shl nuw nsw i64 %address.local, 3, !dbg !13
  %address.local133 = and i64 %address.local132, 1536, !dbg !13
  %address.local134 = getelementptr [2 x i8], ptr addrspace(3) @shm_1, i64 %address.local133, !dbg !13
  %address.local135 = ptrtoint ptr addrspace(3) %address.local134 to i64, !dbg !13
  %address.local136 = trunc i64 %address.local135 to i32, !dbg !13
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local136), !dbg !13
  %39 = zext i32 %38 to i64, !dbg !13
  %40 = inttoptr i64 %39 to ptr addrspace(3), !dbg !13
  %address.local137 = shl nuw nsw i64 %address.local, 7, !dbg !13
  %address.local138 = and i64 %address.local137, 16256, !dbg !13
  %address.local139 = lshr i64 %address.local, 4, !dbg !13
  %address.local140 = and i64 %address.local139, 8, !dbg !13
  %address.local141 = or disjoint i64 %address.local138, %address.local140, !dbg !13
  %address.local142 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %address.local141, !dbg !13
  %41 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !13
  %42 = ptrtoint ptr addrspace(1) %address.local142 to i64, !dbg !13
  %43 = sub i64 %42, %41, !dbg !13
  %dma.buffer.offset949 = trunc i64 %43 to i32, !dbg !13
  %44 = trunc i64 %41 to i32, !dbg !13
  %45 = insertelement <4 x i32> poison, i32 %44, i64 0, !dbg !13
  %46 = lshr i64 %41, 32, !dbg !13
  %47 = trunc i64 %46 to i32, !dbg !13
  %48 = insertelement <4 x i32> %45, i32 %47, i64 1, !dbg !13
  %49 = insertelement <4 x i32> %48, i32 33554432, i64 2, !dbg !13
  %dma.buffer.rsrc950 = insertelement <4 x i32> %49, i32 131072, i64 3, !dbg !13
  %50 = ptrtoint ptr addrspace(3) %40 to i32, !dbg !13
  %dma.buffer.lds951 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !13
  %51 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds951, i32 %dma.buffer.offset949, <4 x i32> %dma.buffer.rsrc950) #10, !dbg !13
  %address.local143 = or disjoint i64 %address.local, 256, !dbg !13
  %address.local144 = shl nuw nsw i64 %address.local143, 3, !dbg !13
  %address.local145 = and i64 %address.local144, 3584, !dbg !13
  %address.local146 = getelementptr [2 x i8], ptr addrspace(3) @shm_1, i64 %address.local145, !dbg !13
  %address.local147 = ptrtoint ptr addrspace(3) %address.local146 to i64, !dbg !13
  %address.local148 = trunc i64 %address.local147 to i32, !dbg !13
  %52 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local148), !dbg !13
  %53 = zext i32 %52 to i64, !dbg !13
  %54 = inttoptr i64 %53 to ptr addrspace(3), !dbg !13
  %address.local149 = shl nuw nsw i64 %address.local, 7, !dbg !13
  %address.local150 = and i64 %address.local149, 16256, !dbg !13
  %address.local151 = lshr i64 %address.local143, 4, !dbg !13
  %address.local152 = and i64 %address.local151, 24, !dbg !13
  %address.local153 = or disjoint i64 %address.local150, %address.local152, !dbg !13
  %address.local154 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %address.local153, !dbg !13
  %55 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !13
  %56 = ptrtoint ptr addrspace(1) %address.local154 to i64, !dbg !13
  %57 = sub i64 %56, %55, !dbg !13
  %dma.buffer.offset946 = trunc i64 %57 to i32, !dbg !13
  %58 = trunc i64 %55 to i32, !dbg !13
  %59 = insertelement <4 x i32> poison, i32 %58, i64 0, !dbg !13
  %60 = lshr i64 %55, 32, !dbg !13
  %61 = trunc i64 %60 to i32, !dbg !13
  %62 = insertelement <4 x i32> %59, i32 %61, i64 1, !dbg !13
  %63 = insertelement <4 x i32> %62, i32 33554432, i64 2, !dbg !13
  %dma.buffer.rsrc947 = insertelement <4 x i32> %63, i32 131072, i64 3, !dbg !13
  %64 = ptrtoint ptr addrspace(3) %54 to i32, !dbg !13
  %dma.buffer.lds948 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %64), !dbg !13
  %65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds948, i32 %dma.buffer.offset946, <4 x i32> %dma.buffer.rsrc947) #10, !dbg !13
  %66 = lshr i64 %30, 12, !dbg !18
  %67 = add nsw i64 %66, %10, !dbg !18
  %68 = lshr i64 %30, 5, !dbg !18
  %69 = and i64 %68, 127, !dbg !18
  %70 = shl i64 %67, 19, !dbg !18
  %71 = and i64 %70, 16252928, !dbg !18
  %72 = shl i64 %68, 19, !dbg !18
  %73 = shl i64 %30, 14, !dbg !18
  %74 = sub i64 %73, %72, !dbg !18
  %75 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %71, !dbg !18
  %76 = getelementptr [2 x i8], ptr addrspace(1) %75, i64 %69, !dbg !18
  %77 = getelementptr [2 x i8], ptr addrspace(1) %76, i64 %74, !dbg !18
  %address.local155 = getelementptr [2 x i8], ptr addrspace(3) @shm_3, i64 %address.local133, !dbg !18
  %address.local156 = ptrtoint ptr addrspace(3) %address.local155 to i64, !dbg !18
  %address.local157 = trunc i64 %address.local156 to i32, !dbg !18
  %78 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local157), !dbg !18
  %79 = zext i32 %78 to i64, !dbg !18
  %80 = inttoptr i64 %79 to ptr addrspace(3), !dbg !18
  %address.local158 = getelementptr [2 x i8], ptr addrspace(1) %77, i64 %address.local141, !dbg !18
  %81 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %82 = ptrtoint ptr addrspace(1) %address.local158 to i64, !dbg !18
  %83 = sub i64 %82, %81, !dbg !18
  %dma.buffer.offset943 = trunc i64 %83 to i32, !dbg !18
  %84 = trunc i64 %81 to i32, !dbg !18
  %85 = insertelement <4 x i32> poison, i32 %84, i64 0, !dbg !18
  %86 = lshr i64 %81, 32, !dbg !18
  %87 = trunc i64 %86 to i32, !dbg !18
  %88 = insertelement <4 x i32> %85, i32 %87, i64 1, !dbg !18
  %89 = insertelement <4 x i32> %88, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc944 = insertelement <4 x i32> %89, i32 131072, i64 3, !dbg !18
  %90 = ptrtoint ptr addrspace(3) %80 to i32, !dbg !18
  %dma.buffer.lds945 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %90), !dbg !18
  %91 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds945, i32 %dma.buffer.offset943, <4 x i32> %dma.buffer.rsrc944) #10, !dbg !18
  %address.local159 = getelementptr [2 x i8], ptr addrspace(3) @shm_3, i64 %address.local145, !dbg !18
  %address.local160 = ptrtoint ptr addrspace(3) %address.local159 to i64, !dbg !18
  %address.local161 = trunc i64 %address.local160 to i32, !dbg !18
  %92 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local161), !dbg !18
  %93 = zext i32 %92 to i64, !dbg !18
  %94 = inttoptr i64 %93 to ptr addrspace(3), !dbg !18
  %address.local162 = getelementptr [2 x i8], ptr addrspace(1) %77, i64 %address.local153, !dbg !18
  %95 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %96 = ptrtoint ptr addrspace(1) %address.local162 to i64, !dbg !18
  %97 = sub i64 %96, %95, !dbg !18
  %dma.buffer.offset940 = trunc i64 %97 to i32, !dbg !18
  %98 = trunc i64 %95 to i32, !dbg !18
  %99 = insertelement <4 x i32> poison, i32 %98, i64 0, !dbg !18
  %100 = lshr i64 %95, 32, !dbg !18
  %101 = trunc i64 %100 to i32, !dbg !18
  %102 = insertelement <4 x i32> %99, i32 %101, i64 1, !dbg !18
  %103 = insertelement <4 x i32> %102, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc941 = insertelement <4 x i32> %103, i32 131072, i64 3, !dbg !18
  %104 = ptrtoint ptr addrspace(3) %94 to i32, !dbg !18
  %dma.buffer.lds942 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %104), !dbg !18
  %105 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds942, i32 %dma.buffer.offset940, <4 x i32> %dma.buffer.rsrc941) #10, !dbg !18
  br label %106, !dbg !23

106:                                              ; preds = %229, %36
  %107 = phi i64 [ %350, %229 ], [ 0, %36 ], !dbg !23
  %108 = phi [2 x <32 x float>] [ %349, %229 ], [ zeroinitializer, %36 ], !dbg !23
  %109 = icmp samesign ult i64 %107, 4, !dbg !23
  br i1 %109, label %110, label %351, !dbg !23

110:                                              ; preds = %106
  %111 = and i64 %107, 1, !dbg !23
  %112 = icmp eq i64 %111, 0, !dbg !23
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %.not69 = icmp eq i64 %107, 3, !dbg !23
  br i1 %.not69, label %229, label %113, !dbg !23

113:                                              ; preds = %110
  %114 = shl nuw nsw i64 %107, 5, !dbg !23
  %115 = and i64 %107, 1, !dbg !23
  %.not70 = icmp eq i64 %115, 0, !dbg !23
  %116 = shl nsw i64 %8, 9, !dbg !13
  %117 = or disjoint i64 %107, %116, !dbg !13
  %118 = shl nsw i64 %10, 14, !dbg !13
  %119 = add nsw i64 %117, %118, !dbg !13
  %120 = shl nsw i64 %6, 19, !dbg !13
  %121 = add nsw i64 %119, %120, !dbg !13
  %122 = add nuw nsw i64 %121, 1, !dbg !13
  %123 = icmp slt i64 %121, -1, !dbg !13
  %124 = sub nuw nsw i64 -2, %121, !dbg !13
  %125 = select i1 %123, i64 %124, i64 %122, !dbg !13
  %126 = sdiv i64 %125, 16384, !dbg !13
  %127 = sext i1 %123 to i64, !dbg !13
  %128 = xor i64 %126, %127, !dbg !13
  %129 = add nuw nsw i64 %107, 1, !dbg !13
  %130 = lshr i64 %129, 2, !dbg !13
  %131 = shl nsw i64 %10, 12, !dbg !13
  %132 = shl nsw i64 %6, 17, !dbg !13
  %.masked = and i64 %18, 3968, !dbg !13
  %133 = or disjoint i64 %130, %.masked, !dbg !13
  %134 = shl i64 %128, 19, !dbg !13
  %135 = and i64 %134, 16252928, !dbg !13
  %136 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %135, !dbg !13
  %.idx71 = shl nuw nsw i64 %133, 8, !dbg !13
  %137 = getelementptr i8, ptr addrspace(1) %136, i64 %.idx71, !dbg !13
  %138 = getelementptr [2 x i8], ptr addrspace(1) %137, i64 %114, !dbg !13
  %.idx72 = mul nuw nsw i64 %130, -256, !dbg !13
  %139 = getelementptr i8, ptr addrspace(1) %138, i64 %.idx72, !dbg !13
  %140 = getelementptr i8, ptr addrspace(1) %139, i64 64, !dbg !13
  %141 = select i1 %.not70, ptr addrspace(3) @shm_2, ptr addrspace(3) @shm_1, !dbg !23
  %address.thread163 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !13, !range !35, !frisk.register_lifetimes !1
  %address.local164 = zext nneg i32 %address.thread163 to i64, !dbg !14
  %address.local165 = shl nuw nsw i64 %address.local164, 3, !dbg !13
  %address.local166 = and i64 %address.local165, 1536, !dbg !13
  %address.local167 = getelementptr [2 x i8], ptr addrspace(3) %141, i64 %address.local166, !dbg !13
  %address.local168 = ptrtoint ptr addrspace(3) %address.local167 to i64, !dbg !13
  %address.local169 = trunc i64 %address.local168 to i32, !dbg !13
  %142 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local169), !dbg !13
  %143 = zext i32 %142 to i64, !dbg !13
  %144 = inttoptr i64 %143 to ptr addrspace(3), !dbg !13
  %address.local170 = shl nuw nsw i64 %address.local164, 7, !dbg !13
  %address.local171 = and i64 %address.local170, 16256, !dbg !13
  %address.local172 = lshr i64 %address.local164, 4, !dbg !13
  %address.local173 = and i64 %address.local172, 8, !dbg !13
  %address.local174 = or disjoint i64 %address.local171, %address.local173, !dbg !13
  %address.local175 = getelementptr [2 x i8], ptr addrspace(1) %140, i64 %address.local174, !dbg !13
  %145 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !13
  %146 = ptrtoint ptr addrspace(1) %address.local175 to i64, !dbg !13
  %147 = sub i64 %146, %145, !dbg !13
  %dma.buffer.offset913 = trunc i64 %147 to i32, !dbg !13
  %148 = trunc i64 %145 to i32, !dbg !13
  %149 = insertelement <4 x i32> poison, i32 %148, i64 0, !dbg !13
  %150 = lshr i64 %145, 32, !dbg !13
  %151 = trunc i64 %150 to i32, !dbg !13
  %152 = insertelement <4 x i32> %149, i32 %151, i64 1, !dbg !13
  %153 = insertelement <4 x i32> %152, i32 33554432, i64 2, !dbg !13
  %dma.buffer.rsrc914 = insertelement <4 x i32> %153, i32 131072, i64 3, !dbg !13
  %154 = ptrtoint ptr addrspace(3) %144 to i32, !dbg !13
  %dma.buffer.lds915 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %154), !dbg !13
  %155 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds915, i32 %dma.buffer.offset913, <4 x i32> %dma.buffer.rsrc914) #10, !dbg !13
  %address.local176 = or disjoint i64 %address.local164, 256, !dbg !13
  %address.local177 = shl nuw nsw i64 %address.local176, 3, !dbg !13
  %address.local178 = and i64 %address.local177, 3584, !dbg !13
  %address.local179 = getelementptr [2 x i8], ptr addrspace(3) %141, i64 %address.local178, !dbg !13
  %address.local180 = ptrtoint ptr addrspace(3) %address.local179 to i64, !dbg !13
  %address.local181 = trunc i64 %address.local180 to i32, !dbg !13
  %156 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local181), !dbg !13
  %157 = zext i32 %156 to i64, !dbg !13
  %158 = inttoptr i64 %157 to ptr addrspace(3), !dbg !13
  %address.local182 = shl nuw nsw i64 %address.local164, 7, !dbg !13
  %address.local183 = and i64 %address.local182, 16256, !dbg !13
  %address.local184 = lshr i64 %address.local176, 4, !dbg !13
  %address.local185 = and i64 %address.local184, 24, !dbg !13
  %address.local186 = or disjoint i64 %address.local183, %address.local185, !dbg !13
  %address.local187 = getelementptr [2 x i8], ptr addrspace(1) %140, i64 %address.local186, !dbg !13
  %159 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !13
  %160 = ptrtoint ptr addrspace(1) %address.local187 to i64, !dbg !13
  %161 = sub i64 %160, %159, !dbg !13
  %dma.buffer.offset910 = trunc i64 %161 to i32, !dbg !13
  %162 = trunc i64 %159 to i32, !dbg !13
  %163 = insertelement <4 x i32> poison, i32 %162, i64 0, !dbg !13
  %164 = lshr i64 %159, 32, !dbg !13
  %165 = trunc i64 %164 to i32, !dbg !13
  %166 = insertelement <4 x i32> %163, i32 %165, i64 1, !dbg !13
  %167 = insertelement <4 x i32> %166, i32 33554432, i64 2, !dbg !13
  %dma.buffer.rsrc911 = insertelement <4 x i32> %167, i32 131072, i64 3, !dbg !13
  %168 = ptrtoint ptr addrspace(3) %158 to i32, !dbg !13
  %dma.buffer.lds912 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %168), !dbg !13
  %169 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds912, i32 %dma.buffer.offset910, <4 x i32> %dma.buffer.rsrc911) #10, !dbg !13
  %170 = shl nuw nsw i64 %107, 10, !dbg !18
  %171 = add nuw i64 %30, %170, !dbg !18
  %172 = add i64 %171, %131, !dbg !18
  %173 = add i64 %172, %132, !dbg !18
  %174 = add i64 %173, 1024, !dbg !18
  %175 = icmp slt i64 %174, 0, !dbg !18
  %176 = sub i64 -1025, %173, !dbg !18
  %177 = select i1 %175, i64 %176, i64 %174, !dbg !18
  %178 = sdiv i64 %177, 4096, !dbg !18
  %.lobit73 = ashr i64 %174, 63, !dbg !18
  %179 = xor i64 %178, %.lobit73, !dbg !18
  %180 = shl nsw i64 %10, 7, !dbg !18
  %181 = or disjoint i64 %114, %180, !dbg !18
  %182 = shl nsw i64 %6, 12, !dbg !18
  %183 = add nsw i64 %181, %182, !dbg !18
  %184 = add nsw i64 %183, %68, !dbg !18
  %185 = add nsw i64 %184, 32, !dbg !18
  %186 = icmp slt i64 %184, -32, !dbg !18
  %187 = sub nsw i64 -33, %184, !dbg !18
  %188 = select i1 %186, i64 %187, i64 %185, !dbg !18
  %189 = sdiv i64 %188, 128, !dbg !18
  %190 = sext i1 %186 to i64, !dbg !18
  %191 = xor i64 %189, %190, !dbg !18
  %192 = shl nsw i64 %191, 7, !dbg !18
  %193 = sub i64 %184, %192, !dbg !18
  %194 = shl i64 %179, 19, !dbg !18
  %195 = and i64 %194, 16252928, !dbg !18
  %196 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %195, !dbg !18
  %197 = getelementptr [2 x i8], ptr addrspace(1) %196, i64 %193, !dbg !18
  %198 = getelementptr i8, ptr addrspace(1) %197, i64 64, !dbg !18
  %199 = getelementptr [2 x i8], ptr addrspace(1) %198, i64 %74, !dbg !18
  %200 = select i1 %.not70, ptr addrspace(3) @shm_4, ptr addrspace(3) @shm_3, !dbg !23
  %address.local188 = getelementptr [2 x i8], ptr addrspace(3) %200, i64 %address.local166, !dbg !18
  %address.local189 = ptrtoint ptr addrspace(3) %address.local188 to i64, !dbg !18
  %address.local190 = trunc i64 %address.local189 to i32, !dbg !18
  %201 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local190), !dbg !18
  %202 = zext i32 %201 to i64, !dbg !18
  %203 = inttoptr i64 %202 to ptr addrspace(3), !dbg !18
  %address.local191 = getelementptr [2 x i8], ptr addrspace(1) %199, i64 %address.local174, !dbg !18
  %204 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %205 = ptrtoint ptr addrspace(1) %address.local191 to i64, !dbg !18
  %206 = sub i64 %205, %204, !dbg !18
  %dma.buffer.offset907 = trunc i64 %206 to i32, !dbg !18
  %207 = trunc i64 %204 to i32, !dbg !18
  %208 = insertelement <4 x i32> poison, i32 %207, i64 0, !dbg !18
  %209 = lshr i64 %204, 32, !dbg !18
  %210 = trunc i64 %209 to i32, !dbg !18
  %211 = insertelement <4 x i32> %208, i32 %210, i64 1, !dbg !18
  %212 = insertelement <4 x i32> %211, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc908 = insertelement <4 x i32> %212, i32 131072, i64 3, !dbg !18
  %213 = ptrtoint ptr addrspace(3) %203 to i32, !dbg !18
  %dma.buffer.lds909 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %213), !dbg !18
  %214 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds909, i32 %dma.buffer.offset907, <4 x i32> %dma.buffer.rsrc908) #10, !dbg !18
  %address.local192 = getelementptr [2 x i8], ptr addrspace(3) %200, i64 %address.local178, !dbg !18
  %address.local193 = ptrtoint ptr addrspace(3) %address.local192 to i64, !dbg !18
  %address.local194 = trunc i64 %address.local193 to i32, !dbg !18
  %215 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local194), !dbg !18
  %216 = zext i32 %215 to i64, !dbg !18
  %217 = inttoptr i64 %216 to ptr addrspace(3), !dbg !18
  %address.local195 = getelementptr [2 x i8], ptr addrspace(1) %199, i64 %address.local186, !dbg !18
  %218 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %219 = ptrtoint ptr addrspace(1) %address.local195 to i64, !dbg !18
  %220 = sub i64 %219, %218, !dbg !18
  %dma.buffer.offset = trunc i64 %220 to i32, !dbg !18
  %221 = trunc i64 %218 to i32, !dbg !18
  %222 = insertelement <4 x i32> poison, i32 %221, i64 0, !dbg !18
  %223 = lshr i64 %218, 32, !dbg !18
  %224 = trunc i64 %223 to i32, !dbg !18
  %225 = insertelement <4 x i32> %222, i32 %224, i64 1, !dbg !18
  %226 = insertelement <4 x i32> %225, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %226, i32 131072, i64 3, !dbg !18
  %227 = ptrtoint ptr addrspace(3) %217 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %227), !dbg !18
  %228 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  br label %229, !dbg !23

229:                                              ; preds = %113, %110
  %230 = select i1 %112, ptr addrspace(3) @shm_1, ptr addrspace(3) @shm_2, !dbg !23
  %address.thread196 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !14, !range !35, !frisk.register_lifetimes !1
  %address.local197 = lshr i32 %address.thread196, 2, !dbg !14
  %address.local198 = and i32 %address.local197, 4, !dbg !14
  %address.local199 = shl nuw nsw i32 %address.thread196, 1, !dbg !14
  %address.local200 = and i32 %address.local199, 384, !dbg !14
  %address.local201 = or disjoint i32 %address.local198, %address.local200, !dbg !14
  %address.local202 = shl nuw nsw i32 %address.thread196, 3, !dbg !14
  %address.local203 = and i32 %address.local202, 120, !dbg !14
  %address.local204 = or disjoint i32 %address.local201, %address.local203, !dbg !14
  %address.local205 = shl nuw nsw i32 %address.thread196, 5, !dbg !14
  %address.local206 = and i32 %address.local205, 1024, !dbg !14
  %address.local207 = or disjoint i32 %address.local204, %address.local206, !dbg !14
  %address.local208 = zext nneg i32 %address.local207 to i64, !dbg !14
  %address.local209 = getelementptr [2 x i8], ptr addrspace(3) %230, i64 %address.local208, !dbg !14
  %231 = load <4 x half>, ptr addrspace(3) %address.local209, align 2, !dbg !14
  %232 = extractvalue [2 x <32 x float>] %108, 0, !dbg !32
  %233 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %234 = fmul <4 x half> %231, splat (half 0xH3015), !dbg !14
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) %230, i64 %address.local208, !dbg !14
  %address.local211 = getelementptr i8, ptr addrspace(3) %address.local210, i64 4096, !dbg !14
  %235 = load <4 x half>, ptr addrspace(3) %address.local211, align 2, !dbg !14
  %236 = fmul <4 x half> %235, splat (half 0xH3015), !dbg !14
  %address.local212 = getelementptr [2 x i8], ptr addrspace(3) %230, i64 %address.local208, !dbg !14
  %address.local213 = getelementptr i8, ptr addrspace(3) %address.local212, i64 1024, !dbg !14
  %237 = load <4 x half>, ptr addrspace(3) %address.local213, align 2, !dbg !14
  %238 = fmul <4 x half> %237, splat (half 0xH3015), !dbg !14
  %239 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %240 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !32
  %241 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !32
  %242 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !32
  %243 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !32
  %244 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !32
  %245 = shufflevector <32 x float> %232, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !32
  %246 = extractvalue [2 x <32 x float>] %108, 1, !dbg !32
  %247 = shufflevector <32 x float> %246, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %248 = shufflevector <32 x float> %246, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %249 = shufflevector <32 x float> %246, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !32
  %250 = shufflevector <32 x float> %246, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !32
  %251 = shufflevector <32 x float> %246, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !32
  %252 = shufflevector <32 x float> %246, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !32
  %253 = shufflevector <32 x float> %246, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !32
  %254 = shufflevector <32 x float> %246, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !32
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %230, i64 %address.local208, !dbg !14
  %address.local215 = getelementptr i8, ptr addrspace(3) %address.local214, i64 5120, !dbg !14
  %255 = load <4 x half>, ptr addrspace(3) %address.local215, align 2, !dbg !14
  %256 = select i1 %112, ptr addrspace(3) @shm_3, ptr addrspace(3) @shm_4, !dbg !23
  %address.local216 = lshr i32 %address.thread196, 2, !dbg !10
  %address.local217 = and i32 %address.local216, 4, !dbg !10
  %address.local218 = shl nuw nsw i32 %address.thread196, 1, !dbg !10
  %address.local219 = and i32 %address.local218, 24, !dbg !10
  %address.local220 = or disjoint i32 %address.local217, %address.local219, !dbg !10
  %address.local221 = shl nuw nsw i32 %address.thread196, 5, !dbg !10
  %address.local222 = and i32 %address.local221, 1024, !dbg !10
  %address.local223 = or disjoint i32 %address.local220, %address.local222, !dbg !10
  %address.local224 = shl nuw nsw i32 %address.thread196, 7, !dbg !10
  %address.local225 = and i32 %address.local224, 384, !dbg !10
  %address.local226 = or disjoint i32 %address.local223, %address.local225, !dbg !10
  %address.local227 = zext nneg i32 %address.local226 to i64, !dbg !10
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local227, !dbg !10
  %257 = load <4 x half>, ptr addrspace(3) %address.local228, align 2, !dbg !10
  %258 = fmul <4 x half> %255, splat (half 0xH3015), !dbg !14
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %257, <4 x float> %233) #9, !dbg !10
  %address.local229 = lshr i32 %address.thread196, 2, !dbg !10
  %address.local230 = and i32 %address.local229, 4, !dbg !10
  %address.local231 = shl nuw nsw i32 %address.thread196, 1, !dbg !10
  %address.local232 = and i32 %address.local231, 24, !dbg !10
  %address.local233 = or disjoint i32 %address.local230, %address.local232, !dbg !10
  %address.local234 = shl nuw nsw i32 %address.thread196, 5, !dbg !10
  %address.local235 = and i32 %address.local234, 1024, !dbg !10
  %address.local236 = or disjoint i32 %address.local233, %address.local235, !dbg !10
  %address.local237 = shl nuw nsw i32 %address.thread196, 7, !dbg !10
  %address.local238 = and i32 %address.local237, 384, !dbg !10
  %address.local239 = or disjoint i32 %address.local236, %address.local238, !dbg !10
  %address.local240 = zext nneg i32 %address.local239 to i64, !dbg !10
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local240, !dbg !10
  %address.local242 = getelementptr i8, ptr addrspace(3) %address.local241, i64 4096, !dbg !10
  %260 = load <4 x half>, ptr addrspace(3) %address.local242, align 2, !dbg !10
  %address.local243 = zext nneg i32 %address.thread196 to i64, !dbg !14
  %address.local244 = shl nuw nsw i64 %address.local243, 7, !dbg !10
  %address.local245 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local244, !dbg !10
  %address.local246 = lshr i64 %address.local243, 2, !dbg !10
  %address.local247 = mul nsw i64 %address.local246, -512, !dbg !10
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local245, i64 %address.local247, !dbg !10
  %address.local249 = shl nuw nsw i64 %address.local243, 1, !dbg !10
  %address.local250 = and i64 %address.local249, 24, !dbg !10
  %address.local251 = getelementptr [2 x i8], ptr addrspace(3) %address.local248, i64 %address.local250, !dbg !10
  %address.local252 = lshr i64 %address.local243, 2, !dbg !14
  %address.local253 = and i64 %address.local252, 12, !dbg !14
  %address.local254 = getelementptr [2 x i8], ptr addrspace(3) %address.local251, i64 %address.local253, !dbg !10
  %address.local255 = and i64 %address.local243, 63, !dbg !14
  %address.local256 = lshr i64 %address.local255, 5, !dbg !14
  %address.local257 = mul nuw nsw i64 %address.local256, 1016, !dbg !14
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) %address.local254, i64 %address.local257, !dbg !10
  %address.local259 = getelementptr i8, ptr addrspace(3) %address.local258, i64 64, !dbg !10
  %261 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !10
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %236, <4 x half> %260, <4 x float> %259) #9, !dbg !10
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %261, <4 x float> %239) #9, !dbg !10
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local244, !dbg !10
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %address.local260, i64 %address.local253, !dbg !10
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local257, !dbg !10
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) %address.local262, i64 %address.local247, !dbg !10
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local250, !dbg !10
  %address.local265 = getelementptr i8, ptr addrspace(3) %address.local264, i64 4160, !dbg !10
  %264 = load <4 x half>, ptr addrspace(3) %address.local265, align 2, !dbg !10
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local244, !dbg !10
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %address.local266, i64 %address.local247, !dbg !10
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) %address.local267, i64 %address.local250, !dbg !10
  %address.local269 = getelementptr [2 x i8], ptr addrspace(3) %address.local268, i64 %address.local253, !dbg !10
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) %address.local269, i64 %address.local257, !dbg !10
  %address.local271 = getelementptr i8, ptr addrspace(3) %address.local270, i64 128, !dbg !10
  %265 = load <4 x half>, ptr addrspace(3) %address.local271, align 2, !dbg !10
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %236, <4 x half> %264, <4 x float> %263) #9, !dbg !10
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %265, <4 x float> %240) #9, !dbg !10
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local244, !dbg !10
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) %address.local272, i64 %address.local253, !dbg !10
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) %address.local273, i64 %address.local257, !dbg !10
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) %address.local274, i64 %address.local247, !dbg !10
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) %address.local275, i64 %address.local250, !dbg !10
  %address.local277 = getelementptr i8, ptr addrspace(3) %address.local276, i64 4224, !dbg !10
  %268 = load <4 x half>, ptr addrspace(3) %address.local277, align 2, !dbg !10
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local244, !dbg !10
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) %address.local278, i64 %address.local247, !dbg !10
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) %address.local279, i64 %address.local250, !dbg !10
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) %address.local280, i64 %address.local253, !dbg !10
  %address.local282 = getelementptr [2 x i8], ptr addrspace(3) %address.local281, i64 %address.local257, !dbg !10
  %address.local283 = getelementptr i8, ptr addrspace(3) %address.local282, i64 192, !dbg !10
  %269 = load <4 x half>, ptr addrspace(3) %address.local283, align 2, !dbg !10
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %236, <4 x half> %268, <4 x float> %267) #9, !dbg !10
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %269, <4 x float> %241) #9, !dbg !10
  %272 = shufflevector <4 x float> %270, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local244, !dbg !10
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) %address.local284, i64 %address.local253, !dbg !10
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) %address.local285, i64 %address.local257, !dbg !10
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) %address.local286, i64 %address.local247, !dbg !10
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) %address.local287, i64 %address.local250, !dbg !10
  %address.local289 = getelementptr i8, ptr addrspace(3) %address.local288, i64 4288, !dbg !10
  %273 = load <4 x half>, ptr addrspace(3) %address.local289, align 2, !dbg !10
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local227, !dbg !10
  %address.local291 = getelementptr i8, ptr addrspace(3) %address.local290, i64 1024, !dbg !10
  %274 = load <4 x half>, ptr addrspace(3) %address.local291, align 2, !dbg !10
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %236, <4 x half> %273, <4 x float> %271) #9, !dbg !10
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %274, <4 x float> %242) #9, !dbg !10
  %277 = shufflevector <4 x float> %275, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %address.local292 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local240, !dbg !10
  %address.local293 = getelementptr i8, ptr addrspace(3) %address.local292, i64 5120, !dbg !10
  %278 = load <4 x half>, ptr addrspace(3) %address.local293, align 2, !dbg !10
  %address.local294 = add nuw nsw i64 %address.local244, 512, !dbg !10
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local294, !dbg !10
  %address.local296 = getelementptr [2 x i8], ptr addrspace(3) %address.local295, i64 %address.local247, !dbg !10
  %address.local297 = getelementptr [2 x i8], ptr addrspace(3) %address.local296, i64 %address.local250, !dbg !10
  %address.local298 = getelementptr [2 x i8], ptr addrspace(3) %address.local297, i64 %address.local253, !dbg !10
  %address.local299 = getelementptr [2 x i8], ptr addrspace(3) %address.local298, i64 %address.local257, !dbg !10
  %address.local300 = getelementptr i8, ptr addrspace(3) %address.local299, i64 64, !dbg !10
  %279 = load <4 x half>, ptr addrspace(3) %address.local300, align 2, !dbg !10
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %236, <4 x half> %278, <4 x float> %276) #9, !dbg !10
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %279, <4 x float> %243) #9, !dbg !10
  %282 = shufflevector <4 x float> %280, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %address.local301 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local294, !dbg !10
  %address.local302 = getelementptr [2 x i8], ptr addrspace(3) %address.local301, i64 %address.local253, !dbg !10
  %address.local303 = getelementptr [2 x i8], ptr addrspace(3) %address.local302, i64 %address.local257, !dbg !10
  %address.local304 = getelementptr [2 x i8], ptr addrspace(3) %address.local303, i64 %address.local247, !dbg !10
  %address.local305 = getelementptr [2 x i8], ptr addrspace(3) %address.local304, i64 %address.local250, !dbg !10
  %address.local306 = getelementptr i8, ptr addrspace(3) %address.local305, i64 4160, !dbg !10
  %283 = load <4 x half>, ptr addrspace(3) %address.local306, align 2, !dbg !10
  %address.local307 = add nuw nsw i64 %address.local244, 512, !dbg !10
  %address.local308 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local307, !dbg !10
  %address.local309 = getelementptr [2 x i8], ptr addrspace(3) %address.local308, i64 %address.local247, !dbg !10
  %address.local310 = getelementptr [2 x i8], ptr addrspace(3) %address.local309, i64 %address.local250, !dbg !10
  %address.local311 = getelementptr [2 x i8], ptr addrspace(3) %address.local310, i64 %address.local253, !dbg !10
  %address.local312 = getelementptr [2 x i8], ptr addrspace(3) %address.local311, i64 %address.local257, !dbg !10
  %address.local313 = getelementptr i8, ptr addrspace(3) %address.local312, i64 128, !dbg !10
  %284 = load <4 x half>, ptr addrspace(3) %address.local313, align 2, !dbg !10
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %236, <4 x half> %283, <4 x float> %281) #9, !dbg !10
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %284, <4 x float> %244) #9, !dbg !10
  %287 = shufflevector <4 x float> %285, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %address.local314 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local307, !dbg !10
  %address.local315 = getelementptr [2 x i8], ptr addrspace(3) %address.local314, i64 %address.local253, !dbg !10
  %address.local316 = getelementptr [2 x i8], ptr addrspace(3) %address.local315, i64 %address.local257, !dbg !10
  %address.local317 = getelementptr [2 x i8], ptr addrspace(3) %address.local316, i64 %address.local247, !dbg !10
  %address.local318 = getelementptr [2 x i8], ptr addrspace(3) %address.local317, i64 %address.local250, !dbg !10
  %address.local319 = getelementptr i8, ptr addrspace(3) %address.local318, i64 4224, !dbg !10
  %288 = load <4 x half>, ptr addrspace(3) %address.local319, align 2, !dbg !10
  %address.local320 = add nuw nsw i64 %address.local244, 512, !dbg !10
  %address.local321 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local320, !dbg !10
  %address.local322 = getelementptr [2 x i8], ptr addrspace(3) %address.local321, i64 %address.local247, !dbg !10
  %address.local323 = getelementptr [2 x i8], ptr addrspace(3) %address.local322, i64 %address.local250, !dbg !10
  %address.local324 = getelementptr [2 x i8], ptr addrspace(3) %address.local323, i64 %address.local253, !dbg !10
  %address.local325 = getelementptr [2 x i8], ptr addrspace(3) %address.local324, i64 %address.local257, !dbg !10
  %address.local326 = getelementptr i8, ptr addrspace(3) %address.local325, i64 192, !dbg !10
  %289 = load <4 x half>, ptr addrspace(3) %address.local326, align 2, !dbg !10
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %236, <4 x half> %288, <4 x float> %286) #9, !dbg !10
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %289, <4 x float> %245) #9, !dbg !10
  %292 = shufflevector <4 x float> %290, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %address.local327 = getelementptr [2 x i8], ptr addrspace(3) %256, i64 %address.local320, !dbg !10
  %address.local328 = getelementptr [2 x i8], ptr addrspace(3) %address.local327, i64 %address.local253, !dbg !10
  %address.local329 = getelementptr [2 x i8], ptr addrspace(3) %address.local328, i64 %address.local257, !dbg !10
  %address.local330 = getelementptr [2 x i8], ptr addrspace(3) %address.local329, i64 %address.local247, !dbg !10
  %address.local331 = getelementptr [2 x i8], ptr addrspace(3) %address.local330, i64 %address.local250, !dbg !10
  %address.local332 = getelementptr i8, ptr addrspace(3) %address.local331, i64 4288, !dbg !10
  %293 = load <4 x half>, ptr addrspace(3) %address.local332, align 2, !dbg !10
  %294 = load <4 x half>, ptr addrspace(3) %address.local228, align 2, !dbg !10
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %236, <4 x half> %293, <4 x float> %291) #9, !dbg !10
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %294, <4 x float> %247) #9, !dbg !10
  %297 = shufflevector <4 x float> %295, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %298 = load <4 x half>, ptr addrspace(3) %address.local242, align 2, !dbg !10
  %299 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !10
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %258, <4 x half> %298, <4 x float> %296) #9, !dbg !10
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %299, <4 x float> %248) #9, !dbg !10
  %302 = load <4 x half>, ptr addrspace(3) %address.local265, align 2, !dbg !10
  %303 = load <4 x half>, ptr addrspace(3) %address.local271, align 2, !dbg !10
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %258, <4 x half> %302, <4 x float> %301) #9, !dbg !10
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %303, <4 x float> %249) #9, !dbg !10
  %306 = load <4 x half>, ptr addrspace(3) %address.local277, align 2, !dbg !10
  %307 = load <4 x half>, ptr addrspace(3) %address.local283, align 2, !dbg !10
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %258, <4 x half> %306, <4 x float> %305) #9, !dbg !10
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %307, <4 x float> %250) #9, !dbg !10
  %310 = shufflevector <4 x float> %308, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %311 = load <4 x half>, ptr addrspace(3) %address.local289, align 2, !dbg !10
  %312 = load <4 x half>, ptr addrspace(3) %address.local291, align 2, !dbg !10
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %258, <4 x half> %311, <4 x float> %309) #9, !dbg !10
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %312, <4 x float> %251) #9, !dbg !10
  %315 = shufflevector <4 x float> %313, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %316 = load <4 x half>, ptr addrspace(3) %address.local293, align 2, !dbg !10
  %317 = load <4 x half>, ptr addrspace(3) %address.local300, align 2, !dbg !10
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %258, <4 x half> %316, <4 x float> %314) #9, !dbg !10
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %317, <4 x float> %252) #9, !dbg !10
  %320 = shufflevector <4 x float> %318, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %321 = load <4 x half>, ptr addrspace(3) %address.local306, align 2, !dbg !10
  %322 = load <4 x half>, ptr addrspace(3) %address.local313, align 2, !dbg !10
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %258, <4 x half> %321, <4 x float> %319) #9, !dbg !10
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %322, <4 x float> %253) #9, !dbg !10
  %325 = shufflevector <4 x float> %323, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %326 = load <4 x half>, ptr addrspace(3) %address.local319, align 2, !dbg !10
  %327 = load <4 x half>, ptr addrspace(3) %address.local326, align 2, !dbg !10
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %258, <4 x half> %326, <4 x float> %324) #9, !dbg !10
  %329 = load <4 x half>, ptr addrspace(3) %address.local332, align 2, !dbg !10
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !10
  %330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %327, <4 x float> %254) #9, !dbg !10
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !32
  %331 = shufflevector <4 x float> %328, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %258, <4 x half> %329, <4 x float> %330) #9, !dbg !10
  %333 = shufflevector <4 x float> %332, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %334 = shufflevector <4 x float> %262, <4 x float> %266, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %335 = shufflevector <32 x float> %334, <32 x float> %272, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %336 = shufflevector <32 x float> %335, <32 x float> %277, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %337 = shufflevector <32 x float> %336, <32 x float> %282, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %338 = shufflevector <32 x float> %337, <32 x float> %287, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %339 = shufflevector <32 x float> %338, <32 x float> %292, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %340 = shufflevector <32 x float> %339, <32 x float> %297, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !23
  %341 = insertvalue [2 x <32 x float>] poison, <32 x float> %340, 0, !dbg !23
  %342 = shufflevector <4 x float> %300, <4 x float> %304, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %343 = shufflevector <32 x float> %342, <32 x float> %310, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %344 = shufflevector <32 x float> %343, <32 x float> %315, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %345 = shufflevector <32 x float> %344, <32 x float> %320, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %346 = shufflevector <32 x float> %345, <32 x float> %325, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %347 = shufflevector <32 x float> %346, <32 x float> %331, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %348 = shufflevector <32 x float> %347, <32 x float> %333, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !23
  %349 = insertvalue [2 x <32 x float>] %341, <32 x float> %348, 1, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %350 = add nuw nsw i64 %107, 1, !dbg !23
  br label %106, !dbg !23

351:                                              ; preds = %106
  %352 = add nsw i64 %68, %10, !dbg !19
  %353 = shl i64 %352, 19, !dbg !19
  %354 = and i64 %353, 16252928, !dbg !19
  %355 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %354, !dbg !19
  %356 = getelementptr [2 x i8], ptr addrspace(1) %355, i64 %74, !dbg !19
  %address.thread333 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !19, !range !35, !frisk.register_lifetimes !1
  %address.local334 = zext nneg i32 %address.thread333 to i64, !dbg !14
  %address.local335 = shl nuw nsw i64 %address.local334, 3, !dbg !13
  %address.local336 = and i64 %address.local335, 1536, !dbg !13
  %address.local337 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local336, !dbg !19
  %address.local338 = ptrtoint ptr addrspace(3) %address.local337 to i64, !dbg !19
  %address.local339 = trunc i64 %address.local338 to i32, !dbg !19
  %357 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local339), !dbg !19
  %358 = zext i32 %357 to i64, !dbg !19
  %359 = inttoptr i64 %358 to ptr addrspace(3), !dbg !19
  %address.local340 = shl nuw nsw i64 %address.local334, 5, !dbg !19
  %address.local341 = and i64 %address.local340, 8064, !dbg !19
  %address.local342 = shl nuw nsw i64 %address.local334, 3, !dbg !19
  %address.local343 = and i64 %address.local342, 24, !dbg !19
  %address.local344 = or disjoint i64 %address.local341, %address.local343, !dbg !19
  %address.local345 = getelementptr [2 x i8], ptr addrspace(1) %356, i64 %address.local344, !dbg !19
  %360 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %361 = ptrtoint ptr addrspace(1) %address.local345 to i64, !dbg !19
  %362 = sub i64 %361, %360, !dbg !19
  %dma.buffer.offset937 = trunc i64 %362 to i32, !dbg !19
  %363 = trunc i64 %360 to i32, !dbg !19
  %364 = insertelement <4 x i32> poison, i32 %363, i64 0, !dbg !19
  %365 = lshr i64 %360, 32, !dbg !19
  %366 = trunc i64 %365 to i32, !dbg !19
  %367 = insertelement <4 x i32> %364, i32 %366, i64 1, !dbg !19
  %368 = insertelement <4 x i32> %367, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc938 = insertelement <4 x i32> %368, i32 131072, i64 3, !dbg !19
  %369 = ptrtoint ptr addrspace(3) %359 to i32, !dbg !19
  %dma.buffer.lds939 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %369), !dbg !19
  %370 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds939, i32 %dma.buffer.offset937, <4 x i32> %dma.buffer.rsrc938) #10, !dbg !19
  %address.local346 = or disjoint i64 %address.local334, 256, !dbg !13
  %address.local347 = shl nuw nsw i64 %address.local346, 3, !dbg !13
  %address.local348 = and i64 %address.local347, 3584, !dbg !13
  %address.local349 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local348, !dbg !19
  %address.local350 = ptrtoint ptr addrspace(3) %address.local349 to i64, !dbg !19
  %address.local351 = trunc i64 %address.local350 to i32, !dbg !19
  %371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local351), !dbg !19
  %372 = zext i32 %371 to i64, !dbg !19
  %373 = inttoptr i64 %372 to ptr addrspace(3), !dbg !19
  %address.local352 = shl nuw nsw i64 %address.local346, 5, !dbg !19
  %address.local353 = and i64 %address.local352, 16256, !dbg !19
  %address.local354 = shl nuw nsw i64 %address.local334, 3, !dbg !19
  %address.local355 = and i64 %address.local354, 24, !dbg !19
  %address.local356 = or disjoint i64 %address.local353, %address.local355, !dbg !19
  %address.local357 = getelementptr [2 x i8], ptr addrspace(1) %356, i64 %address.local356, !dbg !19
  %374 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %375 = ptrtoint ptr addrspace(1) %address.local357 to i64, !dbg !19
  %376 = sub i64 %375, %374, !dbg !19
  %dma.buffer.offset934 = trunc i64 %376 to i32, !dbg !19
  %377 = trunc i64 %374 to i32, !dbg !19
  %378 = insertelement <4 x i32> poison, i32 %377, i64 0, !dbg !19
  %379 = lshr i64 %374, 32, !dbg !19
  %380 = trunc i64 %379 to i32, !dbg !19
  %381 = insertelement <4 x i32> %378, i32 %380, i64 1, !dbg !19
  %382 = insertelement <4 x i32> %381, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc935 = insertelement <4 x i32> %382, i32 131072, i64 3, !dbg !19
  %383 = ptrtoint ptr addrspace(3) %373 to i32, !dbg !19
  %dma.buffer.lds936 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %383), !dbg !19
  %384 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds936, i32 %dma.buffer.offset934, <4 x i32> %dma.buffer.rsrc935) #10, !dbg !19
  %385 = or disjoint i64 %37, 127, !dbg !15
  %386 = icmp uge i64 %11, %385, !dbg !15
  %387 = and i1 %26, %386, !dbg !15
  br i1 %387, label %388, label %389, !dbg !15

388:                                              ; preds = %351
  br label %533, !dbg !15

389:                                              ; preds = %351
  %390 = add nsw i64 %11, %22, !dbg !15
  %391 = shl nuw nsw i64 %23, 4, !dbg !15
  %392 = add nsw i64 %390, %391, !dbg !15
  %393 = and i32 %21, 240, !dbg !15
  %index.compact = zext nneg i32 %393 to i64, !dbg !15
  %394 = sub nsw i64 %392, %index.compact, !dbg !15
  %395 = and i64 %22, 48, !dbg !15
  %396 = or disjoint i64 %37, %395, !dbg !15
  %397 = add nsw i64 %394, 1, !dbg !24
  %398 = lshr i64 %397, 32, !dbg !25
  %compare.high = trunc i64 %398 to i32, !dbg !25
  %399 = lshr i64 %396, 32, !dbg !25
  %compare.high651 = trunc i64 %399 to i32, !dbg !25
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high651, !dbg !25
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high651, !dbg !25
  %compare.low.base = trunc i64 %397 to i32, !dbg !25
  %compare.low.base652 = trunc i64 %396 to i32, !dbg !25
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base652, !dbg !25
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !25
  %400 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 1) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !25
  %compare.low653 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !25
  %compare.result654 = select i1 %compare.high.equal, i1 %compare.low653, i1 %compare.high.order, !dbg !25
  %401 = select i1 %compare.result654, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base655 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 2) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local656 = or i32 %compare.local.base655, 2, !dbg !25
  %compare.low657 = icmp ugt i32 %compare.low.base, %compare.local656, !dbg !25
  %compare.result658 = select i1 %compare.high.equal, i1 %compare.low657, i1 %compare.high.order, !dbg !25
  %402 = select i1 %compare.result658, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base659 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 3) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local660 = or i32 %compare.local.base659, 3, !dbg !25
  %compare.low661 = icmp ugt i32 %compare.low.base, %compare.local660, !dbg !25
  %compare.result662 = select i1 %compare.high.equal, i1 %compare.low661, i1 %compare.high.order, !dbg !25
  %403 = select i1 %compare.result662, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base663 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 4) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local664 = or i32 %compare.local.base663, 4, !dbg !25
  %compare.low665 = icmp ugt i32 %compare.low.base, %compare.local664, !dbg !25
  %compare.result666 = select i1 %compare.high.equal, i1 %compare.low665, i1 %compare.high.order, !dbg !25
  %404 = select i1 %compare.result666, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base667 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 5) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local668 = or i32 %compare.local.base667, 5, !dbg !25
  %compare.low669 = icmp ugt i32 %compare.low.base, %compare.local668, !dbg !25
  %compare.result670 = select i1 %compare.high.equal, i1 %compare.low669, i1 %compare.high.order, !dbg !25
  %405 = select i1 %compare.result670, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base671 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 6) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local672 = or i32 %compare.local.base671, 6, !dbg !25
  %compare.low673 = icmp ugt i32 %compare.low.base, %compare.local672, !dbg !25
  %compare.result674 = select i1 %compare.high.equal, i1 %compare.low673, i1 %compare.high.order, !dbg !25
  %406 = select i1 %compare.result674, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base675 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 7) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local676 = or i32 %compare.local.base675, 7, !dbg !25
  %compare.low677 = icmp ugt i32 %compare.low.base, %compare.local676, !dbg !25
  %compare.result678 = select i1 %compare.high.equal, i1 %compare.low677, i1 %compare.high.order, !dbg !25
  %407 = select i1 %compare.result678, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base679 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 8) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local680 = or i32 %compare.local.base679, 8, !dbg !25
  %compare.low681 = icmp ugt i32 %compare.low.base, %compare.local680, !dbg !25
  %compare.result682 = select i1 %compare.high.equal, i1 %compare.low681, i1 %compare.high.order, !dbg !25
  %408 = select i1 %compare.result682, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base683 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 9) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local684 = or i32 %compare.local.base683, 9, !dbg !25
  %compare.low685 = icmp ugt i32 %compare.low.base, %compare.local684, !dbg !25
  %compare.result686 = select i1 %compare.high.equal, i1 %compare.low685, i1 %compare.high.order, !dbg !25
  %409 = select i1 %compare.result686, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base687 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 10) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local688 = or i32 %compare.local.base687, 10, !dbg !25
  %compare.low689 = icmp ugt i32 %compare.low.base, %compare.local688, !dbg !25
  %compare.result690 = select i1 %compare.high.equal, i1 %compare.low689, i1 %compare.high.order, !dbg !25
  %410 = select i1 %compare.result690, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base691 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 11) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local692 = or i32 %compare.local.base691, 11, !dbg !25
  %compare.low693 = icmp ugt i32 %compare.low.base, %compare.local692, !dbg !25
  %compare.result694 = select i1 %compare.high.equal, i1 %compare.low693, i1 %compare.high.order, !dbg !25
  %411 = select i1 %compare.result694, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base695 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 12) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local696 = or i32 %compare.local.base695, 12, !dbg !25
  %compare.low697 = icmp ugt i32 %compare.low.base, %compare.local696, !dbg !25
  %compare.result698 = select i1 %compare.high.equal, i1 %compare.low697, i1 %compare.high.order, !dbg !25
  %412 = select i1 %compare.result698, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base699 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 13) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local700 = or i32 %compare.local.base699, 13, !dbg !25
  %compare.low701 = icmp ugt i32 %compare.low.base, %compare.local700, !dbg !25
  %compare.result702 = select i1 %compare.high.equal, i1 %compare.low701, i1 %compare.high.order, !dbg !25
  %413 = select i1 %compare.result702, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base703 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 14) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local704 = or i32 %compare.local.base703, 14, !dbg !25
  %compare.low705 = icmp ugt i32 %compare.low.base, %compare.local704, !dbg !25
  %compare.result706 = select i1 %compare.high.equal, i1 %compare.low705, i1 %compare.high.order, !dbg !25
  %414 = select i1 %compare.result706, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base707 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 15) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local708 = or i32 %compare.local.base707, 15, !dbg !25
  %compare.low709 = icmp ugt i32 %compare.low.base, %compare.local708, !dbg !25
  %compare.result710 = select i1 %compare.high.equal, i1 %compare.low709, i1 %compare.high.order, !dbg !25
  %415 = select i1 %compare.result710, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base711 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 16) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local712 = or i32 %compare.local.base711, 64, !dbg !25
  %compare.low713 = icmp ugt i32 %compare.low.base, %compare.local712, !dbg !25
  %compare.result714 = select i1 %compare.high.equal, i1 %compare.low713, i1 %compare.high.order, !dbg !25
  %416 = select i1 %compare.result714, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base715 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 17) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local716 = or i32 %compare.local.base715, 65, !dbg !25
  %compare.low717 = icmp ugt i32 %compare.low.base, %compare.local716, !dbg !25
  %compare.result718 = select i1 %compare.high.equal, i1 %compare.low717, i1 %compare.high.order, !dbg !25
  %417 = select i1 %compare.result718, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base719 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 18) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local720 = or i32 %compare.local.base719, 66, !dbg !25
  %compare.low721 = icmp ugt i32 %compare.low.base, %compare.local720, !dbg !25
  %compare.result722 = select i1 %compare.high.equal, i1 %compare.low721, i1 %compare.high.order, !dbg !25
  %418 = select i1 %compare.result722, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base723 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 19) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local724 = or i32 %compare.local.base723, 67, !dbg !25
  %compare.low725 = icmp ugt i32 %compare.low.base, %compare.local724, !dbg !25
  %compare.result726 = select i1 %compare.high.equal, i1 %compare.low725, i1 %compare.high.order, !dbg !25
  %419 = select i1 %compare.result726, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base727 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 20) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local728 = or i32 %compare.local.base727, 68, !dbg !25
  %compare.low729 = icmp ugt i32 %compare.low.base, %compare.local728, !dbg !25
  %compare.result730 = select i1 %compare.high.equal, i1 %compare.low729, i1 %compare.high.order, !dbg !25
  %420 = select i1 %compare.result730, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base731 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 21) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local732 = or i32 %compare.local.base731, 69, !dbg !25
  %compare.low733 = icmp ugt i32 %compare.low.base, %compare.local732, !dbg !25
  %compare.result734 = select i1 %compare.high.equal, i1 %compare.low733, i1 %compare.high.order, !dbg !25
  %421 = select i1 %compare.result734, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base735 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 22) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local736 = or i32 %compare.local.base735, 70, !dbg !25
  %compare.low737 = icmp ugt i32 %compare.low.base, %compare.local736, !dbg !25
  %compare.result738 = select i1 %compare.high.equal, i1 %compare.low737, i1 %compare.high.order, !dbg !25
  %422 = select i1 %compare.result738, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base739 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 23) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local740 = or i32 %compare.local.base739, 71, !dbg !25
  %compare.low741 = icmp ugt i32 %compare.low.base, %compare.local740, !dbg !25
  %compare.result742 = select i1 %compare.high.equal, i1 %compare.low741, i1 %compare.high.order, !dbg !25
  %423 = select i1 %compare.result742, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base743 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 24) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local744 = or i32 %compare.local.base743, 72, !dbg !25
  %compare.low745 = icmp ugt i32 %compare.low.base, %compare.local744, !dbg !25
  %compare.result746 = select i1 %compare.high.equal, i1 %compare.low745, i1 %compare.high.order, !dbg !25
  %424 = select i1 %compare.result746, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base747 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 25) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local748 = or i32 %compare.local.base747, 73, !dbg !25
  %compare.low749 = icmp ugt i32 %compare.low.base, %compare.local748, !dbg !25
  %compare.result750 = select i1 %compare.high.equal, i1 %compare.low749, i1 %compare.high.order, !dbg !25
  %425 = select i1 %compare.result750, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base751 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 26) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local752 = or i32 %compare.local.base751, 74, !dbg !25
  %compare.low753 = icmp ugt i32 %compare.low.base, %compare.local752, !dbg !25
  %compare.result754 = select i1 %compare.high.equal, i1 %compare.low753, i1 %compare.high.order, !dbg !25
  %426 = select i1 %compare.result754, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base755 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 27) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local756 = or i32 %compare.local.base755, 75, !dbg !25
  %compare.low757 = icmp ugt i32 %compare.low.base, %compare.local756, !dbg !25
  %compare.result758 = select i1 %compare.high.equal, i1 %compare.low757, i1 %compare.high.order, !dbg !25
  %427 = select i1 %compare.result758, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base759 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 28) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local760 = or i32 %compare.local.base759, 76, !dbg !25
  %compare.low761 = icmp ugt i32 %compare.low.base, %compare.local760, !dbg !25
  %compare.result762 = select i1 %compare.high.equal, i1 %compare.low761, i1 %compare.high.order, !dbg !25
  %428 = select i1 %compare.result762, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base763 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 29) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local764 = or i32 %compare.local.base763, 77, !dbg !25
  %compare.low765 = icmp ugt i32 %compare.low.base, %compare.local764, !dbg !25
  %compare.result766 = select i1 %compare.high.equal, i1 %compare.low765, i1 %compare.high.order, !dbg !25
  %429 = select i1 %compare.result766, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base767 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 30) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local768 = or i32 %compare.local.base767, 78, !dbg !25
  %compare.low769 = icmp ugt i32 %compare.low.base, %compare.local768, !dbg !25
  %compare.result770 = select i1 %compare.high.equal, i1 %compare.low769, i1 %compare.high.order, !dbg !25
  %430 = select i1 %compare.result770, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base771 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 31) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local772 = or i32 %compare.local.base771, 79, !dbg !25
  %compare.low773 = icmp ugt i32 %compare.low.base, %compare.local772, !dbg !25
  %compare.result774 = select i1 %compare.high.equal, i1 %compare.low773, i1 %compare.high.order, !dbg !25
  %431 = select i1 %compare.result774, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %432 = add nsw i64 %394, 65, !dbg !24
  %433 = lshr i64 %432, 32, !dbg !25
  %compare.high775 = trunc i64 %433 to i32, !dbg !25
  %434 = lshr i64 %396, 32, !dbg !25
  %compare.high776 = trunc i64 %434 to i32, !dbg !25
  %compare.high.equal777 = icmp eq i32 %compare.high775, %compare.high776, !dbg !25
  %compare.high.order778 = icmp ugt i32 %compare.high775, %compare.high776, !dbg !25
  %compare.low.base779 = trunc i64 %432 to i32, !dbg !25
  %compare.low.base780 = trunc i64 %396 to i32, !dbg !25
  %compare.low781 = icmp ugt i32 %compare.low.base779, %compare.low.base780, !dbg !25
  %compare.result782 = select i1 %compare.high.equal777, i1 %compare.low781, i1 %compare.high.order778, !dbg !25
  %435 = select i1 %compare.result782, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base783 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 32) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local784 = or i32 %compare.local.base783, 1, !dbg !25
  %compare.low785 = icmp ugt i32 %compare.low.base779, %compare.local784, !dbg !25
  %compare.result786 = select i1 %compare.high.equal777, i1 %compare.low785, i1 %compare.high.order778, !dbg !25
  %436 = select i1 %compare.result786, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base787 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 33) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local788 = or i32 %compare.local.base787, 2, !dbg !25
  %compare.low789 = icmp ugt i32 %compare.low.base779, %compare.local788, !dbg !25
  %compare.result790 = select i1 %compare.high.equal777, i1 %compare.low789, i1 %compare.high.order778, !dbg !25
  %437 = select i1 %compare.result790, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base791 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 34) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local792 = or i32 %compare.local.base791, 3, !dbg !25
  %compare.low793 = icmp ugt i32 %compare.low.base779, %compare.local792, !dbg !25
  %compare.result794 = select i1 %compare.high.equal777, i1 %compare.low793, i1 %compare.high.order778, !dbg !25
  %438 = select i1 %compare.result794, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base795 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 35) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local796 = or i32 %compare.local.base795, 4, !dbg !25
  %compare.low797 = icmp ugt i32 %compare.low.base779, %compare.local796, !dbg !25
  %compare.result798 = select i1 %compare.high.equal777, i1 %compare.low797, i1 %compare.high.order778, !dbg !25
  %439 = select i1 %compare.result798, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base799 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 36) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local800 = or i32 %compare.local.base799, 5, !dbg !25
  %compare.low801 = icmp ugt i32 %compare.low.base779, %compare.local800, !dbg !25
  %compare.result802 = select i1 %compare.high.equal777, i1 %compare.low801, i1 %compare.high.order778, !dbg !25
  %440 = select i1 %compare.result802, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base803 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 37) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local804 = or i32 %compare.local.base803, 6, !dbg !25
  %compare.low805 = icmp ugt i32 %compare.low.base779, %compare.local804, !dbg !25
  %compare.result806 = select i1 %compare.high.equal777, i1 %compare.low805, i1 %compare.high.order778, !dbg !25
  %441 = select i1 %compare.result806, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base807 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 38) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local808 = or i32 %compare.local.base807, 7, !dbg !25
  %compare.low809 = icmp ugt i32 %compare.low.base779, %compare.local808, !dbg !25
  %compare.result810 = select i1 %compare.high.equal777, i1 %compare.low809, i1 %compare.high.order778, !dbg !25
  %442 = select i1 %compare.result810, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base811 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 39) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local812 = or i32 %compare.local.base811, 8, !dbg !25
  %compare.low813 = icmp ugt i32 %compare.low.base779, %compare.local812, !dbg !25
  %compare.result814 = select i1 %compare.high.equal777, i1 %compare.low813, i1 %compare.high.order778, !dbg !25
  %443 = select i1 %compare.result814, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base815 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 40) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local816 = or i32 %compare.local.base815, 9, !dbg !25
  %compare.low817 = icmp ugt i32 %compare.low.base779, %compare.local816, !dbg !25
  %compare.result818 = select i1 %compare.high.equal777, i1 %compare.low817, i1 %compare.high.order778, !dbg !25
  %444 = select i1 %compare.result818, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base819 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 41) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local820 = or i32 %compare.local.base819, 10, !dbg !25
  %compare.low821 = icmp ugt i32 %compare.low.base779, %compare.local820, !dbg !25
  %compare.result822 = select i1 %compare.high.equal777, i1 %compare.low821, i1 %compare.high.order778, !dbg !25
  %445 = select i1 %compare.result822, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base823 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 42) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local824 = or i32 %compare.local.base823, 11, !dbg !25
  %compare.low825 = icmp ugt i32 %compare.low.base779, %compare.local824, !dbg !25
  %compare.result826 = select i1 %compare.high.equal777, i1 %compare.low825, i1 %compare.high.order778, !dbg !25
  %446 = select i1 %compare.result826, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base827 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 43) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local828 = or i32 %compare.local.base827, 12, !dbg !25
  %compare.low829 = icmp ugt i32 %compare.low.base779, %compare.local828, !dbg !25
  %compare.result830 = select i1 %compare.high.equal777, i1 %compare.low829, i1 %compare.high.order778, !dbg !25
  %447 = select i1 %compare.result830, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base831 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 44) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local832 = or i32 %compare.local.base831, 13, !dbg !25
  %compare.low833 = icmp ugt i32 %compare.low.base779, %compare.local832, !dbg !25
  %compare.result834 = select i1 %compare.high.equal777, i1 %compare.low833, i1 %compare.high.order778, !dbg !25
  %448 = select i1 %compare.result834, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base835 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 45) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local836 = or i32 %compare.local.base835, 14, !dbg !25
  %compare.low837 = icmp ugt i32 %compare.low.base779, %compare.local836, !dbg !25
  %compare.result838 = select i1 %compare.high.equal777, i1 %compare.low837, i1 %compare.high.order778, !dbg !25
  %449 = select i1 %compare.result838, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base839 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 46) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local840 = or i32 %compare.local.base839, 15, !dbg !25
  %compare.low841 = icmp ugt i32 %compare.low.base779, %compare.local840, !dbg !25
  %compare.result842 = select i1 %compare.high.equal777, i1 %compare.low841, i1 %compare.high.order778, !dbg !25
  %450 = select i1 %compare.result842, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base843 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 47) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local844 = or i32 %compare.local.base843, 64, !dbg !25
  %compare.low845 = icmp ugt i32 %compare.low.base779, %compare.local844, !dbg !25
  %compare.result846 = select i1 %compare.high.equal777, i1 %compare.low845, i1 %compare.high.order778, !dbg !25
  %451 = select i1 %compare.result846, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base847 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 48) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local848 = or i32 %compare.local.base847, 65, !dbg !25
  %compare.low849 = icmp ugt i32 %compare.low.base779, %compare.local848, !dbg !25
  %compare.result850 = select i1 %compare.high.equal777, i1 %compare.low849, i1 %compare.high.order778, !dbg !25
  %452 = select i1 %compare.result850, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base851 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 49) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local852 = or i32 %compare.local.base851, 66, !dbg !25
  %compare.low853 = icmp ugt i32 %compare.low.base779, %compare.local852, !dbg !25
  %compare.result854 = select i1 %compare.high.equal777, i1 %compare.low853, i1 %compare.high.order778, !dbg !25
  %453 = select i1 %compare.result854, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base855 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 50) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local856 = or i32 %compare.local.base855, 67, !dbg !25
  %compare.low857 = icmp ugt i32 %compare.low.base779, %compare.local856, !dbg !25
  %compare.result858 = select i1 %compare.high.equal777, i1 %compare.low857, i1 %compare.high.order778, !dbg !25
  %454 = select i1 %compare.result858, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base859 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 51) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local860 = or i32 %compare.local.base859, 68, !dbg !25
  %compare.low861 = icmp ugt i32 %compare.low.base779, %compare.local860, !dbg !25
  %compare.result862 = select i1 %compare.high.equal777, i1 %compare.low861, i1 %compare.high.order778, !dbg !25
  %455 = select i1 %compare.result862, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base863 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 52) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local864 = or i32 %compare.local.base863, 69, !dbg !25
  %compare.low865 = icmp ugt i32 %compare.low.base779, %compare.local864, !dbg !25
  %compare.result866 = select i1 %compare.high.equal777, i1 %compare.low865, i1 %compare.high.order778, !dbg !25
  %456 = select i1 %compare.result866, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base867 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 53) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local868 = or i32 %compare.local.base867, 70, !dbg !25
  %compare.low869 = icmp ugt i32 %compare.low.base779, %compare.local868, !dbg !25
  %compare.result870 = select i1 %compare.high.equal777, i1 %compare.low869, i1 %compare.high.order778, !dbg !25
  %457 = select i1 %compare.result870, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base871 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 54) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local872 = or i32 %compare.local.base871, 71, !dbg !25
  %compare.low873 = icmp ugt i32 %compare.low.base779, %compare.local872, !dbg !25
  %compare.result874 = select i1 %compare.high.equal777, i1 %compare.low873, i1 %compare.high.order778, !dbg !25
  %458 = select i1 %compare.result874, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base875 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 55) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local876 = or i32 %compare.local.base875, 72, !dbg !25
  %compare.low877 = icmp ugt i32 %compare.low.base779, %compare.local876, !dbg !25
  %compare.result878 = select i1 %compare.high.equal777, i1 %compare.low877, i1 %compare.high.order778, !dbg !25
  %459 = select i1 %compare.result878, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base879 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 56) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local880 = or i32 %compare.local.base879, 73, !dbg !25
  %compare.low881 = icmp ugt i32 %compare.low.base779, %compare.local880, !dbg !25
  %compare.result882 = select i1 %compare.high.equal777, i1 %compare.low881, i1 %compare.high.order778, !dbg !25
  %460 = select i1 %compare.result882, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base883 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 57) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local884 = or i32 %compare.local.base883, 74, !dbg !25
  %compare.low885 = icmp ugt i32 %compare.low.base779, %compare.local884, !dbg !25
  %compare.result886 = select i1 %compare.high.equal777, i1 %compare.low885, i1 %compare.high.order778, !dbg !25
  %461 = select i1 %compare.result886, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base887 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 58) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local888 = or i32 %compare.local.base887, 75, !dbg !25
  %compare.low889 = icmp ugt i32 %compare.low.base779, %compare.local888, !dbg !25
  %compare.result890 = select i1 %compare.high.equal777, i1 %compare.low889, i1 %compare.high.order778, !dbg !25
  %462 = select i1 %compare.result890, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base891 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 59) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local892 = or i32 %compare.local.base891, 76, !dbg !25
  %compare.low893 = icmp ugt i32 %compare.low.base779, %compare.local892, !dbg !25
  %compare.result894 = select i1 %compare.high.equal777, i1 %compare.low893, i1 %compare.high.order778, !dbg !25
  %463 = select i1 %compare.result894, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base895 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 60) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local896 = or i32 %compare.local.base895, 77, !dbg !25
  %compare.low897 = icmp ugt i32 %compare.low.base779, %compare.local896, !dbg !25
  %compare.result898 = select i1 %compare.high.equal777, i1 %compare.low897, i1 %compare.high.order778, !dbg !25
  %464 = select i1 %compare.result898, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base899 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 61) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local900 = or i32 %compare.local.base899, 78, !dbg !25
  %compare.low901 = icmp ugt i32 %compare.low.base779, %compare.local900, !dbg !25
  %compare.result902 = select i1 %compare.high.equal777, i1 %compare.low901, i1 %compare.high.order778, !dbg !25
  %465 = select i1 %compare.result902, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base903 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base780, i32 62) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local904 = or i32 %compare.local.base903, 79, !dbg !25
  %compare.low905 = icmp ugt i32 %compare.low.base779, %compare.local904, !dbg !25
  %compare.result906 = select i1 %compare.high.equal777, i1 %compare.low905, i1 %compare.high.order778, !dbg !25
  %466 = select i1 %compare.result906, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %467 = insertelement <32 x float> poison, float %400, i64 0, !dbg !15
  %468 = insertelement <32 x float> %467, float %401, i64 1, !dbg !15
  %469 = insertelement <32 x float> %468, float %402, i64 2, !dbg !15
  %470 = insertelement <32 x float> %469, float %403, i64 3, !dbg !15
  %471 = insertelement <32 x float> %470, float %404, i64 4, !dbg !15
  %472 = insertelement <32 x float> %471, float %405, i64 5, !dbg !15
  %473 = insertelement <32 x float> %472, float %406, i64 6, !dbg !15
  %474 = insertelement <32 x float> %473, float %407, i64 7, !dbg !15
  %475 = insertelement <32 x float> %474, float %408, i64 8, !dbg !15
  %476 = insertelement <32 x float> %475, float %409, i64 9, !dbg !15
  %477 = insertelement <32 x float> %476, float %410, i64 10, !dbg !15
  %478 = insertelement <32 x float> %477, float %411, i64 11, !dbg !15
  %479 = insertelement <32 x float> %478, float %412, i64 12, !dbg !15
  %480 = insertelement <32 x float> %479, float %413, i64 13, !dbg !15
  %481 = insertelement <32 x float> %480, float %414, i64 14, !dbg !15
  %482 = insertelement <32 x float> %481, float %415, i64 15, !dbg !15
  %483 = insertelement <32 x float> %482, float %416, i64 16, !dbg !15
  %484 = insertelement <32 x float> %483, float %417, i64 17, !dbg !15
  %485 = insertelement <32 x float> %484, float %418, i64 18, !dbg !15
  %486 = insertelement <32 x float> %485, float %419, i64 19, !dbg !15
  %487 = insertelement <32 x float> %486, float %420, i64 20, !dbg !15
  %488 = insertelement <32 x float> %487, float %421, i64 21, !dbg !15
  %489 = insertelement <32 x float> %488, float %422, i64 22, !dbg !15
  %490 = insertelement <32 x float> %489, float %423, i64 23, !dbg !15
  %491 = insertelement <32 x float> %490, float %424, i64 24, !dbg !15
  %492 = insertelement <32 x float> %491, float %425, i64 25, !dbg !15
  %493 = insertelement <32 x float> %492, float %426, i64 26, !dbg !15
  %494 = insertelement <32 x float> %493, float %427, i64 27, !dbg !15
  %495 = insertelement <32 x float> %494, float %428, i64 28, !dbg !15
  %496 = insertelement <32 x float> %495, float %429, i64 29, !dbg !15
  %497 = insertelement <32 x float> %496, float %430, i64 30, !dbg !15
  %498 = insertelement <32 x float> %497, float %431, i64 31, !dbg !15
  %499 = insertvalue [2 x <32 x float>] poison, <32 x float> %498, 0, !dbg !15
  %500 = insertelement <32 x float> poison, float %435, i64 0, !dbg !15
  %501 = insertelement <32 x float> %500, float %436, i64 1, !dbg !15
  %502 = insertelement <32 x float> %501, float %437, i64 2, !dbg !15
  %503 = insertelement <32 x float> %502, float %438, i64 3, !dbg !15
  %504 = insertelement <32 x float> %503, float %439, i64 4, !dbg !15
  %505 = insertelement <32 x float> %504, float %440, i64 5, !dbg !15
  %506 = insertelement <32 x float> %505, float %441, i64 6, !dbg !15
  %507 = insertelement <32 x float> %506, float %442, i64 7, !dbg !15
  %508 = insertelement <32 x float> %507, float %443, i64 8, !dbg !15
  %509 = insertelement <32 x float> %508, float %444, i64 9, !dbg !15
  %510 = insertelement <32 x float> %509, float %445, i64 10, !dbg !15
  %511 = insertelement <32 x float> %510, float %446, i64 11, !dbg !15
  %512 = insertelement <32 x float> %511, float %447, i64 12, !dbg !15
  %513 = insertelement <32 x float> %512, float %448, i64 13, !dbg !15
  %514 = insertelement <32 x float> %513, float %449, i64 14, !dbg !15
  %515 = insertelement <32 x float> %514, float %450, i64 15, !dbg !15
  %516 = insertelement <32 x float> %515, float %451, i64 16, !dbg !15
  %517 = insertelement <32 x float> %516, float %452, i64 17, !dbg !15
  %518 = insertelement <32 x float> %517, float %453, i64 18, !dbg !15
  %519 = insertelement <32 x float> %518, float %454, i64 19, !dbg !15
  %520 = insertelement <32 x float> %519, float %455, i64 20, !dbg !15
  %521 = insertelement <32 x float> %520, float %456, i64 21, !dbg !15
  %522 = insertelement <32 x float> %521, float %457, i64 22, !dbg !15
  %523 = insertelement <32 x float> %522, float %458, i64 23, !dbg !15
  %524 = insertelement <32 x float> %523, float %459, i64 24, !dbg !15
  %525 = insertelement <32 x float> %524, float %460, i64 25, !dbg !15
  %526 = insertelement <32 x float> %525, float %461, i64 26, !dbg !15
  %527 = insertelement <32 x float> %526, float %462, i64 27, !dbg !15
  %528 = insertelement <32 x float> %527, float %463, i64 28, !dbg !15
  %529 = insertelement <32 x float> %528, float %464, i64 29, !dbg !15
  %530 = insertelement <32 x float> %529, float %465, i64 30, !dbg !15
  %531 = insertelement <32 x float> %530, float %466, i64 31, !dbg !15
  %532 = insertvalue [2 x <32 x float>] %499, <32 x float> %531, 1, !dbg !15
  br label %533, !dbg !15

533:                                              ; preds = %388, %389
  %534 = phi [2 x <32 x float>] [ %532, %389 ], [ zeroinitializer, %388 ], !dbg !15
  br label %535, !dbg !15

535:                                              ; preds = %533
  %536 = extractvalue [2 x <32 x float>] %108, 0, !dbg !27
  %537 = extractvalue [2 x <32 x float>] %534, 0, !dbg !27
  %538 = shufflevector <32 x float> %536, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %539 = shufflevector <32 x float> %537, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %540 = fadd <4 x float> %538, %539, !dbg !27
  %541 = extractelement <4 x float> %540, i64 0, !dbg !28
  %542 = call float @llvm.amdgcn.exp2.f32(float %541), !dbg !28
  %543 = insertelement <4 x float> poison, float %542, i64 0, !dbg !30
  %544 = extractelement <4 x float> %540, i64 1, !dbg !28
  %545 = call float @llvm.amdgcn.exp2.f32(float %544), !dbg !28
  %546 = insertelement <4 x float> %543, float %545, i64 1, !dbg !30
  %547 = fadd float %542, %545, !dbg !29
  %548 = extractelement <4 x float> %540, i64 2, !dbg !28
  %549 = call float @llvm.amdgcn.exp2.f32(float %548), !dbg !28
  %550 = insertelement <4 x float> %546, float %549, i64 2, !dbg !30
  %551 = fadd float %547, %549, !dbg !29
  %552 = extractelement <4 x float> %540, i64 3, !dbg !28
  %553 = call float @llvm.amdgcn.exp2.f32(float %552), !dbg !28
  %554 = insertelement <4 x float> %550, float %553, i64 3, !dbg !30
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %554) #9, !dbg !30, !frisk.register_lifetimes !1
  %555 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !30
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %555) #9, !dbg !29, !frisk.register_lifetimes !1
  %556 = fadd float %551, %553, !dbg !29
  %557 = shufflevector <32 x float> %536, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %558 = shufflevector <32 x float> %537, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %559 = fadd <4 x float> %557, %558, !dbg !27
  %560 = extractelement <4 x float> %559, i64 0, !dbg !28
  %561 = call float @llvm.amdgcn.exp2.f32(float %560), !dbg !28
  %562 = insertelement <4 x float> poison, float %561, i64 0, !dbg !30
  %563 = fadd float %556, %561, !dbg !29
  %564 = extractelement <4 x float> %559, i64 1, !dbg !28
  %565 = call float @llvm.amdgcn.exp2.f32(float %564), !dbg !28
  %566 = insertelement <4 x float> %562, float %565, i64 1, !dbg !30
  %567 = fadd float %563, %565, !dbg !29
  %568 = extractelement <4 x float> %559, i64 2, !dbg !28
  %569 = call float @llvm.amdgcn.exp2.f32(float %568), !dbg !28
  %570 = insertelement <4 x float> %566, float %569, i64 2, !dbg !30
  %571 = fadd float %567, %569, !dbg !29
  %572 = extractelement <4 x float> %559, i64 3, !dbg !28
  %573 = call float @llvm.amdgcn.exp2.f32(float %572), !dbg !28
  %574 = insertelement <4 x float> %570, float %573, i64 3, !dbg !30
  %math.result102 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %574) #9, !dbg !30, !frisk.register_lifetimes !1
  %575 = fptrunc <4 x float> %math.result102 to <4 x half>, !dbg !30
  %math.packed103 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %575) #9, !dbg !29, !frisk.register_lifetimes !1
  %576 = fadd float %571, %573, !dbg !29
  %577 = shufflevector <32 x float> %536, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %578 = shufflevector <32 x float> %537, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %579 = fadd <4 x float> %577, %578, !dbg !27
  %580 = extractelement <4 x float> %579, i64 0, !dbg !28
  %581 = call float @llvm.amdgcn.exp2.f32(float %580), !dbg !28
  %582 = insertelement <4 x float> poison, float %581, i64 0, !dbg !30
  %583 = fadd float %576, %581, !dbg !29
  %584 = extractelement <4 x float> %579, i64 1, !dbg !28
  %585 = call float @llvm.amdgcn.exp2.f32(float %584), !dbg !28
  %586 = insertelement <4 x float> %582, float %585, i64 1, !dbg !30
  %587 = fadd float %583, %585, !dbg !29
  %588 = extractelement <4 x float> %579, i64 2, !dbg !28
  %589 = call float @llvm.amdgcn.exp2.f32(float %588), !dbg !28
  %590 = insertelement <4 x float> %586, float %589, i64 2, !dbg !30
  %591 = fadd float %587, %589, !dbg !29
  %592 = extractelement <4 x float> %579, i64 3, !dbg !28
  %593 = call float @llvm.amdgcn.exp2.f32(float %592), !dbg !28
  %594 = insertelement <4 x float> %590, float %593, i64 3, !dbg !30
  %math.result104 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %594) #9, !dbg !30, !frisk.register_lifetimes !1
  %595 = fptrunc <4 x float> %math.result104 to <4 x half>, !dbg !30
  %math.packed105 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %595) #9, !dbg !29, !frisk.register_lifetimes !1
  %596 = fadd float %591, %593, !dbg !29
  %597 = shufflevector <32 x float> %536, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %598 = shufflevector <32 x float> %537, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %599 = fadd <4 x float> %597, %598, !dbg !27
  %600 = extractelement <4 x float> %599, i64 0, !dbg !28
  %601 = call float @llvm.amdgcn.exp2.f32(float %600), !dbg !28
  %602 = insertelement <4 x float> poison, float %601, i64 0, !dbg !30
  %603 = fadd float %596, %601, !dbg !29
  %604 = extractelement <4 x float> %599, i64 1, !dbg !28
  %605 = call float @llvm.amdgcn.exp2.f32(float %604), !dbg !28
  %606 = insertelement <4 x float> %602, float %605, i64 1, !dbg !30
  %607 = fadd float %603, %605, !dbg !29
  %608 = extractelement <4 x float> %599, i64 2, !dbg !28
  %609 = call float @llvm.amdgcn.exp2.f32(float %608), !dbg !28
  %610 = insertelement <4 x float> %606, float %609, i64 2, !dbg !30
  %611 = fadd float %607, %609, !dbg !29
  %612 = extractelement <4 x float> %599, i64 3, !dbg !28
  %613 = call float @llvm.amdgcn.exp2.f32(float %612), !dbg !28
  %614 = insertelement <4 x float> %610, float %613, i64 3, !dbg !30
  %math.result106 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %614) #9, !dbg !30, !frisk.register_lifetimes !1
  %615 = fptrunc <4 x float> %math.result106 to <4 x half>, !dbg !30
  %math.packed107 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %615) #9, !dbg !29, !frisk.register_lifetimes !1
  %616 = fadd float %611, %613, !dbg !29
  %617 = shufflevector <32 x float> %536, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %618 = shufflevector <32 x float> %537, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %619 = fadd <4 x float> %617, %618, !dbg !27
  %620 = extractelement <4 x float> %619, i64 0, !dbg !28
  %621 = call float @llvm.amdgcn.exp2.f32(float %620), !dbg !28
  %622 = insertelement <4 x float> poison, float %621, i64 0, !dbg !30
  %623 = fadd float %616, %621, !dbg !29
  %624 = extractelement <4 x float> %619, i64 1, !dbg !28
  %625 = call float @llvm.amdgcn.exp2.f32(float %624), !dbg !28
  %626 = insertelement <4 x float> %622, float %625, i64 1, !dbg !30
  %627 = fadd float %623, %625, !dbg !29
  %628 = extractelement <4 x float> %619, i64 2, !dbg !28
  %629 = call float @llvm.amdgcn.exp2.f32(float %628), !dbg !28
  %630 = insertelement <4 x float> %626, float %629, i64 2, !dbg !30
  %631 = fadd float %627, %629, !dbg !29
  %632 = extractelement <4 x float> %619, i64 3, !dbg !28
  %633 = call float @llvm.amdgcn.exp2.f32(float %632), !dbg !28
  %634 = insertelement <4 x float> %630, float %633, i64 3, !dbg !30
  %math.result108 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %634) #9, !dbg !30, !frisk.register_lifetimes !1
  %635 = fptrunc <4 x float> %math.result108 to <4 x half>, !dbg !30
  %math.packed109 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %635) #9, !dbg !29, !frisk.register_lifetimes !1
  %636 = fadd float %631, %633, !dbg !29
  %637 = shufflevector <32 x float> %536, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %638 = shufflevector <32 x float> %537, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %639 = fadd <4 x float> %637, %638, !dbg !27
  %640 = extractelement <4 x float> %639, i64 0, !dbg !28
  %641 = call float @llvm.amdgcn.exp2.f32(float %640), !dbg !28
  %642 = insertelement <4 x float> poison, float %641, i64 0, !dbg !30
  %643 = fadd float %636, %641, !dbg !29
  %644 = extractelement <4 x float> %639, i64 1, !dbg !28
  %645 = call float @llvm.amdgcn.exp2.f32(float %644), !dbg !28
  %646 = insertelement <4 x float> %642, float %645, i64 1, !dbg !30
  %647 = fadd float %643, %645, !dbg !29
  %648 = extractelement <4 x float> %639, i64 2, !dbg !28
  %649 = call float @llvm.amdgcn.exp2.f32(float %648), !dbg !28
  %650 = insertelement <4 x float> %646, float %649, i64 2, !dbg !30
  %651 = fadd float %647, %649, !dbg !29
  %652 = extractelement <4 x float> %639, i64 3, !dbg !28
  %653 = call float @llvm.amdgcn.exp2.f32(float %652), !dbg !28
  %654 = insertelement <4 x float> %650, float %653, i64 3, !dbg !30
  %math.result110 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %654) #9, !dbg !30, !frisk.register_lifetimes !1
  %655 = fptrunc <4 x float> %math.result110 to <4 x half>, !dbg !30
  %math.packed111 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %655) #9, !dbg !29, !frisk.register_lifetimes !1
  %656 = fadd float %651, %653, !dbg !29
  %657 = shufflevector <32 x float> %536, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %658 = shufflevector <32 x float> %537, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %659 = fadd <4 x float> %657, %658, !dbg !27
  %660 = extractelement <4 x float> %659, i64 0, !dbg !28
  %661 = call float @llvm.amdgcn.exp2.f32(float %660), !dbg !28
  %662 = insertelement <4 x float> poison, float %661, i64 0, !dbg !30
  %663 = fadd float %656, %661, !dbg !29
  %664 = extractelement <4 x float> %659, i64 1, !dbg !28
  %665 = call float @llvm.amdgcn.exp2.f32(float %664), !dbg !28
  %666 = insertelement <4 x float> %662, float %665, i64 1, !dbg !30
  %667 = fadd float %663, %665, !dbg !29
  %668 = extractelement <4 x float> %659, i64 2, !dbg !28
  %669 = call float @llvm.amdgcn.exp2.f32(float %668), !dbg !28
  %670 = insertelement <4 x float> %666, float %669, i64 2, !dbg !30
  %671 = fadd float %667, %669, !dbg !29
  %672 = extractelement <4 x float> %659, i64 3, !dbg !28
  %673 = call float @llvm.amdgcn.exp2.f32(float %672), !dbg !28
  %674 = insertelement <4 x float> %670, float %673, i64 3, !dbg !30
  %math.result112 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %674) #9, !dbg !30, !frisk.register_lifetimes !1
  %675 = fptrunc <4 x float> %math.result112 to <4 x half>, !dbg !30
  %math.packed113 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %675) #9, !dbg !29, !frisk.register_lifetimes !1
  %676 = fadd float %671, %673, !dbg !29
  %677 = shufflevector <32 x float> %536, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %678 = shufflevector <32 x float> %537, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %679 = fadd <4 x float> %677, %678, !dbg !27
  %680 = extractelement <4 x float> %679, i64 0, !dbg !28
  %681 = call float @llvm.amdgcn.exp2.f32(float %680), !dbg !28
  %682 = insertelement <4 x float> poison, float %681, i64 0, !dbg !30
  %683 = fadd float %676, %681, !dbg !29
  %684 = extractelement <4 x float> %679, i64 1, !dbg !28
  %685 = call float @llvm.amdgcn.exp2.f32(float %684), !dbg !28
  %686 = insertelement <4 x float> %682, float %685, i64 1, !dbg !30
  %687 = fadd float %683, %685, !dbg !29
  %688 = extractelement <4 x float> %679, i64 2, !dbg !28
  %689 = call float @llvm.amdgcn.exp2.f32(float %688), !dbg !28
  %690 = insertelement <4 x float> %686, float %689, i64 2, !dbg !30
  %691 = fadd float %687, %689, !dbg !29
  %692 = extractelement <4 x float> %679, i64 3, !dbg !28
  %693 = call float @llvm.amdgcn.exp2.f32(float %692), !dbg !28
  %694 = insertelement <4 x float> %690, float %693, i64 3, !dbg !30
  %math.result114 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %694) #9, !dbg !30, !frisk.register_lifetimes !1
  %695 = fptrunc <4 x float> %math.result114 to <4 x half>, !dbg !30
  %math.packed115 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %695) #9, !dbg !29, !frisk.register_lifetimes !1
  %696 = fadd float %691, %693, !dbg !29
  %697 = extractvalue [2 x <32 x float>] %108, 1, !dbg !27
  %698 = extractvalue [2 x <32 x float>] %534, 1, !dbg !27
  %699 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %700 = shufflevector <32 x float> %698, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %701 = fadd <4 x float> %699, %700, !dbg !27
  %702 = extractelement <4 x float> %701, i64 0, !dbg !28
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !28
  %704 = insertelement <4 x float> poison, float %703, i64 0, !dbg !30
  %705 = extractelement <4 x float> %701, i64 1, !dbg !28
  %706 = call float @llvm.amdgcn.exp2.f32(float %705), !dbg !28
  %707 = insertelement <4 x float> %704, float %706, i64 1, !dbg !30
  %708 = fadd float %703, %706, !dbg !29
  %709 = extractelement <4 x float> %701, i64 2, !dbg !28
  %710 = call float @llvm.amdgcn.exp2.f32(float %709), !dbg !28
  %711 = insertelement <4 x float> %707, float %710, i64 2, !dbg !30
  %712 = fadd float %708, %710, !dbg !29
  %713 = extractelement <4 x float> %701, i64 3, !dbg !28
  %714 = call float @llvm.amdgcn.exp2.f32(float %713), !dbg !28
  %715 = insertelement <4 x float> %711, float %714, i64 3, !dbg !30
  %math.result116 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %715) #9, !dbg !30, !frisk.register_lifetimes !1
  %716 = fptrunc <4 x float> %math.result116 to <4 x half>, !dbg !30
  %math.packed117 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %716) #9, !dbg !29, !frisk.register_lifetimes !1
  %717 = fadd float %712, %714, !dbg !29
  %718 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %719 = shufflevector <32 x float> %698, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %720 = fadd <4 x float> %718, %719, !dbg !27
  %721 = extractelement <4 x float> %720, i64 0, !dbg !28
  %722 = call float @llvm.amdgcn.exp2.f32(float %721), !dbg !28
  %723 = insertelement <4 x float> poison, float %722, i64 0, !dbg !30
  %724 = fadd float %717, %722, !dbg !29
  %725 = extractelement <4 x float> %720, i64 1, !dbg !28
  %726 = call float @llvm.amdgcn.exp2.f32(float %725), !dbg !28
  %727 = insertelement <4 x float> %723, float %726, i64 1, !dbg !30
  %728 = fadd float %724, %726, !dbg !29
  %729 = extractelement <4 x float> %720, i64 2, !dbg !28
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !28
  %731 = insertelement <4 x float> %727, float %730, i64 2, !dbg !30
  %732 = fadd float %728, %730, !dbg !29
  %733 = extractelement <4 x float> %720, i64 3, !dbg !28
  %734 = call float @llvm.amdgcn.exp2.f32(float %733), !dbg !28
  %735 = insertelement <4 x float> %731, float %734, i64 3, !dbg !30
  %math.result118 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %735) #9, !dbg !30, !frisk.register_lifetimes !1
  %736 = fptrunc <4 x float> %math.result118 to <4 x half>, !dbg !30
  %math.packed119 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %736) #9, !dbg !29, !frisk.register_lifetimes !1
  %737 = fadd float %732, %734, !dbg !29
  %738 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %739 = shufflevector <32 x float> %698, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %740 = fadd <4 x float> %738, %739, !dbg !27
  %741 = extractelement <4 x float> %740, i64 0, !dbg !28
  %742 = call float @llvm.amdgcn.exp2.f32(float %741), !dbg !28
  %743 = insertelement <4 x float> poison, float %742, i64 0, !dbg !30
  %744 = fadd float %737, %742, !dbg !29
  %745 = extractelement <4 x float> %740, i64 1, !dbg !28
  %746 = call float @llvm.amdgcn.exp2.f32(float %745), !dbg !28
  %747 = insertelement <4 x float> %743, float %746, i64 1, !dbg !30
  %748 = fadd float %744, %746, !dbg !29
  %749 = extractelement <4 x float> %740, i64 2, !dbg !28
  %750 = call float @llvm.amdgcn.exp2.f32(float %749), !dbg !28
  %751 = insertelement <4 x float> %747, float %750, i64 2, !dbg !30
  %752 = fadd float %748, %750, !dbg !29
  %753 = extractelement <4 x float> %740, i64 3, !dbg !28
  %754 = call float @llvm.amdgcn.exp2.f32(float %753), !dbg !28
  %755 = insertelement <4 x float> %751, float %754, i64 3, !dbg !30
  %math.result120 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %755) #9, !dbg !30, !frisk.register_lifetimes !1
  %756 = fptrunc <4 x float> %math.result120 to <4 x half>, !dbg !30
  %math.packed121 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %756) #9, !dbg !29, !frisk.register_lifetimes !1
  %757 = fadd float %752, %754, !dbg !29
  %758 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %759 = shufflevector <32 x float> %698, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %760 = fadd <4 x float> %758, %759, !dbg !27
  %761 = extractelement <4 x float> %760, i64 0, !dbg !28
  %762 = call float @llvm.amdgcn.exp2.f32(float %761), !dbg !28
  %763 = insertelement <4 x float> poison, float %762, i64 0, !dbg !30
  %764 = fadd float %757, %762, !dbg !29
  %765 = extractelement <4 x float> %760, i64 1, !dbg !28
  %766 = call float @llvm.amdgcn.exp2.f32(float %765), !dbg !28
  %767 = insertelement <4 x float> %763, float %766, i64 1, !dbg !30
  %768 = fadd float %764, %766, !dbg !29
  %769 = extractelement <4 x float> %760, i64 2, !dbg !28
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !28
  %771 = insertelement <4 x float> %767, float %770, i64 2, !dbg !30
  %772 = fadd float %768, %770, !dbg !29
  %773 = extractelement <4 x float> %760, i64 3, !dbg !28
  %774 = call float @llvm.amdgcn.exp2.f32(float %773), !dbg !28
  %775 = insertelement <4 x float> %771, float %774, i64 3, !dbg !30
  %math.result122 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %775) #9, !dbg !30, !frisk.register_lifetimes !1
  %776 = fptrunc <4 x float> %math.result122 to <4 x half>, !dbg !30
  %math.packed123 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %776) #9, !dbg !29, !frisk.register_lifetimes !1
  %777 = fadd float %772, %774, !dbg !29
  %778 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %779 = shufflevector <32 x float> %698, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %780 = fadd <4 x float> %778, %779, !dbg !27
  %781 = extractelement <4 x float> %780, i64 0, !dbg !28
  %782 = call float @llvm.amdgcn.exp2.f32(float %781), !dbg !28
  %783 = insertelement <4 x float> poison, float %782, i64 0, !dbg !30
  %784 = fadd float %777, %782, !dbg !29
  %785 = extractelement <4 x float> %780, i64 1, !dbg !28
  %786 = call float @llvm.amdgcn.exp2.f32(float %785), !dbg !28
  %787 = insertelement <4 x float> %783, float %786, i64 1, !dbg !30
  %788 = fadd float %784, %786, !dbg !29
  %789 = extractelement <4 x float> %780, i64 2, !dbg !28
  %790 = call float @llvm.amdgcn.exp2.f32(float %789), !dbg !28
  %791 = insertelement <4 x float> %787, float %790, i64 2, !dbg !30
  %792 = fadd float %788, %790, !dbg !29
  %793 = extractelement <4 x float> %780, i64 3, !dbg !28
  %794 = call float @llvm.amdgcn.exp2.f32(float %793), !dbg !28
  %795 = insertelement <4 x float> %791, float %794, i64 3, !dbg !30
  %math.result124 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %795) #9, !dbg !30, !frisk.register_lifetimes !1
  %796 = fptrunc <4 x float> %math.result124 to <4 x half>, !dbg !30
  %math.packed125 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %796) #9, !dbg !29, !frisk.register_lifetimes !1
  %797 = fadd float %792, %794, !dbg !29
  %798 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %799 = shufflevector <32 x float> %698, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %800 = fadd <4 x float> %798, %799, !dbg !27
  %801 = extractelement <4 x float> %800, i64 0, !dbg !28
  %802 = call float @llvm.amdgcn.exp2.f32(float %801), !dbg !28
  %803 = insertelement <4 x float> poison, float %802, i64 0, !dbg !30
  %804 = fadd float %797, %802, !dbg !29
  %805 = extractelement <4 x float> %800, i64 1, !dbg !28
  %806 = call float @llvm.amdgcn.exp2.f32(float %805), !dbg !28
  %807 = insertelement <4 x float> %803, float %806, i64 1, !dbg !30
  %808 = fadd float %804, %806, !dbg !29
  %809 = extractelement <4 x float> %800, i64 2, !dbg !28
  %810 = call float @llvm.amdgcn.exp2.f32(float %809), !dbg !28
  %811 = insertelement <4 x float> %807, float %810, i64 2, !dbg !30
  %812 = fadd float %808, %810, !dbg !29
  %813 = extractelement <4 x float> %800, i64 3, !dbg !28
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !28
  %815 = insertelement <4 x float> %811, float %814, i64 3, !dbg !30
  %math.result126 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %815) #9, !dbg !30, !frisk.register_lifetimes !1
  %816 = fptrunc <4 x float> %math.result126 to <4 x half>, !dbg !30
  %math.packed127 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %816) #9, !dbg !29, !frisk.register_lifetimes !1
  %817 = fadd float %812, %814, !dbg !29
  %818 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %819 = shufflevector <32 x float> %698, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %820 = fadd <4 x float> %818, %819, !dbg !27
  %821 = extractelement <4 x float> %820, i64 0, !dbg !28
  %822 = call float @llvm.amdgcn.exp2.f32(float %821), !dbg !28
  %823 = insertelement <4 x float> poison, float %822, i64 0, !dbg !30
  %824 = fadd float %817, %822, !dbg !29
  %825 = extractelement <4 x float> %820, i64 1, !dbg !28
  %826 = call float @llvm.amdgcn.exp2.f32(float %825), !dbg !28
  %827 = insertelement <4 x float> %823, float %826, i64 1, !dbg !30
  %828 = fadd float %824, %826, !dbg !29
  %829 = extractelement <4 x float> %820, i64 2, !dbg !28
  %830 = call float @llvm.amdgcn.exp2.f32(float %829), !dbg !28
  %831 = insertelement <4 x float> %827, float %830, i64 2, !dbg !30
  %832 = fadd float %828, %830, !dbg !29
  %833 = extractelement <4 x float> %820, i64 3, !dbg !28
  %834 = call float @llvm.amdgcn.exp2.f32(float %833), !dbg !28
  %835 = insertelement <4 x float> %831, float %834, i64 3, !dbg !30
  %math.result128 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %835) #9, !dbg !30, !frisk.register_lifetimes !1
  %836 = fptrunc <4 x float> %math.result128 to <4 x half>, !dbg !30
  %math.packed129 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %836) #9, !dbg !29, !frisk.register_lifetimes !1
  %837 = fadd float %832, %834, !dbg !29
  %838 = shufflevector <32 x float> %697, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %839 = shufflevector <32 x float> %698, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %840 = fadd <4 x float> %838, %839, !dbg !27
  %841 = extractelement <4 x float> %840, i64 0, !dbg !28
  %842 = call float @llvm.amdgcn.exp2.f32(float %841), !dbg !28
  %843 = insertelement <4 x float> poison, float %842, i64 0, !dbg !30
  %844 = fadd float %837, %842, !dbg !29
  %845 = extractelement <4 x float> %840, i64 1, !dbg !28
  %846 = call float @llvm.amdgcn.exp2.f32(float %845), !dbg !28
  %847 = insertelement <4 x float> %843, float %846, i64 1, !dbg !30
  %848 = fadd float %844, %846, !dbg !29
  %849 = extractelement <4 x float> %840, i64 2, !dbg !28
  %850 = call float @llvm.amdgcn.exp2.f32(float %849), !dbg !28
  %851 = insertelement <4 x float> %847, float %850, i64 2, !dbg !30
  %852 = fadd float %848, %850, !dbg !29
  %853 = extractelement <4 x float> %840, i64 3, !dbg !28
  %854 = call float @llvm.amdgcn.exp2.f32(float %853), !dbg !28
  %855 = insertelement <4 x float> %851, float %854, i64 3, !dbg !30
  %math.result130 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %855) #9, !dbg !30, !frisk.register_lifetimes !1
  %856 = fptrunc <4 x float> %math.result130 to <4 x half>, !dbg !30
  %math.packed131 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %856) #9, !dbg !29, !frisk.register_lifetimes !1
  %857 = fadd float %852, %854, !dbg !29
  %858 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !29
  %859 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %858), !dbg !29
  %860 = add nuw nsw i32 %859, 64, !dbg !29
  %861 = xor i32 %859, 16, !dbg !29
  %862 = and i32 %860, 192, !dbg !29
  %863 = icmp samesign ult i32 %861, %862, !dbg !29
  %864 = select i1 %863, i32 %861, i32 %859, !dbg !29
  %865 = shl nuw nsw i32 %864, 2, !dbg !29
  %866 = bitcast float %696 to i32, !dbg !29
  %867 = call i32 @llvm.amdgcn.ds.bpermute(i32 %865, i32 %866), !dbg !29
  %868 = bitcast i32 %867 to float, !dbg !29
  %869 = fadd float %696, %868, !dbg !29
  %870 = xor i32 %859, 32, !dbg !29
  %871 = icmp samesign ult i32 %870, %862, !dbg !29
  %872 = select i1 %871, i32 %870, i32 %859, !dbg !29
  %873 = shl nuw nsw i32 %872, 2, !dbg !29
  %874 = bitcast float %869 to i32, !dbg !29
  %875 = call i32 @llvm.amdgcn.ds.bpermute(i32 %873, i32 %874), !dbg !29
  %876 = bitcast i32 %875 to float, !dbg !29
  %877 = fadd float %869, %876, !dbg !29
  %878 = extractvalue [2 x <1 x float>] %32, 0, !dbg !29
  %879 = extractelement <1 x float> %878, i64 0, !dbg !29
  %880 = fadd float %877, %879, !dbg !29
  %881 = insertelement <1 x float> poison, float %880, i64 0, !dbg !17
  %882 = insertvalue [2 x <1 x float>] poison, <1 x float> %881, 0, !dbg !17
  %883 = bitcast float %857 to i32, !dbg !29
  %884 = call i32 @llvm.amdgcn.ds.bpermute(i32 %865, i32 %883), !dbg !29
  %885 = bitcast i32 %884 to float, !dbg !29
  %886 = fadd float %857, %885, !dbg !29
  %887 = bitcast float %886 to i32, !dbg !29
  %888 = call i32 @llvm.amdgcn.ds.bpermute(i32 %873, i32 %887), !dbg !29
  %889 = bitcast i32 %888 to float, !dbg !29
  %890 = fadd float %886, %889, !dbg !29
  %891 = extractvalue [2 x <1 x float>] %32, 1, !dbg !29
  %892 = extractelement <1 x float> %891, i64 0, !dbg !29
  %893 = fadd float %890, %892, !dbg !29
  %894 = insertelement <1 x float> poison, float %893, i64 0, !dbg !17
  %895 = insertvalue [2 x <1 x float>] %882, <1 x float> %894, 1, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread358 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !16, !range !35, !frisk.register_lifetimes !1
  %address.local359 = shl nuw nsw i32 %address.thread358, 3, !dbg !16
  %address.local360 = and i32 %address.local359, 120, !dbg !16
  %address.local361 = shl nuw nsw i32 %address.thread358, 5, !dbg !16
  %address.local362 = and i32 %address.local361, 1536, !dbg !16
  %address.local363 = or disjoint i32 %address.local360, %address.local362, !dbg !16
  %address.local364 = zext nneg i32 %address.local363 to i64, !dbg !16
  %address.local365 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local364, !dbg !16
  %896 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local365, i32 0) #0, !dbg !16
  %897 = bitcast <4 x i32> %896 to <8 x half>, !dbg !16
  %address.local366 = zext nneg i32 %address.thread358 to i64, !dbg !14
  %address.local367 = and i64 %address.local366, 60, !dbg !16
  %address.local368 = add nuw nsw i64 %address.local367, %address.local366, !dbg !16
  %address.local369 = shl nuw nsw i64 %address.local368, 3, !dbg !16
  %address.local370 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local369, !dbg !16
  %address.local371 = and i64 %address.local366, 63, !dbg !14
  %address.local372 = lshr i64 %address.local371, 4, !dbg !16
  %address.local373 = mul nuw nsw i64 %address.local372, 384, !dbg !16
  %address.local374 = getelementptr [2 x i8], ptr addrspace(3) %address.local370, i64 %address.local373, !dbg !16
  %address.local375 = lshr i64 %address.local366, 2, !dbg !10
  %address.local376 = mul nsw i64 %address.local375, -32, !dbg !16
  %address.local377 = getelementptr [2 x i8], ptr addrspace(3) %address.local374, i64 %address.local376, !dbg !16
  %address.local378 = getelementptr i8, ptr addrspace(3) %address.local377, i64 256, !dbg !16
  %898 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local378, i32 0) #0, !dbg !16
  %899 = bitcast <4 x i32> %898 to <8 x half>, !dbg !16
  %address.local379 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local369, !dbg !16
  %address.local380 = getelementptr [2 x i8], ptr addrspace(3) %address.local379, i64 %address.local373, !dbg !16
  %address.local381 = getelementptr [2 x i8], ptr addrspace(3) %address.local380, i64 %address.local376, !dbg !16
  %address.local382 = getelementptr i8, ptr addrspace(3) %address.local381, i64 512, !dbg !16
  %900 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local382, i32 0) #0, !dbg !16
  %901 = bitcast <4 x i32> %900 to <8 x half>, !dbg !16
  %address.local383 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local369, !dbg !16
  %address.local384 = getelementptr [2 x i8], ptr addrspace(3) %address.local383, i64 %address.local373, !dbg !16
  %address.local385 = getelementptr [2 x i8], ptr addrspace(3) %address.local384, i64 %address.local376, !dbg !16
  %address.local386 = getelementptr i8, ptr addrspace(3) %address.local385, i64 768, !dbg !16
  %902 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local386, i32 0) #0, !dbg !16
  %903 = bitcast <4 x i32> %902 to <8 x half>, !dbg !16
  %address.local387 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local364, !dbg !16
  %address.local388 = getelementptr i8, ptr addrspace(3) %address.local387, i64 4096, !dbg !16
  %904 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local388, i32 0) #0, !dbg !16
  %905 = bitcast <4 x i32> %904 to <8 x half>, !dbg !16
  %address.local389 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local369, !dbg !16
  %address.local390 = getelementptr [2 x i8], ptr addrspace(3) %address.local389, i64 %address.local373, !dbg !16
  %address.local391 = getelementptr [2 x i8], ptr addrspace(3) %address.local390, i64 %address.local376, !dbg !16
  %address.local392 = getelementptr i8, ptr addrspace(3) %address.local391, i64 4352, !dbg !16
  %906 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local392, i32 0) #0, !dbg !16
  %907 = bitcast <4 x i32> %906 to <8 x half>, !dbg !16
  %address.local393 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local369, !dbg !16
  %address.local394 = getelementptr [2 x i8], ptr addrspace(3) %address.local393, i64 %address.local373, !dbg !16
  %address.local395 = getelementptr [2 x i8], ptr addrspace(3) %address.local394, i64 %address.local376, !dbg !16
  %address.local396 = getelementptr i8, ptr addrspace(3) %address.local395, i64 4608, !dbg !16
  %908 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local396, i32 0) #0, !dbg !16
  %909 = bitcast <4 x i32> %908 to <8 x half>, !dbg !16
  %address.local397 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local369, !dbg !16
  %address.local398 = getelementptr [2 x i8], ptr addrspace(3) %address.local397, i64 %address.local373, !dbg !16
  %address.local399 = getelementptr [2 x i8], ptr addrspace(3) %address.local398, i64 %address.local376, !dbg !16
  %address.local400 = getelementptr i8, ptr addrspace(3) %address.local399, i64 4864, !dbg !16
  %910 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local400, i32 0) #0, !dbg !16
  %911 = bitcast <4 x i32> %910 to <8 x half>, !dbg !16
  %912 = add nsw i64 %68, %10, !dbg !19
  %913 = shl i64 %912, 19, !dbg !19
  %914 = and i64 %913, 16252928, !dbg !19
  %915 = add i64 %914, %74, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %916 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %915, !dbg !19
  %917 = getelementptr i8, ptr addrspace(1) %916, i64 64, !dbg !19
  %address.local401 = shl nuw nsw i64 %address.local366, 3, !dbg !13
  %address.local402 = and i64 %address.local401, 1536, !dbg !13
  %address.local403 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local402, !dbg !19
  %address.local404 = ptrtoint ptr addrspace(3) %address.local403 to i64, !dbg !19
  %address.local405 = trunc i64 %address.local404 to i32, !dbg !19
  %918 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local405), !dbg !19
  %919 = zext i32 %918 to i64, !dbg !19
  %920 = inttoptr i64 %919 to ptr addrspace(3), !dbg !19
  %address.local406 = shl nuw nsw i64 %address.local366, 5, !dbg !19
  %address.local407 = and i64 %address.local406, 8064, !dbg !19
  %address.local408 = shl nuw nsw i64 %address.local366, 3, !dbg !19
  %address.local409 = and i64 %address.local408, 24, !dbg !19
  %address.local410 = or disjoint i64 %address.local407, %address.local409, !dbg !19
  %address.local411 = getelementptr [2 x i8], ptr addrspace(1) %917, i64 %address.local410, !dbg !19
  %921 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %922 = ptrtoint ptr addrspace(1) %address.local411 to i64, !dbg !19
  %923 = sub i64 %922, %921, !dbg !19
  %dma.buffer.offset931 = trunc i64 %923 to i32, !dbg !19
  %924 = trunc i64 %921 to i32, !dbg !19
  %925 = insertelement <4 x i32> poison, i32 %924, i64 0, !dbg !19
  %926 = lshr i64 %921, 32, !dbg !19
  %927 = trunc i64 %926 to i32, !dbg !19
  %928 = insertelement <4 x i32> %925, i32 %927, i64 1, !dbg !19
  %929 = insertelement <4 x i32> %928, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc932 = insertelement <4 x i32> %929, i32 131072, i64 3, !dbg !19
  %930 = ptrtoint ptr addrspace(3) %920 to i32, !dbg !19
  %dma.buffer.lds933 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %930), !dbg !19
  %931 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds933, i32 %dma.buffer.offset931, <4 x i32> %dma.buffer.rsrc932) #10, !dbg !19
  %address.local412 = or disjoint i64 %address.local366, 256, !dbg !13
  %address.local413 = shl nuw nsw i64 %address.local412, 3, !dbg !13
  %address.local414 = and i64 %address.local413, 3584, !dbg !13
  %address.local415 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local414, !dbg !19
  %address.local416 = ptrtoint ptr addrspace(3) %address.local415 to i64, !dbg !19
  %address.local417 = trunc i64 %address.local416 to i32, !dbg !19
  %932 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local417), !dbg !19
  %933 = zext i32 %932 to i64, !dbg !19
  %934 = inttoptr i64 %933 to ptr addrspace(3), !dbg !19
  %address.local418 = shl nuw nsw i64 %address.local412, 5, !dbg !19
  %address.local419 = and i64 %address.local418, 16256, !dbg !19
  %address.local420 = shl nuw nsw i64 %address.local366, 3, !dbg !19
  %address.local421 = and i64 %address.local420, 24, !dbg !19
  %address.local422 = or disjoint i64 %address.local419, %address.local421, !dbg !19
  %address.local423 = getelementptr [2 x i8], ptr addrspace(1) %917, i64 %address.local422, !dbg !19
  %935 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %936 = ptrtoint ptr addrspace(1) %address.local423 to i64, !dbg !19
  %937 = sub i64 %936, %935, !dbg !19
  %dma.buffer.offset928 = trunc i64 %937 to i32, !dbg !19
  %938 = trunc i64 %935 to i32, !dbg !19
  %939 = insertelement <4 x i32> poison, i32 %938, i64 0, !dbg !19
  %940 = lshr i64 %935, 32, !dbg !19
  %941 = trunc i64 %940 to i32, !dbg !19
  %942 = insertelement <4 x i32> %939, i32 %941, i64 1, !dbg !19
  %943 = insertelement <4 x i32> %942, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc929 = insertelement <4 x i32> %943, i32 131072, i64 3, !dbg !19
  %944 = ptrtoint ptr addrspace(3) %934 to i32, !dbg !19
  %dma.buffer.lds930 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %944), !dbg !19
  %945 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds930, i32 %dma.buffer.offset928, <4 x i32> %dma.buffer.rsrc929) #10, !dbg !19
  %946 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %947 = extractvalue [2 x <8 x float>] %31, 0, !dbg !31
  %948 = shufflevector <8 x float> %947, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %946, <4 x float> %948) #9, !dbg !16
  %950 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %951 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %950, <4 x float> %949) #9, !dbg !16
  %952 = shufflevector <8 x half> %901, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %953 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %952, <4 x float> %951) #9, !dbg !16
  %954 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %954, <4 x float> %953) #9, !dbg !16
  %956 = shufflevector <8 x half> %905, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %957 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %956, <4 x float> %955) #9, !dbg !16
  %958 = shufflevector <8 x half> %907, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %958, <4 x float> %957) #9, !dbg !16
  %960 = shufflevector <8 x half> %909, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %960, <4 x float> %959) #9, !dbg !16
  %962 = shufflevector <8 x half> %911, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %962, <4 x float> %961) #9, !dbg !16
  %964 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %965 = shufflevector <8 x float> %947, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %966 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %964, <4 x float> %965) #9, !dbg !16
  %967 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %967, <4 x float> %966) #9, !dbg !16
  %969 = shufflevector <8 x half> %901, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %969, <4 x float> %968) #9, !dbg !16
  %971 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %971, <4 x float> %970) #9, !dbg !16
  %973 = shufflevector <8 x half> %905, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %973, <4 x float> %972) #9, !dbg !16
  %975 = shufflevector <8 x half> %907, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %975, <4 x float> %974) #9, !dbg !16
  %977 = shufflevector <8 x half> %909, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %977, <4 x float> %976) #9, !dbg !16
  %979 = shufflevector <8 x half> %911, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %979, <4 x float> %978) #9, !dbg !16
  %981 = extractvalue [2 x <8 x float>] %31, 1, !dbg !31
  %982 = shufflevector <8 x float> %981, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %946, <4 x float> %982) #9, !dbg !16
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %950, <4 x float> %983) #9, !dbg !16
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %952, <4 x float> %984) #9, !dbg !16
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %954, <4 x float> %985) #9, !dbg !16
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %956, <4 x float> %986) #9, !dbg !16
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %958, <4 x float> %987) #9, !dbg !16
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %960, <4 x float> %988) #9, !dbg !16
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %962, <4 x float> %989) #9, !dbg !16
  %991 = shufflevector <8 x float> %981, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %964, <4 x float> %991) #9, !dbg !16
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %967, <4 x float> %992) #9, !dbg !16
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %969, <4 x float> %993) #9, !dbg !16
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %971, <4 x float> %994) #9, !dbg !16
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %973, <4 x float> %995) #9, !dbg !16
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %975, <4 x float> %996) #9, !dbg !16
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %977, <4 x float> %997) #9, !dbg !16
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %979, <4 x float> %998) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1000 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local365, i32 0) #0, !dbg !16
  %1001 = bitcast <4 x i32> %1000 to <8 x half>, !dbg !16
  %1002 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local378, i32 0) #0, !dbg !16
  %1003 = bitcast <4 x i32> %1002 to <8 x half>, !dbg !16
  %1004 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local382, i32 0) #0, !dbg !16
  %1005 = bitcast <4 x i32> %1004 to <8 x half>, !dbg !16
  %1006 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local386, i32 0) #0, !dbg !16
  %1007 = bitcast <4 x i32> %1006 to <8 x half>, !dbg !16
  %1008 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local388, i32 0) #0, !dbg !16
  %1009 = bitcast <4 x i32> %1008 to <8 x half>, !dbg !16
  %1010 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local392, i32 0) #0, !dbg !16
  %1011 = bitcast <4 x i32> %1010 to <8 x half>, !dbg !16
  %1012 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local396, i32 0) #0, !dbg !16
  %1013 = bitcast <4 x i32> %1012 to <8 x half>, !dbg !16
  %1014 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local400, i32 0) #0, !dbg !16
  %1015 = bitcast <4 x i32> %1014 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1016 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %915, !dbg !19
  %1017 = getelementptr i8, ptr addrspace(1) %1016, i64 128, !dbg !19
  %1018 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local405), !dbg !19
  %1019 = zext i32 %1018 to i64, !dbg !19
  %1020 = inttoptr i64 %1019 to ptr addrspace(3), !dbg !19
  %address.local424 = getelementptr [2 x i8], ptr addrspace(1) %1017, i64 %address.local410, !dbg !19
  %1021 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1022 = ptrtoint ptr addrspace(1) %address.local424 to i64, !dbg !19
  %1023 = sub i64 %1022, %1021, !dbg !19
  %dma.buffer.offset925 = trunc i64 %1023 to i32, !dbg !19
  %1024 = trunc i64 %1021 to i32, !dbg !19
  %1025 = insertelement <4 x i32> poison, i32 %1024, i64 0, !dbg !19
  %1026 = lshr i64 %1021, 32, !dbg !19
  %1027 = trunc i64 %1026 to i32, !dbg !19
  %1028 = insertelement <4 x i32> %1025, i32 %1027, i64 1, !dbg !19
  %1029 = insertelement <4 x i32> %1028, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc926 = insertelement <4 x i32> %1029, i32 131072, i64 3, !dbg !19
  %1030 = ptrtoint ptr addrspace(3) %1020 to i32, !dbg !19
  %dma.buffer.lds927 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1030), !dbg !19
  %1031 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds927, i32 %dma.buffer.offset925, <4 x i32> %dma.buffer.rsrc926) #10, !dbg !19
  %1032 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local417), !dbg !19
  %1033 = zext i32 %1032 to i64, !dbg !19
  %1034 = inttoptr i64 %1033 to ptr addrspace(3), !dbg !19
  %address.local425 = getelementptr [2 x i8], ptr addrspace(1) %1017, i64 %address.local422, !dbg !19
  %1035 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1036 = ptrtoint ptr addrspace(1) %address.local425 to i64, !dbg !19
  %1037 = sub i64 %1036, %1035, !dbg !19
  %dma.buffer.offset922 = trunc i64 %1037 to i32, !dbg !19
  %1038 = trunc i64 %1035 to i32, !dbg !19
  %1039 = insertelement <4 x i32> poison, i32 %1038, i64 0, !dbg !19
  %1040 = lshr i64 %1035, 32, !dbg !19
  %1041 = trunc i64 %1040 to i32, !dbg !19
  %1042 = insertelement <4 x i32> %1039, i32 %1041, i64 1, !dbg !19
  %1043 = insertelement <4 x i32> %1042, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc923 = insertelement <4 x i32> %1043, i32 131072, i64 3, !dbg !19
  %1044 = ptrtoint ptr addrspace(3) %1034 to i32, !dbg !19
  %dma.buffer.lds924 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1044), !dbg !19
  %1045 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds924, i32 %dma.buffer.offset922, <4 x i32> %dma.buffer.rsrc923) #10, !dbg !19
  %1046 = shufflevector <8 x half> %1001, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1047 = extractvalue [2 x <8 x float>] %33, 0, !dbg !31
  %1048 = shufflevector <8 x float> %1047, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1046, <4 x float> %1048) #9, !dbg !16
  %1050 = shufflevector <8 x half> %1003, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1050, <4 x float> %1049) #9, !dbg !16
  %1052 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1052, <4 x float> %1051) #9, !dbg !16
  %1054 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1054, <4 x float> %1053) #9, !dbg !16
  %1056 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1056, <4 x float> %1055) #9, !dbg !16
  %1058 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1058, <4 x float> %1057) #9, !dbg !16
  %1060 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1060, <4 x float> %1059) #9, !dbg !16
  %1062 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1062, <4 x float> %1061) #9, !dbg !16
  %1064 = shufflevector <8 x half> %1001, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1065 = shufflevector <8 x float> %1047, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1064, <4 x float> %1065) #9, !dbg !16
  %1067 = shufflevector <8 x half> %1003, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1067, <4 x float> %1066) #9, !dbg !16
  %1069 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1069, <4 x float> %1068) #9, !dbg !16
  %1071 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1071, <4 x float> %1070) #9, !dbg !16
  %1073 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1073, <4 x float> %1072) #9, !dbg !16
  %1075 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1075, <4 x float> %1074) #9, !dbg !16
  %1077 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1077, <4 x float> %1076) #9, !dbg !16
  %1079 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1079, <4 x float> %1078) #9, !dbg !16
  %1081 = extractvalue [2 x <8 x float>] %33, 1, !dbg !31
  %1082 = shufflevector <8 x float> %1081, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1046, <4 x float> %1082) #9, !dbg !16
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1050, <4 x float> %1083) #9, !dbg !16
  %1085 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1052, <4 x float> %1084) #9, !dbg !16
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1054, <4 x float> %1085) #9, !dbg !16
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1056, <4 x float> %1086) #9, !dbg !16
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1058, <4 x float> %1087) #9, !dbg !16
  %1089 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1060, <4 x float> %1088) #9, !dbg !16
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %1062, <4 x float> %1089) #9, !dbg !16
  %1091 = shufflevector <8 x float> %1081, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1064, <4 x float> %1091) #9, !dbg !16
  %1093 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1067, <4 x float> %1092) #9, !dbg !16
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1069, <4 x float> %1093) #9, !dbg !16
  %1095 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1071, <4 x float> %1094) #9, !dbg !16
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1073, <4 x float> %1095) #9, !dbg !16
  %1097 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1075, <4 x float> %1096) #9, !dbg !16
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1077, <4 x float> %1097) #9, !dbg !16
  %1099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %1079, <4 x float> %1098) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1100 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local365, i32 0) #0, !dbg !16
  %1101 = bitcast <4 x i32> %1100 to <8 x half>, !dbg !16
  %1102 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local378, i32 0) #0, !dbg !16
  %1103 = bitcast <4 x i32> %1102 to <8 x half>, !dbg !16
  %1104 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local382, i32 0) #0, !dbg !16
  %1105 = bitcast <4 x i32> %1104 to <8 x half>, !dbg !16
  %1106 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local386, i32 0) #0, !dbg !16
  %1107 = bitcast <4 x i32> %1106 to <8 x half>, !dbg !16
  %1108 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local388, i32 0) #0, !dbg !16
  %1109 = bitcast <4 x i32> %1108 to <8 x half>, !dbg !16
  %1110 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local392, i32 0) #0, !dbg !16
  %1111 = bitcast <4 x i32> %1110 to <8 x half>, !dbg !16
  %1112 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local396, i32 0) #0, !dbg !16
  %1113 = bitcast <4 x i32> %1112 to <8 x half>, !dbg !16
  %1114 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local400, i32 0) #0, !dbg !16
  %1115 = bitcast <4 x i32> %1114 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1116 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %915, !dbg !19
  %1117 = getelementptr i8, ptr addrspace(1) %1116, i64 192, !dbg !19
  %1118 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local405), !dbg !19
  %1119 = zext i32 %1118 to i64, !dbg !19
  %1120 = inttoptr i64 %1119 to ptr addrspace(3), !dbg !19
  %address.local426 = getelementptr [2 x i8], ptr addrspace(1) %1117, i64 %address.local410, !dbg !19
  %1121 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1122 = ptrtoint ptr addrspace(1) %address.local426 to i64, !dbg !19
  %1123 = sub i64 %1122, %1121, !dbg !19
  %dma.buffer.offset919 = trunc i64 %1123 to i32, !dbg !19
  %1124 = trunc i64 %1121 to i32, !dbg !19
  %1125 = insertelement <4 x i32> poison, i32 %1124, i64 0, !dbg !19
  %1126 = lshr i64 %1121, 32, !dbg !19
  %1127 = trunc i64 %1126 to i32, !dbg !19
  %1128 = insertelement <4 x i32> %1125, i32 %1127, i64 1, !dbg !19
  %1129 = insertelement <4 x i32> %1128, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc920 = insertelement <4 x i32> %1129, i32 131072, i64 3, !dbg !19
  %1130 = ptrtoint ptr addrspace(3) %1120 to i32, !dbg !19
  %dma.buffer.lds921 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1130), !dbg !19
  %1131 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds921, i32 %dma.buffer.offset919, <4 x i32> %dma.buffer.rsrc920) #10, !dbg !19
  %1132 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local417), !dbg !19
  %1133 = zext i32 %1132 to i64, !dbg !19
  %1134 = inttoptr i64 %1133 to ptr addrspace(3), !dbg !19
  %address.local427 = getelementptr [2 x i8], ptr addrspace(1) %1117, i64 %address.local422, !dbg !19
  %1135 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1136 = ptrtoint ptr addrspace(1) %address.local427 to i64, !dbg !19
  %1137 = sub i64 %1136, %1135, !dbg !19
  %dma.buffer.offset916 = trunc i64 %1137 to i32, !dbg !19
  %1138 = trunc i64 %1135 to i32, !dbg !19
  %1139 = insertelement <4 x i32> poison, i32 %1138, i64 0, !dbg !19
  %1140 = lshr i64 %1135, 32, !dbg !19
  %1141 = trunc i64 %1140 to i32, !dbg !19
  %1142 = insertelement <4 x i32> %1139, i32 %1141, i64 1, !dbg !19
  %1143 = insertelement <4 x i32> %1142, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc917 = insertelement <4 x i32> %1143, i32 131072, i64 3, !dbg !19
  %1144 = ptrtoint ptr addrspace(3) %1134 to i32, !dbg !19
  %dma.buffer.lds918 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1144), !dbg !19
  %1145 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds918, i32 %dma.buffer.offset916, <4 x i32> %dma.buffer.rsrc917) #10, !dbg !19
  %1146 = shufflevector <8 x half> %1101, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1147 = extractvalue [2 x <8 x float>] %34, 0, !dbg !31
  %1148 = shufflevector <8 x float> %1147, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1146, <4 x float> %1148) #9, !dbg !16
  %1150 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1150, <4 x float> %1149) #9, !dbg !16
  %1152 = shufflevector <8 x half> %1105, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1152, <4 x float> %1151) #9, !dbg !16
  %1154 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1154, <4 x float> %1153) #9, !dbg !16
  %1156 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1156, <4 x float> %1155) #9, !dbg !16
  %1158 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1158, <4 x float> %1157) #9, !dbg !16
  %1160 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1160, <4 x float> %1159) #9, !dbg !16
  %1162 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1162, <4 x float> %1161) #9, !dbg !16
  %1164 = shufflevector <8 x half> %1101, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1165 = shufflevector <8 x float> %1147, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1164, <4 x float> %1165) #9, !dbg !16
  %1167 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1167, <4 x float> %1166) #9, !dbg !16
  %1169 = shufflevector <8 x half> %1105, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1169, <4 x float> %1168) #9, !dbg !16
  %1171 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1171, <4 x float> %1170) #9, !dbg !16
  %1173 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1173, <4 x float> %1172) #9, !dbg !16
  %1175 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1175, <4 x float> %1174) #9, !dbg !16
  %1177 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1177, <4 x float> %1176) #9, !dbg !16
  %1179 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1179, <4 x float> %1178) #9, !dbg !16
  %1181 = extractvalue [2 x <8 x float>] %34, 1, !dbg !31
  %1182 = shufflevector <8 x float> %1181, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1146, <4 x float> %1182) #9, !dbg !16
  %1184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1150, <4 x float> %1183) #9, !dbg !16
  %1185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1152, <4 x float> %1184) #9, !dbg !16
  %1186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1154, <4 x float> %1185) #9, !dbg !16
  %1187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1156, <4 x float> %1186) #9, !dbg !16
  %1188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1158, <4 x float> %1187) #9, !dbg !16
  %1189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1160, <4 x float> %1188) #9, !dbg !16
  %1190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %1162, <4 x float> %1189) #9, !dbg !16
  %1191 = shufflevector <8 x float> %1181, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1164, <4 x float> %1191) #9, !dbg !16
  %1193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1167, <4 x float> %1192) #9, !dbg !16
  %1194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1169, <4 x float> %1193) #9, !dbg !16
  %1195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1171, <4 x float> %1194) #9, !dbg !16
  %1196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1173, <4 x float> %1195) #9, !dbg !16
  %1197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1175, <4 x float> %1196) #9, !dbg !16
  %1198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1177, <4 x float> %1197) #9, !dbg !16
  %1199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %1179, <4 x float> %1198) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1200 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local365, i32 0) #0, !dbg !16
  %1201 = bitcast <4 x i32> %1200 to <8 x half>, !dbg !16
  %1202 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local378, i32 0) #0, !dbg !16
  %1203 = bitcast <4 x i32> %1202 to <8 x half>, !dbg !16
  %1204 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local382, i32 0) #0, !dbg !16
  %1205 = bitcast <4 x i32> %1204 to <8 x half>, !dbg !16
  %1206 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local386, i32 0) #0, !dbg !16
  %1207 = bitcast <4 x i32> %1206 to <8 x half>, !dbg !16
  %1208 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local388, i32 0) #0, !dbg !16
  %1209 = bitcast <4 x i32> %1208 to <8 x half>, !dbg !16
  %1210 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local392, i32 0) #0, !dbg !16
  %1211 = bitcast <4 x i32> %1210 to <8 x half>, !dbg !16
  %1212 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local396, i32 0) #0, !dbg !16
  %1213 = bitcast <4 x i32> %1212 to <8 x half>, !dbg !16
  %1214 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local400, i32 0) #0, !dbg !16
  %1215 = bitcast <4 x i32> %1214 to <8 x half>, !dbg !16
  %1216 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1217 = extractvalue [2 x <8 x float>] %35, 0, !dbg !31
  %1218 = shufflevector <8 x float> %1217, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1216, <4 x float> %1218) #9, !dbg !16
  %1220 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1220, <4 x float> %1219) #9, !dbg !16
  %1222 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1222, <4 x float> %1221) #9, !dbg !16
  %1224 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1224, <4 x float> %1223) #9, !dbg !16
  %1226 = shufflevector <8 x half> %1209, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1226, <4 x float> %1225) #9, !dbg !16
  %1228 = shufflevector <8 x half> %1211, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1228, <4 x float> %1227) #9, !dbg !16
  %1230 = shufflevector <8 x half> %1213, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1230, <4 x float> %1229) #9, !dbg !16
  %1232 = shufflevector <8 x half> %1215, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1232, <4 x float> %1231) #9, !dbg !16
  %1234 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1235 = shufflevector <8 x float> %1217, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1234, <4 x float> %1235) #9, !dbg !16
  %1237 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1237, <4 x float> %1236) #9, !dbg !16
  %1239 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1239, <4 x float> %1238) #9, !dbg !16
  %1241 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1241, <4 x float> %1240) #9, !dbg !16
  %1243 = shufflevector <8 x half> %1209, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1243, <4 x float> %1242) #9, !dbg !16
  %1245 = shufflevector <8 x half> %1211, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1245, <4 x float> %1244) #9, !dbg !16
  %1247 = shufflevector <8 x half> %1213, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1247, <4 x float> %1246) #9, !dbg !16
  %1249 = shufflevector <8 x half> %1215, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1249, <4 x float> %1248) #9, !dbg !16
  %1251 = extractvalue [2 x <8 x float>] %35, 1, !dbg !31
  %1252 = shufflevector <8 x float> %1251, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1216, <4 x float> %1252) #9, !dbg !16
  %1254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1220, <4 x float> %1253) #9, !dbg !16
  %1255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1222, <4 x float> %1254) #9, !dbg !16
  %1256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1224, <4 x float> %1255) #9, !dbg !16
  %1257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1226, <4 x float> %1256) #9, !dbg !16
  %1258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1228, <4 x float> %1257) #9, !dbg !16
  %1259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1230, <4 x float> %1258) #9, !dbg !16
  %1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %1232, <4 x float> %1259) #9, !dbg !16
  %1261 = shufflevector <8 x float> %1251, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1234, <4 x float> %1261) #9, !dbg !16
  %1263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1237, <4 x float> %1262) #9, !dbg !16
  %1264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1239, <4 x float> %1263) #9, !dbg !16
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1241, <4 x float> %1264) #9, !dbg !16
  %1266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1243, <4 x float> %1265) #9, !dbg !16
  %1267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1245, <4 x float> %1266) #9, !dbg !16
  %1268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1247, <4 x float> %1267) #9, !dbg !16
  %1269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed131, <4 x half> %1249, <4 x float> %1268) #9, !dbg !16
  %1270 = shufflevector <4 x float> %963, <4 x float> %980, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1271 = insertvalue [2 x <8 x float>] poison, <8 x float> %1270, 0, !dbg !17
  %1272 = shufflevector <4 x float> %990, <4 x float> %999, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1273 = insertvalue [2 x <8 x float>] %1271, <8 x float> %1272, 1, !dbg !17
  %1274 = shufflevector <4 x float> %1063, <4 x float> %1080, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1275 = insertvalue [2 x <8 x float>] poison, <8 x float> %1274, 0, !dbg !17
  %1276 = shufflevector <4 x float> %1090, <4 x float> %1099, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1277 = insertvalue [2 x <8 x float>] %1275, <8 x float> %1276, 1, !dbg !17
  %1278 = shufflevector <4 x float> %1163, <4 x float> %1180, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1279 = insertvalue [2 x <8 x float>] poison, <8 x float> %1278, 0, !dbg !17
  %1280 = shufflevector <4 x float> %1190, <4 x float> %1199, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1281 = insertvalue [2 x <8 x float>] %1279, <8 x float> %1280, 1, !dbg !17
  %1282 = shufflevector <4 x float> %1233, <4 x float> %1250, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1283 = insertvalue [2 x <8 x float>] poison, <8 x float> %1282, 0, !dbg !17
  %1284 = shufflevector <4 x float> %1260, <4 x float> %1269, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1285 = insertvalue [2 x <8 x float>] %1283, <8 x float> %1284, 1, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1286 = add nuw nsw i64 %30, 1, !dbg !17
  br label %29, !dbg !17

1287:                                             ; preds = %29
  %1288 = and i32 %21, 16, !dbg !17
  %.not3 = icmp eq i32 %1288, 0, !dbg !17
  %1289 = and i32 %21, 32, !dbg !17
  %.not4 = icmp eq i32 %1289, 0, !dbg !17
  %1290 = extractvalue [2 x <8 x float>] %35, 0, !dbg !17
  %1291 = extractelement <8 x float> %1290, i64 0, !dbg !17
  %1292 = extractelement <8 x float> %1290, i64 1, !dbg !17
  %1293 = extractelement <8 x float> %1290, i64 2, !dbg !17
  %1294 = extractelement <8 x float> %1290, i64 3, !dbg !17
  %1295 = select i1 %.not3, float %1292, float %1291, !dbg !17
  %1296 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !17
  %1297 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1296), !dbg !17
  %1298 = add nuw nsw i32 %1297, 64, !dbg !17
  %1299 = xor i32 %1297, 16, !dbg !17
  %1300 = and i32 %1298, 192, !dbg !17
  %1301 = icmp samesign ult i32 %1299, %1300, !dbg !17
  %1302 = select i1 %1301, i32 %1299, i32 %1297, !dbg !17
  %1303 = shl nuw nsw i32 %1302, 2, !dbg !17
  %1304 = bitcast float %1295 to i32, !dbg !17
  %1305 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1304), !dbg !17
  %1306 = bitcast i32 %1305 to float, !dbg !17
  %1307 = select i1 %.not3, float %1291, float %1306, !dbg !17
  %1308 = select i1 %.not3, float %1306, float %1292, !dbg !17
  %1309 = select i1 %.not3, float %1294, float %1293, !dbg !17
  %1310 = bitcast float %1309 to i32, !dbg !17
  %1311 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1310), !dbg !17
  %1312 = bitcast i32 %1311 to float, !dbg !17
  %1313 = select i1 %.not3, float %1293, float %1312, !dbg !17
  %1314 = select i1 %.not3, float %1312, float %1294, !dbg !17
  %1315 = select i1 %.not4, float %1313, float %1307, !dbg !17
  %1316 = xor i32 %1297, 32, !dbg !17
  %1317 = icmp samesign ult i32 %1316, %1300, !dbg !17
  %1318 = select i1 %1317, i32 %1316, i32 %1297, !dbg !17
  %1319 = shl nuw nsw i32 %1318, 2, !dbg !17
  %1320 = bitcast float %1315 to i32, !dbg !17
  %1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1320), !dbg !17
  %1322 = bitcast i32 %1321 to float, !dbg !17
  %1323 = select i1 %.not4, float %1307, float %1322, !dbg !17
  %1324 = select i1 %.not4, float %1322, float %1313, !dbg !17
  %1325 = select i1 %.not4, float %1314, float %1308, !dbg !17
  %1326 = bitcast float %1325 to i32, !dbg !17
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1326), !dbg !17
  %1328 = bitcast i32 %1327 to float, !dbg !17
  %1329 = select i1 %.not4, float %1308, float %1328, !dbg !17
  %1330 = select i1 %.not4, float %1328, float %1314, !dbg !17
  %1331 = extractvalue [2 x <8 x float>] %34, 0, !dbg !17
  %1332 = extractelement <8 x float> %1331, i64 0, !dbg !17
  %1333 = extractelement <8 x float> %1331, i64 1, !dbg !17
  %1334 = extractelement <8 x float> %1331, i64 2, !dbg !17
  %1335 = extractelement <8 x float> %1331, i64 3, !dbg !17
  %1336 = select i1 %.not3, float %1333, float %1332, !dbg !17
  %1337 = bitcast float %1336 to i32, !dbg !17
  %1338 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1337), !dbg !17
  %1339 = bitcast i32 %1338 to float, !dbg !17
  %1340 = select i1 %.not3, float %1332, float %1339, !dbg !17
  %1341 = select i1 %.not3, float %1339, float %1333, !dbg !17
  %1342 = select i1 %.not3, float %1335, float %1334, !dbg !17
  %1343 = bitcast float %1342 to i32, !dbg !17
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1343), !dbg !17
  %1345 = bitcast i32 %1344 to float, !dbg !17
  %1346 = select i1 %.not3, float %1334, float %1345, !dbg !17
  %1347 = select i1 %.not3, float %1345, float %1335, !dbg !17
  %1348 = select i1 %.not4, float %1346, float %1340, !dbg !17
  %1349 = bitcast float %1348 to i32, !dbg !17
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1349), !dbg !17
  %1351 = bitcast i32 %1350 to float, !dbg !17
  %1352 = select i1 %.not4, float %1340, float %1351, !dbg !17
  %1353 = select i1 %.not4, float %1351, float %1346, !dbg !17
  %1354 = select i1 %.not4, float %1347, float %1341, !dbg !17
  %1355 = bitcast float %1354 to i32, !dbg !17
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1355), !dbg !17
  %1357 = bitcast i32 %1356 to float, !dbg !17
  %1358 = select i1 %.not4, float %1341, float %1357, !dbg !17
  %1359 = select i1 %.not4, float %1357, float %1347, !dbg !17
  %1360 = extractvalue [2 x <8 x float>] %33, 0, !dbg !17
  %1361 = extractelement <8 x float> %1360, i64 0, !dbg !17
  %1362 = extractelement <8 x float> %1360, i64 1, !dbg !17
  %1363 = extractelement <8 x float> %1360, i64 2, !dbg !17
  %1364 = extractelement <8 x float> %1360, i64 3, !dbg !17
  %1365 = select i1 %.not3, float %1362, float %1361, !dbg !17
  %1366 = bitcast float %1365 to i32, !dbg !17
  %1367 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1366), !dbg !17
  %1368 = bitcast i32 %1367 to float, !dbg !17
  %1369 = select i1 %.not3, float %1361, float %1368, !dbg !17
  %1370 = select i1 %.not3, float %1368, float %1362, !dbg !17
  %1371 = select i1 %.not3, float %1364, float %1363, !dbg !17
  %1372 = bitcast float %1371 to i32, !dbg !17
  %1373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1372), !dbg !17
  %1374 = bitcast i32 %1373 to float, !dbg !17
  %1375 = select i1 %.not3, float %1363, float %1374, !dbg !17
  %1376 = select i1 %.not3, float %1374, float %1364, !dbg !17
  %1377 = select i1 %.not4, float %1375, float %1369, !dbg !17
  %1378 = bitcast float %1377 to i32, !dbg !17
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1378), !dbg !17
  %1380 = bitcast i32 %1379 to float, !dbg !17
  %1381 = select i1 %.not4, float %1369, float %1380, !dbg !17
  %1382 = select i1 %.not4, float %1380, float %1375, !dbg !17
  %1383 = select i1 %.not4, float %1376, float %1370, !dbg !17
  %1384 = bitcast float %1383 to i32, !dbg !17
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1384), !dbg !17
  %1386 = bitcast i32 %1385 to float, !dbg !17
  %1387 = select i1 %.not4, float %1370, float %1386, !dbg !17
  %1388 = select i1 %.not4, float %1386, float %1376, !dbg !17
  %1389 = extractvalue [2 x <8 x float>] %31, 0, !dbg !17
  %1390 = extractelement <8 x float> %1389, i64 0, !dbg !17
  %1391 = extractelement <8 x float> %1389, i64 1, !dbg !17
  %1392 = extractelement <8 x float> %1389, i64 2, !dbg !17
  %1393 = extractelement <8 x float> %1389, i64 3, !dbg !17
  %1394 = select i1 %.not3, float %1391, float %1390, !dbg !17
  %1395 = bitcast float %1394 to i32, !dbg !17
  %1396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1395), !dbg !17
  %1397 = bitcast i32 %1396 to float, !dbg !17
  %1398 = select i1 %.not3, float %1390, float %1397, !dbg !17
  %1399 = select i1 %.not3, float %1397, float %1391, !dbg !17
  %1400 = select i1 %.not3, float %1393, float %1392, !dbg !17
  %1401 = bitcast float %1400 to i32, !dbg !17
  %1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1401), !dbg !17
  %1403 = bitcast i32 %1402 to float, !dbg !17
  %1404 = select i1 %.not3, float %1392, float %1403, !dbg !17
  %1405 = select i1 %.not3, float %1403, float %1393, !dbg !17
  %1406 = select i1 %.not4, float %1404, float %1398, !dbg !17
  %1407 = bitcast float %1406 to i32, !dbg !17
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1407), !dbg !17
  %1409 = select i1 %.not4, float %1405, float %1399, !dbg !17
  %1410 = bitcast float %1409 to i32, !dbg !17
  %1411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1410), !dbg !17
  %1412 = extractvalue [2 x <1 x float>] %32, 0, !dbg !20
  %1413 = shufflevector <1 x float> %1412, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1414 = extractelement <8 x float> %1290, i64 4, !dbg !17
  %1415 = extractelement <8 x float> %1290, i64 5, !dbg !17
  %1416 = extractelement <8 x float> %1290, i64 6, !dbg !17
  %1417 = extractelement <8 x float> %1290, i64 7, !dbg !17
  %1418 = select i1 %.not3, float %1415, float %1414, !dbg !17
  %1419 = bitcast float %1418 to i32, !dbg !17
  %1420 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1419), !dbg !17
  %1421 = bitcast i32 %1420 to float, !dbg !17
  %1422 = select i1 %.not3, float %1414, float %1421, !dbg !17
  %1423 = select i1 %.not3, float %1421, float %1415, !dbg !17
  %1424 = select i1 %.not3, float %1417, float %1416, !dbg !17
  %1425 = bitcast float %1424 to i32, !dbg !17
  %1426 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1425), !dbg !17
  %1427 = bitcast i32 %1426 to float, !dbg !17
  %1428 = select i1 %.not3, float %1416, float %1427, !dbg !17
  %1429 = select i1 %.not3, float %1427, float %1417, !dbg !17
  %1430 = select i1 %.not4, float %1428, float %1422, !dbg !17
  %1431 = bitcast float %1430 to i32, !dbg !17
  %1432 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1431), !dbg !17
  %1433 = bitcast i32 %1432 to float, !dbg !17
  %1434 = select i1 %.not4, float %1422, float %1433, !dbg !17
  %1435 = select i1 %.not4, float %1433, float %1428, !dbg !17
  %1436 = select i1 %.not4, float %1429, float %1423, !dbg !17
  %1437 = bitcast float %1436 to i32, !dbg !17
  %1438 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1437), !dbg !17
  %1439 = bitcast i32 %1438 to float, !dbg !17
  %1440 = select i1 %.not4, float %1423, float %1439, !dbg !17
  %1441 = select i1 %.not4, float %1439, float %1429, !dbg !17
  %1442 = extractelement <8 x float> %1331, i64 4, !dbg !17
  %1443 = extractelement <8 x float> %1331, i64 5, !dbg !17
  %1444 = extractelement <8 x float> %1331, i64 6, !dbg !17
  %1445 = extractelement <8 x float> %1331, i64 7, !dbg !17
  %1446 = select i1 %.not3, float %1443, float %1442, !dbg !17
  %1447 = bitcast float %1446 to i32, !dbg !17
  %1448 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1447), !dbg !17
  %1449 = bitcast i32 %1448 to float, !dbg !17
  %1450 = select i1 %.not3, float %1442, float %1449, !dbg !17
  %1451 = select i1 %.not3, float %1449, float %1443, !dbg !17
  %1452 = select i1 %.not3, float %1445, float %1444, !dbg !17
  %1453 = bitcast float %1452 to i32, !dbg !17
  %1454 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1453), !dbg !17
  %1455 = bitcast i32 %1454 to float, !dbg !17
  %1456 = select i1 %.not3, float %1444, float %1455, !dbg !17
  %1457 = select i1 %.not3, float %1455, float %1445, !dbg !17
  %1458 = select i1 %.not4, float %1456, float %1450, !dbg !17
  %1459 = bitcast float %1458 to i32, !dbg !17
  %1460 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1459), !dbg !17
  %1461 = bitcast i32 %1460 to float, !dbg !17
  %1462 = select i1 %.not4, float %1450, float %1461, !dbg !17
  %1463 = select i1 %.not4, float %1461, float %1456, !dbg !17
  %1464 = select i1 %.not4, float %1457, float %1451, !dbg !17
  %1465 = bitcast float %1464 to i32, !dbg !17
  %1466 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1465), !dbg !17
  %1467 = bitcast i32 %1466 to float, !dbg !17
  %1468 = select i1 %.not4, float %1451, float %1467, !dbg !17
  %1469 = select i1 %.not4, float %1467, float %1457, !dbg !17
  %1470 = extractelement <8 x float> %1360, i64 4, !dbg !17
  %1471 = extractelement <8 x float> %1360, i64 5, !dbg !17
  %1472 = extractelement <8 x float> %1360, i64 6, !dbg !17
  %1473 = extractelement <8 x float> %1360, i64 7, !dbg !17
  %1474 = select i1 %.not3, float %1471, float %1470, !dbg !17
  %1475 = bitcast float %1474 to i32, !dbg !17
  %1476 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1475), !dbg !17
  %1477 = bitcast i32 %1476 to float, !dbg !17
  %1478 = select i1 %.not3, float %1470, float %1477, !dbg !17
  %1479 = select i1 %.not3, float %1477, float %1471, !dbg !17
  %1480 = select i1 %.not3, float %1473, float %1472, !dbg !17
  %1481 = bitcast float %1480 to i32, !dbg !17
  %1482 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1481), !dbg !17
  %1483 = bitcast i32 %1482 to float, !dbg !17
  %1484 = select i1 %.not3, float %1472, float %1483, !dbg !17
  %1485 = select i1 %.not3, float %1483, float %1473, !dbg !17
  %1486 = select i1 %.not4, float %1484, float %1478, !dbg !17
  %1487 = bitcast float %1486 to i32, !dbg !17
  %1488 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1487), !dbg !17
  %1489 = bitcast i32 %1488 to float, !dbg !17
  %1490 = select i1 %.not4, float %1478, float %1489, !dbg !17
  %1491 = select i1 %.not4, float %1489, float %1484, !dbg !17
  %1492 = select i1 %.not4, float %1485, float %1479, !dbg !17
  %1493 = bitcast float %1492 to i32, !dbg !17
  %1494 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1493), !dbg !17
  %1495 = bitcast i32 %1494 to float, !dbg !17
  %1496 = select i1 %.not4, float %1479, float %1495, !dbg !17
  %1497 = select i1 %.not4, float %1495, float %1485, !dbg !17
  %1498 = extractelement <8 x float> %1389, i64 4, !dbg !17
  %1499 = extractelement <8 x float> %1389, i64 5, !dbg !17
  %1500 = extractelement <8 x float> %1389, i64 6, !dbg !17
  %1501 = extractelement <8 x float> %1389, i64 7, !dbg !17
  %1502 = select i1 %.not3, float %1499, float %1498, !dbg !17
  %1503 = bitcast float %1502 to i32, !dbg !17
  %1504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1503), !dbg !17
  %1505 = bitcast i32 %1504 to float, !dbg !17
  %1506 = select i1 %.not3, float %1498, float %1505, !dbg !17
  %1507 = select i1 %.not3, float %1505, float %1499, !dbg !17
  %1508 = select i1 %.not3, float %1501, float %1500, !dbg !17
  %1509 = bitcast float %1508 to i32, !dbg !17
  %1510 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1509), !dbg !17
  %1511 = bitcast i32 %1510 to float, !dbg !17
  %1512 = select i1 %.not3, float %1500, float %1511, !dbg !17
  %1513 = select i1 %.not3, float %1511, float %1501, !dbg !17
  %1514 = select i1 %.not4, float %1512, float %1506, !dbg !17
  %1515 = bitcast float %1514 to i32, !dbg !17
  %1516 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1515), !dbg !17
  %1517 = select i1 %.not4, float %1513, float %1507, !dbg !17
  %1518 = bitcast float %1517 to i32, !dbg !17
  %1519 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1518), !dbg !17
  %1520 = extractvalue [2 x <8 x float>] %35, 1, !dbg !17
  %1521 = extractelement <8 x float> %1520, i64 0, !dbg !17
  %1522 = extractelement <8 x float> %1520, i64 1, !dbg !17
  %1523 = extractelement <8 x float> %1520, i64 2, !dbg !17
  %1524 = extractelement <8 x float> %1520, i64 3, !dbg !17
  %1525 = select i1 %.not3, float %1522, float %1521, !dbg !17
  %1526 = bitcast float %1525 to i32, !dbg !17
  %1527 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1526), !dbg !17
  %1528 = bitcast i32 %1527 to float, !dbg !17
  %1529 = select i1 %.not3, float %1521, float %1528, !dbg !17
  %1530 = select i1 %.not3, float %1528, float %1522, !dbg !17
  %1531 = select i1 %.not3, float %1524, float %1523, !dbg !17
  %1532 = bitcast float %1531 to i32, !dbg !17
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1532), !dbg !17
  %1534 = bitcast i32 %1533 to float, !dbg !17
  %1535 = select i1 %.not3, float %1523, float %1534, !dbg !17
  %1536 = select i1 %.not3, float %1534, float %1524, !dbg !17
  %1537 = select i1 %.not4, float %1535, float %1529, !dbg !17
  %1538 = bitcast float %1537 to i32, !dbg !17
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1538), !dbg !17
  %1540 = bitcast i32 %1539 to float, !dbg !17
  %1541 = select i1 %.not4, float %1529, float %1540, !dbg !17
  %1542 = select i1 %.not4, float %1540, float %1535, !dbg !17
  %1543 = select i1 %.not4, float %1536, float %1530, !dbg !17
  %1544 = bitcast float %1543 to i32, !dbg !17
  %1545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1544), !dbg !17
  %1546 = bitcast i32 %1545 to float, !dbg !17
  %1547 = select i1 %.not4, float %1530, float %1546, !dbg !17
  %1548 = select i1 %.not4, float %1546, float %1536, !dbg !17
  %1549 = extractvalue [2 x <8 x float>] %34, 1, !dbg !17
  %1550 = extractelement <8 x float> %1549, i64 0, !dbg !17
  %1551 = extractelement <8 x float> %1549, i64 1, !dbg !17
  %1552 = extractelement <8 x float> %1549, i64 2, !dbg !17
  %1553 = extractelement <8 x float> %1549, i64 3, !dbg !17
  %1554 = select i1 %.not3, float %1551, float %1550, !dbg !17
  %1555 = bitcast float %1554 to i32, !dbg !17
  %1556 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1555), !dbg !17
  %1557 = bitcast i32 %1556 to float, !dbg !17
  %1558 = select i1 %.not3, float %1550, float %1557, !dbg !17
  %1559 = select i1 %.not3, float %1557, float %1551, !dbg !17
  %1560 = select i1 %.not3, float %1553, float %1552, !dbg !17
  %1561 = bitcast float %1560 to i32, !dbg !17
  %1562 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1561), !dbg !17
  %1563 = bitcast i32 %1562 to float, !dbg !17
  %1564 = select i1 %.not3, float %1552, float %1563, !dbg !17
  %1565 = select i1 %.not3, float %1563, float %1553, !dbg !17
  %1566 = select i1 %.not4, float %1564, float %1558, !dbg !17
  %1567 = bitcast float %1566 to i32, !dbg !17
  %1568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1567), !dbg !17
  %1569 = bitcast i32 %1568 to float, !dbg !17
  %1570 = select i1 %.not4, float %1558, float %1569, !dbg !17
  %1571 = select i1 %.not4, float %1569, float %1564, !dbg !17
  %1572 = select i1 %.not4, float %1565, float %1559, !dbg !17
  %1573 = bitcast float %1572 to i32, !dbg !17
  %1574 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1573), !dbg !17
  %1575 = bitcast i32 %1574 to float, !dbg !17
  %1576 = select i1 %.not4, float %1559, float %1575, !dbg !17
  %1577 = select i1 %.not4, float %1575, float %1565, !dbg !17
  %1578 = extractvalue [2 x <8 x float>] %33, 1, !dbg !17
  %1579 = extractelement <8 x float> %1578, i64 0, !dbg !17
  %1580 = extractelement <8 x float> %1578, i64 1, !dbg !17
  %1581 = extractelement <8 x float> %1578, i64 2, !dbg !17
  %1582 = extractelement <8 x float> %1578, i64 3, !dbg !17
  %1583 = select i1 %.not3, float %1580, float %1579, !dbg !17
  %1584 = bitcast float %1583 to i32, !dbg !17
  %1585 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1584), !dbg !17
  %1586 = bitcast i32 %1585 to float, !dbg !17
  %1587 = select i1 %.not3, float %1579, float %1586, !dbg !17
  %1588 = select i1 %.not3, float %1586, float %1580, !dbg !17
  %1589 = select i1 %.not3, float %1582, float %1581, !dbg !17
  %1590 = bitcast float %1589 to i32, !dbg !17
  %1591 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1590), !dbg !17
  %1592 = bitcast i32 %1591 to float, !dbg !17
  %1593 = select i1 %.not3, float %1581, float %1592, !dbg !17
  %1594 = select i1 %.not3, float %1592, float %1582, !dbg !17
  %1595 = select i1 %.not4, float %1593, float %1587, !dbg !17
  %1596 = bitcast float %1595 to i32, !dbg !17
  %1597 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1596), !dbg !17
  %1598 = bitcast i32 %1597 to float, !dbg !17
  %1599 = select i1 %.not4, float %1587, float %1598, !dbg !17
  %1600 = select i1 %.not4, float %1598, float %1593, !dbg !17
  %1601 = select i1 %.not4, float %1594, float %1588, !dbg !17
  %1602 = bitcast float %1601 to i32, !dbg !17
  %1603 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1602), !dbg !17
  %1604 = bitcast i32 %1603 to float, !dbg !17
  %1605 = select i1 %.not4, float %1588, float %1604, !dbg !17
  %1606 = select i1 %.not4, float %1604, float %1594, !dbg !17
  %1607 = extractvalue [2 x <8 x float>] %31, 1, !dbg !17
  %1608 = extractelement <8 x float> %1607, i64 0, !dbg !17
  %1609 = extractelement <8 x float> %1607, i64 1, !dbg !17
  %1610 = extractelement <8 x float> %1607, i64 2, !dbg !17
  %1611 = extractelement <8 x float> %1607, i64 3, !dbg !17
  %1612 = select i1 %.not3, float %1609, float %1608, !dbg !17
  %1613 = bitcast float %1612 to i32, !dbg !17
  %1614 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1613), !dbg !17
  %1615 = bitcast i32 %1614 to float, !dbg !17
  %1616 = select i1 %.not3, float %1608, float %1615, !dbg !17
  %1617 = select i1 %.not3, float %1615, float %1609, !dbg !17
  %1618 = select i1 %.not3, float %1611, float %1610, !dbg !17
  %1619 = bitcast float %1618 to i32, !dbg !17
  %1620 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1619), !dbg !17
  %1621 = bitcast i32 %1620 to float, !dbg !17
  %1622 = select i1 %.not3, float %1610, float %1621, !dbg !17
  %1623 = select i1 %.not3, float %1621, float %1611, !dbg !17
  %1624 = select i1 %.not4, float %1622, float %1616, !dbg !17
  %1625 = bitcast float %1624 to i32, !dbg !17
  %1626 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1625), !dbg !17
  %1627 = select i1 %.not4, float %1623, float %1617, !dbg !17
  %1628 = bitcast float %1627 to i32, !dbg !17
  %1629 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1628), !dbg !17
  %1630 = extractvalue [2 x <1 x float>] %32, 1, !dbg !20
  %1631 = shufflevector <1 x float> %1630, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1632 = extractelement <8 x float> %1520, i64 4, !dbg !17
  %1633 = extractelement <8 x float> %1520, i64 5, !dbg !17
  %1634 = extractelement <8 x float> %1520, i64 6, !dbg !17
  %1635 = extractelement <8 x float> %1520, i64 7, !dbg !17
  %1636 = select i1 %.not3, float %1633, float %1632, !dbg !17
  %1637 = bitcast float %1636 to i32, !dbg !17
  %1638 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1637), !dbg !17
  %1639 = bitcast i32 %1638 to float, !dbg !17
  %1640 = select i1 %.not3, float %1632, float %1639, !dbg !17
  %1641 = select i1 %.not3, float %1639, float %1633, !dbg !17
  %1642 = select i1 %.not3, float %1635, float %1634, !dbg !17
  %1643 = bitcast float %1642 to i32, !dbg !17
  %1644 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1643), !dbg !17
  %1645 = bitcast i32 %1644 to float, !dbg !17
  %1646 = select i1 %.not3, float %1634, float %1645, !dbg !17
  %1647 = select i1 %.not3, float %1645, float %1635, !dbg !17
  %1648 = select i1 %.not4, float %1646, float %1640, !dbg !17
  %1649 = bitcast float %1648 to i32, !dbg !17
  %1650 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1649), !dbg !17
  %1651 = bitcast i32 %1650 to float, !dbg !17
  %1652 = select i1 %.not4, float %1640, float %1651, !dbg !17
  %1653 = select i1 %.not4, float %1651, float %1646, !dbg !17
  %1654 = select i1 %.not4, float %1647, float %1641, !dbg !17
  %1655 = bitcast float %1654 to i32, !dbg !17
  %1656 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1655), !dbg !17
  %1657 = bitcast i32 %1656 to float, !dbg !17
  %1658 = select i1 %.not4, float %1641, float %1657, !dbg !17
  %1659 = select i1 %.not4, float %1657, float %1647, !dbg !17
  %1660 = extractelement <8 x float> %1549, i64 4, !dbg !17
  %1661 = extractelement <8 x float> %1549, i64 5, !dbg !17
  %1662 = extractelement <8 x float> %1549, i64 6, !dbg !17
  %1663 = extractelement <8 x float> %1549, i64 7, !dbg !17
  %1664 = select i1 %.not3, float %1661, float %1660, !dbg !17
  %1665 = bitcast float %1664 to i32, !dbg !17
  %1666 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1665), !dbg !17
  %1667 = bitcast i32 %1666 to float, !dbg !17
  %1668 = select i1 %.not3, float %1660, float %1667, !dbg !17
  %1669 = select i1 %.not3, float %1667, float %1661, !dbg !17
  %1670 = select i1 %.not3, float %1663, float %1662, !dbg !17
  %1671 = bitcast float %1670 to i32, !dbg !17
  %1672 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1671), !dbg !17
  %1673 = bitcast i32 %1672 to float, !dbg !17
  %1674 = select i1 %.not3, float %1662, float %1673, !dbg !17
  %1675 = select i1 %.not3, float %1673, float %1663, !dbg !17
  %1676 = select i1 %.not4, float %1674, float %1668, !dbg !17
  %1677 = bitcast float %1676 to i32, !dbg !17
  %1678 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1677), !dbg !17
  %1679 = bitcast i32 %1678 to float, !dbg !17
  %1680 = select i1 %.not4, float %1668, float %1679, !dbg !17
  %1681 = select i1 %.not4, float %1679, float %1674, !dbg !17
  %1682 = select i1 %.not4, float %1675, float %1669, !dbg !17
  %1683 = bitcast float %1682 to i32, !dbg !17
  %1684 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1683), !dbg !17
  %1685 = bitcast i32 %1684 to float, !dbg !17
  %1686 = select i1 %.not4, float %1669, float %1685, !dbg !17
  %1687 = select i1 %.not4, float %1685, float %1675, !dbg !17
  %1688 = extractelement <8 x float> %1578, i64 4, !dbg !17
  %1689 = extractelement <8 x float> %1578, i64 5, !dbg !17
  %1690 = extractelement <8 x float> %1578, i64 6, !dbg !17
  %1691 = extractelement <8 x float> %1578, i64 7, !dbg !17
  %1692 = select i1 %.not3, float %1689, float %1688, !dbg !17
  %1693 = bitcast float %1692 to i32, !dbg !17
  %1694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1693), !dbg !17
  %1695 = bitcast i32 %1694 to float, !dbg !17
  %1696 = select i1 %.not3, float %1688, float %1695, !dbg !17
  %1697 = select i1 %.not3, float %1695, float %1689, !dbg !17
  %1698 = select i1 %.not3, float %1691, float %1690, !dbg !17
  %1699 = bitcast float %1698 to i32, !dbg !17
  %1700 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1699), !dbg !17
  %1701 = bitcast i32 %1700 to float, !dbg !17
  %1702 = select i1 %.not3, float %1690, float %1701, !dbg !17
  %1703 = select i1 %.not3, float %1701, float %1691, !dbg !17
  %1704 = select i1 %.not4, float %1702, float %1696, !dbg !17
  %1705 = bitcast float %1704 to i32, !dbg !17
  %1706 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1705), !dbg !17
  %1707 = bitcast i32 %1706 to float, !dbg !17
  %1708 = select i1 %.not4, float %1696, float %1707, !dbg !17
  %1709 = select i1 %.not4, float %1707, float %1702, !dbg !17
  %1710 = select i1 %.not4, float %1703, float %1697, !dbg !17
  %1711 = bitcast float %1710 to i32, !dbg !17
  %1712 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1711), !dbg !17
  %1713 = bitcast i32 %1712 to float, !dbg !17
  %1714 = select i1 %.not4, float %1697, float %1713, !dbg !17
  %1715 = select i1 %.not4, float %1713, float %1703, !dbg !17
  %1716 = extractelement <8 x float> %1607, i64 4, !dbg !17
  %1717 = extractelement <8 x float> %1607, i64 5, !dbg !17
  %1718 = extractelement <8 x float> %1607, i64 6, !dbg !17
  %1719 = extractelement <8 x float> %1607, i64 7, !dbg !17
  %1720 = select i1 %.not3, float %1717, float %1716, !dbg !17
  %1721 = bitcast float %1720 to i32, !dbg !17
  %1722 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1721), !dbg !17
  %1723 = bitcast i32 %1722 to float, !dbg !17
  %1724 = select i1 %.not3, float %1716, float %1723, !dbg !17
  %1725 = select i1 %.not3, float %1723, float %1717, !dbg !17
  %1726 = select i1 %.not3, float %1719, float %1718, !dbg !17
  %1727 = bitcast float %1726 to i32, !dbg !17
  %1728 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1303, i32 %1727), !dbg !17
  %1729 = bitcast i32 %1728 to float, !dbg !17
  %1730 = select i1 %.not3, float %1718, float %1729, !dbg !17
  %1731 = select i1 %.not3, float %1729, float %1719, !dbg !17
  %1732 = select i1 %.not4, float %1730, float %1724, !dbg !17
  %1733 = bitcast float %1732 to i32, !dbg !17
  %1734 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1733), !dbg !17
  %1735 = select i1 %.not4, float %1731, float %1725, !dbg !17
  %1736 = bitcast float %1735 to i32, !dbg !17
  %1737 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1319, i32 %1736), !dbg !17
  br i1 true, label %1738, label %1800, !dbg !22

1738:                                             ; preds = %1287
  %1739 = bitcast i32 %1734 to float, !dbg !17
  %1740 = select i1 %.not4, float %1724, float %1739, !dbg !17
  %1741 = insertelement <4 x float> poison, float %1740, i64 0, !dbg !20
  %1742 = bitcast i32 %1737 to float, !dbg !17
  %1743 = select i1 %.not4, float %1725, float %1742, !dbg !17
  %1744 = insertelement <4 x float> %1741, float %1743, i64 1, !dbg !20
  %1745 = select i1 %.not4, float %1739, float %1730, !dbg !17
  %1746 = insertelement <4 x float> %1744, float %1745, i64 2, !dbg !20
  %1747 = select i1 %.not4, float %1742, float %1731, !dbg !17
  %1748 = insertelement <4 x float> %1746, float %1747, i64 3, !dbg !20
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1631, !dbg !20
  %normalized = fmul <4 x float> %1748, %row.reciprocal, !dbg !20
  %1749 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !21
  %1750 = extractelement <4 x half> %1749, i64 3, !dbg !21
  %1751 = extractelement <4 x half> %1749, i64 2, !dbg !21
  %1752 = extractelement <4 x half> %1749, i64 1, !dbg !21
  %1753 = extractelement <4 x half> %1749, i64 0, !dbg !21
  %1754 = bitcast i32 %1626 to float, !dbg !17
  %1755 = select i1 %.not4, float %1616, float %1754, !dbg !17
  %1756 = insertelement <4 x float> poison, float %1755, i64 0, !dbg !20
  %1757 = bitcast i32 %1629 to float, !dbg !17
  %1758 = select i1 %.not4, float %1617, float %1757, !dbg !17
  %1759 = insertelement <4 x float> %1756, float %1758, i64 1, !dbg !20
  %1760 = select i1 %.not4, float %1754, float %1622, !dbg !17
  %1761 = insertelement <4 x float> %1759, float %1760, i64 2, !dbg !20
  %1762 = select i1 %.not4, float %1757, float %1623, !dbg !17
  %1763 = insertelement <4 x float> %1761, float %1762, i64 3, !dbg !20
  %normalized74 = fmul <4 x float> %1763, %row.reciprocal, !dbg !20
  %1764 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !21
  %1765 = extractelement <4 x half> %1764, i64 3, !dbg !21
  %1766 = extractelement <4 x half> %1764, i64 2, !dbg !21
  %1767 = extractelement <4 x half> %1764, i64 1, !dbg !21
  %1768 = extractelement <4 x half> %1764, i64 0, !dbg !21
  %1769 = bitcast i32 %1516 to float, !dbg !17
  %1770 = select i1 %.not4, float %1506, float %1769, !dbg !17
  %1771 = insertelement <4 x float> poison, float %1770, i64 0, !dbg !20
  %1772 = bitcast i32 %1519 to float, !dbg !17
  %1773 = select i1 %.not4, float %1507, float %1772, !dbg !17
  %1774 = insertelement <4 x float> %1771, float %1773, i64 1, !dbg !20
  %1775 = select i1 %.not4, float %1769, float %1512, !dbg !17
  %1776 = insertelement <4 x float> %1774, float %1775, i64 2, !dbg !20
  %1777 = select i1 %.not4, float %1772, float %1513, !dbg !17
  %1778 = insertelement <4 x float> %1776, float %1777, i64 3, !dbg !20
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1413, !dbg !20
  %normalized76 = fmul <4 x float> %1778, %row.reciprocal75, !dbg !20
  %1779 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !21
  %1780 = extractelement <4 x half> %1779, i64 3, !dbg !21
  %1781 = extractelement <4 x half> %1779, i64 2, !dbg !21
  %1782 = extractelement <4 x half> %1779, i64 1, !dbg !21
  %1783 = extractelement <4 x half> %1779, i64 0, !dbg !21
  %1784 = bitcast i32 %1408 to float, !dbg !17
  %1785 = select i1 %.not4, float %1398, float %1784, !dbg !17
  %1786 = insertelement <4 x float> poison, float %1785, i64 0, !dbg !20
  %1787 = bitcast i32 %1411 to float, !dbg !17
  %1788 = select i1 %.not4, float %1399, float %1787, !dbg !17
  %1789 = insertelement <4 x float> %1786, float %1788, i64 1, !dbg !20
  %1790 = select i1 %.not4, float %1784, float %1404, !dbg !17
  %1791 = insertelement <4 x float> %1789, float %1790, i64 2, !dbg !20
  %1792 = select i1 %.not4, float %1787, float %1405, !dbg !17
  %1793 = insertelement <4 x float> %1791, float %1792, i64 3, !dbg !20
  %normalized77 = fmul <4 x float> %1793, %row.reciprocal75, !dbg !20
  %1794 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !21
  %1795 = extractelement <4 x half> %1794, i64 3, !dbg !21
  %1796 = extractelement <4 x half> %1794, i64 2, !dbg !21
  %1797 = extractelement <4 x half> %1794, i64 1, !dbg !21
  %1798 = extractelement <4 x half> %1794, i64 0, !dbg !21
  %1799 = shl nuw nsw i64 %17, 19, !dbg !22
  %address.thread428 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !22, !range !35, !frisk.register_lifetimes !1
  %address.local429 = zext nneg i32 %address.thread428 to i64, !dbg !14
  %address.local430 = add nsw i64 %18, %address.local429, !dbg !22
  %address.local431 = lshr i64 %address.local429, 6, !dbg !14
  %address.local432 = shl nuw nsw i64 %address.local431, 4, !dbg !22
  %address.local433 = add nsw i64 %address.local430, %address.local432, !dbg !22
  %address.local434 = and i64 %address.local429, 240, !dbg !22
  %address.local435 = sub nsw i64 %address.local433, %address.local434, !dbg !22
  %address.local436 = add nsw i64 %address.local435, %19, !dbg !22
  %address.local437 = shl nuw nsw i64 %address.local436, 7, !dbg !22
  %address.local438 = add nuw nsw i64 %1799, %address.local437, !dbg !22
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local438, !dbg !22
  %address.local440 = lshr i64 %address.local429, 2, !dbg !14
  %address.local441 = and i64 %address.local440, 12, !dbg !14
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local439, i64 %address.local441, !dbg !22
  store half %1798, ptr addrspace(1) %address.local442, align 2, !dbg !22
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local438, !dbg !22
  %address.local444 = or disjoint i64 %address.local441, 1, !dbg !22
  %address.local445 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local443, i64 %address.local444, !dbg !22
  store half %1797, ptr addrspace(1) %address.local445, align 2, !dbg !22
  %address.local446 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local438, !dbg !22
  %address.local447 = or disjoint i64 %address.local441, 2, !dbg !22
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local446, i64 %address.local447, !dbg !22
  store half %1796, ptr addrspace(1) %address.local448, align 2, !dbg !22
  %address.local449 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local438, !dbg !22
  %address.local450 = or disjoint i64 %address.local441, 3, !dbg !22
  %address.local451 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local449, i64 %address.local450, !dbg !22
  store half %1795, ptr addrspace(1) %address.local451, align 2, !dbg !22
  %address.local452 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local438, !dbg !22
  %address.local453 = or disjoint i64 %address.local441, 16, !dbg !22
  %address.local454 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local452, i64 %address.local453, !dbg !22
  store half %1783, ptr addrspace(1) %address.local454, align 2, !dbg !22
  %address.local455 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local438, !dbg !22
  %address.local456 = or disjoint i64 %address.local441, 17, !dbg !22
  %address.local457 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local455, i64 %address.local456, !dbg !22
  store half %1782, ptr addrspace(1) %address.local457, align 2, !dbg !22
  %address.local458 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local438, !dbg !22
  %address.local459 = or disjoint i64 %address.local441, 18, !dbg !22
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local458, i64 %address.local459, !dbg !22
  store half %1781, ptr addrspace(1) %address.local460, align 2, !dbg !22
  %address.local461 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local438, !dbg !22
  %address.local462 = or disjoint i64 %address.local441, 19, !dbg !22
  %address.local463 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local461, i64 %address.local462, !dbg !22
  store half %1780, ptr addrspace(1) %address.local463, align 2, !dbg !22
  %address.local464 = shl nsw i64 %address.local436, 7, !dbg !22
  %address.local465 = add i64 %address.local464, 8192, !dbg !22
  %address.local466 = add nuw nsw i64 %1799, %address.local465, !dbg !22
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !22
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local467, i64 %address.local441, !dbg !22
  store half %1768, ptr addrspace(1) %address.local468, align 2, !dbg !22
  %address.local469 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !22
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local469, i64 %address.local444, !dbg !22
  store half %1767, ptr addrspace(1) %address.local470, align 2, !dbg !22
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !22
  %address.local472 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local471, i64 %address.local447, !dbg !22
  store half %1766, ptr addrspace(1) %address.local472, align 2, !dbg !22
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !22
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local473, i64 %address.local450, !dbg !22
  store half %1765, ptr addrspace(1) %address.local474, align 2, !dbg !22
  %address.local475 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !22
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local475, i64 %address.local453, !dbg !22
  store half %1753, ptr addrspace(1) %address.local476, align 2, !dbg !22
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !22
  %address.local478 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local477, i64 %address.local456, !dbg !22
  store half %1752, ptr addrspace(1) %address.local478, align 2, !dbg !22
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !22
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local479, i64 %address.local459, !dbg !22
  store half %1751, ptr addrspace(1) %address.local480, align 2, !dbg !22
  %address.local481 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !22
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local481, i64 %address.local462, !dbg !22
  store half %1750, ptr addrspace(1) %address.local482, align 2, !dbg !22
  br label %1800, !dbg !22

1800:                                             ; preds = %1738, %1287
  br i1 true, label %1801, label %1841, !dbg !22

1801:                                             ; preds = %1800
  %1802 = insertelement <4 x float> poison, float %1708, i64 0, !dbg !20
  %1803 = insertelement <4 x float> %1802, float %1714, i64 1, !dbg !20
  %1804 = insertelement <4 x float> %1803, float %1709, i64 2, !dbg !20
  %1805 = insertelement <4 x float> %1804, float %1715, i64 3, !dbg !20
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1631, !dbg !20
  %normalized79 = fmul <4 x float> %1805, %row.reciprocal78, !dbg !20
  %1806 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !21
  %1807 = extractelement <4 x half> %1806, i64 3, !dbg !21
  %1808 = extractelement <4 x half> %1806, i64 2, !dbg !21
  %1809 = extractelement <4 x half> %1806, i64 1, !dbg !21
  %1810 = extractelement <4 x half> %1806, i64 0, !dbg !21
  %1811 = insertelement <4 x float> poison, float %1599, i64 0, !dbg !20
  %1812 = insertelement <4 x float> %1811, float %1605, i64 1, !dbg !20
  %1813 = insertelement <4 x float> %1812, float %1600, i64 2, !dbg !20
  %1814 = insertelement <4 x float> %1813, float %1606, i64 3, !dbg !20
  %normalized80 = fmul <4 x float> %1814, %row.reciprocal78, !dbg !20
  %1815 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !21
  %1816 = extractelement <4 x half> %1815, i64 3, !dbg !21
  %1817 = extractelement <4 x half> %1815, i64 2, !dbg !21
  %1818 = extractelement <4 x half> %1815, i64 1, !dbg !21
  %1819 = extractelement <4 x half> %1815, i64 0, !dbg !21
  %1820 = insertelement <4 x float> poison, float %1490, i64 0, !dbg !20
  %1821 = insertelement <4 x float> %1820, float %1496, i64 1, !dbg !20
  %1822 = insertelement <4 x float> %1821, float %1491, i64 2, !dbg !20
  %1823 = insertelement <4 x float> %1822, float %1497, i64 3, !dbg !20
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1413, !dbg !20
  %normalized82 = fmul <4 x float> %1823, %row.reciprocal81, !dbg !20
  %1824 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !21
  %1825 = extractelement <4 x half> %1824, i64 3, !dbg !21
  %1826 = extractelement <4 x half> %1824, i64 2, !dbg !21
  %1827 = extractelement <4 x half> %1824, i64 1, !dbg !21
  %1828 = extractelement <4 x half> %1824, i64 0, !dbg !21
  %1829 = insertelement <4 x float> poison, float %1381, i64 0, !dbg !20
  %1830 = insertelement <4 x float> %1829, float %1387, i64 1, !dbg !20
  %1831 = insertelement <4 x float> %1830, float %1382, i64 2, !dbg !20
  %1832 = insertelement <4 x float> %1831, float %1388, i64 3, !dbg !20
  %normalized83 = fmul <4 x float> %1832, %row.reciprocal81, !dbg !20
  %1833 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !21
  %1834 = extractelement <4 x half> %1833, i64 3, !dbg !21
  %1835 = extractelement <4 x half> %1833, i64 2, !dbg !21
  %1836 = extractelement <4 x half> %1833, i64 1, !dbg !21
  %1837 = extractelement <4 x half> %1833, i64 0, !dbg !21
  %1838 = add nsw i64 %10, %15, !dbg !22
  %1839 = shl nsw i64 %1838, 19, !dbg !22
  %1840 = and i64 %1839, 16252928, !dbg !22
  %address.thread483 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !22, !range !35, !frisk.register_lifetimes !1
  %address.local484 = zext nneg i32 %address.thread483 to i64, !dbg !14
  %address.local485 = add nsw i64 %18, %address.local484, !dbg !22
  %address.local486 = lshr i64 %address.local484, 6, !dbg !14
  %address.local487 = shl nuw nsw i64 %address.local486, 4, !dbg !22
  %address.local488 = add nsw i64 %address.local485, %address.local487, !dbg !22
  %address.local489 = and i64 %address.local484, 240, !dbg !22
  %address.local490 = sub nsw i64 %address.local488, %address.local489, !dbg !22
  %address.local491 = add nsw i64 %address.local490, %19, !dbg !22
  %address.local492 = shl nuw nsw i64 %address.local491, 7, !dbg !22
  %address.local493 = add nuw nsw i64 %1840, %address.local492, !dbg !22
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local493, !dbg !22
  %address.local495 = lshr i64 %address.local484, 2, !dbg !14
  %address.local496 = and i64 %address.local495, 12, !dbg !14
  %address.local497 = or disjoint i64 %address.local496, 32, !dbg !22
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local494, i64 %address.local497, !dbg !22
  store half %1837, ptr addrspace(1) %address.local498, align 2, !dbg !22
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local493, !dbg !22
  %address.local500 = or disjoint i64 %address.local496, 33, !dbg !22
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local499, i64 %address.local500, !dbg !22
  store half %1836, ptr addrspace(1) %address.local501, align 2, !dbg !22
  %address.local502 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local493, !dbg !22
  %address.local503 = or disjoint i64 %address.local496, 34, !dbg !22
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local502, i64 %address.local503, !dbg !22
  store half %1835, ptr addrspace(1) %address.local504, align 2, !dbg !22
  %address.local505 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local493, !dbg !22
  %address.local506 = or disjoint i64 %address.local496, 35, !dbg !22
  %address.local507 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local505, i64 %address.local506, !dbg !22
  store half %1834, ptr addrspace(1) %address.local507, align 2, !dbg !22
  %address.local508 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local493, !dbg !22
  %address.local509 = or disjoint i64 %address.local496, 48, !dbg !22
  %address.local510 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local508, i64 %address.local509, !dbg !22
  store half %1828, ptr addrspace(1) %address.local510, align 2, !dbg !22
  %address.local511 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local493, !dbg !22
  %address.local512 = or disjoint i64 %address.local496, 49, !dbg !22
  %address.local513 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local511, i64 %address.local512, !dbg !22
  store half %1827, ptr addrspace(1) %address.local513, align 2, !dbg !22
  %address.local514 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local493, !dbg !22
  %address.local515 = or disjoint i64 %address.local496, 50, !dbg !22
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local514, i64 %address.local515, !dbg !22
  store half %1826, ptr addrspace(1) %address.local516, align 2, !dbg !22
  %address.local517 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local493, !dbg !22
  %address.local518 = or i64 %address.local495, 51, !dbg !22
  %address.local519 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local517, i64 %address.local518, !dbg !22
  store half %1825, ptr addrspace(1) %address.local519, align 2, !dbg !22
  %address.local520 = shl nsw i64 %address.local491, 7, !dbg !22
  %address.local521 = add i64 %address.local520, 8192, !dbg !22
  %address.local522 = add nuw nsw i64 %1840, %address.local521, !dbg !22
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local522, !dbg !22
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local523, i64 %address.local497, !dbg !22
  store half %1819, ptr addrspace(1) %address.local524, align 2, !dbg !22
  %address.local525 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local522, !dbg !22
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local525, i64 %address.local500, !dbg !22
  store half %1818, ptr addrspace(1) %address.local526, align 2, !dbg !22
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local522, !dbg !22
  %address.local528 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local527, i64 %address.local503, !dbg !22
  store half %1817, ptr addrspace(1) %address.local528, align 2, !dbg !22
  %address.local529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local522, !dbg !22
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local529, i64 %address.local506, !dbg !22
  store half %1816, ptr addrspace(1) %address.local530, align 2, !dbg !22
  %address.local531 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local522, !dbg !22
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local531, i64 %address.local509, !dbg !22
  store half %1810, ptr addrspace(1) %address.local532, align 2, !dbg !22
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local522, !dbg !22
  %address.local534 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local533, i64 %address.local512, !dbg !22
  store half %1809, ptr addrspace(1) %address.local534, align 2, !dbg !22
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local522, !dbg !22
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local535, i64 %address.local515, !dbg !22
  store half %1808, ptr addrspace(1) %address.local536, align 2, !dbg !22
  %address.local537 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local522, !dbg !22
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local537, i64 %address.local518, !dbg !22
  store half %1807, ptr addrspace(1) %address.local538, align 2, !dbg !22
  br label %1841, !dbg !22

1841:                                             ; preds = %1801, %1800
  br i1 true, label %1842, label %1882, !dbg !22

1842:                                             ; preds = %1841
  %1843 = insertelement <4 x float> poison, float %1680, i64 0, !dbg !20
  %1844 = insertelement <4 x float> %1843, float %1686, i64 1, !dbg !20
  %1845 = insertelement <4 x float> %1844, float %1681, i64 2, !dbg !20
  %1846 = insertelement <4 x float> %1845, float %1687, i64 3, !dbg !20
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1631, !dbg !20
  %normalized85 = fmul <4 x float> %1846, %row.reciprocal84, !dbg !20
  %1847 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !21
  %1848 = extractelement <4 x half> %1847, i64 3, !dbg !21
  %1849 = extractelement <4 x half> %1847, i64 2, !dbg !21
  %1850 = extractelement <4 x half> %1847, i64 1, !dbg !21
  %1851 = extractelement <4 x half> %1847, i64 0, !dbg !21
  %1852 = insertelement <4 x float> poison, float %1570, i64 0, !dbg !20
  %1853 = insertelement <4 x float> %1852, float %1576, i64 1, !dbg !20
  %1854 = insertelement <4 x float> %1853, float %1571, i64 2, !dbg !20
  %1855 = insertelement <4 x float> %1854, float %1577, i64 3, !dbg !20
  %normalized86 = fmul <4 x float> %1855, %row.reciprocal84, !dbg !20
  %1856 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !21
  %1857 = extractelement <4 x half> %1856, i64 3, !dbg !21
  %1858 = extractelement <4 x half> %1856, i64 2, !dbg !21
  %1859 = extractelement <4 x half> %1856, i64 1, !dbg !21
  %1860 = extractelement <4 x half> %1856, i64 0, !dbg !21
  %1861 = insertelement <4 x float> poison, float %1462, i64 0, !dbg !20
  %1862 = insertelement <4 x float> %1861, float %1468, i64 1, !dbg !20
  %1863 = insertelement <4 x float> %1862, float %1463, i64 2, !dbg !20
  %1864 = insertelement <4 x float> %1863, float %1469, i64 3, !dbg !20
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1413, !dbg !20
  %normalized88 = fmul <4 x float> %1864, %row.reciprocal87, !dbg !20
  %1865 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !21
  %1866 = extractelement <4 x half> %1865, i64 3, !dbg !21
  %1867 = extractelement <4 x half> %1865, i64 2, !dbg !21
  %1868 = extractelement <4 x half> %1865, i64 1, !dbg !21
  %1869 = extractelement <4 x half> %1865, i64 0, !dbg !21
  %1870 = insertelement <4 x float> poison, float %1352, i64 0, !dbg !20
  %1871 = insertelement <4 x float> %1870, float %1358, i64 1, !dbg !20
  %1872 = insertelement <4 x float> %1871, float %1353, i64 2, !dbg !20
  %1873 = insertelement <4 x float> %1872, float %1359, i64 3, !dbg !20
  %normalized89 = fmul <4 x float> %1873, %row.reciprocal87, !dbg !20
  %1874 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !21
  %1875 = extractelement <4 x half> %1874, i64 3, !dbg !21
  %1876 = extractelement <4 x half> %1874, i64 2, !dbg !21
  %1877 = extractelement <4 x half> %1874, i64 1, !dbg !21
  %1878 = extractelement <4 x half> %1874, i64 0, !dbg !21
  %1879 = add nsw i64 %10, %15, !dbg !22
  %1880 = shl nsw i64 %1879, 19, !dbg !22
  %1881 = and i64 %1880, 16252928, !dbg !22
  %address.thread539 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !22, !range !35, !frisk.register_lifetimes !1
  %address.local540 = zext nneg i32 %address.thread539 to i64, !dbg !14
  %address.local541 = add nsw i64 %18, %address.local540, !dbg !22
  %address.local542 = lshr i64 %address.local540, 6, !dbg !14
  %address.local543 = shl nuw nsw i64 %address.local542, 4, !dbg !22
  %address.local544 = add nsw i64 %address.local541, %address.local543, !dbg !22
  %address.local545 = and i64 %address.local540, 240, !dbg !22
  %address.local546 = sub nsw i64 %address.local544, %address.local545, !dbg !22
  %address.local547 = add nsw i64 %address.local546, %19, !dbg !22
  %address.local548 = shl nuw nsw i64 %address.local547, 7, !dbg !22
  %address.local549 = add nuw nsw i64 %1881, %address.local548, !dbg !22
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local549, !dbg !22
  %address.local551 = lshr i64 %address.local540, 2, !dbg !14
  %address.local552 = and i64 %address.local551, 12, !dbg !14
  %address.local553 = or disjoint i64 %address.local552, 64, !dbg !22
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local550, i64 %address.local553, !dbg !22
  store half %1878, ptr addrspace(1) %address.local554, align 2, !dbg !22
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local549, !dbg !22
  %address.local556 = or disjoint i64 %address.local552, 65, !dbg !22
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local555, i64 %address.local556, !dbg !22
  store half %1877, ptr addrspace(1) %address.local557, align 2, !dbg !22
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local549, !dbg !22
  %address.local559 = or disjoint i64 %address.local552, 66, !dbg !22
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local558, i64 %address.local559, !dbg !22
  store half %1876, ptr addrspace(1) %address.local560, align 2, !dbg !22
  %address.local561 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local549, !dbg !22
  %address.local562 = or disjoint i64 %address.local552, 67, !dbg !22
  %address.local563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local561, i64 %address.local562, !dbg !22
  store half %1875, ptr addrspace(1) %address.local563, align 2, !dbg !22
  %address.local564 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local549, !dbg !22
  %address.local565 = or disjoint i64 %address.local552, 80, !dbg !22
  %address.local566 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local564, i64 %address.local565, !dbg !22
  store half %1869, ptr addrspace(1) %address.local566, align 2, !dbg !22
  %address.local567 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local549, !dbg !22
  %address.local568 = or disjoint i64 %address.local552, 81, !dbg !22
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local567, i64 %address.local568, !dbg !22
  store half %1868, ptr addrspace(1) %address.local569, align 2, !dbg !22
  %address.local570 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local549, !dbg !22
  %address.local571 = or disjoint i64 %address.local552, 82, !dbg !22
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local570, i64 %address.local571, !dbg !22
  store half %1867, ptr addrspace(1) %address.local572, align 2, !dbg !22
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local549, !dbg !22
  %address.local574 = or disjoint i64 %address.local552, 83, !dbg !22
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local573, i64 %address.local574, !dbg !22
  store half %1866, ptr addrspace(1) %address.local575, align 2, !dbg !22
  %address.local576 = shl nsw i64 %address.local547, 7, !dbg !22
  %address.local577 = add i64 %address.local576, 8192, !dbg !22
  %address.local578 = add nuw nsw i64 %1881, %address.local577, !dbg !22
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local578, !dbg !22
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local579, i64 %address.local553, !dbg !22
  store half %1860, ptr addrspace(1) %address.local580, align 2, !dbg !22
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local578, !dbg !22
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local556, !dbg !22
  store half %1859, ptr addrspace(1) %address.local582, align 2, !dbg !22
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local578, !dbg !22
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local559, !dbg !22
  store half %1858, ptr addrspace(1) %address.local584, align 2, !dbg !22
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local578, !dbg !22
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local585, i64 %address.local562, !dbg !22
  store half %1857, ptr addrspace(1) %address.local586, align 2, !dbg !22
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local578, !dbg !22
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local587, i64 %address.local565, !dbg !22
  store half %1851, ptr addrspace(1) %address.local588, align 2, !dbg !22
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local578, !dbg !22
  %address.local590 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local589, i64 %address.local568, !dbg !22
  store half %1850, ptr addrspace(1) %address.local590, align 2, !dbg !22
  %address.local591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local578, !dbg !22
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local591, i64 %address.local571, !dbg !22
  store half %1849, ptr addrspace(1) %address.local592, align 2, !dbg !22
  %address.local593 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local578, !dbg !22
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local593, i64 %address.local574, !dbg !22
  store half %1848, ptr addrspace(1) %address.local594, align 2, !dbg !22
  br label %1882, !dbg !22

1882:                                             ; preds = %1842, %1841
  br i1 true, label %1883, label %1923, !dbg !22

1883:                                             ; preds = %1882
  %1884 = insertelement <4 x float> poison, float %1652, i64 0, !dbg !20
  %1885 = insertelement <4 x float> %1884, float %1658, i64 1, !dbg !20
  %1886 = insertelement <4 x float> %1885, float %1653, i64 2, !dbg !20
  %1887 = insertelement <4 x float> %1886, float %1659, i64 3, !dbg !20
  %row.reciprocal90 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1631, !dbg !20
  %normalized91 = fmul <4 x float> %1887, %row.reciprocal90, !dbg !20
  %1888 = fptrunc <4 x float> %normalized91 to <4 x half>, !dbg !21
  %1889 = extractelement <4 x half> %1888, i64 3, !dbg !21
  %1890 = extractelement <4 x half> %1888, i64 2, !dbg !21
  %1891 = extractelement <4 x half> %1888, i64 1, !dbg !21
  %1892 = extractelement <4 x half> %1888, i64 0, !dbg !21
  %1893 = insertelement <4 x float> poison, float %1541, i64 0, !dbg !20
  %1894 = insertelement <4 x float> %1893, float %1547, i64 1, !dbg !20
  %1895 = insertelement <4 x float> %1894, float %1542, i64 2, !dbg !20
  %1896 = insertelement <4 x float> %1895, float %1548, i64 3, !dbg !20
  %normalized92 = fmul <4 x float> %1896, %row.reciprocal90, !dbg !20
  %1897 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !21
  %1898 = extractelement <4 x half> %1897, i64 3, !dbg !21
  %1899 = extractelement <4 x half> %1897, i64 2, !dbg !21
  %1900 = extractelement <4 x half> %1897, i64 1, !dbg !21
  %1901 = extractelement <4 x half> %1897, i64 0, !dbg !21
  %1902 = insertelement <4 x float> poison, float %1434, i64 0, !dbg !20
  %1903 = insertelement <4 x float> %1902, float %1440, i64 1, !dbg !20
  %1904 = insertelement <4 x float> %1903, float %1435, i64 2, !dbg !20
  %1905 = insertelement <4 x float> %1904, float %1441, i64 3, !dbg !20
  %row.reciprocal93 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1413, !dbg !20
  %normalized94 = fmul <4 x float> %1905, %row.reciprocal93, !dbg !20
  %1906 = fptrunc <4 x float> %normalized94 to <4 x half>, !dbg !21
  %1907 = extractelement <4 x half> %1906, i64 3, !dbg !21
  %1908 = extractelement <4 x half> %1906, i64 2, !dbg !21
  %1909 = extractelement <4 x half> %1906, i64 1, !dbg !21
  %1910 = extractelement <4 x half> %1906, i64 0, !dbg !21
  %1911 = insertelement <4 x float> poison, float %1323, i64 0, !dbg !20
  %1912 = insertelement <4 x float> %1911, float %1329, i64 1, !dbg !20
  %1913 = insertelement <4 x float> %1912, float %1324, i64 2, !dbg !20
  %1914 = insertelement <4 x float> %1913, float %1330, i64 3, !dbg !20
  %normalized95 = fmul <4 x float> %1914, %row.reciprocal93, !dbg !20
  %1915 = fptrunc <4 x float> %normalized95 to <4 x half>, !dbg !21
  %1916 = extractelement <4 x half> %1915, i64 3, !dbg !21
  %1917 = extractelement <4 x half> %1915, i64 2, !dbg !21
  %1918 = extractelement <4 x half> %1915, i64 1, !dbg !21
  %1919 = extractelement <4 x half> %1915, i64 0, !dbg !21
  %1920 = add nsw i64 %10, %15, !dbg !22
  %1921 = shl nsw i64 %1920, 19, !dbg !22
  %1922 = and i64 %1921, 16252928, !dbg !22
  %address.thread595 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !22, !range !35, !frisk.register_lifetimes !1
  %address.local596 = zext nneg i32 %address.thread595 to i64, !dbg !14
  %address.local597 = add nsw i64 %18, %address.local596, !dbg !22
  %address.local598 = lshr i64 %address.local596, 6, !dbg !14
  %address.local599 = shl nuw nsw i64 %address.local598, 4, !dbg !22
  %address.local600 = add nsw i64 %address.local597, %address.local599, !dbg !22
  %address.local601 = and i64 %address.local596, 240, !dbg !22
  %address.local602 = sub nsw i64 %address.local600, %address.local601, !dbg !22
  %address.local603 = add nsw i64 %address.local602, %19, !dbg !22
  %address.local604 = shl nuw nsw i64 %address.local603, 7, !dbg !22
  %address.local605 = add nuw nsw i64 %1922, %address.local604, !dbg !22
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local605, !dbg !22
  %address.local607 = lshr i64 %address.local596, 2, !dbg !14
  %address.local608 = and i64 %address.local607, 12, !dbg !14
  %address.local609 = or disjoint i64 %address.local608, 96, !dbg !22
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local606, i64 %address.local609, !dbg !22
  store half %1919, ptr addrspace(1) %address.local610, align 2, !dbg !22
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local605, !dbg !22
  %address.local612 = or disjoint i64 %address.local608, 97, !dbg !22
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local611, i64 %address.local612, !dbg !22
  store half %1918, ptr addrspace(1) %address.local613, align 2, !dbg !22
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local605, !dbg !22
  %address.local615 = or disjoint i64 %address.local608, 98, !dbg !22
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local614, i64 %address.local615, !dbg !22
  store half %1917, ptr addrspace(1) %address.local616, align 2, !dbg !22
  %address.local617 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local605, !dbg !22
  %address.local618 = or disjoint i64 %address.local608, 99, !dbg !22
  %address.local619 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local617, i64 %address.local618, !dbg !22
  store half %1916, ptr addrspace(1) %address.local619, align 2, !dbg !22
  %address.local620 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local605, !dbg !22
  %address.local621 = or disjoint i64 %address.local608, 112, !dbg !22
  %address.local622 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local621, !dbg !22
  store half %1910, ptr addrspace(1) %address.local622, align 2, !dbg !22
  %address.local623 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local605, !dbg !22
  %address.local624 = or disjoint i64 %address.local608, 113, !dbg !22
  %address.local625 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local623, i64 %address.local624, !dbg !22
  store half %1909, ptr addrspace(1) %address.local625, align 2, !dbg !22
  %address.local626 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local605, !dbg !22
  %address.local627 = or disjoint i64 %address.local608, 114, !dbg !22
  %address.local628 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local626, i64 %address.local627, !dbg !22
  store half %1908, ptr addrspace(1) %address.local628, align 2, !dbg !22
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local605, !dbg !22
  %address.local630 = or i64 %address.local607, 115, !dbg !22
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local629, i64 %address.local630, !dbg !22
  store half %1907, ptr addrspace(1) %address.local631, align 2, !dbg !22
  %address.local632 = shl nsw i64 %address.local603, 7, !dbg !22
  %address.local633 = add i64 %address.local632, 8192, !dbg !22
  %address.local634 = add nuw nsw i64 %1922, %address.local633, !dbg !22
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local634, !dbg !22
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local635, i64 %address.local609, !dbg !22
  store half %1901, ptr addrspace(1) %address.local636, align 2, !dbg !22
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local634, !dbg !22
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local637, i64 %address.local612, !dbg !22
  store half %1900, ptr addrspace(1) %address.local638, align 2, !dbg !22
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local634, !dbg !22
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local639, i64 %address.local615, !dbg !22
  store half %1899, ptr addrspace(1) %address.local640, align 2, !dbg !22
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local634, !dbg !22
  %address.local642 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local641, i64 %address.local618, !dbg !22
  store half %1898, ptr addrspace(1) %address.local642, align 2, !dbg !22
  %address.local643 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local634, !dbg !22
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local621, !dbg !22
  store half %1892, ptr addrspace(1) %address.local644, align 2, !dbg !22
  %address.local645 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local634, !dbg !22
  %address.local646 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local645, i64 %address.local624, !dbg !22
  store half %1891, ptr addrspace(1) %address.local646, align 2, !dbg !22
  %address.local647 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local634, !dbg !22
  %address.local648 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local647, i64 %address.local627, !dbg !22
  store half %1890, ptr addrspace(1) %address.local648, align 2, !dbg !22
  %address.local649 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local634, !dbg !22
  %address.local650 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local649, i64 %address.local630, !dbg !22
  store half %1889, ptr addrspace(1) %address.local650, align 2, !dbg !22
  br label %1923, !dbg !22

1923:                                             ; preds = %1883, %1882
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

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #6

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #7

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
!llvm.module.flags = !{!34}

!0 = !DIFile(filename: "test_input_spidD.mlir", directory: "3rd/deepgengraph/test")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!10 = !DILocation(line: 69, column: 26, scope: !4)
!12 = !DILocation(line: 46, column: 17, scope: !4)
!13 = !DILocation(line: 66, column: 22, scope: !4)
!14 = !DILocation(line: 68, column: 24, scope: !4)
!15 = !DILocation(line: 73, column: 19, scope: !4)
!16 = !DILocation(line: 93, column: 23, scope: !4)
!17 = !DILocation(line: 56, column: 24, scope: !4)
!18 = !DILocation(line: 67, column: 22, scope: !4)
!19 = !DILocation(line: 92, column: 20, scope: !4)
!20 = !DILocation(line: 97, column: 23, scope: !4)
!21 = !DILocation(line: 98, column: 19, scope: !4)
!22 = !DILocation(line: 99, column: 9, scope: !4)
!23 = !DILocation(line: 59, column: 17, scope: !4)
!24 = !DILocation(line: 75, column: 24, scope: !4)
!25 = !DILocation(line: 76, column: 23, scope: !4)
!26 = !DILocation(line: 77, column: 26, scope: !4)
!27 = !DILocation(line: 84, column: 21, scope: !4)
!28 = !DILocation(line: 85, column: 22, scope: !4)
!29 = !DILocation(line: 86, column: 20, scope: !4)
!30 = !DILocation(line: 87, column: 16, scope: !4)
!31 = !DILocation(line: 94, column: 20, scope: !4)
!32 = !DILocation(line: 70, column: 23, scope: !4)
!33 = !{i32 256, i32 1, i32 1}
!34 = !{i32 2, !"Debug Info Version", i32 3}
!35 = !{i32 0, i32 256}
