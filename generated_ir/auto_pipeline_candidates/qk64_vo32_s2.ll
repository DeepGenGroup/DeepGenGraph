; ModuleID = '3rd/deepgengraph/test/test_input_spidD.mlir'
source_filename = "3rd/deepgengraph/test/test_input_spidD.mlir"

@shm_4 = addrspace(3) global [8192 x half] undef, align 16
@shm_3 = addrspace(3) global [8192 x half] undef, align 16
@shm_2 = addrspace(3) global [8192 x half] undef, align 16
@shm_1 = addrspace(3) global [8192 x half] undef, align 16
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

29:                                               ; preds = %707, %4
  %30 = phi i64 [ %1458, %707 ], [ 0, %4 ], !dbg !17
  %31 = phi [2 x <8 x float>] [ %1445, %707 ], [ zeroinitializer, %4 ], !dbg !17
  %32 = phi [2 x <1 x float>] [ %1067, %707 ], [ zeroinitializer, %4 ], !dbg !17
  %33 = phi [2 x <8 x float>] [ %1449, %707 ], [ zeroinitializer, %4 ], !dbg !17
  %34 = phi [2 x <8 x float>] [ %1453, %707 ], [ zeroinitializer, %4 ], !dbg !17
  %35 = phi [2 x <8 x float>] [ %1457, %707 ], [ zeroinitializer, %4 ], !dbg !17
  %.not = icmp sgt i64 %30, %8, !dbg !17
  br i1 %.not, label %1459, label %36, !dbg !17

36:                                               ; preds = %29
  %37 = shl nsw i64 %30, 7, !dbg !17
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !13, !range !35, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !14
  %address.local141 = shl nuw nsw i64 %address.local, 3, !dbg !13
  %address.local142 = and i64 %address.local141, 1536, !dbg !13
  %address.local143 = getelementptr [2 x i8], ptr addrspace(3) @shm_1, i64 %address.local142, !dbg !13
  %address.local144 = ptrtoint ptr addrspace(3) %address.local143 to i64, !dbg !13
  %address.local145 = trunc i64 %address.local144 to i32, !dbg !13
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !13
  %39 = zext i32 %38 to i64, !dbg !13
  %40 = inttoptr i64 %39 to ptr addrspace(3), !dbg !13
  %address.local146 = shl nuw nsw i64 %address.local, 7, !dbg !13
  %address.local147 = and i64 %address.local146, 16256, !dbg !13
  %address.local148 = lshr i64 %address.local, 4, !dbg !13
  %address.local149 = and i64 %address.local148, 8, !dbg !13
  %address.local150 = or disjoint i64 %address.local147, %address.local149, !dbg !13
  %address.local151 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %address.local150, !dbg !13
  %41 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !13
  %42 = ptrtoint ptr addrspace(1) %address.local151 to i64, !dbg !13
  %43 = sub i64 %42, %41, !dbg !13
  %dma.buffer.offset1134 = trunc i64 %43 to i32, !dbg !13
  %44 = trunc i64 %41 to i32, !dbg !13
  %45 = insertelement <4 x i32> poison, i32 %44, i64 0, !dbg !13
  %46 = lshr i64 %41, 32, !dbg !13
  %47 = trunc i64 %46 to i32, !dbg !13
  %48 = insertelement <4 x i32> %45, i32 %47, i64 1, !dbg !13
  %49 = insertelement <4 x i32> %48, i32 33554432, i64 2, !dbg !13
  %dma.buffer.rsrc1135 = insertelement <4 x i32> %49, i32 131072, i64 3, !dbg !13
  %50 = ptrtoint ptr addrspace(3) %40 to i32, !dbg !13
  %dma.buffer.lds1136 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !13
  %51 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1136, i32 %dma.buffer.offset1134, <4 x i32> %dma.buffer.rsrc1135) #10, !dbg !13
  %address.local152 = or disjoint i64 %address.local, 256, !dbg !13
  %address.local153 = shl nuw nsw i64 %address.local152, 3, !dbg !13
  %address.local154 = and i64 %address.local153, 3584, !dbg !13
  %address.local155 = getelementptr [2 x i8], ptr addrspace(3) @shm_1, i64 %address.local154, !dbg !13
  %address.local156 = ptrtoint ptr addrspace(3) %address.local155 to i64, !dbg !13
  %address.local157 = trunc i64 %address.local156 to i32, !dbg !13
  %52 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local157), !dbg !13
  %53 = zext i32 %52 to i64, !dbg !13
  %54 = inttoptr i64 %53 to ptr addrspace(3), !dbg !13
  %address.local158 = shl nuw nsw i64 %address.local, 7, !dbg !13
  %address.local159 = and i64 %address.local158, 16256, !dbg !13
  %address.local160 = lshr i64 %address.local152, 4, !dbg !13
  %address.local161 = and i64 %address.local160, 24, !dbg !13
  %address.local162 = or disjoint i64 %address.local159, %address.local161, !dbg !13
  %address.local163 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %address.local162, !dbg !13
  %55 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !13
  %56 = ptrtoint ptr addrspace(1) %address.local163 to i64, !dbg !13
  %57 = sub i64 %56, %55, !dbg !13
  %dma.buffer.offset1131 = trunc i64 %57 to i32, !dbg !13
  %58 = trunc i64 %55 to i32, !dbg !13
  %59 = insertelement <4 x i32> poison, i32 %58, i64 0, !dbg !13
  %60 = lshr i64 %55, 32, !dbg !13
  %61 = trunc i64 %60 to i32, !dbg !13
  %62 = insertelement <4 x i32> %59, i32 %61, i64 1, !dbg !13
  %63 = insertelement <4 x i32> %62, i32 33554432, i64 2, !dbg !13
  %dma.buffer.rsrc1132 = insertelement <4 x i32> %63, i32 131072, i64 3, !dbg !13
  %64 = ptrtoint ptr addrspace(3) %54 to i32, !dbg !13
  %dma.buffer.lds1133 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %64), !dbg !13
  %65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1133, i32 %dma.buffer.offset1131, <4 x i32> %dma.buffer.rsrc1132) #10, !dbg !13
  %address.local164 = or disjoint i64 %address.local, 512, !dbg !13
  %address.local165 = shl nuw nsw i64 %address.local164, 3, !dbg !13
  %address.local166 = and i64 %address.local165, 5632, !dbg !13
  %address.local167 = getelementptr [2 x i8], ptr addrspace(3) @shm_1, i64 %address.local166, !dbg !13
  %address.local168 = ptrtoint ptr addrspace(3) %address.local167 to i64, !dbg !13
  %address.local169 = trunc i64 %address.local168 to i32, !dbg !13
  %66 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local169), !dbg !13
  %67 = zext i32 %66 to i64, !dbg !13
  %68 = inttoptr i64 %67 to ptr addrspace(3), !dbg !13
  %address.local170 = shl nuw nsw i64 %address.local, 7, !dbg !13
  %address.local171 = and i64 %address.local170, 16256, !dbg !13
  %address.local172 = lshr i64 %address.local164, 4, !dbg !13
  %address.local173 = and i64 %address.local172, 40, !dbg !13
  %address.local174 = or disjoint i64 %address.local171, %address.local173, !dbg !13
  %address.local175 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %address.local174, !dbg !13
  %69 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !13
  %70 = ptrtoint ptr addrspace(1) %address.local175 to i64, !dbg !13
  %71 = sub i64 %70, %69, !dbg !13
  %dma.buffer.offset1128 = trunc i64 %71 to i32, !dbg !13
  %72 = trunc i64 %69 to i32, !dbg !13
  %73 = insertelement <4 x i32> poison, i32 %72, i64 0, !dbg !13
  %74 = lshr i64 %69, 32, !dbg !13
  %75 = trunc i64 %74 to i32, !dbg !13
  %76 = insertelement <4 x i32> %73, i32 %75, i64 1, !dbg !13
  %77 = insertelement <4 x i32> %76, i32 33554432, i64 2, !dbg !13
  %dma.buffer.rsrc1129 = insertelement <4 x i32> %77, i32 131072, i64 3, !dbg !13
  %78 = ptrtoint ptr addrspace(3) %68 to i32, !dbg !13
  %dma.buffer.lds1130 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %78), !dbg !13
  %79 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1130, i32 %dma.buffer.offset1128, <4 x i32> %dma.buffer.rsrc1129) #10, !dbg !13
  %address.local176 = or disjoint i64 %address.local, 768, !dbg !13
  %address.local177 = shl nuw nsw i64 %address.local176, 3, !dbg !13
  %address.local178 = and i64 %address.local177, 7680, !dbg !13
  %address.local179 = getelementptr [2 x i8], ptr addrspace(3) @shm_1, i64 %address.local178, !dbg !13
  %address.local180 = ptrtoint ptr addrspace(3) %address.local179 to i64, !dbg !13
  %address.local181 = trunc i64 %address.local180 to i32, !dbg !13
  %80 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local181), !dbg !13
  %81 = zext i32 %80 to i64, !dbg !13
  %82 = inttoptr i64 %81 to ptr addrspace(3), !dbg !13
  %address.local182 = shl nuw nsw i64 %address.local, 7, !dbg !13
  %address.local183 = and i64 %address.local182, 16256, !dbg !13
  %address.local184 = lshr i64 %address.local176, 4, !dbg !13
  %address.local185 = and i64 %address.local184, 56, !dbg !13
  %address.local186 = or disjoint i64 %address.local183, %address.local185, !dbg !13
  %address.local187 = getelementptr [2 x i8], ptr addrspace(1) %28, i64 %address.local186, !dbg !13
  %83 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !13
  %84 = ptrtoint ptr addrspace(1) %address.local187 to i64, !dbg !13
  %85 = sub i64 %84, %83, !dbg !13
  %dma.buffer.offset1125 = trunc i64 %85 to i32, !dbg !13
  %86 = trunc i64 %83 to i32, !dbg !13
  %87 = insertelement <4 x i32> poison, i32 %86, i64 0, !dbg !13
  %88 = lshr i64 %83, 32, !dbg !13
  %89 = trunc i64 %88 to i32, !dbg !13
  %90 = insertelement <4 x i32> %87, i32 %89, i64 1, !dbg !13
  %91 = insertelement <4 x i32> %90, i32 33554432, i64 2, !dbg !13
  %dma.buffer.rsrc1126 = insertelement <4 x i32> %91, i32 131072, i64 3, !dbg !13
  %92 = ptrtoint ptr addrspace(3) %82 to i32, !dbg !13
  %dma.buffer.lds1127 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %92), !dbg !13
  %93 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1127, i32 %dma.buffer.offset1125, <4 x i32> %dma.buffer.rsrc1126) #10, !dbg !13
  %94 = lshr i64 %30, 12, !dbg !18
  %95 = add nsw i64 %94, %10, !dbg !18
  %96 = lshr i64 %30, 5, !dbg !18
  %97 = and i64 %96, 127, !dbg !18
  %98 = shl i64 %95, 19, !dbg !18
  %99 = and i64 %98, 16252928, !dbg !18
  %100 = shl i64 %96, 19, !dbg !18
  %101 = shl i64 %30, 14, !dbg !18
  %102 = sub i64 %101, %100, !dbg !18
  %103 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %99, !dbg !18
  %104 = getelementptr [2 x i8], ptr addrspace(1) %103, i64 %97, !dbg !18
  %105 = getelementptr [2 x i8], ptr addrspace(1) %104, i64 %102, !dbg !18
  %address.local188 = getelementptr [2 x i8], ptr addrspace(3) @shm_3, i64 %address.local142, !dbg !18
  %address.local189 = ptrtoint ptr addrspace(3) %address.local188 to i64, !dbg !18
  %address.local190 = trunc i64 %address.local189 to i32, !dbg !18
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local190), !dbg !18
  %107 = zext i32 %106 to i64, !dbg !18
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !18
  %address.local191 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local150, !dbg !18
  %109 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %110 = ptrtoint ptr addrspace(1) %address.local191 to i64, !dbg !18
  %111 = sub i64 %110, %109, !dbg !18
  %dma.buffer.offset1122 = trunc i64 %111 to i32, !dbg !18
  %112 = trunc i64 %109 to i32, !dbg !18
  %113 = insertelement <4 x i32> poison, i32 %112, i64 0, !dbg !18
  %114 = lshr i64 %109, 32, !dbg !18
  %115 = trunc i64 %114 to i32, !dbg !18
  %116 = insertelement <4 x i32> %113, i32 %115, i64 1, !dbg !18
  %117 = insertelement <4 x i32> %116, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1123 = insertelement <4 x i32> %117, i32 131072, i64 3, !dbg !18
  %118 = ptrtoint ptr addrspace(3) %108 to i32, !dbg !18
  %dma.buffer.lds1124 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !18
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1124, i32 %dma.buffer.offset1122, <4 x i32> %dma.buffer.rsrc1123) #10, !dbg !18
  %address.local192 = getelementptr [2 x i8], ptr addrspace(3) @shm_3, i64 %address.local154, !dbg !18
  %address.local193 = ptrtoint ptr addrspace(3) %address.local192 to i64, !dbg !18
  %address.local194 = trunc i64 %address.local193 to i32, !dbg !18
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local194), !dbg !18
  %121 = zext i32 %120 to i64, !dbg !18
  %122 = inttoptr i64 %121 to ptr addrspace(3), !dbg !18
  %address.local195 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local162, !dbg !18
  %123 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %124 = ptrtoint ptr addrspace(1) %address.local195 to i64, !dbg !18
  %125 = sub i64 %124, %123, !dbg !18
  %dma.buffer.offset1119 = trunc i64 %125 to i32, !dbg !18
  %126 = trunc i64 %123 to i32, !dbg !18
  %127 = insertelement <4 x i32> poison, i32 %126, i64 0, !dbg !18
  %128 = lshr i64 %123, 32, !dbg !18
  %129 = trunc i64 %128 to i32, !dbg !18
  %130 = insertelement <4 x i32> %127, i32 %129, i64 1, !dbg !18
  %131 = insertelement <4 x i32> %130, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1120 = insertelement <4 x i32> %131, i32 131072, i64 3, !dbg !18
  %132 = ptrtoint ptr addrspace(3) %122 to i32, !dbg !18
  %dma.buffer.lds1121 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !18
  %133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1121, i32 %dma.buffer.offset1119, <4 x i32> %dma.buffer.rsrc1120) #10, !dbg !18
  %address.local196 = getelementptr [2 x i8], ptr addrspace(3) @shm_3, i64 %address.local166, !dbg !18
  %address.local197 = ptrtoint ptr addrspace(3) %address.local196 to i64, !dbg !18
  %address.local198 = trunc i64 %address.local197 to i32, !dbg !18
  %134 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local198), !dbg !18
  %135 = zext i32 %134 to i64, !dbg !18
  %136 = inttoptr i64 %135 to ptr addrspace(3), !dbg !18
  %address.local199 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local174, !dbg !18
  %137 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %138 = ptrtoint ptr addrspace(1) %address.local199 to i64, !dbg !18
  %139 = sub i64 %138, %137, !dbg !18
  %dma.buffer.offset1116 = trunc i64 %139 to i32, !dbg !18
  %140 = trunc i64 %137 to i32, !dbg !18
  %141 = insertelement <4 x i32> poison, i32 %140, i64 0, !dbg !18
  %142 = lshr i64 %137, 32, !dbg !18
  %143 = trunc i64 %142 to i32, !dbg !18
  %144 = insertelement <4 x i32> %141, i32 %143, i64 1, !dbg !18
  %145 = insertelement <4 x i32> %144, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1117 = insertelement <4 x i32> %145, i32 131072, i64 3, !dbg !18
  %146 = ptrtoint ptr addrspace(3) %136 to i32, !dbg !18
  %dma.buffer.lds1118 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %146), !dbg !18
  %147 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1118, i32 %dma.buffer.offset1116, <4 x i32> %dma.buffer.rsrc1117) #10, !dbg !18
  %address.local200 = getelementptr [2 x i8], ptr addrspace(3) @shm_3, i64 %address.local178, !dbg !18
  %address.local201 = ptrtoint ptr addrspace(3) %address.local200 to i64, !dbg !18
  %address.local202 = trunc i64 %address.local201 to i32, !dbg !18
  %148 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local202), !dbg !18
  %149 = zext i32 %148 to i64, !dbg !18
  %150 = inttoptr i64 %149 to ptr addrspace(3), !dbg !18
  %address.local203 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local186, !dbg !18
  %151 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %152 = ptrtoint ptr addrspace(1) %address.local203 to i64, !dbg !18
  %153 = sub i64 %152, %151, !dbg !18
  %dma.buffer.offset1113 = trunc i64 %153 to i32, !dbg !18
  %154 = trunc i64 %151 to i32, !dbg !18
  %155 = insertelement <4 x i32> poison, i32 %154, i64 0, !dbg !18
  %156 = lshr i64 %151, 32, !dbg !18
  %157 = trunc i64 %156 to i32, !dbg !18
  %158 = insertelement <4 x i32> %155, i32 %157, i64 1, !dbg !18
  %159 = insertelement <4 x i32> %158, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1114 = insertelement <4 x i32> %159, i32 131072, i64 3, !dbg !18
  %160 = ptrtoint ptr addrspace(3) %150 to i32, !dbg !18
  %dma.buffer.lds1115 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %160), !dbg !18
  %161 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1115, i32 %dma.buffer.offset1113, <4 x i32> %dma.buffer.rsrc1114) #10, !dbg !18
  br label %162, !dbg !23

162:                                              ; preds = %329, %36
  %163 = phi i64 [ %522, %329 ], [ 0, %36 ], !dbg !23
  %164 = phi [2 x <32 x float>] [ %521, %329 ], [ zeroinitializer, %36 ], !dbg !23
  %165 = icmp samesign ult i64 %163, 2, !dbg !23
  br i1 %165, label %166, label %523, !dbg !23

166:                                              ; preds = %162
  %167 = icmp eq i64 %163, 0, !dbg !23
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %168 = icmp eq i64 %163, 0, !dbg !23
  br i1 %168, label %169, label %329, !dbg !23

169:                                              ; preds = %166
  %170 = shl nsw i64 %8, 8, !dbg !13
  %171 = shl nsw i64 %10, 13, !dbg !13
  %172 = add nsw i64 %170, %171, !dbg !13
  %173 = shl nsw i64 %6, 18, !dbg !13
  %174 = add nsw i64 %172, %173, !dbg !13
  %175 = or disjoint i64 %174, 1, !dbg !13
  %176 = icmp slt i64 %174, 0, !dbg !13
  %177 = sub nuw nsw i64 -2, %174, !dbg !13
  %178 = select i1 %176, i64 %177, i64 %175, !dbg !13
  %179 = sdiv i64 %178, 8192, !dbg !13
  %.lobit69 = ashr i64 %174, 63, !dbg !13
  %180 = xor i64 %179, %.lobit69, !dbg !13
  %181 = shl nsw i64 %10, 12, !dbg !13
  %182 = shl nsw i64 %6, 17, !dbg !13
  %183 = shl i64 %180, 19, !dbg !13
  %184 = and i64 %183, 16252928, !dbg !13
  %185 = shl nsw i64 %8, 14, !dbg !13
  %186 = and i64 %185, 507904, !dbg !13
  %187 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %184, !dbg !13
  %188 = getelementptr [2 x i8], ptr addrspace(1) %187, i64 %186, !dbg !13
  %189 = getelementptr i8, ptr addrspace(1) %188, i64 128, !dbg !13
  %address.thread204 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !13, !range !35, !frisk.register_lifetimes !1
  %address.local205 = zext nneg i32 %address.thread204 to i64, !dbg !14
  %address.local206 = shl nuw nsw i64 %address.local205, 3, !dbg !13
  %address.local207 = and i64 %address.local206, 1536, !dbg !13
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) @shm_2, i64 %address.local207, !dbg !13
  %address.local209 = ptrtoint ptr addrspace(3) %address.local208 to i64, !dbg !13
  %address.local210 = trunc i64 %address.local209 to i32, !dbg !13
  %190 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local210), !dbg !13
  %191 = zext i32 %190 to i64, !dbg !13
  %192 = inttoptr i64 %191 to ptr addrspace(3), !dbg !13
  %address.local211 = shl nuw nsw i64 %address.local205, 7, !dbg !13
  %address.local212 = and i64 %address.local211, 16256, !dbg !13
  %address.local213 = lshr i64 %address.local205, 4, !dbg !13
  %address.local214 = and i64 %address.local213, 8, !dbg !13
  %address.local215 = or disjoint i64 %address.local212, %address.local214, !dbg !13
  %address.local216 = getelementptr [2 x i8], ptr addrspace(1) %189, i64 %address.local215, !dbg !13
  %193 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !13
  %194 = ptrtoint ptr addrspace(1) %address.local216 to i64, !dbg !13
  %195 = sub i64 %194, %193, !dbg !13
  %dma.buffer.offset1086 = trunc i64 %195 to i32, !dbg !13
  %196 = trunc i64 %193 to i32, !dbg !13
  %197 = insertelement <4 x i32> poison, i32 %196, i64 0, !dbg !13
  %198 = lshr i64 %193, 32, !dbg !13
  %199 = trunc i64 %198 to i32, !dbg !13
  %200 = insertelement <4 x i32> %197, i32 %199, i64 1, !dbg !13
  %201 = insertelement <4 x i32> %200, i32 33554432, i64 2, !dbg !13
  %dma.buffer.rsrc1087 = insertelement <4 x i32> %201, i32 131072, i64 3, !dbg !13
  %202 = ptrtoint ptr addrspace(3) %192 to i32, !dbg !13
  %dma.buffer.lds1088 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %202), !dbg !13
  %203 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1088, i32 %dma.buffer.offset1086, <4 x i32> %dma.buffer.rsrc1087) #10, !dbg !13
  %address.local217 = or disjoint i64 %address.local205, 256, !dbg !13
  %address.local218 = shl nuw nsw i64 %address.local217, 3, !dbg !13
  %address.local219 = and i64 %address.local218, 3584, !dbg !13
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) @shm_2, i64 %address.local219, !dbg !13
  %address.local221 = ptrtoint ptr addrspace(3) %address.local220 to i64, !dbg !13
  %address.local222 = trunc i64 %address.local221 to i32, !dbg !13
  %204 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local222), !dbg !13
  %205 = zext i32 %204 to i64, !dbg !13
  %206 = inttoptr i64 %205 to ptr addrspace(3), !dbg !13
  %address.local223 = shl nuw nsw i64 %address.local205, 7, !dbg !13
  %address.local224 = and i64 %address.local223, 16256, !dbg !13
  %address.local225 = lshr i64 %address.local217, 4, !dbg !13
  %address.local226 = and i64 %address.local225, 24, !dbg !13
  %address.local227 = or disjoint i64 %address.local224, %address.local226, !dbg !13
  %address.local228 = getelementptr [2 x i8], ptr addrspace(1) %189, i64 %address.local227, !dbg !13
  %207 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !13
  %208 = ptrtoint ptr addrspace(1) %address.local228 to i64, !dbg !13
  %209 = sub i64 %208, %207, !dbg !13
  %dma.buffer.offset1083 = trunc i64 %209 to i32, !dbg !13
  %210 = trunc i64 %207 to i32, !dbg !13
  %211 = insertelement <4 x i32> poison, i32 %210, i64 0, !dbg !13
  %212 = lshr i64 %207, 32, !dbg !13
  %213 = trunc i64 %212 to i32, !dbg !13
  %214 = insertelement <4 x i32> %211, i32 %213, i64 1, !dbg !13
  %215 = insertelement <4 x i32> %214, i32 33554432, i64 2, !dbg !13
  %dma.buffer.rsrc1084 = insertelement <4 x i32> %215, i32 131072, i64 3, !dbg !13
  %216 = ptrtoint ptr addrspace(3) %206 to i32, !dbg !13
  %dma.buffer.lds1085 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %216), !dbg !13
  %217 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1085, i32 %dma.buffer.offset1083, <4 x i32> %dma.buffer.rsrc1084) #10, !dbg !13
  %address.local229 = or disjoint i64 %address.local205, 512, !dbg !13
  %address.local230 = shl nuw nsw i64 %address.local229, 3, !dbg !13
  %address.local231 = and i64 %address.local230, 5632, !dbg !13
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) @shm_2, i64 %address.local231, !dbg !13
  %address.local233 = ptrtoint ptr addrspace(3) %address.local232 to i64, !dbg !13
  %address.local234 = trunc i64 %address.local233 to i32, !dbg !13
  %218 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local234), !dbg !13
  %219 = zext i32 %218 to i64, !dbg !13
  %220 = inttoptr i64 %219 to ptr addrspace(3), !dbg !13
  %address.local235 = shl nuw nsw i64 %address.local205, 7, !dbg !13
  %address.local236 = and i64 %address.local235, 16256, !dbg !13
  %address.local237 = lshr i64 %address.local229, 4, !dbg !13
  %address.local238 = and i64 %address.local237, 40, !dbg !13
  %address.local239 = or disjoint i64 %address.local236, %address.local238, !dbg !13
  %address.local240 = getelementptr [2 x i8], ptr addrspace(1) %189, i64 %address.local239, !dbg !13
  %221 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !13
  %222 = ptrtoint ptr addrspace(1) %address.local240 to i64, !dbg !13
  %223 = sub i64 %222, %221, !dbg !13
  %dma.buffer.offset1080 = trunc i64 %223 to i32, !dbg !13
  %224 = trunc i64 %221 to i32, !dbg !13
  %225 = insertelement <4 x i32> poison, i32 %224, i64 0, !dbg !13
  %226 = lshr i64 %221, 32, !dbg !13
  %227 = trunc i64 %226 to i32, !dbg !13
  %228 = insertelement <4 x i32> %225, i32 %227, i64 1, !dbg !13
  %229 = insertelement <4 x i32> %228, i32 33554432, i64 2, !dbg !13
  %dma.buffer.rsrc1081 = insertelement <4 x i32> %229, i32 131072, i64 3, !dbg !13
  %230 = ptrtoint ptr addrspace(3) %220 to i32, !dbg !13
  %dma.buffer.lds1082 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %230), !dbg !13
  %231 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1082, i32 %dma.buffer.offset1080, <4 x i32> %dma.buffer.rsrc1081) #10, !dbg !13
  %address.local241 = or disjoint i64 %address.local205, 768, !dbg !13
  %address.local242 = shl nuw nsw i64 %address.local241, 3, !dbg !13
  %address.local243 = and i64 %address.local242, 7680, !dbg !13
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) @shm_2, i64 %address.local243, !dbg !13
  %address.local245 = ptrtoint ptr addrspace(3) %address.local244 to i64, !dbg !13
  %address.local246 = trunc i64 %address.local245 to i32, !dbg !13
  %232 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local246), !dbg !13
  %233 = zext i32 %232 to i64, !dbg !13
  %234 = inttoptr i64 %233 to ptr addrspace(3), !dbg !13
  %address.local247 = shl nuw nsw i64 %address.local205, 7, !dbg !13
  %address.local248 = and i64 %address.local247, 16256, !dbg !13
  %address.local249 = lshr i64 %address.local241, 4, !dbg !13
  %address.local250 = and i64 %address.local249, 56, !dbg !13
  %address.local251 = or disjoint i64 %address.local248, %address.local250, !dbg !13
  %address.local252 = getelementptr [2 x i8], ptr addrspace(1) %189, i64 %address.local251, !dbg !13
  %235 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !13
  %236 = ptrtoint ptr addrspace(1) %address.local252 to i64, !dbg !13
  %237 = sub i64 %236, %235, !dbg !13
  %dma.buffer.offset1077 = trunc i64 %237 to i32, !dbg !13
  %238 = trunc i64 %235 to i32, !dbg !13
  %239 = insertelement <4 x i32> poison, i32 %238, i64 0, !dbg !13
  %240 = lshr i64 %235, 32, !dbg !13
  %241 = trunc i64 %240 to i32, !dbg !13
  %242 = insertelement <4 x i32> %239, i32 %241, i64 1, !dbg !13
  %243 = insertelement <4 x i32> %242, i32 33554432, i64 2, !dbg !13
  %dma.buffer.rsrc1078 = insertelement <4 x i32> %243, i32 131072, i64 3, !dbg !13
  %244 = ptrtoint ptr addrspace(3) %234 to i32, !dbg !13
  %dma.buffer.lds1079 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %244), !dbg !13
  %245 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1079, i32 %dma.buffer.offset1077, <4 x i32> %dma.buffer.rsrc1078) #10, !dbg !13
  %246 = add i64 %30, %181, !dbg !18
  %247 = add i64 %246, %182, !dbg !18
  %248 = add i64 %247, 2048, !dbg !18
  %249 = icmp slt i64 %248, 0, !dbg !18
  %250 = sub i64 -2049, %247, !dbg !18
  %251 = select i1 %249, i64 %250, i64 %248, !dbg !18
  %252 = sdiv i64 %251, 4096, !dbg !18
  %.lobit70 = ashr i64 %248, 63, !dbg !18
  %253 = xor i64 %252, %.lobit70, !dbg !18
  %254 = shl nsw i64 %10, 7, !dbg !18
  %255 = shl nsw i64 %6, 12, !dbg !18
  %256 = add nsw i64 %254, %255, !dbg !18
  %257 = add nsw i64 %256, %96, !dbg !18
  %258 = add nsw i64 %257, 64, !dbg !18
  %259 = icmp slt i64 %257, -64, !dbg !18
  %260 = sub nsw i64 -65, %257, !dbg !18
  %261 = select i1 %259, i64 %260, i64 %258, !dbg !18
  %262 = sdiv i64 %261, 128, !dbg !18
  %263 = sext i1 %259 to i64, !dbg !18
  %264 = xor i64 %262, %263, !dbg !18
  %265 = shl nsw i64 %264, 7, !dbg !18
  %266 = sub i64 %257, %265, !dbg !18
  %267 = shl i64 %253, 19, !dbg !18
  %268 = and i64 %267, 16252928, !dbg !18
  %269 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %268, !dbg !18
  %270 = getelementptr [2 x i8], ptr addrspace(1) %269, i64 %266, !dbg !18
  %271 = getelementptr i8, ptr addrspace(1) %270, i64 128, !dbg !18
  %272 = getelementptr [2 x i8], ptr addrspace(1) %271, i64 %102, !dbg !18
  %address.local253 = getelementptr [2 x i8], ptr addrspace(3) @shm_4, i64 %address.local207, !dbg !18
  %address.local254 = ptrtoint ptr addrspace(3) %address.local253 to i64, !dbg !18
  %address.local255 = trunc i64 %address.local254 to i32, !dbg !18
  %273 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local255), !dbg !18
  %274 = zext i32 %273 to i64, !dbg !18
  %275 = inttoptr i64 %274 to ptr addrspace(3), !dbg !18
  %address.local256 = getelementptr [2 x i8], ptr addrspace(1) %272, i64 %address.local215, !dbg !18
  %276 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %277 = ptrtoint ptr addrspace(1) %address.local256 to i64, !dbg !18
  %278 = sub i64 %277, %276, !dbg !18
  %dma.buffer.offset1074 = trunc i64 %278 to i32, !dbg !18
  %279 = trunc i64 %276 to i32, !dbg !18
  %280 = insertelement <4 x i32> poison, i32 %279, i64 0, !dbg !18
  %281 = lshr i64 %276, 32, !dbg !18
  %282 = trunc i64 %281 to i32, !dbg !18
  %283 = insertelement <4 x i32> %280, i32 %282, i64 1, !dbg !18
  %284 = insertelement <4 x i32> %283, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1075 = insertelement <4 x i32> %284, i32 131072, i64 3, !dbg !18
  %285 = ptrtoint ptr addrspace(3) %275 to i32, !dbg !18
  %dma.buffer.lds1076 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %285), !dbg !18
  %286 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1076, i32 %dma.buffer.offset1074, <4 x i32> %dma.buffer.rsrc1075) #10, !dbg !18
  %address.local257 = getelementptr [2 x i8], ptr addrspace(3) @shm_4, i64 %address.local219, !dbg !18
  %address.local258 = ptrtoint ptr addrspace(3) %address.local257 to i64, !dbg !18
  %address.local259 = trunc i64 %address.local258 to i32, !dbg !18
  %287 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local259), !dbg !18
  %288 = zext i32 %287 to i64, !dbg !18
  %289 = inttoptr i64 %288 to ptr addrspace(3), !dbg !18
  %address.local260 = getelementptr [2 x i8], ptr addrspace(1) %272, i64 %address.local227, !dbg !18
  %290 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %291 = ptrtoint ptr addrspace(1) %address.local260 to i64, !dbg !18
  %292 = sub i64 %291, %290, !dbg !18
  %dma.buffer.offset1071 = trunc i64 %292 to i32, !dbg !18
  %293 = trunc i64 %290 to i32, !dbg !18
  %294 = insertelement <4 x i32> poison, i32 %293, i64 0, !dbg !18
  %295 = lshr i64 %290, 32, !dbg !18
  %296 = trunc i64 %295 to i32, !dbg !18
  %297 = insertelement <4 x i32> %294, i32 %296, i64 1, !dbg !18
  %298 = insertelement <4 x i32> %297, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1072 = insertelement <4 x i32> %298, i32 131072, i64 3, !dbg !18
  %299 = ptrtoint ptr addrspace(3) %289 to i32, !dbg !18
  %dma.buffer.lds1073 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %299), !dbg !18
  %300 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1073, i32 %dma.buffer.offset1071, <4 x i32> %dma.buffer.rsrc1072) #10, !dbg !18
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) @shm_4, i64 %address.local231, !dbg !18
  %address.local262 = ptrtoint ptr addrspace(3) %address.local261 to i64, !dbg !18
  %address.local263 = trunc i64 %address.local262 to i32, !dbg !18
  %301 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local263), !dbg !18
  %302 = zext i32 %301 to i64, !dbg !18
  %303 = inttoptr i64 %302 to ptr addrspace(3), !dbg !18
  %address.local264 = getelementptr [2 x i8], ptr addrspace(1) %272, i64 %address.local239, !dbg !18
  %304 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %305 = ptrtoint ptr addrspace(1) %address.local264 to i64, !dbg !18
  %306 = sub i64 %305, %304, !dbg !18
  %dma.buffer.offset1068 = trunc i64 %306 to i32, !dbg !18
  %307 = trunc i64 %304 to i32, !dbg !18
  %308 = insertelement <4 x i32> poison, i32 %307, i64 0, !dbg !18
  %309 = lshr i64 %304, 32, !dbg !18
  %310 = trunc i64 %309 to i32, !dbg !18
  %311 = insertelement <4 x i32> %308, i32 %310, i64 1, !dbg !18
  %312 = insertelement <4 x i32> %311, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1069 = insertelement <4 x i32> %312, i32 131072, i64 3, !dbg !18
  %313 = ptrtoint ptr addrspace(3) %303 to i32, !dbg !18
  %dma.buffer.lds1070 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %313), !dbg !18
  %314 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1070, i32 %dma.buffer.offset1068, <4 x i32> %dma.buffer.rsrc1069) #10, !dbg !18
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) @shm_4, i64 %address.local243, !dbg !18
  %address.local266 = ptrtoint ptr addrspace(3) %address.local265 to i64, !dbg !18
  %address.local267 = trunc i64 %address.local266 to i32, !dbg !18
  %315 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local267), !dbg !18
  %316 = zext i32 %315 to i64, !dbg !18
  %317 = inttoptr i64 %316 to ptr addrspace(3), !dbg !18
  %address.local268 = getelementptr [2 x i8], ptr addrspace(1) %272, i64 %address.local251, !dbg !18
  %318 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %319 = ptrtoint ptr addrspace(1) %address.local268 to i64, !dbg !18
  %320 = sub i64 %319, %318, !dbg !18
  %dma.buffer.offset = trunc i64 %320 to i32, !dbg !18
  %321 = trunc i64 %318 to i32, !dbg !18
  %322 = insertelement <4 x i32> poison, i32 %321, i64 0, !dbg !18
  %323 = lshr i64 %318, 32, !dbg !18
  %324 = trunc i64 %323 to i32, !dbg !18
  %325 = insertelement <4 x i32> %322, i32 %324, i64 1, !dbg !18
  %326 = insertelement <4 x i32> %325, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %326, i32 131072, i64 3, !dbg !18
  %327 = ptrtoint ptr addrspace(3) %317 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %327), !dbg !18
  %328 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  br label %329, !dbg !23

329:                                              ; preds = %169, %166
  %330 = select i1 %167, ptr addrspace(3) @shm_1, ptr addrspace(3) @shm_2, !dbg !23
  %address.thread269 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !14, !range !35, !frisk.register_lifetimes !1
  %address.local270 = lshr i32 %address.thread269, 2, !dbg !14
  %address.local271 = and i32 %address.local270, 4, !dbg !14
  %address.local272 = shl nuw nsw i32 %address.thread269, 1, !dbg !14
  %address.local273 = and i32 %address.local272, 384, !dbg !14
  %address.local274 = or disjoint i32 %address.local271, %address.local273, !dbg !14
  %address.local275 = shl nuw nsw i32 %address.thread269, 3, !dbg !14
  %address.local276 = and i32 %address.local275, 120, !dbg !14
  %address.local277 = or disjoint i32 %address.local274, %address.local276, !dbg !14
  %address.local278 = shl nuw nsw i32 %address.thread269, 5, !dbg !14
  %address.local279 = and i32 %address.local278, 1024, !dbg !14
  %address.local280 = or disjoint i32 %address.local277, %address.local279, !dbg !14
  %address.local281 = zext nneg i32 %address.local280 to i64, !dbg !14
  %address.local282 = getelementptr [2 x i8], ptr addrspace(3) %330, i64 %address.local281, !dbg !14
  %331 = load <4 x half>, ptr addrspace(3) %address.local282, align 2, !dbg !14
  %332 = extractvalue [2 x <32 x float>] %164, 0, !dbg !32
  %333 = shufflevector <32 x float> %332, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %334 = fmul <4 x half> %331, splat (half 0xH3015), !dbg !14
  %address.local283 = getelementptr [2 x i8], ptr addrspace(3) %330, i64 %address.local281, !dbg !14
  %address.local284 = getelementptr i8, ptr addrspace(3) %address.local283, i64 4096, !dbg !14
  %335 = load <4 x half>, ptr addrspace(3) %address.local284, align 2, !dbg !14
  %336 = fmul <4 x half> %335, splat (half 0xH3015), !dbg !14
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) %330, i64 %address.local281, !dbg !14
  %address.local286 = getelementptr i8, ptr addrspace(3) %address.local285, i64 8192, !dbg !14
  %337 = load <4 x half>, ptr addrspace(3) %address.local286, align 2, !dbg !14
  %338 = fmul <4 x half> %337, splat (half 0xH3015), !dbg !14
  %339 = shufflevector <32 x float> %332, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %340 = shufflevector <32 x float> %332, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !32
  %341 = shufflevector <32 x float> %332, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !32
  %342 = shufflevector <32 x float> %332, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !32
  %343 = shufflevector <32 x float> %332, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !32
  %344 = shufflevector <32 x float> %332, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !32
  %345 = shufflevector <32 x float> %332, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !32
  %346 = extractvalue [2 x <32 x float>] %164, 1, !dbg !32
  %347 = shufflevector <32 x float> %346, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %348 = shufflevector <32 x float> %346, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %349 = shufflevector <32 x float> %346, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !32
  %350 = shufflevector <32 x float> %346, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !32
  %351 = shufflevector <32 x float> %346, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !32
  %352 = shufflevector <32 x float> %346, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !32
  %353 = shufflevector <32 x float> %346, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !32
  %354 = shufflevector <32 x float> %346, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !32
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) %330, i64 %address.local281, !dbg !14
  %address.local288 = getelementptr i8, ptr addrspace(3) %address.local287, i64 12288, !dbg !14
  %355 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !14
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) %330, i64 %address.local281, !dbg !14
  %address.local290 = getelementptr i8, ptr addrspace(3) %address.local289, i64 1024, !dbg !14
  %356 = load <4 x half>, ptr addrspace(3) %address.local290, align 2, !dbg !14
  %357 = fmul <4 x half> %355, splat (half 0xH3015), !dbg !14
  %358 = fmul <4 x half> %356, splat (half 0xH3015), !dbg !14
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) %330, i64 %address.local281, !dbg !14
  %address.local292 = getelementptr i8, ptr addrspace(3) %address.local291, i64 5120, !dbg !14
  %359 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !14
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) %330, i64 %address.local281, !dbg !14
  %address.local294 = getelementptr i8, ptr addrspace(3) %address.local293, i64 9216, !dbg !14
  %360 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !14
  %361 = fmul <4 x half> %359, splat (half 0xH3015), !dbg !14
  %362 = fmul <4 x half> %360, splat (half 0xH3015), !dbg !14
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) %330, i64 %address.local281, !dbg !14
  %address.local296 = getelementptr i8, ptr addrspace(3) %address.local295, i64 13312, !dbg !14
  %363 = load <4 x half>, ptr addrspace(3) %address.local296, align 2, !dbg !14
  %364 = select i1 %167, ptr addrspace(3) @shm_3, ptr addrspace(3) @shm_4, !dbg !23
  %address.local297 = lshr i32 %address.thread269, 2, !dbg !10
  %address.local298 = and i32 %address.local297, 4, !dbg !10
  %address.local299 = shl nuw nsw i32 %address.thread269, 1, !dbg !10
  %address.local300 = and i32 %address.local299, 24, !dbg !10
  %address.local301 = or disjoint i32 %address.local298, %address.local300, !dbg !10
  %address.local302 = shl nuw nsw i32 %address.thread269, 5, !dbg !10
  %address.local303 = and i32 %address.local302, 1024, !dbg !10
  %address.local304 = or disjoint i32 %address.local301, %address.local303, !dbg !10
  %address.local305 = shl nuw nsw i32 %address.thread269, 7, !dbg !10
  %address.local306 = and i32 %address.local305, 384, !dbg !10
  %address.local307 = or disjoint i32 %address.local304, %address.local306, !dbg !10
  %address.local308 = zext nneg i32 %address.local307 to i64, !dbg !10
  %address.local309 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local308, !dbg !10
  %365 = load <4 x half>, ptr addrspace(3) %address.local309, align 2, !dbg !10
  %366 = fmul <4 x half> %363, splat (half 0xH3015), !dbg !14
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %365, <4 x float> %333) #9, !dbg !10
  %address.local310 = lshr i32 %address.thread269, 2, !dbg !10
  %address.local311 = and i32 %address.local310, 4, !dbg !10
  %address.local312 = shl nuw nsw i32 %address.thread269, 1, !dbg !10
  %address.local313 = and i32 %address.local312, 24, !dbg !10
  %address.local314 = or disjoint i32 %address.local311, %address.local313, !dbg !10
  %address.local315 = shl nuw nsw i32 %address.thread269, 5, !dbg !10
  %address.local316 = and i32 %address.local315, 1024, !dbg !10
  %address.local317 = or disjoint i32 %address.local314, %address.local316, !dbg !10
  %address.local318 = shl nuw nsw i32 %address.thread269, 7, !dbg !10
  %address.local319 = and i32 %address.local318, 384, !dbg !10
  %address.local320 = or disjoint i32 %address.local317, %address.local319, !dbg !10
  %address.local321 = zext nneg i32 %address.local320 to i64, !dbg !10
  %address.local322 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local321, !dbg !10
  %address.local323 = getelementptr i8, ptr addrspace(3) %address.local322, i64 4096, !dbg !10
  %368 = load <4 x half>, ptr addrspace(3) %address.local323, align 2, !dbg !10
  %address.local324 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local321, !dbg !10
  %address.local325 = getelementptr i8, ptr addrspace(3) %address.local324, i64 8192, !dbg !10
  %369 = load <4 x half>, ptr addrspace(3) %address.local325, align 2, !dbg !10
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %368, <4 x float> %367) #9, !dbg !10
  %address.local326 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local321, !dbg !10
  %address.local327 = getelementptr i8, ptr addrspace(3) %address.local326, i64 12288, !dbg !10
  %371 = load <4 x half>, ptr addrspace(3) %address.local327, align 2, !dbg !10
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %338, <4 x half> %369, <4 x float> %370) #9, !dbg !10
  %address.local328 = zext nneg i32 %address.thread269 to i64, !dbg !14
  %address.local329 = shl nuw nsw i64 %address.local328, 7, !dbg !10
  %address.local330 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local329, !dbg !10
  %address.local331 = lshr i64 %address.local328, 2, !dbg !10
  %address.local332 = mul nsw i64 %address.local331, -512, !dbg !10
  %address.local333 = getelementptr [2 x i8], ptr addrspace(3) %address.local330, i64 %address.local332, !dbg !10
  %address.local334 = shl nuw nsw i64 %address.local328, 1, !dbg !10
  %address.local335 = and i64 %address.local334, 24, !dbg !10
  %address.local336 = getelementptr [2 x i8], ptr addrspace(3) %address.local333, i64 %address.local335, !dbg !10
  %address.local337 = lshr i64 %address.local328, 2, !dbg !14
  %address.local338 = and i64 %address.local337, 12, !dbg !14
  %address.local339 = getelementptr [2 x i8], ptr addrspace(3) %address.local336, i64 %address.local338, !dbg !10
  %address.local340 = and i64 %address.local328, 63, !dbg !14
  %address.local341 = lshr i64 %address.local340, 5, !dbg !14
  %address.local342 = mul nuw nsw i64 %address.local341, 1016, !dbg !14
  %address.local343 = getelementptr [2 x i8], ptr addrspace(3) %address.local339, i64 %address.local342, !dbg !10
  %address.local344 = getelementptr i8, ptr addrspace(3) %address.local343, i64 64, !dbg !10
  %373 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !10
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %371, <4 x float> %372) #9, !dbg !10
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %373, <4 x float> %339) #9, !dbg !10
  %address.local345 = lshr i32 %address.thread269, 2, !dbg !10
  %address.local346 = and i32 %address.local345, 4, !dbg !10
  %address.local347 = shl nuw nsw i32 %address.thread269, 1, !dbg !10
  %address.local348 = and i32 %address.local347, 24, !dbg !10
  %address.local349 = or disjoint i32 %address.local346, %address.local348, !dbg !10
  %address.local350 = shl nuw nsw i32 %address.thread269, 5, !dbg !10
  %address.local351 = and i32 %address.local350, 1024, !dbg !10
  %address.local352 = or disjoint i32 %address.local349, %address.local351, !dbg !10
  %address.local353 = shl nuw nsw i32 %address.thread269, 7, !dbg !10
  %address.local354 = and i32 %address.local353, 384, !dbg !10
  %address.local355 = or disjoint i32 %address.local352, %address.local354, !dbg !10
  %address.local356 = zext nneg i32 %address.local355 to i64, !dbg !10
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local356, !dbg !10
  %address.local358 = getelementptr i8, ptr addrspace(3) %address.local357, i64 4160, !dbg !10
  %376 = load <4 x half>, ptr addrspace(3) %address.local358, align 2, !dbg !10
  %address.local359 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local356, !dbg !10
  %address.local360 = getelementptr i8, ptr addrspace(3) %address.local359, i64 8256, !dbg !10
  %377 = load <4 x half>, ptr addrspace(3) %address.local360, align 2, !dbg !10
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %376, <4 x float> %375) #9, !dbg !10
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local356, !dbg !10
  %address.local362 = getelementptr i8, ptr addrspace(3) %address.local361, i64 12352, !dbg !10
  %379 = load <4 x half>, ptr addrspace(3) %address.local362, align 2, !dbg !10
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %338, <4 x half> %377, <4 x float> %378) #9, !dbg !10
  %address.local363 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local329, !dbg !10
  %address.local364 = getelementptr [2 x i8], ptr addrspace(3) %address.local363, i64 %address.local332, !dbg !10
  %address.local365 = getelementptr [2 x i8], ptr addrspace(3) %address.local364, i64 %address.local335, !dbg !10
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) %address.local365, i64 %address.local338, !dbg !10
  %address.local367 = getelementptr [2 x i8], ptr addrspace(3) %address.local366, i64 %address.local342, !dbg !10
  %address.local368 = getelementptr i8, ptr addrspace(3) %address.local367, i64 128, !dbg !10
  %381 = load <4 x half>, ptr addrspace(3) %address.local368, align 2, !dbg !10
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %379, <4 x float> %380) #9, !dbg !10
  %383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %381, <4 x float> %340) #9, !dbg !10
  %address.local369 = lshr i32 %address.thread269, 2, !dbg !10
  %address.local370 = and i32 %address.local369, 4, !dbg !10
  %address.local371 = shl nuw nsw i32 %address.thread269, 1, !dbg !10
  %address.local372 = and i32 %address.local371, 24, !dbg !10
  %address.local373 = or disjoint i32 %address.local370, %address.local372, !dbg !10
  %address.local374 = shl nuw nsw i32 %address.thread269, 5, !dbg !10
  %address.local375 = and i32 %address.local374, 1024, !dbg !10
  %address.local376 = or disjoint i32 %address.local373, %address.local375, !dbg !10
  %address.local377 = shl nuw nsw i32 %address.thread269, 7, !dbg !10
  %address.local378 = and i32 %address.local377, 384, !dbg !10
  %address.local379 = or disjoint i32 %address.local376, %address.local378, !dbg !10
  %address.local380 = zext nneg i32 %address.local379 to i64, !dbg !10
  %address.local381 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local380, !dbg !10
  %address.local382 = getelementptr i8, ptr addrspace(3) %address.local381, i64 4224, !dbg !10
  %384 = load <4 x half>, ptr addrspace(3) %address.local382, align 2, !dbg !10
  %address.local383 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local380, !dbg !10
  %address.local384 = getelementptr i8, ptr addrspace(3) %address.local383, i64 8320, !dbg !10
  %385 = load <4 x half>, ptr addrspace(3) %address.local384, align 2, !dbg !10
  %386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %384, <4 x float> %383) #9, !dbg !10
  %address.local385 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local380, !dbg !10
  %address.local386 = getelementptr i8, ptr addrspace(3) %address.local385, i64 12416, !dbg !10
  %387 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !10
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %338, <4 x half> %385, <4 x float> %386) #9, !dbg !10
  %address.local387 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local329, !dbg !10
  %address.local388 = getelementptr [2 x i8], ptr addrspace(3) %address.local387, i64 %address.local332, !dbg !10
  %address.local389 = getelementptr [2 x i8], ptr addrspace(3) %address.local388, i64 %address.local335, !dbg !10
  %address.local390 = getelementptr [2 x i8], ptr addrspace(3) %address.local389, i64 %address.local338, !dbg !10
  %address.local391 = getelementptr [2 x i8], ptr addrspace(3) %address.local390, i64 %address.local342, !dbg !10
  %address.local392 = getelementptr i8, ptr addrspace(3) %address.local391, i64 192, !dbg !10
  %389 = load <4 x half>, ptr addrspace(3) %address.local392, align 2, !dbg !10
  %390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %387, <4 x float> %388) #9, !dbg !10
  %391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %389, <4 x float> %341) #9, !dbg !10
  %392 = shufflevector <4 x float> %390, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %address.local393 = lshr i32 %address.thread269, 2, !dbg !10
  %address.local394 = and i32 %address.local393, 4, !dbg !10
  %address.local395 = shl nuw nsw i32 %address.thread269, 1, !dbg !10
  %address.local396 = and i32 %address.local395, 24, !dbg !10
  %address.local397 = or disjoint i32 %address.local394, %address.local396, !dbg !10
  %address.local398 = shl nuw nsw i32 %address.thread269, 5, !dbg !10
  %address.local399 = and i32 %address.local398, 1024, !dbg !10
  %address.local400 = or disjoint i32 %address.local397, %address.local399, !dbg !10
  %address.local401 = shl nuw nsw i32 %address.thread269, 7, !dbg !10
  %address.local402 = and i32 %address.local401, 384, !dbg !10
  %address.local403 = or disjoint i32 %address.local400, %address.local402, !dbg !10
  %address.local404 = zext nneg i32 %address.local403 to i64, !dbg !10
  %address.local405 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local404, !dbg !10
  %address.local406 = getelementptr i8, ptr addrspace(3) %address.local405, i64 4288, !dbg !10
  %393 = load <4 x half>, ptr addrspace(3) %address.local406, align 2, !dbg !10
  %address.local407 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local404, !dbg !10
  %address.local408 = getelementptr i8, ptr addrspace(3) %address.local407, i64 8384, !dbg !10
  %394 = load <4 x half>, ptr addrspace(3) %address.local408, align 2, !dbg !10
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %393, <4 x float> %391) #9, !dbg !10
  %address.local409 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local404, !dbg !10
  %address.local410 = getelementptr i8, ptr addrspace(3) %address.local409, i64 12480, !dbg !10
  %396 = load <4 x half>, ptr addrspace(3) %address.local410, align 2, !dbg !10
  %397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %338, <4 x half> %394, <4 x float> %395) #9, !dbg !10
  %address.local411 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local308, !dbg !10
  %address.local412 = getelementptr i8, ptr addrspace(3) %address.local411, i64 1024, !dbg !10
  %398 = load <4 x half>, ptr addrspace(3) %address.local412, align 2, !dbg !10
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %396, <4 x float> %397) #9, !dbg !10
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %398, <4 x float> %342) #9, !dbg !10
  %401 = shufflevector <4 x float> %399, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %address.local413 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local321, !dbg !10
  %address.local414 = getelementptr i8, ptr addrspace(3) %address.local413, i64 5120, !dbg !10
  %402 = load <4 x half>, ptr addrspace(3) %address.local414, align 2, !dbg !10
  %address.local415 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local321, !dbg !10
  %address.local416 = getelementptr i8, ptr addrspace(3) %address.local415, i64 9216, !dbg !10
  %403 = load <4 x half>, ptr addrspace(3) %address.local416, align 2, !dbg !10
  %404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %402, <4 x float> %400) #9, !dbg !10
  %address.local417 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local321, !dbg !10
  %address.local418 = getelementptr i8, ptr addrspace(3) %address.local417, i64 13312, !dbg !10
  %405 = load <4 x half>, ptr addrspace(3) %address.local418, align 2, !dbg !10
  %406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %338, <4 x half> %403, <4 x float> %404) #9, !dbg !10
  %address.local419 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local329, !dbg !10
  %address.local420 = getelementptr [2 x i8], ptr addrspace(3) %address.local419, i64 %address.local332, !dbg !10
  %address.local421 = getelementptr [2 x i8], ptr addrspace(3) %address.local420, i64 %address.local335, !dbg !10
  %address.local422 = getelementptr [2 x i8], ptr addrspace(3) %address.local421, i64 %address.local338, !dbg !10
  %address.local423 = getelementptr [2 x i8], ptr addrspace(3) %address.local422, i64 %address.local342, !dbg !10
  %address.local424 = getelementptr i8, ptr addrspace(3) %address.local423, i64 1088, !dbg !10
  %407 = load <4 x half>, ptr addrspace(3) %address.local424, align 2, !dbg !10
  %408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %405, <4 x float> %406) #9, !dbg !10
  %409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %407, <4 x float> %343) #9, !dbg !10
  %410 = shufflevector <4 x float> %408, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %address.local425 = lshr i32 %address.thread269, 2, !dbg !10
  %address.local426 = and i32 %address.local425, 4, !dbg !10
  %address.local427 = shl nuw nsw i32 %address.thread269, 1, !dbg !10
  %address.local428 = and i32 %address.local427, 24, !dbg !10
  %address.local429 = or disjoint i32 %address.local426, %address.local428, !dbg !10
  %address.local430 = shl nuw nsw i32 %address.thread269, 5, !dbg !10
  %address.local431 = and i32 %address.local430, 1024, !dbg !10
  %address.local432 = or disjoint i32 %address.local429, %address.local431, !dbg !10
  %address.local433 = shl nuw nsw i32 %address.thread269, 7, !dbg !10
  %address.local434 = and i32 %address.local433, 384, !dbg !10
  %address.local435 = or disjoint i32 %address.local432, %address.local434, !dbg !10
  %address.local436 = or disjoint i32 %address.local435, 512, !dbg !10
  %address.local437 = zext nneg i32 %address.local436 to i64, !dbg !10
  %address.local438 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local437, !dbg !10
  %address.local439 = getelementptr i8, ptr addrspace(3) %address.local438, i64 4160, !dbg !10
  %411 = load <4 x half>, ptr addrspace(3) %address.local439, align 2, !dbg !10
  %address.local440 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local437, !dbg !10
  %address.local441 = getelementptr i8, ptr addrspace(3) %address.local440, i64 8256, !dbg !10
  %412 = load <4 x half>, ptr addrspace(3) %address.local441, align 2, !dbg !10
  %413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %411, <4 x float> %409) #9, !dbg !10
  %address.local442 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local437, !dbg !10
  %address.local443 = getelementptr i8, ptr addrspace(3) %address.local442, i64 12352, !dbg !10
  %414 = load <4 x half>, ptr addrspace(3) %address.local443, align 2, !dbg !10
  %415 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %338, <4 x half> %412, <4 x float> %413) #9, !dbg !10
  %address.local444 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local329, !dbg !10
  %address.local445 = getelementptr [2 x i8], ptr addrspace(3) %address.local444, i64 %address.local332, !dbg !10
  %address.local446 = getelementptr [2 x i8], ptr addrspace(3) %address.local445, i64 %address.local335, !dbg !10
  %address.local447 = getelementptr [2 x i8], ptr addrspace(3) %address.local446, i64 %address.local338, !dbg !10
  %address.local448 = getelementptr [2 x i8], ptr addrspace(3) %address.local447, i64 %address.local342, !dbg !10
  %address.local449 = getelementptr i8, ptr addrspace(3) %address.local448, i64 1152, !dbg !10
  %416 = load <4 x half>, ptr addrspace(3) %address.local449, align 2, !dbg !10
  %417 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %414, <4 x float> %415) #9, !dbg !10
  %418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %416, <4 x float> %344) #9, !dbg !10
  %419 = shufflevector <4 x float> %417, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %address.local450 = lshr i32 %address.thread269, 2, !dbg !10
  %address.local451 = and i32 %address.local450, 4, !dbg !10
  %address.local452 = shl nuw nsw i32 %address.thread269, 1, !dbg !10
  %address.local453 = and i32 %address.local452, 24, !dbg !10
  %address.local454 = or disjoint i32 %address.local451, %address.local453, !dbg !10
  %address.local455 = shl nuw nsw i32 %address.thread269, 5, !dbg !10
  %address.local456 = and i32 %address.local455, 1024, !dbg !10
  %address.local457 = or disjoint i32 %address.local454, %address.local456, !dbg !10
  %address.local458 = shl nuw nsw i32 %address.thread269, 7, !dbg !10
  %address.local459 = and i32 %address.local458, 384, !dbg !10
  %address.local460 = or disjoint i32 %address.local457, %address.local459, !dbg !10
  %address.local461 = or disjoint i32 %address.local460, 512, !dbg !10
  %address.local462 = zext nneg i32 %address.local461 to i64, !dbg !10
  %address.local463 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local462, !dbg !10
  %address.local464 = getelementptr i8, ptr addrspace(3) %address.local463, i64 4224, !dbg !10
  %420 = load <4 x half>, ptr addrspace(3) %address.local464, align 2, !dbg !10
  %address.local465 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local462, !dbg !10
  %address.local466 = getelementptr i8, ptr addrspace(3) %address.local465, i64 8320, !dbg !10
  %421 = load <4 x half>, ptr addrspace(3) %address.local466, align 2, !dbg !10
  %422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %420, <4 x float> %418) #9, !dbg !10
  %address.local467 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local462, !dbg !10
  %address.local468 = getelementptr i8, ptr addrspace(3) %address.local467, i64 12416, !dbg !10
  %423 = load <4 x half>, ptr addrspace(3) %address.local468, align 2, !dbg !10
  %424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %338, <4 x half> %421, <4 x float> %422) #9, !dbg !10
  %address.local469 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local329, !dbg !10
  %address.local470 = getelementptr [2 x i8], ptr addrspace(3) %address.local469, i64 %address.local332, !dbg !10
  %address.local471 = getelementptr [2 x i8], ptr addrspace(3) %address.local470, i64 %address.local335, !dbg !10
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) %address.local471, i64 %address.local338, !dbg !10
  %address.local473 = getelementptr [2 x i8], ptr addrspace(3) %address.local472, i64 %address.local342, !dbg !10
  %address.local474 = getelementptr i8, ptr addrspace(3) %address.local473, i64 1216, !dbg !10
  %425 = load <4 x half>, ptr addrspace(3) %address.local474, align 2, !dbg !10
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %423, <4 x float> %424) #9, !dbg !10
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %425, <4 x float> %345) #9, !dbg !10
  %428 = shufflevector <4 x float> %426, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %address.local475 = lshr i32 %address.thread269, 2, !dbg !10
  %address.local476 = and i32 %address.local475, 4, !dbg !10
  %address.local477 = shl nuw nsw i32 %address.thread269, 1, !dbg !10
  %address.local478 = and i32 %address.local477, 24, !dbg !10
  %address.local479 = or disjoint i32 %address.local476, %address.local478, !dbg !10
  %address.local480 = shl nuw nsw i32 %address.thread269, 5, !dbg !10
  %address.local481 = and i32 %address.local480, 1024, !dbg !10
  %address.local482 = or disjoint i32 %address.local479, %address.local481, !dbg !10
  %address.local483 = shl nuw nsw i32 %address.thread269, 7, !dbg !10
  %address.local484 = and i32 %address.local483, 384, !dbg !10
  %address.local485 = or disjoint i32 %address.local482, %address.local484, !dbg !10
  %address.local486 = or disjoint i32 %address.local485, 512, !dbg !10
  %address.local487 = zext nneg i32 %address.local486 to i64, !dbg !10
  %address.local488 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local487, !dbg !10
  %address.local489 = getelementptr i8, ptr addrspace(3) %address.local488, i64 4288, !dbg !10
  %429 = load <4 x half>, ptr addrspace(3) %address.local489, align 2, !dbg !10
  %address.local490 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local487, !dbg !10
  %address.local491 = getelementptr i8, ptr addrspace(3) %address.local490, i64 8384, !dbg !10
  %430 = load <4 x half>, ptr addrspace(3) %address.local491, align 2, !dbg !10
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %429, <4 x float> %427) #9, !dbg !10
  %address.local492 = getelementptr [2 x i8], ptr addrspace(3) %364, i64 %address.local487, !dbg !10
  %address.local493 = getelementptr i8, ptr addrspace(3) %address.local492, i64 12480, !dbg !10
  %432 = load <4 x half>, ptr addrspace(3) %address.local493, align 2, !dbg !10
  %433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %338, <4 x half> %430, <4 x float> %431) #9, !dbg !10
  %434 = load <4 x half>, ptr addrspace(3) %address.local309, align 2, !dbg !10
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %432, <4 x float> %433) #9, !dbg !10
  %436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %434, <4 x float> %347) #9, !dbg !10
  %437 = shufflevector <4 x float> %435, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %438 = load <4 x half>, ptr addrspace(3) %address.local323, align 2, !dbg !10
  %439 = load <4 x half>, ptr addrspace(3) %address.local325, align 2, !dbg !10
  %440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %438, <4 x float> %436) #9, !dbg !10
  %441 = load <4 x half>, ptr addrspace(3) %address.local327, align 2, !dbg !10
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %439, <4 x float> %440) #9, !dbg !10
  %443 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !10
  %444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %441, <4 x float> %442) #9, !dbg !10
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %443, <4 x float> %348) #9, !dbg !10
  %446 = load <4 x half>, ptr addrspace(3) %address.local358, align 2, !dbg !10
  %447 = load <4 x half>, ptr addrspace(3) %address.local360, align 2, !dbg !10
  %448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %446, <4 x float> %445) #9, !dbg !10
  %449 = load <4 x half>, ptr addrspace(3) %address.local362, align 2, !dbg !10
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %447, <4 x float> %448) #9, !dbg !10
  %451 = load <4 x half>, ptr addrspace(3) %address.local368, align 2, !dbg !10
  %452 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %449, <4 x float> %450) #9, !dbg !10
  %453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %451, <4 x float> %349) #9, !dbg !10
  %454 = load <4 x half>, ptr addrspace(3) %address.local382, align 2, !dbg !10
  %455 = load <4 x half>, ptr addrspace(3) %address.local384, align 2, !dbg !10
  %456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %454, <4 x float> %453) #9, !dbg !10
  %457 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !10
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %455, <4 x float> %456) #9, !dbg !10
  %459 = load <4 x half>, ptr addrspace(3) %address.local392, align 2, !dbg !10
  %460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %457, <4 x float> %458) #9, !dbg !10
  %461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %459, <4 x float> %350) #9, !dbg !10
  %462 = shufflevector <4 x float> %460, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %463 = load <4 x half>, ptr addrspace(3) %address.local406, align 2, !dbg !10
  %464 = load <4 x half>, ptr addrspace(3) %address.local408, align 2, !dbg !10
  %465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %463, <4 x float> %461) #9, !dbg !10
  %466 = load <4 x half>, ptr addrspace(3) %address.local410, align 2, !dbg !10
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %464, <4 x float> %465) #9, !dbg !10
  %468 = load <4 x half>, ptr addrspace(3) %address.local412, align 2, !dbg !10
  %469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %466, <4 x float> %467) #9, !dbg !10
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %468, <4 x float> %351) #9, !dbg !10
  %471 = shufflevector <4 x float> %469, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %472 = load <4 x half>, ptr addrspace(3) %address.local414, align 2, !dbg !10
  %473 = load <4 x half>, ptr addrspace(3) %address.local416, align 2, !dbg !10
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %472, <4 x float> %470) #9, !dbg !10
  %475 = load <4 x half>, ptr addrspace(3) %address.local418, align 2, !dbg !10
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %473, <4 x float> %474) #9, !dbg !10
  %477 = load <4 x half>, ptr addrspace(3) %address.local424, align 2, !dbg !10
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %475, <4 x float> %476) #9, !dbg !10
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %477, <4 x float> %352) #9, !dbg !10
  %480 = shufflevector <4 x float> %478, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %481 = load <4 x half>, ptr addrspace(3) %address.local439, align 2, !dbg !10
  %482 = load <4 x half>, ptr addrspace(3) %address.local441, align 2, !dbg !10
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %481, <4 x float> %479) #9, !dbg !10
  %484 = load <4 x half>, ptr addrspace(3) %address.local443, align 2, !dbg !10
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %482, <4 x float> %483) #9, !dbg !10
  %486 = load <4 x half>, ptr addrspace(3) %address.local449, align 2, !dbg !10
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %484, <4 x float> %485) #9, !dbg !10
  %488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %486, <4 x float> %353) #9, !dbg !10
  %489 = shufflevector <4 x float> %487, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %490 = load <4 x half>, ptr addrspace(3) %address.local464, align 2, !dbg !10
  %491 = load <4 x half>, ptr addrspace(3) %address.local466, align 2, !dbg !10
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %490, <4 x float> %488) #9, !dbg !10
  %493 = load <4 x half>, ptr addrspace(3) %address.local468, align 2, !dbg !10
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %491, <4 x float> %492) #9, !dbg !10
  %495 = load <4 x half>, ptr addrspace(3) %address.local474, align 2, !dbg !10
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %493, <4 x float> %494) #9, !dbg !10
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %495, <4 x float> %354) #9, !dbg !10
  %498 = shufflevector <4 x float> %496, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %499 = load <4 x half>, ptr addrspace(3) %address.local489, align 2, !dbg !10
  %500 = load <4 x half>, ptr addrspace(3) %address.local491, align 2, !dbg !10
  %501 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %499, <4 x float> %497) #9, !dbg !10
  %502 = load <4 x half>, ptr addrspace(3) %address.local493, align 2, !dbg !10
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %500, <4 x float> %501) #9, !dbg !10
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %366, <4 x half> %502, <4 x float> %503) #9, !dbg !10
  %505 = shufflevector <4 x float> %504, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !32
  %506 = shufflevector <4 x float> %374, <4 x float> %382, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %507 = shufflevector <32 x float> %506, <32 x float> %392, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %508 = shufflevector <32 x float> %507, <32 x float> %401, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %509 = shufflevector <32 x float> %508, <32 x float> %410, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %510 = shufflevector <32 x float> %509, <32 x float> %419, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %511 = shufflevector <32 x float> %510, <32 x float> %428, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %512 = shufflevector <32 x float> %511, <32 x float> %437, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !23
  %513 = insertvalue [2 x <32 x float>] poison, <32 x float> %512, 0, !dbg !23
  %514 = shufflevector <4 x float> %444, <4 x float> %452, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %515 = shufflevector <32 x float> %514, <32 x float> %462, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %516 = shufflevector <32 x float> %515, <32 x float> %471, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %517 = shufflevector <32 x float> %516, <32 x float> %480, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %518 = shufflevector <32 x float> %517, <32 x float> %489, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %519 = shufflevector <32 x float> %518, <32 x float> %498, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !23
  %520 = shufflevector <32 x float> %519, <32 x float> %505, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !23
  %521 = insertvalue [2 x <32 x float>] %513, <32 x float> %520, 1, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %522 = add nuw nsw i64 %163, 1, !dbg !23
  br label %162, !dbg !23

523:                                              ; preds = %162
  %524 = add nsw i64 %96, %10, !dbg !19
  %525 = shl i64 %524, 19, !dbg !19
  %526 = and i64 %525, 16252928, !dbg !19
  %527 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %526, !dbg !19
  %528 = getelementptr [2 x i8], ptr addrspace(1) %527, i64 %102, !dbg !19
  %address.thread494 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !19, !range !35, !frisk.register_lifetimes !1
  %address.local495 = zext nneg i32 %address.thread494 to i64, !dbg !14
  %address.local496 = shl nuw nsw i64 %address.local495, 3, !dbg !13
  %address.local497 = and i64 %address.local496, 1536, !dbg !19
  %address.local498 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local497, !dbg !19
  %address.local499 = ptrtoint ptr addrspace(3) %address.local498 to i64, !dbg !19
  %address.local500 = trunc i64 %address.local499 to i32, !dbg !19
  %529 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local500), !dbg !19
  %530 = zext i32 %529 to i64, !dbg !19
  %531 = inttoptr i64 %530 to ptr addrspace(3), !dbg !19
  %address.local501 = shl nuw nsw i64 %address.local495, 5, !dbg !19
  %address.local502 = and i64 %address.local501, 8064, !dbg !19
  %address.local503 = shl nuw nsw i64 %address.local495, 3, !dbg !19
  %address.local504 = and i64 %address.local503, 24, !dbg !19
  %address.local505 = or disjoint i64 %address.local502, %address.local504, !dbg !19
  %address.local506 = getelementptr [2 x i8], ptr addrspace(1) %528, i64 %address.local505, !dbg !19
  %532 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %533 = ptrtoint ptr addrspace(1) %address.local506 to i64, !dbg !19
  %534 = sub i64 %533, %532, !dbg !19
  %dma.buffer.offset1110 = trunc i64 %534 to i32, !dbg !19
  %535 = trunc i64 %532 to i32, !dbg !19
  %536 = insertelement <4 x i32> poison, i32 %535, i64 0, !dbg !19
  %537 = lshr i64 %532, 32, !dbg !19
  %538 = trunc i64 %537 to i32, !dbg !19
  %539 = insertelement <4 x i32> %536, i32 %538, i64 1, !dbg !19
  %540 = insertelement <4 x i32> %539, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1111 = insertelement <4 x i32> %540, i32 131072, i64 3, !dbg !19
  %541 = ptrtoint ptr addrspace(3) %531 to i32, !dbg !19
  %dma.buffer.lds1112 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %541), !dbg !19
  %542 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1112, i32 %dma.buffer.offset1110, <4 x i32> %dma.buffer.rsrc1111) #10, !dbg !19
  %address.local507 = or disjoint i64 %address.local495, 256, !dbg !13
  %address.local508 = shl nuw nsw i64 %address.local507, 3, !dbg !13
  %address.local509 = and i64 %address.local508, 3584, !dbg !19
  %address.local510 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local509, !dbg !19
  %address.local511 = ptrtoint ptr addrspace(3) %address.local510 to i64, !dbg !19
  %address.local512 = trunc i64 %address.local511 to i32, !dbg !19
  %543 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local512), !dbg !19
  %544 = zext i32 %543 to i64, !dbg !19
  %545 = inttoptr i64 %544 to ptr addrspace(3), !dbg !19
  %address.local513 = shl nuw nsw i64 %address.local507, 5, !dbg !19
  %address.local514 = and i64 %address.local513, 16256, !dbg !19
  %address.local515 = shl nuw nsw i64 %address.local495, 3, !dbg !19
  %address.local516 = and i64 %address.local515, 24, !dbg !19
  %address.local517 = or disjoint i64 %address.local514, %address.local516, !dbg !19
  %address.local518 = getelementptr [2 x i8], ptr addrspace(1) %528, i64 %address.local517, !dbg !19
  %546 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %547 = ptrtoint ptr addrspace(1) %address.local518 to i64, !dbg !19
  %548 = sub i64 %547, %546, !dbg !19
  %dma.buffer.offset1107 = trunc i64 %548 to i32, !dbg !19
  %549 = trunc i64 %546 to i32, !dbg !19
  %550 = insertelement <4 x i32> poison, i32 %549, i64 0, !dbg !19
  %551 = lshr i64 %546, 32, !dbg !19
  %552 = trunc i64 %551 to i32, !dbg !19
  %553 = insertelement <4 x i32> %550, i32 %552, i64 1, !dbg !19
  %554 = insertelement <4 x i32> %553, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1108 = insertelement <4 x i32> %554, i32 131072, i64 3, !dbg !19
  %555 = ptrtoint ptr addrspace(3) %545 to i32, !dbg !19
  %dma.buffer.lds1109 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %555), !dbg !19
  %556 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1109, i32 %dma.buffer.offset1107, <4 x i32> %dma.buffer.rsrc1108) #10, !dbg !19
  %557 = or disjoint i64 %37, 127, !dbg !15
  %558 = icmp uge i64 %11, %557, !dbg !15
  %559 = and i1 %26, %558, !dbg !15
  br i1 %559, label %560, label %561, !dbg !15

560:                                              ; preds = %523
  br label %705, !dbg !15

561:                                              ; preds = %523
  %562 = add nsw i64 %11, %22, !dbg !15
  %563 = shl nuw nsw i64 %23, 4, !dbg !15
  %564 = add nsw i64 %562, %563, !dbg !15
  %565 = and i32 %21, 240, !dbg !15
  %index.compact = zext nneg i32 %565 to i64, !dbg !15
  %566 = sub nsw i64 %564, %index.compact, !dbg !15
  %567 = and i64 %22, 48, !dbg !15
  %568 = or disjoint i64 %37, %567, !dbg !15
  %569 = add nsw i64 %566, 1, !dbg !24
  %570 = lshr i64 %569, 32, !dbg !25
  %compare.high = trunc i64 %570 to i32, !dbg !25
  %571 = lshr i64 %568, 32, !dbg !25
  %compare.high812 = trunc i64 %571 to i32, !dbg !25
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high812, !dbg !25
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high812, !dbg !25
  %compare.low.base = trunc i64 %569 to i32, !dbg !25
  %compare.low.base813 = trunc i64 %568 to i32, !dbg !25
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base813, !dbg !25
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !25
  %572 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 1) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !25
  %compare.low814 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !25
  %compare.result815 = select i1 %compare.high.equal, i1 %compare.low814, i1 %compare.high.order, !dbg !25
  %573 = select i1 %compare.result815, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base816 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 2) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local817 = or i32 %compare.local.base816, 2, !dbg !25
  %compare.low818 = icmp ugt i32 %compare.low.base, %compare.local817, !dbg !25
  %compare.result819 = select i1 %compare.high.equal, i1 %compare.low818, i1 %compare.high.order, !dbg !25
  %574 = select i1 %compare.result819, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base820 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 3) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local821 = or i32 %compare.local.base820, 3, !dbg !25
  %compare.low822 = icmp ugt i32 %compare.low.base, %compare.local821, !dbg !25
  %compare.result823 = select i1 %compare.high.equal, i1 %compare.low822, i1 %compare.high.order, !dbg !25
  %575 = select i1 %compare.result823, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base824 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 4) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local825 = or i32 %compare.local.base824, 4, !dbg !25
  %compare.low826 = icmp ugt i32 %compare.low.base, %compare.local825, !dbg !25
  %compare.result827 = select i1 %compare.high.equal, i1 %compare.low826, i1 %compare.high.order, !dbg !25
  %576 = select i1 %compare.result827, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base828 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 5) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local829 = or i32 %compare.local.base828, 5, !dbg !25
  %compare.low830 = icmp ugt i32 %compare.low.base, %compare.local829, !dbg !25
  %compare.result831 = select i1 %compare.high.equal, i1 %compare.low830, i1 %compare.high.order, !dbg !25
  %577 = select i1 %compare.result831, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base832 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 6) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local833 = or i32 %compare.local.base832, 6, !dbg !25
  %compare.low834 = icmp ugt i32 %compare.low.base, %compare.local833, !dbg !25
  %compare.result835 = select i1 %compare.high.equal, i1 %compare.low834, i1 %compare.high.order, !dbg !25
  %578 = select i1 %compare.result835, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base836 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 7) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local837 = or i32 %compare.local.base836, 7, !dbg !25
  %compare.low838 = icmp ugt i32 %compare.low.base, %compare.local837, !dbg !25
  %compare.result839 = select i1 %compare.high.equal, i1 %compare.low838, i1 %compare.high.order, !dbg !25
  %579 = select i1 %compare.result839, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base840 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 8) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local841 = or i32 %compare.local.base840, 8, !dbg !25
  %compare.low842 = icmp ugt i32 %compare.low.base, %compare.local841, !dbg !25
  %compare.result843 = select i1 %compare.high.equal, i1 %compare.low842, i1 %compare.high.order, !dbg !25
  %580 = select i1 %compare.result843, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base844 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 9) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local845 = or i32 %compare.local.base844, 9, !dbg !25
  %compare.low846 = icmp ugt i32 %compare.low.base, %compare.local845, !dbg !25
  %compare.result847 = select i1 %compare.high.equal, i1 %compare.low846, i1 %compare.high.order, !dbg !25
  %581 = select i1 %compare.result847, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base848 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 10) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local849 = or i32 %compare.local.base848, 10, !dbg !25
  %compare.low850 = icmp ugt i32 %compare.low.base, %compare.local849, !dbg !25
  %compare.result851 = select i1 %compare.high.equal, i1 %compare.low850, i1 %compare.high.order, !dbg !25
  %582 = select i1 %compare.result851, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base852 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 11) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local853 = or i32 %compare.local.base852, 11, !dbg !25
  %compare.low854 = icmp ugt i32 %compare.low.base, %compare.local853, !dbg !25
  %compare.result855 = select i1 %compare.high.equal, i1 %compare.low854, i1 %compare.high.order, !dbg !25
  %583 = select i1 %compare.result855, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base856 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 12) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local857 = or i32 %compare.local.base856, 12, !dbg !25
  %compare.low858 = icmp ugt i32 %compare.low.base, %compare.local857, !dbg !25
  %compare.result859 = select i1 %compare.high.equal, i1 %compare.low858, i1 %compare.high.order, !dbg !25
  %584 = select i1 %compare.result859, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base860 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 13) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local861 = or i32 %compare.local.base860, 13, !dbg !25
  %compare.low862 = icmp ugt i32 %compare.low.base, %compare.local861, !dbg !25
  %compare.result863 = select i1 %compare.high.equal, i1 %compare.low862, i1 %compare.high.order, !dbg !25
  %585 = select i1 %compare.result863, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base864 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 14) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local865 = or i32 %compare.local.base864, 14, !dbg !25
  %compare.low866 = icmp ugt i32 %compare.low.base, %compare.local865, !dbg !25
  %compare.result867 = select i1 %compare.high.equal, i1 %compare.low866, i1 %compare.high.order, !dbg !25
  %586 = select i1 %compare.result867, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base868 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 15) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local869 = or i32 %compare.local.base868, 15, !dbg !25
  %compare.low870 = icmp ugt i32 %compare.low.base, %compare.local869, !dbg !25
  %compare.result871 = select i1 %compare.high.equal, i1 %compare.low870, i1 %compare.high.order, !dbg !25
  %587 = select i1 %compare.result871, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base872 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 16) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local873 = or i32 %compare.local.base872, 64, !dbg !25
  %compare.low874 = icmp ugt i32 %compare.low.base, %compare.local873, !dbg !25
  %compare.result875 = select i1 %compare.high.equal, i1 %compare.low874, i1 %compare.high.order, !dbg !25
  %588 = select i1 %compare.result875, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base876 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 17) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local877 = or i32 %compare.local.base876, 65, !dbg !25
  %compare.low878 = icmp ugt i32 %compare.low.base, %compare.local877, !dbg !25
  %compare.result879 = select i1 %compare.high.equal, i1 %compare.low878, i1 %compare.high.order, !dbg !25
  %589 = select i1 %compare.result879, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base880 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 18) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local881 = or i32 %compare.local.base880, 66, !dbg !25
  %compare.low882 = icmp ugt i32 %compare.low.base, %compare.local881, !dbg !25
  %compare.result883 = select i1 %compare.high.equal, i1 %compare.low882, i1 %compare.high.order, !dbg !25
  %590 = select i1 %compare.result883, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base884 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 19) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local885 = or i32 %compare.local.base884, 67, !dbg !25
  %compare.low886 = icmp ugt i32 %compare.low.base, %compare.local885, !dbg !25
  %compare.result887 = select i1 %compare.high.equal, i1 %compare.low886, i1 %compare.high.order, !dbg !25
  %591 = select i1 %compare.result887, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base888 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 20) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local889 = or i32 %compare.local.base888, 68, !dbg !25
  %compare.low890 = icmp ugt i32 %compare.low.base, %compare.local889, !dbg !25
  %compare.result891 = select i1 %compare.high.equal, i1 %compare.low890, i1 %compare.high.order, !dbg !25
  %592 = select i1 %compare.result891, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base892 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 21) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local893 = or i32 %compare.local.base892, 69, !dbg !25
  %compare.low894 = icmp ugt i32 %compare.low.base, %compare.local893, !dbg !25
  %compare.result895 = select i1 %compare.high.equal, i1 %compare.low894, i1 %compare.high.order, !dbg !25
  %593 = select i1 %compare.result895, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base896 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 22) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local897 = or i32 %compare.local.base896, 70, !dbg !25
  %compare.low898 = icmp ugt i32 %compare.low.base, %compare.local897, !dbg !25
  %compare.result899 = select i1 %compare.high.equal, i1 %compare.low898, i1 %compare.high.order, !dbg !25
  %594 = select i1 %compare.result899, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base900 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 23) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local901 = or i32 %compare.local.base900, 71, !dbg !25
  %compare.low902 = icmp ugt i32 %compare.low.base, %compare.local901, !dbg !25
  %compare.result903 = select i1 %compare.high.equal, i1 %compare.low902, i1 %compare.high.order, !dbg !25
  %595 = select i1 %compare.result903, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base904 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 24) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local905 = or i32 %compare.local.base904, 72, !dbg !25
  %compare.low906 = icmp ugt i32 %compare.low.base, %compare.local905, !dbg !25
  %compare.result907 = select i1 %compare.high.equal, i1 %compare.low906, i1 %compare.high.order, !dbg !25
  %596 = select i1 %compare.result907, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base908 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 25) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local909 = or i32 %compare.local.base908, 73, !dbg !25
  %compare.low910 = icmp ugt i32 %compare.low.base, %compare.local909, !dbg !25
  %compare.result911 = select i1 %compare.high.equal, i1 %compare.low910, i1 %compare.high.order, !dbg !25
  %597 = select i1 %compare.result911, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base912 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 26) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local913 = or i32 %compare.local.base912, 74, !dbg !25
  %compare.low914 = icmp ugt i32 %compare.low.base, %compare.local913, !dbg !25
  %compare.result915 = select i1 %compare.high.equal, i1 %compare.low914, i1 %compare.high.order, !dbg !25
  %598 = select i1 %compare.result915, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base916 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 27) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local917 = or i32 %compare.local.base916, 75, !dbg !25
  %compare.low918 = icmp ugt i32 %compare.low.base, %compare.local917, !dbg !25
  %compare.result919 = select i1 %compare.high.equal, i1 %compare.low918, i1 %compare.high.order, !dbg !25
  %599 = select i1 %compare.result919, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base920 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 28) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local921 = or i32 %compare.local.base920, 76, !dbg !25
  %compare.low922 = icmp ugt i32 %compare.low.base, %compare.local921, !dbg !25
  %compare.result923 = select i1 %compare.high.equal, i1 %compare.low922, i1 %compare.high.order, !dbg !25
  %600 = select i1 %compare.result923, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base924 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 29) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local925 = or i32 %compare.local.base924, 77, !dbg !25
  %compare.low926 = icmp ugt i32 %compare.low.base, %compare.local925, !dbg !25
  %compare.result927 = select i1 %compare.high.equal, i1 %compare.low926, i1 %compare.high.order, !dbg !25
  %601 = select i1 %compare.result927, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base928 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 30) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local929 = or i32 %compare.local.base928, 78, !dbg !25
  %compare.low930 = icmp ugt i32 %compare.low.base, %compare.local929, !dbg !25
  %compare.result931 = select i1 %compare.high.equal, i1 %compare.low930, i1 %compare.high.order, !dbg !25
  %602 = select i1 %compare.result931, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base932 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base813, i32 31) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local933 = or i32 %compare.local.base932, 79, !dbg !25
  %compare.low934 = icmp ugt i32 %compare.low.base, %compare.local933, !dbg !25
  %compare.result935 = select i1 %compare.high.equal, i1 %compare.low934, i1 %compare.high.order, !dbg !25
  %603 = select i1 %compare.result935, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %604 = add nsw i64 %566, 65, !dbg !24
  %605 = lshr i64 %604, 32, !dbg !25
  %compare.high936 = trunc i64 %605 to i32, !dbg !25
  %606 = lshr i64 %568, 32, !dbg !25
  %compare.high937 = trunc i64 %606 to i32, !dbg !25
  %compare.high.equal938 = icmp eq i32 %compare.high936, %compare.high937, !dbg !25
  %compare.high.order939 = icmp ugt i32 %compare.high936, %compare.high937, !dbg !25
  %compare.low.base940 = trunc i64 %604 to i32, !dbg !25
  %compare.low.base941 = trunc i64 %568 to i32, !dbg !25
  %compare.low942 = icmp ugt i32 %compare.low.base940, %compare.low.base941, !dbg !25
  %compare.result943 = select i1 %compare.high.equal938, i1 %compare.low942, i1 %compare.high.order939, !dbg !25
  %607 = select i1 %compare.result943, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base944 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 32) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local945 = or i32 %compare.local.base944, 1, !dbg !25
  %compare.low946 = icmp ugt i32 %compare.low.base940, %compare.local945, !dbg !25
  %compare.result947 = select i1 %compare.high.equal938, i1 %compare.low946, i1 %compare.high.order939, !dbg !25
  %608 = select i1 %compare.result947, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base948 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 33) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local949 = or i32 %compare.local.base948, 2, !dbg !25
  %compare.low950 = icmp ugt i32 %compare.low.base940, %compare.local949, !dbg !25
  %compare.result951 = select i1 %compare.high.equal938, i1 %compare.low950, i1 %compare.high.order939, !dbg !25
  %609 = select i1 %compare.result951, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base952 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 34) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local953 = or i32 %compare.local.base952, 3, !dbg !25
  %compare.low954 = icmp ugt i32 %compare.low.base940, %compare.local953, !dbg !25
  %compare.result955 = select i1 %compare.high.equal938, i1 %compare.low954, i1 %compare.high.order939, !dbg !25
  %610 = select i1 %compare.result955, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base956 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 35) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local957 = or i32 %compare.local.base956, 4, !dbg !25
  %compare.low958 = icmp ugt i32 %compare.low.base940, %compare.local957, !dbg !25
  %compare.result959 = select i1 %compare.high.equal938, i1 %compare.low958, i1 %compare.high.order939, !dbg !25
  %611 = select i1 %compare.result959, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base960 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 36) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local961 = or i32 %compare.local.base960, 5, !dbg !25
  %compare.low962 = icmp ugt i32 %compare.low.base940, %compare.local961, !dbg !25
  %compare.result963 = select i1 %compare.high.equal938, i1 %compare.low962, i1 %compare.high.order939, !dbg !25
  %612 = select i1 %compare.result963, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base964 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 37) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local965 = or i32 %compare.local.base964, 6, !dbg !25
  %compare.low966 = icmp ugt i32 %compare.low.base940, %compare.local965, !dbg !25
  %compare.result967 = select i1 %compare.high.equal938, i1 %compare.low966, i1 %compare.high.order939, !dbg !25
  %613 = select i1 %compare.result967, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base968 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 38) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local969 = or i32 %compare.local.base968, 7, !dbg !25
  %compare.low970 = icmp ugt i32 %compare.low.base940, %compare.local969, !dbg !25
  %compare.result971 = select i1 %compare.high.equal938, i1 %compare.low970, i1 %compare.high.order939, !dbg !25
  %614 = select i1 %compare.result971, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base972 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 39) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local973 = or i32 %compare.local.base972, 8, !dbg !25
  %compare.low974 = icmp ugt i32 %compare.low.base940, %compare.local973, !dbg !25
  %compare.result975 = select i1 %compare.high.equal938, i1 %compare.low974, i1 %compare.high.order939, !dbg !25
  %615 = select i1 %compare.result975, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base976 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 40) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local977 = or i32 %compare.local.base976, 9, !dbg !25
  %compare.low978 = icmp ugt i32 %compare.low.base940, %compare.local977, !dbg !25
  %compare.result979 = select i1 %compare.high.equal938, i1 %compare.low978, i1 %compare.high.order939, !dbg !25
  %616 = select i1 %compare.result979, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base980 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 41) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local981 = or i32 %compare.local.base980, 10, !dbg !25
  %compare.low982 = icmp ugt i32 %compare.low.base940, %compare.local981, !dbg !25
  %compare.result983 = select i1 %compare.high.equal938, i1 %compare.low982, i1 %compare.high.order939, !dbg !25
  %617 = select i1 %compare.result983, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base984 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 42) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local985 = or i32 %compare.local.base984, 11, !dbg !25
  %compare.low986 = icmp ugt i32 %compare.low.base940, %compare.local985, !dbg !25
  %compare.result987 = select i1 %compare.high.equal938, i1 %compare.low986, i1 %compare.high.order939, !dbg !25
  %618 = select i1 %compare.result987, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base988 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 43) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local989 = or i32 %compare.local.base988, 12, !dbg !25
  %compare.low990 = icmp ugt i32 %compare.low.base940, %compare.local989, !dbg !25
  %compare.result991 = select i1 %compare.high.equal938, i1 %compare.low990, i1 %compare.high.order939, !dbg !25
  %619 = select i1 %compare.result991, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base992 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 44) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local993 = or i32 %compare.local.base992, 13, !dbg !25
  %compare.low994 = icmp ugt i32 %compare.low.base940, %compare.local993, !dbg !25
  %compare.result995 = select i1 %compare.high.equal938, i1 %compare.low994, i1 %compare.high.order939, !dbg !25
  %620 = select i1 %compare.result995, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base996 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 45) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local997 = or i32 %compare.local.base996, 14, !dbg !25
  %compare.low998 = icmp ugt i32 %compare.low.base940, %compare.local997, !dbg !25
  %compare.result999 = select i1 %compare.high.equal938, i1 %compare.low998, i1 %compare.high.order939, !dbg !25
  %621 = select i1 %compare.result999, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1000 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 46) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1001 = or i32 %compare.local.base1000, 15, !dbg !25
  %compare.low1002 = icmp ugt i32 %compare.low.base940, %compare.local1001, !dbg !25
  %compare.result1003 = select i1 %compare.high.equal938, i1 %compare.low1002, i1 %compare.high.order939, !dbg !25
  %622 = select i1 %compare.result1003, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1004 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 47) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1005 = or i32 %compare.local.base1004, 64, !dbg !25
  %compare.low1006 = icmp ugt i32 %compare.low.base940, %compare.local1005, !dbg !25
  %compare.result1007 = select i1 %compare.high.equal938, i1 %compare.low1006, i1 %compare.high.order939, !dbg !25
  %623 = select i1 %compare.result1007, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1008 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 48) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1009 = or i32 %compare.local.base1008, 65, !dbg !25
  %compare.low1010 = icmp ugt i32 %compare.low.base940, %compare.local1009, !dbg !25
  %compare.result1011 = select i1 %compare.high.equal938, i1 %compare.low1010, i1 %compare.high.order939, !dbg !25
  %624 = select i1 %compare.result1011, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1012 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 49) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1013 = or i32 %compare.local.base1012, 66, !dbg !25
  %compare.low1014 = icmp ugt i32 %compare.low.base940, %compare.local1013, !dbg !25
  %compare.result1015 = select i1 %compare.high.equal938, i1 %compare.low1014, i1 %compare.high.order939, !dbg !25
  %625 = select i1 %compare.result1015, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1016 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 50) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1017 = or i32 %compare.local.base1016, 67, !dbg !25
  %compare.low1018 = icmp ugt i32 %compare.low.base940, %compare.local1017, !dbg !25
  %compare.result1019 = select i1 %compare.high.equal938, i1 %compare.low1018, i1 %compare.high.order939, !dbg !25
  %626 = select i1 %compare.result1019, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1020 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 51) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1021 = or i32 %compare.local.base1020, 68, !dbg !25
  %compare.low1022 = icmp ugt i32 %compare.low.base940, %compare.local1021, !dbg !25
  %compare.result1023 = select i1 %compare.high.equal938, i1 %compare.low1022, i1 %compare.high.order939, !dbg !25
  %627 = select i1 %compare.result1023, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1024 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 52) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1025 = or i32 %compare.local.base1024, 69, !dbg !25
  %compare.low1026 = icmp ugt i32 %compare.low.base940, %compare.local1025, !dbg !25
  %compare.result1027 = select i1 %compare.high.equal938, i1 %compare.low1026, i1 %compare.high.order939, !dbg !25
  %628 = select i1 %compare.result1027, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1028 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 53) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1029 = or i32 %compare.local.base1028, 70, !dbg !25
  %compare.low1030 = icmp ugt i32 %compare.low.base940, %compare.local1029, !dbg !25
  %compare.result1031 = select i1 %compare.high.equal938, i1 %compare.low1030, i1 %compare.high.order939, !dbg !25
  %629 = select i1 %compare.result1031, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1032 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 54) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1033 = or i32 %compare.local.base1032, 71, !dbg !25
  %compare.low1034 = icmp ugt i32 %compare.low.base940, %compare.local1033, !dbg !25
  %compare.result1035 = select i1 %compare.high.equal938, i1 %compare.low1034, i1 %compare.high.order939, !dbg !25
  %630 = select i1 %compare.result1035, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1036 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 55) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1037 = or i32 %compare.local.base1036, 72, !dbg !25
  %compare.low1038 = icmp ugt i32 %compare.low.base940, %compare.local1037, !dbg !25
  %compare.result1039 = select i1 %compare.high.equal938, i1 %compare.low1038, i1 %compare.high.order939, !dbg !25
  %631 = select i1 %compare.result1039, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1040 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 56) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1041 = or i32 %compare.local.base1040, 73, !dbg !25
  %compare.low1042 = icmp ugt i32 %compare.low.base940, %compare.local1041, !dbg !25
  %compare.result1043 = select i1 %compare.high.equal938, i1 %compare.low1042, i1 %compare.high.order939, !dbg !25
  %632 = select i1 %compare.result1043, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1044 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 57) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1045 = or i32 %compare.local.base1044, 74, !dbg !25
  %compare.low1046 = icmp ugt i32 %compare.low.base940, %compare.local1045, !dbg !25
  %compare.result1047 = select i1 %compare.high.equal938, i1 %compare.low1046, i1 %compare.high.order939, !dbg !25
  %633 = select i1 %compare.result1047, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1048 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 58) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1049 = or i32 %compare.local.base1048, 75, !dbg !25
  %compare.low1050 = icmp ugt i32 %compare.low.base940, %compare.local1049, !dbg !25
  %compare.result1051 = select i1 %compare.high.equal938, i1 %compare.low1050, i1 %compare.high.order939, !dbg !25
  %634 = select i1 %compare.result1051, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1052 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 59) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1053 = or i32 %compare.local.base1052, 76, !dbg !25
  %compare.low1054 = icmp ugt i32 %compare.low.base940, %compare.local1053, !dbg !25
  %compare.result1055 = select i1 %compare.high.equal938, i1 %compare.low1054, i1 %compare.high.order939, !dbg !25
  %635 = select i1 %compare.result1055, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1056 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 60) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1057 = or i32 %compare.local.base1056, 77, !dbg !25
  %compare.low1058 = icmp ugt i32 %compare.low.base940, %compare.local1057, !dbg !25
  %compare.result1059 = select i1 %compare.high.equal938, i1 %compare.low1058, i1 %compare.high.order939, !dbg !25
  %636 = select i1 %compare.result1059, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1060 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 61) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1061 = or i32 %compare.local.base1060, 78, !dbg !25
  %compare.low1062 = icmp ugt i32 %compare.low.base940, %compare.local1061, !dbg !25
  %compare.result1063 = select i1 %compare.high.equal938, i1 %compare.low1062, i1 %compare.high.order939, !dbg !25
  %637 = select i1 %compare.result1063, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1064 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base941, i32 62) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1065 = or i32 %compare.local.base1064, 79, !dbg !25
  %compare.low1066 = icmp ugt i32 %compare.low.base940, %compare.local1065, !dbg !25
  %compare.result1067 = select i1 %compare.high.equal938, i1 %compare.low1066, i1 %compare.high.order939, !dbg !25
  %638 = select i1 %compare.result1067, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %639 = insertelement <32 x float> poison, float %572, i64 0, !dbg !15
  %640 = insertelement <32 x float> %639, float %573, i64 1, !dbg !15
  %641 = insertelement <32 x float> %640, float %574, i64 2, !dbg !15
  %642 = insertelement <32 x float> %641, float %575, i64 3, !dbg !15
  %643 = insertelement <32 x float> %642, float %576, i64 4, !dbg !15
  %644 = insertelement <32 x float> %643, float %577, i64 5, !dbg !15
  %645 = insertelement <32 x float> %644, float %578, i64 6, !dbg !15
  %646 = insertelement <32 x float> %645, float %579, i64 7, !dbg !15
  %647 = insertelement <32 x float> %646, float %580, i64 8, !dbg !15
  %648 = insertelement <32 x float> %647, float %581, i64 9, !dbg !15
  %649 = insertelement <32 x float> %648, float %582, i64 10, !dbg !15
  %650 = insertelement <32 x float> %649, float %583, i64 11, !dbg !15
  %651 = insertelement <32 x float> %650, float %584, i64 12, !dbg !15
  %652 = insertelement <32 x float> %651, float %585, i64 13, !dbg !15
  %653 = insertelement <32 x float> %652, float %586, i64 14, !dbg !15
  %654 = insertelement <32 x float> %653, float %587, i64 15, !dbg !15
  %655 = insertelement <32 x float> %654, float %588, i64 16, !dbg !15
  %656 = insertelement <32 x float> %655, float %589, i64 17, !dbg !15
  %657 = insertelement <32 x float> %656, float %590, i64 18, !dbg !15
  %658 = insertelement <32 x float> %657, float %591, i64 19, !dbg !15
  %659 = insertelement <32 x float> %658, float %592, i64 20, !dbg !15
  %660 = insertelement <32 x float> %659, float %593, i64 21, !dbg !15
  %661 = insertelement <32 x float> %660, float %594, i64 22, !dbg !15
  %662 = insertelement <32 x float> %661, float %595, i64 23, !dbg !15
  %663 = insertelement <32 x float> %662, float %596, i64 24, !dbg !15
  %664 = insertelement <32 x float> %663, float %597, i64 25, !dbg !15
  %665 = insertelement <32 x float> %664, float %598, i64 26, !dbg !15
  %666 = insertelement <32 x float> %665, float %599, i64 27, !dbg !15
  %667 = insertelement <32 x float> %666, float %600, i64 28, !dbg !15
  %668 = insertelement <32 x float> %667, float %601, i64 29, !dbg !15
  %669 = insertelement <32 x float> %668, float %602, i64 30, !dbg !15
  %670 = insertelement <32 x float> %669, float %603, i64 31, !dbg !15
  %671 = insertvalue [2 x <32 x float>] poison, <32 x float> %670, 0, !dbg !15
  %672 = insertelement <32 x float> poison, float %607, i64 0, !dbg !15
  %673 = insertelement <32 x float> %672, float %608, i64 1, !dbg !15
  %674 = insertelement <32 x float> %673, float %609, i64 2, !dbg !15
  %675 = insertelement <32 x float> %674, float %610, i64 3, !dbg !15
  %676 = insertelement <32 x float> %675, float %611, i64 4, !dbg !15
  %677 = insertelement <32 x float> %676, float %612, i64 5, !dbg !15
  %678 = insertelement <32 x float> %677, float %613, i64 6, !dbg !15
  %679 = insertelement <32 x float> %678, float %614, i64 7, !dbg !15
  %680 = insertelement <32 x float> %679, float %615, i64 8, !dbg !15
  %681 = insertelement <32 x float> %680, float %616, i64 9, !dbg !15
  %682 = insertelement <32 x float> %681, float %617, i64 10, !dbg !15
  %683 = insertelement <32 x float> %682, float %618, i64 11, !dbg !15
  %684 = insertelement <32 x float> %683, float %619, i64 12, !dbg !15
  %685 = insertelement <32 x float> %684, float %620, i64 13, !dbg !15
  %686 = insertelement <32 x float> %685, float %621, i64 14, !dbg !15
  %687 = insertelement <32 x float> %686, float %622, i64 15, !dbg !15
  %688 = insertelement <32 x float> %687, float %623, i64 16, !dbg !15
  %689 = insertelement <32 x float> %688, float %624, i64 17, !dbg !15
  %690 = insertelement <32 x float> %689, float %625, i64 18, !dbg !15
  %691 = insertelement <32 x float> %690, float %626, i64 19, !dbg !15
  %692 = insertelement <32 x float> %691, float %627, i64 20, !dbg !15
  %693 = insertelement <32 x float> %692, float %628, i64 21, !dbg !15
  %694 = insertelement <32 x float> %693, float %629, i64 22, !dbg !15
  %695 = insertelement <32 x float> %694, float %630, i64 23, !dbg !15
  %696 = insertelement <32 x float> %695, float %631, i64 24, !dbg !15
  %697 = insertelement <32 x float> %696, float %632, i64 25, !dbg !15
  %698 = insertelement <32 x float> %697, float %633, i64 26, !dbg !15
  %699 = insertelement <32 x float> %698, float %634, i64 27, !dbg !15
  %700 = insertelement <32 x float> %699, float %635, i64 28, !dbg !15
  %701 = insertelement <32 x float> %700, float %636, i64 29, !dbg !15
  %702 = insertelement <32 x float> %701, float %637, i64 30, !dbg !15
  %703 = insertelement <32 x float> %702, float %638, i64 31, !dbg !15
  %704 = insertvalue [2 x <32 x float>] %671, <32 x float> %703, 1, !dbg !15
  br label %705, !dbg !15

705:                                              ; preds = %560, %561
  %706 = phi [2 x <32 x float>] [ %704, %561 ], [ zeroinitializer, %560 ], !dbg !15
  br label %707, !dbg !15

707:                                              ; preds = %705
  %708 = extractvalue [2 x <32 x float>] %164, 0, !dbg !27
  %709 = extractvalue [2 x <32 x float>] %706, 0, !dbg !27
  %710 = shufflevector <32 x float> %708, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %711 = shufflevector <32 x float> %709, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %712 = fadd <4 x float> %710, %711, !dbg !27
  %713 = extractelement <4 x float> %712, i64 0, !dbg !28
  %714 = call float @llvm.amdgcn.exp2.f32(float %713), !dbg !28
  %715 = insertelement <4 x float> poison, float %714, i64 0, !dbg !30
  %716 = extractelement <4 x float> %712, i64 1, !dbg !28
  %717 = call float @llvm.amdgcn.exp2.f32(float %716), !dbg !28
  %718 = insertelement <4 x float> %715, float %717, i64 1, !dbg !30
  %719 = fadd float %714, %717, !dbg !29
  %720 = extractelement <4 x float> %712, i64 2, !dbg !28
  %721 = call float @llvm.amdgcn.exp2.f32(float %720), !dbg !28
  %722 = insertelement <4 x float> %718, float %721, i64 2, !dbg !30
  %723 = fadd float %719, %721, !dbg !29
  %724 = extractelement <4 x float> %712, i64 3, !dbg !28
  %725 = call float @llvm.amdgcn.exp2.f32(float %724), !dbg !28
  %726 = insertelement <4 x float> %722, float %725, i64 3, !dbg !30
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %726) #9, !dbg !30, !frisk.register_lifetimes !1
  %727 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !30
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %727) #9, !dbg !29, !frisk.register_lifetimes !1
  %728 = fadd float %723, %725, !dbg !29
  %729 = shufflevector <32 x float> %708, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %730 = shufflevector <32 x float> %709, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %731 = fadd <4 x float> %729, %730, !dbg !27
  %732 = extractelement <4 x float> %731, i64 0, !dbg !28
  %733 = call float @llvm.amdgcn.exp2.f32(float %732), !dbg !28
  %734 = insertelement <4 x float> poison, float %733, i64 0, !dbg !30
  %735 = fadd float %728, %733, !dbg !29
  %736 = extractelement <4 x float> %731, i64 1, !dbg !28
  %737 = call float @llvm.amdgcn.exp2.f32(float %736), !dbg !28
  %738 = insertelement <4 x float> %734, float %737, i64 1, !dbg !30
  %739 = fadd float %735, %737, !dbg !29
  %740 = extractelement <4 x float> %731, i64 2, !dbg !28
  %741 = call float @llvm.amdgcn.exp2.f32(float %740), !dbg !28
  %742 = insertelement <4 x float> %738, float %741, i64 2, !dbg !30
  %743 = fadd float %739, %741, !dbg !29
  %744 = extractelement <4 x float> %731, i64 3, !dbg !28
  %745 = call float @llvm.amdgcn.exp2.f32(float %744), !dbg !28
  %746 = insertelement <4 x float> %742, float %745, i64 3, !dbg !30
  %math.result111 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %746) #9, !dbg !30, !frisk.register_lifetimes !1
  %747 = fptrunc <4 x float> %math.result111 to <4 x half>, !dbg !30
  %math.packed112 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %747) #9, !dbg !29, !frisk.register_lifetimes !1
  %748 = fadd float %743, %745, !dbg !29
  %749 = shufflevector <32 x float> %708, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %750 = shufflevector <32 x float> %709, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %751 = fadd <4 x float> %749, %750, !dbg !27
  %752 = extractelement <4 x float> %751, i64 0, !dbg !28
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !28
  %754 = insertelement <4 x float> poison, float %753, i64 0, !dbg !30
  %755 = fadd float %748, %753, !dbg !29
  %756 = extractelement <4 x float> %751, i64 1, !dbg !28
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !28
  %758 = insertelement <4 x float> %754, float %757, i64 1, !dbg !30
  %759 = fadd float %755, %757, !dbg !29
  %760 = extractelement <4 x float> %751, i64 2, !dbg !28
  %761 = call float @llvm.amdgcn.exp2.f32(float %760), !dbg !28
  %762 = insertelement <4 x float> %758, float %761, i64 2, !dbg !30
  %763 = fadd float %759, %761, !dbg !29
  %764 = extractelement <4 x float> %751, i64 3, !dbg !28
  %765 = call float @llvm.amdgcn.exp2.f32(float %764), !dbg !28
  %766 = insertelement <4 x float> %762, float %765, i64 3, !dbg !30
  %math.result113 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %766) #9, !dbg !30, !frisk.register_lifetimes !1
  %767 = fptrunc <4 x float> %math.result113 to <4 x half>, !dbg !30
  %math.packed114 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %767) #9, !dbg !29, !frisk.register_lifetimes !1
  %768 = fadd float %763, %765, !dbg !29
  %769 = shufflevector <32 x float> %708, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %770 = shufflevector <32 x float> %709, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %771 = fadd <4 x float> %769, %770, !dbg !27
  %772 = extractelement <4 x float> %771, i64 0, !dbg !28
  %773 = call float @llvm.amdgcn.exp2.f32(float %772), !dbg !28
  %774 = insertelement <4 x float> poison, float %773, i64 0, !dbg !30
  %775 = fadd float %768, %773, !dbg !29
  %776 = extractelement <4 x float> %771, i64 1, !dbg !28
  %777 = call float @llvm.amdgcn.exp2.f32(float %776), !dbg !28
  %778 = insertelement <4 x float> %774, float %777, i64 1, !dbg !30
  %779 = fadd float %775, %777, !dbg !29
  %780 = extractelement <4 x float> %771, i64 2, !dbg !28
  %781 = call float @llvm.amdgcn.exp2.f32(float %780), !dbg !28
  %782 = insertelement <4 x float> %778, float %781, i64 2, !dbg !30
  %783 = fadd float %779, %781, !dbg !29
  %784 = extractelement <4 x float> %771, i64 3, !dbg !28
  %785 = call float @llvm.amdgcn.exp2.f32(float %784), !dbg !28
  %786 = insertelement <4 x float> %782, float %785, i64 3, !dbg !30
  %math.result115 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %786) #9, !dbg !30, !frisk.register_lifetimes !1
  %787 = fptrunc <4 x float> %math.result115 to <4 x half>, !dbg !30
  %math.packed116 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %787) #9, !dbg !29, !frisk.register_lifetimes !1
  %788 = fadd float %783, %785, !dbg !29
  %789 = shufflevector <32 x float> %708, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %790 = shufflevector <32 x float> %709, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %791 = fadd <4 x float> %789, %790, !dbg !27
  %792 = extractelement <4 x float> %791, i64 0, !dbg !28
  %793 = call float @llvm.amdgcn.exp2.f32(float %792), !dbg !28
  %794 = insertelement <4 x float> poison, float %793, i64 0, !dbg !30
  %795 = fadd float %788, %793, !dbg !29
  %796 = extractelement <4 x float> %791, i64 1, !dbg !28
  %797 = call float @llvm.amdgcn.exp2.f32(float %796), !dbg !28
  %798 = insertelement <4 x float> %794, float %797, i64 1, !dbg !30
  %799 = fadd float %795, %797, !dbg !29
  %800 = extractelement <4 x float> %791, i64 2, !dbg !28
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !28
  %802 = insertelement <4 x float> %798, float %801, i64 2, !dbg !30
  %803 = fadd float %799, %801, !dbg !29
  %804 = extractelement <4 x float> %791, i64 3, !dbg !28
  %805 = call float @llvm.amdgcn.exp2.f32(float %804), !dbg !28
  %806 = insertelement <4 x float> %802, float %805, i64 3, !dbg !30
  %math.result117 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %806) #9, !dbg !30, !frisk.register_lifetimes !1
  %807 = fptrunc <4 x float> %math.result117 to <4 x half>, !dbg !30
  %math.packed118 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %807) #9, !dbg !29, !frisk.register_lifetimes !1
  %808 = fadd float %803, %805, !dbg !29
  %809 = shufflevector <32 x float> %708, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %810 = shufflevector <32 x float> %709, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %811 = fadd <4 x float> %809, %810, !dbg !27
  %812 = extractelement <4 x float> %811, i64 0, !dbg !28
  %813 = call float @llvm.amdgcn.exp2.f32(float %812), !dbg !28
  %814 = insertelement <4 x float> poison, float %813, i64 0, !dbg !30
  %815 = fadd float %808, %813, !dbg !29
  %816 = extractelement <4 x float> %811, i64 1, !dbg !28
  %817 = call float @llvm.amdgcn.exp2.f32(float %816), !dbg !28
  %818 = insertelement <4 x float> %814, float %817, i64 1, !dbg !30
  %819 = fadd float %815, %817, !dbg !29
  %820 = extractelement <4 x float> %811, i64 2, !dbg !28
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !28
  %822 = insertelement <4 x float> %818, float %821, i64 2, !dbg !30
  %823 = fadd float %819, %821, !dbg !29
  %824 = extractelement <4 x float> %811, i64 3, !dbg !28
  %825 = call float @llvm.amdgcn.exp2.f32(float %824), !dbg !28
  %826 = insertelement <4 x float> %822, float %825, i64 3, !dbg !30
  %math.result119 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %826) #9, !dbg !30, !frisk.register_lifetimes !1
  %827 = fptrunc <4 x float> %math.result119 to <4 x half>, !dbg !30
  %math.packed120 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %827) #9, !dbg !29, !frisk.register_lifetimes !1
  %828 = fadd float %823, %825, !dbg !29
  %829 = shufflevector <32 x float> %708, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %830 = shufflevector <32 x float> %709, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %831 = fadd <4 x float> %829, %830, !dbg !27
  %832 = extractelement <4 x float> %831, i64 0, !dbg !28
  %833 = call float @llvm.amdgcn.exp2.f32(float %832), !dbg !28
  %834 = insertelement <4 x float> poison, float %833, i64 0, !dbg !30
  %835 = fadd float %828, %833, !dbg !29
  %836 = extractelement <4 x float> %831, i64 1, !dbg !28
  %837 = call float @llvm.amdgcn.exp2.f32(float %836), !dbg !28
  %838 = insertelement <4 x float> %834, float %837, i64 1, !dbg !30
  %839 = fadd float %835, %837, !dbg !29
  %840 = extractelement <4 x float> %831, i64 2, !dbg !28
  %841 = call float @llvm.amdgcn.exp2.f32(float %840), !dbg !28
  %842 = insertelement <4 x float> %838, float %841, i64 2, !dbg !30
  %843 = fadd float %839, %841, !dbg !29
  %844 = extractelement <4 x float> %831, i64 3, !dbg !28
  %845 = call float @llvm.amdgcn.exp2.f32(float %844), !dbg !28
  %846 = insertelement <4 x float> %842, float %845, i64 3, !dbg !30
  %math.result121 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %846) #9, !dbg !30, !frisk.register_lifetimes !1
  %847 = fptrunc <4 x float> %math.result121 to <4 x half>, !dbg !30
  %math.packed122 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %847) #9, !dbg !29, !frisk.register_lifetimes !1
  %848 = fadd float %843, %845, !dbg !29
  %849 = shufflevector <32 x float> %708, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %850 = shufflevector <32 x float> %709, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %851 = fadd <4 x float> %849, %850, !dbg !27
  %852 = extractelement <4 x float> %851, i64 0, !dbg !28
  %853 = call float @llvm.amdgcn.exp2.f32(float %852), !dbg !28
  %854 = insertelement <4 x float> poison, float %853, i64 0, !dbg !30
  %855 = fadd float %848, %853, !dbg !29
  %856 = extractelement <4 x float> %851, i64 1, !dbg !28
  %857 = call float @llvm.amdgcn.exp2.f32(float %856), !dbg !28
  %858 = insertelement <4 x float> %854, float %857, i64 1, !dbg !30
  %859 = fadd float %855, %857, !dbg !29
  %860 = extractelement <4 x float> %851, i64 2, !dbg !28
  %861 = call float @llvm.amdgcn.exp2.f32(float %860), !dbg !28
  %862 = insertelement <4 x float> %858, float %861, i64 2, !dbg !30
  %863 = fadd float %859, %861, !dbg !29
  %864 = extractelement <4 x float> %851, i64 3, !dbg !28
  %865 = call float @llvm.amdgcn.exp2.f32(float %864), !dbg !28
  %866 = insertelement <4 x float> %862, float %865, i64 3, !dbg !30
  %math.result123 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %866) #9, !dbg !30, !frisk.register_lifetimes !1
  %867 = fptrunc <4 x float> %math.result123 to <4 x half>, !dbg !30
  %math.packed124 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %867) #9, !dbg !29, !frisk.register_lifetimes !1
  %868 = fadd float %863, %865, !dbg !29
  %869 = extractvalue [2 x <32 x float>] %164, 1, !dbg !27
  %870 = extractvalue [2 x <32 x float>] %706, 1, !dbg !27
  %871 = shufflevector <32 x float> %869, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %872 = shufflevector <32 x float> %870, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %873 = fadd <4 x float> %871, %872, !dbg !27
  %874 = extractelement <4 x float> %873, i64 0, !dbg !28
  %875 = call float @llvm.amdgcn.exp2.f32(float %874), !dbg !28
  %876 = insertelement <4 x float> poison, float %875, i64 0, !dbg !30
  %877 = extractelement <4 x float> %873, i64 1, !dbg !28
  %878 = call float @llvm.amdgcn.exp2.f32(float %877), !dbg !28
  %879 = insertelement <4 x float> %876, float %878, i64 1, !dbg !30
  %880 = fadd float %875, %878, !dbg !29
  %881 = extractelement <4 x float> %873, i64 2, !dbg !28
  %882 = call float @llvm.amdgcn.exp2.f32(float %881), !dbg !28
  %883 = insertelement <4 x float> %879, float %882, i64 2, !dbg !30
  %884 = fadd float %880, %882, !dbg !29
  %885 = extractelement <4 x float> %873, i64 3, !dbg !28
  %886 = call float @llvm.amdgcn.exp2.f32(float %885), !dbg !28
  %887 = insertelement <4 x float> %883, float %886, i64 3, !dbg !30
  %math.result125 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %887) #9, !dbg !30, !frisk.register_lifetimes !1
  %888 = fptrunc <4 x float> %math.result125 to <4 x half>, !dbg !30
  %math.packed126 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %888) #9, !dbg !29, !frisk.register_lifetimes !1
  %889 = fadd float %884, %886, !dbg !29
  %890 = shufflevector <32 x float> %869, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %891 = shufflevector <32 x float> %870, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %892 = fadd <4 x float> %890, %891, !dbg !27
  %893 = extractelement <4 x float> %892, i64 0, !dbg !28
  %894 = call float @llvm.amdgcn.exp2.f32(float %893), !dbg !28
  %895 = insertelement <4 x float> poison, float %894, i64 0, !dbg !30
  %896 = fadd float %889, %894, !dbg !29
  %897 = extractelement <4 x float> %892, i64 1, !dbg !28
  %898 = call float @llvm.amdgcn.exp2.f32(float %897), !dbg !28
  %899 = insertelement <4 x float> %895, float %898, i64 1, !dbg !30
  %900 = fadd float %896, %898, !dbg !29
  %901 = extractelement <4 x float> %892, i64 2, !dbg !28
  %902 = call float @llvm.amdgcn.exp2.f32(float %901), !dbg !28
  %903 = insertelement <4 x float> %899, float %902, i64 2, !dbg !30
  %904 = fadd float %900, %902, !dbg !29
  %905 = extractelement <4 x float> %892, i64 3, !dbg !28
  %906 = call float @llvm.amdgcn.exp2.f32(float %905), !dbg !28
  %907 = insertelement <4 x float> %903, float %906, i64 3, !dbg !30
  %math.result127 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %907) #9, !dbg !30, !frisk.register_lifetimes !1
  %908 = fptrunc <4 x float> %math.result127 to <4 x half>, !dbg !30
  %math.packed128 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %908) #9, !dbg !29, !frisk.register_lifetimes !1
  %909 = fadd float %904, %906, !dbg !29
  %910 = shufflevector <32 x float> %869, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %911 = shufflevector <32 x float> %870, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %912 = fadd <4 x float> %910, %911, !dbg !27
  %913 = extractelement <4 x float> %912, i64 0, !dbg !28
  %914 = call float @llvm.amdgcn.exp2.f32(float %913), !dbg !28
  %915 = insertelement <4 x float> poison, float %914, i64 0, !dbg !30
  %916 = fadd float %909, %914, !dbg !29
  %917 = extractelement <4 x float> %912, i64 1, !dbg !28
  %918 = call float @llvm.amdgcn.exp2.f32(float %917), !dbg !28
  %919 = insertelement <4 x float> %915, float %918, i64 1, !dbg !30
  %920 = fadd float %916, %918, !dbg !29
  %921 = extractelement <4 x float> %912, i64 2, !dbg !28
  %922 = call float @llvm.amdgcn.exp2.f32(float %921), !dbg !28
  %923 = insertelement <4 x float> %919, float %922, i64 2, !dbg !30
  %924 = fadd float %920, %922, !dbg !29
  %925 = extractelement <4 x float> %912, i64 3, !dbg !28
  %926 = call float @llvm.amdgcn.exp2.f32(float %925), !dbg !28
  %927 = insertelement <4 x float> %923, float %926, i64 3, !dbg !30
  %math.result129 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %927) #9, !dbg !30, !frisk.register_lifetimes !1
  %928 = fptrunc <4 x float> %math.result129 to <4 x half>, !dbg !30
  %math.packed130 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %928) #9, !dbg !29, !frisk.register_lifetimes !1
  %929 = fadd float %924, %926, !dbg !29
  %930 = shufflevector <32 x float> %869, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %931 = shufflevector <32 x float> %870, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %932 = fadd <4 x float> %930, %931, !dbg !27
  %933 = extractelement <4 x float> %932, i64 0, !dbg !28
  %934 = call float @llvm.amdgcn.exp2.f32(float %933), !dbg !28
  %935 = insertelement <4 x float> poison, float %934, i64 0, !dbg !30
  %936 = fadd float %929, %934, !dbg !29
  %937 = extractelement <4 x float> %932, i64 1, !dbg !28
  %938 = call float @llvm.amdgcn.exp2.f32(float %937), !dbg !28
  %939 = insertelement <4 x float> %935, float %938, i64 1, !dbg !30
  %940 = fadd float %936, %938, !dbg !29
  %941 = extractelement <4 x float> %932, i64 2, !dbg !28
  %942 = call float @llvm.amdgcn.exp2.f32(float %941), !dbg !28
  %943 = insertelement <4 x float> %939, float %942, i64 2, !dbg !30
  %944 = fadd float %940, %942, !dbg !29
  %945 = extractelement <4 x float> %932, i64 3, !dbg !28
  %946 = call float @llvm.amdgcn.exp2.f32(float %945), !dbg !28
  %947 = insertelement <4 x float> %943, float %946, i64 3, !dbg !30
  %math.result131 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %947) #9, !dbg !30, !frisk.register_lifetimes !1
  %948 = fptrunc <4 x float> %math.result131 to <4 x half>, !dbg !30
  %math.packed132 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %948) #9, !dbg !29, !frisk.register_lifetimes !1
  %949 = fadd float %944, %946, !dbg !29
  %950 = shufflevector <32 x float> %869, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %951 = shufflevector <32 x float> %870, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %952 = fadd <4 x float> %950, %951, !dbg !27
  %953 = extractelement <4 x float> %952, i64 0, !dbg !28
  %954 = call float @llvm.amdgcn.exp2.f32(float %953), !dbg !28
  %955 = insertelement <4 x float> poison, float %954, i64 0, !dbg !30
  %956 = fadd float %949, %954, !dbg !29
  %957 = extractelement <4 x float> %952, i64 1, !dbg !28
  %958 = call float @llvm.amdgcn.exp2.f32(float %957), !dbg !28
  %959 = insertelement <4 x float> %955, float %958, i64 1, !dbg !30
  %960 = fadd float %956, %958, !dbg !29
  %961 = extractelement <4 x float> %952, i64 2, !dbg !28
  %962 = call float @llvm.amdgcn.exp2.f32(float %961), !dbg !28
  %963 = insertelement <4 x float> %959, float %962, i64 2, !dbg !30
  %964 = fadd float %960, %962, !dbg !29
  %965 = extractelement <4 x float> %952, i64 3, !dbg !28
  %966 = call float @llvm.amdgcn.exp2.f32(float %965), !dbg !28
  %967 = insertelement <4 x float> %963, float %966, i64 3, !dbg !30
  %math.result133 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %967) #9, !dbg !30, !frisk.register_lifetimes !1
  %968 = fptrunc <4 x float> %math.result133 to <4 x half>, !dbg !30
  %math.packed134 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %968) #9, !dbg !29, !frisk.register_lifetimes !1
  %969 = fadd float %964, %966, !dbg !29
  %970 = shufflevector <32 x float> %869, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %971 = shufflevector <32 x float> %870, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %972 = fadd <4 x float> %970, %971, !dbg !27
  %973 = extractelement <4 x float> %972, i64 0, !dbg !28
  %974 = call float @llvm.amdgcn.exp2.f32(float %973), !dbg !28
  %975 = insertelement <4 x float> poison, float %974, i64 0, !dbg !30
  %976 = fadd float %969, %974, !dbg !29
  %977 = extractelement <4 x float> %972, i64 1, !dbg !28
  %978 = call float @llvm.amdgcn.exp2.f32(float %977), !dbg !28
  %979 = insertelement <4 x float> %975, float %978, i64 1, !dbg !30
  %980 = fadd float %976, %978, !dbg !29
  %981 = extractelement <4 x float> %972, i64 2, !dbg !28
  %982 = call float @llvm.amdgcn.exp2.f32(float %981), !dbg !28
  %983 = insertelement <4 x float> %979, float %982, i64 2, !dbg !30
  %984 = fadd float %980, %982, !dbg !29
  %985 = extractelement <4 x float> %972, i64 3, !dbg !28
  %986 = call float @llvm.amdgcn.exp2.f32(float %985), !dbg !28
  %987 = insertelement <4 x float> %983, float %986, i64 3, !dbg !30
  %math.result135 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %987) #9, !dbg !30, !frisk.register_lifetimes !1
  %988 = fptrunc <4 x float> %math.result135 to <4 x half>, !dbg !30
  %math.packed136 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %988) #9, !dbg !29, !frisk.register_lifetimes !1
  %989 = fadd float %984, %986, !dbg !29
  %990 = shufflevector <32 x float> %869, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %991 = shufflevector <32 x float> %870, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %992 = fadd <4 x float> %990, %991, !dbg !27
  %993 = extractelement <4 x float> %992, i64 0, !dbg !28
  %994 = call float @llvm.amdgcn.exp2.f32(float %993), !dbg !28
  %995 = insertelement <4 x float> poison, float %994, i64 0, !dbg !30
  %996 = fadd float %989, %994, !dbg !29
  %997 = extractelement <4 x float> %992, i64 1, !dbg !28
  %998 = call float @llvm.amdgcn.exp2.f32(float %997), !dbg !28
  %999 = insertelement <4 x float> %995, float %998, i64 1, !dbg !30
  %1000 = fadd float %996, %998, !dbg !29
  %1001 = extractelement <4 x float> %992, i64 2, !dbg !28
  %1002 = call float @llvm.amdgcn.exp2.f32(float %1001), !dbg !28
  %1003 = insertelement <4 x float> %999, float %1002, i64 2, !dbg !30
  %1004 = fadd float %1000, %1002, !dbg !29
  %1005 = extractelement <4 x float> %992, i64 3, !dbg !28
  %1006 = call float @llvm.amdgcn.exp2.f32(float %1005), !dbg !28
  %1007 = insertelement <4 x float> %1003, float %1006, i64 3, !dbg !30
  %math.result137 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1007) #9, !dbg !30, !frisk.register_lifetimes !1
  %1008 = fptrunc <4 x float> %math.result137 to <4 x half>, !dbg !30
  %math.packed138 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1008) #9, !dbg !29, !frisk.register_lifetimes !1
  %1009 = fadd float %1004, %1006, !dbg !29
  %1010 = shufflevector <32 x float> %869, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %1011 = shufflevector <32 x float> %870, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %1012 = fadd <4 x float> %1010, %1011, !dbg !27
  %1013 = extractelement <4 x float> %1012, i64 0, !dbg !28
  %1014 = call float @llvm.amdgcn.exp2.f32(float %1013), !dbg !28
  %1015 = insertelement <4 x float> poison, float %1014, i64 0, !dbg !30
  %1016 = fadd float %1009, %1014, !dbg !29
  %1017 = extractelement <4 x float> %1012, i64 1, !dbg !28
  %1018 = call float @llvm.amdgcn.exp2.f32(float %1017), !dbg !28
  %1019 = insertelement <4 x float> %1015, float %1018, i64 1, !dbg !30
  %1020 = fadd float %1016, %1018, !dbg !29
  %1021 = extractelement <4 x float> %1012, i64 2, !dbg !28
  %1022 = call float @llvm.amdgcn.exp2.f32(float %1021), !dbg !28
  %1023 = insertelement <4 x float> %1019, float %1022, i64 2, !dbg !30
  %1024 = fadd float %1020, %1022, !dbg !29
  %1025 = extractelement <4 x float> %1012, i64 3, !dbg !28
  %1026 = call float @llvm.amdgcn.exp2.f32(float %1025), !dbg !28
  %1027 = insertelement <4 x float> %1023, float %1026, i64 3, !dbg !30
  %math.result139 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1027) #9, !dbg !30, !frisk.register_lifetimes !1
  %1028 = fptrunc <4 x float> %math.result139 to <4 x half>, !dbg !30
  %math.packed140 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1028) #9, !dbg !29, !frisk.register_lifetimes !1
  %1029 = fadd float %1024, %1026, !dbg !29
  %1030 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !29
  %1031 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1030), !dbg !29
  %1032 = add nuw nsw i32 %1031, 64, !dbg !29
  %1033 = xor i32 %1031, 16, !dbg !29
  %1034 = and i32 %1032, 192, !dbg !29
  %1035 = icmp samesign ult i32 %1033, %1034, !dbg !29
  %1036 = select i1 %1035, i32 %1033, i32 %1031, !dbg !29
  %1037 = shl nuw nsw i32 %1036, 2, !dbg !29
  %1038 = bitcast float %868 to i32, !dbg !29
  %1039 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1037, i32 %1038), !dbg !29
  %1040 = bitcast i32 %1039 to float, !dbg !29
  %1041 = fadd float %868, %1040, !dbg !29
  %1042 = xor i32 %1031, 32, !dbg !29
  %1043 = icmp samesign ult i32 %1042, %1034, !dbg !29
  %1044 = select i1 %1043, i32 %1042, i32 %1031, !dbg !29
  %1045 = shl nuw nsw i32 %1044, 2, !dbg !29
  %1046 = bitcast float %1041 to i32, !dbg !29
  %1047 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1045, i32 %1046), !dbg !29
  %1048 = bitcast i32 %1047 to float, !dbg !29
  %1049 = fadd float %1041, %1048, !dbg !29
  %1050 = extractvalue [2 x <1 x float>] %32, 0, !dbg !29
  %1051 = extractelement <1 x float> %1050, i64 0, !dbg !29
  %1052 = fadd float %1049, %1051, !dbg !29
  %1053 = insertelement <1 x float> poison, float %1052, i64 0, !dbg !17
  %1054 = insertvalue [2 x <1 x float>] poison, <1 x float> %1053, 0, !dbg !17
  %1055 = bitcast float %1029 to i32, !dbg !29
  %1056 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1037, i32 %1055), !dbg !29
  %1057 = bitcast i32 %1056 to float, !dbg !29
  %1058 = fadd float %1029, %1057, !dbg !29
  %1059 = bitcast float %1058 to i32, !dbg !29
  %1060 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1045, i32 %1059), !dbg !29
  %1061 = bitcast i32 %1060 to float, !dbg !29
  %1062 = fadd float %1058, %1061, !dbg !29
  %1063 = extractvalue [2 x <1 x float>] %32, 1, !dbg !29
  %1064 = extractelement <1 x float> %1063, i64 0, !dbg !29
  %1065 = fadd float %1062, %1064, !dbg !29
  %1066 = insertelement <1 x float> poison, float %1065, i64 0, !dbg !17
  %1067 = insertvalue [2 x <1 x float>] %1054, <1 x float> %1066, 1, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread519 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !16, !range !35, !frisk.register_lifetimes !1
  %address.local520 = shl nuw nsw i32 %address.thread519, 3, !dbg !16
  %address.local521 = and i32 %address.local520, 120, !dbg !16
  %address.local522 = shl nuw nsw i32 %address.thread519, 5, !dbg !16
  %address.local523 = and i32 %address.local522, 1536, !dbg !16
  %address.local524 = or disjoint i32 %address.local521, %address.local523, !dbg !16
  %address.local525 = zext nneg i32 %address.local524 to i64, !dbg !16
  %address.local526 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local525, !dbg !16
  %1068 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local526, i32 0) #0, !dbg !16
  %1069 = bitcast <4 x i32> %1068 to <8 x half>, !dbg !16
  %address.local527 = zext nneg i32 %address.thread519 to i64, !dbg !14
  %address.local528 = and i64 %address.local527, 60, !dbg !16
  %address.local529 = add nuw nsw i64 %address.local528, %address.local527, !dbg !16
  %address.local530 = shl nuw nsw i64 %address.local529, 3, !dbg !16
  %address.local531 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local530, !dbg !16
  %address.local532 = and i64 %address.local527, 63, !dbg !14
  %address.local533 = lshr i64 %address.local532, 4, !dbg !16
  %address.local534 = mul nuw nsw i64 %address.local533, 384, !dbg !16
  %address.local535 = getelementptr [2 x i8], ptr addrspace(3) %address.local531, i64 %address.local534, !dbg !16
  %address.local536 = lshr i64 %address.local527, 2, !dbg !10
  %address.local537 = mul nsw i64 %address.local536, -32, !dbg !16
  %address.local538 = getelementptr [2 x i8], ptr addrspace(3) %address.local535, i64 %address.local537, !dbg !16
  %address.local539 = getelementptr i8, ptr addrspace(3) %address.local538, i64 256, !dbg !16
  %1070 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local539, i32 0) #0, !dbg !16
  %1071 = bitcast <4 x i32> %1070 to <8 x half>, !dbg !16
  %address.local540 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local530, !dbg !16
  %address.local541 = getelementptr [2 x i8], ptr addrspace(3) %address.local540, i64 %address.local534, !dbg !16
  %address.local542 = getelementptr [2 x i8], ptr addrspace(3) %address.local541, i64 %address.local537, !dbg !16
  %address.local543 = getelementptr i8, ptr addrspace(3) %address.local542, i64 512, !dbg !16
  %1072 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local543, i32 0) #0, !dbg !16
  %1073 = bitcast <4 x i32> %1072 to <8 x half>, !dbg !16
  %address.local544 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local530, !dbg !16
  %address.local545 = getelementptr [2 x i8], ptr addrspace(3) %address.local544, i64 %address.local534, !dbg !16
  %address.local546 = getelementptr [2 x i8], ptr addrspace(3) %address.local545, i64 %address.local537, !dbg !16
  %address.local547 = getelementptr i8, ptr addrspace(3) %address.local546, i64 768, !dbg !16
  %1074 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local547, i32 0) #0, !dbg !16
  %1075 = bitcast <4 x i32> %1074 to <8 x half>, !dbg !16
  %address.local548 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local525, !dbg !16
  %address.local549 = getelementptr i8, ptr addrspace(3) %address.local548, i64 4096, !dbg !16
  %1076 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local549, i32 0) #0, !dbg !16
  %1077 = bitcast <4 x i32> %1076 to <8 x half>, !dbg !16
  %address.local550 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local530, !dbg !16
  %address.local551 = getelementptr [2 x i8], ptr addrspace(3) %address.local550, i64 %address.local534, !dbg !16
  %address.local552 = getelementptr [2 x i8], ptr addrspace(3) %address.local551, i64 %address.local537, !dbg !16
  %address.local553 = getelementptr i8, ptr addrspace(3) %address.local552, i64 4352, !dbg !16
  %1078 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local553, i32 0) #0, !dbg !16
  %1079 = bitcast <4 x i32> %1078 to <8 x half>, !dbg !16
  %address.local554 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local530, !dbg !16
  %address.local555 = getelementptr [2 x i8], ptr addrspace(3) %address.local554, i64 %address.local534, !dbg !16
  %address.local556 = getelementptr [2 x i8], ptr addrspace(3) %address.local555, i64 %address.local537, !dbg !16
  %address.local557 = getelementptr i8, ptr addrspace(3) %address.local556, i64 4608, !dbg !16
  %1080 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local557, i32 0) #0, !dbg !16
  %1081 = bitcast <4 x i32> %1080 to <8 x half>, !dbg !16
  %address.local558 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local530, !dbg !16
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %address.local558, i64 %address.local534, !dbg !16
  %address.local560 = getelementptr [2 x i8], ptr addrspace(3) %address.local559, i64 %address.local537, !dbg !16
  %address.local561 = getelementptr i8, ptr addrspace(3) %address.local560, i64 4864, !dbg !16
  %1082 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local561, i32 0) #0, !dbg !16
  %1083 = bitcast <4 x i32> %1082 to <8 x half>, !dbg !16
  %1084 = add nsw i64 %96, %10, !dbg !19
  %1085 = shl i64 %1084, 19, !dbg !19
  %1086 = and i64 %1085, 16252928, !dbg !19
  %1087 = add i64 %1086, %102, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1088 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1087, !dbg !19
  %1089 = getelementptr i8, ptr addrspace(1) %1088, i64 64, !dbg !19
  %address.local562 = shl nuw nsw i64 %address.local527, 3, !dbg !13
  %address.local563 = and i64 %address.local562, 1536, !dbg !19
  %address.local564 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local563, !dbg !19
  %address.local565 = ptrtoint ptr addrspace(3) %address.local564 to i64, !dbg !19
  %address.local566 = trunc i64 %address.local565 to i32, !dbg !19
  %1090 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local566), !dbg !19
  %1091 = zext i32 %1090 to i64, !dbg !19
  %1092 = inttoptr i64 %1091 to ptr addrspace(3), !dbg !19
  %address.local567 = shl nuw nsw i64 %address.local527, 5, !dbg !19
  %address.local568 = and i64 %address.local567, 8064, !dbg !19
  %address.local569 = shl nuw nsw i64 %address.local527, 3, !dbg !19
  %address.local570 = and i64 %address.local569, 24, !dbg !19
  %address.local571 = or disjoint i64 %address.local568, %address.local570, !dbg !19
  %address.local572 = getelementptr [2 x i8], ptr addrspace(1) %1089, i64 %address.local571, !dbg !19
  %1093 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1094 = ptrtoint ptr addrspace(1) %address.local572 to i64, !dbg !19
  %1095 = sub i64 %1094, %1093, !dbg !19
  %dma.buffer.offset1104 = trunc i64 %1095 to i32, !dbg !19
  %1096 = trunc i64 %1093 to i32, !dbg !19
  %1097 = insertelement <4 x i32> poison, i32 %1096, i64 0, !dbg !19
  %1098 = lshr i64 %1093, 32, !dbg !19
  %1099 = trunc i64 %1098 to i32, !dbg !19
  %1100 = insertelement <4 x i32> %1097, i32 %1099, i64 1, !dbg !19
  %1101 = insertelement <4 x i32> %1100, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1105 = insertelement <4 x i32> %1101, i32 131072, i64 3, !dbg !19
  %1102 = ptrtoint ptr addrspace(3) %1092 to i32, !dbg !19
  %dma.buffer.lds1106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1102), !dbg !19
  %1103 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1106, i32 %dma.buffer.offset1104, <4 x i32> %dma.buffer.rsrc1105) #10, !dbg !19
  %address.local573 = or disjoint i64 %address.local527, 256, !dbg !13
  %address.local574 = shl nuw nsw i64 %address.local573, 3, !dbg !13
  %address.local575 = and i64 %address.local574, 3584, !dbg !19
  %address.local576 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local575, !dbg !19
  %address.local577 = ptrtoint ptr addrspace(3) %address.local576 to i64, !dbg !19
  %address.local578 = trunc i64 %address.local577 to i32, !dbg !19
  %1104 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local578), !dbg !19
  %1105 = zext i32 %1104 to i64, !dbg !19
  %1106 = inttoptr i64 %1105 to ptr addrspace(3), !dbg !19
  %address.local579 = shl nuw nsw i64 %address.local573, 5, !dbg !19
  %address.local580 = and i64 %address.local579, 16256, !dbg !19
  %address.local581 = shl nuw nsw i64 %address.local527, 3, !dbg !19
  %address.local582 = and i64 %address.local581, 24, !dbg !19
  %address.local583 = or disjoint i64 %address.local580, %address.local582, !dbg !19
  %address.local584 = getelementptr [2 x i8], ptr addrspace(1) %1089, i64 %address.local583, !dbg !19
  %1107 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1108 = ptrtoint ptr addrspace(1) %address.local584 to i64, !dbg !19
  %1109 = sub i64 %1108, %1107, !dbg !19
  %dma.buffer.offset1101 = trunc i64 %1109 to i32, !dbg !19
  %1110 = trunc i64 %1107 to i32, !dbg !19
  %1111 = insertelement <4 x i32> poison, i32 %1110, i64 0, !dbg !19
  %1112 = lshr i64 %1107, 32, !dbg !19
  %1113 = trunc i64 %1112 to i32, !dbg !19
  %1114 = insertelement <4 x i32> %1111, i32 %1113, i64 1, !dbg !19
  %1115 = insertelement <4 x i32> %1114, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1102 = insertelement <4 x i32> %1115, i32 131072, i64 3, !dbg !19
  %1116 = ptrtoint ptr addrspace(3) %1106 to i32, !dbg !19
  %dma.buffer.lds1103 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1116), !dbg !19
  %1117 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1103, i32 %dma.buffer.offset1101, <4 x i32> %dma.buffer.rsrc1102) #10, !dbg !19
  %1118 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1119 = extractvalue [2 x <8 x float>] %31, 0, !dbg !31
  %1120 = shufflevector <8 x float> %1119, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1118, <4 x float> %1120) #9, !dbg !16
  %1122 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1122, <4 x float> %1121) #9, !dbg !16
  %1124 = shufflevector <8 x half> %1073, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1124, <4 x float> %1123) #9, !dbg !16
  %1126 = shufflevector <8 x half> %1075, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1126, <4 x float> %1125) #9, !dbg !16
  %1128 = shufflevector <8 x half> %1077, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1128, <4 x float> %1127) #9, !dbg !16
  %1130 = shufflevector <8 x half> %1079, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1130, <4 x float> %1129) #9, !dbg !16
  %1132 = shufflevector <8 x half> %1081, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1132, <4 x float> %1131) #9, !dbg !16
  %1134 = shufflevector <8 x half> %1083, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1134, <4 x float> %1133) #9, !dbg !16
  %1136 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1137 = shufflevector <8 x float> %1119, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1136, <4 x float> %1137) #9, !dbg !16
  %1139 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1139, <4 x float> %1138) #9, !dbg !16
  %1141 = shufflevector <8 x half> %1073, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1141, <4 x float> %1140) #9, !dbg !16
  %1143 = shufflevector <8 x half> %1075, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1143, <4 x float> %1142) #9, !dbg !16
  %1145 = shufflevector <8 x half> %1077, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1145, <4 x float> %1144) #9, !dbg !16
  %1147 = shufflevector <8 x half> %1079, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1147, <4 x float> %1146) #9, !dbg !16
  %1149 = shufflevector <8 x half> %1081, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1149, <4 x float> %1148) #9, !dbg !16
  %1151 = shufflevector <8 x half> %1083, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1151, <4 x float> %1150) #9, !dbg !16
  %1153 = extractvalue [2 x <8 x float>] %31, 1, !dbg !31
  %1154 = shufflevector <8 x float> %1153, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1118, <4 x float> %1154) #9, !dbg !16
  %1156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1122, <4 x float> %1155) #9, !dbg !16
  %1157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1124, <4 x float> %1156) #9, !dbg !16
  %1158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1126, <4 x float> %1157) #9, !dbg !16
  %1159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1128, <4 x float> %1158) #9, !dbg !16
  %1160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1130, <4 x float> %1159) #9, !dbg !16
  %1161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1132, <4 x float> %1160) #9, !dbg !16
  %1162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1134, <4 x float> %1161) #9, !dbg !16
  %1163 = shufflevector <8 x float> %1153, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1136, <4 x float> %1163) #9, !dbg !16
  %1165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1139, <4 x float> %1164) #9, !dbg !16
  %1166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1141, <4 x float> %1165) #9, !dbg !16
  %1167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1143, <4 x float> %1166) #9, !dbg !16
  %1168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1145, <4 x float> %1167) #9, !dbg !16
  %1169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1147, <4 x float> %1168) #9, !dbg !16
  %1170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1149, <4 x float> %1169) #9, !dbg !16
  %1171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1151, <4 x float> %1170) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1172 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local526, i32 0) #0, !dbg !16
  %1173 = bitcast <4 x i32> %1172 to <8 x half>, !dbg !16
  %1174 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local539, i32 0) #0, !dbg !16
  %1175 = bitcast <4 x i32> %1174 to <8 x half>, !dbg !16
  %1176 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local543, i32 0) #0, !dbg !16
  %1177 = bitcast <4 x i32> %1176 to <8 x half>, !dbg !16
  %1178 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local547, i32 0) #0, !dbg !16
  %1179 = bitcast <4 x i32> %1178 to <8 x half>, !dbg !16
  %1180 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local549, i32 0) #0, !dbg !16
  %1181 = bitcast <4 x i32> %1180 to <8 x half>, !dbg !16
  %1182 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local553, i32 0) #0, !dbg !16
  %1183 = bitcast <4 x i32> %1182 to <8 x half>, !dbg !16
  %1184 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local557, i32 0) #0, !dbg !16
  %1185 = bitcast <4 x i32> %1184 to <8 x half>, !dbg !16
  %1186 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local561, i32 0) #0, !dbg !16
  %1187 = bitcast <4 x i32> %1186 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1188 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1087, !dbg !19
  %1189 = getelementptr i8, ptr addrspace(1) %1188, i64 128, !dbg !19
  %1190 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local566), !dbg !19
  %1191 = zext i32 %1190 to i64, !dbg !19
  %1192 = inttoptr i64 %1191 to ptr addrspace(3), !dbg !19
  %address.local585 = getelementptr [2 x i8], ptr addrspace(1) %1189, i64 %address.local571, !dbg !19
  %1193 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1194 = ptrtoint ptr addrspace(1) %address.local585 to i64, !dbg !19
  %1195 = sub i64 %1194, %1193, !dbg !19
  %dma.buffer.offset1098 = trunc i64 %1195 to i32, !dbg !19
  %1196 = trunc i64 %1193 to i32, !dbg !19
  %1197 = insertelement <4 x i32> poison, i32 %1196, i64 0, !dbg !19
  %1198 = lshr i64 %1193, 32, !dbg !19
  %1199 = trunc i64 %1198 to i32, !dbg !19
  %1200 = insertelement <4 x i32> %1197, i32 %1199, i64 1, !dbg !19
  %1201 = insertelement <4 x i32> %1200, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1099 = insertelement <4 x i32> %1201, i32 131072, i64 3, !dbg !19
  %1202 = ptrtoint ptr addrspace(3) %1192 to i32, !dbg !19
  %dma.buffer.lds1100 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1202), !dbg !19
  %1203 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1100, i32 %dma.buffer.offset1098, <4 x i32> %dma.buffer.rsrc1099) #10, !dbg !19
  %1204 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local578), !dbg !19
  %1205 = zext i32 %1204 to i64, !dbg !19
  %1206 = inttoptr i64 %1205 to ptr addrspace(3), !dbg !19
  %address.local586 = getelementptr [2 x i8], ptr addrspace(1) %1189, i64 %address.local583, !dbg !19
  %1207 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1208 = ptrtoint ptr addrspace(1) %address.local586 to i64, !dbg !19
  %1209 = sub i64 %1208, %1207, !dbg !19
  %dma.buffer.offset1095 = trunc i64 %1209 to i32, !dbg !19
  %1210 = trunc i64 %1207 to i32, !dbg !19
  %1211 = insertelement <4 x i32> poison, i32 %1210, i64 0, !dbg !19
  %1212 = lshr i64 %1207, 32, !dbg !19
  %1213 = trunc i64 %1212 to i32, !dbg !19
  %1214 = insertelement <4 x i32> %1211, i32 %1213, i64 1, !dbg !19
  %1215 = insertelement <4 x i32> %1214, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1096 = insertelement <4 x i32> %1215, i32 131072, i64 3, !dbg !19
  %1216 = ptrtoint ptr addrspace(3) %1206 to i32, !dbg !19
  %dma.buffer.lds1097 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1216), !dbg !19
  %1217 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1097, i32 %dma.buffer.offset1095, <4 x i32> %dma.buffer.rsrc1096) #10, !dbg !19
  %1218 = shufflevector <8 x half> %1173, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1219 = extractvalue [2 x <8 x float>] %33, 0, !dbg !31
  %1220 = shufflevector <8 x float> %1219, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1218, <4 x float> %1220) #9, !dbg !16
  %1222 = shufflevector <8 x half> %1175, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1222, <4 x float> %1221) #9, !dbg !16
  %1224 = shufflevector <8 x half> %1177, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1224, <4 x float> %1223) #9, !dbg !16
  %1226 = shufflevector <8 x half> %1179, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1226, <4 x float> %1225) #9, !dbg !16
  %1228 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1228, <4 x float> %1227) #9, !dbg !16
  %1230 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1230, <4 x float> %1229) #9, !dbg !16
  %1232 = shufflevector <8 x half> %1185, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1232, <4 x float> %1231) #9, !dbg !16
  %1234 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1234, <4 x float> %1233) #9, !dbg !16
  %1236 = shufflevector <8 x half> %1173, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1237 = shufflevector <8 x float> %1219, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1236, <4 x float> %1237) #9, !dbg !16
  %1239 = shufflevector <8 x half> %1175, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1239, <4 x float> %1238) #9, !dbg !16
  %1241 = shufflevector <8 x half> %1177, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1241, <4 x float> %1240) #9, !dbg !16
  %1243 = shufflevector <8 x half> %1179, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1243, <4 x float> %1242) #9, !dbg !16
  %1245 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1245, <4 x float> %1244) #9, !dbg !16
  %1247 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1247, <4 x float> %1246) #9, !dbg !16
  %1249 = shufflevector <8 x half> %1185, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1249, <4 x float> %1248) #9, !dbg !16
  %1251 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1251, <4 x float> %1250) #9, !dbg !16
  %1253 = extractvalue [2 x <8 x float>] %33, 1, !dbg !31
  %1254 = shufflevector <8 x float> %1253, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1218, <4 x float> %1254) #9, !dbg !16
  %1256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1222, <4 x float> %1255) #9, !dbg !16
  %1257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1224, <4 x float> %1256) #9, !dbg !16
  %1258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1226, <4 x float> %1257) #9, !dbg !16
  %1259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1228, <4 x float> %1258) #9, !dbg !16
  %1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1230, <4 x float> %1259) #9, !dbg !16
  %1261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1232, <4 x float> %1260) #9, !dbg !16
  %1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1234, <4 x float> %1261) #9, !dbg !16
  %1263 = shufflevector <8 x float> %1253, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1236, <4 x float> %1263) #9, !dbg !16
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1239, <4 x float> %1264) #9, !dbg !16
  %1266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1241, <4 x float> %1265) #9, !dbg !16
  %1267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1243, <4 x float> %1266) #9, !dbg !16
  %1268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1245, <4 x float> %1267) #9, !dbg !16
  %1269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1247, <4 x float> %1268) #9, !dbg !16
  %1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1249, <4 x float> %1269) #9, !dbg !16
  %1271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1251, <4 x float> %1270) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1272 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local526, i32 0) #0, !dbg !16
  %1273 = bitcast <4 x i32> %1272 to <8 x half>, !dbg !16
  %1274 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local539, i32 0) #0, !dbg !16
  %1275 = bitcast <4 x i32> %1274 to <8 x half>, !dbg !16
  %1276 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local543, i32 0) #0, !dbg !16
  %1277 = bitcast <4 x i32> %1276 to <8 x half>, !dbg !16
  %1278 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local547, i32 0) #0, !dbg !16
  %1279 = bitcast <4 x i32> %1278 to <8 x half>, !dbg !16
  %1280 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local549, i32 0) #0, !dbg !16
  %1281 = bitcast <4 x i32> %1280 to <8 x half>, !dbg !16
  %1282 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local553, i32 0) #0, !dbg !16
  %1283 = bitcast <4 x i32> %1282 to <8 x half>, !dbg !16
  %1284 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local557, i32 0) #0, !dbg !16
  %1285 = bitcast <4 x i32> %1284 to <8 x half>, !dbg !16
  %1286 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local561, i32 0) #0, !dbg !16
  %1287 = bitcast <4 x i32> %1286 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1288 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1087, !dbg !19
  %1289 = getelementptr i8, ptr addrspace(1) %1288, i64 192, !dbg !19
  %1290 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local566), !dbg !19
  %1291 = zext i32 %1290 to i64, !dbg !19
  %1292 = inttoptr i64 %1291 to ptr addrspace(3), !dbg !19
  %address.local587 = getelementptr [2 x i8], ptr addrspace(1) %1289, i64 %address.local571, !dbg !19
  %1293 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1294 = ptrtoint ptr addrspace(1) %address.local587 to i64, !dbg !19
  %1295 = sub i64 %1294, %1293, !dbg !19
  %dma.buffer.offset1092 = trunc i64 %1295 to i32, !dbg !19
  %1296 = trunc i64 %1293 to i32, !dbg !19
  %1297 = insertelement <4 x i32> poison, i32 %1296, i64 0, !dbg !19
  %1298 = lshr i64 %1293, 32, !dbg !19
  %1299 = trunc i64 %1298 to i32, !dbg !19
  %1300 = insertelement <4 x i32> %1297, i32 %1299, i64 1, !dbg !19
  %1301 = insertelement <4 x i32> %1300, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1093 = insertelement <4 x i32> %1301, i32 131072, i64 3, !dbg !19
  %1302 = ptrtoint ptr addrspace(3) %1292 to i32, !dbg !19
  %dma.buffer.lds1094 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1302), !dbg !19
  %1303 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1094, i32 %dma.buffer.offset1092, <4 x i32> %dma.buffer.rsrc1093) #10, !dbg !19
  %1304 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local578), !dbg !19
  %1305 = zext i32 %1304 to i64, !dbg !19
  %1306 = inttoptr i64 %1305 to ptr addrspace(3), !dbg !19
  %address.local588 = getelementptr [2 x i8], ptr addrspace(1) %1289, i64 %address.local583, !dbg !19
  %1307 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1308 = ptrtoint ptr addrspace(1) %address.local588 to i64, !dbg !19
  %1309 = sub i64 %1308, %1307, !dbg !19
  %dma.buffer.offset1089 = trunc i64 %1309 to i32, !dbg !19
  %1310 = trunc i64 %1307 to i32, !dbg !19
  %1311 = insertelement <4 x i32> poison, i32 %1310, i64 0, !dbg !19
  %1312 = lshr i64 %1307, 32, !dbg !19
  %1313 = trunc i64 %1312 to i32, !dbg !19
  %1314 = insertelement <4 x i32> %1311, i32 %1313, i64 1, !dbg !19
  %1315 = insertelement <4 x i32> %1314, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1090 = insertelement <4 x i32> %1315, i32 131072, i64 3, !dbg !19
  %1316 = ptrtoint ptr addrspace(3) %1306 to i32, !dbg !19
  %dma.buffer.lds1091 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1316), !dbg !19
  %1317 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1091, i32 %dma.buffer.offset1089, <4 x i32> %dma.buffer.rsrc1090) #10, !dbg !19
  %1318 = shufflevector <8 x half> %1273, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1319 = extractvalue [2 x <8 x float>] %34, 0, !dbg !31
  %1320 = shufflevector <8 x float> %1319, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1318, <4 x float> %1320) #9, !dbg !16
  %1322 = shufflevector <8 x half> %1275, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1322, <4 x float> %1321) #9, !dbg !16
  %1324 = shufflevector <8 x half> %1277, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1324, <4 x float> %1323) #9, !dbg !16
  %1326 = shufflevector <8 x half> %1279, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1326, <4 x float> %1325) #9, !dbg !16
  %1328 = shufflevector <8 x half> %1281, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1328, <4 x float> %1327) #9, !dbg !16
  %1330 = shufflevector <8 x half> %1283, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1330, <4 x float> %1329) #9, !dbg !16
  %1332 = shufflevector <8 x half> %1285, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1332, <4 x float> %1331) #9, !dbg !16
  %1334 = shufflevector <8 x half> %1287, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1334, <4 x float> %1333) #9, !dbg !16
  %1336 = shufflevector <8 x half> %1273, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1337 = shufflevector <8 x float> %1319, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1336, <4 x float> %1337) #9, !dbg !16
  %1339 = shufflevector <8 x half> %1275, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1339, <4 x float> %1338) #9, !dbg !16
  %1341 = shufflevector <8 x half> %1277, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1341, <4 x float> %1340) #9, !dbg !16
  %1343 = shufflevector <8 x half> %1279, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1343, <4 x float> %1342) #9, !dbg !16
  %1345 = shufflevector <8 x half> %1281, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1345, <4 x float> %1344) #9, !dbg !16
  %1347 = shufflevector <8 x half> %1283, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1347, <4 x float> %1346) #9, !dbg !16
  %1349 = shufflevector <8 x half> %1285, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1349, <4 x float> %1348) #9, !dbg !16
  %1351 = shufflevector <8 x half> %1287, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1351, <4 x float> %1350) #9, !dbg !16
  %1353 = extractvalue [2 x <8 x float>] %34, 1, !dbg !31
  %1354 = shufflevector <8 x float> %1353, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1318, <4 x float> %1354) #9, !dbg !16
  %1356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1322, <4 x float> %1355) #9, !dbg !16
  %1357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1324, <4 x float> %1356) #9, !dbg !16
  %1358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1326, <4 x float> %1357) #9, !dbg !16
  %1359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1328, <4 x float> %1358) #9, !dbg !16
  %1360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1330, <4 x float> %1359) #9, !dbg !16
  %1361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1332, <4 x float> %1360) #9, !dbg !16
  %1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1334, <4 x float> %1361) #9, !dbg !16
  %1363 = shufflevector <8 x float> %1353, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1336, <4 x float> %1363) #9, !dbg !16
  %1365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1339, <4 x float> %1364) #9, !dbg !16
  %1366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1341, <4 x float> %1365) #9, !dbg !16
  %1367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1343, <4 x float> %1366) #9, !dbg !16
  %1368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1345, <4 x float> %1367) #9, !dbg !16
  %1369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1347, <4 x float> %1368) #9, !dbg !16
  %1370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1349, <4 x float> %1369) #9, !dbg !16
  %1371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1351, <4 x float> %1370) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1372 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local526, i32 0) #0, !dbg !16
  %1373 = bitcast <4 x i32> %1372 to <8 x half>, !dbg !16
  %1374 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local539, i32 0) #0, !dbg !16
  %1375 = bitcast <4 x i32> %1374 to <8 x half>, !dbg !16
  %1376 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local543, i32 0) #0, !dbg !16
  %1377 = bitcast <4 x i32> %1376 to <8 x half>, !dbg !16
  %1378 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local547, i32 0) #0, !dbg !16
  %1379 = bitcast <4 x i32> %1378 to <8 x half>, !dbg !16
  %1380 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local549, i32 0) #0, !dbg !16
  %1381 = bitcast <4 x i32> %1380 to <8 x half>, !dbg !16
  %1382 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local553, i32 0) #0, !dbg !16
  %1383 = bitcast <4 x i32> %1382 to <8 x half>, !dbg !16
  %1384 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local557, i32 0) #0, !dbg !16
  %1385 = bitcast <4 x i32> %1384 to <8 x half>, !dbg !16
  %1386 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local561, i32 0) #0, !dbg !16
  %1387 = bitcast <4 x i32> %1386 to <8 x half>, !dbg !16
  %1388 = shufflevector <8 x half> %1373, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1389 = extractvalue [2 x <8 x float>] %35, 0, !dbg !31
  %1390 = shufflevector <8 x float> %1389, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1388, <4 x float> %1390) #9, !dbg !16
  %1392 = shufflevector <8 x half> %1375, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1392, <4 x float> %1391) #9, !dbg !16
  %1394 = shufflevector <8 x half> %1377, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1394, <4 x float> %1393) #9, !dbg !16
  %1396 = shufflevector <8 x half> %1379, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1396, <4 x float> %1395) #9, !dbg !16
  %1398 = shufflevector <8 x half> %1381, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1398, <4 x float> %1397) #9, !dbg !16
  %1400 = shufflevector <8 x half> %1383, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1400, <4 x float> %1399) #9, !dbg !16
  %1402 = shufflevector <8 x half> %1385, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1402, <4 x float> %1401) #9, !dbg !16
  %1404 = shufflevector <8 x half> %1387, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1404, <4 x float> %1403) #9, !dbg !16
  %1406 = shufflevector <8 x half> %1373, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1407 = shufflevector <8 x float> %1389, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1406, <4 x float> %1407) #9, !dbg !16
  %1409 = shufflevector <8 x half> %1375, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1409, <4 x float> %1408) #9, !dbg !16
  %1411 = shufflevector <8 x half> %1377, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1411, <4 x float> %1410) #9, !dbg !16
  %1413 = shufflevector <8 x half> %1379, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1413, <4 x float> %1412) #9, !dbg !16
  %1415 = shufflevector <8 x half> %1381, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1416 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1415, <4 x float> %1414) #9, !dbg !16
  %1417 = shufflevector <8 x half> %1383, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1417, <4 x float> %1416) #9, !dbg !16
  %1419 = shufflevector <8 x half> %1385, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1420 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1419, <4 x float> %1418) #9, !dbg !16
  %1421 = shufflevector <8 x half> %1387, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1421, <4 x float> %1420) #9, !dbg !16
  %1423 = extractvalue [2 x <8 x float>] %35, 1, !dbg !31
  %1424 = shufflevector <8 x float> %1423, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1388, <4 x float> %1424) #9, !dbg !16
  %1426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1392, <4 x float> %1425) #9, !dbg !16
  %1427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1394, <4 x float> %1426) #9, !dbg !16
  %1428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1396, <4 x float> %1427) #9, !dbg !16
  %1429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1398, <4 x float> %1428) #9, !dbg !16
  %1430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1400, <4 x float> %1429) #9, !dbg !16
  %1431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1402, <4 x float> %1430) #9, !dbg !16
  %1432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1404, <4 x float> %1431) #9, !dbg !16
  %1433 = shufflevector <8 x float> %1423, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1406, <4 x float> %1433) #9, !dbg !16
  %1435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1409, <4 x float> %1434) #9, !dbg !16
  %1436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1411, <4 x float> %1435) #9, !dbg !16
  %1437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1413, <4 x float> %1436) #9, !dbg !16
  %1438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %1415, <4 x float> %1437) #9, !dbg !16
  %1439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %1417, <4 x float> %1438) #9, !dbg !16
  %1440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed138, <4 x half> %1419, <4 x float> %1439) #9, !dbg !16
  %1441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed140, <4 x half> %1421, <4 x float> %1440) #9, !dbg !16
  %1442 = shufflevector <4 x float> %1135, <4 x float> %1152, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1443 = insertvalue [2 x <8 x float>] poison, <8 x float> %1442, 0, !dbg !17
  %1444 = shufflevector <4 x float> %1162, <4 x float> %1171, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1445 = insertvalue [2 x <8 x float>] %1443, <8 x float> %1444, 1, !dbg !17
  %1446 = shufflevector <4 x float> %1235, <4 x float> %1252, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1447 = insertvalue [2 x <8 x float>] poison, <8 x float> %1446, 0, !dbg !17
  %1448 = shufflevector <4 x float> %1262, <4 x float> %1271, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1449 = insertvalue [2 x <8 x float>] %1447, <8 x float> %1448, 1, !dbg !17
  %1450 = shufflevector <4 x float> %1335, <4 x float> %1352, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1451 = insertvalue [2 x <8 x float>] poison, <8 x float> %1450, 0, !dbg !17
  %1452 = shufflevector <4 x float> %1362, <4 x float> %1371, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1453 = insertvalue [2 x <8 x float>] %1451, <8 x float> %1452, 1, !dbg !17
  %1454 = shufflevector <4 x float> %1405, <4 x float> %1422, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1455 = insertvalue [2 x <8 x float>] poison, <8 x float> %1454, 0, !dbg !17
  %1456 = shufflevector <4 x float> %1432, <4 x float> %1441, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1457 = insertvalue [2 x <8 x float>] %1455, <8 x float> %1456, 1, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1458 = add nuw nsw i64 %30, 1, !dbg !17
  br label %29, !dbg !17

1459:                                             ; preds = %29
  %1460 = and i32 %21, 16, !dbg !17
  %.not3 = icmp eq i32 %1460, 0, !dbg !17
  %1461 = and i32 %21, 32, !dbg !17
  %.not4 = icmp eq i32 %1461, 0, !dbg !17
  %1462 = extractvalue [2 x <8 x float>] %35, 0, !dbg !17
  %1463 = extractelement <8 x float> %1462, i64 0, !dbg !17
  %1464 = extractelement <8 x float> %1462, i64 1, !dbg !17
  %1465 = extractelement <8 x float> %1462, i64 2, !dbg !17
  %1466 = extractelement <8 x float> %1462, i64 3, !dbg !17
  %1467 = select i1 %.not3, float %1464, float %1463, !dbg !17
  %1468 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !17
  %1469 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1468), !dbg !17
  %1470 = add nuw nsw i32 %1469, 64, !dbg !17
  %1471 = xor i32 %1469, 16, !dbg !17
  %1472 = and i32 %1470, 192, !dbg !17
  %1473 = icmp samesign ult i32 %1471, %1472, !dbg !17
  %1474 = select i1 %1473, i32 %1471, i32 %1469, !dbg !17
  %1475 = shl nuw nsw i32 %1474, 2, !dbg !17
  %1476 = bitcast float %1467 to i32, !dbg !17
  %1477 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1476), !dbg !17
  %1478 = bitcast i32 %1477 to float, !dbg !17
  %1479 = select i1 %.not3, float %1463, float %1478, !dbg !17
  %1480 = select i1 %.not3, float %1478, float %1464, !dbg !17
  %1481 = select i1 %.not3, float %1466, float %1465, !dbg !17
  %1482 = bitcast float %1481 to i32, !dbg !17
  %1483 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1482), !dbg !17
  %1484 = bitcast i32 %1483 to float, !dbg !17
  %1485 = select i1 %.not3, float %1465, float %1484, !dbg !17
  %1486 = select i1 %.not3, float %1484, float %1466, !dbg !17
  %1487 = select i1 %.not4, float %1485, float %1479, !dbg !17
  %1488 = xor i32 %1469, 32, !dbg !17
  %1489 = icmp samesign ult i32 %1488, %1472, !dbg !17
  %1490 = select i1 %1489, i32 %1488, i32 %1469, !dbg !17
  %1491 = shl nuw nsw i32 %1490, 2, !dbg !17
  %1492 = bitcast float %1487 to i32, !dbg !17
  %1493 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1492), !dbg !17
  %1494 = bitcast i32 %1493 to float, !dbg !17
  %1495 = select i1 %.not4, float %1479, float %1494, !dbg !17
  %1496 = select i1 %.not4, float %1494, float %1485, !dbg !17
  %1497 = select i1 %.not4, float %1486, float %1480, !dbg !17
  %1498 = bitcast float %1497 to i32, !dbg !17
  %1499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1498), !dbg !17
  %1500 = bitcast i32 %1499 to float, !dbg !17
  %1501 = select i1 %.not4, float %1480, float %1500, !dbg !17
  %1502 = select i1 %.not4, float %1500, float %1486, !dbg !17
  %1503 = extractvalue [2 x <8 x float>] %34, 0, !dbg !17
  %1504 = extractelement <8 x float> %1503, i64 0, !dbg !17
  %1505 = extractelement <8 x float> %1503, i64 1, !dbg !17
  %1506 = extractelement <8 x float> %1503, i64 2, !dbg !17
  %1507 = extractelement <8 x float> %1503, i64 3, !dbg !17
  %1508 = select i1 %.not3, float %1505, float %1504, !dbg !17
  %1509 = bitcast float %1508 to i32, !dbg !17
  %1510 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1509), !dbg !17
  %1511 = bitcast i32 %1510 to float, !dbg !17
  %1512 = select i1 %.not3, float %1504, float %1511, !dbg !17
  %1513 = select i1 %.not3, float %1511, float %1505, !dbg !17
  %1514 = select i1 %.not3, float %1507, float %1506, !dbg !17
  %1515 = bitcast float %1514 to i32, !dbg !17
  %1516 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1515), !dbg !17
  %1517 = bitcast i32 %1516 to float, !dbg !17
  %1518 = select i1 %.not3, float %1506, float %1517, !dbg !17
  %1519 = select i1 %.not3, float %1517, float %1507, !dbg !17
  %1520 = select i1 %.not4, float %1518, float %1512, !dbg !17
  %1521 = bitcast float %1520 to i32, !dbg !17
  %1522 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1521), !dbg !17
  %1523 = bitcast i32 %1522 to float, !dbg !17
  %1524 = select i1 %.not4, float %1512, float %1523, !dbg !17
  %1525 = select i1 %.not4, float %1523, float %1518, !dbg !17
  %1526 = select i1 %.not4, float %1519, float %1513, !dbg !17
  %1527 = bitcast float %1526 to i32, !dbg !17
  %1528 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1527), !dbg !17
  %1529 = bitcast i32 %1528 to float, !dbg !17
  %1530 = select i1 %.not4, float %1513, float %1529, !dbg !17
  %1531 = select i1 %.not4, float %1529, float %1519, !dbg !17
  %1532 = extractvalue [2 x <8 x float>] %33, 0, !dbg !17
  %1533 = extractelement <8 x float> %1532, i64 0, !dbg !17
  %1534 = extractelement <8 x float> %1532, i64 1, !dbg !17
  %1535 = extractelement <8 x float> %1532, i64 2, !dbg !17
  %1536 = extractelement <8 x float> %1532, i64 3, !dbg !17
  %1537 = select i1 %.not3, float %1534, float %1533, !dbg !17
  %1538 = bitcast float %1537 to i32, !dbg !17
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1538), !dbg !17
  %1540 = bitcast i32 %1539 to float, !dbg !17
  %1541 = select i1 %.not3, float %1533, float %1540, !dbg !17
  %1542 = select i1 %.not3, float %1540, float %1534, !dbg !17
  %1543 = select i1 %.not3, float %1536, float %1535, !dbg !17
  %1544 = bitcast float %1543 to i32, !dbg !17
  %1545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1544), !dbg !17
  %1546 = bitcast i32 %1545 to float, !dbg !17
  %1547 = select i1 %.not3, float %1535, float %1546, !dbg !17
  %1548 = select i1 %.not3, float %1546, float %1536, !dbg !17
  %1549 = select i1 %.not4, float %1547, float %1541, !dbg !17
  %1550 = bitcast float %1549 to i32, !dbg !17
  %1551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1550), !dbg !17
  %1552 = bitcast i32 %1551 to float, !dbg !17
  %1553 = select i1 %.not4, float %1541, float %1552, !dbg !17
  %1554 = select i1 %.not4, float %1552, float %1547, !dbg !17
  %1555 = select i1 %.not4, float %1548, float %1542, !dbg !17
  %1556 = bitcast float %1555 to i32, !dbg !17
  %1557 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1556), !dbg !17
  %1558 = bitcast i32 %1557 to float, !dbg !17
  %1559 = select i1 %.not4, float %1542, float %1558, !dbg !17
  %1560 = select i1 %.not4, float %1558, float %1548, !dbg !17
  %1561 = extractvalue [2 x <8 x float>] %31, 0, !dbg !17
  %1562 = extractelement <8 x float> %1561, i64 0, !dbg !17
  %1563 = extractelement <8 x float> %1561, i64 1, !dbg !17
  %1564 = extractelement <8 x float> %1561, i64 2, !dbg !17
  %1565 = extractelement <8 x float> %1561, i64 3, !dbg !17
  %1566 = select i1 %.not3, float %1563, float %1562, !dbg !17
  %1567 = bitcast float %1566 to i32, !dbg !17
  %1568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1567), !dbg !17
  %1569 = bitcast i32 %1568 to float, !dbg !17
  %1570 = select i1 %.not3, float %1562, float %1569, !dbg !17
  %1571 = select i1 %.not3, float %1569, float %1563, !dbg !17
  %1572 = select i1 %.not3, float %1565, float %1564, !dbg !17
  %1573 = bitcast float %1572 to i32, !dbg !17
  %1574 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1573), !dbg !17
  %1575 = bitcast i32 %1574 to float, !dbg !17
  %1576 = select i1 %.not3, float %1564, float %1575, !dbg !17
  %1577 = select i1 %.not3, float %1575, float %1565, !dbg !17
  %1578 = select i1 %.not4, float %1576, float %1570, !dbg !17
  %1579 = bitcast float %1578 to i32, !dbg !17
  %1580 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1579), !dbg !17
  %1581 = select i1 %.not4, float %1577, float %1571, !dbg !17
  %1582 = bitcast float %1581 to i32, !dbg !17
  %1583 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1582), !dbg !17
  %1584 = extractvalue [2 x <1 x float>] %32, 0, !dbg !20
  %1585 = shufflevector <1 x float> %1584, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1586 = extractelement <8 x float> %1462, i64 4, !dbg !17
  %1587 = extractelement <8 x float> %1462, i64 5, !dbg !17
  %1588 = extractelement <8 x float> %1462, i64 6, !dbg !17
  %1589 = extractelement <8 x float> %1462, i64 7, !dbg !17
  %1590 = select i1 %.not3, float %1587, float %1586, !dbg !17
  %1591 = bitcast float %1590 to i32, !dbg !17
  %1592 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1591), !dbg !17
  %1593 = bitcast i32 %1592 to float, !dbg !17
  %1594 = select i1 %.not3, float %1586, float %1593, !dbg !17
  %1595 = select i1 %.not3, float %1593, float %1587, !dbg !17
  %1596 = select i1 %.not3, float %1589, float %1588, !dbg !17
  %1597 = bitcast float %1596 to i32, !dbg !17
  %1598 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1597), !dbg !17
  %1599 = bitcast i32 %1598 to float, !dbg !17
  %1600 = select i1 %.not3, float %1588, float %1599, !dbg !17
  %1601 = select i1 %.not3, float %1599, float %1589, !dbg !17
  %1602 = select i1 %.not4, float %1600, float %1594, !dbg !17
  %1603 = bitcast float %1602 to i32, !dbg !17
  %1604 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1603), !dbg !17
  %1605 = bitcast i32 %1604 to float, !dbg !17
  %1606 = select i1 %.not4, float %1594, float %1605, !dbg !17
  %1607 = select i1 %.not4, float %1605, float %1600, !dbg !17
  %1608 = select i1 %.not4, float %1601, float %1595, !dbg !17
  %1609 = bitcast float %1608 to i32, !dbg !17
  %1610 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1609), !dbg !17
  %1611 = bitcast i32 %1610 to float, !dbg !17
  %1612 = select i1 %.not4, float %1595, float %1611, !dbg !17
  %1613 = select i1 %.not4, float %1611, float %1601, !dbg !17
  %1614 = extractelement <8 x float> %1503, i64 4, !dbg !17
  %1615 = extractelement <8 x float> %1503, i64 5, !dbg !17
  %1616 = extractelement <8 x float> %1503, i64 6, !dbg !17
  %1617 = extractelement <8 x float> %1503, i64 7, !dbg !17
  %1618 = select i1 %.not3, float %1615, float %1614, !dbg !17
  %1619 = bitcast float %1618 to i32, !dbg !17
  %1620 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1619), !dbg !17
  %1621 = bitcast i32 %1620 to float, !dbg !17
  %1622 = select i1 %.not3, float %1614, float %1621, !dbg !17
  %1623 = select i1 %.not3, float %1621, float %1615, !dbg !17
  %1624 = select i1 %.not3, float %1617, float %1616, !dbg !17
  %1625 = bitcast float %1624 to i32, !dbg !17
  %1626 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1625), !dbg !17
  %1627 = bitcast i32 %1626 to float, !dbg !17
  %1628 = select i1 %.not3, float %1616, float %1627, !dbg !17
  %1629 = select i1 %.not3, float %1627, float %1617, !dbg !17
  %1630 = select i1 %.not4, float %1628, float %1622, !dbg !17
  %1631 = bitcast float %1630 to i32, !dbg !17
  %1632 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1631), !dbg !17
  %1633 = bitcast i32 %1632 to float, !dbg !17
  %1634 = select i1 %.not4, float %1622, float %1633, !dbg !17
  %1635 = select i1 %.not4, float %1633, float %1628, !dbg !17
  %1636 = select i1 %.not4, float %1629, float %1623, !dbg !17
  %1637 = bitcast float %1636 to i32, !dbg !17
  %1638 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1637), !dbg !17
  %1639 = bitcast i32 %1638 to float, !dbg !17
  %1640 = select i1 %.not4, float %1623, float %1639, !dbg !17
  %1641 = select i1 %.not4, float %1639, float %1629, !dbg !17
  %1642 = extractelement <8 x float> %1532, i64 4, !dbg !17
  %1643 = extractelement <8 x float> %1532, i64 5, !dbg !17
  %1644 = extractelement <8 x float> %1532, i64 6, !dbg !17
  %1645 = extractelement <8 x float> %1532, i64 7, !dbg !17
  %1646 = select i1 %.not3, float %1643, float %1642, !dbg !17
  %1647 = bitcast float %1646 to i32, !dbg !17
  %1648 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1647), !dbg !17
  %1649 = bitcast i32 %1648 to float, !dbg !17
  %1650 = select i1 %.not3, float %1642, float %1649, !dbg !17
  %1651 = select i1 %.not3, float %1649, float %1643, !dbg !17
  %1652 = select i1 %.not3, float %1645, float %1644, !dbg !17
  %1653 = bitcast float %1652 to i32, !dbg !17
  %1654 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1653), !dbg !17
  %1655 = bitcast i32 %1654 to float, !dbg !17
  %1656 = select i1 %.not3, float %1644, float %1655, !dbg !17
  %1657 = select i1 %.not3, float %1655, float %1645, !dbg !17
  %1658 = select i1 %.not4, float %1656, float %1650, !dbg !17
  %1659 = bitcast float %1658 to i32, !dbg !17
  %1660 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1659), !dbg !17
  %1661 = bitcast i32 %1660 to float, !dbg !17
  %1662 = select i1 %.not4, float %1650, float %1661, !dbg !17
  %1663 = select i1 %.not4, float %1661, float %1656, !dbg !17
  %1664 = select i1 %.not4, float %1657, float %1651, !dbg !17
  %1665 = bitcast float %1664 to i32, !dbg !17
  %1666 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1665), !dbg !17
  %1667 = bitcast i32 %1666 to float, !dbg !17
  %1668 = select i1 %.not4, float %1651, float %1667, !dbg !17
  %1669 = select i1 %.not4, float %1667, float %1657, !dbg !17
  %1670 = extractelement <8 x float> %1561, i64 4, !dbg !17
  %1671 = extractelement <8 x float> %1561, i64 5, !dbg !17
  %1672 = extractelement <8 x float> %1561, i64 6, !dbg !17
  %1673 = extractelement <8 x float> %1561, i64 7, !dbg !17
  %1674 = select i1 %.not3, float %1671, float %1670, !dbg !17
  %1675 = bitcast float %1674 to i32, !dbg !17
  %1676 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1675), !dbg !17
  %1677 = bitcast i32 %1676 to float, !dbg !17
  %1678 = select i1 %.not3, float %1670, float %1677, !dbg !17
  %1679 = select i1 %.not3, float %1677, float %1671, !dbg !17
  %1680 = select i1 %.not3, float %1673, float %1672, !dbg !17
  %1681 = bitcast float %1680 to i32, !dbg !17
  %1682 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1681), !dbg !17
  %1683 = bitcast i32 %1682 to float, !dbg !17
  %1684 = select i1 %.not3, float %1672, float %1683, !dbg !17
  %1685 = select i1 %.not3, float %1683, float %1673, !dbg !17
  %1686 = select i1 %.not4, float %1684, float %1678, !dbg !17
  %1687 = bitcast float %1686 to i32, !dbg !17
  %1688 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1687), !dbg !17
  %1689 = select i1 %.not4, float %1685, float %1679, !dbg !17
  %1690 = bitcast float %1689 to i32, !dbg !17
  %1691 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1690), !dbg !17
  %1692 = extractvalue [2 x <8 x float>] %35, 1, !dbg !17
  %1693 = extractelement <8 x float> %1692, i64 0, !dbg !17
  %1694 = extractelement <8 x float> %1692, i64 1, !dbg !17
  %1695 = extractelement <8 x float> %1692, i64 2, !dbg !17
  %1696 = extractelement <8 x float> %1692, i64 3, !dbg !17
  %1697 = select i1 %.not3, float %1694, float %1693, !dbg !17
  %1698 = bitcast float %1697 to i32, !dbg !17
  %1699 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1698), !dbg !17
  %1700 = bitcast i32 %1699 to float, !dbg !17
  %1701 = select i1 %.not3, float %1693, float %1700, !dbg !17
  %1702 = select i1 %.not3, float %1700, float %1694, !dbg !17
  %1703 = select i1 %.not3, float %1696, float %1695, !dbg !17
  %1704 = bitcast float %1703 to i32, !dbg !17
  %1705 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1704), !dbg !17
  %1706 = bitcast i32 %1705 to float, !dbg !17
  %1707 = select i1 %.not3, float %1695, float %1706, !dbg !17
  %1708 = select i1 %.not3, float %1706, float %1696, !dbg !17
  %1709 = select i1 %.not4, float %1707, float %1701, !dbg !17
  %1710 = bitcast float %1709 to i32, !dbg !17
  %1711 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1710), !dbg !17
  %1712 = bitcast i32 %1711 to float, !dbg !17
  %1713 = select i1 %.not4, float %1701, float %1712, !dbg !17
  %1714 = select i1 %.not4, float %1712, float %1707, !dbg !17
  %1715 = select i1 %.not4, float %1708, float %1702, !dbg !17
  %1716 = bitcast float %1715 to i32, !dbg !17
  %1717 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1716), !dbg !17
  %1718 = bitcast i32 %1717 to float, !dbg !17
  %1719 = select i1 %.not4, float %1702, float %1718, !dbg !17
  %1720 = select i1 %.not4, float %1718, float %1708, !dbg !17
  %1721 = extractvalue [2 x <8 x float>] %34, 1, !dbg !17
  %1722 = extractelement <8 x float> %1721, i64 0, !dbg !17
  %1723 = extractelement <8 x float> %1721, i64 1, !dbg !17
  %1724 = extractelement <8 x float> %1721, i64 2, !dbg !17
  %1725 = extractelement <8 x float> %1721, i64 3, !dbg !17
  %1726 = select i1 %.not3, float %1723, float %1722, !dbg !17
  %1727 = bitcast float %1726 to i32, !dbg !17
  %1728 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1727), !dbg !17
  %1729 = bitcast i32 %1728 to float, !dbg !17
  %1730 = select i1 %.not3, float %1722, float %1729, !dbg !17
  %1731 = select i1 %.not3, float %1729, float %1723, !dbg !17
  %1732 = select i1 %.not3, float %1725, float %1724, !dbg !17
  %1733 = bitcast float %1732 to i32, !dbg !17
  %1734 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1733), !dbg !17
  %1735 = bitcast i32 %1734 to float, !dbg !17
  %1736 = select i1 %.not3, float %1724, float %1735, !dbg !17
  %1737 = select i1 %.not3, float %1735, float %1725, !dbg !17
  %1738 = select i1 %.not4, float %1736, float %1730, !dbg !17
  %1739 = bitcast float %1738 to i32, !dbg !17
  %1740 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1739), !dbg !17
  %1741 = bitcast i32 %1740 to float, !dbg !17
  %1742 = select i1 %.not4, float %1730, float %1741, !dbg !17
  %1743 = select i1 %.not4, float %1741, float %1736, !dbg !17
  %1744 = select i1 %.not4, float %1737, float %1731, !dbg !17
  %1745 = bitcast float %1744 to i32, !dbg !17
  %1746 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1745), !dbg !17
  %1747 = bitcast i32 %1746 to float, !dbg !17
  %1748 = select i1 %.not4, float %1731, float %1747, !dbg !17
  %1749 = select i1 %.not4, float %1747, float %1737, !dbg !17
  %1750 = extractvalue [2 x <8 x float>] %33, 1, !dbg !17
  %1751 = extractelement <8 x float> %1750, i64 0, !dbg !17
  %1752 = extractelement <8 x float> %1750, i64 1, !dbg !17
  %1753 = extractelement <8 x float> %1750, i64 2, !dbg !17
  %1754 = extractelement <8 x float> %1750, i64 3, !dbg !17
  %1755 = select i1 %.not3, float %1752, float %1751, !dbg !17
  %1756 = bitcast float %1755 to i32, !dbg !17
  %1757 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1756), !dbg !17
  %1758 = bitcast i32 %1757 to float, !dbg !17
  %1759 = select i1 %.not3, float %1751, float %1758, !dbg !17
  %1760 = select i1 %.not3, float %1758, float %1752, !dbg !17
  %1761 = select i1 %.not3, float %1754, float %1753, !dbg !17
  %1762 = bitcast float %1761 to i32, !dbg !17
  %1763 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1762), !dbg !17
  %1764 = bitcast i32 %1763 to float, !dbg !17
  %1765 = select i1 %.not3, float %1753, float %1764, !dbg !17
  %1766 = select i1 %.not3, float %1764, float %1754, !dbg !17
  %1767 = select i1 %.not4, float %1765, float %1759, !dbg !17
  %1768 = bitcast float %1767 to i32, !dbg !17
  %1769 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1768), !dbg !17
  %1770 = bitcast i32 %1769 to float, !dbg !17
  %1771 = select i1 %.not4, float %1759, float %1770, !dbg !17
  %1772 = select i1 %.not4, float %1770, float %1765, !dbg !17
  %1773 = select i1 %.not4, float %1766, float %1760, !dbg !17
  %1774 = bitcast float %1773 to i32, !dbg !17
  %1775 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1774), !dbg !17
  %1776 = bitcast i32 %1775 to float, !dbg !17
  %1777 = select i1 %.not4, float %1760, float %1776, !dbg !17
  %1778 = select i1 %.not4, float %1776, float %1766, !dbg !17
  %1779 = extractvalue [2 x <8 x float>] %31, 1, !dbg !17
  %1780 = extractelement <8 x float> %1779, i64 0, !dbg !17
  %1781 = extractelement <8 x float> %1779, i64 1, !dbg !17
  %1782 = extractelement <8 x float> %1779, i64 2, !dbg !17
  %1783 = extractelement <8 x float> %1779, i64 3, !dbg !17
  %1784 = select i1 %.not3, float %1781, float %1780, !dbg !17
  %1785 = bitcast float %1784 to i32, !dbg !17
  %1786 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1785), !dbg !17
  %1787 = bitcast i32 %1786 to float, !dbg !17
  %1788 = select i1 %.not3, float %1780, float %1787, !dbg !17
  %1789 = select i1 %.not3, float %1787, float %1781, !dbg !17
  %1790 = select i1 %.not3, float %1783, float %1782, !dbg !17
  %1791 = bitcast float %1790 to i32, !dbg !17
  %1792 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1791), !dbg !17
  %1793 = bitcast i32 %1792 to float, !dbg !17
  %1794 = select i1 %.not3, float %1782, float %1793, !dbg !17
  %1795 = select i1 %.not3, float %1793, float %1783, !dbg !17
  %1796 = select i1 %.not4, float %1794, float %1788, !dbg !17
  %1797 = bitcast float %1796 to i32, !dbg !17
  %1798 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1797), !dbg !17
  %1799 = select i1 %.not4, float %1795, float %1789, !dbg !17
  %1800 = bitcast float %1799 to i32, !dbg !17
  %1801 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1800), !dbg !17
  %1802 = extractvalue [2 x <1 x float>] %32, 1, !dbg !20
  %1803 = shufflevector <1 x float> %1802, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1804 = extractelement <8 x float> %1692, i64 4, !dbg !17
  %1805 = extractelement <8 x float> %1692, i64 5, !dbg !17
  %1806 = extractelement <8 x float> %1692, i64 6, !dbg !17
  %1807 = extractelement <8 x float> %1692, i64 7, !dbg !17
  %1808 = select i1 %.not3, float %1805, float %1804, !dbg !17
  %1809 = bitcast float %1808 to i32, !dbg !17
  %1810 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1809), !dbg !17
  %1811 = bitcast i32 %1810 to float, !dbg !17
  %1812 = select i1 %.not3, float %1804, float %1811, !dbg !17
  %1813 = select i1 %.not3, float %1811, float %1805, !dbg !17
  %1814 = select i1 %.not3, float %1807, float %1806, !dbg !17
  %1815 = bitcast float %1814 to i32, !dbg !17
  %1816 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1815), !dbg !17
  %1817 = bitcast i32 %1816 to float, !dbg !17
  %1818 = select i1 %.not3, float %1806, float %1817, !dbg !17
  %1819 = select i1 %.not3, float %1817, float %1807, !dbg !17
  %1820 = select i1 %.not4, float %1818, float %1812, !dbg !17
  %1821 = bitcast float %1820 to i32, !dbg !17
  %1822 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1821), !dbg !17
  %1823 = bitcast i32 %1822 to float, !dbg !17
  %1824 = select i1 %.not4, float %1812, float %1823, !dbg !17
  %1825 = select i1 %.not4, float %1823, float %1818, !dbg !17
  %1826 = select i1 %.not4, float %1819, float %1813, !dbg !17
  %1827 = bitcast float %1826 to i32, !dbg !17
  %1828 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1827), !dbg !17
  %1829 = bitcast i32 %1828 to float, !dbg !17
  %1830 = select i1 %.not4, float %1813, float %1829, !dbg !17
  %1831 = select i1 %.not4, float %1829, float %1819, !dbg !17
  %1832 = extractelement <8 x float> %1721, i64 4, !dbg !17
  %1833 = extractelement <8 x float> %1721, i64 5, !dbg !17
  %1834 = extractelement <8 x float> %1721, i64 6, !dbg !17
  %1835 = extractelement <8 x float> %1721, i64 7, !dbg !17
  %1836 = select i1 %.not3, float %1833, float %1832, !dbg !17
  %1837 = bitcast float %1836 to i32, !dbg !17
  %1838 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1837), !dbg !17
  %1839 = bitcast i32 %1838 to float, !dbg !17
  %1840 = select i1 %.not3, float %1832, float %1839, !dbg !17
  %1841 = select i1 %.not3, float %1839, float %1833, !dbg !17
  %1842 = select i1 %.not3, float %1835, float %1834, !dbg !17
  %1843 = bitcast float %1842 to i32, !dbg !17
  %1844 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1843), !dbg !17
  %1845 = bitcast i32 %1844 to float, !dbg !17
  %1846 = select i1 %.not3, float %1834, float %1845, !dbg !17
  %1847 = select i1 %.not3, float %1845, float %1835, !dbg !17
  %1848 = select i1 %.not4, float %1846, float %1840, !dbg !17
  %1849 = bitcast float %1848 to i32, !dbg !17
  %1850 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1849), !dbg !17
  %1851 = bitcast i32 %1850 to float, !dbg !17
  %1852 = select i1 %.not4, float %1840, float %1851, !dbg !17
  %1853 = select i1 %.not4, float %1851, float %1846, !dbg !17
  %1854 = select i1 %.not4, float %1847, float %1841, !dbg !17
  %1855 = bitcast float %1854 to i32, !dbg !17
  %1856 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1855), !dbg !17
  %1857 = bitcast i32 %1856 to float, !dbg !17
  %1858 = select i1 %.not4, float %1841, float %1857, !dbg !17
  %1859 = select i1 %.not4, float %1857, float %1847, !dbg !17
  %1860 = extractelement <8 x float> %1750, i64 4, !dbg !17
  %1861 = extractelement <8 x float> %1750, i64 5, !dbg !17
  %1862 = extractelement <8 x float> %1750, i64 6, !dbg !17
  %1863 = extractelement <8 x float> %1750, i64 7, !dbg !17
  %1864 = select i1 %.not3, float %1861, float %1860, !dbg !17
  %1865 = bitcast float %1864 to i32, !dbg !17
  %1866 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1865), !dbg !17
  %1867 = bitcast i32 %1866 to float, !dbg !17
  %1868 = select i1 %.not3, float %1860, float %1867, !dbg !17
  %1869 = select i1 %.not3, float %1867, float %1861, !dbg !17
  %1870 = select i1 %.not3, float %1863, float %1862, !dbg !17
  %1871 = bitcast float %1870 to i32, !dbg !17
  %1872 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1871), !dbg !17
  %1873 = bitcast i32 %1872 to float, !dbg !17
  %1874 = select i1 %.not3, float %1862, float %1873, !dbg !17
  %1875 = select i1 %.not3, float %1873, float %1863, !dbg !17
  %1876 = select i1 %.not4, float %1874, float %1868, !dbg !17
  %1877 = bitcast float %1876 to i32, !dbg !17
  %1878 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1877), !dbg !17
  %1879 = bitcast i32 %1878 to float, !dbg !17
  %1880 = select i1 %.not4, float %1868, float %1879, !dbg !17
  %1881 = select i1 %.not4, float %1879, float %1874, !dbg !17
  %1882 = select i1 %.not4, float %1875, float %1869, !dbg !17
  %1883 = bitcast float %1882 to i32, !dbg !17
  %1884 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1883), !dbg !17
  %1885 = bitcast i32 %1884 to float, !dbg !17
  %1886 = select i1 %.not4, float %1869, float %1885, !dbg !17
  %1887 = select i1 %.not4, float %1885, float %1875, !dbg !17
  %1888 = extractelement <8 x float> %1779, i64 4, !dbg !17
  %1889 = extractelement <8 x float> %1779, i64 5, !dbg !17
  %1890 = extractelement <8 x float> %1779, i64 6, !dbg !17
  %1891 = extractelement <8 x float> %1779, i64 7, !dbg !17
  %1892 = select i1 %.not3, float %1889, float %1888, !dbg !17
  %1893 = bitcast float %1892 to i32, !dbg !17
  %1894 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1893), !dbg !17
  %1895 = bitcast i32 %1894 to float, !dbg !17
  %1896 = select i1 %.not3, float %1888, float %1895, !dbg !17
  %1897 = select i1 %.not3, float %1895, float %1889, !dbg !17
  %1898 = select i1 %.not3, float %1891, float %1890, !dbg !17
  %1899 = bitcast float %1898 to i32, !dbg !17
  %1900 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1475, i32 %1899), !dbg !17
  %1901 = bitcast i32 %1900 to float, !dbg !17
  %1902 = select i1 %.not3, float %1890, float %1901, !dbg !17
  %1903 = select i1 %.not3, float %1901, float %1891, !dbg !17
  %1904 = select i1 %.not4, float %1902, float %1896, !dbg !17
  %1905 = bitcast float %1904 to i32, !dbg !17
  %1906 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1905), !dbg !17
  %1907 = select i1 %.not4, float %1903, float %1897, !dbg !17
  %1908 = bitcast float %1907 to i32, !dbg !17
  %1909 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1491, i32 %1908), !dbg !17
  br i1 true, label %1910, label %1972, !dbg !22

1910:                                             ; preds = %1459
  %1911 = bitcast i32 %1906 to float, !dbg !17
  %1912 = select i1 %.not4, float %1896, float %1911, !dbg !17
  %1913 = insertelement <4 x float> poison, float %1912, i64 0, !dbg !20
  %1914 = bitcast i32 %1909 to float, !dbg !17
  %1915 = select i1 %.not4, float %1897, float %1914, !dbg !17
  %1916 = insertelement <4 x float> %1913, float %1915, i64 1, !dbg !20
  %1917 = select i1 %.not4, float %1911, float %1902, !dbg !17
  %1918 = insertelement <4 x float> %1916, float %1917, i64 2, !dbg !20
  %1919 = select i1 %.not4, float %1914, float %1903, !dbg !17
  %1920 = insertelement <4 x float> %1918, float %1919, i64 3, !dbg !20
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1803, !dbg !20
  %normalized = fmul <4 x float> %1920, %row.reciprocal, !dbg !20
  %1921 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !21
  %1922 = extractelement <4 x half> %1921, i64 3, !dbg !21
  %1923 = extractelement <4 x half> %1921, i64 2, !dbg !21
  %1924 = extractelement <4 x half> %1921, i64 1, !dbg !21
  %1925 = extractelement <4 x half> %1921, i64 0, !dbg !21
  %1926 = bitcast i32 %1798 to float, !dbg !17
  %1927 = select i1 %.not4, float %1788, float %1926, !dbg !17
  %1928 = insertelement <4 x float> poison, float %1927, i64 0, !dbg !20
  %1929 = bitcast i32 %1801 to float, !dbg !17
  %1930 = select i1 %.not4, float %1789, float %1929, !dbg !17
  %1931 = insertelement <4 x float> %1928, float %1930, i64 1, !dbg !20
  %1932 = select i1 %.not4, float %1926, float %1794, !dbg !17
  %1933 = insertelement <4 x float> %1931, float %1932, i64 2, !dbg !20
  %1934 = select i1 %.not4, float %1929, float %1795, !dbg !17
  %1935 = insertelement <4 x float> %1933, float %1934, i64 3, !dbg !20
  %normalized71 = fmul <4 x float> %1935, %row.reciprocal, !dbg !20
  %1936 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !21
  %1937 = extractelement <4 x half> %1936, i64 3, !dbg !21
  %1938 = extractelement <4 x half> %1936, i64 2, !dbg !21
  %1939 = extractelement <4 x half> %1936, i64 1, !dbg !21
  %1940 = extractelement <4 x half> %1936, i64 0, !dbg !21
  %1941 = bitcast i32 %1688 to float, !dbg !17
  %1942 = select i1 %.not4, float %1678, float %1941, !dbg !17
  %1943 = insertelement <4 x float> poison, float %1942, i64 0, !dbg !20
  %1944 = bitcast i32 %1691 to float, !dbg !17
  %1945 = select i1 %.not4, float %1679, float %1944, !dbg !17
  %1946 = insertelement <4 x float> %1943, float %1945, i64 1, !dbg !20
  %1947 = select i1 %.not4, float %1941, float %1684, !dbg !17
  %1948 = insertelement <4 x float> %1946, float %1947, i64 2, !dbg !20
  %1949 = select i1 %.not4, float %1944, float %1685, !dbg !17
  %1950 = insertelement <4 x float> %1948, float %1949, i64 3, !dbg !20
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1585, !dbg !20
  %normalized73 = fmul <4 x float> %1950, %row.reciprocal72, !dbg !20
  %1951 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !21
  %1952 = extractelement <4 x half> %1951, i64 3, !dbg !21
  %1953 = extractelement <4 x half> %1951, i64 2, !dbg !21
  %1954 = extractelement <4 x half> %1951, i64 1, !dbg !21
  %1955 = extractelement <4 x half> %1951, i64 0, !dbg !21
  %1956 = bitcast i32 %1580 to float, !dbg !17
  %1957 = select i1 %.not4, float %1570, float %1956, !dbg !17
  %1958 = insertelement <4 x float> poison, float %1957, i64 0, !dbg !20
  %1959 = bitcast i32 %1583 to float, !dbg !17
  %1960 = select i1 %.not4, float %1571, float %1959, !dbg !17
  %1961 = insertelement <4 x float> %1958, float %1960, i64 1, !dbg !20
  %1962 = select i1 %.not4, float %1956, float %1576, !dbg !17
  %1963 = insertelement <4 x float> %1961, float %1962, i64 2, !dbg !20
  %1964 = select i1 %.not4, float %1959, float %1577, !dbg !17
  %1965 = insertelement <4 x float> %1963, float %1964, i64 3, !dbg !20
  %normalized74 = fmul <4 x float> %1965, %row.reciprocal72, !dbg !20
  %1966 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !21
  %1967 = extractelement <4 x half> %1966, i64 3, !dbg !21
  %1968 = extractelement <4 x half> %1966, i64 2, !dbg !21
  %1969 = extractelement <4 x half> %1966, i64 1, !dbg !21
  %1970 = extractelement <4 x half> %1966, i64 0, !dbg !21
  %1971 = shl nuw nsw i64 %17, 19, !dbg !22
  %address.thread589 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !22, !range !35, !frisk.register_lifetimes !1
  %address.local590 = zext nneg i32 %address.thread589 to i64, !dbg !14
  %address.local591 = add nsw i64 %18, %address.local590, !dbg !22
  %address.local592 = lshr i64 %address.local590, 6, !dbg !14
  %address.local593 = shl nuw nsw i64 %address.local592, 4, !dbg !22
  %address.local594 = add nsw i64 %address.local591, %address.local593, !dbg !22
  %address.local595 = and i64 %address.local590, 240, !dbg !22
  %address.local596 = sub nsw i64 %address.local594, %address.local595, !dbg !22
  %address.local597 = add nsw i64 %address.local596, %19, !dbg !22
  %address.local598 = shl nuw nsw i64 %address.local597, 7, !dbg !22
  %address.local599 = add nuw nsw i64 %1971, %address.local598, !dbg !22
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !22
  %address.local601 = lshr i64 %address.local590, 2, !dbg !14
  %address.local602 = and i64 %address.local601, 12, !dbg !14
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local602, !dbg !22
  store half %1970, ptr addrspace(1) %address.local603, align 2, !dbg !22
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !22
  %address.local605 = or disjoint i64 %address.local602, 1, !dbg !22
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local604, i64 %address.local605, !dbg !22
  store half %1969, ptr addrspace(1) %address.local606, align 2, !dbg !22
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !22
  %address.local608 = or disjoint i64 %address.local602, 2, !dbg !22
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local607, i64 %address.local608, !dbg !22
  store half %1968, ptr addrspace(1) %address.local609, align 2, !dbg !22
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !22
  %address.local611 = or disjoint i64 %address.local602, 3, !dbg !22
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local610, i64 %address.local611, !dbg !22
  store half %1967, ptr addrspace(1) %address.local612, align 2, !dbg !22
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !22
  %address.local614 = or disjoint i64 %address.local602, 16, !dbg !22
  %address.local615 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local613, i64 %address.local614, !dbg !22
  store half %1955, ptr addrspace(1) %address.local615, align 2, !dbg !22
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !22
  %address.local617 = or disjoint i64 %address.local602, 17, !dbg !22
  %address.local618 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local616, i64 %address.local617, !dbg !22
  store half %1954, ptr addrspace(1) %address.local618, align 2, !dbg !22
  %address.local619 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !22
  %address.local620 = or disjoint i64 %address.local602, 18, !dbg !22
  %address.local621 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local619, i64 %address.local620, !dbg !22
  store half %1953, ptr addrspace(1) %address.local621, align 2, !dbg !22
  %address.local622 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local599, !dbg !22
  %address.local623 = or disjoint i64 %address.local602, 19, !dbg !22
  %address.local624 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local622, i64 %address.local623, !dbg !22
  store half %1952, ptr addrspace(1) %address.local624, align 2, !dbg !22
  %address.local625 = shl nsw i64 %address.local597, 7, !dbg !22
  %address.local626 = add i64 %address.local625, 8192, !dbg !22
  %address.local627 = add nuw nsw i64 %1971, %address.local626, !dbg !22
  %address.local628 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local627, !dbg !22
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local628, i64 %address.local602, !dbg !22
  store half %1940, ptr addrspace(1) %address.local629, align 2, !dbg !22
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local627, !dbg !22
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local630, i64 %address.local605, !dbg !22
  store half %1939, ptr addrspace(1) %address.local631, align 2, !dbg !22
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local627, !dbg !22
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local632, i64 %address.local608, !dbg !22
  store half %1938, ptr addrspace(1) %address.local633, align 2, !dbg !22
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local627, !dbg !22
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local634, i64 %address.local611, !dbg !22
  store half %1937, ptr addrspace(1) %address.local635, align 2, !dbg !22
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local627, !dbg !22
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local636, i64 %address.local614, !dbg !22
  store half %1925, ptr addrspace(1) %address.local637, align 2, !dbg !22
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local627, !dbg !22
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local638, i64 %address.local617, !dbg !22
  store half %1924, ptr addrspace(1) %address.local639, align 2, !dbg !22
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local627, !dbg !22
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local640, i64 %address.local620, !dbg !22
  store half %1923, ptr addrspace(1) %address.local641, align 2, !dbg !22
  %address.local642 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local627, !dbg !22
  %address.local643 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local642, i64 %address.local623, !dbg !22
  store half %1922, ptr addrspace(1) %address.local643, align 2, !dbg !22
  br label %1972, !dbg !22

1972:                                             ; preds = %1910, %1459
  br i1 true, label %1973, label %2013, !dbg !22

1973:                                             ; preds = %1972
  %1974 = insertelement <4 x float> poison, float %1880, i64 0, !dbg !20
  %1975 = insertelement <4 x float> %1974, float %1886, i64 1, !dbg !20
  %1976 = insertelement <4 x float> %1975, float %1881, i64 2, !dbg !20
  %1977 = insertelement <4 x float> %1976, float %1887, i64 3, !dbg !20
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1803, !dbg !20
  %normalized76 = fmul <4 x float> %1977, %row.reciprocal75, !dbg !20
  %1978 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !21
  %1979 = extractelement <4 x half> %1978, i64 3, !dbg !21
  %1980 = extractelement <4 x half> %1978, i64 2, !dbg !21
  %1981 = extractelement <4 x half> %1978, i64 1, !dbg !21
  %1982 = extractelement <4 x half> %1978, i64 0, !dbg !21
  %1983 = insertelement <4 x float> poison, float %1771, i64 0, !dbg !20
  %1984 = insertelement <4 x float> %1983, float %1777, i64 1, !dbg !20
  %1985 = insertelement <4 x float> %1984, float %1772, i64 2, !dbg !20
  %1986 = insertelement <4 x float> %1985, float %1778, i64 3, !dbg !20
  %normalized77 = fmul <4 x float> %1986, %row.reciprocal75, !dbg !20
  %1987 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !21
  %1988 = extractelement <4 x half> %1987, i64 3, !dbg !21
  %1989 = extractelement <4 x half> %1987, i64 2, !dbg !21
  %1990 = extractelement <4 x half> %1987, i64 1, !dbg !21
  %1991 = extractelement <4 x half> %1987, i64 0, !dbg !21
  %1992 = insertelement <4 x float> poison, float %1662, i64 0, !dbg !20
  %1993 = insertelement <4 x float> %1992, float %1668, i64 1, !dbg !20
  %1994 = insertelement <4 x float> %1993, float %1663, i64 2, !dbg !20
  %1995 = insertelement <4 x float> %1994, float %1669, i64 3, !dbg !20
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1585, !dbg !20
  %normalized79 = fmul <4 x float> %1995, %row.reciprocal78, !dbg !20
  %1996 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !21
  %1997 = extractelement <4 x half> %1996, i64 3, !dbg !21
  %1998 = extractelement <4 x half> %1996, i64 2, !dbg !21
  %1999 = extractelement <4 x half> %1996, i64 1, !dbg !21
  %2000 = extractelement <4 x half> %1996, i64 0, !dbg !21
  %2001 = insertelement <4 x float> poison, float %1553, i64 0, !dbg !20
  %2002 = insertelement <4 x float> %2001, float %1559, i64 1, !dbg !20
  %2003 = insertelement <4 x float> %2002, float %1554, i64 2, !dbg !20
  %2004 = insertelement <4 x float> %2003, float %1560, i64 3, !dbg !20
  %normalized80 = fmul <4 x float> %2004, %row.reciprocal78, !dbg !20
  %2005 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !21
  %2006 = extractelement <4 x half> %2005, i64 3, !dbg !21
  %2007 = extractelement <4 x half> %2005, i64 2, !dbg !21
  %2008 = extractelement <4 x half> %2005, i64 1, !dbg !21
  %2009 = extractelement <4 x half> %2005, i64 0, !dbg !21
  %2010 = add nsw i64 %10, %15, !dbg !22
  %2011 = shl nsw i64 %2010, 19, !dbg !22
  %2012 = and i64 %2011, 16252928, !dbg !22
  %address.thread644 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !22, !range !35, !frisk.register_lifetimes !1
  %address.local645 = zext nneg i32 %address.thread644 to i64, !dbg !14
  %address.local646 = add nsw i64 %18, %address.local645, !dbg !22
  %address.local647 = lshr i64 %address.local645, 6, !dbg !14
  %address.local648 = shl nuw nsw i64 %address.local647, 4, !dbg !22
  %address.local649 = add nsw i64 %address.local646, %address.local648, !dbg !22
  %address.local650 = and i64 %address.local645, 240, !dbg !22
  %address.local651 = sub nsw i64 %address.local649, %address.local650, !dbg !22
  %address.local652 = add nsw i64 %address.local651, %19, !dbg !22
  %address.local653 = shl nuw nsw i64 %address.local652, 7, !dbg !22
  %address.local654 = add nuw nsw i64 %2012, %address.local653, !dbg !22
  %address.local655 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local654, !dbg !22
  %address.local656 = lshr i64 %address.local645, 2, !dbg !14
  %address.local657 = and i64 %address.local656, 12, !dbg !14
  %address.local658 = or disjoint i64 %address.local657, 32, !dbg !22
  %address.local659 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local655, i64 %address.local658, !dbg !22
  store half %2009, ptr addrspace(1) %address.local659, align 2, !dbg !22
  %address.local660 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local654, !dbg !22
  %address.local661 = or disjoint i64 %address.local657, 33, !dbg !22
  %address.local662 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local660, i64 %address.local661, !dbg !22
  store half %2008, ptr addrspace(1) %address.local662, align 2, !dbg !22
  %address.local663 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local654, !dbg !22
  %address.local664 = or disjoint i64 %address.local657, 34, !dbg !22
  %address.local665 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local663, i64 %address.local664, !dbg !22
  store half %2007, ptr addrspace(1) %address.local665, align 2, !dbg !22
  %address.local666 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local654, !dbg !22
  %address.local667 = or disjoint i64 %address.local657, 35, !dbg !22
  %address.local668 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local666, i64 %address.local667, !dbg !22
  store half %2006, ptr addrspace(1) %address.local668, align 2, !dbg !22
  %address.local669 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local654, !dbg !22
  %address.local670 = or disjoint i64 %address.local657, 48, !dbg !22
  %address.local671 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local669, i64 %address.local670, !dbg !22
  store half %2000, ptr addrspace(1) %address.local671, align 2, !dbg !22
  %address.local672 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local654, !dbg !22
  %address.local673 = or disjoint i64 %address.local657, 49, !dbg !22
  %address.local674 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local672, i64 %address.local673, !dbg !22
  store half %1999, ptr addrspace(1) %address.local674, align 2, !dbg !22
  %address.local675 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local654, !dbg !22
  %address.local676 = or disjoint i64 %address.local657, 50, !dbg !22
  %address.local677 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local675, i64 %address.local676, !dbg !22
  store half %1998, ptr addrspace(1) %address.local677, align 2, !dbg !22
  %address.local678 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local654, !dbg !22
  %address.local679 = or i64 %address.local656, 51, !dbg !22
  %address.local680 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local678, i64 %address.local679, !dbg !22
  store half %1997, ptr addrspace(1) %address.local680, align 2, !dbg !22
  %address.local681 = shl nsw i64 %address.local652, 7, !dbg !22
  %address.local682 = add i64 %address.local681, 8192, !dbg !22
  %address.local683 = add nuw nsw i64 %2012, %address.local682, !dbg !22
  %address.local684 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local683, !dbg !22
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local684, i64 %address.local658, !dbg !22
  store half %1991, ptr addrspace(1) %address.local685, align 2, !dbg !22
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local683, !dbg !22
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local686, i64 %address.local661, !dbg !22
  store half %1990, ptr addrspace(1) %address.local687, align 2, !dbg !22
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local683, !dbg !22
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local688, i64 %address.local664, !dbg !22
  store half %1989, ptr addrspace(1) %address.local689, align 2, !dbg !22
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local683, !dbg !22
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local690, i64 %address.local667, !dbg !22
  store half %1988, ptr addrspace(1) %address.local691, align 2, !dbg !22
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local683, !dbg !22
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local692, i64 %address.local670, !dbg !22
  store half %1982, ptr addrspace(1) %address.local693, align 2, !dbg !22
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local683, !dbg !22
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local694, i64 %address.local673, !dbg !22
  store half %1981, ptr addrspace(1) %address.local695, align 2, !dbg !22
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local683, !dbg !22
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local696, i64 %address.local676, !dbg !22
  store half %1980, ptr addrspace(1) %address.local697, align 2, !dbg !22
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local683, !dbg !22
  %address.local699 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local698, i64 %address.local679, !dbg !22
  store half %1979, ptr addrspace(1) %address.local699, align 2, !dbg !22
  br label %2013, !dbg !22

2013:                                             ; preds = %1973, %1972
  br i1 true, label %2014, label %2054, !dbg !22

2014:                                             ; preds = %2013
  %2015 = insertelement <4 x float> poison, float %1852, i64 0, !dbg !20
  %2016 = insertelement <4 x float> %2015, float %1858, i64 1, !dbg !20
  %2017 = insertelement <4 x float> %2016, float %1853, i64 2, !dbg !20
  %2018 = insertelement <4 x float> %2017, float %1859, i64 3, !dbg !20
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1803, !dbg !20
  %normalized82 = fmul <4 x float> %2018, %row.reciprocal81, !dbg !20
  %2019 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !21
  %2020 = extractelement <4 x half> %2019, i64 3, !dbg !21
  %2021 = extractelement <4 x half> %2019, i64 2, !dbg !21
  %2022 = extractelement <4 x half> %2019, i64 1, !dbg !21
  %2023 = extractelement <4 x half> %2019, i64 0, !dbg !21
  %2024 = insertelement <4 x float> poison, float %1742, i64 0, !dbg !20
  %2025 = insertelement <4 x float> %2024, float %1748, i64 1, !dbg !20
  %2026 = insertelement <4 x float> %2025, float %1743, i64 2, !dbg !20
  %2027 = insertelement <4 x float> %2026, float %1749, i64 3, !dbg !20
  %normalized83 = fmul <4 x float> %2027, %row.reciprocal81, !dbg !20
  %2028 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !21
  %2029 = extractelement <4 x half> %2028, i64 3, !dbg !21
  %2030 = extractelement <4 x half> %2028, i64 2, !dbg !21
  %2031 = extractelement <4 x half> %2028, i64 1, !dbg !21
  %2032 = extractelement <4 x half> %2028, i64 0, !dbg !21
  %2033 = insertelement <4 x float> poison, float %1634, i64 0, !dbg !20
  %2034 = insertelement <4 x float> %2033, float %1640, i64 1, !dbg !20
  %2035 = insertelement <4 x float> %2034, float %1635, i64 2, !dbg !20
  %2036 = insertelement <4 x float> %2035, float %1641, i64 3, !dbg !20
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1585, !dbg !20
  %normalized85 = fmul <4 x float> %2036, %row.reciprocal84, !dbg !20
  %2037 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !21
  %2038 = extractelement <4 x half> %2037, i64 3, !dbg !21
  %2039 = extractelement <4 x half> %2037, i64 2, !dbg !21
  %2040 = extractelement <4 x half> %2037, i64 1, !dbg !21
  %2041 = extractelement <4 x half> %2037, i64 0, !dbg !21
  %2042 = insertelement <4 x float> poison, float %1524, i64 0, !dbg !20
  %2043 = insertelement <4 x float> %2042, float %1530, i64 1, !dbg !20
  %2044 = insertelement <4 x float> %2043, float %1525, i64 2, !dbg !20
  %2045 = insertelement <4 x float> %2044, float %1531, i64 3, !dbg !20
  %normalized86 = fmul <4 x float> %2045, %row.reciprocal84, !dbg !20
  %2046 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !21
  %2047 = extractelement <4 x half> %2046, i64 3, !dbg !21
  %2048 = extractelement <4 x half> %2046, i64 2, !dbg !21
  %2049 = extractelement <4 x half> %2046, i64 1, !dbg !21
  %2050 = extractelement <4 x half> %2046, i64 0, !dbg !21
  %2051 = add nsw i64 %10, %15, !dbg !22
  %2052 = shl nsw i64 %2051, 19, !dbg !22
  %2053 = and i64 %2052, 16252928, !dbg !22
  %address.thread700 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !22, !range !35, !frisk.register_lifetimes !1
  %address.local701 = zext nneg i32 %address.thread700 to i64, !dbg !14
  %address.local702 = add nsw i64 %18, %address.local701, !dbg !22
  %address.local703 = lshr i64 %address.local701, 6, !dbg !14
  %address.local704 = shl nuw nsw i64 %address.local703, 4, !dbg !22
  %address.local705 = add nsw i64 %address.local702, %address.local704, !dbg !22
  %address.local706 = and i64 %address.local701, 240, !dbg !22
  %address.local707 = sub nsw i64 %address.local705, %address.local706, !dbg !22
  %address.local708 = add nsw i64 %address.local707, %19, !dbg !22
  %address.local709 = shl nuw nsw i64 %address.local708, 7, !dbg !22
  %address.local710 = add nuw nsw i64 %2053, %address.local709, !dbg !22
  %address.local711 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local710, !dbg !22
  %address.local712 = lshr i64 %address.local701, 2, !dbg !14
  %address.local713 = and i64 %address.local712, 12, !dbg !14
  %address.local714 = or disjoint i64 %address.local713, 64, !dbg !22
  %address.local715 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local711, i64 %address.local714, !dbg !22
  store half %2050, ptr addrspace(1) %address.local715, align 2, !dbg !22
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local710, !dbg !22
  %address.local717 = or disjoint i64 %address.local713, 65, !dbg !22
  %address.local718 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local716, i64 %address.local717, !dbg !22
  store half %2049, ptr addrspace(1) %address.local718, align 2, !dbg !22
  %address.local719 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local710, !dbg !22
  %address.local720 = or disjoint i64 %address.local713, 66, !dbg !22
  %address.local721 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local719, i64 %address.local720, !dbg !22
  store half %2048, ptr addrspace(1) %address.local721, align 2, !dbg !22
  %address.local722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local710, !dbg !22
  %address.local723 = or disjoint i64 %address.local713, 67, !dbg !22
  %address.local724 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local722, i64 %address.local723, !dbg !22
  store half %2047, ptr addrspace(1) %address.local724, align 2, !dbg !22
  %address.local725 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local710, !dbg !22
  %address.local726 = or disjoint i64 %address.local713, 80, !dbg !22
  %address.local727 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local725, i64 %address.local726, !dbg !22
  store half %2041, ptr addrspace(1) %address.local727, align 2, !dbg !22
  %address.local728 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local710, !dbg !22
  %address.local729 = or disjoint i64 %address.local713, 81, !dbg !22
  %address.local730 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local728, i64 %address.local729, !dbg !22
  store half %2040, ptr addrspace(1) %address.local730, align 2, !dbg !22
  %address.local731 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local710, !dbg !22
  %address.local732 = or disjoint i64 %address.local713, 82, !dbg !22
  %address.local733 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local731, i64 %address.local732, !dbg !22
  store half %2039, ptr addrspace(1) %address.local733, align 2, !dbg !22
  %address.local734 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local710, !dbg !22
  %address.local735 = or disjoint i64 %address.local713, 83, !dbg !22
  %address.local736 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local734, i64 %address.local735, !dbg !22
  store half %2038, ptr addrspace(1) %address.local736, align 2, !dbg !22
  %address.local737 = shl nsw i64 %address.local708, 7, !dbg !22
  %address.local738 = add i64 %address.local737, 8192, !dbg !22
  %address.local739 = add nuw nsw i64 %2053, %address.local738, !dbg !22
  %address.local740 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local739, !dbg !22
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local740, i64 %address.local714, !dbg !22
  store half %2032, ptr addrspace(1) %address.local741, align 2, !dbg !22
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local739, !dbg !22
  %address.local743 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local742, i64 %address.local717, !dbg !22
  store half %2031, ptr addrspace(1) %address.local743, align 2, !dbg !22
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local739, !dbg !22
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local744, i64 %address.local720, !dbg !22
  store half %2030, ptr addrspace(1) %address.local745, align 2, !dbg !22
  %address.local746 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local739, !dbg !22
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local746, i64 %address.local723, !dbg !22
  store half %2029, ptr addrspace(1) %address.local747, align 2, !dbg !22
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local739, !dbg !22
  %address.local749 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local748, i64 %address.local726, !dbg !22
  store half %2023, ptr addrspace(1) %address.local749, align 2, !dbg !22
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local739, !dbg !22
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local750, i64 %address.local729, !dbg !22
  store half %2022, ptr addrspace(1) %address.local751, align 2, !dbg !22
  %address.local752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local739, !dbg !22
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local752, i64 %address.local732, !dbg !22
  store half %2021, ptr addrspace(1) %address.local753, align 2, !dbg !22
  %address.local754 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local739, !dbg !22
  %address.local755 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local754, i64 %address.local735, !dbg !22
  store half %2020, ptr addrspace(1) %address.local755, align 2, !dbg !22
  br label %2054, !dbg !22

2054:                                             ; preds = %2014, %2013
  br i1 true, label %2055, label %2095, !dbg !22

2055:                                             ; preds = %2054
  %2056 = insertelement <4 x float> poison, float %1824, i64 0, !dbg !20
  %2057 = insertelement <4 x float> %2056, float %1830, i64 1, !dbg !20
  %2058 = insertelement <4 x float> %2057, float %1825, i64 2, !dbg !20
  %2059 = insertelement <4 x float> %2058, float %1831, i64 3, !dbg !20
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1803, !dbg !20
  %normalized88 = fmul <4 x float> %2059, %row.reciprocal87, !dbg !20
  %2060 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !21
  %2061 = extractelement <4 x half> %2060, i64 3, !dbg !21
  %2062 = extractelement <4 x half> %2060, i64 2, !dbg !21
  %2063 = extractelement <4 x half> %2060, i64 1, !dbg !21
  %2064 = extractelement <4 x half> %2060, i64 0, !dbg !21
  %2065 = insertelement <4 x float> poison, float %1713, i64 0, !dbg !20
  %2066 = insertelement <4 x float> %2065, float %1719, i64 1, !dbg !20
  %2067 = insertelement <4 x float> %2066, float %1714, i64 2, !dbg !20
  %2068 = insertelement <4 x float> %2067, float %1720, i64 3, !dbg !20
  %normalized89 = fmul <4 x float> %2068, %row.reciprocal87, !dbg !20
  %2069 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !21
  %2070 = extractelement <4 x half> %2069, i64 3, !dbg !21
  %2071 = extractelement <4 x half> %2069, i64 2, !dbg !21
  %2072 = extractelement <4 x half> %2069, i64 1, !dbg !21
  %2073 = extractelement <4 x half> %2069, i64 0, !dbg !21
  %2074 = insertelement <4 x float> poison, float %1606, i64 0, !dbg !20
  %2075 = insertelement <4 x float> %2074, float %1612, i64 1, !dbg !20
  %2076 = insertelement <4 x float> %2075, float %1607, i64 2, !dbg !20
  %2077 = insertelement <4 x float> %2076, float %1613, i64 3, !dbg !20
  %row.reciprocal90 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1585, !dbg !20
  %normalized91 = fmul <4 x float> %2077, %row.reciprocal90, !dbg !20
  %2078 = fptrunc <4 x float> %normalized91 to <4 x half>, !dbg !21
  %2079 = extractelement <4 x half> %2078, i64 3, !dbg !21
  %2080 = extractelement <4 x half> %2078, i64 2, !dbg !21
  %2081 = extractelement <4 x half> %2078, i64 1, !dbg !21
  %2082 = extractelement <4 x half> %2078, i64 0, !dbg !21
  %2083 = insertelement <4 x float> poison, float %1495, i64 0, !dbg !20
  %2084 = insertelement <4 x float> %2083, float %1501, i64 1, !dbg !20
  %2085 = insertelement <4 x float> %2084, float %1496, i64 2, !dbg !20
  %2086 = insertelement <4 x float> %2085, float %1502, i64 3, !dbg !20
  %normalized92 = fmul <4 x float> %2086, %row.reciprocal90, !dbg !20
  %2087 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !21
  %2088 = extractelement <4 x half> %2087, i64 3, !dbg !21
  %2089 = extractelement <4 x half> %2087, i64 2, !dbg !21
  %2090 = extractelement <4 x half> %2087, i64 1, !dbg !21
  %2091 = extractelement <4 x half> %2087, i64 0, !dbg !21
  %2092 = add nsw i64 %10, %15, !dbg !22
  %2093 = shl nsw i64 %2092, 19, !dbg !22
  %2094 = and i64 %2093, 16252928, !dbg !22
  %address.thread756 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !22, !range !35, !frisk.register_lifetimes !1
  %address.local757 = zext nneg i32 %address.thread756 to i64, !dbg !14
  %address.local758 = add nsw i64 %18, %address.local757, !dbg !22
  %address.local759 = lshr i64 %address.local757, 6, !dbg !14
  %address.local760 = shl nuw nsw i64 %address.local759, 4, !dbg !22
  %address.local761 = add nsw i64 %address.local758, %address.local760, !dbg !22
  %address.local762 = and i64 %address.local757, 240, !dbg !22
  %address.local763 = sub nsw i64 %address.local761, %address.local762, !dbg !22
  %address.local764 = add nsw i64 %address.local763, %19, !dbg !22
  %address.local765 = shl nuw nsw i64 %address.local764, 7, !dbg !22
  %address.local766 = add nuw nsw i64 %2094, %address.local765, !dbg !22
  %address.local767 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local766, !dbg !22
  %address.local768 = lshr i64 %address.local757, 2, !dbg !14
  %address.local769 = and i64 %address.local768, 12, !dbg !14
  %address.local770 = or disjoint i64 %address.local769, 96, !dbg !22
  %address.local771 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local767, i64 %address.local770, !dbg !22
  store half %2091, ptr addrspace(1) %address.local771, align 2, !dbg !22
  %address.local772 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local766, !dbg !22
  %address.local773 = or disjoint i64 %address.local769, 97, !dbg !22
  %address.local774 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local772, i64 %address.local773, !dbg !22
  store half %2090, ptr addrspace(1) %address.local774, align 2, !dbg !22
  %address.local775 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local766, !dbg !22
  %address.local776 = or disjoint i64 %address.local769, 98, !dbg !22
  %address.local777 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local775, i64 %address.local776, !dbg !22
  store half %2089, ptr addrspace(1) %address.local777, align 2, !dbg !22
  %address.local778 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local766, !dbg !22
  %address.local779 = or disjoint i64 %address.local769, 99, !dbg !22
  %address.local780 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local778, i64 %address.local779, !dbg !22
  store half %2088, ptr addrspace(1) %address.local780, align 2, !dbg !22
  %address.local781 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local766, !dbg !22
  %address.local782 = or disjoint i64 %address.local769, 112, !dbg !22
  %address.local783 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local781, i64 %address.local782, !dbg !22
  store half %2082, ptr addrspace(1) %address.local783, align 2, !dbg !22
  %address.local784 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local766, !dbg !22
  %address.local785 = or disjoint i64 %address.local769, 113, !dbg !22
  %address.local786 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local784, i64 %address.local785, !dbg !22
  store half %2081, ptr addrspace(1) %address.local786, align 2, !dbg !22
  %address.local787 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local766, !dbg !22
  %address.local788 = or disjoint i64 %address.local769, 114, !dbg !22
  %address.local789 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local787, i64 %address.local788, !dbg !22
  store half %2080, ptr addrspace(1) %address.local789, align 2, !dbg !22
  %address.local790 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local766, !dbg !22
  %address.local791 = or i64 %address.local768, 115, !dbg !22
  %address.local792 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local790, i64 %address.local791, !dbg !22
  store half %2079, ptr addrspace(1) %address.local792, align 2, !dbg !22
  %address.local793 = shl nsw i64 %address.local764, 7, !dbg !22
  %address.local794 = add i64 %address.local793, 8192, !dbg !22
  %address.local795 = add nuw nsw i64 %2094, %address.local794, !dbg !22
  %address.local796 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local795, !dbg !22
  %address.local797 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local796, i64 %address.local770, !dbg !22
  store half %2073, ptr addrspace(1) %address.local797, align 2, !dbg !22
  %address.local798 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local795, !dbg !22
  %address.local799 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local798, i64 %address.local773, !dbg !22
  store half %2072, ptr addrspace(1) %address.local799, align 2, !dbg !22
  %address.local800 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local795, !dbg !22
  %address.local801 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local800, i64 %address.local776, !dbg !22
  store half %2071, ptr addrspace(1) %address.local801, align 2, !dbg !22
  %address.local802 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local795, !dbg !22
  %address.local803 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local802, i64 %address.local779, !dbg !22
  store half %2070, ptr addrspace(1) %address.local803, align 2, !dbg !22
  %address.local804 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local795, !dbg !22
  %address.local805 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local804, i64 %address.local782, !dbg !22
  store half %2064, ptr addrspace(1) %address.local805, align 2, !dbg !22
  %address.local806 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local795, !dbg !22
  %address.local807 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local806, i64 %address.local785, !dbg !22
  store half %2063, ptr addrspace(1) %address.local807, align 2, !dbg !22
  %address.local808 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local795, !dbg !22
  %address.local809 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local808, i64 %address.local788, !dbg !22
  store half %2062, ptr addrspace(1) %address.local809, align 2, !dbg !22
  %address.local810 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local795, !dbg !22
  %address.local811 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local810, i64 %address.local791, !dbg !22
  store half %2061, ptr addrspace(1) %address.local811, align 2, !dbg !22
  br label %2095, !dbg !22

2095:                                             ; preds = %2055, %2054
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

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #6 = { nocallback nofree nounwind willreturn }
attributes #7 = { convergent nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }

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
