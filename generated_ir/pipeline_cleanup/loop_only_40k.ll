; ModuleID = '3rd/deepgengraph/test/test_input_spidD.mlir'
source_filename = "3rd/deepgengraph/test/test_input_spidD.mlir"

@shm_0 = addrspace(3) global [40960 x i8] undef, align 16

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
  %11 = shl nsw i64 %8, 7, !dbg !9
  %.lobit = ashr i32 %7, 31, !dbg !12
  %12 = xor i32 %7, %.lobit, !dbg !12
  %13 = sdiv i32 %12, 32, !dbg !12
  %.lobit1 = ashr i32 %7, 31, !dbg !12
  %14 = xor i32 %13, %.lobit1, !dbg !12
  %15 = sext i32 %14 to i64, !dbg !12
  %16 = add nsw i64 %10, %15, !dbg !12
  %17 = and i64 %16, 31, !dbg !12
  %18 = shl nsw i64 %8, 7, !dbg !12
  %19 = mul nsw i64 %15, -4096, !dbg !12
  %20 = add nsw i64 %18, %19, !dbg !12
  %21 = shl nsw i64 %20, 7, !dbg !12
  %22 = add nsw i64 %10, %15, !dbg !12
  %23 = and i64 %22, 31, !dbg !12
  %24 = shl nuw nsw i64 %23, 19, !dbg !12
  %25 = add nsw i64 %24, %21, !dbg !12
  %26 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !35
  %27 = zext nneg i32 %26 to i64, !dbg !13
  %28 = lshr i64 %27, 6, !dbg !13
  %29 = or disjoint i64 %11, 127, !dbg !14
  %30 = add nsw i64 %11, 128, !dbg !14
  %31 = icmp uge i64 %30, %29, !dbg !14
  %.idx = shl nuw nsw i64 %17, 20, !dbg !12
  %32 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %33 = getelementptr [2 x i8], ptr addrspace(1) %32, i64 %21, !dbg !12
  %34 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %35 = getelementptr i8, ptr addrspace(1) %34, i64 64, !dbg !12
  %36 = shl nsw i64 %10, 12, !dbg !17
  %37 = shl nsw i64 %6, 17, !dbg !17
  %38 = shl nsw i64 %10, 7, !dbg !17
  %39 = shl nsw i64 %6, 12, !dbg !17
  %40 = add nsw i64 %38, %39, !dbg !17
  %41 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %42 = getelementptr i8, ptr addrspace(1) %41, i64 128, !dbg !12
  %43 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %44 = getelementptr i8, ptr addrspace(1) %43, i64 192, !dbg !12
  br label %45, !dbg !16

45:                                               ; preds = %830, %4
  %46 = phi i64 [ %1563, %830 ], [ 0, %4 ], !dbg !16
  %47 = phi [2 x <8 x float>] [ %1550, %830 ], [ zeroinitializer, %4 ], !dbg !16
  %48 = phi [2 x <1 x float>] [ %1172, %830 ], [ zeroinitializer, %4 ], !dbg !16
  %49 = phi [2 x <8 x float>] [ %1554, %830 ], [ zeroinitializer, %4 ], !dbg !16
  %50 = phi [2 x <8 x float>] [ %1558, %830 ], [ zeroinitializer, %4 ], !dbg !16
  %51 = phi [2 x <8 x float>] [ %1562, %830 ], [ zeroinitializer, %4 ], !dbg !16
  %.not = icmp sgt i64 %46, %8, !dbg !16
  br i1 %.not, label %1564, label %52, !dbg !16

52:                                               ; preds = %45
  %53 = shl nsw i64 %46, 7, !dbg !16
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !12, !range !35, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local174 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local175 = and i64 %address.local174, 1536, !dbg !12
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !12
  %address.local177 = ptrtoint ptr addrspace(3) %address.local176 to i64, !dbg !12
  %address.local178 = trunc i64 %address.local177 to i32, !dbg !12
  %54 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local178), !dbg !12
  %55 = zext i32 %54 to i64, !dbg !12
  %56 = inttoptr i64 %55 to ptr addrspace(3), !dbg !12
  %address.local179 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local180 = and i64 %address.local179, 16256, !dbg !12
  %address.local181 = lshr i64 %address.local, 4, !dbg !12
  %address.local182 = and i64 %address.local181, 8, !dbg !12
  %address.local183 = or disjoint i64 %address.local180, %address.local182, !dbg !12
  %address.local184 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local183, !dbg !12
  %57 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %58 = ptrtoint ptr addrspace(1) %address.local184 to i64, !dbg !12
  %59 = sub i64 %58, %57, !dbg !12
  %dma.buffer.offset1154 = trunc i64 %59 to i32, !dbg !12
  %60 = trunc i64 %57 to i32, !dbg !12
  %61 = insertelement <4 x i32> poison, i32 %60, i64 0, !dbg !12
  %62 = lshr i64 %57, 32, !dbg !12
  %63 = trunc i64 %62 to i32, !dbg !12
  %64 = insertelement <4 x i32> %61, i32 %63, i64 1, !dbg !12
  %65 = insertelement <4 x i32> %64, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1155 = insertelement <4 x i32> %65, i32 131072, i64 3, !dbg !12
  %66 = ptrtoint ptr addrspace(3) %56 to i32, !dbg !12
  %dma.buffer.lds1156 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %66), !dbg !12
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1156, i32 %dma.buffer.offset1154, <4 x i32> %dma.buffer.rsrc1155) #10, !dbg !12
  %address.local185 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local186 = shl nuw nsw i64 %address.local185, 3, !dbg !12
  %address.local187 = and i64 %address.local186, 3584, !dbg !12
  %address.local188 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local187, !dbg !12
  %address.local189 = ptrtoint ptr addrspace(3) %address.local188 to i64, !dbg !12
  %address.local190 = trunc i64 %address.local189 to i32, !dbg !12
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local190), !dbg !12
  %69 = zext i32 %68 to i64, !dbg !12
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !12
  %address.local191 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local192 = and i64 %address.local191, 16256, !dbg !12
  %address.local193 = lshr i64 %address.local185, 4, !dbg !12
  %address.local194 = and i64 %address.local193, 24, !dbg !12
  %address.local195 = or disjoint i64 %address.local192, %address.local194, !dbg !12
  %address.local196 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local195, !dbg !12
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %72 = ptrtoint ptr addrspace(1) %address.local196 to i64, !dbg !12
  %73 = sub i64 %72, %71, !dbg !12
  %dma.buffer.offset1151 = trunc i64 %73 to i32, !dbg !12
  %74 = trunc i64 %71 to i32, !dbg !12
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !12
  %76 = lshr i64 %71, 32, !dbg !12
  %77 = trunc i64 %76 to i32, !dbg !12
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !12
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1152 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !12
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !12
  %dma.buffer.lds1153 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !12
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1153, i32 %dma.buffer.offset1151, <4 x i32> %dma.buffer.rsrc1152) #10, !dbg !12
  %82 = lshr i64 %46, 12, !dbg !17
  %83 = add nsw i64 %82, %10, !dbg !17
  %84 = lshr i64 %46, 5, !dbg !17
  %85 = and i64 %84, 127, !dbg !17
  %86 = shl i64 %83, 19, !dbg !17
  %87 = and i64 %86, 16252928, !dbg !17
  %88 = shl i64 %84, 19, !dbg !17
  %89 = shl i64 %46, 14, !dbg !17
  %90 = sub i64 %89, %88, !dbg !17
  %91 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %87, !dbg !17
  %92 = getelementptr [2 x i8], ptr addrspace(1) %91, i64 %85, !dbg !17
  %93 = getelementptr [2 x i8], ptr addrspace(1) %92, i64 %90, !dbg !17
  %address.local197 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local175, !dbg !17
  %address.local198 = ptrtoint ptr addrspace(3) %address.local197 to i64, !dbg !17
  %address.local199 = trunc i64 %address.local198 to i32, !dbg !17
  %94 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local199), !dbg !17
  %95 = zext i32 %94 to i64, !dbg !17
  %96 = inttoptr i64 %95 to ptr addrspace(3), !dbg !17
  %address.local200 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %address.local183, !dbg !17
  %97 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %98 = ptrtoint ptr addrspace(1) %address.local200 to i64, !dbg !17
  %99 = sub i64 %98, %97, !dbg !17
  %dma.buffer.offset1148 = trunc i64 %99 to i32, !dbg !17
  %100 = trunc i64 %97 to i32, !dbg !17
  %101 = insertelement <4 x i32> poison, i32 %100, i64 0, !dbg !17
  %102 = lshr i64 %97, 32, !dbg !17
  %103 = trunc i64 %102 to i32, !dbg !17
  %104 = insertelement <4 x i32> %101, i32 %103, i64 1, !dbg !17
  %105 = insertelement <4 x i32> %104, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1149 = insertelement <4 x i32> %105, i32 131072, i64 3, !dbg !17
  %106 = ptrtoint ptr addrspace(3) %96 to i32, !dbg !17
  %dma.buffer.lds1150 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %106), !dbg !17
  %107 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1150, i32 %dma.buffer.offset1148, <4 x i32> %dma.buffer.rsrc1149) #10, !dbg !17
  %address.local201 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local187, !dbg !17
  %address.local202 = ptrtoint ptr addrspace(3) %address.local201 to i64, !dbg !17
  %address.local203 = trunc i64 %address.local202 to i32, !dbg !17
  %108 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local203), !dbg !17
  %109 = zext i32 %108 to i64, !dbg !17
  %110 = inttoptr i64 %109 to ptr addrspace(3), !dbg !17
  %address.local204 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %address.local195, !dbg !17
  %111 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %112 = ptrtoint ptr addrspace(1) %address.local204 to i64, !dbg !17
  %113 = sub i64 %112, %111, !dbg !17
  %dma.buffer.offset1145 = trunc i64 %113 to i32, !dbg !17
  %114 = trunc i64 %111 to i32, !dbg !17
  %115 = insertelement <4 x i32> poison, i32 %114, i64 0, !dbg !17
  %116 = lshr i64 %111, 32, !dbg !17
  %117 = trunc i64 %116 to i32, !dbg !17
  %118 = insertelement <4 x i32> %115, i32 %117, i64 1, !dbg !17
  %119 = insertelement <4 x i32> %118, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1146 = insertelement <4 x i32> %119, i32 131072, i64 3, !dbg !17
  %120 = ptrtoint ptr addrspace(3) %110 to i32, !dbg !17
  %dma.buffer.lds1147 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %120), !dbg !17
  %121 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1147, i32 %dma.buffer.offset1145, <4 x i32> %dma.buffer.rsrc1146) #10, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %address.local205 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local175, !dbg !12
  %address.local206 = ptrtoint ptr addrspace(3) %address.local205 to i64, !dbg !12
  %address.local207 = trunc i64 %address.local206 to i32, !dbg !12
  %122 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local207), !dbg !12
  %123 = zext i32 %122 to i64, !dbg !12
  %124 = inttoptr i64 %123 to ptr addrspace(3), !dbg !12
  %address.local208 = getelementptr [2 x i8], ptr addrspace(1) %35, i64 %address.local183, !dbg !12
  %125 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %126 = ptrtoint ptr addrspace(1) %address.local208 to i64, !dbg !12
  %127 = sub i64 %126, %125, !dbg !12
  %dma.buffer.offset1142 = trunc i64 %127 to i32, !dbg !12
  %128 = trunc i64 %125 to i32, !dbg !12
  %129 = insertelement <4 x i32> poison, i32 %128, i64 0, !dbg !12
  %130 = lshr i64 %125, 32, !dbg !12
  %131 = trunc i64 %130 to i32, !dbg !12
  %132 = insertelement <4 x i32> %129, i32 %131, i64 1, !dbg !12
  %133 = insertelement <4 x i32> %132, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1143 = insertelement <4 x i32> %133, i32 131072, i64 3, !dbg !12
  %134 = ptrtoint ptr addrspace(3) %124 to i32, !dbg !12
  %dma.buffer.lds1144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %134), !dbg !12
  %135 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1144, i32 %dma.buffer.offset1142, <4 x i32> %dma.buffer.rsrc1143) #10, !dbg !12
  %address.local209 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local187, !dbg !12
  %address.local210 = ptrtoint ptr addrspace(3) %address.local209 to i64, !dbg !12
  %address.local211 = trunc i64 %address.local210 to i32, !dbg !12
  %136 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local211), !dbg !12
  %137 = zext i32 %136 to i64, !dbg !12
  %138 = inttoptr i64 %137 to ptr addrspace(3), !dbg !12
  %address.local212 = getelementptr [2 x i8], ptr addrspace(1) %35, i64 %address.local195, !dbg !12
  %139 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %140 = ptrtoint ptr addrspace(1) %address.local212 to i64, !dbg !12
  %141 = sub i64 %140, %139, !dbg !12
  %dma.buffer.offset1139 = trunc i64 %141 to i32, !dbg !12
  %142 = trunc i64 %139 to i32, !dbg !12
  %143 = insertelement <4 x i32> poison, i32 %142, i64 0, !dbg !12
  %144 = lshr i64 %139, 32, !dbg !12
  %145 = trunc i64 %144 to i32, !dbg !12
  %146 = insertelement <4 x i32> %143, i32 %145, i64 1, !dbg !12
  %147 = insertelement <4 x i32> %146, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1140 = insertelement <4 x i32> %147, i32 131072, i64 3, !dbg !12
  %148 = ptrtoint ptr addrspace(3) %138 to i32, !dbg !12
  %dma.buffer.lds1141 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %148), !dbg !12
  %149 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1141, i32 %dma.buffer.offset1139, <4 x i32> %dma.buffer.rsrc1140) #10, !dbg !12
  %150 = add i64 %46, %36, !dbg !17
  %151 = add i64 %150, %37, !dbg !17
  %152 = add i64 %151, 1024, !dbg !17
  %153 = icmp slt i64 %152, 0, !dbg !17
  %154 = sub i64 -1025, %151, !dbg !17
  %155 = select i1 %153, i64 %154, i64 %152, !dbg !17
  %156 = sdiv i64 %155, 4096, !dbg !17
  %.lobit4 = ashr i64 %152, 63, !dbg !17
  %157 = xor i64 %156, %.lobit4, !dbg !17
  %158 = add nsw i64 %84, %40, !dbg !17
  %159 = add nsw i64 %158, 32, !dbg !17
  %160 = icmp slt i64 %158, -32, !dbg !17
  %161 = sub nsw i64 -33, %158, !dbg !17
  %162 = select i1 %160, i64 %161, i64 %159, !dbg !17
  %163 = sdiv i64 %162, 128, !dbg !17
  %164 = sext i1 %160 to i64, !dbg !17
  %165 = xor i64 %163, %164, !dbg !17
  %166 = shl nsw i64 %165, 7, !dbg !17
  %167 = sub i64 %40, %166, !dbg !17
  %168 = shl i64 %157, 19, !dbg !17
  %169 = and i64 %168, 16252928, !dbg !17
  %170 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %169, !dbg !17
  %171 = getelementptr [2 x i8], ptr addrspace(1) %170, i64 %84, !dbg !17
  %172 = getelementptr [2 x i8], ptr addrspace(1) %171, i64 %167, !dbg !17
  %173 = getelementptr i8, ptr addrspace(1) %172, i64 64, !dbg !17
  %174 = getelementptr [2 x i8], ptr addrspace(1) %173, i64 %90, !dbg !17
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local175, !dbg !17
  %address.local214 = ptrtoint ptr addrspace(3) %address.local213 to i64, !dbg !17
  %address.local215 = trunc i64 %address.local214 to i32, !dbg !17
  %175 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local215), !dbg !17
  %176 = zext i32 %175 to i64, !dbg !17
  %177 = inttoptr i64 %176 to ptr addrspace(3), !dbg !17
  %address.local216 = getelementptr [2 x i8], ptr addrspace(1) %174, i64 %address.local183, !dbg !17
  %178 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %179 = ptrtoint ptr addrspace(1) %address.local216 to i64, !dbg !17
  %180 = sub i64 %179, %178, !dbg !17
  %dma.buffer.offset1136 = trunc i64 %180 to i32, !dbg !17
  %181 = trunc i64 %178 to i32, !dbg !17
  %182 = insertelement <4 x i32> poison, i32 %181, i64 0, !dbg !17
  %183 = lshr i64 %178, 32, !dbg !17
  %184 = trunc i64 %183 to i32, !dbg !17
  %185 = insertelement <4 x i32> %182, i32 %184, i64 1, !dbg !17
  %186 = insertelement <4 x i32> %185, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1137 = insertelement <4 x i32> %186, i32 131072, i64 3, !dbg !17
  %187 = ptrtoint ptr addrspace(3) %177 to i32, !dbg !17
  %dma.buffer.lds1138 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %187), !dbg !17
  %188 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1138, i32 %dma.buffer.offset1136, <4 x i32> %dma.buffer.rsrc1137) #10, !dbg !17
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local187, !dbg !17
  %address.local218 = ptrtoint ptr addrspace(3) %address.local217 to i64, !dbg !17
  %address.local219 = trunc i64 %address.local218 to i32, !dbg !17
  %189 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local219), !dbg !17
  %190 = zext i32 %189 to i64, !dbg !17
  %191 = inttoptr i64 %190 to ptr addrspace(3), !dbg !17
  %address.local220 = getelementptr [2 x i8], ptr addrspace(1) %174, i64 %address.local195, !dbg !17
  %192 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %193 = ptrtoint ptr addrspace(1) %address.local220 to i64, !dbg !17
  %194 = sub i64 %193, %192, !dbg !17
  %dma.buffer.offset1133 = trunc i64 %194 to i32, !dbg !17
  %195 = trunc i64 %192 to i32, !dbg !17
  %196 = insertelement <4 x i32> poison, i32 %195, i64 0, !dbg !17
  %197 = lshr i64 %192, 32, !dbg !17
  %198 = trunc i64 %197 to i32, !dbg !17
  %199 = insertelement <4 x i32> %196, i32 %198, i64 1, !dbg !17
  %200 = insertelement <4 x i32> %199, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1134 = insertelement <4 x i32> %200, i32 131072, i64 3, !dbg !17
  %201 = ptrtoint ptr addrspace(3) %191 to i32, !dbg !17
  %dma.buffer.lds1135 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %201), !dbg !17
  %202 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1135, i32 %dma.buffer.offset1133, <4 x i32> %dma.buffer.rsrc1134) #10, !dbg !17
  %address.local221 = lshr i32 %address.thread, 2, !dbg !13
  %address.local222 = and i32 %address.local221, 4, !dbg !13
  %address.local223 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local224 = and i32 %address.local223, 384, !dbg !13
  %address.local225 = or disjoint i32 %address.local222, %address.local224, !dbg !13
  %address.local226 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local227 = and i32 %address.local226, 120, !dbg !13
  %address.local228 = or disjoint i32 %address.local225, %address.local227, !dbg !13
  %address.local229 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local230 = and i32 %address.local229, 1024, !dbg !13
  %address.local231 = or disjoint i32 %address.local228, %address.local230, !dbg !13
  %address.local232 = zext nneg i32 %address.local231 to i64, !dbg !13
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local232, !dbg !13
  %203 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !13
  %address.local234 = lshr i32 %address.thread, 2, !dbg !13
  %address.local235 = and i32 %address.local234, 4, !dbg !13
  %address.local236 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local237 = and i32 %address.local236, 384, !dbg !13
  %address.local238 = or disjoint i32 %address.local235, %address.local237, !dbg !13
  %address.local239 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local240 = and i32 %address.local239, 120, !dbg !13
  %address.local241 = or disjoint i32 %address.local238, %address.local240, !dbg !13
  %address.local242 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local243 = and i32 %address.local242, 1024, !dbg !13
  %address.local244 = or disjoint i32 %address.local241, %address.local243, !dbg !13
  %address.local245 = or disjoint i32 %address.local244, 2048, !dbg !13
  %address.local246 = zext nneg i32 %address.local245 to i64, !dbg !13
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local246, !dbg !13
  %204 = load <4 x half>, ptr addrspace(3) %address.local247, align 2, !dbg !13
  %address.local248 = lshr i32 %address.thread, 2, !dbg !13
  %address.local249 = and i32 %address.local248, 4, !dbg !13
  %address.local250 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local251 = and i32 %address.local250, 384, !dbg !13
  %address.local252 = or disjoint i32 %address.local249, %address.local251, !dbg !13
  %address.local253 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local254 = and i32 %address.local253, 120, !dbg !13
  %address.local255 = or disjoint i32 %address.local252, %address.local254, !dbg !13
  %address.local256 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local257 = and i32 %address.local256, 1024, !dbg !13
  %address.local258 = or disjoint i32 %address.local255, %address.local257, !dbg !13
  %address.local259 = or disjoint i32 %address.local258, 512, !dbg !13
  %address.local260 = zext nneg i32 %address.local259 to i64, !dbg !13
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local260, !dbg !13
  %205 = load <4 x half>, ptr addrspace(3) %address.local261, align 2, !dbg !13
  %206 = fmul <4 x half> %203, splat (half 0xH3015), !dbg !13
  %address.local262 = lshr i32 %address.thread, 2, !dbg !13
  %address.local263 = and i32 %address.local262, 4, !dbg !13
  %address.local264 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local265 = and i32 %address.local264, 384, !dbg !13
  %address.local266 = or disjoint i32 %address.local263, %address.local265, !dbg !13
  %address.local267 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local268 = and i32 %address.local267, 120, !dbg !13
  %address.local269 = or disjoint i32 %address.local266, %address.local268, !dbg !13
  %address.local270 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local271 = and i32 %address.local270, 1024, !dbg !13
  %address.local272 = or disjoint i32 %address.local269, %address.local271, !dbg !13
  %address.local273 = or disjoint i32 %address.local272, 2560, !dbg !13
  %address.local274 = zext nneg i32 %address.local273 to i64, !dbg !13
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local274, !dbg !13
  %207 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !13
  %208 = fmul <4 x half> %204, splat (half 0xH3015), !dbg !13
  %address.local276 = lshr i32 %address.thread, 2, !dbg !18
  %address.local277 = and i32 %address.local276, 4, !dbg !18
  %address.local278 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local279 = and i32 %address.local278, 24, !dbg !18
  %address.local280 = or disjoint i32 %address.local277, %address.local279, !dbg !18
  %address.local281 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local282 = and i32 %address.local281, 1024, !dbg !18
  %address.local283 = or disjoint i32 %address.local280, %address.local282, !dbg !18
  %address.local284 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local285 = and i32 %address.local284, 384, !dbg !18
  %address.local286 = or disjoint i32 %address.local283, %address.local285, !dbg !18
  %address.local287 = zext nneg i32 %address.local286 to i64, !dbg !18
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local287, !dbg !18
  %209 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !18
  %210 = fmul <4 x half> %205, splat (half 0xH3015), !dbg !13
  %address.local289 = lshr i32 %address.thread, 2, !dbg !18
  %address.local290 = and i32 %address.local289, 4, !dbg !18
  %address.local291 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local292 = and i32 %address.local291, 24, !dbg !18
  %address.local293 = or disjoint i32 %address.local290, %address.local292, !dbg !18
  %address.local294 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local295 = and i32 %address.local294, 1024, !dbg !18
  %address.local296 = or disjoint i32 %address.local293, %address.local295, !dbg !18
  %address.local297 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local298 = and i32 %address.local297, 384, !dbg !18
  %address.local299 = or disjoint i32 %address.local296, %address.local298, !dbg !18
  %address.local300 = or disjoint i32 %address.local299, 2048, !dbg !18
  %address.local301 = zext nneg i32 %address.local300 to i64, !dbg !18
  %address.local302 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local301, !dbg !18
  %211 = load <4 x half>, ptr addrspace(3) %address.local302, align 2, !dbg !18
  %212 = fmul <4 x half> %207, splat (half 0xH3015), !dbg !13
  %address.local303 = lshr i32 %address.thread, 2, !dbg !18
  %address.local304 = and i32 %address.local303, 4, !dbg !18
  %address.local305 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local306 = and i32 %address.local305, 24, !dbg !18
  %address.local307 = or disjoint i32 %address.local304, %address.local306, !dbg !18
  %address.local308 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local309 = and i32 %address.local308, 1024, !dbg !18
  %address.local310 = or disjoint i32 %address.local307, %address.local309, !dbg !18
  %address.local311 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local312 = and i32 %address.local311, 384, !dbg !18
  %address.local313 = or disjoint i32 %address.local310, %address.local312, !dbg !18
  %address.local314 = or disjoint i32 %address.local313, 32, !dbg !18
  %address.local315 = zext nneg i32 %address.local314 to i64, !dbg !18
  %address.local316 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local315, !dbg !18
  %213 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !18
  %214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %206, <4 x half> %209, <4 x float> zeroinitializer) #9, !dbg !18
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %208, <4 x half> %211, <4 x float> %214) #9, !dbg !18
  %address.local317 = lshr i32 %address.thread, 2, !dbg !18
  %address.local318 = and i32 %address.local317, 4, !dbg !18
  %address.local319 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local320 = and i32 %address.local319, 24, !dbg !18
  %address.local321 = or disjoint i32 %address.local318, %address.local320, !dbg !18
  %address.local322 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local323 = and i32 %address.local322, 1024, !dbg !18
  %address.local324 = or disjoint i32 %address.local321, %address.local323, !dbg !18
  %address.local325 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local326 = and i32 %address.local325, 384, !dbg !18
  %address.local327 = or disjoint i32 %address.local324, %address.local326, !dbg !18
  %address.local328 = or disjoint i32 %address.local327, 2080, !dbg !18
  %address.local329 = zext nneg i32 %address.local328 to i64, !dbg !18
  %address.local330 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local329, !dbg !18
  %216 = load <4 x half>, ptr addrspace(3) %address.local330, align 2, !dbg !18
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %206, <4 x half> %213, <4 x float> zeroinitializer) #9, !dbg !18
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %208, <4 x half> %216, <4 x float> %217) #9, !dbg !18
  %address.local331 = lshr i32 %address.thread, 2, !dbg !18
  %address.local332 = and i32 %address.local331, 4, !dbg !18
  %address.local333 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local334 = and i32 %address.local333, 24, !dbg !18
  %address.local335 = or disjoint i32 %address.local332, %address.local334, !dbg !18
  %address.local336 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local337 = and i32 %address.local336, 1024, !dbg !18
  %address.local338 = or disjoint i32 %address.local335, %address.local337, !dbg !18
  %address.local339 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local340 = and i32 %address.local339, 384, !dbg !18
  %address.local341 = or disjoint i32 %address.local338, %address.local340, !dbg !18
  %address.local342 = or disjoint i32 %address.local341, 64, !dbg !18
  %address.local343 = zext nneg i32 %address.local342 to i64, !dbg !18
  %address.local344 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local343, !dbg !18
  %219 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !18
  %address.local345 = lshr i32 %address.thread, 2, !dbg !18
  %address.local346 = and i32 %address.local345, 4, !dbg !18
  %address.local347 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local348 = and i32 %address.local347, 24, !dbg !18
  %address.local349 = or disjoint i32 %address.local346, %address.local348, !dbg !18
  %address.local350 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local351 = and i32 %address.local350, 1024, !dbg !18
  %address.local352 = or disjoint i32 %address.local349, %address.local351, !dbg !18
  %address.local353 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local354 = and i32 %address.local353, 384, !dbg !18
  %address.local355 = or disjoint i32 %address.local352, %address.local354, !dbg !18
  %address.local356 = or disjoint i32 %address.local355, 2112, !dbg !18
  %address.local357 = zext nneg i32 %address.local356 to i64, !dbg !18
  %address.local358 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local357, !dbg !18
  %220 = load <4 x half>, ptr addrspace(3) %address.local358, align 2, !dbg !18
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %206, <4 x half> %219, <4 x float> zeroinitializer) #9, !dbg !18
  %address.local359 = lshr i32 %address.thread, 2, !dbg !18
  %address.local360 = and i32 %address.local359, 4, !dbg !18
  %address.local361 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local362 = and i32 %address.local361, 24, !dbg !18
  %address.local363 = or disjoint i32 %address.local360, %address.local362, !dbg !18
  %address.local364 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local365 = and i32 %address.local364, 1024, !dbg !18
  %address.local366 = or disjoint i32 %address.local363, %address.local365, !dbg !18
  %address.local367 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local368 = and i32 %address.local367, 384, !dbg !18
  %address.local369 = or disjoint i32 %address.local366, %address.local368, !dbg !18
  %address.local370 = or disjoint i32 %address.local369, 96, !dbg !18
  %address.local371 = zext nneg i32 %address.local370 to i64, !dbg !18
  %address.local372 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local371, !dbg !18
  %222 = load <4 x half>, ptr addrspace(3) %address.local372, align 2, !dbg !18
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %208, <4 x half> %220, <4 x float> %221) #9, !dbg !18
  %address.local373 = lshr i32 %address.thread, 2, !dbg !18
  %address.local374 = and i32 %address.local373, 4, !dbg !18
  %address.local375 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local376 = and i32 %address.local375, 24, !dbg !18
  %address.local377 = or disjoint i32 %address.local374, %address.local376, !dbg !18
  %address.local378 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local379 = and i32 %address.local378, 1024, !dbg !18
  %address.local380 = or disjoint i32 %address.local377, %address.local379, !dbg !18
  %address.local381 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local382 = and i32 %address.local381, 384, !dbg !18
  %address.local383 = or disjoint i32 %address.local380, %address.local382, !dbg !18
  %address.local384 = or disjoint i32 %address.local383, 2144, !dbg !18
  %address.local385 = zext nneg i32 %address.local384 to i64, !dbg !18
  %address.local386 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local385, !dbg !18
  %224 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !18
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %206, <4 x half> %222, <4 x float> zeroinitializer) #9, !dbg !18
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %208, <4 x half> %224, <4 x float> %225) #9, !dbg !18
  %address.local387 = lshr i32 %address.thread, 2, !dbg !18
  %address.local388 = and i32 %address.local387, 4, !dbg !18
  %address.local389 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local390 = and i32 %address.local389, 24, !dbg !18
  %address.local391 = or disjoint i32 %address.local388, %address.local390, !dbg !18
  %address.local392 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local393 = and i32 %address.local392, 1024, !dbg !18
  %address.local394 = or disjoint i32 %address.local391, %address.local393, !dbg !18
  %address.local395 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local396 = and i32 %address.local395, 384, !dbg !18
  %address.local397 = or disjoint i32 %address.local394, %address.local396, !dbg !18
  %address.local398 = or disjoint i32 %address.local397, 512, !dbg !18
  %address.local399 = zext nneg i32 %address.local398 to i64, !dbg !18
  %address.local400 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local399, !dbg !18
  %227 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !18
  %address.local401 = lshr i32 %address.thread, 2, !dbg !18
  %address.local402 = and i32 %address.local401, 4, !dbg !18
  %address.local403 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local404 = and i32 %address.local403, 24, !dbg !18
  %address.local405 = or disjoint i32 %address.local402, %address.local404, !dbg !18
  %address.local406 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local407 = and i32 %address.local406, 1024, !dbg !18
  %address.local408 = or disjoint i32 %address.local405, %address.local407, !dbg !18
  %address.local409 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local410 = and i32 %address.local409, 384, !dbg !18
  %address.local411 = or disjoint i32 %address.local408, %address.local410, !dbg !18
  %address.local412 = or disjoint i32 %address.local411, 2560, !dbg !18
  %address.local413 = zext nneg i32 %address.local412 to i64, !dbg !18
  %address.local414 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local413, !dbg !18
  %228 = load <4 x half>, ptr addrspace(3) %address.local414, align 2, !dbg !18
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %206, <4 x half> %227, <4 x float> zeroinitializer) #9, !dbg !18
  %address.local415 = lshr i32 %address.thread, 2, !dbg !18
  %address.local416 = and i32 %address.local415, 4, !dbg !18
  %address.local417 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local418 = and i32 %address.local417, 24, !dbg !18
  %address.local419 = or disjoint i32 %address.local416, %address.local418, !dbg !18
  %address.local420 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local421 = and i32 %address.local420, 1024, !dbg !18
  %address.local422 = or disjoint i32 %address.local419, %address.local421, !dbg !18
  %address.local423 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local424 = and i32 %address.local423, 384, !dbg !18
  %address.local425 = or disjoint i32 %address.local422, %address.local424, !dbg !18
  %address.local426 = or disjoint i32 %address.local425, 544, !dbg !18
  %address.local427 = zext nneg i32 %address.local426 to i64, !dbg !18
  %address.local428 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local427, !dbg !18
  %230 = load <4 x half>, ptr addrspace(3) %address.local428, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %208, <4 x half> %228, <4 x float> %229) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %address.local429 = lshr i32 %address.thread, 2, !dbg !18
  %address.local430 = and i32 %address.local429, 4, !dbg !18
  %address.local431 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local432 = and i32 %address.local431, 24, !dbg !18
  %address.local433 = or disjoint i32 %address.local430, %address.local432, !dbg !18
  %address.local434 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local435 = and i32 %address.local434, 1024, !dbg !18
  %address.local436 = or disjoint i32 %address.local433, %address.local435, !dbg !18
  %address.local437 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local438 = and i32 %address.local437, 384, !dbg !18
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !18
  %address.local440 = or disjoint i32 %address.local439, 2592, !dbg !18
  %address.local441 = zext nneg i32 %address.local440 to i64, !dbg !18
  %address.local442 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local441, !dbg !18
  %232 = load <4 x half>, ptr addrspace(3) %address.local442, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %206, <4 x half> %230, <4 x float> zeroinitializer) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %208, <4 x half> %232, <4 x float> %233) #9, !dbg !18
  %address.local443 = lshr i32 %address.thread, 2, !dbg !18
  %address.local444 = and i32 %address.local443, 4, !dbg !18
  %address.local445 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local446 = and i32 %address.local445, 24, !dbg !18
  %address.local447 = or disjoint i32 %address.local444, %address.local446, !dbg !18
  %address.local448 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local449 = and i32 %address.local448, 1024, !dbg !18
  %address.local450 = or disjoint i32 %address.local447, %address.local449, !dbg !18
  %address.local451 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local452 = and i32 %address.local451, 384, !dbg !18
  %address.local453 = or disjoint i32 %address.local450, %address.local452, !dbg !18
  %address.local454 = or disjoint i32 %address.local453, 576, !dbg !18
  %address.local455 = zext nneg i32 %address.local454 to i64, !dbg !18
  %address.local456 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local455, !dbg !18
  %235 = load <4 x half>, ptr addrspace(3) %address.local456, align 2, !dbg !18
  %address.local457 = lshr i32 %address.thread, 2, !dbg !18
  %address.local458 = and i32 %address.local457, 4, !dbg !18
  %address.local459 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local460 = and i32 %address.local459, 24, !dbg !18
  %address.local461 = or disjoint i32 %address.local458, %address.local460, !dbg !18
  %address.local462 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local463 = and i32 %address.local462, 1024, !dbg !18
  %address.local464 = or disjoint i32 %address.local461, %address.local463, !dbg !18
  %address.local465 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local466 = and i32 %address.local465, 384, !dbg !18
  %address.local467 = or disjoint i32 %address.local464, %address.local466, !dbg !18
  %address.local468 = or disjoint i32 %address.local467, 2624, !dbg !18
  %address.local469 = zext nneg i32 %address.local468 to i64, !dbg !18
  %address.local470 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local469, !dbg !18
  %236 = load <4 x half>, ptr addrspace(3) %address.local470, align 2, !dbg !18
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %206, <4 x half> %235, <4 x float> zeroinitializer) #9, !dbg !18
  %address.local471 = lshr i32 %address.thread, 2, !dbg !18
  %address.local472 = and i32 %address.local471, 4, !dbg !18
  %address.local473 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local474 = and i32 %address.local473, 24, !dbg !18
  %address.local475 = or disjoint i32 %address.local472, %address.local474, !dbg !18
  %address.local476 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local477 = and i32 %address.local476, 1024, !dbg !18
  %address.local478 = or disjoint i32 %address.local475, %address.local477, !dbg !18
  %address.local479 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local480 = and i32 %address.local479, 384, !dbg !18
  %address.local481 = or disjoint i32 %address.local478, %address.local480, !dbg !18
  %address.local482 = or disjoint i32 %address.local481, 608, !dbg !18
  %address.local483 = zext nneg i32 %address.local482 to i64, !dbg !18
  %address.local484 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local483, !dbg !18
  %238 = load <4 x half>, ptr addrspace(3) %address.local484, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %208, <4 x half> %236, <4 x float> %237) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %address.local485 = lshr i32 %address.thread, 2, !dbg !18
  %address.local486 = and i32 %address.local485, 4, !dbg !18
  %address.local487 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local488 = and i32 %address.local487, 24, !dbg !18
  %address.local489 = or disjoint i32 %address.local486, %address.local488, !dbg !18
  %address.local490 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local491 = and i32 %address.local490, 1024, !dbg !18
  %address.local492 = or disjoint i32 %address.local489, %address.local491, !dbg !18
  %address.local493 = shl nuw nsw i32 %address.thread, 7, !dbg !18
  %address.local494 = and i32 %address.local493, 384, !dbg !18
  %address.local495 = or disjoint i32 %address.local492, %address.local494, !dbg !18
  %address.local496 = or disjoint i32 %address.local495, 2656, !dbg !18
  %address.local497 = zext nneg i32 %address.local496 to i64, !dbg !18
  %address.local498 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local497, !dbg !18
  %240 = load <4 x half>, ptr addrspace(3) %address.local498, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %206, <4 x half> %238, <4 x float> zeroinitializer) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %208, <4 x half> %240, <4 x float> %241) #9, !dbg !18
  %243 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !18
  %244 = load <4 x half>, ptr addrspace(3) %address.local302, align 2, !dbg !18
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %243, <4 x float> zeroinitializer) #9, !dbg !18
  %246 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %244, <4 x float> %245) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %248 = load <4 x half>, ptr addrspace(3) %address.local330, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %246, <4 x float> zeroinitializer) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %248, <4 x float> %249) #9, !dbg !18
  %251 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !18
  %252 = load <4 x half>, ptr addrspace(3) %address.local358, align 2, !dbg !18
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %251, <4 x float> zeroinitializer) #9, !dbg !18
  %254 = load <4 x half>, ptr addrspace(3) %address.local372, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %252, <4 x float> %253) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %256 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %254, <4 x float> zeroinitializer) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %256, <4 x float> %257) #9, !dbg !18
  %259 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !18
  %260 = load <4 x half>, ptr addrspace(3) %address.local414, align 2, !dbg !18
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %259, <4 x float> zeroinitializer) #9, !dbg !18
  %262 = load <4 x half>, ptr addrspace(3) %address.local428, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %260, <4 x float> %261) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %264 = load <4 x half>, ptr addrspace(3) %address.local442, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %262, <4 x float> zeroinitializer) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %264, <4 x float> %265) #9, !dbg !18
  %267 = load <4 x half>, ptr addrspace(3) %address.local456, align 2, !dbg !18
  %268 = load <4 x half>, ptr addrspace(3) %address.local470, align 2, !dbg !18
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %267, <4 x float> zeroinitializer) #9, !dbg !18
  %270 = load <4 x half>, ptr addrspace(3) %address.local484, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %268, <4 x float> %269) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %272 = load <4 x half>, ptr addrspace(3) %address.local498, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %210, <4 x half> %270, <4 x float> zeroinitializer) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %212, <4 x half> %272, <4 x float> %273) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %275 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local178), !dbg !12
  %276 = zext i32 %275 to i64, !dbg !12
  %277 = inttoptr i64 %276 to ptr addrspace(3), !dbg !12
  %address.local499 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %address.local183, !dbg !12
  %278 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %279 = ptrtoint ptr addrspace(1) %address.local499 to i64, !dbg !12
  %280 = sub i64 %279, %278, !dbg !12
  %dma.buffer.offset1130 = trunc i64 %280 to i32, !dbg !12
  %281 = trunc i64 %278 to i32, !dbg !12
  %282 = insertelement <4 x i32> poison, i32 %281, i64 0, !dbg !12
  %283 = lshr i64 %278, 32, !dbg !12
  %284 = trunc i64 %283 to i32, !dbg !12
  %285 = insertelement <4 x i32> %282, i32 %284, i64 1, !dbg !12
  %286 = insertelement <4 x i32> %285, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1131 = insertelement <4 x i32> %286, i32 131072, i64 3, !dbg !12
  %287 = ptrtoint ptr addrspace(3) %277 to i32, !dbg !12
  %dma.buffer.lds1132 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %287), !dbg !12
  %288 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1132, i32 %dma.buffer.offset1130, <4 x i32> %dma.buffer.rsrc1131) #10, !dbg !12
  %289 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local190), !dbg !12
  %290 = zext i32 %289 to i64, !dbg !12
  %291 = inttoptr i64 %290 to ptr addrspace(3), !dbg !12
  %address.local500 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %address.local195, !dbg !12
  %292 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %293 = ptrtoint ptr addrspace(1) %address.local500 to i64, !dbg !12
  %294 = sub i64 %293, %292, !dbg !12
  %dma.buffer.offset1127 = trunc i64 %294 to i32, !dbg !12
  %295 = trunc i64 %292 to i32, !dbg !12
  %296 = insertelement <4 x i32> poison, i32 %295, i64 0, !dbg !12
  %297 = lshr i64 %292, 32, !dbg !12
  %298 = trunc i64 %297 to i32, !dbg !12
  %299 = insertelement <4 x i32> %296, i32 %298, i64 1, !dbg !12
  %300 = insertelement <4 x i32> %299, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1128 = insertelement <4 x i32> %300, i32 131072, i64 3, !dbg !12
  %301 = ptrtoint ptr addrspace(3) %291 to i32, !dbg !12
  %dma.buffer.lds1129 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %301), !dbg !12
  %302 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1129, i32 %dma.buffer.offset1127, <4 x i32> %dma.buffer.rsrc1128) #10, !dbg !12
  %303 = add i64 %151, 2048, !dbg !17
  %304 = icmp slt i64 %303, 0, !dbg !17
  %305 = sub i64 -2049, %151, !dbg !17
  %306 = select i1 %304, i64 %305, i64 %303, !dbg !17
  %307 = sdiv i64 %306, 4096, !dbg !17
  %.lobit5 = ashr i64 %303, 63, !dbg !17
  %308 = xor i64 %307, %.lobit5, !dbg !17
  %309 = add nsw i64 %158, 64, !dbg !17
  %310 = icmp slt i64 %158, -64, !dbg !17
  %311 = sub nsw i64 -65, %158, !dbg !17
  %312 = select i1 %310, i64 %311, i64 %309, !dbg !17
  %313 = sdiv i64 %312, 128, !dbg !17
  %314 = sext i1 %310 to i64, !dbg !17
  %315 = xor i64 %313, %314, !dbg !17
  %316 = shl nsw i64 %315, 7, !dbg !17
  %317 = sub i64 %40, %316, !dbg !17
  %318 = shl i64 %308, 19, !dbg !17
  %319 = and i64 %318, 16252928, !dbg !17
  %320 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %319, !dbg !17
  %321 = getelementptr [2 x i8], ptr addrspace(1) %320, i64 %84, !dbg !17
  %322 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %317, !dbg !17
  %323 = getelementptr i8, ptr addrspace(1) %322, i64 128, !dbg !17
  %324 = getelementptr [2 x i8], ptr addrspace(1) %323, i64 %90, !dbg !17
  %325 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local199), !dbg !17
  %326 = zext i32 %325 to i64, !dbg !17
  %327 = inttoptr i64 %326 to ptr addrspace(3), !dbg !17
  %address.local501 = getelementptr [2 x i8], ptr addrspace(1) %324, i64 %address.local183, !dbg !17
  %328 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %329 = ptrtoint ptr addrspace(1) %address.local501 to i64, !dbg !17
  %330 = sub i64 %329, %328, !dbg !17
  %dma.buffer.offset1124 = trunc i64 %330 to i32, !dbg !17
  %331 = trunc i64 %328 to i32, !dbg !17
  %332 = insertelement <4 x i32> poison, i32 %331, i64 0, !dbg !17
  %333 = lshr i64 %328, 32, !dbg !17
  %334 = trunc i64 %333 to i32, !dbg !17
  %335 = insertelement <4 x i32> %332, i32 %334, i64 1, !dbg !17
  %336 = insertelement <4 x i32> %335, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1125 = insertelement <4 x i32> %336, i32 131072, i64 3, !dbg !17
  %337 = ptrtoint ptr addrspace(3) %327 to i32, !dbg !17
  %dma.buffer.lds1126 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %337), !dbg !17
  %338 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1126, i32 %dma.buffer.offset1124, <4 x i32> %dma.buffer.rsrc1125) #10, !dbg !17
  %339 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local203), !dbg !17
  %340 = zext i32 %339 to i64, !dbg !17
  %341 = inttoptr i64 %340 to ptr addrspace(3), !dbg !17
  %address.local502 = getelementptr [2 x i8], ptr addrspace(1) %324, i64 %address.local195, !dbg !17
  %342 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %343 = ptrtoint ptr addrspace(1) %address.local502 to i64, !dbg !17
  %344 = sub i64 %343, %342, !dbg !17
  %dma.buffer.offset1121 = trunc i64 %344 to i32, !dbg !17
  %345 = trunc i64 %342 to i32, !dbg !17
  %346 = insertelement <4 x i32> poison, i32 %345, i64 0, !dbg !17
  %347 = lshr i64 %342, 32, !dbg !17
  %348 = trunc i64 %347 to i32, !dbg !17
  %349 = insertelement <4 x i32> %346, i32 %348, i64 1, !dbg !17
  %350 = insertelement <4 x i32> %349, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1122 = insertelement <4 x i32> %350, i32 131072, i64 3, !dbg !17
  %351 = ptrtoint ptr addrspace(3) %341 to i32, !dbg !17
  %dma.buffer.lds1123 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %351), !dbg !17
  %352 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1123, i32 %dma.buffer.offset1121, <4 x i32> %dma.buffer.rsrc1122) #10, !dbg !17
  %address.local503 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local232, !dbg !13
  %353 = load <4 x half>, ptr addrspace(3) %address.local503, align 2, !dbg !13
  %354 = fmul <4 x half> %353, splat (half 0xH3015), !dbg !13
  %address.local504 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local246, !dbg !13
  %355 = load <4 x half>, ptr addrspace(3) %address.local504, align 2, !dbg !13
  %address.local505 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local260, !dbg !13
  %356 = load <4 x half>, ptr addrspace(3) %address.local505, align 2, !dbg !13
  %357 = fmul <4 x half> %355, splat (half 0xH3015), !dbg !13
  %358 = fmul <4 x half> %356, splat (half 0xH3015), !dbg !13
  %address.local506 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local274, !dbg !13
  %359 = load <4 x half>, ptr addrspace(3) %address.local506, align 2, !dbg !13
  %address.local507 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local287, !dbg !18
  %360 = load <4 x half>, ptr addrspace(3) %address.local507, align 2, !dbg !18
  %361 = fmul <4 x half> %359, splat (half 0xH3015), !dbg !13
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %354, <4 x half> %360, <4 x float> %215) #9, !dbg !18
  %address.local508 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local301, !dbg !18
  %363 = load <4 x half>, ptr addrspace(3) %address.local508, align 2, !dbg !18
  %address.local509 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local315, !dbg !18
  %364 = load <4 x half>, ptr addrspace(3) %address.local509, align 2, !dbg !18
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %363, <4 x float> %362) #9, !dbg !18
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %354, <4 x half> %364, <4 x float> %218) #9, !dbg !18
  %address.local510 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local329, !dbg !18
  %367 = load <4 x half>, ptr addrspace(3) %address.local510, align 2, !dbg !18
  %address.local511 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local343, !dbg !18
  %368 = load <4 x half>, ptr addrspace(3) %address.local511, align 2, !dbg !18
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %367, <4 x float> %366) #9, !dbg !18
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %354, <4 x half> %368, <4 x float> %223) #9, !dbg !18
  %address.local512 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local357, !dbg !18
  %371 = load <4 x half>, ptr addrspace(3) %address.local512, align 2, !dbg !18
  %address.local513 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local371, !dbg !18
  %372 = load <4 x half>, ptr addrspace(3) %address.local513, align 2, !dbg !18
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %371, <4 x float> %370) #9, !dbg !18
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %354, <4 x half> %372, <4 x float> %226) #9, !dbg !18
  %address.local514 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local385, !dbg !18
  %375 = load <4 x half>, ptr addrspace(3) %address.local514, align 2, !dbg !18
  %address.local515 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local399, !dbg !18
  %376 = load <4 x half>, ptr addrspace(3) %address.local515, align 2, !dbg !18
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %375, <4 x float> %374) #9, !dbg !18
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %354, <4 x half> %376, <4 x float> %231) #9, !dbg !18
  %address.local516 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local413, !dbg !18
  %379 = load <4 x half>, ptr addrspace(3) %address.local516, align 2, !dbg !18
  %address.local517 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local427, !dbg !18
  %380 = load <4 x half>, ptr addrspace(3) %address.local517, align 2, !dbg !18
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %379, <4 x float> %378) #9, !dbg !18
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %354, <4 x half> %380, <4 x float> %234) #9, !dbg !18
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local441, !dbg !18
  %383 = load <4 x half>, ptr addrspace(3) %address.local518, align 2, !dbg !18
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local455, !dbg !18
  %384 = load <4 x half>, ptr addrspace(3) %address.local519, align 2, !dbg !18
  %385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %383, <4 x float> %382) #9, !dbg !18
  %386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %354, <4 x half> %384, <4 x float> %239) #9, !dbg !18
  %address.local520 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local469, !dbg !18
  %387 = load <4 x half>, ptr addrspace(3) %address.local520, align 2, !dbg !18
  %address.local521 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local483, !dbg !18
  %388 = load <4 x half>, ptr addrspace(3) %address.local521, align 2, !dbg !18
  %389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %387, <4 x float> %386) #9, !dbg !18
  %390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %354, <4 x half> %388, <4 x float> %242) #9, !dbg !18
  %address.local522 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local497, !dbg !18
  %391 = load <4 x half>, ptr addrspace(3) %address.local522, align 2, !dbg !18
  %392 = load <4 x half>, ptr addrspace(3) %address.local507, align 2, !dbg !18
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %357, <4 x half> %391, <4 x float> %390) #9, !dbg !18
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %392, <4 x float> %247) #9, !dbg !18
  %395 = load <4 x half>, ptr addrspace(3) %address.local508, align 2, !dbg !18
  %396 = load <4 x half>, ptr addrspace(3) %address.local509, align 2, !dbg !18
  %397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %395, <4 x float> %394) #9, !dbg !18
  %398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %396, <4 x float> %250) #9, !dbg !18
  %399 = load <4 x half>, ptr addrspace(3) %address.local510, align 2, !dbg !18
  %400 = load <4 x half>, ptr addrspace(3) %address.local511, align 2, !dbg !18
  %401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %399, <4 x float> %398) #9, !dbg !18
  %402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %400, <4 x float> %255) #9, !dbg !18
  %403 = load <4 x half>, ptr addrspace(3) %address.local512, align 2, !dbg !18
  %404 = load <4 x half>, ptr addrspace(3) %address.local513, align 2, !dbg !18
  %405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %403, <4 x float> %402) #9, !dbg !18
  %406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %404, <4 x float> %258) #9, !dbg !18
  %407 = load <4 x half>, ptr addrspace(3) %address.local514, align 2, !dbg !18
  %408 = load <4 x half>, ptr addrspace(3) %address.local515, align 2, !dbg !18
  %409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %407, <4 x float> %406) #9, !dbg !18
  %410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %408, <4 x float> %263) #9, !dbg !18
  %411 = load <4 x half>, ptr addrspace(3) %address.local516, align 2, !dbg !18
  %412 = load <4 x half>, ptr addrspace(3) %address.local517, align 2, !dbg !18
  %413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %411, <4 x float> %410) #9, !dbg !18
  %414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %412, <4 x float> %266) #9, !dbg !18
  %415 = load <4 x half>, ptr addrspace(3) %address.local518, align 2, !dbg !18
  %416 = load <4 x half>, ptr addrspace(3) %address.local519, align 2, !dbg !18
  %417 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %415, <4 x float> %414) #9, !dbg !18
  %418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %416, <4 x float> %271) #9, !dbg !18
  %419 = load <4 x half>, ptr addrspace(3) %address.local520, align 2, !dbg !18
  %420 = load <4 x half>, ptr addrspace(3) %address.local521, align 2, !dbg !18
  %421 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %419, <4 x float> %418) #9, !dbg !18
  %422 = load <4 x half>, ptr addrspace(3) %address.local522, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %420, <4 x float> %274) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %361, <4 x half> %422, <4 x float> %423) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %425 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local207), !dbg !12
  %426 = zext i32 %425 to i64, !dbg !12
  %427 = inttoptr i64 %426 to ptr addrspace(3), !dbg !12
  %address.local523 = getelementptr [2 x i8], ptr addrspace(1) %44, i64 %address.local183, !dbg !12
  %428 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %429 = ptrtoint ptr addrspace(1) %address.local523 to i64, !dbg !12
  %430 = sub i64 %429, %428, !dbg !12
  %dma.buffer.offset1118 = trunc i64 %430 to i32, !dbg !12
  %431 = trunc i64 %428 to i32, !dbg !12
  %432 = insertelement <4 x i32> poison, i32 %431, i64 0, !dbg !12
  %433 = lshr i64 %428, 32, !dbg !12
  %434 = trunc i64 %433 to i32, !dbg !12
  %435 = insertelement <4 x i32> %432, i32 %434, i64 1, !dbg !12
  %436 = insertelement <4 x i32> %435, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1119 = insertelement <4 x i32> %436, i32 131072, i64 3, !dbg !12
  %437 = ptrtoint ptr addrspace(3) %427 to i32, !dbg !12
  %dma.buffer.lds1120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %437), !dbg !12
  %438 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1120, i32 %dma.buffer.offset1118, <4 x i32> %dma.buffer.rsrc1119) #10, !dbg !12
  %439 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local211), !dbg !12
  %440 = zext i32 %439 to i64, !dbg !12
  %441 = inttoptr i64 %440 to ptr addrspace(3), !dbg !12
  %address.local524 = getelementptr [2 x i8], ptr addrspace(1) %44, i64 %address.local195, !dbg !12
  %442 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %443 = ptrtoint ptr addrspace(1) %address.local524 to i64, !dbg !12
  %444 = sub i64 %443, %442, !dbg !12
  %dma.buffer.offset1115 = trunc i64 %444 to i32, !dbg !12
  %445 = trunc i64 %442 to i32, !dbg !12
  %446 = insertelement <4 x i32> poison, i32 %445, i64 0, !dbg !12
  %447 = lshr i64 %442, 32, !dbg !12
  %448 = trunc i64 %447 to i32, !dbg !12
  %449 = insertelement <4 x i32> %446, i32 %448, i64 1, !dbg !12
  %450 = insertelement <4 x i32> %449, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1116 = insertelement <4 x i32> %450, i32 131072, i64 3, !dbg !12
  %451 = ptrtoint ptr addrspace(3) %441 to i32, !dbg !12
  %dma.buffer.lds1117 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %451), !dbg !12
  %452 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1117, i32 %dma.buffer.offset1115, <4 x i32> %dma.buffer.rsrc1116) #10, !dbg !12
  %453 = add i64 %151, 3072, !dbg !17
  %454 = icmp slt i64 %453, 0, !dbg !17
  %455 = sub i64 -3073, %151, !dbg !17
  %456 = select i1 %454, i64 %455, i64 %453, !dbg !17
  %457 = sdiv i64 %456, 4096, !dbg !17
  %.lobit6 = ashr i64 %453, 63, !dbg !17
  %458 = xor i64 %457, %.lobit6, !dbg !17
  %459 = add nsw i64 %158, 96, !dbg !17
  %460 = icmp slt i64 %158, -96, !dbg !17
  %461 = sub nsw i64 -97, %158, !dbg !17
  %462 = select i1 %460, i64 %461, i64 %459, !dbg !17
  %463 = sdiv i64 %462, 128, !dbg !17
  %464 = sext i1 %460 to i64, !dbg !17
  %465 = xor i64 %463, %464, !dbg !17
  %466 = shl nsw i64 %465, 7, !dbg !17
  %467 = sub i64 %40, %466, !dbg !17
  %468 = shl i64 %458, 19, !dbg !17
  %469 = and i64 %468, 16252928, !dbg !17
  %470 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %469, !dbg !17
  %471 = getelementptr [2 x i8], ptr addrspace(1) %470, i64 %84, !dbg !17
  %472 = getelementptr [2 x i8], ptr addrspace(1) %471, i64 %467, !dbg !17
  %473 = getelementptr i8, ptr addrspace(1) %472, i64 192, !dbg !17
  %474 = getelementptr [2 x i8], ptr addrspace(1) %473, i64 %90, !dbg !17
  %475 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local215), !dbg !17
  %476 = zext i32 %475 to i64, !dbg !17
  %477 = inttoptr i64 %476 to ptr addrspace(3), !dbg !17
  %address.local525 = getelementptr [2 x i8], ptr addrspace(1) %474, i64 %address.local183, !dbg !17
  %478 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %479 = ptrtoint ptr addrspace(1) %address.local525 to i64, !dbg !17
  %480 = sub i64 %479, %478, !dbg !17
  %dma.buffer.offset1112 = trunc i64 %480 to i32, !dbg !17
  %481 = trunc i64 %478 to i32, !dbg !17
  %482 = insertelement <4 x i32> poison, i32 %481, i64 0, !dbg !17
  %483 = lshr i64 %478, 32, !dbg !17
  %484 = trunc i64 %483 to i32, !dbg !17
  %485 = insertelement <4 x i32> %482, i32 %484, i64 1, !dbg !17
  %486 = insertelement <4 x i32> %485, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1113 = insertelement <4 x i32> %486, i32 131072, i64 3, !dbg !17
  %487 = ptrtoint ptr addrspace(3) %477 to i32, !dbg !17
  %dma.buffer.lds1114 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %487), !dbg !17
  %488 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1114, i32 %dma.buffer.offset1112, <4 x i32> %dma.buffer.rsrc1113) #10, !dbg !17
  %489 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local219), !dbg !17
  %490 = zext i32 %489 to i64, !dbg !17
  %491 = inttoptr i64 %490 to ptr addrspace(3), !dbg !17
  %address.local526 = getelementptr [2 x i8], ptr addrspace(1) %474, i64 %address.local195, !dbg !17
  %492 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %493 = ptrtoint ptr addrspace(1) %address.local526 to i64, !dbg !17
  %494 = sub i64 %493, %492, !dbg !17
  %dma.buffer.offset1109 = trunc i64 %494 to i32, !dbg !17
  %495 = trunc i64 %492 to i32, !dbg !17
  %496 = insertelement <4 x i32> poison, i32 %495, i64 0, !dbg !17
  %497 = lshr i64 %492, 32, !dbg !17
  %498 = trunc i64 %497 to i32, !dbg !17
  %499 = insertelement <4 x i32> %496, i32 %498, i64 1, !dbg !17
  %500 = insertelement <4 x i32> %499, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1110 = insertelement <4 x i32> %500, i32 131072, i64 3, !dbg !17
  %501 = ptrtoint ptr addrspace(3) %491 to i32, !dbg !17
  %dma.buffer.lds1111 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %501), !dbg !17
  %502 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1111, i32 %dma.buffer.offset1109, <4 x i32> %dma.buffer.rsrc1110) #10, !dbg !17
  %503 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !13
  %504 = fmul <4 x half> %503, splat (half 0xH3015), !dbg !13
  %505 = load <4 x half>, ptr addrspace(3) %address.local247, align 2, !dbg !13
  %506 = load <4 x half>, ptr addrspace(3) %address.local261, align 2, !dbg !13
  %507 = fmul <4 x half> %505, splat (half 0xH3015), !dbg !13
  %508 = fmul <4 x half> %506, splat (half 0xH3015), !dbg !13
  %509 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !13
  %510 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !18
  %511 = fmul <4 x half> %509, splat (half 0xH3015), !dbg !13
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %504, <4 x half> %510, <4 x float> %365) #9, !dbg !18
  %513 = load <4 x half>, ptr addrspace(3) %address.local302, align 2, !dbg !18
  %514 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !18
  %515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %507, <4 x half> %513, <4 x float> %512) #9, !dbg !18
  %516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %504, <4 x half> %514, <4 x float> %369) #9, !dbg !18
  %517 = load <4 x half>, ptr addrspace(3) %address.local330, align 2, !dbg !18
  %518 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !18
  %519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %507, <4 x half> %517, <4 x float> %516) #9, !dbg !18
  %520 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %504, <4 x half> %518, <4 x float> %373) #9, !dbg !18
  %521 = load <4 x half>, ptr addrspace(3) %address.local358, align 2, !dbg !18
  %522 = load <4 x half>, ptr addrspace(3) %address.local372, align 2, !dbg !18
  %523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %507, <4 x half> %521, <4 x float> %520) #9, !dbg !18
  %524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %504, <4 x half> %522, <4 x float> %377) #9, !dbg !18
  %525 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !18
  %526 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !18
  %527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %507, <4 x half> %525, <4 x float> %524) #9, !dbg !18
  %528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %504, <4 x half> %526, <4 x float> %381) #9, !dbg !18
  %529 = load <4 x half>, ptr addrspace(3) %address.local414, align 2, !dbg !18
  %530 = load <4 x half>, ptr addrspace(3) %address.local428, align 2, !dbg !18
  %531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %507, <4 x half> %529, <4 x float> %528) #9, !dbg !18
  %532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %504, <4 x half> %530, <4 x float> %385) #9, !dbg !18
  %533 = load <4 x half>, ptr addrspace(3) %address.local442, align 2, !dbg !18
  %534 = load <4 x half>, ptr addrspace(3) %address.local456, align 2, !dbg !18
  %535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %507, <4 x half> %533, <4 x float> %532) #9, !dbg !18
  %536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %504, <4 x half> %534, <4 x float> %389) #9, !dbg !18
  %537 = load <4 x half>, ptr addrspace(3) %address.local470, align 2, !dbg !18
  %538 = load <4 x half>, ptr addrspace(3) %address.local484, align 2, !dbg !18
  %539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %507, <4 x half> %537, <4 x float> %536) #9, !dbg !18
  %540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %504, <4 x half> %538, <4 x float> %393) #9, !dbg !18
  %541 = load <4 x half>, ptr addrspace(3) %address.local498, align 2, !dbg !18
  %542 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !18
  %543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %507, <4 x half> %541, <4 x float> %540) #9, !dbg !18
  %544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %542, <4 x float> %397) #9, !dbg !18
  %545 = load <4 x half>, ptr addrspace(3) %address.local302, align 2, !dbg !18
  %546 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !18
  %547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %511, <4 x half> %545, <4 x float> %544) #9, !dbg !18
  %548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %546, <4 x float> %401) #9, !dbg !18
  %549 = load <4 x half>, ptr addrspace(3) %address.local330, align 2, !dbg !18
  %550 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !18
  %551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %511, <4 x half> %549, <4 x float> %548) #9, !dbg !18
  %552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %550, <4 x float> %405) #9, !dbg !18
  %553 = load <4 x half>, ptr addrspace(3) %address.local358, align 2, !dbg !18
  %554 = load <4 x half>, ptr addrspace(3) %address.local372, align 2, !dbg !18
  %555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %511, <4 x half> %553, <4 x float> %552) #9, !dbg !18
  %556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %554, <4 x float> %409) #9, !dbg !18
  %557 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !18
  %558 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !18
  %559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %511, <4 x half> %557, <4 x float> %556) #9, !dbg !18
  %560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %558, <4 x float> %413) #9, !dbg !18
  %561 = load <4 x half>, ptr addrspace(3) %address.local414, align 2, !dbg !18
  %562 = load <4 x half>, ptr addrspace(3) %address.local428, align 2, !dbg !18
  %563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %511, <4 x half> %561, <4 x float> %560) #9, !dbg !18
  %564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %562, <4 x float> %417) #9, !dbg !18
  %565 = load <4 x half>, ptr addrspace(3) %address.local442, align 2, !dbg !18
  %566 = load <4 x half>, ptr addrspace(3) %address.local456, align 2, !dbg !18
  %567 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %511, <4 x half> %565, <4 x float> %564) #9, !dbg !18
  %568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %566, <4 x float> %421) #9, !dbg !18
  %569 = load <4 x half>, ptr addrspace(3) %address.local470, align 2, !dbg !18
  %570 = load <4 x half>, ptr addrspace(3) %address.local484, align 2, !dbg !18
  %571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %511, <4 x half> %569, <4 x float> %568) #9, !dbg !18
  %572 = load <4 x half>, ptr addrspace(3) %address.local498, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %570, <4 x float> %424) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %511, <4 x half> %572, <4 x float> %573) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %575 = add nsw i64 %84, %10, !dbg !19
  %576 = shl i64 %575, 19, !dbg !19
  %577 = and i64 %576, 16252928, !dbg !19
  %578 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %577, !dbg !19
  %579 = getelementptr [2 x i8], ptr addrspace(1) %578, i64 %90, !dbg !19
  %580 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local178), !dbg !19
  %581 = zext i32 %580 to i64, !dbg !19
  %582 = inttoptr i64 %581 to ptr addrspace(3), !dbg !19
  %address.local527 = shl nuw nsw i64 %address.local, 5, !dbg !19
  %address.local528 = and i64 %address.local527, 8064, !dbg !19
  %address.local529 = shl nuw nsw i64 %address.local, 3, !dbg !19
  %address.local530 = and i64 %address.local529, 24, !dbg !19
  %address.local531 = or disjoint i64 %address.local528, %address.local530, !dbg !19
  %address.local532 = getelementptr [2 x i8], ptr addrspace(1) %579, i64 %address.local531, !dbg !19
  %583 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %584 = ptrtoint ptr addrspace(1) %address.local532 to i64, !dbg !19
  %585 = sub i64 %584, %583, !dbg !19
  %dma.buffer.offset1106 = trunc i64 %585 to i32, !dbg !19
  %586 = trunc i64 %583 to i32, !dbg !19
  %587 = insertelement <4 x i32> poison, i32 %586, i64 0, !dbg !19
  %588 = lshr i64 %583, 32, !dbg !19
  %589 = trunc i64 %588 to i32, !dbg !19
  %590 = insertelement <4 x i32> %587, i32 %589, i64 1, !dbg !19
  %591 = insertelement <4 x i32> %590, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1107 = insertelement <4 x i32> %591, i32 131072, i64 3, !dbg !19
  %592 = ptrtoint ptr addrspace(3) %582 to i32, !dbg !19
  %dma.buffer.lds1108 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %592), !dbg !19
  %593 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1108, i32 %dma.buffer.offset1106, <4 x i32> %dma.buffer.rsrc1107) #10, !dbg !19
  %594 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local190), !dbg !19
  %595 = zext i32 %594 to i64, !dbg !19
  %596 = inttoptr i64 %595 to ptr addrspace(3), !dbg !19
  %address.local533 = shl nuw nsw i64 %address.local185, 5, !dbg !19
  %address.local534 = and i64 %address.local533, 16256, !dbg !19
  %address.local535 = shl nuw nsw i64 %address.local, 3, !dbg !19
  %address.local536 = and i64 %address.local535, 24, !dbg !19
  %address.local537 = or disjoint i64 %address.local534, %address.local536, !dbg !19
  %address.local538 = getelementptr [2 x i8], ptr addrspace(1) %579, i64 %address.local537, !dbg !19
  %597 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %598 = ptrtoint ptr addrspace(1) %address.local538 to i64, !dbg !19
  %599 = sub i64 %598, %597, !dbg !19
  %dma.buffer.offset1103 = trunc i64 %599 to i32, !dbg !19
  %600 = trunc i64 %597 to i32, !dbg !19
  %601 = insertelement <4 x i32> poison, i32 %600, i64 0, !dbg !19
  %602 = lshr i64 %597, 32, !dbg !19
  %603 = trunc i64 %602 to i32, !dbg !19
  %604 = insertelement <4 x i32> %601, i32 %603, i64 1, !dbg !19
  %605 = insertelement <4 x i32> %604, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1104 = insertelement <4 x i32> %605, i32 131072, i64 3, !dbg !19
  %606 = ptrtoint ptr addrspace(3) %596 to i32, !dbg !19
  %dma.buffer.lds1105 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %606), !dbg !19
  %607 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1105, i32 %dma.buffer.offset1103, <4 x i32> %dma.buffer.rsrc1104) #10, !dbg !19
  %608 = load <4 x half>, ptr addrspace(3) %address.local503, align 2, !dbg !13
  %609 = fmul <4 x half> %608, splat (half 0xH3015), !dbg !13
  %610 = load <4 x half>, ptr addrspace(3) %address.local504, align 2, !dbg !13
  %611 = load <4 x half>, ptr addrspace(3) %address.local505, align 2, !dbg !13
  %612 = fmul <4 x half> %610, splat (half 0xH3015), !dbg !13
  %613 = fmul <4 x half> %611, splat (half 0xH3015), !dbg !13
  %614 = load <4 x half>, ptr addrspace(3) %address.local506, align 2, !dbg !13
  %615 = load <4 x half>, ptr addrspace(3) %address.local507, align 2, !dbg !18
  %616 = fmul <4 x half> %614, splat (half 0xH3015), !dbg !13
  %617 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %609, <4 x half> %615, <4 x float> %515) #9, !dbg !18
  %618 = load <4 x half>, ptr addrspace(3) %address.local508, align 2, !dbg !18
  %619 = load <4 x half>, ptr addrspace(3) %address.local509, align 2, !dbg !18
  %620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %612, <4 x half> %618, <4 x float> %617) #9, !dbg !18
  %621 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %609, <4 x half> %619, <4 x float> %519) #9, !dbg !18
  %622 = load <4 x half>, ptr addrspace(3) %address.local510, align 2, !dbg !18
  %623 = load <4 x half>, ptr addrspace(3) %address.local511, align 2, !dbg !18
  %624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %612, <4 x half> %622, <4 x float> %621) #9, !dbg !18
  %625 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %609, <4 x half> %623, <4 x float> %523) #9, !dbg !18
  %626 = load <4 x half>, ptr addrspace(3) %address.local512, align 2, !dbg !18
  %627 = load <4 x half>, ptr addrspace(3) %address.local513, align 2, !dbg !18
  %628 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %612, <4 x half> %626, <4 x float> %625) #9, !dbg !18
  %629 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %609, <4 x half> %627, <4 x float> %527) #9, !dbg !18
  %630 = load <4 x half>, ptr addrspace(3) %address.local514, align 2, !dbg !18
  %631 = load <4 x half>, ptr addrspace(3) %address.local515, align 2, !dbg !18
  %632 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %612, <4 x half> %630, <4 x float> %629) #9, !dbg !18
  %633 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %609, <4 x half> %631, <4 x float> %531) #9, !dbg !18
  %634 = load <4 x half>, ptr addrspace(3) %address.local516, align 2, !dbg !18
  %635 = load <4 x half>, ptr addrspace(3) %address.local517, align 2, !dbg !18
  %636 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %612, <4 x half> %634, <4 x float> %633) #9, !dbg !18
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %609, <4 x half> %635, <4 x float> %535) #9, !dbg !18
  %638 = load <4 x half>, ptr addrspace(3) %address.local518, align 2, !dbg !18
  %639 = load <4 x half>, ptr addrspace(3) %address.local519, align 2, !dbg !18
  %640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %612, <4 x half> %638, <4 x float> %637) #9, !dbg !18
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %609, <4 x half> %639, <4 x float> %539) #9, !dbg !18
  %642 = load <4 x half>, ptr addrspace(3) %address.local520, align 2, !dbg !18
  %643 = load <4 x half>, ptr addrspace(3) %address.local521, align 2, !dbg !18
  %644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %612, <4 x half> %642, <4 x float> %641) #9, !dbg !18
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %609, <4 x half> %643, <4 x float> %543) #9, !dbg !18
  %646 = load <4 x half>, ptr addrspace(3) %address.local522, align 2, !dbg !18
  %647 = load <4 x half>, ptr addrspace(3) %address.local507, align 2, !dbg !18
  %648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %612, <4 x half> %646, <4 x float> %645) #9, !dbg !18
  %649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %647, <4 x float> %547) #9, !dbg !18
  %650 = load <4 x half>, ptr addrspace(3) %address.local508, align 2, !dbg !18
  %651 = load <4 x half>, ptr addrspace(3) %address.local509, align 2, !dbg !18
  %652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %616, <4 x half> %650, <4 x float> %649) #9, !dbg !18
  %653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %651, <4 x float> %551) #9, !dbg !18
  %654 = load <4 x half>, ptr addrspace(3) %address.local510, align 2, !dbg !18
  %655 = load <4 x half>, ptr addrspace(3) %address.local511, align 2, !dbg !18
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %616, <4 x half> %654, <4 x float> %653) #9, !dbg !18
  %657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %655, <4 x float> %555) #9, !dbg !18
  %658 = load <4 x half>, ptr addrspace(3) %address.local512, align 2, !dbg !18
  %659 = load <4 x half>, ptr addrspace(3) %address.local513, align 2, !dbg !18
  %660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %616, <4 x half> %658, <4 x float> %657) #9, !dbg !18
  %661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %659, <4 x float> %559) #9, !dbg !18
  %662 = load <4 x half>, ptr addrspace(3) %address.local514, align 2, !dbg !18
  %663 = load <4 x half>, ptr addrspace(3) %address.local515, align 2, !dbg !18
  %664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %616, <4 x half> %662, <4 x float> %661) #9, !dbg !18
  %665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %663, <4 x float> %563) #9, !dbg !18
  %666 = load <4 x half>, ptr addrspace(3) %address.local516, align 2, !dbg !18
  %667 = load <4 x half>, ptr addrspace(3) %address.local517, align 2, !dbg !18
  %668 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %616, <4 x half> %666, <4 x float> %665) #9, !dbg !18
  %669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %667, <4 x float> %567) #9, !dbg !18
  %670 = load <4 x half>, ptr addrspace(3) %address.local518, align 2, !dbg !18
  %671 = load <4 x half>, ptr addrspace(3) %address.local519, align 2, !dbg !18
  %672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %616, <4 x half> %670, <4 x float> %669) #9, !dbg !18
  %673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %671, <4 x float> %571) #9, !dbg !18
  %674 = load <4 x half>, ptr addrspace(3) %address.local520, align 2, !dbg !18
  %675 = load <4 x half>, ptr addrspace(3) %address.local521, align 2, !dbg !18
  %676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %616, <4 x half> %674, <4 x float> %673) #9, !dbg !18
  %677 = load <4 x half>, ptr addrspace(3) %address.local522, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %675, <4 x float> %574) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %679 = or disjoint i64 %53, 127, !dbg !14
  %680 = icmp uge i64 %11, %679, !dbg !14
  %681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %616, <4 x half> %677, <4 x float> %678) #9, !dbg !18
  %682 = and i1 %31, %680, !dbg !14
  br i1 %682, label %683, label %684, !dbg !14

683:                                              ; preds = %52
  br label %828, !dbg !14

684:                                              ; preds = %52
  %685 = add nsw i64 %11, %27, !dbg !14
  %686 = shl nuw nsw i64 %28, 4, !dbg !14
  %687 = add nsw i64 %685, %686, !dbg !14
  %688 = and i32 %26, 240, !dbg !14
  %index.compact = zext nneg i32 %688 to i64, !dbg !14
  %689 = sub nsw i64 %687, %index.compact, !dbg !14
  %690 = and i64 %27, 48, !dbg !14
  %691 = or disjoint i64 %53, %690, !dbg !14
  %692 = add nsw i64 %689, 1, !dbg !25
  %693 = lshr i64 %692, 32, !dbg !26
  %compare.high = trunc i64 %693 to i32, !dbg !26
  %694 = lshr i64 %691, 32, !dbg !26
  %compare.high832 = trunc i64 %694 to i32, !dbg !26
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high832, !dbg !26
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high832, !dbg !26
  %compare.low.base = trunc i64 %692 to i32, !dbg !26
  %compare.low.base833 = trunc i64 %691 to i32, !dbg !26
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base833, !dbg !26
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !26
  %695 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 1) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !26
  %compare.low834 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !26
  %compare.result835 = select i1 %compare.high.equal, i1 %compare.low834, i1 %compare.high.order, !dbg !26
  %696 = select i1 %compare.result835, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base836 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 2) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local837 = or i32 %compare.local.base836, 2, !dbg !26
  %compare.low838 = icmp ugt i32 %compare.low.base, %compare.local837, !dbg !26
  %compare.result839 = select i1 %compare.high.equal, i1 %compare.low838, i1 %compare.high.order, !dbg !26
  %697 = select i1 %compare.result839, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base840 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 3) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local841 = or i32 %compare.local.base840, 3, !dbg !26
  %compare.low842 = icmp ugt i32 %compare.low.base, %compare.local841, !dbg !26
  %compare.result843 = select i1 %compare.high.equal, i1 %compare.low842, i1 %compare.high.order, !dbg !26
  %698 = select i1 %compare.result843, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base844 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 4) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local845 = or i32 %compare.local.base844, 4, !dbg !26
  %compare.low846 = icmp ugt i32 %compare.low.base, %compare.local845, !dbg !26
  %compare.result847 = select i1 %compare.high.equal, i1 %compare.low846, i1 %compare.high.order, !dbg !26
  %699 = select i1 %compare.result847, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base848 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 5) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local849 = or i32 %compare.local.base848, 5, !dbg !26
  %compare.low850 = icmp ugt i32 %compare.low.base, %compare.local849, !dbg !26
  %compare.result851 = select i1 %compare.high.equal, i1 %compare.low850, i1 %compare.high.order, !dbg !26
  %700 = select i1 %compare.result851, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base852 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 6) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local853 = or i32 %compare.local.base852, 6, !dbg !26
  %compare.low854 = icmp ugt i32 %compare.low.base, %compare.local853, !dbg !26
  %compare.result855 = select i1 %compare.high.equal, i1 %compare.low854, i1 %compare.high.order, !dbg !26
  %701 = select i1 %compare.result855, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base856 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 7) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local857 = or i32 %compare.local.base856, 7, !dbg !26
  %compare.low858 = icmp ugt i32 %compare.low.base, %compare.local857, !dbg !26
  %compare.result859 = select i1 %compare.high.equal, i1 %compare.low858, i1 %compare.high.order, !dbg !26
  %702 = select i1 %compare.result859, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base860 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 8) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local861 = or i32 %compare.local.base860, 8, !dbg !26
  %compare.low862 = icmp ugt i32 %compare.low.base, %compare.local861, !dbg !26
  %compare.result863 = select i1 %compare.high.equal, i1 %compare.low862, i1 %compare.high.order, !dbg !26
  %703 = select i1 %compare.result863, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base864 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 9) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local865 = or i32 %compare.local.base864, 9, !dbg !26
  %compare.low866 = icmp ugt i32 %compare.low.base, %compare.local865, !dbg !26
  %compare.result867 = select i1 %compare.high.equal, i1 %compare.low866, i1 %compare.high.order, !dbg !26
  %704 = select i1 %compare.result867, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base868 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 10) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local869 = or i32 %compare.local.base868, 10, !dbg !26
  %compare.low870 = icmp ugt i32 %compare.low.base, %compare.local869, !dbg !26
  %compare.result871 = select i1 %compare.high.equal, i1 %compare.low870, i1 %compare.high.order, !dbg !26
  %705 = select i1 %compare.result871, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base872 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 11) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local873 = or i32 %compare.local.base872, 11, !dbg !26
  %compare.low874 = icmp ugt i32 %compare.low.base, %compare.local873, !dbg !26
  %compare.result875 = select i1 %compare.high.equal, i1 %compare.low874, i1 %compare.high.order, !dbg !26
  %706 = select i1 %compare.result875, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base876 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 12) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local877 = or i32 %compare.local.base876, 12, !dbg !26
  %compare.low878 = icmp ugt i32 %compare.low.base, %compare.local877, !dbg !26
  %compare.result879 = select i1 %compare.high.equal, i1 %compare.low878, i1 %compare.high.order, !dbg !26
  %707 = select i1 %compare.result879, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base880 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 13) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local881 = or i32 %compare.local.base880, 13, !dbg !26
  %compare.low882 = icmp ugt i32 %compare.low.base, %compare.local881, !dbg !26
  %compare.result883 = select i1 %compare.high.equal, i1 %compare.low882, i1 %compare.high.order, !dbg !26
  %708 = select i1 %compare.result883, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base884 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 14) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local885 = or i32 %compare.local.base884, 14, !dbg !26
  %compare.low886 = icmp ugt i32 %compare.low.base, %compare.local885, !dbg !26
  %compare.result887 = select i1 %compare.high.equal, i1 %compare.low886, i1 %compare.high.order, !dbg !26
  %709 = select i1 %compare.result887, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base888 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 15) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local889 = or i32 %compare.local.base888, 15, !dbg !26
  %compare.low890 = icmp ugt i32 %compare.low.base, %compare.local889, !dbg !26
  %compare.result891 = select i1 %compare.high.equal, i1 %compare.low890, i1 %compare.high.order, !dbg !26
  %710 = select i1 %compare.result891, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base892 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 16) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local893 = or i32 %compare.local.base892, 64, !dbg !26
  %compare.low894 = icmp ugt i32 %compare.low.base, %compare.local893, !dbg !26
  %compare.result895 = select i1 %compare.high.equal, i1 %compare.low894, i1 %compare.high.order, !dbg !26
  %711 = select i1 %compare.result895, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base896 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 17) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local897 = or i32 %compare.local.base896, 65, !dbg !26
  %compare.low898 = icmp ugt i32 %compare.low.base, %compare.local897, !dbg !26
  %compare.result899 = select i1 %compare.high.equal, i1 %compare.low898, i1 %compare.high.order, !dbg !26
  %712 = select i1 %compare.result899, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base900 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 18) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local901 = or i32 %compare.local.base900, 66, !dbg !26
  %compare.low902 = icmp ugt i32 %compare.low.base, %compare.local901, !dbg !26
  %compare.result903 = select i1 %compare.high.equal, i1 %compare.low902, i1 %compare.high.order, !dbg !26
  %713 = select i1 %compare.result903, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base904 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 19) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local905 = or i32 %compare.local.base904, 67, !dbg !26
  %compare.low906 = icmp ugt i32 %compare.low.base, %compare.local905, !dbg !26
  %compare.result907 = select i1 %compare.high.equal, i1 %compare.low906, i1 %compare.high.order, !dbg !26
  %714 = select i1 %compare.result907, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base908 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 20) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local909 = or i32 %compare.local.base908, 68, !dbg !26
  %compare.low910 = icmp ugt i32 %compare.low.base, %compare.local909, !dbg !26
  %compare.result911 = select i1 %compare.high.equal, i1 %compare.low910, i1 %compare.high.order, !dbg !26
  %715 = select i1 %compare.result911, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base912 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 21) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local913 = or i32 %compare.local.base912, 69, !dbg !26
  %compare.low914 = icmp ugt i32 %compare.low.base, %compare.local913, !dbg !26
  %compare.result915 = select i1 %compare.high.equal, i1 %compare.low914, i1 %compare.high.order, !dbg !26
  %716 = select i1 %compare.result915, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base916 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 22) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local917 = or i32 %compare.local.base916, 70, !dbg !26
  %compare.low918 = icmp ugt i32 %compare.low.base, %compare.local917, !dbg !26
  %compare.result919 = select i1 %compare.high.equal, i1 %compare.low918, i1 %compare.high.order, !dbg !26
  %717 = select i1 %compare.result919, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base920 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 23) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local921 = or i32 %compare.local.base920, 71, !dbg !26
  %compare.low922 = icmp ugt i32 %compare.low.base, %compare.local921, !dbg !26
  %compare.result923 = select i1 %compare.high.equal, i1 %compare.low922, i1 %compare.high.order, !dbg !26
  %718 = select i1 %compare.result923, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base924 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 24) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local925 = or i32 %compare.local.base924, 72, !dbg !26
  %compare.low926 = icmp ugt i32 %compare.low.base, %compare.local925, !dbg !26
  %compare.result927 = select i1 %compare.high.equal, i1 %compare.low926, i1 %compare.high.order, !dbg !26
  %719 = select i1 %compare.result927, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base928 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 25) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local929 = or i32 %compare.local.base928, 73, !dbg !26
  %compare.low930 = icmp ugt i32 %compare.low.base, %compare.local929, !dbg !26
  %compare.result931 = select i1 %compare.high.equal, i1 %compare.low930, i1 %compare.high.order, !dbg !26
  %720 = select i1 %compare.result931, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base932 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 26) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local933 = or i32 %compare.local.base932, 74, !dbg !26
  %compare.low934 = icmp ugt i32 %compare.low.base, %compare.local933, !dbg !26
  %compare.result935 = select i1 %compare.high.equal, i1 %compare.low934, i1 %compare.high.order, !dbg !26
  %721 = select i1 %compare.result935, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base936 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 27) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local937 = or i32 %compare.local.base936, 75, !dbg !26
  %compare.low938 = icmp ugt i32 %compare.low.base, %compare.local937, !dbg !26
  %compare.result939 = select i1 %compare.high.equal, i1 %compare.low938, i1 %compare.high.order, !dbg !26
  %722 = select i1 %compare.result939, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base940 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 28) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local941 = or i32 %compare.local.base940, 76, !dbg !26
  %compare.low942 = icmp ugt i32 %compare.low.base, %compare.local941, !dbg !26
  %compare.result943 = select i1 %compare.high.equal, i1 %compare.low942, i1 %compare.high.order, !dbg !26
  %723 = select i1 %compare.result943, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base944 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 29) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local945 = or i32 %compare.local.base944, 77, !dbg !26
  %compare.low946 = icmp ugt i32 %compare.low.base, %compare.local945, !dbg !26
  %compare.result947 = select i1 %compare.high.equal, i1 %compare.low946, i1 %compare.high.order, !dbg !26
  %724 = select i1 %compare.result947, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base948 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 30) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local949 = or i32 %compare.local.base948, 78, !dbg !26
  %compare.low950 = icmp ugt i32 %compare.low.base, %compare.local949, !dbg !26
  %compare.result951 = select i1 %compare.high.equal, i1 %compare.low950, i1 %compare.high.order, !dbg !26
  %725 = select i1 %compare.result951, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base952 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 31) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local953 = or i32 %compare.local.base952, 79, !dbg !26
  %compare.low954 = icmp ugt i32 %compare.low.base, %compare.local953, !dbg !26
  %compare.result955 = select i1 %compare.high.equal, i1 %compare.low954, i1 %compare.high.order, !dbg !26
  %726 = select i1 %compare.result955, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %727 = add nsw i64 %689, 65, !dbg !25
  %728 = lshr i64 %727, 32, !dbg !26
  %compare.high956 = trunc i64 %728 to i32, !dbg !26
  %729 = lshr i64 %691, 32, !dbg !26
  %compare.high957 = trunc i64 %729 to i32, !dbg !26
  %compare.high.equal958 = icmp eq i32 %compare.high956, %compare.high957, !dbg !26
  %compare.high.order959 = icmp ugt i32 %compare.high956, %compare.high957, !dbg !26
  %compare.low.base960 = trunc i64 %727 to i32, !dbg !26
  %compare.low.base961 = trunc i64 %691 to i32, !dbg !26
  %compare.low962 = icmp ugt i32 %compare.low.base960, %compare.low.base961, !dbg !26
  %compare.result963 = select i1 %compare.high.equal958, i1 %compare.low962, i1 %compare.high.order959, !dbg !26
  %730 = select i1 %compare.result963, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base964 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 32) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local965 = or i32 %compare.local.base964, 1, !dbg !26
  %compare.low966 = icmp ugt i32 %compare.low.base960, %compare.local965, !dbg !26
  %compare.result967 = select i1 %compare.high.equal958, i1 %compare.low966, i1 %compare.high.order959, !dbg !26
  %731 = select i1 %compare.result967, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base968 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 33) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local969 = or i32 %compare.local.base968, 2, !dbg !26
  %compare.low970 = icmp ugt i32 %compare.low.base960, %compare.local969, !dbg !26
  %compare.result971 = select i1 %compare.high.equal958, i1 %compare.low970, i1 %compare.high.order959, !dbg !26
  %732 = select i1 %compare.result971, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base972 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 34) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local973 = or i32 %compare.local.base972, 3, !dbg !26
  %compare.low974 = icmp ugt i32 %compare.low.base960, %compare.local973, !dbg !26
  %compare.result975 = select i1 %compare.high.equal958, i1 %compare.low974, i1 %compare.high.order959, !dbg !26
  %733 = select i1 %compare.result975, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base976 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 35) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local977 = or i32 %compare.local.base976, 4, !dbg !26
  %compare.low978 = icmp ugt i32 %compare.low.base960, %compare.local977, !dbg !26
  %compare.result979 = select i1 %compare.high.equal958, i1 %compare.low978, i1 %compare.high.order959, !dbg !26
  %734 = select i1 %compare.result979, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base980 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 36) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local981 = or i32 %compare.local.base980, 5, !dbg !26
  %compare.low982 = icmp ugt i32 %compare.low.base960, %compare.local981, !dbg !26
  %compare.result983 = select i1 %compare.high.equal958, i1 %compare.low982, i1 %compare.high.order959, !dbg !26
  %735 = select i1 %compare.result983, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base984 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 37) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local985 = or i32 %compare.local.base984, 6, !dbg !26
  %compare.low986 = icmp ugt i32 %compare.low.base960, %compare.local985, !dbg !26
  %compare.result987 = select i1 %compare.high.equal958, i1 %compare.low986, i1 %compare.high.order959, !dbg !26
  %736 = select i1 %compare.result987, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base988 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 38) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local989 = or i32 %compare.local.base988, 7, !dbg !26
  %compare.low990 = icmp ugt i32 %compare.low.base960, %compare.local989, !dbg !26
  %compare.result991 = select i1 %compare.high.equal958, i1 %compare.low990, i1 %compare.high.order959, !dbg !26
  %737 = select i1 %compare.result991, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base992 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 39) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local993 = or i32 %compare.local.base992, 8, !dbg !26
  %compare.low994 = icmp ugt i32 %compare.low.base960, %compare.local993, !dbg !26
  %compare.result995 = select i1 %compare.high.equal958, i1 %compare.low994, i1 %compare.high.order959, !dbg !26
  %738 = select i1 %compare.result995, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base996 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 40) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local997 = or i32 %compare.local.base996, 9, !dbg !26
  %compare.low998 = icmp ugt i32 %compare.low.base960, %compare.local997, !dbg !26
  %compare.result999 = select i1 %compare.high.equal958, i1 %compare.low998, i1 %compare.high.order959, !dbg !26
  %739 = select i1 %compare.result999, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1000 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 41) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1001 = or i32 %compare.local.base1000, 10, !dbg !26
  %compare.low1002 = icmp ugt i32 %compare.low.base960, %compare.local1001, !dbg !26
  %compare.result1003 = select i1 %compare.high.equal958, i1 %compare.low1002, i1 %compare.high.order959, !dbg !26
  %740 = select i1 %compare.result1003, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1004 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 42) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1005 = or i32 %compare.local.base1004, 11, !dbg !26
  %compare.low1006 = icmp ugt i32 %compare.low.base960, %compare.local1005, !dbg !26
  %compare.result1007 = select i1 %compare.high.equal958, i1 %compare.low1006, i1 %compare.high.order959, !dbg !26
  %741 = select i1 %compare.result1007, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1008 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 43) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1009 = or i32 %compare.local.base1008, 12, !dbg !26
  %compare.low1010 = icmp ugt i32 %compare.low.base960, %compare.local1009, !dbg !26
  %compare.result1011 = select i1 %compare.high.equal958, i1 %compare.low1010, i1 %compare.high.order959, !dbg !26
  %742 = select i1 %compare.result1011, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1012 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 44) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1013 = or i32 %compare.local.base1012, 13, !dbg !26
  %compare.low1014 = icmp ugt i32 %compare.low.base960, %compare.local1013, !dbg !26
  %compare.result1015 = select i1 %compare.high.equal958, i1 %compare.low1014, i1 %compare.high.order959, !dbg !26
  %743 = select i1 %compare.result1015, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1016 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 45) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1017 = or i32 %compare.local.base1016, 14, !dbg !26
  %compare.low1018 = icmp ugt i32 %compare.low.base960, %compare.local1017, !dbg !26
  %compare.result1019 = select i1 %compare.high.equal958, i1 %compare.low1018, i1 %compare.high.order959, !dbg !26
  %744 = select i1 %compare.result1019, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1020 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 46) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1021 = or i32 %compare.local.base1020, 15, !dbg !26
  %compare.low1022 = icmp ugt i32 %compare.low.base960, %compare.local1021, !dbg !26
  %compare.result1023 = select i1 %compare.high.equal958, i1 %compare.low1022, i1 %compare.high.order959, !dbg !26
  %745 = select i1 %compare.result1023, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1024 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 47) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1025 = or i32 %compare.local.base1024, 64, !dbg !26
  %compare.low1026 = icmp ugt i32 %compare.low.base960, %compare.local1025, !dbg !26
  %compare.result1027 = select i1 %compare.high.equal958, i1 %compare.low1026, i1 %compare.high.order959, !dbg !26
  %746 = select i1 %compare.result1027, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1028 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 48) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1029 = or i32 %compare.local.base1028, 65, !dbg !26
  %compare.low1030 = icmp ugt i32 %compare.low.base960, %compare.local1029, !dbg !26
  %compare.result1031 = select i1 %compare.high.equal958, i1 %compare.low1030, i1 %compare.high.order959, !dbg !26
  %747 = select i1 %compare.result1031, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1032 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 49) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1033 = or i32 %compare.local.base1032, 66, !dbg !26
  %compare.low1034 = icmp ugt i32 %compare.low.base960, %compare.local1033, !dbg !26
  %compare.result1035 = select i1 %compare.high.equal958, i1 %compare.low1034, i1 %compare.high.order959, !dbg !26
  %748 = select i1 %compare.result1035, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1036 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 50) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1037 = or i32 %compare.local.base1036, 67, !dbg !26
  %compare.low1038 = icmp ugt i32 %compare.low.base960, %compare.local1037, !dbg !26
  %compare.result1039 = select i1 %compare.high.equal958, i1 %compare.low1038, i1 %compare.high.order959, !dbg !26
  %749 = select i1 %compare.result1039, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1040 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 51) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1041 = or i32 %compare.local.base1040, 68, !dbg !26
  %compare.low1042 = icmp ugt i32 %compare.low.base960, %compare.local1041, !dbg !26
  %compare.result1043 = select i1 %compare.high.equal958, i1 %compare.low1042, i1 %compare.high.order959, !dbg !26
  %750 = select i1 %compare.result1043, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1044 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 52) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1045 = or i32 %compare.local.base1044, 69, !dbg !26
  %compare.low1046 = icmp ugt i32 %compare.low.base960, %compare.local1045, !dbg !26
  %compare.result1047 = select i1 %compare.high.equal958, i1 %compare.low1046, i1 %compare.high.order959, !dbg !26
  %751 = select i1 %compare.result1047, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1048 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 53) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1049 = or i32 %compare.local.base1048, 70, !dbg !26
  %compare.low1050 = icmp ugt i32 %compare.low.base960, %compare.local1049, !dbg !26
  %compare.result1051 = select i1 %compare.high.equal958, i1 %compare.low1050, i1 %compare.high.order959, !dbg !26
  %752 = select i1 %compare.result1051, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1052 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 54) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1053 = or i32 %compare.local.base1052, 71, !dbg !26
  %compare.low1054 = icmp ugt i32 %compare.low.base960, %compare.local1053, !dbg !26
  %compare.result1055 = select i1 %compare.high.equal958, i1 %compare.low1054, i1 %compare.high.order959, !dbg !26
  %753 = select i1 %compare.result1055, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1056 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 55) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1057 = or i32 %compare.local.base1056, 72, !dbg !26
  %compare.low1058 = icmp ugt i32 %compare.low.base960, %compare.local1057, !dbg !26
  %compare.result1059 = select i1 %compare.high.equal958, i1 %compare.low1058, i1 %compare.high.order959, !dbg !26
  %754 = select i1 %compare.result1059, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1060 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 56) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1061 = or i32 %compare.local.base1060, 73, !dbg !26
  %compare.low1062 = icmp ugt i32 %compare.low.base960, %compare.local1061, !dbg !26
  %compare.result1063 = select i1 %compare.high.equal958, i1 %compare.low1062, i1 %compare.high.order959, !dbg !26
  %755 = select i1 %compare.result1063, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1064 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 57) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1065 = or i32 %compare.local.base1064, 74, !dbg !26
  %compare.low1066 = icmp ugt i32 %compare.low.base960, %compare.local1065, !dbg !26
  %compare.result1067 = select i1 %compare.high.equal958, i1 %compare.low1066, i1 %compare.high.order959, !dbg !26
  %756 = select i1 %compare.result1067, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1068 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 58) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1069 = or i32 %compare.local.base1068, 75, !dbg !26
  %compare.low1070 = icmp ugt i32 %compare.low.base960, %compare.local1069, !dbg !26
  %compare.result1071 = select i1 %compare.high.equal958, i1 %compare.low1070, i1 %compare.high.order959, !dbg !26
  %757 = select i1 %compare.result1071, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1072 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 59) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1073 = or i32 %compare.local.base1072, 76, !dbg !26
  %compare.low1074 = icmp ugt i32 %compare.low.base960, %compare.local1073, !dbg !26
  %compare.result1075 = select i1 %compare.high.equal958, i1 %compare.low1074, i1 %compare.high.order959, !dbg !26
  %758 = select i1 %compare.result1075, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1076 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 60) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1077 = or i32 %compare.local.base1076, 77, !dbg !26
  %compare.low1078 = icmp ugt i32 %compare.low.base960, %compare.local1077, !dbg !26
  %compare.result1079 = select i1 %compare.high.equal958, i1 %compare.low1078, i1 %compare.high.order959, !dbg !26
  %759 = select i1 %compare.result1079, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1080 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 61) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1081 = or i32 %compare.local.base1080, 78, !dbg !26
  %compare.low1082 = icmp ugt i32 %compare.low.base960, %compare.local1081, !dbg !26
  %compare.result1083 = select i1 %compare.high.equal958, i1 %compare.low1082, i1 %compare.high.order959, !dbg !26
  %760 = select i1 %compare.result1083, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %compare.local.base1084 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 62) #9, !dbg !26, !frisk.register_lifetimes !1
  %compare.local1085 = or i32 %compare.local.base1084, 79, !dbg !26
  %compare.low1086 = icmp ugt i32 %compare.low.base960, %compare.local1085, !dbg !26
  %compare.result1087 = select i1 %compare.high.equal958, i1 %compare.low1086, i1 %compare.high.order959, !dbg !26
  %761 = select i1 %compare.result1087, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !27
  %762 = insertelement <32 x float> poison, float %695, i64 0, !dbg !14
  %763 = insertelement <32 x float> %762, float %696, i64 1, !dbg !14
  %764 = insertelement <32 x float> %763, float %697, i64 2, !dbg !14
  %765 = insertelement <32 x float> %764, float %698, i64 3, !dbg !14
  %766 = insertelement <32 x float> %765, float %699, i64 4, !dbg !14
  %767 = insertelement <32 x float> %766, float %700, i64 5, !dbg !14
  %768 = insertelement <32 x float> %767, float %701, i64 6, !dbg !14
  %769 = insertelement <32 x float> %768, float %702, i64 7, !dbg !14
  %770 = insertelement <32 x float> %769, float %703, i64 8, !dbg !14
  %771 = insertelement <32 x float> %770, float %704, i64 9, !dbg !14
  %772 = insertelement <32 x float> %771, float %705, i64 10, !dbg !14
  %773 = insertelement <32 x float> %772, float %706, i64 11, !dbg !14
  %774 = insertelement <32 x float> %773, float %707, i64 12, !dbg !14
  %775 = insertelement <32 x float> %774, float %708, i64 13, !dbg !14
  %776 = insertelement <32 x float> %775, float %709, i64 14, !dbg !14
  %777 = insertelement <32 x float> %776, float %710, i64 15, !dbg !14
  %778 = insertelement <32 x float> %777, float %711, i64 16, !dbg !14
  %779 = insertelement <32 x float> %778, float %712, i64 17, !dbg !14
  %780 = insertelement <32 x float> %779, float %713, i64 18, !dbg !14
  %781 = insertelement <32 x float> %780, float %714, i64 19, !dbg !14
  %782 = insertelement <32 x float> %781, float %715, i64 20, !dbg !14
  %783 = insertelement <32 x float> %782, float %716, i64 21, !dbg !14
  %784 = insertelement <32 x float> %783, float %717, i64 22, !dbg !14
  %785 = insertelement <32 x float> %784, float %718, i64 23, !dbg !14
  %786 = insertelement <32 x float> %785, float %719, i64 24, !dbg !14
  %787 = insertelement <32 x float> %786, float %720, i64 25, !dbg !14
  %788 = insertelement <32 x float> %787, float %721, i64 26, !dbg !14
  %789 = insertelement <32 x float> %788, float %722, i64 27, !dbg !14
  %790 = insertelement <32 x float> %789, float %723, i64 28, !dbg !14
  %791 = insertelement <32 x float> %790, float %724, i64 29, !dbg !14
  %792 = insertelement <32 x float> %791, float %725, i64 30, !dbg !14
  %793 = insertelement <32 x float> %792, float %726, i64 31, !dbg !14
  %794 = insertvalue [2 x <32 x float>] poison, <32 x float> %793, 0, !dbg !14
  %795 = insertelement <32 x float> poison, float %730, i64 0, !dbg !14
  %796 = insertelement <32 x float> %795, float %731, i64 1, !dbg !14
  %797 = insertelement <32 x float> %796, float %732, i64 2, !dbg !14
  %798 = insertelement <32 x float> %797, float %733, i64 3, !dbg !14
  %799 = insertelement <32 x float> %798, float %734, i64 4, !dbg !14
  %800 = insertelement <32 x float> %799, float %735, i64 5, !dbg !14
  %801 = insertelement <32 x float> %800, float %736, i64 6, !dbg !14
  %802 = insertelement <32 x float> %801, float %737, i64 7, !dbg !14
  %803 = insertelement <32 x float> %802, float %738, i64 8, !dbg !14
  %804 = insertelement <32 x float> %803, float %739, i64 9, !dbg !14
  %805 = insertelement <32 x float> %804, float %740, i64 10, !dbg !14
  %806 = insertelement <32 x float> %805, float %741, i64 11, !dbg !14
  %807 = insertelement <32 x float> %806, float %742, i64 12, !dbg !14
  %808 = insertelement <32 x float> %807, float %743, i64 13, !dbg !14
  %809 = insertelement <32 x float> %808, float %744, i64 14, !dbg !14
  %810 = insertelement <32 x float> %809, float %745, i64 15, !dbg !14
  %811 = insertelement <32 x float> %810, float %746, i64 16, !dbg !14
  %812 = insertelement <32 x float> %811, float %747, i64 17, !dbg !14
  %813 = insertelement <32 x float> %812, float %748, i64 18, !dbg !14
  %814 = insertelement <32 x float> %813, float %749, i64 19, !dbg !14
  %815 = insertelement <32 x float> %814, float %750, i64 20, !dbg !14
  %816 = insertelement <32 x float> %815, float %751, i64 21, !dbg !14
  %817 = insertelement <32 x float> %816, float %752, i64 22, !dbg !14
  %818 = insertelement <32 x float> %817, float %753, i64 23, !dbg !14
  %819 = insertelement <32 x float> %818, float %754, i64 24, !dbg !14
  %820 = insertelement <32 x float> %819, float %755, i64 25, !dbg !14
  %821 = insertelement <32 x float> %820, float %756, i64 26, !dbg !14
  %822 = insertelement <32 x float> %821, float %757, i64 27, !dbg !14
  %823 = insertelement <32 x float> %822, float %758, i64 28, !dbg !14
  %824 = insertelement <32 x float> %823, float %759, i64 29, !dbg !14
  %825 = insertelement <32 x float> %824, float %760, i64 30, !dbg !14
  %826 = insertelement <32 x float> %825, float %761, i64 31, !dbg !14
  %827 = insertvalue [2 x <32 x float>] %794, <32 x float> %826, 1, !dbg !14
  br label %828, !dbg !14

828:                                              ; preds = %683, %684
  %829 = phi [2 x <32 x float>] [ %827, %684 ], [ zeroinitializer, %683 ], !dbg !14
  br label %830, !dbg !14

830:                                              ; preds = %828
  %831 = extractvalue [2 x <32 x float>] %829, 0, !dbg !28
  %832 = shufflevector <32 x float> %831, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %833 = fadd <4 x float> %620, %832, !dbg !28
  %834 = extractelement <4 x float> %833, i64 0, !dbg !29
  %835 = call float @llvm.amdgcn.exp2.f32(float %834), !dbg !29
  %836 = insertelement <4 x float> poison, float %835, i64 0, !dbg !31
  %837 = extractelement <4 x float> %833, i64 1, !dbg !29
  %838 = call float @llvm.amdgcn.exp2.f32(float %837), !dbg !29
  %839 = insertelement <4 x float> %836, float %838, i64 1, !dbg !31
  %840 = fadd float %835, %838, !dbg !30
  %841 = extractelement <4 x float> %833, i64 2, !dbg !29
  %842 = call float @llvm.amdgcn.exp2.f32(float %841), !dbg !29
  %843 = insertelement <4 x float> %839, float %842, i64 2, !dbg !31
  %844 = fadd float %840, %842, !dbg !30
  %845 = extractelement <4 x float> %833, i64 3, !dbg !29
  %846 = call float @llvm.amdgcn.exp2.f32(float %845), !dbg !29
  %847 = insertelement <4 x float> %843, float %846, i64 3, !dbg !31
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %847) #9, !dbg !31, !frisk.register_lifetimes !1
  %848 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !31
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %848) #9, !dbg !30, !frisk.register_lifetimes !1
  %849 = fadd float %844, %846, !dbg !30
  %850 = shufflevector <32 x float> %831, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %851 = fadd <4 x float> %624, %850, !dbg !28
  %852 = extractelement <4 x float> %851, i64 0, !dbg !29
  %853 = call float @llvm.amdgcn.exp2.f32(float %852), !dbg !29
  %854 = insertelement <4 x float> poison, float %853, i64 0, !dbg !31
  %855 = fadd float %849, %853, !dbg !30
  %856 = extractelement <4 x float> %851, i64 1, !dbg !29
  %857 = call float @llvm.amdgcn.exp2.f32(float %856), !dbg !29
  %858 = insertelement <4 x float> %854, float %857, i64 1, !dbg !31
  %859 = fadd float %855, %857, !dbg !30
  %860 = extractelement <4 x float> %851, i64 2, !dbg !29
  %861 = call float @llvm.amdgcn.exp2.f32(float %860), !dbg !29
  %862 = insertelement <4 x float> %858, float %861, i64 2, !dbg !31
  %863 = fadd float %859, %861, !dbg !30
  %864 = extractelement <4 x float> %851, i64 3, !dbg !29
  %865 = call float @llvm.amdgcn.exp2.f32(float %864), !dbg !29
  %866 = insertelement <4 x float> %862, float %865, i64 3, !dbg !31
  %math.result144 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %866) #9, !dbg !31, !frisk.register_lifetimes !1
  %867 = fptrunc <4 x float> %math.result144 to <4 x half>, !dbg !31
  %math.packed145 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %867) #9, !dbg !30, !frisk.register_lifetimes !1
  %868 = fadd float %863, %865, !dbg !30
  %869 = shufflevector <32 x float> %831, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !28
  %870 = fadd <4 x float> %628, %869, !dbg !28
  %871 = extractelement <4 x float> %870, i64 0, !dbg !29
  %872 = call float @llvm.amdgcn.exp2.f32(float %871), !dbg !29
  %873 = insertelement <4 x float> poison, float %872, i64 0, !dbg !31
  %874 = fadd float %868, %872, !dbg !30
  %875 = extractelement <4 x float> %870, i64 1, !dbg !29
  %876 = call float @llvm.amdgcn.exp2.f32(float %875), !dbg !29
  %877 = insertelement <4 x float> %873, float %876, i64 1, !dbg !31
  %878 = fadd float %874, %876, !dbg !30
  %879 = extractelement <4 x float> %870, i64 2, !dbg !29
  %880 = call float @llvm.amdgcn.exp2.f32(float %879), !dbg !29
  %881 = insertelement <4 x float> %877, float %880, i64 2, !dbg !31
  %882 = fadd float %878, %880, !dbg !30
  %883 = extractelement <4 x float> %870, i64 3, !dbg !29
  %884 = call float @llvm.amdgcn.exp2.f32(float %883), !dbg !29
  %885 = insertelement <4 x float> %881, float %884, i64 3, !dbg !31
  %math.result146 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %885) #9, !dbg !31, !frisk.register_lifetimes !1
  %886 = fptrunc <4 x float> %math.result146 to <4 x half>, !dbg !31
  %math.packed147 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %886) #9, !dbg !30, !frisk.register_lifetimes !1
  %887 = fadd float %882, %884, !dbg !30
  %888 = shufflevector <32 x float> %831, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !28
  %889 = fadd <4 x float> %632, %888, !dbg !28
  %890 = extractelement <4 x float> %889, i64 0, !dbg !29
  %891 = call float @llvm.amdgcn.exp2.f32(float %890), !dbg !29
  %892 = insertelement <4 x float> poison, float %891, i64 0, !dbg !31
  %893 = fadd float %887, %891, !dbg !30
  %894 = extractelement <4 x float> %889, i64 1, !dbg !29
  %895 = call float @llvm.amdgcn.exp2.f32(float %894), !dbg !29
  %896 = insertelement <4 x float> %892, float %895, i64 1, !dbg !31
  %897 = fadd float %893, %895, !dbg !30
  %898 = extractelement <4 x float> %889, i64 2, !dbg !29
  %899 = call float @llvm.amdgcn.exp2.f32(float %898), !dbg !29
  %900 = insertelement <4 x float> %896, float %899, i64 2, !dbg !31
  %901 = fadd float %897, %899, !dbg !30
  %902 = extractelement <4 x float> %889, i64 3, !dbg !29
  %903 = call float @llvm.amdgcn.exp2.f32(float %902), !dbg !29
  %904 = insertelement <4 x float> %900, float %903, i64 3, !dbg !31
  %math.result148 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %904) #9, !dbg !31, !frisk.register_lifetimes !1
  %905 = fptrunc <4 x float> %math.result148 to <4 x half>, !dbg !31
  %math.packed149 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %905) #9, !dbg !30, !frisk.register_lifetimes !1
  %906 = fadd float %901, %903, !dbg !30
  %907 = shufflevector <32 x float> %831, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !28
  %908 = fadd <4 x float> %636, %907, !dbg !28
  %909 = extractelement <4 x float> %908, i64 0, !dbg !29
  %910 = call float @llvm.amdgcn.exp2.f32(float %909), !dbg !29
  %911 = insertelement <4 x float> poison, float %910, i64 0, !dbg !31
  %912 = fadd float %906, %910, !dbg !30
  %913 = extractelement <4 x float> %908, i64 1, !dbg !29
  %914 = call float @llvm.amdgcn.exp2.f32(float %913), !dbg !29
  %915 = insertelement <4 x float> %911, float %914, i64 1, !dbg !31
  %916 = fadd float %912, %914, !dbg !30
  %917 = extractelement <4 x float> %908, i64 2, !dbg !29
  %918 = call float @llvm.amdgcn.exp2.f32(float %917), !dbg !29
  %919 = insertelement <4 x float> %915, float %918, i64 2, !dbg !31
  %920 = fadd float %916, %918, !dbg !30
  %921 = extractelement <4 x float> %908, i64 3, !dbg !29
  %922 = call float @llvm.amdgcn.exp2.f32(float %921), !dbg !29
  %923 = insertelement <4 x float> %919, float %922, i64 3, !dbg !31
  %math.result150 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %923) #9, !dbg !31, !frisk.register_lifetimes !1
  %924 = fptrunc <4 x float> %math.result150 to <4 x half>, !dbg !31
  %math.packed151 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %924) #9, !dbg !30, !frisk.register_lifetimes !1
  %925 = fadd float %920, %922, !dbg !30
  %926 = shufflevector <32 x float> %831, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !28
  %927 = fadd <4 x float> %640, %926, !dbg !28
  %928 = extractelement <4 x float> %927, i64 0, !dbg !29
  %929 = call float @llvm.amdgcn.exp2.f32(float %928), !dbg !29
  %930 = insertelement <4 x float> poison, float %929, i64 0, !dbg !31
  %931 = fadd float %925, %929, !dbg !30
  %932 = extractelement <4 x float> %927, i64 1, !dbg !29
  %933 = call float @llvm.amdgcn.exp2.f32(float %932), !dbg !29
  %934 = insertelement <4 x float> %930, float %933, i64 1, !dbg !31
  %935 = fadd float %931, %933, !dbg !30
  %936 = extractelement <4 x float> %927, i64 2, !dbg !29
  %937 = call float @llvm.amdgcn.exp2.f32(float %936), !dbg !29
  %938 = insertelement <4 x float> %934, float %937, i64 2, !dbg !31
  %939 = fadd float %935, %937, !dbg !30
  %940 = extractelement <4 x float> %927, i64 3, !dbg !29
  %941 = call float @llvm.amdgcn.exp2.f32(float %940), !dbg !29
  %942 = insertelement <4 x float> %938, float %941, i64 3, !dbg !31
  %math.result152 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %942) #9, !dbg !31, !frisk.register_lifetimes !1
  %943 = fptrunc <4 x float> %math.result152 to <4 x half>, !dbg !31
  %math.packed153 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %943) #9, !dbg !30, !frisk.register_lifetimes !1
  %944 = fadd float %939, %941, !dbg !30
  %945 = shufflevector <32 x float> %831, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !28
  %946 = fadd <4 x float> %644, %945, !dbg !28
  %947 = extractelement <4 x float> %946, i64 0, !dbg !29
  %948 = call float @llvm.amdgcn.exp2.f32(float %947), !dbg !29
  %949 = insertelement <4 x float> poison, float %948, i64 0, !dbg !31
  %950 = fadd float %944, %948, !dbg !30
  %951 = extractelement <4 x float> %946, i64 1, !dbg !29
  %952 = call float @llvm.amdgcn.exp2.f32(float %951), !dbg !29
  %953 = insertelement <4 x float> %949, float %952, i64 1, !dbg !31
  %954 = fadd float %950, %952, !dbg !30
  %955 = extractelement <4 x float> %946, i64 2, !dbg !29
  %956 = call float @llvm.amdgcn.exp2.f32(float %955), !dbg !29
  %957 = insertelement <4 x float> %953, float %956, i64 2, !dbg !31
  %958 = fadd float %954, %956, !dbg !30
  %959 = extractelement <4 x float> %946, i64 3, !dbg !29
  %960 = call float @llvm.amdgcn.exp2.f32(float %959), !dbg !29
  %961 = insertelement <4 x float> %957, float %960, i64 3, !dbg !31
  %math.result154 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %961) #9, !dbg !31, !frisk.register_lifetimes !1
  %962 = fptrunc <4 x float> %math.result154 to <4 x half>, !dbg !31
  %math.packed155 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %962) #9, !dbg !30, !frisk.register_lifetimes !1
  %963 = fadd float %958, %960, !dbg !30
  %964 = shufflevector <32 x float> %831, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !28
  %965 = fadd <4 x float> %648, %964, !dbg !28
  %966 = extractelement <4 x float> %965, i64 0, !dbg !29
  %967 = call float @llvm.amdgcn.exp2.f32(float %966), !dbg !29
  %968 = insertelement <4 x float> poison, float %967, i64 0, !dbg !31
  %969 = fadd float %963, %967, !dbg !30
  %970 = extractelement <4 x float> %965, i64 1, !dbg !29
  %971 = call float @llvm.amdgcn.exp2.f32(float %970), !dbg !29
  %972 = insertelement <4 x float> %968, float %971, i64 1, !dbg !31
  %973 = fadd float %969, %971, !dbg !30
  %974 = extractelement <4 x float> %965, i64 2, !dbg !29
  %975 = call float @llvm.amdgcn.exp2.f32(float %974), !dbg !29
  %976 = insertelement <4 x float> %972, float %975, i64 2, !dbg !31
  %977 = fadd float %973, %975, !dbg !30
  %978 = extractelement <4 x float> %965, i64 3, !dbg !29
  %979 = call float @llvm.amdgcn.exp2.f32(float %978), !dbg !29
  %980 = insertelement <4 x float> %976, float %979, i64 3, !dbg !31
  %math.result156 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %980) #9, !dbg !31, !frisk.register_lifetimes !1
  %981 = fptrunc <4 x float> %math.result156 to <4 x half>, !dbg !31
  %math.packed157 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %981) #9, !dbg !30, !frisk.register_lifetimes !1
  %982 = fadd float %977, %979, !dbg !30
  %983 = extractvalue [2 x <32 x float>] %829, 1, !dbg !28
  %984 = shufflevector <32 x float> %983, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %985 = fadd <4 x float> %652, %984, !dbg !28
  %986 = extractelement <4 x float> %985, i64 0, !dbg !29
  %987 = call float @llvm.amdgcn.exp2.f32(float %986), !dbg !29
  %988 = insertelement <4 x float> poison, float %987, i64 0, !dbg !31
  %989 = extractelement <4 x float> %985, i64 1, !dbg !29
  %990 = call float @llvm.amdgcn.exp2.f32(float %989), !dbg !29
  %991 = insertelement <4 x float> %988, float %990, i64 1, !dbg !31
  %992 = fadd float %987, %990, !dbg !30
  %993 = extractelement <4 x float> %985, i64 2, !dbg !29
  %994 = call float @llvm.amdgcn.exp2.f32(float %993), !dbg !29
  %995 = insertelement <4 x float> %991, float %994, i64 2, !dbg !31
  %996 = fadd float %992, %994, !dbg !30
  %997 = extractelement <4 x float> %985, i64 3, !dbg !29
  %998 = call float @llvm.amdgcn.exp2.f32(float %997), !dbg !29
  %999 = insertelement <4 x float> %995, float %998, i64 3, !dbg !31
  %math.result158 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %999) #9, !dbg !31, !frisk.register_lifetimes !1
  %1000 = fptrunc <4 x float> %math.result158 to <4 x half>, !dbg !31
  %math.packed159 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1000) #9, !dbg !30, !frisk.register_lifetimes !1
  %1001 = fadd float %996, %998, !dbg !30
  %1002 = shufflevector <32 x float> %983, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1003 = fadd <4 x float> %656, %1002, !dbg !28
  %1004 = extractelement <4 x float> %1003, i64 0, !dbg !29
  %1005 = call float @llvm.amdgcn.exp2.f32(float %1004), !dbg !29
  %1006 = insertelement <4 x float> poison, float %1005, i64 0, !dbg !31
  %1007 = fadd float %1001, %1005, !dbg !30
  %1008 = extractelement <4 x float> %1003, i64 1, !dbg !29
  %1009 = call float @llvm.amdgcn.exp2.f32(float %1008), !dbg !29
  %1010 = insertelement <4 x float> %1006, float %1009, i64 1, !dbg !31
  %1011 = fadd float %1007, %1009, !dbg !30
  %1012 = extractelement <4 x float> %1003, i64 2, !dbg !29
  %1013 = call float @llvm.amdgcn.exp2.f32(float %1012), !dbg !29
  %1014 = insertelement <4 x float> %1010, float %1013, i64 2, !dbg !31
  %1015 = fadd float %1011, %1013, !dbg !30
  %1016 = extractelement <4 x float> %1003, i64 3, !dbg !29
  %1017 = call float @llvm.amdgcn.exp2.f32(float %1016), !dbg !29
  %1018 = insertelement <4 x float> %1014, float %1017, i64 3, !dbg !31
  %math.result160 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1018) #9, !dbg !31, !frisk.register_lifetimes !1
  %1019 = fptrunc <4 x float> %math.result160 to <4 x half>, !dbg !31
  %math.packed161 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1019) #9, !dbg !30, !frisk.register_lifetimes !1
  %1020 = fadd float %1015, %1017, !dbg !30
  %1021 = shufflevector <32 x float> %983, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !28
  %1022 = fadd <4 x float> %660, %1021, !dbg !28
  %1023 = extractelement <4 x float> %1022, i64 0, !dbg !29
  %1024 = call float @llvm.amdgcn.exp2.f32(float %1023), !dbg !29
  %1025 = insertelement <4 x float> poison, float %1024, i64 0, !dbg !31
  %1026 = fadd float %1020, %1024, !dbg !30
  %1027 = extractelement <4 x float> %1022, i64 1, !dbg !29
  %1028 = call float @llvm.amdgcn.exp2.f32(float %1027), !dbg !29
  %1029 = insertelement <4 x float> %1025, float %1028, i64 1, !dbg !31
  %1030 = fadd float %1026, %1028, !dbg !30
  %1031 = extractelement <4 x float> %1022, i64 2, !dbg !29
  %1032 = call float @llvm.amdgcn.exp2.f32(float %1031), !dbg !29
  %1033 = insertelement <4 x float> %1029, float %1032, i64 2, !dbg !31
  %1034 = fadd float %1030, %1032, !dbg !30
  %1035 = extractelement <4 x float> %1022, i64 3, !dbg !29
  %1036 = call float @llvm.amdgcn.exp2.f32(float %1035), !dbg !29
  %1037 = insertelement <4 x float> %1033, float %1036, i64 3, !dbg !31
  %math.result162 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1037) #9, !dbg !31, !frisk.register_lifetimes !1
  %1038 = fptrunc <4 x float> %math.result162 to <4 x half>, !dbg !31
  %math.packed163 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1038) #9, !dbg !30, !frisk.register_lifetimes !1
  %1039 = fadd float %1034, %1036, !dbg !30
  %1040 = shufflevector <32 x float> %983, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !28
  %1041 = fadd <4 x float> %664, %1040, !dbg !28
  %1042 = extractelement <4 x float> %1041, i64 0, !dbg !29
  %1043 = call float @llvm.amdgcn.exp2.f32(float %1042), !dbg !29
  %1044 = insertelement <4 x float> poison, float %1043, i64 0, !dbg !31
  %1045 = fadd float %1039, %1043, !dbg !30
  %1046 = extractelement <4 x float> %1041, i64 1, !dbg !29
  %1047 = call float @llvm.amdgcn.exp2.f32(float %1046), !dbg !29
  %1048 = insertelement <4 x float> %1044, float %1047, i64 1, !dbg !31
  %1049 = fadd float %1045, %1047, !dbg !30
  %1050 = extractelement <4 x float> %1041, i64 2, !dbg !29
  %1051 = call float @llvm.amdgcn.exp2.f32(float %1050), !dbg !29
  %1052 = insertelement <4 x float> %1048, float %1051, i64 2, !dbg !31
  %1053 = fadd float %1049, %1051, !dbg !30
  %1054 = extractelement <4 x float> %1041, i64 3, !dbg !29
  %1055 = call float @llvm.amdgcn.exp2.f32(float %1054), !dbg !29
  %1056 = insertelement <4 x float> %1052, float %1055, i64 3, !dbg !31
  %math.result164 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1056) #9, !dbg !31, !frisk.register_lifetimes !1
  %1057 = fptrunc <4 x float> %math.result164 to <4 x half>, !dbg !31
  %math.packed165 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1057) #9, !dbg !30, !frisk.register_lifetimes !1
  %1058 = fadd float %1053, %1055, !dbg !30
  %1059 = shufflevector <32 x float> %983, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !28
  %1060 = fadd <4 x float> %668, %1059, !dbg !28
  %1061 = extractelement <4 x float> %1060, i64 0, !dbg !29
  %1062 = call float @llvm.amdgcn.exp2.f32(float %1061), !dbg !29
  %1063 = insertelement <4 x float> poison, float %1062, i64 0, !dbg !31
  %1064 = fadd float %1058, %1062, !dbg !30
  %1065 = extractelement <4 x float> %1060, i64 1, !dbg !29
  %1066 = call float @llvm.amdgcn.exp2.f32(float %1065), !dbg !29
  %1067 = insertelement <4 x float> %1063, float %1066, i64 1, !dbg !31
  %1068 = fadd float %1064, %1066, !dbg !30
  %1069 = extractelement <4 x float> %1060, i64 2, !dbg !29
  %1070 = call float @llvm.amdgcn.exp2.f32(float %1069), !dbg !29
  %1071 = insertelement <4 x float> %1067, float %1070, i64 2, !dbg !31
  %1072 = fadd float %1068, %1070, !dbg !30
  %1073 = extractelement <4 x float> %1060, i64 3, !dbg !29
  %1074 = call float @llvm.amdgcn.exp2.f32(float %1073), !dbg !29
  %1075 = insertelement <4 x float> %1071, float %1074, i64 3, !dbg !31
  %math.result166 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1075) #9, !dbg !31, !frisk.register_lifetimes !1
  %1076 = fptrunc <4 x float> %math.result166 to <4 x half>, !dbg !31
  %math.packed167 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1076) #9, !dbg !30, !frisk.register_lifetimes !1
  %1077 = fadd float %1072, %1074, !dbg !30
  %1078 = shufflevector <32 x float> %983, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !28
  %1079 = fadd <4 x float> %672, %1078, !dbg !28
  %1080 = extractelement <4 x float> %1079, i64 0, !dbg !29
  %1081 = call float @llvm.amdgcn.exp2.f32(float %1080), !dbg !29
  %1082 = insertelement <4 x float> poison, float %1081, i64 0, !dbg !31
  %1083 = fadd float %1077, %1081, !dbg !30
  %1084 = extractelement <4 x float> %1079, i64 1, !dbg !29
  %1085 = call float @llvm.amdgcn.exp2.f32(float %1084), !dbg !29
  %1086 = insertelement <4 x float> %1082, float %1085, i64 1, !dbg !31
  %1087 = fadd float %1083, %1085, !dbg !30
  %1088 = extractelement <4 x float> %1079, i64 2, !dbg !29
  %1089 = call float @llvm.amdgcn.exp2.f32(float %1088), !dbg !29
  %1090 = insertelement <4 x float> %1086, float %1089, i64 2, !dbg !31
  %1091 = fadd float %1087, %1089, !dbg !30
  %1092 = extractelement <4 x float> %1079, i64 3, !dbg !29
  %1093 = call float @llvm.amdgcn.exp2.f32(float %1092), !dbg !29
  %1094 = insertelement <4 x float> %1090, float %1093, i64 3, !dbg !31
  %math.result168 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1094) #9, !dbg !31, !frisk.register_lifetimes !1
  %1095 = fptrunc <4 x float> %math.result168 to <4 x half>, !dbg !31
  %math.packed169 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1095) #9, !dbg !30, !frisk.register_lifetimes !1
  %1096 = fadd float %1091, %1093, !dbg !30
  %1097 = shufflevector <32 x float> %983, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !28
  %1098 = fadd <4 x float> %676, %1097, !dbg !28
  %1099 = extractelement <4 x float> %1098, i64 0, !dbg !29
  %1100 = call float @llvm.amdgcn.exp2.f32(float %1099), !dbg !29
  %1101 = insertelement <4 x float> poison, float %1100, i64 0, !dbg !31
  %1102 = fadd float %1096, %1100, !dbg !30
  %1103 = extractelement <4 x float> %1098, i64 1, !dbg !29
  %1104 = call float @llvm.amdgcn.exp2.f32(float %1103), !dbg !29
  %1105 = insertelement <4 x float> %1101, float %1104, i64 1, !dbg !31
  %1106 = fadd float %1102, %1104, !dbg !30
  %1107 = extractelement <4 x float> %1098, i64 2, !dbg !29
  %1108 = call float @llvm.amdgcn.exp2.f32(float %1107), !dbg !29
  %1109 = insertelement <4 x float> %1105, float %1108, i64 2, !dbg !31
  %1110 = fadd float %1106, %1108, !dbg !30
  %1111 = extractelement <4 x float> %1098, i64 3, !dbg !29
  %1112 = call float @llvm.amdgcn.exp2.f32(float %1111), !dbg !29
  %1113 = insertelement <4 x float> %1109, float %1112, i64 3, !dbg !31
  %math.result170 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1113) #9, !dbg !31, !frisk.register_lifetimes !1
  %1114 = fptrunc <4 x float> %math.result170 to <4 x half>, !dbg !31
  %math.packed171 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1114) #9, !dbg !30, !frisk.register_lifetimes !1
  %1115 = fadd float %1110, %1112, !dbg !30
  %1116 = shufflevector <32 x float> %983, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !28
  %1117 = fadd <4 x float> %681, %1116, !dbg !28
  %1118 = extractelement <4 x float> %1117, i64 0, !dbg !29
  %1119 = call float @llvm.amdgcn.exp2.f32(float %1118), !dbg !29
  %1120 = insertelement <4 x float> poison, float %1119, i64 0, !dbg !31
  %1121 = fadd float %1115, %1119, !dbg !30
  %1122 = extractelement <4 x float> %1117, i64 1, !dbg !29
  %1123 = call float @llvm.amdgcn.exp2.f32(float %1122), !dbg !29
  %1124 = insertelement <4 x float> %1120, float %1123, i64 1, !dbg !31
  %1125 = fadd float %1121, %1123, !dbg !30
  %1126 = extractelement <4 x float> %1117, i64 2, !dbg !29
  %1127 = call float @llvm.amdgcn.exp2.f32(float %1126), !dbg !29
  %1128 = insertelement <4 x float> %1124, float %1127, i64 2, !dbg !31
  %1129 = fadd float %1125, %1127, !dbg !30
  %1130 = extractelement <4 x float> %1117, i64 3, !dbg !29
  %1131 = call float @llvm.amdgcn.exp2.f32(float %1130), !dbg !29
  %1132 = insertelement <4 x float> %1128, float %1131, i64 3, !dbg !31
  %math.result172 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1132) #9, !dbg !31, !frisk.register_lifetimes !1
  %1133 = fptrunc <4 x float> %math.result172 to <4 x half>, !dbg !31
  %math.packed173 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1133) #9, !dbg !30, !frisk.register_lifetimes !1
  %1134 = fadd float %1129, %1131, !dbg !30
  %1135 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !30
  %1136 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1135), !dbg !30
  %1137 = add nuw nsw i32 %1136, 64, !dbg !30
  %1138 = xor i32 %1136, 16, !dbg !30
  %1139 = and i32 %1137, 192, !dbg !30
  %1140 = icmp samesign ult i32 %1138, %1139, !dbg !30
  %1141 = select i1 %1140, i32 %1138, i32 %1136, !dbg !30
  %1142 = shl nuw nsw i32 %1141, 2, !dbg !30
  %1143 = bitcast float %982 to i32, !dbg !30
  %1144 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1143), !dbg !30
  %1145 = bitcast i32 %1144 to float, !dbg !30
  %1146 = fadd float %982, %1145, !dbg !30
  %1147 = xor i32 %1136, 32, !dbg !30
  %1148 = icmp samesign ult i32 %1147, %1139, !dbg !30
  %1149 = select i1 %1148, i32 %1147, i32 %1136, !dbg !30
  %1150 = shl nuw nsw i32 %1149, 2, !dbg !30
  %1151 = bitcast float %1146 to i32, !dbg !30
  %1152 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1150, i32 %1151), !dbg !30
  %1153 = bitcast i32 %1152 to float, !dbg !30
  %1154 = fadd float %1146, %1153, !dbg !30
  %1155 = extractvalue [2 x <1 x float>] %48, 0, !dbg !30
  %1156 = extractelement <1 x float> %1155, i64 0, !dbg !30
  %1157 = fadd float %1154, %1156, !dbg !30
  %1158 = insertelement <1 x float> poison, float %1157, i64 0, !dbg !16
  %1159 = insertvalue [2 x <1 x float>] poison, <1 x float> %1158, 0, !dbg !16
  %1160 = bitcast float %1134 to i32, !dbg !30
  %1161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1160), !dbg !30
  %1162 = bitcast i32 %1161 to float, !dbg !30
  %1163 = fadd float %1134, %1162, !dbg !30
  %1164 = bitcast float %1163 to i32, !dbg !30
  %1165 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1150, i32 %1164), !dbg !30
  %1166 = bitcast i32 %1165 to float, !dbg !30
  %1167 = fadd float %1163, %1166, !dbg !30
  %1168 = extractvalue [2 x <1 x float>] %48, 1, !dbg !30
  %1169 = extractelement <1 x float> %1168, i64 0, !dbg !30
  %1170 = fadd float %1167, %1169, !dbg !30
  %1171 = insertelement <1 x float> poison, float %1170, i64 0, !dbg !16
  %1172 = insertvalue [2 x <1 x float>] %1159, <1 x float> %1171, 1, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread539 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !15, !range !35, !frisk.register_lifetimes !1
  %address.local540 = shl nuw nsw i32 %address.thread539, 3, !dbg !15
  %address.local541 = and i32 %address.local540, 120, !dbg !15
  %address.local542 = shl nuw nsw i32 %address.thread539, 5, !dbg !15
  %address.local543 = and i32 %address.local542, 1536, !dbg !15
  %address.local544 = or disjoint i32 %address.local541, %address.local543, !dbg !15
  %address.local545 = zext nneg i32 %address.local544 to i64, !dbg !15
  %address.local546 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local545, !dbg !15
  %1173 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local546, i32 0) #0, !dbg !15
  %1174 = bitcast <4 x i32> %1173 to <8 x half>, !dbg !15
  %address.local547 = zext nneg i32 %address.thread539 to i64, !dbg !13
  %address.local548 = and i64 %address.local547, 60, !dbg !15
  %address.local549 = add nuw nsw i64 %address.local548, %address.local547, !dbg !15
  %address.local550 = shl nuw nsw i64 %address.local549, 3, !dbg !15
  %address.local551 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local550, !dbg !15
  %address.local552 = lshr i64 %address.local547, 4, !dbg !15
  %address.local553 = and i64 %address.local552, 3, !dbg !15
  %address.local554 = mul nuw nsw i64 %address.local553, 384, !dbg !15
  %address.local555 = getelementptr [2 x i8], ptr addrspace(3) %address.local551, i64 %address.local554, !dbg !15
  %address.local556 = lshr i64 %address.local547, 2, !dbg !18
  %address.local557 = mul nsw i64 %address.local556, -32, !dbg !15
  %address.local558 = getelementptr [2 x i8], ptr addrspace(3) %address.local555, i64 %address.local557, !dbg !15
  %address.local559 = getelementptr i8, ptr addrspace(3) %address.local558, i64 256, !dbg !15
  %1175 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local559, i32 0) #0, !dbg !15
  %1176 = bitcast <4 x i32> %1175 to <8 x half>, !dbg !15
  %address.local560 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local550, !dbg !15
  %address.local561 = getelementptr [2 x i8], ptr addrspace(3) %address.local560, i64 %address.local554, !dbg !15
  %address.local562 = getelementptr [2 x i8], ptr addrspace(3) %address.local561, i64 %address.local557, !dbg !15
  %address.local563 = getelementptr i8, ptr addrspace(3) %address.local562, i64 512, !dbg !15
  %1177 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local563, i32 0) #0, !dbg !15
  %1178 = bitcast <4 x i32> %1177 to <8 x half>, !dbg !15
  %address.local564 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local550, !dbg !15
  %address.local565 = getelementptr [2 x i8], ptr addrspace(3) %address.local564, i64 %address.local554, !dbg !15
  %address.local566 = getelementptr [2 x i8], ptr addrspace(3) %address.local565, i64 %address.local557, !dbg !15
  %address.local567 = getelementptr i8, ptr addrspace(3) %address.local566, i64 768, !dbg !15
  %1179 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local567, i32 0) #0, !dbg !15
  %1180 = bitcast <4 x i32> %1179 to <8 x half>, !dbg !15
  %address.local568 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local545, !dbg !15
  %address.local569 = getelementptr i8, ptr addrspace(3) %address.local568, i64 4096, !dbg !15
  %1181 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local569, i32 0) #0, !dbg !15
  %1182 = bitcast <4 x i32> %1181 to <8 x half>, !dbg !15
  %address.local570 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local550, !dbg !15
  %address.local571 = getelementptr [2 x i8], ptr addrspace(3) %address.local570, i64 %address.local554, !dbg !15
  %address.local572 = getelementptr [2 x i8], ptr addrspace(3) %address.local571, i64 %address.local557, !dbg !15
  %address.local573 = getelementptr i8, ptr addrspace(3) %address.local572, i64 4352, !dbg !15
  %1183 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local573, i32 0) #0, !dbg !15
  %1184 = bitcast <4 x i32> %1183 to <8 x half>, !dbg !15
  %address.local574 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local550, !dbg !15
  %address.local575 = getelementptr [2 x i8], ptr addrspace(3) %address.local574, i64 %address.local554, !dbg !15
  %address.local576 = getelementptr [2 x i8], ptr addrspace(3) %address.local575, i64 %address.local557, !dbg !15
  %address.local577 = getelementptr i8, ptr addrspace(3) %address.local576, i64 4608, !dbg !15
  %1185 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local577, i32 0) #0, !dbg !15
  %1186 = bitcast <4 x i32> %1185 to <8 x half>, !dbg !15
  %address.local578 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local550, !dbg !15
  %address.local579 = getelementptr [2 x i8], ptr addrspace(3) %address.local578, i64 %address.local554, !dbg !15
  %address.local580 = getelementptr [2 x i8], ptr addrspace(3) %address.local579, i64 %address.local557, !dbg !15
  %address.local581 = getelementptr i8, ptr addrspace(3) %address.local580, i64 4864, !dbg !15
  %1187 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local581, i32 0) #0, !dbg !15
  %1188 = bitcast <4 x i32> %1187 to <8 x half>, !dbg !15
  %1189 = add nsw i64 %84, %10, !dbg !19
  %1190 = shl i64 %1189, 19, !dbg !19
  %1191 = and i64 %1190, 16252928, !dbg !19
  %1192 = add i64 %1191, %90, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1193 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1192, !dbg !19
  %1194 = getelementptr i8, ptr addrspace(1) %1193, i64 64, !dbg !19
  %address.local582 = shl nuw nsw i64 %address.local547, 3, !dbg !12
  %address.local583 = and i64 %address.local582, 1536, !dbg !12
  %address.local584 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local583, !dbg !12
  %address.local585 = ptrtoint ptr addrspace(3) %address.local584 to i64, !dbg !12
  %address.local586 = trunc i64 %address.local585 to i32, !dbg !12
  %1195 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local586), !dbg !19
  %1196 = zext i32 %1195 to i64, !dbg !19
  %1197 = inttoptr i64 %1196 to ptr addrspace(3), !dbg !19
  %address.local587 = shl nuw nsw i64 %address.local547, 5, !dbg !19
  %address.local588 = and i64 %address.local587, 8064, !dbg !19
  %address.local589 = shl nuw nsw i64 %address.local547, 3, !dbg !19
  %address.local590 = and i64 %address.local589, 24, !dbg !19
  %address.local591 = or disjoint i64 %address.local588, %address.local590, !dbg !19
  %address.local592 = getelementptr [2 x i8], ptr addrspace(1) %1194, i64 %address.local591, !dbg !19
  %1198 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1199 = ptrtoint ptr addrspace(1) %address.local592 to i64, !dbg !19
  %1200 = sub i64 %1199, %1198, !dbg !19
  %dma.buffer.offset1100 = trunc i64 %1200 to i32, !dbg !19
  %1201 = trunc i64 %1198 to i32, !dbg !19
  %1202 = insertelement <4 x i32> poison, i32 %1201, i64 0, !dbg !19
  %1203 = lshr i64 %1198, 32, !dbg !19
  %1204 = trunc i64 %1203 to i32, !dbg !19
  %1205 = insertelement <4 x i32> %1202, i32 %1204, i64 1, !dbg !19
  %1206 = insertelement <4 x i32> %1205, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1101 = insertelement <4 x i32> %1206, i32 131072, i64 3, !dbg !19
  %1207 = ptrtoint ptr addrspace(3) %1197 to i32, !dbg !19
  %dma.buffer.lds1102 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1207), !dbg !19
  %1208 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1102, i32 %dma.buffer.offset1100, <4 x i32> %dma.buffer.rsrc1101) #10, !dbg !19
  %address.local593 = or disjoint i64 %address.local547, 256, !dbg !12
  %address.local594 = shl nuw nsw i64 %address.local593, 3, !dbg !12
  %address.local595 = and i64 %address.local594, 3584, !dbg !12
  %address.local596 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local595, !dbg !12
  %address.local597 = ptrtoint ptr addrspace(3) %address.local596 to i64, !dbg !12
  %address.local598 = trunc i64 %address.local597 to i32, !dbg !12
  %1209 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local598), !dbg !19
  %1210 = zext i32 %1209 to i64, !dbg !19
  %1211 = inttoptr i64 %1210 to ptr addrspace(3), !dbg !19
  %address.local599 = shl nuw nsw i64 %address.local593, 5, !dbg !19
  %address.local600 = and i64 %address.local599, 16256, !dbg !19
  %address.local601 = shl nuw nsw i64 %address.local547, 3, !dbg !19
  %address.local602 = and i64 %address.local601, 24, !dbg !19
  %address.local603 = or disjoint i64 %address.local600, %address.local602, !dbg !19
  %address.local604 = getelementptr [2 x i8], ptr addrspace(1) %1194, i64 %address.local603, !dbg !19
  %1212 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1213 = ptrtoint ptr addrspace(1) %address.local604 to i64, !dbg !19
  %1214 = sub i64 %1213, %1212, !dbg !19
  %dma.buffer.offset1097 = trunc i64 %1214 to i32, !dbg !19
  %1215 = trunc i64 %1212 to i32, !dbg !19
  %1216 = insertelement <4 x i32> poison, i32 %1215, i64 0, !dbg !19
  %1217 = lshr i64 %1212, 32, !dbg !19
  %1218 = trunc i64 %1217 to i32, !dbg !19
  %1219 = insertelement <4 x i32> %1216, i32 %1218, i64 1, !dbg !19
  %1220 = insertelement <4 x i32> %1219, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1098 = insertelement <4 x i32> %1220, i32 131072, i64 3, !dbg !19
  %1221 = ptrtoint ptr addrspace(3) %1211 to i32, !dbg !19
  %dma.buffer.lds1099 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1221), !dbg !19
  %1222 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1099, i32 %dma.buffer.offset1097, <4 x i32> %dma.buffer.rsrc1098) #10, !dbg !19
  %1223 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1224 = extractvalue [2 x <8 x float>] %47, 0, !dbg !32
  %1225 = shufflevector <8 x float> %1224, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1223, <4 x float> %1225) #9, !dbg !15
  %1227 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1227, <4 x float> %1226) #9, !dbg !15
  %1229 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1229, <4 x float> %1228) #9, !dbg !15
  %1231 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1231, <4 x float> %1230) #9, !dbg !15
  %1233 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1233, <4 x float> %1232) #9, !dbg !15
  %1235 = shufflevector <8 x half> %1184, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1235, <4 x float> %1234) #9, !dbg !15
  %1237 = shufflevector <8 x half> %1186, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1237, <4 x float> %1236) #9, !dbg !15
  %1239 = shufflevector <8 x half> %1188, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1239, <4 x float> %1238) #9, !dbg !15
  %1241 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1242 = shufflevector <8 x float> %1224, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1241, <4 x float> %1242) #9, !dbg !15
  %1244 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1244, <4 x float> %1243) #9, !dbg !15
  %1246 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1246, <4 x float> %1245) #9, !dbg !15
  %1248 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1248, <4 x float> %1247) #9, !dbg !15
  %1250 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1250, <4 x float> %1249) #9, !dbg !15
  %1252 = shufflevector <8 x half> %1184, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1252, <4 x float> %1251) #9, !dbg !15
  %1254 = shufflevector <8 x half> %1186, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1254, <4 x float> %1253) #9, !dbg !15
  %1256 = shufflevector <8 x half> %1188, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1256, <4 x float> %1255) #9, !dbg !15
  %1258 = extractvalue [2 x <8 x float>] %47, 1, !dbg !32
  %1259 = shufflevector <8 x float> %1258, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1223, <4 x float> %1259) #9, !dbg !15
  %1261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1227, <4 x float> %1260) #9, !dbg !15
  %1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1229, <4 x float> %1261) #9, !dbg !15
  %1263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1231, <4 x float> %1262) #9, !dbg !15
  %1264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1233, <4 x float> %1263) #9, !dbg !15
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1235, <4 x float> %1264) #9, !dbg !15
  %1266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1237, <4 x float> %1265) #9, !dbg !15
  %1267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1239, <4 x float> %1266) #9, !dbg !15
  %1268 = shufflevector <8 x float> %1258, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1241, <4 x float> %1268) #9, !dbg !15
  %1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1244, <4 x float> %1269) #9, !dbg !15
  %1271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1246, <4 x float> %1270) #9, !dbg !15
  %1272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1248, <4 x float> %1271) #9, !dbg !15
  %1273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1250, <4 x float> %1272) #9, !dbg !15
  %1274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1252, <4 x float> %1273) #9, !dbg !15
  %1275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1254, <4 x float> %1274) #9, !dbg !15
  %1276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1256, <4 x float> %1275) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1277 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local546, i32 0) #0, !dbg !15
  %1278 = bitcast <4 x i32> %1277 to <8 x half>, !dbg !15
  %1279 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local559, i32 0) #0, !dbg !15
  %1280 = bitcast <4 x i32> %1279 to <8 x half>, !dbg !15
  %1281 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local563, i32 0) #0, !dbg !15
  %1282 = bitcast <4 x i32> %1281 to <8 x half>, !dbg !15
  %1283 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local567, i32 0) #0, !dbg !15
  %1284 = bitcast <4 x i32> %1283 to <8 x half>, !dbg !15
  %1285 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local569, i32 0) #0, !dbg !15
  %1286 = bitcast <4 x i32> %1285 to <8 x half>, !dbg !15
  %1287 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local573, i32 0) #0, !dbg !15
  %1288 = bitcast <4 x i32> %1287 to <8 x half>, !dbg !15
  %1289 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local577, i32 0) #0, !dbg !15
  %1290 = bitcast <4 x i32> %1289 to <8 x half>, !dbg !15
  %1291 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local581, i32 0) #0, !dbg !15
  %1292 = bitcast <4 x i32> %1291 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1293 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1192, !dbg !19
  %1294 = getelementptr i8, ptr addrspace(1) %1293, i64 128, !dbg !19
  %1295 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local586), !dbg !19
  %1296 = zext i32 %1295 to i64, !dbg !19
  %1297 = inttoptr i64 %1296 to ptr addrspace(3), !dbg !19
  %address.local605 = getelementptr [2 x i8], ptr addrspace(1) %1294, i64 %address.local591, !dbg !19
  %1298 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1299 = ptrtoint ptr addrspace(1) %address.local605 to i64, !dbg !19
  %1300 = sub i64 %1299, %1298, !dbg !19
  %dma.buffer.offset1094 = trunc i64 %1300 to i32, !dbg !19
  %1301 = trunc i64 %1298 to i32, !dbg !19
  %1302 = insertelement <4 x i32> poison, i32 %1301, i64 0, !dbg !19
  %1303 = lshr i64 %1298, 32, !dbg !19
  %1304 = trunc i64 %1303 to i32, !dbg !19
  %1305 = insertelement <4 x i32> %1302, i32 %1304, i64 1, !dbg !19
  %1306 = insertelement <4 x i32> %1305, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1095 = insertelement <4 x i32> %1306, i32 131072, i64 3, !dbg !19
  %1307 = ptrtoint ptr addrspace(3) %1297 to i32, !dbg !19
  %dma.buffer.lds1096 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1307), !dbg !19
  %1308 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1096, i32 %dma.buffer.offset1094, <4 x i32> %dma.buffer.rsrc1095) #10, !dbg !19
  %1309 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local598), !dbg !19
  %1310 = zext i32 %1309 to i64, !dbg !19
  %1311 = inttoptr i64 %1310 to ptr addrspace(3), !dbg !19
  %address.local606 = getelementptr [2 x i8], ptr addrspace(1) %1294, i64 %address.local603, !dbg !19
  %1312 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1313 = ptrtoint ptr addrspace(1) %address.local606 to i64, !dbg !19
  %1314 = sub i64 %1313, %1312, !dbg !19
  %dma.buffer.offset1091 = trunc i64 %1314 to i32, !dbg !19
  %1315 = trunc i64 %1312 to i32, !dbg !19
  %1316 = insertelement <4 x i32> poison, i32 %1315, i64 0, !dbg !19
  %1317 = lshr i64 %1312, 32, !dbg !19
  %1318 = trunc i64 %1317 to i32, !dbg !19
  %1319 = insertelement <4 x i32> %1316, i32 %1318, i64 1, !dbg !19
  %1320 = insertelement <4 x i32> %1319, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1092 = insertelement <4 x i32> %1320, i32 131072, i64 3, !dbg !19
  %1321 = ptrtoint ptr addrspace(3) %1311 to i32, !dbg !19
  %dma.buffer.lds1093 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1321), !dbg !19
  %1322 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1093, i32 %dma.buffer.offset1091, <4 x i32> %dma.buffer.rsrc1092) #10, !dbg !19
  %1323 = shufflevector <8 x half> %1278, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1324 = extractvalue [2 x <8 x float>] %49, 0, !dbg !32
  %1325 = shufflevector <8 x float> %1324, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1323, <4 x float> %1325) #9, !dbg !15
  %1327 = shufflevector <8 x half> %1280, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1327, <4 x float> %1326) #9, !dbg !15
  %1329 = shufflevector <8 x half> %1282, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1329, <4 x float> %1328) #9, !dbg !15
  %1331 = shufflevector <8 x half> %1284, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1331, <4 x float> %1330) #9, !dbg !15
  %1333 = shufflevector <8 x half> %1286, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1333, <4 x float> %1332) #9, !dbg !15
  %1335 = shufflevector <8 x half> %1288, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1335, <4 x float> %1334) #9, !dbg !15
  %1337 = shufflevector <8 x half> %1290, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1337, <4 x float> %1336) #9, !dbg !15
  %1339 = shufflevector <8 x half> %1292, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1339, <4 x float> %1338) #9, !dbg !15
  %1341 = shufflevector <8 x half> %1278, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1342 = shufflevector <8 x float> %1324, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1341, <4 x float> %1342) #9, !dbg !15
  %1344 = shufflevector <8 x half> %1280, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1344, <4 x float> %1343) #9, !dbg !15
  %1346 = shufflevector <8 x half> %1282, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1346, <4 x float> %1345) #9, !dbg !15
  %1348 = shufflevector <8 x half> %1284, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1348, <4 x float> %1347) #9, !dbg !15
  %1350 = shufflevector <8 x half> %1286, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1350, <4 x float> %1349) #9, !dbg !15
  %1352 = shufflevector <8 x half> %1288, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1352, <4 x float> %1351) #9, !dbg !15
  %1354 = shufflevector <8 x half> %1290, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1354, <4 x float> %1353) #9, !dbg !15
  %1356 = shufflevector <8 x half> %1292, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1356, <4 x float> %1355) #9, !dbg !15
  %1358 = extractvalue [2 x <8 x float>] %49, 1, !dbg !32
  %1359 = shufflevector <8 x float> %1358, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1323, <4 x float> %1359) #9, !dbg !15
  %1361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1327, <4 x float> %1360) #9, !dbg !15
  %1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1329, <4 x float> %1361) #9, !dbg !15
  %1363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1331, <4 x float> %1362) #9, !dbg !15
  %1364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1333, <4 x float> %1363) #9, !dbg !15
  %1365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1335, <4 x float> %1364) #9, !dbg !15
  %1366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1337, <4 x float> %1365) #9, !dbg !15
  %1367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1339, <4 x float> %1366) #9, !dbg !15
  %1368 = shufflevector <8 x float> %1358, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1341, <4 x float> %1368) #9, !dbg !15
  %1370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1344, <4 x float> %1369) #9, !dbg !15
  %1371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1346, <4 x float> %1370) #9, !dbg !15
  %1372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1348, <4 x float> %1371) #9, !dbg !15
  %1373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1350, <4 x float> %1372) #9, !dbg !15
  %1374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1352, <4 x float> %1373) #9, !dbg !15
  %1375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1354, <4 x float> %1374) #9, !dbg !15
  %1376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1356, <4 x float> %1375) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1377 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local546, i32 0) #0, !dbg !15
  %1378 = bitcast <4 x i32> %1377 to <8 x half>, !dbg !15
  %1379 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local559, i32 0) #0, !dbg !15
  %1380 = bitcast <4 x i32> %1379 to <8 x half>, !dbg !15
  %1381 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local563, i32 0) #0, !dbg !15
  %1382 = bitcast <4 x i32> %1381 to <8 x half>, !dbg !15
  %1383 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local567, i32 0) #0, !dbg !15
  %1384 = bitcast <4 x i32> %1383 to <8 x half>, !dbg !15
  %1385 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local569, i32 0) #0, !dbg !15
  %1386 = bitcast <4 x i32> %1385 to <8 x half>, !dbg !15
  %1387 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local573, i32 0) #0, !dbg !15
  %1388 = bitcast <4 x i32> %1387 to <8 x half>, !dbg !15
  %1389 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local577, i32 0) #0, !dbg !15
  %1390 = bitcast <4 x i32> %1389 to <8 x half>, !dbg !15
  %1391 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local581, i32 0) #0, !dbg !15
  %1392 = bitcast <4 x i32> %1391 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1393 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1192, !dbg !19
  %1394 = getelementptr i8, ptr addrspace(1) %1393, i64 192, !dbg !19
  %1395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local586), !dbg !19
  %1396 = zext i32 %1395 to i64, !dbg !19
  %1397 = inttoptr i64 %1396 to ptr addrspace(3), !dbg !19
  %address.local607 = getelementptr [2 x i8], ptr addrspace(1) %1394, i64 %address.local591, !dbg !19
  %1398 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1399 = ptrtoint ptr addrspace(1) %address.local607 to i64, !dbg !19
  %1400 = sub i64 %1399, %1398, !dbg !19
  %dma.buffer.offset1088 = trunc i64 %1400 to i32, !dbg !19
  %1401 = trunc i64 %1398 to i32, !dbg !19
  %1402 = insertelement <4 x i32> poison, i32 %1401, i64 0, !dbg !19
  %1403 = lshr i64 %1398, 32, !dbg !19
  %1404 = trunc i64 %1403 to i32, !dbg !19
  %1405 = insertelement <4 x i32> %1402, i32 %1404, i64 1, !dbg !19
  %1406 = insertelement <4 x i32> %1405, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1089 = insertelement <4 x i32> %1406, i32 131072, i64 3, !dbg !19
  %1407 = ptrtoint ptr addrspace(3) %1397 to i32, !dbg !19
  %dma.buffer.lds1090 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1407), !dbg !19
  %1408 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1090, i32 %dma.buffer.offset1088, <4 x i32> %dma.buffer.rsrc1089) #10, !dbg !19
  %1409 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local598), !dbg !19
  %1410 = zext i32 %1409 to i64, !dbg !19
  %1411 = inttoptr i64 %1410 to ptr addrspace(3), !dbg !19
  %address.local608 = getelementptr [2 x i8], ptr addrspace(1) %1394, i64 %address.local603, !dbg !19
  %1412 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1413 = ptrtoint ptr addrspace(1) %address.local608 to i64, !dbg !19
  %1414 = sub i64 %1413, %1412, !dbg !19
  %dma.buffer.offset = trunc i64 %1414 to i32, !dbg !19
  %1415 = trunc i64 %1412 to i32, !dbg !19
  %1416 = insertelement <4 x i32> poison, i32 %1415, i64 0, !dbg !19
  %1417 = lshr i64 %1412, 32, !dbg !19
  %1418 = trunc i64 %1417 to i32, !dbg !19
  %1419 = insertelement <4 x i32> %1416, i32 %1418, i64 1, !dbg !19
  %1420 = insertelement <4 x i32> %1419, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc = insertelement <4 x i32> %1420, i32 131072, i64 3, !dbg !19
  %1421 = ptrtoint ptr addrspace(3) %1411 to i32, !dbg !19
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1421), !dbg !19
  %1422 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !19
  %1423 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1424 = extractvalue [2 x <8 x float>] %50, 0, !dbg !32
  %1425 = shufflevector <8 x float> %1424, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1423, <4 x float> %1425) #9, !dbg !15
  %1427 = shufflevector <8 x half> %1380, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1427, <4 x float> %1426) #9, !dbg !15
  %1429 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1429, <4 x float> %1428) #9, !dbg !15
  %1431 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1431, <4 x float> %1430) #9, !dbg !15
  %1433 = shufflevector <8 x half> %1386, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1433, <4 x float> %1432) #9, !dbg !15
  %1435 = shufflevector <8 x half> %1388, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1435, <4 x float> %1434) #9, !dbg !15
  %1437 = shufflevector <8 x half> %1390, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1437, <4 x float> %1436) #9, !dbg !15
  %1439 = shufflevector <8 x half> %1392, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1439, <4 x float> %1438) #9, !dbg !15
  %1441 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1442 = shufflevector <8 x float> %1424, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1441, <4 x float> %1442) #9, !dbg !15
  %1444 = shufflevector <8 x half> %1380, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1444, <4 x float> %1443) #9, !dbg !15
  %1446 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1446, <4 x float> %1445) #9, !dbg !15
  %1448 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1448, <4 x float> %1447) #9, !dbg !15
  %1450 = shufflevector <8 x half> %1386, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1450, <4 x float> %1449) #9, !dbg !15
  %1452 = shufflevector <8 x half> %1388, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1452, <4 x float> %1451) #9, !dbg !15
  %1454 = shufflevector <8 x half> %1390, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1454, <4 x float> %1453) #9, !dbg !15
  %1456 = shufflevector <8 x half> %1392, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1456, <4 x float> %1455) #9, !dbg !15
  %1458 = extractvalue [2 x <8 x float>] %50, 1, !dbg !32
  %1459 = shufflevector <8 x float> %1458, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1423, <4 x float> %1459) #9, !dbg !15
  %1461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1427, <4 x float> %1460) #9, !dbg !15
  %1462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1429, <4 x float> %1461) #9, !dbg !15
  %1463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1431, <4 x float> %1462) #9, !dbg !15
  %1464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1433, <4 x float> %1463) #9, !dbg !15
  %1465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1435, <4 x float> %1464) #9, !dbg !15
  %1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1437, <4 x float> %1465) #9, !dbg !15
  %1467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1439, <4 x float> %1466) #9, !dbg !15
  %1468 = shufflevector <8 x float> %1458, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1441, <4 x float> %1468) #9, !dbg !15
  %1470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1444, <4 x float> %1469) #9, !dbg !15
  %1471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1446, <4 x float> %1470) #9, !dbg !15
  %1472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1448, <4 x float> %1471) #9, !dbg !15
  %1473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1450, <4 x float> %1472) #9, !dbg !15
  %1474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1452, <4 x float> %1473) #9, !dbg !15
  %1475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1454, <4 x float> %1474) #9, !dbg !15
  %1476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1456, <4 x float> %1475) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1477 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local546, i32 0) #0, !dbg !15
  %1478 = bitcast <4 x i32> %1477 to <8 x half>, !dbg !15
  %1479 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local559, i32 0) #0, !dbg !15
  %1480 = bitcast <4 x i32> %1479 to <8 x half>, !dbg !15
  %1481 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local563, i32 0) #0, !dbg !15
  %1482 = bitcast <4 x i32> %1481 to <8 x half>, !dbg !15
  %1483 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local567, i32 0) #0, !dbg !15
  %1484 = bitcast <4 x i32> %1483 to <8 x half>, !dbg !15
  %1485 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local569, i32 0) #0, !dbg !15
  %1486 = bitcast <4 x i32> %1485 to <8 x half>, !dbg !15
  %1487 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local573, i32 0) #0, !dbg !15
  %1488 = bitcast <4 x i32> %1487 to <8 x half>, !dbg !15
  %1489 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local577, i32 0) #0, !dbg !15
  %1490 = bitcast <4 x i32> %1489 to <8 x half>, !dbg !15
  %1491 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local581, i32 0) #0, !dbg !15
  %1492 = bitcast <4 x i32> %1491 to <8 x half>, !dbg !15
  %1493 = shufflevector <8 x half> %1478, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1494 = extractvalue [2 x <8 x float>] %51, 0, !dbg !32
  %1495 = shufflevector <8 x float> %1494, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1493, <4 x float> %1495) #9, !dbg !15
  %1497 = shufflevector <8 x half> %1480, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1497, <4 x float> %1496) #9, !dbg !15
  %1499 = shufflevector <8 x half> %1482, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1499, <4 x float> %1498) #9, !dbg !15
  %1501 = shufflevector <8 x half> %1484, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1501, <4 x float> %1500) #9, !dbg !15
  %1503 = shufflevector <8 x half> %1486, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1503, <4 x float> %1502) #9, !dbg !15
  %1505 = shufflevector <8 x half> %1488, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1505, <4 x float> %1504) #9, !dbg !15
  %1507 = shufflevector <8 x half> %1490, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1507, <4 x float> %1506) #9, !dbg !15
  %1509 = shufflevector <8 x half> %1492, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1509, <4 x float> %1508) #9, !dbg !15
  %1511 = shufflevector <8 x half> %1478, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1512 = shufflevector <8 x float> %1494, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1511, <4 x float> %1512) #9, !dbg !15
  %1514 = shufflevector <8 x half> %1480, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1514, <4 x float> %1513) #9, !dbg !15
  %1516 = shufflevector <8 x half> %1482, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1516, <4 x float> %1515) #9, !dbg !15
  %1518 = shufflevector <8 x half> %1484, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1518, <4 x float> %1517) #9, !dbg !15
  %1520 = shufflevector <8 x half> %1486, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1520, <4 x float> %1519) #9, !dbg !15
  %1522 = shufflevector <8 x half> %1488, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1522, <4 x float> %1521) #9, !dbg !15
  %1524 = shufflevector <8 x half> %1490, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1524, <4 x float> %1523) #9, !dbg !15
  %1526 = shufflevector <8 x half> %1492, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1526, <4 x float> %1525) #9, !dbg !15
  %1528 = extractvalue [2 x <8 x float>] %51, 1, !dbg !32
  %1529 = shufflevector <8 x float> %1528, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !32
  %1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1493, <4 x float> %1529) #9, !dbg !15
  %1531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1497, <4 x float> %1530) #9, !dbg !15
  %1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1499, <4 x float> %1531) #9, !dbg !15
  %1533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1501, <4 x float> %1532) #9, !dbg !15
  %1534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1503, <4 x float> %1533) #9, !dbg !15
  %1535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1505, <4 x float> %1534) #9, !dbg !15
  %1536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1507, <4 x float> %1535) #9, !dbg !15
  %1537 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1509, <4 x float> %1536) #9, !dbg !15
  %1538 = shufflevector <8 x float> %1528, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !32
  %1539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1511, <4 x float> %1538) #9, !dbg !15
  %1540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1514, <4 x float> %1539) #9, !dbg !15
  %1541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1516, <4 x float> %1540) #9, !dbg !15
  %1542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1518, <4 x float> %1541) #9, !dbg !15
  %1543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1520, <4 x float> %1542) #9, !dbg !15
  %1544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1522, <4 x float> %1543) #9, !dbg !15
  %1545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1524, <4 x float> %1544) #9, !dbg !15
  %1546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1526, <4 x float> %1545) #9, !dbg !15
  %1547 = shufflevector <4 x float> %1240, <4 x float> %1257, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1548 = insertvalue [2 x <8 x float>] poison, <8 x float> %1547, 0, !dbg !16
  %1549 = shufflevector <4 x float> %1267, <4 x float> %1276, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1550 = insertvalue [2 x <8 x float>] %1548, <8 x float> %1549, 1, !dbg !16
  %1551 = shufflevector <4 x float> %1340, <4 x float> %1357, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1552 = insertvalue [2 x <8 x float>] poison, <8 x float> %1551, 0, !dbg !16
  %1553 = shufflevector <4 x float> %1367, <4 x float> %1376, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1554 = insertvalue [2 x <8 x float>] %1552, <8 x float> %1553, 1, !dbg !16
  %1555 = shufflevector <4 x float> %1440, <4 x float> %1457, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1556 = insertvalue [2 x <8 x float>] poison, <8 x float> %1555, 0, !dbg !16
  %1557 = shufflevector <4 x float> %1467, <4 x float> %1476, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1558 = insertvalue [2 x <8 x float>] %1556, <8 x float> %1557, 1, !dbg !16
  %1559 = shufflevector <4 x float> %1510, <4 x float> %1527, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1560 = insertvalue [2 x <8 x float>] poison, <8 x float> %1559, 0, !dbg !16
  %1561 = shufflevector <4 x float> %1537, <4 x float> %1546, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1562 = insertvalue [2 x <8 x float>] %1560, <8 x float> %1561, 1, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1563 = add nuw nsw i64 %46, 1, !dbg !16
  br label %45, !dbg !16

1564:                                             ; preds = %45
  %1565 = and i32 %26, 16, !dbg !16
  %.not2 = icmp eq i32 %1565, 0, !dbg !16
  %1566 = and i32 %26, 32, !dbg !16
  %.not3 = icmp eq i32 %1566, 0, !dbg !16
  %1567 = extractvalue [2 x <8 x float>] %51, 0, !dbg !16
  %1568 = extractelement <8 x float> %1567, i64 0, !dbg !16
  %1569 = extractelement <8 x float> %1567, i64 1, !dbg !16
  %1570 = extractelement <8 x float> %1567, i64 2, !dbg !16
  %1571 = extractelement <8 x float> %1567, i64 3, !dbg !16
  %1572 = select i1 %.not2, float %1569, float %1568, !dbg !16
  %1573 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %1574 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1573), !dbg !16
  %1575 = add nuw nsw i32 %1574, 64, !dbg !16
  %1576 = xor i32 %1574, 16, !dbg !16
  %1577 = and i32 %1575, 192, !dbg !16
  %1578 = icmp samesign ult i32 %1576, %1577, !dbg !16
  %1579 = select i1 %1578, i32 %1576, i32 %1574, !dbg !16
  %1580 = shl nuw nsw i32 %1579, 2, !dbg !16
  %1581 = bitcast float %1572 to i32, !dbg !16
  %1582 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1581), !dbg !16
  %1583 = bitcast i32 %1582 to float, !dbg !16
  %1584 = select i1 %.not2, float %1568, float %1583, !dbg !16
  %1585 = select i1 %.not2, float %1583, float %1569, !dbg !16
  %1586 = select i1 %.not2, float %1571, float %1570, !dbg !16
  %1587 = bitcast float %1586 to i32, !dbg !16
  %1588 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1587), !dbg !16
  %1589 = bitcast i32 %1588 to float, !dbg !16
  %1590 = select i1 %.not2, float %1570, float %1589, !dbg !16
  %1591 = select i1 %.not2, float %1589, float %1571, !dbg !16
  %1592 = select i1 %.not3, float %1590, float %1584, !dbg !16
  %1593 = xor i32 %1574, 32, !dbg !16
  %1594 = icmp samesign ult i32 %1593, %1577, !dbg !16
  %1595 = select i1 %1594, i32 %1593, i32 %1574, !dbg !16
  %1596 = shl nuw nsw i32 %1595, 2, !dbg !16
  %1597 = bitcast float %1592 to i32, !dbg !16
  %1598 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1597), !dbg !16
  %1599 = bitcast i32 %1598 to float, !dbg !16
  %1600 = select i1 %.not3, float %1584, float %1599, !dbg !16
  %1601 = select i1 %.not3, float %1599, float %1590, !dbg !16
  %1602 = select i1 %.not3, float %1591, float %1585, !dbg !16
  %1603 = bitcast float %1602 to i32, !dbg !16
  %1604 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1603), !dbg !16
  %1605 = bitcast i32 %1604 to float, !dbg !16
  %1606 = select i1 %.not3, float %1585, float %1605, !dbg !16
  %1607 = select i1 %.not3, float %1605, float %1591, !dbg !16
  %1608 = extractvalue [2 x <8 x float>] %50, 0, !dbg !16
  %1609 = extractelement <8 x float> %1608, i64 0, !dbg !16
  %1610 = extractelement <8 x float> %1608, i64 1, !dbg !16
  %1611 = extractelement <8 x float> %1608, i64 2, !dbg !16
  %1612 = extractelement <8 x float> %1608, i64 3, !dbg !16
  %1613 = select i1 %.not2, float %1610, float %1609, !dbg !16
  %1614 = bitcast float %1613 to i32, !dbg !16
  %1615 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1614), !dbg !16
  %1616 = bitcast i32 %1615 to float, !dbg !16
  %1617 = select i1 %.not2, float %1609, float %1616, !dbg !16
  %1618 = select i1 %.not2, float %1616, float %1610, !dbg !16
  %1619 = select i1 %.not2, float %1612, float %1611, !dbg !16
  %1620 = bitcast float %1619 to i32, !dbg !16
  %1621 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1620), !dbg !16
  %1622 = bitcast i32 %1621 to float, !dbg !16
  %1623 = select i1 %.not2, float %1611, float %1622, !dbg !16
  %1624 = select i1 %.not2, float %1622, float %1612, !dbg !16
  %1625 = select i1 %.not3, float %1623, float %1617, !dbg !16
  %1626 = bitcast float %1625 to i32, !dbg !16
  %1627 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1626), !dbg !16
  %1628 = bitcast i32 %1627 to float, !dbg !16
  %1629 = select i1 %.not3, float %1617, float %1628, !dbg !16
  %1630 = select i1 %.not3, float %1628, float %1623, !dbg !16
  %1631 = select i1 %.not3, float %1624, float %1618, !dbg !16
  %1632 = bitcast float %1631 to i32, !dbg !16
  %1633 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1632), !dbg !16
  %1634 = bitcast i32 %1633 to float, !dbg !16
  %1635 = select i1 %.not3, float %1618, float %1634, !dbg !16
  %1636 = select i1 %.not3, float %1634, float %1624, !dbg !16
  %1637 = extractvalue [2 x <8 x float>] %49, 0, !dbg !16
  %1638 = extractelement <8 x float> %1637, i64 0, !dbg !16
  %1639 = extractelement <8 x float> %1637, i64 1, !dbg !16
  %1640 = extractelement <8 x float> %1637, i64 2, !dbg !16
  %1641 = extractelement <8 x float> %1637, i64 3, !dbg !16
  %1642 = select i1 %.not2, float %1639, float %1638, !dbg !16
  %1643 = bitcast float %1642 to i32, !dbg !16
  %1644 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1643), !dbg !16
  %1645 = bitcast i32 %1644 to float, !dbg !16
  %1646 = select i1 %.not2, float %1638, float %1645, !dbg !16
  %1647 = select i1 %.not2, float %1645, float %1639, !dbg !16
  %1648 = select i1 %.not2, float %1641, float %1640, !dbg !16
  %1649 = bitcast float %1648 to i32, !dbg !16
  %1650 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1649), !dbg !16
  %1651 = bitcast i32 %1650 to float, !dbg !16
  %1652 = select i1 %.not2, float %1640, float %1651, !dbg !16
  %1653 = select i1 %.not2, float %1651, float %1641, !dbg !16
  %1654 = select i1 %.not3, float %1652, float %1646, !dbg !16
  %1655 = bitcast float %1654 to i32, !dbg !16
  %1656 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1655), !dbg !16
  %1657 = bitcast i32 %1656 to float, !dbg !16
  %1658 = select i1 %.not3, float %1646, float %1657, !dbg !16
  %1659 = select i1 %.not3, float %1657, float %1652, !dbg !16
  %1660 = select i1 %.not3, float %1653, float %1647, !dbg !16
  %1661 = bitcast float %1660 to i32, !dbg !16
  %1662 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1661), !dbg !16
  %1663 = bitcast i32 %1662 to float, !dbg !16
  %1664 = select i1 %.not3, float %1647, float %1663, !dbg !16
  %1665 = select i1 %.not3, float %1663, float %1653, !dbg !16
  %1666 = extractvalue [2 x <8 x float>] %47, 0, !dbg !16
  %1667 = extractelement <8 x float> %1666, i64 0, !dbg !16
  %1668 = extractelement <8 x float> %1666, i64 1, !dbg !16
  %1669 = extractelement <8 x float> %1666, i64 2, !dbg !16
  %1670 = extractelement <8 x float> %1666, i64 3, !dbg !16
  %1671 = select i1 %.not2, float %1668, float %1667, !dbg !16
  %1672 = bitcast float %1671 to i32, !dbg !16
  %1673 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1672), !dbg !16
  %1674 = bitcast i32 %1673 to float, !dbg !16
  %1675 = select i1 %.not2, float %1667, float %1674, !dbg !16
  %1676 = select i1 %.not2, float %1674, float %1668, !dbg !16
  %1677 = select i1 %.not2, float %1670, float %1669, !dbg !16
  %1678 = bitcast float %1677 to i32, !dbg !16
  %1679 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1678), !dbg !16
  %1680 = bitcast i32 %1679 to float, !dbg !16
  %1681 = select i1 %.not2, float %1669, float %1680, !dbg !16
  %1682 = select i1 %.not2, float %1680, float %1670, !dbg !16
  %1683 = select i1 %.not3, float %1681, float %1675, !dbg !16
  %1684 = bitcast float %1683 to i32, !dbg !16
  %1685 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1684), !dbg !16
  %1686 = select i1 %.not3, float %1682, float %1676, !dbg !16
  %1687 = bitcast float %1686 to i32, !dbg !16
  %1688 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1687), !dbg !16
  %1689 = extractvalue [2 x <1 x float>] %48, 0, !dbg !20
  %1690 = shufflevector <1 x float> %1689, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1691 = extractelement <8 x float> %1567, i64 4, !dbg !16
  %1692 = extractelement <8 x float> %1567, i64 5, !dbg !16
  %1693 = extractelement <8 x float> %1567, i64 6, !dbg !16
  %1694 = extractelement <8 x float> %1567, i64 7, !dbg !16
  %1695 = select i1 %.not2, float %1692, float %1691, !dbg !16
  %1696 = bitcast float %1695 to i32, !dbg !16
  %1697 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1696), !dbg !16
  %1698 = bitcast i32 %1697 to float, !dbg !16
  %1699 = select i1 %.not2, float %1691, float %1698, !dbg !16
  %1700 = select i1 %.not2, float %1698, float %1692, !dbg !16
  %1701 = select i1 %.not2, float %1694, float %1693, !dbg !16
  %1702 = bitcast float %1701 to i32, !dbg !16
  %1703 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1702), !dbg !16
  %1704 = bitcast i32 %1703 to float, !dbg !16
  %1705 = select i1 %.not2, float %1693, float %1704, !dbg !16
  %1706 = select i1 %.not2, float %1704, float %1694, !dbg !16
  %1707 = select i1 %.not3, float %1705, float %1699, !dbg !16
  %1708 = bitcast float %1707 to i32, !dbg !16
  %1709 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1708), !dbg !16
  %1710 = bitcast i32 %1709 to float, !dbg !16
  %1711 = select i1 %.not3, float %1699, float %1710, !dbg !16
  %1712 = select i1 %.not3, float %1710, float %1705, !dbg !16
  %1713 = select i1 %.not3, float %1706, float %1700, !dbg !16
  %1714 = bitcast float %1713 to i32, !dbg !16
  %1715 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1714), !dbg !16
  %1716 = bitcast i32 %1715 to float, !dbg !16
  %1717 = select i1 %.not3, float %1700, float %1716, !dbg !16
  %1718 = select i1 %.not3, float %1716, float %1706, !dbg !16
  %1719 = extractelement <8 x float> %1608, i64 4, !dbg !16
  %1720 = extractelement <8 x float> %1608, i64 5, !dbg !16
  %1721 = extractelement <8 x float> %1608, i64 6, !dbg !16
  %1722 = extractelement <8 x float> %1608, i64 7, !dbg !16
  %1723 = select i1 %.not2, float %1720, float %1719, !dbg !16
  %1724 = bitcast float %1723 to i32, !dbg !16
  %1725 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1724), !dbg !16
  %1726 = bitcast i32 %1725 to float, !dbg !16
  %1727 = select i1 %.not2, float %1719, float %1726, !dbg !16
  %1728 = select i1 %.not2, float %1726, float %1720, !dbg !16
  %1729 = select i1 %.not2, float %1722, float %1721, !dbg !16
  %1730 = bitcast float %1729 to i32, !dbg !16
  %1731 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1730), !dbg !16
  %1732 = bitcast i32 %1731 to float, !dbg !16
  %1733 = select i1 %.not2, float %1721, float %1732, !dbg !16
  %1734 = select i1 %.not2, float %1732, float %1722, !dbg !16
  %1735 = select i1 %.not3, float %1733, float %1727, !dbg !16
  %1736 = bitcast float %1735 to i32, !dbg !16
  %1737 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1736), !dbg !16
  %1738 = bitcast i32 %1737 to float, !dbg !16
  %1739 = select i1 %.not3, float %1727, float %1738, !dbg !16
  %1740 = select i1 %.not3, float %1738, float %1733, !dbg !16
  %1741 = select i1 %.not3, float %1734, float %1728, !dbg !16
  %1742 = bitcast float %1741 to i32, !dbg !16
  %1743 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1742), !dbg !16
  %1744 = bitcast i32 %1743 to float, !dbg !16
  %1745 = select i1 %.not3, float %1728, float %1744, !dbg !16
  %1746 = select i1 %.not3, float %1744, float %1734, !dbg !16
  %1747 = extractelement <8 x float> %1637, i64 4, !dbg !16
  %1748 = extractelement <8 x float> %1637, i64 5, !dbg !16
  %1749 = extractelement <8 x float> %1637, i64 6, !dbg !16
  %1750 = extractelement <8 x float> %1637, i64 7, !dbg !16
  %1751 = select i1 %.not2, float %1748, float %1747, !dbg !16
  %1752 = bitcast float %1751 to i32, !dbg !16
  %1753 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1752), !dbg !16
  %1754 = bitcast i32 %1753 to float, !dbg !16
  %1755 = select i1 %.not2, float %1747, float %1754, !dbg !16
  %1756 = select i1 %.not2, float %1754, float %1748, !dbg !16
  %1757 = select i1 %.not2, float %1750, float %1749, !dbg !16
  %1758 = bitcast float %1757 to i32, !dbg !16
  %1759 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1758), !dbg !16
  %1760 = bitcast i32 %1759 to float, !dbg !16
  %1761 = select i1 %.not2, float %1749, float %1760, !dbg !16
  %1762 = select i1 %.not2, float %1760, float %1750, !dbg !16
  %1763 = select i1 %.not3, float %1761, float %1755, !dbg !16
  %1764 = bitcast float %1763 to i32, !dbg !16
  %1765 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1764), !dbg !16
  %1766 = bitcast i32 %1765 to float, !dbg !16
  %1767 = select i1 %.not3, float %1755, float %1766, !dbg !16
  %1768 = select i1 %.not3, float %1766, float %1761, !dbg !16
  %1769 = select i1 %.not3, float %1762, float %1756, !dbg !16
  %1770 = bitcast float %1769 to i32, !dbg !16
  %1771 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1770), !dbg !16
  %1772 = bitcast i32 %1771 to float, !dbg !16
  %1773 = select i1 %.not3, float %1756, float %1772, !dbg !16
  %1774 = select i1 %.not3, float %1772, float %1762, !dbg !16
  %1775 = extractelement <8 x float> %1666, i64 4, !dbg !16
  %1776 = extractelement <8 x float> %1666, i64 5, !dbg !16
  %1777 = extractelement <8 x float> %1666, i64 6, !dbg !16
  %1778 = extractelement <8 x float> %1666, i64 7, !dbg !16
  %1779 = select i1 %.not2, float %1776, float %1775, !dbg !16
  %1780 = bitcast float %1779 to i32, !dbg !16
  %1781 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1780), !dbg !16
  %1782 = bitcast i32 %1781 to float, !dbg !16
  %1783 = select i1 %.not2, float %1775, float %1782, !dbg !16
  %1784 = select i1 %.not2, float %1782, float %1776, !dbg !16
  %1785 = select i1 %.not2, float %1778, float %1777, !dbg !16
  %1786 = bitcast float %1785 to i32, !dbg !16
  %1787 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1786), !dbg !16
  %1788 = bitcast i32 %1787 to float, !dbg !16
  %1789 = select i1 %.not2, float %1777, float %1788, !dbg !16
  %1790 = select i1 %.not2, float %1788, float %1778, !dbg !16
  %1791 = select i1 %.not3, float %1789, float %1783, !dbg !16
  %1792 = bitcast float %1791 to i32, !dbg !16
  %1793 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1792), !dbg !16
  %1794 = select i1 %.not3, float %1790, float %1784, !dbg !16
  %1795 = bitcast float %1794 to i32, !dbg !16
  %1796 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1795), !dbg !16
  %1797 = extractvalue [2 x <8 x float>] %51, 1, !dbg !16
  %1798 = extractelement <8 x float> %1797, i64 0, !dbg !16
  %1799 = extractelement <8 x float> %1797, i64 1, !dbg !16
  %1800 = extractelement <8 x float> %1797, i64 2, !dbg !16
  %1801 = extractelement <8 x float> %1797, i64 3, !dbg !16
  %1802 = select i1 %.not2, float %1799, float %1798, !dbg !16
  %1803 = bitcast float %1802 to i32, !dbg !16
  %1804 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1803), !dbg !16
  %1805 = bitcast i32 %1804 to float, !dbg !16
  %1806 = select i1 %.not2, float %1798, float %1805, !dbg !16
  %1807 = select i1 %.not2, float %1805, float %1799, !dbg !16
  %1808 = select i1 %.not2, float %1801, float %1800, !dbg !16
  %1809 = bitcast float %1808 to i32, !dbg !16
  %1810 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1809), !dbg !16
  %1811 = bitcast i32 %1810 to float, !dbg !16
  %1812 = select i1 %.not2, float %1800, float %1811, !dbg !16
  %1813 = select i1 %.not2, float %1811, float %1801, !dbg !16
  %1814 = select i1 %.not3, float %1812, float %1806, !dbg !16
  %1815 = bitcast float %1814 to i32, !dbg !16
  %1816 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1815), !dbg !16
  %1817 = bitcast i32 %1816 to float, !dbg !16
  %1818 = select i1 %.not3, float %1806, float %1817, !dbg !16
  %1819 = select i1 %.not3, float %1817, float %1812, !dbg !16
  %1820 = select i1 %.not3, float %1813, float %1807, !dbg !16
  %1821 = bitcast float %1820 to i32, !dbg !16
  %1822 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1821), !dbg !16
  %1823 = bitcast i32 %1822 to float, !dbg !16
  %1824 = select i1 %.not3, float %1807, float %1823, !dbg !16
  %1825 = select i1 %.not3, float %1823, float %1813, !dbg !16
  %1826 = extractvalue [2 x <8 x float>] %50, 1, !dbg !16
  %1827 = extractelement <8 x float> %1826, i64 0, !dbg !16
  %1828 = extractelement <8 x float> %1826, i64 1, !dbg !16
  %1829 = extractelement <8 x float> %1826, i64 2, !dbg !16
  %1830 = extractelement <8 x float> %1826, i64 3, !dbg !16
  %1831 = select i1 %.not2, float %1828, float %1827, !dbg !16
  %1832 = bitcast float %1831 to i32, !dbg !16
  %1833 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1832), !dbg !16
  %1834 = bitcast i32 %1833 to float, !dbg !16
  %1835 = select i1 %.not2, float %1827, float %1834, !dbg !16
  %1836 = select i1 %.not2, float %1834, float %1828, !dbg !16
  %1837 = select i1 %.not2, float %1830, float %1829, !dbg !16
  %1838 = bitcast float %1837 to i32, !dbg !16
  %1839 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1838), !dbg !16
  %1840 = bitcast i32 %1839 to float, !dbg !16
  %1841 = select i1 %.not2, float %1829, float %1840, !dbg !16
  %1842 = select i1 %.not2, float %1840, float %1830, !dbg !16
  %1843 = select i1 %.not3, float %1841, float %1835, !dbg !16
  %1844 = bitcast float %1843 to i32, !dbg !16
  %1845 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1844), !dbg !16
  %1846 = bitcast i32 %1845 to float, !dbg !16
  %1847 = select i1 %.not3, float %1835, float %1846, !dbg !16
  %1848 = select i1 %.not3, float %1846, float %1841, !dbg !16
  %1849 = select i1 %.not3, float %1842, float %1836, !dbg !16
  %1850 = bitcast float %1849 to i32, !dbg !16
  %1851 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1850), !dbg !16
  %1852 = bitcast i32 %1851 to float, !dbg !16
  %1853 = select i1 %.not3, float %1836, float %1852, !dbg !16
  %1854 = select i1 %.not3, float %1852, float %1842, !dbg !16
  %1855 = extractvalue [2 x <8 x float>] %49, 1, !dbg !16
  %1856 = extractelement <8 x float> %1855, i64 0, !dbg !16
  %1857 = extractelement <8 x float> %1855, i64 1, !dbg !16
  %1858 = extractelement <8 x float> %1855, i64 2, !dbg !16
  %1859 = extractelement <8 x float> %1855, i64 3, !dbg !16
  %1860 = select i1 %.not2, float %1857, float %1856, !dbg !16
  %1861 = bitcast float %1860 to i32, !dbg !16
  %1862 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1861), !dbg !16
  %1863 = bitcast i32 %1862 to float, !dbg !16
  %1864 = select i1 %.not2, float %1856, float %1863, !dbg !16
  %1865 = select i1 %.not2, float %1863, float %1857, !dbg !16
  %1866 = select i1 %.not2, float %1859, float %1858, !dbg !16
  %1867 = bitcast float %1866 to i32, !dbg !16
  %1868 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1867), !dbg !16
  %1869 = bitcast i32 %1868 to float, !dbg !16
  %1870 = select i1 %.not2, float %1858, float %1869, !dbg !16
  %1871 = select i1 %.not2, float %1869, float %1859, !dbg !16
  %1872 = select i1 %.not3, float %1870, float %1864, !dbg !16
  %1873 = bitcast float %1872 to i32, !dbg !16
  %1874 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1873), !dbg !16
  %1875 = bitcast i32 %1874 to float, !dbg !16
  %1876 = select i1 %.not3, float %1864, float %1875, !dbg !16
  %1877 = select i1 %.not3, float %1875, float %1870, !dbg !16
  %1878 = select i1 %.not3, float %1871, float %1865, !dbg !16
  %1879 = bitcast float %1878 to i32, !dbg !16
  %1880 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1879), !dbg !16
  %1881 = bitcast i32 %1880 to float, !dbg !16
  %1882 = select i1 %.not3, float %1865, float %1881, !dbg !16
  %1883 = select i1 %.not3, float %1881, float %1871, !dbg !16
  %1884 = extractvalue [2 x <8 x float>] %47, 1, !dbg !16
  %1885 = extractelement <8 x float> %1884, i64 0, !dbg !16
  %1886 = extractelement <8 x float> %1884, i64 1, !dbg !16
  %1887 = extractelement <8 x float> %1884, i64 2, !dbg !16
  %1888 = extractelement <8 x float> %1884, i64 3, !dbg !16
  %1889 = select i1 %.not2, float %1886, float %1885, !dbg !16
  %1890 = bitcast float %1889 to i32, !dbg !16
  %1891 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1890), !dbg !16
  %1892 = bitcast i32 %1891 to float, !dbg !16
  %1893 = select i1 %.not2, float %1885, float %1892, !dbg !16
  %1894 = select i1 %.not2, float %1892, float %1886, !dbg !16
  %1895 = select i1 %.not2, float %1888, float %1887, !dbg !16
  %1896 = bitcast float %1895 to i32, !dbg !16
  %1897 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1896), !dbg !16
  %1898 = bitcast i32 %1897 to float, !dbg !16
  %1899 = select i1 %.not2, float %1887, float %1898, !dbg !16
  %1900 = select i1 %.not2, float %1898, float %1888, !dbg !16
  %1901 = select i1 %.not3, float %1899, float %1893, !dbg !16
  %1902 = bitcast float %1901 to i32, !dbg !16
  %1903 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1902), !dbg !16
  %1904 = select i1 %.not3, float %1900, float %1894, !dbg !16
  %1905 = bitcast float %1904 to i32, !dbg !16
  %1906 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1905), !dbg !16
  %1907 = extractvalue [2 x <1 x float>] %48, 1, !dbg !20
  %1908 = shufflevector <1 x float> %1907, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1909 = extractelement <8 x float> %1797, i64 4, !dbg !16
  %1910 = extractelement <8 x float> %1797, i64 5, !dbg !16
  %1911 = extractelement <8 x float> %1797, i64 6, !dbg !16
  %1912 = extractelement <8 x float> %1797, i64 7, !dbg !16
  %1913 = select i1 %.not2, float %1910, float %1909, !dbg !16
  %1914 = bitcast float %1913 to i32, !dbg !16
  %1915 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1914), !dbg !16
  %1916 = bitcast i32 %1915 to float, !dbg !16
  %1917 = select i1 %.not2, float %1909, float %1916, !dbg !16
  %1918 = select i1 %.not2, float %1916, float %1910, !dbg !16
  %1919 = select i1 %.not2, float %1912, float %1911, !dbg !16
  %1920 = bitcast float %1919 to i32, !dbg !16
  %1921 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1920), !dbg !16
  %1922 = bitcast i32 %1921 to float, !dbg !16
  %1923 = select i1 %.not2, float %1911, float %1922, !dbg !16
  %1924 = select i1 %.not2, float %1922, float %1912, !dbg !16
  %1925 = select i1 %.not3, float %1923, float %1917, !dbg !16
  %1926 = bitcast float %1925 to i32, !dbg !16
  %1927 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1926), !dbg !16
  %1928 = bitcast i32 %1927 to float, !dbg !16
  %1929 = select i1 %.not3, float %1917, float %1928, !dbg !16
  %1930 = select i1 %.not3, float %1928, float %1923, !dbg !16
  %1931 = select i1 %.not3, float %1924, float %1918, !dbg !16
  %1932 = bitcast float %1931 to i32, !dbg !16
  %1933 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1932), !dbg !16
  %1934 = bitcast i32 %1933 to float, !dbg !16
  %1935 = select i1 %.not3, float %1918, float %1934, !dbg !16
  %1936 = select i1 %.not3, float %1934, float %1924, !dbg !16
  %1937 = extractelement <8 x float> %1826, i64 4, !dbg !16
  %1938 = extractelement <8 x float> %1826, i64 5, !dbg !16
  %1939 = extractelement <8 x float> %1826, i64 6, !dbg !16
  %1940 = extractelement <8 x float> %1826, i64 7, !dbg !16
  %1941 = select i1 %.not2, float %1938, float %1937, !dbg !16
  %1942 = bitcast float %1941 to i32, !dbg !16
  %1943 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1942), !dbg !16
  %1944 = bitcast i32 %1943 to float, !dbg !16
  %1945 = select i1 %.not2, float %1937, float %1944, !dbg !16
  %1946 = select i1 %.not2, float %1944, float %1938, !dbg !16
  %1947 = select i1 %.not2, float %1940, float %1939, !dbg !16
  %1948 = bitcast float %1947 to i32, !dbg !16
  %1949 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1948), !dbg !16
  %1950 = bitcast i32 %1949 to float, !dbg !16
  %1951 = select i1 %.not2, float %1939, float %1950, !dbg !16
  %1952 = select i1 %.not2, float %1950, float %1940, !dbg !16
  %1953 = select i1 %.not3, float %1951, float %1945, !dbg !16
  %1954 = bitcast float %1953 to i32, !dbg !16
  %1955 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1954), !dbg !16
  %1956 = bitcast i32 %1955 to float, !dbg !16
  %1957 = select i1 %.not3, float %1945, float %1956, !dbg !16
  %1958 = select i1 %.not3, float %1956, float %1951, !dbg !16
  %1959 = select i1 %.not3, float %1952, float %1946, !dbg !16
  %1960 = bitcast float %1959 to i32, !dbg !16
  %1961 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1960), !dbg !16
  %1962 = bitcast i32 %1961 to float, !dbg !16
  %1963 = select i1 %.not3, float %1946, float %1962, !dbg !16
  %1964 = select i1 %.not3, float %1962, float %1952, !dbg !16
  %1965 = extractelement <8 x float> %1855, i64 4, !dbg !16
  %1966 = extractelement <8 x float> %1855, i64 5, !dbg !16
  %1967 = extractelement <8 x float> %1855, i64 6, !dbg !16
  %1968 = extractelement <8 x float> %1855, i64 7, !dbg !16
  %1969 = select i1 %.not2, float %1966, float %1965, !dbg !16
  %1970 = bitcast float %1969 to i32, !dbg !16
  %1971 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1970), !dbg !16
  %1972 = bitcast i32 %1971 to float, !dbg !16
  %1973 = select i1 %.not2, float %1965, float %1972, !dbg !16
  %1974 = select i1 %.not2, float %1972, float %1966, !dbg !16
  %1975 = select i1 %.not2, float %1968, float %1967, !dbg !16
  %1976 = bitcast float %1975 to i32, !dbg !16
  %1977 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1976), !dbg !16
  %1978 = bitcast i32 %1977 to float, !dbg !16
  %1979 = select i1 %.not2, float %1967, float %1978, !dbg !16
  %1980 = select i1 %.not2, float %1978, float %1968, !dbg !16
  %1981 = select i1 %.not3, float %1979, float %1973, !dbg !16
  %1982 = bitcast float %1981 to i32, !dbg !16
  %1983 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1982), !dbg !16
  %1984 = bitcast i32 %1983 to float, !dbg !16
  %1985 = select i1 %.not3, float %1973, float %1984, !dbg !16
  %1986 = select i1 %.not3, float %1984, float %1979, !dbg !16
  %1987 = select i1 %.not3, float %1980, float %1974, !dbg !16
  %1988 = bitcast float %1987 to i32, !dbg !16
  %1989 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1988), !dbg !16
  %1990 = bitcast i32 %1989 to float, !dbg !16
  %1991 = select i1 %.not3, float %1974, float %1990, !dbg !16
  %1992 = select i1 %.not3, float %1990, float %1980, !dbg !16
  %1993 = extractelement <8 x float> %1884, i64 4, !dbg !16
  %1994 = extractelement <8 x float> %1884, i64 5, !dbg !16
  %1995 = extractelement <8 x float> %1884, i64 6, !dbg !16
  %1996 = extractelement <8 x float> %1884, i64 7, !dbg !16
  %1997 = select i1 %.not2, float %1994, float %1993, !dbg !16
  %1998 = bitcast float %1997 to i32, !dbg !16
  %1999 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %1998), !dbg !16
  %2000 = bitcast i32 %1999 to float, !dbg !16
  %2001 = select i1 %.not2, float %1993, float %2000, !dbg !16
  %2002 = select i1 %.not2, float %2000, float %1994, !dbg !16
  %2003 = select i1 %.not2, float %1996, float %1995, !dbg !16
  %2004 = bitcast float %2003 to i32, !dbg !16
  %2005 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1580, i32 %2004), !dbg !16
  %2006 = bitcast i32 %2005 to float, !dbg !16
  %2007 = select i1 %.not2, float %1995, float %2006, !dbg !16
  %2008 = select i1 %.not2, float %2006, float %1996, !dbg !16
  %2009 = select i1 %.not3, float %2007, float %2001, !dbg !16
  %2010 = bitcast float %2009 to i32, !dbg !16
  %2011 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %2010), !dbg !16
  %2012 = select i1 %.not3, float %2008, float %2002, !dbg !16
  %2013 = bitcast float %2012 to i32, !dbg !16
  %2014 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %2013), !dbg !16
  br i1 true, label %2015, label %2077, !dbg !22

2015:                                             ; preds = %1564
  %2016 = bitcast i32 %2011 to float, !dbg !16
  %2017 = select i1 %.not3, float %2001, float %2016, !dbg !16
  %2018 = insertelement <4 x float> poison, float %2017, i64 0, !dbg !20
  %2019 = bitcast i32 %2014 to float, !dbg !16
  %2020 = select i1 %.not3, float %2002, float %2019, !dbg !16
  %2021 = insertelement <4 x float> %2018, float %2020, i64 1, !dbg !20
  %2022 = select i1 %.not3, float %2016, float %2007, !dbg !16
  %2023 = insertelement <4 x float> %2021, float %2022, i64 2, !dbg !20
  %2024 = select i1 %.not3, float %2019, float %2008, !dbg !16
  %2025 = insertelement <4 x float> %2023, float %2024, i64 3, !dbg !20
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1908, !dbg !20
  %normalized = fmul <4 x float> %2025, %row.reciprocal, !dbg !20
  %2026 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !21
  %2027 = extractelement <4 x half> %2026, i64 3, !dbg !21
  %2028 = extractelement <4 x half> %2026, i64 2, !dbg !21
  %2029 = extractelement <4 x half> %2026, i64 1, !dbg !21
  %2030 = extractelement <4 x half> %2026, i64 0, !dbg !21
  %2031 = bitcast i32 %1903 to float, !dbg !16
  %2032 = select i1 %.not3, float %1893, float %2031, !dbg !16
  %2033 = insertelement <4 x float> poison, float %2032, i64 0, !dbg !20
  %2034 = bitcast i32 %1906 to float, !dbg !16
  %2035 = select i1 %.not3, float %1894, float %2034, !dbg !16
  %2036 = insertelement <4 x float> %2033, float %2035, i64 1, !dbg !20
  %2037 = select i1 %.not3, float %2031, float %1899, !dbg !16
  %2038 = insertelement <4 x float> %2036, float %2037, i64 2, !dbg !20
  %2039 = select i1 %.not3, float %2034, float %1900, !dbg !16
  %2040 = insertelement <4 x float> %2038, float %2039, i64 3, !dbg !20
  %normalized71 = fmul <4 x float> %2040, %row.reciprocal, !dbg !20
  %2041 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !21
  %2042 = extractelement <4 x half> %2041, i64 3, !dbg !21
  %2043 = extractelement <4 x half> %2041, i64 2, !dbg !21
  %2044 = extractelement <4 x half> %2041, i64 1, !dbg !21
  %2045 = extractelement <4 x half> %2041, i64 0, !dbg !21
  %2046 = bitcast i32 %1793 to float, !dbg !16
  %2047 = select i1 %.not3, float %1783, float %2046, !dbg !16
  %2048 = insertelement <4 x float> poison, float %2047, i64 0, !dbg !20
  %2049 = bitcast i32 %1796 to float, !dbg !16
  %2050 = select i1 %.not3, float %1784, float %2049, !dbg !16
  %2051 = insertelement <4 x float> %2048, float %2050, i64 1, !dbg !20
  %2052 = select i1 %.not3, float %2046, float %1789, !dbg !16
  %2053 = insertelement <4 x float> %2051, float %2052, i64 2, !dbg !20
  %2054 = select i1 %.not3, float %2049, float %1790, !dbg !16
  %2055 = insertelement <4 x float> %2053, float %2054, i64 3, !dbg !20
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1690, !dbg !20
  %normalized73 = fmul <4 x float> %2055, %row.reciprocal72, !dbg !20
  %2056 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !21
  %2057 = extractelement <4 x half> %2056, i64 3, !dbg !21
  %2058 = extractelement <4 x half> %2056, i64 2, !dbg !21
  %2059 = extractelement <4 x half> %2056, i64 1, !dbg !21
  %2060 = extractelement <4 x half> %2056, i64 0, !dbg !21
  %2061 = bitcast i32 %1685 to float, !dbg !16
  %2062 = select i1 %.not3, float %1675, float %2061, !dbg !16
  %2063 = insertelement <4 x float> poison, float %2062, i64 0, !dbg !20
  %2064 = bitcast i32 %1688 to float, !dbg !16
  %2065 = select i1 %.not3, float %1676, float %2064, !dbg !16
  %2066 = insertelement <4 x float> %2063, float %2065, i64 1, !dbg !20
  %2067 = select i1 %.not3, float %2061, float %1681, !dbg !16
  %2068 = insertelement <4 x float> %2066, float %2067, i64 2, !dbg !20
  %2069 = select i1 %.not3, float %2064, float %1682, !dbg !16
  %2070 = insertelement <4 x float> %2068, float %2069, i64 3, !dbg !20
  %normalized74 = fmul <4 x float> %2070, %row.reciprocal72, !dbg !20
  %2071 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !21
  %2072 = extractelement <4 x half> %2071, i64 3, !dbg !21
  %2073 = extractelement <4 x half> %2071, i64 2, !dbg !21
  %2074 = extractelement <4 x half> %2071, i64 1, !dbg !21
  %2075 = extractelement <4 x half> %2071, i64 0, !dbg !21
  %2076 = shl nuw nsw i64 %17, 19, !dbg !22
  %address.thread609 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !22, !range !35, !frisk.register_lifetimes !1
  %address.local610 = zext nneg i32 %address.thread609 to i64, !dbg !13
  %address.local611 = add nsw i64 %18, %address.local610, !dbg !22
  %address.local612 = lshr i64 %address.local610, 6, !dbg !13
  %address.local613 = shl nuw nsw i64 %address.local612, 4, !dbg !22
  %address.local614 = add nsw i64 %address.local611, %address.local613, !dbg !22
  %address.local615 = and i64 %address.local610, 240, !dbg !22
  %address.local616 = sub nsw i64 %address.local614, %address.local615, !dbg !22
  %address.local617 = add nsw i64 %address.local616, %19, !dbg !22
  %address.local618 = shl nuw nsw i64 %address.local617, 7, !dbg !22
  %address.local619 = add nuw nsw i64 %2076, %address.local618, !dbg !22
  %address.local620 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !22
  %address.local621 = lshr i64 %address.local610, 2, !dbg !13
  %address.local622 = and i64 %address.local621, 12, !dbg !13
  %address.local623 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local622, !dbg !22
  store half %2075, ptr addrspace(1) %address.local623, align 2, !dbg !22
  %address.local624 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !22
  %address.local625 = or disjoint i64 %address.local622, 1, !dbg !22
  %address.local626 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local624, i64 %address.local625, !dbg !22
  store half %2074, ptr addrspace(1) %address.local626, align 2, !dbg !22
  %address.local627 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !22
  %address.local628 = or disjoint i64 %address.local622, 2, !dbg !22
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local627, i64 %address.local628, !dbg !22
  store half %2073, ptr addrspace(1) %address.local629, align 2, !dbg !22
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !22
  %address.local631 = or disjoint i64 %address.local622, 3, !dbg !22
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local630, i64 %address.local631, !dbg !22
  store half %2072, ptr addrspace(1) %address.local632, align 2, !dbg !22
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !22
  %address.local634 = or disjoint i64 %address.local622, 16, !dbg !22
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local633, i64 %address.local634, !dbg !22
  store half %2060, ptr addrspace(1) %address.local635, align 2, !dbg !22
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !22
  %address.local637 = or disjoint i64 %address.local622, 17, !dbg !22
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local636, i64 %address.local637, !dbg !22
  store half %2059, ptr addrspace(1) %address.local638, align 2, !dbg !22
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !22
  %address.local640 = or disjoint i64 %address.local622, 18, !dbg !22
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local639, i64 %address.local640, !dbg !22
  store half %2058, ptr addrspace(1) %address.local641, align 2, !dbg !22
  %address.local642 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !22
  %address.local643 = or disjoint i64 %address.local622, 19, !dbg !22
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local642, i64 %address.local643, !dbg !22
  store half %2057, ptr addrspace(1) %address.local644, align 2, !dbg !22
  %address.local645 = shl nsw i64 %address.local617, 7, !dbg !22
  %address.local646 = add i64 %address.local645, 8192, !dbg !22
  %address.local647 = add nuw nsw i64 %2076, %address.local646, !dbg !22
  %address.local648 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !22
  %address.local649 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local622, !dbg !22
  store half %2045, ptr addrspace(1) %address.local649, align 2, !dbg !22
  %address.local650 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !22
  %address.local651 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local650, i64 %address.local625, !dbg !22
  store half %2044, ptr addrspace(1) %address.local651, align 2, !dbg !22
  %address.local652 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !22
  %address.local653 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local652, i64 %address.local628, !dbg !22
  store half %2043, ptr addrspace(1) %address.local653, align 2, !dbg !22
  %address.local654 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !22
  %address.local655 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local654, i64 %address.local631, !dbg !22
  store half %2042, ptr addrspace(1) %address.local655, align 2, !dbg !22
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !22
  %address.local657 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local656, i64 %address.local634, !dbg !22
  store half %2030, ptr addrspace(1) %address.local657, align 2, !dbg !22
  %address.local658 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !22
  %address.local659 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local658, i64 %address.local637, !dbg !22
  store half %2029, ptr addrspace(1) %address.local659, align 2, !dbg !22
  %address.local660 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !22
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local660, i64 %address.local640, !dbg !22
  store half %2028, ptr addrspace(1) %address.local661, align 2, !dbg !22
  %address.local662 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !22
  %address.local663 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local662, i64 %address.local643, !dbg !22
  store half %2027, ptr addrspace(1) %address.local663, align 2, !dbg !22
  br label %2077, !dbg !22

2077:                                             ; preds = %2015, %1564
  br i1 true, label %2078, label %2116, !dbg !22

2078:                                             ; preds = %2077
  %2079 = insertelement <4 x float> poison, float %1985, i64 0, !dbg !20
  %2080 = insertelement <4 x float> %2079, float %1991, i64 1, !dbg !20
  %2081 = insertelement <4 x float> %2080, float %1986, i64 2, !dbg !20
  %2082 = insertelement <4 x float> %2081, float %1992, i64 3, !dbg !20
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1908, !dbg !20
  %normalized76 = fmul <4 x float> %2082, %row.reciprocal75, !dbg !20
  %2083 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !21
  %2084 = extractelement <4 x half> %2083, i64 3, !dbg !21
  %2085 = extractelement <4 x half> %2083, i64 2, !dbg !21
  %2086 = extractelement <4 x half> %2083, i64 1, !dbg !21
  %2087 = extractelement <4 x half> %2083, i64 0, !dbg !21
  %2088 = insertelement <4 x float> poison, float %1876, i64 0, !dbg !20
  %2089 = insertelement <4 x float> %2088, float %1882, i64 1, !dbg !20
  %2090 = insertelement <4 x float> %2089, float %1877, i64 2, !dbg !20
  %2091 = insertelement <4 x float> %2090, float %1883, i64 3, !dbg !20
  %normalized77 = fmul <4 x float> %2091, %row.reciprocal75, !dbg !20
  %2092 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !21
  %2093 = extractelement <4 x half> %2092, i64 3, !dbg !21
  %2094 = extractelement <4 x half> %2092, i64 2, !dbg !21
  %2095 = extractelement <4 x half> %2092, i64 1, !dbg !21
  %2096 = extractelement <4 x half> %2092, i64 0, !dbg !21
  %2097 = insertelement <4 x float> poison, float %1767, i64 0, !dbg !20
  %2098 = insertelement <4 x float> %2097, float %1773, i64 1, !dbg !20
  %2099 = insertelement <4 x float> %2098, float %1768, i64 2, !dbg !20
  %2100 = insertelement <4 x float> %2099, float %1774, i64 3, !dbg !20
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1690, !dbg !20
  %normalized79 = fmul <4 x float> %2100, %row.reciprocal78, !dbg !20
  %2101 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !21
  %2102 = extractelement <4 x half> %2101, i64 3, !dbg !21
  %2103 = extractelement <4 x half> %2101, i64 2, !dbg !21
  %2104 = extractelement <4 x half> %2101, i64 1, !dbg !21
  %2105 = extractelement <4 x half> %2101, i64 0, !dbg !21
  %2106 = insertelement <4 x float> poison, float %1658, i64 0, !dbg !20
  %2107 = insertelement <4 x float> %2106, float %1664, i64 1, !dbg !20
  %2108 = insertelement <4 x float> %2107, float %1659, i64 2, !dbg !20
  %2109 = insertelement <4 x float> %2108, float %1665, i64 3, !dbg !20
  %normalized80 = fmul <4 x float> %2109, %row.reciprocal78, !dbg !20
  %2110 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !21
  %2111 = extractelement <4 x half> %2110, i64 3, !dbg !21
  %2112 = extractelement <4 x half> %2110, i64 2, !dbg !21
  %2113 = extractelement <4 x half> %2110, i64 1, !dbg !21
  %2114 = extractelement <4 x half> %2110, i64 0, !dbg !21
  %2115 = shl nuw nsw i64 %23, 19, !dbg !22
  %address.thread664 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !22, !range !35, !frisk.register_lifetimes !1
  %address.local665 = zext nneg i32 %address.thread664 to i64, !dbg !13
  %address.local666 = add nsw i64 %18, %address.local665, !dbg !22
  %address.local667 = lshr i64 %address.local665, 6, !dbg !13
  %address.local668 = shl nuw nsw i64 %address.local667, 4, !dbg !22
  %address.local669 = add nsw i64 %address.local666, %address.local668, !dbg !22
  %address.local670 = and i64 %address.local665, 240, !dbg !22
  %address.local671 = sub nsw i64 %address.local669, %address.local670, !dbg !22
  %address.local672 = add nsw i64 %address.local671, %19, !dbg !22
  %address.local673 = shl nuw nsw i64 %address.local672, 7, !dbg !22
  %address.local674 = add nuw nsw i64 %2115, %address.local673, !dbg !22
  %address.local675 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !22
  %address.local676 = lshr i64 %address.local665, 2, !dbg !13
  %address.local677 = and i64 %address.local676, 12, !dbg !13
  %address.local678 = or disjoint i64 %address.local677, 32, !dbg !22
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local675, i64 %address.local678, !dbg !22
  store half %2114, ptr addrspace(1) %address.local679, align 2, !dbg !22
  %address.local680 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !22
  %address.local681 = or disjoint i64 %address.local677, 33, !dbg !22
  %address.local682 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local680, i64 %address.local681, !dbg !22
  store half %2113, ptr addrspace(1) %address.local682, align 2, !dbg !22
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !22
  %address.local684 = or disjoint i64 %address.local677, 34, !dbg !22
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local683, i64 %address.local684, !dbg !22
  store half %2112, ptr addrspace(1) %address.local685, align 2, !dbg !22
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !22
  %address.local687 = or disjoint i64 %address.local677, 35, !dbg !22
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local686, i64 %address.local687, !dbg !22
  store half %2111, ptr addrspace(1) %address.local688, align 2, !dbg !22
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !22
  %address.local690 = or disjoint i64 %address.local677, 48, !dbg !22
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local689, i64 %address.local690, !dbg !22
  store half %2105, ptr addrspace(1) %address.local691, align 2, !dbg !22
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !22
  %address.local693 = or disjoint i64 %address.local677, 49, !dbg !22
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local692, i64 %address.local693, !dbg !22
  store half %2104, ptr addrspace(1) %address.local694, align 2, !dbg !22
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !22
  %address.local696 = or disjoint i64 %address.local677, 50, !dbg !22
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local695, i64 %address.local696, !dbg !22
  store half %2103, ptr addrspace(1) %address.local697, align 2, !dbg !22
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !22
  %address.local699 = or i64 %address.local676, 51, !dbg !22
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local698, i64 %address.local699, !dbg !22
  store half %2102, ptr addrspace(1) %address.local700, align 2, !dbg !22
  %address.local701 = shl nsw i64 %address.local672, 7, !dbg !22
  %address.local702 = add i64 %address.local701, 8192, !dbg !22
  %address.local703 = add nuw nsw i64 %2115, %address.local702, !dbg !22
  %address.local704 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !22
  %address.local705 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local704, i64 %address.local678, !dbg !22
  store half %2096, ptr addrspace(1) %address.local705, align 2, !dbg !22
  %address.local706 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !22
  %address.local707 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local706, i64 %address.local681, !dbg !22
  store half %2095, ptr addrspace(1) %address.local707, align 2, !dbg !22
  %address.local708 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !22
  %address.local709 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local708, i64 %address.local684, !dbg !22
  store half %2094, ptr addrspace(1) %address.local709, align 2, !dbg !22
  %address.local710 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !22
  %address.local711 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local710, i64 %address.local687, !dbg !22
  store half %2093, ptr addrspace(1) %address.local711, align 2, !dbg !22
  %address.local712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !22
  %address.local713 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local712, i64 %address.local690, !dbg !22
  store half %2087, ptr addrspace(1) %address.local713, align 2, !dbg !22
  %address.local714 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !22
  %address.local715 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local714, i64 %address.local693, !dbg !22
  store half %2086, ptr addrspace(1) %address.local715, align 2, !dbg !22
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !22
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local716, i64 %address.local696, !dbg !22
  store half %2085, ptr addrspace(1) %address.local717, align 2, !dbg !22
  %address.local718 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !22
  %address.local719 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local718, i64 %address.local699, !dbg !22
  store half %2084, ptr addrspace(1) %address.local719, align 2, !dbg !22
  br label %2116, !dbg !22

2116:                                             ; preds = %2078, %2077
  br i1 true, label %2117, label %2155, !dbg !22

2117:                                             ; preds = %2116
  %2118 = insertelement <4 x float> poison, float %1957, i64 0, !dbg !20
  %2119 = insertelement <4 x float> %2118, float %1963, i64 1, !dbg !20
  %2120 = insertelement <4 x float> %2119, float %1958, i64 2, !dbg !20
  %2121 = insertelement <4 x float> %2120, float %1964, i64 3, !dbg !20
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1908, !dbg !20
  %normalized82 = fmul <4 x float> %2121, %row.reciprocal81, !dbg !20
  %2122 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !21
  %2123 = extractelement <4 x half> %2122, i64 3, !dbg !21
  %2124 = extractelement <4 x half> %2122, i64 2, !dbg !21
  %2125 = extractelement <4 x half> %2122, i64 1, !dbg !21
  %2126 = extractelement <4 x half> %2122, i64 0, !dbg !21
  %2127 = insertelement <4 x float> poison, float %1847, i64 0, !dbg !20
  %2128 = insertelement <4 x float> %2127, float %1853, i64 1, !dbg !20
  %2129 = insertelement <4 x float> %2128, float %1848, i64 2, !dbg !20
  %2130 = insertelement <4 x float> %2129, float %1854, i64 3, !dbg !20
  %normalized83 = fmul <4 x float> %2130, %row.reciprocal81, !dbg !20
  %2131 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !21
  %2132 = extractelement <4 x half> %2131, i64 3, !dbg !21
  %2133 = extractelement <4 x half> %2131, i64 2, !dbg !21
  %2134 = extractelement <4 x half> %2131, i64 1, !dbg !21
  %2135 = extractelement <4 x half> %2131, i64 0, !dbg !21
  %2136 = insertelement <4 x float> poison, float %1739, i64 0, !dbg !20
  %2137 = insertelement <4 x float> %2136, float %1745, i64 1, !dbg !20
  %2138 = insertelement <4 x float> %2137, float %1740, i64 2, !dbg !20
  %2139 = insertelement <4 x float> %2138, float %1746, i64 3, !dbg !20
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1690, !dbg !20
  %normalized85 = fmul <4 x float> %2139, %row.reciprocal84, !dbg !20
  %2140 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !21
  %2141 = extractelement <4 x half> %2140, i64 3, !dbg !21
  %2142 = extractelement <4 x half> %2140, i64 2, !dbg !21
  %2143 = extractelement <4 x half> %2140, i64 1, !dbg !21
  %2144 = extractelement <4 x half> %2140, i64 0, !dbg !21
  %2145 = insertelement <4 x float> poison, float %1629, i64 0, !dbg !20
  %2146 = insertelement <4 x float> %2145, float %1635, i64 1, !dbg !20
  %2147 = insertelement <4 x float> %2146, float %1630, i64 2, !dbg !20
  %2148 = insertelement <4 x float> %2147, float %1636, i64 3, !dbg !20
  %normalized86 = fmul <4 x float> %2148, %row.reciprocal84, !dbg !20
  %2149 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !21
  %2150 = extractelement <4 x half> %2149, i64 3, !dbg !21
  %2151 = extractelement <4 x half> %2149, i64 2, !dbg !21
  %2152 = extractelement <4 x half> %2149, i64 1, !dbg !21
  %2153 = extractelement <4 x half> %2149, i64 0, !dbg !21
  %2154 = shl nuw nsw i64 %23, 19, !dbg !22
  %address.thread720 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !22, !range !35, !frisk.register_lifetimes !1
  %address.local721 = zext nneg i32 %address.thread720 to i64, !dbg !13
  %address.local722 = add nsw i64 %18, %address.local721, !dbg !22
  %address.local723 = lshr i64 %address.local721, 6, !dbg !13
  %address.local724 = shl nuw nsw i64 %address.local723, 4, !dbg !22
  %address.local725 = add nsw i64 %address.local722, %address.local724, !dbg !22
  %address.local726 = and i64 %address.local721, 240, !dbg !22
  %address.local727 = sub nsw i64 %address.local725, %address.local726, !dbg !22
  %address.local728 = add nsw i64 %address.local727, %19, !dbg !22
  %address.local729 = shl nuw nsw i64 %address.local728, 7, !dbg !22
  %address.local730 = add nuw nsw i64 %2154, %address.local729, !dbg !22
  %address.local731 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !22
  %address.local732 = lshr i64 %address.local721, 2, !dbg !13
  %address.local733 = and i64 %address.local732, 12, !dbg !13
  %address.local734 = or disjoint i64 %address.local733, 64, !dbg !22
  %address.local735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local731, i64 %address.local734, !dbg !22
  store half %2153, ptr addrspace(1) %address.local735, align 2, !dbg !22
  %address.local736 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !22
  %address.local737 = or disjoint i64 %address.local733, 65, !dbg !22
  %address.local738 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local736, i64 %address.local737, !dbg !22
  store half %2152, ptr addrspace(1) %address.local738, align 2, !dbg !22
  %address.local739 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !22
  %address.local740 = or disjoint i64 %address.local733, 66, !dbg !22
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local739, i64 %address.local740, !dbg !22
  store half %2151, ptr addrspace(1) %address.local741, align 2, !dbg !22
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !22
  %address.local743 = or disjoint i64 %address.local733, 67, !dbg !22
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local742, i64 %address.local743, !dbg !22
  store half %2150, ptr addrspace(1) %address.local744, align 2, !dbg !22
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !22
  %address.local746 = or disjoint i64 %address.local733, 80, !dbg !22
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local745, i64 %address.local746, !dbg !22
  store half %2144, ptr addrspace(1) %address.local747, align 2, !dbg !22
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !22
  %address.local749 = or disjoint i64 %address.local733, 81, !dbg !22
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local748, i64 %address.local749, !dbg !22
  store half %2143, ptr addrspace(1) %address.local750, align 2, !dbg !22
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !22
  %address.local752 = or disjoint i64 %address.local733, 82, !dbg !22
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local751, i64 %address.local752, !dbg !22
  store half %2142, ptr addrspace(1) %address.local753, align 2, !dbg !22
  %address.local754 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !22
  %address.local755 = or disjoint i64 %address.local733, 83, !dbg !22
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local754, i64 %address.local755, !dbg !22
  store half %2141, ptr addrspace(1) %address.local756, align 2, !dbg !22
  %address.local757 = shl nsw i64 %address.local728, 7, !dbg !22
  %address.local758 = add i64 %address.local757, 8192, !dbg !22
  %address.local759 = add nuw nsw i64 %2154, %address.local758, !dbg !22
  %address.local760 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !22
  %address.local761 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local760, i64 %address.local734, !dbg !22
  store half %2135, ptr addrspace(1) %address.local761, align 2, !dbg !22
  %address.local762 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !22
  %address.local763 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local762, i64 %address.local737, !dbg !22
  store half %2134, ptr addrspace(1) %address.local763, align 2, !dbg !22
  %address.local764 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !22
  %address.local765 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local764, i64 %address.local740, !dbg !22
  store half %2133, ptr addrspace(1) %address.local765, align 2, !dbg !22
  %address.local766 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !22
  %address.local767 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local743, !dbg !22
  store half %2132, ptr addrspace(1) %address.local767, align 2, !dbg !22
  %address.local768 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !22
  %address.local769 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local768, i64 %address.local746, !dbg !22
  store half %2126, ptr addrspace(1) %address.local769, align 2, !dbg !22
  %address.local770 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !22
  %address.local771 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local770, i64 %address.local749, !dbg !22
  store half %2125, ptr addrspace(1) %address.local771, align 2, !dbg !22
  %address.local772 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !22
  %address.local773 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local772, i64 %address.local752, !dbg !22
  store half %2124, ptr addrspace(1) %address.local773, align 2, !dbg !22
  %address.local774 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !22
  %address.local775 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local774, i64 %address.local755, !dbg !22
  store half %2123, ptr addrspace(1) %address.local775, align 2, !dbg !22
  br label %2155, !dbg !22

2155:                                             ; preds = %2117, %2116
  br i1 true, label %2156, label %2194, !dbg !22

2156:                                             ; preds = %2155
  %2157 = insertelement <4 x float> poison, float %1929, i64 0, !dbg !20
  %2158 = insertelement <4 x float> %2157, float %1935, i64 1, !dbg !20
  %2159 = insertelement <4 x float> %2158, float %1930, i64 2, !dbg !20
  %2160 = insertelement <4 x float> %2159, float %1936, i64 3, !dbg !20
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1908, !dbg !20
  %normalized88 = fmul <4 x float> %2160, %row.reciprocal87, !dbg !20
  %2161 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !21
  %2162 = extractelement <4 x half> %2161, i64 3, !dbg !21
  %2163 = extractelement <4 x half> %2161, i64 2, !dbg !21
  %2164 = extractelement <4 x half> %2161, i64 1, !dbg !21
  %2165 = extractelement <4 x half> %2161, i64 0, !dbg !21
  %2166 = insertelement <4 x float> poison, float %1818, i64 0, !dbg !20
  %2167 = insertelement <4 x float> %2166, float %1824, i64 1, !dbg !20
  %2168 = insertelement <4 x float> %2167, float %1819, i64 2, !dbg !20
  %2169 = insertelement <4 x float> %2168, float %1825, i64 3, !dbg !20
  %normalized89 = fmul <4 x float> %2169, %row.reciprocal87, !dbg !20
  %2170 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !21
  %2171 = extractelement <4 x half> %2170, i64 3, !dbg !21
  %2172 = extractelement <4 x half> %2170, i64 2, !dbg !21
  %2173 = extractelement <4 x half> %2170, i64 1, !dbg !21
  %2174 = extractelement <4 x half> %2170, i64 0, !dbg !21
  %2175 = insertelement <4 x float> poison, float %1711, i64 0, !dbg !20
  %2176 = insertelement <4 x float> %2175, float %1717, i64 1, !dbg !20
  %2177 = insertelement <4 x float> %2176, float %1712, i64 2, !dbg !20
  %2178 = insertelement <4 x float> %2177, float %1718, i64 3, !dbg !20
  %row.reciprocal90 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1690, !dbg !20
  %normalized91 = fmul <4 x float> %2178, %row.reciprocal90, !dbg !20
  %2179 = fptrunc <4 x float> %normalized91 to <4 x half>, !dbg !21
  %2180 = extractelement <4 x half> %2179, i64 3, !dbg !21
  %2181 = extractelement <4 x half> %2179, i64 2, !dbg !21
  %2182 = extractelement <4 x half> %2179, i64 1, !dbg !21
  %2183 = extractelement <4 x half> %2179, i64 0, !dbg !21
  %2184 = insertelement <4 x float> poison, float %1600, i64 0, !dbg !20
  %2185 = insertelement <4 x float> %2184, float %1606, i64 1, !dbg !20
  %2186 = insertelement <4 x float> %2185, float %1601, i64 2, !dbg !20
  %2187 = insertelement <4 x float> %2186, float %1607, i64 3, !dbg !20
  %normalized92 = fmul <4 x float> %2187, %row.reciprocal90, !dbg !20
  %2188 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !21
  %2189 = extractelement <4 x half> %2188, i64 3, !dbg !21
  %2190 = extractelement <4 x half> %2188, i64 2, !dbg !21
  %2191 = extractelement <4 x half> %2188, i64 1, !dbg !21
  %2192 = extractelement <4 x half> %2188, i64 0, !dbg !21
  %2193 = shl nuw nsw i64 %23, 19, !dbg !22
  %address.thread776 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !22, !range !35, !frisk.register_lifetimes !1
  %address.local777 = zext nneg i32 %address.thread776 to i64, !dbg !13
  %address.local778 = add nsw i64 %18, %address.local777, !dbg !22
  %address.local779 = lshr i64 %address.local777, 6, !dbg !13
  %address.local780 = shl nuw nsw i64 %address.local779, 4, !dbg !22
  %address.local781 = add nsw i64 %address.local778, %address.local780, !dbg !22
  %address.local782 = and i64 %address.local777, 240, !dbg !22
  %address.local783 = sub nsw i64 %address.local781, %address.local782, !dbg !22
  %address.local784 = add nsw i64 %address.local783, %19, !dbg !22
  %address.local785 = shl nuw nsw i64 %address.local784, 7, !dbg !22
  %address.local786 = add nuw nsw i64 %2193, %address.local785, !dbg !22
  %address.local787 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !22
  %address.local788 = lshr i64 %address.local777, 2, !dbg !13
  %address.local789 = and i64 %address.local788, 12, !dbg !13
  %address.local790 = or disjoint i64 %address.local789, 96, !dbg !22
  %address.local791 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local787, i64 %address.local790, !dbg !22
  store half %2192, ptr addrspace(1) %address.local791, align 2, !dbg !22
  %address.local792 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !22
  %address.local793 = or disjoint i64 %address.local789, 97, !dbg !22
  %address.local794 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local792, i64 %address.local793, !dbg !22
  store half %2191, ptr addrspace(1) %address.local794, align 2, !dbg !22
  %address.local795 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !22
  %address.local796 = or disjoint i64 %address.local789, 98, !dbg !22
  %address.local797 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local795, i64 %address.local796, !dbg !22
  store half %2190, ptr addrspace(1) %address.local797, align 2, !dbg !22
  %address.local798 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !22
  %address.local799 = or disjoint i64 %address.local789, 99, !dbg !22
  %address.local800 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local798, i64 %address.local799, !dbg !22
  store half %2189, ptr addrspace(1) %address.local800, align 2, !dbg !22
  %address.local801 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !22
  %address.local802 = or disjoint i64 %address.local789, 112, !dbg !22
  %address.local803 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local801, i64 %address.local802, !dbg !22
  store half %2183, ptr addrspace(1) %address.local803, align 2, !dbg !22
  %address.local804 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !22
  %address.local805 = or disjoint i64 %address.local789, 113, !dbg !22
  %address.local806 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local804, i64 %address.local805, !dbg !22
  store half %2182, ptr addrspace(1) %address.local806, align 2, !dbg !22
  %address.local807 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !22
  %address.local808 = or disjoint i64 %address.local789, 114, !dbg !22
  %address.local809 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local807, i64 %address.local808, !dbg !22
  store half %2181, ptr addrspace(1) %address.local809, align 2, !dbg !22
  %address.local810 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !22
  %address.local811 = or i64 %address.local788, 115, !dbg !22
  %address.local812 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local810, i64 %address.local811, !dbg !22
  store half %2180, ptr addrspace(1) %address.local812, align 2, !dbg !22
  %address.local813 = shl nsw i64 %address.local784, 7, !dbg !22
  %address.local814 = add i64 %address.local813, 8192, !dbg !22
  %address.local815 = add nuw nsw i64 %2193, %address.local814, !dbg !22
  %address.local816 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !22
  %address.local817 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local816, i64 %address.local790, !dbg !22
  store half %2174, ptr addrspace(1) %address.local817, align 2, !dbg !22
  %address.local818 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !22
  %address.local819 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local818, i64 %address.local793, !dbg !22
  store half %2173, ptr addrspace(1) %address.local819, align 2, !dbg !22
  %address.local820 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !22
  %address.local821 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local820, i64 %address.local796, !dbg !22
  store half %2172, ptr addrspace(1) %address.local821, align 2, !dbg !22
  %address.local822 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !22
  %address.local823 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local822, i64 %address.local799, !dbg !22
  store half %2171, ptr addrspace(1) %address.local823, align 2, !dbg !22
  %address.local824 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !22
  %address.local825 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local824, i64 %address.local802, !dbg !22
  store half %2165, ptr addrspace(1) %address.local825, align 2, !dbg !22
  %address.local826 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !22
  %address.local827 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local826, i64 %address.local805, !dbg !22
  store half %2164, ptr addrspace(1) %address.local827, align 2, !dbg !22
  %address.local828 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !22
  %address.local829 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local828, i64 %address.local808, !dbg !22
  store half %2163, ptr addrspace(1) %address.local829, align 2, !dbg !22
  %address.local830 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !22
  %address.local831 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local830, i64 %address.local811, !dbg !22
  store half %2162, ptr addrspace(1) %address.local831, align 2, !dbg !22
  br label %2194, !dbg !22

2194:                                             ; preds = %2156, %2155
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
!9 = !DILocation(line: 46, column: 17, scope: !4)
!12 = !DILocation(line: 66, column: 22, scope: !4)
!13 = !DILocation(line: 68, column: 24, scope: !4)
!14 = !DILocation(line: 73, column: 19, scope: !4)
!15 = !DILocation(line: 93, column: 23, scope: !4)
!16 = !DILocation(line: 56, column: 24, scope: !4)
!17 = !DILocation(line: 67, column: 22, scope: !4)
!18 = !DILocation(line: 69, column: 26, scope: !4)
!19 = !DILocation(line: 92, column: 20, scope: !4)
!20 = !DILocation(line: 97, column: 23, scope: !4)
!21 = !DILocation(line: 98, column: 19, scope: !4)
!22 = !DILocation(line: 99, column: 9, scope: !4)
!23 = !DILocation(line: 59, column: 17, scope: !4)
!25 = !DILocation(line: 75, column: 24, scope: !4)
!26 = !DILocation(line: 76, column: 23, scope: !4)
!27 = !DILocation(line: 77, column: 26, scope: !4)
!28 = !DILocation(line: 84, column: 21, scope: !4)
!29 = !DILocation(line: 85, column: 22, scope: !4)
!30 = !DILocation(line: 86, column: 20, scope: !4)
!31 = !DILocation(line: 87, column: 16, scope: !4)
!32 = !DILocation(line: 94, column: 20, scope: !4)
!33 = !{i32 256, i32 1, i32 1}
!34 = !{i32 2, !"Debug Info Version", i32 3}
!35 = !{i32 0, i32 256}
