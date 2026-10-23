; ModuleID = '3rd/deepgengraph/test/test_input_spidD.mlir'
source_filename = "3rd/deepgengraph/test/test_input_spidD.mlir"

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
  %21 = lshr i64 %20, 4, !dbg !13
  %22 = or disjoint i64 %9, 127, !dbg !14
  %23 = add nsw i64 %9, 128, !dbg !14
  %24 = icmp uge i64 %23, %22, !dbg !14
  %25 = lshr i64 %20, 2, !dbg !8
  %26 = shl nuw nsw i64 %20, 10, !dbg !15
  %27 = shl nuw nsw i64 %25, 12, !dbg !15
  %28 = sub nsw i64 %26, %27, !dbg !15
  %29 = shl nuw nsw i64 %20, 1, !dbg !15
  %30 = and i64 %29, 120, !dbg !15
  %31 = or disjoint i64 %28, %30, !dbg !15
  %32 = lshr i64 %20, 2, !dbg !15
  %33 = and i64 %32, 48, !dbg !15
  %34 = mul nsw i64 %21, -16, !dbg !15
  br label %35, !dbg !16

35:                                               ; preds = %1093, %4
  %36 = phi i64 [ %1556, %1093 ], [ 0, %4 ], !dbg !16
  %37 = phi [2 x <16 x float>] [ %1543, %1093 ], [ zeroinitializer, %4 ], !dbg !16
  %38 = phi [2 x <1 x float>] [ %1547, %1093 ], [ zeroinitializer, %4 ], !dbg !16
  %39 = phi [2 x <16 x float>] [ %1555, %1093 ], [ zeroinitializer, %4 ], !dbg !16
  %.not = icmp sgt i64 %36, %6, !dbg !16
  br i1 %.not, label %1557, label %40, !dbg !16

40:                                               ; preds = %35
  %41 = shl nsw i64 %36, 7, !dbg !16
  %42 = lshr i64 %36, 5, !dbg !17
  %43 = shl i64 %42, 12, !dbg !17
  %44 = sub i64 %41, %43, !dbg !17
  br label %45, !dbg !22

45:                                               ; preds = %49, %40
  %46 = phi i64 [ %369, %49 ], [ 0, %40 ], !dbg !22
  %47 = phi [2 x <32 x float>] [ %368, %49 ], [ zeroinitializer, %40 ], !dbg !22
  %48 = icmp samesign ult i64 %46, 2, !dbg !22
  br i1 %48, label %49, label %370, !dbg !22

49:                                               ; preds = %45
  %50 = shl nuw nsw i64 %46, 6, !dbg !22
  %.idx = shl nuw nsw i64 %15, 20, !dbg !12
  %51 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %.idx69 = shl nsw i64 %18, 8, !dbg !12
  %52 = getelementptr i8, ptr addrspace(1) %51, i64 %.idx69, !dbg !12
  %53 = getelementptr [2 x i8], ptr addrspace(1) %52, i64 %50, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !12, !range !34, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local113 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local114 = and i64 %address.local113, 1536, !dbg !12
  %address.local115 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local114, !dbg !12
  %address.local116 = ptrtoint ptr addrspace(3) %address.local115 to i64, !dbg !12
  %address.local117 = trunc i64 %address.local116 to i32, !dbg !12
  %54 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local117), !dbg !12
  %55 = zext i32 %54 to i64, !dbg !12
  %56 = inttoptr i64 %55 to ptr addrspace(3), !dbg !12
  %address.local118 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local119 = and i64 %address.local118, 16256, !dbg !12
  %address.local120 = lshr i64 %address.local, 4, !dbg !12
  %address.local121 = and i64 %address.local120, 8, !dbg !12
  %address.local122 = or disjoint i64 %address.local119, %address.local121, !dbg !12
  %address.local123 = getelementptr [2 x i8], ptr addrspace(1) %53, i64 %address.local122, !dbg !12
  %57 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %58 = ptrtoint ptr addrspace(1) %address.local123 to i64, !dbg !12
  %59 = sub i64 %58, %57, !dbg !12
  %dma.buffer.offset1057 = trunc i64 %59 to i32, !dbg !12
  %60 = trunc i64 %57 to i32, !dbg !12
  %61 = insertelement <4 x i32> poison, i32 %60, i64 0, !dbg !12
  %62 = lshr i64 %57, 32, !dbg !12
  %63 = trunc i64 %62 to i32, !dbg !12
  %64 = insertelement <4 x i32> %61, i32 %63, i64 1, !dbg !12
  %65 = insertelement <4 x i32> %64, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1058 = insertelement <4 x i32> %65, i32 131072, i64 3, !dbg !12
  %66 = ptrtoint ptr addrspace(3) %56 to i32, !dbg !12
  %dma.buffer.lds1059 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %66), !dbg !12
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1059, i32 %dma.buffer.offset1057, <4 x i32> %dma.buffer.rsrc1058) #10, !dbg !12
  %address.local124 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local125 = shl nuw nsw i64 %address.local124, 3, !dbg !12
  %address.local126 = and i64 %address.local125, 3584, !dbg !12
  %address.local127 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local126, !dbg !12
  %address.local128 = ptrtoint ptr addrspace(3) %address.local127 to i64, !dbg !12
  %address.local129 = trunc i64 %address.local128 to i32, !dbg !12
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local129), !dbg !12
  %69 = zext i32 %68 to i64, !dbg !12
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !12
  %address.local130 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local131 = and i64 %address.local130, 16256, !dbg !12
  %address.local132 = lshr i64 %address.local124, 4, !dbg !12
  %address.local133 = and i64 %address.local132, 24, !dbg !12
  %address.local134 = or disjoint i64 %address.local131, %address.local133, !dbg !12
  %address.local135 = getelementptr [2 x i8], ptr addrspace(1) %53, i64 %address.local134, !dbg !12
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %72 = ptrtoint ptr addrspace(1) %address.local135 to i64, !dbg !12
  %73 = sub i64 %72, %71, !dbg !12
  %dma.buffer.offset1054 = trunc i64 %73 to i32, !dbg !12
  %74 = trunc i64 %71 to i32, !dbg !12
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !12
  %76 = lshr i64 %71, 32, !dbg !12
  %77 = trunc i64 %76 to i32, !dbg !12
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !12
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1055 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !12
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !12
  %dma.buffer.lds1056 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !12
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1056, i32 %dma.buffer.offset1054, <4 x i32> %dma.buffer.rsrc1055) #10, !dbg !12
  %address.local136 = or disjoint i64 %address.local, 512, !dbg !12
  %address.local137 = shl nuw nsw i64 %address.local136, 3, !dbg !12
  %address.local138 = and i64 %address.local137, 5632, !dbg !12
  %address.local139 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local138, !dbg !12
  %address.local140 = ptrtoint ptr addrspace(3) %address.local139 to i64, !dbg !12
  %address.local141 = trunc i64 %address.local140 to i32, !dbg !12
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local141), !dbg !12
  %83 = zext i32 %82 to i64, !dbg !12
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !12
  %address.local142 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local143 = and i64 %address.local142, 16256, !dbg !12
  %address.local144 = lshr i64 %address.local136, 4, !dbg !12
  %address.local145 = and i64 %address.local144, 40, !dbg !12
  %address.local146 = or disjoint i64 %address.local143, %address.local145, !dbg !12
  %address.local147 = getelementptr [2 x i8], ptr addrspace(1) %53, i64 %address.local146, !dbg !12
  %85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %86 = ptrtoint ptr addrspace(1) %address.local147 to i64, !dbg !12
  %87 = sub i64 %86, %85, !dbg !12
  %dma.buffer.offset1051 = trunc i64 %87 to i32, !dbg !12
  %88 = trunc i64 %85 to i32, !dbg !12
  %89 = insertelement <4 x i32> poison, i32 %88, i64 0, !dbg !12
  %90 = lshr i64 %85, 32, !dbg !12
  %91 = trunc i64 %90 to i32, !dbg !12
  %92 = insertelement <4 x i32> %89, i32 %91, i64 1, !dbg !12
  %93 = insertelement <4 x i32> %92, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1052 = insertelement <4 x i32> %93, i32 131072, i64 3, !dbg !12
  %94 = ptrtoint ptr addrspace(3) %84 to i32, !dbg !12
  %dma.buffer.lds1053 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !12
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1053, i32 %dma.buffer.offset1051, <4 x i32> %dma.buffer.rsrc1052) #10, !dbg !12
  %address.local148 = or disjoint i64 %address.local, 768, !dbg !12
  %address.local149 = shl nuw nsw i64 %address.local148, 3, !dbg !12
  %address.local150 = and i64 %address.local149, 7680, !dbg !12
  %address.local151 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local150, !dbg !12
  %address.local152 = ptrtoint ptr addrspace(3) %address.local151 to i64, !dbg !12
  %address.local153 = trunc i64 %address.local152 to i32, !dbg !12
  %96 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local153), !dbg !12
  %97 = zext i32 %96 to i64, !dbg !12
  %98 = inttoptr i64 %97 to ptr addrspace(3), !dbg !12
  %address.local154 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local155 = and i64 %address.local154, 16256, !dbg !12
  %address.local156 = lshr i64 %address.local148, 4, !dbg !12
  %address.local157 = and i64 %address.local156, 56, !dbg !12
  %address.local158 = or disjoint i64 %address.local155, %address.local157, !dbg !12
  %address.local159 = getelementptr [2 x i8], ptr addrspace(1) %53, i64 %address.local158, !dbg !12
  %99 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %100 = ptrtoint ptr addrspace(1) %address.local159 to i64, !dbg !12
  %101 = sub i64 %100, %99, !dbg !12
  %dma.buffer.offset1048 = trunc i64 %101 to i32, !dbg !12
  %102 = trunc i64 %99 to i32, !dbg !12
  %103 = insertelement <4 x i32> poison, i32 %102, i64 0, !dbg !12
  %104 = lshr i64 %99, 32, !dbg !12
  %105 = trunc i64 %104 to i32, !dbg !12
  %106 = insertelement <4 x i32> %103, i32 %105, i64 1, !dbg !12
  %107 = insertelement <4 x i32> %106, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1049 = insertelement <4 x i32> %107, i32 131072, i64 3, !dbg !12
  %108 = ptrtoint ptr addrspace(3) %98 to i32, !dbg !12
  %dma.buffer.lds1050 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %108), !dbg !12
  %109 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1050, i32 %dma.buffer.offset1048, <4 x i32> %dma.buffer.rsrc1049) #10, !dbg !12
  %110 = shl nuw nsw i64 %46, 11, !dbg !17
  %111 = add nuw i64 %36, %110, !dbg !17
  %.lobit70 = ashr i64 %111, 63, !dbg !17
  %112 = xor i64 %111, %.lobit70, !dbg !17
  %113 = sdiv i64 %112, 4096, !dbg !17
  %.lobit71 = ashr i64 %111, 63, !dbg !17
  %114 = xor i64 %113, %.lobit71, !dbg !17
  %115 = add nsw i64 %114, %8, !dbg !17
  %116 = add nuw nsw i64 %50, %42, !dbg !17
  %117 = and i64 %116, 127, !dbg !17
  %118 = shl i64 %115, 19, !dbg !17
  %119 = and i64 %118, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %120 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %119, !dbg !17
  %121 = getelementptr [2 x i8], ptr addrspace(1) %120, i64 %117, !dbg !17
  %.idx72 = shl i64 %44, 8, !dbg !17
  %122 = getelementptr i8, ptr addrspace(1) %121, i64 %.idx72, !dbg !17
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local114, !dbg !17
  %address.local161 = ptrtoint ptr addrspace(3) %address.local160 to i64, !dbg !17
  %address.local162 = trunc i64 %address.local161 to i32, !dbg !17
  %123 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local162), !dbg !17
  %124 = zext i32 %123 to i64, !dbg !17
  %125 = inttoptr i64 %124 to ptr addrspace(3), !dbg !17
  %address.local163 = getelementptr [2 x i8], ptr addrspace(1) %122, i64 %address.local122, !dbg !17
  %126 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %127 = ptrtoint ptr addrspace(1) %address.local163 to i64, !dbg !17
  %128 = sub i64 %127, %126, !dbg !17
  %dma.buffer.offset1045 = trunc i64 %128 to i32, !dbg !17
  %129 = trunc i64 %126 to i32, !dbg !17
  %130 = insertelement <4 x i32> poison, i32 %129, i64 0, !dbg !17
  %131 = lshr i64 %126, 32, !dbg !17
  %132 = trunc i64 %131 to i32, !dbg !17
  %133 = insertelement <4 x i32> %130, i32 %132, i64 1, !dbg !17
  %134 = insertelement <4 x i32> %133, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1046 = insertelement <4 x i32> %134, i32 131072, i64 3, !dbg !17
  %135 = ptrtoint ptr addrspace(3) %125 to i32, !dbg !17
  %dma.buffer.lds1047 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %135), !dbg !17
  %136 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1047, i32 %dma.buffer.offset1045, <4 x i32> %dma.buffer.rsrc1046) #10, !dbg !17
  %address.local164 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local126, !dbg !17
  %address.local165 = ptrtoint ptr addrspace(3) %address.local164 to i64, !dbg !17
  %address.local166 = trunc i64 %address.local165 to i32, !dbg !17
  %137 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local166), !dbg !17
  %138 = zext i32 %137 to i64, !dbg !17
  %139 = inttoptr i64 %138 to ptr addrspace(3), !dbg !17
  %address.local167 = getelementptr [2 x i8], ptr addrspace(1) %122, i64 %address.local134, !dbg !17
  %140 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %141 = ptrtoint ptr addrspace(1) %address.local167 to i64, !dbg !17
  %142 = sub i64 %141, %140, !dbg !17
  %dma.buffer.offset1042 = trunc i64 %142 to i32, !dbg !17
  %143 = trunc i64 %140 to i32, !dbg !17
  %144 = insertelement <4 x i32> poison, i32 %143, i64 0, !dbg !17
  %145 = lshr i64 %140, 32, !dbg !17
  %146 = trunc i64 %145 to i32, !dbg !17
  %147 = insertelement <4 x i32> %144, i32 %146, i64 1, !dbg !17
  %148 = insertelement <4 x i32> %147, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1043 = insertelement <4 x i32> %148, i32 131072, i64 3, !dbg !17
  %149 = ptrtoint ptr addrspace(3) %139 to i32, !dbg !17
  %dma.buffer.lds1044 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %149), !dbg !17
  %150 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1044, i32 %dma.buffer.offset1042, <4 x i32> %dma.buffer.rsrc1043) #10, !dbg !17
  %address.local168 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local138, !dbg !17
  %address.local169 = ptrtoint ptr addrspace(3) %address.local168 to i64, !dbg !17
  %address.local170 = trunc i64 %address.local169 to i32, !dbg !17
  %151 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local170), !dbg !17
  %152 = zext i32 %151 to i64, !dbg !17
  %153 = inttoptr i64 %152 to ptr addrspace(3), !dbg !17
  %address.local171 = getelementptr [2 x i8], ptr addrspace(1) %122, i64 %address.local146, !dbg !17
  %154 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %155 = ptrtoint ptr addrspace(1) %address.local171 to i64, !dbg !17
  %156 = sub i64 %155, %154, !dbg !17
  %dma.buffer.offset1039 = trunc i64 %156 to i32, !dbg !17
  %157 = trunc i64 %154 to i32, !dbg !17
  %158 = insertelement <4 x i32> poison, i32 %157, i64 0, !dbg !17
  %159 = lshr i64 %154, 32, !dbg !17
  %160 = trunc i64 %159 to i32, !dbg !17
  %161 = insertelement <4 x i32> %158, i32 %160, i64 1, !dbg !17
  %162 = insertelement <4 x i32> %161, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1040 = insertelement <4 x i32> %162, i32 131072, i64 3, !dbg !17
  %163 = ptrtoint ptr addrspace(3) %153 to i32, !dbg !17
  %dma.buffer.lds1041 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %163), !dbg !17
  %164 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1041, i32 %dma.buffer.offset1039, <4 x i32> %dma.buffer.rsrc1040) #10, !dbg !17
  %address.local172 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local150, !dbg !17
  %address.local173 = ptrtoint ptr addrspace(3) %address.local172 to i64, !dbg !17
  %address.local174 = trunc i64 %address.local173 to i32, !dbg !17
  %165 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local174), !dbg !17
  %166 = zext i32 %165 to i64, !dbg !17
  %167 = inttoptr i64 %166 to ptr addrspace(3), !dbg !17
  %address.local175 = getelementptr [2 x i8], ptr addrspace(1) %122, i64 %address.local158, !dbg !17
  %168 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %169 = ptrtoint ptr addrspace(1) %address.local175 to i64, !dbg !17
  %170 = sub i64 %169, %168, !dbg !17
  %dma.buffer.offset = trunc i64 %170 to i32, !dbg !17
  %171 = trunc i64 %168 to i32, !dbg !17
  %172 = insertelement <4 x i32> poison, i32 %171, i64 0, !dbg !17
  %173 = lshr i64 %168, 32, !dbg !17
  %174 = trunc i64 %173 to i32, !dbg !17
  %175 = insertelement <4 x i32> %172, i32 %174, i64 1, !dbg !17
  %176 = insertelement <4 x i32> %175, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %176, i32 131072, i64 3, !dbg !17
  %177 = ptrtoint ptr addrspace(3) %167 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %177), !dbg !17
  %178 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %address.local176 = lshr i32 %address.thread, 2, !dbg !13
  %address.local177 = and i32 %address.local176, 4, !dbg !13
  %address.local178 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local179 = and i32 %address.local178, 384, !dbg !13
  %address.local180 = or disjoint i32 %address.local177, %address.local179, !dbg !13
  %address.local181 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local182 = and i32 %address.local181, 120, !dbg !13
  %address.local183 = or disjoint i32 %address.local180, %address.local182, !dbg !13
  %address.local184 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local185 = and i32 %address.local184, 1024, !dbg !13
  %address.local186 = or disjoint i32 %address.local183, %address.local185, !dbg !13
  %address.local187 = zext nneg i32 %address.local186 to i64, !dbg !13
  %address.local188 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local187, !dbg !13
  %179 = load <4 x half>, ptr addrspace(3) %address.local188, align 2, !dbg !13
  %address.local189 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local187, !dbg !13
  %address.local190 = getelementptr i8, ptr addrspace(3) %address.local189, i64 4096, !dbg !13
  %180 = load <4 x half>, ptr addrspace(3) %address.local190, align 2, !dbg !13
  %address.local191 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local187, !dbg !13
  %address.local192 = getelementptr i8, ptr addrspace(3) %address.local191, i64 8192, !dbg !13
  %181 = load <4 x half>, ptr addrspace(3) %address.local192, align 2, !dbg !13
  %182 = fmul <4 x half> %179, splat (half 0xH3015), !dbg !13
  %address.local193 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local187, !dbg !13
  %address.local194 = getelementptr i8, ptr addrspace(3) %address.local193, i64 12288, !dbg !13
  %183 = load <4 x half>, ptr addrspace(3) %address.local194, align 2, !dbg !13
  %184 = fmul <4 x half> %180, splat (half 0xH3015), !dbg !13
  %address.local195 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local187, !dbg !13
  %address.local196 = getelementptr i8, ptr addrspace(3) %address.local195, i64 1024, !dbg !13
  %185 = load <4 x half>, ptr addrspace(3) %address.local196, align 2, !dbg !13
  %186 = fmul <4 x half> %181, splat (half 0xH3015), !dbg !13
  %address.local197 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local187, !dbg !13
  %address.local198 = getelementptr i8, ptr addrspace(3) %address.local197, i64 5120, !dbg !13
  %187 = load <4 x half>, ptr addrspace(3) %address.local198, align 2, !dbg !13
  %188 = fmul <4 x half> %183, splat (half 0xH3015), !dbg !13
  %address.local199 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local187, !dbg !13
  %address.local200 = getelementptr i8, ptr addrspace(3) %address.local199, i64 9216, !dbg !13
  %189 = load <4 x half>, ptr addrspace(3) %address.local200, align 2, !dbg !13
  %190 = fmul <4 x half> %185, splat (half 0xH3015), !dbg !13
  %address.local201 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local187, !dbg !13
  %address.local202 = getelementptr i8, ptr addrspace(3) %address.local201, i64 13312, !dbg !13
  %191 = load <4 x half>, ptr addrspace(3) %address.local202, align 2, !dbg !13
  %192 = fmul <4 x half> %187, splat (half 0xH3015), !dbg !13
  %193 = fmul <4 x half> %189, splat (half 0xH3015), !dbg !13
  %194 = fmul <4 x half> %191, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local203 = lshr i32 %address.thread, 2, !dbg !8
  %address.local204 = and i32 %address.local203, 4, !dbg !8
  %address.local205 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local206 = and i32 %address.local205, 24, !dbg !8
  %address.local207 = or disjoint i32 %address.local204, %address.local206, !dbg !8
  %address.local208 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local209 = and i32 %address.local208, 1024, !dbg !8
  %address.local210 = or disjoint i32 %address.local207, %address.local209, !dbg !8
  %address.local211 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local212 = and i32 %address.local211, 384, !dbg !8
  %address.local213 = or disjoint i32 %address.local210, %address.local212, !dbg !8
  %address.local214 = zext nneg i32 %address.local213 to i64, !dbg !8
  %address.local215 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local214, !dbg !8
  %195 = load <4 x half>, ptr addrspace(3) %address.local215, align 2, !dbg !8
  %196 = extractvalue [2 x <32 x float>] %47, 0, !dbg !31
  %197 = shufflevector <32 x float> %196, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %198 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %195, <4 x float> %197) #11, !dbg !8
  %address.local216 = lshr i32 %address.thread, 2, !dbg !8
  %address.local217 = and i32 %address.local216, 4, !dbg !8
  %address.local218 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local219 = and i32 %address.local218, 24, !dbg !8
  %address.local220 = or disjoint i32 %address.local217, %address.local219, !dbg !8
  %address.local221 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local222 = and i32 %address.local221, 1024, !dbg !8
  %address.local223 = or disjoint i32 %address.local220, %address.local222, !dbg !8
  %address.local224 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local225 = and i32 %address.local224, 384, !dbg !8
  %address.local226 = or disjoint i32 %address.local223, %address.local225, !dbg !8
  %address.local227 = zext nneg i32 %address.local226 to i64, !dbg !8
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local227, !dbg !8
  %address.local229 = getelementptr i8, ptr addrspace(3) %address.local228, i64 4096, !dbg !8
  %199 = load <4 x half>, ptr addrspace(3) %address.local229, align 2, !dbg !8
  %200 = shufflevector <32 x float> %196, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %201 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %199, <4 x float> %198) #11, !dbg !8
  %202 = shufflevector <32 x float> %196, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %203 = shufflevector <32 x float> %196, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %204 = shufflevector <32 x float> %196, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %205 = shufflevector <32 x float> %196, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %206 = shufflevector <32 x float> %196, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %207 = shufflevector <32 x float> %196, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %208 = extractvalue [2 x <32 x float>] %47, 1, !dbg !31
  %209 = shufflevector <32 x float> %208, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %210 = shufflevector <32 x float> %208, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %211 = shufflevector <32 x float> %208, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %212 = shufflevector <32 x float> %208, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %213 = shufflevector <32 x float> %208, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %214 = shufflevector <32 x float> %208, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %215 = shufflevector <32 x float> %208, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %216 = shufflevector <32 x float> %208, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local227, !dbg !8
  %address.local231 = getelementptr i8, ptr addrspace(3) %address.local230, i64 8192, !dbg !8
  %217 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !8
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local227, !dbg !8
  %address.local233 = getelementptr i8, ptr addrspace(3) %address.local232, i64 12288, !dbg !8
  %218 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !8
  %219 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %217, <4 x float> %201) #11, !dbg !8
  %address.local234 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local234, !dbg !8
  %address.local236 = lshr i64 %address.local, 2, !dbg !8
  %address.local237 = mul nsw i64 %address.local236, -512, !dbg !8
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local237, !dbg !8
  %address.local239 = shl nuw nsw i64 %address.local, 1, !dbg !8
  %address.local240 = and i64 %address.local239, 24, !dbg !8
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local238, i64 %address.local240, !dbg !8
  %address.local242 = lshr i64 %address.local, 2, !dbg !13
  %address.local243 = and i64 %address.local242, 12, !dbg !13
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local243, !dbg !8
  %address.local245 = and i64 %address.local, 32, !dbg !13
  %address.local246 = icmp eq i64 %address.local245, 0, !dbg !13
  %address.local247 = select i1 %address.local246, i64 0, i64 1016, !dbg !13
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local244, i64 %address.local247, !dbg !8
  %address.local249 = getelementptr i8, ptr addrspace(3) %address.local248, i64 64, !dbg !8
  %220 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !8
  %221 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %188, <4 x half> %218, <4 x float> %219) #11, !dbg !8
  %address.local250 = lshr i32 %address.thread, 2, !dbg !8
  %address.local251 = and i32 %address.local250, 4, !dbg !8
  %address.local252 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local253 = and i32 %address.local252, 24, !dbg !8
  %address.local254 = or disjoint i32 %address.local251, %address.local253, !dbg !8
  %address.local255 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local256 = and i32 %address.local255, 1024, !dbg !8
  %address.local257 = or disjoint i32 %address.local254, %address.local256, !dbg !8
  %address.local258 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local259 = and i32 %address.local258, 384, !dbg !8
  %address.local260 = or disjoint i32 %address.local257, %address.local259, !dbg !8
  %address.local261 = zext nneg i32 %address.local260 to i64, !dbg !8
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local261, !dbg !8
  %address.local263 = getelementptr i8, ptr addrspace(3) %address.local262, i64 4160, !dbg !8
  %222 = load <4 x half>, ptr addrspace(3) %address.local263, align 2, !dbg !8
  %223 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %220, <4 x float> %200) #11, !dbg !8
  %224 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %222, <4 x float> %223) #11, !dbg !8
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local261, !dbg !8
  %address.local265 = getelementptr i8, ptr addrspace(3) %address.local264, i64 8256, !dbg !8
  %225 = load <4 x half>, ptr addrspace(3) %address.local265, align 2, !dbg !8
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local261, !dbg !8
  %address.local267 = getelementptr i8, ptr addrspace(3) %address.local266, i64 12352, !dbg !8
  %226 = load <4 x half>, ptr addrspace(3) %address.local267, align 2, !dbg !8
  %227 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %225, <4 x float> %224) #11, !dbg !8
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local234, !dbg !8
  %address.local269 = getelementptr [2 x i8], ptr addrspace(3) %address.local268, i64 %address.local237, !dbg !8
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) %address.local269, i64 %address.local240, !dbg !8
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local243, !dbg !8
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) %address.local271, i64 %address.local247, !dbg !8
  %address.local273 = getelementptr i8, ptr addrspace(3) %address.local272, i64 128, !dbg !8
  %228 = load <4 x half>, ptr addrspace(3) %address.local273, align 2, !dbg !8
  %229 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %188, <4 x half> %226, <4 x float> %227) #11, !dbg !8
  %address.local274 = lshr i32 %address.thread, 2, !dbg !8
  %address.local275 = and i32 %address.local274, 4, !dbg !8
  %address.local276 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local277 = and i32 %address.local276, 24, !dbg !8
  %address.local278 = or disjoint i32 %address.local275, %address.local277, !dbg !8
  %address.local279 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local280 = and i32 %address.local279, 1024, !dbg !8
  %address.local281 = or disjoint i32 %address.local278, %address.local280, !dbg !8
  %address.local282 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local283 = and i32 %address.local282, 384, !dbg !8
  %address.local284 = or disjoint i32 %address.local281, %address.local283, !dbg !8
  %address.local285 = zext nneg i32 %address.local284 to i64, !dbg !8
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local285, !dbg !8
  %address.local287 = getelementptr i8, ptr addrspace(3) %address.local286, i64 4224, !dbg !8
  %230 = load <4 x half>, ptr addrspace(3) %address.local287, align 2, !dbg !8
  %231 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %228, <4 x float> %202) #11, !dbg !8
  %232 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %230, <4 x float> %231) #11, !dbg !8
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local285, !dbg !8
  %address.local289 = getelementptr i8, ptr addrspace(3) %address.local288, i64 8320, !dbg !8
  %233 = load <4 x half>, ptr addrspace(3) %address.local289, align 2, !dbg !8
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local285, !dbg !8
  %address.local291 = getelementptr i8, ptr addrspace(3) %address.local290, i64 12416, !dbg !8
  %234 = load <4 x half>, ptr addrspace(3) %address.local291, align 2, !dbg !8
  %235 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %233, <4 x float> %232) #11, !dbg !8
  %address.local292 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local234, !dbg !8
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) %address.local292, i64 %address.local237, !dbg !8
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) %address.local293, i64 %address.local240, !dbg !8
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) %address.local294, i64 %address.local243, !dbg !8
  %address.local296 = getelementptr [2 x i8], ptr addrspace(3) %address.local295, i64 %address.local247, !dbg !8
  %address.local297 = getelementptr i8, ptr addrspace(3) %address.local296, i64 192, !dbg !8
  %236 = load <4 x half>, ptr addrspace(3) %address.local297, align 2, !dbg !8
  %237 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %188, <4 x half> %234, <4 x float> %235) #11, !dbg !8
  %address.local298 = lshr i32 %address.thread, 2, !dbg !8
  %address.local299 = and i32 %address.local298, 4, !dbg !8
  %address.local300 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local301 = and i32 %address.local300, 24, !dbg !8
  %address.local302 = or disjoint i32 %address.local299, %address.local301, !dbg !8
  %address.local303 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local304 = and i32 %address.local303, 1024, !dbg !8
  %address.local305 = or disjoint i32 %address.local302, %address.local304, !dbg !8
  %address.local306 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local307 = and i32 %address.local306, 384, !dbg !8
  %address.local308 = or disjoint i32 %address.local305, %address.local307, !dbg !8
  %address.local309 = zext nneg i32 %address.local308 to i64, !dbg !8
  %address.local310 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local309, !dbg !8
  %address.local311 = getelementptr i8, ptr addrspace(3) %address.local310, i64 4288, !dbg !8
  %238 = load <4 x half>, ptr addrspace(3) %address.local311, align 2, !dbg !8
  %239 = shufflevector <4 x float> %237, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %240 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %236, <4 x float> %203) #11, !dbg !8
  %241 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %238, <4 x float> %240) #11, !dbg !8
  %address.local312 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local309, !dbg !8
  %address.local313 = getelementptr i8, ptr addrspace(3) %address.local312, i64 8384, !dbg !8
  %242 = load <4 x half>, ptr addrspace(3) %address.local313, align 2, !dbg !8
  %address.local314 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local309, !dbg !8
  %address.local315 = getelementptr i8, ptr addrspace(3) %address.local314, i64 12480, !dbg !8
  %243 = load <4 x half>, ptr addrspace(3) %address.local315, align 2, !dbg !8
  %244 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %242, <4 x float> %241) #11, !dbg !8
  %address.local316 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local214, !dbg !8
  %address.local317 = getelementptr i8, ptr addrspace(3) %address.local316, i64 1024, !dbg !8
  %245 = load <4 x half>, ptr addrspace(3) %address.local317, align 2, !dbg !8
  %246 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %188, <4 x half> %243, <4 x float> %244) #11, !dbg !8
  %address.local318 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local227, !dbg !8
  %address.local319 = getelementptr i8, ptr addrspace(3) %address.local318, i64 5120, !dbg !8
  %247 = load <4 x half>, ptr addrspace(3) %address.local319, align 2, !dbg !8
  %248 = shufflevector <4 x float> %246, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %249 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %245, <4 x float> %204) #11, !dbg !8
  %250 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %247, <4 x float> %249) #11, !dbg !8
  %address.local320 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local227, !dbg !8
  %address.local321 = getelementptr i8, ptr addrspace(3) %address.local320, i64 9216, !dbg !8
  %251 = load <4 x half>, ptr addrspace(3) %address.local321, align 2, !dbg !8
  %address.local322 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local227, !dbg !8
  %address.local323 = getelementptr i8, ptr addrspace(3) %address.local322, i64 13312, !dbg !8
  %252 = load <4 x half>, ptr addrspace(3) %address.local323, align 2, !dbg !8
  %253 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %251, <4 x float> %250) #11, !dbg !8
  %address.local324 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local234, !dbg !8
  %address.local325 = getelementptr [2 x i8], ptr addrspace(3) %address.local324, i64 %address.local237, !dbg !8
  %address.local326 = getelementptr [2 x i8], ptr addrspace(3) %address.local325, i64 %address.local240, !dbg !8
  %address.local327 = getelementptr [2 x i8], ptr addrspace(3) %address.local326, i64 %address.local243, !dbg !8
  %address.local328 = getelementptr [2 x i8], ptr addrspace(3) %address.local327, i64 %address.local247, !dbg !8
  %address.local329 = getelementptr i8, ptr addrspace(3) %address.local328, i64 1088, !dbg !8
  %254 = load <4 x half>, ptr addrspace(3) %address.local329, align 2, !dbg !8
  %255 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %188, <4 x half> %252, <4 x float> %253) #11, !dbg !8
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
  %address.local341 = or disjoint i32 %address.local340, 512, !dbg !8
  %address.local342 = zext nneg i32 %address.local341 to i64, !dbg !8
  %address.local343 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local342, !dbg !8
  %address.local344 = getelementptr i8, ptr addrspace(3) %address.local343, i64 4160, !dbg !8
  %256 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !8
  %257 = shufflevector <4 x float> %255, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %258 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %254, <4 x float> %205) #11, !dbg !8
  %259 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %256, <4 x float> %258) #11, !dbg !8
  %address.local345 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local342, !dbg !8
  %address.local346 = getelementptr i8, ptr addrspace(3) %address.local345, i64 8256, !dbg !8
  %260 = load <4 x half>, ptr addrspace(3) %address.local346, align 2, !dbg !8
  %address.local347 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local342, !dbg !8
  %address.local348 = getelementptr i8, ptr addrspace(3) %address.local347, i64 12352, !dbg !8
  %261 = load <4 x half>, ptr addrspace(3) %address.local348, align 2, !dbg !8
  %262 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %260, <4 x float> %259) #11, !dbg !8
  %address.local349 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local234, !dbg !8
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) %address.local349, i64 %address.local237, !dbg !8
  %address.local351 = getelementptr [2 x i8], ptr addrspace(3) %address.local350, i64 %address.local240, !dbg !8
  %address.local352 = getelementptr [2 x i8], ptr addrspace(3) %address.local351, i64 %address.local243, !dbg !8
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) %address.local352, i64 %address.local247, !dbg !8
  %address.local354 = getelementptr i8, ptr addrspace(3) %address.local353, i64 1152, !dbg !8
  %263 = load <4 x half>, ptr addrspace(3) %address.local354, align 2, !dbg !8
  %264 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %188, <4 x half> %261, <4 x float> %262) #11, !dbg !8
  %address.local355 = lshr i32 %address.thread, 2, !dbg !8
  %address.local356 = and i32 %address.local355, 4, !dbg !8
  %address.local357 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local358 = and i32 %address.local357, 24, !dbg !8
  %address.local359 = or disjoint i32 %address.local356, %address.local358, !dbg !8
  %address.local360 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local361 = and i32 %address.local360, 1024, !dbg !8
  %address.local362 = or disjoint i32 %address.local359, %address.local361, !dbg !8
  %address.local363 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local364 = and i32 %address.local363, 384, !dbg !8
  %address.local365 = or disjoint i32 %address.local362, %address.local364, !dbg !8
  %address.local366 = or disjoint i32 %address.local365, 512, !dbg !8
  %address.local367 = zext nneg i32 %address.local366 to i64, !dbg !8
  %address.local368 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local367, !dbg !8
  %address.local369 = getelementptr i8, ptr addrspace(3) %address.local368, i64 4224, !dbg !8
  %265 = load <4 x half>, ptr addrspace(3) %address.local369, align 2, !dbg !8
  %266 = shufflevector <4 x float> %264, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %267 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %263, <4 x float> %206) #11, !dbg !8
  %268 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %265, <4 x float> %267) #11, !dbg !8
  %address.local370 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local367, !dbg !8
  %address.local371 = getelementptr i8, ptr addrspace(3) %address.local370, i64 8320, !dbg !8
  %269 = load <4 x half>, ptr addrspace(3) %address.local371, align 2, !dbg !8
  %address.local372 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local367, !dbg !8
  %address.local373 = getelementptr i8, ptr addrspace(3) %address.local372, i64 12416, !dbg !8
  %270 = load <4 x half>, ptr addrspace(3) %address.local373, align 2, !dbg !8
  %271 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %269, <4 x float> %268) #11, !dbg !8
  %address.local374 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local234, !dbg !8
  %address.local375 = getelementptr [2 x i8], ptr addrspace(3) %address.local374, i64 %address.local237, !dbg !8
  %address.local376 = getelementptr [2 x i8], ptr addrspace(3) %address.local375, i64 %address.local240, !dbg !8
  %address.local377 = getelementptr [2 x i8], ptr addrspace(3) %address.local376, i64 %address.local243, !dbg !8
  %address.local378 = getelementptr [2 x i8], ptr addrspace(3) %address.local377, i64 %address.local247, !dbg !8
  %address.local379 = getelementptr i8, ptr addrspace(3) %address.local378, i64 1216, !dbg !8
  %272 = load <4 x half>, ptr addrspace(3) %address.local379, align 2, !dbg !8
  %273 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %188, <4 x half> %270, <4 x float> %271) #11, !dbg !8
  %address.local380 = lshr i32 %address.thread, 2, !dbg !8
  %address.local381 = and i32 %address.local380, 4, !dbg !8
  %address.local382 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local383 = and i32 %address.local382, 24, !dbg !8
  %address.local384 = or disjoint i32 %address.local381, %address.local383, !dbg !8
  %address.local385 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local386 = and i32 %address.local385, 1024, !dbg !8
  %address.local387 = or disjoint i32 %address.local384, %address.local386, !dbg !8
  %address.local388 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local389 = and i32 %address.local388, 384, !dbg !8
  %address.local390 = or disjoint i32 %address.local387, %address.local389, !dbg !8
  %address.local391 = or disjoint i32 %address.local390, 512, !dbg !8
  %address.local392 = zext nneg i32 %address.local391 to i64, !dbg !8
  %address.local393 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local392, !dbg !8
  %address.local394 = getelementptr i8, ptr addrspace(3) %address.local393, i64 4288, !dbg !8
  %274 = load <4 x half>, ptr addrspace(3) %address.local394, align 2, !dbg !8
  %275 = shufflevector <4 x float> %273, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %276 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %272, <4 x float> %207) #11, !dbg !8
  %277 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %274, <4 x float> %276) #11, !dbg !8
  %address.local395 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local392, !dbg !8
  %address.local396 = getelementptr i8, ptr addrspace(3) %address.local395, i64 8384, !dbg !8
  %278 = load <4 x half>, ptr addrspace(3) %address.local396, align 2, !dbg !8
  %address.local397 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local392, !dbg !8
  %address.local398 = getelementptr i8, ptr addrspace(3) %address.local397, i64 12480, !dbg !8
  %279 = load <4 x half>, ptr addrspace(3) %address.local398, align 2, !dbg !8
  %280 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %278, <4 x float> %277) #11, !dbg !8
  %281 = load <4 x half>, ptr addrspace(3) %address.local215, align 2, !dbg !8
  %282 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %188, <4 x half> %279, <4 x float> %280) #11, !dbg !8
  %283 = load <4 x half>, ptr addrspace(3) %address.local229, align 2, !dbg !8
  %284 = shufflevector <4 x float> %282, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %285 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %190, <4 x half> %281, <4 x float> %209) #11, !dbg !8
  %286 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %192, <4 x half> %283, <4 x float> %285) #11, !dbg !8
  %287 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !8
  %288 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !8
  %289 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %287, <4 x float> %286) #11, !dbg !8
  %290 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !8
  %291 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %194, <4 x half> %288, <4 x float> %289) #11, !dbg !8
  %292 = load <4 x half>, ptr addrspace(3) %address.local263, align 2, !dbg !8
  %293 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %190, <4 x half> %290, <4 x float> %210) #11, !dbg !8
  %294 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %192, <4 x half> %292, <4 x float> %293) #11, !dbg !8
  %295 = load <4 x half>, ptr addrspace(3) %address.local265, align 2, !dbg !8
  %296 = load <4 x half>, ptr addrspace(3) %address.local267, align 2, !dbg !8
  %297 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %295, <4 x float> %294) #11, !dbg !8
  %298 = load <4 x half>, ptr addrspace(3) %address.local273, align 2, !dbg !8
  %299 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %194, <4 x half> %296, <4 x float> %297) #11, !dbg !8
  %300 = load <4 x half>, ptr addrspace(3) %address.local287, align 2, !dbg !8
  %301 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %190, <4 x half> %298, <4 x float> %211) #11, !dbg !8
  %302 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %192, <4 x half> %300, <4 x float> %301) #11, !dbg !8
  %303 = load <4 x half>, ptr addrspace(3) %address.local289, align 2, !dbg !8
  %304 = load <4 x half>, ptr addrspace(3) %address.local291, align 2, !dbg !8
  %305 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %303, <4 x float> %302) #11, !dbg !8
  %306 = load <4 x half>, ptr addrspace(3) %address.local297, align 2, !dbg !8
  %307 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %194, <4 x half> %304, <4 x float> %305) #11, !dbg !8
  %308 = load <4 x half>, ptr addrspace(3) %address.local311, align 2, !dbg !8
  %309 = shufflevector <4 x float> %307, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %310 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %190, <4 x half> %306, <4 x float> %212) #11, !dbg !8
  %311 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %192, <4 x half> %308, <4 x float> %310) #11, !dbg !8
  %312 = load <4 x half>, ptr addrspace(3) %address.local313, align 2, !dbg !8
  %313 = load <4 x half>, ptr addrspace(3) %address.local315, align 2, !dbg !8
  %314 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %312, <4 x float> %311) #11, !dbg !8
  %315 = load <4 x half>, ptr addrspace(3) %address.local317, align 2, !dbg !8
  %316 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %194, <4 x half> %313, <4 x float> %314) #11, !dbg !8
  %317 = load <4 x half>, ptr addrspace(3) %address.local319, align 2, !dbg !8
  %318 = shufflevector <4 x float> %316, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %319 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %190, <4 x half> %315, <4 x float> %213) #11, !dbg !8
  %320 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %192, <4 x half> %317, <4 x float> %319) #11, !dbg !8
  %321 = load <4 x half>, ptr addrspace(3) %address.local321, align 2, !dbg !8
  %322 = load <4 x half>, ptr addrspace(3) %address.local323, align 2, !dbg !8
  %323 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %321, <4 x float> %320) #11, !dbg !8
  %324 = load <4 x half>, ptr addrspace(3) %address.local329, align 2, !dbg !8
  %325 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %194, <4 x half> %322, <4 x float> %323) #11, !dbg !8
  %326 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !8
  %327 = shufflevector <4 x float> %325, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %328 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %190, <4 x half> %324, <4 x float> %214) #11, !dbg !8
  %329 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %192, <4 x half> %326, <4 x float> %328) #11, !dbg !8
  %330 = load <4 x half>, ptr addrspace(3) %address.local346, align 2, !dbg !8
  %331 = load <4 x half>, ptr addrspace(3) %address.local348, align 2, !dbg !8
  %332 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %330, <4 x float> %329) #11, !dbg !8
  %333 = load <4 x half>, ptr addrspace(3) %address.local354, align 2, !dbg !8
  %334 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %194, <4 x half> %331, <4 x float> %332) #11, !dbg !8
  %335 = load <4 x half>, ptr addrspace(3) %address.local369, align 2, !dbg !8
  %336 = shufflevector <4 x float> %334, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %337 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %190, <4 x half> %333, <4 x float> %215) #11, !dbg !8
  %338 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %192, <4 x half> %335, <4 x float> %337) #11, !dbg !8
  %339 = load <4 x half>, ptr addrspace(3) %address.local371, align 2, !dbg !8
  %340 = load <4 x half>, ptr addrspace(3) %address.local373, align 2, !dbg !8
  %341 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %339, <4 x float> %338) #11, !dbg !8
  %342 = load <4 x half>, ptr addrspace(3) %address.local379, align 2, !dbg !8
  %343 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %194, <4 x half> %340, <4 x float> %341) #11, !dbg !8
  %344 = load <4 x half>, ptr addrspace(3) %address.local394, align 2, !dbg !8
  %345 = shufflevector <4 x float> %343, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %346 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %190, <4 x half> %342, <4 x float> %216) #11, !dbg !8
  %347 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %192, <4 x half> %344, <4 x float> %346) #11, !dbg !8
  %348 = load <4 x half>, ptr addrspace(3) %address.local396, align 2, !dbg !8
  %349 = load <4 x half>, ptr addrspace(3) %address.local398, align 2, !dbg !8
  %350 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %348, <4 x float> %347) #11, !dbg !8
  %351 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %194, <4 x half> %349, <4 x float> %350) #11, !dbg !8
  %352 = shufflevector <4 x float> %351, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %353 = shufflevector <4 x float> %221, <4 x float> %229, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %354 = shufflevector <32 x float> %353, <32 x float> %239, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %355 = shufflevector <32 x float> %354, <32 x float> %248, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %356 = shufflevector <32 x float> %355, <32 x float> %257, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %357 = shufflevector <32 x float> %356, <32 x float> %266, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %358 = shufflevector <32 x float> %357, <32 x float> %275, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %359 = shufflevector <32 x float> %358, <32 x float> %284, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %360 = insertvalue [2 x <32 x float>] poison, <32 x float> %359, 0, !dbg !22
  %361 = shufflevector <4 x float> %291, <4 x float> %299, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %362 = shufflevector <32 x float> %361, <32 x float> %309, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %363 = shufflevector <32 x float> %362, <32 x float> %318, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %364 = shufflevector <32 x float> %363, <32 x float> %327, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %365 = shufflevector <32 x float> %364, <32 x float> %336, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %366 = shufflevector <32 x float> %365, <32 x float> %345, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %367 = shufflevector <32 x float> %366, <32 x float> %352, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %368 = insertvalue [2 x <32 x float>] %360, <32 x float> %367, 1, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %369 = add nuw nsw i64 %46, 1, !dbg !22
  br label %45, !dbg !22

370:                                              ; preds = %45
  %371 = add nsw i64 %42, %8, !dbg !18
  %372 = shl i64 %371, 19, !dbg !18
  %373 = and i64 %372, 16252928, !dbg !18
  %374 = shl nsw i64 %44, 7, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %375 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %373, !dbg !18
  %376 = getelementptr [2 x i8], ptr addrspace(1) %375, i64 %374, !dbg !18
  %address.thread399 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !18, !range !34, !frisk.register_lifetimes !1
  %address.local400 = zext nneg i32 %address.thread399 to i64, !dbg !13
  %address.local401 = shl nuw nsw i64 %address.local400, 3, !dbg !12
  %address.local402 = and i64 %address.local401, 1536, !dbg !12
  %address.local403 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local402, !dbg !12
  %address.local404 = ptrtoint ptr addrspace(3) %address.local403 to i64, !dbg !12
  %address.local405 = trunc i64 %address.local404 to i32, !dbg !12
  %377 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local405), !dbg !18
  %378 = zext i32 %377 to i64, !dbg !18
  %379 = inttoptr i64 %378 to ptr addrspace(3), !dbg !18
  %address.local406 = shl nuw nsw i64 %address.local400, 7, !dbg !12
  %address.local407 = and i64 %address.local406, 16256, !dbg !12
  %address.local408 = lshr i64 %address.local400, 4, !dbg !12
  %address.local409 = and i64 %address.local408, 8, !dbg !12
  %address.local410 = or disjoint i64 %address.local407, %address.local409, !dbg !12
  %address.local411 = getelementptr [2 x i8], ptr addrspace(1) %376, i64 %address.local410, !dbg !18
  %380 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %381 = ptrtoint ptr addrspace(1) %address.local411 to i64, !dbg !18
  %382 = sub i64 %381, %380, !dbg !18
  %dma.buffer.offset1081 = trunc i64 %382 to i32, !dbg !18
  %383 = trunc i64 %380 to i32, !dbg !18
  %384 = insertelement <4 x i32> poison, i32 %383, i64 0, !dbg !18
  %385 = lshr i64 %380, 32, !dbg !18
  %386 = trunc i64 %385 to i32, !dbg !18
  %387 = insertelement <4 x i32> %384, i32 %386, i64 1, !dbg !18
  %388 = insertelement <4 x i32> %387, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1082 = insertelement <4 x i32> %388, i32 131072, i64 3, !dbg !18
  %389 = ptrtoint ptr addrspace(3) %379 to i32, !dbg !18
  %dma.buffer.lds1083 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %389), !dbg !18
  %390 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1083, i32 %dma.buffer.offset1081, <4 x i32> %dma.buffer.rsrc1082) #10, !dbg !18
  %address.local412 = or disjoint i64 %address.local400, 256, !dbg !12
  %address.local413 = shl nuw nsw i64 %address.local412, 3, !dbg !12
  %address.local414 = and i64 %address.local413, 3584, !dbg !12
  %address.local415 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local414, !dbg !12
  %address.local416 = ptrtoint ptr addrspace(3) %address.local415 to i64, !dbg !12
  %address.local417 = trunc i64 %address.local416 to i32, !dbg !12
  %391 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local417), !dbg !18
  %392 = zext i32 %391 to i64, !dbg !18
  %393 = inttoptr i64 %392 to ptr addrspace(3), !dbg !18
  %address.local418 = shl nuw nsw i64 %address.local400, 7, !dbg !12
  %address.local419 = and i64 %address.local418, 16256, !dbg !12
  %address.local420 = lshr i64 %address.local412, 4, !dbg !12
  %address.local421 = and i64 %address.local420, 24, !dbg !12
  %address.local422 = or disjoint i64 %address.local419, %address.local421, !dbg !12
  %address.local423 = getelementptr [2 x i8], ptr addrspace(1) %376, i64 %address.local422, !dbg !18
  %394 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %395 = ptrtoint ptr addrspace(1) %address.local423 to i64, !dbg !18
  %396 = sub i64 %395, %394, !dbg !18
  %dma.buffer.offset1078 = trunc i64 %396 to i32, !dbg !18
  %397 = trunc i64 %394 to i32, !dbg !18
  %398 = insertelement <4 x i32> poison, i32 %397, i64 0, !dbg !18
  %399 = lshr i64 %394, 32, !dbg !18
  %400 = trunc i64 %399 to i32, !dbg !18
  %401 = insertelement <4 x i32> %398, i32 %400, i64 1, !dbg !18
  %402 = insertelement <4 x i32> %401, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1079 = insertelement <4 x i32> %402, i32 131072, i64 3, !dbg !18
  %403 = ptrtoint ptr addrspace(3) %393 to i32, !dbg !18
  %dma.buffer.lds1080 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %403), !dbg !18
  %404 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1080, i32 %dma.buffer.offset1078, <4 x i32> %dma.buffer.rsrc1079) #10, !dbg !18
  %address.local424 = or disjoint i64 %address.local400, 512, !dbg !12
  %address.local425 = shl nuw nsw i64 %address.local424, 3, !dbg !12
  %address.local426 = and i64 %address.local425, 5632, !dbg !12
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local426, !dbg !12
  %address.local428 = ptrtoint ptr addrspace(3) %address.local427 to i64, !dbg !12
  %address.local429 = trunc i64 %address.local428 to i32, !dbg !12
  %405 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !18
  %406 = zext i32 %405 to i64, !dbg !18
  %407 = inttoptr i64 %406 to ptr addrspace(3), !dbg !18
  %address.local430 = shl nuw nsw i64 %address.local400, 7, !dbg !12
  %address.local431 = and i64 %address.local430, 16256, !dbg !12
  %address.local432 = lshr i64 %address.local424, 4, !dbg !12
  %address.local433 = and i64 %address.local432, 40, !dbg !12
  %address.local434 = or disjoint i64 %address.local431, %address.local433, !dbg !12
  %address.local435 = getelementptr [2 x i8], ptr addrspace(1) %376, i64 %address.local434, !dbg !18
  %408 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %409 = ptrtoint ptr addrspace(1) %address.local435 to i64, !dbg !18
  %410 = sub i64 %409, %408, !dbg !18
  %dma.buffer.offset1075 = trunc i64 %410 to i32, !dbg !18
  %411 = trunc i64 %408 to i32, !dbg !18
  %412 = insertelement <4 x i32> poison, i32 %411, i64 0, !dbg !18
  %413 = lshr i64 %408, 32, !dbg !18
  %414 = trunc i64 %413 to i32, !dbg !18
  %415 = insertelement <4 x i32> %412, i32 %414, i64 1, !dbg !18
  %416 = insertelement <4 x i32> %415, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1076 = insertelement <4 x i32> %416, i32 131072, i64 3, !dbg !18
  %417 = ptrtoint ptr addrspace(3) %407 to i32, !dbg !18
  %dma.buffer.lds1077 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %417), !dbg !18
  %418 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1077, i32 %dma.buffer.offset1075, <4 x i32> %dma.buffer.rsrc1076) #10, !dbg !18
  %address.local436 = or disjoint i64 %address.local400, 768, !dbg !12
  %address.local437 = shl nuw nsw i64 %address.local436, 3, !dbg !12
  %address.local438 = and i64 %address.local437, 7680, !dbg !12
  %address.local439 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local438, !dbg !12
  %address.local440 = ptrtoint ptr addrspace(3) %address.local439 to i64, !dbg !12
  %address.local441 = trunc i64 %address.local440 to i32, !dbg !12
  %419 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local441), !dbg !18
  %420 = zext i32 %419 to i64, !dbg !18
  %421 = inttoptr i64 %420 to ptr addrspace(3), !dbg !18
  %address.local442 = shl nuw nsw i64 %address.local400, 7, !dbg !12
  %address.local443 = and i64 %address.local442, 16256, !dbg !12
  %address.local444 = lshr i64 %address.local436, 4, !dbg !12
  %address.local445 = and i64 %address.local444, 56, !dbg !12
  %address.local446 = or disjoint i64 %address.local443, %address.local445, !dbg !12
  %address.local447 = getelementptr [2 x i8], ptr addrspace(1) %376, i64 %address.local446, !dbg !18
  %422 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %423 = ptrtoint ptr addrspace(1) %address.local447 to i64, !dbg !18
  %424 = sub i64 %423, %422, !dbg !18
  %dma.buffer.offset1072 = trunc i64 %424 to i32, !dbg !18
  %425 = trunc i64 %422 to i32, !dbg !18
  %426 = insertelement <4 x i32> poison, i32 %425, i64 0, !dbg !18
  %427 = lshr i64 %422, 32, !dbg !18
  %428 = trunc i64 %427 to i32, !dbg !18
  %429 = insertelement <4 x i32> %426, i32 %428, i64 1, !dbg !18
  %430 = insertelement <4 x i32> %429, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1073 = insertelement <4 x i32> %430, i32 131072, i64 3, !dbg !18
  %431 = ptrtoint ptr addrspace(3) %421 to i32, !dbg !18
  %dma.buffer.lds1074 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %431), !dbg !18
  %432 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1074, i32 %dma.buffer.offset1072, <4 x i32> %dma.buffer.rsrc1073) #10, !dbg !18
  %433 = or disjoint i64 %41, 127, !dbg !14
  %434 = icmp uge i64 %9, %433, !dbg !14
  %435 = and i1 %24, %434, !dbg !14
  br i1 %435, label %436, label %437, !dbg !14

436:                                              ; preds = %370
  br label %606, !dbg !14

437:                                              ; preds = %370
  %438 = add nsw i64 %9, %20, !dbg !14
  %439 = add nsw i64 %438, %33, !dbg !14
  %440 = add nsw i64 %439, %34, !dbg !14
  %441 = and i64 %20, 48, !dbg !14
  %442 = or disjoint i64 %41, %441, !dbg !14
  %443 = add nsw i64 %440, 1, !dbg !23
  %.not4 = icmp ugt i64 %443, %442, !dbg !24
  %444 = select i1 %.not4, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %445 = or disjoint i64 %442, 1, !dbg !14
  %.not5 = icmp ugt i64 %443, %445, !dbg !24
  %446 = select i1 %.not5, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %447 = or disjoint i64 %442, 2, !dbg !14
  %.not6 = icmp ugt i64 %443, %447, !dbg !24
  %448 = select i1 %.not6, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %449 = or disjoint i64 %442, 3, !dbg !14
  %.not7 = icmp ugt i64 %443, %449, !dbg !24
  %450 = select i1 %.not7, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %451 = or disjoint i64 %442, 4, !dbg !14
  %.not8 = icmp ugt i64 %443, %451, !dbg !24
  %452 = select i1 %.not8, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %453 = or disjoint i64 %442, 5, !dbg !14
  %.not9 = icmp ugt i64 %443, %453, !dbg !24
  %454 = select i1 %.not9, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %455 = or disjoint i64 %442, 6, !dbg !14
  %.not10 = icmp ugt i64 %443, %455, !dbg !24
  %456 = select i1 %.not10, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %457 = or disjoint i64 %442, 7, !dbg !14
  %.not11 = icmp ugt i64 %443, %457, !dbg !24
  %458 = select i1 %.not11, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %459 = or disjoint i64 %442, 8, !dbg !14
  %.not12 = icmp ugt i64 %443, %459, !dbg !24
  %460 = select i1 %.not12, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %461 = or disjoint i64 %442, 9, !dbg !14
  %.not13 = icmp ugt i64 %443, %461, !dbg !24
  %462 = select i1 %.not13, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %463 = or disjoint i64 %442, 10, !dbg !14
  %.not14 = icmp ugt i64 %443, %463, !dbg !24
  %464 = select i1 %.not14, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %465 = or disjoint i64 %442, 11, !dbg !14
  %.not15 = icmp ugt i64 %443, %465, !dbg !24
  %466 = select i1 %.not15, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %467 = or disjoint i64 %442, 12, !dbg !14
  %.not16 = icmp ugt i64 %443, %467, !dbg !24
  %468 = select i1 %.not16, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %469 = or disjoint i64 %442, 13, !dbg !14
  %.not17 = icmp ugt i64 %443, %469, !dbg !24
  %470 = select i1 %.not17, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %471 = or disjoint i64 %442, 14, !dbg !14
  %.not18 = icmp ugt i64 %443, %471, !dbg !24
  %472 = select i1 %.not18, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %473 = or disjoint i64 %442, 15, !dbg !14
  %.not19 = icmp ugt i64 %443, %473, !dbg !24
  %474 = select i1 %.not19, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %475 = or disjoint i64 %442, 64, !dbg !14
  %.not20 = icmp ugt i64 %443, %475, !dbg !24
  %476 = select i1 %.not20, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %477 = or disjoint i64 %442, 65, !dbg !14
  %.not21 = icmp ugt i64 %443, %477, !dbg !24
  %478 = select i1 %.not21, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %479 = or disjoint i64 %442, 66, !dbg !14
  %.not22 = icmp ugt i64 %443, %479, !dbg !24
  %480 = select i1 %.not22, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %481 = or disjoint i64 %442, 67, !dbg !14
  %.not23 = icmp ugt i64 %443, %481, !dbg !24
  %482 = select i1 %.not23, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %483 = or disjoint i64 %442, 68, !dbg !14
  %.not24 = icmp ugt i64 %443, %483, !dbg !24
  %484 = select i1 %.not24, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %485 = or disjoint i64 %442, 69, !dbg !14
  %.not25 = icmp ugt i64 %443, %485, !dbg !24
  %486 = select i1 %.not25, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %487 = or disjoint i64 %442, 70, !dbg !14
  %.not26 = icmp ugt i64 %443, %487, !dbg !24
  %488 = select i1 %.not26, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %489 = or disjoint i64 %442, 71, !dbg !14
  %.not27 = icmp ugt i64 %443, %489, !dbg !24
  %490 = select i1 %.not27, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %491 = or disjoint i64 %442, 72, !dbg !14
  %.not28 = icmp ugt i64 %443, %491, !dbg !24
  %492 = select i1 %.not28, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %493 = or disjoint i64 %442, 73, !dbg !14
  %.not29 = icmp ugt i64 %443, %493, !dbg !24
  %494 = select i1 %.not29, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %495 = or disjoint i64 %442, 74, !dbg !14
  %.not30 = icmp ugt i64 %443, %495, !dbg !24
  %496 = select i1 %.not30, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %497 = or disjoint i64 %442, 75, !dbg !14
  %.not31 = icmp ugt i64 %443, %497, !dbg !24
  %498 = select i1 %.not31, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %499 = or disjoint i64 %442, 76, !dbg !14
  %.not32 = icmp ugt i64 %443, %499, !dbg !24
  %500 = select i1 %.not32, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %501 = or disjoint i64 %442, 77, !dbg !14
  %.not33 = icmp ugt i64 %443, %501, !dbg !24
  %502 = select i1 %.not33, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %503 = or disjoint i64 %442, 78, !dbg !14
  %.not34 = icmp ugt i64 %443, %503, !dbg !24
  %504 = select i1 %.not34, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %505 = or disjoint i64 %442, 79, !dbg !14
  %.not35 = icmp ugt i64 %443, %505, !dbg !24
  %506 = select i1 %.not35, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %507 = add nsw i64 %440, 65, !dbg !23
  %.not36 = icmp ugt i64 %507, %442, !dbg !24
  %508 = select i1 %.not36, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not37 = icmp ugt i64 %507, %445, !dbg !24
  %509 = select i1 %.not37, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not38 = icmp ugt i64 %507, %447, !dbg !24
  %510 = select i1 %.not38, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not39 = icmp ugt i64 %507, %449, !dbg !24
  %511 = select i1 %.not39, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not40 = icmp ugt i64 %507, %451, !dbg !24
  %512 = select i1 %.not40, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not41 = icmp ugt i64 %507, %453, !dbg !24
  %513 = select i1 %.not41, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not42 = icmp ugt i64 %507, %455, !dbg !24
  %514 = select i1 %.not42, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not43 = icmp ugt i64 %507, %457, !dbg !24
  %515 = select i1 %.not43, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not44 = icmp ugt i64 %507, %459, !dbg !24
  %516 = select i1 %.not44, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not45 = icmp ugt i64 %507, %461, !dbg !24
  %517 = select i1 %.not45, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not46 = icmp ugt i64 %507, %463, !dbg !24
  %518 = select i1 %.not46, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not47 = icmp ugt i64 %507, %465, !dbg !24
  %519 = select i1 %.not47, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not48 = icmp ugt i64 %507, %467, !dbg !24
  %520 = select i1 %.not48, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not49 = icmp ugt i64 %507, %469, !dbg !24
  %521 = select i1 %.not49, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not50 = icmp ugt i64 %507, %471, !dbg !24
  %522 = select i1 %.not50, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not51 = icmp ugt i64 %507, %473, !dbg !24
  %523 = select i1 %.not51, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not52 = icmp ugt i64 %507, %475, !dbg !24
  %524 = select i1 %.not52, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not53 = icmp ugt i64 %507, %477, !dbg !24
  %525 = select i1 %.not53, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not54 = icmp ugt i64 %507, %479, !dbg !24
  %526 = select i1 %.not54, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not55 = icmp ugt i64 %507, %481, !dbg !24
  %527 = select i1 %.not55, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not56 = icmp ugt i64 %507, %483, !dbg !24
  %528 = select i1 %.not56, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not57 = icmp ugt i64 %507, %485, !dbg !24
  %529 = select i1 %.not57, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not58 = icmp ugt i64 %507, %487, !dbg !24
  %530 = select i1 %.not58, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not59 = icmp ugt i64 %507, %489, !dbg !24
  %531 = select i1 %.not59, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not60 = icmp ugt i64 %507, %491, !dbg !24
  %532 = select i1 %.not60, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not61 = icmp ugt i64 %507, %493, !dbg !24
  %533 = select i1 %.not61, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not62 = icmp ugt i64 %507, %495, !dbg !24
  %534 = select i1 %.not62, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not63 = icmp ugt i64 %507, %497, !dbg !24
  %535 = select i1 %.not63, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not64 = icmp ugt i64 %507, %499, !dbg !24
  %536 = select i1 %.not64, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not65 = icmp ugt i64 %507, %501, !dbg !24
  %537 = select i1 %.not65, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not66 = icmp ugt i64 %507, %503, !dbg !24
  %538 = select i1 %.not66, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not67 = icmp ugt i64 %507, %505, !dbg !24
  %539 = select i1 %.not67, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %540 = insertelement <32 x float> poison, float %444, i64 0, !dbg !14
  %541 = insertelement <32 x float> %540, float %446, i64 1, !dbg !14
  %542 = insertelement <32 x float> %541, float %448, i64 2, !dbg !14
  %543 = insertelement <32 x float> %542, float %450, i64 3, !dbg !14
  %544 = insertelement <32 x float> %543, float %452, i64 4, !dbg !14
  %545 = insertelement <32 x float> %544, float %454, i64 5, !dbg !14
  %546 = insertelement <32 x float> %545, float %456, i64 6, !dbg !14
  %547 = insertelement <32 x float> %546, float %458, i64 7, !dbg !14
  %548 = insertelement <32 x float> %547, float %460, i64 8, !dbg !14
  %549 = insertelement <32 x float> %548, float %462, i64 9, !dbg !14
  %550 = insertelement <32 x float> %549, float %464, i64 10, !dbg !14
  %551 = insertelement <32 x float> %550, float %466, i64 11, !dbg !14
  %552 = insertelement <32 x float> %551, float %468, i64 12, !dbg !14
  %553 = insertelement <32 x float> %552, float %470, i64 13, !dbg !14
  %554 = insertelement <32 x float> %553, float %472, i64 14, !dbg !14
  %555 = insertelement <32 x float> %554, float %474, i64 15, !dbg !14
  %556 = insertelement <32 x float> %555, float %476, i64 16, !dbg !14
  %557 = insertelement <32 x float> %556, float %478, i64 17, !dbg !14
  %558 = insertelement <32 x float> %557, float %480, i64 18, !dbg !14
  %559 = insertelement <32 x float> %558, float %482, i64 19, !dbg !14
  %560 = insertelement <32 x float> %559, float %484, i64 20, !dbg !14
  %561 = insertelement <32 x float> %560, float %486, i64 21, !dbg !14
  %562 = insertelement <32 x float> %561, float %488, i64 22, !dbg !14
  %563 = insertelement <32 x float> %562, float %490, i64 23, !dbg !14
  %564 = insertelement <32 x float> %563, float %492, i64 24, !dbg !14
  %565 = insertelement <32 x float> %564, float %494, i64 25, !dbg !14
  %566 = insertelement <32 x float> %565, float %496, i64 26, !dbg !14
  %567 = insertelement <32 x float> %566, float %498, i64 27, !dbg !14
  %568 = insertelement <32 x float> %567, float %500, i64 28, !dbg !14
  %569 = insertelement <32 x float> %568, float %502, i64 29, !dbg !14
  %570 = insertelement <32 x float> %569, float %504, i64 30, !dbg !14
  %571 = insertelement <32 x float> %570, float %506, i64 31, !dbg !14
  %572 = insertvalue [2 x <32 x float>] poison, <32 x float> %571, 0, !dbg !14
  %573 = insertelement <32 x float> poison, float %508, i64 0, !dbg !14
  %574 = insertelement <32 x float> %573, float %509, i64 1, !dbg !14
  %575 = insertelement <32 x float> %574, float %510, i64 2, !dbg !14
  %576 = insertelement <32 x float> %575, float %511, i64 3, !dbg !14
  %577 = insertelement <32 x float> %576, float %512, i64 4, !dbg !14
  %578 = insertelement <32 x float> %577, float %513, i64 5, !dbg !14
  %579 = insertelement <32 x float> %578, float %514, i64 6, !dbg !14
  %580 = insertelement <32 x float> %579, float %515, i64 7, !dbg !14
  %581 = insertelement <32 x float> %580, float %516, i64 8, !dbg !14
  %582 = insertelement <32 x float> %581, float %517, i64 9, !dbg !14
  %583 = insertelement <32 x float> %582, float %518, i64 10, !dbg !14
  %584 = insertelement <32 x float> %583, float %519, i64 11, !dbg !14
  %585 = insertelement <32 x float> %584, float %520, i64 12, !dbg !14
  %586 = insertelement <32 x float> %585, float %521, i64 13, !dbg !14
  %587 = insertelement <32 x float> %586, float %522, i64 14, !dbg !14
  %588 = insertelement <32 x float> %587, float %523, i64 15, !dbg !14
  %589 = insertelement <32 x float> %588, float %524, i64 16, !dbg !14
  %590 = insertelement <32 x float> %589, float %525, i64 17, !dbg !14
  %591 = insertelement <32 x float> %590, float %526, i64 18, !dbg !14
  %592 = insertelement <32 x float> %591, float %527, i64 19, !dbg !14
  %593 = insertelement <32 x float> %592, float %528, i64 20, !dbg !14
  %594 = insertelement <32 x float> %593, float %529, i64 21, !dbg !14
  %595 = insertelement <32 x float> %594, float %530, i64 22, !dbg !14
  %596 = insertelement <32 x float> %595, float %531, i64 23, !dbg !14
  %597 = insertelement <32 x float> %596, float %532, i64 24, !dbg !14
  %598 = insertelement <32 x float> %597, float %533, i64 25, !dbg !14
  %599 = insertelement <32 x float> %598, float %534, i64 26, !dbg !14
  %600 = insertelement <32 x float> %599, float %535, i64 27, !dbg !14
  %601 = insertelement <32 x float> %600, float %536, i64 28, !dbg !14
  %602 = insertelement <32 x float> %601, float %537, i64 29, !dbg !14
  %603 = insertelement <32 x float> %602, float %538, i64 30, !dbg !14
  %604 = insertelement <32 x float> %603, float %539, i64 31, !dbg !14
  %605 = insertvalue [2 x <32 x float>] %572, <32 x float> %604, 1, !dbg !14
  br label %606, !dbg !14

606:                                              ; preds = %436, %437
  %607 = phi [2 x <32 x float>] [ %605, %437 ], [ zeroinitializer, %436 ], !dbg !14
  br label %608, !dbg !14

608:                                              ; preds = %606
  %609 = extractvalue [2 x <32 x float>] %47, 0, !dbg !26
  %610 = extractvalue [2 x <32 x float>] %607, 0, !dbg !26
  %611 = shufflevector <32 x float> %609, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %612 = shufflevector <32 x float> %610, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %613 = fadd <4 x float> %611, %612, !dbg !26
  %614 = extractelement <4 x float> %613, i64 0, !dbg !27
  %615 = call float @llvm.amdgcn.exp2.f32(float %614), !dbg !27
  %616 = extractelement <4 x float> %613, i64 1, !dbg !27
  %617 = call float @llvm.amdgcn.exp2.f32(float %616), !dbg !27
  %618 = fadd float %615, %617, !dbg !28
  %619 = extractelement <4 x float> %613, i64 2, !dbg !27
  %620 = call float @llvm.amdgcn.exp2.f32(float %619), !dbg !27
  %621 = fadd float %618, %620, !dbg !28
  %622 = extractelement <4 x float> %613, i64 3, !dbg !27
  %623 = call float @llvm.amdgcn.exp2.f32(float %622), !dbg !27
  %624 = fadd float %621, %623, !dbg !28
  %625 = shufflevector <32 x float> %609, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %626 = shufflevector <32 x float> %610, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %627 = fadd <4 x float> %625, %626, !dbg !26
  %628 = extractelement <4 x float> %627, i64 0, !dbg !27
  %629 = call float @llvm.amdgcn.exp2.f32(float %628), !dbg !27
  %630 = fadd float %624, %629, !dbg !28
  %631 = extractelement <4 x float> %627, i64 1, !dbg !27
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !27
  %633 = fadd float %630, %632, !dbg !28
  %634 = extractelement <4 x float> %627, i64 2, !dbg !27
  %635 = call float @llvm.amdgcn.exp2.f32(float %634), !dbg !27
  %636 = fadd float %633, %635, !dbg !28
  %637 = extractelement <4 x float> %627, i64 3, !dbg !27
  %638 = call float @llvm.amdgcn.exp2.f32(float %637), !dbg !27
  %639 = fadd float %636, %638, !dbg !28
  %640 = shufflevector <32 x float> %609, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %641 = shufflevector <32 x float> %610, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %642 = fadd <4 x float> %640, %641, !dbg !26
  %643 = extractelement <4 x float> %642, i64 0, !dbg !27
  %644 = call float @llvm.amdgcn.exp2.f32(float %643), !dbg !27
  %645 = fadd float %639, %644, !dbg !28
  %646 = extractelement <4 x float> %642, i64 1, !dbg !27
  %647 = call float @llvm.amdgcn.exp2.f32(float %646), !dbg !27
  %648 = fadd float %645, %647, !dbg !28
  %649 = extractelement <4 x float> %642, i64 2, !dbg !27
  %650 = call float @llvm.amdgcn.exp2.f32(float %649), !dbg !27
  %651 = fadd float %648, %650, !dbg !28
  %652 = extractelement <4 x float> %642, i64 3, !dbg !27
  %653 = call float @llvm.amdgcn.exp2.f32(float %652), !dbg !27
  %654 = fadd float %651, %653, !dbg !28
  %655 = shufflevector <32 x float> %609, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %656 = shufflevector <32 x float> %610, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %657 = fadd <4 x float> %655, %656, !dbg !26
  %658 = extractelement <4 x float> %657, i64 0, !dbg !27
  %659 = call float @llvm.amdgcn.exp2.f32(float %658), !dbg !27
  %660 = fadd float %654, %659, !dbg !28
  %661 = extractelement <4 x float> %657, i64 1, !dbg !27
  %662 = call float @llvm.amdgcn.exp2.f32(float %661), !dbg !27
  %663 = fadd float %660, %662, !dbg !28
  %664 = extractelement <4 x float> %657, i64 2, !dbg !27
  %665 = call float @llvm.amdgcn.exp2.f32(float %664), !dbg !27
  %666 = fadd float %663, %665, !dbg !28
  %667 = extractelement <4 x float> %657, i64 3, !dbg !27
  %668 = call float @llvm.amdgcn.exp2.f32(float %667), !dbg !27
  %669 = fadd float %666, %668, !dbg !28
  %670 = shufflevector <32 x float> %609, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %671 = shufflevector <32 x float> %610, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %672 = fadd <4 x float> %670, %671, !dbg !26
  %673 = extractelement <4 x float> %672, i64 0, !dbg !27
  %674 = call float @llvm.amdgcn.exp2.f32(float %673), !dbg !27
  %675 = fadd float %669, %674, !dbg !28
  %676 = extractelement <4 x float> %672, i64 1, !dbg !27
  %677 = call float @llvm.amdgcn.exp2.f32(float %676), !dbg !27
  %678 = fadd float %675, %677, !dbg !28
  %679 = extractelement <4 x float> %672, i64 2, !dbg !27
  %680 = call float @llvm.amdgcn.exp2.f32(float %679), !dbg !27
  %681 = fadd float %678, %680, !dbg !28
  %682 = extractelement <4 x float> %672, i64 3, !dbg !27
  %683 = call float @llvm.amdgcn.exp2.f32(float %682), !dbg !27
  %684 = fadd float %681, %683, !dbg !28
  %685 = shufflevector <32 x float> %609, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %686 = shufflevector <32 x float> %610, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %687 = fadd <4 x float> %685, %686, !dbg !26
  %688 = extractelement <4 x float> %687, i64 0, !dbg !27
  %689 = call float @llvm.amdgcn.exp2.f32(float %688), !dbg !27
  %690 = fadd float %684, %689, !dbg !28
  %691 = extractelement <4 x float> %687, i64 1, !dbg !27
  %692 = call float @llvm.amdgcn.exp2.f32(float %691), !dbg !27
  %693 = fadd float %690, %692, !dbg !28
  %694 = extractelement <4 x float> %687, i64 2, !dbg !27
  %695 = call float @llvm.amdgcn.exp2.f32(float %694), !dbg !27
  %696 = fadd float %693, %695, !dbg !28
  %697 = extractelement <4 x float> %687, i64 3, !dbg !27
  %698 = call float @llvm.amdgcn.exp2.f32(float %697), !dbg !27
  %699 = fadd float %696, %698, !dbg !28
  %700 = shufflevector <32 x float> %609, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %701 = shufflevector <32 x float> %610, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %702 = fadd <4 x float> %700, %701, !dbg !26
  %703 = extractelement <4 x float> %702, i64 0, !dbg !27
  %704 = call float @llvm.amdgcn.exp2.f32(float %703), !dbg !27
  %705 = fadd float %699, %704, !dbg !28
  %706 = extractelement <4 x float> %702, i64 1, !dbg !27
  %707 = call float @llvm.amdgcn.exp2.f32(float %706), !dbg !27
  %708 = fadd float %705, %707, !dbg !28
  %709 = extractelement <4 x float> %702, i64 2, !dbg !27
  %710 = call float @llvm.amdgcn.exp2.f32(float %709), !dbg !27
  %711 = fadd float %708, %710, !dbg !28
  %712 = extractelement <4 x float> %702, i64 3, !dbg !27
  %713 = call float @llvm.amdgcn.exp2.f32(float %712), !dbg !27
  %714 = fadd float %711, %713, !dbg !28
  %715 = shufflevector <32 x float> %609, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %716 = shufflevector <32 x float> %610, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %717 = fadd <4 x float> %715, %716, !dbg !26
  %718 = extractelement <4 x float> %717, i64 0, !dbg !27
  %719 = call float @llvm.amdgcn.exp2.f32(float %718), !dbg !27
  %720 = fadd float %714, %719, !dbg !28
  %721 = extractelement <4 x float> %717, i64 1, !dbg !27
  %722 = call float @llvm.amdgcn.exp2.f32(float %721), !dbg !27
  %723 = fadd float %720, %722, !dbg !28
  %724 = extractelement <4 x float> %717, i64 2, !dbg !27
  %725 = call float @llvm.amdgcn.exp2.f32(float %724), !dbg !27
  %726 = fadd float %723, %725, !dbg !28
  %727 = extractelement <4 x float> %717, i64 3, !dbg !27
  %728 = call float @llvm.amdgcn.exp2.f32(float %727), !dbg !27
  %729 = fadd float %726, %728, !dbg !28
  %730 = extractvalue [2 x <32 x float>] %47, 1, !dbg !26
  %731 = extractvalue [2 x <32 x float>] %607, 1, !dbg !26
  %732 = shufflevector <32 x float> %730, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %733 = shufflevector <32 x float> %731, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %734 = fadd <4 x float> %732, %733, !dbg !26
  %735 = extractelement <4 x float> %734, i64 0, !dbg !27
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !27
  %737 = extractelement <4 x float> %734, i64 1, !dbg !27
  %738 = call float @llvm.amdgcn.exp2.f32(float %737), !dbg !27
  %739 = fadd float %736, %738, !dbg !28
  %740 = extractelement <4 x float> %734, i64 2, !dbg !27
  %741 = call float @llvm.amdgcn.exp2.f32(float %740), !dbg !27
  %742 = fadd float %739, %741, !dbg !28
  %743 = extractelement <4 x float> %734, i64 3, !dbg !27
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !27
  %745 = fadd float %742, %744, !dbg !28
  %746 = shufflevector <32 x float> %730, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %747 = shufflevector <32 x float> %731, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %748 = fadd <4 x float> %746, %747, !dbg !26
  %749 = extractelement <4 x float> %748, i64 0, !dbg !27
  %750 = call float @llvm.amdgcn.exp2.f32(float %749), !dbg !27
  %751 = fadd float %745, %750, !dbg !28
  %752 = extractelement <4 x float> %748, i64 1, !dbg !27
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !27
  %754 = fadd float %751, %753, !dbg !28
  %755 = extractelement <4 x float> %748, i64 2, !dbg !27
  %756 = call float @llvm.amdgcn.exp2.f32(float %755), !dbg !27
  %757 = fadd float %754, %756, !dbg !28
  %758 = extractelement <4 x float> %748, i64 3, !dbg !27
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !27
  %760 = fadd float %757, %759, !dbg !28
  %761 = shufflevector <32 x float> %730, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %762 = shufflevector <32 x float> %731, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %763 = fadd <4 x float> %761, %762, !dbg !26
  %764 = extractelement <4 x float> %763, i64 0, !dbg !27
  %765 = call float @llvm.amdgcn.exp2.f32(float %764), !dbg !27
  %766 = fadd float %760, %765, !dbg !28
  %767 = extractelement <4 x float> %763, i64 1, !dbg !27
  %768 = call float @llvm.amdgcn.exp2.f32(float %767), !dbg !27
  %769 = fadd float %766, %768, !dbg !28
  %770 = extractelement <4 x float> %763, i64 2, !dbg !27
  %771 = call float @llvm.amdgcn.exp2.f32(float %770), !dbg !27
  %772 = fadd float %769, %771, !dbg !28
  %773 = extractelement <4 x float> %763, i64 3, !dbg !27
  %774 = call float @llvm.amdgcn.exp2.f32(float %773), !dbg !27
  %775 = fadd float %772, %774, !dbg !28
  %776 = shufflevector <32 x float> %730, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %777 = shufflevector <32 x float> %731, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %778 = fadd <4 x float> %776, %777, !dbg !26
  %779 = extractelement <4 x float> %778, i64 0, !dbg !27
  %780 = call float @llvm.amdgcn.exp2.f32(float %779), !dbg !27
  %781 = fadd float %775, %780, !dbg !28
  %782 = extractelement <4 x float> %778, i64 1, !dbg !27
  %783 = call float @llvm.amdgcn.exp2.f32(float %782), !dbg !27
  %784 = fadd float %781, %783, !dbg !28
  %785 = extractelement <4 x float> %778, i64 2, !dbg !27
  %786 = call float @llvm.amdgcn.exp2.f32(float %785), !dbg !27
  %787 = fadd float %784, %786, !dbg !28
  %788 = extractelement <4 x float> %778, i64 3, !dbg !27
  %789 = call float @llvm.amdgcn.exp2.f32(float %788), !dbg !27
  %790 = fadd float %787, %789, !dbg !28
  %791 = shufflevector <32 x float> %730, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %792 = shufflevector <32 x float> %731, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %793 = fadd <4 x float> %791, %792, !dbg !26
  %794 = extractelement <4 x float> %793, i64 0, !dbg !27
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !27
  %796 = fadd float %790, %795, !dbg !28
  %797 = extractelement <4 x float> %793, i64 1, !dbg !27
  %798 = call float @llvm.amdgcn.exp2.f32(float %797), !dbg !27
  %799 = fadd float %796, %798, !dbg !28
  %800 = extractelement <4 x float> %793, i64 2, !dbg !27
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !27
  %802 = fadd float %799, %801, !dbg !28
  %803 = extractelement <4 x float> %793, i64 3, !dbg !27
  %804 = call float @llvm.amdgcn.exp2.f32(float %803), !dbg !27
  %805 = fadd float %802, %804, !dbg !28
  %806 = shufflevector <32 x float> %730, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %807 = shufflevector <32 x float> %731, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %808 = fadd <4 x float> %806, %807, !dbg !26
  %809 = extractelement <4 x float> %808, i64 0, !dbg !27
  %810 = call float @llvm.amdgcn.exp2.f32(float %809), !dbg !27
  %811 = fadd float %805, %810, !dbg !28
  %812 = extractelement <4 x float> %808, i64 1, !dbg !27
  %813 = call float @llvm.amdgcn.exp2.f32(float %812), !dbg !27
  %814 = fadd float %811, %813, !dbg !28
  %815 = extractelement <4 x float> %808, i64 2, !dbg !27
  %816 = call float @llvm.amdgcn.exp2.f32(float %815), !dbg !27
  %817 = fadd float %814, %816, !dbg !28
  %818 = extractelement <4 x float> %808, i64 3, !dbg !27
  %819 = call float @llvm.amdgcn.exp2.f32(float %818), !dbg !27
  %820 = fadd float %817, %819, !dbg !28
  %821 = shufflevector <32 x float> %730, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %822 = shufflevector <32 x float> %731, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %823 = fadd <4 x float> %821, %822, !dbg !26
  %824 = extractelement <4 x float> %823, i64 0, !dbg !27
  %825 = call float @llvm.amdgcn.exp2.f32(float %824), !dbg !27
  %826 = fadd float %820, %825, !dbg !28
  %827 = extractelement <4 x float> %823, i64 1, !dbg !27
  %828 = call float @llvm.amdgcn.exp2.f32(float %827), !dbg !27
  %829 = fadd float %826, %828, !dbg !28
  %830 = extractelement <4 x float> %823, i64 2, !dbg !27
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !27
  %832 = fadd float %829, %831, !dbg !28
  %833 = extractelement <4 x float> %823, i64 3, !dbg !27
  %834 = call float @llvm.amdgcn.exp2.f32(float %833), !dbg !27
  %835 = fadd float %832, %834, !dbg !28
  %836 = shufflevector <32 x float> %730, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %837 = shufflevector <32 x float> %731, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %838 = fadd <4 x float> %836, %837, !dbg !26
  %839 = extractelement <4 x float> %838, i64 0, !dbg !27
  %840 = call float @llvm.amdgcn.exp2.f32(float %839), !dbg !27
  %841 = fadd float %835, %840, !dbg !28
  %842 = extractelement <4 x float> %838, i64 1, !dbg !27
  %843 = call float @llvm.amdgcn.exp2.f32(float %842), !dbg !27
  %844 = fadd float %841, %843, !dbg !28
  %845 = extractelement <4 x float> %838, i64 2, !dbg !27
  %846 = call float @llvm.amdgcn.exp2.f32(float %845), !dbg !27
  %847 = fadd float %844, %846, !dbg !28
  %848 = extractelement <4 x float> %838, i64 3, !dbg !27
  %849 = call float @llvm.amdgcn.exp2.f32(float %848), !dbg !27
  %850 = fadd float %847, %849, !dbg !28
  %851 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %852 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %851), !dbg !28
  %853 = add nuw nsw i32 %852, 64, !dbg !28
  %854 = xor i32 %852, 16, !dbg !28
  %855 = and i32 %853, 192, !dbg !28
  %856 = icmp samesign ult i32 %854, %855, !dbg !28
  %857 = select i1 %856, i32 %854, i32 %852, !dbg !28
  %858 = shl nuw nsw i32 %857, 2, !dbg !28
  %859 = bitcast float %729 to i32, !dbg !28
  %860 = call i32 @llvm.amdgcn.ds.bpermute(i32 %858, i32 %859), !dbg !28
  %861 = bitcast i32 %860 to float, !dbg !28
  %862 = fadd float %729, %861, !dbg !28
  %863 = xor i32 %852, 32, !dbg !28
  %864 = icmp samesign ult i32 %863, %855, !dbg !28
  %865 = select i1 %864, i32 %863, i32 %852, !dbg !28
  %866 = shl nuw nsw i32 %865, 2, !dbg !28
  %867 = bitcast float %862 to i32, !dbg !28
  %868 = call i32 @llvm.amdgcn.ds.bpermute(i32 %866, i32 %867), !dbg !28
  %869 = bitcast i32 %868 to float, !dbg !28
  %870 = fadd float %862, %869, !dbg !28
  %871 = extractvalue [2 x <1 x float>] %38, 0, !dbg !28
  %872 = extractelement <1 x float> %871, i64 0, !dbg !28
  %873 = fadd float %870, %872, !dbg !28
  %874 = bitcast float %850 to i32, !dbg !28
  %875 = call i32 @llvm.amdgcn.ds.bpermute(i32 %858, i32 %874), !dbg !28
  %876 = bitcast i32 %875 to float, !dbg !28
  %877 = fadd float %850, %876, !dbg !28
  %878 = bitcast float %877 to i32, !dbg !28
  %879 = call i32 @llvm.amdgcn.ds.bpermute(i32 %866, i32 %878), !dbg !28
  %880 = bitcast i32 %879 to float, !dbg !28
  %881 = fadd float %877, %880, !dbg !28
  %882 = extractvalue [2 x <1 x float>] %38, 1, !dbg !28
  %883 = extractelement <1 x float> %882, i64 0, !dbg !28
  %884 = fadd float %881, %883, !dbg !28
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %885 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %886 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %885, i32 0) #0, !dbg !15
  %887 = bitcast <4 x i32> %886 to <8 x half>, !dbg !15
  %888 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %889 = getelementptr i8, ptr addrspace(3) %888, i64 8192, !dbg !15
  %890 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %889, i32 0) #0, !dbg !15
  %891 = bitcast <4 x i32> %890 to <8 x half>, !dbg !15
  %892 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %893 = getelementptr i8, ptr addrspace(3) %892, i64 256, !dbg !15
  %894 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %893, i32 0) #0, !dbg !15
  %895 = bitcast <4 x i32> %894 to <8 x half>, !dbg !15
  %896 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %897 = getelementptr i8, ptr addrspace(3) %896, i64 8448, !dbg !15
  %898 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %897, i32 0) #0, !dbg !15
  %899 = bitcast <4 x i32> %898 to <8 x half>, !dbg !15
  %900 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %901 = getelementptr i8, ptr addrspace(3) %900, i64 512, !dbg !15
  %902 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %901, i32 0) #0, !dbg !15
  %903 = bitcast <4 x i32> %902 to <8 x half>, !dbg !15
  %904 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %905 = getelementptr i8, ptr addrspace(3) %904, i64 8704, !dbg !15
  %906 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %905, i32 0) #0, !dbg !15
  %907 = bitcast <4 x i32> %906 to <8 x half>, !dbg !15
  %908 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %909 = getelementptr i8, ptr addrspace(3) %908, i64 768, !dbg !15
  %910 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %909, i32 0) #0, !dbg !15
  %911 = bitcast <4 x i32> %910 to <8 x half>, !dbg !15
  %912 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %913 = getelementptr i8, ptr addrspace(3) %912, i64 8960, !dbg !15
  %914 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %913, i32 0) #0, !dbg !15
  %915 = bitcast <4 x i32> %914 to <8 x half>, !dbg !15
  %916 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %917 = getelementptr i8, ptr addrspace(3) %916, i64 1024, !dbg !15
  %918 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %917, i32 0) #0, !dbg !15
  %919 = bitcast <4 x i32> %918 to <8 x half>, !dbg !15
  %920 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %921 = getelementptr i8, ptr addrspace(3) %920, i64 9216, !dbg !15
  %922 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %921, i32 0) #0, !dbg !15
  %923 = bitcast <4 x i32> %922 to <8 x half>, !dbg !15
  %924 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %925 = getelementptr i8, ptr addrspace(3) %924, i64 1280, !dbg !15
  %926 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %925, i32 0) #0, !dbg !15
  %927 = bitcast <4 x i32> %926 to <8 x half>, !dbg !15
  %928 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %929 = getelementptr i8, ptr addrspace(3) %928, i64 9472, !dbg !15
  %930 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %929, i32 0) #0, !dbg !15
  %931 = bitcast <4 x i32> %930 to <8 x half>, !dbg !15
  %932 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %933 = getelementptr i8, ptr addrspace(3) %932, i64 1536, !dbg !15
  %934 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %933, i32 0) #0, !dbg !15
  %935 = bitcast <4 x i32> %934 to <8 x half>, !dbg !15
  %936 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %937 = getelementptr i8, ptr addrspace(3) %936, i64 9728, !dbg !15
  %938 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %937, i32 0) #0, !dbg !15
  %939 = bitcast <4 x i32> %938 to <8 x half>, !dbg !15
  %940 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %941 = getelementptr i8, ptr addrspace(3) %940, i64 1792, !dbg !15
  %942 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %941, i32 0) #0, !dbg !15
  %943 = bitcast <4 x i32> %942 to <8 x half>, !dbg !15
  %944 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %31, !dbg !15
  %945 = getelementptr i8, ptr addrspace(3) %944, i64 9984, !dbg !15
  %946 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %945, i32 0) #0, !dbg !15
  %947 = bitcast <4 x i32> %946 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  br i1 true, label %948, label %1093, !dbg !15

948:                                              ; preds = %608
  %949 = insertelement <4 x float> poison, float %840, i64 0, !dbg !27
  %950 = insertelement <4 x float> %949, float %843, i64 1, !dbg !27
  %951 = insertelement <4 x float> %950, float %846, i64 2, !dbg !27
  %952 = insertelement <4 x float> %951, float %849, i64 3, !dbg !27
  %953 = fptrunc <4 x float> %952 to <4 x half>, !dbg !29
  %954 = extractelement <4 x half> %953, i64 3, !dbg !29
  %955 = extractelement <4 x half> %953, i64 2, !dbg !29
  %956 = extractelement <4 x half> %953, i64 1, !dbg !29
  %957 = extractelement <4 x half> %953, i64 0, !dbg !29
  %958 = insertelement <4 x float> poison, float %825, i64 0, !dbg !27
  %959 = insertelement <4 x float> %958, float %828, i64 1, !dbg !27
  %960 = insertelement <4 x float> %959, float %831, i64 2, !dbg !27
  %961 = insertelement <4 x float> %960, float %834, i64 3, !dbg !27
  %962 = fptrunc <4 x float> %961 to <4 x half>, !dbg !29
  %963 = extractelement <4 x half> %962, i64 3, !dbg !29
  %964 = extractelement <4 x half> %962, i64 2, !dbg !29
  %965 = extractelement <4 x half> %962, i64 1, !dbg !29
  %966 = extractelement <4 x half> %962, i64 0, !dbg !29
  %967 = insertelement <4 x float> poison, float %810, i64 0, !dbg !27
  %968 = insertelement <4 x float> %967, float %813, i64 1, !dbg !27
  %969 = insertelement <4 x float> %968, float %816, i64 2, !dbg !27
  %970 = insertelement <4 x float> %969, float %819, i64 3, !dbg !27
  %971 = fptrunc <4 x float> %970 to <4 x half>, !dbg !29
  %972 = extractelement <4 x half> %971, i64 3, !dbg !29
  %973 = extractelement <4 x half> %971, i64 2, !dbg !29
  %974 = extractelement <4 x half> %971, i64 1, !dbg !29
  %975 = extractelement <4 x half> %971, i64 0, !dbg !29
  %976 = insertelement <4 x float> poison, float %795, i64 0, !dbg !27
  %977 = insertelement <4 x float> %976, float %798, i64 1, !dbg !27
  %978 = insertelement <4 x float> %977, float %801, i64 2, !dbg !27
  %979 = insertelement <4 x float> %978, float %804, i64 3, !dbg !27
  %980 = fptrunc <4 x float> %979 to <4 x half>, !dbg !29
  %981 = extractelement <4 x half> %980, i64 3, !dbg !29
  %982 = extractelement <4 x half> %980, i64 2, !dbg !29
  %983 = extractelement <4 x half> %980, i64 1, !dbg !29
  %984 = extractelement <4 x half> %980, i64 0, !dbg !29
  %985 = insertelement <4 x float> poison, float %780, i64 0, !dbg !27
  %986 = insertelement <4 x float> %985, float %783, i64 1, !dbg !27
  %987 = insertelement <4 x float> %986, float %786, i64 2, !dbg !27
  %988 = insertelement <4 x float> %987, float %789, i64 3, !dbg !27
  %989 = fptrunc <4 x float> %988 to <4 x half>, !dbg !29
  %990 = extractelement <4 x half> %989, i64 3, !dbg !29
  %991 = extractelement <4 x half> %989, i64 2, !dbg !29
  %992 = extractelement <4 x half> %989, i64 1, !dbg !29
  %993 = extractelement <4 x half> %989, i64 0, !dbg !29
  %994 = insertelement <4 x float> poison, float %765, i64 0, !dbg !27
  %995 = insertelement <4 x float> %994, float %768, i64 1, !dbg !27
  %996 = insertelement <4 x float> %995, float %771, i64 2, !dbg !27
  %997 = insertelement <4 x float> %996, float %774, i64 3, !dbg !27
  %998 = fptrunc <4 x float> %997 to <4 x half>, !dbg !29
  %999 = extractelement <4 x half> %998, i64 3, !dbg !29
  %1000 = extractelement <4 x half> %998, i64 2, !dbg !29
  %1001 = extractelement <4 x half> %998, i64 1, !dbg !29
  %1002 = extractelement <4 x half> %998, i64 0, !dbg !29
  %1003 = insertelement <4 x float> poison, float %750, i64 0, !dbg !27
  %1004 = insertelement <4 x float> %1003, float %753, i64 1, !dbg !27
  %1005 = insertelement <4 x float> %1004, float %756, i64 2, !dbg !27
  %1006 = insertelement <4 x float> %1005, float %759, i64 3, !dbg !27
  %1007 = fptrunc <4 x float> %1006 to <4 x half>, !dbg !29
  %1008 = extractelement <4 x half> %1007, i64 3, !dbg !29
  %1009 = extractelement <4 x half> %1007, i64 2, !dbg !29
  %1010 = extractelement <4 x half> %1007, i64 1, !dbg !29
  %1011 = extractelement <4 x half> %1007, i64 0, !dbg !29
  %1012 = insertelement <4 x float> poison, float %736, i64 0, !dbg !27
  %1013 = insertelement <4 x float> %1012, float %738, i64 1, !dbg !27
  %1014 = insertelement <4 x float> %1013, float %741, i64 2, !dbg !27
  %1015 = insertelement <4 x float> %1014, float %744, i64 3, !dbg !27
  %1016 = fptrunc <4 x float> %1015 to <4 x half>, !dbg !29
  %1017 = extractelement <4 x half> %1016, i64 3, !dbg !29
  %1018 = extractelement <4 x half> %1016, i64 2, !dbg !29
  %1019 = extractelement <4 x half> %1016, i64 1, !dbg !29
  %1020 = extractelement <4 x half> %1016, i64 0, !dbg !29
  %1021 = insertelement <4 x float> poison, float %719, i64 0, !dbg !27
  %1022 = insertelement <4 x float> %1021, float %722, i64 1, !dbg !27
  %1023 = insertelement <4 x float> %1022, float %725, i64 2, !dbg !27
  %1024 = insertelement <4 x float> %1023, float %728, i64 3, !dbg !27
  %1025 = fptrunc <4 x float> %1024 to <4 x half>, !dbg !29
  %1026 = extractelement <4 x half> %1025, i64 3, !dbg !29
  %1027 = extractelement <4 x half> %1025, i64 2, !dbg !29
  %1028 = extractelement <4 x half> %1025, i64 1, !dbg !29
  %1029 = extractelement <4 x half> %1025, i64 0, !dbg !29
  %1030 = insertelement <4 x float> poison, float %704, i64 0, !dbg !27
  %1031 = insertelement <4 x float> %1030, float %707, i64 1, !dbg !27
  %1032 = insertelement <4 x float> %1031, float %710, i64 2, !dbg !27
  %1033 = insertelement <4 x float> %1032, float %713, i64 3, !dbg !27
  %1034 = fptrunc <4 x float> %1033 to <4 x half>, !dbg !29
  %1035 = extractelement <4 x half> %1034, i64 3, !dbg !29
  %1036 = extractelement <4 x half> %1034, i64 2, !dbg !29
  %1037 = extractelement <4 x half> %1034, i64 1, !dbg !29
  %1038 = extractelement <4 x half> %1034, i64 0, !dbg !29
  %1039 = insertelement <4 x float> poison, float %689, i64 0, !dbg !27
  %1040 = insertelement <4 x float> %1039, float %692, i64 1, !dbg !27
  %1041 = insertelement <4 x float> %1040, float %695, i64 2, !dbg !27
  %1042 = insertelement <4 x float> %1041, float %698, i64 3, !dbg !27
  %1043 = fptrunc <4 x float> %1042 to <4 x half>, !dbg !29
  %1044 = extractelement <4 x half> %1043, i64 3, !dbg !29
  %1045 = extractelement <4 x half> %1043, i64 2, !dbg !29
  %1046 = extractelement <4 x half> %1043, i64 1, !dbg !29
  %1047 = extractelement <4 x half> %1043, i64 0, !dbg !29
  %1048 = insertelement <4 x float> poison, float %674, i64 0, !dbg !27
  %1049 = insertelement <4 x float> %1048, float %677, i64 1, !dbg !27
  %1050 = insertelement <4 x float> %1049, float %680, i64 2, !dbg !27
  %1051 = insertelement <4 x float> %1050, float %683, i64 3, !dbg !27
  %1052 = fptrunc <4 x float> %1051 to <4 x half>, !dbg !29
  %1053 = extractelement <4 x half> %1052, i64 3, !dbg !29
  %1054 = extractelement <4 x half> %1052, i64 2, !dbg !29
  %1055 = extractelement <4 x half> %1052, i64 1, !dbg !29
  %1056 = extractelement <4 x half> %1052, i64 0, !dbg !29
  %1057 = insertelement <4 x float> poison, float %659, i64 0, !dbg !27
  %1058 = insertelement <4 x float> %1057, float %662, i64 1, !dbg !27
  %1059 = insertelement <4 x float> %1058, float %665, i64 2, !dbg !27
  %1060 = insertelement <4 x float> %1059, float %668, i64 3, !dbg !27
  %1061 = fptrunc <4 x float> %1060 to <4 x half>, !dbg !29
  %1062 = extractelement <4 x half> %1061, i64 3, !dbg !29
  %1063 = extractelement <4 x half> %1061, i64 2, !dbg !29
  %1064 = extractelement <4 x half> %1061, i64 1, !dbg !29
  %1065 = extractelement <4 x half> %1061, i64 0, !dbg !29
  %1066 = insertelement <4 x float> poison, float %644, i64 0, !dbg !27
  %1067 = insertelement <4 x float> %1066, float %647, i64 1, !dbg !27
  %1068 = insertelement <4 x float> %1067, float %650, i64 2, !dbg !27
  %1069 = insertelement <4 x float> %1068, float %653, i64 3, !dbg !27
  %1070 = fptrunc <4 x float> %1069 to <4 x half>, !dbg !29
  %1071 = extractelement <4 x half> %1070, i64 3, !dbg !29
  %1072 = extractelement <4 x half> %1070, i64 2, !dbg !29
  %1073 = extractelement <4 x half> %1070, i64 1, !dbg !29
  %1074 = extractelement <4 x half> %1070, i64 0, !dbg !29
  %1075 = insertelement <4 x float> poison, float %629, i64 0, !dbg !27
  %1076 = insertelement <4 x float> %1075, float %632, i64 1, !dbg !27
  %1077 = insertelement <4 x float> %1076, float %635, i64 2, !dbg !27
  %1078 = insertelement <4 x float> %1077, float %638, i64 3, !dbg !27
  %1079 = fptrunc <4 x float> %1078 to <4 x half>, !dbg !29
  %1080 = extractelement <4 x half> %1079, i64 3, !dbg !29
  %1081 = extractelement <4 x half> %1079, i64 2, !dbg !29
  %1082 = extractelement <4 x half> %1079, i64 1, !dbg !29
  %1083 = extractelement <4 x half> %1079, i64 0, !dbg !29
  %1084 = insertelement <4 x float> poison, float %615, i64 0, !dbg !27
  %1085 = insertelement <4 x float> %1084, float %617, i64 1, !dbg !27
  %1086 = insertelement <4 x float> %1085, float %620, i64 2, !dbg !27
  %1087 = insertelement <4 x float> %1086, float %623, i64 3, !dbg !27
  %1088 = fptrunc <4 x float> %1087 to <4 x half>, !dbg !29
  %1089 = extractelement <4 x half> %1088, i64 3, !dbg !29
  %1090 = extractelement <4 x half> %1088, i64 2, !dbg !29
  %1091 = extractelement <4 x half> %1088, i64 1, !dbg !29
  %1092 = extractelement <4 x half> %1088, i64 0, !dbg !29
  %address.thread448 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !15, !range !34, !frisk.register_lifetimes !1
  %address.local449 = shl nuw nsw i32 %address.thread448, 5, !dbg !15
  %address.local450 = and i32 %address.local449, 6144, !dbg !15
  %address.local451 = shl nuw nsw i32 %address.thread448, 7, !dbg !15
  %address.local452 = and i32 %address.local451, 1920, !dbg !15
  %address.local453 = or disjoint i32 %address.local450, %address.local452, !dbg !15
  %address.local454 = zext nneg i32 %address.local453 to i64, !dbg !15
  %address.local455 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local456 = zext nneg i32 %address.thread448 to i64, !dbg !13
  %address.local457 = and i64 %address.local456, 48, !dbg !15
  %address.local458 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local455, i64 %address.local457, !dbg !15
  store half %1092, ptr addrspace(3) %address.local458, align 2, !dbg !15
  %address.local459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local460 = or disjoint i64 %address.local457, 1, !dbg !15
  %address.local461 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local459, i64 %address.local460, !dbg !15
  store half %1091, ptr addrspace(3) %address.local461, align 2, !dbg !15
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local463 = or disjoint i64 %address.local457, 2, !dbg !15
  %address.local464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local462, i64 %address.local463, !dbg !15
  store half %1090, ptr addrspace(3) %address.local464, align 2, !dbg !15
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local466 = or disjoint i64 %address.local457, 3, !dbg !15
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local465, i64 %address.local466, !dbg !15
  store half %1089, ptr addrspace(3) %address.local467, align 2, !dbg !15
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local469 = or disjoint i64 %address.local457, 4, !dbg !15
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local468, i64 %address.local469, !dbg !15
  store half %1083, ptr addrspace(3) %address.local470, align 2, !dbg !15
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local472 = or disjoint i64 %address.local457, 5, !dbg !15
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local471, i64 %address.local472, !dbg !15
  store half %1082, ptr addrspace(3) %address.local473, align 2, !dbg !15
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local475 = or disjoint i64 %address.local457, 6, !dbg !15
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local474, i64 %address.local475, !dbg !15
  store half %1081, ptr addrspace(3) %address.local476, align 2, !dbg !15
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local478 = or disjoint i64 %address.local457, 7, !dbg !15
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local477, i64 %address.local478, !dbg !15
  store half %1080, ptr addrspace(3) %address.local479, align 2, !dbg !15
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local481 = or disjoint i64 %address.local457, 8, !dbg !15
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local480, i64 %address.local481, !dbg !15
  store half %1074, ptr addrspace(3) %address.local482, align 2, !dbg !15
  %address.local483 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local484 = or disjoint i64 %address.local457, 9, !dbg !15
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local483, i64 %address.local484, !dbg !15
  store half %1073, ptr addrspace(3) %address.local485, align 2, !dbg !15
  %address.local486 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local487 = or disjoint i64 %address.local457, 10, !dbg !15
  %address.local488 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local486, i64 %address.local487, !dbg !15
  store half %1072, ptr addrspace(3) %address.local488, align 2, !dbg !15
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local490 = or disjoint i64 %address.local457, 11, !dbg !15
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local489, i64 %address.local490, !dbg !15
  store half %1071, ptr addrspace(3) %address.local491, align 2, !dbg !15
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local493 = or disjoint i64 %address.local457, 12, !dbg !15
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local492, i64 %address.local493, !dbg !15
  store half %1065, ptr addrspace(3) %address.local494, align 2, !dbg !15
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local496 = or disjoint i64 %address.local457, 13, !dbg !15
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local495, i64 %address.local496, !dbg !15
  store half %1064, ptr addrspace(3) %address.local497, align 2, !dbg !15
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local499 = or disjoint i64 %address.local457, 14, !dbg !15
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local498, i64 %address.local499, !dbg !15
  store half %1063, ptr addrspace(3) %address.local500, align 2, !dbg !15
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local502 = or disjoint i64 %address.local457, 15, !dbg !15
  %address.local503 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local501, i64 %address.local502, !dbg !15
  store half %1062, ptr addrspace(3) %address.local503, align 2, !dbg !15
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local505 = or disjoint i64 %address.local457, 64, !dbg !15
  %address.local506 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local504, i64 %address.local505, !dbg !15
  store half %1056, ptr addrspace(3) %address.local506, align 2, !dbg !15
  %address.local507 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local508 = or disjoint i64 %address.local457, 65, !dbg !15
  %address.local509 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local507, i64 %address.local508, !dbg !15
  store half %1055, ptr addrspace(3) %address.local509, align 2, !dbg !15
  %address.local510 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local511 = or disjoint i64 %address.local457, 66, !dbg !15
  %address.local512 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local510, i64 %address.local511, !dbg !15
  store half %1054, ptr addrspace(3) %address.local512, align 2, !dbg !15
  %address.local513 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local514 = or disjoint i64 %address.local457, 67, !dbg !15
  %address.local515 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local513, i64 %address.local514, !dbg !15
  store half %1053, ptr addrspace(3) %address.local515, align 2, !dbg !15
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local517 = or disjoint i64 %address.local457, 68, !dbg !15
  %address.local518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local516, i64 %address.local517, !dbg !15
  store half %1047, ptr addrspace(3) %address.local518, align 2, !dbg !15
  %address.local519 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local520 = or disjoint i64 %address.local457, 69, !dbg !15
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local519, i64 %address.local520, !dbg !15
  store half %1046, ptr addrspace(3) %address.local521, align 2, !dbg !15
  %address.local522 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local523 = or disjoint i64 %address.local457, 70, !dbg !15
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local522, i64 %address.local523, !dbg !15
  store half %1045, ptr addrspace(3) %address.local524, align 2, !dbg !15
  %address.local525 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local526 = or disjoint i64 %address.local457, 71, !dbg !15
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local525, i64 %address.local526, !dbg !15
  store half %1044, ptr addrspace(3) %address.local527, align 2, !dbg !15
  %address.local528 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local529 = or disjoint i64 %address.local457, 72, !dbg !15
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local528, i64 %address.local529, !dbg !15
  store half %1038, ptr addrspace(3) %address.local530, align 2, !dbg !15
  %address.local531 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local532 = or disjoint i64 %address.local457, 73, !dbg !15
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local531, i64 %address.local532, !dbg !15
  store half %1037, ptr addrspace(3) %address.local533, align 2, !dbg !15
  %address.local534 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local535 = or disjoint i64 %address.local457, 74, !dbg !15
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local534, i64 %address.local535, !dbg !15
  store half %1036, ptr addrspace(3) %address.local536, align 2, !dbg !15
  %address.local537 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local538 = or disjoint i64 %address.local457, 75, !dbg !15
  %address.local539 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local537, i64 %address.local538, !dbg !15
  store half %1035, ptr addrspace(3) %address.local539, align 2, !dbg !15
  %address.local540 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local541 = or disjoint i64 %address.local457, 76, !dbg !15
  %address.local542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local540, i64 %address.local541, !dbg !15
  store half %1029, ptr addrspace(3) %address.local542, align 2, !dbg !15
  %address.local543 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local544 = or disjoint i64 %address.local457, 77, !dbg !15
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local543, i64 %address.local544, !dbg !15
  store half %1028, ptr addrspace(3) %address.local545, align 2, !dbg !15
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local547 = or disjoint i64 %address.local457, 78, !dbg !15
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local546, i64 %address.local547, !dbg !15
  store half %1027, ptr addrspace(3) %address.local548, align 2, !dbg !15
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local454, !dbg !15
  %address.local550 = or disjoint i64 %address.local457, 79, !dbg !15
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local549, i64 %address.local550, !dbg !15
  store half %1026, ptr addrspace(3) %address.local551, align 2, !dbg !15
  %address.local552 = shl nuw nsw i32 %address.thread448, 5, !dbg !15
  %address.local553 = and i32 %address.local552, 6144, !dbg !15
  %address.local554 = shl nuw nsw i32 %address.thread448, 7, !dbg !15
  %address.local555 = and i32 %address.local554, 1920, !dbg !15
  %address.local556 = or disjoint i32 %address.local553, %address.local555, !dbg !15
  %address.local557 = or disjoint i32 %address.local556, 8192, !dbg !15
  %address.local558 = zext nneg i32 %address.local557 to i64, !dbg !15
  %address.local559 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local559, i64 %address.local457, !dbg !15
  store half %1020, ptr addrspace(3) %address.local560, align 2, !dbg !15
  %address.local561 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local562 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local561, i64 %address.local460, !dbg !15
  store half %1019, ptr addrspace(3) %address.local562, align 2, !dbg !15
  %address.local563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local564 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local563, i64 %address.local463, !dbg !15
  store half %1018, ptr addrspace(3) %address.local564, align 2, !dbg !15
  %address.local565 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local566 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local565, i64 %address.local466, !dbg !15
  store half %1017, ptr addrspace(3) %address.local566, align 2, !dbg !15
  %address.local567 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local568 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local567, i64 %address.local469, !dbg !15
  store half %1011, ptr addrspace(3) %address.local568, align 2, !dbg !15
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local570 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local569, i64 %address.local472, !dbg !15
  store half %1010, ptr addrspace(3) %address.local570, align 2, !dbg !15
  %address.local571 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local571, i64 %address.local475, !dbg !15
  store half %1009, ptr addrspace(3) %address.local572, align 2, !dbg !15
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local573, i64 %address.local478, !dbg !15
  store half %1008, ptr addrspace(3) %address.local574, align 2, !dbg !15
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local575, i64 %address.local481, !dbg !15
  store half %1002, ptr addrspace(3) %address.local576, align 2, !dbg !15
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local577, i64 %address.local484, !dbg !15
  store half %1001, ptr addrspace(3) %address.local578, align 2, !dbg !15
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local579, i64 %address.local487, !dbg !15
  store half %1000, ptr addrspace(3) %address.local580, align 2, !dbg !15
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local581, i64 %address.local490, !dbg !15
  store half %999, ptr addrspace(3) %address.local582, align 2, !dbg !15
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local583, i64 %address.local493, !dbg !15
  store half %993, ptr addrspace(3) %address.local584, align 2, !dbg !15
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local585, i64 %address.local496, !dbg !15
  store half %992, ptr addrspace(3) %address.local586, align 2, !dbg !15
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local587, i64 %address.local499, !dbg !15
  store half %991, ptr addrspace(3) %address.local588, align 2, !dbg !15
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local590 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local589, i64 %address.local502, !dbg !15
  store half %990, ptr addrspace(3) %address.local590, align 2, !dbg !15
  %address.local591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local591, i64 %address.local505, !dbg !15
  store half %984, ptr addrspace(3) %address.local592, align 2, !dbg !15
  %address.local593 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local593, i64 %address.local508, !dbg !15
  store half %983, ptr addrspace(3) %address.local594, align 2, !dbg !15
  %address.local595 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local596 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local595, i64 %address.local511, !dbg !15
  store half %982, ptr addrspace(3) %address.local596, align 2, !dbg !15
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local598 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local597, i64 %address.local514, !dbg !15
  store half %981, ptr addrspace(3) %address.local598, align 2, !dbg !15
  %address.local599 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local599, i64 %address.local517, !dbg !15
  store half %975, ptr addrspace(3) %address.local600, align 2, !dbg !15
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local601, i64 %address.local520, !dbg !15
  store half %974, ptr addrspace(3) %address.local602, align 2, !dbg !15
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local603, i64 %address.local523, !dbg !15
  store half %973, ptr addrspace(3) %address.local604, align 2, !dbg !15
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local605, i64 %address.local526, !dbg !15
  store half %972, ptr addrspace(3) %address.local606, align 2, !dbg !15
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local607, i64 %address.local529, !dbg !15
  store half %966, ptr addrspace(3) %address.local608, align 2, !dbg !15
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local609, i64 %address.local532, !dbg !15
  store half %965, ptr addrspace(3) %address.local610, align 2, !dbg !15
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local611, i64 %address.local535, !dbg !15
  store half %964, ptr addrspace(3) %address.local612, align 2, !dbg !15
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local613, i64 %address.local538, !dbg !15
  store half %963, ptr addrspace(3) %address.local614, align 2, !dbg !15
  %address.local615 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local615, i64 %address.local541, !dbg !15
  store half %957, ptr addrspace(3) %address.local616, align 2, !dbg !15
  %address.local617 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local618 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local617, i64 %address.local544, !dbg !15
  store half %956, ptr addrspace(3) %address.local618, align 2, !dbg !15
  %address.local619 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local620 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local619, i64 %address.local547, !dbg !15
  store half %955, ptr addrspace(3) %address.local620, align 2, !dbg !15
  %address.local621 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local558, !dbg !15
  %address.local622 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local621, i64 %address.local550, !dbg !15
  store half %954, ptr addrspace(3) %address.local622, align 2, !dbg !15
  br label %1093, !dbg !15

1093:                                             ; preds = %948, %608
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.thread623 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !15, !range !34, !frisk.register_lifetimes !1
  %address.local624 = shl nuw nsw i32 %address.thread623, 5, !dbg !15
  %address.local625 = and i32 %address.local624, 6144, !dbg !15
  %address.local626 = shl nuw nsw i32 %address.thread623, 7, !dbg !15
  %address.local627 = and i32 %address.local626, 1920, !dbg !15
  %address.local628 = or disjoint i32 %address.local625, %address.local627, !dbg !15
  %address.local629 = zext nneg i32 %address.local628 to i64, !dbg !15
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local631 = zext nneg i32 %address.thread623 to i64, !dbg !13
  %address.local632 = lshr i64 %address.local631, 2, !dbg !13
  %address.local633 = and i64 %address.local632, 12, !dbg !13
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local630, i64 %address.local633, !dbg !15
  %1094 = load half, ptr addrspace(3) %address.local634, align 2, !dbg !15
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local636 = or disjoint i64 %address.local633, 1, !dbg !15
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local635, i64 %address.local636, !dbg !15
  %1095 = load half, ptr addrspace(3) %address.local637, align 2, !dbg !15
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local639 = or disjoint i64 %address.local633, 2, !dbg !15
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local638, i64 %address.local639, !dbg !15
  %1096 = load half, ptr addrspace(3) %address.local640, align 2, !dbg !15
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local642 = or disjoint i64 %address.local633, 3, !dbg !15
  %address.local643 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local641, i64 %address.local642, !dbg !15
  %1097 = load half, ptr addrspace(3) %address.local643, align 2, !dbg !15
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local645 = or disjoint i64 %address.local633, 16, !dbg !15
  %address.local646 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local644, i64 %address.local645, !dbg !15
  %1098 = load half, ptr addrspace(3) %address.local646, align 2, !dbg !15
  %address.local647 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local648 = or disjoint i64 %address.local633, 17, !dbg !15
  %address.local649 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local647, i64 %address.local648, !dbg !15
  %1099 = load half, ptr addrspace(3) %address.local649, align 2, !dbg !15
  %address.local650 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local651 = or disjoint i64 %address.local633, 18, !dbg !15
  %address.local652 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local650, i64 %address.local651, !dbg !15
  %1100 = load half, ptr addrspace(3) %address.local652, align 2, !dbg !15
  %address.local653 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local654 = or disjoint i64 %address.local633, 19, !dbg !15
  %address.local655 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local653, i64 %address.local654, !dbg !15
  %1101 = load half, ptr addrspace(3) %address.local655, align 2, !dbg !15
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local657 = or disjoint i64 %address.local633, 32, !dbg !15
  %address.local658 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local656, i64 %address.local657, !dbg !15
  %1102 = load half, ptr addrspace(3) %address.local658, align 2, !dbg !15
  %address.local659 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local660 = or disjoint i64 %address.local633, 33, !dbg !15
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local659, i64 %address.local660, !dbg !15
  %1103 = load half, ptr addrspace(3) %address.local661, align 2, !dbg !15
  %address.local662 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local663 = or disjoint i64 %address.local633, 34, !dbg !15
  %address.local664 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local662, i64 %address.local663, !dbg !15
  %1104 = load half, ptr addrspace(3) %address.local664, align 2, !dbg !15
  %address.local665 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local666 = or disjoint i64 %address.local633, 35, !dbg !15
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local665, i64 %address.local666, !dbg !15
  %1105 = load half, ptr addrspace(3) %address.local667, align 2, !dbg !15
  %address.local668 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local669 = or disjoint i64 %address.local633, 48, !dbg !15
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local668, i64 %address.local669, !dbg !15
  %1106 = load half, ptr addrspace(3) %address.local670, align 2, !dbg !15
  %address.local671 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local672 = or disjoint i64 %address.local633, 49, !dbg !15
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local671, i64 %address.local672, !dbg !15
  %1107 = load half, ptr addrspace(3) %address.local673, align 2, !dbg !15
  %address.local674 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local675 = or disjoint i64 %address.local633, 50, !dbg !15
  %address.local676 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local674, i64 %address.local675, !dbg !15
  %1108 = load half, ptr addrspace(3) %address.local676, align 2, !dbg !15
  %address.local677 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local678 = or i64 %address.local632, 51, !dbg !15
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local677, i64 %address.local678, !dbg !15
  %1109 = load half, ptr addrspace(3) %address.local679, align 2, !dbg !15
  %address.local680 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local681 = or disjoint i64 %address.local633, 64, !dbg !15
  %address.local682 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local680, i64 %address.local681, !dbg !15
  %1110 = load half, ptr addrspace(3) %address.local682, align 2, !dbg !15
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local684 = or disjoint i64 %address.local633, 65, !dbg !15
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local683, i64 %address.local684, !dbg !15
  %1111 = load half, ptr addrspace(3) %address.local685, align 2, !dbg !15
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local687 = or disjoint i64 %address.local633, 66, !dbg !15
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local686, i64 %address.local687, !dbg !15
  %1112 = load half, ptr addrspace(3) %address.local688, align 2, !dbg !15
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local690 = or disjoint i64 %address.local633, 67, !dbg !15
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local689, i64 %address.local690, !dbg !15
  %1113 = load half, ptr addrspace(3) %address.local691, align 2, !dbg !15
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local693 = or disjoint i64 %address.local633, 80, !dbg !15
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local692, i64 %address.local693, !dbg !15
  %1114 = load half, ptr addrspace(3) %address.local694, align 2, !dbg !15
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local696 = or disjoint i64 %address.local633, 81, !dbg !15
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local695, i64 %address.local696, !dbg !15
  %1115 = load half, ptr addrspace(3) %address.local697, align 2, !dbg !15
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local699 = or disjoint i64 %address.local633, 82, !dbg !15
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local698, i64 %address.local699, !dbg !15
  %1116 = load half, ptr addrspace(3) %address.local700, align 2, !dbg !15
  %address.local701 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local702 = or disjoint i64 %address.local633, 83, !dbg !15
  %address.local703 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local701, i64 %address.local702, !dbg !15
  %1117 = load half, ptr addrspace(3) %address.local703, align 2, !dbg !15
  %address.local704 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local705 = or disjoint i64 %address.local633, 96, !dbg !15
  %address.local706 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local704, i64 %address.local705, !dbg !15
  %1118 = load half, ptr addrspace(3) %address.local706, align 2, !dbg !15
  %address.local707 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local708 = or disjoint i64 %address.local633, 97, !dbg !15
  %address.local709 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local707, i64 %address.local708, !dbg !15
  %1119 = load half, ptr addrspace(3) %address.local709, align 2, !dbg !15
  %address.local710 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local711 = or disjoint i64 %address.local633, 98, !dbg !15
  %address.local712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local710, i64 %address.local711, !dbg !15
  %1120 = load half, ptr addrspace(3) %address.local712, align 2, !dbg !15
  %address.local713 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local714 = or disjoint i64 %address.local633, 99, !dbg !15
  %address.local715 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local713, i64 %address.local714, !dbg !15
  %1121 = load half, ptr addrspace(3) %address.local715, align 2, !dbg !15
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local717 = or disjoint i64 %address.local633, 112, !dbg !15
  %address.local718 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local716, i64 %address.local717, !dbg !15
  %1122 = load half, ptr addrspace(3) %address.local718, align 2, !dbg !15
  %address.local719 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local720 = or disjoint i64 %address.local633, 113, !dbg !15
  %address.local721 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local719, i64 %address.local720, !dbg !15
  %1123 = load half, ptr addrspace(3) %address.local721, align 2, !dbg !15
  %address.local722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local723 = or disjoint i64 %address.local633, 114, !dbg !15
  %address.local724 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local722, i64 %address.local723, !dbg !15
  %1124 = load half, ptr addrspace(3) %address.local724, align 2, !dbg !15
  %address.local725 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local629, !dbg !15
  %address.local726 = or i64 %address.local632, 115, !dbg !15
  %address.local727 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local725, i64 %address.local726, !dbg !15
  %1125 = load half, ptr addrspace(3) %address.local727, align 2, !dbg !15
  %address.local728 = shl nuw nsw i32 %address.thread623, 5, !dbg !15
  %address.local729 = and i32 %address.local728, 6144, !dbg !15
  %address.local730 = shl nuw nsw i32 %address.thread623, 7, !dbg !15
  %address.local731 = and i32 %address.local730, 1920, !dbg !15
  %address.local732 = or disjoint i32 %address.local729, %address.local731, !dbg !15
  %address.local733 = or disjoint i32 %address.local732, 8192, !dbg !15
  %address.local734 = zext nneg i32 %address.local733 to i64, !dbg !15
  %address.local735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local736 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local735, i64 %address.local633, !dbg !15
  %1126 = load half, ptr addrspace(3) %address.local736, align 2, !dbg !15
  %address.local737 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local738 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local737, i64 %address.local636, !dbg !15
  %1127 = load half, ptr addrspace(3) %address.local738, align 2, !dbg !15
  %address.local739 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local740 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local739, i64 %address.local639, !dbg !15
  %1128 = load half, ptr addrspace(3) %address.local740, align 2, !dbg !15
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local741, i64 %address.local642, !dbg !15
  %1129 = load half, ptr addrspace(3) %address.local742, align 2, !dbg !15
  %address.local743 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local743, i64 %address.local645, !dbg !15
  %1130 = load half, ptr addrspace(3) %address.local744, align 2, !dbg !15
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local746 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local745, i64 %address.local648, !dbg !15
  %1131 = load half, ptr addrspace(3) %address.local746, align 2, !dbg !15
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local747, i64 %address.local651, !dbg !15
  %1132 = load half, ptr addrspace(3) %address.local748, align 2, !dbg !15
  %address.local749 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local749, i64 %address.local654, !dbg !15
  %1133 = load half, ptr addrspace(3) %address.local750, align 2, !dbg !15
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local751, i64 %address.local657, !dbg !15
  %1134 = load half, ptr addrspace(3) %address.local752, align 2, !dbg !15
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local754 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local753, i64 %address.local660, !dbg !15
  %1135 = load half, ptr addrspace(3) %address.local754, align 2, !dbg !15
  %address.local755 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local755, i64 %address.local663, !dbg !15
  %1136 = load half, ptr addrspace(3) %address.local756, align 2, !dbg !15
  %address.local757 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local758 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local757, i64 %address.local666, !dbg !15
  %1137 = load half, ptr addrspace(3) %address.local758, align 2, !dbg !15
  %address.local759 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local760 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local759, i64 %address.local669, !dbg !15
  %1138 = load half, ptr addrspace(3) %address.local760, align 2, !dbg !15
  %address.local761 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local762 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local761, i64 %address.local672, !dbg !15
  %1139 = load half, ptr addrspace(3) %address.local762, align 2, !dbg !15
  %address.local763 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local764 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local763, i64 %address.local675, !dbg !15
  %1140 = load half, ptr addrspace(3) %address.local764, align 2, !dbg !15
  %address.local765 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local766 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local765, i64 %address.local678, !dbg !15
  %1141 = load half, ptr addrspace(3) %address.local766, align 2, !dbg !15
  %address.local767 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local768 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local767, i64 %address.local681, !dbg !15
  %1142 = load half, ptr addrspace(3) %address.local768, align 2, !dbg !15
  %address.local769 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local770 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local769, i64 %address.local684, !dbg !15
  %1143 = load half, ptr addrspace(3) %address.local770, align 2, !dbg !15
  %address.local771 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local772 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local771, i64 %address.local687, !dbg !15
  %1144 = load half, ptr addrspace(3) %address.local772, align 2, !dbg !15
  %address.local773 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local774 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local773, i64 %address.local690, !dbg !15
  %1145 = load half, ptr addrspace(3) %address.local774, align 2, !dbg !15
  %address.local775 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local776 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local775, i64 %address.local693, !dbg !15
  %1146 = load half, ptr addrspace(3) %address.local776, align 2, !dbg !15
  %address.local777 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local778 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local777, i64 %address.local696, !dbg !15
  %1147 = load half, ptr addrspace(3) %address.local778, align 2, !dbg !15
  %address.local779 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local780 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local779, i64 %address.local699, !dbg !15
  %1148 = load half, ptr addrspace(3) %address.local780, align 2, !dbg !15
  %address.local781 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local782 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local781, i64 %address.local702, !dbg !15
  %1149 = load half, ptr addrspace(3) %address.local782, align 2, !dbg !15
  %address.local783 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local784 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local783, i64 %address.local705, !dbg !15
  %1150 = load half, ptr addrspace(3) %address.local784, align 2, !dbg !15
  %address.local785 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local786 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local785, i64 %address.local708, !dbg !15
  %1151 = load half, ptr addrspace(3) %address.local786, align 2, !dbg !15
  %address.local787 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local788 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local787, i64 %address.local711, !dbg !15
  %1152 = load half, ptr addrspace(3) %address.local788, align 2, !dbg !15
  %address.local789 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local790 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local789, i64 %address.local714, !dbg !15
  %1153 = load half, ptr addrspace(3) %address.local790, align 2, !dbg !15
  %address.local791 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local792 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local791, i64 %address.local717, !dbg !15
  %1154 = load half, ptr addrspace(3) %address.local792, align 2, !dbg !15
  %address.local793 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local794 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local793, i64 %address.local720, !dbg !15
  %1155 = load half, ptr addrspace(3) %address.local794, align 2, !dbg !15
  %address.local795 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local796 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local795, i64 %address.local723, !dbg !15
  %1156 = load half, ptr addrspace(3) %address.local796, align 2, !dbg !15
  %address.local797 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local734, !dbg !15
  %address.local798 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %address.local797, i64 %address.local726, !dbg !15
  %1157 = load half, ptr addrspace(3) %address.local798, align 2, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1158 = add nsw i64 %42, %8, !dbg !18
  %1159 = shl i64 %1158, 19, !dbg !18
  %1160 = and i64 %1159, 16252928, !dbg !18
  %1161 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1160, !dbg !18
  %1162 = getelementptr [2 x i8], ptr addrspace(1) %1161, i64 %374, !dbg !18
  %1163 = getelementptr i8, ptr addrspace(1) %1162, i64 128, !dbg !18
  %address.local799 = shl nuw nsw i64 %address.local631, 3, !dbg !12
  %address.local800 = and i64 %address.local799, 1536, !dbg !12
  %address.local801 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local800, !dbg !12
  %address.local802 = ptrtoint ptr addrspace(3) %address.local801 to i64, !dbg !12
  %address.local803 = trunc i64 %address.local802 to i32, !dbg !12
  %1164 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local803), !dbg !18
  %1165 = zext i32 %1164 to i64, !dbg !18
  %1166 = inttoptr i64 %1165 to ptr addrspace(3), !dbg !18
  %address.local804 = shl nuw nsw i64 %address.local631, 7, !dbg !12
  %address.local805 = and i64 %address.local804, 16256, !dbg !12
  %address.local806 = lshr i64 %address.local631, 4, !dbg !12
  %address.local807 = and i64 %address.local806, 8, !dbg !12
  %address.local808 = or disjoint i64 %address.local805, %address.local807, !dbg !12
  %address.local809 = getelementptr [2 x i8], ptr addrspace(1) %1163, i64 %address.local808, !dbg !18
  %1167 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1168 = ptrtoint ptr addrspace(1) %address.local809 to i64, !dbg !18
  %1169 = sub i64 %1168, %1167, !dbg !18
  %dma.buffer.offset1069 = trunc i64 %1169 to i32, !dbg !18
  %1170 = trunc i64 %1167 to i32, !dbg !18
  %1171 = insertelement <4 x i32> poison, i32 %1170, i64 0, !dbg !18
  %1172 = lshr i64 %1167, 32, !dbg !18
  %1173 = trunc i64 %1172 to i32, !dbg !18
  %1174 = insertelement <4 x i32> %1171, i32 %1173, i64 1, !dbg !18
  %1175 = insertelement <4 x i32> %1174, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1070 = insertelement <4 x i32> %1175, i32 131072, i64 3, !dbg !18
  %1176 = ptrtoint ptr addrspace(3) %1166 to i32, !dbg !18
  %dma.buffer.lds1071 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1176), !dbg !18
  %1177 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1071, i32 %dma.buffer.offset1069, <4 x i32> %dma.buffer.rsrc1070) #10, !dbg !18
  %address.local810 = or disjoint i64 %address.local631, 256, !dbg !12
  %address.local811 = shl nuw nsw i64 %address.local810, 3, !dbg !12
  %address.local812 = and i64 %address.local811, 3584, !dbg !12
  %address.local813 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local812, !dbg !12
  %address.local814 = ptrtoint ptr addrspace(3) %address.local813 to i64, !dbg !12
  %address.local815 = trunc i64 %address.local814 to i32, !dbg !12
  %1178 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local815), !dbg !18
  %1179 = zext i32 %1178 to i64, !dbg !18
  %1180 = inttoptr i64 %1179 to ptr addrspace(3), !dbg !18
  %address.local816 = shl nuw nsw i64 %address.local631, 7, !dbg !12
  %address.local817 = and i64 %address.local816, 16256, !dbg !12
  %address.local818 = lshr i64 %address.local810, 4, !dbg !12
  %address.local819 = and i64 %address.local818, 24, !dbg !12
  %address.local820 = or disjoint i64 %address.local817, %address.local819, !dbg !12
  %address.local821 = getelementptr [2 x i8], ptr addrspace(1) %1163, i64 %address.local820, !dbg !18
  %1181 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1182 = ptrtoint ptr addrspace(1) %address.local821 to i64, !dbg !18
  %1183 = sub i64 %1182, %1181, !dbg !18
  %dma.buffer.offset1066 = trunc i64 %1183 to i32, !dbg !18
  %1184 = trunc i64 %1181 to i32, !dbg !18
  %1185 = insertelement <4 x i32> poison, i32 %1184, i64 0, !dbg !18
  %1186 = lshr i64 %1181, 32, !dbg !18
  %1187 = trunc i64 %1186 to i32, !dbg !18
  %1188 = insertelement <4 x i32> %1185, i32 %1187, i64 1, !dbg !18
  %1189 = insertelement <4 x i32> %1188, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1067 = insertelement <4 x i32> %1189, i32 131072, i64 3, !dbg !18
  %1190 = ptrtoint ptr addrspace(3) %1180 to i32, !dbg !18
  %dma.buffer.lds1068 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1190), !dbg !18
  %1191 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1068, i32 %dma.buffer.offset1066, <4 x i32> %dma.buffer.rsrc1067) #10, !dbg !18
  %address.local822 = or disjoint i64 %address.local631, 512, !dbg !12
  %address.local823 = shl nuw nsw i64 %address.local822, 3, !dbg !12
  %address.local824 = and i64 %address.local823, 5632, !dbg !12
  %address.local825 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local824, !dbg !12
  %address.local826 = ptrtoint ptr addrspace(3) %address.local825 to i64, !dbg !12
  %address.local827 = trunc i64 %address.local826 to i32, !dbg !12
  %1192 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local827), !dbg !18
  %1193 = zext i32 %1192 to i64, !dbg !18
  %1194 = inttoptr i64 %1193 to ptr addrspace(3), !dbg !18
  %address.local828 = shl nuw nsw i64 %address.local631, 7, !dbg !12
  %address.local829 = and i64 %address.local828, 16256, !dbg !12
  %address.local830 = lshr i64 %address.local822, 4, !dbg !12
  %address.local831 = and i64 %address.local830, 40, !dbg !12
  %address.local832 = or disjoint i64 %address.local829, %address.local831, !dbg !12
  %address.local833 = getelementptr [2 x i8], ptr addrspace(1) %1163, i64 %address.local832, !dbg !18
  %1195 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1196 = ptrtoint ptr addrspace(1) %address.local833 to i64, !dbg !18
  %1197 = sub i64 %1196, %1195, !dbg !18
  %dma.buffer.offset1063 = trunc i64 %1197 to i32, !dbg !18
  %1198 = trunc i64 %1195 to i32, !dbg !18
  %1199 = insertelement <4 x i32> poison, i32 %1198, i64 0, !dbg !18
  %1200 = lshr i64 %1195, 32, !dbg !18
  %1201 = trunc i64 %1200 to i32, !dbg !18
  %1202 = insertelement <4 x i32> %1199, i32 %1201, i64 1, !dbg !18
  %1203 = insertelement <4 x i32> %1202, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1064 = insertelement <4 x i32> %1203, i32 131072, i64 3, !dbg !18
  %1204 = ptrtoint ptr addrspace(3) %1194 to i32, !dbg !18
  %dma.buffer.lds1065 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1204), !dbg !18
  %1205 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1065, i32 %dma.buffer.offset1063, <4 x i32> %dma.buffer.rsrc1064) #10, !dbg !18
  %address.local834 = or disjoint i64 %address.local631, 768, !dbg !12
  %address.local835 = shl nuw nsw i64 %address.local834, 3, !dbg !12
  %address.local836 = and i64 %address.local835, 7680, !dbg !12
  %address.local837 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local836, !dbg !12
  %address.local838 = ptrtoint ptr addrspace(3) %address.local837 to i64, !dbg !12
  %address.local839 = trunc i64 %address.local838 to i32, !dbg !12
  %1206 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local839), !dbg !18
  %1207 = zext i32 %1206 to i64, !dbg !18
  %1208 = inttoptr i64 %1207 to ptr addrspace(3), !dbg !18
  %address.local840 = shl nuw nsw i64 %address.local631, 7, !dbg !12
  %address.local841 = and i64 %address.local840, 16256, !dbg !12
  %address.local842 = lshr i64 %address.local834, 4, !dbg !12
  %address.local843 = and i64 %address.local842, 56, !dbg !12
  %address.local844 = or disjoint i64 %address.local841, %address.local843, !dbg !12
  %address.local845 = getelementptr [2 x i8], ptr addrspace(1) %1163, i64 %address.local844, !dbg !18
  %1209 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1210 = ptrtoint ptr addrspace(1) %address.local845 to i64, !dbg !18
  %1211 = sub i64 %1210, %1209, !dbg !18
  %dma.buffer.offset1060 = trunc i64 %1211 to i32, !dbg !18
  %1212 = trunc i64 %1209 to i32, !dbg !18
  %1213 = insertelement <4 x i32> poison, i32 %1212, i64 0, !dbg !18
  %1214 = lshr i64 %1209, 32, !dbg !18
  %1215 = trunc i64 %1214 to i32, !dbg !18
  %1216 = insertelement <4 x i32> %1213, i32 %1215, i64 1, !dbg !18
  %1217 = insertelement <4 x i32> %1216, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1061 = insertelement <4 x i32> %1217, i32 131072, i64 3, !dbg !18
  %1218 = ptrtoint ptr addrspace(3) %1208 to i32, !dbg !18
  %dma.buffer.lds1062 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1218), !dbg !18
  %1219 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1062, i32 %dma.buffer.offset1060, <4 x i32> %dma.buffer.rsrc1061) #10, !dbg !18
  %1220 = insertelement <4 x half> poison, half %1094, i64 0, !dbg !15
  %1221 = insertelement <4 x half> %1220, half %1095, i64 1, !dbg !15
  %1222 = insertelement <4 x half> %1221, half %1096, i64 2, !dbg !15
  %1223 = insertelement <4 x half> %1222, half %1097, i64 3, !dbg !15
  %1224 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1225 = extractvalue [2 x <16 x float>] %37, 0, !dbg !30
  %1226 = shufflevector <16 x float> %1225, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1227 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1223, <4 x half> %1224, <4 x float> %1226) #11, !dbg !15
  %1228 = insertelement <4 x half> poison, half %1098, i64 0, !dbg !15
  %1229 = insertelement <4 x half> %1228, half %1099, i64 1, !dbg !15
  %1230 = insertelement <4 x half> %1229, half %1100, i64 2, !dbg !15
  %1231 = insertelement <4 x half> %1230, half %1101, i64 3, !dbg !15
  %1232 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1233 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1231, <4 x half> %1232, <4 x float> %1227) #11, !dbg !15
  %1234 = insertelement <4 x half> poison, half %1102, i64 0, !dbg !15
  %1235 = insertelement <4 x half> %1234, half %1103, i64 1, !dbg !15
  %1236 = insertelement <4 x half> %1235, half %1104, i64 2, !dbg !15
  %1237 = insertelement <4 x half> %1236, half %1105, i64 3, !dbg !15
  %1238 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1239 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1237, <4 x half> %1238, <4 x float> %1233) #11, !dbg !15
  %1240 = insertelement <4 x half> poison, half %1106, i64 0, !dbg !15
  %1241 = insertelement <4 x half> %1240, half %1107, i64 1, !dbg !15
  %1242 = insertelement <4 x half> %1241, half %1108, i64 2, !dbg !15
  %1243 = insertelement <4 x half> %1242, half %1109, i64 3, !dbg !15
  %1244 = shufflevector <8 x half> %911, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1245 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1243, <4 x half> %1244, <4 x float> %1239) #11, !dbg !15
  %1246 = insertelement <4 x half> poison, half %1110, i64 0, !dbg !15
  %1247 = insertelement <4 x half> %1246, half %1111, i64 1, !dbg !15
  %1248 = insertelement <4 x half> %1247, half %1112, i64 2, !dbg !15
  %1249 = insertelement <4 x half> %1248, half %1113, i64 3, !dbg !15
  %1250 = shufflevector <8 x half> %919, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1251 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1249, <4 x half> %1250, <4 x float> %1245) #11, !dbg !15
  %1252 = insertelement <4 x half> poison, half %1114, i64 0, !dbg !15
  %1253 = insertelement <4 x half> %1252, half %1115, i64 1, !dbg !15
  %1254 = insertelement <4 x half> %1253, half %1116, i64 2, !dbg !15
  %1255 = insertelement <4 x half> %1254, half %1117, i64 3, !dbg !15
  %1256 = shufflevector <8 x half> %927, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1257 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1255, <4 x half> %1256, <4 x float> %1251) #11, !dbg !15
  %1258 = insertelement <4 x half> poison, half %1118, i64 0, !dbg !15
  %1259 = insertelement <4 x half> %1258, half %1119, i64 1, !dbg !15
  %1260 = insertelement <4 x half> %1259, half %1120, i64 2, !dbg !15
  %1261 = insertelement <4 x half> %1260, half %1121, i64 3, !dbg !15
  %1262 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1263 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1261, <4 x half> %1262, <4 x float> %1257) #11, !dbg !15
  %1264 = insertelement <4 x half> poison, half %1122, i64 0, !dbg !15
  %1265 = insertelement <4 x half> %1264, half %1123, i64 1, !dbg !15
  %1266 = insertelement <4 x half> %1265, half %1124, i64 2, !dbg !15
  %1267 = insertelement <4 x half> %1266, half %1125, i64 3, !dbg !15
  %1268 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1269 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1267, <4 x half> %1268, <4 x float> %1263) #11, !dbg !15
  %1270 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1271 = shufflevector <16 x float> %1225, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1272 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1223, <4 x half> %1270, <4 x float> %1271) #11, !dbg !15
  %1273 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1274 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1231, <4 x half> %1273, <4 x float> %1272) #11, !dbg !15
  %1275 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1276 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1237, <4 x half> %1275, <4 x float> %1274) #11, !dbg !15
  %1277 = shufflevector <8 x half> %911, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1278 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1243, <4 x half> %1277, <4 x float> %1276) #11, !dbg !15
  %1279 = shufflevector <8 x half> %919, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1280 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1249, <4 x half> %1279, <4 x float> %1278) #11, !dbg !15
  %1281 = shufflevector <8 x half> %927, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1282 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1255, <4 x half> %1281, <4 x float> %1280) #11, !dbg !15
  %1283 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1284 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1261, <4 x half> %1283, <4 x float> %1282) #11, !dbg !15
  %1285 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1286 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1267, <4 x half> %1285, <4 x float> %1284) #11, !dbg !15
  %1287 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1288 = shufflevector <16 x float> %1225, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %1289 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1223, <4 x half> %1287, <4 x float> %1288) #11, !dbg !15
  %1290 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1291 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1231, <4 x half> %1290, <4 x float> %1289) #11, !dbg !15
  %1292 = shufflevector <8 x half> %907, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1293 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1237, <4 x half> %1292, <4 x float> %1291) #11, !dbg !15
  %1294 = shufflevector <8 x half> %915, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1295 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1243, <4 x half> %1294, <4 x float> %1293) #11, !dbg !15
  %1296 = shufflevector <8 x half> %923, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1297 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1249, <4 x half> %1296, <4 x float> %1295) #11, !dbg !15
  %1298 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1299 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1255, <4 x half> %1298, <4 x float> %1297) #11, !dbg !15
  %1300 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1301 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1261, <4 x half> %1300, <4 x float> %1299) #11, !dbg !15
  %1302 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1303 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1267, <4 x half> %1302, <4 x float> %1301) #11, !dbg !15
  %1304 = shufflevector <4 x float> %1303, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1305 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1306 = shufflevector <16 x float> %1225, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %1307 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1223, <4 x half> %1305, <4 x float> %1306) #11, !dbg !15
  %1308 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1309 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1231, <4 x half> %1308, <4 x float> %1307) #11, !dbg !15
  %1310 = shufflevector <8 x half> %907, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1311 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1237, <4 x half> %1310, <4 x float> %1309) #11, !dbg !15
  %1312 = shufflevector <8 x half> %915, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1313 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1243, <4 x half> %1312, <4 x float> %1311) #11, !dbg !15
  %1314 = shufflevector <8 x half> %923, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1315 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1249, <4 x half> %1314, <4 x float> %1313) #11, !dbg !15
  %1316 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1317 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1255, <4 x half> %1316, <4 x float> %1315) #11, !dbg !15
  %1318 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1319 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1261, <4 x half> %1318, <4 x float> %1317) #11, !dbg !15
  %1320 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1321 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1267, <4 x half> %1320, <4 x float> %1319) #11, !dbg !15
  %1322 = shufflevector <4 x float> %1321, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1323 = insertelement <4 x half> poison, half %1126, i64 0, !dbg !15
  %1324 = insertelement <4 x half> %1323, half %1127, i64 1, !dbg !15
  %1325 = insertelement <4 x half> %1324, half %1128, i64 2, !dbg !15
  %1326 = insertelement <4 x half> %1325, half %1129, i64 3, !dbg !15
  %1327 = extractvalue [2 x <16 x float>] %37, 1, !dbg !30
  %1328 = shufflevector <16 x float> %1327, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1329 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1326, <4 x half> %1224, <4 x float> %1328) #11, !dbg !15
  %1330 = insertelement <4 x half> poison, half %1130, i64 0, !dbg !15
  %1331 = insertelement <4 x half> %1330, half %1131, i64 1, !dbg !15
  %1332 = insertelement <4 x half> %1331, half %1132, i64 2, !dbg !15
  %1333 = insertelement <4 x half> %1332, half %1133, i64 3, !dbg !15
  %1334 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1333, <4 x half> %1232, <4 x float> %1329) #11, !dbg !15
  %1335 = insertelement <4 x half> poison, half %1134, i64 0, !dbg !15
  %1336 = insertelement <4 x half> %1335, half %1135, i64 1, !dbg !15
  %1337 = insertelement <4 x half> %1336, half %1136, i64 2, !dbg !15
  %1338 = insertelement <4 x half> %1337, half %1137, i64 3, !dbg !15
  %1339 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1338, <4 x half> %1238, <4 x float> %1334) #11, !dbg !15
  %1340 = insertelement <4 x half> poison, half %1138, i64 0, !dbg !15
  %1341 = insertelement <4 x half> %1340, half %1139, i64 1, !dbg !15
  %1342 = insertelement <4 x half> %1341, half %1140, i64 2, !dbg !15
  %1343 = insertelement <4 x half> %1342, half %1141, i64 3, !dbg !15
  %1344 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1244, <4 x float> %1339) #11, !dbg !15
  %1345 = insertelement <4 x half> poison, half %1142, i64 0, !dbg !15
  %1346 = insertelement <4 x half> %1345, half %1143, i64 1, !dbg !15
  %1347 = insertelement <4 x half> %1346, half %1144, i64 2, !dbg !15
  %1348 = insertelement <4 x half> %1347, half %1145, i64 3, !dbg !15
  %1349 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1348, <4 x half> %1250, <4 x float> %1344) #11, !dbg !15
  %1350 = insertelement <4 x half> poison, half %1146, i64 0, !dbg !15
  %1351 = insertelement <4 x half> %1350, half %1147, i64 1, !dbg !15
  %1352 = insertelement <4 x half> %1351, half %1148, i64 2, !dbg !15
  %1353 = insertelement <4 x half> %1352, half %1149, i64 3, !dbg !15
  %1354 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1353, <4 x half> %1256, <4 x float> %1349) #11, !dbg !15
  %1355 = insertelement <4 x half> poison, half %1150, i64 0, !dbg !15
  %1356 = insertelement <4 x half> %1355, half %1151, i64 1, !dbg !15
  %1357 = insertelement <4 x half> %1356, half %1152, i64 2, !dbg !15
  %1358 = insertelement <4 x half> %1357, half %1153, i64 3, !dbg !15
  %1359 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1358, <4 x half> %1262, <4 x float> %1354) #11, !dbg !15
  %1360 = insertelement <4 x half> poison, half %1154, i64 0, !dbg !15
  %1361 = insertelement <4 x half> %1360, half %1155, i64 1, !dbg !15
  %1362 = insertelement <4 x half> %1361, half %1156, i64 2, !dbg !15
  %1363 = insertelement <4 x half> %1362, half %1157, i64 3, !dbg !15
  %1364 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1363, <4 x half> %1268, <4 x float> %1359) #11, !dbg !15
  %1365 = shufflevector <16 x float> %1327, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1366 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1326, <4 x half> %1270, <4 x float> %1365) #11, !dbg !15
  %1367 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1333, <4 x half> %1273, <4 x float> %1366) #11, !dbg !15
  %1368 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1338, <4 x half> %1275, <4 x float> %1367) #11, !dbg !15
  %1369 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1277, <4 x float> %1368) #11, !dbg !15
  %1370 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1348, <4 x half> %1279, <4 x float> %1369) #11, !dbg !15
  %1371 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1353, <4 x half> %1281, <4 x float> %1370) #11, !dbg !15
  %1372 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1358, <4 x half> %1283, <4 x float> %1371) #11, !dbg !15
  %1373 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1363, <4 x half> %1285, <4 x float> %1372) #11, !dbg !15
  %1374 = shufflevector <16 x float> %1327, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %1375 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1326, <4 x half> %1287, <4 x float> %1374) #11, !dbg !15
  %1376 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1333, <4 x half> %1290, <4 x float> %1375) #11, !dbg !15
  %1377 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1338, <4 x half> %1292, <4 x float> %1376) #11, !dbg !15
  %1378 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1294, <4 x float> %1377) #11, !dbg !15
  %1379 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1348, <4 x half> %1296, <4 x float> %1378) #11, !dbg !15
  %1380 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1353, <4 x half> %1298, <4 x float> %1379) #11, !dbg !15
  %1381 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1358, <4 x half> %1300, <4 x float> %1380) #11, !dbg !15
  %1382 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1363, <4 x half> %1302, <4 x float> %1381) #11, !dbg !15
  %1383 = shufflevector <4 x float> %1382, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1384 = shufflevector <16 x float> %1327, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %1385 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1326, <4 x half> %1305, <4 x float> %1384) #11, !dbg !15
  %1386 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1333, <4 x half> %1308, <4 x float> %1385) #11, !dbg !15
  %1387 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1338, <4 x half> %1310, <4 x float> %1386) #11, !dbg !15
  %1388 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1312, <4 x float> %1387) #11, !dbg !15
  %1389 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1348, <4 x half> %1314, <4 x float> %1388) #11, !dbg !15
  %1390 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1353, <4 x half> %1316, <4 x float> %1389) #11, !dbg !15
  %1391 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1358, <4 x half> %1318, <4 x float> %1390) #11, !dbg !15
  %1392 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1363, <4 x half> %1320, <4 x float> %1391) #11, !dbg !15
  %1393 = shufflevector <4 x float> %1392, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1394 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %885, i32 0) #0, !dbg !15
  %1395 = bitcast <4 x i32> %1394 to <8 x half>, !dbg !15
  %1396 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %889, i32 0) #0, !dbg !15
  %1397 = bitcast <4 x i32> %1396 to <8 x half>, !dbg !15
  %1398 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %893, i32 0) #0, !dbg !15
  %1399 = bitcast <4 x i32> %1398 to <8 x half>, !dbg !15
  %1400 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %897, i32 0) #0, !dbg !15
  %1401 = bitcast <4 x i32> %1400 to <8 x half>, !dbg !15
  %1402 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %901, i32 0) #0, !dbg !15
  %1403 = bitcast <4 x i32> %1402 to <8 x half>, !dbg !15
  %1404 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %905, i32 0) #0, !dbg !15
  %1405 = bitcast <4 x i32> %1404 to <8 x half>, !dbg !15
  %1406 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %909, i32 0) #0, !dbg !15
  %1407 = bitcast <4 x i32> %1406 to <8 x half>, !dbg !15
  %1408 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %913, i32 0) #0, !dbg !15
  %1409 = bitcast <4 x i32> %1408 to <8 x half>, !dbg !15
  %1410 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %917, i32 0) #0, !dbg !15
  %1411 = bitcast <4 x i32> %1410 to <8 x half>, !dbg !15
  %1412 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %921, i32 0) #0, !dbg !15
  %1413 = bitcast <4 x i32> %1412 to <8 x half>, !dbg !15
  %1414 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %925, i32 0) #0, !dbg !15
  %1415 = bitcast <4 x i32> %1414 to <8 x half>, !dbg !15
  %1416 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %929, i32 0) #0, !dbg !15
  %1417 = bitcast <4 x i32> %1416 to <8 x half>, !dbg !15
  %1418 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %933, i32 0) #0, !dbg !15
  %1419 = bitcast <4 x i32> %1418 to <8 x half>, !dbg !15
  %1420 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %937, i32 0) #0, !dbg !15
  %1421 = bitcast <4 x i32> %1420 to <8 x half>, !dbg !15
  %1422 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %941, i32 0) #0, !dbg !15
  %1423 = bitcast <4 x i32> %1422 to <8 x half>, !dbg !15
  %1424 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %945, i32 0) #0, !dbg !15
  %1425 = bitcast <4 x i32> %1424 to <8 x half>, !dbg !15
  %1426 = shufflevector <8 x half> %1395, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1427 = extractvalue [2 x <16 x float>] %39, 0, !dbg !30
  %1428 = shufflevector <16 x float> %1427, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1429 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1223, <4 x half> %1426, <4 x float> %1428) #11, !dbg !15
  %1430 = shufflevector <8 x half> %1399, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1431 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1231, <4 x half> %1430, <4 x float> %1429) #11, !dbg !15
  %1432 = shufflevector <8 x half> %1403, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1433 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1237, <4 x half> %1432, <4 x float> %1431) #11, !dbg !15
  %1434 = shufflevector <8 x half> %1407, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1435 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1243, <4 x half> %1434, <4 x float> %1433) #11, !dbg !15
  %1436 = shufflevector <8 x half> %1411, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1437 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1249, <4 x half> %1436, <4 x float> %1435) #11, !dbg !15
  %1438 = shufflevector <8 x half> %1415, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1439 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1255, <4 x half> %1438, <4 x float> %1437) #11, !dbg !15
  %1440 = shufflevector <8 x half> %1419, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1441 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1261, <4 x half> %1440, <4 x float> %1439) #11, !dbg !15
  %1442 = shufflevector <8 x half> %1423, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1443 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1267, <4 x half> %1442, <4 x float> %1441) #11, !dbg !15
  %1444 = shufflevector <8 x half> %1395, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1445 = shufflevector <16 x float> %1427, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1446 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1223, <4 x half> %1444, <4 x float> %1445) #11, !dbg !15
  %1447 = shufflevector <8 x half> %1399, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1448 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1231, <4 x half> %1447, <4 x float> %1446) #11, !dbg !15
  %1449 = shufflevector <8 x half> %1403, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1450 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1237, <4 x half> %1449, <4 x float> %1448) #11, !dbg !15
  %1451 = shufflevector <8 x half> %1407, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1452 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1243, <4 x half> %1451, <4 x float> %1450) #11, !dbg !15
  %1453 = shufflevector <8 x half> %1411, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1454 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1249, <4 x half> %1453, <4 x float> %1452) #11, !dbg !15
  %1455 = shufflevector <8 x half> %1415, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1456 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1255, <4 x half> %1455, <4 x float> %1454) #11, !dbg !15
  %1457 = shufflevector <8 x half> %1419, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1458 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1261, <4 x half> %1457, <4 x float> %1456) #11, !dbg !15
  %1459 = shufflevector <8 x half> %1423, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1460 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1267, <4 x half> %1459, <4 x float> %1458) #11, !dbg !15
  %1461 = shufflevector <8 x half> %1397, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1462 = shufflevector <16 x float> %1427, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %1463 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1223, <4 x half> %1461, <4 x float> %1462) #11, !dbg !15
  %1464 = shufflevector <8 x half> %1401, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1465 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1231, <4 x half> %1464, <4 x float> %1463) #11, !dbg !15
  %1466 = shufflevector <8 x half> %1405, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1467 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1237, <4 x half> %1466, <4 x float> %1465) #11, !dbg !15
  %1468 = shufflevector <8 x half> %1409, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1469 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1243, <4 x half> %1468, <4 x float> %1467) #11, !dbg !15
  %1470 = shufflevector <8 x half> %1413, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1471 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1249, <4 x half> %1470, <4 x float> %1469) #11, !dbg !15
  %1472 = shufflevector <8 x half> %1417, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1473 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1255, <4 x half> %1472, <4 x float> %1471) #11, !dbg !15
  %1474 = shufflevector <8 x half> %1421, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1475 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1261, <4 x half> %1474, <4 x float> %1473) #11, !dbg !15
  %1476 = shufflevector <8 x half> %1425, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1477 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1267, <4 x half> %1476, <4 x float> %1475) #11, !dbg !15
  %1478 = shufflevector <4 x float> %1477, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1479 = shufflevector <8 x half> %1397, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1480 = shufflevector <16 x float> %1427, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %1481 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1223, <4 x half> %1479, <4 x float> %1480) #11, !dbg !15
  %1482 = shufflevector <8 x half> %1401, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1483 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1231, <4 x half> %1482, <4 x float> %1481) #11, !dbg !15
  %1484 = shufflevector <8 x half> %1405, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1485 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1237, <4 x half> %1484, <4 x float> %1483) #11, !dbg !15
  %1486 = shufflevector <8 x half> %1409, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1487 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1243, <4 x half> %1486, <4 x float> %1485) #11, !dbg !15
  %1488 = shufflevector <8 x half> %1413, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1489 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1249, <4 x half> %1488, <4 x float> %1487) #11, !dbg !15
  %1490 = shufflevector <8 x half> %1417, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1491 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1255, <4 x half> %1490, <4 x float> %1489) #11, !dbg !15
  %1492 = shufflevector <8 x half> %1421, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1493 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1261, <4 x half> %1492, <4 x float> %1491) #11, !dbg !15
  %1494 = shufflevector <8 x half> %1425, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1495 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1267, <4 x half> %1494, <4 x float> %1493) #11, !dbg !15
  %1496 = shufflevector <4 x float> %1495, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1497 = extractvalue [2 x <16 x float>] %39, 1, !dbg !30
  %1498 = shufflevector <16 x float> %1497, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1499 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1326, <4 x half> %1426, <4 x float> %1498) #11, !dbg !15
  %1500 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1333, <4 x half> %1430, <4 x float> %1499) #11, !dbg !15
  %1501 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1338, <4 x half> %1432, <4 x float> %1500) #11, !dbg !15
  %1502 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1434, <4 x float> %1501) #11, !dbg !15
  %1503 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1348, <4 x half> %1436, <4 x float> %1502) #11, !dbg !15
  %1504 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1353, <4 x half> %1438, <4 x float> %1503) #11, !dbg !15
  %1505 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1358, <4 x half> %1440, <4 x float> %1504) #11, !dbg !15
  %1506 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1363, <4 x half> %1442, <4 x float> %1505) #11, !dbg !15
  %1507 = shufflevector <16 x float> %1497, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1508 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1326, <4 x half> %1444, <4 x float> %1507) #11, !dbg !15
  %1509 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1333, <4 x half> %1447, <4 x float> %1508) #11, !dbg !15
  %1510 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1338, <4 x half> %1449, <4 x float> %1509) #11, !dbg !15
  %1511 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1451, <4 x float> %1510) #11, !dbg !15
  %1512 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1348, <4 x half> %1453, <4 x float> %1511) #11, !dbg !15
  %1513 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1353, <4 x half> %1455, <4 x float> %1512) #11, !dbg !15
  %1514 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1358, <4 x half> %1457, <4 x float> %1513) #11, !dbg !15
  %1515 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1363, <4 x half> %1459, <4 x float> %1514) #11, !dbg !15
  %1516 = shufflevector <16 x float> %1497, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %1517 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1326, <4 x half> %1461, <4 x float> %1516) #11, !dbg !15
  %1518 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1333, <4 x half> %1464, <4 x float> %1517) #11, !dbg !15
  %1519 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1338, <4 x half> %1466, <4 x float> %1518) #11, !dbg !15
  %1520 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1468, <4 x float> %1519) #11, !dbg !15
  %1521 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1348, <4 x half> %1470, <4 x float> %1520) #11, !dbg !15
  %1522 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1353, <4 x half> %1472, <4 x float> %1521) #11, !dbg !15
  %1523 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1358, <4 x half> %1474, <4 x float> %1522) #11, !dbg !15
  %1524 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1363, <4 x half> %1476, <4 x float> %1523) #11, !dbg !15
  %1525 = shufflevector <4 x float> %1524, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1526 = shufflevector <16 x float> %1497, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %1527 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1326, <4 x half> %1479, <4 x float> %1526) #11, !dbg !15
  %1528 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1333, <4 x half> %1482, <4 x float> %1527) #11, !dbg !15
  %1529 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1338, <4 x half> %1484, <4 x float> %1528) #11, !dbg !15
  %1530 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1343, <4 x half> %1486, <4 x float> %1529) #11, !dbg !15
  %1531 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1348, <4 x half> %1488, <4 x float> %1530) #11, !dbg !15
  %1532 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1353, <4 x half> %1490, <4 x float> %1531) #11, !dbg !15
  %1533 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1358, <4 x half> %1492, <4 x float> %1532) #11, !dbg !15
  %1534 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %1363, <4 x half> %1494, <4 x float> %1533) #11, !dbg !15
  %1535 = shufflevector <4 x float> %1534, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %1536 = shufflevector <4 x float> %1269, <4 x float> %1286, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %1537 = shufflevector <16 x float> %1536, <16 x float> %1304, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %1538 = shufflevector <16 x float> %1537, <16 x float> %1322, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !16
  %1539 = insertvalue [2 x <16 x float>] poison, <16 x float> %1538, 0, !dbg !16
  %1540 = shufflevector <4 x float> %1364, <4 x float> %1373, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %1541 = shufflevector <16 x float> %1540, <16 x float> %1383, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %1542 = shufflevector <16 x float> %1541, <16 x float> %1393, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !16
  %1543 = insertvalue [2 x <16 x float>] %1539, <16 x float> %1542, 1, !dbg !16
  %1544 = insertelement <1 x float> poison, float %873, i64 0, !dbg !16
  %1545 = insertvalue [2 x <1 x float>] poison, <1 x float> %1544, 0, !dbg !16
  %1546 = insertelement <1 x float> poison, float %884, i64 0, !dbg !16
  %1547 = insertvalue [2 x <1 x float>] %1545, <1 x float> %1546, 1, !dbg !16
  %1548 = shufflevector <4 x float> %1443, <4 x float> %1460, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %1549 = shufflevector <16 x float> %1548, <16 x float> %1478, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %1550 = shufflevector <16 x float> %1549, <16 x float> %1496, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !16
  %1551 = insertvalue [2 x <16 x float>] poison, <16 x float> %1550, 0, !dbg !16
  %1552 = shufflevector <4 x float> %1506, <4 x float> %1515, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %1553 = shufflevector <16 x float> %1552, <16 x float> %1525, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %1554 = shufflevector <16 x float> %1553, <16 x float> %1535, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !16
  %1555 = insertvalue [2 x <16 x float>] %1551, <16 x float> %1554, 1, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1556 = add nuw nsw i64 %36, 1, !dbg !16
  br label %35, !dbg !16

1557:                                             ; preds = %35
  %1558 = and i32 %19, 16, !dbg !16
  %.not2 = icmp eq i32 %1558, 0, !dbg !16
  %1559 = and i32 %19, 32, !dbg !16
  %.not3 = icmp eq i32 %1559, 0, !dbg !16
  %1560 = extractvalue [2 x <16 x float>] %39, 0, !dbg !16
  %1561 = extractelement <16 x float> %1560, i64 0, !dbg !16
  %1562 = extractelement <16 x float> %1560, i64 1, !dbg !16
  %1563 = extractelement <16 x float> %1560, i64 2, !dbg !16
  %1564 = extractelement <16 x float> %1560, i64 3, !dbg !16
  %1565 = select i1 %.not2, float %1562, float %1561, !dbg !16
  %1566 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %1567 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1566), !dbg !16
  %1568 = add nuw nsw i32 %1567, 64, !dbg !16
  %1569 = xor i32 %1567, 16, !dbg !16
  %1570 = and i32 %1568, 192, !dbg !16
  %1571 = icmp samesign ult i32 %1569, %1570, !dbg !16
  %1572 = select i1 %1571, i32 %1569, i32 %1567, !dbg !16
  %1573 = shl nuw nsw i32 %1572, 2, !dbg !16
  %1574 = bitcast float %1565 to i32, !dbg !16
  %1575 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1574), !dbg !16
  %1576 = bitcast i32 %1575 to float, !dbg !16
  %1577 = select i1 %.not2, float %1561, float %1576, !dbg !16
  %1578 = select i1 %.not2, float %1576, float %1562, !dbg !16
  %1579 = select i1 %.not2, float %1564, float %1563, !dbg !16
  %1580 = bitcast float %1579 to i32, !dbg !16
  %1581 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1580), !dbg !16
  %1582 = bitcast i32 %1581 to float, !dbg !16
  %1583 = select i1 %.not2, float %1563, float %1582, !dbg !16
  %1584 = select i1 %.not2, float %1582, float %1564, !dbg !16
  %1585 = select i1 %.not3, float %1583, float %1577, !dbg !16
  %1586 = xor i32 %1567, 32, !dbg !16
  %1587 = icmp samesign ult i32 %1586, %1570, !dbg !16
  %1588 = select i1 %1587, i32 %1586, i32 %1567, !dbg !16
  %1589 = shl nuw nsw i32 %1588, 2, !dbg !16
  %1590 = bitcast float %1585 to i32, !dbg !16
  %1591 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1590), !dbg !16
  %1592 = bitcast i32 %1591 to float, !dbg !16
  %1593 = select i1 %.not3, float %1577, float %1592, !dbg !16
  %1594 = select i1 %.not3, float %1592, float %1583, !dbg !16
  %1595 = select i1 %.not3, float %1584, float %1578, !dbg !16
  %1596 = bitcast float %1595 to i32, !dbg !16
  %1597 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1596), !dbg !16
  %1598 = bitcast i32 %1597 to float, !dbg !16
  %1599 = select i1 %.not3, float %1578, float %1598, !dbg !16
  %1600 = select i1 %.not3, float %1598, float %1584, !dbg !16
  %1601 = extractvalue [2 x <16 x float>] %37, 0, !dbg !16
  %1602 = extractelement <16 x float> %1601, i64 0, !dbg !16
  %1603 = extractelement <16 x float> %1601, i64 1, !dbg !16
  %1604 = extractelement <16 x float> %1601, i64 2, !dbg !16
  %1605 = extractelement <16 x float> %1601, i64 3, !dbg !16
  %1606 = select i1 %.not2, float %1603, float %1602, !dbg !16
  %1607 = bitcast float %1606 to i32, !dbg !16
  %1608 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1607), !dbg !16
  %1609 = bitcast i32 %1608 to float, !dbg !16
  %1610 = select i1 %.not2, float %1602, float %1609, !dbg !16
  %1611 = select i1 %.not2, float %1609, float %1603, !dbg !16
  %1612 = select i1 %.not2, float %1605, float %1604, !dbg !16
  %1613 = bitcast float %1612 to i32, !dbg !16
  %1614 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1613), !dbg !16
  %1615 = bitcast i32 %1614 to float, !dbg !16
  %1616 = select i1 %.not2, float %1604, float %1615, !dbg !16
  %1617 = select i1 %.not2, float %1615, float %1605, !dbg !16
  %1618 = select i1 %.not3, float %1616, float %1610, !dbg !16
  %1619 = bitcast float %1618 to i32, !dbg !16
  %1620 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1619), !dbg !16
  %1621 = select i1 %.not3, float %1617, float %1611, !dbg !16
  %1622 = bitcast float %1621 to i32, !dbg !16
  %1623 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1622), !dbg !16
  %1624 = extractvalue [2 x <1 x float>] %38, 0, !dbg !19
  %1625 = shufflevector <1 x float> %1624, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1626 = extractelement <16 x float> %1560, i64 4, !dbg !16
  %1627 = extractelement <16 x float> %1560, i64 5, !dbg !16
  %1628 = extractelement <16 x float> %1560, i64 6, !dbg !16
  %1629 = extractelement <16 x float> %1560, i64 7, !dbg !16
  %1630 = select i1 %.not2, float %1627, float %1626, !dbg !16
  %1631 = bitcast float %1630 to i32, !dbg !16
  %1632 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1631), !dbg !16
  %1633 = bitcast i32 %1632 to float, !dbg !16
  %1634 = select i1 %.not2, float %1626, float %1633, !dbg !16
  %1635 = select i1 %.not2, float %1633, float %1627, !dbg !16
  %1636 = select i1 %.not2, float %1629, float %1628, !dbg !16
  %1637 = bitcast float %1636 to i32, !dbg !16
  %1638 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1637), !dbg !16
  %1639 = bitcast i32 %1638 to float, !dbg !16
  %1640 = select i1 %.not2, float %1628, float %1639, !dbg !16
  %1641 = select i1 %.not2, float %1639, float %1629, !dbg !16
  %1642 = select i1 %.not3, float %1640, float %1634, !dbg !16
  %1643 = bitcast float %1642 to i32, !dbg !16
  %1644 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1643), !dbg !16
  %1645 = bitcast i32 %1644 to float, !dbg !16
  %1646 = select i1 %.not3, float %1634, float %1645, !dbg !16
  %1647 = select i1 %.not3, float %1645, float %1640, !dbg !16
  %1648 = select i1 %.not3, float %1641, float %1635, !dbg !16
  %1649 = bitcast float %1648 to i32, !dbg !16
  %1650 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1649), !dbg !16
  %1651 = bitcast i32 %1650 to float, !dbg !16
  %1652 = select i1 %.not3, float %1635, float %1651, !dbg !16
  %1653 = select i1 %.not3, float %1651, float %1641, !dbg !16
  %1654 = extractelement <16 x float> %1601, i64 4, !dbg !16
  %1655 = extractelement <16 x float> %1601, i64 5, !dbg !16
  %1656 = extractelement <16 x float> %1601, i64 6, !dbg !16
  %1657 = extractelement <16 x float> %1601, i64 7, !dbg !16
  %1658 = select i1 %.not2, float %1655, float %1654, !dbg !16
  %1659 = bitcast float %1658 to i32, !dbg !16
  %1660 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1659), !dbg !16
  %1661 = bitcast i32 %1660 to float, !dbg !16
  %1662 = select i1 %.not2, float %1654, float %1661, !dbg !16
  %1663 = select i1 %.not2, float %1661, float %1655, !dbg !16
  %1664 = select i1 %.not2, float %1657, float %1656, !dbg !16
  %1665 = bitcast float %1664 to i32, !dbg !16
  %1666 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1665), !dbg !16
  %1667 = bitcast i32 %1666 to float, !dbg !16
  %1668 = select i1 %.not2, float %1656, float %1667, !dbg !16
  %1669 = select i1 %.not2, float %1667, float %1657, !dbg !16
  %1670 = select i1 %.not3, float %1668, float %1662, !dbg !16
  %1671 = bitcast float %1670 to i32, !dbg !16
  %1672 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1671), !dbg !16
  %1673 = select i1 %.not3, float %1669, float %1663, !dbg !16
  %1674 = bitcast float %1673 to i32, !dbg !16
  %1675 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1674), !dbg !16
  %1676 = extractelement <16 x float> %1560, i64 8, !dbg !16
  %1677 = extractelement <16 x float> %1560, i64 9, !dbg !16
  %1678 = extractelement <16 x float> %1560, i64 10, !dbg !16
  %1679 = extractelement <16 x float> %1560, i64 11, !dbg !16
  %1680 = select i1 %.not2, float %1677, float %1676, !dbg !16
  %1681 = bitcast float %1680 to i32, !dbg !16
  %1682 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1681), !dbg !16
  %1683 = bitcast i32 %1682 to float, !dbg !16
  %1684 = select i1 %.not2, float %1676, float %1683, !dbg !16
  %1685 = select i1 %.not2, float %1683, float %1677, !dbg !16
  %1686 = select i1 %.not2, float %1679, float %1678, !dbg !16
  %1687 = bitcast float %1686 to i32, !dbg !16
  %1688 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1687), !dbg !16
  %1689 = bitcast i32 %1688 to float, !dbg !16
  %1690 = select i1 %.not2, float %1678, float %1689, !dbg !16
  %1691 = select i1 %.not2, float %1689, float %1679, !dbg !16
  %1692 = select i1 %.not3, float %1690, float %1684, !dbg !16
  %1693 = bitcast float %1692 to i32, !dbg !16
  %1694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1693), !dbg !16
  %1695 = bitcast i32 %1694 to float, !dbg !16
  %1696 = select i1 %.not3, float %1684, float %1695, !dbg !16
  %1697 = select i1 %.not3, float %1695, float %1690, !dbg !16
  %1698 = select i1 %.not3, float %1691, float %1685, !dbg !16
  %1699 = bitcast float %1698 to i32, !dbg !16
  %1700 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1699), !dbg !16
  %1701 = bitcast i32 %1700 to float, !dbg !16
  %1702 = select i1 %.not3, float %1685, float %1701, !dbg !16
  %1703 = select i1 %.not3, float %1701, float %1691, !dbg !16
  %1704 = extractelement <16 x float> %1601, i64 8, !dbg !16
  %1705 = extractelement <16 x float> %1601, i64 9, !dbg !16
  %1706 = extractelement <16 x float> %1601, i64 10, !dbg !16
  %1707 = extractelement <16 x float> %1601, i64 11, !dbg !16
  %1708 = select i1 %.not2, float %1705, float %1704, !dbg !16
  %1709 = bitcast float %1708 to i32, !dbg !16
  %1710 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1709), !dbg !16
  %1711 = bitcast i32 %1710 to float, !dbg !16
  %1712 = select i1 %.not2, float %1704, float %1711, !dbg !16
  %1713 = select i1 %.not2, float %1711, float %1705, !dbg !16
  %1714 = select i1 %.not2, float %1707, float %1706, !dbg !16
  %1715 = bitcast float %1714 to i32, !dbg !16
  %1716 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1715), !dbg !16
  %1717 = bitcast i32 %1716 to float, !dbg !16
  %1718 = select i1 %.not2, float %1706, float %1717, !dbg !16
  %1719 = select i1 %.not2, float %1717, float %1707, !dbg !16
  %1720 = select i1 %.not3, float %1718, float %1712, !dbg !16
  %1721 = bitcast float %1720 to i32, !dbg !16
  %1722 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1721), !dbg !16
  %1723 = select i1 %.not3, float %1719, float %1713, !dbg !16
  %1724 = bitcast float %1723 to i32, !dbg !16
  %1725 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1724), !dbg !16
  %1726 = extractelement <16 x float> %1560, i64 12, !dbg !16
  %1727 = extractelement <16 x float> %1560, i64 13, !dbg !16
  %1728 = extractelement <16 x float> %1560, i64 14, !dbg !16
  %1729 = extractelement <16 x float> %1560, i64 15, !dbg !16
  %1730 = select i1 %.not2, float %1727, float %1726, !dbg !16
  %1731 = bitcast float %1730 to i32, !dbg !16
  %1732 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1731), !dbg !16
  %1733 = bitcast i32 %1732 to float, !dbg !16
  %1734 = select i1 %.not2, float %1726, float %1733, !dbg !16
  %1735 = select i1 %.not2, float %1733, float %1727, !dbg !16
  %1736 = select i1 %.not2, float %1729, float %1728, !dbg !16
  %1737 = bitcast float %1736 to i32, !dbg !16
  %1738 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1737), !dbg !16
  %1739 = bitcast i32 %1738 to float, !dbg !16
  %1740 = select i1 %.not2, float %1728, float %1739, !dbg !16
  %1741 = select i1 %.not2, float %1739, float %1729, !dbg !16
  %1742 = select i1 %.not3, float %1740, float %1734, !dbg !16
  %1743 = bitcast float %1742 to i32, !dbg !16
  %1744 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1743), !dbg !16
  %1745 = bitcast i32 %1744 to float, !dbg !16
  %1746 = select i1 %.not3, float %1734, float %1745, !dbg !16
  %1747 = select i1 %.not3, float %1745, float %1740, !dbg !16
  %1748 = select i1 %.not3, float %1741, float %1735, !dbg !16
  %1749 = bitcast float %1748 to i32, !dbg !16
  %1750 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1749), !dbg !16
  %1751 = bitcast i32 %1750 to float, !dbg !16
  %1752 = select i1 %.not3, float %1735, float %1751, !dbg !16
  %1753 = select i1 %.not3, float %1751, float %1741, !dbg !16
  %1754 = extractelement <16 x float> %1601, i64 12, !dbg !16
  %1755 = extractelement <16 x float> %1601, i64 13, !dbg !16
  %1756 = extractelement <16 x float> %1601, i64 14, !dbg !16
  %1757 = extractelement <16 x float> %1601, i64 15, !dbg !16
  %1758 = select i1 %.not2, float %1755, float %1754, !dbg !16
  %1759 = bitcast float %1758 to i32, !dbg !16
  %1760 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1759), !dbg !16
  %1761 = bitcast i32 %1760 to float, !dbg !16
  %1762 = select i1 %.not2, float %1754, float %1761, !dbg !16
  %1763 = select i1 %.not2, float %1761, float %1755, !dbg !16
  %1764 = select i1 %.not2, float %1757, float %1756, !dbg !16
  %1765 = bitcast float %1764 to i32, !dbg !16
  %1766 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1765), !dbg !16
  %1767 = bitcast i32 %1766 to float, !dbg !16
  %1768 = select i1 %.not2, float %1756, float %1767, !dbg !16
  %1769 = select i1 %.not2, float %1767, float %1757, !dbg !16
  %1770 = select i1 %.not3, float %1768, float %1762, !dbg !16
  %1771 = bitcast float %1770 to i32, !dbg !16
  %1772 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1771), !dbg !16
  %1773 = select i1 %.not3, float %1769, float %1763, !dbg !16
  %1774 = bitcast float %1773 to i32, !dbg !16
  %1775 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1774), !dbg !16
  %1776 = extractvalue [2 x <16 x float>] %39, 1, !dbg !16
  %1777 = extractelement <16 x float> %1776, i64 0, !dbg !16
  %1778 = extractelement <16 x float> %1776, i64 1, !dbg !16
  %1779 = extractelement <16 x float> %1776, i64 2, !dbg !16
  %1780 = extractelement <16 x float> %1776, i64 3, !dbg !16
  %1781 = select i1 %.not2, float %1778, float %1777, !dbg !16
  %1782 = bitcast float %1781 to i32, !dbg !16
  %1783 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1782), !dbg !16
  %1784 = bitcast i32 %1783 to float, !dbg !16
  %1785 = select i1 %.not2, float %1777, float %1784, !dbg !16
  %1786 = select i1 %.not2, float %1784, float %1778, !dbg !16
  %1787 = select i1 %.not2, float %1780, float %1779, !dbg !16
  %1788 = bitcast float %1787 to i32, !dbg !16
  %1789 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1788), !dbg !16
  %1790 = bitcast i32 %1789 to float, !dbg !16
  %1791 = select i1 %.not2, float %1779, float %1790, !dbg !16
  %1792 = select i1 %.not2, float %1790, float %1780, !dbg !16
  %1793 = select i1 %.not3, float %1791, float %1785, !dbg !16
  %1794 = bitcast float %1793 to i32, !dbg !16
  %1795 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1794), !dbg !16
  %1796 = bitcast i32 %1795 to float, !dbg !16
  %1797 = select i1 %.not3, float %1785, float %1796, !dbg !16
  %1798 = select i1 %.not3, float %1796, float %1791, !dbg !16
  %1799 = select i1 %.not3, float %1792, float %1786, !dbg !16
  %1800 = bitcast float %1799 to i32, !dbg !16
  %1801 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1800), !dbg !16
  %1802 = bitcast i32 %1801 to float, !dbg !16
  %1803 = select i1 %.not3, float %1786, float %1802, !dbg !16
  %1804 = select i1 %.not3, float %1802, float %1792, !dbg !16
  %1805 = extractvalue [2 x <16 x float>] %37, 1, !dbg !16
  %1806 = extractelement <16 x float> %1805, i64 0, !dbg !16
  %1807 = extractelement <16 x float> %1805, i64 1, !dbg !16
  %1808 = extractelement <16 x float> %1805, i64 2, !dbg !16
  %1809 = extractelement <16 x float> %1805, i64 3, !dbg !16
  %1810 = select i1 %.not2, float %1807, float %1806, !dbg !16
  %1811 = bitcast float %1810 to i32, !dbg !16
  %1812 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1811), !dbg !16
  %1813 = bitcast i32 %1812 to float, !dbg !16
  %1814 = select i1 %.not2, float %1806, float %1813, !dbg !16
  %1815 = select i1 %.not2, float %1813, float %1807, !dbg !16
  %1816 = select i1 %.not2, float %1809, float %1808, !dbg !16
  %1817 = bitcast float %1816 to i32, !dbg !16
  %1818 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1817), !dbg !16
  %1819 = bitcast i32 %1818 to float, !dbg !16
  %1820 = select i1 %.not2, float %1808, float %1819, !dbg !16
  %1821 = select i1 %.not2, float %1819, float %1809, !dbg !16
  %1822 = select i1 %.not3, float %1820, float %1814, !dbg !16
  %1823 = bitcast float %1822 to i32, !dbg !16
  %1824 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1823), !dbg !16
  %1825 = select i1 %.not3, float %1821, float %1815, !dbg !16
  %1826 = bitcast float %1825 to i32, !dbg !16
  %1827 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1826), !dbg !16
  %1828 = extractvalue [2 x <1 x float>] %38, 1, !dbg !19
  %1829 = shufflevector <1 x float> %1828, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1830 = extractelement <16 x float> %1776, i64 4, !dbg !16
  %1831 = extractelement <16 x float> %1776, i64 5, !dbg !16
  %1832 = extractelement <16 x float> %1776, i64 6, !dbg !16
  %1833 = extractelement <16 x float> %1776, i64 7, !dbg !16
  %1834 = select i1 %.not2, float %1831, float %1830, !dbg !16
  %1835 = bitcast float %1834 to i32, !dbg !16
  %1836 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1835), !dbg !16
  %1837 = bitcast i32 %1836 to float, !dbg !16
  %1838 = select i1 %.not2, float %1830, float %1837, !dbg !16
  %1839 = select i1 %.not2, float %1837, float %1831, !dbg !16
  %1840 = select i1 %.not2, float %1833, float %1832, !dbg !16
  %1841 = bitcast float %1840 to i32, !dbg !16
  %1842 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1841), !dbg !16
  %1843 = bitcast i32 %1842 to float, !dbg !16
  %1844 = select i1 %.not2, float %1832, float %1843, !dbg !16
  %1845 = select i1 %.not2, float %1843, float %1833, !dbg !16
  %1846 = select i1 %.not3, float %1844, float %1838, !dbg !16
  %1847 = bitcast float %1846 to i32, !dbg !16
  %1848 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1847), !dbg !16
  %1849 = bitcast i32 %1848 to float, !dbg !16
  %1850 = select i1 %.not3, float %1838, float %1849, !dbg !16
  %1851 = select i1 %.not3, float %1849, float %1844, !dbg !16
  %1852 = select i1 %.not3, float %1845, float %1839, !dbg !16
  %1853 = bitcast float %1852 to i32, !dbg !16
  %1854 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1853), !dbg !16
  %1855 = bitcast i32 %1854 to float, !dbg !16
  %1856 = select i1 %.not3, float %1839, float %1855, !dbg !16
  %1857 = select i1 %.not3, float %1855, float %1845, !dbg !16
  %1858 = extractelement <16 x float> %1805, i64 4, !dbg !16
  %1859 = extractelement <16 x float> %1805, i64 5, !dbg !16
  %1860 = extractelement <16 x float> %1805, i64 6, !dbg !16
  %1861 = extractelement <16 x float> %1805, i64 7, !dbg !16
  %1862 = select i1 %.not2, float %1859, float %1858, !dbg !16
  %1863 = bitcast float %1862 to i32, !dbg !16
  %1864 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1863), !dbg !16
  %1865 = bitcast i32 %1864 to float, !dbg !16
  %1866 = select i1 %.not2, float %1858, float %1865, !dbg !16
  %1867 = select i1 %.not2, float %1865, float %1859, !dbg !16
  %1868 = select i1 %.not2, float %1861, float %1860, !dbg !16
  %1869 = bitcast float %1868 to i32, !dbg !16
  %1870 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1869), !dbg !16
  %1871 = bitcast i32 %1870 to float, !dbg !16
  %1872 = select i1 %.not2, float %1860, float %1871, !dbg !16
  %1873 = select i1 %.not2, float %1871, float %1861, !dbg !16
  %1874 = select i1 %.not3, float %1872, float %1866, !dbg !16
  %1875 = bitcast float %1874 to i32, !dbg !16
  %1876 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1875), !dbg !16
  %1877 = select i1 %.not3, float %1873, float %1867, !dbg !16
  %1878 = bitcast float %1877 to i32, !dbg !16
  %1879 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1878), !dbg !16
  %1880 = extractelement <16 x float> %1776, i64 8, !dbg !16
  %1881 = extractelement <16 x float> %1776, i64 9, !dbg !16
  %1882 = extractelement <16 x float> %1776, i64 10, !dbg !16
  %1883 = extractelement <16 x float> %1776, i64 11, !dbg !16
  %1884 = select i1 %.not2, float %1881, float %1880, !dbg !16
  %1885 = bitcast float %1884 to i32, !dbg !16
  %1886 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1885), !dbg !16
  %1887 = bitcast i32 %1886 to float, !dbg !16
  %1888 = select i1 %.not2, float %1880, float %1887, !dbg !16
  %1889 = select i1 %.not2, float %1887, float %1881, !dbg !16
  %1890 = select i1 %.not2, float %1883, float %1882, !dbg !16
  %1891 = bitcast float %1890 to i32, !dbg !16
  %1892 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1891), !dbg !16
  %1893 = bitcast i32 %1892 to float, !dbg !16
  %1894 = select i1 %.not2, float %1882, float %1893, !dbg !16
  %1895 = select i1 %.not2, float %1893, float %1883, !dbg !16
  %1896 = select i1 %.not3, float %1894, float %1888, !dbg !16
  %1897 = bitcast float %1896 to i32, !dbg !16
  %1898 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1897), !dbg !16
  %1899 = bitcast i32 %1898 to float, !dbg !16
  %1900 = select i1 %.not3, float %1888, float %1899, !dbg !16
  %1901 = select i1 %.not3, float %1899, float %1894, !dbg !16
  %1902 = select i1 %.not3, float %1895, float %1889, !dbg !16
  %1903 = bitcast float %1902 to i32, !dbg !16
  %1904 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1903), !dbg !16
  %1905 = bitcast i32 %1904 to float, !dbg !16
  %1906 = select i1 %.not3, float %1889, float %1905, !dbg !16
  %1907 = select i1 %.not3, float %1905, float %1895, !dbg !16
  %1908 = extractelement <16 x float> %1805, i64 8, !dbg !16
  %1909 = extractelement <16 x float> %1805, i64 9, !dbg !16
  %1910 = extractelement <16 x float> %1805, i64 10, !dbg !16
  %1911 = extractelement <16 x float> %1805, i64 11, !dbg !16
  %1912 = select i1 %.not2, float %1909, float %1908, !dbg !16
  %1913 = bitcast float %1912 to i32, !dbg !16
  %1914 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1913), !dbg !16
  %1915 = bitcast i32 %1914 to float, !dbg !16
  %1916 = select i1 %.not2, float %1908, float %1915, !dbg !16
  %1917 = select i1 %.not2, float %1915, float %1909, !dbg !16
  %1918 = select i1 %.not2, float %1911, float %1910, !dbg !16
  %1919 = bitcast float %1918 to i32, !dbg !16
  %1920 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1919), !dbg !16
  %1921 = bitcast i32 %1920 to float, !dbg !16
  %1922 = select i1 %.not2, float %1910, float %1921, !dbg !16
  %1923 = select i1 %.not2, float %1921, float %1911, !dbg !16
  %1924 = select i1 %.not3, float %1922, float %1916, !dbg !16
  %1925 = bitcast float %1924 to i32, !dbg !16
  %1926 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1925), !dbg !16
  %1927 = select i1 %.not3, float %1923, float %1917, !dbg !16
  %1928 = bitcast float %1927 to i32, !dbg !16
  %1929 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1928), !dbg !16
  %1930 = extractelement <16 x float> %1776, i64 12, !dbg !16
  %1931 = extractelement <16 x float> %1776, i64 13, !dbg !16
  %1932 = extractelement <16 x float> %1776, i64 14, !dbg !16
  %1933 = extractelement <16 x float> %1776, i64 15, !dbg !16
  %1934 = select i1 %.not2, float %1931, float %1930, !dbg !16
  %1935 = bitcast float %1934 to i32, !dbg !16
  %1936 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1935), !dbg !16
  %1937 = bitcast i32 %1936 to float, !dbg !16
  %1938 = select i1 %.not2, float %1930, float %1937, !dbg !16
  %1939 = select i1 %.not2, float %1937, float %1931, !dbg !16
  %1940 = select i1 %.not2, float %1933, float %1932, !dbg !16
  %1941 = bitcast float %1940 to i32, !dbg !16
  %1942 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1941), !dbg !16
  %1943 = bitcast i32 %1942 to float, !dbg !16
  %1944 = select i1 %.not2, float %1932, float %1943, !dbg !16
  %1945 = select i1 %.not2, float %1943, float %1933, !dbg !16
  %1946 = select i1 %.not3, float %1944, float %1938, !dbg !16
  %1947 = bitcast float %1946 to i32, !dbg !16
  %1948 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1947), !dbg !16
  %1949 = bitcast i32 %1948 to float, !dbg !16
  %1950 = select i1 %.not3, float %1938, float %1949, !dbg !16
  %1951 = select i1 %.not3, float %1949, float %1944, !dbg !16
  %1952 = select i1 %.not3, float %1945, float %1939, !dbg !16
  %1953 = bitcast float %1952 to i32, !dbg !16
  %1954 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1953), !dbg !16
  %1955 = bitcast i32 %1954 to float, !dbg !16
  %1956 = select i1 %.not3, float %1939, float %1955, !dbg !16
  %1957 = select i1 %.not3, float %1955, float %1945, !dbg !16
  %1958 = extractelement <16 x float> %1805, i64 12, !dbg !16
  %1959 = extractelement <16 x float> %1805, i64 13, !dbg !16
  %1960 = extractelement <16 x float> %1805, i64 14, !dbg !16
  %1961 = extractelement <16 x float> %1805, i64 15, !dbg !16
  %1962 = select i1 %.not2, float %1959, float %1958, !dbg !16
  %1963 = bitcast float %1962 to i32, !dbg !16
  %1964 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1963), !dbg !16
  %1965 = bitcast i32 %1964 to float, !dbg !16
  %1966 = select i1 %.not2, float %1958, float %1965, !dbg !16
  %1967 = select i1 %.not2, float %1965, float %1959, !dbg !16
  %1968 = select i1 %.not2, float %1961, float %1960, !dbg !16
  %1969 = bitcast float %1968 to i32, !dbg !16
  %1970 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1573, i32 %1969), !dbg !16
  %1971 = bitcast i32 %1970 to float, !dbg !16
  %1972 = select i1 %.not2, float %1960, float %1971, !dbg !16
  %1973 = select i1 %.not2, float %1971, float %1961, !dbg !16
  %1974 = select i1 %.not3, float %1972, float %1966, !dbg !16
  %1975 = bitcast float %1974 to i32, !dbg !16
  %1976 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1975), !dbg !16
  %1977 = select i1 %.not3, float %1973, float %1967, !dbg !16
  %1978 = bitcast float %1977 to i32, !dbg !16
  %1979 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1589, i32 %1978), !dbg !16
  br i1 true, label %1980, label %2104, !dbg !21

1980:                                             ; preds = %1557
  %1981 = bitcast i32 %1976 to float, !dbg !16
  %1982 = select i1 %.not3, float %1966, float %1981, !dbg !16
  %1983 = insertelement <4 x float> poison, float %1982, i64 0, !dbg !19
  %1984 = bitcast i32 %1979 to float, !dbg !16
  %1985 = select i1 %.not3, float %1967, float %1984, !dbg !16
  %1986 = insertelement <4 x float> %1983, float %1985, i64 1, !dbg !19
  %1987 = select i1 %.not3, float %1981, float %1972, !dbg !16
  %1988 = insertelement <4 x float> %1986, float %1987, i64 2, !dbg !19
  %1989 = select i1 %.not3, float %1984, float %1973, !dbg !16
  %1990 = insertelement <4 x float> %1988, float %1989, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1829, !dbg !19
  %normalized = fmul <4 x float> %1990, %row.reciprocal, !dbg !19
  %1991 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1992 = extractelement <4 x half> %1991, i64 3, !dbg !20
  %1993 = extractelement <4 x half> %1991, i64 2, !dbg !20
  %1994 = extractelement <4 x half> %1991, i64 1, !dbg !20
  %1995 = extractelement <4 x half> %1991, i64 0, !dbg !20
  %1996 = bitcast i32 %1926 to float, !dbg !16
  %1997 = select i1 %.not3, float %1916, float %1996, !dbg !16
  %1998 = insertelement <4 x float> poison, float %1997, i64 0, !dbg !19
  %1999 = bitcast i32 %1929 to float, !dbg !16
  %2000 = select i1 %.not3, float %1917, float %1999, !dbg !16
  %2001 = insertelement <4 x float> %1998, float %2000, i64 1, !dbg !19
  %2002 = select i1 %.not3, float %1996, float %1922, !dbg !16
  %2003 = insertelement <4 x float> %2001, float %2002, i64 2, !dbg !19
  %2004 = select i1 %.not3, float %1999, float %1923, !dbg !16
  %2005 = insertelement <4 x float> %2003, float %2004, i64 3, !dbg !19
  %normalized73 = fmul <4 x float> %2005, %row.reciprocal, !dbg !19
  %2006 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !20
  %2007 = extractelement <4 x half> %2006, i64 3, !dbg !20
  %2008 = extractelement <4 x half> %2006, i64 2, !dbg !20
  %2009 = extractelement <4 x half> %2006, i64 1, !dbg !20
  %2010 = extractelement <4 x half> %2006, i64 0, !dbg !20
  %2011 = bitcast i32 %1876 to float, !dbg !16
  %2012 = select i1 %.not3, float %1866, float %2011, !dbg !16
  %2013 = insertelement <4 x float> poison, float %2012, i64 0, !dbg !19
  %2014 = bitcast i32 %1879 to float, !dbg !16
  %2015 = select i1 %.not3, float %1867, float %2014, !dbg !16
  %2016 = insertelement <4 x float> %2013, float %2015, i64 1, !dbg !19
  %2017 = select i1 %.not3, float %2011, float %1872, !dbg !16
  %2018 = insertelement <4 x float> %2016, float %2017, i64 2, !dbg !19
  %2019 = select i1 %.not3, float %2014, float %1873, !dbg !16
  %2020 = insertelement <4 x float> %2018, float %2019, i64 3, !dbg !19
  %normalized74 = fmul <4 x float> %2020, %row.reciprocal, !dbg !19
  %2021 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %2022 = extractelement <4 x half> %2021, i64 3, !dbg !20
  %2023 = extractelement <4 x half> %2021, i64 2, !dbg !20
  %2024 = extractelement <4 x half> %2021, i64 1, !dbg !20
  %2025 = extractelement <4 x half> %2021, i64 0, !dbg !20
  %2026 = bitcast i32 %1824 to float, !dbg !16
  %2027 = select i1 %.not3, float %1814, float %2026, !dbg !16
  %2028 = insertelement <4 x float> poison, float %2027, i64 0, !dbg !19
  %2029 = bitcast i32 %1827 to float, !dbg !16
  %2030 = select i1 %.not3, float %1815, float %2029, !dbg !16
  %2031 = insertelement <4 x float> %2028, float %2030, i64 1, !dbg !19
  %2032 = select i1 %.not3, float %2026, float %1820, !dbg !16
  %2033 = insertelement <4 x float> %2031, float %2032, i64 2, !dbg !19
  %2034 = select i1 %.not3, float %2029, float %1821, !dbg !16
  %2035 = insertelement <4 x float> %2033, float %2034, i64 3, !dbg !19
  %normalized75 = fmul <4 x float> %2035, %row.reciprocal, !dbg !19
  %2036 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !20
  %2037 = extractelement <4 x half> %2036, i64 3, !dbg !20
  %2038 = extractelement <4 x half> %2036, i64 2, !dbg !20
  %2039 = extractelement <4 x half> %2036, i64 1, !dbg !20
  %2040 = extractelement <4 x half> %2036, i64 0, !dbg !20
  %2041 = bitcast i32 %1772 to float, !dbg !16
  %2042 = select i1 %.not3, float %1762, float %2041, !dbg !16
  %2043 = insertelement <4 x float> poison, float %2042, i64 0, !dbg !19
  %2044 = bitcast i32 %1775 to float, !dbg !16
  %2045 = select i1 %.not3, float %1763, float %2044, !dbg !16
  %2046 = insertelement <4 x float> %2043, float %2045, i64 1, !dbg !19
  %2047 = select i1 %.not3, float %2041, float %1768, !dbg !16
  %2048 = insertelement <4 x float> %2046, float %2047, i64 2, !dbg !19
  %2049 = select i1 %.not3, float %2044, float %1769, !dbg !16
  %2050 = insertelement <4 x float> %2048, float %2049, i64 3, !dbg !19
  %row.reciprocal76 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1625, !dbg !19
  %normalized77 = fmul <4 x float> %2050, %row.reciprocal76, !dbg !19
  %2051 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !20
  %2052 = extractelement <4 x half> %2051, i64 3, !dbg !20
  %2053 = extractelement <4 x half> %2051, i64 2, !dbg !20
  %2054 = extractelement <4 x half> %2051, i64 1, !dbg !20
  %2055 = extractelement <4 x half> %2051, i64 0, !dbg !20
  %2056 = bitcast i32 %1722 to float, !dbg !16
  %2057 = select i1 %.not3, float %1712, float %2056, !dbg !16
  %2058 = insertelement <4 x float> poison, float %2057, i64 0, !dbg !19
  %2059 = bitcast i32 %1725 to float, !dbg !16
  %2060 = select i1 %.not3, float %1713, float %2059, !dbg !16
  %2061 = insertelement <4 x float> %2058, float %2060, i64 1, !dbg !19
  %2062 = select i1 %.not3, float %2056, float %1718, !dbg !16
  %2063 = insertelement <4 x float> %2061, float %2062, i64 2, !dbg !19
  %2064 = select i1 %.not3, float %2059, float %1719, !dbg !16
  %2065 = insertelement <4 x float> %2063, float %2064, i64 3, !dbg !19
  %normalized78 = fmul <4 x float> %2065, %row.reciprocal76, !dbg !19
  %2066 = fptrunc <4 x float> %normalized78 to <4 x half>, !dbg !20
  %2067 = extractelement <4 x half> %2066, i64 3, !dbg !20
  %2068 = extractelement <4 x half> %2066, i64 2, !dbg !20
  %2069 = extractelement <4 x half> %2066, i64 1, !dbg !20
  %2070 = extractelement <4 x half> %2066, i64 0, !dbg !20
  %2071 = bitcast i32 %1672 to float, !dbg !16
  %2072 = select i1 %.not3, float %1662, float %2071, !dbg !16
  %2073 = insertelement <4 x float> poison, float %2072, i64 0, !dbg !19
  %2074 = bitcast i32 %1675 to float, !dbg !16
  %2075 = select i1 %.not3, float %1663, float %2074, !dbg !16
  %2076 = insertelement <4 x float> %2073, float %2075, i64 1, !dbg !19
  %2077 = select i1 %.not3, float %2071, float %1668, !dbg !16
  %2078 = insertelement <4 x float> %2076, float %2077, i64 2, !dbg !19
  %2079 = select i1 %.not3, float %2074, float %1669, !dbg !16
  %2080 = insertelement <4 x float> %2078, float %2079, i64 3, !dbg !19
  %normalized79 = fmul <4 x float> %2080, %row.reciprocal76, !dbg !19
  %2081 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !20
  %2082 = extractelement <4 x half> %2081, i64 3, !dbg !20
  %2083 = extractelement <4 x half> %2081, i64 2, !dbg !20
  %2084 = extractelement <4 x half> %2081, i64 1, !dbg !20
  %2085 = extractelement <4 x half> %2081, i64 0, !dbg !20
  %2086 = bitcast i32 %1620 to float, !dbg !16
  %2087 = select i1 %.not3, float %1610, float %2086, !dbg !16
  %2088 = insertelement <4 x float> poison, float %2087, i64 0, !dbg !19
  %2089 = bitcast i32 %1623 to float, !dbg !16
  %2090 = select i1 %.not3, float %1611, float %2089, !dbg !16
  %2091 = insertelement <4 x float> %2088, float %2090, i64 1, !dbg !19
  %2092 = select i1 %.not3, float %2086, float %1616, !dbg !16
  %2093 = insertelement <4 x float> %2091, float %2092, i64 2, !dbg !19
  %2094 = select i1 %.not3, float %2089, float %1617, !dbg !16
  %2095 = insertelement <4 x float> %2093, float %2094, i64 3, !dbg !19
  %normalized80 = fmul <4 x float> %2095, %row.reciprocal76, !dbg !19
  %2096 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !20
  %2097 = extractelement <4 x half> %2096, i64 3, !dbg !20
  %2098 = extractelement <4 x half> %2096, i64 2, !dbg !20
  %2099 = extractelement <4 x half> %2096, i64 1, !dbg !20
  %2100 = extractelement <4 x half> %2096, i64 0, !dbg !20
  %2101 = add nsw i64 %8, %13, !dbg !21
  %2102 = shl nsw i64 %2101, 19, !dbg !21
  %2103 = and i64 %2102, 16252928, !dbg !21
  %address.thread846 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local847 = zext nneg i32 %address.thread846 to i64, !dbg !13
  %address.local848 = add nsw i64 %16, %address.local847, !dbg !21
  %address.local849 = lshr i64 %address.local847, 2, !dbg !15
  %address.local850 = and i64 %address.local849, 48, !dbg !15
  %address.local851 = add nsw i64 %address.local848, %address.local850, !dbg !21
  %address.local852 = lshr i64 %address.local847, 4, !dbg !13
  %address.local853 = mul nsw i64 %address.local852, -16, !dbg !15
  %address.local854 = add nsw i64 %address.local851, %address.local853, !dbg !21
  %address.local855 = add nsw i64 %address.local854, %17, !dbg !21
  %address.local856 = shl nuw nsw i64 %address.local855, 7, !dbg !21
  %address.local857 = add nuw nsw i64 %2103, %address.local856, !dbg !21
  %address.local858 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local859 = lshr i64 %address.local847, 2, !dbg !13
  %address.local860 = and i64 %address.local859, 12, !dbg !13
  %address.local861 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local858, i64 %address.local860, !dbg !21
  store half %2100, ptr addrspace(1) %address.local861, align 2, !dbg !21
  %address.local862 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local863 = or disjoint i64 %address.local860, 1, !dbg !15
  %address.local864 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local862, i64 %address.local863, !dbg !21
  store half %2099, ptr addrspace(1) %address.local864, align 2, !dbg !21
  %address.local865 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local866 = or disjoint i64 %address.local860, 2, !dbg !15
  %address.local867 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local865, i64 %address.local866, !dbg !21
  store half %2098, ptr addrspace(1) %address.local867, align 2, !dbg !21
  %address.local868 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local869 = or disjoint i64 %address.local860, 3, !dbg !15
  %address.local870 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local868, i64 %address.local869, !dbg !21
  store half %2097, ptr addrspace(1) %address.local870, align 2, !dbg !21
  %address.local871 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local872 = or disjoint i64 %address.local860, 16, !dbg !15
  %address.local873 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local871, i64 %address.local872, !dbg !21
  store half %2085, ptr addrspace(1) %address.local873, align 2, !dbg !21
  %address.local874 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local875 = or disjoint i64 %address.local860, 17, !dbg !15
  %address.local876 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local874, i64 %address.local875, !dbg !21
  store half %2084, ptr addrspace(1) %address.local876, align 2, !dbg !21
  %address.local877 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local878 = or disjoint i64 %address.local860, 18, !dbg !15
  %address.local879 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local877, i64 %address.local878, !dbg !21
  store half %2083, ptr addrspace(1) %address.local879, align 2, !dbg !21
  %address.local880 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local881 = or disjoint i64 %address.local860, 19, !dbg !15
  %address.local882 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local881, !dbg !21
  store half %2082, ptr addrspace(1) %address.local882, align 2, !dbg !21
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local884 = or disjoint i64 %address.local860, 32, !dbg !15
  %address.local885 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local883, i64 %address.local884, !dbg !21
  store half %2070, ptr addrspace(1) %address.local885, align 2, !dbg !21
  %address.local886 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local887 = or disjoint i64 %address.local860, 33, !dbg !15
  %address.local888 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local886, i64 %address.local887, !dbg !21
  store half %2069, ptr addrspace(1) %address.local888, align 2, !dbg !21
  %address.local889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local890 = or disjoint i64 %address.local860, 34, !dbg !15
  %address.local891 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local889, i64 %address.local890, !dbg !21
  store half %2068, ptr addrspace(1) %address.local891, align 2, !dbg !21
  %address.local892 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local893 = or disjoint i64 %address.local860, 35, !dbg !15
  %address.local894 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local892, i64 %address.local893, !dbg !21
  store half %2067, ptr addrspace(1) %address.local894, align 2, !dbg !21
  %address.local895 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local896 = or disjoint i64 %address.local860, 48, !dbg !15
  %address.local897 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local895, i64 %address.local896, !dbg !21
  store half %2055, ptr addrspace(1) %address.local897, align 2, !dbg !21
  %address.local898 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local899 = or disjoint i64 %address.local860, 49, !dbg !15
  %address.local900 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local898, i64 %address.local899, !dbg !21
  store half %2054, ptr addrspace(1) %address.local900, align 2, !dbg !21
  %address.local901 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local902 = or disjoint i64 %address.local860, 50, !dbg !15
  %address.local903 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local901, i64 %address.local902, !dbg !21
  store half %2053, ptr addrspace(1) %address.local903, align 2, !dbg !21
  %address.local904 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local857, !dbg !21
  %address.local905 = or i64 %address.local859, 51, !dbg !15
  %address.local906 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local904, i64 %address.local905, !dbg !21
  store half %2052, ptr addrspace(1) %address.local906, align 2, !dbg !21
  %address.local907 = shl nsw i64 %address.local855, 7, !dbg !21
  %address.local908 = add i64 %address.local907, 8192, !dbg !21
  %address.local909 = add nuw nsw i64 %2103, %address.local908, !dbg !21
  %address.local910 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local911 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local910, i64 %address.local860, !dbg !21
  store half %2040, ptr addrspace(1) %address.local911, align 2, !dbg !21
  %address.local912 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local913 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local912, i64 %address.local863, !dbg !21
  store half %2039, ptr addrspace(1) %address.local913, align 2, !dbg !21
  %address.local914 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local915 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local914, i64 %address.local866, !dbg !21
  store half %2038, ptr addrspace(1) %address.local915, align 2, !dbg !21
  %address.local916 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local917 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local869, !dbg !21
  store half %2037, ptr addrspace(1) %address.local917, align 2, !dbg !21
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local919 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local918, i64 %address.local872, !dbg !21
  store half %2025, ptr addrspace(1) %address.local919, align 2, !dbg !21
  %address.local920 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local921 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local920, i64 %address.local875, !dbg !21
  store half %2024, ptr addrspace(1) %address.local921, align 2, !dbg !21
  %address.local922 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local923 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local878, !dbg !21
  store half %2023, ptr addrspace(1) %address.local923, align 2, !dbg !21
  %address.local924 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local925 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local924, i64 %address.local881, !dbg !21
  store half %2022, ptr addrspace(1) %address.local925, align 2, !dbg !21
  %address.local926 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local926, i64 %address.local884, !dbg !21
  store half %2010, ptr addrspace(1) %address.local927, align 2, !dbg !21
  %address.local928 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local929 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local928, i64 %address.local887, !dbg !21
  store half %2009, ptr addrspace(1) %address.local929, align 2, !dbg !21
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local931 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local930, i64 %address.local890, !dbg !21
  store half %2008, ptr addrspace(1) %address.local931, align 2, !dbg !21
  %address.local932 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local933 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local932, i64 %address.local893, !dbg !21
  store half %2007, ptr addrspace(1) %address.local933, align 2, !dbg !21
  %address.local934 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local934, i64 %address.local896, !dbg !21
  store half %1995, ptr addrspace(1) %address.local935, align 2, !dbg !21
  %address.local936 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local936, i64 %address.local899, !dbg !21
  store half %1994, ptr addrspace(1) %address.local937, align 2, !dbg !21
  %address.local938 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local938, i64 %address.local902, !dbg !21
  store half %1993, ptr addrspace(1) %address.local939, align 2, !dbg !21
  %address.local940 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local909, !dbg !21
  %address.local941 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local940, i64 %address.local905, !dbg !21
  store half %1992, ptr addrspace(1) %address.local941, align 2, !dbg !21
  br label %2104, !dbg !21

2104:                                             ; preds = %1980, %1557
  br i1 true, label %2105, label %2179, !dbg !21

2105:                                             ; preds = %2104
  %2106 = insertelement <4 x float> poison, float %1950, i64 0, !dbg !19
  %2107 = insertelement <4 x float> %2106, float %1956, i64 1, !dbg !19
  %2108 = insertelement <4 x float> %2107, float %1951, i64 2, !dbg !19
  %2109 = insertelement <4 x float> %2108, float %1957, i64 3, !dbg !19
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1829, !dbg !19
  %normalized82 = fmul <4 x float> %2109, %row.reciprocal81, !dbg !19
  %2110 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !20
  %2111 = extractelement <4 x half> %2110, i64 3, !dbg !20
  %2112 = extractelement <4 x half> %2110, i64 2, !dbg !20
  %2113 = extractelement <4 x half> %2110, i64 1, !dbg !20
  %2114 = extractelement <4 x half> %2110, i64 0, !dbg !20
  %2115 = insertelement <4 x float> poison, float %1900, i64 0, !dbg !19
  %2116 = insertelement <4 x float> %2115, float %1906, i64 1, !dbg !19
  %2117 = insertelement <4 x float> %2116, float %1901, i64 2, !dbg !19
  %2118 = insertelement <4 x float> %2117, float %1907, i64 3, !dbg !19
  %normalized83 = fmul <4 x float> %2118, %row.reciprocal81, !dbg !19
  %2119 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !20
  %2120 = extractelement <4 x half> %2119, i64 3, !dbg !20
  %2121 = extractelement <4 x half> %2119, i64 2, !dbg !20
  %2122 = extractelement <4 x half> %2119, i64 1, !dbg !20
  %2123 = extractelement <4 x half> %2119, i64 0, !dbg !20
  %2124 = insertelement <4 x float> poison, float %1850, i64 0, !dbg !19
  %2125 = insertelement <4 x float> %2124, float %1856, i64 1, !dbg !19
  %2126 = insertelement <4 x float> %2125, float %1851, i64 2, !dbg !19
  %2127 = insertelement <4 x float> %2126, float %1857, i64 3, !dbg !19
  %normalized84 = fmul <4 x float> %2127, %row.reciprocal81, !dbg !19
  %2128 = fptrunc <4 x float> %normalized84 to <4 x half>, !dbg !20
  %2129 = extractelement <4 x half> %2128, i64 3, !dbg !20
  %2130 = extractelement <4 x half> %2128, i64 2, !dbg !20
  %2131 = extractelement <4 x half> %2128, i64 1, !dbg !20
  %2132 = extractelement <4 x half> %2128, i64 0, !dbg !20
  %2133 = insertelement <4 x float> poison, float %1797, i64 0, !dbg !19
  %2134 = insertelement <4 x float> %2133, float %1803, i64 1, !dbg !19
  %2135 = insertelement <4 x float> %2134, float %1798, i64 2, !dbg !19
  %2136 = insertelement <4 x float> %2135, float %1804, i64 3, !dbg !19
  %normalized85 = fmul <4 x float> %2136, %row.reciprocal81, !dbg !19
  %2137 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !20
  %2138 = extractelement <4 x half> %2137, i64 3, !dbg !20
  %2139 = extractelement <4 x half> %2137, i64 2, !dbg !20
  %2140 = extractelement <4 x half> %2137, i64 1, !dbg !20
  %2141 = extractelement <4 x half> %2137, i64 0, !dbg !20
  %2142 = insertelement <4 x float> poison, float %1746, i64 0, !dbg !19
  %2143 = insertelement <4 x float> %2142, float %1752, i64 1, !dbg !19
  %2144 = insertelement <4 x float> %2143, float %1747, i64 2, !dbg !19
  %2145 = insertelement <4 x float> %2144, float %1753, i64 3, !dbg !19
  %row.reciprocal86 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1625, !dbg !19
  %normalized87 = fmul <4 x float> %2145, %row.reciprocal86, !dbg !19
  %2146 = fptrunc <4 x float> %normalized87 to <4 x half>, !dbg !20
  %2147 = extractelement <4 x half> %2146, i64 3, !dbg !20
  %2148 = extractelement <4 x half> %2146, i64 2, !dbg !20
  %2149 = extractelement <4 x half> %2146, i64 1, !dbg !20
  %2150 = extractelement <4 x half> %2146, i64 0, !dbg !20
  %2151 = insertelement <4 x float> poison, float %1696, i64 0, !dbg !19
  %2152 = insertelement <4 x float> %2151, float %1702, i64 1, !dbg !19
  %2153 = insertelement <4 x float> %2152, float %1697, i64 2, !dbg !19
  %2154 = insertelement <4 x float> %2153, float %1703, i64 3, !dbg !19
  %normalized88 = fmul <4 x float> %2154, %row.reciprocal86, !dbg !19
  %2155 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !20
  %2156 = extractelement <4 x half> %2155, i64 3, !dbg !20
  %2157 = extractelement <4 x half> %2155, i64 2, !dbg !20
  %2158 = extractelement <4 x half> %2155, i64 1, !dbg !20
  %2159 = extractelement <4 x half> %2155, i64 0, !dbg !20
  %2160 = insertelement <4 x float> poison, float %1646, i64 0, !dbg !19
  %2161 = insertelement <4 x float> %2160, float %1652, i64 1, !dbg !19
  %2162 = insertelement <4 x float> %2161, float %1647, i64 2, !dbg !19
  %2163 = insertelement <4 x float> %2162, float %1653, i64 3, !dbg !19
  %normalized89 = fmul <4 x float> %2163, %row.reciprocal86, !dbg !19
  %2164 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !20
  %2165 = extractelement <4 x half> %2164, i64 3, !dbg !20
  %2166 = extractelement <4 x half> %2164, i64 2, !dbg !20
  %2167 = extractelement <4 x half> %2164, i64 1, !dbg !20
  %2168 = extractelement <4 x half> %2164, i64 0, !dbg !20
  %2169 = insertelement <4 x float> poison, float %1593, i64 0, !dbg !19
  %2170 = insertelement <4 x float> %2169, float %1599, i64 1, !dbg !19
  %2171 = insertelement <4 x float> %2170, float %1594, i64 2, !dbg !19
  %2172 = insertelement <4 x float> %2171, float %1600, i64 3, !dbg !19
  %normalized90 = fmul <4 x float> %2172, %row.reciprocal86, !dbg !19
  %2173 = fptrunc <4 x float> %normalized90 to <4 x half>, !dbg !20
  %2174 = extractelement <4 x half> %2173, i64 3, !dbg !20
  %2175 = extractelement <4 x half> %2173, i64 2, !dbg !20
  %2176 = extractelement <4 x half> %2173, i64 1, !dbg !20
  %2177 = extractelement <4 x half> %2173, i64 0, !dbg !20
  %2178 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread942 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local943 = zext nneg i32 %address.thread942 to i64, !dbg !13
  %address.local944 = add nsw i64 %16, %address.local943, !dbg !21
  %address.local945 = lshr i64 %address.local943, 2, !dbg !15
  %address.local946 = and i64 %address.local945, 48, !dbg !15
  %address.local947 = add nsw i64 %address.local944, %address.local946, !dbg !21
  %address.local948 = lshr i64 %address.local943, 4, !dbg !13
  %address.local949 = mul nsw i64 %address.local948, -16, !dbg !15
  %address.local950 = add nsw i64 %address.local947, %address.local949, !dbg !21
  %address.local951 = add nsw i64 %address.local950, %17, !dbg !21
  %address.local952 = shl nuw nsw i64 %address.local951, 7, !dbg !21
  %address.local953 = add nuw nsw i64 %2178, %address.local952, !dbg !21
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local955 = lshr i64 %address.local943, 2, !dbg !13
  %address.local956 = and i64 %address.local955, 12, !dbg !13
  %address.local957 = or disjoint i64 %address.local956, 64, !dbg !15
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local954, i64 %address.local957, !dbg !21
  store half %2177, ptr addrspace(1) %address.local958, align 2, !dbg !21
  %address.local959 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local960 = or disjoint i64 %address.local956, 65, !dbg !15
  %address.local961 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local959, i64 %address.local960, !dbg !21
  store half %2176, ptr addrspace(1) %address.local961, align 2, !dbg !21
  %address.local962 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local963 = or disjoint i64 %address.local956, 66, !dbg !15
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local963, !dbg !21
  store half %2175, ptr addrspace(1) %address.local964, align 2, !dbg !21
  %address.local965 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local966 = or disjoint i64 %address.local956, 67, !dbg !15
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local965, i64 %address.local966, !dbg !21
  store half %2174, ptr addrspace(1) %address.local967, align 2, !dbg !21
  %address.local968 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local969 = or disjoint i64 %address.local956, 80, !dbg !15
  %address.local970 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local968, i64 %address.local969, !dbg !21
  store half %2168, ptr addrspace(1) %address.local970, align 2, !dbg !21
  %address.local971 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local972 = or disjoint i64 %address.local956, 81, !dbg !15
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local971, i64 %address.local972, !dbg !21
  store half %2167, ptr addrspace(1) %address.local973, align 2, !dbg !21
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local975 = or disjoint i64 %address.local956, 82, !dbg !15
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local974, i64 %address.local975, !dbg !21
  store half %2166, ptr addrspace(1) %address.local976, align 2, !dbg !21
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local978 = or disjoint i64 %address.local956, 83, !dbg !15
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local977, i64 %address.local978, !dbg !21
  store half %2165, ptr addrspace(1) %address.local979, align 2, !dbg !21
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local981 = or disjoint i64 %address.local956, 96, !dbg !15
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local980, i64 %address.local981, !dbg !21
  store half %2159, ptr addrspace(1) %address.local982, align 2, !dbg !21
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local984 = or disjoint i64 %address.local956, 97, !dbg !15
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local983, i64 %address.local984, !dbg !21
  store half %2158, ptr addrspace(1) %address.local985, align 2, !dbg !21
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local987 = or disjoint i64 %address.local956, 98, !dbg !15
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local986, i64 %address.local987, !dbg !21
  store half %2157, ptr addrspace(1) %address.local988, align 2, !dbg !21
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local990 = or disjoint i64 %address.local956, 99, !dbg !15
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local989, i64 %address.local990, !dbg !21
  store half %2156, ptr addrspace(1) %address.local991, align 2, !dbg !21
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local993 = or disjoint i64 %address.local956, 112, !dbg !15
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local992, i64 %address.local993, !dbg !21
  store half %2150, ptr addrspace(1) %address.local994, align 2, !dbg !21
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local996 = or disjoint i64 %address.local956, 113, !dbg !15
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local995, i64 %address.local996, !dbg !21
  store half %2149, ptr addrspace(1) %address.local997, align 2, !dbg !21
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local999 = or disjoint i64 %address.local956, 114, !dbg !15
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local998, i64 %address.local999, !dbg !21
  store half %2148, ptr addrspace(1) %address.local1000, align 2, !dbg !21
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local953, !dbg !21
  %address.local1002 = or i64 %address.local955, 115, !dbg !15
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1001, i64 %address.local1002, !dbg !21
  store half %2147, ptr addrspace(1) %address.local1003, align 2, !dbg !21
  %address.local1004 = shl nsw i64 %address.local951, 7, !dbg !21
  %address.local1005 = add i64 %address.local1004, 8192, !dbg !21
  %address.local1006 = add nuw nsw i64 %2178, %address.local1005, !dbg !21
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1007, i64 %address.local957, !dbg !21
  store half %2141, ptr addrspace(1) %address.local1008, align 2, !dbg !21
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1009, i64 %address.local960, !dbg !21
  store half %2140, ptr addrspace(1) %address.local1010, align 2, !dbg !21
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1011, i64 %address.local963, !dbg !21
  store half %2139, ptr addrspace(1) %address.local1012, align 2, !dbg !21
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local966, !dbg !21
  store half %2138, ptr addrspace(1) %address.local1014, align 2, !dbg !21
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1015, i64 %address.local969, !dbg !21
  store half %2132, ptr addrspace(1) %address.local1016, align 2, !dbg !21
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local972, !dbg !21
  store half %2131, ptr addrspace(1) %address.local1018, align 2, !dbg !21
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1019, i64 %address.local975, !dbg !21
  store half %2130, ptr addrspace(1) %address.local1020, align 2, !dbg !21
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1021, i64 %address.local978, !dbg !21
  store half %2129, ptr addrspace(1) %address.local1022, align 2, !dbg !21
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local981, !dbg !21
  store half %2123, ptr addrspace(1) %address.local1024, align 2, !dbg !21
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1025, i64 %address.local984, !dbg !21
  store half %2122, ptr addrspace(1) %address.local1026, align 2, !dbg !21
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1027, i64 %address.local987, !dbg !21
  store half %2121, ptr addrspace(1) %address.local1028, align 2, !dbg !21
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1029, i64 %address.local990, !dbg !21
  store half %2120, ptr addrspace(1) %address.local1030, align 2, !dbg !21
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local993, !dbg !21
  store half %2114, ptr addrspace(1) %address.local1032, align 2, !dbg !21
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local996, !dbg !21
  store half %2113, ptr addrspace(1) %address.local1034, align 2, !dbg !21
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1035, i64 %address.local999, !dbg !21
  store half %2112, ptr addrspace(1) %address.local1036, align 2, !dbg !21
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1006, !dbg !21
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1037, i64 %address.local1002, !dbg !21
  store half %2111, ptr addrspace(1) %address.local1038, align 2, !dbg !21
  br label %2179, !dbg !21

2179:                                             ; preds = %2105, %2104
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

!0 = !DIFile(filename: "test_input_spidD.mlir", directory: "3rd/deepgengraph/test")
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
