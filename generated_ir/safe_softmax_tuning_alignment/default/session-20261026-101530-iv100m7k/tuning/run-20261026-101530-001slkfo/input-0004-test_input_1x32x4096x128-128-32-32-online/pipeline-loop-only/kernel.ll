; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/pipeline-loop-only/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/pipeline-loop-only/input.mlir"

@shm_0 = addrspace(3) global [20480 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !38 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %10 = sext i32 %9 to i64, !dbg !5
  %11 = shl nsw i64 %10, 7, !dbg !9
  %.lobit = ashr i32 %9, 31, !dbg !12
  %12 = xor i32 %9, %.lobit, !dbg !12
  %13 = sdiv i32 %12, 32, !dbg !12
  %.lobit1 = ashr i32 %9, 31, !dbg !12
  %14 = xor i32 %13, %.lobit1, !dbg !12
  %15 = sext i32 %14 to i64, !dbg !12
  %16 = add nsw i64 %8, %15, !dbg !12
  %17 = and i64 %16, 31, !dbg !12
  %18 = shl nsw i64 %10, 7, !dbg !12
  %19 = mul nsw i64 %15, -4096, !dbg !12
  %20 = add nsw i64 %18, %19, !dbg !12
  %21 = shl nsw i64 %20, 7, !dbg !12
  %22 = add nsw i64 %8, %15, !dbg !12
  %23 = and i64 %22, 31, !dbg !12
  %24 = shl nuw nsw i64 %23, 19, !dbg !12
  %25 = add nsw i64 %24, %21, !dbg !12
  %26 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !40
  %27 = zext nneg i32 %26 to i64, !dbg !13
  %28 = lshr i64 %27, 6, !dbg !13
  %29 = lshr i64 %27, 2, !dbg !13
  %30 = and i64 %29, 12, !dbg !13
  %31 = and i32 %26, 16, !dbg !14
  %.not = icmp eq i32 %31, 0, !dbg !14
  %32 = and i32 %26, 32, !dbg !14
  %.not2 = icmp eq i32 %32, 0, !dbg !14
  %33 = or disjoint i64 %11, 127, !dbg !15
  %34 = add nsw i64 %11, 128, !dbg !15
  %35 = icmp uge i64 %34, %33, !dbg !15
  %36 = shl nsw i64 %10, 2, !dbg !7
  %37 = add nsw i64 %36, 4, !dbg !7
  %.idx = shl nuw nsw i64 %17, 20, !dbg !12
  %38 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %39 = getelementptr [2 x i8], ptr addrspace(1) %38, i64 %21, !dbg !12
  %40 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %41 = getelementptr i8, ptr addrspace(1) %40, i64 64, !dbg !12
  %42 = shl nsw i64 %8, 14, !dbg !17
  %43 = shl nsw i64 %6, 19, !dbg !17
  %44 = shl nsw i64 %8, 7, !dbg !17
  %45 = shl nsw i64 %6, 12, !dbg !17
  %46 = add nsw i64 %44, %45, !dbg !17
  %47 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %48 = getelementptr i8, ptr addrspace(1) %47, i64 128, !dbg !12
  %49 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %50 = getelementptr i8, ptr addrspace(1) %49, i64 192, !dbg !12
  br label %51, !dbg !7

51:                                               ; preds = %1033, %4
  %52 = phi i64 [ %1104, %1033 ], [ 0, %4 ], !dbg !7
  %53 = phi [2 x <8 x float>] [ %1085, %1033 ], [ zeroinitializer, %4 ], !dbg !7
  %54 = phi [2 x <1 x float>] [ %1089, %1033 ], [ zeroinitializer, %4 ], !dbg !7
  %55 = phi [2 x <1 x float>] [ %1091, %1033 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %56 = phi [2 x <8 x float>] [ %1095, %1033 ], [ zeroinitializer, %4 ], !dbg !7
  %57 = phi [2 x <8 x float>] [ %1099, %1033 ], [ zeroinitializer, %4 ], !dbg !7
  %58 = phi [2 x <8 x float>] [ %1103, %1033 ], [ zeroinitializer, %4 ], !dbg !7
  %59 = icmp slt i64 %52, %37, !dbg !7
  br i1 %59, label %60, label %1105, !dbg !7

60:                                               ; preds = %51
  %61 = shl nsw i64 %52, 5, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !12, !range !40, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local60 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local61 = and i64 %address.local60, 1536, !dbg !12
  %address.local62 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local61, !dbg !12
  %address.local63 = ptrtoint ptr addrspace(3) %address.local62 to i64, !dbg !12
  %address.local64 = trunc i64 %address.local63 to i32, !dbg !12
  %62 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local64), !dbg !12
  %63 = zext i32 %62 to i64, !dbg !12
  %64 = inttoptr i64 %63 to ptr addrspace(3), !dbg !12
  %address.local65 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local66 = and i64 %address.local65, 16256, !dbg !12
  %address.local67 = lshr i64 %address.local, 4, !dbg !12
  %address.local68 = and i64 %address.local67, 8, !dbg !12
  %address.local69 = or disjoint i64 %address.local66, %address.local68, !dbg !12
  %address.local70 = getelementptr [2 x i8], ptr addrspace(1) %39, i64 %address.local69, !dbg !12
  %65 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %66 = ptrtoint ptr addrspace(1) %address.local70 to i64, !dbg !12
  %67 = sub i64 %66, %65, !dbg !12
  %dma.buffer.offset861 = trunc i64 %67 to i32, !dbg !12
  %68 = trunc i64 %65 to i32, !dbg !12
  %69 = insertelement <4 x i32> poison, i32 %68, i64 0, !dbg !12
  %70 = lshr i64 %65, 32, !dbg !12
  %71 = trunc i64 %70 to i32, !dbg !12
  %72 = insertelement <4 x i32> %69, i32 %71, i64 1, !dbg !12
  %73 = insertelement <4 x i32> %72, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc862 = insertelement <4 x i32> %73, i32 131072, i64 3, !dbg !12
  %74 = ptrtoint ptr addrspace(3) %64 to i32, !dbg !12
  %dma.buffer.lds863 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %74), !dbg !12
  %75 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds863, i32 %dma.buffer.offset861, <4 x i32> %dma.buffer.rsrc862) #10, !dbg !12
  %address.local71 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local72 = shl nuw nsw i64 %address.local71, 3, !dbg !12
  %address.local73 = and i64 %address.local72, 3584, !dbg !12
  %address.local74 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local73, !dbg !12
  %address.local75 = ptrtoint ptr addrspace(3) %address.local74 to i64, !dbg !12
  %address.local76 = trunc i64 %address.local75 to i32, !dbg !12
  %76 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local76), !dbg !12
  %77 = zext i32 %76 to i64, !dbg !12
  %78 = inttoptr i64 %77 to ptr addrspace(3), !dbg !12
  %address.local77 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local78 = and i64 %address.local77, 16256, !dbg !12
  %address.local79 = lshr i64 %address.local71, 4, !dbg !12
  %address.local80 = and i64 %address.local79, 24, !dbg !12
  %address.local81 = or disjoint i64 %address.local78, %address.local80, !dbg !12
  %address.local82 = getelementptr [2 x i8], ptr addrspace(1) %39, i64 %address.local81, !dbg !12
  %79 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %80 = ptrtoint ptr addrspace(1) %address.local82 to i64, !dbg !12
  %81 = sub i64 %80, %79, !dbg !12
  %dma.buffer.offset858 = trunc i64 %81 to i32, !dbg !12
  %82 = trunc i64 %79 to i32, !dbg !12
  %83 = insertelement <4 x i32> poison, i32 %82, i64 0, !dbg !12
  %84 = lshr i64 %79, 32, !dbg !12
  %85 = trunc i64 %84 to i32, !dbg !12
  %86 = insertelement <4 x i32> %83, i32 %85, i64 1, !dbg !12
  %87 = insertelement <4 x i32> %86, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc859 = insertelement <4 x i32> %87, i32 131072, i64 3, !dbg !12
  %88 = ptrtoint ptr addrspace(3) %78 to i32, !dbg !12
  %dma.buffer.lds860 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %88), !dbg !12
  %89 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds860, i32 %dma.buffer.offset858, <4 x i32> %dma.buffer.rsrc859) #10, !dbg !12
  %90 = lshr i64 %52, 14, !dbg !17
  %91 = add nsw i64 %90, %8, !dbg !17
  %92 = lshr i64 %52, 7, !dbg !17
  %93 = and i64 %92, 127, !dbg !17
  %94 = shl i64 %91, 19, !dbg !17
  %95 = and i64 %94, 16252928, !dbg !17
  %96 = shl i64 %92, 19, !dbg !17
  %97 = shl i64 %52, 12, !dbg !17
  %98 = sub i64 %97, %96, !dbg !17
  %99 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %95, !dbg !17
  %100 = getelementptr [2 x i8], ptr addrspace(1) %99, i64 %93, !dbg !17
  %101 = getelementptr [2 x i8], ptr addrspace(1) %100, i64 %98, !dbg !17
  br label %102, !dbg !17

102:                                              ; preds = %105, %60
  %103 = phi i64 [ %131, %105 ], [ %27, %60 ], !dbg !17
  %104 = icmp samesign ult i64 %103, 128, !dbg !17
  br i1 %104, label %105, label %132, !dbg !17

105:                                              ; preds = %102
  %106 = lshr i64 %103, 2, !dbg !17
  %107 = and i64 %106, 24, !dbg !17
  %108 = shl nuw nsw i64 %103, 3, !dbg !17
  %109 = and i64 %108, 512, !dbg !17
  %110 = shl nuw nsw i64 %103, 7, !dbg !17
  %111 = and i64 %110, 3968, !dbg !17
  %112 = getelementptr [2 x i8], ptr addrspace(1) %101, i64 %107, !dbg !17
  %113 = getelementptr [2 x i8], ptr addrspace(1) %112, i64 %111, !dbg !17
  %114 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %109, !dbg !17
  %115 = ptrtoint ptr addrspace(3) %114 to i64, !dbg !17
  %116 = trunc i64 %115 to i32, !dbg !17
  %117 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %116), !dbg !17
  %118 = zext i32 %117 to i64, !dbg !17
  %119 = inttoptr i64 %118 to ptr addrspace(3), !dbg !17
  %120 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %121 = ptrtoint ptr addrspace(1) %113 to i64, !dbg !17
  %122 = sub i64 %121, %120, !dbg !17
  %dma.buffer.offset = trunc i64 %122 to i32, !dbg !17
  %123 = trunc i64 %120 to i32, !dbg !17
  %124 = insertelement <4 x i32> poison, i32 %123, i64 0, !dbg !17
  %125 = lshr i64 %120, 32, !dbg !17
  %126 = trunc i64 %125 to i32, !dbg !17
  %127 = insertelement <4 x i32> %124, i32 %126, i64 1, !dbg !17
  %128 = insertelement <4 x i32> %127, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %128, i32 131072, i64 3, !dbg !17
  %129 = ptrtoint ptr addrspace(3) %119 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %129), !dbg !17
  %130 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %131 = or disjoint i64 %27, 256, !dbg !17
  br label %102, !dbg !17

132:                                              ; preds = %102
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %address.thread83 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !12, !range !40, !frisk.register_lifetimes !1
  %address.local84 = zext nneg i32 %address.thread83 to i64, !dbg !13
  %address.local85 = shl nuw nsw i64 %address.local84, 3, !dbg !12
  %address.local86 = and i64 %address.local85, 1536, !dbg !12
  %address.local87 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local86, !dbg !12
  %address.local88 = ptrtoint ptr addrspace(3) %address.local87 to i64, !dbg !12
  %address.local89 = trunc i64 %address.local88 to i32, !dbg !12
  %133 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local89), !dbg !12
  %134 = zext i32 %133 to i64, !dbg !12
  %135 = inttoptr i64 %134 to ptr addrspace(3), !dbg !12
  %address.local90 = shl nuw nsw i64 %address.local84, 7, !dbg !12
  %address.local91 = and i64 %address.local90, 16256, !dbg !12
  %address.local92 = lshr i64 %address.local84, 4, !dbg !12
  %address.local93 = and i64 %address.local92, 8, !dbg !12
  %address.local94 = or disjoint i64 %address.local91, %address.local93, !dbg !12
  %address.local95 = getelementptr [2 x i8], ptr addrspace(1) %41, i64 %address.local94, !dbg !12
  %136 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %137 = ptrtoint ptr addrspace(1) %address.local95 to i64, !dbg !12
  %138 = sub i64 %137, %136, !dbg !12
  %dma.buffer.offset855 = trunc i64 %138 to i32, !dbg !12
  %139 = trunc i64 %136 to i32, !dbg !12
  %140 = insertelement <4 x i32> poison, i32 %139, i64 0, !dbg !12
  %141 = lshr i64 %136, 32, !dbg !12
  %142 = trunc i64 %141 to i32, !dbg !12
  %143 = insertelement <4 x i32> %140, i32 %142, i64 1, !dbg !12
  %144 = insertelement <4 x i32> %143, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc856 = insertelement <4 x i32> %144, i32 131072, i64 3, !dbg !12
  %145 = ptrtoint ptr addrspace(3) %135 to i32, !dbg !12
  %dma.buffer.lds857 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %145), !dbg !12
  %146 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds857, i32 %dma.buffer.offset855, <4 x i32> %dma.buffer.rsrc856) #10, !dbg !12
  %address.local96 = or disjoint i64 %address.local84, 256, !dbg !12
  %address.local97 = shl nuw nsw i64 %address.local96, 3, !dbg !12
  %address.local98 = and i64 %address.local97, 3584, !dbg !12
  %address.local99 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local98, !dbg !12
  %address.local100 = ptrtoint ptr addrspace(3) %address.local99 to i64, !dbg !12
  %address.local101 = trunc i64 %address.local100 to i32, !dbg !12
  %147 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local101), !dbg !12
  %148 = zext i32 %147 to i64, !dbg !12
  %149 = inttoptr i64 %148 to ptr addrspace(3), !dbg !12
  %address.local102 = shl nuw nsw i64 %address.local84, 7, !dbg !12
  %address.local103 = and i64 %address.local102, 16256, !dbg !12
  %address.local104 = lshr i64 %address.local96, 4, !dbg !12
  %address.local105 = and i64 %address.local104, 24, !dbg !12
  %address.local106 = or disjoint i64 %address.local103, %address.local105, !dbg !12
  %address.local107 = getelementptr [2 x i8], ptr addrspace(1) %41, i64 %address.local106, !dbg !12
  %150 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %151 = ptrtoint ptr addrspace(1) %address.local107 to i64, !dbg !12
  %152 = sub i64 %151, %150, !dbg !12
  %dma.buffer.offset852 = trunc i64 %152 to i32, !dbg !12
  %153 = trunc i64 %150 to i32, !dbg !12
  %154 = insertelement <4 x i32> poison, i32 %153, i64 0, !dbg !12
  %155 = lshr i64 %150, 32, !dbg !12
  %156 = trunc i64 %155 to i32, !dbg !12
  %157 = insertelement <4 x i32> %154, i32 %156, i64 1, !dbg !12
  %158 = insertelement <4 x i32> %157, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc853 = insertelement <4 x i32> %158, i32 131072, i64 3, !dbg !12
  %159 = ptrtoint ptr addrspace(3) %149 to i32, !dbg !12
  %dma.buffer.lds854 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %159), !dbg !12
  %160 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds854, i32 %dma.buffer.offset852, <4 x i32> %dma.buffer.rsrc853) #10, !dbg !12
  %161 = add i64 %52, %42, !dbg !17
  %162 = add i64 %161, %43, !dbg !17
  %163 = add i64 %162, 4096, !dbg !17
  %164 = icmp slt i64 %163, 0, !dbg !17
  %165 = sub i64 -4097, %162, !dbg !17
  %166 = select i1 %164, i64 %165, i64 %163, !dbg !17
  %167 = sdiv i64 %166, 16384, !dbg !17
  %.lobit3 = ashr i64 %163, 63, !dbg !17
  %168 = xor i64 %167, %.lobit3, !dbg !17
  %169 = add nsw i64 %92, %46, !dbg !17
  %170 = add nsw i64 %169, 32, !dbg !17
  %171 = icmp slt i64 %169, -32, !dbg !17
  %172 = sub nsw i64 -33, %169, !dbg !17
  %173 = select i1 %171, i64 %172, i64 %170, !dbg !17
  %174 = sdiv i64 %173, 128, !dbg !17
  %175 = sext i1 %171 to i64, !dbg !17
  %176 = xor i64 %174, %175, !dbg !17
  %177 = shl nsw i64 %176, 7, !dbg !17
  %178 = sub i64 %46, %177, !dbg !17
  %179 = shl i64 %168, 19, !dbg !17
  %180 = and i64 %179, 16252928, !dbg !17
  %181 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %180, !dbg !17
  %182 = getelementptr [2 x i8], ptr addrspace(1) %181, i64 %92, !dbg !17
  %183 = getelementptr [2 x i8], ptr addrspace(1) %182, i64 %178, !dbg !17
  %184 = getelementptr i8, ptr addrspace(1) %183, i64 64, !dbg !17
  %185 = getelementptr [2 x i8], ptr addrspace(1) %184, i64 %98, !dbg !17
  br label %186, !dbg !17

186:                                              ; preds = %189, %132
  %187 = phi i64 [ %215, %189 ], [ %27, %132 ], !dbg !17
  %188 = icmp samesign ult i64 %187, 128, !dbg !17
  br i1 %188, label %189, label %216, !dbg !17

189:                                              ; preds = %186
  %190 = lshr i64 %187, 2, !dbg !17
  %191 = and i64 %190, 24, !dbg !17
  %192 = shl nuw nsw i64 %187, 3, !dbg !17
  %193 = and i64 %192, 512, !dbg !17
  %194 = shl nuw nsw i64 %187, 7, !dbg !17
  %195 = and i64 %194, 3968, !dbg !17
  %196 = getelementptr [2 x i8], ptr addrspace(1) %185, i64 %191, !dbg !17
  %197 = getelementptr [2 x i8], ptr addrspace(1) %196, i64 %195, !dbg !17
  %198 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 18432), i64 %193, !dbg !17
  %199 = ptrtoint ptr addrspace(3) %198 to i64, !dbg !17
  %200 = trunc i64 %199 to i32, !dbg !17
  %201 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %200), !dbg !17
  %202 = zext i32 %201 to i64, !dbg !17
  %203 = inttoptr i64 %202 to ptr addrspace(3), !dbg !17
  %204 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %205 = ptrtoint ptr addrspace(1) %197 to i64, !dbg !17
  %206 = sub i64 %205, %204, !dbg !17
  %dma.buffer.offset819 = trunc i64 %206 to i32, !dbg !17
  %207 = trunc i64 %204 to i32, !dbg !17
  %208 = insertelement <4 x i32> poison, i32 %207, i64 0, !dbg !17
  %209 = lshr i64 %204, 32, !dbg !17
  %210 = trunc i64 %209 to i32, !dbg !17
  %211 = insertelement <4 x i32> %208, i32 %210, i64 1, !dbg !17
  %212 = insertelement <4 x i32> %211, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc820 = insertelement <4 x i32> %212, i32 131072, i64 3, !dbg !17
  %213 = ptrtoint ptr addrspace(3) %203 to i32, !dbg !17
  %dma.buffer.lds821 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %213), !dbg !17
  %214 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds821, i32 %dma.buffer.offset819, <4 x i32> %dma.buffer.rsrc820) #10, !dbg !17
  %215 = or disjoint i64 %27, 256, !dbg !17
  br label %186, !dbg !17

216:                                              ; preds = %186
  %address.thread108 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !13, !range !40, !frisk.register_lifetimes !1
  %address.local109 = lshr i32 %address.thread108, 2, !dbg !13
  %address.local110 = and i32 %address.local109, 4, !dbg !13
  %address.local111 = shl nuw nsw i32 %address.thread108, 1, !dbg !13
  %address.local112 = and i32 %address.local111, 384, !dbg !13
  %address.local113 = or disjoint i32 %address.local110, %address.local112, !dbg !13
  %address.local114 = shl nuw nsw i32 %address.thread108, 3, !dbg !13
  %address.local115 = and i32 %address.local114, 120, !dbg !13
  %address.local116 = or disjoint i32 %address.local113, %address.local115, !dbg !13
  %address.local117 = shl nuw nsw i32 %address.thread108, 5, !dbg !13
  %address.local118 = and i32 %address.local117, 1024, !dbg !13
  %address.local119 = or disjoint i32 %address.local116, %address.local118, !dbg !13
  %address.local120 = zext nneg i32 %address.local119 to i64, !dbg !13
  %address.local121 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local120, !dbg !13
  %217 = load <4 x half>, ptr addrspace(3) %address.local121, align 2, !dbg !13
  %address.local122 = lshr i32 %address.thread108, 2, !dbg !13
  %address.local123 = and i32 %address.local122, 4, !dbg !13
  %address.local124 = shl nuw nsw i32 %address.thread108, 1, !dbg !13
  %address.local125 = and i32 %address.local124, 384, !dbg !13
  %address.local126 = or disjoint i32 %address.local123, %address.local125, !dbg !13
  %address.local127 = shl nuw nsw i32 %address.thread108, 3, !dbg !13
  %address.local128 = and i32 %address.local127, 120, !dbg !13
  %address.local129 = or disjoint i32 %address.local126, %address.local128, !dbg !13
  %address.local130 = shl nuw nsw i32 %address.thread108, 5, !dbg !13
  %address.local131 = and i32 %address.local130, 1024, !dbg !13
  %address.local132 = or disjoint i32 %address.local129, %address.local131, !dbg !13
  %address.local133 = or disjoint i32 %address.local132, 2048, !dbg !13
  %address.local134 = zext nneg i32 %address.local133 to i64, !dbg !13
  %address.local135 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local134, !dbg !13
  %218 = load <4 x half>, ptr addrspace(3) %address.local135, align 2, !dbg !13
  %address.local136 = lshr i32 %address.thread108, 2, !dbg !13
  %address.local137 = and i32 %address.local136, 4, !dbg !13
  %address.local138 = shl nuw nsw i32 %address.thread108, 1, !dbg !13
  %address.local139 = and i32 %address.local138, 384, !dbg !13
  %address.local140 = or disjoint i32 %address.local137, %address.local139, !dbg !13
  %address.local141 = shl nuw nsw i32 %address.thread108, 3, !dbg !13
  %address.local142 = and i32 %address.local141, 120, !dbg !13
  %address.local143 = or disjoint i32 %address.local140, %address.local142, !dbg !13
  %address.local144 = shl nuw nsw i32 %address.thread108, 5, !dbg !13
  %address.local145 = and i32 %address.local144, 1024, !dbg !13
  %address.local146 = or disjoint i32 %address.local143, %address.local145, !dbg !13
  %address.local147 = or disjoint i32 %address.local146, 512, !dbg !13
  %address.local148 = zext nneg i32 %address.local147 to i64, !dbg !13
  %address.local149 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local148, !dbg !13
  %219 = load <4 x half>, ptr addrspace(3) %address.local149, align 2, !dbg !13
  %220 = fmul <4 x half> %217, splat (half 0xH3015), !dbg !13
  %address.local150 = lshr i32 %address.thread108, 2, !dbg !13
  %address.local151 = and i32 %address.local150, 4, !dbg !13
  %address.local152 = shl nuw nsw i32 %address.thread108, 1, !dbg !13
  %address.local153 = and i32 %address.local152, 384, !dbg !13
  %address.local154 = or disjoint i32 %address.local151, %address.local153, !dbg !13
  %address.local155 = shl nuw nsw i32 %address.thread108, 3, !dbg !13
  %address.local156 = and i32 %address.local155, 120, !dbg !13
  %address.local157 = or disjoint i32 %address.local154, %address.local156, !dbg !13
  %address.local158 = shl nuw nsw i32 %address.thread108, 5, !dbg !13
  %address.local159 = and i32 %address.local158, 1024, !dbg !13
  %address.local160 = or disjoint i32 %address.local157, %address.local159, !dbg !13
  %address.local161 = or disjoint i32 %address.local160, 2560, !dbg !13
  %address.local162 = zext nneg i32 %address.local161 to i64, !dbg !13
  %address.local163 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local162, !dbg !13
  %221 = load <4 x half>, ptr addrspace(3) %address.local163, align 2, !dbg !13
  %222 = fmul <4 x half> %218, splat (half 0xH3015), !dbg !13
  %address.local164 = lshr i32 %address.thread108, 2, !dbg !18
  %address.local165 = and i32 %address.local164, 4, !dbg !18
  %address.local166 = shl nuw nsw i32 %address.thread108, 3, !dbg !18
  %address.local167 = and i32 %address.local166, 376, !dbg !18
  %address.local168 = or disjoint i32 %address.local165, %address.local167, !dbg !18
  %address.local169 = zext nneg i32 %address.local168 to i64, !dbg !18
  %address.local170 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local169, !dbg !18
  %223 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !18
  %224 = fmul <4 x half> %219, splat (half 0xH3015), !dbg !13
  %address.local171 = lshr i32 %address.thread108, 2, !dbg !18
  %address.local172 = and i32 %address.local171, 4, !dbg !18
  %address.local173 = shl nuw nsw i32 %address.thread108, 3, !dbg !18
  %address.local174 = and i32 %address.local173, 376, !dbg !18
  %address.local175 = or disjoint i32 %address.local172, %address.local174, !dbg !18
  %address.local176 = or disjoint i32 %address.local175, 512, !dbg !18
  %address.local177 = zext nneg i32 %address.local176 to i64, !dbg !18
  %address.local178 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local177, !dbg !18
  %225 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !18
  %226 = fmul <4 x half> %221, splat (half 0xH3015), !dbg !13
  %address.local179 = lshr i32 %address.thread108, 2, !dbg !18
  %address.local180 = and i32 %address.local179, 4, !dbg !18
  %address.local181 = shl nuw nsw i32 %address.thread108, 3, !dbg !18
  %address.local182 = and i32 %address.local181, 376, !dbg !18
  %address.local183 = or disjoint i32 %address.local180, %address.local182, !dbg !18
  %address.local184 = or disjoint i32 %address.local183, 128, !dbg !18
  %address.local185 = zext nneg i32 %address.local184 to i64, !dbg !18
  %address.local186 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local185, !dbg !18
  %227 = load <4 x half>, ptr addrspace(3) %address.local186, align 2, !dbg !18
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %220, <4 x half> %223, <4 x float> zeroinitializer) #9, !dbg !18
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %222, <4 x half> %225, <4 x float> %228) #9, !dbg !18
  %address.local187 = lshr i32 %address.thread108, 2, !dbg !18
  %address.local188 = and i32 %address.local187, 4, !dbg !18
  %address.local189 = shl nuw nsw i32 %address.thread108, 3, !dbg !18
  %address.local190 = and i32 %address.local189, 376, !dbg !18
  %address.local191 = or disjoint i32 %address.local188, %address.local190, !dbg !18
  %address.local192 = or disjoint i32 %address.local191, 640, !dbg !18
  %address.local193 = zext nneg i32 %address.local192 to i64, !dbg !18
  %address.local194 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local193, !dbg !18
  %230 = load <4 x half>, ptr addrspace(3) %address.local194, align 2, !dbg !18
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %220, <4 x half> %227, <4 x float> zeroinitializer) #9, !dbg !18
  %232 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %222, <4 x half> %230, <4 x float> %231) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %232, <4 x float> zeroinitializer) #9, !dbg !18
  %235 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !18
  %236 = load <4 x half>, ptr addrspace(3) %address.local186, align 2, !dbg !18
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %235, <4 x float> %234) #9, !dbg !18
  %238 = load <4 x half>, ptr addrspace(3) %address.local194, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %236, <4 x float> zeroinitializer) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %226, <4 x half> %238, <4 x float> %239) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %address.local195 = zext nneg i32 %address.thread108 to i64, !dbg !13
  %address.local196 = shl nuw nsw i64 %address.local195, 3, !dbg !12
  %address.local197 = and i64 %address.local196, 1536, !dbg !12
  %address.local198 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local197, !dbg !12
  %address.local199 = ptrtoint ptr addrspace(3) %address.local198 to i64, !dbg !12
  %address.local200 = trunc i64 %address.local199 to i32, !dbg !12
  %241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local200), !dbg !12
  %242 = zext i32 %241 to i64, !dbg !12
  %243 = inttoptr i64 %242 to ptr addrspace(3), !dbg !12
  %address.local201 = shl nuw nsw i64 %address.local195, 7, !dbg !12
  %address.local202 = and i64 %address.local201, 16256, !dbg !12
  %address.local203 = lshr i64 %address.local195, 4, !dbg !12
  %address.local204 = and i64 %address.local203, 8, !dbg !12
  %address.local205 = or disjoint i64 %address.local202, %address.local204, !dbg !12
  %address.local206 = getelementptr [2 x i8], ptr addrspace(1) %48, i64 %address.local205, !dbg !12
  %244 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %245 = ptrtoint ptr addrspace(1) %address.local206 to i64, !dbg !12
  %246 = sub i64 %245, %244, !dbg !12
  %dma.buffer.offset849 = trunc i64 %246 to i32, !dbg !12
  %247 = trunc i64 %244 to i32, !dbg !12
  %248 = insertelement <4 x i32> poison, i32 %247, i64 0, !dbg !12
  %249 = lshr i64 %244, 32, !dbg !12
  %250 = trunc i64 %249 to i32, !dbg !12
  %251 = insertelement <4 x i32> %248, i32 %250, i64 1, !dbg !12
  %252 = insertelement <4 x i32> %251, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc850 = insertelement <4 x i32> %252, i32 131072, i64 3, !dbg !12
  %253 = ptrtoint ptr addrspace(3) %243 to i32, !dbg !12
  %dma.buffer.lds851 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %253), !dbg !12
  %254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds851, i32 %dma.buffer.offset849, <4 x i32> %dma.buffer.rsrc850) #10, !dbg !12
  %address.local207 = or disjoint i64 %address.local195, 256, !dbg !12
  %address.local208 = shl nuw nsw i64 %address.local207, 3, !dbg !12
  %address.local209 = and i64 %address.local208, 3584, !dbg !12
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local209, !dbg !12
  %address.local211 = ptrtoint ptr addrspace(3) %address.local210 to i64, !dbg !12
  %address.local212 = trunc i64 %address.local211 to i32, !dbg !12
  %255 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local212), !dbg !12
  %256 = zext i32 %255 to i64, !dbg !12
  %257 = inttoptr i64 %256 to ptr addrspace(3), !dbg !12
  %address.local213 = shl nuw nsw i64 %address.local195, 7, !dbg !12
  %address.local214 = and i64 %address.local213, 16256, !dbg !12
  %address.local215 = lshr i64 %address.local207, 4, !dbg !12
  %address.local216 = and i64 %address.local215, 24, !dbg !12
  %address.local217 = or disjoint i64 %address.local214, %address.local216, !dbg !12
  %address.local218 = getelementptr [2 x i8], ptr addrspace(1) %48, i64 %address.local217, !dbg !12
  %258 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %259 = ptrtoint ptr addrspace(1) %address.local218 to i64, !dbg !12
  %260 = sub i64 %259, %258, !dbg !12
  %dma.buffer.offset846 = trunc i64 %260 to i32, !dbg !12
  %261 = trunc i64 %258 to i32, !dbg !12
  %262 = insertelement <4 x i32> poison, i32 %261, i64 0, !dbg !12
  %263 = lshr i64 %258, 32, !dbg !12
  %264 = trunc i64 %263 to i32, !dbg !12
  %265 = insertelement <4 x i32> %262, i32 %264, i64 1, !dbg !12
  %266 = insertelement <4 x i32> %265, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc847 = insertelement <4 x i32> %266, i32 131072, i64 3, !dbg !12
  %267 = ptrtoint ptr addrspace(3) %257 to i32, !dbg !12
  %dma.buffer.lds848 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %267), !dbg !12
  %268 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds848, i32 %dma.buffer.offset846, <4 x i32> %dma.buffer.rsrc847) #10, !dbg !12
  %269 = add i64 %162, 8192, !dbg !17
  %270 = icmp slt i64 %269, 0, !dbg !17
  %271 = sub i64 -8193, %162, !dbg !17
  %272 = select i1 %270, i64 %271, i64 %269, !dbg !17
  %273 = sdiv i64 %272, 16384, !dbg !17
  %.lobit4 = ashr i64 %269, 63, !dbg !17
  %274 = xor i64 %273, %.lobit4, !dbg !17
  %275 = add nsw i64 %169, 64, !dbg !17
  %276 = icmp slt i64 %169, -64, !dbg !17
  %277 = sub nsw i64 -65, %169, !dbg !17
  %278 = select i1 %276, i64 %277, i64 %275, !dbg !17
  %279 = sdiv i64 %278, 128, !dbg !17
  %280 = sext i1 %276 to i64, !dbg !17
  %281 = xor i64 %279, %280, !dbg !17
  %282 = shl nsw i64 %281, 7, !dbg !17
  %283 = sub i64 %46, %282, !dbg !17
  %284 = shl i64 %274, 19, !dbg !17
  %285 = and i64 %284, 16252928, !dbg !17
  %286 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %285, !dbg !17
  %287 = getelementptr [2 x i8], ptr addrspace(1) %286, i64 %92, !dbg !17
  %288 = getelementptr [2 x i8], ptr addrspace(1) %287, i64 %283, !dbg !17
  %289 = getelementptr i8, ptr addrspace(1) %288, i64 128, !dbg !17
  %290 = getelementptr [2 x i8], ptr addrspace(1) %289, i64 %98, !dbg !17
  br label %291, !dbg !17

291:                                              ; preds = %294, %216
  %292 = phi i64 [ %320, %294 ], [ %27, %216 ], !dbg !17
  %293 = icmp samesign ult i64 %292, 128, !dbg !17
  br i1 %293, label %294, label %321, !dbg !17

294:                                              ; preds = %291
  %295 = lshr i64 %292, 2, !dbg !17
  %296 = and i64 %295, 24, !dbg !17
  %297 = shl nuw nsw i64 %292, 3, !dbg !17
  %298 = and i64 %297, 512, !dbg !17
  %299 = shl nuw nsw i64 %292, 7, !dbg !17
  %300 = and i64 %299, 3968, !dbg !17
  %301 = getelementptr [2 x i8], ptr addrspace(1) %290, i64 %296, !dbg !17
  %302 = getelementptr [2 x i8], ptr addrspace(1) %301, i64 %300, !dbg !17
  %303 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %298, !dbg !17
  %304 = ptrtoint ptr addrspace(3) %303 to i64, !dbg !17
  %305 = trunc i64 %304 to i32, !dbg !17
  %306 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %305), !dbg !17
  %307 = zext i32 %306 to i64, !dbg !17
  %308 = inttoptr i64 %307 to ptr addrspace(3), !dbg !17
  %309 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %310 = ptrtoint ptr addrspace(1) %302 to i64, !dbg !17
  %311 = sub i64 %310, %309, !dbg !17
  %dma.buffer.offset822 = trunc i64 %311 to i32, !dbg !17
  %312 = trunc i64 %309 to i32, !dbg !17
  %313 = insertelement <4 x i32> poison, i32 %312, i64 0, !dbg !17
  %314 = lshr i64 %309, 32, !dbg !17
  %315 = trunc i64 %314 to i32, !dbg !17
  %316 = insertelement <4 x i32> %313, i32 %315, i64 1, !dbg !17
  %317 = insertelement <4 x i32> %316, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc823 = insertelement <4 x i32> %317, i32 131072, i64 3, !dbg !17
  %318 = ptrtoint ptr addrspace(3) %308 to i32, !dbg !17
  %dma.buffer.lds824 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %318), !dbg !17
  %319 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds824, i32 %dma.buffer.offset822, <4 x i32> %dma.buffer.rsrc823) #10, !dbg !17
  %320 = or disjoint i64 %27, 256, !dbg !17
  br label %291, !dbg !17

321:                                              ; preds = %291
  %address.thread219 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !13, !range !40, !frisk.register_lifetimes !1
  %address.local220 = lshr i32 %address.thread219, 2, !dbg !13
  %address.local221 = and i32 %address.local220, 4, !dbg !13
  %address.local222 = shl nuw nsw i32 %address.thread219, 1, !dbg !13
  %address.local223 = and i32 %address.local222, 384, !dbg !13
  %address.local224 = or disjoint i32 %address.local221, %address.local223, !dbg !13
  %address.local225 = shl nuw nsw i32 %address.thread219, 3, !dbg !13
  %address.local226 = and i32 %address.local225, 120, !dbg !13
  %address.local227 = or disjoint i32 %address.local224, %address.local226, !dbg !13
  %address.local228 = shl nuw nsw i32 %address.thread219, 5, !dbg !13
  %address.local229 = and i32 %address.local228, 1024, !dbg !13
  %address.local230 = or disjoint i32 %address.local227, %address.local229, !dbg !13
  %address.local231 = zext nneg i32 %address.local230 to i64, !dbg !13
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local231, !dbg !13
  %322 = load <4 x half>, ptr addrspace(3) %address.local232, align 2, !dbg !13
  %address.local233 = lshr i32 %address.thread219, 2, !dbg !13
  %address.local234 = and i32 %address.local233, 4, !dbg !13
  %address.local235 = shl nuw nsw i32 %address.thread219, 1, !dbg !13
  %address.local236 = and i32 %address.local235, 384, !dbg !13
  %address.local237 = or disjoint i32 %address.local234, %address.local236, !dbg !13
  %address.local238 = shl nuw nsw i32 %address.thread219, 3, !dbg !13
  %address.local239 = and i32 %address.local238, 120, !dbg !13
  %address.local240 = or disjoint i32 %address.local237, %address.local239, !dbg !13
  %address.local241 = shl nuw nsw i32 %address.thread219, 5, !dbg !13
  %address.local242 = and i32 %address.local241, 1024, !dbg !13
  %address.local243 = or disjoint i32 %address.local240, %address.local242, !dbg !13
  %address.local244 = or disjoint i32 %address.local243, 2048, !dbg !13
  %address.local245 = zext nneg i32 %address.local244 to i64, !dbg !13
  %address.local246 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local245, !dbg !13
  %323 = load <4 x half>, ptr addrspace(3) %address.local246, align 2, !dbg !13
  %324 = fmul <4 x half> %322, splat (half 0xH3015), !dbg !13
  %325 = fmul <4 x half> %323, splat (half 0xH3015), !dbg !13
  %address.local247 = lshr i32 %address.thread219, 2, !dbg !13
  %address.local248 = and i32 %address.local247, 4, !dbg !13
  %address.local249 = shl nuw nsw i32 %address.thread219, 1, !dbg !13
  %address.local250 = and i32 %address.local249, 384, !dbg !13
  %address.local251 = or disjoint i32 %address.local248, %address.local250, !dbg !13
  %address.local252 = shl nuw nsw i32 %address.thread219, 3, !dbg !13
  %address.local253 = and i32 %address.local252, 120, !dbg !13
  %address.local254 = or disjoint i32 %address.local251, %address.local253, !dbg !13
  %address.local255 = shl nuw nsw i32 %address.thread219, 5, !dbg !13
  %address.local256 = and i32 %address.local255, 1024, !dbg !13
  %address.local257 = or disjoint i32 %address.local254, %address.local256, !dbg !13
  %address.local258 = or disjoint i32 %address.local257, 512, !dbg !13
  %address.local259 = zext nneg i32 %address.local258 to i64, !dbg !13
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local259, !dbg !13
  %326 = load <4 x half>, ptr addrspace(3) %address.local260, align 2, !dbg !13
  %address.local261 = lshr i32 %address.thread219, 2, !dbg !13
  %address.local262 = and i32 %address.local261, 4, !dbg !13
  %address.local263 = shl nuw nsw i32 %address.thread219, 1, !dbg !13
  %address.local264 = and i32 %address.local263, 384, !dbg !13
  %address.local265 = or disjoint i32 %address.local262, %address.local264, !dbg !13
  %address.local266 = shl nuw nsw i32 %address.thread219, 3, !dbg !13
  %address.local267 = and i32 %address.local266, 120, !dbg !13
  %address.local268 = or disjoint i32 %address.local265, %address.local267, !dbg !13
  %address.local269 = shl nuw nsw i32 %address.thread219, 5, !dbg !13
  %address.local270 = and i32 %address.local269, 1024, !dbg !13
  %address.local271 = or disjoint i32 %address.local268, %address.local270, !dbg !13
  %address.local272 = or disjoint i32 %address.local271, 2560, !dbg !13
  %address.local273 = zext nneg i32 %address.local272 to i64, !dbg !13
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local273, !dbg !13
  %327 = load <4 x half>, ptr addrspace(3) %address.local274, align 2, !dbg !13
  %328 = fmul <4 x half> %326, splat (half 0xH3015), !dbg !13
  %329 = fmul <4 x half> %327, splat (half 0xH3015), !dbg !13
  %address.local275 = lshr i32 %address.thread219, 2, !dbg !18
  %address.local276 = and i32 %address.local275, 4, !dbg !18
  %address.local277 = shl nuw nsw i32 %address.thread219, 3, !dbg !18
  %address.local278 = and i32 %address.local277, 376, !dbg !18
  %address.local279 = or disjoint i32 %address.local276, %address.local278, !dbg !18
  %address.local280 = zext nneg i32 %address.local279 to i64, !dbg !18
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 18432), i64 %address.local280, !dbg !18
  %330 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !18
  %address.local282 = lshr i32 %address.thread219, 2, !dbg !18
  %address.local283 = and i32 %address.local282, 4, !dbg !18
  %address.local284 = shl nuw nsw i32 %address.thread219, 3, !dbg !18
  %address.local285 = and i32 %address.local284, 376, !dbg !18
  %address.local286 = or disjoint i32 %address.local283, %address.local285, !dbg !18
  %address.local287 = or disjoint i32 %address.local286, 512, !dbg !18
  %address.local288 = zext nneg i32 %address.local287 to i64, !dbg !18
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 18432), i64 %address.local288, !dbg !18
  %331 = load <4 x half>, ptr addrspace(3) %address.local289, align 2, !dbg !18
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %324, <4 x half> %330, <4 x float> %229) #9, !dbg !18
  %address.local290 = lshr i32 %address.thread219, 2, !dbg !18
  %address.local291 = and i32 %address.local290, 4, !dbg !18
  %address.local292 = shl nuw nsw i32 %address.thread219, 3, !dbg !18
  %address.local293 = and i32 %address.local292, 376, !dbg !18
  %address.local294 = or disjoint i32 %address.local291, %address.local293, !dbg !18
  %address.local295 = or disjoint i32 %address.local294, 128, !dbg !18
  %address.local296 = zext nneg i32 %address.local295 to i64, !dbg !18
  %address.local297 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 18432), i64 %address.local296, !dbg !18
  %333 = load <4 x half>, ptr addrspace(3) %address.local297, align 2, !dbg !18
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %331, <4 x float> %332) #9, !dbg !18
  %address.local298 = lshr i32 %address.thread219, 2, !dbg !18
  %address.local299 = and i32 %address.local298, 4, !dbg !18
  %address.local300 = shl nuw nsw i32 %address.thread219, 3, !dbg !18
  %address.local301 = and i32 %address.local300, 376, !dbg !18
  %address.local302 = or disjoint i32 %address.local299, %address.local301, !dbg !18
  %address.local303 = or disjoint i32 %address.local302, 640, !dbg !18
  %address.local304 = zext nneg i32 %address.local303 to i64, !dbg !18
  %address.local305 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 18432), i64 %address.local304, !dbg !18
  %335 = load <4 x half>, ptr addrspace(3) %address.local305, align 2, !dbg !18
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %324, <4 x half> %333, <4 x float> %233) #9, !dbg !18
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %335, <4 x float> %336) #9, !dbg !18
  %338 = load <4 x half>, ptr addrspace(3) %address.local281, align 2, !dbg !18
  %339 = load <4 x half>, ptr addrspace(3) %address.local289, align 2, !dbg !18
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %338, <4 x float> %237) #9, !dbg !18
  %341 = load <4 x half>, ptr addrspace(3) %address.local297, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %339, <4 x float> %340) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %343 = load <4 x half>, ptr addrspace(3) %address.local305, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %341, <4 x float> %240) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %343, <4 x float> %344) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %address.local306 = zext nneg i32 %address.thread219 to i64, !dbg !13
  %address.local307 = shl nuw nsw i64 %address.local306, 3, !dbg !12
  %address.local308 = and i64 %address.local307, 1536, !dbg !12
  %address.local309 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local308, !dbg !12
  %address.local310 = ptrtoint ptr addrspace(3) %address.local309 to i64, !dbg !12
  %address.local311 = trunc i64 %address.local310 to i32, !dbg !12
  %346 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local311), !dbg !12
  %347 = zext i32 %346 to i64, !dbg !12
  %348 = inttoptr i64 %347 to ptr addrspace(3), !dbg !12
  %address.local312 = shl nuw nsw i64 %address.local306, 7, !dbg !12
  %address.local313 = and i64 %address.local312, 16256, !dbg !12
  %address.local314 = lshr i64 %address.local306, 4, !dbg !12
  %address.local315 = and i64 %address.local314, 8, !dbg !12
  %address.local316 = or disjoint i64 %address.local313, %address.local315, !dbg !12
  %address.local317 = getelementptr [2 x i8], ptr addrspace(1) %50, i64 %address.local316, !dbg !12
  %349 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %350 = ptrtoint ptr addrspace(1) %address.local317 to i64, !dbg !12
  %351 = sub i64 %350, %349, !dbg !12
  %dma.buffer.offset843 = trunc i64 %351 to i32, !dbg !12
  %352 = trunc i64 %349 to i32, !dbg !12
  %353 = insertelement <4 x i32> poison, i32 %352, i64 0, !dbg !12
  %354 = lshr i64 %349, 32, !dbg !12
  %355 = trunc i64 %354 to i32, !dbg !12
  %356 = insertelement <4 x i32> %353, i32 %355, i64 1, !dbg !12
  %357 = insertelement <4 x i32> %356, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc844 = insertelement <4 x i32> %357, i32 131072, i64 3, !dbg !12
  %358 = ptrtoint ptr addrspace(3) %348 to i32, !dbg !12
  %dma.buffer.lds845 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %358), !dbg !12
  %359 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds845, i32 %dma.buffer.offset843, <4 x i32> %dma.buffer.rsrc844) #10, !dbg !12
  %address.local318 = or disjoint i64 %address.local306, 256, !dbg !12
  %address.local319 = shl nuw nsw i64 %address.local318, 3, !dbg !12
  %address.local320 = and i64 %address.local319, 3584, !dbg !12
  %address.local321 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local320, !dbg !12
  %address.local322 = ptrtoint ptr addrspace(3) %address.local321 to i64, !dbg !12
  %address.local323 = trunc i64 %address.local322 to i32, !dbg !12
  %360 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local323), !dbg !12
  %361 = zext i32 %360 to i64, !dbg !12
  %362 = inttoptr i64 %361 to ptr addrspace(3), !dbg !12
  %address.local324 = shl nuw nsw i64 %address.local306, 7, !dbg !12
  %address.local325 = and i64 %address.local324, 16256, !dbg !12
  %address.local326 = lshr i64 %address.local318, 4, !dbg !12
  %address.local327 = and i64 %address.local326, 24, !dbg !12
  %address.local328 = or disjoint i64 %address.local325, %address.local327, !dbg !12
  %address.local329 = getelementptr [2 x i8], ptr addrspace(1) %50, i64 %address.local328, !dbg !12
  %363 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %364 = ptrtoint ptr addrspace(1) %address.local329 to i64, !dbg !12
  %365 = sub i64 %364, %363, !dbg !12
  %dma.buffer.offset840 = trunc i64 %365 to i32, !dbg !12
  %366 = trunc i64 %363 to i32, !dbg !12
  %367 = insertelement <4 x i32> poison, i32 %366, i64 0, !dbg !12
  %368 = lshr i64 %363, 32, !dbg !12
  %369 = trunc i64 %368 to i32, !dbg !12
  %370 = insertelement <4 x i32> %367, i32 %369, i64 1, !dbg !12
  %371 = insertelement <4 x i32> %370, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc841 = insertelement <4 x i32> %371, i32 131072, i64 3, !dbg !12
  %372 = ptrtoint ptr addrspace(3) %362 to i32, !dbg !12
  %dma.buffer.lds842 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %372), !dbg !12
  %373 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds842, i32 %dma.buffer.offset840, <4 x i32> %dma.buffer.rsrc841) #10, !dbg !12
  %374 = add i64 %162, 12288, !dbg !17
  %375 = icmp slt i64 %374, 0, !dbg !17
  %376 = sub i64 -12289, %162, !dbg !17
  %377 = select i1 %375, i64 %376, i64 %374, !dbg !17
  %378 = sdiv i64 %377, 16384, !dbg !17
  %.lobit5 = ashr i64 %374, 63, !dbg !17
  %379 = xor i64 %378, %.lobit5, !dbg !17
  %380 = add nsw i64 %169, 96, !dbg !17
  %381 = icmp slt i64 %169, -96, !dbg !17
  %382 = sub nsw i64 -97, %169, !dbg !17
  %383 = select i1 %381, i64 %382, i64 %380, !dbg !17
  %384 = sdiv i64 %383, 128, !dbg !17
  %385 = sext i1 %381 to i64, !dbg !17
  %386 = xor i64 %384, %385, !dbg !17
  %387 = shl nsw i64 %386, 7, !dbg !17
  %388 = sub i64 %46, %387, !dbg !17
  %389 = shl i64 %379, 19, !dbg !17
  %390 = and i64 %389, 16252928, !dbg !17
  %391 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %390, !dbg !17
  %392 = getelementptr [2 x i8], ptr addrspace(1) %391, i64 %92, !dbg !17
  %393 = getelementptr [2 x i8], ptr addrspace(1) %392, i64 %388, !dbg !17
  %394 = getelementptr i8, ptr addrspace(1) %393, i64 192, !dbg !17
  %395 = getelementptr [2 x i8], ptr addrspace(1) %394, i64 %98, !dbg !17
  br label %396, !dbg !17

396:                                              ; preds = %399, %321
  %397 = phi i64 [ %425, %399 ], [ %27, %321 ], !dbg !17
  %398 = icmp samesign ult i64 %397, 128, !dbg !17
  br i1 %398, label %399, label %426, !dbg !17

399:                                              ; preds = %396
  %400 = lshr i64 %397, 2, !dbg !17
  %401 = and i64 %400, 24, !dbg !17
  %402 = shl nuw nsw i64 %397, 3, !dbg !17
  %403 = and i64 %402, 512, !dbg !17
  %404 = shl nuw nsw i64 %397, 7, !dbg !17
  %405 = and i64 %404, 3968, !dbg !17
  %406 = getelementptr [2 x i8], ptr addrspace(1) %395, i64 %401, !dbg !17
  %407 = getelementptr [2 x i8], ptr addrspace(1) %406, i64 %405, !dbg !17
  %408 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 18432), i64 %403, !dbg !17
  %409 = ptrtoint ptr addrspace(3) %408 to i64, !dbg !17
  %410 = trunc i64 %409 to i32, !dbg !17
  %411 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %410), !dbg !17
  %412 = zext i32 %411 to i64, !dbg !17
  %413 = inttoptr i64 %412 to ptr addrspace(3), !dbg !17
  %414 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %415 = ptrtoint ptr addrspace(1) %407 to i64, !dbg !17
  %416 = sub i64 %415, %414, !dbg !17
  %dma.buffer.offset825 = trunc i64 %416 to i32, !dbg !17
  %417 = trunc i64 %414 to i32, !dbg !17
  %418 = insertelement <4 x i32> poison, i32 %417, i64 0, !dbg !17
  %419 = lshr i64 %414, 32, !dbg !17
  %420 = trunc i64 %419 to i32, !dbg !17
  %421 = insertelement <4 x i32> %418, i32 %420, i64 1, !dbg !17
  %422 = insertelement <4 x i32> %421, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc826 = insertelement <4 x i32> %422, i32 131072, i64 3, !dbg !17
  %423 = ptrtoint ptr addrspace(3) %413 to i32, !dbg !17
  %dma.buffer.lds827 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %423), !dbg !17
  %424 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds827, i32 %dma.buffer.offset825, <4 x i32> %dma.buffer.rsrc826) #10, !dbg !17
  %425 = or disjoint i64 %27, 256, !dbg !17
  br label %396, !dbg !17

426:                                              ; preds = %396
  %address.thread330 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !13, !range !40, !frisk.register_lifetimes !1
  %address.local331 = lshr i32 %address.thread330, 2, !dbg !13
  %address.local332 = and i32 %address.local331, 4, !dbg !13
  %address.local333 = shl nuw nsw i32 %address.thread330, 1, !dbg !13
  %address.local334 = and i32 %address.local333, 384, !dbg !13
  %address.local335 = or disjoint i32 %address.local332, %address.local334, !dbg !13
  %address.local336 = shl nuw nsw i32 %address.thread330, 3, !dbg !13
  %address.local337 = and i32 %address.local336, 120, !dbg !13
  %address.local338 = or disjoint i32 %address.local335, %address.local337, !dbg !13
  %address.local339 = shl nuw nsw i32 %address.thread330, 5, !dbg !13
  %address.local340 = and i32 %address.local339, 1024, !dbg !13
  %address.local341 = or disjoint i32 %address.local338, %address.local340, !dbg !13
  %address.local342 = zext nneg i32 %address.local341 to i64, !dbg !13
  %address.local343 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local342, !dbg !13
  %427 = load <4 x half>, ptr addrspace(3) %address.local343, align 2, !dbg !13
  %address.local344 = lshr i32 %address.thread330, 2, !dbg !13
  %address.local345 = and i32 %address.local344, 4, !dbg !13
  %address.local346 = shl nuw nsw i32 %address.thread330, 1, !dbg !13
  %address.local347 = and i32 %address.local346, 384, !dbg !13
  %address.local348 = or disjoint i32 %address.local345, %address.local347, !dbg !13
  %address.local349 = shl nuw nsw i32 %address.thread330, 3, !dbg !13
  %address.local350 = and i32 %address.local349, 120, !dbg !13
  %address.local351 = or disjoint i32 %address.local348, %address.local350, !dbg !13
  %address.local352 = shl nuw nsw i32 %address.thread330, 5, !dbg !13
  %address.local353 = and i32 %address.local352, 1024, !dbg !13
  %address.local354 = or disjoint i32 %address.local351, %address.local353, !dbg !13
  %address.local355 = or disjoint i32 %address.local354, 2048, !dbg !13
  %address.local356 = zext nneg i32 %address.local355 to i64, !dbg !13
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local356, !dbg !13
  %428 = load <4 x half>, ptr addrspace(3) %address.local357, align 2, !dbg !13
  %429 = fmul <4 x half> %427, splat (half 0xH3015), !dbg !13
  %430 = fmul <4 x half> %428, splat (half 0xH3015), !dbg !13
  %address.local358 = lshr i32 %address.thread330, 2, !dbg !13
  %address.local359 = and i32 %address.local358, 4, !dbg !13
  %address.local360 = shl nuw nsw i32 %address.thread330, 1, !dbg !13
  %address.local361 = and i32 %address.local360, 384, !dbg !13
  %address.local362 = or disjoint i32 %address.local359, %address.local361, !dbg !13
  %address.local363 = shl nuw nsw i32 %address.thread330, 3, !dbg !13
  %address.local364 = and i32 %address.local363, 120, !dbg !13
  %address.local365 = or disjoint i32 %address.local362, %address.local364, !dbg !13
  %address.local366 = shl nuw nsw i32 %address.thread330, 5, !dbg !13
  %address.local367 = and i32 %address.local366, 1024, !dbg !13
  %address.local368 = or disjoint i32 %address.local365, %address.local367, !dbg !13
  %address.local369 = or disjoint i32 %address.local368, 512, !dbg !13
  %address.local370 = zext nneg i32 %address.local369 to i64, !dbg !13
  %address.local371 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local370, !dbg !13
  %431 = load <4 x half>, ptr addrspace(3) %address.local371, align 2, !dbg !13
  %address.local372 = lshr i32 %address.thread330, 2, !dbg !13
  %address.local373 = and i32 %address.local372, 4, !dbg !13
  %address.local374 = shl nuw nsw i32 %address.thread330, 1, !dbg !13
  %address.local375 = and i32 %address.local374, 384, !dbg !13
  %address.local376 = or disjoint i32 %address.local373, %address.local375, !dbg !13
  %address.local377 = shl nuw nsw i32 %address.thread330, 3, !dbg !13
  %address.local378 = and i32 %address.local377, 120, !dbg !13
  %address.local379 = or disjoint i32 %address.local376, %address.local378, !dbg !13
  %address.local380 = shl nuw nsw i32 %address.thread330, 5, !dbg !13
  %address.local381 = and i32 %address.local380, 1024, !dbg !13
  %address.local382 = or disjoint i32 %address.local379, %address.local381, !dbg !13
  %address.local383 = or disjoint i32 %address.local382, 2560, !dbg !13
  %address.local384 = zext nneg i32 %address.local383 to i64, !dbg !13
  %address.local385 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local384, !dbg !13
  %432 = load <4 x half>, ptr addrspace(3) %address.local385, align 2, !dbg !13
  %433 = fmul <4 x half> %431, splat (half 0xH3015), !dbg !13
  %434 = fmul <4 x half> %432, splat (half 0xH3015), !dbg !13
  %address.local386 = lshr i32 %address.thread330, 2, !dbg !18
  %address.local387 = and i32 %address.local386, 4, !dbg !18
  %address.local388 = shl nuw nsw i32 %address.thread330, 3, !dbg !18
  %address.local389 = and i32 %address.local388, 376, !dbg !18
  %address.local390 = or disjoint i32 %address.local387, %address.local389, !dbg !18
  %address.local391 = zext nneg i32 %address.local390 to i64, !dbg !18
  %address.local392 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local391, !dbg !18
  %435 = load <4 x half>, ptr addrspace(3) %address.local392, align 2, !dbg !18
  %address.local393 = lshr i32 %address.thread330, 2, !dbg !18
  %address.local394 = and i32 %address.local393, 4, !dbg !18
  %address.local395 = shl nuw nsw i32 %address.thread330, 3, !dbg !18
  %address.local396 = and i32 %address.local395, 376, !dbg !18
  %address.local397 = or disjoint i32 %address.local394, %address.local396, !dbg !18
  %address.local398 = or disjoint i32 %address.local397, 512, !dbg !18
  %address.local399 = zext nneg i32 %address.local398 to i64, !dbg !18
  %address.local400 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local399, !dbg !18
  %436 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !18
  %437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %429, <4 x half> %435, <4 x float> %334) #9, !dbg !18
  %address.local401 = lshr i32 %address.thread330, 2, !dbg !18
  %address.local402 = and i32 %address.local401, 4, !dbg !18
  %address.local403 = shl nuw nsw i32 %address.thread330, 3, !dbg !18
  %address.local404 = and i32 %address.local403, 376, !dbg !18
  %address.local405 = or disjoint i32 %address.local402, %address.local404, !dbg !18
  %address.local406 = or disjoint i32 %address.local405, 128, !dbg !18
  %address.local407 = zext nneg i32 %address.local406 to i64, !dbg !18
  %address.local408 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local407, !dbg !18
  %438 = load <4 x half>, ptr addrspace(3) %address.local408, align 2, !dbg !18
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %430, <4 x half> %436, <4 x float> %437) #9, !dbg !18
  %address.local409 = lshr i32 %address.thread330, 2, !dbg !18
  %address.local410 = and i32 %address.local409, 4, !dbg !18
  %address.local411 = shl nuw nsw i32 %address.thread330, 3, !dbg !18
  %address.local412 = and i32 %address.local411, 376, !dbg !18
  %address.local413 = or disjoint i32 %address.local410, %address.local412, !dbg !18
  %address.local414 = or disjoint i32 %address.local413, 640, !dbg !18
  %address.local415 = zext nneg i32 %address.local414 to i64, !dbg !18
  %address.local416 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local415, !dbg !18
  %440 = load <4 x half>, ptr addrspace(3) %address.local416, align 2, !dbg !18
  %441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %429, <4 x half> %438, <4 x float> %337) #9, !dbg !18
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %430, <4 x half> %440, <4 x float> %441) #9, !dbg !18
  %443 = load <4 x half>, ptr addrspace(3) %address.local392, align 2, !dbg !18
  %444 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !18
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %433, <4 x half> %443, <4 x float> %342) #9, !dbg !18
  %446 = load <4 x half>, ptr addrspace(3) %address.local408, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %434, <4 x half> %444, <4 x float> %445) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %448 = load <4 x half>, ptr addrspace(3) %address.local416, align 2, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %433, <4 x half> %446, <4 x float> %345) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %434, <4 x half> %448, <4 x float> %449) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %451 = add nsw i64 %92, %8, !dbg !23
  %452 = shl i64 %451, 19, !dbg !23
  %453 = and i64 %452, 16252928, !dbg !23
  %454 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %453, !dbg !23
  %455 = getelementptr [2 x i8], ptr addrspace(1) %454, i64 %98, !dbg !23
  br label %456, !dbg !23

456:                                              ; preds = %459, %426
  %457 = phi i64 [ %485, %459 ], [ %27, %426 ], !dbg !23
  %458 = icmp samesign ult i64 %457, 128, !dbg !23
  br i1 %458, label %459, label %486, !dbg !23

459:                                              ; preds = %456
  %460 = shl nuw nsw i64 %457, 3, !dbg !23
  %461 = and i64 %460, 24, !dbg !23
  %462 = shl nuw nsw i64 %457, 3, !dbg !23
  %463 = and i64 %462, 512, !dbg !23
  %464 = shl nuw nsw i64 %457, 5, !dbg !23
  %465 = and i64 %464, 3968, !dbg !23
  %466 = getelementptr [2 x i8], ptr addrspace(1) %455, i64 %465, !dbg !23
  %467 = getelementptr [2 x i8], ptr addrspace(1) %466, i64 %461, !dbg !23
  %468 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %463, !dbg !23
  %469 = ptrtoint ptr addrspace(3) %468 to i64, !dbg !23
  %470 = trunc i64 %469 to i32, !dbg !23
  %471 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %470), !dbg !23
  %472 = zext i32 %471 to i64, !dbg !23
  %473 = inttoptr i64 %472 to ptr addrspace(3), !dbg !23
  %474 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !23
  %475 = ptrtoint ptr addrspace(1) %467 to i64, !dbg !23
  %476 = sub i64 %475, %474, !dbg !23
  %dma.buffer.offset828 = trunc i64 %476 to i32, !dbg !23
  %477 = trunc i64 %474 to i32, !dbg !23
  %478 = insertelement <4 x i32> poison, i32 %477, i64 0, !dbg !23
  %479 = lshr i64 %474, 32, !dbg !23
  %480 = trunc i64 %479 to i32, !dbg !23
  %481 = insertelement <4 x i32> %478, i32 %480, i64 1, !dbg !23
  %482 = insertelement <4 x i32> %481, i32 33554432, i64 2, !dbg !23
  %dma.buffer.rsrc829 = insertelement <4 x i32> %482, i32 131072, i64 3, !dbg !23
  %483 = ptrtoint ptr addrspace(3) %473 to i32, !dbg !23
  %dma.buffer.lds830 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %483), !dbg !23
  %484 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds830, i32 %dma.buffer.offset828, <4 x i32> %dma.buffer.rsrc829) #10, !dbg !23
  %485 = or disjoint i64 %27, 256, !dbg !23
  br label %456, !dbg !23

486:                                              ; preds = %456
  %address.thread417 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !13, !range !40, !frisk.register_lifetimes !1
  %address.local418 = lshr i32 %address.thread417, 2, !dbg !13
  %address.local419 = and i32 %address.local418, 4, !dbg !13
  %address.local420 = shl nuw nsw i32 %address.thread417, 1, !dbg !13
  %address.local421 = and i32 %address.local420, 384, !dbg !13
  %address.local422 = or disjoint i32 %address.local419, %address.local421, !dbg !13
  %address.local423 = shl nuw nsw i32 %address.thread417, 3, !dbg !13
  %address.local424 = and i32 %address.local423, 120, !dbg !13
  %address.local425 = or disjoint i32 %address.local422, %address.local424, !dbg !13
  %address.local426 = shl nuw nsw i32 %address.thread417, 5, !dbg !13
  %address.local427 = and i32 %address.local426, 1024, !dbg !13
  %address.local428 = or disjoint i32 %address.local425, %address.local427, !dbg !13
  %address.local429 = zext nneg i32 %address.local428 to i64, !dbg !13
  %address.local430 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local429, !dbg !13
  %487 = load <4 x half>, ptr addrspace(3) %address.local430, align 2, !dbg !13
  %address.local431 = lshr i32 %address.thread417, 2, !dbg !13
  %address.local432 = and i32 %address.local431, 4, !dbg !13
  %address.local433 = shl nuw nsw i32 %address.thread417, 1, !dbg !13
  %address.local434 = and i32 %address.local433, 384, !dbg !13
  %address.local435 = or disjoint i32 %address.local432, %address.local434, !dbg !13
  %address.local436 = shl nuw nsw i32 %address.thread417, 3, !dbg !13
  %address.local437 = and i32 %address.local436, 120, !dbg !13
  %address.local438 = or disjoint i32 %address.local435, %address.local437, !dbg !13
  %address.local439 = shl nuw nsw i32 %address.thread417, 5, !dbg !13
  %address.local440 = and i32 %address.local439, 1024, !dbg !13
  %address.local441 = or disjoint i32 %address.local438, %address.local440, !dbg !13
  %address.local442 = or disjoint i32 %address.local441, 2048, !dbg !13
  %address.local443 = zext nneg i32 %address.local442 to i64, !dbg !13
  %address.local444 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local443, !dbg !13
  %488 = load <4 x half>, ptr addrspace(3) %address.local444, align 2, !dbg !13
  %489 = fmul <4 x half> %487, splat (half 0xH3015), !dbg !13
  %490 = fmul <4 x half> %488, splat (half 0xH3015), !dbg !13
  %address.local445 = lshr i32 %address.thread417, 2, !dbg !13
  %address.local446 = and i32 %address.local445, 4, !dbg !13
  %address.local447 = shl nuw nsw i32 %address.thread417, 1, !dbg !13
  %address.local448 = and i32 %address.local447, 384, !dbg !13
  %address.local449 = or disjoint i32 %address.local446, %address.local448, !dbg !13
  %address.local450 = shl nuw nsw i32 %address.thread417, 3, !dbg !13
  %address.local451 = and i32 %address.local450, 120, !dbg !13
  %address.local452 = or disjoint i32 %address.local449, %address.local451, !dbg !13
  %address.local453 = shl nuw nsw i32 %address.thread417, 5, !dbg !13
  %address.local454 = and i32 %address.local453, 1024, !dbg !13
  %address.local455 = or disjoint i32 %address.local452, %address.local454, !dbg !13
  %address.local456 = or disjoint i32 %address.local455, 512, !dbg !13
  %address.local457 = zext nneg i32 %address.local456 to i64, !dbg !13
  %address.local458 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local457, !dbg !13
  %491 = load <4 x half>, ptr addrspace(3) %address.local458, align 2, !dbg !13
  %address.local459 = lshr i32 %address.thread417, 2, !dbg !13
  %address.local460 = and i32 %address.local459, 4, !dbg !13
  %address.local461 = shl nuw nsw i32 %address.thread417, 1, !dbg !13
  %address.local462 = and i32 %address.local461, 384, !dbg !13
  %address.local463 = or disjoint i32 %address.local460, %address.local462, !dbg !13
  %address.local464 = shl nuw nsw i32 %address.thread417, 3, !dbg !13
  %address.local465 = and i32 %address.local464, 120, !dbg !13
  %address.local466 = or disjoint i32 %address.local463, %address.local465, !dbg !13
  %address.local467 = shl nuw nsw i32 %address.thread417, 5, !dbg !13
  %address.local468 = and i32 %address.local467, 1024, !dbg !13
  %address.local469 = or disjoint i32 %address.local466, %address.local468, !dbg !13
  %address.local470 = or disjoint i32 %address.local469, 2560, !dbg !13
  %address.local471 = zext nneg i32 %address.local470 to i64, !dbg !13
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local471, !dbg !13
  %492 = load <4 x half>, ptr addrspace(3) %address.local472, align 2, !dbg !13
  %493 = fmul <4 x half> %491, splat (half 0xH3015), !dbg !13
  %494 = fmul <4 x half> %492, splat (half 0xH3015), !dbg !13
  %address.local473 = lshr i32 %address.thread417, 2, !dbg !18
  %address.local474 = and i32 %address.local473, 4, !dbg !18
  %address.local475 = shl nuw nsw i32 %address.thread417, 3, !dbg !18
  %address.local476 = and i32 %address.local475, 376, !dbg !18
  %address.local477 = or disjoint i32 %address.local474, %address.local476, !dbg !18
  %address.local478 = zext nneg i32 %address.local477 to i64, !dbg !18
  %address.local479 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 18432), i64 %address.local478, !dbg !18
  %495 = load <4 x half>, ptr addrspace(3) %address.local479, align 2, !dbg !18
  %address.local480 = lshr i32 %address.thread417, 2, !dbg !18
  %address.local481 = and i32 %address.local480, 4, !dbg !18
  %address.local482 = shl nuw nsw i32 %address.thread417, 3, !dbg !18
  %address.local483 = and i32 %address.local482, 376, !dbg !18
  %address.local484 = or disjoint i32 %address.local481, %address.local483, !dbg !18
  %address.local485 = or disjoint i32 %address.local484, 512, !dbg !18
  %address.local486 = zext nneg i32 %address.local485 to i64, !dbg !18
  %address.local487 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 18432), i64 %address.local486, !dbg !18
  %496 = load <4 x half>, ptr addrspace(3) %address.local487, align 2, !dbg !18
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %489, <4 x half> %495, <4 x float> %439) #9, !dbg !18
  %address.local488 = lshr i32 %address.thread417, 2, !dbg !18
  %address.local489 = and i32 %address.local488, 4, !dbg !18
  %address.local490 = shl nuw nsw i32 %address.thread417, 3, !dbg !18
  %address.local491 = and i32 %address.local490, 376, !dbg !18
  %address.local492 = or disjoint i32 %address.local489, %address.local491, !dbg !18
  %address.local493 = or disjoint i32 %address.local492, 128, !dbg !18
  %address.local494 = zext nneg i32 %address.local493 to i64, !dbg !18
  %address.local495 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 18432), i64 %address.local494, !dbg !18
  %498 = load <4 x half>, ptr addrspace(3) %address.local495, align 2, !dbg !18
  %499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %490, <4 x half> %496, <4 x float> %497) #9, !dbg !18
  %address.local496 = lshr i32 %address.thread417, 2, !dbg !18
  %address.local497 = and i32 %address.local496, 4, !dbg !18
  %address.local498 = shl nuw nsw i32 %address.thread417, 3, !dbg !18
  %address.local499 = and i32 %address.local498, 376, !dbg !18
  %address.local500 = or disjoint i32 %address.local497, %address.local499, !dbg !18
  %address.local501 = or disjoint i32 %address.local500, 640, !dbg !18
  %address.local502 = zext nneg i32 %address.local501 to i64, !dbg !18
  %address.local503 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 18432), i64 %address.local502, !dbg !18
  %500 = load <4 x half>, ptr addrspace(3) %address.local503, align 2, !dbg !18
  %501 = extractelement <4 x float> %499, i64 0, !dbg !22
  %502 = extractelement <4 x float> %499, i64 1, !dbg !22
  %503 = select i1 %.not, float %502, float %501, !dbg !14
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %489, <4 x half> %498, <4 x float> %442) #9, !dbg !18
  %505 = extractelement <4 x float> %499, i64 2, !dbg !22
  %506 = extractelement <4 x float> %499, i64 3, !dbg !22
  %507 = load <4 x half>, ptr addrspace(3) %address.local479, align 2, !dbg !18
  %508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %490, <4 x half> %500, <4 x float> %504) #9, !dbg !18
  %509 = load <4 x half>, ptr addrspace(3) %address.local487, align 2, !dbg !18
  %510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %493, <4 x half> %507, <4 x float> %447) #9, !dbg !18
  %511 = extractelement <4 x float> %508, i64 0, !dbg !22
  %512 = extractelement <4 x float> %508, i64 1, !dbg !22
  %513 = extractelement <4 x float> %508, i64 2, !dbg !22
  %514 = extractelement <4 x float> %508, i64 3, !dbg !22
  %515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %494, <4 x half> %509, <4 x float> %510) #9, !dbg !18
  %516 = load <4 x half>, ptr addrspace(3) %address.local495, align 2, !dbg !18
  %517 = load <4 x half>, ptr addrspace(3) %address.local503, align 2, !dbg !18
  %518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %493, <4 x half> %516, <4 x float> %450) #9, !dbg !18
  %519 = extractelement <4 x float> %515, i64 0, !dbg !22
  %520 = extractelement <4 x float> %515, i64 1, !dbg !22
  %521 = extractelement <4 x float> %515, i64 2, !dbg !22
  %522 = extractelement <4 x float> %515, i64 3, !dbg !22
  %523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %494, <4 x half> %517, <4 x float> %518) #9, !dbg !18
  %524 = extractelement <4 x float> %523, i64 0, !dbg !22
  %525 = extractelement <4 x float> %523, i64 1, !dbg !22
  %526 = extractelement <4 x float> %523, i64 2, !dbg !22
  %527 = extractelement <4 x float> %523, i64 3, !dbg !22
  %528 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !14
  %529 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %528), !dbg !14
  %530 = add nuw nsw i32 %529, 64, !dbg !14
  %531 = xor i32 %529, 16, !dbg !14
  %532 = and i32 %530, 192, !dbg !14
  %533 = icmp samesign ult i32 %531, %532, !dbg !14
  %534 = select i1 %533, i32 %531, i32 %529, !dbg !14
  %535 = shl nuw nsw i32 %534, 2, !dbg !14
  %536 = bitcast float %503 to i32, !dbg !14
  %537 = call i32 @llvm.amdgcn.ds.bpermute(i32 %535, i32 %536), !dbg !14
  %538 = bitcast i32 %537 to float, !dbg !14
  %539 = select i1 %.not, float %501, float %538, !dbg !14
  %540 = select i1 %.not, float %538, float %502, !dbg !14
  %541 = select i1 %.not, float %506, float %505, !dbg !14
  %542 = bitcast float %541 to i32, !dbg !14
  %543 = call i32 @llvm.amdgcn.ds.bpermute(i32 %535, i32 %542), !dbg !14
  %544 = bitcast i32 %543 to float, !dbg !14
  %545 = select i1 %.not, float %505, float %544, !dbg !14
  %546 = select i1 %.not, float %544, float %506, !dbg !14
  %547 = select i1 %.not2, float %545, float %539, !dbg !14
  %548 = xor i32 %529, 32, !dbg !14
  %549 = icmp samesign ult i32 %548, %532, !dbg !14
  %550 = select i1 %549, i32 %548, i32 %529, !dbg !14
  %551 = shl nuw nsw i32 %550, 2, !dbg !14
  %552 = bitcast float %547 to i32, !dbg !14
  %553 = call i32 @llvm.amdgcn.ds.bpermute(i32 %551, i32 %552), !dbg !14
  %554 = bitcast i32 %553 to float, !dbg !14
  %555 = select i1 %.not2, float %539, float %554, !dbg !14
  %556 = select i1 %.not2, float %554, float %545, !dbg !14
  %557 = select i1 %.not2, float %546, float %540, !dbg !14
  %558 = bitcast float %557 to i32, !dbg !14
  %559 = call i32 @llvm.amdgcn.ds.bpermute(i32 %551, i32 %558), !dbg !14
  %560 = bitcast i32 %559 to float, !dbg !14
  %561 = select i1 %.not2, float %540, float %560, !dbg !14
  %562 = select i1 %.not2, float %560, float %546, !dbg !14
  %563 = select i1 %.not, float %512, float %511, !dbg !14
  %564 = bitcast float %563 to i32, !dbg !14
  %565 = call i32 @llvm.amdgcn.ds.bpermute(i32 %535, i32 %564), !dbg !14
  %566 = bitcast i32 %565 to float, !dbg !14
  %567 = select i1 %.not, float %511, float %566, !dbg !14
  %568 = select i1 %.not, float %566, float %512, !dbg !14
  %569 = select i1 %.not, float %514, float %513, !dbg !14
  %570 = bitcast float %569 to i32, !dbg !14
  %571 = call i32 @llvm.amdgcn.ds.bpermute(i32 %535, i32 %570), !dbg !14
  %572 = bitcast i32 %571 to float, !dbg !14
  %573 = select i1 %.not, float %513, float %572, !dbg !14
  %574 = select i1 %.not, float %572, float %514, !dbg !14
  %575 = select i1 %.not2, float %573, float %567, !dbg !14
  %576 = bitcast float %575 to i32, !dbg !14
  %577 = call i32 @llvm.amdgcn.ds.bpermute(i32 %551, i32 %576), !dbg !14
  %578 = bitcast i32 %577 to float, !dbg !14
  %579 = select i1 %.not2, float %567, float %578, !dbg !14
  %580 = select i1 %.not2, float %578, float %573, !dbg !14
  %581 = select i1 %.not2, float %574, float %568, !dbg !14
  %582 = bitcast float %581 to i32, !dbg !14
  %583 = call i32 @llvm.amdgcn.ds.bpermute(i32 %551, i32 %582), !dbg !14
  %584 = bitcast i32 %583 to float, !dbg !14
  %585 = select i1 %.not2, float %568, float %584, !dbg !14
  %586 = select i1 %.not2, float %584, float %574, !dbg !14
  %587 = select i1 %.not, float %520, float %519, !dbg !14
  %588 = bitcast float %587 to i32, !dbg !14
  %589 = call i32 @llvm.amdgcn.ds.bpermute(i32 %535, i32 %588), !dbg !14
  %590 = bitcast i32 %589 to float, !dbg !14
  %591 = select i1 %.not, float %519, float %590, !dbg !14
  %592 = select i1 %.not, float %590, float %520, !dbg !14
  %593 = select i1 %.not, float %522, float %521, !dbg !14
  %594 = bitcast float %593 to i32, !dbg !14
  %595 = call i32 @llvm.amdgcn.ds.bpermute(i32 %535, i32 %594), !dbg !14
  %596 = bitcast i32 %595 to float, !dbg !14
  %597 = select i1 %.not, float %521, float %596, !dbg !14
  %598 = select i1 %.not, float %596, float %522, !dbg !14
  %599 = select i1 %.not2, float %597, float %591, !dbg !14
  %600 = bitcast float %599 to i32, !dbg !14
  %601 = call i32 @llvm.amdgcn.ds.bpermute(i32 %551, i32 %600), !dbg !14
  %602 = bitcast i32 %601 to float, !dbg !14
  %603 = select i1 %.not2, float %591, float %602, !dbg !14
  %604 = select i1 %.not2, float %602, float %597, !dbg !14
  %605 = select i1 %.not2, float %598, float %592, !dbg !14
  %606 = bitcast float %605 to i32, !dbg !14
  %607 = call i32 @llvm.amdgcn.ds.bpermute(i32 %551, i32 %606), !dbg !14
  %608 = bitcast i32 %607 to float, !dbg !14
  %609 = select i1 %.not2, float %592, float %608, !dbg !14
  %610 = select i1 %.not2, float %608, float %598, !dbg !14
  %611 = select i1 %.not, float %525, float %524, !dbg !14
  %612 = bitcast float %611 to i32, !dbg !14
  %613 = call i32 @llvm.amdgcn.ds.bpermute(i32 %535, i32 %612), !dbg !14
  %614 = bitcast i32 %613 to float, !dbg !14
  %615 = select i1 %.not, float %524, float %614, !dbg !14
  %616 = select i1 %.not, float %614, float %525, !dbg !14
  %617 = select i1 %.not, float %527, float %526, !dbg !14
  %618 = bitcast float %617 to i32, !dbg !14
  %619 = call i32 @llvm.amdgcn.ds.bpermute(i32 %535, i32 %618), !dbg !14
  %620 = bitcast i32 %619 to float, !dbg !14
  %621 = select i1 %.not, float %526, float %620, !dbg !14
  %622 = select i1 %.not, float %620, float %527, !dbg !14
  %623 = select i1 %.not2, float %621, float %615, !dbg !14
  %624 = bitcast float %623 to i32, !dbg !14
  %625 = call i32 @llvm.amdgcn.ds.bpermute(i32 %551, i32 %624), !dbg !14
  %626 = bitcast i32 %625 to float, !dbg !14
  %627 = select i1 %.not2, float %615, float %626, !dbg !14
  %628 = select i1 %.not2, float %626, float %621, !dbg !14
  %629 = select i1 %.not2, float %622, float %616, !dbg !14
  %630 = bitcast float %629 to i32, !dbg !14
  %631 = call i32 @llvm.amdgcn.ds.bpermute(i32 %551, i32 %630), !dbg !14
  %632 = bitcast i32 %631 to float, !dbg !14
  %633 = select i1 %.not2, float %616, float %632, !dbg !14
  %634 = select i1 %.not2, float %632, float %622, !dbg !14
  %635 = or disjoint i64 %61, 31, !dbg !15
  %636 = icmp uge i64 %11, %635, !dbg !15
  %637 = and i1 %35, %636, !dbg !15
  br i1 %637, label %638, label %639, !dbg !15

638:                                              ; preds = %486
  br label %686, !dbg !15

639:                                              ; preds = %486
  %640 = add nsw i64 %11, %27, !dbg !15
  %641 = shl nuw nsw i64 %28, 4, !dbg !15
  %642 = add nsw i64 %640, %641, !dbg !15
  %643 = and i32 %26, 240, !dbg !15
  %index.compact = zext nneg i32 %643 to i64, !dbg !15
  %644 = sub nsw i64 %642, %index.compact, !dbg !15
  %645 = or disjoint i64 %61, %30, !dbg !15
  %646 = add nsw i64 %644, 1, !dbg !24
  %647 = lshr i64 %646, 32, !dbg !25
  %compare.high = trunc i64 %647 to i32, !dbg !25
  %648 = lshr i64 %645, 32, !dbg !25
  %compare.high755 = trunc i64 %648 to i32, !dbg !25
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high755, !dbg !25
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high755, !dbg !25
  %compare.low.base = trunc i64 %646 to i32, !dbg !25
  %compare.low.base756 = trunc i64 %645 to i32, !dbg !25
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base756, !dbg !25
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !25
  %649 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base756, i32 1) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !25
  %compare.low757 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !25
  %compare.result758 = select i1 %compare.high.equal, i1 %compare.low757, i1 %compare.high.order, !dbg !25
  %650 = select i1 %compare.result758, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base759 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base756, i32 2) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local760 = or i32 %compare.local.base759, 2, !dbg !25
  %compare.low761 = icmp ugt i32 %compare.low.base, %compare.local760, !dbg !25
  %compare.result762 = select i1 %compare.high.equal, i1 %compare.low761, i1 %compare.high.order, !dbg !25
  %651 = select i1 %compare.result762, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base763 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base756, i32 3) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local764 = or i32 %compare.local.base763, 3, !dbg !25
  %compare.low765 = icmp ugt i32 %compare.low.base, %compare.local764, !dbg !25
  %compare.result766 = select i1 %compare.high.equal, i1 %compare.low765, i1 %compare.high.order, !dbg !25
  %652 = select i1 %compare.result766, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base767 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base756, i32 4) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local768 = or i32 %compare.local.base767, 16, !dbg !25
  %compare.low769 = icmp ugt i32 %compare.low.base, %compare.local768, !dbg !25
  %compare.result770 = select i1 %compare.high.equal, i1 %compare.low769, i1 %compare.high.order, !dbg !25
  %653 = select i1 %compare.result770, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base771 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base756, i32 5) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local772 = or i32 %compare.local.base771, 17, !dbg !25
  %compare.low773 = icmp ugt i32 %compare.low.base, %compare.local772, !dbg !25
  %compare.result774 = select i1 %compare.high.equal, i1 %compare.low773, i1 %compare.high.order, !dbg !25
  %654 = select i1 %compare.result774, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base775 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base756, i32 6) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local776 = or i32 %compare.local.base775, 18, !dbg !25
  %compare.low777 = icmp ugt i32 %compare.low.base, %compare.local776, !dbg !25
  %compare.result778 = select i1 %compare.high.equal, i1 %compare.low777, i1 %compare.high.order, !dbg !25
  %655 = select i1 %compare.result778, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base779 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base756, i32 7) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local780 = or i32 %compare.local.base779, 19, !dbg !25
  %compare.low781 = icmp ugt i32 %compare.low.base, %compare.local780, !dbg !25
  %compare.result782 = select i1 %compare.high.equal, i1 %compare.low781, i1 %compare.high.order, !dbg !25
  %656 = select i1 %compare.result782, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %657 = add nsw i64 %644, 65, !dbg !24
  %658 = lshr i64 %657, 32, !dbg !25
  %compare.high783 = trunc i64 %658 to i32, !dbg !25
  %659 = lshr i64 %645, 32, !dbg !25
  %compare.high784 = trunc i64 %659 to i32, !dbg !25
  %compare.high.equal785 = icmp eq i32 %compare.high783, %compare.high784, !dbg !25
  %compare.high.order786 = icmp ugt i32 %compare.high783, %compare.high784, !dbg !25
  %compare.low.base787 = trunc i64 %657 to i32, !dbg !25
  %compare.low.base788 = trunc i64 %645 to i32, !dbg !25
  %compare.low789 = icmp ugt i32 %compare.low.base787, %compare.low.base788, !dbg !25
  %compare.result790 = select i1 %compare.high.equal785, i1 %compare.low789, i1 %compare.high.order786, !dbg !25
  %660 = select i1 %compare.result790, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base791 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base788, i32 8) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local792 = or i32 %compare.local.base791, 1, !dbg !25
  %compare.low793 = icmp ugt i32 %compare.low.base787, %compare.local792, !dbg !25
  %compare.result794 = select i1 %compare.high.equal785, i1 %compare.low793, i1 %compare.high.order786, !dbg !25
  %661 = select i1 %compare.result794, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base795 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base788, i32 9) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local796 = or i32 %compare.local.base795, 2, !dbg !25
  %compare.low797 = icmp ugt i32 %compare.low.base787, %compare.local796, !dbg !25
  %compare.result798 = select i1 %compare.high.equal785, i1 %compare.low797, i1 %compare.high.order786, !dbg !25
  %662 = select i1 %compare.result798, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base799 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base788, i32 10) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local800 = or i32 %compare.local.base799, 3, !dbg !25
  %compare.low801 = icmp ugt i32 %compare.low.base787, %compare.local800, !dbg !25
  %compare.result802 = select i1 %compare.high.equal785, i1 %compare.low801, i1 %compare.high.order786, !dbg !25
  %663 = select i1 %compare.result802, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base803 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base788, i32 11) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local804 = or i32 %compare.local.base803, 16, !dbg !25
  %compare.low805 = icmp ugt i32 %compare.low.base787, %compare.local804, !dbg !25
  %compare.result806 = select i1 %compare.high.equal785, i1 %compare.low805, i1 %compare.high.order786, !dbg !25
  %664 = select i1 %compare.result806, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base807 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base788, i32 12) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local808 = or i32 %compare.local.base807, 17, !dbg !25
  %compare.low809 = icmp ugt i32 %compare.low.base787, %compare.local808, !dbg !25
  %compare.result810 = select i1 %compare.high.equal785, i1 %compare.low809, i1 %compare.high.order786, !dbg !25
  %665 = select i1 %compare.result810, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base811 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base788, i32 13) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local812 = or i32 %compare.local.base811, 18, !dbg !25
  %compare.low813 = icmp ugt i32 %compare.low.base787, %compare.local812, !dbg !25
  %compare.result814 = select i1 %compare.high.equal785, i1 %compare.low813, i1 %compare.high.order786, !dbg !25
  %666 = select i1 %compare.result814, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base815 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base788, i32 14) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local816 = or i32 %compare.local.base815, 19, !dbg !25
  %compare.low817 = icmp ugt i32 %compare.low.base787, %compare.local816, !dbg !25
  %compare.result818 = select i1 %compare.high.equal785, i1 %compare.low817, i1 %compare.high.order786, !dbg !25
  %667 = select i1 %compare.result818, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %668 = insertelement <8 x float> poison, float %649, i64 0, !dbg !15
  %669 = insertelement <8 x float> %668, float %650, i64 1, !dbg !15
  %670 = insertelement <8 x float> %669, float %651, i64 2, !dbg !15
  %671 = insertelement <8 x float> %670, float %652, i64 3, !dbg !15
  %672 = insertelement <8 x float> %671, float %653, i64 4, !dbg !15
  %673 = insertelement <8 x float> %672, float %654, i64 5, !dbg !15
  %674 = insertelement <8 x float> %673, float %655, i64 6, !dbg !15
  %675 = insertelement <8 x float> %674, float %656, i64 7, !dbg !15
  %676 = insertvalue [2 x <8 x float>] poison, <8 x float> %675, 0, !dbg !15
  %677 = insertelement <8 x float> poison, float %660, i64 0, !dbg !15
  %678 = insertelement <8 x float> %677, float %661, i64 1, !dbg !15
  %679 = insertelement <8 x float> %678, float %662, i64 2, !dbg !15
  %680 = insertelement <8 x float> %679, float %663, i64 3, !dbg !15
  %681 = insertelement <8 x float> %680, float %664, i64 4, !dbg !15
  %682 = insertelement <8 x float> %681, float %665, i64 5, !dbg !15
  %683 = insertelement <8 x float> %682, float %666, i64 6, !dbg !15
  %684 = insertelement <8 x float> %683, float %667, i64 7, !dbg !15
  %685 = insertvalue [2 x <8 x float>] %676, <8 x float> %684, 1, !dbg !15
  br label %686, !dbg !15

686:                                              ; preds = %638, %639
  %687 = phi [2 x <8 x float>] [ %685, %639 ], [ zeroinitializer, %638 ], !dbg !15
  br label %688, !dbg !15

688:                                              ; preds = %686
  %689 = extractvalue [2 x <8 x float>] %687, 0, !dbg !27
  %690 = insertelement <4 x float> poison, float %555, i64 0, !dbg !27
  %691 = insertelement <4 x float> %690, float %561, i64 1, !dbg !27
  %692 = insertelement <4 x float> %691, float %556, i64 2, !dbg !27
  %693 = insertelement <4 x float> %692, float %562, i64 3, !dbg !27
  %694 = shufflevector <8 x float> %689, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %695 = fadd <4 x float> %693, %694, !dbg !27
  %696 = extractelement <4 x float> %695, i64 0, !dbg !27
  %697 = extractelement <4 x float> %695, i64 1, !dbg !27
  %698 = extractelement <4 x float> %695, i64 2, !dbg !27
  %699 = extractelement <4 x float> %695, i64 3, !dbg !27
  %700 = insertelement <4 x float> poison, float %579, i64 0, !dbg !27
  %701 = insertelement <4 x float> %700, float %585, i64 1, !dbg !27
  %702 = insertelement <4 x float> %701, float %580, i64 2, !dbg !27
  %703 = insertelement <4 x float> %702, float %586, i64 3, !dbg !27
  %704 = shufflevector <8 x float> %689, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %705 = fadd <4 x float> %703, %704, !dbg !27
  %706 = extractelement <4 x float> %705, i64 0, !dbg !27
  %707 = extractelement <4 x float> %705, i64 1, !dbg !27
  %708 = extractelement <4 x float> %705, i64 2, !dbg !27
  %709 = extractelement <4 x float> %705, i64 3, !dbg !27
  %710 = extractvalue [2 x <8 x float>] %687, 1, !dbg !27
  %711 = insertelement <4 x float> poison, float %603, i64 0, !dbg !27
  %712 = insertelement <4 x float> %711, float %609, i64 1, !dbg !27
  %713 = insertelement <4 x float> %712, float %604, i64 2, !dbg !27
  %714 = insertelement <4 x float> %713, float %610, i64 3, !dbg !27
  %715 = shufflevector <8 x float> %710, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %716 = fadd <4 x float> %714, %715, !dbg !27
  %717 = extractelement <4 x float> %716, i64 0, !dbg !27
  %718 = extractelement <4 x float> %716, i64 1, !dbg !27
  %719 = extractelement <4 x float> %716, i64 2, !dbg !27
  %720 = extractelement <4 x float> %716, i64 3, !dbg !27
  %721 = insertelement <4 x float> poison, float %627, i64 0, !dbg !27
  %722 = insertelement <4 x float> %721, float %633, i64 1, !dbg !27
  %723 = insertelement <4 x float> %722, float %628, i64 2, !dbg !27
  %724 = insertelement <4 x float> %723, float %634, i64 3, !dbg !27
  %725 = shufflevector <8 x float> %710, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %726 = fadd <4 x float> %724, %725, !dbg !27
  %727 = extractelement <4 x float> %726, i64 0, !dbg !27
  %728 = extractelement <4 x float> %726, i64 1, !dbg !27
  %729 = extractelement <4 x float> %726, i64 2, !dbg !27
  %730 = extractelement <4 x float> %726, i64 3, !dbg !27
  %731 = call float @llvm.maxnum.f32(float %696, float 0xFFF0000000000000), !dbg !28
  %732 = call float @llvm.maxnum.f32(float %731, float %697), !dbg !28
  %733 = call float @llvm.maxnum.f32(float %732, float %698), !dbg !28
  %734 = call float @llvm.maxnum.f32(float %733, float %699), !dbg !28
  %735 = call float @llvm.maxnum.f32(float %734, float %706), !dbg !28
  %736 = call float @llvm.maxnum.f32(float %735, float %707), !dbg !28
  %737 = call float @llvm.maxnum.f32(float %736, float %708), !dbg !28
  %738 = call float @llvm.maxnum.f32(float %737, float %709), !dbg !28
  %739 = bitcast float %738 to i32, !dbg !28
  %740 = call i32 @llvm.amdgcn.ds.bpermute(i32 %535, i32 %739), !dbg !28
  %741 = bitcast i32 %740 to float, !dbg !28
  %742 = call float @llvm.maxnum.f32(float %738, float %741), !dbg !28
  %743 = bitcast float %742 to i32, !dbg !28
  %744 = call i32 @llvm.amdgcn.ds.bpermute(i32 %551, i32 %743), !dbg !28
  %745 = bitcast i32 %744 to float, !dbg !28
  %746 = call float @llvm.maxnum.f32(float %742, float %745), !dbg !28
  %747 = extractvalue [2 x <1 x float>] %55, 0, !dbg !28
  %748 = insertelement <1 x float> poison, float %746, i64 0, !dbg !28
  %749 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %748, <1 x float> %747), !dbg !28
  %750 = call float @llvm.maxnum.f32(float %717, float 0xFFF0000000000000), !dbg !28
  %751 = call float @llvm.maxnum.f32(float %750, float %718), !dbg !28
  %752 = call float @llvm.maxnum.f32(float %751, float %719), !dbg !28
  %753 = call float @llvm.maxnum.f32(float %752, float %720), !dbg !28
  %754 = call float @llvm.maxnum.f32(float %753, float %727), !dbg !28
  %755 = call float @llvm.maxnum.f32(float %754, float %728), !dbg !28
  %756 = call float @llvm.maxnum.f32(float %755, float %729), !dbg !28
  %757 = call float @llvm.maxnum.f32(float %756, float %730), !dbg !28
  %758 = bitcast float %757 to i32, !dbg !28
  %759 = call i32 @llvm.amdgcn.ds.bpermute(i32 %535, i32 %758), !dbg !28
  %760 = bitcast i32 %759 to float, !dbg !28
  %761 = call float @llvm.maxnum.f32(float %757, float %760), !dbg !28
  %762 = bitcast float %761 to i32, !dbg !28
  %763 = call i32 @llvm.amdgcn.ds.bpermute(i32 %551, i32 %762), !dbg !28
  %764 = bitcast i32 %763 to float, !dbg !28
  %765 = call float @llvm.maxnum.f32(float %761, float %764), !dbg !28
  %766 = extractvalue [2 x <1 x float>] %55, 1, !dbg !28
  %767 = insertelement <1 x float> poison, float %765, i64 0, !dbg !28
  %768 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %767, <1 x float> %766), !dbg !28
  %769 = shufflevector <1 x float> %749, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !29
  %770 = fsub <4 x float> %695, %769, !dbg !29
  %771 = fsub <4 x float> %705, %769, !dbg !29
  %772 = shufflevector <1 x float> %768, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !29
  %773 = fsub <4 x float> %716, %772, !dbg !29
  %774 = fsub <4 x float> %726, %772, !dbg !29
  %775 = fsub <1 x float> %747, %749, !dbg !30
  %776 = extractelement <1 x float> %775, i64 0, !dbg !31
  %777 = call float @llvm.amdgcn.exp2.f32(float %776), !dbg !31
  %778 = extractvalue [2 x <1 x float>] %54, 0, !dbg !32
  %779 = extractelement <1 x float> %778, i64 0, !dbg !32
  %780 = fmul float %779, %777, !dbg !32
  %781 = fsub <1 x float> %766, %768, !dbg !30
  %782 = extractelement <1 x float> %781, i64 0, !dbg !31
  %783 = call float @llvm.amdgcn.exp2.f32(float %782), !dbg !31
  %784 = extractvalue [2 x <1 x float>] %54, 1, !dbg !32
  %785 = extractelement <1 x float> %784, i64 0, !dbg !32
  %786 = fmul float %785, %783, !dbg !32
  %787 = extractelement <4 x float> %770, i64 0, !dbg !33
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !33
  %789 = insertelement <4 x float> poison, float %788, i64 0, !dbg !35
  %790 = extractelement <4 x float> %770, i64 1, !dbg !33
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !33
  %792 = insertelement <4 x float> %789, float %791, i64 1, !dbg !35
  %793 = fadd float %788, %791, !dbg !34
  %794 = extractelement <4 x float> %770, i64 2, !dbg !33
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !33
  %796 = insertelement <4 x float> %792, float %795, i64 2, !dbg !35
  %797 = fadd float %793, %795, !dbg !34
  %798 = extractelement <4 x float> %770, i64 3, !dbg !33
  %799 = call float @llvm.amdgcn.exp2.f32(float %798), !dbg !33
  %800 = insertelement <4 x float> %796, float %799, i64 3, !dbg !35
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %800) #9, !dbg !35, !frisk.register_lifetimes !1
  %801 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !35
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %801) #9, !dbg !34, !frisk.register_lifetimes !1
  %802 = fadd float %797, %799, !dbg !34
  %803 = extractelement <4 x float> %771, i64 0, !dbg !33
  %804 = call float @llvm.amdgcn.exp2.f32(float %803), !dbg !33
  %805 = insertelement <4 x float> poison, float %804, i64 0, !dbg !35
  %806 = fadd float %802, %804, !dbg !34
  %807 = extractelement <4 x float> %771, i64 1, !dbg !33
  %808 = call float @llvm.amdgcn.exp2.f32(float %807), !dbg !33
  %809 = insertelement <4 x float> %805, float %808, i64 1, !dbg !35
  %810 = fadd float %806, %808, !dbg !34
  %811 = extractelement <4 x float> %771, i64 2, !dbg !33
  %812 = call float @llvm.amdgcn.exp2.f32(float %811), !dbg !33
  %813 = insertelement <4 x float> %809, float %812, i64 2, !dbg !35
  %814 = fadd float %810, %812, !dbg !34
  %815 = extractelement <4 x float> %771, i64 3, !dbg !33
  %816 = call float @llvm.amdgcn.exp2.f32(float %815), !dbg !33
  %817 = insertelement <4 x float> %813, float %816, i64 3, !dbg !35
  %math.result54 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %817) #9, !dbg !35, !frisk.register_lifetimes !1
  %818 = fptrunc <4 x float> %math.result54 to <4 x half>, !dbg !35
  %math.packed55 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %818) #9, !dbg !34, !frisk.register_lifetimes !1
  %819 = fadd float %814, %816, !dbg !34
  %820 = extractelement <4 x float> %773, i64 0, !dbg !33
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !33
  %822 = insertelement <4 x float> poison, float %821, i64 0, !dbg !35
  %823 = extractelement <4 x float> %773, i64 1, !dbg !33
  %824 = call float @llvm.amdgcn.exp2.f32(float %823), !dbg !33
  %825 = insertelement <4 x float> %822, float %824, i64 1, !dbg !35
  %826 = fadd float %821, %824, !dbg !34
  %827 = extractelement <4 x float> %773, i64 2, !dbg !33
  %828 = call float @llvm.amdgcn.exp2.f32(float %827), !dbg !33
  %829 = insertelement <4 x float> %825, float %828, i64 2, !dbg !35
  %830 = fadd float %826, %828, !dbg !34
  %831 = extractelement <4 x float> %773, i64 3, !dbg !33
  %832 = call float @llvm.amdgcn.exp2.f32(float %831), !dbg !33
  %833 = insertelement <4 x float> %829, float %832, i64 3, !dbg !35
  %math.result56 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %833) #9, !dbg !35, !frisk.register_lifetimes !1
  %834 = fptrunc <4 x float> %math.result56 to <4 x half>, !dbg !35
  %math.packed57 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %834) #9, !dbg !34, !frisk.register_lifetimes !1
  %835 = fadd float %830, %832, !dbg !34
  %836 = extractelement <4 x float> %774, i64 0, !dbg !33
  %837 = call float @llvm.amdgcn.exp2.f32(float %836), !dbg !33
  %838 = insertelement <4 x float> poison, float %837, i64 0, !dbg !35
  %839 = fadd float %835, %837, !dbg !34
  %840 = extractelement <4 x float> %774, i64 1, !dbg !33
  %841 = call float @llvm.amdgcn.exp2.f32(float %840), !dbg !33
  %842 = insertelement <4 x float> %838, float %841, i64 1, !dbg !35
  %843 = fadd float %839, %841, !dbg !34
  %844 = extractelement <4 x float> %774, i64 2, !dbg !33
  %845 = call float @llvm.amdgcn.exp2.f32(float %844), !dbg !33
  %846 = insertelement <4 x float> %842, float %845, i64 2, !dbg !35
  %847 = fadd float %843, %845, !dbg !34
  %848 = extractelement <4 x float> %774, i64 3, !dbg !33
  %849 = call float @llvm.amdgcn.exp2.f32(float %848), !dbg !33
  %850 = insertelement <4 x float> %846, float %849, i64 3, !dbg !35
  %math.result58 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %850) #9, !dbg !35, !frisk.register_lifetimes !1
  %851 = fptrunc <4 x float> %math.result58 to <4 x half>, !dbg !35
  %math.packed59 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %851) #9, !dbg !34, !frisk.register_lifetimes !1
  %852 = fadd float %847, %849, !dbg !34
  %853 = bitcast float %819 to i32, !dbg !34
  %854 = call i32 @llvm.amdgcn.ds.bpermute(i32 %535, i32 %853), !dbg !34
  %855 = bitcast i32 %854 to float, !dbg !34
  %856 = fadd float %819, %855, !dbg !34
  %857 = bitcast float %856 to i32, !dbg !34
  %858 = call i32 @llvm.amdgcn.ds.bpermute(i32 %551, i32 %857), !dbg !34
  %859 = bitcast i32 %858 to float, !dbg !34
  %860 = fadd float %856, %859, !dbg !34
  %861 = fadd float %860, %780, !dbg !34
  %862 = bitcast float %852 to i32, !dbg !34
  %863 = call i32 @llvm.amdgcn.ds.bpermute(i32 %535, i32 %862), !dbg !34
  %864 = bitcast i32 %863 to float, !dbg !34
  %865 = fadd float %852, %864, !dbg !34
  %866 = bitcast float %865 to i32, !dbg !34
  %867 = call i32 @llvm.amdgcn.ds.bpermute(i32 %551, i32 %866), !dbg !34
  %868 = bitcast i32 %867 to float, !dbg !34
  %869 = fadd float %865, %868, !dbg !34
  %870 = fadd float %869, %786, !dbg !34
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %address.thread504 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local505 = shl nuw nsw i32 %address.thread504, 3, !dbg !16
  %address.local506 = and i32 %address.local505, 504, !dbg !16
  %address.local507 = zext nneg i32 %address.local506 to i64, !dbg !16
  %address.local508 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local507, !dbg !16
  %871 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local508, i32 0) #0, !dbg !16
  %872 = bitcast <4 x i32> %871 to <8 x half>, !dbg !16
  %address.local509 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local507, !dbg !16
  %address.local510 = getelementptr i8, ptr addrspace(3) %address.local509, i64 1024, !dbg !16
  %873 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local510, i32 0) #0, !dbg !16
  %874 = bitcast <4 x i32> %873 to <8 x half>, !dbg !16
  %875 = add nsw i64 %92, %8, !dbg !23
  %876 = shl i64 %875, 19, !dbg !23
  %877 = and i64 %876, 16252928, !dbg !23
  %878 = add i64 %877, %98, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %879 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %878, !dbg !23
  %880 = getelementptr i8, ptr addrspace(1) %879, i64 64, !dbg !23
  br label %881, !dbg !23

881:                                              ; preds = %884, %688
  %882 = phi i64 [ %910, %884 ], [ %27, %688 ], !dbg !23
  %883 = icmp samesign ult i64 %882, 128, !dbg !23
  br i1 %883, label %884, label %911, !dbg !23

884:                                              ; preds = %881
  %885 = shl nuw nsw i64 %882, 3, !dbg !23
  %886 = and i64 %885, 24, !dbg !23
  %887 = shl nuw nsw i64 %882, 3, !dbg !23
  %888 = and i64 %887, 512, !dbg !23
  %889 = shl nuw nsw i64 %882, 5, !dbg !23
  %890 = and i64 %889, 3968, !dbg !23
  %891 = getelementptr [2 x i8], ptr addrspace(1) %880, i64 %890, !dbg !23
  %892 = getelementptr [2 x i8], ptr addrspace(1) %891, i64 %886, !dbg !23
  %893 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %888, !dbg !23
  %894 = ptrtoint ptr addrspace(3) %893 to i64, !dbg !23
  %895 = trunc i64 %894 to i32, !dbg !23
  %896 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %895), !dbg !23
  %897 = zext i32 %896 to i64, !dbg !23
  %898 = inttoptr i64 %897 to ptr addrspace(3), !dbg !23
  %899 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !23
  %900 = ptrtoint ptr addrspace(1) %892 to i64, !dbg !23
  %901 = sub i64 %900, %899, !dbg !23
  %dma.buffer.offset831 = trunc i64 %901 to i32, !dbg !23
  %902 = trunc i64 %899 to i32, !dbg !23
  %903 = insertelement <4 x i32> poison, i32 %902, i64 0, !dbg !23
  %904 = lshr i64 %899, 32, !dbg !23
  %905 = trunc i64 %904 to i32, !dbg !23
  %906 = insertelement <4 x i32> %903, i32 %905, i64 1, !dbg !23
  %907 = insertelement <4 x i32> %906, i32 33554432, i64 2, !dbg !23
  %dma.buffer.rsrc832 = insertelement <4 x i32> %907, i32 131072, i64 3, !dbg !23
  %908 = ptrtoint ptr addrspace(3) %898 to i32, !dbg !23
  %dma.buffer.lds833 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %908), !dbg !23
  %909 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds833, i32 %dma.buffer.offset831, <4 x i32> %dma.buffer.rsrc832) #10, !dbg !23
  %910 = or disjoint i64 %27, 256, !dbg !23
  br label %881, !dbg !23

911:                                              ; preds = %881
  %912 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %913 = extractvalue [2 x <8 x float>] %53, 0, !dbg !36
  %914 = shufflevector <8 x float> %913, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %915 = insertelement <4 x float> poison, float %777, i64 0, !dbg !36
  %916 = shufflevector <4 x float> %915, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %917 = fmul <4 x float> %914, %916, !dbg !36
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %912, <4 x float> %917) #9, !dbg !16
  %919 = shufflevector <8 x half> %874, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %919, <4 x float> %918) #9, !dbg !16
  %921 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %922 = shufflevector <8 x float> %913, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %923 = fmul <4 x float> %922, %916, !dbg !36
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %921, <4 x float> %923) #9, !dbg !16
  %925 = shufflevector <8 x half> %874, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %925, <4 x float> %924) #9, !dbg !16
  %927 = extractvalue [2 x <8 x float>] %53, 1, !dbg !36
  %928 = shufflevector <8 x float> %927, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %929 = insertelement <4 x float> poison, float %783, i64 0, !dbg !36
  %930 = shufflevector <4 x float> %929, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %931 = fmul <4 x float> %928, %930, !dbg !36
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %912, <4 x float> %931) #9, !dbg !16
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %919, <4 x float> %932) #9, !dbg !16
  %934 = shufflevector <8 x float> %927, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %935 = fmul <4 x float> %934, %930, !dbg !36
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %921, <4 x float> %935) #9, !dbg !16
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %925, <4 x float> %936) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %address.thread511 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local512 = shl nuw nsw i32 %address.thread511, 3, !dbg !16
  %address.local513 = and i32 %address.local512, 504, !dbg !16
  %address.local514 = zext nneg i32 %address.local513 to i64, !dbg !16
  %address.local515 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local514, !dbg !16
  %938 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local515, i32 0) #0, !dbg !16
  %939 = bitcast <4 x i32> %938 to <8 x half>, !dbg !16
  %address.local516 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local514, !dbg !16
  %address.local517 = getelementptr i8, ptr addrspace(3) %address.local516, i64 1024, !dbg !16
  %940 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local517, i32 0) #0, !dbg !16
  %941 = bitcast <4 x i32> %940 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %942 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %878, !dbg !23
  %943 = getelementptr i8, ptr addrspace(1) %942, i64 128, !dbg !23
  br label %944, !dbg !23

944:                                              ; preds = %947, %911
  %945 = phi i64 [ %973, %947 ], [ %27, %911 ], !dbg !23
  %946 = icmp samesign ult i64 %945, 128, !dbg !23
  br i1 %946, label %947, label %974, !dbg !23

947:                                              ; preds = %944
  %948 = shl nuw nsw i64 %945, 3, !dbg !23
  %949 = and i64 %948, 24, !dbg !23
  %950 = shl nuw nsw i64 %945, 3, !dbg !23
  %951 = and i64 %950, 512, !dbg !23
  %952 = shl nuw nsw i64 %945, 5, !dbg !23
  %953 = and i64 %952, 3968, !dbg !23
  %954 = getelementptr [2 x i8], ptr addrspace(1) %943, i64 %953, !dbg !23
  %955 = getelementptr [2 x i8], ptr addrspace(1) %954, i64 %949, !dbg !23
  %956 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %951, !dbg !23
  %957 = ptrtoint ptr addrspace(3) %956 to i64, !dbg !23
  %958 = trunc i64 %957 to i32, !dbg !23
  %959 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %958), !dbg !23
  %960 = zext i32 %959 to i64, !dbg !23
  %961 = inttoptr i64 %960 to ptr addrspace(3), !dbg !23
  %962 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !23
  %963 = ptrtoint ptr addrspace(1) %955 to i64, !dbg !23
  %964 = sub i64 %963, %962, !dbg !23
  %dma.buffer.offset834 = trunc i64 %964 to i32, !dbg !23
  %965 = trunc i64 %962 to i32, !dbg !23
  %966 = insertelement <4 x i32> poison, i32 %965, i64 0, !dbg !23
  %967 = lshr i64 %962, 32, !dbg !23
  %968 = trunc i64 %967 to i32, !dbg !23
  %969 = insertelement <4 x i32> %966, i32 %968, i64 1, !dbg !23
  %970 = insertelement <4 x i32> %969, i32 33554432, i64 2, !dbg !23
  %dma.buffer.rsrc835 = insertelement <4 x i32> %970, i32 131072, i64 3, !dbg !23
  %971 = ptrtoint ptr addrspace(3) %961 to i32, !dbg !23
  %dma.buffer.lds836 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %971), !dbg !23
  %972 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds836, i32 %dma.buffer.offset834, <4 x i32> %dma.buffer.rsrc835) #10, !dbg !23
  %973 = or disjoint i64 %27, 256, !dbg !23
  br label %944, !dbg !23

974:                                              ; preds = %944
  %975 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %976 = extractvalue [2 x <8 x float>] %56, 0, !dbg !36
  %977 = shufflevector <8 x float> %976, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %978 = fmul <4 x float> %977, %916, !dbg !36
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %975, <4 x float> %978) #9, !dbg !16
  %980 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %980, <4 x float> %979) #9, !dbg !16
  %982 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %983 = shufflevector <8 x float> %976, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %984 = fmul <4 x float> %983, %916, !dbg !36
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %982, <4 x float> %984) #9, !dbg !16
  %986 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %986, <4 x float> %985) #9, !dbg !16
  %988 = extractvalue [2 x <8 x float>] %56, 1, !dbg !36
  %989 = shufflevector <8 x float> %988, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %990 = fmul <4 x float> %989, %930, !dbg !36
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %975, <4 x float> %990) #9, !dbg !16
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %980, <4 x float> %991) #9, !dbg !16
  %993 = shufflevector <8 x float> %988, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %994 = fmul <4 x float> %993, %930, !dbg !36
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %982, <4 x float> %994) #9, !dbg !16
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %986, <4 x float> %995) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %address.thread518 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local519 = shl nuw nsw i32 %address.thread518, 3, !dbg !16
  %address.local520 = and i32 %address.local519, 504, !dbg !16
  %address.local521 = zext nneg i32 %address.local520 to i64, !dbg !16
  %address.local522 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local521, !dbg !16
  %997 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local522, i32 0) #0, !dbg !16
  %998 = bitcast <4 x i32> %997 to <8 x half>, !dbg !16
  %address.local523 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local521, !dbg !16
  %address.local524 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !16
  %999 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local524, i32 0) #0, !dbg !16
  %1000 = bitcast <4 x i32> %999 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %1001 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %878, !dbg !23
  %1002 = getelementptr i8, ptr addrspace(1) %1001, i64 192, !dbg !23
  br label %1003, !dbg !23

1003:                                             ; preds = %1006, %974
  %1004 = phi i64 [ %1032, %1006 ], [ %27, %974 ], !dbg !23
  %1005 = icmp samesign ult i64 %1004, 128, !dbg !23
  br i1 %1005, label %1006, label %1033, !dbg !23

1006:                                             ; preds = %1003
  %1007 = shl nuw nsw i64 %1004, 3, !dbg !23
  %1008 = and i64 %1007, 24, !dbg !23
  %1009 = shl nuw nsw i64 %1004, 3, !dbg !23
  %1010 = and i64 %1009, 512, !dbg !23
  %1011 = shl nuw nsw i64 %1004, 5, !dbg !23
  %1012 = and i64 %1011, 3968, !dbg !23
  %1013 = getelementptr [2 x i8], ptr addrspace(1) %1002, i64 %1012, !dbg !23
  %1014 = getelementptr [2 x i8], ptr addrspace(1) %1013, i64 %1008, !dbg !23
  %1015 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %1010, !dbg !23
  %1016 = ptrtoint ptr addrspace(3) %1015 to i64, !dbg !23
  %1017 = trunc i64 %1016 to i32, !dbg !23
  %1018 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1017), !dbg !23
  %1019 = zext i32 %1018 to i64, !dbg !23
  %1020 = inttoptr i64 %1019 to ptr addrspace(3), !dbg !23
  %1021 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !23
  %1022 = ptrtoint ptr addrspace(1) %1014 to i64, !dbg !23
  %1023 = sub i64 %1022, %1021, !dbg !23
  %dma.buffer.offset837 = trunc i64 %1023 to i32, !dbg !23
  %1024 = trunc i64 %1021 to i32, !dbg !23
  %1025 = insertelement <4 x i32> poison, i32 %1024, i64 0, !dbg !23
  %1026 = lshr i64 %1021, 32, !dbg !23
  %1027 = trunc i64 %1026 to i32, !dbg !23
  %1028 = insertelement <4 x i32> %1025, i32 %1027, i64 1, !dbg !23
  %1029 = insertelement <4 x i32> %1028, i32 33554432, i64 2, !dbg !23
  %dma.buffer.rsrc838 = insertelement <4 x i32> %1029, i32 131072, i64 3, !dbg !23
  %1030 = ptrtoint ptr addrspace(3) %1020 to i32, !dbg !23
  %dma.buffer.lds839 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1030), !dbg !23
  %1031 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds839, i32 %dma.buffer.offset837, <4 x i32> %dma.buffer.rsrc838) #10, !dbg !23
  %1032 = or disjoint i64 %27, 256, !dbg !23
  br label %1003, !dbg !23

1033:                                             ; preds = %1003
  %1034 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1035 = extractvalue [2 x <8 x float>] %57, 0, !dbg !36
  %1036 = shufflevector <8 x float> %1035, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1037 = fmul <4 x float> %1036, %916, !dbg !36
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1034, <4 x float> %1037) #9, !dbg !16
  %1039 = shufflevector <8 x half> %1000, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %1039, <4 x float> %1038) #9, !dbg !16
  %1041 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1042 = shufflevector <8 x float> %1035, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1043 = fmul <4 x float> %1042, %916, !dbg !36
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1041, <4 x float> %1043) #9, !dbg !16
  %1045 = shufflevector <8 x half> %1000, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %1045, <4 x float> %1044) #9, !dbg !16
  %1047 = extractvalue [2 x <8 x float>] %57, 1, !dbg !36
  %1048 = shufflevector <8 x float> %1047, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1049 = fmul <4 x float> %1048, %930, !dbg !36
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %1034, <4 x float> %1049) #9, !dbg !16
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %1039, <4 x float> %1050) #9, !dbg !16
  %1052 = shufflevector <8 x float> %1047, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1053 = fmul <4 x float> %1052, %930, !dbg !36
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %1041, <4 x float> %1053) #9, !dbg !16
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %1045, <4 x float> %1054) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %address.thread525 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local526 = shl nuw nsw i32 %address.thread525, 3, !dbg !16
  %address.local527 = and i32 %address.local526, 504, !dbg !16
  %address.local528 = zext nneg i32 %address.local527 to i64, !dbg !16
  %address.local529 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local528, !dbg !16
  %1056 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local529, i32 0) #0, !dbg !16
  %1057 = bitcast <4 x i32> %1056 to <8 x half>, !dbg !16
  %address.local530 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local528, !dbg !16
  %address.local531 = getelementptr i8, ptr addrspace(3) %address.local530, i64 1024, !dbg !16
  %1058 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local531, i32 0) #0, !dbg !16
  %1059 = bitcast <4 x i32> %1058 to <8 x half>, !dbg !16
  %1060 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1061 = extractvalue [2 x <8 x float>] %58, 0, !dbg !36
  %1062 = shufflevector <8 x float> %1061, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1063 = fmul <4 x float> %1062, %916, !dbg !36
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1060, <4 x float> %1063) #9, !dbg !16
  %1065 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %1065, <4 x float> %1064) #9, !dbg !16
  %1067 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1068 = shufflevector <8 x float> %1061, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1069 = fmul <4 x float> %1068, %916, !dbg !36
  %1070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1067, <4 x float> %1069) #9, !dbg !16
  %1071 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %1071, <4 x float> %1070) #9, !dbg !16
  %1073 = extractvalue [2 x <8 x float>] %58, 1, !dbg !36
  %1074 = shufflevector <8 x float> %1073, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1075 = fmul <4 x float> %1074, %930, !dbg !36
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %1060, <4 x float> %1075) #9, !dbg !16
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %1065, <4 x float> %1076) #9, !dbg !16
  %1078 = shufflevector <8 x float> %1073, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1079 = fmul <4 x float> %1078, %930, !dbg !36
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %1067, <4 x float> %1079) #9, !dbg !16
  %1081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %1071, <4 x float> %1080) #9, !dbg !16
  %1082 = shufflevector <4 x float> %920, <4 x float> %926, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1083 = insertvalue [2 x <8 x float>] poison, <8 x float> %1082, 0, !dbg !7
  %1084 = shufflevector <4 x float> %933, <4 x float> %937, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1085 = insertvalue [2 x <8 x float>] %1083, <8 x float> %1084, 1, !dbg !7
  %1086 = insertelement <1 x float> poison, float %861, i64 0, !dbg !7
  %1087 = insertvalue [2 x <1 x float>] poison, <1 x float> %1086, 0, !dbg !7
  %1088 = insertelement <1 x float> poison, float %870, i64 0, !dbg !7
  %1089 = insertvalue [2 x <1 x float>] %1087, <1 x float> %1088, 1, !dbg !7
  %1090 = insertvalue [2 x <1 x float>] poison, <1 x float> %749, 0, !dbg !7
  %1091 = insertvalue [2 x <1 x float>] %1090, <1 x float> %768, 1, !dbg !7
  %1092 = shufflevector <4 x float> %981, <4 x float> %987, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1093 = insertvalue [2 x <8 x float>] poison, <8 x float> %1092, 0, !dbg !7
  %1094 = shufflevector <4 x float> %992, <4 x float> %996, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1095 = insertvalue [2 x <8 x float>] %1093, <8 x float> %1094, 1, !dbg !7
  %1096 = shufflevector <4 x float> %1040, <4 x float> %1046, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1097 = insertvalue [2 x <8 x float>] poison, <8 x float> %1096, 0, !dbg !7
  %1098 = shufflevector <4 x float> %1051, <4 x float> %1055, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1099 = insertvalue [2 x <8 x float>] %1097, <8 x float> %1098, 1, !dbg !7
  %1100 = shufflevector <4 x float> %1066, <4 x float> %1072, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1101 = insertvalue [2 x <8 x float>] poison, <8 x float> %1100, 0, !dbg !7
  %1102 = shufflevector <4 x float> %1077, <4 x float> %1081, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1103 = insertvalue [2 x <8 x float>] %1101, <8 x float> %1102, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1104 = add nuw nsw i64 %52, 1, !dbg !7
  br label %51, !dbg !7

1105:                                             ; preds = %51
  %1106 = extractvalue [2 x <8 x float>] %58, 0, !dbg !7
  %1107 = extractelement <8 x float> %1106, i64 0, !dbg !7
  %1108 = extractelement <8 x float> %1106, i64 1, !dbg !7
  %1109 = extractelement <8 x float> %1106, i64 2, !dbg !7
  %1110 = extractelement <8 x float> %1106, i64 3, !dbg !7
  %1111 = select i1 %.not, float %1108, float %1107, !dbg !7
  %1112 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1113 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1112), !dbg !7
  %1114 = add nuw nsw i32 %1113, 64, !dbg !7
  %1115 = xor i32 %1113, 16, !dbg !7
  %1116 = and i32 %1114, 192, !dbg !7
  %1117 = icmp samesign ult i32 %1115, %1116, !dbg !7
  %1118 = select i1 %1117, i32 %1115, i32 %1113, !dbg !7
  %1119 = shl nuw nsw i32 %1118, 2, !dbg !7
  %1120 = bitcast float %1111 to i32, !dbg !7
  %1121 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1120), !dbg !7
  %1122 = bitcast i32 %1121 to float, !dbg !7
  %1123 = select i1 %.not, float %1107, float %1122, !dbg !7
  %1124 = select i1 %.not, float %1122, float %1108, !dbg !7
  %1125 = select i1 %.not, float %1110, float %1109, !dbg !7
  %1126 = bitcast float %1125 to i32, !dbg !7
  %1127 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1126), !dbg !7
  %1128 = bitcast i32 %1127 to float, !dbg !7
  %1129 = select i1 %.not, float %1109, float %1128, !dbg !7
  %1130 = select i1 %.not, float %1128, float %1110, !dbg !7
  %1131 = select i1 %.not2, float %1129, float %1123, !dbg !7
  %1132 = xor i32 %1113, 32, !dbg !7
  %1133 = icmp samesign ult i32 %1132, %1116, !dbg !7
  %1134 = select i1 %1133, i32 %1132, i32 %1113, !dbg !7
  %1135 = shl nuw nsw i32 %1134, 2, !dbg !7
  %1136 = bitcast float %1131 to i32, !dbg !7
  %1137 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1136), !dbg !7
  %1138 = bitcast i32 %1137 to float, !dbg !7
  %1139 = select i1 %.not2, float %1123, float %1138, !dbg !7
  %1140 = select i1 %.not2, float %1138, float %1129, !dbg !7
  %1141 = select i1 %.not2, float %1130, float %1124, !dbg !7
  %1142 = bitcast float %1141 to i32, !dbg !7
  %1143 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1142), !dbg !7
  %1144 = bitcast i32 %1143 to float, !dbg !7
  %1145 = select i1 %.not2, float %1124, float %1144, !dbg !7
  %1146 = select i1 %.not2, float %1144, float %1130, !dbg !7
  %1147 = extractvalue [2 x <8 x float>] %57, 0, !dbg !7
  %1148 = extractelement <8 x float> %1147, i64 0, !dbg !7
  %1149 = extractelement <8 x float> %1147, i64 1, !dbg !7
  %1150 = extractelement <8 x float> %1147, i64 2, !dbg !7
  %1151 = extractelement <8 x float> %1147, i64 3, !dbg !7
  %1152 = select i1 %.not, float %1149, float %1148, !dbg !7
  %1153 = bitcast float %1152 to i32, !dbg !7
  %1154 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1153), !dbg !7
  %1155 = bitcast i32 %1154 to float, !dbg !7
  %1156 = select i1 %.not, float %1148, float %1155, !dbg !7
  %1157 = select i1 %.not, float %1155, float %1149, !dbg !7
  %1158 = select i1 %.not, float %1151, float %1150, !dbg !7
  %1159 = bitcast float %1158 to i32, !dbg !7
  %1160 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1159), !dbg !7
  %1161 = bitcast i32 %1160 to float, !dbg !7
  %1162 = select i1 %.not, float %1150, float %1161, !dbg !7
  %1163 = select i1 %.not, float %1161, float %1151, !dbg !7
  %1164 = select i1 %.not2, float %1162, float %1156, !dbg !7
  %1165 = bitcast float %1164 to i32, !dbg !7
  %1166 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1165), !dbg !7
  %1167 = bitcast i32 %1166 to float, !dbg !7
  %1168 = select i1 %.not2, float %1156, float %1167, !dbg !7
  %1169 = select i1 %.not2, float %1167, float %1162, !dbg !7
  %1170 = select i1 %.not2, float %1163, float %1157, !dbg !7
  %1171 = bitcast float %1170 to i32, !dbg !7
  %1172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1171), !dbg !7
  %1173 = bitcast i32 %1172 to float, !dbg !7
  %1174 = select i1 %.not2, float %1157, float %1173, !dbg !7
  %1175 = select i1 %.not2, float %1173, float %1163, !dbg !7
  %1176 = extractvalue [2 x <8 x float>] %56, 0, !dbg !7
  %1177 = extractelement <8 x float> %1176, i64 0, !dbg !7
  %1178 = extractelement <8 x float> %1176, i64 1, !dbg !7
  %1179 = extractelement <8 x float> %1176, i64 2, !dbg !7
  %1180 = extractelement <8 x float> %1176, i64 3, !dbg !7
  %1181 = select i1 %.not, float %1178, float %1177, !dbg !7
  %1182 = bitcast float %1181 to i32, !dbg !7
  %1183 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1182), !dbg !7
  %1184 = bitcast i32 %1183 to float, !dbg !7
  %1185 = select i1 %.not, float %1177, float %1184, !dbg !7
  %1186 = select i1 %.not, float %1184, float %1178, !dbg !7
  %1187 = select i1 %.not, float %1180, float %1179, !dbg !7
  %1188 = bitcast float %1187 to i32, !dbg !7
  %1189 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1188), !dbg !7
  %1190 = bitcast i32 %1189 to float, !dbg !7
  %1191 = select i1 %.not, float %1179, float %1190, !dbg !7
  %1192 = select i1 %.not, float %1190, float %1180, !dbg !7
  %1193 = select i1 %.not2, float %1191, float %1185, !dbg !7
  %1194 = bitcast float %1193 to i32, !dbg !7
  %1195 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1194), !dbg !7
  %1196 = bitcast i32 %1195 to float, !dbg !7
  %1197 = select i1 %.not2, float %1185, float %1196, !dbg !7
  %1198 = select i1 %.not2, float %1196, float %1191, !dbg !7
  %1199 = select i1 %.not2, float %1192, float %1186, !dbg !7
  %1200 = bitcast float %1199 to i32, !dbg !7
  %1201 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1200), !dbg !7
  %1202 = bitcast i32 %1201 to float, !dbg !7
  %1203 = select i1 %.not2, float %1186, float %1202, !dbg !7
  %1204 = select i1 %.not2, float %1202, float %1192, !dbg !7
  %1205 = extractvalue [2 x <8 x float>] %53, 0, !dbg !7
  %1206 = extractelement <8 x float> %1205, i64 0, !dbg !7
  %1207 = extractelement <8 x float> %1205, i64 1, !dbg !7
  %1208 = extractelement <8 x float> %1205, i64 2, !dbg !7
  %1209 = extractelement <8 x float> %1205, i64 3, !dbg !7
  %1210 = select i1 %.not, float %1207, float %1206, !dbg !7
  %1211 = bitcast float %1210 to i32, !dbg !7
  %1212 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1211), !dbg !7
  %1213 = bitcast i32 %1212 to float, !dbg !7
  %1214 = select i1 %.not, float %1206, float %1213, !dbg !7
  %1215 = select i1 %.not, float %1213, float %1207, !dbg !7
  %1216 = select i1 %.not, float %1209, float %1208, !dbg !7
  %1217 = bitcast float %1216 to i32, !dbg !7
  %1218 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1217), !dbg !7
  %1219 = bitcast i32 %1218 to float, !dbg !7
  %1220 = select i1 %.not, float %1208, float %1219, !dbg !7
  %1221 = select i1 %.not, float %1219, float %1209, !dbg !7
  %1222 = select i1 %.not2, float %1220, float %1214, !dbg !7
  %1223 = bitcast float %1222 to i32, !dbg !7
  %1224 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1223), !dbg !7
  %1225 = select i1 %.not2, float %1221, float %1215, !dbg !7
  %1226 = bitcast float %1225 to i32, !dbg !7
  %1227 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1226), !dbg !7
  %1228 = extractvalue [2 x <1 x float>] %54, 0, !dbg !19
  %1229 = shufflevector <1 x float> %1228, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1230 = extractelement <8 x float> %1106, i64 4, !dbg !7
  %1231 = extractelement <8 x float> %1106, i64 5, !dbg !7
  %1232 = extractelement <8 x float> %1106, i64 6, !dbg !7
  %1233 = extractelement <8 x float> %1106, i64 7, !dbg !7
  %1234 = select i1 %.not, float %1231, float %1230, !dbg !7
  %1235 = bitcast float %1234 to i32, !dbg !7
  %1236 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1235), !dbg !7
  %1237 = bitcast i32 %1236 to float, !dbg !7
  %1238 = select i1 %.not, float %1230, float %1237, !dbg !7
  %1239 = select i1 %.not, float %1237, float %1231, !dbg !7
  %1240 = select i1 %.not, float %1233, float %1232, !dbg !7
  %1241 = bitcast float %1240 to i32, !dbg !7
  %1242 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1241), !dbg !7
  %1243 = bitcast i32 %1242 to float, !dbg !7
  %1244 = select i1 %.not, float %1232, float %1243, !dbg !7
  %1245 = select i1 %.not, float %1243, float %1233, !dbg !7
  %1246 = select i1 %.not2, float %1244, float %1238, !dbg !7
  %1247 = bitcast float %1246 to i32, !dbg !7
  %1248 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1247), !dbg !7
  %1249 = bitcast i32 %1248 to float, !dbg !7
  %1250 = select i1 %.not2, float %1238, float %1249, !dbg !7
  %1251 = select i1 %.not2, float %1249, float %1244, !dbg !7
  %1252 = select i1 %.not2, float %1245, float %1239, !dbg !7
  %1253 = bitcast float %1252 to i32, !dbg !7
  %1254 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1253), !dbg !7
  %1255 = bitcast i32 %1254 to float, !dbg !7
  %1256 = select i1 %.not2, float %1239, float %1255, !dbg !7
  %1257 = select i1 %.not2, float %1255, float %1245, !dbg !7
  %1258 = extractelement <8 x float> %1147, i64 4, !dbg !7
  %1259 = extractelement <8 x float> %1147, i64 5, !dbg !7
  %1260 = extractelement <8 x float> %1147, i64 6, !dbg !7
  %1261 = extractelement <8 x float> %1147, i64 7, !dbg !7
  %1262 = select i1 %.not, float %1259, float %1258, !dbg !7
  %1263 = bitcast float %1262 to i32, !dbg !7
  %1264 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1263), !dbg !7
  %1265 = bitcast i32 %1264 to float, !dbg !7
  %1266 = select i1 %.not, float %1258, float %1265, !dbg !7
  %1267 = select i1 %.not, float %1265, float %1259, !dbg !7
  %1268 = select i1 %.not, float %1261, float %1260, !dbg !7
  %1269 = bitcast float %1268 to i32, !dbg !7
  %1270 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1269), !dbg !7
  %1271 = bitcast i32 %1270 to float, !dbg !7
  %1272 = select i1 %.not, float %1260, float %1271, !dbg !7
  %1273 = select i1 %.not, float %1271, float %1261, !dbg !7
  %1274 = select i1 %.not2, float %1272, float %1266, !dbg !7
  %1275 = bitcast float %1274 to i32, !dbg !7
  %1276 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1275), !dbg !7
  %1277 = bitcast i32 %1276 to float, !dbg !7
  %1278 = select i1 %.not2, float %1266, float %1277, !dbg !7
  %1279 = select i1 %.not2, float %1277, float %1272, !dbg !7
  %1280 = select i1 %.not2, float %1273, float %1267, !dbg !7
  %1281 = bitcast float %1280 to i32, !dbg !7
  %1282 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1281), !dbg !7
  %1283 = bitcast i32 %1282 to float, !dbg !7
  %1284 = select i1 %.not2, float %1267, float %1283, !dbg !7
  %1285 = select i1 %.not2, float %1283, float %1273, !dbg !7
  %1286 = extractelement <8 x float> %1176, i64 4, !dbg !7
  %1287 = extractelement <8 x float> %1176, i64 5, !dbg !7
  %1288 = extractelement <8 x float> %1176, i64 6, !dbg !7
  %1289 = extractelement <8 x float> %1176, i64 7, !dbg !7
  %1290 = select i1 %.not, float %1287, float %1286, !dbg !7
  %1291 = bitcast float %1290 to i32, !dbg !7
  %1292 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1291), !dbg !7
  %1293 = bitcast i32 %1292 to float, !dbg !7
  %1294 = select i1 %.not, float %1286, float %1293, !dbg !7
  %1295 = select i1 %.not, float %1293, float %1287, !dbg !7
  %1296 = select i1 %.not, float %1289, float %1288, !dbg !7
  %1297 = bitcast float %1296 to i32, !dbg !7
  %1298 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1297), !dbg !7
  %1299 = bitcast i32 %1298 to float, !dbg !7
  %1300 = select i1 %.not, float %1288, float %1299, !dbg !7
  %1301 = select i1 %.not, float %1299, float %1289, !dbg !7
  %1302 = select i1 %.not2, float %1300, float %1294, !dbg !7
  %1303 = bitcast float %1302 to i32, !dbg !7
  %1304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1303), !dbg !7
  %1305 = bitcast i32 %1304 to float, !dbg !7
  %1306 = select i1 %.not2, float %1294, float %1305, !dbg !7
  %1307 = select i1 %.not2, float %1305, float %1300, !dbg !7
  %1308 = select i1 %.not2, float %1301, float %1295, !dbg !7
  %1309 = bitcast float %1308 to i32, !dbg !7
  %1310 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1309), !dbg !7
  %1311 = bitcast i32 %1310 to float, !dbg !7
  %1312 = select i1 %.not2, float %1295, float %1311, !dbg !7
  %1313 = select i1 %.not2, float %1311, float %1301, !dbg !7
  %1314 = extractelement <8 x float> %1205, i64 4, !dbg !7
  %1315 = extractelement <8 x float> %1205, i64 5, !dbg !7
  %1316 = extractelement <8 x float> %1205, i64 6, !dbg !7
  %1317 = extractelement <8 x float> %1205, i64 7, !dbg !7
  %1318 = select i1 %.not, float %1315, float %1314, !dbg !7
  %1319 = bitcast float %1318 to i32, !dbg !7
  %1320 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1319), !dbg !7
  %1321 = bitcast i32 %1320 to float, !dbg !7
  %1322 = select i1 %.not, float %1314, float %1321, !dbg !7
  %1323 = select i1 %.not, float %1321, float %1315, !dbg !7
  %1324 = select i1 %.not, float %1317, float %1316, !dbg !7
  %1325 = bitcast float %1324 to i32, !dbg !7
  %1326 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1325), !dbg !7
  %1327 = bitcast i32 %1326 to float, !dbg !7
  %1328 = select i1 %.not, float %1316, float %1327, !dbg !7
  %1329 = select i1 %.not, float %1327, float %1317, !dbg !7
  %1330 = select i1 %.not2, float %1328, float %1322, !dbg !7
  %1331 = bitcast float %1330 to i32, !dbg !7
  %1332 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1331), !dbg !7
  %1333 = select i1 %.not2, float %1329, float %1323, !dbg !7
  %1334 = bitcast float %1333 to i32, !dbg !7
  %1335 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1334), !dbg !7
  %1336 = extractvalue [2 x <8 x float>] %58, 1, !dbg !7
  %1337 = extractelement <8 x float> %1336, i64 0, !dbg !7
  %1338 = extractelement <8 x float> %1336, i64 1, !dbg !7
  %1339 = extractelement <8 x float> %1336, i64 2, !dbg !7
  %1340 = extractelement <8 x float> %1336, i64 3, !dbg !7
  %1341 = select i1 %.not, float %1338, float %1337, !dbg !7
  %1342 = bitcast float %1341 to i32, !dbg !7
  %1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1342), !dbg !7
  %1344 = bitcast i32 %1343 to float, !dbg !7
  %1345 = select i1 %.not, float %1337, float %1344, !dbg !7
  %1346 = select i1 %.not, float %1344, float %1338, !dbg !7
  %1347 = select i1 %.not, float %1340, float %1339, !dbg !7
  %1348 = bitcast float %1347 to i32, !dbg !7
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1348), !dbg !7
  %1350 = bitcast i32 %1349 to float, !dbg !7
  %1351 = select i1 %.not, float %1339, float %1350, !dbg !7
  %1352 = select i1 %.not, float %1350, float %1340, !dbg !7
  %1353 = select i1 %.not2, float %1351, float %1345, !dbg !7
  %1354 = bitcast float %1353 to i32, !dbg !7
  %1355 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1354), !dbg !7
  %1356 = bitcast i32 %1355 to float, !dbg !7
  %1357 = select i1 %.not2, float %1345, float %1356, !dbg !7
  %1358 = select i1 %.not2, float %1356, float %1351, !dbg !7
  %1359 = select i1 %.not2, float %1352, float %1346, !dbg !7
  %1360 = bitcast float %1359 to i32, !dbg !7
  %1361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1360), !dbg !7
  %1362 = bitcast i32 %1361 to float, !dbg !7
  %1363 = select i1 %.not2, float %1346, float %1362, !dbg !7
  %1364 = select i1 %.not2, float %1362, float %1352, !dbg !7
  %1365 = extractvalue [2 x <8 x float>] %57, 1, !dbg !7
  %1366 = extractelement <8 x float> %1365, i64 0, !dbg !7
  %1367 = extractelement <8 x float> %1365, i64 1, !dbg !7
  %1368 = extractelement <8 x float> %1365, i64 2, !dbg !7
  %1369 = extractelement <8 x float> %1365, i64 3, !dbg !7
  %1370 = select i1 %.not, float %1367, float %1366, !dbg !7
  %1371 = bitcast float %1370 to i32, !dbg !7
  %1372 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1371), !dbg !7
  %1373 = bitcast i32 %1372 to float, !dbg !7
  %1374 = select i1 %.not, float %1366, float %1373, !dbg !7
  %1375 = select i1 %.not, float %1373, float %1367, !dbg !7
  %1376 = select i1 %.not, float %1369, float %1368, !dbg !7
  %1377 = bitcast float %1376 to i32, !dbg !7
  %1378 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1377), !dbg !7
  %1379 = bitcast i32 %1378 to float, !dbg !7
  %1380 = select i1 %.not, float %1368, float %1379, !dbg !7
  %1381 = select i1 %.not, float %1379, float %1369, !dbg !7
  %1382 = select i1 %.not2, float %1380, float %1374, !dbg !7
  %1383 = bitcast float %1382 to i32, !dbg !7
  %1384 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1383), !dbg !7
  %1385 = bitcast i32 %1384 to float, !dbg !7
  %1386 = select i1 %.not2, float %1374, float %1385, !dbg !7
  %1387 = select i1 %.not2, float %1385, float %1380, !dbg !7
  %1388 = select i1 %.not2, float %1381, float %1375, !dbg !7
  %1389 = bitcast float %1388 to i32, !dbg !7
  %1390 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1389), !dbg !7
  %1391 = bitcast i32 %1390 to float, !dbg !7
  %1392 = select i1 %.not2, float %1375, float %1391, !dbg !7
  %1393 = select i1 %.not2, float %1391, float %1381, !dbg !7
  %1394 = extractvalue [2 x <8 x float>] %56, 1, !dbg !7
  %1395 = extractelement <8 x float> %1394, i64 0, !dbg !7
  %1396 = extractelement <8 x float> %1394, i64 1, !dbg !7
  %1397 = extractelement <8 x float> %1394, i64 2, !dbg !7
  %1398 = extractelement <8 x float> %1394, i64 3, !dbg !7
  %1399 = select i1 %.not, float %1396, float %1395, !dbg !7
  %1400 = bitcast float %1399 to i32, !dbg !7
  %1401 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1400), !dbg !7
  %1402 = bitcast i32 %1401 to float, !dbg !7
  %1403 = select i1 %.not, float %1395, float %1402, !dbg !7
  %1404 = select i1 %.not, float %1402, float %1396, !dbg !7
  %1405 = select i1 %.not, float %1398, float %1397, !dbg !7
  %1406 = bitcast float %1405 to i32, !dbg !7
  %1407 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1406), !dbg !7
  %1408 = bitcast i32 %1407 to float, !dbg !7
  %1409 = select i1 %.not, float %1397, float %1408, !dbg !7
  %1410 = select i1 %.not, float %1408, float %1398, !dbg !7
  %1411 = select i1 %.not2, float %1409, float %1403, !dbg !7
  %1412 = bitcast float %1411 to i32, !dbg !7
  %1413 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1412), !dbg !7
  %1414 = bitcast i32 %1413 to float, !dbg !7
  %1415 = select i1 %.not2, float %1403, float %1414, !dbg !7
  %1416 = select i1 %.not2, float %1414, float %1409, !dbg !7
  %1417 = select i1 %.not2, float %1410, float %1404, !dbg !7
  %1418 = bitcast float %1417 to i32, !dbg !7
  %1419 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1418), !dbg !7
  %1420 = bitcast i32 %1419 to float, !dbg !7
  %1421 = select i1 %.not2, float %1404, float %1420, !dbg !7
  %1422 = select i1 %.not2, float %1420, float %1410, !dbg !7
  %1423 = extractvalue [2 x <8 x float>] %53, 1, !dbg !7
  %1424 = extractelement <8 x float> %1423, i64 0, !dbg !7
  %1425 = extractelement <8 x float> %1423, i64 1, !dbg !7
  %1426 = extractelement <8 x float> %1423, i64 2, !dbg !7
  %1427 = extractelement <8 x float> %1423, i64 3, !dbg !7
  %1428 = select i1 %.not, float %1425, float %1424, !dbg !7
  %1429 = bitcast float %1428 to i32, !dbg !7
  %1430 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1429), !dbg !7
  %1431 = bitcast i32 %1430 to float, !dbg !7
  %1432 = select i1 %.not, float %1424, float %1431, !dbg !7
  %1433 = select i1 %.not, float %1431, float %1425, !dbg !7
  %1434 = select i1 %.not, float %1427, float %1426, !dbg !7
  %1435 = bitcast float %1434 to i32, !dbg !7
  %1436 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1435), !dbg !7
  %1437 = bitcast i32 %1436 to float, !dbg !7
  %1438 = select i1 %.not, float %1426, float %1437, !dbg !7
  %1439 = select i1 %.not, float %1437, float %1427, !dbg !7
  %1440 = select i1 %.not2, float %1438, float %1432, !dbg !7
  %1441 = bitcast float %1440 to i32, !dbg !7
  %1442 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1441), !dbg !7
  %1443 = select i1 %.not2, float %1439, float %1433, !dbg !7
  %1444 = bitcast float %1443 to i32, !dbg !7
  %1445 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1444), !dbg !7
  %1446 = extractvalue [2 x <1 x float>] %54, 1, !dbg !19
  %1447 = shufflevector <1 x float> %1446, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1448 = extractelement <8 x float> %1336, i64 4, !dbg !7
  %1449 = extractelement <8 x float> %1336, i64 5, !dbg !7
  %1450 = extractelement <8 x float> %1336, i64 6, !dbg !7
  %1451 = extractelement <8 x float> %1336, i64 7, !dbg !7
  %1452 = select i1 %.not, float %1449, float %1448, !dbg !7
  %1453 = bitcast float %1452 to i32, !dbg !7
  %1454 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1453), !dbg !7
  %1455 = bitcast i32 %1454 to float, !dbg !7
  %1456 = select i1 %.not, float %1448, float %1455, !dbg !7
  %1457 = select i1 %.not, float %1455, float %1449, !dbg !7
  %1458 = select i1 %.not, float %1451, float %1450, !dbg !7
  %1459 = bitcast float %1458 to i32, !dbg !7
  %1460 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1459), !dbg !7
  %1461 = bitcast i32 %1460 to float, !dbg !7
  %1462 = select i1 %.not, float %1450, float %1461, !dbg !7
  %1463 = select i1 %.not, float %1461, float %1451, !dbg !7
  %1464 = select i1 %.not2, float %1462, float %1456, !dbg !7
  %1465 = bitcast float %1464 to i32, !dbg !7
  %1466 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1465), !dbg !7
  %1467 = bitcast i32 %1466 to float, !dbg !7
  %1468 = select i1 %.not2, float %1456, float %1467, !dbg !7
  %1469 = select i1 %.not2, float %1467, float %1462, !dbg !7
  %1470 = select i1 %.not2, float %1463, float %1457, !dbg !7
  %1471 = bitcast float %1470 to i32, !dbg !7
  %1472 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1471), !dbg !7
  %1473 = bitcast i32 %1472 to float, !dbg !7
  %1474 = select i1 %.not2, float %1457, float %1473, !dbg !7
  %1475 = select i1 %.not2, float %1473, float %1463, !dbg !7
  %1476 = extractelement <8 x float> %1365, i64 4, !dbg !7
  %1477 = extractelement <8 x float> %1365, i64 5, !dbg !7
  %1478 = extractelement <8 x float> %1365, i64 6, !dbg !7
  %1479 = extractelement <8 x float> %1365, i64 7, !dbg !7
  %1480 = select i1 %.not, float %1477, float %1476, !dbg !7
  %1481 = bitcast float %1480 to i32, !dbg !7
  %1482 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1481), !dbg !7
  %1483 = bitcast i32 %1482 to float, !dbg !7
  %1484 = select i1 %.not, float %1476, float %1483, !dbg !7
  %1485 = select i1 %.not, float %1483, float %1477, !dbg !7
  %1486 = select i1 %.not, float %1479, float %1478, !dbg !7
  %1487 = bitcast float %1486 to i32, !dbg !7
  %1488 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1487), !dbg !7
  %1489 = bitcast i32 %1488 to float, !dbg !7
  %1490 = select i1 %.not, float %1478, float %1489, !dbg !7
  %1491 = select i1 %.not, float %1489, float %1479, !dbg !7
  %1492 = select i1 %.not2, float %1490, float %1484, !dbg !7
  %1493 = bitcast float %1492 to i32, !dbg !7
  %1494 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1493), !dbg !7
  %1495 = bitcast i32 %1494 to float, !dbg !7
  %1496 = select i1 %.not2, float %1484, float %1495, !dbg !7
  %1497 = select i1 %.not2, float %1495, float %1490, !dbg !7
  %1498 = select i1 %.not2, float %1491, float %1485, !dbg !7
  %1499 = bitcast float %1498 to i32, !dbg !7
  %1500 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1499), !dbg !7
  %1501 = bitcast i32 %1500 to float, !dbg !7
  %1502 = select i1 %.not2, float %1485, float %1501, !dbg !7
  %1503 = select i1 %.not2, float %1501, float %1491, !dbg !7
  %1504 = extractelement <8 x float> %1394, i64 4, !dbg !7
  %1505 = extractelement <8 x float> %1394, i64 5, !dbg !7
  %1506 = extractelement <8 x float> %1394, i64 6, !dbg !7
  %1507 = extractelement <8 x float> %1394, i64 7, !dbg !7
  %1508 = select i1 %.not, float %1505, float %1504, !dbg !7
  %1509 = bitcast float %1508 to i32, !dbg !7
  %1510 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1509), !dbg !7
  %1511 = bitcast i32 %1510 to float, !dbg !7
  %1512 = select i1 %.not, float %1504, float %1511, !dbg !7
  %1513 = select i1 %.not, float %1511, float %1505, !dbg !7
  %1514 = select i1 %.not, float %1507, float %1506, !dbg !7
  %1515 = bitcast float %1514 to i32, !dbg !7
  %1516 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1515), !dbg !7
  %1517 = bitcast i32 %1516 to float, !dbg !7
  %1518 = select i1 %.not, float %1506, float %1517, !dbg !7
  %1519 = select i1 %.not, float %1517, float %1507, !dbg !7
  %1520 = select i1 %.not2, float %1518, float %1512, !dbg !7
  %1521 = bitcast float %1520 to i32, !dbg !7
  %1522 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1521), !dbg !7
  %1523 = bitcast i32 %1522 to float, !dbg !7
  %1524 = select i1 %.not2, float %1512, float %1523, !dbg !7
  %1525 = select i1 %.not2, float %1523, float %1518, !dbg !7
  %1526 = select i1 %.not2, float %1519, float %1513, !dbg !7
  %1527 = bitcast float %1526 to i32, !dbg !7
  %1528 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1527), !dbg !7
  %1529 = bitcast i32 %1528 to float, !dbg !7
  %1530 = select i1 %.not2, float %1513, float %1529, !dbg !7
  %1531 = select i1 %.not2, float %1529, float %1519, !dbg !7
  %1532 = extractelement <8 x float> %1423, i64 4, !dbg !7
  %1533 = extractelement <8 x float> %1423, i64 5, !dbg !7
  %1534 = extractelement <8 x float> %1423, i64 6, !dbg !7
  %1535 = extractelement <8 x float> %1423, i64 7, !dbg !7
  %1536 = select i1 %.not, float %1533, float %1532, !dbg !7
  %1537 = bitcast float %1536 to i32, !dbg !7
  %1538 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1537), !dbg !7
  %1539 = bitcast i32 %1538 to float, !dbg !7
  %1540 = select i1 %.not, float %1532, float %1539, !dbg !7
  %1541 = select i1 %.not, float %1539, float %1533, !dbg !7
  %1542 = select i1 %.not, float %1535, float %1534, !dbg !7
  %1543 = bitcast float %1542 to i32, !dbg !7
  %1544 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1119, i32 %1543), !dbg !7
  %1545 = bitcast i32 %1544 to float, !dbg !7
  %1546 = select i1 %.not, float %1534, float %1545, !dbg !7
  %1547 = select i1 %.not, float %1545, float %1535, !dbg !7
  %1548 = select i1 %.not2, float %1546, float %1540, !dbg !7
  %1549 = bitcast float %1548 to i32, !dbg !7
  %1550 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1549), !dbg !7
  %1551 = select i1 %.not2, float %1547, float %1541, !dbg !7
  %1552 = bitcast float %1551 to i32, !dbg !7
  %1553 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1552), !dbg !7
  br i1 true, label %1554, label %1616, !dbg !21

1554:                                             ; preds = %1105
  %1555 = bitcast i32 %1550 to float, !dbg !7
  %1556 = select i1 %.not2, float %1540, float %1555, !dbg !7
  %1557 = insertelement <4 x float> poison, float %1556, i64 0, !dbg !19
  %1558 = bitcast i32 %1553 to float, !dbg !7
  %1559 = select i1 %.not2, float %1541, float %1558, !dbg !7
  %1560 = insertelement <4 x float> %1557, float %1559, i64 1, !dbg !19
  %1561 = select i1 %.not2, float %1555, float %1546, !dbg !7
  %1562 = insertelement <4 x float> %1560, float %1561, i64 2, !dbg !19
  %1563 = select i1 %.not2, float %1558, float %1547, !dbg !7
  %1564 = insertelement <4 x float> %1562, float %1563, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1447, !dbg !19
  %normalized = fmul <4 x float> %1564, %row.reciprocal, !dbg !19
  %1565 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1566 = extractelement <4 x half> %1565, i64 3, !dbg !20
  %1567 = extractelement <4 x half> %1565, i64 2, !dbg !20
  %1568 = extractelement <4 x half> %1565, i64 1, !dbg !20
  %1569 = extractelement <4 x half> %1565, i64 0, !dbg !20
  %1570 = bitcast i32 %1442 to float, !dbg !7
  %1571 = select i1 %.not2, float %1432, float %1570, !dbg !7
  %1572 = insertelement <4 x float> poison, float %1571, i64 0, !dbg !19
  %1573 = bitcast i32 %1445 to float, !dbg !7
  %1574 = select i1 %.not2, float %1433, float %1573, !dbg !7
  %1575 = insertelement <4 x float> %1572, float %1574, i64 1, !dbg !19
  %1576 = select i1 %.not2, float %1570, float %1438, !dbg !7
  %1577 = insertelement <4 x float> %1575, float %1576, i64 2, !dbg !19
  %1578 = select i1 %.not2, float %1573, float %1439, !dbg !7
  %1579 = insertelement <4 x float> %1577, float %1578, i64 3, !dbg !19
  %normalized22 = fmul <4 x float> %1579, %row.reciprocal, !dbg !19
  %1580 = fptrunc <4 x float> %normalized22 to <4 x half>, !dbg !20
  %1581 = extractelement <4 x half> %1580, i64 3, !dbg !20
  %1582 = extractelement <4 x half> %1580, i64 2, !dbg !20
  %1583 = extractelement <4 x half> %1580, i64 1, !dbg !20
  %1584 = extractelement <4 x half> %1580, i64 0, !dbg !20
  %1585 = bitcast i32 %1332 to float, !dbg !7
  %1586 = select i1 %.not2, float %1322, float %1585, !dbg !7
  %1587 = insertelement <4 x float> poison, float %1586, i64 0, !dbg !19
  %1588 = bitcast i32 %1335 to float, !dbg !7
  %1589 = select i1 %.not2, float %1323, float %1588, !dbg !7
  %1590 = insertelement <4 x float> %1587, float %1589, i64 1, !dbg !19
  %1591 = select i1 %.not2, float %1585, float %1328, !dbg !7
  %1592 = insertelement <4 x float> %1590, float %1591, i64 2, !dbg !19
  %1593 = select i1 %.not2, float %1588, float %1329, !dbg !7
  %1594 = insertelement <4 x float> %1592, float %1593, i64 3, !dbg !19
  %row.reciprocal23 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1229, !dbg !19
  %normalized24 = fmul <4 x float> %1594, %row.reciprocal23, !dbg !19
  %1595 = fptrunc <4 x float> %normalized24 to <4 x half>, !dbg !20
  %1596 = extractelement <4 x half> %1595, i64 3, !dbg !20
  %1597 = extractelement <4 x half> %1595, i64 2, !dbg !20
  %1598 = extractelement <4 x half> %1595, i64 1, !dbg !20
  %1599 = extractelement <4 x half> %1595, i64 0, !dbg !20
  %1600 = bitcast i32 %1224 to float, !dbg !7
  %1601 = select i1 %.not2, float %1214, float %1600, !dbg !7
  %1602 = insertelement <4 x float> poison, float %1601, i64 0, !dbg !19
  %1603 = bitcast i32 %1227 to float, !dbg !7
  %1604 = select i1 %.not2, float %1215, float %1603, !dbg !7
  %1605 = insertelement <4 x float> %1602, float %1604, i64 1, !dbg !19
  %1606 = select i1 %.not2, float %1600, float %1220, !dbg !7
  %1607 = insertelement <4 x float> %1605, float %1606, i64 2, !dbg !19
  %1608 = select i1 %.not2, float %1603, float %1221, !dbg !7
  %1609 = insertelement <4 x float> %1607, float %1608, i64 3, !dbg !19
  %normalized25 = fmul <4 x float> %1609, %row.reciprocal23, !dbg !19
  %1610 = fptrunc <4 x float> %normalized25 to <4 x half>, !dbg !20
  %1611 = extractelement <4 x half> %1610, i64 3, !dbg !20
  %1612 = extractelement <4 x half> %1610, i64 2, !dbg !20
  %1613 = extractelement <4 x half> %1610, i64 1, !dbg !20
  %1614 = extractelement <4 x half> %1610, i64 0, !dbg !20
  %1615 = shl nuw nsw i64 %17, 19, !dbg !21
  %address.thread532 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local533 = zext nneg i32 %address.thread532 to i64, !dbg !13
  %address.local534 = add nsw i64 %18, %address.local533, !dbg !21
  %address.local535 = lshr i64 %address.local533, 6, !dbg !13
  %address.local536 = shl nuw nsw i64 %address.local535, 4, !dbg !21
  %address.local537 = add nsw i64 %address.local534, %address.local536, !dbg !21
  %address.local538 = and i64 %address.local533, 240, !dbg !21
  %address.local539 = sub nsw i64 %address.local537, %address.local538, !dbg !21
  %address.local540 = add nsw i64 %address.local539, %19, !dbg !21
  %address.local541 = shl nuw nsw i64 %address.local540, 7, !dbg !21
  %address.local542 = add nuw nsw i64 %1615, %address.local541, !dbg !21
  %address.local543 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local542, !dbg !21
  %address.local544 = lshr i64 %address.local533, 2, !dbg !13
  %address.local545 = and i64 %address.local544, 12, !dbg !13
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local543, i64 %address.local545, !dbg !21
  store half %1614, ptr addrspace(1) %address.local546, align 2, !dbg !21
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local542, !dbg !21
  %address.local548 = or disjoint i64 %address.local545, 1, !dbg !21
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local547, i64 %address.local548, !dbg !21
  store half %1613, ptr addrspace(1) %address.local549, align 2, !dbg !21
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local542, !dbg !21
  %address.local551 = or disjoint i64 %address.local545, 2, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local550, i64 %address.local551, !dbg !21
  store half %1612, ptr addrspace(1) %address.local552, align 2, !dbg !21
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local542, !dbg !21
  %address.local554 = or disjoint i64 %address.local545, 3, !dbg !21
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local553, i64 %address.local554, !dbg !21
  store half %1611, ptr addrspace(1) %address.local555, align 2, !dbg !21
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local542, !dbg !21
  %address.local557 = or disjoint i64 %address.local545, 16, !dbg !21
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local556, i64 %address.local557, !dbg !21
  store half %1599, ptr addrspace(1) %address.local558, align 2, !dbg !21
  %address.local559 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local542, !dbg !21
  %address.local560 = or disjoint i64 %address.local545, 17, !dbg !21
  %address.local561 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local559, i64 %address.local560, !dbg !21
  store half %1598, ptr addrspace(1) %address.local561, align 2, !dbg !21
  %address.local562 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local542, !dbg !21
  %address.local563 = or disjoint i64 %address.local545, 18, !dbg !21
  %address.local564 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local562, i64 %address.local563, !dbg !21
  store half %1597, ptr addrspace(1) %address.local564, align 2, !dbg !21
  %address.local565 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local542, !dbg !21
  %address.local566 = or disjoint i64 %address.local545, 19, !dbg !21
  %address.local567 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local565, i64 %address.local566, !dbg !21
  store half %1596, ptr addrspace(1) %address.local567, align 2, !dbg !21
  %address.local568 = shl nsw i64 %address.local540, 7, !dbg !21
  %address.local569 = add i64 %address.local568, 8192, !dbg !21
  %address.local570 = add nuw nsw i64 %1615, %address.local569, !dbg !21
  %address.local571 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local570, !dbg !21
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local571, i64 %address.local545, !dbg !21
  store half %1584, ptr addrspace(1) %address.local572, align 2, !dbg !21
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local570, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local573, i64 %address.local548, !dbg !21
  store half %1583, ptr addrspace(1) %address.local574, align 2, !dbg !21
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local570, !dbg !21
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local551, !dbg !21
  store half %1582, ptr addrspace(1) %address.local576, align 2, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local570, !dbg !21
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local554, !dbg !21
  store half %1581, ptr addrspace(1) %address.local578, align 2, !dbg !21
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local570, !dbg !21
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local579, i64 %address.local557, !dbg !21
  store half %1569, ptr addrspace(1) %address.local580, align 2, !dbg !21
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local570, !dbg !21
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local560, !dbg !21
  store half %1568, ptr addrspace(1) %address.local582, align 2, !dbg !21
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local570, !dbg !21
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local563, !dbg !21
  store half %1567, ptr addrspace(1) %address.local584, align 2, !dbg !21
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local570, !dbg !21
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local585, i64 %address.local566, !dbg !21
  store half %1566, ptr addrspace(1) %address.local586, align 2, !dbg !21
  br label %1616, !dbg !21

1616:                                             ; preds = %1554, %1105
  br i1 true, label %1617, label %1655, !dbg !21

1617:                                             ; preds = %1616
  %1618 = insertelement <4 x float> poison, float %1524, i64 0, !dbg !19
  %1619 = insertelement <4 x float> %1618, float %1530, i64 1, !dbg !19
  %1620 = insertelement <4 x float> %1619, float %1525, i64 2, !dbg !19
  %1621 = insertelement <4 x float> %1620, float %1531, i64 3, !dbg !19
  %row.reciprocal26 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1447, !dbg !19
  %normalized27 = fmul <4 x float> %1621, %row.reciprocal26, !dbg !19
  %1622 = fptrunc <4 x float> %normalized27 to <4 x half>, !dbg !20
  %1623 = extractelement <4 x half> %1622, i64 3, !dbg !20
  %1624 = extractelement <4 x half> %1622, i64 2, !dbg !20
  %1625 = extractelement <4 x half> %1622, i64 1, !dbg !20
  %1626 = extractelement <4 x half> %1622, i64 0, !dbg !20
  %1627 = insertelement <4 x float> poison, float %1415, i64 0, !dbg !19
  %1628 = insertelement <4 x float> %1627, float %1421, i64 1, !dbg !19
  %1629 = insertelement <4 x float> %1628, float %1416, i64 2, !dbg !19
  %1630 = insertelement <4 x float> %1629, float %1422, i64 3, !dbg !19
  %normalized28 = fmul <4 x float> %1630, %row.reciprocal26, !dbg !19
  %1631 = fptrunc <4 x float> %normalized28 to <4 x half>, !dbg !20
  %1632 = extractelement <4 x half> %1631, i64 3, !dbg !20
  %1633 = extractelement <4 x half> %1631, i64 2, !dbg !20
  %1634 = extractelement <4 x half> %1631, i64 1, !dbg !20
  %1635 = extractelement <4 x half> %1631, i64 0, !dbg !20
  %1636 = insertelement <4 x float> poison, float %1306, i64 0, !dbg !19
  %1637 = insertelement <4 x float> %1636, float %1312, i64 1, !dbg !19
  %1638 = insertelement <4 x float> %1637, float %1307, i64 2, !dbg !19
  %1639 = insertelement <4 x float> %1638, float %1313, i64 3, !dbg !19
  %row.reciprocal29 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1229, !dbg !19
  %normalized30 = fmul <4 x float> %1639, %row.reciprocal29, !dbg !19
  %1640 = fptrunc <4 x float> %normalized30 to <4 x half>, !dbg !20
  %1641 = extractelement <4 x half> %1640, i64 3, !dbg !20
  %1642 = extractelement <4 x half> %1640, i64 2, !dbg !20
  %1643 = extractelement <4 x half> %1640, i64 1, !dbg !20
  %1644 = extractelement <4 x half> %1640, i64 0, !dbg !20
  %1645 = insertelement <4 x float> poison, float %1197, i64 0, !dbg !19
  %1646 = insertelement <4 x float> %1645, float %1203, i64 1, !dbg !19
  %1647 = insertelement <4 x float> %1646, float %1198, i64 2, !dbg !19
  %1648 = insertelement <4 x float> %1647, float %1204, i64 3, !dbg !19
  %normalized31 = fmul <4 x float> %1648, %row.reciprocal29, !dbg !19
  %1649 = fptrunc <4 x float> %normalized31 to <4 x half>, !dbg !20
  %1650 = extractelement <4 x half> %1649, i64 3, !dbg !20
  %1651 = extractelement <4 x half> %1649, i64 2, !dbg !20
  %1652 = extractelement <4 x half> %1649, i64 1, !dbg !20
  %1653 = extractelement <4 x half> %1649, i64 0, !dbg !20
  %1654 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread587 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local588 = zext nneg i32 %address.thread587 to i64, !dbg !13
  %address.local589 = add nsw i64 %18, %address.local588, !dbg !21
  %address.local590 = lshr i64 %address.local588, 6, !dbg !13
  %address.local591 = shl nuw nsw i64 %address.local590, 4, !dbg !21
  %address.local592 = add nsw i64 %address.local589, %address.local591, !dbg !21
  %address.local593 = and i64 %address.local588, 240, !dbg !21
  %address.local594 = sub nsw i64 %address.local592, %address.local593, !dbg !21
  %address.local595 = add nsw i64 %address.local594, %19, !dbg !21
  %address.local596 = shl nuw nsw i64 %address.local595, 7, !dbg !21
  %address.local597 = add nuw nsw i64 %1654, %address.local596, !dbg !21
  %address.local598 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local599 = lshr i64 %address.local588, 2, !dbg !13
  %address.local600 = and i64 %address.local599, 12, !dbg !13
  %address.local601 = or disjoint i64 %address.local600, 32, !dbg !21
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local598, i64 %address.local601, !dbg !21
  store half %1653, ptr addrspace(1) %address.local602, align 2, !dbg !21
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local604 = or disjoint i64 %address.local600, 33, !dbg !21
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local603, i64 %address.local604, !dbg !21
  store half %1652, ptr addrspace(1) %address.local605, align 2, !dbg !21
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local607 = or disjoint i64 %address.local600, 34, !dbg !21
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local606, i64 %address.local607, !dbg !21
  store half %1651, ptr addrspace(1) %address.local608, align 2, !dbg !21
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local610 = or disjoint i64 %address.local600, 35, !dbg !21
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local609, i64 %address.local610, !dbg !21
  store half %1650, ptr addrspace(1) %address.local611, align 2, !dbg !21
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local613 = or disjoint i64 %address.local600, 48, !dbg !21
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local612, i64 %address.local613, !dbg !21
  store half %1644, ptr addrspace(1) %address.local614, align 2, !dbg !21
  %address.local615 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local616 = or disjoint i64 %address.local600, 49, !dbg !21
  %address.local617 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local616, !dbg !21
  store half %1643, ptr addrspace(1) %address.local617, align 2, !dbg !21
  %address.local618 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local619 = or disjoint i64 %address.local600, 50, !dbg !21
  %address.local620 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local618, i64 %address.local619, !dbg !21
  store half %1642, ptr addrspace(1) %address.local620, align 2, !dbg !21
  %address.local621 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local622 = or i64 %address.local599, 51, !dbg !21
  %address.local623 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local621, i64 %address.local622, !dbg !21
  store half %1641, ptr addrspace(1) %address.local623, align 2, !dbg !21
  %address.local624 = shl nsw i64 %address.local595, 7, !dbg !21
  %address.local625 = add i64 %address.local624, 8192, !dbg !21
  %address.local626 = add nuw nsw i64 %1654, %address.local625, !dbg !21
  %address.local627 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local626, !dbg !21
  %address.local628 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local627, i64 %address.local601, !dbg !21
  store half %1635, ptr addrspace(1) %address.local628, align 2, !dbg !21
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local626, !dbg !21
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local629, i64 %address.local604, !dbg !21
  store half %1634, ptr addrspace(1) %address.local630, align 2, !dbg !21
  %address.local631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local626, !dbg !21
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local607, !dbg !21
  store half %1633, ptr addrspace(1) %address.local632, align 2, !dbg !21
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local626, !dbg !21
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local633, i64 %address.local610, !dbg !21
  store half %1632, ptr addrspace(1) %address.local634, align 2, !dbg !21
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local626, !dbg !21
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local635, i64 %address.local613, !dbg !21
  store half %1626, ptr addrspace(1) %address.local636, align 2, !dbg !21
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local626, !dbg !21
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local637, i64 %address.local616, !dbg !21
  store half %1625, ptr addrspace(1) %address.local638, align 2, !dbg !21
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local626, !dbg !21
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local639, i64 %address.local619, !dbg !21
  store half %1624, ptr addrspace(1) %address.local640, align 2, !dbg !21
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local626, !dbg !21
  %address.local642 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local641, i64 %address.local622, !dbg !21
  store half %1623, ptr addrspace(1) %address.local642, align 2, !dbg !21
  br label %1655, !dbg !21

1655:                                             ; preds = %1617, %1616
  br i1 true, label %1656, label %1694, !dbg !21

1656:                                             ; preds = %1655
  %1657 = insertelement <4 x float> poison, float %1496, i64 0, !dbg !19
  %1658 = insertelement <4 x float> %1657, float %1502, i64 1, !dbg !19
  %1659 = insertelement <4 x float> %1658, float %1497, i64 2, !dbg !19
  %1660 = insertelement <4 x float> %1659, float %1503, i64 3, !dbg !19
  %row.reciprocal32 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1447, !dbg !19
  %normalized33 = fmul <4 x float> %1660, %row.reciprocal32, !dbg !19
  %1661 = fptrunc <4 x float> %normalized33 to <4 x half>, !dbg !20
  %1662 = extractelement <4 x half> %1661, i64 3, !dbg !20
  %1663 = extractelement <4 x half> %1661, i64 2, !dbg !20
  %1664 = extractelement <4 x half> %1661, i64 1, !dbg !20
  %1665 = extractelement <4 x half> %1661, i64 0, !dbg !20
  %1666 = insertelement <4 x float> poison, float %1386, i64 0, !dbg !19
  %1667 = insertelement <4 x float> %1666, float %1392, i64 1, !dbg !19
  %1668 = insertelement <4 x float> %1667, float %1387, i64 2, !dbg !19
  %1669 = insertelement <4 x float> %1668, float %1393, i64 3, !dbg !19
  %normalized34 = fmul <4 x float> %1669, %row.reciprocal32, !dbg !19
  %1670 = fptrunc <4 x float> %normalized34 to <4 x half>, !dbg !20
  %1671 = extractelement <4 x half> %1670, i64 3, !dbg !20
  %1672 = extractelement <4 x half> %1670, i64 2, !dbg !20
  %1673 = extractelement <4 x half> %1670, i64 1, !dbg !20
  %1674 = extractelement <4 x half> %1670, i64 0, !dbg !20
  %1675 = insertelement <4 x float> poison, float %1278, i64 0, !dbg !19
  %1676 = insertelement <4 x float> %1675, float %1284, i64 1, !dbg !19
  %1677 = insertelement <4 x float> %1676, float %1279, i64 2, !dbg !19
  %1678 = insertelement <4 x float> %1677, float %1285, i64 3, !dbg !19
  %row.reciprocal35 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1229, !dbg !19
  %normalized36 = fmul <4 x float> %1678, %row.reciprocal35, !dbg !19
  %1679 = fptrunc <4 x float> %normalized36 to <4 x half>, !dbg !20
  %1680 = extractelement <4 x half> %1679, i64 3, !dbg !20
  %1681 = extractelement <4 x half> %1679, i64 2, !dbg !20
  %1682 = extractelement <4 x half> %1679, i64 1, !dbg !20
  %1683 = extractelement <4 x half> %1679, i64 0, !dbg !20
  %1684 = insertelement <4 x float> poison, float %1168, i64 0, !dbg !19
  %1685 = insertelement <4 x float> %1684, float %1174, i64 1, !dbg !19
  %1686 = insertelement <4 x float> %1685, float %1169, i64 2, !dbg !19
  %1687 = insertelement <4 x float> %1686, float %1175, i64 3, !dbg !19
  %normalized37 = fmul <4 x float> %1687, %row.reciprocal35, !dbg !19
  %1688 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !20
  %1689 = extractelement <4 x half> %1688, i64 3, !dbg !20
  %1690 = extractelement <4 x half> %1688, i64 2, !dbg !20
  %1691 = extractelement <4 x half> %1688, i64 1, !dbg !20
  %1692 = extractelement <4 x half> %1688, i64 0, !dbg !20
  %1693 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread643 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local644 = zext nneg i32 %address.thread643 to i64, !dbg !13
  %address.local645 = add nsw i64 %18, %address.local644, !dbg !21
  %address.local646 = lshr i64 %address.local644, 6, !dbg !13
  %address.local647 = shl nuw nsw i64 %address.local646, 4, !dbg !21
  %address.local648 = add nsw i64 %address.local645, %address.local647, !dbg !21
  %address.local649 = and i64 %address.local644, 240, !dbg !21
  %address.local650 = sub nsw i64 %address.local648, %address.local649, !dbg !21
  %address.local651 = add nsw i64 %address.local650, %19, !dbg !21
  %address.local652 = shl nuw nsw i64 %address.local651, 7, !dbg !21
  %address.local653 = add nuw nsw i64 %1693, %address.local652, !dbg !21
  %address.local654 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local653, !dbg !21
  %address.local655 = lshr i64 %address.local644, 2, !dbg !13
  %address.local656 = and i64 %address.local655, 12, !dbg !13
  %address.local657 = or disjoint i64 %address.local656, 64, !dbg !21
  %address.local658 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local654, i64 %address.local657, !dbg !21
  store half %1692, ptr addrspace(1) %address.local658, align 2, !dbg !21
  %address.local659 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local653, !dbg !21
  %address.local660 = or disjoint i64 %address.local656, 65, !dbg !21
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local660, !dbg !21
  store half %1691, ptr addrspace(1) %address.local661, align 2, !dbg !21
  %address.local662 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local653, !dbg !21
  %address.local663 = or disjoint i64 %address.local656, 66, !dbg !21
  %address.local664 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local662, i64 %address.local663, !dbg !21
  store half %1690, ptr addrspace(1) %address.local664, align 2, !dbg !21
  %address.local665 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local653, !dbg !21
  %address.local666 = or disjoint i64 %address.local656, 67, !dbg !21
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local665, i64 %address.local666, !dbg !21
  store half %1689, ptr addrspace(1) %address.local667, align 2, !dbg !21
  %address.local668 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local653, !dbg !21
  %address.local669 = or disjoint i64 %address.local656, 80, !dbg !21
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local668, i64 %address.local669, !dbg !21
  store half %1683, ptr addrspace(1) %address.local670, align 2, !dbg !21
  %address.local671 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local653, !dbg !21
  %address.local672 = or disjoint i64 %address.local656, 81, !dbg !21
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local671, i64 %address.local672, !dbg !21
  store half %1682, ptr addrspace(1) %address.local673, align 2, !dbg !21
  %address.local674 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local653, !dbg !21
  %address.local675 = or disjoint i64 %address.local656, 82, !dbg !21
  %address.local676 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local674, i64 %address.local675, !dbg !21
  store half %1681, ptr addrspace(1) %address.local676, align 2, !dbg !21
  %address.local677 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local653, !dbg !21
  %address.local678 = or disjoint i64 %address.local656, 83, !dbg !21
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local677, i64 %address.local678, !dbg !21
  store half %1680, ptr addrspace(1) %address.local679, align 2, !dbg !21
  %address.local680 = shl nsw i64 %address.local651, 7, !dbg !21
  %address.local681 = add i64 %address.local680, 8192, !dbg !21
  %address.local682 = add nuw nsw i64 %1693, %address.local681, !dbg !21
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local682, !dbg !21
  %address.local684 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local683, i64 %address.local657, !dbg !21
  store half %1674, ptr addrspace(1) %address.local684, align 2, !dbg !21
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local682, !dbg !21
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local685, i64 %address.local660, !dbg !21
  store half %1673, ptr addrspace(1) %address.local686, align 2, !dbg !21
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local682, !dbg !21
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local687, i64 %address.local663, !dbg !21
  store half %1672, ptr addrspace(1) %address.local688, align 2, !dbg !21
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local682, !dbg !21
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local689, i64 %address.local666, !dbg !21
  store half %1671, ptr addrspace(1) %address.local690, align 2, !dbg !21
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local682, !dbg !21
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local691, i64 %address.local669, !dbg !21
  store half %1665, ptr addrspace(1) %address.local692, align 2, !dbg !21
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local682, !dbg !21
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local693, i64 %address.local672, !dbg !21
  store half %1664, ptr addrspace(1) %address.local694, align 2, !dbg !21
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local682, !dbg !21
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local695, i64 %address.local675, !dbg !21
  store half %1663, ptr addrspace(1) %address.local696, align 2, !dbg !21
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local682, !dbg !21
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local697, i64 %address.local678, !dbg !21
  store half %1662, ptr addrspace(1) %address.local698, align 2, !dbg !21
  br label %1694, !dbg !21

1694:                                             ; preds = %1656, %1655
  br i1 true, label %1695, label %1733, !dbg !21

1695:                                             ; preds = %1694
  %1696 = insertelement <4 x float> poison, float %1468, i64 0, !dbg !19
  %1697 = insertelement <4 x float> %1696, float %1474, i64 1, !dbg !19
  %1698 = insertelement <4 x float> %1697, float %1469, i64 2, !dbg !19
  %1699 = insertelement <4 x float> %1698, float %1475, i64 3, !dbg !19
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1447, !dbg !19
  %normalized39 = fmul <4 x float> %1699, %row.reciprocal38, !dbg !19
  %1700 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !20
  %1701 = extractelement <4 x half> %1700, i64 3, !dbg !20
  %1702 = extractelement <4 x half> %1700, i64 2, !dbg !20
  %1703 = extractelement <4 x half> %1700, i64 1, !dbg !20
  %1704 = extractelement <4 x half> %1700, i64 0, !dbg !20
  %1705 = insertelement <4 x float> poison, float %1357, i64 0, !dbg !19
  %1706 = insertelement <4 x float> %1705, float %1363, i64 1, !dbg !19
  %1707 = insertelement <4 x float> %1706, float %1358, i64 2, !dbg !19
  %1708 = insertelement <4 x float> %1707, float %1364, i64 3, !dbg !19
  %normalized40 = fmul <4 x float> %1708, %row.reciprocal38, !dbg !19
  %1709 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !20
  %1710 = extractelement <4 x half> %1709, i64 3, !dbg !20
  %1711 = extractelement <4 x half> %1709, i64 2, !dbg !20
  %1712 = extractelement <4 x half> %1709, i64 1, !dbg !20
  %1713 = extractelement <4 x half> %1709, i64 0, !dbg !20
  %1714 = insertelement <4 x float> poison, float %1250, i64 0, !dbg !19
  %1715 = insertelement <4 x float> %1714, float %1256, i64 1, !dbg !19
  %1716 = insertelement <4 x float> %1715, float %1251, i64 2, !dbg !19
  %1717 = insertelement <4 x float> %1716, float %1257, i64 3, !dbg !19
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1229, !dbg !19
  %normalized42 = fmul <4 x float> %1717, %row.reciprocal41, !dbg !19
  %1718 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !20
  %1719 = extractelement <4 x half> %1718, i64 3, !dbg !20
  %1720 = extractelement <4 x half> %1718, i64 2, !dbg !20
  %1721 = extractelement <4 x half> %1718, i64 1, !dbg !20
  %1722 = extractelement <4 x half> %1718, i64 0, !dbg !20
  %1723 = insertelement <4 x float> poison, float %1139, i64 0, !dbg !19
  %1724 = insertelement <4 x float> %1723, float %1145, i64 1, !dbg !19
  %1725 = insertelement <4 x float> %1724, float %1140, i64 2, !dbg !19
  %1726 = insertelement <4 x float> %1725, float %1146, i64 3, !dbg !19
  %normalized43 = fmul <4 x float> %1726, %row.reciprocal41, !dbg !19
  %1727 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !20
  %1728 = extractelement <4 x half> %1727, i64 3, !dbg !20
  %1729 = extractelement <4 x half> %1727, i64 2, !dbg !20
  %1730 = extractelement <4 x half> %1727, i64 1, !dbg !20
  %1731 = extractelement <4 x half> %1727, i64 0, !dbg !20
  %1732 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread699 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local700 = zext nneg i32 %address.thread699 to i64, !dbg !13
  %address.local701 = add nsw i64 %18, %address.local700, !dbg !21
  %address.local702 = lshr i64 %address.local700, 6, !dbg !13
  %address.local703 = shl nuw nsw i64 %address.local702, 4, !dbg !21
  %address.local704 = add nsw i64 %address.local701, %address.local703, !dbg !21
  %address.local705 = and i64 %address.local700, 240, !dbg !21
  %address.local706 = sub nsw i64 %address.local704, %address.local705, !dbg !21
  %address.local707 = add nsw i64 %address.local706, %19, !dbg !21
  %address.local708 = shl nuw nsw i64 %address.local707, 7, !dbg !21
  %address.local709 = add nuw nsw i64 %1732, %address.local708, !dbg !21
  %address.local710 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local709, !dbg !21
  %address.local711 = lshr i64 %address.local700, 2, !dbg !13
  %address.local712 = and i64 %address.local711, 12, !dbg !13
  %address.local713 = or disjoint i64 %address.local712, 96, !dbg !21
  %address.local714 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local710, i64 %address.local713, !dbg !21
  store half %1731, ptr addrspace(1) %address.local714, align 2, !dbg !21
  %address.local715 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local709, !dbg !21
  %address.local716 = or disjoint i64 %address.local712, 97, !dbg !21
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local715, i64 %address.local716, !dbg !21
  store half %1730, ptr addrspace(1) %address.local717, align 2, !dbg !21
  %address.local718 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local709, !dbg !21
  %address.local719 = or disjoint i64 %address.local712, 98, !dbg !21
  %address.local720 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local718, i64 %address.local719, !dbg !21
  store half %1729, ptr addrspace(1) %address.local720, align 2, !dbg !21
  %address.local721 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local709, !dbg !21
  %address.local722 = or disjoint i64 %address.local712, 99, !dbg !21
  %address.local723 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local721, i64 %address.local722, !dbg !21
  store half %1728, ptr addrspace(1) %address.local723, align 2, !dbg !21
  %address.local724 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local709, !dbg !21
  %address.local725 = or disjoint i64 %address.local712, 112, !dbg !21
  %address.local726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local724, i64 %address.local725, !dbg !21
  store half %1722, ptr addrspace(1) %address.local726, align 2, !dbg !21
  %address.local727 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local709, !dbg !21
  %address.local728 = or disjoint i64 %address.local712, 113, !dbg !21
  %address.local729 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local727, i64 %address.local728, !dbg !21
  store half %1721, ptr addrspace(1) %address.local729, align 2, !dbg !21
  %address.local730 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local709, !dbg !21
  %address.local731 = or disjoint i64 %address.local712, 114, !dbg !21
  %address.local732 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local730, i64 %address.local731, !dbg !21
  store half %1720, ptr addrspace(1) %address.local732, align 2, !dbg !21
  %address.local733 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local709, !dbg !21
  %address.local734 = or i64 %address.local711, 115, !dbg !21
  %address.local735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local733, i64 %address.local734, !dbg !21
  store half %1719, ptr addrspace(1) %address.local735, align 2, !dbg !21
  %address.local736 = shl nsw i64 %address.local707, 7, !dbg !21
  %address.local737 = add i64 %address.local736, 8192, !dbg !21
  %address.local738 = add nuw nsw i64 %1732, %address.local737, !dbg !21
  %address.local739 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local738, !dbg !21
  %address.local740 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local739, i64 %address.local713, !dbg !21
  store half %1713, ptr addrspace(1) %address.local740, align 2, !dbg !21
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local738, !dbg !21
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local741, i64 %address.local716, !dbg !21
  store half %1712, ptr addrspace(1) %address.local742, align 2, !dbg !21
  %address.local743 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local738, !dbg !21
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local743, i64 %address.local719, !dbg !21
  store half %1711, ptr addrspace(1) %address.local744, align 2, !dbg !21
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local738, !dbg !21
  %address.local746 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local745, i64 %address.local722, !dbg !21
  store half %1710, ptr addrspace(1) %address.local746, align 2, !dbg !21
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local738, !dbg !21
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local747, i64 %address.local725, !dbg !21
  store half %1704, ptr addrspace(1) %address.local748, align 2, !dbg !21
  %address.local749 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local738, !dbg !21
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local749, i64 %address.local728, !dbg !21
  store half %1703, ptr addrspace(1) %address.local750, align 2, !dbg !21
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local738, !dbg !21
  %address.local752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local751, i64 %address.local731, !dbg !21
  store half %1702, ptr addrspace(1) %address.local752, align 2, !dbg !21
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local738, !dbg !21
  %address.local754 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local753, i64 %address.local734, !dbg !21
  store half %1701, ptr addrspace(1) %address.local754, align 2, !dbg !21
  br label %1733, !dbg !21

1733:                                             ; preds = %1695, %1694
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
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

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
!llvm.module.flags = !{!39}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/pipeline-loop-only")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!9 = !DILocation(line: 48, column: 17, scope: !4)
!12 = !DILocation(line: 70, column: 22, scope: !4)
!13 = !DILocation(line: 72, column: 24, scope: !4)
!14 = !DILocation(line: 63, column: 17, scope: !4)
!15 = !DILocation(line: 77, column: 19, scope: !4)
!16 = !DILocation(line: 102, column: 23, scope: !4)
!17 = !DILocation(line: 71, column: 22, scope: !4)
!18 = !DILocation(line: 73, column: 26, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
!22 = !DILocation(line: 74, column: 23, scope: !4)
!23 = !DILocation(line: 101, column: 20, scope: !4)
!24 = !DILocation(line: 79, column: 24, scope: !4)
!25 = !DILocation(line: 80, column: 23, scope: !4)
!26 = !DILocation(line: 81, column: 26, scope: !4)
!27 = !DILocation(line: 88, column: 21, scope: !4)
!28 = !DILocation(line: 89, column: 20, scope: !4)
!29 = !DILocation(line: 90, column: 22, scope: !4)
!30 = !DILocation(line: 91, column: 23, scope: !4)
!31 = !DILocation(line: 92, column: 20, scope: !4)
!32 = !DILocation(line: 93, column: 22, scope: !4)
!33 = !DILocation(line: 94, column: 22, scope: !4)
!34 = !DILocation(line: 95, column: 20, scope: !4)
!35 = !DILocation(line: 96, column: 16, scope: !4)
!36 = !DILocation(line: 103, column: 22, scope: !4)
!38 = !{i32 256, i32 1, i32 1}
!39 = !{i32 2, !"Debug Info Version", i32 3}
!40 = !{i32 0, i32 256}
