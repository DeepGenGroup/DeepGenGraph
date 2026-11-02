; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/online-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/online-baseline/input.mlir"

@shm_0 = addrspace(3) global [8192 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !38 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 6, !dbg !5
  %10 = add nsw i64 %9, %6, !dbg !5
  %11 = lshr i64 %10, 5, !dbg !5
  %12 = and i64 %11, 576460752303423484, !dbg !5
  %13 = and i64 %6, 3, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = sub nsw i64 63, %14, !dbg !5
  %16 = lshr i64 %10, 2, !dbg !5
  %17 = icmp ugt i64 %10, 2047, !dbg !8
  %18 = add nsw i64 %14, -64, !dbg !8
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !8
  %20 = sdiv i64 %19, 64, !dbg !8
  %21 = sext i1 %17 to i64, !dbg !8
  %22 = xor i64 %20, %21, !dbg !8
  %23 = add nsw i64 %16, %22, !dbg !8
  %24 = and i64 %23, 31, !dbg !8
  %25 = shl nsw i64 %15, 6, !dbg !8
  %26 = mul nsw i64 %22, -4096, !dbg !8
  %27 = add i64 %25, %26, !dbg !8
  %28 = shl nuw nsw i64 %24, 19, !dbg !8
  %29 = shl nsw i64 %27, 7, !dbg !8
  %30 = add i64 %28, %29, !dbg !8
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !40
  %32 = zext nneg i32 %31 to i64, !dbg !8
  %33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !8, !range !40, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local39 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local40 = and i64 %address.local39, 1536, !dbg !8
  %address.local41 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local40, !dbg !8
  %address.local42 = ptrtoint ptr addrspace(3) %address.local41 to i64, !dbg !8
  %address.local43 = trunc i64 %address.local42 to i32, !dbg !8
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local43), !dbg !8
  %35 = zext i32 %34 to i64, !dbg !8
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !8
  %address.local44 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local45 = and i64 %address.local44, 8064, !dbg !8
  %address.local46 = lshr i64 %address.local, 3, !dbg !8
  %address.local47 = and i64 %address.local46, 24, !dbg !8
  %address.local48 = or disjoint i64 %address.local45, %address.local47, !dbg !8
  %address.local49 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local48, !dbg !8
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %38 = ptrtoint ptr addrspace(1) %address.local49 to i64, !dbg !8
  %39 = sub i64 %38, %37, !dbg !8
  %dma.buffer.offset489 = trunc i64 %39 to i32, !dbg !8
  %40 = trunc i64 %37 to i32, !dbg !8
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !8
  %42 = lshr i64 %37, 32, !dbg !8
  %43 = trunc i64 %42 to i32, !dbg !8
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !8
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc490 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !8
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !8
  %dma.buffer.lds491 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds491, i32 %dma.buffer.offset489, <4 x i32> %dma.buffer.rsrc490) #10, !dbg !8
  %48 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %49 = sext i32 %48 to i64, !dbg !5
  %50 = shl i64 %15, 6, !dbg !10
  %51 = lshr i64 %32, 6, !dbg !11
  %52 = lshr i64 %32, 2, !dbg !11
  %53 = and i64 %52, 12, !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %54 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %55 = getelementptr i8, ptr addrspace(1) %54, i64 64, !dbg !8
  %address.local50 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local40, !dbg !8
  %address.local51 = ptrtoint ptr addrspace(3) %address.local50 to i64, !dbg !8
  %address.local52 = trunc i64 %address.local51 to i32, !dbg !8
  %56 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local52), !dbg !8
  %57 = zext i32 %56 to i64, !dbg !8
  %58 = inttoptr i64 %57 to ptr addrspace(3), !dbg !8
  %address.local53 = getelementptr [2 x i8], ptr addrspace(1) %55, i64 %address.local48, !dbg !8
  %59 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %60 = ptrtoint ptr addrspace(1) %address.local53 to i64, !dbg !8
  %61 = sub i64 %60, %59, !dbg !8
  %dma.buffer.offset486 = trunc i64 %61 to i32, !dbg !8
  %62 = trunc i64 %59 to i32, !dbg !8
  %63 = insertelement <4 x i32> poison, i32 %62, i64 0, !dbg !8
  %64 = lshr i64 %59, 32, !dbg !8
  %65 = trunc i64 %64 to i32, !dbg !8
  %66 = insertelement <4 x i32> %63, i32 %65, i64 1, !dbg !8
  %67 = insertelement <4 x i32> %66, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc487 = insertelement <4 x i32> %67, i32 131072, i64 3, !dbg !8
  %68 = ptrtoint ptr addrspace(3) %58 to i32, !dbg !8
  %dma.buffer.lds488 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %68), !dbg !8
  %69 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds488, i32 %dma.buffer.offset486, <4 x i32> %dma.buffer.rsrc487) #10, !dbg !8
  %address.local54 = lshr i32 %address.thread, 2, !dbg !11
  %address.local55 = and i32 %address.local54, 4, !dbg !11
  %address.local56 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local57 = and i32 %address.local56, 384, !dbg !11
  %address.local58 = or disjoint i32 %address.local55, %address.local57, !dbg !11
  %address.local59 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local60 = and i32 %address.local59, 120, !dbg !11
  %address.local61 = or disjoint i32 %address.local58, %address.local60, !dbg !11
  %address.local62 = shl nuw nsw i32 %address.thread, 4, !dbg !11
  %address.local63 = and i32 %address.local62, 512, !dbg !11
  %address.local64 = or disjoint i32 %address.local61, %address.local63, !dbg !11
  %address.local65 = zext nneg i32 %address.local64 to i64, !dbg !11
  %address.local66 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local65, !dbg !11
  %70 = load <4 x half>, ptr addrspace(3) %address.local66, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local66, i64 2048, !dbg !11
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %72 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !11
  %73 = fmul <4 x half> %71, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %74 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %75 = getelementptr i8, ptr addrspace(1) %74, i64 128, !dbg !8
  %76 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local43), !dbg !8
  %77 = zext i32 %76 to i64, !dbg !8
  %78 = inttoptr i64 %77 to ptr addrspace(3), !dbg !8
  %address.local81 = getelementptr [2 x i8], ptr addrspace(1) %75, i64 %address.local48, !dbg !8
  %79 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %80 = ptrtoint ptr addrspace(1) %address.local81 to i64, !dbg !8
  %81 = sub i64 %80, %79, !dbg !8
  %dma.buffer.offset483 = trunc i64 %81 to i32, !dbg !8
  %82 = trunc i64 %79 to i32, !dbg !8
  %83 = insertelement <4 x i32> poison, i32 %82, i64 0, !dbg !8
  %84 = lshr i64 %79, 32, !dbg !8
  %85 = trunc i64 %84 to i32, !dbg !8
  %86 = insertelement <4 x i32> %83, i32 %85, i64 1, !dbg !8
  %87 = insertelement <4 x i32> %86, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc484 = insertelement <4 x i32> %87, i32 131072, i64 3, !dbg !8
  %88 = ptrtoint ptr addrspace(3) %78 to i32, !dbg !8
  %dma.buffer.lds485 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %88), !dbg !8
  %89 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds485, i32 %dma.buffer.offset483, <4 x i32> %dma.buffer.rsrc484) #10, !dbg !8
  %lds.const.gep440 = getelementptr i8, ptr addrspace(3) %address.local66, i64 4096, !dbg !11
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep440, align 8, !dbg !11
  %lds.const.gep441 = getelementptr i8, ptr addrspace(3) %address.local66, i64 6144, !dbg !11
  %91 = load <4 x half>, ptr addrspace(3) %lds.const.gep441, align 8, !dbg !11
  %92 = fmul <4 x half> %90, splat (half 0xH3015), !dbg !11
  %93 = fmul <4 x half> %91, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %94 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %95 = getelementptr i8, ptr addrspace(1) %94, i64 192, !dbg !8
  %96 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local52), !dbg !8
  %97 = zext i32 %96 to i64, !dbg !8
  %98 = inttoptr i64 %97 to ptr addrspace(3), !dbg !8
  %address.local84 = getelementptr [2 x i8], ptr addrspace(1) %95, i64 %address.local48, !dbg !8
  %99 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %100 = ptrtoint ptr addrspace(1) %address.local84 to i64, !dbg !8
  %101 = sub i64 %100, %99, !dbg !8
  %dma.buffer.offset480 = trunc i64 %101 to i32, !dbg !8
  %102 = trunc i64 %99 to i32, !dbg !8
  %103 = insertelement <4 x i32> poison, i32 %102, i64 0, !dbg !8
  %104 = lshr i64 %99, 32, !dbg !8
  %105 = trunc i64 %104 to i32, !dbg !8
  %106 = insertelement <4 x i32> %103, i32 %105, i64 1, !dbg !8
  %107 = insertelement <4 x i32> %106, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc481 = insertelement <4 x i32> %107, i32 131072, i64 3, !dbg !8
  %108 = ptrtoint ptr addrspace(3) %98 to i32, !dbg !8
  %dma.buffer.lds482 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %108), !dbg !8
  %109 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds482, i32 %dma.buffer.offset480, <4 x i32> %dma.buffer.rsrc481) #10, !dbg !8
  %110 = load <4 x half>, ptr addrspace(3) %address.local66, align 8, !dbg !11
  %lds.const.gep442 = getelementptr i8, ptr addrspace(3) %address.local66, i64 2048, !dbg !11
  %111 = load <4 x half>, ptr addrspace(3) %lds.const.gep442, align 8, !dbg !11
  %112 = fmul <4 x half> %110, splat (half 0xH3015), !dbg !11
  %113 = fmul <4 x half> %111, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep443 = getelementptr i8, ptr addrspace(3) %address.local66, i64 4096, !dbg !11
  %114 = load <4 x half>, ptr addrspace(3) %lds.const.gep443, align 8, !dbg !11
  %lds.const.gep444 = getelementptr i8, ptr addrspace(3) %address.local66, i64 6144, !dbg !11
  %115 = load <4 x half>, ptr addrspace(3) %lds.const.gep444, align 8, !dbg !11
  %116 = and i32 %31, 16, !dbg !14
  %.not = icmp eq i32 %116, 0, !dbg !14
  %117 = and i32 %31, 32, !dbg !14
  %118 = fmul <4 x half> %114, splat (half 0xH3015), !dbg !11
  %119 = fmul <4 x half> %115, splat (half 0xH3015), !dbg !11
  %.not1 = icmp eq i32 %117, 0, !dbg !14
  %120 = or disjoint i64 %50, 63, !dbg !15
  %121 = add i64 %50, 64, !dbg !15
  %122 = icmp uge i64 %121, %120, !dbg !15
  %123 = shl nsw i64 %15, 1, !dbg !7
  %124 = add nsw i64 %123, 2, !dbg !7
  %125 = shl i64 %16, 7, !dbg !18
  %126 = and i64 %125, 3968, !dbg !18
  %127 = shl nsw i64 %49, 12, !dbg !18
  %128 = or disjoint i64 %126, %127, !dbg !18
  br label %129, !dbg !7

129:                                              ; preds = %763, %4
  %130 = phi i64 [ %793, %763 ], [ 0, %4 ], !dbg !7
  %131 = phi <4 x float> [ %674, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %132 = phi <4 x float> [ %679, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %133 = phi <8 x float> [ %788, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %134 = phi <1 x float> [ %789, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %135 = phi <1 x float> [ %576, %763 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %136 = phi <4 x float> [ %721, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %137 = phi <4 x float> [ %726, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %138 = phi <8 x float> [ %790, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %139 = phi <4 x float> [ %768, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %140 = phi <4 x float> [ %773, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %141 = phi <8 x float> [ %791, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %142 = phi <4 x float> [ %782, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %143 = phi <4 x float> [ %787, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %144 = phi <8 x float> [ %792, %763 ], [ zeroinitializer, %4 ], !dbg !7
  %145 = icmp slt i64 %130, %124, !dbg !7
  br i1 %145, label %146, label %794, !dbg !7

146:                                              ; preds = %129
  %147 = shl nsw i64 %130, 5, !dbg !7
  %148 = lshr i64 %130, 14, !dbg !18
  %149 = add nuw nsw i64 %148, %16, !dbg !18
  %150 = lshr i64 %130, 7, !dbg !18
  %151 = and i64 %150, 127, !dbg !18
  %152 = shl i64 %149, 19, !dbg !18
  %153 = and i64 %152, 16252928, !dbg !18
  %154 = shl i64 %150, 19, !dbg !18
  %155 = shl i64 %130, 12, !dbg !18
  %156 = sub i64 %155, %154, !dbg !18
  %157 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %153, !dbg !18
  %158 = getelementptr [2 x i8], ptr addrspace(1) %157, i64 %151, !dbg !18
  %159 = getelementptr [2 x i8], ptr addrspace(1) %158, i64 %156, !dbg !18
  br label %160, !dbg !18

160:                                              ; preds = %163, %146
  %161 = phi i64 [ %193, %163 ], [ %32, %146 ], !dbg !18
  %162 = icmp samesign ult i64 %161, 128, !dbg !18
  br i1 %162, label %163, label %194, !dbg !18

163:                                              ; preds = %160
  %164 = and i64 %161, 27, !dbg !18
  %165 = lshr i64 %161, 2, !dbg !18
  %166 = and i64 %165, 4, !dbg !18
  %167 = add nuw nsw i64 %161, %166, !dbg !18
  %168 = and i64 %167, 4, !dbg !18
  %169 = or disjoint i64 %164, %168, !dbg !18
  %170 = lshr i64 %161, 2, !dbg !18
  %171 = and i64 %170, 24, !dbg !18
  %172 = shl nuw nsw i64 %161, 3, !dbg !18
  %173 = and i64 %172, 512, !dbg !18
  %174 = getelementptr [2 x i8], ptr addrspace(1) %159, i64 %171, !dbg !18
  %.idx14 = shl nuw nsw i64 %169, 8, !dbg !18
  %175 = getelementptr i8, ptr addrspace(1) %174, i64 %.idx14, !dbg !18
  %176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %173, !dbg !18
  %177 = ptrtoint ptr addrspace(3) %176 to i64, !dbg !18
  %178 = trunc i64 %177 to i32, !dbg !18
  %179 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %178), !dbg !18
  %180 = zext i32 %179 to i64, !dbg !18
  %181 = inttoptr i64 %180 to ptr addrspace(3), !dbg !18
  %182 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %183 = ptrtoint ptr addrspace(1) %175 to i64, !dbg !18
  %184 = sub i64 %183, %182, !dbg !18
  %dma.buffer.offset = trunc i64 %184 to i32, !dbg !18
  %185 = trunc i64 %182 to i32, !dbg !18
  %186 = insertelement <4 x i32> poison, i32 %185, i64 0, !dbg !18
  %187 = lshr i64 %182, 32, !dbg !18
  %188 = trunc i64 %187 to i32, !dbg !18
  %189 = insertelement <4 x i32> %186, i32 %188, i64 1, !dbg !18
  %190 = insertelement <4 x i32> %189, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %190, i32 131072, i64 3, !dbg !18
  %191 = ptrtoint ptr addrspace(3) %181 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %191), !dbg !18
  %192 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %193 = or disjoint i64 %32, 256, !dbg !18
  br label %160, !dbg !18

194:                                              ; preds = %160
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %195 = add nuw i64 %130, 4096, !dbg !18
  %196 = icmp slt i64 %195, 0, !dbg !18
  %197 = sub nuw i64 -4097, %130, !dbg !18
  %198 = select i1 %196, i64 %197, i64 %195, !dbg !18
  %199 = sdiv i64 %198, 16384, !dbg !18
  %.lobit = ashr i64 %195, 63, !dbg !18
  %200 = xor i64 %199, %.lobit, !dbg !18
  %201 = add nsw i64 %200, %16, !dbg !18
  %202 = add nsw i64 %150, %128, !dbg !18
  %203 = add nsw i64 %202, 32, !dbg !18
  %204 = icmp slt i64 %202, -32, !dbg !18
  %205 = sub nsw i64 -33, %202, !dbg !18
  %206 = select i1 %204, i64 %205, i64 %203, !dbg !18
  %207 = sdiv i64 %206, 128, !dbg !18
  %208 = sext i1 %204 to i64, !dbg !18
  %209 = xor i64 %207, %208, !dbg !18
  %210 = shl nsw i64 %209, 7, !dbg !18
  %211 = sub i64 %128, %210, !dbg !18
  %212 = shl i64 %201, 19, !dbg !18
  %213 = and i64 %212, 16252928, !dbg !18
  %214 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %213, !dbg !18
  %215 = getelementptr [2 x i8], ptr addrspace(1) %214, i64 %150, !dbg !18
  %216 = getelementptr [2 x i8], ptr addrspace(1) %215, i64 %211, !dbg !18
  %217 = getelementptr i8, ptr addrspace(1) %216, i64 64, !dbg !18
  %218 = getelementptr [2 x i8], ptr addrspace(1) %217, i64 %156, !dbg !18
  br label %219, !dbg !18

219:                                              ; preds = %222, %194
  %220 = phi i64 [ %252, %222 ], [ %32, %194 ], !dbg !18
  %221 = icmp samesign ult i64 %220, 128, !dbg !18
  br i1 %221, label %222, label %253, !dbg !18

222:                                              ; preds = %219
  %223 = and i64 %220, 27, !dbg !18
  %224 = lshr i64 %220, 2, !dbg !18
  %225 = and i64 %224, 4, !dbg !18
  %226 = add nuw nsw i64 %220, %225, !dbg !18
  %227 = and i64 %226, 4, !dbg !18
  %228 = or disjoint i64 %223, %227, !dbg !18
  %229 = lshr i64 %220, 2, !dbg !18
  %230 = and i64 %229, 24, !dbg !18
  %231 = shl nuw nsw i64 %220, 3, !dbg !18
  %232 = and i64 %231, 512, !dbg !18
  %233 = getelementptr [2 x i8], ptr addrspace(1) %218, i64 %230, !dbg !18
  %.idx13 = shl nuw nsw i64 %228, 8, !dbg !18
  %234 = getelementptr i8, ptr addrspace(1) %233, i64 %.idx13, !dbg !18
  %235 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 2048), i64 %232, !dbg !18
  %236 = ptrtoint ptr addrspace(3) %235 to i64, !dbg !18
  %237 = trunc i64 %236 to i32, !dbg !18
  %238 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %237), !dbg !18
  %239 = zext i32 %238 to i64, !dbg !18
  %240 = inttoptr i64 %239 to ptr addrspace(3), !dbg !18
  %241 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %242 = ptrtoint ptr addrspace(1) %234 to i64, !dbg !18
  %243 = sub i64 %242, %241, !dbg !18
  %dma.buffer.offset459 = trunc i64 %243 to i32, !dbg !18
  %244 = trunc i64 %241 to i32, !dbg !18
  %245 = insertelement <4 x i32> poison, i32 %244, i64 0, !dbg !18
  %246 = lshr i64 %241, 32, !dbg !18
  %247 = trunc i64 %246 to i32, !dbg !18
  %248 = insertelement <4 x i32> %245, i32 %247, i64 1, !dbg !18
  %249 = insertelement <4 x i32> %248, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc460 = insertelement <4 x i32> %249, i32 131072, i64 3, !dbg !18
  %250 = ptrtoint ptr addrspace(3) %240 to i32, !dbg !18
  %dma.buffer.lds461 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %250), !dbg !18
  %251 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds461, i32 %dma.buffer.offset459, <4 x i32> %dma.buffer.rsrc460) #10, !dbg !18
  %252 = or disjoint i64 %32, 256, !dbg !18
  br label %219, !dbg !18

253:                                              ; preds = %219
  %address.thread85 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !17, !range !40, !frisk.register_lifetimes !1
  %address.local86 = lshr i32 %address.thread85, 2, !dbg !17
  %address.local87 = and i32 %address.local86, 4, !dbg !17
  %address.local88 = shl nuw nsw i32 %address.thread85, 3, !dbg !17
  %address.local89 = and i32 %address.local88, 376, !dbg !17
  %address.local90 = or disjoint i32 %address.local87, %address.local89, !dbg !17
  %address.local91 = zext nneg i32 %address.local90 to i64, !dbg !17
  %address.local92 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local91, !dbg !17
  %254 = load <4 x half>, ptr addrspace(3) %address.local92, align 8, !dbg !17
  %lds.const.gep445 = getelementptr i8, ptr addrspace(3) %address.local92, i64 1024, !dbg !17
  %255 = load <4 x half>, ptr addrspace(3) %lds.const.gep445, align 8, !dbg !17
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %254, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local101 = lshr i32 %address.thread85, 2, !dbg !17
  %address.local102 = and i32 %address.local101, 4, !dbg !17
  %address.local103 = shl nuw nsw i32 %address.thread85, 3, !dbg !17
  %address.local104 = and i32 %address.local103, 344, !dbg !17
  %address.local105 = or disjoint i32 %address.local102, %address.local104, !dbg !17
  %address.local106 = or disjoint i32 %address.local105, 160, !dbg !17
  %address.local107 = shl nuw nsw i32 %address.thread85, 3, !dbg !17
  %address.local108 = and i32 %address.local107, 32, !dbg !17
  %address.local109 = sub nuw nsw i32 %address.local106, %address.local108, !dbg !17
  %address.local110 = zext nneg i32 %address.local109 to i64, !dbg !17
  %address.local111 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local110, !dbg !17
  %257 = load <4 x half>, ptr addrspace(3) %address.local111, align 8, !dbg !17
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %255, <4 x float> %256) #9, !dbg !17
  %lds.const.gep446 = getelementptr i8, ptr addrspace(3) %address.local111, i64 1024, !dbg !17
  %259 = load <4 x half>, ptr addrspace(3) %lds.const.gep446, align 8, !dbg !17
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %257, <4 x float> zeroinitializer) #9, !dbg !17
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %259, <4 x float> %260) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %262 = add nuw i64 %130, 8192, !dbg !18
  %263 = icmp slt i64 %262, 0, !dbg !18
  %264 = sub nuw i64 -8193, %130, !dbg !18
  %265 = select i1 %263, i64 %264, i64 %262, !dbg !18
  %266 = sdiv i64 %265, 16384, !dbg !18
  %.lobit2 = ashr i64 %262, 63, !dbg !18
  %267 = xor i64 %266, %.lobit2, !dbg !18
  %268 = add nsw i64 %267, %16, !dbg !18
  %269 = add nsw i64 %202, 64, !dbg !18
  %270 = icmp slt i64 %202, -64, !dbg !18
  %271 = sub nsw i64 -65, %202, !dbg !18
  %272 = select i1 %270, i64 %271, i64 %269, !dbg !18
  %273 = sdiv i64 %272, 128, !dbg !18
  %274 = sext i1 %270 to i64, !dbg !18
  %275 = xor i64 %273, %274, !dbg !18
  %276 = shl nsw i64 %275, 7, !dbg !18
  %277 = sub i64 %128, %276, !dbg !18
  %278 = shl i64 %268, 19, !dbg !18
  %279 = and i64 %278, 16252928, !dbg !18
  %280 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %279, !dbg !18
  %281 = getelementptr [2 x i8], ptr addrspace(1) %280, i64 %150, !dbg !18
  %282 = getelementptr [2 x i8], ptr addrspace(1) %281, i64 %277, !dbg !18
  %283 = getelementptr i8, ptr addrspace(1) %282, i64 128, !dbg !18
  %284 = getelementptr [2 x i8], ptr addrspace(1) %283, i64 %156, !dbg !18
  br label %285, !dbg !18

285:                                              ; preds = %288, %253
  %286 = phi i64 [ %318, %288 ], [ %32, %253 ], !dbg !18
  %287 = icmp samesign ult i64 %286, 128, !dbg !18
  br i1 %287, label %288, label %319, !dbg !18

288:                                              ; preds = %285
  %289 = and i64 %286, 27, !dbg !18
  %290 = lshr i64 %286, 2, !dbg !18
  %291 = and i64 %290, 4, !dbg !18
  %292 = add nuw nsw i64 %286, %291, !dbg !18
  %293 = and i64 %292, 4, !dbg !18
  %294 = or disjoint i64 %289, %293, !dbg !18
  %295 = lshr i64 %286, 2, !dbg !18
  %296 = and i64 %295, 24, !dbg !18
  %297 = shl nuw nsw i64 %286, 3, !dbg !18
  %298 = and i64 %297, 512, !dbg !18
  %299 = getelementptr [2 x i8], ptr addrspace(1) %284, i64 %296, !dbg !18
  %.idx12 = shl nuw nsw i64 %294, 8, !dbg !18
  %300 = getelementptr i8, ptr addrspace(1) %299, i64 %.idx12, !dbg !18
  %301 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %298, !dbg !18
  %302 = ptrtoint ptr addrspace(3) %301 to i64, !dbg !18
  %303 = trunc i64 %302 to i32, !dbg !18
  %304 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %303), !dbg !18
  %305 = zext i32 %304 to i64, !dbg !18
  %306 = inttoptr i64 %305 to ptr addrspace(3), !dbg !18
  %307 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %308 = ptrtoint ptr addrspace(1) %300 to i64, !dbg !18
  %309 = sub i64 %308, %307, !dbg !18
  %dma.buffer.offset462 = trunc i64 %309 to i32, !dbg !18
  %310 = trunc i64 %307 to i32, !dbg !18
  %311 = insertelement <4 x i32> poison, i32 %310, i64 0, !dbg !18
  %312 = lshr i64 %307, 32, !dbg !18
  %313 = trunc i64 %312 to i32, !dbg !18
  %314 = insertelement <4 x i32> %311, i32 %313, i64 1, !dbg !18
  %315 = insertelement <4 x i32> %314, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc463 = insertelement <4 x i32> %315, i32 131072, i64 3, !dbg !18
  %316 = ptrtoint ptr addrspace(3) %306 to i32, !dbg !18
  %dma.buffer.lds464 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %316), !dbg !18
  %317 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds464, i32 %dma.buffer.offset462, <4 x i32> %dma.buffer.rsrc463) #10, !dbg !18
  %318 = or disjoint i64 %32, 256, !dbg !18
  br label %285, !dbg !18

319:                                              ; preds = %285
  %lds.const.gep447 = getelementptr i8, ptr addrspace(3) %address.local92, i64 2048, !dbg !17
  %320 = load <4 x half>, ptr addrspace(3) %lds.const.gep447, align 8, !dbg !17
  %lds.const.gep448 = getelementptr i8, ptr addrspace(3) %address.local92, i64 3072, !dbg !17
  %321 = load <4 x half>, ptr addrspace(3) %lds.const.gep448, align 8, !dbg !17
  %322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %320, <4 x float> %258) #9, !dbg !17
  %lds.const.gep449 = getelementptr i8, ptr addrspace(3) %address.local111, i64 2048, !dbg !17
  %323 = load <4 x half>, ptr addrspace(3) %lds.const.gep449, align 8, !dbg !17
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %321, <4 x float> %322) #9, !dbg !17
  %lds.const.gep450 = getelementptr i8, ptr addrspace(3) %address.local111, i64 3072, !dbg !17
  %325 = load <4 x half>, ptr addrspace(3) %lds.const.gep450, align 8, !dbg !17
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %323, <4 x float> %261) #9, !dbg !17
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %325, <4 x float> %326) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %328 = add nuw i64 %130, 12288, !dbg !18
  %329 = icmp slt i64 %328, 0, !dbg !18
  %330 = sub nuw i64 -12289, %130, !dbg !18
  %331 = select i1 %329, i64 %330, i64 %328, !dbg !18
  %332 = sdiv i64 %331, 16384, !dbg !18
  %.lobit3 = ashr i64 %328, 63, !dbg !18
  %333 = xor i64 %332, %.lobit3, !dbg !18
  %334 = add nsw i64 %333, %16, !dbg !18
  %335 = add nsw i64 %202, 96, !dbg !18
  %336 = icmp slt i64 %202, -96, !dbg !18
  %337 = sub nsw i64 -97, %202, !dbg !18
  %338 = select i1 %336, i64 %337, i64 %335, !dbg !18
  %339 = sdiv i64 %338, 128, !dbg !18
  %340 = sext i1 %336 to i64, !dbg !18
  %341 = xor i64 %339, %340, !dbg !18
  %342 = shl nsw i64 %341, 7, !dbg !18
  %343 = sub i64 %128, %342, !dbg !18
  %344 = shl i64 %334, 19, !dbg !18
  %345 = and i64 %344, 16252928, !dbg !18
  %346 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %345, !dbg !18
  %347 = getelementptr [2 x i8], ptr addrspace(1) %346, i64 %150, !dbg !18
  %348 = getelementptr [2 x i8], ptr addrspace(1) %347, i64 %343, !dbg !18
  %349 = getelementptr i8, ptr addrspace(1) %348, i64 192, !dbg !18
  %350 = getelementptr [2 x i8], ptr addrspace(1) %349, i64 %156, !dbg !18
  br label %351, !dbg !18

351:                                              ; preds = %354, %319
  %352 = phi i64 [ %384, %354 ], [ %32, %319 ], !dbg !18
  %353 = icmp samesign ult i64 %352, 128, !dbg !18
  br i1 %353, label %354, label %385, !dbg !18

354:                                              ; preds = %351
  %355 = and i64 %352, 27, !dbg !18
  %356 = lshr i64 %352, 2, !dbg !18
  %357 = and i64 %356, 4, !dbg !18
  %358 = add nuw nsw i64 %352, %357, !dbg !18
  %359 = and i64 %358, 4, !dbg !18
  %360 = or disjoint i64 %355, %359, !dbg !18
  %361 = lshr i64 %352, 2, !dbg !18
  %362 = and i64 %361, 24, !dbg !18
  %363 = shl nuw nsw i64 %352, 3, !dbg !18
  %364 = and i64 %363, 512, !dbg !18
  %365 = getelementptr [2 x i8], ptr addrspace(1) %350, i64 %362, !dbg !18
  %.idx = shl nuw nsw i64 %360, 8, !dbg !18
  %366 = getelementptr i8, ptr addrspace(1) %365, i64 %.idx, !dbg !18
  %367 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 2048), i64 %364, !dbg !18
  %368 = ptrtoint ptr addrspace(3) %367 to i64, !dbg !18
  %369 = trunc i64 %368 to i32, !dbg !18
  %370 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %369), !dbg !18
  %371 = zext i32 %370 to i64, !dbg !18
  %372 = inttoptr i64 %371 to ptr addrspace(3), !dbg !18
  %373 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !18
  %374 = ptrtoint ptr addrspace(1) %366 to i64, !dbg !18
  %375 = sub i64 %374, %373, !dbg !18
  %dma.buffer.offset465 = trunc i64 %375 to i32, !dbg !18
  %376 = trunc i64 %373 to i32, !dbg !18
  %377 = insertelement <4 x i32> poison, i32 %376, i64 0, !dbg !18
  %378 = lshr i64 %373, 32, !dbg !18
  %379 = trunc i64 %378 to i32, !dbg !18
  %380 = insertelement <4 x i32> %377, i32 %379, i64 1, !dbg !18
  %381 = insertelement <4 x i32> %380, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc466 = insertelement <4 x i32> %381, i32 131072, i64 3, !dbg !18
  %382 = ptrtoint ptr addrspace(3) %372 to i32, !dbg !18
  %dma.buffer.lds467 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %382), !dbg !18
  %383 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds467, i32 %dma.buffer.offset465, <4 x i32> %dma.buffer.rsrc466) #10, !dbg !18
  %384 = or disjoint i64 %32, 256, !dbg !18
  br label %351, !dbg !18

385:                                              ; preds = %351
  %lds.const.gep451 = getelementptr i8, ptr addrspace(3) %address.local92, i64 0, !dbg !17
  %386 = load <4 x half>, ptr addrspace(3) %lds.const.gep451, align 8, !dbg !17
  %lds.const.gep452 = getelementptr i8, ptr addrspace(3) %address.local92, i64 1024, !dbg !17
  %387 = load <4 x half>, ptr addrspace(3) %lds.const.gep452, align 8, !dbg !17
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %386, <4 x float> %324) #9, !dbg !17
  %lds.const.gep453 = getelementptr i8, ptr addrspace(3) %address.local111, i64 0, !dbg !17
  %389 = load <4 x half>, ptr addrspace(3) %lds.const.gep453, align 8, !dbg !17
  %390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %387, <4 x float> %388) #9, !dbg !17
  %lds.const.gep454 = getelementptr i8, ptr addrspace(3) %address.local111, i64 1024, !dbg !17
  %391 = load <4 x half>, ptr addrspace(3) %lds.const.gep454, align 8, !dbg !17
  %392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %389, <4 x float> %327) #9, !dbg !17
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %113, <4 x half> %391, <4 x float> %392) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %394 = add nuw nsw i64 %150, %16, !dbg !19
  %395 = shl i64 %394, 19, !dbg !19
  %396 = and i64 %395, 16252928, !dbg !19
  %397 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %396, !dbg !19
  %398 = getelementptr [2 x i8], ptr addrspace(1) %397, i64 %156, !dbg !19
  br label %399, !dbg !19

399:                                              ; preds = %402, %385
  %400 = phi i64 [ %428, %402 ], [ %32, %385 ], !dbg !19
  %401 = icmp samesign ult i64 %400, 128, !dbg !19
  br i1 %401, label %402, label %429, !dbg !19

402:                                              ; preds = %399
  %403 = shl nuw nsw i64 %400, 3, !dbg !19
  %404 = and i64 %403, 24, !dbg !19
  %405 = shl nuw nsw i64 %400, 3, !dbg !19
  %406 = and i64 %405, 512, !dbg !19
  %407 = shl nuw nsw i64 %400, 5, !dbg !19
  %408 = and i64 %407, 3968, !dbg !19
  %409 = getelementptr [2 x i8], ptr addrspace(1) %398, i64 %408, !dbg !19
  %410 = getelementptr [2 x i8], ptr addrspace(1) %409, i64 %404, !dbg !19
  %411 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %406, !dbg !19
  %412 = ptrtoint ptr addrspace(3) %411 to i64, !dbg !19
  %413 = trunc i64 %412 to i32, !dbg !19
  %414 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %413), !dbg !19
  %415 = zext i32 %414 to i64, !dbg !19
  %416 = inttoptr i64 %415 to ptr addrspace(3), !dbg !19
  %417 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %418 = ptrtoint ptr addrspace(1) %410 to i64, !dbg !19
  %419 = sub i64 %418, %417, !dbg !19
  %dma.buffer.offset468 = trunc i64 %419 to i32, !dbg !19
  %420 = trunc i64 %417 to i32, !dbg !19
  %421 = insertelement <4 x i32> poison, i32 %420, i64 0, !dbg !19
  %422 = lshr i64 %417, 32, !dbg !19
  %423 = trunc i64 %422 to i32, !dbg !19
  %424 = insertelement <4 x i32> %421, i32 %423, i64 1, !dbg !19
  %425 = insertelement <4 x i32> %424, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc469 = insertelement <4 x i32> %425, i32 131072, i64 3, !dbg !19
  %426 = ptrtoint ptr addrspace(3) %416 to i32, !dbg !19
  %dma.buffer.lds470 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %426), !dbg !19
  %427 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds470, i32 %dma.buffer.offset468, <4 x i32> %dma.buffer.rsrc469) #10, !dbg !19
  %428 = or disjoint i64 %32, 256, !dbg !19
  br label %399, !dbg !19

429:                                              ; preds = %399
  %lds.const.gep455 = getelementptr i8, ptr addrspace(3) %address.local92, i64 2048, !dbg !17
  %430 = load <4 x half>, ptr addrspace(3) %lds.const.gep455, align 8, !dbg !17
  %lds.const.gep456 = getelementptr i8, ptr addrspace(3) %address.local92, i64 3072, !dbg !17
  %431 = load <4 x half>, ptr addrspace(3) %lds.const.gep456, align 8, !dbg !17
  %432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %430, <4 x float> %390) #9, !dbg !17
  %lds.const.gep457 = getelementptr i8, ptr addrspace(3) %address.local111, i64 2048, !dbg !17
  %433 = load <4 x half>, ptr addrspace(3) %lds.const.gep457, align 8, !dbg !17
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %431, <4 x float> %432) #9, !dbg !17
  %lds.const.gep458 = getelementptr i8, ptr addrspace(3) %address.local111, i64 3072, !dbg !17
  %435 = load <4 x half>, ptr addrspace(3) %lds.const.gep458, align 8, !dbg !17
  %436 = extractelement <4 x float> %434, i64 0, !dbg !23
  %437 = extractelement <4 x float> %434, i64 1, !dbg !23
  %438 = select i1 %.not, float %437, float %436, !dbg !14
  %439 = extractelement <4 x float> %434, i64 2, !dbg !23
  %440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %433, <4 x float> %393) #9, !dbg !17
  %441 = extractelement <4 x float> %434, i64 3, !dbg !23
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %435, <4 x float> %440) #9, !dbg !17
  %443 = extractelement <4 x float> %442, i64 0, !dbg !23
  %444 = extractelement <4 x float> %442, i64 1, !dbg !23
  %445 = extractelement <4 x float> %442, i64 2, !dbg !23
  %446 = extractelement <4 x float> %442, i64 3, !dbg !23
  %447 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !14
  %448 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %447), !dbg !14
  %449 = add nuw nsw i32 %448, 64, !dbg !14
  %450 = xor i32 %448, 16, !dbg !14
  %451 = and i32 %449, 192, !dbg !14
  %452 = icmp samesign ult i32 %450, %451, !dbg !14
  %453 = select i1 %452, i32 %450, i32 %448, !dbg !14
  %454 = shl nuw nsw i32 %453, 2, !dbg !14
  %455 = bitcast float %438 to i32, !dbg !14
  %456 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %455), !dbg !14
  %457 = bitcast i32 %456 to float, !dbg !14
  %458 = select i1 %.not, float %436, float %457, !dbg !14
  %459 = select i1 %.not, float %457, float %437, !dbg !14
  %460 = select i1 %.not, float %441, float %439, !dbg !14
  %461 = bitcast float %460 to i32, !dbg !14
  %462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %461), !dbg !14
  %463 = bitcast i32 %462 to float, !dbg !14
  %464 = select i1 %.not, float %439, float %463, !dbg !14
  %465 = select i1 %.not, float %463, float %441, !dbg !14
  %466 = select i1 %.not1, float %464, float %458, !dbg !14
  %467 = xor i32 %448, 32, !dbg !14
  %468 = icmp samesign ult i32 %467, %451, !dbg !14
  %469 = select i1 %468, i32 %467, i32 %448, !dbg !14
  %470 = shl nuw nsw i32 %469, 2, !dbg !14
  %471 = bitcast float %466 to i32, !dbg !14
  %472 = call i32 @llvm.amdgcn.ds.bpermute(i32 %470, i32 %471), !dbg !14
  %473 = bitcast i32 %472 to float, !dbg !14
  %474 = select i1 %.not1, float %458, float %473, !dbg !14
  %475 = select i1 %.not1, float %473, float %464, !dbg !14
  %476 = select i1 %.not1, float %465, float %459, !dbg !14
  %477 = bitcast float %476 to i32, !dbg !14
  %478 = call i32 @llvm.amdgcn.ds.bpermute(i32 %470, i32 %477), !dbg !14
  %479 = bitcast i32 %478 to float, !dbg !14
  %480 = select i1 %.not1, float %459, float %479, !dbg !14
  %481 = select i1 %.not1, float %479, float %465, !dbg !14
  %482 = select i1 %.not, float %444, float %443, !dbg !14
  %483 = bitcast float %482 to i32, !dbg !14
  %484 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %483), !dbg !14
  %485 = bitcast i32 %484 to float, !dbg !14
  %486 = select i1 %.not, float %443, float %485, !dbg !14
  %487 = select i1 %.not, float %485, float %444, !dbg !14
  %488 = select i1 %.not, float %446, float %445, !dbg !14
  %489 = bitcast float %488 to i32, !dbg !14
  %490 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %489), !dbg !14
  %491 = bitcast i32 %490 to float, !dbg !14
  %492 = select i1 %.not, float %445, float %491, !dbg !14
  %493 = select i1 %.not, float %491, float %446, !dbg !14
  %494 = select i1 %.not1, float %492, float %486, !dbg !14
  %495 = bitcast float %494 to i32, !dbg !14
  %496 = call i32 @llvm.amdgcn.ds.bpermute(i32 %470, i32 %495), !dbg !14
  %497 = bitcast i32 %496 to float, !dbg !14
  %498 = select i1 %.not1, float %486, float %497, !dbg !14
  %499 = select i1 %.not1, float %497, float %492, !dbg !14
  %500 = select i1 %.not1, float %493, float %487, !dbg !14
  %501 = bitcast float %500 to i32, !dbg !14
  %502 = call i32 @llvm.amdgcn.ds.bpermute(i32 %470, i32 %501), !dbg !14
  %503 = bitcast i32 %502 to float, !dbg !14
  %504 = select i1 %.not1, float %487, float %503, !dbg !14
  %505 = select i1 %.not1, float %503, float %493, !dbg !14
  %506 = or disjoint i64 %147, 31, !dbg !15
  %507 = icmp uge i64 %50, %506, !dbg !15
  %508 = and i1 %122, %507, !dbg !15
  br i1 %508, label %509, label %510, !dbg !15

509:                                              ; preds = %429
  br label %536, !dbg !15

510:                                              ; preds = %429
  %511 = add i64 %50, %32, !dbg !15
  %512 = shl nuw nsw i64 %51, 4, !dbg !15
  %513 = add i64 %511, %512, !dbg !15
  %514 = and i32 %31, 240, !dbg !15
  %index.compact = zext nneg i32 %514 to i64, !dbg !15
  %515 = sub i64 %513, %index.compact, !dbg !15
  %516 = add i64 %515, 1, !dbg !24
  %517 = or disjoint i64 %147, %53, !dbg !15
  %518 = lshr i64 %516, 32, !dbg !25
  %compare.high = trunc i64 %518 to i32, !dbg !25
  %519 = lshr i64 %517, 32, !dbg !25
  %compare.high412 = trunc i64 %519 to i32, !dbg !25
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high412, !dbg !25
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high412, !dbg !25
  %compare.low.base = trunc i64 %516 to i32, !dbg !25
  %compare.low.base413 = trunc i64 %517 to i32, !dbg !25
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base413, !dbg !25
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !25
  %520 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base413, i32 1) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !25
  %compare.low414 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !25
  %compare.result415 = select i1 %compare.high.equal, i1 %compare.low414, i1 %compare.high.order, !dbg !25
  %521 = select i1 %compare.result415, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base416 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base413, i32 2) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local417 = or i32 %compare.local.base416, 2, !dbg !25
  %compare.low418 = icmp ugt i32 %compare.low.base, %compare.local417, !dbg !25
  %compare.result419 = select i1 %compare.high.equal, i1 %compare.low418, i1 %compare.high.order, !dbg !25
  %522 = select i1 %compare.result419, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base420 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base413, i32 3) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local421 = or i32 %compare.local.base420, 3, !dbg !25
  %compare.low422 = icmp ugt i32 %compare.low.base, %compare.local421, !dbg !25
  %compare.result423 = select i1 %compare.high.equal, i1 %compare.low422, i1 %compare.high.order, !dbg !25
  %523 = select i1 %compare.result423, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base424 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base413, i32 4) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local425 = or i32 %compare.local.base424, 16, !dbg !25
  %compare.low426 = icmp ugt i32 %compare.low.base, %compare.local425, !dbg !25
  %compare.result427 = select i1 %compare.high.equal, i1 %compare.low426, i1 %compare.high.order, !dbg !25
  %524 = select i1 %compare.result427, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base428 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base413, i32 5) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local429 = or i32 %compare.local.base428, 17, !dbg !25
  %compare.low430 = icmp ugt i32 %compare.low.base, %compare.local429, !dbg !25
  %compare.result431 = select i1 %compare.high.equal, i1 %compare.low430, i1 %compare.high.order, !dbg !25
  %525 = select i1 %compare.result431, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base432 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base413, i32 6) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local433 = or i32 %compare.local.base432, 18, !dbg !25
  %compare.low434 = icmp ugt i32 %compare.low.base, %compare.local433, !dbg !25
  %compare.result435 = select i1 %compare.high.equal, i1 %compare.low434, i1 %compare.high.order, !dbg !25
  %526 = select i1 %compare.result435, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base436 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base413, i32 7) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local437 = or i32 %compare.local.base436, 19, !dbg !25
  %compare.low438 = icmp ugt i32 %compare.low.base, %compare.local437, !dbg !25
  %compare.result439 = select i1 %compare.high.equal, i1 %compare.low438, i1 %compare.high.order, !dbg !25
  %527 = select i1 %compare.result439, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %528 = insertelement <8 x float> poison, float %520, i64 0, !dbg !15
  %529 = insertelement <8 x float> %528, float %521, i64 1, !dbg !15
  %530 = insertelement <8 x float> %529, float %522, i64 2, !dbg !15
  %531 = insertelement <8 x float> %530, float %523, i64 3, !dbg !15
  %532 = insertelement <8 x float> %531, float %524, i64 4, !dbg !15
  %533 = insertelement <8 x float> %532, float %525, i64 5, !dbg !15
  %534 = insertelement <8 x float> %533, float %526, i64 6, !dbg !15
  %535 = insertelement <8 x float> %534, float %527, i64 7, !dbg !15
  br label %536, !dbg !15

536:                                              ; preds = %509, %510
  %537 = phi <8 x float> [ %535, %510 ], [ zeroinitializer, %509 ], !dbg !15
  br label %538, !dbg !15

538:                                              ; preds = %536
  %539 = insertelement <4 x float> poison, float %474, i64 0, !dbg !27
  %540 = insertelement <4 x float> %539, float %480, i64 1, !dbg !27
  %541 = insertelement <4 x float> %540, float %475, i64 2, !dbg !27
  %542 = insertelement <4 x float> %541, float %481, i64 3, !dbg !27
  %543 = shufflevector <8 x float> %537, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %544 = fadd <4 x float> %542, %543, !dbg !27
  %545 = extractelement <4 x float> %544, i64 0, !dbg !27
  %546 = extractelement <4 x float> %544, i64 1, !dbg !27
  %547 = extractelement <4 x float> %544, i64 2, !dbg !27
  %548 = extractelement <4 x float> %544, i64 3, !dbg !27
  %549 = insertelement <4 x float> poison, float %498, i64 0, !dbg !27
  %550 = insertelement <4 x float> %549, float %504, i64 1, !dbg !27
  %551 = insertelement <4 x float> %550, float %499, i64 2, !dbg !27
  %552 = insertelement <4 x float> %551, float %505, i64 3, !dbg !27
  %553 = shufflevector <8 x float> %537, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %554 = fadd <4 x float> %552, %553, !dbg !27
  %555 = extractelement <4 x float> %554, i64 0, !dbg !27
  %556 = extractelement <4 x float> %554, i64 1, !dbg !27
  %557 = extractelement <4 x float> %554, i64 2, !dbg !27
  %558 = extractelement <4 x float> %554, i64 3, !dbg !27
  %559 = call float @llvm.maxnum.f32(float %545, float 0xFFF0000000000000), !dbg !28
  %560 = call float @llvm.maxnum.f32(float %559, float %546), !dbg !28
  %561 = call float @llvm.maxnum.f32(float %560, float %547), !dbg !28
  %562 = call float @llvm.maxnum.f32(float %561, float %548), !dbg !28
  %563 = call float @llvm.maxnum.f32(float %562, float %555), !dbg !28
  %564 = call float @llvm.maxnum.f32(float %563, float %556), !dbg !28
  %565 = call float @llvm.maxnum.f32(float %564, float %557), !dbg !28
  %566 = call float @llvm.maxnum.f32(float %565, float %558), !dbg !28
  %567 = bitcast float %566 to i32, !dbg !28
  %568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %567), !dbg !28
  %569 = bitcast i32 %568 to float, !dbg !28
  %570 = call float @llvm.maxnum.f32(float %566, float %569), !dbg !28
  %571 = bitcast float %570 to i32, !dbg !28
  %572 = call i32 @llvm.amdgcn.ds.bpermute(i32 %470, i32 %571), !dbg !28
  %573 = bitcast i32 %572 to float, !dbg !28
  %574 = call float @llvm.maxnum.f32(float %570, float %573), !dbg !28
  %575 = insertelement <1 x float> poison, float %574, i64 0, !dbg !28
  %576 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %575, <1 x float> %135), !dbg !28
  %577 = shufflevector <1 x float> %576, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !29
  %578 = fsub <4 x float> %544, %577, !dbg !29
  %579 = fsub <4 x float> %554, %577, !dbg !29
  %580 = fsub <1 x float> %135, %576, !dbg !30
  %581 = extractelement <1 x float> %580, i64 0, !dbg !31
  %582 = call float @llvm.amdgcn.exp2.f32(float %581), !dbg !31
  %583 = extractelement <1 x float> %134, i64 0, !dbg !32
  %584 = fmul float %583, %582, !dbg !32
  %585 = extractelement <4 x float> %578, i64 0, !dbg !33
  %586 = call float @llvm.amdgcn.exp2.f32(float %585), !dbg !33
  %587 = insertelement <4 x float> poison, float %586, i64 0, !dbg !35
  %588 = extractelement <4 x float> %578, i64 1, !dbg !33
  %589 = call float @llvm.amdgcn.exp2.f32(float %588), !dbg !33
  %590 = insertelement <4 x float> %587, float %589, i64 1, !dbg !35
  %591 = fadd float %586, %589, !dbg !34
  %592 = extractelement <4 x float> %578, i64 2, !dbg !33
  %593 = call float @llvm.amdgcn.exp2.f32(float %592), !dbg !33
  %594 = insertelement <4 x float> %590, float %593, i64 2, !dbg !35
  %595 = fadd float %591, %593, !dbg !34
  %596 = extractelement <4 x float> %578, i64 3, !dbg !33
  %597 = call float @llvm.amdgcn.exp2.f32(float %596), !dbg !33
  %598 = insertelement <4 x float> %594, float %597, i64 3, !dbg !35
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %598) #9, !dbg !35, !frisk.register_lifetimes !1
  %599 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !35
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %599) #9, !dbg !34, !frisk.register_lifetimes !1
  %600 = fadd float %595, %597, !dbg !34
  %601 = extractelement <4 x float> %579, i64 0, !dbg !33
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !33
  %603 = insertelement <4 x float> poison, float %602, i64 0, !dbg !35
  %604 = fadd float %600, %602, !dbg !34
  %605 = extractelement <4 x float> %579, i64 1, !dbg !33
  %606 = call float @llvm.amdgcn.exp2.f32(float %605), !dbg !33
  %607 = insertelement <4 x float> %603, float %606, i64 1, !dbg !35
  %608 = fadd float %604, %606, !dbg !34
  %609 = extractelement <4 x float> %579, i64 2, !dbg !33
  %610 = call float @llvm.amdgcn.exp2.f32(float %609), !dbg !33
  %611 = insertelement <4 x float> %607, float %610, i64 2, !dbg !35
  %612 = fadd float %608, %610, !dbg !34
  %613 = extractelement <4 x float> %579, i64 3, !dbg !33
  %614 = call float @llvm.amdgcn.exp2.f32(float %613), !dbg !33
  %615 = insertelement <4 x float> %611, float %614, i64 3, !dbg !35
  %math.result37 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %615) #9, !dbg !35, !frisk.register_lifetimes !1
  %616 = fptrunc <4 x float> %math.result37 to <4 x half>, !dbg !35
  %math.packed38 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %616) #9, !dbg !34, !frisk.register_lifetimes !1
  %617 = fadd float %612, %614, !dbg !34
  %618 = bitcast float %617 to i32, !dbg !34
  %619 = call i32 @llvm.amdgcn.ds.bpermute(i32 %454, i32 %618), !dbg !34
  %620 = bitcast i32 %619 to float, !dbg !34
  %621 = fadd float %617, %620, !dbg !34
  %622 = bitcast float %621 to i32, !dbg !34
  %623 = call i32 @llvm.amdgcn.ds.bpermute(i32 %470, i32 %622), !dbg !34
  %624 = bitcast i32 %623 to float, !dbg !34
  %625 = fadd float %621, %624, !dbg !34
  %626 = fadd float %625, %584, !dbg !34
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread237 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local238 = shl nuw nsw i32 %address.thread237, 3, !dbg !16
  %address.local239 = and i32 %address.local238, 504, !dbg !16
  %address.local240 = zext nneg i32 %address.local239 to i64, !dbg !16
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local240, !dbg !16
  %627 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local241, i32 0) #0, !dbg !16
  %628 = bitcast <4 x i32> %627 to <8 x half>, !dbg !16
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local240, !dbg !16
  %address.local243 = getelementptr i8, ptr addrspace(3) %address.local242, i64 1024, !dbg !16
  %629 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local243, i32 0) #0, !dbg !16
  %630 = bitcast <4 x i32> %629 to <8 x half>, !dbg !16
  %631 = add nuw nsw i64 %150, %16, !dbg !19
  %632 = shl i64 %631, 19, !dbg !19
  %633 = and i64 %632, 16252928, !dbg !19
  %634 = add i64 %633, %156, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %635 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %634, !dbg !19
  %636 = getelementptr i8, ptr addrspace(1) %635, i64 64, !dbg !19
  br label %637, !dbg !19

637:                                              ; preds = %640, %538
  %638 = phi i64 [ %666, %640 ], [ %32, %538 ], !dbg !19
  %639 = icmp samesign ult i64 %638, 128, !dbg !19
  br i1 %639, label %640, label %667, !dbg !19

640:                                              ; preds = %637
  %641 = shl nuw nsw i64 %638, 3, !dbg !19
  %642 = and i64 %641, 24, !dbg !19
  %643 = shl nuw nsw i64 %638, 3, !dbg !19
  %644 = and i64 %643, 512, !dbg !19
  %645 = shl nuw nsw i64 %638, 5, !dbg !19
  %646 = and i64 %645, 3968, !dbg !19
  %647 = getelementptr [2 x i8], ptr addrspace(1) %636, i64 %646, !dbg !19
  %648 = getelementptr [2 x i8], ptr addrspace(1) %647, i64 %642, !dbg !19
  %649 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %644, !dbg !19
  %650 = ptrtoint ptr addrspace(3) %649 to i64, !dbg !19
  %651 = trunc i64 %650 to i32, !dbg !19
  %652 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %651), !dbg !19
  %653 = zext i32 %652 to i64, !dbg !19
  %654 = inttoptr i64 %653 to ptr addrspace(3), !dbg !19
  %655 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %656 = ptrtoint ptr addrspace(1) %648 to i64, !dbg !19
  %657 = sub i64 %656, %655, !dbg !19
  %dma.buffer.offset471 = trunc i64 %657 to i32, !dbg !19
  %658 = trunc i64 %655 to i32, !dbg !19
  %659 = insertelement <4 x i32> poison, i32 %658, i64 0, !dbg !19
  %660 = lshr i64 %655, 32, !dbg !19
  %661 = trunc i64 %660 to i32, !dbg !19
  %662 = insertelement <4 x i32> %659, i32 %661, i64 1, !dbg !19
  %663 = insertelement <4 x i32> %662, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc472 = insertelement <4 x i32> %663, i32 131072, i64 3, !dbg !19
  %664 = ptrtoint ptr addrspace(3) %654 to i32, !dbg !19
  %dma.buffer.lds473 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %664), !dbg !19
  %665 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds473, i32 %dma.buffer.offset471, <4 x i32> %dma.buffer.rsrc472) #10, !dbg !19
  %666 = or disjoint i64 %32, 256, !dbg !19
  br label %637, !dbg !19

667:                                              ; preds = %637
  %668 = shufflevector <8 x half> %628, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %669 = insertelement <4 x float> poison, float %582, i64 0, !dbg !36
  %670 = shufflevector <4 x float> %669, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %671 = fmul <4 x float> %131, %670, !dbg !36
  %672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %668, <4 x float> %671) #9, !dbg !16
  %673 = shufflevector <8 x half> %630, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %673, <4 x float> %672) #9, !dbg !16
  %675 = shufflevector <8 x half> %628, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %676 = fmul <4 x float> %132, %670, !dbg !36
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %675, <4 x float> %676) #9, !dbg !16
  %678 = shufflevector <8 x half> %630, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %678, <4 x float> %677) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread244 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local245 = shl nuw nsw i32 %address.thread244, 3, !dbg !16
  %address.local246 = and i32 %address.local245, 504, !dbg !16
  %address.local247 = zext nneg i32 %address.local246 to i64, !dbg !16
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local247, !dbg !16
  %680 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local248, i32 0) #0, !dbg !16
  %681 = bitcast <4 x i32> %680 to <8 x half>, !dbg !16
  %address.local249 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local247, !dbg !16
  %address.local250 = getelementptr i8, ptr addrspace(3) %address.local249, i64 1024, !dbg !16
  %682 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local250, i32 0) #0, !dbg !16
  %683 = bitcast <4 x i32> %682 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %684 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %634, !dbg !19
  %685 = getelementptr i8, ptr addrspace(1) %684, i64 128, !dbg !19
  br label %686, !dbg !19

686:                                              ; preds = %689, %667
  %687 = phi i64 [ %715, %689 ], [ %32, %667 ], !dbg !19
  %688 = icmp samesign ult i64 %687, 128, !dbg !19
  br i1 %688, label %689, label %716, !dbg !19

689:                                              ; preds = %686
  %690 = shl nuw nsw i64 %687, 3, !dbg !19
  %691 = and i64 %690, 24, !dbg !19
  %692 = shl nuw nsw i64 %687, 3, !dbg !19
  %693 = and i64 %692, 512, !dbg !19
  %694 = shl nuw nsw i64 %687, 5, !dbg !19
  %695 = and i64 %694, 3968, !dbg !19
  %696 = getelementptr [2 x i8], ptr addrspace(1) %685, i64 %695, !dbg !19
  %697 = getelementptr [2 x i8], ptr addrspace(1) %696, i64 %691, !dbg !19
  %698 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %693, !dbg !19
  %699 = ptrtoint ptr addrspace(3) %698 to i64, !dbg !19
  %700 = trunc i64 %699 to i32, !dbg !19
  %701 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %700), !dbg !19
  %702 = zext i32 %701 to i64, !dbg !19
  %703 = inttoptr i64 %702 to ptr addrspace(3), !dbg !19
  %704 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %705 = ptrtoint ptr addrspace(1) %697 to i64, !dbg !19
  %706 = sub i64 %705, %704, !dbg !19
  %dma.buffer.offset474 = trunc i64 %706 to i32, !dbg !19
  %707 = trunc i64 %704 to i32, !dbg !19
  %708 = insertelement <4 x i32> poison, i32 %707, i64 0, !dbg !19
  %709 = lshr i64 %704, 32, !dbg !19
  %710 = trunc i64 %709 to i32, !dbg !19
  %711 = insertelement <4 x i32> %708, i32 %710, i64 1, !dbg !19
  %712 = insertelement <4 x i32> %711, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc475 = insertelement <4 x i32> %712, i32 131072, i64 3, !dbg !19
  %713 = ptrtoint ptr addrspace(3) %703 to i32, !dbg !19
  %dma.buffer.lds476 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %713), !dbg !19
  %714 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds476, i32 %dma.buffer.offset474, <4 x i32> %dma.buffer.rsrc475) #10, !dbg !19
  %715 = or disjoint i64 %32, 256, !dbg !19
  br label %686, !dbg !19

716:                                              ; preds = %686
  %717 = shufflevector <8 x half> %681, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %718 = fmul <4 x float> %136, %670, !dbg !36
  %719 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %717, <4 x float> %718) #9, !dbg !16
  %720 = shufflevector <8 x half> %683, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %721 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %720, <4 x float> %719) #9, !dbg !16
  %722 = shufflevector <8 x half> %681, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %723 = fmul <4 x float> %137, %670, !dbg !36
  %724 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %722, <4 x float> %723) #9, !dbg !16
  %725 = shufflevector <8 x half> %683, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %726 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %725, <4 x float> %724) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread251 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local252 = shl nuw nsw i32 %address.thread251, 3, !dbg !16
  %address.local253 = and i32 %address.local252, 504, !dbg !16
  %address.local254 = zext nneg i32 %address.local253 to i64, !dbg !16
  %address.local255 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local254, !dbg !16
  %727 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local255, i32 0) #0, !dbg !16
  %728 = bitcast <4 x i32> %727 to <8 x half>, !dbg !16
  %address.local256 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local254, !dbg !16
  %address.local257 = getelementptr i8, ptr addrspace(3) %address.local256, i64 1024, !dbg !16
  %729 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local257, i32 0) #0, !dbg !16
  %730 = bitcast <4 x i32> %729 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %731 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %634, !dbg !19
  %732 = getelementptr i8, ptr addrspace(1) %731, i64 192, !dbg !19
  br label %733, !dbg !19

733:                                              ; preds = %736, %716
  %734 = phi i64 [ %762, %736 ], [ %32, %716 ], !dbg !19
  %735 = icmp samesign ult i64 %734, 128, !dbg !19
  br i1 %735, label %736, label %763, !dbg !19

736:                                              ; preds = %733
  %737 = shl nuw nsw i64 %734, 3, !dbg !19
  %738 = and i64 %737, 24, !dbg !19
  %739 = shl nuw nsw i64 %734, 3, !dbg !19
  %740 = and i64 %739, 512, !dbg !19
  %741 = shl nuw nsw i64 %734, 5, !dbg !19
  %742 = and i64 %741, 3968, !dbg !19
  %743 = getelementptr [2 x i8], ptr addrspace(1) %732, i64 %742, !dbg !19
  %744 = getelementptr [2 x i8], ptr addrspace(1) %743, i64 %738, !dbg !19
  %745 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %740, !dbg !19
  %746 = ptrtoint ptr addrspace(3) %745 to i64, !dbg !19
  %747 = trunc i64 %746 to i32, !dbg !19
  %748 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %747), !dbg !19
  %749 = zext i32 %748 to i64, !dbg !19
  %750 = inttoptr i64 %749 to ptr addrspace(3), !dbg !19
  %751 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %752 = ptrtoint ptr addrspace(1) %744 to i64, !dbg !19
  %753 = sub i64 %752, %751, !dbg !19
  %dma.buffer.offset477 = trunc i64 %753 to i32, !dbg !19
  %754 = trunc i64 %751 to i32, !dbg !19
  %755 = insertelement <4 x i32> poison, i32 %754, i64 0, !dbg !19
  %756 = lshr i64 %751, 32, !dbg !19
  %757 = trunc i64 %756 to i32, !dbg !19
  %758 = insertelement <4 x i32> %755, i32 %757, i64 1, !dbg !19
  %759 = insertelement <4 x i32> %758, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc478 = insertelement <4 x i32> %759, i32 131072, i64 3, !dbg !19
  %760 = ptrtoint ptr addrspace(3) %750 to i32, !dbg !19
  %dma.buffer.lds479 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %760), !dbg !19
  %761 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds479, i32 %dma.buffer.offset477, <4 x i32> %dma.buffer.rsrc478) #10, !dbg !19
  %762 = or disjoint i64 %32, 256, !dbg !19
  br label %733, !dbg !19

763:                                              ; preds = %733
  %764 = shufflevector <8 x half> %728, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %765 = fmul <4 x float> %139, %670, !dbg !36
  %766 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %764, <4 x float> %765) #9, !dbg !16
  %767 = shufflevector <8 x half> %730, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %768 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %767, <4 x float> %766) #9, !dbg !16
  %769 = shufflevector <8 x half> %728, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %770 = fmul <4 x float> %140, %670, !dbg !36
  %771 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %769, <4 x float> %770) #9, !dbg !16
  %772 = shufflevector <8 x half> %730, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %773 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %772, <4 x float> %771) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread258 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local259 = shl nuw nsw i32 %address.thread258, 3, !dbg !16
  %address.local260 = and i32 %address.local259, 504, !dbg !16
  %address.local261 = zext nneg i32 %address.local260 to i64, !dbg !16
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local261, !dbg !16
  %774 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local262, i32 0) #0, !dbg !16
  %775 = bitcast <4 x i32> %774 to <8 x half>, !dbg !16
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local261, !dbg !16
  %address.local264 = getelementptr i8, ptr addrspace(3) %address.local263, i64 1024, !dbg !16
  %776 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local264, i32 0) #0, !dbg !16
  %777 = bitcast <4 x i32> %776 to <8 x half>, !dbg !16
  %778 = shufflevector <8 x half> %775, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %779 = fmul <4 x float> %142, %670, !dbg !36
  %780 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %778, <4 x float> %779) #9, !dbg !16
  %781 = shufflevector <8 x half> %777, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %782 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %781, <4 x float> %780) #9, !dbg !16
  %783 = shufflevector <8 x half> %775, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %784 = fmul <4 x float> %143, %670, !dbg !36
  %785 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %783, <4 x float> %784) #9, !dbg !16
  %786 = shufflevector <8 x half> %777, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %787 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed38, <4 x half> %786, <4 x float> %785) #9, !dbg !16
  %788 = shufflevector <4 x float> %674, <4 x float> %679, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %789 = insertelement <1 x float> poison, float %626, i64 0, !dbg !7
  %790 = shufflevector <4 x float> %721, <4 x float> %726, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %791 = shufflevector <4 x float> %768, <4 x float> %773, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %792 = shufflevector <4 x float> %782, <4 x float> %787, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %793 = add nuw nsw i64 %130, 1, !dbg !7
  br label %129, !dbg !7

794:                                              ; preds = %129
  %795 = extractelement <8 x float> %144, i64 0, !dbg !7
  %796 = extractelement <8 x float> %144, i64 1, !dbg !7
  %797 = extractelement <8 x float> %144, i64 2, !dbg !7
  %798 = extractelement <8 x float> %144, i64 3, !dbg !7
  %799 = select i1 %.not, float %796, float %795, !dbg !7
  %800 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %801 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %800), !dbg !7
  %802 = add nuw nsw i32 %801, 64, !dbg !7
  %803 = xor i32 %801, 16, !dbg !7
  %804 = and i32 %802, 192, !dbg !7
  %805 = icmp samesign ult i32 %803, %804, !dbg !7
  %806 = select i1 %805, i32 %803, i32 %801, !dbg !7
  %807 = shl nuw nsw i32 %806, 2, !dbg !7
  %808 = bitcast float %799 to i32, !dbg !7
  %809 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %808), !dbg !7
  %810 = bitcast i32 %809 to float, !dbg !7
  %811 = select i1 %.not, float %795, float %810, !dbg !7
  %812 = select i1 %.not, float %810, float %796, !dbg !7
  %813 = select i1 %.not, float %798, float %797, !dbg !7
  %814 = bitcast float %813 to i32, !dbg !7
  %815 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %814), !dbg !7
  %816 = bitcast i32 %815 to float, !dbg !7
  %817 = select i1 %.not, float %797, float %816, !dbg !7
  %818 = select i1 %.not, float %816, float %798, !dbg !7
  %819 = select i1 %.not1, float %817, float %811, !dbg !7
  %820 = xor i32 %801, 32, !dbg !7
  %821 = icmp samesign ult i32 %820, %804, !dbg !7
  %822 = select i1 %821, i32 %820, i32 %801, !dbg !7
  %823 = shl nuw nsw i32 %822, 2, !dbg !7
  %824 = bitcast float %819 to i32, !dbg !7
  %825 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %824), !dbg !7
  %826 = bitcast i32 %825 to float, !dbg !7
  %827 = select i1 %.not1, float %811, float %826, !dbg !7
  %828 = select i1 %.not1, float %826, float %817, !dbg !7
  %829 = select i1 %.not1, float %818, float %812, !dbg !7
  %830 = bitcast float %829 to i32, !dbg !7
  %831 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %830), !dbg !7
  %832 = bitcast i32 %831 to float, !dbg !7
  %833 = select i1 %.not1, float %812, float %832, !dbg !7
  %834 = select i1 %.not1, float %832, float %818, !dbg !7
  %835 = extractelement <8 x float> %141, i64 0, !dbg !7
  %836 = extractelement <8 x float> %141, i64 1, !dbg !7
  %837 = extractelement <8 x float> %141, i64 2, !dbg !7
  %838 = extractelement <8 x float> %141, i64 3, !dbg !7
  %839 = select i1 %.not, float %836, float %835, !dbg !7
  %840 = bitcast float %839 to i32, !dbg !7
  %841 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %840), !dbg !7
  %842 = bitcast i32 %841 to float, !dbg !7
  %843 = select i1 %.not, float %835, float %842, !dbg !7
  %844 = select i1 %.not, float %842, float %836, !dbg !7
  %845 = select i1 %.not, float %838, float %837, !dbg !7
  %846 = bitcast float %845 to i32, !dbg !7
  %847 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %846), !dbg !7
  %848 = bitcast i32 %847 to float, !dbg !7
  %849 = select i1 %.not, float %837, float %848, !dbg !7
  %850 = select i1 %.not, float %848, float %838, !dbg !7
  %851 = select i1 %.not1, float %849, float %843, !dbg !7
  %852 = bitcast float %851 to i32, !dbg !7
  %853 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %852), !dbg !7
  %854 = bitcast i32 %853 to float, !dbg !7
  %855 = select i1 %.not1, float %843, float %854, !dbg !7
  %856 = select i1 %.not1, float %854, float %849, !dbg !7
  %857 = select i1 %.not1, float %850, float %844, !dbg !7
  %858 = bitcast float %857 to i32, !dbg !7
  %859 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %858), !dbg !7
  %860 = bitcast i32 %859 to float, !dbg !7
  %861 = select i1 %.not1, float %844, float %860, !dbg !7
  %862 = select i1 %.not1, float %860, float %850, !dbg !7
  %863 = extractelement <8 x float> %138, i64 0, !dbg !7
  %864 = extractelement <8 x float> %138, i64 1, !dbg !7
  %865 = extractelement <8 x float> %138, i64 2, !dbg !7
  %866 = extractelement <8 x float> %138, i64 3, !dbg !7
  %867 = select i1 %.not, float %864, float %863, !dbg !7
  %868 = bitcast float %867 to i32, !dbg !7
  %869 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %868), !dbg !7
  %870 = bitcast i32 %869 to float, !dbg !7
  %871 = select i1 %.not, float %863, float %870, !dbg !7
  %872 = select i1 %.not, float %870, float %864, !dbg !7
  %873 = select i1 %.not, float %866, float %865, !dbg !7
  %874 = bitcast float %873 to i32, !dbg !7
  %875 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %874), !dbg !7
  %876 = bitcast i32 %875 to float, !dbg !7
  %877 = select i1 %.not, float %865, float %876, !dbg !7
  %878 = select i1 %.not, float %876, float %866, !dbg !7
  %879 = select i1 %.not1, float %877, float %871, !dbg !7
  %880 = bitcast float %879 to i32, !dbg !7
  %881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %880), !dbg !7
  %882 = bitcast i32 %881 to float, !dbg !7
  %883 = select i1 %.not1, float %871, float %882, !dbg !7
  %884 = select i1 %.not1, float %882, float %877, !dbg !7
  %885 = select i1 %.not1, float %878, float %872, !dbg !7
  %886 = bitcast float %885 to i32, !dbg !7
  %887 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %886), !dbg !7
  %888 = bitcast i32 %887 to float, !dbg !7
  %889 = select i1 %.not1, float %872, float %888, !dbg !7
  %890 = select i1 %.not1, float %888, float %878, !dbg !7
  %891 = extractelement <8 x float> %133, i64 0, !dbg !7
  %892 = extractelement <8 x float> %133, i64 1, !dbg !7
  %893 = extractelement <8 x float> %133, i64 2, !dbg !7
  %894 = extractelement <8 x float> %133, i64 3, !dbg !7
  %895 = select i1 %.not, float %892, float %891, !dbg !7
  %896 = bitcast float %895 to i32, !dbg !7
  %897 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %896), !dbg !7
  %898 = bitcast i32 %897 to float, !dbg !7
  %899 = select i1 %.not, float %891, float %898, !dbg !7
  %900 = select i1 %.not, float %898, float %892, !dbg !7
  %901 = select i1 %.not, float %894, float %893, !dbg !7
  %902 = bitcast float %901 to i32, !dbg !7
  %903 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %902), !dbg !7
  %904 = bitcast i32 %903 to float, !dbg !7
  %905 = select i1 %.not, float %893, float %904, !dbg !7
  %906 = select i1 %.not, float %904, float %894, !dbg !7
  %907 = select i1 %.not1, float %905, float %899, !dbg !7
  %908 = bitcast float %907 to i32, !dbg !7
  %909 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %908), !dbg !7
  %910 = select i1 %.not1, float %906, float %900, !dbg !7
  %911 = bitcast float %910 to i32, !dbg !7
  %912 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %911), !dbg !7
  %913 = shufflevector <1 x float> %134, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %914 = extractelement <8 x float> %144, i64 4, !dbg !7
  %915 = extractelement <8 x float> %144, i64 5, !dbg !7
  %916 = extractelement <8 x float> %144, i64 6, !dbg !7
  %917 = extractelement <8 x float> %144, i64 7, !dbg !7
  %918 = select i1 %.not, float %915, float %914, !dbg !7
  %919 = bitcast float %918 to i32, !dbg !7
  %920 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %919), !dbg !7
  %921 = bitcast i32 %920 to float, !dbg !7
  %922 = select i1 %.not, float %914, float %921, !dbg !7
  %923 = select i1 %.not, float %921, float %915, !dbg !7
  %924 = select i1 %.not, float %917, float %916, !dbg !7
  %925 = bitcast float %924 to i32, !dbg !7
  %926 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %925), !dbg !7
  %927 = bitcast i32 %926 to float, !dbg !7
  %928 = select i1 %.not, float %916, float %927, !dbg !7
  %929 = select i1 %.not, float %927, float %917, !dbg !7
  %930 = select i1 %.not1, float %928, float %922, !dbg !7
  %931 = bitcast float %930 to i32, !dbg !7
  %932 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %931), !dbg !7
  %933 = bitcast i32 %932 to float, !dbg !7
  %934 = select i1 %.not1, float %922, float %933, !dbg !7
  %935 = select i1 %.not1, float %933, float %928, !dbg !7
  %936 = select i1 %.not1, float %929, float %923, !dbg !7
  %937 = bitcast float %936 to i32, !dbg !7
  %938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %937), !dbg !7
  %939 = bitcast i32 %938 to float, !dbg !7
  %940 = select i1 %.not1, float %923, float %939, !dbg !7
  %941 = select i1 %.not1, float %939, float %929, !dbg !7
  %942 = extractelement <8 x float> %141, i64 4, !dbg !7
  %943 = extractelement <8 x float> %141, i64 5, !dbg !7
  %944 = extractelement <8 x float> %141, i64 6, !dbg !7
  %945 = extractelement <8 x float> %141, i64 7, !dbg !7
  %946 = select i1 %.not, float %943, float %942, !dbg !7
  %947 = bitcast float %946 to i32, !dbg !7
  %948 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %947), !dbg !7
  %949 = bitcast i32 %948 to float, !dbg !7
  %950 = select i1 %.not, float %942, float %949, !dbg !7
  %951 = select i1 %.not, float %949, float %943, !dbg !7
  %952 = select i1 %.not, float %945, float %944, !dbg !7
  %953 = bitcast float %952 to i32, !dbg !7
  %954 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %953), !dbg !7
  %955 = bitcast i32 %954 to float, !dbg !7
  %956 = select i1 %.not, float %944, float %955, !dbg !7
  %957 = select i1 %.not, float %955, float %945, !dbg !7
  %958 = select i1 %.not1, float %956, float %950, !dbg !7
  %959 = bitcast float %958 to i32, !dbg !7
  %960 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %959), !dbg !7
  %961 = bitcast i32 %960 to float, !dbg !7
  %962 = select i1 %.not1, float %950, float %961, !dbg !7
  %963 = select i1 %.not1, float %961, float %956, !dbg !7
  %964 = select i1 %.not1, float %957, float %951, !dbg !7
  %965 = bitcast float %964 to i32, !dbg !7
  %966 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %965), !dbg !7
  %967 = bitcast i32 %966 to float, !dbg !7
  %968 = select i1 %.not1, float %951, float %967, !dbg !7
  %969 = select i1 %.not1, float %967, float %957, !dbg !7
  %970 = extractelement <8 x float> %138, i64 4, !dbg !7
  %971 = extractelement <8 x float> %138, i64 5, !dbg !7
  %972 = extractelement <8 x float> %138, i64 6, !dbg !7
  %973 = extractelement <8 x float> %138, i64 7, !dbg !7
  %974 = select i1 %.not, float %971, float %970, !dbg !7
  %975 = bitcast float %974 to i32, !dbg !7
  %976 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %975), !dbg !7
  %977 = bitcast i32 %976 to float, !dbg !7
  %978 = select i1 %.not, float %970, float %977, !dbg !7
  %979 = select i1 %.not, float %977, float %971, !dbg !7
  %980 = select i1 %.not, float %973, float %972, !dbg !7
  %981 = bitcast float %980 to i32, !dbg !7
  %982 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %981), !dbg !7
  %983 = bitcast i32 %982 to float, !dbg !7
  %984 = select i1 %.not, float %972, float %983, !dbg !7
  %985 = select i1 %.not, float %983, float %973, !dbg !7
  %986 = select i1 %.not1, float %984, float %978, !dbg !7
  %987 = bitcast float %986 to i32, !dbg !7
  %988 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %987), !dbg !7
  %989 = bitcast i32 %988 to float, !dbg !7
  %990 = select i1 %.not1, float %978, float %989, !dbg !7
  %991 = select i1 %.not1, float %989, float %984, !dbg !7
  %992 = select i1 %.not1, float %985, float %979, !dbg !7
  %993 = bitcast float %992 to i32, !dbg !7
  %994 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %993), !dbg !7
  %995 = bitcast i32 %994 to float, !dbg !7
  %996 = select i1 %.not1, float %979, float %995, !dbg !7
  %997 = select i1 %.not1, float %995, float %985, !dbg !7
  %998 = extractelement <8 x float> %133, i64 4, !dbg !7
  %999 = extractelement <8 x float> %133, i64 5, !dbg !7
  %1000 = extractelement <8 x float> %133, i64 6, !dbg !7
  %1001 = extractelement <8 x float> %133, i64 7, !dbg !7
  %1002 = select i1 %.not, float %999, float %998, !dbg !7
  %1003 = bitcast float %1002 to i32, !dbg !7
  %1004 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %1003), !dbg !7
  %1005 = bitcast i32 %1004 to float, !dbg !7
  %1006 = select i1 %.not, float %998, float %1005, !dbg !7
  %1007 = select i1 %.not, float %1005, float %999, !dbg !7
  %1008 = select i1 %.not, float %1001, float %1000, !dbg !7
  %1009 = bitcast float %1008 to i32, !dbg !7
  %1010 = call i32 @llvm.amdgcn.ds.bpermute(i32 %807, i32 %1009), !dbg !7
  %1011 = bitcast i32 %1010 to float, !dbg !7
  %1012 = select i1 %.not, float %1000, float %1011, !dbg !7
  %1013 = select i1 %.not, float %1011, float %1001, !dbg !7
  %1014 = select i1 %.not1, float %1012, float %1006, !dbg !7
  %1015 = bitcast float %1014 to i32, !dbg !7
  %1016 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %1015), !dbg !7
  %1017 = select i1 %.not1, float %1013, float %1007, !dbg !7
  %1018 = bitcast float %1017 to i32, !dbg !7
  %1019 = call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %1018), !dbg !7
  br i1 true, label %1020, label %1054, !dbg !22

1020:                                             ; preds = %794
  %1021 = bitcast i32 %1016 to float, !dbg !7
  %1022 = select i1 %.not1, float %1006, float %1021, !dbg !7
  %1023 = insertelement <4 x float> poison, float %1022, i64 0, !dbg !20
  %1024 = bitcast i32 %1019 to float, !dbg !7
  %1025 = select i1 %.not1, float %1007, float %1024, !dbg !7
  %1026 = insertelement <4 x float> %1023, float %1025, i64 1, !dbg !20
  %1027 = select i1 %.not1, float %1021, float %1012, !dbg !7
  %1028 = insertelement <4 x float> %1026, float %1027, i64 2, !dbg !20
  %1029 = select i1 %.not1, float %1024, float %1013, !dbg !7
  %1030 = insertelement <4 x float> %1028, float %1029, i64 3, !dbg !20
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %913, !dbg !20
  %normalized = fmul <4 x float> %1030, %row.reciprocal, !dbg !20
  %1031 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !21
  %1032 = extractelement <4 x half> %1031, i64 3, !dbg !21
  %1033 = extractelement <4 x half> %1031, i64 2, !dbg !21
  %1034 = extractelement <4 x half> %1031, i64 1, !dbg !21
  %1035 = extractelement <4 x half> %1031, i64 0, !dbg !21
  %1036 = bitcast i32 %909 to float, !dbg !7
  %1037 = select i1 %.not1, float %899, float %1036, !dbg !7
  %1038 = insertelement <4 x float> poison, float %1037, i64 0, !dbg !20
  %1039 = bitcast i32 %912 to float, !dbg !7
  %1040 = select i1 %.not1, float %900, float %1039, !dbg !7
  %1041 = insertelement <4 x float> %1038, float %1040, i64 1, !dbg !20
  %1042 = select i1 %.not1, float %1036, float %905, !dbg !7
  %1043 = insertelement <4 x float> %1041, float %1042, i64 2, !dbg !20
  %1044 = select i1 %.not1, float %1039, float %906, !dbg !7
  %1045 = insertelement <4 x float> %1043, float %1044, i64 3, !dbg !20
  %normalized15 = fmul <4 x float> %1045, %row.reciprocal, !dbg !20
  %1046 = fptrunc <4 x float> %normalized15 to <4 x half>, !dbg !21
  %1047 = extractelement <4 x half> %1046, i64 3, !dbg !21
  %1048 = extractelement <4 x half> %1046, i64 2, !dbg !21
  %1049 = extractelement <4 x half> %1046, i64 1, !dbg !21
  %1050 = extractelement <4 x half> %1046, i64 0, !dbg !21
  %1051 = add nsw i64 %16, %22, !dbg !22
  %1052 = shl i64 %1051, 19, !dbg !22
  %1053 = and i64 %1052, 16252928, !dbg !22
  %address.thread265 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !22, !range !40, !frisk.register_lifetimes !1
  %address.local266 = zext nneg i32 %address.thread265 to i64, !dbg !8
  %address.local267 = add i64 %25, %address.local266, !dbg !22
  %address.local268 = lshr i64 %address.local266, 6, !dbg !11
  %address.local269 = shl nuw nsw i64 %address.local268, 4, !dbg !22
  %address.local270 = add i64 %address.local267, %address.local269, !dbg !22
  %address.local271 = and i64 %address.local266, 240, !dbg !22
  %address.local272 = sub i64 %address.local270, %address.local271, !dbg !22
  %address.local273 = add i64 %address.local272, %26, !dbg !22
  %address.local274 = shl nuw nsw i64 %address.local273, 7, !dbg !22
  %address.local275 = add nuw nsw i64 %1053, %address.local274, !dbg !22
  %address.local276 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !22
  %address.local277 = lshr i64 %address.local266, 2, !dbg !11
  %address.local278 = and i64 %address.local277, 12, !dbg !11
  %address.local279 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local276, i64 %address.local278, !dbg !22
  store half %1050, ptr addrspace(1) %address.local279, align 2, !dbg !22
  %address.local280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !22
  %address.local281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local280, i64 %address.local278, !dbg !22
  %address.local282 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local281, i64 2, !dbg !22
  store half %1049, ptr addrspace(1) %address.local282, align 2, !dbg !22
  %address.local283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !22
  %address.local284 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local283, i64 %address.local278, !dbg !22
  %address.local285 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local284, i64 4, !dbg !22
  store half %1048, ptr addrspace(1) %address.local285, align 2, !dbg !22
  %address.local286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !22
  %address.local287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local286, i64 %address.local278, !dbg !22
  %address.local288 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local287, i64 6, !dbg !22
  store half %1047, ptr addrspace(1) %address.local288, align 2, !dbg !22
  %address.local289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !22
  %address.local290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local289, i64 %address.local278, !dbg !22
  %address.local291 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local290, i64 32, !dbg !22
  store half %1035, ptr addrspace(1) %address.local291, align 2, !dbg !22
  %address.local292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !22
  %address.local293 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local292, i64 %address.local278, !dbg !22
  %address.local294 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local293, i64 34, !dbg !22
  store half %1034, ptr addrspace(1) %address.local294, align 2, !dbg !22
  %address.local295 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !22
  %address.local296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local295, i64 %address.local278, !dbg !22
  %address.local297 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local296, i64 36, !dbg !22
  store half %1033, ptr addrspace(1) %address.local297, align 2, !dbg !22
  %address.local298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !22
  %address.local299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local298, i64 %address.local278, !dbg !22
  %address.local300 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local299, i64 38, !dbg !22
  store half %1032, ptr addrspace(1) %address.local300, align 2, !dbg !22
  br label %1054, !dbg !22

1054:                                             ; preds = %1020, %794
  br i1 true, label %1055, label %1075, !dbg !22

1055:                                             ; preds = %1054
  %1056 = insertelement <4 x float> poison, float %990, i64 0, !dbg !20
  %1057 = insertelement <4 x float> %1056, float %996, i64 1, !dbg !20
  %1058 = insertelement <4 x float> %1057, float %991, i64 2, !dbg !20
  %1059 = insertelement <4 x float> %1058, float %997, i64 3, !dbg !20
  %row.reciprocal16 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %913, !dbg !20
  %normalized17 = fmul <4 x float> %1059, %row.reciprocal16, !dbg !20
  %1060 = fptrunc <4 x float> %normalized17 to <4 x half>, !dbg !21
  %1061 = extractelement <4 x half> %1060, i64 3, !dbg !21
  %1062 = extractelement <4 x half> %1060, i64 2, !dbg !21
  %1063 = extractelement <4 x half> %1060, i64 1, !dbg !21
  %1064 = extractelement <4 x half> %1060, i64 0, !dbg !21
  %1065 = insertelement <4 x float> poison, float %883, i64 0, !dbg !20
  %1066 = insertelement <4 x float> %1065, float %889, i64 1, !dbg !20
  %1067 = insertelement <4 x float> %1066, float %884, i64 2, !dbg !20
  %1068 = insertelement <4 x float> %1067, float %890, i64 3, !dbg !20
  %normalized18 = fmul <4 x float> %1068, %row.reciprocal16, !dbg !20
  %1069 = fptrunc <4 x float> %normalized18 to <4 x half>, !dbg !21
  %1070 = extractelement <4 x half> %1069, i64 3, !dbg !21
  %1071 = extractelement <4 x half> %1069, i64 2, !dbg !21
  %1072 = extractelement <4 x half> %1069, i64 1, !dbg !21
  %1073 = extractelement <4 x half> %1069, i64 0, !dbg !21
  %1074 = shl nuw nsw i64 %24, 19, !dbg !22
  %address.thread301 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !22, !range !40, !frisk.register_lifetimes !1
  %address.local302 = zext nneg i32 %address.thread301 to i64, !dbg !8
  %address.local303 = add i64 %25, %address.local302, !dbg !22
  %address.local304 = lshr i64 %address.local302, 6, !dbg !11
  %address.local305 = shl nuw nsw i64 %address.local304, 4, !dbg !22
  %address.local306 = add i64 %address.local303, %address.local305, !dbg !22
  %address.local307 = and i64 %address.local302, 240, !dbg !22
  %address.local308 = sub i64 %address.local306, %address.local307, !dbg !22
  %address.local309 = add i64 %address.local308, %26, !dbg !22
  %address.local310 = shl nuw nsw i64 %address.local309, 7, !dbg !22
  %address.local311 = add nuw nsw i64 %1074, %address.local310, !dbg !22
  %address.local312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local311, !dbg !22
  %address.local313 = lshr i64 %address.local302, 2, !dbg !11
  %address.local314 = and i64 %address.local313, 12, !dbg !11
  %address.local315 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local312, i64 %address.local314, !dbg !22
  %address.local316 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local315, i64 64, !dbg !22
  store half %1073, ptr addrspace(1) %address.local316, align 2, !dbg !22
  %address.local317 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local311, !dbg !22
  %address.local318 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local317, i64 %address.local314, !dbg !22
  %address.local319 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local318, i64 66, !dbg !22
  store half %1072, ptr addrspace(1) %address.local319, align 2, !dbg !22
  %address.local320 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local311, !dbg !22
  %address.local321 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local320, i64 %address.local314, !dbg !22
  %address.local322 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local321, i64 68, !dbg !22
  store half %1071, ptr addrspace(1) %address.local322, align 2, !dbg !22
  %address.local323 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local311, !dbg !22
  %address.local324 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local323, i64 %address.local314, !dbg !22
  %address.local325 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local324, i64 70, !dbg !22
  store half %1070, ptr addrspace(1) %address.local325, align 2, !dbg !22
  %address.local326 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local311, !dbg !22
  %address.local327 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local326, i64 %address.local314, !dbg !22
  %address.local328 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local327, i64 96, !dbg !22
  store half %1064, ptr addrspace(1) %address.local328, align 2, !dbg !22
  %address.local329 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local311, !dbg !22
  %address.local330 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local329, i64 %address.local314, !dbg !22
  %address.local331 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local330, i64 98, !dbg !22
  store half %1063, ptr addrspace(1) %address.local331, align 2, !dbg !22
  %address.local332 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local311, !dbg !22
  %address.local333 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local332, i64 %address.local314, !dbg !22
  %address.local334 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local333, i64 100, !dbg !22
  store half %1062, ptr addrspace(1) %address.local334, align 2, !dbg !22
  %address.local335 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local311, !dbg !22
  %address.local336 = or i64 %address.local313, 51, !dbg !22
  %address.local337 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local335, i64 %address.local336, !dbg !22
  store half %1061, ptr addrspace(1) %address.local337, align 2, !dbg !22
  br label %1075, !dbg !22

1075:                                             ; preds = %1055, %1054
  br i1 true, label %1076, label %1096, !dbg !22

1076:                                             ; preds = %1075
  %1077 = insertelement <4 x float> poison, float %962, i64 0, !dbg !20
  %1078 = insertelement <4 x float> %1077, float %968, i64 1, !dbg !20
  %1079 = insertelement <4 x float> %1078, float %963, i64 2, !dbg !20
  %1080 = insertelement <4 x float> %1079, float %969, i64 3, !dbg !20
  %row.reciprocal19 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %913, !dbg !20
  %normalized20 = fmul <4 x float> %1080, %row.reciprocal19, !dbg !20
  %1081 = fptrunc <4 x float> %normalized20 to <4 x half>, !dbg !21
  %1082 = extractelement <4 x half> %1081, i64 3, !dbg !21
  %1083 = extractelement <4 x half> %1081, i64 2, !dbg !21
  %1084 = extractelement <4 x half> %1081, i64 1, !dbg !21
  %1085 = extractelement <4 x half> %1081, i64 0, !dbg !21
  %1086 = insertelement <4 x float> poison, float %855, i64 0, !dbg !20
  %1087 = insertelement <4 x float> %1086, float %861, i64 1, !dbg !20
  %1088 = insertelement <4 x float> %1087, float %856, i64 2, !dbg !20
  %1089 = insertelement <4 x float> %1088, float %862, i64 3, !dbg !20
  %normalized21 = fmul <4 x float> %1089, %row.reciprocal19, !dbg !20
  %1090 = fptrunc <4 x float> %normalized21 to <4 x half>, !dbg !21
  %1091 = extractelement <4 x half> %1090, i64 3, !dbg !21
  %1092 = extractelement <4 x half> %1090, i64 2, !dbg !21
  %1093 = extractelement <4 x half> %1090, i64 1, !dbg !21
  %1094 = extractelement <4 x half> %1090, i64 0, !dbg !21
  %1095 = shl nuw nsw i64 %24, 19, !dbg !22
  %address.thread338 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !22, !range !40, !frisk.register_lifetimes !1
  %address.local339 = zext nneg i32 %address.thread338 to i64, !dbg !8
  %address.local340 = add i64 %25, %address.local339, !dbg !22
  %address.local341 = lshr i64 %address.local339, 6, !dbg !11
  %address.local342 = shl nuw nsw i64 %address.local341, 4, !dbg !22
  %address.local343 = add i64 %address.local340, %address.local342, !dbg !22
  %address.local344 = and i64 %address.local339, 240, !dbg !22
  %address.local345 = sub i64 %address.local343, %address.local344, !dbg !22
  %address.local346 = add i64 %address.local345, %26, !dbg !22
  %address.local347 = shl nuw nsw i64 %address.local346, 7, !dbg !22
  %address.local348 = add nuw nsw i64 %1095, %address.local347, !dbg !22
  %address.local349 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local348, !dbg !22
  %address.local350 = lshr i64 %address.local339, 2, !dbg !11
  %address.local351 = and i64 %address.local350, 12, !dbg !11
  %address.local352 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local349, i64 %address.local351, !dbg !22
  %address.local353 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local352, i64 128, !dbg !22
  store half %1094, ptr addrspace(1) %address.local353, align 2, !dbg !22
  %address.local354 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local348, !dbg !22
  %address.local355 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local354, i64 %address.local351, !dbg !22
  %address.local356 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local355, i64 130, !dbg !22
  store half %1093, ptr addrspace(1) %address.local356, align 2, !dbg !22
  %address.local357 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local348, !dbg !22
  %address.local358 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local357, i64 %address.local351, !dbg !22
  %address.local359 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local358, i64 132, !dbg !22
  store half %1092, ptr addrspace(1) %address.local359, align 2, !dbg !22
  %address.local360 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local348, !dbg !22
  %address.local361 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local360, i64 %address.local351, !dbg !22
  %address.local362 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local361, i64 134, !dbg !22
  store half %1091, ptr addrspace(1) %address.local362, align 2, !dbg !22
  %address.local363 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local348, !dbg !22
  %address.local364 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local363, i64 %address.local351, !dbg !22
  %address.local365 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local364, i64 160, !dbg !22
  store half %1085, ptr addrspace(1) %address.local365, align 2, !dbg !22
  %address.local366 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local348, !dbg !22
  %address.local367 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local366, i64 %address.local351, !dbg !22
  %address.local368 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local367, i64 162, !dbg !22
  store half %1084, ptr addrspace(1) %address.local368, align 2, !dbg !22
  %address.local369 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local348, !dbg !22
  %address.local370 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local369, i64 %address.local351, !dbg !22
  %address.local371 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local370, i64 164, !dbg !22
  store half %1083, ptr addrspace(1) %address.local371, align 2, !dbg !22
  %address.local372 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local348, !dbg !22
  %address.local373 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local372, i64 %address.local351, !dbg !22
  %address.local374 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local373, i64 166, !dbg !22
  store half %1082, ptr addrspace(1) %address.local374, align 2, !dbg !22
  br label %1096, !dbg !22

1096:                                             ; preds = %1076, %1075
  br i1 true, label %1097, label %1117, !dbg !22

1097:                                             ; preds = %1096
  %1098 = insertelement <4 x float> poison, float %934, i64 0, !dbg !20
  %1099 = insertelement <4 x float> %1098, float %940, i64 1, !dbg !20
  %1100 = insertelement <4 x float> %1099, float %935, i64 2, !dbg !20
  %1101 = insertelement <4 x float> %1100, float %941, i64 3, !dbg !20
  %row.reciprocal22 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %913, !dbg !20
  %normalized23 = fmul <4 x float> %1101, %row.reciprocal22, !dbg !20
  %1102 = fptrunc <4 x float> %normalized23 to <4 x half>, !dbg !21
  %1103 = extractelement <4 x half> %1102, i64 3, !dbg !21
  %1104 = extractelement <4 x half> %1102, i64 2, !dbg !21
  %1105 = extractelement <4 x half> %1102, i64 1, !dbg !21
  %1106 = extractelement <4 x half> %1102, i64 0, !dbg !21
  %1107 = insertelement <4 x float> poison, float %827, i64 0, !dbg !20
  %1108 = insertelement <4 x float> %1107, float %833, i64 1, !dbg !20
  %1109 = insertelement <4 x float> %1108, float %828, i64 2, !dbg !20
  %1110 = insertelement <4 x float> %1109, float %834, i64 3, !dbg !20
  %normalized24 = fmul <4 x float> %1110, %row.reciprocal22, !dbg !20
  %1111 = fptrunc <4 x float> %normalized24 to <4 x half>, !dbg !21
  %1112 = extractelement <4 x half> %1111, i64 3, !dbg !21
  %1113 = extractelement <4 x half> %1111, i64 2, !dbg !21
  %1114 = extractelement <4 x half> %1111, i64 1, !dbg !21
  %1115 = extractelement <4 x half> %1111, i64 0, !dbg !21
  %1116 = shl nuw nsw i64 %24, 19, !dbg !22
  %address.thread375 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !22, !range !40, !frisk.register_lifetimes !1
  %address.local376 = zext nneg i32 %address.thread375 to i64, !dbg !8
  %address.local377 = add i64 %25, %address.local376, !dbg !22
  %address.local378 = lshr i64 %address.local376, 6, !dbg !11
  %address.local379 = shl nuw nsw i64 %address.local378, 4, !dbg !22
  %address.local380 = add i64 %address.local377, %address.local379, !dbg !22
  %address.local381 = and i64 %address.local376, 240, !dbg !22
  %address.local382 = sub i64 %address.local380, %address.local381, !dbg !22
  %address.local383 = add i64 %address.local382, %26, !dbg !22
  %address.local384 = shl nuw nsw i64 %address.local383, 7, !dbg !22
  %address.local385 = add nuw nsw i64 %1116, %address.local384, !dbg !22
  %address.local386 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local385, !dbg !22
  %address.local387 = lshr i64 %address.local376, 2, !dbg !11
  %address.local388 = and i64 %address.local387, 12, !dbg !11
  %address.local389 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local386, i64 %address.local388, !dbg !22
  %address.local390 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local389, i64 192, !dbg !22
  store half %1115, ptr addrspace(1) %address.local390, align 2, !dbg !22
  %address.local391 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local385, !dbg !22
  %address.local392 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local391, i64 %address.local388, !dbg !22
  %address.local393 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local392, i64 194, !dbg !22
  store half %1114, ptr addrspace(1) %address.local393, align 2, !dbg !22
  %address.local394 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local385, !dbg !22
  %address.local395 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local394, i64 %address.local388, !dbg !22
  %address.local396 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local395, i64 196, !dbg !22
  store half %1113, ptr addrspace(1) %address.local396, align 2, !dbg !22
  %address.local397 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local385, !dbg !22
  %address.local398 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local397, i64 %address.local388, !dbg !22
  %address.local399 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local398, i64 198, !dbg !22
  store half %1112, ptr addrspace(1) %address.local399, align 2, !dbg !22
  %address.local400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local385, !dbg !22
  %address.local401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local388, !dbg !22
  %address.local402 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local401, i64 224, !dbg !22
  store half %1106, ptr addrspace(1) %address.local402, align 2, !dbg !22
  %address.local403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local385, !dbg !22
  %address.local404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local403, i64 %address.local388, !dbg !22
  %address.local405 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local404, i64 226, !dbg !22
  store half %1105, ptr addrspace(1) %address.local405, align 2, !dbg !22
  %address.local406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local385, !dbg !22
  %address.local407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local406, i64 %address.local388, !dbg !22
  %address.local408 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local407, i64 228, !dbg !22
  store half %1104, ptr addrspace(1) %address.local408, align 2, !dbg !22
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local385, !dbg !22
  %address.local410 = or i64 %address.local387, 115, !dbg !22
  %address.local411 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local409, i64 %address.local410, !dbg !22
  store half %1103, ptr addrspace(1) %address.local411, align 2, !dbg !22
  br label %1117, !dbg !22

1117:                                             ; preds = %1097, %1096
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
!llvm.module.flags = !{!39}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/online-baseline")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 70, column: 22, scope: !4)
!10 = !DILocation(line: 48, column: 17, scope: !4)
!11 = !DILocation(line: 72, column: 24, scope: !4)
!14 = !DILocation(line: 63, column: 17, scope: !4)
!15 = !DILocation(line: 77, column: 19, scope: !4)
!16 = !DILocation(line: 102, column: 23, scope: !4)
!17 = !DILocation(line: 73, column: 26, scope: !4)
!18 = !DILocation(line: 71, column: 22, scope: !4)
!19 = !DILocation(line: 101, column: 20, scope: !4)
!20 = !DILocation(line: 107, column: 23, scope: !4)
!21 = !DILocation(line: 108, column: 19, scope: !4)
!22 = !DILocation(line: 109, column: 9, scope: !4)
!23 = !DILocation(line: 74, column: 23, scope: !4)
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
