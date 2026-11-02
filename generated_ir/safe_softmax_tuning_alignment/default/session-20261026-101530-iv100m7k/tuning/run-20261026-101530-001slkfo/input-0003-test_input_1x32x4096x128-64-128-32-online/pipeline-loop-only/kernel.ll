; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/pipeline-loop-only/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/pipeline-loop-only/input.mlir"

@shm_0 = addrspace(3) global [24576 x i8] undef, align 16

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
  %11 = shl nsw i64 %10, 6, !dbg !9
  %12 = icmp slt i32 %9, 0, !dbg !12
  %13 = xor i64 %10, -1, !dbg !12
  %14 = select i1 %12, i64 %13, i64 %10, !dbg !12
  %15 = sdiv i64 %14, 64, !dbg !12
  %.lobit = ashr i32 %9, 31, !dbg !12
  %16 = sext i32 %.lobit to i64, !dbg !12
  %17 = xor i64 %15, %16, !dbg !12
  %18 = add nsw i64 %17, %8, !dbg !12
  %19 = and i64 %18, 31, !dbg !12
  %20 = shl nsw i64 %10, 6, !dbg !12
  %21 = mul nsw i64 %17, -4096, !dbg !12
  %22 = add nsw i64 %20, %21, !dbg !12
  %23 = shl nsw i64 %22, 7, !dbg !12
  %24 = add nsw i64 %17, %8, !dbg !12
  %25 = and i64 %24, 31, !dbg !12
  %26 = shl nuw nsw i64 %25, 19, !dbg !12
  %27 = add i64 %26, %23, !dbg !12
  %28 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !40
  %29 = zext nneg i32 %28 to i64, !dbg !13
  %30 = lshr i64 %29, 6, !dbg !13
  %31 = or disjoint i64 %11, 63, !dbg !14
  %32 = add nsw i64 %11, 64, !dbg !14
  %33 = icmp uge i64 %32, %31, !dbg !14
  %34 = icmp slt i32 %9, 0, !dbg !7
  %35 = select i1 %34, i64 %13, i64 %10, !dbg !7
  %36 = sdiv i64 %35, 2, !dbg !7
  %37 = sub nsw i64 0, %36, !dbg !7
  %38 = add nsw i64 %36, 1, !dbg !7
  %39 = select i1 %34, i64 %37, i64 %38, !dbg !7
  %.idx = shl nuw nsw i64 %19, 20, !dbg !12
  %40 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %41 = getelementptr [2 x i8], ptr addrspace(1) %40, i64 %23, !dbg !12
  %42 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %27, !dbg !12
  %43 = getelementptr i8, ptr addrspace(1) %42, i64 64, !dbg !12
  %44 = shl nsw i64 %8, 12, !dbg !16
  %45 = shl nsw i64 %6, 17, !dbg !16
  %46 = shl nsw i64 %8, 7, !dbg !16
  %47 = shl nsw i64 %6, 12, !dbg !16
  %48 = add nsw i64 %46, %47, !dbg !16
  %49 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %27, !dbg !12
  %50 = getelementptr i8, ptr addrspace(1) %49, i64 128, !dbg !12
  %51 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %27, !dbg !12
  %52 = getelementptr i8, ptr addrspace(1) %51, i64 192, !dbg !12
  br label %53, !dbg !7

53:                                               ; preds = %579, %4
  %54 = phi i64 [ %1141, %579 ], [ 0, %4 ], !dbg !7
  %55 = phi <4 x float> [ %916, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %56 = phi <4 x float> [ %932, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %57 = phi <8 x float> [ %1137, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %58 = phi <1 x float> [ %850, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %59 = phi <1 x float> [ %681, %579 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %60 = phi <4 x float> [ %994, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %61 = phi <4 x float> [ %1010, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %62 = phi <8 x float> [ %1138, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %63 = phi <4 x float> [ %1072, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %64 = phi <4 x float> [ %1088, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %65 = phi <8 x float> [ %1139, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %66 = phi <4 x float> [ %1120, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %67 = phi <4 x float> [ %1136, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %68 = phi <8 x float> [ %1140, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %69 = icmp slt i64 %54, %39, !dbg !7
  br i1 %69, label %70, label %1142, !dbg !7

70:                                               ; preds = %53
  %71 = shl nsw i64 %54, 7, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %28) #9, !dbg !12, !range !40, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local110 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local111 = and i64 %address.local110, 1536, !dbg !12
  %address.local112 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local111, !dbg !12
  %address.local113 = ptrtoint ptr addrspace(3) %address.local112 to i64, !dbg !12
  %address.local114 = trunc i64 %address.local113 to i32, !dbg !12
  %72 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local114), !dbg !12
  %73 = zext i32 %72 to i64, !dbg !12
  %74 = inttoptr i64 %73 to ptr addrspace(3), !dbg !12
  %address.local115 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local116 = and i64 %address.local115, 8064, !dbg !12
  %address.local117 = lshr i64 %address.local, 3, !dbg !12
  %address.local118 = and i64 %address.local117, 24, !dbg !12
  %address.local119 = or disjoint i64 %address.local116, %address.local118, !dbg !12
  %address.local120 = getelementptr [2 x i8], ptr addrspace(1) %41, i64 %address.local119, !dbg !12
  %75 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %76 = ptrtoint ptr addrspace(1) %address.local120 to i64, !dbg !12
  %77 = sub i64 %76, %75, !dbg !12
  %dma.buffer.offset836 = trunc i64 %77 to i32, !dbg !12
  %78 = trunc i64 %75 to i32, !dbg !12
  %79 = insertelement <4 x i32> poison, i32 %78, i64 0, !dbg !12
  %80 = lshr i64 %75, 32, !dbg !12
  %81 = trunc i64 %80 to i32, !dbg !12
  %82 = insertelement <4 x i32> %79, i32 %81, i64 1, !dbg !12
  %83 = insertelement <4 x i32> %82, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc837 = insertelement <4 x i32> %83, i32 131072, i64 3, !dbg !12
  %84 = ptrtoint ptr addrspace(3) %74 to i32, !dbg !12
  %dma.buffer.lds838 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %84), !dbg !12
  %85 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds838, i32 %dma.buffer.offset836, <4 x i32> %dma.buffer.rsrc837) #10, !dbg !12
  %86 = lshr i64 %54, 12, !dbg !16
  %87 = add nsw i64 %86, %8, !dbg !16
  %88 = lshr i64 %54, 5, !dbg !16
  %89 = and i64 %88, 127, !dbg !16
  %90 = shl i64 %87, 19, !dbg !16
  %91 = and i64 %90, 16252928, !dbg !16
  %92 = shl i64 %88, 19, !dbg !16
  %93 = shl i64 %54, 14, !dbg !16
  %94 = sub i64 %93, %92, !dbg !16
  %95 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %91, !dbg !16
  %96 = getelementptr [2 x i8], ptr addrspace(1) %95, i64 %89, !dbg !16
  %97 = getelementptr [2 x i8], ptr addrspace(1) %96, i64 %94, !dbg !16
  %address.local121 = and i64 %address.local110, 1536, !dbg !16
  %address.local122 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local121, !dbg !16
  %address.local123 = ptrtoint ptr addrspace(3) %address.local122 to i64, !dbg !16
  %address.local124 = trunc i64 %address.local123 to i32, !dbg !16
  %98 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local124), !dbg !16
  %99 = zext i32 %98 to i64, !dbg !16
  %100 = inttoptr i64 %99 to ptr addrspace(3), !dbg !16
  %address.local125 = lshr i64 %address.local, 4, !dbg !16
  %address.local126 = and i64 %address.local125, 8, !dbg !16
  %address.local127 = shl nuw nsw i64 %address.local, 7, !dbg !16
  %address.local128 = and i64 %address.local127, 16256, !dbg !16
  %address.local129 = or disjoint i64 %address.local126, %address.local128, !dbg !16
  %address.local130 = getelementptr [2 x i8], ptr addrspace(1) %97, i64 %address.local129, !dbg !16
  %101 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %102 = ptrtoint ptr addrspace(1) %address.local130 to i64, !dbg !16
  %103 = sub i64 %102, %101, !dbg !16
  %dma.buffer.offset833 = trunc i64 %103 to i32, !dbg !16
  %104 = trunc i64 %101 to i32, !dbg !16
  %105 = insertelement <4 x i32> poison, i32 %104, i64 0, !dbg !16
  %106 = lshr i64 %101, 32, !dbg !16
  %107 = trunc i64 %106 to i32, !dbg !16
  %108 = insertelement <4 x i32> %105, i32 %107, i64 1, !dbg !16
  %109 = insertelement <4 x i32> %108, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc834 = insertelement <4 x i32> %109, i32 131072, i64 3, !dbg !16
  %110 = ptrtoint ptr addrspace(3) %100 to i32, !dbg !16
  %dma.buffer.lds835 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %110), !dbg !16
  %111 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds835, i32 %dma.buffer.offset833, <4 x i32> %dma.buffer.rsrc834) #10, !dbg !16
  %address.local131 = or disjoint i64 %address.local, 256, !dbg !16
  %address.local132 = shl nuw nsw i64 %address.local131, 3, !dbg !16
  %address.local133 = and i64 %address.local132, 3584, !dbg !16
  %address.local134 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local133, !dbg !16
  %address.local135 = ptrtoint ptr addrspace(3) %address.local134 to i64, !dbg !16
  %address.local136 = trunc i64 %address.local135 to i32, !dbg !16
  %112 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local136), !dbg !16
  %113 = zext i32 %112 to i64, !dbg !16
  %114 = inttoptr i64 %113 to ptr addrspace(3), !dbg !16
  %address.local137 = lshr i64 %address.local131, 4, !dbg !16
  %address.local138 = and i64 %address.local137, 24, !dbg !16
  %address.local139 = shl nuw nsw i64 %address.local, 7, !dbg !16
  %address.local140 = and i64 %address.local139, 16256, !dbg !16
  %address.local141 = or disjoint i64 %address.local138, %address.local140, !dbg !16
  %address.local142 = getelementptr [2 x i8], ptr addrspace(1) %97, i64 %address.local141, !dbg !16
  %115 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %116 = ptrtoint ptr addrspace(1) %address.local142 to i64, !dbg !16
  %117 = sub i64 %116, %115, !dbg !16
  %dma.buffer.offset830 = trunc i64 %117 to i32, !dbg !16
  %118 = trunc i64 %115 to i32, !dbg !16
  %119 = insertelement <4 x i32> poison, i32 %118, i64 0, !dbg !16
  %120 = lshr i64 %115, 32, !dbg !16
  %121 = trunc i64 %120 to i32, !dbg !16
  %122 = insertelement <4 x i32> %119, i32 %121, i64 1, !dbg !16
  %123 = insertelement <4 x i32> %122, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc831 = insertelement <4 x i32> %123, i32 131072, i64 3, !dbg !16
  %124 = ptrtoint ptr addrspace(3) %114 to i32, !dbg !16
  %dma.buffer.lds832 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %124), !dbg !16
  %125 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds832, i32 %dma.buffer.offset830, <4 x i32> %dma.buffer.rsrc831) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %address.local143 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 20480), i64 %address.local111, !dbg !12
  %address.local144 = ptrtoint ptr addrspace(3) %address.local143 to i64, !dbg !12
  %address.local145 = trunc i64 %address.local144 to i32, !dbg !12
  %126 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !12
  %127 = zext i32 %126 to i64, !dbg !12
  %128 = inttoptr i64 %127 to ptr addrspace(3), !dbg !12
  %address.local146 = getelementptr [2 x i8], ptr addrspace(1) %43, i64 %address.local119, !dbg !12
  %129 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %130 = ptrtoint ptr addrspace(1) %address.local146 to i64, !dbg !12
  %131 = sub i64 %130, %129, !dbg !12
  %dma.buffer.offset827 = trunc i64 %131 to i32, !dbg !12
  %132 = trunc i64 %129 to i32, !dbg !12
  %133 = insertelement <4 x i32> poison, i32 %132, i64 0, !dbg !12
  %134 = lshr i64 %129, 32, !dbg !12
  %135 = trunc i64 %134 to i32, !dbg !12
  %136 = insertelement <4 x i32> %133, i32 %135, i64 1, !dbg !12
  %137 = insertelement <4 x i32> %136, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc828 = insertelement <4 x i32> %137, i32 131072, i64 3, !dbg !12
  %138 = ptrtoint ptr addrspace(3) %128 to i32, !dbg !12
  %dma.buffer.lds829 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %138), !dbg !12
  %139 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds829, i32 %dma.buffer.offset827, <4 x i32> %dma.buffer.rsrc828) #10, !dbg !12
  %140 = add i64 %54, %44, !dbg !16
  %141 = add i64 %140, %45, !dbg !16
  %142 = add i64 %141, 1024, !dbg !16
  %143 = icmp slt i64 %142, 0, !dbg !16
  %144 = sub i64 -1025, %141, !dbg !16
  %145 = select i1 %143, i64 %144, i64 %142, !dbg !16
  %146 = sdiv i64 %145, 4096, !dbg !16
  %.lobit2 = ashr i64 %142, 63, !dbg !16
  %147 = xor i64 %146, %.lobit2, !dbg !16
  %148 = add nsw i64 %88, %48, !dbg !16
  %149 = add nsw i64 %148, 32, !dbg !16
  %150 = icmp slt i64 %148, -32, !dbg !16
  %151 = sub nsw i64 -33, %148, !dbg !16
  %152 = select i1 %150, i64 %151, i64 %149, !dbg !16
  %153 = sdiv i64 %152, 128, !dbg !16
  %154 = sext i1 %150 to i64, !dbg !16
  %155 = xor i64 %153, %154, !dbg !16
  %156 = shl nsw i64 %155, 7, !dbg !16
  %157 = sub i64 %48, %156, !dbg !16
  %158 = shl i64 %147, 19, !dbg !16
  %159 = and i64 %158, 16252928, !dbg !16
  %160 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %159, !dbg !16
  %161 = getelementptr [2 x i8], ptr addrspace(1) %160, i64 %88, !dbg !16
  %162 = getelementptr [2 x i8], ptr addrspace(1) %161, i64 %157, !dbg !16
  %163 = getelementptr i8, ptr addrspace(1) %162, i64 64, !dbg !16
  %164 = getelementptr [2 x i8], ptr addrspace(1) %163, i64 %94, !dbg !16
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local121, !dbg !16
  %address.local148 = ptrtoint ptr addrspace(3) %address.local147 to i64, !dbg !16
  %address.local149 = trunc i64 %address.local148 to i32, !dbg !16
  %165 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local149), !dbg !16
  %166 = zext i32 %165 to i64, !dbg !16
  %167 = inttoptr i64 %166 to ptr addrspace(3), !dbg !16
  %address.local150 = getelementptr [2 x i8], ptr addrspace(1) %164, i64 %address.local129, !dbg !16
  %168 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %169 = ptrtoint ptr addrspace(1) %address.local150 to i64, !dbg !16
  %170 = sub i64 %169, %168, !dbg !16
  %dma.buffer.offset824 = trunc i64 %170 to i32, !dbg !16
  %171 = trunc i64 %168 to i32, !dbg !16
  %172 = insertelement <4 x i32> poison, i32 %171, i64 0, !dbg !16
  %173 = lshr i64 %168, 32, !dbg !16
  %174 = trunc i64 %173 to i32, !dbg !16
  %175 = insertelement <4 x i32> %172, i32 %174, i64 1, !dbg !16
  %176 = insertelement <4 x i32> %175, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc825 = insertelement <4 x i32> %176, i32 131072, i64 3, !dbg !16
  %177 = ptrtoint ptr addrspace(3) %167 to i32, !dbg !16
  %dma.buffer.lds826 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %177), !dbg !16
  %178 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds826, i32 %dma.buffer.offset824, <4 x i32> %dma.buffer.rsrc825) #10, !dbg !16
  %address.local151 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local133, !dbg !16
  %address.local152 = ptrtoint ptr addrspace(3) %address.local151 to i64, !dbg !16
  %address.local153 = trunc i64 %address.local152 to i32, !dbg !16
  %179 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local153), !dbg !16
  %180 = zext i32 %179 to i64, !dbg !16
  %181 = inttoptr i64 %180 to ptr addrspace(3), !dbg !16
  %address.local154 = getelementptr [2 x i8], ptr addrspace(1) %164, i64 %address.local141, !dbg !16
  %182 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %183 = ptrtoint ptr addrspace(1) %address.local154 to i64, !dbg !16
  %184 = sub i64 %183, %182, !dbg !16
  %dma.buffer.offset821 = trunc i64 %184 to i32, !dbg !16
  %185 = trunc i64 %182 to i32, !dbg !16
  %186 = insertelement <4 x i32> poison, i32 %185, i64 0, !dbg !16
  %187 = lshr i64 %182, 32, !dbg !16
  %188 = trunc i64 %187 to i32, !dbg !16
  %189 = insertelement <4 x i32> %186, i32 %188, i64 1, !dbg !16
  %190 = insertelement <4 x i32> %189, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc822 = insertelement <4 x i32> %190, i32 131072, i64 3, !dbg !16
  %191 = ptrtoint ptr addrspace(3) %181 to i32, !dbg !16
  %dma.buffer.lds823 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %191), !dbg !16
  %192 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds823, i32 %dma.buffer.offset821, <4 x i32> %dma.buffer.rsrc822) #10, !dbg !16
  %address.local155 = lshr i32 %address.thread, 2, !dbg !13
  %address.local156 = and i32 %address.local155, 4, !dbg !13
  %address.local157 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local158 = and i32 %address.local157, 384, !dbg !13
  %address.local159 = or disjoint i32 %address.local156, %address.local158, !dbg !13
  %address.local160 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local161 = and i32 %address.local160, 120, !dbg !13
  %address.local162 = or disjoint i32 %address.local159, %address.local161, !dbg !13
  %address.local163 = shl nuw nsw i32 %address.thread, 4, !dbg !13
  %address.local164 = and i32 %address.local163, 512, !dbg !13
  %address.local165 = or disjoint i32 %address.local162, %address.local164, !dbg !13
  %address.local166 = zext nneg i32 %address.local165 to i64, !dbg !13
  %address.local167 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local166, !dbg !13
  %193 = load <4 x half>, ptr addrspace(3) %address.local167, align 2, !dbg !13
  %address.local168 = lshr i32 %address.thread, 2, !dbg !13
  %address.local169 = and i32 %address.local168, 4, !dbg !13
  %address.local170 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local171 = and i32 %address.local170, 384, !dbg !13
  %address.local172 = or disjoint i32 %address.local169, %address.local171, !dbg !13
  %address.local173 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local174 = and i32 %address.local173, 120, !dbg !13
  %address.local175 = or disjoint i32 %address.local172, %address.local174, !dbg !13
  %address.local176 = shl nuw nsw i32 %address.thread, 4, !dbg !13
  %address.local177 = and i32 %address.local176, 512, !dbg !13
  %address.local178 = or disjoint i32 %address.local175, %address.local177, !dbg !13
  %address.local179 = or disjoint i32 %address.local178, 1024, !dbg !13
  %address.local180 = zext nneg i32 %address.local179 to i64, !dbg !13
  %address.local181 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local180, !dbg !13
  %194 = load <4 x half>, ptr addrspace(3) %address.local181, align 2, !dbg !13
  %address.local182 = lshr i32 %address.thread, 2, !dbg !17
  %address.local183 = and i32 %address.local182, 4, !dbg !17
  %address.local184 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local185 = and i32 %address.local184, 24, !dbg !17
  %address.local186 = or disjoint i32 %address.local183, %address.local185, !dbg !17
  %address.local187 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local188 = and i32 %address.local187, 1024, !dbg !17
  %address.local189 = or disjoint i32 %address.local186, %address.local188, !dbg !17
  %address.local190 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local191 = and i32 %address.local190, 384, !dbg !17
  %address.local192 = or disjoint i32 %address.local189, %address.local191, !dbg !17
  %address.local193 = zext nneg i32 %address.local192 to i64, !dbg !17
  %address.local194 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local193, !dbg !17
  %195 = load <4 x half>, ptr addrspace(3) %address.local194, align 2, !dbg !17
  %196 = fmul <4 x half> %193, splat (half 0xH3015), !dbg !13
  %address.local195 = lshr i32 %address.thread, 2, !dbg !17
  %address.local196 = and i32 %address.local195, 4, !dbg !17
  %address.local197 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local198 = and i32 %address.local197, 24, !dbg !17
  %address.local199 = or disjoint i32 %address.local196, %address.local198, !dbg !17
  %address.local200 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local201 = and i32 %address.local200, 1024, !dbg !17
  %address.local202 = or disjoint i32 %address.local199, %address.local201, !dbg !17
  %address.local203 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local204 = and i32 %address.local203, 384, !dbg !17
  %address.local205 = or disjoint i32 %address.local202, %address.local204, !dbg !17
  %address.local206 = or disjoint i32 %address.local205, 2048, !dbg !17
  %address.local207 = zext nneg i32 %address.local206 to i64, !dbg !17
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local207, !dbg !17
  %197 = load <4 x half>, ptr addrspace(3) %address.local208, align 2, !dbg !17
  %198 = fmul <4 x half> %194, splat (half 0xH3015), !dbg !13
  %address.local209 = lshr i32 %address.thread, 2, !dbg !17
  %address.local210 = and i32 %address.local209, 4, !dbg !17
  %address.local211 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local212 = and i32 %address.local211, 24, !dbg !17
  %address.local213 = or disjoint i32 %address.local210, %address.local212, !dbg !17
  %address.local214 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local215 = and i32 %address.local214, 1024, !dbg !17
  %address.local216 = or disjoint i32 %address.local213, %address.local215, !dbg !17
  %address.local217 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local218 = and i32 %address.local217, 384, !dbg !17
  %address.local219 = or disjoint i32 %address.local216, %address.local218, !dbg !17
  %address.local220 = or disjoint i32 %address.local219, 32, !dbg !17
  %address.local221 = zext nneg i32 %address.local220 to i64, !dbg !17
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local221, !dbg !17
  %199 = load <4 x half>, ptr addrspace(3) %address.local222, align 2, !dbg !17
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %195, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local223 = lshr i32 %address.thread, 2, !dbg !17
  %address.local224 = and i32 %address.local223, 4, !dbg !17
  %address.local225 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local226 = and i32 %address.local225, 24, !dbg !17
  %address.local227 = or disjoint i32 %address.local224, %address.local226, !dbg !17
  %address.local228 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local229 = and i32 %address.local228, 1024, !dbg !17
  %address.local230 = or disjoint i32 %address.local227, %address.local229, !dbg !17
  %address.local231 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local232 = and i32 %address.local231, 384, !dbg !17
  %address.local233 = or disjoint i32 %address.local230, %address.local232, !dbg !17
  %address.local234 = or disjoint i32 %address.local233, 2080, !dbg !17
  %address.local235 = zext nneg i32 %address.local234 to i64, !dbg !17
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local235, !dbg !17
  %201 = load <4 x half>, ptr addrspace(3) %address.local236, align 2, !dbg !17
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %198, <4 x half> %197, <4 x float> %200) #9, !dbg !17
  %address.local237 = lshr i32 %address.thread, 2, !dbg !17
  %address.local238 = and i32 %address.local237, 4, !dbg !17
  %address.local239 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local240 = and i32 %address.local239, 24, !dbg !17
  %address.local241 = or disjoint i32 %address.local238, %address.local240, !dbg !17
  %address.local242 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local243 = and i32 %address.local242, 1024, !dbg !17
  %address.local244 = or disjoint i32 %address.local241, %address.local243, !dbg !17
  %address.local245 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local246 = and i32 %address.local245, 384, !dbg !17
  %address.local247 = or disjoint i32 %address.local244, %address.local246, !dbg !17
  %address.local248 = or disjoint i32 %address.local247, 64, !dbg !17
  %address.local249 = zext nneg i32 %address.local248 to i64, !dbg !17
  %address.local250 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local249, !dbg !17
  %203 = load <4 x half>, ptr addrspace(3) %address.local250, align 2, !dbg !17
  %address.local251 = lshr i32 %address.thread, 2, !dbg !17
  %address.local252 = and i32 %address.local251, 4, !dbg !17
  %address.local253 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local254 = and i32 %address.local253, 24, !dbg !17
  %address.local255 = or disjoint i32 %address.local252, %address.local254, !dbg !17
  %address.local256 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local257 = and i32 %address.local256, 1024, !dbg !17
  %address.local258 = or disjoint i32 %address.local255, %address.local257, !dbg !17
  %address.local259 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local260 = and i32 %address.local259, 384, !dbg !17
  %address.local261 = or disjoint i32 %address.local258, %address.local260, !dbg !17
  %address.local262 = or disjoint i32 %address.local261, 2112, !dbg !17
  %address.local263 = zext nneg i32 %address.local262 to i64, !dbg !17
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local263, !dbg !17
  %204 = load <4 x half>, ptr addrspace(3) %address.local264, align 2, !dbg !17
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %199, <4 x float> zeroinitializer) #9, !dbg !17
  %206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %198, <4 x half> %201, <4 x float> %205) #9, !dbg !17
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %203, <4 x float> zeroinitializer) #9, !dbg !17
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %198, <4 x half> %204, <4 x float> %207) #9, !dbg !17
  %address.local265 = lshr i32 %address.thread, 2, !dbg !17
  %address.local266 = and i32 %address.local265, 4, !dbg !17
  %address.local267 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local268 = and i32 %address.local267, 24, !dbg !17
  %address.local269 = or disjoint i32 %address.local266, %address.local268, !dbg !17
  %address.local270 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local271 = and i32 %address.local270, 1024, !dbg !17
  %address.local272 = or disjoint i32 %address.local269, %address.local271, !dbg !17
  %address.local273 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local274 = and i32 %address.local273, 384, !dbg !17
  %address.local275 = or disjoint i32 %address.local272, %address.local274, !dbg !17
  %address.local276 = or disjoint i32 %address.local275, 96, !dbg !17
  %address.local277 = zext nneg i32 %address.local276 to i64, !dbg !17
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local277, !dbg !17
  %209 = load <4 x half>, ptr addrspace(3) %address.local278, align 2, !dbg !17
  %address.local279 = lshr i32 %address.thread, 2, !dbg !17
  %address.local280 = and i32 %address.local279, 4, !dbg !17
  %address.local281 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local282 = and i32 %address.local281, 24, !dbg !17
  %address.local283 = or disjoint i32 %address.local280, %address.local282, !dbg !17
  %address.local284 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local285 = and i32 %address.local284, 1024, !dbg !17
  %address.local286 = or disjoint i32 %address.local283, %address.local285, !dbg !17
  %address.local287 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local288 = and i32 %address.local287, 384, !dbg !17
  %address.local289 = or disjoint i32 %address.local286, %address.local288, !dbg !17
  %address.local290 = or disjoint i32 %address.local289, 2144, !dbg !17
  %address.local291 = zext nneg i32 %address.local290 to i64, !dbg !17
  %address.local292 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local291, !dbg !17
  %210 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !17
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %209, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local293 = lshr i32 %address.thread, 2, !dbg !17
  %address.local294 = and i32 %address.local293, 4, !dbg !17
  %address.local295 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local296 = and i32 %address.local295, 24, !dbg !17
  %address.local297 = or disjoint i32 %address.local294, %address.local296, !dbg !17
  %address.local298 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local299 = and i32 %address.local298, 1024, !dbg !17
  %address.local300 = or disjoint i32 %address.local297, %address.local299, !dbg !17
  %address.local301 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local302 = and i32 %address.local301, 384, !dbg !17
  %address.local303 = or disjoint i32 %address.local300, %address.local302, !dbg !17
  %address.local304 = or disjoint i32 %address.local303, 512, !dbg !17
  %address.local305 = zext nneg i32 %address.local304 to i64, !dbg !17
  %address.local306 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local305, !dbg !17
  %212 = load <4 x half>, ptr addrspace(3) %address.local306, align 2, !dbg !17
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %198, <4 x half> %210, <4 x float> %211) #9, !dbg !17
  %address.local307 = lshr i32 %address.thread, 2, !dbg !17
  %address.local308 = and i32 %address.local307, 4, !dbg !17
  %address.local309 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local310 = and i32 %address.local309, 24, !dbg !17
  %address.local311 = or disjoint i32 %address.local308, %address.local310, !dbg !17
  %address.local312 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local313 = and i32 %address.local312, 1024, !dbg !17
  %address.local314 = or disjoint i32 %address.local311, %address.local313, !dbg !17
  %address.local315 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local316 = and i32 %address.local315, 384, !dbg !17
  %address.local317 = or disjoint i32 %address.local314, %address.local316, !dbg !17
  %address.local318 = or disjoint i32 %address.local317, 2560, !dbg !17
  %address.local319 = zext nneg i32 %address.local318 to i64, !dbg !17
  %address.local320 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local319, !dbg !17
  %214 = load <4 x half>, ptr addrspace(3) %address.local320, align 2, !dbg !17
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %212, <4 x float> zeroinitializer) #9, !dbg !17
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %198, <4 x half> %214, <4 x float> %215) #9, !dbg !17
  %address.local321 = lshr i32 %address.thread, 2, !dbg !17
  %address.local322 = and i32 %address.local321, 4, !dbg !17
  %address.local323 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local324 = and i32 %address.local323, 24, !dbg !17
  %address.local325 = or disjoint i32 %address.local322, %address.local324, !dbg !17
  %address.local326 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local327 = and i32 %address.local326, 1024, !dbg !17
  %address.local328 = or disjoint i32 %address.local325, %address.local327, !dbg !17
  %address.local329 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local330 = and i32 %address.local329, 384, !dbg !17
  %address.local331 = or disjoint i32 %address.local328, %address.local330, !dbg !17
  %address.local332 = or disjoint i32 %address.local331, 544, !dbg !17
  %address.local333 = zext nneg i32 %address.local332 to i64, !dbg !17
  %address.local334 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local333, !dbg !17
  %217 = load <4 x half>, ptr addrspace(3) %address.local334, align 2, !dbg !17
  %address.local335 = lshr i32 %address.thread, 2, !dbg !17
  %address.local336 = and i32 %address.local335, 4, !dbg !17
  %address.local337 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local338 = and i32 %address.local337, 24, !dbg !17
  %address.local339 = or disjoint i32 %address.local336, %address.local338, !dbg !17
  %address.local340 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local341 = and i32 %address.local340, 1024, !dbg !17
  %address.local342 = or disjoint i32 %address.local339, %address.local341, !dbg !17
  %address.local343 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local344 = and i32 %address.local343, 384, !dbg !17
  %address.local345 = or disjoint i32 %address.local342, %address.local344, !dbg !17
  %address.local346 = or disjoint i32 %address.local345, 2592, !dbg !17
  %address.local347 = zext nneg i32 %address.local346 to i64, !dbg !17
  %address.local348 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local347, !dbg !17
  %218 = load <4 x half>, ptr addrspace(3) %address.local348, align 2, !dbg !17
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %217, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local349 = lshr i32 %address.thread, 2, !dbg !17
  %address.local350 = and i32 %address.local349, 4, !dbg !17
  %address.local351 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local352 = and i32 %address.local351, 24, !dbg !17
  %address.local353 = or disjoint i32 %address.local350, %address.local352, !dbg !17
  %address.local354 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local355 = and i32 %address.local354, 1024, !dbg !17
  %address.local356 = or disjoint i32 %address.local353, %address.local355, !dbg !17
  %address.local357 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local358 = and i32 %address.local357, 384, !dbg !17
  %address.local359 = or disjoint i32 %address.local356, %address.local358, !dbg !17
  %address.local360 = or disjoint i32 %address.local359, 576, !dbg !17
  %address.local361 = zext nneg i32 %address.local360 to i64, !dbg !17
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local361, !dbg !17
  %220 = load <4 x half>, ptr addrspace(3) %address.local362, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %198, <4 x half> %218, <4 x float> %219) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %address.local363 = lshr i32 %address.thread, 2, !dbg !17
  %address.local364 = and i32 %address.local363, 4, !dbg !17
  %address.local365 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local366 = and i32 %address.local365, 24, !dbg !17
  %address.local367 = or disjoint i32 %address.local364, %address.local366, !dbg !17
  %address.local368 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local369 = and i32 %address.local368, 1024, !dbg !17
  %address.local370 = or disjoint i32 %address.local367, %address.local369, !dbg !17
  %address.local371 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local372 = and i32 %address.local371, 384, !dbg !17
  %address.local373 = or disjoint i32 %address.local370, %address.local372, !dbg !17
  %address.local374 = or disjoint i32 %address.local373, 2624, !dbg !17
  %address.local375 = zext nneg i32 %address.local374 to i64, !dbg !17
  %address.local376 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local375, !dbg !17
  %222 = load <4 x half>, ptr addrspace(3) %address.local376, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %220, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %198, <4 x half> %222, <4 x float> %223) #9, !dbg !17
  %address.local377 = lshr i32 %address.thread, 2, !dbg !17
  %address.local378 = and i32 %address.local377, 4, !dbg !17
  %address.local379 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local380 = and i32 %address.local379, 24, !dbg !17
  %address.local381 = or disjoint i32 %address.local378, %address.local380, !dbg !17
  %address.local382 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local383 = and i32 %address.local382, 1024, !dbg !17
  %address.local384 = or disjoint i32 %address.local381, %address.local383, !dbg !17
  %address.local385 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local386 = and i32 %address.local385, 384, !dbg !17
  %address.local387 = or disjoint i32 %address.local384, %address.local386, !dbg !17
  %address.local388 = or disjoint i32 %address.local387, 608, !dbg !17
  %address.local389 = zext nneg i32 %address.local388 to i64, !dbg !17
  %address.local390 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local389, !dbg !17
  %225 = load <4 x half>, ptr addrspace(3) %address.local390, align 2, !dbg !17
  %address.local391 = lshr i32 %address.thread, 2, !dbg !17
  %address.local392 = and i32 %address.local391, 4, !dbg !17
  %address.local393 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local394 = and i32 %address.local393, 24, !dbg !17
  %address.local395 = or disjoint i32 %address.local392, %address.local394, !dbg !17
  %address.local396 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local397 = and i32 %address.local396, 1024, !dbg !17
  %address.local398 = or disjoint i32 %address.local395, %address.local397, !dbg !17
  %address.local399 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local400 = and i32 %address.local399, 384, !dbg !17
  %address.local401 = or disjoint i32 %address.local398, %address.local400, !dbg !17
  %address.local402 = or disjoint i32 %address.local401, 2656, !dbg !17
  %address.local403 = zext nneg i32 %address.local402 to i64, !dbg !17
  %address.local404 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local403, !dbg !17
  %226 = load <4 x half>, ptr addrspace(3) %address.local404, align 2, !dbg !17
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %196, <4 x half> %225, <4 x float> zeroinitializer) #9, !dbg !17
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %198, <4 x half> %226, <4 x float> %227) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %229 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local114), !dbg !12
  %230 = zext i32 %229 to i64, !dbg !12
  %231 = inttoptr i64 %230 to ptr addrspace(3), !dbg !12
  %address.local405 = getelementptr [2 x i8], ptr addrspace(1) %50, i64 %address.local119, !dbg !12
  %232 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %233 = ptrtoint ptr addrspace(1) %address.local405 to i64, !dbg !12
  %234 = sub i64 %233, %232, !dbg !12
  %dma.buffer.offset818 = trunc i64 %234 to i32, !dbg !12
  %235 = trunc i64 %232 to i32, !dbg !12
  %236 = insertelement <4 x i32> poison, i32 %235, i64 0, !dbg !12
  %237 = lshr i64 %232, 32, !dbg !12
  %238 = trunc i64 %237 to i32, !dbg !12
  %239 = insertelement <4 x i32> %236, i32 %238, i64 1, !dbg !12
  %240 = insertelement <4 x i32> %239, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc819 = insertelement <4 x i32> %240, i32 131072, i64 3, !dbg !12
  %241 = ptrtoint ptr addrspace(3) %231 to i32, !dbg !12
  %dma.buffer.lds820 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %241), !dbg !12
  %242 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds820, i32 %dma.buffer.offset818, <4 x i32> %dma.buffer.rsrc819) #10, !dbg !12
  %243 = add i64 %141, 2048, !dbg !16
  %244 = icmp slt i64 %243, 0, !dbg !16
  %245 = sub i64 -2049, %141, !dbg !16
  %246 = select i1 %244, i64 %245, i64 %243, !dbg !16
  %247 = sdiv i64 %246, 4096, !dbg !16
  %.lobit3 = ashr i64 %243, 63, !dbg !16
  %248 = xor i64 %247, %.lobit3, !dbg !16
  %249 = add nsw i64 %148, 64, !dbg !16
  %250 = icmp slt i64 %148, -64, !dbg !16
  %251 = sub nsw i64 -65, %148, !dbg !16
  %252 = select i1 %250, i64 %251, i64 %249, !dbg !16
  %253 = sdiv i64 %252, 128, !dbg !16
  %254 = sext i1 %250 to i64, !dbg !16
  %255 = xor i64 %253, %254, !dbg !16
  %256 = shl nsw i64 %255, 7, !dbg !16
  %257 = sub i64 %48, %256, !dbg !16
  %258 = shl i64 %248, 19, !dbg !16
  %259 = and i64 %258, 16252928, !dbg !16
  %260 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %259, !dbg !16
  %261 = getelementptr [2 x i8], ptr addrspace(1) %260, i64 %88, !dbg !16
  %262 = getelementptr [2 x i8], ptr addrspace(1) %261, i64 %257, !dbg !16
  %263 = getelementptr i8, ptr addrspace(1) %262, i64 128, !dbg !16
  %264 = getelementptr [2 x i8], ptr addrspace(1) %263, i64 %94, !dbg !16
  %265 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local124), !dbg !16
  %266 = zext i32 %265 to i64, !dbg !16
  %267 = inttoptr i64 %266 to ptr addrspace(3), !dbg !16
  %address.local406 = getelementptr [2 x i8], ptr addrspace(1) %264, i64 %address.local129, !dbg !16
  %268 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %269 = ptrtoint ptr addrspace(1) %address.local406 to i64, !dbg !16
  %270 = sub i64 %269, %268, !dbg !16
  %dma.buffer.offset815 = trunc i64 %270 to i32, !dbg !16
  %271 = trunc i64 %268 to i32, !dbg !16
  %272 = insertelement <4 x i32> poison, i32 %271, i64 0, !dbg !16
  %273 = lshr i64 %268, 32, !dbg !16
  %274 = trunc i64 %273 to i32, !dbg !16
  %275 = insertelement <4 x i32> %272, i32 %274, i64 1, !dbg !16
  %276 = insertelement <4 x i32> %275, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc816 = insertelement <4 x i32> %276, i32 131072, i64 3, !dbg !16
  %277 = ptrtoint ptr addrspace(3) %267 to i32, !dbg !16
  %dma.buffer.lds817 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %277), !dbg !16
  %278 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds817, i32 %dma.buffer.offset815, <4 x i32> %dma.buffer.rsrc816) #10, !dbg !16
  %279 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local136), !dbg !16
  %280 = zext i32 %279 to i64, !dbg !16
  %281 = inttoptr i64 %280 to ptr addrspace(3), !dbg !16
  %address.local407 = getelementptr [2 x i8], ptr addrspace(1) %264, i64 %address.local141, !dbg !16
  %282 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %283 = ptrtoint ptr addrspace(1) %address.local407 to i64, !dbg !16
  %284 = sub i64 %283, %282, !dbg !16
  %dma.buffer.offset812 = trunc i64 %284 to i32, !dbg !16
  %285 = trunc i64 %282 to i32, !dbg !16
  %286 = insertelement <4 x i32> poison, i32 %285, i64 0, !dbg !16
  %287 = lshr i64 %282, 32, !dbg !16
  %288 = trunc i64 %287 to i32, !dbg !16
  %289 = insertelement <4 x i32> %286, i32 %288, i64 1, !dbg !16
  %290 = insertelement <4 x i32> %289, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc813 = insertelement <4 x i32> %290, i32 131072, i64 3, !dbg !16
  %291 = ptrtoint ptr addrspace(3) %281 to i32, !dbg !16
  %dma.buffer.lds814 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %291), !dbg !16
  %292 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds814, i32 %dma.buffer.offset812, <4 x i32> %dma.buffer.rsrc813) #10, !dbg !16
  %address.local408 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 20480), i64 %address.local166, !dbg !13
  %293 = load <4 x half>, ptr addrspace(3) %address.local408, align 2, !dbg !13
  %294 = fmul <4 x half> %293, splat (half 0xH3015), !dbg !13
  %address.local409 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 20480), i64 %address.local180, !dbg !13
  %295 = load <4 x half>, ptr addrspace(3) %address.local409, align 2, !dbg !13
  %address.local410 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local193, !dbg !17
  %296 = load <4 x half>, ptr addrspace(3) %address.local410, align 2, !dbg !17
  %address.local411 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !17
  %297 = load <4 x half>, ptr addrspace(3) %address.local411, align 2, !dbg !17
  %298 = fmul <4 x half> %295, splat (half 0xH3015), !dbg !13
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %296, <4 x float> %202) #9, !dbg !17
  %address.local412 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local221, !dbg !17
  %300 = load <4 x half>, ptr addrspace(3) %address.local412, align 2, !dbg !17
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %297, <4 x float> %299) #9, !dbg !17
  %address.local413 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local235, !dbg !17
  %302 = load <4 x half>, ptr addrspace(3) %address.local413, align 2, !dbg !17
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %300, <4 x float> %206) #9, !dbg !17
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %302, <4 x float> %303) #9, !dbg !17
  %address.local414 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local249, !dbg !17
  %305 = load <4 x half>, ptr addrspace(3) %address.local414, align 2, !dbg !17
  %address.local415 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local263, !dbg !17
  %306 = load <4 x half>, ptr addrspace(3) %address.local415, align 2, !dbg !17
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %305, <4 x float> %208) #9, !dbg !17
  %address.local416 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local277, !dbg !17
  %308 = load <4 x half>, ptr addrspace(3) %address.local416, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %306, <4 x float> %307) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %address.local417 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local291, !dbg !17
  %310 = load <4 x half>, ptr addrspace(3) %address.local417, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %308, <4 x float> %213) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %310, <4 x float> %311) #9, !dbg !17
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local305, !dbg !17
  %313 = load <4 x half>, ptr addrspace(3) %address.local418, align 2, !dbg !17
  %address.local419 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local319, !dbg !17
  %314 = load <4 x half>, ptr addrspace(3) %address.local419, align 2, !dbg !17
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %313, <4 x float> %216) #9, !dbg !17
  %address.local420 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local333, !dbg !17
  %316 = load <4 x half>, ptr addrspace(3) %address.local420, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %314, <4 x float> %315) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %address.local421 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local347, !dbg !17
  %318 = load <4 x half>, ptr addrspace(3) %address.local421, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %316, <4 x float> %221) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %318, <4 x float> %319) #9, !dbg !17
  %address.local422 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local361, !dbg !17
  %321 = load <4 x half>, ptr addrspace(3) %address.local422, align 2, !dbg !17
  %address.local423 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local375, !dbg !17
  %322 = load <4 x half>, ptr addrspace(3) %address.local423, align 2, !dbg !17
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %321, <4 x float> %224) #9, !dbg !17
  %address.local424 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local389, !dbg !17
  %324 = load <4 x half>, ptr addrspace(3) %address.local424, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %322, <4 x float> %323) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %address.local425 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local403, !dbg !17
  %326 = load <4 x half>, ptr addrspace(3) %address.local425, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %294, <4 x half> %324, <4 x float> %228) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %298, <4 x half> %326, <4 x float> %327) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %329 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !12
  %330 = zext i32 %329 to i64, !dbg !12
  %331 = inttoptr i64 %330 to ptr addrspace(3), !dbg !12
  %address.local426 = getelementptr [2 x i8], ptr addrspace(1) %52, i64 %address.local119, !dbg !12
  %332 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %333 = ptrtoint ptr addrspace(1) %address.local426 to i64, !dbg !12
  %334 = sub i64 %333, %332, !dbg !12
  %dma.buffer.offset809 = trunc i64 %334 to i32, !dbg !12
  %335 = trunc i64 %332 to i32, !dbg !12
  %336 = insertelement <4 x i32> poison, i32 %335, i64 0, !dbg !12
  %337 = lshr i64 %332, 32, !dbg !12
  %338 = trunc i64 %337 to i32, !dbg !12
  %339 = insertelement <4 x i32> %336, i32 %338, i64 1, !dbg !12
  %340 = insertelement <4 x i32> %339, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc810 = insertelement <4 x i32> %340, i32 131072, i64 3, !dbg !12
  %341 = ptrtoint ptr addrspace(3) %331 to i32, !dbg !12
  %dma.buffer.lds811 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %341), !dbg !12
  %342 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds811, i32 %dma.buffer.offset809, <4 x i32> %dma.buffer.rsrc810) #10, !dbg !12
  %343 = add i64 %141, 3072, !dbg !16
  %344 = icmp slt i64 %343, 0, !dbg !16
  %345 = sub i64 -3073, %141, !dbg !16
  %346 = select i1 %344, i64 %345, i64 %343, !dbg !16
  %347 = sdiv i64 %346, 4096, !dbg !16
  %.lobit4 = ashr i64 %343, 63, !dbg !16
  %348 = xor i64 %347, %.lobit4, !dbg !16
  %349 = add nsw i64 %148, 96, !dbg !16
  %350 = icmp slt i64 %148, -96, !dbg !16
  %351 = sub nsw i64 -97, %148, !dbg !16
  %352 = select i1 %350, i64 %351, i64 %349, !dbg !16
  %353 = sdiv i64 %352, 128, !dbg !16
  %354 = sext i1 %350 to i64, !dbg !16
  %355 = xor i64 %353, %354, !dbg !16
  %356 = shl nsw i64 %355, 7, !dbg !16
  %357 = sub i64 %48, %356, !dbg !16
  %358 = shl i64 %348, 19, !dbg !16
  %359 = and i64 %358, 16252928, !dbg !16
  %360 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %359, !dbg !16
  %361 = getelementptr [2 x i8], ptr addrspace(1) %360, i64 %88, !dbg !16
  %362 = getelementptr [2 x i8], ptr addrspace(1) %361, i64 %357, !dbg !16
  %363 = getelementptr i8, ptr addrspace(1) %362, i64 192, !dbg !16
  %364 = getelementptr [2 x i8], ptr addrspace(1) %363, i64 %94, !dbg !16
  %365 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local149), !dbg !16
  %366 = zext i32 %365 to i64, !dbg !16
  %367 = inttoptr i64 %366 to ptr addrspace(3), !dbg !16
  %address.local427 = getelementptr [2 x i8], ptr addrspace(1) %364, i64 %address.local129, !dbg !16
  %368 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %369 = ptrtoint ptr addrspace(1) %address.local427 to i64, !dbg !16
  %370 = sub i64 %369, %368, !dbg !16
  %dma.buffer.offset806 = trunc i64 %370 to i32, !dbg !16
  %371 = trunc i64 %368 to i32, !dbg !16
  %372 = insertelement <4 x i32> poison, i32 %371, i64 0, !dbg !16
  %373 = lshr i64 %368, 32, !dbg !16
  %374 = trunc i64 %373 to i32, !dbg !16
  %375 = insertelement <4 x i32> %372, i32 %374, i64 1, !dbg !16
  %376 = insertelement <4 x i32> %375, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc807 = insertelement <4 x i32> %376, i32 131072, i64 3, !dbg !16
  %377 = ptrtoint ptr addrspace(3) %367 to i32, !dbg !16
  %dma.buffer.lds808 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %377), !dbg !16
  %378 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds808, i32 %dma.buffer.offset806, <4 x i32> %dma.buffer.rsrc807) #10, !dbg !16
  %379 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local153), !dbg !16
  %380 = zext i32 %379 to i64, !dbg !16
  %381 = inttoptr i64 %380 to ptr addrspace(3), !dbg !16
  %address.local428 = getelementptr [2 x i8], ptr addrspace(1) %364, i64 %address.local141, !dbg !16
  %382 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %383 = ptrtoint ptr addrspace(1) %address.local428 to i64, !dbg !16
  %384 = sub i64 %383, %382, !dbg !16
  %dma.buffer.offset803 = trunc i64 %384 to i32, !dbg !16
  %385 = trunc i64 %382 to i32, !dbg !16
  %386 = insertelement <4 x i32> poison, i32 %385, i64 0, !dbg !16
  %387 = lshr i64 %382, 32, !dbg !16
  %388 = trunc i64 %387 to i32, !dbg !16
  %389 = insertelement <4 x i32> %386, i32 %388, i64 1, !dbg !16
  %390 = insertelement <4 x i32> %389, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc804 = insertelement <4 x i32> %390, i32 131072, i64 3, !dbg !16
  %391 = ptrtoint ptr addrspace(3) %381 to i32, !dbg !16
  %dma.buffer.lds805 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %391), !dbg !16
  %392 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds805, i32 %dma.buffer.offset803, <4 x i32> %dma.buffer.rsrc804) #10, !dbg !16
  %393 = load <4 x half>, ptr addrspace(3) %address.local167, align 2, !dbg !13
  %394 = fmul <4 x half> %393, splat (half 0xH3015), !dbg !13
  %395 = load <4 x half>, ptr addrspace(3) %address.local181, align 2, !dbg !13
  %396 = load <4 x half>, ptr addrspace(3) %address.local194, align 2, !dbg !17
  %397 = load <4 x half>, ptr addrspace(3) %address.local208, align 2, !dbg !17
  %398 = fmul <4 x half> %395, splat (half 0xH3015), !dbg !13
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %396, <4 x float> %301) #9, !dbg !17
  %400 = load <4 x half>, ptr addrspace(3) %address.local222, align 2, !dbg !17
  %401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %397, <4 x float> %399) #9, !dbg !17
  %402 = load <4 x half>, ptr addrspace(3) %address.local236, align 2, !dbg !17
  %403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %400, <4 x float> %304) #9, !dbg !17
  %404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %402, <4 x float> %403) #9, !dbg !17
  %405 = load <4 x half>, ptr addrspace(3) %address.local250, align 2, !dbg !17
  %406 = load <4 x half>, ptr addrspace(3) %address.local264, align 2, !dbg !17
  %407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %405, <4 x float> %309) #9, !dbg !17
  %408 = load <4 x half>, ptr addrspace(3) %address.local278, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %406, <4 x float> %407) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %410 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %408, <4 x float> %312) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %410, <4 x float> %411) #9, !dbg !17
  %413 = load <4 x half>, ptr addrspace(3) %address.local306, align 2, !dbg !17
  %414 = load <4 x half>, ptr addrspace(3) %address.local320, align 2, !dbg !17
  %415 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %413, <4 x float> %317) #9, !dbg !17
  %416 = load <4 x half>, ptr addrspace(3) %address.local334, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %417 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %414, <4 x float> %415) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %418 = load <4 x half>, ptr addrspace(3) %address.local348, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %416, <4 x float> %320) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %420 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %418, <4 x float> %419) #9, !dbg !17
  %421 = load <4 x half>, ptr addrspace(3) %address.local362, align 2, !dbg !17
  %422 = load <4 x half>, ptr addrspace(3) %address.local376, align 2, !dbg !17
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %421, <4 x float> %325) #9, !dbg !17
  %424 = load <4 x half>, ptr addrspace(3) %address.local390, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %422, <4 x float> %423) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %426 = load <4 x half>, ptr addrspace(3) %address.local404, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %394, <4 x half> %424, <4 x float> %328) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %398, <4 x half> %426, <4 x float> %427) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %429 = add nsw i64 %88, %8, !dbg !18
  %430 = shl i64 %429, 19, !dbg !18
  %431 = and i64 %430, 16252928, !dbg !18
  %432 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %431, !dbg !18
  %433 = getelementptr [2 x i8], ptr addrspace(1) %432, i64 %94, !dbg !18
  %434 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local124), !dbg !18
  %435 = zext i32 %434 to i64, !dbg !18
  %436 = inttoptr i64 %435 to ptr addrspace(3), !dbg !18
  %address.local429 = shl nuw nsw i64 %address.local, 5, !dbg !18
  %address.local430 = and i64 %address.local429, 8064, !dbg !18
  %address.local431 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local432 = and i64 %address.local431, 24, !dbg !18
  %address.local433 = or disjoint i64 %address.local430, %address.local432, !dbg !18
  %address.local434 = getelementptr [2 x i8], ptr addrspace(1) %433, i64 %address.local433, !dbg !18
  %437 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %438 = ptrtoint ptr addrspace(1) %address.local434 to i64, !dbg !18
  %439 = sub i64 %438, %437, !dbg !18
  %dma.buffer.offset800 = trunc i64 %439 to i32, !dbg !18
  %440 = trunc i64 %437 to i32, !dbg !18
  %441 = insertelement <4 x i32> poison, i32 %440, i64 0, !dbg !18
  %442 = lshr i64 %437, 32, !dbg !18
  %443 = trunc i64 %442 to i32, !dbg !18
  %444 = insertelement <4 x i32> %441, i32 %443, i64 1, !dbg !18
  %445 = insertelement <4 x i32> %444, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc801 = insertelement <4 x i32> %445, i32 131072, i64 3, !dbg !18
  %446 = ptrtoint ptr addrspace(3) %436 to i32, !dbg !18
  %dma.buffer.lds802 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %446), !dbg !18
  %447 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds802, i32 %dma.buffer.offset800, <4 x i32> %dma.buffer.rsrc801) #10, !dbg !18
  %448 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local136), !dbg !18
  %449 = zext i32 %448 to i64, !dbg !18
  %450 = inttoptr i64 %449 to ptr addrspace(3), !dbg !18
  %address.local435 = shl nuw nsw i64 %address.local131, 5, !dbg !18
  %address.local436 = and i64 %address.local435, 16256, !dbg !18
  %address.local437 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local438 = and i64 %address.local437, 24, !dbg !18
  %address.local439 = or disjoint i64 %address.local436, %address.local438, !dbg !18
  %address.local440 = getelementptr [2 x i8], ptr addrspace(1) %433, i64 %address.local439, !dbg !18
  %451 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %452 = ptrtoint ptr addrspace(1) %address.local440 to i64, !dbg !18
  %453 = sub i64 %452, %451, !dbg !18
  %dma.buffer.offset797 = trunc i64 %453 to i32, !dbg !18
  %454 = trunc i64 %451 to i32, !dbg !18
  %455 = insertelement <4 x i32> poison, i32 %454, i64 0, !dbg !18
  %456 = lshr i64 %451, 32, !dbg !18
  %457 = trunc i64 %456 to i32, !dbg !18
  %458 = insertelement <4 x i32> %455, i32 %457, i64 1, !dbg !18
  %459 = insertelement <4 x i32> %458, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc798 = insertelement <4 x i32> %459, i32 131072, i64 3, !dbg !18
  %460 = ptrtoint ptr addrspace(3) %450 to i32, !dbg !18
  %dma.buffer.lds799 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %460), !dbg !18
  %461 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds799, i32 %dma.buffer.offset797, <4 x i32> %dma.buffer.rsrc798) #10, !dbg !18
  %462 = load <4 x half>, ptr addrspace(3) %address.local408, align 2, !dbg !13
  %463 = fmul <4 x half> %462, splat (half 0xH3015), !dbg !13
  %464 = load <4 x half>, ptr addrspace(3) %address.local409, align 2, !dbg !13
  %465 = load <4 x half>, ptr addrspace(3) %address.local410, align 2, !dbg !17
  %466 = load <4 x half>, ptr addrspace(3) %address.local411, align 2, !dbg !17
  %467 = fmul <4 x half> %464, splat (half 0xH3015), !dbg !13
  %468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %463, <4 x half> %465, <4 x float> %401) #9, !dbg !17
  %469 = load <4 x half>, ptr addrspace(3) %address.local412, align 2, !dbg !17
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %466, <4 x float> %468) #9, !dbg !17
  %471 = load <4 x half>, ptr addrspace(3) %address.local413, align 2, !dbg !17
  %472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %463, <4 x half> %469, <4 x float> %404) #9, !dbg !17
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %471, <4 x float> %472) #9, !dbg !17
  %474 = load <4 x half>, ptr addrspace(3) %address.local414, align 2, !dbg !17
  %475 = load <4 x half>, ptr addrspace(3) %address.local415, align 2, !dbg !17
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %463, <4 x half> %474, <4 x float> %409) #9, !dbg !17
  %477 = load <4 x half>, ptr addrspace(3) %address.local416, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %475, <4 x float> %476) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %479 = load <4 x half>, ptr addrspace(3) %address.local417, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %463, <4 x half> %477, <4 x float> %412) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %479, <4 x float> %480) #9, !dbg !17
  %482 = load <4 x half>, ptr addrspace(3) %address.local418, align 2, !dbg !17
  %483 = load <4 x half>, ptr addrspace(3) %address.local419, align 2, !dbg !17
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %463, <4 x half> %482, <4 x float> %417) #9, !dbg !17
  %485 = load <4 x half>, ptr addrspace(3) %address.local420, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %483, <4 x float> %484) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %487 = load <4 x half>, ptr addrspace(3) %address.local421, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %463, <4 x half> %485, <4 x float> %420) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %487, <4 x float> %488) #9, !dbg !17
  %490 = load <4 x half>, ptr addrspace(3) %address.local422, align 2, !dbg !17
  %491 = load <4 x half>, ptr addrspace(3) %address.local423, align 2, !dbg !17
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %463, <4 x half> %490, <4 x float> %425) #9, !dbg !17
  %493 = load <4 x half>, ptr addrspace(3) %address.local424, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %491, <4 x float> %492) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %495 = load <4 x half>, ptr addrspace(3) %address.local425, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %463, <4 x half> %493, <4 x float> %428) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %467, <4 x half> %495, <4 x float> %496) #9, !dbg !17
  %498 = or disjoint i64 %71, 127, !dbg !14
  %499 = icmp uge i64 %11, %498, !dbg !14
  %500 = and i1 %33, %499, !dbg !14
  br i1 %500, label %501, label %502, !dbg !14

501:                                              ; preds = %70
  br label %577, !dbg !14

502:                                              ; preds = %70
  %503 = add nsw i64 %11, %29, !dbg !14
  %504 = shl nuw nsw i64 %30, 4, !dbg !14
  %505 = add nsw i64 %503, %504, !dbg !14
  %506 = and i32 %28, 240, !dbg !14
  %index.compact = zext nneg i32 %506 to i64, !dbg !14
  %507 = sub nsw i64 %505, %index.compact, !dbg !14
  %508 = add nsw i64 %507, 1, !dbg !24
  %509 = and i64 %29, 48, !dbg !14
  %510 = or disjoint i64 %71, %509, !dbg !14
  %511 = lshr i64 %508, 32, !dbg !25
  %compare.high = trunc i64 %511 to i32, !dbg !25
  %512 = lshr i64 %510, 32, !dbg !25
  %compare.high658 = trunc i64 %512 to i32, !dbg !25
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high658, !dbg !25
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high658, !dbg !25
  %compare.low.base = trunc i64 %508 to i32, !dbg !25
  %compare.low.base659 = trunc i64 %510 to i32, !dbg !25
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base659, !dbg !25
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !25
  %513 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 1) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !25
  %compare.low660 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !25
  %compare.result661 = select i1 %compare.high.equal, i1 %compare.low660, i1 %compare.high.order, !dbg !25
  %514 = select i1 %compare.result661, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base662 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 2) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local663 = or i32 %compare.local.base662, 2, !dbg !25
  %compare.low664 = icmp ugt i32 %compare.low.base, %compare.local663, !dbg !25
  %compare.result665 = select i1 %compare.high.equal, i1 %compare.low664, i1 %compare.high.order, !dbg !25
  %515 = select i1 %compare.result665, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base666 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 3) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local667 = or i32 %compare.local.base666, 3, !dbg !25
  %compare.low668 = icmp ugt i32 %compare.low.base, %compare.local667, !dbg !25
  %compare.result669 = select i1 %compare.high.equal, i1 %compare.low668, i1 %compare.high.order, !dbg !25
  %516 = select i1 %compare.result669, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base670 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 4) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local671 = or i32 %compare.local.base670, 4, !dbg !25
  %compare.low672 = icmp ugt i32 %compare.low.base, %compare.local671, !dbg !25
  %compare.result673 = select i1 %compare.high.equal, i1 %compare.low672, i1 %compare.high.order, !dbg !25
  %517 = select i1 %compare.result673, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base674 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 5) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local675 = or i32 %compare.local.base674, 5, !dbg !25
  %compare.low676 = icmp ugt i32 %compare.low.base, %compare.local675, !dbg !25
  %compare.result677 = select i1 %compare.high.equal, i1 %compare.low676, i1 %compare.high.order, !dbg !25
  %518 = select i1 %compare.result677, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base678 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 6) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local679 = or i32 %compare.local.base678, 6, !dbg !25
  %compare.low680 = icmp ugt i32 %compare.low.base, %compare.local679, !dbg !25
  %compare.result681 = select i1 %compare.high.equal, i1 %compare.low680, i1 %compare.high.order, !dbg !25
  %519 = select i1 %compare.result681, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base682 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 7) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local683 = or i32 %compare.local.base682, 7, !dbg !25
  %compare.low684 = icmp ugt i32 %compare.low.base, %compare.local683, !dbg !25
  %compare.result685 = select i1 %compare.high.equal, i1 %compare.low684, i1 %compare.high.order, !dbg !25
  %520 = select i1 %compare.result685, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base686 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 8) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local687 = or i32 %compare.local.base686, 8, !dbg !25
  %compare.low688 = icmp ugt i32 %compare.low.base, %compare.local687, !dbg !25
  %compare.result689 = select i1 %compare.high.equal, i1 %compare.low688, i1 %compare.high.order, !dbg !25
  %521 = select i1 %compare.result689, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base690 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 9) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local691 = or i32 %compare.local.base690, 9, !dbg !25
  %compare.low692 = icmp ugt i32 %compare.low.base, %compare.local691, !dbg !25
  %compare.result693 = select i1 %compare.high.equal, i1 %compare.low692, i1 %compare.high.order, !dbg !25
  %522 = select i1 %compare.result693, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base694 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 10) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local695 = or i32 %compare.local.base694, 10, !dbg !25
  %compare.low696 = icmp ugt i32 %compare.low.base, %compare.local695, !dbg !25
  %compare.result697 = select i1 %compare.high.equal, i1 %compare.low696, i1 %compare.high.order, !dbg !25
  %523 = select i1 %compare.result697, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base698 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 11) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local699 = or i32 %compare.local.base698, 11, !dbg !25
  %compare.low700 = icmp ugt i32 %compare.low.base, %compare.local699, !dbg !25
  %compare.result701 = select i1 %compare.high.equal, i1 %compare.low700, i1 %compare.high.order, !dbg !25
  %524 = select i1 %compare.result701, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base702 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 12) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local703 = or i32 %compare.local.base702, 12, !dbg !25
  %compare.low704 = icmp ugt i32 %compare.low.base, %compare.local703, !dbg !25
  %compare.result705 = select i1 %compare.high.equal, i1 %compare.low704, i1 %compare.high.order, !dbg !25
  %525 = select i1 %compare.result705, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base706 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 13) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local707 = or i32 %compare.local.base706, 13, !dbg !25
  %compare.low708 = icmp ugt i32 %compare.low.base, %compare.local707, !dbg !25
  %compare.result709 = select i1 %compare.high.equal, i1 %compare.low708, i1 %compare.high.order, !dbg !25
  %526 = select i1 %compare.result709, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base710 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 14) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local711 = or i32 %compare.local.base710, 14, !dbg !25
  %compare.low712 = icmp ugt i32 %compare.low.base, %compare.local711, !dbg !25
  %compare.result713 = select i1 %compare.high.equal, i1 %compare.low712, i1 %compare.high.order, !dbg !25
  %527 = select i1 %compare.result713, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base714 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 15) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local715 = or i32 %compare.local.base714, 15, !dbg !25
  %compare.low716 = icmp ugt i32 %compare.low.base, %compare.local715, !dbg !25
  %compare.result717 = select i1 %compare.high.equal, i1 %compare.low716, i1 %compare.high.order, !dbg !25
  %528 = select i1 %compare.result717, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base718 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 16) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local719 = or i32 %compare.local.base718, 64, !dbg !25
  %compare.low720 = icmp ugt i32 %compare.low.base, %compare.local719, !dbg !25
  %compare.result721 = select i1 %compare.high.equal, i1 %compare.low720, i1 %compare.high.order, !dbg !25
  %529 = select i1 %compare.result721, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base722 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 17) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local723 = or i32 %compare.local.base722, 65, !dbg !25
  %compare.low724 = icmp ugt i32 %compare.low.base, %compare.local723, !dbg !25
  %compare.result725 = select i1 %compare.high.equal, i1 %compare.low724, i1 %compare.high.order, !dbg !25
  %530 = select i1 %compare.result725, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base726 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 18) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local727 = or i32 %compare.local.base726, 66, !dbg !25
  %compare.low728 = icmp ugt i32 %compare.low.base, %compare.local727, !dbg !25
  %compare.result729 = select i1 %compare.high.equal, i1 %compare.low728, i1 %compare.high.order, !dbg !25
  %531 = select i1 %compare.result729, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base730 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 19) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local731 = or i32 %compare.local.base730, 67, !dbg !25
  %compare.low732 = icmp ugt i32 %compare.low.base, %compare.local731, !dbg !25
  %compare.result733 = select i1 %compare.high.equal, i1 %compare.low732, i1 %compare.high.order, !dbg !25
  %532 = select i1 %compare.result733, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base734 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 20) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local735 = or i32 %compare.local.base734, 68, !dbg !25
  %compare.low736 = icmp ugt i32 %compare.low.base, %compare.local735, !dbg !25
  %compare.result737 = select i1 %compare.high.equal, i1 %compare.low736, i1 %compare.high.order, !dbg !25
  %533 = select i1 %compare.result737, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base738 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 21) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local739 = or i32 %compare.local.base738, 69, !dbg !25
  %compare.low740 = icmp ugt i32 %compare.low.base, %compare.local739, !dbg !25
  %compare.result741 = select i1 %compare.high.equal, i1 %compare.low740, i1 %compare.high.order, !dbg !25
  %534 = select i1 %compare.result741, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base742 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 22) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local743 = or i32 %compare.local.base742, 70, !dbg !25
  %compare.low744 = icmp ugt i32 %compare.low.base, %compare.local743, !dbg !25
  %compare.result745 = select i1 %compare.high.equal, i1 %compare.low744, i1 %compare.high.order, !dbg !25
  %535 = select i1 %compare.result745, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base746 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 23) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local747 = or i32 %compare.local.base746, 71, !dbg !25
  %compare.low748 = icmp ugt i32 %compare.low.base, %compare.local747, !dbg !25
  %compare.result749 = select i1 %compare.high.equal, i1 %compare.low748, i1 %compare.high.order, !dbg !25
  %536 = select i1 %compare.result749, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base750 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 24) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local751 = or i32 %compare.local.base750, 72, !dbg !25
  %compare.low752 = icmp ugt i32 %compare.low.base, %compare.local751, !dbg !25
  %compare.result753 = select i1 %compare.high.equal, i1 %compare.low752, i1 %compare.high.order, !dbg !25
  %537 = select i1 %compare.result753, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base754 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 25) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local755 = or i32 %compare.local.base754, 73, !dbg !25
  %compare.low756 = icmp ugt i32 %compare.low.base, %compare.local755, !dbg !25
  %compare.result757 = select i1 %compare.high.equal, i1 %compare.low756, i1 %compare.high.order, !dbg !25
  %538 = select i1 %compare.result757, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base758 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 26) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local759 = or i32 %compare.local.base758, 74, !dbg !25
  %compare.low760 = icmp ugt i32 %compare.low.base, %compare.local759, !dbg !25
  %compare.result761 = select i1 %compare.high.equal, i1 %compare.low760, i1 %compare.high.order, !dbg !25
  %539 = select i1 %compare.result761, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base762 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 27) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local763 = or i32 %compare.local.base762, 75, !dbg !25
  %compare.low764 = icmp ugt i32 %compare.low.base, %compare.local763, !dbg !25
  %compare.result765 = select i1 %compare.high.equal, i1 %compare.low764, i1 %compare.high.order, !dbg !25
  %540 = select i1 %compare.result765, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base766 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 28) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local767 = or i32 %compare.local.base766, 76, !dbg !25
  %compare.low768 = icmp ugt i32 %compare.low.base, %compare.local767, !dbg !25
  %compare.result769 = select i1 %compare.high.equal, i1 %compare.low768, i1 %compare.high.order, !dbg !25
  %541 = select i1 %compare.result769, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base770 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 29) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local771 = or i32 %compare.local.base770, 77, !dbg !25
  %compare.low772 = icmp ugt i32 %compare.low.base, %compare.local771, !dbg !25
  %compare.result773 = select i1 %compare.high.equal, i1 %compare.low772, i1 %compare.high.order, !dbg !25
  %542 = select i1 %compare.result773, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base774 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 30) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local775 = or i32 %compare.local.base774, 78, !dbg !25
  %compare.low776 = icmp ugt i32 %compare.low.base, %compare.local775, !dbg !25
  %compare.result777 = select i1 %compare.high.equal, i1 %compare.low776, i1 %compare.high.order, !dbg !25
  %543 = select i1 %compare.result777, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base778 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base659, i32 31) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local779 = or i32 %compare.local.base778, 79, !dbg !25
  %compare.low780 = icmp ugt i32 %compare.low.base, %compare.local779, !dbg !25
  %compare.result781 = select i1 %compare.high.equal, i1 %compare.low780, i1 %compare.high.order, !dbg !25
  %544 = select i1 %compare.result781, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %545 = insertelement <32 x float> poison, float %513, i64 0, !dbg !14
  %546 = insertelement <32 x float> %545, float %514, i64 1, !dbg !14
  %547 = insertelement <32 x float> %546, float %515, i64 2, !dbg !14
  %548 = insertelement <32 x float> %547, float %516, i64 3, !dbg !14
  %549 = insertelement <32 x float> %548, float %517, i64 4, !dbg !14
  %550 = insertelement <32 x float> %549, float %518, i64 5, !dbg !14
  %551 = insertelement <32 x float> %550, float %519, i64 6, !dbg !14
  %552 = insertelement <32 x float> %551, float %520, i64 7, !dbg !14
  %553 = insertelement <32 x float> %552, float %521, i64 8, !dbg !14
  %554 = insertelement <32 x float> %553, float %522, i64 9, !dbg !14
  %555 = insertelement <32 x float> %554, float %523, i64 10, !dbg !14
  %556 = insertelement <32 x float> %555, float %524, i64 11, !dbg !14
  %557 = insertelement <32 x float> %556, float %525, i64 12, !dbg !14
  %558 = insertelement <32 x float> %557, float %526, i64 13, !dbg !14
  %559 = insertelement <32 x float> %558, float %527, i64 14, !dbg !14
  %560 = insertelement <32 x float> %559, float %528, i64 15, !dbg !14
  %561 = insertelement <32 x float> %560, float %529, i64 16, !dbg !14
  %562 = insertelement <32 x float> %561, float %530, i64 17, !dbg !14
  %563 = insertelement <32 x float> %562, float %531, i64 18, !dbg !14
  %564 = insertelement <32 x float> %563, float %532, i64 19, !dbg !14
  %565 = insertelement <32 x float> %564, float %533, i64 20, !dbg !14
  %566 = insertelement <32 x float> %565, float %534, i64 21, !dbg !14
  %567 = insertelement <32 x float> %566, float %535, i64 22, !dbg !14
  %568 = insertelement <32 x float> %567, float %536, i64 23, !dbg !14
  %569 = insertelement <32 x float> %568, float %537, i64 24, !dbg !14
  %570 = insertelement <32 x float> %569, float %538, i64 25, !dbg !14
  %571 = insertelement <32 x float> %570, float %539, i64 26, !dbg !14
  %572 = insertelement <32 x float> %571, float %540, i64 27, !dbg !14
  %573 = insertelement <32 x float> %572, float %541, i64 28, !dbg !14
  %574 = insertelement <32 x float> %573, float %542, i64 29, !dbg !14
  %575 = insertelement <32 x float> %574, float %543, i64 30, !dbg !14
  %576 = insertelement <32 x float> %575, float %544, i64 31, !dbg !14
  br label %577, !dbg !14

577:                                              ; preds = %501, %502
  %578 = phi <32 x float> [ %576, %502 ], [ zeroinitializer, %501 ], !dbg !14
  br label %579, !dbg !14

579:                                              ; preds = %577
  %580 = shufflevector <32 x float> %578, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %581 = fadd <4 x float> %470, %580, !dbg !27
  %582 = extractelement <4 x float> %581, i64 0, !dbg !27
  %583 = extractelement <4 x float> %581, i64 1, !dbg !27
  %584 = extractelement <4 x float> %581, i64 2, !dbg !27
  %585 = extractelement <4 x float> %581, i64 3, !dbg !27
  %586 = shufflevector <32 x float> %578, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %587 = fadd <4 x float> %473, %586, !dbg !27
  %588 = extractelement <4 x float> %587, i64 0, !dbg !27
  %589 = extractelement <4 x float> %587, i64 1, !dbg !27
  %590 = extractelement <4 x float> %587, i64 2, !dbg !27
  %591 = extractelement <4 x float> %587, i64 3, !dbg !27
  %592 = shufflevector <32 x float> %578, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %593 = fadd <4 x float> %478, %592, !dbg !27
  %594 = extractelement <4 x float> %593, i64 0, !dbg !27
  %595 = extractelement <4 x float> %593, i64 1, !dbg !27
  %596 = extractelement <4 x float> %593, i64 2, !dbg !27
  %597 = extractelement <4 x float> %593, i64 3, !dbg !27
  %598 = shufflevector <32 x float> %578, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %599 = fadd <4 x float> %481, %598, !dbg !27
  %600 = extractelement <4 x float> %599, i64 0, !dbg !27
  %601 = extractelement <4 x float> %599, i64 1, !dbg !27
  %602 = extractelement <4 x float> %599, i64 2, !dbg !27
  %603 = extractelement <4 x float> %599, i64 3, !dbg !27
  %604 = shufflevector <32 x float> %578, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %605 = fadd <4 x float> %486, %604, !dbg !27
  %606 = extractelement <4 x float> %605, i64 0, !dbg !27
  %607 = extractelement <4 x float> %605, i64 1, !dbg !27
  %608 = extractelement <4 x float> %605, i64 2, !dbg !27
  %609 = extractelement <4 x float> %605, i64 3, !dbg !27
  %610 = shufflevector <32 x float> %578, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %611 = fadd <4 x float> %489, %610, !dbg !27
  %612 = extractelement <4 x float> %611, i64 0, !dbg !27
  %613 = extractelement <4 x float> %611, i64 1, !dbg !27
  %614 = extractelement <4 x float> %611, i64 2, !dbg !27
  %615 = extractelement <4 x float> %611, i64 3, !dbg !27
  %616 = shufflevector <32 x float> %578, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %617 = fadd <4 x float> %494, %616, !dbg !27
  %618 = extractelement <4 x float> %617, i64 0, !dbg !27
  %619 = extractelement <4 x float> %617, i64 1, !dbg !27
  %620 = extractelement <4 x float> %617, i64 2, !dbg !27
  %621 = extractelement <4 x float> %617, i64 3, !dbg !27
  %622 = shufflevector <32 x float> %578, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %623 = fadd <4 x float> %497, %622, !dbg !27
  %624 = extractelement <4 x float> %623, i64 0, !dbg !27
  %625 = extractelement <4 x float> %623, i64 1, !dbg !27
  %626 = extractelement <4 x float> %623, i64 2, !dbg !27
  %627 = extractelement <4 x float> %623, i64 3, !dbg !27
  %628 = call float @llvm.maxnum.f32(float %582, float 0xFFF0000000000000), !dbg !28
  %629 = call float @llvm.maxnum.f32(float %628, float %583), !dbg !28
  %630 = call float @llvm.maxnum.f32(float %629, float %584), !dbg !28
  %631 = call float @llvm.maxnum.f32(float %630, float %585), !dbg !28
  %632 = call float @llvm.maxnum.f32(float %631, float %588), !dbg !28
  %633 = call float @llvm.maxnum.f32(float %632, float %589), !dbg !28
  %634 = call float @llvm.maxnum.f32(float %633, float %590), !dbg !28
  %635 = call float @llvm.maxnum.f32(float %634, float %591), !dbg !28
  %636 = call float @llvm.maxnum.f32(float %635, float %594), !dbg !28
  %637 = call float @llvm.maxnum.f32(float %636, float %595), !dbg !28
  %638 = call float @llvm.maxnum.f32(float %637, float %596), !dbg !28
  %639 = call float @llvm.maxnum.f32(float %638, float %597), !dbg !28
  %640 = call float @llvm.maxnum.f32(float %639, float %600), !dbg !28
  %641 = call float @llvm.maxnum.f32(float %640, float %601), !dbg !28
  %642 = call float @llvm.maxnum.f32(float %641, float %602), !dbg !28
  %643 = call float @llvm.maxnum.f32(float %642, float %603), !dbg !28
  %644 = call float @llvm.maxnum.f32(float %643, float %606), !dbg !28
  %645 = call float @llvm.maxnum.f32(float %644, float %607), !dbg !28
  %646 = call float @llvm.maxnum.f32(float %645, float %608), !dbg !28
  %647 = call float @llvm.maxnum.f32(float %646, float %609), !dbg !28
  %648 = call float @llvm.maxnum.f32(float %647, float %612), !dbg !28
  %649 = call float @llvm.maxnum.f32(float %648, float %613), !dbg !28
  %650 = call float @llvm.maxnum.f32(float %649, float %614), !dbg !28
  %651 = call float @llvm.maxnum.f32(float %650, float %615), !dbg !28
  %652 = call float @llvm.maxnum.f32(float %651, float %618), !dbg !28
  %653 = call float @llvm.maxnum.f32(float %652, float %619), !dbg !28
  %654 = call float @llvm.maxnum.f32(float %653, float %620), !dbg !28
  %655 = call float @llvm.maxnum.f32(float %654, float %621), !dbg !28
  %656 = call float @llvm.maxnum.f32(float %655, float %624), !dbg !28
  %657 = call float @llvm.maxnum.f32(float %656, float %625), !dbg !28
  %658 = call float @llvm.maxnum.f32(float %657, float %626), !dbg !28
  %659 = call float @llvm.maxnum.f32(float %658, float %627), !dbg !28
  %660 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %661 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %660), !dbg !28
  %662 = add nuw nsw i32 %661, 64, !dbg !28
  %663 = xor i32 %661, 16, !dbg !28
  %664 = and i32 %662, 192, !dbg !28
  %665 = icmp samesign ult i32 %663, %664, !dbg !28
  %666 = select i1 %665, i32 %663, i32 %661, !dbg !28
  %667 = shl nuw nsw i32 %666, 2, !dbg !28
  %668 = bitcast float %659 to i32, !dbg !28
  %669 = call i32 @llvm.amdgcn.ds.bpermute(i32 %667, i32 %668), !dbg !28
  %670 = bitcast i32 %669 to float, !dbg !28
  %671 = call float @llvm.maxnum.f32(float %659, float %670), !dbg !28
  %672 = xor i32 %661, 32, !dbg !28
  %673 = icmp samesign ult i32 %672, %664, !dbg !28
  %674 = select i1 %673, i32 %672, i32 %661, !dbg !28
  %675 = shl nuw nsw i32 %674, 2, !dbg !28
  %676 = bitcast float %671 to i32, !dbg !28
  %677 = call i32 @llvm.amdgcn.ds.bpermute(i32 %675, i32 %676), !dbg !28
  %678 = bitcast i32 %677 to float, !dbg !28
  %679 = call float @llvm.maxnum.f32(float %671, float %678), !dbg !28
  %680 = insertelement <1 x float> poison, float %679, i64 0, !dbg !28
  %681 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %680, <1 x float> %59), !dbg !28
  %682 = shufflevector <1 x float> %681, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !29
  %683 = fsub <4 x float> %581, %682, !dbg !29
  %684 = fsub <4 x float> %587, %682, !dbg !29
  %685 = fsub <4 x float> %593, %682, !dbg !29
  %686 = fsub <4 x float> %599, %682, !dbg !29
  %687 = fsub <4 x float> %605, %682, !dbg !29
  %688 = fsub <4 x float> %611, %682, !dbg !29
  %689 = fsub <4 x float> %617, %682, !dbg !29
  %690 = fsub <4 x float> %623, %682, !dbg !29
  %691 = fsub <1 x float> %59, %681, !dbg !30
  %692 = extractelement <1 x float> %691, i64 0, !dbg !31
  %693 = call float @llvm.amdgcn.exp2.f32(float %692), !dbg !31
  %694 = insertelement <4 x float> poison, float %693, i64 0, !dbg !36
  %695 = shufflevector <4 x float> %694, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %696 = fmul <4 x float> %67, %695, !dbg !36
  %697 = fmul <4 x float> %66, %695, !dbg !36
  %698 = fmul <4 x float> %64, %695, !dbg !36
  %699 = fmul <4 x float> %63, %695, !dbg !36
  %700 = fmul <4 x float> %61, %695, !dbg !36
  %701 = fmul <4 x float> %60, %695, !dbg !36
  %702 = fmul <4 x float> %56, %695, !dbg !36
  %703 = fmul <4 x float> %55, %695, !dbg !36
  %704 = extractelement <1 x float> %58, i64 0, !dbg !32
  %705 = fmul float %704, %693, !dbg !32
  %706 = extractelement <4 x float> %683, i64 0, !dbg !33
  %707 = call float @llvm.amdgcn.exp2.f32(float %706), !dbg !33
  %708 = insertelement <4 x float> poison, float %707, i64 0, !dbg !35
  %709 = extractelement <4 x float> %683, i64 1, !dbg !33
  %710 = call float @llvm.amdgcn.exp2.f32(float %709), !dbg !33
  %711 = insertelement <4 x float> %708, float %710, i64 1, !dbg !35
  %712 = fadd float %707, %710, !dbg !34
  %713 = extractelement <4 x float> %683, i64 2, !dbg !33
  %714 = call float @llvm.amdgcn.exp2.f32(float %713), !dbg !33
  %715 = insertelement <4 x float> %711, float %714, i64 2, !dbg !35
  %716 = fadd float %712, %714, !dbg !34
  %717 = extractelement <4 x float> %683, i64 3, !dbg !33
  %718 = call float @llvm.amdgcn.exp2.f32(float %717), !dbg !33
  %719 = insertelement <4 x float> %715, float %718, i64 3, !dbg !35
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %719) #9, !dbg !35, !frisk.register_lifetimes !1
  %720 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !35
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %720) #9, !dbg !34, !frisk.register_lifetimes !1
  %721 = fadd float %716, %718, !dbg !34
  %722 = extractelement <4 x float> %684, i64 0, !dbg !33
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !33
  %724 = insertelement <4 x float> poison, float %723, i64 0, !dbg !35
  %725 = fadd float %721, %723, !dbg !34
  %726 = extractelement <4 x float> %684, i64 1, !dbg !33
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !33
  %728 = insertelement <4 x float> %724, float %727, i64 1, !dbg !35
  %729 = fadd float %725, %727, !dbg !34
  %730 = extractelement <4 x float> %684, i64 2, !dbg !33
  %731 = call float @llvm.amdgcn.exp2.f32(float %730), !dbg !33
  %732 = insertelement <4 x float> %728, float %731, i64 2, !dbg !35
  %733 = fadd float %729, %731, !dbg !34
  %734 = extractelement <4 x float> %684, i64 3, !dbg !33
  %735 = call float @llvm.amdgcn.exp2.f32(float %734), !dbg !33
  %736 = insertelement <4 x float> %732, float %735, i64 3, !dbg !35
  %math.result96 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %736) #9, !dbg !35, !frisk.register_lifetimes !1
  %737 = fptrunc <4 x float> %math.result96 to <4 x half>, !dbg !35
  %math.packed97 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %737) #9, !dbg !34, !frisk.register_lifetimes !1
  %738 = fadd float %733, %735, !dbg !34
  %739 = extractelement <4 x float> %685, i64 0, !dbg !33
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !33
  %741 = insertelement <4 x float> poison, float %740, i64 0, !dbg !35
  %742 = fadd float %738, %740, !dbg !34
  %743 = extractelement <4 x float> %685, i64 1, !dbg !33
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !33
  %745 = insertelement <4 x float> %741, float %744, i64 1, !dbg !35
  %746 = fadd float %742, %744, !dbg !34
  %747 = extractelement <4 x float> %685, i64 2, !dbg !33
  %748 = call float @llvm.amdgcn.exp2.f32(float %747), !dbg !33
  %749 = insertelement <4 x float> %745, float %748, i64 2, !dbg !35
  %750 = fadd float %746, %748, !dbg !34
  %751 = extractelement <4 x float> %685, i64 3, !dbg !33
  %752 = call float @llvm.amdgcn.exp2.f32(float %751), !dbg !33
  %753 = insertelement <4 x float> %749, float %752, i64 3, !dbg !35
  %math.result98 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %753) #9, !dbg !35, !frisk.register_lifetimes !1
  %754 = fptrunc <4 x float> %math.result98 to <4 x half>, !dbg !35
  %math.packed99 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %754) #9, !dbg !34, !frisk.register_lifetimes !1
  %755 = fadd float %750, %752, !dbg !34
  %756 = extractelement <4 x float> %686, i64 0, !dbg !33
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !33
  %758 = insertelement <4 x float> poison, float %757, i64 0, !dbg !35
  %759 = fadd float %755, %757, !dbg !34
  %760 = extractelement <4 x float> %686, i64 1, !dbg !33
  %761 = call float @llvm.amdgcn.exp2.f32(float %760), !dbg !33
  %762 = insertelement <4 x float> %758, float %761, i64 1, !dbg !35
  %763 = fadd float %759, %761, !dbg !34
  %764 = extractelement <4 x float> %686, i64 2, !dbg !33
  %765 = call float @llvm.amdgcn.exp2.f32(float %764), !dbg !33
  %766 = insertelement <4 x float> %762, float %765, i64 2, !dbg !35
  %767 = fadd float %763, %765, !dbg !34
  %768 = extractelement <4 x float> %686, i64 3, !dbg !33
  %769 = call float @llvm.amdgcn.exp2.f32(float %768), !dbg !33
  %770 = insertelement <4 x float> %766, float %769, i64 3, !dbg !35
  %math.result100 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %770) #9, !dbg !35, !frisk.register_lifetimes !1
  %771 = fptrunc <4 x float> %math.result100 to <4 x half>, !dbg !35
  %math.packed101 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %771) #9, !dbg !34, !frisk.register_lifetimes !1
  %772 = fadd float %767, %769, !dbg !34
  %773 = extractelement <4 x float> %687, i64 0, !dbg !33
  %774 = call float @llvm.amdgcn.exp2.f32(float %773), !dbg !33
  %775 = insertelement <4 x float> poison, float %774, i64 0, !dbg !35
  %776 = fadd float %772, %774, !dbg !34
  %777 = extractelement <4 x float> %687, i64 1, !dbg !33
  %778 = call float @llvm.amdgcn.exp2.f32(float %777), !dbg !33
  %779 = insertelement <4 x float> %775, float %778, i64 1, !dbg !35
  %780 = fadd float %776, %778, !dbg !34
  %781 = extractelement <4 x float> %687, i64 2, !dbg !33
  %782 = call float @llvm.amdgcn.exp2.f32(float %781), !dbg !33
  %783 = insertelement <4 x float> %779, float %782, i64 2, !dbg !35
  %784 = fadd float %780, %782, !dbg !34
  %785 = extractelement <4 x float> %687, i64 3, !dbg !33
  %786 = call float @llvm.amdgcn.exp2.f32(float %785), !dbg !33
  %787 = insertelement <4 x float> %783, float %786, i64 3, !dbg !35
  %math.result102 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %787) #9, !dbg !35, !frisk.register_lifetimes !1
  %788 = fptrunc <4 x float> %math.result102 to <4 x half>, !dbg !35
  %math.packed103 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %788) #9, !dbg !34, !frisk.register_lifetimes !1
  %789 = fadd float %784, %786, !dbg !34
  %790 = extractelement <4 x float> %688, i64 0, !dbg !33
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !33
  %792 = insertelement <4 x float> poison, float %791, i64 0, !dbg !35
  %793 = fadd float %789, %791, !dbg !34
  %794 = extractelement <4 x float> %688, i64 1, !dbg !33
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !33
  %796 = insertelement <4 x float> %792, float %795, i64 1, !dbg !35
  %797 = fadd float %793, %795, !dbg !34
  %798 = extractelement <4 x float> %688, i64 2, !dbg !33
  %799 = call float @llvm.amdgcn.exp2.f32(float %798), !dbg !33
  %800 = insertelement <4 x float> %796, float %799, i64 2, !dbg !35
  %801 = fadd float %797, %799, !dbg !34
  %802 = extractelement <4 x float> %688, i64 3, !dbg !33
  %803 = call float @llvm.amdgcn.exp2.f32(float %802), !dbg !33
  %804 = insertelement <4 x float> %800, float %803, i64 3, !dbg !35
  %math.result104 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %804) #9, !dbg !35, !frisk.register_lifetimes !1
  %805 = fptrunc <4 x float> %math.result104 to <4 x half>, !dbg !35
  %math.packed105 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %805) #9, !dbg !34, !frisk.register_lifetimes !1
  %806 = fadd float %801, %803, !dbg !34
  %807 = extractelement <4 x float> %689, i64 0, !dbg !33
  %808 = call float @llvm.amdgcn.exp2.f32(float %807), !dbg !33
  %809 = insertelement <4 x float> poison, float %808, i64 0, !dbg !35
  %810 = fadd float %806, %808, !dbg !34
  %811 = extractelement <4 x float> %689, i64 1, !dbg !33
  %812 = call float @llvm.amdgcn.exp2.f32(float %811), !dbg !33
  %813 = insertelement <4 x float> %809, float %812, i64 1, !dbg !35
  %814 = fadd float %810, %812, !dbg !34
  %815 = extractelement <4 x float> %689, i64 2, !dbg !33
  %816 = call float @llvm.amdgcn.exp2.f32(float %815), !dbg !33
  %817 = insertelement <4 x float> %813, float %816, i64 2, !dbg !35
  %818 = fadd float %814, %816, !dbg !34
  %819 = extractelement <4 x float> %689, i64 3, !dbg !33
  %820 = call float @llvm.amdgcn.exp2.f32(float %819), !dbg !33
  %821 = insertelement <4 x float> %817, float %820, i64 3, !dbg !35
  %math.result106 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %821) #9, !dbg !35, !frisk.register_lifetimes !1
  %822 = fptrunc <4 x float> %math.result106 to <4 x half>, !dbg !35
  %math.packed107 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %822) #9, !dbg !34, !frisk.register_lifetimes !1
  %823 = fadd float %818, %820, !dbg !34
  %824 = extractelement <4 x float> %690, i64 0, !dbg !33
  %825 = call float @llvm.amdgcn.exp2.f32(float %824), !dbg !33
  %826 = insertelement <4 x float> poison, float %825, i64 0, !dbg !35
  %827 = fadd float %823, %825, !dbg !34
  %828 = extractelement <4 x float> %690, i64 1, !dbg !33
  %829 = call float @llvm.amdgcn.exp2.f32(float %828), !dbg !33
  %830 = insertelement <4 x float> %826, float %829, i64 1, !dbg !35
  %831 = fadd float %827, %829, !dbg !34
  %832 = extractelement <4 x float> %690, i64 2, !dbg !33
  %833 = call float @llvm.amdgcn.exp2.f32(float %832), !dbg !33
  %834 = insertelement <4 x float> %830, float %833, i64 2, !dbg !35
  %835 = fadd float %831, %833, !dbg !34
  %836 = extractelement <4 x float> %690, i64 3, !dbg !33
  %837 = call float @llvm.amdgcn.exp2.f32(float %836), !dbg !33
  %838 = insertelement <4 x float> %834, float %837, i64 3, !dbg !35
  %math.result108 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %838) #9, !dbg !35, !frisk.register_lifetimes !1
  %839 = fptrunc <4 x float> %math.result108 to <4 x half>, !dbg !35
  %math.packed109 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %839) #9, !dbg !34, !frisk.register_lifetimes !1
  %840 = fadd float %835, %837, !dbg !34
  %841 = bitcast float %840 to i32, !dbg !34
  %842 = call i32 @llvm.amdgcn.ds.bpermute(i32 %667, i32 %841), !dbg !34
  %843 = bitcast i32 %842 to float, !dbg !34
  %844 = fadd float %840, %843, !dbg !34
  %845 = bitcast float %844 to i32, !dbg !34
  %846 = call i32 @llvm.amdgcn.ds.bpermute(i32 %675, i32 %845), !dbg !34
  %847 = bitcast i32 %846 to float, !dbg !34
  %848 = fadd float %844, %847, !dbg !34
  %849 = fadd float %848, %705, !dbg !34
  %850 = insertelement <1 x float> poison, float %849, i64 0, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread441 = call i32 asm sideeffect "", "=v,0"(i32 %28) #9, !dbg !15, !range !40, !frisk.register_lifetimes !1
  %address.local442 = shl nuw nsw i32 %address.thread441, 3, !dbg !15
  %address.local443 = and i32 %address.local442, 120, !dbg !15
  %address.local444 = shl nuw nsw i32 %address.thread441, 5, !dbg !15
  %address.local445 = and i32 %address.local444, 1536, !dbg !15
  %address.local446 = or disjoint i32 %address.local443, %address.local445, !dbg !15
  %address.local447 = zext nneg i32 %address.local446 to i64, !dbg !15
  %address.local448 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local447, !dbg !15
  %851 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local448, i32 0) #0, !dbg !15
  %852 = bitcast <4 x i32> %851 to <8 x half>, !dbg !15
  %address.local449 = zext nneg i32 %address.thread441 to i64, !dbg !13
  %address.local450 = and i64 %address.local449, 60, !dbg !15
  %address.local451 = add nuw nsw i64 %address.local450, %address.local449, !dbg !15
  %address.local452 = shl nuw nsw i64 %address.local451, 3, !dbg !15
  %address.local453 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local452, !dbg !15
  %address.local454 = lshr i64 %address.local449, 4, !dbg !15
  %address.local455 = and i64 %address.local454, 3, !dbg !15
  %address.local456 = mul nuw nsw i64 %address.local455, 384, !dbg !15
  %address.local457 = getelementptr [2 x i8], ptr addrspace(3) %address.local453, i64 %address.local456, !dbg !15
  %address.local458 = lshr i64 %address.local449, 2, !dbg !17
  %address.local459 = mul nsw i64 %address.local458, -32, !dbg !15
  %address.local460 = getelementptr [2 x i8], ptr addrspace(3) %address.local457, i64 %address.local459, !dbg !15
  %address.local461 = getelementptr i8, ptr addrspace(3) %address.local460, i64 256, !dbg !15
  %853 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local461, i32 0) #0, !dbg !15
  %854 = bitcast <4 x i32> %853 to <8 x half>, !dbg !15
  %address.local462 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local452, !dbg !15
  %address.local463 = getelementptr [2 x i8], ptr addrspace(3) %address.local462, i64 %address.local456, !dbg !15
  %address.local464 = getelementptr [2 x i8], ptr addrspace(3) %address.local463, i64 %address.local459, !dbg !15
  %address.local465 = getelementptr i8, ptr addrspace(3) %address.local464, i64 512, !dbg !15
  %855 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local465, i32 0) #0, !dbg !15
  %856 = bitcast <4 x i32> %855 to <8 x half>, !dbg !15
  %address.local466 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local452, !dbg !15
  %address.local467 = getelementptr [2 x i8], ptr addrspace(3) %address.local466, i64 %address.local456, !dbg !15
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) %address.local467, i64 %address.local459, !dbg !15
  %address.local469 = getelementptr i8, ptr addrspace(3) %address.local468, i64 768, !dbg !15
  %857 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local469, i32 0) #0, !dbg !15
  %858 = bitcast <4 x i32> %857 to <8 x half>, !dbg !15
  %address.local470 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local447, !dbg !15
  %address.local471 = getelementptr i8, ptr addrspace(3) %address.local470, i64 4096, !dbg !15
  %859 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local471, i32 0) #0, !dbg !15
  %860 = bitcast <4 x i32> %859 to <8 x half>, !dbg !15
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local452, !dbg !15
  %address.local473 = getelementptr [2 x i8], ptr addrspace(3) %address.local472, i64 %address.local456, !dbg !15
  %address.local474 = getelementptr [2 x i8], ptr addrspace(3) %address.local473, i64 %address.local459, !dbg !15
  %address.local475 = getelementptr i8, ptr addrspace(3) %address.local474, i64 4352, !dbg !15
  %861 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local475, i32 0) #0, !dbg !15
  %862 = bitcast <4 x i32> %861 to <8 x half>, !dbg !15
  %address.local476 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local452, !dbg !15
  %address.local477 = getelementptr [2 x i8], ptr addrspace(3) %address.local476, i64 %address.local456, !dbg !15
  %address.local478 = getelementptr [2 x i8], ptr addrspace(3) %address.local477, i64 %address.local459, !dbg !15
  %address.local479 = getelementptr i8, ptr addrspace(3) %address.local478, i64 4608, !dbg !15
  %863 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local479, i32 0) #0, !dbg !15
  %864 = bitcast <4 x i32> %863 to <8 x half>, !dbg !15
  %address.local480 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local452, !dbg !15
  %address.local481 = getelementptr [2 x i8], ptr addrspace(3) %address.local480, i64 %address.local456, !dbg !15
  %address.local482 = getelementptr [2 x i8], ptr addrspace(3) %address.local481, i64 %address.local459, !dbg !15
  %address.local483 = getelementptr i8, ptr addrspace(3) %address.local482, i64 4864, !dbg !15
  %865 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local483, i32 0) #0, !dbg !15
  %866 = bitcast <4 x i32> %865 to <8 x half>, !dbg !15
  %867 = add nsw i64 %88, %8, !dbg !18
  %868 = shl i64 %867, 19, !dbg !18
  %869 = and i64 %868, 16252928, !dbg !18
  %870 = add i64 %869, %94, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %871 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %870, !dbg !18
  %872 = getelementptr i8, ptr addrspace(1) %871, i64 64, !dbg !18
  %address.local484 = shl nuw nsw i64 %address.local449, 3, !dbg !12
  %address.local485 = and i64 %address.local484, 1536, !dbg !16
  %address.local486 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local485, !dbg !16
  %address.local487 = ptrtoint ptr addrspace(3) %address.local486 to i64, !dbg !16
  %address.local488 = trunc i64 %address.local487 to i32, !dbg !16
  %873 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local488), !dbg !18
  %874 = zext i32 %873 to i64, !dbg !18
  %875 = inttoptr i64 %874 to ptr addrspace(3), !dbg !18
  %address.local489 = shl nuw nsw i64 %address.local449, 5, !dbg !18
  %address.local490 = and i64 %address.local489, 8064, !dbg !18
  %address.local491 = shl nuw nsw i64 %address.local449, 3, !dbg !18
  %address.local492 = and i64 %address.local491, 24, !dbg !18
  %address.local493 = or disjoint i64 %address.local490, %address.local492, !dbg !18
  %address.local494 = getelementptr [2 x i8], ptr addrspace(1) %872, i64 %address.local493, !dbg !18
  %876 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %877 = ptrtoint ptr addrspace(1) %address.local494 to i64, !dbg !18
  %878 = sub i64 %877, %876, !dbg !18
  %dma.buffer.offset794 = trunc i64 %878 to i32, !dbg !18
  %879 = trunc i64 %876 to i32, !dbg !18
  %880 = insertelement <4 x i32> poison, i32 %879, i64 0, !dbg !18
  %881 = lshr i64 %876, 32, !dbg !18
  %882 = trunc i64 %881 to i32, !dbg !18
  %883 = insertelement <4 x i32> %880, i32 %882, i64 1, !dbg !18
  %884 = insertelement <4 x i32> %883, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc795 = insertelement <4 x i32> %884, i32 131072, i64 3, !dbg !18
  %885 = ptrtoint ptr addrspace(3) %875 to i32, !dbg !18
  %dma.buffer.lds796 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %885), !dbg !18
  %886 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds796, i32 %dma.buffer.offset794, <4 x i32> %dma.buffer.rsrc795) #10, !dbg !18
  %address.local495 = or disjoint i64 %address.local449, 256, !dbg !16
  %address.local496 = shl nuw nsw i64 %address.local495, 3, !dbg !16
  %address.local497 = and i64 %address.local496, 3584, !dbg !16
  %address.local498 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local497, !dbg !16
  %address.local499 = ptrtoint ptr addrspace(3) %address.local498 to i64, !dbg !16
  %address.local500 = trunc i64 %address.local499 to i32, !dbg !16
  %887 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local500), !dbg !18
  %888 = zext i32 %887 to i64, !dbg !18
  %889 = inttoptr i64 %888 to ptr addrspace(3), !dbg !18
  %address.local501 = shl nuw nsw i64 %address.local495, 5, !dbg !18
  %address.local502 = and i64 %address.local501, 16256, !dbg !18
  %address.local503 = shl nuw nsw i64 %address.local449, 3, !dbg !18
  %address.local504 = and i64 %address.local503, 24, !dbg !18
  %address.local505 = or disjoint i64 %address.local502, %address.local504, !dbg !18
  %address.local506 = getelementptr [2 x i8], ptr addrspace(1) %872, i64 %address.local505, !dbg !18
  %890 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %891 = ptrtoint ptr addrspace(1) %address.local506 to i64, !dbg !18
  %892 = sub i64 %891, %890, !dbg !18
  %dma.buffer.offset791 = trunc i64 %892 to i32, !dbg !18
  %893 = trunc i64 %890 to i32, !dbg !18
  %894 = insertelement <4 x i32> poison, i32 %893, i64 0, !dbg !18
  %895 = lshr i64 %890, 32, !dbg !18
  %896 = trunc i64 %895 to i32, !dbg !18
  %897 = insertelement <4 x i32> %894, i32 %896, i64 1, !dbg !18
  %898 = insertelement <4 x i32> %897, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc792 = insertelement <4 x i32> %898, i32 131072, i64 3, !dbg !18
  %899 = ptrtoint ptr addrspace(3) %889 to i32, !dbg !18
  %dma.buffer.lds793 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %899), !dbg !18
  %900 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds793, i32 %dma.buffer.offset791, <4 x i32> %dma.buffer.rsrc792) #10, !dbg !18
  %901 = shufflevector <8 x half> %852, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %901, <4 x float> %703) #9, !dbg !15
  %903 = shufflevector <8 x half> %854, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %903, <4 x float> %902) #9, !dbg !15
  %905 = shufflevector <8 x half> %856, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %905, <4 x float> %904) #9, !dbg !15
  %907 = shufflevector <8 x half> %858, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %907, <4 x float> %906) #9, !dbg !15
  %909 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %909, <4 x float> %908) #9, !dbg !15
  %911 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %911, <4 x float> %910) #9, !dbg !15
  %913 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %913, <4 x float> %912) #9, !dbg !15
  %915 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %915, <4 x float> %914) #9, !dbg !15
  %917 = shufflevector <8 x half> %852, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %917, <4 x float> %702) #9, !dbg !15
  %919 = shufflevector <8 x half> %854, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %919, <4 x float> %918) #9, !dbg !15
  %921 = shufflevector <8 x half> %856, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %921, <4 x float> %920) #9, !dbg !15
  %923 = shufflevector <8 x half> %858, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %923, <4 x float> %922) #9, !dbg !15
  %925 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %925, <4 x float> %924) #9, !dbg !15
  %927 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %927, <4 x float> %926) #9, !dbg !15
  %929 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %929, <4 x float> %928) #9, !dbg !15
  %931 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %931, <4 x float> %930) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %933 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local448, i32 0) #0, !dbg !15
  %934 = bitcast <4 x i32> %933 to <8 x half>, !dbg !15
  %935 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local461, i32 0) #0, !dbg !15
  %936 = bitcast <4 x i32> %935 to <8 x half>, !dbg !15
  %937 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local465, i32 0) #0, !dbg !15
  %938 = bitcast <4 x i32> %937 to <8 x half>, !dbg !15
  %939 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local469, i32 0) #0, !dbg !15
  %940 = bitcast <4 x i32> %939 to <8 x half>, !dbg !15
  %941 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local471, i32 0) #0, !dbg !15
  %942 = bitcast <4 x i32> %941 to <8 x half>, !dbg !15
  %943 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local475, i32 0) #0, !dbg !15
  %944 = bitcast <4 x i32> %943 to <8 x half>, !dbg !15
  %945 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local479, i32 0) #0, !dbg !15
  %946 = bitcast <4 x i32> %945 to <8 x half>, !dbg !15
  %947 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local483, i32 0) #0, !dbg !15
  %948 = bitcast <4 x i32> %947 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %949 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %870, !dbg !18
  %950 = getelementptr i8, ptr addrspace(1) %949, i64 128, !dbg !18
  %951 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local488), !dbg !18
  %952 = zext i32 %951 to i64, !dbg !18
  %953 = inttoptr i64 %952 to ptr addrspace(3), !dbg !18
  %address.local507 = getelementptr [2 x i8], ptr addrspace(1) %950, i64 %address.local493, !dbg !18
  %954 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %955 = ptrtoint ptr addrspace(1) %address.local507 to i64, !dbg !18
  %956 = sub i64 %955, %954, !dbg !18
  %dma.buffer.offset788 = trunc i64 %956 to i32, !dbg !18
  %957 = trunc i64 %954 to i32, !dbg !18
  %958 = insertelement <4 x i32> poison, i32 %957, i64 0, !dbg !18
  %959 = lshr i64 %954, 32, !dbg !18
  %960 = trunc i64 %959 to i32, !dbg !18
  %961 = insertelement <4 x i32> %958, i32 %960, i64 1, !dbg !18
  %962 = insertelement <4 x i32> %961, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc789 = insertelement <4 x i32> %962, i32 131072, i64 3, !dbg !18
  %963 = ptrtoint ptr addrspace(3) %953 to i32, !dbg !18
  %dma.buffer.lds790 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %963), !dbg !18
  %964 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds790, i32 %dma.buffer.offset788, <4 x i32> %dma.buffer.rsrc789) #10, !dbg !18
  %965 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local500), !dbg !18
  %966 = zext i32 %965 to i64, !dbg !18
  %967 = inttoptr i64 %966 to ptr addrspace(3), !dbg !18
  %address.local508 = getelementptr [2 x i8], ptr addrspace(1) %950, i64 %address.local505, !dbg !18
  %968 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %969 = ptrtoint ptr addrspace(1) %address.local508 to i64, !dbg !18
  %970 = sub i64 %969, %968, !dbg !18
  %dma.buffer.offset785 = trunc i64 %970 to i32, !dbg !18
  %971 = trunc i64 %968 to i32, !dbg !18
  %972 = insertelement <4 x i32> poison, i32 %971, i64 0, !dbg !18
  %973 = lshr i64 %968, 32, !dbg !18
  %974 = trunc i64 %973 to i32, !dbg !18
  %975 = insertelement <4 x i32> %972, i32 %974, i64 1, !dbg !18
  %976 = insertelement <4 x i32> %975, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc786 = insertelement <4 x i32> %976, i32 131072, i64 3, !dbg !18
  %977 = ptrtoint ptr addrspace(3) %967 to i32, !dbg !18
  %dma.buffer.lds787 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %977), !dbg !18
  %978 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds787, i32 %dma.buffer.offset785, <4 x i32> %dma.buffer.rsrc786) #10, !dbg !18
  %979 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %979, <4 x float> %701) #9, !dbg !15
  %981 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %981, <4 x float> %980) #9, !dbg !15
  %983 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %983, <4 x float> %982) #9, !dbg !15
  %985 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %985, <4 x float> %984) #9, !dbg !15
  %987 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %987, <4 x float> %986) #9, !dbg !15
  %989 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %989, <4 x float> %988) #9, !dbg !15
  %991 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %991, <4 x float> %990) #9, !dbg !15
  %993 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %993, <4 x float> %992) #9, !dbg !15
  %995 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %995, <4 x float> %700) #9, !dbg !15
  %997 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %997, <4 x float> %996) #9, !dbg !15
  %999 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1000 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %999, <4 x float> %998) #9, !dbg !15
  %1001 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1001, <4 x float> %1000) #9, !dbg !15
  %1003 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1003, <4 x float> %1002) #9, !dbg !15
  %1005 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1006 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1005, <4 x float> %1004) #9, !dbg !15
  %1007 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1007, <4 x float> %1006) #9, !dbg !15
  %1009 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1009, <4 x float> %1008) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1011 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local448, i32 0) #0, !dbg !15
  %1012 = bitcast <4 x i32> %1011 to <8 x half>, !dbg !15
  %1013 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local461, i32 0) #0, !dbg !15
  %1014 = bitcast <4 x i32> %1013 to <8 x half>, !dbg !15
  %1015 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local465, i32 0) #0, !dbg !15
  %1016 = bitcast <4 x i32> %1015 to <8 x half>, !dbg !15
  %1017 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local469, i32 0) #0, !dbg !15
  %1018 = bitcast <4 x i32> %1017 to <8 x half>, !dbg !15
  %1019 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local471, i32 0) #0, !dbg !15
  %1020 = bitcast <4 x i32> %1019 to <8 x half>, !dbg !15
  %1021 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local475, i32 0) #0, !dbg !15
  %1022 = bitcast <4 x i32> %1021 to <8 x half>, !dbg !15
  %1023 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local479, i32 0) #0, !dbg !15
  %1024 = bitcast <4 x i32> %1023 to <8 x half>, !dbg !15
  %1025 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local483, i32 0) #0, !dbg !15
  %1026 = bitcast <4 x i32> %1025 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1027 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %870, !dbg !18
  %1028 = getelementptr i8, ptr addrspace(1) %1027, i64 192, !dbg !18
  %1029 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local488), !dbg !18
  %1030 = zext i32 %1029 to i64, !dbg !18
  %1031 = inttoptr i64 %1030 to ptr addrspace(3), !dbg !18
  %address.local509 = getelementptr [2 x i8], ptr addrspace(1) %1028, i64 %address.local493, !dbg !18
  %1032 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1033 = ptrtoint ptr addrspace(1) %address.local509 to i64, !dbg !18
  %1034 = sub i64 %1033, %1032, !dbg !18
  %dma.buffer.offset782 = trunc i64 %1034 to i32, !dbg !18
  %1035 = trunc i64 %1032 to i32, !dbg !18
  %1036 = insertelement <4 x i32> poison, i32 %1035, i64 0, !dbg !18
  %1037 = lshr i64 %1032, 32, !dbg !18
  %1038 = trunc i64 %1037 to i32, !dbg !18
  %1039 = insertelement <4 x i32> %1036, i32 %1038, i64 1, !dbg !18
  %1040 = insertelement <4 x i32> %1039, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc783 = insertelement <4 x i32> %1040, i32 131072, i64 3, !dbg !18
  %1041 = ptrtoint ptr addrspace(3) %1031 to i32, !dbg !18
  %dma.buffer.lds784 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1041), !dbg !18
  %1042 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds784, i32 %dma.buffer.offset782, <4 x i32> %dma.buffer.rsrc783) #10, !dbg !18
  %1043 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local500), !dbg !18
  %1044 = zext i32 %1043 to i64, !dbg !18
  %1045 = inttoptr i64 %1044 to ptr addrspace(3), !dbg !18
  %address.local510 = getelementptr [2 x i8], ptr addrspace(1) %1028, i64 %address.local505, !dbg !18
  %1046 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1047 = ptrtoint ptr addrspace(1) %address.local510 to i64, !dbg !18
  %1048 = sub i64 %1047, %1046, !dbg !18
  %dma.buffer.offset = trunc i64 %1048 to i32, !dbg !18
  %1049 = trunc i64 %1046 to i32, !dbg !18
  %1050 = insertelement <4 x i32> poison, i32 %1049, i64 0, !dbg !18
  %1051 = lshr i64 %1046, 32, !dbg !18
  %1052 = trunc i64 %1051 to i32, !dbg !18
  %1053 = insertelement <4 x i32> %1050, i32 %1052, i64 1, !dbg !18
  %1054 = insertelement <4 x i32> %1053, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1054, i32 131072, i64 3, !dbg !18
  %1055 = ptrtoint ptr addrspace(3) %1045 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1055), !dbg !18
  %1056 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1057 = shufflevector <8 x half> %1012, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1057, <4 x float> %699) #9, !dbg !15
  %1059 = shufflevector <8 x half> %1014, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1059, <4 x float> %1058) #9, !dbg !15
  %1061 = shufflevector <8 x half> %1016, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %1061, <4 x float> %1060) #9, !dbg !15
  %1063 = shufflevector <8 x half> %1018, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1063, <4 x float> %1062) #9, !dbg !15
  %1065 = shufflevector <8 x half> %1020, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1065, <4 x float> %1064) #9, !dbg !15
  %1067 = shufflevector <8 x half> %1022, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1067, <4 x float> %1066) #9, !dbg !15
  %1069 = shufflevector <8 x half> %1024, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1069, <4 x float> %1068) #9, !dbg !15
  %1071 = shufflevector <8 x half> %1026, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1071, <4 x float> %1070) #9, !dbg !15
  %1073 = shufflevector <8 x half> %1012, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1073, <4 x float> %698) #9, !dbg !15
  %1075 = shufflevector <8 x half> %1014, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1075, <4 x float> %1074) #9, !dbg !15
  %1077 = shufflevector <8 x half> %1016, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %1077, <4 x float> %1076) #9, !dbg !15
  %1079 = shufflevector <8 x half> %1018, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1079, <4 x float> %1078) #9, !dbg !15
  %1081 = shufflevector <8 x half> %1020, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1081, <4 x float> %1080) #9, !dbg !15
  %1083 = shufflevector <8 x half> %1022, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1083, <4 x float> %1082) #9, !dbg !15
  %1085 = shufflevector <8 x half> %1024, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1085, <4 x float> %1084) #9, !dbg !15
  %1087 = shufflevector <8 x half> %1026, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1087, <4 x float> %1086) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1089 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local448, i32 0) #0, !dbg !15
  %1090 = bitcast <4 x i32> %1089 to <8 x half>, !dbg !15
  %1091 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local461, i32 0) #0, !dbg !15
  %1092 = bitcast <4 x i32> %1091 to <8 x half>, !dbg !15
  %1093 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local465, i32 0) #0, !dbg !15
  %1094 = bitcast <4 x i32> %1093 to <8 x half>, !dbg !15
  %1095 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local469, i32 0) #0, !dbg !15
  %1096 = bitcast <4 x i32> %1095 to <8 x half>, !dbg !15
  %1097 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local471, i32 0) #0, !dbg !15
  %1098 = bitcast <4 x i32> %1097 to <8 x half>, !dbg !15
  %1099 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local475, i32 0) #0, !dbg !15
  %1100 = bitcast <4 x i32> %1099 to <8 x half>, !dbg !15
  %1101 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local479, i32 0) #0, !dbg !15
  %1102 = bitcast <4 x i32> %1101 to <8 x half>, !dbg !15
  %1103 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local483, i32 0) #0, !dbg !15
  %1104 = bitcast <4 x i32> %1103 to <8 x half>, !dbg !15
  %1105 = shufflevector <8 x half> %1090, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1105, <4 x float> %697) #9, !dbg !15
  %1107 = shufflevector <8 x half> %1092, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1107, <4 x float> %1106) #9, !dbg !15
  %1109 = shufflevector <8 x half> %1094, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %1109, <4 x float> %1108) #9, !dbg !15
  %1111 = shufflevector <8 x half> %1096, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1111, <4 x float> %1110) #9, !dbg !15
  %1113 = shufflevector <8 x half> %1098, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1113, <4 x float> %1112) #9, !dbg !15
  %1115 = shufflevector <8 x half> %1100, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1115, <4 x float> %1114) #9, !dbg !15
  %1117 = shufflevector <8 x half> %1102, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1117, <4 x float> %1116) #9, !dbg !15
  %1119 = shufflevector <8 x half> %1104, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1119, <4 x float> %1118) #9, !dbg !15
  %1121 = shufflevector <8 x half> %1090, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1121, <4 x float> %696) #9, !dbg !15
  %1123 = shufflevector <8 x half> %1092, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1123, <4 x float> %1122) #9, !dbg !15
  %1125 = shufflevector <8 x half> %1094, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %1125, <4 x float> %1124) #9, !dbg !15
  %1127 = shufflevector <8 x half> %1096, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1127, <4 x float> %1126) #9, !dbg !15
  %1129 = shufflevector <8 x half> %1098, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1129, <4 x float> %1128) #9, !dbg !15
  %1131 = shufflevector <8 x half> %1100, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1131, <4 x float> %1130) #9, !dbg !15
  %1133 = shufflevector <8 x half> %1102, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1133, <4 x float> %1132) #9, !dbg !15
  %1135 = shufflevector <8 x half> %1104, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1135, <4 x float> %1134) #9, !dbg !15
  %1137 = shufflevector <4 x float> %916, <4 x float> %932, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1138 = shufflevector <4 x float> %994, <4 x float> %1010, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1139 = shufflevector <4 x float> %1072, <4 x float> %1088, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1140 = shufflevector <4 x float> %1120, <4 x float> %1136, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1141 = add nuw nsw i64 %54, 1, !dbg !7
  br label %53, !dbg !7

1142:                                             ; preds = %53
  %1143 = and i32 %28, 16, !dbg !7
  %.not = icmp eq i32 %1143, 0, !dbg !7
  %1144 = and i32 %28, 32, !dbg !7
  %.not1 = icmp eq i32 %1144, 0, !dbg !7
  %1145 = extractelement <8 x float> %68, i64 0, !dbg !7
  %1146 = extractelement <8 x float> %68, i64 1, !dbg !7
  %1147 = extractelement <8 x float> %68, i64 2, !dbg !7
  %1148 = extractelement <8 x float> %68, i64 3, !dbg !7
  %1149 = select i1 %.not, float %1146, float %1145, !dbg !7
  %1150 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1151 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1150), !dbg !7
  %1152 = add nuw nsw i32 %1151, 64, !dbg !7
  %1153 = xor i32 %1151, 16, !dbg !7
  %1154 = and i32 %1152, 192, !dbg !7
  %1155 = icmp samesign ult i32 %1153, %1154, !dbg !7
  %1156 = select i1 %1155, i32 %1153, i32 %1151, !dbg !7
  %1157 = shl nuw nsw i32 %1156, 2, !dbg !7
  %1158 = bitcast float %1149 to i32, !dbg !7
  %1159 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1158), !dbg !7
  %1160 = bitcast i32 %1159 to float, !dbg !7
  %1161 = select i1 %.not, float %1145, float %1160, !dbg !7
  %1162 = select i1 %.not, float %1160, float %1146, !dbg !7
  %1163 = select i1 %.not, float %1148, float %1147, !dbg !7
  %1164 = bitcast float %1163 to i32, !dbg !7
  %1165 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1164), !dbg !7
  %1166 = bitcast i32 %1165 to float, !dbg !7
  %1167 = select i1 %.not, float %1147, float %1166, !dbg !7
  %1168 = select i1 %.not, float %1166, float %1148, !dbg !7
  %1169 = select i1 %.not1, float %1167, float %1161, !dbg !7
  %1170 = xor i32 %1151, 32, !dbg !7
  %1171 = icmp samesign ult i32 %1170, %1154, !dbg !7
  %1172 = select i1 %1171, i32 %1170, i32 %1151, !dbg !7
  %1173 = shl nuw nsw i32 %1172, 2, !dbg !7
  %1174 = bitcast float %1169 to i32, !dbg !7
  %1175 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1174), !dbg !7
  %1176 = bitcast i32 %1175 to float, !dbg !7
  %1177 = select i1 %.not1, float %1161, float %1176, !dbg !7
  %1178 = select i1 %.not1, float %1176, float %1167, !dbg !7
  %1179 = select i1 %.not1, float %1168, float %1162, !dbg !7
  %1180 = bitcast float %1179 to i32, !dbg !7
  %1181 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1180), !dbg !7
  %1182 = bitcast i32 %1181 to float, !dbg !7
  %1183 = select i1 %.not1, float %1162, float %1182, !dbg !7
  %1184 = select i1 %.not1, float %1182, float %1168, !dbg !7
  %1185 = extractelement <8 x float> %65, i64 0, !dbg !7
  %1186 = extractelement <8 x float> %65, i64 1, !dbg !7
  %1187 = extractelement <8 x float> %65, i64 2, !dbg !7
  %1188 = extractelement <8 x float> %65, i64 3, !dbg !7
  %1189 = select i1 %.not, float %1186, float %1185, !dbg !7
  %1190 = bitcast float %1189 to i32, !dbg !7
  %1191 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1190), !dbg !7
  %1192 = bitcast i32 %1191 to float, !dbg !7
  %1193 = select i1 %.not, float %1185, float %1192, !dbg !7
  %1194 = select i1 %.not, float %1192, float %1186, !dbg !7
  %1195 = select i1 %.not, float %1188, float %1187, !dbg !7
  %1196 = bitcast float %1195 to i32, !dbg !7
  %1197 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1196), !dbg !7
  %1198 = bitcast i32 %1197 to float, !dbg !7
  %1199 = select i1 %.not, float %1187, float %1198, !dbg !7
  %1200 = select i1 %.not, float %1198, float %1188, !dbg !7
  %1201 = select i1 %.not1, float %1199, float %1193, !dbg !7
  %1202 = bitcast float %1201 to i32, !dbg !7
  %1203 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1202), !dbg !7
  %1204 = bitcast i32 %1203 to float, !dbg !7
  %1205 = select i1 %.not1, float %1193, float %1204, !dbg !7
  %1206 = select i1 %.not1, float %1204, float %1199, !dbg !7
  %1207 = select i1 %.not1, float %1200, float %1194, !dbg !7
  %1208 = bitcast float %1207 to i32, !dbg !7
  %1209 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1208), !dbg !7
  %1210 = bitcast i32 %1209 to float, !dbg !7
  %1211 = select i1 %.not1, float %1194, float %1210, !dbg !7
  %1212 = select i1 %.not1, float %1210, float %1200, !dbg !7
  %1213 = extractelement <8 x float> %62, i64 0, !dbg !7
  %1214 = extractelement <8 x float> %62, i64 1, !dbg !7
  %1215 = extractelement <8 x float> %62, i64 2, !dbg !7
  %1216 = extractelement <8 x float> %62, i64 3, !dbg !7
  %1217 = select i1 %.not, float %1214, float %1213, !dbg !7
  %1218 = bitcast float %1217 to i32, !dbg !7
  %1219 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1218), !dbg !7
  %1220 = bitcast i32 %1219 to float, !dbg !7
  %1221 = select i1 %.not, float %1213, float %1220, !dbg !7
  %1222 = select i1 %.not, float %1220, float %1214, !dbg !7
  %1223 = select i1 %.not, float %1216, float %1215, !dbg !7
  %1224 = bitcast float %1223 to i32, !dbg !7
  %1225 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1224), !dbg !7
  %1226 = bitcast i32 %1225 to float, !dbg !7
  %1227 = select i1 %.not, float %1215, float %1226, !dbg !7
  %1228 = select i1 %.not, float %1226, float %1216, !dbg !7
  %1229 = select i1 %.not1, float %1227, float %1221, !dbg !7
  %1230 = bitcast float %1229 to i32, !dbg !7
  %1231 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1230), !dbg !7
  %1232 = bitcast i32 %1231 to float, !dbg !7
  %1233 = select i1 %.not1, float %1221, float %1232, !dbg !7
  %1234 = select i1 %.not1, float %1232, float %1227, !dbg !7
  %1235 = select i1 %.not1, float %1228, float %1222, !dbg !7
  %1236 = bitcast float %1235 to i32, !dbg !7
  %1237 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1236), !dbg !7
  %1238 = bitcast i32 %1237 to float, !dbg !7
  %1239 = select i1 %.not1, float %1222, float %1238, !dbg !7
  %1240 = select i1 %.not1, float %1238, float %1228, !dbg !7
  %1241 = extractelement <8 x float> %57, i64 0, !dbg !7
  %1242 = extractelement <8 x float> %57, i64 1, !dbg !7
  %1243 = extractelement <8 x float> %57, i64 2, !dbg !7
  %1244 = extractelement <8 x float> %57, i64 3, !dbg !7
  %1245 = select i1 %.not, float %1242, float %1241, !dbg !7
  %1246 = bitcast float %1245 to i32, !dbg !7
  %1247 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1246), !dbg !7
  %1248 = bitcast i32 %1247 to float, !dbg !7
  %1249 = select i1 %.not, float %1241, float %1248, !dbg !7
  %1250 = select i1 %.not, float %1248, float %1242, !dbg !7
  %1251 = select i1 %.not, float %1244, float %1243, !dbg !7
  %1252 = bitcast float %1251 to i32, !dbg !7
  %1253 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1252), !dbg !7
  %1254 = bitcast i32 %1253 to float, !dbg !7
  %1255 = select i1 %.not, float %1243, float %1254, !dbg !7
  %1256 = select i1 %.not, float %1254, float %1244, !dbg !7
  %1257 = select i1 %.not1, float %1255, float %1249, !dbg !7
  %1258 = bitcast float %1257 to i32, !dbg !7
  %1259 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1258), !dbg !7
  %1260 = select i1 %.not1, float %1256, float %1250, !dbg !7
  %1261 = bitcast float %1260 to i32, !dbg !7
  %1262 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1261), !dbg !7
  %1263 = shufflevector <1 x float> %58, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1264 = extractelement <8 x float> %68, i64 4, !dbg !7
  %1265 = extractelement <8 x float> %68, i64 5, !dbg !7
  %1266 = extractelement <8 x float> %68, i64 6, !dbg !7
  %1267 = extractelement <8 x float> %68, i64 7, !dbg !7
  %1268 = select i1 %.not, float %1265, float %1264, !dbg !7
  %1269 = bitcast float %1268 to i32, !dbg !7
  %1270 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1269), !dbg !7
  %1271 = bitcast i32 %1270 to float, !dbg !7
  %1272 = select i1 %.not, float %1264, float %1271, !dbg !7
  %1273 = select i1 %.not, float %1271, float %1265, !dbg !7
  %1274 = select i1 %.not, float %1267, float %1266, !dbg !7
  %1275 = bitcast float %1274 to i32, !dbg !7
  %1276 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1275), !dbg !7
  %1277 = bitcast i32 %1276 to float, !dbg !7
  %1278 = select i1 %.not, float %1266, float %1277, !dbg !7
  %1279 = select i1 %.not, float %1277, float %1267, !dbg !7
  %1280 = select i1 %.not1, float %1278, float %1272, !dbg !7
  %1281 = bitcast float %1280 to i32, !dbg !7
  %1282 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1281), !dbg !7
  %1283 = bitcast i32 %1282 to float, !dbg !7
  %1284 = select i1 %.not1, float %1272, float %1283, !dbg !7
  %1285 = select i1 %.not1, float %1283, float %1278, !dbg !7
  %1286 = select i1 %.not1, float %1279, float %1273, !dbg !7
  %1287 = bitcast float %1286 to i32, !dbg !7
  %1288 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1287), !dbg !7
  %1289 = bitcast i32 %1288 to float, !dbg !7
  %1290 = select i1 %.not1, float %1273, float %1289, !dbg !7
  %1291 = select i1 %.not1, float %1289, float %1279, !dbg !7
  %1292 = extractelement <8 x float> %65, i64 4, !dbg !7
  %1293 = extractelement <8 x float> %65, i64 5, !dbg !7
  %1294 = extractelement <8 x float> %65, i64 6, !dbg !7
  %1295 = extractelement <8 x float> %65, i64 7, !dbg !7
  %1296 = select i1 %.not, float %1293, float %1292, !dbg !7
  %1297 = bitcast float %1296 to i32, !dbg !7
  %1298 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1297), !dbg !7
  %1299 = bitcast i32 %1298 to float, !dbg !7
  %1300 = select i1 %.not, float %1292, float %1299, !dbg !7
  %1301 = select i1 %.not, float %1299, float %1293, !dbg !7
  %1302 = select i1 %.not, float %1295, float %1294, !dbg !7
  %1303 = bitcast float %1302 to i32, !dbg !7
  %1304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1303), !dbg !7
  %1305 = bitcast i32 %1304 to float, !dbg !7
  %1306 = select i1 %.not, float %1294, float %1305, !dbg !7
  %1307 = select i1 %.not, float %1305, float %1295, !dbg !7
  %1308 = select i1 %.not1, float %1306, float %1300, !dbg !7
  %1309 = bitcast float %1308 to i32, !dbg !7
  %1310 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1309), !dbg !7
  %1311 = bitcast i32 %1310 to float, !dbg !7
  %1312 = select i1 %.not1, float %1300, float %1311, !dbg !7
  %1313 = select i1 %.not1, float %1311, float %1306, !dbg !7
  %1314 = select i1 %.not1, float %1307, float %1301, !dbg !7
  %1315 = bitcast float %1314 to i32, !dbg !7
  %1316 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1315), !dbg !7
  %1317 = bitcast i32 %1316 to float, !dbg !7
  %1318 = select i1 %.not1, float %1301, float %1317, !dbg !7
  %1319 = select i1 %.not1, float %1317, float %1307, !dbg !7
  %1320 = extractelement <8 x float> %62, i64 4, !dbg !7
  %1321 = extractelement <8 x float> %62, i64 5, !dbg !7
  %1322 = extractelement <8 x float> %62, i64 6, !dbg !7
  %1323 = extractelement <8 x float> %62, i64 7, !dbg !7
  %1324 = select i1 %.not, float %1321, float %1320, !dbg !7
  %1325 = bitcast float %1324 to i32, !dbg !7
  %1326 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1325), !dbg !7
  %1327 = bitcast i32 %1326 to float, !dbg !7
  %1328 = select i1 %.not, float %1320, float %1327, !dbg !7
  %1329 = select i1 %.not, float %1327, float %1321, !dbg !7
  %1330 = select i1 %.not, float %1323, float %1322, !dbg !7
  %1331 = bitcast float %1330 to i32, !dbg !7
  %1332 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1331), !dbg !7
  %1333 = bitcast i32 %1332 to float, !dbg !7
  %1334 = select i1 %.not, float %1322, float %1333, !dbg !7
  %1335 = select i1 %.not, float %1333, float %1323, !dbg !7
  %1336 = select i1 %.not1, float %1334, float %1328, !dbg !7
  %1337 = bitcast float %1336 to i32, !dbg !7
  %1338 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1337), !dbg !7
  %1339 = bitcast i32 %1338 to float, !dbg !7
  %1340 = select i1 %.not1, float %1328, float %1339, !dbg !7
  %1341 = select i1 %.not1, float %1339, float %1334, !dbg !7
  %1342 = select i1 %.not1, float %1335, float %1329, !dbg !7
  %1343 = bitcast float %1342 to i32, !dbg !7
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1343), !dbg !7
  %1345 = bitcast i32 %1344 to float, !dbg !7
  %1346 = select i1 %.not1, float %1329, float %1345, !dbg !7
  %1347 = select i1 %.not1, float %1345, float %1335, !dbg !7
  %1348 = extractelement <8 x float> %57, i64 4, !dbg !7
  %1349 = extractelement <8 x float> %57, i64 5, !dbg !7
  %1350 = extractelement <8 x float> %57, i64 6, !dbg !7
  %1351 = extractelement <8 x float> %57, i64 7, !dbg !7
  %1352 = select i1 %.not, float %1349, float %1348, !dbg !7
  %1353 = bitcast float %1352 to i32, !dbg !7
  %1354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1353), !dbg !7
  %1355 = bitcast i32 %1354 to float, !dbg !7
  %1356 = select i1 %.not, float %1348, float %1355, !dbg !7
  %1357 = select i1 %.not, float %1355, float %1349, !dbg !7
  %1358 = select i1 %.not, float %1351, float %1350, !dbg !7
  %1359 = bitcast float %1358 to i32, !dbg !7
  %1360 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1157, i32 %1359), !dbg !7
  %1361 = bitcast i32 %1360 to float, !dbg !7
  %1362 = select i1 %.not, float %1350, float %1361, !dbg !7
  %1363 = select i1 %.not, float %1361, float %1351, !dbg !7
  %1364 = select i1 %.not1, float %1362, float %1356, !dbg !7
  %1365 = bitcast float %1364 to i32, !dbg !7
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1365), !dbg !7
  %1367 = select i1 %.not1, float %1363, float %1357, !dbg !7
  %1368 = bitcast float %1367 to i32, !dbg !7
  %1369 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1173, i32 %1368), !dbg !7
  br i1 true, label %1370, label %1402, !dbg !21

1370:                                             ; preds = %1142
  %1371 = bitcast i32 %1366 to float, !dbg !7
  %1372 = select i1 %.not1, float %1356, float %1371, !dbg !7
  %1373 = insertelement <4 x float> poison, float %1372, i64 0, !dbg !19
  %1374 = bitcast i32 %1369 to float, !dbg !7
  %1375 = select i1 %.not1, float %1357, float %1374, !dbg !7
  %1376 = insertelement <4 x float> %1373, float %1375, i64 1, !dbg !19
  %1377 = select i1 %.not1, float %1371, float %1362, !dbg !7
  %1378 = insertelement <4 x float> %1376, float %1377, i64 2, !dbg !19
  %1379 = select i1 %.not1, float %1374, float %1363, !dbg !7
  %1380 = insertelement <4 x float> %1378, float %1379, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1263, !dbg !19
  %normalized = fmul <4 x float> %1380, %row.reciprocal, !dbg !19
  %1381 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1382 = extractelement <4 x half> %1381, i64 3, !dbg !20
  %1383 = extractelement <4 x half> %1381, i64 2, !dbg !20
  %1384 = extractelement <4 x half> %1381, i64 1, !dbg !20
  %1385 = extractelement <4 x half> %1381, i64 0, !dbg !20
  %1386 = bitcast i32 %1259 to float, !dbg !7
  %1387 = select i1 %.not1, float %1249, float %1386, !dbg !7
  %1388 = insertelement <4 x float> poison, float %1387, i64 0, !dbg !19
  %1389 = bitcast i32 %1262 to float, !dbg !7
  %1390 = select i1 %.not1, float %1250, float %1389, !dbg !7
  %1391 = insertelement <4 x float> %1388, float %1390, i64 1, !dbg !19
  %1392 = select i1 %.not1, float %1386, float %1255, !dbg !7
  %1393 = insertelement <4 x float> %1391, float %1392, i64 2, !dbg !19
  %1394 = select i1 %.not1, float %1389, float %1256, !dbg !7
  %1395 = insertelement <4 x float> %1393, float %1394, i64 3, !dbg !19
  %normalized37 = fmul <4 x float> %1395, %row.reciprocal, !dbg !19
  %1396 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !20
  %1397 = extractelement <4 x half> %1396, i64 3, !dbg !20
  %1398 = extractelement <4 x half> %1396, i64 2, !dbg !20
  %1399 = extractelement <4 x half> %1396, i64 1, !dbg !20
  %1400 = extractelement <4 x half> %1396, i64 0, !dbg !20
  %1401 = shl nuw nsw i64 %19, 19, !dbg !21
  %address.thread511 = call i32 asm sideeffect "", "=v,0"(i32 %28) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local512 = zext nneg i32 %address.thread511 to i64, !dbg !13
  %address.local513 = add nsw i64 %20, %address.local512, !dbg !21
  %address.local514 = lshr i64 %address.local512, 6, !dbg !13
  %address.local515 = shl nuw nsw i64 %address.local514, 4, !dbg !21
  %address.local516 = add nsw i64 %address.local513, %address.local515, !dbg !21
  %address.local517 = and i64 %address.local512, 240, !dbg !21
  %address.local518 = sub nsw i64 %address.local516, %address.local517, !dbg !21
  %address.local519 = add nsw i64 %address.local518, %21, !dbg !21
  %address.local520 = shl nuw nsw i64 %address.local519, 7, !dbg !21
  %address.local521 = add nuw nsw i64 %1401, %address.local520, !dbg !21
  %address.local522 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local521, !dbg !21
  %address.local523 = lshr i64 %address.local512, 2, !dbg !13
  %address.local524 = and i64 %address.local523, 12, !dbg !13
  %address.local525 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local522, i64 %address.local524, !dbg !21
  store half %1400, ptr addrspace(1) %address.local525, align 2, !dbg !21
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local521, !dbg !21
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local526, i64 %address.local524, !dbg !21
  %address.local528 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local527, i64 2, !dbg !21
  store half %1399, ptr addrspace(1) %address.local528, align 2, !dbg !21
  %address.local529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local521, !dbg !21
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local529, i64 %address.local524, !dbg !21
  %address.local531 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local530, i64 4, !dbg !21
  store half %1398, ptr addrspace(1) %address.local531, align 2, !dbg !21
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local521, !dbg !21
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local532, i64 %address.local524, !dbg !21
  %address.local534 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local533, i64 6, !dbg !21
  store half %1397, ptr addrspace(1) %address.local534, align 2, !dbg !21
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local521, !dbg !21
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local535, i64 %address.local524, !dbg !21
  %address.local537 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local536, i64 32, !dbg !21
  store half %1385, ptr addrspace(1) %address.local537, align 2, !dbg !21
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local521, !dbg !21
  %address.local539 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local538, i64 %address.local524, !dbg !21
  %address.local540 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local539, i64 34, !dbg !21
  store half %1384, ptr addrspace(1) %address.local540, align 2, !dbg !21
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local521, !dbg !21
  %address.local542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local541, i64 %address.local524, !dbg !21
  %address.local543 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local542, i64 36, !dbg !21
  store half %1383, ptr addrspace(1) %address.local543, align 2, !dbg !21
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local521, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local544, i64 %address.local524, !dbg !21
  %address.local546 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local545, i64 38, !dbg !21
  store half %1382, ptr addrspace(1) %address.local546, align 2, !dbg !21
  br label %1402, !dbg !21

1402:                                             ; preds = %1370, %1142
  br i1 true, label %1403, label %1423, !dbg !21

1403:                                             ; preds = %1402
  %1404 = insertelement <4 x float> poison, float %1340, i64 0, !dbg !19
  %1405 = insertelement <4 x float> %1404, float %1346, i64 1, !dbg !19
  %1406 = insertelement <4 x float> %1405, float %1341, i64 2, !dbg !19
  %1407 = insertelement <4 x float> %1406, float %1347, i64 3, !dbg !19
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1263, !dbg !19
  %normalized39 = fmul <4 x float> %1407, %row.reciprocal38, !dbg !19
  %1408 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !20
  %1409 = extractelement <4 x half> %1408, i64 3, !dbg !20
  %1410 = extractelement <4 x half> %1408, i64 2, !dbg !20
  %1411 = extractelement <4 x half> %1408, i64 1, !dbg !20
  %1412 = extractelement <4 x half> %1408, i64 0, !dbg !20
  %1413 = insertelement <4 x float> poison, float %1233, i64 0, !dbg !19
  %1414 = insertelement <4 x float> %1413, float %1239, i64 1, !dbg !19
  %1415 = insertelement <4 x float> %1414, float %1234, i64 2, !dbg !19
  %1416 = insertelement <4 x float> %1415, float %1240, i64 3, !dbg !19
  %normalized40 = fmul <4 x float> %1416, %row.reciprocal38, !dbg !19
  %1417 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !20
  %1418 = extractelement <4 x half> %1417, i64 3, !dbg !20
  %1419 = extractelement <4 x half> %1417, i64 2, !dbg !20
  %1420 = extractelement <4 x half> %1417, i64 1, !dbg !20
  %1421 = extractelement <4 x half> %1417, i64 0, !dbg !20
  %1422 = shl nuw nsw i64 %25, 19, !dbg !21
  %address.thread547 = call i32 asm sideeffect "", "=v,0"(i32 %28) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local548 = zext nneg i32 %address.thread547 to i64, !dbg !13
  %address.local549 = add nsw i64 %20, %address.local548, !dbg !21
  %address.local550 = lshr i64 %address.local548, 6, !dbg !13
  %address.local551 = shl nuw nsw i64 %address.local550, 4, !dbg !21
  %address.local552 = add nsw i64 %address.local549, %address.local551, !dbg !21
  %address.local553 = and i64 %address.local548, 240, !dbg !21
  %address.local554 = sub nsw i64 %address.local552, %address.local553, !dbg !21
  %address.local555 = add nsw i64 %address.local554, %21, !dbg !21
  %address.local556 = shl nuw nsw i64 %address.local555, 7, !dbg !21
  %address.local557 = add nuw nsw i64 %1422, %address.local556, !dbg !21
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local557, !dbg !21
  %address.local559 = lshr i64 %address.local548, 2, !dbg !13
  %address.local560 = and i64 %address.local559, 12, !dbg !13
  %address.local561 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local558, i64 %address.local560, !dbg !21
  %address.local562 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local561, i64 64, !dbg !21
  store half %1421, ptr addrspace(1) %address.local562, align 2, !dbg !21
  %address.local563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local557, !dbg !21
  %address.local564 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local563, i64 %address.local560, !dbg !21
  %address.local565 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local564, i64 66, !dbg !21
  store half %1420, ptr addrspace(1) %address.local565, align 2, !dbg !21
  %address.local566 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local557, !dbg !21
  %address.local567 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local566, i64 %address.local560, !dbg !21
  %address.local568 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local567, i64 68, !dbg !21
  store half %1419, ptr addrspace(1) %address.local568, align 2, !dbg !21
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local557, !dbg !21
  %address.local570 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local569, i64 %address.local560, !dbg !21
  %address.local571 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local570, i64 70, !dbg !21
  store half %1418, ptr addrspace(1) %address.local571, align 2, !dbg !21
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local557, !dbg !21
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local572, i64 %address.local560, !dbg !21
  %address.local574 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local573, i64 96, !dbg !21
  store half %1412, ptr addrspace(1) %address.local574, align 2, !dbg !21
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local557, !dbg !21
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local560, !dbg !21
  %address.local577 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local576, i64 98, !dbg !21
  store half %1411, ptr addrspace(1) %address.local577, align 2, !dbg !21
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local557, !dbg !21
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local578, i64 %address.local560, !dbg !21
  %address.local580 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local579, i64 100, !dbg !21
  store half %1410, ptr addrspace(1) %address.local580, align 2, !dbg !21
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local557, !dbg !21
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local560, !dbg !21
  %address.local583 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local582, i64 102, !dbg !21
  store half %1409, ptr addrspace(1) %address.local583, align 2, !dbg !21
  br label %1423, !dbg !21

1423:                                             ; preds = %1403, %1402
  br i1 true, label %1424, label %1444, !dbg !21

1424:                                             ; preds = %1423
  %1425 = insertelement <4 x float> poison, float %1312, i64 0, !dbg !19
  %1426 = insertelement <4 x float> %1425, float %1318, i64 1, !dbg !19
  %1427 = insertelement <4 x float> %1426, float %1313, i64 2, !dbg !19
  %1428 = insertelement <4 x float> %1427, float %1319, i64 3, !dbg !19
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1263, !dbg !19
  %normalized42 = fmul <4 x float> %1428, %row.reciprocal41, !dbg !19
  %1429 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !20
  %1430 = extractelement <4 x half> %1429, i64 3, !dbg !20
  %1431 = extractelement <4 x half> %1429, i64 2, !dbg !20
  %1432 = extractelement <4 x half> %1429, i64 1, !dbg !20
  %1433 = extractelement <4 x half> %1429, i64 0, !dbg !20
  %1434 = insertelement <4 x float> poison, float %1205, i64 0, !dbg !19
  %1435 = insertelement <4 x float> %1434, float %1211, i64 1, !dbg !19
  %1436 = insertelement <4 x float> %1435, float %1206, i64 2, !dbg !19
  %1437 = insertelement <4 x float> %1436, float %1212, i64 3, !dbg !19
  %normalized43 = fmul <4 x float> %1437, %row.reciprocal41, !dbg !19
  %1438 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !20
  %1439 = extractelement <4 x half> %1438, i64 3, !dbg !20
  %1440 = extractelement <4 x half> %1438, i64 2, !dbg !20
  %1441 = extractelement <4 x half> %1438, i64 1, !dbg !20
  %1442 = extractelement <4 x half> %1438, i64 0, !dbg !20
  %1443 = shl nuw nsw i64 %25, 19, !dbg !21
  %address.thread584 = call i32 asm sideeffect "", "=v,0"(i32 %28) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local585 = zext nneg i32 %address.thread584 to i64, !dbg !13
  %address.local586 = add nsw i64 %20, %address.local585, !dbg !21
  %address.local587 = lshr i64 %address.local585, 6, !dbg !13
  %address.local588 = shl nuw nsw i64 %address.local587, 4, !dbg !21
  %address.local589 = add nsw i64 %address.local586, %address.local588, !dbg !21
  %address.local590 = and i64 %address.local585, 240, !dbg !21
  %address.local591 = sub nsw i64 %address.local589, %address.local590, !dbg !21
  %address.local592 = add nsw i64 %address.local591, %21, !dbg !21
  %address.local593 = shl nuw nsw i64 %address.local592, 7, !dbg !21
  %address.local594 = add nuw nsw i64 %1443, %address.local593, !dbg !21
  %address.local595 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local594, !dbg !21
  %address.local596 = lshr i64 %address.local585, 2, !dbg !13
  %address.local597 = and i64 %address.local596, 12, !dbg !13
  %address.local598 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local595, i64 %address.local597, !dbg !21
  %address.local599 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local598, i64 128, !dbg !21
  store half %1442, ptr addrspace(1) %address.local599, align 2, !dbg !21
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local594, !dbg !21
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local597, !dbg !21
  %address.local602 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local601, i64 130, !dbg !21
  store half %1441, ptr addrspace(1) %address.local602, align 2, !dbg !21
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local594, !dbg !21
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local603, i64 %address.local597, !dbg !21
  %address.local605 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local604, i64 132, !dbg !21
  store half %1440, ptr addrspace(1) %address.local605, align 2, !dbg !21
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local594, !dbg !21
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local606, i64 %address.local597, !dbg !21
  %address.local608 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local607, i64 134, !dbg !21
  store half %1439, ptr addrspace(1) %address.local608, align 2, !dbg !21
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local594, !dbg !21
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local609, i64 %address.local597, !dbg !21
  %address.local611 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local610, i64 160, !dbg !21
  store half %1433, ptr addrspace(1) %address.local611, align 2, !dbg !21
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local594, !dbg !21
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local612, i64 %address.local597, !dbg !21
  %address.local614 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local613, i64 162, !dbg !21
  store half %1432, ptr addrspace(1) %address.local614, align 2, !dbg !21
  %address.local615 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local594, !dbg !21
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local597, !dbg !21
  %address.local617 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local616, i64 164, !dbg !21
  store half %1431, ptr addrspace(1) %address.local617, align 2, !dbg !21
  %address.local618 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local594, !dbg !21
  %address.local619 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local618, i64 %address.local597, !dbg !21
  %address.local620 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local619, i64 166, !dbg !21
  store half %1430, ptr addrspace(1) %address.local620, align 2, !dbg !21
  br label %1444, !dbg !21

1444:                                             ; preds = %1424, %1423
  br i1 true, label %1445, label %1465, !dbg !21

1445:                                             ; preds = %1444
  %1446 = insertelement <4 x float> poison, float %1284, i64 0, !dbg !19
  %1447 = insertelement <4 x float> %1446, float %1290, i64 1, !dbg !19
  %1448 = insertelement <4 x float> %1447, float %1285, i64 2, !dbg !19
  %1449 = insertelement <4 x float> %1448, float %1291, i64 3, !dbg !19
  %row.reciprocal44 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1263, !dbg !19
  %normalized45 = fmul <4 x float> %1449, %row.reciprocal44, !dbg !19
  %1450 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !20
  %1451 = extractelement <4 x half> %1450, i64 3, !dbg !20
  %1452 = extractelement <4 x half> %1450, i64 2, !dbg !20
  %1453 = extractelement <4 x half> %1450, i64 1, !dbg !20
  %1454 = extractelement <4 x half> %1450, i64 0, !dbg !20
  %1455 = insertelement <4 x float> poison, float %1177, i64 0, !dbg !19
  %1456 = insertelement <4 x float> %1455, float %1183, i64 1, !dbg !19
  %1457 = insertelement <4 x float> %1456, float %1178, i64 2, !dbg !19
  %1458 = insertelement <4 x float> %1457, float %1184, i64 3, !dbg !19
  %normalized46 = fmul <4 x float> %1458, %row.reciprocal44, !dbg !19
  %1459 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !20
  %1460 = extractelement <4 x half> %1459, i64 3, !dbg !20
  %1461 = extractelement <4 x half> %1459, i64 2, !dbg !20
  %1462 = extractelement <4 x half> %1459, i64 1, !dbg !20
  %1463 = extractelement <4 x half> %1459, i64 0, !dbg !20
  %1464 = shl nuw nsw i64 %25, 19, !dbg !21
  %address.thread621 = call i32 asm sideeffect "", "=v,0"(i32 %28) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local622 = zext nneg i32 %address.thread621 to i64, !dbg !13
  %address.local623 = add nsw i64 %20, %address.local622, !dbg !21
  %address.local624 = lshr i64 %address.local622, 6, !dbg !13
  %address.local625 = shl nuw nsw i64 %address.local624, 4, !dbg !21
  %address.local626 = add nsw i64 %address.local623, %address.local625, !dbg !21
  %address.local627 = and i64 %address.local622, 240, !dbg !21
  %address.local628 = sub nsw i64 %address.local626, %address.local627, !dbg !21
  %address.local629 = add nsw i64 %address.local628, %21, !dbg !21
  %address.local630 = shl nuw nsw i64 %address.local629, 7, !dbg !21
  %address.local631 = add nuw nsw i64 %1464, %address.local630, !dbg !21
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local631, !dbg !21
  %address.local633 = lshr i64 %address.local622, 2, !dbg !13
  %address.local634 = and i64 %address.local633, 12, !dbg !13
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local632, i64 %address.local634, !dbg !21
  %address.local636 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local635, i64 192, !dbg !21
  store half %1463, ptr addrspace(1) %address.local636, align 2, !dbg !21
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local631, !dbg !21
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local637, i64 %address.local634, !dbg !21
  %address.local639 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local638, i64 194, !dbg !21
  store half %1462, ptr addrspace(1) %address.local639, align 2, !dbg !21
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local631, !dbg !21
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local640, i64 %address.local634, !dbg !21
  %address.local642 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local641, i64 196, !dbg !21
  store half %1461, ptr addrspace(1) %address.local642, align 2, !dbg !21
  %address.local643 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local631, !dbg !21
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local634, !dbg !21
  %address.local645 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local644, i64 198, !dbg !21
  store half %1460, ptr addrspace(1) %address.local645, align 2, !dbg !21
  %address.local646 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local631, !dbg !21
  %address.local647 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local646, i64 %address.local634, !dbg !21
  %address.local648 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local647, i64 224, !dbg !21
  store half %1454, ptr addrspace(1) %address.local648, align 2, !dbg !21
  %address.local649 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local631, !dbg !21
  %address.local650 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local649, i64 %address.local634, !dbg !21
  %address.local651 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local650, i64 226, !dbg !21
  store half %1453, ptr addrspace(1) %address.local651, align 2, !dbg !21
  %address.local652 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local631, !dbg !21
  %address.local653 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local652, i64 %address.local634, !dbg !21
  %address.local654 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local653, i64 228, !dbg !21
  store half %1452, ptr addrspace(1) %address.local654, align 2, !dbg !21
  %address.local655 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local631, !dbg !21
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local655, i64 %address.local634, !dbg !21
  %address.local657 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local656, i64 230, !dbg !21
  store half %1451, ptr addrspace(1) %address.local657, align 2, !dbg !21
  br label %1465, !dbg !21

1465:                                             ; preds = %1445, %1444
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/pipeline-loop-only")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!9 = !DILocation(line: 48, column: 17, scope: !4)
!12 = !DILocation(line: 70, column: 22, scope: !4)
!13 = !DILocation(line: 72, column: 24, scope: !4)
!14 = !DILocation(line: 77, column: 19, scope: !4)
!15 = !DILocation(line: 102, column: 23, scope: !4)
!16 = !DILocation(line: 71, column: 22, scope: !4)
!17 = !DILocation(line: 73, column: 26, scope: !4)
!18 = !DILocation(line: 101, column: 20, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
!22 = !DILocation(line: 63, column: 17, scope: !4)
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
