; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0005-test_input_1x32x4096x128-128-64-32-online/pipeline-loop-only/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0005-test_input_1x32x4096x128-128-64-32-online/pipeline-loop-only/input.mlir"

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
  %29 = or disjoint i64 %11, 127, !dbg !14
  %30 = add nsw i64 %11, 128, !dbg !14
  %31 = icmp uge i64 %30, %29, !dbg !14
  %32 = shl nsw i64 %10, 1, !dbg !7
  %33 = add nsw i64 %32, 2, !dbg !7
  %.idx = shl nuw nsw i64 %17, 20, !dbg !12
  %34 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %35 = getelementptr [2 x i8], ptr addrspace(1) %34, i64 %21, !dbg !12
  %36 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %37 = getelementptr i8, ptr addrspace(1) %36, i64 64, !dbg !12
  %38 = shl nsw i64 %8, 13, !dbg !16
  %39 = shl nsw i64 %6, 18, !dbg !16
  %40 = shl nsw i64 %8, 7, !dbg !16
  %41 = shl nsw i64 %6, 12, !dbg !16
  %42 = add nsw i64 %40, %41, !dbg !16
  %43 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %44 = getelementptr i8, ptr addrspace(1) %43, i64 128, !dbg !12
  %45 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %46 = getelementptr i8, ptr addrspace(1) %45, i64 192, !dbg !12
  br label %47, !dbg !7

47:                                               ; preds = %572, %4
  %48 = phi i64 [ %1109, %572 ], [ 0, %4 ], !dbg !7
  %49 = phi [2 x <8 x float>] [ %1094, %572 ], [ zeroinitializer, %4 ], !dbg !7
  %50 = phi [2 x <1 x float>] [ %870, %572 ], [ zeroinitializer, %4 ], !dbg !7
  %51 = phi [2 x <1 x float>] [ %1096, %572 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %52 = phi [2 x <8 x float>] [ %1100, %572 ], [ zeroinitializer, %4 ], !dbg !7
  %53 = phi [2 x <8 x float>] [ %1104, %572 ], [ zeroinitializer, %4 ], !dbg !7
  %54 = phi [2 x <8 x float>] [ %1108, %572 ], [ zeroinitializer, %4 ], !dbg !7
  %55 = icmp slt i64 %48, %33, !dbg !7
  br i1 %55, label %56, label %1110, !dbg !7

56:                                               ; preds = %47
  %57 = shl nsw i64 %48, 6, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !12, !range !40, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local98 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local99 = and i64 %address.local98, 1536, !dbg !12
  %address.local100 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local99, !dbg !12
  %address.local101 = ptrtoint ptr addrspace(3) %address.local100 to i64, !dbg !12
  %address.local102 = trunc i64 %address.local101 to i32, !dbg !12
  %58 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local102), !dbg !12
  %59 = zext i32 %58 to i64, !dbg !12
  %60 = inttoptr i64 %59 to ptr addrspace(3), !dbg !12
  %address.local103 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local104 = and i64 %address.local103, 16256, !dbg !12
  %address.local105 = lshr i64 %address.local, 4, !dbg !12
  %address.local106 = and i64 %address.local105, 8, !dbg !12
  %address.local107 = or disjoint i64 %address.local104, %address.local106, !dbg !12
  %address.local108 = getelementptr [2 x i8], ptr addrspace(1) %35, i64 %address.local107, !dbg !12
  %61 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %62 = ptrtoint ptr addrspace(1) %address.local108 to i64, !dbg !12
  %63 = sub i64 %62, %61, !dbg !12
  %dma.buffer.offset767 = trunc i64 %63 to i32, !dbg !12
  %64 = trunc i64 %61 to i32, !dbg !12
  %65 = insertelement <4 x i32> poison, i32 %64, i64 0, !dbg !12
  %66 = lshr i64 %61, 32, !dbg !12
  %67 = trunc i64 %66 to i32, !dbg !12
  %68 = insertelement <4 x i32> %65, i32 %67, i64 1, !dbg !12
  %69 = insertelement <4 x i32> %68, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc768 = insertelement <4 x i32> %69, i32 131072, i64 3, !dbg !12
  %70 = ptrtoint ptr addrspace(3) %60 to i32, !dbg !12
  %dma.buffer.lds769 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %70), !dbg !12
  %71 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds769, i32 %dma.buffer.offset767, <4 x i32> %dma.buffer.rsrc768) #10, !dbg !12
  %address.local109 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local110 = shl nuw nsw i64 %address.local109, 3, !dbg !12
  %address.local111 = and i64 %address.local110, 3584, !dbg !12
  %address.local112 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local111, !dbg !12
  %address.local113 = ptrtoint ptr addrspace(3) %address.local112 to i64, !dbg !12
  %address.local114 = trunc i64 %address.local113 to i32, !dbg !12
  %72 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local114), !dbg !12
  %73 = zext i32 %72 to i64, !dbg !12
  %74 = inttoptr i64 %73 to ptr addrspace(3), !dbg !12
  %address.local115 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local116 = and i64 %address.local115, 16256, !dbg !12
  %address.local117 = lshr i64 %address.local109, 4, !dbg !12
  %address.local118 = and i64 %address.local117, 24, !dbg !12
  %address.local119 = or disjoint i64 %address.local116, %address.local118, !dbg !12
  %address.local120 = getelementptr [2 x i8], ptr addrspace(1) %35, i64 %address.local119, !dbg !12
  %75 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %76 = ptrtoint ptr addrspace(1) %address.local120 to i64, !dbg !12
  %77 = sub i64 %76, %75, !dbg !12
  %dma.buffer.offset764 = trunc i64 %77 to i32, !dbg !12
  %78 = trunc i64 %75 to i32, !dbg !12
  %79 = insertelement <4 x i32> poison, i32 %78, i64 0, !dbg !12
  %80 = lshr i64 %75, 32, !dbg !12
  %81 = trunc i64 %80 to i32, !dbg !12
  %82 = insertelement <4 x i32> %79, i32 %81, i64 1, !dbg !12
  %83 = insertelement <4 x i32> %82, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc765 = insertelement <4 x i32> %83, i32 131072, i64 3, !dbg !12
  %84 = ptrtoint ptr addrspace(3) %74 to i32, !dbg !12
  %dma.buffer.lds766 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %84), !dbg !12
  %85 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds766, i32 %dma.buffer.offset764, <4 x i32> %dma.buffer.rsrc765) #10, !dbg !12
  %86 = lshr i64 %48, 13, !dbg !16
  %87 = add nsw i64 %86, %8, !dbg !16
  %88 = lshr i64 %48, 6, !dbg !16
  %89 = and i64 %88, 127, !dbg !16
  %90 = shl i64 %87, 19, !dbg !16
  %91 = and i64 %90, 16252928, !dbg !16
  %92 = shl i64 %88, 19, !dbg !16
  %93 = shl i64 %48, 13, !dbg !16
  %94 = sub i64 %93, %92, !dbg !16
  %95 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %91, !dbg !16
  %96 = getelementptr [2 x i8], ptr addrspace(1) %95, i64 %89, !dbg !16
  %97 = getelementptr [2 x i8], ptr addrspace(1) %96, i64 %94, !dbg !16
  %address.local121 = and i64 %address.local98, 1536, !dbg !16
  %address.local122 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local121, !dbg !16
  %address.local123 = ptrtoint ptr addrspace(3) %address.local122 to i64, !dbg !16
  %address.local124 = trunc i64 %address.local123 to i32, !dbg !16
  %98 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local124), !dbg !16
  %99 = zext i32 %98 to i64, !dbg !16
  %100 = inttoptr i64 %99 to ptr addrspace(3), !dbg !16
  %address.local125 = lshr i64 %address.local, 3, !dbg !16
  %address.local126 = and i64 %address.local125, 24, !dbg !16
  %address.local127 = shl nuw nsw i64 %address.local, 7, !dbg !16
  %address.local128 = and i64 %address.local127, 8064, !dbg !16
  %address.local129 = or disjoint i64 %address.local126, %address.local128, !dbg !16
  %address.local130 = getelementptr [2 x i8], ptr addrspace(1) %97, i64 %address.local129, !dbg !16
  %101 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %102 = ptrtoint ptr addrspace(1) %address.local130 to i64, !dbg !16
  %103 = sub i64 %102, %101, !dbg !16
  %dma.buffer.offset761 = trunc i64 %103 to i32, !dbg !16
  %104 = trunc i64 %101 to i32, !dbg !16
  %105 = insertelement <4 x i32> poison, i32 %104, i64 0, !dbg !16
  %106 = lshr i64 %101, 32, !dbg !16
  %107 = trunc i64 %106 to i32, !dbg !16
  %108 = insertelement <4 x i32> %105, i32 %107, i64 1, !dbg !16
  %109 = insertelement <4 x i32> %108, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc762 = insertelement <4 x i32> %109, i32 131072, i64 3, !dbg !16
  %110 = ptrtoint ptr addrspace(3) %100 to i32, !dbg !16
  %dma.buffer.lds763 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %110), !dbg !16
  %111 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds763, i32 %dma.buffer.offset761, <4 x i32> %dma.buffer.rsrc762) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %address.local131 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local99, !dbg !12
  %address.local132 = ptrtoint ptr addrspace(3) %address.local131 to i64, !dbg !12
  %address.local133 = trunc i64 %address.local132 to i32, !dbg !12
  %112 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !12
  %113 = zext i32 %112 to i64, !dbg !12
  %114 = inttoptr i64 %113 to ptr addrspace(3), !dbg !12
  %address.local134 = getelementptr [2 x i8], ptr addrspace(1) %37, i64 %address.local107, !dbg !12
  %115 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %116 = ptrtoint ptr addrspace(1) %address.local134 to i64, !dbg !12
  %117 = sub i64 %116, %115, !dbg !12
  %dma.buffer.offset758 = trunc i64 %117 to i32, !dbg !12
  %118 = trunc i64 %115 to i32, !dbg !12
  %119 = insertelement <4 x i32> poison, i32 %118, i64 0, !dbg !12
  %120 = lshr i64 %115, 32, !dbg !12
  %121 = trunc i64 %120 to i32, !dbg !12
  %122 = insertelement <4 x i32> %119, i32 %121, i64 1, !dbg !12
  %123 = insertelement <4 x i32> %122, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc759 = insertelement <4 x i32> %123, i32 131072, i64 3, !dbg !12
  %124 = ptrtoint ptr addrspace(3) %114 to i32, !dbg !12
  %dma.buffer.lds760 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %124), !dbg !12
  %125 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds760, i32 %dma.buffer.offset758, <4 x i32> %dma.buffer.rsrc759) #10, !dbg !12
  %address.local135 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local111, !dbg !12
  %address.local136 = ptrtoint ptr addrspace(3) %address.local135 to i64, !dbg !12
  %address.local137 = trunc i64 %address.local136 to i32, !dbg !12
  %126 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local137), !dbg !12
  %127 = zext i32 %126 to i64, !dbg !12
  %128 = inttoptr i64 %127 to ptr addrspace(3), !dbg !12
  %address.local138 = getelementptr [2 x i8], ptr addrspace(1) %37, i64 %address.local119, !dbg !12
  %129 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %130 = ptrtoint ptr addrspace(1) %address.local138 to i64, !dbg !12
  %131 = sub i64 %130, %129, !dbg !12
  %dma.buffer.offset755 = trunc i64 %131 to i32, !dbg !12
  %132 = trunc i64 %129 to i32, !dbg !12
  %133 = insertelement <4 x i32> poison, i32 %132, i64 0, !dbg !12
  %134 = lshr i64 %129, 32, !dbg !12
  %135 = trunc i64 %134 to i32, !dbg !12
  %136 = insertelement <4 x i32> %133, i32 %135, i64 1, !dbg !12
  %137 = insertelement <4 x i32> %136, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc756 = insertelement <4 x i32> %137, i32 131072, i64 3, !dbg !12
  %138 = ptrtoint ptr addrspace(3) %128 to i32, !dbg !12
  %dma.buffer.lds757 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %138), !dbg !12
  %139 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds757, i32 %dma.buffer.offset755, <4 x i32> %dma.buffer.rsrc756) #10, !dbg !12
  %140 = add i64 %48, %38, !dbg !16
  %141 = add i64 %140, %39, !dbg !16
  %142 = add i64 %141, 2048, !dbg !16
  %143 = icmp slt i64 %142, 0, !dbg !16
  %144 = sub i64 -2049, %141, !dbg !16
  %145 = select i1 %143, i64 %144, i64 %142, !dbg !16
  %146 = sdiv i64 %145, 8192, !dbg !16
  %.lobit3 = ashr i64 %142, 63, !dbg !16
  %147 = xor i64 %146, %.lobit3, !dbg !16
  %148 = add nsw i64 %88, %42, !dbg !16
  %149 = add nsw i64 %148, 32, !dbg !16
  %150 = icmp slt i64 %148, -32, !dbg !16
  %151 = sub nsw i64 -33, %148, !dbg !16
  %152 = select i1 %150, i64 %151, i64 %149, !dbg !16
  %153 = sdiv i64 %152, 128, !dbg !16
  %154 = sext i1 %150 to i64, !dbg !16
  %155 = xor i64 %153, %154, !dbg !16
  %156 = shl nsw i64 %155, 7, !dbg !16
  %157 = sub i64 %42, %156, !dbg !16
  %158 = shl i64 %147, 19, !dbg !16
  %159 = and i64 %158, 16252928, !dbg !16
  %160 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %159, !dbg !16
  %161 = getelementptr [2 x i8], ptr addrspace(1) %160, i64 %88, !dbg !16
  %162 = getelementptr [2 x i8], ptr addrspace(1) %161, i64 %157, !dbg !16
  %163 = getelementptr i8, ptr addrspace(1) %162, i64 64, !dbg !16
  %164 = getelementptr [2 x i8], ptr addrspace(1) %163, i64 %94, !dbg !16
  %address.local139 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 20480), i64 %address.local121, !dbg !16
  %address.local140 = ptrtoint ptr addrspace(3) %address.local139 to i64, !dbg !16
  %address.local141 = trunc i64 %address.local140 to i32, !dbg !16
  %165 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local141), !dbg !16
  %166 = zext i32 %165 to i64, !dbg !16
  %167 = inttoptr i64 %166 to ptr addrspace(3), !dbg !16
  %address.local142 = getelementptr [2 x i8], ptr addrspace(1) %164, i64 %address.local129, !dbg !16
  %168 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %169 = ptrtoint ptr addrspace(1) %address.local142 to i64, !dbg !16
  %170 = sub i64 %169, %168, !dbg !16
  %dma.buffer.offset752 = trunc i64 %170 to i32, !dbg !16
  %171 = trunc i64 %168 to i32, !dbg !16
  %172 = insertelement <4 x i32> poison, i32 %171, i64 0, !dbg !16
  %173 = lshr i64 %168, 32, !dbg !16
  %174 = trunc i64 %173 to i32, !dbg !16
  %175 = insertelement <4 x i32> %172, i32 %174, i64 1, !dbg !16
  %176 = insertelement <4 x i32> %175, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc753 = insertelement <4 x i32> %176, i32 131072, i64 3, !dbg !16
  %177 = ptrtoint ptr addrspace(3) %167 to i32, !dbg !16
  %dma.buffer.lds754 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %177), !dbg !16
  %178 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds754, i32 %dma.buffer.offset752, <4 x i32> %dma.buffer.rsrc753) #10, !dbg !16
  %address.local143 = lshr i32 %address.thread, 2, !dbg !13
  %address.local144 = and i32 %address.local143, 4, !dbg !13
  %address.local145 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local146 = and i32 %address.local145, 384, !dbg !13
  %address.local147 = or disjoint i32 %address.local144, %address.local146, !dbg !13
  %address.local148 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local149 = and i32 %address.local148, 120, !dbg !13
  %address.local150 = or disjoint i32 %address.local147, %address.local149, !dbg !13
  %address.local151 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local152 = and i32 %address.local151, 1024, !dbg !13
  %address.local153 = or disjoint i32 %address.local150, %address.local152, !dbg !13
  %address.local154 = zext nneg i32 %address.local153 to i64, !dbg !13
  %address.local155 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local154, !dbg !13
  %179 = load <4 x half>, ptr addrspace(3) %address.local155, align 2, !dbg !13
  %address.local156 = lshr i32 %address.thread, 2, !dbg !13
  %address.local157 = and i32 %address.local156, 4, !dbg !13
  %address.local158 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local159 = and i32 %address.local158, 384, !dbg !13
  %address.local160 = or disjoint i32 %address.local157, %address.local159, !dbg !13
  %address.local161 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local162 = and i32 %address.local161, 120, !dbg !13
  %address.local163 = or disjoint i32 %address.local160, %address.local162, !dbg !13
  %address.local164 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local165 = and i32 %address.local164, 1024, !dbg !13
  %address.local166 = or disjoint i32 %address.local163, %address.local165, !dbg !13
  %address.local167 = or disjoint i32 %address.local166, 2048, !dbg !13
  %address.local168 = zext nneg i32 %address.local167 to i64, !dbg !13
  %address.local169 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local168, !dbg !13
  %180 = load <4 x half>, ptr addrspace(3) %address.local169, align 2, !dbg !13
  %address.local170 = lshr i32 %address.thread, 2, !dbg !13
  %address.local171 = and i32 %address.local170, 4, !dbg !13
  %address.local172 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local173 = and i32 %address.local172, 384, !dbg !13
  %address.local174 = or disjoint i32 %address.local171, %address.local173, !dbg !13
  %address.local175 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local176 = and i32 %address.local175, 120, !dbg !13
  %address.local177 = or disjoint i32 %address.local174, %address.local176, !dbg !13
  %address.local178 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local179 = and i32 %address.local178, 1024, !dbg !13
  %address.local180 = or disjoint i32 %address.local177, %address.local179, !dbg !13
  %address.local181 = or disjoint i32 %address.local180, 512, !dbg !13
  %address.local182 = zext nneg i32 %address.local181 to i64, !dbg !13
  %address.local183 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local182, !dbg !13
  %181 = load <4 x half>, ptr addrspace(3) %address.local183, align 2, !dbg !13
  %182 = fmul <4 x half> %179, splat (half 0xH3015), !dbg !13
  %address.local184 = lshr i32 %address.thread, 2, !dbg !13
  %address.local185 = and i32 %address.local184, 4, !dbg !13
  %address.local186 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local187 = and i32 %address.local186, 384, !dbg !13
  %address.local188 = or disjoint i32 %address.local185, %address.local187, !dbg !13
  %address.local189 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local190 = and i32 %address.local189, 120, !dbg !13
  %address.local191 = or disjoint i32 %address.local188, %address.local190, !dbg !13
  %address.local192 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local193 = and i32 %address.local192, 1024, !dbg !13
  %address.local194 = or disjoint i32 %address.local191, %address.local193, !dbg !13
  %address.local195 = or disjoint i32 %address.local194, 2560, !dbg !13
  %address.local196 = zext nneg i32 %address.local195 to i64, !dbg !13
  %address.local197 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local196, !dbg !13
  %183 = load <4 x half>, ptr addrspace(3) %address.local197, align 2, !dbg !13
  %184 = fmul <4 x half> %180, splat (half 0xH3015), !dbg !13
  %address.local198 = lshr i32 %address.thread, 2, !dbg !17
  %address.local199 = and i32 %address.local198, 4, !dbg !17
  %address.local200 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local201 = and i32 %address.local200, 24, !dbg !17
  %address.local202 = or disjoint i32 %address.local199, %address.local201, !dbg !17
  %address.local203 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local204 = and i32 %address.local203, 512, !dbg !17
  %address.local205 = or disjoint i32 %address.local202, %address.local204, !dbg !17
  %address.local206 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local207 = and i32 %address.local206, 384, !dbg !17
  %address.local208 = or disjoint i32 %address.local205, %address.local207, !dbg !17
  %address.local209 = zext nneg i32 %address.local208 to i64, !dbg !17
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local209, !dbg !17
  %185 = load <4 x half>, ptr addrspace(3) %address.local210, align 2, !dbg !17
  %186 = fmul <4 x half> %181, splat (half 0xH3015), !dbg !13
  %address.local211 = lshr i32 %address.thread, 2, !dbg !17
  %address.local212 = and i32 %address.local211, 4, !dbg !17
  %address.local213 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local214 = and i32 %address.local213, 24, !dbg !17
  %address.local215 = or disjoint i32 %address.local212, %address.local214, !dbg !17
  %address.local216 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local217 = and i32 %address.local216, 512, !dbg !17
  %address.local218 = or disjoint i32 %address.local215, %address.local217, !dbg !17
  %address.local219 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local220 = and i32 %address.local219, 384, !dbg !17
  %address.local221 = or disjoint i32 %address.local218, %address.local220, !dbg !17
  %address.local222 = or disjoint i32 %address.local221, 1024, !dbg !17
  %address.local223 = zext nneg i32 %address.local222 to i64, !dbg !17
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local223, !dbg !17
  %187 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !17
  %188 = fmul <4 x half> %183, splat (half 0xH3015), !dbg !13
  %address.local225 = lshr i32 %address.thread, 2, !dbg !17
  %address.local226 = and i32 %address.local225, 4, !dbg !17
  %address.local227 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local228 = and i32 %address.local227, 24, !dbg !17
  %address.local229 = or disjoint i32 %address.local226, %address.local228, !dbg !17
  %address.local230 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local231 = and i32 %address.local230, 512, !dbg !17
  %address.local232 = or disjoint i32 %address.local229, %address.local231, !dbg !17
  %address.local233 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local234 = and i32 %address.local233, 384, !dbg !17
  %address.local235 = or disjoint i32 %address.local232, %address.local234, !dbg !17
  %address.local236 = or disjoint i32 %address.local235, 32, !dbg !17
  %address.local237 = zext nneg i32 %address.local236 to i64, !dbg !17
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local237, !dbg !17
  %189 = load <4 x half>, ptr addrspace(3) %address.local238, align 2, !dbg !17
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %185, <4 x float> zeroinitializer) #9, !dbg !17
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %187, <4 x float> %190) #9, !dbg !17
  %address.local239 = lshr i32 %address.thread, 2, !dbg !17
  %address.local240 = and i32 %address.local239, 4, !dbg !17
  %address.local241 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local242 = and i32 %address.local241, 24, !dbg !17
  %address.local243 = or disjoint i32 %address.local240, %address.local242, !dbg !17
  %address.local244 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local245 = and i32 %address.local244, 512, !dbg !17
  %address.local246 = or disjoint i32 %address.local243, %address.local245, !dbg !17
  %address.local247 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local248 = and i32 %address.local247, 384, !dbg !17
  %address.local249 = or disjoint i32 %address.local246, %address.local248, !dbg !17
  %address.local250 = or disjoint i32 %address.local249, 1056, !dbg !17
  %address.local251 = zext nneg i32 %address.local250 to i64, !dbg !17
  %address.local252 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local251, !dbg !17
  %192 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !17
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %189, <4 x float> zeroinitializer) #9, !dbg !17
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %192, <4 x float> %193) #9, !dbg !17
  %address.local253 = lshr i32 %address.thread, 2, !dbg !17
  %address.local254 = and i32 %address.local253, 4, !dbg !17
  %address.local255 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local256 = and i32 %address.local255, 24, !dbg !17
  %address.local257 = or disjoint i32 %address.local254, %address.local256, !dbg !17
  %address.local258 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local259 = and i32 %address.local258, 512, !dbg !17
  %address.local260 = or disjoint i32 %address.local257, %address.local259, !dbg !17
  %address.local261 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local262 = and i32 %address.local261, 384, !dbg !17
  %address.local263 = or disjoint i32 %address.local260, %address.local262, !dbg !17
  %address.local264 = or disjoint i32 %address.local263, 64, !dbg !17
  %address.local265 = zext nneg i32 %address.local264 to i64, !dbg !17
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local265, !dbg !17
  %195 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !17
  %address.local267 = lshr i32 %address.thread, 2, !dbg !17
  %address.local268 = and i32 %address.local267, 4, !dbg !17
  %address.local269 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local270 = and i32 %address.local269, 24, !dbg !17
  %address.local271 = or disjoint i32 %address.local268, %address.local270, !dbg !17
  %address.local272 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local273 = and i32 %address.local272, 512, !dbg !17
  %address.local274 = or disjoint i32 %address.local271, %address.local273, !dbg !17
  %address.local275 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local276 = and i32 %address.local275, 384, !dbg !17
  %address.local277 = or disjoint i32 %address.local274, %address.local276, !dbg !17
  %address.local278 = or disjoint i32 %address.local277, 1088, !dbg !17
  %address.local279 = zext nneg i32 %address.local278 to i64, !dbg !17
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local279, !dbg !17
  %196 = load <4 x half>, ptr addrspace(3) %address.local280, align 2, !dbg !17
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %195, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local281 = lshr i32 %address.thread, 2, !dbg !17
  %address.local282 = and i32 %address.local281, 4, !dbg !17
  %address.local283 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local284 = and i32 %address.local283, 24, !dbg !17
  %address.local285 = or disjoint i32 %address.local282, %address.local284, !dbg !17
  %address.local286 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local287 = and i32 %address.local286, 512, !dbg !17
  %address.local288 = or disjoint i32 %address.local285, %address.local287, !dbg !17
  %address.local289 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local290 = and i32 %address.local289, 384, !dbg !17
  %address.local291 = or disjoint i32 %address.local288, %address.local290, !dbg !17
  %address.local292 = or disjoint i32 %address.local291, 96, !dbg !17
  %address.local293 = zext nneg i32 %address.local292 to i64, !dbg !17
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local293, !dbg !17
  %198 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !17
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %196, <4 x float> %197) #9, !dbg !17
  %address.local295 = lshr i32 %address.thread, 2, !dbg !17
  %address.local296 = and i32 %address.local295, 4, !dbg !17
  %address.local297 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local298 = and i32 %address.local297, 24, !dbg !17
  %address.local299 = or disjoint i32 %address.local296, %address.local298, !dbg !17
  %address.local300 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local301 = and i32 %address.local300, 512, !dbg !17
  %address.local302 = or disjoint i32 %address.local299, %address.local301, !dbg !17
  %address.local303 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local304 = and i32 %address.local303, 384, !dbg !17
  %address.local305 = or disjoint i32 %address.local302, %address.local304, !dbg !17
  %address.local306 = or disjoint i32 %address.local305, 1120, !dbg !17
  %address.local307 = zext nneg i32 %address.local306 to i64, !dbg !17
  %address.local308 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local307, !dbg !17
  %200 = load <4 x half>, ptr addrspace(3) %address.local308, align 2, !dbg !17
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %198, <4 x float> zeroinitializer) #9, !dbg !17
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %200, <4 x float> %201) #9, !dbg !17
  %203 = load <4 x half>, ptr addrspace(3) %address.local210, align 2, !dbg !17
  %204 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !17
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %203, <4 x float> zeroinitializer) #9, !dbg !17
  %206 = load <4 x half>, ptr addrspace(3) %address.local238, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %188, <4 x half> %204, <4 x float> %205) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %208 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %206, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %188, <4 x half> %208, <4 x float> %209) #9, !dbg !17
  %211 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !17
  %212 = load <4 x half>, ptr addrspace(3) %address.local280, align 2, !dbg !17
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %211, <4 x float> zeroinitializer) #9, !dbg !17
  %214 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %188, <4 x half> %212, <4 x float> %213) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %216 = load <4 x half>, ptr addrspace(3) %address.local308, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %214, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %188, <4 x half> %216, <4 x float> %217) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %219 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local102), !dbg !12
  %220 = zext i32 %219 to i64, !dbg !12
  %221 = inttoptr i64 %220 to ptr addrspace(3), !dbg !12
  %address.local309 = getelementptr [2 x i8], ptr addrspace(1) %44, i64 %address.local107, !dbg !12
  %222 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %223 = ptrtoint ptr addrspace(1) %address.local309 to i64, !dbg !12
  %224 = sub i64 %223, %222, !dbg !12
  %dma.buffer.offset749 = trunc i64 %224 to i32, !dbg !12
  %225 = trunc i64 %222 to i32, !dbg !12
  %226 = insertelement <4 x i32> poison, i32 %225, i64 0, !dbg !12
  %227 = lshr i64 %222, 32, !dbg !12
  %228 = trunc i64 %227 to i32, !dbg !12
  %229 = insertelement <4 x i32> %226, i32 %228, i64 1, !dbg !12
  %230 = insertelement <4 x i32> %229, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc750 = insertelement <4 x i32> %230, i32 131072, i64 3, !dbg !12
  %231 = ptrtoint ptr addrspace(3) %221 to i32, !dbg !12
  %dma.buffer.lds751 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %231), !dbg !12
  %232 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds751, i32 %dma.buffer.offset749, <4 x i32> %dma.buffer.rsrc750) #10, !dbg !12
  %233 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local114), !dbg !12
  %234 = zext i32 %233 to i64, !dbg !12
  %235 = inttoptr i64 %234 to ptr addrspace(3), !dbg !12
  %address.local310 = getelementptr [2 x i8], ptr addrspace(1) %44, i64 %address.local119, !dbg !12
  %236 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %237 = ptrtoint ptr addrspace(1) %address.local310 to i64, !dbg !12
  %238 = sub i64 %237, %236, !dbg !12
  %dma.buffer.offset746 = trunc i64 %238 to i32, !dbg !12
  %239 = trunc i64 %236 to i32, !dbg !12
  %240 = insertelement <4 x i32> poison, i32 %239, i64 0, !dbg !12
  %241 = lshr i64 %236, 32, !dbg !12
  %242 = trunc i64 %241 to i32, !dbg !12
  %243 = insertelement <4 x i32> %240, i32 %242, i64 1, !dbg !12
  %244 = insertelement <4 x i32> %243, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc747 = insertelement <4 x i32> %244, i32 131072, i64 3, !dbg !12
  %245 = ptrtoint ptr addrspace(3) %235 to i32, !dbg !12
  %dma.buffer.lds748 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %245), !dbg !12
  %246 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds748, i32 %dma.buffer.offset746, <4 x i32> %dma.buffer.rsrc747) #10, !dbg !12
  %247 = add i64 %141, 4096, !dbg !16
  %248 = icmp slt i64 %247, 0, !dbg !16
  %249 = sub i64 -4097, %141, !dbg !16
  %250 = select i1 %248, i64 %249, i64 %247, !dbg !16
  %251 = sdiv i64 %250, 8192, !dbg !16
  %.lobit4 = ashr i64 %247, 63, !dbg !16
  %252 = xor i64 %251, %.lobit4, !dbg !16
  %253 = add nsw i64 %148, 64, !dbg !16
  %254 = icmp slt i64 %148, -64, !dbg !16
  %255 = sub nsw i64 -65, %148, !dbg !16
  %256 = select i1 %254, i64 %255, i64 %253, !dbg !16
  %257 = sdiv i64 %256, 128, !dbg !16
  %258 = sext i1 %254 to i64, !dbg !16
  %259 = xor i64 %257, %258, !dbg !16
  %260 = shl nsw i64 %259, 7, !dbg !16
  %261 = sub i64 %42, %260, !dbg !16
  %262 = shl i64 %252, 19, !dbg !16
  %263 = and i64 %262, 16252928, !dbg !16
  %264 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %263, !dbg !16
  %265 = getelementptr [2 x i8], ptr addrspace(1) %264, i64 %88, !dbg !16
  %266 = getelementptr [2 x i8], ptr addrspace(1) %265, i64 %261, !dbg !16
  %267 = getelementptr i8, ptr addrspace(1) %266, i64 128, !dbg !16
  %268 = getelementptr [2 x i8], ptr addrspace(1) %267, i64 %94, !dbg !16
  %269 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local124), !dbg !16
  %270 = zext i32 %269 to i64, !dbg !16
  %271 = inttoptr i64 %270 to ptr addrspace(3), !dbg !16
  %address.local311 = getelementptr [2 x i8], ptr addrspace(1) %268, i64 %address.local129, !dbg !16
  %272 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %273 = ptrtoint ptr addrspace(1) %address.local311 to i64, !dbg !16
  %274 = sub i64 %273, %272, !dbg !16
  %dma.buffer.offset743 = trunc i64 %274 to i32, !dbg !16
  %275 = trunc i64 %272 to i32, !dbg !16
  %276 = insertelement <4 x i32> poison, i32 %275, i64 0, !dbg !16
  %277 = lshr i64 %272, 32, !dbg !16
  %278 = trunc i64 %277 to i32, !dbg !16
  %279 = insertelement <4 x i32> %276, i32 %278, i64 1, !dbg !16
  %280 = insertelement <4 x i32> %279, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc744 = insertelement <4 x i32> %280, i32 131072, i64 3, !dbg !16
  %281 = ptrtoint ptr addrspace(3) %271 to i32, !dbg !16
  %dma.buffer.lds745 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %281), !dbg !16
  %282 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds745, i32 %dma.buffer.offset743, <4 x i32> %dma.buffer.rsrc744) #10, !dbg !16
  %address.local312 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local154, !dbg !13
  %283 = load <4 x half>, ptr addrspace(3) %address.local312, align 2, !dbg !13
  %284 = fmul <4 x half> %283, splat (half 0xH3015), !dbg !13
  %address.local313 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local168, !dbg !13
  %285 = load <4 x half>, ptr addrspace(3) %address.local313, align 2, !dbg !13
  %address.local314 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local182, !dbg !13
  %286 = load <4 x half>, ptr addrspace(3) %address.local314, align 2, !dbg !13
  %287 = fmul <4 x half> %285, splat (half 0xH3015), !dbg !13
  %288 = fmul <4 x half> %286, splat (half 0xH3015), !dbg !13
  %address.local315 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local196, !dbg !13
  %289 = load <4 x half>, ptr addrspace(3) %address.local315, align 2, !dbg !13
  %address.local316 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 20480), i64 %address.local209, !dbg !17
  %290 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !17
  %291 = fmul <4 x half> %289, splat (half 0xH3015), !dbg !13
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %284, <4 x half> %290, <4 x float> %191) #9, !dbg !17
  %address.local317 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 20480), i64 %address.local223, !dbg !17
  %293 = load <4 x half>, ptr addrspace(3) %address.local317, align 2, !dbg !17
  %address.local318 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 20480), i64 %address.local237, !dbg !17
  %294 = load <4 x half>, ptr addrspace(3) %address.local318, align 2, !dbg !17
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %293, <4 x float> %292) #9, !dbg !17
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %284, <4 x half> %294, <4 x float> %194) #9, !dbg !17
  %address.local319 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 20480), i64 %address.local251, !dbg !17
  %297 = load <4 x half>, ptr addrspace(3) %address.local319, align 2, !dbg !17
  %address.local320 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 20480), i64 %address.local265, !dbg !17
  %298 = load <4 x half>, ptr addrspace(3) %address.local320, align 2, !dbg !17
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %297, <4 x float> %296) #9, !dbg !17
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %284, <4 x half> %298, <4 x float> %199) #9, !dbg !17
  %address.local321 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 20480), i64 %address.local279, !dbg !17
  %301 = load <4 x half>, ptr addrspace(3) %address.local321, align 2, !dbg !17
  %address.local322 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 20480), i64 %address.local293, !dbg !17
  %302 = load <4 x half>, ptr addrspace(3) %address.local322, align 2, !dbg !17
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %301, <4 x float> %300) #9, !dbg !17
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %284, <4 x half> %302, <4 x float> %202) #9, !dbg !17
  %address.local323 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 20480), i64 %address.local307, !dbg !17
  %305 = load <4 x half>, ptr addrspace(3) %address.local323, align 2, !dbg !17
  %306 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !17
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %287, <4 x half> %305, <4 x float> %304) #9, !dbg !17
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %288, <4 x half> %306, <4 x float> %207) #9, !dbg !17
  %309 = load <4 x half>, ptr addrspace(3) %address.local317, align 2, !dbg !17
  %310 = load <4 x half>, ptr addrspace(3) %address.local318, align 2, !dbg !17
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %291, <4 x half> %309, <4 x float> %308) #9, !dbg !17
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %288, <4 x half> %310, <4 x float> %210) #9, !dbg !17
  %313 = load <4 x half>, ptr addrspace(3) %address.local319, align 2, !dbg !17
  %314 = load <4 x half>, ptr addrspace(3) %address.local320, align 2, !dbg !17
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %291, <4 x half> %313, <4 x float> %312) #9, !dbg !17
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %288, <4 x half> %314, <4 x float> %215) #9, !dbg !17
  %317 = load <4 x half>, ptr addrspace(3) %address.local321, align 2, !dbg !17
  %318 = load <4 x half>, ptr addrspace(3) %address.local322, align 2, !dbg !17
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %291, <4 x half> %317, <4 x float> %316) #9, !dbg !17
  %320 = load <4 x half>, ptr addrspace(3) %address.local323, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %288, <4 x half> %318, <4 x float> %218) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %291, <4 x half> %320, <4 x float> %321) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %323 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !12
  %324 = zext i32 %323 to i64, !dbg !12
  %325 = inttoptr i64 %324 to ptr addrspace(3), !dbg !12
  %address.local324 = getelementptr [2 x i8], ptr addrspace(1) %46, i64 %address.local107, !dbg !12
  %326 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %327 = ptrtoint ptr addrspace(1) %address.local324 to i64, !dbg !12
  %328 = sub i64 %327, %326, !dbg !12
  %dma.buffer.offset740 = trunc i64 %328 to i32, !dbg !12
  %329 = trunc i64 %326 to i32, !dbg !12
  %330 = insertelement <4 x i32> poison, i32 %329, i64 0, !dbg !12
  %331 = lshr i64 %326, 32, !dbg !12
  %332 = trunc i64 %331 to i32, !dbg !12
  %333 = insertelement <4 x i32> %330, i32 %332, i64 1, !dbg !12
  %334 = insertelement <4 x i32> %333, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc741 = insertelement <4 x i32> %334, i32 131072, i64 3, !dbg !12
  %335 = ptrtoint ptr addrspace(3) %325 to i32, !dbg !12
  %dma.buffer.lds742 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %335), !dbg !12
  %336 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds742, i32 %dma.buffer.offset740, <4 x i32> %dma.buffer.rsrc741) #10, !dbg !12
  %337 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local137), !dbg !12
  %338 = zext i32 %337 to i64, !dbg !12
  %339 = inttoptr i64 %338 to ptr addrspace(3), !dbg !12
  %address.local325 = getelementptr [2 x i8], ptr addrspace(1) %46, i64 %address.local119, !dbg !12
  %340 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %341 = ptrtoint ptr addrspace(1) %address.local325 to i64, !dbg !12
  %342 = sub i64 %341, %340, !dbg !12
  %dma.buffer.offset737 = trunc i64 %342 to i32, !dbg !12
  %343 = trunc i64 %340 to i32, !dbg !12
  %344 = insertelement <4 x i32> poison, i32 %343, i64 0, !dbg !12
  %345 = lshr i64 %340, 32, !dbg !12
  %346 = trunc i64 %345 to i32, !dbg !12
  %347 = insertelement <4 x i32> %344, i32 %346, i64 1, !dbg !12
  %348 = insertelement <4 x i32> %347, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc738 = insertelement <4 x i32> %348, i32 131072, i64 3, !dbg !12
  %349 = ptrtoint ptr addrspace(3) %339 to i32, !dbg !12
  %dma.buffer.lds739 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %349), !dbg !12
  %350 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds739, i32 %dma.buffer.offset737, <4 x i32> %dma.buffer.rsrc738) #10, !dbg !12
  %351 = add i64 %141, 6144, !dbg !16
  %352 = icmp slt i64 %351, 0, !dbg !16
  %353 = sub i64 -6145, %141, !dbg !16
  %354 = select i1 %352, i64 %353, i64 %351, !dbg !16
  %355 = sdiv i64 %354, 8192, !dbg !16
  %.lobit5 = ashr i64 %351, 63, !dbg !16
  %356 = xor i64 %355, %.lobit5, !dbg !16
  %357 = add nsw i64 %148, 96, !dbg !16
  %358 = icmp slt i64 %148, -96, !dbg !16
  %359 = sub nsw i64 -97, %148, !dbg !16
  %360 = select i1 %358, i64 %359, i64 %357, !dbg !16
  %361 = sdiv i64 %360, 128, !dbg !16
  %362 = sext i1 %358 to i64, !dbg !16
  %363 = xor i64 %361, %362, !dbg !16
  %364 = shl nsw i64 %363, 7, !dbg !16
  %365 = sub i64 %42, %364, !dbg !16
  %366 = shl i64 %356, 19, !dbg !16
  %367 = and i64 %366, 16252928, !dbg !16
  %368 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %367, !dbg !16
  %369 = getelementptr [2 x i8], ptr addrspace(1) %368, i64 %88, !dbg !16
  %370 = getelementptr [2 x i8], ptr addrspace(1) %369, i64 %365, !dbg !16
  %371 = getelementptr i8, ptr addrspace(1) %370, i64 192, !dbg !16
  %372 = getelementptr [2 x i8], ptr addrspace(1) %371, i64 %94, !dbg !16
  %373 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local141), !dbg !16
  %374 = zext i32 %373 to i64, !dbg !16
  %375 = inttoptr i64 %374 to ptr addrspace(3), !dbg !16
  %address.local326 = getelementptr [2 x i8], ptr addrspace(1) %372, i64 %address.local129, !dbg !16
  %376 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %377 = ptrtoint ptr addrspace(1) %address.local326 to i64, !dbg !16
  %378 = sub i64 %377, %376, !dbg !16
  %dma.buffer.offset734 = trunc i64 %378 to i32, !dbg !16
  %379 = trunc i64 %376 to i32, !dbg !16
  %380 = insertelement <4 x i32> poison, i32 %379, i64 0, !dbg !16
  %381 = lshr i64 %376, 32, !dbg !16
  %382 = trunc i64 %381 to i32, !dbg !16
  %383 = insertelement <4 x i32> %380, i32 %382, i64 1, !dbg !16
  %384 = insertelement <4 x i32> %383, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc735 = insertelement <4 x i32> %384, i32 131072, i64 3, !dbg !16
  %385 = ptrtoint ptr addrspace(3) %375 to i32, !dbg !16
  %dma.buffer.lds736 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %385), !dbg !16
  %386 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds736, i32 %dma.buffer.offset734, <4 x i32> %dma.buffer.rsrc735) #10, !dbg !16
  %387 = load <4 x half>, ptr addrspace(3) %address.local155, align 2, !dbg !13
  %388 = fmul <4 x half> %387, splat (half 0xH3015), !dbg !13
  %389 = load <4 x half>, ptr addrspace(3) %address.local169, align 2, !dbg !13
  %390 = load <4 x half>, ptr addrspace(3) %address.local183, align 2, !dbg !13
  %391 = fmul <4 x half> %389, splat (half 0xH3015), !dbg !13
  %392 = fmul <4 x half> %390, splat (half 0xH3015), !dbg !13
  %393 = load <4 x half>, ptr addrspace(3) %address.local197, align 2, !dbg !13
  %394 = load <4 x half>, ptr addrspace(3) %address.local210, align 2, !dbg !17
  %395 = fmul <4 x half> %393, splat (half 0xH3015), !dbg !13
  %396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %388, <4 x half> %394, <4 x float> %295) #9, !dbg !17
  %397 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !17
  %398 = load <4 x half>, ptr addrspace(3) %address.local238, align 2, !dbg !17
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %391, <4 x half> %397, <4 x float> %396) #9, !dbg !17
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %388, <4 x half> %398, <4 x float> %299) #9, !dbg !17
  %401 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !17
  %402 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !17
  %403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %391, <4 x half> %401, <4 x float> %400) #9, !dbg !17
  %404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %388, <4 x half> %402, <4 x float> %303) #9, !dbg !17
  %405 = load <4 x half>, ptr addrspace(3) %address.local280, align 2, !dbg !17
  %406 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !17
  %407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %391, <4 x half> %405, <4 x float> %404) #9, !dbg !17
  %408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %388, <4 x half> %406, <4 x float> %307) #9, !dbg !17
  %409 = load <4 x half>, ptr addrspace(3) %address.local308, align 2, !dbg !17
  %410 = load <4 x half>, ptr addrspace(3) %address.local210, align 2, !dbg !17
  %411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %391, <4 x half> %409, <4 x float> %408) #9, !dbg !17
  %412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %392, <4 x half> %410, <4 x float> %311) #9, !dbg !17
  %413 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !17
  %414 = load <4 x half>, ptr addrspace(3) %address.local238, align 2, !dbg !17
  %415 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %395, <4 x half> %413, <4 x float> %412) #9, !dbg !17
  %416 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %392, <4 x half> %414, <4 x float> %315) #9, !dbg !17
  %417 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !17
  %418 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !17
  %419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %395, <4 x half> %417, <4 x float> %416) #9, !dbg !17
  %420 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %392, <4 x half> %418, <4 x float> %319) #9, !dbg !17
  %421 = load <4 x half>, ptr addrspace(3) %address.local280, align 2, !dbg !17
  %422 = load <4 x half>, ptr addrspace(3) %address.local294, align 2, !dbg !17
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %395, <4 x half> %421, <4 x float> %420) #9, !dbg !17
  %424 = load <4 x half>, ptr addrspace(3) %address.local308, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %392, <4 x half> %422, <4 x float> %322) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %395, <4 x half> %424, <4 x float> %425) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %427 = add nsw i64 %88, %8, !dbg !18
  %428 = shl i64 %427, 19, !dbg !18
  %429 = and i64 %428, 16252928, !dbg !18
  %430 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %429, !dbg !18
  %431 = getelementptr [2 x i8], ptr addrspace(1) %430, i64 %94, !dbg !18
  %address.local327 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local121, !dbg !18
  %address.local328 = ptrtoint ptr addrspace(3) %address.local327 to i64, !dbg !18
  %address.local329 = trunc i64 %address.local328 to i32, !dbg !18
  %432 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local329), !dbg !18
  %433 = zext i32 %432 to i64, !dbg !18
  %434 = inttoptr i64 %433 to ptr addrspace(3), !dbg !18
  %address.local330 = shl nuw nsw i64 %address.local, 5, !dbg !18
  %address.local331 = and i64 %address.local330, 8064, !dbg !18
  %address.local332 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local333 = and i64 %address.local332, 24, !dbg !18
  %address.local334 = or disjoint i64 %address.local331, %address.local333, !dbg !18
  %address.local335 = getelementptr [2 x i8], ptr addrspace(1) %431, i64 %address.local334, !dbg !18
  %435 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %436 = ptrtoint ptr addrspace(1) %address.local335 to i64, !dbg !18
  %437 = sub i64 %436, %435, !dbg !18
  %dma.buffer.offset731 = trunc i64 %437 to i32, !dbg !18
  %438 = trunc i64 %435 to i32, !dbg !18
  %439 = insertelement <4 x i32> poison, i32 %438, i64 0, !dbg !18
  %440 = lshr i64 %435, 32, !dbg !18
  %441 = trunc i64 %440 to i32, !dbg !18
  %442 = insertelement <4 x i32> %439, i32 %441, i64 1, !dbg !18
  %443 = insertelement <4 x i32> %442, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc732 = insertelement <4 x i32> %443, i32 131072, i64 3, !dbg !18
  %444 = ptrtoint ptr addrspace(3) %434 to i32, !dbg !18
  %dma.buffer.lds733 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %444), !dbg !18
  %445 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds733, i32 %dma.buffer.offset731, <4 x i32> %dma.buffer.rsrc732) #10, !dbg !18
  %446 = load <4 x half>, ptr addrspace(3) %address.local312, align 2, !dbg !13
  %447 = fmul <4 x half> %446, splat (half 0xH3015), !dbg !13
  %448 = load <4 x half>, ptr addrspace(3) %address.local313, align 2, !dbg !13
  %449 = load <4 x half>, ptr addrspace(3) %address.local314, align 2, !dbg !13
  %450 = fmul <4 x half> %448, splat (half 0xH3015), !dbg !13
  %451 = fmul <4 x half> %449, splat (half 0xH3015), !dbg !13
  %452 = load <4 x half>, ptr addrspace(3) %address.local315, align 2, !dbg !13
  %453 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !17
  %454 = fmul <4 x half> %452, splat (half 0xH3015), !dbg !13
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %453, <4 x float> %399) #9, !dbg !17
  %456 = load <4 x half>, ptr addrspace(3) %address.local317, align 2, !dbg !17
  %457 = load <4 x half>, ptr addrspace(3) %address.local318, align 2, !dbg !17
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %456, <4 x float> %455) #9, !dbg !17
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %457, <4 x float> %403) #9, !dbg !17
  %460 = load <4 x half>, ptr addrspace(3) %address.local319, align 2, !dbg !17
  %461 = load <4 x half>, ptr addrspace(3) %address.local320, align 2, !dbg !17
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %460, <4 x float> %459) #9, !dbg !17
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %461, <4 x float> %407) #9, !dbg !17
  %464 = load <4 x half>, ptr addrspace(3) %address.local321, align 2, !dbg !17
  %465 = load <4 x half>, ptr addrspace(3) %address.local322, align 2, !dbg !17
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %464, <4 x float> %463) #9, !dbg !17
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %465, <4 x float> %411) #9, !dbg !17
  %468 = load <4 x half>, ptr addrspace(3) %address.local323, align 2, !dbg !17
  %469 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !17
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %450, <4 x half> %468, <4 x float> %467) #9, !dbg !17
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %451, <4 x half> %469, <4 x float> %415) #9, !dbg !17
  %472 = load <4 x half>, ptr addrspace(3) %address.local317, align 2, !dbg !17
  %473 = load <4 x half>, ptr addrspace(3) %address.local318, align 2, !dbg !17
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %454, <4 x half> %472, <4 x float> %471) #9, !dbg !17
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %451, <4 x half> %473, <4 x float> %419) #9, !dbg !17
  %476 = load <4 x half>, ptr addrspace(3) %address.local319, align 2, !dbg !17
  %477 = load <4 x half>, ptr addrspace(3) %address.local320, align 2, !dbg !17
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %454, <4 x half> %476, <4 x float> %475) #9, !dbg !17
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %451, <4 x half> %477, <4 x float> %423) #9, !dbg !17
  %480 = load <4 x half>, ptr addrspace(3) %address.local321, align 2, !dbg !17
  %481 = load <4 x half>, ptr addrspace(3) %address.local322, align 2, !dbg !17
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %454, <4 x half> %480, <4 x float> %479) #9, !dbg !17
  %483 = load <4 x half>, ptr addrspace(3) %address.local323, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %451, <4 x half> %481, <4 x float> %426) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %485 = or disjoint i64 %57, 63, !dbg !14
  %486 = icmp uge i64 %11, %485, !dbg !14
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %454, <4 x half> %483, <4 x float> %484) #9, !dbg !17
  %488 = and i1 %31, %486, !dbg !14
  br i1 %488, label %489, label %490, !dbg !14

489:                                              ; preds = %56
  br label %570, !dbg !14

490:                                              ; preds = %56
  %491 = add nsw i64 %11, %27, !dbg !14
  %492 = shl nuw nsw i64 %28, 4, !dbg !14
  %493 = add nsw i64 %491, %492, !dbg !14
  %494 = and i32 %26, 240, !dbg !14
  %index.compact = zext nneg i32 %494 to i64, !dbg !14
  %495 = sub nsw i64 %493, %index.compact, !dbg !14
  %496 = and i64 %27, 48, !dbg !14
  %497 = or disjoint i64 %57, %496, !dbg !14
  %498 = add nsw i64 %495, 1, !dbg !24
  %499 = lshr i64 %498, 32, !dbg !25
  %compare.high = trunc i64 %499 to i32, !dbg !25
  %500 = lshr i64 %497, 32, !dbg !25
  %compare.high597 = trunc i64 %500 to i32, !dbg !25
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high597, !dbg !25
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high597, !dbg !25
  %compare.low.base = trunc i64 %498 to i32, !dbg !25
  %compare.low.base598 = trunc i64 %497 to i32, !dbg !25
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base598, !dbg !25
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !25
  %501 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 1) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !25
  %compare.low599 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !25
  %compare.result600 = select i1 %compare.high.equal, i1 %compare.low599, i1 %compare.high.order, !dbg !25
  %502 = select i1 %compare.result600, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base601 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 2) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local602 = or i32 %compare.local.base601, 2, !dbg !25
  %compare.low603 = icmp ugt i32 %compare.low.base, %compare.local602, !dbg !25
  %compare.result604 = select i1 %compare.high.equal, i1 %compare.low603, i1 %compare.high.order, !dbg !25
  %503 = select i1 %compare.result604, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base605 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 3) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local606 = or i32 %compare.local.base605, 3, !dbg !25
  %compare.low607 = icmp ugt i32 %compare.low.base, %compare.local606, !dbg !25
  %compare.result608 = select i1 %compare.high.equal, i1 %compare.low607, i1 %compare.high.order, !dbg !25
  %504 = select i1 %compare.result608, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base609 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 4) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local610 = or i32 %compare.local.base609, 4, !dbg !25
  %compare.low611 = icmp ugt i32 %compare.low.base, %compare.local610, !dbg !25
  %compare.result612 = select i1 %compare.high.equal, i1 %compare.low611, i1 %compare.high.order, !dbg !25
  %505 = select i1 %compare.result612, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base613 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 5) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local614 = or i32 %compare.local.base613, 5, !dbg !25
  %compare.low615 = icmp ugt i32 %compare.low.base, %compare.local614, !dbg !25
  %compare.result616 = select i1 %compare.high.equal, i1 %compare.low615, i1 %compare.high.order, !dbg !25
  %506 = select i1 %compare.result616, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base617 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 6) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local618 = or i32 %compare.local.base617, 6, !dbg !25
  %compare.low619 = icmp ugt i32 %compare.low.base, %compare.local618, !dbg !25
  %compare.result620 = select i1 %compare.high.equal, i1 %compare.low619, i1 %compare.high.order, !dbg !25
  %507 = select i1 %compare.result620, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base621 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 7) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local622 = or i32 %compare.local.base621, 7, !dbg !25
  %compare.low623 = icmp ugt i32 %compare.low.base, %compare.local622, !dbg !25
  %compare.result624 = select i1 %compare.high.equal, i1 %compare.low623, i1 %compare.high.order, !dbg !25
  %508 = select i1 %compare.result624, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base625 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 8) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local626 = or i32 %compare.local.base625, 8, !dbg !25
  %compare.low627 = icmp ugt i32 %compare.low.base, %compare.local626, !dbg !25
  %compare.result628 = select i1 %compare.high.equal, i1 %compare.low627, i1 %compare.high.order, !dbg !25
  %509 = select i1 %compare.result628, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base629 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 9) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local630 = or i32 %compare.local.base629, 9, !dbg !25
  %compare.low631 = icmp ugt i32 %compare.low.base, %compare.local630, !dbg !25
  %compare.result632 = select i1 %compare.high.equal, i1 %compare.low631, i1 %compare.high.order, !dbg !25
  %510 = select i1 %compare.result632, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base633 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 10) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local634 = or i32 %compare.local.base633, 10, !dbg !25
  %compare.low635 = icmp ugt i32 %compare.low.base, %compare.local634, !dbg !25
  %compare.result636 = select i1 %compare.high.equal, i1 %compare.low635, i1 %compare.high.order, !dbg !25
  %511 = select i1 %compare.result636, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base637 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 11) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local638 = or i32 %compare.local.base637, 11, !dbg !25
  %compare.low639 = icmp ugt i32 %compare.low.base, %compare.local638, !dbg !25
  %compare.result640 = select i1 %compare.high.equal, i1 %compare.low639, i1 %compare.high.order, !dbg !25
  %512 = select i1 %compare.result640, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base641 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 12) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local642 = or i32 %compare.local.base641, 12, !dbg !25
  %compare.low643 = icmp ugt i32 %compare.low.base, %compare.local642, !dbg !25
  %compare.result644 = select i1 %compare.high.equal, i1 %compare.low643, i1 %compare.high.order, !dbg !25
  %513 = select i1 %compare.result644, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base645 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 13) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local646 = or i32 %compare.local.base645, 13, !dbg !25
  %compare.low647 = icmp ugt i32 %compare.low.base, %compare.local646, !dbg !25
  %compare.result648 = select i1 %compare.high.equal, i1 %compare.low647, i1 %compare.high.order, !dbg !25
  %514 = select i1 %compare.result648, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base649 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 14) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local650 = or i32 %compare.local.base649, 14, !dbg !25
  %compare.low651 = icmp ugt i32 %compare.low.base, %compare.local650, !dbg !25
  %compare.result652 = select i1 %compare.high.equal, i1 %compare.low651, i1 %compare.high.order, !dbg !25
  %515 = select i1 %compare.result652, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base653 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base598, i32 15) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local654 = or i32 %compare.local.base653, 15, !dbg !25
  %compare.low655 = icmp ugt i32 %compare.low.base, %compare.local654, !dbg !25
  %compare.result656 = select i1 %compare.high.equal, i1 %compare.low655, i1 %compare.high.order, !dbg !25
  %516 = select i1 %compare.result656, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %517 = add nsw i64 %495, 65, !dbg !24
  %518 = lshr i64 %517, 32, !dbg !25
  %compare.high657 = trunc i64 %518 to i32, !dbg !25
  %519 = lshr i64 %497, 32, !dbg !25
  %compare.high658 = trunc i64 %519 to i32, !dbg !25
  %compare.high.equal659 = icmp eq i32 %compare.high657, %compare.high658, !dbg !25
  %compare.high.order660 = icmp ugt i32 %compare.high657, %compare.high658, !dbg !25
  %compare.low.base661 = trunc i64 %517 to i32, !dbg !25
  %compare.low.base662 = trunc i64 %497 to i32, !dbg !25
  %compare.low663 = icmp ugt i32 %compare.low.base661, %compare.low.base662, !dbg !25
  %compare.result664 = select i1 %compare.high.equal659, i1 %compare.low663, i1 %compare.high.order660, !dbg !25
  %520 = select i1 %compare.result664, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base665 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 16) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local666 = or i32 %compare.local.base665, 1, !dbg !25
  %compare.low667 = icmp ugt i32 %compare.low.base661, %compare.local666, !dbg !25
  %compare.result668 = select i1 %compare.high.equal659, i1 %compare.low667, i1 %compare.high.order660, !dbg !25
  %521 = select i1 %compare.result668, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base669 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 17) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local670 = or i32 %compare.local.base669, 2, !dbg !25
  %compare.low671 = icmp ugt i32 %compare.low.base661, %compare.local670, !dbg !25
  %compare.result672 = select i1 %compare.high.equal659, i1 %compare.low671, i1 %compare.high.order660, !dbg !25
  %522 = select i1 %compare.result672, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base673 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 18) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local674 = or i32 %compare.local.base673, 3, !dbg !25
  %compare.low675 = icmp ugt i32 %compare.low.base661, %compare.local674, !dbg !25
  %compare.result676 = select i1 %compare.high.equal659, i1 %compare.low675, i1 %compare.high.order660, !dbg !25
  %523 = select i1 %compare.result676, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base677 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 19) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local678 = or i32 %compare.local.base677, 4, !dbg !25
  %compare.low679 = icmp ugt i32 %compare.low.base661, %compare.local678, !dbg !25
  %compare.result680 = select i1 %compare.high.equal659, i1 %compare.low679, i1 %compare.high.order660, !dbg !25
  %524 = select i1 %compare.result680, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base681 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 20) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local682 = or i32 %compare.local.base681, 5, !dbg !25
  %compare.low683 = icmp ugt i32 %compare.low.base661, %compare.local682, !dbg !25
  %compare.result684 = select i1 %compare.high.equal659, i1 %compare.low683, i1 %compare.high.order660, !dbg !25
  %525 = select i1 %compare.result684, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base685 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 21) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local686 = or i32 %compare.local.base685, 6, !dbg !25
  %compare.low687 = icmp ugt i32 %compare.low.base661, %compare.local686, !dbg !25
  %compare.result688 = select i1 %compare.high.equal659, i1 %compare.low687, i1 %compare.high.order660, !dbg !25
  %526 = select i1 %compare.result688, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base689 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 22) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local690 = or i32 %compare.local.base689, 7, !dbg !25
  %compare.low691 = icmp ugt i32 %compare.low.base661, %compare.local690, !dbg !25
  %compare.result692 = select i1 %compare.high.equal659, i1 %compare.low691, i1 %compare.high.order660, !dbg !25
  %527 = select i1 %compare.result692, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base693 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 23) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local694 = or i32 %compare.local.base693, 8, !dbg !25
  %compare.low695 = icmp ugt i32 %compare.low.base661, %compare.local694, !dbg !25
  %compare.result696 = select i1 %compare.high.equal659, i1 %compare.low695, i1 %compare.high.order660, !dbg !25
  %528 = select i1 %compare.result696, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base697 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 24) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local698 = or i32 %compare.local.base697, 9, !dbg !25
  %compare.low699 = icmp ugt i32 %compare.low.base661, %compare.local698, !dbg !25
  %compare.result700 = select i1 %compare.high.equal659, i1 %compare.low699, i1 %compare.high.order660, !dbg !25
  %529 = select i1 %compare.result700, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base701 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 25) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local702 = or i32 %compare.local.base701, 10, !dbg !25
  %compare.low703 = icmp ugt i32 %compare.low.base661, %compare.local702, !dbg !25
  %compare.result704 = select i1 %compare.high.equal659, i1 %compare.low703, i1 %compare.high.order660, !dbg !25
  %530 = select i1 %compare.result704, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base705 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 26) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local706 = or i32 %compare.local.base705, 11, !dbg !25
  %compare.low707 = icmp ugt i32 %compare.low.base661, %compare.local706, !dbg !25
  %compare.result708 = select i1 %compare.high.equal659, i1 %compare.low707, i1 %compare.high.order660, !dbg !25
  %531 = select i1 %compare.result708, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base709 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 27) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local710 = or i32 %compare.local.base709, 12, !dbg !25
  %compare.low711 = icmp ugt i32 %compare.low.base661, %compare.local710, !dbg !25
  %compare.result712 = select i1 %compare.high.equal659, i1 %compare.low711, i1 %compare.high.order660, !dbg !25
  %532 = select i1 %compare.result712, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base713 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 28) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local714 = or i32 %compare.local.base713, 13, !dbg !25
  %compare.low715 = icmp ugt i32 %compare.low.base661, %compare.local714, !dbg !25
  %compare.result716 = select i1 %compare.high.equal659, i1 %compare.low715, i1 %compare.high.order660, !dbg !25
  %533 = select i1 %compare.result716, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base717 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 29) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local718 = or i32 %compare.local.base717, 14, !dbg !25
  %compare.low719 = icmp ugt i32 %compare.low.base661, %compare.local718, !dbg !25
  %compare.result720 = select i1 %compare.high.equal659, i1 %compare.low719, i1 %compare.high.order660, !dbg !25
  %534 = select i1 %compare.result720, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base721 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base662, i32 30) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local722 = or i32 %compare.local.base721, 15, !dbg !25
  %compare.low723 = icmp ugt i32 %compare.low.base661, %compare.local722, !dbg !25
  %compare.result724 = select i1 %compare.high.equal659, i1 %compare.low723, i1 %compare.high.order660, !dbg !25
  %535 = select i1 %compare.result724, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %536 = insertelement <16 x float> poison, float %501, i64 0, !dbg !14
  %537 = insertelement <16 x float> %536, float %502, i64 1, !dbg !14
  %538 = insertelement <16 x float> %537, float %503, i64 2, !dbg !14
  %539 = insertelement <16 x float> %538, float %504, i64 3, !dbg !14
  %540 = insertelement <16 x float> %539, float %505, i64 4, !dbg !14
  %541 = insertelement <16 x float> %540, float %506, i64 5, !dbg !14
  %542 = insertelement <16 x float> %541, float %507, i64 6, !dbg !14
  %543 = insertelement <16 x float> %542, float %508, i64 7, !dbg !14
  %544 = insertelement <16 x float> %543, float %509, i64 8, !dbg !14
  %545 = insertelement <16 x float> %544, float %510, i64 9, !dbg !14
  %546 = insertelement <16 x float> %545, float %511, i64 10, !dbg !14
  %547 = insertelement <16 x float> %546, float %512, i64 11, !dbg !14
  %548 = insertelement <16 x float> %547, float %513, i64 12, !dbg !14
  %549 = insertelement <16 x float> %548, float %514, i64 13, !dbg !14
  %550 = insertelement <16 x float> %549, float %515, i64 14, !dbg !14
  %551 = insertelement <16 x float> %550, float %516, i64 15, !dbg !14
  %552 = insertvalue [2 x <16 x float>] poison, <16 x float> %551, 0, !dbg !14
  %553 = insertelement <16 x float> poison, float %520, i64 0, !dbg !14
  %554 = insertelement <16 x float> %553, float %521, i64 1, !dbg !14
  %555 = insertelement <16 x float> %554, float %522, i64 2, !dbg !14
  %556 = insertelement <16 x float> %555, float %523, i64 3, !dbg !14
  %557 = insertelement <16 x float> %556, float %524, i64 4, !dbg !14
  %558 = insertelement <16 x float> %557, float %525, i64 5, !dbg !14
  %559 = insertelement <16 x float> %558, float %526, i64 6, !dbg !14
  %560 = insertelement <16 x float> %559, float %527, i64 7, !dbg !14
  %561 = insertelement <16 x float> %560, float %528, i64 8, !dbg !14
  %562 = insertelement <16 x float> %561, float %529, i64 9, !dbg !14
  %563 = insertelement <16 x float> %562, float %530, i64 10, !dbg !14
  %564 = insertelement <16 x float> %563, float %531, i64 11, !dbg !14
  %565 = insertelement <16 x float> %564, float %532, i64 12, !dbg !14
  %566 = insertelement <16 x float> %565, float %533, i64 13, !dbg !14
  %567 = insertelement <16 x float> %566, float %534, i64 14, !dbg !14
  %568 = insertelement <16 x float> %567, float %535, i64 15, !dbg !14
  %569 = insertvalue [2 x <16 x float>] %552, <16 x float> %568, 1, !dbg !14
  br label %570, !dbg !14

570:                                              ; preds = %489, %490
  %571 = phi [2 x <16 x float>] [ %569, %490 ], [ zeroinitializer, %489 ], !dbg !14
  br label %572, !dbg !14

572:                                              ; preds = %570
  %573 = extractvalue [2 x <16 x float>] %571, 0, !dbg !27
  %574 = shufflevector <16 x float> %573, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %575 = fadd <4 x float> %458, %574, !dbg !27
  %576 = extractelement <4 x float> %575, i64 0, !dbg !27
  %577 = extractelement <4 x float> %575, i64 1, !dbg !27
  %578 = extractelement <4 x float> %575, i64 2, !dbg !27
  %579 = extractelement <4 x float> %575, i64 3, !dbg !27
  %580 = shufflevector <16 x float> %573, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %581 = fadd <4 x float> %462, %580, !dbg !27
  %582 = extractelement <4 x float> %581, i64 0, !dbg !27
  %583 = extractelement <4 x float> %581, i64 1, !dbg !27
  %584 = extractelement <4 x float> %581, i64 2, !dbg !27
  %585 = extractelement <4 x float> %581, i64 3, !dbg !27
  %586 = shufflevector <16 x float> %573, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %587 = fadd <4 x float> %466, %586, !dbg !27
  %588 = extractelement <4 x float> %587, i64 0, !dbg !27
  %589 = extractelement <4 x float> %587, i64 1, !dbg !27
  %590 = extractelement <4 x float> %587, i64 2, !dbg !27
  %591 = extractelement <4 x float> %587, i64 3, !dbg !27
  %592 = shufflevector <16 x float> %573, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %593 = fadd <4 x float> %470, %592, !dbg !27
  %594 = extractelement <4 x float> %593, i64 0, !dbg !27
  %595 = extractelement <4 x float> %593, i64 1, !dbg !27
  %596 = extractelement <4 x float> %593, i64 2, !dbg !27
  %597 = extractelement <4 x float> %593, i64 3, !dbg !27
  %598 = extractvalue [2 x <16 x float>] %571, 1, !dbg !27
  %599 = shufflevector <16 x float> %598, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %600 = fadd <4 x float> %474, %599, !dbg !27
  %601 = extractelement <4 x float> %600, i64 0, !dbg !27
  %602 = extractelement <4 x float> %600, i64 1, !dbg !27
  %603 = extractelement <4 x float> %600, i64 2, !dbg !27
  %604 = extractelement <4 x float> %600, i64 3, !dbg !27
  %605 = shufflevector <16 x float> %598, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %606 = fadd <4 x float> %478, %605, !dbg !27
  %607 = extractelement <4 x float> %606, i64 0, !dbg !27
  %608 = extractelement <4 x float> %606, i64 1, !dbg !27
  %609 = extractelement <4 x float> %606, i64 2, !dbg !27
  %610 = extractelement <4 x float> %606, i64 3, !dbg !27
  %611 = shufflevector <16 x float> %598, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %612 = fadd <4 x float> %482, %611, !dbg !27
  %613 = extractelement <4 x float> %612, i64 0, !dbg !27
  %614 = extractelement <4 x float> %612, i64 1, !dbg !27
  %615 = extractelement <4 x float> %612, i64 2, !dbg !27
  %616 = extractelement <4 x float> %612, i64 3, !dbg !27
  %617 = shufflevector <16 x float> %598, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %618 = fadd <4 x float> %487, %617, !dbg !27
  %619 = extractelement <4 x float> %618, i64 0, !dbg !27
  %620 = extractelement <4 x float> %618, i64 1, !dbg !27
  %621 = extractelement <4 x float> %618, i64 2, !dbg !27
  %622 = extractelement <4 x float> %618, i64 3, !dbg !27
  %623 = call float @llvm.maxnum.f32(float %576, float 0xFFF0000000000000), !dbg !28
  %624 = call float @llvm.maxnum.f32(float %623, float %577), !dbg !28
  %625 = call float @llvm.maxnum.f32(float %624, float %578), !dbg !28
  %626 = call float @llvm.maxnum.f32(float %625, float %579), !dbg !28
  %627 = call float @llvm.maxnum.f32(float %626, float %582), !dbg !28
  %628 = call float @llvm.maxnum.f32(float %627, float %583), !dbg !28
  %629 = call float @llvm.maxnum.f32(float %628, float %584), !dbg !28
  %630 = call float @llvm.maxnum.f32(float %629, float %585), !dbg !28
  %631 = call float @llvm.maxnum.f32(float %630, float %588), !dbg !28
  %632 = call float @llvm.maxnum.f32(float %631, float %589), !dbg !28
  %633 = call float @llvm.maxnum.f32(float %632, float %590), !dbg !28
  %634 = call float @llvm.maxnum.f32(float %633, float %591), !dbg !28
  %635 = call float @llvm.maxnum.f32(float %634, float %594), !dbg !28
  %636 = call float @llvm.maxnum.f32(float %635, float %595), !dbg !28
  %637 = call float @llvm.maxnum.f32(float %636, float %596), !dbg !28
  %638 = call float @llvm.maxnum.f32(float %637, float %597), !dbg !28
  %639 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %640 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %639), !dbg !28
  %641 = add nuw nsw i32 %640, 64, !dbg !28
  %642 = xor i32 %640, 16, !dbg !28
  %643 = and i32 %641, 192, !dbg !28
  %644 = icmp samesign ult i32 %642, %643, !dbg !28
  %645 = select i1 %644, i32 %642, i32 %640, !dbg !28
  %646 = shl nuw nsw i32 %645, 2, !dbg !28
  %647 = bitcast float %638 to i32, !dbg !28
  %648 = call i32 @llvm.amdgcn.ds.bpermute(i32 %646, i32 %647), !dbg !28
  %649 = bitcast i32 %648 to float, !dbg !28
  %650 = call float @llvm.maxnum.f32(float %638, float %649), !dbg !28
  %651 = xor i32 %640, 32, !dbg !28
  %652 = icmp samesign ult i32 %651, %643, !dbg !28
  %653 = select i1 %652, i32 %651, i32 %640, !dbg !28
  %654 = shl nuw nsw i32 %653, 2, !dbg !28
  %655 = bitcast float %650 to i32, !dbg !28
  %656 = call i32 @llvm.amdgcn.ds.bpermute(i32 %654, i32 %655), !dbg !28
  %657 = bitcast i32 %656 to float, !dbg !28
  %658 = call float @llvm.maxnum.f32(float %650, float %657), !dbg !28
  %659 = extractvalue [2 x <1 x float>] %51, 0, !dbg !28
  %660 = insertelement <1 x float> poison, float %658, i64 0, !dbg !28
  %661 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %660, <1 x float> %659), !dbg !28
  %662 = call float @llvm.maxnum.f32(float %601, float 0xFFF0000000000000), !dbg !28
  %663 = call float @llvm.maxnum.f32(float %662, float %602), !dbg !28
  %664 = call float @llvm.maxnum.f32(float %663, float %603), !dbg !28
  %665 = call float @llvm.maxnum.f32(float %664, float %604), !dbg !28
  %666 = call float @llvm.maxnum.f32(float %665, float %607), !dbg !28
  %667 = call float @llvm.maxnum.f32(float %666, float %608), !dbg !28
  %668 = call float @llvm.maxnum.f32(float %667, float %609), !dbg !28
  %669 = call float @llvm.maxnum.f32(float %668, float %610), !dbg !28
  %670 = call float @llvm.maxnum.f32(float %669, float %613), !dbg !28
  %671 = call float @llvm.maxnum.f32(float %670, float %614), !dbg !28
  %672 = call float @llvm.maxnum.f32(float %671, float %615), !dbg !28
  %673 = call float @llvm.maxnum.f32(float %672, float %616), !dbg !28
  %674 = call float @llvm.maxnum.f32(float %673, float %619), !dbg !28
  %675 = call float @llvm.maxnum.f32(float %674, float %620), !dbg !28
  %676 = call float @llvm.maxnum.f32(float %675, float %621), !dbg !28
  %677 = call float @llvm.maxnum.f32(float %676, float %622), !dbg !28
  %678 = bitcast float %677 to i32, !dbg !28
  %679 = call i32 @llvm.amdgcn.ds.bpermute(i32 %646, i32 %678), !dbg !28
  %680 = bitcast i32 %679 to float, !dbg !28
  %681 = call float @llvm.maxnum.f32(float %677, float %680), !dbg !28
  %682 = bitcast float %681 to i32, !dbg !28
  %683 = call i32 @llvm.amdgcn.ds.bpermute(i32 %654, i32 %682), !dbg !28
  %684 = bitcast i32 %683 to float, !dbg !28
  %685 = call float @llvm.maxnum.f32(float %681, float %684), !dbg !28
  %686 = extractvalue [2 x <1 x float>] %51, 1, !dbg !28
  %687 = insertelement <1 x float> poison, float %685, i64 0, !dbg !28
  %688 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %687, <1 x float> %686), !dbg !28
  %689 = shufflevector <1 x float> %661, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !29
  %690 = fsub <4 x float> %575, %689, !dbg !29
  %691 = fsub <4 x float> %581, %689, !dbg !29
  %692 = fsub <4 x float> %587, %689, !dbg !29
  %693 = fsub <4 x float> %593, %689, !dbg !29
  %694 = shufflevector <1 x float> %688, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !29
  %695 = fsub <4 x float> %600, %694, !dbg !29
  %696 = fsub <4 x float> %606, %694, !dbg !29
  %697 = fsub <4 x float> %612, %694, !dbg !29
  %698 = fsub <4 x float> %618, %694, !dbg !29
  %699 = fsub <1 x float> %659, %661, !dbg !30
  %700 = extractelement <1 x float> %699, i64 0, !dbg !31
  %701 = call float @llvm.amdgcn.exp2.f32(float %700), !dbg !31
  %702 = insertelement <4 x float> poison, float %701, i64 0, !dbg !36
  %703 = shufflevector <4 x float> %702, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %704 = extractvalue [2 x <1 x float>] %50, 0, !dbg !32
  %705 = extractelement <1 x float> %704, i64 0, !dbg !32
  %706 = fmul float %705, %701, !dbg !32
  %707 = fsub <1 x float> %686, %688, !dbg !30
  %708 = extractelement <1 x float> %707, i64 0, !dbg !31
  %709 = call float @llvm.amdgcn.exp2.f32(float %708), !dbg !31
  %710 = insertelement <4 x float> poison, float %709, i64 0, !dbg !36
  %711 = shufflevector <4 x float> %710, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %712 = extractvalue [2 x <1 x float>] %50, 1, !dbg !32
  %713 = extractelement <1 x float> %712, i64 0, !dbg !32
  %714 = fmul float %713, %709, !dbg !32
  %715 = extractelement <4 x float> %690, i64 0, !dbg !33
  %716 = call float @llvm.amdgcn.exp2.f32(float %715), !dbg !33
  %717 = insertelement <4 x float> poison, float %716, i64 0, !dbg !35
  %718 = extractelement <4 x float> %690, i64 1, !dbg !33
  %719 = call float @llvm.amdgcn.exp2.f32(float %718), !dbg !33
  %720 = insertelement <4 x float> %717, float %719, i64 1, !dbg !35
  %721 = fadd float %716, %719, !dbg !34
  %722 = extractelement <4 x float> %690, i64 2, !dbg !33
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !33
  %724 = insertelement <4 x float> %720, float %723, i64 2, !dbg !35
  %725 = fadd float %721, %723, !dbg !34
  %726 = extractelement <4 x float> %690, i64 3, !dbg !33
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !33
  %728 = insertelement <4 x float> %724, float %727, i64 3, !dbg !35
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %728) #9, !dbg !35, !frisk.register_lifetimes !1
  %729 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !35
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %729) #9, !dbg !34, !frisk.register_lifetimes !1
  %730 = fadd float %725, %727, !dbg !34
  %731 = extractelement <4 x float> %691, i64 0, !dbg !33
  %732 = call float @llvm.amdgcn.exp2.f32(float %731), !dbg !33
  %733 = insertelement <4 x float> poison, float %732, i64 0, !dbg !35
  %734 = fadd float %730, %732, !dbg !34
  %735 = extractelement <4 x float> %691, i64 1, !dbg !33
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !33
  %737 = insertelement <4 x float> %733, float %736, i64 1, !dbg !35
  %738 = fadd float %734, %736, !dbg !34
  %739 = extractelement <4 x float> %691, i64 2, !dbg !33
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !33
  %741 = insertelement <4 x float> %737, float %740, i64 2, !dbg !35
  %742 = fadd float %738, %740, !dbg !34
  %743 = extractelement <4 x float> %691, i64 3, !dbg !33
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !33
  %745 = insertelement <4 x float> %741, float %744, i64 3, !dbg !35
  %math.result84 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %745) #9, !dbg !35, !frisk.register_lifetimes !1
  %746 = fptrunc <4 x float> %math.result84 to <4 x half>, !dbg !35
  %math.packed85 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %746) #9, !dbg !34, !frisk.register_lifetimes !1
  %747 = fadd float %742, %744, !dbg !34
  %748 = extractelement <4 x float> %692, i64 0, !dbg !33
  %749 = call float @llvm.amdgcn.exp2.f32(float %748), !dbg !33
  %750 = insertelement <4 x float> poison, float %749, i64 0, !dbg !35
  %751 = fadd float %747, %749, !dbg !34
  %752 = extractelement <4 x float> %692, i64 1, !dbg !33
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !33
  %754 = insertelement <4 x float> %750, float %753, i64 1, !dbg !35
  %755 = fadd float %751, %753, !dbg !34
  %756 = extractelement <4 x float> %692, i64 2, !dbg !33
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !33
  %758 = insertelement <4 x float> %754, float %757, i64 2, !dbg !35
  %759 = fadd float %755, %757, !dbg !34
  %760 = extractelement <4 x float> %692, i64 3, !dbg !33
  %761 = call float @llvm.amdgcn.exp2.f32(float %760), !dbg !33
  %762 = insertelement <4 x float> %758, float %761, i64 3, !dbg !35
  %math.result86 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %762) #9, !dbg !35, !frisk.register_lifetimes !1
  %763 = fptrunc <4 x float> %math.result86 to <4 x half>, !dbg !35
  %math.packed87 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %763) #9, !dbg !34, !frisk.register_lifetimes !1
  %764 = fadd float %759, %761, !dbg !34
  %765 = extractelement <4 x float> %693, i64 0, !dbg !33
  %766 = call float @llvm.amdgcn.exp2.f32(float %765), !dbg !33
  %767 = insertelement <4 x float> poison, float %766, i64 0, !dbg !35
  %768 = fadd float %764, %766, !dbg !34
  %769 = extractelement <4 x float> %693, i64 1, !dbg !33
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !33
  %771 = insertelement <4 x float> %767, float %770, i64 1, !dbg !35
  %772 = fadd float %768, %770, !dbg !34
  %773 = extractelement <4 x float> %693, i64 2, !dbg !33
  %774 = call float @llvm.amdgcn.exp2.f32(float %773), !dbg !33
  %775 = insertelement <4 x float> %771, float %774, i64 2, !dbg !35
  %776 = fadd float %772, %774, !dbg !34
  %777 = extractelement <4 x float> %693, i64 3, !dbg !33
  %778 = call float @llvm.amdgcn.exp2.f32(float %777), !dbg !33
  %779 = insertelement <4 x float> %775, float %778, i64 3, !dbg !35
  %math.result88 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %779) #9, !dbg !35, !frisk.register_lifetimes !1
  %780 = fptrunc <4 x float> %math.result88 to <4 x half>, !dbg !35
  %math.packed89 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %780) #9, !dbg !34, !frisk.register_lifetimes !1
  %781 = fadd float %776, %778, !dbg !34
  %782 = extractelement <4 x float> %695, i64 0, !dbg !33
  %783 = call float @llvm.amdgcn.exp2.f32(float %782), !dbg !33
  %784 = insertelement <4 x float> poison, float %783, i64 0, !dbg !35
  %785 = extractelement <4 x float> %695, i64 1, !dbg !33
  %786 = call float @llvm.amdgcn.exp2.f32(float %785), !dbg !33
  %787 = insertelement <4 x float> %784, float %786, i64 1, !dbg !35
  %788 = fadd float %783, %786, !dbg !34
  %789 = extractelement <4 x float> %695, i64 2, !dbg !33
  %790 = call float @llvm.amdgcn.exp2.f32(float %789), !dbg !33
  %791 = insertelement <4 x float> %787, float %790, i64 2, !dbg !35
  %792 = fadd float %788, %790, !dbg !34
  %793 = extractelement <4 x float> %695, i64 3, !dbg !33
  %794 = call float @llvm.amdgcn.exp2.f32(float %793), !dbg !33
  %795 = insertelement <4 x float> %791, float %794, i64 3, !dbg !35
  %math.result90 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %795) #9, !dbg !35, !frisk.register_lifetimes !1
  %796 = fptrunc <4 x float> %math.result90 to <4 x half>, !dbg !35
  %math.packed91 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %796) #9, !dbg !34, !frisk.register_lifetimes !1
  %797 = fadd float %792, %794, !dbg !34
  %798 = extractelement <4 x float> %696, i64 0, !dbg !33
  %799 = call float @llvm.amdgcn.exp2.f32(float %798), !dbg !33
  %800 = insertelement <4 x float> poison, float %799, i64 0, !dbg !35
  %801 = fadd float %797, %799, !dbg !34
  %802 = extractelement <4 x float> %696, i64 1, !dbg !33
  %803 = call float @llvm.amdgcn.exp2.f32(float %802), !dbg !33
  %804 = insertelement <4 x float> %800, float %803, i64 1, !dbg !35
  %805 = fadd float %801, %803, !dbg !34
  %806 = extractelement <4 x float> %696, i64 2, !dbg !33
  %807 = call float @llvm.amdgcn.exp2.f32(float %806), !dbg !33
  %808 = insertelement <4 x float> %804, float %807, i64 2, !dbg !35
  %809 = fadd float %805, %807, !dbg !34
  %810 = extractelement <4 x float> %696, i64 3, !dbg !33
  %811 = call float @llvm.amdgcn.exp2.f32(float %810), !dbg !33
  %812 = insertelement <4 x float> %808, float %811, i64 3, !dbg !35
  %math.result92 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %812) #9, !dbg !35, !frisk.register_lifetimes !1
  %813 = fptrunc <4 x float> %math.result92 to <4 x half>, !dbg !35
  %math.packed93 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %813) #9, !dbg !34, !frisk.register_lifetimes !1
  %814 = fadd float %809, %811, !dbg !34
  %815 = extractelement <4 x float> %697, i64 0, !dbg !33
  %816 = call float @llvm.amdgcn.exp2.f32(float %815), !dbg !33
  %817 = insertelement <4 x float> poison, float %816, i64 0, !dbg !35
  %818 = fadd float %814, %816, !dbg !34
  %819 = extractelement <4 x float> %697, i64 1, !dbg !33
  %820 = call float @llvm.amdgcn.exp2.f32(float %819), !dbg !33
  %821 = insertelement <4 x float> %817, float %820, i64 1, !dbg !35
  %822 = fadd float %818, %820, !dbg !34
  %823 = extractelement <4 x float> %697, i64 2, !dbg !33
  %824 = call float @llvm.amdgcn.exp2.f32(float %823), !dbg !33
  %825 = insertelement <4 x float> %821, float %824, i64 2, !dbg !35
  %826 = fadd float %822, %824, !dbg !34
  %827 = extractelement <4 x float> %697, i64 3, !dbg !33
  %828 = call float @llvm.amdgcn.exp2.f32(float %827), !dbg !33
  %829 = insertelement <4 x float> %825, float %828, i64 3, !dbg !35
  %math.result94 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %829) #9, !dbg !35, !frisk.register_lifetimes !1
  %830 = fptrunc <4 x float> %math.result94 to <4 x half>, !dbg !35
  %math.packed95 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %830) #9, !dbg !34, !frisk.register_lifetimes !1
  %831 = fadd float %826, %828, !dbg !34
  %832 = extractelement <4 x float> %698, i64 0, !dbg !33
  %833 = call float @llvm.amdgcn.exp2.f32(float %832), !dbg !33
  %834 = insertelement <4 x float> poison, float %833, i64 0, !dbg !35
  %835 = fadd float %831, %833, !dbg !34
  %836 = extractelement <4 x float> %698, i64 1, !dbg !33
  %837 = call float @llvm.amdgcn.exp2.f32(float %836), !dbg !33
  %838 = insertelement <4 x float> %834, float %837, i64 1, !dbg !35
  %839 = fadd float %835, %837, !dbg !34
  %840 = extractelement <4 x float> %698, i64 2, !dbg !33
  %841 = call float @llvm.amdgcn.exp2.f32(float %840), !dbg !33
  %842 = insertelement <4 x float> %838, float %841, i64 2, !dbg !35
  %843 = fadd float %839, %841, !dbg !34
  %844 = extractelement <4 x float> %698, i64 3, !dbg !33
  %845 = call float @llvm.amdgcn.exp2.f32(float %844), !dbg !33
  %846 = insertelement <4 x float> %842, float %845, i64 3, !dbg !35
  %math.result96 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %846) #9, !dbg !35, !frisk.register_lifetimes !1
  %847 = fptrunc <4 x float> %math.result96 to <4 x half>, !dbg !35
  %math.packed97 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %847) #9, !dbg !34, !frisk.register_lifetimes !1
  %848 = fadd float %843, %845, !dbg !34
  %849 = bitcast float %781 to i32, !dbg !34
  %850 = call i32 @llvm.amdgcn.ds.bpermute(i32 %646, i32 %849), !dbg !34
  %851 = bitcast i32 %850 to float, !dbg !34
  %852 = fadd float %781, %851, !dbg !34
  %853 = bitcast float %852 to i32, !dbg !34
  %854 = call i32 @llvm.amdgcn.ds.bpermute(i32 %654, i32 %853), !dbg !34
  %855 = bitcast i32 %854 to float, !dbg !34
  %856 = fadd float %852, %855, !dbg !34
  %857 = fadd float %856, %706, !dbg !34
  %858 = insertelement <1 x float> poison, float %857, i64 0, !dbg !7
  %859 = insertvalue [2 x <1 x float>] poison, <1 x float> %858, 0, !dbg !7
  %860 = bitcast float %848 to i32, !dbg !34
  %861 = call i32 @llvm.amdgcn.ds.bpermute(i32 %646, i32 %860), !dbg !34
  %862 = bitcast i32 %861 to float, !dbg !34
  %863 = fadd float %848, %862, !dbg !34
  %864 = bitcast float %863 to i32, !dbg !34
  %865 = call i32 @llvm.amdgcn.ds.bpermute(i32 %654, i32 %864), !dbg !34
  %866 = bitcast i32 %865 to float, !dbg !34
  %867 = fadd float %863, %866, !dbg !34
  %868 = fadd float %867, %714, !dbg !34
  %869 = insertelement <1 x float> poison, float %868, i64 0, !dbg !7
  %870 = insertvalue [2 x <1 x float>] %859, <1 x float> %869, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread336 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !15, !range !40, !frisk.register_lifetimes !1
  %address.local337 = zext nneg i32 %address.thread336 to i64, !dbg !13
  %address.local338 = and i64 %address.local337, 60, !dbg !15
  %address.local339 = add nuw nsw i64 %address.local338, %address.local337, !dbg !15
  %address.local340 = shl nuw nsw i64 %address.local339, 3, !dbg !15
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local340, !dbg !15
  %address.local342 = lshr i64 %address.local337, 4, !dbg !15
  %address.local343 = and i64 %address.local342, 3, !dbg !15
  %address.local344 = mul nuw nsw i64 %address.local343, 384, !dbg !15
  %address.local345 = getelementptr [2 x i8], ptr addrspace(3) %address.local341, i64 %address.local344, !dbg !15
  %address.local346 = lshr i64 %address.local337, 2, !dbg !17
  %address.local347 = mul nsw i64 %address.local346, -32, !dbg !15
  %address.local348 = getelementptr [2 x i8], ptr addrspace(3) %address.local345, i64 %address.local347, !dbg !15
  %871 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !15
  %872 = bitcast <4 x i32> %871 to <8 x half>, !dbg !15
  %address.local349 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local340, !dbg !15
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) %address.local349, i64 %address.local344, !dbg !15
  %address.local351 = getelementptr [2 x i8], ptr addrspace(3) %address.local350, i64 %address.local347, !dbg !15
  %address.local352 = getelementptr i8, ptr addrspace(3) %address.local351, i64 256, !dbg !15
  %873 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !15
  %874 = bitcast <4 x i32> %873 to <8 x half>, !dbg !15
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local340, !dbg !15
  %address.local354 = getelementptr [2 x i8], ptr addrspace(3) %address.local353, i64 %address.local344, !dbg !15
  %address.local355 = getelementptr [2 x i8], ptr addrspace(3) %address.local354, i64 %address.local347, !dbg !15
  %address.local356 = getelementptr i8, ptr addrspace(3) %address.local355, i64 512, !dbg !15
  %875 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local356, i32 0) #0, !dbg !15
  %876 = bitcast <4 x i32> %875 to <8 x half>, !dbg !15
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local340, !dbg !15
  %address.local358 = getelementptr [2 x i8], ptr addrspace(3) %address.local357, i64 %address.local344, !dbg !15
  %address.local359 = getelementptr [2 x i8], ptr addrspace(3) %address.local358, i64 %address.local347, !dbg !15
  %address.local360 = getelementptr i8, ptr addrspace(3) %address.local359, i64 768, !dbg !15
  %877 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local360, i32 0) #0, !dbg !15
  %878 = bitcast <4 x i32> %877 to <8 x half>, !dbg !15
  %879 = add nsw i64 %88, %8, !dbg !18
  %880 = shl i64 %879, 19, !dbg !18
  %881 = and i64 %880, 16252928, !dbg !18
  %882 = add i64 %881, %94, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %883 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %882, !dbg !18
  %884 = getelementptr i8, ptr addrspace(1) %883, i64 64, !dbg !18
  %address.local361 = shl nuw nsw i64 %address.local337, 3, !dbg !12
  %address.local362 = and i64 %address.local361, 1536, !dbg !16
  %address.local363 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local362, !dbg !18
  %address.local364 = ptrtoint ptr addrspace(3) %address.local363 to i64, !dbg !18
  %address.local365 = trunc i64 %address.local364 to i32, !dbg !18
  %885 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local365), !dbg !18
  %886 = zext i32 %885 to i64, !dbg !18
  %887 = inttoptr i64 %886 to ptr addrspace(3), !dbg !18
  %address.local366 = shl nuw nsw i64 %address.local337, 5, !dbg !18
  %address.local367 = and i64 %address.local366, 8064, !dbg !18
  %address.local368 = shl nuw nsw i64 %address.local337, 3, !dbg !18
  %address.local369 = and i64 %address.local368, 24, !dbg !18
  %address.local370 = or disjoint i64 %address.local367, %address.local369, !dbg !18
  %address.local371 = getelementptr [2 x i8], ptr addrspace(1) %884, i64 %address.local370, !dbg !18
  %888 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %889 = ptrtoint ptr addrspace(1) %address.local371 to i64, !dbg !18
  %890 = sub i64 %889, %888, !dbg !18
  %dma.buffer.offset728 = trunc i64 %890 to i32, !dbg !18
  %891 = trunc i64 %888 to i32, !dbg !18
  %892 = insertelement <4 x i32> poison, i32 %891, i64 0, !dbg !18
  %893 = lshr i64 %888, 32, !dbg !18
  %894 = trunc i64 %893 to i32, !dbg !18
  %895 = insertelement <4 x i32> %892, i32 %894, i64 1, !dbg !18
  %896 = insertelement <4 x i32> %895, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc729 = insertelement <4 x i32> %896, i32 131072, i64 3, !dbg !18
  %897 = ptrtoint ptr addrspace(3) %887 to i32, !dbg !18
  %dma.buffer.lds730 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %897), !dbg !18
  %898 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds730, i32 %dma.buffer.offset728, <4 x i32> %dma.buffer.rsrc729) #10, !dbg !18
  %899 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %900 = extractvalue [2 x <8 x float>] %49, 0, !dbg !36
  %901 = shufflevector <8 x float> %900, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %902 = fmul <4 x float> %901, %703, !dbg !36
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %899, <4 x float> %902) #9, !dbg !15
  %904 = shufflevector <8 x half> %874, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed85, <4 x half> %904, <4 x float> %903) #9, !dbg !15
  %906 = shufflevector <8 x half> %876, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed87, <4 x half> %906, <4 x float> %905) #9, !dbg !15
  %908 = shufflevector <8 x half> %878, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed89, <4 x half> %908, <4 x float> %907) #9, !dbg !15
  %910 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %911 = shufflevector <8 x float> %900, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %912 = fmul <4 x float> %911, %703, !dbg !36
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %910, <4 x float> %912) #9, !dbg !15
  %914 = shufflevector <8 x half> %874, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed85, <4 x half> %914, <4 x float> %913) #9, !dbg !15
  %916 = shufflevector <8 x half> %876, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed87, <4 x half> %916, <4 x float> %915) #9, !dbg !15
  %918 = shufflevector <8 x half> %878, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed89, <4 x half> %918, <4 x float> %917) #9, !dbg !15
  %920 = extractvalue [2 x <8 x float>] %49, 1, !dbg !36
  %921 = shufflevector <8 x float> %920, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %922 = fmul <4 x float> %921, %711, !dbg !36
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed91, <4 x half> %899, <4 x float> %922) #9, !dbg !15
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed93, <4 x half> %904, <4 x float> %923) #9, !dbg !15
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed95, <4 x half> %906, <4 x float> %924) #9, !dbg !15
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %908, <4 x float> %925) #9, !dbg !15
  %927 = shufflevector <8 x float> %920, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %928 = fmul <4 x float> %927, %711, !dbg !36
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed91, <4 x half> %910, <4 x float> %928) #9, !dbg !15
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed93, <4 x half> %914, <4 x float> %929) #9, !dbg !15
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed95, <4 x half> %916, <4 x float> %930) #9, !dbg !15
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %918, <4 x float> %931) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %933 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !15
  %934 = bitcast <4 x i32> %933 to <8 x half>, !dbg !15
  %935 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !15
  %936 = bitcast <4 x i32> %935 to <8 x half>, !dbg !15
  %937 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local356, i32 0) #0, !dbg !15
  %938 = bitcast <4 x i32> %937 to <8 x half>, !dbg !15
  %939 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local360, i32 0) #0, !dbg !15
  %940 = bitcast <4 x i32> %939 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %941 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %882, !dbg !18
  %942 = getelementptr i8, ptr addrspace(1) %941, i64 128, !dbg !18
  %943 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local365), !dbg !18
  %944 = zext i32 %943 to i64, !dbg !18
  %945 = inttoptr i64 %944 to ptr addrspace(3), !dbg !18
  %address.local372 = getelementptr [2 x i8], ptr addrspace(1) %942, i64 %address.local370, !dbg !18
  %946 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %947 = ptrtoint ptr addrspace(1) %address.local372 to i64, !dbg !18
  %948 = sub i64 %947, %946, !dbg !18
  %dma.buffer.offset725 = trunc i64 %948 to i32, !dbg !18
  %949 = trunc i64 %946 to i32, !dbg !18
  %950 = insertelement <4 x i32> poison, i32 %949, i64 0, !dbg !18
  %951 = lshr i64 %946, 32, !dbg !18
  %952 = trunc i64 %951 to i32, !dbg !18
  %953 = insertelement <4 x i32> %950, i32 %952, i64 1, !dbg !18
  %954 = insertelement <4 x i32> %953, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc726 = insertelement <4 x i32> %954, i32 131072, i64 3, !dbg !18
  %955 = ptrtoint ptr addrspace(3) %945 to i32, !dbg !18
  %dma.buffer.lds727 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %955), !dbg !18
  %956 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds727, i32 %dma.buffer.offset725, <4 x i32> %dma.buffer.rsrc726) #10, !dbg !18
  %957 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %958 = extractvalue [2 x <8 x float>] %52, 0, !dbg !36
  %959 = shufflevector <8 x float> %958, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %960 = fmul <4 x float> %959, %703, !dbg !36
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %957, <4 x float> %960) #9, !dbg !15
  %962 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed85, <4 x half> %962, <4 x float> %961) #9, !dbg !15
  %964 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed87, <4 x half> %964, <4 x float> %963) #9, !dbg !15
  %966 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed89, <4 x half> %966, <4 x float> %965) #9, !dbg !15
  %968 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %969 = shufflevector <8 x float> %958, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %970 = fmul <4 x float> %969, %703, !dbg !36
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %968, <4 x float> %970) #9, !dbg !15
  %972 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed85, <4 x half> %972, <4 x float> %971) #9, !dbg !15
  %974 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed87, <4 x half> %974, <4 x float> %973) #9, !dbg !15
  %976 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed89, <4 x half> %976, <4 x float> %975) #9, !dbg !15
  %978 = extractvalue [2 x <8 x float>] %52, 1, !dbg !36
  %979 = shufflevector <8 x float> %978, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %980 = fmul <4 x float> %979, %711, !dbg !36
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed91, <4 x half> %957, <4 x float> %980) #9, !dbg !15
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed93, <4 x half> %962, <4 x float> %981) #9, !dbg !15
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed95, <4 x half> %964, <4 x float> %982) #9, !dbg !15
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %966, <4 x float> %983) #9, !dbg !15
  %985 = shufflevector <8 x float> %978, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %986 = fmul <4 x float> %985, %711, !dbg !36
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed91, <4 x half> %968, <4 x float> %986) #9, !dbg !15
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed93, <4 x half> %972, <4 x float> %987) #9, !dbg !15
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed95, <4 x half> %974, <4 x float> %988) #9, !dbg !15
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %976, <4 x float> %989) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %991 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !15
  %992 = bitcast <4 x i32> %991 to <8 x half>, !dbg !15
  %993 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !15
  %994 = bitcast <4 x i32> %993 to <8 x half>, !dbg !15
  %995 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local356, i32 0) #0, !dbg !15
  %996 = bitcast <4 x i32> %995 to <8 x half>, !dbg !15
  %997 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local360, i32 0) #0, !dbg !15
  %998 = bitcast <4 x i32> %997 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %999 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %882, !dbg !18
  %1000 = getelementptr i8, ptr addrspace(1) %999, i64 192, !dbg !18
  %1001 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local365), !dbg !18
  %1002 = zext i32 %1001 to i64, !dbg !18
  %1003 = inttoptr i64 %1002 to ptr addrspace(3), !dbg !18
  %address.local373 = getelementptr [2 x i8], ptr addrspace(1) %1000, i64 %address.local370, !dbg !18
  %1004 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1005 = ptrtoint ptr addrspace(1) %address.local373 to i64, !dbg !18
  %1006 = sub i64 %1005, %1004, !dbg !18
  %dma.buffer.offset = trunc i64 %1006 to i32, !dbg !18
  %1007 = trunc i64 %1004 to i32, !dbg !18
  %1008 = insertelement <4 x i32> poison, i32 %1007, i64 0, !dbg !18
  %1009 = lshr i64 %1004, 32, !dbg !18
  %1010 = trunc i64 %1009 to i32, !dbg !18
  %1011 = insertelement <4 x i32> %1008, i32 %1010, i64 1, !dbg !18
  %1012 = insertelement <4 x i32> %1011, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1012, i32 131072, i64 3, !dbg !18
  %1013 = ptrtoint ptr addrspace(3) %1003 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1013), !dbg !18
  %1014 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1015 = shufflevector <8 x half> %992, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1016 = extractvalue [2 x <8 x float>] %53, 0, !dbg !36
  %1017 = shufflevector <8 x float> %1016, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1018 = fmul <4 x float> %1017, %703, !dbg !36
  %1019 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1015, <4 x float> %1018) #9, !dbg !15
  %1020 = shufflevector <8 x half> %994, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed85, <4 x half> %1020, <4 x float> %1019) #9, !dbg !15
  %1022 = shufflevector <8 x half> %996, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed87, <4 x half> %1022, <4 x float> %1021) #9, !dbg !15
  %1024 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed89, <4 x half> %1024, <4 x float> %1023) #9, !dbg !15
  %1026 = shufflevector <8 x half> %992, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1027 = shufflevector <8 x float> %1016, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1028 = fmul <4 x float> %1027, %703, !dbg !36
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1026, <4 x float> %1028) #9, !dbg !15
  %1030 = shufflevector <8 x half> %994, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1031 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed85, <4 x half> %1030, <4 x float> %1029) #9, !dbg !15
  %1032 = shufflevector <8 x half> %996, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed87, <4 x half> %1032, <4 x float> %1031) #9, !dbg !15
  %1034 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed89, <4 x half> %1034, <4 x float> %1033) #9, !dbg !15
  %1036 = extractvalue [2 x <8 x float>] %53, 1, !dbg !36
  %1037 = shufflevector <8 x float> %1036, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1038 = fmul <4 x float> %1037, %711, !dbg !36
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed91, <4 x half> %1015, <4 x float> %1038) #9, !dbg !15
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed93, <4 x half> %1020, <4 x float> %1039) #9, !dbg !15
  %1041 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed95, <4 x half> %1022, <4 x float> %1040) #9, !dbg !15
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1024, <4 x float> %1041) #9, !dbg !15
  %1043 = shufflevector <8 x float> %1036, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1044 = fmul <4 x float> %1043, %711, !dbg !36
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed91, <4 x half> %1026, <4 x float> %1044) #9, !dbg !15
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed93, <4 x half> %1030, <4 x float> %1045) #9, !dbg !15
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed95, <4 x half> %1032, <4 x float> %1046) #9, !dbg !15
  %1048 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1034, <4 x float> %1047) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1049 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !15
  %1050 = bitcast <4 x i32> %1049 to <8 x half>, !dbg !15
  %1051 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !15
  %1052 = bitcast <4 x i32> %1051 to <8 x half>, !dbg !15
  %1053 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local356, i32 0) #0, !dbg !15
  %1054 = bitcast <4 x i32> %1053 to <8 x half>, !dbg !15
  %1055 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local360, i32 0) #0, !dbg !15
  %1056 = bitcast <4 x i32> %1055 to <8 x half>, !dbg !15
  %1057 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1058 = extractvalue [2 x <8 x float>] %54, 0, !dbg !36
  %1059 = shufflevector <8 x float> %1058, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1060 = fmul <4 x float> %1059, %703, !dbg !36
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1057, <4 x float> %1060) #9, !dbg !15
  %1062 = shufflevector <8 x half> %1052, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed85, <4 x half> %1062, <4 x float> %1061) #9, !dbg !15
  %1064 = shufflevector <8 x half> %1054, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed87, <4 x half> %1064, <4 x float> %1063) #9, !dbg !15
  %1066 = shufflevector <8 x half> %1056, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed89, <4 x half> %1066, <4 x float> %1065) #9, !dbg !15
  %1068 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1069 = shufflevector <8 x float> %1058, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1070 = fmul <4 x float> %1069, %703, !dbg !36
  %1071 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1068, <4 x float> %1070) #9, !dbg !15
  %1072 = shufflevector <8 x half> %1052, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed85, <4 x half> %1072, <4 x float> %1071) #9, !dbg !15
  %1074 = shufflevector <8 x half> %1054, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed87, <4 x half> %1074, <4 x float> %1073) #9, !dbg !15
  %1076 = shufflevector <8 x half> %1056, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed89, <4 x half> %1076, <4 x float> %1075) #9, !dbg !15
  %1078 = extractvalue [2 x <8 x float>] %54, 1, !dbg !36
  %1079 = shufflevector <8 x float> %1078, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1080 = fmul <4 x float> %1079, %711, !dbg !36
  %1081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed91, <4 x half> %1057, <4 x float> %1080) #9, !dbg !15
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed93, <4 x half> %1062, <4 x float> %1081) #9, !dbg !15
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed95, <4 x half> %1064, <4 x float> %1082) #9, !dbg !15
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1066, <4 x float> %1083) #9, !dbg !15
  %1085 = shufflevector <8 x float> %1078, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1086 = fmul <4 x float> %1085, %711, !dbg !36
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed91, <4 x half> %1068, <4 x float> %1086) #9, !dbg !15
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed93, <4 x half> %1072, <4 x float> %1087) #9, !dbg !15
  %1089 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed95, <4 x half> %1074, <4 x float> %1088) #9, !dbg !15
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1076, <4 x float> %1089) #9, !dbg !15
  %1091 = shufflevector <4 x float> %909, <4 x float> %919, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1092 = insertvalue [2 x <8 x float>] poison, <8 x float> %1091, 0, !dbg !7
  %1093 = shufflevector <4 x float> %926, <4 x float> %932, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1094 = insertvalue [2 x <8 x float>] %1092, <8 x float> %1093, 1, !dbg !7
  %1095 = insertvalue [2 x <1 x float>] poison, <1 x float> %661, 0, !dbg !7
  %1096 = insertvalue [2 x <1 x float>] %1095, <1 x float> %688, 1, !dbg !7
  %1097 = shufflevector <4 x float> %967, <4 x float> %977, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1098 = insertvalue [2 x <8 x float>] poison, <8 x float> %1097, 0, !dbg !7
  %1099 = shufflevector <4 x float> %984, <4 x float> %990, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1100 = insertvalue [2 x <8 x float>] %1098, <8 x float> %1099, 1, !dbg !7
  %1101 = shufflevector <4 x float> %1025, <4 x float> %1035, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1102 = insertvalue [2 x <8 x float>] poison, <8 x float> %1101, 0, !dbg !7
  %1103 = shufflevector <4 x float> %1042, <4 x float> %1048, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1104 = insertvalue [2 x <8 x float>] %1102, <8 x float> %1103, 1, !dbg !7
  %1105 = shufflevector <4 x float> %1067, <4 x float> %1077, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1106 = insertvalue [2 x <8 x float>] poison, <8 x float> %1105, 0, !dbg !7
  %1107 = shufflevector <4 x float> %1084, <4 x float> %1090, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1108 = insertvalue [2 x <8 x float>] %1106, <8 x float> %1107, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1109 = add nuw nsw i64 %48, 1, !dbg !7
  br label %47, !dbg !7

1110:                                             ; preds = %47
  %1111 = and i32 %26, 16, !dbg !7
  %.not = icmp eq i32 %1111, 0, !dbg !7
  %1112 = and i32 %26, 32, !dbg !7
  %.not2 = icmp eq i32 %1112, 0, !dbg !7
  %1113 = extractvalue [2 x <8 x float>] %54, 0, !dbg !7
  %1114 = extractelement <8 x float> %1113, i64 0, !dbg !7
  %1115 = extractelement <8 x float> %1113, i64 1, !dbg !7
  %1116 = extractelement <8 x float> %1113, i64 2, !dbg !7
  %1117 = extractelement <8 x float> %1113, i64 3, !dbg !7
  %1118 = select i1 %.not, float %1115, float %1114, !dbg !7
  %1119 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1120 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1119), !dbg !7
  %1121 = add nuw nsw i32 %1120, 64, !dbg !7
  %1122 = xor i32 %1120, 16, !dbg !7
  %1123 = and i32 %1121, 192, !dbg !7
  %1124 = icmp samesign ult i32 %1122, %1123, !dbg !7
  %1125 = select i1 %1124, i32 %1122, i32 %1120, !dbg !7
  %1126 = shl nuw nsw i32 %1125, 2, !dbg !7
  %1127 = bitcast float %1118 to i32, !dbg !7
  %1128 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1127), !dbg !7
  %1129 = bitcast i32 %1128 to float, !dbg !7
  %1130 = select i1 %.not, float %1114, float %1129, !dbg !7
  %1131 = select i1 %.not, float %1129, float %1115, !dbg !7
  %1132 = select i1 %.not, float %1117, float %1116, !dbg !7
  %1133 = bitcast float %1132 to i32, !dbg !7
  %1134 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1133), !dbg !7
  %1135 = bitcast i32 %1134 to float, !dbg !7
  %1136 = select i1 %.not, float %1116, float %1135, !dbg !7
  %1137 = select i1 %.not, float %1135, float %1117, !dbg !7
  %1138 = select i1 %.not2, float %1136, float %1130, !dbg !7
  %1139 = xor i32 %1120, 32, !dbg !7
  %1140 = icmp samesign ult i32 %1139, %1123, !dbg !7
  %1141 = select i1 %1140, i32 %1139, i32 %1120, !dbg !7
  %1142 = shl nuw nsw i32 %1141, 2, !dbg !7
  %1143 = bitcast float %1138 to i32, !dbg !7
  %1144 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1143), !dbg !7
  %1145 = bitcast i32 %1144 to float, !dbg !7
  %1146 = select i1 %.not2, float %1130, float %1145, !dbg !7
  %1147 = select i1 %.not2, float %1145, float %1136, !dbg !7
  %1148 = select i1 %.not2, float %1137, float %1131, !dbg !7
  %1149 = bitcast float %1148 to i32, !dbg !7
  %1150 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1149), !dbg !7
  %1151 = bitcast i32 %1150 to float, !dbg !7
  %1152 = select i1 %.not2, float %1131, float %1151, !dbg !7
  %1153 = select i1 %.not2, float %1151, float %1137, !dbg !7
  %1154 = extractvalue [2 x <8 x float>] %53, 0, !dbg !7
  %1155 = extractelement <8 x float> %1154, i64 0, !dbg !7
  %1156 = extractelement <8 x float> %1154, i64 1, !dbg !7
  %1157 = extractelement <8 x float> %1154, i64 2, !dbg !7
  %1158 = extractelement <8 x float> %1154, i64 3, !dbg !7
  %1159 = select i1 %.not, float %1156, float %1155, !dbg !7
  %1160 = bitcast float %1159 to i32, !dbg !7
  %1161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1160), !dbg !7
  %1162 = bitcast i32 %1161 to float, !dbg !7
  %1163 = select i1 %.not, float %1155, float %1162, !dbg !7
  %1164 = select i1 %.not, float %1162, float %1156, !dbg !7
  %1165 = select i1 %.not, float %1158, float %1157, !dbg !7
  %1166 = bitcast float %1165 to i32, !dbg !7
  %1167 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1166), !dbg !7
  %1168 = bitcast i32 %1167 to float, !dbg !7
  %1169 = select i1 %.not, float %1157, float %1168, !dbg !7
  %1170 = select i1 %.not, float %1168, float %1158, !dbg !7
  %1171 = select i1 %.not2, float %1169, float %1163, !dbg !7
  %1172 = bitcast float %1171 to i32, !dbg !7
  %1173 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1172), !dbg !7
  %1174 = bitcast i32 %1173 to float, !dbg !7
  %1175 = select i1 %.not2, float %1163, float %1174, !dbg !7
  %1176 = select i1 %.not2, float %1174, float %1169, !dbg !7
  %1177 = select i1 %.not2, float %1170, float %1164, !dbg !7
  %1178 = bitcast float %1177 to i32, !dbg !7
  %1179 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1178), !dbg !7
  %1180 = bitcast i32 %1179 to float, !dbg !7
  %1181 = select i1 %.not2, float %1164, float %1180, !dbg !7
  %1182 = select i1 %.not2, float %1180, float %1170, !dbg !7
  %1183 = extractvalue [2 x <8 x float>] %52, 0, !dbg !7
  %1184 = extractelement <8 x float> %1183, i64 0, !dbg !7
  %1185 = extractelement <8 x float> %1183, i64 1, !dbg !7
  %1186 = extractelement <8 x float> %1183, i64 2, !dbg !7
  %1187 = extractelement <8 x float> %1183, i64 3, !dbg !7
  %1188 = select i1 %.not, float %1185, float %1184, !dbg !7
  %1189 = bitcast float %1188 to i32, !dbg !7
  %1190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1189), !dbg !7
  %1191 = bitcast i32 %1190 to float, !dbg !7
  %1192 = select i1 %.not, float %1184, float %1191, !dbg !7
  %1193 = select i1 %.not, float %1191, float %1185, !dbg !7
  %1194 = select i1 %.not, float %1187, float %1186, !dbg !7
  %1195 = bitcast float %1194 to i32, !dbg !7
  %1196 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1195), !dbg !7
  %1197 = bitcast i32 %1196 to float, !dbg !7
  %1198 = select i1 %.not, float %1186, float %1197, !dbg !7
  %1199 = select i1 %.not, float %1197, float %1187, !dbg !7
  %1200 = select i1 %.not2, float %1198, float %1192, !dbg !7
  %1201 = bitcast float %1200 to i32, !dbg !7
  %1202 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1201), !dbg !7
  %1203 = bitcast i32 %1202 to float, !dbg !7
  %1204 = select i1 %.not2, float %1192, float %1203, !dbg !7
  %1205 = select i1 %.not2, float %1203, float %1198, !dbg !7
  %1206 = select i1 %.not2, float %1199, float %1193, !dbg !7
  %1207 = bitcast float %1206 to i32, !dbg !7
  %1208 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1207), !dbg !7
  %1209 = bitcast i32 %1208 to float, !dbg !7
  %1210 = select i1 %.not2, float %1193, float %1209, !dbg !7
  %1211 = select i1 %.not2, float %1209, float %1199, !dbg !7
  %1212 = extractvalue [2 x <8 x float>] %49, 0, !dbg !7
  %1213 = extractelement <8 x float> %1212, i64 0, !dbg !7
  %1214 = extractelement <8 x float> %1212, i64 1, !dbg !7
  %1215 = extractelement <8 x float> %1212, i64 2, !dbg !7
  %1216 = extractelement <8 x float> %1212, i64 3, !dbg !7
  %1217 = select i1 %.not, float %1214, float %1213, !dbg !7
  %1218 = bitcast float %1217 to i32, !dbg !7
  %1219 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1218), !dbg !7
  %1220 = bitcast i32 %1219 to float, !dbg !7
  %1221 = select i1 %.not, float %1213, float %1220, !dbg !7
  %1222 = select i1 %.not, float %1220, float %1214, !dbg !7
  %1223 = select i1 %.not, float %1216, float %1215, !dbg !7
  %1224 = bitcast float %1223 to i32, !dbg !7
  %1225 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1224), !dbg !7
  %1226 = bitcast i32 %1225 to float, !dbg !7
  %1227 = select i1 %.not, float %1215, float %1226, !dbg !7
  %1228 = select i1 %.not, float %1226, float %1216, !dbg !7
  %1229 = select i1 %.not2, float %1227, float %1221, !dbg !7
  %1230 = bitcast float %1229 to i32, !dbg !7
  %1231 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1230), !dbg !7
  %1232 = select i1 %.not2, float %1228, float %1222, !dbg !7
  %1233 = bitcast float %1232 to i32, !dbg !7
  %1234 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1233), !dbg !7
  %1235 = extractvalue [2 x <1 x float>] %50, 0, !dbg !19
  %1236 = shufflevector <1 x float> %1235, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1237 = extractelement <8 x float> %1113, i64 4, !dbg !7
  %1238 = extractelement <8 x float> %1113, i64 5, !dbg !7
  %1239 = extractelement <8 x float> %1113, i64 6, !dbg !7
  %1240 = extractelement <8 x float> %1113, i64 7, !dbg !7
  %1241 = select i1 %.not, float %1238, float %1237, !dbg !7
  %1242 = bitcast float %1241 to i32, !dbg !7
  %1243 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1242), !dbg !7
  %1244 = bitcast i32 %1243 to float, !dbg !7
  %1245 = select i1 %.not, float %1237, float %1244, !dbg !7
  %1246 = select i1 %.not, float %1244, float %1238, !dbg !7
  %1247 = select i1 %.not, float %1240, float %1239, !dbg !7
  %1248 = bitcast float %1247 to i32, !dbg !7
  %1249 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1248), !dbg !7
  %1250 = bitcast i32 %1249 to float, !dbg !7
  %1251 = select i1 %.not, float %1239, float %1250, !dbg !7
  %1252 = select i1 %.not, float %1250, float %1240, !dbg !7
  %1253 = select i1 %.not2, float %1251, float %1245, !dbg !7
  %1254 = bitcast float %1253 to i32, !dbg !7
  %1255 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1254), !dbg !7
  %1256 = bitcast i32 %1255 to float, !dbg !7
  %1257 = select i1 %.not2, float %1245, float %1256, !dbg !7
  %1258 = select i1 %.not2, float %1256, float %1251, !dbg !7
  %1259 = select i1 %.not2, float %1252, float %1246, !dbg !7
  %1260 = bitcast float %1259 to i32, !dbg !7
  %1261 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1260), !dbg !7
  %1262 = bitcast i32 %1261 to float, !dbg !7
  %1263 = select i1 %.not2, float %1246, float %1262, !dbg !7
  %1264 = select i1 %.not2, float %1262, float %1252, !dbg !7
  %1265 = extractelement <8 x float> %1154, i64 4, !dbg !7
  %1266 = extractelement <8 x float> %1154, i64 5, !dbg !7
  %1267 = extractelement <8 x float> %1154, i64 6, !dbg !7
  %1268 = extractelement <8 x float> %1154, i64 7, !dbg !7
  %1269 = select i1 %.not, float %1266, float %1265, !dbg !7
  %1270 = bitcast float %1269 to i32, !dbg !7
  %1271 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1270), !dbg !7
  %1272 = bitcast i32 %1271 to float, !dbg !7
  %1273 = select i1 %.not, float %1265, float %1272, !dbg !7
  %1274 = select i1 %.not, float %1272, float %1266, !dbg !7
  %1275 = select i1 %.not, float %1268, float %1267, !dbg !7
  %1276 = bitcast float %1275 to i32, !dbg !7
  %1277 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1276), !dbg !7
  %1278 = bitcast i32 %1277 to float, !dbg !7
  %1279 = select i1 %.not, float %1267, float %1278, !dbg !7
  %1280 = select i1 %.not, float %1278, float %1268, !dbg !7
  %1281 = select i1 %.not2, float %1279, float %1273, !dbg !7
  %1282 = bitcast float %1281 to i32, !dbg !7
  %1283 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1282), !dbg !7
  %1284 = bitcast i32 %1283 to float, !dbg !7
  %1285 = select i1 %.not2, float %1273, float %1284, !dbg !7
  %1286 = select i1 %.not2, float %1284, float %1279, !dbg !7
  %1287 = select i1 %.not2, float %1280, float %1274, !dbg !7
  %1288 = bitcast float %1287 to i32, !dbg !7
  %1289 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1288), !dbg !7
  %1290 = bitcast i32 %1289 to float, !dbg !7
  %1291 = select i1 %.not2, float %1274, float %1290, !dbg !7
  %1292 = select i1 %.not2, float %1290, float %1280, !dbg !7
  %1293 = extractelement <8 x float> %1183, i64 4, !dbg !7
  %1294 = extractelement <8 x float> %1183, i64 5, !dbg !7
  %1295 = extractelement <8 x float> %1183, i64 6, !dbg !7
  %1296 = extractelement <8 x float> %1183, i64 7, !dbg !7
  %1297 = select i1 %.not, float %1294, float %1293, !dbg !7
  %1298 = bitcast float %1297 to i32, !dbg !7
  %1299 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1298), !dbg !7
  %1300 = bitcast i32 %1299 to float, !dbg !7
  %1301 = select i1 %.not, float %1293, float %1300, !dbg !7
  %1302 = select i1 %.not, float %1300, float %1294, !dbg !7
  %1303 = select i1 %.not, float %1296, float %1295, !dbg !7
  %1304 = bitcast float %1303 to i32, !dbg !7
  %1305 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1304), !dbg !7
  %1306 = bitcast i32 %1305 to float, !dbg !7
  %1307 = select i1 %.not, float %1295, float %1306, !dbg !7
  %1308 = select i1 %.not, float %1306, float %1296, !dbg !7
  %1309 = select i1 %.not2, float %1307, float %1301, !dbg !7
  %1310 = bitcast float %1309 to i32, !dbg !7
  %1311 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1310), !dbg !7
  %1312 = bitcast i32 %1311 to float, !dbg !7
  %1313 = select i1 %.not2, float %1301, float %1312, !dbg !7
  %1314 = select i1 %.not2, float %1312, float %1307, !dbg !7
  %1315 = select i1 %.not2, float %1308, float %1302, !dbg !7
  %1316 = bitcast float %1315 to i32, !dbg !7
  %1317 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1316), !dbg !7
  %1318 = bitcast i32 %1317 to float, !dbg !7
  %1319 = select i1 %.not2, float %1302, float %1318, !dbg !7
  %1320 = select i1 %.not2, float %1318, float %1308, !dbg !7
  %1321 = extractelement <8 x float> %1212, i64 4, !dbg !7
  %1322 = extractelement <8 x float> %1212, i64 5, !dbg !7
  %1323 = extractelement <8 x float> %1212, i64 6, !dbg !7
  %1324 = extractelement <8 x float> %1212, i64 7, !dbg !7
  %1325 = select i1 %.not, float %1322, float %1321, !dbg !7
  %1326 = bitcast float %1325 to i32, !dbg !7
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1326), !dbg !7
  %1328 = bitcast i32 %1327 to float, !dbg !7
  %1329 = select i1 %.not, float %1321, float %1328, !dbg !7
  %1330 = select i1 %.not, float %1328, float %1322, !dbg !7
  %1331 = select i1 %.not, float %1324, float %1323, !dbg !7
  %1332 = bitcast float %1331 to i32, !dbg !7
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1332), !dbg !7
  %1334 = bitcast i32 %1333 to float, !dbg !7
  %1335 = select i1 %.not, float %1323, float %1334, !dbg !7
  %1336 = select i1 %.not, float %1334, float %1324, !dbg !7
  %1337 = select i1 %.not2, float %1335, float %1329, !dbg !7
  %1338 = bitcast float %1337 to i32, !dbg !7
  %1339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1338), !dbg !7
  %1340 = select i1 %.not2, float %1336, float %1330, !dbg !7
  %1341 = bitcast float %1340 to i32, !dbg !7
  %1342 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1341), !dbg !7
  %1343 = extractvalue [2 x <8 x float>] %54, 1, !dbg !7
  %1344 = extractelement <8 x float> %1343, i64 0, !dbg !7
  %1345 = extractelement <8 x float> %1343, i64 1, !dbg !7
  %1346 = extractelement <8 x float> %1343, i64 2, !dbg !7
  %1347 = extractelement <8 x float> %1343, i64 3, !dbg !7
  %1348 = select i1 %.not, float %1345, float %1344, !dbg !7
  %1349 = bitcast float %1348 to i32, !dbg !7
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1349), !dbg !7
  %1351 = bitcast i32 %1350 to float, !dbg !7
  %1352 = select i1 %.not, float %1344, float %1351, !dbg !7
  %1353 = select i1 %.not, float %1351, float %1345, !dbg !7
  %1354 = select i1 %.not, float %1347, float %1346, !dbg !7
  %1355 = bitcast float %1354 to i32, !dbg !7
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1355), !dbg !7
  %1357 = bitcast i32 %1356 to float, !dbg !7
  %1358 = select i1 %.not, float %1346, float %1357, !dbg !7
  %1359 = select i1 %.not, float %1357, float %1347, !dbg !7
  %1360 = select i1 %.not2, float %1358, float %1352, !dbg !7
  %1361 = bitcast float %1360 to i32, !dbg !7
  %1362 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1361), !dbg !7
  %1363 = bitcast i32 %1362 to float, !dbg !7
  %1364 = select i1 %.not2, float %1352, float %1363, !dbg !7
  %1365 = select i1 %.not2, float %1363, float %1358, !dbg !7
  %1366 = select i1 %.not2, float %1359, float %1353, !dbg !7
  %1367 = bitcast float %1366 to i32, !dbg !7
  %1368 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1367), !dbg !7
  %1369 = bitcast i32 %1368 to float, !dbg !7
  %1370 = select i1 %.not2, float %1353, float %1369, !dbg !7
  %1371 = select i1 %.not2, float %1369, float %1359, !dbg !7
  %1372 = extractvalue [2 x <8 x float>] %53, 1, !dbg !7
  %1373 = extractelement <8 x float> %1372, i64 0, !dbg !7
  %1374 = extractelement <8 x float> %1372, i64 1, !dbg !7
  %1375 = extractelement <8 x float> %1372, i64 2, !dbg !7
  %1376 = extractelement <8 x float> %1372, i64 3, !dbg !7
  %1377 = select i1 %.not, float %1374, float %1373, !dbg !7
  %1378 = bitcast float %1377 to i32, !dbg !7
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1378), !dbg !7
  %1380 = bitcast i32 %1379 to float, !dbg !7
  %1381 = select i1 %.not, float %1373, float %1380, !dbg !7
  %1382 = select i1 %.not, float %1380, float %1374, !dbg !7
  %1383 = select i1 %.not, float %1376, float %1375, !dbg !7
  %1384 = bitcast float %1383 to i32, !dbg !7
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1384), !dbg !7
  %1386 = bitcast i32 %1385 to float, !dbg !7
  %1387 = select i1 %.not, float %1375, float %1386, !dbg !7
  %1388 = select i1 %.not, float %1386, float %1376, !dbg !7
  %1389 = select i1 %.not2, float %1387, float %1381, !dbg !7
  %1390 = bitcast float %1389 to i32, !dbg !7
  %1391 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1390), !dbg !7
  %1392 = bitcast i32 %1391 to float, !dbg !7
  %1393 = select i1 %.not2, float %1381, float %1392, !dbg !7
  %1394 = select i1 %.not2, float %1392, float %1387, !dbg !7
  %1395 = select i1 %.not2, float %1388, float %1382, !dbg !7
  %1396 = bitcast float %1395 to i32, !dbg !7
  %1397 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1396), !dbg !7
  %1398 = bitcast i32 %1397 to float, !dbg !7
  %1399 = select i1 %.not2, float %1382, float %1398, !dbg !7
  %1400 = select i1 %.not2, float %1398, float %1388, !dbg !7
  %1401 = extractvalue [2 x <8 x float>] %52, 1, !dbg !7
  %1402 = extractelement <8 x float> %1401, i64 0, !dbg !7
  %1403 = extractelement <8 x float> %1401, i64 1, !dbg !7
  %1404 = extractelement <8 x float> %1401, i64 2, !dbg !7
  %1405 = extractelement <8 x float> %1401, i64 3, !dbg !7
  %1406 = select i1 %.not, float %1403, float %1402, !dbg !7
  %1407 = bitcast float %1406 to i32, !dbg !7
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1407), !dbg !7
  %1409 = bitcast i32 %1408 to float, !dbg !7
  %1410 = select i1 %.not, float %1402, float %1409, !dbg !7
  %1411 = select i1 %.not, float %1409, float %1403, !dbg !7
  %1412 = select i1 %.not, float %1405, float %1404, !dbg !7
  %1413 = bitcast float %1412 to i32, !dbg !7
  %1414 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1413), !dbg !7
  %1415 = bitcast i32 %1414 to float, !dbg !7
  %1416 = select i1 %.not, float %1404, float %1415, !dbg !7
  %1417 = select i1 %.not, float %1415, float %1405, !dbg !7
  %1418 = select i1 %.not2, float %1416, float %1410, !dbg !7
  %1419 = bitcast float %1418 to i32, !dbg !7
  %1420 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1419), !dbg !7
  %1421 = bitcast i32 %1420 to float, !dbg !7
  %1422 = select i1 %.not2, float %1410, float %1421, !dbg !7
  %1423 = select i1 %.not2, float %1421, float %1416, !dbg !7
  %1424 = select i1 %.not2, float %1417, float %1411, !dbg !7
  %1425 = bitcast float %1424 to i32, !dbg !7
  %1426 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1425), !dbg !7
  %1427 = bitcast i32 %1426 to float, !dbg !7
  %1428 = select i1 %.not2, float %1411, float %1427, !dbg !7
  %1429 = select i1 %.not2, float %1427, float %1417, !dbg !7
  %1430 = extractvalue [2 x <8 x float>] %49, 1, !dbg !7
  %1431 = extractelement <8 x float> %1430, i64 0, !dbg !7
  %1432 = extractelement <8 x float> %1430, i64 1, !dbg !7
  %1433 = extractelement <8 x float> %1430, i64 2, !dbg !7
  %1434 = extractelement <8 x float> %1430, i64 3, !dbg !7
  %1435 = select i1 %.not, float %1432, float %1431, !dbg !7
  %1436 = bitcast float %1435 to i32, !dbg !7
  %1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1436), !dbg !7
  %1438 = bitcast i32 %1437 to float, !dbg !7
  %1439 = select i1 %.not, float %1431, float %1438, !dbg !7
  %1440 = select i1 %.not, float %1438, float %1432, !dbg !7
  %1441 = select i1 %.not, float %1434, float %1433, !dbg !7
  %1442 = bitcast float %1441 to i32, !dbg !7
  %1443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1442), !dbg !7
  %1444 = bitcast i32 %1443 to float, !dbg !7
  %1445 = select i1 %.not, float %1433, float %1444, !dbg !7
  %1446 = select i1 %.not, float %1444, float %1434, !dbg !7
  %1447 = select i1 %.not2, float %1445, float %1439, !dbg !7
  %1448 = bitcast float %1447 to i32, !dbg !7
  %1449 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1448), !dbg !7
  %1450 = select i1 %.not2, float %1446, float %1440, !dbg !7
  %1451 = bitcast float %1450 to i32, !dbg !7
  %1452 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1451), !dbg !7
  %1453 = extractvalue [2 x <1 x float>] %50, 1, !dbg !19
  %1454 = shufflevector <1 x float> %1453, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1455 = extractelement <8 x float> %1343, i64 4, !dbg !7
  %1456 = extractelement <8 x float> %1343, i64 5, !dbg !7
  %1457 = extractelement <8 x float> %1343, i64 6, !dbg !7
  %1458 = extractelement <8 x float> %1343, i64 7, !dbg !7
  %1459 = select i1 %.not, float %1456, float %1455, !dbg !7
  %1460 = bitcast float %1459 to i32, !dbg !7
  %1461 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1460), !dbg !7
  %1462 = bitcast i32 %1461 to float, !dbg !7
  %1463 = select i1 %.not, float %1455, float %1462, !dbg !7
  %1464 = select i1 %.not, float %1462, float %1456, !dbg !7
  %1465 = select i1 %.not, float %1458, float %1457, !dbg !7
  %1466 = bitcast float %1465 to i32, !dbg !7
  %1467 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1466), !dbg !7
  %1468 = bitcast i32 %1467 to float, !dbg !7
  %1469 = select i1 %.not, float %1457, float %1468, !dbg !7
  %1470 = select i1 %.not, float %1468, float %1458, !dbg !7
  %1471 = select i1 %.not2, float %1469, float %1463, !dbg !7
  %1472 = bitcast float %1471 to i32, !dbg !7
  %1473 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1472), !dbg !7
  %1474 = bitcast i32 %1473 to float, !dbg !7
  %1475 = select i1 %.not2, float %1463, float %1474, !dbg !7
  %1476 = select i1 %.not2, float %1474, float %1469, !dbg !7
  %1477 = select i1 %.not2, float %1470, float %1464, !dbg !7
  %1478 = bitcast float %1477 to i32, !dbg !7
  %1479 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1478), !dbg !7
  %1480 = bitcast i32 %1479 to float, !dbg !7
  %1481 = select i1 %.not2, float %1464, float %1480, !dbg !7
  %1482 = select i1 %.not2, float %1480, float %1470, !dbg !7
  %1483 = extractelement <8 x float> %1372, i64 4, !dbg !7
  %1484 = extractelement <8 x float> %1372, i64 5, !dbg !7
  %1485 = extractelement <8 x float> %1372, i64 6, !dbg !7
  %1486 = extractelement <8 x float> %1372, i64 7, !dbg !7
  %1487 = select i1 %.not, float %1484, float %1483, !dbg !7
  %1488 = bitcast float %1487 to i32, !dbg !7
  %1489 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1488), !dbg !7
  %1490 = bitcast i32 %1489 to float, !dbg !7
  %1491 = select i1 %.not, float %1483, float %1490, !dbg !7
  %1492 = select i1 %.not, float %1490, float %1484, !dbg !7
  %1493 = select i1 %.not, float %1486, float %1485, !dbg !7
  %1494 = bitcast float %1493 to i32, !dbg !7
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1494), !dbg !7
  %1496 = bitcast i32 %1495 to float, !dbg !7
  %1497 = select i1 %.not, float %1485, float %1496, !dbg !7
  %1498 = select i1 %.not, float %1496, float %1486, !dbg !7
  %1499 = select i1 %.not2, float %1497, float %1491, !dbg !7
  %1500 = bitcast float %1499 to i32, !dbg !7
  %1501 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1500), !dbg !7
  %1502 = bitcast i32 %1501 to float, !dbg !7
  %1503 = select i1 %.not2, float %1491, float %1502, !dbg !7
  %1504 = select i1 %.not2, float %1502, float %1497, !dbg !7
  %1505 = select i1 %.not2, float %1498, float %1492, !dbg !7
  %1506 = bitcast float %1505 to i32, !dbg !7
  %1507 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1506), !dbg !7
  %1508 = bitcast i32 %1507 to float, !dbg !7
  %1509 = select i1 %.not2, float %1492, float %1508, !dbg !7
  %1510 = select i1 %.not2, float %1508, float %1498, !dbg !7
  %1511 = extractelement <8 x float> %1401, i64 4, !dbg !7
  %1512 = extractelement <8 x float> %1401, i64 5, !dbg !7
  %1513 = extractelement <8 x float> %1401, i64 6, !dbg !7
  %1514 = extractelement <8 x float> %1401, i64 7, !dbg !7
  %1515 = select i1 %.not, float %1512, float %1511, !dbg !7
  %1516 = bitcast float %1515 to i32, !dbg !7
  %1517 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1516), !dbg !7
  %1518 = bitcast i32 %1517 to float, !dbg !7
  %1519 = select i1 %.not, float %1511, float %1518, !dbg !7
  %1520 = select i1 %.not, float %1518, float %1512, !dbg !7
  %1521 = select i1 %.not, float %1514, float %1513, !dbg !7
  %1522 = bitcast float %1521 to i32, !dbg !7
  %1523 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1522), !dbg !7
  %1524 = bitcast i32 %1523 to float, !dbg !7
  %1525 = select i1 %.not, float %1513, float %1524, !dbg !7
  %1526 = select i1 %.not, float %1524, float %1514, !dbg !7
  %1527 = select i1 %.not2, float %1525, float %1519, !dbg !7
  %1528 = bitcast float %1527 to i32, !dbg !7
  %1529 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1528), !dbg !7
  %1530 = bitcast i32 %1529 to float, !dbg !7
  %1531 = select i1 %.not2, float %1519, float %1530, !dbg !7
  %1532 = select i1 %.not2, float %1530, float %1525, !dbg !7
  %1533 = select i1 %.not2, float %1526, float %1520, !dbg !7
  %1534 = bitcast float %1533 to i32, !dbg !7
  %1535 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1534), !dbg !7
  %1536 = bitcast i32 %1535 to float, !dbg !7
  %1537 = select i1 %.not2, float %1520, float %1536, !dbg !7
  %1538 = select i1 %.not2, float %1536, float %1526, !dbg !7
  %1539 = extractelement <8 x float> %1430, i64 4, !dbg !7
  %1540 = extractelement <8 x float> %1430, i64 5, !dbg !7
  %1541 = extractelement <8 x float> %1430, i64 6, !dbg !7
  %1542 = extractelement <8 x float> %1430, i64 7, !dbg !7
  %1543 = select i1 %.not, float %1540, float %1539, !dbg !7
  %1544 = bitcast float %1543 to i32, !dbg !7
  %1545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1544), !dbg !7
  %1546 = bitcast i32 %1545 to float, !dbg !7
  %1547 = select i1 %.not, float %1539, float %1546, !dbg !7
  %1548 = select i1 %.not, float %1546, float %1540, !dbg !7
  %1549 = select i1 %.not, float %1542, float %1541, !dbg !7
  %1550 = bitcast float %1549 to i32, !dbg !7
  %1551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1126, i32 %1550), !dbg !7
  %1552 = bitcast i32 %1551 to float, !dbg !7
  %1553 = select i1 %.not, float %1541, float %1552, !dbg !7
  %1554 = select i1 %.not, float %1552, float %1542, !dbg !7
  %1555 = select i1 %.not2, float %1553, float %1547, !dbg !7
  %1556 = bitcast float %1555 to i32, !dbg !7
  %1557 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1556), !dbg !7
  %1558 = select i1 %.not2, float %1554, float %1548, !dbg !7
  %1559 = bitcast float %1558 to i32, !dbg !7
  %1560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1559), !dbg !7
  br i1 true, label %1561, label %1623, !dbg !21

1561:                                             ; preds = %1110
  %1562 = bitcast i32 %1557 to float, !dbg !7
  %1563 = select i1 %.not2, float %1547, float %1562, !dbg !7
  %1564 = insertelement <4 x float> poison, float %1563, i64 0, !dbg !19
  %1565 = bitcast i32 %1560 to float, !dbg !7
  %1566 = select i1 %.not2, float %1548, float %1565, !dbg !7
  %1567 = insertelement <4 x float> %1564, float %1566, i64 1, !dbg !19
  %1568 = select i1 %.not2, float %1562, float %1553, !dbg !7
  %1569 = insertelement <4 x float> %1567, float %1568, i64 2, !dbg !19
  %1570 = select i1 %.not2, float %1565, float %1554, !dbg !7
  %1571 = insertelement <4 x float> %1569, float %1570, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1454, !dbg !19
  %normalized = fmul <4 x float> %1571, %row.reciprocal, !dbg !19
  %1572 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1573 = extractelement <4 x half> %1572, i64 3, !dbg !20
  %1574 = extractelement <4 x half> %1572, i64 2, !dbg !20
  %1575 = extractelement <4 x half> %1572, i64 1, !dbg !20
  %1576 = extractelement <4 x half> %1572, i64 0, !dbg !20
  %1577 = bitcast i32 %1449 to float, !dbg !7
  %1578 = select i1 %.not2, float %1439, float %1577, !dbg !7
  %1579 = insertelement <4 x float> poison, float %1578, i64 0, !dbg !19
  %1580 = bitcast i32 %1452 to float, !dbg !7
  %1581 = select i1 %.not2, float %1440, float %1580, !dbg !7
  %1582 = insertelement <4 x float> %1579, float %1581, i64 1, !dbg !19
  %1583 = select i1 %.not2, float %1577, float %1445, !dbg !7
  %1584 = insertelement <4 x float> %1582, float %1583, i64 2, !dbg !19
  %1585 = select i1 %.not2, float %1580, float %1446, !dbg !7
  %1586 = insertelement <4 x float> %1584, float %1585, i64 3, !dbg !19
  %normalized38 = fmul <4 x float> %1586, %row.reciprocal, !dbg !19
  %1587 = fptrunc <4 x float> %normalized38 to <4 x half>, !dbg !20
  %1588 = extractelement <4 x half> %1587, i64 3, !dbg !20
  %1589 = extractelement <4 x half> %1587, i64 2, !dbg !20
  %1590 = extractelement <4 x half> %1587, i64 1, !dbg !20
  %1591 = extractelement <4 x half> %1587, i64 0, !dbg !20
  %1592 = bitcast i32 %1339 to float, !dbg !7
  %1593 = select i1 %.not2, float %1329, float %1592, !dbg !7
  %1594 = insertelement <4 x float> poison, float %1593, i64 0, !dbg !19
  %1595 = bitcast i32 %1342 to float, !dbg !7
  %1596 = select i1 %.not2, float %1330, float %1595, !dbg !7
  %1597 = insertelement <4 x float> %1594, float %1596, i64 1, !dbg !19
  %1598 = select i1 %.not2, float %1592, float %1335, !dbg !7
  %1599 = insertelement <4 x float> %1597, float %1598, i64 2, !dbg !19
  %1600 = select i1 %.not2, float %1595, float %1336, !dbg !7
  %1601 = insertelement <4 x float> %1599, float %1600, i64 3, !dbg !19
  %row.reciprocal39 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1236, !dbg !19
  %normalized40 = fmul <4 x float> %1601, %row.reciprocal39, !dbg !19
  %1602 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !20
  %1603 = extractelement <4 x half> %1602, i64 3, !dbg !20
  %1604 = extractelement <4 x half> %1602, i64 2, !dbg !20
  %1605 = extractelement <4 x half> %1602, i64 1, !dbg !20
  %1606 = extractelement <4 x half> %1602, i64 0, !dbg !20
  %1607 = bitcast i32 %1231 to float, !dbg !7
  %1608 = select i1 %.not2, float %1221, float %1607, !dbg !7
  %1609 = insertelement <4 x float> poison, float %1608, i64 0, !dbg !19
  %1610 = bitcast i32 %1234 to float, !dbg !7
  %1611 = select i1 %.not2, float %1222, float %1610, !dbg !7
  %1612 = insertelement <4 x float> %1609, float %1611, i64 1, !dbg !19
  %1613 = select i1 %.not2, float %1607, float %1227, !dbg !7
  %1614 = insertelement <4 x float> %1612, float %1613, i64 2, !dbg !19
  %1615 = select i1 %.not2, float %1610, float %1228, !dbg !7
  %1616 = insertelement <4 x float> %1614, float %1615, i64 3, !dbg !19
  %normalized41 = fmul <4 x float> %1616, %row.reciprocal39, !dbg !19
  %1617 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !20
  %1618 = extractelement <4 x half> %1617, i64 3, !dbg !20
  %1619 = extractelement <4 x half> %1617, i64 2, !dbg !20
  %1620 = extractelement <4 x half> %1617, i64 1, !dbg !20
  %1621 = extractelement <4 x half> %1617, i64 0, !dbg !20
  %1622 = shl nuw nsw i64 %17, 19, !dbg !21
  %address.thread374 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local375 = zext nneg i32 %address.thread374 to i64, !dbg !13
  %address.local376 = add nsw i64 %18, %address.local375, !dbg !21
  %address.local377 = lshr i64 %address.local375, 6, !dbg !13
  %address.local378 = shl nuw nsw i64 %address.local377, 4, !dbg !21
  %address.local379 = add nsw i64 %address.local376, %address.local378, !dbg !21
  %address.local380 = and i64 %address.local375, 240, !dbg !21
  %address.local381 = sub nsw i64 %address.local379, %address.local380, !dbg !21
  %address.local382 = add nsw i64 %address.local381, %19, !dbg !21
  %address.local383 = shl nuw nsw i64 %address.local382, 7, !dbg !21
  %address.local384 = add nuw nsw i64 %1622, %address.local383, !dbg !21
  %address.local385 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local384, !dbg !21
  %address.local386 = lshr i64 %address.local375, 2, !dbg !13
  %address.local387 = and i64 %address.local386, 12, !dbg !13
  %address.local388 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local385, i64 %address.local387, !dbg !21
  store half %1621, ptr addrspace(1) %address.local388, align 2, !dbg !21
  %address.local389 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local384, !dbg !21
  %address.local390 = or disjoint i64 %address.local387, 1, !dbg !21
  %address.local391 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local389, i64 %address.local390, !dbg !21
  store half %1620, ptr addrspace(1) %address.local391, align 2, !dbg !21
  %address.local392 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local384, !dbg !21
  %address.local393 = or disjoint i64 %address.local387, 2, !dbg !21
  %address.local394 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local392, i64 %address.local393, !dbg !21
  store half %1619, ptr addrspace(1) %address.local394, align 2, !dbg !21
  %address.local395 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local384, !dbg !21
  %address.local396 = or disjoint i64 %address.local387, 3, !dbg !21
  %address.local397 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local395, i64 %address.local396, !dbg !21
  store half %1618, ptr addrspace(1) %address.local397, align 2, !dbg !21
  %address.local398 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local384, !dbg !21
  %address.local399 = or disjoint i64 %address.local387, 16, !dbg !21
  %address.local400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local398, i64 %address.local399, !dbg !21
  store half %1606, ptr addrspace(1) %address.local400, align 2, !dbg !21
  %address.local401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local384, !dbg !21
  %address.local402 = or disjoint i64 %address.local387, 17, !dbg !21
  %address.local403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local401, i64 %address.local402, !dbg !21
  store half %1605, ptr addrspace(1) %address.local403, align 2, !dbg !21
  %address.local404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local384, !dbg !21
  %address.local405 = or disjoint i64 %address.local387, 18, !dbg !21
  %address.local406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local404, i64 %address.local405, !dbg !21
  store half %1604, ptr addrspace(1) %address.local406, align 2, !dbg !21
  %address.local407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local384, !dbg !21
  %address.local408 = or disjoint i64 %address.local387, 19, !dbg !21
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local407, i64 %address.local408, !dbg !21
  store half %1603, ptr addrspace(1) %address.local409, align 2, !dbg !21
  %address.local410 = shl nsw i64 %address.local382, 7, !dbg !21
  %address.local411 = add i64 %address.local410, 8192, !dbg !21
  %address.local412 = add nuw nsw i64 %1622, %address.local411, !dbg !21
  %address.local413 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local412, !dbg !21
  %address.local414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local413, i64 %address.local387, !dbg !21
  store half %1591, ptr addrspace(1) %address.local414, align 2, !dbg !21
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local412, !dbg !21
  %address.local416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local415, i64 %address.local390, !dbg !21
  store half %1590, ptr addrspace(1) %address.local416, align 2, !dbg !21
  %address.local417 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local412, !dbg !21
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local417, i64 %address.local393, !dbg !21
  store half %1589, ptr addrspace(1) %address.local418, align 2, !dbg !21
  %address.local419 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local412, !dbg !21
  %address.local420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local419, i64 %address.local396, !dbg !21
  store half %1588, ptr addrspace(1) %address.local420, align 2, !dbg !21
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local412, !dbg !21
  %address.local422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local421, i64 %address.local399, !dbg !21
  store half %1576, ptr addrspace(1) %address.local422, align 2, !dbg !21
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local412, !dbg !21
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local423, i64 %address.local402, !dbg !21
  store half %1575, ptr addrspace(1) %address.local424, align 2, !dbg !21
  %address.local425 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local412, !dbg !21
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local425, i64 %address.local405, !dbg !21
  store half %1574, ptr addrspace(1) %address.local426, align 2, !dbg !21
  %address.local427 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local412, !dbg !21
  %address.local428 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local427, i64 %address.local408, !dbg !21
  store half %1573, ptr addrspace(1) %address.local428, align 2, !dbg !21
  br label %1623, !dbg !21

1623:                                             ; preds = %1561, %1110
  br i1 true, label %1624, label %1662, !dbg !21

1624:                                             ; preds = %1623
  %1625 = insertelement <4 x float> poison, float %1531, i64 0, !dbg !19
  %1626 = insertelement <4 x float> %1625, float %1537, i64 1, !dbg !19
  %1627 = insertelement <4 x float> %1626, float %1532, i64 2, !dbg !19
  %1628 = insertelement <4 x float> %1627, float %1538, i64 3, !dbg !19
  %row.reciprocal42 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1454, !dbg !19
  %normalized43 = fmul <4 x float> %1628, %row.reciprocal42, !dbg !19
  %1629 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !20
  %1630 = extractelement <4 x half> %1629, i64 3, !dbg !20
  %1631 = extractelement <4 x half> %1629, i64 2, !dbg !20
  %1632 = extractelement <4 x half> %1629, i64 1, !dbg !20
  %1633 = extractelement <4 x half> %1629, i64 0, !dbg !20
  %1634 = insertelement <4 x float> poison, float %1422, i64 0, !dbg !19
  %1635 = insertelement <4 x float> %1634, float %1428, i64 1, !dbg !19
  %1636 = insertelement <4 x float> %1635, float %1423, i64 2, !dbg !19
  %1637 = insertelement <4 x float> %1636, float %1429, i64 3, !dbg !19
  %normalized44 = fmul <4 x float> %1637, %row.reciprocal42, !dbg !19
  %1638 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !20
  %1639 = extractelement <4 x half> %1638, i64 3, !dbg !20
  %1640 = extractelement <4 x half> %1638, i64 2, !dbg !20
  %1641 = extractelement <4 x half> %1638, i64 1, !dbg !20
  %1642 = extractelement <4 x half> %1638, i64 0, !dbg !20
  %1643 = insertelement <4 x float> poison, float %1313, i64 0, !dbg !19
  %1644 = insertelement <4 x float> %1643, float %1319, i64 1, !dbg !19
  %1645 = insertelement <4 x float> %1644, float %1314, i64 2, !dbg !19
  %1646 = insertelement <4 x float> %1645, float %1320, i64 3, !dbg !19
  %row.reciprocal45 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1236, !dbg !19
  %normalized46 = fmul <4 x float> %1646, %row.reciprocal45, !dbg !19
  %1647 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !20
  %1648 = extractelement <4 x half> %1647, i64 3, !dbg !20
  %1649 = extractelement <4 x half> %1647, i64 2, !dbg !20
  %1650 = extractelement <4 x half> %1647, i64 1, !dbg !20
  %1651 = extractelement <4 x half> %1647, i64 0, !dbg !20
  %1652 = insertelement <4 x float> poison, float %1204, i64 0, !dbg !19
  %1653 = insertelement <4 x float> %1652, float %1210, i64 1, !dbg !19
  %1654 = insertelement <4 x float> %1653, float %1205, i64 2, !dbg !19
  %1655 = insertelement <4 x float> %1654, float %1211, i64 3, !dbg !19
  %normalized47 = fmul <4 x float> %1655, %row.reciprocal45, !dbg !19
  %1656 = fptrunc <4 x float> %normalized47 to <4 x half>, !dbg !20
  %1657 = extractelement <4 x half> %1656, i64 3, !dbg !20
  %1658 = extractelement <4 x half> %1656, i64 2, !dbg !20
  %1659 = extractelement <4 x half> %1656, i64 1, !dbg !20
  %1660 = extractelement <4 x half> %1656, i64 0, !dbg !20
  %1661 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread429 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local430 = zext nneg i32 %address.thread429 to i64, !dbg !13
  %address.local431 = add nsw i64 %18, %address.local430, !dbg !21
  %address.local432 = lshr i64 %address.local430, 6, !dbg !13
  %address.local433 = shl nuw nsw i64 %address.local432, 4, !dbg !21
  %address.local434 = add nsw i64 %address.local431, %address.local433, !dbg !21
  %address.local435 = and i64 %address.local430, 240, !dbg !21
  %address.local436 = sub nsw i64 %address.local434, %address.local435, !dbg !21
  %address.local437 = add nsw i64 %address.local436, %19, !dbg !21
  %address.local438 = shl nuw nsw i64 %address.local437, 7, !dbg !21
  %address.local439 = add nuw nsw i64 %1661, %address.local438, !dbg !21
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local439, !dbg !21
  %address.local441 = lshr i64 %address.local430, 2, !dbg !13
  %address.local442 = and i64 %address.local441, 12, !dbg !13
  %address.local443 = or disjoint i64 %address.local442, 32, !dbg !21
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local440, i64 %address.local443, !dbg !21
  store half %1660, ptr addrspace(1) %address.local444, align 2, !dbg !21
  %address.local445 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local439, !dbg !21
  %address.local446 = or disjoint i64 %address.local442, 33, !dbg !21
  %address.local447 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local445, i64 %address.local446, !dbg !21
  store half %1659, ptr addrspace(1) %address.local447, align 2, !dbg !21
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local439, !dbg !21
  %address.local449 = or disjoint i64 %address.local442, 34, !dbg !21
  %address.local450 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local448, i64 %address.local449, !dbg !21
  store half %1658, ptr addrspace(1) %address.local450, align 2, !dbg !21
  %address.local451 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local439, !dbg !21
  %address.local452 = or disjoint i64 %address.local442, 35, !dbg !21
  %address.local453 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local451, i64 %address.local452, !dbg !21
  store half %1657, ptr addrspace(1) %address.local453, align 2, !dbg !21
  %address.local454 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local439, !dbg !21
  %address.local455 = or disjoint i64 %address.local442, 48, !dbg !21
  %address.local456 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local454, i64 %address.local455, !dbg !21
  store half %1651, ptr addrspace(1) %address.local456, align 2, !dbg !21
  %address.local457 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local439, !dbg !21
  %address.local458 = or disjoint i64 %address.local442, 49, !dbg !21
  %address.local459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local457, i64 %address.local458, !dbg !21
  store half %1650, ptr addrspace(1) %address.local459, align 2, !dbg !21
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local439, !dbg !21
  %address.local461 = or disjoint i64 %address.local442, 50, !dbg !21
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local461, !dbg !21
  store half %1649, ptr addrspace(1) %address.local462, align 2, !dbg !21
  %address.local463 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local439, !dbg !21
  %address.local464 = or i64 %address.local441, 51, !dbg !21
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local463, i64 %address.local464, !dbg !21
  store half %1648, ptr addrspace(1) %address.local465, align 2, !dbg !21
  %address.local466 = shl nsw i64 %address.local437, 7, !dbg !21
  %address.local467 = add i64 %address.local466, 8192, !dbg !21
  %address.local468 = add nuw nsw i64 %1661, %address.local467, !dbg !21
  %address.local469 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local468, !dbg !21
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local469, i64 %address.local443, !dbg !21
  store half %1642, ptr addrspace(1) %address.local470, align 2, !dbg !21
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local468, !dbg !21
  %address.local472 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local471, i64 %address.local446, !dbg !21
  store half %1641, ptr addrspace(1) %address.local472, align 2, !dbg !21
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local468, !dbg !21
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local473, i64 %address.local449, !dbg !21
  store half %1640, ptr addrspace(1) %address.local474, align 2, !dbg !21
  %address.local475 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local468, !dbg !21
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local475, i64 %address.local452, !dbg !21
  store half %1639, ptr addrspace(1) %address.local476, align 2, !dbg !21
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local468, !dbg !21
  %address.local478 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local477, i64 %address.local455, !dbg !21
  store half %1633, ptr addrspace(1) %address.local478, align 2, !dbg !21
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local468, !dbg !21
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local479, i64 %address.local458, !dbg !21
  store half %1632, ptr addrspace(1) %address.local480, align 2, !dbg !21
  %address.local481 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local468, !dbg !21
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local481, i64 %address.local461, !dbg !21
  store half %1631, ptr addrspace(1) %address.local482, align 2, !dbg !21
  %address.local483 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local468, !dbg !21
  %address.local484 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local483, i64 %address.local464, !dbg !21
  store half %1630, ptr addrspace(1) %address.local484, align 2, !dbg !21
  br label %1662, !dbg !21

1662:                                             ; preds = %1624, %1623
  br i1 true, label %1663, label %1701, !dbg !21

1663:                                             ; preds = %1662
  %1664 = insertelement <4 x float> poison, float %1503, i64 0, !dbg !19
  %1665 = insertelement <4 x float> %1664, float %1509, i64 1, !dbg !19
  %1666 = insertelement <4 x float> %1665, float %1504, i64 2, !dbg !19
  %1667 = insertelement <4 x float> %1666, float %1510, i64 3, !dbg !19
  %row.reciprocal48 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1454, !dbg !19
  %normalized49 = fmul <4 x float> %1667, %row.reciprocal48, !dbg !19
  %1668 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !20
  %1669 = extractelement <4 x half> %1668, i64 3, !dbg !20
  %1670 = extractelement <4 x half> %1668, i64 2, !dbg !20
  %1671 = extractelement <4 x half> %1668, i64 1, !dbg !20
  %1672 = extractelement <4 x half> %1668, i64 0, !dbg !20
  %1673 = insertelement <4 x float> poison, float %1393, i64 0, !dbg !19
  %1674 = insertelement <4 x float> %1673, float %1399, i64 1, !dbg !19
  %1675 = insertelement <4 x float> %1674, float %1394, i64 2, !dbg !19
  %1676 = insertelement <4 x float> %1675, float %1400, i64 3, !dbg !19
  %normalized50 = fmul <4 x float> %1676, %row.reciprocal48, !dbg !19
  %1677 = fptrunc <4 x float> %normalized50 to <4 x half>, !dbg !20
  %1678 = extractelement <4 x half> %1677, i64 3, !dbg !20
  %1679 = extractelement <4 x half> %1677, i64 2, !dbg !20
  %1680 = extractelement <4 x half> %1677, i64 1, !dbg !20
  %1681 = extractelement <4 x half> %1677, i64 0, !dbg !20
  %1682 = insertelement <4 x float> poison, float %1285, i64 0, !dbg !19
  %1683 = insertelement <4 x float> %1682, float %1291, i64 1, !dbg !19
  %1684 = insertelement <4 x float> %1683, float %1286, i64 2, !dbg !19
  %1685 = insertelement <4 x float> %1684, float %1292, i64 3, !dbg !19
  %row.reciprocal51 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1236, !dbg !19
  %normalized52 = fmul <4 x float> %1685, %row.reciprocal51, !dbg !19
  %1686 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !20
  %1687 = extractelement <4 x half> %1686, i64 3, !dbg !20
  %1688 = extractelement <4 x half> %1686, i64 2, !dbg !20
  %1689 = extractelement <4 x half> %1686, i64 1, !dbg !20
  %1690 = extractelement <4 x half> %1686, i64 0, !dbg !20
  %1691 = insertelement <4 x float> poison, float %1175, i64 0, !dbg !19
  %1692 = insertelement <4 x float> %1691, float %1181, i64 1, !dbg !19
  %1693 = insertelement <4 x float> %1692, float %1176, i64 2, !dbg !19
  %1694 = insertelement <4 x float> %1693, float %1182, i64 3, !dbg !19
  %normalized53 = fmul <4 x float> %1694, %row.reciprocal51, !dbg !19
  %1695 = fptrunc <4 x float> %normalized53 to <4 x half>, !dbg !20
  %1696 = extractelement <4 x half> %1695, i64 3, !dbg !20
  %1697 = extractelement <4 x half> %1695, i64 2, !dbg !20
  %1698 = extractelement <4 x half> %1695, i64 1, !dbg !20
  %1699 = extractelement <4 x half> %1695, i64 0, !dbg !20
  %1700 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread485 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local486 = zext nneg i32 %address.thread485 to i64, !dbg !13
  %address.local487 = add nsw i64 %18, %address.local486, !dbg !21
  %address.local488 = lshr i64 %address.local486, 6, !dbg !13
  %address.local489 = shl nuw nsw i64 %address.local488, 4, !dbg !21
  %address.local490 = add nsw i64 %address.local487, %address.local489, !dbg !21
  %address.local491 = and i64 %address.local486, 240, !dbg !21
  %address.local492 = sub nsw i64 %address.local490, %address.local491, !dbg !21
  %address.local493 = add nsw i64 %address.local492, %19, !dbg !21
  %address.local494 = shl nuw nsw i64 %address.local493, 7, !dbg !21
  %address.local495 = add nuw nsw i64 %1700, %address.local494, !dbg !21
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local495, !dbg !21
  %address.local497 = lshr i64 %address.local486, 2, !dbg !13
  %address.local498 = and i64 %address.local497, 12, !dbg !13
  %address.local499 = or disjoint i64 %address.local498, 64, !dbg !21
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local496, i64 %address.local499, !dbg !21
  store half %1699, ptr addrspace(1) %address.local500, align 2, !dbg !21
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local495, !dbg !21
  %address.local502 = or disjoint i64 %address.local498, 65, !dbg !21
  %address.local503 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local501, i64 %address.local502, !dbg !21
  store half %1698, ptr addrspace(1) %address.local503, align 2, !dbg !21
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local495, !dbg !21
  %address.local505 = or disjoint i64 %address.local498, 66, !dbg !21
  %address.local506 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local504, i64 %address.local505, !dbg !21
  store half %1697, ptr addrspace(1) %address.local506, align 2, !dbg !21
  %address.local507 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local495, !dbg !21
  %address.local508 = or disjoint i64 %address.local498, 67, !dbg !21
  %address.local509 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local507, i64 %address.local508, !dbg !21
  store half %1696, ptr addrspace(1) %address.local509, align 2, !dbg !21
  %address.local510 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local495, !dbg !21
  %address.local511 = or disjoint i64 %address.local498, 80, !dbg !21
  %address.local512 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local510, i64 %address.local511, !dbg !21
  store half %1690, ptr addrspace(1) %address.local512, align 2, !dbg !21
  %address.local513 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local495, !dbg !21
  %address.local514 = or disjoint i64 %address.local498, 81, !dbg !21
  %address.local515 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local513, i64 %address.local514, !dbg !21
  store half %1689, ptr addrspace(1) %address.local515, align 2, !dbg !21
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local495, !dbg !21
  %address.local517 = or disjoint i64 %address.local498, 82, !dbg !21
  %address.local518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local516, i64 %address.local517, !dbg !21
  store half %1688, ptr addrspace(1) %address.local518, align 2, !dbg !21
  %address.local519 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local495, !dbg !21
  %address.local520 = or disjoint i64 %address.local498, 83, !dbg !21
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local519, i64 %address.local520, !dbg !21
  store half %1687, ptr addrspace(1) %address.local521, align 2, !dbg !21
  %address.local522 = shl nsw i64 %address.local493, 7, !dbg !21
  %address.local523 = add i64 %address.local522, 8192, !dbg !21
  %address.local524 = add nuw nsw i64 %1700, %address.local523, !dbg !21
  %address.local525 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local524, !dbg !21
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local525, i64 %address.local499, !dbg !21
  store half %1681, ptr addrspace(1) %address.local526, align 2, !dbg !21
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local524, !dbg !21
  %address.local528 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local527, i64 %address.local502, !dbg !21
  store half %1680, ptr addrspace(1) %address.local528, align 2, !dbg !21
  %address.local529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local524, !dbg !21
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local529, i64 %address.local505, !dbg !21
  store half %1679, ptr addrspace(1) %address.local530, align 2, !dbg !21
  %address.local531 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local524, !dbg !21
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local531, i64 %address.local508, !dbg !21
  store half %1678, ptr addrspace(1) %address.local532, align 2, !dbg !21
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local524, !dbg !21
  %address.local534 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local533, i64 %address.local511, !dbg !21
  store half %1672, ptr addrspace(1) %address.local534, align 2, !dbg !21
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local524, !dbg !21
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local535, i64 %address.local514, !dbg !21
  store half %1671, ptr addrspace(1) %address.local536, align 2, !dbg !21
  %address.local537 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local524, !dbg !21
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local537, i64 %address.local517, !dbg !21
  store half %1670, ptr addrspace(1) %address.local538, align 2, !dbg !21
  %address.local539 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local524, !dbg !21
  %address.local540 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local539, i64 %address.local520, !dbg !21
  store half %1669, ptr addrspace(1) %address.local540, align 2, !dbg !21
  br label %1701, !dbg !21

1701:                                             ; preds = %1663, %1662
  br i1 true, label %1702, label %1740, !dbg !21

1702:                                             ; preds = %1701
  %1703 = insertelement <4 x float> poison, float %1475, i64 0, !dbg !19
  %1704 = insertelement <4 x float> %1703, float %1481, i64 1, !dbg !19
  %1705 = insertelement <4 x float> %1704, float %1476, i64 2, !dbg !19
  %1706 = insertelement <4 x float> %1705, float %1482, i64 3, !dbg !19
  %row.reciprocal54 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1454, !dbg !19
  %normalized55 = fmul <4 x float> %1706, %row.reciprocal54, !dbg !19
  %1707 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !20
  %1708 = extractelement <4 x half> %1707, i64 3, !dbg !20
  %1709 = extractelement <4 x half> %1707, i64 2, !dbg !20
  %1710 = extractelement <4 x half> %1707, i64 1, !dbg !20
  %1711 = extractelement <4 x half> %1707, i64 0, !dbg !20
  %1712 = insertelement <4 x float> poison, float %1364, i64 0, !dbg !19
  %1713 = insertelement <4 x float> %1712, float %1370, i64 1, !dbg !19
  %1714 = insertelement <4 x float> %1713, float %1365, i64 2, !dbg !19
  %1715 = insertelement <4 x float> %1714, float %1371, i64 3, !dbg !19
  %normalized56 = fmul <4 x float> %1715, %row.reciprocal54, !dbg !19
  %1716 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !20
  %1717 = extractelement <4 x half> %1716, i64 3, !dbg !20
  %1718 = extractelement <4 x half> %1716, i64 2, !dbg !20
  %1719 = extractelement <4 x half> %1716, i64 1, !dbg !20
  %1720 = extractelement <4 x half> %1716, i64 0, !dbg !20
  %1721 = insertelement <4 x float> poison, float %1257, i64 0, !dbg !19
  %1722 = insertelement <4 x float> %1721, float %1263, i64 1, !dbg !19
  %1723 = insertelement <4 x float> %1722, float %1258, i64 2, !dbg !19
  %1724 = insertelement <4 x float> %1723, float %1264, i64 3, !dbg !19
  %row.reciprocal57 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1236, !dbg !19
  %normalized58 = fmul <4 x float> %1724, %row.reciprocal57, !dbg !19
  %1725 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !20
  %1726 = extractelement <4 x half> %1725, i64 3, !dbg !20
  %1727 = extractelement <4 x half> %1725, i64 2, !dbg !20
  %1728 = extractelement <4 x half> %1725, i64 1, !dbg !20
  %1729 = extractelement <4 x half> %1725, i64 0, !dbg !20
  %1730 = insertelement <4 x float> poison, float %1146, i64 0, !dbg !19
  %1731 = insertelement <4 x float> %1730, float %1152, i64 1, !dbg !19
  %1732 = insertelement <4 x float> %1731, float %1147, i64 2, !dbg !19
  %1733 = insertelement <4 x float> %1732, float %1153, i64 3, !dbg !19
  %normalized59 = fmul <4 x float> %1733, %row.reciprocal57, !dbg !19
  %1734 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !20
  %1735 = extractelement <4 x half> %1734, i64 3, !dbg !20
  %1736 = extractelement <4 x half> %1734, i64 2, !dbg !20
  %1737 = extractelement <4 x half> %1734, i64 1, !dbg !20
  %1738 = extractelement <4 x half> %1734, i64 0, !dbg !20
  %1739 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread541 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local542 = zext nneg i32 %address.thread541 to i64, !dbg !13
  %address.local543 = add nsw i64 %18, %address.local542, !dbg !21
  %address.local544 = lshr i64 %address.local542, 6, !dbg !13
  %address.local545 = shl nuw nsw i64 %address.local544, 4, !dbg !21
  %address.local546 = add nsw i64 %address.local543, %address.local545, !dbg !21
  %address.local547 = and i64 %address.local542, 240, !dbg !21
  %address.local548 = sub nsw i64 %address.local546, %address.local547, !dbg !21
  %address.local549 = add nsw i64 %address.local548, %19, !dbg !21
  %address.local550 = shl nuw nsw i64 %address.local549, 7, !dbg !21
  %address.local551 = add nuw nsw i64 %1739, %address.local550, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local551, !dbg !21
  %address.local553 = lshr i64 %address.local542, 2, !dbg !13
  %address.local554 = and i64 %address.local553, 12, !dbg !13
  %address.local555 = or disjoint i64 %address.local554, 96, !dbg !21
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local552, i64 %address.local555, !dbg !21
  store half %1738, ptr addrspace(1) %address.local556, align 2, !dbg !21
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local551, !dbg !21
  %address.local558 = or disjoint i64 %address.local554, 97, !dbg !21
  %address.local559 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local557, i64 %address.local558, !dbg !21
  store half %1737, ptr addrspace(1) %address.local559, align 2, !dbg !21
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local551, !dbg !21
  %address.local561 = or disjoint i64 %address.local554, 98, !dbg !21
  %address.local562 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local560, i64 %address.local561, !dbg !21
  store half %1736, ptr addrspace(1) %address.local562, align 2, !dbg !21
  %address.local563 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local551, !dbg !21
  %address.local564 = or disjoint i64 %address.local554, 99, !dbg !21
  %address.local565 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local563, i64 %address.local564, !dbg !21
  store half %1735, ptr addrspace(1) %address.local565, align 2, !dbg !21
  %address.local566 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local551, !dbg !21
  %address.local567 = or disjoint i64 %address.local554, 112, !dbg !21
  %address.local568 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local566, i64 %address.local567, !dbg !21
  store half %1729, ptr addrspace(1) %address.local568, align 2, !dbg !21
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local551, !dbg !21
  %address.local570 = or disjoint i64 %address.local554, 113, !dbg !21
  %address.local571 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local569, i64 %address.local570, !dbg !21
  store half %1728, ptr addrspace(1) %address.local571, align 2, !dbg !21
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local551, !dbg !21
  %address.local573 = or disjoint i64 %address.local554, 114, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local572, i64 %address.local573, !dbg !21
  store half %1727, ptr addrspace(1) %address.local574, align 2, !dbg !21
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local551, !dbg !21
  %address.local576 = or i64 %address.local553, 115, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local576, !dbg !21
  store half %1726, ptr addrspace(1) %address.local577, align 2, !dbg !21
  %address.local578 = shl nsw i64 %address.local549, 7, !dbg !21
  %address.local579 = add i64 %address.local578, 8192, !dbg !21
  %address.local580 = add nuw nsw i64 %1739, %address.local579, !dbg !21
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local555, !dbg !21
  store half %1720, ptr addrspace(1) %address.local582, align 2, !dbg !21
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local558, !dbg !21
  store half %1719, ptr addrspace(1) %address.local584, align 2, !dbg !21
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local585, i64 %address.local561, !dbg !21
  store half %1718, ptr addrspace(1) %address.local586, align 2, !dbg !21
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local587, i64 %address.local564, !dbg !21
  store half %1717, ptr addrspace(1) %address.local588, align 2, !dbg !21
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local590 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local589, i64 %address.local567, !dbg !21
  store half %1711, ptr addrspace(1) %address.local590, align 2, !dbg !21
  %address.local591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local591, i64 %address.local570, !dbg !21
  store half %1710, ptr addrspace(1) %address.local592, align 2, !dbg !21
  %address.local593 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local593, i64 %address.local573, !dbg !21
  store half %1709, ptr addrspace(1) %address.local594, align 2, !dbg !21
  %address.local595 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local580, !dbg !21
  %address.local596 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local595, i64 %address.local576, !dbg !21
  store half %1708, ptr addrspace(1) %address.local596, align 2, !dbg !21
  br label %1740, !dbg !21

1740:                                             ; preds = %1702, %1701
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0005-test_input_1x32x4096x128-128-64-32-online/pipeline-loop-only")
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
