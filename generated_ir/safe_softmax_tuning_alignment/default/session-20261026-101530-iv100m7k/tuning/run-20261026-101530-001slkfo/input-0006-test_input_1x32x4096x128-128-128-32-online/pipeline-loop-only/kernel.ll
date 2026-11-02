; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0006-test_input_1x32x4096x128-128-128-32-online/pipeline-loop-only/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0006-test_input_1x32x4096x128-128-128-32-online/pipeline-loop-only/input.mlir"

@shm_0 = addrspace(3) global [32768 x i8] undef, align 16

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
  %.idx = shl nuw nsw i64 %17, 20, !dbg !12
  %32 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %33 = getelementptr [2 x i8], ptr addrspace(1) %32, i64 %21, !dbg !12
  %34 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %35 = getelementptr i8, ptr addrspace(1) %34, i64 64, !dbg !12
  %36 = shl nsw i64 %8, 12, !dbg !16
  %37 = shl nsw i64 %6, 17, !dbg !16
  %38 = shl nsw i64 %8, 7, !dbg !16
  %39 = shl nsw i64 %6, 12, !dbg !16
  %40 = add nsw i64 %38, %39, !dbg !16
  %41 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %42 = getelementptr i8, ptr addrspace(1) %41, i64 128, !dbg !12
  %43 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %44 = getelementptr i8, ptr addrspace(1) %43, i64 192, !dbg !12
  br label %45, !dbg !7

45:                                               ; preds = %831, %4
  %46 = phi i64 [ %1762, %831 ], [ 0, %4 ], !dbg !7
  %47 = phi [2 x <8 x float>] [ %1747, %831 ], [ zeroinitializer, %4 ], !dbg !7
  %48 = phi [2 x <1 x float>] [ %1353, %831 ], [ zeroinitializer, %4 ], !dbg !7
  %49 = phi [2 x <1 x float>] [ %1749, %831 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %50 = phi [2 x <8 x float>] [ %1753, %831 ], [ zeroinitializer, %4 ], !dbg !7
  %51 = phi [2 x <8 x float>] [ %1757, %831 ], [ zeroinitializer, %4 ], !dbg !7
  %52 = phi [2 x <8 x float>] [ %1761, %831 ], [ zeroinitializer, %4 ], !dbg !7
  %.not = icmp sgt i64 %46, %10, !dbg !7
  br i1 %.not, label %1763, label %53, !dbg !7

53:                                               ; preds = %45
  %54 = shl nsw i64 %46, 7, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !12, !range !40, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local174 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local175 = and i64 %address.local174, 1536, !dbg !12
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !12
  %address.local177 = ptrtoint ptr addrspace(3) %address.local176 to i64, !dbg !12
  %address.local178 = trunc i64 %address.local177 to i32, !dbg !12
  %55 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local178), !dbg !12
  %56 = zext i32 %55 to i64, !dbg !12
  %57 = inttoptr i64 %56 to ptr addrspace(3), !dbg !12
  %address.local179 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local180 = and i64 %address.local179, 16256, !dbg !12
  %address.local181 = lshr i64 %address.local, 4, !dbg !12
  %address.local182 = and i64 %address.local181, 8, !dbg !12
  %address.local183 = or disjoint i64 %address.local180, %address.local182, !dbg !12
  %address.local184 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local183, !dbg !12
  %58 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %59 = ptrtoint ptr addrspace(1) %address.local184 to i64, !dbg !12
  %60 = sub i64 %59, %58, !dbg !12
  %dma.buffer.offset1154 = trunc i64 %60 to i32, !dbg !12
  %61 = trunc i64 %58 to i32, !dbg !12
  %62 = insertelement <4 x i32> poison, i32 %61, i64 0, !dbg !12
  %63 = lshr i64 %58, 32, !dbg !12
  %64 = trunc i64 %63 to i32, !dbg !12
  %65 = insertelement <4 x i32> %62, i32 %64, i64 1, !dbg !12
  %66 = insertelement <4 x i32> %65, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1155 = insertelement <4 x i32> %66, i32 131072, i64 3, !dbg !12
  %67 = ptrtoint ptr addrspace(3) %57 to i32, !dbg !12
  %dma.buffer.lds1156 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %67), !dbg !12
  %68 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1156, i32 %dma.buffer.offset1154, <4 x i32> %dma.buffer.rsrc1155) #10, !dbg !12
  %address.local185 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local186 = shl nuw nsw i64 %address.local185, 3, !dbg !12
  %address.local187 = and i64 %address.local186, 3584, !dbg !12
  %address.local188 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local187, !dbg !12
  %address.local189 = ptrtoint ptr addrspace(3) %address.local188 to i64, !dbg !12
  %address.local190 = trunc i64 %address.local189 to i32, !dbg !12
  %69 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local190), !dbg !12
  %70 = zext i32 %69 to i64, !dbg !12
  %71 = inttoptr i64 %70 to ptr addrspace(3), !dbg !12
  %address.local191 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local192 = and i64 %address.local191, 16256, !dbg !12
  %address.local193 = lshr i64 %address.local185, 4, !dbg !12
  %address.local194 = and i64 %address.local193, 24, !dbg !12
  %address.local195 = or disjoint i64 %address.local192, %address.local194, !dbg !12
  %address.local196 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local195, !dbg !12
  %72 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %73 = ptrtoint ptr addrspace(1) %address.local196 to i64, !dbg !12
  %74 = sub i64 %73, %72, !dbg !12
  %dma.buffer.offset1151 = trunc i64 %74 to i32, !dbg !12
  %75 = trunc i64 %72 to i32, !dbg !12
  %76 = insertelement <4 x i32> poison, i32 %75, i64 0, !dbg !12
  %77 = lshr i64 %72, 32, !dbg !12
  %78 = trunc i64 %77 to i32, !dbg !12
  %79 = insertelement <4 x i32> %76, i32 %78, i64 1, !dbg !12
  %80 = insertelement <4 x i32> %79, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1152 = insertelement <4 x i32> %80, i32 131072, i64 3, !dbg !12
  %81 = ptrtoint ptr addrspace(3) %71 to i32, !dbg !12
  %dma.buffer.lds1153 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %81), !dbg !12
  %82 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1153, i32 %dma.buffer.offset1151, <4 x i32> %dma.buffer.rsrc1152) #10, !dbg !12
  %83 = lshr i64 %46, 12, !dbg !16
  %84 = add nsw i64 %83, %8, !dbg !16
  %85 = lshr i64 %46, 5, !dbg !16
  %86 = and i64 %85, 127, !dbg !16
  %87 = shl i64 %84, 19, !dbg !16
  %88 = and i64 %87, 16252928, !dbg !16
  %89 = shl i64 %85, 19, !dbg !16
  %90 = shl i64 %46, 14, !dbg !16
  %91 = sub i64 %90, %89, !dbg !16
  %92 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %88, !dbg !16
  %93 = getelementptr [2 x i8], ptr addrspace(1) %92, i64 %86, !dbg !16
  %94 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %91, !dbg !16
  %address.local197 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local175, !dbg !16
  %address.local198 = ptrtoint ptr addrspace(3) %address.local197 to i64, !dbg !16
  %address.local199 = trunc i64 %address.local198 to i32, !dbg !16
  %95 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local199), !dbg !16
  %96 = zext i32 %95 to i64, !dbg !16
  %97 = inttoptr i64 %96 to ptr addrspace(3), !dbg !16
  %address.local200 = getelementptr [2 x i8], ptr addrspace(1) %94, i64 %address.local183, !dbg !16
  %98 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %99 = ptrtoint ptr addrspace(1) %address.local200 to i64, !dbg !16
  %100 = sub i64 %99, %98, !dbg !16
  %dma.buffer.offset1148 = trunc i64 %100 to i32, !dbg !16
  %101 = trunc i64 %98 to i32, !dbg !16
  %102 = insertelement <4 x i32> poison, i32 %101, i64 0, !dbg !16
  %103 = lshr i64 %98, 32, !dbg !16
  %104 = trunc i64 %103 to i32, !dbg !16
  %105 = insertelement <4 x i32> %102, i32 %104, i64 1, !dbg !16
  %106 = insertelement <4 x i32> %105, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1149 = insertelement <4 x i32> %106, i32 131072, i64 3, !dbg !16
  %107 = ptrtoint ptr addrspace(3) %97 to i32, !dbg !16
  %dma.buffer.lds1150 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %107), !dbg !16
  %108 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1150, i32 %dma.buffer.offset1148, <4 x i32> %dma.buffer.rsrc1149) #10, !dbg !16
  %address.local201 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local187, !dbg !16
  %address.local202 = ptrtoint ptr addrspace(3) %address.local201 to i64, !dbg !16
  %address.local203 = trunc i64 %address.local202 to i32, !dbg !16
  %109 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local203), !dbg !16
  %110 = zext i32 %109 to i64, !dbg !16
  %111 = inttoptr i64 %110 to ptr addrspace(3), !dbg !16
  %address.local204 = getelementptr [2 x i8], ptr addrspace(1) %94, i64 %address.local195, !dbg !16
  %112 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %113 = ptrtoint ptr addrspace(1) %address.local204 to i64, !dbg !16
  %114 = sub i64 %113, %112, !dbg !16
  %dma.buffer.offset1145 = trunc i64 %114 to i32, !dbg !16
  %115 = trunc i64 %112 to i32, !dbg !16
  %116 = insertelement <4 x i32> poison, i32 %115, i64 0, !dbg !16
  %117 = lshr i64 %112, 32, !dbg !16
  %118 = trunc i64 %117 to i32, !dbg !16
  %119 = insertelement <4 x i32> %116, i32 %118, i64 1, !dbg !16
  %120 = insertelement <4 x i32> %119, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1146 = insertelement <4 x i32> %120, i32 131072, i64 3, !dbg !16
  %121 = ptrtoint ptr addrspace(3) %111 to i32, !dbg !16
  %dma.buffer.lds1147 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %121), !dbg !16
  %122 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1147, i32 %dma.buffer.offset1145, <4 x i32> %dma.buffer.rsrc1146) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %address.local205 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local175, !dbg !12
  %address.local206 = ptrtoint ptr addrspace(3) %address.local205 to i64, !dbg !12
  %address.local207 = trunc i64 %address.local206 to i32, !dbg !12
  %123 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local207), !dbg !12
  %124 = zext i32 %123 to i64, !dbg !12
  %125 = inttoptr i64 %124 to ptr addrspace(3), !dbg !12
  %address.local208 = getelementptr [2 x i8], ptr addrspace(1) %35, i64 %address.local183, !dbg !12
  %126 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %127 = ptrtoint ptr addrspace(1) %address.local208 to i64, !dbg !12
  %128 = sub i64 %127, %126, !dbg !12
  %dma.buffer.offset1142 = trunc i64 %128 to i32, !dbg !12
  %129 = trunc i64 %126 to i32, !dbg !12
  %130 = insertelement <4 x i32> poison, i32 %129, i64 0, !dbg !12
  %131 = lshr i64 %126, 32, !dbg !12
  %132 = trunc i64 %131 to i32, !dbg !12
  %133 = insertelement <4 x i32> %130, i32 %132, i64 1, !dbg !12
  %134 = insertelement <4 x i32> %133, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1143 = insertelement <4 x i32> %134, i32 131072, i64 3, !dbg !12
  %135 = ptrtoint ptr addrspace(3) %125 to i32, !dbg !12
  %dma.buffer.lds1144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %135), !dbg !12
  %136 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1144, i32 %dma.buffer.offset1142, <4 x i32> %dma.buffer.rsrc1143) #10, !dbg !12
  %address.local209 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local187, !dbg !12
  %address.local210 = ptrtoint ptr addrspace(3) %address.local209 to i64, !dbg !12
  %address.local211 = trunc i64 %address.local210 to i32, !dbg !12
  %137 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local211), !dbg !12
  %138 = zext i32 %137 to i64, !dbg !12
  %139 = inttoptr i64 %138 to ptr addrspace(3), !dbg !12
  %address.local212 = getelementptr [2 x i8], ptr addrspace(1) %35, i64 %address.local195, !dbg !12
  %140 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %141 = ptrtoint ptr addrspace(1) %address.local212 to i64, !dbg !12
  %142 = sub i64 %141, %140, !dbg !12
  %dma.buffer.offset1139 = trunc i64 %142 to i32, !dbg !12
  %143 = trunc i64 %140 to i32, !dbg !12
  %144 = insertelement <4 x i32> poison, i32 %143, i64 0, !dbg !12
  %145 = lshr i64 %140, 32, !dbg !12
  %146 = trunc i64 %145 to i32, !dbg !12
  %147 = insertelement <4 x i32> %144, i32 %146, i64 1, !dbg !12
  %148 = insertelement <4 x i32> %147, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1140 = insertelement <4 x i32> %148, i32 131072, i64 3, !dbg !12
  %149 = ptrtoint ptr addrspace(3) %139 to i32, !dbg !12
  %dma.buffer.lds1141 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %149), !dbg !12
  %150 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1141, i32 %dma.buffer.offset1139, <4 x i32> %dma.buffer.rsrc1140) #10, !dbg !12
  %151 = add i64 %46, %36, !dbg !16
  %152 = add i64 %151, %37, !dbg !16
  %153 = add i64 %152, 1024, !dbg !16
  %154 = icmp slt i64 %153, 0, !dbg !16
  %155 = sub i64 -1025, %152, !dbg !16
  %156 = select i1 %154, i64 %155, i64 %153, !dbg !16
  %157 = sdiv i64 %156, 4096, !dbg !16
  %.lobit4 = ashr i64 %153, 63, !dbg !16
  %158 = xor i64 %157, %.lobit4, !dbg !16
  %159 = add nsw i64 %85, %40, !dbg !16
  %160 = add nsw i64 %159, 32, !dbg !16
  %161 = icmp slt i64 %159, -32, !dbg !16
  %162 = sub nsw i64 -33, %159, !dbg !16
  %163 = select i1 %161, i64 %162, i64 %160, !dbg !16
  %164 = sdiv i64 %163, 128, !dbg !16
  %165 = sext i1 %161 to i64, !dbg !16
  %166 = xor i64 %164, %165, !dbg !16
  %167 = shl nsw i64 %166, 7, !dbg !16
  %168 = sub i64 %40, %167, !dbg !16
  %169 = shl i64 %158, 19, !dbg !16
  %170 = and i64 %169, 16252928, !dbg !16
  %171 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %170, !dbg !16
  %172 = getelementptr [2 x i8], ptr addrspace(1) %171, i64 %85, !dbg !16
  %173 = getelementptr [2 x i8], ptr addrspace(1) %172, i64 %168, !dbg !16
  %174 = getelementptr i8, ptr addrspace(1) %173, i64 64, !dbg !16
  %175 = getelementptr [2 x i8], ptr addrspace(1) %174, i64 %91, !dbg !16
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local175, !dbg !16
  %address.local214 = ptrtoint ptr addrspace(3) %address.local213 to i64, !dbg !16
  %address.local215 = trunc i64 %address.local214 to i32, !dbg !16
  %176 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local215), !dbg !16
  %177 = zext i32 %176 to i64, !dbg !16
  %178 = inttoptr i64 %177 to ptr addrspace(3), !dbg !16
  %address.local216 = getelementptr [2 x i8], ptr addrspace(1) %175, i64 %address.local183, !dbg !16
  %179 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %180 = ptrtoint ptr addrspace(1) %address.local216 to i64, !dbg !16
  %181 = sub i64 %180, %179, !dbg !16
  %dma.buffer.offset1136 = trunc i64 %181 to i32, !dbg !16
  %182 = trunc i64 %179 to i32, !dbg !16
  %183 = insertelement <4 x i32> poison, i32 %182, i64 0, !dbg !16
  %184 = lshr i64 %179, 32, !dbg !16
  %185 = trunc i64 %184 to i32, !dbg !16
  %186 = insertelement <4 x i32> %183, i32 %185, i64 1, !dbg !16
  %187 = insertelement <4 x i32> %186, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1137 = insertelement <4 x i32> %187, i32 131072, i64 3, !dbg !16
  %188 = ptrtoint ptr addrspace(3) %178 to i32, !dbg !16
  %dma.buffer.lds1138 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %188), !dbg !16
  %189 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1138, i32 %dma.buffer.offset1136, <4 x i32> %dma.buffer.rsrc1137) #10, !dbg !16
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local187, !dbg !16
  %address.local218 = ptrtoint ptr addrspace(3) %address.local217 to i64, !dbg !16
  %address.local219 = trunc i64 %address.local218 to i32, !dbg !16
  %190 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local219), !dbg !16
  %191 = zext i32 %190 to i64, !dbg !16
  %192 = inttoptr i64 %191 to ptr addrspace(3), !dbg !16
  %address.local220 = getelementptr [2 x i8], ptr addrspace(1) %175, i64 %address.local195, !dbg !16
  %193 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %194 = ptrtoint ptr addrspace(1) %address.local220 to i64, !dbg !16
  %195 = sub i64 %194, %193, !dbg !16
  %dma.buffer.offset1133 = trunc i64 %195 to i32, !dbg !16
  %196 = trunc i64 %193 to i32, !dbg !16
  %197 = insertelement <4 x i32> poison, i32 %196, i64 0, !dbg !16
  %198 = lshr i64 %193, 32, !dbg !16
  %199 = trunc i64 %198 to i32, !dbg !16
  %200 = insertelement <4 x i32> %197, i32 %199, i64 1, !dbg !16
  %201 = insertelement <4 x i32> %200, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1134 = insertelement <4 x i32> %201, i32 131072, i64 3, !dbg !16
  %202 = ptrtoint ptr addrspace(3) %192 to i32, !dbg !16
  %dma.buffer.lds1135 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %202), !dbg !16
  %203 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1135, i32 %dma.buffer.offset1133, <4 x i32> %dma.buffer.rsrc1134) #10, !dbg !16
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
  %204 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !13
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
  %205 = load <4 x half>, ptr addrspace(3) %address.local247, align 2, !dbg !13
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
  %206 = load <4 x half>, ptr addrspace(3) %address.local261, align 2, !dbg !13
  %207 = fmul <4 x half> %204, splat (half 0xH3015), !dbg !13
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
  %208 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !13
  %209 = fmul <4 x half> %205, splat (half 0xH3015), !dbg !13
  %address.local276 = lshr i32 %address.thread, 2, !dbg !17
  %address.local277 = and i32 %address.local276, 4, !dbg !17
  %address.local278 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local279 = and i32 %address.local278, 24, !dbg !17
  %address.local280 = or disjoint i32 %address.local277, %address.local279, !dbg !17
  %address.local281 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local282 = and i32 %address.local281, 1024, !dbg !17
  %address.local283 = or disjoint i32 %address.local280, %address.local282, !dbg !17
  %address.local284 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local285 = and i32 %address.local284, 384, !dbg !17
  %address.local286 = or disjoint i32 %address.local283, %address.local285, !dbg !17
  %address.local287 = zext nneg i32 %address.local286 to i64, !dbg !17
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local287, !dbg !17
  %210 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !17
  %211 = fmul <4 x half> %206, splat (half 0xH3015), !dbg !13
  %address.local289 = lshr i32 %address.thread, 2, !dbg !17
  %address.local290 = and i32 %address.local289, 4, !dbg !17
  %address.local291 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local292 = and i32 %address.local291, 24, !dbg !17
  %address.local293 = or disjoint i32 %address.local290, %address.local292, !dbg !17
  %address.local294 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local295 = and i32 %address.local294, 1024, !dbg !17
  %address.local296 = or disjoint i32 %address.local293, %address.local295, !dbg !17
  %address.local297 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local298 = and i32 %address.local297, 384, !dbg !17
  %address.local299 = or disjoint i32 %address.local296, %address.local298, !dbg !17
  %address.local300 = or disjoint i32 %address.local299, 2048, !dbg !17
  %address.local301 = zext nneg i32 %address.local300 to i64, !dbg !17
  %address.local302 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local301, !dbg !17
  %212 = load <4 x half>, ptr addrspace(3) %address.local302, align 2, !dbg !17
  %213 = fmul <4 x half> %208, splat (half 0xH3015), !dbg !13
  %address.local303 = lshr i32 %address.thread, 2, !dbg !17
  %address.local304 = and i32 %address.local303, 4, !dbg !17
  %address.local305 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local306 = and i32 %address.local305, 24, !dbg !17
  %address.local307 = or disjoint i32 %address.local304, %address.local306, !dbg !17
  %address.local308 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local309 = and i32 %address.local308, 1024, !dbg !17
  %address.local310 = or disjoint i32 %address.local307, %address.local309, !dbg !17
  %address.local311 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local312 = and i32 %address.local311, 384, !dbg !17
  %address.local313 = or disjoint i32 %address.local310, %address.local312, !dbg !17
  %address.local314 = or disjoint i32 %address.local313, 32, !dbg !17
  %address.local315 = zext nneg i32 %address.local314 to i64, !dbg !17
  %address.local316 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local315, !dbg !17
  %214 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !17
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %207, <4 x half> %210, <4 x float> zeroinitializer) #9, !dbg !17
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %209, <4 x half> %212, <4 x float> %215) #9, !dbg !17
  %address.local317 = lshr i32 %address.thread, 2, !dbg !17
  %address.local318 = and i32 %address.local317, 4, !dbg !17
  %address.local319 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local320 = and i32 %address.local319, 24, !dbg !17
  %address.local321 = or disjoint i32 %address.local318, %address.local320, !dbg !17
  %address.local322 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local323 = and i32 %address.local322, 1024, !dbg !17
  %address.local324 = or disjoint i32 %address.local321, %address.local323, !dbg !17
  %address.local325 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local326 = and i32 %address.local325, 384, !dbg !17
  %address.local327 = or disjoint i32 %address.local324, %address.local326, !dbg !17
  %address.local328 = or disjoint i32 %address.local327, 2080, !dbg !17
  %address.local329 = zext nneg i32 %address.local328 to i64, !dbg !17
  %address.local330 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local329, !dbg !17
  %217 = load <4 x half>, ptr addrspace(3) %address.local330, align 2, !dbg !17
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %207, <4 x half> %214, <4 x float> zeroinitializer) #9, !dbg !17
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %209, <4 x half> %217, <4 x float> %218) #9, !dbg !17
  %address.local331 = lshr i32 %address.thread, 2, !dbg !17
  %address.local332 = and i32 %address.local331, 4, !dbg !17
  %address.local333 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local334 = and i32 %address.local333, 24, !dbg !17
  %address.local335 = or disjoint i32 %address.local332, %address.local334, !dbg !17
  %address.local336 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local337 = and i32 %address.local336, 1024, !dbg !17
  %address.local338 = or disjoint i32 %address.local335, %address.local337, !dbg !17
  %address.local339 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local340 = and i32 %address.local339, 384, !dbg !17
  %address.local341 = or disjoint i32 %address.local338, %address.local340, !dbg !17
  %address.local342 = or disjoint i32 %address.local341, 64, !dbg !17
  %address.local343 = zext nneg i32 %address.local342 to i64, !dbg !17
  %address.local344 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local343, !dbg !17
  %220 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !17
  %address.local345 = lshr i32 %address.thread, 2, !dbg !17
  %address.local346 = and i32 %address.local345, 4, !dbg !17
  %address.local347 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local348 = and i32 %address.local347, 24, !dbg !17
  %address.local349 = or disjoint i32 %address.local346, %address.local348, !dbg !17
  %address.local350 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local351 = and i32 %address.local350, 1024, !dbg !17
  %address.local352 = or disjoint i32 %address.local349, %address.local351, !dbg !17
  %address.local353 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local354 = and i32 %address.local353, 384, !dbg !17
  %address.local355 = or disjoint i32 %address.local352, %address.local354, !dbg !17
  %address.local356 = or disjoint i32 %address.local355, 2112, !dbg !17
  %address.local357 = zext nneg i32 %address.local356 to i64, !dbg !17
  %address.local358 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local357, !dbg !17
  %221 = load <4 x half>, ptr addrspace(3) %address.local358, align 2, !dbg !17
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %207, <4 x half> %220, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local359 = lshr i32 %address.thread, 2, !dbg !17
  %address.local360 = and i32 %address.local359, 4, !dbg !17
  %address.local361 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local362 = and i32 %address.local361, 24, !dbg !17
  %address.local363 = or disjoint i32 %address.local360, %address.local362, !dbg !17
  %address.local364 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local365 = and i32 %address.local364, 1024, !dbg !17
  %address.local366 = or disjoint i32 %address.local363, %address.local365, !dbg !17
  %address.local367 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local368 = and i32 %address.local367, 384, !dbg !17
  %address.local369 = or disjoint i32 %address.local366, %address.local368, !dbg !17
  %address.local370 = or disjoint i32 %address.local369, 96, !dbg !17
  %address.local371 = zext nneg i32 %address.local370 to i64, !dbg !17
  %address.local372 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local371, !dbg !17
  %223 = load <4 x half>, ptr addrspace(3) %address.local372, align 2, !dbg !17
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %209, <4 x half> %221, <4 x float> %222) #9, !dbg !17
  %address.local373 = lshr i32 %address.thread, 2, !dbg !17
  %address.local374 = and i32 %address.local373, 4, !dbg !17
  %address.local375 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local376 = and i32 %address.local375, 24, !dbg !17
  %address.local377 = or disjoint i32 %address.local374, %address.local376, !dbg !17
  %address.local378 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local379 = and i32 %address.local378, 1024, !dbg !17
  %address.local380 = or disjoint i32 %address.local377, %address.local379, !dbg !17
  %address.local381 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local382 = and i32 %address.local381, 384, !dbg !17
  %address.local383 = or disjoint i32 %address.local380, %address.local382, !dbg !17
  %address.local384 = or disjoint i32 %address.local383, 2144, !dbg !17
  %address.local385 = zext nneg i32 %address.local384 to i64, !dbg !17
  %address.local386 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local385, !dbg !17
  %225 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !17
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %207, <4 x half> %223, <4 x float> zeroinitializer) #9, !dbg !17
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %209, <4 x half> %225, <4 x float> %226) #9, !dbg !17
  %address.local387 = lshr i32 %address.thread, 2, !dbg !17
  %address.local388 = and i32 %address.local387, 4, !dbg !17
  %address.local389 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local390 = and i32 %address.local389, 24, !dbg !17
  %address.local391 = or disjoint i32 %address.local388, %address.local390, !dbg !17
  %address.local392 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local393 = and i32 %address.local392, 1024, !dbg !17
  %address.local394 = or disjoint i32 %address.local391, %address.local393, !dbg !17
  %address.local395 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local396 = and i32 %address.local395, 384, !dbg !17
  %address.local397 = or disjoint i32 %address.local394, %address.local396, !dbg !17
  %address.local398 = or disjoint i32 %address.local397, 512, !dbg !17
  %address.local399 = zext nneg i32 %address.local398 to i64, !dbg !17
  %address.local400 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local399, !dbg !17
  %228 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !17
  %address.local401 = lshr i32 %address.thread, 2, !dbg !17
  %address.local402 = and i32 %address.local401, 4, !dbg !17
  %address.local403 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local404 = and i32 %address.local403, 24, !dbg !17
  %address.local405 = or disjoint i32 %address.local402, %address.local404, !dbg !17
  %address.local406 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local407 = and i32 %address.local406, 1024, !dbg !17
  %address.local408 = or disjoint i32 %address.local405, %address.local407, !dbg !17
  %address.local409 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local410 = and i32 %address.local409, 384, !dbg !17
  %address.local411 = or disjoint i32 %address.local408, %address.local410, !dbg !17
  %address.local412 = or disjoint i32 %address.local411, 2560, !dbg !17
  %address.local413 = zext nneg i32 %address.local412 to i64, !dbg !17
  %address.local414 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local413, !dbg !17
  %229 = load <4 x half>, ptr addrspace(3) %address.local414, align 2, !dbg !17
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %207, <4 x half> %228, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local415 = lshr i32 %address.thread, 2, !dbg !17
  %address.local416 = and i32 %address.local415, 4, !dbg !17
  %address.local417 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local418 = and i32 %address.local417, 24, !dbg !17
  %address.local419 = or disjoint i32 %address.local416, %address.local418, !dbg !17
  %address.local420 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local421 = and i32 %address.local420, 1024, !dbg !17
  %address.local422 = or disjoint i32 %address.local419, %address.local421, !dbg !17
  %address.local423 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local424 = and i32 %address.local423, 384, !dbg !17
  %address.local425 = or disjoint i32 %address.local422, %address.local424, !dbg !17
  %address.local426 = or disjoint i32 %address.local425, 544, !dbg !17
  %address.local427 = zext nneg i32 %address.local426 to i64, !dbg !17
  %address.local428 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local427, !dbg !17
  %231 = load <4 x half>, ptr addrspace(3) %address.local428, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %209, <4 x half> %229, <4 x float> %230) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %address.local429 = lshr i32 %address.thread, 2, !dbg !17
  %address.local430 = and i32 %address.local429, 4, !dbg !17
  %address.local431 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local432 = and i32 %address.local431, 24, !dbg !17
  %address.local433 = or disjoint i32 %address.local430, %address.local432, !dbg !17
  %address.local434 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local435 = and i32 %address.local434, 1024, !dbg !17
  %address.local436 = or disjoint i32 %address.local433, %address.local435, !dbg !17
  %address.local437 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local438 = and i32 %address.local437, 384, !dbg !17
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !17
  %address.local440 = or disjoint i32 %address.local439, 2592, !dbg !17
  %address.local441 = zext nneg i32 %address.local440 to i64, !dbg !17
  %address.local442 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local441, !dbg !17
  %233 = load <4 x half>, ptr addrspace(3) %address.local442, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %207, <4 x half> %231, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %209, <4 x half> %233, <4 x float> %234) #9, !dbg !17
  %address.local443 = lshr i32 %address.thread, 2, !dbg !17
  %address.local444 = and i32 %address.local443, 4, !dbg !17
  %address.local445 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local446 = and i32 %address.local445, 24, !dbg !17
  %address.local447 = or disjoint i32 %address.local444, %address.local446, !dbg !17
  %address.local448 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local449 = and i32 %address.local448, 1024, !dbg !17
  %address.local450 = or disjoint i32 %address.local447, %address.local449, !dbg !17
  %address.local451 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local452 = and i32 %address.local451, 384, !dbg !17
  %address.local453 = or disjoint i32 %address.local450, %address.local452, !dbg !17
  %address.local454 = or disjoint i32 %address.local453, 576, !dbg !17
  %address.local455 = zext nneg i32 %address.local454 to i64, !dbg !17
  %address.local456 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local455, !dbg !17
  %236 = load <4 x half>, ptr addrspace(3) %address.local456, align 2, !dbg !17
  %address.local457 = lshr i32 %address.thread, 2, !dbg !17
  %address.local458 = and i32 %address.local457, 4, !dbg !17
  %address.local459 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local460 = and i32 %address.local459, 24, !dbg !17
  %address.local461 = or disjoint i32 %address.local458, %address.local460, !dbg !17
  %address.local462 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local463 = and i32 %address.local462, 1024, !dbg !17
  %address.local464 = or disjoint i32 %address.local461, %address.local463, !dbg !17
  %address.local465 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local466 = and i32 %address.local465, 384, !dbg !17
  %address.local467 = or disjoint i32 %address.local464, %address.local466, !dbg !17
  %address.local468 = or disjoint i32 %address.local467, 2624, !dbg !17
  %address.local469 = zext nneg i32 %address.local468 to i64, !dbg !17
  %address.local470 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local469, !dbg !17
  %237 = load <4 x half>, ptr addrspace(3) %address.local470, align 2, !dbg !17
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %207, <4 x half> %236, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local471 = lshr i32 %address.thread, 2, !dbg !17
  %address.local472 = and i32 %address.local471, 4, !dbg !17
  %address.local473 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local474 = and i32 %address.local473, 24, !dbg !17
  %address.local475 = or disjoint i32 %address.local472, %address.local474, !dbg !17
  %address.local476 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local477 = and i32 %address.local476, 1024, !dbg !17
  %address.local478 = or disjoint i32 %address.local475, %address.local477, !dbg !17
  %address.local479 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local480 = and i32 %address.local479, 384, !dbg !17
  %address.local481 = or disjoint i32 %address.local478, %address.local480, !dbg !17
  %address.local482 = or disjoint i32 %address.local481, 608, !dbg !17
  %address.local483 = zext nneg i32 %address.local482 to i64, !dbg !17
  %address.local484 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local483, !dbg !17
  %239 = load <4 x half>, ptr addrspace(3) %address.local484, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %209, <4 x half> %237, <4 x float> %238) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %address.local485 = lshr i32 %address.thread, 2, !dbg !17
  %address.local486 = and i32 %address.local485, 4, !dbg !17
  %address.local487 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local488 = and i32 %address.local487, 24, !dbg !17
  %address.local489 = or disjoint i32 %address.local486, %address.local488, !dbg !17
  %address.local490 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local491 = and i32 %address.local490, 1024, !dbg !17
  %address.local492 = or disjoint i32 %address.local489, %address.local491, !dbg !17
  %address.local493 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local494 = and i32 %address.local493, 384, !dbg !17
  %address.local495 = or disjoint i32 %address.local492, %address.local494, !dbg !17
  %address.local496 = or disjoint i32 %address.local495, 2656, !dbg !17
  %address.local497 = zext nneg i32 %address.local496 to i64, !dbg !17
  %address.local498 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local497, !dbg !17
  %241 = load <4 x half>, ptr addrspace(3) %address.local498, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %207, <4 x half> %239, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %209, <4 x half> %241, <4 x float> %242) #9, !dbg !17
  %244 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !17
  %245 = load <4 x half>, ptr addrspace(3) %address.local302, align 2, !dbg !17
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %211, <4 x half> %244, <4 x float> zeroinitializer) #9, !dbg !17
  %247 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %245, <4 x float> %246) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %249 = load <4 x half>, ptr addrspace(3) %address.local330, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %211, <4 x half> %247, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %249, <4 x float> %250) #9, !dbg !17
  %252 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !17
  %253 = load <4 x half>, ptr addrspace(3) %address.local358, align 2, !dbg !17
  %254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %211, <4 x half> %252, <4 x float> zeroinitializer) #9, !dbg !17
  %255 = load <4 x half>, ptr addrspace(3) %address.local372, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %253, <4 x float> %254) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %257 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %211, <4 x half> %255, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %257, <4 x float> %258) #9, !dbg !17
  %260 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !17
  %261 = load <4 x half>, ptr addrspace(3) %address.local414, align 2, !dbg !17
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %211, <4 x half> %260, <4 x float> zeroinitializer) #9, !dbg !17
  %263 = load <4 x half>, ptr addrspace(3) %address.local428, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %261, <4 x float> %262) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %265 = load <4 x half>, ptr addrspace(3) %address.local442, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %211, <4 x half> %263, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %265, <4 x float> %266) #9, !dbg !17
  %268 = load <4 x half>, ptr addrspace(3) %address.local456, align 2, !dbg !17
  %269 = load <4 x half>, ptr addrspace(3) %address.local470, align 2, !dbg !17
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %211, <4 x half> %268, <4 x float> zeroinitializer) #9, !dbg !17
  %271 = load <4 x half>, ptr addrspace(3) %address.local484, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %269, <4 x float> %270) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %273 = load <4 x half>, ptr addrspace(3) %address.local498, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %211, <4 x half> %271, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %213, <4 x half> %273, <4 x float> %274) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %276 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local178), !dbg !12
  %277 = zext i32 %276 to i64, !dbg !12
  %278 = inttoptr i64 %277 to ptr addrspace(3), !dbg !12
  %address.local499 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %address.local183, !dbg !12
  %279 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %280 = ptrtoint ptr addrspace(1) %address.local499 to i64, !dbg !12
  %281 = sub i64 %280, %279, !dbg !12
  %dma.buffer.offset1130 = trunc i64 %281 to i32, !dbg !12
  %282 = trunc i64 %279 to i32, !dbg !12
  %283 = insertelement <4 x i32> poison, i32 %282, i64 0, !dbg !12
  %284 = lshr i64 %279, 32, !dbg !12
  %285 = trunc i64 %284 to i32, !dbg !12
  %286 = insertelement <4 x i32> %283, i32 %285, i64 1, !dbg !12
  %287 = insertelement <4 x i32> %286, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1131 = insertelement <4 x i32> %287, i32 131072, i64 3, !dbg !12
  %288 = ptrtoint ptr addrspace(3) %278 to i32, !dbg !12
  %dma.buffer.lds1132 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %288), !dbg !12
  %289 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1132, i32 %dma.buffer.offset1130, <4 x i32> %dma.buffer.rsrc1131) #10, !dbg !12
  %290 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local190), !dbg !12
  %291 = zext i32 %290 to i64, !dbg !12
  %292 = inttoptr i64 %291 to ptr addrspace(3), !dbg !12
  %address.local500 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %address.local195, !dbg !12
  %293 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %294 = ptrtoint ptr addrspace(1) %address.local500 to i64, !dbg !12
  %295 = sub i64 %294, %293, !dbg !12
  %dma.buffer.offset1127 = trunc i64 %295 to i32, !dbg !12
  %296 = trunc i64 %293 to i32, !dbg !12
  %297 = insertelement <4 x i32> poison, i32 %296, i64 0, !dbg !12
  %298 = lshr i64 %293, 32, !dbg !12
  %299 = trunc i64 %298 to i32, !dbg !12
  %300 = insertelement <4 x i32> %297, i32 %299, i64 1, !dbg !12
  %301 = insertelement <4 x i32> %300, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1128 = insertelement <4 x i32> %301, i32 131072, i64 3, !dbg !12
  %302 = ptrtoint ptr addrspace(3) %292 to i32, !dbg !12
  %dma.buffer.lds1129 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %302), !dbg !12
  %303 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1129, i32 %dma.buffer.offset1127, <4 x i32> %dma.buffer.rsrc1128) #10, !dbg !12
  %304 = add i64 %152, 2048, !dbg !16
  %305 = icmp slt i64 %304, 0, !dbg !16
  %306 = sub i64 -2049, %152, !dbg !16
  %307 = select i1 %305, i64 %306, i64 %304, !dbg !16
  %308 = sdiv i64 %307, 4096, !dbg !16
  %.lobit5 = ashr i64 %304, 63, !dbg !16
  %309 = xor i64 %308, %.lobit5, !dbg !16
  %310 = add nsw i64 %159, 64, !dbg !16
  %311 = icmp slt i64 %159, -64, !dbg !16
  %312 = sub nsw i64 -65, %159, !dbg !16
  %313 = select i1 %311, i64 %312, i64 %310, !dbg !16
  %314 = sdiv i64 %313, 128, !dbg !16
  %315 = sext i1 %311 to i64, !dbg !16
  %316 = xor i64 %314, %315, !dbg !16
  %317 = shl nsw i64 %316, 7, !dbg !16
  %318 = sub i64 %40, %317, !dbg !16
  %319 = shl i64 %309, 19, !dbg !16
  %320 = and i64 %319, 16252928, !dbg !16
  %321 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %320, !dbg !16
  %322 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %85, !dbg !16
  %323 = getelementptr [2 x i8], ptr addrspace(1) %322, i64 %318, !dbg !16
  %324 = getelementptr i8, ptr addrspace(1) %323, i64 128, !dbg !16
  %325 = getelementptr [2 x i8], ptr addrspace(1) %324, i64 %91, !dbg !16
  %326 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local199), !dbg !16
  %327 = zext i32 %326 to i64, !dbg !16
  %328 = inttoptr i64 %327 to ptr addrspace(3), !dbg !16
  %address.local501 = getelementptr [2 x i8], ptr addrspace(1) %325, i64 %address.local183, !dbg !16
  %329 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %330 = ptrtoint ptr addrspace(1) %address.local501 to i64, !dbg !16
  %331 = sub i64 %330, %329, !dbg !16
  %dma.buffer.offset1124 = trunc i64 %331 to i32, !dbg !16
  %332 = trunc i64 %329 to i32, !dbg !16
  %333 = insertelement <4 x i32> poison, i32 %332, i64 0, !dbg !16
  %334 = lshr i64 %329, 32, !dbg !16
  %335 = trunc i64 %334 to i32, !dbg !16
  %336 = insertelement <4 x i32> %333, i32 %335, i64 1, !dbg !16
  %337 = insertelement <4 x i32> %336, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1125 = insertelement <4 x i32> %337, i32 131072, i64 3, !dbg !16
  %338 = ptrtoint ptr addrspace(3) %328 to i32, !dbg !16
  %dma.buffer.lds1126 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %338), !dbg !16
  %339 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1126, i32 %dma.buffer.offset1124, <4 x i32> %dma.buffer.rsrc1125) #10, !dbg !16
  %340 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local203), !dbg !16
  %341 = zext i32 %340 to i64, !dbg !16
  %342 = inttoptr i64 %341 to ptr addrspace(3), !dbg !16
  %address.local502 = getelementptr [2 x i8], ptr addrspace(1) %325, i64 %address.local195, !dbg !16
  %343 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %344 = ptrtoint ptr addrspace(1) %address.local502 to i64, !dbg !16
  %345 = sub i64 %344, %343, !dbg !16
  %dma.buffer.offset1121 = trunc i64 %345 to i32, !dbg !16
  %346 = trunc i64 %343 to i32, !dbg !16
  %347 = insertelement <4 x i32> poison, i32 %346, i64 0, !dbg !16
  %348 = lshr i64 %343, 32, !dbg !16
  %349 = trunc i64 %348 to i32, !dbg !16
  %350 = insertelement <4 x i32> %347, i32 %349, i64 1, !dbg !16
  %351 = insertelement <4 x i32> %350, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1122 = insertelement <4 x i32> %351, i32 131072, i64 3, !dbg !16
  %352 = ptrtoint ptr addrspace(3) %342 to i32, !dbg !16
  %dma.buffer.lds1123 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %352), !dbg !16
  %353 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1123, i32 %dma.buffer.offset1121, <4 x i32> %dma.buffer.rsrc1122) #10, !dbg !16
  %address.local503 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local232, !dbg !13
  %354 = load <4 x half>, ptr addrspace(3) %address.local503, align 2, !dbg !13
  %355 = fmul <4 x half> %354, splat (half 0xH3015), !dbg !13
  %address.local504 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local246, !dbg !13
  %356 = load <4 x half>, ptr addrspace(3) %address.local504, align 2, !dbg !13
  %address.local505 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local260, !dbg !13
  %357 = load <4 x half>, ptr addrspace(3) %address.local505, align 2, !dbg !13
  %358 = fmul <4 x half> %356, splat (half 0xH3015), !dbg !13
  %359 = fmul <4 x half> %357, splat (half 0xH3015), !dbg !13
  %address.local506 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local274, !dbg !13
  %360 = load <4 x half>, ptr addrspace(3) %address.local506, align 2, !dbg !13
  %address.local507 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local287, !dbg !17
  %361 = load <4 x half>, ptr addrspace(3) %address.local507, align 2, !dbg !17
  %362 = fmul <4 x half> %360, splat (half 0xH3015), !dbg !13
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %355, <4 x half> %361, <4 x float> %216) #9, !dbg !17
  %address.local508 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local301, !dbg !17
  %364 = load <4 x half>, ptr addrspace(3) %address.local508, align 2, !dbg !17
  %address.local509 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local315, !dbg !17
  %365 = load <4 x half>, ptr addrspace(3) %address.local509, align 2, !dbg !17
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %364, <4 x float> %363) #9, !dbg !17
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %355, <4 x half> %365, <4 x float> %219) #9, !dbg !17
  %address.local510 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local329, !dbg !17
  %368 = load <4 x half>, ptr addrspace(3) %address.local510, align 2, !dbg !17
  %address.local511 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local343, !dbg !17
  %369 = load <4 x half>, ptr addrspace(3) %address.local511, align 2, !dbg !17
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %368, <4 x float> %367) #9, !dbg !17
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %355, <4 x half> %369, <4 x float> %224) #9, !dbg !17
  %address.local512 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local357, !dbg !17
  %372 = load <4 x half>, ptr addrspace(3) %address.local512, align 2, !dbg !17
  %address.local513 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local371, !dbg !17
  %373 = load <4 x half>, ptr addrspace(3) %address.local513, align 2, !dbg !17
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %372, <4 x float> %371) #9, !dbg !17
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %355, <4 x half> %373, <4 x float> %227) #9, !dbg !17
  %address.local514 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local385, !dbg !17
  %376 = load <4 x half>, ptr addrspace(3) %address.local514, align 2, !dbg !17
  %address.local515 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local399, !dbg !17
  %377 = load <4 x half>, ptr addrspace(3) %address.local515, align 2, !dbg !17
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %376, <4 x float> %375) #9, !dbg !17
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %355, <4 x half> %377, <4 x float> %232) #9, !dbg !17
  %address.local516 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local413, !dbg !17
  %380 = load <4 x half>, ptr addrspace(3) %address.local516, align 2, !dbg !17
  %address.local517 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local427, !dbg !17
  %381 = load <4 x half>, ptr addrspace(3) %address.local517, align 2, !dbg !17
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %380, <4 x float> %379) #9, !dbg !17
  %383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %355, <4 x half> %381, <4 x float> %235) #9, !dbg !17
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local441, !dbg !17
  %384 = load <4 x half>, ptr addrspace(3) %address.local518, align 2, !dbg !17
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local455, !dbg !17
  %385 = load <4 x half>, ptr addrspace(3) %address.local519, align 2, !dbg !17
  %386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %384, <4 x float> %383) #9, !dbg !17
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %355, <4 x half> %385, <4 x float> %240) #9, !dbg !17
  %address.local520 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local469, !dbg !17
  %388 = load <4 x half>, ptr addrspace(3) %address.local520, align 2, !dbg !17
  %address.local521 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local483, !dbg !17
  %389 = load <4 x half>, ptr addrspace(3) %address.local521, align 2, !dbg !17
  %390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %388, <4 x float> %387) #9, !dbg !17
  %391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %355, <4 x half> %389, <4 x float> %243) #9, !dbg !17
  %address.local522 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local497, !dbg !17
  %392 = load <4 x half>, ptr addrspace(3) %address.local522, align 2, !dbg !17
  %393 = load <4 x half>, ptr addrspace(3) %address.local507, align 2, !dbg !17
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %392, <4 x float> %391) #9, !dbg !17
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %393, <4 x float> %248) #9, !dbg !17
  %396 = load <4 x half>, ptr addrspace(3) %address.local508, align 2, !dbg !17
  %397 = load <4 x half>, ptr addrspace(3) %address.local509, align 2, !dbg !17
  %398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %396, <4 x float> %395) #9, !dbg !17
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %397, <4 x float> %251) #9, !dbg !17
  %400 = load <4 x half>, ptr addrspace(3) %address.local510, align 2, !dbg !17
  %401 = load <4 x half>, ptr addrspace(3) %address.local511, align 2, !dbg !17
  %402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %400, <4 x float> %399) #9, !dbg !17
  %403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %401, <4 x float> %256) #9, !dbg !17
  %404 = load <4 x half>, ptr addrspace(3) %address.local512, align 2, !dbg !17
  %405 = load <4 x half>, ptr addrspace(3) %address.local513, align 2, !dbg !17
  %406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %404, <4 x float> %403) #9, !dbg !17
  %407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %405, <4 x float> %259) #9, !dbg !17
  %408 = load <4 x half>, ptr addrspace(3) %address.local514, align 2, !dbg !17
  %409 = load <4 x half>, ptr addrspace(3) %address.local515, align 2, !dbg !17
  %410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %408, <4 x float> %407) #9, !dbg !17
  %411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %409, <4 x float> %264) #9, !dbg !17
  %412 = load <4 x half>, ptr addrspace(3) %address.local516, align 2, !dbg !17
  %413 = load <4 x half>, ptr addrspace(3) %address.local517, align 2, !dbg !17
  %414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %412, <4 x float> %411) #9, !dbg !17
  %415 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %413, <4 x float> %267) #9, !dbg !17
  %416 = load <4 x half>, ptr addrspace(3) %address.local518, align 2, !dbg !17
  %417 = load <4 x half>, ptr addrspace(3) %address.local519, align 2, !dbg !17
  %418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %416, <4 x float> %415) #9, !dbg !17
  %419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %417, <4 x float> %272) #9, !dbg !17
  %420 = load <4 x half>, ptr addrspace(3) %address.local520, align 2, !dbg !17
  %421 = load <4 x half>, ptr addrspace(3) %address.local521, align 2, !dbg !17
  %422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %420, <4 x float> %419) #9, !dbg !17
  %423 = load <4 x half>, ptr addrspace(3) %address.local522, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %421, <4 x float> %275) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %362, <4 x half> %423, <4 x float> %424) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %426 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local207), !dbg !12
  %427 = zext i32 %426 to i64, !dbg !12
  %428 = inttoptr i64 %427 to ptr addrspace(3), !dbg !12
  %address.local523 = getelementptr [2 x i8], ptr addrspace(1) %44, i64 %address.local183, !dbg !12
  %429 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %430 = ptrtoint ptr addrspace(1) %address.local523 to i64, !dbg !12
  %431 = sub i64 %430, %429, !dbg !12
  %dma.buffer.offset1118 = trunc i64 %431 to i32, !dbg !12
  %432 = trunc i64 %429 to i32, !dbg !12
  %433 = insertelement <4 x i32> poison, i32 %432, i64 0, !dbg !12
  %434 = lshr i64 %429, 32, !dbg !12
  %435 = trunc i64 %434 to i32, !dbg !12
  %436 = insertelement <4 x i32> %433, i32 %435, i64 1, !dbg !12
  %437 = insertelement <4 x i32> %436, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1119 = insertelement <4 x i32> %437, i32 131072, i64 3, !dbg !12
  %438 = ptrtoint ptr addrspace(3) %428 to i32, !dbg !12
  %dma.buffer.lds1120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %438), !dbg !12
  %439 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1120, i32 %dma.buffer.offset1118, <4 x i32> %dma.buffer.rsrc1119) #10, !dbg !12
  %440 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local211), !dbg !12
  %441 = zext i32 %440 to i64, !dbg !12
  %442 = inttoptr i64 %441 to ptr addrspace(3), !dbg !12
  %address.local524 = getelementptr [2 x i8], ptr addrspace(1) %44, i64 %address.local195, !dbg !12
  %443 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %444 = ptrtoint ptr addrspace(1) %address.local524 to i64, !dbg !12
  %445 = sub i64 %444, %443, !dbg !12
  %dma.buffer.offset1115 = trunc i64 %445 to i32, !dbg !12
  %446 = trunc i64 %443 to i32, !dbg !12
  %447 = insertelement <4 x i32> poison, i32 %446, i64 0, !dbg !12
  %448 = lshr i64 %443, 32, !dbg !12
  %449 = trunc i64 %448 to i32, !dbg !12
  %450 = insertelement <4 x i32> %447, i32 %449, i64 1, !dbg !12
  %451 = insertelement <4 x i32> %450, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc1116 = insertelement <4 x i32> %451, i32 131072, i64 3, !dbg !12
  %452 = ptrtoint ptr addrspace(3) %442 to i32, !dbg !12
  %dma.buffer.lds1117 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %452), !dbg !12
  %453 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1117, i32 %dma.buffer.offset1115, <4 x i32> %dma.buffer.rsrc1116) #10, !dbg !12
  %454 = add i64 %152, 3072, !dbg !16
  %455 = icmp slt i64 %454, 0, !dbg !16
  %456 = sub i64 -3073, %152, !dbg !16
  %457 = select i1 %455, i64 %456, i64 %454, !dbg !16
  %458 = sdiv i64 %457, 4096, !dbg !16
  %.lobit6 = ashr i64 %454, 63, !dbg !16
  %459 = xor i64 %458, %.lobit6, !dbg !16
  %460 = add nsw i64 %159, 96, !dbg !16
  %461 = icmp slt i64 %159, -96, !dbg !16
  %462 = sub nsw i64 -97, %159, !dbg !16
  %463 = select i1 %461, i64 %462, i64 %460, !dbg !16
  %464 = sdiv i64 %463, 128, !dbg !16
  %465 = sext i1 %461 to i64, !dbg !16
  %466 = xor i64 %464, %465, !dbg !16
  %467 = shl nsw i64 %466, 7, !dbg !16
  %468 = sub i64 %40, %467, !dbg !16
  %469 = shl i64 %459, 19, !dbg !16
  %470 = and i64 %469, 16252928, !dbg !16
  %471 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %470, !dbg !16
  %472 = getelementptr [2 x i8], ptr addrspace(1) %471, i64 %85, !dbg !16
  %473 = getelementptr [2 x i8], ptr addrspace(1) %472, i64 %468, !dbg !16
  %474 = getelementptr i8, ptr addrspace(1) %473, i64 192, !dbg !16
  %475 = getelementptr [2 x i8], ptr addrspace(1) %474, i64 %91, !dbg !16
  %476 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local215), !dbg !16
  %477 = zext i32 %476 to i64, !dbg !16
  %478 = inttoptr i64 %477 to ptr addrspace(3), !dbg !16
  %address.local525 = getelementptr [2 x i8], ptr addrspace(1) %475, i64 %address.local183, !dbg !16
  %479 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %480 = ptrtoint ptr addrspace(1) %address.local525 to i64, !dbg !16
  %481 = sub i64 %480, %479, !dbg !16
  %dma.buffer.offset1112 = trunc i64 %481 to i32, !dbg !16
  %482 = trunc i64 %479 to i32, !dbg !16
  %483 = insertelement <4 x i32> poison, i32 %482, i64 0, !dbg !16
  %484 = lshr i64 %479, 32, !dbg !16
  %485 = trunc i64 %484 to i32, !dbg !16
  %486 = insertelement <4 x i32> %483, i32 %485, i64 1, !dbg !16
  %487 = insertelement <4 x i32> %486, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1113 = insertelement <4 x i32> %487, i32 131072, i64 3, !dbg !16
  %488 = ptrtoint ptr addrspace(3) %478 to i32, !dbg !16
  %dma.buffer.lds1114 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %488), !dbg !16
  %489 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1114, i32 %dma.buffer.offset1112, <4 x i32> %dma.buffer.rsrc1113) #10, !dbg !16
  %490 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local219), !dbg !16
  %491 = zext i32 %490 to i64, !dbg !16
  %492 = inttoptr i64 %491 to ptr addrspace(3), !dbg !16
  %address.local526 = getelementptr [2 x i8], ptr addrspace(1) %475, i64 %address.local195, !dbg !16
  %493 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %494 = ptrtoint ptr addrspace(1) %address.local526 to i64, !dbg !16
  %495 = sub i64 %494, %493, !dbg !16
  %dma.buffer.offset1109 = trunc i64 %495 to i32, !dbg !16
  %496 = trunc i64 %493 to i32, !dbg !16
  %497 = insertelement <4 x i32> poison, i32 %496, i64 0, !dbg !16
  %498 = lshr i64 %493, 32, !dbg !16
  %499 = trunc i64 %498 to i32, !dbg !16
  %500 = insertelement <4 x i32> %497, i32 %499, i64 1, !dbg !16
  %501 = insertelement <4 x i32> %500, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1110 = insertelement <4 x i32> %501, i32 131072, i64 3, !dbg !16
  %502 = ptrtoint ptr addrspace(3) %492 to i32, !dbg !16
  %dma.buffer.lds1111 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %502), !dbg !16
  %503 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1111, i32 %dma.buffer.offset1109, <4 x i32> %dma.buffer.rsrc1110) #10, !dbg !16
  %504 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !13
  %505 = fmul <4 x half> %504, splat (half 0xH3015), !dbg !13
  %506 = load <4 x half>, ptr addrspace(3) %address.local247, align 2, !dbg !13
  %507 = load <4 x half>, ptr addrspace(3) %address.local261, align 2, !dbg !13
  %508 = fmul <4 x half> %506, splat (half 0xH3015), !dbg !13
  %509 = fmul <4 x half> %507, splat (half 0xH3015), !dbg !13
  %510 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !13
  %511 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !17
  %512 = fmul <4 x half> %510, splat (half 0xH3015), !dbg !13
  %513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %505, <4 x half> %511, <4 x float> %366) #9, !dbg !17
  %514 = load <4 x half>, ptr addrspace(3) %address.local302, align 2, !dbg !17
  %515 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !17
  %516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %514, <4 x float> %513) #9, !dbg !17
  %517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %505, <4 x half> %515, <4 x float> %370) #9, !dbg !17
  %518 = load <4 x half>, ptr addrspace(3) %address.local330, align 2, !dbg !17
  %519 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !17
  %520 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %518, <4 x float> %517) #9, !dbg !17
  %521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %505, <4 x half> %519, <4 x float> %374) #9, !dbg !17
  %522 = load <4 x half>, ptr addrspace(3) %address.local358, align 2, !dbg !17
  %523 = load <4 x half>, ptr addrspace(3) %address.local372, align 2, !dbg !17
  %524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %522, <4 x float> %521) #9, !dbg !17
  %525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %505, <4 x half> %523, <4 x float> %378) #9, !dbg !17
  %526 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !17
  %527 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !17
  %528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %526, <4 x float> %525) #9, !dbg !17
  %529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %505, <4 x half> %527, <4 x float> %382) #9, !dbg !17
  %530 = load <4 x half>, ptr addrspace(3) %address.local414, align 2, !dbg !17
  %531 = load <4 x half>, ptr addrspace(3) %address.local428, align 2, !dbg !17
  %532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %530, <4 x float> %529) #9, !dbg !17
  %533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %505, <4 x half> %531, <4 x float> %386) #9, !dbg !17
  %534 = load <4 x half>, ptr addrspace(3) %address.local442, align 2, !dbg !17
  %535 = load <4 x half>, ptr addrspace(3) %address.local456, align 2, !dbg !17
  %536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %534, <4 x float> %533) #9, !dbg !17
  %537 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %505, <4 x half> %535, <4 x float> %390) #9, !dbg !17
  %538 = load <4 x half>, ptr addrspace(3) %address.local470, align 2, !dbg !17
  %539 = load <4 x half>, ptr addrspace(3) %address.local484, align 2, !dbg !17
  %540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %538, <4 x float> %537) #9, !dbg !17
  %541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %505, <4 x half> %539, <4 x float> %394) #9, !dbg !17
  %542 = load <4 x half>, ptr addrspace(3) %address.local498, align 2, !dbg !17
  %543 = load <4 x half>, ptr addrspace(3) %address.local288, align 2, !dbg !17
  %544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %508, <4 x half> %542, <4 x float> %541) #9, !dbg !17
  %545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %509, <4 x half> %543, <4 x float> %398) #9, !dbg !17
  %546 = load <4 x half>, ptr addrspace(3) %address.local302, align 2, !dbg !17
  %547 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !17
  %548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %512, <4 x half> %546, <4 x float> %545) #9, !dbg !17
  %549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %509, <4 x half> %547, <4 x float> %402) #9, !dbg !17
  %550 = load <4 x half>, ptr addrspace(3) %address.local330, align 2, !dbg !17
  %551 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !17
  %552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %512, <4 x half> %550, <4 x float> %549) #9, !dbg !17
  %553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %509, <4 x half> %551, <4 x float> %406) #9, !dbg !17
  %554 = load <4 x half>, ptr addrspace(3) %address.local358, align 2, !dbg !17
  %555 = load <4 x half>, ptr addrspace(3) %address.local372, align 2, !dbg !17
  %556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %512, <4 x half> %554, <4 x float> %553) #9, !dbg !17
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %509, <4 x half> %555, <4 x float> %410) #9, !dbg !17
  %558 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !17
  %559 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !17
  %560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %512, <4 x half> %558, <4 x float> %557) #9, !dbg !17
  %561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %509, <4 x half> %559, <4 x float> %414) #9, !dbg !17
  %562 = load <4 x half>, ptr addrspace(3) %address.local414, align 2, !dbg !17
  %563 = load <4 x half>, ptr addrspace(3) %address.local428, align 2, !dbg !17
  %564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %512, <4 x half> %562, <4 x float> %561) #9, !dbg !17
  %565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %509, <4 x half> %563, <4 x float> %418) #9, !dbg !17
  %566 = load <4 x half>, ptr addrspace(3) %address.local442, align 2, !dbg !17
  %567 = load <4 x half>, ptr addrspace(3) %address.local456, align 2, !dbg !17
  %568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %512, <4 x half> %566, <4 x float> %565) #9, !dbg !17
  %569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %509, <4 x half> %567, <4 x float> %422) #9, !dbg !17
  %570 = load <4 x half>, ptr addrspace(3) %address.local470, align 2, !dbg !17
  %571 = load <4 x half>, ptr addrspace(3) %address.local484, align 2, !dbg !17
  %572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %512, <4 x half> %570, <4 x float> %569) #9, !dbg !17
  %573 = load <4 x half>, ptr addrspace(3) %address.local498, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %509, <4 x half> %571, <4 x float> %425) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %575 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %512, <4 x half> %573, <4 x float> %574) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %576 = add nsw i64 %85, %8, !dbg !18
  %577 = shl i64 %576, 19, !dbg !18
  %578 = and i64 %577, 16252928, !dbg !18
  %579 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %578, !dbg !18
  %580 = getelementptr [2 x i8], ptr addrspace(1) %579, i64 %91, !dbg !18
  %581 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local178), !dbg !18
  %582 = zext i32 %581 to i64, !dbg !18
  %583 = inttoptr i64 %582 to ptr addrspace(3), !dbg !18
  %address.local527 = shl nuw nsw i64 %address.local, 5, !dbg !18
  %address.local528 = and i64 %address.local527, 8064, !dbg !18
  %address.local529 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local530 = and i64 %address.local529, 24, !dbg !18
  %address.local531 = or disjoint i64 %address.local528, %address.local530, !dbg !18
  %address.local532 = getelementptr [2 x i8], ptr addrspace(1) %580, i64 %address.local531, !dbg !18
  %584 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %585 = ptrtoint ptr addrspace(1) %address.local532 to i64, !dbg !18
  %586 = sub i64 %585, %584, !dbg !18
  %dma.buffer.offset1106 = trunc i64 %586 to i32, !dbg !18
  %587 = trunc i64 %584 to i32, !dbg !18
  %588 = insertelement <4 x i32> poison, i32 %587, i64 0, !dbg !18
  %589 = lshr i64 %584, 32, !dbg !18
  %590 = trunc i64 %589 to i32, !dbg !18
  %591 = insertelement <4 x i32> %588, i32 %590, i64 1, !dbg !18
  %592 = insertelement <4 x i32> %591, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1107 = insertelement <4 x i32> %592, i32 131072, i64 3, !dbg !18
  %593 = ptrtoint ptr addrspace(3) %583 to i32, !dbg !18
  %dma.buffer.lds1108 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %593), !dbg !18
  %594 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1108, i32 %dma.buffer.offset1106, <4 x i32> %dma.buffer.rsrc1107) #10, !dbg !18
  %595 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local190), !dbg !18
  %596 = zext i32 %595 to i64, !dbg !18
  %597 = inttoptr i64 %596 to ptr addrspace(3), !dbg !18
  %address.local533 = shl nuw nsw i64 %address.local185, 5, !dbg !18
  %address.local534 = and i64 %address.local533, 16256, !dbg !18
  %address.local535 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local536 = and i64 %address.local535, 24, !dbg !18
  %address.local537 = or disjoint i64 %address.local534, %address.local536, !dbg !18
  %address.local538 = getelementptr [2 x i8], ptr addrspace(1) %580, i64 %address.local537, !dbg !18
  %598 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %599 = ptrtoint ptr addrspace(1) %address.local538 to i64, !dbg !18
  %600 = sub i64 %599, %598, !dbg !18
  %dma.buffer.offset1103 = trunc i64 %600 to i32, !dbg !18
  %601 = trunc i64 %598 to i32, !dbg !18
  %602 = insertelement <4 x i32> poison, i32 %601, i64 0, !dbg !18
  %603 = lshr i64 %598, 32, !dbg !18
  %604 = trunc i64 %603 to i32, !dbg !18
  %605 = insertelement <4 x i32> %602, i32 %604, i64 1, !dbg !18
  %606 = insertelement <4 x i32> %605, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1104 = insertelement <4 x i32> %606, i32 131072, i64 3, !dbg !18
  %607 = ptrtoint ptr addrspace(3) %597 to i32, !dbg !18
  %dma.buffer.lds1105 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %607), !dbg !18
  %608 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1105, i32 %dma.buffer.offset1103, <4 x i32> %dma.buffer.rsrc1104) #10, !dbg !18
  %609 = load <4 x half>, ptr addrspace(3) %address.local503, align 2, !dbg !13
  %610 = fmul <4 x half> %609, splat (half 0xH3015), !dbg !13
  %611 = load <4 x half>, ptr addrspace(3) %address.local504, align 2, !dbg !13
  %612 = load <4 x half>, ptr addrspace(3) %address.local505, align 2, !dbg !13
  %613 = fmul <4 x half> %611, splat (half 0xH3015), !dbg !13
  %614 = fmul <4 x half> %612, splat (half 0xH3015), !dbg !13
  %615 = load <4 x half>, ptr addrspace(3) %address.local506, align 2, !dbg !13
  %616 = load <4 x half>, ptr addrspace(3) %address.local507, align 2, !dbg !17
  %617 = fmul <4 x half> %615, splat (half 0xH3015), !dbg !13
  %618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %610, <4 x half> %616, <4 x float> %516) #9, !dbg !17
  %619 = load <4 x half>, ptr addrspace(3) %address.local508, align 2, !dbg !17
  %620 = load <4 x half>, ptr addrspace(3) %address.local509, align 2, !dbg !17
  %621 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %619, <4 x float> %618) #9, !dbg !17
  %622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %610, <4 x half> %620, <4 x float> %520) #9, !dbg !17
  %623 = load <4 x half>, ptr addrspace(3) %address.local510, align 2, !dbg !17
  %624 = load <4 x half>, ptr addrspace(3) %address.local511, align 2, !dbg !17
  %625 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %623, <4 x float> %622) #9, !dbg !17
  %626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %610, <4 x half> %624, <4 x float> %524) #9, !dbg !17
  %627 = load <4 x half>, ptr addrspace(3) %address.local512, align 2, !dbg !17
  %628 = load <4 x half>, ptr addrspace(3) %address.local513, align 2, !dbg !17
  %629 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %627, <4 x float> %626) #9, !dbg !17
  %630 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %610, <4 x half> %628, <4 x float> %528) #9, !dbg !17
  %631 = load <4 x half>, ptr addrspace(3) %address.local514, align 2, !dbg !17
  %632 = load <4 x half>, ptr addrspace(3) %address.local515, align 2, !dbg !17
  %633 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %631, <4 x float> %630) #9, !dbg !17
  %634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %610, <4 x half> %632, <4 x float> %532) #9, !dbg !17
  %635 = load <4 x half>, ptr addrspace(3) %address.local516, align 2, !dbg !17
  %636 = load <4 x half>, ptr addrspace(3) %address.local517, align 2, !dbg !17
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %635, <4 x float> %634) #9, !dbg !17
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %610, <4 x half> %636, <4 x float> %536) #9, !dbg !17
  %639 = load <4 x half>, ptr addrspace(3) %address.local518, align 2, !dbg !17
  %640 = load <4 x half>, ptr addrspace(3) %address.local519, align 2, !dbg !17
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %639, <4 x float> %638) #9, !dbg !17
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %610, <4 x half> %640, <4 x float> %540) #9, !dbg !17
  %643 = load <4 x half>, ptr addrspace(3) %address.local520, align 2, !dbg !17
  %644 = load <4 x half>, ptr addrspace(3) %address.local521, align 2, !dbg !17
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %643, <4 x float> %642) #9, !dbg !17
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %610, <4 x half> %644, <4 x float> %544) #9, !dbg !17
  %647 = load <4 x half>, ptr addrspace(3) %address.local522, align 2, !dbg !17
  %648 = load <4 x half>, ptr addrspace(3) %address.local507, align 2, !dbg !17
  %649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %613, <4 x half> %647, <4 x float> %646) #9, !dbg !17
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %614, <4 x half> %648, <4 x float> %548) #9, !dbg !17
  %651 = load <4 x half>, ptr addrspace(3) %address.local508, align 2, !dbg !17
  %652 = load <4 x half>, ptr addrspace(3) %address.local509, align 2, !dbg !17
  %653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %617, <4 x half> %651, <4 x float> %650) #9, !dbg !17
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %614, <4 x half> %652, <4 x float> %552) #9, !dbg !17
  %655 = load <4 x half>, ptr addrspace(3) %address.local510, align 2, !dbg !17
  %656 = load <4 x half>, ptr addrspace(3) %address.local511, align 2, !dbg !17
  %657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %617, <4 x half> %655, <4 x float> %654) #9, !dbg !17
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %614, <4 x half> %656, <4 x float> %556) #9, !dbg !17
  %659 = load <4 x half>, ptr addrspace(3) %address.local512, align 2, !dbg !17
  %660 = load <4 x half>, ptr addrspace(3) %address.local513, align 2, !dbg !17
  %661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %617, <4 x half> %659, <4 x float> %658) #9, !dbg !17
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %614, <4 x half> %660, <4 x float> %560) #9, !dbg !17
  %663 = load <4 x half>, ptr addrspace(3) %address.local514, align 2, !dbg !17
  %664 = load <4 x half>, ptr addrspace(3) %address.local515, align 2, !dbg !17
  %665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %617, <4 x half> %663, <4 x float> %662) #9, !dbg !17
  %666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %614, <4 x half> %664, <4 x float> %564) #9, !dbg !17
  %667 = load <4 x half>, ptr addrspace(3) %address.local516, align 2, !dbg !17
  %668 = load <4 x half>, ptr addrspace(3) %address.local517, align 2, !dbg !17
  %669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %617, <4 x half> %667, <4 x float> %666) #9, !dbg !17
  %670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %614, <4 x half> %668, <4 x float> %568) #9, !dbg !17
  %671 = load <4 x half>, ptr addrspace(3) %address.local518, align 2, !dbg !17
  %672 = load <4 x half>, ptr addrspace(3) %address.local519, align 2, !dbg !17
  %673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %617, <4 x half> %671, <4 x float> %670) #9, !dbg !17
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %614, <4 x half> %672, <4 x float> %572) #9, !dbg !17
  %675 = load <4 x half>, ptr addrspace(3) %address.local520, align 2, !dbg !17
  %676 = load <4 x half>, ptr addrspace(3) %address.local521, align 2, !dbg !17
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %617, <4 x half> %675, <4 x float> %674) #9, !dbg !17
  %678 = load <4 x half>, ptr addrspace(3) %address.local522, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %614, <4 x half> %676, <4 x float> %575) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %680 = or disjoint i64 %54, 127, !dbg !14
  %681 = icmp uge i64 %11, %680, !dbg !14
  %682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %617, <4 x half> %678, <4 x float> %679) #9, !dbg !17
  %683 = and i1 %31, %681, !dbg !14
  br i1 %683, label %684, label %685, !dbg !14

684:                                              ; preds = %53
  br label %829, !dbg !14

685:                                              ; preds = %53
  %686 = add nsw i64 %11, %27, !dbg !14
  %687 = shl nuw nsw i64 %28, 4, !dbg !14
  %688 = add nsw i64 %686, %687, !dbg !14
  %689 = and i32 %26, 240, !dbg !14
  %index.compact = zext nneg i32 %689 to i64, !dbg !14
  %690 = sub nsw i64 %688, %index.compact, !dbg !14
  %691 = and i64 %27, 48, !dbg !14
  %692 = or disjoint i64 %54, %691, !dbg !14
  %693 = add nsw i64 %690, 1, !dbg !24
  %694 = lshr i64 %693, 32, !dbg !25
  %compare.high = trunc i64 %694 to i32, !dbg !25
  %695 = lshr i64 %692, 32, !dbg !25
  %compare.high832 = trunc i64 %695 to i32, !dbg !25
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high832, !dbg !25
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high832, !dbg !25
  %compare.low.base = trunc i64 %693 to i32, !dbg !25
  %compare.low.base833 = trunc i64 %692 to i32, !dbg !25
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base833, !dbg !25
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !25
  %696 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 1) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !25
  %compare.low834 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !25
  %compare.result835 = select i1 %compare.high.equal, i1 %compare.low834, i1 %compare.high.order, !dbg !25
  %697 = select i1 %compare.result835, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base836 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 2) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local837 = or i32 %compare.local.base836, 2, !dbg !25
  %compare.low838 = icmp ugt i32 %compare.low.base, %compare.local837, !dbg !25
  %compare.result839 = select i1 %compare.high.equal, i1 %compare.low838, i1 %compare.high.order, !dbg !25
  %698 = select i1 %compare.result839, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base840 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 3) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local841 = or i32 %compare.local.base840, 3, !dbg !25
  %compare.low842 = icmp ugt i32 %compare.low.base, %compare.local841, !dbg !25
  %compare.result843 = select i1 %compare.high.equal, i1 %compare.low842, i1 %compare.high.order, !dbg !25
  %699 = select i1 %compare.result843, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base844 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 4) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local845 = or i32 %compare.local.base844, 4, !dbg !25
  %compare.low846 = icmp ugt i32 %compare.low.base, %compare.local845, !dbg !25
  %compare.result847 = select i1 %compare.high.equal, i1 %compare.low846, i1 %compare.high.order, !dbg !25
  %700 = select i1 %compare.result847, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base848 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 5) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local849 = or i32 %compare.local.base848, 5, !dbg !25
  %compare.low850 = icmp ugt i32 %compare.low.base, %compare.local849, !dbg !25
  %compare.result851 = select i1 %compare.high.equal, i1 %compare.low850, i1 %compare.high.order, !dbg !25
  %701 = select i1 %compare.result851, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base852 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 6) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local853 = or i32 %compare.local.base852, 6, !dbg !25
  %compare.low854 = icmp ugt i32 %compare.low.base, %compare.local853, !dbg !25
  %compare.result855 = select i1 %compare.high.equal, i1 %compare.low854, i1 %compare.high.order, !dbg !25
  %702 = select i1 %compare.result855, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base856 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 7) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local857 = or i32 %compare.local.base856, 7, !dbg !25
  %compare.low858 = icmp ugt i32 %compare.low.base, %compare.local857, !dbg !25
  %compare.result859 = select i1 %compare.high.equal, i1 %compare.low858, i1 %compare.high.order, !dbg !25
  %703 = select i1 %compare.result859, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base860 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 8) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local861 = or i32 %compare.local.base860, 8, !dbg !25
  %compare.low862 = icmp ugt i32 %compare.low.base, %compare.local861, !dbg !25
  %compare.result863 = select i1 %compare.high.equal, i1 %compare.low862, i1 %compare.high.order, !dbg !25
  %704 = select i1 %compare.result863, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base864 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 9) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local865 = or i32 %compare.local.base864, 9, !dbg !25
  %compare.low866 = icmp ugt i32 %compare.low.base, %compare.local865, !dbg !25
  %compare.result867 = select i1 %compare.high.equal, i1 %compare.low866, i1 %compare.high.order, !dbg !25
  %705 = select i1 %compare.result867, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base868 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 10) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local869 = or i32 %compare.local.base868, 10, !dbg !25
  %compare.low870 = icmp ugt i32 %compare.low.base, %compare.local869, !dbg !25
  %compare.result871 = select i1 %compare.high.equal, i1 %compare.low870, i1 %compare.high.order, !dbg !25
  %706 = select i1 %compare.result871, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base872 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 11) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local873 = or i32 %compare.local.base872, 11, !dbg !25
  %compare.low874 = icmp ugt i32 %compare.low.base, %compare.local873, !dbg !25
  %compare.result875 = select i1 %compare.high.equal, i1 %compare.low874, i1 %compare.high.order, !dbg !25
  %707 = select i1 %compare.result875, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base876 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 12) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local877 = or i32 %compare.local.base876, 12, !dbg !25
  %compare.low878 = icmp ugt i32 %compare.low.base, %compare.local877, !dbg !25
  %compare.result879 = select i1 %compare.high.equal, i1 %compare.low878, i1 %compare.high.order, !dbg !25
  %708 = select i1 %compare.result879, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base880 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 13) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local881 = or i32 %compare.local.base880, 13, !dbg !25
  %compare.low882 = icmp ugt i32 %compare.low.base, %compare.local881, !dbg !25
  %compare.result883 = select i1 %compare.high.equal, i1 %compare.low882, i1 %compare.high.order, !dbg !25
  %709 = select i1 %compare.result883, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base884 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 14) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local885 = or i32 %compare.local.base884, 14, !dbg !25
  %compare.low886 = icmp ugt i32 %compare.low.base, %compare.local885, !dbg !25
  %compare.result887 = select i1 %compare.high.equal, i1 %compare.low886, i1 %compare.high.order, !dbg !25
  %710 = select i1 %compare.result887, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base888 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 15) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local889 = or i32 %compare.local.base888, 15, !dbg !25
  %compare.low890 = icmp ugt i32 %compare.low.base, %compare.local889, !dbg !25
  %compare.result891 = select i1 %compare.high.equal, i1 %compare.low890, i1 %compare.high.order, !dbg !25
  %711 = select i1 %compare.result891, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base892 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 16) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local893 = or i32 %compare.local.base892, 64, !dbg !25
  %compare.low894 = icmp ugt i32 %compare.low.base, %compare.local893, !dbg !25
  %compare.result895 = select i1 %compare.high.equal, i1 %compare.low894, i1 %compare.high.order, !dbg !25
  %712 = select i1 %compare.result895, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base896 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 17) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local897 = or i32 %compare.local.base896, 65, !dbg !25
  %compare.low898 = icmp ugt i32 %compare.low.base, %compare.local897, !dbg !25
  %compare.result899 = select i1 %compare.high.equal, i1 %compare.low898, i1 %compare.high.order, !dbg !25
  %713 = select i1 %compare.result899, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base900 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 18) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local901 = or i32 %compare.local.base900, 66, !dbg !25
  %compare.low902 = icmp ugt i32 %compare.low.base, %compare.local901, !dbg !25
  %compare.result903 = select i1 %compare.high.equal, i1 %compare.low902, i1 %compare.high.order, !dbg !25
  %714 = select i1 %compare.result903, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base904 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 19) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local905 = or i32 %compare.local.base904, 67, !dbg !25
  %compare.low906 = icmp ugt i32 %compare.low.base, %compare.local905, !dbg !25
  %compare.result907 = select i1 %compare.high.equal, i1 %compare.low906, i1 %compare.high.order, !dbg !25
  %715 = select i1 %compare.result907, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base908 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 20) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local909 = or i32 %compare.local.base908, 68, !dbg !25
  %compare.low910 = icmp ugt i32 %compare.low.base, %compare.local909, !dbg !25
  %compare.result911 = select i1 %compare.high.equal, i1 %compare.low910, i1 %compare.high.order, !dbg !25
  %716 = select i1 %compare.result911, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base912 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 21) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local913 = or i32 %compare.local.base912, 69, !dbg !25
  %compare.low914 = icmp ugt i32 %compare.low.base, %compare.local913, !dbg !25
  %compare.result915 = select i1 %compare.high.equal, i1 %compare.low914, i1 %compare.high.order, !dbg !25
  %717 = select i1 %compare.result915, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base916 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 22) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local917 = or i32 %compare.local.base916, 70, !dbg !25
  %compare.low918 = icmp ugt i32 %compare.low.base, %compare.local917, !dbg !25
  %compare.result919 = select i1 %compare.high.equal, i1 %compare.low918, i1 %compare.high.order, !dbg !25
  %718 = select i1 %compare.result919, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base920 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 23) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local921 = or i32 %compare.local.base920, 71, !dbg !25
  %compare.low922 = icmp ugt i32 %compare.low.base, %compare.local921, !dbg !25
  %compare.result923 = select i1 %compare.high.equal, i1 %compare.low922, i1 %compare.high.order, !dbg !25
  %719 = select i1 %compare.result923, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base924 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 24) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local925 = or i32 %compare.local.base924, 72, !dbg !25
  %compare.low926 = icmp ugt i32 %compare.low.base, %compare.local925, !dbg !25
  %compare.result927 = select i1 %compare.high.equal, i1 %compare.low926, i1 %compare.high.order, !dbg !25
  %720 = select i1 %compare.result927, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base928 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 25) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local929 = or i32 %compare.local.base928, 73, !dbg !25
  %compare.low930 = icmp ugt i32 %compare.low.base, %compare.local929, !dbg !25
  %compare.result931 = select i1 %compare.high.equal, i1 %compare.low930, i1 %compare.high.order, !dbg !25
  %721 = select i1 %compare.result931, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base932 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 26) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local933 = or i32 %compare.local.base932, 74, !dbg !25
  %compare.low934 = icmp ugt i32 %compare.low.base, %compare.local933, !dbg !25
  %compare.result935 = select i1 %compare.high.equal, i1 %compare.low934, i1 %compare.high.order, !dbg !25
  %722 = select i1 %compare.result935, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base936 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 27) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local937 = or i32 %compare.local.base936, 75, !dbg !25
  %compare.low938 = icmp ugt i32 %compare.low.base, %compare.local937, !dbg !25
  %compare.result939 = select i1 %compare.high.equal, i1 %compare.low938, i1 %compare.high.order, !dbg !25
  %723 = select i1 %compare.result939, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base940 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 28) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local941 = or i32 %compare.local.base940, 76, !dbg !25
  %compare.low942 = icmp ugt i32 %compare.low.base, %compare.local941, !dbg !25
  %compare.result943 = select i1 %compare.high.equal, i1 %compare.low942, i1 %compare.high.order, !dbg !25
  %724 = select i1 %compare.result943, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base944 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 29) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local945 = or i32 %compare.local.base944, 77, !dbg !25
  %compare.low946 = icmp ugt i32 %compare.low.base, %compare.local945, !dbg !25
  %compare.result947 = select i1 %compare.high.equal, i1 %compare.low946, i1 %compare.high.order, !dbg !25
  %725 = select i1 %compare.result947, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base948 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 30) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local949 = or i32 %compare.local.base948, 78, !dbg !25
  %compare.low950 = icmp ugt i32 %compare.low.base, %compare.local949, !dbg !25
  %compare.result951 = select i1 %compare.high.equal, i1 %compare.low950, i1 %compare.high.order, !dbg !25
  %726 = select i1 %compare.result951, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base952 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 31) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local953 = or i32 %compare.local.base952, 79, !dbg !25
  %compare.low954 = icmp ugt i32 %compare.low.base, %compare.local953, !dbg !25
  %compare.result955 = select i1 %compare.high.equal, i1 %compare.low954, i1 %compare.high.order, !dbg !25
  %727 = select i1 %compare.result955, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %728 = add nsw i64 %690, 65, !dbg !24
  %729 = lshr i64 %728, 32, !dbg !25
  %compare.high956 = trunc i64 %729 to i32, !dbg !25
  %730 = lshr i64 %692, 32, !dbg !25
  %compare.high957 = trunc i64 %730 to i32, !dbg !25
  %compare.high.equal958 = icmp eq i32 %compare.high956, %compare.high957, !dbg !25
  %compare.high.order959 = icmp ugt i32 %compare.high956, %compare.high957, !dbg !25
  %compare.low.base960 = trunc i64 %728 to i32, !dbg !25
  %compare.low.base961 = trunc i64 %692 to i32, !dbg !25
  %compare.low962 = icmp ugt i32 %compare.low.base960, %compare.low.base961, !dbg !25
  %compare.result963 = select i1 %compare.high.equal958, i1 %compare.low962, i1 %compare.high.order959, !dbg !25
  %731 = select i1 %compare.result963, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base964 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 32) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local965 = or i32 %compare.local.base964, 1, !dbg !25
  %compare.low966 = icmp ugt i32 %compare.low.base960, %compare.local965, !dbg !25
  %compare.result967 = select i1 %compare.high.equal958, i1 %compare.low966, i1 %compare.high.order959, !dbg !25
  %732 = select i1 %compare.result967, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base968 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 33) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local969 = or i32 %compare.local.base968, 2, !dbg !25
  %compare.low970 = icmp ugt i32 %compare.low.base960, %compare.local969, !dbg !25
  %compare.result971 = select i1 %compare.high.equal958, i1 %compare.low970, i1 %compare.high.order959, !dbg !25
  %733 = select i1 %compare.result971, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base972 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 34) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local973 = or i32 %compare.local.base972, 3, !dbg !25
  %compare.low974 = icmp ugt i32 %compare.low.base960, %compare.local973, !dbg !25
  %compare.result975 = select i1 %compare.high.equal958, i1 %compare.low974, i1 %compare.high.order959, !dbg !25
  %734 = select i1 %compare.result975, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base976 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 35) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local977 = or i32 %compare.local.base976, 4, !dbg !25
  %compare.low978 = icmp ugt i32 %compare.low.base960, %compare.local977, !dbg !25
  %compare.result979 = select i1 %compare.high.equal958, i1 %compare.low978, i1 %compare.high.order959, !dbg !25
  %735 = select i1 %compare.result979, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base980 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 36) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local981 = or i32 %compare.local.base980, 5, !dbg !25
  %compare.low982 = icmp ugt i32 %compare.low.base960, %compare.local981, !dbg !25
  %compare.result983 = select i1 %compare.high.equal958, i1 %compare.low982, i1 %compare.high.order959, !dbg !25
  %736 = select i1 %compare.result983, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base984 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 37) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local985 = or i32 %compare.local.base984, 6, !dbg !25
  %compare.low986 = icmp ugt i32 %compare.low.base960, %compare.local985, !dbg !25
  %compare.result987 = select i1 %compare.high.equal958, i1 %compare.low986, i1 %compare.high.order959, !dbg !25
  %737 = select i1 %compare.result987, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base988 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 38) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local989 = or i32 %compare.local.base988, 7, !dbg !25
  %compare.low990 = icmp ugt i32 %compare.low.base960, %compare.local989, !dbg !25
  %compare.result991 = select i1 %compare.high.equal958, i1 %compare.low990, i1 %compare.high.order959, !dbg !25
  %738 = select i1 %compare.result991, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base992 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 39) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local993 = or i32 %compare.local.base992, 8, !dbg !25
  %compare.low994 = icmp ugt i32 %compare.low.base960, %compare.local993, !dbg !25
  %compare.result995 = select i1 %compare.high.equal958, i1 %compare.low994, i1 %compare.high.order959, !dbg !25
  %739 = select i1 %compare.result995, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base996 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 40) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local997 = or i32 %compare.local.base996, 9, !dbg !25
  %compare.low998 = icmp ugt i32 %compare.low.base960, %compare.local997, !dbg !25
  %compare.result999 = select i1 %compare.high.equal958, i1 %compare.low998, i1 %compare.high.order959, !dbg !25
  %740 = select i1 %compare.result999, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1000 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 41) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1001 = or i32 %compare.local.base1000, 10, !dbg !25
  %compare.low1002 = icmp ugt i32 %compare.low.base960, %compare.local1001, !dbg !25
  %compare.result1003 = select i1 %compare.high.equal958, i1 %compare.low1002, i1 %compare.high.order959, !dbg !25
  %741 = select i1 %compare.result1003, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1004 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 42) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1005 = or i32 %compare.local.base1004, 11, !dbg !25
  %compare.low1006 = icmp ugt i32 %compare.low.base960, %compare.local1005, !dbg !25
  %compare.result1007 = select i1 %compare.high.equal958, i1 %compare.low1006, i1 %compare.high.order959, !dbg !25
  %742 = select i1 %compare.result1007, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1008 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 43) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1009 = or i32 %compare.local.base1008, 12, !dbg !25
  %compare.low1010 = icmp ugt i32 %compare.low.base960, %compare.local1009, !dbg !25
  %compare.result1011 = select i1 %compare.high.equal958, i1 %compare.low1010, i1 %compare.high.order959, !dbg !25
  %743 = select i1 %compare.result1011, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1012 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 44) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1013 = or i32 %compare.local.base1012, 13, !dbg !25
  %compare.low1014 = icmp ugt i32 %compare.low.base960, %compare.local1013, !dbg !25
  %compare.result1015 = select i1 %compare.high.equal958, i1 %compare.low1014, i1 %compare.high.order959, !dbg !25
  %744 = select i1 %compare.result1015, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1016 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 45) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1017 = or i32 %compare.local.base1016, 14, !dbg !25
  %compare.low1018 = icmp ugt i32 %compare.low.base960, %compare.local1017, !dbg !25
  %compare.result1019 = select i1 %compare.high.equal958, i1 %compare.low1018, i1 %compare.high.order959, !dbg !25
  %745 = select i1 %compare.result1019, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1020 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 46) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1021 = or i32 %compare.local.base1020, 15, !dbg !25
  %compare.low1022 = icmp ugt i32 %compare.low.base960, %compare.local1021, !dbg !25
  %compare.result1023 = select i1 %compare.high.equal958, i1 %compare.low1022, i1 %compare.high.order959, !dbg !25
  %746 = select i1 %compare.result1023, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1024 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 47) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1025 = or i32 %compare.local.base1024, 64, !dbg !25
  %compare.low1026 = icmp ugt i32 %compare.low.base960, %compare.local1025, !dbg !25
  %compare.result1027 = select i1 %compare.high.equal958, i1 %compare.low1026, i1 %compare.high.order959, !dbg !25
  %747 = select i1 %compare.result1027, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1028 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 48) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1029 = or i32 %compare.local.base1028, 65, !dbg !25
  %compare.low1030 = icmp ugt i32 %compare.low.base960, %compare.local1029, !dbg !25
  %compare.result1031 = select i1 %compare.high.equal958, i1 %compare.low1030, i1 %compare.high.order959, !dbg !25
  %748 = select i1 %compare.result1031, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1032 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 49) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1033 = or i32 %compare.local.base1032, 66, !dbg !25
  %compare.low1034 = icmp ugt i32 %compare.low.base960, %compare.local1033, !dbg !25
  %compare.result1035 = select i1 %compare.high.equal958, i1 %compare.low1034, i1 %compare.high.order959, !dbg !25
  %749 = select i1 %compare.result1035, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1036 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 50) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1037 = or i32 %compare.local.base1036, 67, !dbg !25
  %compare.low1038 = icmp ugt i32 %compare.low.base960, %compare.local1037, !dbg !25
  %compare.result1039 = select i1 %compare.high.equal958, i1 %compare.low1038, i1 %compare.high.order959, !dbg !25
  %750 = select i1 %compare.result1039, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1040 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 51) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1041 = or i32 %compare.local.base1040, 68, !dbg !25
  %compare.low1042 = icmp ugt i32 %compare.low.base960, %compare.local1041, !dbg !25
  %compare.result1043 = select i1 %compare.high.equal958, i1 %compare.low1042, i1 %compare.high.order959, !dbg !25
  %751 = select i1 %compare.result1043, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1044 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 52) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1045 = or i32 %compare.local.base1044, 69, !dbg !25
  %compare.low1046 = icmp ugt i32 %compare.low.base960, %compare.local1045, !dbg !25
  %compare.result1047 = select i1 %compare.high.equal958, i1 %compare.low1046, i1 %compare.high.order959, !dbg !25
  %752 = select i1 %compare.result1047, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1048 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 53) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1049 = or i32 %compare.local.base1048, 70, !dbg !25
  %compare.low1050 = icmp ugt i32 %compare.low.base960, %compare.local1049, !dbg !25
  %compare.result1051 = select i1 %compare.high.equal958, i1 %compare.low1050, i1 %compare.high.order959, !dbg !25
  %753 = select i1 %compare.result1051, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1052 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 54) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1053 = or i32 %compare.local.base1052, 71, !dbg !25
  %compare.low1054 = icmp ugt i32 %compare.low.base960, %compare.local1053, !dbg !25
  %compare.result1055 = select i1 %compare.high.equal958, i1 %compare.low1054, i1 %compare.high.order959, !dbg !25
  %754 = select i1 %compare.result1055, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1056 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 55) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1057 = or i32 %compare.local.base1056, 72, !dbg !25
  %compare.low1058 = icmp ugt i32 %compare.low.base960, %compare.local1057, !dbg !25
  %compare.result1059 = select i1 %compare.high.equal958, i1 %compare.low1058, i1 %compare.high.order959, !dbg !25
  %755 = select i1 %compare.result1059, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1060 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 56) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1061 = or i32 %compare.local.base1060, 73, !dbg !25
  %compare.low1062 = icmp ugt i32 %compare.low.base960, %compare.local1061, !dbg !25
  %compare.result1063 = select i1 %compare.high.equal958, i1 %compare.low1062, i1 %compare.high.order959, !dbg !25
  %756 = select i1 %compare.result1063, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1064 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 57) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1065 = or i32 %compare.local.base1064, 74, !dbg !25
  %compare.low1066 = icmp ugt i32 %compare.low.base960, %compare.local1065, !dbg !25
  %compare.result1067 = select i1 %compare.high.equal958, i1 %compare.low1066, i1 %compare.high.order959, !dbg !25
  %757 = select i1 %compare.result1067, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1068 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 58) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1069 = or i32 %compare.local.base1068, 75, !dbg !25
  %compare.low1070 = icmp ugt i32 %compare.low.base960, %compare.local1069, !dbg !25
  %compare.result1071 = select i1 %compare.high.equal958, i1 %compare.low1070, i1 %compare.high.order959, !dbg !25
  %758 = select i1 %compare.result1071, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1072 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 59) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1073 = or i32 %compare.local.base1072, 76, !dbg !25
  %compare.low1074 = icmp ugt i32 %compare.low.base960, %compare.local1073, !dbg !25
  %compare.result1075 = select i1 %compare.high.equal958, i1 %compare.low1074, i1 %compare.high.order959, !dbg !25
  %759 = select i1 %compare.result1075, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1076 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 60) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1077 = or i32 %compare.local.base1076, 77, !dbg !25
  %compare.low1078 = icmp ugt i32 %compare.low.base960, %compare.local1077, !dbg !25
  %compare.result1079 = select i1 %compare.high.equal958, i1 %compare.low1078, i1 %compare.high.order959, !dbg !25
  %760 = select i1 %compare.result1079, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1080 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 61) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1081 = or i32 %compare.local.base1080, 78, !dbg !25
  %compare.low1082 = icmp ugt i32 %compare.low.base960, %compare.local1081, !dbg !25
  %compare.result1083 = select i1 %compare.high.equal958, i1 %compare.low1082, i1 %compare.high.order959, !dbg !25
  %761 = select i1 %compare.result1083, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1084 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base961, i32 62) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1085 = or i32 %compare.local.base1084, 79, !dbg !25
  %compare.low1086 = icmp ugt i32 %compare.low.base960, %compare.local1085, !dbg !25
  %compare.result1087 = select i1 %compare.high.equal958, i1 %compare.low1086, i1 %compare.high.order959, !dbg !25
  %762 = select i1 %compare.result1087, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %763 = insertelement <32 x float> poison, float %696, i64 0, !dbg !14
  %764 = insertelement <32 x float> %763, float %697, i64 1, !dbg !14
  %765 = insertelement <32 x float> %764, float %698, i64 2, !dbg !14
  %766 = insertelement <32 x float> %765, float %699, i64 3, !dbg !14
  %767 = insertelement <32 x float> %766, float %700, i64 4, !dbg !14
  %768 = insertelement <32 x float> %767, float %701, i64 5, !dbg !14
  %769 = insertelement <32 x float> %768, float %702, i64 6, !dbg !14
  %770 = insertelement <32 x float> %769, float %703, i64 7, !dbg !14
  %771 = insertelement <32 x float> %770, float %704, i64 8, !dbg !14
  %772 = insertelement <32 x float> %771, float %705, i64 9, !dbg !14
  %773 = insertelement <32 x float> %772, float %706, i64 10, !dbg !14
  %774 = insertelement <32 x float> %773, float %707, i64 11, !dbg !14
  %775 = insertelement <32 x float> %774, float %708, i64 12, !dbg !14
  %776 = insertelement <32 x float> %775, float %709, i64 13, !dbg !14
  %777 = insertelement <32 x float> %776, float %710, i64 14, !dbg !14
  %778 = insertelement <32 x float> %777, float %711, i64 15, !dbg !14
  %779 = insertelement <32 x float> %778, float %712, i64 16, !dbg !14
  %780 = insertelement <32 x float> %779, float %713, i64 17, !dbg !14
  %781 = insertelement <32 x float> %780, float %714, i64 18, !dbg !14
  %782 = insertelement <32 x float> %781, float %715, i64 19, !dbg !14
  %783 = insertelement <32 x float> %782, float %716, i64 20, !dbg !14
  %784 = insertelement <32 x float> %783, float %717, i64 21, !dbg !14
  %785 = insertelement <32 x float> %784, float %718, i64 22, !dbg !14
  %786 = insertelement <32 x float> %785, float %719, i64 23, !dbg !14
  %787 = insertelement <32 x float> %786, float %720, i64 24, !dbg !14
  %788 = insertelement <32 x float> %787, float %721, i64 25, !dbg !14
  %789 = insertelement <32 x float> %788, float %722, i64 26, !dbg !14
  %790 = insertelement <32 x float> %789, float %723, i64 27, !dbg !14
  %791 = insertelement <32 x float> %790, float %724, i64 28, !dbg !14
  %792 = insertelement <32 x float> %791, float %725, i64 29, !dbg !14
  %793 = insertelement <32 x float> %792, float %726, i64 30, !dbg !14
  %794 = insertelement <32 x float> %793, float %727, i64 31, !dbg !14
  %795 = insertvalue [2 x <32 x float>] poison, <32 x float> %794, 0, !dbg !14
  %796 = insertelement <32 x float> poison, float %731, i64 0, !dbg !14
  %797 = insertelement <32 x float> %796, float %732, i64 1, !dbg !14
  %798 = insertelement <32 x float> %797, float %733, i64 2, !dbg !14
  %799 = insertelement <32 x float> %798, float %734, i64 3, !dbg !14
  %800 = insertelement <32 x float> %799, float %735, i64 4, !dbg !14
  %801 = insertelement <32 x float> %800, float %736, i64 5, !dbg !14
  %802 = insertelement <32 x float> %801, float %737, i64 6, !dbg !14
  %803 = insertelement <32 x float> %802, float %738, i64 7, !dbg !14
  %804 = insertelement <32 x float> %803, float %739, i64 8, !dbg !14
  %805 = insertelement <32 x float> %804, float %740, i64 9, !dbg !14
  %806 = insertelement <32 x float> %805, float %741, i64 10, !dbg !14
  %807 = insertelement <32 x float> %806, float %742, i64 11, !dbg !14
  %808 = insertelement <32 x float> %807, float %743, i64 12, !dbg !14
  %809 = insertelement <32 x float> %808, float %744, i64 13, !dbg !14
  %810 = insertelement <32 x float> %809, float %745, i64 14, !dbg !14
  %811 = insertelement <32 x float> %810, float %746, i64 15, !dbg !14
  %812 = insertelement <32 x float> %811, float %747, i64 16, !dbg !14
  %813 = insertelement <32 x float> %812, float %748, i64 17, !dbg !14
  %814 = insertelement <32 x float> %813, float %749, i64 18, !dbg !14
  %815 = insertelement <32 x float> %814, float %750, i64 19, !dbg !14
  %816 = insertelement <32 x float> %815, float %751, i64 20, !dbg !14
  %817 = insertelement <32 x float> %816, float %752, i64 21, !dbg !14
  %818 = insertelement <32 x float> %817, float %753, i64 22, !dbg !14
  %819 = insertelement <32 x float> %818, float %754, i64 23, !dbg !14
  %820 = insertelement <32 x float> %819, float %755, i64 24, !dbg !14
  %821 = insertelement <32 x float> %820, float %756, i64 25, !dbg !14
  %822 = insertelement <32 x float> %821, float %757, i64 26, !dbg !14
  %823 = insertelement <32 x float> %822, float %758, i64 27, !dbg !14
  %824 = insertelement <32 x float> %823, float %759, i64 28, !dbg !14
  %825 = insertelement <32 x float> %824, float %760, i64 29, !dbg !14
  %826 = insertelement <32 x float> %825, float %761, i64 30, !dbg !14
  %827 = insertelement <32 x float> %826, float %762, i64 31, !dbg !14
  %828 = insertvalue [2 x <32 x float>] %795, <32 x float> %827, 1, !dbg !14
  br label %829, !dbg !14

829:                                              ; preds = %684, %685
  %830 = phi [2 x <32 x float>] [ %828, %685 ], [ zeroinitializer, %684 ], !dbg !14
  br label %831, !dbg !14

831:                                              ; preds = %829
  %832 = extractvalue [2 x <32 x float>] %830, 0, !dbg !27
  %833 = shufflevector <32 x float> %832, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %834 = fadd <4 x float> %621, %833, !dbg !27
  %835 = extractelement <4 x float> %834, i64 0, !dbg !27
  %836 = extractelement <4 x float> %834, i64 1, !dbg !27
  %837 = extractelement <4 x float> %834, i64 2, !dbg !27
  %838 = extractelement <4 x float> %834, i64 3, !dbg !27
  %839 = shufflevector <32 x float> %832, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %840 = fadd <4 x float> %625, %839, !dbg !27
  %841 = extractelement <4 x float> %840, i64 0, !dbg !27
  %842 = extractelement <4 x float> %840, i64 1, !dbg !27
  %843 = extractelement <4 x float> %840, i64 2, !dbg !27
  %844 = extractelement <4 x float> %840, i64 3, !dbg !27
  %845 = shufflevector <32 x float> %832, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %846 = fadd <4 x float> %629, %845, !dbg !27
  %847 = extractelement <4 x float> %846, i64 0, !dbg !27
  %848 = extractelement <4 x float> %846, i64 1, !dbg !27
  %849 = extractelement <4 x float> %846, i64 2, !dbg !27
  %850 = extractelement <4 x float> %846, i64 3, !dbg !27
  %851 = shufflevector <32 x float> %832, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %852 = fadd <4 x float> %633, %851, !dbg !27
  %853 = extractelement <4 x float> %852, i64 0, !dbg !27
  %854 = extractelement <4 x float> %852, i64 1, !dbg !27
  %855 = extractelement <4 x float> %852, i64 2, !dbg !27
  %856 = extractelement <4 x float> %852, i64 3, !dbg !27
  %857 = shufflevector <32 x float> %832, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %858 = fadd <4 x float> %637, %857, !dbg !27
  %859 = extractelement <4 x float> %858, i64 0, !dbg !27
  %860 = extractelement <4 x float> %858, i64 1, !dbg !27
  %861 = extractelement <4 x float> %858, i64 2, !dbg !27
  %862 = extractelement <4 x float> %858, i64 3, !dbg !27
  %863 = shufflevector <32 x float> %832, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %864 = fadd <4 x float> %641, %863, !dbg !27
  %865 = extractelement <4 x float> %864, i64 0, !dbg !27
  %866 = extractelement <4 x float> %864, i64 1, !dbg !27
  %867 = extractelement <4 x float> %864, i64 2, !dbg !27
  %868 = extractelement <4 x float> %864, i64 3, !dbg !27
  %869 = shufflevector <32 x float> %832, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %870 = fadd <4 x float> %645, %869, !dbg !27
  %871 = extractelement <4 x float> %870, i64 0, !dbg !27
  %872 = extractelement <4 x float> %870, i64 1, !dbg !27
  %873 = extractelement <4 x float> %870, i64 2, !dbg !27
  %874 = extractelement <4 x float> %870, i64 3, !dbg !27
  %875 = shufflevector <32 x float> %832, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %876 = fadd <4 x float> %649, %875, !dbg !27
  %877 = extractelement <4 x float> %876, i64 0, !dbg !27
  %878 = extractelement <4 x float> %876, i64 1, !dbg !27
  %879 = extractelement <4 x float> %876, i64 2, !dbg !27
  %880 = extractelement <4 x float> %876, i64 3, !dbg !27
  %881 = extractvalue [2 x <32 x float>] %830, 1, !dbg !27
  %882 = shufflevector <32 x float> %881, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %883 = fadd <4 x float> %653, %882, !dbg !27
  %884 = extractelement <4 x float> %883, i64 0, !dbg !27
  %885 = extractelement <4 x float> %883, i64 1, !dbg !27
  %886 = extractelement <4 x float> %883, i64 2, !dbg !27
  %887 = extractelement <4 x float> %883, i64 3, !dbg !27
  %888 = shufflevector <32 x float> %881, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %889 = fadd <4 x float> %657, %888, !dbg !27
  %890 = extractelement <4 x float> %889, i64 0, !dbg !27
  %891 = extractelement <4 x float> %889, i64 1, !dbg !27
  %892 = extractelement <4 x float> %889, i64 2, !dbg !27
  %893 = extractelement <4 x float> %889, i64 3, !dbg !27
  %894 = shufflevector <32 x float> %881, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %895 = fadd <4 x float> %661, %894, !dbg !27
  %896 = extractelement <4 x float> %895, i64 0, !dbg !27
  %897 = extractelement <4 x float> %895, i64 1, !dbg !27
  %898 = extractelement <4 x float> %895, i64 2, !dbg !27
  %899 = extractelement <4 x float> %895, i64 3, !dbg !27
  %900 = shufflevector <32 x float> %881, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %901 = fadd <4 x float> %665, %900, !dbg !27
  %902 = extractelement <4 x float> %901, i64 0, !dbg !27
  %903 = extractelement <4 x float> %901, i64 1, !dbg !27
  %904 = extractelement <4 x float> %901, i64 2, !dbg !27
  %905 = extractelement <4 x float> %901, i64 3, !dbg !27
  %906 = shufflevector <32 x float> %881, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %907 = fadd <4 x float> %669, %906, !dbg !27
  %908 = extractelement <4 x float> %907, i64 0, !dbg !27
  %909 = extractelement <4 x float> %907, i64 1, !dbg !27
  %910 = extractelement <4 x float> %907, i64 2, !dbg !27
  %911 = extractelement <4 x float> %907, i64 3, !dbg !27
  %912 = shufflevector <32 x float> %881, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %913 = fadd <4 x float> %673, %912, !dbg !27
  %914 = extractelement <4 x float> %913, i64 0, !dbg !27
  %915 = extractelement <4 x float> %913, i64 1, !dbg !27
  %916 = extractelement <4 x float> %913, i64 2, !dbg !27
  %917 = extractelement <4 x float> %913, i64 3, !dbg !27
  %918 = shufflevector <32 x float> %881, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %919 = fadd <4 x float> %677, %918, !dbg !27
  %920 = extractelement <4 x float> %919, i64 0, !dbg !27
  %921 = extractelement <4 x float> %919, i64 1, !dbg !27
  %922 = extractelement <4 x float> %919, i64 2, !dbg !27
  %923 = extractelement <4 x float> %919, i64 3, !dbg !27
  %924 = shufflevector <32 x float> %881, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %925 = fadd <4 x float> %682, %924, !dbg !27
  %926 = extractelement <4 x float> %925, i64 0, !dbg !27
  %927 = extractelement <4 x float> %925, i64 1, !dbg !27
  %928 = extractelement <4 x float> %925, i64 2, !dbg !27
  %929 = extractelement <4 x float> %925, i64 3, !dbg !27
  %930 = call float @llvm.maxnum.f32(float %835, float 0xFFF0000000000000), !dbg !28
  %931 = call float @llvm.maxnum.f32(float %930, float %836), !dbg !28
  %932 = call float @llvm.maxnum.f32(float %931, float %837), !dbg !28
  %933 = call float @llvm.maxnum.f32(float %932, float %838), !dbg !28
  %934 = call float @llvm.maxnum.f32(float %933, float %841), !dbg !28
  %935 = call float @llvm.maxnum.f32(float %934, float %842), !dbg !28
  %936 = call float @llvm.maxnum.f32(float %935, float %843), !dbg !28
  %937 = call float @llvm.maxnum.f32(float %936, float %844), !dbg !28
  %938 = call float @llvm.maxnum.f32(float %937, float %847), !dbg !28
  %939 = call float @llvm.maxnum.f32(float %938, float %848), !dbg !28
  %940 = call float @llvm.maxnum.f32(float %939, float %849), !dbg !28
  %941 = call float @llvm.maxnum.f32(float %940, float %850), !dbg !28
  %942 = call float @llvm.maxnum.f32(float %941, float %853), !dbg !28
  %943 = call float @llvm.maxnum.f32(float %942, float %854), !dbg !28
  %944 = call float @llvm.maxnum.f32(float %943, float %855), !dbg !28
  %945 = call float @llvm.maxnum.f32(float %944, float %856), !dbg !28
  %946 = call float @llvm.maxnum.f32(float %945, float %859), !dbg !28
  %947 = call float @llvm.maxnum.f32(float %946, float %860), !dbg !28
  %948 = call float @llvm.maxnum.f32(float %947, float %861), !dbg !28
  %949 = call float @llvm.maxnum.f32(float %948, float %862), !dbg !28
  %950 = call float @llvm.maxnum.f32(float %949, float %865), !dbg !28
  %951 = call float @llvm.maxnum.f32(float %950, float %866), !dbg !28
  %952 = call float @llvm.maxnum.f32(float %951, float %867), !dbg !28
  %953 = call float @llvm.maxnum.f32(float %952, float %868), !dbg !28
  %954 = call float @llvm.maxnum.f32(float %953, float %871), !dbg !28
  %955 = call float @llvm.maxnum.f32(float %954, float %872), !dbg !28
  %956 = call float @llvm.maxnum.f32(float %955, float %873), !dbg !28
  %957 = call float @llvm.maxnum.f32(float %956, float %874), !dbg !28
  %958 = call float @llvm.maxnum.f32(float %957, float %877), !dbg !28
  %959 = call float @llvm.maxnum.f32(float %958, float %878), !dbg !28
  %960 = call float @llvm.maxnum.f32(float %959, float %879), !dbg !28
  %961 = call float @llvm.maxnum.f32(float %960, float %880), !dbg !28
  %962 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %963 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %962), !dbg !28
  %964 = add nuw nsw i32 %963, 64, !dbg !28
  %965 = xor i32 %963, 16, !dbg !28
  %966 = and i32 %964, 192, !dbg !28
  %967 = icmp samesign ult i32 %965, %966, !dbg !28
  %968 = select i1 %967, i32 %965, i32 %963, !dbg !28
  %969 = shl nuw nsw i32 %968, 2, !dbg !28
  %970 = bitcast float %961 to i32, !dbg !28
  %971 = call i32 @llvm.amdgcn.ds.bpermute(i32 %969, i32 %970), !dbg !28
  %972 = bitcast i32 %971 to float, !dbg !28
  %973 = call float @llvm.maxnum.f32(float %961, float %972), !dbg !28
  %974 = xor i32 %963, 32, !dbg !28
  %975 = icmp samesign ult i32 %974, %966, !dbg !28
  %976 = select i1 %975, i32 %974, i32 %963, !dbg !28
  %977 = shl nuw nsw i32 %976, 2, !dbg !28
  %978 = bitcast float %973 to i32, !dbg !28
  %979 = call i32 @llvm.amdgcn.ds.bpermute(i32 %977, i32 %978), !dbg !28
  %980 = bitcast i32 %979 to float, !dbg !28
  %981 = call float @llvm.maxnum.f32(float %973, float %980), !dbg !28
  %982 = extractvalue [2 x <1 x float>] %49, 0, !dbg !28
  %983 = insertelement <1 x float> poison, float %981, i64 0, !dbg !28
  %984 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %983, <1 x float> %982), !dbg !28
  %985 = call float @llvm.maxnum.f32(float %884, float 0xFFF0000000000000), !dbg !28
  %986 = call float @llvm.maxnum.f32(float %985, float %885), !dbg !28
  %987 = call float @llvm.maxnum.f32(float %986, float %886), !dbg !28
  %988 = call float @llvm.maxnum.f32(float %987, float %887), !dbg !28
  %989 = call float @llvm.maxnum.f32(float %988, float %890), !dbg !28
  %990 = call float @llvm.maxnum.f32(float %989, float %891), !dbg !28
  %991 = call float @llvm.maxnum.f32(float %990, float %892), !dbg !28
  %992 = call float @llvm.maxnum.f32(float %991, float %893), !dbg !28
  %993 = call float @llvm.maxnum.f32(float %992, float %896), !dbg !28
  %994 = call float @llvm.maxnum.f32(float %993, float %897), !dbg !28
  %995 = call float @llvm.maxnum.f32(float %994, float %898), !dbg !28
  %996 = call float @llvm.maxnum.f32(float %995, float %899), !dbg !28
  %997 = call float @llvm.maxnum.f32(float %996, float %902), !dbg !28
  %998 = call float @llvm.maxnum.f32(float %997, float %903), !dbg !28
  %999 = call float @llvm.maxnum.f32(float %998, float %904), !dbg !28
  %1000 = call float @llvm.maxnum.f32(float %999, float %905), !dbg !28
  %1001 = call float @llvm.maxnum.f32(float %1000, float %908), !dbg !28
  %1002 = call float @llvm.maxnum.f32(float %1001, float %909), !dbg !28
  %1003 = call float @llvm.maxnum.f32(float %1002, float %910), !dbg !28
  %1004 = call float @llvm.maxnum.f32(float %1003, float %911), !dbg !28
  %1005 = call float @llvm.maxnum.f32(float %1004, float %914), !dbg !28
  %1006 = call float @llvm.maxnum.f32(float %1005, float %915), !dbg !28
  %1007 = call float @llvm.maxnum.f32(float %1006, float %916), !dbg !28
  %1008 = call float @llvm.maxnum.f32(float %1007, float %917), !dbg !28
  %1009 = call float @llvm.maxnum.f32(float %1008, float %920), !dbg !28
  %1010 = call float @llvm.maxnum.f32(float %1009, float %921), !dbg !28
  %1011 = call float @llvm.maxnum.f32(float %1010, float %922), !dbg !28
  %1012 = call float @llvm.maxnum.f32(float %1011, float %923), !dbg !28
  %1013 = call float @llvm.maxnum.f32(float %1012, float %926), !dbg !28
  %1014 = call float @llvm.maxnum.f32(float %1013, float %927), !dbg !28
  %1015 = call float @llvm.maxnum.f32(float %1014, float %928), !dbg !28
  %1016 = call float @llvm.maxnum.f32(float %1015, float %929), !dbg !28
  %1017 = bitcast float %1016 to i32, !dbg !28
  %1018 = call i32 @llvm.amdgcn.ds.bpermute(i32 %969, i32 %1017), !dbg !28
  %1019 = bitcast i32 %1018 to float, !dbg !28
  %1020 = call float @llvm.maxnum.f32(float %1016, float %1019), !dbg !28
  %1021 = bitcast float %1020 to i32, !dbg !28
  %1022 = call i32 @llvm.amdgcn.ds.bpermute(i32 %977, i32 %1021), !dbg !28
  %1023 = bitcast i32 %1022 to float, !dbg !28
  %1024 = call float @llvm.maxnum.f32(float %1020, float %1023), !dbg !28
  %1025 = extractvalue [2 x <1 x float>] %49, 1, !dbg !28
  %1026 = insertelement <1 x float> poison, float %1024, i64 0, !dbg !28
  %1027 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1026, <1 x float> %1025), !dbg !28
  %1028 = shufflevector <1 x float> %984, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !29
  %1029 = fsub <4 x float> %834, %1028, !dbg !29
  %1030 = fsub <4 x float> %840, %1028, !dbg !29
  %1031 = fsub <4 x float> %846, %1028, !dbg !29
  %1032 = fsub <4 x float> %852, %1028, !dbg !29
  %1033 = fsub <4 x float> %858, %1028, !dbg !29
  %1034 = fsub <4 x float> %864, %1028, !dbg !29
  %1035 = fsub <4 x float> %870, %1028, !dbg !29
  %1036 = fsub <4 x float> %876, %1028, !dbg !29
  %1037 = shufflevector <1 x float> %1027, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !29
  %1038 = fsub <4 x float> %883, %1037, !dbg !29
  %1039 = fsub <4 x float> %889, %1037, !dbg !29
  %1040 = fsub <4 x float> %895, %1037, !dbg !29
  %1041 = fsub <4 x float> %901, %1037, !dbg !29
  %1042 = fsub <4 x float> %907, %1037, !dbg !29
  %1043 = fsub <4 x float> %913, %1037, !dbg !29
  %1044 = fsub <4 x float> %919, %1037, !dbg !29
  %1045 = fsub <4 x float> %925, %1037, !dbg !29
  %1046 = fsub <1 x float> %982, %984, !dbg !30
  %1047 = extractelement <1 x float> %1046, i64 0, !dbg !31
  %1048 = call float @llvm.amdgcn.exp2.f32(float %1047), !dbg !31
  %1049 = insertelement <4 x float> poison, float %1048, i64 0, !dbg !36
  %1050 = shufflevector <4 x float> %1049, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1051 = extractvalue [2 x <1 x float>] %48, 0, !dbg !32
  %1052 = extractelement <1 x float> %1051, i64 0, !dbg !32
  %1053 = fmul float %1052, %1048, !dbg !32
  %1054 = fsub <1 x float> %1025, %1027, !dbg !30
  %1055 = extractelement <1 x float> %1054, i64 0, !dbg !31
  %1056 = call float @llvm.amdgcn.exp2.f32(float %1055), !dbg !31
  %1057 = insertelement <4 x float> poison, float %1056, i64 0, !dbg !36
  %1058 = shufflevector <4 x float> %1057, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %1059 = extractvalue [2 x <1 x float>] %48, 1, !dbg !32
  %1060 = extractelement <1 x float> %1059, i64 0, !dbg !32
  %1061 = fmul float %1060, %1056, !dbg !32
  %1062 = extractelement <4 x float> %1029, i64 0, !dbg !33
  %1063 = call float @llvm.amdgcn.exp2.f32(float %1062), !dbg !33
  %1064 = insertelement <4 x float> poison, float %1063, i64 0, !dbg !35
  %1065 = extractelement <4 x float> %1029, i64 1, !dbg !33
  %1066 = call float @llvm.amdgcn.exp2.f32(float %1065), !dbg !33
  %1067 = insertelement <4 x float> %1064, float %1066, i64 1, !dbg !35
  %1068 = fadd float %1063, %1066, !dbg !34
  %1069 = extractelement <4 x float> %1029, i64 2, !dbg !33
  %1070 = call float @llvm.amdgcn.exp2.f32(float %1069), !dbg !33
  %1071 = insertelement <4 x float> %1067, float %1070, i64 2, !dbg !35
  %1072 = fadd float %1068, %1070, !dbg !34
  %1073 = extractelement <4 x float> %1029, i64 3, !dbg !33
  %1074 = call float @llvm.amdgcn.exp2.f32(float %1073), !dbg !33
  %1075 = insertelement <4 x float> %1071, float %1074, i64 3, !dbg !35
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1075) #9, !dbg !35, !frisk.register_lifetimes !1
  %1076 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !35
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1076) #9, !dbg !34, !frisk.register_lifetimes !1
  %1077 = fadd float %1072, %1074, !dbg !34
  %1078 = extractelement <4 x float> %1030, i64 0, !dbg !33
  %1079 = call float @llvm.amdgcn.exp2.f32(float %1078), !dbg !33
  %1080 = insertelement <4 x float> poison, float %1079, i64 0, !dbg !35
  %1081 = fadd float %1077, %1079, !dbg !34
  %1082 = extractelement <4 x float> %1030, i64 1, !dbg !33
  %1083 = call float @llvm.amdgcn.exp2.f32(float %1082), !dbg !33
  %1084 = insertelement <4 x float> %1080, float %1083, i64 1, !dbg !35
  %1085 = fadd float %1081, %1083, !dbg !34
  %1086 = extractelement <4 x float> %1030, i64 2, !dbg !33
  %1087 = call float @llvm.amdgcn.exp2.f32(float %1086), !dbg !33
  %1088 = insertelement <4 x float> %1084, float %1087, i64 2, !dbg !35
  %1089 = fadd float %1085, %1087, !dbg !34
  %1090 = extractelement <4 x float> %1030, i64 3, !dbg !33
  %1091 = call float @llvm.amdgcn.exp2.f32(float %1090), !dbg !33
  %1092 = insertelement <4 x float> %1088, float %1091, i64 3, !dbg !35
  %math.result144 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1092) #9, !dbg !35, !frisk.register_lifetimes !1
  %1093 = fptrunc <4 x float> %math.result144 to <4 x half>, !dbg !35
  %math.packed145 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1093) #9, !dbg !34, !frisk.register_lifetimes !1
  %1094 = fadd float %1089, %1091, !dbg !34
  %1095 = extractelement <4 x float> %1031, i64 0, !dbg !33
  %1096 = call float @llvm.amdgcn.exp2.f32(float %1095), !dbg !33
  %1097 = insertelement <4 x float> poison, float %1096, i64 0, !dbg !35
  %1098 = fadd float %1094, %1096, !dbg !34
  %1099 = extractelement <4 x float> %1031, i64 1, !dbg !33
  %1100 = call float @llvm.amdgcn.exp2.f32(float %1099), !dbg !33
  %1101 = insertelement <4 x float> %1097, float %1100, i64 1, !dbg !35
  %1102 = fadd float %1098, %1100, !dbg !34
  %1103 = extractelement <4 x float> %1031, i64 2, !dbg !33
  %1104 = call float @llvm.amdgcn.exp2.f32(float %1103), !dbg !33
  %1105 = insertelement <4 x float> %1101, float %1104, i64 2, !dbg !35
  %1106 = fadd float %1102, %1104, !dbg !34
  %1107 = extractelement <4 x float> %1031, i64 3, !dbg !33
  %1108 = call float @llvm.amdgcn.exp2.f32(float %1107), !dbg !33
  %1109 = insertelement <4 x float> %1105, float %1108, i64 3, !dbg !35
  %math.result146 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1109) #9, !dbg !35, !frisk.register_lifetimes !1
  %1110 = fptrunc <4 x float> %math.result146 to <4 x half>, !dbg !35
  %math.packed147 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1110) #9, !dbg !34, !frisk.register_lifetimes !1
  %1111 = fadd float %1106, %1108, !dbg !34
  %1112 = extractelement <4 x float> %1032, i64 0, !dbg !33
  %1113 = call float @llvm.amdgcn.exp2.f32(float %1112), !dbg !33
  %1114 = insertelement <4 x float> poison, float %1113, i64 0, !dbg !35
  %1115 = fadd float %1111, %1113, !dbg !34
  %1116 = extractelement <4 x float> %1032, i64 1, !dbg !33
  %1117 = call float @llvm.amdgcn.exp2.f32(float %1116), !dbg !33
  %1118 = insertelement <4 x float> %1114, float %1117, i64 1, !dbg !35
  %1119 = fadd float %1115, %1117, !dbg !34
  %1120 = extractelement <4 x float> %1032, i64 2, !dbg !33
  %1121 = call float @llvm.amdgcn.exp2.f32(float %1120), !dbg !33
  %1122 = insertelement <4 x float> %1118, float %1121, i64 2, !dbg !35
  %1123 = fadd float %1119, %1121, !dbg !34
  %1124 = extractelement <4 x float> %1032, i64 3, !dbg !33
  %1125 = call float @llvm.amdgcn.exp2.f32(float %1124), !dbg !33
  %1126 = insertelement <4 x float> %1122, float %1125, i64 3, !dbg !35
  %math.result148 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1126) #9, !dbg !35, !frisk.register_lifetimes !1
  %1127 = fptrunc <4 x float> %math.result148 to <4 x half>, !dbg !35
  %math.packed149 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1127) #9, !dbg !34, !frisk.register_lifetimes !1
  %1128 = fadd float %1123, %1125, !dbg !34
  %1129 = extractelement <4 x float> %1033, i64 0, !dbg !33
  %1130 = call float @llvm.amdgcn.exp2.f32(float %1129), !dbg !33
  %1131 = insertelement <4 x float> poison, float %1130, i64 0, !dbg !35
  %1132 = fadd float %1128, %1130, !dbg !34
  %1133 = extractelement <4 x float> %1033, i64 1, !dbg !33
  %1134 = call float @llvm.amdgcn.exp2.f32(float %1133), !dbg !33
  %1135 = insertelement <4 x float> %1131, float %1134, i64 1, !dbg !35
  %1136 = fadd float %1132, %1134, !dbg !34
  %1137 = extractelement <4 x float> %1033, i64 2, !dbg !33
  %1138 = call float @llvm.amdgcn.exp2.f32(float %1137), !dbg !33
  %1139 = insertelement <4 x float> %1135, float %1138, i64 2, !dbg !35
  %1140 = fadd float %1136, %1138, !dbg !34
  %1141 = extractelement <4 x float> %1033, i64 3, !dbg !33
  %1142 = call float @llvm.amdgcn.exp2.f32(float %1141), !dbg !33
  %1143 = insertelement <4 x float> %1139, float %1142, i64 3, !dbg !35
  %math.result150 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1143) #9, !dbg !35, !frisk.register_lifetimes !1
  %1144 = fptrunc <4 x float> %math.result150 to <4 x half>, !dbg !35
  %math.packed151 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1144) #9, !dbg !34, !frisk.register_lifetimes !1
  %1145 = fadd float %1140, %1142, !dbg !34
  %1146 = extractelement <4 x float> %1034, i64 0, !dbg !33
  %1147 = call float @llvm.amdgcn.exp2.f32(float %1146), !dbg !33
  %1148 = insertelement <4 x float> poison, float %1147, i64 0, !dbg !35
  %1149 = fadd float %1145, %1147, !dbg !34
  %1150 = extractelement <4 x float> %1034, i64 1, !dbg !33
  %1151 = call float @llvm.amdgcn.exp2.f32(float %1150), !dbg !33
  %1152 = insertelement <4 x float> %1148, float %1151, i64 1, !dbg !35
  %1153 = fadd float %1149, %1151, !dbg !34
  %1154 = extractelement <4 x float> %1034, i64 2, !dbg !33
  %1155 = call float @llvm.amdgcn.exp2.f32(float %1154), !dbg !33
  %1156 = insertelement <4 x float> %1152, float %1155, i64 2, !dbg !35
  %1157 = fadd float %1153, %1155, !dbg !34
  %1158 = extractelement <4 x float> %1034, i64 3, !dbg !33
  %1159 = call float @llvm.amdgcn.exp2.f32(float %1158), !dbg !33
  %1160 = insertelement <4 x float> %1156, float %1159, i64 3, !dbg !35
  %math.result152 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1160) #9, !dbg !35, !frisk.register_lifetimes !1
  %1161 = fptrunc <4 x float> %math.result152 to <4 x half>, !dbg !35
  %math.packed153 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1161) #9, !dbg !34, !frisk.register_lifetimes !1
  %1162 = fadd float %1157, %1159, !dbg !34
  %1163 = extractelement <4 x float> %1035, i64 0, !dbg !33
  %1164 = call float @llvm.amdgcn.exp2.f32(float %1163), !dbg !33
  %1165 = insertelement <4 x float> poison, float %1164, i64 0, !dbg !35
  %1166 = fadd float %1162, %1164, !dbg !34
  %1167 = extractelement <4 x float> %1035, i64 1, !dbg !33
  %1168 = call float @llvm.amdgcn.exp2.f32(float %1167), !dbg !33
  %1169 = insertelement <4 x float> %1165, float %1168, i64 1, !dbg !35
  %1170 = fadd float %1166, %1168, !dbg !34
  %1171 = extractelement <4 x float> %1035, i64 2, !dbg !33
  %1172 = call float @llvm.amdgcn.exp2.f32(float %1171), !dbg !33
  %1173 = insertelement <4 x float> %1169, float %1172, i64 2, !dbg !35
  %1174 = fadd float %1170, %1172, !dbg !34
  %1175 = extractelement <4 x float> %1035, i64 3, !dbg !33
  %1176 = call float @llvm.amdgcn.exp2.f32(float %1175), !dbg !33
  %1177 = insertelement <4 x float> %1173, float %1176, i64 3, !dbg !35
  %math.result154 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1177) #9, !dbg !35, !frisk.register_lifetimes !1
  %1178 = fptrunc <4 x float> %math.result154 to <4 x half>, !dbg !35
  %math.packed155 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1178) #9, !dbg !34, !frisk.register_lifetimes !1
  %1179 = fadd float %1174, %1176, !dbg !34
  %1180 = extractelement <4 x float> %1036, i64 0, !dbg !33
  %1181 = call float @llvm.amdgcn.exp2.f32(float %1180), !dbg !33
  %1182 = insertelement <4 x float> poison, float %1181, i64 0, !dbg !35
  %1183 = fadd float %1179, %1181, !dbg !34
  %1184 = extractelement <4 x float> %1036, i64 1, !dbg !33
  %1185 = call float @llvm.amdgcn.exp2.f32(float %1184), !dbg !33
  %1186 = insertelement <4 x float> %1182, float %1185, i64 1, !dbg !35
  %1187 = fadd float %1183, %1185, !dbg !34
  %1188 = extractelement <4 x float> %1036, i64 2, !dbg !33
  %1189 = call float @llvm.amdgcn.exp2.f32(float %1188), !dbg !33
  %1190 = insertelement <4 x float> %1186, float %1189, i64 2, !dbg !35
  %1191 = fadd float %1187, %1189, !dbg !34
  %1192 = extractelement <4 x float> %1036, i64 3, !dbg !33
  %1193 = call float @llvm.amdgcn.exp2.f32(float %1192), !dbg !33
  %1194 = insertelement <4 x float> %1190, float %1193, i64 3, !dbg !35
  %math.result156 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1194) #9, !dbg !35, !frisk.register_lifetimes !1
  %1195 = fptrunc <4 x float> %math.result156 to <4 x half>, !dbg !35
  %math.packed157 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1195) #9, !dbg !34, !frisk.register_lifetimes !1
  %1196 = fadd float %1191, %1193, !dbg !34
  %1197 = extractelement <4 x float> %1038, i64 0, !dbg !33
  %1198 = call float @llvm.amdgcn.exp2.f32(float %1197), !dbg !33
  %1199 = insertelement <4 x float> poison, float %1198, i64 0, !dbg !35
  %1200 = extractelement <4 x float> %1038, i64 1, !dbg !33
  %1201 = call float @llvm.amdgcn.exp2.f32(float %1200), !dbg !33
  %1202 = insertelement <4 x float> %1199, float %1201, i64 1, !dbg !35
  %1203 = fadd float %1198, %1201, !dbg !34
  %1204 = extractelement <4 x float> %1038, i64 2, !dbg !33
  %1205 = call float @llvm.amdgcn.exp2.f32(float %1204), !dbg !33
  %1206 = insertelement <4 x float> %1202, float %1205, i64 2, !dbg !35
  %1207 = fadd float %1203, %1205, !dbg !34
  %1208 = extractelement <4 x float> %1038, i64 3, !dbg !33
  %1209 = call float @llvm.amdgcn.exp2.f32(float %1208), !dbg !33
  %1210 = insertelement <4 x float> %1206, float %1209, i64 3, !dbg !35
  %math.result158 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1210) #9, !dbg !35, !frisk.register_lifetimes !1
  %1211 = fptrunc <4 x float> %math.result158 to <4 x half>, !dbg !35
  %math.packed159 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1211) #9, !dbg !34, !frisk.register_lifetimes !1
  %1212 = fadd float %1207, %1209, !dbg !34
  %1213 = extractelement <4 x float> %1039, i64 0, !dbg !33
  %1214 = call float @llvm.amdgcn.exp2.f32(float %1213), !dbg !33
  %1215 = insertelement <4 x float> poison, float %1214, i64 0, !dbg !35
  %1216 = fadd float %1212, %1214, !dbg !34
  %1217 = extractelement <4 x float> %1039, i64 1, !dbg !33
  %1218 = call float @llvm.amdgcn.exp2.f32(float %1217), !dbg !33
  %1219 = insertelement <4 x float> %1215, float %1218, i64 1, !dbg !35
  %1220 = fadd float %1216, %1218, !dbg !34
  %1221 = extractelement <4 x float> %1039, i64 2, !dbg !33
  %1222 = call float @llvm.amdgcn.exp2.f32(float %1221), !dbg !33
  %1223 = insertelement <4 x float> %1219, float %1222, i64 2, !dbg !35
  %1224 = fadd float %1220, %1222, !dbg !34
  %1225 = extractelement <4 x float> %1039, i64 3, !dbg !33
  %1226 = call float @llvm.amdgcn.exp2.f32(float %1225), !dbg !33
  %1227 = insertelement <4 x float> %1223, float %1226, i64 3, !dbg !35
  %math.result160 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1227) #9, !dbg !35, !frisk.register_lifetimes !1
  %1228 = fptrunc <4 x float> %math.result160 to <4 x half>, !dbg !35
  %math.packed161 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1228) #9, !dbg !34, !frisk.register_lifetimes !1
  %1229 = fadd float %1224, %1226, !dbg !34
  %1230 = extractelement <4 x float> %1040, i64 0, !dbg !33
  %1231 = call float @llvm.amdgcn.exp2.f32(float %1230), !dbg !33
  %1232 = insertelement <4 x float> poison, float %1231, i64 0, !dbg !35
  %1233 = fadd float %1229, %1231, !dbg !34
  %1234 = extractelement <4 x float> %1040, i64 1, !dbg !33
  %1235 = call float @llvm.amdgcn.exp2.f32(float %1234), !dbg !33
  %1236 = insertelement <4 x float> %1232, float %1235, i64 1, !dbg !35
  %1237 = fadd float %1233, %1235, !dbg !34
  %1238 = extractelement <4 x float> %1040, i64 2, !dbg !33
  %1239 = call float @llvm.amdgcn.exp2.f32(float %1238), !dbg !33
  %1240 = insertelement <4 x float> %1236, float %1239, i64 2, !dbg !35
  %1241 = fadd float %1237, %1239, !dbg !34
  %1242 = extractelement <4 x float> %1040, i64 3, !dbg !33
  %1243 = call float @llvm.amdgcn.exp2.f32(float %1242), !dbg !33
  %1244 = insertelement <4 x float> %1240, float %1243, i64 3, !dbg !35
  %math.result162 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1244) #9, !dbg !35, !frisk.register_lifetimes !1
  %1245 = fptrunc <4 x float> %math.result162 to <4 x half>, !dbg !35
  %math.packed163 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1245) #9, !dbg !34, !frisk.register_lifetimes !1
  %1246 = fadd float %1241, %1243, !dbg !34
  %1247 = extractelement <4 x float> %1041, i64 0, !dbg !33
  %1248 = call float @llvm.amdgcn.exp2.f32(float %1247), !dbg !33
  %1249 = insertelement <4 x float> poison, float %1248, i64 0, !dbg !35
  %1250 = fadd float %1246, %1248, !dbg !34
  %1251 = extractelement <4 x float> %1041, i64 1, !dbg !33
  %1252 = call float @llvm.amdgcn.exp2.f32(float %1251), !dbg !33
  %1253 = insertelement <4 x float> %1249, float %1252, i64 1, !dbg !35
  %1254 = fadd float %1250, %1252, !dbg !34
  %1255 = extractelement <4 x float> %1041, i64 2, !dbg !33
  %1256 = call float @llvm.amdgcn.exp2.f32(float %1255), !dbg !33
  %1257 = insertelement <4 x float> %1253, float %1256, i64 2, !dbg !35
  %1258 = fadd float %1254, %1256, !dbg !34
  %1259 = extractelement <4 x float> %1041, i64 3, !dbg !33
  %1260 = call float @llvm.amdgcn.exp2.f32(float %1259), !dbg !33
  %1261 = insertelement <4 x float> %1257, float %1260, i64 3, !dbg !35
  %math.result164 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1261) #9, !dbg !35, !frisk.register_lifetimes !1
  %1262 = fptrunc <4 x float> %math.result164 to <4 x half>, !dbg !35
  %math.packed165 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1262) #9, !dbg !34, !frisk.register_lifetimes !1
  %1263 = fadd float %1258, %1260, !dbg !34
  %1264 = extractelement <4 x float> %1042, i64 0, !dbg !33
  %1265 = call float @llvm.amdgcn.exp2.f32(float %1264), !dbg !33
  %1266 = insertelement <4 x float> poison, float %1265, i64 0, !dbg !35
  %1267 = fadd float %1263, %1265, !dbg !34
  %1268 = extractelement <4 x float> %1042, i64 1, !dbg !33
  %1269 = call float @llvm.amdgcn.exp2.f32(float %1268), !dbg !33
  %1270 = insertelement <4 x float> %1266, float %1269, i64 1, !dbg !35
  %1271 = fadd float %1267, %1269, !dbg !34
  %1272 = extractelement <4 x float> %1042, i64 2, !dbg !33
  %1273 = call float @llvm.amdgcn.exp2.f32(float %1272), !dbg !33
  %1274 = insertelement <4 x float> %1270, float %1273, i64 2, !dbg !35
  %1275 = fadd float %1271, %1273, !dbg !34
  %1276 = extractelement <4 x float> %1042, i64 3, !dbg !33
  %1277 = call float @llvm.amdgcn.exp2.f32(float %1276), !dbg !33
  %1278 = insertelement <4 x float> %1274, float %1277, i64 3, !dbg !35
  %math.result166 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1278) #9, !dbg !35, !frisk.register_lifetimes !1
  %1279 = fptrunc <4 x float> %math.result166 to <4 x half>, !dbg !35
  %math.packed167 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1279) #9, !dbg !34, !frisk.register_lifetimes !1
  %1280 = fadd float %1275, %1277, !dbg !34
  %1281 = extractelement <4 x float> %1043, i64 0, !dbg !33
  %1282 = call float @llvm.amdgcn.exp2.f32(float %1281), !dbg !33
  %1283 = insertelement <4 x float> poison, float %1282, i64 0, !dbg !35
  %1284 = fadd float %1280, %1282, !dbg !34
  %1285 = extractelement <4 x float> %1043, i64 1, !dbg !33
  %1286 = call float @llvm.amdgcn.exp2.f32(float %1285), !dbg !33
  %1287 = insertelement <4 x float> %1283, float %1286, i64 1, !dbg !35
  %1288 = fadd float %1284, %1286, !dbg !34
  %1289 = extractelement <4 x float> %1043, i64 2, !dbg !33
  %1290 = call float @llvm.amdgcn.exp2.f32(float %1289), !dbg !33
  %1291 = insertelement <4 x float> %1287, float %1290, i64 2, !dbg !35
  %1292 = fadd float %1288, %1290, !dbg !34
  %1293 = extractelement <4 x float> %1043, i64 3, !dbg !33
  %1294 = call float @llvm.amdgcn.exp2.f32(float %1293), !dbg !33
  %1295 = insertelement <4 x float> %1291, float %1294, i64 3, !dbg !35
  %math.result168 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1295) #9, !dbg !35, !frisk.register_lifetimes !1
  %1296 = fptrunc <4 x float> %math.result168 to <4 x half>, !dbg !35
  %math.packed169 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1296) #9, !dbg !34, !frisk.register_lifetimes !1
  %1297 = fadd float %1292, %1294, !dbg !34
  %1298 = extractelement <4 x float> %1044, i64 0, !dbg !33
  %1299 = call float @llvm.amdgcn.exp2.f32(float %1298), !dbg !33
  %1300 = insertelement <4 x float> poison, float %1299, i64 0, !dbg !35
  %1301 = fadd float %1297, %1299, !dbg !34
  %1302 = extractelement <4 x float> %1044, i64 1, !dbg !33
  %1303 = call float @llvm.amdgcn.exp2.f32(float %1302), !dbg !33
  %1304 = insertelement <4 x float> %1300, float %1303, i64 1, !dbg !35
  %1305 = fadd float %1301, %1303, !dbg !34
  %1306 = extractelement <4 x float> %1044, i64 2, !dbg !33
  %1307 = call float @llvm.amdgcn.exp2.f32(float %1306), !dbg !33
  %1308 = insertelement <4 x float> %1304, float %1307, i64 2, !dbg !35
  %1309 = fadd float %1305, %1307, !dbg !34
  %1310 = extractelement <4 x float> %1044, i64 3, !dbg !33
  %1311 = call float @llvm.amdgcn.exp2.f32(float %1310), !dbg !33
  %1312 = insertelement <4 x float> %1308, float %1311, i64 3, !dbg !35
  %math.result170 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1312) #9, !dbg !35, !frisk.register_lifetimes !1
  %1313 = fptrunc <4 x float> %math.result170 to <4 x half>, !dbg !35
  %math.packed171 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1313) #9, !dbg !34, !frisk.register_lifetimes !1
  %1314 = fadd float %1309, %1311, !dbg !34
  %1315 = extractelement <4 x float> %1045, i64 0, !dbg !33
  %1316 = call float @llvm.amdgcn.exp2.f32(float %1315), !dbg !33
  %1317 = insertelement <4 x float> poison, float %1316, i64 0, !dbg !35
  %1318 = fadd float %1314, %1316, !dbg !34
  %1319 = extractelement <4 x float> %1045, i64 1, !dbg !33
  %1320 = call float @llvm.amdgcn.exp2.f32(float %1319), !dbg !33
  %1321 = insertelement <4 x float> %1317, float %1320, i64 1, !dbg !35
  %1322 = fadd float %1318, %1320, !dbg !34
  %1323 = extractelement <4 x float> %1045, i64 2, !dbg !33
  %1324 = call float @llvm.amdgcn.exp2.f32(float %1323), !dbg !33
  %1325 = insertelement <4 x float> %1321, float %1324, i64 2, !dbg !35
  %1326 = fadd float %1322, %1324, !dbg !34
  %1327 = extractelement <4 x float> %1045, i64 3, !dbg !33
  %1328 = call float @llvm.amdgcn.exp2.f32(float %1327), !dbg !33
  %1329 = insertelement <4 x float> %1325, float %1328, i64 3, !dbg !35
  %math.result172 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1329) #9, !dbg !35, !frisk.register_lifetimes !1
  %1330 = fptrunc <4 x float> %math.result172 to <4 x half>, !dbg !35
  %math.packed173 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1330) #9, !dbg !34, !frisk.register_lifetimes !1
  %1331 = fadd float %1326, %1328, !dbg !34
  %1332 = bitcast float %1196 to i32, !dbg !34
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %969, i32 %1332), !dbg !34
  %1334 = bitcast i32 %1333 to float, !dbg !34
  %1335 = fadd float %1196, %1334, !dbg !34
  %1336 = bitcast float %1335 to i32, !dbg !34
  %1337 = call i32 @llvm.amdgcn.ds.bpermute(i32 %977, i32 %1336), !dbg !34
  %1338 = bitcast i32 %1337 to float, !dbg !34
  %1339 = fadd float %1335, %1338, !dbg !34
  %1340 = fadd float %1339, %1053, !dbg !34
  %1341 = insertelement <1 x float> poison, float %1340, i64 0, !dbg !7
  %1342 = insertvalue [2 x <1 x float>] poison, <1 x float> %1341, 0, !dbg !7
  %1343 = bitcast float %1331 to i32, !dbg !34
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %969, i32 %1343), !dbg !34
  %1345 = bitcast i32 %1344 to float, !dbg !34
  %1346 = fadd float %1331, %1345, !dbg !34
  %1347 = bitcast float %1346 to i32, !dbg !34
  %1348 = call i32 @llvm.amdgcn.ds.bpermute(i32 %977, i32 %1347), !dbg !34
  %1349 = bitcast i32 %1348 to float, !dbg !34
  %1350 = fadd float %1346, %1349, !dbg !34
  %1351 = fadd float %1350, %1061, !dbg !34
  %1352 = insertelement <1 x float> poison, float %1351, i64 0, !dbg !7
  %1353 = insertvalue [2 x <1 x float>] %1342, <1 x float> %1352, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread539 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !15, !range !40, !frisk.register_lifetimes !1
  %address.local540 = shl nuw nsw i32 %address.thread539, 3, !dbg !15
  %address.local541 = and i32 %address.local540, 120, !dbg !15
  %address.local542 = shl nuw nsw i32 %address.thread539, 5, !dbg !15
  %address.local543 = and i32 %address.local542, 1536, !dbg !15
  %address.local544 = or disjoint i32 %address.local541, %address.local543, !dbg !15
  %address.local545 = zext nneg i32 %address.local544 to i64, !dbg !15
  %address.local546 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local545, !dbg !15
  %1354 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local546, i32 0) #0, !dbg !15
  %1355 = bitcast <4 x i32> %1354 to <8 x half>, !dbg !15
  %address.local547 = zext nneg i32 %address.thread539 to i64, !dbg !13
  %address.local548 = and i64 %address.local547, 60, !dbg !15
  %address.local549 = add nuw nsw i64 %address.local548, %address.local547, !dbg !15
  %address.local550 = shl nuw nsw i64 %address.local549, 3, !dbg !15
  %address.local551 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local550, !dbg !15
  %address.local552 = lshr i64 %address.local547, 4, !dbg !15
  %address.local553 = and i64 %address.local552, 3, !dbg !15
  %address.local554 = mul nuw nsw i64 %address.local553, 384, !dbg !15
  %address.local555 = getelementptr [2 x i8], ptr addrspace(3) %address.local551, i64 %address.local554, !dbg !15
  %address.local556 = lshr i64 %address.local547, 2, !dbg !17
  %address.local557 = mul nsw i64 %address.local556, -32, !dbg !15
  %address.local558 = getelementptr [2 x i8], ptr addrspace(3) %address.local555, i64 %address.local557, !dbg !15
  %address.local559 = getelementptr i8, ptr addrspace(3) %address.local558, i64 256, !dbg !15
  %1356 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local559, i32 0) #0, !dbg !15
  %1357 = bitcast <4 x i32> %1356 to <8 x half>, !dbg !15
  %address.local560 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local550, !dbg !15
  %address.local561 = getelementptr [2 x i8], ptr addrspace(3) %address.local560, i64 %address.local554, !dbg !15
  %address.local562 = getelementptr [2 x i8], ptr addrspace(3) %address.local561, i64 %address.local557, !dbg !15
  %address.local563 = getelementptr i8, ptr addrspace(3) %address.local562, i64 512, !dbg !15
  %1358 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local563, i32 0) #0, !dbg !15
  %1359 = bitcast <4 x i32> %1358 to <8 x half>, !dbg !15
  %address.local564 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local550, !dbg !15
  %address.local565 = getelementptr [2 x i8], ptr addrspace(3) %address.local564, i64 %address.local554, !dbg !15
  %address.local566 = getelementptr [2 x i8], ptr addrspace(3) %address.local565, i64 %address.local557, !dbg !15
  %address.local567 = getelementptr i8, ptr addrspace(3) %address.local566, i64 768, !dbg !15
  %1360 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local567, i32 0) #0, !dbg !15
  %1361 = bitcast <4 x i32> %1360 to <8 x half>, !dbg !15
  %address.local568 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local545, !dbg !15
  %address.local569 = getelementptr i8, ptr addrspace(3) %address.local568, i64 4096, !dbg !15
  %1362 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local569, i32 0) #0, !dbg !15
  %1363 = bitcast <4 x i32> %1362 to <8 x half>, !dbg !15
  %address.local570 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local550, !dbg !15
  %address.local571 = getelementptr [2 x i8], ptr addrspace(3) %address.local570, i64 %address.local554, !dbg !15
  %address.local572 = getelementptr [2 x i8], ptr addrspace(3) %address.local571, i64 %address.local557, !dbg !15
  %address.local573 = getelementptr i8, ptr addrspace(3) %address.local572, i64 4352, !dbg !15
  %1364 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local573, i32 0) #0, !dbg !15
  %1365 = bitcast <4 x i32> %1364 to <8 x half>, !dbg !15
  %address.local574 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local550, !dbg !15
  %address.local575 = getelementptr [2 x i8], ptr addrspace(3) %address.local574, i64 %address.local554, !dbg !15
  %address.local576 = getelementptr [2 x i8], ptr addrspace(3) %address.local575, i64 %address.local557, !dbg !15
  %address.local577 = getelementptr i8, ptr addrspace(3) %address.local576, i64 4608, !dbg !15
  %1366 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local577, i32 0) #0, !dbg !15
  %1367 = bitcast <4 x i32> %1366 to <8 x half>, !dbg !15
  %address.local578 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local550, !dbg !15
  %address.local579 = getelementptr [2 x i8], ptr addrspace(3) %address.local578, i64 %address.local554, !dbg !15
  %address.local580 = getelementptr [2 x i8], ptr addrspace(3) %address.local579, i64 %address.local557, !dbg !15
  %address.local581 = getelementptr i8, ptr addrspace(3) %address.local580, i64 4864, !dbg !15
  %1368 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local581, i32 0) #0, !dbg !15
  %1369 = bitcast <4 x i32> %1368 to <8 x half>, !dbg !15
  %1370 = add nsw i64 %85, %8, !dbg !18
  %1371 = shl i64 %1370, 19, !dbg !18
  %1372 = and i64 %1371, 16252928, !dbg !18
  %1373 = add i64 %1372, %91, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1374 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1373, !dbg !18
  %1375 = getelementptr i8, ptr addrspace(1) %1374, i64 64, !dbg !18
  %address.local582 = shl nuw nsw i64 %address.local547, 3, !dbg !12
  %address.local583 = and i64 %address.local582, 1536, !dbg !12
  %address.local584 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local583, !dbg !12
  %address.local585 = ptrtoint ptr addrspace(3) %address.local584 to i64, !dbg !12
  %address.local586 = trunc i64 %address.local585 to i32, !dbg !12
  %1376 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local586), !dbg !18
  %1377 = zext i32 %1376 to i64, !dbg !18
  %1378 = inttoptr i64 %1377 to ptr addrspace(3), !dbg !18
  %address.local587 = shl nuw nsw i64 %address.local547, 5, !dbg !18
  %address.local588 = and i64 %address.local587, 8064, !dbg !18
  %address.local589 = shl nuw nsw i64 %address.local547, 3, !dbg !18
  %address.local590 = and i64 %address.local589, 24, !dbg !18
  %address.local591 = or disjoint i64 %address.local588, %address.local590, !dbg !18
  %address.local592 = getelementptr [2 x i8], ptr addrspace(1) %1375, i64 %address.local591, !dbg !18
  %1379 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1380 = ptrtoint ptr addrspace(1) %address.local592 to i64, !dbg !18
  %1381 = sub i64 %1380, %1379, !dbg !18
  %dma.buffer.offset1100 = trunc i64 %1381 to i32, !dbg !18
  %1382 = trunc i64 %1379 to i32, !dbg !18
  %1383 = insertelement <4 x i32> poison, i32 %1382, i64 0, !dbg !18
  %1384 = lshr i64 %1379, 32, !dbg !18
  %1385 = trunc i64 %1384 to i32, !dbg !18
  %1386 = insertelement <4 x i32> %1383, i32 %1385, i64 1, !dbg !18
  %1387 = insertelement <4 x i32> %1386, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1101 = insertelement <4 x i32> %1387, i32 131072, i64 3, !dbg !18
  %1388 = ptrtoint ptr addrspace(3) %1378 to i32, !dbg !18
  %dma.buffer.lds1102 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1388), !dbg !18
  %1389 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1102, i32 %dma.buffer.offset1100, <4 x i32> %dma.buffer.rsrc1101) #10, !dbg !18
  %address.local593 = or disjoint i64 %address.local547, 256, !dbg !12
  %address.local594 = shl nuw nsw i64 %address.local593, 3, !dbg !12
  %address.local595 = and i64 %address.local594, 3584, !dbg !12
  %address.local596 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local595, !dbg !12
  %address.local597 = ptrtoint ptr addrspace(3) %address.local596 to i64, !dbg !12
  %address.local598 = trunc i64 %address.local597 to i32, !dbg !12
  %1390 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local598), !dbg !18
  %1391 = zext i32 %1390 to i64, !dbg !18
  %1392 = inttoptr i64 %1391 to ptr addrspace(3), !dbg !18
  %address.local599 = shl nuw nsw i64 %address.local593, 5, !dbg !18
  %address.local600 = and i64 %address.local599, 16256, !dbg !18
  %address.local601 = shl nuw nsw i64 %address.local547, 3, !dbg !18
  %address.local602 = and i64 %address.local601, 24, !dbg !18
  %address.local603 = or disjoint i64 %address.local600, %address.local602, !dbg !18
  %address.local604 = getelementptr [2 x i8], ptr addrspace(1) %1375, i64 %address.local603, !dbg !18
  %1393 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1394 = ptrtoint ptr addrspace(1) %address.local604 to i64, !dbg !18
  %1395 = sub i64 %1394, %1393, !dbg !18
  %dma.buffer.offset1097 = trunc i64 %1395 to i32, !dbg !18
  %1396 = trunc i64 %1393 to i32, !dbg !18
  %1397 = insertelement <4 x i32> poison, i32 %1396, i64 0, !dbg !18
  %1398 = lshr i64 %1393, 32, !dbg !18
  %1399 = trunc i64 %1398 to i32, !dbg !18
  %1400 = insertelement <4 x i32> %1397, i32 %1399, i64 1, !dbg !18
  %1401 = insertelement <4 x i32> %1400, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1098 = insertelement <4 x i32> %1401, i32 131072, i64 3, !dbg !18
  %1402 = ptrtoint ptr addrspace(3) %1392 to i32, !dbg !18
  %dma.buffer.lds1099 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1402), !dbg !18
  %1403 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1099, i32 %dma.buffer.offset1097, <4 x i32> %dma.buffer.rsrc1098) #10, !dbg !18
  %1404 = shufflevector <8 x half> %1355, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1405 = extractvalue [2 x <8 x float>] %47, 0, !dbg !36
  %1406 = shufflevector <8 x float> %1405, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1407 = fmul <4 x float> %1406, %1050, !dbg !36
  %1408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1404, <4 x float> %1407) #9, !dbg !15
  %1409 = shufflevector <8 x half> %1357, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1409, <4 x float> %1408) #9, !dbg !15
  %1411 = shufflevector <8 x half> %1359, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1411, <4 x float> %1410) #9, !dbg !15
  %1413 = shufflevector <8 x half> %1361, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1413, <4 x float> %1412) #9, !dbg !15
  %1415 = shufflevector <8 x half> %1363, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1416 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1415, <4 x float> %1414) #9, !dbg !15
  %1417 = shufflevector <8 x half> %1365, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1417, <4 x float> %1416) #9, !dbg !15
  %1419 = shufflevector <8 x half> %1367, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1420 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1419, <4 x float> %1418) #9, !dbg !15
  %1421 = shufflevector <8 x half> %1369, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1421, <4 x float> %1420) #9, !dbg !15
  %1423 = shufflevector <8 x half> %1355, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1424 = shufflevector <8 x float> %1405, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1425 = fmul <4 x float> %1424, %1050, !dbg !36
  %1426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1423, <4 x float> %1425) #9, !dbg !15
  %1427 = shufflevector <8 x half> %1357, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1427, <4 x float> %1426) #9, !dbg !15
  %1429 = shufflevector <8 x half> %1359, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1429, <4 x float> %1428) #9, !dbg !15
  %1431 = shufflevector <8 x half> %1361, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1431, <4 x float> %1430) #9, !dbg !15
  %1433 = shufflevector <8 x half> %1363, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1433, <4 x float> %1432) #9, !dbg !15
  %1435 = shufflevector <8 x half> %1365, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1435, <4 x float> %1434) #9, !dbg !15
  %1437 = shufflevector <8 x half> %1367, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1437, <4 x float> %1436) #9, !dbg !15
  %1439 = shufflevector <8 x half> %1369, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1439, <4 x float> %1438) #9, !dbg !15
  %1441 = extractvalue [2 x <8 x float>] %47, 1, !dbg !36
  %1442 = shufflevector <8 x float> %1441, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1443 = fmul <4 x float> %1442, %1058, !dbg !36
  %1444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1404, <4 x float> %1443) #9, !dbg !15
  %1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1409, <4 x float> %1444) #9, !dbg !15
  %1446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1411, <4 x float> %1445) #9, !dbg !15
  %1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1413, <4 x float> %1446) #9, !dbg !15
  %1448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1415, <4 x float> %1447) #9, !dbg !15
  %1449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1417, <4 x float> %1448) #9, !dbg !15
  %1450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1419, <4 x float> %1449) #9, !dbg !15
  %1451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1421, <4 x float> %1450) #9, !dbg !15
  %1452 = shufflevector <8 x float> %1441, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1453 = fmul <4 x float> %1452, %1058, !dbg !36
  %1454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1423, <4 x float> %1453) #9, !dbg !15
  %1455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1427, <4 x float> %1454) #9, !dbg !15
  %1456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1429, <4 x float> %1455) #9, !dbg !15
  %1457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1431, <4 x float> %1456) #9, !dbg !15
  %1458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1433, <4 x float> %1457) #9, !dbg !15
  %1459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1435, <4 x float> %1458) #9, !dbg !15
  %1460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1437, <4 x float> %1459) #9, !dbg !15
  %1461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1439, <4 x float> %1460) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1462 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local546, i32 0) #0, !dbg !15
  %1463 = bitcast <4 x i32> %1462 to <8 x half>, !dbg !15
  %1464 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local559, i32 0) #0, !dbg !15
  %1465 = bitcast <4 x i32> %1464 to <8 x half>, !dbg !15
  %1466 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local563, i32 0) #0, !dbg !15
  %1467 = bitcast <4 x i32> %1466 to <8 x half>, !dbg !15
  %1468 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local567, i32 0) #0, !dbg !15
  %1469 = bitcast <4 x i32> %1468 to <8 x half>, !dbg !15
  %1470 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local569, i32 0) #0, !dbg !15
  %1471 = bitcast <4 x i32> %1470 to <8 x half>, !dbg !15
  %1472 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local573, i32 0) #0, !dbg !15
  %1473 = bitcast <4 x i32> %1472 to <8 x half>, !dbg !15
  %1474 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local577, i32 0) #0, !dbg !15
  %1475 = bitcast <4 x i32> %1474 to <8 x half>, !dbg !15
  %1476 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local581, i32 0) #0, !dbg !15
  %1477 = bitcast <4 x i32> %1476 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1478 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1373, !dbg !18
  %1479 = getelementptr i8, ptr addrspace(1) %1478, i64 128, !dbg !18
  %1480 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local586), !dbg !18
  %1481 = zext i32 %1480 to i64, !dbg !18
  %1482 = inttoptr i64 %1481 to ptr addrspace(3), !dbg !18
  %address.local605 = getelementptr [2 x i8], ptr addrspace(1) %1479, i64 %address.local591, !dbg !18
  %1483 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1484 = ptrtoint ptr addrspace(1) %address.local605 to i64, !dbg !18
  %1485 = sub i64 %1484, %1483, !dbg !18
  %dma.buffer.offset1094 = trunc i64 %1485 to i32, !dbg !18
  %1486 = trunc i64 %1483 to i32, !dbg !18
  %1487 = insertelement <4 x i32> poison, i32 %1486, i64 0, !dbg !18
  %1488 = lshr i64 %1483, 32, !dbg !18
  %1489 = trunc i64 %1488 to i32, !dbg !18
  %1490 = insertelement <4 x i32> %1487, i32 %1489, i64 1, !dbg !18
  %1491 = insertelement <4 x i32> %1490, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1095 = insertelement <4 x i32> %1491, i32 131072, i64 3, !dbg !18
  %1492 = ptrtoint ptr addrspace(3) %1482 to i32, !dbg !18
  %dma.buffer.lds1096 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1492), !dbg !18
  %1493 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1096, i32 %dma.buffer.offset1094, <4 x i32> %dma.buffer.rsrc1095) #10, !dbg !18
  %1494 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local598), !dbg !18
  %1495 = zext i32 %1494 to i64, !dbg !18
  %1496 = inttoptr i64 %1495 to ptr addrspace(3), !dbg !18
  %address.local606 = getelementptr [2 x i8], ptr addrspace(1) %1479, i64 %address.local603, !dbg !18
  %1497 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1498 = ptrtoint ptr addrspace(1) %address.local606 to i64, !dbg !18
  %1499 = sub i64 %1498, %1497, !dbg !18
  %dma.buffer.offset1091 = trunc i64 %1499 to i32, !dbg !18
  %1500 = trunc i64 %1497 to i32, !dbg !18
  %1501 = insertelement <4 x i32> poison, i32 %1500, i64 0, !dbg !18
  %1502 = lshr i64 %1497, 32, !dbg !18
  %1503 = trunc i64 %1502 to i32, !dbg !18
  %1504 = insertelement <4 x i32> %1501, i32 %1503, i64 1, !dbg !18
  %1505 = insertelement <4 x i32> %1504, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1092 = insertelement <4 x i32> %1505, i32 131072, i64 3, !dbg !18
  %1506 = ptrtoint ptr addrspace(3) %1496 to i32, !dbg !18
  %dma.buffer.lds1093 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1506), !dbg !18
  %1507 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1093, i32 %dma.buffer.offset1091, <4 x i32> %dma.buffer.rsrc1092) #10, !dbg !18
  %1508 = shufflevector <8 x half> %1463, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1509 = extractvalue [2 x <8 x float>] %50, 0, !dbg !36
  %1510 = shufflevector <8 x float> %1509, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1511 = fmul <4 x float> %1510, %1050, !dbg !36
  %1512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1508, <4 x float> %1511) #9, !dbg !15
  %1513 = shufflevector <8 x half> %1465, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1513, <4 x float> %1512) #9, !dbg !15
  %1515 = shufflevector <8 x half> %1467, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1515, <4 x float> %1514) #9, !dbg !15
  %1517 = shufflevector <8 x half> %1469, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1517, <4 x float> %1516) #9, !dbg !15
  %1519 = shufflevector <8 x half> %1471, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1520 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1519, <4 x float> %1518) #9, !dbg !15
  %1521 = shufflevector <8 x half> %1473, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1521, <4 x float> %1520) #9, !dbg !15
  %1523 = shufflevector <8 x half> %1475, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1523, <4 x float> %1522) #9, !dbg !15
  %1525 = shufflevector <8 x half> %1477, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1525, <4 x float> %1524) #9, !dbg !15
  %1527 = shufflevector <8 x half> %1463, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1528 = shufflevector <8 x float> %1509, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1529 = fmul <4 x float> %1528, %1050, !dbg !36
  %1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1527, <4 x float> %1529) #9, !dbg !15
  %1531 = shufflevector <8 x half> %1465, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1531, <4 x float> %1530) #9, !dbg !15
  %1533 = shufflevector <8 x half> %1467, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1533, <4 x float> %1532) #9, !dbg !15
  %1535 = shufflevector <8 x half> %1469, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1535, <4 x float> %1534) #9, !dbg !15
  %1537 = shufflevector <8 x half> %1471, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1537, <4 x float> %1536) #9, !dbg !15
  %1539 = shufflevector <8 x half> %1473, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1539, <4 x float> %1538) #9, !dbg !15
  %1541 = shufflevector <8 x half> %1475, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1541, <4 x float> %1540) #9, !dbg !15
  %1543 = shufflevector <8 x half> %1477, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1543, <4 x float> %1542) #9, !dbg !15
  %1545 = extractvalue [2 x <8 x float>] %50, 1, !dbg !36
  %1546 = shufflevector <8 x float> %1545, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1547 = fmul <4 x float> %1546, %1058, !dbg !36
  %1548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1508, <4 x float> %1547) #9, !dbg !15
  %1549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1513, <4 x float> %1548) #9, !dbg !15
  %1550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1515, <4 x float> %1549) #9, !dbg !15
  %1551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1517, <4 x float> %1550) #9, !dbg !15
  %1552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1519, <4 x float> %1551) #9, !dbg !15
  %1553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1521, <4 x float> %1552) #9, !dbg !15
  %1554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1523, <4 x float> %1553) #9, !dbg !15
  %1555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1525, <4 x float> %1554) #9, !dbg !15
  %1556 = shufflevector <8 x float> %1545, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1557 = fmul <4 x float> %1556, %1058, !dbg !36
  %1558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1527, <4 x float> %1557) #9, !dbg !15
  %1559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1531, <4 x float> %1558) #9, !dbg !15
  %1560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1533, <4 x float> %1559) #9, !dbg !15
  %1561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1535, <4 x float> %1560) #9, !dbg !15
  %1562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1537, <4 x float> %1561) #9, !dbg !15
  %1563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1539, <4 x float> %1562) #9, !dbg !15
  %1564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1541, <4 x float> %1563) #9, !dbg !15
  %1565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1543, <4 x float> %1564) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1566 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local546, i32 0) #0, !dbg !15
  %1567 = bitcast <4 x i32> %1566 to <8 x half>, !dbg !15
  %1568 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local559, i32 0) #0, !dbg !15
  %1569 = bitcast <4 x i32> %1568 to <8 x half>, !dbg !15
  %1570 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local563, i32 0) #0, !dbg !15
  %1571 = bitcast <4 x i32> %1570 to <8 x half>, !dbg !15
  %1572 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local567, i32 0) #0, !dbg !15
  %1573 = bitcast <4 x i32> %1572 to <8 x half>, !dbg !15
  %1574 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local569, i32 0) #0, !dbg !15
  %1575 = bitcast <4 x i32> %1574 to <8 x half>, !dbg !15
  %1576 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local573, i32 0) #0, !dbg !15
  %1577 = bitcast <4 x i32> %1576 to <8 x half>, !dbg !15
  %1578 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local577, i32 0) #0, !dbg !15
  %1579 = bitcast <4 x i32> %1578 to <8 x half>, !dbg !15
  %1580 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local581, i32 0) #0, !dbg !15
  %1581 = bitcast <4 x i32> %1580 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1582 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1373, !dbg !18
  %1583 = getelementptr i8, ptr addrspace(1) %1582, i64 192, !dbg !18
  %1584 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local586), !dbg !18
  %1585 = zext i32 %1584 to i64, !dbg !18
  %1586 = inttoptr i64 %1585 to ptr addrspace(3), !dbg !18
  %address.local607 = getelementptr [2 x i8], ptr addrspace(1) %1583, i64 %address.local591, !dbg !18
  %1587 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1588 = ptrtoint ptr addrspace(1) %address.local607 to i64, !dbg !18
  %1589 = sub i64 %1588, %1587, !dbg !18
  %dma.buffer.offset1088 = trunc i64 %1589 to i32, !dbg !18
  %1590 = trunc i64 %1587 to i32, !dbg !18
  %1591 = insertelement <4 x i32> poison, i32 %1590, i64 0, !dbg !18
  %1592 = lshr i64 %1587, 32, !dbg !18
  %1593 = trunc i64 %1592 to i32, !dbg !18
  %1594 = insertelement <4 x i32> %1591, i32 %1593, i64 1, !dbg !18
  %1595 = insertelement <4 x i32> %1594, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1089 = insertelement <4 x i32> %1595, i32 131072, i64 3, !dbg !18
  %1596 = ptrtoint ptr addrspace(3) %1586 to i32, !dbg !18
  %dma.buffer.lds1090 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1596), !dbg !18
  %1597 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1090, i32 %dma.buffer.offset1088, <4 x i32> %dma.buffer.rsrc1089) #10, !dbg !18
  %1598 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local598), !dbg !18
  %1599 = zext i32 %1598 to i64, !dbg !18
  %1600 = inttoptr i64 %1599 to ptr addrspace(3), !dbg !18
  %address.local608 = getelementptr [2 x i8], ptr addrspace(1) %1583, i64 %address.local603, !dbg !18
  %1601 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1602 = ptrtoint ptr addrspace(1) %address.local608 to i64, !dbg !18
  %1603 = sub i64 %1602, %1601, !dbg !18
  %dma.buffer.offset = trunc i64 %1603 to i32, !dbg !18
  %1604 = trunc i64 %1601 to i32, !dbg !18
  %1605 = insertelement <4 x i32> poison, i32 %1604, i64 0, !dbg !18
  %1606 = lshr i64 %1601, 32, !dbg !18
  %1607 = trunc i64 %1606 to i32, !dbg !18
  %1608 = insertelement <4 x i32> %1605, i32 %1607, i64 1, !dbg !18
  %1609 = insertelement <4 x i32> %1608, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1609, i32 131072, i64 3, !dbg !18
  %1610 = ptrtoint ptr addrspace(3) %1600 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1610), !dbg !18
  %1611 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1612 = shufflevector <8 x half> %1567, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1613 = extractvalue [2 x <8 x float>] %51, 0, !dbg !36
  %1614 = shufflevector <8 x float> %1613, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1615 = fmul <4 x float> %1614, %1050, !dbg !36
  %1616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1612, <4 x float> %1615) #9, !dbg !15
  %1617 = shufflevector <8 x half> %1569, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1617, <4 x float> %1616) #9, !dbg !15
  %1619 = shufflevector <8 x half> %1571, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1619, <4 x float> %1618) #9, !dbg !15
  %1621 = shufflevector <8 x half> %1573, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1621, <4 x float> %1620) #9, !dbg !15
  %1623 = shufflevector <8 x half> %1575, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1623, <4 x float> %1622) #9, !dbg !15
  %1625 = shufflevector <8 x half> %1577, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1625, <4 x float> %1624) #9, !dbg !15
  %1627 = shufflevector <8 x half> %1579, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1628 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1627, <4 x float> %1626) #9, !dbg !15
  %1629 = shufflevector <8 x half> %1581, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1630 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1629, <4 x float> %1628) #9, !dbg !15
  %1631 = shufflevector <8 x half> %1567, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1632 = shufflevector <8 x float> %1613, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1633 = fmul <4 x float> %1632, %1050, !dbg !36
  %1634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1631, <4 x float> %1633) #9, !dbg !15
  %1635 = shufflevector <8 x half> %1569, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1636 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1635, <4 x float> %1634) #9, !dbg !15
  %1637 = shufflevector <8 x half> %1571, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1637, <4 x float> %1636) #9, !dbg !15
  %1639 = shufflevector <8 x half> %1573, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1639, <4 x float> %1638) #9, !dbg !15
  %1641 = shufflevector <8 x half> %1575, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1641, <4 x float> %1640) #9, !dbg !15
  %1643 = shufflevector <8 x half> %1577, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1643, <4 x float> %1642) #9, !dbg !15
  %1645 = shufflevector <8 x half> %1579, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1645, <4 x float> %1644) #9, !dbg !15
  %1647 = shufflevector <8 x half> %1581, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1647, <4 x float> %1646) #9, !dbg !15
  %1649 = extractvalue [2 x <8 x float>] %51, 1, !dbg !36
  %1650 = shufflevector <8 x float> %1649, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1651 = fmul <4 x float> %1650, %1058, !dbg !36
  %1652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1612, <4 x float> %1651) #9, !dbg !15
  %1653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1617, <4 x float> %1652) #9, !dbg !15
  %1654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1619, <4 x float> %1653) #9, !dbg !15
  %1655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1621, <4 x float> %1654) #9, !dbg !15
  %1656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1623, <4 x float> %1655) #9, !dbg !15
  %1657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1625, <4 x float> %1656) #9, !dbg !15
  %1658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1627, <4 x float> %1657) #9, !dbg !15
  %1659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1629, <4 x float> %1658) #9, !dbg !15
  %1660 = shufflevector <8 x float> %1649, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1661 = fmul <4 x float> %1660, %1058, !dbg !36
  %1662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1631, <4 x float> %1661) #9, !dbg !15
  %1663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1635, <4 x float> %1662) #9, !dbg !15
  %1664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1637, <4 x float> %1663) #9, !dbg !15
  %1665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1639, <4 x float> %1664) #9, !dbg !15
  %1666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1641, <4 x float> %1665) #9, !dbg !15
  %1667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1643, <4 x float> %1666) #9, !dbg !15
  %1668 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1645, <4 x float> %1667) #9, !dbg !15
  %1669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1647, <4 x float> %1668) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1670 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local546, i32 0) #0, !dbg !15
  %1671 = bitcast <4 x i32> %1670 to <8 x half>, !dbg !15
  %1672 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local559, i32 0) #0, !dbg !15
  %1673 = bitcast <4 x i32> %1672 to <8 x half>, !dbg !15
  %1674 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local563, i32 0) #0, !dbg !15
  %1675 = bitcast <4 x i32> %1674 to <8 x half>, !dbg !15
  %1676 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local567, i32 0) #0, !dbg !15
  %1677 = bitcast <4 x i32> %1676 to <8 x half>, !dbg !15
  %1678 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local569, i32 0) #0, !dbg !15
  %1679 = bitcast <4 x i32> %1678 to <8 x half>, !dbg !15
  %1680 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local573, i32 0) #0, !dbg !15
  %1681 = bitcast <4 x i32> %1680 to <8 x half>, !dbg !15
  %1682 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local577, i32 0) #0, !dbg !15
  %1683 = bitcast <4 x i32> %1682 to <8 x half>, !dbg !15
  %1684 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local581, i32 0) #0, !dbg !15
  %1685 = bitcast <4 x i32> %1684 to <8 x half>, !dbg !15
  %1686 = shufflevector <8 x half> %1671, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1687 = extractvalue [2 x <8 x float>] %52, 0, !dbg !36
  %1688 = shufflevector <8 x float> %1687, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1689 = fmul <4 x float> %1688, %1050, !dbg !36
  %1690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1686, <4 x float> %1689) #9, !dbg !15
  %1691 = shufflevector <8 x half> %1673, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1692 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1691, <4 x float> %1690) #9, !dbg !15
  %1693 = shufflevector <8 x half> %1675, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1693, <4 x float> %1692) #9, !dbg !15
  %1695 = shufflevector <8 x half> %1677, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1695, <4 x float> %1694) #9, !dbg !15
  %1697 = shufflevector <8 x half> %1679, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1697, <4 x float> %1696) #9, !dbg !15
  %1699 = shufflevector <8 x half> %1681, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1700 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1699, <4 x float> %1698) #9, !dbg !15
  %1701 = shufflevector <8 x half> %1683, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1702 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1701, <4 x float> %1700) #9, !dbg !15
  %1703 = shufflevector <8 x half> %1685, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1704 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1703, <4 x float> %1702) #9, !dbg !15
  %1705 = shufflevector <8 x half> %1671, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1706 = shufflevector <8 x float> %1687, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1707 = fmul <4 x float> %1706, %1050, !dbg !36
  %1708 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1705, <4 x float> %1707) #9, !dbg !15
  %1709 = shufflevector <8 x half> %1673, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1710 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1709, <4 x float> %1708) #9, !dbg !15
  %1711 = shufflevector <8 x half> %1675, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1712 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1711, <4 x float> %1710) #9, !dbg !15
  %1713 = shufflevector <8 x half> %1677, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1714 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1713, <4 x float> %1712) #9, !dbg !15
  %1715 = shufflevector <8 x half> %1679, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1716 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1715, <4 x float> %1714) #9, !dbg !15
  %1717 = shufflevector <8 x half> %1681, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1718 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1717, <4 x float> %1716) #9, !dbg !15
  %1719 = shufflevector <8 x half> %1683, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1720 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1719, <4 x float> %1718) #9, !dbg !15
  %1721 = shufflevector <8 x half> %1685, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1722 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1721, <4 x float> %1720) #9, !dbg !15
  %1723 = extractvalue [2 x <8 x float>] %52, 1, !dbg !36
  %1724 = shufflevector <8 x float> %1723, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1725 = fmul <4 x float> %1724, %1058, !dbg !36
  %1726 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1686, <4 x float> %1725) #9, !dbg !15
  %1727 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1691, <4 x float> %1726) #9, !dbg !15
  %1728 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1693, <4 x float> %1727) #9, !dbg !15
  %1729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1695, <4 x float> %1728) #9, !dbg !15
  %1730 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1697, <4 x float> %1729) #9, !dbg !15
  %1731 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1699, <4 x float> %1730) #9, !dbg !15
  %1732 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1701, <4 x float> %1731) #9, !dbg !15
  %1733 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1703, <4 x float> %1732) #9, !dbg !15
  %1734 = shufflevector <8 x float> %1723, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1735 = fmul <4 x float> %1734, %1058, !dbg !36
  %1736 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1705, <4 x float> %1735) #9, !dbg !15
  %1737 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1709, <4 x float> %1736) #9, !dbg !15
  %1738 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1711, <4 x float> %1737) #9, !dbg !15
  %1739 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1713, <4 x float> %1738) #9, !dbg !15
  %1740 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1715, <4 x float> %1739) #9, !dbg !15
  %1741 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1717, <4 x float> %1740) #9, !dbg !15
  %1742 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1719, <4 x float> %1741) #9, !dbg !15
  %1743 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1721, <4 x float> %1742) #9, !dbg !15
  %1744 = shufflevector <4 x float> %1422, <4 x float> %1440, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1745 = insertvalue [2 x <8 x float>] poison, <8 x float> %1744, 0, !dbg !7
  %1746 = shufflevector <4 x float> %1451, <4 x float> %1461, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1747 = insertvalue [2 x <8 x float>] %1745, <8 x float> %1746, 1, !dbg !7
  %1748 = insertvalue [2 x <1 x float>] poison, <1 x float> %984, 0, !dbg !7
  %1749 = insertvalue [2 x <1 x float>] %1748, <1 x float> %1027, 1, !dbg !7
  %1750 = shufflevector <4 x float> %1526, <4 x float> %1544, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1751 = insertvalue [2 x <8 x float>] poison, <8 x float> %1750, 0, !dbg !7
  %1752 = shufflevector <4 x float> %1555, <4 x float> %1565, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1753 = insertvalue [2 x <8 x float>] %1751, <8 x float> %1752, 1, !dbg !7
  %1754 = shufflevector <4 x float> %1630, <4 x float> %1648, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1755 = insertvalue [2 x <8 x float>] poison, <8 x float> %1754, 0, !dbg !7
  %1756 = shufflevector <4 x float> %1659, <4 x float> %1669, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1757 = insertvalue [2 x <8 x float>] %1755, <8 x float> %1756, 1, !dbg !7
  %1758 = shufflevector <4 x float> %1704, <4 x float> %1722, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1759 = insertvalue [2 x <8 x float>] poison, <8 x float> %1758, 0, !dbg !7
  %1760 = shufflevector <4 x float> %1733, <4 x float> %1743, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1761 = insertvalue [2 x <8 x float>] %1759, <8 x float> %1760, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1762 = add nuw nsw i64 %46, 1, !dbg !7
  br label %45, !dbg !7

1763:                                             ; preds = %45
  %1764 = and i32 %26, 16, !dbg !7
  %.not2 = icmp eq i32 %1764, 0, !dbg !7
  %1765 = and i32 %26, 32, !dbg !7
  %.not3 = icmp eq i32 %1765, 0, !dbg !7
  %1766 = extractvalue [2 x <8 x float>] %52, 0, !dbg !7
  %1767 = extractelement <8 x float> %1766, i64 0, !dbg !7
  %1768 = extractelement <8 x float> %1766, i64 1, !dbg !7
  %1769 = extractelement <8 x float> %1766, i64 2, !dbg !7
  %1770 = extractelement <8 x float> %1766, i64 3, !dbg !7
  %1771 = select i1 %.not2, float %1768, float %1767, !dbg !7
  %1772 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1773 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1772), !dbg !7
  %1774 = add nuw nsw i32 %1773, 64, !dbg !7
  %1775 = xor i32 %1773, 16, !dbg !7
  %1776 = and i32 %1774, 192, !dbg !7
  %1777 = icmp samesign ult i32 %1775, %1776, !dbg !7
  %1778 = select i1 %1777, i32 %1775, i32 %1773, !dbg !7
  %1779 = shl nuw nsw i32 %1778, 2, !dbg !7
  %1780 = bitcast float %1771 to i32, !dbg !7
  %1781 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1780), !dbg !7
  %1782 = bitcast i32 %1781 to float, !dbg !7
  %1783 = select i1 %.not2, float %1767, float %1782, !dbg !7
  %1784 = select i1 %.not2, float %1782, float %1768, !dbg !7
  %1785 = select i1 %.not2, float %1770, float %1769, !dbg !7
  %1786 = bitcast float %1785 to i32, !dbg !7
  %1787 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1786), !dbg !7
  %1788 = bitcast i32 %1787 to float, !dbg !7
  %1789 = select i1 %.not2, float %1769, float %1788, !dbg !7
  %1790 = select i1 %.not2, float %1788, float %1770, !dbg !7
  %1791 = select i1 %.not3, float %1789, float %1783, !dbg !7
  %1792 = xor i32 %1773, 32, !dbg !7
  %1793 = icmp samesign ult i32 %1792, %1776, !dbg !7
  %1794 = select i1 %1793, i32 %1792, i32 %1773, !dbg !7
  %1795 = shl nuw nsw i32 %1794, 2, !dbg !7
  %1796 = bitcast float %1791 to i32, !dbg !7
  %1797 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1796), !dbg !7
  %1798 = bitcast i32 %1797 to float, !dbg !7
  %1799 = select i1 %.not3, float %1783, float %1798, !dbg !7
  %1800 = select i1 %.not3, float %1798, float %1789, !dbg !7
  %1801 = select i1 %.not3, float %1790, float %1784, !dbg !7
  %1802 = bitcast float %1801 to i32, !dbg !7
  %1803 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1802), !dbg !7
  %1804 = bitcast i32 %1803 to float, !dbg !7
  %1805 = select i1 %.not3, float %1784, float %1804, !dbg !7
  %1806 = select i1 %.not3, float %1804, float %1790, !dbg !7
  %1807 = extractvalue [2 x <8 x float>] %51, 0, !dbg !7
  %1808 = extractelement <8 x float> %1807, i64 0, !dbg !7
  %1809 = extractelement <8 x float> %1807, i64 1, !dbg !7
  %1810 = extractelement <8 x float> %1807, i64 2, !dbg !7
  %1811 = extractelement <8 x float> %1807, i64 3, !dbg !7
  %1812 = select i1 %.not2, float %1809, float %1808, !dbg !7
  %1813 = bitcast float %1812 to i32, !dbg !7
  %1814 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1813), !dbg !7
  %1815 = bitcast i32 %1814 to float, !dbg !7
  %1816 = select i1 %.not2, float %1808, float %1815, !dbg !7
  %1817 = select i1 %.not2, float %1815, float %1809, !dbg !7
  %1818 = select i1 %.not2, float %1811, float %1810, !dbg !7
  %1819 = bitcast float %1818 to i32, !dbg !7
  %1820 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1819), !dbg !7
  %1821 = bitcast i32 %1820 to float, !dbg !7
  %1822 = select i1 %.not2, float %1810, float %1821, !dbg !7
  %1823 = select i1 %.not2, float %1821, float %1811, !dbg !7
  %1824 = select i1 %.not3, float %1822, float %1816, !dbg !7
  %1825 = bitcast float %1824 to i32, !dbg !7
  %1826 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1825), !dbg !7
  %1827 = bitcast i32 %1826 to float, !dbg !7
  %1828 = select i1 %.not3, float %1816, float %1827, !dbg !7
  %1829 = select i1 %.not3, float %1827, float %1822, !dbg !7
  %1830 = select i1 %.not3, float %1823, float %1817, !dbg !7
  %1831 = bitcast float %1830 to i32, !dbg !7
  %1832 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1831), !dbg !7
  %1833 = bitcast i32 %1832 to float, !dbg !7
  %1834 = select i1 %.not3, float %1817, float %1833, !dbg !7
  %1835 = select i1 %.not3, float %1833, float %1823, !dbg !7
  %1836 = extractvalue [2 x <8 x float>] %50, 0, !dbg !7
  %1837 = extractelement <8 x float> %1836, i64 0, !dbg !7
  %1838 = extractelement <8 x float> %1836, i64 1, !dbg !7
  %1839 = extractelement <8 x float> %1836, i64 2, !dbg !7
  %1840 = extractelement <8 x float> %1836, i64 3, !dbg !7
  %1841 = select i1 %.not2, float %1838, float %1837, !dbg !7
  %1842 = bitcast float %1841 to i32, !dbg !7
  %1843 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1842), !dbg !7
  %1844 = bitcast i32 %1843 to float, !dbg !7
  %1845 = select i1 %.not2, float %1837, float %1844, !dbg !7
  %1846 = select i1 %.not2, float %1844, float %1838, !dbg !7
  %1847 = select i1 %.not2, float %1840, float %1839, !dbg !7
  %1848 = bitcast float %1847 to i32, !dbg !7
  %1849 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1848), !dbg !7
  %1850 = bitcast i32 %1849 to float, !dbg !7
  %1851 = select i1 %.not2, float %1839, float %1850, !dbg !7
  %1852 = select i1 %.not2, float %1850, float %1840, !dbg !7
  %1853 = select i1 %.not3, float %1851, float %1845, !dbg !7
  %1854 = bitcast float %1853 to i32, !dbg !7
  %1855 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1854), !dbg !7
  %1856 = bitcast i32 %1855 to float, !dbg !7
  %1857 = select i1 %.not3, float %1845, float %1856, !dbg !7
  %1858 = select i1 %.not3, float %1856, float %1851, !dbg !7
  %1859 = select i1 %.not3, float %1852, float %1846, !dbg !7
  %1860 = bitcast float %1859 to i32, !dbg !7
  %1861 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1860), !dbg !7
  %1862 = bitcast i32 %1861 to float, !dbg !7
  %1863 = select i1 %.not3, float %1846, float %1862, !dbg !7
  %1864 = select i1 %.not3, float %1862, float %1852, !dbg !7
  %1865 = extractvalue [2 x <8 x float>] %47, 0, !dbg !7
  %1866 = extractelement <8 x float> %1865, i64 0, !dbg !7
  %1867 = extractelement <8 x float> %1865, i64 1, !dbg !7
  %1868 = extractelement <8 x float> %1865, i64 2, !dbg !7
  %1869 = extractelement <8 x float> %1865, i64 3, !dbg !7
  %1870 = select i1 %.not2, float %1867, float %1866, !dbg !7
  %1871 = bitcast float %1870 to i32, !dbg !7
  %1872 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1871), !dbg !7
  %1873 = bitcast i32 %1872 to float, !dbg !7
  %1874 = select i1 %.not2, float %1866, float %1873, !dbg !7
  %1875 = select i1 %.not2, float %1873, float %1867, !dbg !7
  %1876 = select i1 %.not2, float %1869, float %1868, !dbg !7
  %1877 = bitcast float %1876 to i32, !dbg !7
  %1878 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1877), !dbg !7
  %1879 = bitcast i32 %1878 to float, !dbg !7
  %1880 = select i1 %.not2, float %1868, float %1879, !dbg !7
  %1881 = select i1 %.not2, float %1879, float %1869, !dbg !7
  %1882 = select i1 %.not3, float %1880, float %1874, !dbg !7
  %1883 = bitcast float %1882 to i32, !dbg !7
  %1884 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1883), !dbg !7
  %1885 = select i1 %.not3, float %1881, float %1875, !dbg !7
  %1886 = bitcast float %1885 to i32, !dbg !7
  %1887 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1886), !dbg !7
  %1888 = extractvalue [2 x <1 x float>] %48, 0, !dbg !19
  %1889 = shufflevector <1 x float> %1888, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1890 = extractelement <8 x float> %1766, i64 4, !dbg !7
  %1891 = extractelement <8 x float> %1766, i64 5, !dbg !7
  %1892 = extractelement <8 x float> %1766, i64 6, !dbg !7
  %1893 = extractelement <8 x float> %1766, i64 7, !dbg !7
  %1894 = select i1 %.not2, float %1891, float %1890, !dbg !7
  %1895 = bitcast float %1894 to i32, !dbg !7
  %1896 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1895), !dbg !7
  %1897 = bitcast i32 %1896 to float, !dbg !7
  %1898 = select i1 %.not2, float %1890, float %1897, !dbg !7
  %1899 = select i1 %.not2, float %1897, float %1891, !dbg !7
  %1900 = select i1 %.not2, float %1893, float %1892, !dbg !7
  %1901 = bitcast float %1900 to i32, !dbg !7
  %1902 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1901), !dbg !7
  %1903 = bitcast i32 %1902 to float, !dbg !7
  %1904 = select i1 %.not2, float %1892, float %1903, !dbg !7
  %1905 = select i1 %.not2, float %1903, float %1893, !dbg !7
  %1906 = select i1 %.not3, float %1904, float %1898, !dbg !7
  %1907 = bitcast float %1906 to i32, !dbg !7
  %1908 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1907), !dbg !7
  %1909 = bitcast i32 %1908 to float, !dbg !7
  %1910 = select i1 %.not3, float %1898, float %1909, !dbg !7
  %1911 = select i1 %.not3, float %1909, float %1904, !dbg !7
  %1912 = select i1 %.not3, float %1905, float %1899, !dbg !7
  %1913 = bitcast float %1912 to i32, !dbg !7
  %1914 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1913), !dbg !7
  %1915 = bitcast i32 %1914 to float, !dbg !7
  %1916 = select i1 %.not3, float %1899, float %1915, !dbg !7
  %1917 = select i1 %.not3, float %1915, float %1905, !dbg !7
  %1918 = extractelement <8 x float> %1807, i64 4, !dbg !7
  %1919 = extractelement <8 x float> %1807, i64 5, !dbg !7
  %1920 = extractelement <8 x float> %1807, i64 6, !dbg !7
  %1921 = extractelement <8 x float> %1807, i64 7, !dbg !7
  %1922 = select i1 %.not2, float %1919, float %1918, !dbg !7
  %1923 = bitcast float %1922 to i32, !dbg !7
  %1924 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1923), !dbg !7
  %1925 = bitcast i32 %1924 to float, !dbg !7
  %1926 = select i1 %.not2, float %1918, float %1925, !dbg !7
  %1927 = select i1 %.not2, float %1925, float %1919, !dbg !7
  %1928 = select i1 %.not2, float %1921, float %1920, !dbg !7
  %1929 = bitcast float %1928 to i32, !dbg !7
  %1930 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1929), !dbg !7
  %1931 = bitcast i32 %1930 to float, !dbg !7
  %1932 = select i1 %.not2, float %1920, float %1931, !dbg !7
  %1933 = select i1 %.not2, float %1931, float %1921, !dbg !7
  %1934 = select i1 %.not3, float %1932, float %1926, !dbg !7
  %1935 = bitcast float %1934 to i32, !dbg !7
  %1936 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1935), !dbg !7
  %1937 = bitcast i32 %1936 to float, !dbg !7
  %1938 = select i1 %.not3, float %1926, float %1937, !dbg !7
  %1939 = select i1 %.not3, float %1937, float %1932, !dbg !7
  %1940 = select i1 %.not3, float %1933, float %1927, !dbg !7
  %1941 = bitcast float %1940 to i32, !dbg !7
  %1942 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1941), !dbg !7
  %1943 = bitcast i32 %1942 to float, !dbg !7
  %1944 = select i1 %.not3, float %1927, float %1943, !dbg !7
  %1945 = select i1 %.not3, float %1943, float %1933, !dbg !7
  %1946 = extractelement <8 x float> %1836, i64 4, !dbg !7
  %1947 = extractelement <8 x float> %1836, i64 5, !dbg !7
  %1948 = extractelement <8 x float> %1836, i64 6, !dbg !7
  %1949 = extractelement <8 x float> %1836, i64 7, !dbg !7
  %1950 = select i1 %.not2, float %1947, float %1946, !dbg !7
  %1951 = bitcast float %1950 to i32, !dbg !7
  %1952 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1951), !dbg !7
  %1953 = bitcast i32 %1952 to float, !dbg !7
  %1954 = select i1 %.not2, float %1946, float %1953, !dbg !7
  %1955 = select i1 %.not2, float %1953, float %1947, !dbg !7
  %1956 = select i1 %.not2, float %1949, float %1948, !dbg !7
  %1957 = bitcast float %1956 to i32, !dbg !7
  %1958 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1957), !dbg !7
  %1959 = bitcast i32 %1958 to float, !dbg !7
  %1960 = select i1 %.not2, float %1948, float %1959, !dbg !7
  %1961 = select i1 %.not2, float %1959, float %1949, !dbg !7
  %1962 = select i1 %.not3, float %1960, float %1954, !dbg !7
  %1963 = bitcast float %1962 to i32, !dbg !7
  %1964 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1963), !dbg !7
  %1965 = bitcast i32 %1964 to float, !dbg !7
  %1966 = select i1 %.not3, float %1954, float %1965, !dbg !7
  %1967 = select i1 %.not3, float %1965, float %1960, !dbg !7
  %1968 = select i1 %.not3, float %1961, float %1955, !dbg !7
  %1969 = bitcast float %1968 to i32, !dbg !7
  %1970 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1969), !dbg !7
  %1971 = bitcast i32 %1970 to float, !dbg !7
  %1972 = select i1 %.not3, float %1955, float %1971, !dbg !7
  %1973 = select i1 %.not3, float %1971, float %1961, !dbg !7
  %1974 = extractelement <8 x float> %1865, i64 4, !dbg !7
  %1975 = extractelement <8 x float> %1865, i64 5, !dbg !7
  %1976 = extractelement <8 x float> %1865, i64 6, !dbg !7
  %1977 = extractelement <8 x float> %1865, i64 7, !dbg !7
  %1978 = select i1 %.not2, float %1975, float %1974, !dbg !7
  %1979 = bitcast float %1978 to i32, !dbg !7
  %1980 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1979), !dbg !7
  %1981 = bitcast i32 %1980 to float, !dbg !7
  %1982 = select i1 %.not2, float %1974, float %1981, !dbg !7
  %1983 = select i1 %.not2, float %1981, float %1975, !dbg !7
  %1984 = select i1 %.not2, float %1977, float %1976, !dbg !7
  %1985 = bitcast float %1984 to i32, !dbg !7
  %1986 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %1985), !dbg !7
  %1987 = bitcast i32 %1986 to float, !dbg !7
  %1988 = select i1 %.not2, float %1976, float %1987, !dbg !7
  %1989 = select i1 %.not2, float %1987, float %1977, !dbg !7
  %1990 = select i1 %.not3, float %1988, float %1982, !dbg !7
  %1991 = bitcast float %1990 to i32, !dbg !7
  %1992 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1991), !dbg !7
  %1993 = select i1 %.not3, float %1989, float %1983, !dbg !7
  %1994 = bitcast float %1993 to i32, !dbg !7
  %1995 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %1994), !dbg !7
  %1996 = extractvalue [2 x <8 x float>] %52, 1, !dbg !7
  %1997 = extractelement <8 x float> %1996, i64 0, !dbg !7
  %1998 = extractelement <8 x float> %1996, i64 1, !dbg !7
  %1999 = extractelement <8 x float> %1996, i64 2, !dbg !7
  %2000 = extractelement <8 x float> %1996, i64 3, !dbg !7
  %2001 = select i1 %.not2, float %1998, float %1997, !dbg !7
  %2002 = bitcast float %2001 to i32, !dbg !7
  %2003 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2002), !dbg !7
  %2004 = bitcast i32 %2003 to float, !dbg !7
  %2005 = select i1 %.not2, float %1997, float %2004, !dbg !7
  %2006 = select i1 %.not2, float %2004, float %1998, !dbg !7
  %2007 = select i1 %.not2, float %2000, float %1999, !dbg !7
  %2008 = bitcast float %2007 to i32, !dbg !7
  %2009 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2008), !dbg !7
  %2010 = bitcast i32 %2009 to float, !dbg !7
  %2011 = select i1 %.not2, float %1999, float %2010, !dbg !7
  %2012 = select i1 %.not2, float %2010, float %2000, !dbg !7
  %2013 = select i1 %.not3, float %2011, float %2005, !dbg !7
  %2014 = bitcast float %2013 to i32, !dbg !7
  %2015 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2014), !dbg !7
  %2016 = bitcast i32 %2015 to float, !dbg !7
  %2017 = select i1 %.not3, float %2005, float %2016, !dbg !7
  %2018 = select i1 %.not3, float %2016, float %2011, !dbg !7
  %2019 = select i1 %.not3, float %2012, float %2006, !dbg !7
  %2020 = bitcast float %2019 to i32, !dbg !7
  %2021 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2020), !dbg !7
  %2022 = bitcast i32 %2021 to float, !dbg !7
  %2023 = select i1 %.not3, float %2006, float %2022, !dbg !7
  %2024 = select i1 %.not3, float %2022, float %2012, !dbg !7
  %2025 = extractvalue [2 x <8 x float>] %51, 1, !dbg !7
  %2026 = extractelement <8 x float> %2025, i64 0, !dbg !7
  %2027 = extractelement <8 x float> %2025, i64 1, !dbg !7
  %2028 = extractelement <8 x float> %2025, i64 2, !dbg !7
  %2029 = extractelement <8 x float> %2025, i64 3, !dbg !7
  %2030 = select i1 %.not2, float %2027, float %2026, !dbg !7
  %2031 = bitcast float %2030 to i32, !dbg !7
  %2032 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2031), !dbg !7
  %2033 = bitcast i32 %2032 to float, !dbg !7
  %2034 = select i1 %.not2, float %2026, float %2033, !dbg !7
  %2035 = select i1 %.not2, float %2033, float %2027, !dbg !7
  %2036 = select i1 %.not2, float %2029, float %2028, !dbg !7
  %2037 = bitcast float %2036 to i32, !dbg !7
  %2038 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2037), !dbg !7
  %2039 = bitcast i32 %2038 to float, !dbg !7
  %2040 = select i1 %.not2, float %2028, float %2039, !dbg !7
  %2041 = select i1 %.not2, float %2039, float %2029, !dbg !7
  %2042 = select i1 %.not3, float %2040, float %2034, !dbg !7
  %2043 = bitcast float %2042 to i32, !dbg !7
  %2044 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2043), !dbg !7
  %2045 = bitcast i32 %2044 to float, !dbg !7
  %2046 = select i1 %.not3, float %2034, float %2045, !dbg !7
  %2047 = select i1 %.not3, float %2045, float %2040, !dbg !7
  %2048 = select i1 %.not3, float %2041, float %2035, !dbg !7
  %2049 = bitcast float %2048 to i32, !dbg !7
  %2050 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2049), !dbg !7
  %2051 = bitcast i32 %2050 to float, !dbg !7
  %2052 = select i1 %.not3, float %2035, float %2051, !dbg !7
  %2053 = select i1 %.not3, float %2051, float %2041, !dbg !7
  %2054 = extractvalue [2 x <8 x float>] %50, 1, !dbg !7
  %2055 = extractelement <8 x float> %2054, i64 0, !dbg !7
  %2056 = extractelement <8 x float> %2054, i64 1, !dbg !7
  %2057 = extractelement <8 x float> %2054, i64 2, !dbg !7
  %2058 = extractelement <8 x float> %2054, i64 3, !dbg !7
  %2059 = select i1 %.not2, float %2056, float %2055, !dbg !7
  %2060 = bitcast float %2059 to i32, !dbg !7
  %2061 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2060), !dbg !7
  %2062 = bitcast i32 %2061 to float, !dbg !7
  %2063 = select i1 %.not2, float %2055, float %2062, !dbg !7
  %2064 = select i1 %.not2, float %2062, float %2056, !dbg !7
  %2065 = select i1 %.not2, float %2058, float %2057, !dbg !7
  %2066 = bitcast float %2065 to i32, !dbg !7
  %2067 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2066), !dbg !7
  %2068 = bitcast i32 %2067 to float, !dbg !7
  %2069 = select i1 %.not2, float %2057, float %2068, !dbg !7
  %2070 = select i1 %.not2, float %2068, float %2058, !dbg !7
  %2071 = select i1 %.not3, float %2069, float %2063, !dbg !7
  %2072 = bitcast float %2071 to i32, !dbg !7
  %2073 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2072), !dbg !7
  %2074 = bitcast i32 %2073 to float, !dbg !7
  %2075 = select i1 %.not3, float %2063, float %2074, !dbg !7
  %2076 = select i1 %.not3, float %2074, float %2069, !dbg !7
  %2077 = select i1 %.not3, float %2070, float %2064, !dbg !7
  %2078 = bitcast float %2077 to i32, !dbg !7
  %2079 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2078), !dbg !7
  %2080 = bitcast i32 %2079 to float, !dbg !7
  %2081 = select i1 %.not3, float %2064, float %2080, !dbg !7
  %2082 = select i1 %.not3, float %2080, float %2070, !dbg !7
  %2083 = extractvalue [2 x <8 x float>] %47, 1, !dbg !7
  %2084 = extractelement <8 x float> %2083, i64 0, !dbg !7
  %2085 = extractelement <8 x float> %2083, i64 1, !dbg !7
  %2086 = extractelement <8 x float> %2083, i64 2, !dbg !7
  %2087 = extractelement <8 x float> %2083, i64 3, !dbg !7
  %2088 = select i1 %.not2, float %2085, float %2084, !dbg !7
  %2089 = bitcast float %2088 to i32, !dbg !7
  %2090 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2089), !dbg !7
  %2091 = bitcast i32 %2090 to float, !dbg !7
  %2092 = select i1 %.not2, float %2084, float %2091, !dbg !7
  %2093 = select i1 %.not2, float %2091, float %2085, !dbg !7
  %2094 = select i1 %.not2, float %2087, float %2086, !dbg !7
  %2095 = bitcast float %2094 to i32, !dbg !7
  %2096 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2095), !dbg !7
  %2097 = bitcast i32 %2096 to float, !dbg !7
  %2098 = select i1 %.not2, float %2086, float %2097, !dbg !7
  %2099 = select i1 %.not2, float %2097, float %2087, !dbg !7
  %2100 = select i1 %.not3, float %2098, float %2092, !dbg !7
  %2101 = bitcast float %2100 to i32, !dbg !7
  %2102 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2101), !dbg !7
  %2103 = select i1 %.not3, float %2099, float %2093, !dbg !7
  %2104 = bitcast float %2103 to i32, !dbg !7
  %2105 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2104), !dbg !7
  %2106 = extractvalue [2 x <1 x float>] %48, 1, !dbg !19
  %2107 = shufflevector <1 x float> %2106, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %2108 = extractelement <8 x float> %1996, i64 4, !dbg !7
  %2109 = extractelement <8 x float> %1996, i64 5, !dbg !7
  %2110 = extractelement <8 x float> %1996, i64 6, !dbg !7
  %2111 = extractelement <8 x float> %1996, i64 7, !dbg !7
  %2112 = select i1 %.not2, float %2109, float %2108, !dbg !7
  %2113 = bitcast float %2112 to i32, !dbg !7
  %2114 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2113), !dbg !7
  %2115 = bitcast i32 %2114 to float, !dbg !7
  %2116 = select i1 %.not2, float %2108, float %2115, !dbg !7
  %2117 = select i1 %.not2, float %2115, float %2109, !dbg !7
  %2118 = select i1 %.not2, float %2111, float %2110, !dbg !7
  %2119 = bitcast float %2118 to i32, !dbg !7
  %2120 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2119), !dbg !7
  %2121 = bitcast i32 %2120 to float, !dbg !7
  %2122 = select i1 %.not2, float %2110, float %2121, !dbg !7
  %2123 = select i1 %.not2, float %2121, float %2111, !dbg !7
  %2124 = select i1 %.not3, float %2122, float %2116, !dbg !7
  %2125 = bitcast float %2124 to i32, !dbg !7
  %2126 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2125), !dbg !7
  %2127 = bitcast i32 %2126 to float, !dbg !7
  %2128 = select i1 %.not3, float %2116, float %2127, !dbg !7
  %2129 = select i1 %.not3, float %2127, float %2122, !dbg !7
  %2130 = select i1 %.not3, float %2123, float %2117, !dbg !7
  %2131 = bitcast float %2130 to i32, !dbg !7
  %2132 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2131), !dbg !7
  %2133 = bitcast i32 %2132 to float, !dbg !7
  %2134 = select i1 %.not3, float %2117, float %2133, !dbg !7
  %2135 = select i1 %.not3, float %2133, float %2123, !dbg !7
  %2136 = extractelement <8 x float> %2025, i64 4, !dbg !7
  %2137 = extractelement <8 x float> %2025, i64 5, !dbg !7
  %2138 = extractelement <8 x float> %2025, i64 6, !dbg !7
  %2139 = extractelement <8 x float> %2025, i64 7, !dbg !7
  %2140 = select i1 %.not2, float %2137, float %2136, !dbg !7
  %2141 = bitcast float %2140 to i32, !dbg !7
  %2142 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2141), !dbg !7
  %2143 = bitcast i32 %2142 to float, !dbg !7
  %2144 = select i1 %.not2, float %2136, float %2143, !dbg !7
  %2145 = select i1 %.not2, float %2143, float %2137, !dbg !7
  %2146 = select i1 %.not2, float %2139, float %2138, !dbg !7
  %2147 = bitcast float %2146 to i32, !dbg !7
  %2148 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2147), !dbg !7
  %2149 = bitcast i32 %2148 to float, !dbg !7
  %2150 = select i1 %.not2, float %2138, float %2149, !dbg !7
  %2151 = select i1 %.not2, float %2149, float %2139, !dbg !7
  %2152 = select i1 %.not3, float %2150, float %2144, !dbg !7
  %2153 = bitcast float %2152 to i32, !dbg !7
  %2154 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2153), !dbg !7
  %2155 = bitcast i32 %2154 to float, !dbg !7
  %2156 = select i1 %.not3, float %2144, float %2155, !dbg !7
  %2157 = select i1 %.not3, float %2155, float %2150, !dbg !7
  %2158 = select i1 %.not3, float %2151, float %2145, !dbg !7
  %2159 = bitcast float %2158 to i32, !dbg !7
  %2160 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2159), !dbg !7
  %2161 = bitcast i32 %2160 to float, !dbg !7
  %2162 = select i1 %.not3, float %2145, float %2161, !dbg !7
  %2163 = select i1 %.not3, float %2161, float %2151, !dbg !7
  %2164 = extractelement <8 x float> %2054, i64 4, !dbg !7
  %2165 = extractelement <8 x float> %2054, i64 5, !dbg !7
  %2166 = extractelement <8 x float> %2054, i64 6, !dbg !7
  %2167 = extractelement <8 x float> %2054, i64 7, !dbg !7
  %2168 = select i1 %.not2, float %2165, float %2164, !dbg !7
  %2169 = bitcast float %2168 to i32, !dbg !7
  %2170 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2169), !dbg !7
  %2171 = bitcast i32 %2170 to float, !dbg !7
  %2172 = select i1 %.not2, float %2164, float %2171, !dbg !7
  %2173 = select i1 %.not2, float %2171, float %2165, !dbg !7
  %2174 = select i1 %.not2, float %2167, float %2166, !dbg !7
  %2175 = bitcast float %2174 to i32, !dbg !7
  %2176 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2175), !dbg !7
  %2177 = bitcast i32 %2176 to float, !dbg !7
  %2178 = select i1 %.not2, float %2166, float %2177, !dbg !7
  %2179 = select i1 %.not2, float %2177, float %2167, !dbg !7
  %2180 = select i1 %.not3, float %2178, float %2172, !dbg !7
  %2181 = bitcast float %2180 to i32, !dbg !7
  %2182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2181), !dbg !7
  %2183 = bitcast i32 %2182 to float, !dbg !7
  %2184 = select i1 %.not3, float %2172, float %2183, !dbg !7
  %2185 = select i1 %.not3, float %2183, float %2178, !dbg !7
  %2186 = select i1 %.not3, float %2179, float %2173, !dbg !7
  %2187 = bitcast float %2186 to i32, !dbg !7
  %2188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2187), !dbg !7
  %2189 = bitcast i32 %2188 to float, !dbg !7
  %2190 = select i1 %.not3, float %2173, float %2189, !dbg !7
  %2191 = select i1 %.not3, float %2189, float %2179, !dbg !7
  %2192 = extractelement <8 x float> %2083, i64 4, !dbg !7
  %2193 = extractelement <8 x float> %2083, i64 5, !dbg !7
  %2194 = extractelement <8 x float> %2083, i64 6, !dbg !7
  %2195 = extractelement <8 x float> %2083, i64 7, !dbg !7
  %2196 = select i1 %.not2, float %2193, float %2192, !dbg !7
  %2197 = bitcast float %2196 to i32, !dbg !7
  %2198 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2197), !dbg !7
  %2199 = bitcast i32 %2198 to float, !dbg !7
  %2200 = select i1 %.not2, float %2192, float %2199, !dbg !7
  %2201 = select i1 %.not2, float %2199, float %2193, !dbg !7
  %2202 = select i1 %.not2, float %2195, float %2194, !dbg !7
  %2203 = bitcast float %2202 to i32, !dbg !7
  %2204 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1779, i32 %2203), !dbg !7
  %2205 = bitcast i32 %2204 to float, !dbg !7
  %2206 = select i1 %.not2, float %2194, float %2205, !dbg !7
  %2207 = select i1 %.not2, float %2205, float %2195, !dbg !7
  %2208 = select i1 %.not3, float %2206, float %2200, !dbg !7
  %2209 = bitcast float %2208 to i32, !dbg !7
  %2210 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2209), !dbg !7
  %2211 = select i1 %.not3, float %2207, float %2201, !dbg !7
  %2212 = bitcast float %2211 to i32, !dbg !7
  %2213 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1795, i32 %2212), !dbg !7
  br i1 true, label %2214, label %2276, !dbg !21

2214:                                             ; preds = %1763
  %2215 = bitcast i32 %2210 to float, !dbg !7
  %2216 = select i1 %.not3, float %2200, float %2215, !dbg !7
  %2217 = insertelement <4 x float> poison, float %2216, i64 0, !dbg !19
  %2218 = bitcast i32 %2213 to float, !dbg !7
  %2219 = select i1 %.not3, float %2201, float %2218, !dbg !7
  %2220 = insertelement <4 x float> %2217, float %2219, i64 1, !dbg !19
  %2221 = select i1 %.not3, float %2215, float %2206, !dbg !7
  %2222 = insertelement <4 x float> %2220, float %2221, i64 2, !dbg !19
  %2223 = select i1 %.not3, float %2218, float %2207, !dbg !7
  %2224 = insertelement <4 x float> %2222, float %2223, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2107, !dbg !19
  %normalized = fmul <4 x float> %2224, %row.reciprocal, !dbg !19
  %2225 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %2226 = extractelement <4 x half> %2225, i64 3, !dbg !20
  %2227 = extractelement <4 x half> %2225, i64 2, !dbg !20
  %2228 = extractelement <4 x half> %2225, i64 1, !dbg !20
  %2229 = extractelement <4 x half> %2225, i64 0, !dbg !20
  %2230 = bitcast i32 %2102 to float, !dbg !7
  %2231 = select i1 %.not3, float %2092, float %2230, !dbg !7
  %2232 = insertelement <4 x float> poison, float %2231, i64 0, !dbg !19
  %2233 = bitcast i32 %2105 to float, !dbg !7
  %2234 = select i1 %.not3, float %2093, float %2233, !dbg !7
  %2235 = insertelement <4 x float> %2232, float %2234, i64 1, !dbg !19
  %2236 = select i1 %.not3, float %2230, float %2098, !dbg !7
  %2237 = insertelement <4 x float> %2235, float %2236, i64 2, !dbg !19
  %2238 = select i1 %.not3, float %2233, float %2099, !dbg !7
  %2239 = insertelement <4 x float> %2237, float %2238, i64 3, !dbg !19
  %normalized71 = fmul <4 x float> %2239, %row.reciprocal, !dbg !19
  %2240 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !20
  %2241 = extractelement <4 x half> %2240, i64 3, !dbg !20
  %2242 = extractelement <4 x half> %2240, i64 2, !dbg !20
  %2243 = extractelement <4 x half> %2240, i64 1, !dbg !20
  %2244 = extractelement <4 x half> %2240, i64 0, !dbg !20
  %2245 = bitcast i32 %1992 to float, !dbg !7
  %2246 = select i1 %.not3, float %1982, float %2245, !dbg !7
  %2247 = insertelement <4 x float> poison, float %2246, i64 0, !dbg !19
  %2248 = bitcast i32 %1995 to float, !dbg !7
  %2249 = select i1 %.not3, float %1983, float %2248, !dbg !7
  %2250 = insertelement <4 x float> %2247, float %2249, i64 1, !dbg !19
  %2251 = select i1 %.not3, float %2245, float %1988, !dbg !7
  %2252 = insertelement <4 x float> %2250, float %2251, i64 2, !dbg !19
  %2253 = select i1 %.not3, float %2248, float %1989, !dbg !7
  %2254 = insertelement <4 x float> %2252, float %2253, i64 3, !dbg !19
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1889, !dbg !19
  %normalized73 = fmul <4 x float> %2254, %row.reciprocal72, !dbg !19
  %2255 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !20
  %2256 = extractelement <4 x half> %2255, i64 3, !dbg !20
  %2257 = extractelement <4 x half> %2255, i64 2, !dbg !20
  %2258 = extractelement <4 x half> %2255, i64 1, !dbg !20
  %2259 = extractelement <4 x half> %2255, i64 0, !dbg !20
  %2260 = bitcast i32 %1884 to float, !dbg !7
  %2261 = select i1 %.not3, float %1874, float %2260, !dbg !7
  %2262 = insertelement <4 x float> poison, float %2261, i64 0, !dbg !19
  %2263 = bitcast i32 %1887 to float, !dbg !7
  %2264 = select i1 %.not3, float %1875, float %2263, !dbg !7
  %2265 = insertelement <4 x float> %2262, float %2264, i64 1, !dbg !19
  %2266 = select i1 %.not3, float %2260, float %1880, !dbg !7
  %2267 = insertelement <4 x float> %2265, float %2266, i64 2, !dbg !19
  %2268 = select i1 %.not3, float %2263, float %1881, !dbg !7
  %2269 = insertelement <4 x float> %2267, float %2268, i64 3, !dbg !19
  %normalized74 = fmul <4 x float> %2269, %row.reciprocal72, !dbg !19
  %2270 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %2271 = extractelement <4 x half> %2270, i64 3, !dbg !20
  %2272 = extractelement <4 x half> %2270, i64 2, !dbg !20
  %2273 = extractelement <4 x half> %2270, i64 1, !dbg !20
  %2274 = extractelement <4 x half> %2270, i64 0, !dbg !20
  %2275 = shl nuw nsw i64 %17, 19, !dbg !21
  %address.thread609 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local610 = zext nneg i32 %address.thread609 to i64, !dbg !13
  %address.local611 = add nsw i64 %18, %address.local610, !dbg !21
  %address.local612 = lshr i64 %address.local610, 6, !dbg !13
  %address.local613 = shl nuw nsw i64 %address.local612, 4, !dbg !21
  %address.local614 = add nsw i64 %address.local611, %address.local613, !dbg !21
  %address.local615 = and i64 %address.local610, 240, !dbg !21
  %address.local616 = sub nsw i64 %address.local614, %address.local615, !dbg !21
  %address.local617 = add nsw i64 %address.local616, %19, !dbg !21
  %address.local618 = shl nuw nsw i64 %address.local617, 7, !dbg !21
  %address.local619 = add nuw nsw i64 %2275, %address.local618, !dbg !21
  %address.local620 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !21
  %address.local621 = lshr i64 %address.local610, 2, !dbg !13
  %address.local622 = and i64 %address.local621, 12, !dbg !13
  %address.local623 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local622, !dbg !21
  store half %2274, ptr addrspace(1) %address.local623, align 2, !dbg !21
  %address.local624 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !21
  %address.local625 = or disjoint i64 %address.local622, 1, !dbg !21
  %address.local626 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local624, i64 %address.local625, !dbg !21
  store half %2273, ptr addrspace(1) %address.local626, align 2, !dbg !21
  %address.local627 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !21
  %address.local628 = or disjoint i64 %address.local622, 2, !dbg !21
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local627, i64 %address.local628, !dbg !21
  store half %2272, ptr addrspace(1) %address.local629, align 2, !dbg !21
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !21
  %address.local631 = or disjoint i64 %address.local622, 3, !dbg !21
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local630, i64 %address.local631, !dbg !21
  store half %2271, ptr addrspace(1) %address.local632, align 2, !dbg !21
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !21
  %address.local634 = or disjoint i64 %address.local622, 16, !dbg !21
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local633, i64 %address.local634, !dbg !21
  store half %2259, ptr addrspace(1) %address.local635, align 2, !dbg !21
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !21
  %address.local637 = or disjoint i64 %address.local622, 17, !dbg !21
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local636, i64 %address.local637, !dbg !21
  store half %2258, ptr addrspace(1) %address.local638, align 2, !dbg !21
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !21
  %address.local640 = or disjoint i64 %address.local622, 18, !dbg !21
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local639, i64 %address.local640, !dbg !21
  store half %2257, ptr addrspace(1) %address.local641, align 2, !dbg !21
  %address.local642 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local619, !dbg !21
  %address.local643 = or disjoint i64 %address.local622, 19, !dbg !21
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local642, i64 %address.local643, !dbg !21
  store half %2256, ptr addrspace(1) %address.local644, align 2, !dbg !21
  %address.local645 = shl nsw i64 %address.local617, 7, !dbg !21
  %address.local646 = add i64 %address.local645, 8192, !dbg !21
  %address.local647 = add nuw nsw i64 %2275, %address.local646, !dbg !21
  %address.local648 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !21
  %address.local649 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local622, !dbg !21
  store half %2244, ptr addrspace(1) %address.local649, align 2, !dbg !21
  %address.local650 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !21
  %address.local651 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local650, i64 %address.local625, !dbg !21
  store half %2243, ptr addrspace(1) %address.local651, align 2, !dbg !21
  %address.local652 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !21
  %address.local653 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local652, i64 %address.local628, !dbg !21
  store half %2242, ptr addrspace(1) %address.local653, align 2, !dbg !21
  %address.local654 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !21
  %address.local655 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local654, i64 %address.local631, !dbg !21
  store half %2241, ptr addrspace(1) %address.local655, align 2, !dbg !21
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !21
  %address.local657 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local656, i64 %address.local634, !dbg !21
  store half %2229, ptr addrspace(1) %address.local657, align 2, !dbg !21
  %address.local658 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !21
  %address.local659 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local658, i64 %address.local637, !dbg !21
  store half %2228, ptr addrspace(1) %address.local659, align 2, !dbg !21
  %address.local660 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !21
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local660, i64 %address.local640, !dbg !21
  store half %2227, ptr addrspace(1) %address.local661, align 2, !dbg !21
  %address.local662 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local647, !dbg !21
  %address.local663 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local662, i64 %address.local643, !dbg !21
  store half %2226, ptr addrspace(1) %address.local663, align 2, !dbg !21
  br label %2276, !dbg !21

2276:                                             ; preds = %2214, %1763
  br i1 true, label %2277, label %2315, !dbg !21

2277:                                             ; preds = %2276
  %2278 = insertelement <4 x float> poison, float %2184, i64 0, !dbg !19
  %2279 = insertelement <4 x float> %2278, float %2190, i64 1, !dbg !19
  %2280 = insertelement <4 x float> %2279, float %2185, i64 2, !dbg !19
  %2281 = insertelement <4 x float> %2280, float %2191, i64 3, !dbg !19
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2107, !dbg !19
  %normalized76 = fmul <4 x float> %2281, %row.reciprocal75, !dbg !19
  %2282 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !20
  %2283 = extractelement <4 x half> %2282, i64 3, !dbg !20
  %2284 = extractelement <4 x half> %2282, i64 2, !dbg !20
  %2285 = extractelement <4 x half> %2282, i64 1, !dbg !20
  %2286 = extractelement <4 x half> %2282, i64 0, !dbg !20
  %2287 = insertelement <4 x float> poison, float %2075, i64 0, !dbg !19
  %2288 = insertelement <4 x float> %2287, float %2081, i64 1, !dbg !19
  %2289 = insertelement <4 x float> %2288, float %2076, i64 2, !dbg !19
  %2290 = insertelement <4 x float> %2289, float %2082, i64 3, !dbg !19
  %normalized77 = fmul <4 x float> %2290, %row.reciprocal75, !dbg !19
  %2291 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !20
  %2292 = extractelement <4 x half> %2291, i64 3, !dbg !20
  %2293 = extractelement <4 x half> %2291, i64 2, !dbg !20
  %2294 = extractelement <4 x half> %2291, i64 1, !dbg !20
  %2295 = extractelement <4 x half> %2291, i64 0, !dbg !20
  %2296 = insertelement <4 x float> poison, float %1966, i64 0, !dbg !19
  %2297 = insertelement <4 x float> %2296, float %1972, i64 1, !dbg !19
  %2298 = insertelement <4 x float> %2297, float %1967, i64 2, !dbg !19
  %2299 = insertelement <4 x float> %2298, float %1973, i64 3, !dbg !19
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1889, !dbg !19
  %normalized79 = fmul <4 x float> %2299, %row.reciprocal78, !dbg !19
  %2300 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !20
  %2301 = extractelement <4 x half> %2300, i64 3, !dbg !20
  %2302 = extractelement <4 x half> %2300, i64 2, !dbg !20
  %2303 = extractelement <4 x half> %2300, i64 1, !dbg !20
  %2304 = extractelement <4 x half> %2300, i64 0, !dbg !20
  %2305 = insertelement <4 x float> poison, float %1857, i64 0, !dbg !19
  %2306 = insertelement <4 x float> %2305, float %1863, i64 1, !dbg !19
  %2307 = insertelement <4 x float> %2306, float %1858, i64 2, !dbg !19
  %2308 = insertelement <4 x float> %2307, float %1864, i64 3, !dbg !19
  %normalized80 = fmul <4 x float> %2308, %row.reciprocal78, !dbg !19
  %2309 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !20
  %2310 = extractelement <4 x half> %2309, i64 3, !dbg !20
  %2311 = extractelement <4 x half> %2309, i64 2, !dbg !20
  %2312 = extractelement <4 x half> %2309, i64 1, !dbg !20
  %2313 = extractelement <4 x half> %2309, i64 0, !dbg !20
  %2314 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread664 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local665 = zext nneg i32 %address.thread664 to i64, !dbg !13
  %address.local666 = add nsw i64 %18, %address.local665, !dbg !21
  %address.local667 = lshr i64 %address.local665, 6, !dbg !13
  %address.local668 = shl nuw nsw i64 %address.local667, 4, !dbg !21
  %address.local669 = add nsw i64 %address.local666, %address.local668, !dbg !21
  %address.local670 = and i64 %address.local665, 240, !dbg !21
  %address.local671 = sub nsw i64 %address.local669, %address.local670, !dbg !21
  %address.local672 = add nsw i64 %address.local671, %19, !dbg !21
  %address.local673 = shl nuw nsw i64 %address.local672, 7, !dbg !21
  %address.local674 = add nuw nsw i64 %2314, %address.local673, !dbg !21
  %address.local675 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !21
  %address.local676 = lshr i64 %address.local665, 2, !dbg !13
  %address.local677 = and i64 %address.local676, 12, !dbg !13
  %address.local678 = or disjoint i64 %address.local677, 32, !dbg !21
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local675, i64 %address.local678, !dbg !21
  store half %2313, ptr addrspace(1) %address.local679, align 2, !dbg !21
  %address.local680 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !21
  %address.local681 = or disjoint i64 %address.local677, 33, !dbg !21
  %address.local682 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local680, i64 %address.local681, !dbg !21
  store half %2312, ptr addrspace(1) %address.local682, align 2, !dbg !21
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !21
  %address.local684 = or disjoint i64 %address.local677, 34, !dbg !21
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local683, i64 %address.local684, !dbg !21
  store half %2311, ptr addrspace(1) %address.local685, align 2, !dbg !21
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !21
  %address.local687 = or disjoint i64 %address.local677, 35, !dbg !21
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local686, i64 %address.local687, !dbg !21
  store half %2310, ptr addrspace(1) %address.local688, align 2, !dbg !21
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !21
  %address.local690 = or disjoint i64 %address.local677, 48, !dbg !21
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local689, i64 %address.local690, !dbg !21
  store half %2304, ptr addrspace(1) %address.local691, align 2, !dbg !21
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !21
  %address.local693 = or disjoint i64 %address.local677, 49, !dbg !21
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local692, i64 %address.local693, !dbg !21
  store half %2303, ptr addrspace(1) %address.local694, align 2, !dbg !21
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !21
  %address.local696 = or disjoint i64 %address.local677, 50, !dbg !21
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local695, i64 %address.local696, !dbg !21
  store half %2302, ptr addrspace(1) %address.local697, align 2, !dbg !21
  %address.local698 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local674, !dbg !21
  %address.local699 = or i64 %address.local676, 51, !dbg !21
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local698, i64 %address.local699, !dbg !21
  store half %2301, ptr addrspace(1) %address.local700, align 2, !dbg !21
  %address.local701 = shl nsw i64 %address.local672, 7, !dbg !21
  %address.local702 = add i64 %address.local701, 8192, !dbg !21
  %address.local703 = add nuw nsw i64 %2314, %address.local702, !dbg !21
  %address.local704 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !21
  %address.local705 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local704, i64 %address.local678, !dbg !21
  store half %2295, ptr addrspace(1) %address.local705, align 2, !dbg !21
  %address.local706 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !21
  %address.local707 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local706, i64 %address.local681, !dbg !21
  store half %2294, ptr addrspace(1) %address.local707, align 2, !dbg !21
  %address.local708 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !21
  %address.local709 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local708, i64 %address.local684, !dbg !21
  store half %2293, ptr addrspace(1) %address.local709, align 2, !dbg !21
  %address.local710 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !21
  %address.local711 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local710, i64 %address.local687, !dbg !21
  store half %2292, ptr addrspace(1) %address.local711, align 2, !dbg !21
  %address.local712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !21
  %address.local713 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local712, i64 %address.local690, !dbg !21
  store half %2286, ptr addrspace(1) %address.local713, align 2, !dbg !21
  %address.local714 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !21
  %address.local715 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local714, i64 %address.local693, !dbg !21
  store half %2285, ptr addrspace(1) %address.local715, align 2, !dbg !21
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !21
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local716, i64 %address.local696, !dbg !21
  store half %2284, ptr addrspace(1) %address.local717, align 2, !dbg !21
  %address.local718 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local703, !dbg !21
  %address.local719 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local718, i64 %address.local699, !dbg !21
  store half %2283, ptr addrspace(1) %address.local719, align 2, !dbg !21
  br label %2315, !dbg !21

2315:                                             ; preds = %2277, %2276
  br i1 true, label %2316, label %2354, !dbg !21

2316:                                             ; preds = %2315
  %2317 = insertelement <4 x float> poison, float %2156, i64 0, !dbg !19
  %2318 = insertelement <4 x float> %2317, float %2162, i64 1, !dbg !19
  %2319 = insertelement <4 x float> %2318, float %2157, i64 2, !dbg !19
  %2320 = insertelement <4 x float> %2319, float %2163, i64 3, !dbg !19
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2107, !dbg !19
  %normalized82 = fmul <4 x float> %2320, %row.reciprocal81, !dbg !19
  %2321 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !20
  %2322 = extractelement <4 x half> %2321, i64 3, !dbg !20
  %2323 = extractelement <4 x half> %2321, i64 2, !dbg !20
  %2324 = extractelement <4 x half> %2321, i64 1, !dbg !20
  %2325 = extractelement <4 x half> %2321, i64 0, !dbg !20
  %2326 = insertelement <4 x float> poison, float %2046, i64 0, !dbg !19
  %2327 = insertelement <4 x float> %2326, float %2052, i64 1, !dbg !19
  %2328 = insertelement <4 x float> %2327, float %2047, i64 2, !dbg !19
  %2329 = insertelement <4 x float> %2328, float %2053, i64 3, !dbg !19
  %normalized83 = fmul <4 x float> %2329, %row.reciprocal81, !dbg !19
  %2330 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !20
  %2331 = extractelement <4 x half> %2330, i64 3, !dbg !20
  %2332 = extractelement <4 x half> %2330, i64 2, !dbg !20
  %2333 = extractelement <4 x half> %2330, i64 1, !dbg !20
  %2334 = extractelement <4 x half> %2330, i64 0, !dbg !20
  %2335 = insertelement <4 x float> poison, float %1938, i64 0, !dbg !19
  %2336 = insertelement <4 x float> %2335, float %1944, i64 1, !dbg !19
  %2337 = insertelement <4 x float> %2336, float %1939, i64 2, !dbg !19
  %2338 = insertelement <4 x float> %2337, float %1945, i64 3, !dbg !19
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1889, !dbg !19
  %normalized85 = fmul <4 x float> %2338, %row.reciprocal84, !dbg !19
  %2339 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !20
  %2340 = extractelement <4 x half> %2339, i64 3, !dbg !20
  %2341 = extractelement <4 x half> %2339, i64 2, !dbg !20
  %2342 = extractelement <4 x half> %2339, i64 1, !dbg !20
  %2343 = extractelement <4 x half> %2339, i64 0, !dbg !20
  %2344 = insertelement <4 x float> poison, float %1828, i64 0, !dbg !19
  %2345 = insertelement <4 x float> %2344, float %1834, i64 1, !dbg !19
  %2346 = insertelement <4 x float> %2345, float %1829, i64 2, !dbg !19
  %2347 = insertelement <4 x float> %2346, float %1835, i64 3, !dbg !19
  %normalized86 = fmul <4 x float> %2347, %row.reciprocal84, !dbg !19
  %2348 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !20
  %2349 = extractelement <4 x half> %2348, i64 3, !dbg !20
  %2350 = extractelement <4 x half> %2348, i64 2, !dbg !20
  %2351 = extractelement <4 x half> %2348, i64 1, !dbg !20
  %2352 = extractelement <4 x half> %2348, i64 0, !dbg !20
  %2353 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread720 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local721 = zext nneg i32 %address.thread720 to i64, !dbg !13
  %address.local722 = add nsw i64 %18, %address.local721, !dbg !21
  %address.local723 = lshr i64 %address.local721, 6, !dbg !13
  %address.local724 = shl nuw nsw i64 %address.local723, 4, !dbg !21
  %address.local725 = add nsw i64 %address.local722, %address.local724, !dbg !21
  %address.local726 = and i64 %address.local721, 240, !dbg !21
  %address.local727 = sub nsw i64 %address.local725, %address.local726, !dbg !21
  %address.local728 = add nsw i64 %address.local727, %19, !dbg !21
  %address.local729 = shl nuw nsw i64 %address.local728, 7, !dbg !21
  %address.local730 = add nuw nsw i64 %2353, %address.local729, !dbg !21
  %address.local731 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !21
  %address.local732 = lshr i64 %address.local721, 2, !dbg !13
  %address.local733 = and i64 %address.local732, 12, !dbg !13
  %address.local734 = or disjoint i64 %address.local733, 64, !dbg !21
  %address.local735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local731, i64 %address.local734, !dbg !21
  store half %2352, ptr addrspace(1) %address.local735, align 2, !dbg !21
  %address.local736 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !21
  %address.local737 = or disjoint i64 %address.local733, 65, !dbg !21
  %address.local738 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local736, i64 %address.local737, !dbg !21
  store half %2351, ptr addrspace(1) %address.local738, align 2, !dbg !21
  %address.local739 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !21
  %address.local740 = or disjoint i64 %address.local733, 66, !dbg !21
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local739, i64 %address.local740, !dbg !21
  store half %2350, ptr addrspace(1) %address.local741, align 2, !dbg !21
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !21
  %address.local743 = or disjoint i64 %address.local733, 67, !dbg !21
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local742, i64 %address.local743, !dbg !21
  store half %2349, ptr addrspace(1) %address.local744, align 2, !dbg !21
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !21
  %address.local746 = or disjoint i64 %address.local733, 80, !dbg !21
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local745, i64 %address.local746, !dbg !21
  store half %2343, ptr addrspace(1) %address.local747, align 2, !dbg !21
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !21
  %address.local749 = or disjoint i64 %address.local733, 81, !dbg !21
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local748, i64 %address.local749, !dbg !21
  store half %2342, ptr addrspace(1) %address.local750, align 2, !dbg !21
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !21
  %address.local752 = or disjoint i64 %address.local733, 82, !dbg !21
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local751, i64 %address.local752, !dbg !21
  store half %2341, ptr addrspace(1) %address.local753, align 2, !dbg !21
  %address.local754 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local730, !dbg !21
  %address.local755 = or disjoint i64 %address.local733, 83, !dbg !21
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local754, i64 %address.local755, !dbg !21
  store half %2340, ptr addrspace(1) %address.local756, align 2, !dbg !21
  %address.local757 = shl nsw i64 %address.local728, 7, !dbg !21
  %address.local758 = add i64 %address.local757, 8192, !dbg !21
  %address.local759 = add nuw nsw i64 %2353, %address.local758, !dbg !21
  %address.local760 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !21
  %address.local761 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local760, i64 %address.local734, !dbg !21
  store half %2334, ptr addrspace(1) %address.local761, align 2, !dbg !21
  %address.local762 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !21
  %address.local763 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local762, i64 %address.local737, !dbg !21
  store half %2333, ptr addrspace(1) %address.local763, align 2, !dbg !21
  %address.local764 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !21
  %address.local765 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local764, i64 %address.local740, !dbg !21
  store half %2332, ptr addrspace(1) %address.local765, align 2, !dbg !21
  %address.local766 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !21
  %address.local767 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local743, !dbg !21
  store half %2331, ptr addrspace(1) %address.local767, align 2, !dbg !21
  %address.local768 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !21
  %address.local769 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local768, i64 %address.local746, !dbg !21
  store half %2325, ptr addrspace(1) %address.local769, align 2, !dbg !21
  %address.local770 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !21
  %address.local771 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local770, i64 %address.local749, !dbg !21
  store half %2324, ptr addrspace(1) %address.local771, align 2, !dbg !21
  %address.local772 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !21
  %address.local773 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local772, i64 %address.local752, !dbg !21
  store half %2323, ptr addrspace(1) %address.local773, align 2, !dbg !21
  %address.local774 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local759, !dbg !21
  %address.local775 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local774, i64 %address.local755, !dbg !21
  store half %2322, ptr addrspace(1) %address.local775, align 2, !dbg !21
  br label %2354, !dbg !21

2354:                                             ; preds = %2316, %2315
  br i1 true, label %2355, label %2393, !dbg !21

2355:                                             ; preds = %2354
  %2356 = insertelement <4 x float> poison, float %2128, i64 0, !dbg !19
  %2357 = insertelement <4 x float> %2356, float %2134, i64 1, !dbg !19
  %2358 = insertelement <4 x float> %2357, float %2129, i64 2, !dbg !19
  %2359 = insertelement <4 x float> %2358, float %2135, i64 3, !dbg !19
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2107, !dbg !19
  %normalized88 = fmul <4 x float> %2359, %row.reciprocal87, !dbg !19
  %2360 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !20
  %2361 = extractelement <4 x half> %2360, i64 3, !dbg !20
  %2362 = extractelement <4 x half> %2360, i64 2, !dbg !20
  %2363 = extractelement <4 x half> %2360, i64 1, !dbg !20
  %2364 = extractelement <4 x half> %2360, i64 0, !dbg !20
  %2365 = insertelement <4 x float> poison, float %2017, i64 0, !dbg !19
  %2366 = insertelement <4 x float> %2365, float %2023, i64 1, !dbg !19
  %2367 = insertelement <4 x float> %2366, float %2018, i64 2, !dbg !19
  %2368 = insertelement <4 x float> %2367, float %2024, i64 3, !dbg !19
  %normalized89 = fmul <4 x float> %2368, %row.reciprocal87, !dbg !19
  %2369 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !20
  %2370 = extractelement <4 x half> %2369, i64 3, !dbg !20
  %2371 = extractelement <4 x half> %2369, i64 2, !dbg !20
  %2372 = extractelement <4 x half> %2369, i64 1, !dbg !20
  %2373 = extractelement <4 x half> %2369, i64 0, !dbg !20
  %2374 = insertelement <4 x float> poison, float %1910, i64 0, !dbg !19
  %2375 = insertelement <4 x float> %2374, float %1916, i64 1, !dbg !19
  %2376 = insertelement <4 x float> %2375, float %1911, i64 2, !dbg !19
  %2377 = insertelement <4 x float> %2376, float %1917, i64 3, !dbg !19
  %row.reciprocal90 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1889, !dbg !19
  %normalized91 = fmul <4 x float> %2377, %row.reciprocal90, !dbg !19
  %2378 = fptrunc <4 x float> %normalized91 to <4 x half>, !dbg !20
  %2379 = extractelement <4 x half> %2378, i64 3, !dbg !20
  %2380 = extractelement <4 x half> %2378, i64 2, !dbg !20
  %2381 = extractelement <4 x half> %2378, i64 1, !dbg !20
  %2382 = extractelement <4 x half> %2378, i64 0, !dbg !20
  %2383 = insertelement <4 x float> poison, float %1799, i64 0, !dbg !19
  %2384 = insertelement <4 x float> %2383, float %1805, i64 1, !dbg !19
  %2385 = insertelement <4 x float> %2384, float %1800, i64 2, !dbg !19
  %2386 = insertelement <4 x float> %2385, float %1806, i64 3, !dbg !19
  %normalized92 = fmul <4 x float> %2386, %row.reciprocal90, !dbg !19
  %2387 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !20
  %2388 = extractelement <4 x half> %2387, i64 3, !dbg !20
  %2389 = extractelement <4 x half> %2387, i64 2, !dbg !20
  %2390 = extractelement <4 x half> %2387, i64 1, !dbg !20
  %2391 = extractelement <4 x half> %2387, i64 0, !dbg !20
  %2392 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread776 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local777 = zext nneg i32 %address.thread776 to i64, !dbg !13
  %address.local778 = add nsw i64 %18, %address.local777, !dbg !21
  %address.local779 = lshr i64 %address.local777, 6, !dbg !13
  %address.local780 = shl nuw nsw i64 %address.local779, 4, !dbg !21
  %address.local781 = add nsw i64 %address.local778, %address.local780, !dbg !21
  %address.local782 = and i64 %address.local777, 240, !dbg !21
  %address.local783 = sub nsw i64 %address.local781, %address.local782, !dbg !21
  %address.local784 = add nsw i64 %address.local783, %19, !dbg !21
  %address.local785 = shl nuw nsw i64 %address.local784, 7, !dbg !21
  %address.local786 = add nuw nsw i64 %2392, %address.local785, !dbg !21
  %address.local787 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !21
  %address.local788 = lshr i64 %address.local777, 2, !dbg !13
  %address.local789 = and i64 %address.local788, 12, !dbg !13
  %address.local790 = or disjoint i64 %address.local789, 96, !dbg !21
  %address.local791 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local787, i64 %address.local790, !dbg !21
  store half %2391, ptr addrspace(1) %address.local791, align 2, !dbg !21
  %address.local792 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !21
  %address.local793 = or disjoint i64 %address.local789, 97, !dbg !21
  %address.local794 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local792, i64 %address.local793, !dbg !21
  store half %2390, ptr addrspace(1) %address.local794, align 2, !dbg !21
  %address.local795 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !21
  %address.local796 = or disjoint i64 %address.local789, 98, !dbg !21
  %address.local797 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local795, i64 %address.local796, !dbg !21
  store half %2389, ptr addrspace(1) %address.local797, align 2, !dbg !21
  %address.local798 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !21
  %address.local799 = or disjoint i64 %address.local789, 99, !dbg !21
  %address.local800 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local798, i64 %address.local799, !dbg !21
  store half %2388, ptr addrspace(1) %address.local800, align 2, !dbg !21
  %address.local801 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !21
  %address.local802 = or disjoint i64 %address.local789, 112, !dbg !21
  %address.local803 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local801, i64 %address.local802, !dbg !21
  store half %2382, ptr addrspace(1) %address.local803, align 2, !dbg !21
  %address.local804 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !21
  %address.local805 = or disjoint i64 %address.local789, 113, !dbg !21
  %address.local806 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local804, i64 %address.local805, !dbg !21
  store half %2381, ptr addrspace(1) %address.local806, align 2, !dbg !21
  %address.local807 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !21
  %address.local808 = or disjoint i64 %address.local789, 114, !dbg !21
  %address.local809 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local807, i64 %address.local808, !dbg !21
  store half %2380, ptr addrspace(1) %address.local809, align 2, !dbg !21
  %address.local810 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local786, !dbg !21
  %address.local811 = or i64 %address.local788, 115, !dbg !21
  %address.local812 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local810, i64 %address.local811, !dbg !21
  store half %2379, ptr addrspace(1) %address.local812, align 2, !dbg !21
  %address.local813 = shl nsw i64 %address.local784, 7, !dbg !21
  %address.local814 = add i64 %address.local813, 8192, !dbg !21
  %address.local815 = add nuw nsw i64 %2392, %address.local814, !dbg !21
  %address.local816 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !21
  %address.local817 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local816, i64 %address.local790, !dbg !21
  store half %2373, ptr addrspace(1) %address.local817, align 2, !dbg !21
  %address.local818 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !21
  %address.local819 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local818, i64 %address.local793, !dbg !21
  store half %2372, ptr addrspace(1) %address.local819, align 2, !dbg !21
  %address.local820 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !21
  %address.local821 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local820, i64 %address.local796, !dbg !21
  store half %2371, ptr addrspace(1) %address.local821, align 2, !dbg !21
  %address.local822 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !21
  %address.local823 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local822, i64 %address.local799, !dbg !21
  store half %2370, ptr addrspace(1) %address.local823, align 2, !dbg !21
  %address.local824 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !21
  %address.local825 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local824, i64 %address.local802, !dbg !21
  store half %2364, ptr addrspace(1) %address.local825, align 2, !dbg !21
  %address.local826 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !21
  %address.local827 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local826, i64 %address.local805, !dbg !21
  store half %2363, ptr addrspace(1) %address.local827, align 2, !dbg !21
  %address.local828 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !21
  %address.local829 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local828, i64 %address.local808, !dbg !21
  store half %2362, ptr addrspace(1) %address.local829, align 2, !dbg !21
  %address.local830 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local815, !dbg !21
  %address.local831 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local830, i64 %address.local811, !dbg !21
  store half %2361, ptr addrspace(1) %address.local831, align 2, !dbg !21
  br label %2393, !dbg !21

2393:                                             ; preds = %2355, %2354
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0006-test_input_1x32x4096x128-128-128-32-online/pipeline-loop-only")
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
