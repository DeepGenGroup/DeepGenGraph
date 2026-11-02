; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0006-test_input_1x32x4096x128-128-128-32-online/group8-reverse/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0006-test_input_1x32x4096x128-128-128-32-online/group8-reverse/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 5, !dbg !5
  %10 = add nsw i64 %9, %6, !dbg !5
  %11 = lshr i64 %10, 5, !dbg !5
  %12 = and i64 %11, 576460752303423480, !dbg !5
  %13 = and i64 %6, 7, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = lshr i64 %10, 3, !dbg !5
  %16 = sub nsw i64 31, %14, !dbg !5
  %17 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %18 = sext i32 %17 to i64, !dbg !5
  %19 = shl i64 %16, 7, !dbg !9
  %20 = icmp ugt i64 %10, 1023, !dbg !11
  %21 = add nsw i64 %14, -32, !dbg !11
  %22 = select i1 %20, i64 %21, i64 %16, !dbg !11
  %23 = sdiv i64 %22, 32, !dbg !11
  %24 = sext i1 %20 to i64, !dbg !11
  %25 = xor i64 %23, %24, !dbg !11
  %26 = add nsw i64 %15, %25, !dbg !11
  %27 = and i64 %26, 31, !dbg !11
  %28 = shl nsw i64 %16, 7, !dbg !11
  %29 = mul nsw i64 %25, -4096, !dbg !11
  %30 = add i64 %28, %29, !dbg !11
  %31 = shl nuw nsw i64 %27, 19, !dbg !11
  %32 = shl nsw i64 %30, 7, !dbg !11
  %33 = add i64 %31, %32, !dbg !11
  %34 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !39
  %35 = zext nneg i32 %34 to i64, !dbg !12
  %36 = lshr i64 %35, 6, !dbg !12
  %37 = or disjoint i64 %19, 127, !dbg !13
  %38 = add i64 %19, 128, !dbg !13
  %39 = icmp uge i64 %38, %37, !dbg !13
  %40 = sub nsw i64 32, %14, !dbg !15
  %41 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %42 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %43 = getelementptr i8, ptr addrspace(1) %42, i64 64, !dbg !11
  %44 = shl i64 %15, 7, !dbg !16
  %45 = and i64 %44, 3968, !dbg !16
  %46 = shl nsw i64 %18, 12, !dbg !16
  %47 = or disjoint i64 %45, %46, !dbg !16
  %48 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %49 = getelementptr i8, ptr addrspace(1) %48, i64 128, !dbg !11
  %50 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %51 = getelementptr i8, ptr addrspace(1) %50, i64 192, !dbg !11
  br label %52, !dbg !7

52:                                               ; preds = %840, %4
  %53 = phi i64 [ %1771, %840 ], [ 0, %4 ], !dbg !7
  %54 = phi [2 x <8 x float>] [ %1756, %840 ], [ zeroinitializer, %4 ], !dbg !7
  %55 = phi [2 x <1 x float>] [ %1362, %840 ], [ zeroinitializer, %4 ], !dbg !7
  %56 = phi [2 x <1 x float>] [ %1758, %840 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %57 = phi [2 x <8 x float>] [ %1762, %840 ], [ zeroinitializer, %4 ], !dbg !7
  %58 = phi [2 x <8 x float>] [ %1766, %840 ], [ zeroinitializer, %4 ], !dbg !7
  %59 = phi [2 x <8 x float>] [ %1770, %840 ], [ zeroinitializer, %4 ], !dbg !7
  %60 = icmp slt i64 %53, %40, !dbg !7
  br i1 %60, label %61, label %1772, !dbg !7

61:                                               ; preds = %52
  %62 = shl nsw i64 %53, 7, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local126 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local127 = and i64 %address.local126, 1536, !dbg !11
  %address.local128 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local127, !dbg !11
  %address.local129 = ptrtoint ptr addrspace(3) %address.local128 to i64, !dbg !11
  %address.local130 = trunc i64 %address.local129 to i32, !dbg !11
  %63 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local130), !dbg !11
  %64 = zext i32 %63 to i64, !dbg !11
  %65 = inttoptr i64 %64 to ptr addrspace(3), !dbg !11
  %address.local131 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local132 = and i64 %address.local131, 16256, !dbg !11
  %address.local133 = lshr i64 %address.local, 4, !dbg !11
  %address.local134 = and i64 %address.local133, 8, !dbg !11
  %address.local135 = or disjoint i64 %address.local132, %address.local134, !dbg !11
  %address.local136 = getelementptr [2 x i8], ptr addrspace(1) %41, i64 %address.local135, !dbg !11
  %66 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %67 = ptrtoint ptr addrspace(1) %address.local136 to i64, !dbg !11
  %68 = sub i64 %67, %66, !dbg !11
  %dma.buffer.offset931 = trunc i64 %68 to i32, !dbg !11
  %69 = trunc i64 %66 to i32, !dbg !11
  %70 = insertelement <4 x i32> poison, i32 %69, i64 0, !dbg !11
  %71 = lshr i64 %66, 32, !dbg !11
  %72 = trunc i64 %71 to i32, !dbg !11
  %73 = insertelement <4 x i32> %70, i32 %72, i64 1, !dbg !11
  %74 = insertelement <4 x i32> %73, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc932 = insertelement <4 x i32> %74, i32 131072, i64 3, !dbg !11
  %75 = ptrtoint ptr addrspace(3) %65 to i32, !dbg !11
  %dma.buffer.lds933 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %75), !dbg !11
  %76 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds933, i32 %dma.buffer.offset931, <4 x i32> %dma.buffer.rsrc932) #10, !dbg !11
  %address.local137 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local138 = shl nuw nsw i64 %address.local137, 3, !dbg !11
  %address.local139 = and i64 %address.local138, 3584, !dbg !11
  %address.local140 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local139, !dbg !11
  %address.local141 = ptrtoint ptr addrspace(3) %address.local140 to i64, !dbg !11
  %address.local142 = trunc i64 %address.local141 to i32, !dbg !11
  %77 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local142), !dbg !11
  %78 = zext i32 %77 to i64, !dbg !11
  %79 = inttoptr i64 %78 to ptr addrspace(3), !dbg !11
  %address.local143 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local144 = and i64 %address.local143, 16256, !dbg !11
  %address.local145 = lshr i64 %address.local137, 4, !dbg !11
  %address.local146 = and i64 %address.local145, 24, !dbg !11
  %address.local147 = or disjoint i64 %address.local144, %address.local146, !dbg !11
  %address.local148 = getelementptr [2 x i8], ptr addrspace(1) %41, i64 %address.local147, !dbg !11
  %80 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %81 = ptrtoint ptr addrspace(1) %address.local148 to i64, !dbg !11
  %82 = sub i64 %81, %80, !dbg !11
  %dma.buffer.offset928 = trunc i64 %82 to i32, !dbg !11
  %83 = trunc i64 %80 to i32, !dbg !11
  %84 = insertelement <4 x i32> poison, i32 %83, i64 0, !dbg !11
  %85 = lshr i64 %80, 32, !dbg !11
  %86 = trunc i64 %85 to i32, !dbg !11
  %87 = insertelement <4 x i32> %84, i32 %86, i64 1, !dbg !11
  %88 = insertelement <4 x i32> %87, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc929 = insertelement <4 x i32> %88, i32 131072, i64 3, !dbg !11
  %89 = ptrtoint ptr addrspace(3) %79 to i32, !dbg !11
  %dma.buffer.lds930 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %89), !dbg !11
  %90 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds930, i32 %dma.buffer.offset928, <4 x i32> %dma.buffer.rsrc929) #10, !dbg !11
  %91 = lshr i64 %53, 12, !dbg !16
  %92 = add nuw nsw i64 %91, %15, !dbg !16
  %93 = lshr i64 %53, 5, !dbg !16
  %94 = and i64 %93, 127, !dbg !16
  %95 = shl i64 %92, 19, !dbg !16
  %96 = and i64 %95, 16252928, !dbg !16
  %97 = shl i64 %93, 19, !dbg !16
  %98 = shl i64 %53, 14, !dbg !16
  %99 = sub i64 %98, %97, !dbg !16
  %100 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %96, !dbg !16
  %101 = getelementptr [2 x i8], ptr addrspace(1) %100, i64 %94, !dbg !16
  %102 = getelementptr [2 x i8], ptr addrspace(1) %101, i64 %99, !dbg !16
  %address.local149 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local127, !dbg !16
  %address.local150 = ptrtoint ptr addrspace(3) %address.local149 to i64, !dbg !16
  %address.local151 = trunc i64 %address.local150 to i32, !dbg !16
  %103 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local151), !dbg !16
  %104 = zext i32 %103 to i64, !dbg !16
  %105 = inttoptr i64 %104 to ptr addrspace(3), !dbg !16
  %address.local152 = getelementptr [2 x i8], ptr addrspace(1) %102, i64 %address.local135, !dbg !16
  %106 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %107 = ptrtoint ptr addrspace(1) %address.local152 to i64, !dbg !16
  %108 = sub i64 %107, %106, !dbg !16
  %dma.buffer.offset925 = trunc i64 %108 to i32, !dbg !16
  %109 = trunc i64 %106 to i32, !dbg !16
  %110 = insertelement <4 x i32> poison, i32 %109, i64 0, !dbg !16
  %111 = lshr i64 %106, 32, !dbg !16
  %112 = trunc i64 %111 to i32, !dbg !16
  %113 = insertelement <4 x i32> %110, i32 %112, i64 1, !dbg !16
  %114 = insertelement <4 x i32> %113, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc926 = insertelement <4 x i32> %114, i32 131072, i64 3, !dbg !16
  %115 = ptrtoint ptr addrspace(3) %105 to i32, !dbg !16
  %dma.buffer.lds927 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %115), !dbg !16
  %116 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds927, i32 %dma.buffer.offset925, <4 x i32> %dma.buffer.rsrc926) #10, !dbg !16
  %address.local153 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local139, !dbg !16
  %address.local154 = ptrtoint ptr addrspace(3) %address.local153 to i64, !dbg !16
  %address.local155 = trunc i64 %address.local154 to i32, !dbg !16
  %117 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local155), !dbg !16
  %118 = zext i32 %117 to i64, !dbg !16
  %119 = inttoptr i64 %118 to ptr addrspace(3), !dbg !16
  %address.local156 = getelementptr [2 x i8], ptr addrspace(1) %102, i64 %address.local147, !dbg !16
  %120 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %121 = ptrtoint ptr addrspace(1) %address.local156 to i64, !dbg !16
  %122 = sub i64 %121, %120, !dbg !16
  %dma.buffer.offset922 = trunc i64 %122 to i32, !dbg !16
  %123 = trunc i64 %120 to i32, !dbg !16
  %124 = insertelement <4 x i32> poison, i32 %123, i64 0, !dbg !16
  %125 = lshr i64 %120, 32, !dbg !16
  %126 = trunc i64 %125 to i32, !dbg !16
  %127 = insertelement <4 x i32> %124, i32 %126, i64 1, !dbg !16
  %128 = insertelement <4 x i32> %127, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc923 = insertelement <4 x i32> %128, i32 131072, i64 3, !dbg !16
  %129 = ptrtoint ptr addrspace(3) %119 to i32, !dbg !16
  %dma.buffer.lds924 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %129), !dbg !16
  %130 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds924, i32 %dma.buffer.offset922, <4 x i32> %dma.buffer.rsrc923) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local157 = lshr i32 %address.thread, 2, !dbg !12
  %address.local158 = and i32 %address.local157, 4, !dbg !12
  %address.local159 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local160 = and i32 %address.local159, 384, !dbg !12
  %address.local161 = or disjoint i32 %address.local158, %address.local160, !dbg !12
  %address.local162 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local163 = and i32 %address.local162, 120, !dbg !12
  %address.local164 = or disjoint i32 %address.local161, %address.local163, !dbg !12
  %address.local165 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local166 = and i32 %address.local165, 1024, !dbg !12
  %address.local167 = or disjoint i32 %address.local164, %address.local166, !dbg !12
  %address.local168 = zext nneg i32 %address.local167 to i64, !dbg !12
  %address.local169 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local168, !dbg !12
  %131 = load <4 x half>, ptr addrspace(3) %address.local169, align 2, !dbg !12
  %address.local170 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local168, !dbg !12
  %address.local171 = getelementptr i8, ptr addrspace(3) %address.local170, i64 4096, !dbg !12
  %132 = load <4 x half>, ptr addrspace(3) %address.local171, align 2, !dbg !12
  %address.local172 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local168, !dbg !12
  %address.local173 = getelementptr i8, ptr addrspace(3) %address.local172, i64 1024, !dbg !12
  %133 = load <4 x half>, ptr addrspace(3) %address.local173, align 2, !dbg !12
  %134 = fmul <4 x half> %131, splat (half 0xH3015), !dbg !12
  %address.local174 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local168, !dbg !12
  %address.local175 = getelementptr i8, ptr addrspace(3) %address.local174, i64 5120, !dbg !12
  %135 = load <4 x half>, ptr addrspace(3) %address.local175, align 2, !dbg !12
  %136 = fmul <4 x half> %132, splat (half 0xH3015), !dbg !12
  %137 = fmul <4 x half> %133, splat (half 0xH3015), !dbg !12
  %138 = fmul <4 x half> %135, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local176 = lshr i32 %address.thread, 2, !dbg !17
  %address.local177 = and i32 %address.local176, 4, !dbg !17
  %address.local178 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local179 = and i32 %address.local178, 24, !dbg !17
  %address.local180 = or disjoint i32 %address.local177, %address.local179, !dbg !17
  %address.local181 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local182 = and i32 %address.local181, 1024, !dbg !17
  %address.local183 = or disjoint i32 %address.local180, %address.local182, !dbg !17
  %address.local184 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local185 = and i32 %address.local184, 384, !dbg !17
  %address.local186 = or disjoint i32 %address.local183, %address.local185, !dbg !17
  %address.local187 = zext nneg i32 %address.local186 to i64, !dbg !17
  %address.local188 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local187, !dbg !17
  %139 = load <4 x half>, ptr addrspace(3) %address.local188, align 2, !dbg !17
  %140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %139, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local189 = lshr i32 %address.thread, 2, !dbg !17
  %address.local190 = and i32 %address.local189, 4, !dbg !17
  %address.local191 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local192 = and i32 %address.local191, 24, !dbg !17
  %address.local193 = or disjoint i32 %address.local190, %address.local192, !dbg !17
  %address.local194 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local195 = and i32 %address.local194, 1024, !dbg !17
  %address.local196 = or disjoint i32 %address.local193, %address.local195, !dbg !17
  %address.local197 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local198 = and i32 %address.local197, 384, !dbg !17
  %address.local199 = or disjoint i32 %address.local196, %address.local198, !dbg !17
  %address.local200 = zext nneg i32 %address.local199 to i64, !dbg !17
  %address.local201 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local200, !dbg !17
  %address.local202 = getelementptr i8, ptr addrspace(3) %address.local201, i64 4096, !dbg !17
  %141 = load <4 x half>, ptr addrspace(3) %address.local202, align 2, !dbg !17
  %address.local203 = shl nuw nsw i64 %address.local, 7, !dbg !17
  %address.local204 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local203, !dbg !17
  %address.local205 = lshr i64 %address.local, 2, !dbg !17
  %address.local206 = mul nsw i64 %address.local205, -512, !dbg !17
  %address.local207 = getelementptr [2 x i8], ptr addrspace(3) %address.local204, i64 %address.local206, !dbg !17
  %address.local208 = shl nuw nsw i64 %address.local, 1, !dbg !17
  %address.local209 = and i64 %address.local208, 24, !dbg !17
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) %address.local207, i64 %address.local209, !dbg !17
  %address.local211 = lshr i64 %address.local, 2, !dbg !12
  %address.local212 = and i64 %address.local211, 12, !dbg !12
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) %address.local210, i64 %address.local212, !dbg !17
  %address.local214 = and i64 %address.local, 63, !dbg !12
  %address.local215 = lshr i64 %address.local214, 5, !dbg !12
  %address.local216 = mul nuw nsw i64 %address.local215, 1016, !dbg !12
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local213, i64 %address.local216, !dbg !17
  %address.local218 = getelementptr i8, ptr addrspace(3) %address.local217, i64 64, !dbg !17
  %142 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !17
  %143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %141, <4 x float> %140) #9, !dbg !17
  %144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %142, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local219 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local203, !dbg !17
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local219, i64 %address.local212, !dbg !17
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) %address.local220, i64 %address.local216, !dbg !17
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) %address.local221, i64 %address.local206, !dbg !17
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) %address.local222, i64 %address.local209, !dbg !17
  %address.local224 = getelementptr i8, ptr addrspace(3) %address.local223, i64 4160, !dbg !17
  %145 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !17
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local203, !dbg !17
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %address.local225, i64 %address.local206, !dbg !17
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local226, i64 %address.local209, !dbg !17
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) %address.local227, i64 %address.local212, !dbg !17
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local216, !dbg !17
  %address.local230 = getelementptr i8, ptr addrspace(3) %address.local229, i64 128, !dbg !17
  %146 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !17
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %145, <4 x float> %144) #9, !dbg !17
  %148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %146, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local231 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local203, !dbg !17
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) %address.local231, i64 %address.local212, !dbg !17
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) %address.local232, i64 %address.local216, !dbg !17
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) %address.local233, i64 %address.local206, !dbg !17
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) %address.local234, i64 %address.local209, !dbg !17
  %address.local236 = getelementptr i8, ptr addrspace(3) %address.local235, i64 4224, !dbg !17
  %149 = load <4 x half>, ptr addrspace(3) %address.local236, align 2, !dbg !17
  %address.local237 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local203, !dbg !17
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) %address.local237, i64 %address.local206, !dbg !17
  %address.local239 = getelementptr [2 x i8], ptr addrspace(3) %address.local238, i64 %address.local209, !dbg !17
  %address.local240 = getelementptr [2 x i8], ptr addrspace(3) %address.local239, i64 %address.local212, !dbg !17
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local240, i64 %address.local216, !dbg !17
  %address.local242 = getelementptr i8, ptr addrspace(3) %address.local241, i64 192, !dbg !17
  %150 = load <4 x half>, ptr addrspace(3) %address.local242, align 2, !dbg !17
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %149, <4 x float> %148) #9, !dbg !17
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %150, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local243 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local203, !dbg !17
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) %address.local243, i64 %address.local212, !dbg !17
  %address.local245 = getelementptr [2 x i8], ptr addrspace(3) %address.local244, i64 %address.local216, !dbg !17
  %address.local246 = getelementptr [2 x i8], ptr addrspace(3) %address.local245, i64 %address.local206, !dbg !17
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) %address.local246, i64 %address.local209, !dbg !17
  %address.local248 = getelementptr i8, ptr addrspace(3) %address.local247, i64 4288, !dbg !17
  %153 = load <4 x half>, ptr addrspace(3) %address.local248, align 2, !dbg !17
  %address.local249 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local187, !dbg !17
  %address.local250 = getelementptr i8, ptr addrspace(3) %address.local249, i64 1024, !dbg !17
  %154 = load <4 x half>, ptr addrspace(3) %address.local250, align 2, !dbg !17
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %153, <4 x float> %152) #9, !dbg !17
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %154, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local251 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local200, !dbg !17
  %address.local252 = getelementptr i8, ptr addrspace(3) %address.local251, i64 5120, !dbg !17
  %157 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !17
  %address.local253 = add nuw nsw i64 %address.local203, 512, !dbg !17
  %address.local254 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local253, !dbg !17
  %address.local255 = getelementptr [2 x i8], ptr addrspace(3) %address.local254, i64 %address.local206, !dbg !17
  %address.local256 = getelementptr [2 x i8], ptr addrspace(3) %address.local255, i64 %address.local209, !dbg !17
  %address.local257 = getelementptr [2 x i8], ptr addrspace(3) %address.local256, i64 %address.local212, !dbg !17
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) %address.local257, i64 %address.local216, !dbg !17
  %address.local259 = getelementptr i8, ptr addrspace(3) %address.local258, i64 64, !dbg !17
  %158 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !17
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %157, <4 x float> %156) #9, !dbg !17
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %158, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local253, !dbg !17
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %address.local260, i64 %address.local212, !dbg !17
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local216, !dbg !17
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) %address.local262, i64 %address.local206, !dbg !17
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local209, !dbg !17
  %address.local265 = getelementptr i8, ptr addrspace(3) %address.local264, i64 4160, !dbg !17
  %161 = load <4 x half>, ptr addrspace(3) %address.local265, align 2, !dbg !17
  %address.local266 = add nuw nsw i64 %address.local203, 512, !dbg !17
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local266, !dbg !17
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) %address.local267, i64 %address.local206, !dbg !17
  %address.local269 = getelementptr [2 x i8], ptr addrspace(3) %address.local268, i64 %address.local209, !dbg !17
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) %address.local269, i64 %address.local212, !dbg !17
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local216, !dbg !17
  %address.local272 = getelementptr i8, ptr addrspace(3) %address.local271, i64 128, !dbg !17
  %162 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !17
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %161, <4 x float> %160) #9, !dbg !17
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %162, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local266, !dbg !17
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) %address.local273, i64 %address.local212, !dbg !17
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) %address.local274, i64 %address.local216, !dbg !17
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) %address.local275, i64 %address.local206, !dbg !17
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) %address.local276, i64 %address.local209, !dbg !17
  %address.local278 = getelementptr i8, ptr addrspace(3) %address.local277, i64 4224, !dbg !17
  %165 = load <4 x half>, ptr addrspace(3) %address.local278, align 2, !dbg !17
  %address.local279 = add nuw nsw i64 %address.local203, 512, !dbg !17
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local279, !dbg !17
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) %address.local280, i64 %address.local206, !dbg !17
  %address.local282 = getelementptr [2 x i8], ptr addrspace(3) %address.local281, i64 %address.local209, !dbg !17
  %address.local283 = getelementptr [2 x i8], ptr addrspace(3) %address.local282, i64 %address.local212, !dbg !17
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) %address.local283, i64 %address.local216, !dbg !17
  %address.local285 = getelementptr i8, ptr addrspace(3) %address.local284, i64 192, !dbg !17
  %166 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !17
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %165, <4 x float> %164) #9, !dbg !17
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %134, <4 x half> %166, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local279, !dbg !17
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) %address.local286, i64 %address.local212, !dbg !17
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) %address.local287, i64 %address.local216, !dbg !17
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) %address.local288, i64 %address.local206, !dbg !17
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) %address.local289, i64 %address.local209, !dbg !17
  %address.local291 = getelementptr i8, ptr addrspace(3) %address.local290, i64 4288, !dbg !17
  %169 = load <4 x half>, ptr addrspace(3) %address.local291, align 2, !dbg !17
  %170 = load <4 x half>, ptr addrspace(3) %address.local188, align 2, !dbg !17
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %169, <4 x float> %168) #9, !dbg !17
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %170, <4 x float> zeroinitializer) #9, !dbg !17
  %173 = load <4 x half>, ptr addrspace(3) %address.local202, align 2, !dbg !17
  %174 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !17
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %173, <4 x float> %172) #9, !dbg !17
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %174, <4 x float> zeroinitializer) #9, !dbg !17
  %177 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !17
  %178 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !17
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %177, <4 x float> %176) #9, !dbg !17
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %178, <4 x float> zeroinitializer) #9, !dbg !17
  %181 = load <4 x half>, ptr addrspace(3) %address.local236, align 2, !dbg !17
  %182 = load <4 x half>, ptr addrspace(3) %address.local242, align 2, !dbg !17
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %181, <4 x float> %180) #9, !dbg !17
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %182, <4 x float> zeroinitializer) #9, !dbg !17
  %185 = load <4 x half>, ptr addrspace(3) %address.local248, align 2, !dbg !17
  %186 = load <4 x half>, ptr addrspace(3) %address.local250, align 2, !dbg !17
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %185, <4 x float> %184) #9, !dbg !17
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %186, <4 x float> zeroinitializer) #9, !dbg !17
  %189 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !17
  %190 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !17
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %189, <4 x float> %188) #9, !dbg !17
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %190, <4 x float> zeroinitializer) #9, !dbg !17
  %193 = load <4 x half>, ptr addrspace(3) %address.local265, align 2, !dbg !17
  %194 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !17
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %193, <4 x float> %192) #9, !dbg !17
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %194, <4 x float> zeroinitializer) #9, !dbg !17
  %197 = load <4 x half>, ptr addrspace(3) %address.local278, align 2, !dbg !17
  %198 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !17
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %197, <4 x float> %196) #9, !dbg !17
  %200 = load <4 x half>, ptr addrspace(3) %address.local291, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %198, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %200, <4 x float> %201) #9, !dbg !17
  %203 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local130), !dbg !11
  %204 = zext i32 %203 to i64, !dbg !11
  %205 = inttoptr i64 %204 to ptr addrspace(3), !dbg !11
  %address.local292 = getelementptr [2 x i8], ptr addrspace(1) %43, i64 %address.local135, !dbg !11
  %206 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %207 = ptrtoint ptr addrspace(1) %address.local292 to i64, !dbg !11
  %208 = sub i64 %207, %206, !dbg !11
  %dma.buffer.offset919 = trunc i64 %208 to i32, !dbg !11
  %209 = trunc i64 %206 to i32, !dbg !11
  %210 = insertelement <4 x i32> poison, i32 %209, i64 0, !dbg !11
  %211 = lshr i64 %206, 32, !dbg !11
  %212 = trunc i64 %211 to i32, !dbg !11
  %213 = insertelement <4 x i32> %210, i32 %212, i64 1, !dbg !11
  %214 = insertelement <4 x i32> %213, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc920 = insertelement <4 x i32> %214, i32 131072, i64 3, !dbg !11
  %215 = ptrtoint ptr addrspace(3) %205 to i32, !dbg !11
  %dma.buffer.lds921 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %215), !dbg !11
  %216 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds921, i32 %dma.buffer.offset919, <4 x i32> %dma.buffer.rsrc920) #10, !dbg !11
  %217 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local142), !dbg !11
  %218 = zext i32 %217 to i64, !dbg !11
  %219 = inttoptr i64 %218 to ptr addrspace(3), !dbg !11
  %address.local293 = getelementptr [2 x i8], ptr addrspace(1) %43, i64 %address.local147, !dbg !11
  %220 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %221 = ptrtoint ptr addrspace(1) %address.local293 to i64, !dbg !11
  %222 = sub i64 %221, %220, !dbg !11
  %dma.buffer.offset916 = trunc i64 %222 to i32, !dbg !11
  %223 = trunc i64 %220 to i32, !dbg !11
  %224 = insertelement <4 x i32> poison, i32 %223, i64 0, !dbg !11
  %225 = lshr i64 %220, 32, !dbg !11
  %226 = trunc i64 %225 to i32, !dbg !11
  %227 = insertelement <4 x i32> %224, i32 %226, i64 1, !dbg !11
  %228 = insertelement <4 x i32> %227, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc917 = insertelement <4 x i32> %228, i32 131072, i64 3, !dbg !11
  %229 = ptrtoint ptr addrspace(3) %219 to i32, !dbg !11
  %dma.buffer.lds918 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %229), !dbg !11
  %230 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds918, i32 %dma.buffer.offset916, <4 x i32> %dma.buffer.rsrc917) #10, !dbg !11
  %231 = add nuw i64 %53, 1024, !dbg !16
  %232 = icmp slt i64 %231, 0, !dbg !16
  %233 = sub nuw i64 -1025, %53, !dbg !16
  %234 = select i1 %232, i64 %233, i64 %231, !dbg !16
  %235 = sdiv i64 %234, 4096, !dbg !16
  %.lobit = ashr i64 %231, 63, !dbg !16
  %236 = xor i64 %235, %.lobit, !dbg !16
  %237 = add nsw i64 %236, %15, !dbg !16
  %238 = add nsw i64 %93, %47, !dbg !16
  %239 = add nsw i64 %238, 32, !dbg !16
  %240 = icmp slt i64 %238, -32, !dbg !16
  %241 = sub nsw i64 -33, %238, !dbg !16
  %242 = select i1 %240, i64 %241, i64 %239, !dbg !16
  %243 = sdiv i64 %242, 128, !dbg !16
  %244 = sext i1 %240 to i64, !dbg !16
  %245 = xor i64 %243, %244, !dbg !16
  %246 = shl nsw i64 %245, 7, !dbg !16
  %247 = sub i64 %47, %246, !dbg !16
  %248 = shl i64 %237, 19, !dbg !16
  %249 = and i64 %248, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %250 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %249, !dbg !16
  %251 = getelementptr [2 x i8], ptr addrspace(1) %250, i64 %93, !dbg !16
  %252 = getelementptr [2 x i8], ptr addrspace(1) %251, i64 %247, !dbg !16
  %253 = getelementptr i8, ptr addrspace(1) %252, i64 64, !dbg !16
  %254 = getelementptr [2 x i8], ptr addrspace(1) %253, i64 %99, !dbg !16
  %255 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local151), !dbg !16
  %256 = zext i32 %255 to i64, !dbg !16
  %257 = inttoptr i64 %256 to ptr addrspace(3), !dbg !16
  %address.local294 = getelementptr [2 x i8], ptr addrspace(1) %254, i64 %address.local135, !dbg !16
  %258 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %259 = ptrtoint ptr addrspace(1) %address.local294 to i64, !dbg !16
  %260 = sub i64 %259, %258, !dbg !16
  %dma.buffer.offset913 = trunc i64 %260 to i32, !dbg !16
  %261 = trunc i64 %258 to i32, !dbg !16
  %262 = insertelement <4 x i32> poison, i32 %261, i64 0, !dbg !16
  %263 = lshr i64 %258, 32, !dbg !16
  %264 = trunc i64 %263 to i32, !dbg !16
  %265 = insertelement <4 x i32> %262, i32 %264, i64 1, !dbg !16
  %266 = insertelement <4 x i32> %265, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc914 = insertelement <4 x i32> %266, i32 131072, i64 3, !dbg !16
  %267 = ptrtoint ptr addrspace(3) %257 to i32, !dbg !16
  %dma.buffer.lds915 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %267), !dbg !16
  %268 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds915, i32 %dma.buffer.offset913, <4 x i32> %dma.buffer.rsrc914) #10, !dbg !16
  %269 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local155), !dbg !16
  %270 = zext i32 %269 to i64, !dbg !16
  %271 = inttoptr i64 %270 to ptr addrspace(3), !dbg !16
  %address.local295 = getelementptr [2 x i8], ptr addrspace(1) %254, i64 %address.local147, !dbg !16
  %272 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %273 = ptrtoint ptr addrspace(1) %address.local295 to i64, !dbg !16
  %274 = sub i64 %273, %272, !dbg !16
  %dma.buffer.offset910 = trunc i64 %274 to i32, !dbg !16
  %275 = trunc i64 %272 to i32, !dbg !16
  %276 = insertelement <4 x i32> poison, i32 %275, i64 0, !dbg !16
  %277 = lshr i64 %272, 32, !dbg !16
  %278 = trunc i64 %277 to i32, !dbg !16
  %279 = insertelement <4 x i32> %276, i32 %278, i64 1, !dbg !16
  %280 = insertelement <4 x i32> %279, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc911 = insertelement <4 x i32> %280, i32 131072, i64 3, !dbg !16
  %281 = ptrtoint ptr addrspace(3) %271 to i32, !dbg !16
  %dma.buffer.lds912 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %281), !dbg !16
  %282 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds912, i32 %dma.buffer.offset910, <4 x i32> %dma.buffer.rsrc911) #10, !dbg !16
  %283 = load <4 x half>, ptr addrspace(3) %address.local169, align 2, !dbg !12
  %284 = load <4 x half>, ptr addrspace(3) %address.local171, align 2, !dbg !12
  %285 = load <4 x half>, ptr addrspace(3) %address.local173, align 2, !dbg !12
  %286 = fmul <4 x half> %283, splat (half 0xH3015), !dbg !12
  %287 = load <4 x half>, ptr addrspace(3) %address.local175, align 2, !dbg !12
  %288 = fmul <4 x half> %284, splat (half 0xH3015), !dbg !12
  %289 = fmul <4 x half> %285, splat (half 0xH3015), !dbg !12
  %290 = fmul <4 x half> %287, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %291 = load <4 x half>, ptr addrspace(3) %address.local188, align 2, !dbg !17
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %291, <4 x float> %143) #9, !dbg !17
  %293 = load <4 x half>, ptr addrspace(3) %address.local202, align 2, !dbg !17
  %294 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !17
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %288, <4 x half> %293, <4 x float> %292) #9, !dbg !17
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %294, <4 x float> %147) #9, !dbg !17
  %297 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !17
  %298 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !17
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %288, <4 x half> %297, <4 x float> %296) #9, !dbg !17
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %298, <4 x float> %151) #9, !dbg !17
  %301 = load <4 x half>, ptr addrspace(3) %address.local236, align 2, !dbg !17
  %302 = load <4 x half>, ptr addrspace(3) %address.local242, align 2, !dbg !17
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %288, <4 x half> %301, <4 x float> %300) #9, !dbg !17
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %302, <4 x float> %155) #9, !dbg !17
  %305 = load <4 x half>, ptr addrspace(3) %address.local248, align 2, !dbg !17
  %306 = load <4 x half>, ptr addrspace(3) %address.local250, align 2, !dbg !17
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %288, <4 x half> %305, <4 x float> %304) #9, !dbg !17
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %306, <4 x float> %159) #9, !dbg !17
  %309 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !17
  %310 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !17
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %288, <4 x half> %309, <4 x float> %308) #9, !dbg !17
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %310, <4 x float> %163) #9, !dbg !17
  %313 = load <4 x half>, ptr addrspace(3) %address.local265, align 2, !dbg !17
  %314 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !17
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %288, <4 x half> %313, <4 x float> %312) #9, !dbg !17
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %314, <4 x float> %167) #9, !dbg !17
  %317 = load <4 x half>, ptr addrspace(3) %address.local278, align 2, !dbg !17
  %318 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !17
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %288, <4 x half> %317, <4 x float> %316) #9, !dbg !17
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %286, <4 x half> %318, <4 x float> %171) #9, !dbg !17
  %321 = load <4 x half>, ptr addrspace(3) %address.local291, align 2, !dbg !17
  %322 = load <4 x half>, ptr addrspace(3) %address.local188, align 2, !dbg !17
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %288, <4 x half> %321, <4 x float> %320) #9, !dbg !17
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %289, <4 x half> %322, <4 x float> %175) #9, !dbg !17
  %325 = load <4 x half>, ptr addrspace(3) %address.local202, align 2, !dbg !17
  %326 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !17
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %290, <4 x half> %325, <4 x float> %324) #9, !dbg !17
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %289, <4 x half> %326, <4 x float> %179) #9, !dbg !17
  %329 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !17
  %330 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !17
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %290, <4 x half> %329, <4 x float> %328) #9, !dbg !17
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %289, <4 x half> %330, <4 x float> %183) #9, !dbg !17
  %333 = load <4 x half>, ptr addrspace(3) %address.local236, align 2, !dbg !17
  %334 = load <4 x half>, ptr addrspace(3) %address.local242, align 2, !dbg !17
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %290, <4 x half> %333, <4 x float> %332) #9, !dbg !17
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %289, <4 x half> %334, <4 x float> %187) #9, !dbg !17
  %337 = load <4 x half>, ptr addrspace(3) %address.local248, align 2, !dbg !17
  %338 = load <4 x half>, ptr addrspace(3) %address.local250, align 2, !dbg !17
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %290, <4 x half> %337, <4 x float> %336) #9, !dbg !17
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %289, <4 x half> %338, <4 x float> %191) #9, !dbg !17
  %341 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !17
  %342 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !17
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %290, <4 x half> %341, <4 x float> %340) #9, !dbg !17
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %289, <4 x half> %342, <4 x float> %195) #9, !dbg !17
  %345 = load <4 x half>, ptr addrspace(3) %address.local265, align 2, !dbg !17
  %346 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !17
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %290, <4 x half> %345, <4 x float> %344) #9, !dbg !17
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %289, <4 x half> %346, <4 x float> %199) #9, !dbg !17
  %349 = load <4 x half>, ptr addrspace(3) %address.local278, align 2, !dbg !17
  %350 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !17
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %290, <4 x half> %349, <4 x float> %348) #9, !dbg !17
  %352 = load <4 x half>, ptr addrspace(3) %address.local291, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %289, <4 x half> %350, <4 x float> %202) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %290, <4 x half> %352, <4 x float> %353) #9, !dbg !17
  %355 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local130), !dbg !11
  %356 = zext i32 %355 to i64, !dbg !11
  %357 = inttoptr i64 %356 to ptr addrspace(3), !dbg !11
  %address.local296 = getelementptr [2 x i8], ptr addrspace(1) %49, i64 %address.local135, !dbg !11
  %358 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %359 = ptrtoint ptr addrspace(1) %address.local296 to i64, !dbg !11
  %360 = sub i64 %359, %358, !dbg !11
  %dma.buffer.offset907 = trunc i64 %360 to i32, !dbg !11
  %361 = trunc i64 %358 to i32, !dbg !11
  %362 = insertelement <4 x i32> poison, i32 %361, i64 0, !dbg !11
  %363 = lshr i64 %358, 32, !dbg !11
  %364 = trunc i64 %363 to i32, !dbg !11
  %365 = insertelement <4 x i32> %362, i32 %364, i64 1, !dbg !11
  %366 = insertelement <4 x i32> %365, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc908 = insertelement <4 x i32> %366, i32 131072, i64 3, !dbg !11
  %367 = ptrtoint ptr addrspace(3) %357 to i32, !dbg !11
  %dma.buffer.lds909 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %367), !dbg !11
  %368 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds909, i32 %dma.buffer.offset907, <4 x i32> %dma.buffer.rsrc908) #10, !dbg !11
  %369 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local142), !dbg !11
  %370 = zext i32 %369 to i64, !dbg !11
  %371 = inttoptr i64 %370 to ptr addrspace(3), !dbg !11
  %address.local297 = getelementptr [2 x i8], ptr addrspace(1) %49, i64 %address.local147, !dbg !11
  %372 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %373 = ptrtoint ptr addrspace(1) %address.local297 to i64, !dbg !11
  %374 = sub i64 %373, %372, !dbg !11
  %dma.buffer.offset904 = trunc i64 %374 to i32, !dbg !11
  %375 = trunc i64 %372 to i32, !dbg !11
  %376 = insertelement <4 x i32> poison, i32 %375, i64 0, !dbg !11
  %377 = lshr i64 %372, 32, !dbg !11
  %378 = trunc i64 %377 to i32, !dbg !11
  %379 = insertelement <4 x i32> %376, i32 %378, i64 1, !dbg !11
  %380 = insertelement <4 x i32> %379, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc905 = insertelement <4 x i32> %380, i32 131072, i64 3, !dbg !11
  %381 = ptrtoint ptr addrspace(3) %371 to i32, !dbg !11
  %dma.buffer.lds906 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %381), !dbg !11
  %382 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds906, i32 %dma.buffer.offset904, <4 x i32> %dma.buffer.rsrc905) #10, !dbg !11
  %383 = add nuw i64 %53, 2048, !dbg !16
  %384 = icmp slt i64 %383, 0, !dbg !16
  %385 = sub nuw i64 -2049, %53, !dbg !16
  %386 = select i1 %384, i64 %385, i64 %383, !dbg !16
  %387 = sdiv i64 %386, 4096, !dbg !16
  %.lobit2 = ashr i64 %383, 63, !dbg !16
  %388 = xor i64 %387, %.lobit2, !dbg !16
  %389 = add nsw i64 %388, %15, !dbg !16
  %390 = add nsw i64 %238, 64, !dbg !16
  %391 = icmp slt i64 %238, -64, !dbg !16
  %392 = sub nsw i64 -65, %238, !dbg !16
  %393 = select i1 %391, i64 %392, i64 %390, !dbg !16
  %394 = sdiv i64 %393, 128, !dbg !16
  %395 = sext i1 %391 to i64, !dbg !16
  %396 = xor i64 %394, %395, !dbg !16
  %397 = shl nsw i64 %396, 7, !dbg !16
  %398 = sub i64 %47, %397, !dbg !16
  %399 = shl i64 %389, 19, !dbg !16
  %400 = and i64 %399, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %401 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %400, !dbg !16
  %402 = getelementptr [2 x i8], ptr addrspace(1) %401, i64 %93, !dbg !16
  %403 = getelementptr [2 x i8], ptr addrspace(1) %402, i64 %398, !dbg !16
  %404 = getelementptr i8, ptr addrspace(1) %403, i64 128, !dbg !16
  %405 = getelementptr [2 x i8], ptr addrspace(1) %404, i64 %99, !dbg !16
  %406 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local151), !dbg !16
  %407 = zext i32 %406 to i64, !dbg !16
  %408 = inttoptr i64 %407 to ptr addrspace(3), !dbg !16
  %address.local298 = getelementptr [2 x i8], ptr addrspace(1) %405, i64 %address.local135, !dbg !16
  %409 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %410 = ptrtoint ptr addrspace(1) %address.local298 to i64, !dbg !16
  %411 = sub i64 %410, %409, !dbg !16
  %dma.buffer.offset901 = trunc i64 %411 to i32, !dbg !16
  %412 = trunc i64 %409 to i32, !dbg !16
  %413 = insertelement <4 x i32> poison, i32 %412, i64 0, !dbg !16
  %414 = lshr i64 %409, 32, !dbg !16
  %415 = trunc i64 %414 to i32, !dbg !16
  %416 = insertelement <4 x i32> %413, i32 %415, i64 1, !dbg !16
  %417 = insertelement <4 x i32> %416, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc902 = insertelement <4 x i32> %417, i32 131072, i64 3, !dbg !16
  %418 = ptrtoint ptr addrspace(3) %408 to i32, !dbg !16
  %dma.buffer.lds903 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %418), !dbg !16
  %419 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds903, i32 %dma.buffer.offset901, <4 x i32> %dma.buffer.rsrc902) #10, !dbg !16
  %420 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local155), !dbg !16
  %421 = zext i32 %420 to i64, !dbg !16
  %422 = inttoptr i64 %421 to ptr addrspace(3), !dbg !16
  %address.local299 = getelementptr [2 x i8], ptr addrspace(1) %405, i64 %address.local147, !dbg !16
  %423 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %424 = ptrtoint ptr addrspace(1) %address.local299 to i64, !dbg !16
  %425 = sub i64 %424, %423, !dbg !16
  %dma.buffer.offset898 = trunc i64 %425 to i32, !dbg !16
  %426 = trunc i64 %423 to i32, !dbg !16
  %427 = insertelement <4 x i32> poison, i32 %426, i64 0, !dbg !16
  %428 = lshr i64 %423, 32, !dbg !16
  %429 = trunc i64 %428 to i32, !dbg !16
  %430 = insertelement <4 x i32> %427, i32 %429, i64 1, !dbg !16
  %431 = insertelement <4 x i32> %430, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc899 = insertelement <4 x i32> %431, i32 131072, i64 3, !dbg !16
  %432 = ptrtoint ptr addrspace(3) %422 to i32, !dbg !16
  %dma.buffer.lds900 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %432), !dbg !16
  %433 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds900, i32 %dma.buffer.offset898, <4 x i32> %dma.buffer.rsrc899) #10, !dbg !16
  %434 = load <4 x half>, ptr addrspace(3) %address.local169, align 2, !dbg !12
  %435 = load <4 x half>, ptr addrspace(3) %address.local171, align 2, !dbg !12
  %436 = load <4 x half>, ptr addrspace(3) %address.local173, align 2, !dbg !12
  %437 = fmul <4 x half> %434, splat (half 0xH3015), !dbg !12
  %438 = load <4 x half>, ptr addrspace(3) %address.local175, align 2, !dbg !12
  %439 = fmul <4 x half> %435, splat (half 0xH3015), !dbg !12
  %440 = fmul <4 x half> %436, splat (half 0xH3015), !dbg !12
  %441 = fmul <4 x half> %438, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %442 = load <4 x half>, ptr addrspace(3) %address.local188, align 2, !dbg !17
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %442, <4 x float> %295) #9, !dbg !17
  %444 = load <4 x half>, ptr addrspace(3) %address.local202, align 2, !dbg !17
  %445 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !17
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %439, <4 x half> %444, <4 x float> %443) #9, !dbg !17
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %445, <4 x float> %299) #9, !dbg !17
  %448 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !17
  %449 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !17
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %439, <4 x half> %448, <4 x float> %447) #9, !dbg !17
  %451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %449, <4 x float> %303) #9, !dbg !17
  %452 = load <4 x half>, ptr addrspace(3) %address.local236, align 2, !dbg !17
  %453 = load <4 x half>, ptr addrspace(3) %address.local242, align 2, !dbg !17
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %439, <4 x half> %452, <4 x float> %451) #9, !dbg !17
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %453, <4 x float> %307) #9, !dbg !17
  %456 = load <4 x half>, ptr addrspace(3) %address.local248, align 2, !dbg !17
  %457 = load <4 x half>, ptr addrspace(3) %address.local250, align 2, !dbg !17
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %439, <4 x half> %456, <4 x float> %455) #9, !dbg !17
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %457, <4 x float> %311) #9, !dbg !17
  %460 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !17
  %461 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !17
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %439, <4 x half> %460, <4 x float> %459) #9, !dbg !17
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %461, <4 x float> %315) #9, !dbg !17
  %464 = load <4 x half>, ptr addrspace(3) %address.local265, align 2, !dbg !17
  %465 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !17
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %439, <4 x half> %464, <4 x float> %463) #9, !dbg !17
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %465, <4 x float> %319) #9, !dbg !17
  %468 = load <4 x half>, ptr addrspace(3) %address.local278, align 2, !dbg !17
  %469 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !17
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %439, <4 x half> %468, <4 x float> %467) #9, !dbg !17
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %469, <4 x float> %323) #9, !dbg !17
  %472 = load <4 x half>, ptr addrspace(3) %address.local291, align 2, !dbg !17
  %473 = load <4 x half>, ptr addrspace(3) %address.local188, align 2, !dbg !17
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %439, <4 x half> %472, <4 x float> %471) #9, !dbg !17
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %440, <4 x half> %473, <4 x float> %327) #9, !dbg !17
  %476 = load <4 x half>, ptr addrspace(3) %address.local202, align 2, !dbg !17
  %477 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !17
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %441, <4 x half> %476, <4 x float> %475) #9, !dbg !17
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %440, <4 x half> %477, <4 x float> %331) #9, !dbg !17
  %480 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !17
  %481 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !17
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %441, <4 x half> %480, <4 x float> %479) #9, !dbg !17
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %440, <4 x half> %481, <4 x float> %335) #9, !dbg !17
  %484 = load <4 x half>, ptr addrspace(3) %address.local236, align 2, !dbg !17
  %485 = load <4 x half>, ptr addrspace(3) %address.local242, align 2, !dbg !17
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %441, <4 x half> %484, <4 x float> %483) #9, !dbg !17
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %440, <4 x half> %485, <4 x float> %339) #9, !dbg !17
  %488 = load <4 x half>, ptr addrspace(3) %address.local248, align 2, !dbg !17
  %489 = load <4 x half>, ptr addrspace(3) %address.local250, align 2, !dbg !17
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %441, <4 x half> %488, <4 x float> %487) #9, !dbg !17
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %440, <4 x half> %489, <4 x float> %343) #9, !dbg !17
  %492 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !17
  %493 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !17
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %441, <4 x half> %492, <4 x float> %491) #9, !dbg !17
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %440, <4 x half> %493, <4 x float> %347) #9, !dbg !17
  %496 = load <4 x half>, ptr addrspace(3) %address.local265, align 2, !dbg !17
  %497 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !17
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %441, <4 x half> %496, <4 x float> %495) #9, !dbg !17
  %499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %440, <4 x half> %497, <4 x float> %351) #9, !dbg !17
  %500 = load <4 x half>, ptr addrspace(3) %address.local278, align 2, !dbg !17
  %501 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !17
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %441, <4 x half> %500, <4 x float> %499) #9, !dbg !17
  %503 = load <4 x half>, ptr addrspace(3) %address.local291, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %440, <4 x half> %501, <4 x float> %354) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %505 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %441, <4 x half> %503, <4 x float> %504) #9, !dbg !17
  %506 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local130), !dbg !11
  %507 = zext i32 %506 to i64, !dbg !11
  %508 = inttoptr i64 %507 to ptr addrspace(3), !dbg !11
  %address.local300 = getelementptr [2 x i8], ptr addrspace(1) %51, i64 %address.local135, !dbg !11
  %509 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %510 = ptrtoint ptr addrspace(1) %address.local300 to i64, !dbg !11
  %511 = sub i64 %510, %509, !dbg !11
  %dma.buffer.offset895 = trunc i64 %511 to i32, !dbg !11
  %512 = trunc i64 %509 to i32, !dbg !11
  %513 = insertelement <4 x i32> poison, i32 %512, i64 0, !dbg !11
  %514 = lshr i64 %509, 32, !dbg !11
  %515 = trunc i64 %514 to i32, !dbg !11
  %516 = insertelement <4 x i32> %513, i32 %515, i64 1, !dbg !11
  %517 = insertelement <4 x i32> %516, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc896 = insertelement <4 x i32> %517, i32 131072, i64 3, !dbg !11
  %518 = ptrtoint ptr addrspace(3) %508 to i32, !dbg !11
  %dma.buffer.lds897 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %518), !dbg !11
  %519 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds897, i32 %dma.buffer.offset895, <4 x i32> %dma.buffer.rsrc896) #10, !dbg !11
  %520 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local142), !dbg !11
  %521 = zext i32 %520 to i64, !dbg !11
  %522 = inttoptr i64 %521 to ptr addrspace(3), !dbg !11
  %address.local301 = getelementptr [2 x i8], ptr addrspace(1) %51, i64 %address.local147, !dbg !11
  %523 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %524 = ptrtoint ptr addrspace(1) %address.local301 to i64, !dbg !11
  %525 = sub i64 %524, %523, !dbg !11
  %dma.buffer.offset892 = trunc i64 %525 to i32, !dbg !11
  %526 = trunc i64 %523 to i32, !dbg !11
  %527 = insertelement <4 x i32> poison, i32 %526, i64 0, !dbg !11
  %528 = lshr i64 %523, 32, !dbg !11
  %529 = trunc i64 %528 to i32, !dbg !11
  %530 = insertelement <4 x i32> %527, i32 %529, i64 1, !dbg !11
  %531 = insertelement <4 x i32> %530, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc893 = insertelement <4 x i32> %531, i32 131072, i64 3, !dbg !11
  %532 = ptrtoint ptr addrspace(3) %522 to i32, !dbg !11
  %dma.buffer.lds894 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %532), !dbg !11
  %533 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds894, i32 %dma.buffer.offset892, <4 x i32> %dma.buffer.rsrc893) #10, !dbg !11
  %534 = add nuw i64 %53, 3072, !dbg !16
  %535 = icmp slt i64 %534, 0, !dbg !16
  %536 = sub nuw i64 -3073, %53, !dbg !16
  %537 = select i1 %535, i64 %536, i64 %534, !dbg !16
  %538 = sdiv i64 %537, 4096, !dbg !16
  %.lobit3 = ashr i64 %534, 63, !dbg !16
  %539 = xor i64 %538, %.lobit3, !dbg !16
  %540 = add nsw i64 %539, %15, !dbg !16
  %541 = add nsw i64 %238, 96, !dbg !16
  %542 = icmp slt i64 %238, -96, !dbg !16
  %543 = sub nsw i64 -97, %238, !dbg !16
  %544 = select i1 %542, i64 %543, i64 %541, !dbg !16
  %545 = sdiv i64 %544, 128, !dbg !16
  %546 = sext i1 %542 to i64, !dbg !16
  %547 = xor i64 %545, %546, !dbg !16
  %548 = shl nsw i64 %547, 7, !dbg !16
  %549 = sub i64 %47, %548, !dbg !16
  %550 = shl i64 %540, 19, !dbg !16
  %551 = and i64 %550, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %552 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %551, !dbg !16
  %553 = getelementptr [2 x i8], ptr addrspace(1) %552, i64 %93, !dbg !16
  %554 = getelementptr [2 x i8], ptr addrspace(1) %553, i64 %549, !dbg !16
  %555 = getelementptr i8, ptr addrspace(1) %554, i64 192, !dbg !16
  %556 = getelementptr [2 x i8], ptr addrspace(1) %555, i64 %99, !dbg !16
  %557 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local151), !dbg !16
  %558 = zext i32 %557 to i64, !dbg !16
  %559 = inttoptr i64 %558 to ptr addrspace(3), !dbg !16
  %address.local302 = getelementptr [2 x i8], ptr addrspace(1) %556, i64 %address.local135, !dbg !16
  %560 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %561 = ptrtoint ptr addrspace(1) %address.local302 to i64, !dbg !16
  %562 = sub i64 %561, %560, !dbg !16
  %dma.buffer.offset889 = trunc i64 %562 to i32, !dbg !16
  %563 = trunc i64 %560 to i32, !dbg !16
  %564 = insertelement <4 x i32> poison, i32 %563, i64 0, !dbg !16
  %565 = lshr i64 %560, 32, !dbg !16
  %566 = trunc i64 %565 to i32, !dbg !16
  %567 = insertelement <4 x i32> %564, i32 %566, i64 1, !dbg !16
  %568 = insertelement <4 x i32> %567, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc890 = insertelement <4 x i32> %568, i32 131072, i64 3, !dbg !16
  %569 = ptrtoint ptr addrspace(3) %559 to i32, !dbg !16
  %dma.buffer.lds891 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %569), !dbg !16
  %570 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds891, i32 %dma.buffer.offset889, <4 x i32> %dma.buffer.rsrc890) #10, !dbg !16
  %571 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local155), !dbg !16
  %572 = zext i32 %571 to i64, !dbg !16
  %573 = inttoptr i64 %572 to ptr addrspace(3), !dbg !16
  %address.local303 = getelementptr [2 x i8], ptr addrspace(1) %556, i64 %address.local147, !dbg !16
  %574 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %575 = ptrtoint ptr addrspace(1) %address.local303 to i64, !dbg !16
  %576 = sub i64 %575, %574, !dbg !16
  %dma.buffer.offset886 = trunc i64 %576 to i32, !dbg !16
  %577 = trunc i64 %574 to i32, !dbg !16
  %578 = insertelement <4 x i32> poison, i32 %577, i64 0, !dbg !16
  %579 = lshr i64 %574, 32, !dbg !16
  %580 = trunc i64 %579 to i32, !dbg !16
  %581 = insertelement <4 x i32> %578, i32 %580, i64 1, !dbg !16
  %582 = insertelement <4 x i32> %581, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc887 = insertelement <4 x i32> %582, i32 131072, i64 3, !dbg !16
  %583 = ptrtoint ptr addrspace(3) %573 to i32, !dbg !16
  %dma.buffer.lds888 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %583), !dbg !16
  %584 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds888, i32 %dma.buffer.offset886, <4 x i32> %dma.buffer.rsrc887) #10, !dbg !16
  %585 = load <4 x half>, ptr addrspace(3) %address.local169, align 2, !dbg !12
  %586 = load <4 x half>, ptr addrspace(3) %address.local171, align 2, !dbg !12
  %587 = load <4 x half>, ptr addrspace(3) %address.local173, align 2, !dbg !12
  %588 = fmul <4 x half> %585, splat (half 0xH3015), !dbg !12
  %589 = load <4 x half>, ptr addrspace(3) %address.local175, align 2, !dbg !12
  %590 = fmul <4 x half> %586, splat (half 0xH3015), !dbg !12
  %591 = fmul <4 x half> %587, splat (half 0xH3015), !dbg !12
  %592 = fmul <4 x half> %589, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %593 = add nuw nsw i64 %93, %15, !dbg !18
  %594 = shl i64 %593, 19, !dbg !18
  %595 = and i64 %594, 16252928, !dbg !18
  %596 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %595, !dbg !18
  %597 = getelementptr [2 x i8], ptr addrspace(1) %596, i64 %99, !dbg !18
  %598 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local130), !dbg !18
  %599 = zext i32 %598 to i64, !dbg !18
  %600 = inttoptr i64 %599 to ptr addrspace(3), !dbg !18
  %address.local304 = shl nuw nsw i64 %address.local, 5, !dbg !18
  %address.local305 = and i64 %address.local304, 8064, !dbg !18
  %address.local306 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local307 = and i64 %address.local306, 24, !dbg !18
  %address.local308 = or disjoint i64 %address.local305, %address.local307, !dbg !18
  %address.local309 = getelementptr [2 x i8], ptr addrspace(1) %597, i64 %address.local308, !dbg !18
  %601 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %602 = ptrtoint ptr addrspace(1) %address.local309 to i64, !dbg !18
  %603 = sub i64 %602, %601, !dbg !18
  %dma.buffer.offset883 = trunc i64 %603 to i32, !dbg !18
  %604 = trunc i64 %601 to i32, !dbg !18
  %605 = insertelement <4 x i32> poison, i32 %604, i64 0, !dbg !18
  %606 = lshr i64 %601, 32, !dbg !18
  %607 = trunc i64 %606 to i32, !dbg !18
  %608 = insertelement <4 x i32> %605, i32 %607, i64 1, !dbg !18
  %609 = insertelement <4 x i32> %608, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc884 = insertelement <4 x i32> %609, i32 131072, i64 3, !dbg !18
  %610 = ptrtoint ptr addrspace(3) %600 to i32, !dbg !18
  %dma.buffer.lds885 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %610), !dbg !18
  %611 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds885, i32 %dma.buffer.offset883, <4 x i32> %dma.buffer.rsrc884) #10, !dbg !18
  %612 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local142), !dbg !18
  %613 = zext i32 %612 to i64, !dbg !18
  %614 = inttoptr i64 %613 to ptr addrspace(3), !dbg !18
  %address.local310 = shl nuw nsw i64 %address.local137, 5, !dbg !18
  %address.local311 = and i64 %address.local310, 16256, !dbg !18
  %address.local312 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local313 = and i64 %address.local312, 24, !dbg !18
  %address.local314 = or disjoint i64 %address.local311, %address.local313, !dbg !18
  %address.local315 = getelementptr [2 x i8], ptr addrspace(1) %597, i64 %address.local314, !dbg !18
  %615 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %616 = ptrtoint ptr addrspace(1) %address.local315 to i64, !dbg !18
  %617 = sub i64 %616, %615, !dbg !18
  %dma.buffer.offset880 = trunc i64 %617 to i32, !dbg !18
  %618 = trunc i64 %615 to i32, !dbg !18
  %619 = insertelement <4 x i32> poison, i32 %618, i64 0, !dbg !18
  %620 = lshr i64 %615, 32, !dbg !18
  %621 = trunc i64 %620 to i32, !dbg !18
  %622 = insertelement <4 x i32> %619, i32 %621, i64 1, !dbg !18
  %623 = insertelement <4 x i32> %622, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc881 = insertelement <4 x i32> %623, i32 131072, i64 3, !dbg !18
  %624 = ptrtoint ptr addrspace(3) %614 to i32, !dbg !18
  %dma.buffer.lds882 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %624), !dbg !18
  %625 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds882, i32 %dma.buffer.offset880, <4 x i32> %dma.buffer.rsrc881) #10, !dbg !18
  %626 = load <4 x half>, ptr addrspace(3) %address.local188, align 2, !dbg !17
  %627 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %588, <4 x half> %626, <4 x float> %446) #9, !dbg !17
  %628 = load <4 x half>, ptr addrspace(3) %address.local202, align 2, !dbg !17
  %629 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !17
  %630 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %590, <4 x half> %628, <4 x float> %627) #9, !dbg !17
  %631 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %588, <4 x half> %629, <4 x float> %450) #9, !dbg !17
  %632 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !17
  %633 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !17
  %634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %590, <4 x half> %632, <4 x float> %631) #9, !dbg !17
  %635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %588, <4 x half> %633, <4 x float> %454) #9, !dbg !17
  %636 = load <4 x half>, ptr addrspace(3) %address.local236, align 2, !dbg !17
  %637 = load <4 x half>, ptr addrspace(3) %address.local242, align 2, !dbg !17
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %590, <4 x half> %636, <4 x float> %635) #9, !dbg !17
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %588, <4 x half> %637, <4 x float> %458) #9, !dbg !17
  %640 = load <4 x half>, ptr addrspace(3) %address.local248, align 2, !dbg !17
  %641 = load <4 x half>, ptr addrspace(3) %address.local250, align 2, !dbg !17
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %590, <4 x half> %640, <4 x float> %639) #9, !dbg !17
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %588, <4 x half> %641, <4 x float> %462) #9, !dbg !17
  %644 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !17
  %645 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !17
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %590, <4 x half> %644, <4 x float> %643) #9, !dbg !17
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %588, <4 x half> %645, <4 x float> %466) #9, !dbg !17
  %648 = load <4 x half>, ptr addrspace(3) %address.local265, align 2, !dbg !17
  %649 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !17
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %590, <4 x half> %648, <4 x float> %647) #9, !dbg !17
  %651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %588, <4 x half> %649, <4 x float> %470) #9, !dbg !17
  %652 = load <4 x half>, ptr addrspace(3) %address.local278, align 2, !dbg !17
  %653 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !17
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %590, <4 x half> %652, <4 x float> %651) #9, !dbg !17
  %655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %588, <4 x half> %653, <4 x float> %474) #9, !dbg !17
  %656 = load <4 x half>, ptr addrspace(3) %address.local291, align 2, !dbg !17
  %657 = load <4 x half>, ptr addrspace(3) %address.local188, align 2, !dbg !17
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %590, <4 x half> %656, <4 x float> %655) #9, !dbg !17
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %591, <4 x half> %657, <4 x float> %478) #9, !dbg !17
  %660 = load <4 x half>, ptr addrspace(3) %address.local202, align 2, !dbg !17
  %661 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !17
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %592, <4 x half> %660, <4 x float> %659) #9, !dbg !17
  %663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %591, <4 x half> %661, <4 x float> %482) #9, !dbg !17
  %664 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !17
  %665 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !17
  %666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %592, <4 x half> %664, <4 x float> %663) #9, !dbg !17
  %667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %591, <4 x half> %665, <4 x float> %486) #9, !dbg !17
  %668 = load <4 x half>, ptr addrspace(3) %address.local236, align 2, !dbg !17
  %669 = load <4 x half>, ptr addrspace(3) %address.local242, align 2, !dbg !17
  %670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %592, <4 x half> %668, <4 x float> %667) #9, !dbg !17
  %671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %591, <4 x half> %669, <4 x float> %490) #9, !dbg !17
  %672 = load <4 x half>, ptr addrspace(3) %address.local248, align 2, !dbg !17
  %673 = load <4 x half>, ptr addrspace(3) %address.local250, align 2, !dbg !17
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %592, <4 x half> %672, <4 x float> %671) #9, !dbg !17
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %591, <4 x half> %673, <4 x float> %494) #9, !dbg !17
  %676 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !17
  %677 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !17
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %592, <4 x half> %676, <4 x float> %675) #9, !dbg !17
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %591, <4 x half> %677, <4 x float> %498) #9, !dbg !17
  %680 = load <4 x half>, ptr addrspace(3) %address.local265, align 2, !dbg !17
  %681 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !17
  %682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %592, <4 x half> %680, <4 x float> %679) #9, !dbg !17
  %683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %591, <4 x half> %681, <4 x float> %502) #9, !dbg !17
  %684 = load <4 x half>, ptr addrspace(3) %address.local278, align 2, !dbg !17
  %685 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !17
  %686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %592, <4 x half> %684, <4 x float> %683) #9, !dbg !17
  %687 = load <4 x half>, ptr addrspace(3) %address.local291, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %688 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %591, <4 x half> %685, <4 x float> %505) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %689 = or disjoint i64 %62, 127, !dbg !13
  %690 = icmp uge i64 %19, %689, !dbg !13
  %691 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %592, <4 x half> %687, <4 x float> %688) #9, !dbg !17
  %692 = and i1 %39, %690, !dbg !13
  br i1 %692, label %693, label %694, !dbg !13

693:                                              ; preds = %61
  br label %838, !dbg !13

694:                                              ; preds = %61
  %695 = add i64 %19, %35, !dbg !13
  %696 = shl nuw nsw i64 %36, 4, !dbg !13
  %697 = add i64 %695, %696, !dbg !13
  %698 = and i32 %34, 240, !dbg !13
  %index.compact = zext nneg i32 %698 to i64, !dbg !13
  %699 = sub i64 %697, %index.compact, !dbg !13
  %700 = and i64 %35, 48, !dbg !13
  %701 = or disjoint i64 %62, %700, !dbg !13
  %702 = add i64 %699, 1, !dbg !23
  %703 = lshr i64 %702, 32, !dbg !24
  %compare.high = trunc i64 %703 to i32, !dbg !24
  %704 = lshr i64 %701, 32, !dbg !24
  %compare.high609 = trunc i64 %704 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high609, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high609, !dbg !24
  %compare.low.base = trunc i64 %702 to i32, !dbg !24
  %compare.low.base610 = trunc i64 %701 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base610, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %705 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low611 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result612 = select i1 %compare.high.equal, i1 %compare.low611, i1 %compare.high.order, !dbg !24
  %706 = select i1 %compare.result612, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base613 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local614 = or i32 %compare.local.base613, 2, !dbg !24
  %compare.low615 = icmp ugt i32 %compare.low.base, %compare.local614, !dbg !24
  %compare.result616 = select i1 %compare.high.equal, i1 %compare.low615, i1 %compare.high.order, !dbg !24
  %707 = select i1 %compare.result616, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base617 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local618 = or i32 %compare.local.base617, 3, !dbg !24
  %compare.low619 = icmp ugt i32 %compare.low.base, %compare.local618, !dbg !24
  %compare.result620 = select i1 %compare.high.equal, i1 %compare.low619, i1 %compare.high.order, !dbg !24
  %708 = select i1 %compare.result620, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base621 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local622 = or i32 %compare.local.base621, 4, !dbg !24
  %compare.low623 = icmp ugt i32 %compare.low.base, %compare.local622, !dbg !24
  %compare.result624 = select i1 %compare.high.equal, i1 %compare.low623, i1 %compare.high.order, !dbg !24
  %709 = select i1 %compare.result624, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base625 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local626 = or i32 %compare.local.base625, 5, !dbg !24
  %compare.low627 = icmp ugt i32 %compare.low.base, %compare.local626, !dbg !24
  %compare.result628 = select i1 %compare.high.equal, i1 %compare.low627, i1 %compare.high.order, !dbg !24
  %710 = select i1 %compare.result628, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base629 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local630 = or i32 %compare.local.base629, 6, !dbg !24
  %compare.low631 = icmp ugt i32 %compare.low.base, %compare.local630, !dbg !24
  %compare.result632 = select i1 %compare.high.equal, i1 %compare.low631, i1 %compare.high.order, !dbg !24
  %711 = select i1 %compare.result632, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base633 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local634 = or i32 %compare.local.base633, 7, !dbg !24
  %compare.low635 = icmp ugt i32 %compare.low.base, %compare.local634, !dbg !24
  %compare.result636 = select i1 %compare.high.equal, i1 %compare.low635, i1 %compare.high.order, !dbg !24
  %712 = select i1 %compare.result636, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base637 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local638 = or i32 %compare.local.base637, 8, !dbg !24
  %compare.low639 = icmp ugt i32 %compare.low.base, %compare.local638, !dbg !24
  %compare.result640 = select i1 %compare.high.equal, i1 %compare.low639, i1 %compare.high.order, !dbg !24
  %713 = select i1 %compare.result640, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base641 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local642 = or i32 %compare.local.base641, 9, !dbg !24
  %compare.low643 = icmp ugt i32 %compare.low.base, %compare.local642, !dbg !24
  %compare.result644 = select i1 %compare.high.equal, i1 %compare.low643, i1 %compare.high.order, !dbg !24
  %714 = select i1 %compare.result644, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base645 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local646 = or i32 %compare.local.base645, 10, !dbg !24
  %compare.low647 = icmp ugt i32 %compare.low.base, %compare.local646, !dbg !24
  %compare.result648 = select i1 %compare.high.equal, i1 %compare.low647, i1 %compare.high.order, !dbg !24
  %715 = select i1 %compare.result648, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base649 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local650 = or i32 %compare.local.base649, 11, !dbg !24
  %compare.low651 = icmp ugt i32 %compare.low.base, %compare.local650, !dbg !24
  %compare.result652 = select i1 %compare.high.equal, i1 %compare.low651, i1 %compare.high.order, !dbg !24
  %716 = select i1 %compare.result652, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base653 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local654 = or i32 %compare.local.base653, 12, !dbg !24
  %compare.low655 = icmp ugt i32 %compare.low.base, %compare.local654, !dbg !24
  %compare.result656 = select i1 %compare.high.equal, i1 %compare.low655, i1 %compare.high.order, !dbg !24
  %717 = select i1 %compare.result656, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base657 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local658 = or i32 %compare.local.base657, 13, !dbg !24
  %compare.low659 = icmp ugt i32 %compare.low.base, %compare.local658, !dbg !24
  %compare.result660 = select i1 %compare.high.equal, i1 %compare.low659, i1 %compare.high.order, !dbg !24
  %718 = select i1 %compare.result660, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base661 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local662 = or i32 %compare.local.base661, 14, !dbg !24
  %compare.low663 = icmp ugt i32 %compare.low.base, %compare.local662, !dbg !24
  %compare.result664 = select i1 %compare.high.equal, i1 %compare.low663, i1 %compare.high.order, !dbg !24
  %719 = select i1 %compare.result664, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base665 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local666 = or i32 %compare.local.base665, 15, !dbg !24
  %compare.low667 = icmp ugt i32 %compare.low.base, %compare.local666, !dbg !24
  %compare.result668 = select i1 %compare.high.equal, i1 %compare.low667, i1 %compare.high.order, !dbg !24
  %720 = select i1 %compare.result668, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base669 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local670 = or i32 %compare.local.base669, 64, !dbg !24
  %compare.low671 = icmp ugt i32 %compare.low.base, %compare.local670, !dbg !24
  %compare.result672 = select i1 %compare.high.equal, i1 %compare.low671, i1 %compare.high.order, !dbg !24
  %721 = select i1 %compare.result672, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base673 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local674 = or i32 %compare.local.base673, 65, !dbg !24
  %compare.low675 = icmp ugt i32 %compare.low.base, %compare.local674, !dbg !24
  %compare.result676 = select i1 %compare.high.equal, i1 %compare.low675, i1 %compare.high.order, !dbg !24
  %722 = select i1 %compare.result676, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base677 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local678 = or i32 %compare.local.base677, 66, !dbg !24
  %compare.low679 = icmp ugt i32 %compare.low.base, %compare.local678, !dbg !24
  %compare.result680 = select i1 %compare.high.equal, i1 %compare.low679, i1 %compare.high.order, !dbg !24
  %723 = select i1 %compare.result680, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base681 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local682 = or i32 %compare.local.base681, 67, !dbg !24
  %compare.low683 = icmp ugt i32 %compare.low.base, %compare.local682, !dbg !24
  %compare.result684 = select i1 %compare.high.equal, i1 %compare.low683, i1 %compare.high.order, !dbg !24
  %724 = select i1 %compare.result684, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base685 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local686 = or i32 %compare.local.base685, 68, !dbg !24
  %compare.low687 = icmp ugt i32 %compare.low.base, %compare.local686, !dbg !24
  %compare.result688 = select i1 %compare.high.equal, i1 %compare.low687, i1 %compare.high.order, !dbg !24
  %725 = select i1 %compare.result688, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base689 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local690 = or i32 %compare.local.base689, 69, !dbg !24
  %compare.low691 = icmp ugt i32 %compare.low.base, %compare.local690, !dbg !24
  %compare.result692 = select i1 %compare.high.equal, i1 %compare.low691, i1 %compare.high.order, !dbg !24
  %726 = select i1 %compare.result692, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base693 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local694 = or i32 %compare.local.base693, 70, !dbg !24
  %compare.low695 = icmp ugt i32 %compare.low.base, %compare.local694, !dbg !24
  %compare.result696 = select i1 %compare.high.equal, i1 %compare.low695, i1 %compare.high.order, !dbg !24
  %727 = select i1 %compare.result696, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base697 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local698 = or i32 %compare.local.base697, 71, !dbg !24
  %compare.low699 = icmp ugt i32 %compare.low.base, %compare.local698, !dbg !24
  %compare.result700 = select i1 %compare.high.equal, i1 %compare.low699, i1 %compare.high.order, !dbg !24
  %728 = select i1 %compare.result700, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base701 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local702 = or i32 %compare.local.base701, 72, !dbg !24
  %compare.low703 = icmp ugt i32 %compare.low.base, %compare.local702, !dbg !24
  %compare.result704 = select i1 %compare.high.equal, i1 %compare.low703, i1 %compare.high.order, !dbg !24
  %729 = select i1 %compare.result704, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base705 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local706 = or i32 %compare.local.base705, 73, !dbg !24
  %compare.low707 = icmp ugt i32 %compare.low.base, %compare.local706, !dbg !24
  %compare.result708 = select i1 %compare.high.equal, i1 %compare.low707, i1 %compare.high.order, !dbg !24
  %730 = select i1 %compare.result708, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base709 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local710 = or i32 %compare.local.base709, 74, !dbg !24
  %compare.low711 = icmp ugt i32 %compare.low.base, %compare.local710, !dbg !24
  %compare.result712 = select i1 %compare.high.equal, i1 %compare.low711, i1 %compare.high.order, !dbg !24
  %731 = select i1 %compare.result712, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base713 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local714 = or i32 %compare.local.base713, 75, !dbg !24
  %compare.low715 = icmp ugt i32 %compare.low.base, %compare.local714, !dbg !24
  %compare.result716 = select i1 %compare.high.equal, i1 %compare.low715, i1 %compare.high.order, !dbg !24
  %732 = select i1 %compare.result716, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base717 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local718 = or i32 %compare.local.base717, 76, !dbg !24
  %compare.low719 = icmp ugt i32 %compare.low.base, %compare.local718, !dbg !24
  %compare.result720 = select i1 %compare.high.equal, i1 %compare.low719, i1 %compare.high.order, !dbg !24
  %733 = select i1 %compare.result720, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base721 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local722 = or i32 %compare.local.base721, 77, !dbg !24
  %compare.low723 = icmp ugt i32 %compare.low.base, %compare.local722, !dbg !24
  %compare.result724 = select i1 %compare.high.equal, i1 %compare.low723, i1 %compare.high.order, !dbg !24
  %734 = select i1 %compare.result724, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base725 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local726 = or i32 %compare.local.base725, 78, !dbg !24
  %compare.low727 = icmp ugt i32 %compare.low.base, %compare.local726, !dbg !24
  %compare.result728 = select i1 %compare.high.equal, i1 %compare.low727, i1 %compare.high.order, !dbg !24
  %735 = select i1 %compare.result728, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base729 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base610, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local730 = or i32 %compare.local.base729, 79, !dbg !24
  %compare.low731 = icmp ugt i32 %compare.low.base, %compare.local730, !dbg !24
  %compare.result732 = select i1 %compare.high.equal, i1 %compare.low731, i1 %compare.high.order, !dbg !24
  %736 = select i1 %compare.result732, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %737 = add i64 %699, 65, !dbg !23
  %738 = lshr i64 %737, 32, !dbg !24
  %compare.high733 = trunc i64 %738 to i32, !dbg !24
  %739 = lshr i64 %701, 32, !dbg !24
  %compare.high734 = trunc i64 %739 to i32, !dbg !24
  %compare.high.equal735 = icmp eq i32 %compare.high733, %compare.high734, !dbg !24
  %compare.high.order736 = icmp ugt i32 %compare.high733, %compare.high734, !dbg !24
  %compare.low.base737 = trunc i64 %737 to i32, !dbg !24
  %compare.low.base738 = trunc i64 %701 to i32, !dbg !24
  %compare.low739 = icmp ugt i32 %compare.low.base737, %compare.low.base738, !dbg !24
  %compare.result740 = select i1 %compare.high.equal735, i1 %compare.low739, i1 %compare.high.order736, !dbg !24
  %740 = select i1 %compare.result740, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base741 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local742 = or i32 %compare.local.base741, 1, !dbg !24
  %compare.low743 = icmp ugt i32 %compare.low.base737, %compare.local742, !dbg !24
  %compare.result744 = select i1 %compare.high.equal735, i1 %compare.low743, i1 %compare.high.order736, !dbg !24
  %741 = select i1 %compare.result744, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base745 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local746 = or i32 %compare.local.base745, 2, !dbg !24
  %compare.low747 = icmp ugt i32 %compare.low.base737, %compare.local746, !dbg !24
  %compare.result748 = select i1 %compare.high.equal735, i1 %compare.low747, i1 %compare.high.order736, !dbg !24
  %742 = select i1 %compare.result748, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base749 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local750 = or i32 %compare.local.base749, 3, !dbg !24
  %compare.low751 = icmp ugt i32 %compare.low.base737, %compare.local750, !dbg !24
  %compare.result752 = select i1 %compare.high.equal735, i1 %compare.low751, i1 %compare.high.order736, !dbg !24
  %743 = select i1 %compare.result752, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base753 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local754 = or i32 %compare.local.base753, 4, !dbg !24
  %compare.low755 = icmp ugt i32 %compare.low.base737, %compare.local754, !dbg !24
  %compare.result756 = select i1 %compare.high.equal735, i1 %compare.low755, i1 %compare.high.order736, !dbg !24
  %744 = select i1 %compare.result756, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base757 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local758 = or i32 %compare.local.base757, 5, !dbg !24
  %compare.low759 = icmp ugt i32 %compare.low.base737, %compare.local758, !dbg !24
  %compare.result760 = select i1 %compare.high.equal735, i1 %compare.low759, i1 %compare.high.order736, !dbg !24
  %745 = select i1 %compare.result760, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base761 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local762 = or i32 %compare.local.base761, 6, !dbg !24
  %compare.low763 = icmp ugt i32 %compare.low.base737, %compare.local762, !dbg !24
  %compare.result764 = select i1 %compare.high.equal735, i1 %compare.low763, i1 %compare.high.order736, !dbg !24
  %746 = select i1 %compare.result764, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base765 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local766 = or i32 %compare.local.base765, 7, !dbg !24
  %compare.low767 = icmp ugt i32 %compare.low.base737, %compare.local766, !dbg !24
  %compare.result768 = select i1 %compare.high.equal735, i1 %compare.low767, i1 %compare.high.order736, !dbg !24
  %747 = select i1 %compare.result768, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base769 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local770 = or i32 %compare.local.base769, 8, !dbg !24
  %compare.low771 = icmp ugt i32 %compare.low.base737, %compare.local770, !dbg !24
  %compare.result772 = select i1 %compare.high.equal735, i1 %compare.low771, i1 %compare.high.order736, !dbg !24
  %748 = select i1 %compare.result772, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base773 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local774 = or i32 %compare.local.base773, 9, !dbg !24
  %compare.low775 = icmp ugt i32 %compare.low.base737, %compare.local774, !dbg !24
  %compare.result776 = select i1 %compare.high.equal735, i1 %compare.low775, i1 %compare.high.order736, !dbg !24
  %749 = select i1 %compare.result776, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base777 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local778 = or i32 %compare.local.base777, 10, !dbg !24
  %compare.low779 = icmp ugt i32 %compare.low.base737, %compare.local778, !dbg !24
  %compare.result780 = select i1 %compare.high.equal735, i1 %compare.low779, i1 %compare.high.order736, !dbg !24
  %750 = select i1 %compare.result780, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base781 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local782 = or i32 %compare.local.base781, 11, !dbg !24
  %compare.low783 = icmp ugt i32 %compare.low.base737, %compare.local782, !dbg !24
  %compare.result784 = select i1 %compare.high.equal735, i1 %compare.low783, i1 %compare.high.order736, !dbg !24
  %751 = select i1 %compare.result784, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base785 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local786 = or i32 %compare.local.base785, 12, !dbg !24
  %compare.low787 = icmp ugt i32 %compare.low.base737, %compare.local786, !dbg !24
  %compare.result788 = select i1 %compare.high.equal735, i1 %compare.low787, i1 %compare.high.order736, !dbg !24
  %752 = select i1 %compare.result788, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base789 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local790 = or i32 %compare.local.base789, 13, !dbg !24
  %compare.low791 = icmp ugt i32 %compare.low.base737, %compare.local790, !dbg !24
  %compare.result792 = select i1 %compare.high.equal735, i1 %compare.low791, i1 %compare.high.order736, !dbg !24
  %753 = select i1 %compare.result792, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base793 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local794 = or i32 %compare.local.base793, 14, !dbg !24
  %compare.low795 = icmp ugt i32 %compare.low.base737, %compare.local794, !dbg !24
  %compare.result796 = select i1 %compare.high.equal735, i1 %compare.low795, i1 %compare.high.order736, !dbg !24
  %754 = select i1 %compare.result796, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base797 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local798 = or i32 %compare.local.base797, 15, !dbg !24
  %compare.low799 = icmp ugt i32 %compare.low.base737, %compare.local798, !dbg !24
  %compare.result800 = select i1 %compare.high.equal735, i1 %compare.low799, i1 %compare.high.order736, !dbg !24
  %755 = select i1 %compare.result800, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base801 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local802 = or i32 %compare.local.base801, 64, !dbg !24
  %compare.low803 = icmp ugt i32 %compare.low.base737, %compare.local802, !dbg !24
  %compare.result804 = select i1 %compare.high.equal735, i1 %compare.low803, i1 %compare.high.order736, !dbg !24
  %756 = select i1 %compare.result804, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base805 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local806 = or i32 %compare.local.base805, 65, !dbg !24
  %compare.low807 = icmp ugt i32 %compare.low.base737, %compare.local806, !dbg !24
  %compare.result808 = select i1 %compare.high.equal735, i1 %compare.low807, i1 %compare.high.order736, !dbg !24
  %757 = select i1 %compare.result808, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base809 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local810 = or i32 %compare.local.base809, 66, !dbg !24
  %compare.low811 = icmp ugt i32 %compare.low.base737, %compare.local810, !dbg !24
  %compare.result812 = select i1 %compare.high.equal735, i1 %compare.low811, i1 %compare.high.order736, !dbg !24
  %758 = select i1 %compare.result812, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base813 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local814 = or i32 %compare.local.base813, 67, !dbg !24
  %compare.low815 = icmp ugt i32 %compare.low.base737, %compare.local814, !dbg !24
  %compare.result816 = select i1 %compare.high.equal735, i1 %compare.low815, i1 %compare.high.order736, !dbg !24
  %759 = select i1 %compare.result816, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base817 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local818 = or i32 %compare.local.base817, 68, !dbg !24
  %compare.low819 = icmp ugt i32 %compare.low.base737, %compare.local818, !dbg !24
  %compare.result820 = select i1 %compare.high.equal735, i1 %compare.low819, i1 %compare.high.order736, !dbg !24
  %760 = select i1 %compare.result820, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base821 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local822 = or i32 %compare.local.base821, 69, !dbg !24
  %compare.low823 = icmp ugt i32 %compare.low.base737, %compare.local822, !dbg !24
  %compare.result824 = select i1 %compare.high.equal735, i1 %compare.low823, i1 %compare.high.order736, !dbg !24
  %761 = select i1 %compare.result824, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base825 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local826 = or i32 %compare.local.base825, 70, !dbg !24
  %compare.low827 = icmp ugt i32 %compare.low.base737, %compare.local826, !dbg !24
  %compare.result828 = select i1 %compare.high.equal735, i1 %compare.low827, i1 %compare.high.order736, !dbg !24
  %762 = select i1 %compare.result828, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base829 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local830 = or i32 %compare.local.base829, 71, !dbg !24
  %compare.low831 = icmp ugt i32 %compare.low.base737, %compare.local830, !dbg !24
  %compare.result832 = select i1 %compare.high.equal735, i1 %compare.low831, i1 %compare.high.order736, !dbg !24
  %763 = select i1 %compare.result832, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base833 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local834 = or i32 %compare.local.base833, 72, !dbg !24
  %compare.low835 = icmp ugt i32 %compare.low.base737, %compare.local834, !dbg !24
  %compare.result836 = select i1 %compare.high.equal735, i1 %compare.low835, i1 %compare.high.order736, !dbg !24
  %764 = select i1 %compare.result836, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base837 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local838 = or i32 %compare.local.base837, 73, !dbg !24
  %compare.low839 = icmp ugt i32 %compare.low.base737, %compare.local838, !dbg !24
  %compare.result840 = select i1 %compare.high.equal735, i1 %compare.low839, i1 %compare.high.order736, !dbg !24
  %765 = select i1 %compare.result840, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base841 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local842 = or i32 %compare.local.base841, 74, !dbg !24
  %compare.low843 = icmp ugt i32 %compare.low.base737, %compare.local842, !dbg !24
  %compare.result844 = select i1 %compare.high.equal735, i1 %compare.low843, i1 %compare.high.order736, !dbg !24
  %766 = select i1 %compare.result844, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base845 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local846 = or i32 %compare.local.base845, 75, !dbg !24
  %compare.low847 = icmp ugt i32 %compare.low.base737, %compare.local846, !dbg !24
  %compare.result848 = select i1 %compare.high.equal735, i1 %compare.low847, i1 %compare.high.order736, !dbg !24
  %767 = select i1 %compare.result848, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base849 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local850 = or i32 %compare.local.base849, 76, !dbg !24
  %compare.low851 = icmp ugt i32 %compare.low.base737, %compare.local850, !dbg !24
  %compare.result852 = select i1 %compare.high.equal735, i1 %compare.low851, i1 %compare.high.order736, !dbg !24
  %768 = select i1 %compare.result852, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base853 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local854 = or i32 %compare.local.base853, 77, !dbg !24
  %compare.low855 = icmp ugt i32 %compare.low.base737, %compare.local854, !dbg !24
  %compare.result856 = select i1 %compare.high.equal735, i1 %compare.low855, i1 %compare.high.order736, !dbg !24
  %769 = select i1 %compare.result856, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base857 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local858 = or i32 %compare.local.base857, 78, !dbg !24
  %compare.low859 = icmp ugt i32 %compare.low.base737, %compare.local858, !dbg !24
  %compare.result860 = select i1 %compare.high.equal735, i1 %compare.low859, i1 %compare.high.order736, !dbg !24
  %770 = select i1 %compare.result860, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base861 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base738, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local862 = or i32 %compare.local.base861, 79, !dbg !24
  %compare.low863 = icmp ugt i32 %compare.low.base737, %compare.local862, !dbg !24
  %compare.result864 = select i1 %compare.high.equal735, i1 %compare.low863, i1 %compare.high.order736, !dbg !24
  %771 = select i1 %compare.result864, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %772 = insertelement <32 x float> poison, float %705, i64 0, !dbg !13
  %773 = insertelement <32 x float> %772, float %706, i64 1, !dbg !13
  %774 = insertelement <32 x float> %773, float %707, i64 2, !dbg !13
  %775 = insertelement <32 x float> %774, float %708, i64 3, !dbg !13
  %776 = insertelement <32 x float> %775, float %709, i64 4, !dbg !13
  %777 = insertelement <32 x float> %776, float %710, i64 5, !dbg !13
  %778 = insertelement <32 x float> %777, float %711, i64 6, !dbg !13
  %779 = insertelement <32 x float> %778, float %712, i64 7, !dbg !13
  %780 = insertelement <32 x float> %779, float %713, i64 8, !dbg !13
  %781 = insertelement <32 x float> %780, float %714, i64 9, !dbg !13
  %782 = insertelement <32 x float> %781, float %715, i64 10, !dbg !13
  %783 = insertelement <32 x float> %782, float %716, i64 11, !dbg !13
  %784 = insertelement <32 x float> %783, float %717, i64 12, !dbg !13
  %785 = insertelement <32 x float> %784, float %718, i64 13, !dbg !13
  %786 = insertelement <32 x float> %785, float %719, i64 14, !dbg !13
  %787 = insertelement <32 x float> %786, float %720, i64 15, !dbg !13
  %788 = insertelement <32 x float> %787, float %721, i64 16, !dbg !13
  %789 = insertelement <32 x float> %788, float %722, i64 17, !dbg !13
  %790 = insertelement <32 x float> %789, float %723, i64 18, !dbg !13
  %791 = insertelement <32 x float> %790, float %724, i64 19, !dbg !13
  %792 = insertelement <32 x float> %791, float %725, i64 20, !dbg !13
  %793 = insertelement <32 x float> %792, float %726, i64 21, !dbg !13
  %794 = insertelement <32 x float> %793, float %727, i64 22, !dbg !13
  %795 = insertelement <32 x float> %794, float %728, i64 23, !dbg !13
  %796 = insertelement <32 x float> %795, float %729, i64 24, !dbg !13
  %797 = insertelement <32 x float> %796, float %730, i64 25, !dbg !13
  %798 = insertelement <32 x float> %797, float %731, i64 26, !dbg !13
  %799 = insertelement <32 x float> %798, float %732, i64 27, !dbg !13
  %800 = insertelement <32 x float> %799, float %733, i64 28, !dbg !13
  %801 = insertelement <32 x float> %800, float %734, i64 29, !dbg !13
  %802 = insertelement <32 x float> %801, float %735, i64 30, !dbg !13
  %803 = insertelement <32 x float> %802, float %736, i64 31, !dbg !13
  %804 = insertvalue [2 x <32 x float>] poison, <32 x float> %803, 0, !dbg !13
  %805 = insertelement <32 x float> poison, float %740, i64 0, !dbg !13
  %806 = insertelement <32 x float> %805, float %741, i64 1, !dbg !13
  %807 = insertelement <32 x float> %806, float %742, i64 2, !dbg !13
  %808 = insertelement <32 x float> %807, float %743, i64 3, !dbg !13
  %809 = insertelement <32 x float> %808, float %744, i64 4, !dbg !13
  %810 = insertelement <32 x float> %809, float %745, i64 5, !dbg !13
  %811 = insertelement <32 x float> %810, float %746, i64 6, !dbg !13
  %812 = insertelement <32 x float> %811, float %747, i64 7, !dbg !13
  %813 = insertelement <32 x float> %812, float %748, i64 8, !dbg !13
  %814 = insertelement <32 x float> %813, float %749, i64 9, !dbg !13
  %815 = insertelement <32 x float> %814, float %750, i64 10, !dbg !13
  %816 = insertelement <32 x float> %815, float %751, i64 11, !dbg !13
  %817 = insertelement <32 x float> %816, float %752, i64 12, !dbg !13
  %818 = insertelement <32 x float> %817, float %753, i64 13, !dbg !13
  %819 = insertelement <32 x float> %818, float %754, i64 14, !dbg !13
  %820 = insertelement <32 x float> %819, float %755, i64 15, !dbg !13
  %821 = insertelement <32 x float> %820, float %756, i64 16, !dbg !13
  %822 = insertelement <32 x float> %821, float %757, i64 17, !dbg !13
  %823 = insertelement <32 x float> %822, float %758, i64 18, !dbg !13
  %824 = insertelement <32 x float> %823, float %759, i64 19, !dbg !13
  %825 = insertelement <32 x float> %824, float %760, i64 20, !dbg !13
  %826 = insertelement <32 x float> %825, float %761, i64 21, !dbg !13
  %827 = insertelement <32 x float> %826, float %762, i64 22, !dbg !13
  %828 = insertelement <32 x float> %827, float %763, i64 23, !dbg !13
  %829 = insertelement <32 x float> %828, float %764, i64 24, !dbg !13
  %830 = insertelement <32 x float> %829, float %765, i64 25, !dbg !13
  %831 = insertelement <32 x float> %830, float %766, i64 26, !dbg !13
  %832 = insertelement <32 x float> %831, float %767, i64 27, !dbg !13
  %833 = insertelement <32 x float> %832, float %768, i64 28, !dbg !13
  %834 = insertelement <32 x float> %833, float %769, i64 29, !dbg !13
  %835 = insertelement <32 x float> %834, float %770, i64 30, !dbg !13
  %836 = insertelement <32 x float> %835, float %771, i64 31, !dbg !13
  %837 = insertvalue [2 x <32 x float>] %804, <32 x float> %836, 1, !dbg !13
  br label %838, !dbg !13

838:                                              ; preds = %693, %694
  %839 = phi [2 x <32 x float>] [ %837, %694 ], [ zeroinitializer, %693 ], !dbg !13
  br label %840, !dbg !13

840:                                              ; preds = %838
  %841 = extractvalue [2 x <32 x float>] %839, 0, !dbg !26
  %842 = shufflevector <32 x float> %841, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %843 = fadd <4 x float> %630, %842, !dbg !26
  %844 = extractelement <4 x float> %843, i64 0, !dbg !26
  %845 = extractelement <4 x float> %843, i64 1, !dbg !26
  %846 = extractelement <4 x float> %843, i64 2, !dbg !26
  %847 = extractelement <4 x float> %843, i64 3, !dbg !26
  %848 = shufflevector <32 x float> %841, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %849 = fadd <4 x float> %634, %848, !dbg !26
  %850 = extractelement <4 x float> %849, i64 0, !dbg !26
  %851 = extractelement <4 x float> %849, i64 1, !dbg !26
  %852 = extractelement <4 x float> %849, i64 2, !dbg !26
  %853 = extractelement <4 x float> %849, i64 3, !dbg !26
  %854 = shufflevector <32 x float> %841, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %855 = fadd <4 x float> %638, %854, !dbg !26
  %856 = extractelement <4 x float> %855, i64 0, !dbg !26
  %857 = extractelement <4 x float> %855, i64 1, !dbg !26
  %858 = extractelement <4 x float> %855, i64 2, !dbg !26
  %859 = extractelement <4 x float> %855, i64 3, !dbg !26
  %860 = shufflevector <32 x float> %841, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %861 = fadd <4 x float> %642, %860, !dbg !26
  %862 = extractelement <4 x float> %861, i64 0, !dbg !26
  %863 = extractelement <4 x float> %861, i64 1, !dbg !26
  %864 = extractelement <4 x float> %861, i64 2, !dbg !26
  %865 = extractelement <4 x float> %861, i64 3, !dbg !26
  %866 = shufflevector <32 x float> %841, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %867 = fadd <4 x float> %646, %866, !dbg !26
  %868 = extractelement <4 x float> %867, i64 0, !dbg !26
  %869 = extractelement <4 x float> %867, i64 1, !dbg !26
  %870 = extractelement <4 x float> %867, i64 2, !dbg !26
  %871 = extractelement <4 x float> %867, i64 3, !dbg !26
  %872 = shufflevector <32 x float> %841, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %873 = fadd <4 x float> %650, %872, !dbg !26
  %874 = extractelement <4 x float> %873, i64 0, !dbg !26
  %875 = extractelement <4 x float> %873, i64 1, !dbg !26
  %876 = extractelement <4 x float> %873, i64 2, !dbg !26
  %877 = extractelement <4 x float> %873, i64 3, !dbg !26
  %878 = shufflevector <32 x float> %841, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %879 = fadd <4 x float> %654, %878, !dbg !26
  %880 = extractelement <4 x float> %879, i64 0, !dbg !26
  %881 = extractelement <4 x float> %879, i64 1, !dbg !26
  %882 = extractelement <4 x float> %879, i64 2, !dbg !26
  %883 = extractelement <4 x float> %879, i64 3, !dbg !26
  %884 = shufflevector <32 x float> %841, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %885 = fadd <4 x float> %658, %884, !dbg !26
  %886 = extractelement <4 x float> %885, i64 0, !dbg !26
  %887 = extractelement <4 x float> %885, i64 1, !dbg !26
  %888 = extractelement <4 x float> %885, i64 2, !dbg !26
  %889 = extractelement <4 x float> %885, i64 3, !dbg !26
  %890 = extractvalue [2 x <32 x float>] %839, 1, !dbg !26
  %891 = shufflevector <32 x float> %890, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %892 = fadd <4 x float> %662, %891, !dbg !26
  %893 = extractelement <4 x float> %892, i64 0, !dbg !26
  %894 = extractelement <4 x float> %892, i64 1, !dbg !26
  %895 = extractelement <4 x float> %892, i64 2, !dbg !26
  %896 = extractelement <4 x float> %892, i64 3, !dbg !26
  %897 = shufflevector <32 x float> %890, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %898 = fadd <4 x float> %666, %897, !dbg !26
  %899 = extractelement <4 x float> %898, i64 0, !dbg !26
  %900 = extractelement <4 x float> %898, i64 1, !dbg !26
  %901 = extractelement <4 x float> %898, i64 2, !dbg !26
  %902 = extractelement <4 x float> %898, i64 3, !dbg !26
  %903 = shufflevector <32 x float> %890, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %904 = fadd <4 x float> %670, %903, !dbg !26
  %905 = extractelement <4 x float> %904, i64 0, !dbg !26
  %906 = extractelement <4 x float> %904, i64 1, !dbg !26
  %907 = extractelement <4 x float> %904, i64 2, !dbg !26
  %908 = extractelement <4 x float> %904, i64 3, !dbg !26
  %909 = shufflevector <32 x float> %890, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %910 = fadd <4 x float> %674, %909, !dbg !26
  %911 = extractelement <4 x float> %910, i64 0, !dbg !26
  %912 = extractelement <4 x float> %910, i64 1, !dbg !26
  %913 = extractelement <4 x float> %910, i64 2, !dbg !26
  %914 = extractelement <4 x float> %910, i64 3, !dbg !26
  %915 = shufflevector <32 x float> %890, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %916 = fadd <4 x float> %678, %915, !dbg !26
  %917 = extractelement <4 x float> %916, i64 0, !dbg !26
  %918 = extractelement <4 x float> %916, i64 1, !dbg !26
  %919 = extractelement <4 x float> %916, i64 2, !dbg !26
  %920 = extractelement <4 x float> %916, i64 3, !dbg !26
  %921 = shufflevector <32 x float> %890, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %922 = fadd <4 x float> %682, %921, !dbg !26
  %923 = extractelement <4 x float> %922, i64 0, !dbg !26
  %924 = extractelement <4 x float> %922, i64 1, !dbg !26
  %925 = extractelement <4 x float> %922, i64 2, !dbg !26
  %926 = extractelement <4 x float> %922, i64 3, !dbg !26
  %927 = shufflevector <32 x float> %890, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %928 = fadd <4 x float> %686, %927, !dbg !26
  %929 = extractelement <4 x float> %928, i64 0, !dbg !26
  %930 = extractelement <4 x float> %928, i64 1, !dbg !26
  %931 = extractelement <4 x float> %928, i64 2, !dbg !26
  %932 = extractelement <4 x float> %928, i64 3, !dbg !26
  %933 = shufflevector <32 x float> %890, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %934 = fadd <4 x float> %691, %933, !dbg !26
  %935 = extractelement <4 x float> %934, i64 0, !dbg !26
  %936 = extractelement <4 x float> %934, i64 1, !dbg !26
  %937 = extractelement <4 x float> %934, i64 2, !dbg !26
  %938 = extractelement <4 x float> %934, i64 3, !dbg !26
  %939 = call float @llvm.maxnum.f32(float %844, float 0xFFF0000000000000), !dbg !27
  %940 = call float @llvm.maxnum.f32(float %939, float %845), !dbg !27
  %941 = call float @llvm.maxnum.f32(float %940, float %846), !dbg !27
  %942 = call float @llvm.maxnum.f32(float %941, float %847), !dbg !27
  %943 = call float @llvm.maxnum.f32(float %942, float %850), !dbg !27
  %944 = call float @llvm.maxnum.f32(float %943, float %851), !dbg !27
  %945 = call float @llvm.maxnum.f32(float %944, float %852), !dbg !27
  %946 = call float @llvm.maxnum.f32(float %945, float %853), !dbg !27
  %947 = call float @llvm.maxnum.f32(float %946, float %856), !dbg !27
  %948 = call float @llvm.maxnum.f32(float %947, float %857), !dbg !27
  %949 = call float @llvm.maxnum.f32(float %948, float %858), !dbg !27
  %950 = call float @llvm.maxnum.f32(float %949, float %859), !dbg !27
  %951 = call float @llvm.maxnum.f32(float %950, float %862), !dbg !27
  %952 = call float @llvm.maxnum.f32(float %951, float %863), !dbg !27
  %953 = call float @llvm.maxnum.f32(float %952, float %864), !dbg !27
  %954 = call float @llvm.maxnum.f32(float %953, float %865), !dbg !27
  %955 = call float @llvm.maxnum.f32(float %954, float %868), !dbg !27
  %956 = call float @llvm.maxnum.f32(float %955, float %869), !dbg !27
  %957 = call float @llvm.maxnum.f32(float %956, float %870), !dbg !27
  %958 = call float @llvm.maxnum.f32(float %957, float %871), !dbg !27
  %959 = call float @llvm.maxnum.f32(float %958, float %874), !dbg !27
  %960 = call float @llvm.maxnum.f32(float %959, float %875), !dbg !27
  %961 = call float @llvm.maxnum.f32(float %960, float %876), !dbg !27
  %962 = call float @llvm.maxnum.f32(float %961, float %877), !dbg !27
  %963 = call float @llvm.maxnum.f32(float %962, float %880), !dbg !27
  %964 = call float @llvm.maxnum.f32(float %963, float %881), !dbg !27
  %965 = call float @llvm.maxnum.f32(float %964, float %882), !dbg !27
  %966 = call float @llvm.maxnum.f32(float %965, float %883), !dbg !27
  %967 = call float @llvm.maxnum.f32(float %966, float %886), !dbg !27
  %968 = call float @llvm.maxnum.f32(float %967, float %887), !dbg !27
  %969 = call float @llvm.maxnum.f32(float %968, float %888), !dbg !27
  %970 = call float @llvm.maxnum.f32(float %969, float %889), !dbg !27
  %971 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %972 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %971), !dbg !27
  %973 = add nuw nsw i32 %972, 64, !dbg !27
  %974 = xor i32 %972, 16, !dbg !27
  %975 = and i32 %973, 192, !dbg !27
  %976 = icmp samesign ult i32 %974, %975, !dbg !27
  %977 = select i1 %976, i32 %974, i32 %972, !dbg !27
  %978 = shl nuw nsw i32 %977, 2, !dbg !27
  %979 = bitcast float %970 to i32, !dbg !27
  %980 = call i32 @llvm.amdgcn.ds.bpermute(i32 %978, i32 %979), !dbg !27
  %981 = bitcast i32 %980 to float, !dbg !27
  %982 = call float @llvm.maxnum.f32(float %970, float %981), !dbg !27
  %983 = xor i32 %972, 32, !dbg !27
  %984 = icmp samesign ult i32 %983, %975, !dbg !27
  %985 = select i1 %984, i32 %983, i32 %972, !dbg !27
  %986 = shl nuw nsw i32 %985, 2, !dbg !27
  %987 = bitcast float %982 to i32, !dbg !27
  %988 = call i32 @llvm.amdgcn.ds.bpermute(i32 %986, i32 %987), !dbg !27
  %989 = bitcast i32 %988 to float, !dbg !27
  %990 = call float @llvm.maxnum.f32(float %982, float %989), !dbg !27
  %991 = extractvalue [2 x <1 x float>] %56, 0, !dbg !27
  %992 = insertelement <1 x float> poison, float %990, i64 0, !dbg !27
  %993 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %992, <1 x float> %991), !dbg !27
  %994 = call float @llvm.maxnum.f32(float %893, float 0xFFF0000000000000), !dbg !27
  %995 = call float @llvm.maxnum.f32(float %994, float %894), !dbg !27
  %996 = call float @llvm.maxnum.f32(float %995, float %895), !dbg !27
  %997 = call float @llvm.maxnum.f32(float %996, float %896), !dbg !27
  %998 = call float @llvm.maxnum.f32(float %997, float %899), !dbg !27
  %999 = call float @llvm.maxnum.f32(float %998, float %900), !dbg !27
  %1000 = call float @llvm.maxnum.f32(float %999, float %901), !dbg !27
  %1001 = call float @llvm.maxnum.f32(float %1000, float %902), !dbg !27
  %1002 = call float @llvm.maxnum.f32(float %1001, float %905), !dbg !27
  %1003 = call float @llvm.maxnum.f32(float %1002, float %906), !dbg !27
  %1004 = call float @llvm.maxnum.f32(float %1003, float %907), !dbg !27
  %1005 = call float @llvm.maxnum.f32(float %1004, float %908), !dbg !27
  %1006 = call float @llvm.maxnum.f32(float %1005, float %911), !dbg !27
  %1007 = call float @llvm.maxnum.f32(float %1006, float %912), !dbg !27
  %1008 = call float @llvm.maxnum.f32(float %1007, float %913), !dbg !27
  %1009 = call float @llvm.maxnum.f32(float %1008, float %914), !dbg !27
  %1010 = call float @llvm.maxnum.f32(float %1009, float %917), !dbg !27
  %1011 = call float @llvm.maxnum.f32(float %1010, float %918), !dbg !27
  %1012 = call float @llvm.maxnum.f32(float %1011, float %919), !dbg !27
  %1013 = call float @llvm.maxnum.f32(float %1012, float %920), !dbg !27
  %1014 = call float @llvm.maxnum.f32(float %1013, float %923), !dbg !27
  %1015 = call float @llvm.maxnum.f32(float %1014, float %924), !dbg !27
  %1016 = call float @llvm.maxnum.f32(float %1015, float %925), !dbg !27
  %1017 = call float @llvm.maxnum.f32(float %1016, float %926), !dbg !27
  %1018 = call float @llvm.maxnum.f32(float %1017, float %929), !dbg !27
  %1019 = call float @llvm.maxnum.f32(float %1018, float %930), !dbg !27
  %1020 = call float @llvm.maxnum.f32(float %1019, float %931), !dbg !27
  %1021 = call float @llvm.maxnum.f32(float %1020, float %932), !dbg !27
  %1022 = call float @llvm.maxnum.f32(float %1021, float %935), !dbg !27
  %1023 = call float @llvm.maxnum.f32(float %1022, float %936), !dbg !27
  %1024 = call float @llvm.maxnum.f32(float %1023, float %937), !dbg !27
  %1025 = call float @llvm.maxnum.f32(float %1024, float %938), !dbg !27
  %1026 = bitcast float %1025 to i32, !dbg !27
  %1027 = call i32 @llvm.amdgcn.ds.bpermute(i32 %978, i32 %1026), !dbg !27
  %1028 = bitcast i32 %1027 to float, !dbg !27
  %1029 = call float @llvm.maxnum.f32(float %1025, float %1028), !dbg !27
  %1030 = bitcast float %1029 to i32, !dbg !27
  %1031 = call i32 @llvm.amdgcn.ds.bpermute(i32 %986, i32 %1030), !dbg !27
  %1032 = bitcast i32 %1031 to float, !dbg !27
  %1033 = call float @llvm.maxnum.f32(float %1029, float %1032), !dbg !27
  %1034 = extractvalue [2 x <1 x float>] %56, 1, !dbg !27
  %1035 = insertelement <1 x float> poison, float %1033, i64 0, !dbg !27
  %1036 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1035, <1 x float> %1034), !dbg !27
  %1037 = shufflevector <1 x float> %993, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1038 = fsub <4 x float> %843, %1037, !dbg !28
  %1039 = fsub <4 x float> %849, %1037, !dbg !28
  %1040 = fsub <4 x float> %855, %1037, !dbg !28
  %1041 = fsub <4 x float> %861, %1037, !dbg !28
  %1042 = fsub <4 x float> %867, %1037, !dbg !28
  %1043 = fsub <4 x float> %873, %1037, !dbg !28
  %1044 = fsub <4 x float> %879, %1037, !dbg !28
  %1045 = fsub <4 x float> %885, %1037, !dbg !28
  %1046 = shufflevector <1 x float> %1036, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1047 = fsub <4 x float> %892, %1046, !dbg !28
  %1048 = fsub <4 x float> %898, %1046, !dbg !28
  %1049 = fsub <4 x float> %904, %1046, !dbg !28
  %1050 = fsub <4 x float> %910, %1046, !dbg !28
  %1051 = fsub <4 x float> %916, %1046, !dbg !28
  %1052 = fsub <4 x float> %922, %1046, !dbg !28
  %1053 = fsub <4 x float> %928, %1046, !dbg !28
  %1054 = fsub <4 x float> %934, %1046, !dbg !28
  %1055 = fsub <1 x float> %991, %993, !dbg !29
  %1056 = extractelement <1 x float> %1055, i64 0, !dbg !30
  %1057 = call float @llvm.amdgcn.exp2.f32(float %1056), !dbg !30
  %1058 = insertelement <4 x float> poison, float %1057, i64 0, !dbg !35
  %1059 = shufflevector <4 x float> %1058, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %1060 = extractvalue [2 x <1 x float>] %55, 0, !dbg !31
  %1061 = extractelement <1 x float> %1060, i64 0, !dbg !31
  %1062 = fmul float %1061, %1057, !dbg !31
  %1063 = fsub <1 x float> %1034, %1036, !dbg !29
  %1064 = extractelement <1 x float> %1063, i64 0, !dbg !30
  %1065 = call float @llvm.amdgcn.exp2.f32(float %1064), !dbg !30
  %1066 = insertelement <4 x float> poison, float %1065, i64 0, !dbg !35
  %1067 = shufflevector <4 x float> %1066, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %1068 = extractvalue [2 x <1 x float>] %55, 1, !dbg !31
  %1069 = extractelement <1 x float> %1068, i64 0, !dbg !31
  %1070 = fmul float %1069, %1065, !dbg !31
  %1071 = extractelement <4 x float> %1038, i64 0, !dbg !32
  %1072 = call float @llvm.amdgcn.exp2.f32(float %1071), !dbg !32
  %1073 = insertelement <4 x float> poison, float %1072, i64 0, !dbg !34
  %1074 = extractelement <4 x float> %1038, i64 1, !dbg !32
  %1075 = call float @llvm.amdgcn.exp2.f32(float %1074), !dbg !32
  %1076 = insertelement <4 x float> %1073, float %1075, i64 1, !dbg !34
  %1077 = fadd float %1072, %1075, !dbg !33
  %1078 = extractelement <4 x float> %1038, i64 2, !dbg !32
  %1079 = call float @llvm.amdgcn.exp2.f32(float %1078), !dbg !32
  %1080 = insertelement <4 x float> %1076, float %1079, i64 2, !dbg !34
  %1081 = fadd float %1077, %1079, !dbg !33
  %1082 = extractelement <4 x float> %1038, i64 3, !dbg !32
  %1083 = call float @llvm.amdgcn.exp2.f32(float %1082), !dbg !32
  %1084 = insertelement <4 x float> %1080, float %1083, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1084) #9, !dbg !34, !frisk.register_lifetimes !1
  %1085 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1085) #9, !dbg !33, !frisk.register_lifetimes !1
  %1086 = fadd float %1081, %1083, !dbg !33
  %1087 = extractelement <4 x float> %1039, i64 0, !dbg !32
  %1088 = call float @llvm.amdgcn.exp2.f32(float %1087), !dbg !32
  %1089 = insertelement <4 x float> poison, float %1088, i64 0, !dbg !34
  %1090 = fadd float %1086, %1088, !dbg !33
  %1091 = extractelement <4 x float> %1039, i64 1, !dbg !32
  %1092 = call float @llvm.amdgcn.exp2.f32(float %1091), !dbg !32
  %1093 = insertelement <4 x float> %1089, float %1092, i64 1, !dbg !34
  %1094 = fadd float %1090, %1092, !dbg !33
  %1095 = extractelement <4 x float> %1039, i64 2, !dbg !32
  %1096 = call float @llvm.amdgcn.exp2.f32(float %1095), !dbg !32
  %1097 = insertelement <4 x float> %1093, float %1096, i64 2, !dbg !34
  %1098 = fadd float %1094, %1096, !dbg !33
  %1099 = extractelement <4 x float> %1039, i64 3, !dbg !32
  %1100 = call float @llvm.amdgcn.exp2.f32(float %1099), !dbg !32
  %1101 = insertelement <4 x float> %1097, float %1100, i64 3, !dbg !34
  %math.result96 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1101) #9, !dbg !34, !frisk.register_lifetimes !1
  %1102 = fptrunc <4 x float> %math.result96 to <4 x half>, !dbg !34
  %math.packed97 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1102) #9, !dbg !33, !frisk.register_lifetimes !1
  %1103 = fadd float %1098, %1100, !dbg !33
  %1104 = extractelement <4 x float> %1040, i64 0, !dbg !32
  %1105 = call float @llvm.amdgcn.exp2.f32(float %1104), !dbg !32
  %1106 = insertelement <4 x float> poison, float %1105, i64 0, !dbg !34
  %1107 = fadd float %1103, %1105, !dbg !33
  %1108 = extractelement <4 x float> %1040, i64 1, !dbg !32
  %1109 = call float @llvm.amdgcn.exp2.f32(float %1108), !dbg !32
  %1110 = insertelement <4 x float> %1106, float %1109, i64 1, !dbg !34
  %1111 = fadd float %1107, %1109, !dbg !33
  %1112 = extractelement <4 x float> %1040, i64 2, !dbg !32
  %1113 = call float @llvm.amdgcn.exp2.f32(float %1112), !dbg !32
  %1114 = insertelement <4 x float> %1110, float %1113, i64 2, !dbg !34
  %1115 = fadd float %1111, %1113, !dbg !33
  %1116 = extractelement <4 x float> %1040, i64 3, !dbg !32
  %1117 = call float @llvm.amdgcn.exp2.f32(float %1116), !dbg !32
  %1118 = insertelement <4 x float> %1114, float %1117, i64 3, !dbg !34
  %math.result98 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1118) #9, !dbg !34, !frisk.register_lifetimes !1
  %1119 = fptrunc <4 x float> %math.result98 to <4 x half>, !dbg !34
  %math.packed99 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1119) #9, !dbg !33, !frisk.register_lifetimes !1
  %1120 = fadd float %1115, %1117, !dbg !33
  %1121 = extractelement <4 x float> %1041, i64 0, !dbg !32
  %1122 = call float @llvm.amdgcn.exp2.f32(float %1121), !dbg !32
  %1123 = insertelement <4 x float> poison, float %1122, i64 0, !dbg !34
  %1124 = fadd float %1120, %1122, !dbg !33
  %1125 = extractelement <4 x float> %1041, i64 1, !dbg !32
  %1126 = call float @llvm.amdgcn.exp2.f32(float %1125), !dbg !32
  %1127 = insertelement <4 x float> %1123, float %1126, i64 1, !dbg !34
  %1128 = fadd float %1124, %1126, !dbg !33
  %1129 = extractelement <4 x float> %1041, i64 2, !dbg !32
  %1130 = call float @llvm.amdgcn.exp2.f32(float %1129), !dbg !32
  %1131 = insertelement <4 x float> %1127, float %1130, i64 2, !dbg !34
  %1132 = fadd float %1128, %1130, !dbg !33
  %1133 = extractelement <4 x float> %1041, i64 3, !dbg !32
  %1134 = call float @llvm.amdgcn.exp2.f32(float %1133), !dbg !32
  %1135 = insertelement <4 x float> %1131, float %1134, i64 3, !dbg !34
  %math.result100 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1135) #9, !dbg !34, !frisk.register_lifetimes !1
  %1136 = fptrunc <4 x float> %math.result100 to <4 x half>, !dbg !34
  %math.packed101 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1136) #9, !dbg !33, !frisk.register_lifetimes !1
  %1137 = fadd float %1132, %1134, !dbg !33
  %1138 = extractelement <4 x float> %1042, i64 0, !dbg !32
  %1139 = call float @llvm.amdgcn.exp2.f32(float %1138), !dbg !32
  %1140 = insertelement <4 x float> poison, float %1139, i64 0, !dbg !34
  %1141 = fadd float %1137, %1139, !dbg !33
  %1142 = extractelement <4 x float> %1042, i64 1, !dbg !32
  %1143 = call float @llvm.amdgcn.exp2.f32(float %1142), !dbg !32
  %1144 = insertelement <4 x float> %1140, float %1143, i64 1, !dbg !34
  %1145 = fadd float %1141, %1143, !dbg !33
  %1146 = extractelement <4 x float> %1042, i64 2, !dbg !32
  %1147 = call float @llvm.amdgcn.exp2.f32(float %1146), !dbg !32
  %1148 = insertelement <4 x float> %1144, float %1147, i64 2, !dbg !34
  %1149 = fadd float %1145, %1147, !dbg !33
  %1150 = extractelement <4 x float> %1042, i64 3, !dbg !32
  %1151 = call float @llvm.amdgcn.exp2.f32(float %1150), !dbg !32
  %1152 = insertelement <4 x float> %1148, float %1151, i64 3, !dbg !34
  %math.result102 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1152) #9, !dbg !34, !frisk.register_lifetimes !1
  %1153 = fptrunc <4 x float> %math.result102 to <4 x half>, !dbg !34
  %math.packed103 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1153) #9, !dbg !33, !frisk.register_lifetimes !1
  %1154 = fadd float %1149, %1151, !dbg !33
  %1155 = extractelement <4 x float> %1043, i64 0, !dbg !32
  %1156 = call float @llvm.amdgcn.exp2.f32(float %1155), !dbg !32
  %1157 = insertelement <4 x float> poison, float %1156, i64 0, !dbg !34
  %1158 = fadd float %1154, %1156, !dbg !33
  %1159 = extractelement <4 x float> %1043, i64 1, !dbg !32
  %1160 = call float @llvm.amdgcn.exp2.f32(float %1159), !dbg !32
  %1161 = insertelement <4 x float> %1157, float %1160, i64 1, !dbg !34
  %1162 = fadd float %1158, %1160, !dbg !33
  %1163 = extractelement <4 x float> %1043, i64 2, !dbg !32
  %1164 = call float @llvm.amdgcn.exp2.f32(float %1163), !dbg !32
  %1165 = insertelement <4 x float> %1161, float %1164, i64 2, !dbg !34
  %1166 = fadd float %1162, %1164, !dbg !33
  %1167 = extractelement <4 x float> %1043, i64 3, !dbg !32
  %1168 = call float @llvm.amdgcn.exp2.f32(float %1167), !dbg !32
  %1169 = insertelement <4 x float> %1165, float %1168, i64 3, !dbg !34
  %math.result104 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1169) #9, !dbg !34, !frisk.register_lifetimes !1
  %1170 = fptrunc <4 x float> %math.result104 to <4 x half>, !dbg !34
  %math.packed105 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1170) #9, !dbg !33, !frisk.register_lifetimes !1
  %1171 = fadd float %1166, %1168, !dbg !33
  %1172 = extractelement <4 x float> %1044, i64 0, !dbg !32
  %1173 = call float @llvm.amdgcn.exp2.f32(float %1172), !dbg !32
  %1174 = insertelement <4 x float> poison, float %1173, i64 0, !dbg !34
  %1175 = fadd float %1171, %1173, !dbg !33
  %1176 = extractelement <4 x float> %1044, i64 1, !dbg !32
  %1177 = call float @llvm.amdgcn.exp2.f32(float %1176), !dbg !32
  %1178 = insertelement <4 x float> %1174, float %1177, i64 1, !dbg !34
  %1179 = fadd float %1175, %1177, !dbg !33
  %1180 = extractelement <4 x float> %1044, i64 2, !dbg !32
  %1181 = call float @llvm.amdgcn.exp2.f32(float %1180), !dbg !32
  %1182 = insertelement <4 x float> %1178, float %1181, i64 2, !dbg !34
  %1183 = fadd float %1179, %1181, !dbg !33
  %1184 = extractelement <4 x float> %1044, i64 3, !dbg !32
  %1185 = call float @llvm.amdgcn.exp2.f32(float %1184), !dbg !32
  %1186 = insertelement <4 x float> %1182, float %1185, i64 3, !dbg !34
  %math.result106 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1186) #9, !dbg !34, !frisk.register_lifetimes !1
  %1187 = fptrunc <4 x float> %math.result106 to <4 x half>, !dbg !34
  %math.packed107 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1187) #9, !dbg !33, !frisk.register_lifetimes !1
  %1188 = fadd float %1183, %1185, !dbg !33
  %1189 = extractelement <4 x float> %1045, i64 0, !dbg !32
  %1190 = call float @llvm.amdgcn.exp2.f32(float %1189), !dbg !32
  %1191 = insertelement <4 x float> poison, float %1190, i64 0, !dbg !34
  %1192 = fadd float %1188, %1190, !dbg !33
  %1193 = extractelement <4 x float> %1045, i64 1, !dbg !32
  %1194 = call float @llvm.amdgcn.exp2.f32(float %1193), !dbg !32
  %1195 = insertelement <4 x float> %1191, float %1194, i64 1, !dbg !34
  %1196 = fadd float %1192, %1194, !dbg !33
  %1197 = extractelement <4 x float> %1045, i64 2, !dbg !32
  %1198 = call float @llvm.amdgcn.exp2.f32(float %1197), !dbg !32
  %1199 = insertelement <4 x float> %1195, float %1198, i64 2, !dbg !34
  %1200 = fadd float %1196, %1198, !dbg !33
  %1201 = extractelement <4 x float> %1045, i64 3, !dbg !32
  %1202 = call float @llvm.amdgcn.exp2.f32(float %1201), !dbg !32
  %1203 = insertelement <4 x float> %1199, float %1202, i64 3, !dbg !34
  %math.result108 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1203) #9, !dbg !34, !frisk.register_lifetimes !1
  %1204 = fptrunc <4 x float> %math.result108 to <4 x half>, !dbg !34
  %math.packed109 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1204) #9, !dbg !33, !frisk.register_lifetimes !1
  %1205 = fadd float %1200, %1202, !dbg !33
  %1206 = extractelement <4 x float> %1047, i64 0, !dbg !32
  %1207 = call float @llvm.amdgcn.exp2.f32(float %1206), !dbg !32
  %1208 = insertelement <4 x float> poison, float %1207, i64 0, !dbg !34
  %1209 = extractelement <4 x float> %1047, i64 1, !dbg !32
  %1210 = call float @llvm.amdgcn.exp2.f32(float %1209), !dbg !32
  %1211 = insertelement <4 x float> %1208, float %1210, i64 1, !dbg !34
  %1212 = fadd float %1207, %1210, !dbg !33
  %1213 = extractelement <4 x float> %1047, i64 2, !dbg !32
  %1214 = call float @llvm.amdgcn.exp2.f32(float %1213), !dbg !32
  %1215 = insertelement <4 x float> %1211, float %1214, i64 2, !dbg !34
  %1216 = fadd float %1212, %1214, !dbg !33
  %1217 = extractelement <4 x float> %1047, i64 3, !dbg !32
  %1218 = call float @llvm.amdgcn.exp2.f32(float %1217), !dbg !32
  %1219 = insertelement <4 x float> %1215, float %1218, i64 3, !dbg !34
  %math.result110 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1219) #9, !dbg !34, !frisk.register_lifetimes !1
  %1220 = fptrunc <4 x float> %math.result110 to <4 x half>, !dbg !34
  %math.packed111 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1220) #9, !dbg !33, !frisk.register_lifetimes !1
  %1221 = fadd float %1216, %1218, !dbg !33
  %1222 = extractelement <4 x float> %1048, i64 0, !dbg !32
  %1223 = call float @llvm.amdgcn.exp2.f32(float %1222), !dbg !32
  %1224 = insertelement <4 x float> poison, float %1223, i64 0, !dbg !34
  %1225 = fadd float %1221, %1223, !dbg !33
  %1226 = extractelement <4 x float> %1048, i64 1, !dbg !32
  %1227 = call float @llvm.amdgcn.exp2.f32(float %1226), !dbg !32
  %1228 = insertelement <4 x float> %1224, float %1227, i64 1, !dbg !34
  %1229 = fadd float %1225, %1227, !dbg !33
  %1230 = extractelement <4 x float> %1048, i64 2, !dbg !32
  %1231 = call float @llvm.amdgcn.exp2.f32(float %1230), !dbg !32
  %1232 = insertelement <4 x float> %1228, float %1231, i64 2, !dbg !34
  %1233 = fadd float %1229, %1231, !dbg !33
  %1234 = extractelement <4 x float> %1048, i64 3, !dbg !32
  %1235 = call float @llvm.amdgcn.exp2.f32(float %1234), !dbg !32
  %1236 = insertelement <4 x float> %1232, float %1235, i64 3, !dbg !34
  %math.result112 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1236) #9, !dbg !34, !frisk.register_lifetimes !1
  %1237 = fptrunc <4 x float> %math.result112 to <4 x half>, !dbg !34
  %math.packed113 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1237) #9, !dbg !33, !frisk.register_lifetimes !1
  %1238 = fadd float %1233, %1235, !dbg !33
  %1239 = extractelement <4 x float> %1049, i64 0, !dbg !32
  %1240 = call float @llvm.amdgcn.exp2.f32(float %1239), !dbg !32
  %1241 = insertelement <4 x float> poison, float %1240, i64 0, !dbg !34
  %1242 = fadd float %1238, %1240, !dbg !33
  %1243 = extractelement <4 x float> %1049, i64 1, !dbg !32
  %1244 = call float @llvm.amdgcn.exp2.f32(float %1243), !dbg !32
  %1245 = insertelement <4 x float> %1241, float %1244, i64 1, !dbg !34
  %1246 = fadd float %1242, %1244, !dbg !33
  %1247 = extractelement <4 x float> %1049, i64 2, !dbg !32
  %1248 = call float @llvm.amdgcn.exp2.f32(float %1247), !dbg !32
  %1249 = insertelement <4 x float> %1245, float %1248, i64 2, !dbg !34
  %1250 = fadd float %1246, %1248, !dbg !33
  %1251 = extractelement <4 x float> %1049, i64 3, !dbg !32
  %1252 = call float @llvm.amdgcn.exp2.f32(float %1251), !dbg !32
  %1253 = insertelement <4 x float> %1249, float %1252, i64 3, !dbg !34
  %math.result114 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1253) #9, !dbg !34, !frisk.register_lifetimes !1
  %1254 = fptrunc <4 x float> %math.result114 to <4 x half>, !dbg !34
  %math.packed115 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1254) #9, !dbg !33, !frisk.register_lifetimes !1
  %1255 = fadd float %1250, %1252, !dbg !33
  %1256 = extractelement <4 x float> %1050, i64 0, !dbg !32
  %1257 = call float @llvm.amdgcn.exp2.f32(float %1256), !dbg !32
  %1258 = insertelement <4 x float> poison, float %1257, i64 0, !dbg !34
  %1259 = fadd float %1255, %1257, !dbg !33
  %1260 = extractelement <4 x float> %1050, i64 1, !dbg !32
  %1261 = call float @llvm.amdgcn.exp2.f32(float %1260), !dbg !32
  %1262 = insertelement <4 x float> %1258, float %1261, i64 1, !dbg !34
  %1263 = fadd float %1259, %1261, !dbg !33
  %1264 = extractelement <4 x float> %1050, i64 2, !dbg !32
  %1265 = call float @llvm.amdgcn.exp2.f32(float %1264), !dbg !32
  %1266 = insertelement <4 x float> %1262, float %1265, i64 2, !dbg !34
  %1267 = fadd float %1263, %1265, !dbg !33
  %1268 = extractelement <4 x float> %1050, i64 3, !dbg !32
  %1269 = call float @llvm.amdgcn.exp2.f32(float %1268), !dbg !32
  %1270 = insertelement <4 x float> %1266, float %1269, i64 3, !dbg !34
  %math.result116 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1270) #9, !dbg !34, !frisk.register_lifetimes !1
  %1271 = fptrunc <4 x float> %math.result116 to <4 x half>, !dbg !34
  %math.packed117 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1271) #9, !dbg !33, !frisk.register_lifetimes !1
  %1272 = fadd float %1267, %1269, !dbg !33
  %1273 = extractelement <4 x float> %1051, i64 0, !dbg !32
  %1274 = call float @llvm.amdgcn.exp2.f32(float %1273), !dbg !32
  %1275 = insertelement <4 x float> poison, float %1274, i64 0, !dbg !34
  %1276 = fadd float %1272, %1274, !dbg !33
  %1277 = extractelement <4 x float> %1051, i64 1, !dbg !32
  %1278 = call float @llvm.amdgcn.exp2.f32(float %1277), !dbg !32
  %1279 = insertelement <4 x float> %1275, float %1278, i64 1, !dbg !34
  %1280 = fadd float %1276, %1278, !dbg !33
  %1281 = extractelement <4 x float> %1051, i64 2, !dbg !32
  %1282 = call float @llvm.amdgcn.exp2.f32(float %1281), !dbg !32
  %1283 = insertelement <4 x float> %1279, float %1282, i64 2, !dbg !34
  %1284 = fadd float %1280, %1282, !dbg !33
  %1285 = extractelement <4 x float> %1051, i64 3, !dbg !32
  %1286 = call float @llvm.amdgcn.exp2.f32(float %1285), !dbg !32
  %1287 = insertelement <4 x float> %1283, float %1286, i64 3, !dbg !34
  %math.result118 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1287) #9, !dbg !34, !frisk.register_lifetimes !1
  %1288 = fptrunc <4 x float> %math.result118 to <4 x half>, !dbg !34
  %math.packed119 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1288) #9, !dbg !33, !frisk.register_lifetimes !1
  %1289 = fadd float %1284, %1286, !dbg !33
  %1290 = extractelement <4 x float> %1052, i64 0, !dbg !32
  %1291 = call float @llvm.amdgcn.exp2.f32(float %1290), !dbg !32
  %1292 = insertelement <4 x float> poison, float %1291, i64 0, !dbg !34
  %1293 = fadd float %1289, %1291, !dbg !33
  %1294 = extractelement <4 x float> %1052, i64 1, !dbg !32
  %1295 = call float @llvm.amdgcn.exp2.f32(float %1294), !dbg !32
  %1296 = insertelement <4 x float> %1292, float %1295, i64 1, !dbg !34
  %1297 = fadd float %1293, %1295, !dbg !33
  %1298 = extractelement <4 x float> %1052, i64 2, !dbg !32
  %1299 = call float @llvm.amdgcn.exp2.f32(float %1298), !dbg !32
  %1300 = insertelement <4 x float> %1296, float %1299, i64 2, !dbg !34
  %1301 = fadd float %1297, %1299, !dbg !33
  %1302 = extractelement <4 x float> %1052, i64 3, !dbg !32
  %1303 = call float @llvm.amdgcn.exp2.f32(float %1302), !dbg !32
  %1304 = insertelement <4 x float> %1300, float %1303, i64 3, !dbg !34
  %math.result120 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1304) #9, !dbg !34, !frisk.register_lifetimes !1
  %1305 = fptrunc <4 x float> %math.result120 to <4 x half>, !dbg !34
  %math.packed121 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1305) #9, !dbg !33, !frisk.register_lifetimes !1
  %1306 = fadd float %1301, %1303, !dbg !33
  %1307 = extractelement <4 x float> %1053, i64 0, !dbg !32
  %1308 = call float @llvm.amdgcn.exp2.f32(float %1307), !dbg !32
  %1309 = insertelement <4 x float> poison, float %1308, i64 0, !dbg !34
  %1310 = fadd float %1306, %1308, !dbg !33
  %1311 = extractelement <4 x float> %1053, i64 1, !dbg !32
  %1312 = call float @llvm.amdgcn.exp2.f32(float %1311), !dbg !32
  %1313 = insertelement <4 x float> %1309, float %1312, i64 1, !dbg !34
  %1314 = fadd float %1310, %1312, !dbg !33
  %1315 = extractelement <4 x float> %1053, i64 2, !dbg !32
  %1316 = call float @llvm.amdgcn.exp2.f32(float %1315), !dbg !32
  %1317 = insertelement <4 x float> %1313, float %1316, i64 2, !dbg !34
  %1318 = fadd float %1314, %1316, !dbg !33
  %1319 = extractelement <4 x float> %1053, i64 3, !dbg !32
  %1320 = call float @llvm.amdgcn.exp2.f32(float %1319), !dbg !32
  %1321 = insertelement <4 x float> %1317, float %1320, i64 3, !dbg !34
  %math.result122 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1321) #9, !dbg !34, !frisk.register_lifetimes !1
  %1322 = fptrunc <4 x float> %math.result122 to <4 x half>, !dbg !34
  %math.packed123 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1322) #9, !dbg !33, !frisk.register_lifetimes !1
  %1323 = fadd float %1318, %1320, !dbg !33
  %1324 = extractelement <4 x float> %1054, i64 0, !dbg !32
  %1325 = call float @llvm.amdgcn.exp2.f32(float %1324), !dbg !32
  %1326 = insertelement <4 x float> poison, float %1325, i64 0, !dbg !34
  %1327 = fadd float %1323, %1325, !dbg !33
  %1328 = extractelement <4 x float> %1054, i64 1, !dbg !32
  %1329 = call float @llvm.amdgcn.exp2.f32(float %1328), !dbg !32
  %1330 = insertelement <4 x float> %1326, float %1329, i64 1, !dbg !34
  %1331 = fadd float %1327, %1329, !dbg !33
  %1332 = extractelement <4 x float> %1054, i64 2, !dbg !32
  %1333 = call float @llvm.amdgcn.exp2.f32(float %1332), !dbg !32
  %1334 = insertelement <4 x float> %1330, float %1333, i64 2, !dbg !34
  %1335 = fadd float %1331, %1333, !dbg !33
  %1336 = extractelement <4 x float> %1054, i64 3, !dbg !32
  %1337 = call float @llvm.amdgcn.exp2.f32(float %1336), !dbg !32
  %1338 = insertelement <4 x float> %1334, float %1337, i64 3, !dbg !34
  %math.result124 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1338) #9, !dbg !34, !frisk.register_lifetimes !1
  %1339 = fptrunc <4 x float> %math.result124 to <4 x half>, !dbg !34
  %math.packed125 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1339) #9, !dbg !33, !frisk.register_lifetimes !1
  %1340 = fadd float %1335, %1337, !dbg !33
  %1341 = bitcast float %1205 to i32, !dbg !33
  %1342 = call i32 @llvm.amdgcn.ds.bpermute(i32 %978, i32 %1341), !dbg !33
  %1343 = bitcast i32 %1342 to float, !dbg !33
  %1344 = fadd float %1205, %1343, !dbg !33
  %1345 = bitcast float %1344 to i32, !dbg !33
  %1346 = call i32 @llvm.amdgcn.ds.bpermute(i32 %986, i32 %1345), !dbg !33
  %1347 = bitcast i32 %1346 to float, !dbg !33
  %1348 = fadd float %1344, %1347, !dbg !33
  %1349 = fadd float %1348, %1062, !dbg !33
  %1350 = insertelement <1 x float> poison, float %1349, i64 0, !dbg !7
  %1351 = insertvalue [2 x <1 x float>] poison, <1 x float> %1350, 0, !dbg !7
  %1352 = bitcast float %1340 to i32, !dbg !33
  %1353 = call i32 @llvm.amdgcn.ds.bpermute(i32 %978, i32 %1352), !dbg !33
  %1354 = bitcast i32 %1353 to float, !dbg !33
  %1355 = fadd float %1340, %1354, !dbg !33
  %1356 = bitcast float %1355 to i32, !dbg !33
  %1357 = call i32 @llvm.amdgcn.ds.bpermute(i32 %986, i32 %1356), !dbg !33
  %1358 = bitcast i32 %1357 to float, !dbg !33
  %1359 = fadd float %1355, %1358, !dbg !33
  %1360 = fadd float %1359, %1070, !dbg !33
  %1361 = insertelement <1 x float> poison, float %1360, i64 0, !dbg !7
  %1362 = insertvalue [2 x <1 x float>] %1351, <1 x float> %1361, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread316 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !14, !range !39, !frisk.register_lifetimes !1
  %address.local317 = shl nuw nsw i32 %address.thread316, 3, !dbg !14
  %address.local318 = and i32 %address.local317, 120, !dbg !14
  %address.local319 = shl nuw nsw i32 %address.thread316, 5, !dbg !14
  %address.local320 = and i32 %address.local319, 1536, !dbg !14
  %address.local321 = or disjoint i32 %address.local318, %address.local320, !dbg !14
  %address.local322 = zext nneg i32 %address.local321 to i64, !dbg !14
  %address.local323 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local322, !dbg !14
  %1363 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local323, i32 0) #0, !dbg !14
  %1364 = bitcast <4 x i32> %1363 to <8 x half>, !dbg !14
  %address.local324 = zext nneg i32 %address.thread316 to i64, !dbg !12
  %address.local325 = and i64 %address.local324, 60, !dbg !14
  %address.local326 = add nuw nsw i64 %address.local325, %address.local324, !dbg !14
  %address.local327 = shl nuw nsw i64 %address.local326, 3, !dbg !14
  %address.local328 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local327, !dbg !14
  %address.local329 = and i64 %address.local324, 63, !dbg !12
  %address.local330 = lshr i64 %address.local329, 4, !dbg !14
  %address.local331 = mul nuw nsw i64 %address.local330, 384, !dbg !14
  %address.local332 = getelementptr [2 x i8], ptr addrspace(3) %address.local328, i64 %address.local331, !dbg !14
  %address.local333 = lshr i64 %address.local324, 2, !dbg !17
  %address.local334 = mul nsw i64 %address.local333, -32, !dbg !14
  %address.local335 = getelementptr [2 x i8], ptr addrspace(3) %address.local332, i64 %address.local334, !dbg !14
  %address.local336 = getelementptr i8, ptr addrspace(3) %address.local335, i64 256, !dbg !14
  %1365 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local336, i32 0) #0, !dbg !14
  %1366 = bitcast <4 x i32> %1365 to <8 x half>, !dbg !14
  %address.local337 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local327, !dbg !14
  %address.local338 = getelementptr [2 x i8], ptr addrspace(3) %address.local337, i64 %address.local331, !dbg !14
  %address.local339 = getelementptr [2 x i8], ptr addrspace(3) %address.local338, i64 %address.local334, !dbg !14
  %address.local340 = getelementptr i8, ptr addrspace(3) %address.local339, i64 512, !dbg !14
  %1367 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local340, i32 0) #0, !dbg !14
  %1368 = bitcast <4 x i32> %1367 to <8 x half>, !dbg !14
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local327, !dbg !14
  %address.local342 = getelementptr [2 x i8], ptr addrspace(3) %address.local341, i64 %address.local331, !dbg !14
  %address.local343 = getelementptr [2 x i8], ptr addrspace(3) %address.local342, i64 %address.local334, !dbg !14
  %address.local344 = getelementptr i8, ptr addrspace(3) %address.local343, i64 768, !dbg !14
  %1369 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !14
  %1370 = bitcast <4 x i32> %1369 to <8 x half>, !dbg !14
  %address.local345 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local322, !dbg !14
  %address.local346 = getelementptr i8, ptr addrspace(3) %address.local345, i64 4096, !dbg !14
  %1371 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local346, i32 0) #0, !dbg !14
  %1372 = bitcast <4 x i32> %1371 to <8 x half>, !dbg !14
  %address.local347 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local327, !dbg !14
  %address.local348 = getelementptr [2 x i8], ptr addrspace(3) %address.local347, i64 %address.local331, !dbg !14
  %address.local349 = getelementptr [2 x i8], ptr addrspace(3) %address.local348, i64 %address.local334, !dbg !14
  %address.local350 = getelementptr i8, ptr addrspace(3) %address.local349, i64 4352, !dbg !14
  %1373 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local350, i32 0) #0, !dbg !14
  %1374 = bitcast <4 x i32> %1373 to <8 x half>, !dbg !14
  %address.local351 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local327, !dbg !14
  %address.local352 = getelementptr [2 x i8], ptr addrspace(3) %address.local351, i64 %address.local331, !dbg !14
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) %address.local352, i64 %address.local334, !dbg !14
  %address.local354 = getelementptr i8, ptr addrspace(3) %address.local353, i64 4608, !dbg !14
  %1375 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !14
  %1376 = bitcast <4 x i32> %1375 to <8 x half>, !dbg !14
  %address.local355 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local327, !dbg !14
  %address.local356 = getelementptr [2 x i8], ptr addrspace(3) %address.local355, i64 %address.local331, !dbg !14
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) %address.local356, i64 %address.local334, !dbg !14
  %address.local358 = getelementptr i8, ptr addrspace(3) %address.local357, i64 4864, !dbg !14
  %1377 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !14
  %1378 = bitcast <4 x i32> %1377 to <8 x half>, !dbg !14
  %1379 = add nuw nsw i64 %93, %15, !dbg !18
  %1380 = shl i64 %1379, 19, !dbg !18
  %1381 = and i64 %1380, 16252928, !dbg !18
  %1382 = add i64 %1381, %99, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1383 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1382, !dbg !18
  %1384 = getelementptr i8, ptr addrspace(1) %1383, i64 64, !dbg !18
  %address.local359 = shl nuw nsw i64 %address.local324, 3, !dbg !11
  %address.local360 = and i64 %address.local359, 1536, !dbg !11
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local360, !dbg !11
  %address.local362 = ptrtoint ptr addrspace(3) %address.local361 to i64, !dbg !11
  %address.local363 = trunc i64 %address.local362 to i32, !dbg !11
  %1385 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local363), !dbg !18
  %1386 = zext i32 %1385 to i64, !dbg !18
  %1387 = inttoptr i64 %1386 to ptr addrspace(3), !dbg !18
  %address.local364 = shl nuw nsw i64 %address.local324, 5, !dbg !18
  %address.local365 = and i64 %address.local364, 8064, !dbg !18
  %address.local366 = shl nuw nsw i64 %address.local324, 3, !dbg !18
  %address.local367 = and i64 %address.local366, 24, !dbg !18
  %address.local368 = or disjoint i64 %address.local365, %address.local367, !dbg !18
  %address.local369 = getelementptr [2 x i8], ptr addrspace(1) %1384, i64 %address.local368, !dbg !18
  %1388 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1389 = ptrtoint ptr addrspace(1) %address.local369 to i64, !dbg !18
  %1390 = sub i64 %1389, %1388, !dbg !18
  %dma.buffer.offset877 = trunc i64 %1390 to i32, !dbg !18
  %1391 = trunc i64 %1388 to i32, !dbg !18
  %1392 = insertelement <4 x i32> poison, i32 %1391, i64 0, !dbg !18
  %1393 = lshr i64 %1388, 32, !dbg !18
  %1394 = trunc i64 %1393 to i32, !dbg !18
  %1395 = insertelement <4 x i32> %1392, i32 %1394, i64 1, !dbg !18
  %1396 = insertelement <4 x i32> %1395, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc878 = insertelement <4 x i32> %1396, i32 131072, i64 3, !dbg !18
  %1397 = ptrtoint ptr addrspace(3) %1387 to i32, !dbg !18
  %dma.buffer.lds879 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1397), !dbg !18
  %1398 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds879, i32 %dma.buffer.offset877, <4 x i32> %dma.buffer.rsrc878) #10, !dbg !18
  %address.local370 = or disjoint i64 %address.local324, 256, !dbg !11
  %address.local371 = shl nuw nsw i64 %address.local370, 3, !dbg !11
  %address.local372 = and i64 %address.local371, 3584, !dbg !11
  %address.local373 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local372, !dbg !11
  %address.local374 = ptrtoint ptr addrspace(3) %address.local373 to i64, !dbg !11
  %address.local375 = trunc i64 %address.local374 to i32, !dbg !11
  %1399 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local375), !dbg !18
  %1400 = zext i32 %1399 to i64, !dbg !18
  %1401 = inttoptr i64 %1400 to ptr addrspace(3), !dbg !18
  %address.local376 = shl nuw nsw i64 %address.local370, 5, !dbg !18
  %address.local377 = and i64 %address.local376, 16256, !dbg !18
  %address.local378 = shl nuw nsw i64 %address.local324, 3, !dbg !18
  %address.local379 = and i64 %address.local378, 24, !dbg !18
  %address.local380 = or disjoint i64 %address.local377, %address.local379, !dbg !18
  %address.local381 = getelementptr [2 x i8], ptr addrspace(1) %1384, i64 %address.local380, !dbg !18
  %1402 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1403 = ptrtoint ptr addrspace(1) %address.local381 to i64, !dbg !18
  %1404 = sub i64 %1403, %1402, !dbg !18
  %dma.buffer.offset874 = trunc i64 %1404 to i32, !dbg !18
  %1405 = trunc i64 %1402 to i32, !dbg !18
  %1406 = insertelement <4 x i32> poison, i32 %1405, i64 0, !dbg !18
  %1407 = lshr i64 %1402, 32, !dbg !18
  %1408 = trunc i64 %1407 to i32, !dbg !18
  %1409 = insertelement <4 x i32> %1406, i32 %1408, i64 1, !dbg !18
  %1410 = insertelement <4 x i32> %1409, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc875 = insertelement <4 x i32> %1410, i32 131072, i64 3, !dbg !18
  %1411 = ptrtoint ptr addrspace(3) %1401 to i32, !dbg !18
  %dma.buffer.lds876 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1411), !dbg !18
  %1412 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds876, i32 %dma.buffer.offset874, <4 x i32> %dma.buffer.rsrc875) #10, !dbg !18
  %1413 = shufflevector <8 x half> %1364, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1414 = extractvalue [2 x <8 x float>] %54, 0, !dbg !35
  %1415 = shufflevector <8 x float> %1414, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1416 = fmul <4 x float> %1415, %1059, !dbg !35
  %1417 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1413, <4 x float> %1416) #9, !dbg !14
  %1418 = shufflevector <8 x half> %1366, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1418, <4 x float> %1417) #9, !dbg !14
  %1420 = shufflevector <8 x half> %1368, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1421 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %1420, <4 x float> %1419) #9, !dbg !14
  %1422 = shufflevector <8 x half> %1370, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1422, <4 x float> %1421) #9, !dbg !14
  %1424 = shufflevector <8 x half> %1372, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1424, <4 x float> %1423) #9, !dbg !14
  %1426 = shufflevector <8 x half> %1374, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1426, <4 x float> %1425) #9, !dbg !14
  %1428 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1428, <4 x float> %1427) #9, !dbg !14
  %1430 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1430, <4 x float> %1429) #9, !dbg !14
  %1432 = shufflevector <8 x half> %1364, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1433 = shufflevector <8 x float> %1414, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1434 = fmul <4 x float> %1433, %1059, !dbg !35
  %1435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1432, <4 x float> %1434) #9, !dbg !14
  %1436 = shufflevector <8 x half> %1366, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1436, <4 x float> %1435) #9, !dbg !14
  %1438 = shufflevector <8 x half> %1368, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %1438, <4 x float> %1437) #9, !dbg !14
  %1440 = shufflevector <8 x half> %1370, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1440, <4 x float> %1439) #9, !dbg !14
  %1442 = shufflevector <8 x half> %1372, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1442, <4 x float> %1441) #9, !dbg !14
  %1444 = shufflevector <8 x half> %1374, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1444, <4 x float> %1443) #9, !dbg !14
  %1446 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1446, <4 x float> %1445) #9, !dbg !14
  %1448 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1448, <4 x float> %1447) #9, !dbg !14
  %1450 = extractvalue [2 x <8 x float>] %54, 1, !dbg !35
  %1451 = shufflevector <8 x float> %1450, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1452 = fmul <4 x float> %1451, %1067, !dbg !35
  %1453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1413, <4 x float> %1452) #9, !dbg !14
  %1454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1418, <4 x float> %1453) #9, !dbg !14
  %1455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1420, <4 x float> %1454) #9, !dbg !14
  %1456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1422, <4 x float> %1455) #9, !dbg !14
  %1457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1424, <4 x float> %1456) #9, !dbg !14
  %1458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1426, <4 x float> %1457) #9, !dbg !14
  %1459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1428, <4 x float> %1458) #9, !dbg !14
  %1460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1430, <4 x float> %1459) #9, !dbg !14
  %1461 = shufflevector <8 x float> %1450, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1462 = fmul <4 x float> %1461, %1067, !dbg !35
  %1463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1432, <4 x float> %1462) #9, !dbg !14
  %1464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1436, <4 x float> %1463) #9, !dbg !14
  %1465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1438, <4 x float> %1464) #9, !dbg !14
  %1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1440, <4 x float> %1465) #9, !dbg !14
  %1467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1442, <4 x float> %1466) #9, !dbg !14
  %1468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1444, <4 x float> %1467) #9, !dbg !14
  %1469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1446, <4 x float> %1468) #9, !dbg !14
  %1470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1448, <4 x float> %1469) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1471 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local323, i32 0) #0, !dbg !14
  %1472 = bitcast <4 x i32> %1471 to <8 x half>, !dbg !14
  %1473 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local336, i32 0) #0, !dbg !14
  %1474 = bitcast <4 x i32> %1473 to <8 x half>, !dbg !14
  %1475 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local340, i32 0) #0, !dbg !14
  %1476 = bitcast <4 x i32> %1475 to <8 x half>, !dbg !14
  %1477 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !14
  %1478 = bitcast <4 x i32> %1477 to <8 x half>, !dbg !14
  %1479 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local346, i32 0) #0, !dbg !14
  %1480 = bitcast <4 x i32> %1479 to <8 x half>, !dbg !14
  %1481 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local350, i32 0) #0, !dbg !14
  %1482 = bitcast <4 x i32> %1481 to <8 x half>, !dbg !14
  %1483 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !14
  %1484 = bitcast <4 x i32> %1483 to <8 x half>, !dbg !14
  %1485 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !14
  %1486 = bitcast <4 x i32> %1485 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1487 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1382, !dbg !18
  %1488 = getelementptr i8, ptr addrspace(1) %1487, i64 128, !dbg !18
  %1489 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local363), !dbg !18
  %1490 = zext i32 %1489 to i64, !dbg !18
  %1491 = inttoptr i64 %1490 to ptr addrspace(3), !dbg !18
  %address.local382 = getelementptr [2 x i8], ptr addrspace(1) %1488, i64 %address.local368, !dbg !18
  %1492 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1493 = ptrtoint ptr addrspace(1) %address.local382 to i64, !dbg !18
  %1494 = sub i64 %1493, %1492, !dbg !18
  %dma.buffer.offset871 = trunc i64 %1494 to i32, !dbg !18
  %1495 = trunc i64 %1492 to i32, !dbg !18
  %1496 = insertelement <4 x i32> poison, i32 %1495, i64 0, !dbg !18
  %1497 = lshr i64 %1492, 32, !dbg !18
  %1498 = trunc i64 %1497 to i32, !dbg !18
  %1499 = insertelement <4 x i32> %1496, i32 %1498, i64 1, !dbg !18
  %1500 = insertelement <4 x i32> %1499, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc872 = insertelement <4 x i32> %1500, i32 131072, i64 3, !dbg !18
  %1501 = ptrtoint ptr addrspace(3) %1491 to i32, !dbg !18
  %dma.buffer.lds873 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1501), !dbg !18
  %1502 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds873, i32 %dma.buffer.offset871, <4 x i32> %dma.buffer.rsrc872) #10, !dbg !18
  %1503 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local375), !dbg !18
  %1504 = zext i32 %1503 to i64, !dbg !18
  %1505 = inttoptr i64 %1504 to ptr addrspace(3), !dbg !18
  %address.local383 = getelementptr [2 x i8], ptr addrspace(1) %1488, i64 %address.local380, !dbg !18
  %1506 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1507 = ptrtoint ptr addrspace(1) %address.local383 to i64, !dbg !18
  %1508 = sub i64 %1507, %1506, !dbg !18
  %dma.buffer.offset868 = trunc i64 %1508 to i32, !dbg !18
  %1509 = trunc i64 %1506 to i32, !dbg !18
  %1510 = insertelement <4 x i32> poison, i32 %1509, i64 0, !dbg !18
  %1511 = lshr i64 %1506, 32, !dbg !18
  %1512 = trunc i64 %1511 to i32, !dbg !18
  %1513 = insertelement <4 x i32> %1510, i32 %1512, i64 1, !dbg !18
  %1514 = insertelement <4 x i32> %1513, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc869 = insertelement <4 x i32> %1514, i32 131072, i64 3, !dbg !18
  %1515 = ptrtoint ptr addrspace(3) %1505 to i32, !dbg !18
  %dma.buffer.lds870 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1515), !dbg !18
  %1516 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds870, i32 %dma.buffer.offset868, <4 x i32> %dma.buffer.rsrc869) #10, !dbg !18
  %1517 = shufflevector <8 x half> %1472, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1518 = extractvalue [2 x <8 x float>] %57, 0, !dbg !35
  %1519 = shufflevector <8 x float> %1518, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1520 = fmul <4 x float> %1519, %1059, !dbg !35
  %1521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1517, <4 x float> %1520) #9, !dbg !14
  %1522 = shufflevector <8 x half> %1474, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1522, <4 x float> %1521) #9, !dbg !14
  %1524 = shufflevector <8 x half> %1476, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %1524, <4 x float> %1523) #9, !dbg !14
  %1526 = shufflevector <8 x half> %1478, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1526, <4 x float> %1525) #9, !dbg !14
  %1528 = shufflevector <8 x half> %1480, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1528, <4 x float> %1527) #9, !dbg !14
  %1530 = shufflevector <8 x half> %1482, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1530, <4 x float> %1529) #9, !dbg !14
  %1532 = shufflevector <8 x half> %1484, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1532, <4 x float> %1531) #9, !dbg !14
  %1534 = shufflevector <8 x half> %1486, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1534, <4 x float> %1533) #9, !dbg !14
  %1536 = shufflevector <8 x half> %1472, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1537 = shufflevector <8 x float> %1518, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1538 = fmul <4 x float> %1537, %1059, !dbg !35
  %1539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1536, <4 x float> %1538) #9, !dbg !14
  %1540 = shufflevector <8 x half> %1474, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1540, <4 x float> %1539) #9, !dbg !14
  %1542 = shufflevector <8 x half> %1476, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %1542, <4 x float> %1541) #9, !dbg !14
  %1544 = shufflevector <8 x half> %1478, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1544, <4 x float> %1543) #9, !dbg !14
  %1546 = shufflevector <8 x half> %1480, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1546, <4 x float> %1545) #9, !dbg !14
  %1548 = shufflevector <8 x half> %1482, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1548, <4 x float> %1547) #9, !dbg !14
  %1550 = shufflevector <8 x half> %1484, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1550, <4 x float> %1549) #9, !dbg !14
  %1552 = shufflevector <8 x half> %1486, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1552, <4 x float> %1551) #9, !dbg !14
  %1554 = extractvalue [2 x <8 x float>] %57, 1, !dbg !35
  %1555 = shufflevector <8 x float> %1554, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1556 = fmul <4 x float> %1555, %1067, !dbg !35
  %1557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1517, <4 x float> %1556) #9, !dbg !14
  %1558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1522, <4 x float> %1557) #9, !dbg !14
  %1559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1524, <4 x float> %1558) #9, !dbg !14
  %1560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1526, <4 x float> %1559) #9, !dbg !14
  %1561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1528, <4 x float> %1560) #9, !dbg !14
  %1562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1530, <4 x float> %1561) #9, !dbg !14
  %1563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1532, <4 x float> %1562) #9, !dbg !14
  %1564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1534, <4 x float> %1563) #9, !dbg !14
  %1565 = shufflevector <8 x float> %1554, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1566 = fmul <4 x float> %1565, %1067, !dbg !35
  %1567 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1536, <4 x float> %1566) #9, !dbg !14
  %1568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1540, <4 x float> %1567) #9, !dbg !14
  %1569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1542, <4 x float> %1568) #9, !dbg !14
  %1570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1544, <4 x float> %1569) #9, !dbg !14
  %1571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1546, <4 x float> %1570) #9, !dbg !14
  %1572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1548, <4 x float> %1571) #9, !dbg !14
  %1573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1550, <4 x float> %1572) #9, !dbg !14
  %1574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1552, <4 x float> %1573) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1575 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local323, i32 0) #0, !dbg !14
  %1576 = bitcast <4 x i32> %1575 to <8 x half>, !dbg !14
  %1577 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local336, i32 0) #0, !dbg !14
  %1578 = bitcast <4 x i32> %1577 to <8 x half>, !dbg !14
  %1579 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local340, i32 0) #0, !dbg !14
  %1580 = bitcast <4 x i32> %1579 to <8 x half>, !dbg !14
  %1581 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !14
  %1582 = bitcast <4 x i32> %1581 to <8 x half>, !dbg !14
  %1583 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local346, i32 0) #0, !dbg !14
  %1584 = bitcast <4 x i32> %1583 to <8 x half>, !dbg !14
  %1585 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local350, i32 0) #0, !dbg !14
  %1586 = bitcast <4 x i32> %1585 to <8 x half>, !dbg !14
  %1587 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !14
  %1588 = bitcast <4 x i32> %1587 to <8 x half>, !dbg !14
  %1589 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !14
  %1590 = bitcast <4 x i32> %1589 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1591 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1382, !dbg !18
  %1592 = getelementptr i8, ptr addrspace(1) %1591, i64 192, !dbg !18
  %1593 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local363), !dbg !18
  %1594 = zext i32 %1593 to i64, !dbg !18
  %1595 = inttoptr i64 %1594 to ptr addrspace(3), !dbg !18
  %address.local384 = getelementptr [2 x i8], ptr addrspace(1) %1592, i64 %address.local368, !dbg !18
  %1596 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1597 = ptrtoint ptr addrspace(1) %address.local384 to i64, !dbg !18
  %1598 = sub i64 %1597, %1596, !dbg !18
  %dma.buffer.offset865 = trunc i64 %1598 to i32, !dbg !18
  %1599 = trunc i64 %1596 to i32, !dbg !18
  %1600 = insertelement <4 x i32> poison, i32 %1599, i64 0, !dbg !18
  %1601 = lshr i64 %1596, 32, !dbg !18
  %1602 = trunc i64 %1601 to i32, !dbg !18
  %1603 = insertelement <4 x i32> %1600, i32 %1602, i64 1, !dbg !18
  %1604 = insertelement <4 x i32> %1603, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc866 = insertelement <4 x i32> %1604, i32 131072, i64 3, !dbg !18
  %1605 = ptrtoint ptr addrspace(3) %1595 to i32, !dbg !18
  %dma.buffer.lds867 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1605), !dbg !18
  %1606 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds867, i32 %dma.buffer.offset865, <4 x i32> %dma.buffer.rsrc866) #10, !dbg !18
  %1607 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local375), !dbg !18
  %1608 = zext i32 %1607 to i64, !dbg !18
  %1609 = inttoptr i64 %1608 to ptr addrspace(3), !dbg !18
  %address.local385 = getelementptr [2 x i8], ptr addrspace(1) %1592, i64 %address.local380, !dbg !18
  %1610 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1611 = ptrtoint ptr addrspace(1) %address.local385 to i64, !dbg !18
  %1612 = sub i64 %1611, %1610, !dbg !18
  %dma.buffer.offset = trunc i64 %1612 to i32, !dbg !18
  %1613 = trunc i64 %1610 to i32, !dbg !18
  %1614 = insertelement <4 x i32> poison, i32 %1613, i64 0, !dbg !18
  %1615 = lshr i64 %1610, 32, !dbg !18
  %1616 = trunc i64 %1615 to i32, !dbg !18
  %1617 = insertelement <4 x i32> %1614, i32 %1616, i64 1, !dbg !18
  %1618 = insertelement <4 x i32> %1617, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1618, i32 131072, i64 3, !dbg !18
  %1619 = ptrtoint ptr addrspace(3) %1609 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1619), !dbg !18
  %1620 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1621 = shufflevector <8 x half> %1576, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1622 = extractvalue [2 x <8 x float>] %58, 0, !dbg !35
  %1623 = shufflevector <8 x float> %1622, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1624 = fmul <4 x float> %1623, %1059, !dbg !35
  %1625 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1621, <4 x float> %1624) #9, !dbg !14
  %1626 = shufflevector <8 x half> %1578, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1627 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1626, <4 x float> %1625) #9, !dbg !14
  %1628 = shufflevector <8 x half> %1580, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1629 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %1628, <4 x float> %1627) #9, !dbg !14
  %1630 = shufflevector <8 x half> %1582, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1631 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1630, <4 x float> %1629) #9, !dbg !14
  %1632 = shufflevector <8 x half> %1584, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1633 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1632, <4 x float> %1631) #9, !dbg !14
  %1634 = shufflevector <8 x half> %1586, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1634, <4 x float> %1633) #9, !dbg !14
  %1636 = shufflevector <8 x half> %1588, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1636, <4 x float> %1635) #9, !dbg !14
  %1638 = shufflevector <8 x half> %1590, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1638, <4 x float> %1637) #9, !dbg !14
  %1640 = shufflevector <8 x half> %1576, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1641 = shufflevector <8 x float> %1622, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1642 = fmul <4 x float> %1641, %1059, !dbg !35
  %1643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1640, <4 x float> %1642) #9, !dbg !14
  %1644 = shufflevector <8 x half> %1578, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1644, <4 x float> %1643) #9, !dbg !14
  %1646 = shufflevector <8 x half> %1580, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %1646, <4 x float> %1645) #9, !dbg !14
  %1648 = shufflevector <8 x half> %1582, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1648, <4 x float> %1647) #9, !dbg !14
  %1650 = shufflevector <8 x half> %1584, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1650, <4 x float> %1649) #9, !dbg !14
  %1652 = shufflevector <8 x half> %1586, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1652, <4 x float> %1651) #9, !dbg !14
  %1654 = shufflevector <8 x half> %1588, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1654, <4 x float> %1653) #9, !dbg !14
  %1656 = shufflevector <8 x half> %1590, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1656, <4 x float> %1655) #9, !dbg !14
  %1658 = extractvalue [2 x <8 x float>] %58, 1, !dbg !35
  %1659 = shufflevector <8 x float> %1658, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1660 = fmul <4 x float> %1659, %1067, !dbg !35
  %1661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1621, <4 x float> %1660) #9, !dbg !14
  %1662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1626, <4 x float> %1661) #9, !dbg !14
  %1663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1628, <4 x float> %1662) #9, !dbg !14
  %1664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1630, <4 x float> %1663) #9, !dbg !14
  %1665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1632, <4 x float> %1664) #9, !dbg !14
  %1666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1634, <4 x float> %1665) #9, !dbg !14
  %1667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1636, <4 x float> %1666) #9, !dbg !14
  %1668 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1638, <4 x float> %1667) #9, !dbg !14
  %1669 = shufflevector <8 x float> %1658, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1670 = fmul <4 x float> %1669, %1067, !dbg !35
  %1671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1640, <4 x float> %1670) #9, !dbg !14
  %1672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1644, <4 x float> %1671) #9, !dbg !14
  %1673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1646, <4 x float> %1672) #9, !dbg !14
  %1674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1648, <4 x float> %1673) #9, !dbg !14
  %1675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1650, <4 x float> %1674) #9, !dbg !14
  %1676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1652, <4 x float> %1675) #9, !dbg !14
  %1677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1654, <4 x float> %1676) #9, !dbg !14
  %1678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1656, <4 x float> %1677) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1679 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local323, i32 0) #0, !dbg !14
  %1680 = bitcast <4 x i32> %1679 to <8 x half>, !dbg !14
  %1681 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local336, i32 0) #0, !dbg !14
  %1682 = bitcast <4 x i32> %1681 to <8 x half>, !dbg !14
  %1683 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local340, i32 0) #0, !dbg !14
  %1684 = bitcast <4 x i32> %1683 to <8 x half>, !dbg !14
  %1685 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !14
  %1686 = bitcast <4 x i32> %1685 to <8 x half>, !dbg !14
  %1687 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local346, i32 0) #0, !dbg !14
  %1688 = bitcast <4 x i32> %1687 to <8 x half>, !dbg !14
  %1689 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local350, i32 0) #0, !dbg !14
  %1690 = bitcast <4 x i32> %1689 to <8 x half>, !dbg !14
  %1691 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !14
  %1692 = bitcast <4 x i32> %1691 to <8 x half>, !dbg !14
  %1693 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !14
  %1694 = bitcast <4 x i32> %1693 to <8 x half>, !dbg !14
  %1695 = shufflevector <8 x half> %1680, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1696 = extractvalue [2 x <8 x float>] %59, 0, !dbg !35
  %1697 = shufflevector <8 x float> %1696, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1698 = fmul <4 x float> %1697, %1059, !dbg !35
  %1699 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1695, <4 x float> %1698) #9, !dbg !14
  %1700 = shufflevector <8 x half> %1682, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1701 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1700, <4 x float> %1699) #9, !dbg !14
  %1702 = shufflevector <8 x half> %1684, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1703 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %1702, <4 x float> %1701) #9, !dbg !14
  %1704 = shufflevector <8 x half> %1686, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1705 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1704, <4 x float> %1703) #9, !dbg !14
  %1706 = shufflevector <8 x half> %1688, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1707 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1706, <4 x float> %1705) #9, !dbg !14
  %1708 = shufflevector <8 x half> %1690, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1708, <4 x float> %1707) #9, !dbg !14
  %1710 = shufflevector <8 x half> %1692, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1711 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1710, <4 x float> %1709) #9, !dbg !14
  %1712 = shufflevector <8 x half> %1694, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1713 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1712, <4 x float> %1711) #9, !dbg !14
  %1714 = shufflevector <8 x half> %1680, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1715 = shufflevector <8 x float> %1696, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1716 = fmul <4 x float> %1715, %1059, !dbg !35
  %1717 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1714, <4 x float> %1716) #9, !dbg !14
  %1718 = shufflevector <8 x half> %1682, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1719 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed97, <4 x half> %1718, <4 x float> %1717) #9, !dbg !14
  %1720 = shufflevector <8 x half> %1684, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1721 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed99, <4 x half> %1720, <4 x float> %1719) #9, !dbg !14
  %1722 = shufflevector <8 x half> %1686, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1723 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1722, <4 x float> %1721) #9, !dbg !14
  %1724 = shufflevector <8 x half> %1688, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1725 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1724, <4 x float> %1723) #9, !dbg !14
  %1726 = shufflevector <8 x half> %1690, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1727 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1726, <4 x float> %1725) #9, !dbg !14
  %1728 = shufflevector <8 x half> %1692, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1728, <4 x float> %1727) #9, !dbg !14
  %1730 = shufflevector <8 x half> %1694, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1731 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1730, <4 x float> %1729) #9, !dbg !14
  %1732 = extractvalue [2 x <8 x float>] %59, 1, !dbg !35
  %1733 = shufflevector <8 x float> %1732, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1734 = fmul <4 x float> %1733, %1067, !dbg !35
  %1735 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1695, <4 x float> %1734) #9, !dbg !14
  %1736 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1700, <4 x float> %1735) #9, !dbg !14
  %1737 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1702, <4 x float> %1736) #9, !dbg !14
  %1738 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1704, <4 x float> %1737) #9, !dbg !14
  %1739 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1706, <4 x float> %1738) #9, !dbg !14
  %1740 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1708, <4 x float> %1739) #9, !dbg !14
  %1741 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1710, <4 x float> %1740) #9, !dbg !14
  %1742 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1712, <4 x float> %1741) #9, !dbg !14
  %1743 = shufflevector <8 x float> %1732, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1744 = fmul <4 x float> %1743, %1067, !dbg !35
  %1745 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1714, <4 x float> %1744) #9, !dbg !14
  %1746 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1718, <4 x float> %1745) #9, !dbg !14
  %1747 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1720, <4 x float> %1746) #9, !dbg !14
  %1748 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1722, <4 x float> %1747) #9, !dbg !14
  %1749 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1724, <4 x float> %1748) #9, !dbg !14
  %1750 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1726, <4 x float> %1749) #9, !dbg !14
  %1751 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1728, <4 x float> %1750) #9, !dbg !14
  %1752 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1730, <4 x float> %1751) #9, !dbg !14
  %1753 = shufflevector <4 x float> %1431, <4 x float> %1449, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1754 = insertvalue [2 x <8 x float>] poison, <8 x float> %1753, 0, !dbg !7
  %1755 = shufflevector <4 x float> %1460, <4 x float> %1470, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1756 = insertvalue [2 x <8 x float>] %1754, <8 x float> %1755, 1, !dbg !7
  %1757 = insertvalue [2 x <1 x float>] poison, <1 x float> %993, 0, !dbg !7
  %1758 = insertvalue [2 x <1 x float>] %1757, <1 x float> %1036, 1, !dbg !7
  %1759 = shufflevector <4 x float> %1535, <4 x float> %1553, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1760 = insertvalue [2 x <8 x float>] poison, <8 x float> %1759, 0, !dbg !7
  %1761 = shufflevector <4 x float> %1564, <4 x float> %1574, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1762 = insertvalue [2 x <8 x float>] %1760, <8 x float> %1761, 1, !dbg !7
  %1763 = shufflevector <4 x float> %1639, <4 x float> %1657, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1764 = insertvalue [2 x <8 x float>] poison, <8 x float> %1763, 0, !dbg !7
  %1765 = shufflevector <4 x float> %1668, <4 x float> %1678, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1766 = insertvalue [2 x <8 x float>] %1764, <8 x float> %1765, 1, !dbg !7
  %1767 = shufflevector <4 x float> %1713, <4 x float> %1731, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1768 = insertvalue [2 x <8 x float>] poison, <8 x float> %1767, 0, !dbg !7
  %1769 = shufflevector <4 x float> %1742, <4 x float> %1752, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1770 = insertvalue [2 x <8 x float>] %1768, <8 x float> %1769, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1771 = add nuw nsw i64 %53, 1, !dbg !7
  br label %52, !dbg !7

1772:                                             ; preds = %52
  %1773 = and i32 %34, 16, !dbg !7
  %.not = icmp eq i32 %1773, 0, !dbg !7
  %1774 = and i32 %34, 32, !dbg !7
  %.not1 = icmp eq i32 %1774, 0, !dbg !7
  %1775 = extractvalue [2 x <8 x float>] %59, 0, !dbg !7
  %1776 = extractelement <8 x float> %1775, i64 0, !dbg !7
  %1777 = extractelement <8 x float> %1775, i64 1, !dbg !7
  %1778 = extractelement <8 x float> %1775, i64 2, !dbg !7
  %1779 = extractelement <8 x float> %1775, i64 3, !dbg !7
  %1780 = select i1 %.not, float %1777, float %1776, !dbg !7
  %1781 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1782 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1781), !dbg !7
  %1783 = add nuw nsw i32 %1782, 64, !dbg !7
  %1784 = xor i32 %1782, 16, !dbg !7
  %1785 = and i32 %1783, 192, !dbg !7
  %1786 = icmp samesign ult i32 %1784, %1785, !dbg !7
  %1787 = select i1 %1786, i32 %1784, i32 %1782, !dbg !7
  %1788 = shl nuw nsw i32 %1787, 2, !dbg !7
  %1789 = bitcast float %1780 to i32, !dbg !7
  %1790 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1789), !dbg !7
  %1791 = bitcast i32 %1790 to float, !dbg !7
  %1792 = select i1 %.not, float %1776, float %1791, !dbg !7
  %1793 = select i1 %.not, float %1791, float %1777, !dbg !7
  %1794 = select i1 %.not, float %1779, float %1778, !dbg !7
  %1795 = bitcast float %1794 to i32, !dbg !7
  %1796 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1795), !dbg !7
  %1797 = bitcast i32 %1796 to float, !dbg !7
  %1798 = select i1 %.not, float %1778, float %1797, !dbg !7
  %1799 = select i1 %.not, float %1797, float %1779, !dbg !7
  %1800 = select i1 %.not1, float %1798, float %1792, !dbg !7
  %1801 = xor i32 %1782, 32, !dbg !7
  %1802 = icmp samesign ult i32 %1801, %1785, !dbg !7
  %1803 = select i1 %1802, i32 %1801, i32 %1782, !dbg !7
  %1804 = shl nuw nsw i32 %1803, 2, !dbg !7
  %1805 = bitcast float %1800 to i32, !dbg !7
  %1806 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1805), !dbg !7
  %1807 = bitcast i32 %1806 to float, !dbg !7
  %1808 = select i1 %.not1, float %1792, float %1807, !dbg !7
  %1809 = select i1 %.not1, float %1807, float %1798, !dbg !7
  %1810 = select i1 %.not1, float %1799, float %1793, !dbg !7
  %1811 = bitcast float %1810 to i32, !dbg !7
  %1812 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1811), !dbg !7
  %1813 = bitcast i32 %1812 to float, !dbg !7
  %1814 = select i1 %.not1, float %1793, float %1813, !dbg !7
  %1815 = select i1 %.not1, float %1813, float %1799, !dbg !7
  %1816 = extractvalue [2 x <8 x float>] %58, 0, !dbg !7
  %1817 = extractelement <8 x float> %1816, i64 0, !dbg !7
  %1818 = extractelement <8 x float> %1816, i64 1, !dbg !7
  %1819 = extractelement <8 x float> %1816, i64 2, !dbg !7
  %1820 = extractelement <8 x float> %1816, i64 3, !dbg !7
  %1821 = select i1 %.not, float %1818, float %1817, !dbg !7
  %1822 = bitcast float %1821 to i32, !dbg !7
  %1823 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1822), !dbg !7
  %1824 = bitcast i32 %1823 to float, !dbg !7
  %1825 = select i1 %.not, float %1817, float %1824, !dbg !7
  %1826 = select i1 %.not, float %1824, float %1818, !dbg !7
  %1827 = select i1 %.not, float %1820, float %1819, !dbg !7
  %1828 = bitcast float %1827 to i32, !dbg !7
  %1829 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1828), !dbg !7
  %1830 = bitcast i32 %1829 to float, !dbg !7
  %1831 = select i1 %.not, float %1819, float %1830, !dbg !7
  %1832 = select i1 %.not, float %1830, float %1820, !dbg !7
  %1833 = select i1 %.not1, float %1831, float %1825, !dbg !7
  %1834 = bitcast float %1833 to i32, !dbg !7
  %1835 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1834), !dbg !7
  %1836 = bitcast i32 %1835 to float, !dbg !7
  %1837 = select i1 %.not1, float %1825, float %1836, !dbg !7
  %1838 = select i1 %.not1, float %1836, float %1831, !dbg !7
  %1839 = select i1 %.not1, float %1832, float %1826, !dbg !7
  %1840 = bitcast float %1839 to i32, !dbg !7
  %1841 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1840), !dbg !7
  %1842 = bitcast i32 %1841 to float, !dbg !7
  %1843 = select i1 %.not1, float %1826, float %1842, !dbg !7
  %1844 = select i1 %.not1, float %1842, float %1832, !dbg !7
  %1845 = extractvalue [2 x <8 x float>] %57, 0, !dbg !7
  %1846 = extractelement <8 x float> %1845, i64 0, !dbg !7
  %1847 = extractelement <8 x float> %1845, i64 1, !dbg !7
  %1848 = extractelement <8 x float> %1845, i64 2, !dbg !7
  %1849 = extractelement <8 x float> %1845, i64 3, !dbg !7
  %1850 = select i1 %.not, float %1847, float %1846, !dbg !7
  %1851 = bitcast float %1850 to i32, !dbg !7
  %1852 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1851), !dbg !7
  %1853 = bitcast i32 %1852 to float, !dbg !7
  %1854 = select i1 %.not, float %1846, float %1853, !dbg !7
  %1855 = select i1 %.not, float %1853, float %1847, !dbg !7
  %1856 = select i1 %.not, float %1849, float %1848, !dbg !7
  %1857 = bitcast float %1856 to i32, !dbg !7
  %1858 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1857), !dbg !7
  %1859 = bitcast i32 %1858 to float, !dbg !7
  %1860 = select i1 %.not, float %1848, float %1859, !dbg !7
  %1861 = select i1 %.not, float %1859, float %1849, !dbg !7
  %1862 = select i1 %.not1, float %1860, float %1854, !dbg !7
  %1863 = bitcast float %1862 to i32, !dbg !7
  %1864 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1863), !dbg !7
  %1865 = bitcast i32 %1864 to float, !dbg !7
  %1866 = select i1 %.not1, float %1854, float %1865, !dbg !7
  %1867 = select i1 %.not1, float %1865, float %1860, !dbg !7
  %1868 = select i1 %.not1, float %1861, float %1855, !dbg !7
  %1869 = bitcast float %1868 to i32, !dbg !7
  %1870 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1869), !dbg !7
  %1871 = bitcast i32 %1870 to float, !dbg !7
  %1872 = select i1 %.not1, float %1855, float %1871, !dbg !7
  %1873 = select i1 %.not1, float %1871, float %1861, !dbg !7
  %1874 = extractvalue [2 x <8 x float>] %54, 0, !dbg !7
  %1875 = extractelement <8 x float> %1874, i64 0, !dbg !7
  %1876 = extractelement <8 x float> %1874, i64 1, !dbg !7
  %1877 = extractelement <8 x float> %1874, i64 2, !dbg !7
  %1878 = extractelement <8 x float> %1874, i64 3, !dbg !7
  %1879 = select i1 %.not, float %1876, float %1875, !dbg !7
  %1880 = bitcast float %1879 to i32, !dbg !7
  %1881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1880), !dbg !7
  %1882 = bitcast i32 %1881 to float, !dbg !7
  %1883 = select i1 %.not, float %1875, float %1882, !dbg !7
  %1884 = select i1 %.not, float %1882, float %1876, !dbg !7
  %1885 = select i1 %.not, float %1878, float %1877, !dbg !7
  %1886 = bitcast float %1885 to i32, !dbg !7
  %1887 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1886), !dbg !7
  %1888 = bitcast i32 %1887 to float, !dbg !7
  %1889 = select i1 %.not, float %1877, float %1888, !dbg !7
  %1890 = select i1 %.not, float %1888, float %1878, !dbg !7
  %1891 = select i1 %.not1, float %1889, float %1883, !dbg !7
  %1892 = bitcast float %1891 to i32, !dbg !7
  %1893 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1892), !dbg !7
  %1894 = select i1 %.not1, float %1890, float %1884, !dbg !7
  %1895 = bitcast float %1894 to i32, !dbg !7
  %1896 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1895), !dbg !7
  %1897 = extractvalue [2 x <1 x float>] %55, 0, !dbg !19
  %1898 = shufflevector <1 x float> %1897, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1899 = extractelement <8 x float> %1775, i64 4, !dbg !7
  %1900 = extractelement <8 x float> %1775, i64 5, !dbg !7
  %1901 = extractelement <8 x float> %1775, i64 6, !dbg !7
  %1902 = extractelement <8 x float> %1775, i64 7, !dbg !7
  %1903 = select i1 %.not, float %1900, float %1899, !dbg !7
  %1904 = bitcast float %1903 to i32, !dbg !7
  %1905 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1904), !dbg !7
  %1906 = bitcast i32 %1905 to float, !dbg !7
  %1907 = select i1 %.not, float %1899, float %1906, !dbg !7
  %1908 = select i1 %.not, float %1906, float %1900, !dbg !7
  %1909 = select i1 %.not, float %1902, float %1901, !dbg !7
  %1910 = bitcast float %1909 to i32, !dbg !7
  %1911 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1910), !dbg !7
  %1912 = bitcast i32 %1911 to float, !dbg !7
  %1913 = select i1 %.not, float %1901, float %1912, !dbg !7
  %1914 = select i1 %.not, float %1912, float %1902, !dbg !7
  %1915 = select i1 %.not1, float %1913, float %1907, !dbg !7
  %1916 = bitcast float %1915 to i32, !dbg !7
  %1917 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1916), !dbg !7
  %1918 = bitcast i32 %1917 to float, !dbg !7
  %1919 = select i1 %.not1, float %1907, float %1918, !dbg !7
  %1920 = select i1 %.not1, float %1918, float %1913, !dbg !7
  %1921 = select i1 %.not1, float %1914, float %1908, !dbg !7
  %1922 = bitcast float %1921 to i32, !dbg !7
  %1923 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1922), !dbg !7
  %1924 = bitcast i32 %1923 to float, !dbg !7
  %1925 = select i1 %.not1, float %1908, float %1924, !dbg !7
  %1926 = select i1 %.not1, float %1924, float %1914, !dbg !7
  %1927 = extractelement <8 x float> %1816, i64 4, !dbg !7
  %1928 = extractelement <8 x float> %1816, i64 5, !dbg !7
  %1929 = extractelement <8 x float> %1816, i64 6, !dbg !7
  %1930 = extractelement <8 x float> %1816, i64 7, !dbg !7
  %1931 = select i1 %.not, float %1928, float %1927, !dbg !7
  %1932 = bitcast float %1931 to i32, !dbg !7
  %1933 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1932), !dbg !7
  %1934 = bitcast i32 %1933 to float, !dbg !7
  %1935 = select i1 %.not, float %1927, float %1934, !dbg !7
  %1936 = select i1 %.not, float %1934, float %1928, !dbg !7
  %1937 = select i1 %.not, float %1930, float %1929, !dbg !7
  %1938 = bitcast float %1937 to i32, !dbg !7
  %1939 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1938), !dbg !7
  %1940 = bitcast i32 %1939 to float, !dbg !7
  %1941 = select i1 %.not, float %1929, float %1940, !dbg !7
  %1942 = select i1 %.not, float %1940, float %1930, !dbg !7
  %1943 = select i1 %.not1, float %1941, float %1935, !dbg !7
  %1944 = bitcast float %1943 to i32, !dbg !7
  %1945 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1944), !dbg !7
  %1946 = bitcast i32 %1945 to float, !dbg !7
  %1947 = select i1 %.not1, float %1935, float %1946, !dbg !7
  %1948 = select i1 %.not1, float %1946, float %1941, !dbg !7
  %1949 = select i1 %.not1, float %1942, float %1936, !dbg !7
  %1950 = bitcast float %1949 to i32, !dbg !7
  %1951 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1950), !dbg !7
  %1952 = bitcast i32 %1951 to float, !dbg !7
  %1953 = select i1 %.not1, float %1936, float %1952, !dbg !7
  %1954 = select i1 %.not1, float %1952, float %1942, !dbg !7
  %1955 = extractelement <8 x float> %1845, i64 4, !dbg !7
  %1956 = extractelement <8 x float> %1845, i64 5, !dbg !7
  %1957 = extractelement <8 x float> %1845, i64 6, !dbg !7
  %1958 = extractelement <8 x float> %1845, i64 7, !dbg !7
  %1959 = select i1 %.not, float %1956, float %1955, !dbg !7
  %1960 = bitcast float %1959 to i32, !dbg !7
  %1961 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1960), !dbg !7
  %1962 = bitcast i32 %1961 to float, !dbg !7
  %1963 = select i1 %.not, float %1955, float %1962, !dbg !7
  %1964 = select i1 %.not, float %1962, float %1956, !dbg !7
  %1965 = select i1 %.not, float %1958, float %1957, !dbg !7
  %1966 = bitcast float %1965 to i32, !dbg !7
  %1967 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1966), !dbg !7
  %1968 = bitcast i32 %1967 to float, !dbg !7
  %1969 = select i1 %.not, float %1957, float %1968, !dbg !7
  %1970 = select i1 %.not, float %1968, float %1958, !dbg !7
  %1971 = select i1 %.not1, float %1969, float %1963, !dbg !7
  %1972 = bitcast float %1971 to i32, !dbg !7
  %1973 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1972), !dbg !7
  %1974 = bitcast i32 %1973 to float, !dbg !7
  %1975 = select i1 %.not1, float %1963, float %1974, !dbg !7
  %1976 = select i1 %.not1, float %1974, float %1969, !dbg !7
  %1977 = select i1 %.not1, float %1970, float %1964, !dbg !7
  %1978 = bitcast float %1977 to i32, !dbg !7
  %1979 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %1978), !dbg !7
  %1980 = bitcast i32 %1979 to float, !dbg !7
  %1981 = select i1 %.not1, float %1964, float %1980, !dbg !7
  %1982 = select i1 %.not1, float %1980, float %1970, !dbg !7
  %1983 = extractelement <8 x float> %1874, i64 4, !dbg !7
  %1984 = extractelement <8 x float> %1874, i64 5, !dbg !7
  %1985 = extractelement <8 x float> %1874, i64 6, !dbg !7
  %1986 = extractelement <8 x float> %1874, i64 7, !dbg !7
  %1987 = select i1 %.not, float %1984, float %1983, !dbg !7
  %1988 = bitcast float %1987 to i32, !dbg !7
  %1989 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1988), !dbg !7
  %1990 = bitcast i32 %1989 to float, !dbg !7
  %1991 = select i1 %.not, float %1983, float %1990, !dbg !7
  %1992 = select i1 %.not, float %1990, float %1984, !dbg !7
  %1993 = select i1 %.not, float %1986, float %1985, !dbg !7
  %1994 = bitcast float %1993 to i32, !dbg !7
  %1995 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %1994), !dbg !7
  %1996 = bitcast i32 %1995 to float, !dbg !7
  %1997 = select i1 %.not, float %1985, float %1996, !dbg !7
  %1998 = select i1 %.not, float %1996, float %1986, !dbg !7
  %1999 = select i1 %.not1, float %1997, float %1991, !dbg !7
  %2000 = bitcast float %1999 to i32, !dbg !7
  %2001 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2000), !dbg !7
  %2002 = select i1 %.not1, float %1998, float %1992, !dbg !7
  %2003 = bitcast float %2002 to i32, !dbg !7
  %2004 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2003), !dbg !7
  %2005 = extractvalue [2 x <8 x float>] %59, 1, !dbg !7
  %2006 = extractelement <8 x float> %2005, i64 0, !dbg !7
  %2007 = extractelement <8 x float> %2005, i64 1, !dbg !7
  %2008 = extractelement <8 x float> %2005, i64 2, !dbg !7
  %2009 = extractelement <8 x float> %2005, i64 3, !dbg !7
  %2010 = select i1 %.not, float %2007, float %2006, !dbg !7
  %2011 = bitcast float %2010 to i32, !dbg !7
  %2012 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2011), !dbg !7
  %2013 = bitcast i32 %2012 to float, !dbg !7
  %2014 = select i1 %.not, float %2006, float %2013, !dbg !7
  %2015 = select i1 %.not, float %2013, float %2007, !dbg !7
  %2016 = select i1 %.not, float %2009, float %2008, !dbg !7
  %2017 = bitcast float %2016 to i32, !dbg !7
  %2018 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2017), !dbg !7
  %2019 = bitcast i32 %2018 to float, !dbg !7
  %2020 = select i1 %.not, float %2008, float %2019, !dbg !7
  %2021 = select i1 %.not, float %2019, float %2009, !dbg !7
  %2022 = select i1 %.not1, float %2020, float %2014, !dbg !7
  %2023 = bitcast float %2022 to i32, !dbg !7
  %2024 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2023), !dbg !7
  %2025 = bitcast i32 %2024 to float, !dbg !7
  %2026 = select i1 %.not1, float %2014, float %2025, !dbg !7
  %2027 = select i1 %.not1, float %2025, float %2020, !dbg !7
  %2028 = select i1 %.not1, float %2021, float %2015, !dbg !7
  %2029 = bitcast float %2028 to i32, !dbg !7
  %2030 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2029), !dbg !7
  %2031 = bitcast i32 %2030 to float, !dbg !7
  %2032 = select i1 %.not1, float %2015, float %2031, !dbg !7
  %2033 = select i1 %.not1, float %2031, float %2021, !dbg !7
  %2034 = extractvalue [2 x <8 x float>] %58, 1, !dbg !7
  %2035 = extractelement <8 x float> %2034, i64 0, !dbg !7
  %2036 = extractelement <8 x float> %2034, i64 1, !dbg !7
  %2037 = extractelement <8 x float> %2034, i64 2, !dbg !7
  %2038 = extractelement <8 x float> %2034, i64 3, !dbg !7
  %2039 = select i1 %.not, float %2036, float %2035, !dbg !7
  %2040 = bitcast float %2039 to i32, !dbg !7
  %2041 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2040), !dbg !7
  %2042 = bitcast i32 %2041 to float, !dbg !7
  %2043 = select i1 %.not, float %2035, float %2042, !dbg !7
  %2044 = select i1 %.not, float %2042, float %2036, !dbg !7
  %2045 = select i1 %.not, float %2038, float %2037, !dbg !7
  %2046 = bitcast float %2045 to i32, !dbg !7
  %2047 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2046), !dbg !7
  %2048 = bitcast i32 %2047 to float, !dbg !7
  %2049 = select i1 %.not, float %2037, float %2048, !dbg !7
  %2050 = select i1 %.not, float %2048, float %2038, !dbg !7
  %2051 = select i1 %.not1, float %2049, float %2043, !dbg !7
  %2052 = bitcast float %2051 to i32, !dbg !7
  %2053 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2052), !dbg !7
  %2054 = bitcast i32 %2053 to float, !dbg !7
  %2055 = select i1 %.not1, float %2043, float %2054, !dbg !7
  %2056 = select i1 %.not1, float %2054, float %2049, !dbg !7
  %2057 = select i1 %.not1, float %2050, float %2044, !dbg !7
  %2058 = bitcast float %2057 to i32, !dbg !7
  %2059 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2058), !dbg !7
  %2060 = bitcast i32 %2059 to float, !dbg !7
  %2061 = select i1 %.not1, float %2044, float %2060, !dbg !7
  %2062 = select i1 %.not1, float %2060, float %2050, !dbg !7
  %2063 = extractvalue [2 x <8 x float>] %57, 1, !dbg !7
  %2064 = extractelement <8 x float> %2063, i64 0, !dbg !7
  %2065 = extractelement <8 x float> %2063, i64 1, !dbg !7
  %2066 = extractelement <8 x float> %2063, i64 2, !dbg !7
  %2067 = extractelement <8 x float> %2063, i64 3, !dbg !7
  %2068 = select i1 %.not, float %2065, float %2064, !dbg !7
  %2069 = bitcast float %2068 to i32, !dbg !7
  %2070 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2069), !dbg !7
  %2071 = bitcast i32 %2070 to float, !dbg !7
  %2072 = select i1 %.not, float %2064, float %2071, !dbg !7
  %2073 = select i1 %.not, float %2071, float %2065, !dbg !7
  %2074 = select i1 %.not, float %2067, float %2066, !dbg !7
  %2075 = bitcast float %2074 to i32, !dbg !7
  %2076 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2075), !dbg !7
  %2077 = bitcast i32 %2076 to float, !dbg !7
  %2078 = select i1 %.not, float %2066, float %2077, !dbg !7
  %2079 = select i1 %.not, float %2077, float %2067, !dbg !7
  %2080 = select i1 %.not1, float %2078, float %2072, !dbg !7
  %2081 = bitcast float %2080 to i32, !dbg !7
  %2082 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2081), !dbg !7
  %2083 = bitcast i32 %2082 to float, !dbg !7
  %2084 = select i1 %.not1, float %2072, float %2083, !dbg !7
  %2085 = select i1 %.not1, float %2083, float %2078, !dbg !7
  %2086 = select i1 %.not1, float %2079, float %2073, !dbg !7
  %2087 = bitcast float %2086 to i32, !dbg !7
  %2088 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2087), !dbg !7
  %2089 = bitcast i32 %2088 to float, !dbg !7
  %2090 = select i1 %.not1, float %2073, float %2089, !dbg !7
  %2091 = select i1 %.not1, float %2089, float %2079, !dbg !7
  %2092 = extractvalue [2 x <8 x float>] %54, 1, !dbg !7
  %2093 = extractelement <8 x float> %2092, i64 0, !dbg !7
  %2094 = extractelement <8 x float> %2092, i64 1, !dbg !7
  %2095 = extractelement <8 x float> %2092, i64 2, !dbg !7
  %2096 = extractelement <8 x float> %2092, i64 3, !dbg !7
  %2097 = select i1 %.not, float %2094, float %2093, !dbg !7
  %2098 = bitcast float %2097 to i32, !dbg !7
  %2099 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2098), !dbg !7
  %2100 = bitcast i32 %2099 to float, !dbg !7
  %2101 = select i1 %.not, float %2093, float %2100, !dbg !7
  %2102 = select i1 %.not, float %2100, float %2094, !dbg !7
  %2103 = select i1 %.not, float %2096, float %2095, !dbg !7
  %2104 = bitcast float %2103 to i32, !dbg !7
  %2105 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2104), !dbg !7
  %2106 = bitcast i32 %2105 to float, !dbg !7
  %2107 = select i1 %.not, float %2095, float %2106, !dbg !7
  %2108 = select i1 %.not, float %2106, float %2096, !dbg !7
  %2109 = select i1 %.not1, float %2107, float %2101, !dbg !7
  %2110 = bitcast float %2109 to i32, !dbg !7
  %2111 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2110), !dbg !7
  %2112 = select i1 %.not1, float %2108, float %2102, !dbg !7
  %2113 = bitcast float %2112 to i32, !dbg !7
  %2114 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2113), !dbg !7
  %2115 = extractvalue [2 x <1 x float>] %55, 1, !dbg !19
  %2116 = shufflevector <1 x float> %2115, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %2117 = extractelement <8 x float> %2005, i64 4, !dbg !7
  %2118 = extractelement <8 x float> %2005, i64 5, !dbg !7
  %2119 = extractelement <8 x float> %2005, i64 6, !dbg !7
  %2120 = extractelement <8 x float> %2005, i64 7, !dbg !7
  %2121 = select i1 %.not, float %2118, float %2117, !dbg !7
  %2122 = bitcast float %2121 to i32, !dbg !7
  %2123 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2122), !dbg !7
  %2124 = bitcast i32 %2123 to float, !dbg !7
  %2125 = select i1 %.not, float %2117, float %2124, !dbg !7
  %2126 = select i1 %.not, float %2124, float %2118, !dbg !7
  %2127 = select i1 %.not, float %2120, float %2119, !dbg !7
  %2128 = bitcast float %2127 to i32, !dbg !7
  %2129 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2128), !dbg !7
  %2130 = bitcast i32 %2129 to float, !dbg !7
  %2131 = select i1 %.not, float %2119, float %2130, !dbg !7
  %2132 = select i1 %.not, float %2130, float %2120, !dbg !7
  %2133 = select i1 %.not1, float %2131, float %2125, !dbg !7
  %2134 = bitcast float %2133 to i32, !dbg !7
  %2135 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2134), !dbg !7
  %2136 = bitcast i32 %2135 to float, !dbg !7
  %2137 = select i1 %.not1, float %2125, float %2136, !dbg !7
  %2138 = select i1 %.not1, float %2136, float %2131, !dbg !7
  %2139 = select i1 %.not1, float %2132, float %2126, !dbg !7
  %2140 = bitcast float %2139 to i32, !dbg !7
  %2141 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2140), !dbg !7
  %2142 = bitcast i32 %2141 to float, !dbg !7
  %2143 = select i1 %.not1, float %2126, float %2142, !dbg !7
  %2144 = select i1 %.not1, float %2142, float %2132, !dbg !7
  %2145 = extractelement <8 x float> %2034, i64 4, !dbg !7
  %2146 = extractelement <8 x float> %2034, i64 5, !dbg !7
  %2147 = extractelement <8 x float> %2034, i64 6, !dbg !7
  %2148 = extractelement <8 x float> %2034, i64 7, !dbg !7
  %2149 = select i1 %.not, float %2146, float %2145, !dbg !7
  %2150 = bitcast float %2149 to i32, !dbg !7
  %2151 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2150), !dbg !7
  %2152 = bitcast i32 %2151 to float, !dbg !7
  %2153 = select i1 %.not, float %2145, float %2152, !dbg !7
  %2154 = select i1 %.not, float %2152, float %2146, !dbg !7
  %2155 = select i1 %.not, float %2148, float %2147, !dbg !7
  %2156 = bitcast float %2155 to i32, !dbg !7
  %2157 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2156), !dbg !7
  %2158 = bitcast i32 %2157 to float, !dbg !7
  %2159 = select i1 %.not, float %2147, float %2158, !dbg !7
  %2160 = select i1 %.not, float %2158, float %2148, !dbg !7
  %2161 = select i1 %.not1, float %2159, float %2153, !dbg !7
  %2162 = bitcast float %2161 to i32, !dbg !7
  %2163 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2162), !dbg !7
  %2164 = bitcast i32 %2163 to float, !dbg !7
  %2165 = select i1 %.not1, float %2153, float %2164, !dbg !7
  %2166 = select i1 %.not1, float %2164, float %2159, !dbg !7
  %2167 = select i1 %.not1, float %2160, float %2154, !dbg !7
  %2168 = bitcast float %2167 to i32, !dbg !7
  %2169 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2168), !dbg !7
  %2170 = bitcast i32 %2169 to float, !dbg !7
  %2171 = select i1 %.not1, float %2154, float %2170, !dbg !7
  %2172 = select i1 %.not1, float %2170, float %2160, !dbg !7
  %2173 = extractelement <8 x float> %2063, i64 4, !dbg !7
  %2174 = extractelement <8 x float> %2063, i64 5, !dbg !7
  %2175 = extractelement <8 x float> %2063, i64 6, !dbg !7
  %2176 = extractelement <8 x float> %2063, i64 7, !dbg !7
  %2177 = select i1 %.not, float %2174, float %2173, !dbg !7
  %2178 = bitcast float %2177 to i32, !dbg !7
  %2179 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2178), !dbg !7
  %2180 = bitcast i32 %2179 to float, !dbg !7
  %2181 = select i1 %.not, float %2173, float %2180, !dbg !7
  %2182 = select i1 %.not, float %2180, float %2174, !dbg !7
  %2183 = select i1 %.not, float %2176, float %2175, !dbg !7
  %2184 = bitcast float %2183 to i32, !dbg !7
  %2185 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2184), !dbg !7
  %2186 = bitcast i32 %2185 to float, !dbg !7
  %2187 = select i1 %.not, float %2175, float %2186, !dbg !7
  %2188 = select i1 %.not, float %2186, float %2176, !dbg !7
  %2189 = select i1 %.not1, float %2187, float %2181, !dbg !7
  %2190 = bitcast float %2189 to i32, !dbg !7
  %2191 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2190), !dbg !7
  %2192 = bitcast i32 %2191 to float, !dbg !7
  %2193 = select i1 %.not1, float %2181, float %2192, !dbg !7
  %2194 = select i1 %.not1, float %2192, float %2187, !dbg !7
  %2195 = select i1 %.not1, float %2188, float %2182, !dbg !7
  %2196 = bitcast float %2195 to i32, !dbg !7
  %2197 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2196), !dbg !7
  %2198 = bitcast i32 %2197 to float, !dbg !7
  %2199 = select i1 %.not1, float %2182, float %2198, !dbg !7
  %2200 = select i1 %.not1, float %2198, float %2188, !dbg !7
  %2201 = extractelement <8 x float> %2092, i64 4, !dbg !7
  %2202 = extractelement <8 x float> %2092, i64 5, !dbg !7
  %2203 = extractelement <8 x float> %2092, i64 6, !dbg !7
  %2204 = extractelement <8 x float> %2092, i64 7, !dbg !7
  %2205 = select i1 %.not, float %2202, float %2201, !dbg !7
  %2206 = bitcast float %2205 to i32, !dbg !7
  %2207 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2206), !dbg !7
  %2208 = bitcast i32 %2207 to float, !dbg !7
  %2209 = select i1 %.not, float %2201, float %2208, !dbg !7
  %2210 = select i1 %.not, float %2208, float %2202, !dbg !7
  %2211 = select i1 %.not, float %2204, float %2203, !dbg !7
  %2212 = bitcast float %2211 to i32, !dbg !7
  %2213 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1788, i32 %2212), !dbg !7
  %2214 = bitcast i32 %2213 to float, !dbg !7
  %2215 = select i1 %.not, float %2203, float %2214, !dbg !7
  %2216 = select i1 %.not, float %2214, float %2204, !dbg !7
  %2217 = select i1 %.not1, float %2215, float %2209, !dbg !7
  %2218 = bitcast float %2217 to i32, !dbg !7
  %2219 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2218), !dbg !7
  %2220 = select i1 %.not1, float %2216, float %2210, !dbg !7
  %2221 = bitcast float %2220 to i32, !dbg !7
  %2222 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1804, i32 %2221), !dbg !7
  br i1 true, label %2223, label %2287, !dbg !21

2223:                                             ; preds = %1772
  %2224 = bitcast i32 %2219 to float, !dbg !7
  %2225 = select i1 %.not1, float %2209, float %2224, !dbg !7
  %2226 = insertelement <4 x float> poison, float %2225, i64 0, !dbg !19
  %2227 = bitcast i32 %2222 to float, !dbg !7
  %2228 = select i1 %.not1, float %2210, float %2227, !dbg !7
  %2229 = insertelement <4 x float> %2226, float %2228, i64 1, !dbg !19
  %2230 = select i1 %.not1, float %2224, float %2215, !dbg !7
  %2231 = insertelement <4 x float> %2229, float %2230, i64 2, !dbg !19
  %2232 = select i1 %.not1, float %2227, float %2216, !dbg !7
  %2233 = insertelement <4 x float> %2231, float %2232, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2116, !dbg !19
  %normalized = fmul <4 x float> %2233, %row.reciprocal, !dbg !19
  %2234 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %2235 = extractelement <4 x half> %2234, i64 3, !dbg !20
  %2236 = extractelement <4 x half> %2234, i64 2, !dbg !20
  %2237 = extractelement <4 x half> %2234, i64 1, !dbg !20
  %2238 = extractelement <4 x half> %2234, i64 0, !dbg !20
  %2239 = bitcast i32 %2111 to float, !dbg !7
  %2240 = select i1 %.not1, float %2101, float %2239, !dbg !7
  %2241 = insertelement <4 x float> poison, float %2240, i64 0, !dbg !19
  %2242 = bitcast i32 %2114 to float, !dbg !7
  %2243 = select i1 %.not1, float %2102, float %2242, !dbg !7
  %2244 = insertelement <4 x float> %2241, float %2243, i64 1, !dbg !19
  %2245 = select i1 %.not1, float %2239, float %2107, !dbg !7
  %2246 = insertelement <4 x float> %2244, float %2245, i64 2, !dbg !19
  %2247 = select i1 %.not1, float %2242, float %2108, !dbg !7
  %2248 = insertelement <4 x float> %2246, float %2247, i64 3, !dbg !19
  %normalized68 = fmul <4 x float> %2248, %row.reciprocal, !dbg !19
  %2249 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !20
  %2250 = extractelement <4 x half> %2249, i64 3, !dbg !20
  %2251 = extractelement <4 x half> %2249, i64 2, !dbg !20
  %2252 = extractelement <4 x half> %2249, i64 1, !dbg !20
  %2253 = extractelement <4 x half> %2249, i64 0, !dbg !20
  %2254 = bitcast i32 %2001 to float, !dbg !7
  %2255 = select i1 %.not1, float %1991, float %2254, !dbg !7
  %2256 = insertelement <4 x float> poison, float %2255, i64 0, !dbg !19
  %2257 = bitcast i32 %2004 to float, !dbg !7
  %2258 = select i1 %.not1, float %1992, float %2257, !dbg !7
  %2259 = insertelement <4 x float> %2256, float %2258, i64 1, !dbg !19
  %2260 = select i1 %.not1, float %2254, float %1997, !dbg !7
  %2261 = insertelement <4 x float> %2259, float %2260, i64 2, !dbg !19
  %2262 = select i1 %.not1, float %2257, float %1998, !dbg !7
  %2263 = insertelement <4 x float> %2261, float %2262, i64 3, !dbg !19
  %row.reciprocal69 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1898, !dbg !19
  %normalized70 = fmul <4 x float> %2263, %row.reciprocal69, !dbg !19
  %2264 = fptrunc <4 x float> %normalized70 to <4 x half>, !dbg !20
  %2265 = extractelement <4 x half> %2264, i64 3, !dbg !20
  %2266 = extractelement <4 x half> %2264, i64 2, !dbg !20
  %2267 = extractelement <4 x half> %2264, i64 1, !dbg !20
  %2268 = extractelement <4 x half> %2264, i64 0, !dbg !20
  %2269 = bitcast i32 %1893 to float, !dbg !7
  %2270 = select i1 %.not1, float %1883, float %2269, !dbg !7
  %2271 = insertelement <4 x float> poison, float %2270, i64 0, !dbg !19
  %2272 = bitcast i32 %1896 to float, !dbg !7
  %2273 = select i1 %.not1, float %1884, float %2272, !dbg !7
  %2274 = insertelement <4 x float> %2271, float %2273, i64 1, !dbg !19
  %2275 = select i1 %.not1, float %2269, float %1889, !dbg !7
  %2276 = insertelement <4 x float> %2274, float %2275, i64 2, !dbg !19
  %2277 = select i1 %.not1, float %2272, float %1890, !dbg !7
  %2278 = insertelement <4 x float> %2276, float %2277, i64 3, !dbg !19
  %normalized71 = fmul <4 x float> %2278, %row.reciprocal69, !dbg !19
  %2279 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !20
  %2280 = extractelement <4 x half> %2279, i64 3, !dbg !20
  %2281 = extractelement <4 x half> %2279, i64 2, !dbg !20
  %2282 = extractelement <4 x half> %2279, i64 1, !dbg !20
  %2283 = extractelement <4 x half> %2279, i64 0, !dbg !20
  %2284 = add nsw i64 %15, %25, !dbg !21
  %2285 = shl i64 %2284, 19, !dbg !21
  %2286 = and i64 %2285, 16252928, !dbg !21
  %address.thread386 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local387 = zext nneg i32 %address.thread386 to i64, !dbg !12
  %address.local388 = add i64 %28, %address.local387, !dbg !21
  %address.local389 = lshr i64 %address.local387, 6, !dbg !12
  %address.local390 = shl nuw nsw i64 %address.local389, 4, !dbg !21
  %address.local391 = add i64 %address.local388, %address.local390, !dbg !21
  %address.local392 = and i64 %address.local387, 240, !dbg !21
  %address.local393 = sub i64 %address.local391, %address.local392, !dbg !21
  %address.local394 = add i64 %address.local393, %29, !dbg !21
  %address.local395 = shl nuw nsw i64 %address.local394, 7, !dbg !21
  %address.local396 = add nuw nsw i64 %2286, %address.local395, !dbg !21
  %address.local397 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local396, !dbg !21
  %address.local398 = lshr i64 %address.local387, 2, !dbg !12
  %address.local399 = and i64 %address.local398, 12, !dbg !12
  %address.local400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local397, i64 %address.local399, !dbg !21
  store half %2283, ptr addrspace(1) %address.local400, align 2, !dbg !21
  %address.local401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local396, !dbg !21
  %address.local402 = or disjoint i64 %address.local399, 1, !dbg !21
  %address.local403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local401, i64 %address.local402, !dbg !21
  store half %2282, ptr addrspace(1) %address.local403, align 2, !dbg !21
  %address.local404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local396, !dbg !21
  %address.local405 = or disjoint i64 %address.local399, 2, !dbg !21
  %address.local406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local404, i64 %address.local405, !dbg !21
  store half %2281, ptr addrspace(1) %address.local406, align 2, !dbg !21
  %address.local407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local396, !dbg !21
  %address.local408 = or disjoint i64 %address.local399, 3, !dbg !21
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local407, i64 %address.local408, !dbg !21
  store half %2280, ptr addrspace(1) %address.local409, align 2, !dbg !21
  %address.local410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local396, !dbg !21
  %address.local411 = or disjoint i64 %address.local399, 16, !dbg !21
  %address.local412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local410, i64 %address.local411, !dbg !21
  store half %2268, ptr addrspace(1) %address.local412, align 2, !dbg !21
  %address.local413 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local396, !dbg !21
  %address.local414 = or disjoint i64 %address.local399, 17, !dbg !21
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local413, i64 %address.local414, !dbg !21
  store half %2267, ptr addrspace(1) %address.local415, align 2, !dbg !21
  %address.local416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local396, !dbg !21
  %address.local417 = or disjoint i64 %address.local399, 18, !dbg !21
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local416, i64 %address.local417, !dbg !21
  store half %2266, ptr addrspace(1) %address.local418, align 2, !dbg !21
  %address.local419 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local396, !dbg !21
  %address.local420 = or disjoint i64 %address.local399, 19, !dbg !21
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local419, i64 %address.local420, !dbg !21
  store half %2265, ptr addrspace(1) %address.local421, align 2, !dbg !21
  %address.local422 = shl i64 %address.local394, 7, !dbg !21
  %address.local423 = add i64 %address.local422, 8192, !dbg !21
  %address.local424 = add nuw nsw i64 %2286, %address.local423, !dbg !21
  %address.local425 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local424, !dbg !21
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local425, i64 %address.local399, !dbg !21
  store half %2253, ptr addrspace(1) %address.local426, align 2, !dbg !21
  %address.local427 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local424, !dbg !21
  %address.local428 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local427, i64 %address.local402, !dbg !21
  store half %2252, ptr addrspace(1) %address.local428, align 2, !dbg !21
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local424, !dbg !21
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local429, i64 %address.local405, !dbg !21
  store half %2251, ptr addrspace(1) %address.local430, align 2, !dbg !21
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local424, !dbg !21
  %address.local432 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local431, i64 %address.local408, !dbg !21
  store half %2250, ptr addrspace(1) %address.local432, align 2, !dbg !21
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local424, !dbg !21
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local411, !dbg !21
  store half %2238, ptr addrspace(1) %address.local434, align 2, !dbg !21
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local424, !dbg !21
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local435, i64 %address.local414, !dbg !21
  store half %2237, ptr addrspace(1) %address.local436, align 2, !dbg !21
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local424, !dbg !21
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local437, i64 %address.local417, !dbg !21
  store half %2236, ptr addrspace(1) %address.local438, align 2, !dbg !21
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local424, !dbg !21
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local439, i64 %address.local420, !dbg !21
  store half %2235, ptr addrspace(1) %address.local440, align 2, !dbg !21
  br label %2287, !dbg !21

2287:                                             ; preds = %2223, %1772
  br i1 true, label %2288, label %2326, !dbg !21

2288:                                             ; preds = %2287
  %2289 = insertelement <4 x float> poison, float %2193, i64 0, !dbg !19
  %2290 = insertelement <4 x float> %2289, float %2199, i64 1, !dbg !19
  %2291 = insertelement <4 x float> %2290, float %2194, i64 2, !dbg !19
  %2292 = insertelement <4 x float> %2291, float %2200, i64 3, !dbg !19
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2116, !dbg !19
  %normalized73 = fmul <4 x float> %2292, %row.reciprocal72, !dbg !19
  %2293 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !20
  %2294 = extractelement <4 x half> %2293, i64 3, !dbg !20
  %2295 = extractelement <4 x half> %2293, i64 2, !dbg !20
  %2296 = extractelement <4 x half> %2293, i64 1, !dbg !20
  %2297 = extractelement <4 x half> %2293, i64 0, !dbg !20
  %2298 = insertelement <4 x float> poison, float %2084, i64 0, !dbg !19
  %2299 = insertelement <4 x float> %2298, float %2090, i64 1, !dbg !19
  %2300 = insertelement <4 x float> %2299, float %2085, i64 2, !dbg !19
  %2301 = insertelement <4 x float> %2300, float %2091, i64 3, !dbg !19
  %normalized74 = fmul <4 x float> %2301, %row.reciprocal72, !dbg !19
  %2302 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %2303 = extractelement <4 x half> %2302, i64 3, !dbg !20
  %2304 = extractelement <4 x half> %2302, i64 2, !dbg !20
  %2305 = extractelement <4 x half> %2302, i64 1, !dbg !20
  %2306 = extractelement <4 x half> %2302, i64 0, !dbg !20
  %2307 = insertelement <4 x float> poison, float %1975, i64 0, !dbg !19
  %2308 = insertelement <4 x float> %2307, float %1981, i64 1, !dbg !19
  %2309 = insertelement <4 x float> %2308, float %1976, i64 2, !dbg !19
  %2310 = insertelement <4 x float> %2309, float %1982, i64 3, !dbg !19
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1898, !dbg !19
  %normalized76 = fmul <4 x float> %2310, %row.reciprocal75, !dbg !19
  %2311 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !20
  %2312 = extractelement <4 x half> %2311, i64 3, !dbg !20
  %2313 = extractelement <4 x half> %2311, i64 2, !dbg !20
  %2314 = extractelement <4 x half> %2311, i64 1, !dbg !20
  %2315 = extractelement <4 x half> %2311, i64 0, !dbg !20
  %2316 = insertelement <4 x float> poison, float %1866, i64 0, !dbg !19
  %2317 = insertelement <4 x float> %2316, float %1872, i64 1, !dbg !19
  %2318 = insertelement <4 x float> %2317, float %1867, i64 2, !dbg !19
  %2319 = insertelement <4 x float> %2318, float %1873, i64 3, !dbg !19
  %normalized77 = fmul <4 x float> %2319, %row.reciprocal75, !dbg !19
  %2320 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !20
  %2321 = extractelement <4 x half> %2320, i64 3, !dbg !20
  %2322 = extractelement <4 x half> %2320, i64 2, !dbg !20
  %2323 = extractelement <4 x half> %2320, i64 1, !dbg !20
  %2324 = extractelement <4 x half> %2320, i64 0, !dbg !20
  %2325 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread441 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local442 = zext nneg i32 %address.thread441 to i64, !dbg !12
  %address.local443 = add i64 %28, %address.local442, !dbg !21
  %address.local444 = lshr i64 %address.local442, 6, !dbg !12
  %address.local445 = shl nuw nsw i64 %address.local444, 4, !dbg !21
  %address.local446 = add i64 %address.local443, %address.local445, !dbg !21
  %address.local447 = and i64 %address.local442, 240, !dbg !21
  %address.local448 = sub i64 %address.local446, %address.local447, !dbg !21
  %address.local449 = add i64 %address.local448, %29, !dbg !21
  %address.local450 = shl nuw nsw i64 %address.local449, 7, !dbg !21
  %address.local451 = add nuw nsw i64 %2325, %address.local450, !dbg !21
  %address.local452 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local451, !dbg !21
  %address.local453 = lshr i64 %address.local442, 2, !dbg !12
  %address.local454 = and i64 %address.local453, 12, !dbg !12
  %address.local455 = or disjoint i64 %address.local454, 32, !dbg !21
  %address.local456 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local452, i64 %address.local455, !dbg !21
  store half %2324, ptr addrspace(1) %address.local456, align 2, !dbg !21
  %address.local457 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local451, !dbg !21
  %address.local458 = or disjoint i64 %address.local454, 33, !dbg !21
  %address.local459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local457, i64 %address.local458, !dbg !21
  store half %2323, ptr addrspace(1) %address.local459, align 2, !dbg !21
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local451, !dbg !21
  %address.local461 = or disjoint i64 %address.local454, 34, !dbg !21
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local461, !dbg !21
  store half %2322, ptr addrspace(1) %address.local462, align 2, !dbg !21
  %address.local463 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local451, !dbg !21
  %address.local464 = or disjoint i64 %address.local454, 35, !dbg !21
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local463, i64 %address.local464, !dbg !21
  store half %2321, ptr addrspace(1) %address.local465, align 2, !dbg !21
  %address.local466 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local451, !dbg !21
  %address.local467 = or disjoint i64 %address.local454, 48, !dbg !21
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local466, i64 %address.local467, !dbg !21
  store half %2315, ptr addrspace(1) %address.local468, align 2, !dbg !21
  %address.local469 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local451, !dbg !21
  %address.local470 = or disjoint i64 %address.local454, 49, !dbg !21
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local469, i64 %address.local470, !dbg !21
  store half %2314, ptr addrspace(1) %address.local471, align 2, !dbg !21
  %address.local472 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local451, !dbg !21
  %address.local473 = or disjoint i64 %address.local454, 50, !dbg !21
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local472, i64 %address.local473, !dbg !21
  store half %2313, ptr addrspace(1) %address.local474, align 2, !dbg !21
  %address.local475 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local451, !dbg !21
  %address.local476 = or i64 %address.local453, 51, !dbg !21
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local475, i64 %address.local476, !dbg !21
  store half %2312, ptr addrspace(1) %address.local477, align 2, !dbg !21
  %address.local478 = shl i64 %address.local449, 7, !dbg !21
  %address.local479 = add i64 %address.local478, 8192, !dbg !21
  %address.local480 = add nuw nsw i64 %2325, %address.local479, !dbg !21
  %address.local481 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local480, !dbg !21
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local481, i64 %address.local455, !dbg !21
  store half %2306, ptr addrspace(1) %address.local482, align 2, !dbg !21
  %address.local483 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local480, !dbg !21
  %address.local484 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local483, i64 %address.local458, !dbg !21
  store half %2305, ptr addrspace(1) %address.local484, align 2, !dbg !21
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local480, !dbg !21
  %address.local486 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local485, i64 %address.local461, !dbg !21
  store half %2304, ptr addrspace(1) %address.local486, align 2, !dbg !21
  %address.local487 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local480, !dbg !21
  %address.local488 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local487, i64 %address.local464, !dbg !21
  store half %2303, ptr addrspace(1) %address.local488, align 2, !dbg !21
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local480, !dbg !21
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local467, !dbg !21
  store half %2297, ptr addrspace(1) %address.local490, align 2, !dbg !21
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local480, !dbg !21
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local491, i64 %address.local470, !dbg !21
  store half %2296, ptr addrspace(1) %address.local492, align 2, !dbg !21
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local480, !dbg !21
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local493, i64 %address.local473, !dbg !21
  store half %2295, ptr addrspace(1) %address.local494, align 2, !dbg !21
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local480, !dbg !21
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local495, i64 %address.local476, !dbg !21
  store half %2294, ptr addrspace(1) %address.local496, align 2, !dbg !21
  br label %2326, !dbg !21

2326:                                             ; preds = %2288, %2287
  br i1 true, label %2327, label %2365, !dbg !21

2327:                                             ; preds = %2326
  %2328 = insertelement <4 x float> poison, float %2165, i64 0, !dbg !19
  %2329 = insertelement <4 x float> %2328, float %2171, i64 1, !dbg !19
  %2330 = insertelement <4 x float> %2329, float %2166, i64 2, !dbg !19
  %2331 = insertelement <4 x float> %2330, float %2172, i64 3, !dbg !19
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2116, !dbg !19
  %normalized79 = fmul <4 x float> %2331, %row.reciprocal78, !dbg !19
  %2332 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !20
  %2333 = extractelement <4 x half> %2332, i64 3, !dbg !20
  %2334 = extractelement <4 x half> %2332, i64 2, !dbg !20
  %2335 = extractelement <4 x half> %2332, i64 1, !dbg !20
  %2336 = extractelement <4 x half> %2332, i64 0, !dbg !20
  %2337 = insertelement <4 x float> poison, float %2055, i64 0, !dbg !19
  %2338 = insertelement <4 x float> %2337, float %2061, i64 1, !dbg !19
  %2339 = insertelement <4 x float> %2338, float %2056, i64 2, !dbg !19
  %2340 = insertelement <4 x float> %2339, float %2062, i64 3, !dbg !19
  %normalized80 = fmul <4 x float> %2340, %row.reciprocal78, !dbg !19
  %2341 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !20
  %2342 = extractelement <4 x half> %2341, i64 3, !dbg !20
  %2343 = extractelement <4 x half> %2341, i64 2, !dbg !20
  %2344 = extractelement <4 x half> %2341, i64 1, !dbg !20
  %2345 = extractelement <4 x half> %2341, i64 0, !dbg !20
  %2346 = insertelement <4 x float> poison, float %1947, i64 0, !dbg !19
  %2347 = insertelement <4 x float> %2346, float %1953, i64 1, !dbg !19
  %2348 = insertelement <4 x float> %2347, float %1948, i64 2, !dbg !19
  %2349 = insertelement <4 x float> %2348, float %1954, i64 3, !dbg !19
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1898, !dbg !19
  %normalized82 = fmul <4 x float> %2349, %row.reciprocal81, !dbg !19
  %2350 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !20
  %2351 = extractelement <4 x half> %2350, i64 3, !dbg !20
  %2352 = extractelement <4 x half> %2350, i64 2, !dbg !20
  %2353 = extractelement <4 x half> %2350, i64 1, !dbg !20
  %2354 = extractelement <4 x half> %2350, i64 0, !dbg !20
  %2355 = insertelement <4 x float> poison, float %1837, i64 0, !dbg !19
  %2356 = insertelement <4 x float> %2355, float %1843, i64 1, !dbg !19
  %2357 = insertelement <4 x float> %2356, float %1838, i64 2, !dbg !19
  %2358 = insertelement <4 x float> %2357, float %1844, i64 3, !dbg !19
  %normalized83 = fmul <4 x float> %2358, %row.reciprocal81, !dbg !19
  %2359 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !20
  %2360 = extractelement <4 x half> %2359, i64 3, !dbg !20
  %2361 = extractelement <4 x half> %2359, i64 2, !dbg !20
  %2362 = extractelement <4 x half> %2359, i64 1, !dbg !20
  %2363 = extractelement <4 x half> %2359, i64 0, !dbg !20
  %2364 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread497 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local498 = zext nneg i32 %address.thread497 to i64, !dbg !12
  %address.local499 = add i64 %28, %address.local498, !dbg !21
  %address.local500 = lshr i64 %address.local498, 6, !dbg !12
  %address.local501 = shl nuw nsw i64 %address.local500, 4, !dbg !21
  %address.local502 = add i64 %address.local499, %address.local501, !dbg !21
  %address.local503 = and i64 %address.local498, 240, !dbg !21
  %address.local504 = sub i64 %address.local502, %address.local503, !dbg !21
  %address.local505 = add i64 %address.local504, %29, !dbg !21
  %address.local506 = shl nuw nsw i64 %address.local505, 7, !dbg !21
  %address.local507 = add nuw nsw i64 %2364, %address.local506, !dbg !21
  %address.local508 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local507, !dbg !21
  %address.local509 = lshr i64 %address.local498, 2, !dbg !12
  %address.local510 = and i64 %address.local509, 12, !dbg !12
  %address.local511 = or disjoint i64 %address.local510, 64, !dbg !21
  %address.local512 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local508, i64 %address.local511, !dbg !21
  store half %2363, ptr addrspace(1) %address.local512, align 2, !dbg !21
  %address.local513 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local507, !dbg !21
  %address.local514 = or disjoint i64 %address.local510, 65, !dbg !21
  %address.local515 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local513, i64 %address.local514, !dbg !21
  store half %2362, ptr addrspace(1) %address.local515, align 2, !dbg !21
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local507, !dbg !21
  %address.local517 = or disjoint i64 %address.local510, 66, !dbg !21
  %address.local518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local516, i64 %address.local517, !dbg !21
  store half %2361, ptr addrspace(1) %address.local518, align 2, !dbg !21
  %address.local519 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local507, !dbg !21
  %address.local520 = or disjoint i64 %address.local510, 67, !dbg !21
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local519, i64 %address.local520, !dbg !21
  store half %2360, ptr addrspace(1) %address.local521, align 2, !dbg !21
  %address.local522 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local507, !dbg !21
  %address.local523 = or disjoint i64 %address.local510, 80, !dbg !21
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local522, i64 %address.local523, !dbg !21
  store half %2354, ptr addrspace(1) %address.local524, align 2, !dbg !21
  %address.local525 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local507, !dbg !21
  %address.local526 = or disjoint i64 %address.local510, 81, !dbg !21
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local525, i64 %address.local526, !dbg !21
  store half %2353, ptr addrspace(1) %address.local527, align 2, !dbg !21
  %address.local528 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local507, !dbg !21
  %address.local529 = or disjoint i64 %address.local510, 82, !dbg !21
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local528, i64 %address.local529, !dbg !21
  store half %2352, ptr addrspace(1) %address.local530, align 2, !dbg !21
  %address.local531 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local507, !dbg !21
  %address.local532 = or disjoint i64 %address.local510, 83, !dbg !21
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local531, i64 %address.local532, !dbg !21
  store half %2351, ptr addrspace(1) %address.local533, align 2, !dbg !21
  %address.local534 = shl i64 %address.local505, 7, !dbg !21
  %address.local535 = add i64 %address.local534, 8192, !dbg !21
  %address.local536 = add nuw nsw i64 %2364, %address.local535, !dbg !21
  %address.local537 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local536, !dbg !21
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local537, i64 %address.local511, !dbg !21
  store half %2345, ptr addrspace(1) %address.local538, align 2, !dbg !21
  %address.local539 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local536, !dbg !21
  %address.local540 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local539, i64 %address.local514, !dbg !21
  store half %2344, ptr addrspace(1) %address.local540, align 2, !dbg !21
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local536, !dbg !21
  %address.local542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local541, i64 %address.local517, !dbg !21
  store half %2343, ptr addrspace(1) %address.local542, align 2, !dbg !21
  %address.local543 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local536, !dbg !21
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local543, i64 %address.local520, !dbg !21
  store half %2342, ptr addrspace(1) %address.local544, align 2, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local536, !dbg !21
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local523, !dbg !21
  store half %2336, ptr addrspace(1) %address.local546, align 2, !dbg !21
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local536, !dbg !21
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local547, i64 %address.local526, !dbg !21
  store half %2335, ptr addrspace(1) %address.local548, align 2, !dbg !21
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local536, !dbg !21
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local549, i64 %address.local529, !dbg !21
  store half %2334, ptr addrspace(1) %address.local550, align 2, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local536, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local551, i64 %address.local532, !dbg !21
  store half %2333, ptr addrspace(1) %address.local552, align 2, !dbg !21
  br label %2365, !dbg !21

2365:                                             ; preds = %2327, %2326
  br i1 true, label %2366, label %2404, !dbg !21

2366:                                             ; preds = %2365
  %2367 = insertelement <4 x float> poison, float %2137, i64 0, !dbg !19
  %2368 = insertelement <4 x float> %2367, float %2143, i64 1, !dbg !19
  %2369 = insertelement <4 x float> %2368, float %2138, i64 2, !dbg !19
  %2370 = insertelement <4 x float> %2369, float %2144, i64 3, !dbg !19
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2116, !dbg !19
  %normalized85 = fmul <4 x float> %2370, %row.reciprocal84, !dbg !19
  %2371 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !20
  %2372 = extractelement <4 x half> %2371, i64 3, !dbg !20
  %2373 = extractelement <4 x half> %2371, i64 2, !dbg !20
  %2374 = extractelement <4 x half> %2371, i64 1, !dbg !20
  %2375 = extractelement <4 x half> %2371, i64 0, !dbg !20
  %2376 = insertelement <4 x float> poison, float %2026, i64 0, !dbg !19
  %2377 = insertelement <4 x float> %2376, float %2032, i64 1, !dbg !19
  %2378 = insertelement <4 x float> %2377, float %2027, i64 2, !dbg !19
  %2379 = insertelement <4 x float> %2378, float %2033, i64 3, !dbg !19
  %normalized86 = fmul <4 x float> %2379, %row.reciprocal84, !dbg !19
  %2380 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !20
  %2381 = extractelement <4 x half> %2380, i64 3, !dbg !20
  %2382 = extractelement <4 x half> %2380, i64 2, !dbg !20
  %2383 = extractelement <4 x half> %2380, i64 1, !dbg !20
  %2384 = extractelement <4 x half> %2380, i64 0, !dbg !20
  %2385 = insertelement <4 x float> poison, float %1919, i64 0, !dbg !19
  %2386 = insertelement <4 x float> %2385, float %1925, i64 1, !dbg !19
  %2387 = insertelement <4 x float> %2386, float %1920, i64 2, !dbg !19
  %2388 = insertelement <4 x float> %2387, float %1926, i64 3, !dbg !19
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1898, !dbg !19
  %normalized88 = fmul <4 x float> %2388, %row.reciprocal87, !dbg !19
  %2389 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !20
  %2390 = extractelement <4 x half> %2389, i64 3, !dbg !20
  %2391 = extractelement <4 x half> %2389, i64 2, !dbg !20
  %2392 = extractelement <4 x half> %2389, i64 1, !dbg !20
  %2393 = extractelement <4 x half> %2389, i64 0, !dbg !20
  %2394 = insertelement <4 x float> poison, float %1808, i64 0, !dbg !19
  %2395 = insertelement <4 x float> %2394, float %1814, i64 1, !dbg !19
  %2396 = insertelement <4 x float> %2395, float %1809, i64 2, !dbg !19
  %2397 = insertelement <4 x float> %2396, float %1815, i64 3, !dbg !19
  %normalized89 = fmul <4 x float> %2397, %row.reciprocal87, !dbg !19
  %2398 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !20
  %2399 = extractelement <4 x half> %2398, i64 3, !dbg !20
  %2400 = extractelement <4 x half> %2398, i64 2, !dbg !20
  %2401 = extractelement <4 x half> %2398, i64 1, !dbg !20
  %2402 = extractelement <4 x half> %2398, i64 0, !dbg !20
  %2403 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread553 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local554 = zext nneg i32 %address.thread553 to i64, !dbg !12
  %address.local555 = add i64 %28, %address.local554, !dbg !21
  %address.local556 = lshr i64 %address.local554, 6, !dbg !12
  %address.local557 = shl nuw nsw i64 %address.local556, 4, !dbg !21
  %address.local558 = add i64 %address.local555, %address.local557, !dbg !21
  %address.local559 = and i64 %address.local554, 240, !dbg !21
  %address.local560 = sub i64 %address.local558, %address.local559, !dbg !21
  %address.local561 = add i64 %address.local560, %29, !dbg !21
  %address.local562 = shl nuw nsw i64 %address.local561, 7, !dbg !21
  %address.local563 = add nuw nsw i64 %2403, %address.local562, !dbg !21
  %address.local564 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local563, !dbg !21
  %address.local565 = lshr i64 %address.local554, 2, !dbg !12
  %address.local566 = and i64 %address.local565, 12, !dbg !12
  %address.local567 = or disjoint i64 %address.local566, 96, !dbg !21
  %address.local568 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local564, i64 %address.local567, !dbg !21
  store half %2402, ptr addrspace(1) %address.local568, align 2, !dbg !21
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local563, !dbg !21
  %address.local570 = or disjoint i64 %address.local566, 97, !dbg !21
  %address.local571 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local569, i64 %address.local570, !dbg !21
  store half %2401, ptr addrspace(1) %address.local571, align 2, !dbg !21
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local563, !dbg !21
  %address.local573 = or disjoint i64 %address.local566, 98, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local572, i64 %address.local573, !dbg !21
  store half %2400, ptr addrspace(1) %address.local574, align 2, !dbg !21
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local563, !dbg !21
  %address.local576 = or disjoint i64 %address.local566, 99, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local576, !dbg !21
  store half %2399, ptr addrspace(1) %address.local577, align 2, !dbg !21
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local563, !dbg !21
  %address.local579 = or disjoint i64 %address.local566, 112, !dbg !21
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local578, i64 %address.local579, !dbg !21
  store half %2393, ptr addrspace(1) %address.local580, align 2, !dbg !21
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local563, !dbg !21
  %address.local582 = or disjoint i64 %address.local566, 113, !dbg !21
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local582, !dbg !21
  store half %2392, ptr addrspace(1) %address.local583, align 2, !dbg !21
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local563, !dbg !21
  %address.local585 = or disjoint i64 %address.local566, 114, !dbg !21
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local584, i64 %address.local585, !dbg !21
  store half %2391, ptr addrspace(1) %address.local586, align 2, !dbg !21
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local563, !dbg !21
  %address.local588 = or i64 %address.local565, 115, !dbg !21
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local587, i64 %address.local588, !dbg !21
  store half %2390, ptr addrspace(1) %address.local589, align 2, !dbg !21
  %address.local590 = shl i64 %address.local561, 7, !dbg !21
  %address.local591 = add i64 %address.local590, 8192, !dbg !21
  %address.local592 = add nuw nsw i64 %2403, %address.local591, !dbg !21
  %address.local593 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local592, !dbg !21
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local593, i64 %address.local567, !dbg !21
  store half %2384, ptr addrspace(1) %address.local594, align 2, !dbg !21
  %address.local595 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local592, !dbg !21
  %address.local596 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local595, i64 %address.local570, !dbg !21
  store half %2383, ptr addrspace(1) %address.local596, align 2, !dbg !21
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local592, !dbg !21
  %address.local598 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local597, i64 %address.local573, !dbg !21
  store half %2382, ptr addrspace(1) %address.local598, align 2, !dbg !21
  %address.local599 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local592, !dbg !21
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local599, i64 %address.local576, !dbg !21
  store half %2381, ptr addrspace(1) %address.local600, align 2, !dbg !21
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local592, !dbg !21
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local601, i64 %address.local579, !dbg !21
  store half %2375, ptr addrspace(1) %address.local602, align 2, !dbg !21
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local592, !dbg !21
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local603, i64 %address.local582, !dbg !21
  store half %2374, ptr addrspace(1) %address.local604, align 2, !dbg !21
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local592, !dbg !21
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local605, i64 %address.local585, !dbg !21
  store half %2373, ptr addrspace(1) %address.local606, align 2, !dbg !21
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local592, !dbg !21
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local607, i64 %address.local588, !dbg !21
  store half %2372, ptr addrspace(1) %address.local608, align 2, !dbg !21
  br label %2404, !dbg !21

2404:                                             ; preds = %2366, %2365
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

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
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0006-test_input_1x32x4096x128-128-128-32-online/group8-reverse")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!9 = !DILocation(line: 48, column: 17, scope: !4)
!11 = !DILocation(line: 70, column: 22, scope: !4)
!12 = !DILocation(line: 72, column: 24, scope: !4)
!13 = !DILocation(line: 77, column: 19, scope: !4)
!14 = !DILocation(line: 102, column: 23, scope: !4)
!15 = !DILocation(line: 0, scope: !4)
!16 = !DILocation(line: 71, column: 22, scope: !4)
!17 = !DILocation(line: 73, column: 26, scope: !4)
!18 = !DILocation(line: 101, column: 20, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
!23 = !DILocation(line: 79, column: 24, scope: !4)
!24 = !DILocation(line: 80, column: 23, scope: !4)
!25 = !DILocation(line: 81, column: 26, scope: !4)
!26 = !DILocation(line: 88, column: 21, scope: !4)
!27 = !DILocation(line: 89, column: 20, scope: !4)
!28 = !DILocation(line: 90, column: 22, scope: !4)
!29 = !DILocation(line: 91, column: 23, scope: !4)
!30 = !DILocation(line: 92, column: 20, scope: !4)
!31 = !DILocation(line: 93, column: 22, scope: !4)
!32 = !DILocation(line: 94, column: 22, scope: !4)
!33 = !DILocation(line: 95, column: 20, scope: !4)
!34 = !DILocation(line: 96, column: 16, scope: !4)
!35 = !DILocation(line: 103, column: 22, scope: !4)
!37 = !{i32 256, i32 1, i32 1}
!38 = !{i32 2, !"Debug Info Version", i32 3}
!39 = !{i32 0, i32 256}
