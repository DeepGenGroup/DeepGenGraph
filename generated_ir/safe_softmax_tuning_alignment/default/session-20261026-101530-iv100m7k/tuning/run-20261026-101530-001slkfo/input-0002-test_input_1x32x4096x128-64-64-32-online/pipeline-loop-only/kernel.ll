; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/pipeline-loop-only/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/pipeline-loop-only/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

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
  %.lobit = ashr i32 %9, 31, !dbg !12
  %12 = xor i32 %9, %.lobit, !dbg !12
  %13 = sdiv i32 %12, 64, !dbg !12
  %.lobit1 = ashr i32 %9, 31, !dbg !12
  %14 = xor i32 %13, %.lobit1, !dbg !12
  %15 = sext i32 %14 to i64, !dbg !12
  %16 = add nsw i64 %8, %15, !dbg !12
  %17 = and i64 %16, 31, !dbg !12
  %18 = shl nsw i64 %10, 6, !dbg !12
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
  %29 = or disjoint i64 %11, 63, !dbg !14
  %30 = add nsw i64 %11, 64, !dbg !14
  %31 = icmp uge i64 %30, %29, !dbg !14
  %.idx = shl nuw nsw i64 %17, 20, !dbg !12
  %32 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %33 = getelementptr [2 x i8], ptr addrspace(1) %32, i64 %21, !dbg !12
  %34 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %35 = getelementptr i8, ptr addrspace(1) %34, i64 64, !dbg !12
  %36 = shl nsw i64 %8, 13, !dbg !16
  %37 = shl nsw i64 %6, 18, !dbg !16
  %38 = shl nsw i64 %8, 7, !dbg !16
  %39 = shl nsw i64 %6, 12, !dbg !16
  %40 = add nsw i64 %38, %39, !dbg !16
  %41 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %42 = getelementptr i8, ptr addrspace(1) %41, i64 128, !dbg !12
  %43 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %25, !dbg !12
  %44 = getelementptr i8, ptr addrspace(1) %43, i64 192, !dbg !12
  br label %45, !dbg !7

45:                                               ; preds = %404, %4
  %46 = phi i64 [ %716, %404 ], [ 0, %4 ], !dbg !7
  %47 = phi <4 x float> [ %599, %404 ], [ zeroinitializer, %4 ], !dbg !7
  %48 = phi <4 x float> [ %607, %404 ], [ zeroinitializer, %4 ], !dbg !7
  %49 = phi <8 x float> [ %712, %404 ], [ zeroinitializer, %4 ], !dbg !7
  %50 = phi <1 x float> [ %563, %404 ], [ zeroinitializer, %4 ], !dbg !7
  %51 = phi <1 x float> [ %466, %404 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %52 = phi <4 x float> [ %639, %404 ], [ zeroinitializer, %4 ], !dbg !7
  %53 = phi <4 x float> [ %647, %404 ], [ zeroinitializer, %4 ], !dbg !7
  %54 = phi <8 x float> [ %713, %404 ], [ zeroinitializer, %4 ], !dbg !7
  %55 = phi <4 x float> [ %679, %404 ], [ zeroinitializer, %4 ], !dbg !7
  %56 = phi <4 x float> [ %687, %404 ], [ zeroinitializer, %4 ], !dbg !7
  %57 = phi <8 x float> [ %714, %404 ], [ zeroinitializer, %4 ], !dbg !7
  %58 = phi <4 x float> [ %703, %404 ], [ zeroinitializer, %4 ], !dbg !7
  %59 = phi <4 x float> [ %711, %404 ], [ zeroinitializer, %4 ], !dbg !7
  %60 = phi <8 x float> [ %715, %404 ], [ zeroinitializer, %4 ], !dbg !7
  %.not = icmp sgt i64 %46, %10, !dbg !7
  br i1 %.not, label %717, label %61, !dbg !7

61:                                               ; preds = %45
  %62 = shl nsw i64 %46, 6, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !12, !range !40, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local61 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local62 = and i64 %address.local61, 1536, !dbg !12
  %address.local63 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local62, !dbg !12
  %address.local64 = ptrtoint ptr addrspace(3) %address.local63 to i64, !dbg !12
  %address.local65 = trunc i64 %address.local64 to i32, !dbg !12
  %63 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local65), !dbg !12
  %64 = zext i32 %63 to i64, !dbg !12
  %65 = inttoptr i64 %64 to ptr addrspace(3), !dbg !12
  %address.local66 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local67 = and i64 %address.local66, 8064, !dbg !12
  %address.local68 = lshr i64 %address.local, 3, !dbg !12
  %address.local69 = and i64 %address.local68, 24, !dbg !12
  %address.local70 = or disjoint i64 %address.local67, %address.local69, !dbg !12
  %address.local71 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local70, !dbg !12
  %66 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %67 = ptrtoint ptr addrspace(1) %address.local71 to i64, !dbg !12
  %68 = sub i64 %67, %66, !dbg !12
  %dma.buffer.offset517 = trunc i64 %68 to i32, !dbg !12
  %69 = trunc i64 %66 to i32, !dbg !12
  %70 = insertelement <4 x i32> poison, i32 %69, i64 0, !dbg !12
  %71 = lshr i64 %66, 32, !dbg !12
  %72 = trunc i64 %71 to i32, !dbg !12
  %73 = insertelement <4 x i32> %70, i32 %72, i64 1, !dbg !12
  %74 = insertelement <4 x i32> %73, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc518 = insertelement <4 x i32> %74, i32 131072, i64 3, !dbg !12
  %75 = ptrtoint ptr addrspace(3) %65 to i32, !dbg !12
  %dma.buffer.lds519 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %75), !dbg !12
  %76 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds519, i32 %dma.buffer.offset517, <4 x i32> %dma.buffer.rsrc518) #10, !dbg !12
  %77 = lshr i64 %46, 13, !dbg !16
  %78 = add nsw i64 %77, %8, !dbg !16
  %79 = lshr i64 %46, 6, !dbg !16
  %80 = and i64 %79, 127, !dbg !16
  %81 = shl i64 %78, 19, !dbg !16
  %82 = and i64 %81, 16252928, !dbg !16
  %83 = shl i64 %79, 19, !dbg !16
  %84 = shl i64 %46, 13, !dbg !16
  %85 = sub i64 %84, %83, !dbg !16
  %86 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %82, !dbg !16
  %87 = getelementptr [2 x i8], ptr addrspace(1) %86, i64 %80, !dbg !16
  %88 = getelementptr [2 x i8], ptr addrspace(1) %87, i64 %85, !dbg !16
  %address.local72 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local62, !dbg !16
  %address.local73 = ptrtoint ptr addrspace(3) %address.local72 to i64, !dbg !16
  %address.local74 = trunc i64 %address.local73 to i32, !dbg !16
  %89 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local74), !dbg !16
  %90 = zext i32 %89 to i64, !dbg !16
  %91 = inttoptr i64 %90 to ptr addrspace(3), !dbg !16
  %address.local75 = getelementptr [2 x i8], ptr addrspace(1) %88, i64 %address.local70, !dbg !16
  %92 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %93 = ptrtoint ptr addrspace(1) %address.local75 to i64, !dbg !16
  %94 = sub i64 %93, %92, !dbg !16
  %dma.buffer.offset514 = trunc i64 %94 to i32, !dbg !16
  %95 = trunc i64 %92 to i32, !dbg !16
  %96 = insertelement <4 x i32> poison, i32 %95, i64 0, !dbg !16
  %97 = lshr i64 %92, 32, !dbg !16
  %98 = trunc i64 %97 to i32, !dbg !16
  %99 = insertelement <4 x i32> %96, i32 %98, i64 1, !dbg !16
  %100 = insertelement <4 x i32> %99, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc515 = insertelement <4 x i32> %100, i32 131072, i64 3, !dbg !16
  %101 = ptrtoint ptr addrspace(3) %91 to i32, !dbg !16
  %dma.buffer.lds516 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %101), !dbg !16
  %102 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds516, i32 %dma.buffer.offset514, <4 x i32> %dma.buffer.rsrc515) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %address.local76 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local62, !dbg !12
  %address.local77 = ptrtoint ptr addrspace(3) %address.local76 to i64, !dbg !12
  %address.local78 = trunc i64 %address.local77 to i32, !dbg !12
  %103 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local78), !dbg !12
  %104 = zext i32 %103 to i64, !dbg !12
  %105 = inttoptr i64 %104 to ptr addrspace(3), !dbg !12
  %address.local79 = getelementptr [2 x i8], ptr addrspace(1) %35, i64 %address.local70, !dbg !12
  %106 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %107 = ptrtoint ptr addrspace(1) %address.local79 to i64, !dbg !12
  %108 = sub i64 %107, %106, !dbg !12
  %dma.buffer.offset511 = trunc i64 %108 to i32, !dbg !12
  %109 = trunc i64 %106 to i32, !dbg !12
  %110 = insertelement <4 x i32> poison, i32 %109, i64 0, !dbg !12
  %111 = lshr i64 %106, 32, !dbg !12
  %112 = trunc i64 %111 to i32, !dbg !12
  %113 = insertelement <4 x i32> %110, i32 %112, i64 1, !dbg !12
  %114 = insertelement <4 x i32> %113, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc512 = insertelement <4 x i32> %114, i32 131072, i64 3, !dbg !12
  %115 = ptrtoint ptr addrspace(3) %105 to i32, !dbg !12
  %dma.buffer.lds513 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %115), !dbg !12
  %116 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds513, i32 %dma.buffer.offset511, <4 x i32> %dma.buffer.rsrc512) #10, !dbg !12
  %117 = add i64 %46, %36, !dbg !16
  %118 = add i64 %117, %37, !dbg !16
  %119 = add i64 %118, 2048, !dbg !16
  %120 = icmp slt i64 %119, 0, !dbg !16
  %121 = sub i64 -2049, %118, !dbg !16
  %122 = select i1 %120, i64 %121, i64 %119, !dbg !16
  %123 = sdiv i64 %122, 8192, !dbg !16
  %.lobit4 = ashr i64 %119, 63, !dbg !16
  %124 = xor i64 %123, %.lobit4, !dbg !16
  %125 = add nsw i64 %79, %40, !dbg !16
  %126 = add nsw i64 %125, 32, !dbg !16
  %127 = icmp slt i64 %125, -32, !dbg !16
  %128 = sub nsw i64 -33, %125, !dbg !16
  %129 = select i1 %127, i64 %128, i64 %126, !dbg !16
  %130 = sdiv i64 %129, 128, !dbg !16
  %131 = sext i1 %127 to i64, !dbg !16
  %132 = xor i64 %130, %131, !dbg !16
  %133 = shl nsw i64 %132, 7, !dbg !16
  %134 = sub i64 %40, %133, !dbg !16
  %135 = shl i64 %124, 19, !dbg !16
  %136 = and i64 %135, 16252928, !dbg !16
  %137 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %136, !dbg !16
  %138 = getelementptr [2 x i8], ptr addrspace(1) %137, i64 %79, !dbg !16
  %139 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %134, !dbg !16
  %140 = getelementptr i8, ptr addrspace(1) %139, i64 64, !dbg !16
  %141 = getelementptr [2 x i8], ptr addrspace(1) %140, i64 %85, !dbg !16
  %address.local80 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 12288), i64 %address.local62, !dbg !16
  %address.local81 = ptrtoint ptr addrspace(3) %address.local80 to i64, !dbg !16
  %address.local82 = trunc i64 %address.local81 to i32, !dbg !16
  %142 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local82), !dbg !16
  %143 = zext i32 %142 to i64, !dbg !16
  %144 = inttoptr i64 %143 to ptr addrspace(3), !dbg !16
  %address.local83 = getelementptr [2 x i8], ptr addrspace(1) %141, i64 %address.local70, !dbg !16
  %145 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %146 = ptrtoint ptr addrspace(1) %address.local83 to i64, !dbg !16
  %147 = sub i64 %146, %145, !dbg !16
  %dma.buffer.offset508 = trunc i64 %147 to i32, !dbg !16
  %148 = trunc i64 %145 to i32, !dbg !16
  %149 = insertelement <4 x i32> poison, i32 %148, i64 0, !dbg !16
  %150 = lshr i64 %145, 32, !dbg !16
  %151 = trunc i64 %150 to i32, !dbg !16
  %152 = insertelement <4 x i32> %149, i32 %151, i64 1, !dbg !16
  %153 = insertelement <4 x i32> %152, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc509 = insertelement <4 x i32> %153, i32 131072, i64 3, !dbg !16
  %154 = ptrtoint ptr addrspace(3) %144 to i32, !dbg !16
  %dma.buffer.lds510 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %154), !dbg !16
  %155 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds510, i32 %dma.buffer.offset508, <4 x i32> %dma.buffer.rsrc509) #10, !dbg !16
  %address.local84 = lshr i32 %address.thread, 2, !dbg !13
  %address.local85 = and i32 %address.local84, 4, !dbg !13
  %address.local86 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local87 = and i32 %address.local86, 384, !dbg !13
  %address.local88 = or disjoint i32 %address.local85, %address.local87, !dbg !13
  %address.local89 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local90 = and i32 %address.local89, 120, !dbg !13
  %address.local91 = or disjoint i32 %address.local88, %address.local90, !dbg !13
  %address.local92 = shl nuw nsw i32 %address.thread, 4, !dbg !13
  %address.local93 = and i32 %address.local92, 512, !dbg !13
  %address.local94 = or disjoint i32 %address.local91, %address.local93, !dbg !13
  %address.local95 = zext nneg i32 %address.local94 to i64, !dbg !13
  %address.local96 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local95, !dbg !13
  %156 = load <4 x half>, ptr addrspace(3) %address.local96, align 2, !dbg !13
  %address.local97 = lshr i32 %address.thread, 2, !dbg !13
  %address.local98 = and i32 %address.local97, 4, !dbg !13
  %address.local99 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local100 = and i32 %address.local99, 384, !dbg !13
  %address.local101 = or disjoint i32 %address.local98, %address.local100, !dbg !13
  %address.local102 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local103 = and i32 %address.local102, 120, !dbg !13
  %address.local104 = or disjoint i32 %address.local101, %address.local103, !dbg !13
  %address.local105 = shl nuw nsw i32 %address.thread, 4, !dbg !13
  %address.local106 = and i32 %address.local105, 512, !dbg !13
  %address.local107 = or disjoint i32 %address.local104, %address.local106, !dbg !13
  %address.local108 = or disjoint i32 %address.local107, 1024, !dbg !13
  %address.local109 = zext nneg i32 %address.local108 to i64, !dbg !13
  %address.local110 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local109, !dbg !13
  %157 = load <4 x half>, ptr addrspace(3) %address.local110, align 2, !dbg !13
  %address.local111 = lshr i32 %address.thread, 2, !dbg !17
  %address.local112 = and i32 %address.local111, 4, !dbg !17
  %address.local113 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local114 = and i32 %address.local113, 24, !dbg !17
  %address.local115 = or disjoint i32 %address.local112, %address.local114, !dbg !17
  %address.local116 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local117 = and i32 %address.local116, 512, !dbg !17
  %address.local118 = or disjoint i32 %address.local115, %address.local117, !dbg !17
  %address.local119 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local120 = and i32 %address.local119, 384, !dbg !17
  %address.local121 = or disjoint i32 %address.local118, %address.local120, !dbg !17
  %address.local122 = zext nneg i32 %address.local121 to i64, !dbg !17
  %address.local123 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local122, !dbg !17
  %158 = load <4 x half>, ptr addrspace(3) %address.local123, align 2, !dbg !17
  %159 = fmul <4 x half> %156, splat (half 0xH3015), !dbg !13
  %address.local124 = lshr i32 %address.thread, 2, !dbg !17
  %address.local125 = and i32 %address.local124, 4, !dbg !17
  %address.local126 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local127 = and i32 %address.local126, 24, !dbg !17
  %address.local128 = or disjoint i32 %address.local125, %address.local127, !dbg !17
  %address.local129 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local130 = and i32 %address.local129, 512, !dbg !17
  %address.local131 = or disjoint i32 %address.local128, %address.local130, !dbg !17
  %address.local132 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local133 = and i32 %address.local132, 384, !dbg !17
  %address.local134 = or disjoint i32 %address.local131, %address.local133, !dbg !17
  %address.local135 = or disjoint i32 %address.local134, 1024, !dbg !17
  %address.local136 = zext nneg i32 %address.local135 to i64, !dbg !17
  %address.local137 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local136, !dbg !17
  %160 = load <4 x half>, ptr addrspace(3) %address.local137, align 2, !dbg !17
  %161 = fmul <4 x half> %157, splat (half 0xH3015), !dbg !13
  %address.local138 = lshr i32 %address.thread, 2, !dbg !17
  %address.local139 = and i32 %address.local138, 4, !dbg !17
  %address.local140 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local141 = and i32 %address.local140, 24, !dbg !17
  %address.local142 = or disjoint i32 %address.local139, %address.local141, !dbg !17
  %address.local143 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local144 = and i32 %address.local143, 512, !dbg !17
  %address.local145 = or disjoint i32 %address.local142, %address.local144, !dbg !17
  %address.local146 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local147 = and i32 %address.local146, 384, !dbg !17
  %address.local148 = or disjoint i32 %address.local145, %address.local147, !dbg !17
  %address.local149 = or disjoint i32 %address.local148, 32, !dbg !17
  %address.local150 = zext nneg i32 %address.local149 to i64, !dbg !17
  %address.local151 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local150, !dbg !17
  %162 = load <4 x half>, ptr addrspace(3) %address.local151, align 2, !dbg !17
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %159, <4 x half> %158, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local152 = lshr i32 %address.thread, 2, !dbg !17
  %address.local153 = and i32 %address.local152, 4, !dbg !17
  %address.local154 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local155 = and i32 %address.local154, 24, !dbg !17
  %address.local156 = or disjoint i32 %address.local153, %address.local155, !dbg !17
  %address.local157 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local158 = and i32 %address.local157, 512, !dbg !17
  %address.local159 = or disjoint i32 %address.local156, %address.local158, !dbg !17
  %address.local160 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local161 = and i32 %address.local160, 384, !dbg !17
  %address.local162 = or disjoint i32 %address.local159, %address.local161, !dbg !17
  %address.local163 = or disjoint i32 %address.local162, 1056, !dbg !17
  %address.local164 = zext nneg i32 %address.local163 to i64, !dbg !17
  %address.local165 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local164, !dbg !17
  %164 = load <4 x half>, ptr addrspace(3) %address.local165, align 2, !dbg !17
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %161, <4 x half> %160, <4 x float> %163) #9, !dbg !17
  %address.local166 = lshr i32 %address.thread, 2, !dbg !17
  %address.local167 = and i32 %address.local166, 4, !dbg !17
  %address.local168 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local169 = and i32 %address.local168, 24, !dbg !17
  %address.local170 = or disjoint i32 %address.local167, %address.local169, !dbg !17
  %address.local171 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local172 = and i32 %address.local171, 512, !dbg !17
  %address.local173 = or disjoint i32 %address.local170, %address.local172, !dbg !17
  %address.local174 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local175 = and i32 %address.local174, 384, !dbg !17
  %address.local176 = or disjoint i32 %address.local173, %address.local175, !dbg !17
  %address.local177 = or disjoint i32 %address.local176, 64, !dbg !17
  %address.local178 = zext nneg i32 %address.local177 to i64, !dbg !17
  %address.local179 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local178, !dbg !17
  %166 = load <4 x half>, ptr addrspace(3) %address.local179, align 2, !dbg !17
  %address.local180 = lshr i32 %address.thread, 2, !dbg !17
  %address.local181 = and i32 %address.local180, 4, !dbg !17
  %address.local182 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local183 = and i32 %address.local182, 24, !dbg !17
  %address.local184 = or disjoint i32 %address.local181, %address.local183, !dbg !17
  %address.local185 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local186 = and i32 %address.local185, 512, !dbg !17
  %address.local187 = or disjoint i32 %address.local184, %address.local186, !dbg !17
  %address.local188 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local189 = and i32 %address.local188, 384, !dbg !17
  %address.local190 = or disjoint i32 %address.local187, %address.local189, !dbg !17
  %address.local191 = or disjoint i32 %address.local190, 1088, !dbg !17
  %address.local192 = zext nneg i32 %address.local191 to i64, !dbg !17
  %address.local193 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local192, !dbg !17
  %167 = load <4 x half>, ptr addrspace(3) %address.local193, align 2, !dbg !17
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %159, <4 x half> %162, <4 x float> zeroinitializer) #9, !dbg !17
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %161, <4 x half> %164, <4 x float> %168) #9, !dbg !17
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %159, <4 x half> %166, <4 x float> zeroinitializer) #9, !dbg !17
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %161, <4 x half> %167, <4 x float> %170) #9, !dbg !17
  %address.local194 = lshr i32 %address.thread, 2, !dbg !17
  %address.local195 = and i32 %address.local194, 4, !dbg !17
  %address.local196 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local197 = and i32 %address.local196, 24, !dbg !17
  %address.local198 = or disjoint i32 %address.local195, %address.local197, !dbg !17
  %address.local199 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local200 = and i32 %address.local199, 512, !dbg !17
  %address.local201 = or disjoint i32 %address.local198, %address.local200, !dbg !17
  %address.local202 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local203 = and i32 %address.local202, 384, !dbg !17
  %address.local204 = or disjoint i32 %address.local201, %address.local203, !dbg !17
  %address.local205 = or disjoint i32 %address.local204, 96, !dbg !17
  %address.local206 = zext nneg i32 %address.local205 to i64, !dbg !17
  %address.local207 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !17
  %172 = load <4 x half>, ptr addrspace(3) %address.local207, align 2, !dbg !17
  %address.local208 = lshr i32 %address.thread, 2, !dbg !17
  %address.local209 = and i32 %address.local208, 4, !dbg !17
  %address.local210 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local211 = and i32 %address.local210, 24, !dbg !17
  %address.local212 = or disjoint i32 %address.local209, %address.local211, !dbg !17
  %address.local213 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local214 = and i32 %address.local213, 512, !dbg !17
  %address.local215 = or disjoint i32 %address.local212, %address.local214, !dbg !17
  %address.local216 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local217 = and i32 %address.local216, 384, !dbg !17
  %address.local218 = or disjoint i32 %address.local215, %address.local217, !dbg !17
  %address.local219 = or disjoint i32 %address.local218, 1120, !dbg !17
  %address.local220 = zext nneg i32 %address.local219 to i64, !dbg !17
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local220, !dbg !17
  %173 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !17
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %159, <4 x half> %172, <4 x float> zeroinitializer) #9, !dbg !17
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %161, <4 x half> %173, <4 x float> %174) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %176 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local65), !dbg !12
  %177 = zext i32 %176 to i64, !dbg !12
  %178 = inttoptr i64 %177 to ptr addrspace(3), !dbg !12
  %address.local222 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %address.local70, !dbg !12
  %179 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %180 = ptrtoint ptr addrspace(1) %address.local222 to i64, !dbg !12
  %181 = sub i64 %180, %179, !dbg !12
  %dma.buffer.offset505 = trunc i64 %181 to i32, !dbg !12
  %182 = trunc i64 %179 to i32, !dbg !12
  %183 = insertelement <4 x i32> poison, i32 %182, i64 0, !dbg !12
  %184 = lshr i64 %179, 32, !dbg !12
  %185 = trunc i64 %184 to i32, !dbg !12
  %186 = insertelement <4 x i32> %183, i32 %185, i64 1, !dbg !12
  %187 = insertelement <4 x i32> %186, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc506 = insertelement <4 x i32> %187, i32 131072, i64 3, !dbg !12
  %188 = ptrtoint ptr addrspace(3) %178 to i32, !dbg !12
  %dma.buffer.lds507 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %188), !dbg !12
  %189 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds507, i32 %dma.buffer.offset505, <4 x i32> %dma.buffer.rsrc506) #10, !dbg !12
  %190 = add i64 %118, 4096, !dbg !16
  %191 = icmp slt i64 %190, 0, !dbg !16
  %192 = sub i64 -4097, %118, !dbg !16
  %193 = select i1 %191, i64 %192, i64 %190, !dbg !16
  %194 = sdiv i64 %193, 8192, !dbg !16
  %.lobit5 = ashr i64 %190, 63, !dbg !16
  %195 = xor i64 %194, %.lobit5, !dbg !16
  %196 = add nsw i64 %125, 64, !dbg !16
  %197 = icmp slt i64 %125, -64, !dbg !16
  %198 = sub nsw i64 -65, %125, !dbg !16
  %199 = select i1 %197, i64 %198, i64 %196, !dbg !16
  %200 = sdiv i64 %199, 128, !dbg !16
  %201 = sext i1 %197 to i64, !dbg !16
  %202 = xor i64 %200, %201, !dbg !16
  %203 = shl nsw i64 %202, 7, !dbg !16
  %204 = sub i64 %40, %203, !dbg !16
  %205 = shl i64 %195, 19, !dbg !16
  %206 = and i64 %205, 16252928, !dbg !16
  %207 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %206, !dbg !16
  %208 = getelementptr [2 x i8], ptr addrspace(1) %207, i64 %79, !dbg !16
  %209 = getelementptr [2 x i8], ptr addrspace(1) %208, i64 %204, !dbg !16
  %210 = getelementptr i8, ptr addrspace(1) %209, i64 128, !dbg !16
  %211 = getelementptr [2 x i8], ptr addrspace(1) %210, i64 %85, !dbg !16
  %212 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local74), !dbg !16
  %213 = zext i32 %212 to i64, !dbg !16
  %214 = inttoptr i64 %213 to ptr addrspace(3), !dbg !16
  %address.local223 = getelementptr [2 x i8], ptr addrspace(1) %211, i64 %address.local70, !dbg !16
  %215 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %216 = ptrtoint ptr addrspace(1) %address.local223 to i64, !dbg !16
  %217 = sub i64 %216, %215, !dbg !16
  %dma.buffer.offset502 = trunc i64 %217 to i32, !dbg !16
  %218 = trunc i64 %215 to i32, !dbg !16
  %219 = insertelement <4 x i32> poison, i32 %218, i64 0, !dbg !16
  %220 = lshr i64 %215, 32, !dbg !16
  %221 = trunc i64 %220 to i32, !dbg !16
  %222 = insertelement <4 x i32> %219, i32 %221, i64 1, !dbg !16
  %223 = insertelement <4 x i32> %222, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc503 = insertelement <4 x i32> %223, i32 131072, i64 3, !dbg !16
  %224 = ptrtoint ptr addrspace(3) %214 to i32, !dbg !16
  %dma.buffer.lds504 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %224), !dbg !16
  %225 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds504, i32 %dma.buffer.offset502, <4 x i32> %dma.buffer.rsrc503) #10, !dbg !16
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local95, !dbg !13
  %226 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !13
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local109, !dbg !13
  %227 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !13
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 12288), i64 %address.local122, !dbg !17
  %228 = load <4 x half>, ptr addrspace(3) %address.local226, align 2, !dbg !17
  %229 = fmul <4 x half> %226, splat (half 0xH3015), !dbg !13
  %230 = fmul <4 x half> %227, splat (half 0xH3015), !dbg !13
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %229, <4 x half> %228, <4 x float> %165) #9, !dbg !17
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 12288), i64 %address.local136, !dbg !17
  %232 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !17
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 12288), i64 %address.local150, !dbg !17
  %233 = load <4 x half>, ptr addrspace(3) %address.local228, align 2, !dbg !17
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %230, <4 x half> %232, <4 x float> %231) #9, !dbg !17
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %229, <4 x half> %233, <4 x float> %169) #9, !dbg !17
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 12288), i64 %address.local164, !dbg !17
  %236 = load <4 x half>, ptr addrspace(3) %address.local229, align 2, !dbg !17
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 12288), i64 %address.local178, !dbg !17
  %237 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !17
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %230, <4 x half> %236, <4 x float> %235) #9, !dbg !17
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %229, <4 x half> %237, <4 x float> %171) #9, !dbg !17
  %address.local231 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 12288), i64 %address.local192, !dbg !17
  %240 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !17
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 12288), i64 %address.local206, !dbg !17
  %241 = load <4 x half>, ptr addrspace(3) %address.local232, align 2, !dbg !17
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %230, <4 x half> %240, <4 x float> %239) #9, !dbg !17
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 12288), i64 %address.local220, !dbg !17
  %243 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %229, <4 x half> %241, <4 x float> %175) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %230, <4 x half> %243, <4 x float> %244) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %246 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local78), !dbg !12
  %247 = zext i32 %246 to i64, !dbg !12
  %248 = inttoptr i64 %247 to ptr addrspace(3), !dbg !12
  %address.local234 = getelementptr [2 x i8], ptr addrspace(1) %44, i64 %address.local70, !dbg !12
  %249 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %250 = ptrtoint ptr addrspace(1) %address.local234 to i64, !dbg !12
  %251 = sub i64 %250, %249, !dbg !12
  %dma.buffer.offset499 = trunc i64 %251 to i32, !dbg !12
  %252 = trunc i64 %249 to i32, !dbg !12
  %253 = insertelement <4 x i32> poison, i32 %252, i64 0, !dbg !12
  %254 = lshr i64 %249, 32, !dbg !12
  %255 = trunc i64 %254 to i32, !dbg !12
  %256 = insertelement <4 x i32> %253, i32 %255, i64 1, !dbg !12
  %257 = insertelement <4 x i32> %256, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc500 = insertelement <4 x i32> %257, i32 131072, i64 3, !dbg !12
  %258 = ptrtoint ptr addrspace(3) %248 to i32, !dbg !12
  %dma.buffer.lds501 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %258), !dbg !12
  %259 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds501, i32 %dma.buffer.offset499, <4 x i32> %dma.buffer.rsrc500) #10, !dbg !12
  %260 = add i64 %118, 6144, !dbg !16
  %261 = icmp slt i64 %260, 0, !dbg !16
  %262 = sub i64 -6145, %118, !dbg !16
  %263 = select i1 %261, i64 %262, i64 %260, !dbg !16
  %264 = sdiv i64 %263, 8192, !dbg !16
  %.lobit6 = ashr i64 %260, 63, !dbg !16
  %265 = xor i64 %264, %.lobit6, !dbg !16
  %266 = add nsw i64 %125, 96, !dbg !16
  %267 = icmp slt i64 %125, -96, !dbg !16
  %268 = sub nsw i64 -97, %125, !dbg !16
  %269 = select i1 %267, i64 %268, i64 %266, !dbg !16
  %270 = sdiv i64 %269, 128, !dbg !16
  %271 = sext i1 %267 to i64, !dbg !16
  %272 = xor i64 %270, %271, !dbg !16
  %273 = shl nsw i64 %272, 7, !dbg !16
  %274 = sub i64 %40, %273, !dbg !16
  %275 = shl i64 %265, 19, !dbg !16
  %276 = and i64 %275, 16252928, !dbg !16
  %277 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %276, !dbg !16
  %278 = getelementptr [2 x i8], ptr addrspace(1) %277, i64 %79, !dbg !16
  %279 = getelementptr [2 x i8], ptr addrspace(1) %278, i64 %274, !dbg !16
  %280 = getelementptr i8, ptr addrspace(1) %279, i64 192, !dbg !16
  %281 = getelementptr [2 x i8], ptr addrspace(1) %280, i64 %85, !dbg !16
  %282 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local82), !dbg !16
  %283 = zext i32 %282 to i64, !dbg !16
  %284 = inttoptr i64 %283 to ptr addrspace(3), !dbg !16
  %address.local235 = getelementptr [2 x i8], ptr addrspace(1) %281, i64 %address.local70, !dbg !16
  %285 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %286 = ptrtoint ptr addrspace(1) %address.local235 to i64, !dbg !16
  %287 = sub i64 %286, %285, !dbg !16
  %dma.buffer.offset496 = trunc i64 %287 to i32, !dbg !16
  %288 = trunc i64 %285 to i32, !dbg !16
  %289 = insertelement <4 x i32> poison, i32 %288, i64 0, !dbg !16
  %290 = lshr i64 %285, 32, !dbg !16
  %291 = trunc i64 %290 to i32, !dbg !16
  %292 = insertelement <4 x i32> %289, i32 %291, i64 1, !dbg !16
  %293 = insertelement <4 x i32> %292, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc497 = insertelement <4 x i32> %293, i32 131072, i64 3, !dbg !16
  %294 = ptrtoint ptr addrspace(3) %284 to i32, !dbg !16
  %dma.buffer.lds498 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %294), !dbg !16
  %295 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds498, i32 %dma.buffer.offset496, <4 x i32> %dma.buffer.rsrc497) #10, !dbg !16
  %296 = load <4 x half>, ptr addrspace(3) %address.local96, align 2, !dbg !13
  %297 = load <4 x half>, ptr addrspace(3) %address.local110, align 2, !dbg !13
  %298 = load <4 x half>, ptr addrspace(3) %address.local123, align 2, !dbg !17
  %299 = fmul <4 x half> %296, splat (half 0xH3015), !dbg !13
  %300 = fmul <4 x half> %297, splat (half 0xH3015), !dbg !13
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %298, <4 x float> %234) #9, !dbg !17
  %302 = load <4 x half>, ptr addrspace(3) %address.local137, align 2, !dbg !17
  %303 = load <4 x half>, ptr addrspace(3) %address.local151, align 2, !dbg !17
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %300, <4 x half> %302, <4 x float> %301) #9, !dbg !17
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %303, <4 x float> %238) #9, !dbg !17
  %306 = load <4 x half>, ptr addrspace(3) %address.local165, align 2, !dbg !17
  %307 = load <4 x half>, ptr addrspace(3) %address.local179, align 2, !dbg !17
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %300, <4 x half> %306, <4 x float> %305) #9, !dbg !17
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %307, <4 x float> %242) #9, !dbg !17
  %310 = load <4 x half>, ptr addrspace(3) %address.local193, align 2, !dbg !17
  %311 = load <4 x half>, ptr addrspace(3) %address.local207, align 2, !dbg !17
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %300, <4 x half> %310, <4 x float> %309) #9, !dbg !17
  %313 = load <4 x half>, ptr addrspace(3) %address.local221, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %299, <4 x half> %311, <4 x float> %245) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %300, <4 x half> %313, <4 x float> %314) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %316 = add nsw i64 %79, %8, !dbg !18
  %317 = shl i64 %316, 19, !dbg !18
  %318 = and i64 %317, 16252928, !dbg !18
  %319 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %318, !dbg !18
  %320 = getelementptr [2 x i8], ptr addrspace(1) %319, i64 %85, !dbg !18
  %321 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local65), !dbg !18
  %322 = zext i32 %321 to i64, !dbg !18
  %323 = inttoptr i64 %322 to ptr addrspace(3), !dbg !18
  %address.local236 = shl nuw nsw i64 %address.local, 5, !dbg !18
  %address.local237 = and i64 %address.local236, 8064, !dbg !18
  %address.local238 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local239 = and i64 %address.local238, 24, !dbg !18
  %address.local240 = or disjoint i64 %address.local237, %address.local239, !dbg !18
  %address.local241 = getelementptr [2 x i8], ptr addrspace(1) %320, i64 %address.local240, !dbg !18
  %324 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %325 = ptrtoint ptr addrspace(1) %address.local241 to i64, !dbg !18
  %326 = sub i64 %325, %324, !dbg !18
  %dma.buffer.offset493 = trunc i64 %326 to i32, !dbg !18
  %327 = trunc i64 %324 to i32, !dbg !18
  %328 = insertelement <4 x i32> poison, i32 %327, i64 0, !dbg !18
  %329 = lshr i64 %324, 32, !dbg !18
  %330 = trunc i64 %329 to i32, !dbg !18
  %331 = insertelement <4 x i32> %328, i32 %330, i64 1, !dbg !18
  %332 = insertelement <4 x i32> %331, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc494 = insertelement <4 x i32> %332, i32 131072, i64 3, !dbg !18
  %333 = ptrtoint ptr addrspace(3) %323 to i32, !dbg !18
  %dma.buffer.lds495 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %333), !dbg !18
  %334 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds495, i32 %dma.buffer.offset493, <4 x i32> %dma.buffer.rsrc494) #10, !dbg !18
  %335 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !13
  %336 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !13
  %337 = load <4 x half>, ptr addrspace(3) %address.local226, align 2, !dbg !17
  %338 = fmul <4 x half> %335, splat (half 0xH3015), !dbg !13
  %339 = fmul <4 x half> %336, splat (half 0xH3015), !dbg !13
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %338, <4 x half> %337, <4 x float> %304) #9, !dbg !17
  %341 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !17
  %342 = load <4 x half>, ptr addrspace(3) %address.local228, align 2, !dbg !17
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %339, <4 x half> %341, <4 x float> %340) #9, !dbg !17
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %338, <4 x half> %342, <4 x float> %308) #9, !dbg !17
  %345 = load <4 x half>, ptr addrspace(3) %address.local229, align 2, !dbg !17
  %346 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !17
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %339, <4 x half> %345, <4 x float> %344) #9, !dbg !17
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %338, <4 x half> %346, <4 x float> %312) #9, !dbg !17
  %349 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !17
  %350 = load <4 x half>, ptr addrspace(3) %address.local232, align 2, !dbg !17
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %339, <4 x half> %349, <4 x float> %348) #9, !dbg !17
  %352 = load <4 x half>, ptr addrspace(3) %address.local233, align 2, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %338, <4 x half> %350, <4 x float> %315) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %354 = or disjoint i64 %62, 63, !dbg !14
  %355 = icmp uge i64 %11, %354, !dbg !14
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %339, <4 x half> %352, <4 x float> %353) #9, !dbg !17
  %357 = and i1 %31, %355, !dbg !14
  br i1 %357, label %358, label %359, !dbg !14

358:                                              ; preds = %61
  br label %402, !dbg !14

359:                                              ; preds = %61
  %360 = add nsw i64 %11, %27, !dbg !14
  %361 = shl nuw nsw i64 %28, 4, !dbg !14
  %362 = add nsw i64 %360, %361, !dbg !14
  %363 = and i32 %26, 240, !dbg !14
  %index.compact = zext nneg i32 %363 to i64, !dbg !14
  %364 = sub nsw i64 %362, %index.compact, !dbg !14
  %365 = add nsw i64 %364, 1, !dbg !24
  %366 = and i64 %27, 48, !dbg !14
  %367 = or disjoint i64 %62, %366, !dbg !14
  %368 = lshr i64 %365, 32, !dbg !25
  %compare.high = trunc i64 %368 to i32, !dbg !25
  %369 = lshr i64 %367, 32, !dbg !25
  %compare.high427 = trunc i64 %369 to i32, !dbg !25
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high427, !dbg !25
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high427, !dbg !25
  %compare.low.base = trunc i64 %365 to i32, !dbg !25
  %compare.low.base428 = trunc i64 %367 to i32, !dbg !25
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base428, !dbg !25
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !25
  %370 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 1) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !25
  %compare.low429 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !25
  %compare.result430 = select i1 %compare.high.equal, i1 %compare.low429, i1 %compare.high.order, !dbg !25
  %371 = select i1 %compare.result430, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base431 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 2) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local432 = or i32 %compare.local.base431, 2, !dbg !25
  %compare.low433 = icmp ugt i32 %compare.low.base, %compare.local432, !dbg !25
  %compare.result434 = select i1 %compare.high.equal, i1 %compare.low433, i1 %compare.high.order, !dbg !25
  %372 = select i1 %compare.result434, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base435 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 3) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local436 = or i32 %compare.local.base435, 3, !dbg !25
  %compare.low437 = icmp ugt i32 %compare.low.base, %compare.local436, !dbg !25
  %compare.result438 = select i1 %compare.high.equal, i1 %compare.low437, i1 %compare.high.order, !dbg !25
  %373 = select i1 %compare.result438, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base439 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 4) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local440 = or i32 %compare.local.base439, 4, !dbg !25
  %compare.low441 = icmp ugt i32 %compare.low.base, %compare.local440, !dbg !25
  %compare.result442 = select i1 %compare.high.equal, i1 %compare.low441, i1 %compare.high.order, !dbg !25
  %374 = select i1 %compare.result442, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base443 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 5) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local444 = or i32 %compare.local.base443, 5, !dbg !25
  %compare.low445 = icmp ugt i32 %compare.low.base, %compare.local444, !dbg !25
  %compare.result446 = select i1 %compare.high.equal, i1 %compare.low445, i1 %compare.high.order, !dbg !25
  %375 = select i1 %compare.result446, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base447 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 6) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local448 = or i32 %compare.local.base447, 6, !dbg !25
  %compare.low449 = icmp ugt i32 %compare.low.base, %compare.local448, !dbg !25
  %compare.result450 = select i1 %compare.high.equal, i1 %compare.low449, i1 %compare.high.order, !dbg !25
  %376 = select i1 %compare.result450, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base451 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 7) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local452 = or i32 %compare.local.base451, 7, !dbg !25
  %compare.low453 = icmp ugt i32 %compare.low.base, %compare.local452, !dbg !25
  %compare.result454 = select i1 %compare.high.equal, i1 %compare.low453, i1 %compare.high.order, !dbg !25
  %377 = select i1 %compare.result454, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base455 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 8) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local456 = or i32 %compare.local.base455, 8, !dbg !25
  %compare.low457 = icmp ugt i32 %compare.low.base, %compare.local456, !dbg !25
  %compare.result458 = select i1 %compare.high.equal, i1 %compare.low457, i1 %compare.high.order, !dbg !25
  %378 = select i1 %compare.result458, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base459 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 9) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local460 = or i32 %compare.local.base459, 9, !dbg !25
  %compare.low461 = icmp ugt i32 %compare.low.base, %compare.local460, !dbg !25
  %compare.result462 = select i1 %compare.high.equal, i1 %compare.low461, i1 %compare.high.order, !dbg !25
  %379 = select i1 %compare.result462, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base463 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 10) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local464 = or i32 %compare.local.base463, 10, !dbg !25
  %compare.low465 = icmp ugt i32 %compare.low.base, %compare.local464, !dbg !25
  %compare.result466 = select i1 %compare.high.equal, i1 %compare.low465, i1 %compare.high.order, !dbg !25
  %380 = select i1 %compare.result466, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base467 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 11) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local468 = or i32 %compare.local.base467, 11, !dbg !25
  %compare.low469 = icmp ugt i32 %compare.low.base, %compare.local468, !dbg !25
  %compare.result470 = select i1 %compare.high.equal, i1 %compare.low469, i1 %compare.high.order, !dbg !25
  %381 = select i1 %compare.result470, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base471 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 12) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local472 = or i32 %compare.local.base471, 12, !dbg !25
  %compare.low473 = icmp ugt i32 %compare.low.base, %compare.local472, !dbg !25
  %compare.result474 = select i1 %compare.high.equal, i1 %compare.low473, i1 %compare.high.order, !dbg !25
  %382 = select i1 %compare.result474, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base475 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 13) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local476 = or i32 %compare.local.base475, 13, !dbg !25
  %compare.low477 = icmp ugt i32 %compare.low.base, %compare.local476, !dbg !25
  %compare.result478 = select i1 %compare.high.equal, i1 %compare.low477, i1 %compare.high.order, !dbg !25
  %383 = select i1 %compare.result478, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base479 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 14) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local480 = or i32 %compare.local.base479, 14, !dbg !25
  %compare.low481 = icmp ugt i32 %compare.low.base, %compare.local480, !dbg !25
  %compare.result482 = select i1 %compare.high.equal, i1 %compare.low481, i1 %compare.high.order, !dbg !25
  %384 = select i1 %compare.result482, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base483 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base428, i32 15) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local484 = or i32 %compare.local.base483, 15, !dbg !25
  %compare.low485 = icmp ugt i32 %compare.low.base, %compare.local484, !dbg !25
  %compare.result486 = select i1 %compare.high.equal, i1 %compare.low485, i1 %compare.high.order, !dbg !25
  %385 = select i1 %compare.result486, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %386 = insertelement <16 x float> poison, float %370, i64 0, !dbg !14
  %387 = insertelement <16 x float> %386, float %371, i64 1, !dbg !14
  %388 = insertelement <16 x float> %387, float %372, i64 2, !dbg !14
  %389 = insertelement <16 x float> %388, float %373, i64 3, !dbg !14
  %390 = insertelement <16 x float> %389, float %374, i64 4, !dbg !14
  %391 = insertelement <16 x float> %390, float %375, i64 5, !dbg !14
  %392 = insertelement <16 x float> %391, float %376, i64 6, !dbg !14
  %393 = insertelement <16 x float> %392, float %377, i64 7, !dbg !14
  %394 = insertelement <16 x float> %393, float %378, i64 8, !dbg !14
  %395 = insertelement <16 x float> %394, float %379, i64 9, !dbg !14
  %396 = insertelement <16 x float> %395, float %380, i64 10, !dbg !14
  %397 = insertelement <16 x float> %396, float %381, i64 11, !dbg !14
  %398 = insertelement <16 x float> %397, float %382, i64 12, !dbg !14
  %399 = insertelement <16 x float> %398, float %383, i64 13, !dbg !14
  %400 = insertelement <16 x float> %399, float %384, i64 14, !dbg !14
  %401 = insertelement <16 x float> %400, float %385, i64 15, !dbg !14
  br label %402, !dbg !14

402:                                              ; preds = %358, %359
  %403 = phi <16 x float> [ %401, %359 ], [ zeroinitializer, %358 ], !dbg !14
  br label %404, !dbg !14

404:                                              ; preds = %402
  %405 = shufflevector <16 x float> %403, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %406 = fadd <4 x float> %343, %405, !dbg !27
  %407 = extractelement <4 x float> %406, i64 0, !dbg !27
  %408 = extractelement <4 x float> %406, i64 1, !dbg !27
  %409 = extractelement <4 x float> %406, i64 2, !dbg !27
  %410 = extractelement <4 x float> %406, i64 3, !dbg !27
  %411 = shufflevector <16 x float> %403, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %412 = fadd <4 x float> %347, %411, !dbg !27
  %413 = extractelement <4 x float> %412, i64 0, !dbg !27
  %414 = extractelement <4 x float> %412, i64 1, !dbg !27
  %415 = extractelement <4 x float> %412, i64 2, !dbg !27
  %416 = extractelement <4 x float> %412, i64 3, !dbg !27
  %417 = shufflevector <16 x float> %403, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %418 = fadd <4 x float> %351, %417, !dbg !27
  %419 = extractelement <4 x float> %418, i64 0, !dbg !27
  %420 = extractelement <4 x float> %418, i64 1, !dbg !27
  %421 = extractelement <4 x float> %418, i64 2, !dbg !27
  %422 = extractelement <4 x float> %418, i64 3, !dbg !27
  %423 = shufflevector <16 x float> %403, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %424 = fadd <4 x float> %356, %423, !dbg !27
  %425 = extractelement <4 x float> %424, i64 0, !dbg !27
  %426 = extractelement <4 x float> %424, i64 1, !dbg !27
  %427 = extractelement <4 x float> %424, i64 2, !dbg !27
  %428 = extractelement <4 x float> %424, i64 3, !dbg !27
  %429 = call float @llvm.maxnum.f32(float %407, float 0xFFF0000000000000), !dbg !28
  %430 = call float @llvm.maxnum.f32(float %429, float %408), !dbg !28
  %431 = call float @llvm.maxnum.f32(float %430, float %409), !dbg !28
  %432 = call float @llvm.maxnum.f32(float %431, float %410), !dbg !28
  %433 = call float @llvm.maxnum.f32(float %432, float %413), !dbg !28
  %434 = call float @llvm.maxnum.f32(float %433, float %414), !dbg !28
  %435 = call float @llvm.maxnum.f32(float %434, float %415), !dbg !28
  %436 = call float @llvm.maxnum.f32(float %435, float %416), !dbg !28
  %437 = call float @llvm.maxnum.f32(float %436, float %419), !dbg !28
  %438 = call float @llvm.maxnum.f32(float %437, float %420), !dbg !28
  %439 = call float @llvm.maxnum.f32(float %438, float %421), !dbg !28
  %440 = call float @llvm.maxnum.f32(float %439, float %422), !dbg !28
  %441 = call float @llvm.maxnum.f32(float %440, float %425), !dbg !28
  %442 = call float @llvm.maxnum.f32(float %441, float %426), !dbg !28
  %443 = call float @llvm.maxnum.f32(float %442, float %427), !dbg !28
  %444 = call float @llvm.maxnum.f32(float %443, float %428), !dbg !28
  %445 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %446 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %445), !dbg !28
  %447 = add nuw nsw i32 %446, 64, !dbg !28
  %448 = xor i32 %446, 16, !dbg !28
  %449 = and i32 %447, 192, !dbg !28
  %450 = icmp samesign ult i32 %448, %449, !dbg !28
  %451 = select i1 %450, i32 %448, i32 %446, !dbg !28
  %452 = shl nuw nsw i32 %451, 2, !dbg !28
  %453 = bitcast float %444 to i32, !dbg !28
  %454 = call i32 @llvm.amdgcn.ds.bpermute(i32 %452, i32 %453), !dbg !28
  %455 = bitcast i32 %454 to float, !dbg !28
  %456 = call float @llvm.maxnum.f32(float %444, float %455), !dbg !28
  %457 = xor i32 %446, 32, !dbg !28
  %458 = icmp samesign ult i32 %457, %449, !dbg !28
  %459 = select i1 %458, i32 %457, i32 %446, !dbg !28
  %460 = shl nuw nsw i32 %459, 2, !dbg !28
  %461 = bitcast float %456 to i32, !dbg !28
  %462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %460, i32 %461), !dbg !28
  %463 = bitcast i32 %462 to float, !dbg !28
  %464 = call float @llvm.maxnum.f32(float %456, float %463), !dbg !28
  %465 = insertelement <1 x float> poison, float %464, i64 0, !dbg !28
  %466 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %465, <1 x float> %51), !dbg !28
  %467 = shufflevector <1 x float> %466, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !29
  %468 = fsub <4 x float> %406, %467, !dbg !29
  %469 = fsub <4 x float> %412, %467, !dbg !29
  %470 = fsub <4 x float> %418, %467, !dbg !29
  %471 = fsub <4 x float> %424, %467, !dbg !29
  %472 = fsub <1 x float> %51, %466, !dbg !30
  %473 = extractelement <1 x float> %472, i64 0, !dbg !31
  %474 = call float @llvm.amdgcn.exp2.f32(float %473), !dbg !31
  %475 = insertelement <4 x float> poison, float %474, i64 0, !dbg !36
  %476 = shufflevector <4 x float> %475, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %477 = fmul <4 x float> %59, %476, !dbg !36
  %478 = fmul <4 x float> %58, %476, !dbg !36
  %479 = fmul <4 x float> %56, %476, !dbg !36
  %480 = fmul <4 x float> %55, %476, !dbg !36
  %481 = fmul <4 x float> %53, %476, !dbg !36
  %482 = fmul <4 x float> %52, %476, !dbg !36
  %483 = fmul <4 x float> %48, %476, !dbg !36
  %484 = fmul <4 x float> %47, %476, !dbg !36
  %485 = extractelement <1 x float> %50, i64 0, !dbg !32
  %486 = fmul float %485, %474, !dbg !32
  %487 = extractelement <4 x float> %468, i64 0, !dbg !33
  %488 = call float @llvm.amdgcn.exp2.f32(float %487), !dbg !33
  %489 = insertelement <4 x float> poison, float %488, i64 0, !dbg !35
  %490 = extractelement <4 x float> %468, i64 1, !dbg !33
  %491 = call float @llvm.amdgcn.exp2.f32(float %490), !dbg !33
  %492 = insertelement <4 x float> %489, float %491, i64 1, !dbg !35
  %493 = fadd float %488, %491, !dbg !34
  %494 = extractelement <4 x float> %468, i64 2, !dbg !33
  %495 = call float @llvm.amdgcn.exp2.f32(float %494), !dbg !33
  %496 = insertelement <4 x float> %492, float %495, i64 2, !dbg !35
  %497 = fadd float %493, %495, !dbg !34
  %498 = extractelement <4 x float> %468, i64 3, !dbg !33
  %499 = call float @llvm.amdgcn.exp2.f32(float %498), !dbg !33
  %500 = insertelement <4 x float> %496, float %499, i64 3, !dbg !35
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %500) #9, !dbg !35, !frisk.register_lifetimes !1
  %501 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !35
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %501) #9, !dbg !34, !frisk.register_lifetimes !1
  %502 = fadd float %497, %499, !dbg !34
  %503 = extractelement <4 x float> %469, i64 0, !dbg !33
  %504 = call float @llvm.amdgcn.exp2.f32(float %503), !dbg !33
  %505 = insertelement <4 x float> poison, float %504, i64 0, !dbg !35
  %506 = fadd float %502, %504, !dbg !34
  %507 = extractelement <4 x float> %469, i64 1, !dbg !33
  %508 = call float @llvm.amdgcn.exp2.f32(float %507), !dbg !33
  %509 = insertelement <4 x float> %505, float %508, i64 1, !dbg !35
  %510 = fadd float %506, %508, !dbg !34
  %511 = extractelement <4 x float> %469, i64 2, !dbg !33
  %512 = call float @llvm.amdgcn.exp2.f32(float %511), !dbg !33
  %513 = insertelement <4 x float> %509, float %512, i64 2, !dbg !35
  %514 = fadd float %510, %512, !dbg !34
  %515 = extractelement <4 x float> %469, i64 3, !dbg !33
  %516 = call float @llvm.amdgcn.exp2.f32(float %515), !dbg !33
  %517 = insertelement <4 x float> %513, float %516, i64 3, !dbg !35
  %math.result55 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %517) #9, !dbg !35, !frisk.register_lifetimes !1
  %518 = fptrunc <4 x float> %math.result55 to <4 x half>, !dbg !35
  %math.packed56 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %518) #9, !dbg !34, !frisk.register_lifetimes !1
  %519 = fadd float %514, %516, !dbg !34
  %520 = extractelement <4 x float> %470, i64 0, !dbg !33
  %521 = call float @llvm.amdgcn.exp2.f32(float %520), !dbg !33
  %522 = insertelement <4 x float> poison, float %521, i64 0, !dbg !35
  %523 = fadd float %519, %521, !dbg !34
  %524 = extractelement <4 x float> %470, i64 1, !dbg !33
  %525 = call float @llvm.amdgcn.exp2.f32(float %524), !dbg !33
  %526 = insertelement <4 x float> %522, float %525, i64 1, !dbg !35
  %527 = fadd float %523, %525, !dbg !34
  %528 = extractelement <4 x float> %470, i64 2, !dbg !33
  %529 = call float @llvm.amdgcn.exp2.f32(float %528), !dbg !33
  %530 = insertelement <4 x float> %526, float %529, i64 2, !dbg !35
  %531 = fadd float %527, %529, !dbg !34
  %532 = extractelement <4 x float> %470, i64 3, !dbg !33
  %533 = call float @llvm.amdgcn.exp2.f32(float %532), !dbg !33
  %534 = insertelement <4 x float> %530, float %533, i64 3, !dbg !35
  %math.result57 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %534) #9, !dbg !35, !frisk.register_lifetimes !1
  %535 = fptrunc <4 x float> %math.result57 to <4 x half>, !dbg !35
  %math.packed58 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %535) #9, !dbg !34, !frisk.register_lifetimes !1
  %536 = fadd float %531, %533, !dbg !34
  %537 = extractelement <4 x float> %471, i64 0, !dbg !33
  %538 = call float @llvm.amdgcn.exp2.f32(float %537), !dbg !33
  %539 = insertelement <4 x float> poison, float %538, i64 0, !dbg !35
  %540 = fadd float %536, %538, !dbg !34
  %541 = extractelement <4 x float> %471, i64 1, !dbg !33
  %542 = call float @llvm.amdgcn.exp2.f32(float %541), !dbg !33
  %543 = insertelement <4 x float> %539, float %542, i64 1, !dbg !35
  %544 = fadd float %540, %542, !dbg !34
  %545 = extractelement <4 x float> %471, i64 2, !dbg !33
  %546 = call float @llvm.amdgcn.exp2.f32(float %545), !dbg !33
  %547 = insertelement <4 x float> %543, float %546, i64 2, !dbg !35
  %548 = fadd float %544, %546, !dbg !34
  %549 = extractelement <4 x float> %471, i64 3, !dbg !33
  %550 = call float @llvm.amdgcn.exp2.f32(float %549), !dbg !33
  %551 = insertelement <4 x float> %547, float %550, i64 3, !dbg !35
  %math.result59 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %551) #9, !dbg !35, !frisk.register_lifetimes !1
  %552 = fptrunc <4 x float> %math.result59 to <4 x half>, !dbg !35
  %math.packed60 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %552) #9, !dbg !34, !frisk.register_lifetimes !1
  %553 = fadd float %548, %550, !dbg !34
  %554 = bitcast float %553 to i32, !dbg !34
  %555 = call i32 @llvm.amdgcn.ds.bpermute(i32 %452, i32 %554), !dbg !34
  %556 = bitcast i32 %555 to float, !dbg !34
  %557 = fadd float %553, %556, !dbg !34
  %558 = bitcast float %557 to i32, !dbg !34
  %559 = call i32 @llvm.amdgcn.ds.bpermute(i32 %460, i32 %558), !dbg !34
  %560 = bitcast i32 %559 to float, !dbg !34
  %561 = fadd float %557, %560, !dbg !34
  %562 = fadd float %561, %486, !dbg !34
  %563 = insertelement <1 x float> poison, float %562, i64 0, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread242 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !15, !range !40, !frisk.register_lifetimes !1
  %address.local243 = zext nneg i32 %address.thread242 to i64, !dbg !13
  %address.local244 = and i64 %address.local243, 60, !dbg !15
  %address.local245 = add nuw nsw i64 %address.local244, %address.local243, !dbg !15
  %address.local246 = shl nuw nsw i64 %address.local245, 3, !dbg !15
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local246, !dbg !15
  %address.local248 = lshr i64 %address.local243, 4, !dbg !15
  %address.local249 = and i64 %address.local248, 3, !dbg !15
  %address.local250 = mul nuw nsw i64 %address.local249, 384, !dbg !15
  %address.local251 = getelementptr [2 x i8], ptr addrspace(3) %address.local247, i64 %address.local250, !dbg !15
  %address.local252 = lshr i64 %address.local243, 2, !dbg !17
  %address.local253 = mul nsw i64 %address.local252, -32, !dbg !15
  %address.local254 = getelementptr [2 x i8], ptr addrspace(3) %address.local251, i64 %address.local253, !dbg !15
  %564 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local254, i32 0) #0, !dbg !15
  %565 = bitcast <4 x i32> %564 to <8 x half>, !dbg !15
  %address.local255 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local246, !dbg !15
  %address.local256 = getelementptr [2 x i8], ptr addrspace(3) %address.local255, i64 %address.local250, !dbg !15
  %address.local257 = getelementptr [2 x i8], ptr addrspace(3) %address.local256, i64 %address.local253, !dbg !15
  %address.local258 = getelementptr i8, ptr addrspace(3) %address.local257, i64 256, !dbg !15
  %566 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local258, i32 0) #0, !dbg !15
  %567 = bitcast <4 x i32> %566 to <8 x half>, !dbg !15
  %address.local259 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local246, !dbg !15
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) %address.local259, i64 %address.local250, !dbg !15
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %address.local260, i64 %address.local253, !dbg !15
  %address.local262 = getelementptr i8, ptr addrspace(3) %address.local261, i64 512, !dbg !15
  %568 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local262, i32 0) #0, !dbg !15
  %569 = bitcast <4 x i32> %568 to <8 x half>, !dbg !15
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local246, !dbg !15
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local250, !dbg !15
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local253, !dbg !15
  %address.local266 = getelementptr i8, ptr addrspace(3) %address.local265, i64 768, !dbg !15
  %570 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local266, i32 0) #0, !dbg !15
  %571 = bitcast <4 x i32> %570 to <8 x half>, !dbg !15
  %572 = add nsw i64 %79, %8, !dbg !18
  %573 = shl i64 %572, 19, !dbg !18
  %574 = and i64 %573, 16252928, !dbg !18
  %575 = add i64 %574, %85, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %576 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %575, !dbg !18
  %577 = getelementptr i8, ptr addrspace(1) %576, i64 64, !dbg !18
  %address.local267 = shl nuw nsw i64 %address.local243, 3, !dbg !12
  %address.local268 = and i64 %address.local267, 1536, !dbg !12
  %address.local269 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local268, !dbg !12
  %address.local270 = ptrtoint ptr addrspace(3) %address.local269 to i64, !dbg !12
  %address.local271 = trunc i64 %address.local270 to i32, !dbg !12
  %578 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local271), !dbg !18
  %579 = zext i32 %578 to i64, !dbg !18
  %580 = inttoptr i64 %579 to ptr addrspace(3), !dbg !18
  %address.local272 = shl nuw nsw i64 %address.local243, 5, !dbg !18
  %address.local273 = and i64 %address.local272, 8064, !dbg !18
  %address.local274 = shl nuw nsw i64 %address.local243, 3, !dbg !18
  %address.local275 = and i64 %address.local274, 24, !dbg !18
  %address.local276 = or disjoint i64 %address.local273, %address.local275, !dbg !18
  %address.local277 = getelementptr [2 x i8], ptr addrspace(1) %577, i64 %address.local276, !dbg !18
  %581 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %582 = ptrtoint ptr addrspace(1) %address.local277 to i64, !dbg !18
  %583 = sub i64 %582, %581, !dbg !18
  %dma.buffer.offset490 = trunc i64 %583 to i32, !dbg !18
  %584 = trunc i64 %581 to i32, !dbg !18
  %585 = insertelement <4 x i32> poison, i32 %584, i64 0, !dbg !18
  %586 = lshr i64 %581, 32, !dbg !18
  %587 = trunc i64 %586 to i32, !dbg !18
  %588 = insertelement <4 x i32> %585, i32 %587, i64 1, !dbg !18
  %589 = insertelement <4 x i32> %588, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc491 = insertelement <4 x i32> %589, i32 131072, i64 3, !dbg !18
  %590 = ptrtoint ptr addrspace(3) %580 to i32, !dbg !18
  %dma.buffer.lds492 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %590), !dbg !18
  %591 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds492, i32 %dma.buffer.offset490, <4 x i32> %dma.buffer.rsrc491) #10, !dbg !18
  %592 = shufflevector <8 x half> %565, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %592, <4 x float> %484) #9, !dbg !15
  %594 = shufflevector <8 x half> %567, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %595 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed56, <4 x half> %594, <4 x float> %593) #9, !dbg !15
  %596 = shufflevector <8 x half> %569, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed58, <4 x half> %596, <4 x float> %595) #9, !dbg !15
  %598 = shufflevector <8 x half> %571, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %599 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %598, <4 x float> %597) #9, !dbg !15
  %600 = shufflevector <8 x half> %565, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %601 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %600, <4 x float> %483) #9, !dbg !15
  %602 = shufflevector <8 x half> %567, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %603 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed56, <4 x half> %602, <4 x float> %601) #9, !dbg !15
  %604 = shufflevector <8 x half> %569, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %605 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed58, <4 x half> %604, <4 x float> %603) #9, !dbg !15
  %606 = shufflevector <8 x half> %571, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %607 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %606, <4 x float> %605) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %608 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local254, i32 0) #0, !dbg !15
  %609 = bitcast <4 x i32> %608 to <8 x half>, !dbg !15
  %610 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local258, i32 0) #0, !dbg !15
  %611 = bitcast <4 x i32> %610 to <8 x half>, !dbg !15
  %612 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local262, i32 0) #0, !dbg !15
  %613 = bitcast <4 x i32> %612 to <8 x half>, !dbg !15
  %614 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local266, i32 0) #0, !dbg !15
  %615 = bitcast <4 x i32> %614 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %616 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %575, !dbg !18
  %617 = getelementptr i8, ptr addrspace(1) %616, i64 128, !dbg !18
  %618 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local271), !dbg !18
  %619 = zext i32 %618 to i64, !dbg !18
  %620 = inttoptr i64 %619 to ptr addrspace(3), !dbg !18
  %address.local278 = getelementptr [2 x i8], ptr addrspace(1) %617, i64 %address.local276, !dbg !18
  %621 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %622 = ptrtoint ptr addrspace(1) %address.local278 to i64, !dbg !18
  %623 = sub i64 %622, %621, !dbg !18
  %dma.buffer.offset487 = trunc i64 %623 to i32, !dbg !18
  %624 = trunc i64 %621 to i32, !dbg !18
  %625 = insertelement <4 x i32> poison, i32 %624, i64 0, !dbg !18
  %626 = lshr i64 %621, 32, !dbg !18
  %627 = trunc i64 %626 to i32, !dbg !18
  %628 = insertelement <4 x i32> %625, i32 %627, i64 1, !dbg !18
  %629 = insertelement <4 x i32> %628, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc488 = insertelement <4 x i32> %629, i32 131072, i64 3, !dbg !18
  %630 = ptrtoint ptr addrspace(3) %620 to i32, !dbg !18
  %dma.buffer.lds489 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %630), !dbg !18
  %631 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds489, i32 %dma.buffer.offset487, <4 x i32> %dma.buffer.rsrc488) #10, !dbg !18
  %632 = shufflevector <8 x half> %609, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %633 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %632, <4 x float> %482) #9, !dbg !15
  %634 = shufflevector <8 x half> %611, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed56, <4 x half> %634, <4 x float> %633) #9, !dbg !15
  %636 = shufflevector <8 x half> %613, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed58, <4 x half> %636, <4 x float> %635) #9, !dbg !15
  %638 = shufflevector <8 x half> %615, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %638, <4 x float> %637) #9, !dbg !15
  %640 = shufflevector <8 x half> %609, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %640, <4 x float> %481) #9, !dbg !15
  %642 = shufflevector <8 x half> %611, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed56, <4 x half> %642, <4 x float> %641) #9, !dbg !15
  %644 = shufflevector <8 x half> %613, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed58, <4 x half> %644, <4 x float> %643) #9, !dbg !15
  %646 = shufflevector <8 x half> %615, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %646, <4 x float> %645) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %648 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local254, i32 0) #0, !dbg !15
  %649 = bitcast <4 x i32> %648 to <8 x half>, !dbg !15
  %650 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local258, i32 0) #0, !dbg !15
  %651 = bitcast <4 x i32> %650 to <8 x half>, !dbg !15
  %652 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local262, i32 0) #0, !dbg !15
  %653 = bitcast <4 x i32> %652 to <8 x half>, !dbg !15
  %654 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local266, i32 0) #0, !dbg !15
  %655 = bitcast <4 x i32> %654 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %656 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %575, !dbg !18
  %657 = getelementptr i8, ptr addrspace(1) %656, i64 192, !dbg !18
  %658 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local271), !dbg !18
  %659 = zext i32 %658 to i64, !dbg !18
  %660 = inttoptr i64 %659 to ptr addrspace(3), !dbg !18
  %address.local279 = getelementptr [2 x i8], ptr addrspace(1) %657, i64 %address.local276, !dbg !18
  %661 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %662 = ptrtoint ptr addrspace(1) %address.local279 to i64, !dbg !18
  %663 = sub i64 %662, %661, !dbg !18
  %dma.buffer.offset = trunc i64 %663 to i32, !dbg !18
  %664 = trunc i64 %661 to i32, !dbg !18
  %665 = insertelement <4 x i32> poison, i32 %664, i64 0, !dbg !18
  %666 = lshr i64 %661, 32, !dbg !18
  %667 = trunc i64 %666 to i32, !dbg !18
  %668 = insertelement <4 x i32> %665, i32 %667, i64 1, !dbg !18
  %669 = insertelement <4 x i32> %668, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %669, i32 131072, i64 3, !dbg !18
  %670 = ptrtoint ptr addrspace(3) %660 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %670), !dbg !18
  %671 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %672 = shufflevector <8 x half> %649, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %672, <4 x float> %480) #9, !dbg !15
  %674 = shufflevector <8 x half> %651, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed56, <4 x half> %674, <4 x float> %673) #9, !dbg !15
  %676 = shufflevector <8 x half> %653, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed58, <4 x half> %676, <4 x float> %675) #9, !dbg !15
  %678 = shufflevector <8 x half> %655, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %678, <4 x float> %677) #9, !dbg !15
  %680 = shufflevector <8 x half> %649, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %680, <4 x float> %479) #9, !dbg !15
  %682 = shufflevector <8 x half> %651, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed56, <4 x half> %682, <4 x float> %681) #9, !dbg !15
  %684 = shufflevector <8 x half> %653, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %685 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed58, <4 x half> %684, <4 x float> %683) #9, !dbg !15
  %686 = shufflevector <8 x half> %655, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %687 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %686, <4 x float> %685) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %688 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local254, i32 0) #0, !dbg !15
  %689 = bitcast <4 x i32> %688 to <8 x half>, !dbg !15
  %690 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local258, i32 0) #0, !dbg !15
  %691 = bitcast <4 x i32> %690 to <8 x half>, !dbg !15
  %692 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local262, i32 0) #0, !dbg !15
  %693 = bitcast <4 x i32> %692 to <8 x half>, !dbg !15
  %694 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local266, i32 0) #0, !dbg !15
  %695 = bitcast <4 x i32> %694 to <8 x half>, !dbg !15
  %696 = shufflevector <8 x half> %689, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %697 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %696, <4 x float> %478) #9, !dbg !15
  %698 = shufflevector <8 x half> %691, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %699 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed56, <4 x half> %698, <4 x float> %697) #9, !dbg !15
  %700 = shufflevector <8 x half> %693, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %701 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed58, <4 x half> %700, <4 x float> %699) #9, !dbg !15
  %702 = shufflevector <8 x half> %695, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %703 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %702, <4 x float> %701) #9, !dbg !15
  %704 = shufflevector <8 x half> %689, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %705 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %704, <4 x float> %477) #9, !dbg !15
  %706 = shufflevector <8 x half> %691, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %707 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed56, <4 x half> %706, <4 x float> %705) #9, !dbg !15
  %708 = shufflevector <8 x half> %693, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed58, <4 x half> %708, <4 x float> %707) #9, !dbg !15
  %710 = shufflevector <8 x half> %695, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %711 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %710, <4 x float> %709) #9, !dbg !15
  %712 = shufflevector <4 x float> %599, <4 x float> %607, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %713 = shufflevector <4 x float> %639, <4 x float> %647, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %714 = shufflevector <4 x float> %679, <4 x float> %687, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %715 = shufflevector <4 x float> %703, <4 x float> %711, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %716 = add nuw nsw i64 %46, 1, !dbg !7
  br label %45, !dbg !7

717:                                              ; preds = %45
  %718 = and i32 %26, 16, !dbg !7
  %.not2 = icmp eq i32 %718, 0, !dbg !7
  %719 = and i32 %26, 32, !dbg !7
  %.not3 = icmp eq i32 %719, 0, !dbg !7
  %720 = extractelement <8 x float> %60, i64 0, !dbg !7
  %721 = extractelement <8 x float> %60, i64 1, !dbg !7
  %722 = extractelement <8 x float> %60, i64 2, !dbg !7
  %723 = extractelement <8 x float> %60, i64 3, !dbg !7
  %724 = select i1 %.not2, float %721, float %720, !dbg !7
  %725 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %726 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %725), !dbg !7
  %727 = add nuw nsw i32 %726, 64, !dbg !7
  %728 = xor i32 %726, 16, !dbg !7
  %729 = and i32 %727, 192, !dbg !7
  %730 = icmp samesign ult i32 %728, %729, !dbg !7
  %731 = select i1 %730, i32 %728, i32 %726, !dbg !7
  %732 = shl nuw nsw i32 %731, 2, !dbg !7
  %733 = bitcast float %724 to i32, !dbg !7
  %734 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %733), !dbg !7
  %735 = bitcast i32 %734 to float, !dbg !7
  %736 = select i1 %.not2, float %720, float %735, !dbg !7
  %737 = select i1 %.not2, float %735, float %721, !dbg !7
  %738 = select i1 %.not2, float %723, float %722, !dbg !7
  %739 = bitcast float %738 to i32, !dbg !7
  %740 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %739), !dbg !7
  %741 = bitcast i32 %740 to float, !dbg !7
  %742 = select i1 %.not2, float %722, float %741, !dbg !7
  %743 = select i1 %.not2, float %741, float %723, !dbg !7
  %744 = select i1 %.not3, float %742, float %736, !dbg !7
  %745 = xor i32 %726, 32, !dbg !7
  %746 = icmp samesign ult i32 %745, %729, !dbg !7
  %747 = select i1 %746, i32 %745, i32 %726, !dbg !7
  %748 = shl nuw nsw i32 %747, 2, !dbg !7
  %749 = bitcast float %744 to i32, !dbg !7
  %750 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %749), !dbg !7
  %751 = bitcast i32 %750 to float, !dbg !7
  %752 = select i1 %.not3, float %736, float %751, !dbg !7
  %753 = select i1 %.not3, float %751, float %742, !dbg !7
  %754 = select i1 %.not3, float %743, float %737, !dbg !7
  %755 = bitcast float %754 to i32, !dbg !7
  %756 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %755), !dbg !7
  %757 = bitcast i32 %756 to float, !dbg !7
  %758 = select i1 %.not3, float %737, float %757, !dbg !7
  %759 = select i1 %.not3, float %757, float %743, !dbg !7
  %760 = extractelement <8 x float> %57, i64 0, !dbg !7
  %761 = extractelement <8 x float> %57, i64 1, !dbg !7
  %762 = extractelement <8 x float> %57, i64 2, !dbg !7
  %763 = extractelement <8 x float> %57, i64 3, !dbg !7
  %764 = select i1 %.not2, float %761, float %760, !dbg !7
  %765 = bitcast float %764 to i32, !dbg !7
  %766 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %765), !dbg !7
  %767 = bitcast i32 %766 to float, !dbg !7
  %768 = select i1 %.not2, float %760, float %767, !dbg !7
  %769 = select i1 %.not2, float %767, float %761, !dbg !7
  %770 = select i1 %.not2, float %763, float %762, !dbg !7
  %771 = bitcast float %770 to i32, !dbg !7
  %772 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %771), !dbg !7
  %773 = bitcast i32 %772 to float, !dbg !7
  %774 = select i1 %.not2, float %762, float %773, !dbg !7
  %775 = select i1 %.not2, float %773, float %763, !dbg !7
  %776 = select i1 %.not3, float %774, float %768, !dbg !7
  %777 = bitcast float %776 to i32, !dbg !7
  %778 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %777), !dbg !7
  %779 = bitcast i32 %778 to float, !dbg !7
  %780 = select i1 %.not3, float %768, float %779, !dbg !7
  %781 = select i1 %.not3, float %779, float %774, !dbg !7
  %782 = select i1 %.not3, float %775, float %769, !dbg !7
  %783 = bitcast float %782 to i32, !dbg !7
  %784 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %783), !dbg !7
  %785 = bitcast i32 %784 to float, !dbg !7
  %786 = select i1 %.not3, float %769, float %785, !dbg !7
  %787 = select i1 %.not3, float %785, float %775, !dbg !7
  %788 = extractelement <8 x float> %54, i64 0, !dbg !7
  %789 = extractelement <8 x float> %54, i64 1, !dbg !7
  %790 = extractelement <8 x float> %54, i64 2, !dbg !7
  %791 = extractelement <8 x float> %54, i64 3, !dbg !7
  %792 = select i1 %.not2, float %789, float %788, !dbg !7
  %793 = bitcast float %792 to i32, !dbg !7
  %794 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %793), !dbg !7
  %795 = bitcast i32 %794 to float, !dbg !7
  %796 = select i1 %.not2, float %788, float %795, !dbg !7
  %797 = select i1 %.not2, float %795, float %789, !dbg !7
  %798 = select i1 %.not2, float %791, float %790, !dbg !7
  %799 = bitcast float %798 to i32, !dbg !7
  %800 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %799), !dbg !7
  %801 = bitcast i32 %800 to float, !dbg !7
  %802 = select i1 %.not2, float %790, float %801, !dbg !7
  %803 = select i1 %.not2, float %801, float %791, !dbg !7
  %804 = select i1 %.not3, float %802, float %796, !dbg !7
  %805 = bitcast float %804 to i32, !dbg !7
  %806 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %805), !dbg !7
  %807 = bitcast i32 %806 to float, !dbg !7
  %808 = select i1 %.not3, float %796, float %807, !dbg !7
  %809 = select i1 %.not3, float %807, float %802, !dbg !7
  %810 = select i1 %.not3, float %803, float %797, !dbg !7
  %811 = bitcast float %810 to i32, !dbg !7
  %812 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %811), !dbg !7
  %813 = bitcast i32 %812 to float, !dbg !7
  %814 = select i1 %.not3, float %797, float %813, !dbg !7
  %815 = select i1 %.not3, float %813, float %803, !dbg !7
  %816 = extractelement <8 x float> %49, i64 0, !dbg !7
  %817 = extractelement <8 x float> %49, i64 1, !dbg !7
  %818 = extractelement <8 x float> %49, i64 2, !dbg !7
  %819 = extractelement <8 x float> %49, i64 3, !dbg !7
  %820 = select i1 %.not2, float %817, float %816, !dbg !7
  %821 = bitcast float %820 to i32, !dbg !7
  %822 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %821), !dbg !7
  %823 = bitcast i32 %822 to float, !dbg !7
  %824 = select i1 %.not2, float %816, float %823, !dbg !7
  %825 = select i1 %.not2, float %823, float %817, !dbg !7
  %826 = select i1 %.not2, float %819, float %818, !dbg !7
  %827 = bitcast float %826 to i32, !dbg !7
  %828 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %827), !dbg !7
  %829 = bitcast i32 %828 to float, !dbg !7
  %830 = select i1 %.not2, float %818, float %829, !dbg !7
  %831 = select i1 %.not2, float %829, float %819, !dbg !7
  %832 = select i1 %.not3, float %830, float %824, !dbg !7
  %833 = bitcast float %832 to i32, !dbg !7
  %834 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %833), !dbg !7
  %835 = select i1 %.not3, float %831, float %825, !dbg !7
  %836 = bitcast float %835 to i32, !dbg !7
  %837 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %836), !dbg !7
  %838 = shufflevector <1 x float> %50, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %839 = extractelement <8 x float> %60, i64 4, !dbg !7
  %840 = extractelement <8 x float> %60, i64 5, !dbg !7
  %841 = extractelement <8 x float> %60, i64 6, !dbg !7
  %842 = extractelement <8 x float> %60, i64 7, !dbg !7
  %843 = select i1 %.not2, float %840, float %839, !dbg !7
  %844 = bitcast float %843 to i32, !dbg !7
  %845 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %844), !dbg !7
  %846 = bitcast i32 %845 to float, !dbg !7
  %847 = select i1 %.not2, float %839, float %846, !dbg !7
  %848 = select i1 %.not2, float %846, float %840, !dbg !7
  %849 = select i1 %.not2, float %842, float %841, !dbg !7
  %850 = bitcast float %849 to i32, !dbg !7
  %851 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %850), !dbg !7
  %852 = bitcast i32 %851 to float, !dbg !7
  %853 = select i1 %.not2, float %841, float %852, !dbg !7
  %854 = select i1 %.not2, float %852, float %842, !dbg !7
  %855 = select i1 %.not3, float %853, float %847, !dbg !7
  %856 = bitcast float %855 to i32, !dbg !7
  %857 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %856), !dbg !7
  %858 = bitcast i32 %857 to float, !dbg !7
  %859 = select i1 %.not3, float %847, float %858, !dbg !7
  %860 = select i1 %.not3, float %858, float %853, !dbg !7
  %861 = select i1 %.not3, float %854, float %848, !dbg !7
  %862 = bitcast float %861 to i32, !dbg !7
  %863 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %862), !dbg !7
  %864 = bitcast i32 %863 to float, !dbg !7
  %865 = select i1 %.not3, float %848, float %864, !dbg !7
  %866 = select i1 %.not3, float %864, float %854, !dbg !7
  %867 = extractelement <8 x float> %57, i64 4, !dbg !7
  %868 = extractelement <8 x float> %57, i64 5, !dbg !7
  %869 = extractelement <8 x float> %57, i64 6, !dbg !7
  %870 = extractelement <8 x float> %57, i64 7, !dbg !7
  %871 = select i1 %.not2, float %868, float %867, !dbg !7
  %872 = bitcast float %871 to i32, !dbg !7
  %873 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %872), !dbg !7
  %874 = bitcast i32 %873 to float, !dbg !7
  %875 = select i1 %.not2, float %867, float %874, !dbg !7
  %876 = select i1 %.not2, float %874, float %868, !dbg !7
  %877 = select i1 %.not2, float %870, float %869, !dbg !7
  %878 = bitcast float %877 to i32, !dbg !7
  %879 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %878), !dbg !7
  %880 = bitcast i32 %879 to float, !dbg !7
  %881 = select i1 %.not2, float %869, float %880, !dbg !7
  %882 = select i1 %.not2, float %880, float %870, !dbg !7
  %883 = select i1 %.not3, float %881, float %875, !dbg !7
  %884 = bitcast float %883 to i32, !dbg !7
  %885 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %884), !dbg !7
  %886 = bitcast i32 %885 to float, !dbg !7
  %887 = select i1 %.not3, float %875, float %886, !dbg !7
  %888 = select i1 %.not3, float %886, float %881, !dbg !7
  %889 = select i1 %.not3, float %882, float %876, !dbg !7
  %890 = bitcast float %889 to i32, !dbg !7
  %891 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %890), !dbg !7
  %892 = bitcast i32 %891 to float, !dbg !7
  %893 = select i1 %.not3, float %876, float %892, !dbg !7
  %894 = select i1 %.not3, float %892, float %882, !dbg !7
  %895 = extractelement <8 x float> %54, i64 4, !dbg !7
  %896 = extractelement <8 x float> %54, i64 5, !dbg !7
  %897 = extractelement <8 x float> %54, i64 6, !dbg !7
  %898 = extractelement <8 x float> %54, i64 7, !dbg !7
  %899 = select i1 %.not2, float %896, float %895, !dbg !7
  %900 = bitcast float %899 to i32, !dbg !7
  %901 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %900), !dbg !7
  %902 = bitcast i32 %901 to float, !dbg !7
  %903 = select i1 %.not2, float %895, float %902, !dbg !7
  %904 = select i1 %.not2, float %902, float %896, !dbg !7
  %905 = select i1 %.not2, float %898, float %897, !dbg !7
  %906 = bitcast float %905 to i32, !dbg !7
  %907 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %906), !dbg !7
  %908 = bitcast i32 %907 to float, !dbg !7
  %909 = select i1 %.not2, float %897, float %908, !dbg !7
  %910 = select i1 %.not2, float %908, float %898, !dbg !7
  %911 = select i1 %.not3, float %909, float %903, !dbg !7
  %912 = bitcast float %911 to i32, !dbg !7
  %913 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %912), !dbg !7
  %914 = bitcast i32 %913 to float, !dbg !7
  %915 = select i1 %.not3, float %903, float %914, !dbg !7
  %916 = select i1 %.not3, float %914, float %909, !dbg !7
  %917 = select i1 %.not3, float %910, float %904, !dbg !7
  %918 = bitcast float %917 to i32, !dbg !7
  %919 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %918), !dbg !7
  %920 = bitcast i32 %919 to float, !dbg !7
  %921 = select i1 %.not3, float %904, float %920, !dbg !7
  %922 = select i1 %.not3, float %920, float %910, !dbg !7
  %923 = extractelement <8 x float> %49, i64 4, !dbg !7
  %924 = extractelement <8 x float> %49, i64 5, !dbg !7
  %925 = extractelement <8 x float> %49, i64 6, !dbg !7
  %926 = extractelement <8 x float> %49, i64 7, !dbg !7
  %927 = select i1 %.not2, float %924, float %923, !dbg !7
  %928 = bitcast float %927 to i32, !dbg !7
  %929 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %928), !dbg !7
  %930 = bitcast i32 %929 to float, !dbg !7
  %931 = select i1 %.not2, float %923, float %930, !dbg !7
  %932 = select i1 %.not2, float %930, float %924, !dbg !7
  %933 = select i1 %.not2, float %926, float %925, !dbg !7
  %934 = bitcast float %933 to i32, !dbg !7
  %935 = call i32 @llvm.amdgcn.ds.bpermute(i32 %732, i32 %934), !dbg !7
  %936 = bitcast i32 %935 to float, !dbg !7
  %937 = select i1 %.not2, float %925, float %936, !dbg !7
  %938 = select i1 %.not2, float %936, float %926, !dbg !7
  %939 = select i1 %.not3, float %937, float %931, !dbg !7
  %940 = bitcast float %939 to i32, !dbg !7
  %941 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %940), !dbg !7
  %942 = select i1 %.not3, float %938, float %932, !dbg !7
  %943 = bitcast float %942 to i32, !dbg !7
  %944 = call i32 @llvm.amdgcn.ds.bpermute(i32 %748, i32 %943), !dbg !7
  br i1 true, label %945, label %977, !dbg !21

945:                                              ; preds = %717
  %946 = bitcast i32 %941 to float, !dbg !7
  %947 = select i1 %.not3, float %931, float %946, !dbg !7
  %948 = insertelement <4 x float> poison, float %947, i64 0, !dbg !19
  %949 = bitcast i32 %944 to float, !dbg !7
  %950 = select i1 %.not3, float %932, float %949, !dbg !7
  %951 = insertelement <4 x float> %948, float %950, i64 1, !dbg !19
  %952 = select i1 %.not3, float %946, float %937, !dbg !7
  %953 = insertelement <4 x float> %951, float %952, i64 2, !dbg !19
  %954 = select i1 %.not3, float %949, float %938, !dbg !7
  %955 = insertelement <4 x float> %953, float %954, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %838, !dbg !19
  %normalized = fmul <4 x float> %955, %row.reciprocal, !dbg !19
  %956 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %957 = extractelement <4 x half> %956, i64 3, !dbg !20
  %958 = extractelement <4 x half> %956, i64 2, !dbg !20
  %959 = extractelement <4 x half> %956, i64 1, !dbg !20
  %960 = extractelement <4 x half> %956, i64 0, !dbg !20
  %961 = bitcast i32 %834 to float, !dbg !7
  %962 = select i1 %.not3, float %824, float %961, !dbg !7
  %963 = insertelement <4 x float> poison, float %962, i64 0, !dbg !19
  %964 = bitcast i32 %837 to float, !dbg !7
  %965 = select i1 %.not3, float %825, float %964, !dbg !7
  %966 = insertelement <4 x float> %963, float %965, i64 1, !dbg !19
  %967 = select i1 %.not3, float %961, float %830, !dbg !7
  %968 = insertelement <4 x float> %966, float %967, i64 2, !dbg !19
  %969 = select i1 %.not3, float %964, float %831, !dbg !7
  %970 = insertelement <4 x float> %968, float %969, i64 3, !dbg !19
  %normalized23 = fmul <4 x float> %970, %row.reciprocal, !dbg !19
  %971 = fptrunc <4 x float> %normalized23 to <4 x half>, !dbg !20
  %972 = extractelement <4 x half> %971, i64 3, !dbg !20
  %973 = extractelement <4 x half> %971, i64 2, !dbg !20
  %974 = extractelement <4 x half> %971, i64 1, !dbg !20
  %975 = extractelement <4 x half> %971, i64 0, !dbg !20
  %976 = shl nuw nsw i64 %17, 19, !dbg !21
  %address.thread280 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local281 = zext nneg i32 %address.thread280 to i64, !dbg !13
  %address.local282 = add nsw i64 %18, %address.local281, !dbg !21
  %address.local283 = lshr i64 %address.local281, 6, !dbg !13
  %address.local284 = shl nuw nsw i64 %address.local283, 4, !dbg !21
  %address.local285 = add nsw i64 %address.local282, %address.local284, !dbg !21
  %address.local286 = and i64 %address.local281, 240, !dbg !21
  %address.local287 = sub nsw i64 %address.local285, %address.local286, !dbg !21
  %address.local288 = add nsw i64 %address.local287, %19, !dbg !21
  %address.local289 = shl nuw nsw i64 %address.local288, 7, !dbg !21
  %address.local290 = add nuw nsw i64 %976, %address.local289, !dbg !21
  %address.local291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local292 = lshr i64 %address.local281, 2, !dbg !13
  %address.local293 = and i64 %address.local292, 12, !dbg !13
  %address.local294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local291, i64 %address.local293, !dbg !21
  store half %975, ptr addrspace(1) %address.local294, align 2, !dbg !21
  %address.local295 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local295, i64 %address.local293, !dbg !21
  %address.local297 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local296, i64 2, !dbg !21
  store half %974, ptr addrspace(1) %address.local297, align 2, !dbg !21
  %address.local298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local298, i64 %address.local293, !dbg !21
  %address.local300 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local299, i64 4, !dbg !21
  store half %973, ptr addrspace(1) %address.local300, align 2, !dbg !21
  %address.local301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local301, i64 %address.local293, !dbg !21
  %address.local303 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local302, i64 6, !dbg !21
  store half %972, ptr addrspace(1) %address.local303, align 2, !dbg !21
  %address.local304 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local305 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local304, i64 %address.local293, !dbg !21
  %address.local306 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local305, i64 32, !dbg !21
  store half %960, ptr addrspace(1) %address.local306, align 2, !dbg !21
  %address.local307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local307, i64 %address.local293, !dbg !21
  %address.local309 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local308, i64 34, !dbg !21
  store half %959, ptr addrspace(1) %address.local309, align 2, !dbg !21
  %address.local310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local310, i64 %address.local293, !dbg !21
  %address.local312 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local311, i64 36, !dbg !21
  store half %958, ptr addrspace(1) %address.local312, align 2, !dbg !21
  %address.local313 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local290, !dbg !21
  %address.local314 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local313, i64 %address.local293, !dbg !21
  %address.local315 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local314, i64 38, !dbg !21
  store half %957, ptr addrspace(1) %address.local315, align 2, !dbg !21
  br label %977, !dbg !21

977:                                              ; preds = %945, %717
  br i1 true, label %978, label %998, !dbg !21

978:                                              ; preds = %977
  %979 = insertelement <4 x float> poison, float %915, i64 0, !dbg !19
  %980 = insertelement <4 x float> %979, float %921, i64 1, !dbg !19
  %981 = insertelement <4 x float> %980, float %916, i64 2, !dbg !19
  %982 = insertelement <4 x float> %981, float %922, i64 3, !dbg !19
  %row.reciprocal24 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %838, !dbg !19
  %normalized25 = fmul <4 x float> %982, %row.reciprocal24, !dbg !19
  %983 = fptrunc <4 x float> %normalized25 to <4 x half>, !dbg !20
  %984 = extractelement <4 x half> %983, i64 3, !dbg !20
  %985 = extractelement <4 x half> %983, i64 2, !dbg !20
  %986 = extractelement <4 x half> %983, i64 1, !dbg !20
  %987 = extractelement <4 x half> %983, i64 0, !dbg !20
  %988 = insertelement <4 x float> poison, float %808, i64 0, !dbg !19
  %989 = insertelement <4 x float> %988, float %814, i64 1, !dbg !19
  %990 = insertelement <4 x float> %989, float %809, i64 2, !dbg !19
  %991 = insertelement <4 x float> %990, float %815, i64 3, !dbg !19
  %normalized26 = fmul <4 x float> %991, %row.reciprocal24, !dbg !19
  %992 = fptrunc <4 x float> %normalized26 to <4 x half>, !dbg !20
  %993 = extractelement <4 x half> %992, i64 3, !dbg !20
  %994 = extractelement <4 x half> %992, i64 2, !dbg !20
  %995 = extractelement <4 x half> %992, i64 1, !dbg !20
  %996 = extractelement <4 x half> %992, i64 0, !dbg !20
  %997 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread316 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local317 = zext nneg i32 %address.thread316 to i64, !dbg !13
  %address.local318 = add nsw i64 %18, %address.local317, !dbg !21
  %address.local319 = lshr i64 %address.local317, 6, !dbg !13
  %address.local320 = shl nuw nsw i64 %address.local319, 4, !dbg !21
  %address.local321 = add nsw i64 %address.local318, %address.local320, !dbg !21
  %address.local322 = and i64 %address.local317, 240, !dbg !21
  %address.local323 = sub nsw i64 %address.local321, %address.local322, !dbg !21
  %address.local324 = add nsw i64 %address.local323, %19, !dbg !21
  %address.local325 = shl nuw nsw i64 %address.local324, 7, !dbg !21
  %address.local326 = add nuw nsw i64 %997, %address.local325, !dbg !21
  %address.local327 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local326, !dbg !21
  %address.local328 = lshr i64 %address.local317, 2, !dbg !13
  %address.local329 = and i64 %address.local328, 12, !dbg !13
  %address.local330 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local327, i64 %address.local329, !dbg !21
  %address.local331 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local330, i64 64, !dbg !21
  store half %996, ptr addrspace(1) %address.local331, align 2, !dbg !21
  %address.local332 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local326, !dbg !21
  %address.local333 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local332, i64 %address.local329, !dbg !21
  %address.local334 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local333, i64 66, !dbg !21
  store half %995, ptr addrspace(1) %address.local334, align 2, !dbg !21
  %address.local335 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local326, !dbg !21
  %address.local336 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local335, i64 %address.local329, !dbg !21
  %address.local337 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local336, i64 68, !dbg !21
  store half %994, ptr addrspace(1) %address.local337, align 2, !dbg !21
  %address.local338 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local326, !dbg !21
  %address.local339 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local338, i64 %address.local329, !dbg !21
  %address.local340 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local339, i64 70, !dbg !21
  store half %993, ptr addrspace(1) %address.local340, align 2, !dbg !21
  %address.local341 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local326, !dbg !21
  %address.local342 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local341, i64 %address.local329, !dbg !21
  %address.local343 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local342, i64 96, !dbg !21
  store half %987, ptr addrspace(1) %address.local343, align 2, !dbg !21
  %address.local344 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local326, !dbg !21
  %address.local345 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local344, i64 %address.local329, !dbg !21
  %address.local346 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local345, i64 98, !dbg !21
  store half %986, ptr addrspace(1) %address.local346, align 2, !dbg !21
  %address.local347 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local326, !dbg !21
  %address.local348 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local347, i64 %address.local329, !dbg !21
  %address.local349 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local348, i64 100, !dbg !21
  store half %985, ptr addrspace(1) %address.local349, align 2, !dbg !21
  %address.local350 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local326, !dbg !21
  %address.local351 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local350, i64 %address.local329, !dbg !21
  %address.local352 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local351, i64 102, !dbg !21
  store half %984, ptr addrspace(1) %address.local352, align 2, !dbg !21
  br label %998, !dbg !21

998:                                              ; preds = %978, %977
  br i1 true, label %999, label %1019, !dbg !21

999:                                              ; preds = %998
  %1000 = insertelement <4 x float> poison, float %887, i64 0, !dbg !19
  %1001 = insertelement <4 x float> %1000, float %893, i64 1, !dbg !19
  %1002 = insertelement <4 x float> %1001, float %888, i64 2, !dbg !19
  %1003 = insertelement <4 x float> %1002, float %894, i64 3, !dbg !19
  %row.reciprocal27 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %838, !dbg !19
  %normalized28 = fmul <4 x float> %1003, %row.reciprocal27, !dbg !19
  %1004 = fptrunc <4 x float> %normalized28 to <4 x half>, !dbg !20
  %1005 = extractelement <4 x half> %1004, i64 3, !dbg !20
  %1006 = extractelement <4 x half> %1004, i64 2, !dbg !20
  %1007 = extractelement <4 x half> %1004, i64 1, !dbg !20
  %1008 = extractelement <4 x half> %1004, i64 0, !dbg !20
  %1009 = insertelement <4 x float> poison, float %780, i64 0, !dbg !19
  %1010 = insertelement <4 x float> %1009, float %786, i64 1, !dbg !19
  %1011 = insertelement <4 x float> %1010, float %781, i64 2, !dbg !19
  %1012 = insertelement <4 x float> %1011, float %787, i64 3, !dbg !19
  %normalized29 = fmul <4 x float> %1012, %row.reciprocal27, !dbg !19
  %1013 = fptrunc <4 x float> %normalized29 to <4 x half>, !dbg !20
  %1014 = extractelement <4 x half> %1013, i64 3, !dbg !20
  %1015 = extractelement <4 x half> %1013, i64 2, !dbg !20
  %1016 = extractelement <4 x half> %1013, i64 1, !dbg !20
  %1017 = extractelement <4 x half> %1013, i64 0, !dbg !20
  %1018 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread353 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local354 = zext nneg i32 %address.thread353 to i64, !dbg !13
  %address.local355 = add nsw i64 %18, %address.local354, !dbg !21
  %address.local356 = lshr i64 %address.local354, 6, !dbg !13
  %address.local357 = shl nuw nsw i64 %address.local356, 4, !dbg !21
  %address.local358 = add nsw i64 %address.local355, %address.local357, !dbg !21
  %address.local359 = and i64 %address.local354, 240, !dbg !21
  %address.local360 = sub nsw i64 %address.local358, %address.local359, !dbg !21
  %address.local361 = add nsw i64 %address.local360, %19, !dbg !21
  %address.local362 = shl nuw nsw i64 %address.local361, 7, !dbg !21
  %address.local363 = add nuw nsw i64 %1018, %address.local362, !dbg !21
  %address.local364 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local363, !dbg !21
  %address.local365 = lshr i64 %address.local354, 2, !dbg !13
  %address.local366 = and i64 %address.local365, 12, !dbg !13
  %address.local367 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local364, i64 %address.local366, !dbg !21
  %address.local368 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local367, i64 128, !dbg !21
  store half %1017, ptr addrspace(1) %address.local368, align 2, !dbg !21
  %address.local369 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local363, !dbg !21
  %address.local370 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local369, i64 %address.local366, !dbg !21
  %address.local371 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local370, i64 130, !dbg !21
  store half %1016, ptr addrspace(1) %address.local371, align 2, !dbg !21
  %address.local372 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local363, !dbg !21
  %address.local373 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local372, i64 %address.local366, !dbg !21
  %address.local374 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local373, i64 132, !dbg !21
  store half %1015, ptr addrspace(1) %address.local374, align 2, !dbg !21
  %address.local375 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local363, !dbg !21
  %address.local376 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local375, i64 %address.local366, !dbg !21
  %address.local377 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local376, i64 134, !dbg !21
  store half %1014, ptr addrspace(1) %address.local377, align 2, !dbg !21
  %address.local378 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local363, !dbg !21
  %address.local379 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local378, i64 %address.local366, !dbg !21
  %address.local380 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local379, i64 160, !dbg !21
  store half %1008, ptr addrspace(1) %address.local380, align 2, !dbg !21
  %address.local381 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local363, !dbg !21
  %address.local382 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local381, i64 %address.local366, !dbg !21
  %address.local383 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local382, i64 162, !dbg !21
  store half %1007, ptr addrspace(1) %address.local383, align 2, !dbg !21
  %address.local384 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local363, !dbg !21
  %address.local385 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local384, i64 %address.local366, !dbg !21
  %address.local386 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local385, i64 164, !dbg !21
  store half %1006, ptr addrspace(1) %address.local386, align 2, !dbg !21
  %address.local387 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local363, !dbg !21
  %address.local388 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local387, i64 %address.local366, !dbg !21
  %address.local389 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local388, i64 166, !dbg !21
  store half %1005, ptr addrspace(1) %address.local389, align 2, !dbg !21
  br label %1019, !dbg !21

1019:                                             ; preds = %999, %998
  br i1 true, label %1020, label %1040, !dbg !21

1020:                                             ; preds = %1019
  %1021 = insertelement <4 x float> poison, float %859, i64 0, !dbg !19
  %1022 = insertelement <4 x float> %1021, float %865, i64 1, !dbg !19
  %1023 = insertelement <4 x float> %1022, float %860, i64 2, !dbg !19
  %1024 = insertelement <4 x float> %1023, float %866, i64 3, !dbg !19
  %row.reciprocal30 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %838, !dbg !19
  %normalized31 = fmul <4 x float> %1024, %row.reciprocal30, !dbg !19
  %1025 = fptrunc <4 x float> %normalized31 to <4 x half>, !dbg !20
  %1026 = extractelement <4 x half> %1025, i64 3, !dbg !20
  %1027 = extractelement <4 x half> %1025, i64 2, !dbg !20
  %1028 = extractelement <4 x half> %1025, i64 1, !dbg !20
  %1029 = extractelement <4 x half> %1025, i64 0, !dbg !20
  %1030 = insertelement <4 x float> poison, float %752, i64 0, !dbg !19
  %1031 = insertelement <4 x float> %1030, float %758, i64 1, !dbg !19
  %1032 = insertelement <4 x float> %1031, float %753, i64 2, !dbg !19
  %1033 = insertelement <4 x float> %1032, float %759, i64 3, !dbg !19
  %normalized32 = fmul <4 x float> %1033, %row.reciprocal30, !dbg !19
  %1034 = fptrunc <4 x float> %normalized32 to <4 x half>, !dbg !20
  %1035 = extractelement <4 x half> %1034, i64 3, !dbg !20
  %1036 = extractelement <4 x half> %1034, i64 2, !dbg !20
  %1037 = extractelement <4 x half> %1034, i64 1, !dbg !20
  %1038 = extractelement <4 x half> %1034, i64 0, !dbg !20
  %1039 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread390 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local391 = zext nneg i32 %address.thread390 to i64, !dbg !13
  %address.local392 = add nsw i64 %18, %address.local391, !dbg !21
  %address.local393 = lshr i64 %address.local391, 6, !dbg !13
  %address.local394 = shl nuw nsw i64 %address.local393, 4, !dbg !21
  %address.local395 = add nsw i64 %address.local392, %address.local394, !dbg !21
  %address.local396 = and i64 %address.local391, 240, !dbg !21
  %address.local397 = sub nsw i64 %address.local395, %address.local396, !dbg !21
  %address.local398 = add nsw i64 %address.local397, %19, !dbg !21
  %address.local399 = shl nuw nsw i64 %address.local398, 7, !dbg !21
  %address.local400 = add nuw nsw i64 %1039, %address.local399, !dbg !21
  %address.local401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local400, !dbg !21
  %address.local402 = lshr i64 %address.local391, 2, !dbg !13
  %address.local403 = and i64 %address.local402, 12, !dbg !13
  %address.local404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local401, i64 %address.local403, !dbg !21
  %address.local405 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local404, i64 192, !dbg !21
  store half %1038, ptr addrspace(1) %address.local405, align 2, !dbg !21
  %address.local406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local400, !dbg !21
  %address.local407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local406, i64 %address.local403, !dbg !21
  %address.local408 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local407, i64 194, !dbg !21
  store half %1037, ptr addrspace(1) %address.local408, align 2, !dbg !21
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local400, !dbg !21
  %address.local410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local409, i64 %address.local403, !dbg !21
  %address.local411 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local410, i64 196, !dbg !21
  store half %1036, ptr addrspace(1) %address.local411, align 2, !dbg !21
  %address.local412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local400, !dbg !21
  %address.local413 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local412, i64 %address.local403, !dbg !21
  %address.local414 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local413, i64 198, !dbg !21
  store half %1035, ptr addrspace(1) %address.local414, align 2, !dbg !21
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local400, !dbg !21
  %address.local416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local415, i64 %address.local403, !dbg !21
  %address.local417 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local416, i64 224, !dbg !21
  store half %1029, ptr addrspace(1) %address.local417, align 2, !dbg !21
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local400, !dbg !21
  %address.local419 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local418, i64 %address.local403, !dbg !21
  %address.local420 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local419, i64 226, !dbg !21
  store half %1028, ptr addrspace(1) %address.local420, align 2, !dbg !21
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local400, !dbg !21
  %address.local422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local421, i64 %address.local403, !dbg !21
  %address.local423 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local422, i64 228, !dbg !21
  store half %1027, ptr addrspace(1) %address.local423, align 2, !dbg !21
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local400, !dbg !21
  %address.local425 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local424, i64 %address.local403, !dbg !21
  %address.local426 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local425, i64 230, !dbg !21
  store half %1026, ptr addrspace(1) %address.local426, align 2, !dbg !21
  br label %1040, !dbg !21

1040:                                             ; preds = %1020, %1019
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/pipeline-loop-only")
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
