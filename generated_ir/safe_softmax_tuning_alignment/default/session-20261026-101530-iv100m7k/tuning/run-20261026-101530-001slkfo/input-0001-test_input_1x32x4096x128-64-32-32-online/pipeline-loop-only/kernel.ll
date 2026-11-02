; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/pipeline-loop-only/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/pipeline-loop-only/input.mlir"

@shm_0 = addrspace(3) global [12288 x i8] undef, align 16

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
  %29 = lshr i64 %27, 2, !dbg !13
  %30 = and i64 %29, 12, !dbg !13
  %31 = and i32 %26, 16, !dbg !14
  %.not = icmp eq i32 %31, 0, !dbg !14
  %32 = and i32 %26, 32, !dbg !14
  %.not2 = icmp eq i32 %32, 0, !dbg !14
  %33 = or disjoint i64 %11, 63, !dbg !15
  %34 = add nsw i64 %11, 64, !dbg !15
  %35 = icmp uge i64 %34, %33, !dbg !15
  %36 = shl nsw i64 %10, 1, !dbg !7
  %37 = add nsw i64 %36, 2, !dbg !7
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

51:                                               ; preds = %740, %4
  %52 = phi i64 [ %770, %740 ], [ 0, %4 ], !dbg !7
  %53 = phi <4 x float> [ %651, %740 ], [ zeroinitializer, %4 ], !dbg !7
  %54 = phi <4 x float> [ %656, %740 ], [ zeroinitializer, %4 ], !dbg !7
  %55 = phi <8 x float> [ %765, %740 ], [ zeroinitializer, %4 ], !dbg !7
  %56 = phi <1 x float> [ %766, %740 ], [ zeroinitializer, %4 ], !dbg !7
  %57 = phi <1 x float> [ %553, %740 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %58 = phi <4 x float> [ %698, %740 ], [ zeroinitializer, %4 ], !dbg !7
  %59 = phi <4 x float> [ %703, %740 ], [ zeroinitializer, %4 ], !dbg !7
  %60 = phi <8 x float> [ %767, %740 ], [ zeroinitializer, %4 ], !dbg !7
  %61 = phi <4 x float> [ %745, %740 ], [ zeroinitializer, %4 ], !dbg !7
  %62 = phi <4 x float> [ %750, %740 ], [ zeroinitializer, %4 ], !dbg !7
  %63 = phi <8 x float> [ %768, %740 ], [ zeroinitializer, %4 ], !dbg !7
  %64 = phi <4 x float> [ %759, %740 ], [ zeroinitializer, %4 ], !dbg !7
  %65 = phi <4 x float> [ %764, %740 ], [ zeroinitializer, %4 ], !dbg !7
  %66 = phi <8 x float> [ %769, %740 ], [ zeroinitializer, %4 ], !dbg !7
  %67 = icmp slt i64 %52, %37, !dbg !7
  br i1 %67, label %68, label %771, !dbg !7

68:                                               ; preds = %51
  %69 = shl nsw i64 %52, 5, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !12, !range !40, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local34 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local35 = and i64 %address.local34, 1536, !dbg !12
  %address.local36 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local35, !dbg !12
  %address.local37 = ptrtoint ptr addrspace(3) %address.local36 to i64, !dbg !12
  %address.local38 = trunc i64 %address.local37 to i32, !dbg !12
  %70 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local38), !dbg !12
  %71 = zext i32 %70 to i64, !dbg !12
  %72 = inttoptr i64 %71 to ptr addrspace(3), !dbg !12
  %address.local39 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local40 = and i64 %address.local39, 8064, !dbg !12
  %address.local41 = lshr i64 %address.local, 3, !dbg !12
  %address.local42 = and i64 %address.local41, 24, !dbg !12
  %address.local43 = or disjoint i64 %address.local40, %address.local42, !dbg !12
  %address.local44 = getelementptr [2 x i8], ptr addrspace(1) %39, i64 %address.local43, !dbg !12
  %73 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %74 = ptrtoint ptr addrspace(1) %address.local44 to i64, !dbg !12
  %75 = sub i64 %74, %73, !dbg !12
  %dma.buffer.offset551 = trunc i64 %75 to i32, !dbg !12
  %76 = trunc i64 %73 to i32, !dbg !12
  %77 = insertelement <4 x i32> poison, i32 %76, i64 0, !dbg !12
  %78 = lshr i64 %73, 32, !dbg !12
  %79 = trunc i64 %78 to i32, !dbg !12
  %80 = insertelement <4 x i32> %77, i32 %79, i64 1, !dbg !12
  %81 = insertelement <4 x i32> %80, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc552 = insertelement <4 x i32> %81, i32 131072, i64 3, !dbg !12
  %82 = ptrtoint ptr addrspace(3) %72 to i32, !dbg !12
  %dma.buffer.lds553 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %82), !dbg !12
  %83 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds553, i32 %dma.buffer.offset551, <4 x i32> %dma.buffer.rsrc552) #10, !dbg !12
  %84 = lshr i64 %52, 14, !dbg !17
  %85 = add nsw i64 %84, %8, !dbg !17
  %86 = lshr i64 %52, 7, !dbg !17
  %87 = and i64 %86, 127, !dbg !17
  %88 = shl i64 %85, 19, !dbg !17
  %89 = and i64 %88, 16252928, !dbg !17
  %90 = shl i64 %86, 19, !dbg !17
  %91 = shl i64 %52, 12, !dbg !17
  %92 = sub i64 %91, %90, !dbg !17
  %93 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %89, !dbg !17
  %94 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %87, !dbg !17
  %95 = getelementptr [2 x i8], ptr addrspace(1) %94, i64 %92, !dbg !17
  br label %96, !dbg !17

96:                                               ; preds = %99, %68
  %97 = phi i64 [ %125, %99 ], [ %27, %68 ], !dbg !17
  %98 = icmp samesign ult i64 %97, 128, !dbg !17
  br i1 %98, label %99, label %126, !dbg !17

99:                                               ; preds = %96
  %100 = lshr i64 %97, 2, !dbg !17
  %101 = and i64 %100, 24, !dbg !17
  %102 = shl nuw nsw i64 %97, 3, !dbg !17
  %103 = and i64 %102, 512, !dbg !17
  %104 = shl nuw nsw i64 %97, 7, !dbg !17
  %105 = and i64 %104, 3968, !dbg !17
  %106 = getelementptr [2 x i8], ptr addrspace(1) %95, i64 %101, !dbg !17
  %107 = getelementptr [2 x i8], ptr addrspace(1) %106, i64 %105, !dbg !17
  %108 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %103, !dbg !17
  %109 = ptrtoint ptr addrspace(3) %108 to i64, !dbg !17
  %110 = trunc i64 %109 to i32, !dbg !17
  %111 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %110), !dbg !17
  %112 = zext i32 %111 to i64, !dbg !17
  %113 = inttoptr i64 %112 to ptr addrspace(3), !dbg !17
  %114 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %115 = ptrtoint ptr addrspace(1) %107 to i64, !dbg !17
  %116 = sub i64 %115, %114, !dbg !17
  %dma.buffer.offset = trunc i64 %116 to i32, !dbg !17
  %117 = trunc i64 %114 to i32, !dbg !17
  %118 = insertelement <4 x i32> poison, i32 %117, i64 0, !dbg !17
  %119 = lshr i64 %114, 32, !dbg !17
  %120 = trunc i64 %119 to i32, !dbg !17
  %121 = insertelement <4 x i32> %118, i32 %120, i64 1, !dbg !17
  %122 = insertelement <4 x i32> %121, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %122, i32 131072, i64 3, !dbg !17
  %123 = ptrtoint ptr addrspace(3) %113 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %123), !dbg !17
  %124 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %125 = or disjoint i64 %27, 256, !dbg !17
  br label %96, !dbg !17

126:                                              ; preds = %96
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %address.thread45 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !12, !range !40, !frisk.register_lifetimes !1
  %address.local46 = zext nneg i32 %address.thread45 to i64, !dbg !13
  %address.local47 = shl nuw nsw i64 %address.local46, 3, !dbg !12
  %address.local48 = and i64 %address.local47, 1536, !dbg !12
  %address.local49 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local48, !dbg !12
  %address.local50 = ptrtoint ptr addrspace(3) %address.local49 to i64, !dbg !12
  %address.local51 = trunc i64 %address.local50 to i32, !dbg !12
  %127 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local51), !dbg !12
  %128 = zext i32 %127 to i64, !dbg !12
  %129 = inttoptr i64 %128 to ptr addrspace(3), !dbg !12
  %address.local52 = shl nuw nsw i64 %address.local46, 7, !dbg !12
  %address.local53 = and i64 %address.local52, 8064, !dbg !12
  %address.local54 = lshr i64 %address.local46, 3, !dbg !12
  %address.local55 = and i64 %address.local54, 24, !dbg !12
  %address.local56 = or disjoint i64 %address.local53, %address.local55, !dbg !12
  %address.local57 = getelementptr [2 x i8], ptr addrspace(1) %41, i64 %address.local56, !dbg !12
  %130 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %131 = ptrtoint ptr addrspace(1) %address.local57 to i64, !dbg !12
  %132 = sub i64 %131, %130, !dbg !12
  %dma.buffer.offset548 = trunc i64 %132 to i32, !dbg !12
  %133 = trunc i64 %130 to i32, !dbg !12
  %134 = insertelement <4 x i32> poison, i32 %133, i64 0, !dbg !12
  %135 = lshr i64 %130, 32, !dbg !12
  %136 = trunc i64 %135 to i32, !dbg !12
  %137 = insertelement <4 x i32> %134, i32 %136, i64 1, !dbg !12
  %138 = insertelement <4 x i32> %137, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc549 = insertelement <4 x i32> %138, i32 131072, i64 3, !dbg !12
  %139 = ptrtoint ptr addrspace(3) %129 to i32, !dbg !12
  %dma.buffer.lds550 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %139), !dbg !12
  %140 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds550, i32 %dma.buffer.offset548, <4 x i32> %dma.buffer.rsrc549) #10, !dbg !12
  %141 = add i64 %52, %42, !dbg !17
  %142 = add i64 %141, %43, !dbg !17
  %143 = add i64 %142, 4096, !dbg !17
  %144 = icmp slt i64 %143, 0, !dbg !17
  %145 = sub i64 -4097, %142, !dbg !17
  %146 = select i1 %144, i64 %145, i64 %143, !dbg !17
  %147 = sdiv i64 %146, 16384, !dbg !17
  %.lobit3 = ashr i64 %143, 63, !dbg !17
  %148 = xor i64 %147, %.lobit3, !dbg !17
  %149 = add nsw i64 %86, %46, !dbg !17
  %150 = add nsw i64 %149, 32, !dbg !17
  %151 = icmp slt i64 %149, -32, !dbg !17
  %152 = sub nsw i64 -33, %149, !dbg !17
  %153 = select i1 %151, i64 %152, i64 %150, !dbg !17
  %154 = sdiv i64 %153, 128, !dbg !17
  %155 = sext i1 %151 to i64, !dbg !17
  %156 = xor i64 %154, %155, !dbg !17
  %157 = shl nsw i64 %156, 7, !dbg !17
  %158 = sub i64 %46, %157, !dbg !17
  %159 = shl i64 %148, 19, !dbg !17
  %160 = and i64 %159, 16252928, !dbg !17
  %161 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %160, !dbg !17
  %162 = getelementptr [2 x i8], ptr addrspace(1) %161, i64 %86, !dbg !17
  %163 = getelementptr [2 x i8], ptr addrspace(1) %162, i64 %158, !dbg !17
  %164 = getelementptr i8, ptr addrspace(1) %163, i64 64, !dbg !17
  %165 = getelementptr [2 x i8], ptr addrspace(1) %164, i64 %92, !dbg !17
  br label %166, !dbg !17

166:                                              ; preds = %169, %126
  %167 = phi i64 [ %195, %169 ], [ %27, %126 ], !dbg !17
  %168 = icmp samesign ult i64 %167, 128, !dbg !17
  br i1 %168, label %169, label %196, !dbg !17

169:                                              ; preds = %166
  %170 = lshr i64 %167, 2, !dbg !17
  %171 = and i64 %170, 24, !dbg !17
  %172 = shl nuw nsw i64 %167, 3, !dbg !17
  %173 = and i64 %172, 512, !dbg !17
  %174 = shl nuw nsw i64 %167, 7, !dbg !17
  %175 = and i64 %174, 3968, !dbg !17
  %176 = getelementptr [2 x i8], ptr addrspace(1) %165, i64 %171, !dbg !17
  %177 = getelementptr [2 x i8], ptr addrspace(1) %176, i64 %175, !dbg !17
  %178 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 10240), i64 %173, !dbg !17
  %179 = ptrtoint ptr addrspace(3) %178 to i64, !dbg !17
  %180 = trunc i64 %179 to i32, !dbg !17
  %181 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %180), !dbg !17
  %182 = zext i32 %181 to i64, !dbg !17
  %183 = inttoptr i64 %182 to ptr addrspace(3), !dbg !17
  %184 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %185 = ptrtoint ptr addrspace(1) %177 to i64, !dbg !17
  %186 = sub i64 %185, %184, !dbg !17
  %dma.buffer.offset521 = trunc i64 %186 to i32, !dbg !17
  %187 = trunc i64 %184 to i32, !dbg !17
  %188 = insertelement <4 x i32> poison, i32 %187, i64 0, !dbg !17
  %189 = lshr i64 %184, 32, !dbg !17
  %190 = trunc i64 %189 to i32, !dbg !17
  %191 = insertelement <4 x i32> %188, i32 %190, i64 1, !dbg !17
  %192 = insertelement <4 x i32> %191, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc522 = insertelement <4 x i32> %192, i32 131072, i64 3, !dbg !17
  %193 = ptrtoint ptr addrspace(3) %183 to i32, !dbg !17
  %dma.buffer.lds523 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %193), !dbg !17
  %194 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds523, i32 %dma.buffer.offset521, <4 x i32> %dma.buffer.rsrc522) #10, !dbg !17
  %195 = or disjoint i64 %27, 256, !dbg !17
  br label %166, !dbg !17

196:                                              ; preds = %166
  %address.thread58 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !13, !range !40, !frisk.register_lifetimes !1
  %address.local59 = lshr i32 %address.thread58, 2, !dbg !13
  %address.local60 = and i32 %address.local59, 4, !dbg !13
  %address.local61 = shl nuw nsw i32 %address.thread58, 1, !dbg !13
  %address.local62 = and i32 %address.local61, 384, !dbg !13
  %address.local63 = or disjoint i32 %address.local60, %address.local62, !dbg !13
  %address.local64 = shl nuw nsw i32 %address.thread58, 3, !dbg !13
  %address.local65 = and i32 %address.local64, 120, !dbg !13
  %address.local66 = or disjoint i32 %address.local63, %address.local65, !dbg !13
  %address.local67 = shl nuw nsw i32 %address.thread58, 4, !dbg !13
  %address.local68 = and i32 %address.local67, 512, !dbg !13
  %address.local69 = or disjoint i32 %address.local66, %address.local68, !dbg !13
  %address.local70 = zext nneg i32 %address.local69 to i64, !dbg !13
  %address.local71 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local70, !dbg !13
  %197 = load <4 x half>, ptr addrspace(3) %address.local71, align 2, !dbg !13
  %address.local72 = lshr i32 %address.thread58, 2, !dbg !13
  %address.local73 = and i32 %address.local72, 4, !dbg !13
  %address.local74 = shl nuw nsw i32 %address.thread58, 1, !dbg !13
  %address.local75 = and i32 %address.local74, 384, !dbg !13
  %address.local76 = or disjoint i32 %address.local73, %address.local75, !dbg !13
  %address.local77 = shl nuw nsw i32 %address.thread58, 3, !dbg !13
  %address.local78 = and i32 %address.local77, 120, !dbg !13
  %address.local79 = or disjoint i32 %address.local76, %address.local78, !dbg !13
  %address.local80 = shl nuw nsw i32 %address.thread58, 4, !dbg !13
  %address.local81 = and i32 %address.local80, 512, !dbg !13
  %address.local82 = or disjoint i32 %address.local79, %address.local81, !dbg !13
  %address.local83 = or disjoint i32 %address.local82, 1024, !dbg !13
  %address.local84 = zext nneg i32 %address.local83 to i64, !dbg !13
  %address.local85 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local84, !dbg !13
  %198 = load <4 x half>, ptr addrspace(3) %address.local85, align 2, !dbg !13
  %address.local86 = lshr i32 %address.thread58, 2, !dbg !18
  %address.local87 = and i32 %address.local86, 4, !dbg !18
  %address.local88 = shl nuw nsw i32 %address.thread58, 3, !dbg !18
  %address.local89 = and i32 %address.local88, 376, !dbg !18
  %address.local90 = or disjoint i32 %address.local87, %address.local89, !dbg !18
  %address.local91 = zext nneg i32 %address.local90 to i64, !dbg !18
  %address.local92 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local91, !dbg !18
  %199 = load <4 x half>, ptr addrspace(3) %address.local92, align 2, !dbg !18
  %200 = fmul <4 x half> %197, splat (half 0xH3015), !dbg !13
  %address.local93 = lshr i32 %address.thread58, 2, !dbg !18
  %address.local94 = and i32 %address.local93, 4, !dbg !18
  %address.local95 = shl nuw nsw i32 %address.thread58, 3, !dbg !18
  %address.local96 = and i32 %address.local95, 376, !dbg !18
  %address.local97 = or disjoint i32 %address.local94, %address.local96, !dbg !18
  %address.local98 = or disjoint i32 %address.local97, 512, !dbg !18
  %address.local99 = zext nneg i32 %address.local98 to i64, !dbg !18
  %address.local100 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local99, !dbg !18
  %201 = load <4 x half>, ptr addrspace(3) %address.local100, align 2, !dbg !18
  %202 = fmul <4 x half> %198, splat (half 0xH3015), !dbg !13
  %address.local101 = lshr i32 %address.thread58, 2, !dbg !18
  %address.local102 = and i32 %address.local101, 4, !dbg !18
  %address.local103 = shl nuw nsw i32 %address.thread58, 3, !dbg !18
  %address.local104 = and i32 %address.local103, 376, !dbg !18
  %address.local105 = or disjoint i32 %address.local102, %address.local104, !dbg !18
  %address.local106 = or disjoint i32 %address.local105, 128, !dbg !18
  %address.local107 = zext nneg i32 %address.local106 to i64, !dbg !18
  %address.local108 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local107, !dbg !18
  %203 = load <4 x half>, ptr addrspace(3) %address.local108, align 2, !dbg !18
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %200, <4 x half> %199, <4 x float> zeroinitializer) #9, !dbg !18
  %address.local109 = lshr i32 %address.thread58, 2, !dbg !18
  %address.local110 = and i32 %address.local109, 4, !dbg !18
  %address.local111 = shl nuw nsw i32 %address.thread58, 3, !dbg !18
  %address.local112 = and i32 %address.local111, 376, !dbg !18
  %address.local113 = or disjoint i32 %address.local110, %address.local112, !dbg !18
  %address.local114 = or disjoint i32 %address.local113, 640, !dbg !18
  %address.local115 = zext nneg i32 %address.local114 to i64, !dbg !18
  %address.local116 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local115, !dbg !18
  %205 = load <4 x half>, ptr addrspace(3) %address.local116, align 2, !dbg !18
  %206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %202, <4 x half> %201, <4 x float> %204) #9, !dbg !18
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %200, <4 x half> %203, <4 x float> zeroinitializer) #9, !dbg !18
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %202, <4 x half> %205, <4 x float> %207) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %address.local117 = zext nneg i32 %address.thread58 to i64, !dbg !13
  %address.local118 = shl nuw nsw i64 %address.local117, 3, !dbg !12
  %address.local119 = and i64 %address.local118, 1536, !dbg !12
  %address.local120 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local119, !dbg !12
  %address.local121 = ptrtoint ptr addrspace(3) %address.local120 to i64, !dbg !12
  %address.local122 = trunc i64 %address.local121 to i32, !dbg !12
  %209 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local122), !dbg !12
  %210 = zext i32 %209 to i64, !dbg !12
  %211 = inttoptr i64 %210 to ptr addrspace(3), !dbg !12
  %address.local123 = shl nuw nsw i64 %address.local117, 7, !dbg !12
  %address.local124 = and i64 %address.local123, 8064, !dbg !12
  %address.local125 = lshr i64 %address.local117, 3, !dbg !12
  %address.local126 = and i64 %address.local125, 24, !dbg !12
  %address.local127 = or disjoint i64 %address.local124, %address.local126, !dbg !12
  %address.local128 = getelementptr [2 x i8], ptr addrspace(1) %48, i64 %address.local127, !dbg !12
  %212 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %213 = ptrtoint ptr addrspace(1) %address.local128 to i64, !dbg !12
  %214 = sub i64 %213, %212, !dbg !12
  %dma.buffer.offset545 = trunc i64 %214 to i32, !dbg !12
  %215 = trunc i64 %212 to i32, !dbg !12
  %216 = insertelement <4 x i32> poison, i32 %215, i64 0, !dbg !12
  %217 = lshr i64 %212, 32, !dbg !12
  %218 = trunc i64 %217 to i32, !dbg !12
  %219 = insertelement <4 x i32> %216, i32 %218, i64 1, !dbg !12
  %220 = insertelement <4 x i32> %219, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc546 = insertelement <4 x i32> %220, i32 131072, i64 3, !dbg !12
  %221 = ptrtoint ptr addrspace(3) %211 to i32, !dbg !12
  %dma.buffer.lds547 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %221), !dbg !12
  %222 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds547, i32 %dma.buffer.offset545, <4 x i32> %dma.buffer.rsrc546) #10, !dbg !12
  %223 = add i64 %142, 8192, !dbg !17
  %224 = icmp slt i64 %223, 0, !dbg !17
  %225 = sub i64 -8193, %142, !dbg !17
  %226 = select i1 %224, i64 %225, i64 %223, !dbg !17
  %227 = sdiv i64 %226, 16384, !dbg !17
  %.lobit4 = ashr i64 %223, 63, !dbg !17
  %228 = xor i64 %227, %.lobit4, !dbg !17
  %229 = add nsw i64 %149, 64, !dbg !17
  %230 = icmp slt i64 %149, -64, !dbg !17
  %231 = sub nsw i64 -65, %149, !dbg !17
  %232 = select i1 %230, i64 %231, i64 %229, !dbg !17
  %233 = sdiv i64 %232, 128, !dbg !17
  %234 = sext i1 %230 to i64, !dbg !17
  %235 = xor i64 %233, %234, !dbg !17
  %236 = shl nsw i64 %235, 7, !dbg !17
  %237 = sub i64 %46, %236, !dbg !17
  %238 = shl i64 %228, 19, !dbg !17
  %239 = and i64 %238, 16252928, !dbg !17
  %240 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %239, !dbg !17
  %241 = getelementptr [2 x i8], ptr addrspace(1) %240, i64 %86, !dbg !17
  %242 = getelementptr [2 x i8], ptr addrspace(1) %241, i64 %237, !dbg !17
  %243 = getelementptr i8, ptr addrspace(1) %242, i64 128, !dbg !17
  %244 = getelementptr [2 x i8], ptr addrspace(1) %243, i64 %92, !dbg !17
  br label %245, !dbg !17

245:                                              ; preds = %248, %196
  %246 = phi i64 [ %274, %248 ], [ %27, %196 ], !dbg !17
  %247 = icmp samesign ult i64 %246, 128, !dbg !17
  br i1 %247, label %248, label %275, !dbg !17

248:                                              ; preds = %245
  %249 = lshr i64 %246, 2, !dbg !17
  %250 = and i64 %249, 24, !dbg !17
  %251 = shl nuw nsw i64 %246, 3, !dbg !17
  %252 = and i64 %251, 512, !dbg !17
  %253 = shl nuw nsw i64 %246, 7, !dbg !17
  %254 = and i64 %253, 3968, !dbg !17
  %255 = getelementptr [2 x i8], ptr addrspace(1) %244, i64 %250, !dbg !17
  %256 = getelementptr [2 x i8], ptr addrspace(1) %255, i64 %254, !dbg !17
  %257 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %252, !dbg !17
  %258 = ptrtoint ptr addrspace(3) %257 to i64, !dbg !17
  %259 = trunc i64 %258 to i32, !dbg !17
  %260 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %259), !dbg !17
  %261 = zext i32 %260 to i64, !dbg !17
  %262 = inttoptr i64 %261 to ptr addrspace(3), !dbg !17
  %263 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %264 = ptrtoint ptr addrspace(1) %256 to i64, !dbg !17
  %265 = sub i64 %264, %263, !dbg !17
  %dma.buffer.offset524 = trunc i64 %265 to i32, !dbg !17
  %266 = trunc i64 %263 to i32, !dbg !17
  %267 = insertelement <4 x i32> poison, i32 %266, i64 0, !dbg !17
  %268 = lshr i64 %263, 32, !dbg !17
  %269 = trunc i64 %268 to i32, !dbg !17
  %270 = insertelement <4 x i32> %267, i32 %269, i64 1, !dbg !17
  %271 = insertelement <4 x i32> %270, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc525 = insertelement <4 x i32> %271, i32 131072, i64 3, !dbg !17
  %272 = ptrtoint ptr addrspace(3) %262 to i32, !dbg !17
  %dma.buffer.lds526 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %272), !dbg !17
  %273 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds526, i32 %dma.buffer.offset524, <4 x i32> %dma.buffer.rsrc525) #10, !dbg !17
  %274 = or disjoint i64 %27, 256, !dbg !17
  br label %245, !dbg !17

275:                                              ; preds = %245
  %address.thread129 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !13, !range !40, !frisk.register_lifetimes !1
  %address.local130 = lshr i32 %address.thread129, 2, !dbg !13
  %address.local131 = and i32 %address.local130, 4, !dbg !13
  %address.local132 = shl nuw nsw i32 %address.thread129, 1, !dbg !13
  %address.local133 = and i32 %address.local132, 384, !dbg !13
  %address.local134 = or disjoint i32 %address.local131, %address.local133, !dbg !13
  %address.local135 = shl nuw nsw i32 %address.thread129, 3, !dbg !13
  %address.local136 = and i32 %address.local135, 120, !dbg !13
  %address.local137 = or disjoint i32 %address.local134, %address.local136, !dbg !13
  %address.local138 = shl nuw nsw i32 %address.thread129, 4, !dbg !13
  %address.local139 = and i32 %address.local138, 512, !dbg !13
  %address.local140 = or disjoint i32 %address.local137, %address.local139, !dbg !13
  %address.local141 = zext nneg i32 %address.local140 to i64, !dbg !13
  %address.local142 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local141, !dbg !13
  %276 = load <4 x half>, ptr addrspace(3) %address.local142, align 2, !dbg !13
  %address.local143 = lshr i32 %address.thread129, 2, !dbg !13
  %address.local144 = and i32 %address.local143, 4, !dbg !13
  %address.local145 = shl nuw nsw i32 %address.thread129, 1, !dbg !13
  %address.local146 = and i32 %address.local145, 384, !dbg !13
  %address.local147 = or disjoint i32 %address.local144, %address.local146, !dbg !13
  %address.local148 = shl nuw nsw i32 %address.thread129, 3, !dbg !13
  %address.local149 = and i32 %address.local148, 120, !dbg !13
  %address.local150 = or disjoint i32 %address.local147, %address.local149, !dbg !13
  %address.local151 = shl nuw nsw i32 %address.thread129, 4, !dbg !13
  %address.local152 = and i32 %address.local151, 512, !dbg !13
  %address.local153 = or disjoint i32 %address.local150, %address.local152, !dbg !13
  %address.local154 = or disjoint i32 %address.local153, 1024, !dbg !13
  %address.local155 = zext nneg i32 %address.local154 to i64, !dbg !13
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local155, !dbg !13
  %277 = load <4 x half>, ptr addrspace(3) %address.local156, align 2, !dbg !13
  %address.local157 = lshr i32 %address.thread129, 2, !dbg !18
  %address.local158 = and i32 %address.local157, 4, !dbg !18
  %address.local159 = shl nuw nsw i32 %address.thread129, 3, !dbg !18
  %address.local160 = and i32 %address.local159, 376, !dbg !18
  %address.local161 = or disjoint i32 %address.local158, %address.local160, !dbg !18
  %address.local162 = zext nneg i32 %address.local161 to i64, !dbg !18
  %address.local163 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 10240), i64 %address.local162, !dbg !18
  %278 = load <4 x half>, ptr addrspace(3) %address.local163, align 2, !dbg !18
  %279 = fmul <4 x half> %276, splat (half 0xH3015), !dbg !13
  %address.local164 = lshr i32 %address.thread129, 2, !dbg !18
  %address.local165 = and i32 %address.local164, 4, !dbg !18
  %address.local166 = shl nuw nsw i32 %address.thread129, 3, !dbg !18
  %address.local167 = and i32 %address.local166, 376, !dbg !18
  %address.local168 = or disjoint i32 %address.local165, %address.local167, !dbg !18
  %address.local169 = or disjoint i32 %address.local168, 512, !dbg !18
  %address.local170 = zext nneg i32 %address.local169 to i64, !dbg !18
  %address.local171 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 10240), i64 %address.local170, !dbg !18
  %280 = load <4 x half>, ptr addrspace(3) %address.local171, align 2, !dbg !18
  %281 = fmul <4 x half> %277, splat (half 0xH3015), !dbg !13
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %279, <4 x half> %278, <4 x float> %206) #9, !dbg !18
  %address.local172 = lshr i32 %address.thread129, 2, !dbg !18
  %address.local173 = and i32 %address.local172, 4, !dbg !18
  %address.local174 = shl nuw nsw i32 %address.thread129, 3, !dbg !18
  %address.local175 = and i32 %address.local174, 376, !dbg !18
  %address.local176 = or disjoint i32 %address.local173, %address.local175, !dbg !18
  %address.local177 = or disjoint i32 %address.local176, 128, !dbg !18
  %address.local178 = zext nneg i32 %address.local177 to i64, !dbg !18
  %address.local179 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 10240), i64 %address.local178, !dbg !18
  %283 = load <4 x half>, ptr addrspace(3) %address.local179, align 2, !dbg !18
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %281, <4 x half> %280, <4 x float> %282) #9, !dbg !18
  %address.local180 = lshr i32 %address.thread129, 2, !dbg !18
  %address.local181 = and i32 %address.local180, 4, !dbg !18
  %address.local182 = shl nuw nsw i32 %address.thread129, 3, !dbg !18
  %address.local183 = and i32 %address.local182, 376, !dbg !18
  %address.local184 = or disjoint i32 %address.local181, %address.local183, !dbg !18
  %address.local185 = or disjoint i32 %address.local184, 640, !dbg !18
  %address.local186 = zext nneg i32 %address.local185 to i64, !dbg !18
  %address.local187 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 10240), i64 %address.local186, !dbg !18
  %285 = load <4 x half>, ptr addrspace(3) %address.local187, align 2, !dbg !18
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %279, <4 x half> %283, <4 x float> %208) #9, !dbg !18
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %281, <4 x half> %285, <4 x float> %286) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %address.local188 = zext nneg i32 %address.thread129 to i64, !dbg !13
  %address.local189 = shl nuw nsw i64 %address.local188, 3, !dbg !12
  %address.local190 = and i64 %address.local189, 1536, !dbg !12
  %address.local191 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local190, !dbg !12
  %address.local192 = ptrtoint ptr addrspace(3) %address.local191 to i64, !dbg !12
  %address.local193 = trunc i64 %address.local192 to i32, !dbg !12
  %288 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local193), !dbg !12
  %289 = zext i32 %288 to i64, !dbg !12
  %290 = inttoptr i64 %289 to ptr addrspace(3), !dbg !12
  %address.local194 = shl nuw nsw i64 %address.local188, 7, !dbg !12
  %address.local195 = and i64 %address.local194, 8064, !dbg !12
  %address.local196 = lshr i64 %address.local188, 3, !dbg !12
  %address.local197 = and i64 %address.local196, 24, !dbg !12
  %address.local198 = or disjoint i64 %address.local195, %address.local197, !dbg !12
  %address.local199 = getelementptr [2 x i8], ptr addrspace(1) %50, i64 %address.local198, !dbg !12
  %291 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %292 = ptrtoint ptr addrspace(1) %address.local199 to i64, !dbg !12
  %293 = sub i64 %292, %291, !dbg !12
  %dma.buffer.offset542 = trunc i64 %293 to i32, !dbg !12
  %294 = trunc i64 %291 to i32, !dbg !12
  %295 = insertelement <4 x i32> poison, i32 %294, i64 0, !dbg !12
  %296 = lshr i64 %291, 32, !dbg !12
  %297 = trunc i64 %296 to i32, !dbg !12
  %298 = insertelement <4 x i32> %295, i32 %297, i64 1, !dbg !12
  %299 = insertelement <4 x i32> %298, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc543 = insertelement <4 x i32> %299, i32 131072, i64 3, !dbg !12
  %300 = ptrtoint ptr addrspace(3) %290 to i32, !dbg !12
  %dma.buffer.lds544 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %300), !dbg !12
  %301 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds544, i32 %dma.buffer.offset542, <4 x i32> %dma.buffer.rsrc543) #10, !dbg !12
  %302 = add i64 %142, 12288, !dbg !17
  %303 = icmp slt i64 %302, 0, !dbg !17
  %304 = sub i64 -12289, %142, !dbg !17
  %305 = select i1 %303, i64 %304, i64 %302, !dbg !17
  %306 = sdiv i64 %305, 16384, !dbg !17
  %.lobit5 = ashr i64 %302, 63, !dbg !17
  %307 = xor i64 %306, %.lobit5, !dbg !17
  %308 = add nsw i64 %149, 96, !dbg !17
  %309 = icmp slt i64 %149, -96, !dbg !17
  %310 = sub nsw i64 -97, %149, !dbg !17
  %311 = select i1 %309, i64 %310, i64 %308, !dbg !17
  %312 = sdiv i64 %311, 128, !dbg !17
  %313 = sext i1 %309 to i64, !dbg !17
  %314 = xor i64 %312, %313, !dbg !17
  %315 = shl nsw i64 %314, 7, !dbg !17
  %316 = sub i64 %46, %315, !dbg !17
  %317 = shl i64 %307, 19, !dbg !17
  %318 = and i64 %317, 16252928, !dbg !17
  %319 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %318, !dbg !17
  %320 = getelementptr [2 x i8], ptr addrspace(1) %319, i64 %86, !dbg !17
  %321 = getelementptr [2 x i8], ptr addrspace(1) %320, i64 %316, !dbg !17
  %322 = getelementptr i8, ptr addrspace(1) %321, i64 192, !dbg !17
  %323 = getelementptr [2 x i8], ptr addrspace(1) %322, i64 %92, !dbg !17
  br label %324, !dbg !17

324:                                              ; preds = %327, %275
  %325 = phi i64 [ %353, %327 ], [ %27, %275 ], !dbg !17
  %326 = icmp samesign ult i64 %325, 128, !dbg !17
  br i1 %326, label %327, label %354, !dbg !17

327:                                              ; preds = %324
  %328 = lshr i64 %325, 2, !dbg !17
  %329 = and i64 %328, 24, !dbg !17
  %330 = shl nuw nsw i64 %325, 3, !dbg !17
  %331 = and i64 %330, 512, !dbg !17
  %332 = shl nuw nsw i64 %325, 7, !dbg !17
  %333 = and i64 %332, 3968, !dbg !17
  %334 = getelementptr [2 x i8], ptr addrspace(1) %323, i64 %329, !dbg !17
  %335 = getelementptr [2 x i8], ptr addrspace(1) %334, i64 %333, !dbg !17
  %336 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 10240), i64 %331, !dbg !17
  %337 = ptrtoint ptr addrspace(3) %336 to i64, !dbg !17
  %338 = trunc i64 %337 to i32, !dbg !17
  %339 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %338), !dbg !17
  %340 = zext i32 %339 to i64, !dbg !17
  %341 = inttoptr i64 %340 to ptr addrspace(3), !dbg !17
  %342 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %343 = ptrtoint ptr addrspace(1) %335 to i64, !dbg !17
  %344 = sub i64 %343, %342, !dbg !17
  %dma.buffer.offset527 = trunc i64 %344 to i32, !dbg !17
  %345 = trunc i64 %342 to i32, !dbg !17
  %346 = insertelement <4 x i32> poison, i32 %345, i64 0, !dbg !17
  %347 = lshr i64 %342, 32, !dbg !17
  %348 = trunc i64 %347 to i32, !dbg !17
  %349 = insertelement <4 x i32> %346, i32 %348, i64 1, !dbg !17
  %350 = insertelement <4 x i32> %349, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc528 = insertelement <4 x i32> %350, i32 131072, i64 3, !dbg !17
  %351 = ptrtoint ptr addrspace(3) %341 to i32, !dbg !17
  %dma.buffer.lds529 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %351), !dbg !17
  %352 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds529, i32 %dma.buffer.offset527, <4 x i32> %dma.buffer.rsrc528) #10, !dbg !17
  %353 = or disjoint i64 %27, 256, !dbg !17
  br label %324, !dbg !17

354:                                              ; preds = %324
  %address.thread200 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !13, !range !40, !frisk.register_lifetimes !1
  %address.local201 = lshr i32 %address.thread200, 2, !dbg !13
  %address.local202 = and i32 %address.local201, 4, !dbg !13
  %address.local203 = shl nuw nsw i32 %address.thread200, 1, !dbg !13
  %address.local204 = and i32 %address.local203, 384, !dbg !13
  %address.local205 = or disjoint i32 %address.local202, %address.local204, !dbg !13
  %address.local206 = shl nuw nsw i32 %address.thread200, 3, !dbg !13
  %address.local207 = and i32 %address.local206, 120, !dbg !13
  %address.local208 = or disjoint i32 %address.local205, %address.local207, !dbg !13
  %address.local209 = shl nuw nsw i32 %address.thread200, 4, !dbg !13
  %address.local210 = and i32 %address.local209, 512, !dbg !13
  %address.local211 = or disjoint i32 %address.local208, %address.local210, !dbg !13
  %address.local212 = zext nneg i32 %address.local211 to i64, !dbg !13
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local212, !dbg !13
  %355 = load <4 x half>, ptr addrspace(3) %address.local213, align 2, !dbg !13
  %address.local214 = lshr i32 %address.thread200, 2, !dbg !13
  %address.local215 = and i32 %address.local214, 4, !dbg !13
  %address.local216 = shl nuw nsw i32 %address.thread200, 1, !dbg !13
  %address.local217 = and i32 %address.local216, 384, !dbg !13
  %address.local218 = or disjoint i32 %address.local215, %address.local217, !dbg !13
  %address.local219 = shl nuw nsw i32 %address.thread200, 3, !dbg !13
  %address.local220 = and i32 %address.local219, 120, !dbg !13
  %address.local221 = or disjoint i32 %address.local218, %address.local220, !dbg !13
  %address.local222 = shl nuw nsw i32 %address.thread200, 4, !dbg !13
  %address.local223 = and i32 %address.local222, 512, !dbg !13
  %address.local224 = or disjoint i32 %address.local221, %address.local223, !dbg !13
  %address.local225 = or disjoint i32 %address.local224, 1024, !dbg !13
  %address.local226 = zext nneg i32 %address.local225 to i64, !dbg !13
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local226, !dbg !13
  %356 = load <4 x half>, ptr addrspace(3) %address.local227, align 2, !dbg !13
  %address.local228 = lshr i32 %address.thread200, 2, !dbg !18
  %address.local229 = and i32 %address.local228, 4, !dbg !18
  %address.local230 = shl nuw nsw i32 %address.thread200, 3, !dbg !18
  %address.local231 = and i32 %address.local230, 376, !dbg !18
  %address.local232 = or disjoint i32 %address.local229, %address.local231, !dbg !18
  %address.local233 = zext nneg i32 %address.local232 to i64, !dbg !18
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local233, !dbg !18
  %357 = load <4 x half>, ptr addrspace(3) %address.local234, align 2, !dbg !18
  %358 = fmul <4 x half> %355, splat (half 0xH3015), !dbg !13
  %address.local235 = lshr i32 %address.thread200, 2, !dbg !18
  %address.local236 = and i32 %address.local235, 4, !dbg !18
  %address.local237 = shl nuw nsw i32 %address.thread200, 3, !dbg !18
  %address.local238 = and i32 %address.local237, 376, !dbg !18
  %address.local239 = or disjoint i32 %address.local236, %address.local238, !dbg !18
  %address.local240 = or disjoint i32 %address.local239, 512, !dbg !18
  %address.local241 = zext nneg i32 %address.local240 to i64, !dbg !18
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local241, !dbg !18
  %359 = load <4 x half>, ptr addrspace(3) %address.local242, align 2, !dbg !18
  %360 = fmul <4 x half> %356, splat (half 0xH3015), !dbg !13
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %357, <4 x float> %284) #9, !dbg !18
  %address.local243 = lshr i32 %address.thread200, 2, !dbg !18
  %address.local244 = and i32 %address.local243, 4, !dbg !18
  %address.local245 = shl nuw nsw i32 %address.thread200, 3, !dbg !18
  %address.local246 = and i32 %address.local245, 376, !dbg !18
  %address.local247 = or disjoint i32 %address.local244, %address.local246, !dbg !18
  %address.local248 = or disjoint i32 %address.local247, 128, !dbg !18
  %address.local249 = zext nneg i32 %address.local248 to i64, !dbg !18
  %address.local250 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local249, !dbg !18
  %362 = load <4 x half>, ptr addrspace(3) %address.local250, align 2, !dbg !18
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %360, <4 x half> %359, <4 x float> %361) #9, !dbg !18
  %address.local251 = lshr i32 %address.thread200, 2, !dbg !18
  %address.local252 = and i32 %address.local251, 4, !dbg !18
  %address.local253 = shl nuw nsw i32 %address.thread200, 3, !dbg !18
  %address.local254 = and i32 %address.local253, 376, !dbg !18
  %address.local255 = or disjoint i32 %address.local252, %address.local254, !dbg !18
  %address.local256 = or disjoint i32 %address.local255, 640, !dbg !18
  %address.local257 = zext nneg i32 %address.local256 to i64, !dbg !18
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local257, !dbg !18
  %364 = load <4 x half>, ptr addrspace(3) %address.local258, align 2, !dbg !18
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %362, <4 x float> %287) #9, !dbg !18
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %360, <4 x half> %364, <4 x float> %365) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %367 = add nsw i64 %86, %8, !dbg !23
  %368 = shl i64 %367, 19, !dbg !23
  %369 = and i64 %368, 16252928, !dbg !23
  %370 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %369, !dbg !23
  %371 = getelementptr [2 x i8], ptr addrspace(1) %370, i64 %92, !dbg !23
  br label %372, !dbg !23

372:                                              ; preds = %375, %354
  %373 = phi i64 [ %401, %375 ], [ %27, %354 ], !dbg !23
  %374 = icmp samesign ult i64 %373, 128, !dbg !23
  br i1 %374, label %375, label %402, !dbg !23

375:                                              ; preds = %372
  %376 = shl nuw nsw i64 %373, 3, !dbg !23
  %377 = and i64 %376, 24, !dbg !23
  %378 = shl nuw nsw i64 %373, 3, !dbg !23
  %379 = and i64 %378, 512, !dbg !23
  %380 = shl nuw nsw i64 %373, 5, !dbg !23
  %381 = and i64 %380, 3968, !dbg !23
  %382 = getelementptr [2 x i8], ptr addrspace(1) %371, i64 %381, !dbg !23
  %383 = getelementptr [2 x i8], ptr addrspace(1) %382, i64 %377, !dbg !23
  %384 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %379, !dbg !23
  %385 = ptrtoint ptr addrspace(3) %384 to i64, !dbg !23
  %386 = trunc i64 %385 to i32, !dbg !23
  %387 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %386), !dbg !23
  %388 = zext i32 %387 to i64, !dbg !23
  %389 = inttoptr i64 %388 to ptr addrspace(3), !dbg !23
  %390 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !23
  %391 = ptrtoint ptr addrspace(1) %383 to i64, !dbg !23
  %392 = sub i64 %391, %390, !dbg !23
  %dma.buffer.offset530 = trunc i64 %392 to i32, !dbg !23
  %393 = trunc i64 %390 to i32, !dbg !23
  %394 = insertelement <4 x i32> poison, i32 %393, i64 0, !dbg !23
  %395 = lshr i64 %390, 32, !dbg !23
  %396 = trunc i64 %395 to i32, !dbg !23
  %397 = insertelement <4 x i32> %394, i32 %396, i64 1, !dbg !23
  %398 = insertelement <4 x i32> %397, i32 33554432, i64 2, !dbg !23
  %dma.buffer.rsrc531 = insertelement <4 x i32> %398, i32 131072, i64 3, !dbg !23
  %399 = ptrtoint ptr addrspace(3) %389 to i32, !dbg !23
  %dma.buffer.lds532 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %399), !dbg !23
  %400 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds532, i32 %dma.buffer.offset530, <4 x i32> %dma.buffer.rsrc531) #10, !dbg !23
  %401 = or disjoint i64 %27, 256, !dbg !23
  br label %372, !dbg !23

402:                                              ; preds = %372
  %address.thread259 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !13, !range !40, !frisk.register_lifetimes !1
  %address.local260 = lshr i32 %address.thread259, 2, !dbg !13
  %address.local261 = and i32 %address.local260, 4, !dbg !13
  %address.local262 = shl nuw nsw i32 %address.thread259, 1, !dbg !13
  %address.local263 = and i32 %address.local262, 384, !dbg !13
  %address.local264 = or disjoint i32 %address.local261, %address.local263, !dbg !13
  %address.local265 = shl nuw nsw i32 %address.thread259, 3, !dbg !13
  %address.local266 = and i32 %address.local265, 120, !dbg !13
  %address.local267 = or disjoint i32 %address.local264, %address.local266, !dbg !13
  %address.local268 = shl nuw nsw i32 %address.thread259, 4, !dbg !13
  %address.local269 = and i32 %address.local268, 512, !dbg !13
  %address.local270 = or disjoint i32 %address.local267, %address.local269, !dbg !13
  %address.local271 = zext nneg i32 %address.local270 to i64, !dbg !13
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local271, !dbg !13
  %403 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !13
  %address.local273 = lshr i32 %address.thread259, 2, !dbg !13
  %address.local274 = and i32 %address.local273, 4, !dbg !13
  %address.local275 = shl nuw nsw i32 %address.thread259, 1, !dbg !13
  %address.local276 = and i32 %address.local275, 384, !dbg !13
  %address.local277 = or disjoint i32 %address.local274, %address.local276, !dbg !13
  %address.local278 = shl nuw nsw i32 %address.thread259, 3, !dbg !13
  %address.local279 = and i32 %address.local278, 120, !dbg !13
  %address.local280 = or disjoint i32 %address.local277, %address.local279, !dbg !13
  %address.local281 = shl nuw nsw i32 %address.thread259, 4, !dbg !13
  %address.local282 = and i32 %address.local281, 512, !dbg !13
  %address.local283 = or disjoint i32 %address.local280, %address.local282, !dbg !13
  %address.local284 = or disjoint i32 %address.local283, 1024, !dbg !13
  %address.local285 = zext nneg i32 %address.local284 to i64, !dbg !13
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local285, !dbg !13
  %404 = load <4 x half>, ptr addrspace(3) %address.local286, align 2, !dbg !13
  %address.local287 = lshr i32 %address.thread259, 2, !dbg !18
  %address.local288 = and i32 %address.local287, 4, !dbg !18
  %address.local289 = shl nuw nsw i32 %address.thread259, 3, !dbg !18
  %address.local290 = and i32 %address.local289, 376, !dbg !18
  %address.local291 = or disjoint i32 %address.local288, %address.local290, !dbg !18
  %address.local292 = zext nneg i32 %address.local291 to i64, !dbg !18
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 10240), i64 %address.local292, !dbg !18
  %405 = load <4 x half>, ptr addrspace(3) %address.local293, align 2, !dbg !18
  %406 = fmul <4 x half> %403, splat (half 0xH3015), !dbg !13
  %address.local294 = lshr i32 %address.thread259, 2, !dbg !18
  %address.local295 = and i32 %address.local294, 4, !dbg !18
  %address.local296 = shl nuw nsw i32 %address.thread259, 3, !dbg !18
  %address.local297 = and i32 %address.local296, 376, !dbg !18
  %address.local298 = or disjoint i32 %address.local295, %address.local297, !dbg !18
  %address.local299 = or disjoint i32 %address.local298, 512, !dbg !18
  %address.local300 = zext nneg i32 %address.local299 to i64, !dbg !18
  %address.local301 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 10240), i64 %address.local300, !dbg !18
  %407 = load <4 x half>, ptr addrspace(3) %address.local301, align 2, !dbg !18
  %408 = fmul <4 x half> %404, splat (half 0xH3015), !dbg !13
  %409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %406, <4 x half> %405, <4 x float> %363) #9, !dbg !18
  %address.local302 = lshr i32 %address.thread259, 2, !dbg !18
  %address.local303 = and i32 %address.local302, 4, !dbg !18
  %address.local304 = shl nuw nsw i32 %address.thread259, 3, !dbg !18
  %address.local305 = and i32 %address.local304, 376, !dbg !18
  %address.local306 = or disjoint i32 %address.local303, %address.local305, !dbg !18
  %address.local307 = or disjoint i32 %address.local306, 128, !dbg !18
  %address.local308 = zext nneg i32 %address.local307 to i64, !dbg !18
  %address.local309 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 10240), i64 %address.local308, !dbg !18
  %410 = load <4 x half>, ptr addrspace(3) %address.local309, align 2, !dbg !18
  %411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %408, <4 x half> %407, <4 x float> %409) #9, !dbg !18
  %address.local310 = lshr i32 %address.thread259, 2, !dbg !18
  %address.local311 = and i32 %address.local310, 4, !dbg !18
  %address.local312 = shl nuw nsw i32 %address.thread259, 3, !dbg !18
  %address.local313 = and i32 %address.local312, 376, !dbg !18
  %address.local314 = or disjoint i32 %address.local311, %address.local313, !dbg !18
  %address.local315 = or disjoint i32 %address.local314, 640, !dbg !18
  %address.local316 = zext nneg i32 %address.local315 to i64, !dbg !18
  %address.local317 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 10240), i64 %address.local316, !dbg !18
  %412 = load <4 x half>, ptr addrspace(3) %address.local317, align 2, !dbg !18
  %413 = extractelement <4 x float> %411, i64 0, !dbg !22
  %414 = extractelement <4 x float> %411, i64 1, !dbg !22
  %415 = select i1 %.not, float %414, float %413, !dbg !14
  %416 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %406, <4 x half> %410, <4 x float> %366) #9, !dbg !18
  %417 = extractelement <4 x float> %411, i64 2, !dbg !22
  %418 = extractelement <4 x float> %411, i64 3, !dbg !22
  %419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %408, <4 x half> %412, <4 x float> %416) #9, !dbg !18
  %420 = extractelement <4 x float> %419, i64 0, !dbg !22
  %421 = extractelement <4 x float> %419, i64 1, !dbg !22
  %422 = extractelement <4 x float> %419, i64 2, !dbg !22
  %423 = extractelement <4 x float> %419, i64 3, !dbg !22
  %424 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !14
  %425 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %424), !dbg !14
  %426 = add nuw nsw i32 %425, 64, !dbg !14
  %427 = xor i32 %425, 16, !dbg !14
  %428 = and i32 %426, 192, !dbg !14
  %429 = icmp samesign ult i32 %427, %428, !dbg !14
  %430 = select i1 %429, i32 %427, i32 %425, !dbg !14
  %431 = shl nuw nsw i32 %430, 2, !dbg !14
  %432 = bitcast float %415 to i32, !dbg !14
  %433 = call i32 @llvm.amdgcn.ds.bpermute(i32 %431, i32 %432), !dbg !14
  %434 = bitcast i32 %433 to float, !dbg !14
  %435 = select i1 %.not, float %413, float %434, !dbg !14
  %436 = select i1 %.not, float %434, float %414, !dbg !14
  %437 = select i1 %.not, float %418, float %417, !dbg !14
  %438 = bitcast float %437 to i32, !dbg !14
  %439 = call i32 @llvm.amdgcn.ds.bpermute(i32 %431, i32 %438), !dbg !14
  %440 = bitcast i32 %439 to float, !dbg !14
  %441 = select i1 %.not, float %417, float %440, !dbg !14
  %442 = select i1 %.not, float %440, float %418, !dbg !14
  %443 = select i1 %.not2, float %441, float %435, !dbg !14
  %444 = xor i32 %425, 32, !dbg !14
  %445 = icmp samesign ult i32 %444, %428, !dbg !14
  %446 = select i1 %445, i32 %444, i32 %425, !dbg !14
  %447 = shl nuw nsw i32 %446, 2, !dbg !14
  %448 = bitcast float %443 to i32, !dbg !14
  %449 = call i32 @llvm.amdgcn.ds.bpermute(i32 %447, i32 %448), !dbg !14
  %450 = bitcast i32 %449 to float, !dbg !14
  %451 = select i1 %.not2, float %435, float %450, !dbg !14
  %452 = select i1 %.not2, float %450, float %441, !dbg !14
  %453 = select i1 %.not2, float %442, float %436, !dbg !14
  %454 = bitcast float %453 to i32, !dbg !14
  %455 = call i32 @llvm.amdgcn.ds.bpermute(i32 %447, i32 %454), !dbg !14
  %456 = bitcast i32 %455 to float, !dbg !14
  %457 = select i1 %.not2, float %436, float %456, !dbg !14
  %458 = select i1 %.not2, float %456, float %442, !dbg !14
  %459 = select i1 %.not, float %421, float %420, !dbg !14
  %460 = bitcast float %459 to i32, !dbg !14
  %461 = call i32 @llvm.amdgcn.ds.bpermute(i32 %431, i32 %460), !dbg !14
  %462 = bitcast i32 %461 to float, !dbg !14
  %463 = select i1 %.not, float %420, float %462, !dbg !14
  %464 = select i1 %.not, float %462, float %421, !dbg !14
  %465 = select i1 %.not, float %423, float %422, !dbg !14
  %466 = bitcast float %465 to i32, !dbg !14
  %467 = call i32 @llvm.amdgcn.ds.bpermute(i32 %431, i32 %466), !dbg !14
  %468 = bitcast i32 %467 to float, !dbg !14
  %469 = select i1 %.not, float %422, float %468, !dbg !14
  %470 = select i1 %.not, float %468, float %423, !dbg !14
  %471 = select i1 %.not2, float %469, float %463, !dbg !14
  %472 = bitcast float %471 to i32, !dbg !14
  %473 = call i32 @llvm.amdgcn.ds.bpermute(i32 %447, i32 %472), !dbg !14
  %474 = bitcast i32 %473 to float, !dbg !14
  %475 = select i1 %.not2, float %463, float %474, !dbg !14
  %476 = select i1 %.not2, float %474, float %469, !dbg !14
  %477 = select i1 %.not2, float %470, float %464, !dbg !14
  %478 = bitcast float %477 to i32, !dbg !14
  %479 = call i32 @llvm.amdgcn.ds.bpermute(i32 %447, i32 %478), !dbg !14
  %480 = bitcast i32 %479 to float, !dbg !14
  %481 = select i1 %.not2, float %464, float %480, !dbg !14
  %482 = select i1 %.not2, float %480, float %470, !dbg !14
  %483 = or disjoint i64 %69, 31, !dbg !15
  %484 = icmp uge i64 %11, %483, !dbg !15
  %485 = and i1 %35, %484, !dbg !15
  br i1 %485, label %486, label %487, !dbg !15

486:                                              ; preds = %402
  br label %513, !dbg !15

487:                                              ; preds = %402
  %488 = add nsw i64 %11, %27, !dbg !15
  %489 = shl nuw nsw i64 %28, 4, !dbg !15
  %490 = add nsw i64 %488, %489, !dbg !15
  %491 = and i32 %26, 240, !dbg !15
  %index.compact = zext nneg i32 %491 to i64, !dbg !15
  %492 = sub nsw i64 %490, %index.compact, !dbg !15
  %493 = add nsw i64 %492, 1, !dbg !24
  %494 = or disjoint i64 %69, %30, !dbg !15
  %495 = lshr i64 %493, 32, !dbg !25
  %compare.high = trunc i64 %495 to i32, !dbg !25
  %496 = lshr i64 %494, 32, !dbg !25
  %compare.high493 = trunc i64 %496 to i32, !dbg !25
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high493, !dbg !25
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high493, !dbg !25
  %compare.low.base = trunc i64 %493 to i32, !dbg !25
  %compare.low.base494 = trunc i64 %494 to i32, !dbg !25
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base494, !dbg !25
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !25
  %497 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base494, i32 1) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !25
  %compare.low495 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !25
  %compare.result496 = select i1 %compare.high.equal, i1 %compare.low495, i1 %compare.high.order, !dbg !25
  %498 = select i1 %compare.result496, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base497 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base494, i32 2) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local498 = or i32 %compare.local.base497, 2, !dbg !25
  %compare.low499 = icmp ugt i32 %compare.low.base, %compare.local498, !dbg !25
  %compare.result500 = select i1 %compare.high.equal, i1 %compare.low499, i1 %compare.high.order, !dbg !25
  %499 = select i1 %compare.result500, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base501 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base494, i32 3) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local502 = or i32 %compare.local.base501, 3, !dbg !25
  %compare.low503 = icmp ugt i32 %compare.low.base, %compare.local502, !dbg !25
  %compare.result504 = select i1 %compare.high.equal, i1 %compare.low503, i1 %compare.high.order, !dbg !25
  %500 = select i1 %compare.result504, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base505 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base494, i32 4) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local506 = or i32 %compare.local.base505, 16, !dbg !25
  %compare.low507 = icmp ugt i32 %compare.low.base, %compare.local506, !dbg !25
  %compare.result508 = select i1 %compare.high.equal, i1 %compare.low507, i1 %compare.high.order, !dbg !25
  %501 = select i1 %compare.result508, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base509 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base494, i32 5) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local510 = or i32 %compare.local.base509, 17, !dbg !25
  %compare.low511 = icmp ugt i32 %compare.low.base, %compare.local510, !dbg !25
  %compare.result512 = select i1 %compare.high.equal, i1 %compare.low511, i1 %compare.high.order, !dbg !25
  %502 = select i1 %compare.result512, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base513 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base494, i32 6) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local514 = or i32 %compare.local.base513, 18, !dbg !25
  %compare.low515 = icmp ugt i32 %compare.low.base, %compare.local514, !dbg !25
  %compare.result516 = select i1 %compare.high.equal, i1 %compare.low515, i1 %compare.high.order, !dbg !25
  %503 = select i1 %compare.result516, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base517 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base494, i32 7) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local518 = or i32 %compare.local.base517, 19, !dbg !25
  %compare.low519 = icmp ugt i32 %compare.low.base, %compare.local518, !dbg !25
  %compare.result520 = select i1 %compare.high.equal, i1 %compare.low519, i1 %compare.high.order, !dbg !25
  %504 = select i1 %compare.result520, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %505 = insertelement <8 x float> poison, float %497, i64 0, !dbg !15
  %506 = insertelement <8 x float> %505, float %498, i64 1, !dbg !15
  %507 = insertelement <8 x float> %506, float %499, i64 2, !dbg !15
  %508 = insertelement <8 x float> %507, float %500, i64 3, !dbg !15
  %509 = insertelement <8 x float> %508, float %501, i64 4, !dbg !15
  %510 = insertelement <8 x float> %509, float %502, i64 5, !dbg !15
  %511 = insertelement <8 x float> %510, float %503, i64 6, !dbg !15
  %512 = insertelement <8 x float> %511, float %504, i64 7, !dbg !15
  br label %513, !dbg !15

513:                                              ; preds = %486, %487
  %514 = phi <8 x float> [ %512, %487 ], [ zeroinitializer, %486 ], !dbg !15
  br label %515, !dbg !15

515:                                              ; preds = %513
  %516 = insertelement <4 x float> poison, float %451, i64 0, !dbg !27
  %517 = insertelement <4 x float> %516, float %457, i64 1, !dbg !27
  %518 = insertelement <4 x float> %517, float %452, i64 2, !dbg !27
  %519 = insertelement <4 x float> %518, float %458, i64 3, !dbg !27
  %520 = shufflevector <8 x float> %514, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %521 = fadd <4 x float> %519, %520, !dbg !27
  %522 = extractelement <4 x float> %521, i64 0, !dbg !27
  %523 = extractelement <4 x float> %521, i64 1, !dbg !27
  %524 = extractelement <4 x float> %521, i64 2, !dbg !27
  %525 = extractelement <4 x float> %521, i64 3, !dbg !27
  %526 = insertelement <4 x float> poison, float %475, i64 0, !dbg !27
  %527 = insertelement <4 x float> %526, float %481, i64 1, !dbg !27
  %528 = insertelement <4 x float> %527, float %476, i64 2, !dbg !27
  %529 = insertelement <4 x float> %528, float %482, i64 3, !dbg !27
  %530 = shufflevector <8 x float> %514, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %531 = fadd <4 x float> %529, %530, !dbg !27
  %532 = extractelement <4 x float> %531, i64 0, !dbg !27
  %533 = extractelement <4 x float> %531, i64 1, !dbg !27
  %534 = extractelement <4 x float> %531, i64 2, !dbg !27
  %535 = extractelement <4 x float> %531, i64 3, !dbg !27
  %536 = call float @llvm.maxnum.f32(float %522, float 0xFFF0000000000000), !dbg !28
  %537 = call float @llvm.maxnum.f32(float %536, float %523), !dbg !28
  %538 = call float @llvm.maxnum.f32(float %537, float %524), !dbg !28
  %539 = call float @llvm.maxnum.f32(float %538, float %525), !dbg !28
  %540 = call float @llvm.maxnum.f32(float %539, float %532), !dbg !28
  %541 = call float @llvm.maxnum.f32(float %540, float %533), !dbg !28
  %542 = call float @llvm.maxnum.f32(float %541, float %534), !dbg !28
  %543 = call float @llvm.maxnum.f32(float %542, float %535), !dbg !28
  %544 = bitcast float %543 to i32, !dbg !28
  %545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %431, i32 %544), !dbg !28
  %546 = bitcast i32 %545 to float, !dbg !28
  %547 = call float @llvm.maxnum.f32(float %543, float %546), !dbg !28
  %548 = bitcast float %547 to i32, !dbg !28
  %549 = call i32 @llvm.amdgcn.ds.bpermute(i32 %447, i32 %548), !dbg !28
  %550 = bitcast i32 %549 to float, !dbg !28
  %551 = call float @llvm.maxnum.f32(float %547, float %550), !dbg !28
  %552 = insertelement <1 x float> poison, float %551, i64 0, !dbg !28
  %553 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %552, <1 x float> %57), !dbg !28
  %554 = shufflevector <1 x float> %553, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !29
  %555 = fsub <4 x float> %521, %554, !dbg !29
  %556 = fsub <4 x float> %531, %554, !dbg !29
  %557 = fsub <1 x float> %57, %553, !dbg !30
  %558 = extractelement <1 x float> %557, i64 0, !dbg !31
  %559 = call float @llvm.amdgcn.exp2.f32(float %558), !dbg !31
  %560 = extractelement <1 x float> %56, i64 0, !dbg !32
  %561 = fmul float %560, %559, !dbg !32
  %562 = extractelement <4 x float> %555, i64 0, !dbg !33
  %563 = call float @llvm.amdgcn.exp2.f32(float %562), !dbg !33
  %564 = insertelement <4 x float> poison, float %563, i64 0, !dbg !35
  %565 = extractelement <4 x float> %555, i64 1, !dbg !33
  %566 = call float @llvm.amdgcn.exp2.f32(float %565), !dbg !33
  %567 = insertelement <4 x float> %564, float %566, i64 1, !dbg !35
  %568 = fadd float %563, %566, !dbg !34
  %569 = extractelement <4 x float> %555, i64 2, !dbg !33
  %570 = call float @llvm.amdgcn.exp2.f32(float %569), !dbg !33
  %571 = insertelement <4 x float> %567, float %570, i64 2, !dbg !35
  %572 = fadd float %568, %570, !dbg !34
  %573 = extractelement <4 x float> %555, i64 3, !dbg !33
  %574 = call float @llvm.amdgcn.exp2.f32(float %573), !dbg !33
  %575 = insertelement <4 x float> %571, float %574, i64 3, !dbg !35
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %575) #9, !dbg !35, !frisk.register_lifetimes !1
  %576 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !35
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %576) #9, !dbg !34, !frisk.register_lifetimes !1
  %577 = fadd float %572, %574, !dbg !34
  %578 = extractelement <4 x float> %556, i64 0, !dbg !33
  %579 = call float @llvm.amdgcn.exp2.f32(float %578), !dbg !33
  %580 = insertelement <4 x float> poison, float %579, i64 0, !dbg !35
  %581 = fadd float %577, %579, !dbg !34
  %582 = extractelement <4 x float> %556, i64 1, !dbg !33
  %583 = call float @llvm.amdgcn.exp2.f32(float %582), !dbg !33
  %584 = insertelement <4 x float> %580, float %583, i64 1, !dbg !35
  %585 = fadd float %581, %583, !dbg !34
  %586 = extractelement <4 x float> %556, i64 2, !dbg !33
  %587 = call float @llvm.amdgcn.exp2.f32(float %586), !dbg !33
  %588 = insertelement <4 x float> %584, float %587, i64 2, !dbg !35
  %589 = fadd float %585, %587, !dbg !34
  %590 = extractelement <4 x float> %556, i64 3, !dbg !33
  %591 = call float @llvm.amdgcn.exp2.f32(float %590), !dbg !33
  %592 = insertelement <4 x float> %588, float %591, i64 3, !dbg !35
  %math.result32 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %592) #9, !dbg !35, !frisk.register_lifetimes !1
  %593 = fptrunc <4 x float> %math.result32 to <4 x half>, !dbg !35
  %math.packed33 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %593) #9, !dbg !34, !frisk.register_lifetimes !1
  %594 = fadd float %589, %591, !dbg !34
  %595 = bitcast float %594 to i32, !dbg !34
  %596 = call i32 @llvm.amdgcn.ds.bpermute(i32 %431, i32 %595), !dbg !34
  %597 = bitcast i32 %596 to float, !dbg !34
  %598 = fadd float %594, %597, !dbg !34
  %599 = bitcast float %598 to i32, !dbg !34
  %600 = call i32 @llvm.amdgcn.ds.bpermute(i32 %447, i32 %599), !dbg !34
  %601 = bitcast i32 %600 to float, !dbg !34
  %602 = fadd float %598, %601, !dbg !34
  %603 = fadd float %602, %561, !dbg !34
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %address.thread318 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local319 = shl nuw nsw i32 %address.thread318, 3, !dbg !16
  %address.local320 = and i32 %address.local319, 504, !dbg !16
  %address.local321 = zext nneg i32 %address.local320 to i64, !dbg !16
  %address.local322 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local321, !dbg !16
  %604 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local322, i32 0) #0, !dbg !16
  %605 = bitcast <4 x i32> %604 to <8 x half>, !dbg !16
  %address.local323 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local321, !dbg !16
  %address.local324 = getelementptr i8, ptr addrspace(3) %address.local323, i64 1024, !dbg !16
  %606 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local324, i32 0) #0, !dbg !16
  %607 = bitcast <4 x i32> %606 to <8 x half>, !dbg !16
  %608 = add nsw i64 %86, %8, !dbg !23
  %609 = shl i64 %608, 19, !dbg !23
  %610 = and i64 %609, 16252928, !dbg !23
  %611 = add i64 %610, %92, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %612 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %611, !dbg !23
  %613 = getelementptr i8, ptr addrspace(1) %612, i64 64, !dbg !23
  br label %614, !dbg !23

614:                                              ; preds = %617, %515
  %615 = phi i64 [ %643, %617 ], [ %27, %515 ], !dbg !23
  %616 = icmp samesign ult i64 %615, 128, !dbg !23
  br i1 %616, label %617, label %644, !dbg !23

617:                                              ; preds = %614
  %618 = shl nuw nsw i64 %615, 3, !dbg !23
  %619 = and i64 %618, 24, !dbg !23
  %620 = shl nuw nsw i64 %615, 3, !dbg !23
  %621 = and i64 %620, 512, !dbg !23
  %622 = shl nuw nsw i64 %615, 5, !dbg !23
  %623 = and i64 %622, 3968, !dbg !23
  %624 = getelementptr [2 x i8], ptr addrspace(1) %613, i64 %623, !dbg !23
  %625 = getelementptr [2 x i8], ptr addrspace(1) %624, i64 %619, !dbg !23
  %626 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %621, !dbg !23
  %627 = ptrtoint ptr addrspace(3) %626 to i64, !dbg !23
  %628 = trunc i64 %627 to i32, !dbg !23
  %629 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %628), !dbg !23
  %630 = zext i32 %629 to i64, !dbg !23
  %631 = inttoptr i64 %630 to ptr addrspace(3), !dbg !23
  %632 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !23
  %633 = ptrtoint ptr addrspace(1) %625 to i64, !dbg !23
  %634 = sub i64 %633, %632, !dbg !23
  %dma.buffer.offset533 = trunc i64 %634 to i32, !dbg !23
  %635 = trunc i64 %632 to i32, !dbg !23
  %636 = insertelement <4 x i32> poison, i32 %635, i64 0, !dbg !23
  %637 = lshr i64 %632, 32, !dbg !23
  %638 = trunc i64 %637 to i32, !dbg !23
  %639 = insertelement <4 x i32> %636, i32 %638, i64 1, !dbg !23
  %640 = insertelement <4 x i32> %639, i32 33554432, i64 2, !dbg !23
  %dma.buffer.rsrc534 = insertelement <4 x i32> %640, i32 131072, i64 3, !dbg !23
  %641 = ptrtoint ptr addrspace(3) %631 to i32, !dbg !23
  %dma.buffer.lds535 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %641), !dbg !23
  %642 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds535, i32 %dma.buffer.offset533, <4 x i32> %dma.buffer.rsrc534) #10, !dbg !23
  %643 = or disjoint i64 %27, 256, !dbg !23
  br label %614, !dbg !23

644:                                              ; preds = %614
  %645 = shufflevector <8 x half> %605, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %646 = insertelement <4 x float> poison, float %559, i64 0, !dbg !36
  %647 = shufflevector <4 x float> %646, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %648 = fmul <4 x float> %53, %647, !dbg !36
  %649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %645, <4 x float> %648) #9, !dbg !16
  %650 = shufflevector <8 x half> %607, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %650, <4 x float> %649) #9, !dbg !16
  %652 = shufflevector <8 x half> %605, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %653 = fmul <4 x float> %54, %647, !dbg !36
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %652, <4 x float> %653) #9, !dbg !16
  %655 = shufflevector <8 x half> %607, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %655, <4 x float> %654) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %address.thread325 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local326 = shl nuw nsw i32 %address.thread325, 3, !dbg !16
  %address.local327 = and i32 %address.local326, 504, !dbg !16
  %address.local328 = zext nneg i32 %address.local327 to i64, !dbg !16
  %address.local329 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local328, !dbg !16
  %657 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local329, i32 0) #0, !dbg !16
  %658 = bitcast <4 x i32> %657 to <8 x half>, !dbg !16
  %address.local330 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local328, !dbg !16
  %address.local331 = getelementptr i8, ptr addrspace(3) %address.local330, i64 1024, !dbg !16
  %659 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !16
  %660 = bitcast <4 x i32> %659 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %661 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %611, !dbg !23
  %662 = getelementptr i8, ptr addrspace(1) %661, i64 128, !dbg !23
  br label %663, !dbg !23

663:                                              ; preds = %666, %644
  %664 = phi i64 [ %692, %666 ], [ %27, %644 ], !dbg !23
  %665 = icmp samesign ult i64 %664, 128, !dbg !23
  br i1 %665, label %666, label %693, !dbg !23

666:                                              ; preds = %663
  %667 = shl nuw nsw i64 %664, 3, !dbg !23
  %668 = and i64 %667, 24, !dbg !23
  %669 = shl nuw nsw i64 %664, 3, !dbg !23
  %670 = and i64 %669, 512, !dbg !23
  %671 = shl nuw nsw i64 %664, 5, !dbg !23
  %672 = and i64 %671, 3968, !dbg !23
  %673 = getelementptr [2 x i8], ptr addrspace(1) %662, i64 %672, !dbg !23
  %674 = getelementptr [2 x i8], ptr addrspace(1) %673, i64 %668, !dbg !23
  %675 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %670, !dbg !23
  %676 = ptrtoint ptr addrspace(3) %675 to i64, !dbg !23
  %677 = trunc i64 %676 to i32, !dbg !23
  %678 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %677), !dbg !23
  %679 = zext i32 %678 to i64, !dbg !23
  %680 = inttoptr i64 %679 to ptr addrspace(3), !dbg !23
  %681 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !23
  %682 = ptrtoint ptr addrspace(1) %674 to i64, !dbg !23
  %683 = sub i64 %682, %681, !dbg !23
  %dma.buffer.offset536 = trunc i64 %683 to i32, !dbg !23
  %684 = trunc i64 %681 to i32, !dbg !23
  %685 = insertelement <4 x i32> poison, i32 %684, i64 0, !dbg !23
  %686 = lshr i64 %681, 32, !dbg !23
  %687 = trunc i64 %686 to i32, !dbg !23
  %688 = insertelement <4 x i32> %685, i32 %687, i64 1, !dbg !23
  %689 = insertelement <4 x i32> %688, i32 33554432, i64 2, !dbg !23
  %dma.buffer.rsrc537 = insertelement <4 x i32> %689, i32 131072, i64 3, !dbg !23
  %690 = ptrtoint ptr addrspace(3) %680 to i32, !dbg !23
  %dma.buffer.lds538 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %690), !dbg !23
  %691 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds538, i32 %dma.buffer.offset536, <4 x i32> %dma.buffer.rsrc537) #10, !dbg !23
  %692 = or disjoint i64 %27, 256, !dbg !23
  br label %663, !dbg !23

693:                                              ; preds = %663
  %694 = shufflevector <8 x half> %658, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %695 = fmul <4 x float> %58, %647, !dbg !36
  %696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %694, <4 x float> %695) #9, !dbg !16
  %697 = shufflevector <8 x half> %660, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %697, <4 x float> %696) #9, !dbg !16
  %699 = shufflevector <8 x half> %658, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %700 = fmul <4 x float> %59, %647, !dbg !36
  %701 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %699, <4 x float> %700) #9, !dbg !16
  %702 = shufflevector <8 x half> %660, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %703 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %702, <4 x float> %701) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %address.thread332 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local333 = shl nuw nsw i32 %address.thread332, 3, !dbg !16
  %address.local334 = and i32 %address.local333, 504, !dbg !16
  %address.local335 = zext nneg i32 %address.local334 to i64, !dbg !16
  %address.local336 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !16
  %704 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local336, i32 0) #0, !dbg !16
  %705 = bitcast <4 x i32> %704 to <8 x half>, !dbg !16
  %address.local337 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !16
  %address.local338 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !16
  %706 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local338, i32 0) #0, !dbg !16
  %707 = bitcast <4 x i32> %706 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %708 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %611, !dbg !23
  %709 = getelementptr i8, ptr addrspace(1) %708, i64 192, !dbg !23
  br label %710, !dbg !23

710:                                              ; preds = %713, %693
  %711 = phi i64 [ %739, %713 ], [ %27, %693 ], !dbg !23
  %712 = icmp samesign ult i64 %711, 128, !dbg !23
  br i1 %712, label %713, label %740, !dbg !23

713:                                              ; preds = %710
  %714 = shl nuw nsw i64 %711, 3, !dbg !23
  %715 = and i64 %714, 24, !dbg !23
  %716 = shl nuw nsw i64 %711, 3, !dbg !23
  %717 = and i64 %716, 512, !dbg !23
  %718 = shl nuw nsw i64 %711, 5, !dbg !23
  %719 = and i64 %718, 3968, !dbg !23
  %720 = getelementptr [2 x i8], ptr addrspace(1) %709, i64 %719, !dbg !23
  %721 = getelementptr [2 x i8], ptr addrspace(1) %720, i64 %715, !dbg !23
  %722 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %717, !dbg !23
  %723 = ptrtoint ptr addrspace(3) %722 to i64, !dbg !23
  %724 = trunc i64 %723 to i32, !dbg !23
  %725 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %724), !dbg !23
  %726 = zext i32 %725 to i64, !dbg !23
  %727 = inttoptr i64 %726 to ptr addrspace(3), !dbg !23
  %728 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !23
  %729 = ptrtoint ptr addrspace(1) %721 to i64, !dbg !23
  %730 = sub i64 %729, %728, !dbg !23
  %dma.buffer.offset539 = trunc i64 %730 to i32, !dbg !23
  %731 = trunc i64 %728 to i32, !dbg !23
  %732 = insertelement <4 x i32> poison, i32 %731, i64 0, !dbg !23
  %733 = lshr i64 %728, 32, !dbg !23
  %734 = trunc i64 %733 to i32, !dbg !23
  %735 = insertelement <4 x i32> %732, i32 %734, i64 1, !dbg !23
  %736 = insertelement <4 x i32> %735, i32 33554432, i64 2, !dbg !23
  %dma.buffer.rsrc540 = insertelement <4 x i32> %736, i32 131072, i64 3, !dbg !23
  %737 = ptrtoint ptr addrspace(3) %727 to i32, !dbg !23
  %dma.buffer.lds541 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %737), !dbg !23
  %738 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds541, i32 %dma.buffer.offset539, <4 x i32> %dma.buffer.rsrc540) #10, !dbg !23
  %739 = or disjoint i64 %27, 256, !dbg !23
  br label %710, !dbg !23

740:                                              ; preds = %710
  %741 = shufflevector <8 x half> %705, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %742 = fmul <4 x float> %61, %647, !dbg !36
  %743 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %741, <4 x float> %742) #9, !dbg !16
  %744 = shufflevector <8 x half> %707, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %745 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %744, <4 x float> %743) #9, !dbg !16
  %746 = shufflevector <8 x half> %705, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %747 = fmul <4 x float> %62, %647, !dbg !36
  %748 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %746, <4 x float> %747) #9, !dbg !16
  %749 = shufflevector <8 x half> %707, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %750 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %749, <4 x float> %748) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !23
  fence syncscope("workgroup") release, !dbg !23
  call void @llvm.amdgcn.s.barrier(), !dbg !23
  fence syncscope("workgroup") acquire, !dbg !23
  %address.thread339 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local340 = shl nuw nsw i32 %address.thread339, 3, !dbg !16
  %address.local341 = and i32 %address.local340, 504, !dbg !16
  %address.local342 = zext nneg i32 %address.local341 to i64, !dbg !16
  %address.local343 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local342, !dbg !16
  %751 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #0, !dbg !16
  %752 = bitcast <4 x i32> %751 to <8 x half>, !dbg !16
  %address.local344 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local342, !dbg !16
  %address.local345 = getelementptr i8, ptr addrspace(3) %address.local344, i64 1024, !dbg !16
  %753 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local345, i32 0) #0, !dbg !16
  %754 = bitcast <4 x i32> %753 to <8 x half>, !dbg !16
  %755 = shufflevector <8 x half> %752, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %756 = fmul <4 x float> %64, %647, !dbg !36
  %757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %755, <4 x float> %756) #9, !dbg !16
  %758 = shufflevector <8 x half> %754, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %759 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %758, <4 x float> %757) #9, !dbg !16
  %760 = shufflevector <8 x half> %752, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %761 = fmul <4 x float> %65, %647, !dbg !36
  %762 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %760, <4 x float> %761) #9, !dbg !16
  %763 = shufflevector <8 x half> %754, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %764 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed33, <4 x half> %763, <4 x float> %762) #9, !dbg !16
  %765 = shufflevector <4 x float> %651, <4 x float> %656, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %766 = insertelement <1 x float> poison, float %603, i64 0, !dbg !7
  %767 = shufflevector <4 x float> %698, <4 x float> %703, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %768 = shufflevector <4 x float> %745, <4 x float> %750, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %769 = shufflevector <4 x float> %759, <4 x float> %764, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %770 = add nuw nsw i64 %52, 1, !dbg !7
  br label %51, !dbg !7

771:                                              ; preds = %51
  %772 = extractelement <8 x float> %66, i64 0, !dbg !7
  %773 = extractelement <8 x float> %66, i64 1, !dbg !7
  %774 = extractelement <8 x float> %66, i64 2, !dbg !7
  %775 = extractelement <8 x float> %66, i64 3, !dbg !7
  %776 = select i1 %.not, float %773, float %772, !dbg !7
  %777 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %778 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %777), !dbg !7
  %779 = add nuw nsw i32 %778, 64, !dbg !7
  %780 = xor i32 %778, 16, !dbg !7
  %781 = and i32 %779, 192, !dbg !7
  %782 = icmp samesign ult i32 %780, %781, !dbg !7
  %783 = select i1 %782, i32 %780, i32 %778, !dbg !7
  %784 = shl nuw nsw i32 %783, 2, !dbg !7
  %785 = bitcast float %776 to i32, !dbg !7
  %786 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %785), !dbg !7
  %787 = bitcast i32 %786 to float, !dbg !7
  %788 = select i1 %.not, float %772, float %787, !dbg !7
  %789 = select i1 %.not, float %787, float %773, !dbg !7
  %790 = select i1 %.not, float %775, float %774, !dbg !7
  %791 = bitcast float %790 to i32, !dbg !7
  %792 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %791), !dbg !7
  %793 = bitcast i32 %792 to float, !dbg !7
  %794 = select i1 %.not, float %774, float %793, !dbg !7
  %795 = select i1 %.not, float %793, float %775, !dbg !7
  %796 = select i1 %.not2, float %794, float %788, !dbg !7
  %797 = xor i32 %778, 32, !dbg !7
  %798 = icmp samesign ult i32 %797, %781, !dbg !7
  %799 = select i1 %798, i32 %797, i32 %778, !dbg !7
  %800 = shl nuw nsw i32 %799, 2, !dbg !7
  %801 = bitcast float %796 to i32, !dbg !7
  %802 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %801), !dbg !7
  %803 = bitcast i32 %802 to float, !dbg !7
  %804 = select i1 %.not2, float %788, float %803, !dbg !7
  %805 = select i1 %.not2, float %803, float %794, !dbg !7
  %806 = select i1 %.not2, float %795, float %789, !dbg !7
  %807 = bitcast float %806 to i32, !dbg !7
  %808 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %807), !dbg !7
  %809 = bitcast i32 %808 to float, !dbg !7
  %810 = select i1 %.not2, float %789, float %809, !dbg !7
  %811 = select i1 %.not2, float %809, float %795, !dbg !7
  %812 = extractelement <8 x float> %63, i64 0, !dbg !7
  %813 = extractelement <8 x float> %63, i64 1, !dbg !7
  %814 = extractelement <8 x float> %63, i64 2, !dbg !7
  %815 = extractelement <8 x float> %63, i64 3, !dbg !7
  %816 = select i1 %.not, float %813, float %812, !dbg !7
  %817 = bitcast float %816 to i32, !dbg !7
  %818 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %817), !dbg !7
  %819 = bitcast i32 %818 to float, !dbg !7
  %820 = select i1 %.not, float %812, float %819, !dbg !7
  %821 = select i1 %.not, float %819, float %813, !dbg !7
  %822 = select i1 %.not, float %815, float %814, !dbg !7
  %823 = bitcast float %822 to i32, !dbg !7
  %824 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %823), !dbg !7
  %825 = bitcast i32 %824 to float, !dbg !7
  %826 = select i1 %.not, float %814, float %825, !dbg !7
  %827 = select i1 %.not, float %825, float %815, !dbg !7
  %828 = select i1 %.not2, float %826, float %820, !dbg !7
  %829 = bitcast float %828 to i32, !dbg !7
  %830 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %829), !dbg !7
  %831 = bitcast i32 %830 to float, !dbg !7
  %832 = select i1 %.not2, float %820, float %831, !dbg !7
  %833 = select i1 %.not2, float %831, float %826, !dbg !7
  %834 = select i1 %.not2, float %827, float %821, !dbg !7
  %835 = bitcast float %834 to i32, !dbg !7
  %836 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %835), !dbg !7
  %837 = bitcast i32 %836 to float, !dbg !7
  %838 = select i1 %.not2, float %821, float %837, !dbg !7
  %839 = select i1 %.not2, float %837, float %827, !dbg !7
  %840 = extractelement <8 x float> %60, i64 0, !dbg !7
  %841 = extractelement <8 x float> %60, i64 1, !dbg !7
  %842 = extractelement <8 x float> %60, i64 2, !dbg !7
  %843 = extractelement <8 x float> %60, i64 3, !dbg !7
  %844 = select i1 %.not, float %841, float %840, !dbg !7
  %845 = bitcast float %844 to i32, !dbg !7
  %846 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %845), !dbg !7
  %847 = bitcast i32 %846 to float, !dbg !7
  %848 = select i1 %.not, float %840, float %847, !dbg !7
  %849 = select i1 %.not, float %847, float %841, !dbg !7
  %850 = select i1 %.not, float %843, float %842, !dbg !7
  %851 = bitcast float %850 to i32, !dbg !7
  %852 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %851), !dbg !7
  %853 = bitcast i32 %852 to float, !dbg !7
  %854 = select i1 %.not, float %842, float %853, !dbg !7
  %855 = select i1 %.not, float %853, float %843, !dbg !7
  %856 = select i1 %.not2, float %854, float %848, !dbg !7
  %857 = bitcast float %856 to i32, !dbg !7
  %858 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %857), !dbg !7
  %859 = bitcast i32 %858 to float, !dbg !7
  %860 = select i1 %.not2, float %848, float %859, !dbg !7
  %861 = select i1 %.not2, float %859, float %854, !dbg !7
  %862 = select i1 %.not2, float %855, float %849, !dbg !7
  %863 = bitcast float %862 to i32, !dbg !7
  %864 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %863), !dbg !7
  %865 = bitcast i32 %864 to float, !dbg !7
  %866 = select i1 %.not2, float %849, float %865, !dbg !7
  %867 = select i1 %.not2, float %865, float %855, !dbg !7
  %868 = extractelement <8 x float> %55, i64 0, !dbg !7
  %869 = extractelement <8 x float> %55, i64 1, !dbg !7
  %870 = extractelement <8 x float> %55, i64 2, !dbg !7
  %871 = extractelement <8 x float> %55, i64 3, !dbg !7
  %872 = select i1 %.not, float %869, float %868, !dbg !7
  %873 = bitcast float %872 to i32, !dbg !7
  %874 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %873), !dbg !7
  %875 = bitcast i32 %874 to float, !dbg !7
  %876 = select i1 %.not, float %868, float %875, !dbg !7
  %877 = select i1 %.not, float %875, float %869, !dbg !7
  %878 = select i1 %.not, float %871, float %870, !dbg !7
  %879 = bitcast float %878 to i32, !dbg !7
  %880 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %879), !dbg !7
  %881 = bitcast i32 %880 to float, !dbg !7
  %882 = select i1 %.not, float %870, float %881, !dbg !7
  %883 = select i1 %.not, float %881, float %871, !dbg !7
  %884 = select i1 %.not2, float %882, float %876, !dbg !7
  %885 = bitcast float %884 to i32, !dbg !7
  %886 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %885), !dbg !7
  %887 = select i1 %.not2, float %883, float %877, !dbg !7
  %888 = bitcast float %887 to i32, !dbg !7
  %889 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %888), !dbg !7
  %890 = shufflevector <1 x float> %56, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %891 = extractelement <8 x float> %66, i64 4, !dbg !7
  %892 = extractelement <8 x float> %66, i64 5, !dbg !7
  %893 = extractelement <8 x float> %66, i64 6, !dbg !7
  %894 = extractelement <8 x float> %66, i64 7, !dbg !7
  %895 = select i1 %.not, float %892, float %891, !dbg !7
  %896 = bitcast float %895 to i32, !dbg !7
  %897 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %896), !dbg !7
  %898 = bitcast i32 %897 to float, !dbg !7
  %899 = select i1 %.not, float %891, float %898, !dbg !7
  %900 = select i1 %.not, float %898, float %892, !dbg !7
  %901 = select i1 %.not, float %894, float %893, !dbg !7
  %902 = bitcast float %901 to i32, !dbg !7
  %903 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %902), !dbg !7
  %904 = bitcast i32 %903 to float, !dbg !7
  %905 = select i1 %.not, float %893, float %904, !dbg !7
  %906 = select i1 %.not, float %904, float %894, !dbg !7
  %907 = select i1 %.not2, float %905, float %899, !dbg !7
  %908 = bitcast float %907 to i32, !dbg !7
  %909 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %908), !dbg !7
  %910 = bitcast i32 %909 to float, !dbg !7
  %911 = select i1 %.not2, float %899, float %910, !dbg !7
  %912 = select i1 %.not2, float %910, float %905, !dbg !7
  %913 = select i1 %.not2, float %906, float %900, !dbg !7
  %914 = bitcast float %913 to i32, !dbg !7
  %915 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %914), !dbg !7
  %916 = bitcast i32 %915 to float, !dbg !7
  %917 = select i1 %.not2, float %900, float %916, !dbg !7
  %918 = select i1 %.not2, float %916, float %906, !dbg !7
  %919 = extractelement <8 x float> %63, i64 4, !dbg !7
  %920 = extractelement <8 x float> %63, i64 5, !dbg !7
  %921 = extractelement <8 x float> %63, i64 6, !dbg !7
  %922 = extractelement <8 x float> %63, i64 7, !dbg !7
  %923 = select i1 %.not, float %920, float %919, !dbg !7
  %924 = bitcast float %923 to i32, !dbg !7
  %925 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %924), !dbg !7
  %926 = bitcast i32 %925 to float, !dbg !7
  %927 = select i1 %.not, float %919, float %926, !dbg !7
  %928 = select i1 %.not, float %926, float %920, !dbg !7
  %929 = select i1 %.not, float %922, float %921, !dbg !7
  %930 = bitcast float %929 to i32, !dbg !7
  %931 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %930), !dbg !7
  %932 = bitcast i32 %931 to float, !dbg !7
  %933 = select i1 %.not, float %921, float %932, !dbg !7
  %934 = select i1 %.not, float %932, float %922, !dbg !7
  %935 = select i1 %.not2, float %933, float %927, !dbg !7
  %936 = bitcast float %935 to i32, !dbg !7
  %937 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %936), !dbg !7
  %938 = bitcast i32 %937 to float, !dbg !7
  %939 = select i1 %.not2, float %927, float %938, !dbg !7
  %940 = select i1 %.not2, float %938, float %933, !dbg !7
  %941 = select i1 %.not2, float %934, float %928, !dbg !7
  %942 = bitcast float %941 to i32, !dbg !7
  %943 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %942), !dbg !7
  %944 = bitcast i32 %943 to float, !dbg !7
  %945 = select i1 %.not2, float %928, float %944, !dbg !7
  %946 = select i1 %.not2, float %944, float %934, !dbg !7
  %947 = extractelement <8 x float> %60, i64 4, !dbg !7
  %948 = extractelement <8 x float> %60, i64 5, !dbg !7
  %949 = extractelement <8 x float> %60, i64 6, !dbg !7
  %950 = extractelement <8 x float> %60, i64 7, !dbg !7
  %951 = select i1 %.not, float %948, float %947, !dbg !7
  %952 = bitcast float %951 to i32, !dbg !7
  %953 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %952), !dbg !7
  %954 = bitcast i32 %953 to float, !dbg !7
  %955 = select i1 %.not, float %947, float %954, !dbg !7
  %956 = select i1 %.not, float %954, float %948, !dbg !7
  %957 = select i1 %.not, float %950, float %949, !dbg !7
  %958 = bitcast float %957 to i32, !dbg !7
  %959 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %958), !dbg !7
  %960 = bitcast i32 %959 to float, !dbg !7
  %961 = select i1 %.not, float %949, float %960, !dbg !7
  %962 = select i1 %.not, float %960, float %950, !dbg !7
  %963 = select i1 %.not2, float %961, float %955, !dbg !7
  %964 = bitcast float %963 to i32, !dbg !7
  %965 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %964), !dbg !7
  %966 = bitcast i32 %965 to float, !dbg !7
  %967 = select i1 %.not2, float %955, float %966, !dbg !7
  %968 = select i1 %.not2, float %966, float %961, !dbg !7
  %969 = select i1 %.not2, float %962, float %956, !dbg !7
  %970 = bitcast float %969 to i32, !dbg !7
  %971 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %970), !dbg !7
  %972 = bitcast i32 %971 to float, !dbg !7
  %973 = select i1 %.not2, float %956, float %972, !dbg !7
  %974 = select i1 %.not2, float %972, float %962, !dbg !7
  %975 = extractelement <8 x float> %55, i64 4, !dbg !7
  %976 = extractelement <8 x float> %55, i64 5, !dbg !7
  %977 = extractelement <8 x float> %55, i64 6, !dbg !7
  %978 = extractelement <8 x float> %55, i64 7, !dbg !7
  %979 = select i1 %.not, float %976, float %975, !dbg !7
  %980 = bitcast float %979 to i32, !dbg !7
  %981 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %980), !dbg !7
  %982 = bitcast i32 %981 to float, !dbg !7
  %983 = select i1 %.not, float %975, float %982, !dbg !7
  %984 = select i1 %.not, float %982, float %976, !dbg !7
  %985 = select i1 %.not, float %978, float %977, !dbg !7
  %986 = bitcast float %985 to i32, !dbg !7
  %987 = call i32 @llvm.amdgcn.ds.bpermute(i32 %784, i32 %986), !dbg !7
  %988 = bitcast i32 %987 to float, !dbg !7
  %989 = select i1 %.not, float %977, float %988, !dbg !7
  %990 = select i1 %.not, float %988, float %978, !dbg !7
  %991 = select i1 %.not2, float %989, float %983, !dbg !7
  %992 = bitcast float %991 to i32, !dbg !7
  %993 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %992), !dbg !7
  %994 = select i1 %.not2, float %990, float %984, !dbg !7
  %995 = bitcast float %994 to i32, !dbg !7
  %996 = call i32 @llvm.amdgcn.ds.bpermute(i32 %800, i32 %995), !dbg !7
  br i1 true, label %997, label %1029, !dbg !21

997:                                              ; preds = %771
  %998 = bitcast i32 %993 to float, !dbg !7
  %999 = select i1 %.not2, float %983, float %998, !dbg !7
  %1000 = insertelement <4 x float> poison, float %999, i64 0, !dbg !19
  %1001 = bitcast i32 %996 to float, !dbg !7
  %1002 = select i1 %.not2, float %984, float %1001, !dbg !7
  %1003 = insertelement <4 x float> %1000, float %1002, i64 1, !dbg !19
  %1004 = select i1 %.not2, float %998, float %989, !dbg !7
  %1005 = insertelement <4 x float> %1003, float %1004, i64 2, !dbg !19
  %1006 = select i1 %.not2, float %1001, float %990, !dbg !7
  %1007 = insertelement <4 x float> %1005, float %1006, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %890, !dbg !19
  %normalized = fmul <4 x float> %1007, %row.reciprocal, !dbg !19
  %1008 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1009 = extractelement <4 x half> %1008, i64 3, !dbg !20
  %1010 = extractelement <4 x half> %1008, i64 2, !dbg !20
  %1011 = extractelement <4 x half> %1008, i64 1, !dbg !20
  %1012 = extractelement <4 x half> %1008, i64 0, !dbg !20
  %1013 = bitcast i32 %886 to float, !dbg !7
  %1014 = select i1 %.not2, float %876, float %1013, !dbg !7
  %1015 = insertelement <4 x float> poison, float %1014, i64 0, !dbg !19
  %1016 = bitcast i32 %889 to float, !dbg !7
  %1017 = select i1 %.not2, float %877, float %1016, !dbg !7
  %1018 = insertelement <4 x float> %1015, float %1017, i64 1, !dbg !19
  %1019 = select i1 %.not2, float %1013, float %882, !dbg !7
  %1020 = insertelement <4 x float> %1018, float %1019, i64 2, !dbg !19
  %1021 = select i1 %.not2, float %1016, float %883, !dbg !7
  %1022 = insertelement <4 x float> %1020, float %1021, i64 3, !dbg !19
  %normalized14 = fmul <4 x float> %1022, %row.reciprocal, !dbg !19
  %1023 = fptrunc <4 x float> %normalized14 to <4 x half>, !dbg !20
  %1024 = extractelement <4 x half> %1023, i64 3, !dbg !20
  %1025 = extractelement <4 x half> %1023, i64 2, !dbg !20
  %1026 = extractelement <4 x half> %1023, i64 1, !dbg !20
  %1027 = extractelement <4 x half> %1023, i64 0, !dbg !20
  %1028 = shl nuw nsw i64 %17, 19, !dbg !21
  %address.thread346 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local347 = zext nneg i32 %address.thread346 to i64, !dbg !13
  %address.local348 = add nsw i64 %18, %address.local347, !dbg !21
  %address.local349 = lshr i64 %address.local347, 6, !dbg !13
  %address.local350 = shl nuw nsw i64 %address.local349, 4, !dbg !21
  %address.local351 = add nsw i64 %address.local348, %address.local350, !dbg !21
  %address.local352 = and i64 %address.local347, 240, !dbg !21
  %address.local353 = sub nsw i64 %address.local351, %address.local352, !dbg !21
  %address.local354 = add nsw i64 %address.local353, %19, !dbg !21
  %address.local355 = shl nuw nsw i64 %address.local354, 7, !dbg !21
  %address.local356 = add nuw nsw i64 %1028, %address.local355, !dbg !21
  %address.local357 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local356, !dbg !21
  %address.local358 = lshr i64 %address.local347, 2, !dbg !13
  %address.local359 = and i64 %address.local358, 12, !dbg !13
  %address.local360 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local357, i64 %address.local359, !dbg !21
  store half %1027, ptr addrspace(1) %address.local360, align 2, !dbg !21
  %address.local361 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local356, !dbg !21
  %address.local362 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local361, i64 %address.local359, !dbg !21
  %address.local363 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local362, i64 2, !dbg !21
  store half %1026, ptr addrspace(1) %address.local363, align 2, !dbg !21
  %address.local364 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local356, !dbg !21
  %address.local365 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local364, i64 %address.local359, !dbg !21
  %address.local366 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local365, i64 4, !dbg !21
  store half %1025, ptr addrspace(1) %address.local366, align 2, !dbg !21
  %address.local367 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local356, !dbg !21
  %address.local368 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local367, i64 %address.local359, !dbg !21
  %address.local369 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local368, i64 6, !dbg !21
  store half %1024, ptr addrspace(1) %address.local369, align 2, !dbg !21
  %address.local370 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local356, !dbg !21
  %address.local371 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local370, i64 %address.local359, !dbg !21
  %address.local372 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local371, i64 32, !dbg !21
  store half %1012, ptr addrspace(1) %address.local372, align 2, !dbg !21
  %address.local373 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local356, !dbg !21
  %address.local374 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local373, i64 %address.local359, !dbg !21
  %address.local375 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local374, i64 34, !dbg !21
  store half %1011, ptr addrspace(1) %address.local375, align 2, !dbg !21
  %address.local376 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local356, !dbg !21
  %address.local377 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local376, i64 %address.local359, !dbg !21
  %address.local378 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local377, i64 36, !dbg !21
  store half %1010, ptr addrspace(1) %address.local378, align 2, !dbg !21
  %address.local379 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local356, !dbg !21
  %address.local380 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local379, i64 %address.local359, !dbg !21
  %address.local381 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local380, i64 38, !dbg !21
  store half %1009, ptr addrspace(1) %address.local381, align 2, !dbg !21
  br label %1029, !dbg !21

1029:                                             ; preds = %997, %771
  br i1 true, label %1030, label %1050, !dbg !21

1030:                                             ; preds = %1029
  %1031 = insertelement <4 x float> poison, float %967, i64 0, !dbg !19
  %1032 = insertelement <4 x float> %1031, float %973, i64 1, !dbg !19
  %1033 = insertelement <4 x float> %1032, float %968, i64 2, !dbg !19
  %1034 = insertelement <4 x float> %1033, float %974, i64 3, !dbg !19
  %row.reciprocal15 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %890, !dbg !19
  %normalized16 = fmul <4 x float> %1034, %row.reciprocal15, !dbg !19
  %1035 = fptrunc <4 x float> %normalized16 to <4 x half>, !dbg !20
  %1036 = extractelement <4 x half> %1035, i64 3, !dbg !20
  %1037 = extractelement <4 x half> %1035, i64 2, !dbg !20
  %1038 = extractelement <4 x half> %1035, i64 1, !dbg !20
  %1039 = extractelement <4 x half> %1035, i64 0, !dbg !20
  %1040 = insertelement <4 x float> poison, float %860, i64 0, !dbg !19
  %1041 = insertelement <4 x float> %1040, float %866, i64 1, !dbg !19
  %1042 = insertelement <4 x float> %1041, float %861, i64 2, !dbg !19
  %1043 = insertelement <4 x float> %1042, float %867, i64 3, !dbg !19
  %normalized17 = fmul <4 x float> %1043, %row.reciprocal15, !dbg !19
  %1044 = fptrunc <4 x float> %normalized17 to <4 x half>, !dbg !20
  %1045 = extractelement <4 x half> %1044, i64 3, !dbg !20
  %1046 = extractelement <4 x half> %1044, i64 2, !dbg !20
  %1047 = extractelement <4 x half> %1044, i64 1, !dbg !20
  %1048 = extractelement <4 x half> %1044, i64 0, !dbg !20
  %1049 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread382 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local383 = zext nneg i32 %address.thread382 to i64, !dbg !13
  %address.local384 = add nsw i64 %18, %address.local383, !dbg !21
  %address.local385 = lshr i64 %address.local383, 6, !dbg !13
  %address.local386 = shl nuw nsw i64 %address.local385, 4, !dbg !21
  %address.local387 = add nsw i64 %address.local384, %address.local386, !dbg !21
  %address.local388 = and i64 %address.local383, 240, !dbg !21
  %address.local389 = sub nsw i64 %address.local387, %address.local388, !dbg !21
  %address.local390 = add nsw i64 %address.local389, %19, !dbg !21
  %address.local391 = shl nuw nsw i64 %address.local390, 7, !dbg !21
  %address.local392 = add nuw nsw i64 %1049, %address.local391, !dbg !21
  %address.local393 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local392, !dbg !21
  %address.local394 = lshr i64 %address.local383, 2, !dbg !13
  %address.local395 = and i64 %address.local394, 12, !dbg !13
  %address.local396 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local393, i64 %address.local395, !dbg !21
  %address.local397 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local396, i64 64, !dbg !21
  store half %1048, ptr addrspace(1) %address.local397, align 2, !dbg !21
  %address.local398 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local392, !dbg !21
  %address.local399 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local398, i64 %address.local395, !dbg !21
  %address.local400 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local399, i64 66, !dbg !21
  store half %1047, ptr addrspace(1) %address.local400, align 2, !dbg !21
  %address.local401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local392, !dbg !21
  %address.local402 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local401, i64 %address.local395, !dbg !21
  %address.local403 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local402, i64 68, !dbg !21
  store half %1046, ptr addrspace(1) %address.local403, align 2, !dbg !21
  %address.local404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local392, !dbg !21
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local404, i64 %address.local395, !dbg !21
  %address.local406 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local405, i64 70, !dbg !21
  store half %1045, ptr addrspace(1) %address.local406, align 2, !dbg !21
  %address.local407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local392, !dbg !21
  %address.local408 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local407, i64 %address.local395, !dbg !21
  %address.local409 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local408, i64 96, !dbg !21
  store half %1039, ptr addrspace(1) %address.local409, align 2, !dbg !21
  %address.local410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local392, !dbg !21
  %address.local411 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local410, i64 %address.local395, !dbg !21
  %address.local412 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local411, i64 98, !dbg !21
  store half %1038, ptr addrspace(1) %address.local412, align 2, !dbg !21
  %address.local413 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local392, !dbg !21
  %address.local414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local413, i64 %address.local395, !dbg !21
  %address.local415 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local414, i64 100, !dbg !21
  store half %1037, ptr addrspace(1) %address.local415, align 2, !dbg !21
  %address.local416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local392, !dbg !21
  %address.local417 = or i64 %address.local394, 51, !dbg !21
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local416, i64 %address.local417, !dbg !21
  store half %1036, ptr addrspace(1) %address.local418, align 2, !dbg !21
  br label %1050, !dbg !21

1050:                                             ; preds = %1030, %1029
  br i1 true, label %1051, label %1071, !dbg !21

1051:                                             ; preds = %1050
  %1052 = insertelement <4 x float> poison, float %939, i64 0, !dbg !19
  %1053 = insertelement <4 x float> %1052, float %945, i64 1, !dbg !19
  %1054 = insertelement <4 x float> %1053, float %940, i64 2, !dbg !19
  %1055 = insertelement <4 x float> %1054, float %946, i64 3, !dbg !19
  %row.reciprocal18 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %890, !dbg !19
  %normalized19 = fmul <4 x float> %1055, %row.reciprocal18, !dbg !19
  %1056 = fptrunc <4 x float> %normalized19 to <4 x half>, !dbg !20
  %1057 = extractelement <4 x half> %1056, i64 3, !dbg !20
  %1058 = extractelement <4 x half> %1056, i64 2, !dbg !20
  %1059 = extractelement <4 x half> %1056, i64 1, !dbg !20
  %1060 = extractelement <4 x half> %1056, i64 0, !dbg !20
  %1061 = insertelement <4 x float> poison, float %832, i64 0, !dbg !19
  %1062 = insertelement <4 x float> %1061, float %838, i64 1, !dbg !19
  %1063 = insertelement <4 x float> %1062, float %833, i64 2, !dbg !19
  %1064 = insertelement <4 x float> %1063, float %839, i64 3, !dbg !19
  %normalized20 = fmul <4 x float> %1064, %row.reciprocal18, !dbg !19
  %1065 = fptrunc <4 x float> %normalized20 to <4 x half>, !dbg !20
  %1066 = extractelement <4 x half> %1065, i64 3, !dbg !20
  %1067 = extractelement <4 x half> %1065, i64 2, !dbg !20
  %1068 = extractelement <4 x half> %1065, i64 1, !dbg !20
  %1069 = extractelement <4 x half> %1065, i64 0, !dbg !20
  %1070 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread419 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local420 = zext nneg i32 %address.thread419 to i64, !dbg !13
  %address.local421 = add nsw i64 %18, %address.local420, !dbg !21
  %address.local422 = lshr i64 %address.local420, 6, !dbg !13
  %address.local423 = shl nuw nsw i64 %address.local422, 4, !dbg !21
  %address.local424 = add nsw i64 %address.local421, %address.local423, !dbg !21
  %address.local425 = and i64 %address.local420, 240, !dbg !21
  %address.local426 = sub nsw i64 %address.local424, %address.local425, !dbg !21
  %address.local427 = add nsw i64 %address.local426, %19, !dbg !21
  %address.local428 = shl nuw nsw i64 %address.local427, 7, !dbg !21
  %address.local429 = add nuw nsw i64 %1070, %address.local428, !dbg !21
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local431 = lshr i64 %address.local420, 2, !dbg !13
  %address.local432 = and i64 %address.local431, 12, !dbg !13
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local430, i64 %address.local432, !dbg !21
  %address.local434 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local433, i64 128, !dbg !21
  store half %1069, ptr addrspace(1) %address.local434, align 2, !dbg !21
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local435, i64 %address.local432, !dbg !21
  %address.local437 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local436, i64 130, !dbg !21
  store half %1068, ptr addrspace(1) %address.local437, align 2, !dbg !21
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local438, i64 %address.local432, !dbg !21
  %address.local440 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local439, i64 132, !dbg !21
  store half %1067, ptr addrspace(1) %address.local440, align 2, !dbg !21
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local441, i64 %address.local432, !dbg !21
  %address.local443 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local442, i64 134, !dbg !21
  store half %1066, ptr addrspace(1) %address.local443, align 2, !dbg !21
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local445 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local444, i64 %address.local432, !dbg !21
  %address.local446 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local445, i64 160, !dbg !21
  store half %1060, ptr addrspace(1) %address.local446, align 2, !dbg !21
  %address.local447 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local447, i64 %address.local432, !dbg !21
  %address.local449 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local448, i64 162, !dbg !21
  store half %1059, ptr addrspace(1) %address.local449, align 2, !dbg !21
  %address.local450 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local451 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local450, i64 %address.local432, !dbg !21
  %address.local452 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local451, i64 164, !dbg !21
  store half %1058, ptr addrspace(1) %address.local452, align 2, !dbg !21
  %address.local453 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local454 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local453, i64 %address.local432, !dbg !21
  %address.local455 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local454, i64 166, !dbg !21
  store half %1057, ptr addrspace(1) %address.local455, align 2, !dbg !21
  br label %1071, !dbg !21

1071:                                             ; preds = %1051, %1050
  br i1 true, label %1072, label %1092, !dbg !21

1072:                                             ; preds = %1071
  %1073 = insertelement <4 x float> poison, float %911, i64 0, !dbg !19
  %1074 = insertelement <4 x float> %1073, float %917, i64 1, !dbg !19
  %1075 = insertelement <4 x float> %1074, float %912, i64 2, !dbg !19
  %1076 = insertelement <4 x float> %1075, float %918, i64 3, !dbg !19
  %row.reciprocal21 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %890, !dbg !19
  %normalized22 = fmul <4 x float> %1076, %row.reciprocal21, !dbg !19
  %1077 = fptrunc <4 x float> %normalized22 to <4 x half>, !dbg !20
  %1078 = extractelement <4 x half> %1077, i64 3, !dbg !20
  %1079 = extractelement <4 x half> %1077, i64 2, !dbg !20
  %1080 = extractelement <4 x half> %1077, i64 1, !dbg !20
  %1081 = extractelement <4 x half> %1077, i64 0, !dbg !20
  %1082 = insertelement <4 x float> poison, float %804, i64 0, !dbg !19
  %1083 = insertelement <4 x float> %1082, float %810, i64 1, !dbg !19
  %1084 = insertelement <4 x float> %1083, float %805, i64 2, !dbg !19
  %1085 = insertelement <4 x float> %1084, float %811, i64 3, !dbg !19
  %normalized23 = fmul <4 x float> %1085, %row.reciprocal21, !dbg !19
  %1086 = fptrunc <4 x float> %normalized23 to <4 x half>, !dbg !20
  %1087 = extractelement <4 x half> %1086, i64 3, !dbg !20
  %1088 = extractelement <4 x half> %1086, i64 2, !dbg !20
  %1089 = extractelement <4 x half> %1086, i64 1, !dbg !20
  %1090 = extractelement <4 x half> %1086, i64 0, !dbg !20
  %1091 = shl nuw nsw i64 %23, 19, !dbg !21
  %address.thread456 = call i32 asm sideeffect "", "=v,0"(i32 %26) #9, !dbg !21, !range !40, !frisk.register_lifetimes !1
  %address.local457 = zext nneg i32 %address.thread456 to i64, !dbg !13
  %address.local458 = add nsw i64 %18, %address.local457, !dbg !21
  %address.local459 = lshr i64 %address.local457, 6, !dbg !13
  %address.local460 = shl nuw nsw i64 %address.local459, 4, !dbg !21
  %address.local461 = add nsw i64 %address.local458, %address.local460, !dbg !21
  %address.local462 = and i64 %address.local457, 240, !dbg !21
  %address.local463 = sub nsw i64 %address.local461, %address.local462, !dbg !21
  %address.local464 = add nsw i64 %address.local463, %19, !dbg !21
  %address.local465 = shl nuw nsw i64 %address.local464, 7, !dbg !21
  %address.local466 = add nuw nsw i64 %1091, %address.local465, !dbg !21
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !21
  %address.local468 = lshr i64 %address.local457, 2, !dbg !13
  %address.local469 = and i64 %address.local468, 12, !dbg !13
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local467, i64 %address.local469, !dbg !21
  %address.local471 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local470, i64 192, !dbg !21
  store half %1090, ptr addrspace(1) %address.local471, align 2, !dbg !21
  %address.local472 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !21
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local472, i64 %address.local469, !dbg !21
  %address.local474 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local473, i64 194, !dbg !21
  store half %1089, ptr addrspace(1) %address.local474, align 2, !dbg !21
  %address.local475 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !21
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local475, i64 %address.local469, !dbg !21
  %address.local477 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local476, i64 196, !dbg !21
  store half %1088, ptr addrspace(1) %address.local477, align 2, !dbg !21
  %address.local478 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !21
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local478, i64 %address.local469, !dbg !21
  %address.local480 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local479, i64 198, !dbg !21
  store half %1087, ptr addrspace(1) %address.local480, align 2, !dbg !21
  %address.local481 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !21
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local481, i64 %address.local469, !dbg !21
  %address.local483 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local482, i64 224, !dbg !21
  store half %1081, ptr addrspace(1) %address.local483, align 2, !dbg !21
  %address.local484 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !21
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local484, i64 %address.local469, !dbg !21
  %address.local486 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local485, i64 226, !dbg !21
  store half %1080, ptr addrspace(1) %address.local486, align 2, !dbg !21
  %address.local487 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !21
  %address.local488 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local487, i64 %address.local469, !dbg !21
  %address.local489 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local488, i64 228, !dbg !21
  store half %1079, ptr addrspace(1) %address.local489, align 2, !dbg !21
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local466, !dbg !21
  %address.local491 = or i64 %address.local468, 115, !dbg !21
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local490, i64 %address.local491, !dbg !21
  store half %1078, ptr addrspace(1) %address.local492, align 2, !dbg !21
  br label %1092, !dbg !21

1092:                                             ; preds = %1072, %1071
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

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!39}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0001-test_input_1x32x4096x128-64-32-32-online/pipeline-loop-only")
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
