; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/retained32-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/retained32-baseline/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !38 {
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
  %15 = sub nsw i64 31, %14, !dbg !5
  %16 = lshr i64 %10, 3, !dbg !5
  %17 = icmp ugt i64 %10, 1023, !dbg !9
  %18 = add nsw i64 %14, -32, !dbg !9
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !9
  %20 = sdiv i64 %19, 32, !dbg !9
  %21 = sext i1 %17 to i64, !dbg !9
  %22 = xor i64 %20, %21, !dbg !9
  %23 = add nsw i64 %16, %22, !dbg !9
  %24 = and i64 %23, 31, !dbg !9
  %25 = shl nsw i64 %15, 7, !dbg !9
  %26 = mul nsw i64 %22, -4096, !dbg !9
  %27 = add i64 %25, %26, !dbg !9
  %28 = shl nuw nsw i64 %24, 19, !dbg !9
  %29 = shl nsw i64 %27, 7, !dbg !9
  %30 = add i64 %28, %29, !dbg !9
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !9, !range !40
  %32 = zext nneg i32 %31 to i64, !dbg !9
  %33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !9
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !9, !range !40, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !9
  %address.local65 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local66 = and i64 %address.local65, 1536, !dbg !9
  %address.local67 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local66, !dbg !9
  %address.local68 = ptrtoint ptr addrspace(3) %address.local67 to i64, !dbg !9
  %address.local69 = trunc i64 %address.local68 to i32, !dbg !9
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local69), !dbg !9
  %35 = zext i32 %34 to i64, !dbg !9
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !9
  %address.local70 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local71 = and i64 %address.local70, 16256, !dbg !9
  %address.local72 = lshr i64 %address.local, 4, !dbg !9
  %address.local73 = and i64 %address.local72, 8, !dbg !9
  %address.local74 = or disjoint i64 %address.local71, %address.local73, !dbg !9
  %address.local75 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local74, !dbg !9
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %38 = ptrtoint ptr addrspace(1) %address.local75 to i64, !dbg !9
  %39 = sub i64 %38, %37, !dbg !9
  %dma.buffer.offset709 = trunc i64 %39 to i32, !dbg !9
  %40 = trunc i64 %37 to i32, !dbg !9
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !9
  %42 = lshr i64 %37, 32, !dbg !9
  %43 = trunc i64 %42 to i32, !dbg !9
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !9
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc710 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !9
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !9
  %dma.buffer.lds711 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !9
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds711, i32 %dma.buffer.offset709, <4 x i32> %dma.buffer.rsrc710) #10, !dbg !9
  %address.local76 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local77 = shl nuw nsw i64 %address.local76, 3, !dbg !9
  %address.local78 = and i64 %address.local77, 3584, !dbg !9
  %address.local79 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local78, !dbg !9
  %address.local80 = ptrtoint ptr addrspace(3) %address.local79 to i64, !dbg !9
  %address.local81 = trunc i64 %address.local80 to i32, !dbg !9
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local81), !dbg !9
  %49 = zext i32 %48 to i64, !dbg !9
  %50 = inttoptr i64 %49 to ptr addrspace(3), !dbg !9
  %address.local82 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local83 = and i64 %address.local82, 16256, !dbg !9
  %address.local84 = lshr i64 %address.local76, 4, !dbg !9
  %address.local85 = and i64 %address.local84, 24, !dbg !9
  %address.local86 = or disjoint i64 %address.local83, %address.local85, !dbg !9
  %address.local87 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local86, !dbg !9
  %51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %52 = ptrtoint ptr addrspace(1) %address.local87 to i64, !dbg !9
  %53 = sub i64 %52, %51, !dbg !9
  %dma.buffer.offset706 = trunc i64 %53 to i32, !dbg !9
  %54 = trunc i64 %51 to i32, !dbg !9
  %55 = insertelement <4 x i32> poison, i32 %54, i64 0, !dbg !9
  %56 = lshr i64 %51, 32, !dbg !9
  %57 = trunc i64 %56 to i32, !dbg !9
  %58 = insertelement <4 x i32> %55, i32 %57, i64 1, !dbg !9
  %59 = insertelement <4 x i32> %58, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc707 = insertelement <4 x i32> %59, i32 131072, i64 3, !dbg !9
  %60 = ptrtoint ptr addrspace(3) %50 to i32, !dbg !9
  %dma.buffer.lds708 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !9
  %61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds708, i32 %dma.buffer.offset706, <4 x i32> %dma.buffer.rsrc707) #10, !dbg !9
  %62 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %63 = sext i32 %62 to i64, !dbg !5
  %64 = shl i64 %15, 7, !dbg !11
  %65 = lshr i64 %32, 6, !dbg !8
  %66 = lshr i64 %32, 2, !dbg !8
  %67 = and i64 %66, 12, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %68 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !9
  %69 = getelementptr i8, ptr addrspace(1) %68, i64 64, !dbg !9
  %address.local88 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local66, !dbg !9
  %address.local89 = ptrtoint ptr addrspace(3) %address.local88 to i64, !dbg !9
  %address.local90 = trunc i64 %address.local89 to i32, !dbg !9
  %70 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local90), !dbg !9
  %71 = zext i32 %70 to i64, !dbg !9
  %72 = inttoptr i64 %71 to ptr addrspace(3), !dbg !9
  %address.local91 = getelementptr [2 x i8], ptr addrspace(1) %69, i64 %address.local74, !dbg !9
  %73 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %74 = ptrtoint ptr addrspace(1) %address.local91 to i64, !dbg !9
  %75 = sub i64 %74, %73, !dbg !9
  %dma.buffer.offset703 = trunc i64 %75 to i32, !dbg !9
  %76 = trunc i64 %73 to i32, !dbg !9
  %77 = insertelement <4 x i32> poison, i32 %76, i64 0, !dbg !9
  %78 = lshr i64 %73, 32, !dbg !9
  %79 = trunc i64 %78 to i32, !dbg !9
  %80 = insertelement <4 x i32> %77, i32 %79, i64 1, !dbg !9
  %81 = insertelement <4 x i32> %80, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc704 = insertelement <4 x i32> %81, i32 131072, i64 3, !dbg !9
  %82 = ptrtoint ptr addrspace(3) %72 to i32, !dbg !9
  %dma.buffer.lds705 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %82), !dbg !9
  %83 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds705, i32 %dma.buffer.offset703, <4 x i32> %dma.buffer.rsrc704) #10, !dbg !9
  %address.local92 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local78, !dbg !9
  %address.local93 = ptrtoint ptr addrspace(3) %address.local92 to i64, !dbg !9
  %address.local94 = trunc i64 %address.local93 to i32, !dbg !9
  %84 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local94), !dbg !9
  %85 = zext i32 %84 to i64, !dbg !9
  %86 = inttoptr i64 %85 to ptr addrspace(3), !dbg !9
  %address.local95 = getelementptr [2 x i8], ptr addrspace(1) %69, i64 %address.local86, !dbg !9
  %87 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %88 = ptrtoint ptr addrspace(1) %address.local95 to i64, !dbg !9
  %89 = sub i64 %88, %87, !dbg !9
  %dma.buffer.offset700 = trunc i64 %89 to i32, !dbg !9
  %90 = trunc i64 %87 to i32, !dbg !9
  %91 = insertelement <4 x i32> poison, i32 %90, i64 0, !dbg !9
  %92 = lshr i64 %87, 32, !dbg !9
  %93 = trunc i64 %92 to i32, !dbg !9
  %94 = insertelement <4 x i32> %91, i32 %93, i64 1, !dbg !9
  %95 = insertelement <4 x i32> %94, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc701 = insertelement <4 x i32> %95, i32 131072, i64 3, !dbg !9
  %96 = ptrtoint ptr addrspace(3) %86 to i32, !dbg !9
  %dma.buffer.lds702 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %96), !dbg !9
  %97 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds702, i32 %dma.buffer.offset700, <4 x i32> %dma.buffer.rsrc701) #10, !dbg !9
  %address.local96 = lshr i32 %address.thread, 2, !dbg !8
  %address.local97 = and i32 %address.local96, 4, !dbg !8
  %address.local98 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local99 = and i32 %address.local98, 384, !dbg !8
  %address.local100 = or disjoint i32 %address.local97, %address.local99, !dbg !8
  %address.local101 = shl nuw nsw i32 %address.thread, 3, !dbg !8
  %address.local102 = and i32 %address.local101, 120, !dbg !8
  %address.local103 = or disjoint i32 %address.local100, %address.local102, !dbg !8
  %address.local104 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local105 = and i32 %address.local104, 1024, !dbg !8
  %address.local106 = or disjoint i32 %address.local103, %address.local105, !dbg !8
  %address.local107 = zext nneg i32 %address.local106 to i64, !dbg !8
  %address.local108 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local107, !dbg !8
  %98 = load <4 x half>, ptr addrspace(3) %address.local108, align 8, !dbg !8
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local108, i64 4096, !dbg !8
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !8
  %100 = fmul <4 x half> %98, splat (half 0xH3015), !dbg !8
  %lds.const.gep626 = getelementptr i8, ptr addrspace(3) %address.local108, i64 1024, !dbg !8
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep626, align 8, !dbg !8
  %102 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !8
  %lds.const.gep627 = getelementptr i8, ptr addrspace(3) %address.local108, i64 5120, !dbg !8
  %103 = load <4 x half>, ptr addrspace(3) %lds.const.gep627, align 8, !dbg !8
  %104 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !8
  %105 = fmul <4 x half> %103, splat (half 0xH3015), !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %106 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !9
  %107 = getelementptr i8, ptr addrspace(1) %106, i64 128, !dbg !9
  %108 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local69), !dbg !9
  %109 = zext i32 %108 to i64, !dbg !9
  %110 = inttoptr i64 %109 to ptr addrspace(3), !dbg !9
  %address.local151 = getelementptr [2 x i8], ptr addrspace(1) %107, i64 %address.local74, !dbg !9
  %111 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %112 = ptrtoint ptr addrspace(1) %address.local151 to i64, !dbg !9
  %113 = sub i64 %112, %111, !dbg !9
  %dma.buffer.offset697 = trunc i64 %113 to i32, !dbg !9
  %114 = trunc i64 %111 to i32, !dbg !9
  %115 = insertelement <4 x i32> poison, i32 %114, i64 0, !dbg !9
  %116 = lshr i64 %111, 32, !dbg !9
  %117 = trunc i64 %116 to i32, !dbg !9
  %118 = insertelement <4 x i32> %115, i32 %117, i64 1, !dbg !9
  %119 = insertelement <4 x i32> %118, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc698 = insertelement <4 x i32> %119, i32 131072, i64 3, !dbg !9
  %120 = ptrtoint ptr addrspace(3) %110 to i32, !dbg !9
  %dma.buffer.lds699 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %120), !dbg !9
  %121 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds699, i32 %dma.buffer.offset697, <4 x i32> %dma.buffer.rsrc698) #10, !dbg !9
  %122 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local81), !dbg !9
  %123 = zext i32 %122 to i64, !dbg !9
  %124 = inttoptr i64 %123 to ptr addrspace(3), !dbg !9
  %address.local152 = getelementptr [2 x i8], ptr addrspace(1) %107, i64 %address.local86, !dbg !9
  %125 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %126 = ptrtoint ptr addrspace(1) %address.local152 to i64, !dbg !9
  %127 = sub i64 %126, %125, !dbg !9
  %dma.buffer.offset694 = trunc i64 %127 to i32, !dbg !9
  %128 = trunc i64 %125 to i32, !dbg !9
  %129 = insertelement <4 x i32> poison, i32 %128, i64 0, !dbg !9
  %130 = lshr i64 %125, 32, !dbg !9
  %131 = trunc i64 %130 to i32, !dbg !9
  %132 = insertelement <4 x i32> %129, i32 %131, i64 1, !dbg !9
  %133 = insertelement <4 x i32> %132, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc695 = insertelement <4 x i32> %133, i32 131072, i64 3, !dbg !9
  %134 = ptrtoint ptr addrspace(3) %124 to i32, !dbg !9
  %dma.buffer.lds696 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %134), !dbg !9
  %135 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds696, i32 %dma.buffer.offset694, <4 x i32> %dma.buffer.rsrc695) #10, !dbg !9
  %lds.const.gep628 = getelementptr i8, ptr addrspace(3) %address.local108, i64 8192, !dbg !8
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep628, align 8, !dbg !8
  %lds.const.gep629 = getelementptr i8, ptr addrspace(3) %address.local108, i64 12288, !dbg !8
  %137 = load <4 x half>, ptr addrspace(3) %lds.const.gep629, align 8, !dbg !8
  %lds.const.gep630 = getelementptr i8, ptr addrspace(3) %address.local108, i64 9216, !dbg !8
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep630, align 8, !dbg !8
  %139 = fmul <4 x half> %136, splat (half 0xH3015), !dbg !8
  %lds.const.gep631 = getelementptr i8, ptr addrspace(3) %address.local108, i64 13312, !dbg !8
  %140 = load <4 x half>, ptr addrspace(3) %lds.const.gep631, align 8, !dbg !8
  %141 = fmul <4 x half> %137, splat (half 0xH3015), !dbg !8
  %142 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !8
  %143 = fmul <4 x half> %140, splat (half 0xH3015), !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %144 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !9
  %145 = getelementptr i8, ptr addrspace(1) %144, i64 192, !dbg !9
  %146 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local90), !dbg !9
  %147 = zext i32 %146 to i64, !dbg !9
  %148 = inttoptr i64 %147 to ptr addrspace(3), !dbg !9
  %address.local157 = getelementptr [2 x i8], ptr addrspace(1) %145, i64 %address.local74, !dbg !9
  %149 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %150 = ptrtoint ptr addrspace(1) %address.local157 to i64, !dbg !9
  %151 = sub i64 %150, %149, !dbg !9
  %dma.buffer.offset691 = trunc i64 %151 to i32, !dbg !9
  %152 = trunc i64 %149 to i32, !dbg !9
  %153 = insertelement <4 x i32> poison, i32 %152, i64 0, !dbg !9
  %154 = lshr i64 %149, 32, !dbg !9
  %155 = trunc i64 %154 to i32, !dbg !9
  %156 = insertelement <4 x i32> %153, i32 %155, i64 1, !dbg !9
  %157 = insertelement <4 x i32> %156, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc692 = insertelement <4 x i32> %157, i32 131072, i64 3, !dbg !9
  %158 = ptrtoint ptr addrspace(3) %148 to i32, !dbg !9
  %dma.buffer.lds693 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %158), !dbg !9
  %159 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds693, i32 %dma.buffer.offset691, <4 x i32> %dma.buffer.rsrc692) #10, !dbg !9
  %160 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local94), !dbg !9
  %161 = zext i32 %160 to i64, !dbg !9
  %162 = inttoptr i64 %161 to ptr addrspace(3), !dbg !9
  %address.local158 = getelementptr [2 x i8], ptr addrspace(1) %145, i64 %address.local86, !dbg !9
  %163 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %164 = ptrtoint ptr addrspace(1) %address.local158 to i64, !dbg !9
  %165 = sub i64 %164, %163, !dbg !9
  %dma.buffer.offset688 = trunc i64 %165 to i32, !dbg !9
  %166 = trunc i64 %163 to i32, !dbg !9
  %167 = insertelement <4 x i32> poison, i32 %166, i64 0, !dbg !9
  %168 = lshr i64 %163, 32, !dbg !9
  %169 = trunc i64 %168 to i32, !dbg !9
  %170 = insertelement <4 x i32> %167, i32 %169, i64 1, !dbg !9
  %171 = insertelement <4 x i32> %170, i32 33554432, i64 2, !dbg !9
  %dma.buffer.rsrc689 = insertelement <4 x i32> %171, i32 131072, i64 3, !dbg !9
  %172 = ptrtoint ptr addrspace(3) %162 to i32, !dbg !9
  %dma.buffer.lds690 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %172), !dbg !9
  %173 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds690, i32 %dma.buffer.offset688, <4 x i32> %dma.buffer.rsrc689) #10, !dbg !9
  %174 = load <4 x half>, ptr addrspace(3) %address.local108, align 8, !dbg !8
  %lds.const.gep632 = getelementptr i8, ptr addrspace(3) %address.local108, i64 4096, !dbg !8
  %175 = load <4 x half>, ptr addrspace(3) %lds.const.gep632, align 8, !dbg !8
  %lds.const.gep633 = getelementptr i8, ptr addrspace(3) %address.local108, i64 1024, !dbg !8
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep633, align 8, !dbg !8
  %177 = fmul <4 x half> %174, splat (half 0xH3015), !dbg !8
  %lds.const.gep634 = getelementptr i8, ptr addrspace(3) %address.local108, i64 5120, !dbg !8
  %178 = load <4 x half>, ptr addrspace(3) %lds.const.gep634, align 8, !dbg !8
  %179 = fmul <4 x half> %175, splat (half 0xH3015), !dbg !8
  %180 = fmul <4 x half> %176, splat (half 0xH3015), !dbg !8
  %181 = fmul <4 x half> %178, splat (half 0xH3015), !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %lds.const.gep635 = getelementptr i8, ptr addrspace(3) %address.local108, i64 8192, !dbg !8
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep635, align 8, !dbg !8
  %lds.const.gep636 = getelementptr i8, ptr addrspace(3) %address.local108, i64 12288, !dbg !8
  %183 = load <4 x half>, ptr addrspace(3) %lds.const.gep636, align 8, !dbg !8
  %lds.const.gep637 = getelementptr i8, ptr addrspace(3) %address.local108, i64 9216, !dbg !8
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep637, align 8, !dbg !8
  %185 = fmul <4 x half> %182, splat (half 0xH3015), !dbg !8
  %lds.const.gep638 = getelementptr i8, ptr addrspace(3) %address.local108, i64 13312, !dbg !8
  %186 = load <4 x half>, ptr addrspace(3) %lds.const.gep638, align 8, !dbg !8
  %187 = fmul <4 x half> %183, splat (half 0xH3015), !dbg !8
  %188 = fmul <4 x half> %184, splat (half 0xH3015), !dbg !8
  %189 = fmul <4 x half> %186, splat (half 0xH3015), !dbg !8
  %190 = and i32 %31, 16, !dbg !14
  %.not = icmp eq i32 %190, 0, !dbg !14
  %191 = and i32 %31, 32, !dbg !14
  %.not1 = icmp eq i32 %191, 0, !dbg !14
  %192 = or disjoint i64 %64, 127, !dbg !15
  %193 = add i64 %64, 128, !dbg !15
  %194 = icmp uge i64 %193, %192, !dbg !15
  %195 = shl nsw i64 %15, 2, !dbg !7
  %196 = add nsw i64 %195, 4, !dbg !7
  %197 = shl i64 %16, 7, !dbg !17
  %198 = and i64 %197, 3968, !dbg !17
  %199 = shl nsw i64 %63, 12, !dbg !17
  %200 = or disjoint i64 %198, %199, !dbg !17
  br label %201, !dbg !7

201:                                              ; preds = %1056, %4
  %202 = phi i64 [ %1127, %1056 ], [ 0, %4 ], !dbg !7
  %203 = phi [2 x <8 x float>] [ %1108, %1056 ], [ zeroinitializer, %4 ], !dbg !7
  %204 = phi [2 x <1 x float>] [ %1112, %1056 ], [ zeroinitializer, %4 ], !dbg !7
  %205 = phi [2 x <1 x float>] [ %1114, %1056 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %206 = phi [2 x <8 x float>] [ %1118, %1056 ], [ zeroinitializer, %4 ], !dbg !7
  %207 = phi [2 x <8 x float>] [ %1122, %1056 ], [ zeroinitializer, %4 ], !dbg !7
  %208 = phi [2 x <8 x float>] [ %1126, %1056 ], [ zeroinitializer, %4 ], !dbg !7
  %209 = icmp slt i64 %202, %196, !dbg !7
  br i1 %209, label %210, label %1128, !dbg !7

210:                                              ; preds = %201
  %211 = shl nsw i64 %202, 5, !dbg !7
  %212 = lshr i64 %202, 14, !dbg !17
  %213 = add nuw nsw i64 %212, %16, !dbg !17
  %214 = lshr i64 %202, 7, !dbg !17
  %215 = and i64 %214, 127, !dbg !17
  %216 = shl i64 %213, 19, !dbg !17
  %217 = and i64 %216, 16252928, !dbg !17
  %218 = shl i64 %214, 19, !dbg !17
  %219 = shl i64 %202, 12, !dbg !17
  %220 = sub i64 %219, %218, !dbg !17
  %221 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %217, !dbg !17
  %222 = getelementptr [2 x i8], ptr addrspace(1) %221, i64 %215, !dbg !17
  %223 = getelementptr [2 x i8], ptr addrspace(1) %222, i64 %220, !dbg !17
  br label %224, !dbg !17

224:                                              ; preds = %227, %210
  %225 = phi i64 [ %257, %227 ], [ %32, %210 ], !dbg !17
  %226 = icmp samesign ult i64 %225, 128, !dbg !17
  br i1 %226, label %227, label %258, !dbg !17

227:                                              ; preds = %224
  %228 = and i64 %225, 27, !dbg !17
  %229 = lshr i64 %225, 2, !dbg !17
  %230 = and i64 %229, 4, !dbg !17
  %231 = add nuw nsw i64 %225, %230, !dbg !17
  %232 = and i64 %231, 4, !dbg !17
  %233 = or disjoint i64 %228, %232, !dbg !17
  %234 = lshr i64 %225, 2, !dbg !17
  %235 = and i64 %234, 24, !dbg !17
  %236 = shl nuw nsw i64 %225, 3, !dbg !17
  %237 = and i64 %236, 512, !dbg !17
  %238 = getelementptr [2 x i8], ptr addrspace(1) %223, i64 %235, !dbg !17
  %.idx22 = shl nuw nsw i64 %233, 8, !dbg !17
  %239 = getelementptr i8, ptr addrspace(1) %238, i64 %.idx22, !dbg !17
  %240 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %237, !dbg !17
  %241 = ptrtoint ptr addrspace(3) %240 to i64, !dbg !17
  %242 = trunc i64 %241 to i32, !dbg !17
  %243 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %242), !dbg !17
  %244 = zext i32 %243 to i64, !dbg !17
  %245 = inttoptr i64 %244 to ptr addrspace(3), !dbg !17
  %246 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %247 = ptrtoint ptr addrspace(1) %239 to i64, !dbg !17
  %248 = sub i64 %247, %246, !dbg !17
  %dma.buffer.offset = trunc i64 %248 to i32, !dbg !17
  %249 = trunc i64 %246 to i32, !dbg !17
  %250 = insertelement <4 x i32> poison, i32 %249, i64 0, !dbg !17
  %251 = lshr i64 %246, 32, !dbg !17
  %252 = trunc i64 %251 to i32, !dbg !17
  %253 = insertelement <4 x i32> %250, i32 %252, i64 1, !dbg !17
  %254 = insertelement <4 x i32> %253, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %254, i32 131072, i64 3, !dbg !17
  %255 = ptrtoint ptr addrspace(3) %245 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %255), !dbg !17
  %256 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %257 = or disjoint i64 %32, 256, !dbg !17
  br label %224, !dbg !17

258:                                              ; preds = %224
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %259 = add nuw i64 %202, 4096, !dbg !17
  %260 = icmp slt i64 %259, 0, !dbg !17
  %261 = sub nuw i64 -4097, %202, !dbg !17
  %262 = select i1 %260, i64 %261, i64 %259, !dbg !17
  %263 = sdiv i64 %262, 16384, !dbg !17
  %.lobit = ashr i64 %259, 63, !dbg !17
  %264 = xor i64 %263, %.lobit, !dbg !17
  %265 = add nsw i64 %264, %16, !dbg !17
  %266 = add nsw i64 %214, %200, !dbg !17
  %267 = add nsw i64 %266, 32, !dbg !17
  %268 = icmp slt i64 %266, -32, !dbg !17
  %269 = sub nsw i64 -33, %266, !dbg !17
  %270 = select i1 %268, i64 %269, i64 %267, !dbg !17
  %271 = sdiv i64 %270, 128, !dbg !17
  %272 = sext i1 %268 to i64, !dbg !17
  %273 = xor i64 %271, %272, !dbg !17
  %274 = shl nsw i64 %273, 7, !dbg !17
  %275 = sub i64 %200, %274, !dbg !17
  %276 = shl i64 %265, 19, !dbg !17
  %277 = and i64 %276, 16252928, !dbg !17
  %278 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %277, !dbg !17
  %279 = getelementptr [2 x i8], ptr addrspace(1) %278, i64 %214, !dbg !17
  %280 = getelementptr [2 x i8], ptr addrspace(1) %279, i64 %275, !dbg !17
  %281 = getelementptr i8, ptr addrspace(1) %280, i64 64, !dbg !17
  %282 = getelementptr [2 x i8], ptr addrspace(1) %281, i64 %220, !dbg !17
  br label %283, !dbg !17

283:                                              ; preds = %286, %258
  %284 = phi i64 [ %316, %286 ], [ %32, %258 ], !dbg !17
  %285 = icmp samesign ult i64 %284, 128, !dbg !17
  br i1 %285, label %286, label %317, !dbg !17

286:                                              ; preds = %283
  %287 = and i64 %284, 27, !dbg !17
  %288 = lshr i64 %284, 2, !dbg !17
  %289 = and i64 %288, 4, !dbg !17
  %290 = add nuw nsw i64 %284, %289, !dbg !17
  %291 = and i64 %290, 4, !dbg !17
  %292 = or disjoint i64 %287, %291, !dbg !17
  %293 = lshr i64 %284, 2, !dbg !17
  %294 = and i64 %293, 24, !dbg !17
  %295 = shl nuw nsw i64 %284, 3, !dbg !17
  %296 = and i64 %295, 512, !dbg !17
  %297 = getelementptr [2 x i8], ptr addrspace(1) %282, i64 %294, !dbg !17
  %.idx21 = shl nuw nsw i64 %292, 8, !dbg !17
  %298 = getelementptr i8, ptr addrspace(1) %297, i64 %.idx21, !dbg !17
  %299 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 2048), i64 %296, !dbg !17
  %300 = ptrtoint ptr addrspace(3) %299 to i64, !dbg !17
  %301 = trunc i64 %300 to i32, !dbg !17
  %302 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %301), !dbg !17
  %303 = zext i32 %302 to i64, !dbg !17
  %304 = inttoptr i64 %303 to ptr addrspace(3), !dbg !17
  %305 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %306 = ptrtoint ptr addrspace(1) %298 to i64, !dbg !17
  %307 = sub i64 %306, %305, !dbg !17
  %dma.buffer.offset667 = trunc i64 %307 to i32, !dbg !17
  %308 = trunc i64 %305 to i32, !dbg !17
  %309 = insertelement <4 x i32> poison, i32 %308, i64 0, !dbg !17
  %310 = lshr i64 %305, 32, !dbg !17
  %311 = trunc i64 %310 to i32, !dbg !17
  %312 = insertelement <4 x i32> %309, i32 %311, i64 1, !dbg !17
  %313 = insertelement <4 x i32> %312, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc668 = insertelement <4 x i32> %313, i32 131072, i64 3, !dbg !17
  %314 = ptrtoint ptr addrspace(3) %304 to i32, !dbg !17
  %dma.buffer.lds669 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %314), !dbg !17
  %315 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds669, i32 %dma.buffer.offset667, <4 x i32> %dma.buffer.rsrc668) #10, !dbg !17
  %316 = or disjoint i64 %32, 256, !dbg !17
  br label %283, !dbg !17

317:                                              ; preds = %283
  %address.thread159 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !18, !range !40, !frisk.register_lifetimes !1
  %address.local160 = lshr i32 %address.thread159, 2, !dbg !18
  %address.local161 = and i32 %address.local160, 4, !dbg !18
  %address.local162 = shl nuw nsw i32 %address.thread159, 3, !dbg !18
  %address.local163 = and i32 %address.local162, 376, !dbg !18
  %address.local164 = or disjoint i32 %address.local161, %address.local163, !dbg !18
  %address.local165 = zext nneg i32 %address.local164 to i64, !dbg !18
  %address.local166 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local165, !dbg !18
  %318 = load <4 x half>, ptr addrspace(3) %address.local166, align 8, !dbg !18
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %318, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep639 = getelementptr i8, ptr addrspace(3) %address.local166, i64 1024, !dbg !18
  %320 = load <4 x half>, ptr addrspace(3) %lds.const.gep639, align 8, !dbg !18
  %address.local175 = lshr i32 %address.thread159, 2, !dbg !18
  %address.local176 = and i32 %address.local175, 4, !dbg !18
  %address.local177 = shl nuw nsw i32 %address.thread159, 3, !dbg !18
  %address.local178 = and i32 %address.local177, 344, !dbg !18
  %address.local179 = or disjoint i32 %address.local176, %address.local178, !dbg !18
  %address.local180 = or disjoint i32 %address.local179, 160, !dbg !18
  %address.local181 = shl nuw nsw i32 %address.thread159, 3, !dbg !18
  %address.local182 = and i32 %address.local181, 32, !dbg !18
  %address.local183 = sub nuw nsw i32 %address.local180, %address.local182, !dbg !18
  %address.local184 = zext nneg i32 %address.local183 to i64, !dbg !18
  %address.local185 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local184, !dbg !18
  %321 = load <4 x half>, ptr addrspace(3) %address.local185, align 8, !dbg !18
  %322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %320, <4 x float> %319) #9, !dbg !18
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %321, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep640 = getelementptr i8, ptr addrspace(3) %address.local185, i64 1024, !dbg !18
  %324 = load <4 x half>, ptr addrspace(3) %lds.const.gep640, align 8, !dbg !18
  %325 = load <4 x half>, ptr addrspace(3) %address.local166, align 8, !dbg !18
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %324, <4 x float> %323) #9, !dbg !18
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %325, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep641 = getelementptr i8, ptr addrspace(3) %address.local166, i64 1024, !dbg !18
  %328 = load <4 x half>, ptr addrspace(3) %lds.const.gep641, align 8, !dbg !18
  %329 = load <4 x half>, ptr addrspace(3) %address.local185, align 8, !dbg !18
  %330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %328, <4 x float> %327) #9, !dbg !18
  %lds.const.gep642 = getelementptr i8, ptr addrspace(3) %address.local185, i64 1024, !dbg !18
  %331 = load <4 x half>, ptr addrspace(3) %lds.const.gep642, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %329, <4 x float> zeroinitializer) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %331, <4 x float> %332) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %334 = add nuw i64 %202, 8192, !dbg !17
  %335 = icmp slt i64 %334, 0, !dbg !17
  %336 = sub nuw i64 -8193, %202, !dbg !17
  %337 = select i1 %335, i64 %336, i64 %334, !dbg !17
  %338 = sdiv i64 %337, 16384, !dbg !17
  %.lobit2 = ashr i64 %334, 63, !dbg !17
  %339 = xor i64 %338, %.lobit2, !dbg !17
  %340 = add nsw i64 %339, %16, !dbg !17
  %341 = add nsw i64 %266, 64, !dbg !17
  %342 = icmp slt i64 %266, -64, !dbg !17
  %343 = sub nsw i64 -65, %266, !dbg !17
  %344 = select i1 %342, i64 %343, i64 %341, !dbg !17
  %345 = sdiv i64 %344, 128, !dbg !17
  %346 = sext i1 %342 to i64, !dbg !17
  %347 = xor i64 %345, %346, !dbg !17
  %348 = shl nsw i64 %347, 7, !dbg !17
  %349 = sub i64 %200, %348, !dbg !17
  %350 = shl i64 %340, 19, !dbg !17
  %351 = and i64 %350, 16252928, !dbg !17
  %352 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %351, !dbg !17
  %353 = getelementptr [2 x i8], ptr addrspace(1) %352, i64 %214, !dbg !17
  %354 = getelementptr [2 x i8], ptr addrspace(1) %353, i64 %349, !dbg !17
  %355 = getelementptr i8, ptr addrspace(1) %354, i64 128, !dbg !17
  %356 = getelementptr [2 x i8], ptr addrspace(1) %355, i64 %220, !dbg !17
  br label %357, !dbg !17

357:                                              ; preds = %360, %317
  %358 = phi i64 [ %390, %360 ], [ %32, %317 ], !dbg !17
  %359 = icmp samesign ult i64 %358, 128, !dbg !17
  br i1 %359, label %360, label %391, !dbg !17

360:                                              ; preds = %357
  %361 = and i64 %358, 27, !dbg !17
  %362 = lshr i64 %358, 2, !dbg !17
  %363 = and i64 %362, 4, !dbg !17
  %364 = add nuw nsw i64 %358, %363, !dbg !17
  %365 = and i64 %364, 4, !dbg !17
  %366 = or disjoint i64 %361, %365, !dbg !17
  %367 = lshr i64 %358, 2, !dbg !17
  %368 = and i64 %367, 24, !dbg !17
  %369 = shl nuw nsw i64 %358, 3, !dbg !17
  %370 = and i64 %369, 512, !dbg !17
  %371 = getelementptr [2 x i8], ptr addrspace(1) %356, i64 %368, !dbg !17
  %.idx20 = shl nuw nsw i64 %366, 8, !dbg !17
  %372 = getelementptr i8, ptr addrspace(1) %371, i64 %.idx20, !dbg !17
  %373 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %370, !dbg !17
  %374 = ptrtoint ptr addrspace(3) %373 to i64, !dbg !17
  %375 = trunc i64 %374 to i32, !dbg !17
  %376 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %375), !dbg !17
  %377 = zext i32 %376 to i64, !dbg !17
  %378 = inttoptr i64 %377 to ptr addrspace(3), !dbg !17
  %379 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %380 = ptrtoint ptr addrspace(1) %372 to i64, !dbg !17
  %381 = sub i64 %380, %379, !dbg !17
  %dma.buffer.offset670 = trunc i64 %381 to i32, !dbg !17
  %382 = trunc i64 %379 to i32, !dbg !17
  %383 = insertelement <4 x i32> poison, i32 %382, i64 0, !dbg !17
  %384 = lshr i64 %379, 32, !dbg !17
  %385 = trunc i64 %384 to i32, !dbg !17
  %386 = insertelement <4 x i32> %383, i32 %385, i64 1, !dbg !17
  %387 = insertelement <4 x i32> %386, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc671 = insertelement <4 x i32> %387, i32 131072, i64 3, !dbg !17
  %388 = ptrtoint ptr addrspace(3) %378 to i32, !dbg !17
  %dma.buffer.lds672 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %388), !dbg !17
  %389 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds672, i32 %dma.buffer.offset670, <4 x i32> %dma.buffer.rsrc671) #10, !dbg !17
  %390 = or disjoint i64 %32, 256, !dbg !17
  br label %357, !dbg !17

391:                                              ; preds = %357
  %lds.const.gep643 = getelementptr i8, ptr addrspace(3) %address.local166, i64 2048, !dbg !18
  %392 = load <4 x half>, ptr addrspace(3) %lds.const.gep643, align 8, !dbg !18
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %392, <4 x float> %322) #9, !dbg !18
  %lds.const.gep644 = getelementptr i8, ptr addrspace(3) %address.local166, i64 3072, !dbg !18
  %394 = load <4 x half>, ptr addrspace(3) %lds.const.gep644, align 8, !dbg !18
  %lds.const.gep645 = getelementptr i8, ptr addrspace(3) %address.local185, i64 2048, !dbg !18
  %395 = load <4 x half>, ptr addrspace(3) %lds.const.gep645, align 8, !dbg !18
  %396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %394, <4 x float> %393) #9, !dbg !18
  %397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %395, <4 x float> %326) #9, !dbg !18
  %lds.const.gep646 = getelementptr i8, ptr addrspace(3) %address.local185, i64 3072, !dbg !18
  %398 = load <4 x half>, ptr addrspace(3) %lds.const.gep646, align 8, !dbg !18
  %lds.const.gep647 = getelementptr i8, ptr addrspace(3) %address.local166, i64 2048, !dbg !18
  %399 = load <4 x half>, ptr addrspace(3) %lds.const.gep647, align 8, !dbg !18
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %398, <4 x float> %397) #9, !dbg !18
  %401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %399, <4 x float> %330) #9, !dbg !18
  %lds.const.gep648 = getelementptr i8, ptr addrspace(3) %address.local166, i64 3072, !dbg !18
  %402 = load <4 x half>, ptr addrspace(3) %lds.const.gep648, align 8, !dbg !18
  %lds.const.gep649 = getelementptr i8, ptr addrspace(3) %address.local185, i64 2048, !dbg !18
  %403 = load <4 x half>, ptr addrspace(3) %lds.const.gep649, align 8, !dbg !18
  %404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %402, <4 x float> %401) #9, !dbg !18
  %lds.const.gep650 = getelementptr i8, ptr addrspace(3) %address.local185, i64 3072, !dbg !18
  %405 = load <4 x half>, ptr addrspace(3) %lds.const.gep650, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %403, <4 x float> %333) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %405, <4 x float> %406) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %408 = add nuw i64 %202, 12288, !dbg !17
  %409 = icmp slt i64 %408, 0, !dbg !17
  %410 = sub nuw i64 -12289, %202, !dbg !17
  %411 = select i1 %409, i64 %410, i64 %408, !dbg !17
  %412 = sdiv i64 %411, 16384, !dbg !17
  %.lobit3 = ashr i64 %408, 63, !dbg !17
  %413 = xor i64 %412, %.lobit3, !dbg !17
  %414 = add nsw i64 %413, %16, !dbg !17
  %415 = add nsw i64 %266, 96, !dbg !17
  %416 = icmp slt i64 %266, -96, !dbg !17
  %417 = sub nsw i64 -97, %266, !dbg !17
  %418 = select i1 %416, i64 %417, i64 %415, !dbg !17
  %419 = sdiv i64 %418, 128, !dbg !17
  %420 = sext i1 %416 to i64, !dbg !17
  %421 = xor i64 %419, %420, !dbg !17
  %422 = shl nsw i64 %421, 7, !dbg !17
  %423 = sub i64 %200, %422, !dbg !17
  %424 = shl i64 %414, 19, !dbg !17
  %425 = and i64 %424, 16252928, !dbg !17
  %426 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %425, !dbg !17
  %427 = getelementptr [2 x i8], ptr addrspace(1) %426, i64 %214, !dbg !17
  %428 = getelementptr [2 x i8], ptr addrspace(1) %427, i64 %423, !dbg !17
  %429 = getelementptr i8, ptr addrspace(1) %428, i64 192, !dbg !17
  %430 = getelementptr [2 x i8], ptr addrspace(1) %429, i64 %220, !dbg !17
  br label %431, !dbg !17

431:                                              ; preds = %434, %391
  %432 = phi i64 [ %464, %434 ], [ %32, %391 ], !dbg !17
  %433 = icmp samesign ult i64 %432, 128, !dbg !17
  br i1 %433, label %434, label %465, !dbg !17

434:                                              ; preds = %431
  %435 = and i64 %432, 27, !dbg !17
  %436 = lshr i64 %432, 2, !dbg !17
  %437 = and i64 %436, 4, !dbg !17
  %438 = add nuw nsw i64 %432, %437, !dbg !17
  %439 = and i64 %438, 4, !dbg !17
  %440 = or disjoint i64 %435, %439, !dbg !17
  %441 = lshr i64 %432, 2, !dbg !17
  %442 = and i64 %441, 24, !dbg !17
  %443 = shl nuw nsw i64 %432, 3, !dbg !17
  %444 = and i64 %443, 512, !dbg !17
  %445 = getelementptr [2 x i8], ptr addrspace(1) %430, i64 %442, !dbg !17
  %.idx = shl nuw nsw i64 %440, 8, !dbg !17
  %446 = getelementptr i8, ptr addrspace(1) %445, i64 %.idx, !dbg !17
  %447 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 2048), i64 %444, !dbg !17
  %448 = ptrtoint ptr addrspace(3) %447 to i64, !dbg !17
  %449 = trunc i64 %448 to i32, !dbg !17
  %450 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %449), !dbg !17
  %451 = zext i32 %450 to i64, !dbg !17
  %452 = inttoptr i64 %451 to ptr addrspace(3), !dbg !17
  %453 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %454 = ptrtoint ptr addrspace(1) %446 to i64, !dbg !17
  %455 = sub i64 %454, %453, !dbg !17
  %dma.buffer.offset673 = trunc i64 %455 to i32, !dbg !17
  %456 = trunc i64 %453 to i32, !dbg !17
  %457 = insertelement <4 x i32> poison, i32 %456, i64 0, !dbg !17
  %458 = lshr i64 %453, 32, !dbg !17
  %459 = trunc i64 %458 to i32, !dbg !17
  %460 = insertelement <4 x i32> %457, i32 %459, i64 1, !dbg !17
  %461 = insertelement <4 x i32> %460, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc674 = insertelement <4 x i32> %461, i32 131072, i64 3, !dbg !17
  %462 = ptrtoint ptr addrspace(3) %452 to i32, !dbg !17
  %dma.buffer.lds675 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %462), !dbg !17
  %463 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds675, i32 %dma.buffer.offset673, <4 x i32> %dma.buffer.rsrc674) #10, !dbg !17
  %464 = or disjoint i64 %32, 256, !dbg !17
  br label %431, !dbg !17

465:                                              ; preds = %431
  %lds.const.gep651 = getelementptr i8, ptr addrspace(3) %address.local166, i64 0, !dbg !18
  %466 = load <4 x half>, ptr addrspace(3) %lds.const.gep651, align 8, !dbg !18
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %466, <4 x float> %396) #9, !dbg !18
  %lds.const.gep652 = getelementptr i8, ptr addrspace(3) %address.local166, i64 1024, !dbg !18
  %468 = load <4 x half>, ptr addrspace(3) %lds.const.gep652, align 8, !dbg !18
  %lds.const.gep653 = getelementptr i8, ptr addrspace(3) %address.local185, i64 0, !dbg !18
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep653, align 8, !dbg !18
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %468, <4 x float> %467) #9, !dbg !18
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %469, <4 x float> %400) #9, !dbg !18
  %lds.const.gep654 = getelementptr i8, ptr addrspace(3) %address.local185, i64 1024, !dbg !18
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep654, align 8, !dbg !18
  %lds.const.gep655 = getelementptr i8, ptr addrspace(3) %address.local166, i64 0, !dbg !18
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep655, align 8, !dbg !18
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %472, <4 x float> %471) #9, !dbg !18
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %473, <4 x float> %404) #9, !dbg !18
  %lds.const.gep656 = getelementptr i8, ptr addrspace(3) %address.local166, i64 1024, !dbg !18
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep656, align 8, !dbg !18
  %lds.const.gep657 = getelementptr i8, ptr addrspace(3) %address.local185, i64 0, !dbg !18
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep657, align 8, !dbg !18
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %181, <4 x half> %476, <4 x float> %475) #9, !dbg !18
  %lds.const.gep658 = getelementptr i8, ptr addrspace(3) %address.local185, i64 1024, !dbg !18
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep658, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %180, <4 x half> %477, <4 x float> %407) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %181, <4 x half> %479, <4 x float> %480) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %482 = add nuw nsw i64 %214, %16, !dbg !19
  %483 = shl i64 %482, 19, !dbg !19
  %484 = and i64 %483, 16252928, !dbg !19
  %485 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %484, !dbg !19
  %486 = getelementptr [2 x i8], ptr addrspace(1) %485, i64 %220, !dbg !19
  br label %487, !dbg !19

487:                                              ; preds = %490, %465
  %488 = phi i64 [ %516, %490 ], [ %32, %465 ], !dbg !19
  %489 = icmp samesign ult i64 %488, 128, !dbg !19
  br i1 %489, label %490, label %517, !dbg !19

490:                                              ; preds = %487
  %491 = shl nuw nsw i64 %488, 3, !dbg !19
  %492 = and i64 %491, 24, !dbg !19
  %493 = shl nuw nsw i64 %488, 3, !dbg !19
  %494 = and i64 %493, 512, !dbg !19
  %495 = shl nuw nsw i64 %488, 5, !dbg !19
  %496 = and i64 %495, 3968, !dbg !19
  %497 = getelementptr [2 x i8], ptr addrspace(1) %486, i64 %496, !dbg !19
  %498 = getelementptr [2 x i8], ptr addrspace(1) %497, i64 %492, !dbg !19
  %499 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %494, !dbg !19
  %500 = ptrtoint ptr addrspace(3) %499 to i64, !dbg !19
  %501 = trunc i64 %500 to i32, !dbg !19
  %502 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %501), !dbg !19
  %503 = zext i32 %502 to i64, !dbg !19
  %504 = inttoptr i64 %503 to ptr addrspace(3), !dbg !19
  %505 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %506 = ptrtoint ptr addrspace(1) %498 to i64, !dbg !19
  %507 = sub i64 %506, %505, !dbg !19
  %dma.buffer.offset676 = trunc i64 %507 to i32, !dbg !19
  %508 = trunc i64 %505 to i32, !dbg !19
  %509 = insertelement <4 x i32> poison, i32 %508, i64 0, !dbg !19
  %510 = lshr i64 %505, 32, !dbg !19
  %511 = trunc i64 %510 to i32, !dbg !19
  %512 = insertelement <4 x i32> %509, i32 %511, i64 1, !dbg !19
  %513 = insertelement <4 x i32> %512, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc677 = insertelement <4 x i32> %513, i32 131072, i64 3, !dbg !19
  %514 = ptrtoint ptr addrspace(3) %504 to i32, !dbg !19
  %dma.buffer.lds678 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %514), !dbg !19
  %515 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds678, i32 %dma.buffer.offset676, <4 x i32> %dma.buffer.rsrc677) #10, !dbg !19
  %516 = or disjoint i64 %32, 256, !dbg !19
  br label %487, !dbg !19

517:                                              ; preds = %487
  %lds.const.gep659 = getelementptr i8, ptr addrspace(3) %address.local166, i64 2048, !dbg !18
  %518 = load <4 x half>, ptr addrspace(3) %lds.const.gep659, align 8, !dbg !18
  %519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %518, <4 x float> %470) #9, !dbg !18
  %lds.const.gep660 = getelementptr i8, ptr addrspace(3) %address.local166, i64 3072, !dbg !18
  %520 = load <4 x half>, ptr addrspace(3) %lds.const.gep660, align 8, !dbg !18
  %lds.const.gep661 = getelementptr i8, ptr addrspace(3) %address.local185, i64 2048, !dbg !18
  %521 = load <4 x half>, ptr addrspace(3) %lds.const.gep661, align 8, !dbg !18
  %522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %520, <4 x float> %519) #9, !dbg !18
  %523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %521, <4 x float> %474) #9, !dbg !18
  %524 = extractelement <4 x float> %522, i64 0, !dbg !23
  %525 = extractelement <4 x float> %522, i64 1, !dbg !23
  %526 = select i1 %.not, float %525, float %524, !dbg !14
  %527 = extractelement <4 x float> %522, i64 2, !dbg !23
  %528 = extractelement <4 x float> %522, i64 3, !dbg !23
  %lds.const.gep662 = getelementptr i8, ptr addrspace(3) %address.local185, i64 3072, !dbg !18
  %529 = load <4 x half>, ptr addrspace(3) %lds.const.gep662, align 8, !dbg !18
  %lds.const.gep663 = getelementptr i8, ptr addrspace(3) %address.local166, i64 2048, !dbg !18
  %530 = load <4 x half>, ptr addrspace(3) %lds.const.gep663, align 8, !dbg !18
  %531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %529, <4 x float> %523) #9, !dbg !18
  %532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %188, <4 x half> %530, <4 x float> %478) #9, !dbg !18
  %533 = extractelement <4 x float> %531, i64 0, !dbg !23
  %534 = extractelement <4 x float> %531, i64 1, !dbg !23
  %535 = extractelement <4 x float> %531, i64 2, !dbg !23
  %536 = extractelement <4 x float> %531, i64 3, !dbg !23
  %lds.const.gep664 = getelementptr i8, ptr addrspace(3) %address.local166, i64 3072, !dbg !18
  %537 = load <4 x half>, ptr addrspace(3) %lds.const.gep664, align 8, !dbg !18
  %lds.const.gep665 = getelementptr i8, ptr addrspace(3) %address.local185, i64 2048, !dbg !18
  %538 = load <4 x half>, ptr addrspace(3) %lds.const.gep665, align 8, !dbg !18
  %539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %189, <4 x half> %537, <4 x float> %532) #9, !dbg !18
  %lds.const.gep666 = getelementptr i8, ptr addrspace(3) %address.local185, i64 3072, !dbg !18
  %540 = load <4 x half>, ptr addrspace(3) %lds.const.gep666, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %188, <4 x half> %538, <4 x float> %481) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !23
  %542 = extractelement <4 x float> %539, i64 0, !dbg !23
  %543 = extractelement <4 x float> %539, i64 1, !dbg !23
  %544 = extractelement <4 x float> %539, i64 2, !dbg !23
  %545 = extractelement <4 x float> %539, i64 3, !dbg !23
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %189, <4 x half> %540, <4 x float> %541) #9, !dbg !18
  %547 = extractelement <4 x float> %546, i64 0, !dbg !23
  %548 = extractelement <4 x float> %546, i64 1, !dbg !23
  %549 = extractelement <4 x float> %546, i64 2, !dbg !23
  %550 = extractelement <4 x float> %546, i64 3, !dbg !23
  %551 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !14
  %552 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %551), !dbg !14
  %553 = add nuw nsw i32 %552, 64, !dbg !14
  %554 = xor i32 %552, 16, !dbg !14
  %555 = and i32 %553, 192, !dbg !14
  %556 = icmp samesign ult i32 %554, %555, !dbg !14
  %557 = select i1 %556, i32 %554, i32 %552, !dbg !14
  %558 = shl nuw nsw i32 %557, 2, !dbg !14
  %559 = bitcast float %526 to i32, !dbg !14
  %560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %559), !dbg !14
  %561 = bitcast i32 %560 to float, !dbg !14
  %562 = select i1 %.not, float %524, float %561, !dbg !14
  %563 = select i1 %.not, float %561, float %525, !dbg !14
  %564 = select i1 %.not, float %528, float %527, !dbg !14
  %565 = bitcast float %564 to i32, !dbg !14
  %566 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %565), !dbg !14
  %567 = bitcast i32 %566 to float, !dbg !14
  %568 = select i1 %.not, float %527, float %567, !dbg !14
  %569 = select i1 %.not, float %567, float %528, !dbg !14
  %570 = select i1 %.not1, float %568, float %562, !dbg !14
  %571 = xor i32 %552, 32, !dbg !14
  %572 = icmp samesign ult i32 %571, %555, !dbg !14
  %573 = select i1 %572, i32 %571, i32 %552, !dbg !14
  %574 = shl nuw nsw i32 %573, 2, !dbg !14
  %575 = bitcast float %570 to i32, !dbg !14
  %576 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %575), !dbg !14
  %577 = bitcast i32 %576 to float, !dbg !14
  %578 = select i1 %.not1, float %562, float %577, !dbg !14
  %579 = select i1 %.not1, float %577, float %568, !dbg !14
  %580 = select i1 %.not1, float %569, float %563, !dbg !14
  %581 = bitcast float %580 to i32, !dbg !14
  %582 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %581), !dbg !14
  %583 = bitcast i32 %582 to float, !dbg !14
  %584 = select i1 %.not1, float %563, float %583, !dbg !14
  %585 = select i1 %.not1, float %583, float %569, !dbg !14
  %586 = select i1 %.not, float %534, float %533, !dbg !14
  %587 = bitcast float %586 to i32, !dbg !14
  %588 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %587), !dbg !14
  %589 = bitcast i32 %588 to float, !dbg !14
  %590 = select i1 %.not, float %533, float %589, !dbg !14
  %591 = select i1 %.not, float %589, float %534, !dbg !14
  %592 = select i1 %.not, float %536, float %535, !dbg !14
  %593 = bitcast float %592 to i32, !dbg !14
  %594 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %593), !dbg !14
  %595 = bitcast i32 %594 to float, !dbg !14
  %596 = select i1 %.not, float %535, float %595, !dbg !14
  %597 = select i1 %.not, float %595, float %536, !dbg !14
  %598 = select i1 %.not1, float %596, float %590, !dbg !14
  %599 = bitcast float %598 to i32, !dbg !14
  %600 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %599), !dbg !14
  %601 = bitcast i32 %600 to float, !dbg !14
  %602 = select i1 %.not1, float %590, float %601, !dbg !14
  %603 = select i1 %.not1, float %601, float %596, !dbg !14
  %604 = select i1 %.not1, float %597, float %591, !dbg !14
  %605 = bitcast float %604 to i32, !dbg !14
  %606 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %605), !dbg !14
  %607 = bitcast i32 %606 to float, !dbg !14
  %608 = select i1 %.not1, float %591, float %607, !dbg !14
  %609 = select i1 %.not1, float %607, float %597, !dbg !14
  %610 = select i1 %.not, float %543, float %542, !dbg !14
  %611 = bitcast float %610 to i32, !dbg !14
  %612 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %611), !dbg !14
  %613 = bitcast i32 %612 to float, !dbg !14
  %614 = select i1 %.not, float %542, float %613, !dbg !14
  %615 = select i1 %.not, float %613, float %543, !dbg !14
  %616 = select i1 %.not, float %545, float %544, !dbg !14
  %617 = bitcast float %616 to i32, !dbg !14
  %618 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %617), !dbg !14
  %619 = bitcast i32 %618 to float, !dbg !14
  %620 = select i1 %.not, float %544, float %619, !dbg !14
  %621 = select i1 %.not, float %619, float %545, !dbg !14
  %622 = select i1 %.not1, float %620, float %614, !dbg !14
  %623 = bitcast float %622 to i32, !dbg !14
  %624 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %623), !dbg !14
  %625 = bitcast i32 %624 to float, !dbg !14
  %626 = select i1 %.not1, float %614, float %625, !dbg !14
  %627 = select i1 %.not1, float %625, float %620, !dbg !14
  %628 = select i1 %.not1, float %621, float %615, !dbg !14
  %629 = bitcast float %628 to i32, !dbg !14
  %630 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %629), !dbg !14
  %631 = bitcast i32 %630 to float, !dbg !14
  %632 = select i1 %.not1, float %615, float %631, !dbg !14
  %633 = select i1 %.not1, float %631, float %621, !dbg !14
  %634 = select i1 %.not, float %548, float %547, !dbg !14
  %635 = bitcast float %634 to i32, !dbg !14
  %636 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %635), !dbg !14
  %637 = bitcast i32 %636 to float, !dbg !14
  %638 = select i1 %.not, float %547, float %637, !dbg !14
  %639 = select i1 %.not, float %637, float %548, !dbg !14
  %640 = select i1 %.not, float %550, float %549, !dbg !14
  %641 = bitcast float %640 to i32, !dbg !14
  %642 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %641), !dbg !14
  %643 = bitcast i32 %642 to float, !dbg !14
  %644 = select i1 %.not, float %549, float %643, !dbg !14
  %645 = select i1 %.not, float %643, float %550, !dbg !14
  %646 = select i1 %.not1, float %644, float %638, !dbg !14
  %647 = bitcast float %646 to i32, !dbg !14
  %648 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %647), !dbg !14
  %649 = bitcast i32 %648 to float, !dbg !14
  %650 = select i1 %.not1, float %638, float %649, !dbg !14
  %651 = select i1 %.not1, float %649, float %644, !dbg !14
  %652 = select i1 %.not1, float %645, float %639, !dbg !14
  %653 = bitcast float %652 to i32, !dbg !14
  %654 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %653), !dbg !14
  %655 = bitcast i32 %654 to float, !dbg !14
  %656 = select i1 %.not1, float %639, float %655, !dbg !14
  %657 = select i1 %.not1, float %655, float %645, !dbg !14
  %658 = or disjoint i64 %211, 31, !dbg !15
  %659 = icmp uge i64 %64, %658, !dbg !15
  %660 = and i1 %194, %659, !dbg !15
  br i1 %660, label %661, label %662, !dbg !15

661:                                              ; preds = %517
  br label %709, !dbg !15

662:                                              ; preds = %517
  %663 = add i64 %64, %32, !dbg !15
  %664 = shl nuw nsw i64 %65, 4, !dbg !15
  %665 = add i64 %663, %664, !dbg !15
  %666 = and i32 %31, 240, !dbg !15
  %index.compact = zext nneg i32 %666 to i64, !dbg !15
  %667 = sub i64 %665, %index.compact, !dbg !15
  %668 = or disjoint i64 %211, %67, !dbg !15
  %669 = add i64 %667, 1, !dbg !24
  %670 = lshr i64 %669, 32, !dbg !25
  %compare.high = trunc i64 %670 to i32, !dbg !25
  %671 = lshr i64 %668, 32, !dbg !25
  %compare.high562 = trunc i64 %671 to i32, !dbg !25
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high562, !dbg !25
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high562, !dbg !25
  %compare.low.base = trunc i64 %669 to i32, !dbg !25
  %compare.low.base563 = trunc i64 %668 to i32, !dbg !25
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base563, !dbg !25
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !25
  %672 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 1) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !25
  %compare.low564 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !25
  %compare.result565 = select i1 %compare.high.equal, i1 %compare.low564, i1 %compare.high.order, !dbg !25
  %673 = select i1 %compare.result565, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base566 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 2) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local567 = or i32 %compare.local.base566, 2, !dbg !25
  %compare.low568 = icmp ugt i32 %compare.low.base, %compare.local567, !dbg !25
  %compare.result569 = select i1 %compare.high.equal, i1 %compare.low568, i1 %compare.high.order, !dbg !25
  %674 = select i1 %compare.result569, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base570 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 3) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local571 = or i32 %compare.local.base570, 3, !dbg !25
  %compare.low572 = icmp ugt i32 %compare.low.base, %compare.local571, !dbg !25
  %compare.result573 = select i1 %compare.high.equal, i1 %compare.low572, i1 %compare.high.order, !dbg !25
  %675 = select i1 %compare.result573, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base574 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 4) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local575 = or i32 %compare.local.base574, 16, !dbg !25
  %compare.low576 = icmp ugt i32 %compare.low.base, %compare.local575, !dbg !25
  %compare.result577 = select i1 %compare.high.equal, i1 %compare.low576, i1 %compare.high.order, !dbg !25
  %676 = select i1 %compare.result577, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base578 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 5) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local579 = or i32 %compare.local.base578, 17, !dbg !25
  %compare.low580 = icmp ugt i32 %compare.low.base, %compare.local579, !dbg !25
  %compare.result581 = select i1 %compare.high.equal, i1 %compare.low580, i1 %compare.high.order, !dbg !25
  %677 = select i1 %compare.result581, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base582 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 6) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local583 = or i32 %compare.local.base582, 18, !dbg !25
  %compare.low584 = icmp ugt i32 %compare.low.base, %compare.local583, !dbg !25
  %compare.result585 = select i1 %compare.high.equal, i1 %compare.low584, i1 %compare.high.order, !dbg !25
  %678 = select i1 %compare.result585, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base586 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 7) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local587 = or i32 %compare.local.base586, 19, !dbg !25
  %compare.low588 = icmp ugt i32 %compare.low.base, %compare.local587, !dbg !25
  %compare.result589 = select i1 %compare.high.equal, i1 %compare.low588, i1 %compare.high.order, !dbg !25
  %679 = select i1 %compare.result589, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %680 = add i64 %667, 65, !dbg !24
  %681 = lshr i64 %680, 32, !dbg !25
  %compare.high590 = trunc i64 %681 to i32, !dbg !25
  %682 = lshr i64 %668, 32, !dbg !25
  %compare.high591 = trunc i64 %682 to i32, !dbg !25
  %compare.high.equal592 = icmp eq i32 %compare.high590, %compare.high591, !dbg !25
  %compare.high.order593 = icmp ugt i32 %compare.high590, %compare.high591, !dbg !25
  %compare.low.base594 = trunc i64 %680 to i32, !dbg !25
  %compare.low.base595 = trunc i64 %668 to i32, !dbg !25
  %compare.low596 = icmp ugt i32 %compare.low.base594, %compare.low.base595, !dbg !25
  %compare.result597 = select i1 %compare.high.equal592, i1 %compare.low596, i1 %compare.high.order593, !dbg !25
  %683 = select i1 %compare.result597, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base598 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base595, i32 8) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local599 = or i32 %compare.local.base598, 1, !dbg !25
  %compare.low600 = icmp ugt i32 %compare.low.base594, %compare.local599, !dbg !25
  %compare.result601 = select i1 %compare.high.equal592, i1 %compare.low600, i1 %compare.high.order593, !dbg !25
  %684 = select i1 %compare.result601, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base602 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base595, i32 9) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local603 = or i32 %compare.local.base602, 2, !dbg !25
  %compare.low604 = icmp ugt i32 %compare.low.base594, %compare.local603, !dbg !25
  %compare.result605 = select i1 %compare.high.equal592, i1 %compare.low604, i1 %compare.high.order593, !dbg !25
  %685 = select i1 %compare.result605, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base606 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base595, i32 10) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local607 = or i32 %compare.local.base606, 3, !dbg !25
  %compare.low608 = icmp ugt i32 %compare.low.base594, %compare.local607, !dbg !25
  %compare.result609 = select i1 %compare.high.equal592, i1 %compare.low608, i1 %compare.high.order593, !dbg !25
  %686 = select i1 %compare.result609, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base610 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base595, i32 11) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local611 = or i32 %compare.local.base610, 16, !dbg !25
  %compare.low612 = icmp ugt i32 %compare.low.base594, %compare.local611, !dbg !25
  %compare.result613 = select i1 %compare.high.equal592, i1 %compare.low612, i1 %compare.high.order593, !dbg !25
  %687 = select i1 %compare.result613, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base614 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base595, i32 12) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local615 = or i32 %compare.local.base614, 17, !dbg !25
  %compare.low616 = icmp ugt i32 %compare.low.base594, %compare.local615, !dbg !25
  %compare.result617 = select i1 %compare.high.equal592, i1 %compare.low616, i1 %compare.high.order593, !dbg !25
  %688 = select i1 %compare.result617, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base618 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base595, i32 13) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local619 = or i32 %compare.local.base618, 18, !dbg !25
  %compare.low620 = icmp ugt i32 %compare.low.base594, %compare.local619, !dbg !25
  %compare.result621 = select i1 %compare.high.equal592, i1 %compare.low620, i1 %compare.high.order593, !dbg !25
  %689 = select i1 %compare.result621, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base622 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base595, i32 14) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local623 = or i32 %compare.local.base622, 19, !dbg !25
  %compare.low624 = icmp ugt i32 %compare.low.base594, %compare.local623, !dbg !25
  %compare.result625 = select i1 %compare.high.equal592, i1 %compare.low624, i1 %compare.high.order593, !dbg !25
  %690 = select i1 %compare.result625, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %691 = insertelement <8 x float> poison, float %672, i64 0, !dbg !15
  %692 = insertelement <8 x float> %691, float %673, i64 1, !dbg !15
  %693 = insertelement <8 x float> %692, float %674, i64 2, !dbg !15
  %694 = insertelement <8 x float> %693, float %675, i64 3, !dbg !15
  %695 = insertelement <8 x float> %694, float %676, i64 4, !dbg !15
  %696 = insertelement <8 x float> %695, float %677, i64 5, !dbg !15
  %697 = insertelement <8 x float> %696, float %678, i64 6, !dbg !15
  %698 = insertelement <8 x float> %697, float %679, i64 7, !dbg !15
  %699 = insertvalue [2 x <8 x float>] poison, <8 x float> %698, 0, !dbg !15
  %700 = insertelement <8 x float> poison, float %683, i64 0, !dbg !15
  %701 = insertelement <8 x float> %700, float %684, i64 1, !dbg !15
  %702 = insertelement <8 x float> %701, float %685, i64 2, !dbg !15
  %703 = insertelement <8 x float> %702, float %686, i64 3, !dbg !15
  %704 = insertelement <8 x float> %703, float %687, i64 4, !dbg !15
  %705 = insertelement <8 x float> %704, float %688, i64 5, !dbg !15
  %706 = insertelement <8 x float> %705, float %689, i64 6, !dbg !15
  %707 = insertelement <8 x float> %706, float %690, i64 7, !dbg !15
  %708 = insertvalue [2 x <8 x float>] %699, <8 x float> %707, 1, !dbg !15
  br label %709, !dbg !15

709:                                              ; preds = %661, %662
  %710 = phi [2 x <8 x float>] [ %708, %662 ], [ zeroinitializer, %661 ], !dbg !15
  br label %711, !dbg !15

711:                                              ; preds = %709
  %712 = extractvalue [2 x <8 x float>] %710, 0, !dbg !27
  %713 = insertelement <4 x float> poison, float %578, i64 0, !dbg !27
  %714 = insertelement <4 x float> %713, float %584, i64 1, !dbg !27
  %715 = insertelement <4 x float> %714, float %579, i64 2, !dbg !27
  %716 = insertelement <4 x float> %715, float %585, i64 3, !dbg !27
  %717 = shufflevector <8 x float> %712, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %718 = fadd <4 x float> %716, %717, !dbg !27
  %719 = extractelement <4 x float> %718, i64 0, !dbg !27
  %720 = extractelement <4 x float> %718, i64 1, !dbg !27
  %721 = extractelement <4 x float> %718, i64 2, !dbg !27
  %722 = extractelement <4 x float> %718, i64 3, !dbg !27
  %723 = insertelement <4 x float> poison, float %602, i64 0, !dbg !27
  %724 = insertelement <4 x float> %723, float %608, i64 1, !dbg !27
  %725 = insertelement <4 x float> %724, float %603, i64 2, !dbg !27
  %726 = insertelement <4 x float> %725, float %609, i64 3, !dbg !27
  %727 = shufflevector <8 x float> %712, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %728 = fadd <4 x float> %726, %727, !dbg !27
  %729 = extractelement <4 x float> %728, i64 0, !dbg !27
  %730 = extractelement <4 x float> %728, i64 1, !dbg !27
  %731 = extractelement <4 x float> %728, i64 2, !dbg !27
  %732 = extractelement <4 x float> %728, i64 3, !dbg !27
  %733 = extractvalue [2 x <8 x float>] %710, 1, !dbg !27
  %734 = insertelement <4 x float> poison, float %626, i64 0, !dbg !27
  %735 = insertelement <4 x float> %734, float %632, i64 1, !dbg !27
  %736 = insertelement <4 x float> %735, float %627, i64 2, !dbg !27
  %737 = insertelement <4 x float> %736, float %633, i64 3, !dbg !27
  %738 = shufflevector <8 x float> %733, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %739 = fadd <4 x float> %737, %738, !dbg !27
  %740 = extractelement <4 x float> %739, i64 0, !dbg !27
  %741 = extractelement <4 x float> %739, i64 1, !dbg !27
  %742 = extractelement <4 x float> %739, i64 2, !dbg !27
  %743 = extractelement <4 x float> %739, i64 3, !dbg !27
  %744 = insertelement <4 x float> poison, float %650, i64 0, !dbg !27
  %745 = insertelement <4 x float> %744, float %656, i64 1, !dbg !27
  %746 = insertelement <4 x float> %745, float %651, i64 2, !dbg !27
  %747 = insertelement <4 x float> %746, float %657, i64 3, !dbg !27
  %748 = shufflevector <8 x float> %733, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %749 = fadd <4 x float> %747, %748, !dbg !27
  %750 = extractelement <4 x float> %749, i64 0, !dbg !27
  %751 = extractelement <4 x float> %749, i64 1, !dbg !27
  %752 = extractelement <4 x float> %749, i64 2, !dbg !27
  %753 = extractelement <4 x float> %749, i64 3, !dbg !27
  %754 = call float @llvm.maxnum.f32(float %719, float 0xFFF0000000000000), !dbg !28
  %755 = call float @llvm.maxnum.f32(float %754, float %720), !dbg !28
  %756 = call float @llvm.maxnum.f32(float %755, float %721), !dbg !28
  %757 = call float @llvm.maxnum.f32(float %756, float %722), !dbg !28
  %758 = call float @llvm.maxnum.f32(float %757, float %729), !dbg !28
  %759 = call float @llvm.maxnum.f32(float %758, float %730), !dbg !28
  %760 = call float @llvm.maxnum.f32(float %759, float %731), !dbg !28
  %761 = call float @llvm.maxnum.f32(float %760, float %732), !dbg !28
  %762 = bitcast float %761 to i32, !dbg !28
  %763 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %762), !dbg !28
  %764 = bitcast i32 %763 to float, !dbg !28
  %765 = call float @llvm.maxnum.f32(float %761, float %764), !dbg !28
  %766 = bitcast float %765 to i32, !dbg !28
  %767 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %766), !dbg !28
  %768 = bitcast i32 %767 to float, !dbg !28
  %769 = call float @llvm.maxnum.f32(float %765, float %768), !dbg !28
  %770 = extractvalue [2 x <1 x float>] %205, 0, !dbg !28
  %771 = insertelement <1 x float> poison, float %769, i64 0, !dbg !28
  %772 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %771, <1 x float> %770), !dbg !28
  %773 = call float @llvm.maxnum.f32(float %740, float 0xFFF0000000000000), !dbg !28
  %774 = call float @llvm.maxnum.f32(float %773, float %741), !dbg !28
  %775 = call float @llvm.maxnum.f32(float %774, float %742), !dbg !28
  %776 = call float @llvm.maxnum.f32(float %775, float %743), !dbg !28
  %777 = call float @llvm.maxnum.f32(float %776, float %750), !dbg !28
  %778 = call float @llvm.maxnum.f32(float %777, float %751), !dbg !28
  %779 = call float @llvm.maxnum.f32(float %778, float %752), !dbg !28
  %780 = call float @llvm.maxnum.f32(float %779, float %753), !dbg !28
  %781 = bitcast float %780 to i32, !dbg !28
  %782 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %781), !dbg !28
  %783 = bitcast i32 %782 to float, !dbg !28
  %784 = call float @llvm.maxnum.f32(float %780, float %783), !dbg !28
  %785 = bitcast float %784 to i32, !dbg !28
  %786 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %785), !dbg !28
  %787 = bitcast i32 %786 to float, !dbg !28
  %788 = call float @llvm.maxnum.f32(float %784, float %787), !dbg !28
  %789 = extractvalue [2 x <1 x float>] %205, 1, !dbg !28
  %790 = insertelement <1 x float> poison, float %788, i64 0, !dbg !28
  %791 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %790, <1 x float> %789), !dbg !28
  %792 = shufflevector <1 x float> %772, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !29
  %793 = fsub <4 x float> %718, %792, !dbg !29
  %794 = fsub <4 x float> %728, %792, !dbg !29
  %795 = shufflevector <1 x float> %791, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !29
  %796 = fsub <4 x float> %739, %795, !dbg !29
  %797 = fsub <4 x float> %749, %795, !dbg !29
  %798 = fsub <1 x float> %770, %772, !dbg !30
  %799 = extractelement <1 x float> %798, i64 0, !dbg !31
  %800 = call float @llvm.amdgcn.exp2.f32(float %799), !dbg !31
  %801 = extractvalue [2 x <1 x float>] %204, 0, !dbg !32
  %802 = extractelement <1 x float> %801, i64 0, !dbg !32
  %803 = fmul float %802, %800, !dbg !32
  %804 = fsub <1 x float> %789, %791, !dbg !30
  %805 = extractelement <1 x float> %804, i64 0, !dbg !31
  %806 = call float @llvm.amdgcn.exp2.f32(float %805), !dbg !31
  %807 = extractvalue [2 x <1 x float>] %204, 1, !dbg !32
  %808 = extractelement <1 x float> %807, i64 0, !dbg !32
  %809 = fmul float %808, %806, !dbg !32
  %810 = extractelement <4 x float> %793, i64 0, !dbg !33
  %811 = call float @llvm.amdgcn.exp2.f32(float %810), !dbg !33
  %812 = insertelement <4 x float> poison, float %811, i64 0, !dbg !35
  %813 = extractelement <4 x float> %793, i64 1, !dbg !33
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !33
  %815 = insertelement <4 x float> %812, float %814, i64 1, !dbg !35
  %816 = fadd float %811, %814, !dbg !34
  %817 = extractelement <4 x float> %793, i64 2, !dbg !33
  %818 = call float @llvm.amdgcn.exp2.f32(float %817), !dbg !33
  %819 = insertelement <4 x float> %815, float %818, i64 2, !dbg !35
  %820 = fadd float %816, %818, !dbg !34
  %821 = extractelement <4 x float> %793, i64 3, !dbg !33
  %822 = call float @llvm.amdgcn.exp2.f32(float %821), !dbg !33
  %823 = insertelement <4 x float> %819, float %822, i64 3, !dbg !35
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %823) #9, !dbg !35, !frisk.register_lifetimes !1
  %824 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !35
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %824) #9, !dbg !34, !frisk.register_lifetimes !1
  %825 = fadd float %820, %822, !dbg !34
  %826 = extractelement <4 x float> %794, i64 0, !dbg !33
  %827 = call float @llvm.amdgcn.exp2.f32(float %826), !dbg !33
  %828 = insertelement <4 x float> poison, float %827, i64 0, !dbg !35
  %829 = fadd float %825, %827, !dbg !34
  %830 = extractelement <4 x float> %794, i64 1, !dbg !33
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !33
  %832 = insertelement <4 x float> %828, float %831, i64 1, !dbg !35
  %833 = fadd float %829, %831, !dbg !34
  %834 = extractelement <4 x float> %794, i64 2, !dbg !33
  %835 = call float @llvm.amdgcn.exp2.f32(float %834), !dbg !33
  %836 = insertelement <4 x float> %832, float %835, i64 2, !dbg !35
  %837 = fadd float %833, %835, !dbg !34
  %838 = extractelement <4 x float> %794, i64 3, !dbg !33
  %839 = call float @llvm.amdgcn.exp2.f32(float %838), !dbg !33
  %840 = insertelement <4 x float> %836, float %839, i64 3, !dbg !35
  %math.result59 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %840) #9, !dbg !35, !frisk.register_lifetimes !1
  %841 = fptrunc <4 x float> %math.result59 to <4 x half>, !dbg !35
  %math.packed60 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %841) #9, !dbg !34, !frisk.register_lifetimes !1
  %842 = fadd float %837, %839, !dbg !34
  %843 = extractelement <4 x float> %796, i64 0, !dbg !33
  %844 = call float @llvm.amdgcn.exp2.f32(float %843), !dbg !33
  %845 = insertelement <4 x float> poison, float %844, i64 0, !dbg !35
  %846 = extractelement <4 x float> %796, i64 1, !dbg !33
  %847 = call float @llvm.amdgcn.exp2.f32(float %846), !dbg !33
  %848 = insertelement <4 x float> %845, float %847, i64 1, !dbg !35
  %849 = fadd float %844, %847, !dbg !34
  %850 = extractelement <4 x float> %796, i64 2, !dbg !33
  %851 = call float @llvm.amdgcn.exp2.f32(float %850), !dbg !33
  %852 = insertelement <4 x float> %848, float %851, i64 2, !dbg !35
  %853 = fadd float %849, %851, !dbg !34
  %854 = extractelement <4 x float> %796, i64 3, !dbg !33
  %855 = call float @llvm.amdgcn.exp2.f32(float %854), !dbg !33
  %856 = insertelement <4 x float> %852, float %855, i64 3, !dbg !35
  %math.result61 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %856) #9, !dbg !35, !frisk.register_lifetimes !1
  %857 = fptrunc <4 x float> %math.result61 to <4 x half>, !dbg !35
  %math.packed62 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %857) #9, !dbg !34, !frisk.register_lifetimes !1
  %858 = fadd float %853, %855, !dbg !34
  %859 = extractelement <4 x float> %797, i64 0, !dbg !33
  %860 = call float @llvm.amdgcn.exp2.f32(float %859), !dbg !33
  %861 = insertelement <4 x float> poison, float %860, i64 0, !dbg !35
  %862 = fadd float %858, %860, !dbg !34
  %863 = extractelement <4 x float> %797, i64 1, !dbg !33
  %864 = call float @llvm.amdgcn.exp2.f32(float %863), !dbg !33
  %865 = insertelement <4 x float> %861, float %864, i64 1, !dbg !35
  %866 = fadd float %862, %864, !dbg !34
  %867 = extractelement <4 x float> %797, i64 2, !dbg !33
  %868 = call float @llvm.amdgcn.exp2.f32(float %867), !dbg !33
  %869 = insertelement <4 x float> %865, float %868, i64 2, !dbg !35
  %870 = fadd float %866, %868, !dbg !34
  %871 = extractelement <4 x float> %797, i64 3, !dbg !33
  %872 = call float @llvm.amdgcn.exp2.f32(float %871), !dbg !33
  %873 = insertelement <4 x float> %869, float %872, i64 3, !dbg !35
  %math.result63 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %873) #9, !dbg !35, !frisk.register_lifetimes !1
  %874 = fptrunc <4 x float> %math.result63 to <4 x half>, !dbg !35
  %math.packed64 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %874) #9, !dbg !34, !frisk.register_lifetimes !1
  %875 = fadd float %870, %872, !dbg !34
  %876 = bitcast float %842 to i32, !dbg !34
  %877 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %876), !dbg !34
  %878 = bitcast i32 %877 to float, !dbg !34
  %879 = fadd float %842, %878, !dbg !34
  %880 = bitcast float %879 to i32, !dbg !34
  %881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %880), !dbg !34
  %882 = bitcast i32 %881 to float, !dbg !34
  %883 = fadd float %879, %882, !dbg !34
  %884 = fadd float %883, %803, !dbg !34
  %885 = bitcast float %875 to i32, !dbg !34
  %886 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %885), !dbg !34
  %887 = bitcast i32 %886 to float, !dbg !34
  %888 = fadd float %875, %887, !dbg !34
  %889 = bitcast float %888 to i32, !dbg !34
  %890 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %889), !dbg !34
  %891 = bitcast i32 %890 to float, !dbg !34
  %892 = fadd float %888, %891, !dbg !34
  %893 = fadd float %892, %809, !dbg !34
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread311 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local312 = shl nuw nsw i32 %address.thread311, 3, !dbg !16
  %address.local313 = and i32 %address.local312, 504, !dbg !16
  %address.local314 = zext nneg i32 %address.local313 to i64, !dbg !16
  %address.local315 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local314, !dbg !16
  %894 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local315, i32 0) #0, !dbg !16
  %895 = bitcast <4 x i32> %894 to <8 x half>, !dbg !16
  %address.local316 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local314, !dbg !16
  %address.local317 = getelementptr i8, ptr addrspace(3) %address.local316, i64 1024, !dbg !16
  %896 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local317, i32 0) #0, !dbg !16
  %897 = bitcast <4 x i32> %896 to <8 x half>, !dbg !16
  %898 = add nuw nsw i64 %214, %16, !dbg !19
  %899 = shl i64 %898, 19, !dbg !19
  %900 = and i64 %899, 16252928, !dbg !19
  %901 = add i64 %900, %220, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %902 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %901, !dbg !19
  %903 = getelementptr i8, ptr addrspace(1) %902, i64 64, !dbg !19
  br label %904, !dbg !19

904:                                              ; preds = %907, %711
  %905 = phi i64 [ %933, %907 ], [ %32, %711 ], !dbg !19
  %906 = icmp samesign ult i64 %905, 128, !dbg !19
  br i1 %906, label %907, label %934, !dbg !19

907:                                              ; preds = %904
  %908 = shl nuw nsw i64 %905, 3, !dbg !19
  %909 = and i64 %908, 24, !dbg !19
  %910 = shl nuw nsw i64 %905, 3, !dbg !19
  %911 = and i64 %910, 512, !dbg !19
  %912 = shl nuw nsw i64 %905, 5, !dbg !19
  %913 = and i64 %912, 3968, !dbg !19
  %914 = getelementptr [2 x i8], ptr addrspace(1) %903, i64 %913, !dbg !19
  %915 = getelementptr [2 x i8], ptr addrspace(1) %914, i64 %909, !dbg !19
  %916 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %911, !dbg !19
  %917 = ptrtoint ptr addrspace(3) %916 to i64, !dbg !19
  %918 = trunc i64 %917 to i32, !dbg !19
  %919 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %918), !dbg !19
  %920 = zext i32 %919 to i64, !dbg !19
  %921 = inttoptr i64 %920 to ptr addrspace(3), !dbg !19
  %922 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %923 = ptrtoint ptr addrspace(1) %915 to i64, !dbg !19
  %924 = sub i64 %923, %922, !dbg !19
  %dma.buffer.offset679 = trunc i64 %924 to i32, !dbg !19
  %925 = trunc i64 %922 to i32, !dbg !19
  %926 = insertelement <4 x i32> poison, i32 %925, i64 0, !dbg !19
  %927 = lshr i64 %922, 32, !dbg !19
  %928 = trunc i64 %927 to i32, !dbg !19
  %929 = insertelement <4 x i32> %926, i32 %928, i64 1, !dbg !19
  %930 = insertelement <4 x i32> %929, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc680 = insertelement <4 x i32> %930, i32 131072, i64 3, !dbg !19
  %931 = ptrtoint ptr addrspace(3) %921 to i32, !dbg !19
  %dma.buffer.lds681 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %931), !dbg !19
  %932 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds681, i32 %dma.buffer.offset679, <4 x i32> %dma.buffer.rsrc680) #10, !dbg !19
  %933 = or disjoint i64 %32, 256, !dbg !19
  br label %904, !dbg !19

934:                                              ; preds = %904
  %935 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %936 = extractvalue [2 x <8 x float>] %203, 0, !dbg !36
  %937 = shufflevector <8 x float> %936, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %938 = insertelement <4 x float> poison, float %800, i64 0, !dbg !36
  %939 = shufflevector <4 x float> %938, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %940 = fmul <4 x float> %937, %939, !dbg !36
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %935, <4 x float> %940) #9, !dbg !16
  %942 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %942, <4 x float> %941) #9, !dbg !16
  %944 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %945 = shufflevector <8 x float> %936, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %946 = fmul <4 x float> %945, %939, !dbg !36
  %947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %944, <4 x float> %946) #9, !dbg !16
  %948 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %948, <4 x float> %947) #9, !dbg !16
  %950 = extractvalue [2 x <8 x float>] %203, 1, !dbg !36
  %951 = shufflevector <8 x float> %950, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %952 = insertelement <4 x float> poison, float %806, i64 0, !dbg !36
  %953 = shufflevector <4 x float> %952, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %954 = fmul <4 x float> %951, %953, !dbg !36
  %955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed62, <4 x half> %935, <4 x float> %954) #9, !dbg !16
  %956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %942, <4 x float> %955) #9, !dbg !16
  %957 = shufflevector <8 x float> %950, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %958 = fmul <4 x float> %957, %953, !dbg !36
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed62, <4 x half> %944, <4 x float> %958) #9, !dbg !16
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %948, <4 x float> %959) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread318 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local319 = shl nuw nsw i32 %address.thread318, 3, !dbg !16
  %address.local320 = and i32 %address.local319, 504, !dbg !16
  %address.local321 = zext nneg i32 %address.local320 to i64, !dbg !16
  %address.local322 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local321, !dbg !16
  %961 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local322, i32 0) #0, !dbg !16
  %962 = bitcast <4 x i32> %961 to <8 x half>, !dbg !16
  %address.local323 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local321, !dbg !16
  %address.local324 = getelementptr i8, ptr addrspace(3) %address.local323, i64 1024, !dbg !16
  %963 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local324, i32 0) #0, !dbg !16
  %964 = bitcast <4 x i32> %963 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %965 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %901, !dbg !19
  %966 = getelementptr i8, ptr addrspace(1) %965, i64 128, !dbg !19
  br label %967, !dbg !19

967:                                              ; preds = %970, %934
  %968 = phi i64 [ %996, %970 ], [ %32, %934 ], !dbg !19
  %969 = icmp samesign ult i64 %968, 128, !dbg !19
  br i1 %969, label %970, label %997, !dbg !19

970:                                              ; preds = %967
  %971 = shl nuw nsw i64 %968, 3, !dbg !19
  %972 = and i64 %971, 24, !dbg !19
  %973 = shl nuw nsw i64 %968, 3, !dbg !19
  %974 = and i64 %973, 512, !dbg !19
  %975 = shl nuw nsw i64 %968, 5, !dbg !19
  %976 = and i64 %975, 3968, !dbg !19
  %977 = getelementptr [2 x i8], ptr addrspace(1) %966, i64 %976, !dbg !19
  %978 = getelementptr [2 x i8], ptr addrspace(1) %977, i64 %972, !dbg !19
  %979 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %974, !dbg !19
  %980 = ptrtoint ptr addrspace(3) %979 to i64, !dbg !19
  %981 = trunc i64 %980 to i32, !dbg !19
  %982 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %981), !dbg !19
  %983 = zext i32 %982 to i64, !dbg !19
  %984 = inttoptr i64 %983 to ptr addrspace(3), !dbg !19
  %985 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %986 = ptrtoint ptr addrspace(1) %978 to i64, !dbg !19
  %987 = sub i64 %986, %985, !dbg !19
  %dma.buffer.offset682 = trunc i64 %987 to i32, !dbg !19
  %988 = trunc i64 %985 to i32, !dbg !19
  %989 = insertelement <4 x i32> poison, i32 %988, i64 0, !dbg !19
  %990 = lshr i64 %985, 32, !dbg !19
  %991 = trunc i64 %990 to i32, !dbg !19
  %992 = insertelement <4 x i32> %989, i32 %991, i64 1, !dbg !19
  %993 = insertelement <4 x i32> %992, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc683 = insertelement <4 x i32> %993, i32 131072, i64 3, !dbg !19
  %994 = ptrtoint ptr addrspace(3) %984 to i32, !dbg !19
  %dma.buffer.lds684 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %994), !dbg !19
  %995 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds684, i32 %dma.buffer.offset682, <4 x i32> %dma.buffer.rsrc683) #10, !dbg !19
  %996 = or disjoint i64 %32, 256, !dbg !19
  br label %967, !dbg !19

997:                                              ; preds = %967
  %998 = shufflevector <8 x half> %962, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %999 = extractvalue [2 x <8 x float>] %206, 0, !dbg !36
  %1000 = shufflevector <8 x float> %999, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1001 = fmul <4 x float> %1000, %939, !dbg !36
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %998, <4 x float> %1001) #9, !dbg !16
  %1003 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %1003, <4 x float> %1002) #9, !dbg !16
  %1005 = shufflevector <8 x half> %962, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1006 = shufflevector <8 x float> %999, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1007 = fmul <4 x float> %1006, %939, !dbg !36
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1005, <4 x float> %1007) #9, !dbg !16
  %1009 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %1009, <4 x float> %1008) #9, !dbg !16
  %1011 = extractvalue [2 x <8 x float>] %206, 1, !dbg !36
  %1012 = shufflevector <8 x float> %1011, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1013 = fmul <4 x float> %1012, %953, !dbg !36
  %1014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed62, <4 x half> %998, <4 x float> %1013) #9, !dbg !16
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %1003, <4 x float> %1014) #9, !dbg !16
  %1016 = shufflevector <8 x float> %1011, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1017 = fmul <4 x float> %1016, %953, !dbg !36
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed62, <4 x half> %1005, <4 x float> %1017) #9, !dbg !16
  %1019 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %1009, <4 x float> %1018) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread325 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local326 = shl nuw nsw i32 %address.thread325, 3, !dbg !16
  %address.local327 = and i32 %address.local326, 504, !dbg !16
  %address.local328 = zext nneg i32 %address.local327 to i64, !dbg !16
  %address.local329 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local328, !dbg !16
  %1020 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local329, i32 0) #0, !dbg !16
  %1021 = bitcast <4 x i32> %1020 to <8 x half>, !dbg !16
  %address.local330 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local328, !dbg !16
  %address.local331 = getelementptr i8, ptr addrspace(3) %address.local330, i64 1024, !dbg !16
  %1022 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !16
  %1023 = bitcast <4 x i32> %1022 to <8 x half>, !dbg !16
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1024 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %901, !dbg !19
  %1025 = getelementptr i8, ptr addrspace(1) %1024, i64 192, !dbg !19
  br label %1026, !dbg !19

1026:                                             ; preds = %1029, %997
  %1027 = phi i64 [ %1055, %1029 ], [ %32, %997 ], !dbg !19
  %1028 = icmp samesign ult i64 %1027, 128, !dbg !19
  br i1 %1028, label %1029, label %1056, !dbg !19

1029:                                             ; preds = %1026
  %1030 = shl nuw nsw i64 %1027, 3, !dbg !19
  %1031 = and i64 %1030, 24, !dbg !19
  %1032 = shl nuw nsw i64 %1027, 3, !dbg !19
  %1033 = and i64 %1032, 512, !dbg !19
  %1034 = shl nuw nsw i64 %1027, 5, !dbg !19
  %1035 = and i64 %1034, 3968, !dbg !19
  %1036 = getelementptr [2 x i8], ptr addrspace(1) %1025, i64 %1035, !dbg !19
  %1037 = getelementptr [2 x i8], ptr addrspace(1) %1036, i64 %1031, !dbg !19
  %1038 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %1033, !dbg !19
  %1039 = ptrtoint ptr addrspace(3) %1038 to i64, !dbg !19
  %1040 = trunc i64 %1039 to i32, !dbg !19
  %1041 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1040), !dbg !19
  %1042 = zext i32 %1041 to i64, !dbg !19
  %1043 = inttoptr i64 %1042 to ptr addrspace(3), !dbg !19
  %1044 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1045 = ptrtoint ptr addrspace(1) %1037 to i64, !dbg !19
  %1046 = sub i64 %1045, %1044, !dbg !19
  %dma.buffer.offset685 = trunc i64 %1046 to i32, !dbg !19
  %1047 = trunc i64 %1044 to i32, !dbg !19
  %1048 = insertelement <4 x i32> poison, i32 %1047, i64 0, !dbg !19
  %1049 = lshr i64 %1044, 32, !dbg !19
  %1050 = trunc i64 %1049 to i32, !dbg !19
  %1051 = insertelement <4 x i32> %1048, i32 %1050, i64 1, !dbg !19
  %1052 = insertelement <4 x i32> %1051, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc686 = insertelement <4 x i32> %1052, i32 131072, i64 3, !dbg !19
  %1053 = ptrtoint ptr addrspace(3) %1043 to i32, !dbg !19
  %dma.buffer.lds687 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1053), !dbg !19
  %1054 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds687, i32 %dma.buffer.offset685, <4 x i32> %dma.buffer.rsrc686) #10, !dbg !19
  %1055 = or disjoint i64 %32, 256, !dbg !19
  br label %1026, !dbg !19

1056:                                             ; preds = %1026
  %1057 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1058 = extractvalue [2 x <8 x float>] %207, 0, !dbg !36
  %1059 = shufflevector <8 x float> %1058, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1060 = fmul <4 x float> %1059, %939, !dbg !36
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1057, <4 x float> %1060) #9, !dbg !16
  %1062 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %1062, <4 x float> %1061) #9, !dbg !16
  %1064 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1065 = shufflevector <8 x float> %1058, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1066 = fmul <4 x float> %1065, %939, !dbg !36
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1064, <4 x float> %1066) #9, !dbg !16
  %1068 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %1068, <4 x float> %1067) #9, !dbg !16
  %1070 = extractvalue [2 x <8 x float>] %207, 1, !dbg !36
  %1071 = shufflevector <8 x float> %1070, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1072 = fmul <4 x float> %1071, %953, !dbg !36
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed62, <4 x half> %1057, <4 x float> %1072) #9, !dbg !16
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %1062, <4 x float> %1073) #9, !dbg !16
  %1075 = shufflevector <8 x float> %1070, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1076 = fmul <4 x float> %1075, %953, !dbg !36
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed62, <4 x half> %1064, <4 x float> %1076) #9, !dbg !16
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %1068, <4 x float> %1077) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread332 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !40, !frisk.register_lifetimes !1
  %address.local333 = shl nuw nsw i32 %address.thread332, 3, !dbg !16
  %address.local334 = and i32 %address.local333, 504, !dbg !16
  %address.local335 = zext nneg i32 %address.local334 to i64, !dbg !16
  %address.local336 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !16
  %1079 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local336, i32 0) #0, !dbg !16
  %1080 = bitcast <4 x i32> %1079 to <8 x half>, !dbg !16
  %address.local337 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !16
  %address.local338 = getelementptr i8, ptr addrspace(3) %address.local337, i64 1024, !dbg !16
  %1081 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local338, i32 0) #0, !dbg !16
  %1082 = bitcast <4 x i32> %1081 to <8 x half>, !dbg !16
  %1083 = shufflevector <8 x half> %1080, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1084 = extractvalue [2 x <8 x float>] %208, 0, !dbg !36
  %1085 = shufflevector <8 x float> %1084, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1086 = fmul <4 x float> %1085, %939, !dbg !36
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1083, <4 x float> %1086) #9, !dbg !16
  %1088 = shufflevector <8 x half> %1082, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %1089 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %1088, <4 x float> %1087) #9, !dbg !16
  %1090 = shufflevector <8 x half> %1080, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1091 = shufflevector <8 x float> %1084, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1092 = fmul <4 x float> %1091, %939, !dbg !36
  %1093 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1090, <4 x float> %1092) #9, !dbg !16
  %1094 = shufflevector <8 x half> %1082, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1095 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed60, <4 x half> %1094, <4 x float> %1093) #9, !dbg !16
  %1096 = extractvalue [2 x <8 x float>] %208, 1, !dbg !36
  %1097 = shufflevector <8 x float> %1096, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %1098 = fmul <4 x float> %1097, %953, !dbg !36
  %1099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed62, <4 x half> %1083, <4 x float> %1098) #9, !dbg !16
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %1088, <4 x float> %1099) #9, !dbg !16
  %1101 = shufflevector <8 x float> %1096, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %1102 = fmul <4 x float> %1101, %953, !dbg !36
  %1103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed62, <4 x half> %1090, <4 x float> %1102) #9, !dbg !16
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %1094, <4 x float> %1103) #9, !dbg !16
  %1105 = shufflevector <4 x float> %943, <4 x float> %949, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1106 = insertvalue [2 x <8 x float>] poison, <8 x float> %1105, 0, !dbg !7
  %1107 = shufflevector <4 x float> %956, <4 x float> %960, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1108 = insertvalue [2 x <8 x float>] %1106, <8 x float> %1107, 1, !dbg !7
  %1109 = insertelement <1 x float> poison, float %884, i64 0, !dbg !7
  %1110 = insertvalue [2 x <1 x float>] poison, <1 x float> %1109, 0, !dbg !7
  %1111 = insertelement <1 x float> poison, float %893, i64 0, !dbg !7
  %1112 = insertvalue [2 x <1 x float>] %1110, <1 x float> %1111, 1, !dbg !7
  %1113 = insertvalue [2 x <1 x float>] poison, <1 x float> %772, 0, !dbg !7
  %1114 = insertvalue [2 x <1 x float>] %1113, <1 x float> %791, 1, !dbg !7
  %1115 = shufflevector <4 x float> %1004, <4 x float> %1010, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1116 = insertvalue [2 x <8 x float>] poison, <8 x float> %1115, 0, !dbg !7
  %1117 = shufflevector <4 x float> %1015, <4 x float> %1019, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1118 = insertvalue [2 x <8 x float>] %1116, <8 x float> %1117, 1, !dbg !7
  %1119 = shufflevector <4 x float> %1063, <4 x float> %1069, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1120 = insertvalue [2 x <8 x float>] poison, <8 x float> %1119, 0, !dbg !7
  %1121 = shufflevector <4 x float> %1074, <4 x float> %1078, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1122 = insertvalue [2 x <8 x float>] %1120, <8 x float> %1121, 1, !dbg !7
  %1123 = shufflevector <4 x float> %1089, <4 x float> %1095, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1124 = insertvalue [2 x <8 x float>] poison, <8 x float> %1123, 0, !dbg !7
  %1125 = shufflevector <4 x float> %1100, <4 x float> %1104, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1126 = insertvalue [2 x <8 x float>] %1124, <8 x float> %1125, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1127 = add nuw nsw i64 %202, 1, !dbg !7
  br label %201, !dbg !7

1128:                                             ; preds = %201
  %1129 = extractvalue [2 x <8 x float>] %208, 0, !dbg !7
  %1130 = extractelement <8 x float> %1129, i64 0, !dbg !7
  %1131 = extractelement <8 x float> %1129, i64 1, !dbg !7
  %1132 = extractelement <8 x float> %1129, i64 2, !dbg !7
  %1133 = extractelement <8 x float> %1129, i64 3, !dbg !7
  %1134 = select i1 %.not, float %1131, float %1130, !dbg !7
  %1135 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1136 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1135), !dbg !7
  %1137 = add nuw nsw i32 %1136, 64, !dbg !7
  %1138 = xor i32 %1136, 16, !dbg !7
  %1139 = and i32 %1137, 192, !dbg !7
  %1140 = icmp samesign ult i32 %1138, %1139, !dbg !7
  %1141 = select i1 %1140, i32 %1138, i32 %1136, !dbg !7
  %1142 = shl nuw nsw i32 %1141, 2, !dbg !7
  %1143 = bitcast float %1134 to i32, !dbg !7
  %1144 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1143), !dbg !7
  %1145 = bitcast i32 %1144 to float, !dbg !7
  %1146 = select i1 %.not, float %1130, float %1145, !dbg !7
  %1147 = select i1 %.not, float %1145, float %1131, !dbg !7
  %1148 = select i1 %.not, float %1133, float %1132, !dbg !7
  %1149 = bitcast float %1148 to i32, !dbg !7
  %1150 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1149), !dbg !7
  %1151 = bitcast i32 %1150 to float, !dbg !7
  %1152 = select i1 %.not, float %1132, float %1151, !dbg !7
  %1153 = select i1 %.not, float %1151, float %1133, !dbg !7
  %1154 = select i1 %.not1, float %1152, float %1146, !dbg !7
  %1155 = xor i32 %1136, 32, !dbg !7
  %1156 = icmp samesign ult i32 %1155, %1139, !dbg !7
  %1157 = select i1 %1156, i32 %1155, i32 %1136, !dbg !7
  %1158 = shl nuw nsw i32 %1157, 2, !dbg !7
  %1159 = bitcast float %1154 to i32, !dbg !7
  %1160 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1159), !dbg !7
  %1161 = bitcast i32 %1160 to float, !dbg !7
  %1162 = select i1 %.not1, float %1146, float %1161, !dbg !7
  %1163 = select i1 %.not1, float %1161, float %1152, !dbg !7
  %1164 = select i1 %.not1, float %1153, float %1147, !dbg !7
  %1165 = bitcast float %1164 to i32, !dbg !7
  %1166 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1165), !dbg !7
  %1167 = bitcast i32 %1166 to float, !dbg !7
  %1168 = select i1 %.not1, float %1147, float %1167, !dbg !7
  %1169 = select i1 %.not1, float %1167, float %1153, !dbg !7
  %1170 = extractvalue [2 x <8 x float>] %207, 0, !dbg !7
  %1171 = extractelement <8 x float> %1170, i64 0, !dbg !7
  %1172 = extractelement <8 x float> %1170, i64 1, !dbg !7
  %1173 = extractelement <8 x float> %1170, i64 2, !dbg !7
  %1174 = extractelement <8 x float> %1170, i64 3, !dbg !7
  %1175 = select i1 %.not, float %1172, float %1171, !dbg !7
  %1176 = bitcast float %1175 to i32, !dbg !7
  %1177 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1176), !dbg !7
  %1178 = bitcast i32 %1177 to float, !dbg !7
  %1179 = select i1 %.not, float %1171, float %1178, !dbg !7
  %1180 = select i1 %.not, float %1178, float %1172, !dbg !7
  %1181 = select i1 %.not, float %1174, float %1173, !dbg !7
  %1182 = bitcast float %1181 to i32, !dbg !7
  %1183 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1182), !dbg !7
  %1184 = bitcast i32 %1183 to float, !dbg !7
  %1185 = select i1 %.not, float %1173, float %1184, !dbg !7
  %1186 = select i1 %.not, float %1184, float %1174, !dbg !7
  %1187 = select i1 %.not1, float %1185, float %1179, !dbg !7
  %1188 = bitcast float %1187 to i32, !dbg !7
  %1189 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1188), !dbg !7
  %1190 = bitcast i32 %1189 to float, !dbg !7
  %1191 = select i1 %.not1, float %1179, float %1190, !dbg !7
  %1192 = select i1 %.not1, float %1190, float %1185, !dbg !7
  %1193 = select i1 %.not1, float %1186, float %1180, !dbg !7
  %1194 = bitcast float %1193 to i32, !dbg !7
  %1195 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1194), !dbg !7
  %1196 = bitcast i32 %1195 to float, !dbg !7
  %1197 = select i1 %.not1, float %1180, float %1196, !dbg !7
  %1198 = select i1 %.not1, float %1196, float %1186, !dbg !7
  %1199 = extractvalue [2 x <8 x float>] %206, 0, !dbg !7
  %1200 = extractelement <8 x float> %1199, i64 0, !dbg !7
  %1201 = extractelement <8 x float> %1199, i64 1, !dbg !7
  %1202 = extractelement <8 x float> %1199, i64 2, !dbg !7
  %1203 = extractelement <8 x float> %1199, i64 3, !dbg !7
  %1204 = select i1 %.not, float %1201, float %1200, !dbg !7
  %1205 = bitcast float %1204 to i32, !dbg !7
  %1206 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1205), !dbg !7
  %1207 = bitcast i32 %1206 to float, !dbg !7
  %1208 = select i1 %.not, float %1200, float %1207, !dbg !7
  %1209 = select i1 %.not, float %1207, float %1201, !dbg !7
  %1210 = select i1 %.not, float %1203, float %1202, !dbg !7
  %1211 = bitcast float %1210 to i32, !dbg !7
  %1212 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1211), !dbg !7
  %1213 = bitcast i32 %1212 to float, !dbg !7
  %1214 = select i1 %.not, float %1202, float %1213, !dbg !7
  %1215 = select i1 %.not, float %1213, float %1203, !dbg !7
  %1216 = select i1 %.not1, float %1214, float %1208, !dbg !7
  %1217 = bitcast float %1216 to i32, !dbg !7
  %1218 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1217), !dbg !7
  %1219 = bitcast i32 %1218 to float, !dbg !7
  %1220 = select i1 %.not1, float %1208, float %1219, !dbg !7
  %1221 = select i1 %.not1, float %1219, float %1214, !dbg !7
  %1222 = select i1 %.not1, float %1215, float %1209, !dbg !7
  %1223 = bitcast float %1222 to i32, !dbg !7
  %1224 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1223), !dbg !7
  %1225 = bitcast i32 %1224 to float, !dbg !7
  %1226 = select i1 %.not1, float %1209, float %1225, !dbg !7
  %1227 = select i1 %.not1, float %1225, float %1215, !dbg !7
  %1228 = extractvalue [2 x <8 x float>] %203, 0, !dbg !7
  %1229 = extractelement <8 x float> %1228, i64 0, !dbg !7
  %1230 = extractelement <8 x float> %1228, i64 1, !dbg !7
  %1231 = extractelement <8 x float> %1228, i64 2, !dbg !7
  %1232 = extractelement <8 x float> %1228, i64 3, !dbg !7
  %1233 = select i1 %.not, float %1230, float %1229, !dbg !7
  %1234 = bitcast float %1233 to i32, !dbg !7
  %1235 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1234), !dbg !7
  %1236 = bitcast i32 %1235 to float, !dbg !7
  %1237 = select i1 %.not, float %1229, float %1236, !dbg !7
  %1238 = select i1 %.not, float %1236, float %1230, !dbg !7
  %1239 = select i1 %.not, float %1232, float %1231, !dbg !7
  %1240 = bitcast float %1239 to i32, !dbg !7
  %1241 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1240), !dbg !7
  %1242 = bitcast i32 %1241 to float, !dbg !7
  %1243 = select i1 %.not, float %1231, float %1242, !dbg !7
  %1244 = select i1 %.not, float %1242, float %1232, !dbg !7
  %1245 = select i1 %.not1, float %1243, float %1237, !dbg !7
  %1246 = bitcast float %1245 to i32, !dbg !7
  %1247 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1246), !dbg !7
  %1248 = select i1 %.not1, float %1244, float %1238, !dbg !7
  %1249 = bitcast float %1248 to i32, !dbg !7
  %1250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1249), !dbg !7
  %1251 = extractvalue [2 x <1 x float>] %204, 0, !dbg !20
  %1252 = shufflevector <1 x float> %1251, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1253 = extractelement <8 x float> %1129, i64 4, !dbg !7
  %1254 = extractelement <8 x float> %1129, i64 5, !dbg !7
  %1255 = extractelement <8 x float> %1129, i64 6, !dbg !7
  %1256 = extractelement <8 x float> %1129, i64 7, !dbg !7
  %1257 = select i1 %.not, float %1254, float %1253, !dbg !7
  %1258 = bitcast float %1257 to i32, !dbg !7
  %1259 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1258), !dbg !7
  %1260 = bitcast i32 %1259 to float, !dbg !7
  %1261 = select i1 %.not, float %1253, float %1260, !dbg !7
  %1262 = select i1 %.not, float %1260, float %1254, !dbg !7
  %1263 = select i1 %.not, float %1256, float %1255, !dbg !7
  %1264 = bitcast float %1263 to i32, !dbg !7
  %1265 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1264), !dbg !7
  %1266 = bitcast i32 %1265 to float, !dbg !7
  %1267 = select i1 %.not, float %1255, float %1266, !dbg !7
  %1268 = select i1 %.not, float %1266, float %1256, !dbg !7
  %1269 = select i1 %.not1, float %1267, float %1261, !dbg !7
  %1270 = bitcast float %1269 to i32, !dbg !7
  %1271 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1270), !dbg !7
  %1272 = bitcast i32 %1271 to float, !dbg !7
  %1273 = select i1 %.not1, float %1261, float %1272, !dbg !7
  %1274 = select i1 %.not1, float %1272, float %1267, !dbg !7
  %1275 = select i1 %.not1, float %1268, float %1262, !dbg !7
  %1276 = bitcast float %1275 to i32, !dbg !7
  %1277 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1276), !dbg !7
  %1278 = bitcast i32 %1277 to float, !dbg !7
  %1279 = select i1 %.not1, float %1262, float %1278, !dbg !7
  %1280 = select i1 %.not1, float %1278, float %1268, !dbg !7
  %1281 = extractelement <8 x float> %1170, i64 4, !dbg !7
  %1282 = extractelement <8 x float> %1170, i64 5, !dbg !7
  %1283 = extractelement <8 x float> %1170, i64 6, !dbg !7
  %1284 = extractelement <8 x float> %1170, i64 7, !dbg !7
  %1285 = select i1 %.not, float %1282, float %1281, !dbg !7
  %1286 = bitcast float %1285 to i32, !dbg !7
  %1287 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1286), !dbg !7
  %1288 = bitcast i32 %1287 to float, !dbg !7
  %1289 = select i1 %.not, float %1281, float %1288, !dbg !7
  %1290 = select i1 %.not, float %1288, float %1282, !dbg !7
  %1291 = select i1 %.not, float %1284, float %1283, !dbg !7
  %1292 = bitcast float %1291 to i32, !dbg !7
  %1293 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1292), !dbg !7
  %1294 = bitcast i32 %1293 to float, !dbg !7
  %1295 = select i1 %.not, float %1283, float %1294, !dbg !7
  %1296 = select i1 %.not, float %1294, float %1284, !dbg !7
  %1297 = select i1 %.not1, float %1295, float %1289, !dbg !7
  %1298 = bitcast float %1297 to i32, !dbg !7
  %1299 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1298), !dbg !7
  %1300 = bitcast i32 %1299 to float, !dbg !7
  %1301 = select i1 %.not1, float %1289, float %1300, !dbg !7
  %1302 = select i1 %.not1, float %1300, float %1295, !dbg !7
  %1303 = select i1 %.not1, float %1296, float %1290, !dbg !7
  %1304 = bitcast float %1303 to i32, !dbg !7
  %1305 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1304), !dbg !7
  %1306 = bitcast i32 %1305 to float, !dbg !7
  %1307 = select i1 %.not1, float %1290, float %1306, !dbg !7
  %1308 = select i1 %.not1, float %1306, float %1296, !dbg !7
  %1309 = extractelement <8 x float> %1199, i64 4, !dbg !7
  %1310 = extractelement <8 x float> %1199, i64 5, !dbg !7
  %1311 = extractelement <8 x float> %1199, i64 6, !dbg !7
  %1312 = extractelement <8 x float> %1199, i64 7, !dbg !7
  %1313 = select i1 %.not, float %1310, float %1309, !dbg !7
  %1314 = bitcast float %1313 to i32, !dbg !7
  %1315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1314), !dbg !7
  %1316 = bitcast i32 %1315 to float, !dbg !7
  %1317 = select i1 %.not, float %1309, float %1316, !dbg !7
  %1318 = select i1 %.not, float %1316, float %1310, !dbg !7
  %1319 = select i1 %.not, float %1312, float %1311, !dbg !7
  %1320 = bitcast float %1319 to i32, !dbg !7
  %1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1320), !dbg !7
  %1322 = bitcast i32 %1321 to float, !dbg !7
  %1323 = select i1 %.not, float %1311, float %1322, !dbg !7
  %1324 = select i1 %.not, float %1322, float %1312, !dbg !7
  %1325 = select i1 %.not1, float %1323, float %1317, !dbg !7
  %1326 = bitcast float %1325 to i32, !dbg !7
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1326), !dbg !7
  %1328 = bitcast i32 %1327 to float, !dbg !7
  %1329 = select i1 %.not1, float %1317, float %1328, !dbg !7
  %1330 = select i1 %.not1, float %1328, float %1323, !dbg !7
  %1331 = select i1 %.not1, float %1324, float %1318, !dbg !7
  %1332 = bitcast float %1331 to i32, !dbg !7
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1332), !dbg !7
  %1334 = bitcast i32 %1333 to float, !dbg !7
  %1335 = select i1 %.not1, float %1318, float %1334, !dbg !7
  %1336 = select i1 %.not1, float %1334, float %1324, !dbg !7
  %1337 = extractelement <8 x float> %1228, i64 4, !dbg !7
  %1338 = extractelement <8 x float> %1228, i64 5, !dbg !7
  %1339 = extractelement <8 x float> %1228, i64 6, !dbg !7
  %1340 = extractelement <8 x float> %1228, i64 7, !dbg !7
  %1341 = select i1 %.not, float %1338, float %1337, !dbg !7
  %1342 = bitcast float %1341 to i32, !dbg !7
  %1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1342), !dbg !7
  %1344 = bitcast i32 %1343 to float, !dbg !7
  %1345 = select i1 %.not, float %1337, float %1344, !dbg !7
  %1346 = select i1 %.not, float %1344, float %1338, !dbg !7
  %1347 = select i1 %.not, float %1340, float %1339, !dbg !7
  %1348 = bitcast float %1347 to i32, !dbg !7
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1348), !dbg !7
  %1350 = bitcast i32 %1349 to float, !dbg !7
  %1351 = select i1 %.not, float %1339, float %1350, !dbg !7
  %1352 = select i1 %.not, float %1350, float %1340, !dbg !7
  %1353 = select i1 %.not1, float %1351, float %1345, !dbg !7
  %1354 = bitcast float %1353 to i32, !dbg !7
  %1355 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1354), !dbg !7
  %1356 = select i1 %.not1, float %1352, float %1346, !dbg !7
  %1357 = bitcast float %1356 to i32, !dbg !7
  %1358 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1357), !dbg !7
  %1359 = extractvalue [2 x <8 x float>] %208, 1, !dbg !7
  %1360 = extractelement <8 x float> %1359, i64 0, !dbg !7
  %1361 = extractelement <8 x float> %1359, i64 1, !dbg !7
  %1362 = extractelement <8 x float> %1359, i64 2, !dbg !7
  %1363 = extractelement <8 x float> %1359, i64 3, !dbg !7
  %1364 = select i1 %.not, float %1361, float %1360, !dbg !7
  %1365 = bitcast float %1364 to i32, !dbg !7
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1365), !dbg !7
  %1367 = bitcast i32 %1366 to float, !dbg !7
  %1368 = select i1 %.not, float %1360, float %1367, !dbg !7
  %1369 = select i1 %.not, float %1367, float %1361, !dbg !7
  %1370 = select i1 %.not, float %1363, float %1362, !dbg !7
  %1371 = bitcast float %1370 to i32, !dbg !7
  %1372 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1371), !dbg !7
  %1373 = bitcast i32 %1372 to float, !dbg !7
  %1374 = select i1 %.not, float %1362, float %1373, !dbg !7
  %1375 = select i1 %.not, float %1373, float %1363, !dbg !7
  %1376 = select i1 %.not1, float %1374, float %1368, !dbg !7
  %1377 = bitcast float %1376 to i32, !dbg !7
  %1378 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1377), !dbg !7
  %1379 = bitcast i32 %1378 to float, !dbg !7
  %1380 = select i1 %.not1, float %1368, float %1379, !dbg !7
  %1381 = select i1 %.not1, float %1379, float %1374, !dbg !7
  %1382 = select i1 %.not1, float %1375, float %1369, !dbg !7
  %1383 = bitcast float %1382 to i32, !dbg !7
  %1384 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1383), !dbg !7
  %1385 = bitcast i32 %1384 to float, !dbg !7
  %1386 = select i1 %.not1, float %1369, float %1385, !dbg !7
  %1387 = select i1 %.not1, float %1385, float %1375, !dbg !7
  %1388 = extractvalue [2 x <8 x float>] %207, 1, !dbg !7
  %1389 = extractelement <8 x float> %1388, i64 0, !dbg !7
  %1390 = extractelement <8 x float> %1388, i64 1, !dbg !7
  %1391 = extractelement <8 x float> %1388, i64 2, !dbg !7
  %1392 = extractelement <8 x float> %1388, i64 3, !dbg !7
  %1393 = select i1 %.not, float %1390, float %1389, !dbg !7
  %1394 = bitcast float %1393 to i32, !dbg !7
  %1395 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1394), !dbg !7
  %1396 = bitcast i32 %1395 to float, !dbg !7
  %1397 = select i1 %.not, float %1389, float %1396, !dbg !7
  %1398 = select i1 %.not, float %1396, float %1390, !dbg !7
  %1399 = select i1 %.not, float %1392, float %1391, !dbg !7
  %1400 = bitcast float %1399 to i32, !dbg !7
  %1401 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1400), !dbg !7
  %1402 = bitcast i32 %1401 to float, !dbg !7
  %1403 = select i1 %.not, float %1391, float %1402, !dbg !7
  %1404 = select i1 %.not, float %1402, float %1392, !dbg !7
  %1405 = select i1 %.not1, float %1403, float %1397, !dbg !7
  %1406 = bitcast float %1405 to i32, !dbg !7
  %1407 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1406), !dbg !7
  %1408 = bitcast i32 %1407 to float, !dbg !7
  %1409 = select i1 %.not1, float %1397, float %1408, !dbg !7
  %1410 = select i1 %.not1, float %1408, float %1403, !dbg !7
  %1411 = select i1 %.not1, float %1404, float %1398, !dbg !7
  %1412 = bitcast float %1411 to i32, !dbg !7
  %1413 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1412), !dbg !7
  %1414 = bitcast i32 %1413 to float, !dbg !7
  %1415 = select i1 %.not1, float %1398, float %1414, !dbg !7
  %1416 = select i1 %.not1, float %1414, float %1404, !dbg !7
  %1417 = extractvalue [2 x <8 x float>] %206, 1, !dbg !7
  %1418 = extractelement <8 x float> %1417, i64 0, !dbg !7
  %1419 = extractelement <8 x float> %1417, i64 1, !dbg !7
  %1420 = extractelement <8 x float> %1417, i64 2, !dbg !7
  %1421 = extractelement <8 x float> %1417, i64 3, !dbg !7
  %1422 = select i1 %.not, float %1419, float %1418, !dbg !7
  %1423 = bitcast float %1422 to i32, !dbg !7
  %1424 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1423), !dbg !7
  %1425 = bitcast i32 %1424 to float, !dbg !7
  %1426 = select i1 %.not, float %1418, float %1425, !dbg !7
  %1427 = select i1 %.not, float %1425, float %1419, !dbg !7
  %1428 = select i1 %.not, float %1421, float %1420, !dbg !7
  %1429 = bitcast float %1428 to i32, !dbg !7
  %1430 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1429), !dbg !7
  %1431 = bitcast i32 %1430 to float, !dbg !7
  %1432 = select i1 %.not, float %1420, float %1431, !dbg !7
  %1433 = select i1 %.not, float %1431, float %1421, !dbg !7
  %1434 = select i1 %.not1, float %1432, float %1426, !dbg !7
  %1435 = bitcast float %1434 to i32, !dbg !7
  %1436 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1435), !dbg !7
  %1437 = bitcast i32 %1436 to float, !dbg !7
  %1438 = select i1 %.not1, float %1426, float %1437, !dbg !7
  %1439 = select i1 %.not1, float %1437, float %1432, !dbg !7
  %1440 = select i1 %.not1, float %1433, float %1427, !dbg !7
  %1441 = bitcast float %1440 to i32, !dbg !7
  %1442 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1441), !dbg !7
  %1443 = bitcast i32 %1442 to float, !dbg !7
  %1444 = select i1 %.not1, float %1427, float %1443, !dbg !7
  %1445 = select i1 %.not1, float %1443, float %1433, !dbg !7
  %1446 = extractvalue [2 x <8 x float>] %203, 1, !dbg !7
  %1447 = extractelement <8 x float> %1446, i64 0, !dbg !7
  %1448 = extractelement <8 x float> %1446, i64 1, !dbg !7
  %1449 = extractelement <8 x float> %1446, i64 2, !dbg !7
  %1450 = extractelement <8 x float> %1446, i64 3, !dbg !7
  %1451 = select i1 %.not, float %1448, float %1447, !dbg !7
  %1452 = bitcast float %1451 to i32, !dbg !7
  %1453 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1452), !dbg !7
  %1454 = bitcast i32 %1453 to float, !dbg !7
  %1455 = select i1 %.not, float %1447, float %1454, !dbg !7
  %1456 = select i1 %.not, float %1454, float %1448, !dbg !7
  %1457 = select i1 %.not, float %1450, float %1449, !dbg !7
  %1458 = bitcast float %1457 to i32, !dbg !7
  %1459 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1458), !dbg !7
  %1460 = bitcast i32 %1459 to float, !dbg !7
  %1461 = select i1 %.not, float %1449, float %1460, !dbg !7
  %1462 = select i1 %.not, float %1460, float %1450, !dbg !7
  %1463 = select i1 %.not1, float %1461, float %1455, !dbg !7
  %1464 = bitcast float %1463 to i32, !dbg !7
  %1465 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1464), !dbg !7
  %1466 = select i1 %.not1, float %1462, float %1456, !dbg !7
  %1467 = bitcast float %1466 to i32, !dbg !7
  %1468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1467), !dbg !7
  %1469 = extractvalue [2 x <1 x float>] %204, 1, !dbg !20
  %1470 = shufflevector <1 x float> %1469, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1471 = extractelement <8 x float> %1359, i64 4, !dbg !7
  %1472 = extractelement <8 x float> %1359, i64 5, !dbg !7
  %1473 = extractelement <8 x float> %1359, i64 6, !dbg !7
  %1474 = extractelement <8 x float> %1359, i64 7, !dbg !7
  %1475 = select i1 %.not, float %1472, float %1471, !dbg !7
  %1476 = bitcast float %1475 to i32, !dbg !7
  %1477 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1476), !dbg !7
  %1478 = bitcast i32 %1477 to float, !dbg !7
  %1479 = select i1 %.not, float %1471, float %1478, !dbg !7
  %1480 = select i1 %.not, float %1478, float %1472, !dbg !7
  %1481 = select i1 %.not, float %1474, float %1473, !dbg !7
  %1482 = bitcast float %1481 to i32, !dbg !7
  %1483 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1482), !dbg !7
  %1484 = bitcast i32 %1483 to float, !dbg !7
  %1485 = select i1 %.not, float %1473, float %1484, !dbg !7
  %1486 = select i1 %.not, float %1484, float %1474, !dbg !7
  %1487 = select i1 %.not1, float %1485, float %1479, !dbg !7
  %1488 = bitcast float %1487 to i32, !dbg !7
  %1489 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1488), !dbg !7
  %1490 = bitcast i32 %1489 to float, !dbg !7
  %1491 = select i1 %.not1, float %1479, float %1490, !dbg !7
  %1492 = select i1 %.not1, float %1490, float %1485, !dbg !7
  %1493 = select i1 %.not1, float %1486, float %1480, !dbg !7
  %1494 = bitcast float %1493 to i32, !dbg !7
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1494), !dbg !7
  %1496 = bitcast i32 %1495 to float, !dbg !7
  %1497 = select i1 %.not1, float %1480, float %1496, !dbg !7
  %1498 = select i1 %.not1, float %1496, float %1486, !dbg !7
  %1499 = extractelement <8 x float> %1388, i64 4, !dbg !7
  %1500 = extractelement <8 x float> %1388, i64 5, !dbg !7
  %1501 = extractelement <8 x float> %1388, i64 6, !dbg !7
  %1502 = extractelement <8 x float> %1388, i64 7, !dbg !7
  %1503 = select i1 %.not, float %1500, float %1499, !dbg !7
  %1504 = bitcast float %1503 to i32, !dbg !7
  %1505 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1504), !dbg !7
  %1506 = bitcast i32 %1505 to float, !dbg !7
  %1507 = select i1 %.not, float %1499, float %1506, !dbg !7
  %1508 = select i1 %.not, float %1506, float %1500, !dbg !7
  %1509 = select i1 %.not, float %1502, float %1501, !dbg !7
  %1510 = bitcast float %1509 to i32, !dbg !7
  %1511 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1510), !dbg !7
  %1512 = bitcast i32 %1511 to float, !dbg !7
  %1513 = select i1 %.not, float %1501, float %1512, !dbg !7
  %1514 = select i1 %.not, float %1512, float %1502, !dbg !7
  %1515 = select i1 %.not1, float %1513, float %1507, !dbg !7
  %1516 = bitcast float %1515 to i32, !dbg !7
  %1517 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1516), !dbg !7
  %1518 = bitcast i32 %1517 to float, !dbg !7
  %1519 = select i1 %.not1, float %1507, float %1518, !dbg !7
  %1520 = select i1 %.not1, float %1518, float %1513, !dbg !7
  %1521 = select i1 %.not1, float %1514, float %1508, !dbg !7
  %1522 = bitcast float %1521 to i32, !dbg !7
  %1523 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1522), !dbg !7
  %1524 = bitcast i32 %1523 to float, !dbg !7
  %1525 = select i1 %.not1, float %1508, float %1524, !dbg !7
  %1526 = select i1 %.not1, float %1524, float %1514, !dbg !7
  %1527 = extractelement <8 x float> %1417, i64 4, !dbg !7
  %1528 = extractelement <8 x float> %1417, i64 5, !dbg !7
  %1529 = extractelement <8 x float> %1417, i64 6, !dbg !7
  %1530 = extractelement <8 x float> %1417, i64 7, !dbg !7
  %1531 = select i1 %.not, float %1528, float %1527, !dbg !7
  %1532 = bitcast float %1531 to i32, !dbg !7
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1532), !dbg !7
  %1534 = bitcast i32 %1533 to float, !dbg !7
  %1535 = select i1 %.not, float %1527, float %1534, !dbg !7
  %1536 = select i1 %.not, float %1534, float %1528, !dbg !7
  %1537 = select i1 %.not, float %1530, float %1529, !dbg !7
  %1538 = bitcast float %1537 to i32, !dbg !7
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1538), !dbg !7
  %1540 = bitcast i32 %1539 to float, !dbg !7
  %1541 = select i1 %.not, float %1529, float %1540, !dbg !7
  %1542 = select i1 %.not, float %1540, float %1530, !dbg !7
  %1543 = select i1 %.not1, float %1541, float %1535, !dbg !7
  %1544 = bitcast float %1543 to i32, !dbg !7
  %1545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1544), !dbg !7
  %1546 = bitcast i32 %1545 to float, !dbg !7
  %1547 = select i1 %.not1, float %1535, float %1546, !dbg !7
  %1548 = select i1 %.not1, float %1546, float %1541, !dbg !7
  %1549 = select i1 %.not1, float %1542, float %1536, !dbg !7
  %1550 = bitcast float %1549 to i32, !dbg !7
  %1551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1550), !dbg !7
  %1552 = bitcast i32 %1551 to float, !dbg !7
  %1553 = select i1 %.not1, float %1536, float %1552, !dbg !7
  %1554 = select i1 %.not1, float %1552, float %1542, !dbg !7
  %1555 = extractelement <8 x float> %1446, i64 4, !dbg !7
  %1556 = extractelement <8 x float> %1446, i64 5, !dbg !7
  %1557 = extractelement <8 x float> %1446, i64 6, !dbg !7
  %1558 = extractelement <8 x float> %1446, i64 7, !dbg !7
  %1559 = select i1 %.not, float %1556, float %1555, !dbg !7
  %1560 = bitcast float %1559 to i32, !dbg !7
  %1561 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1560), !dbg !7
  %1562 = bitcast i32 %1561 to float, !dbg !7
  %1563 = select i1 %.not, float %1555, float %1562, !dbg !7
  %1564 = select i1 %.not, float %1562, float %1556, !dbg !7
  %1565 = select i1 %.not, float %1558, float %1557, !dbg !7
  %1566 = bitcast float %1565 to i32, !dbg !7
  %1567 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1566), !dbg !7
  %1568 = bitcast i32 %1567 to float, !dbg !7
  %1569 = select i1 %.not, float %1557, float %1568, !dbg !7
  %1570 = select i1 %.not, float %1568, float %1558, !dbg !7
  %1571 = select i1 %.not1, float %1569, float %1563, !dbg !7
  %1572 = bitcast float %1571 to i32, !dbg !7
  %1573 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1572), !dbg !7
  %1574 = select i1 %.not1, float %1570, float %1564, !dbg !7
  %1575 = bitcast float %1574 to i32, !dbg !7
  %1576 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1575), !dbg !7
  br i1 true, label %1577, label %1641, !dbg !22

1577:                                             ; preds = %1128
  %1578 = bitcast i32 %1573 to float, !dbg !7
  %1579 = select i1 %.not1, float %1563, float %1578, !dbg !7
  %1580 = insertelement <4 x float> poison, float %1579, i64 0, !dbg !20
  %1581 = bitcast i32 %1576 to float, !dbg !7
  %1582 = select i1 %.not1, float %1564, float %1581, !dbg !7
  %1583 = insertelement <4 x float> %1580, float %1582, i64 1, !dbg !20
  %1584 = select i1 %.not1, float %1578, float %1569, !dbg !7
  %1585 = insertelement <4 x float> %1583, float %1584, i64 2, !dbg !20
  %1586 = select i1 %.not1, float %1581, float %1570, !dbg !7
  %1587 = insertelement <4 x float> %1585, float %1586, i64 3, !dbg !20
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1470, !dbg !20
  %normalized = fmul <4 x float> %1587, %row.reciprocal, !dbg !20
  %1588 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !21
  %1589 = extractelement <4 x half> %1588, i64 3, !dbg !21
  %1590 = extractelement <4 x half> %1588, i64 2, !dbg !21
  %1591 = extractelement <4 x half> %1588, i64 1, !dbg !21
  %1592 = extractelement <4 x half> %1588, i64 0, !dbg !21
  %1593 = bitcast i32 %1465 to float, !dbg !7
  %1594 = select i1 %.not1, float %1455, float %1593, !dbg !7
  %1595 = insertelement <4 x float> poison, float %1594, i64 0, !dbg !20
  %1596 = bitcast i32 %1468 to float, !dbg !7
  %1597 = select i1 %.not1, float %1456, float %1596, !dbg !7
  %1598 = insertelement <4 x float> %1595, float %1597, i64 1, !dbg !20
  %1599 = select i1 %.not1, float %1593, float %1461, !dbg !7
  %1600 = insertelement <4 x float> %1598, float %1599, i64 2, !dbg !20
  %1601 = select i1 %.not1, float %1596, float %1462, !dbg !7
  %1602 = insertelement <4 x float> %1600, float %1601, i64 3, !dbg !20
  %normalized23 = fmul <4 x float> %1602, %row.reciprocal, !dbg !20
  %1603 = fptrunc <4 x float> %normalized23 to <4 x half>, !dbg !21
  %1604 = extractelement <4 x half> %1603, i64 3, !dbg !21
  %1605 = extractelement <4 x half> %1603, i64 2, !dbg !21
  %1606 = extractelement <4 x half> %1603, i64 1, !dbg !21
  %1607 = extractelement <4 x half> %1603, i64 0, !dbg !21
  %1608 = bitcast i32 %1355 to float, !dbg !7
  %1609 = select i1 %.not1, float %1345, float %1608, !dbg !7
  %1610 = insertelement <4 x float> poison, float %1609, i64 0, !dbg !20
  %1611 = bitcast i32 %1358 to float, !dbg !7
  %1612 = select i1 %.not1, float %1346, float %1611, !dbg !7
  %1613 = insertelement <4 x float> %1610, float %1612, i64 1, !dbg !20
  %1614 = select i1 %.not1, float %1608, float %1351, !dbg !7
  %1615 = insertelement <4 x float> %1613, float %1614, i64 2, !dbg !20
  %1616 = select i1 %.not1, float %1611, float %1352, !dbg !7
  %1617 = insertelement <4 x float> %1615, float %1616, i64 3, !dbg !20
  %row.reciprocal24 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1252, !dbg !20
  %normalized25 = fmul <4 x float> %1617, %row.reciprocal24, !dbg !20
  %1618 = fptrunc <4 x float> %normalized25 to <4 x half>, !dbg !21
  %1619 = extractelement <4 x half> %1618, i64 3, !dbg !21
  %1620 = extractelement <4 x half> %1618, i64 2, !dbg !21
  %1621 = extractelement <4 x half> %1618, i64 1, !dbg !21
  %1622 = extractelement <4 x half> %1618, i64 0, !dbg !21
  %1623 = bitcast i32 %1247 to float, !dbg !7
  %1624 = select i1 %.not1, float %1237, float %1623, !dbg !7
  %1625 = insertelement <4 x float> poison, float %1624, i64 0, !dbg !20
  %1626 = bitcast i32 %1250 to float, !dbg !7
  %1627 = select i1 %.not1, float %1238, float %1626, !dbg !7
  %1628 = insertelement <4 x float> %1625, float %1627, i64 1, !dbg !20
  %1629 = select i1 %.not1, float %1623, float %1243, !dbg !7
  %1630 = insertelement <4 x float> %1628, float %1629, i64 2, !dbg !20
  %1631 = select i1 %.not1, float %1626, float %1244, !dbg !7
  %1632 = insertelement <4 x float> %1630, float %1631, i64 3, !dbg !20
  %normalized26 = fmul <4 x float> %1632, %row.reciprocal24, !dbg !20
  %1633 = fptrunc <4 x float> %normalized26 to <4 x half>, !dbg !21
  %1634 = extractelement <4 x half> %1633, i64 3, !dbg !21
  %1635 = extractelement <4 x half> %1633, i64 2, !dbg !21
  %1636 = extractelement <4 x half> %1633, i64 1, !dbg !21
  %1637 = extractelement <4 x half> %1633, i64 0, !dbg !21
  %1638 = add nsw i64 %16, %22, !dbg !22
  %1639 = shl i64 %1638, 19, !dbg !22
  %1640 = and i64 %1639, 16252928, !dbg !22
  %address.thread339 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !22, !range !40, !frisk.register_lifetimes !1
  %address.local340 = zext nneg i32 %address.thread339 to i64, !dbg !9
  %address.local341 = add i64 %25, %address.local340, !dbg !22
  %address.local342 = lshr i64 %address.local340, 6, !dbg !8
  %address.local343 = shl nuw nsw i64 %address.local342, 4, !dbg !22
  %address.local344 = add i64 %address.local341, %address.local343, !dbg !22
  %address.local345 = and i64 %address.local340, 240, !dbg !22
  %address.local346 = sub i64 %address.local344, %address.local345, !dbg !22
  %address.local347 = add i64 %address.local346, %26, !dbg !22
  %address.local348 = shl nuw nsw i64 %address.local347, 7, !dbg !22
  %address.local349 = add nuw nsw i64 %1640, %address.local348, !dbg !22
  %address.local350 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local349, !dbg !22
  %address.local351 = lshr i64 %address.local340, 2, !dbg !8
  %address.local352 = and i64 %address.local351, 12, !dbg !8
  %address.local353 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local350, i64 %address.local352, !dbg !22
  store half %1637, ptr addrspace(1) %address.local353, align 2, !dbg !22
  %address.local354 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local349, !dbg !22
  %address.local355 = or disjoint i64 %address.local352, 1, !dbg !22
  %address.local356 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local354, i64 %address.local355, !dbg !22
  store half %1636, ptr addrspace(1) %address.local356, align 2, !dbg !22
  %address.local357 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local349, !dbg !22
  %address.local358 = or disjoint i64 %address.local352, 2, !dbg !22
  %address.local359 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local357, i64 %address.local358, !dbg !22
  store half %1635, ptr addrspace(1) %address.local359, align 2, !dbg !22
  %address.local360 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local349, !dbg !22
  %address.local361 = or disjoint i64 %address.local352, 3, !dbg !22
  %address.local362 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local360, i64 %address.local361, !dbg !22
  store half %1634, ptr addrspace(1) %address.local362, align 2, !dbg !22
  %address.local363 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local349, !dbg !22
  %address.local364 = or disjoint i64 %address.local352, 16, !dbg !22
  %address.local365 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local363, i64 %address.local364, !dbg !22
  store half %1622, ptr addrspace(1) %address.local365, align 2, !dbg !22
  %address.local366 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local349, !dbg !22
  %address.local367 = or disjoint i64 %address.local352, 17, !dbg !22
  %address.local368 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local366, i64 %address.local367, !dbg !22
  store half %1621, ptr addrspace(1) %address.local368, align 2, !dbg !22
  %address.local369 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local349, !dbg !22
  %address.local370 = or disjoint i64 %address.local352, 18, !dbg !22
  %address.local371 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local369, i64 %address.local370, !dbg !22
  store half %1620, ptr addrspace(1) %address.local371, align 2, !dbg !22
  %address.local372 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local349, !dbg !22
  %address.local373 = or disjoint i64 %address.local352, 19, !dbg !22
  %address.local374 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local372, i64 %address.local373, !dbg !22
  store half %1619, ptr addrspace(1) %address.local374, align 2, !dbg !22
  %address.local375 = shl i64 %address.local347, 7, !dbg !22
  %address.local376 = add i64 %address.local375, 8192, !dbg !22
  %address.local377 = add nuw nsw i64 %1640, %address.local376, !dbg !22
  %address.local378 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local377, !dbg !22
  %address.local379 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local378, i64 %address.local352, !dbg !22
  store half %1607, ptr addrspace(1) %address.local379, align 2, !dbg !22
  %address.local380 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local377, !dbg !22
  %address.local381 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local380, i64 %address.local355, !dbg !22
  store half %1606, ptr addrspace(1) %address.local381, align 2, !dbg !22
  %address.local382 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local377, !dbg !22
  %address.local383 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local382, i64 %address.local358, !dbg !22
  store half %1605, ptr addrspace(1) %address.local383, align 2, !dbg !22
  %address.local384 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local377, !dbg !22
  %address.local385 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local384, i64 %address.local361, !dbg !22
  store half %1604, ptr addrspace(1) %address.local385, align 2, !dbg !22
  %address.local386 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local377, !dbg !22
  %address.local387 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local386, i64 %address.local364, !dbg !22
  store half %1592, ptr addrspace(1) %address.local387, align 2, !dbg !22
  %address.local388 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local377, !dbg !22
  %address.local389 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local388, i64 %address.local367, !dbg !22
  store half %1591, ptr addrspace(1) %address.local389, align 2, !dbg !22
  %address.local390 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local377, !dbg !22
  %address.local391 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local390, i64 %address.local370, !dbg !22
  store half %1590, ptr addrspace(1) %address.local391, align 2, !dbg !22
  %address.local392 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local377, !dbg !22
  %address.local393 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local392, i64 %address.local373, !dbg !22
  store half %1589, ptr addrspace(1) %address.local393, align 2, !dbg !22
  br label %1641, !dbg !22

1641:                                             ; preds = %1577, %1128
  br i1 true, label %1642, label %1680, !dbg !22

1642:                                             ; preds = %1641
  %1643 = insertelement <4 x float> poison, float %1547, i64 0, !dbg !20
  %1644 = insertelement <4 x float> %1643, float %1553, i64 1, !dbg !20
  %1645 = insertelement <4 x float> %1644, float %1548, i64 2, !dbg !20
  %1646 = insertelement <4 x float> %1645, float %1554, i64 3, !dbg !20
  %row.reciprocal27 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1470, !dbg !20
  %normalized28 = fmul <4 x float> %1646, %row.reciprocal27, !dbg !20
  %1647 = fptrunc <4 x float> %normalized28 to <4 x half>, !dbg !21
  %1648 = extractelement <4 x half> %1647, i64 3, !dbg !21
  %1649 = extractelement <4 x half> %1647, i64 2, !dbg !21
  %1650 = extractelement <4 x half> %1647, i64 1, !dbg !21
  %1651 = extractelement <4 x half> %1647, i64 0, !dbg !21
  %1652 = insertelement <4 x float> poison, float %1438, i64 0, !dbg !20
  %1653 = insertelement <4 x float> %1652, float %1444, i64 1, !dbg !20
  %1654 = insertelement <4 x float> %1653, float %1439, i64 2, !dbg !20
  %1655 = insertelement <4 x float> %1654, float %1445, i64 3, !dbg !20
  %normalized29 = fmul <4 x float> %1655, %row.reciprocal27, !dbg !20
  %1656 = fptrunc <4 x float> %normalized29 to <4 x half>, !dbg !21
  %1657 = extractelement <4 x half> %1656, i64 3, !dbg !21
  %1658 = extractelement <4 x half> %1656, i64 2, !dbg !21
  %1659 = extractelement <4 x half> %1656, i64 1, !dbg !21
  %1660 = extractelement <4 x half> %1656, i64 0, !dbg !21
  %1661 = insertelement <4 x float> poison, float %1329, i64 0, !dbg !20
  %1662 = insertelement <4 x float> %1661, float %1335, i64 1, !dbg !20
  %1663 = insertelement <4 x float> %1662, float %1330, i64 2, !dbg !20
  %1664 = insertelement <4 x float> %1663, float %1336, i64 3, !dbg !20
  %row.reciprocal30 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1252, !dbg !20
  %normalized31 = fmul <4 x float> %1664, %row.reciprocal30, !dbg !20
  %1665 = fptrunc <4 x float> %normalized31 to <4 x half>, !dbg !21
  %1666 = extractelement <4 x half> %1665, i64 3, !dbg !21
  %1667 = extractelement <4 x half> %1665, i64 2, !dbg !21
  %1668 = extractelement <4 x half> %1665, i64 1, !dbg !21
  %1669 = extractelement <4 x half> %1665, i64 0, !dbg !21
  %1670 = insertelement <4 x float> poison, float %1220, i64 0, !dbg !20
  %1671 = insertelement <4 x float> %1670, float %1226, i64 1, !dbg !20
  %1672 = insertelement <4 x float> %1671, float %1221, i64 2, !dbg !20
  %1673 = insertelement <4 x float> %1672, float %1227, i64 3, !dbg !20
  %normalized32 = fmul <4 x float> %1673, %row.reciprocal30, !dbg !20
  %1674 = fptrunc <4 x float> %normalized32 to <4 x half>, !dbg !21
  %1675 = extractelement <4 x half> %1674, i64 3, !dbg !21
  %1676 = extractelement <4 x half> %1674, i64 2, !dbg !21
  %1677 = extractelement <4 x half> %1674, i64 1, !dbg !21
  %1678 = extractelement <4 x half> %1674, i64 0, !dbg !21
  %1679 = shl nuw nsw i64 %24, 19, !dbg !22
  %address.thread394 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !22, !range !40, !frisk.register_lifetimes !1
  %address.local395 = zext nneg i32 %address.thread394 to i64, !dbg !9
  %address.local396 = add i64 %25, %address.local395, !dbg !22
  %address.local397 = lshr i64 %address.local395, 6, !dbg !8
  %address.local398 = shl nuw nsw i64 %address.local397, 4, !dbg !22
  %address.local399 = add i64 %address.local396, %address.local398, !dbg !22
  %address.local400 = and i64 %address.local395, 240, !dbg !22
  %address.local401 = sub i64 %address.local399, %address.local400, !dbg !22
  %address.local402 = add i64 %address.local401, %26, !dbg !22
  %address.local403 = shl nuw nsw i64 %address.local402, 7, !dbg !22
  %address.local404 = add nuw nsw i64 %1679, %address.local403, !dbg !22
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !22
  %address.local406 = lshr i64 %address.local395, 2, !dbg !8
  %address.local407 = and i64 %address.local406, 12, !dbg !8
  %address.local408 = or disjoint i64 %address.local407, 32, !dbg !22
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local408, !dbg !22
  store half %1678, ptr addrspace(1) %address.local409, align 2, !dbg !22
  %address.local410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !22
  %address.local411 = or disjoint i64 %address.local407, 33, !dbg !22
  %address.local412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local410, i64 %address.local411, !dbg !22
  store half %1677, ptr addrspace(1) %address.local412, align 2, !dbg !22
  %address.local413 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !22
  %address.local414 = or disjoint i64 %address.local407, 34, !dbg !22
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local413, i64 %address.local414, !dbg !22
  store half %1676, ptr addrspace(1) %address.local415, align 2, !dbg !22
  %address.local416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !22
  %address.local417 = or disjoint i64 %address.local407, 35, !dbg !22
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local416, i64 %address.local417, !dbg !22
  store half %1675, ptr addrspace(1) %address.local418, align 2, !dbg !22
  %address.local419 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !22
  %address.local420 = or disjoint i64 %address.local407, 48, !dbg !22
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local419, i64 %address.local420, !dbg !22
  store half %1669, ptr addrspace(1) %address.local421, align 2, !dbg !22
  %address.local422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !22
  %address.local423 = or disjoint i64 %address.local407, 49, !dbg !22
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local422, i64 %address.local423, !dbg !22
  store half %1668, ptr addrspace(1) %address.local424, align 2, !dbg !22
  %address.local425 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !22
  %address.local426 = or disjoint i64 %address.local407, 50, !dbg !22
  %address.local427 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local425, i64 %address.local426, !dbg !22
  store half %1667, ptr addrspace(1) %address.local427, align 2, !dbg !22
  %address.local428 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !22
  %address.local429 = or i64 %address.local406, 51, !dbg !22
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local429, !dbg !22
  store half %1666, ptr addrspace(1) %address.local430, align 2, !dbg !22
  %address.local431 = shl i64 %address.local402, 7, !dbg !22
  %address.local432 = add i64 %address.local431, 8192, !dbg !22
  %address.local433 = add nuw nsw i64 %1679, %address.local432, !dbg !22
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local433, !dbg !22
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local434, i64 %address.local408, !dbg !22
  store half %1660, ptr addrspace(1) %address.local435, align 2, !dbg !22
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local433, !dbg !22
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local436, i64 %address.local411, !dbg !22
  store half %1659, ptr addrspace(1) %address.local437, align 2, !dbg !22
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local433, !dbg !22
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local438, i64 %address.local414, !dbg !22
  store half %1658, ptr addrspace(1) %address.local439, align 2, !dbg !22
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local433, !dbg !22
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local440, i64 %address.local417, !dbg !22
  store half %1657, ptr addrspace(1) %address.local441, align 2, !dbg !22
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local433, !dbg !22
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local442, i64 %address.local420, !dbg !22
  store half %1651, ptr addrspace(1) %address.local443, align 2, !dbg !22
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local433, !dbg !22
  %address.local445 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local444, i64 %address.local423, !dbg !22
  store half %1650, ptr addrspace(1) %address.local445, align 2, !dbg !22
  %address.local446 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local433, !dbg !22
  %address.local447 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local446, i64 %address.local426, !dbg !22
  store half %1649, ptr addrspace(1) %address.local447, align 2, !dbg !22
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local433, !dbg !22
  %address.local449 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local448, i64 %address.local429, !dbg !22
  store half %1648, ptr addrspace(1) %address.local449, align 2, !dbg !22
  br label %1680, !dbg !22

1680:                                             ; preds = %1642, %1641
  br i1 true, label %1681, label %1719, !dbg !22

1681:                                             ; preds = %1680
  %1682 = insertelement <4 x float> poison, float %1519, i64 0, !dbg !20
  %1683 = insertelement <4 x float> %1682, float %1525, i64 1, !dbg !20
  %1684 = insertelement <4 x float> %1683, float %1520, i64 2, !dbg !20
  %1685 = insertelement <4 x float> %1684, float %1526, i64 3, !dbg !20
  %row.reciprocal33 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1470, !dbg !20
  %normalized34 = fmul <4 x float> %1685, %row.reciprocal33, !dbg !20
  %1686 = fptrunc <4 x float> %normalized34 to <4 x half>, !dbg !21
  %1687 = extractelement <4 x half> %1686, i64 3, !dbg !21
  %1688 = extractelement <4 x half> %1686, i64 2, !dbg !21
  %1689 = extractelement <4 x half> %1686, i64 1, !dbg !21
  %1690 = extractelement <4 x half> %1686, i64 0, !dbg !21
  %1691 = insertelement <4 x float> poison, float %1409, i64 0, !dbg !20
  %1692 = insertelement <4 x float> %1691, float %1415, i64 1, !dbg !20
  %1693 = insertelement <4 x float> %1692, float %1410, i64 2, !dbg !20
  %1694 = insertelement <4 x float> %1693, float %1416, i64 3, !dbg !20
  %normalized35 = fmul <4 x float> %1694, %row.reciprocal33, !dbg !20
  %1695 = fptrunc <4 x float> %normalized35 to <4 x half>, !dbg !21
  %1696 = extractelement <4 x half> %1695, i64 3, !dbg !21
  %1697 = extractelement <4 x half> %1695, i64 2, !dbg !21
  %1698 = extractelement <4 x half> %1695, i64 1, !dbg !21
  %1699 = extractelement <4 x half> %1695, i64 0, !dbg !21
  %1700 = insertelement <4 x float> poison, float %1301, i64 0, !dbg !20
  %1701 = insertelement <4 x float> %1700, float %1307, i64 1, !dbg !20
  %1702 = insertelement <4 x float> %1701, float %1302, i64 2, !dbg !20
  %1703 = insertelement <4 x float> %1702, float %1308, i64 3, !dbg !20
  %row.reciprocal36 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1252, !dbg !20
  %normalized37 = fmul <4 x float> %1703, %row.reciprocal36, !dbg !20
  %1704 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !21
  %1705 = extractelement <4 x half> %1704, i64 3, !dbg !21
  %1706 = extractelement <4 x half> %1704, i64 2, !dbg !21
  %1707 = extractelement <4 x half> %1704, i64 1, !dbg !21
  %1708 = extractelement <4 x half> %1704, i64 0, !dbg !21
  %1709 = insertelement <4 x float> poison, float %1191, i64 0, !dbg !20
  %1710 = insertelement <4 x float> %1709, float %1197, i64 1, !dbg !20
  %1711 = insertelement <4 x float> %1710, float %1192, i64 2, !dbg !20
  %1712 = insertelement <4 x float> %1711, float %1198, i64 3, !dbg !20
  %normalized38 = fmul <4 x float> %1712, %row.reciprocal36, !dbg !20
  %1713 = fptrunc <4 x float> %normalized38 to <4 x half>, !dbg !21
  %1714 = extractelement <4 x half> %1713, i64 3, !dbg !21
  %1715 = extractelement <4 x half> %1713, i64 2, !dbg !21
  %1716 = extractelement <4 x half> %1713, i64 1, !dbg !21
  %1717 = extractelement <4 x half> %1713, i64 0, !dbg !21
  %1718 = shl nuw nsw i64 %24, 19, !dbg !22
  %address.thread450 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !22, !range !40, !frisk.register_lifetimes !1
  %address.local451 = zext nneg i32 %address.thread450 to i64, !dbg !9
  %address.local452 = add i64 %25, %address.local451, !dbg !22
  %address.local453 = lshr i64 %address.local451, 6, !dbg !8
  %address.local454 = shl nuw nsw i64 %address.local453, 4, !dbg !22
  %address.local455 = add i64 %address.local452, %address.local454, !dbg !22
  %address.local456 = and i64 %address.local451, 240, !dbg !22
  %address.local457 = sub i64 %address.local455, %address.local456, !dbg !22
  %address.local458 = add i64 %address.local457, %26, !dbg !22
  %address.local459 = shl nuw nsw i64 %address.local458, 7, !dbg !22
  %address.local460 = add nuw nsw i64 %1718, %address.local459, !dbg !22
  %address.local461 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local460, !dbg !22
  %address.local462 = lshr i64 %address.local451, 2, !dbg !8
  %address.local463 = and i64 %address.local462, 12, !dbg !8
  %address.local464 = or disjoint i64 %address.local463, 64, !dbg !22
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local461, i64 %address.local464, !dbg !22
  store half %1717, ptr addrspace(1) %address.local465, align 2, !dbg !22
  %address.local466 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local460, !dbg !22
  %address.local467 = or disjoint i64 %address.local463, 65, !dbg !22
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local466, i64 %address.local467, !dbg !22
  store half %1716, ptr addrspace(1) %address.local468, align 2, !dbg !22
  %address.local469 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local460, !dbg !22
  %address.local470 = or disjoint i64 %address.local463, 66, !dbg !22
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local469, i64 %address.local470, !dbg !22
  store half %1715, ptr addrspace(1) %address.local471, align 2, !dbg !22
  %address.local472 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local460, !dbg !22
  %address.local473 = or disjoint i64 %address.local463, 67, !dbg !22
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local472, i64 %address.local473, !dbg !22
  store half %1714, ptr addrspace(1) %address.local474, align 2, !dbg !22
  %address.local475 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local460, !dbg !22
  %address.local476 = or disjoint i64 %address.local463, 80, !dbg !22
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local475, i64 %address.local476, !dbg !22
  store half %1708, ptr addrspace(1) %address.local477, align 2, !dbg !22
  %address.local478 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local460, !dbg !22
  %address.local479 = or disjoint i64 %address.local463, 81, !dbg !22
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local478, i64 %address.local479, !dbg !22
  store half %1707, ptr addrspace(1) %address.local480, align 2, !dbg !22
  %address.local481 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local460, !dbg !22
  %address.local482 = or disjoint i64 %address.local463, 82, !dbg !22
  %address.local483 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local481, i64 %address.local482, !dbg !22
  store half %1706, ptr addrspace(1) %address.local483, align 2, !dbg !22
  %address.local484 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local460, !dbg !22
  %address.local485 = or disjoint i64 %address.local463, 83, !dbg !22
  %address.local486 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local484, i64 %address.local485, !dbg !22
  store half %1705, ptr addrspace(1) %address.local486, align 2, !dbg !22
  %address.local487 = shl i64 %address.local458, 7, !dbg !22
  %address.local488 = add i64 %address.local487, 8192, !dbg !22
  %address.local489 = add nuw nsw i64 %1718, %address.local488, !dbg !22
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local489, !dbg !22
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local490, i64 %address.local464, !dbg !22
  store half %1699, ptr addrspace(1) %address.local491, align 2, !dbg !22
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local489, !dbg !22
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local492, i64 %address.local467, !dbg !22
  store half %1698, ptr addrspace(1) %address.local493, align 2, !dbg !22
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local489, !dbg !22
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local494, i64 %address.local470, !dbg !22
  store half %1697, ptr addrspace(1) %address.local495, align 2, !dbg !22
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local489, !dbg !22
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local496, i64 %address.local473, !dbg !22
  store half %1696, ptr addrspace(1) %address.local497, align 2, !dbg !22
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local489, !dbg !22
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local498, i64 %address.local476, !dbg !22
  store half %1690, ptr addrspace(1) %address.local499, align 2, !dbg !22
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local489, !dbg !22
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local500, i64 %address.local479, !dbg !22
  store half %1689, ptr addrspace(1) %address.local501, align 2, !dbg !22
  %address.local502 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local489, !dbg !22
  %address.local503 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local502, i64 %address.local482, !dbg !22
  store half %1688, ptr addrspace(1) %address.local503, align 2, !dbg !22
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local489, !dbg !22
  %address.local505 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local504, i64 %address.local485, !dbg !22
  store half %1687, ptr addrspace(1) %address.local505, align 2, !dbg !22
  br label %1719, !dbg !22

1719:                                             ; preds = %1681, %1680
  br i1 true, label %1720, label %1758, !dbg !22

1720:                                             ; preds = %1719
  %1721 = insertelement <4 x float> poison, float %1491, i64 0, !dbg !20
  %1722 = insertelement <4 x float> %1721, float %1497, i64 1, !dbg !20
  %1723 = insertelement <4 x float> %1722, float %1492, i64 2, !dbg !20
  %1724 = insertelement <4 x float> %1723, float %1498, i64 3, !dbg !20
  %row.reciprocal39 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1470, !dbg !20
  %normalized40 = fmul <4 x float> %1724, %row.reciprocal39, !dbg !20
  %1725 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !21
  %1726 = extractelement <4 x half> %1725, i64 3, !dbg !21
  %1727 = extractelement <4 x half> %1725, i64 2, !dbg !21
  %1728 = extractelement <4 x half> %1725, i64 1, !dbg !21
  %1729 = extractelement <4 x half> %1725, i64 0, !dbg !21
  %1730 = insertelement <4 x float> poison, float %1380, i64 0, !dbg !20
  %1731 = insertelement <4 x float> %1730, float %1386, i64 1, !dbg !20
  %1732 = insertelement <4 x float> %1731, float %1381, i64 2, !dbg !20
  %1733 = insertelement <4 x float> %1732, float %1387, i64 3, !dbg !20
  %normalized41 = fmul <4 x float> %1733, %row.reciprocal39, !dbg !20
  %1734 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !21
  %1735 = extractelement <4 x half> %1734, i64 3, !dbg !21
  %1736 = extractelement <4 x half> %1734, i64 2, !dbg !21
  %1737 = extractelement <4 x half> %1734, i64 1, !dbg !21
  %1738 = extractelement <4 x half> %1734, i64 0, !dbg !21
  %1739 = insertelement <4 x float> poison, float %1273, i64 0, !dbg !20
  %1740 = insertelement <4 x float> %1739, float %1279, i64 1, !dbg !20
  %1741 = insertelement <4 x float> %1740, float %1274, i64 2, !dbg !20
  %1742 = insertelement <4 x float> %1741, float %1280, i64 3, !dbg !20
  %row.reciprocal42 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1252, !dbg !20
  %normalized43 = fmul <4 x float> %1742, %row.reciprocal42, !dbg !20
  %1743 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !21
  %1744 = extractelement <4 x half> %1743, i64 3, !dbg !21
  %1745 = extractelement <4 x half> %1743, i64 2, !dbg !21
  %1746 = extractelement <4 x half> %1743, i64 1, !dbg !21
  %1747 = extractelement <4 x half> %1743, i64 0, !dbg !21
  %1748 = insertelement <4 x float> poison, float %1162, i64 0, !dbg !20
  %1749 = insertelement <4 x float> %1748, float %1168, i64 1, !dbg !20
  %1750 = insertelement <4 x float> %1749, float %1163, i64 2, !dbg !20
  %1751 = insertelement <4 x float> %1750, float %1169, i64 3, !dbg !20
  %normalized44 = fmul <4 x float> %1751, %row.reciprocal42, !dbg !20
  %1752 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !21
  %1753 = extractelement <4 x half> %1752, i64 3, !dbg !21
  %1754 = extractelement <4 x half> %1752, i64 2, !dbg !21
  %1755 = extractelement <4 x half> %1752, i64 1, !dbg !21
  %1756 = extractelement <4 x half> %1752, i64 0, !dbg !21
  %1757 = shl nuw nsw i64 %24, 19, !dbg !22
  %address.thread506 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !22, !range !40, !frisk.register_lifetimes !1
  %address.local507 = zext nneg i32 %address.thread506 to i64, !dbg !9
  %address.local508 = add i64 %25, %address.local507, !dbg !22
  %address.local509 = lshr i64 %address.local507, 6, !dbg !8
  %address.local510 = shl nuw nsw i64 %address.local509, 4, !dbg !22
  %address.local511 = add i64 %address.local508, %address.local510, !dbg !22
  %address.local512 = and i64 %address.local507, 240, !dbg !22
  %address.local513 = sub i64 %address.local511, %address.local512, !dbg !22
  %address.local514 = add i64 %address.local513, %26, !dbg !22
  %address.local515 = shl nuw nsw i64 %address.local514, 7, !dbg !22
  %address.local516 = add nuw nsw i64 %1757, %address.local515, !dbg !22
  %address.local517 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local516, !dbg !22
  %address.local518 = lshr i64 %address.local507, 2, !dbg !8
  %address.local519 = and i64 %address.local518, 12, !dbg !8
  %address.local520 = or disjoint i64 %address.local519, 96, !dbg !22
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local517, i64 %address.local520, !dbg !22
  store half %1756, ptr addrspace(1) %address.local521, align 2, !dbg !22
  %address.local522 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local516, !dbg !22
  %address.local523 = or disjoint i64 %address.local519, 97, !dbg !22
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local522, i64 %address.local523, !dbg !22
  store half %1755, ptr addrspace(1) %address.local524, align 2, !dbg !22
  %address.local525 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local516, !dbg !22
  %address.local526 = or disjoint i64 %address.local519, 98, !dbg !22
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local525, i64 %address.local526, !dbg !22
  store half %1754, ptr addrspace(1) %address.local527, align 2, !dbg !22
  %address.local528 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local516, !dbg !22
  %address.local529 = or disjoint i64 %address.local519, 99, !dbg !22
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local528, i64 %address.local529, !dbg !22
  store half %1753, ptr addrspace(1) %address.local530, align 2, !dbg !22
  %address.local531 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local516, !dbg !22
  %address.local532 = or disjoint i64 %address.local519, 112, !dbg !22
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local531, i64 %address.local532, !dbg !22
  store half %1747, ptr addrspace(1) %address.local533, align 2, !dbg !22
  %address.local534 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local516, !dbg !22
  %address.local535 = or disjoint i64 %address.local519, 113, !dbg !22
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local534, i64 %address.local535, !dbg !22
  store half %1746, ptr addrspace(1) %address.local536, align 2, !dbg !22
  %address.local537 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local516, !dbg !22
  %address.local538 = or disjoint i64 %address.local519, 114, !dbg !22
  %address.local539 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local537, i64 %address.local538, !dbg !22
  store half %1745, ptr addrspace(1) %address.local539, align 2, !dbg !22
  %address.local540 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local516, !dbg !22
  %address.local541 = or i64 %address.local518, 115, !dbg !22
  %address.local542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local540, i64 %address.local541, !dbg !22
  store half %1744, ptr addrspace(1) %address.local542, align 2, !dbg !22
  %address.local543 = shl i64 %address.local514, 7, !dbg !22
  %address.local544 = add i64 %address.local543, 8192, !dbg !22
  %address.local545 = add nuw nsw i64 %1757, %address.local544, !dbg !22
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local545, !dbg !22
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local546, i64 %address.local520, !dbg !22
  store half %1738, ptr addrspace(1) %address.local547, align 2, !dbg !22
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local545, !dbg !22
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local548, i64 %address.local523, !dbg !22
  store half %1737, ptr addrspace(1) %address.local549, align 2, !dbg !22
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local545, !dbg !22
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local550, i64 %address.local526, !dbg !22
  store half %1736, ptr addrspace(1) %address.local551, align 2, !dbg !22
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local545, !dbg !22
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local552, i64 %address.local529, !dbg !22
  store half %1735, ptr addrspace(1) %address.local553, align 2, !dbg !22
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local545, !dbg !22
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local554, i64 %address.local532, !dbg !22
  store half %1729, ptr addrspace(1) %address.local555, align 2, !dbg !22
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local545, !dbg !22
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local556, i64 %address.local535, !dbg !22
  store half %1728, ptr addrspace(1) %address.local557, align 2, !dbg !22
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local545, !dbg !22
  %address.local559 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local558, i64 %address.local538, !dbg !22
  store half %1727, ptr addrspace(1) %address.local559, align 2, !dbg !22
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local545, !dbg !22
  %address.local561 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local560, i64 %address.local541, !dbg !22
  store half %1726, ptr addrspace(1) %address.local561, align 2, !dbg !22
  br label %1758, !dbg !22

1758:                                             ; preds = %1720, %1719
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

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!39}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/retained32-baseline")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 72, column: 24, scope: !4)
!9 = !DILocation(line: 70, column: 22, scope: !4)
!11 = !DILocation(line: 48, column: 17, scope: !4)
!14 = !DILocation(line: 63, column: 17, scope: !4)
!15 = !DILocation(line: 77, column: 19, scope: !4)
!16 = !DILocation(line: 102, column: 23, scope: !4)
!17 = !DILocation(line: 71, column: 22, scope: !4)
!18 = !DILocation(line: 73, column: 26, scope: !4)
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
