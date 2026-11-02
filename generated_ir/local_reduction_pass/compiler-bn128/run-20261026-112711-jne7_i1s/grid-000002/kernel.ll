; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/local_reduction_pass/compiler-bn128/run-20261026-112711-jne7_i1s/grid-000002/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/local_reduction_pass/compiler-bn128/run-20261026-112711-jne7_i1s/grid-000002/input.mlir"

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
  %12 = and i64 %11, 576460752303423484, !dbg !5
  %13 = and i64 %6, 3, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = sub nsw i64 31, %14, !dbg !5
  %16 = lshr i64 %10, 2, !dbg !5
  %17 = icmp ugt i64 %10, 1023, !dbg !8
  %18 = add nsw i64 %14, -32, !dbg !8
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !8
  %20 = sdiv i64 %19, 32, !dbg !8
  %21 = sext i1 %17 to i64, !dbg !8
  %22 = xor i64 %20, %21, !dbg !8
  %23 = add nsw i64 %16, %22, !dbg !8
  %24 = and i64 %23, 31, !dbg !8
  %25 = shl nsw i64 %15, 7, !dbg !8
  %26 = mul nsw i64 %22, -4096, !dbg !8
  %27 = add i64 %25, %26, !dbg !8
  %28 = shl nuw nsw i64 %24, 19, !dbg !8
  %29 = shl nsw i64 %27, 7, !dbg !8
  %30 = add i64 %28, %29, !dbg !8
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !39
  %32 = zext nneg i32 %31 to i64, !dbg !8
  %33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !8, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local372 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local373 = and i64 %address.local372, 1536, !dbg !8
  %address.local374 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local373, !dbg !8
  %address.local375 = ptrtoint ptr addrspace(3) %address.local374 to i64, !dbg !8
  %address.local376 = trunc i64 %address.local375 to i32, !dbg !8
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local376), !dbg !8
  %35 = zext i32 %34 to i64, !dbg !8
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !8
  %address.local377 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local378 = and i64 %address.local377, 16256, !dbg !8
  %address.local379 = lshr i64 %address.local, 4, !dbg !8
  %address.local380 = and i64 %address.local379, 8, !dbg !8
  %address.local381 = or disjoint i64 %address.local378, %address.local380, !dbg !8
  %address.local382 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local381, !dbg !8
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %38 = ptrtoint ptr addrspace(1) %address.local382 to i64, !dbg !8
  %39 = sub i64 %38, %37, !dbg !8
  %dma.buffer.offset1645 = trunc i64 %39 to i32, !dbg !8
  %40 = trunc i64 %37 to i32, !dbg !8
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !8
  %42 = lshr i64 %37, 32, !dbg !8
  %43 = trunc i64 %42 to i32, !dbg !8
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !8
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1646 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !8
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !8
  %dma.buffer.lds1647 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1647, i32 %dma.buffer.offset1645, <4 x i32> %dma.buffer.rsrc1646) #10, !dbg !8
  %address.local383 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local384 = shl nuw nsw i64 %address.local383, 3, !dbg !8
  %address.local385 = and i64 %address.local384, 3584, !dbg !8
  %address.local386 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local385, !dbg !8
  %address.local387 = ptrtoint ptr addrspace(3) %address.local386 to i64, !dbg !8
  %address.local388 = trunc i64 %address.local387 to i32, !dbg !8
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !8
  %49 = zext i32 %48 to i64, !dbg !8
  %50 = inttoptr i64 %49 to ptr addrspace(3), !dbg !8
  %address.local389 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local390 = and i64 %address.local389, 16256, !dbg !8
  %address.local391 = lshr i64 %address.local383, 4, !dbg !8
  %address.local392 = and i64 %address.local391, 24, !dbg !8
  %address.local393 = or disjoint i64 %address.local390, %address.local392, !dbg !8
  %address.local394 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local393, !dbg !8
  %51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %52 = ptrtoint ptr addrspace(1) %address.local394 to i64, !dbg !8
  %53 = sub i64 %52, %51, !dbg !8
  %dma.buffer.offset1642 = trunc i64 %53 to i32, !dbg !8
  %54 = trunc i64 %51 to i32, !dbg !8
  %55 = insertelement <4 x i32> poison, i32 %54, i64 0, !dbg !8
  %56 = lshr i64 %51, 32, !dbg !8
  %57 = trunc i64 %56 to i32, !dbg !8
  %58 = insertelement <4 x i32> %55, i32 %57, i64 1, !dbg !8
  %59 = insertelement <4 x i32> %58, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1643 = insertelement <4 x i32> %59, i32 131072, i64 3, !dbg !8
  %60 = ptrtoint ptr addrspace(3) %50 to i32, !dbg !8
  %dma.buffer.lds1644 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !8
  %61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1644, i32 %dma.buffer.offset1642, <4 x i32> %dma.buffer.rsrc1643) #10, !dbg !8
  %62 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %63 = sext i32 %62 to i64, !dbg !5
  %64 = shl i64 %15, 7, !dbg !10
  %65 = lshr i64 %32, 6, !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %66 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %67 = getelementptr i8, ptr addrspace(1) %66, i64 64, !dbg !8
  %address.local395 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local373, !dbg !8
  %address.local396 = ptrtoint ptr addrspace(3) %address.local395 to i64, !dbg !8
  %address.local397 = trunc i64 %address.local396 to i32, !dbg !8
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local397), !dbg !8
  %69 = zext i32 %68 to i64, !dbg !8
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !8
  %address.local398 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local381, !dbg !8
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %72 = ptrtoint ptr addrspace(1) %address.local398 to i64, !dbg !8
  %73 = sub i64 %72, %71, !dbg !8
  %dma.buffer.offset1639 = trunc i64 %73 to i32, !dbg !8
  %74 = trunc i64 %71 to i32, !dbg !8
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !8
  %76 = lshr i64 %71, 32, !dbg !8
  %77 = trunc i64 %76 to i32, !dbg !8
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !8
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1640 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !8
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !8
  %dma.buffer.lds1641 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !8
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1641, i32 %dma.buffer.offset1639, <4 x i32> %dma.buffer.rsrc1640) #10, !dbg !8
  %address.local399 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local385, !dbg !8
  %address.local400 = ptrtoint ptr addrspace(3) %address.local399 to i64, !dbg !8
  %address.local401 = trunc i64 %address.local400 to i32, !dbg !8
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local401), !dbg !8
  %83 = zext i32 %82 to i64, !dbg !8
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !8
  %address.local402 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local393, !dbg !8
  %85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %86 = ptrtoint ptr addrspace(1) %address.local402 to i64, !dbg !8
  %87 = sub i64 %86, %85, !dbg !8
  %dma.buffer.offset1636 = trunc i64 %87 to i32, !dbg !8
  %88 = trunc i64 %85 to i32, !dbg !8
  %89 = insertelement <4 x i32> poison, i32 %88, i64 0, !dbg !8
  %90 = lshr i64 %85, 32, !dbg !8
  %91 = trunc i64 %90 to i32, !dbg !8
  %92 = insertelement <4 x i32> %89, i32 %91, i64 1, !dbg !8
  %93 = insertelement <4 x i32> %92, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1637 = insertelement <4 x i32> %93, i32 131072, i64 3, !dbg !8
  %94 = ptrtoint ptr addrspace(3) %84 to i32, !dbg !8
  %dma.buffer.lds1638 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !8
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1638, i32 %dma.buffer.offset1636, <4 x i32> %dma.buffer.rsrc1637) #10, !dbg !8
  %address.local403 = lshr i32 %address.thread, 2, !dbg !11
  %address.local404 = and i32 %address.local403, 4, !dbg !11
  %address.local405 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local406 = and i32 %address.local405, 384, !dbg !11
  %address.local407 = or disjoint i32 %address.local404, %address.local406, !dbg !11
  %address.local408 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local409 = and i32 %address.local408, 120, !dbg !11
  %address.local410 = or disjoint i32 %address.local407, %address.local409, !dbg !11
  %address.local411 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local412 = and i32 %address.local411, 1024, !dbg !11
  %address.local413 = or disjoint i32 %address.local410, %address.local412, !dbg !11
  %address.local414 = zext nneg i32 %address.local413 to i64, !dbg !11
  %address.local415 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local414, !dbg !11
  %96 = load <4 x half>, ptr addrspace(3) %address.local415, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local415, i64 4096, !dbg !11
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %98 = fmul <4 x half> %96, splat (half 0xH3015), !dbg !11
  %lds.const.gep1446 = getelementptr i8, ptr addrspace(3) %address.local415, i64 1024, !dbg !11
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1446, align 8, !dbg !11
  %100 = fmul <4 x half> %97, splat (half 0xH3015), !dbg !11
  %lds.const.gep1447 = getelementptr i8, ptr addrspace(3) %address.local415, i64 5120, !dbg !11
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1447, align 8, !dbg !11
  %102 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !11
  %103 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %105 = getelementptr i8, ptr addrspace(1) %104, i64 128, !dbg !8
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local376), !dbg !8
  %107 = zext i32 %106 to i64, !dbg !8
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !8
  %address.local458 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local381, !dbg !8
  %109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %110 = ptrtoint ptr addrspace(1) %address.local458 to i64, !dbg !8
  %111 = sub i64 %110, %109, !dbg !8
  %dma.buffer.offset1633 = trunc i64 %111 to i32, !dbg !8
  %112 = trunc i64 %109 to i32, !dbg !8
  %113 = insertelement <4 x i32> poison, i32 %112, i64 0, !dbg !8
  %114 = lshr i64 %109, 32, !dbg !8
  %115 = trunc i64 %114 to i32, !dbg !8
  %116 = insertelement <4 x i32> %113, i32 %115, i64 1, !dbg !8
  %117 = insertelement <4 x i32> %116, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1634 = insertelement <4 x i32> %117, i32 131072, i64 3, !dbg !8
  %118 = ptrtoint ptr addrspace(3) %108 to i32, !dbg !8
  %dma.buffer.lds1635 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !8
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1635, i32 %dma.buffer.offset1633, <4 x i32> %dma.buffer.rsrc1634) #10, !dbg !8
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local388), !dbg !8
  %121 = zext i32 %120 to i64, !dbg !8
  %122 = inttoptr i64 %121 to ptr addrspace(3), !dbg !8
  %address.local459 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local393, !dbg !8
  %123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %124 = ptrtoint ptr addrspace(1) %address.local459 to i64, !dbg !8
  %125 = sub i64 %124, %123, !dbg !8
  %dma.buffer.offset1630 = trunc i64 %125 to i32, !dbg !8
  %126 = trunc i64 %123 to i32, !dbg !8
  %127 = insertelement <4 x i32> poison, i32 %126, i64 0, !dbg !8
  %128 = lshr i64 %123, 32, !dbg !8
  %129 = trunc i64 %128 to i32, !dbg !8
  %130 = insertelement <4 x i32> %127, i32 %129, i64 1, !dbg !8
  %131 = insertelement <4 x i32> %130, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1631 = insertelement <4 x i32> %131, i32 131072, i64 3, !dbg !8
  %132 = ptrtoint ptr addrspace(3) %122 to i32, !dbg !8
  %dma.buffer.lds1632 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !8
  %133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1632, i32 %dma.buffer.offset1630, <4 x i32> %dma.buffer.rsrc1631) #10, !dbg !8
  %lds.const.gep1448 = getelementptr i8, ptr addrspace(3) %address.local415, i64 8192, !dbg !11
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1448, align 8, !dbg !11
  %lds.const.gep1449 = getelementptr i8, ptr addrspace(3) %address.local415, i64 12288, !dbg !11
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1449, align 8, !dbg !11
  %lds.const.gep1450 = getelementptr i8, ptr addrspace(3) %address.local415, i64 9216, !dbg !11
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1450, align 8, !dbg !11
  %137 = fmul <4 x half> %134, splat (half 0xH3015), !dbg !11
  %lds.const.gep1451 = getelementptr i8, ptr addrspace(3) %address.local415, i64 13312, !dbg !11
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep1451, align 8, !dbg !11
  %139 = fmul <4 x half> %135, splat (half 0xH3015), !dbg !11
  %140 = fmul <4 x half> %136, splat (half 0xH3015), !dbg !11
  %141 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %143 = getelementptr i8, ptr addrspace(1) %142, i64 192, !dbg !8
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local397), !dbg !8
  %145 = zext i32 %144 to i64, !dbg !8
  %146 = inttoptr i64 %145 to ptr addrspace(3), !dbg !8
  %address.local464 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local381, !dbg !8
  %147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %148 = ptrtoint ptr addrspace(1) %address.local464 to i64, !dbg !8
  %149 = sub i64 %148, %147, !dbg !8
  %dma.buffer.offset1627 = trunc i64 %149 to i32, !dbg !8
  %150 = trunc i64 %147 to i32, !dbg !8
  %151 = insertelement <4 x i32> poison, i32 %150, i64 0, !dbg !8
  %152 = lshr i64 %147, 32, !dbg !8
  %153 = trunc i64 %152 to i32, !dbg !8
  %154 = insertelement <4 x i32> %151, i32 %153, i64 1, !dbg !8
  %155 = insertelement <4 x i32> %154, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1628 = insertelement <4 x i32> %155, i32 131072, i64 3, !dbg !8
  %156 = ptrtoint ptr addrspace(3) %146 to i32, !dbg !8
  %dma.buffer.lds1629 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %156), !dbg !8
  %157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1629, i32 %dma.buffer.offset1627, <4 x i32> %dma.buffer.rsrc1628) #10, !dbg !8
  %158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local401), !dbg !8
  %159 = zext i32 %158 to i64, !dbg !8
  %160 = inttoptr i64 %159 to ptr addrspace(3), !dbg !8
  %address.local465 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local393, !dbg !8
  %161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %162 = ptrtoint ptr addrspace(1) %address.local465 to i64, !dbg !8
  %163 = sub i64 %162, %161, !dbg !8
  %dma.buffer.offset1624 = trunc i64 %163 to i32, !dbg !8
  %164 = trunc i64 %161 to i32, !dbg !8
  %165 = insertelement <4 x i32> poison, i32 %164, i64 0, !dbg !8
  %166 = lshr i64 %161, 32, !dbg !8
  %167 = trunc i64 %166 to i32, !dbg !8
  %168 = insertelement <4 x i32> %165, i32 %167, i64 1, !dbg !8
  %169 = insertelement <4 x i32> %168, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1625 = insertelement <4 x i32> %169, i32 131072, i64 3, !dbg !8
  %170 = ptrtoint ptr addrspace(3) %160 to i32, !dbg !8
  %dma.buffer.lds1626 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !8
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1626, i32 %dma.buffer.offset1624, <4 x i32> %dma.buffer.rsrc1625) #10, !dbg !8
  %172 = load <4 x half>, ptr addrspace(3) %address.local415, align 8, !dbg !11
  %lds.const.gep1452 = getelementptr i8, ptr addrspace(3) %address.local415, i64 4096, !dbg !11
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1452, align 8, !dbg !11
  %lds.const.gep1453 = getelementptr i8, ptr addrspace(3) %address.local415, i64 1024, !dbg !11
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1453, align 8, !dbg !11
  %175 = fmul <4 x half> %172, splat (half 0xH3015), !dbg !11
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local415, i64 5120, !dbg !11
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !11
  %177 = fmul <4 x half> %173, splat (half 0xH3015), !dbg !11
  %178 = fmul <4 x half> %174, splat (half 0xH3015), !dbg !11
  %179 = fmul <4 x half> %176, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local415, i64 8192, !dbg !11
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !11
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %address.local415, i64 12288, !dbg !11
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !11
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %address.local415, i64 9216, !dbg !11
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !11
  %183 = fmul <4 x half> %180, splat (half 0xH3015), !dbg !11
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local415, i64 13312, !dbg !11
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !11
  %185 = fmul <4 x half> %181, splat (half 0xH3015), !dbg !11
  %186 = fmul <4 x half> %182, splat (half 0xH3015), !dbg !11
  %187 = fmul <4 x half> %184, splat (half 0xH3015), !dbg !11
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %188 = or disjoint i64 %64, 127, !dbg !13
  %189 = add i64 %64, 128, !dbg !13
  %190 = icmp uge i64 %189, %188, !dbg !13
  %191 = sub nsw i64 32, %14, !dbg !15
  %192 = shl i64 %16, 7, !dbg !16
  %193 = and i64 %192, 3968, !dbg !16
  %194 = shl nsw i64 %63, 12, !dbg !16
  %195 = or disjoint i64 %193, %194, !dbg !16
  %196 = lshr i64 %32, 2, !dbg !17
  %197 = shl nuw nsw i64 %32, 2, !dbg !17
  %198 = mul nsw i64 %196, -16, !dbg !17
  %199 = add nsw i64 %197, %198, !dbg !17
  %200 = and i32 %31, 1, !dbg !17
  %index.compact312 = zext nneg i32 %200 to i64, !dbg !17
  %201 = or disjoint i64 %199, %index.compact312, !dbg !17
  %202 = add nuw nsw i64 %201, 1, !dbg !17
  %203 = sdiv i64 %202, 2, !dbg !17
  br label %204, !dbg !7

204:                                              ; preds = %845, %4
  %205 = phi i64 [ %1742, %845 ], [ 0, %4 ], !dbg !7
  %206 = phi [2 x <8 x float>] [ %1727, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %207 = phi [2 x <1 x float>] [ %1333, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %208 = phi [2 x <1 x float>] [ %1729, %845 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %209 = phi [2 x <8 x float>] [ %1733, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %210 = phi [2 x <8 x float>] [ %1737, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %211 = phi [2 x <8 x float>] [ %1741, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %212 = icmp slt i64 %205, %191, !dbg !7
  br i1 %212, label %213, label %1743, !dbg !7

213:                                              ; preds = %204
  %214 = shl nsw i64 %205, 7, !dbg !7
  %215 = lshr i64 %205, 12, !dbg !16
  %216 = add nuw nsw i64 %215, %16, !dbg !16
  %217 = lshr i64 %205, 5, !dbg !16
  %218 = and i64 %217, 127, !dbg !16
  %219 = shl i64 %216, 19, !dbg !16
  %220 = and i64 %219, 16252928, !dbg !16
  %221 = shl i64 %217, 19, !dbg !16
  %222 = shl i64 %205, 14, !dbg !16
  %223 = sub i64 %222, %221, !dbg !16
  %224 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %220, !dbg !16
  %225 = getelementptr [2 x i8], ptr addrspace(1) %224, i64 %218, !dbg !16
  %226 = getelementptr [2 x i8], ptr addrspace(1) %225, i64 %223, !dbg !16
  %address.thread466 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local467 = zext nneg i32 %address.thread466 to i64, !dbg !8
  %address.local468 = shl nuw nsw i64 %address.local467, 3, !dbg !8
  %address.local469 = and i64 %address.local468, 1536, !dbg !8
  %address.local470 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local469, !dbg !8
  %address.local471 = ptrtoint ptr addrspace(3) %address.local470 to i64, !dbg !8
  %address.local472 = trunc i64 %address.local471 to i32, !dbg !8
  %227 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local472), !dbg !16
  %228 = zext i32 %227 to i64, !dbg !16
  %229 = inttoptr i64 %228 to ptr addrspace(3), !dbg !16
  %address.local473 = lshr i64 %address.local467, 4, !dbg !16
  %address.local474 = and i64 %address.local473, 8, !dbg !16
  %address.local475 = and i64 %address.local467, 123, !dbg !16
  %address.local476 = lshr i64 %address.local467, 2, !dbg !16
  %address.local477 = and i64 %address.local476, 4, !dbg !16
  %address.local478 = add nuw nsw i64 %address.local477, %address.local467, !dbg !16
  %address.local479 = and i64 %address.local478, 4, !dbg !16
  %address.local480 = or disjoint i64 %address.local475, %address.local479, !dbg !16
  %address.local481 = shl nuw nsw i64 %address.local480, 7, !dbg !16
  %address.local482 = or disjoint i64 %address.local474, %address.local481, !dbg !16
  %address.local483 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %address.local482, !dbg !16
  %230 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %231 = ptrtoint ptr addrspace(1) %address.local483 to i64, !dbg !16
  %232 = sub i64 %231, %230, !dbg !16
  %dma.buffer.offset1621 = trunc i64 %232 to i32, !dbg !16
  %233 = trunc i64 %230 to i32, !dbg !16
  %234 = insertelement <4 x i32> poison, i32 %233, i64 0, !dbg !16
  %235 = lshr i64 %230, 32, !dbg !16
  %236 = trunc i64 %235 to i32, !dbg !16
  %237 = insertelement <4 x i32> %234, i32 %236, i64 1, !dbg !16
  %238 = insertelement <4 x i32> %237, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1622 = insertelement <4 x i32> %238, i32 131072, i64 3, !dbg !16
  %239 = ptrtoint ptr addrspace(3) %229 to i32, !dbg !16
  %dma.buffer.lds1623 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %239), !dbg !16
  %240 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1623, i32 %dma.buffer.offset1621, <4 x i32> %dma.buffer.rsrc1622) #10, !dbg !16
  %address.local484 = or disjoint i64 %address.local467, 256, !dbg !8
  %address.local485 = shl nuw nsw i64 %address.local484, 3, !dbg !8
  %address.local486 = and i64 %address.local485, 3584, !dbg !8
  %address.local487 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local486, !dbg !8
  %address.local488 = ptrtoint ptr addrspace(3) %address.local487 to i64, !dbg !8
  %address.local489 = trunc i64 %address.local488 to i32, !dbg !8
  %241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local489), !dbg !16
  %242 = zext i32 %241 to i64, !dbg !16
  %243 = inttoptr i64 %242 to ptr addrspace(3), !dbg !16
  %address.local490 = lshr i64 %address.local484, 4, !dbg !16
  %address.local491 = and i64 %address.local490, 24, !dbg !16
  %address.local492 = and i64 %address.local467, 123, !dbg !16
  %address.local493 = lshr i64 %address.local467, 2, !dbg !16
  %address.local494 = and i64 %address.local493, 4, !dbg !16
  %address.local495 = add nuw nsw i64 %address.local494, %address.local467, !dbg !16
  %address.local496 = and i64 %address.local495, 4, !dbg !16
  %address.local497 = or disjoint i64 %address.local492, %address.local496, !dbg !16
  %address.local498 = shl nuw nsw i64 %address.local497, 7, !dbg !16
  %address.local499 = or disjoint i64 %address.local491, %address.local498, !dbg !16
  %address.local500 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %address.local499, !dbg !16
  %244 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %245 = ptrtoint ptr addrspace(1) %address.local500 to i64, !dbg !16
  %246 = sub i64 %245, %244, !dbg !16
  %dma.buffer.offset1618 = trunc i64 %246 to i32, !dbg !16
  %247 = trunc i64 %244 to i32, !dbg !16
  %248 = insertelement <4 x i32> poison, i32 %247, i64 0, !dbg !16
  %249 = lshr i64 %244, 32, !dbg !16
  %250 = trunc i64 %249 to i32, !dbg !16
  %251 = insertelement <4 x i32> %248, i32 %250, i64 1, !dbg !16
  %252 = insertelement <4 x i32> %251, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1619 = insertelement <4 x i32> %252, i32 131072, i64 3, !dbg !16
  %253 = ptrtoint ptr addrspace(3) %243 to i32, !dbg !16
  %dma.buffer.lds1620 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %253), !dbg !16
  %254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1620, i32 %dma.buffer.offset1618, <4 x i32> %dma.buffer.rsrc1619) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %255 = add nuw i64 %205, 1024, !dbg !16
  %256 = icmp slt i64 %255, 0, !dbg !16
  %257 = sub nuw i64 -1025, %205, !dbg !16
  %258 = select i1 %256, i64 %257, i64 %255, !dbg !16
  %259 = sdiv i64 %258, 4096, !dbg !16
  %.lobit37 = ashr i64 %255, 63, !dbg !16
  %260 = xor i64 %259, %.lobit37, !dbg !16
  %261 = add nsw i64 %260, %16, !dbg !16
  %262 = add nsw i64 %217, %195, !dbg !16
  %263 = add nsw i64 %262, 32, !dbg !16
  %264 = icmp slt i64 %262, -32, !dbg !16
  %265 = sub nsw i64 -33, %262, !dbg !16
  %266 = select i1 %264, i64 %265, i64 %263, !dbg !16
  %267 = sdiv i64 %266, 128, !dbg !16
  %268 = sext i1 %264 to i64, !dbg !16
  %269 = xor i64 %267, %268, !dbg !16
  %270 = shl nsw i64 %269, 7, !dbg !16
  %271 = sub i64 %195, %270, !dbg !16
  %272 = shl i64 %261, 19, !dbg !16
  %273 = and i64 %272, 16252928, !dbg !16
  %274 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %273, !dbg !16
  %275 = getelementptr [2 x i8], ptr addrspace(1) %274, i64 %217, !dbg !16
  %276 = getelementptr [2 x i8], ptr addrspace(1) %275, i64 %271, !dbg !16
  %277 = getelementptr i8, ptr addrspace(1) %276, i64 64, !dbg !16
  %278 = getelementptr [2 x i8], ptr addrspace(1) %277, i64 %223, !dbg !16
  %address.local501 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local469, !dbg !8
  %address.local502 = ptrtoint ptr addrspace(3) %address.local501 to i64, !dbg !8
  %address.local503 = trunc i64 %address.local502 to i32, !dbg !8
  %279 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local503), !dbg !16
  %280 = zext i32 %279 to i64, !dbg !16
  %281 = inttoptr i64 %280 to ptr addrspace(3), !dbg !16
  %address.local504 = getelementptr [2 x i8], ptr addrspace(1) %278, i64 %address.local482, !dbg !16
  %282 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %283 = ptrtoint ptr addrspace(1) %address.local504 to i64, !dbg !16
  %284 = sub i64 %283, %282, !dbg !16
  %dma.buffer.offset1615 = trunc i64 %284 to i32, !dbg !16
  %285 = trunc i64 %282 to i32, !dbg !16
  %286 = insertelement <4 x i32> poison, i32 %285, i64 0, !dbg !16
  %287 = lshr i64 %282, 32, !dbg !16
  %288 = trunc i64 %287 to i32, !dbg !16
  %289 = insertelement <4 x i32> %286, i32 %288, i64 1, !dbg !16
  %290 = insertelement <4 x i32> %289, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1616 = insertelement <4 x i32> %290, i32 131072, i64 3, !dbg !16
  %291 = ptrtoint ptr addrspace(3) %281 to i32, !dbg !16
  %dma.buffer.lds1617 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %291), !dbg !16
  %292 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1617, i32 %dma.buffer.offset1615, <4 x i32> %dma.buffer.rsrc1616) #10, !dbg !16
  %address.local505 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local486, !dbg !8
  %address.local506 = ptrtoint ptr addrspace(3) %address.local505 to i64, !dbg !8
  %address.local507 = trunc i64 %address.local506 to i32, !dbg !8
  %293 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local507), !dbg !16
  %294 = zext i32 %293 to i64, !dbg !16
  %295 = inttoptr i64 %294 to ptr addrspace(3), !dbg !16
  %address.local508 = getelementptr [2 x i8], ptr addrspace(1) %278, i64 %address.local499, !dbg !16
  %296 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %297 = ptrtoint ptr addrspace(1) %address.local508 to i64, !dbg !16
  %298 = sub i64 %297, %296, !dbg !16
  %dma.buffer.offset1612 = trunc i64 %298 to i32, !dbg !16
  %299 = trunc i64 %296 to i32, !dbg !16
  %300 = insertelement <4 x i32> poison, i32 %299, i64 0, !dbg !16
  %301 = lshr i64 %296, 32, !dbg !16
  %302 = trunc i64 %301 to i32, !dbg !16
  %303 = insertelement <4 x i32> %300, i32 %302, i64 1, !dbg !16
  %304 = insertelement <4 x i32> %303, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1613 = insertelement <4 x i32> %304, i32 131072, i64 3, !dbg !16
  %305 = ptrtoint ptr addrspace(3) %295 to i32, !dbg !16
  %dma.buffer.lds1614 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %305), !dbg !16
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1614, i32 %dma.buffer.offset1612, <4 x i32> %dma.buffer.rsrc1613) #10, !dbg !16
  %address.local509 = mul nuw nsw i64 %address.local467, 160, !dbg !17
  %address.local510 = lshr i64 %address.local467, 2, !dbg !17
  %address.local511 = mul nsw i64 %address.local510, -640, !dbg !17
  %address.local512 = add nsw i64 %address.local509, %address.local511, !dbg !17
  %address.local513 = shl nuw nsw i64 %address.local467, 1, !dbg !17
  %address.local514 = and i64 %address.local513, 24, !dbg !17
  %address.local515 = or disjoint i64 %address.local512, %address.local514, !dbg !17
  %address.local516 = shl i64 %address.local467, 62, !dbg !17
  %address.local517 = ashr exact i64 %address.local516, 57, !dbg !17
  %address.local518 = and i64 %address.local517, -64, !dbg !17
  %address.local519 = add nsw i64 %address.local515, %address.local518, !dbg !17
  %address.local520 = lshr i64 %address.local467, 2, !dbg !11
  %address.local521 = and i64 %address.local520, 12, !dbg !11
  %address.local522 = add nsw i64 %address.local519, %address.local521, !dbg !17
  %address.local523 = and i64 %address.local467, 63, !dbg !11
  %address.local524 = lshr i64 %address.local523, 5, !dbg !11
  %address.local525 = mul nuw nsw i64 %address.local524, 1016, !dbg !11
  %address.local526 = add i64 %address.local522, %address.local525, !dbg !17
  %address.local527 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local526, !dbg !17
  %307 = load <4 x half>, ptr addrspace(3) %address.local527, align 8, !dbg !17
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %307, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local527, i64 4096, !dbg !17
  %309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !17
  %address.local538 = shl nuw nsw i32 %address.thread466, 1, !dbg !17
  %address.local539 = and i32 %address.local538, 24, !dbg !17
  %address.local540 = shl nuw nsw i32 %address.thread466, 8, !dbg !17
  %address.local541 = and i32 %address.local540, 768, !dbg !17
  %address.local542 = or disjoint i32 %address.local539, %address.local541, !dbg !17
  %address.local543 = shl nuw nsw i32 %address.thread466, 5, !dbg !17
  %address.local544 = and i32 %address.local543, 96, !dbg !17
  %address.local545 = or disjoint i32 %address.local542, %address.local544, !dbg !17
  %address.local546 = zext nneg i32 %address.local545 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %203) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local547 = lshr i32 %address.thread466, 1, !dbg !17
  %address.local548 = and i32 %address.local547, 1, !dbg !17
  %address.local549 = zext nneg i32 %address.local548 to i64, !dbg !17
  %address.local550 = add nsw i64 %address.base, %address.local549, !dbg !17
  %address.local551 = shl i64 %address.local550, 6, !dbg !17
  %address.local552 = sub i64 %address.local546, %address.local551, !dbg !17
  %address.local553 = add i64 %address.local552, %address.local521, !dbg !17
  %address.local554 = add i64 %address.local553, %address.local525, !dbg !17
  %address.local555 = add i64 %address.local554, 32, !dbg !17
  %address.local556 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local555, !dbg !17
  %310 = load <4 x half>, ptr addrspace(3) %address.local556, align 8, !dbg !17
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %309, <4 x float> %308) #9, !dbg !17
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %310, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %address.local556, i64 4096, !dbg !17
  %313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !17
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %address.local527, i64 128, !dbg !17
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !17
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %313, <4 x float> %312) #9, !dbg !17
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %314, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local527, i64 4224, !dbg !17
  %317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !17
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local556, i64 128, !dbg !17
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !17
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %317, <4 x float> %316) #9, !dbg !17
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %318, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %address.local556, i64 4224, !dbg !17
  %321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !17
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %address.local527, i64 1024, !dbg !17
  %322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !17
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %321, <4 x float> %320) #9, !dbg !17
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %322, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local527, i64 5120, !dbg !17
  %325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !17
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local556, i64 1024, !dbg !17
  %326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !17
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %325, <4 x float> %324) #9, !dbg !17
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %326, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %address.local556, i64 5120, !dbg !17
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !17
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %address.local527, i64 1152, !dbg !17
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !17
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %329, <4 x float> %328) #9, !dbg !17
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %330, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1470 = getelementptr i8, ptr addrspace(3) %address.local527, i64 5248, !dbg !17
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1470, align 8, !dbg !17
  %lds.const.gep1471 = getelementptr i8, ptr addrspace(3) %address.local556, i64 1152, !dbg !17
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1471, align 8, !dbg !17
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %333, <4 x float> %332) #9, !dbg !17
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %334, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1472 = getelementptr i8, ptr addrspace(3) %address.local556, i64 5248, !dbg !17
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1472, align 8, !dbg !17
  %338 = load <4 x half>, ptr addrspace(3) %address.local527, align 8, !dbg !17
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %337, <4 x float> %336) #9, !dbg !17
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %338, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1473 = getelementptr i8, ptr addrspace(3) %address.local527, i64 4096, !dbg !17
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1473, align 8, !dbg !17
  %342 = load <4 x half>, ptr addrspace(3) %address.local556, align 8, !dbg !17
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %341, <4 x float> %340) #9, !dbg !17
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %342, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1474 = getelementptr i8, ptr addrspace(3) %address.local556, i64 4096, !dbg !17
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1474, align 8, !dbg !17
  %lds.const.gep1475 = getelementptr i8, ptr addrspace(3) %address.local527, i64 128, !dbg !17
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1475, align 8, !dbg !17
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %345, <4 x float> %344) #9, !dbg !17
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %346, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1476 = getelementptr i8, ptr addrspace(3) %address.local527, i64 4224, !dbg !17
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1476, align 8, !dbg !17
  %lds.const.gep1477 = getelementptr i8, ptr addrspace(3) %address.local556, i64 128, !dbg !17
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1477, align 8, !dbg !17
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %349, <4 x float> %348) #9, !dbg !17
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %350, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1478 = getelementptr i8, ptr addrspace(3) %address.local556, i64 4224, !dbg !17
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !17
  %lds.const.gep1479 = getelementptr i8, ptr addrspace(3) %address.local527, i64 1024, !dbg !17
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !17
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %353, <4 x float> %352) #9, !dbg !17
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %354, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1480 = getelementptr i8, ptr addrspace(3) %address.local527, i64 5120, !dbg !17
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1480, align 8, !dbg !17
  %lds.const.gep1481 = getelementptr i8, ptr addrspace(3) %address.local556, i64 1024, !dbg !17
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1481, align 8, !dbg !17
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %357, <4 x float> %356) #9, !dbg !17
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %358, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1482 = getelementptr i8, ptr addrspace(3) %address.local556, i64 5120, !dbg !17
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1482, align 8, !dbg !17
  %lds.const.gep1483 = getelementptr i8, ptr addrspace(3) %address.local527, i64 1152, !dbg !17
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1483, align 8, !dbg !17
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %361, <4 x float> %360) #9, !dbg !17
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %362, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1484 = getelementptr i8, ptr addrspace(3) %address.local527, i64 5248, !dbg !17
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1484, align 8, !dbg !17
  %lds.const.gep1485 = getelementptr i8, ptr addrspace(3) %address.local556, i64 1152, !dbg !17
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1485, align 8, !dbg !17
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %365, <4 x float> %364) #9, !dbg !17
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %366, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1486 = getelementptr i8, ptr addrspace(3) %address.local556, i64 5248, !dbg !17
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1486, align 8, !dbg !17
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %369, <4 x float> %368) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %371 = add nuw i64 %205, 2048, !dbg !16
  %372 = icmp slt i64 %371, 0, !dbg !16
  %373 = sub nuw i64 -2049, %205, !dbg !16
  %374 = select i1 %372, i64 %373, i64 %371, !dbg !16
  %375 = sdiv i64 %374, 4096, !dbg !16
  %.lobit38 = ashr i64 %371, 63, !dbg !16
  %376 = xor i64 %375, %.lobit38, !dbg !16
  %377 = add nsw i64 %376, %16, !dbg !16
  %378 = add nsw i64 %262, 64, !dbg !16
  %379 = icmp slt i64 %262, -64, !dbg !16
  %380 = sub nsw i64 -65, %262, !dbg !16
  %381 = select i1 %379, i64 %380, i64 %378, !dbg !16
  %382 = sdiv i64 %381, 128, !dbg !16
  %383 = sext i1 %379 to i64, !dbg !16
  %384 = xor i64 %382, %383, !dbg !16
  %385 = shl nsw i64 %384, 7, !dbg !16
  %386 = sub i64 %195, %385, !dbg !16
  %387 = shl i64 %377, 19, !dbg !16
  %388 = and i64 %387, 16252928, !dbg !16
  %389 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %388, !dbg !16
  %390 = getelementptr [2 x i8], ptr addrspace(1) %389, i64 %217, !dbg !16
  %391 = getelementptr [2 x i8], ptr addrspace(1) %390, i64 %386, !dbg !16
  %392 = getelementptr i8, ptr addrspace(1) %391, i64 128, !dbg !16
  %393 = getelementptr [2 x i8], ptr addrspace(1) %392, i64 %223, !dbg !16
  %394 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local472), !dbg !16
  %395 = zext i32 %394 to i64, !dbg !16
  %396 = inttoptr i64 %395 to ptr addrspace(3), !dbg !16
  %address.local865 = getelementptr [2 x i8], ptr addrspace(1) %393, i64 %address.local482, !dbg !16
  %397 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %398 = ptrtoint ptr addrspace(1) %address.local865 to i64, !dbg !16
  %399 = sub i64 %398, %397, !dbg !16
  %dma.buffer.offset1609 = trunc i64 %399 to i32, !dbg !16
  %400 = trunc i64 %397 to i32, !dbg !16
  %401 = insertelement <4 x i32> poison, i32 %400, i64 0, !dbg !16
  %402 = lshr i64 %397, 32, !dbg !16
  %403 = trunc i64 %402 to i32, !dbg !16
  %404 = insertelement <4 x i32> %401, i32 %403, i64 1, !dbg !16
  %405 = insertelement <4 x i32> %404, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1610 = insertelement <4 x i32> %405, i32 131072, i64 3, !dbg !16
  %406 = ptrtoint ptr addrspace(3) %396 to i32, !dbg !16
  %dma.buffer.lds1611 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %406), !dbg !16
  %407 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1611, i32 %dma.buffer.offset1609, <4 x i32> %dma.buffer.rsrc1610) #10, !dbg !16
  %408 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local489), !dbg !16
  %409 = zext i32 %408 to i64, !dbg !16
  %410 = inttoptr i64 %409 to ptr addrspace(3), !dbg !16
  %address.local866 = getelementptr [2 x i8], ptr addrspace(1) %393, i64 %address.local499, !dbg !16
  %411 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %412 = ptrtoint ptr addrspace(1) %address.local866 to i64, !dbg !16
  %413 = sub i64 %412, %411, !dbg !16
  %dma.buffer.offset1606 = trunc i64 %413 to i32, !dbg !16
  %414 = trunc i64 %411 to i32, !dbg !16
  %415 = insertelement <4 x i32> poison, i32 %414, i64 0, !dbg !16
  %416 = lshr i64 %411, 32, !dbg !16
  %417 = trunc i64 %416 to i32, !dbg !16
  %418 = insertelement <4 x i32> %415, i32 %417, i64 1, !dbg !16
  %419 = insertelement <4 x i32> %418, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1607 = insertelement <4 x i32> %419, i32 131072, i64 3, !dbg !16
  %420 = ptrtoint ptr addrspace(3) %410 to i32, !dbg !16
  %dma.buffer.lds1608 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %420), !dbg !16
  %421 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1608, i32 %dma.buffer.offset1606, <4 x i32> %dma.buffer.rsrc1607) #10, !dbg !16
  %lds.const.gep1487 = getelementptr i8, ptr addrspace(3) %address.local527, i64 8192, !dbg !17
  %422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1487, align 8, !dbg !17
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %422, <4 x float> %311) #9, !dbg !17
  %lds.const.gep1488 = getelementptr i8, ptr addrspace(3) %address.local527, i64 12288, !dbg !17
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1488, align 8, !dbg !17
  %lds.const.gep1489 = getelementptr i8, ptr addrspace(3) %address.local556, i64 8192, !dbg !17
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1489, align 8, !dbg !17
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %424, <4 x float> %423) #9, !dbg !17
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %425, <4 x float> %315) #9, !dbg !17
  %lds.const.gep1490 = getelementptr i8, ptr addrspace(3) %address.local556, i64 12288, !dbg !17
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1490, align 8, !dbg !17
  %lds.const.gep1491 = getelementptr i8, ptr addrspace(3) %address.local527, i64 8320, !dbg !17
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1491, align 8, !dbg !17
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %428, <4 x float> %427) #9, !dbg !17
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %429, <4 x float> %319) #9, !dbg !17
  %lds.const.gep1492 = getelementptr i8, ptr addrspace(3) %address.local527, i64 12416, !dbg !17
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1492, align 8, !dbg !17
  %lds.const.gep1493 = getelementptr i8, ptr addrspace(3) %address.local556, i64 8320, !dbg !17
  %433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1493, align 8, !dbg !17
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %432, <4 x float> %431) #9, !dbg !17
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %433, <4 x float> %323) #9, !dbg !17
  %lds.const.gep1494 = getelementptr i8, ptr addrspace(3) %address.local556, i64 12416, !dbg !17
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1494, align 8, !dbg !17
  %lds.const.gep1495 = getelementptr i8, ptr addrspace(3) %address.local527, i64 9216, !dbg !17
  %437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1495, align 8, !dbg !17
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %436, <4 x float> %435) #9, !dbg !17
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %437, <4 x float> %327) #9, !dbg !17
  %lds.const.gep1496 = getelementptr i8, ptr addrspace(3) %address.local527, i64 13312, !dbg !17
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1496, align 8, !dbg !17
  %lds.const.gep1497 = getelementptr i8, ptr addrspace(3) %address.local556, i64 9216, !dbg !17
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1497, align 8, !dbg !17
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %440, <4 x float> %439) #9, !dbg !17
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %441, <4 x float> %331) #9, !dbg !17
  %lds.const.gep1498 = getelementptr i8, ptr addrspace(3) %address.local556, i64 13312, !dbg !17
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1498, align 8, !dbg !17
  %lds.const.gep1499 = getelementptr i8, ptr addrspace(3) %address.local527, i64 9344, !dbg !17
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1499, align 8, !dbg !17
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %444, <4 x float> %443) #9, !dbg !17
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %445, <4 x float> %335) #9, !dbg !17
  %lds.const.gep1500 = getelementptr i8, ptr addrspace(3) %address.local527, i64 13440, !dbg !17
  %448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1500, align 8, !dbg !17
  %lds.const.gep1501 = getelementptr i8, ptr addrspace(3) %address.local556, i64 9344, !dbg !17
  %449 = load <4 x half>, ptr addrspace(3) %lds.const.gep1501, align 8, !dbg !17
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %448, <4 x float> %447) #9, !dbg !17
  %451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %449, <4 x float> %339) #9, !dbg !17
  %lds.const.gep1502 = getelementptr i8, ptr addrspace(3) %address.local556, i64 13440, !dbg !17
  %452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1502, align 8, !dbg !17
  %lds.const.gep1503 = getelementptr i8, ptr addrspace(3) %address.local527, i64 8192, !dbg !17
  %453 = load <4 x half>, ptr addrspace(3) %lds.const.gep1503, align 8, !dbg !17
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %452, <4 x float> %451) #9, !dbg !17
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %453, <4 x float> %343) #9, !dbg !17
  %lds.const.gep1504 = getelementptr i8, ptr addrspace(3) %address.local527, i64 12288, !dbg !17
  %456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1504, align 8, !dbg !17
  %lds.const.gep1505 = getelementptr i8, ptr addrspace(3) %address.local556, i64 8192, !dbg !17
  %457 = load <4 x half>, ptr addrspace(3) %lds.const.gep1505, align 8, !dbg !17
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %456, <4 x float> %455) #9, !dbg !17
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %457, <4 x float> %347) #9, !dbg !17
  %lds.const.gep1506 = getelementptr i8, ptr addrspace(3) %address.local556, i64 12288, !dbg !17
  %460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1506, align 8, !dbg !17
  %lds.const.gep1507 = getelementptr i8, ptr addrspace(3) %address.local527, i64 8320, !dbg !17
  %461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1507, align 8, !dbg !17
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %460, <4 x float> %459) #9, !dbg !17
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %461, <4 x float> %351) #9, !dbg !17
  %lds.const.gep1508 = getelementptr i8, ptr addrspace(3) %address.local527, i64 12416, !dbg !17
  %464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1508, align 8, !dbg !17
  %lds.const.gep1509 = getelementptr i8, ptr addrspace(3) %address.local556, i64 8320, !dbg !17
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1509, align 8, !dbg !17
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %464, <4 x float> %463) #9, !dbg !17
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %465, <4 x float> %355) #9, !dbg !17
  %lds.const.gep1510 = getelementptr i8, ptr addrspace(3) %address.local556, i64 12416, !dbg !17
  %468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1510, align 8, !dbg !17
  %lds.const.gep1511 = getelementptr i8, ptr addrspace(3) %address.local527, i64 9216, !dbg !17
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1511, align 8, !dbg !17
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %468, <4 x float> %467) #9, !dbg !17
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %469, <4 x float> %359) #9, !dbg !17
  %lds.const.gep1512 = getelementptr i8, ptr addrspace(3) %address.local527, i64 13312, !dbg !17
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1512, align 8, !dbg !17
  %lds.const.gep1513 = getelementptr i8, ptr addrspace(3) %address.local556, i64 9216, !dbg !17
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1513, align 8, !dbg !17
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %472, <4 x float> %471) #9, !dbg !17
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %473, <4 x float> %363) #9, !dbg !17
  %lds.const.gep1514 = getelementptr i8, ptr addrspace(3) %address.local556, i64 13312, !dbg !17
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1514, align 8, !dbg !17
  %lds.const.gep1515 = getelementptr i8, ptr addrspace(3) %address.local527, i64 9344, !dbg !17
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !17
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %476, <4 x float> %475) #9, !dbg !17
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %477, <4 x float> %367) #9, !dbg !17
  %lds.const.gep1516 = getelementptr i8, ptr addrspace(3) %address.local527, i64 13440, !dbg !17
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !17
  %lds.const.gep1517 = getelementptr i8, ptr addrspace(3) %address.local556, i64 9344, !dbg !17
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !17
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %480, <4 x float> %479) #9, !dbg !17
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %481, <4 x float> %370) #9, !dbg !17
  %lds.const.gep1518 = getelementptr i8, ptr addrspace(3) %address.local556, i64 13440, !dbg !17
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !17
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %484, <4 x float> %483) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %486 = add nuw i64 %205, 3072, !dbg !16
  %487 = icmp slt i64 %486, 0, !dbg !16
  %488 = sub nuw i64 -3073, %205, !dbg !16
  %489 = select i1 %487, i64 %488, i64 %486, !dbg !16
  %490 = sdiv i64 %489, 4096, !dbg !16
  %.lobit39 = ashr i64 %486, 63, !dbg !16
  %491 = xor i64 %490, %.lobit39, !dbg !16
  %492 = add nsw i64 %491, %16, !dbg !16
  %493 = add nsw i64 %262, 96, !dbg !16
  %494 = icmp slt i64 %262, -96, !dbg !16
  %495 = sub nsw i64 -97, %262, !dbg !16
  %496 = select i1 %494, i64 %495, i64 %493, !dbg !16
  %497 = sdiv i64 %496, 128, !dbg !16
  %498 = sext i1 %494 to i64, !dbg !16
  %499 = xor i64 %497, %498, !dbg !16
  %500 = shl nsw i64 %499, 7, !dbg !16
  %501 = sub i64 %195, %500, !dbg !16
  %502 = shl i64 %492, 19, !dbg !16
  %503 = and i64 %502, 16252928, !dbg !16
  %504 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %503, !dbg !16
  %505 = getelementptr [2 x i8], ptr addrspace(1) %504, i64 %217, !dbg !16
  %506 = getelementptr [2 x i8], ptr addrspace(1) %505, i64 %501, !dbg !16
  %507 = getelementptr i8, ptr addrspace(1) %506, i64 192, !dbg !16
  %508 = getelementptr [2 x i8], ptr addrspace(1) %507, i64 %223, !dbg !16
  %509 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local503), !dbg !16
  %510 = zext i32 %509 to i64, !dbg !16
  %511 = inttoptr i64 %510 to ptr addrspace(3), !dbg !16
  %address.local883 = getelementptr [2 x i8], ptr addrspace(1) %508, i64 %address.local482, !dbg !16
  %512 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %513 = ptrtoint ptr addrspace(1) %address.local883 to i64, !dbg !16
  %514 = sub i64 %513, %512, !dbg !16
  %dma.buffer.offset1603 = trunc i64 %514 to i32, !dbg !16
  %515 = trunc i64 %512 to i32, !dbg !16
  %516 = insertelement <4 x i32> poison, i32 %515, i64 0, !dbg !16
  %517 = lshr i64 %512, 32, !dbg !16
  %518 = trunc i64 %517 to i32, !dbg !16
  %519 = insertelement <4 x i32> %516, i32 %518, i64 1, !dbg !16
  %520 = insertelement <4 x i32> %519, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1604 = insertelement <4 x i32> %520, i32 131072, i64 3, !dbg !16
  %521 = ptrtoint ptr addrspace(3) %511 to i32, !dbg !16
  %dma.buffer.lds1605 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %521), !dbg !16
  %522 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1605, i32 %dma.buffer.offset1603, <4 x i32> %dma.buffer.rsrc1604) #10, !dbg !16
  %523 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local507), !dbg !16
  %524 = zext i32 %523 to i64, !dbg !16
  %525 = inttoptr i64 %524 to ptr addrspace(3), !dbg !16
  %address.local884 = getelementptr [2 x i8], ptr addrspace(1) %508, i64 %address.local499, !dbg !16
  %526 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %527 = ptrtoint ptr addrspace(1) %address.local884 to i64, !dbg !16
  %528 = sub i64 %527, %526, !dbg !16
  %dma.buffer.offset1600 = trunc i64 %528 to i32, !dbg !16
  %529 = trunc i64 %526 to i32, !dbg !16
  %530 = insertelement <4 x i32> poison, i32 %529, i64 0, !dbg !16
  %531 = lshr i64 %526, 32, !dbg !16
  %532 = trunc i64 %531 to i32, !dbg !16
  %533 = insertelement <4 x i32> %530, i32 %532, i64 1, !dbg !16
  %534 = insertelement <4 x i32> %533, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1601 = insertelement <4 x i32> %534, i32 131072, i64 3, !dbg !16
  %535 = ptrtoint ptr addrspace(3) %525 to i32, !dbg !16
  %dma.buffer.lds1602 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %535), !dbg !16
  %536 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1602, i32 %dma.buffer.offset1600, <4 x i32> %dma.buffer.rsrc1601) #10, !dbg !16
  %537 = load <4 x half>, ptr addrspace(3) %address.local527, align 8, !dbg !17
  %538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %537, <4 x float> %426) #9, !dbg !17
  %lds.const.gep1519 = getelementptr i8, ptr addrspace(3) %address.local527, i64 4096, !dbg !17
  %539 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !17
  %540 = load <4 x half>, ptr addrspace(3) %address.local556, align 8, !dbg !17
  %541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %539, <4 x float> %538) #9, !dbg !17
  %542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %540, <4 x float> %430) #9, !dbg !17
  %lds.const.gep1520 = getelementptr i8, ptr addrspace(3) %address.local556, i64 4096, !dbg !17
  %543 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !17
  %lds.const.gep1521 = getelementptr i8, ptr addrspace(3) %address.local527, i64 128, !dbg !17
  %544 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !17
  %545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %543, <4 x float> %542) #9, !dbg !17
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %544, <4 x float> %434) #9, !dbg !17
  %lds.const.gep1522 = getelementptr i8, ptr addrspace(3) %address.local527, i64 4224, !dbg !17
  %547 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !17
  %lds.const.gep1523 = getelementptr i8, ptr addrspace(3) %address.local556, i64 128, !dbg !17
  %548 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !17
  %549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %547, <4 x float> %546) #9, !dbg !17
  %550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %548, <4 x float> %438) #9, !dbg !17
  %lds.const.gep1524 = getelementptr i8, ptr addrspace(3) %address.local556, i64 4224, !dbg !17
  %551 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !17
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local527, i64 1024, !dbg !17
  %552 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !17
  %553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %551, <4 x float> %550) #9, !dbg !17
  %554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %552, <4 x float> %442) #9, !dbg !17
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local527, i64 5120, !dbg !17
  %555 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !17
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local556, i64 1024, !dbg !17
  %556 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !17
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %555, <4 x float> %554) #9, !dbg !17
  %558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %556, <4 x float> %446) #9, !dbg !17
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local556, i64 5120, !dbg !17
  %559 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !17
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local527, i64 1152, !dbg !17
  %560 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !17
  %561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %559, <4 x float> %558) #9, !dbg !17
  %562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %560, <4 x float> %450) #9, !dbg !17
  %lds.const.gep1530 = getelementptr i8, ptr addrspace(3) %address.local527, i64 5248, !dbg !17
  %563 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !17
  %lds.const.gep1531 = getelementptr i8, ptr addrspace(3) %address.local556, i64 1152, !dbg !17
  %564 = load <4 x half>, ptr addrspace(3) %lds.const.gep1531, align 8, !dbg !17
  %565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %563, <4 x float> %562) #9, !dbg !17
  %566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %564, <4 x float> %454) #9, !dbg !17
  %lds.const.gep1532 = getelementptr i8, ptr addrspace(3) %address.local556, i64 5248, !dbg !17
  %567 = load <4 x half>, ptr addrspace(3) %lds.const.gep1532, align 8, !dbg !17
  %568 = load <4 x half>, ptr addrspace(3) %address.local527, align 8, !dbg !17
  %569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %567, <4 x float> %566) #9, !dbg !17
  %570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %568, <4 x float> %458) #9, !dbg !17
  %lds.const.gep1533 = getelementptr i8, ptr addrspace(3) %address.local527, i64 4096, !dbg !17
  %571 = load <4 x half>, ptr addrspace(3) %lds.const.gep1533, align 8, !dbg !17
  %572 = load <4 x half>, ptr addrspace(3) %address.local556, align 8, !dbg !17
  %573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %571, <4 x float> %570) #9, !dbg !17
  %574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %572, <4 x float> %462) #9, !dbg !17
  %lds.const.gep1534 = getelementptr i8, ptr addrspace(3) %address.local556, i64 4096, !dbg !17
  %575 = load <4 x half>, ptr addrspace(3) %lds.const.gep1534, align 8, !dbg !17
  %lds.const.gep1535 = getelementptr i8, ptr addrspace(3) %address.local527, i64 128, !dbg !17
  %576 = load <4 x half>, ptr addrspace(3) %lds.const.gep1535, align 8, !dbg !17
  %577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %575, <4 x float> %574) #9, !dbg !17
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %576, <4 x float> %466) #9, !dbg !17
  %lds.const.gep1536 = getelementptr i8, ptr addrspace(3) %address.local527, i64 4224, !dbg !17
  %579 = load <4 x half>, ptr addrspace(3) %lds.const.gep1536, align 8, !dbg !17
  %lds.const.gep1537 = getelementptr i8, ptr addrspace(3) %address.local556, i64 128, !dbg !17
  %580 = load <4 x half>, ptr addrspace(3) %lds.const.gep1537, align 8, !dbg !17
  %581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %579, <4 x float> %578) #9, !dbg !17
  %582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %580, <4 x float> %470) #9, !dbg !17
  %lds.const.gep1538 = getelementptr i8, ptr addrspace(3) %address.local556, i64 4224, !dbg !17
  %583 = load <4 x half>, ptr addrspace(3) %lds.const.gep1538, align 8, !dbg !17
  %lds.const.gep1539 = getelementptr i8, ptr addrspace(3) %address.local527, i64 1024, !dbg !17
  %584 = load <4 x half>, ptr addrspace(3) %lds.const.gep1539, align 8, !dbg !17
  %585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %583, <4 x float> %582) #9, !dbg !17
  %586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %584, <4 x float> %474) #9, !dbg !17
  %lds.const.gep1540 = getelementptr i8, ptr addrspace(3) %address.local527, i64 5120, !dbg !17
  %587 = load <4 x half>, ptr addrspace(3) %lds.const.gep1540, align 8, !dbg !17
  %lds.const.gep1541 = getelementptr i8, ptr addrspace(3) %address.local556, i64 1024, !dbg !17
  %588 = load <4 x half>, ptr addrspace(3) %lds.const.gep1541, align 8, !dbg !17
  %589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %587, <4 x float> %586) #9, !dbg !17
  %590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %588, <4 x float> %478) #9, !dbg !17
  %lds.const.gep1542 = getelementptr i8, ptr addrspace(3) %address.local556, i64 5120, !dbg !17
  %591 = load <4 x half>, ptr addrspace(3) %lds.const.gep1542, align 8, !dbg !17
  %lds.const.gep1543 = getelementptr i8, ptr addrspace(3) %address.local527, i64 1152, !dbg !17
  %592 = load <4 x half>, ptr addrspace(3) %lds.const.gep1543, align 8, !dbg !17
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %591, <4 x float> %590) #9, !dbg !17
  %594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %592, <4 x float> %482) #9, !dbg !17
  %lds.const.gep1544 = getelementptr i8, ptr addrspace(3) %address.local527, i64 5248, !dbg !17
  %595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1544, align 8, !dbg !17
  %lds.const.gep1545 = getelementptr i8, ptr addrspace(3) %address.local556, i64 1152, !dbg !17
  %596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1545, align 8, !dbg !17
  %597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %595, <4 x float> %594) #9, !dbg !17
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %596, <4 x float> %485) #9, !dbg !17
  %lds.const.gep1546 = getelementptr i8, ptr addrspace(3) %address.local556, i64 5248, !dbg !17
  %599 = load <4 x half>, ptr addrspace(3) %lds.const.gep1546, align 8, !dbg !17
  %600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %599, <4 x float> %598) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %601 = add nuw nsw i64 %217, %16, !dbg !18
  %602 = shl i64 %601, 19, !dbg !18
  %603 = and i64 %602, 16252928, !dbg !18
  %604 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %603, !dbg !18
  %605 = getelementptr [2 x i8], ptr addrspace(1) %604, i64 %223, !dbg !18
  %606 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local472), !dbg !18
  %607 = zext i32 %606 to i64, !dbg !18
  %608 = inttoptr i64 %607 to ptr addrspace(3), !dbg !18
  %address.local885 = shl nuw nsw i64 %address.local467, 5, !dbg !18
  %address.local886 = and i64 %address.local885, 8064, !dbg !18
  %address.local887 = shl nuw nsw i64 %address.local467, 3, !dbg !18
  %address.local888 = and i64 %address.local887, 24, !dbg !18
  %address.local889 = or disjoint i64 %address.local886, %address.local888, !dbg !18
  %address.local890 = getelementptr [2 x i8], ptr addrspace(1) %605, i64 %address.local889, !dbg !18
  %609 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %610 = ptrtoint ptr addrspace(1) %address.local890 to i64, !dbg !18
  %611 = sub i64 %610, %609, !dbg !18
  %dma.buffer.offset1597 = trunc i64 %611 to i32, !dbg !18
  %612 = trunc i64 %609 to i32, !dbg !18
  %613 = insertelement <4 x i32> poison, i32 %612, i64 0, !dbg !18
  %614 = lshr i64 %609, 32, !dbg !18
  %615 = trunc i64 %614 to i32, !dbg !18
  %616 = insertelement <4 x i32> %613, i32 %615, i64 1, !dbg !18
  %617 = insertelement <4 x i32> %616, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1598 = insertelement <4 x i32> %617, i32 131072, i64 3, !dbg !18
  %618 = ptrtoint ptr addrspace(3) %608 to i32, !dbg !18
  %dma.buffer.lds1599 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %618), !dbg !18
  %619 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1599, i32 %dma.buffer.offset1597, <4 x i32> %dma.buffer.rsrc1598) #10, !dbg !18
  %620 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local489), !dbg !18
  %621 = zext i32 %620 to i64, !dbg !18
  %622 = inttoptr i64 %621 to ptr addrspace(3), !dbg !18
  %address.local891 = shl nuw nsw i64 %address.local484, 5, !dbg !18
  %address.local892 = and i64 %address.local891, 16256, !dbg !18
  %address.local893 = shl nuw nsw i64 %address.local467, 3, !dbg !18
  %address.local894 = and i64 %address.local893, 24, !dbg !18
  %address.local895 = or disjoint i64 %address.local892, %address.local894, !dbg !18
  %address.local896 = getelementptr [2 x i8], ptr addrspace(1) %605, i64 %address.local895, !dbg !18
  %623 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %624 = ptrtoint ptr addrspace(1) %address.local896 to i64, !dbg !18
  %625 = sub i64 %624, %623, !dbg !18
  %dma.buffer.offset1594 = trunc i64 %625 to i32, !dbg !18
  %626 = trunc i64 %623 to i32, !dbg !18
  %627 = insertelement <4 x i32> poison, i32 %626, i64 0, !dbg !18
  %628 = lshr i64 %623, 32, !dbg !18
  %629 = trunc i64 %628 to i32, !dbg !18
  %630 = insertelement <4 x i32> %627, i32 %629, i64 1, !dbg !18
  %631 = insertelement <4 x i32> %630, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1595 = insertelement <4 x i32> %631, i32 131072, i64 3, !dbg !18
  %632 = ptrtoint ptr addrspace(3) %622 to i32, !dbg !18
  %dma.buffer.lds1596 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %632), !dbg !18
  %633 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1596, i32 %dma.buffer.offset1594, <4 x i32> %dma.buffer.rsrc1595) #10, !dbg !18
  %lds.const.gep1547 = getelementptr i8, ptr addrspace(3) %address.local527, i64 8192, !dbg !17
  %634 = load <4 x half>, ptr addrspace(3) %lds.const.gep1547, align 8, !dbg !17
  %635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %634, <4 x float> %541) #9, !dbg !17
  %lds.const.gep1548 = getelementptr i8, ptr addrspace(3) %address.local527, i64 12288, !dbg !17
  %636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1548, align 8, !dbg !17
  %lds.const.gep1549 = getelementptr i8, ptr addrspace(3) %address.local556, i64 8192, !dbg !17
  %637 = load <4 x half>, ptr addrspace(3) %lds.const.gep1549, align 8, !dbg !17
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %636, <4 x float> %635) #9, !dbg !17
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %637, <4 x float> %545) #9, !dbg !17
  %lds.const.gep1550 = getelementptr i8, ptr addrspace(3) %address.local556, i64 12288, !dbg !17
  %640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1550, align 8, !dbg !17
  %lds.const.gep1551 = getelementptr i8, ptr addrspace(3) %address.local527, i64 8320, !dbg !17
  %641 = load <4 x half>, ptr addrspace(3) %lds.const.gep1551, align 8, !dbg !17
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %640, <4 x float> %639) #9, !dbg !17
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %641, <4 x float> %549) #9, !dbg !17
  %lds.const.gep1552 = getelementptr i8, ptr addrspace(3) %address.local527, i64 12416, !dbg !17
  %644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1552, align 8, !dbg !17
  %lds.const.gep1553 = getelementptr i8, ptr addrspace(3) %address.local556, i64 8320, !dbg !17
  %645 = load <4 x half>, ptr addrspace(3) %lds.const.gep1553, align 8, !dbg !17
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %644, <4 x float> %643) #9, !dbg !17
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %645, <4 x float> %553) #9, !dbg !17
  %lds.const.gep1554 = getelementptr i8, ptr addrspace(3) %address.local556, i64 12416, !dbg !17
  %648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1554, align 8, !dbg !17
  %lds.const.gep1555 = getelementptr i8, ptr addrspace(3) %address.local527, i64 9216, !dbg !17
  %649 = load <4 x half>, ptr addrspace(3) %lds.const.gep1555, align 8, !dbg !17
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %648, <4 x float> %647) #9, !dbg !17
  %651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %649, <4 x float> %557) #9, !dbg !17
  %lds.const.gep1556 = getelementptr i8, ptr addrspace(3) %address.local527, i64 13312, !dbg !17
  %652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1556, align 8, !dbg !17
  %lds.const.gep1557 = getelementptr i8, ptr addrspace(3) %address.local556, i64 9216, !dbg !17
  %653 = load <4 x half>, ptr addrspace(3) %lds.const.gep1557, align 8, !dbg !17
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %652, <4 x float> %651) #9, !dbg !17
  %655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %653, <4 x float> %561) #9, !dbg !17
  %lds.const.gep1558 = getelementptr i8, ptr addrspace(3) %address.local556, i64 13312, !dbg !17
  %656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1558, align 8, !dbg !17
  %lds.const.gep1559 = getelementptr i8, ptr addrspace(3) %address.local527, i64 9344, !dbg !17
  %657 = load <4 x half>, ptr addrspace(3) %lds.const.gep1559, align 8, !dbg !17
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %656, <4 x float> %655) #9, !dbg !17
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %657, <4 x float> %565) #9, !dbg !17
  %lds.const.gep1560 = getelementptr i8, ptr addrspace(3) %address.local527, i64 13440, !dbg !17
  %660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1560, align 8, !dbg !17
  %lds.const.gep1561 = getelementptr i8, ptr addrspace(3) %address.local556, i64 9344, !dbg !17
  %661 = load <4 x half>, ptr addrspace(3) %lds.const.gep1561, align 8, !dbg !17
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %660, <4 x float> %659) #9, !dbg !17
  %663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %661, <4 x float> %569) #9, !dbg !17
  %lds.const.gep1562 = getelementptr i8, ptr addrspace(3) %address.local556, i64 13440, !dbg !17
  %664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1562, align 8, !dbg !17
  %lds.const.gep1563 = getelementptr i8, ptr addrspace(3) %address.local527, i64 8192, !dbg !17
  %665 = load <4 x half>, ptr addrspace(3) %lds.const.gep1563, align 8, !dbg !17
  %666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %664, <4 x float> %663) #9, !dbg !17
  %667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %665, <4 x float> %573) #9, !dbg !17
  %lds.const.gep1564 = getelementptr i8, ptr addrspace(3) %address.local527, i64 12288, !dbg !17
  %668 = load <4 x half>, ptr addrspace(3) %lds.const.gep1564, align 8, !dbg !17
  %lds.const.gep1565 = getelementptr i8, ptr addrspace(3) %address.local556, i64 8192, !dbg !17
  %669 = load <4 x half>, ptr addrspace(3) %lds.const.gep1565, align 8, !dbg !17
  %670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %668, <4 x float> %667) #9, !dbg !17
  %671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %669, <4 x float> %577) #9, !dbg !17
  %lds.const.gep1566 = getelementptr i8, ptr addrspace(3) %address.local556, i64 12288, !dbg !17
  %672 = load <4 x half>, ptr addrspace(3) %lds.const.gep1566, align 8, !dbg !17
  %lds.const.gep1567 = getelementptr i8, ptr addrspace(3) %address.local527, i64 8320, !dbg !17
  %673 = load <4 x half>, ptr addrspace(3) %lds.const.gep1567, align 8, !dbg !17
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %672, <4 x float> %671) #9, !dbg !17
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %673, <4 x float> %581) #9, !dbg !17
  %lds.const.gep1568 = getelementptr i8, ptr addrspace(3) %address.local527, i64 12416, !dbg !17
  %676 = load <4 x half>, ptr addrspace(3) %lds.const.gep1568, align 8, !dbg !17
  %lds.const.gep1569 = getelementptr i8, ptr addrspace(3) %address.local556, i64 8320, !dbg !17
  %677 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !17
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %676, <4 x float> %675) #9, !dbg !17
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %677, <4 x float> %585) #9, !dbg !17
  %lds.const.gep1570 = getelementptr i8, ptr addrspace(3) %address.local556, i64 12416, !dbg !17
  %680 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !17
  %lds.const.gep1571 = getelementptr i8, ptr addrspace(3) %address.local527, i64 9216, !dbg !17
  %681 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !17
  %682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %680, <4 x float> %679) #9, !dbg !17
  %683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %681, <4 x float> %589) #9, !dbg !17
  %lds.const.gep1572 = getelementptr i8, ptr addrspace(3) %address.local527, i64 13312, !dbg !17
  %684 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !17
  %lds.const.gep1573 = getelementptr i8, ptr addrspace(3) %address.local556, i64 9216, !dbg !17
  %685 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !17
  %686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %684, <4 x float> %683) #9, !dbg !17
  %687 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %685, <4 x float> %593) #9, !dbg !17
  %lds.const.gep1574 = getelementptr i8, ptr addrspace(3) %address.local556, i64 13312, !dbg !17
  %688 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !17
  %lds.const.gep1575 = getelementptr i8, ptr addrspace(3) %address.local527, i64 9344, !dbg !17
  %689 = load <4 x half>, ptr addrspace(3) %lds.const.gep1575, align 8, !dbg !17
  %690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %688, <4 x float> %687) #9, !dbg !17
  %691 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %689, <4 x float> %597) #9, !dbg !17
  %lds.const.gep1576 = getelementptr i8, ptr addrspace(3) %address.local527, i64 13440, !dbg !17
  %692 = load <4 x half>, ptr addrspace(3) %lds.const.gep1576, align 8, !dbg !17
  %lds.const.gep1577 = getelementptr i8, ptr addrspace(3) %address.local556, i64 9344, !dbg !17
  %693 = load <4 x half>, ptr addrspace(3) %lds.const.gep1577, align 8, !dbg !17
  %694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %692, <4 x float> %691) #9, !dbg !17
  %695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %693, <4 x float> %600) #9, !dbg !17
  %lds.const.gep1578 = getelementptr i8, ptr addrspace(3) %address.local556, i64 13440, !dbg !17
  %696 = load <4 x half>, ptr addrspace(3) %lds.const.gep1578, align 8, !dbg !17
  %697 = or disjoint i64 %214, 127, !dbg !13
  %698 = icmp uge i64 %64, %697, !dbg !13
  %699 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %696, <4 x float> %695) #9, !dbg !17
  %700 = and i1 %190, %698, !dbg !13
  br i1 %700, label %701, label %702, !dbg !13

701:                                              ; preds = %213
  br label %828, !dbg !13

702:                                              ; preds = %213
  %703 = add i64 %64, %32, !dbg !13
  %704 = shl nuw nsw i64 %65, 4, !dbg !13
  %705 = add i64 %703, %704, !dbg !13
  %706 = and i32 %31, 240, !dbg !13
  %index.compact = zext nneg i32 %706 to i64, !dbg !13
  %707 = sub i64 %705, %index.compact, !dbg !13
  %708 = and i64 %32, 48, !dbg !13
  %709 = or disjoint i64 %214, %708, !dbg !13
  %710 = add i64 %707, 1, !dbg !23
  %711 = lshr i64 %710, 32, !dbg !24
  %compare.high = trunc i64 %711 to i32, !dbg !24
  %712 = lshr i64 %709, 32, !dbg !24
  %compare.high1190 = trunc i64 %712 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1190, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1190, !dbg !24
  %compare.low.base = trunc i64 %710 to i32, !dbg !24
  %compare.low.base1191 = trunc i64 %709 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1191, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %713 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low1192 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result1193 = select i1 %compare.high.equal, i1 %compare.low1192, i1 %compare.high.order, !dbg !24
  %714 = select i1 %compare.result1193, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1194 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1195 = or i32 %compare.local.base1194, 2, !dbg !24
  %compare.low1196 = icmp ugt i32 %compare.low.base, %compare.local1195, !dbg !24
  %compare.result1197 = select i1 %compare.high.equal, i1 %compare.low1196, i1 %compare.high.order, !dbg !24
  %715 = select i1 %compare.result1197, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1198 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1199 = or i32 %compare.local.base1198, 3, !dbg !24
  %compare.low1200 = icmp ugt i32 %compare.low.base, %compare.local1199, !dbg !24
  %compare.result1201 = select i1 %compare.high.equal, i1 %compare.low1200, i1 %compare.high.order, !dbg !24
  %716 = select i1 %compare.result1201, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1202 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1203 = or i32 %compare.local.base1202, 4, !dbg !24
  %compare.low1204 = icmp ugt i32 %compare.low.base, %compare.local1203, !dbg !24
  %compare.result1205 = select i1 %compare.high.equal, i1 %compare.low1204, i1 %compare.high.order, !dbg !24
  %717 = select i1 %compare.result1205, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1206 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1207 = or i32 %compare.local.base1206, 5, !dbg !24
  %compare.low1208 = icmp ugt i32 %compare.low.base, %compare.local1207, !dbg !24
  %compare.result1209 = select i1 %compare.high.equal, i1 %compare.low1208, i1 %compare.high.order, !dbg !24
  %718 = select i1 %compare.result1209, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1210 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1211 = or i32 %compare.local.base1210, 6, !dbg !24
  %compare.low1212 = icmp ugt i32 %compare.low.base, %compare.local1211, !dbg !24
  %compare.result1213 = select i1 %compare.high.equal, i1 %compare.low1212, i1 %compare.high.order, !dbg !24
  %719 = select i1 %compare.result1213, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1214 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1215 = or i32 %compare.local.base1214, 7, !dbg !24
  %compare.low1216 = icmp ugt i32 %compare.low.base, %compare.local1215, !dbg !24
  %compare.result1217 = select i1 %compare.high.equal, i1 %compare.low1216, i1 %compare.high.order, !dbg !24
  %720 = select i1 %compare.result1217, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1218 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1219 = or i32 %compare.local.base1218, 8, !dbg !24
  %compare.low1220 = icmp ugt i32 %compare.low.base, %compare.local1219, !dbg !24
  %compare.result1221 = select i1 %compare.high.equal, i1 %compare.low1220, i1 %compare.high.order, !dbg !24
  %721 = select i1 %compare.result1221, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1222 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1223 = or i32 %compare.local.base1222, 9, !dbg !24
  %compare.low1224 = icmp ugt i32 %compare.low.base, %compare.local1223, !dbg !24
  %compare.result1225 = select i1 %compare.high.equal, i1 %compare.low1224, i1 %compare.high.order, !dbg !24
  %722 = select i1 %compare.result1225, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1226 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1227 = or i32 %compare.local.base1226, 10, !dbg !24
  %compare.low1228 = icmp ugt i32 %compare.low.base, %compare.local1227, !dbg !24
  %compare.result1229 = select i1 %compare.high.equal, i1 %compare.low1228, i1 %compare.high.order, !dbg !24
  %723 = select i1 %compare.result1229, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1230 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1231 = or i32 %compare.local.base1230, 11, !dbg !24
  %compare.low1232 = icmp ugt i32 %compare.low.base, %compare.local1231, !dbg !24
  %compare.result1233 = select i1 %compare.high.equal, i1 %compare.low1232, i1 %compare.high.order, !dbg !24
  %724 = select i1 %compare.result1233, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1234 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1235 = or i32 %compare.local.base1234, 12, !dbg !24
  %compare.low1236 = icmp ugt i32 %compare.low.base, %compare.local1235, !dbg !24
  %compare.result1237 = select i1 %compare.high.equal, i1 %compare.low1236, i1 %compare.high.order, !dbg !24
  %725 = select i1 %compare.result1237, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1238 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1239 = or i32 %compare.local.base1238, 13, !dbg !24
  %compare.low1240 = icmp ugt i32 %compare.low.base, %compare.local1239, !dbg !24
  %compare.result1241 = select i1 %compare.high.equal, i1 %compare.low1240, i1 %compare.high.order, !dbg !24
  %726 = select i1 %compare.result1241, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1242 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1243 = or i32 %compare.local.base1242, 14, !dbg !24
  %compare.low1244 = icmp ugt i32 %compare.low.base, %compare.local1243, !dbg !24
  %compare.result1245 = select i1 %compare.high.equal, i1 %compare.low1244, i1 %compare.high.order, !dbg !24
  %727 = select i1 %compare.result1245, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1246 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1247 = or i32 %compare.local.base1246, 15, !dbg !24
  %compare.low1248 = icmp ugt i32 %compare.low.base, %compare.local1247, !dbg !24
  %compare.result1249 = select i1 %compare.high.equal, i1 %compare.low1248, i1 %compare.high.order, !dbg !24
  %728 = select i1 %compare.result1249, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1250 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1251 = or i32 %compare.local.base1250, 64, !dbg !24
  %compare.low1252 = icmp ugt i32 %compare.low.base, %compare.local1251, !dbg !24
  %compare.result1253 = select i1 %compare.high.equal, i1 %compare.low1252, i1 %compare.high.order, !dbg !24
  %729 = select i1 %compare.result1253, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1254 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1255 = or i32 %compare.local.base1254, 65, !dbg !24
  %compare.low1256 = icmp ugt i32 %compare.low.base, %compare.local1255, !dbg !24
  %compare.result1257 = select i1 %compare.high.equal, i1 %compare.low1256, i1 %compare.high.order, !dbg !24
  %730 = select i1 %compare.result1257, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1258 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1259 = or i32 %compare.local.base1258, 66, !dbg !24
  %compare.low1260 = icmp ugt i32 %compare.low.base, %compare.local1259, !dbg !24
  %compare.result1261 = select i1 %compare.high.equal, i1 %compare.low1260, i1 %compare.high.order, !dbg !24
  %731 = select i1 %compare.result1261, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1262 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1263 = or i32 %compare.local.base1262, 67, !dbg !24
  %compare.low1264 = icmp ugt i32 %compare.low.base, %compare.local1263, !dbg !24
  %compare.result1265 = select i1 %compare.high.equal, i1 %compare.low1264, i1 %compare.high.order, !dbg !24
  %732 = select i1 %compare.result1265, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1266 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1267 = or i32 %compare.local.base1266, 68, !dbg !24
  %compare.low1268 = icmp ugt i32 %compare.low.base, %compare.local1267, !dbg !24
  %compare.result1269 = select i1 %compare.high.equal, i1 %compare.low1268, i1 %compare.high.order, !dbg !24
  %733 = select i1 %compare.result1269, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1270 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1271 = or i32 %compare.local.base1270, 69, !dbg !24
  %compare.low1272 = icmp ugt i32 %compare.low.base, %compare.local1271, !dbg !24
  %compare.result1273 = select i1 %compare.high.equal, i1 %compare.low1272, i1 %compare.high.order, !dbg !24
  %734 = select i1 %compare.result1273, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1274 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1275 = or i32 %compare.local.base1274, 70, !dbg !24
  %compare.low1276 = icmp ugt i32 %compare.low.base, %compare.local1275, !dbg !24
  %compare.result1277 = select i1 %compare.high.equal, i1 %compare.low1276, i1 %compare.high.order, !dbg !24
  %735 = select i1 %compare.result1277, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1278 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1279 = or i32 %compare.local.base1278, 71, !dbg !24
  %compare.low1280 = icmp ugt i32 %compare.low.base, %compare.local1279, !dbg !24
  %compare.result1281 = select i1 %compare.high.equal, i1 %compare.low1280, i1 %compare.high.order, !dbg !24
  %736 = select i1 %compare.result1281, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1282 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1283 = or i32 %compare.local.base1282, 72, !dbg !24
  %compare.low1284 = icmp ugt i32 %compare.low.base, %compare.local1283, !dbg !24
  %compare.result1285 = select i1 %compare.high.equal, i1 %compare.low1284, i1 %compare.high.order, !dbg !24
  %737 = select i1 %compare.result1285, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1286 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1287 = or i32 %compare.local.base1286, 73, !dbg !24
  %compare.low1288 = icmp ugt i32 %compare.low.base, %compare.local1287, !dbg !24
  %compare.result1289 = select i1 %compare.high.equal, i1 %compare.low1288, i1 %compare.high.order, !dbg !24
  %738 = select i1 %compare.result1289, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1290 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1291 = or i32 %compare.local.base1290, 74, !dbg !24
  %compare.low1292 = icmp ugt i32 %compare.low.base, %compare.local1291, !dbg !24
  %compare.result1293 = select i1 %compare.high.equal, i1 %compare.low1292, i1 %compare.high.order, !dbg !24
  %739 = select i1 %compare.result1293, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1294 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1295 = or i32 %compare.local.base1294, 75, !dbg !24
  %compare.low1296 = icmp ugt i32 %compare.low.base, %compare.local1295, !dbg !24
  %compare.result1297 = select i1 %compare.high.equal, i1 %compare.low1296, i1 %compare.high.order, !dbg !24
  %740 = select i1 %compare.result1297, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1298 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1299 = or i32 %compare.local.base1298, 76, !dbg !24
  %compare.low1300 = icmp ugt i32 %compare.low.base, %compare.local1299, !dbg !24
  %compare.result1301 = select i1 %compare.high.equal, i1 %compare.low1300, i1 %compare.high.order, !dbg !24
  %741 = select i1 %compare.result1301, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1302 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1303 = or i32 %compare.local.base1302, 77, !dbg !24
  %compare.low1304 = icmp ugt i32 %compare.low.base, %compare.local1303, !dbg !24
  %compare.result1305 = select i1 %compare.high.equal, i1 %compare.low1304, i1 %compare.high.order, !dbg !24
  %742 = select i1 %compare.result1305, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1306 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1307 = or i32 %compare.local.base1306, 78, !dbg !24
  %compare.low1308 = icmp ugt i32 %compare.low.base, %compare.local1307, !dbg !24
  %compare.result1309 = select i1 %compare.high.equal, i1 %compare.low1308, i1 %compare.high.order, !dbg !24
  %743 = select i1 %compare.result1309, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1310 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1191, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1311 = or i32 %compare.local.base1310, 79, !dbg !24
  %compare.low1312 = icmp ugt i32 %compare.low.base, %compare.local1311, !dbg !24
  %compare.result1313 = select i1 %compare.high.equal, i1 %compare.low1312, i1 %compare.high.order, !dbg !24
  %744 = select i1 %compare.result1313, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %745 = add i64 %707, 65, !dbg !23
  %746 = lshr i64 %745, 32, !dbg !24
  %compare.high1314 = trunc i64 %746 to i32, !dbg !24
  %747 = lshr i64 %709, 32, !dbg !24
  %compare.high1315 = trunc i64 %747 to i32, !dbg !24
  %compare.high.equal1316 = icmp eq i32 %compare.high1314, %compare.high1315, !dbg !24
  %compare.high.order1317 = icmp ugt i32 %compare.high1314, %compare.high1315, !dbg !24
  %compare.low.base1318 = trunc i64 %745 to i32, !dbg !24
  %compare.low.base1319 = trunc i64 %709 to i32, !dbg !24
  %compare.low1320 = icmp ugt i32 %compare.low.base1318, %compare.low.base1319, !dbg !24
  %compare.result1321 = select i1 %compare.high.equal1316, i1 %compare.low1320, i1 %compare.high.order1317, !dbg !24
  %748 = select i1 %compare.result1321, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1322 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1323 = or i32 %compare.local.base1322, 1, !dbg !24
  %compare.low1324 = icmp ugt i32 %compare.low.base1318, %compare.local1323, !dbg !24
  %compare.result1325 = select i1 %compare.high.equal1316, i1 %compare.low1324, i1 %compare.high.order1317, !dbg !24
  %749 = select i1 %compare.result1325, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1326 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1327 = or i32 %compare.local.base1326, 2, !dbg !24
  %compare.low1328 = icmp ugt i32 %compare.low.base1318, %compare.local1327, !dbg !24
  %compare.result1329 = select i1 %compare.high.equal1316, i1 %compare.low1328, i1 %compare.high.order1317, !dbg !24
  %750 = select i1 %compare.result1329, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1330 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1331 = or i32 %compare.local.base1330, 3, !dbg !24
  %compare.low1332 = icmp ugt i32 %compare.low.base1318, %compare.local1331, !dbg !24
  %compare.result1333 = select i1 %compare.high.equal1316, i1 %compare.low1332, i1 %compare.high.order1317, !dbg !24
  %751 = select i1 %compare.result1333, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1334 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1335 = or i32 %compare.local.base1334, 4, !dbg !24
  %compare.low1336 = icmp ugt i32 %compare.low.base1318, %compare.local1335, !dbg !24
  %compare.result1337 = select i1 %compare.high.equal1316, i1 %compare.low1336, i1 %compare.high.order1317, !dbg !24
  %752 = select i1 %compare.result1337, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1338 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1339 = or i32 %compare.local.base1338, 5, !dbg !24
  %compare.low1340 = icmp ugt i32 %compare.low.base1318, %compare.local1339, !dbg !24
  %compare.result1341 = select i1 %compare.high.equal1316, i1 %compare.low1340, i1 %compare.high.order1317, !dbg !24
  %753 = select i1 %compare.result1341, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1342 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1343 = or i32 %compare.local.base1342, 6, !dbg !24
  %compare.low1344 = icmp ugt i32 %compare.low.base1318, %compare.local1343, !dbg !24
  %compare.result1345 = select i1 %compare.high.equal1316, i1 %compare.low1344, i1 %compare.high.order1317, !dbg !24
  %754 = select i1 %compare.result1345, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1346 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1347 = or i32 %compare.local.base1346, 7, !dbg !24
  %compare.low1348 = icmp ugt i32 %compare.low.base1318, %compare.local1347, !dbg !24
  %compare.result1349 = select i1 %compare.high.equal1316, i1 %compare.low1348, i1 %compare.high.order1317, !dbg !24
  %755 = select i1 %compare.result1349, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1350 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1351 = or i32 %compare.local.base1350, 8, !dbg !24
  %compare.low1352 = icmp ugt i32 %compare.low.base1318, %compare.local1351, !dbg !24
  %compare.result1353 = select i1 %compare.high.equal1316, i1 %compare.low1352, i1 %compare.high.order1317, !dbg !24
  %756 = select i1 %compare.result1353, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1354 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1355 = or i32 %compare.local.base1354, 9, !dbg !24
  %compare.low1356 = icmp ugt i32 %compare.low.base1318, %compare.local1355, !dbg !24
  %compare.result1357 = select i1 %compare.high.equal1316, i1 %compare.low1356, i1 %compare.high.order1317, !dbg !24
  %757 = select i1 %compare.result1357, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1358 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1359 = or i32 %compare.local.base1358, 10, !dbg !24
  %compare.low1360 = icmp ugt i32 %compare.low.base1318, %compare.local1359, !dbg !24
  %compare.result1361 = select i1 %compare.high.equal1316, i1 %compare.low1360, i1 %compare.high.order1317, !dbg !24
  %758 = select i1 %compare.result1361, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1362 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1363 = or i32 %compare.local.base1362, 11, !dbg !24
  %compare.low1364 = icmp ugt i32 %compare.low.base1318, %compare.local1363, !dbg !24
  %compare.result1365 = select i1 %compare.high.equal1316, i1 %compare.low1364, i1 %compare.high.order1317, !dbg !24
  %759 = select i1 %compare.result1365, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1366 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1367 = or i32 %compare.local.base1366, 12, !dbg !24
  %compare.low1368 = icmp ugt i32 %compare.low.base1318, %compare.local1367, !dbg !24
  %compare.result1369 = select i1 %compare.high.equal1316, i1 %compare.low1368, i1 %compare.high.order1317, !dbg !24
  %760 = select i1 %compare.result1369, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1370 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1371 = or i32 %compare.local.base1370, 13, !dbg !24
  %compare.low1372 = icmp ugt i32 %compare.low.base1318, %compare.local1371, !dbg !24
  %compare.result1373 = select i1 %compare.high.equal1316, i1 %compare.low1372, i1 %compare.high.order1317, !dbg !24
  %761 = select i1 %compare.result1373, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1374 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1375 = or i32 %compare.local.base1374, 14, !dbg !24
  %compare.low1376 = icmp ugt i32 %compare.low.base1318, %compare.local1375, !dbg !24
  %compare.result1377 = select i1 %compare.high.equal1316, i1 %compare.low1376, i1 %compare.high.order1317, !dbg !24
  %762 = select i1 %compare.result1377, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1378 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1379 = or i32 %compare.local.base1378, 15, !dbg !24
  %compare.low1380 = icmp ugt i32 %compare.low.base1318, %compare.local1379, !dbg !24
  %compare.result1381 = select i1 %compare.high.equal1316, i1 %compare.low1380, i1 %compare.high.order1317, !dbg !24
  %763 = select i1 %compare.result1381, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1382 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1383 = or i32 %compare.local.base1382, 64, !dbg !24
  %compare.low1384 = icmp ugt i32 %compare.low.base1318, %compare.local1383, !dbg !24
  %compare.result1385 = select i1 %compare.high.equal1316, i1 %compare.low1384, i1 %compare.high.order1317, !dbg !24
  %764 = select i1 %compare.result1385, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1386 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1387 = or i32 %compare.local.base1386, 65, !dbg !24
  %compare.low1388 = icmp ugt i32 %compare.low.base1318, %compare.local1387, !dbg !24
  %compare.result1389 = select i1 %compare.high.equal1316, i1 %compare.low1388, i1 %compare.high.order1317, !dbg !24
  %765 = select i1 %compare.result1389, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1390 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1391 = or i32 %compare.local.base1390, 66, !dbg !24
  %compare.low1392 = icmp ugt i32 %compare.low.base1318, %compare.local1391, !dbg !24
  %compare.result1393 = select i1 %compare.high.equal1316, i1 %compare.low1392, i1 %compare.high.order1317, !dbg !24
  %766 = select i1 %compare.result1393, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1394 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1395 = or i32 %compare.local.base1394, 67, !dbg !24
  %compare.low1396 = icmp ugt i32 %compare.low.base1318, %compare.local1395, !dbg !24
  %compare.result1397 = select i1 %compare.high.equal1316, i1 %compare.low1396, i1 %compare.high.order1317, !dbg !24
  %767 = select i1 %compare.result1397, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1398 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1399 = or i32 %compare.local.base1398, 68, !dbg !24
  %compare.low1400 = icmp ugt i32 %compare.low.base1318, %compare.local1399, !dbg !24
  %compare.result1401 = select i1 %compare.high.equal1316, i1 %compare.low1400, i1 %compare.high.order1317, !dbg !24
  %768 = select i1 %compare.result1401, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1402 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1403 = or i32 %compare.local.base1402, 69, !dbg !24
  %compare.low1404 = icmp ugt i32 %compare.low.base1318, %compare.local1403, !dbg !24
  %compare.result1405 = select i1 %compare.high.equal1316, i1 %compare.low1404, i1 %compare.high.order1317, !dbg !24
  %769 = select i1 %compare.result1405, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1406 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1407 = or i32 %compare.local.base1406, 70, !dbg !24
  %compare.low1408 = icmp ugt i32 %compare.low.base1318, %compare.local1407, !dbg !24
  %compare.result1409 = select i1 %compare.high.equal1316, i1 %compare.low1408, i1 %compare.high.order1317, !dbg !24
  %770 = select i1 %compare.result1409, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1410 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1411 = or i32 %compare.local.base1410, 71, !dbg !24
  %compare.low1412 = icmp ugt i32 %compare.low.base1318, %compare.local1411, !dbg !24
  %compare.result1413 = select i1 %compare.high.equal1316, i1 %compare.low1412, i1 %compare.high.order1317, !dbg !24
  %771 = select i1 %compare.result1413, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1414 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1415 = or i32 %compare.local.base1414, 72, !dbg !24
  %compare.low1416 = icmp ugt i32 %compare.low.base1318, %compare.local1415, !dbg !24
  %compare.result1417 = select i1 %compare.high.equal1316, i1 %compare.low1416, i1 %compare.high.order1317, !dbg !24
  %772 = select i1 %compare.result1417, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1418 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1419 = or i32 %compare.local.base1418, 73, !dbg !24
  %compare.low1420 = icmp ugt i32 %compare.low.base1318, %compare.local1419, !dbg !24
  %compare.result1421 = select i1 %compare.high.equal1316, i1 %compare.low1420, i1 %compare.high.order1317, !dbg !24
  %773 = select i1 %compare.result1421, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1422 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1423 = or i32 %compare.local.base1422, 74, !dbg !24
  %compare.low1424 = icmp ugt i32 %compare.low.base1318, %compare.local1423, !dbg !24
  %compare.result1425 = select i1 %compare.high.equal1316, i1 %compare.low1424, i1 %compare.high.order1317, !dbg !24
  %774 = select i1 %compare.result1425, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1426 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1427 = or i32 %compare.local.base1426, 75, !dbg !24
  %compare.low1428 = icmp ugt i32 %compare.low.base1318, %compare.local1427, !dbg !24
  %compare.result1429 = select i1 %compare.high.equal1316, i1 %compare.low1428, i1 %compare.high.order1317, !dbg !24
  %775 = select i1 %compare.result1429, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1430 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1431 = or i32 %compare.local.base1430, 76, !dbg !24
  %compare.low1432 = icmp ugt i32 %compare.low.base1318, %compare.local1431, !dbg !24
  %compare.result1433 = select i1 %compare.high.equal1316, i1 %compare.low1432, i1 %compare.high.order1317, !dbg !24
  %776 = select i1 %compare.result1433, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1434 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1435 = or i32 %compare.local.base1434, 77, !dbg !24
  %compare.low1436 = icmp ugt i32 %compare.low.base1318, %compare.local1435, !dbg !24
  %compare.result1437 = select i1 %compare.high.equal1316, i1 %compare.low1436, i1 %compare.high.order1317, !dbg !24
  %777 = select i1 %compare.result1437, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1438 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1439 = or i32 %compare.local.base1438, 78, !dbg !24
  %compare.low1440 = icmp ugt i32 %compare.low.base1318, %compare.local1439, !dbg !24
  %compare.result1441 = select i1 %compare.high.equal1316, i1 %compare.low1440, i1 %compare.high.order1317, !dbg !24
  %778 = select i1 %compare.result1441, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1442 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1319, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1443 = or i32 %compare.local.base1442, 79, !dbg !24
  %compare.low1444 = icmp ugt i32 %compare.low.base1318, %compare.local1443, !dbg !24
  %compare.result1445 = select i1 %compare.high.equal1316, i1 %compare.low1444, i1 %compare.high.order1317, !dbg !24
  %779 = select i1 %compare.result1445, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %780 = insertelement <4 x float> poison, float %713, i64 0, !dbg !13
  %781 = insertelement <4 x float> %780, float %714, i64 1, !dbg !13
  %782 = insertelement <4 x float> %781, float %715, i64 2, !dbg !13
  %.masked126 = insertelement <4 x float> %782, float %716, i64 3, !dbg !13
  %783 = insertelement <4 x float> poison, float %717, i64 0, !dbg !13
  %784 = insertelement <4 x float> %783, float %718, i64 1, !dbg !13
  %785 = insertelement <4 x float> %784, float %719, i64 2, !dbg !13
  %.masked128 = insertelement <4 x float> %785, float %720, i64 3, !dbg !13
  %786 = insertelement <4 x float> poison, float %721, i64 0, !dbg !13
  %787 = insertelement <4 x float> %786, float %722, i64 1, !dbg !13
  %788 = insertelement <4 x float> %787, float %723, i64 2, !dbg !13
  %.masked130 = insertelement <4 x float> %788, float %724, i64 3, !dbg !13
  %789 = insertelement <4 x float> poison, float %725, i64 0, !dbg !13
  %790 = insertelement <4 x float> %789, float %726, i64 1, !dbg !13
  %791 = insertelement <4 x float> %790, float %727, i64 2, !dbg !13
  %.masked132 = insertelement <4 x float> %791, float %728, i64 3, !dbg !13
  %792 = insertelement <4 x float> poison, float %729, i64 0, !dbg !13
  %793 = insertelement <4 x float> %792, float %730, i64 1, !dbg !13
  %794 = insertelement <4 x float> %793, float %731, i64 2, !dbg !13
  %.masked134 = insertelement <4 x float> %794, float %732, i64 3, !dbg !13
  %795 = insertelement <4 x float> poison, float %733, i64 0, !dbg !13
  %796 = insertelement <4 x float> %795, float %734, i64 1, !dbg !13
  %797 = insertelement <4 x float> %796, float %735, i64 2, !dbg !13
  %.masked136 = insertelement <4 x float> %797, float %736, i64 3, !dbg !13
  %798 = insertelement <4 x float> poison, float %737, i64 0, !dbg !13
  %799 = insertelement <4 x float> %798, float %738, i64 1, !dbg !13
  %800 = insertelement <4 x float> %799, float %739, i64 2, !dbg !13
  %.masked138 = insertelement <4 x float> %800, float %740, i64 3, !dbg !13
  %801 = insertelement <4 x float> poison, float %741, i64 0, !dbg !13
  %802 = insertelement <4 x float> %801, float %742, i64 1, !dbg !13
  %803 = insertelement <4 x float> %802, float %743, i64 2, !dbg !13
  %.masked140 = insertelement <4 x float> %803, float %744, i64 3, !dbg !13
  %804 = insertelement <4 x float> poison, float %748, i64 0, !dbg !13
  %805 = insertelement <4 x float> %804, float %749, i64 1, !dbg !13
  %806 = insertelement <4 x float> %805, float %750, i64 2, !dbg !13
  %.masked143 = insertelement <4 x float> %806, float %751, i64 3, !dbg !13
  %807 = insertelement <4 x float> poison, float %752, i64 0, !dbg !13
  %808 = insertelement <4 x float> %807, float %753, i64 1, !dbg !13
  %809 = insertelement <4 x float> %808, float %754, i64 2, !dbg !13
  %.masked145 = insertelement <4 x float> %809, float %755, i64 3, !dbg !13
  %810 = insertelement <4 x float> poison, float %756, i64 0, !dbg !13
  %811 = insertelement <4 x float> %810, float %757, i64 1, !dbg !13
  %812 = insertelement <4 x float> %811, float %758, i64 2, !dbg !13
  %.masked147 = insertelement <4 x float> %812, float %759, i64 3, !dbg !13
  %813 = insertelement <4 x float> poison, float %760, i64 0, !dbg !13
  %814 = insertelement <4 x float> %813, float %761, i64 1, !dbg !13
  %815 = insertelement <4 x float> %814, float %762, i64 2, !dbg !13
  %.masked149 = insertelement <4 x float> %815, float %763, i64 3, !dbg !13
  %816 = insertelement <4 x float> poison, float %764, i64 0, !dbg !13
  %817 = insertelement <4 x float> %816, float %765, i64 1, !dbg !13
  %818 = insertelement <4 x float> %817, float %766, i64 2, !dbg !13
  %.masked151 = insertelement <4 x float> %818, float %767, i64 3, !dbg !13
  %819 = insertelement <4 x float> poison, float %768, i64 0, !dbg !13
  %820 = insertelement <4 x float> %819, float %769, i64 1, !dbg !13
  %821 = insertelement <4 x float> %820, float %770, i64 2, !dbg !13
  %.masked153 = insertelement <4 x float> %821, float %771, i64 3, !dbg !13
  %822 = insertelement <4 x float> poison, float %772, i64 0, !dbg !13
  %823 = insertelement <4 x float> %822, float %773, i64 1, !dbg !13
  %824 = insertelement <4 x float> %823, float %774, i64 2, !dbg !13
  %.masked155 = insertelement <4 x float> %824, float %775, i64 3, !dbg !13
  %825 = insertelement <4 x float> poison, float %776, i64 0, !dbg !13
  %826 = insertelement <4 x float> %825, float %777, i64 1, !dbg !13
  %827 = insertelement <4 x float> %826, float %778, i64 2, !dbg !13
  %.masked157 = insertelement <4 x float> %827, float %779, i64 3, !dbg !13
  %.masked127 = fadd <4 x float> %638, %.masked126, !dbg !26
  %.masked129 = fadd <4 x float> %642, %.masked128, !dbg !26
  %.masked131 = fadd <4 x float> %646, %.masked130, !dbg !26
  %.masked133 = fadd <4 x float> %650, %.masked132, !dbg !26
  %.masked135 = fadd <4 x float> %654, %.masked134, !dbg !26
  %.masked137 = fadd <4 x float> %658, %.masked136, !dbg !26
  %.masked139 = fadd <4 x float> %662, %.masked138, !dbg !26
  %.masked141 = fadd <4 x float> %666, %.masked140, !dbg !26
  %.masked144 = fadd <4 x float> %670, %.masked143, !dbg !26
  %.masked146 = fadd <4 x float> %674, %.masked145, !dbg !26
  %.masked148 = fadd <4 x float> %678, %.masked147, !dbg !26
  %.masked150 = fadd <4 x float> %682, %.masked149, !dbg !26
  %.masked152 = fadd <4 x float> %686, %.masked151, !dbg !26
  %.masked154 = fadd <4 x float> %690, %.masked153, !dbg !26
  %.masked156 = fadd <4 x float> %694, %.masked155, !dbg !26
  %.masked158 = fadd <4 x float> %699, %.masked157, !dbg !26
  br label %828, !dbg !13

828:                                              ; preds = %701, %702
  %829 = phi <4 x float> [ %.masked127, %702 ], [ %638, %701 ], !dbg !26
  %830 = phi <4 x float> [ %.masked129, %702 ], [ %642, %701 ], !dbg !26
  %831 = phi <4 x float> [ %.masked131, %702 ], [ %646, %701 ], !dbg !26
  %832 = phi <4 x float> [ %.masked133, %702 ], [ %650, %701 ], !dbg !26
  %833 = phi <4 x float> [ %.masked135, %702 ], [ %654, %701 ], !dbg !26
  %834 = phi <4 x float> [ %.masked137, %702 ], [ %658, %701 ], !dbg !26
  %835 = phi <4 x float> [ %.masked139, %702 ], [ %662, %701 ], !dbg !26
  %836 = phi <4 x float> [ %.masked141, %702 ], [ %666, %701 ], !dbg !26
  %837 = phi <4 x float> [ %.masked144, %702 ], [ %670, %701 ], !dbg !26
  %838 = phi <4 x float> [ %.masked146, %702 ], [ %674, %701 ], !dbg !26
  %839 = phi <4 x float> [ %.masked148, %702 ], [ %678, %701 ], !dbg !26
  %840 = phi <4 x float> [ %.masked150, %702 ], [ %682, %701 ], !dbg !26
  %841 = phi <4 x float> [ %.masked152, %702 ], [ %686, %701 ], !dbg !26
  %842 = phi <4 x float> [ %.masked154, %702 ], [ %690, %701 ], !dbg !26
  %843 = phi <4 x float> [ %.masked156, %702 ], [ %694, %701 ], !dbg !26
  %844 = phi <4 x float> [ %.masked158, %702 ], [ %699, %701 ], !dbg !26
  br label %845, !dbg !13

845:                                              ; preds = %828
  %846 = extractelement <4 x float> %829, i64 0, !dbg !26
  %847 = extractelement <4 x float> %829, i64 1, !dbg !26
  %848 = extractelement <4 x float> %829, i64 2, !dbg !26
  %849 = extractelement <4 x float> %829, i64 3, !dbg !26
  %850 = extractelement <4 x float> %830, i64 0, !dbg !26
  %851 = extractelement <4 x float> %830, i64 1, !dbg !26
  %852 = extractelement <4 x float> %830, i64 2, !dbg !26
  %853 = extractelement <4 x float> %830, i64 3, !dbg !26
  %854 = extractelement <4 x float> %831, i64 0, !dbg !26
  %855 = extractelement <4 x float> %831, i64 1, !dbg !26
  %856 = extractelement <4 x float> %831, i64 2, !dbg !26
  %857 = extractelement <4 x float> %831, i64 3, !dbg !26
  %858 = extractelement <4 x float> %832, i64 0, !dbg !26
  %859 = extractelement <4 x float> %832, i64 1, !dbg !26
  %860 = extractelement <4 x float> %832, i64 2, !dbg !26
  %861 = extractelement <4 x float> %832, i64 3, !dbg !26
  %862 = extractelement <4 x float> %833, i64 0, !dbg !26
  %863 = extractelement <4 x float> %833, i64 1, !dbg !26
  %864 = extractelement <4 x float> %833, i64 2, !dbg !26
  %865 = extractelement <4 x float> %833, i64 3, !dbg !26
  %866 = extractelement <4 x float> %834, i64 0, !dbg !26
  %867 = extractelement <4 x float> %834, i64 1, !dbg !26
  %868 = extractelement <4 x float> %834, i64 2, !dbg !26
  %869 = extractelement <4 x float> %834, i64 3, !dbg !26
  %870 = extractelement <4 x float> %835, i64 0, !dbg !26
  %871 = extractelement <4 x float> %835, i64 1, !dbg !26
  %872 = extractelement <4 x float> %835, i64 2, !dbg !26
  %873 = extractelement <4 x float> %835, i64 3, !dbg !26
  %874 = extractelement <4 x float> %836, i64 0, !dbg !26
  %875 = extractelement <4 x float> %836, i64 1, !dbg !26
  %876 = extractelement <4 x float> %836, i64 2, !dbg !26
  %877 = extractelement <4 x float> %836, i64 3, !dbg !26
  %878 = extractelement <4 x float> %837, i64 0, !dbg !26
  %879 = extractelement <4 x float> %837, i64 1, !dbg !26
  %880 = extractelement <4 x float> %837, i64 2, !dbg !26
  %881 = extractelement <4 x float> %837, i64 3, !dbg !26
  %882 = extractelement <4 x float> %838, i64 0, !dbg !26
  %883 = extractelement <4 x float> %838, i64 1, !dbg !26
  %884 = extractelement <4 x float> %838, i64 2, !dbg !26
  %885 = extractelement <4 x float> %838, i64 3, !dbg !26
  %886 = extractelement <4 x float> %839, i64 0, !dbg !26
  %887 = extractelement <4 x float> %839, i64 1, !dbg !26
  %888 = extractelement <4 x float> %839, i64 2, !dbg !26
  %889 = extractelement <4 x float> %839, i64 3, !dbg !26
  %890 = extractelement <4 x float> %840, i64 0, !dbg !26
  %891 = extractelement <4 x float> %840, i64 1, !dbg !26
  %892 = extractelement <4 x float> %840, i64 2, !dbg !26
  %893 = extractelement <4 x float> %840, i64 3, !dbg !26
  %894 = extractelement <4 x float> %841, i64 0, !dbg !26
  %895 = extractelement <4 x float> %841, i64 1, !dbg !26
  %896 = extractelement <4 x float> %841, i64 2, !dbg !26
  %897 = extractelement <4 x float> %841, i64 3, !dbg !26
  %898 = extractelement <4 x float> %842, i64 0, !dbg !26
  %899 = extractelement <4 x float> %842, i64 1, !dbg !26
  %900 = extractelement <4 x float> %842, i64 2, !dbg !26
  %901 = extractelement <4 x float> %842, i64 3, !dbg !26
  %902 = extractelement <4 x float> %843, i64 0, !dbg !26
  %903 = extractelement <4 x float> %843, i64 1, !dbg !26
  %904 = extractelement <4 x float> %843, i64 2, !dbg !26
  %905 = extractelement <4 x float> %843, i64 3, !dbg !26
  %906 = extractelement <4 x float> %844, i64 0, !dbg !26
  %907 = extractelement <4 x float> %844, i64 1, !dbg !26
  %908 = extractelement <4 x float> %844, i64 2, !dbg !26
  %909 = extractelement <4 x float> %844, i64 3, !dbg !26
  %910 = call float @llvm.maxnum.f32(float %846, float 0xFFF0000000000000), !dbg !27
  %911 = call float @llvm.maxnum.f32(float %910, float %847), !dbg !27
  %912 = call float @llvm.maxnum.f32(float %911, float %848), !dbg !27
  %913 = call float @llvm.maxnum.f32(float %912, float %849), !dbg !27
  %914 = call float @llvm.maxnum.f32(float %913, float %850), !dbg !27
  %915 = call float @llvm.maxnum.f32(float %914, float %851), !dbg !27
  %916 = call float @llvm.maxnum.f32(float %915, float %852), !dbg !27
  %917 = call float @llvm.maxnum.f32(float %916, float %853), !dbg !27
  %918 = call float @llvm.maxnum.f32(float %917, float %854), !dbg !27
  %919 = call float @llvm.maxnum.f32(float %918, float %855), !dbg !27
  %920 = call float @llvm.maxnum.f32(float %919, float %856), !dbg !27
  %921 = call float @llvm.maxnum.f32(float %920, float %857), !dbg !27
  %922 = call float @llvm.maxnum.f32(float %921, float %858), !dbg !27
  %923 = call float @llvm.maxnum.f32(float %922, float %859), !dbg !27
  %924 = call float @llvm.maxnum.f32(float %923, float %860), !dbg !27
  %925 = call float @llvm.maxnum.f32(float %924, float %861), !dbg !27
  %926 = call float @llvm.maxnum.f32(float %925, float %862), !dbg !27
  %927 = call float @llvm.maxnum.f32(float %926, float %863), !dbg !27
  %928 = call float @llvm.maxnum.f32(float %927, float %864), !dbg !27
  %929 = call float @llvm.maxnum.f32(float %928, float %865), !dbg !27
  %930 = call float @llvm.maxnum.f32(float %929, float %866), !dbg !27
  %931 = call float @llvm.maxnum.f32(float %930, float %867), !dbg !27
  %932 = call float @llvm.maxnum.f32(float %931, float %868), !dbg !27
  %933 = call float @llvm.maxnum.f32(float %932, float %869), !dbg !27
  %934 = call float @llvm.maxnum.f32(float %933, float %870), !dbg !27
  %935 = call float @llvm.maxnum.f32(float %934, float %871), !dbg !27
  %936 = call float @llvm.maxnum.f32(float %935, float %872), !dbg !27
  %937 = call float @llvm.maxnum.f32(float %936, float %873), !dbg !27
  %938 = call float @llvm.maxnum.f32(float %937, float %874), !dbg !27
  %939 = call float @llvm.maxnum.f32(float %938, float %875), !dbg !27
  %940 = call float @llvm.maxnum.f32(float %939, float %876), !dbg !27
  %941 = call float @llvm.maxnum.f32(float %940, float %877), !dbg !27
  %942 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %943 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %942), !dbg !27
  %944 = add nuw nsw i32 %943, 64, !dbg !27
  %945 = xor i32 %943, 16, !dbg !27
  %946 = and i32 %944, 192, !dbg !27
  %947 = icmp samesign ult i32 %945, %946, !dbg !27
  %948 = select i1 %947, i32 %945, i32 %943, !dbg !27
  %949 = shl nuw nsw i32 %948, 2, !dbg !27
  %950 = bitcast float %941 to i32, !dbg !27
  %951 = call i32 @llvm.amdgcn.ds.bpermute(i32 %949, i32 %950), !dbg !27
  %952 = bitcast i32 %951 to float, !dbg !27
  %953 = call float @llvm.maxnum.f32(float %941, float %952), !dbg !27
  %954 = xor i32 %943, 32, !dbg !27
  %955 = icmp samesign ult i32 %954, %946, !dbg !27
  %956 = select i1 %955, i32 %954, i32 %943, !dbg !27
  %957 = shl nuw nsw i32 %956, 2, !dbg !27
  %958 = bitcast float %953 to i32, !dbg !27
  %959 = call i32 @llvm.amdgcn.ds.bpermute(i32 %957, i32 %958), !dbg !27
  %960 = bitcast i32 %959 to float, !dbg !27
  %961 = call float @llvm.maxnum.f32(float %953, float %960), !dbg !27
  %962 = extractvalue [2 x <1 x float>] %208, 0, !dbg !27
  %963 = insertelement <1 x float> poison, float %961, i64 0, !dbg !27
  %964 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %963, <1 x float> %962), !dbg !27
  %965 = call float @llvm.maxnum.f32(float %878, float 0xFFF0000000000000), !dbg !27
  %966 = call float @llvm.maxnum.f32(float %965, float %879), !dbg !27
  %967 = call float @llvm.maxnum.f32(float %966, float %880), !dbg !27
  %968 = call float @llvm.maxnum.f32(float %967, float %881), !dbg !27
  %969 = call float @llvm.maxnum.f32(float %968, float %882), !dbg !27
  %970 = call float @llvm.maxnum.f32(float %969, float %883), !dbg !27
  %971 = call float @llvm.maxnum.f32(float %970, float %884), !dbg !27
  %972 = call float @llvm.maxnum.f32(float %971, float %885), !dbg !27
  %973 = call float @llvm.maxnum.f32(float %972, float %886), !dbg !27
  %974 = call float @llvm.maxnum.f32(float %973, float %887), !dbg !27
  %975 = call float @llvm.maxnum.f32(float %974, float %888), !dbg !27
  %976 = call float @llvm.maxnum.f32(float %975, float %889), !dbg !27
  %977 = call float @llvm.maxnum.f32(float %976, float %890), !dbg !27
  %978 = call float @llvm.maxnum.f32(float %977, float %891), !dbg !27
  %979 = call float @llvm.maxnum.f32(float %978, float %892), !dbg !27
  %980 = call float @llvm.maxnum.f32(float %979, float %893), !dbg !27
  %981 = call float @llvm.maxnum.f32(float %980, float %894), !dbg !27
  %982 = call float @llvm.maxnum.f32(float %981, float %895), !dbg !27
  %983 = call float @llvm.maxnum.f32(float %982, float %896), !dbg !27
  %984 = call float @llvm.maxnum.f32(float %983, float %897), !dbg !27
  %985 = call float @llvm.maxnum.f32(float %984, float %898), !dbg !27
  %986 = call float @llvm.maxnum.f32(float %985, float %899), !dbg !27
  %987 = call float @llvm.maxnum.f32(float %986, float %900), !dbg !27
  %988 = call float @llvm.maxnum.f32(float %987, float %901), !dbg !27
  %989 = call float @llvm.maxnum.f32(float %988, float %902), !dbg !27
  %990 = call float @llvm.maxnum.f32(float %989, float %903), !dbg !27
  %991 = call float @llvm.maxnum.f32(float %990, float %904), !dbg !27
  %992 = call float @llvm.maxnum.f32(float %991, float %905), !dbg !27
  %993 = call float @llvm.maxnum.f32(float %992, float %906), !dbg !27
  %994 = call float @llvm.maxnum.f32(float %993, float %907), !dbg !27
  %995 = call float @llvm.maxnum.f32(float %994, float %908), !dbg !27
  %996 = call float @llvm.maxnum.f32(float %995, float %909), !dbg !27
  %997 = bitcast float %996 to i32, !dbg !27
  %998 = call i32 @llvm.amdgcn.ds.bpermute(i32 %949, i32 %997), !dbg !27
  %999 = bitcast i32 %998 to float, !dbg !27
  %1000 = call float @llvm.maxnum.f32(float %996, float %999), !dbg !27
  %1001 = bitcast float %1000 to i32, !dbg !27
  %1002 = call i32 @llvm.amdgcn.ds.bpermute(i32 %957, i32 %1001), !dbg !27
  %1003 = bitcast i32 %1002 to float, !dbg !27
  %1004 = call float @llvm.maxnum.f32(float %1000, float %1003), !dbg !27
  %1005 = extractvalue [2 x <1 x float>] %208, 1, !dbg !27
  %1006 = insertelement <1 x float> poison, float %1004, i64 0, !dbg !27
  %1007 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1006, <1 x float> %1005), !dbg !27
  %1008 = shufflevector <1 x float> %964, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1009 = fsub <4 x float> %829, %1008, !dbg !28
  %1010 = fsub <4 x float> %830, %1008, !dbg !28
  %1011 = fsub <4 x float> %831, %1008, !dbg !28
  %1012 = fsub <4 x float> %832, %1008, !dbg !28
  %1013 = fsub <4 x float> %833, %1008, !dbg !28
  %1014 = fsub <4 x float> %834, %1008, !dbg !28
  %1015 = fsub <4 x float> %835, %1008, !dbg !28
  %1016 = fsub <4 x float> %836, %1008, !dbg !28
  %1017 = shufflevector <1 x float> %1007, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1018 = fsub <4 x float> %837, %1017, !dbg !28
  %1019 = fsub <4 x float> %838, %1017, !dbg !28
  %1020 = fsub <4 x float> %839, %1017, !dbg !28
  %1021 = fsub <4 x float> %840, %1017, !dbg !28
  %1022 = fsub <4 x float> %841, %1017, !dbg !28
  %1023 = fsub <4 x float> %842, %1017, !dbg !28
  %1024 = fsub <4 x float> %843, %1017, !dbg !28
  %1025 = fsub <4 x float> %844, %1017, !dbg !28
  %1026 = fsub <1 x float> %962, %964, !dbg !29
  %1027 = extractelement <1 x float> %1026, i64 0, !dbg !30
  %1028 = call float @llvm.amdgcn.exp2.f32(float %1027), !dbg !30
  %1029 = insertelement <4 x float> poison, float %1028, i64 0, !dbg !35
  %1030 = shufflevector <4 x float> %1029, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %1031 = extractvalue [2 x <1 x float>] %207, 0, !dbg !31
  %1032 = extractelement <1 x float> %1031, i64 0, !dbg !31
  %1033 = fmul float %1032, %1028, !dbg !31
  %1034 = fsub <1 x float> %1005, %1007, !dbg !29
  %1035 = extractelement <1 x float> %1034, i64 0, !dbg !30
  %1036 = call float @llvm.amdgcn.exp2.f32(float %1035), !dbg !30
  %1037 = insertelement <4 x float> poison, float %1036, i64 0, !dbg !35
  %1038 = shufflevector <4 x float> %1037, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %1039 = extractvalue [2 x <1 x float>] %207, 1, !dbg !31
  %1040 = extractelement <1 x float> %1039, i64 0, !dbg !31
  %1041 = fmul float %1040, %1036, !dbg !31
  %1042 = extractelement <4 x float> %1009, i64 0, !dbg !32
  %1043 = call float @llvm.amdgcn.exp2.f32(float %1042), !dbg !32
  %1044 = insertelement <4 x float> poison, float %1043, i64 0, !dbg !34
  %1045 = extractelement <4 x float> %1009, i64 1, !dbg !32
  %1046 = call float @llvm.amdgcn.exp2.f32(float %1045), !dbg !32
  %1047 = insertelement <4 x float> %1044, float %1046, i64 1, !dbg !34
  %1048 = fadd float %1043, %1046, !dbg !33
  %1049 = extractelement <4 x float> %1009, i64 2, !dbg !32
  %1050 = call float @llvm.amdgcn.exp2.f32(float %1049), !dbg !32
  %1051 = insertelement <4 x float> %1047, float %1050, i64 2, !dbg !34
  %1052 = fadd float %1048, %1050, !dbg !33
  %1053 = extractelement <4 x float> %1009, i64 3, !dbg !32
  %1054 = call float @llvm.amdgcn.exp2.f32(float %1053), !dbg !32
  %1055 = insertelement <4 x float> %1051, float %1054, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1055) #9, !dbg !34, !frisk.register_lifetimes !1
  %1056 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1056) #9, !dbg !33, !frisk.register_lifetimes !1
  %1057 = fadd float %1052, %1054, !dbg !33
  %1058 = extractelement <4 x float> %1010, i64 0, !dbg !32
  %1059 = call float @llvm.amdgcn.exp2.f32(float %1058), !dbg !32
  %1060 = insertelement <4 x float> poison, float %1059, i64 0, !dbg !34
  %1061 = fadd float %1057, %1059, !dbg !33
  %1062 = extractelement <4 x float> %1010, i64 1, !dbg !32
  %1063 = call float @llvm.amdgcn.exp2.f32(float %1062), !dbg !32
  %1064 = insertelement <4 x float> %1060, float %1063, i64 1, !dbg !34
  %1065 = fadd float %1061, %1063, !dbg !33
  %1066 = extractelement <4 x float> %1010, i64 2, !dbg !32
  %1067 = call float @llvm.amdgcn.exp2.f32(float %1066), !dbg !32
  %1068 = insertelement <4 x float> %1064, float %1067, i64 2, !dbg !34
  %1069 = fadd float %1065, %1067, !dbg !33
  %1070 = extractelement <4 x float> %1010, i64 3, !dbg !32
  %1071 = call float @llvm.amdgcn.exp2.f32(float %1070), !dbg !32
  %1072 = insertelement <4 x float> %1068, float %1071, i64 3, !dbg !34
  %math.result342 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1072) #9, !dbg !34, !frisk.register_lifetimes !1
  %1073 = fptrunc <4 x float> %math.result342 to <4 x half>, !dbg !34
  %math.packed343 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1073) #9, !dbg !33, !frisk.register_lifetimes !1
  %1074 = fadd float %1069, %1071, !dbg !33
  %1075 = extractelement <4 x float> %1011, i64 0, !dbg !32
  %1076 = call float @llvm.amdgcn.exp2.f32(float %1075), !dbg !32
  %1077 = insertelement <4 x float> poison, float %1076, i64 0, !dbg !34
  %1078 = fadd float %1074, %1076, !dbg !33
  %1079 = extractelement <4 x float> %1011, i64 1, !dbg !32
  %1080 = call float @llvm.amdgcn.exp2.f32(float %1079), !dbg !32
  %1081 = insertelement <4 x float> %1077, float %1080, i64 1, !dbg !34
  %1082 = fadd float %1078, %1080, !dbg !33
  %1083 = extractelement <4 x float> %1011, i64 2, !dbg !32
  %1084 = call float @llvm.amdgcn.exp2.f32(float %1083), !dbg !32
  %1085 = insertelement <4 x float> %1081, float %1084, i64 2, !dbg !34
  %1086 = fadd float %1082, %1084, !dbg !33
  %1087 = extractelement <4 x float> %1011, i64 3, !dbg !32
  %1088 = call float @llvm.amdgcn.exp2.f32(float %1087), !dbg !32
  %1089 = insertelement <4 x float> %1085, float %1088, i64 3, !dbg !34
  %math.result344 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1089) #9, !dbg !34, !frisk.register_lifetimes !1
  %1090 = fptrunc <4 x float> %math.result344 to <4 x half>, !dbg !34
  %math.packed345 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1090) #9, !dbg !33, !frisk.register_lifetimes !1
  %1091 = fadd float %1086, %1088, !dbg !33
  %1092 = extractelement <4 x float> %1012, i64 0, !dbg !32
  %1093 = call float @llvm.amdgcn.exp2.f32(float %1092), !dbg !32
  %1094 = insertelement <4 x float> poison, float %1093, i64 0, !dbg !34
  %1095 = fadd float %1091, %1093, !dbg !33
  %1096 = extractelement <4 x float> %1012, i64 1, !dbg !32
  %1097 = call float @llvm.amdgcn.exp2.f32(float %1096), !dbg !32
  %1098 = insertelement <4 x float> %1094, float %1097, i64 1, !dbg !34
  %1099 = fadd float %1095, %1097, !dbg !33
  %1100 = extractelement <4 x float> %1012, i64 2, !dbg !32
  %1101 = call float @llvm.amdgcn.exp2.f32(float %1100), !dbg !32
  %1102 = insertelement <4 x float> %1098, float %1101, i64 2, !dbg !34
  %1103 = fadd float %1099, %1101, !dbg !33
  %1104 = extractelement <4 x float> %1012, i64 3, !dbg !32
  %1105 = call float @llvm.amdgcn.exp2.f32(float %1104), !dbg !32
  %1106 = insertelement <4 x float> %1102, float %1105, i64 3, !dbg !34
  %math.result346 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1106) #9, !dbg !34, !frisk.register_lifetimes !1
  %1107 = fptrunc <4 x float> %math.result346 to <4 x half>, !dbg !34
  %math.packed347 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1107) #9, !dbg !33, !frisk.register_lifetimes !1
  %1108 = fadd float %1103, %1105, !dbg !33
  %1109 = extractelement <4 x float> %1013, i64 0, !dbg !32
  %1110 = call float @llvm.amdgcn.exp2.f32(float %1109), !dbg !32
  %1111 = insertelement <4 x float> poison, float %1110, i64 0, !dbg !34
  %1112 = fadd float %1108, %1110, !dbg !33
  %1113 = extractelement <4 x float> %1013, i64 1, !dbg !32
  %1114 = call float @llvm.amdgcn.exp2.f32(float %1113), !dbg !32
  %1115 = insertelement <4 x float> %1111, float %1114, i64 1, !dbg !34
  %1116 = fadd float %1112, %1114, !dbg !33
  %1117 = extractelement <4 x float> %1013, i64 2, !dbg !32
  %1118 = call float @llvm.amdgcn.exp2.f32(float %1117), !dbg !32
  %1119 = insertelement <4 x float> %1115, float %1118, i64 2, !dbg !34
  %1120 = fadd float %1116, %1118, !dbg !33
  %1121 = extractelement <4 x float> %1013, i64 3, !dbg !32
  %1122 = call float @llvm.amdgcn.exp2.f32(float %1121), !dbg !32
  %1123 = insertelement <4 x float> %1119, float %1122, i64 3, !dbg !34
  %math.result348 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1123) #9, !dbg !34, !frisk.register_lifetimes !1
  %1124 = fptrunc <4 x float> %math.result348 to <4 x half>, !dbg !34
  %math.packed349 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1124) #9, !dbg !33, !frisk.register_lifetimes !1
  %1125 = fadd float %1120, %1122, !dbg !33
  %1126 = extractelement <4 x float> %1014, i64 0, !dbg !32
  %1127 = call float @llvm.amdgcn.exp2.f32(float %1126), !dbg !32
  %1128 = insertelement <4 x float> poison, float %1127, i64 0, !dbg !34
  %1129 = fadd float %1125, %1127, !dbg !33
  %1130 = extractelement <4 x float> %1014, i64 1, !dbg !32
  %1131 = call float @llvm.amdgcn.exp2.f32(float %1130), !dbg !32
  %1132 = insertelement <4 x float> %1128, float %1131, i64 1, !dbg !34
  %1133 = fadd float %1129, %1131, !dbg !33
  %1134 = extractelement <4 x float> %1014, i64 2, !dbg !32
  %1135 = call float @llvm.amdgcn.exp2.f32(float %1134), !dbg !32
  %1136 = insertelement <4 x float> %1132, float %1135, i64 2, !dbg !34
  %1137 = fadd float %1133, %1135, !dbg !33
  %1138 = extractelement <4 x float> %1014, i64 3, !dbg !32
  %1139 = call float @llvm.amdgcn.exp2.f32(float %1138), !dbg !32
  %1140 = insertelement <4 x float> %1136, float %1139, i64 3, !dbg !34
  %math.result350 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1140) #9, !dbg !34, !frisk.register_lifetimes !1
  %1141 = fptrunc <4 x float> %math.result350 to <4 x half>, !dbg !34
  %math.packed351 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1141) #9, !dbg !33, !frisk.register_lifetimes !1
  %1142 = fadd float %1137, %1139, !dbg !33
  %1143 = extractelement <4 x float> %1015, i64 0, !dbg !32
  %1144 = call float @llvm.amdgcn.exp2.f32(float %1143), !dbg !32
  %1145 = insertelement <4 x float> poison, float %1144, i64 0, !dbg !34
  %1146 = fadd float %1142, %1144, !dbg !33
  %1147 = extractelement <4 x float> %1015, i64 1, !dbg !32
  %1148 = call float @llvm.amdgcn.exp2.f32(float %1147), !dbg !32
  %1149 = insertelement <4 x float> %1145, float %1148, i64 1, !dbg !34
  %1150 = fadd float %1146, %1148, !dbg !33
  %1151 = extractelement <4 x float> %1015, i64 2, !dbg !32
  %1152 = call float @llvm.amdgcn.exp2.f32(float %1151), !dbg !32
  %1153 = insertelement <4 x float> %1149, float %1152, i64 2, !dbg !34
  %1154 = fadd float %1150, %1152, !dbg !33
  %1155 = extractelement <4 x float> %1015, i64 3, !dbg !32
  %1156 = call float @llvm.amdgcn.exp2.f32(float %1155), !dbg !32
  %1157 = insertelement <4 x float> %1153, float %1156, i64 3, !dbg !34
  %math.result352 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1157) #9, !dbg !34, !frisk.register_lifetimes !1
  %1158 = fptrunc <4 x float> %math.result352 to <4 x half>, !dbg !34
  %math.packed353 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1158) #9, !dbg !33, !frisk.register_lifetimes !1
  %1159 = fadd float %1154, %1156, !dbg !33
  %1160 = extractelement <4 x float> %1016, i64 0, !dbg !32
  %1161 = call float @llvm.amdgcn.exp2.f32(float %1160), !dbg !32
  %1162 = insertelement <4 x float> poison, float %1161, i64 0, !dbg !34
  %1163 = fadd float %1159, %1161, !dbg !33
  %1164 = extractelement <4 x float> %1016, i64 1, !dbg !32
  %1165 = call float @llvm.amdgcn.exp2.f32(float %1164), !dbg !32
  %1166 = insertelement <4 x float> %1162, float %1165, i64 1, !dbg !34
  %1167 = fadd float %1163, %1165, !dbg !33
  %1168 = extractelement <4 x float> %1016, i64 2, !dbg !32
  %1169 = call float @llvm.amdgcn.exp2.f32(float %1168), !dbg !32
  %1170 = insertelement <4 x float> %1166, float %1169, i64 2, !dbg !34
  %1171 = fadd float %1167, %1169, !dbg !33
  %1172 = extractelement <4 x float> %1016, i64 3, !dbg !32
  %1173 = call float @llvm.amdgcn.exp2.f32(float %1172), !dbg !32
  %1174 = insertelement <4 x float> %1170, float %1173, i64 3, !dbg !34
  %math.result354 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1174) #9, !dbg !34, !frisk.register_lifetimes !1
  %1175 = fptrunc <4 x float> %math.result354 to <4 x half>, !dbg !34
  %math.packed355 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1175) #9, !dbg !33, !frisk.register_lifetimes !1
  %1176 = fadd float %1171, %1173, !dbg !33
  %1177 = extractelement <4 x float> %1018, i64 0, !dbg !32
  %1178 = call float @llvm.amdgcn.exp2.f32(float %1177), !dbg !32
  %1179 = insertelement <4 x float> poison, float %1178, i64 0, !dbg !34
  %1180 = extractelement <4 x float> %1018, i64 1, !dbg !32
  %1181 = call float @llvm.amdgcn.exp2.f32(float %1180), !dbg !32
  %1182 = insertelement <4 x float> %1179, float %1181, i64 1, !dbg !34
  %1183 = fadd float %1178, %1181, !dbg !33
  %1184 = extractelement <4 x float> %1018, i64 2, !dbg !32
  %1185 = call float @llvm.amdgcn.exp2.f32(float %1184), !dbg !32
  %1186 = insertelement <4 x float> %1182, float %1185, i64 2, !dbg !34
  %1187 = fadd float %1183, %1185, !dbg !33
  %1188 = extractelement <4 x float> %1018, i64 3, !dbg !32
  %1189 = call float @llvm.amdgcn.exp2.f32(float %1188), !dbg !32
  %1190 = insertelement <4 x float> %1186, float %1189, i64 3, !dbg !34
  %math.result356 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1190) #9, !dbg !34, !frisk.register_lifetimes !1
  %1191 = fptrunc <4 x float> %math.result356 to <4 x half>, !dbg !34
  %math.packed357 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1191) #9, !dbg !33, !frisk.register_lifetimes !1
  %1192 = fadd float %1187, %1189, !dbg !33
  %1193 = extractelement <4 x float> %1019, i64 0, !dbg !32
  %1194 = call float @llvm.amdgcn.exp2.f32(float %1193), !dbg !32
  %1195 = insertelement <4 x float> poison, float %1194, i64 0, !dbg !34
  %1196 = fadd float %1192, %1194, !dbg !33
  %1197 = extractelement <4 x float> %1019, i64 1, !dbg !32
  %1198 = call float @llvm.amdgcn.exp2.f32(float %1197), !dbg !32
  %1199 = insertelement <4 x float> %1195, float %1198, i64 1, !dbg !34
  %1200 = fadd float %1196, %1198, !dbg !33
  %1201 = extractelement <4 x float> %1019, i64 2, !dbg !32
  %1202 = call float @llvm.amdgcn.exp2.f32(float %1201), !dbg !32
  %1203 = insertelement <4 x float> %1199, float %1202, i64 2, !dbg !34
  %1204 = fadd float %1200, %1202, !dbg !33
  %1205 = extractelement <4 x float> %1019, i64 3, !dbg !32
  %1206 = call float @llvm.amdgcn.exp2.f32(float %1205), !dbg !32
  %1207 = insertelement <4 x float> %1203, float %1206, i64 3, !dbg !34
  %math.result358 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1207) #9, !dbg !34, !frisk.register_lifetimes !1
  %1208 = fptrunc <4 x float> %math.result358 to <4 x half>, !dbg !34
  %math.packed359 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1208) #9, !dbg !33, !frisk.register_lifetimes !1
  %1209 = fadd float %1204, %1206, !dbg !33
  %1210 = extractelement <4 x float> %1020, i64 0, !dbg !32
  %1211 = call float @llvm.amdgcn.exp2.f32(float %1210), !dbg !32
  %1212 = insertelement <4 x float> poison, float %1211, i64 0, !dbg !34
  %1213 = fadd float %1209, %1211, !dbg !33
  %1214 = extractelement <4 x float> %1020, i64 1, !dbg !32
  %1215 = call float @llvm.amdgcn.exp2.f32(float %1214), !dbg !32
  %1216 = insertelement <4 x float> %1212, float %1215, i64 1, !dbg !34
  %1217 = fadd float %1213, %1215, !dbg !33
  %1218 = extractelement <4 x float> %1020, i64 2, !dbg !32
  %1219 = call float @llvm.amdgcn.exp2.f32(float %1218), !dbg !32
  %1220 = insertelement <4 x float> %1216, float %1219, i64 2, !dbg !34
  %1221 = fadd float %1217, %1219, !dbg !33
  %1222 = extractelement <4 x float> %1020, i64 3, !dbg !32
  %1223 = call float @llvm.amdgcn.exp2.f32(float %1222), !dbg !32
  %1224 = insertelement <4 x float> %1220, float %1223, i64 3, !dbg !34
  %math.result360 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1224) #9, !dbg !34, !frisk.register_lifetimes !1
  %1225 = fptrunc <4 x float> %math.result360 to <4 x half>, !dbg !34
  %math.packed361 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1225) #9, !dbg !33, !frisk.register_lifetimes !1
  %1226 = fadd float %1221, %1223, !dbg !33
  %1227 = extractelement <4 x float> %1021, i64 0, !dbg !32
  %1228 = call float @llvm.amdgcn.exp2.f32(float %1227), !dbg !32
  %1229 = insertelement <4 x float> poison, float %1228, i64 0, !dbg !34
  %1230 = fadd float %1226, %1228, !dbg !33
  %1231 = extractelement <4 x float> %1021, i64 1, !dbg !32
  %1232 = call float @llvm.amdgcn.exp2.f32(float %1231), !dbg !32
  %1233 = insertelement <4 x float> %1229, float %1232, i64 1, !dbg !34
  %1234 = fadd float %1230, %1232, !dbg !33
  %1235 = extractelement <4 x float> %1021, i64 2, !dbg !32
  %1236 = call float @llvm.amdgcn.exp2.f32(float %1235), !dbg !32
  %1237 = insertelement <4 x float> %1233, float %1236, i64 2, !dbg !34
  %1238 = fadd float %1234, %1236, !dbg !33
  %1239 = extractelement <4 x float> %1021, i64 3, !dbg !32
  %1240 = call float @llvm.amdgcn.exp2.f32(float %1239), !dbg !32
  %1241 = insertelement <4 x float> %1237, float %1240, i64 3, !dbg !34
  %math.result362 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1241) #9, !dbg !34, !frisk.register_lifetimes !1
  %1242 = fptrunc <4 x float> %math.result362 to <4 x half>, !dbg !34
  %math.packed363 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1242) #9, !dbg !33, !frisk.register_lifetimes !1
  %1243 = fadd float %1238, %1240, !dbg !33
  %1244 = extractelement <4 x float> %1022, i64 0, !dbg !32
  %1245 = call float @llvm.amdgcn.exp2.f32(float %1244), !dbg !32
  %1246 = insertelement <4 x float> poison, float %1245, i64 0, !dbg !34
  %1247 = fadd float %1243, %1245, !dbg !33
  %1248 = extractelement <4 x float> %1022, i64 1, !dbg !32
  %1249 = call float @llvm.amdgcn.exp2.f32(float %1248), !dbg !32
  %1250 = insertelement <4 x float> %1246, float %1249, i64 1, !dbg !34
  %1251 = fadd float %1247, %1249, !dbg !33
  %1252 = extractelement <4 x float> %1022, i64 2, !dbg !32
  %1253 = call float @llvm.amdgcn.exp2.f32(float %1252), !dbg !32
  %1254 = insertelement <4 x float> %1250, float %1253, i64 2, !dbg !34
  %1255 = fadd float %1251, %1253, !dbg !33
  %1256 = extractelement <4 x float> %1022, i64 3, !dbg !32
  %1257 = call float @llvm.amdgcn.exp2.f32(float %1256), !dbg !32
  %1258 = insertelement <4 x float> %1254, float %1257, i64 3, !dbg !34
  %math.result364 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1258) #9, !dbg !34, !frisk.register_lifetimes !1
  %1259 = fptrunc <4 x float> %math.result364 to <4 x half>, !dbg !34
  %math.packed365 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1259) #9, !dbg !33, !frisk.register_lifetimes !1
  %1260 = fadd float %1255, %1257, !dbg !33
  %1261 = extractelement <4 x float> %1023, i64 0, !dbg !32
  %1262 = call float @llvm.amdgcn.exp2.f32(float %1261), !dbg !32
  %1263 = insertelement <4 x float> poison, float %1262, i64 0, !dbg !34
  %1264 = fadd float %1260, %1262, !dbg !33
  %1265 = extractelement <4 x float> %1023, i64 1, !dbg !32
  %1266 = call float @llvm.amdgcn.exp2.f32(float %1265), !dbg !32
  %1267 = insertelement <4 x float> %1263, float %1266, i64 1, !dbg !34
  %1268 = fadd float %1264, %1266, !dbg !33
  %1269 = extractelement <4 x float> %1023, i64 2, !dbg !32
  %1270 = call float @llvm.amdgcn.exp2.f32(float %1269), !dbg !32
  %1271 = insertelement <4 x float> %1267, float %1270, i64 2, !dbg !34
  %1272 = fadd float %1268, %1270, !dbg !33
  %1273 = extractelement <4 x float> %1023, i64 3, !dbg !32
  %1274 = call float @llvm.amdgcn.exp2.f32(float %1273), !dbg !32
  %1275 = insertelement <4 x float> %1271, float %1274, i64 3, !dbg !34
  %math.result366 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1275) #9, !dbg !34, !frisk.register_lifetimes !1
  %1276 = fptrunc <4 x float> %math.result366 to <4 x half>, !dbg !34
  %math.packed367 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1276) #9, !dbg !33, !frisk.register_lifetimes !1
  %1277 = fadd float %1272, %1274, !dbg !33
  %1278 = extractelement <4 x float> %1024, i64 0, !dbg !32
  %1279 = call float @llvm.amdgcn.exp2.f32(float %1278), !dbg !32
  %1280 = insertelement <4 x float> poison, float %1279, i64 0, !dbg !34
  %1281 = fadd float %1277, %1279, !dbg !33
  %1282 = extractelement <4 x float> %1024, i64 1, !dbg !32
  %1283 = call float @llvm.amdgcn.exp2.f32(float %1282), !dbg !32
  %1284 = insertelement <4 x float> %1280, float %1283, i64 1, !dbg !34
  %1285 = fadd float %1281, %1283, !dbg !33
  %1286 = extractelement <4 x float> %1024, i64 2, !dbg !32
  %1287 = call float @llvm.amdgcn.exp2.f32(float %1286), !dbg !32
  %1288 = insertelement <4 x float> %1284, float %1287, i64 2, !dbg !34
  %1289 = fadd float %1285, %1287, !dbg !33
  %1290 = extractelement <4 x float> %1024, i64 3, !dbg !32
  %1291 = call float @llvm.amdgcn.exp2.f32(float %1290), !dbg !32
  %1292 = insertelement <4 x float> %1288, float %1291, i64 3, !dbg !34
  %math.result368 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1292) #9, !dbg !34, !frisk.register_lifetimes !1
  %1293 = fptrunc <4 x float> %math.result368 to <4 x half>, !dbg !34
  %math.packed369 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1293) #9, !dbg !33, !frisk.register_lifetimes !1
  %1294 = fadd float %1289, %1291, !dbg !33
  %1295 = extractelement <4 x float> %1025, i64 0, !dbg !32
  %1296 = call float @llvm.amdgcn.exp2.f32(float %1295), !dbg !32
  %1297 = insertelement <4 x float> poison, float %1296, i64 0, !dbg !34
  %1298 = fadd float %1294, %1296, !dbg !33
  %1299 = extractelement <4 x float> %1025, i64 1, !dbg !32
  %1300 = call float @llvm.amdgcn.exp2.f32(float %1299), !dbg !32
  %1301 = insertelement <4 x float> %1297, float %1300, i64 1, !dbg !34
  %1302 = fadd float %1298, %1300, !dbg !33
  %1303 = extractelement <4 x float> %1025, i64 2, !dbg !32
  %1304 = call float @llvm.amdgcn.exp2.f32(float %1303), !dbg !32
  %1305 = insertelement <4 x float> %1301, float %1304, i64 2, !dbg !34
  %1306 = fadd float %1302, %1304, !dbg !33
  %1307 = extractelement <4 x float> %1025, i64 3, !dbg !32
  %1308 = call float @llvm.amdgcn.exp2.f32(float %1307), !dbg !32
  %1309 = insertelement <4 x float> %1305, float %1308, i64 3, !dbg !34
  %math.result370 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1309) #9, !dbg !34, !frisk.register_lifetimes !1
  %1310 = fptrunc <4 x float> %math.result370 to <4 x half>, !dbg !34
  %math.packed371 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1310) #9, !dbg !33, !frisk.register_lifetimes !1
  %1311 = fadd float %1306, %1308, !dbg !33
  %1312 = bitcast float %1176 to i32, !dbg !33
  %1313 = call i32 @llvm.amdgcn.ds.bpermute(i32 %949, i32 %1312), !dbg !33
  %1314 = bitcast i32 %1313 to float, !dbg !33
  %1315 = fadd float %1176, %1314, !dbg !33
  %1316 = bitcast float %1315 to i32, !dbg !33
  %1317 = call i32 @llvm.amdgcn.ds.bpermute(i32 %957, i32 %1316), !dbg !33
  %1318 = bitcast i32 %1317 to float, !dbg !33
  %1319 = fadd float %1315, %1318, !dbg !33
  %1320 = fadd float %1319, %1033, !dbg !33
  %1321 = insertelement <1 x float> poison, float %1320, i64 0, !dbg !7
  %1322 = insertvalue [2 x <1 x float>] poison, <1 x float> %1321, 0, !dbg !7
  %1323 = bitcast float %1311 to i32, !dbg !33
  %1324 = call i32 @llvm.amdgcn.ds.bpermute(i32 %949, i32 %1323), !dbg !33
  %1325 = bitcast i32 %1324 to float, !dbg !33
  %1326 = fadd float %1311, %1325, !dbg !33
  %1327 = bitcast float %1326 to i32, !dbg !33
  %1328 = call i32 @llvm.amdgcn.ds.bpermute(i32 %957, i32 %1327), !dbg !33
  %1329 = bitcast i32 %1328 to float, !dbg !33
  %1330 = fadd float %1326, %1329, !dbg !33
  %1331 = fadd float %1330, %1041, !dbg !33
  %1332 = insertelement <1 x float> poison, float %1331, i64 0, !dbg !7
  %1333 = insertvalue [2 x <1 x float>] %1322, <1 x float> %1332, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread897 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !14, !range !39, !frisk.register_lifetimes !1
  %address.local898 = shl nuw nsw i32 %address.thread897, 3, !dbg !14
  %address.local899 = and i32 %address.local898, 120, !dbg !14
  %address.local900 = shl nuw nsw i32 %address.thread897, 5, !dbg !14
  %address.local901 = and i32 %address.local900, 1536, !dbg !14
  %address.local902 = or disjoint i32 %address.local899, %address.local901, !dbg !14
  %address.local903 = zext nneg i32 %address.local902 to i64, !dbg !14
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !14
  %1334 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local904, i32 0) #0, !dbg !14
  %1335 = bitcast <4 x i32> %1334 to <8 x half>, !dbg !14
  %address.local905 = zext nneg i32 %address.thread897 to i64, !dbg !8
  %address.local906 = and i64 %address.local905, 60, !dbg !14
  %address.local907 = add nuw nsw i64 %address.local906, %address.local905, !dbg !14
  %address.local908 = shl nuw nsw i64 %address.local907, 3, !dbg !14
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !14
  %address.local910 = and i64 %address.local905, 63, !dbg !11
  %address.local911 = lshr i64 %address.local910, 4, !dbg !14
  %address.local912 = mul nuw nsw i64 %address.local911, 384, !dbg !14
  %address.local913 = getelementptr [2 x i8], ptr addrspace(3) %address.local909, i64 %address.local912, !dbg !14
  %address.local914 = lshr i64 %address.local905, 2, !dbg !17
  %address.local915 = mul nsw i64 %address.local914, -32, !dbg !14
  %address.local916 = getelementptr [2 x i8], ptr addrspace(3) %address.local913, i64 %address.local915, !dbg !14
  %address.local917 = getelementptr i8, ptr addrspace(3) %address.local916, i64 256, !dbg !14
  %1336 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local917, i32 0) #0, !dbg !14
  %1337 = bitcast <4 x i32> %1336 to <8 x half>, !dbg !14
  %address.local918 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !14
  %address.local919 = getelementptr [2 x i8], ptr addrspace(3) %address.local918, i64 %address.local912, !dbg !14
  %address.local920 = getelementptr [2 x i8], ptr addrspace(3) %address.local919, i64 %address.local915, !dbg !14
  %address.local921 = getelementptr i8, ptr addrspace(3) %address.local920, i64 512, !dbg !14
  %1338 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local921, i32 0) #0, !dbg !14
  %1339 = bitcast <4 x i32> %1338 to <8 x half>, !dbg !14
  %address.local922 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !14
  %address.local923 = getelementptr [2 x i8], ptr addrspace(3) %address.local922, i64 %address.local912, !dbg !14
  %address.local924 = getelementptr [2 x i8], ptr addrspace(3) %address.local923, i64 %address.local915, !dbg !14
  %address.local925 = getelementptr i8, ptr addrspace(3) %address.local924, i64 768, !dbg !14
  %1340 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local925, i32 0) #0, !dbg !14
  %1341 = bitcast <4 x i32> %1340 to <8 x half>, !dbg !14
  %address.local926 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !14
  %address.local927 = getelementptr i8, ptr addrspace(3) %address.local926, i64 4096, !dbg !14
  %1342 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local927, i32 0) #0, !dbg !14
  %1343 = bitcast <4 x i32> %1342 to <8 x half>, !dbg !14
  %address.local928 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !14
  %address.local929 = getelementptr [2 x i8], ptr addrspace(3) %address.local928, i64 %address.local912, !dbg !14
  %address.local930 = getelementptr [2 x i8], ptr addrspace(3) %address.local929, i64 %address.local915, !dbg !14
  %address.local931 = getelementptr i8, ptr addrspace(3) %address.local930, i64 4352, !dbg !14
  %1344 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local931, i32 0) #0, !dbg !14
  %1345 = bitcast <4 x i32> %1344 to <8 x half>, !dbg !14
  %address.local932 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !14
  %address.local933 = getelementptr [2 x i8], ptr addrspace(3) %address.local932, i64 %address.local912, !dbg !14
  %address.local934 = getelementptr [2 x i8], ptr addrspace(3) %address.local933, i64 %address.local915, !dbg !14
  %address.local935 = getelementptr i8, ptr addrspace(3) %address.local934, i64 4608, !dbg !14
  %1346 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !14
  %1347 = bitcast <4 x i32> %1346 to <8 x half>, !dbg !14
  %address.local936 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !14
  %address.local937 = getelementptr [2 x i8], ptr addrspace(3) %address.local936, i64 %address.local912, !dbg !14
  %address.local938 = getelementptr [2 x i8], ptr addrspace(3) %address.local937, i64 %address.local915, !dbg !14
  %address.local939 = getelementptr i8, ptr addrspace(3) %address.local938, i64 4864, !dbg !14
  %1348 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #0, !dbg !14
  %1349 = bitcast <4 x i32> %1348 to <8 x half>, !dbg !14
  %1350 = add nuw nsw i64 %217, %16, !dbg !18
  %1351 = shl i64 %1350, 19, !dbg !18
  %1352 = and i64 %1351, 16252928, !dbg !18
  %1353 = add i64 %1352, %223, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1354 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1353, !dbg !18
  %1355 = getelementptr i8, ptr addrspace(1) %1354, i64 64, !dbg !18
  %address.local940 = shl nuw nsw i64 %address.local905, 3, !dbg !8
  %address.local941 = and i64 %address.local940, 1536, !dbg !8
  %address.local942 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local941, !dbg !8
  %address.local943 = ptrtoint ptr addrspace(3) %address.local942 to i64, !dbg !8
  %address.local944 = trunc i64 %address.local943 to i32, !dbg !8
  %1356 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local944), !dbg !18
  %1357 = zext i32 %1356 to i64, !dbg !18
  %1358 = inttoptr i64 %1357 to ptr addrspace(3), !dbg !18
  %address.local945 = shl nuw nsw i64 %address.local905, 5, !dbg !18
  %address.local946 = and i64 %address.local945, 8064, !dbg !18
  %address.local947 = shl nuw nsw i64 %address.local905, 3, !dbg !18
  %address.local948 = and i64 %address.local947, 24, !dbg !18
  %address.local949 = or disjoint i64 %address.local946, %address.local948, !dbg !18
  %address.local950 = getelementptr [2 x i8], ptr addrspace(1) %1355, i64 %address.local949, !dbg !18
  %1359 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1360 = ptrtoint ptr addrspace(1) %address.local950 to i64, !dbg !18
  %1361 = sub i64 %1360, %1359, !dbg !18
  %dma.buffer.offset1591 = trunc i64 %1361 to i32, !dbg !18
  %1362 = trunc i64 %1359 to i32, !dbg !18
  %1363 = insertelement <4 x i32> poison, i32 %1362, i64 0, !dbg !18
  %1364 = lshr i64 %1359, 32, !dbg !18
  %1365 = trunc i64 %1364 to i32, !dbg !18
  %1366 = insertelement <4 x i32> %1363, i32 %1365, i64 1, !dbg !18
  %1367 = insertelement <4 x i32> %1366, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1592 = insertelement <4 x i32> %1367, i32 131072, i64 3, !dbg !18
  %1368 = ptrtoint ptr addrspace(3) %1358 to i32, !dbg !18
  %dma.buffer.lds1593 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1368), !dbg !18
  %1369 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1591, <4 x i32> %dma.buffer.rsrc1592) #10, !dbg !18
  %address.local951 = or disjoint i64 %address.local905, 256, !dbg !8
  %address.local952 = shl nuw nsw i64 %address.local951, 3, !dbg !8
  %address.local953 = and i64 %address.local952, 3584, !dbg !8
  %address.local954 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local953, !dbg !8
  %address.local955 = ptrtoint ptr addrspace(3) %address.local954 to i64, !dbg !8
  %address.local956 = trunc i64 %address.local955 to i32, !dbg !8
  %1370 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local956), !dbg !18
  %1371 = zext i32 %1370 to i64, !dbg !18
  %1372 = inttoptr i64 %1371 to ptr addrspace(3), !dbg !18
  %address.local957 = shl nuw nsw i64 %address.local951, 5, !dbg !18
  %address.local958 = and i64 %address.local957, 16256, !dbg !18
  %address.local959 = shl nuw nsw i64 %address.local905, 3, !dbg !18
  %address.local960 = and i64 %address.local959, 24, !dbg !18
  %address.local961 = or disjoint i64 %address.local958, %address.local960, !dbg !18
  %address.local962 = getelementptr [2 x i8], ptr addrspace(1) %1355, i64 %address.local961, !dbg !18
  %1373 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1374 = ptrtoint ptr addrspace(1) %address.local962 to i64, !dbg !18
  %1375 = sub i64 %1374, %1373, !dbg !18
  %dma.buffer.offset1588 = trunc i64 %1375 to i32, !dbg !18
  %1376 = trunc i64 %1373 to i32, !dbg !18
  %1377 = insertelement <4 x i32> poison, i32 %1376, i64 0, !dbg !18
  %1378 = lshr i64 %1373, 32, !dbg !18
  %1379 = trunc i64 %1378 to i32, !dbg !18
  %1380 = insertelement <4 x i32> %1377, i32 %1379, i64 1, !dbg !18
  %1381 = insertelement <4 x i32> %1380, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1589 = insertelement <4 x i32> %1381, i32 131072, i64 3, !dbg !18
  %1382 = ptrtoint ptr addrspace(3) %1372 to i32, !dbg !18
  %dma.buffer.lds1590 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1382), !dbg !18
  %1383 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1589) #10, !dbg !18
  %1384 = shufflevector <8 x half> %1335, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1385 = extractvalue [2 x <8 x float>] %206, 0, !dbg !35
  %1386 = shufflevector <8 x float> %1385, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1387 = fmul <4 x float> %1386, %1030, !dbg !35
  %1388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1384, <4 x float> %1387) #9, !dbg !14
  %1389 = shufflevector <8 x half> %1337, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed343, <4 x half> %1389, <4 x float> %1388) #9, !dbg !14
  %1391 = shufflevector <8 x half> %1339, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed345, <4 x half> %1391, <4 x float> %1390) #9, !dbg !14
  %1393 = shufflevector <8 x half> %1341, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed347, <4 x half> %1393, <4 x float> %1392) #9, !dbg !14
  %1395 = shufflevector <8 x half> %1343, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed349, <4 x half> %1395, <4 x float> %1394) #9, !dbg !14
  %1397 = shufflevector <8 x half> %1345, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed351, <4 x half> %1397, <4 x float> %1396) #9, !dbg !14
  %1399 = shufflevector <8 x half> %1347, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed353, <4 x half> %1399, <4 x float> %1398) #9, !dbg !14
  %1401 = shufflevector <8 x half> %1349, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed355, <4 x half> %1401, <4 x float> %1400) #9, !dbg !14
  %1403 = shufflevector <8 x half> %1335, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1404 = shufflevector <8 x float> %1385, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1405 = fmul <4 x float> %1404, %1030, !dbg !35
  %1406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1403, <4 x float> %1405) #9, !dbg !14
  %1407 = shufflevector <8 x half> %1337, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed343, <4 x half> %1407, <4 x float> %1406) #9, !dbg !14
  %1409 = shufflevector <8 x half> %1339, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed345, <4 x half> %1409, <4 x float> %1408) #9, !dbg !14
  %1411 = shufflevector <8 x half> %1341, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed347, <4 x half> %1411, <4 x float> %1410) #9, !dbg !14
  %1413 = shufflevector <8 x half> %1343, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed349, <4 x half> %1413, <4 x float> %1412) #9, !dbg !14
  %1415 = shufflevector <8 x half> %1345, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1416 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed351, <4 x half> %1415, <4 x float> %1414) #9, !dbg !14
  %1417 = shufflevector <8 x half> %1347, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed353, <4 x half> %1417, <4 x float> %1416) #9, !dbg !14
  %1419 = shufflevector <8 x half> %1349, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1420 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed355, <4 x half> %1419, <4 x float> %1418) #9, !dbg !14
  %1421 = extractvalue [2 x <8 x float>] %206, 1, !dbg !35
  %1422 = shufflevector <8 x float> %1421, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1423 = fmul <4 x float> %1422, %1038, !dbg !35
  %1424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed357, <4 x half> %1384, <4 x float> %1423) #9, !dbg !14
  %1425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed359, <4 x half> %1389, <4 x float> %1424) #9, !dbg !14
  %1426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed361, <4 x half> %1391, <4 x float> %1425) #9, !dbg !14
  %1427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed363, <4 x half> %1393, <4 x float> %1426) #9, !dbg !14
  %1428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed365, <4 x half> %1395, <4 x float> %1427) #9, !dbg !14
  %1429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed367, <4 x half> %1397, <4 x float> %1428) #9, !dbg !14
  %1430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %1399, <4 x float> %1429) #9, !dbg !14
  %1431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %1401, <4 x float> %1430) #9, !dbg !14
  %1432 = shufflevector <8 x float> %1421, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1433 = fmul <4 x float> %1432, %1038, !dbg !35
  %1434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed357, <4 x half> %1403, <4 x float> %1433) #9, !dbg !14
  %1435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed359, <4 x half> %1407, <4 x float> %1434) #9, !dbg !14
  %1436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed361, <4 x half> %1409, <4 x float> %1435) #9, !dbg !14
  %1437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed363, <4 x half> %1411, <4 x float> %1436) #9, !dbg !14
  %1438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed365, <4 x half> %1413, <4 x float> %1437) #9, !dbg !14
  %1439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed367, <4 x half> %1415, <4 x float> %1438) #9, !dbg !14
  %1440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %1417, <4 x float> %1439) #9, !dbg !14
  %1441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %1419, <4 x float> %1440) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1442 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local904, i32 0) #0, !dbg !14
  %1443 = bitcast <4 x i32> %1442 to <8 x half>, !dbg !14
  %1444 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local917, i32 0) #0, !dbg !14
  %1445 = bitcast <4 x i32> %1444 to <8 x half>, !dbg !14
  %1446 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local921, i32 0) #0, !dbg !14
  %1447 = bitcast <4 x i32> %1446 to <8 x half>, !dbg !14
  %1448 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local925, i32 0) #0, !dbg !14
  %1449 = bitcast <4 x i32> %1448 to <8 x half>, !dbg !14
  %1450 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local927, i32 0) #0, !dbg !14
  %1451 = bitcast <4 x i32> %1450 to <8 x half>, !dbg !14
  %1452 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local931, i32 0) #0, !dbg !14
  %1453 = bitcast <4 x i32> %1452 to <8 x half>, !dbg !14
  %1454 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !14
  %1455 = bitcast <4 x i32> %1454 to <8 x half>, !dbg !14
  %1456 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #0, !dbg !14
  %1457 = bitcast <4 x i32> %1456 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1458 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1353, !dbg !18
  %1459 = getelementptr i8, ptr addrspace(1) %1458, i64 128, !dbg !18
  %1460 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local944), !dbg !18
  %1461 = zext i32 %1460 to i64, !dbg !18
  %1462 = inttoptr i64 %1461 to ptr addrspace(3), !dbg !18
  %address.local963 = getelementptr [2 x i8], ptr addrspace(1) %1459, i64 %address.local949, !dbg !18
  %1463 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1464 = ptrtoint ptr addrspace(1) %address.local963 to i64, !dbg !18
  %1465 = sub i64 %1464, %1463, !dbg !18
  %dma.buffer.offset1585 = trunc i64 %1465 to i32, !dbg !18
  %1466 = trunc i64 %1463 to i32, !dbg !18
  %1467 = insertelement <4 x i32> poison, i32 %1466, i64 0, !dbg !18
  %1468 = lshr i64 %1463, 32, !dbg !18
  %1469 = trunc i64 %1468 to i32, !dbg !18
  %1470 = insertelement <4 x i32> %1467, i32 %1469, i64 1, !dbg !18
  %1471 = insertelement <4 x i32> %1470, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1586 = insertelement <4 x i32> %1471, i32 131072, i64 3, !dbg !18
  %1472 = ptrtoint ptr addrspace(3) %1462 to i32, !dbg !18
  %dma.buffer.lds1587 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1472), !dbg !18
  %1473 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1586) #10, !dbg !18
  %1474 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local956), !dbg !18
  %1475 = zext i32 %1474 to i64, !dbg !18
  %1476 = inttoptr i64 %1475 to ptr addrspace(3), !dbg !18
  %address.local964 = getelementptr [2 x i8], ptr addrspace(1) %1459, i64 %address.local961, !dbg !18
  %1477 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1478 = ptrtoint ptr addrspace(1) %address.local964 to i64, !dbg !18
  %1479 = sub i64 %1478, %1477, !dbg !18
  %dma.buffer.offset1582 = trunc i64 %1479 to i32, !dbg !18
  %1480 = trunc i64 %1477 to i32, !dbg !18
  %1481 = insertelement <4 x i32> poison, i32 %1480, i64 0, !dbg !18
  %1482 = lshr i64 %1477, 32, !dbg !18
  %1483 = trunc i64 %1482 to i32, !dbg !18
  %1484 = insertelement <4 x i32> %1481, i32 %1483, i64 1, !dbg !18
  %1485 = insertelement <4 x i32> %1484, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1583 = insertelement <4 x i32> %1485, i32 131072, i64 3, !dbg !18
  %1486 = ptrtoint ptr addrspace(3) %1476 to i32, !dbg !18
  %dma.buffer.lds1584 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1486), !dbg !18
  %1487 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1583) #10, !dbg !18
  %1488 = shufflevector <8 x half> %1443, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1489 = extractvalue [2 x <8 x float>] %209, 0, !dbg !35
  %1490 = shufflevector <8 x float> %1489, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1491 = fmul <4 x float> %1490, %1030, !dbg !35
  %1492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1488, <4 x float> %1491) #9, !dbg !14
  %1493 = shufflevector <8 x half> %1445, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed343, <4 x half> %1493, <4 x float> %1492) #9, !dbg !14
  %1495 = shufflevector <8 x half> %1447, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed345, <4 x half> %1495, <4 x float> %1494) #9, !dbg !14
  %1497 = shufflevector <8 x half> %1449, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed347, <4 x half> %1497, <4 x float> %1496) #9, !dbg !14
  %1499 = shufflevector <8 x half> %1451, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed349, <4 x half> %1499, <4 x float> %1498) #9, !dbg !14
  %1501 = shufflevector <8 x half> %1453, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed351, <4 x half> %1501, <4 x float> %1500) #9, !dbg !14
  %1503 = shufflevector <8 x half> %1455, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed353, <4 x half> %1503, <4 x float> %1502) #9, !dbg !14
  %1505 = shufflevector <8 x half> %1457, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed355, <4 x half> %1505, <4 x float> %1504) #9, !dbg !14
  %1507 = shufflevector <8 x half> %1443, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1508 = shufflevector <8 x float> %1489, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1509 = fmul <4 x float> %1508, %1030, !dbg !35
  %1510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1507, <4 x float> %1509) #9, !dbg !14
  %1511 = shufflevector <8 x half> %1445, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed343, <4 x half> %1511, <4 x float> %1510) #9, !dbg !14
  %1513 = shufflevector <8 x half> %1447, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed345, <4 x half> %1513, <4 x float> %1512) #9, !dbg !14
  %1515 = shufflevector <8 x half> %1449, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed347, <4 x half> %1515, <4 x float> %1514) #9, !dbg !14
  %1517 = shufflevector <8 x half> %1451, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed349, <4 x half> %1517, <4 x float> %1516) #9, !dbg !14
  %1519 = shufflevector <8 x half> %1453, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1520 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed351, <4 x half> %1519, <4 x float> %1518) #9, !dbg !14
  %1521 = shufflevector <8 x half> %1455, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed353, <4 x half> %1521, <4 x float> %1520) #9, !dbg !14
  %1523 = shufflevector <8 x half> %1457, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed355, <4 x half> %1523, <4 x float> %1522) #9, !dbg !14
  %1525 = extractvalue [2 x <8 x float>] %209, 1, !dbg !35
  %1526 = shufflevector <8 x float> %1525, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1527 = fmul <4 x float> %1526, %1038, !dbg !35
  %1528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed357, <4 x half> %1488, <4 x float> %1527) #9, !dbg !14
  %1529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed359, <4 x half> %1493, <4 x float> %1528) #9, !dbg !14
  %1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed361, <4 x half> %1495, <4 x float> %1529) #9, !dbg !14
  %1531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed363, <4 x half> %1497, <4 x float> %1530) #9, !dbg !14
  %1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed365, <4 x half> %1499, <4 x float> %1531) #9, !dbg !14
  %1533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed367, <4 x half> %1501, <4 x float> %1532) #9, !dbg !14
  %1534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %1503, <4 x float> %1533) #9, !dbg !14
  %1535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %1505, <4 x float> %1534) #9, !dbg !14
  %1536 = shufflevector <8 x float> %1525, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1537 = fmul <4 x float> %1536, %1038, !dbg !35
  %1538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed357, <4 x half> %1507, <4 x float> %1537) #9, !dbg !14
  %1539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed359, <4 x half> %1511, <4 x float> %1538) #9, !dbg !14
  %1540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed361, <4 x half> %1513, <4 x float> %1539) #9, !dbg !14
  %1541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed363, <4 x half> %1515, <4 x float> %1540) #9, !dbg !14
  %1542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed365, <4 x half> %1517, <4 x float> %1541) #9, !dbg !14
  %1543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed367, <4 x half> %1519, <4 x float> %1542) #9, !dbg !14
  %1544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %1521, <4 x float> %1543) #9, !dbg !14
  %1545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %1523, <4 x float> %1544) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1546 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local904, i32 0) #0, !dbg !14
  %1547 = bitcast <4 x i32> %1546 to <8 x half>, !dbg !14
  %1548 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local917, i32 0) #0, !dbg !14
  %1549 = bitcast <4 x i32> %1548 to <8 x half>, !dbg !14
  %1550 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local921, i32 0) #0, !dbg !14
  %1551 = bitcast <4 x i32> %1550 to <8 x half>, !dbg !14
  %1552 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local925, i32 0) #0, !dbg !14
  %1553 = bitcast <4 x i32> %1552 to <8 x half>, !dbg !14
  %1554 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local927, i32 0) #0, !dbg !14
  %1555 = bitcast <4 x i32> %1554 to <8 x half>, !dbg !14
  %1556 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local931, i32 0) #0, !dbg !14
  %1557 = bitcast <4 x i32> %1556 to <8 x half>, !dbg !14
  %1558 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !14
  %1559 = bitcast <4 x i32> %1558 to <8 x half>, !dbg !14
  %1560 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #0, !dbg !14
  %1561 = bitcast <4 x i32> %1560 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1562 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1353, !dbg !18
  %1563 = getelementptr i8, ptr addrspace(1) %1562, i64 192, !dbg !18
  %1564 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local944), !dbg !18
  %1565 = zext i32 %1564 to i64, !dbg !18
  %1566 = inttoptr i64 %1565 to ptr addrspace(3), !dbg !18
  %address.local965 = getelementptr [2 x i8], ptr addrspace(1) %1563, i64 %address.local949, !dbg !18
  %1567 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1568 = ptrtoint ptr addrspace(1) %address.local965 to i64, !dbg !18
  %1569 = sub i64 %1568, %1567, !dbg !18
  %dma.buffer.offset1579 = trunc i64 %1569 to i32, !dbg !18
  %1570 = trunc i64 %1567 to i32, !dbg !18
  %1571 = insertelement <4 x i32> poison, i32 %1570, i64 0, !dbg !18
  %1572 = lshr i64 %1567, 32, !dbg !18
  %1573 = trunc i64 %1572 to i32, !dbg !18
  %1574 = insertelement <4 x i32> %1571, i32 %1573, i64 1, !dbg !18
  %1575 = insertelement <4 x i32> %1574, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1580 = insertelement <4 x i32> %1575, i32 131072, i64 3, !dbg !18
  %1576 = ptrtoint ptr addrspace(3) %1566 to i32, !dbg !18
  %dma.buffer.lds1581 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1576), !dbg !18
  %1577 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1580) #10, !dbg !18
  %1578 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local956), !dbg !18
  %1579 = zext i32 %1578 to i64, !dbg !18
  %1580 = inttoptr i64 %1579 to ptr addrspace(3), !dbg !18
  %address.local966 = getelementptr [2 x i8], ptr addrspace(1) %1563, i64 %address.local961, !dbg !18
  %1581 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1582 = ptrtoint ptr addrspace(1) %address.local966 to i64, !dbg !18
  %1583 = sub i64 %1582, %1581, !dbg !18
  %dma.buffer.offset = trunc i64 %1583 to i32, !dbg !18
  %1584 = trunc i64 %1581 to i32, !dbg !18
  %1585 = insertelement <4 x i32> poison, i32 %1584, i64 0, !dbg !18
  %1586 = lshr i64 %1581, 32, !dbg !18
  %1587 = trunc i64 %1586 to i32, !dbg !18
  %1588 = insertelement <4 x i32> %1585, i32 %1587, i64 1, !dbg !18
  %1589 = insertelement <4 x i32> %1588, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1589, i32 131072, i64 3, !dbg !18
  %1590 = ptrtoint ptr addrspace(3) %1580 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1590), !dbg !18
  %1591 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1592 = shufflevector <8 x half> %1547, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1593 = extractvalue [2 x <8 x float>] %210, 0, !dbg !35
  %1594 = shufflevector <8 x float> %1593, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1595 = fmul <4 x float> %1594, %1030, !dbg !35
  %1596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1592, <4 x float> %1595) #9, !dbg !14
  %1597 = shufflevector <8 x half> %1549, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed343, <4 x half> %1597, <4 x float> %1596) #9, !dbg !14
  %1599 = shufflevector <8 x half> %1551, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed345, <4 x half> %1599, <4 x float> %1598) #9, !dbg !14
  %1601 = shufflevector <8 x half> %1553, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1602 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed347, <4 x half> %1601, <4 x float> %1600) #9, !dbg !14
  %1603 = shufflevector <8 x half> %1555, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1604 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed349, <4 x half> %1603, <4 x float> %1602) #9, !dbg !14
  %1605 = shufflevector <8 x half> %1557, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1606 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed351, <4 x half> %1605, <4 x float> %1604) #9, !dbg !14
  %1607 = shufflevector <8 x half> %1559, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed353, <4 x half> %1607, <4 x float> %1606) #9, !dbg !14
  %1609 = shufflevector <8 x half> %1561, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed355, <4 x half> %1609, <4 x float> %1608) #9, !dbg !14
  %1611 = shufflevector <8 x half> %1547, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1612 = shufflevector <8 x float> %1593, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1613 = fmul <4 x float> %1612, %1030, !dbg !35
  %1614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1611, <4 x float> %1613) #9, !dbg !14
  %1615 = shufflevector <8 x half> %1549, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed343, <4 x half> %1615, <4 x float> %1614) #9, !dbg !14
  %1617 = shufflevector <8 x half> %1551, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed345, <4 x half> %1617, <4 x float> %1616) #9, !dbg !14
  %1619 = shufflevector <8 x half> %1553, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed347, <4 x half> %1619, <4 x float> %1618) #9, !dbg !14
  %1621 = shufflevector <8 x half> %1555, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed349, <4 x half> %1621, <4 x float> %1620) #9, !dbg !14
  %1623 = shufflevector <8 x half> %1557, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed351, <4 x half> %1623, <4 x float> %1622) #9, !dbg !14
  %1625 = shufflevector <8 x half> %1559, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed353, <4 x half> %1625, <4 x float> %1624) #9, !dbg !14
  %1627 = shufflevector <8 x half> %1561, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1628 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed355, <4 x half> %1627, <4 x float> %1626) #9, !dbg !14
  %1629 = extractvalue [2 x <8 x float>] %210, 1, !dbg !35
  %1630 = shufflevector <8 x float> %1629, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1631 = fmul <4 x float> %1630, %1038, !dbg !35
  %1632 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed357, <4 x half> %1592, <4 x float> %1631) #9, !dbg !14
  %1633 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed359, <4 x half> %1597, <4 x float> %1632) #9, !dbg !14
  %1634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed361, <4 x half> %1599, <4 x float> %1633) #9, !dbg !14
  %1635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed363, <4 x half> %1601, <4 x float> %1634) #9, !dbg !14
  %1636 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed365, <4 x half> %1603, <4 x float> %1635) #9, !dbg !14
  %1637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed367, <4 x half> %1605, <4 x float> %1636) #9, !dbg !14
  %1638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %1607, <4 x float> %1637) #9, !dbg !14
  %1639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %1609, <4 x float> %1638) #9, !dbg !14
  %1640 = shufflevector <8 x float> %1629, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1641 = fmul <4 x float> %1640, %1038, !dbg !35
  %1642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed357, <4 x half> %1611, <4 x float> %1641) #9, !dbg !14
  %1643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed359, <4 x half> %1615, <4 x float> %1642) #9, !dbg !14
  %1644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed361, <4 x half> %1617, <4 x float> %1643) #9, !dbg !14
  %1645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed363, <4 x half> %1619, <4 x float> %1644) #9, !dbg !14
  %1646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed365, <4 x half> %1621, <4 x float> %1645) #9, !dbg !14
  %1647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed367, <4 x half> %1623, <4 x float> %1646) #9, !dbg !14
  %1648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %1625, <4 x float> %1647) #9, !dbg !14
  %1649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %1627, <4 x float> %1648) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1650 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local904, i32 0) #0, !dbg !14
  %1651 = bitcast <4 x i32> %1650 to <8 x half>, !dbg !14
  %1652 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local917, i32 0) #0, !dbg !14
  %1653 = bitcast <4 x i32> %1652 to <8 x half>, !dbg !14
  %1654 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local921, i32 0) #0, !dbg !14
  %1655 = bitcast <4 x i32> %1654 to <8 x half>, !dbg !14
  %1656 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local925, i32 0) #0, !dbg !14
  %1657 = bitcast <4 x i32> %1656 to <8 x half>, !dbg !14
  %1658 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local927, i32 0) #0, !dbg !14
  %1659 = bitcast <4 x i32> %1658 to <8 x half>, !dbg !14
  %1660 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local931, i32 0) #0, !dbg !14
  %1661 = bitcast <4 x i32> %1660 to <8 x half>, !dbg !14
  %1662 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !14
  %1663 = bitcast <4 x i32> %1662 to <8 x half>, !dbg !14
  %1664 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #0, !dbg !14
  %1665 = bitcast <4 x i32> %1664 to <8 x half>, !dbg !14
  %1666 = shufflevector <8 x half> %1651, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1667 = extractvalue [2 x <8 x float>] %211, 0, !dbg !35
  %1668 = shufflevector <8 x float> %1667, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1669 = fmul <4 x float> %1668, %1030, !dbg !35
  %1670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1666, <4 x float> %1669) #9, !dbg !14
  %1671 = shufflevector <8 x half> %1653, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed343, <4 x half> %1671, <4 x float> %1670) #9, !dbg !14
  %1673 = shufflevector <8 x half> %1655, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed345, <4 x half> %1673, <4 x float> %1672) #9, !dbg !14
  %1675 = shufflevector <8 x half> %1657, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed347, <4 x half> %1675, <4 x float> %1674) #9, !dbg !14
  %1677 = shufflevector <8 x half> %1659, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed349, <4 x half> %1677, <4 x float> %1676) #9, !dbg !14
  %1679 = shufflevector <8 x half> %1661, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1680 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed351, <4 x half> %1679, <4 x float> %1678) #9, !dbg !14
  %1681 = shufflevector <8 x half> %1663, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed353, <4 x half> %1681, <4 x float> %1680) #9, !dbg !14
  %1683 = shufflevector <8 x half> %1665, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1684 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed355, <4 x half> %1683, <4 x float> %1682) #9, !dbg !14
  %1685 = shufflevector <8 x half> %1651, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1686 = shufflevector <8 x float> %1667, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1687 = fmul <4 x float> %1686, %1030, !dbg !35
  %1688 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1685, <4 x float> %1687) #9, !dbg !14
  %1689 = shufflevector <8 x half> %1653, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed343, <4 x half> %1689, <4 x float> %1688) #9, !dbg !14
  %1691 = shufflevector <8 x half> %1655, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1692 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed345, <4 x half> %1691, <4 x float> %1690) #9, !dbg !14
  %1693 = shufflevector <8 x half> %1657, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed347, <4 x half> %1693, <4 x float> %1692) #9, !dbg !14
  %1695 = shufflevector <8 x half> %1659, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed349, <4 x half> %1695, <4 x float> %1694) #9, !dbg !14
  %1697 = shufflevector <8 x half> %1661, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed351, <4 x half> %1697, <4 x float> %1696) #9, !dbg !14
  %1699 = shufflevector <8 x half> %1663, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1700 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed353, <4 x half> %1699, <4 x float> %1698) #9, !dbg !14
  %1701 = shufflevector <8 x half> %1665, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1702 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed355, <4 x half> %1701, <4 x float> %1700) #9, !dbg !14
  %1703 = extractvalue [2 x <8 x float>] %211, 1, !dbg !35
  %1704 = shufflevector <8 x float> %1703, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1705 = fmul <4 x float> %1704, %1038, !dbg !35
  %1706 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed357, <4 x half> %1666, <4 x float> %1705) #9, !dbg !14
  %1707 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed359, <4 x half> %1671, <4 x float> %1706) #9, !dbg !14
  %1708 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed361, <4 x half> %1673, <4 x float> %1707) #9, !dbg !14
  %1709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed363, <4 x half> %1675, <4 x float> %1708) #9, !dbg !14
  %1710 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed365, <4 x half> %1677, <4 x float> %1709) #9, !dbg !14
  %1711 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed367, <4 x half> %1679, <4 x float> %1710) #9, !dbg !14
  %1712 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %1681, <4 x float> %1711) #9, !dbg !14
  %1713 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %1683, <4 x float> %1712) #9, !dbg !14
  %1714 = shufflevector <8 x float> %1703, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1715 = fmul <4 x float> %1714, %1038, !dbg !35
  %1716 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed357, <4 x half> %1685, <4 x float> %1715) #9, !dbg !14
  %1717 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed359, <4 x half> %1689, <4 x float> %1716) #9, !dbg !14
  %1718 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed361, <4 x half> %1691, <4 x float> %1717) #9, !dbg !14
  %1719 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed363, <4 x half> %1693, <4 x float> %1718) #9, !dbg !14
  %1720 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed365, <4 x half> %1695, <4 x float> %1719) #9, !dbg !14
  %1721 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed367, <4 x half> %1697, <4 x float> %1720) #9, !dbg !14
  %1722 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %1699, <4 x float> %1721) #9, !dbg !14
  %1723 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %1701, <4 x float> %1722) #9, !dbg !14
  %1724 = shufflevector <4 x float> %1402, <4 x float> %1420, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1725 = insertvalue [2 x <8 x float>] poison, <8 x float> %1724, 0, !dbg !7
  %1726 = shufflevector <4 x float> %1431, <4 x float> %1441, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1727 = insertvalue [2 x <8 x float>] %1725, <8 x float> %1726, 1, !dbg !7
  %1728 = insertvalue [2 x <1 x float>] poison, <1 x float> %964, 0, !dbg !7
  %1729 = insertvalue [2 x <1 x float>] %1728, <1 x float> %1007, 1, !dbg !7
  %1730 = shufflevector <4 x float> %1506, <4 x float> %1524, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1731 = insertvalue [2 x <8 x float>] poison, <8 x float> %1730, 0, !dbg !7
  %1732 = shufflevector <4 x float> %1535, <4 x float> %1545, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1733 = insertvalue [2 x <8 x float>] %1731, <8 x float> %1732, 1, !dbg !7
  %1734 = shufflevector <4 x float> %1610, <4 x float> %1628, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1735 = insertvalue [2 x <8 x float>] poison, <8 x float> %1734, 0, !dbg !7
  %1736 = shufflevector <4 x float> %1639, <4 x float> %1649, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1737 = insertvalue [2 x <8 x float>] %1735, <8 x float> %1736, 1, !dbg !7
  %1738 = shufflevector <4 x float> %1684, <4 x float> %1702, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1739 = insertvalue [2 x <8 x float>] poison, <8 x float> %1738, 0, !dbg !7
  %1740 = shufflevector <4 x float> %1713, <4 x float> %1723, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1741 = insertvalue [2 x <8 x float>] %1739, <8 x float> %1740, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1742 = add nuw nsw i64 %205, 1, !dbg !7
  br label %204, !dbg !7

1743:                                             ; preds = %204
  %1744 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %1744, 0, !dbg !7
  %1745 = and i32 %31, 32, !dbg !7
  %.not36 = icmp eq i32 %1745, 0, !dbg !7
  %1746 = extractvalue [2 x <8 x float>] %211, 0, !dbg !7
  %1747 = extractelement <8 x float> %1746, i64 0, !dbg !7
  %1748 = extractelement <8 x float> %1746, i64 1, !dbg !7
  %1749 = extractelement <8 x float> %1746, i64 2, !dbg !7
  %1750 = extractelement <8 x float> %1746, i64 3, !dbg !7
  %1751 = select i1 %.not, float %1748, float %1747, !dbg !7
  %1752 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1753 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1752), !dbg !7
  %1754 = add nuw nsw i32 %1753, 64, !dbg !7
  %1755 = xor i32 %1753, 16, !dbg !7
  %1756 = and i32 %1754, 192, !dbg !7
  %1757 = icmp samesign ult i32 %1755, %1756, !dbg !7
  %1758 = select i1 %1757, i32 %1755, i32 %1753, !dbg !7
  %1759 = shl nuw nsw i32 %1758, 2, !dbg !7
  %1760 = bitcast float %1751 to i32, !dbg !7
  %1761 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1760), !dbg !7
  %1762 = bitcast i32 %1761 to float, !dbg !7
  %1763 = select i1 %.not, float %1747, float %1762, !dbg !7
  %1764 = select i1 %.not, float %1762, float %1748, !dbg !7
  %1765 = select i1 %.not, float %1750, float %1749, !dbg !7
  %1766 = bitcast float %1765 to i32, !dbg !7
  %1767 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1766), !dbg !7
  %1768 = bitcast i32 %1767 to float, !dbg !7
  %1769 = select i1 %.not, float %1749, float %1768, !dbg !7
  %1770 = select i1 %.not, float %1768, float %1750, !dbg !7
  %1771 = select i1 %.not36, float %1769, float %1763, !dbg !7
  %1772 = xor i32 %1753, 32, !dbg !7
  %1773 = icmp samesign ult i32 %1772, %1756, !dbg !7
  %1774 = select i1 %1773, i32 %1772, i32 %1753, !dbg !7
  %1775 = shl nuw nsw i32 %1774, 2, !dbg !7
  %1776 = bitcast float %1771 to i32, !dbg !7
  %1777 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1776), !dbg !7
  %1778 = bitcast i32 %1777 to float, !dbg !7
  %1779 = select i1 %.not36, float %1763, float %1778, !dbg !7
  %1780 = select i1 %.not36, float %1778, float %1769, !dbg !7
  %1781 = select i1 %.not36, float %1770, float %1764, !dbg !7
  %1782 = bitcast float %1781 to i32, !dbg !7
  %1783 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1782), !dbg !7
  %1784 = bitcast i32 %1783 to float, !dbg !7
  %1785 = select i1 %.not36, float %1764, float %1784, !dbg !7
  %1786 = select i1 %.not36, float %1784, float %1770, !dbg !7
  %1787 = extractvalue [2 x <8 x float>] %210, 0, !dbg !7
  %1788 = extractelement <8 x float> %1787, i64 0, !dbg !7
  %1789 = extractelement <8 x float> %1787, i64 1, !dbg !7
  %1790 = extractelement <8 x float> %1787, i64 2, !dbg !7
  %1791 = extractelement <8 x float> %1787, i64 3, !dbg !7
  %1792 = select i1 %.not, float %1789, float %1788, !dbg !7
  %1793 = bitcast float %1792 to i32, !dbg !7
  %1794 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1793), !dbg !7
  %1795 = bitcast i32 %1794 to float, !dbg !7
  %1796 = select i1 %.not, float %1788, float %1795, !dbg !7
  %1797 = select i1 %.not, float %1795, float %1789, !dbg !7
  %1798 = select i1 %.not, float %1791, float %1790, !dbg !7
  %1799 = bitcast float %1798 to i32, !dbg !7
  %1800 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1799), !dbg !7
  %1801 = bitcast i32 %1800 to float, !dbg !7
  %1802 = select i1 %.not, float %1790, float %1801, !dbg !7
  %1803 = select i1 %.not, float %1801, float %1791, !dbg !7
  %1804 = select i1 %.not36, float %1802, float %1796, !dbg !7
  %1805 = bitcast float %1804 to i32, !dbg !7
  %1806 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1805), !dbg !7
  %1807 = bitcast i32 %1806 to float, !dbg !7
  %1808 = select i1 %.not36, float %1796, float %1807, !dbg !7
  %1809 = select i1 %.not36, float %1807, float %1802, !dbg !7
  %1810 = select i1 %.not36, float %1803, float %1797, !dbg !7
  %1811 = bitcast float %1810 to i32, !dbg !7
  %1812 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1811), !dbg !7
  %1813 = bitcast i32 %1812 to float, !dbg !7
  %1814 = select i1 %.not36, float %1797, float %1813, !dbg !7
  %1815 = select i1 %.not36, float %1813, float %1803, !dbg !7
  %1816 = extractvalue [2 x <8 x float>] %209, 0, !dbg !7
  %1817 = extractelement <8 x float> %1816, i64 0, !dbg !7
  %1818 = extractelement <8 x float> %1816, i64 1, !dbg !7
  %1819 = extractelement <8 x float> %1816, i64 2, !dbg !7
  %1820 = extractelement <8 x float> %1816, i64 3, !dbg !7
  %1821 = select i1 %.not, float %1818, float %1817, !dbg !7
  %1822 = bitcast float %1821 to i32, !dbg !7
  %1823 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1822), !dbg !7
  %1824 = bitcast i32 %1823 to float, !dbg !7
  %1825 = select i1 %.not, float %1817, float %1824, !dbg !7
  %1826 = select i1 %.not, float %1824, float %1818, !dbg !7
  %1827 = select i1 %.not, float %1820, float %1819, !dbg !7
  %1828 = bitcast float %1827 to i32, !dbg !7
  %1829 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1828), !dbg !7
  %1830 = bitcast i32 %1829 to float, !dbg !7
  %1831 = select i1 %.not, float %1819, float %1830, !dbg !7
  %1832 = select i1 %.not, float %1830, float %1820, !dbg !7
  %1833 = select i1 %.not36, float %1831, float %1825, !dbg !7
  %1834 = bitcast float %1833 to i32, !dbg !7
  %1835 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1834), !dbg !7
  %1836 = bitcast i32 %1835 to float, !dbg !7
  %1837 = select i1 %.not36, float %1825, float %1836, !dbg !7
  %1838 = select i1 %.not36, float %1836, float %1831, !dbg !7
  %1839 = select i1 %.not36, float %1832, float %1826, !dbg !7
  %1840 = bitcast float %1839 to i32, !dbg !7
  %1841 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1840), !dbg !7
  %1842 = bitcast i32 %1841 to float, !dbg !7
  %1843 = select i1 %.not36, float %1826, float %1842, !dbg !7
  %1844 = select i1 %.not36, float %1842, float %1832, !dbg !7
  %1845 = extractvalue [2 x <8 x float>] %206, 0, !dbg !7
  %1846 = extractelement <8 x float> %1845, i64 0, !dbg !7
  %1847 = extractelement <8 x float> %1845, i64 1, !dbg !7
  %1848 = extractelement <8 x float> %1845, i64 2, !dbg !7
  %1849 = extractelement <8 x float> %1845, i64 3, !dbg !7
  %1850 = select i1 %.not, float %1847, float %1846, !dbg !7
  %1851 = bitcast float %1850 to i32, !dbg !7
  %1852 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1851), !dbg !7
  %1853 = bitcast i32 %1852 to float, !dbg !7
  %1854 = select i1 %.not, float %1846, float %1853, !dbg !7
  %1855 = select i1 %.not, float %1853, float %1847, !dbg !7
  %1856 = select i1 %.not, float %1849, float %1848, !dbg !7
  %1857 = bitcast float %1856 to i32, !dbg !7
  %1858 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1857), !dbg !7
  %1859 = bitcast i32 %1858 to float, !dbg !7
  %1860 = select i1 %.not, float %1848, float %1859, !dbg !7
  %1861 = select i1 %.not, float %1859, float %1849, !dbg !7
  %1862 = select i1 %.not36, float %1860, float %1854, !dbg !7
  %1863 = bitcast float %1862 to i32, !dbg !7
  %1864 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1863), !dbg !7
  %1865 = select i1 %.not36, float %1861, float %1855, !dbg !7
  %1866 = bitcast float %1865 to i32, !dbg !7
  %1867 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1866), !dbg !7
  %1868 = extractvalue [2 x <1 x float>] %207, 0, !dbg !19
  %1869 = shufflevector <1 x float> %1868, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1870 = extractelement <8 x float> %1746, i64 4, !dbg !7
  %1871 = extractelement <8 x float> %1746, i64 5, !dbg !7
  %1872 = extractelement <8 x float> %1746, i64 6, !dbg !7
  %1873 = extractelement <8 x float> %1746, i64 7, !dbg !7
  %1874 = select i1 %.not, float %1871, float %1870, !dbg !7
  %1875 = bitcast float %1874 to i32, !dbg !7
  %1876 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1875), !dbg !7
  %1877 = bitcast i32 %1876 to float, !dbg !7
  %1878 = select i1 %.not, float %1870, float %1877, !dbg !7
  %1879 = select i1 %.not, float %1877, float %1871, !dbg !7
  %1880 = select i1 %.not, float %1873, float %1872, !dbg !7
  %1881 = bitcast float %1880 to i32, !dbg !7
  %1882 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1881), !dbg !7
  %1883 = bitcast i32 %1882 to float, !dbg !7
  %1884 = select i1 %.not, float %1872, float %1883, !dbg !7
  %1885 = select i1 %.not, float %1883, float %1873, !dbg !7
  %1886 = select i1 %.not36, float %1884, float %1878, !dbg !7
  %1887 = bitcast float %1886 to i32, !dbg !7
  %1888 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1887), !dbg !7
  %1889 = bitcast i32 %1888 to float, !dbg !7
  %1890 = select i1 %.not36, float %1878, float %1889, !dbg !7
  %1891 = select i1 %.not36, float %1889, float %1884, !dbg !7
  %1892 = select i1 %.not36, float %1885, float %1879, !dbg !7
  %1893 = bitcast float %1892 to i32, !dbg !7
  %1894 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1893), !dbg !7
  %1895 = bitcast i32 %1894 to float, !dbg !7
  %1896 = select i1 %.not36, float %1879, float %1895, !dbg !7
  %1897 = select i1 %.not36, float %1895, float %1885, !dbg !7
  %1898 = extractelement <8 x float> %1787, i64 4, !dbg !7
  %1899 = extractelement <8 x float> %1787, i64 5, !dbg !7
  %1900 = extractelement <8 x float> %1787, i64 6, !dbg !7
  %1901 = extractelement <8 x float> %1787, i64 7, !dbg !7
  %1902 = select i1 %.not, float %1899, float %1898, !dbg !7
  %1903 = bitcast float %1902 to i32, !dbg !7
  %1904 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1903), !dbg !7
  %1905 = bitcast i32 %1904 to float, !dbg !7
  %1906 = select i1 %.not, float %1898, float %1905, !dbg !7
  %1907 = select i1 %.not, float %1905, float %1899, !dbg !7
  %1908 = select i1 %.not, float %1901, float %1900, !dbg !7
  %1909 = bitcast float %1908 to i32, !dbg !7
  %1910 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1909), !dbg !7
  %1911 = bitcast i32 %1910 to float, !dbg !7
  %1912 = select i1 %.not, float %1900, float %1911, !dbg !7
  %1913 = select i1 %.not, float %1911, float %1901, !dbg !7
  %1914 = select i1 %.not36, float %1912, float %1906, !dbg !7
  %1915 = bitcast float %1914 to i32, !dbg !7
  %1916 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1915), !dbg !7
  %1917 = bitcast i32 %1916 to float, !dbg !7
  %1918 = select i1 %.not36, float %1906, float %1917, !dbg !7
  %1919 = select i1 %.not36, float %1917, float %1912, !dbg !7
  %1920 = select i1 %.not36, float %1913, float %1907, !dbg !7
  %1921 = bitcast float %1920 to i32, !dbg !7
  %1922 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1921), !dbg !7
  %1923 = bitcast i32 %1922 to float, !dbg !7
  %1924 = select i1 %.not36, float %1907, float %1923, !dbg !7
  %1925 = select i1 %.not36, float %1923, float %1913, !dbg !7
  %1926 = extractelement <8 x float> %1816, i64 4, !dbg !7
  %1927 = extractelement <8 x float> %1816, i64 5, !dbg !7
  %1928 = extractelement <8 x float> %1816, i64 6, !dbg !7
  %1929 = extractelement <8 x float> %1816, i64 7, !dbg !7
  %1930 = select i1 %.not, float %1927, float %1926, !dbg !7
  %1931 = bitcast float %1930 to i32, !dbg !7
  %1932 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1931), !dbg !7
  %1933 = bitcast i32 %1932 to float, !dbg !7
  %1934 = select i1 %.not, float %1926, float %1933, !dbg !7
  %1935 = select i1 %.not, float %1933, float %1927, !dbg !7
  %1936 = select i1 %.not, float %1929, float %1928, !dbg !7
  %1937 = bitcast float %1936 to i32, !dbg !7
  %1938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1937), !dbg !7
  %1939 = bitcast i32 %1938 to float, !dbg !7
  %1940 = select i1 %.not, float %1928, float %1939, !dbg !7
  %1941 = select i1 %.not, float %1939, float %1929, !dbg !7
  %1942 = select i1 %.not36, float %1940, float %1934, !dbg !7
  %1943 = bitcast float %1942 to i32, !dbg !7
  %1944 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1943), !dbg !7
  %1945 = bitcast i32 %1944 to float, !dbg !7
  %1946 = select i1 %.not36, float %1934, float %1945, !dbg !7
  %1947 = select i1 %.not36, float %1945, float %1940, !dbg !7
  %1948 = select i1 %.not36, float %1941, float %1935, !dbg !7
  %1949 = bitcast float %1948 to i32, !dbg !7
  %1950 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1949), !dbg !7
  %1951 = bitcast i32 %1950 to float, !dbg !7
  %1952 = select i1 %.not36, float %1935, float %1951, !dbg !7
  %1953 = select i1 %.not36, float %1951, float %1941, !dbg !7
  %1954 = extractelement <8 x float> %1845, i64 4, !dbg !7
  %1955 = extractelement <8 x float> %1845, i64 5, !dbg !7
  %1956 = extractelement <8 x float> %1845, i64 6, !dbg !7
  %1957 = extractelement <8 x float> %1845, i64 7, !dbg !7
  %1958 = select i1 %.not, float %1955, float %1954, !dbg !7
  %1959 = bitcast float %1958 to i32, !dbg !7
  %1960 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1959), !dbg !7
  %1961 = bitcast i32 %1960 to float, !dbg !7
  %1962 = select i1 %.not, float %1954, float %1961, !dbg !7
  %1963 = select i1 %.not, float %1961, float %1955, !dbg !7
  %1964 = select i1 %.not, float %1957, float %1956, !dbg !7
  %1965 = bitcast float %1964 to i32, !dbg !7
  %1966 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1965), !dbg !7
  %1967 = bitcast i32 %1966 to float, !dbg !7
  %1968 = select i1 %.not, float %1956, float %1967, !dbg !7
  %1969 = select i1 %.not, float %1967, float %1957, !dbg !7
  %1970 = select i1 %.not36, float %1968, float %1962, !dbg !7
  %1971 = bitcast float %1970 to i32, !dbg !7
  %1972 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1971), !dbg !7
  %1973 = select i1 %.not36, float %1969, float %1963, !dbg !7
  %1974 = bitcast float %1973 to i32, !dbg !7
  %1975 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1974), !dbg !7
  %1976 = extractvalue [2 x <8 x float>] %211, 1, !dbg !7
  %1977 = extractelement <8 x float> %1976, i64 0, !dbg !7
  %1978 = extractelement <8 x float> %1976, i64 1, !dbg !7
  %1979 = extractelement <8 x float> %1976, i64 2, !dbg !7
  %1980 = extractelement <8 x float> %1976, i64 3, !dbg !7
  %1981 = select i1 %.not, float %1978, float %1977, !dbg !7
  %1982 = bitcast float %1981 to i32, !dbg !7
  %1983 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1982), !dbg !7
  %1984 = bitcast i32 %1983 to float, !dbg !7
  %1985 = select i1 %.not, float %1977, float %1984, !dbg !7
  %1986 = select i1 %.not, float %1984, float %1978, !dbg !7
  %1987 = select i1 %.not, float %1980, float %1979, !dbg !7
  %1988 = bitcast float %1987 to i32, !dbg !7
  %1989 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %1988), !dbg !7
  %1990 = bitcast i32 %1989 to float, !dbg !7
  %1991 = select i1 %.not, float %1979, float %1990, !dbg !7
  %1992 = select i1 %.not, float %1990, float %1980, !dbg !7
  %1993 = select i1 %.not36, float %1991, float %1985, !dbg !7
  %1994 = bitcast float %1993 to i32, !dbg !7
  %1995 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %1994), !dbg !7
  %1996 = bitcast i32 %1995 to float, !dbg !7
  %1997 = select i1 %.not36, float %1985, float %1996, !dbg !7
  %1998 = select i1 %.not36, float %1996, float %1991, !dbg !7
  %1999 = select i1 %.not36, float %1992, float %1986, !dbg !7
  %2000 = bitcast float %1999 to i32, !dbg !7
  %2001 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2000), !dbg !7
  %2002 = bitcast i32 %2001 to float, !dbg !7
  %2003 = select i1 %.not36, float %1986, float %2002, !dbg !7
  %2004 = select i1 %.not36, float %2002, float %1992, !dbg !7
  %2005 = extractvalue [2 x <8 x float>] %210, 1, !dbg !7
  %2006 = extractelement <8 x float> %2005, i64 0, !dbg !7
  %2007 = extractelement <8 x float> %2005, i64 1, !dbg !7
  %2008 = extractelement <8 x float> %2005, i64 2, !dbg !7
  %2009 = extractelement <8 x float> %2005, i64 3, !dbg !7
  %2010 = select i1 %.not, float %2007, float %2006, !dbg !7
  %2011 = bitcast float %2010 to i32, !dbg !7
  %2012 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %2011), !dbg !7
  %2013 = bitcast i32 %2012 to float, !dbg !7
  %2014 = select i1 %.not, float %2006, float %2013, !dbg !7
  %2015 = select i1 %.not, float %2013, float %2007, !dbg !7
  %2016 = select i1 %.not, float %2009, float %2008, !dbg !7
  %2017 = bitcast float %2016 to i32, !dbg !7
  %2018 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %2017), !dbg !7
  %2019 = bitcast i32 %2018 to float, !dbg !7
  %2020 = select i1 %.not, float %2008, float %2019, !dbg !7
  %2021 = select i1 %.not, float %2019, float %2009, !dbg !7
  %2022 = select i1 %.not36, float %2020, float %2014, !dbg !7
  %2023 = bitcast float %2022 to i32, !dbg !7
  %2024 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2023), !dbg !7
  %2025 = bitcast i32 %2024 to float, !dbg !7
  %2026 = select i1 %.not36, float %2014, float %2025, !dbg !7
  %2027 = select i1 %.not36, float %2025, float %2020, !dbg !7
  %2028 = select i1 %.not36, float %2021, float %2015, !dbg !7
  %2029 = bitcast float %2028 to i32, !dbg !7
  %2030 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2029), !dbg !7
  %2031 = bitcast i32 %2030 to float, !dbg !7
  %2032 = select i1 %.not36, float %2015, float %2031, !dbg !7
  %2033 = select i1 %.not36, float %2031, float %2021, !dbg !7
  %2034 = extractvalue [2 x <8 x float>] %209, 1, !dbg !7
  %2035 = extractelement <8 x float> %2034, i64 0, !dbg !7
  %2036 = extractelement <8 x float> %2034, i64 1, !dbg !7
  %2037 = extractelement <8 x float> %2034, i64 2, !dbg !7
  %2038 = extractelement <8 x float> %2034, i64 3, !dbg !7
  %2039 = select i1 %.not, float %2036, float %2035, !dbg !7
  %2040 = bitcast float %2039 to i32, !dbg !7
  %2041 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %2040), !dbg !7
  %2042 = bitcast i32 %2041 to float, !dbg !7
  %2043 = select i1 %.not, float %2035, float %2042, !dbg !7
  %2044 = select i1 %.not, float %2042, float %2036, !dbg !7
  %2045 = select i1 %.not, float %2038, float %2037, !dbg !7
  %2046 = bitcast float %2045 to i32, !dbg !7
  %2047 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %2046), !dbg !7
  %2048 = bitcast i32 %2047 to float, !dbg !7
  %2049 = select i1 %.not, float %2037, float %2048, !dbg !7
  %2050 = select i1 %.not, float %2048, float %2038, !dbg !7
  %2051 = select i1 %.not36, float %2049, float %2043, !dbg !7
  %2052 = bitcast float %2051 to i32, !dbg !7
  %2053 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2052), !dbg !7
  %2054 = bitcast i32 %2053 to float, !dbg !7
  %2055 = select i1 %.not36, float %2043, float %2054, !dbg !7
  %2056 = select i1 %.not36, float %2054, float %2049, !dbg !7
  %2057 = select i1 %.not36, float %2050, float %2044, !dbg !7
  %2058 = bitcast float %2057 to i32, !dbg !7
  %2059 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2058), !dbg !7
  %2060 = bitcast i32 %2059 to float, !dbg !7
  %2061 = select i1 %.not36, float %2044, float %2060, !dbg !7
  %2062 = select i1 %.not36, float %2060, float %2050, !dbg !7
  %2063 = extractvalue [2 x <8 x float>] %206, 1, !dbg !7
  %2064 = extractelement <8 x float> %2063, i64 0, !dbg !7
  %2065 = extractelement <8 x float> %2063, i64 1, !dbg !7
  %2066 = extractelement <8 x float> %2063, i64 2, !dbg !7
  %2067 = extractelement <8 x float> %2063, i64 3, !dbg !7
  %2068 = select i1 %.not, float %2065, float %2064, !dbg !7
  %2069 = bitcast float %2068 to i32, !dbg !7
  %2070 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %2069), !dbg !7
  %2071 = bitcast i32 %2070 to float, !dbg !7
  %2072 = select i1 %.not, float %2064, float %2071, !dbg !7
  %2073 = select i1 %.not, float %2071, float %2065, !dbg !7
  %2074 = select i1 %.not, float %2067, float %2066, !dbg !7
  %2075 = bitcast float %2074 to i32, !dbg !7
  %2076 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %2075), !dbg !7
  %2077 = bitcast i32 %2076 to float, !dbg !7
  %2078 = select i1 %.not, float %2066, float %2077, !dbg !7
  %2079 = select i1 %.not, float %2077, float %2067, !dbg !7
  %2080 = select i1 %.not36, float %2078, float %2072, !dbg !7
  %2081 = bitcast float %2080 to i32, !dbg !7
  %2082 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2081), !dbg !7
  %2083 = select i1 %.not36, float %2079, float %2073, !dbg !7
  %2084 = bitcast float %2083 to i32, !dbg !7
  %2085 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2084), !dbg !7
  %2086 = extractvalue [2 x <1 x float>] %207, 1, !dbg !19
  %2087 = shufflevector <1 x float> %2086, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %2088 = extractelement <8 x float> %1976, i64 4, !dbg !7
  %2089 = extractelement <8 x float> %1976, i64 5, !dbg !7
  %2090 = extractelement <8 x float> %1976, i64 6, !dbg !7
  %2091 = extractelement <8 x float> %1976, i64 7, !dbg !7
  %2092 = select i1 %.not, float %2089, float %2088, !dbg !7
  %2093 = bitcast float %2092 to i32, !dbg !7
  %2094 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %2093), !dbg !7
  %2095 = bitcast i32 %2094 to float, !dbg !7
  %2096 = select i1 %.not, float %2088, float %2095, !dbg !7
  %2097 = select i1 %.not, float %2095, float %2089, !dbg !7
  %2098 = select i1 %.not, float %2091, float %2090, !dbg !7
  %2099 = bitcast float %2098 to i32, !dbg !7
  %2100 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %2099), !dbg !7
  %2101 = bitcast i32 %2100 to float, !dbg !7
  %2102 = select i1 %.not, float %2090, float %2101, !dbg !7
  %2103 = select i1 %.not, float %2101, float %2091, !dbg !7
  %2104 = select i1 %.not36, float %2102, float %2096, !dbg !7
  %2105 = bitcast float %2104 to i32, !dbg !7
  %2106 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2105), !dbg !7
  %2107 = bitcast i32 %2106 to float, !dbg !7
  %2108 = select i1 %.not36, float %2096, float %2107, !dbg !7
  %2109 = select i1 %.not36, float %2107, float %2102, !dbg !7
  %2110 = select i1 %.not36, float %2103, float %2097, !dbg !7
  %2111 = bitcast float %2110 to i32, !dbg !7
  %2112 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2111), !dbg !7
  %2113 = bitcast i32 %2112 to float, !dbg !7
  %2114 = select i1 %.not36, float %2097, float %2113, !dbg !7
  %2115 = select i1 %.not36, float %2113, float %2103, !dbg !7
  %2116 = extractelement <8 x float> %2005, i64 4, !dbg !7
  %2117 = extractelement <8 x float> %2005, i64 5, !dbg !7
  %2118 = extractelement <8 x float> %2005, i64 6, !dbg !7
  %2119 = extractelement <8 x float> %2005, i64 7, !dbg !7
  %2120 = select i1 %.not, float %2117, float %2116, !dbg !7
  %2121 = bitcast float %2120 to i32, !dbg !7
  %2122 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %2121), !dbg !7
  %2123 = bitcast i32 %2122 to float, !dbg !7
  %2124 = select i1 %.not, float %2116, float %2123, !dbg !7
  %2125 = select i1 %.not, float %2123, float %2117, !dbg !7
  %2126 = select i1 %.not, float %2119, float %2118, !dbg !7
  %2127 = bitcast float %2126 to i32, !dbg !7
  %2128 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %2127), !dbg !7
  %2129 = bitcast i32 %2128 to float, !dbg !7
  %2130 = select i1 %.not, float %2118, float %2129, !dbg !7
  %2131 = select i1 %.not, float %2129, float %2119, !dbg !7
  %2132 = select i1 %.not36, float %2130, float %2124, !dbg !7
  %2133 = bitcast float %2132 to i32, !dbg !7
  %2134 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2133), !dbg !7
  %2135 = bitcast i32 %2134 to float, !dbg !7
  %2136 = select i1 %.not36, float %2124, float %2135, !dbg !7
  %2137 = select i1 %.not36, float %2135, float %2130, !dbg !7
  %2138 = select i1 %.not36, float %2131, float %2125, !dbg !7
  %2139 = bitcast float %2138 to i32, !dbg !7
  %2140 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2139), !dbg !7
  %2141 = bitcast i32 %2140 to float, !dbg !7
  %2142 = select i1 %.not36, float %2125, float %2141, !dbg !7
  %2143 = select i1 %.not36, float %2141, float %2131, !dbg !7
  %2144 = extractelement <8 x float> %2034, i64 4, !dbg !7
  %2145 = extractelement <8 x float> %2034, i64 5, !dbg !7
  %2146 = extractelement <8 x float> %2034, i64 6, !dbg !7
  %2147 = extractelement <8 x float> %2034, i64 7, !dbg !7
  %2148 = select i1 %.not, float %2145, float %2144, !dbg !7
  %2149 = bitcast float %2148 to i32, !dbg !7
  %2150 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %2149), !dbg !7
  %2151 = bitcast i32 %2150 to float, !dbg !7
  %2152 = select i1 %.not, float %2144, float %2151, !dbg !7
  %2153 = select i1 %.not, float %2151, float %2145, !dbg !7
  %2154 = select i1 %.not, float %2147, float %2146, !dbg !7
  %2155 = bitcast float %2154 to i32, !dbg !7
  %2156 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %2155), !dbg !7
  %2157 = bitcast i32 %2156 to float, !dbg !7
  %2158 = select i1 %.not, float %2146, float %2157, !dbg !7
  %2159 = select i1 %.not, float %2157, float %2147, !dbg !7
  %2160 = select i1 %.not36, float %2158, float %2152, !dbg !7
  %2161 = bitcast float %2160 to i32, !dbg !7
  %2162 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2161), !dbg !7
  %2163 = bitcast i32 %2162 to float, !dbg !7
  %2164 = select i1 %.not36, float %2152, float %2163, !dbg !7
  %2165 = select i1 %.not36, float %2163, float %2158, !dbg !7
  %2166 = select i1 %.not36, float %2159, float %2153, !dbg !7
  %2167 = bitcast float %2166 to i32, !dbg !7
  %2168 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2167), !dbg !7
  %2169 = bitcast i32 %2168 to float, !dbg !7
  %2170 = select i1 %.not36, float %2153, float %2169, !dbg !7
  %2171 = select i1 %.not36, float %2169, float %2159, !dbg !7
  %2172 = extractelement <8 x float> %2063, i64 4, !dbg !7
  %2173 = extractelement <8 x float> %2063, i64 5, !dbg !7
  %2174 = extractelement <8 x float> %2063, i64 6, !dbg !7
  %2175 = extractelement <8 x float> %2063, i64 7, !dbg !7
  %2176 = select i1 %.not, float %2173, float %2172, !dbg !7
  %2177 = bitcast float %2176 to i32, !dbg !7
  %2178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %2177), !dbg !7
  %2179 = bitcast i32 %2178 to float, !dbg !7
  %2180 = select i1 %.not, float %2172, float %2179, !dbg !7
  %2181 = select i1 %.not, float %2179, float %2173, !dbg !7
  %2182 = select i1 %.not, float %2175, float %2174, !dbg !7
  %2183 = bitcast float %2182 to i32, !dbg !7
  %2184 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1759, i32 %2183), !dbg !7
  %2185 = bitcast i32 %2184 to float, !dbg !7
  %2186 = select i1 %.not, float %2174, float %2185, !dbg !7
  %2187 = select i1 %.not, float %2185, float %2175, !dbg !7
  %2188 = select i1 %.not36, float %2186, float %2180, !dbg !7
  %2189 = bitcast float %2188 to i32, !dbg !7
  %2190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2189), !dbg !7
  %2191 = select i1 %.not36, float %2187, float %2181, !dbg !7
  %2192 = bitcast float %2191 to i32, !dbg !7
  %2193 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1775, i32 %2192), !dbg !7
  br i1 true, label %2194, label %2258, !dbg !21

2194:                                             ; preds = %1743
  %2195 = bitcast i32 %2190 to float, !dbg !7
  %2196 = select i1 %.not36, float %2180, float %2195, !dbg !7
  %2197 = insertelement <4 x float> poison, float %2196, i64 0, !dbg !19
  %2198 = bitcast i32 %2193 to float, !dbg !7
  %2199 = select i1 %.not36, float %2181, float %2198, !dbg !7
  %2200 = insertelement <4 x float> %2197, float %2199, i64 1, !dbg !19
  %2201 = select i1 %.not36, float %2195, float %2186, !dbg !7
  %2202 = insertelement <4 x float> %2200, float %2201, i64 2, !dbg !19
  %2203 = select i1 %.not36, float %2198, float %2187, !dbg !7
  %2204 = insertelement <4 x float> %2202, float %2203, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2087, !dbg !19
  %normalized = fmul <4 x float> %2204, %row.reciprocal, !dbg !19
  %2205 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %2206 = extractelement <4 x half> %2205, i64 3, !dbg !20
  %2207 = extractelement <4 x half> %2205, i64 2, !dbg !20
  %2208 = extractelement <4 x half> %2205, i64 1, !dbg !20
  %2209 = extractelement <4 x half> %2205, i64 0, !dbg !20
  %2210 = bitcast i32 %2082 to float, !dbg !7
  %2211 = select i1 %.not36, float %2072, float %2210, !dbg !7
  %2212 = insertelement <4 x float> poison, float %2211, i64 0, !dbg !19
  %2213 = bitcast i32 %2085 to float, !dbg !7
  %2214 = select i1 %.not36, float %2073, float %2213, !dbg !7
  %2215 = insertelement <4 x float> %2212, float %2214, i64 1, !dbg !19
  %2216 = select i1 %.not36, float %2210, float %2078, !dbg !7
  %2217 = insertelement <4 x float> %2215, float %2216, i64 2, !dbg !19
  %2218 = select i1 %.not36, float %2213, float %2079, !dbg !7
  %2219 = insertelement <4 x float> %2217, float %2218, i64 3, !dbg !19
  %normalized104 = fmul <4 x float> %2219, %row.reciprocal, !dbg !19
  %2220 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !20
  %2221 = extractelement <4 x half> %2220, i64 3, !dbg !20
  %2222 = extractelement <4 x half> %2220, i64 2, !dbg !20
  %2223 = extractelement <4 x half> %2220, i64 1, !dbg !20
  %2224 = extractelement <4 x half> %2220, i64 0, !dbg !20
  %2225 = bitcast i32 %1972 to float, !dbg !7
  %2226 = select i1 %.not36, float %1962, float %2225, !dbg !7
  %2227 = insertelement <4 x float> poison, float %2226, i64 0, !dbg !19
  %2228 = bitcast i32 %1975 to float, !dbg !7
  %2229 = select i1 %.not36, float %1963, float %2228, !dbg !7
  %2230 = insertelement <4 x float> %2227, float %2229, i64 1, !dbg !19
  %2231 = select i1 %.not36, float %2225, float %1968, !dbg !7
  %2232 = insertelement <4 x float> %2230, float %2231, i64 2, !dbg !19
  %2233 = select i1 %.not36, float %2228, float %1969, !dbg !7
  %2234 = insertelement <4 x float> %2232, float %2233, i64 3, !dbg !19
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1869, !dbg !19
  %normalized106 = fmul <4 x float> %2234, %row.reciprocal105, !dbg !19
  %2235 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !20
  %2236 = extractelement <4 x half> %2235, i64 3, !dbg !20
  %2237 = extractelement <4 x half> %2235, i64 2, !dbg !20
  %2238 = extractelement <4 x half> %2235, i64 1, !dbg !20
  %2239 = extractelement <4 x half> %2235, i64 0, !dbg !20
  %2240 = bitcast i32 %1864 to float, !dbg !7
  %2241 = select i1 %.not36, float %1854, float %2240, !dbg !7
  %2242 = insertelement <4 x float> poison, float %2241, i64 0, !dbg !19
  %2243 = bitcast i32 %1867 to float, !dbg !7
  %2244 = select i1 %.not36, float %1855, float %2243, !dbg !7
  %2245 = insertelement <4 x float> %2242, float %2244, i64 1, !dbg !19
  %2246 = select i1 %.not36, float %2240, float %1860, !dbg !7
  %2247 = insertelement <4 x float> %2245, float %2246, i64 2, !dbg !19
  %2248 = select i1 %.not36, float %2243, float %1861, !dbg !7
  %2249 = insertelement <4 x float> %2247, float %2248, i64 3, !dbg !19
  %normalized107 = fmul <4 x float> %2249, %row.reciprocal105, !dbg !19
  %2250 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !20
  %2251 = extractelement <4 x half> %2250, i64 3, !dbg !20
  %2252 = extractelement <4 x half> %2250, i64 2, !dbg !20
  %2253 = extractelement <4 x half> %2250, i64 1, !dbg !20
  %2254 = extractelement <4 x half> %2250, i64 0, !dbg !20
  %2255 = add nsw i64 %16, %22, !dbg !21
  %2256 = shl i64 %2255, 19, !dbg !21
  %2257 = and i64 %2256, 16252928, !dbg !21
  %address.thread967 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local968 = zext nneg i32 %address.thread967 to i64, !dbg !8
  %address.local969 = add i64 %25, %address.local968, !dbg !21
  %address.local970 = lshr i64 %address.local968, 6, !dbg !11
  %address.local971 = shl nuw nsw i64 %address.local970, 4, !dbg !21
  %address.local972 = add i64 %address.local969, %address.local971, !dbg !21
  %address.local973 = and i64 %address.local968, 240, !dbg !21
  %address.local974 = sub i64 %address.local972, %address.local973, !dbg !21
  %address.local975 = add i64 %address.local974, %26, !dbg !21
  %address.local976 = shl nuw nsw i64 %address.local975, 7, !dbg !21
  %address.local977 = add nuw nsw i64 %2257, %address.local976, !dbg !21
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local979 = lshr i64 %address.local968, 2, !dbg !11
  %address.local980 = and i64 %address.local979, 12, !dbg !11
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local978, i64 %address.local980, !dbg !21
  store half %2254, ptr addrspace(1) %address.local981, align 2, !dbg !21
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local983 = or disjoint i64 %address.local980, 1, !dbg !21
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local983, !dbg !21
  store half %2253, ptr addrspace(1) %address.local984, align 2, !dbg !21
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local986 = or disjoint i64 %address.local980, 2, !dbg !21
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local986, !dbg !21
  store half %2252, ptr addrspace(1) %address.local987, align 2, !dbg !21
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local989 = or disjoint i64 %address.local980, 3, !dbg !21
  %address.local990 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local989, !dbg !21
  store half %2251, ptr addrspace(1) %address.local990, align 2, !dbg !21
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local992 = or disjoint i64 %address.local980, 16, !dbg !21
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local991, i64 %address.local992, !dbg !21
  store half %2239, ptr addrspace(1) %address.local993, align 2, !dbg !21
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local995 = or disjoint i64 %address.local980, 17, !dbg !21
  %address.local996 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local994, i64 %address.local995, !dbg !21
  store half %2238, ptr addrspace(1) %address.local996, align 2, !dbg !21
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local998 = or disjoint i64 %address.local980, 18, !dbg !21
  %address.local999 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local998, !dbg !21
  store half %2237, ptr addrspace(1) %address.local999, align 2, !dbg !21
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local977, !dbg !21
  %address.local1001 = or disjoint i64 %address.local980, 19, !dbg !21
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1000, i64 %address.local1001, !dbg !21
  store half %2236, ptr addrspace(1) %address.local1002, align 2, !dbg !21
  %address.local1003 = shl i64 %address.local975, 7, !dbg !21
  %address.local1004 = add i64 %address.local1003, 8192, !dbg !21
  %address.local1005 = add nuw nsw i64 %2257, %address.local1004, !dbg !21
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !21
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1006, i64 %address.local980, !dbg !21
  store half %2224, ptr addrspace(1) %address.local1007, align 2, !dbg !21
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !21
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1008, i64 %address.local983, !dbg !21
  store half %2223, ptr addrspace(1) %address.local1009, align 2, !dbg !21
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !21
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local986, !dbg !21
  store half %2222, ptr addrspace(1) %address.local1011, align 2, !dbg !21
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !21
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1012, i64 %address.local989, !dbg !21
  store half %2221, ptr addrspace(1) %address.local1013, align 2, !dbg !21
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !21
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1014, i64 %address.local992, !dbg !21
  store half %2209, ptr addrspace(1) %address.local1015, align 2, !dbg !21
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !21
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local995, !dbg !21
  store half %2208, ptr addrspace(1) %address.local1017, align 2, !dbg !21
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !21
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1018, i64 %address.local998, !dbg !21
  store half %2207, ptr addrspace(1) %address.local1019, align 2, !dbg !21
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !21
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1020, i64 %address.local1001, !dbg !21
  store half %2206, ptr addrspace(1) %address.local1021, align 2, !dbg !21
  br label %2258, !dbg !21

2258:                                             ; preds = %2194, %1743
  br i1 true, label %2259, label %2297, !dbg !21

2259:                                             ; preds = %2258
  %2260 = insertelement <4 x float> poison, float %2164, i64 0, !dbg !19
  %2261 = insertelement <4 x float> %2260, float %2170, i64 1, !dbg !19
  %2262 = insertelement <4 x float> %2261, float %2165, i64 2, !dbg !19
  %2263 = insertelement <4 x float> %2262, float %2171, i64 3, !dbg !19
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2087, !dbg !19
  %normalized109 = fmul <4 x float> %2263, %row.reciprocal108, !dbg !19
  %2264 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !20
  %2265 = extractelement <4 x half> %2264, i64 3, !dbg !20
  %2266 = extractelement <4 x half> %2264, i64 2, !dbg !20
  %2267 = extractelement <4 x half> %2264, i64 1, !dbg !20
  %2268 = extractelement <4 x half> %2264, i64 0, !dbg !20
  %2269 = insertelement <4 x float> poison, float %2055, i64 0, !dbg !19
  %2270 = insertelement <4 x float> %2269, float %2061, i64 1, !dbg !19
  %2271 = insertelement <4 x float> %2270, float %2056, i64 2, !dbg !19
  %2272 = insertelement <4 x float> %2271, float %2062, i64 3, !dbg !19
  %normalized110 = fmul <4 x float> %2272, %row.reciprocal108, !dbg !19
  %2273 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %2274 = extractelement <4 x half> %2273, i64 3, !dbg !20
  %2275 = extractelement <4 x half> %2273, i64 2, !dbg !20
  %2276 = extractelement <4 x half> %2273, i64 1, !dbg !20
  %2277 = extractelement <4 x half> %2273, i64 0, !dbg !20
  %2278 = insertelement <4 x float> poison, float %1946, i64 0, !dbg !19
  %2279 = insertelement <4 x float> %2278, float %1952, i64 1, !dbg !19
  %2280 = insertelement <4 x float> %2279, float %1947, i64 2, !dbg !19
  %2281 = insertelement <4 x float> %2280, float %1953, i64 3, !dbg !19
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1869, !dbg !19
  %normalized112 = fmul <4 x float> %2281, %row.reciprocal111, !dbg !19
  %2282 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !20
  %2283 = extractelement <4 x half> %2282, i64 3, !dbg !20
  %2284 = extractelement <4 x half> %2282, i64 2, !dbg !20
  %2285 = extractelement <4 x half> %2282, i64 1, !dbg !20
  %2286 = extractelement <4 x half> %2282, i64 0, !dbg !20
  %2287 = insertelement <4 x float> poison, float %1837, i64 0, !dbg !19
  %2288 = insertelement <4 x float> %2287, float %1843, i64 1, !dbg !19
  %2289 = insertelement <4 x float> %2288, float %1838, i64 2, !dbg !19
  %2290 = insertelement <4 x float> %2289, float %1844, i64 3, !dbg !19
  %normalized113 = fmul <4 x float> %2290, %row.reciprocal111, !dbg !19
  %2291 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %2292 = extractelement <4 x half> %2291, i64 3, !dbg !20
  %2293 = extractelement <4 x half> %2291, i64 2, !dbg !20
  %2294 = extractelement <4 x half> %2291, i64 1, !dbg !20
  %2295 = extractelement <4 x half> %2291, i64 0, !dbg !20
  %2296 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1022 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1023 = zext nneg i32 %address.thread1022 to i64, !dbg !8
  %address.local1024 = add i64 %25, %address.local1023, !dbg !21
  %address.local1025 = lshr i64 %address.local1023, 6, !dbg !11
  %address.local1026 = shl nuw nsw i64 %address.local1025, 4, !dbg !21
  %address.local1027 = add i64 %address.local1024, %address.local1026, !dbg !21
  %address.local1028 = and i64 %address.local1023, 240, !dbg !21
  %address.local1029 = sub i64 %address.local1027, %address.local1028, !dbg !21
  %address.local1030 = add i64 %address.local1029, %26, !dbg !21
  %address.local1031 = shl nuw nsw i64 %address.local1030, 7, !dbg !21
  %address.local1032 = add nuw nsw i64 %2296, %address.local1031, !dbg !21
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !21
  %address.local1034 = lshr i64 %address.local1023, 2, !dbg !11
  %address.local1035 = and i64 %address.local1034, 12, !dbg !11
  %address.local1036 = or disjoint i64 %address.local1035, 32, !dbg !21
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local1036, !dbg !21
  store half %2295, ptr addrspace(1) %address.local1037, align 2, !dbg !21
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !21
  %address.local1039 = or disjoint i64 %address.local1035, 33, !dbg !21
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1038, i64 %address.local1039, !dbg !21
  store half %2294, ptr addrspace(1) %address.local1040, align 2, !dbg !21
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !21
  %address.local1042 = or disjoint i64 %address.local1035, 34, !dbg !21
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1041, i64 %address.local1042, !dbg !21
  store half %2293, ptr addrspace(1) %address.local1043, align 2, !dbg !21
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !21
  %address.local1045 = or disjoint i64 %address.local1035, 35, !dbg !21
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1044, i64 %address.local1045, !dbg !21
  store half %2292, ptr addrspace(1) %address.local1046, align 2, !dbg !21
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !21
  %address.local1048 = or disjoint i64 %address.local1035, 48, !dbg !21
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1047, i64 %address.local1048, !dbg !21
  store half %2286, ptr addrspace(1) %address.local1049, align 2, !dbg !21
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !21
  %address.local1051 = or disjoint i64 %address.local1035, 49, !dbg !21
  %address.local1052 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1051, !dbg !21
  store half %2285, ptr addrspace(1) %address.local1052, align 2, !dbg !21
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !21
  %address.local1054 = or disjoint i64 %address.local1035, 50, !dbg !21
  %address.local1055 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1054, !dbg !21
  store half %2284, ptr addrspace(1) %address.local1055, align 2, !dbg !21
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !21
  %address.local1057 = or i64 %address.local1034, 51, !dbg !21
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1057, !dbg !21
  store half %2283, ptr addrspace(1) %address.local1058, align 2, !dbg !21
  %address.local1059 = shl i64 %address.local1030, 7, !dbg !21
  %address.local1060 = add i64 %address.local1059, 8192, !dbg !21
  %address.local1061 = add nuw nsw i64 %2296, %address.local1060, !dbg !21
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !21
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1062, i64 %address.local1036, !dbg !21
  store half %2277, ptr addrspace(1) %address.local1063, align 2, !dbg !21
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !21
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1064, i64 %address.local1039, !dbg !21
  store half %2276, ptr addrspace(1) %address.local1065, align 2, !dbg !21
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !21
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1066, i64 %address.local1042, !dbg !21
  store half %2275, ptr addrspace(1) %address.local1067, align 2, !dbg !21
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !21
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1068, i64 %address.local1045, !dbg !21
  store half %2274, ptr addrspace(1) %address.local1069, align 2, !dbg !21
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !21
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1048, !dbg !21
  store half %2268, ptr addrspace(1) %address.local1071, align 2, !dbg !21
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !21
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1072, i64 %address.local1051, !dbg !21
  store half %2267, ptr addrspace(1) %address.local1073, align 2, !dbg !21
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !21
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1074, i64 %address.local1054, !dbg !21
  store half %2266, ptr addrspace(1) %address.local1075, align 2, !dbg !21
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !21
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1076, i64 %address.local1057, !dbg !21
  store half %2265, ptr addrspace(1) %address.local1077, align 2, !dbg !21
  br label %2297, !dbg !21

2297:                                             ; preds = %2259, %2258
  br i1 true, label %2298, label %2336, !dbg !21

2298:                                             ; preds = %2297
  %2299 = insertelement <4 x float> poison, float %2136, i64 0, !dbg !19
  %2300 = insertelement <4 x float> %2299, float %2142, i64 1, !dbg !19
  %2301 = insertelement <4 x float> %2300, float %2137, i64 2, !dbg !19
  %2302 = insertelement <4 x float> %2301, float %2143, i64 3, !dbg !19
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2087, !dbg !19
  %normalized115 = fmul <4 x float> %2302, %row.reciprocal114, !dbg !19
  %2303 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !20
  %2304 = extractelement <4 x half> %2303, i64 3, !dbg !20
  %2305 = extractelement <4 x half> %2303, i64 2, !dbg !20
  %2306 = extractelement <4 x half> %2303, i64 1, !dbg !20
  %2307 = extractelement <4 x half> %2303, i64 0, !dbg !20
  %2308 = insertelement <4 x float> poison, float %2026, i64 0, !dbg !19
  %2309 = insertelement <4 x float> %2308, float %2032, i64 1, !dbg !19
  %2310 = insertelement <4 x float> %2309, float %2027, i64 2, !dbg !19
  %2311 = insertelement <4 x float> %2310, float %2033, i64 3, !dbg !19
  %normalized116 = fmul <4 x float> %2311, %row.reciprocal114, !dbg !19
  %2312 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %2313 = extractelement <4 x half> %2312, i64 3, !dbg !20
  %2314 = extractelement <4 x half> %2312, i64 2, !dbg !20
  %2315 = extractelement <4 x half> %2312, i64 1, !dbg !20
  %2316 = extractelement <4 x half> %2312, i64 0, !dbg !20
  %2317 = insertelement <4 x float> poison, float %1918, i64 0, !dbg !19
  %2318 = insertelement <4 x float> %2317, float %1924, i64 1, !dbg !19
  %2319 = insertelement <4 x float> %2318, float %1919, i64 2, !dbg !19
  %2320 = insertelement <4 x float> %2319, float %1925, i64 3, !dbg !19
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1869, !dbg !19
  %normalized118 = fmul <4 x float> %2320, %row.reciprocal117, !dbg !19
  %2321 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !20
  %2322 = extractelement <4 x half> %2321, i64 3, !dbg !20
  %2323 = extractelement <4 x half> %2321, i64 2, !dbg !20
  %2324 = extractelement <4 x half> %2321, i64 1, !dbg !20
  %2325 = extractelement <4 x half> %2321, i64 0, !dbg !20
  %2326 = insertelement <4 x float> poison, float %1808, i64 0, !dbg !19
  %2327 = insertelement <4 x float> %2326, float %1814, i64 1, !dbg !19
  %2328 = insertelement <4 x float> %2327, float %1809, i64 2, !dbg !19
  %2329 = insertelement <4 x float> %2328, float %1815, i64 3, !dbg !19
  %normalized119 = fmul <4 x float> %2329, %row.reciprocal117, !dbg !19
  %2330 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %2331 = extractelement <4 x half> %2330, i64 3, !dbg !20
  %2332 = extractelement <4 x half> %2330, i64 2, !dbg !20
  %2333 = extractelement <4 x half> %2330, i64 1, !dbg !20
  %2334 = extractelement <4 x half> %2330, i64 0, !dbg !20
  %2335 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1078 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1079 = zext nneg i32 %address.thread1078 to i64, !dbg !8
  %address.local1080 = add i64 %25, %address.local1079, !dbg !21
  %address.local1081 = lshr i64 %address.local1079, 6, !dbg !11
  %address.local1082 = shl nuw nsw i64 %address.local1081, 4, !dbg !21
  %address.local1083 = add i64 %address.local1080, %address.local1082, !dbg !21
  %address.local1084 = and i64 %address.local1079, 240, !dbg !21
  %address.local1085 = sub i64 %address.local1083, %address.local1084, !dbg !21
  %address.local1086 = add i64 %address.local1085, %26, !dbg !21
  %address.local1087 = shl nuw nsw i64 %address.local1086, 7, !dbg !21
  %address.local1088 = add nuw nsw i64 %2335, %address.local1087, !dbg !21
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !21
  %address.local1090 = lshr i64 %address.local1079, 2, !dbg !11
  %address.local1091 = and i64 %address.local1090, 12, !dbg !11
  %address.local1092 = or disjoint i64 %address.local1091, 64, !dbg !21
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1092, !dbg !21
  store half %2334, ptr addrspace(1) %address.local1093, align 2, !dbg !21
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !21
  %address.local1095 = or disjoint i64 %address.local1091, 65, !dbg !21
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1094, i64 %address.local1095, !dbg !21
  store half %2333, ptr addrspace(1) %address.local1096, align 2, !dbg !21
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !21
  %address.local1098 = or disjoint i64 %address.local1091, 66, !dbg !21
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1097, i64 %address.local1098, !dbg !21
  store half %2332, ptr addrspace(1) %address.local1099, align 2, !dbg !21
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !21
  %address.local1101 = or disjoint i64 %address.local1091, 67, !dbg !21
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1100, i64 %address.local1101, !dbg !21
  store half %2331, ptr addrspace(1) %address.local1102, align 2, !dbg !21
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !21
  %address.local1104 = or disjoint i64 %address.local1091, 80, !dbg !21
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1103, i64 %address.local1104, !dbg !21
  store half %2325, ptr addrspace(1) %address.local1105, align 2, !dbg !21
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !21
  %address.local1107 = or disjoint i64 %address.local1091, 81, !dbg !21
  %address.local1108 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1106, i64 %address.local1107, !dbg !21
  store half %2324, ptr addrspace(1) %address.local1108, align 2, !dbg !21
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !21
  %address.local1110 = or disjoint i64 %address.local1091, 82, !dbg !21
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1109, i64 %address.local1110, !dbg !21
  store half %2323, ptr addrspace(1) %address.local1111, align 2, !dbg !21
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !21
  %address.local1113 = or disjoint i64 %address.local1091, 83, !dbg !21
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1112, i64 %address.local1113, !dbg !21
  store half %2322, ptr addrspace(1) %address.local1114, align 2, !dbg !21
  %address.local1115 = shl i64 %address.local1086, 7, !dbg !21
  %address.local1116 = add i64 %address.local1115, 8192, !dbg !21
  %address.local1117 = add nuw nsw i64 %2335, %address.local1116, !dbg !21
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1117, !dbg !21
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1118, i64 %address.local1092, !dbg !21
  store half %2316, ptr addrspace(1) %address.local1119, align 2, !dbg !21
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1117, !dbg !21
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1120, i64 %address.local1095, !dbg !21
  store half %2315, ptr addrspace(1) %address.local1121, align 2, !dbg !21
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1117, !dbg !21
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1122, i64 %address.local1098, !dbg !21
  store half %2314, ptr addrspace(1) %address.local1123, align 2, !dbg !21
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1117, !dbg !21
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1124, i64 %address.local1101, !dbg !21
  store half %2313, ptr addrspace(1) %address.local1125, align 2, !dbg !21
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1117, !dbg !21
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1126, i64 %address.local1104, !dbg !21
  store half %2307, ptr addrspace(1) %address.local1127, align 2, !dbg !21
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1117, !dbg !21
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1128, i64 %address.local1107, !dbg !21
  store half %2306, ptr addrspace(1) %address.local1129, align 2, !dbg !21
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1117, !dbg !21
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1130, i64 %address.local1110, !dbg !21
  store half %2305, ptr addrspace(1) %address.local1131, align 2, !dbg !21
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1117, !dbg !21
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1132, i64 %address.local1113, !dbg !21
  store half %2304, ptr addrspace(1) %address.local1133, align 2, !dbg !21
  br label %2336, !dbg !21

2336:                                             ; preds = %2298, %2297
  br i1 true, label %2337, label %2375, !dbg !21

2337:                                             ; preds = %2336
  %2338 = insertelement <4 x float> poison, float %2108, i64 0, !dbg !19
  %2339 = insertelement <4 x float> %2338, float %2114, i64 1, !dbg !19
  %2340 = insertelement <4 x float> %2339, float %2109, i64 2, !dbg !19
  %2341 = insertelement <4 x float> %2340, float %2115, i64 3, !dbg !19
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2087, !dbg !19
  %normalized121 = fmul <4 x float> %2341, %row.reciprocal120, !dbg !19
  %2342 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !20
  %2343 = extractelement <4 x half> %2342, i64 3, !dbg !20
  %2344 = extractelement <4 x half> %2342, i64 2, !dbg !20
  %2345 = extractelement <4 x half> %2342, i64 1, !dbg !20
  %2346 = extractelement <4 x half> %2342, i64 0, !dbg !20
  %2347 = insertelement <4 x float> poison, float %1997, i64 0, !dbg !19
  %2348 = insertelement <4 x float> %2347, float %2003, i64 1, !dbg !19
  %2349 = insertelement <4 x float> %2348, float %1998, i64 2, !dbg !19
  %2350 = insertelement <4 x float> %2349, float %2004, i64 3, !dbg !19
  %normalized122 = fmul <4 x float> %2350, %row.reciprocal120, !dbg !19
  %2351 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %2352 = extractelement <4 x half> %2351, i64 3, !dbg !20
  %2353 = extractelement <4 x half> %2351, i64 2, !dbg !20
  %2354 = extractelement <4 x half> %2351, i64 1, !dbg !20
  %2355 = extractelement <4 x half> %2351, i64 0, !dbg !20
  %2356 = insertelement <4 x float> poison, float %1890, i64 0, !dbg !19
  %2357 = insertelement <4 x float> %2356, float %1896, i64 1, !dbg !19
  %2358 = insertelement <4 x float> %2357, float %1891, i64 2, !dbg !19
  %2359 = insertelement <4 x float> %2358, float %1897, i64 3, !dbg !19
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1869, !dbg !19
  %normalized124 = fmul <4 x float> %2359, %row.reciprocal123, !dbg !19
  %2360 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %2361 = extractelement <4 x half> %2360, i64 3, !dbg !20
  %2362 = extractelement <4 x half> %2360, i64 2, !dbg !20
  %2363 = extractelement <4 x half> %2360, i64 1, !dbg !20
  %2364 = extractelement <4 x half> %2360, i64 0, !dbg !20
  %2365 = insertelement <4 x float> poison, float %1779, i64 0, !dbg !19
  %2366 = insertelement <4 x float> %2365, float %1785, i64 1, !dbg !19
  %2367 = insertelement <4 x float> %2366, float %1780, i64 2, !dbg !19
  %2368 = insertelement <4 x float> %2367, float %1786, i64 3, !dbg !19
  %normalized125 = fmul <4 x float> %2368, %row.reciprocal123, !dbg !19
  %2369 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %2370 = extractelement <4 x half> %2369, i64 3, !dbg !20
  %2371 = extractelement <4 x half> %2369, i64 2, !dbg !20
  %2372 = extractelement <4 x half> %2369, i64 1, !dbg !20
  %2373 = extractelement <4 x half> %2369, i64 0, !dbg !20
  %2374 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1134 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1135 = zext nneg i32 %address.thread1134 to i64, !dbg !8
  %address.local1136 = add i64 %25, %address.local1135, !dbg !21
  %address.local1137 = lshr i64 %address.local1135, 6, !dbg !11
  %address.local1138 = shl nuw nsw i64 %address.local1137, 4, !dbg !21
  %address.local1139 = add i64 %address.local1136, %address.local1138, !dbg !21
  %address.local1140 = and i64 %address.local1135, 240, !dbg !21
  %address.local1141 = sub i64 %address.local1139, %address.local1140, !dbg !21
  %address.local1142 = add i64 %address.local1141, %26, !dbg !21
  %address.local1143 = shl nuw nsw i64 %address.local1142, 7, !dbg !21
  %address.local1144 = add nuw nsw i64 %2374, %address.local1143, !dbg !21
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !21
  %address.local1146 = lshr i64 %address.local1135, 2, !dbg !11
  %address.local1147 = and i64 %address.local1146, 12, !dbg !11
  %address.local1148 = or disjoint i64 %address.local1147, 96, !dbg !21
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1148, !dbg !21
  store half %2373, ptr addrspace(1) %address.local1149, align 2, !dbg !21
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !21
  %address.local1151 = or disjoint i64 %address.local1147, 97, !dbg !21
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1150, i64 %address.local1151, !dbg !21
  store half %2372, ptr addrspace(1) %address.local1152, align 2, !dbg !21
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !21
  %address.local1154 = or disjoint i64 %address.local1147, 98, !dbg !21
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1154, !dbg !21
  store half %2371, ptr addrspace(1) %address.local1155, align 2, !dbg !21
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !21
  %address.local1157 = or disjoint i64 %address.local1147, 99, !dbg !21
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1156, i64 %address.local1157, !dbg !21
  store half %2370, ptr addrspace(1) %address.local1158, align 2, !dbg !21
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !21
  %address.local1160 = or disjoint i64 %address.local1147, 112, !dbg !21
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1159, i64 %address.local1160, !dbg !21
  store half %2364, ptr addrspace(1) %address.local1161, align 2, !dbg !21
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !21
  %address.local1163 = or disjoint i64 %address.local1147, 113, !dbg !21
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1162, i64 %address.local1163, !dbg !21
  store half %2363, ptr addrspace(1) %address.local1164, align 2, !dbg !21
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !21
  %address.local1166 = or disjoint i64 %address.local1147, 114, !dbg !21
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1165, i64 %address.local1166, !dbg !21
  store half %2362, ptr addrspace(1) %address.local1167, align 2, !dbg !21
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !21
  %address.local1169 = or i64 %address.local1146, 115, !dbg !21
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1168, i64 %address.local1169, !dbg !21
  store half %2361, ptr addrspace(1) %address.local1170, align 2, !dbg !21
  %address.local1171 = shl i64 %address.local1142, 7, !dbg !21
  %address.local1172 = add i64 %address.local1171, 8192, !dbg !21
  %address.local1173 = add nuw nsw i64 %2374, %address.local1172, !dbg !21
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1173, !dbg !21
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1174, i64 %address.local1148, !dbg !21
  store half %2355, ptr addrspace(1) %address.local1175, align 2, !dbg !21
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1173, !dbg !21
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1176, i64 %address.local1151, !dbg !21
  store half %2354, ptr addrspace(1) %address.local1177, align 2, !dbg !21
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1173, !dbg !21
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1178, i64 %address.local1154, !dbg !21
  store half %2353, ptr addrspace(1) %address.local1179, align 2, !dbg !21
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1173, !dbg !21
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1180, i64 %address.local1157, !dbg !21
  store half %2352, ptr addrspace(1) %address.local1181, align 2, !dbg !21
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1173, !dbg !21
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1182, i64 %address.local1160, !dbg !21
  store half %2346, ptr addrspace(1) %address.local1183, align 2, !dbg !21
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1173, !dbg !21
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1184, i64 %address.local1163, !dbg !21
  store half %2345, ptr addrspace(1) %address.local1185, align 2, !dbg !21
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1173, !dbg !21
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1186, i64 %address.local1166, !dbg !21
  store half %2344, ptr addrspace(1) %address.local1187, align 2, !dbg !21
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1173, !dbg !21
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1188, i64 %address.local1169, !dbg !21
  store half %2343, ptr addrspace(1) %address.local1189, align 2, !dbg !21
  br label %2375, !dbg !21

2375:                                             ; preds = %2337, %2336
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
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/local_reduction_pass/compiler-bn128/run-20261026-112711-jne7_i1s/grid-000002")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 70, column: 22, scope: !4)
!10 = !DILocation(line: 48, column: 17, scope: !4)
!11 = !DILocation(line: 72, column: 24, scope: !4)
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
