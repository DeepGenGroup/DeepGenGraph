; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/exp2_fixed/run-20261027-123528-0u0so_ep/grid-000002/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/exp2_fixed/run-20261027-123528-0u0so_ep/grid-000002/input.mlir"

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
  %address.local505 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local506 = and i64 %address.local505, 1536, !dbg !8
  %address.local507 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local506, !dbg !8
  %address.local508 = ptrtoint ptr addrspace(3) %address.local507 to i64, !dbg !8
  %address.local509 = trunc i64 %address.local508 to i32, !dbg !8
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local509), !dbg !8
  %35 = zext i32 %34 to i64, !dbg !8
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !8
  %address.local510 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local511 = and i64 %address.local510, 16256, !dbg !8
  %address.local512 = lshr i64 %address.local, 4, !dbg !8
  %address.local513 = and i64 %address.local512, 8, !dbg !8
  %address.local514 = or disjoint i64 %address.local511, %address.local513, !dbg !8
  %address.local515 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local514, !dbg !8
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %38 = ptrtoint ptr addrspace(1) %address.local515 to i64, !dbg !8
  %39 = sub i64 %38, %37, !dbg !8
  %dma.buffer.offset1778 = trunc i64 %39 to i32, !dbg !8
  %40 = trunc i64 %37 to i32, !dbg !8
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !8
  %42 = lshr i64 %37, 32, !dbg !8
  %43 = trunc i64 %42 to i32, !dbg !8
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !8
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1779 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !8
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !8
  %dma.buffer.lds1780 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1780, i32 %dma.buffer.offset1778, <4 x i32> %dma.buffer.rsrc1779) #10, !dbg !8
  %address.local516 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %address.local520 = ptrtoint ptr addrspace(3) %address.local519 to i64, !dbg !8
  %address.local521 = trunc i64 %address.local520 to i32, !dbg !8
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !8
  %49 = zext i32 %48 to i64, !dbg !8
  %50 = inttoptr i64 %49 to ptr addrspace(3), !dbg !8
  %address.local522 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local523 = and i64 %address.local522, 16256, !dbg !8
  %address.local524 = lshr i64 %address.local516, 4, !dbg !8
  %address.local525 = and i64 %address.local524, 24, !dbg !8
  %address.local526 = or disjoint i64 %address.local523, %address.local525, !dbg !8
  %address.local527 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local526, !dbg !8
  %51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %52 = ptrtoint ptr addrspace(1) %address.local527 to i64, !dbg !8
  %53 = sub i64 %52, %51, !dbg !8
  %dma.buffer.offset1775 = trunc i64 %53 to i32, !dbg !8
  %54 = trunc i64 %51 to i32, !dbg !8
  %55 = insertelement <4 x i32> poison, i32 %54, i64 0, !dbg !8
  %56 = lshr i64 %51, 32, !dbg !8
  %57 = trunc i64 %56 to i32, !dbg !8
  %58 = insertelement <4 x i32> %55, i32 %57, i64 1, !dbg !8
  %59 = insertelement <4 x i32> %58, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1776 = insertelement <4 x i32> %59, i32 131072, i64 3, !dbg !8
  %60 = ptrtoint ptr addrspace(3) %50 to i32, !dbg !8
  %dma.buffer.lds1777 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !8
  %61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1777, i32 %dma.buffer.offset1775, <4 x i32> %dma.buffer.rsrc1776) #10, !dbg !8
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
  %address.local528 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local506, !dbg !8
  %address.local529 = ptrtoint ptr addrspace(3) %address.local528 to i64, !dbg !8
  %address.local530 = trunc i64 %address.local529 to i32, !dbg !8
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local530), !dbg !8
  %69 = zext i32 %68 to i64, !dbg !8
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !8
  %address.local531 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local514, !dbg !8
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %72 = ptrtoint ptr addrspace(1) %address.local531 to i64, !dbg !8
  %73 = sub i64 %72, %71, !dbg !8
  %dma.buffer.offset1772 = trunc i64 %73 to i32, !dbg !8
  %74 = trunc i64 %71 to i32, !dbg !8
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !8
  %76 = lshr i64 %71, 32, !dbg !8
  %77 = trunc i64 %76 to i32, !dbg !8
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !8
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1773 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !8
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !8
  %dma.buffer.lds1774 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !8
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1774, i32 %dma.buffer.offset1772, <4 x i32> %dma.buffer.rsrc1773) #10, !dbg !8
  %address.local532 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %address.local533 = ptrtoint ptr addrspace(3) %address.local532 to i64, !dbg !8
  %address.local534 = trunc i64 %address.local533 to i32, !dbg !8
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local534), !dbg !8
  %83 = zext i32 %82 to i64, !dbg !8
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !8
  %address.local535 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local526, !dbg !8
  %85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %86 = ptrtoint ptr addrspace(1) %address.local535 to i64, !dbg !8
  %87 = sub i64 %86, %85, !dbg !8
  %dma.buffer.offset1769 = trunc i64 %87 to i32, !dbg !8
  %88 = trunc i64 %85 to i32, !dbg !8
  %89 = insertelement <4 x i32> poison, i32 %88, i64 0, !dbg !8
  %90 = lshr i64 %85, 32, !dbg !8
  %91 = trunc i64 %90 to i32, !dbg !8
  %92 = insertelement <4 x i32> %89, i32 %91, i64 1, !dbg !8
  %93 = insertelement <4 x i32> %92, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1770 = insertelement <4 x i32> %93, i32 131072, i64 3, !dbg !8
  %94 = ptrtoint ptr addrspace(3) %84 to i32, !dbg !8
  %dma.buffer.lds1771 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !8
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1771, i32 %dma.buffer.offset1769, <4 x i32> %dma.buffer.rsrc1770) #10, !dbg !8
  %address.local536 = lshr i32 %address.thread, 2, !dbg !11
  %address.local537 = and i32 %address.local536, 4, !dbg !11
  %address.local538 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local539 = and i32 %address.local538, 384, !dbg !11
  %address.local540 = or disjoint i32 %address.local537, %address.local539, !dbg !11
  %address.local541 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local542 = and i32 %address.local541, 120, !dbg !11
  %address.local543 = or disjoint i32 %address.local540, %address.local542, !dbg !11
  %address.local544 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local545 = and i32 %address.local544, 1024, !dbg !11
  %address.local546 = or disjoint i32 %address.local543, %address.local545, !dbg !11
  %address.local547 = zext nneg i32 %address.local546 to i64, !dbg !11
  %address.local548 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local547, !dbg !11
  %96 = load <4 x half>, ptr addrspace(3) %address.local548, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local548, i64 4096, !dbg !11
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %98 = fmul <4 x half> %96, splat (half 0xH3015), !dbg !11
  %lds.const.gep1579 = getelementptr i8, ptr addrspace(3) %address.local548, i64 1024, !dbg !11
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !11
  %100 = fmul <4 x half> %97, splat (half 0xH3015), !dbg !11
  %lds.const.gep1580 = getelementptr i8, ptr addrspace(3) %address.local548, i64 5120, !dbg !11
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !11
  %102 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !11
  %103 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %105 = getelementptr i8, ptr addrspace(1) %104, i64 128, !dbg !8
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local509), !dbg !8
  %107 = zext i32 %106 to i64, !dbg !8
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !8
  %address.local591 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local514, !dbg !8
  %109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %110 = ptrtoint ptr addrspace(1) %address.local591 to i64, !dbg !8
  %111 = sub i64 %110, %109, !dbg !8
  %dma.buffer.offset1766 = trunc i64 %111 to i32, !dbg !8
  %112 = trunc i64 %109 to i32, !dbg !8
  %113 = insertelement <4 x i32> poison, i32 %112, i64 0, !dbg !8
  %114 = lshr i64 %109, 32, !dbg !8
  %115 = trunc i64 %114 to i32, !dbg !8
  %116 = insertelement <4 x i32> %113, i32 %115, i64 1, !dbg !8
  %117 = insertelement <4 x i32> %116, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1767 = insertelement <4 x i32> %117, i32 131072, i64 3, !dbg !8
  %118 = ptrtoint ptr addrspace(3) %108 to i32, !dbg !8
  %dma.buffer.lds1768 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !8
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1768, i32 %dma.buffer.offset1766, <4 x i32> %dma.buffer.rsrc1767) #10, !dbg !8
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !8
  %121 = zext i32 %120 to i64, !dbg !8
  %122 = inttoptr i64 %121 to ptr addrspace(3), !dbg !8
  %address.local592 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local526, !dbg !8
  %123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %124 = ptrtoint ptr addrspace(1) %address.local592 to i64, !dbg !8
  %125 = sub i64 %124, %123, !dbg !8
  %dma.buffer.offset1763 = trunc i64 %125 to i32, !dbg !8
  %126 = trunc i64 %123 to i32, !dbg !8
  %127 = insertelement <4 x i32> poison, i32 %126, i64 0, !dbg !8
  %128 = lshr i64 %123, 32, !dbg !8
  %129 = trunc i64 %128 to i32, !dbg !8
  %130 = insertelement <4 x i32> %127, i32 %129, i64 1, !dbg !8
  %131 = insertelement <4 x i32> %130, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1764 = insertelement <4 x i32> %131, i32 131072, i64 3, !dbg !8
  %132 = ptrtoint ptr addrspace(3) %122 to i32, !dbg !8
  %dma.buffer.lds1765 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !8
  %133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1765, i32 %dma.buffer.offset1763, <4 x i32> %dma.buffer.rsrc1764) #10, !dbg !8
  %lds.const.gep1581 = getelementptr i8, ptr addrspace(3) %address.local548, i64 8192, !dbg !11
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !11
  %lds.const.gep1582 = getelementptr i8, ptr addrspace(3) %address.local548, i64 12288, !dbg !11
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !11
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %address.local548, i64 9216, !dbg !11
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !11
  %137 = fmul <4 x half> %134, splat (half 0xH3015), !dbg !11
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local548, i64 13312, !dbg !11
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !11
  %139 = fmul <4 x half> %135, splat (half 0xH3015), !dbg !11
  %140 = fmul <4 x half> %136, splat (half 0xH3015), !dbg !11
  %141 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %143 = getelementptr i8, ptr addrspace(1) %142, i64 192, !dbg !8
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local530), !dbg !8
  %145 = zext i32 %144 to i64, !dbg !8
  %146 = inttoptr i64 %145 to ptr addrspace(3), !dbg !8
  %address.local597 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local514, !dbg !8
  %147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %148 = ptrtoint ptr addrspace(1) %address.local597 to i64, !dbg !8
  %149 = sub i64 %148, %147, !dbg !8
  %dma.buffer.offset1760 = trunc i64 %149 to i32, !dbg !8
  %150 = trunc i64 %147 to i32, !dbg !8
  %151 = insertelement <4 x i32> poison, i32 %150, i64 0, !dbg !8
  %152 = lshr i64 %147, 32, !dbg !8
  %153 = trunc i64 %152 to i32, !dbg !8
  %154 = insertelement <4 x i32> %151, i32 %153, i64 1, !dbg !8
  %155 = insertelement <4 x i32> %154, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1761 = insertelement <4 x i32> %155, i32 131072, i64 3, !dbg !8
  %156 = ptrtoint ptr addrspace(3) %146 to i32, !dbg !8
  %dma.buffer.lds1762 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %156), !dbg !8
  %157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1762, i32 %dma.buffer.offset1760, <4 x i32> %dma.buffer.rsrc1761) #10, !dbg !8
  %158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local534), !dbg !8
  %159 = zext i32 %158 to i64, !dbg !8
  %160 = inttoptr i64 %159 to ptr addrspace(3), !dbg !8
  %address.local598 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local526, !dbg !8
  %161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %162 = ptrtoint ptr addrspace(1) %address.local598 to i64, !dbg !8
  %163 = sub i64 %162, %161, !dbg !8
  %dma.buffer.offset1757 = trunc i64 %163 to i32, !dbg !8
  %164 = trunc i64 %161 to i32, !dbg !8
  %165 = insertelement <4 x i32> poison, i32 %164, i64 0, !dbg !8
  %166 = lshr i64 %161, 32, !dbg !8
  %167 = trunc i64 %166 to i32, !dbg !8
  %168 = insertelement <4 x i32> %165, i32 %167, i64 1, !dbg !8
  %169 = insertelement <4 x i32> %168, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1758 = insertelement <4 x i32> %169, i32 131072, i64 3, !dbg !8
  %170 = ptrtoint ptr addrspace(3) %160 to i32, !dbg !8
  %dma.buffer.lds1759 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !8
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1759, i32 %dma.buffer.offset1757, <4 x i32> %dma.buffer.rsrc1758) #10, !dbg !8
  %172 = load <4 x half>, ptr addrspace(3) %address.local548, align 8, !dbg !11
  %lds.const.gep1585 = getelementptr i8, ptr addrspace(3) %address.local548, i64 4096, !dbg !11
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1585, align 8, !dbg !11
  %lds.const.gep1586 = getelementptr i8, ptr addrspace(3) %address.local548, i64 1024, !dbg !11
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1586, align 8, !dbg !11
  %175 = fmul <4 x half> %172, splat (half 0xH3015), !dbg !11
  %lds.const.gep1587 = getelementptr i8, ptr addrspace(3) %address.local548, i64 5120, !dbg !11
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1587, align 8, !dbg !11
  %177 = fmul <4 x half> %173, splat (half 0xH3015), !dbg !11
  %178 = fmul <4 x half> %174, splat (half 0xH3015), !dbg !11
  %179 = fmul <4 x half> %176, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1588 = getelementptr i8, ptr addrspace(3) %address.local548, i64 8192, !dbg !11
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1588, align 8, !dbg !11
  %lds.const.gep1589 = getelementptr i8, ptr addrspace(3) %address.local548, i64 12288, !dbg !11
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !11
  %lds.const.gep1590 = getelementptr i8, ptr addrspace(3) %address.local548, i64 9216, !dbg !11
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !11
  %183 = fmul <4 x half> %180, splat (half 0xH3015), !dbg !11
  %lds.const.gep1591 = getelementptr i8, ptr addrspace(3) %address.local548, i64 13312, !dbg !11
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !11
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
  %index.compact445 = zext nneg i32 %200 to i64, !dbg !17
  %201 = or disjoint i64 %199, %index.compact445, !dbg !17
  %202 = add nuw nsw i64 %201, 1, !dbg !17
  %203 = sdiv i64 %202, 2, !dbg !17
  br label %204, !dbg !7

204:                                              ; preds = %845, %4
  %205 = phi i64 [ %1606, %845 ], [ 0, %4 ], !dbg !7
  %206 = phi [2 x <8 x float>] [ %1591, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %207 = phi [2 x <1 x float>] [ %1197, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %208 = phi [2 x <1 x float>] [ %1593, %845 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %209 = phi [2 x <8 x float>] [ %1597, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %210 = phi [2 x <8 x float>] [ %1601, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %211 = phi [2 x <8 x float>] [ %1605, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %212 = icmp slt i64 %205, %191, !dbg !7
  br i1 %212, label %213, label %1607, !dbg !7

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
  %address.thread599 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local600 = zext nneg i32 %address.thread599 to i64, !dbg !8
  %address.local601 = shl nuw nsw i64 %address.local600, 3, !dbg !8
  %address.local602 = and i64 %address.local601, 1536, !dbg !8
  %address.local603 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local602, !dbg !8
  %address.local604 = ptrtoint ptr addrspace(3) %address.local603 to i64, !dbg !8
  %address.local605 = trunc i64 %address.local604 to i32, !dbg !8
  %227 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local605), !dbg !16
  %228 = zext i32 %227 to i64, !dbg !16
  %229 = inttoptr i64 %228 to ptr addrspace(3), !dbg !16
  %address.local606 = lshr i64 %address.local600, 4, !dbg !16
  %address.local607 = and i64 %address.local606, 8, !dbg !16
  %address.local608 = and i64 %address.local600, 123, !dbg !16
  %address.local609 = lshr i64 %address.local600, 2, !dbg !16
  %address.local610 = and i64 %address.local609, 4, !dbg !16
  %address.local611 = add nuw nsw i64 %address.local610, %address.local600, !dbg !16
  %address.local612 = and i64 %address.local611, 4, !dbg !16
  %address.local613 = or disjoint i64 %address.local608, %address.local612, !dbg !16
  %address.local614 = shl nuw nsw i64 %address.local613, 7, !dbg !16
  %address.local615 = or disjoint i64 %address.local607, %address.local614, !dbg !16
  %address.local616 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %address.local615, !dbg !16
  %230 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %231 = ptrtoint ptr addrspace(1) %address.local616 to i64, !dbg !16
  %232 = sub i64 %231, %230, !dbg !16
  %dma.buffer.offset1754 = trunc i64 %232 to i32, !dbg !16
  %233 = trunc i64 %230 to i32, !dbg !16
  %234 = insertelement <4 x i32> poison, i32 %233, i64 0, !dbg !16
  %235 = lshr i64 %230, 32, !dbg !16
  %236 = trunc i64 %235 to i32, !dbg !16
  %237 = insertelement <4 x i32> %234, i32 %236, i64 1, !dbg !16
  %238 = insertelement <4 x i32> %237, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1755 = insertelement <4 x i32> %238, i32 131072, i64 3, !dbg !16
  %239 = ptrtoint ptr addrspace(3) %229 to i32, !dbg !16
  %dma.buffer.lds1756 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %239), !dbg !16
  %240 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1756, i32 %dma.buffer.offset1754, <4 x i32> %dma.buffer.rsrc1755) #10, !dbg !16
  %address.local617 = or disjoint i64 %address.local600, 256, !dbg !8
  %address.local618 = shl nuw nsw i64 %address.local617, 3, !dbg !8
  %address.local619 = and i64 %address.local618, 3584, !dbg !8
  %address.local620 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local619, !dbg !8
  %address.local621 = ptrtoint ptr addrspace(3) %address.local620 to i64, !dbg !8
  %address.local622 = trunc i64 %address.local621 to i32, !dbg !8
  %241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local622), !dbg !16
  %242 = zext i32 %241 to i64, !dbg !16
  %243 = inttoptr i64 %242 to ptr addrspace(3), !dbg !16
  %address.local623 = lshr i64 %address.local617, 4, !dbg !16
  %address.local624 = and i64 %address.local623, 24, !dbg !16
  %address.local625 = and i64 %address.local600, 123, !dbg !16
  %address.local626 = lshr i64 %address.local600, 2, !dbg !16
  %address.local627 = and i64 %address.local626, 4, !dbg !16
  %address.local628 = add nuw nsw i64 %address.local627, %address.local600, !dbg !16
  %address.local629 = and i64 %address.local628, 4, !dbg !16
  %address.local630 = or disjoint i64 %address.local625, %address.local629, !dbg !16
  %address.local631 = shl nuw nsw i64 %address.local630, 7, !dbg !16
  %address.local632 = or disjoint i64 %address.local624, %address.local631, !dbg !16
  %address.local633 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %address.local632, !dbg !16
  %244 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %245 = ptrtoint ptr addrspace(1) %address.local633 to i64, !dbg !16
  %246 = sub i64 %245, %244, !dbg !16
  %dma.buffer.offset1751 = trunc i64 %246 to i32, !dbg !16
  %247 = trunc i64 %244 to i32, !dbg !16
  %248 = insertelement <4 x i32> poison, i32 %247, i64 0, !dbg !16
  %249 = lshr i64 %244, 32, !dbg !16
  %250 = trunc i64 %249 to i32, !dbg !16
  %251 = insertelement <4 x i32> %248, i32 %250, i64 1, !dbg !16
  %252 = insertelement <4 x i32> %251, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1752 = insertelement <4 x i32> %252, i32 131072, i64 3, !dbg !16
  %253 = ptrtoint ptr addrspace(3) %243 to i32, !dbg !16
  %dma.buffer.lds1753 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %253), !dbg !16
  %254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1753, i32 %dma.buffer.offset1751, <4 x i32> %dma.buffer.rsrc1752) #10, !dbg !16
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
  %address.local634 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local602, !dbg !8
  %address.local635 = ptrtoint ptr addrspace(3) %address.local634 to i64, !dbg !8
  %address.local636 = trunc i64 %address.local635 to i32, !dbg !8
  %279 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local636), !dbg !16
  %280 = zext i32 %279 to i64, !dbg !16
  %281 = inttoptr i64 %280 to ptr addrspace(3), !dbg !16
  %address.local637 = getelementptr [2 x i8], ptr addrspace(1) %278, i64 %address.local615, !dbg !16
  %282 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %283 = ptrtoint ptr addrspace(1) %address.local637 to i64, !dbg !16
  %284 = sub i64 %283, %282, !dbg !16
  %dma.buffer.offset1748 = trunc i64 %284 to i32, !dbg !16
  %285 = trunc i64 %282 to i32, !dbg !16
  %286 = insertelement <4 x i32> poison, i32 %285, i64 0, !dbg !16
  %287 = lshr i64 %282, 32, !dbg !16
  %288 = trunc i64 %287 to i32, !dbg !16
  %289 = insertelement <4 x i32> %286, i32 %288, i64 1, !dbg !16
  %290 = insertelement <4 x i32> %289, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1749 = insertelement <4 x i32> %290, i32 131072, i64 3, !dbg !16
  %291 = ptrtoint ptr addrspace(3) %281 to i32, !dbg !16
  %dma.buffer.lds1750 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %291), !dbg !16
  %292 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1750, i32 %dma.buffer.offset1748, <4 x i32> %dma.buffer.rsrc1749) #10, !dbg !16
  %address.local638 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local619, !dbg !8
  %address.local639 = ptrtoint ptr addrspace(3) %address.local638 to i64, !dbg !8
  %address.local640 = trunc i64 %address.local639 to i32, !dbg !8
  %293 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local640), !dbg !16
  %294 = zext i32 %293 to i64, !dbg !16
  %295 = inttoptr i64 %294 to ptr addrspace(3), !dbg !16
  %address.local641 = getelementptr [2 x i8], ptr addrspace(1) %278, i64 %address.local632, !dbg !16
  %296 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %297 = ptrtoint ptr addrspace(1) %address.local641 to i64, !dbg !16
  %298 = sub i64 %297, %296, !dbg !16
  %dma.buffer.offset1745 = trunc i64 %298 to i32, !dbg !16
  %299 = trunc i64 %296 to i32, !dbg !16
  %300 = insertelement <4 x i32> poison, i32 %299, i64 0, !dbg !16
  %301 = lshr i64 %296, 32, !dbg !16
  %302 = trunc i64 %301 to i32, !dbg !16
  %303 = insertelement <4 x i32> %300, i32 %302, i64 1, !dbg !16
  %304 = insertelement <4 x i32> %303, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1746 = insertelement <4 x i32> %304, i32 131072, i64 3, !dbg !16
  %305 = ptrtoint ptr addrspace(3) %295 to i32, !dbg !16
  %dma.buffer.lds1747 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %305), !dbg !16
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1747, i32 %dma.buffer.offset1745, <4 x i32> %dma.buffer.rsrc1746) #10, !dbg !16
  %address.local642 = mul nuw nsw i64 %address.local600, 160, !dbg !17
  %address.local643 = lshr i64 %address.local600, 2, !dbg !17
  %address.local644 = mul nsw i64 %address.local643, -640, !dbg !17
  %address.local645 = add nsw i64 %address.local642, %address.local644, !dbg !17
  %address.local646 = shl nuw nsw i64 %address.local600, 1, !dbg !17
  %address.local647 = and i64 %address.local646, 24, !dbg !17
  %address.local648 = or disjoint i64 %address.local645, %address.local647, !dbg !17
  %address.local649 = shl i64 %address.local600, 62, !dbg !17
  %address.local650 = ashr exact i64 %address.local649, 57, !dbg !17
  %address.local651 = and i64 %address.local650, -64, !dbg !17
  %address.local652 = add nsw i64 %address.local648, %address.local651, !dbg !17
  %address.local653 = lshr i64 %address.local600, 2, !dbg !11
  %address.local654 = and i64 %address.local653, 12, !dbg !11
  %address.local655 = add nsw i64 %address.local652, %address.local654, !dbg !17
  %address.local656 = and i64 %address.local600, 63, !dbg !11
  %address.local657 = lshr i64 %address.local656, 5, !dbg !11
  %address.local658 = mul nuw nsw i64 %address.local657, 1016, !dbg !11
  %address.local659 = add i64 %address.local655, %address.local658, !dbg !17
  %address.local660 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local659, !dbg !17
  %307 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !17
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %307, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1592 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4096, !dbg !17
  %309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !17
  %address.local671 = shl nuw nsw i32 %address.thread599, 1, !dbg !17
  %address.local672 = and i32 %address.local671, 24, !dbg !17
  %address.local673 = shl nuw nsw i32 %address.thread599, 8, !dbg !17
  %address.local674 = and i32 %address.local673, 768, !dbg !17
  %address.local675 = or disjoint i32 %address.local672, %address.local674, !dbg !17
  %address.local676 = shl nuw nsw i32 %address.thread599, 5, !dbg !17
  %address.local677 = and i32 %address.local676, 96, !dbg !17
  %address.local678 = or disjoint i32 %address.local675, %address.local677, !dbg !17
  %address.local679 = zext nneg i32 %address.local678 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %203) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local680 = lshr i32 %address.thread599, 1, !dbg !17
  %address.local681 = and i32 %address.local680, 1, !dbg !17
  %address.local682 = zext nneg i32 %address.local681 to i64, !dbg !17
  %address.local683 = add nsw i64 %address.base, %address.local682, !dbg !17
  %address.local684 = shl i64 %address.local683, 6, !dbg !17
  %address.local685 = sub i64 %address.local679, %address.local684, !dbg !17
  %address.local686 = add i64 %address.local685, %address.local654, !dbg !17
  %address.local687 = add i64 %address.local686, %address.local658, !dbg !17
  %address.local688 = add i64 %address.local687, 32, !dbg !17
  %address.local689 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local688, !dbg !17
  %310 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !17
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %309, <4 x float> %308) #9, !dbg !17
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %310, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %address.local689, i64 4096, !dbg !17
  %313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !17
  %lds.const.gep1594 = getelementptr i8, ptr addrspace(3) %address.local660, i64 128, !dbg !17
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !17
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %313, <4 x float> %312) #9, !dbg !17
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %314, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1595 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4224, !dbg !17
  %317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !17
  %lds.const.gep1596 = getelementptr i8, ptr addrspace(3) %address.local689, i64 128, !dbg !17
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !17
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %317, <4 x float> %316) #9, !dbg !17
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %318, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1597 = getelementptr i8, ptr addrspace(3) %address.local689, i64 4224, !dbg !17
  %321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !17
  %lds.const.gep1598 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1024, !dbg !17
  %322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !17
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %321, <4 x float> %320) #9, !dbg !17
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %322, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1599 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5120, !dbg !17
  %325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !17
  %lds.const.gep1600 = getelementptr i8, ptr addrspace(3) %address.local689, i64 1024, !dbg !17
  %326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !17
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %325, <4 x float> %324) #9, !dbg !17
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %326, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1601 = getelementptr i8, ptr addrspace(3) %address.local689, i64 5120, !dbg !17
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !17
  %lds.const.gep1602 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1152, !dbg !17
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !17
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %329, <4 x float> %328) #9, !dbg !17
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %330, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1603 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5248, !dbg !17
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !17
  %lds.const.gep1604 = getelementptr i8, ptr addrspace(3) %address.local689, i64 1152, !dbg !17
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !17
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %333, <4 x float> %332) #9, !dbg !17
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %334, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1605 = getelementptr i8, ptr addrspace(3) %address.local689, i64 5248, !dbg !17
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !17
  %338 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !17
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %337, <4 x float> %336) #9, !dbg !17
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %338, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1606 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4096, !dbg !17
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1606, align 8, !dbg !17
  %342 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !17
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %341, <4 x float> %340) #9, !dbg !17
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %342, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1607 = getelementptr i8, ptr addrspace(3) %address.local689, i64 4096, !dbg !17
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1607, align 8, !dbg !17
  %lds.const.gep1608 = getelementptr i8, ptr addrspace(3) %address.local660, i64 128, !dbg !17
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1608, align 8, !dbg !17
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %345, <4 x float> %344) #9, !dbg !17
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %346, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1609 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4224, !dbg !17
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1609, align 8, !dbg !17
  %lds.const.gep1610 = getelementptr i8, ptr addrspace(3) %address.local689, i64 128, !dbg !17
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1610, align 8, !dbg !17
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %349, <4 x float> %348) #9, !dbg !17
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %350, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1611 = getelementptr i8, ptr addrspace(3) %address.local689, i64 4224, !dbg !17
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1611, align 8, !dbg !17
  %lds.const.gep1612 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1024, !dbg !17
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1612, align 8, !dbg !17
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %353, <4 x float> %352) #9, !dbg !17
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %354, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1613 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5120, !dbg !17
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1613, align 8, !dbg !17
  %lds.const.gep1614 = getelementptr i8, ptr addrspace(3) %address.local689, i64 1024, !dbg !17
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1614, align 8, !dbg !17
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %357, <4 x float> %356) #9, !dbg !17
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %358, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1615 = getelementptr i8, ptr addrspace(3) %address.local689, i64 5120, !dbg !17
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1615, align 8, !dbg !17
  %lds.const.gep1616 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1152, !dbg !17
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1616, align 8, !dbg !17
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %361, <4 x float> %360) #9, !dbg !17
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %362, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1617 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5248, !dbg !17
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1617, align 8, !dbg !17
  %lds.const.gep1618 = getelementptr i8, ptr addrspace(3) %address.local689, i64 1152, !dbg !17
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1618, align 8, !dbg !17
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %365, <4 x float> %364) #9, !dbg !17
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %366, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1619 = getelementptr i8, ptr addrspace(3) %address.local689, i64 5248, !dbg !17
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1619, align 8, !dbg !17
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
  %394 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local605), !dbg !16
  %395 = zext i32 %394 to i64, !dbg !16
  %396 = inttoptr i64 %395 to ptr addrspace(3), !dbg !16
  %address.local998 = getelementptr [2 x i8], ptr addrspace(1) %393, i64 %address.local615, !dbg !16
  %397 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %398 = ptrtoint ptr addrspace(1) %address.local998 to i64, !dbg !16
  %399 = sub i64 %398, %397, !dbg !16
  %dma.buffer.offset1742 = trunc i64 %399 to i32, !dbg !16
  %400 = trunc i64 %397 to i32, !dbg !16
  %401 = insertelement <4 x i32> poison, i32 %400, i64 0, !dbg !16
  %402 = lshr i64 %397, 32, !dbg !16
  %403 = trunc i64 %402 to i32, !dbg !16
  %404 = insertelement <4 x i32> %401, i32 %403, i64 1, !dbg !16
  %405 = insertelement <4 x i32> %404, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1743 = insertelement <4 x i32> %405, i32 131072, i64 3, !dbg !16
  %406 = ptrtoint ptr addrspace(3) %396 to i32, !dbg !16
  %dma.buffer.lds1744 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %406), !dbg !16
  %407 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1744, i32 %dma.buffer.offset1742, <4 x i32> %dma.buffer.rsrc1743) #10, !dbg !16
  %408 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local622), !dbg !16
  %409 = zext i32 %408 to i64, !dbg !16
  %410 = inttoptr i64 %409 to ptr addrspace(3), !dbg !16
  %address.local999 = getelementptr [2 x i8], ptr addrspace(1) %393, i64 %address.local632, !dbg !16
  %411 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %412 = ptrtoint ptr addrspace(1) %address.local999 to i64, !dbg !16
  %413 = sub i64 %412, %411, !dbg !16
  %dma.buffer.offset1739 = trunc i64 %413 to i32, !dbg !16
  %414 = trunc i64 %411 to i32, !dbg !16
  %415 = insertelement <4 x i32> poison, i32 %414, i64 0, !dbg !16
  %416 = lshr i64 %411, 32, !dbg !16
  %417 = trunc i64 %416 to i32, !dbg !16
  %418 = insertelement <4 x i32> %415, i32 %417, i64 1, !dbg !16
  %419 = insertelement <4 x i32> %418, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1740 = insertelement <4 x i32> %419, i32 131072, i64 3, !dbg !16
  %420 = ptrtoint ptr addrspace(3) %410 to i32, !dbg !16
  %dma.buffer.lds1741 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %420), !dbg !16
  %421 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1741, i32 %dma.buffer.offset1739, <4 x i32> %dma.buffer.rsrc1740) #10, !dbg !16
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8192, !dbg !17
  %422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !17
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %422, <4 x float> %311) #9, !dbg !17
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12288, !dbg !17
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !17
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %address.local689, i64 8192, !dbg !17
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !17
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %424, <4 x float> %423) #9, !dbg !17
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %425, <4 x float> %315) #9, !dbg !17
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %address.local689, i64 12288, !dbg !17
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !17
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8320, !dbg !17
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !17
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %428, <4 x float> %427) #9, !dbg !17
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %429, <4 x float> %319) #9, !dbg !17
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12416, !dbg !17
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !17
  %lds.const.gep1626 = getelementptr i8, ptr addrspace(3) %address.local689, i64 8320, !dbg !17
  %433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !17
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %432, <4 x float> %431) #9, !dbg !17
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %433, <4 x float> %323) #9, !dbg !17
  %lds.const.gep1627 = getelementptr i8, ptr addrspace(3) %address.local689, i64 12416, !dbg !17
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !17
  %lds.const.gep1628 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9216, !dbg !17
  %437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !17
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %436, <4 x float> %435) #9, !dbg !17
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %437, <4 x float> %327) #9, !dbg !17
  %lds.const.gep1629 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13312, !dbg !17
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !17
  %lds.const.gep1630 = getelementptr i8, ptr addrspace(3) %address.local689, i64 9216, !dbg !17
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !17
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %440, <4 x float> %439) #9, !dbg !17
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %441, <4 x float> %331) #9, !dbg !17
  %lds.const.gep1631 = getelementptr i8, ptr addrspace(3) %address.local689, i64 13312, !dbg !17
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !17
  %lds.const.gep1632 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9344, !dbg !17
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !17
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %444, <4 x float> %443) #9, !dbg !17
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %445, <4 x float> %335) #9, !dbg !17
  %lds.const.gep1633 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13440, !dbg !17
  %448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !17
  %lds.const.gep1634 = getelementptr i8, ptr addrspace(3) %address.local689, i64 9344, !dbg !17
  %449 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !17
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %448, <4 x float> %447) #9, !dbg !17
  %451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %449, <4 x float> %339) #9, !dbg !17
  %lds.const.gep1635 = getelementptr i8, ptr addrspace(3) %address.local689, i64 13440, !dbg !17
  %452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !17
  %lds.const.gep1636 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8192, !dbg !17
  %453 = load <4 x half>, ptr addrspace(3) %lds.const.gep1636, align 8, !dbg !17
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %452, <4 x float> %451) #9, !dbg !17
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %453, <4 x float> %343) #9, !dbg !17
  %lds.const.gep1637 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12288, !dbg !17
  %456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1637, align 8, !dbg !17
  %lds.const.gep1638 = getelementptr i8, ptr addrspace(3) %address.local689, i64 8192, !dbg !17
  %457 = load <4 x half>, ptr addrspace(3) %lds.const.gep1638, align 8, !dbg !17
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %456, <4 x float> %455) #9, !dbg !17
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %457, <4 x float> %347) #9, !dbg !17
  %lds.const.gep1639 = getelementptr i8, ptr addrspace(3) %address.local689, i64 12288, !dbg !17
  %460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1639, align 8, !dbg !17
  %lds.const.gep1640 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8320, !dbg !17
  %461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1640, align 8, !dbg !17
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %460, <4 x float> %459) #9, !dbg !17
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %461, <4 x float> %351) #9, !dbg !17
  %lds.const.gep1641 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12416, !dbg !17
  %464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1641, align 8, !dbg !17
  %lds.const.gep1642 = getelementptr i8, ptr addrspace(3) %address.local689, i64 8320, !dbg !17
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1642, align 8, !dbg !17
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %464, <4 x float> %463) #9, !dbg !17
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %465, <4 x float> %355) #9, !dbg !17
  %lds.const.gep1643 = getelementptr i8, ptr addrspace(3) %address.local689, i64 12416, !dbg !17
  %468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1643, align 8, !dbg !17
  %lds.const.gep1644 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9216, !dbg !17
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1644, align 8, !dbg !17
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %468, <4 x float> %467) #9, !dbg !17
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %469, <4 x float> %359) #9, !dbg !17
  %lds.const.gep1645 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13312, !dbg !17
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1645, align 8, !dbg !17
  %lds.const.gep1646 = getelementptr i8, ptr addrspace(3) %address.local689, i64 9216, !dbg !17
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1646, align 8, !dbg !17
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %472, <4 x float> %471) #9, !dbg !17
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %473, <4 x float> %363) #9, !dbg !17
  %lds.const.gep1647 = getelementptr i8, ptr addrspace(3) %address.local689, i64 13312, !dbg !17
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1647, align 8, !dbg !17
  %lds.const.gep1648 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9344, !dbg !17
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1648, align 8, !dbg !17
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %476, <4 x float> %475) #9, !dbg !17
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %477, <4 x float> %367) #9, !dbg !17
  %lds.const.gep1649 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13440, !dbg !17
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1649, align 8, !dbg !17
  %lds.const.gep1650 = getelementptr i8, ptr addrspace(3) %address.local689, i64 9344, !dbg !17
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1650, align 8, !dbg !17
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %480, <4 x float> %479) #9, !dbg !17
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %481, <4 x float> %370) #9, !dbg !17
  %lds.const.gep1651 = getelementptr i8, ptr addrspace(3) %address.local689, i64 13440, !dbg !17
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1651, align 8, !dbg !17
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
  %509 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local636), !dbg !16
  %510 = zext i32 %509 to i64, !dbg !16
  %511 = inttoptr i64 %510 to ptr addrspace(3), !dbg !16
  %address.local1016 = getelementptr [2 x i8], ptr addrspace(1) %508, i64 %address.local615, !dbg !16
  %512 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %513 = ptrtoint ptr addrspace(1) %address.local1016 to i64, !dbg !16
  %514 = sub i64 %513, %512, !dbg !16
  %dma.buffer.offset1736 = trunc i64 %514 to i32, !dbg !16
  %515 = trunc i64 %512 to i32, !dbg !16
  %516 = insertelement <4 x i32> poison, i32 %515, i64 0, !dbg !16
  %517 = lshr i64 %512, 32, !dbg !16
  %518 = trunc i64 %517 to i32, !dbg !16
  %519 = insertelement <4 x i32> %516, i32 %518, i64 1, !dbg !16
  %520 = insertelement <4 x i32> %519, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1737 = insertelement <4 x i32> %520, i32 131072, i64 3, !dbg !16
  %521 = ptrtoint ptr addrspace(3) %511 to i32, !dbg !16
  %dma.buffer.lds1738 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %521), !dbg !16
  %522 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1738, i32 %dma.buffer.offset1736, <4 x i32> %dma.buffer.rsrc1737) #10, !dbg !16
  %523 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local640), !dbg !16
  %524 = zext i32 %523 to i64, !dbg !16
  %525 = inttoptr i64 %524 to ptr addrspace(3), !dbg !16
  %address.local1017 = getelementptr [2 x i8], ptr addrspace(1) %508, i64 %address.local632, !dbg !16
  %526 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %527 = ptrtoint ptr addrspace(1) %address.local1017 to i64, !dbg !16
  %528 = sub i64 %527, %526, !dbg !16
  %dma.buffer.offset1733 = trunc i64 %528 to i32, !dbg !16
  %529 = trunc i64 %526 to i32, !dbg !16
  %530 = insertelement <4 x i32> poison, i32 %529, i64 0, !dbg !16
  %531 = lshr i64 %526, 32, !dbg !16
  %532 = trunc i64 %531 to i32, !dbg !16
  %533 = insertelement <4 x i32> %530, i32 %532, i64 1, !dbg !16
  %534 = insertelement <4 x i32> %533, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1734 = insertelement <4 x i32> %534, i32 131072, i64 3, !dbg !16
  %535 = ptrtoint ptr addrspace(3) %525 to i32, !dbg !16
  %dma.buffer.lds1735 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %535), !dbg !16
  %536 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1735, i32 %dma.buffer.offset1733, <4 x i32> %dma.buffer.rsrc1734) #10, !dbg !16
  %537 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !17
  %538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %537, <4 x float> %426) #9, !dbg !17
  %lds.const.gep1652 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4096, !dbg !17
  %539 = load <4 x half>, ptr addrspace(3) %lds.const.gep1652, align 8, !dbg !17
  %540 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !17
  %541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %539, <4 x float> %538) #9, !dbg !17
  %542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %540, <4 x float> %430) #9, !dbg !17
  %lds.const.gep1653 = getelementptr i8, ptr addrspace(3) %address.local689, i64 4096, !dbg !17
  %543 = load <4 x half>, ptr addrspace(3) %lds.const.gep1653, align 8, !dbg !17
  %lds.const.gep1654 = getelementptr i8, ptr addrspace(3) %address.local660, i64 128, !dbg !17
  %544 = load <4 x half>, ptr addrspace(3) %lds.const.gep1654, align 8, !dbg !17
  %545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %543, <4 x float> %542) #9, !dbg !17
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %544, <4 x float> %434) #9, !dbg !17
  %lds.const.gep1655 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4224, !dbg !17
  %547 = load <4 x half>, ptr addrspace(3) %lds.const.gep1655, align 8, !dbg !17
  %lds.const.gep1656 = getelementptr i8, ptr addrspace(3) %address.local689, i64 128, !dbg !17
  %548 = load <4 x half>, ptr addrspace(3) %lds.const.gep1656, align 8, !dbg !17
  %549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %547, <4 x float> %546) #9, !dbg !17
  %550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %548, <4 x float> %438) #9, !dbg !17
  %lds.const.gep1657 = getelementptr i8, ptr addrspace(3) %address.local689, i64 4224, !dbg !17
  %551 = load <4 x half>, ptr addrspace(3) %lds.const.gep1657, align 8, !dbg !17
  %lds.const.gep1658 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1024, !dbg !17
  %552 = load <4 x half>, ptr addrspace(3) %lds.const.gep1658, align 8, !dbg !17
  %553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %551, <4 x float> %550) #9, !dbg !17
  %554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %552, <4 x float> %442) #9, !dbg !17
  %lds.const.gep1659 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5120, !dbg !17
  %555 = load <4 x half>, ptr addrspace(3) %lds.const.gep1659, align 8, !dbg !17
  %lds.const.gep1660 = getelementptr i8, ptr addrspace(3) %address.local689, i64 1024, !dbg !17
  %556 = load <4 x half>, ptr addrspace(3) %lds.const.gep1660, align 8, !dbg !17
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %555, <4 x float> %554) #9, !dbg !17
  %558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %556, <4 x float> %446) #9, !dbg !17
  %lds.const.gep1661 = getelementptr i8, ptr addrspace(3) %address.local689, i64 5120, !dbg !17
  %559 = load <4 x half>, ptr addrspace(3) %lds.const.gep1661, align 8, !dbg !17
  %lds.const.gep1662 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1152, !dbg !17
  %560 = load <4 x half>, ptr addrspace(3) %lds.const.gep1662, align 8, !dbg !17
  %561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %559, <4 x float> %558) #9, !dbg !17
  %562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %560, <4 x float> %450) #9, !dbg !17
  %lds.const.gep1663 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5248, !dbg !17
  %563 = load <4 x half>, ptr addrspace(3) %lds.const.gep1663, align 8, !dbg !17
  %lds.const.gep1664 = getelementptr i8, ptr addrspace(3) %address.local689, i64 1152, !dbg !17
  %564 = load <4 x half>, ptr addrspace(3) %lds.const.gep1664, align 8, !dbg !17
  %565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %563, <4 x float> %562) #9, !dbg !17
  %566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %564, <4 x float> %454) #9, !dbg !17
  %lds.const.gep1665 = getelementptr i8, ptr addrspace(3) %address.local689, i64 5248, !dbg !17
  %567 = load <4 x half>, ptr addrspace(3) %lds.const.gep1665, align 8, !dbg !17
  %568 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !17
  %569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %567, <4 x float> %566) #9, !dbg !17
  %570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %568, <4 x float> %458) #9, !dbg !17
  %lds.const.gep1666 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4096, !dbg !17
  %571 = load <4 x half>, ptr addrspace(3) %lds.const.gep1666, align 8, !dbg !17
  %572 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !17
  %573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %571, <4 x float> %570) #9, !dbg !17
  %574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %572, <4 x float> %462) #9, !dbg !17
  %lds.const.gep1667 = getelementptr i8, ptr addrspace(3) %address.local689, i64 4096, !dbg !17
  %575 = load <4 x half>, ptr addrspace(3) %lds.const.gep1667, align 8, !dbg !17
  %lds.const.gep1668 = getelementptr i8, ptr addrspace(3) %address.local660, i64 128, !dbg !17
  %576 = load <4 x half>, ptr addrspace(3) %lds.const.gep1668, align 8, !dbg !17
  %577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %575, <4 x float> %574) #9, !dbg !17
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %576, <4 x float> %466) #9, !dbg !17
  %lds.const.gep1669 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4224, !dbg !17
  %579 = load <4 x half>, ptr addrspace(3) %lds.const.gep1669, align 8, !dbg !17
  %lds.const.gep1670 = getelementptr i8, ptr addrspace(3) %address.local689, i64 128, !dbg !17
  %580 = load <4 x half>, ptr addrspace(3) %lds.const.gep1670, align 8, !dbg !17
  %581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %579, <4 x float> %578) #9, !dbg !17
  %582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %580, <4 x float> %470) #9, !dbg !17
  %lds.const.gep1671 = getelementptr i8, ptr addrspace(3) %address.local689, i64 4224, !dbg !17
  %583 = load <4 x half>, ptr addrspace(3) %lds.const.gep1671, align 8, !dbg !17
  %lds.const.gep1672 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1024, !dbg !17
  %584 = load <4 x half>, ptr addrspace(3) %lds.const.gep1672, align 8, !dbg !17
  %585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %583, <4 x float> %582) #9, !dbg !17
  %586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %584, <4 x float> %474) #9, !dbg !17
  %lds.const.gep1673 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5120, !dbg !17
  %587 = load <4 x half>, ptr addrspace(3) %lds.const.gep1673, align 8, !dbg !17
  %lds.const.gep1674 = getelementptr i8, ptr addrspace(3) %address.local689, i64 1024, !dbg !17
  %588 = load <4 x half>, ptr addrspace(3) %lds.const.gep1674, align 8, !dbg !17
  %589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %587, <4 x float> %586) #9, !dbg !17
  %590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %588, <4 x float> %478) #9, !dbg !17
  %lds.const.gep1675 = getelementptr i8, ptr addrspace(3) %address.local689, i64 5120, !dbg !17
  %591 = load <4 x half>, ptr addrspace(3) %lds.const.gep1675, align 8, !dbg !17
  %lds.const.gep1676 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1152, !dbg !17
  %592 = load <4 x half>, ptr addrspace(3) %lds.const.gep1676, align 8, !dbg !17
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %591, <4 x float> %590) #9, !dbg !17
  %594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %592, <4 x float> %482) #9, !dbg !17
  %lds.const.gep1677 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5248, !dbg !17
  %595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1677, align 8, !dbg !17
  %lds.const.gep1678 = getelementptr i8, ptr addrspace(3) %address.local689, i64 1152, !dbg !17
  %596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1678, align 8, !dbg !17
  %597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %595, <4 x float> %594) #9, !dbg !17
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %596, <4 x float> %485) #9, !dbg !17
  %lds.const.gep1679 = getelementptr i8, ptr addrspace(3) %address.local689, i64 5248, !dbg !17
  %599 = load <4 x half>, ptr addrspace(3) %lds.const.gep1679, align 8, !dbg !17
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
  %606 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local605), !dbg !18
  %607 = zext i32 %606 to i64, !dbg !18
  %608 = inttoptr i64 %607 to ptr addrspace(3), !dbg !18
  %address.local1018 = shl nuw nsw i64 %address.local600, 5, !dbg !18
  %address.local1019 = and i64 %address.local1018, 8064, !dbg !18
  %address.local1020 = shl nuw nsw i64 %address.local600, 3, !dbg !18
  %address.local1021 = and i64 %address.local1020, 24, !dbg !18
  %address.local1022 = or disjoint i64 %address.local1019, %address.local1021, !dbg !18
  %address.local1023 = getelementptr [2 x i8], ptr addrspace(1) %605, i64 %address.local1022, !dbg !18
  %609 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %610 = ptrtoint ptr addrspace(1) %address.local1023 to i64, !dbg !18
  %611 = sub i64 %610, %609, !dbg !18
  %dma.buffer.offset1730 = trunc i64 %611 to i32, !dbg !18
  %612 = trunc i64 %609 to i32, !dbg !18
  %613 = insertelement <4 x i32> poison, i32 %612, i64 0, !dbg !18
  %614 = lshr i64 %609, 32, !dbg !18
  %615 = trunc i64 %614 to i32, !dbg !18
  %616 = insertelement <4 x i32> %613, i32 %615, i64 1, !dbg !18
  %617 = insertelement <4 x i32> %616, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1731 = insertelement <4 x i32> %617, i32 131072, i64 3, !dbg !18
  %618 = ptrtoint ptr addrspace(3) %608 to i32, !dbg !18
  %dma.buffer.lds1732 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %618), !dbg !18
  %619 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1732, i32 %dma.buffer.offset1730, <4 x i32> %dma.buffer.rsrc1731) #10, !dbg !18
  %620 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local622), !dbg !18
  %621 = zext i32 %620 to i64, !dbg !18
  %622 = inttoptr i64 %621 to ptr addrspace(3), !dbg !18
  %address.local1024 = shl nuw nsw i64 %address.local617, 5, !dbg !18
  %address.local1025 = and i64 %address.local1024, 16256, !dbg !18
  %address.local1026 = shl nuw nsw i64 %address.local600, 3, !dbg !18
  %address.local1027 = and i64 %address.local1026, 24, !dbg !18
  %address.local1028 = or disjoint i64 %address.local1025, %address.local1027, !dbg !18
  %address.local1029 = getelementptr [2 x i8], ptr addrspace(1) %605, i64 %address.local1028, !dbg !18
  %623 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %624 = ptrtoint ptr addrspace(1) %address.local1029 to i64, !dbg !18
  %625 = sub i64 %624, %623, !dbg !18
  %dma.buffer.offset1727 = trunc i64 %625 to i32, !dbg !18
  %626 = trunc i64 %623 to i32, !dbg !18
  %627 = insertelement <4 x i32> poison, i32 %626, i64 0, !dbg !18
  %628 = lshr i64 %623, 32, !dbg !18
  %629 = trunc i64 %628 to i32, !dbg !18
  %630 = insertelement <4 x i32> %627, i32 %629, i64 1, !dbg !18
  %631 = insertelement <4 x i32> %630, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1728 = insertelement <4 x i32> %631, i32 131072, i64 3, !dbg !18
  %632 = ptrtoint ptr addrspace(3) %622 to i32, !dbg !18
  %dma.buffer.lds1729 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %632), !dbg !18
  %633 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1729, i32 %dma.buffer.offset1727, <4 x i32> %dma.buffer.rsrc1728) #10, !dbg !18
  %lds.const.gep1680 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8192, !dbg !17
  %634 = load <4 x half>, ptr addrspace(3) %lds.const.gep1680, align 8, !dbg !17
  %635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %634, <4 x float> %541) #9, !dbg !17
  %lds.const.gep1681 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12288, !dbg !17
  %636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1681, align 8, !dbg !17
  %lds.const.gep1682 = getelementptr i8, ptr addrspace(3) %address.local689, i64 8192, !dbg !17
  %637 = load <4 x half>, ptr addrspace(3) %lds.const.gep1682, align 8, !dbg !17
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %636, <4 x float> %635) #9, !dbg !17
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %637, <4 x float> %545) #9, !dbg !17
  %lds.const.gep1683 = getelementptr i8, ptr addrspace(3) %address.local689, i64 12288, !dbg !17
  %640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1683, align 8, !dbg !17
  %lds.const.gep1684 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8320, !dbg !17
  %641 = load <4 x half>, ptr addrspace(3) %lds.const.gep1684, align 8, !dbg !17
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %640, <4 x float> %639) #9, !dbg !17
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %641, <4 x float> %549) #9, !dbg !17
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12416, !dbg !17
  %644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1685, align 8, !dbg !17
  %lds.const.gep1686 = getelementptr i8, ptr addrspace(3) %address.local689, i64 8320, !dbg !17
  %645 = load <4 x half>, ptr addrspace(3) %lds.const.gep1686, align 8, !dbg !17
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %644, <4 x float> %643) #9, !dbg !17
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %645, <4 x float> %553) #9, !dbg !17
  %lds.const.gep1687 = getelementptr i8, ptr addrspace(3) %address.local689, i64 12416, !dbg !17
  %648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1687, align 8, !dbg !17
  %lds.const.gep1688 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9216, !dbg !17
  %649 = load <4 x half>, ptr addrspace(3) %lds.const.gep1688, align 8, !dbg !17
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %648, <4 x float> %647) #9, !dbg !17
  %651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %649, <4 x float> %557) #9, !dbg !17
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13312, !dbg !17
  %652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1689, align 8, !dbg !17
  %lds.const.gep1690 = getelementptr i8, ptr addrspace(3) %address.local689, i64 9216, !dbg !17
  %653 = load <4 x half>, ptr addrspace(3) %lds.const.gep1690, align 8, !dbg !17
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %652, <4 x float> %651) #9, !dbg !17
  %655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %653, <4 x float> %561) #9, !dbg !17
  %lds.const.gep1691 = getelementptr i8, ptr addrspace(3) %address.local689, i64 13312, !dbg !17
  %656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1691, align 8, !dbg !17
  %lds.const.gep1692 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9344, !dbg !17
  %657 = load <4 x half>, ptr addrspace(3) %lds.const.gep1692, align 8, !dbg !17
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %656, <4 x float> %655) #9, !dbg !17
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %657, <4 x float> %565) #9, !dbg !17
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13440, !dbg !17
  %660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1693, align 8, !dbg !17
  %lds.const.gep1694 = getelementptr i8, ptr addrspace(3) %address.local689, i64 9344, !dbg !17
  %661 = load <4 x half>, ptr addrspace(3) %lds.const.gep1694, align 8, !dbg !17
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %660, <4 x float> %659) #9, !dbg !17
  %663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %661, <4 x float> %569) #9, !dbg !17
  %lds.const.gep1695 = getelementptr i8, ptr addrspace(3) %address.local689, i64 13440, !dbg !17
  %664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1695, align 8, !dbg !17
  %lds.const.gep1696 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8192, !dbg !17
  %665 = load <4 x half>, ptr addrspace(3) %lds.const.gep1696, align 8, !dbg !17
  %666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %664, <4 x float> %663) #9, !dbg !17
  %667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %665, <4 x float> %573) #9, !dbg !17
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12288, !dbg !17
  %668 = load <4 x half>, ptr addrspace(3) %lds.const.gep1697, align 8, !dbg !17
  %lds.const.gep1698 = getelementptr i8, ptr addrspace(3) %address.local689, i64 8192, !dbg !17
  %669 = load <4 x half>, ptr addrspace(3) %lds.const.gep1698, align 8, !dbg !17
  %670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %668, <4 x float> %667) #9, !dbg !17
  %671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %669, <4 x float> %577) #9, !dbg !17
  %lds.const.gep1699 = getelementptr i8, ptr addrspace(3) %address.local689, i64 12288, !dbg !17
  %672 = load <4 x half>, ptr addrspace(3) %lds.const.gep1699, align 8, !dbg !17
  %lds.const.gep1700 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8320, !dbg !17
  %673 = load <4 x half>, ptr addrspace(3) %lds.const.gep1700, align 8, !dbg !17
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %672, <4 x float> %671) #9, !dbg !17
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %673, <4 x float> %581) #9, !dbg !17
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12416, !dbg !17
  %676 = load <4 x half>, ptr addrspace(3) %lds.const.gep1701, align 8, !dbg !17
  %lds.const.gep1702 = getelementptr i8, ptr addrspace(3) %address.local689, i64 8320, !dbg !17
  %677 = load <4 x half>, ptr addrspace(3) %lds.const.gep1702, align 8, !dbg !17
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %676, <4 x float> %675) #9, !dbg !17
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %677, <4 x float> %585) #9, !dbg !17
  %lds.const.gep1703 = getelementptr i8, ptr addrspace(3) %address.local689, i64 12416, !dbg !17
  %680 = load <4 x half>, ptr addrspace(3) %lds.const.gep1703, align 8, !dbg !17
  %lds.const.gep1704 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9216, !dbg !17
  %681 = load <4 x half>, ptr addrspace(3) %lds.const.gep1704, align 8, !dbg !17
  %682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %680, <4 x float> %679) #9, !dbg !17
  %683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %681, <4 x float> %589) #9, !dbg !17
  %lds.const.gep1705 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13312, !dbg !17
  %684 = load <4 x half>, ptr addrspace(3) %lds.const.gep1705, align 8, !dbg !17
  %lds.const.gep1706 = getelementptr i8, ptr addrspace(3) %address.local689, i64 9216, !dbg !17
  %685 = load <4 x half>, ptr addrspace(3) %lds.const.gep1706, align 8, !dbg !17
  %686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %684, <4 x float> %683) #9, !dbg !17
  %687 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %685, <4 x float> %593) #9, !dbg !17
  %lds.const.gep1707 = getelementptr i8, ptr addrspace(3) %address.local689, i64 13312, !dbg !17
  %688 = load <4 x half>, ptr addrspace(3) %lds.const.gep1707, align 8, !dbg !17
  %lds.const.gep1708 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9344, !dbg !17
  %689 = load <4 x half>, ptr addrspace(3) %lds.const.gep1708, align 8, !dbg !17
  %690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %688, <4 x float> %687) #9, !dbg !17
  %691 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %689, <4 x float> %597) #9, !dbg !17
  %lds.const.gep1709 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13440, !dbg !17
  %692 = load <4 x half>, ptr addrspace(3) %lds.const.gep1709, align 8, !dbg !17
  %lds.const.gep1710 = getelementptr i8, ptr addrspace(3) %address.local689, i64 9344, !dbg !17
  %693 = load <4 x half>, ptr addrspace(3) %lds.const.gep1710, align 8, !dbg !17
  %694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %692, <4 x float> %691) #9, !dbg !17
  %695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %693, <4 x float> %600) #9, !dbg !17
  %lds.const.gep1711 = getelementptr i8, ptr addrspace(3) %address.local689, i64 13440, !dbg !17
  %696 = load <4 x half>, ptr addrspace(3) %lds.const.gep1711, align 8, !dbg !17
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
  %compare.high1323 = trunc i64 %712 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1323, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1323, !dbg !24
  %compare.low.base = trunc i64 %710 to i32, !dbg !24
  %compare.low.base1324 = trunc i64 %709 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1324, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %713 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low1325 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result1326 = select i1 %compare.high.equal, i1 %compare.low1325, i1 %compare.high.order, !dbg !24
  %714 = select i1 %compare.result1326, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1327 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1328 = or i32 %compare.local.base1327, 2, !dbg !24
  %compare.low1329 = icmp ugt i32 %compare.low.base, %compare.local1328, !dbg !24
  %compare.result1330 = select i1 %compare.high.equal, i1 %compare.low1329, i1 %compare.high.order, !dbg !24
  %715 = select i1 %compare.result1330, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1331 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1332 = or i32 %compare.local.base1331, 3, !dbg !24
  %compare.low1333 = icmp ugt i32 %compare.low.base, %compare.local1332, !dbg !24
  %compare.result1334 = select i1 %compare.high.equal, i1 %compare.low1333, i1 %compare.high.order, !dbg !24
  %716 = select i1 %compare.result1334, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1335 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1336 = or i32 %compare.local.base1335, 4, !dbg !24
  %compare.low1337 = icmp ugt i32 %compare.low.base, %compare.local1336, !dbg !24
  %compare.result1338 = select i1 %compare.high.equal, i1 %compare.low1337, i1 %compare.high.order, !dbg !24
  %717 = select i1 %compare.result1338, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1339 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1340 = or i32 %compare.local.base1339, 5, !dbg !24
  %compare.low1341 = icmp ugt i32 %compare.low.base, %compare.local1340, !dbg !24
  %compare.result1342 = select i1 %compare.high.equal, i1 %compare.low1341, i1 %compare.high.order, !dbg !24
  %718 = select i1 %compare.result1342, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1343 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1344 = or i32 %compare.local.base1343, 6, !dbg !24
  %compare.low1345 = icmp ugt i32 %compare.low.base, %compare.local1344, !dbg !24
  %compare.result1346 = select i1 %compare.high.equal, i1 %compare.low1345, i1 %compare.high.order, !dbg !24
  %719 = select i1 %compare.result1346, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1347 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1348 = or i32 %compare.local.base1347, 7, !dbg !24
  %compare.low1349 = icmp ugt i32 %compare.low.base, %compare.local1348, !dbg !24
  %compare.result1350 = select i1 %compare.high.equal, i1 %compare.low1349, i1 %compare.high.order, !dbg !24
  %720 = select i1 %compare.result1350, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1351 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1352 = or i32 %compare.local.base1351, 8, !dbg !24
  %compare.low1353 = icmp ugt i32 %compare.low.base, %compare.local1352, !dbg !24
  %compare.result1354 = select i1 %compare.high.equal, i1 %compare.low1353, i1 %compare.high.order, !dbg !24
  %721 = select i1 %compare.result1354, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1355 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1356 = or i32 %compare.local.base1355, 9, !dbg !24
  %compare.low1357 = icmp ugt i32 %compare.low.base, %compare.local1356, !dbg !24
  %compare.result1358 = select i1 %compare.high.equal, i1 %compare.low1357, i1 %compare.high.order, !dbg !24
  %722 = select i1 %compare.result1358, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1359 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1360 = or i32 %compare.local.base1359, 10, !dbg !24
  %compare.low1361 = icmp ugt i32 %compare.low.base, %compare.local1360, !dbg !24
  %compare.result1362 = select i1 %compare.high.equal, i1 %compare.low1361, i1 %compare.high.order, !dbg !24
  %723 = select i1 %compare.result1362, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1363 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1364 = or i32 %compare.local.base1363, 11, !dbg !24
  %compare.low1365 = icmp ugt i32 %compare.low.base, %compare.local1364, !dbg !24
  %compare.result1366 = select i1 %compare.high.equal, i1 %compare.low1365, i1 %compare.high.order, !dbg !24
  %724 = select i1 %compare.result1366, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1367 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1368 = or i32 %compare.local.base1367, 12, !dbg !24
  %compare.low1369 = icmp ugt i32 %compare.low.base, %compare.local1368, !dbg !24
  %compare.result1370 = select i1 %compare.high.equal, i1 %compare.low1369, i1 %compare.high.order, !dbg !24
  %725 = select i1 %compare.result1370, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1371 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1372 = or i32 %compare.local.base1371, 13, !dbg !24
  %compare.low1373 = icmp ugt i32 %compare.low.base, %compare.local1372, !dbg !24
  %compare.result1374 = select i1 %compare.high.equal, i1 %compare.low1373, i1 %compare.high.order, !dbg !24
  %726 = select i1 %compare.result1374, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1375 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1376 = or i32 %compare.local.base1375, 14, !dbg !24
  %compare.low1377 = icmp ugt i32 %compare.low.base, %compare.local1376, !dbg !24
  %compare.result1378 = select i1 %compare.high.equal, i1 %compare.low1377, i1 %compare.high.order, !dbg !24
  %727 = select i1 %compare.result1378, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1379 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1380 = or i32 %compare.local.base1379, 15, !dbg !24
  %compare.low1381 = icmp ugt i32 %compare.low.base, %compare.local1380, !dbg !24
  %compare.result1382 = select i1 %compare.high.equal, i1 %compare.low1381, i1 %compare.high.order, !dbg !24
  %728 = select i1 %compare.result1382, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1383 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1384 = or i32 %compare.local.base1383, 64, !dbg !24
  %compare.low1385 = icmp ugt i32 %compare.low.base, %compare.local1384, !dbg !24
  %compare.result1386 = select i1 %compare.high.equal, i1 %compare.low1385, i1 %compare.high.order, !dbg !24
  %729 = select i1 %compare.result1386, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1387 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1388 = or i32 %compare.local.base1387, 65, !dbg !24
  %compare.low1389 = icmp ugt i32 %compare.low.base, %compare.local1388, !dbg !24
  %compare.result1390 = select i1 %compare.high.equal, i1 %compare.low1389, i1 %compare.high.order, !dbg !24
  %730 = select i1 %compare.result1390, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1391 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1392 = or i32 %compare.local.base1391, 66, !dbg !24
  %compare.low1393 = icmp ugt i32 %compare.low.base, %compare.local1392, !dbg !24
  %compare.result1394 = select i1 %compare.high.equal, i1 %compare.low1393, i1 %compare.high.order, !dbg !24
  %731 = select i1 %compare.result1394, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1395 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1396 = or i32 %compare.local.base1395, 67, !dbg !24
  %compare.low1397 = icmp ugt i32 %compare.low.base, %compare.local1396, !dbg !24
  %compare.result1398 = select i1 %compare.high.equal, i1 %compare.low1397, i1 %compare.high.order, !dbg !24
  %732 = select i1 %compare.result1398, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1399 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1400 = or i32 %compare.local.base1399, 68, !dbg !24
  %compare.low1401 = icmp ugt i32 %compare.low.base, %compare.local1400, !dbg !24
  %compare.result1402 = select i1 %compare.high.equal, i1 %compare.low1401, i1 %compare.high.order, !dbg !24
  %733 = select i1 %compare.result1402, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1403 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1404 = or i32 %compare.local.base1403, 69, !dbg !24
  %compare.low1405 = icmp ugt i32 %compare.low.base, %compare.local1404, !dbg !24
  %compare.result1406 = select i1 %compare.high.equal, i1 %compare.low1405, i1 %compare.high.order, !dbg !24
  %734 = select i1 %compare.result1406, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1407 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1408 = or i32 %compare.local.base1407, 70, !dbg !24
  %compare.low1409 = icmp ugt i32 %compare.low.base, %compare.local1408, !dbg !24
  %compare.result1410 = select i1 %compare.high.equal, i1 %compare.low1409, i1 %compare.high.order, !dbg !24
  %735 = select i1 %compare.result1410, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1411 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1412 = or i32 %compare.local.base1411, 71, !dbg !24
  %compare.low1413 = icmp ugt i32 %compare.low.base, %compare.local1412, !dbg !24
  %compare.result1414 = select i1 %compare.high.equal, i1 %compare.low1413, i1 %compare.high.order, !dbg !24
  %736 = select i1 %compare.result1414, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1415 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1416 = or i32 %compare.local.base1415, 72, !dbg !24
  %compare.low1417 = icmp ugt i32 %compare.low.base, %compare.local1416, !dbg !24
  %compare.result1418 = select i1 %compare.high.equal, i1 %compare.low1417, i1 %compare.high.order, !dbg !24
  %737 = select i1 %compare.result1418, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1419 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1420 = or i32 %compare.local.base1419, 73, !dbg !24
  %compare.low1421 = icmp ugt i32 %compare.low.base, %compare.local1420, !dbg !24
  %compare.result1422 = select i1 %compare.high.equal, i1 %compare.low1421, i1 %compare.high.order, !dbg !24
  %738 = select i1 %compare.result1422, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1423 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1424 = or i32 %compare.local.base1423, 74, !dbg !24
  %compare.low1425 = icmp ugt i32 %compare.low.base, %compare.local1424, !dbg !24
  %compare.result1426 = select i1 %compare.high.equal, i1 %compare.low1425, i1 %compare.high.order, !dbg !24
  %739 = select i1 %compare.result1426, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1427 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1428 = or i32 %compare.local.base1427, 75, !dbg !24
  %compare.low1429 = icmp ugt i32 %compare.low.base, %compare.local1428, !dbg !24
  %compare.result1430 = select i1 %compare.high.equal, i1 %compare.low1429, i1 %compare.high.order, !dbg !24
  %740 = select i1 %compare.result1430, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1431 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1432 = or i32 %compare.local.base1431, 76, !dbg !24
  %compare.low1433 = icmp ugt i32 %compare.low.base, %compare.local1432, !dbg !24
  %compare.result1434 = select i1 %compare.high.equal, i1 %compare.low1433, i1 %compare.high.order, !dbg !24
  %741 = select i1 %compare.result1434, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1435 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1436 = or i32 %compare.local.base1435, 77, !dbg !24
  %compare.low1437 = icmp ugt i32 %compare.low.base, %compare.local1436, !dbg !24
  %compare.result1438 = select i1 %compare.high.equal, i1 %compare.low1437, i1 %compare.high.order, !dbg !24
  %742 = select i1 %compare.result1438, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1439 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1440 = or i32 %compare.local.base1439, 78, !dbg !24
  %compare.low1441 = icmp ugt i32 %compare.low.base, %compare.local1440, !dbg !24
  %compare.result1442 = select i1 %compare.high.equal, i1 %compare.low1441, i1 %compare.high.order, !dbg !24
  %743 = select i1 %compare.result1442, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1443 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1444 = or i32 %compare.local.base1443, 79, !dbg !24
  %compare.low1445 = icmp ugt i32 %compare.low.base, %compare.local1444, !dbg !24
  %compare.result1446 = select i1 %compare.high.equal, i1 %compare.low1445, i1 %compare.high.order, !dbg !24
  %744 = select i1 %compare.result1446, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %745 = add i64 %707, 65, !dbg !23
  %746 = lshr i64 %745, 32, !dbg !24
  %compare.high1447 = trunc i64 %746 to i32, !dbg !24
  %747 = lshr i64 %709, 32, !dbg !24
  %compare.high1448 = trunc i64 %747 to i32, !dbg !24
  %compare.high.equal1449 = icmp eq i32 %compare.high1447, %compare.high1448, !dbg !24
  %compare.high.order1450 = icmp ugt i32 %compare.high1447, %compare.high1448, !dbg !24
  %compare.low.base1451 = trunc i64 %745 to i32, !dbg !24
  %compare.low.base1452 = trunc i64 %709 to i32, !dbg !24
  %compare.low1453 = icmp ugt i32 %compare.low.base1451, %compare.low.base1452, !dbg !24
  %compare.result1454 = select i1 %compare.high.equal1449, i1 %compare.low1453, i1 %compare.high.order1450, !dbg !24
  %748 = select i1 %compare.result1454, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1455 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1456 = or i32 %compare.local.base1455, 1, !dbg !24
  %compare.low1457 = icmp ugt i32 %compare.low.base1451, %compare.local1456, !dbg !24
  %compare.result1458 = select i1 %compare.high.equal1449, i1 %compare.low1457, i1 %compare.high.order1450, !dbg !24
  %749 = select i1 %compare.result1458, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1459 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1460 = or i32 %compare.local.base1459, 2, !dbg !24
  %compare.low1461 = icmp ugt i32 %compare.low.base1451, %compare.local1460, !dbg !24
  %compare.result1462 = select i1 %compare.high.equal1449, i1 %compare.low1461, i1 %compare.high.order1450, !dbg !24
  %750 = select i1 %compare.result1462, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1463 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1464 = or i32 %compare.local.base1463, 3, !dbg !24
  %compare.low1465 = icmp ugt i32 %compare.low.base1451, %compare.local1464, !dbg !24
  %compare.result1466 = select i1 %compare.high.equal1449, i1 %compare.low1465, i1 %compare.high.order1450, !dbg !24
  %751 = select i1 %compare.result1466, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1467 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1468 = or i32 %compare.local.base1467, 4, !dbg !24
  %compare.low1469 = icmp ugt i32 %compare.low.base1451, %compare.local1468, !dbg !24
  %compare.result1470 = select i1 %compare.high.equal1449, i1 %compare.low1469, i1 %compare.high.order1450, !dbg !24
  %752 = select i1 %compare.result1470, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1471 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1472 = or i32 %compare.local.base1471, 5, !dbg !24
  %compare.low1473 = icmp ugt i32 %compare.low.base1451, %compare.local1472, !dbg !24
  %compare.result1474 = select i1 %compare.high.equal1449, i1 %compare.low1473, i1 %compare.high.order1450, !dbg !24
  %753 = select i1 %compare.result1474, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1475 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1476 = or i32 %compare.local.base1475, 6, !dbg !24
  %compare.low1477 = icmp ugt i32 %compare.low.base1451, %compare.local1476, !dbg !24
  %compare.result1478 = select i1 %compare.high.equal1449, i1 %compare.low1477, i1 %compare.high.order1450, !dbg !24
  %754 = select i1 %compare.result1478, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1479 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1480 = or i32 %compare.local.base1479, 7, !dbg !24
  %compare.low1481 = icmp ugt i32 %compare.low.base1451, %compare.local1480, !dbg !24
  %compare.result1482 = select i1 %compare.high.equal1449, i1 %compare.low1481, i1 %compare.high.order1450, !dbg !24
  %755 = select i1 %compare.result1482, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1483 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1484 = or i32 %compare.local.base1483, 8, !dbg !24
  %compare.low1485 = icmp ugt i32 %compare.low.base1451, %compare.local1484, !dbg !24
  %compare.result1486 = select i1 %compare.high.equal1449, i1 %compare.low1485, i1 %compare.high.order1450, !dbg !24
  %756 = select i1 %compare.result1486, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1487 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1488 = or i32 %compare.local.base1487, 9, !dbg !24
  %compare.low1489 = icmp ugt i32 %compare.low.base1451, %compare.local1488, !dbg !24
  %compare.result1490 = select i1 %compare.high.equal1449, i1 %compare.low1489, i1 %compare.high.order1450, !dbg !24
  %757 = select i1 %compare.result1490, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1491 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1492 = or i32 %compare.local.base1491, 10, !dbg !24
  %compare.low1493 = icmp ugt i32 %compare.low.base1451, %compare.local1492, !dbg !24
  %compare.result1494 = select i1 %compare.high.equal1449, i1 %compare.low1493, i1 %compare.high.order1450, !dbg !24
  %758 = select i1 %compare.result1494, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1495 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1496 = or i32 %compare.local.base1495, 11, !dbg !24
  %compare.low1497 = icmp ugt i32 %compare.low.base1451, %compare.local1496, !dbg !24
  %compare.result1498 = select i1 %compare.high.equal1449, i1 %compare.low1497, i1 %compare.high.order1450, !dbg !24
  %759 = select i1 %compare.result1498, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1499 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1500 = or i32 %compare.local.base1499, 12, !dbg !24
  %compare.low1501 = icmp ugt i32 %compare.low.base1451, %compare.local1500, !dbg !24
  %compare.result1502 = select i1 %compare.high.equal1449, i1 %compare.low1501, i1 %compare.high.order1450, !dbg !24
  %760 = select i1 %compare.result1502, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1503 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1504 = or i32 %compare.local.base1503, 13, !dbg !24
  %compare.low1505 = icmp ugt i32 %compare.low.base1451, %compare.local1504, !dbg !24
  %compare.result1506 = select i1 %compare.high.equal1449, i1 %compare.low1505, i1 %compare.high.order1450, !dbg !24
  %761 = select i1 %compare.result1506, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1507 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1508 = or i32 %compare.local.base1507, 14, !dbg !24
  %compare.low1509 = icmp ugt i32 %compare.low.base1451, %compare.local1508, !dbg !24
  %compare.result1510 = select i1 %compare.high.equal1449, i1 %compare.low1509, i1 %compare.high.order1450, !dbg !24
  %762 = select i1 %compare.result1510, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1511 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1512 = or i32 %compare.local.base1511, 15, !dbg !24
  %compare.low1513 = icmp ugt i32 %compare.low.base1451, %compare.local1512, !dbg !24
  %compare.result1514 = select i1 %compare.high.equal1449, i1 %compare.low1513, i1 %compare.high.order1450, !dbg !24
  %763 = select i1 %compare.result1514, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1515 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1516 = or i32 %compare.local.base1515, 64, !dbg !24
  %compare.low1517 = icmp ugt i32 %compare.low.base1451, %compare.local1516, !dbg !24
  %compare.result1518 = select i1 %compare.high.equal1449, i1 %compare.low1517, i1 %compare.high.order1450, !dbg !24
  %764 = select i1 %compare.result1518, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1519 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1520 = or i32 %compare.local.base1519, 65, !dbg !24
  %compare.low1521 = icmp ugt i32 %compare.low.base1451, %compare.local1520, !dbg !24
  %compare.result1522 = select i1 %compare.high.equal1449, i1 %compare.low1521, i1 %compare.high.order1450, !dbg !24
  %765 = select i1 %compare.result1522, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1523 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1524 = or i32 %compare.local.base1523, 66, !dbg !24
  %compare.low1525 = icmp ugt i32 %compare.low.base1451, %compare.local1524, !dbg !24
  %compare.result1526 = select i1 %compare.high.equal1449, i1 %compare.low1525, i1 %compare.high.order1450, !dbg !24
  %766 = select i1 %compare.result1526, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1527 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1528 = or i32 %compare.local.base1527, 67, !dbg !24
  %compare.low1529 = icmp ugt i32 %compare.low.base1451, %compare.local1528, !dbg !24
  %compare.result1530 = select i1 %compare.high.equal1449, i1 %compare.low1529, i1 %compare.high.order1450, !dbg !24
  %767 = select i1 %compare.result1530, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1531 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1532 = or i32 %compare.local.base1531, 68, !dbg !24
  %compare.low1533 = icmp ugt i32 %compare.low.base1451, %compare.local1532, !dbg !24
  %compare.result1534 = select i1 %compare.high.equal1449, i1 %compare.low1533, i1 %compare.high.order1450, !dbg !24
  %768 = select i1 %compare.result1534, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1535 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1536 = or i32 %compare.local.base1535, 69, !dbg !24
  %compare.low1537 = icmp ugt i32 %compare.low.base1451, %compare.local1536, !dbg !24
  %compare.result1538 = select i1 %compare.high.equal1449, i1 %compare.low1537, i1 %compare.high.order1450, !dbg !24
  %769 = select i1 %compare.result1538, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1539 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1540 = or i32 %compare.local.base1539, 70, !dbg !24
  %compare.low1541 = icmp ugt i32 %compare.low.base1451, %compare.local1540, !dbg !24
  %compare.result1542 = select i1 %compare.high.equal1449, i1 %compare.low1541, i1 %compare.high.order1450, !dbg !24
  %770 = select i1 %compare.result1542, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1543 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1544 = or i32 %compare.local.base1543, 71, !dbg !24
  %compare.low1545 = icmp ugt i32 %compare.low.base1451, %compare.local1544, !dbg !24
  %compare.result1546 = select i1 %compare.high.equal1449, i1 %compare.low1545, i1 %compare.high.order1450, !dbg !24
  %771 = select i1 %compare.result1546, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1547 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1548 = or i32 %compare.local.base1547, 72, !dbg !24
  %compare.low1549 = icmp ugt i32 %compare.low.base1451, %compare.local1548, !dbg !24
  %compare.result1550 = select i1 %compare.high.equal1449, i1 %compare.low1549, i1 %compare.high.order1450, !dbg !24
  %772 = select i1 %compare.result1550, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1551 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1552 = or i32 %compare.local.base1551, 73, !dbg !24
  %compare.low1553 = icmp ugt i32 %compare.low.base1451, %compare.local1552, !dbg !24
  %compare.result1554 = select i1 %compare.high.equal1449, i1 %compare.low1553, i1 %compare.high.order1450, !dbg !24
  %773 = select i1 %compare.result1554, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1555 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1556 = or i32 %compare.local.base1555, 74, !dbg !24
  %compare.low1557 = icmp ugt i32 %compare.low.base1451, %compare.local1556, !dbg !24
  %compare.result1558 = select i1 %compare.high.equal1449, i1 %compare.low1557, i1 %compare.high.order1450, !dbg !24
  %774 = select i1 %compare.result1558, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1559 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1560 = or i32 %compare.local.base1559, 75, !dbg !24
  %compare.low1561 = icmp ugt i32 %compare.low.base1451, %compare.local1560, !dbg !24
  %compare.result1562 = select i1 %compare.high.equal1449, i1 %compare.low1561, i1 %compare.high.order1450, !dbg !24
  %775 = select i1 %compare.result1562, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1563 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1564 = or i32 %compare.local.base1563, 76, !dbg !24
  %compare.low1565 = icmp ugt i32 %compare.low.base1451, %compare.local1564, !dbg !24
  %compare.result1566 = select i1 %compare.high.equal1449, i1 %compare.low1565, i1 %compare.high.order1450, !dbg !24
  %776 = select i1 %compare.result1566, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1567 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1568 = or i32 %compare.local.base1567, 77, !dbg !24
  %compare.low1569 = icmp ugt i32 %compare.low.base1451, %compare.local1568, !dbg !24
  %compare.result1570 = select i1 %compare.high.equal1449, i1 %compare.low1569, i1 %compare.high.order1450, !dbg !24
  %777 = select i1 %compare.result1570, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1571 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1572 = or i32 %compare.local.base1571, 78, !dbg !24
  %compare.low1573 = icmp ugt i32 %compare.low.base1451, %compare.local1572, !dbg !24
  %compare.result1574 = select i1 %compare.high.equal1449, i1 %compare.low1573, i1 %compare.high.order1450, !dbg !24
  %778 = select i1 %compare.result1574, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1575 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1576 = or i32 %compare.local.base1575, 79, !dbg !24
  %compare.low1577 = icmp ugt i32 %compare.low.base1451, %compare.local1576, !dbg !24
  %compare.result1578 = select i1 %compare.high.equal1449, i1 %compare.low1577, i1 %compare.high.order1450, !dbg !24
  %779 = select i1 %compare.result1578, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
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
  %910 = call float @llvm.maxnum.f32(float %846, float 0xFFF0000000000000), !dbg !27, !frisk.local_reduction !1
  %local.reduce251 = call float @llvm.maxnum.f32(float %910, float %850), !dbg !27, !frisk.local_reduction !1
  %local.reduce252 = call float @llvm.maxnum.f32(float %847, float %851), !dbg !27, !frisk.local_reduction !1
  %local.reduce253 = call float @llvm.maxnum.f32(float %848, float %852), !dbg !27, !frisk.local_reduction !1
  %local.reduce254 = call float @llvm.maxnum.f32(float %849, float %853), !dbg !27, !frisk.local_reduction !1
  %local.reduce255 = call float @llvm.maxnum.f32(float %local.reduce251, float %854), !dbg !27, !frisk.local_reduction !1
  %local.reduce256 = call float @llvm.maxnum.f32(float %local.reduce252, float %855), !dbg !27, !frisk.local_reduction !1
  %local.reduce257 = call float @llvm.maxnum.f32(float %local.reduce253, float %856), !dbg !27, !frisk.local_reduction !1
  %local.reduce258 = call float @llvm.maxnum.f32(float %local.reduce254, float %857), !dbg !27, !frisk.local_reduction !1
  %local.reduce259 = call float @llvm.maxnum.f32(float %local.reduce255, float %858), !dbg !27, !frisk.local_reduction !1
  %local.reduce260 = call float @llvm.maxnum.f32(float %local.reduce256, float %859), !dbg !27, !frisk.local_reduction !1
  %local.reduce261 = call float @llvm.maxnum.f32(float %local.reduce257, float %860), !dbg !27, !frisk.local_reduction !1
  %local.reduce262 = call float @llvm.maxnum.f32(float %local.reduce258, float %861), !dbg !27, !frisk.local_reduction !1
  %local.reduce263 = call float @llvm.maxnum.f32(float %local.reduce259, float %862), !dbg !27, !frisk.local_reduction !1
  %local.reduce264 = call float @llvm.maxnum.f32(float %local.reduce260, float %863), !dbg !27, !frisk.local_reduction !1
  %local.reduce265 = call float @llvm.maxnum.f32(float %local.reduce261, float %864), !dbg !27, !frisk.local_reduction !1
  %local.reduce266 = call float @llvm.maxnum.f32(float %local.reduce262, float %865), !dbg !27, !frisk.local_reduction !1
  %local.reduce267 = call float @llvm.maxnum.f32(float %local.reduce263, float %866), !dbg !27, !frisk.local_reduction !1
  %local.reduce268 = call float @llvm.maxnum.f32(float %local.reduce264, float %867), !dbg !27, !frisk.local_reduction !1
  %local.reduce269 = call float @llvm.maxnum.f32(float %local.reduce265, float %868), !dbg !27, !frisk.local_reduction !1
  %local.reduce270 = call float @llvm.maxnum.f32(float %local.reduce266, float %869), !dbg !27, !frisk.local_reduction !1
  %local.reduce271 = call float @llvm.maxnum.f32(float %local.reduce267, float %870), !dbg !27, !frisk.local_reduction !1
  %local.reduce272 = call float @llvm.maxnum.f32(float %local.reduce268, float %871), !dbg !27, !frisk.local_reduction !1
  %local.reduce273 = call float @llvm.maxnum.f32(float %local.reduce269, float %872), !dbg !27, !frisk.local_reduction !1
  %local.reduce274 = call float @llvm.maxnum.f32(float %local.reduce270, float %873), !dbg !27, !frisk.local_reduction !1
  %local.reduce275 = call float @llvm.maxnum.f32(float %local.reduce271, float %874), !dbg !27, !frisk.local_reduction !1
  %local.reduce276 = call float @llvm.maxnum.f32(float %local.reduce272, float %875), !dbg !27, !frisk.local_reduction !1
  %local.reduce277 = call float @llvm.maxnum.f32(float %local.reduce273, float %876), !dbg !27, !frisk.local_reduction !1
  %local.reduce278 = call float @llvm.maxnum.f32(float %local.reduce274, float %877), !dbg !27, !frisk.local_reduction !1
  %local.reduce279 = call float @llvm.maxnum.f32(float %local.reduce275, float %local.reduce276), !dbg !27, !frisk.local_reduction !1
  %local.reduce280 = call float @llvm.maxnum.f32(float %local.reduce277, float %local.reduce278), !dbg !27, !frisk.local_reduction !1
  %911 = call float @llvm.maxnum.f32(float %local.reduce279, float %local.reduce280), !dbg !27, !frisk.local_reduction !1
  %912 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %913 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %912), !dbg !27
  %914 = add nuw nsw i32 %913, 64, !dbg !27
  %915 = xor i32 %913, 16, !dbg !27
  %916 = and i32 %914, 192, !dbg !27
  %917 = icmp samesign ult i32 %915, %916, !dbg !27
  %918 = select i1 %917, i32 %915, i32 %913, !dbg !27
  %919 = shl nuw nsw i32 %918, 2, !dbg !27
  %920 = bitcast float %911 to i32, !dbg !27
  %921 = call i32 @llvm.amdgcn.ds.bpermute(i32 %919, i32 %920), !dbg !27
  %922 = bitcast i32 %921 to float, !dbg !27
  %923 = call float @llvm.maxnum.f32(float %911, float %922), !dbg !27
  %924 = xor i32 %913, 32, !dbg !27
  %925 = icmp samesign ult i32 %924, %916, !dbg !27
  %926 = select i1 %925, i32 %924, i32 %913, !dbg !27
  %927 = shl nuw nsw i32 %926, 2, !dbg !27
  %928 = bitcast float %923 to i32, !dbg !27
  %929 = call i32 @llvm.amdgcn.ds.bpermute(i32 %927, i32 %928), !dbg !27
  %930 = bitcast i32 %929 to float, !dbg !27
  %931 = call float @llvm.maxnum.f32(float %923, float %930), !dbg !27
  %932 = extractvalue [2 x <1 x float>] %208, 0, !dbg !27
  %933 = insertelement <1 x float> poison, float %931, i64 0, !dbg !27
  %934 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %933, <1 x float> %932), !dbg !27
  %935 = call float @llvm.maxnum.f32(float %878, float 0xFFF0000000000000), !dbg !27, !frisk.local_reduction !1
  %local.reduce220 = call float @llvm.maxnum.f32(float %935, float %882), !dbg !27, !frisk.local_reduction !1
  %local.reduce221 = call float @llvm.maxnum.f32(float %879, float %883), !dbg !27, !frisk.local_reduction !1
  %local.reduce222 = call float @llvm.maxnum.f32(float %880, float %884), !dbg !27, !frisk.local_reduction !1
  %local.reduce223 = call float @llvm.maxnum.f32(float %881, float %885), !dbg !27, !frisk.local_reduction !1
  %local.reduce224 = call float @llvm.maxnum.f32(float %local.reduce220, float %886), !dbg !27, !frisk.local_reduction !1
  %local.reduce225 = call float @llvm.maxnum.f32(float %local.reduce221, float %887), !dbg !27, !frisk.local_reduction !1
  %local.reduce226 = call float @llvm.maxnum.f32(float %local.reduce222, float %888), !dbg !27, !frisk.local_reduction !1
  %local.reduce227 = call float @llvm.maxnum.f32(float %local.reduce223, float %889), !dbg !27, !frisk.local_reduction !1
  %local.reduce228 = call float @llvm.maxnum.f32(float %local.reduce224, float %890), !dbg !27, !frisk.local_reduction !1
  %local.reduce229 = call float @llvm.maxnum.f32(float %local.reduce225, float %891), !dbg !27, !frisk.local_reduction !1
  %local.reduce230 = call float @llvm.maxnum.f32(float %local.reduce226, float %892), !dbg !27, !frisk.local_reduction !1
  %local.reduce231 = call float @llvm.maxnum.f32(float %local.reduce227, float %893), !dbg !27, !frisk.local_reduction !1
  %local.reduce232 = call float @llvm.maxnum.f32(float %local.reduce228, float %894), !dbg !27, !frisk.local_reduction !1
  %local.reduce233 = call float @llvm.maxnum.f32(float %local.reduce229, float %895), !dbg !27, !frisk.local_reduction !1
  %local.reduce234 = call float @llvm.maxnum.f32(float %local.reduce230, float %896), !dbg !27, !frisk.local_reduction !1
  %local.reduce235 = call float @llvm.maxnum.f32(float %local.reduce231, float %897), !dbg !27, !frisk.local_reduction !1
  %local.reduce236 = call float @llvm.maxnum.f32(float %local.reduce232, float %898), !dbg !27, !frisk.local_reduction !1
  %local.reduce237 = call float @llvm.maxnum.f32(float %local.reduce233, float %899), !dbg !27, !frisk.local_reduction !1
  %local.reduce238 = call float @llvm.maxnum.f32(float %local.reduce234, float %900), !dbg !27, !frisk.local_reduction !1
  %local.reduce239 = call float @llvm.maxnum.f32(float %local.reduce235, float %901), !dbg !27, !frisk.local_reduction !1
  %local.reduce240 = call float @llvm.maxnum.f32(float %local.reduce236, float %902), !dbg !27, !frisk.local_reduction !1
  %local.reduce241 = call float @llvm.maxnum.f32(float %local.reduce237, float %903), !dbg !27, !frisk.local_reduction !1
  %local.reduce242 = call float @llvm.maxnum.f32(float %local.reduce238, float %904), !dbg !27, !frisk.local_reduction !1
  %local.reduce243 = call float @llvm.maxnum.f32(float %local.reduce239, float %905), !dbg !27, !frisk.local_reduction !1
  %local.reduce244 = call float @llvm.maxnum.f32(float %local.reduce240, float %906), !dbg !27, !frisk.local_reduction !1
  %local.reduce245 = call float @llvm.maxnum.f32(float %local.reduce241, float %907), !dbg !27, !frisk.local_reduction !1
  %local.reduce246 = call float @llvm.maxnum.f32(float %local.reduce242, float %908), !dbg !27, !frisk.local_reduction !1
  %local.reduce247 = call float @llvm.maxnum.f32(float %local.reduce243, float %909), !dbg !27, !frisk.local_reduction !1
  %local.reduce248 = call float @llvm.maxnum.f32(float %local.reduce244, float %local.reduce245), !dbg !27, !frisk.local_reduction !1
  %local.reduce249 = call float @llvm.maxnum.f32(float %local.reduce246, float %local.reduce247), !dbg !27, !frisk.local_reduction !1
  %936 = call float @llvm.maxnum.f32(float %local.reduce248, float %local.reduce249), !dbg !27, !frisk.local_reduction !1
  %937 = bitcast float %936 to i32, !dbg !27
  %938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %919, i32 %937), !dbg !27
  %939 = bitcast i32 %938 to float, !dbg !27
  %940 = call float @llvm.maxnum.f32(float %936, float %939), !dbg !27
  %941 = bitcast float %940 to i32, !dbg !27
  %942 = call i32 @llvm.amdgcn.ds.bpermute(i32 %927, i32 %941), !dbg !27
  %943 = bitcast i32 %942 to float, !dbg !27
  %944 = call float @llvm.maxnum.f32(float %940, float %943), !dbg !27
  %945 = extractvalue [2 x <1 x float>] %208, 1, !dbg !27
  %946 = insertelement <1 x float> poison, float %944, i64 0, !dbg !27
  %947 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %946, <1 x float> %945), !dbg !27
  %948 = shufflevector <1 x float> %934, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %949 = fsub <4 x float> %829, %948, !dbg !28
  %950 = fsub <4 x float> %830, %948, !dbg !28
  %951 = fsub <4 x float> %831, %948, !dbg !28
  %952 = fsub <4 x float> %832, %948, !dbg !28
  %953 = fsub <4 x float> %833, %948, !dbg !28
  %954 = fsub <4 x float> %834, %948, !dbg !28
  %955 = fsub <4 x float> %835, %948, !dbg !28
  %956 = fsub <4 x float> %836, %948, !dbg !28
  %957 = shufflevector <1 x float> %947, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %958 = fsub <4 x float> %837, %957, !dbg !28
  %959 = fsub <4 x float> %838, %957, !dbg !28
  %960 = fsub <4 x float> %839, %957, !dbg !28
  %961 = fsub <4 x float> %840, %957, !dbg !28
  %962 = fsub <4 x float> %841, %957, !dbg !28
  %963 = fsub <4 x float> %842, %957, !dbg !28
  %964 = fsub <4 x float> %843, %957, !dbg !28
  %965 = fsub <4 x float> %844, %957, !dbg !28
  %966 = fsub <1 x float> %932, %934, !dbg !29
  %967 = extractelement <1 x float> %966, i64 0, !dbg !30
  %968 = call float @llvm.amdgcn.exp2.f32(float %967), !dbg !30
  %969 = insertelement <4 x float> poison, float %968, i64 0, !dbg !35
  %970 = shufflevector <4 x float> %969, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %971 = extractvalue [2 x <1 x float>] %207, 0, !dbg !31
  %972 = extractelement <1 x float> %971, i64 0, !dbg !31
  %973 = fmul float %972, %968, !dbg !31
  %974 = fsub <1 x float> %945, %947, !dbg !29
  %975 = extractelement <1 x float> %974, i64 0, !dbg !30
  %976 = call float @llvm.amdgcn.exp2.f32(float %975), !dbg !30
  %977 = insertelement <4 x float> poison, float %976, i64 0, !dbg !35
  %978 = shufflevector <4 x float> %977, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %979 = extractvalue [2 x <1 x float>] %207, 1, !dbg !31
  %980 = extractelement <1 x float> %979, i64 0, !dbg !31
  %981 = fmul float %980, %976, !dbg !31
  %982 = extractelement <4 x float> %949, i64 0, !dbg !32
  %983 = call float @llvm.amdgcn.exp2.f32(float %982), !dbg !32
  %984 = insertelement <4 x float> poison, float %983, i64 0, !dbg !34
  %985 = extractelement <4 x float> %949, i64 1, !dbg !32
  %986 = call float @llvm.amdgcn.exp2.f32(float %985), !dbg !32
  %987 = insertelement <4 x float> %984, float %986, i64 1, !dbg !34
  %988 = extractelement <4 x float> %949, i64 2, !dbg !32
  %989 = call float @llvm.amdgcn.exp2.f32(float %988), !dbg !32
  %990 = insertelement <4 x float> %987, float %989, i64 2, !dbg !34
  %991 = extractelement <4 x float> %949, i64 3, !dbg !32
  %992 = call float @llvm.amdgcn.exp2.f32(float %991), !dbg !32
  %993 = insertelement <4 x float> %990, float %992, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %993) #9, !dbg !34, !frisk.register_lifetimes !1
  %994 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %994) #9, !dbg !32, !frisk.register_lifetimes !1
  %995 = extractelement <4 x float> %950, i64 0, !dbg !32
  %996 = call float @llvm.amdgcn.exp2.f32(float %995), !dbg !32
  %997 = insertelement <4 x float> poison, float %996, i64 0, !dbg !34
  %local.reduce189 = fadd float %983, %996, !dbg !33, !frisk.local_reduction !1
  %998 = extractelement <4 x float> %950, i64 1, !dbg !32
  %999 = call float @llvm.amdgcn.exp2.f32(float %998), !dbg !32
  %1000 = insertelement <4 x float> %997, float %999, i64 1, !dbg !34
  %local.reduce190 = fadd float %986, %999, !dbg !33, !frisk.local_reduction !1
  %1001 = extractelement <4 x float> %950, i64 2, !dbg !32
  %1002 = call float @llvm.amdgcn.exp2.f32(float %1001), !dbg !32
  %1003 = insertelement <4 x float> %1000, float %1002, i64 2, !dbg !34
  %local.reduce191 = fadd float %989, %1002, !dbg !33, !frisk.local_reduction !1
  %1004 = extractelement <4 x float> %950, i64 3, !dbg !32
  %1005 = call float @llvm.amdgcn.exp2.f32(float %1004), !dbg !32
  %1006 = insertelement <4 x float> %1003, float %1005, i64 3, !dbg !34
  %math.result475 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1006) #9, !dbg !34, !frisk.register_lifetimes !1
  %1007 = fptrunc <4 x float> %math.result475 to <4 x half>, !dbg !34
  %math.packed476 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1007) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce192 = fadd float %992, %1005, !dbg !33, !frisk.local_reduction !1
  %1008 = extractelement <4 x float> %951, i64 0, !dbg !32
  %1009 = call float @llvm.amdgcn.exp2.f32(float %1008), !dbg !32
  %1010 = insertelement <4 x float> poison, float %1009, i64 0, !dbg !34
  %local.reduce193 = fadd float %local.reduce189, %1009, !dbg !33, !frisk.local_reduction !1
  %1011 = extractelement <4 x float> %951, i64 1, !dbg !32
  %1012 = call float @llvm.amdgcn.exp2.f32(float %1011), !dbg !32
  %1013 = insertelement <4 x float> %1010, float %1012, i64 1, !dbg !34
  %local.reduce194 = fadd float %local.reduce190, %1012, !dbg !33, !frisk.local_reduction !1
  %1014 = extractelement <4 x float> %951, i64 2, !dbg !32
  %1015 = call float @llvm.amdgcn.exp2.f32(float %1014), !dbg !32
  %1016 = insertelement <4 x float> %1013, float %1015, i64 2, !dbg !34
  %local.reduce195 = fadd float %local.reduce191, %1015, !dbg !33, !frisk.local_reduction !1
  %1017 = extractelement <4 x float> %951, i64 3, !dbg !32
  %1018 = call float @llvm.amdgcn.exp2.f32(float %1017), !dbg !32
  %1019 = insertelement <4 x float> %1016, float %1018, i64 3, !dbg !34
  %math.result477 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1019) #9, !dbg !34, !frisk.register_lifetimes !1
  %1020 = fptrunc <4 x float> %math.result477 to <4 x half>, !dbg !34
  %math.packed478 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1020) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce196 = fadd float %local.reduce192, %1018, !dbg !33, !frisk.local_reduction !1
  %1021 = extractelement <4 x float> %952, i64 0, !dbg !32
  %1022 = call float @llvm.amdgcn.exp2.f32(float %1021), !dbg !32
  %1023 = insertelement <4 x float> poison, float %1022, i64 0, !dbg !34
  %local.reduce197 = fadd float %local.reduce193, %1022, !dbg !33, !frisk.local_reduction !1
  %1024 = extractelement <4 x float> %952, i64 1, !dbg !32
  %1025 = call float @llvm.amdgcn.exp2.f32(float %1024), !dbg !32
  %1026 = insertelement <4 x float> %1023, float %1025, i64 1, !dbg !34
  %local.reduce198 = fadd float %local.reduce194, %1025, !dbg !33, !frisk.local_reduction !1
  %1027 = extractelement <4 x float> %952, i64 2, !dbg !32
  %1028 = call float @llvm.amdgcn.exp2.f32(float %1027), !dbg !32
  %1029 = insertelement <4 x float> %1026, float %1028, i64 2, !dbg !34
  %local.reduce199 = fadd float %local.reduce195, %1028, !dbg !33, !frisk.local_reduction !1
  %1030 = extractelement <4 x float> %952, i64 3, !dbg !32
  %1031 = call float @llvm.amdgcn.exp2.f32(float %1030), !dbg !32
  %1032 = insertelement <4 x float> %1029, float %1031, i64 3, !dbg !34
  %math.result479 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1032) #9, !dbg !34, !frisk.register_lifetimes !1
  %1033 = fptrunc <4 x float> %math.result479 to <4 x half>, !dbg !34
  %math.packed480 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1033) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce200 = fadd float %local.reduce196, %1031, !dbg !33, !frisk.local_reduction !1
  %1034 = extractelement <4 x float> %953, i64 0, !dbg !32
  %1035 = call float @llvm.amdgcn.exp2.f32(float %1034), !dbg !32
  %1036 = insertelement <4 x float> poison, float %1035, i64 0, !dbg !34
  %local.reduce201 = fadd float %local.reduce197, %1035, !dbg !33, !frisk.local_reduction !1
  %1037 = extractelement <4 x float> %953, i64 1, !dbg !32
  %1038 = call float @llvm.amdgcn.exp2.f32(float %1037), !dbg !32
  %1039 = insertelement <4 x float> %1036, float %1038, i64 1, !dbg !34
  %local.reduce202 = fadd float %local.reduce198, %1038, !dbg !33, !frisk.local_reduction !1
  %1040 = extractelement <4 x float> %953, i64 2, !dbg !32
  %1041 = call float @llvm.amdgcn.exp2.f32(float %1040), !dbg !32
  %1042 = insertelement <4 x float> %1039, float %1041, i64 2, !dbg !34
  %local.reduce203 = fadd float %local.reduce199, %1041, !dbg !33, !frisk.local_reduction !1
  %1043 = extractelement <4 x float> %953, i64 3, !dbg !32
  %1044 = call float @llvm.amdgcn.exp2.f32(float %1043), !dbg !32
  %1045 = insertelement <4 x float> %1042, float %1044, i64 3, !dbg !34
  %math.result481 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1045) #9, !dbg !34, !frisk.register_lifetimes !1
  %1046 = fptrunc <4 x float> %math.result481 to <4 x half>, !dbg !34
  %math.packed482 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1046) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce204 = fadd float %local.reduce200, %1044, !dbg !33, !frisk.local_reduction !1
  %1047 = extractelement <4 x float> %954, i64 0, !dbg !32
  %1048 = call float @llvm.amdgcn.exp2.f32(float %1047), !dbg !32
  %1049 = insertelement <4 x float> poison, float %1048, i64 0, !dbg !34
  %local.reduce205 = fadd float %local.reduce201, %1048, !dbg !33, !frisk.local_reduction !1
  %1050 = extractelement <4 x float> %954, i64 1, !dbg !32
  %1051 = call float @llvm.amdgcn.exp2.f32(float %1050), !dbg !32
  %1052 = insertelement <4 x float> %1049, float %1051, i64 1, !dbg !34
  %local.reduce206 = fadd float %local.reduce202, %1051, !dbg !33, !frisk.local_reduction !1
  %1053 = extractelement <4 x float> %954, i64 2, !dbg !32
  %1054 = call float @llvm.amdgcn.exp2.f32(float %1053), !dbg !32
  %1055 = insertelement <4 x float> %1052, float %1054, i64 2, !dbg !34
  %local.reduce207 = fadd float %local.reduce203, %1054, !dbg !33, !frisk.local_reduction !1
  %1056 = extractelement <4 x float> %954, i64 3, !dbg !32
  %1057 = call float @llvm.amdgcn.exp2.f32(float %1056), !dbg !32
  %1058 = insertelement <4 x float> %1055, float %1057, i64 3, !dbg !34
  %math.result483 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1058) #9, !dbg !34, !frisk.register_lifetimes !1
  %1059 = fptrunc <4 x float> %math.result483 to <4 x half>, !dbg !34
  %math.packed484 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1059) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce208 = fadd float %local.reduce204, %1057, !dbg !33, !frisk.local_reduction !1
  %1060 = extractelement <4 x float> %955, i64 0, !dbg !32
  %1061 = call float @llvm.amdgcn.exp2.f32(float %1060), !dbg !32
  %1062 = insertelement <4 x float> poison, float %1061, i64 0, !dbg !34
  %local.reduce209 = fadd float %local.reduce205, %1061, !dbg !33, !frisk.local_reduction !1
  %1063 = extractelement <4 x float> %955, i64 1, !dbg !32
  %1064 = call float @llvm.amdgcn.exp2.f32(float %1063), !dbg !32
  %1065 = insertelement <4 x float> %1062, float %1064, i64 1, !dbg !34
  %local.reduce210 = fadd float %local.reduce206, %1064, !dbg !33, !frisk.local_reduction !1
  %1066 = extractelement <4 x float> %955, i64 2, !dbg !32
  %1067 = call float @llvm.amdgcn.exp2.f32(float %1066), !dbg !32
  %1068 = insertelement <4 x float> %1065, float %1067, i64 2, !dbg !34
  %local.reduce211 = fadd float %local.reduce207, %1067, !dbg !33, !frisk.local_reduction !1
  %1069 = extractelement <4 x float> %955, i64 3, !dbg !32
  %1070 = call float @llvm.amdgcn.exp2.f32(float %1069), !dbg !32
  %1071 = insertelement <4 x float> %1068, float %1070, i64 3, !dbg !34
  %math.result485 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1071) #9, !dbg !34, !frisk.register_lifetimes !1
  %1072 = fptrunc <4 x float> %math.result485 to <4 x half>, !dbg !34
  %math.packed486 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1072) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce212 = fadd float %local.reduce208, %1070, !dbg !33, !frisk.local_reduction !1
  %1073 = extractelement <4 x float> %956, i64 0, !dbg !32
  %1074 = call float @llvm.amdgcn.exp2.f32(float %1073), !dbg !32
  %1075 = insertelement <4 x float> poison, float %1074, i64 0, !dbg !34
  %local.reduce213 = fadd float %local.reduce209, %1074, !dbg !33, !frisk.local_reduction !1
  %1076 = extractelement <4 x float> %956, i64 1, !dbg !32
  %1077 = call float @llvm.amdgcn.exp2.f32(float %1076), !dbg !32
  %1078 = insertelement <4 x float> %1075, float %1077, i64 1, !dbg !34
  %local.reduce214 = fadd float %local.reduce210, %1077, !dbg !33, !frisk.local_reduction !1
  %local.reduce217 = fadd float %local.reduce213, %local.reduce214, !dbg !33, !frisk.local_reduction !1
  %1079 = extractelement <4 x float> %956, i64 2, !dbg !32
  %1080 = call float @llvm.amdgcn.exp2.f32(float %1079), !dbg !32
  %1081 = insertelement <4 x float> %1078, float %1080, i64 2, !dbg !34
  %local.reduce215 = fadd float %local.reduce211, %1080, !dbg !33, !frisk.local_reduction !1
  %1082 = extractelement <4 x float> %956, i64 3, !dbg !32
  %1083 = call float @llvm.amdgcn.exp2.f32(float %1082), !dbg !32
  %1084 = insertelement <4 x float> %1081, float %1083, i64 3, !dbg !34
  %math.result487 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1084) #9, !dbg !34, !frisk.register_lifetimes !1
  %1085 = fptrunc <4 x float> %math.result487 to <4 x half>, !dbg !34
  %math.packed488 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1085) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce216 = fadd float %local.reduce212, %1083, !dbg !33, !frisk.local_reduction !1
  %local.reduce218 = fadd float %local.reduce215, %local.reduce216, !dbg !33, !frisk.local_reduction !1
  %1086 = fadd float %local.reduce217, %local.reduce218, !dbg !33, !frisk.local_reduction !1
  %1087 = fadd float %1086, %973, !dbg !33
  %1088 = insertelement <1 x float> poison, float %1087, i64 0, !dbg !7
  %1089 = insertvalue [2 x <1 x float>] poison, <1 x float> %1088, 0, !dbg !7
  %1090 = extractelement <4 x float> %958, i64 0, !dbg !32
  %1091 = call float @llvm.amdgcn.exp2.f32(float %1090), !dbg !32
  %1092 = insertelement <4 x float> poison, float %1091, i64 0, !dbg !34
  %1093 = extractelement <4 x float> %958, i64 1, !dbg !32
  %1094 = call float @llvm.amdgcn.exp2.f32(float %1093), !dbg !32
  %1095 = insertelement <4 x float> %1092, float %1094, i64 1, !dbg !34
  %1096 = extractelement <4 x float> %958, i64 2, !dbg !32
  %1097 = call float @llvm.amdgcn.exp2.f32(float %1096), !dbg !32
  %1098 = insertelement <4 x float> %1095, float %1097, i64 2, !dbg !34
  %1099 = extractelement <4 x float> %958, i64 3, !dbg !32
  %1100 = call float @llvm.amdgcn.exp2.f32(float %1099), !dbg !32
  %1101 = insertelement <4 x float> %1098, float %1100, i64 3, !dbg !34
  %math.result489 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1101) #9, !dbg !34, !frisk.register_lifetimes !1
  %1102 = fptrunc <4 x float> %math.result489 to <4 x half>, !dbg !34
  %math.packed490 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1102) #9, !dbg !32, !frisk.register_lifetimes !1
  %1103 = extractelement <4 x float> %959, i64 0, !dbg !32
  %1104 = call float @llvm.amdgcn.exp2.f32(float %1103), !dbg !32
  %1105 = insertelement <4 x float> poison, float %1104, i64 0, !dbg !34
  %local.reduce = fadd float %1091, %1104, !dbg !33, !frisk.local_reduction !1
  %1106 = extractelement <4 x float> %959, i64 1, !dbg !32
  %1107 = call float @llvm.amdgcn.exp2.f32(float %1106), !dbg !32
  %1108 = insertelement <4 x float> %1105, float %1107, i64 1, !dbg !34
  %local.reduce159 = fadd float %1094, %1107, !dbg !33, !frisk.local_reduction !1
  %1109 = extractelement <4 x float> %959, i64 2, !dbg !32
  %1110 = call float @llvm.amdgcn.exp2.f32(float %1109), !dbg !32
  %1111 = insertelement <4 x float> %1108, float %1110, i64 2, !dbg !34
  %local.reduce160 = fadd float %1097, %1110, !dbg !33, !frisk.local_reduction !1
  %1112 = extractelement <4 x float> %959, i64 3, !dbg !32
  %1113 = call float @llvm.amdgcn.exp2.f32(float %1112), !dbg !32
  %1114 = insertelement <4 x float> %1111, float %1113, i64 3, !dbg !34
  %math.result491 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1114) #9, !dbg !34, !frisk.register_lifetimes !1
  %1115 = fptrunc <4 x float> %math.result491 to <4 x half>, !dbg !34
  %math.packed492 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1115) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce161 = fadd float %1100, %1113, !dbg !33, !frisk.local_reduction !1
  %1116 = extractelement <4 x float> %960, i64 0, !dbg !32
  %1117 = call float @llvm.amdgcn.exp2.f32(float %1116), !dbg !32
  %1118 = insertelement <4 x float> poison, float %1117, i64 0, !dbg !34
  %local.reduce162 = fadd float %local.reduce, %1117, !dbg !33, !frisk.local_reduction !1
  %1119 = extractelement <4 x float> %960, i64 1, !dbg !32
  %1120 = call float @llvm.amdgcn.exp2.f32(float %1119), !dbg !32
  %1121 = insertelement <4 x float> %1118, float %1120, i64 1, !dbg !34
  %local.reduce163 = fadd float %local.reduce159, %1120, !dbg !33, !frisk.local_reduction !1
  %1122 = extractelement <4 x float> %960, i64 2, !dbg !32
  %1123 = call float @llvm.amdgcn.exp2.f32(float %1122), !dbg !32
  %1124 = insertelement <4 x float> %1121, float %1123, i64 2, !dbg !34
  %local.reduce164 = fadd float %local.reduce160, %1123, !dbg !33, !frisk.local_reduction !1
  %1125 = extractelement <4 x float> %960, i64 3, !dbg !32
  %1126 = call float @llvm.amdgcn.exp2.f32(float %1125), !dbg !32
  %1127 = insertelement <4 x float> %1124, float %1126, i64 3, !dbg !34
  %math.result493 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1127) #9, !dbg !34, !frisk.register_lifetimes !1
  %1128 = fptrunc <4 x float> %math.result493 to <4 x half>, !dbg !34
  %math.packed494 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1128) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce165 = fadd float %local.reduce161, %1126, !dbg !33, !frisk.local_reduction !1
  %1129 = extractelement <4 x float> %961, i64 0, !dbg !32
  %1130 = call float @llvm.amdgcn.exp2.f32(float %1129), !dbg !32
  %1131 = insertelement <4 x float> poison, float %1130, i64 0, !dbg !34
  %local.reduce166 = fadd float %local.reduce162, %1130, !dbg !33, !frisk.local_reduction !1
  %1132 = extractelement <4 x float> %961, i64 1, !dbg !32
  %1133 = call float @llvm.amdgcn.exp2.f32(float %1132), !dbg !32
  %1134 = insertelement <4 x float> %1131, float %1133, i64 1, !dbg !34
  %local.reduce167 = fadd float %local.reduce163, %1133, !dbg !33, !frisk.local_reduction !1
  %1135 = extractelement <4 x float> %961, i64 2, !dbg !32
  %1136 = call float @llvm.amdgcn.exp2.f32(float %1135), !dbg !32
  %1137 = insertelement <4 x float> %1134, float %1136, i64 2, !dbg !34
  %local.reduce168 = fadd float %local.reduce164, %1136, !dbg !33, !frisk.local_reduction !1
  %1138 = extractelement <4 x float> %961, i64 3, !dbg !32
  %1139 = call float @llvm.amdgcn.exp2.f32(float %1138), !dbg !32
  %1140 = insertelement <4 x float> %1137, float %1139, i64 3, !dbg !34
  %math.result495 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1140) #9, !dbg !34, !frisk.register_lifetimes !1
  %1141 = fptrunc <4 x float> %math.result495 to <4 x half>, !dbg !34
  %math.packed496 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1141) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce169 = fadd float %local.reduce165, %1139, !dbg !33, !frisk.local_reduction !1
  %1142 = extractelement <4 x float> %962, i64 0, !dbg !32
  %1143 = call float @llvm.amdgcn.exp2.f32(float %1142), !dbg !32
  %1144 = insertelement <4 x float> poison, float %1143, i64 0, !dbg !34
  %local.reduce170 = fadd float %local.reduce166, %1143, !dbg !33, !frisk.local_reduction !1
  %1145 = extractelement <4 x float> %962, i64 1, !dbg !32
  %1146 = call float @llvm.amdgcn.exp2.f32(float %1145), !dbg !32
  %1147 = insertelement <4 x float> %1144, float %1146, i64 1, !dbg !34
  %local.reduce171 = fadd float %local.reduce167, %1146, !dbg !33, !frisk.local_reduction !1
  %1148 = extractelement <4 x float> %962, i64 2, !dbg !32
  %1149 = call float @llvm.amdgcn.exp2.f32(float %1148), !dbg !32
  %1150 = insertelement <4 x float> %1147, float %1149, i64 2, !dbg !34
  %local.reduce172 = fadd float %local.reduce168, %1149, !dbg !33, !frisk.local_reduction !1
  %1151 = extractelement <4 x float> %962, i64 3, !dbg !32
  %1152 = call float @llvm.amdgcn.exp2.f32(float %1151), !dbg !32
  %1153 = insertelement <4 x float> %1150, float %1152, i64 3, !dbg !34
  %math.result497 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1153) #9, !dbg !34, !frisk.register_lifetimes !1
  %1154 = fptrunc <4 x float> %math.result497 to <4 x half>, !dbg !34
  %math.packed498 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1154) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce173 = fadd float %local.reduce169, %1152, !dbg !33, !frisk.local_reduction !1
  %1155 = extractelement <4 x float> %963, i64 0, !dbg !32
  %1156 = call float @llvm.amdgcn.exp2.f32(float %1155), !dbg !32
  %1157 = insertelement <4 x float> poison, float %1156, i64 0, !dbg !34
  %local.reduce174 = fadd float %local.reduce170, %1156, !dbg !33, !frisk.local_reduction !1
  %1158 = extractelement <4 x float> %963, i64 1, !dbg !32
  %1159 = call float @llvm.amdgcn.exp2.f32(float %1158), !dbg !32
  %1160 = insertelement <4 x float> %1157, float %1159, i64 1, !dbg !34
  %local.reduce175 = fadd float %local.reduce171, %1159, !dbg !33, !frisk.local_reduction !1
  %1161 = extractelement <4 x float> %963, i64 2, !dbg !32
  %1162 = call float @llvm.amdgcn.exp2.f32(float %1161), !dbg !32
  %1163 = insertelement <4 x float> %1160, float %1162, i64 2, !dbg !34
  %local.reduce176 = fadd float %local.reduce172, %1162, !dbg !33, !frisk.local_reduction !1
  %1164 = extractelement <4 x float> %963, i64 3, !dbg !32
  %1165 = call float @llvm.amdgcn.exp2.f32(float %1164), !dbg !32
  %1166 = insertelement <4 x float> %1163, float %1165, i64 3, !dbg !34
  %math.result499 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1166) #9, !dbg !34, !frisk.register_lifetimes !1
  %1167 = fptrunc <4 x float> %math.result499 to <4 x half>, !dbg !34
  %math.packed500 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1167) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce177 = fadd float %local.reduce173, %1165, !dbg !33, !frisk.local_reduction !1
  %1168 = extractelement <4 x float> %964, i64 0, !dbg !32
  %1169 = call float @llvm.amdgcn.exp2.f32(float %1168), !dbg !32
  %1170 = insertelement <4 x float> poison, float %1169, i64 0, !dbg !34
  %local.reduce178 = fadd float %local.reduce174, %1169, !dbg !33, !frisk.local_reduction !1
  %1171 = extractelement <4 x float> %964, i64 1, !dbg !32
  %1172 = call float @llvm.amdgcn.exp2.f32(float %1171), !dbg !32
  %1173 = insertelement <4 x float> %1170, float %1172, i64 1, !dbg !34
  %local.reduce179 = fadd float %local.reduce175, %1172, !dbg !33, !frisk.local_reduction !1
  %1174 = extractelement <4 x float> %964, i64 2, !dbg !32
  %1175 = call float @llvm.amdgcn.exp2.f32(float %1174), !dbg !32
  %1176 = insertelement <4 x float> %1173, float %1175, i64 2, !dbg !34
  %local.reduce180 = fadd float %local.reduce176, %1175, !dbg !33, !frisk.local_reduction !1
  %1177 = extractelement <4 x float> %964, i64 3, !dbg !32
  %1178 = call float @llvm.amdgcn.exp2.f32(float %1177), !dbg !32
  %1179 = insertelement <4 x float> %1176, float %1178, i64 3, !dbg !34
  %math.result501 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1179) #9, !dbg !34, !frisk.register_lifetimes !1
  %1180 = fptrunc <4 x float> %math.result501 to <4 x half>, !dbg !34
  %math.packed502 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1180) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce181 = fadd float %local.reduce177, %1178, !dbg !33, !frisk.local_reduction !1
  %1181 = extractelement <4 x float> %965, i64 0, !dbg !32
  %1182 = call float @llvm.amdgcn.exp2.f32(float %1181), !dbg !32
  %1183 = insertelement <4 x float> poison, float %1182, i64 0, !dbg !34
  %local.reduce182 = fadd float %local.reduce178, %1182, !dbg !33, !frisk.local_reduction !1
  %1184 = extractelement <4 x float> %965, i64 1, !dbg !32
  %1185 = call float @llvm.amdgcn.exp2.f32(float %1184), !dbg !32
  %1186 = insertelement <4 x float> %1183, float %1185, i64 1, !dbg !34
  %local.reduce183 = fadd float %local.reduce179, %1185, !dbg !33, !frisk.local_reduction !1
  %local.reduce186 = fadd float %local.reduce182, %local.reduce183, !dbg !33, !frisk.local_reduction !1
  %1187 = extractelement <4 x float> %965, i64 2, !dbg !32
  %1188 = call float @llvm.amdgcn.exp2.f32(float %1187), !dbg !32
  %1189 = insertelement <4 x float> %1186, float %1188, i64 2, !dbg !34
  %local.reduce184 = fadd float %local.reduce180, %1188, !dbg !33, !frisk.local_reduction !1
  %1190 = extractelement <4 x float> %965, i64 3, !dbg !32
  %1191 = call float @llvm.amdgcn.exp2.f32(float %1190), !dbg !32
  %1192 = insertelement <4 x float> %1189, float %1191, i64 3, !dbg !34
  %math.result503 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1192) #9, !dbg !34, !frisk.register_lifetimes !1
  %1193 = fptrunc <4 x float> %math.result503 to <4 x half>, !dbg !34
  %math.packed504 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1193) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce185 = fadd float %local.reduce181, %1191, !dbg !33, !frisk.local_reduction !1
  %local.reduce187 = fadd float %local.reduce184, %local.reduce185, !dbg !33, !frisk.local_reduction !1
  %1194 = fadd float %local.reduce186, %local.reduce187, !dbg !33, !frisk.local_reduction !1
  %1195 = fadd float %1194, %981, !dbg !33
  %1196 = insertelement <1 x float> poison, float %1195, i64 0, !dbg !7
  %1197 = insertvalue [2 x <1 x float>] %1089, <1 x float> %1196, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread1030 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !14, !range !39, !frisk.register_lifetimes !1
  %address.local1031 = shl nuw nsw i32 %address.thread1030, 3, !dbg !14
  %address.local1032 = and i32 %address.local1031, 120, !dbg !14
  %address.local1033 = shl nuw nsw i32 %address.thread1030, 5, !dbg !14
  %address.local1034 = and i32 %address.local1033, 1536, !dbg !14
  %address.local1035 = or disjoint i32 %address.local1032, %address.local1034, !dbg !14
  %address.local1036 = zext nneg i32 %address.local1035 to i64, !dbg !14
  %address.local1037 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1036, !dbg !14
  %1198 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #0, !dbg !14
  %1199 = bitcast <4 x i32> %1198 to <8 x half>, !dbg !14
  %address.local1038 = zext nneg i32 %address.thread1030 to i64, !dbg !8
  %address.local1039 = and i64 %address.local1038, 60, !dbg !14
  %address.local1040 = add nuw nsw i64 %address.local1039, %address.local1038, !dbg !14
  %address.local1041 = shl nuw nsw i64 %address.local1040, 3, !dbg !14
  %address.local1042 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1043 = and i64 %address.local1038, 63, !dbg !11
  %address.local1044 = lshr i64 %address.local1043, 4, !dbg !14
  %address.local1045 = mul nuw nsw i64 %address.local1044, 384, !dbg !14
  %address.local1046 = getelementptr [2 x i8], ptr addrspace(3) %address.local1042, i64 %address.local1045, !dbg !14
  %address.local1047 = lshr i64 %address.local1038, 2, !dbg !17
  %address.local1048 = mul nsw i64 %address.local1047, -32, !dbg !14
  %address.local1049 = getelementptr [2 x i8], ptr addrspace(3) %address.local1046, i64 %address.local1048, !dbg !14
  %address.local1050 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 256, !dbg !14
  %1200 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1201 = bitcast <4 x i32> %1200 to <8 x half>, !dbg !14
  %address.local1051 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1052 = getelementptr [2 x i8], ptr addrspace(3) %address.local1051, i64 %address.local1045, !dbg !14
  %address.local1053 = getelementptr [2 x i8], ptr addrspace(3) %address.local1052, i64 %address.local1048, !dbg !14
  %address.local1054 = getelementptr i8, ptr addrspace(3) %address.local1053, i64 512, !dbg !14
  %1202 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1203 = bitcast <4 x i32> %1202 to <8 x half>, !dbg !14
  %address.local1055 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1056 = getelementptr [2 x i8], ptr addrspace(3) %address.local1055, i64 %address.local1045, !dbg !14
  %address.local1057 = getelementptr [2 x i8], ptr addrspace(3) %address.local1056, i64 %address.local1048, !dbg !14
  %address.local1058 = getelementptr i8, ptr addrspace(3) %address.local1057, i64 768, !dbg !14
  %1204 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1205 = bitcast <4 x i32> %1204 to <8 x half>, !dbg !14
  %address.local1059 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1036, !dbg !14
  %address.local1060 = getelementptr i8, ptr addrspace(3) %address.local1059, i64 4096, !dbg !14
  %1206 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #0, !dbg !14
  %1207 = bitcast <4 x i32> %1206 to <8 x half>, !dbg !14
  %address.local1061 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1062 = getelementptr [2 x i8], ptr addrspace(3) %address.local1061, i64 %address.local1045, !dbg !14
  %address.local1063 = getelementptr [2 x i8], ptr addrspace(3) %address.local1062, i64 %address.local1048, !dbg !14
  %address.local1064 = getelementptr i8, ptr addrspace(3) %address.local1063, i64 4352, !dbg !14
  %1208 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1209 = bitcast <4 x i32> %1208 to <8 x half>, !dbg !14
  %address.local1065 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1066 = getelementptr [2 x i8], ptr addrspace(3) %address.local1065, i64 %address.local1045, !dbg !14
  %address.local1067 = getelementptr [2 x i8], ptr addrspace(3) %address.local1066, i64 %address.local1048, !dbg !14
  %address.local1068 = getelementptr i8, ptr addrspace(3) %address.local1067, i64 4608, !dbg !14
  %1210 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #0, !dbg !14
  %1211 = bitcast <4 x i32> %1210 to <8 x half>, !dbg !14
  %address.local1069 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1070 = getelementptr [2 x i8], ptr addrspace(3) %address.local1069, i64 %address.local1045, !dbg !14
  %address.local1071 = getelementptr [2 x i8], ptr addrspace(3) %address.local1070, i64 %address.local1048, !dbg !14
  %address.local1072 = getelementptr i8, ptr addrspace(3) %address.local1071, i64 4864, !dbg !14
  %1212 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #0, !dbg !14
  %1213 = bitcast <4 x i32> %1212 to <8 x half>, !dbg !14
  %1214 = add nuw nsw i64 %217, %16, !dbg !18
  %1215 = shl i64 %1214, 19, !dbg !18
  %1216 = and i64 %1215, 16252928, !dbg !18
  %1217 = add i64 %1216, %223, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1218 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1217, !dbg !18
  %1219 = getelementptr i8, ptr addrspace(1) %1218, i64 64, !dbg !18
  %address.local1073 = shl nuw nsw i64 %address.local1038, 3, !dbg !8
  %address.local1074 = and i64 %address.local1073, 1536, !dbg !8
  %address.local1075 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1074, !dbg !8
  %address.local1076 = ptrtoint ptr addrspace(3) %address.local1075 to i64, !dbg !8
  %address.local1077 = trunc i64 %address.local1076 to i32, !dbg !8
  %1220 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1077), !dbg !18
  %1221 = zext i32 %1220 to i64, !dbg !18
  %1222 = inttoptr i64 %1221 to ptr addrspace(3), !dbg !18
  %address.local1078 = shl nuw nsw i64 %address.local1038, 5, !dbg !18
  %address.local1079 = and i64 %address.local1078, 8064, !dbg !18
  %address.local1080 = shl nuw nsw i64 %address.local1038, 3, !dbg !18
  %address.local1081 = and i64 %address.local1080, 24, !dbg !18
  %address.local1082 = or disjoint i64 %address.local1079, %address.local1081, !dbg !18
  %address.local1083 = getelementptr [2 x i8], ptr addrspace(1) %1219, i64 %address.local1082, !dbg !18
  %1223 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1224 = ptrtoint ptr addrspace(1) %address.local1083 to i64, !dbg !18
  %1225 = sub i64 %1224, %1223, !dbg !18
  %dma.buffer.offset1724 = trunc i64 %1225 to i32, !dbg !18
  %1226 = trunc i64 %1223 to i32, !dbg !18
  %1227 = insertelement <4 x i32> poison, i32 %1226, i64 0, !dbg !18
  %1228 = lshr i64 %1223, 32, !dbg !18
  %1229 = trunc i64 %1228 to i32, !dbg !18
  %1230 = insertelement <4 x i32> %1227, i32 %1229, i64 1, !dbg !18
  %1231 = insertelement <4 x i32> %1230, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1725 = insertelement <4 x i32> %1231, i32 131072, i64 3, !dbg !18
  %1232 = ptrtoint ptr addrspace(3) %1222 to i32, !dbg !18
  %dma.buffer.lds1726 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1232), !dbg !18
  %1233 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1724, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !18
  %address.local1084 = or disjoint i64 %address.local1038, 256, !dbg !8
  %address.local1085 = shl nuw nsw i64 %address.local1084, 3, !dbg !8
  %address.local1086 = and i64 %address.local1085, 3584, !dbg !8
  %address.local1087 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1086, !dbg !8
  %address.local1088 = ptrtoint ptr addrspace(3) %address.local1087 to i64, !dbg !8
  %address.local1089 = trunc i64 %address.local1088 to i32, !dbg !8
  %1234 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1089), !dbg !18
  %1235 = zext i32 %1234 to i64, !dbg !18
  %1236 = inttoptr i64 %1235 to ptr addrspace(3), !dbg !18
  %address.local1090 = shl nuw nsw i64 %address.local1084, 5, !dbg !18
  %address.local1091 = and i64 %address.local1090, 16256, !dbg !18
  %address.local1092 = shl nuw nsw i64 %address.local1038, 3, !dbg !18
  %address.local1093 = and i64 %address.local1092, 24, !dbg !18
  %address.local1094 = or disjoint i64 %address.local1091, %address.local1093, !dbg !18
  %address.local1095 = getelementptr [2 x i8], ptr addrspace(1) %1219, i64 %address.local1094, !dbg !18
  %1237 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1238 = ptrtoint ptr addrspace(1) %address.local1095 to i64, !dbg !18
  %1239 = sub i64 %1238, %1237, !dbg !18
  %dma.buffer.offset1721 = trunc i64 %1239 to i32, !dbg !18
  %1240 = trunc i64 %1237 to i32, !dbg !18
  %1241 = insertelement <4 x i32> poison, i32 %1240, i64 0, !dbg !18
  %1242 = lshr i64 %1237, 32, !dbg !18
  %1243 = trunc i64 %1242 to i32, !dbg !18
  %1244 = insertelement <4 x i32> %1241, i32 %1243, i64 1, !dbg !18
  %1245 = insertelement <4 x i32> %1244, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1722 = insertelement <4 x i32> %1245, i32 131072, i64 3, !dbg !18
  %1246 = ptrtoint ptr addrspace(3) %1236 to i32, !dbg !18
  %dma.buffer.lds1723 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1246), !dbg !18
  %1247 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset1721, <4 x i32> %dma.buffer.rsrc1722) #10, !dbg !18
  %1248 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1249 = extractvalue [2 x <8 x float>] %206, 0, !dbg !35
  %1250 = shufflevector <8 x float> %1249, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1251 = fmul <4 x float> %1250, %970, !dbg !35
  %1252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1248, <4 x float> %1251) #9, !dbg !14
  %1253 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1253, <4 x float> %1252) #9, !dbg !14
  %1255 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1255, <4 x float> %1254) #9, !dbg !14
  %1257 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1257, <4 x float> %1256) #9, !dbg !14
  %1259 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1259, <4 x float> %1258) #9, !dbg !14
  %1261 = shufflevector <8 x half> %1209, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1261, <4 x float> %1260) #9, !dbg !14
  %1263 = shufflevector <8 x half> %1211, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1263, <4 x float> %1262) #9, !dbg !14
  %1265 = shufflevector <8 x half> %1213, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1265, <4 x float> %1264) #9, !dbg !14
  %1267 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1268 = shufflevector <8 x float> %1249, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1269 = fmul <4 x float> %1268, %970, !dbg !35
  %1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1267, <4 x float> %1269) #9, !dbg !14
  %1271 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1271, <4 x float> %1270) #9, !dbg !14
  %1273 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1273, <4 x float> %1272) #9, !dbg !14
  %1275 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1275, <4 x float> %1274) #9, !dbg !14
  %1277 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1277, <4 x float> %1276) #9, !dbg !14
  %1279 = shufflevector <8 x half> %1209, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1279, <4 x float> %1278) #9, !dbg !14
  %1281 = shufflevector <8 x half> %1211, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1281, <4 x float> %1280) #9, !dbg !14
  %1283 = shufflevector <8 x half> %1213, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1283, <4 x float> %1282) #9, !dbg !14
  %1285 = extractvalue [2 x <8 x float>] %206, 1, !dbg !35
  %1286 = shufflevector <8 x float> %1285, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1287 = fmul <4 x float> %1286, %978, !dbg !35
  %1288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1248, <4 x float> %1287) #9, !dbg !14
  %1289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1253, <4 x float> %1288) #9, !dbg !14
  %1290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1255, <4 x float> %1289) #9, !dbg !14
  %1291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1257, <4 x float> %1290) #9, !dbg !14
  %1292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1259, <4 x float> %1291) #9, !dbg !14
  %1293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1261, <4 x float> %1292) #9, !dbg !14
  %1294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1263, <4 x float> %1293) #9, !dbg !14
  %1295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1265, <4 x float> %1294) #9, !dbg !14
  %1296 = shufflevector <8 x float> %1285, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1297 = fmul <4 x float> %1296, %978, !dbg !35
  %1298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1267, <4 x float> %1297) #9, !dbg !14
  %1299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1271, <4 x float> %1298) #9, !dbg !14
  %1300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1273, <4 x float> %1299) #9, !dbg !14
  %1301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1275, <4 x float> %1300) #9, !dbg !14
  %1302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1277, <4 x float> %1301) #9, !dbg !14
  %1303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1279, <4 x float> %1302) #9, !dbg !14
  %1304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1281, <4 x float> %1303) #9, !dbg !14
  %1305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1283, <4 x float> %1304) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1306 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #0, !dbg !14
  %1307 = bitcast <4 x i32> %1306 to <8 x half>, !dbg !14
  %1308 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1309 = bitcast <4 x i32> %1308 to <8 x half>, !dbg !14
  %1310 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1311 = bitcast <4 x i32> %1310 to <8 x half>, !dbg !14
  %1312 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1313 = bitcast <4 x i32> %1312 to <8 x half>, !dbg !14
  %1314 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #0, !dbg !14
  %1315 = bitcast <4 x i32> %1314 to <8 x half>, !dbg !14
  %1316 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1317 = bitcast <4 x i32> %1316 to <8 x half>, !dbg !14
  %1318 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #0, !dbg !14
  %1319 = bitcast <4 x i32> %1318 to <8 x half>, !dbg !14
  %1320 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #0, !dbg !14
  %1321 = bitcast <4 x i32> %1320 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1322 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1217, !dbg !18
  %1323 = getelementptr i8, ptr addrspace(1) %1322, i64 128, !dbg !18
  %1324 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1077), !dbg !18
  %1325 = zext i32 %1324 to i64, !dbg !18
  %1326 = inttoptr i64 %1325 to ptr addrspace(3), !dbg !18
  %address.local1096 = getelementptr [2 x i8], ptr addrspace(1) %1323, i64 %address.local1082, !dbg !18
  %1327 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1328 = ptrtoint ptr addrspace(1) %address.local1096 to i64, !dbg !18
  %1329 = sub i64 %1328, %1327, !dbg !18
  %dma.buffer.offset1718 = trunc i64 %1329 to i32, !dbg !18
  %1330 = trunc i64 %1327 to i32, !dbg !18
  %1331 = insertelement <4 x i32> poison, i32 %1330, i64 0, !dbg !18
  %1332 = lshr i64 %1327, 32, !dbg !18
  %1333 = trunc i64 %1332 to i32, !dbg !18
  %1334 = insertelement <4 x i32> %1331, i32 %1333, i64 1, !dbg !18
  %1335 = insertelement <4 x i32> %1334, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1719 = insertelement <4 x i32> %1335, i32 131072, i64 3, !dbg !18
  %1336 = ptrtoint ptr addrspace(3) %1326 to i32, !dbg !18
  %dma.buffer.lds1720 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1336), !dbg !18
  %1337 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1720, i32 %dma.buffer.offset1718, <4 x i32> %dma.buffer.rsrc1719) #10, !dbg !18
  %1338 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1089), !dbg !18
  %1339 = zext i32 %1338 to i64, !dbg !18
  %1340 = inttoptr i64 %1339 to ptr addrspace(3), !dbg !18
  %address.local1097 = getelementptr [2 x i8], ptr addrspace(1) %1323, i64 %address.local1094, !dbg !18
  %1341 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1342 = ptrtoint ptr addrspace(1) %address.local1097 to i64, !dbg !18
  %1343 = sub i64 %1342, %1341, !dbg !18
  %dma.buffer.offset1715 = trunc i64 %1343 to i32, !dbg !18
  %1344 = trunc i64 %1341 to i32, !dbg !18
  %1345 = insertelement <4 x i32> poison, i32 %1344, i64 0, !dbg !18
  %1346 = lshr i64 %1341, 32, !dbg !18
  %1347 = trunc i64 %1346 to i32, !dbg !18
  %1348 = insertelement <4 x i32> %1345, i32 %1347, i64 1, !dbg !18
  %1349 = insertelement <4 x i32> %1348, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1716 = insertelement <4 x i32> %1349, i32 131072, i64 3, !dbg !18
  %1350 = ptrtoint ptr addrspace(3) %1340 to i32, !dbg !18
  %dma.buffer.lds1717 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1350), !dbg !18
  %1351 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1717, i32 %dma.buffer.offset1715, <4 x i32> %dma.buffer.rsrc1716) #10, !dbg !18
  %1352 = shufflevector <8 x half> %1307, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1353 = extractvalue [2 x <8 x float>] %209, 0, !dbg !35
  %1354 = shufflevector <8 x float> %1353, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1355 = fmul <4 x float> %1354, %970, !dbg !35
  %1356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1352, <4 x float> %1355) #9, !dbg !14
  %1357 = shufflevector <8 x half> %1309, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1357, <4 x float> %1356) #9, !dbg !14
  %1359 = shufflevector <8 x half> %1311, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1359, <4 x float> %1358) #9, !dbg !14
  %1361 = shufflevector <8 x half> %1313, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1361, <4 x float> %1360) #9, !dbg !14
  %1363 = shufflevector <8 x half> %1315, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1363, <4 x float> %1362) #9, !dbg !14
  %1365 = shufflevector <8 x half> %1317, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1365, <4 x float> %1364) #9, !dbg !14
  %1367 = shufflevector <8 x half> %1319, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1367, <4 x float> %1366) #9, !dbg !14
  %1369 = shufflevector <8 x half> %1321, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1369, <4 x float> %1368) #9, !dbg !14
  %1371 = shufflevector <8 x half> %1307, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1372 = shufflevector <8 x float> %1353, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1373 = fmul <4 x float> %1372, %970, !dbg !35
  %1374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1371, <4 x float> %1373) #9, !dbg !14
  %1375 = shufflevector <8 x half> %1309, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1375, <4 x float> %1374) #9, !dbg !14
  %1377 = shufflevector <8 x half> %1311, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1377, <4 x float> %1376) #9, !dbg !14
  %1379 = shufflevector <8 x half> %1313, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1379, <4 x float> %1378) #9, !dbg !14
  %1381 = shufflevector <8 x half> %1315, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1381, <4 x float> %1380) #9, !dbg !14
  %1383 = shufflevector <8 x half> %1317, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1383, <4 x float> %1382) #9, !dbg !14
  %1385 = shufflevector <8 x half> %1319, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1385, <4 x float> %1384) #9, !dbg !14
  %1387 = shufflevector <8 x half> %1321, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1387, <4 x float> %1386) #9, !dbg !14
  %1389 = extractvalue [2 x <8 x float>] %209, 1, !dbg !35
  %1390 = shufflevector <8 x float> %1389, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1391 = fmul <4 x float> %1390, %978, !dbg !35
  %1392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1352, <4 x float> %1391) #9, !dbg !14
  %1393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1357, <4 x float> %1392) #9, !dbg !14
  %1394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1359, <4 x float> %1393) #9, !dbg !14
  %1395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1361, <4 x float> %1394) #9, !dbg !14
  %1396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1363, <4 x float> %1395) #9, !dbg !14
  %1397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1365, <4 x float> %1396) #9, !dbg !14
  %1398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1367, <4 x float> %1397) #9, !dbg !14
  %1399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1369, <4 x float> %1398) #9, !dbg !14
  %1400 = shufflevector <8 x float> %1389, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1401 = fmul <4 x float> %1400, %978, !dbg !35
  %1402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1371, <4 x float> %1401) #9, !dbg !14
  %1403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1375, <4 x float> %1402) #9, !dbg !14
  %1404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1377, <4 x float> %1403) #9, !dbg !14
  %1405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1379, <4 x float> %1404) #9, !dbg !14
  %1406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1381, <4 x float> %1405) #9, !dbg !14
  %1407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1383, <4 x float> %1406) #9, !dbg !14
  %1408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1385, <4 x float> %1407) #9, !dbg !14
  %1409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1387, <4 x float> %1408) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1410 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #0, !dbg !14
  %1411 = bitcast <4 x i32> %1410 to <8 x half>, !dbg !14
  %1412 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1413 = bitcast <4 x i32> %1412 to <8 x half>, !dbg !14
  %1414 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1415 = bitcast <4 x i32> %1414 to <8 x half>, !dbg !14
  %1416 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1417 = bitcast <4 x i32> %1416 to <8 x half>, !dbg !14
  %1418 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #0, !dbg !14
  %1419 = bitcast <4 x i32> %1418 to <8 x half>, !dbg !14
  %1420 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1421 = bitcast <4 x i32> %1420 to <8 x half>, !dbg !14
  %1422 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #0, !dbg !14
  %1423 = bitcast <4 x i32> %1422 to <8 x half>, !dbg !14
  %1424 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #0, !dbg !14
  %1425 = bitcast <4 x i32> %1424 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1426 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1217, !dbg !18
  %1427 = getelementptr i8, ptr addrspace(1) %1426, i64 192, !dbg !18
  %1428 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1077), !dbg !18
  %1429 = zext i32 %1428 to i64, !dbg !18
  %1430 = inttoptr i64 %1429 to ptr addrspace(3), !dbg !18
  %address.local1098 = getelementptr [2 x i8], ptr addrspace(1) %1427, i64 %address.local1082, !dbg !18
  %1431 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1432 = ptrtoint ptr addrspace(1) %address.local1098 to i64, !dbg !18
  %1433 = sub i64 %1432, %1431, !dbg !18
  %dma.buffer.offset1712 = trunc i64 %1433 to i32, !dbg !18
  %1434 = trunc i64 %1431 to i32, !dbg !18
  %1435 = insertelement <4 x i32> poison, i32 %1434, i64 0, !dbg !18
  %1436 = lshr i64 %1431, 32, !dbg !18
  %1437 = trunc i64 %1436 to i32, !dbg !18
  %1438 = insertelement <4 x i32> %1435, i32 %1437, i64 1, !dbg !18
  %1439 = insertelement <4 x i32> %1438, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1713 = insertelement <4 x i32> %1439, i32 131072, i64 3, !dbg !18
  %1440 = ptrtoint ptr addrspace(3) %1430 to i32, !dbg !18
  %dma.buffer.lds1714 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1440), !dbg !18
  %1441 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1714, i32 %dma.buffer.offset1712, <4 x i32> %dma.buffer.rsrc1713) #10, !dbg !18
  %1442 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1089), !dbg !18
  %1443 = zext i32 %1442 to i64, !dbg !18
  %1444 = inttoptr i64 %1443 to ptr addrspace(3), !dbg !18
  %address.local1099 = getelementptr [2 x i8], ptr addrspace(1) %1427, i64 %address.local1094, !dbg !18
  %1445 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1446 = ptrtoint ptr addrspace(1) %address.local1099 to i64, !dbg !18
  %1447 = sub i64 %1446, %1445, !dbg !18
  %dma.buffer.offset = trunc i64 %1447 to i32, !dbg !18
  %1448 = trunc i64 %1445 to i32, !dbg !18
  %1449 = insertelement <4 x i32> poison, i32 %1448, i64 0, !dbg !18
  %1450 = lshr i64 %1445, 32, !dbg !18
  %1451 = trunc i64 %1450 to i32, !dbg !18
  %1452 = insertelement <4 x i32> %1449, i32 %1451, i64 1, !dbg !18
  %1453 = insertelement <4 x i32> %1452, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1453, i32 131072, i64 3, !dbg !18
  %1454 = ptrtoint ptr addrspace(3) %1444 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1454), !dbg !18
  %1455 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1456 = shufflevector <8 x half> %1411, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1457 = extractvalue [2 x <8 x float>] %210, 0, !dbg !35
  %1458 = shufflevector <8 x float> %1457, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1459 = fmul <4 x float> %1458, %970, !dbg !35
  %1460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1456, <4 x float> %1459) #9, !dbg !14
  %1461 = shufflevector <8 x half> %1413, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1461, <4 x float> %1460) #9, !dbg !14
  %1463 = shufflevector <8 x half> %1415, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1463, <4 x float> %1462) #9, !dbg !14
  %1465 = shufflevector <8 x half> %1417, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1465, <4 x float> %1464) #9, !dbg !14
  %1467 = shufflevector <8 x half> %1419, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1467, <4 x float> %1466) #9, !dbg !14
  %1469 = shufflevector <8 x half> %1421, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1469, <4 x float> %1468) #9, !dbg !14
  %1471 = shufflevector <8 x half> %1423, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1471, <4 x float> %1470) #9, !dbg !14
  %1473 = shufflevector <8 x half> %1425, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1473, <4 x float> %1472) #9, !dbg !14
  %1475 = shufflevector <8 x half> %1411, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1476 = shufflevector <8 x float> %1457, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1477 = fmul <4 x float> %1476, %970, !dbg !35
  %1478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1475, <4 x float> %1477) #9, !dbg !14
  %1479 = shufflevector <8 x half> %1413, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1479, <4 x float> %1478) #9, !dbg !14
  %1481 = shufflevector <8 x half> %1415, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1481, <4 x float> %1480) #9, !dbg !14
  %1483 = shufflevector <8 x half> %1417, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1483, <4 x float> %1482) #9, !dbg !14
  %1485 = shufflevector <8 x half> %1419, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1485, <4 x float> %1484) #9, !dbg !14
  %1487 = shufflevector <8 x half> %1421, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1487, <4 x float> %1486) #9, !dbg !14
  %1489 = shufflevector <8 x half> %1423, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1489, <4 x float> %1488) #9, !dbg !14
  %1491 = shufflevector <8 x half> %1425, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1491, <4 x float> %1490) #9, !dbg !14
  %1493 = extractvalue [2 x <8 x float>] %210, 1, !dbg !35
  %1494 = shufflevector <8 x float> %1493, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1495 = fmul <4 x float> %1494, %978, !dbg !35
  %1496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1456, <4 x float> %1495) #9, !dbg !14
  %1497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1461, <4 x float> %1496) #9, !dbg !14
  %1498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1463, <4 x float> %1497) #9, !dbg !14
  %1499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1465, <4 x float> %1498) #9, !dbg !14
  %1500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1467, <4 x float> %1499) #9, !dbg !14
  %1501 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1469, <4 x float> %1500) #9, !dbg !14
  %1502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1471, <4 x float> %1501) #9, !dbg !14
  %1503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1473, <4 x float> %1502) #9, !dbg !14
  %1504 = shufflevector <8 x float> %1493, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1505 = fmul <4 x float> %1504, %978, !dbg !35
  %1506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1475, <4 x float> %1505) #9, !dbg !14
  %1507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1479, <4 x float> %1506) #9, !dbg !14
  %1508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1481, <4 x float> %1507) #9, !dbg !14
  %1509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1483, <4 x float> %1508) #9, !dbg !14
  %1510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1485, <4 x float> %1509) #9, !dbg !14
  %1511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1487, <4 x float> %1510) #9, !dbg !14
  %1512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1489, <4 x float> %1511) #9, !dbg !14
  %1513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1491, <4 x float> %1512) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1514 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #0, !dbg !14
  %1515 = bitcast <4 x i32> %1514 to <8 x half>, !dbg !14
  %1516 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1517 = bitcast <4 x i32> %1516 to <8 x half>, !dbg !14
  %1518 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1519 = bitcast <4 x i32> %1518 to <8 x half>, !dbg !14
  %1520 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1521 = bitcast <4 x i32> %1520 to <8 x half>, !dbg !14
  %1522 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #0, !dbg !14
  %1523 = bitcast <4 x i32> %1522 to <8 x half>, !dbg !14
  %1524 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1525 = bitcast <4 x i32> %1524 to <8 x half>, !dbg !14
  %1526 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #0, !dbg !14
  %1527 = bitcast <4 x i32> %1526 to <8 x half>, !dbg !14
  %1528 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #0, !dbg !14
  %1529 = bitcast <4 x i32> %1528 to <8 x half>, !dbg !14
  %1530 = shufflevector <8 x half> %1515, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1531 = extractvalue [2 x <8 x float>] %211, 0, !dbg !35
  %1532 = shufflevector <8 x float> %1531, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1533 = fmul <4 x float> %1532, %970, !dbg !35
  %1534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1530, <4 x float> %1533) #9, !dbg !14
  %1535 = shufflevector <8 x half> %1517, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1535, <4 x float> %1534) #9, !dbg !14
  %1537 = shufflevector <8 x half> %1519, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1537, <4 x float> %1536) #9, !dbg !14
  %1539 = shufflevector <8 x half> %1521, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1539, <4 x float> %1538) #9, !dbg !14
  %1541 = shufflevector <8 x half> %1523, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1541, <4 x float> %1540) #9, !dbg !14
  %1543 = shufflevector <8 x half> %1525, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1543, <4 x float> %1542) #9, !dbg !14
  %1545 = shufflevector <8 x half> %1527, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1545, <4 x float> %1544) #9, !dbg !14
  %1547 = shufflevector <8 x half> %1529, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1547, <4 x float> %1546) #9, !dbg !14
  %1549 = shufflevector <8 x half> %1515, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1550 = shufflevector <8 x float> %1531, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1551 = fmul <4 x float> %1550, %970, !dbg !35
  %1552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1549, <4 x float> %1551) #9, !dbg !14
  %1553 = shufflevector <8 x half> %1517, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1553, <4 x float> %1552) #9, !dbg !14
  %1555 = shufflevector <8 x half> %1519, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1555, <4 x float> %1554) #9, !dbg !14
  %1557 = shufflevector <8 x half> %1521, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1557, <4 x float> %1556) #9, !dbg !14
  %1559 = shufflevector <8 x half> %1523, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1559, <4 x float> %1558) #9, !dbg !14
  %1561 = shufflevector <8 x half> %1525, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1561, <4 x float> %1560) #9, !dbg !14
  %1563 = shufflevector <8 x half> %1527, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1563, <4 x float> %1562) #9, !dbg !14
  %1565 = shufflevector <8 x half> %1529, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1565, <4 x float> %1564) #9, !dbg !14
  %1567 = extractvalue [2 x <8 x float>] %211, 1, !dbg !35
  %1568 = shufflevector <8 x float> %1567, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1569 = fmul <4 x float> %1568, %978, !dbg !35
  %1570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1530, <4 x float> %1569) #9, !dbg !14
  %1571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1535, <4 x float> %1570) #9, !dbg !14
  %1572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1537, <4 x float> %1571) #9, !dbg !14
  %1573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1539, <4 x float> %1572) #9, !dbg !14
  %1574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1541, <4 x float> %1573) #9, !dbg !14
  %1575 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1543, <4 x float> %1574) #9, !dbg !14
  %1576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1545, <4 x float> %1575) #9, !dbg !14
  %1577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1547, <4 x float> %1576) #9, !dbg !14
  %1578 = shufflevector <8 x float> %1567, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1579 = fmul <4 x float> %1578, %978, !dbg !35
  %1580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1549, <4 x float> %1579) #9, !dbg !14
  %1581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1553, <4 x float> %1580) #9, !dbg !14
  %1582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1555, <4 x float> %1581) #9, !dbg !14
  %1583 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1557, <4 x float> %1582) #9, !dbg !14
  %1584 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1559, <4 x float> %1583) #9, !dbg !14
  %1585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1561, <4 x float> %1584) #9, !dbg !14
  %1586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1563, <4 x float> %1585) #9, !dbg !14
  %1587 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1565, <4 x float> %1586) #9, !dbg !14
  %1588 = shufflevector <4 x float> %1266, <4 x float> %1284, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1589 = insertvalue [2 x <8 x float>] poison, <8 x float> %1588, 0, !dbg !7
  %1590 = shufflevector <4 x float> %1295, <4 x float> %1305, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1591 = insertvalue [2 x <8 x float>] %1589, <8 x float> %1590, 1, !dbg !7
  %1592 = insertvalue [2 x <1 x float>] poison, <1 x float> %934, 0, !dbg !7
  %1593 = insertvalue [2 x <1 x float>] %1592, <1 x float> %947, 1, !dbg !7
  %1594 = shufflevector <4 x float> %1370, <4 x float> %1388, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1595 = insertvalue [2 x <8 x float>] poison, <8 x float> %1594, 0, !dbg !7
  %1596 = shufflevector <4 x float> %1399, <4 x float> %1409, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1597 = insertvalue [2 x <8 x float>] %1595, <8 x float> %1596, 1, !dbg !7
  %1598 = shufflevector <4 x float> %1474, <4 x float> %1492, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1599 = insertvalue [2 x <8 x float>] poison, <8 x float> %1598, 0, !dbg !7
  %1600 = shufflevector <4 x float> %1503, <4 x float> %1513, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1601 = insertvalue [2 x <8 x float>] %1599, <8 x float> %1600, 1, !dbg !7
  %1602 = shufflevector <4 x float> %1548, <4 x float> %1566, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1603 = insertvalue [2 x <8 x float>] poison, <8 x float> %1602, 0, !dbg !7
  %1604 = shufflevector <4 x float> %1577, <4 x float> %1587, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1605 = insertvalue [2 x <8 x float>] %1603, <8 x float> %1604, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1606 = add nuw nsw i64 %205, 1, !dbg !7
  br label %204, !dbg !7

1607:                                             ; preds = %204
  %lane.sum.field284 = extractvalue [2 x <1 x float>] %207, 1, !dbg !7
  %lane.sum.local285 = extractelement <1 x float> %lane.sum.field284, i64 0, !dbg !7
  %lane.sum.lane.lo286 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane287 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo286), !dbg !7
  %1608 = shl nuw nsw i32 %lane.sum.lane287, 2, !dbg !7
  %1609 = xor i32 %1608, 64, !dbg !7
  %1610 = bitcast float %lane.sum.local285 to i32, !dbg !7
  %lane.sum.shuffle288 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1609, i32 %1610), !dbg !7
  %1611 = bitcast i32 %lane.sum.shuffle288 to float, !dbg !7
  %lane.sum.final289 = fadd float %lane.sum.local285, %1611, !dbg !7
  %1612 = shl nuw nsw i32 %lane.sum.lane287, 2, !dbg !7
  %1613 = xor i32 %1612, 128, !dbg !7
  %1614 = bitcast float %lane.sum.final289 to i32, !dbg !7
  %lane.sum.shuffle290 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1614), !dbg !7
  %1615 = bitcast i32 %lane.sum.shuffle290 to float, !dbg !7
  %lane.sum.final291 = fadd float %lane.sum.final289, %1615, !dbg !7
  %1616 = insertelement <1 x float> poison, float %lane.sum.final291, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %207, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %lane.sum.lane.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo), !dbg !7
  %1617 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1618 = xor i32 %1617, 64, !dbg !7
  %1619 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1618, i32 %1619), !dbg !7
  %1620 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1620, !dbg !7
  %1621 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1622 = xor i32 %1621, 128, !dbg !7
  %1623 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle282 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1622, i32 %1623), !dbg !7
  %1624 = bitcast i32 %lane.sum.shuffle282 to float, !dbg !7
  %lane.sum.final283 = fadd float %lane.sum.final, %1624, !dbg !7
  %1625 = insertelement <1 x float> poison, float %lane.sum.final283, i64 0, !dbg !7
  %1626 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %1626, 0, !dbg !7
  %1627 = and i32 %31, 32, !dbg !7
  %.not36 = icmp eq i32 %1627, 0, !dbg !7
  %1628 = extractvalue [2 x <8 x float>] %211, 0, !dbg !7
  %1629 = extractelement <8 x float> %1628, i64 0, !dbg !7
  %1630 = extractelement <8 x float> %1628, i64 1, !dbg !7
  %1631 = extractelement <8 x float> %1628, i64 2, !dbg !7
  %1632 = extractelement <8 x float> %1628, i64 3, !dbg !7
  %1633 = select i1 %.not, float %1630, float %1629, !dbg !7
  %1634 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1635 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1634), !dbg !7
  %1636 = add nuw nsw i32 %1635, 64, !dbg !7
  %1637 = xor i32 %1635, 16, !dbg !7
  %1638 = and i32 %1636, 192, !dbg !7
  %1639 = icmp samesign ult i32 %1637, %1638, !dbg !7
  %1640 = select i1 %1639, i32 %1637, i32 %1635, !dbg !7
  %1641 = shl nuw nsw i32 %1640, 2, !dbg !7
  %1642 = bitcast float %1633 to i32, !dbg !7
  %1643 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1642), !dbg !7
  %1644 = bitcast i32 %1643 to float, !dbg !7
  %1645 = select i1 %.not, float %1629, float %1644, !dbg !7
  %1646 = select i1 %.not, float %1644, float %1630, !dbg !7
  %1647 = select i1 %.not, float %1632, float %1631, !dbg !7
  %1648 = bitcast float %1647 to i32, !dbg !7
  %1649 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1648), !dbg !7
  %1650 = bitcast i32 %1649 to float, !dbg !7
  %1651 = select i1 %.not, float %1631, float %1650, !dbg !7
  %1652 = select i1 %.not, float %1650, float %1632, !dbg !7
  %1653 = select i1 %.not36, float %1651, float %1645, !dbg !7
  %1654 = xor i32 %1635, 32, !dbg !7
  %1655 = icmp samesign ult i32 %1654, %1638, !dbg !7
  %1656 = select i1 %1655, i32 %1654, i32 %1635, !dbg !7
  %1657 = shl nuw nsw i32 %1656, 2, !dbg !7
  %1658 = bitcast float %1653 to i32, !dbg !7
  %1659 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1658), !dbg !7
  %1660 = bitcast i32 %1659 to float, !dbg !7
  %1661 = select i1 %.not36, float %1645, float %1660, !dbg !7
  %1662 = select i1 %.not36, float %1660, float %1651, !dbg !7
  %1663 = select i1 %.not36, float %1652, float %1646, !dbg !7
  %1664 = bitcast float %1663 to i32, !dbg !7
  %1665 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1664), !dbg !7
  %1666 = bitcast i32 %1665 to float, !dbg !7
  %1667 = select i1 %.not36, float %1646, float %1666, !dbg !7
  %1668 = select i1 %.not36, float %1666, float %1652, !dbg !7
  %1669 = extractvalue [2 x <8 x float>] %210, 0, !dbg !7
  %1670 = extractelement <8 x float> %1669, i64 0, !dbg !7
  %1671 = extractelement <8 x float> %1669, i64 1, !dbg !7
  %1672 = extractelement <8 x float> %1669, i64 2, !dbg !7
  %1673 = extractelement <8 x float> %1669, i64 3, !dbg !7
  %1674 = select i1 %.not, float %1671, float %1670, !dbg !7
  %1675 = bitcast float %1674 to i32, !dbg !7
  %1676 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1675), !dbg !7
  %1677 = bitcast i32 %1676 to float, !dbg !7
  %1678 = select i1 %.not, float %1670, float %1677, !dbg !7
  %1679 = select i1 %.not, float %1677, float %1671, !dbg !7
  %1680 = select i1 %.not, float %1673, float %1672, !dbg !7
  %1681 = bitcast float %1680 to i32, !dbg !7
  %1682 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1681), !dbg !7
  %1683 = bitcast i32 %1682 to float, !dbg !7
  %1684 = select i1 %.not, float %1672, float %1683, !dbg !7
  %1685 = select i1 %.not, float %1683, float %1673, !dbg !7
  %1686 = select i1 %.not36, float %1684, float %1678, !dbg !7
  %1687 = bitcast float %1686 to i32, !dbg !7
  %1688 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1687), !dbg !7
  %1689 = bitcast i32 %1688 to float, !dbg !7
  %1690 = select i1 %.not36, float %1678, float %1689, !dbg !7
  %1691 = select i1 %.not36, float %1689, float %1684, !dbg !7
  %1692 = select i1 %.not36, float %1685, float %1679, !dbg !7
  %1693 = bitcast float %1692 to i32, !dbg !7
  %1694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1693), !dbg !7
  %1695 = bitcast i32 %1694 to float, !dbg !7
  %1696 = select i1 %.not36, float %1679, float %1695, !dbg !7
  %1697 = select i1 %.not36, float %1695, float %1685, !dbg !7
  %1698 = extractvalue [2 x <8 x float>] %209, 0, !dbg !7
  %1699 = extractelement <8 x float> %1698, i64 0, !dbg !7
  %1700 = extractelement <8 x float> %1698, i64 1, !dbg !7
  %1701 = extractelement <8 x float> %1698, i64 2, !dbg !7
  %1702 = extractelement <8 x float> %1698, i64 3, !dbg !7
  %1703 = select i1 %.not, float %1700, float %1699, !dbg !7
  %1704 = bitcast float %1703 to i32, !dbg !7
  %1705 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1704), !dbg !7
  %1706 = bitcast i32 %1705 to float, !dbg !7
  %1707 = select i1 %.not, float %1699, float %1706, !dbg !7
  %1708 = select i1 %.not, float %1706, float %1700, !dbg !7
  %1709 = select i1 %.not, float %1702, float %1701, !dbg !7
  %1710 = bitcast float %1709 to i32, !dbg !7
  %1711 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1710), !dbg !7
  %1712 = bitcast i32 %1711 to float, !dbg !7
  %1713 = select i1 %.not, float %1701, float %1712, !dbg !7
  %1714 = select i1 %.not, float %1712, float %1702, !dbg !7
  %1715 = select i1 %.not36, float %1713, float %1707, !dbg !7
  %1716 = bitcast float %1715 to i32, !dbg !7
  %1717 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1716), !dbg !7
  %1718 = bitcast i32 %1717 to float, !dbg !7
  %1719 = select i1 %.not36, float %1707, float %1718, !dbg !7
  %1720 = select i1 %.not36, float %1718, float %1713, !dbg !7
  %1721 = select i1 %.not36, float %1714, float %1708, !dbg !7
  %1722 = bitcast float %1721 to i32, !dbg !7
  %1723 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1722), !dbg !7
  %1724 = bitcast i32 %1723 to float, !dbg !7
  %1725 = select i1 %.not36, float %1708, float %1724, !dbg !7
  %1726 = select i1 %.not36, float %1724, float %1714, !dbg !7
  %1727 = extractvalue [2 x <8 x float>] %206, 0, !dbg !7
  %1728 = extractelement <8 x float> %1727, i64 0, !dbg !7
  %1729 = extractelement <8 x float> %1727, i64 1, !dbg !7
  %1730 = extractelement <8 x float> %1727, i64 2, !dbg !7
  %1731 = extractelement <8 x float> %1727, i64 3, !dbg !7
  %1732 = select i1 %.not, float %1729, float %1728, !dbg !7
  %1733 = bitcast float %1732 to i32, !dbg !7
  %1734 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1733), !dbg !7
  %1735 = bitcast i32 %1734 to float, !dbg !7
  %1736 = select i1 %.not, float %1728, float %1735, !dbg !7
  %1737 = select i1 %.not, float %1735, float %1729, !dbg !7
  %1738 = select i1 %.not, float %1731, float %1730, !dbg !7
  %1739 = bitcast float %1738 to i32, !dbg !7
  %1740 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1739), !dbg !7
  %1741 = bitcast i32 %1740 to float, !dbg !7
  %1742 = select i1 %.not, float %1730, float %1741, !dbg !7
  %1743 = select i1 %.not, float %1741, float %1731, !dbg !7
  %1744 = select i1 %.not36, float %1742, float %1736, !dbg !7
  %1745 = bitcast float %1744 to i32, !dbg !7
  %1746 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1745), !dbg !7
  %1747 = select i1 %.not36, float %1743, float %1737, !dbg !7
  %1748 = bitcast float %1747 to i32, !dbg !7
  %1749 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1748), !dbg !7
  %1750 = shufflevector <1 x float> %1625, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1751 = extractelement <8 x float> %1628, i64 4, !dbg !7
  %1752 = extractelement <8 x float> %1628, i64 5, !dbg !7
  %1753 = extractelement <8 x float> %1628, i64 6, !dbg !7
  %1754 = extractelement <8 x float> %1628, i64 7, !dbg !7
  %1755 = select i1 %.not, float %1752, float %1751, !dbg !7
  %1756 = bitcast float %1755 to i32, !dbg !7
  %1757 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1756), !dbg !7
  %1758 = bitcast i32 %1757 to float, !dbg !7
  %1759 = select i1 %.not, float %1751, float %1758, !dbg !7
  %1760 = select i1 %.not, float %1758, float %1752, !dbg !7
  %1761 = select i1 %.not, float %1754, float %1753, !dbg !7
  %1762 = bitcast float %1761 to i32, !dbg !7
  %1763 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1762), !dbg !7
  %1764 = bitcast i32 %1763 to float, !dbg !7
  %1765 = select i1 %.not, float %1753, float %1764, !dbg !7
  %1766 = select i1 %.not, float %1764, float %1754, !dbg !7
  %1767 = select i1 %.not36, float %1765, float %1759, !dbg !7
  %1768 = bitcast float %1767 to i32, !dbg !7
  %1769 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1768), !dbg !7
  %1770 = bitcast i32 %1769 to float, !dbg !7
  %1771 = select i1 %.not36, float %1759, float %1770, !dbg !7
  %1772 = select i1 %.not36, float %1770, float %1765, !dbg !7
  %1773 = select i1 %.not36, float %1766, float %1760, !dbg !7
  %1774 = bitcast float %1773 to i32, !dbg !7
  %1775 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1774), !dbg !7
  %1776 = bitcast i32 %1775 to float, !dbg !7
  %1777 = select i1 %.not36, float %1760, float %1776, !dbg !7
  %1778 = select i1 %.not36, float %1776, float %1766, !dbg !7
  %1779 = extractelement <8 x float> %1669, i64 4, !dbg !7
  %1780 = extractelement <8 x float> %1669, i64 5, !dbg !7
  %1781 = extractelement <8 x float> %1669, i64 6, !dbg !7
  %1782 = extractelement <8 x float> %1669, i64 7, !dbg !7
  %1783 = select i1 %.not, float %1780, float %1779, !dbg !7
  %1784 = bitcast float %1783 to i32, !dbg !7
  %1785 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1784), !dbg !7
  %1786 = bitcast i32 %1785 to float, !dbg !7
  %1787 = select i1 %.not, float %1779, float %1786, !dbg !7
  %1788 = select i1 %.not, float %1786, float %1780, !dbg !7
  %1789 = select i1 %.not, float %1782, float %1781, !dbg !7
  %1790 = bitcast float %1789 to i32, !dbg !7
  %1791 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1790), !dbg !7
  %1792 = bitcast i32 %1791 to float, !dbg !7
  %1793 = select i1 %.not, float %1781, float %1792, !dbg !7
  %1794 = select i1 %.not, float %1792, float %1782, !dbg !7
  %1795 = select i1 %.not36, float %1793, float %1787, !dbg !7
  %1796 = bitcast float %1795 to i32, !dbg !7
  %1797 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1796), !dbg !7
  %1798 = bitcast i32 %1797 to float, !dbg !7
  %1799 = select i1 %.not36, float %1787, float %1798, !dbg !7
  %1800 = select i1 %.not36, float %1798, float %1793, !dbg !7
  %1801 = select i1 %.not36, float %1794, float %1788, !dbg !7
  %1802 = bitcast float %1801 to i32, !dbg !7
  %1803 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1802), !dbg !7
  %1804 = bitcast i32 %1803 to float, !dbg !7
  %1805 = select i1 %.not36, float %1788, float %1804, !dbg !7
  %1806 = select i1 %.not36, float %1804, float %1794, !dbg !7
  %1807 = extractelement <8 x float> %1698, i64 4, !dbg !7
  %1808 = extractelement <8 x float> %1698, i64 5, !dbg !7
  %1809 = extractelement <8 x float> %1698, i64 6, !dbg !7
  %1810 = extractelement <8 x float> %1698, i64 7, !dbg !7
  %1811 = select i1 %.not, float %1808, float %1807, !dbg !7
  %1812 = bitcast float %1811 to i32, !dbg !7
  %1813 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1812), !dbg !7
  %1814 = bitcast i32 %1813 to float, !dbg !7
  %1815 = select i1 %.not, float %1807, float %1814, !dbg !7
  %1816 = select i1 %.not, float %1814, float %1808, !dbg !7
  %1817 = select i1 %.not, float %1810, float %1809, !dbg !7
  %1818 = bitcast float %1817 to i32, !dbg !7
  %1819 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1818), !dbg !7
  %1820 = bitcast i32 %1819 to float, !dbg !7
  %1821 = select i1 %.not, float %1809, float %1820, !dbg !7
  %1822 = select i1 %.not, float %1820, float %1810, !dbg !7
  %1823 = select i1 %.not36, float %1821, float %1815, !dbg !7
  %1824 = bitcast float %1823 to i32, !dbg !7
  %1825 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1824), !dbg !7
  %1826 = bitcast i32 %1825 to float, !dbg !7
  %1827 = select i1 %.not36, float %1815, float %1826, !dbg !7
  %1828 = select i1 %.not36, float %1826, float %1821, !dbg !7
  %1829 = select i1 %.not36, float %1822, float %1816, !dbg !7
  %1830 = bitcast float %1829 to i32, !dbg !7
  %1831 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1830), !dbg !7
  %1832 = bitcast i32 %1831 to float, !dbg !7
  %1833 = select i1 %.not36, float %1816, float %1832, !dbg !7
  %1834 = select i1 %.not36, float %1832, float %1822, !dbg !7
  %1835 = extractelement <8 x float> %1727, i64 4, !dbg !7
  %1836 = extractelement <8 x float> %1727, i64 5, !dbg !7
  %1837 = extractelement <8 x float> %1727, i64 6, !dbg !7
  %1838 = extractelement <8 x float> %1727, i64 7, !dbg !7
  %1839 = select i1 %.not, float %1836, float %1835, !dbg !7
  %1840 = bitcast float %1839 to i32, !dbg !7
  %1841 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1840), !dbg !7
  %1842 = bitcast i32 %1841 to float, !dbg !7
  %1843 = select i1 %.not, float %1835, float %1842, !dbg !7
  %1844 = select i1 %.not, float %1842, float %1836, !dbg !7
  %1845 = select i1 %.not, float %1838, float %1837, !dbg !7
  %1846 = bitcast float %1845 to i32, !dbg !7
  %1847 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1846), !dbg !7
  %1848 = bitcast i32 %1847 to float, !dbg !7
  %1849 = select i1 %.not, float %1837, float %1848, !dbg !7
  %1850 = select i1 %.not, float %1848, float %1838, !dbg !7
  %1851 = select i1 %.not36, float %1849, float %1843, !dbg !7
  %1852 = bitcast float %1851 to i32, !dbg !7
  %1853 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1852), !dbg !7
  %1854 = select i1 %.not36, float %1850, float %1844, !dbg !7
  %1855 = bitcast float %1854 to i32, !dbg !7
  %1856 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1855), !dbg !7
  %1857 = extractvalue [2 x <8 x float>] %211, 1, !dbg !7
  %1858 = extractelement <8 x float> %1857, i64 0, !dbg !7
  %1859 = extractelement <8 x float> %1857, i64 1, !dbg !7
  %1860 = extractelement <8 x float> %1857, i64 2, !dbg !7
  %1861 = extractelement <8 x float> %1857, i64 3, !dbg !7
  %1862 = select i1 %.not, float %1859, float %1858, !dbg !7
  %1863 = bitcast float %1862 to i32, !dbg !7
  %1864 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1863), !dbg !7
  %1865 = bitcast i32 %1864 to float, !dbg !7
  %1866 = select i1 %.not, float %1858, float %1865, !dbg !7
  %1867 = select i1 %.not, float %1865, float %1859, !dbg !7
  %1868 = select i1 %.not, float %1861, float %1860, !dbg !7
  %1869 = bitcast float %1868 to i32, !dbg !7
  %1870 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1869), !dbg !7
  %1871 = bitcast i32 %1870 to float, !dbg !7
  %1872 = select i1 %.not, float %1860, float %1871, !dbg !7
  %1873 = select i1 %.not, float %1871, float %1861, !dbg !7
  %1874 = select i1 %.not36, float %1872, float %1866, !dbg !7
  %1875 = bitcast float %1874 to i32, !dbg !7
  %1876 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1875), !dbg !7
  %1877 = bitcast i32 %1876 to float, !dbg !7
  %1878 = select i1 %.not36, float %1866, float %1877, !dbg !7
  %1879 = select i1 %.not36, float %1877, float %1872, !dbg !7
  %1880 = select i1 %.not36, float %1873, float %1867, !dbg !7
  %1881 = bitcast float %1880 to i32, !dbg !7
  %1882 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1881), !dbg !7
  %1883 = bitcast i32 %1882 to float, !dbg !7
  %1884 = select i1 %.not36, float %1867, float %1883, !dbg !7
  %1885 = select i1 %.not36, float %1883, float %1873, !dbg !7
  %1886 = extractvalue [2 x <8 x float>] %210, 1, !dbg !7
  %1887 = extractelement <8 x float> %1886, i64 0, !dbg !7
  %1888 = extractelement <8 x float> %1886, i64 1, !dbg !7
  %1889 = extractelement <8 x float> %1886, i64 2, !dbg !7
  %1890 = extractelement <8 x float> %1886, i64 3, !dbg !7
  %1891 = select i1 %.not, float %1888, float %1887, !dbg !7
  %1892 = bitcast float %1891 to i32, !dbg !7
  %1893 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1892), !dbg !7
  %1894 = bitcast i32 %1893 to float, !dbg !7
  %1895 = select i1 %.not, float %1887, float %1894, !dbg !7
  %1896 = select i1 %.not, float %1894, float %1888, !dbg !7
  %1897 = select i1 %.not, float %1890, float %1889, !dbg !7
  %1898 = bitcast float %1897 to i32, !dbg !7
  %1899 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1898), !dbg !7
  %1900 = bitcast i32 %1899 to float, !dbg !7
  %1901 = select i1 %.not, float %1889, float %1900, !dbg !7
  %1902 = select i1 %.not, float %1900, float %1890, !dbg !7
  %1903 = select i1 %.not36, float %1901, float %1895, !dbg !7
  %1904 = bitcast float %1903 to i32, !dbg !7
  %1905 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1904), !dbg !7
  %1906 = bitcast i32 %1905 to float, !dbg !7
  %1907 = select i1 %.not36, float %1895, float %1906, !dbg !7
  %1908 = select i1 %.not36, float %1906, float %1901, !dbg !7
  %1909 = select i1 %.not36, float %1902, float %1896, !dbg !7
  %1910 = bitcast float %1909 to i32, !dbg !7
  %1911 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1910), !dbg !7
  %1912 = bitcast i32 %1911 to float, !dbg !7
  %1913 = select i1 %.not36, float %1896, float %1912, !dbg !7
  %1914 = select i1 %.not36, float %1912, float %1902, !dbg !7
  %1915 = extractvalue [2 x <8 x float>] %209, 1, !dbg !7
  %1916 = extractelement <8 x float> %1915, i64 0, !dbg !7
  %1917 = extractelement <8 x float> %1915, i64 1, !dbg !7
  %1918 = extractelement <8 x float> %1915, i64 2, !dbg !7
  %1919 = extractelement <8 x float> %1915, i64 3, !dbg !7
  %1920 = select i1 %.not, float %1917, float %1916, !dbg !7
  %1921 = bitcast float %1920 to i32, !dbg !7
  %1922 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1921), !dbg !7
  %1923 = bitcast i32 %1922 to float, !dbg !7
  %1924 = select i1 %.not, float %1916, float %1923, !dbg !7
  %1925 = select i1 %.not, float %1923, float %1917, !dbg !7
  %1926 = select i1 %.not, float %1919, float %1918, !dbg !7
  %1927 = bitcast float %1926 to i32, !dbg !7
  %1928 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1927), !dbg !7
  %1929 = bitcast i32 %1928 to float, !dbg !7
  %1930 = select i1 %.not, float %1918, float %1929, !dbg !7
  %1931 = select i1 %.not, float %1929, float %1919, !dbg !7
  %1932 = select i1 %.not36, float %1930, float %1924, !dbg !7
  %1933 = bitcast float %1932 to i32, !dbg !7
  %1934 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1933), !dbg !7
  %1935 = bitcast i32 %1934 to float, !dbg !7
  %1936 = select i1 %.not36, float %1924, float %1935, !dbg !7
  %1937 = select i1 %.not36, float %1935, float %1930, !dbg !7
  %1938 = select i1 %.not36, float %1931, float %1925, !dbg !7
  %1939 = bitcast float %1938 to i32, !dbg !7
  %1940 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1939), !dbg !7
  %1941 = bitcast i32 %1940 to float, !dbg !7
  %1942 = select i1 %.not36, float %1925, float %1941, !dbg !7
  %1943 = select i1 %.not36, float %1941, float %1931, !dbg !7
  %1944 = extractvalue [2 x <8 x float>] %206, 1, !dbg !7
  %1945 = extractelement <8 x float> %1944, i64 0, !dbg !7
  %1946 = extractelement <8 x float> %1944, i64 1, !dbg !7
  %1947 = extractelement <8 x float> %1944, i64 2, !dbg !7
  %1948 = extractelement <8 x float> %1944, i64 3, !dbg !7
  %1949 = select i1 %.not, float %1946, float %1945, !dbg !7
  %1950 = bitcast float %1949 to i32, !dbg !7
  %1951 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1950), !dbg !7
  %1952 = bitcast i32 %1951 to float, !dbg !7
  %1953 = select i1 %.not, float %1945, float %1952, !dbg !7
  %1954 = select i1 %.not, float %1952, float %1946, !dbg !7
  %1955 = select i1 %.not, float %1948, float %1947, !dbg !7
  %1956 = bitcast float %1955 to i32, !dbg !7
  %1957 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1956), !dbg !7
  %1958 = bitcast i32 %1957 to float, !dbg !7
  %1959 = select i1 %.not, float %1947, float %1958, !dbg !7
  %1960 = select i1 %.not, float %1958, float %1948, !dbg !7
  %1961 = select i1 %.not36, float %1959, float %1953, !dbg !7
  %1962 = bitcast float %1961 to i32, !dbg !7
  %1963 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1962), !dbg !7
  %1964 = select i1 %.not36, float %1960, float %1954, !dbg !7
  %1965 = bitcast float %1964 to i32, !dbg !7
  %1966 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1965), !dbg !7
  %1967 = shufflevector <1 x float> %1616, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1968 = extractelement <8 x float> %1857, i64 4, !dbg !7
  %1969 = extractelement <8 x float> %1857, i64 5, !dbg !7
  %1970 = extractelement <8 x float> %1857, i64 6, !dbg !7
  %1971 = extractelement <8 x float> %1857, i64 7, !dbg !7
  %1972 = select i1 %.not, float %1969, float %1968, !dbg !7
  %1973 = bitcast float %1972 to i32, !dbg !7
  %1974 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1973), !dbg !7
  %1975 = bitcast i32 %1974 to float, !dbg !7
  %1976 = select i1 %.not, float %1968, float %1975, !dbg !7
  %1977 = select i1 %.not, float %1975, float %1969, !dbg !7
  %1978 = select i1 %.not, float %1971, float %1970, !dbg !7
  %1979 = bitcast float %1978 to i32, !dbg !7
  %1980 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %1979), !dbg !7
  %1981 = bitcast i32 %1980 to float, !dbg !7
  %1982 = select i1 %.not, float %1970, float %1981, !dbg !7
  %1983 = select i1 %.not, float %1981, float %1971, !dbg !7
  %1984 = select i1 %.not36, float %1982, float %1976, !dbg !7
  %1985 = bitcast float %1984 to i32, !dbg !7
  %1986 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1985), !dbg !7
  %1987 = bitcast i32 %1986 to float, !dbg !7
  %1988 = select i1 %.not36, float %1976, float %1987, !dbg !7
  %1989 = select i1 %.not36, float %1987, float %1982, !dbg !7
  %1990 = select i1 %.not36, float %1983, float %1977, !dbg !7
  %1991 = bitcast float %1990 to i32, !dbg !7
  %1992 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %1991), !dbg !7
  %1993 = bitcast i32 %1992 to float, !dbg !7
  %1994 = select i1 %.not36, float %1977, float %1993, !dbg !7
  %1995 = select i1 %.not36, float %1993, float %1983, !dbg !7
  %1996 = extractelement <8 x float> %1886, i64 4, !dbg !7
  %1997 = extractelement <8 x float> %1886, i64 5, !dbg !7
  %1998 = extractelement <8 x float> %1886, i64 6, !dbg !7
  %1999 = extractelement <8 x float> %1886, i64 7, !dbg !7
  %2000 = select i1 %.not, float %1997, float %1996, !dbg !7
  %2001 = bitcast float %2000 to i32, !dbg !7
  %2002 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %2001), !dbg !7
  %2003 = bitcast i32 %2002 to float, !dbg !7
  %2004 = select i1 %.not, float %1996, float %2003, !dbg !7
  %2005 = select i1 %.not, float %2003, float %1997, !dbg !7
  %2006 = select i1 %.not, float %1999, float %1998, !dbg !7
  %2007 = bitcast float %2006 to i32, !dbg !7
  %2008 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %2007), !dbg !7
  %2009 = bitcast i32 %2008 to float, !dbg !7
  %2010 = select i1 %.not, float %1998, float %2009, !dbg !7
  %2011 = select i1 %.not, float %2009, float %1999, !dbg !7
  %2012 = select i1 %.not36, float %2010, float %2004, !dbg !7
  %2013 = bitcast float %2012 to i32, !dbg !7
  %2014 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %2013), !dbg !7
  %2015 = bitcast i32 %2014 to float, !dbg !7
  %2016 = select i1 %.not36, float %2004, float %2015, !dbg !7
  %2017 = select i1 %.not36, float %2015, float %2010, !dbg !7
  %2018 = select i1 %.not36, float %2011, float %2005, !dbg !7
  %2019 = bitcast float %2018 to i32, !dbg !7
  %2020 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %2019), !dbg !7
  %2021 = bitcast i32 %2020 to float, !dbg !7
  %2022 = select i1 %.not36, float %2005, float %2021, !dbg !7
  %2023 = select i1 %.not36, float %2021, float %2011, !dbg !7
  %2024 = extractelement <8 x float> %1915, i64 4, !dbg !7
  %2025 = extractelement <8 x float> %1915, i64 5, !dbg !7
  %2026 = extractelement <8 x float> %1915, i64 6, !dbg !7
  %2027 = extractelement <8 x float> %1915, i64 7, !dbg !7
  %2028 = select i1 %.not, float %2025, float %2024, !dbg !7
  %2029 = bitcast float %2028 to i32, !dbg !7
  %2030 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %2029), !dbg !7
  %2031 = bitcast i32 %2030 to float, !dbg !7
  %2032 = select i1 %.not, float %2024, float %2031, !dbg !7
  %2033 = select i1 %.not, float %2031, float %2025, !dbg !7
  %2034 = select i1 %.not, float %2027, float %2026, !dbg !7
  %2035 = bitcast float %2034 to i32, !dbg !7
  %2036 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %2035), !dbg !7
  %2037 = bitcast i32 %2036 to float, !dbg !7
  %2038 = select i1 %.not, float %2026, float %2037, !dbg !7
  %2039 = select i1 %.not, float %2037, float %2027, !dbg !7
  %2040 = select i1 %.not36, float %2038, float %2032, !dbg !7
  %2041 = bitcast float %2040 to i32, !dbg !7
  %2042 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %2041), !dbg !7
  %2043 = bitcast i32 %2042 to float, !dbg !7
  %2044 = select i1 %.not36, float %2032, float %2043, !dbg !7
  %2045 = select i1 %.not36, float %2043, float %2038, !dbg !7
  %2046 = select i1 %.not36, float %2039, float %2033, !dbg !7
  %2047 = bitcast float %2046 to i32, !dbg !7
  %2048 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %2047), !dbg !7
  %2049 = bitcast i32 %2048 to float, !dbg !7
  %2050 = select i1 %.not36, float %2033, float %2049, !dbg !7
  %2051 = select i1 %.not36, float %2049, float %2039, !dbg !7
  %2052 = extractelement <8 x float> %1944, i64 4, !dbg !7
  %2053 = extractelement <8 x float> %1944, i64 5, !dbg !7
  %2054 = extractelement <8 x float> %1944, i64 6, !dbg !7
  %2055 = extractelement <8 x float> %1944, i64 7, !dbg !7
  %2056 = select i1 %.not, float %2053, float %2052, !dbg !7
  %2057 = bitcast float %2056 to i32, !dbg !7
  %2058 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %2057), !dbg !7
  %2059 = bitcast i32 %2058 to float, !dbg !7
  %2060 = select i1 %.not, float %2052, float %2059, !dbg !7
  %2061 = select i1 %.not, float %2059, float %2053, !dbg !7
  %2062 = select i1 %.not, float %2055, float %2054, !dbg !7
  %2063 = bitcast float %2062 to i32, !dbg !7
  %2064 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1641, i32 %2063), !dbg !7
  %2065 = bitcast i32 %2064 to float, !dbg !7
  %2066 = select i1 %.not, float %2054, float %2065, !dbg !7
  %2067 = select i1 %.not, float %2065, float %2055, !dbg !7
  %2068 = select i1 %.not36, float %2066, float %2060, !dbg !7
  %2069 = bitcast float %2068 to i32, !dbg !7
  %2070 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %2069), !dbg !7
  %2071 = select i1 %.not36, float %2067, float %2061, !dbg !7
  %2072 = bitcast float %2071 to i32, !dbg !7
  %2073 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1657, i32 %2072), !dbg !7
  br i1 true, label %2074, label %2138, !dbg !21

2074:                                             ; preds = %1607
  %2075 = bitcast i32 %2070 to float, !dbg !7
  %2076 = select i1 %.not36, float %2060, float %2075, !dbg !7
  %2077 = insertelement <4 x float> poison, float %2076, i64 0, !dbg !19
  %2078 = bitcast i32 %2073 to float, !dbg !7
  %2079 = select i1 %.not36, float %2061, float %2078, !dbg !7
  %2080 = insertelement <4 x float> %2077, float %2079, i64 1, !dbg !19
  %2081 = select i1 %.not36, float %2075, float %2066, !dbg !7
  %2082 = insertelement <4 x float> %2080, float %2081, i64 2, !dbg !19
  %2083 = select i1 %.not36, float %2078, float %2067, !dbg !7
  %2084 = insertelement <4 x float> %2082, float %2083, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1967, !dbg !19
  %normalized = fmul <4 x float> %2084, %row.reciprocal, !dbg !19
  %2085 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %2086 = extractelement <4 x half> %2085, i64 3, !dbg !20
  %2087 = extractelement <4 x half> %2085, i64 2, !dbg !20
  %2088 = extractelement <4 x half> %2085, i64 1, !dbg !20
  %2089 = extractelement <4 x half> %2085, i64 0, !dbg !20
  %2090 = bitcast i32 %1963 to float, !dbg !7
  %2091 = select i1 %.not36, float %1953, float %2090, !dbg !7
  %2092 = insertelement <4 x float> poison, float %2091, i64 0, !dbg !19
  %2093 = bitcast i32 %1966 to float, !dbg !7
  %2094 = select i1 %.not36, float %1954, float %2093, !dbg !7
  %2095 = insertelement <4 x float> %2092, float %2094, i64 1, !dbg !19
  %2096 = select i1 %.not36, float %2090, float %1959, !dbg !7
  %2097 = insertelement <4 x float> %2095, float %2096, i64 2, !dbg !19
  %2098 = select i1 %.not36, float %2093, float %1960, !dbg !7
  %2099 = insertelement <4 x float> %2097, float %2098, i64 3, !dbg !19
  %normalized104 = fmul <4 x float> %2099, %row.reciprocal, !dbg !19
  %2100 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !20
  %2101 = extractelement <4 x half> %2100, i64 3, !dbg !20
  %2102 = extractelement <4 x half> %2100, i64 2, !dbg !20
  %2103 = extractelement <4 x half> %2100, i64 1, !dbg !20
  %2104 = extractelement <4 x half> %2100, i64 0, !dbg !20
  %2105 = bitcast i32 %1853 to float, !dbg !7
  %2106 = select i1 %.not36, float %1843, float %2105, !dbg !7
  %2107 = insertelement <4 x float> poison, float %2106, i64 0, !dbg !19
  %2108 = bitcast i32 %1856 to float, !dbg !7
  %2109 = select i1 %.not36, float %1844, float %2108, !dbg !7
  %2110 = insertelement <4 x float> %2107, float %2109, i64 1, !dbg !19
  %2111 = select i1 %.not36, float %2105, float %1849, !dbg !7
  %2112 = insertelement <4 x float> %2110, float %2111, i64 2, !dbg !19
  %2113 = select i1 %.not36, float %2108, float %1850, !dbg !7
  %2114 = insertelement <4 x float> %2112, float %2113, i64 3, !dbg !19
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1750, !dbg !19
  %normalized106 = fmul <4 x float> %2114, %row.reciprocal105, !dbg !19
  %2115 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !20
  %2116 = extractelement <4 x half> %2115, i64 3, !dbg !20
  %2117 = extractelement <4 x half> %2115, i64 2, !dbg !20
  %2118 = extractelement <4 x half> %2115, i64 1, !dbg !20
  %2119 = extractelement <4 x half> %2115, i64 0, !dbg !20
  %2120 = bitcast i32 %1746 to float, !dbg !7
  %2121 = select i1 %.not36, float %1736, float %2120, !dbg !7
  %2122 = insertelement <4 x float> poison, float %2121, i64 0, !dbg !19
  %2123 = bitcast i32 %1749 to float, !dbg !7
  %2124 = select i1 %.not36, float %1737, float %2123, !dbg !7
  %2125 = insertelement <4 x float> %2122, float %2124, i64 1, !dbg !19
  %2126 = select i1 %.not36, float %2120, float %1742, !dbg !7
  %2127 = insertelement <4 x float> %2125, float %2126, i64 2, !dbg !19
  %2128 = select i1 %.not36, float %2123, float %1743, !dbg !7
  %2129 = insertelement <4 x float> %2127, float %2128, i64 3, !dbg !19
  %normalized107 = fmul <4 x float> %2129, %row.reciprocal105, !dbg !19
  %2130 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !20
  %2131 = extractelement <4 x half> %2130, i64 3, !dbg !20
  %2132 = extractelement <4 x half> %2130, i64 2, !dbg !20
  %2133 = extractelement <4 x half> %2130, i64 1, !dbg !20
  %2134 = extractelement <4 x half> %2130, i64 0, !dbg !20
  %2135 = add nsw i64 %16, %22, !dbg !21
  %2136 = shl i64 %2135, 19, !dbg !21
  %2137 = and i64 %2136, 16252928, !dbg !21
  %address.thread1100 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1101 = zext nneg i32 %address.thread1100 to i64, !dbg !8
  %address.local1102 = add i64 %25, %address.local1101, !dbg !21
  %address.local1103 = lshr i64 %address.local1101, 6, !dbg !11
  %address.local1104 = shl nuw nsw i64 %address.local1103, 4, !dbg !21
  %address.local1105 = add i64 %address.local1102, %address.local1104, !dbg !21
  %address.local1106 = and i64 %address.local1101, 240, !dbg !21
  %address.local1107 = sub i64 %address.local1105, %address.local1106, !dbg !21
  %address.local1108 = add i64 %address.local1107, %26, !dbg !21
  %address.local1109 = shl nuw nsw i64 %address.local1108, 7, !dbg !21
  %address.local1110 = add nuw nsw i64 %2137, %address.local1109, !dbg !21
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1112 = lshr i64 %address.local1101, 2, !dbg !11
  %address.local1113 = and i64 %address.local1112, 12, !dbg !11
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1113, !dbg !21
  store half %2134, ptr addrspace(1) %address.local1114, align 2, !dbg !21
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1116 = or disjoint i64 %address.local1113, 1, !dbg !21
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1115, i64 %address.local1116, !dbg !21
  store half %2133, ptr addrspace(1) %address.local1117, align 2, !dbg !21
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1119 = or disjoint i64 %address.local1113, 2, !dbg !21
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1118, i64 %address.local1119, !dbg !21
  store half %2132, ptr addrspace(1) %address.local1120, align 2, !dbg !21
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1122 = or disjoint i64 %address.local1113, 3, !dbg !21
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1121, i64 %address.local1122, !dbg !21
  store half %2131, ptr addrspace(1) %address.local1123, align 2, !dbg !21
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1125 = or disjoint i64 %address.local1113, 16, !dbg !21
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1124, i64 %address.local1125, !dbg !21
  store half %2119, ptr addrspace(1) %address.local1126, align 2, !dbg !21
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1128 = or disjoint i64 %address.local1113, 17, !dbg !21
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1128, !dbg !21
  store half %2118, ptr addrspace(1) %address.local1129, align 2, !dbg !21
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1131 = or disjoint i64 %address.local1113, 18, !dbg !21
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1130, i64 %address.local1131, !dbg !21
  store half %2117, ptr addrspace(1) %address.local1132, align 2, !dbg !21
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1134 = or disjoint i64 %address.local1113, 19, !dbg !21
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1133, i64 %address.local1134, !dbg !21
  store half %2116, ptr addrspace(1) %address.local1135, align 2, !dbg !21
  %address.local1136 = shl i64 %address.local1108, 7, !dbg !21
  %address.local1137 = add i64 %address.local1136, 8192, !dbg !21
  %address.local1138 = add nuw nsw i64 %2137, %address.local1137, !dbg !21
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1113, !dbg !21
  store half %2104, ptr addrspace(1) %address.local1140, align 2, !dbg !21
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1116, !dbg !21
  store half %2103, ptr addrspace(1) %address.local1142, align 2, !dbg !21
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1119, !dbg !21
  store half %2102, ptr addrspace(1) %address.local1144, align 2, !dbg !21
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1122, !dbg !21
  store half %2101, ptr addrspace(1) %address.local1146, align 2, !dbg !21
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1125, !dbg !21
  store half %2089, ptr addrspace(1) %address.local1148, align 2, !dbg !21
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1128, !dbg !21
  store half %2088, ptr addrspace(1) %address.local1150, align 2, !dbg !21
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1131, !dbg !21
  store half %2087, ptr addrspace(1) %address.local1152, align 2, !dbg !21
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1134, !dbg !21
  store half %2086, ptr addrspace(1) %address.local1154, align 2, !dbg !21
  br label %2138, !dbg !21

2138:                                             ; preds = %2074, %1607
  br i1 true, label %2139, label %2177, !dbg !21

2139:                                             ; preds = %2138
  %2140 = insertelement <4 x float> poison, float %2044, i64 0, !dbg !19
  %2141 = insertelement <4 x float> %2140, float %2050, i64 1, !dbg !19
  %2142 = insertelement <4 x float> %2141, float %2045, i64 2, !dbg !19
  %2143 = insertelement <4 x float> %2142, float %2051, i64 3, !dbg !19
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1967, !dbg !19
  %normalized109 = fmul <4 x float> %2143, %row.reciprocal108, !dbg !19
  %2144 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !20
  %2145 = extractelement <4 x half> %2144, i64 3, !dbg !20
  %2146 = extractelement <4 x half> %2144, i64 2, !dbg !20
  %2147 = extractelement <4 x half> %2144, i64 1, !dbg !20
  %2148 = extractelement <4 x half> %2144, i64 0, !dbg !20
  %2149 = insertelement <4 x float> poison, float %1936, i64 0, !dbg !19
  %2150 = insertelement <4 x float> %2149, float %1942, i64 1, !dbg !19
  %2151 = insertelement <4 x float> %2150, float %1937, i64 2, !dbg !19
  %2152 = insertelement <4 x float> %2151, float %1943, i64 3, !dbg !19
  %normalized110 = fmul <4 x float> %2152, %row.reciprocal108, !dbg !19
  %2153 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %2154 = extractelement <4 x half> %2153, i64 3, !dbg !20
  %2155 = extractelement <4 x half> %2153, i64 2, !dbg !20
  %2156 = extractelement <4 x half> %2153, i64 1, !dbg !20
  %2157 = extractelement <4 x half> %2153, i64 0, !dbg !20
  %2158 = insertelement <4 x float> poison, float %1827, i64 0, !dbg !19
  %2159 = insertelement <4 x float> %2158, float %1833, i64 1, !dbg !19
  %2160 = insertelement <4 x float> %2159, float %1828, i64 2, !dbg !19
  %2161 = insertelement <4 x float> %2160, float %1834, i64 3, !dbg !19
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1750, !dbg !19
  %normalized112 = fmul <4 x float> %2161, %row.reciprocal111, !dbg !19
  %2162 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !20
  %2163 = extractelement <4 x half> %2162, i64 3, !dbg !20
  %2164 = extractelement <4 x half> %2162, i64 2, !dbg !20
  %2165 = extractelement <4 x half> %2162, i64 1, !dbg !20
  %2166 = extractelement <4 x half> %2162, i64 0, !dbg !20
  %2167 = insertelement <4 x float> poison, float %1719, i64 0, !dbg !19
  %2168 = insertelement <4 x float> %2167, float %1725, i64 1, !dbg !19
  %2169 = insertelement <4 x float> %2168, float %1720, i64 2, !dbg !19
  %2170 = insertelement <4 x float> %2169, float %1726, i64 3, !dbg !19
  %normalized113 = fmul <4 x float> %2170, %row.reciprocal111, !dbg !19
  %2171 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %2172 = extractelement <4 x half> %2171, i64 3, !dbg !20
  %2173 = extractelement <4 x half> %2171, i64 2, !dbg !20
  %2174 = extractelement <4 x half> %2171, i64 1, !dbg !20
  %2175 = extractelement <4 x half> %2171, i64 0, !dbg !20
  %2176 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1155 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1156 = zext nneg i32 %address.thread1155 to i64, !dbg !8
  %address.local1157 = add i64 %25, %address.local1156, !dbg !21
  %address.local1158 = lshr i64 %address.local1156, 6, !dbg !11
  %address.local1159 = shl nuw nsw i64 %address.local1158, 4, !dbg !21
  %address.local1160 = add i64 %address.local1157, %address.local1159, !dbg !21
  %address.local1161 = and i64 %address.local1156, 240, !dbg !21
  %address.local1162 = sub i64 %address.local1160, %address.local1161, !dbg !21
  %address.local1163 = add i64 %address.local1162, %26, !dbg !21
  %address.local1164 = shl nuw nsw i64 %address.local1163, 7, !dbg !21
  %address.local1165 = add nuw nsw i64 %2176, %address.local1164, !dbg !21
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1167 = lshr i64 %address.local1156, 2, !dbg !11
  %address.local1168 = and i64 %address.local1167, 12, !dbg !11
  %address.local1169 = or disjoint i64 %address.local1168, 32, !dbg !21
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1166, i64 %address.local1169, !dbg !21
  store half %2175, ptr addrspace(1) %address.local1170, align 2, !dbg !21
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1172 = or disjoint i64 %address.local1168, 33, !dbg !21
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1171, i64 %address.local1172, !dbg !21
  store half %2174, ptr addrspace(1) %address.local1173, align 2, !dbg !21
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1175 = or disjoint i64 %address.local1168, 34, !dbg !21
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1174, i64 %address.local1175, !dbg !21
  store half %2173, ptr addrspace(1) %address.local1176, align 2, !dbg !21
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1178 = or disjoint i64 %address.local1168, 35, !dbg !21
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1177, i64 %address.local1178, !dbg !21
  store half %2172, ptr addrspace(1) %address.local1179, align 2, !dbg !21
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1181 = or disjoint i64 %address.local1168, 48, !dbg !21
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1180, i64 %address.local1181, !dbg !21
  store half %2166, ptr addrspace(1) %address.local1182, align 2, !dbg !21
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1184 = or disjoint i64 %address.local1168, 49, !dbg !21
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1184, !dbg !21
  store half %2165, ptr addrspace(1) %address.local1185, align 2, !dbg !21
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1187 = or disjoint i64 %address.local1168, 50, !dbg !21
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1186, i64 %address.local1187, !dbg !21
  store half %2164, ptr addrspace(1) %address.local1188, align 2, !dbg !21
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1190 = or i64 %address.local1167, 51, !dbg !21
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1189, i64 %address.local1190, !dbg !21
  store half %2163, ptr addrspace(1) %address.local1191, align 2, !dbg !21
  %address.local1192 = shl i64 %address.local1163, 7, !dbg !21
  %address.local1193 = add i64 %address.local1192, 8192, !dbg !21
  %address.local1194 = add nuw nsw i64 %2176, %address.local1193, !dbg !21
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1195, i64 %address.local1169, !dbg !21
  store half %2157, ptr addrspace(1) %address.local1196, align 2, !dbg !21
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1197, i64 %address.local1172, !dbg !21
  store half %2156, ptr addrspace(1) %address.local1198, align 2, !dbg !21
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1199, i64 %address.local1175, !dbg !21
  store half %2155, ptr addrspace(1) %address.local1200, align 2, !dbg !21
  %address.local1201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1201, i64 %address.local1178, !dbg !21
  store half %2154, ptr addrspace(1) %address.local1202, align 2, !dbg !21
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1203, i64 %address.local1181, !dbg !21
  store half %2148, ptr addrspace(1) %address.local1204, align 2, !dbg !21
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1205, i64 %address.local1184, !dbg !21
  store half %2147, ptr addrspace(1) %address.local1206, align 2, !dbg !21
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1207, i64 %address.local1187, !dbg !21
  store half %2146, ptr addrspace(1) %address.local1208, align 2, !dbg !21
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1209, i64 %address.local1190, !dbg !21
  store half %2145, ptr addrspace(1) %address.local1210, align 2, !dbg !21
  br label %2177, !dbg !21

2177:                                             ; preds = %2139, %2138
  br i1 true, label %2178, label %2216, !dbg !21

2178:                                             ; preds = %2177
  %2179 = insertelement <4 x float> poison, float %2016, i64 0, !dbg !19
  %2180 = insertelement <4 x float> %2179, float %2022, i64 1, !dbg !19
  %2181 = insertelement <4 x float> %2180, float %2017, i64 2, !dbg !19
  %2182 = insertelement <4 x float> %2181, float %2023, i64 3, !dbg !19
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1967, !dbg !19
  %normalized115 = fmul <4 x float> %2182, %row.reciprocal114, !dbg !19
  %2183 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !20
  %2184 = extractelement <4 x half> %2183, i64 3, !dbg !20
  %2185 = extractelement <4 x half> %2183, i64 2, !dbg !20
  %2186 = extractelement <4 x half> %2183, i64 1, !dbg !20
  %2187 = extractelement <4 x half> %2183, i64 0, !dbg !20
  %2188 = insertelement <4 x float> poison, float %1907, i64 0, !dbg !19
  %2189 = insertelement <4 x float> %2188, float %1913, i64 1, !dbg !19
  %2190 = insertelement <4 x float> %2189, float %1908, i64 2, !dbg !19
  %2191 = insertelement <4 x float> %2190, float %1914, i64 3, !dbg !19
  %normalized116 = fmul <4 x float> %2191, %row.reciprocal114, !dbg !19
  %2192 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %2193 = extractelement <4 x half> %2192, i64 3, !dbg !20
  %2194 = extractelement <4 x half> %2192, i64 2, !dbg !20
  %2195 = extractelement <4 x half> %2192, i64 1, !dbg !20
  %2196 = extractelement <4 x half> %2192, i64 0, !dbg !20
  %2197 = insertelement <4 x float> poison, float %1799, i64 0, !dbg !19
  %2198 = insertelement <4 x float> %2197, float %1805, i64 1, !dbg !19
  %2199 = insertelement <4 x float> %2198, float %1800, i64 2, !dbg !19
  %2200 = insertelement <4 x float> %2199, float %1806, i64 3, !dbg !19
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1750, !dbg !19
  %normalized118 = fmul <4 x float> %2200, %row.reciprocal117, !dbg !19
  %2201 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !20
  %2202 = extractelement <4 x half> %2201, i64 3, !dbg !20
  %2203 = extractelement <4 x half> %2201, i64 2, !dbg !20
  %2204 = extractelement <4 x half> %2201, i64 1, !dbg !20
  %2205 = extractelement <4 x half> %2201, i64 0, !dbg !20
  %2206 = insertelement <4 x float> poison, float %1690, i64 0, !dbg !19
  %2207 = insertelement <4 x float> %2206, float %1696, i64 1, !dbg !19
  %2208 = insertelement <4 x float> %2207, float %1691, i64 2, !dbg !19
  %2209 = insertelement <4 x float> %2208, float %1697, i64 3, !dbg !19
  %normalized119 = fmul <4 x float> %2209, %row.reciprocal117, !dbg !19
  %2210 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %2211 = extractelement <4 x half> %2210, i64 3, !dbg !20
  %2212 = extractelement <4 x half> %2210, i64 2, !dbg !20
  %2213 = extractelement <4 x half> %2210, i64 1, !dbg !20
  %2214 = extractelement <4 x half> %2210, i64 0, !dbg !20
  %2215 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1211 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1212 = zext nneg i32 %address.thread1211 to i64, !dbg !8
  %address.local1213 = add i64 %25, %address.local1212, !dbg !21
  %address.local1214 = lshr i64 %address.local1212, 6, !dbg !11
  %address.local1215 = shl nuw nsw i64 %address.local1214, 4, !dbg !21
  %address.local1216 = add i64 %address.local1213, %address.local1215, !dbg !21
  %address.local1217 = and i64 %address.local1212, 240, !dbg !21
  %address.local1218 = sub i64 %address.local1216, %address.local1217, !dbg !21
  %address.local1219 = add i64 %address.local1218, %26, !dbg !21
  %address.local1220 = shl nuw nsw i64 %address.local1219, 7, !dbg !21
  %address.local1221 = add nuw nsw i64 %2215, %address.local1220, !dbg !21
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1223 = lshr i64 %address.local1212, 2, !dbg !11
  %address.local1224 = and i64 %address.local1223, 12, !dbg !11
  %address.local1225 = or disjoint i64 %address.local1224, 64, !dbg !21
  %address.local1226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1222, i64 %address.local1225, !dbg !21
  store half %2214, ptr addrspace(1) %address.local1226, align 2, !dbg !21
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1228 = or disjoint i64 %address.local1224, 65, !dbg !21
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1227, i64 %address.local1228, !dbg !21
  store half %2213, ptr addrspace(1) %address.local1229, align 2, !dbg !21
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1231 = or disjoint i64 %address.local1224, 66, !dbg !21
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1230, i64 %address.local1231, !dbg !21
  store half %2212, ptr addrspace(1) %address.local1232, align 2, !dbg !21
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1234 = or disjoint i64 %address.local1224, 67, !dbg !21
  %address.local1235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1233, i64 %address.local1234, !dbg !21
  store half %2211, ptr addrspace(1) %address.local1235, align 2, !dbg !21
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1237 = or disjoint i64 %address.local1224, 80, !dbg !21
  %address.local1238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1236, i64 %address.local1237, !dbg !21
  store half %2205, ptr addrspace(1) %address.local1238, align 2, !dbg !21
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1240 = or disjoint i64 %address.local1224, 81, !dbg !21
  %address.local1241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1239, i64 %address.local1240, !dbg !21
  store half %2204, ptr addrspace(1) %address.local1241, align 2, !dbg !21
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1243 = or disjoint i64 %address.local1224, 82, !dbg !21
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1242, i64 %address.local1243, !dbg !21
  store half %2203, ptr addrspace(1) %address.local1244, align 2, !dbg !21
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1246 = or disjoint i64 %address.local1224, 83, !dbg !21
  %address.local1247 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1245, i64 %address.local1246, !dbg !21
  store half %2202, ptr addrspace(1) %address.local1247, align 2, !dbg !21
  %address.local1248 = shl i64 %address.local1219, 7, !dbg !21
  %address.local1249 = add i64 %address.local1248, 8192, !dbg !21
  %address.local1250 = add nuw nsw i64 %2215, %address.local1249, !dbg !21
  %address.local1251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1252 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1251, i64 %address.local1225, !dbg !21
  store half %2196, ptr addrspace(1) %address.local1252, align 2, !dbg !21
  %address.local1253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1253, i64 %address.local1228, !dbg !21
  store half %2195, ptr addrspace(1) %address.local1254, align 2, !dbg !21
  %address.local1255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1255, i64 %address.local1231, !dbg !21
  store half %2194, ptr addrspace(1) %address.local1256, align 2, !dbg !21
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1257, i64 %address.local1234, !dbg !21
  store half %2193, ptr addrspace(1) %address.local1258, align 2, !dbg !21
  %address.local1259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1259, i64 %address.local1237, !dbg !21
  store half %2187, ptr addrspace(1) %address.local1260, align 2, !dbg !21
  %address.local1261 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1261, i64 %address.local1240, !dbg !21
  store half %2186, ptr addrspace(1) %address.local1262, align 2, !dbg !21
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1263, i64 %address.local1243, !dbg !21
  store half %2185, ptr addrspace(1) %address.local1264, align 2, !dbg !21
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1265, i64 %address.local1246, !dbg !21
  store half %2184, ptr addrspace(1) %address.local1266, align 2, !dbg !21
  br label %2216, !dbg !21

2216:                                             ; preds = %2178, %2177
  br i1 true, label %2217, label %2255, !dbg !21

2217:                                             ; preds = %2216
  %2218 = insertelement <4 x float> poison, float %1988, i64 0, !dbg !19
  %2219 = insertelement <4 x float> %2218, float %1994, i64 1, !dbg !19
  %2220 = insertelement <4 x float> %2219, float %1989, i64 2, !dbg !19
  %2221 = insertelement <4 x float> %2220, float %1995, i64 3, !dbg !19
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1967, !dbg !19
  %normalized121 = fmul <4 x float> %2221, %row.reciprocal120, !dbg !19
  %2222 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !20
  %2223 = extractelement <4 x half> %2222, i64 3, !dbg !20
  %2224 = extractelement <4 x half> %2222, i64 2, !dbg !20
  %2225 = extractelement <4 x half> %2222, i64 1, !dbg !20
  %2226 = extractelement <4 x half> %2222, i64 0, !dbg !20
  %2227 = insertelement <4 x float> poison, float %1878, i64 0, !dbg !19
  %2228 = insertelement <4 x float> %2227, float %1884, i64 1, !dbg !19
  %2229 = insertelement <4 x float> %2228, float %1879, i64 2, !dbg !19
  %2230 = insertelement <4 x float> %2229, float %1885, i64 3, !dbg !19
  %normalized122 = fmul <4 x float> %2230, %row.reciprocal120, !dbg !19
  %2231 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %2232 = extractelement <4 x half> %2231, i64 3, !dbg !20
  %2233 = extractelement <4 x half> %2231, i64 2, !dbg !20
  %2234 = extractelement <4 x half> %2231, i64 1, !dbg !20
  %2235 = extractelement <4 x half> %2231, i64 0, !dbg !20
  %2236 = insertelement <4 x float> poison, float %1771, i64 0, !dbg !19
  %2237 = insertelement <4 x float> %2236, float %1777, i64 1, !dbg !19
  %2238 = insertelement <4 x float> %2237, float %1772, i64 2, !dbg !19
  %2239 = insertelement <4 x float> %2238, float %1778, i64 3, !dbg !19
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1750, !dbg !19
  %normalized124 = fmul <4 x float> %2239, %row.reciprocal123, !dbg !19
  %2240 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %2241 = extractelement <4 x half> %2240, i64 3, !dbg !20
  %2242 = extractelement <4 x half> %2240, i64 2, !dbg !20
  %2243 = extractelement <4 x half> %2240, i64 1, !dbg !20
  %2244 = extractelement <4 x half> %2240, i64 0, !dbg !20
  %2245 = insertelement <4 x float> poison, float %1661, i64 0, !dbg !19
  %2246 = insertelement <4 x float> %2245, float %1667, i64 1, !dbg !19
  %2247 = insertelement <4 x float> %2246, float %1662, i64 2, !dbg !19
  %2248 = insertelement <4 x float> %2247, float %1668, i64 3, !dbg !19
  %normalized125 = fmul <4 x float> %2248, %row.reciprocal123, !dbg !19
  %2249 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %2250 = extractelement <4 x half> %2249, i64 3, !dbg !20
  %2251 = extractelement <4 x half> %2249, i64 2, !dbg !20
  %2252 = extractelement <4 x half> %2249, i64 1, !dbg !20
  %2253 = extractelement <4 x half> %2249, i64 0, !dbg !20
  %2254 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1267 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1268 = zext nneg i32 %address.thread1267 to i64, !dbg !8
  %address.local1269 = add i64 %25, %address.local1268, !dbg !21
  %address.local1270 = lshr i64 %address.local1268, 6, !dbg !11
  %address.local1271 = shl nuw nsw i64 %address.local1270, 4, !dbg !21
  %address.local1272 = add i64 %address.local1269, %address.local1271, !dbg !21
  %address.local1273 = and i64 %address.local1268, 240, !dbg !21
  %address.local1274 = sub i64 %address.local1272, %address.local1273, !dbg !21
  %address.local1275 = add i64 %address.local1274, %26, !dbg !21
  %address.local1276 = shl nuw nsw i64 %address.local1275, 7, !dbg !21
  %address.local1277 = add nuw nsw i64 %2254, %address.local1276, !dbg !21
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1279 = lshr i64 %address.local1268, 2, !dbg !11
  %address.local1280 = and i64 %address.local1279, 12, !dbg !11
  %address.local1281 = or disjoint i64 %address.local1280, 96, !dbg !21
  %address.local1282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1278, i64 %address.local1281, !dbg !21
  store half %2253, ptr addrspace(1) %address.local1282, align 2, !dbg !21
  %address.local1283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1284 = or disjoint i64 %address.local1280, 97, !dbg !21
  %address.local1285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1283, i64 %address.local1284, !dbg !21
  store half %2252, ptr addrspace(1) %address.local1285, align 2, !dbg !21
  %address.local1286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1287 = or disjoint i64 %address.local1280, 98, !dbg !21
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1286, i64 %address.local1287, !dbg !21
  store half %2251, ptr addrspace(1) %address.local1288, align 2, !dbg !21
  %address.local1289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1290 = or disjoint i64 %address.local1280, 99, !dbg !21
  %address.local1291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1289, i64 %address.local1290, !dbg !21
  store half %2250, ptr addrspace(1) %address.local1291, align 2, !dbg !21
  %address.local1292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1293 = or disjoint i64 %address.local1280, 112, !dbg !21
  %address.local1294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1292, i64 %address.local1293, !dbg !21
  store half %2244, ptr addrspace(1) %address.local1294, align 2, !dbg !21
  %address.local1295 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1296 = or disjoint i64 %address.local1280, 113, !dbg !21
  %address.local1297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1295, i64 %address.local1296, !dbg !21
  store half %2243, ptr addrspace(1) %address.local1297, align 2, !dbg !21
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1299 = or disjoint i64 %address.local1280, 114, !dbg !21
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1298, i64 %address.local1299, !dbg !21
  store half %2242, ptr addrspace(1) %address.local1300, align 2, !dbg !21
  %address.local1301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1302 = or i64 %address.local1279, 115, !dbg !21
  %address.local1303 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1301, i64 %address.local1302, !dbg !21
  store half %2241, ptr addrspace(1) %address.local1303, align 2, !dbg !21
  %address.local1304 = shl i64 %address.local1275, 7, !dbg !21
  %address.local1305 = add i64 %address.local1304, 8192, !dbg !21
  %address.local1306 = add nuw nsw i64 %2254, %address.local1305, !dbg !21
  %address.local1307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1307, i64 %address.local1281, !dbg !21
  store half %2235, ptr addrspace(1) %address.local1308, align 2, !dbg !21
  %address.local1309 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1309, i64 %address.local1284, !dbg !21
  store half %2234, ptr addrspace(1) %address.local1310, align 2, !dbg !21
  %address.local1311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1311, i64 %address.local1287, !dbg !21
  store half %2233, ptr addrspace(1) %address.local1312, align 2, !dbg !21
  %address.local1313 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1314 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1313, i64 %address.local1290, !dbg !21
  store half %2232, ptr addrspace(1) %address.local1314, align 2, !dbg !21
  %address.local1315 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1315, i64 %address.local1293, !dbg !21
  store half %2226, ptr addrspace(1) %address.local1316, align 2, !dbg !21
  %address.local1317 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1318 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1317, i64 %address.local1296, !dbg !21
  store half %2225, ptr addrspace(1) %address.local1318, align 2, !dbg !21
  %address.local1319 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1320 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1319, i64 %address.local1299, !dbg !21
  store half %2224, ptr addrspace(1) %address.local1320, align 2, !dbg !21
  %address.local1321 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1322 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1321, i64 %address.local1302, !dbg !21
  store half %2223, ptr addrspace(1) %address.local1322, align 2, !dbg !21
  br label %2255, !dbg !21

2255:                                             ; preds = %2217, %2216
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/exp2_fixed/run-20261027-123528-0u0so_ep/grid-000002")
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
