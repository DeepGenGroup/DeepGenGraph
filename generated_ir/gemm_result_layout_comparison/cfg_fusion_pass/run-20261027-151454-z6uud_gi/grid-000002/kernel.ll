; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run-20261027-151454-z6uud_gi/grid-000002/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run-20261027-151454-z6uud_gi/grid-000002/input.mlir"

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
  %dma.buffer.offset1780 = trunc i64 %39 to i32, !dbg !8
  %40 = trunc i64 %37 to i32, !dbg !8
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !8
  %42 = lshr i64 %37, 32, !dbg !8
  %43 = trunc i64 %42 to i32, !dbg !8
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !8
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1781 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !8
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !8
  %dma.buffer.lds1782 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1782, i32 %dma.buffer.offset1780, <4 x i32> %dma.buffer.rsrc1781) #10, !dbg !8
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
  %dma.buffer.offset1777 = trunc i64 %53 to i32, !dbg !8
  %54 = trunc i64 %51 to i32, !dbg !8
  %55 = insertelement <4 x i32> poison, i32 %54, i64 0, !dbg !8
  %56 = lshr i64 %51, 32, !dbg !8
  %57 = trunc i64 %56 to i32, !dbg !8
  %58 = insertelement <4 x i32> %55, i32 %57, i64 1, !dbg !8
  %59 = insertelement <4 x i32> %58, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1778 = insertelement <4 x i32> %59, i32 131072, i64 3, !dbg !8
  %60 = ptrtoint ptr addrspace(3) %50 to i32, !dbg !8
  %dma.buffer.lds1779 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !8
  %61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1779, i32 %dma.buffer.offset1777, <4 x i32> %dma.buffer.rsrc1778) #10, !dbg !8
  %62 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %63 = sext i32 %62 to i64, !dbg !5
  %64 = shl i64 %15, 7, !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %65 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %66 = getelementptr i8, ptr addrspace(1) %65, i64 64, !dbg !8
  %address.local528 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local506, !dbg !8
  %address.local529 = ptrtoint ptr addrspace(3) %address.local528 to i64, !dbg !8
  %address.local530 = trunc i64 %address.local529 to i32, !dbg !8
  %67 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local530), !dbg !8
  %68 = zext i32 %67 to i64, !dbg !8
  %69 = inttoptr i64 %68 to ptr addrspace(3), !dbg !8
  %address.local531 = getelementptr [2 x i8], ptr addrspace(1) %66, i64 %address.local514, !dbg !8
  %70 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %71 = ptrtoint ptr addrspace(1) %address.local531 to i64, !dbg !8
  %72 = sub i64 %71, %70, !dbg !8
  %dma.buffer.offset1774 = trunc i64 %72 to i32, !dbg !8
  %73 = trunc i64 %70 to i32, !dbg !8
  %74 = insertelement <4 x i32> poison, i32 %73, i64 0, !dbg !8
  %75 = lshr i64 %70, 32, !dbg !8
  %76 = trunc i64 %75 to i32, !dbg !8
  %77 = insertelement <4 x i32> %74, i32 %76, i64 1, !dbg !8
  %78 = insertelement <4 x i32> %77, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1775 = insertelement <4 x i32> %78, i32 131072, i64 3, !dbg !8
  %79 = ptrtoint ptr addrspace(3) %69 to i32, !dbg !8
  %dma.buffer.lds1776 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %79), !dbg !8
  %80 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1776, i32 %dma.buffer.offset1774, <4 x i32> %dma.buffer.rsrc1775) #10, !dbg !8
  %address.local532 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %address.local533 = ptrtoint ptr addrspace(3) %address.local532 to i64, !dbg !8
  %address.local534 = trunc i64 %address.local533 to i32, !dbg !8
  %81 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local534), !dbg !8
  %82 = zext i32 %81 to i64, !dbg !8
  %83 = inttoptr i64 %82 to ptr addrspace(3), !dbg !8
  %address.local535 = getelementptr [2 x i8], ptr addrspace(1) %66, i64 %address.local526, !dbg !8
  %84 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %85 = ptrtoint ptr addrspace(1) %address.local535 to i64, !dbg !8
  %86 = sub i64 %85, %84, !dbg !8
  %dma.buffer.offset1771 = trunc i64 %86 to i32, !dbg !8
  %87 = trunc i64 %84 to i32, !dbg !8
  %88 = insertelement <4 x i32> poison, i32 %87, i64 0, !dbg !8
  %89 = lshr i64 %84, 32, !dbg !8
  %90 = trunc i64 %89 to i32, !dbg !8
  %91 = insertelement <4 x i32> %88, i32 %90, i64 1, !dbg !8
  %92 = insertelement <4 x i32> %91, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1772 = insertelement <4 x i32> %92, i32 131072, i64 3, !dbg !8
  %93 = ptrtoint ptr addrspace(3) %83 to i32, !dbg !8
  %dma.buffer.lds1773 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %93), !dbg !8
  %94 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1773, i32 %dma.buffer.offset1771, <4 x i32> %dma.buffer.rsrc1772) #10, !dbg !8
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
  %95 = load <4 x half>, ptr addrspace(3) %address.local548, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local548, i64 4096, !dbg !11
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %97 = fmul <4 x half> %95, splat (half 0xH3015), !dbg !11
  %lds.const.gep1579 = getelementptr i8, ptr addrspace(3) %address.local548, i64 1024, !dbg !11
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !11
  %99 = fmul <4 x half> %96, splat (half 0xH3015), !dbg !11
  %lds.const.gep1580 = getelementptr i8, ptr addrspace(3) %address.local548, i64 5120, !dbg !11
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !11
  %101 = fmul <4 x half> %98, splat (half 0xH3015), !dbg !11
  %102 = fmul <4 x half> %100, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %103 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %104 = getelementptr i8, ptr addrspace(1) %103, i64 128, !dbg !8
  %105 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local509), !dbg !8
  %106 = zext i32 %105 to i64, !dbg !8
  %107 = inttoptr i64 %106 to ptr addrspace(3), !dbg !8
  %address.local591 = getelementptr [2 x i8], ptr addrspace(1) %104, i64 %address.local514, !dbg !8
  %108 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %109 = ptrtoint ptr addrspace(1) %address.local591 to i64, !dbg !8
  %110 = sub i64 %109, %108, !dbg !8
  %dma.buffer.offset1768 = trunc i64 %110 to i32, !dbg !8
  %111 = trunc i64 %108 to i32, !dbg !8
  %112 = insertelement <4 x i32> poison, i32 %111, i64 0, !dbg !8
  %113 = lshr i64 %108, 32, !dbg !8
  %114 = trunc i64 %113 to i32, !dbg !8
  %115 = insertelement <4 x i32> %112, i32 %114, i64 1, !dbg !8
  %116 = insertelement <4 x i32> %115, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1769 = insertelement <4 x i32> %116, i32 131072, i64 3, !dbg !8
  %117 = ptrtoint ptr addrspace(3) %107 to i32, !dbg !8
  %dma.buffer.lds1770 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %117), !dbg !8
  %118 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1770, i32 %dma.buffer.offset1768, <4 x i32> %dma.buffer.rsrc1769) #10, !dbg !8
  %119 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !8
  %120 = zext i32 %119 to i64, !dbg !8
  %121 = inttoptr i64 %120 to ptr addrspace(3), !dbg !8
  %address.local592 = getelementptr [2 x i8], ptr addrspace(1) %104, i64 %address.local526, !dbg !8
  %122 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %123 = ptrtoint ptr addrspace(1) %address.local592 to i64, !dbg !8
  %124 = sub i64 %123, %122, !dbg !8
  %dma.buffer.offset1765 = trunc i64 %124 to i32, !dbg !8
  %125 = trunc i64 %122 to i32, !dbg !8
  %126 = insertelement <4 x i32> poison, i32 %125, i64 0, !dbg !8
  %127 = lshr i64 %122, 32, !dbg !8
  %128 = trunc i64 %127 to i32, !dbg !8
  %129 = insertelement <4 x i32> %126, i32 %128, i64 1, !dbg !8
  %130 = insertelement <4 x i32> %129, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1766 = insertelement <4 x i32> %130, i32 131072, i64 3, !dbg !8
  %131 = ptrtoint ptr addrspace(3) %121 to i32, !dbg !8
  %dma.buffer.lds1767 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %131), !dbg !8
  %132 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1767, i32 %dma.buffer.offset1765, <4 x i32> %dma.buffer.rsrc1766) #10, !dbg !8
  %lds.const.gep1581 = getelementptr i8, ptr addrspace(3) %address.local548, i64 8192, !dbg !11
  %133 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !11
  %lds.const.gep1582 = getelementptr i8, ptr addrspace(3) %address.local548, i64 12288, !dbg !11
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !11
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %address.local548, i64 9216, !dbg !11
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !11
  %136 = fmul <4 x half> %133, splat (half 0xH3015), !dbg !11
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local548, i64 13312, !dbg !11
  %137 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !11
  %138 = fmul <4 x half> %134, splat (half 0xH3015), !dbg !11
  %139 = fmul <4 x half> %135, splat (half 0xH3015), !dbg !11
  %140 = fmul <4 x half> %137, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %141 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %142 = getelementptr i8, ptr addrspace(1) %141, i64 192, !dbg !8
  %143 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local530), !dbg !8
  %144 = zext i32 %143 to i64, !dbg !8
  %145 = inttoptr i64 %144 to ptr addrspace(3), !dbg !8
  %address.local597 = getelementptr [2 x i8], ptr addrspace(1) %142, i64 %address.local514, !dbg !8
  %146 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %147 = ptrtoint ptr addrspace(1) %address.local597 to i64, !dbg !8
  %148 = sub i64 %147, %146, !dbg !8
  %dma.buffer.offset1762 = trunc i64 %148 to i32, !dbg !8
  %149 = trunc i64 %146 to i32, !dbg !8
  %150 = insertelement <4 x i32> poison, i32 %149, i64 0, !dbg !8
  %151 = lshr i64 %146, 32, !dbg !8
  %152 = trunc i64 %151 to i32, !dbg !8
  %153 = insertelement <4 x i32> %150, i32 %152, i64 1, !dbg !8
  %154 = insertelement <4 x i32> %153, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1763 = insertelement <4 x i32> %154, i32 131072, i64 3, !dbg !8
  %155 = ptrtoint ptr addrspace(3) %145 to i32, !dbg !8
  %dma.buffer.lds1764 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %155), !dbg !8
  %156 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1764, i32 %dma.buffer.offset1762, <4 x i32> %dma.buffer.rsrc1763) #10, !dbg !8
  %157 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local534), !dbg !8
  %158 = zext i32 %157 to i64, !dbg !8
  %159 = inttoptr i64 %158 to ptr addrspace(3), !dbg !8
  %address.local598 = getelementptr [2 x i8], ptr addrspace(1) %142, i64 %address.local526, !dbg !8
  %160 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %161 = ptrtoint ptr addrspace(1) %address.local598 to i64, !dbg !8
  %162 = sub i64 %161, %160, !dbg !8
  %dma.buffer.offset1759 = trunc i64 %162 to i32, !dbg !8
  %163 = trunc i64 %160 to i32, !dbg !8
  %164 = insertelement <4 x i32> poison, i32 %163, i64 0, !dbg !8
  %165 = lshr i64 %160, 32, !dbg !8
  %166 = trunc i64 %165 to i32, !dbg !8
  %167 = insertelement <4 x i32> %164, i32 %166, i64 1, !dbg !8
  %168 = insertelement <4 x i32> %167, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1760 = insertelement <4 x i32> %168, i32 131072, i64 3, !dbg !8
  %169 = ptrtoint ptr addrspace(3) %159 to i32, !dbg !8
  %dma.buffer.lds1761 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %169), !dbg !8
  %170 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1761, i32 %dma.buffer.offset1759, <4 x i32> %dma.buffer.rsrc1760) #10, !dbg !8
  %171 = load <4 x half>, ptr addrspace(3) %address.local548, align 8, !dbg !11
  %lds.const.gep1585 = getelementptr i8, ptr addrspace(3) %address.local548, i64 4096, !dbg !11
  %172 = load <4 x half>, ptr addrspace(3) %lds.const.gep1585, align 8, !dbg !11
  %lds.const.gep1586 = getelementptr i8, ptr addrspace(3) %address.local548, i64 1024, !dbg !11
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1586, align 8, !dbg !11
  %174 = fmul <4 x half> %171, splat (half 0xH3015), !dbg !11
  %lds.const.gep1587 = getelementptr i8, ptr addrspace(3) %address.local548, i64 5120, !dbg !11
  %175 = load <4 x half>, ptr addrspace(3) %lds.const.gep1587, align 8, !dbg !11
  %176 = fmul <4 x half> %172, splat (half 0xH3015), !dbg !11
  %177 = fmul <4 x half> %173, splat (half 0xH3015), !dbg !11
  %178 = fmul <4 x half> %175, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1588 = getelementptr i8, ptr addrspace(3) %address.local548, i64 8192, !dbg !11
  %179 = load <4 x half>, ptr addrspace(3) %lds.const.gep1588, align 8, !dbg !11
  %lds.const.gep1589 = getelementptr i8, ptr addrspace(3) %address.local548, i64 12288, !dbg !11
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !11
  %lds.const.gep1590 = getelementptr i8, ptr addrspace(3) %address.local548, i64 9216, !dbg !11
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !11
  %182 = fmul <4 x half> %179, splat (half 0xH3015), !dbg !11
  %lds.const.gep1591 = getelementptr i8, ptr addrspace(3) %address.local548, i64 13312, !dbg !11
  %183 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !11
  %184 = fmul <4 x half> %180, splat (half 0xH3015), !dbg !11
  %185 = fmul <4 x half> %181, splat (half 0xH3015), !dbg !11
  %186 = fmul <4 x half> %183, splat (half 0xH3015), !dbg !11
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %187 = or disjoint i64 %64, 127, !dbg !13
  %188 = add i64 %64, 128, !dbg !13
  %189 = icmp uge i64 %188, %187, !dbg !13
  %190 = sub nsw i64 32, %14, !dbg !15
  %191 = shl i64 %16, 7, !dbg !16
  %192 = and i64 %191, 3968, !dbg !16
  %193 = shl nsw i64 %63, 12, !dbg !16
  %194 = or disjoint i64 %192, %193, !dbg !16
  %195 = shl nuw nsw i64 %32, 2, !dbg !17
  %196 = shl nuw nsw i64 %32, 2, !dbg !17
  %197 = and i64 %196, 1008, !dbg !17
  %198 = sub nsw i64 %195, %197, !dbg !17
  %199 = and i32 %31, 1, !dbg !17
  %index.compact445 = zext nneg i32 %199 to i64, !dbg !17
  %200 = or disjoint i64 %198, %index.compact445, !dbg !17
  %201 = add nuw nsw i64 %200, 1, !dbg !17
  %202 = sdiv i64 %201, 2, !dbg !17
  br label %203, !dbg !7

203:                                              ; preds = %838, %4
  %204 = phi i64 [ %1615, %838 ], [ 0, %4 ], !dbg !7
  %205 = phi [2 x <8 x float>] [ %1600, %838 ], [ zeroinitializer, %4 ], !dbg !7
  %206 = phi [2 x <1 x float>] [ %1206, %838 ], [ zeroinitializer, %4 ], !dbg !7
  %207 = phi [2 x <1 x float>] [ %1602, %838 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %208 = phi [2 x <8 x float>] [ %1606, %838 ], [ zeroinitializer, %4 ], !dbg !7
  %209 = phi [2 x <8 x float>] [ %1610, %838 ], [ zeroinitializer, %4 ], !dbg !7
  %210 = phi [2 x <8 x float>] [ %1614, %838 ], [ zeroinitializer, %4 ], !dbg !7
  %211 = icmp slt i64 %204, %190, !dbg !7
  br i1 %211, label %212, label %1616, !dbg !7

212:                                              ; preds = %203
  %213 = shl nsw i64 %204, 7, !dbg !7
  %214 = lshr i64 %204, 12, !dbg !16
  %215 = add nuw nsw i64 %214, %16, !dbg !16
  %216 = lshr i64 %204, 5, !dbg !16
  %217 = and i64 %216, 127, !dbg !16
  %218 = shl i64 %215, 19, !dbg !16
  %219 = and i64 %218, 16252928, !dbg !16
  %220 = shl i64 %216, 19, !dbg !16
  %221 = shl i64 %204, 14, !dbg !16
  %222 = sub i64 %221, %220, !dbg !16
  %223 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %219, !dbg !16
  %224 = getelementptr [2 x i8], ptr addrspace(1) %223, i64 %217, !dbg !16
  %225 = getelementptr [2 x i8], ptr addrspace(1) %224, i64 %222, !dbg !16
  %address.thread599 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local600 = zext nneg i32 %address.thread599 to i64, !dbg !8
  %address.local601 = shl nuw nsw i64 %address.local600, 3, !dbg !8
  %address.local602 = and i64 %address.local601, 1536, !dbg !8
  %address.local603 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local602, !dbg !8
  %address.local604 = ptrtoint ptr addrspace(3) %address.local603 to i64, !dbg !8
  %address.local605 = trunc i64 %address.local604 to i32, !dbg !8
  %226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local605), !dbg !16
  %227 = zext i32 %226 to i64, !dbg !16
  %228 = inttoptr i64 %227 to ptr addrspace(3), !dbg !16
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
  %address.local616 = getelementptr [2 x i8], ptr addrspace(1) %225, i64 %address.local615, !dbg !16
  %229 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %230 = ptrtoint ptr addrspace(1) %address.local616 to i64, !dbg !16
  %231 = sub i64 %230, %229, !dbg !16
  %dma.buffer.offset1756 = trunc i64 %231 to i32, !dbg !16
  %232 = trunc i64 %229 to i32, !dbg !16
  %233 = insertelement <4 x i32> poison, i32 %232, i64 0, !dbg !16
  %234 = lshr i64 %229, 32, !dbg !16
  %235 = trunc i64 %234 to i32, !dbg !16
  %236 = insertelement <4 x i32> %233, i32 %235, i64 1, !dbg !16
  %237 = insertelement <4 x i32> %236, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1757 = insertelement <4 x i32> %237, i32 131072, i64 3, !dbg !16
  %238 = ptrtoint ptr addrspace(3) %228 to i32, !dbg !16
  %dma.buffer.lds1758 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %238), !dbg !16
  %239 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1758, i32 %dma.buffer.offset1756, <4 x i32> %dma.buffer.rsrc1757) #10, !dbg !16
  %address.local617 = or disjoint i64 %address.local600, 256, !dbg !8
  %address.local618 = shl nuw nsw i64 %address.local617, 3, !dbg !8
  %address.local619 = and i64 %address.local618, 3584, !dbg !8
  %address.local620 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local619, !dbg !8
  %address.local621 = ptrtoint ptr addrspace(3) %address.local620 to i64, !dbg !8
  %address.local622 = trunc i64 %address.local621 to i32, !dbg !8
  %240 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local622), !dbg !16
  %241 = zext i32 %240 to i64, !dbg !16
  %242 = inttoptr i64 %241 to ptr addrspace(3), !dbg !16
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
  %address.local633 = getelementptr [2 x i8], ptr addrspace(1) %225, i64 %address.local632, !dbg !16
  %243 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %244 = ptrtoint ptr addrspace(1) %address.local633 to i64, !dbg !16
  %245 = sub i64 %244, %243, !dbg !16
  %dma.buffer.offset1753 = trunc i64 %245 to i32, !dbg !16
  %246 = trunc i64 %243 to i32, !dbg !16
  %247 = insertelement <4 x i32> poison, i32 %246, i64 0, !dbg !16
  %248 = lshr i64 %243, 32, !dbg !16
  %249 = trunc i64 %248 to i32, !dbg !16
  %250 = insertelement <4 x i32> %247, i32 %249, i64 1, !dbg !16
  %251 = insertelement <4 x i32> %250, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1754 = insertelement <4 x i32> %251, i32 131072, i64 3, !dbg !16
  %252 = ptrtoint ptr addrspace(3) %242 to i32, !dbg !16
  %dma.buffer.lds1755 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %252), !dbg !16
  %253 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1755, i32 %dma.buffer.offset1753, <4 x i32> %dma.buffer.rsrc1754) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %254 = add nuw i64 %204, 1024, !dbg !16
  %255 = icmp slt i64 %254, 0, !dbg !16
  %256 = sub nuw i64 -1025, %204, !dbg !16
  %257 = select i1 %255, i64 %256, i64 %254, !dbg !16
  %258 = sdiv i64 %257, 4096, !dbg !16
  %.lobit37 = ashr i64 %254, 63, !dbg !16
  %259 = xor i64 %258, %.lobit37, !dbg !16
  %260 = add nsw i64 %259, %16, !dbg !16
  %261 = add nsw i64 %216, %194, !dbg !16
  %262 = add nsw i64 %261, 32, !dbg !16
  %263 = icmp slt i64 %261, -32, !dbg !16
  %264 = sub nsw i64 -33, %261, !dbg !16
  %265 = select i1 %263, i64 %264, i64 %262, !dbg !16
  %266 = sdiv i64 %265, 128, !dbg !16
  %267 = sext i1 %263 to i64, !dbg !16
  %268 = xor i64 %266, %267, !dbg !16
  %269 = shl nsw i64 %268, 7, !dbg !16
  %270 = sub i64 %194, %269, !dbg !16
  %271 = shl i64 %260, 19, !dbg !16
  %272 = and i64 %271, 16252928, !dbg !16
  %273 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %272, !dbg !16
  %274 = getelementptr [2 x i8], ptr addrspace(1) %273, i64 %216, !dbg !16
  %275 = getelementptr [2 x i8], ptr addrspace(1) %274, i64 %270, !dbg !16
  %276 = getelementptr i8, ptr addrspace(1) %275, i64 64, !dbg !16
  %277 = getelementptr [2 x i8], ptr addrspace(1) %276, i64 %222, !dbg !16
  %address.local634 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local602, !dbg !8
  %address.local635 = ptrtoint ptr addrspace(3) %address.local634 to i64, !dbg !8
  %address.local636 = trunc i64 %address.local635 to i32, !dbg !8
  %278 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local636), !dbg !16
  %279 = zext i32 %278 to i64, !dbg !16
  %280 = inttoptr i64 %279 to ptr addrspace(3), !dbg !16
  %address.local637 = getelementptr [2 x i8], ptr addrspace(1) %277, i64 %address.local615, !dbg !16
  %281 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %282 = ptrtoint ptr addrspace(1) %address.local637 to i64, !dbg !16
  %283 = sub i64 %282, %281, !dbg !16
  %dma.buffer.offset1750 = trunc i64 %283 to i32, !dbg !16
  %284 = trunc i64 %281 to i32, !dbg !16
  %285 = insertelement <4 x i32> poison, i32 %284, i64 0, !dbg !16
  %286 = lshr i64 %281, 32, !dbg !16
  %287 = trunc i64 %286 to i32, !dbg !16
  %288 = insertelement <4 x i32> %285, i32 %287, i64 1, !dbg !16
  %289 = insertelement <4 x i32> %288, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1751 = insertelement <4 x i32> %289, i32 131072, i64 3, !dbg !16
  %290 = ptrtoint ptr addrspace(3) %280 to i32, !dbg !16
  %dma.buffer.lds1752 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %290), !dbg !16
  %291 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1752, i32 %dma.buffer.offset1750, <4 x i32> %dma.buffer.rsrc1751) #10, !dbg !16
  %address.local638 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local619, !dbg !8
  %address.local639 = ptrtoint ptr addrspace(3) %address.local638 to i64, !dbg !8
  %address.local640 = trunc i64 %address.local639 to i32, !dbg !8
  %292 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local640), !dbg !16
  %293 = zext i32 %292 to i64, !dbg !16
  %294 = inttoptr i64 %293 to ptr addrspace(3), !dbg !16
  %address.local641 = getelementptr [2 x i8], ptr addrspace(1) %277, i64 %address.local632, !dbg !16
  %295 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %296 = ptrtoint ptr addrspace(1) %address.local641 to i64, !dbg !16
  %297 = sub i64 %296, %295, !dbg !16
  %dma.buffer.offset1747 = trunc i64 %297 to i32, !dbg !16
  %298 = trunc i64 %295 to i32, !dbg !16
  %299 = insertelement <4 x i32> poison, i32 %298, i64 0, !dbg !16
  %300 = lshr i64 %295, 32, !dbg !16
  %301 = trunc i64 %300 to i32, !dbg !16
  %302 = insertelement <4 x i32> %299, i32 %301, i64 1, !dbg !16
  %303 = insertelement <4 x i32> %302, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1748 = insertelement <4 x i32> %303, i32 131072, i64 3, !dbg !16
  %304 = ptrtoint ptr addrspace(3) %294 to i32, !dbg !16
  %dma.buffer.lds1749 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %304), !dbg !16
  %305 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1749, i32 %dma.buffer.offset1747, <4 x i32> %dma.buffer.rsrc1748) #10, !dbg !16
  %address.local643 = lshr i64 %address.local600, 2, !dbg !17
  %address.local646 = shl nuw nsw i64 %address.local600, 1, !dbg !17
  %address.local647 = and i64 %address.local646, 24, !dbg !17
  %address.local649 = shl i64 %address.local600, 62, !dbg !17
  %address.local650 = ashr exact i64 %address.local649, 57, !dbg !17
  %address.local651 = and i64 %address.local650, -64, !dbg !17
  %address.local653 = lshr i64 %address.local600, 2, !dbg !11
  %address.local654 = and i64 %address.local653, 12, !dbg !11
  %306 = and i64 %address.local600, 32, !dbg !11
  %.not1712 = icmp eq i64 %306, 0, !dbg !11
  %address.local658 = select i1 %.not1712, i64 0, i64 1016, !dbg !11
  %.idx = mul nuw nsw i64 %address.local600, 320, !dbg !17
  %307 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !17
  %.idx1713 = mul nsw i64 %address.local643, -1280, !dbg !17
  %308 = getelementptr i8, ptr addrspace(3) %307, i64 %.idx1713, !dbg !17
  %309 = getelementptr [2 x i8], ptr addrspace(3) %308, i64 %address.local647, !dbg !17
  %310 = getelementptr [2 x i8], ptr addrspace(3) %309, i64 %address.local651, !dbg !17
  %311 = getelementptr [2 x i8], ptr addrspace(3) %310, i64 %address.local654, !dbg !17
  %address.local660 = getelementptr [2 x i8], ptr addrspace(3) %311, i64 %address.local658, !dbg !17
  %312 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !17
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %312, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1592 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4096, !dbg !17
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !17
  %address.local671 = shl nuw nsw i32 %address.thread599, 1, !dbg !17
  %address.local672 = and i32 %address.local671, 24, !dbg !17
  %address.local673 = shl nuw nsw i32 %address.thread599, 8, !dbg !17
  %address.local674 = and i32 %address.local673, 768, !dbg !17
  %address.local675 = or disjoint i32 %address.local672, %address.local674, !dbg !17
  %address.local676 = shl nuw nsw i32 %address.thread599, 5, !dbg !17
  %address.local677 = and i32 %address.local676, 96, !dbg !17
  %address.local678 = or disjoint i32 %address.local675, %address.local677, !dbg !17
  %address.local679 = zext nneg i32 %address.local678 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %202) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local680 = lshr i32 %address.thread599, 1, !dbg !17
  %address.local681 = and i32 %address.local680, 1, !dbg !17
  %address.local682 = zext nneg i32 %address.local681 to i64, !dbg !17
  %address.local683 = add nsw i64 %address.base, %address.local682, !dbg !17
  %address.local684 = shl i64 %address.local683, 6, !dbg !17
  %address.local685 = sub i64 %address.local679, %address.local684, !dbg !17
  %315 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local685, !dbg !17
  %316 = getelementptr [2 x i8], ptr addrspace(3) %315, i64 %address.local654, !dbg !17
  %317 = getelementptr [2 x i8], ptr addrspace(3) %316, i64 %address.local658, !dbg !17
  %address.local689 = getelementptr i8, ptr addrspace(3) %317, i64 64, !dbg !17
  %318 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !17
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %314, <4 x float> %313) #9, !dbg !17
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %318, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %317, i64 4160, !dbg !17
  %321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !17
  %lds.const.gep1594 = getelementptr i8, ptr addrspace(3) %address.local660, i64 128, !dbg !17
  %322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !17
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %321, <4 x float> %320) #9, !dbg !17
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %322, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1595 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4224, !dbg !17
  %325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !17
  %lds.const.gep1596 = getelementptr i8, ptr addrspace(3) %317, i64 192, !dbg !17
  %326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !17
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %325, <4 x float> %324) #9, !dbg !17
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %326, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1597 = getelementptr i8, ptr addrspace(3) %317, i64 4288, !dbg !17
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !17
  %lds.const.gep1598 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1024, !dbg !17
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !17
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %329, <4 x float> %328) #9, !dbg !17
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %330, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1599 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5120, !dbg !17
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !17
  %lds.const.gep1600 = getelementptr i8, ptr addrspace(3) %317, i64 1088, !dbg !17
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !17
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %333, <4 x float> %332) #9, !dbg !17
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %334, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1601 = getelementptr i8, ptr addrspace(3) %317, i64 5184, !dbg !17
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !17
  %lds.const.gep1602 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1152, !dbg !17
  %338 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !17
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %337, <4 x float> %336) #9, !dbg !17
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %338, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1603 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5248, !dbg !17
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !17
  %lds.const.gep1604 = getelementptr i8, ptr addrspace(3) %317, i64 1216, !dbg !17
  %342 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !17
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %341, <4 x float> %340) #9, !dbg !17
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %342, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1605 = getelementptr i8, ptr addrspace(3) %317, i64 5312, !dbg !17
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !17
  %346 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !17
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %345, <4 x float> %344) #9, !dbg !17
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %346, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1606 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4096, !dbg !17
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1606, align 8, !dbg !17
  %350 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !17
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %349, <4 x float> %348) #9, !dbg !17
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %350, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1607 = getelementptr i8, ptr addrspace(3) %317, i64 4160, !dbg !17
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1607, align 8, !dbg !17
  %lds.const.gep1608 = getelementptr i8, ptr addrspace(3) %address.local660, i64 128, !dbg !17
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1608, align 8, !dbg !17
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %353, <4 x float> %352) #9, !dbg !17
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %354, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1609 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4224, !dbg !17
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1609, align 8, !dbg !17
  %lds.const.gep1610 = getelementptr i8, ptr addrspace(3) %317, i64 192, !dbg !17
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1610, align 8, !dbg !17
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %357, <4 x float> %356) #9, !dbg !17
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %358, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1611 = getelementptr i8, ptr addrspace(3) %317, i64 4288, !dbg !17
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1611, align 8, !dbg !17
  %lds.const.gep1612 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1024, !dbg !17
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1612, align 8, !dbg !17
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %361, <4 x float> %360) #9, !dbg !17
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %362, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1613 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5120, !dbg !17
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1613, align 8, !dbg !17
  %lds.const.gep1614 = getelementptr i8, ptr addrspace(3) %317, i64 1088, !dbg !17
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1614, align 8, !dbg !17
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %365, <4 x float> %364) #9, !dbg !17
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %366, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1615 = getelementptr i8, ptr addrspace(3) %317, i64 5184, !dbg !17
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1615, align 8, !dbg !17
  %lds.const.gep1616 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1152, !dbg !17
  %370 = load <4 x half>, ptr addrspace(3) %lds.const.gep1616, align 8, !dbg !17
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %369, <4 x float> %368) #9, !dbg !17
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %370, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1617 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5248, !dbg !17
  %373 = load <4 x half>, ptr addrspace(3) %lds.const.gep1617, align 8, !dbg !17
  %lds.const.gep1618 = getelementptr i8, ptr addrspace(3) %317, i64 1216, !dbg !17
  %374 = load <4 x half>, ptr addrspace(3) %lds.const.gep1618, align 8, !dbg !17
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %373, <4 x float> %372) #9, !dbg !17
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %374, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1619 = getelementptr i8, ptr addrspace(3) %317, i64 5312, !dbg !17
  %377 = load <4 x half>, ptr addrspace(3) %lds.const.gep1619, align 8, !dbg !17
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %377, <4 x float> %376) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %379 = add nuw i64 %204, 2048, !dbg !16
  %380 = icmp slt i64 %379, 0, !dbg !16
  %381 = sub nuw i64 -2049, %204, !dbg !16
  %382 = select i1 %380, i64 %381, i64 %379, !dbg !16
  %383 = sdiv i64 %382, 4096, !dbg !16
  %.lobit38 = ashr i64 %379, 63, !dbg !16
  %384 = xor i64 %383, %.lobit38, !dbg !16
  %385 = add nsw i64 %384, %16, !dbg !16
  %386 = add nsw i64 %261, 64, !dbg !16
  %387 = icmp slt i64 %261, -64, !dbg !16
  %388 = sub nsw i64 -65, %261, !dbg !16
  %389 = select i1 %387, i64 %388, i64 %386, !dbg !16
  %390 = sdiv i64 %389, 128, !dbg !16
  %391 = sext i1 %387 to i64, !dbg !16
  %392 = xor i64 %390, %391, !dbg !16
  %393 = shl nsw i64 %392, 7, !dbg !16
  %394 = sub i64 %194, %393, !dbg !16
  %395 = shl i64 %385, 19, !dbg !16
  %396 = and i64 %395, 16252928, !dbg !16
  %397 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %396, !dbg !16
  %398 = getelementptr [2 x i8], ptr addrspace(1) %397, i64 %216, !dbg !16
  %399 = getelementptr [2 x i8], ptr addrspace(1) %398, i64 %394, !dbg !16
  %400 = getelementptr i8, ptr addrspace(1) %399, i64 128, !dbg !16
  %401 = getelementptr [2 x i8], ptr addrspace(1) %400, i64 %222, !dbg !16
  %402 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local605), !dbg !16
  %403 = zext i32 %402 to i64, !dbg !16
  %404 = inttoptr i64 %403 to ptr addrspace(3), !dbg !16
  %address.local998 = getelementptr [2 x i8], ptr addrspace(1) %401, i64 %address.local615, !dbg !16
  %405 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %406 = ptrtoint ptr addrspace(1) %address.local998 to i64, !dbg !16
  %407 = sub i64 %406, %405, !dbg !16
  %dma.buffer.offset1744 = trunc i64 %407 to i32, !dbg !16
  %408 = trunc i64 %405 to i32, !dbg !16
  %409 = insertelement <4 x i32> poison, i32 %408, i64 0, !dbg !16
  %410 = lshr i64 %405, 32, !dbg !16
  %411 = trunc i64 %410 to i32, !dbg !16
  %412 = insertelement <4 x i32> %409, i32 %411, i64 1, !dbg !16
  %413 = insertelement <4 x i32> %412, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1745 = insertelement <4 x i32> %413, i32 131072, i64 3, !dbg !16
  %414 = ptrtoint ptr addrspace(3) %404 to i32, !dbg !16
  %dma.buffer.lds1746 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %414), !dbg !16
  %415 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1746, i32 %dma.buffer.offset1744, <4 x i32> %dma.buffer.rsrc1745) #10, !dbg !16
  %416 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local622), !dbg !16
  %417 = zext i32 %416 to i64, !dbg !16
  %418 = inttoptr i64 %417 to ptr addrspace(3), !dbg !16
  %address.local999 = getelementptr [2 x i8], ptr addrspace(1) %401, i64 %address.local632, !dbg !16
  %419 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %420 = ptrtoint ptr addrspace(1) %address.local999 to i64, !dbg !16
  %421 = sub i64 %420, %419, !dbg !16
  %dma.buffer.offset1741 = trunc i64 %421 to i32, !dbg !16
  %422 = trunc i64 %419 to i32, !dbg !16
  %423 = insertelement <4 x i32> poison, i32 %422, i64 0, !dbg !16
  %424 = lshr i64 %419, 32, !dbg !16
  %425 = trunc i64 %424 to i32, !dbg !16
  %426 = insertelement <4 x i32> %423, i32 %425, i64 1, !dbg !16
  %427 = insertelement <4 x i32> %426, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1742 = insertelement <4 x i32> %427, i32 131072, i64 3, !dbg !16
  %428 = ptrtoint ptr addrspace(3) %418 to i32, !dbg !16
  %dma.buffer.lds1743 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %428), !dbg !16
  %429 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1743, i32 %dma.buffer.offset1741, <4 x i32> %dma.buffer.rsrc1742) #10, !dbg !16
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8192, !dbg !17
  %430 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !17
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %430, <4 x float> %319) #9, !dbg !17
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12288, !dbg !17
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !17
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %317, i64 8256, !dbg !17
  %433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !17
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %432, <4 x float> %431) #9, !dbg !17
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %433, <4 x float> %323) #9, !dbg !17
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %317, i64 12352, !dbg !17
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !17
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8320, !dbg !17
  %437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !17
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %436, <4 x float> %435) #9, !dbg !17
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %437, <4 x float> %327) #9, !dbg !17
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12416, !dbg !17
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !17
  %lds.const.gep1626 = getelementptr i8, ptr addrspace(3) %317, i64 8384, !dbg !17
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !17
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %440, <4 x float> %439) #9, !dbg !17
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %441, <4 x float> %331) #9, !dbg !17
  %lds.const.gep1627 = getelementptr i8, ptr addrspace(3) %317, i64 12480, !dbg !17
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !17
  %lds.const.gep1628 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9216, !dbg !17
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !17
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %444, <4 x float> %443) #9, !dbg !17
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %445, <4 x float> %335) #9, !dbg !17
  %lds.const.gep1629 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13312, !dbg !17
  %448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !17
  %lds.const.gep1630 = getelementptr i8, ptr addrspace(3) %317, i64 9280, !dbg !17
  %449 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !17
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %448, <4 x float> %447) #9, !dbg !17
  %451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %449, <4 x float> %339) #9, !dbg !17
  %lds.const.gep1631 = getelementptr i8, ptr addrspace(3) %317, i64 13376, !dbg !17
  %452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !17
  %lds.const.gep1632 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9344, !dbg !17
  %453 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !17
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %452, <4 x float> %451) #9, !dbg !17
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %453, <4 x float> %343) #9, !dbg !17
  %lds.const.gep1633 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13440, !dbg !17
  %456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !17
  %lds.const.gep1634 = getelementptr i8, ptr addrspace(3) %317, i64 9408, !dbg !17
  %457 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !17
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %456, <4 x float> %455) #9, !dbg !17
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %136, <4 x half> %457, <4 x float> %347) #9, !dbg !17
  %lds.const.gep1635 = getelementptr i8, ptr addrspace(3) %317, i64 13504, !dbg !17
  %460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !17
  %lds.const.gep1636 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8192, !dbg !17
  %461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1636, align 8, !dbg !17
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %138, <4 x half> %460, <4 x float> %459) #9, !dbg !17
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %461, <4 x float> %351) #9, !dbg !17
  %lds.const.gep1637 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12288, !dbg !17
  %464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1637, align 8, !dbg !17
  %lds.const.gep1638 = getelementptr i8, ptr addrspace(3) %317, i64 8256, !dbg !17
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1638, align 8, !dbg !17
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %464, <4 x float> %463) #9, !dbg !17
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %465, <4 x float> %355) #9, !dbg !17
  %lds.const.gep1639 = getelementptr i8, ptr addrspace(3) %317, i64 12352, !dbg !17
  %468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1639, align 8, !dbg !17
  %lds.const.gep1640 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8320, !dbg !17
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1640, align 8, !dbg !17
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %468, <4 x float> %467) #9, !dbg !17
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %469, <4 x float> %359) #9, !dbg !17
  %lds.const.gep1641 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12416, !dbg !17
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1641, align 8, !dbg !17
  %lds.const.gep1642 = getelementptr i8, ptr addrspace(3) %317, i64 8384, !dbg !17
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1642, align 8, !dbg !17
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %472, <4 x float> %471) #9, !dbg !17
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %473, <4 x float> %363) #9, !dbg !17
  %lds.const.gep1643 = getelementptr i8, ptr addrspace(3) %317, i64 12480, !dbg !17
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1643, align 8, !dbg !17
  %lds.const.gep1644 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9216, !dbg !17
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1644, align 8, !dbg !17
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %476, <4 x float> %475) #9, !dbg !17
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %477, <4 x float> %367) #9, !dbg !17
  %lds.const.gep1645 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13312, !dbg !17
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1645, align 8, !dbg !17
  %lds.const.gep1646 = getelementptr i8, ptr addrspace(3) %317, i64 9280, !dbg !17
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1646, align 8, !dbg !17
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %480, <4 x float> %479) #9, !dbg !17
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %481, <4 x float> %371) #9, !dbg !17
  %lds.const.gep1647 = getelementptr i8, ptr addrspace(3) %317, i64 13376, !dbg !17
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1647, align 8, !dbg !17
  %lds.const.gep1648 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9344, !dbg !17
  %485 = load <4 x half>, ptr addrspace(3) %lds.const.gep1648, align 8, !dbg !17
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %484, <4 x float> %483) #9, !dbg !17
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %485, <4 x float> %375) #9, !dbg !17
  %lds.const.gep1649 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13440, !dbg !17
  %488 = load <4 x half>, ptr addrspace(3) %lds.const.gep1649, align 8, !dbg !17
  %lds.const.gep1650 = getelementptr i8, ptr addrspace(3) %317, i64 9408, !dbg !17
  %489 = load <4 x half>, ptr addrspace(3) %lds.const.gep1650, align 8, !dbg !17
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %488, <4 x float> %487) #9, !dbg !17
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %489, <4 x float> %378) #9, !dbg !17
  %lds.const.gep1651 = getelementptr i8, ptr addrspace(3) %317, i64 13504, !dbg !17
  %492 = load <4 x half>, ptr addrspace(3) %lds.const.gep1651, align 8, !dbg !17
  %493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %492, <4 x float> %491) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %494 = add nuw i64 %204, 3072, !dbg !16
  %495 = icmp slt i64 %494, 0, !dbg !16
  %496 = sub nuw i64 -3073, %204, !dbg !16
  %497 = select i1 %495, i64 %496, i64 %494, !dbg !16
  %498 = sdiv i64 %497, 4096, !dbg !16
  %.lobit39 = ashr i64 %494, 63, !dbg !16
  %499 = xor i64 %498, %.lobit39, !dbg !16
  %500 = add nsw i64 %499, %16, !dbg !16
  %501 = add nsw i64 %261, 96, !dbg !16
  %502 = icmp slt i64 %261, -96, !dbg !16
  %503 = sub nsw i64 -97, %261, !dbg !16
  %504 = select i1 %502, i64 %503, i64 %501, !dbg !16
  %505 = sdiv i64 %504, 128, !dbg !16
  %506 = sext i1 %502 to i64, !dbg !16
  %507 = xor i64 %505, %506, !dbg !16
  %508 = shl nsw i64 %507, 7, !dbg !16
  %509 = sub i64 %194, %508, !dbg !16
  %510 = shl i64 %500, 19, !dbg !16
  %511 = and i64 %510, 16252928, !dbg !16
  %512 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %511, !dbg !16
  %513 = getelementptr [2 x i8], ptr addrspace(1) %512, i64 %216, !dbg !16
  %514 = getelementptr [2 x i8], ptr addrspace(1) %513, i64 %509, !dbg !16
  %515 = getelementptr i8, ptr addrspace(1) %514, i64 192, !dbg !16
  %516 = getelementptr [2 x i8], ptr addrspace(1) %515, i64 %222, !dbg !16
  %517 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local636), !dbg !16
  %518 = zext i32 %517 to i64, !dbg !16
  %519 = inttoptr i64 %518 to ptr addrspace(3), !dbg !16
  %address.local1016 = getelementptr [2 x i8], ptr addrspace(1) %516, i64 %address.local615, !dbg !16
  %520 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %521 = ptrtoint ptr addrspace(1) %address.local1016 to i64, !dbg !16
  %522 = sub i64 %521, %520, !dbg !16
  %dma.buffer.offset1738 = trunc i64 %522 to i32, !dbg !16
  %523 = trunc i64 %520 to i32, !dbg !16
  %524 = insertelement <4 x i32> poison, i32 %523, i64 0, !dbg !16
  %525 = lshr i64 %520, 32, !dbg !16
  %526 = trunc i64 %525 to i32, !dbg !16
  %527 = insertelement <4 x i32> %524, i32 %526, i64 1, !dbg !16
  %528 = insertelement <4 x i32> %527, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1739 = insertelement <4 x i32> %528, i32 131072, i64 3, !dbg !16
  %529 = ptrtoint ptr addrspace(3) %519 to i32, !dbg !16
  %dma.buffer.lds1740 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %529), !dbg !16
  %530 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1740, i32 %dma.buffer.offset1738, <4 x i32> %dma.buffer.rsrc1739) #10, !dbg !16
  %531 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local640), !dbg !16
  %532 = zext i32 %531 to i64, !dbg !16
  %533 = inttoptr i64 %532 to ptr addrspace(3), !dbg !16
  %address.local1017 = getelementptr [2 x i8], ptr addrspace(1) %516, i64 %address.local632, !dbg !16
  %534 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %535 = ptrtoint ptr addrspace(1) %address.local1017 to i64, !dbg !16
  %536 = sub i64 %535, %534, !dbg !16
  %dma.buffer.offset1735 = trunc i64 %536 to i32, !dbg !16
  %537 = trunc i64 %534 to i32, !dbg !16
  %538 = insertelement <4 x i32> poison, i32 %537, i64 0, !dbg !16
  %539 = lshr i64 %534, 32, !dbg !16
  %540 = trunc i64 %539 to i32, !dbg !16
  %541 = insertelement <4 x i32> %538, i32 %540, i64 1, !dbg !16
  %542 = insertelement <4 x i32> %541, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1736 = insertelement <4 x i32> %542, i32 131072, i64 3, !dbg !16
  %543 = ptrtoint ptr addrspace(3) %533 to i32, !dbg !16
  %dma.buffer.lds1737 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %543), !dbg !16
  %544 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1737, i32 %dma.buffer.offset1735, <4 x i32> %dma.buffer.rsrc1736) #10, !dbg !16
  %545 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !17
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %545, <4 x float> %434) #9, !dbg !17
  %lds.const.gep1652 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4096, !dbg !17
  %547 = load <4 x half>, ptr addrspace(3) %lds.const.gep1652, align 8, !dbg !17
  %548 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !17
  %549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %547, <4 x float> %546) #9, !dbg !17
  %550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %548, <4 x float> %438) #9, !dbg !17
  %lds.const.gep1653 = getelementptr i8, ptr addrspace(3) %317, i64 4160, !dbg !17
  %551 = load <4 x half>, ptr addrspace(3) %lds.const.gep1653, align 8, !dbg !17
  %lds.const.gep1654 = getelementptr i8, ptr addrspace(3) %address.local660, i64 128, !dbg !17
  %552 = load <4 x half>, ptr addrspace(3) %lds.const.gep1654, align 8, !dbg !17
  %553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %551, <4 x float> %550) #9, !dbg !17
  %554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %552, <4 x float> %442) #9, !dbg !17
  %lds.const.gep1655 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4224, !dbg !17
  %555 = load <4 x half>, ptr addrspace(3) %lds.const.gep1655, align 8, !dbg !17
  %lds.const.gep1656 = getelementptr i8, ptr addrspace(3) %317, i64 192, !dbg !17
  %556 = load <4 x half>, ptr addrspace(3) %lds.const.gep1656, align 8, !dbg !17
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %555, <4 x float> %554) #9, !dbg !17
  %558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %556, <4 x float> %446) #9, !dbg !17
  %lds.const.gep1657 = getelementptr i8, ptr addrspace(3) %317, i64 4288, !dbg !17
  %559 = load <4 x half>, ptr addrspace(3) %lds.const.gep1657, align 8, !dbg !17
  %lds.const.gep1658 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1024, !dbg !17
  %560 = load <4 x half>, ptr addrspace(3) %lds.const.gep1658, align 8, !dbg !17
  %561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %559, <4 x float> %558) #9, !dbg !17
  %562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %560, <4 x float> %450) #9, !dbg !17
  %lds.const.gep1659 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5120, !dbg !17
  %563 = load <4 x half>, ptr addrspace(3) %lds.const.gep1659, align 8, !dbg !17
  %lds.const.gep1660 = getelementptr i8, ptr addrspace(3) %317, i64 1088, !dbg !17
  %564 = load <4 x half>, ptr addrspace(3) %lds.const.gep1660, align 8, !dbg !17
  %565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %563, <4 x float> %562) #9, !dbg !17
  %566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %564, <4 x float> %454) #9, !dbg !17
  %lds.const.gep1661 = getelementptr i8, ptr addrspace(3) %317, i64 5184, !dbg !17
  %567 = load <4 x half>, ptr addrspace(3) %lds.const.gep1661, align 8, !dbg !17
  %lds.const.gep1662 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1152, !dbg !17
  %568 = load <4 x half>, ptr addrspace(3) %lds.const.gep1662, align 8, !dbg !17
  %569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %567, <4 x float> %566) #9, !dbg !17
  %570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %568, <4 x float> %458) #9, !dbg !17
  %lds.const.gep1663 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5248, !dbg !17
  %571 = load <4 x half>, ptr addrspace(3) %lds.const.gep1663, align 8, !dbg !17
  %lds.const.gep1664 = getelementptr i8, ptr addrspace(3) %317, i64 1216, !dbg !17
  %572 = load <4 x half>, ptr addrspace(3) %lds.const.gep1664, align 8, !dbg !17
  %573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %571, <4 x float> %570) #9, !dbg !17
  %574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %174, <4 x half> %572, <4 x float> %462) #9, !dbg !17
  %lds.const.gep1665 = getelementptr i8, ptr addrspace(3) %317, i64 5312, !dbg !17
  %575 = load <4 x half>, ptr addrspace(3) %lds.const.gep1665, align 8, !dbg !17
  %576 = load <4 x half>, ptr addrspace(3) %address.local660, align 8, !dbg !17
  %577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %176, <4 x half> %575, <4 x float> %574) #9, !dbg !17
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %576, <4 x float> %466) #9, !dbg !17
  %lds.const.gep1666 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4096, !dbg !17
  %579 = load <4 x half>, ptr addrspace(3) %lds.const.gep1666, align 8, !dbg !17
  %580 = load <4 x half>, ptr addrspace(3) %address.local689, align 8, !dbg !17
  %581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %579, <4 x float> %578) #9, !dbg !17
  %582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %580, <4 x float> %470) #9, !dbg !17
  %lds.const.gep1667 = getelementptr i8, ptr addrspace(3) %317, i64 4160, !dbg !17
  %583 = load <4 x half>, ptr addrspace(3) %lds.const.gep1667, align 8, !dbg !17
  %lds.const.gep1668 = getelementptr i8, ptr addrspace(3) %address.local660, i64 128, !dbg !17
  %584 = load <4 x half>, ptr addrspace(3) %lds.const.gep1668, align 8, !dbg !17
  %585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %583, <4 x float> %582) #9, !dbg !17
  %586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %584, <4 x float> %474) #9, !dbg !17
  %lds.const.gep1669 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4224, !dbg !17
  %587 = load <4 x half>, ptr addrspace(3) %lds.const.gep1669, align 8, !dbg !17
  %lds.const.gep1670 = getelementptr i8, ptr addrspace(3) %317, i64 192, !dbg !17
  %588 = load <4 x half>, ptr addrspace(3) %lds.const.gep1670, align 8, !dbg !17
  %589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %587, <4 x float> %586) #9, !dbg !17
  %590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %588, <4 x float> %478) #9, !dbg !17
  %lds.const.gep1671 = getelementptr i8, ptr addrspace(3) %317, i64 4288, !dbg !17
  %591 = load <4 x half>, ptr addrspace(3) %lds.const.gep1671, align 8, !dbg !17
  %lds.const.gep1672 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1024, !dbg !17
  %592 = load <4 x half>, ptr addrspace(3) %lds.const.gep1672, align 8, !dbg !17
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %591, <4 x float> %590) #9, !dbg !17
  %594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %592, <4 x float> %482) #9, !dbg !17
  %lds.const.gep1673 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5120, !dbg !17
  %595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1673, align 8, !dbg !17
  %lds.const.gep1674 = getelementptr i8, ptr addrspace(3) %317, i64 1088, !dbg !17
  %596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1674, align 8, !dbg !17
  %597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %595, <4 x float> %594) #9, !dbg !17
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %596, <4 x float> %486) #9, !dbg !17
  %lds.const.gep1675 = getelementptr i8, ptr addrspace(3) %317, i64 5184, !dbg !17
  %599 = load <4 x half>, ptr addrspace(3) %lds.const.gep1675, align 8, !dbg !17
  %lds.const.gep1676 = getelementptr i8, ptr addrspace(3) %address.local660, i64 1152, !dbg !17
  %600 = load <4 x half>, ptr addrspace(3) %lds.const.gep1676, align 8, !dbg !17
  %601 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %599, <4 x float> %598) #9, !dbg !17
  %602 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %600, <4 x float> %490) #9, !dbg !17
  %lds.const.gep1677 = getelementptr i8, ptr addrspace(3) %address.local660, i64 5248, !dbg !17
  %603 = load <4 x half>, ptr addrspace(3) %lds.const.gep1677, align 8, !dbg !17
  %lds.const.gep1678 = getelementptr i8, ptr addrspace(3) %317, i64 1216, !dbg !17
  %604 = load <4 x half>, ptr addrspace(3) %lds.const.gep1678, align 8, !dbg !17
  %605 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %603, <4 x float> %602) #9, !dbg !17
  %606 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %604, <4 x float> %493) #9, !dbg !17
  %lds.const.gep1679 = getelementptr i8, ptr addrspace(3) %317, i64 5312, !dbg !17
  %607 = load <4 x half>, ptr addrspace(3) %lds.const.gep1679, align 8, !dbg !17
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %607, <4 x float> %606) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %609 = add nuw nsw i64 %216, %16, !dbg !18
  %610 = shl i64 %609, 19, !dbg !18
  %611 = and i64 %610, 16252928, !dbg !18
  %612 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %611, !dbg !18
  %613 = getelementptr [2 x i8], ptr addrspace(1) %612, i64 %222, !dbg !18
  %614 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local605), !dbg !18
  %615 = zext i32 %614 to i64, !dbg !18
  %616 = inttoptr i64 %615 to ptr addrspace(3), !dbg !18
  %address.local1018 = shl nuw nsw i64 %address.local600, 5, !dbg !18
  %address.local1019 = and i64 %address.local1018, 8064, !dbg !18
  %address.local1020 = shl nuw nsw i64 %address.local600, 3, !dbg !18
  %address.local1021 = and i64 %address.local1020, 24, !dbg !18
  %617 = getelementptr [2 x i8], ptr addrspace(1) %613, i64 %address.local1019, !dbg !18
  %address.local1023 = getelementptr [2 x i8], ptr addrspace(1) %617, i64 %address.local1021, !dbg !18
  %618 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %619 = ptrtoint ptr addrspace(1) %address.local1023 to i64, !dbg !18
  %620 = sub i64 %619, %618, !dbg !18
  %dma.buffer.offset1732 = trunc i64 %620 to i32, !dbg !18
  %621 = trunc i64 %618 to i32, !dbg !18
  %622 = insertelement <4 x i32> poison, i32 %621, i64 0, !dbg !18
  %623 = lshr i64 %618, 32, !dbg !18
  %624 = trunc i64 %623 to i32, !dbg !18
  %625 = insertelement <4 x i32> %622, i32 %624, i64 1, !dbg !18
  %626 = insertelement <4 x i32> %625, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1733 = insertelement <4 x i32> %626, i32 131072, i64 3, !dbg !18
  %627 = ptrtoint ptr addrspace(3) %616 to i32, !dbg !18
  %dma.buffer.lds1734 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %627), !dbg !18
  %628 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1734, i32 %dma.buffer.offset1732, <4 x i32> %dma.buffer.rsrc1733) #10, !dbg !18
  %629 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local622), !dbg !18
  %630 = zext i32 %629 to i64, !dbg !18
  %631 = inttoptr i64 %630 to ptr addrspace(3), !dbg !18
  %address.local1024 = shl nuw nsw i64 %address.local617, 5, !dbg !18
  %address.local1025 = and i64 %address.local1024, 16256, !dbg !18
  %address.local1026 = shl nuw nsw i64 %address.local600, 3, !dbg !18
  %address.local1027 = and i64 %address.local1026, 24, !dbg !18
  %632 = getelementptr [2 x i8], ptr addrspace(1) %613, i64 %address.local1025, !dbg !18
  %address.local1029 = getelementptr [2 x i8], ptr addrspace(1) %632, i64 %address.local1027, !dbg !18
  %633 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %634 = ptrtoint ptr addrspace(1) %address.local1029 to i64, !dbg !18
  %635 = sub i64 %634, %633, !dbg !18
  %dma.buffer.offset1729 = trunc i64 %635 to i32, !dbg !18
  %636 = trunc i64 %633 to i32, !dbg !18
  %637 = insertelement <4 x i32> poison, i32 %636, i64 0, !dbg !18
  %638 = lshr i64 %633, 32, !dbg !18
  %639 = trunc i64 %638 to i32, !dbg !18
  %640 = insertelement <4 x i32> %637, i32 %639, i64 1, !dbg !18
  %641 = insertelement <4 x i32> %640, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1730 = insertelement <4 x i32> %641, i32 131072, i64 3, !dbg !18
  %642 = ptrtoint ptr addrspace(3) %631 to i32, !dbg !18
  %dma.buffer.lds1731 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %642), !dbg !18
  %643 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1731, i32 %dma.buffer.offset1729, <4 x i32> %dma.buffer.rsrc1730) #10, !dbg !18
  %lds.const.gep1680 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8192, !dbg !17
  %644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1680, align 8, !dbg !17
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %644, <4 x float> %549) #9, !dbg !17
  %lds.const.gep1681 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12288, !dbg !17
  %646 = load <4 x half>, ptr addrspace(3) %lds.const.gep1681, align 8, !dbg !17
  %lds.const.gep1682 = getelementptr i8, ptr addrspace(3) %317, i64 8256, !dbg !17
  %647 = load <4 x half>, ptr addrspace(3) %lds.const.gep1682, align 8, !dbg !17
  %648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %646, <4 x float> %645) #9, !dbg !17
  %649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %647, <4 x float> %553) #9, !dbg !17
  %lds.const.gep1683 = getelementptr i8, ptr addrspace(3) %317, i64 12352, !dbg !17
  %650 = load <4 x half>, ptr addrspace(3) %lds.const.gep1683, align 8, !dbg !17
  %lds.const.gep1684 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8320, !dbg !17
  %651 = load <4 x half>, ptr addrspace(3) %lds.const.gep1684, align 8, !dbg !17
  %652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %650, <4 x float> %649) #9, !dbg !17
  %653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %651, <4 x float> %557) #9, !dbg !17
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12416, !dbg !17
  %654 = load <4 x half>, ptr addrspace(3) %lds.const.gep1685, align 8, !dbg !17
  %lds.const.gep1686 = getelementptr i8, ptr addrspace(3) %317, i64 8384, !dbg !17
  %655 = load <4 x half>, ptr addrspace(3) %lds.const.gep1686, align 8, !dbg !17
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %654, <4 x float> %653) #9, !dbg !17
  %657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %655, <4 x float> %561) #9, !dbg !17
  %lds.const.gep1687 = getelementptr i8, ptr addrspace(3) %317, i64 12480, !dbg !17
  %658 = load <4 x half>, ptr addrspace(3) %lds.const.gep1687, align 8, !dbg !17
  %lds.const.gep1688 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9216, !dbg !17
  %659 = load <4 x half>, ptr addrspace(3) %lds.const.gep1688, align 8, !dbg !17
  %660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %658, <4 x float> %657) #9, !dbg !17
  %661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %659, <4 x float> %565) #9, !dbg !17
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13312, !dbg !17
  %662 = load <4 x half>, ptr addrspace(3) %lds.const.gep1689, align 8, !dbg !17
  %lds.const.gep1690 = getelementptr i8, ptr addrspace(3) %317, i64 9280, !dbg !17
  %663 = load <4 x half>, ptr addrspace(3) %lds.const.gep1690, align 8, !dbg !17
  %664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %662, <4 x float> %661) #9, !dbg !17
  %665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %663, <4 x float> %569) #9, !dbg !17
  %lds.const.gep1691 = getelementptr i8, ptr addrspace(3) %317, i64 13376, !dbg !17
  %666 = load <4 x half>, ptr addrspace(3) %lds.const.gep1691, align 8, !dbg !17
  %lds.const.gep1692 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9344, !dbg !17
  %667 = load <4 x half>, ptr addrspace(3) %lds.const.gep1692, align 8, !dbg !17
  %668 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %666, <4 x float> %665) #9, !dbg !17
  %669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %667, <4 x float> %573) #9, !dbg !17
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13440, !dbg !17
  %670 = load <4 x half>, ptr addrspace(3) %lds.const.gep1693, align 8, !dbg !17
  %lds.const.gep1694 = getelementptr i8, ptr addrspace(3) %317, i64 9408, !dbg !17
  %671 = load <4 x half>, ptr addrspace(3) %lds.const.gep1694, align 8, !dbg !17
  %672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %670, <4 x float> %669) #9, !dbg !17
  %673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %182, <4 x half> %671, <4 x float> %577) #9, !dbg !17
  %lds.const.gep1695 = getelementptr i8, ptr addrspace(3) %317, i64 13504, !dbg !17
  %674 = load <4 x half>, ptr addrspace(3) %lds.const.gep1695, align 8, !dbg !17
  %lds.const.gep1696 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8192, !dbg !17
  %675 = load <4 x half>, ptr addrspace(3) %lds.const.gep1696, align 8, !dbg !17
  %676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %184, <4 x half> %674, <4 x float> %673) #9, !dbg !17
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %675, <4 x float> %581) #9, !dbg !17
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12288, !dbg !17
  %678 = load <4 x half>, ptr addrspace(3) %lds.const.gep1697, align 8, !dbg !17
  %lds.const.gep1698 = getelementptr i8, ptr addrspace(3) %317, i64 8256, !dbg !17
  %679 = load <4 x half>, ptr addrspace(3) %lds.const.gep1698, align 8, !dbg !17
  %680 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %678, <4 x float> %677) #9, !dbg !17
  %681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %679, <4 x float> %585) #9, !dbg !17
  %lds.const.gep1699 = getelementptr i8, ptr addrspace(3) %317, i64 12352, !dbg !17
  %682 = load <4 x half>, ptr addrspace(3) %lds.const.gep1699, align 8, !dbg !17
  %lds.const.gep1700 = getelementptr i8, ptr addrspace(3) %address.local660, i64 8320, !dbg !17
  %683 = load <4 x half>, ptr addrspace(3) %lds.const.gep1700, align 8, !dbg !17
  %684 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %682, <4 x float> %681) #9, !dbg !17
  %685 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %683, <4 x float> %589) #9, !dbg !17
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local660, i64 12416, !dbg !17
  %686 = load <4 x half>, ptr addrspace(3) %lds.const.gep1701, align 8, !dbg !17
  %lds.const.gep1702 = getelementptr i8, ptr addrspace(3) %317, i64 8384, !dbg !17
  %687 = load <4 x half>, ptr addrspace(3) %lds.const.gep1702, align 8, !dbg !17
  %688 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %686, <4 x float> %685) #9, !dbg !17
  %689 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %687, <4 x float> %593) #9, !dbg !17
  %lds.const.gep1703 = getelementptr i8, ptr addrspace(3) %317, i64 12480, !dbg !17
  %690 = load <4 x half>, ptr addrspace(3) %lds.const.gep1703, align 8, !dbg !17
  %lds.const.gep1704 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9216, !dbg !17
  %691 = load <4 x half>, ptr addrspace(3) %lds.const.gep1704, align 8, !dbg !17
  %692 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %690, <4 x float> %689) #9, !dbg !17
  %693 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %691, <4 x float> %597) #9, !dbg !17
  %lds.const.gep1705 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13312, !dbg !17
  %694 = load <4 x half>, ptr addrspace(3) %lds.const.gep1705, align 8, !dbg !17
  %lds.const.gep1706 = getelementptr i8, ptr addrspace(3) %317, i64 9280, !dbg !17
  %695 = load <4 x half>, ptr addrspace(3) %lds.const.gep1706, align 8, !dbg !17
  %696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %694, <4 x float> %693) #9, !dbg !17
  %697 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %695, <4 x float> %601) #9, !dbg !17
  %lds.const.gep1707 = getelementptr i8, ptr addrspace(3) %317, i64 13376, !dbg !17
  %698 = load <4 x half>, ptr addrspace(3) %lds.const.gep1707, align 8, !dbg !17
  %lds.const.gep1708 = getelementptr i8, ptr addrspace(3) %address.local660, i64 9344, !dbg !17
  %699 = load <4 x half>, ptr addrspace(3) %lds.const.gep1708, align 8, !dbg !17
  %700 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %698, <4 x float> %697) #9, !dbg !17
  %701 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %699, <4 x float> %605) #9, !dbg !17
  %lds.const.gep1709 = getelementptr i8, ptr addrspace(3) %address.local660, i64 13440, !dbg !17
  %702 = load <4 x half>, ptr addrspace(3) %lds.const.gep1709, align 8, !dbg !17
  %lds.const.gep1710 = getelementptr i8, ptr addrspace(3) %317, i64 9408, !dbg !17
  %703 = load <4 x half>, ptr addrspace(3) %lds.const.gep1710, align 8, !dbg !17
  %704 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %702, <4 x float> %701) #9, !dbg !17
  %705 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %703, <4 x float> %608) #9, !dbg !17
  %lds.const.gep1711 = getelementptr i8, ptr addrspace(3) %317, i64 13504, !dbg !17
  %706 = load <4 x half>, ptr addrspace(3) %lds.const.gep1711, align 8, !dbg !17
  %707 = or disjoint i64 %213, 127, !dbg !13
  %708 = icmp uge i64 %64, %707, !dbg !13
  %709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %706, <4 x float> %705) #9, !dbg !17
  %710 = and i1 %189, %708, !dbg !13
  br i1 %710, label %838, label %711, !dbg !13

711:                                              ; preds = %212
  %712 = add i64 %64, %32, !dbg !13
  %713 = lshr i64 %32, 2, !dbg !13
  %714 = and i64 %713, 48, !dbg !13
  %715 = add i64 %712, %714, !dbg !13
  %716 = and i32 %31, 240, !dbg !13
  %index.compact = zext nneg i32 %716 to i64, !dbg !13
  %717 = sub i64 %715, %index.compact, !dbg !13
  %718 = and i64 %32, 48, !dbg !13
  %719 = or disjoint i64 %213, %718, !dbg !13
  %720 = add i64 %717, 1, !dbg !23
  %721 = lshr i64 %720, 32, !dbg !24
  %compare.high = trunc nuw i64 %721 to i32, !dbg !24
  %722 = lshr i64 %213, 32, !dbg !24
  %compare.high1323 = trunc nuw i64 %722 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1323, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1323, !dbg !24
  %compare.low.base = trunc i64 %720 to i32, !dbg !24
  %compare.low.base1324 = trunc i64 %719 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1324, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %723 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low1325 = icmp ult i32 %compare.local, %compare.low.base, !dbg !24
  %compare.result1326 = select i1 %compare.high.equal, i1 %compare.low1325, i1 %compare.high.order, !dbg !24
  %724 = select i1 %compare.result1326, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1327 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1328 = or i32 %compare.local.base1327, 2, !dbg !24
  %compare.low1329 = icmp ult i32 %compare.local1328, %compare.low.base, !dbg !24
  %compare.result1330 = select i1 %compare.high.equal, i1 %compare.low1329, i1 %compare.high.order, !dbg !24
  %725 = select i1 %compare.result1330, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1331 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1332 = or i32 %compare.local.base1331, 3, !dbg !24
  %compare.low1333 = icmp ult i32 %compare.local1332, %compare.low.base, !dbg !24
  %compare.result1334 = select i1 %compare.high.equal, i1 %compare.low1333, i1 %compare.high.order, !dbg !24
  %726 = select i1 %compare.result1334, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1335 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1336 = or i32 %compare.local.base1335, 4, !dbg !24
  %compare.low1337 = icmp ult i32 %compare.local1336, %compare.low.base, !dbg !24
  %compare.result1338 = select i1 %compare.high.equal, i1 %compare.low1337, i1 %compare.high.order, !dbg !24
  %727 = select i1 %compare.result1338, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1339 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1340 = or i32 %compare.local.base1339, 5, !dbg !24
  %compare.low1341 = icmp ult i32 %compare.local1340, %compare.low.base, !dbg !24
  %compare.result1342 = select i1 %compare.high.equal, i1 %compare.low1341, i1 %compare.high.order, !dbg !24
  %728 = select i1 %compare.result1342, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1343 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1344 = or i32 %compare.local.base1343, 6, !dbg !24
  %compare.low1345 = icmp ult i32 %compare.local1344, %compare.low.base, !dbg !24
  %compare.result1346 = select i1 %compare.high.equal, i1 %compare.low1345, i1 %compare.high.order, !dbg !24
  %729 = select i1 %compare.result1346, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1347 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1348 = or i32 %compare.local.base1347, 7, !dbg !24
  %compare.low1349 = icmp ult i32 %compare.local1348, %compare.low.base, !dbg !24
  %compare.result1350 = select i1 %compare.high.equal, i1 %compare.low1349, i1 %compare.high.order, !dbg !24
  %730 = select i1 %compare.result1350, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1351 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1352 = or i32 %compare.local.base1351, 8, !dbg !24
  %compare.low1353 = icmp ult i32 %compare.local1352, %compare.low.base, !dbg !24
  %compare.result1354 = select i1 %compare.high.equal, i1 %compare.low1353, i1 %compare.high.order, !dbg !24
  %731 = select i1 %compare.result1354, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1355 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1356 = or i32 %compare.local.base1355, 9, !dbg !24
  %compare.low1357 = icmp ult i32 %compare.local1356, %compare.low.base, !dbg !24
  %compare.result1358 = select i1 %compare.high.equal, i1 %compare.low1357, i1 %compare.high.order, !dbg !24
  %732 = select i1 %compare.result1358, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1359 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1360 = or i32 %compare.local.base1359, 10, !dbg !24
  %compare.low1361 = icmp ult i32 %compare.local1360, %compare.low.base, !dbg !24
  %compare.result1362 = select i1 %compare.high.equal, i1 %compare.low1361, i1 %compare.high.order, !dbg !24
  %733 = select i1 %compare.result1362, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1363 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1364 = or i32 %compare.local.base1363, 11, !dbg !24
  %compare.low1365 = icmp ult i32 %compare.local1364, %compare.low.base, !dbg !24
  %compare.result1366 = select i1 %compare.high.equal, i1 %compare.low1365, i1 %compare.high.order, !dbg !24
  %734 = select i1 %compare.result1366, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1367 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1368 = or i32 %compare.local.base1367, 12, !dbg !24
  %compare.low1369 = icmp ult i32 %compare.local1368, %compare.low.base, !dbg !24
  %compare.result1370 = select i1 %compare.high.equal, i1 %compare.low1369, i1 %compare.high.order, !dbg !24
  %735 = select i1 %compare.result1370, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1371 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1372 = or i32 %compare.local.base1371, 13, !dbg !24
  %compare.low1373 = icmp ult i32 %compare.local1372, %compare.low.base, !dbg !24
  %compare.result1374 = select i1 %compare.high.equal, i1 %compare.low1373, i1 %compare.high.order, !dbg !24
  %736 = select i1 %compare.result1374, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1375 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1376 = or i32 %compare.local.base1375, 14, !dbg !24
  %compare.low1377 = icmp ult i32 %compare.local1376, %compare.low.base, !dbg !24
  %compare.result1378 = select i1 %compare.high.equal, i1 %compare.low1377, i1 %compare.high.order, !dbg !24
  %737 = select i1 %compare.result1378, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1379 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1380 = or i32 %compare.local.base1379, 15, !dbg !24
  %compare.low1381 = icmp ult i32 %compare.local1380, %compare.low.base, !dbg !24
  %compare.result1382 = select i1 %compare.high.equal, i1 %compare.low1381, i1 %compare.high.order, !dbg !24
  %738 = select i1 %compare.result1382, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1383 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1384 = or i32 %compare.local.base1383, 64, !dbg !24
  %compare.low1385 = icmp ult i32 %compare.local1384, %compare.low.base, !dbg !24
  %compare.result1386 = select i1 %compare.high.equal, i1 %compare.low1385, i1 %compare.high.order, !dbg !24
  %739 = select i1 %compare.result1386, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1387 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1388 = or i32 %compare.local.base1387, 65, !dbg !24
  %compare.low1389 = icmp ult i32 %compare.local1388, %compare.low.base, !dbg !24
  %compare.result1390 = select i1 %compare.high.equal, i1 %compare.low1389, i1 %compare.high.order, !dbg !24
  %740 = select i1 %compare.result1390, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1391 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1392 = or i32 %compare.local.base1391, 66, !dbg !24
  %compare.low1393 = icmp ult i32 %compare.local1392, %compare.low.base, !dbg !24
  %compare.result1394 = select i1 %compare.high.equal, i1 %compare.low1393, i1 %compare.high.order, !dbg !24
  %741 = select i1 %compare.result1394, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1395 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1396 = or i32 %compare.local.base1395, 67, !dbg !24
  %compare.low1397 = icmp ult i32 %compare.local1396, %compare.low.base, !dbg !24
  %compare.result1398 = select i1 %compare.high.equal, i1 %compare.low1397, i1 %compare.high.order, !dbg !24
  %742 = select i1 %compare.result1398, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1399 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1400 = or i32 %compare.local.base1399, 68, !dbg !24
  %compare.low1401 = icmp ult i32 %compare.local1400, %compare.low.base, !dbg !24
  %compare.result1402 = select i1 %compare.high.equal, i1 %compare.low1401, i1 %compare.high.order, !dbg !24
  %743 = select i1 %compare.result1402, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1403 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1404 = or i32 %compare.local.base1403, 69, !dbg !24
  %compare.low1405 = icmp ult i32 %compare.local1404, %compare.low.base, !dbg !24
  %compare.result1406 = select i1 %compare.high.equal, i1 %compare.low1405, i1 %compare.high.order, !dbg !24
  %744 = select i1 %compare.result1406, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1407 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1408 = or i32 %compare.local.base1407, 70, !dbg !24
  %compare.low1409 = icmp ult i32 %compare.local1408, %compare.low.base, !dbg !24
  %compare.result1410 = select i1 %compare.high.equal, i1 %compare.low1409, i1 %compare.high.order, !dbg !24
  %745 = select i1 %compare.result1410, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1411 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1412 = or i32 %compare.local.base1411, 71, !dbg !24
  %compare.low1413 = icmp ult i32 %compare.local1412, %compare.low.base, !dbg !24
  %compare.result1414 = select i1 %compare.high.equal, i1 %compare.low1413, i1 %compare.high.order, !dbg !24
  %746 = select i1 %compare.result1414, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1415 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1416 = or i32 %compare.local.base1415, 72, !dbg !24
  %compare.low1417 = icmp ult i32 %compare.local1416, %compare.low.base, !dbg !24
  %compare.result1418 = select i1 %compare.high.equal, i1 %compare.low1417, i1 %compare.high.order, !dbg !24
  %747 = select i1 %compare.result1418, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1419 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1420 = or i32 %compare.local.base1419, 73, !dbg !24
  %compare.low1421 = icmp ult i32 %compare.local1420, %compare.low.base, !dbg !24
  %compare.result1422 = select i1 %compare.high.equal, i1 %compare.low1421, i1 %compare.high.order, !dbg !24
  %748 = select i1 %compare.result1422, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1423 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1424 = or i32 %compare.local.base1423, 74, !dbg !24
  %compare.low1425 = icmp ult i32 %compare.local1424, %compare.low.base, !dbg !24
  %compare.result1426 = select i1 %compare.high.equal, i1 %compare.low1425, i1 %compare.high.order, !dbg !24
  %749 = select i1 %compare.result1426, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1427 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1428 = or i32 %compare.local.base1427, 75, !dbg !24
  %compare.low1429 = icmp ult i32 %compare.local1428, %compare.low.base, !dbg !24
  %compare.result1430 = select i1 %compare.high.equal, i1 %compare.low1429, i1 %compare.high.order, !dbg !24
  %750 = select i1 %compare.result1430, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1431 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1432 = or i32 %compare.local.base1431, 76, !dbg !24
  %compare.low1433 = icmp ult i32 %compare.local1432, %compare.low.base, !dbg !24
  %compare.result1434 = select i1 %compare.high.equal, i1 %compare.low1433, i1 %compare.high.order, !dbg !24
  %751 = select i1 %compare.result1434, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1435 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1436 = or i32 %compare.local.base1435, 77, !dbg !24
  %compare.low1437 = icmp ult i32 %compare.local1436, %compare.low.base, !dbg !24
  %compare.result1438 = select i1 %compare.high.equal, i1 %compare.low1437, i1 %compare.high.order, !dbg !24
  %752 = select i1 %compare.result1438, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1439 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1440 = or i32 %compare.local.base1439, 78, !dbg !24
  %compare.low1441 = icmp ult i32 %compare.local1440, %compare.low.base, !dbg !24
  %compare.result1442 = select i1 %compare.high.equal, i1 %compare.low1441, i1 %compare.high.order, !dbg !24
  %753 = select i1 %compare.result1442, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1443 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1324, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1444 = or i32 %compare.local.base1443, 79, !dbg !24
  %compare.low1445 = icmp ult i32 %compare.local1444, %compare.low.base, !dbg !24
  %compare.result1446 = select i1 %compare.high.equal, i1 %compare.low1445, i1 %compare.high.order, !dbg !24
  %754 = select i1 %compare.result1446, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %755 = add i64 %717, 65, !dbg !23
  %756 = lshr i64 %755, 32, !dbg !24
  %compare.high1447 = trunc nuw i64 %756 to i32, !dbg !24
  %757 = lshr i64 %213, 32, !dbg !24
  %compare.high1448 = trunc nuw i64 %757 to i32, !dbg !24
  %compare.high.equal1449 = icmp eq i32 %compare.high1447, %compare.high1448, !dbg !24
  %compare.high.order1450 = icmp ugt i32 %compare.high1447, %compare.high1448, !dbg !24
  %compare.low.base1451 = trunc i64 %755 to i32, !dbg !24
  %compare.low.base1452 = trunc i64 %719 to i32, !dbg !24
  %compare.low1453 = icmp ugt i32 %compare.low.base1451, %compare.low.base1452, !dbg !24
  %compare.result1454 = select i1 %compare.high.equal1449, i1 %compare.low1453, i1 %compare.high.order1450, !dbg !24
  %758 = select i1 %compare.result1454, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1455 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1456 = or i32 %compare.local.base1455, 1, !dbg !24
  %compare.low1457 = icmp ult i32 %compare.local1456, %compare.low.base1451, !dbg !24
  %compare.result1458 = select i1 %compare.high.equal1449, i1 %compare.low1457, i1 %compare.high.order1450, !dbg !24
  %759 = select i1 %compare.result1458, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1459 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1460 = or i32 %compare.local.base1459, 2, !dbg !24
  %compare.low1461 = icmp ult i32 %compare.local1460, %compare.low.base1451, !dbg !24
  %compare.result1462 = select i1 %compare.high.equal1449, i1 %compare.low1461, i1 %compare.high.order1450, !dbg !24
  %760 = select i1 %compare.result1462, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1463 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1464 = or i32 %compare.local.base1463, 3, !dbg !24
  %compare.low1465 = icmp ult i32 %compare.local1464, %compare.low.base1451, !dbg !24
  %compare.result1466 = select i1 %compare.high.equal1449, i1 %compare.low1465, i1 %compare.high.order1450, !dbg !24
  %761 = select i1 %compare.result1466, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1467 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1468 = or i32 %compare.local.base1467, 4, !dbg !24
  %compare.low1469 = icmp ult i32 %compare.local1468, %compare.low.base1451, !dbg !24
  %compare.result1470 = select i1 %compare.high.equal1449, i1 %compare.low1469, i1 %compare.high.order1450, !dbg !24
  %762 = select i1 %compare.result1470, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1471 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1472 = or i32 %compare.local.base1471, 5, !dbg !24
  %compare.low1473 = icmp ult i32 %compare.local1472, %compare.low.base1451, !dbg !24
  %compare.result1474 = select i1 %compare.high.equal1449, i1 %compare.low1473, i1 %compare.high.order1450, !dbg !24
  %763 = select i1 %compare.result1474, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1475 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1476 = or i32 %compare.local.base1475, 6, !dbg !24
  %compare.low1477 = icmp ult i32 %compare.local1476, %compare.low.base1451, !dbg !24
  %compare.result1478 = select i1 %compare.high.equal1449, i1 %compare.low1477, i1 %compare.high.order1450, !dbg !24
  %764 = select i1 %compare.result1478, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1479 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1480 = or i32 %compare.local.base1479, 7, !dbg !24
  %compare.low1481 = icmp ult i32 %compare.local1480, %compare.low.base1451, !dbg !24
  %compare.result1482 = select i1 %compare.high.equal1449, i1 %compare.low1481, i1 %compare.high.order1450, !dbg !24
  %765 = select i1 %compare.result1482, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1483 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1484 = or i32 %compare.local.base1483, 8, !dbg !24
  %compare.low1485 = icmp ult i32 %compare.local1484, %compare.low.base1451, !dbg !24
  %compare.result1486 = select i1 %compare.high.equal1449, i1 %compare.low1485, i1 %compare.high.order1450, !dbg !24
  %766 = select i1 %compare.result1486, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1487 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1488 = or i32 %compare.local.base1487, 9, !dbg !24
  %compare.low1489 = icmp ult i32 %compare.local1488, %compare.low.base1451, !dbg !24
  %compare.result1490 = select i1 %compare.high.equal1449, i1 %compare.low1489, i1 %compare.high.order1450, !dbg !24
  %767 = select i1 %compare.result1490, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1491 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1492 = or i32 %compare.local.base1491, 10, !dbg !24
  %compare.low1493 = icmp ult i32 %compare.local1492, %compare.low.base1451, !dbg !24
  %compare.result1494 = select i1 %compare.high.equal1449, i1 %compare.low1493, i1 %compare.high.order1450, !dbg !24
  %768 = select i1 %compare.result1494, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1495 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1496 = or i32 %compare.local.base1495, 11, !dbg !24
  %compare.low1497 = icmp ult i32 %compare.local1496, %compare.low.base1451, !dbg !24
  %compare.result1498 = select i1 %compare.high.equal1449, i1 %compare.low1497, i1 %compare.high.order1450, !dbg !24
  %769 = select i1 %compare.result1498, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1499 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1500 = or i32 %compare.local.base1499, 12, !dbg !24
  %compare.low1501 = icmp ult i32 %compare.local1500, %compare.low.base1451, !dbg !24
  %compare.result1502 = select i1 %compare.high.equal1449, i1 %compare.low1501, i1 %compare.high.order1450, !dbg !24
  %770 = select i1 %compare.result1502, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1503 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1504 = or i32 %compare.local.base1503, 13, !dbg !24
  %compare.low1505 = icmp ult i32 %compare.local1504, %compare.low.base1451, !dbg !24
  %compare.result1506 = select i1 %compare.high.equal1449, i1 %compare.low1505, i1 %compare.high.order1450, !dbg !24
  %771 = select i1 %compare.result1506, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1507 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1508 = or i32 %compare.local.base1507, 14, !dbg !24
  %compare.low1509 = icmp ult i32 %compare.local1508, %compare.low.base1451, !dbg !24
  %compare.result1510 = select i1 %compare.high.equal1449, i1 %compare.low1509, i1 %compare.high.order1450, !dbg !24
  %772 = select i1 %compare.result1510, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1511 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1512 = or i32 %compare.local.base1511, 15, !dbg !24
  %compare.low1513 = icmp ult i32 %compare.local1512, %compare.low.base1451, !dbg !24
  %compare.result1514 = select i1 %compare.high.equal1449, i1 %compare.low1513, i1 %compare.high.order1450, !dbg !24
  %773 = select i1 %compare.result1514, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1515 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1516 = or i32 %compare.local.base1515, 64, !dbg !24
  %compare.low1517 = icmp ult i32 %compare.local1516, %compare.low.base1451, !dbg !24
  %compare.result1518 = select i1 %compare.high.equal1449, i1 %compare.low1517, i1 %compare.high.order1450, !dbg !24
  %774 = select i1 %compare.result1518, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1519 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1520 = or i32 %compare.local.base1519, 65, !dbg !24
  %compare.low1521 = icmp ult i32 %compare.local1520, %compare.low.base1451, !dbg !24
  %compare.result1522 = select i1 %compare.high.equal1449, i1 %compare.low1521, i1 %compare.high.order1450, !dbg !24
  %775 = select i1 %compare.result1522, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1523 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1524 = or i32 %compare.local.base1523, 66, !dbg !24
  %compare.low1525 = icmp ult i32 %compare.local1524, %compare.low.base1451, !dbg !24
  %compare.result1526 = select i1 %compare.high.equal1449, i1 %compare.low1525, i1 %compare.high.order1450, !dbg !24
  %776 = select i1 %compare.result1526, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1527 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1528 = or i32 %compare.local.base1527, 67, !dbg !24
  %compare.low1529 = icmp ult i32 %compare.local1528, %compare.low.base1451, !dbg !24
  %compare.result1530 = select i1 %compare.high.equal1449, i1 %compare.low1529, i1 %compare.high.order1450, !dbg !24
  %777 = select i1 %compare.result1530, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1531 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1532 = or i32 %compare.local.base1531, 68, !dbg !24
  %compare.low1533 = icmp ult i32 %compare.local1532, %compare.low.base1451, !dbg !24
  %compare.result1534 = select i1 %compare.high.equal1449, i1 %compare.low1533, i1 %compare.high.order1450, !dbg !24
  %778 = select i1 %compare.result1534, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1535 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1536 = or i32 %compare.local.base1535, 69, !dbg !24
  %compare.low1537 = icmp ult i32 %compare.local1536, %compare.low.base1451, !dbg !24
  %compare.result1538 = select i1 %compare.high.equal1449, i1 %compare.low1537, i1 %compare.high.order1450, !dbg !24
  %779 = select i1 %compare.result1538, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1539 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1540 = or i32 %compare.local.base1539, 70, !dbg !24
  %compare.low1541 = icmp ult i32 %compare.local1540, %compare.low.base1451, !dbg !24
  %compare.result1542 = select i1 %compare.high.equal1449, i1 %compare.low1541, i1 %compare.high.order1450, !dbg !24
  %780 = select i1 %compare.result1542, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1543 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1544 = or i32 %compare.local.base1543, 71, !dbg !24
  %compare.low1545 = icmp ult i32 %compare.local1544, %compare.low.base1451, !dbg !24
  %compare.result1546 = select i1 %compare.high.equal1449, i1 %compare.low1545, i1 %compare.high.order1450, !dbg !24
  %781 = select i1 %compare.result1546, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1547 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1548 = or i32 %compare.local.base1547, 72, !dbg !24
  %compare.low1549 = icmp ult i32 %compare.local1548, %compare.low.base1451, !dbg !24
  %compare.result1550 = select i1 %compare.high.equal1449, i1 %compare.low1549, i1 %compare.high.order1450, !dbg !24
  %782 = select i1 %compare.result1550, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1551 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1552 = or i32 %compare.local.base1551, 73, !dbg !24
  %compare.low1553 = icmp ult i32 %compare.local1552, %compare.low.base1451, !dbg !24
  %compare.result1554 = select i1 %compare.high.equal1449, i1 %compare.low1553, i1 %compare.high.order1450, !dbg !24
  %783 = select i1 %compare.result1554, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1555 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1556 = or i32 %compare.local.base1555, 74, !dbg !24
  %compare.low1557 = icmp ult i32 %compare.local1556, %compare.low.base1451, !dbg !24
  %compare.result1558 = select i1 %compare.high.equal1449, i1 %compare.low1557, i1 %compare.high.order1450, !dbg !24
  %784 = select i1 %compare.result1558, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1559 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1560 = or i32 %compare.local.base1559, 75, !dbg !24
  %compare.low1561 = icmp ult i32 %compare.local1560, %compare.low.base1451, !dbg !24
  %compare.result1562 = select i1 %compare.high.equal1449, i1 %compare.low1561, i1 %compare.high.order1450, !dbg !24
  %785 = select i1 %compare.result1562, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1563 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1564 = or i32 %compare.local.base1563, 76, !dbg !24
  %compare.low1565 = icmp ult i32 %compare.local1564, %compare.low.base1451, !dbg !24
  %compare.result1566 = select i1 %compare.high.equal1449, i1 %compare.low1565, i1 %compare.high.order1450, !dbg !24
  %786 = select i1 %compare.result1566, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1567 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1568 = or i32 %compare.local.base1567, 77, !dbg !24
  %compare.low1569 = icmp ult i32 %compare.local1568, %compare.low.base1451, !dbg !24
  %compare.result1570 = select i1 %compare.high.equal1449, i1 %compare.low1569, i1 %compare.high.order1450, !dbg !24
  %787 = select i1 %compare.result1570, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1571 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1572 = or i32 %compare.local.base1571, 78, !dbg !24
  %compare.low1573 = icmp ult i32 %compare.local1572, %compare.low.base1451, !dbg !24
  %compare.result1574 = select i1 %compare.high.equal1449, i1 %compare.low1573, i1 %compare.high.order1450, !dbg !24
  %788 = select i1 %compare.result1574, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1575 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1452, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1576 = or i32 %compare.local.base1575, 79, !dbg !24
  %compare.low1577 = icmp ult i32 %compare.local1576, %compare.low.base1451, !dbg !24
  %compare.result1578 = select i1 %compare.high.equal1449, i1 %compare.low1577, i1 %compare.high.order1450, !dbg !24
  %789 = select i1 %compare.result1578, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %790 = insertelement <4 x float> poison, float %723, i64 0, !dbg !13
  %791 = insertelement <4 x float> %790, float %724, i64 1, !dbg !13
  %792 = insertelement <4 x float> %791, float %725, i64 2, !dbg !13
  %.masked126 = insertelement <4 x float> %792, float %726, i64 3, !dbg !13
  %793 = insertelement <4 x float> poison, float %727, i64 0, !dbg !13
  %794 = insertelement <4 x float> %793, float %728, i64 1, !dbg !13
  %795 = insertelement <4 x float> %794, float %729, i64 2, !dbg !13
  %.masked128 = insertelement <4 x float> %795, float %730, i64 3, !dbg !13
  %796 = insertelement <4 x float> poison, float %731, i64 0, !dbg !13
  %797 = insertelement <4 x float> %796, float %732, i64 1, !dbg !13
  %798 = insertelement <4 x float> %797, float %733, i64 2, !dbg !13
  %.masked130 = insertelement <4 x float> %798, float %734, i64 3, !dbg !13
  %799 = insertelement <4 x float> poison, float %735, i64 0, !dbg !13
  %800 = insertelement <4 x float> %799, float %736, i64 1, !dbg !13
  %801 = insertelement <4 x float> %800, float %737, i64 2, !dbg !13
  %.masked132 = insertelement <4 x float> %801, float %738, i64 3, !dbg !13
  %802 = insertelement <4 x float> poison, float %739, i64 0, !dbg !13
  %803 = insertelement <4 x float> %802, float %740, i64 1, !dbg !13
  %804 = insertelement <4 x float> %803, float %741, i64 2, !dbg !13
  %.masked134 = insertelement <4 x float> %804, float %742, i64 3, !dbg !13
  %805 = insertelement <4 x float> poison, float %743, i64 0, !dbg !13
  %806 = insertelement <4 x float> %805, float %744, i64 1, !dbg !13
  %807 = insertelement <4 x float> %806, float %745, i64 2, !dbg !13
  %.masked136 = insertelement <4 x float> %807, float %746, i64 3, !dbg !13
  %808 = insertelement <4 x float> poison, float %747, i64 0, !dbg !13
  %809 = insertelement <4 x float> %808, float %748, i64 1, !dbg !13
  %810 = insertelement <4 x float> %809, float %749, i64 2, !dbg !13
  %.masked138 = insertelement <4 x float> %810, float %750, i64 3, !dbg !13
  %811 = insertelement <4 x float> poison, float %751, i64 0, !dbg !13
  %812 = insertelement <4 x float> %811, float %752, i64 1, !dbg !13
  %813 = insertelement <4 x float> %812, float %753, i64 2, !dbg !13
  %.masked140 = insertelement <4 x float> %813, float %754, i64 3, !dbg !13
  %814 = insertelement <4 x float> poison, float %758, i64 0, !dbg !13
  %815 = insertelement <4 x float> %814, float %759, i64 1, !dbg !13
  %816 = insertelement <4 x float> %815, float %760, i64 2, !dbg !13
  %.masked143 = insertelement <4 x float> %816, float %761, i64 3, !dbg !13
  %817 = insertelement <4 x float> poison, float %762, i64 0, !dbg !13
  %818 = insertelement <4 x float> %817, float %763, i64 1, !dbg !13
  %819 = insertelement <4 x float> %818, float %764, i64 2, !dbg !13
  %.masked145 = insertelement <4 x float> %819, float %765, i64 3, !dbg !13
  %820 = insertelement <4 x float> poison, float %766, i64 0, !dbg !13
  %821 = insertelement <4 x float> %820, float %767, i64 1, !dbg !13
  %822 = insertelement <4 x float> %821, float %768, i64 2, !dbg !13
  %.masked147 = insertelement <4 x float> %822, float %769, i64 3, !dbg !13
  %823 = insertelement <4 x float> poison, float %770, i64 0, !dbg !13
  %824 = insertelement <4 x float> %823, float %771, i64 1, !dbg !13
  %825 = insertelement <4 x float> %824, float %772, i64 2, !dbg !13
  %.masked149 = insertelement <4 x float> %825, float %773, i64 3, !dbg !13
  %826 = insertelement <4 x float> poison, float %774, i64 0, !dbg !13
  %827 = insertelement <4 x float> %826, float %775, i64 1, !dbg !13
  %828 = insertelement <4 x float> %827, float %776, i64 2, !dbg !13
  %.masked151 = insertelement <4 x float> %828, float %777, i64 3, !dbg !13
  %829 = insertelement <4 x float> poison, float %778, i64 0, !dbg !13
  %830 = insertelement <4 x float> %829, float %779, i64 1, !dbg !13
  %831 = insertelement <4 x float> %830, float %780, i64 2, !dbg !13
  %.masked153 = insertelement <4 x float> %831, float %781, i64 3, !dbg !13
  %832 = insertelement <4 x float> poison, float %782, i64 0, !dbg !13
  %833 = insertelement <4 x float> %832, float %783, i64 1, !dbg !13
  %834 = insertelement <4 x float> %833, float %784, i64 2, !dbg !13
  %.masked155 = insertelement <4 x float> %834, float %785, i64 3, !dbg !13
  %835 = insertelement <4 x float> poison, float %786, i64 0, !dbg !13
  %836 = insertelement <4 x float> %835, float %787, i64 1, !dbg !13
  %837 = insertelement <4 x float> %836, float %788, i64 2, !dbg !13
  %.masked157 = insertelement <4 x float> %837, float %789, i64 3, !dbg !13
  %.masked127 = fadd <4 x float> %648, %.masked126, !dbg !26
  %.masked129 = fadd <4 x float> %652, %.masked128, !dbg !26
  %.masked131 = fadd <4 x float> %656, %.masked130, !dbg !26
  %.masked133 = fadd <4 x float> %660, %.masked132, !dbg !26
  %.masked135 = fadd <4 x float> %664, %.masked134, !dbg !26
  %.masked137 = fadd <4 x float> %668, %.masked136, !dbg !26
  %.masked139 = fadd <4 x float> %672, %.masked138, !dbg !26
  %.masked141 = fadd <4 x float> %676, %.masked140, !dbg !26
  %.masked144 = fadd <4 x float> %680, %.masked143, !dbg !26
  %.masked146 = fadd <4 x float> %684, %.masked145, !dbg !26
  %.masked148 = fadd <4 x float> %688, %.masked147, !dbg !26
  %.masked150 = fadd <4 x float> %692, %.masked149, !dbg !26
  %.masked152 = fadd <4 x float> %696, %.masked151, !dbg !26
  %.masked154 = fadd <4 x float> %700, %.masked153, !dbg !26
  %.masked156 = fadd <4 x float> %704, %.masked155, !dbg !26
  %.masked158 = fadd <4 x float> %709, %.masked157, !dbg !26
  br label %838, !dbg !13

838:                                              ; preds = %711, %212
  %839 = phi <4 x float> [ %.masked127, %711 ], [ %648, %212 ], !dbg !26
  %840 = phi <4 x float> [ %.masked129, %711 ], [ %652, %212 ], !dbg !26
  %841 = phi <4 x float> [ %.masked131, %711 ], [ %656, %212 ], !dbg !26
  %842 = phi <4 x float> [ %.masked133, %711 ], [ %660, %212 ], !dbg !26
  %843 = phi <4 x float> [ %.masked135, %711 ], [ %664, %212 ], !dbg !26
  %844 = phi <4 x float> [ %.masked137, %711 ], [ %668, %212 ], !dbg !26
  %845 = phi <4 x float> [ %.masked139, %711 ], [ %672, %212 ], !dbg !26
  %846 = phi <4 x float> [ %.masked141, %711 ], [ %676, %212 ], !dbg !26
  %847 = phi <4 x float> [ %.masked144, %711 ], [ %680, %212 ], !dbg !26
  %848 = phi <4 x float> [ %.masked146, %711 ], [ %684, %212 ], !dbg !26
  %849 = phi <4 x float> [ %.masked148, %711 ], [ %688, %212 ], !dbg !26
  %850 = phi <4 x float> [ %.masked150, %711 ], [ %692, %212 ], !dbg !26
  %851 = phi <4 x float> [ %.masked152, %711 ], [ %696, %212 ], !dbg !26
  %852 = phi <4 x float> [ %.masked154, %711 ], [ %700, %212 ], !dbg !26
  %853 = phi <4 x float> [ %.masked156, %711 ], [ %704, %212 ], !dbg !26
  %854 = phi <4 x float> [ %.masked158, %711 ], [ %709, %212 ], !dbg !26
  %855 = extractelement <4 x float> %839, i64 0, !dbg !26
  %856 = extractelement <4 x float> %839, i64 1, !dbg !26
  %857 = extractelement <4 x float> %839, i64 2, !dbg !26
  %858 = extractelement <4 x float> %839, i64 3, !dbg !26
  %859 = extractelement <4 x float> %840, i64 0, !dbg !26
  %860 = extractelement <4 x float> %840, i64 1, !dbg !26
  %861 = extractelement <4 x float> %840, i64 2, !dbg !26
  %862 = extractelement <4 x float> %840, i64 3, !dbg !26
  %863 = extractelement <4 x float> %841, i64 0, !dbg !26
  %864 = extractelement <4 x float> %841, i64 1, !dbg !26
  %865 = extractelement <4 x float> %841, i64 2, !dbg !26
  %866 = extractelement <4 x float> %841, i64 3, !dbg !26
  %867 = extractelement <4 x float> %842, i64 0, !dbg !26
  %868 = extractelement <4 x float> %842, i64 1, !dbg !26
  %869 = extractelement <4 x float> %842, i64 2, !dbg !26
  %870 = extractelement <4 x float> %842, i64 3, !dbg !26
  %871 = extractelement <4 x float> %843, i64 0, !dbg !26
  %872 = extractelement <4 x float> %843, i64 1, !dbg !26
  %873 = extractelement <4 x float> %843, i64 2, !dbg !26
  %874 = extractelement <4 x float> %843, i64 3, !dbg !26
  %875 = extractelement <4 x float> %844, i64 0, !dbg !26
  %876 = extractelement <4 x float> %844, i64 1, !dbg !26
  %877 = extractelement <4 x float> %844, i64 2, !dbg !26
  %878 = extractelement <4 x float> %844, i64 3, !dbg !26
  %879 = extractelement <4 x float> %845, i64 0, !dbg !26
  %880 = extractelement <4 x float> %845, i64 1, !dbg !26
  %881 = extractelement <4 x float> %845, i64 2, !dbg !26
  %882 = extractelement <4 x float> %845, i64 3, !dbg !26
  %883 = extractelement <4 x float> %846, i64 0, !dbg !26
  %884 = extractelement <4 x float> %846, i64 1, !dbg !26
  %885 = extractelement <4 x float> %846, i64 2, !dbg !26
  %886 = extractelement <4 x float> %846, i64 3, !dbg !26
  %887 = extractelement <4 x float> %847, i64 0, !dbg !26
  %888 = extractelement <4 x float> %847, i64 1, !dbg !26
  %889 = extractelement <4 x float> %847, i64 2, !dbg !26
  %890 = extractelement <4 x float> %847, i64 3, !dbg !26
  %891 = extractelement <4 x float> %848, i64 0, !dbg !26
  %892 = extractelement <4 x float> %848, i64 1, !dbg !26
  %893 = extractelement <4 x float> %848, i64 2, !dbg !26
  %894 = extractelement <4 x float> %848, i64 3, !dbg !26
  %895 = extractelement <4 x float> %849, i64 0, !dbg !26
  %896 = extractelement <4 x float> %849, i64 1, !dbg !26
  %897 = extractelement <4 x float> %849, i64 2, !dbg !26
  %898 = extractelement <4 x float> %849, i64 3, !dbg !26
  %899 = extractelement <4 x float> %850, i64 0, !dbg !26
  %900 = extractelement <4 x float> %850, i64 1, !dbg !26
  %901 = extractelement <4 x float> %850, i64 2, !dbg !26
  %902 = extractelement <4 x float> %850, i64 3, !dbg !26
  %903 = extractelement <4 x float> %851, i64 0, !dbg !26
  %904 = extractelement <4 x float> %851, i64 1, !dbg !26
  %905 = extractelement <4 x float> %851, i64 2, !dbg !26
  %906 = extractelement <4 x float> %851, i64 3, !dbg !26
  %907 = extractelement <4 x float> %852, i64 0, !dbg !26
  %908 = extractelement <4 x float> %852, i64 1, !dbg !26
  %909 = extractelement <4 x float> %852, i64 2, !dbg !26
  %910 = extractelement <4 x float> %852, i64 3, !dbg !26
  %911 = extractelement <4 x float> %853, i64 0, !dbg !26
  %912 = extractelement <4 x float> %853, i64 1, !dbg !26
  %913 = extractelement <4 x float> %853, i64 2, !dbg !26
  %914 = extractelement <4 x float> %853, i64 3, !dbg !26
  %915 = extractelement <4 x float> %854, i64 0, !dbg !26
  %916 = extractelement <4 x float> %854, i64 1, !dbg !26
  %917 = extractelement <4 x float> %854, i64 2, !dbg !26
  %918 = extractelement <4 x float> %854, i64 3, !dbg !26
  %919 = call float @llvm.maxnum.f32(float %855, float 0xFFF0000000000000), !dbg !27, !frisk.local_reduction !1
  %local.reduce251 = call float @llvm.maxnum.f32(float %919, float %859), !dbg !27, !frisk.local_reduction !1
  %local.reduce252 = call float @llvm.maxnum.f32(float %856, float %860), !dbg !27, !frisk.local_reduction !1
  %local.reduce253 = call float @llvm.maxnum.f32(float %857, float %861), !dbg !27, !frisk.local_reduction !1
  %local.reduce254 = call float @llvm.maxnum.f32(float %858, float %862), !dbg !27, !frisk.local_reduction !1
  %local.reduce255 = call float @llvm.maxnum.f32(float %local.reduce251, float %863), !dbg !27, !frisk.local_reduction !1
  %local.reduce256 = call float @llvm.maxnum.f32(float %local.reduce252, float %864), !dbg !27, !frisk.local_reduction !1
  %local.reduce257 = call float @llvm.maxnum.f32(float %local.reduce253, float %865), !dbg !27, !frisk.local_reduction !1
  %local.reduce258 = call float @llvm.maxnum.f32(float %local.reduce254, float %866), !dbg !27, !frisk.local_reduction !1
  %local.reduce259 = call float @llvm.maxnum.f32(float %local.reduce255, float %867), !dbg !27, !frisk.local_reduction !1
  %local.reduce260 = call float @llvm.maxnum.f32(float %local.reduce256, float %868), !dbg !27, !frisk.local_reduction !1
  %local.reduce261 = call float @llvm.maxnum.f32(float %local.reduce257, float %869), !dbg !27, !frisk.local_reduction !1
  %local.reduce262 = call float @llvm.maxnum.f32(float %local.reduce258, float %870), !dbg !27, !frisk.local_reduction !1
  %local.reduce263 = call float @llvm.maxnum.f32(float %local.reduce259, float %871), !dbg !27, !frisk.local_reduction !1
  %local.reduce264 = call float @llvm.maxnum.f32(float %local.reduce260, float %872), !dbg !27, !frisk.local_reduction !1
  %local.reduce265 = call float @llvm.maxnum.f32(float %local.reduce261, float %873), !dbg !27, !frisk.local_reduction !1
  %local.reduce266 = call float @llvm.maxnum.f32(float %local.reduce262, float %874), !dbg !27, !frisk.local_reduction !1
  %local.reduce267 = call float @llvm.maxnum.f32(float %local.reduce263, float %875), !dbg !27, !frisk.local_reduction !1
  %local.reduce268 = call float @llvm.maxnum.f32(float %local.reduce264, float %876), !dbg !27, !frisk.local_reduction !1
  %local.reduce269 = call float @llvm.maxnum.f32(float %local.reduce265, float %877), !dbg !27, !frisk.local_reduction !1
  %local.reduce270 = call float @llvm.maxnum.f32(float %local.reduce266, float %878), !dbg !27, !frisk.local_reduction !1
  %local.reduce271 = call float @llvm.maxnum.f32(float %local.reduce267, float %879), !dbg !27, !frisk.local_reduction !1
  %local.reduce272 = call float @llvm.maxnum.f32(float %local.reduce268, float %880), !dbg !27, !frisk.local_reduction !1
  %local.reduce273 = call float @llvm.maxnum.f32(float %local.reduce269, float %881), !dbg !27, !frisk.local_reduction !1
  %local.reduce274 = call float @llvm.maxnum.f32(float %local.reduce270, float %882), !dbg !27, !frisk.local_reduction !1
  %local.reduce275 = call float @llvm.maxnum.f32(float %local.reduce271, float %883), !dbg !27, !frisk.local_reduction !1
  %local.reduce276 = call float @llvm.maxnum.f32(float %local.reduce272, float %884), !dbg !27, !frisk.local_reduction !1
  %local.reduce277 = call float @llvm.maxnum.f32(float %local.reduce273, float %885), !dbg !27, !frisk.local_reduction !1
  %local.reduce278 = call float @llvm.maxnum.f32(float %local.reduce274, float %886), !dbg !27, !frisk.local_reduction !1
  %local.reduce279 = call float @llvm.maxnum.f32(float %local.reduce275, float %local.reduce276), !dbg !27, !frisk.local_reduction !1
  %local.reduce280 = call float @llvm.maxnum.f32(float %local.reduce277, float %local.reduce278), !dbg !27, !frisk.local_reduction !1
  %920 = call float @llvm.maxnum.f32(float %local.reduce279, float %local.reduce280), !dbg !27, !frisk.local_reduction !1
  %921 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %922 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %921), !dbg !27
  %923 = add nuw nsw i32 %922, 64, !dbg !27
  %924 = xor i32 %922, 16, !dbg !27
  %925 = and i32 %923, 192, !dbg !27
  %926 = icmp samesign ult i32 %924, %925, !dbg !27
  %927 = select i1 %926, i32 %924, i32 %922, !dbg !27
  %928 = shl nuw nsw i32 %927, 2, !dbg !27
  %929 = bitcast float %920 to i32, !dbg !27
  %930 = call i32 @llvm.amdgcn.ds.bpermute(i32 %928, i32 %929), !dbg !27
  %931 = bitcast i32 %930 to float, !dbg !27
  %932 = call float @llvm.maxnum.f32(float %920, float %931), !dbg !27
  %933 = xor i32 %922, 32, !dbg !27
  %934 = icmp samesign ult i32 %933, %925, !dbg !27
  %935 = select i1 %934, i32 %933, i32 %922, !dbg !27
  %936 = shl nuw nsw i32 %935, 2, !dbg !27
  %937 = bitcast float %932 to i32, !dbg !27
  %938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %936, i32 %937), !dbg !27
  %939 = bitcast i32 %938 to float, !dbg !27
  %940 = call float @llvm.maxnum.f32(float %932, float %939), !dbg !27
  %941 = extractvalue [2 x <1 x float>] %207, 0, !dbg !27
  %942 = insertelement <1 x float> poison, float %940, i64 0, !dbg !27
  %943 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %942, <1 x float> %941), !dbg !27
  %944 = call float @llvm.maxnum.f32(float %887, float 0xFFF0000000000000), !dbg !27, !frisk.local_reduction !1
  %local.reduce220 = call float @llvm.maxnum.f32(float %944, float %891), !dbg !27, !frisk.local_reduction !1
  %local.reduce221 = call float @llvm.maxnum.f32(float %888, float %892), !dbg !27, !frisk.local_reduction !1
  %local.reduce222 = call float @llvm.maxnum.f32(float %889, float %893), !dbg !27, !frisk.local_reduction !1
  %local.reduce223 = call float @llvm.maxnum.f32(float %890, float %894), !dbg !27, !frisk.local_reduction !1
  %local.reduce224 = call float @llvm.maxnum.f32(float %local.reduce220, float %895), !dbg !27, !frisk.local_reduction !1
  %local.reduce225 = call float @llvm.maxnum.f32(float %local.reduce221, float %896), !dbg !27, !frisk.local_reduction !1
  %local.reduce226 = call float @llvm.maxnum.f32(float %local.reduce222, float %897), !dbg !27, !frisk.local_reduction !1
  %local.reduce227 = call float @llvm.maxnum.f32(float %local.reduce223, float %898), !dbg !27, !frisk.local_reduction !1
  %local.reduce228 = call float @llvm.maxnum.f32(float %local.reduce224, float %899), !dbg !27, !frisk.local_reduction !1
  %local.reduce229 = call float @llvm.maxnum.f32(float %local.reduce225, float %900), !dbg !27, !frisk.local_reduction !1
  %local.reduce230 = call float @llvm.maxnum.f32(float %local.reduce226, float %901), !dbg !27, !frisk.local_reduction !1
  %local.reduce231 = call float @llvm.maxnum.f32(float %local.reduce227, float %902), !dbg !27, !frisk.local_reduction !1
  %local.reduce232 = call float @llvm.maxnum.f32(float %local.reduce228, float %903), !dbg !27, !frisk.local_reduction !1
  %local.reduce233 = call float @llvm.maxnum.f32(float %local.reduce229, float %904), !dbg !27, !frisk.local_reduction !1
  %local.reduce234 = call float @llvm.maxnum.f32(float %local.reduce230, float %905), !dbg !27, !frisk.local_reduction !1
  %local.reduce235 = call float @llvm.maxnum.f32(float %local.reduce231, float %906), !dbg !27, !frisk.local_reduction !1
  %local.reduce236 = call float @llvm.maxnum.f32(float %local.reduce232, float %907), !dbg !27, !frisk.local_reduction !1
  %local.reduce237 = call float @llvm.maxnum.f32(float %local.reduce233, float %908), !dbg !27, !frisk.local_reduction !1
  %local.reduce238 = call float @llvm.maxnum.f32(float %local.reduce234, float %909), !dbg !27, !frisk.local_reduction !1
  %local.reduce239 = call float @llvm.maxnum.f32(float %local.reduce235, float %910), !dbg !27, !frisk.local_reduction !1
  %local.reduce240 = call float @llvm.maxnum.f32(float %local.reduce236, float %911), !dbg !27, !frisk.local_reduction !1
  %local.reduce241 = call float @llvm.maxnum.f32(float %local.reduce237, float %912), !dbg !27, !frisk.local_reduction !1
  %local.reduce242 = call float @llvm.maxnum.f32(float %local.reduce238, float %913), !dbg !27, !frisk.local_reduction !1
  %local.reduce243 = call float @llvm.maxnum.f32(float %local.reduce239, float %914), !dbg !27, !frisk.local_reduction !1
  %local.reduce244 = call float @llvm.maxnum.f32(float %local.reduce240, float %915), !dbg !27, !frisk.local_reduction !1
  %local.reduce245 = call float @llvm.maxnum.f32(float %local.reduce241, float %916), !dbg !27, !frisk.local_reduction !1
  %local.reduce246 = call float @llvm.maxnum.f32(float %local.reduce242, float %917), !dbg !27, !frisk.local_reduction !1
  %local.reduce247 = call float @llvm.maxnum.f32(float %local.reduce243, float %918), !dbg !27, !frisk.local_reduction !1
  %local.reduce248 = call float @llvm.maxnum.f32(float %local.reduce244, float %local.reduce245), !dbg !27, !frisk.local_reduction !1
  %local.reduce249 = call float @llvm.maxnum.f32(float %local.reduce246, float %local.reduce247), !dbg !27, !frisk.local_reduction !1
  %945 = call float @llvm.maxnum.f32(float %local.reduce248, float %local.reduce249), !dbg !27, !frisk.local_reduction !1
  %946 = bitcast float %945 to i32, !dbg !27
  %947 = call i32 @llvm.amdgcn.ds.bpermute(i32 %928, i32 %946), !dbg !27
  %948 = bitcast i32 %947 to float, !dbg !27
  %949 = call float @llvm.maxnum.f32(float %945, float %948), !dbg !27
  %950 = bitcast float %949 to i32, !dbg !27
  %951 = call i32 @llvm.amdgcn.ds.bpermute(i32 %936, i32 %950), !dbg !27
  %952 = bitcast i32 %951 to float, !dbg !27
  %953 = call float @llvm.maxnum.f32(float %949, float %952), !dbg !27
  %954 = extractvalue [2 x <1 x float>] %207, 1, !dbg !27
  %955 = insertelement <1 x float> poison, float %953, i64 0, !dbg !27
  %956 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %955, <1 x float> %954), !dbg !27
  %957 = shufflevector <1 x float> %943, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %958 = fsub <4 x float> %839, %957, !dbg !28
  %959 = fsub <4 x float> %840, %957, !dbg !28
  %960 = fsub <4 x float> %841, %957, !dbg !28
  %961 = fsub <4 x float> %842, %957, !dbg !28
  %962 = fsub <4 x float> %843, %957, !dbg !28
  %963 = fsub <4 x float> %844, %957, !dbg !28
  %964 = fsub <4 x float> %845, %957, !dbg !28
  %965 = fsub <4 x float> %846, %957, !dbg !28
  %966 = shufflevector <1 x float> %956, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %967 = fsub <4 x float> %847, %966, !dbg !28
  %968 = fsub <4 x float> %848, %966, !dbg !28
  %969 = fsub <4 x float> %849, %966, !dbg !28
  %970 = fsub <4 x float> %850, %966, !dbg !28
  %971 = fsub <4 x float> %851, %966, !dbg !28
  %972 = fsub <4 x float> %852, %966, !dbg !28
  %973 = fsub <4 x float> %853, %966, !dbg !28
  %974 = fsub <4 x float> %854, %966, !dbg !28
  %975 = fsub <1 x float> %941, %943, !dbg !29
  %976 = extractelement <1 x float> %975, i64 0, !dbg !30
  %977 = call float @llvm.amdgcn.exp2.f32(float %976), !dbg !30
  %978 = insertelement <4 x float> poison, float %977, i64 0, !dbg !35
  %979 = shufflevector <4 x float> %978, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %980 = extractvalue [2 x <1 x float>] %206, 0, !dbg !31
  %981 = extractelement <1 x float> %980, i64 0, !dbg !31
  %982 = fmul float %981, %977, !dbg !31
  %983 = fsub <1 x float> %954, %956, !dbg !29
  %984 = extractelement <1 x float> %983, i64 0, !dbg !30
  %985 = call float @llvm.amdgcn.exp2.f32(float %984), !dbg !30
  %986 = insertelement <4 x float> poison, float %985, i64 0, !dbg !35
  %987 = shufflevector <4 x float> %986, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %988 = extractvalue [2 x <1 x float>] %206, 1, !dbg !31
  %989 = extractelement <1 x float> %988, i64 0, !dbg !31
  %990 = fmul float %989, %985, !dbg !31
  %991 = extractelement <4 x float> %958, i64 0, !dbg !32
  %992 = call float @llvm.amdgcn.exp2.f32(float %991), !dbg !32
  %993 = insertelement <4 x float> poison, float %992, i64 0, !dbg !34
  %994 = extractelement <4 x float> %958, i64 1, !dbg !32
  %995 = call float @llvm.amdgcn.exp2.f32(float %994), !dbg !32
  %996 = insertelement <4 x float> %993, float %995, i64 1, !dbg !34
  %997 = extractelement <4 x float> %958, i64 2, !dbg !32
  %998 = call float @llvm.amdgcn.exp2.f32(float %997), !dbg !32
  %999 = insertelement <4 x float> %996, float %998, i64 2, !dbg !34
  %1000 = extractelement <4 x float> %958, i64 3, !dbg !32
  %1001 = call float @llvm.amdgcn.exp2.f32(float %1000), !dbg !32
  %1002 = insertelement <4 x float> %999, float %1001, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1002) #9, !dbg !34, !frisk.register_lifetimes !1
  %1003 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1003) #9, !dbg !32, !frisk.register_lifetimes !1
  %1004 = extractelement <4 x float> %959, i64 0, !dbg !32
  %1005 = call float @llvm.amdgcn.exp2.f32(float %1004), !dbg !32
  %1006 = insertelement <4 x float> poison, float %1005, i64 0, !dbg !34
  %local.reduce189 = fadd float %992, %1005, !dbg !33, !frisk.local_reduction !1
  %1007 = extractelement <4 x float> %959, i64 1, !dbg !32
  %1008 = call float @llvm.amdgcn.exp2.f32(float %1007), !dbg !32
  %1009 = insertelement <4 x float> %1006, float %1008, i64 1, !dbg !34
  %local.reduce190 = fadd float %995, %1008, !dbg !33, !frisk.local_reduction !1
  %1010 = extractelement <4 x float> %959, i64 2, !dbg !32
  %1011 = call float @llvm.amdgcn.exp2.f32(float %1010), !dbg !32
  %1012 = insertelement <4 x float> %1009, float %1011, i64 2, !dbg !34
  %local.reduce191 = fadd float %998, %1011, !dbg !33, !frisk.local_reduction !1
  %1013 = extractelement <4 x float> %959, i64 3, !dbg !32
  %1014 = call float @llvm.amdgcn.exp2.f32(float %1013), !dbg !32
  %1015 = insertelement <4 x float> %1012, float %1014, i64 3, !dbg !34
  %math.result475 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1015) #9, !dbg !34, !frisk.register_lifetimes !1
  %1016 = fptrunc <4 x float> %math.result475 to <4 x half>, !dbg !34
  %math.packed476 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1016) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce192 = fadd float %1001, %1014, !dbg !33, !frisk.local_reduction !1
  %1017 = extractelement <4 x float> %960, i64 0, !dbg !32
  %1018 = call float @llvm.amdgcn.exp2.f32(float %1017), !dbg !32
  %1019 = insertelement <4 x float> poison, float %1018, i64 0, !dbg !34
  %local.reduce193 = fadd float %local.reduce189, %1018, !dbg !33, !frisk.local_reduction !1
  %1020 = extractelement <4 x float> %960, i64 1, !dbg !32
  %1021 = call float @llvm.amdgcn.exp2.f32(float %1020), !dbg !32
  %1022 = insertelement <4 x float> %1019, float %1021, i64 1, !dbg !34
  %local.reduce194 = fadd float %local.reduce190, %1021, !dbg !33, !frisk.local_reduction !1
  %1023 = extractelement <4 x float> %960, i64 2, !dbg !32
  %1024 = call float @llvm.amdgcn.exp2.f32(float %1023), !dbg !32
  %1025 = insertelement <4 x float> %1022, float %1024, i64 2, !dbg !34
  %local.reduce195 = fadd float %local.reduce191, %1024, !dbg !33, !frisk.local_reduction !1
  %1026 = extractelement <4 x float> %960, i64 3, !dbg !32
  %1027 = call float @llvm.amdgcn.exp2.f32(float %1026), !dbg !32
  %1028 = insertelement <4 x float> %1025, float %1027, i64 3, !dbg !34
  %math.result477 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1028) #9, !dbg !34, !frisk.register_lifetimes !1
  %1029 = fptrunc <4 x float> %math.result477 to <4 x half>, !dbg !34
  %math.packed478 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1029) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce196 = fadd float %local.reduce192, %1027, !dbg !33, !frisk.local_reduction !1
  %1030 = extractelement <4 x float> %961, i64 0, !dbg !32
  %1031 = call float @llvm.amdgcn.exp2.f32(float %1030), !dbg !32
  %1032 = insertelement <4 x float> poison, float %1031, i64 0, !dbg !34
  %local.reduce197 = fadd float %local.reduce193, %1031, !dbg !33, !frisk.local_reduction !1
  %1033 = extractelement <4 x float> %961, i64 1, !dbg !32
  %1034 = call float @llvm.amdgcn.exp2.f32(float %1033), !dbg !32
  %1035 = insertelement <4 x float> %1032, float %1034, i64 1, !dbg !34
  %local.reduce198 = fadd float %local.reduce194, %1034, !dbg !33, !frisk.local_reduction !1
  %1036 = extractelement <4 x float> %961, i64 2, !dbg !32
  %1037 = call float @llvm.amdgcn.exp2.f32(float %1036), !dbg !32
  %1038 = insertelement <4 x float> %1035, float %1037, i64 2, !dbg !34
  %local.reduce199 = fadd float %local.reduce195, %1037, !dbg !33, !frisk.local_reduction !1
  %1039 = extractelement <4 x float> %961, i64 3, !dbg !32
  %1040 = call float @llvm.amdgcn.exp2.f32(float %1039), !dbg !32
  %1041 = insertelement <4 x float> %1038, float %1040, i64 3, !dbg !34
  %math.result479 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1041) #9, !dbg !34, !frisk.register_lifetimes !1
  %1042 = fptrunc <4 x float> %math.result479 to <4 x half>, !dbg !34
  %math.packed480 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1042) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce200 = fadd float %local.reduce196, %1040, !dbg !33, !frisk.local_reduction !1
  %1043 = extractelement <4 x float> %962, i64 0, !dbg !32
  %1044 = call float @llvm.amdgcn.exp2.f32(float %1043), !dbg !32
  %1045 = insertelement <4 x float> poison, float %1044, i64 0, !dbg !34
  %local.reduce201 = fadd float %local.reduce197, %1044, !dbg !33, !frisk.local_reduction !1
  %1046 = extractelement <4 x float> %962, i64 1, !dbg !32
  %1047 = call float @llvm.amdgcn.exp2.f32(float %1046), !dbg !32
  %1048 = insertelement <4 x float> %1045, float %1047, i64 1, !dbg !34
  %local.reduce202 = fadd float %local.reduce198, %1047, !dbg !33, !frisk.local_reduction !1
  %1049 = extractelement <4 x float> %962, i64 2, !dbg !32
  %1050 = call float @llvm.amdgcn.exp2.f32(float %1049), !dbg !32
  %1051 = insertelement <4 x float> %1048, float %1050, i64 2, !dbg !34
  %local.reduce203 = fadd float %local.reduce199, %1050, !dbg !33, !frisk.local_reduction !1
  %1052 = extractelement <4 x float> %962, i64 3, !dbg !32
  %1053 = call float @llvm.amdgcn.exp2.f32(float %1052), !dbg !32
  %1054 = insertelement <4 x float> %1051, float %1053, i64 3, !dbg !34
  %math.result481 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1054) #9, !dbg !34, !frisk.register_lifetimes !1
  %1055 = fptrunc <4 x float> %math.result481 to <4 x half>, !dbg !34
  %math.packed482 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1055) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce204 = fadd float %local.reduce200, %1053, !dbg !33, !frisk.local_reduction !1
  %1056 = extractelement <4 x float> %963, i64 0, !dbg !32
  %1057 = call float @llvm.amdgcn.exp2.f32(float %1056), !dbg !32
  %1058 = insertelement <4 x float> poison, float %1057, i64 0, !dbg !34
  %local.reduce205 = fadd float %local.reduce201, %1057, !dbg !33, !frisk.local_reduction !1
  %1059 = extractelement <4 x float> %963, i64 1, !dbg !32
  %1060 = call float @llvm.amdgcn.exp2.f32(float %1059), !dbg !32
  %1061 = insertelement <4 x float> %1058, float %1060, i64 1, !dbg !34
  %local.reduce206 = fadd float %local.reduce202, %1060, !dbg !33, !frisk.local_reduction !1
  %1062 = extractelement <4 x float> %963, i64 2, !dbg !32
  %1063 = call float @llvm.amdgcn.exp2.f32(float %1062), !dbg !32
  %1064 = insertelement <4 x float> %1061, float %1063, i64 2, !dbg !34
  %local.reduce207 = fadd float %local.reduce203, %1063, !dbg !33, !frisk.local_reduction !1
  %1065 = extractelement <4 x float> %963, i64 3, !dbg !32
  %1066 = call float @llvm.amdgcn.exp2.f32(float %1065), !dbg !32
  %1067 = insertelement <4 x float> %1064, float %1066, i64 3, !dbg !34
  %math.result483 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1067) #9, !dbg !34, !frisk.register_lifetimes !1
  %1068 = fptrunc <4 x float> %math.result483 to <4 x half>, !dbg !34
  %math.packed484 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1068) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce208 = fadd float %local.reduce204, %1066, !dbg !33, !frisk.local_reduction !1
  %1069 = extractelement <4 x float> %964, i64 0, !dbg !32
  %1070 = call float @llvm.amdgcn.exp2.f32(float %1069), !dbg !32
  %1071 = insertelement <4 x float> poison, float %1070, i64 0, !dbg !34
  %local.reduce209 = fadd float %local.reduce205, %1070, !dbg !33, !frisk.local_reduction !1
  %1072 = extractelement <4 x float> %964, i64 1, !dbg !32
  %1073 = call float @llvm.amdgcn.exp2.f32(float %1072), !dbg !32
  %1074 = insertelement <4 x float> %1071, float %1073, i64 1, !dbg !34
  %local.reduce210 = fadd float %local.reduce206, %1073, !dbg !33, !frisk.local_reduction !1
  %1075 = extractelement <4 x float> %964, i64 2, !dbg !32
  %1076 = call float @llvm.amdgcn.exp2.f32(float %1075), !dbg !32
  %1077 = insertelement <4 x float> %1074, float %1076, i64 2, !dbg !34
  %local.reduce211 = fadd float %local.reduce207, %1076, !dbg !33, !frisk.local_reduction !1
  %1078 = extractelement <4 x float> %964, i64 3, !dbg !32
  %1079 = call float @llvm.amdgcn.exp2.f32(float %1078), !dbg !32
  %1080 = insertelement <4 x float> %1077, float %1079, i64 3, !dbg !34
  %math.result485 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1080) #9, !dbg !34, !frisk.register_lifetimes !1
  %1081 = fptrunc <4 x float> %math.result485 to <4 x half>, !dbg !34
  %math.packed486 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1081) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce212 = fadd float %local.reduce208, %1079, !dbg !33, !frisk.local_reduction !1
  %1082 = extractelement <4 x float> %965, i64 0, !dbg !32
  %1083 = call float @llvm.amdgcn.exp2.f32(float %1082), !dbg !32
  %1084 = insertelement <4 x float> poison, float %1083, i64 0, !dbg !34
  %local.reduce213 = fadd float %local.reduce209, %1083, !dbg !33, !frisk.local_reduction !1
  %1085 = extractelement <4 x float> %965, i64 1, !dbg !32
  %1086 = call float @llvm.amdgcn.exp2.f32(float %1085), !dbg !32
  %1087 = insertelement <4 x float> %1084, float %1086, i64 1, !dbg !34
  %local.reduce214 = fadd float %local.reduce210, %1086, !dbg !33, !frisk.local_reduction !1
  %local.reduce217 = fadd float %local.reduce213, %local.reduce214, !dbg !33, !frisk.local_reduction !1
  %1088 = extractelement <4 x float> %965, i64 2, !dbg !32
  %1089 = call float @llvm.amdgcn.exp2.f32(float %1088), !dbg !32
  %1090 = insertelement <4 x float> %1087, float %1089, i64 2, !dbg !34
  %local.reduce215 = fadd float %local.reduce211, %1089, !dbg !33, !frisk.local_reduction !1
  %1091 = extractelement <4 x float> %965, i64 3, !dbg !32
  %1092 = call float @llvm.amdgcn.exp2.f32(float %1091), !dbg !32
  %1093 = insertelement <4 x float> %1090, float %1092, i64 3, !dbg !34
  %math.result487 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1093) #9, !dbg !34, !frisk.register_lifetimes !1
  %1094 = fptrunc <4 x float> %math.result487 to <4 x half>, !dbg !34
  %math.packed488 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1094) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce216 = fadd float %local.reduce212, %1092, !dbg !33, !frisk.local_reduction !1
  %local.reduce218 = fadd float %local.reduce215, %local.reduce216, !dbg !33, !frisk.local_reduction !1
  %1095 = fadd float %local.reduce217, %local.reduce218, !dbg !33, !frisk.local_reduction !1
  %1096 = fadd float %1095, %982, !dbg !33
  %1097 = insertelement <1 x float> poison, float %1096, i64 0, !dbg !7
  %1098 = insertvalue [2 x <1 x float>] poison, <1 x float> %1097, 0, !dbg !7
  %1099 = extractelement <4 x float> %967, i64 0, !dbg !32
  %1100 = call float @llvm.amdgcn.exp2.f32(float %1099), !dbg !32
  %1101 = insertelement <4 x float> poison, float %1100, i64 0, !dbg !34
  %1102 = extractelement <4 x float> %967, i64 1, !dbg !32
  %1103 = call float @llvm.amdgcn.exp2.f32(float %1102), !dbg !32
  %1104 = insertelement <4 x float> %1101, float %1103, i64 1, !dbg !34
  %1105 = extractelement <4 x float> %967, i64 2, !dbg !32
  %1106 = call float @llvm.amdgcn.exp2.f32(float %1105), !dbg !32
  %1107 = insertelement <4 x float> %1104, float %1106, i64 2, !dbg !34
  %1108 = extractelement <4 x float> %967, i64 3, !dbg !32
  %1109 = call float @llvm.amdgcn.exp2.f32(float %1108), !dbg !32
  %1110 = insertelement <4 x float> %1107, float %1109, i64 3, !dbg !34
  %math.result489 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1110) #9, !dbg !34, !frisk.register_lifetimes !1
  %1111 = fptrunc <4 x float> %math.result489 to <4 x half>, !dbg !34
  %math.packed490 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1111) #9, !dbg !32, !frisk.register_lifetimes !1
  %1112 = extractelement <4 x float> %968, i64 0, !dbg !32
  %1113 = call float @llvm.amdgcn.exp2.f32(float %1112), !dbg !32
  %1114 = insertelement <4 x float> poison, float %1113, i64 0, !dbg !34
  %local.reduce = fadd float %1100, %1113, !dbg !33, !frisk.local_reduction !1
  %1115 = extractelement <4 x float> %968, i64 1, !dbg !32
  %1116 = call float @llvm.amdgcn.exp2.f32(float %1115), !dbg !32
  %1117 = insertelement <4 x float> %1114, float %1116, i64 1, !dbg !34
  %local.reduce159 = fadd float %1103, %1116, !dbg !33, !frisk.local_reduction !1
  %1118 = extractelement <4 x float> %968, i64 2, !dbg !32
  %1119 = call float @llvm.amdgcn.exp2.f32(float %1118), !dbg !32
  %1120 = insertelement <4 x float> %1117, float %1119, i64 2, !dbg !34
  %local.reduce160 = fadd float %1106, %1119, !dbg !33, !frisk.local_reduction !1
  %1121 = extractelement <4 x float> %968, i64 3, !dbg !32
  %1122 = call float @llvm.amdgcn.exp2.f32(float %1121), !dbg !32
  %1123 = insertelement <4 x float> %1120, float %1122, i64 3, !dbg !34
  %math.result491 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1123) #9, !dbg !34, !frisk.register_lifetimes !1
  %1124 = fptrunc <4 x float> %math.result491 to <4 x half>, !dbg !34
  %math.packed492 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1124) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce161 = fadd float %1109, %1122, !dbg !33, !frisk.local_reduction !1
  %1125 = extractelement <4 x float> %969, i64 0, !dbg !32
  %1126 = call float @llvm.amdgcn.exp2.f32(float %1125), !dbg !32
  %1127 = insertelement <4 x float> poison, float %1126, i64 0, !dbg !34
  %local.reduce162 = fadd float %local.reduce, %1126, !dbg !33, !frisk.local_reduction !1
  %1128 = extractelement <4 x float> %969, i64 1, !dbg !32
  %1129 = call float @llvm.amdgcn.exp2.f32(float %1128), !dbg !32
  %1130 = insertelement <4 x float> %1127, float %1129, i64 1, !dbg !34
  %local.reduce163 = fadd float %local.reduce159, %1129, !dbg !33, !frisk.local_reduction !1
  %1131 = extractelement <4 x float> %969, i64 2, !dbg !32
  %1132 = call float @llvm.amdgcn.exp2.f32(float %1131), !dbg !32
  %1133 = insertelement <4 x float> %1130, float %1132, i64 2, !dbg !34
  %local.reduce164 = fadd float %local.reduce160, %1132, !dbg !33, !frisk.local_reduction !1
  %1134 = extractelement <4 x float> %969, i64 3, !dbg !32
  %1135 = call float @llvm.amdgcn.exp2.f32(float %1134), !dbg !32
  %1136 = insertelement <4 x float> %1133, float %1135, i64 3, !dbg !34
  %math.result493 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1136) #9, !dbg !34, !frisk.register_lifetimes !1
  %1137 = fptrunc <4 x float> %math.result493 to <4 x half>, !dbg !34
  %math.packed494 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1137) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce165 = fadd float %local.reduce161, %1135, !dbg !33, !frisk.local_reduction !1
  %1138 = extractelement <4 x float> %970, i64 0, !dbg !32
  %1139 = call float @llvm.amdgcn.exp2.f32(float %1138), !dbg !32
  %1140 = insertelement <4 x float> poison, float %1139, i64 0, !dbg !34
  %local.reduce166 = fadd float %local.reduce162, %1139, !dbg !33, !frisk.local_reduction !1
  %1141 = extractelement <4 x float> %970, i64 1, !dbg !32
  %1142 = call float @llvm.amdgcn.exp2.f32(float %1141), !dbg !32
  %1143 = insertelement <4 x float> %1140, float %1142, i64 1, !dbg !34
  %local.reduce167 = fadd float %local.reduce163, %1142, !dbg !33, !frisk.local_reduction !1
  %1144 = extractelement <4 x float> %970, i64 2, !dbg !32
  %1145 = call float @llvm.amdgcn.exp2.f32(float %1144), !dbg !32
  %1146 = insertelement <4 x float> %1143, float %1145, i64 2, !dbg !34
  %local.reduce168 = fadd float %local.reduce164, %1145, !dbg !33, !frisk.local_reduction !1
  %1147 = extractelement <4 x float> %970, i64 3, !dbg !32
  %1148 = call float @llvm.amdgcn.exp2.f32(float %1147), !dbg !32
  %1149 = insertelement <4 x float> %1146, float %1148, i64 3, !dbg !34
  %math.result495 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1149) #9, !dbg !34, !frisk.register_lifetimes !1
  %1150 = fptrunc <4 x float> %math.result495 to <4 x half>, !dbg !34
  %math.packed496 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1150) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce169 = fadd float %local.reduce165, %1148, !dbg !33, !frisk.local_reduction !1
  %1151 = extractelement <4 x float> %971, i64 0, !dbg !32
  %1152 = call float @llvm.amdgcn.exp2.f32(float %1151), !dbg !32
  %1153 = insertelement <4 x float> poison, float %1152, i64 0, !dbg !34
  %local.reduce170 = fadd float %local.reduce166, %1152, !dbg !33, !frisk.local_reduction !1
  %1154 = extractelement <4 x float> %971, i64 1, !dbg !32
  %1155 = call float @llvm.amdgcn.exp2.f32(float %1154), !dbg !32
  %1156 = insertelement <4 x float> %1153, float %1155, i64 1, !dbg !34
  %local.reduce171 = fadd float %local.reduce167, %1155, !dbg !33, !frisk.local_reduction !1
  %1157 = extractelement <4 x float> %971, i64 2, !dbg !32
  %1158 = call float @llvm.amdgcn.exp2.f32(float %1157), !dbg !32
  %1159 = insertelement <4 x float> %1156, float %1158, i64 2, !dbg !34
  %local.reduce172 = fadd float %local.reduce168, %1158, !dbg !33, !frisk.local_reduction !1
  %1160 = extractelement <4 x float> %971, i64 3, !dbg !32
  %1161 = call float @llvm.amdgcn.exp2.f32(float %1160), !dbg !32
  %1162 = insertelement <4 x float> %1159, float %1161, i64 3, !dbg !34
  %math.result497 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1162) #9, !dbg !34, !frisk.register_lifetimes !1
  %1163 = fptrunc <4 x float> %math.result497 to <4 x half>, !dbg !34
  %math.packed498 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1163) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce173 = fadd float %local.reduce169, %1161, !dbg !33, !frisk.local_reduction !1
  %1164 = extractelement <4 x float> %972, i64 0, !dbg !32
  %1165 = call float @llvm.amdgcn.exp2.f32(float %1164), !dbg !32
  %1166 = insertelement <4 x float> poison, float %1165, i64 0, !dbg !34
  %local.reduce174 = fadd float %local.reduce170, %1165, !dbg !33, !frisk.local_reduction !1
  %1167 = extractelement <4 x float> %972, i64 1, !dbg !32
  %1168 = call float @llvm.amdgcn.exp2.f32(float %1167), !dbg !32
  %1169 = insertelement <4 x float> %1166, float %1168, i64 1, !dbg !34
  %local.reduce175 = fadd float %local.reduce171, %1168, !dbg !33, !frisk.local_reduction !1
  %1170 = extractelement <4 x float> %972, i64 2, !dbg !32
  %1171 = call float @llvm.amdgcn.exp2.f32(float %1170), !dbg !32
  %1172 = insertelement <4 x float> %1169, float %1171, i64 2, !dbg !34
  %local.reduce176 = fadd float %local.reduce172, %1171, !dbg !33, !frisk.local_reduction !1
  %1173 = extractelement <4 x float> %972, i64 3, !dbg !32
  %1174 = call float @llvm.amdgcn.exp2.f32(float %1173), !dbg !32
  %1175 = insertelement <4 x float> %1172, float %1174, i64 3, !dbg !34
  %math.result499 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1175) #9, !dbg !34, !frisk.register_lifetimes !1
  %1176 = fptrunc <4 x float> %math.result499 to <4 x half>, !dbg !34
  %math.packed500 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1176) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce177 = fadd float %local.reduce173, %1174, !dbg !33, !frisk.local_reduction !1
  %1177 = extractelement <4 x float> %973, i64 0, !dbg !32
  %1178 = call float @llvm.amdgcn.exp2.f32(float %1177), !dbg !32
  %1179 = insertelement <4 x float> poison, float %1178, i64 0, !dbg !34
  %local.reduce178 = fadd float %local.reduce174, %1178, !dbg !33, !frisk.local_reduction !1
  %1180 = extractelement <4 x float> %973, i64 1, !dbg !32
  %1181 = call float @llvm.amdgcn.exp2.f32(float %1180), !dbg !32
  %1182 = insertelement <4 x float> %1179, float %1181, i64 1, !dbg !34
  %local.reduce179 = fadd float %local.reduce175, %1181, !dbg !33, !frisk.local_reduction !1
  %1183 = extractelement <4 x float> %973, i64 2, !dbg !32
  %1184 = call float @llvm.amdgcn.exp2.f32(float %1183), !dbg !32
  %1185 = insertelement <4 x float> %1182, float %1184, i64 2, !dbg !34
  %local.reduce180 = fadd float %local.reduce176, %1184, !dbg !33, !frisk.local_reduction !1
  %1186 = extractelement <4 x float> %973, i64 3, !dbg !32
  %1187 = call float @llvm.amdgcn.exp2.f32(float %1186), !dbg !32
  %1188 = insertelement <4 x float> %1185, float %1187, i64 3, !dbg !34
  %math.result501 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1188) #9, !dbg !34, !frisk.register_lifetimes !1
  %1189 = fptrunc <4 x float> %math.result501 to <4 x half>, !dbg !34
  %math.packed502 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1189) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce181 = fadd float %local.reduce177, %1187, !dbg !33, !frisk.local_reduction !1
  %1190 = extractelement <4 x float> %974, i64 0, !dbg !32
  %1191 = call float @llvm.amdgcn.exp2.f32(float %1190), !dbg !32
  %1192 = insertelement <4 x float> poison, float %1191, i64 0, !dbg !34
  %local.reduce182 = fadd float %local.reduce178, %1191, !dbg !33, !frisk.local_reduction !1
  %1193 = extractelement <4 x float> %974, i64 1, !dbg !32
  %1194 = call float @llvm.amdgcn.exp2.f32(float %1193), !dbg !32
  %1195 = insertelement <4 x float> %1192, float %1194, i64 1, !dbg !34
  %local.reduce183 = fadd float %local.reduce179, %1194, !dbg !33, !frisk.local_reduction !1
  %local.reduce186 = fadd float %local.reduce182, %local.reduce183, !dbg !33, !frisk.local_reduction !1
  %1196 = extractelement <4 x float> %974, i64 2, !dbg !32
  %1197 = call float @llvm.amdgcn.exp2.f32(float %1196), !dbg !32
  %1198 = insertelement <4 x float> %1195, float %1197, i64 2, !dbg !34
  %local.reduce184 = fadd float %local.reduce180, %1197, !dbg !33, !frisk.local_reduction !1
  %1199 = extractelement <4 x float> %974, i64 3, !dbg !32
  %1200 = call float @llvm.amdgcn.exp2.f32(float %1199), !dbg !32
  %1201 = insertelement <4 x float> %1198, float %1200, i64 3, !dbg !34
  %math.result503 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1201) #9, !dbg !34, !frisk.register_lifetimes !1
  %1202 = fptrunc <4 x float> %math.result503 to <4 x half>, !dbg !34
  %math.packed504 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1202) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce185 = fadd float %local.reduce181, %1200, !dbg !33, !frisk.local_reduction !1
  %local.reduce187 = fadd float %local.reduce184, %local.reduce185, !dbg !33, !frisk.local_reduction !1
  %1203 = fadd float %local.reduce186, %local.reduce187, !dbg !33, !frisk.local_reduction !1
  %1204 = fadd float %1203, %990, !dbg !33
  %1205 = insertelement <1 x float> poison, float %1204, i64 0, !dbg !7
  %1206 = insertvalue [2 x <1 x float>] %1098, <1 x float> %1205, 1, !dbg !7
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
  %1207 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #0, !dbg !14
  %1208 = bitcast <4 x i32> %1207 to <8 x half>, !dbg !14
  %address.local1038 = zext nneg i32 %address.thread1030 to i64, !dbg !8
  %address.local1039 = and i64 %address.local1038, 60, !dbg !14
  %address.local1040 = add nuw nsw i64 %address.local1039, %address.local1038, !dbg !14
  %address.local1041 = shl nuw nsw i64 %address.local1040, 3, !dbg !14
  %address.local1042 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1043 = lshr i64 %address.local1038, 4, !dbg !14
  %address.local1044 = and i64 %address.local1043, 3, !dbg !14
  %address.local1045 = mul nuw nsw i64 %address.local1044, 384, !dbg !14
  %address.local1046 = getelementptr [2 x i8], ptr addrspace(3) %address.local1042, i64 %address.local1045, !dbg !14
  %address.local1047 = lshr i64 %address.local1038, 2, !dbg !17
  %address.local1048 = mul nsw i64 %address.local1047, -32, !dbg !14
  %address.local1049 = getelementptr [2 x i8], ptr addrspace(3) %address.local1046, i64 %address.local1048, !dbg !14
  %address.local1050 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 256, !dbg !14
  %1209 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1210 = bitcast <4 x i32> %1209 to <8 x half>, !dbg !14
  %address.local1051 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1052 = getelementptr [2 x i8], ptr addrspace(3) %address.local1051, i64 %address.local1045, !dbg !14
  %address.local1053 = getelementptr [2 x i8], ptr addrspace(3) %address.local1052, i64 %address.local1048, !dbg !14
  %address.local1054 = getelementptr i8, ptr addrspace(3) %address.local1053, i64 512, !dbg !14
  %1211 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1212 = bitcast <4 x i32> %1211 to <8 x half>, !dbg !14
  %address.local1055 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1056 = getelementptr [2 x i8], ptr addrspace(3) %address.local1055, i64 %address.local1045, !dbg !14
  %address.local1057 = getelementptr [2 x i8], ptr addrspace(3) %address.local1056, i64 %address.local1048, !dbg !14
  %address.local1058 = getelementptr i8, ptr addrspace(3) %address.local1057, i64 768, !dbg !14
  %1213 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1214 = bitcast <4 x i32> %1213 to <8 x half>, !dbg !14
  %address.local1059 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1036, !dbg !14
  %address.local1060 = getelementptr i8, ptr addrspace(3) %address.local1059, i64 4096, !dbg !14
  %1215 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #0, !dbg !14
  %1216 = bitcast <4 x i32> %1215 to <8 x half>, !dbg !14
  %address.local1061 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1062 = getelementptr [2 x i8], ptr addrspace(3) %address.local1061, i64 %address.local1045, !dbg !14
  %address.local1063 = getelementptr [2 x i8], ptr addrspace(3) %address.local1062, i64 %address.local1048, !dbg !14
  %address.local1064 = getelementptr i8, ptr addrspace(3) %address.local1063, i64 4352, !dbg !14
  %1217 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1218 = bitcast <4 x i32> %1217 to <8 x half>, !dbg !14
  %address.local1065 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1066 = getelementptr [2 x i8], ptr addrspace(3) %address.local1065, i64 %address.local1045, !dbg !14
  %address.local1067 = getelementptr [2 x i8], ptr addrspace(3) %address.local1066, i64 %address.local1048, !dbg !14
  %address.local1068 = getelementptr i8, ptr addrspace(3) %address.local1067, i64 4608, !dbg !14
  %1219 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #0, !dbg !14
  %1220 = bitcast <4 x i32> %1219 to <8 x half>, !dbg !14
  %address.local1069 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1041, !dbg !14
  %address.local1070 = getelementptr [2 x i8], ptr addrspace(3) %address.local1069, i64 %address.local1045, !dbg !14
  %address.local1071 = getelementptr [2 x i8], ptr addrspace(3) %address.local1070, i64 %address.local1048, !dbg !14
  %address.local1072 = getelementptr i8, ptr addrspace(3) %address.local1071, i64 4864, !dbg !14
  %1221 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #0, !dbg !14
  %1222 = bitcast <4 x i32> %1221 to <8 x half>, !dbg !14
  %1223 = add nuw nsw i64 %216, %16, !dbg !18
  %1224 = shl i64 %1223, 19, !dbg !18
  %1225 = and i64 %1224, 16252928, !dbg !18
  %1226 = add i64 %1225, %222, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1227 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1226, !dbg !18
  %1228 = getelementptr i8, ptr addrspace(1) %1227, i64 64, !dbg !18
  %address.local1073 = shl nuw nsw i64 %address.local1038, 3, !dbg !8
  %address.local1074 = and i64 %address.local1073, 1536, !dbg !8
  %address.local1075 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1074, !dbg !8
  %address.local1076 = ptrtoint ptr addrspace(3) %address.local1075 to i64, !dbg !8
  %address.local1077 = trunc i64 %address.local1076 to i32, !dbg !8
  %1229 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1077), !dbg !18
  %1230 = zext i32 %1229 to i64, !dbg !18
  %1231 = inttoptr i64 %1230 to ptr addrspace(3), !dbg !18
  %address.local1078 = shl nuw nsw i64 %address.local1038, 5, !dbg !18
  %address.local1079 = and i64 %address.local1078, 8064, !dbg !18
  %address.local1080 = shl nuw nsw i64 %address.local1038, 3, !dbg !18
  %address.local1081 = and i64 %address.local1080, 24, !dbg !18
  %address.local1082 = or disjoint i64 %address.local1079, %address.local1081, !dbg !18
  %address.local1083 = getelementptr [2 x i8], ptr addrspace(1) %1228, i64 %address.local1082, !dbg !18
  %1232 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1233 = ptrtoint ptr addrspace(1) %address.local1083 to i64, !dbg !18
  %1234 = sub i64 %1233, %1232, !dbg !18
  %dma.buffer.offset1726 = trunc i64 %1234 to i32, !dbg !18
  %1235 = trunc i64 %1232 to i32, !dbg !18
  %1236 = insertelement <4 x i32> poison, i32 %1235, i64 0, !dbg !18
  %1237 = lshr i64 %1232, 32, !dbg !18
  %1238 = trunc i64 %1237 to i32, !dbg !18
  %1239 = insertelement <4 x i32> %1236, i32 %1238, i64 1, !dbg !18
  %1240 = insertelement <4 x i32> %1239, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1727 = insertelement <4 x i32> %1240, i32 131072, i64 3, !dbg !18
  %1241 = ptrtoint ptr addrspace(3) %1231 to i32, !dbg !18
  %dma.buffer.lds1728 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1241), !dbg !18
  %1242 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1728, i32 %dma.buffer.offset1726, <4 x i32> %dma.buffer.rsrc1727) #10, !dbg !18
  %address.local1084 = or disjoint i64 %address.local1038, 256, !dbg !8
  %address.local1085 = shl nuw nsw i64 %address.local1084, 3, !dbg !8
  %address.local1086 = and i64 %address.local1085, 3584, !dbg !8
  %address.local1087 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1086, !dbg !8
  %address.local1088 = ptrtoint ptr addrspace(3) %address.local1087 to i64, !dbg !8
  %address.local1089 = trunc i64 %address.local1088 to i32, !dbg !8
  %1243 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1089), !dbg !18
  %1244 = zext i32 %1243 to i64, !dbg !18
  %1245 = inttoptr i64 %1244 to ptr addrspace(3), !dbg !18
  %address.local1090 = shl nuw nsw i64 %address.local1084, 5, !dbg !18
  %address.local1091 = and i64 %address.local1090, 16256, !dbg !18
  %address.local1092 = shl nuw nsw i64 %address.local1038, 3, !dbg !18
  %address.local1093 = and i64 %address.local1092, 24, !dbg !18
  %address.local1094 = or disjoint i64 %address.local1091, %address.local1093, !dbg !18
  %address.local1095 = getelementptr [2 x i8], ptr addrspace(1) %1228, i64 %address.local1094, !dbg !18
  %1246 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1247 = ptrtoint ptr addrspace(1) %address.local1095 to i64, !dbg !18
  %1248 = sub i64 %1247, %1246, !dbg !18
  %dma.buffer.offset1723 = trunc i64 %1248 to i32, !dbg !18
  %1249 = trunc i64 %1246 to i32, !dbg !18
  %1250 = insertelement <4 x i32> poison, i32 %1249, i64 0, !dbg !18
  %1251 = lshr i64 %1246, 32, !dbg !18
  %1252 = trunc i64 %1251 to i32, !dbg !18
  %1253 = insertelement <4 x i32> %1250, i32 %1252, i64 1, !dbg !18
  %1254 = insertelement <4 x i32> %1253, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1724 = insertelement <4 x i32> %1254, i32 131072, i64 3, !dbg !18
  %1255 = ptrtoint ptr addrspace(3) %1245 to i32, !dbg !18
  %dma.buffer.lds1725 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1255), !dbg !18
  %1256 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1725, i32 %dma.buffer.offset1723, <4 x i32> %dma.buffer.rsrc1724) #10, !dbg !18
  %1257 = shufflevector <8 x half> %1208, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1258 = extractvalue [2 x <8 x float>] %205, 0, !dbg !35
  %1259 = shufflevector <8 x float> %1258, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1260 = fmul <4 x float> %1259, %979, !dbg !35
  %1261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1257, <4 x float> %1260) #9, !dbg !14
  %1262 = shufflevector <8 x half> %1210, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1262, <4 x float> %1261) #9, !dbg !14
  %1264 = shufflevector <8 x half> %1212, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1264, <4 x float> %1263) #9, !dbg !14
  %1266 = shufflevector <8 x half> %1214, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1266, <4 x float> %1265) #9, !dbg !14
  %1268 = shufflevector <8 x half> %1216, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1268, <4 x float> %1267) #9, !dbg !14
  %1270 = shufflevector <8 x half> %1218, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1270, <4 x float> %1269) #9, !dbg !14
  %1272 = shufflevector <8 x half> %1220, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1272, <4 x float> %1271) #9, !dbg !14
  %1274 = shufflevector <8 x half> %1222, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1274, <4 x float> %1273) #9, !dbg !14
  %1276 = shufflevector <8 x half> %1208, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1277 = shufflevector <8 x float> %1258, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1278 = fmul <4 x float> %1277, %979, !dbg !35
  %1279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1276, <4 x float> %1278) #9, !dbg !14
  %1280 = shufflevector <8 x half> %1210, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1280, <4 x float> %1279) #9, !dbg !14
  %1282 = shufflevector <8 x half> %1212, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1282, <4 x float> %1281) #9, !dbg !14
  %1284 = shufflevector <8 x half> %1214, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1284, <4 x float> %1283) #9, !dbg !14
  %1286 = shufflevector <8 x half> %1216, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1286, <4 x float> %1285) #9, !dbg !14
  %1288 = shufflevector <8 x half> %1218, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1288, <4 x float> %1287) #9, !dbg !14
  %1290 = shufflevector <8 x half> %1220, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1290, <4 x float> %1289) #9, !dbg !14
  %1292 = shufflevector <8 x half> %1222, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1292, <4 x float> %1291) #9, !dbg !14
  %1294 = extractvalue [2 x <8 x float>] %205, 1, !dbg !35
  %1295 = shufflevector <8 x float> %1294, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1296 = fmul <4 x float> %1295, %987, !dbg !35
  %1297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1257, <4 x float> %1296) #9, !dbg !14
  %1298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1262, <4 x float> %1297) #9, !dbg !14
  %1299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1264, <4 x float> %1298) #9, !dbg !14
  %1300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1266, <4 x float> %1299) #9, !dbg !14
  %1301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1268, <4 x float> %1300) #9, !dbg !14
  %1302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1270, <4 x float> %1301) #9, !dbg !14
  %1303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1272, <4 x float> %1302) #9, !dbg !14
  %1304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1274, <4 x float> %1303) #9, !dbg !14
  %1305 = shufflevector <8 x float> %1294, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1306 = fmul <4 x float> %1305, %987, !dbg !35
  %1307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1276, <4 x float> %1306) #9, !dbg !14
  %1308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1280, <4 x float> %1307) #9, !dbg !14
  %1309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1282, <4 x float> %1308) #9, !dbg !14
  %1310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1284, <4 x float> %1309) #9, !dbg !14
  %1311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1286, <4 x float> %1310) #9, !dbg !14
  %1312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1288, <4 x float> %1311) #9, !dbg !14
  %1313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1290, <4 x float> %1312) #9, !dbg !14
  %1314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1292, <4 x float> %1313) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1315 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #0, !dbg !14
  %1316 = bitcast <4 x i32> %1315 to <8 x half>, !dbg !14
  %1317 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1318 = bitcast <4 x i32> %1317 to <8 x half>, !dbg !14
  %1319 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1320 = bitcast <4 x i32> %1319 to <8 x half>, !dbg !14
  %1321 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1322 = bitcast <4 x i32> %1321 to <8 x half>, !dbg !14
  %1323 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #0, !dbg !14
  %1324 = bitcast <4 x i32> %1323 to <8 x half>, !dbg !14
  %1325 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1326 = bitcast <4 x i32> %1325 to <8 x half>, !dbg !14
  %1327 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #0, !dbg !14
  %1328 = bitcast <4 x i32> %1327 to <8 x half>, !dbg !14
  %1329 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #0, !dbg !14
  %1330 = bitcast <4 x i32> %1329 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1331 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1226, !dbg !18
  %1332 = getelementptr i8, ptr addrspace(1) %1331, i64 128, !dbg !18
  %1333 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1077), !dbg !18
  %1334 = zext i32 %1333 to i64, !dbg !18
  %1335 = inttoptr i64 %1334 to ptr addrspace(3), !dbg !18
  %address.local1096 = getelementptr [2 x i8], ptr addrspace(1) %1332, i64 %address.local1082, !dbg !18
  %1336 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1337 = ptrtoint ptr addrspace(1) %address.local1096 to i64, !dbg !18
  %1338 = sub i64 %1337, %1336, !dbg !18
  %dma.buffer.offset1720 = trunc i64 %1338 to i32, !dbg !18
  %1339 = trunc i64 %1336 to i32, !dbg !18
  %1340 = insertelement <4 x i32> poison, i32 %1339, i64 0, !dbg !18
  %1341 = lshr i64 %1336, 32, !dbg !18
  %1342 = trunc i64 %1341 to i32, !dbg !18
  %1343 = insertelement <4 x i32> %1340, i32 %1342, i64 1, !dbg !18
  %1344 = insertelement <4 x i32> %1343, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1721 = insertelement <4 x i32> %1344, i32 131072, i64 3, !dbg !18
  %1345 = ptrtoint ptr addrspace(3) %1335 to i32, !dbg !18
  %dma.buffer.lds1722 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1345), !dbg !18
  %1346 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1722, i32 %dma.buffer.offset1720, <4 x i32> %dma.buffer.rsrc1721) #10, !dbg !18
  %1347 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1089), !dbg !18
  %1348 = zext i32 %1347 to i64, !dbg !18
  %1349 = inttoptr i64 %1348 to ptr addrspace(3), !dbg !18
  %address.local1097 = getelementptr [2 x i8], ptr addrspace(1) %1332, i64 %address.local1094, !dbg !18
  %1350 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1351 = ptrtoint ptr addrspace(1) %address.local1097 to i64, !dbg !18
  %1352 = sub i64 %1351, %1350, !dbg !18
  %dma.buffer.offset1717 = trunc i64 %1352 to i32, !dbg !18
  %1353 = trunc i64 %1350 to i32, !dbg !18
  %1354 = insertelement <4 x i32> poison, i32 %1353, i64 0, !dbg !18
  %1355 = lshr i64 %1350, 32, !dbg !18
  %1356 = trunc i64 %1355 to i32, !dbg !18
  %1357 = insertelement <4 x i32> %1354, i32 %1356, i64 1, !dbg !18
  %1358 = insertelement <4 x i32> %1357, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1718 = insertelement <4 x i32> %1358, i32 131072, i64 3, !dbg !18
  %1359 = ptrtoint ptr addrspace(3) %1349 to i32, !dbg !18
  %dma.buffer.lds1719 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1359), !dbg !18
  %1360 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1719, i32 %dma.buffer.offset1717, <4 x i32> %dma.buffer.rsrc1718) #10, !dbg !18
  %1361 = shufflevector <8 x half> %1316, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1362 = extractvalue [2 x <8 x float>] %208, 0, !dbg !35
  %1363 = shufflevector <8 x float> %1362, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1364 = fmul <4 x float> %1363, %979, !dbg !35
  %1365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1361, <4 x float> %1364) #9, !dbg !14
  %1366 = shufflevector <8 x half> %1318, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1366, <4 x float> %1365) #9, !dbg !14
  %1368 = shufflevector <8 x half> %1320, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1368, <4 x float> %1367) #9, !dbg !14
  %1370 = shufflevector <8 x half> %1322, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1370, <4 x float> %1369) #9, !dbg !14
  %1372 = shufflevector <8 x half> %1324, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1372, <4 x float> %1371) #9, !dbg !14
  %1374 = shufflevector <8 x half> %1326, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1374, <4 x float> %1373) #9, !dbg !14
  %1376 = shufflevector <8 x half> %1328, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1376, <4 x float> %1375) #9, !dbg !14
  %1378 = shufflevector <8 x half> %1330, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1378, <4 x float> %1377) #9, !dbg !14
  %1380 = shufflevector <8 x half> %1316, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1381 = shufflevector <8 x float> %1362, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1382 = fmul <4 x float> %1381, %979, !dbg !35
  %1383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1380, <4 x float> %1382) #9, !dbg !14
  %1384 = shufflevector <8 x half> %1318, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1384, <4 x float> %1383) #9, !dbg !14
  %1386 = shufflevector <8 x half> %1320, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1386, <4 x float> %1385) #9, !dbg !14
  %1388 = shufflevector <8 x half> %1322, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1388, <4 x float> %1387) #9, !dbg !14
  %1390 = shufflevector <8 x half> %1324, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1390, <4 x float> %1389) #9, !dbg !14
  %1392 = shufflevector <8 x half> %1326, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1392, <4 x float> %1391) #9, !dbg !14
  %1394 = shufflevector <8 x half> %1328, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1394, <4 x float> %1393) #9, !dbg !14
  %1396 = shufflevector <8 x half> %1330, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1396, <4 x float> %1395) #9, !dbg !14
  %1398 = extractvalue [2 x <8 x float>] %208, 1, !dbg !35
  %1399 = shufflevector <8 x float> %1398, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1400 = fmul <4 x float> %1399, %987, !dbg !35
  %1401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1361, <4 x float> %1400) #9, !dbg !14
  %1402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1366, <4 x float> %1401) #9, !dbg !14
  %1403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1368, <4 x float> %1402) #9, !dbg !14
  %1404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1370, <4 x float> %1403) #9, !dbg !14
  %1405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1372, <4 x float> %1404) #9, !dbg !14
  %1406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1374, <4 x float> %1405) #9, !dbg !14
  %1407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1376, <4 x float> %1406) #9, !dbg !14
  %1408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1378, <4 x float> %1407) #9, !dbg !14
  %1409 = shufflevector <8 x float> %1398, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1410 = fmul <4 x float> %1409, %987, !dbg !35
  %1411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1380, <4 x float> %1410) #9, !dbg !14
  %1412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1384, <4 x float> %1411) #9, !dbg !14
  %1413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1386, <4 x float> %1412) #9, !dbg !14
  %1414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1388, <4 x float> %1413) #9, !dbg !14
  %1415 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1390, <4 x float> %1414) #9, !dbg !14
  %1416 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1392, <4 x float> %1415) #9, !dbg !14
  %1417 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1394, <4 x float> %1416) #9, !dbg !14
  %1418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1396, <4 x float> %1417) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1419 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #0, !dbg !14
  %1420 = bitcast <4 x i32> %1419 to <8 x half>, !dbg !14
  %1421 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1422 = bitcast <4 x i32> %1421 to <8 x half>, !dbg !14
  %1423 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1424 = bitcast <4 x i32> %1423 to <8 x half>, !dbg !14
  %1425 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1426 = bitcast <4 x i32> %1425 to <8 x half>, !dbg !14
  %1427 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #0, !dbg !14
  %1428 = bitcast <4 x i32> %1427 to <8 x half>, !dbg !14
  %1429 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1430 = bitcast <4 x i32> %1429 to <8 x half>, !dbg !14
  %1431 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #0, !dbg !14
  %1432 = bitcast <4 x i32> %1431 to <8 x half>, !dbg !14
  %1433 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #0, !dbg !14
  %1434 = bitcast <4 x i32> %1433 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1435 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1226, !dbg !18
  %1436 = getelementptr i8, ptr addrspace(1) %1435, i64 192, !dbg !18
  %1437 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1077), !dbg !18
  %1438 = zext i32 %1437 to i64, !dbg !18
  %1439 = inttoptr i64 %1438 to ptr addrspace(3), !dbg !18
  %address.local1098 = getelementptr [2 x i8], ptr addrspace(1) %1436, i64 %address.local1082, !dbg !18
  %1440 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1441 = ptrtoint ptr addrspace(1) %address.local1098 to i64, !dbg !18
  %1442 = sub i64 %1441, %1440, !dbg !18
  %dma.buffer.offset1714 = trunc i64 %1442 to i32, !dbg !18
  %1443 = trunc i64 %1440 to i32, !dbg !18
  %1444 = insertelement <4 x i32> poison, i32 %1443, i64 0, !dbg !18
  %1445 = lshr i64 %1440, 32, !dbg !18
  %1446 = trunc i64 %1445 to i32, !dbg !18
  %1447 = insertelement <4 x i32> %1444, i32 %1446, i64 1, !dbg !18
  %1448 = insertelement <4 x i32> %1447, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1715 = insertelement <4 x i32> %1448, i32 131072, i64 3, !dbg !18
  %1449 = ptrtoint ptr addrspace(3) %1439 to i32, !dbg !18
  %dma.buffer.lds1716 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1449), !dbg !18
  %1450 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1716, i32 %dma.buffer.offset1714, <4 x i32> %dma.buffer.rsrc1715) #10, !dbg !18
  %1451 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1089), !dbg !18
  %1452 = zext i32 %1451 to i64, !dbg !18
  %1453 = inttoptr i64 %1452 to ptr addrspace(3), !dbg !18
  %address.local1099 = getelementptr [2 x i8], ptr addrspace(1) %1436, i64 %address.local1094, !dbg !18
  %1454 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1455 = ptrtoint ptr addrspace(1) %address.local1099 to i64, !dbg !18
  %1456 = sub i64 %1455, %1454, !dbg !18
  %dma.buffer.offset = trunc i64 %1456 to i32, !dbg !18
  %1457 = trunc i64 %1454 to i32, !dbg !18
  %1458 = insertelement <4 x i32> poison, i32 %1457, i64 0, !dbg !18
  %1459 = lshr i64 %1454, 32, !dbg !18
  %1460 = trunc i64 %1459 to i32, !dbg !18
  %1461 = insertelement <4 x i32> %1458, i32 %1460, i64 1, !dbg !18
  %1462 = insertelement <4 x i32> %1461, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1462, i32 131072, i64 3, !dbg !18
  %1463 = ptrtoint ptr addrspace(3) %1453 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1463), !dbg !18
  %1464 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1465 = shufflevector <8 x half> %1420, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1466 = extractvalue [2 x <8 x float>] %209, 0, !dbg !35
  %1467 = shufflevector <8 x float> %1466, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1468 = fmul <4 x float> %1467, %979, !dbg !35
  %1469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1465, <4 x float> %1468) #9, !dbg !14
  %1470 = shufflevector <8 x half> %1422, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1470, <4 x float> %1469) #9, !dbg !14
  %1472 = shufflevector <8 x half> %1424, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1472, <4 x float> %1471) #9, !dbg !14
  %1474 = shufflevector <8 x half> %1426, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1474, <4 x float> %1473) #9, !dbg !14
  %1476 = shufflevector <8 x half> %1428, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1476, <4 x float> %1475) #9, !dbg !14
  %1478 = shufflevector <8 x half> %1430, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1478, <4 x float> %1477) #9, !dbg !14
  %1480 = shufflevector <8 x half> %1432, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1480, <4 x float> %1479) #9, !dbg !14
  %1482 = shufflevector <8 x half> %1434, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1482, <4 x float> %1481) #9, !dbg !14
  %1484 = shufflevector <8 x half> %1420, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1485 = shufflevector <8 x float> %1466, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1486 = fmul <4 x float> %1485, %979, !dbg !35
  %1487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1484, <4 x float> %1486) #9, !dbg !14
  %1488 = shufflevector <8 x half> %1422, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1488, <4 x float> %1487) #9, !dbg !14
  %1490 = shufflevector <8 x half> %1424, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1490, <4 x float> %1489) #9, !dbg !14
  %1492 = shufflevector <8 x half> %1426, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1492, <4 x float> %1491) #9, !dbg !14
  %1494 = shufflevector <8 x half> %1428, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1494, <4 x float> %1493) #9, !dbg !14
  %1496 = shufflevector <8 x half> %1430, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1496, <4 x float> %1495) #9, !dbg !14
  %1498 = shufflevector <8 x half> %1432, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1498, <4 x float> %1497) #9, !dbg !14
  %1500 = shufflevector <8 x half> %1434, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1501 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1500, <4 x float> %1499) #9, !dbg !14
  %1502 = extractvalue [2 x <8 x float>] %209, 1, !dbg !35
  %1503 = shufflevector <8 x float> %1502, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1504 = fmul <4 x float> %1503, %987, !dbg !35
  %1505 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1465, <4 x float> %1504) #9, !dbg !14
  %1506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1470, <4 x float> %1505) #9, !dbg !14
  %1507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1472, <4 x float> %1506) #9, !dbg !14
  %1508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1474, <4 x float> %1507) #9, !dbg !14
  %1509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1476, <4 x float> %1508) #9, !dbg !14
  %1510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1478, <4 x float> %1509) #9, !dbg !14
  %1511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1480, <4 x float> %1510) #9, !dbg !14
  %1512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1482, <4 x float> %1511) #9, !dbg !14
  %1513 = shufflevector <8 x float> %1502, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1514 = fmul <4 x float> %1513, %987, !dbg !35
  %1515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1484, <4 x float> %1514) #9, !dbg !14
  %1516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1488, <4 x float> %1515) #9, !dbg !14
  %1517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1490, <4 x float> %1516) #9, !dbg !14
  %1518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1492, <4 x float> %1517) #9, !dbg !14
  %1519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1494, <4 x float> %1518) #9, !dbg !14
  %1520 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1496, <4 x float> %1519) #9, !dbg !14
  %1521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1498, <4 x float> %1520) #9, !dbg !14
  %1522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1500, <4 x float> %1521) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1523 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1037, i32 0) #0, !dbg !14
  %1524 = bitcast <4 x i32> %1523 to <8 x half>, !dbg !14
  %1525 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1526 = bitcast <4 x i32> %1525 to <8 x half>, !dbg !14
  %1527 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1528 = bitcast <4 x i32> %1527 to <8 x half>, !dbg !14
  %1529 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1530 = bitcast <4 x i32> %1529 to <8 x half>, !dbg !14
  %1531 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1060, i32 0) #0, !dbg !14
  %1532 = bitcast <4 x i32> %1531 to <8 x half>, !dbg !14
  %1533 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1534 = bitcast <4 x i32> %1533 to <8 x half>, !dbg !14
  %1535 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1068, i32 0) #0, !dbg !14
  %1536 = bitcast <4 x i32> %1535 to <8 x half>, !dbg !14
  %1537 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1072, i32 0) #0, !dbg !14
  %1538 = bitcast <4 x i32> %1537 to <8 x half>, !dbg !14
  %1539 = shufflevector <8 x half> %1524, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1540 = extractvalue [2 x <8 x float>] %210, 0, !dbg !35
  %1541 = shufflevector <8 x float> %1540, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1542 = fmul <4 x float> %1541, %979, !dbg !35
  %1543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1539, <4 x float> %1542) #9, !dbg !14
  %1544 = shufflevector <8 x half> %1526, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1544, <4 x float> %1543) #9, !dbg !14
  %1546 = shufflevector <8 x half> %1528, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1546, <4 x float> %1545) #9, !dbg !14
  %1548 = shufflevector <8 x half> %1530, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1548, <4 x float> %1547) #9, !dbg !14
  %1550 = shufflevector <8 x half> %1532, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1550, <4 x float> %1549) #9, !dbg !14
  %1552 = shufflevector <8 x half> %1534, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1552, <4 x float> %1551) #9, !dbg !14
  %1554 = shufflevector <8 x half> %1536, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1554, <4 x float> %1553) #9, !dbg !14
  %1556 = shufflevector <8 x half> %1538, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1556, <4 x float> %1555) #9, !dbg !14
  %1558 = shufflevector <8 x half> %1524, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1559 = shufflevector <8 x float> %1540, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1560 = fmul <4 x float> %1559, %979, !dbg !35
  %1561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1558, <4 x float> %1560) #9, !dbg !14
  %1562 = shufflevector <8 x half> %1526, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1562, <4 x float> %1561) #9, !dbg !14
  %1564 = shufflevector <8 x half> %1528, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1564, <4 x float> %1563) #9, !dbg !14
  %1566 = shufflevector <8 x half> %1530, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1567 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1566, <4 x float> %1565) #9, !dbg !14
  %1568 = shufflevector <8 x half> %1532, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1568, <4 x float> %1567) #9, !dbg !14
  %1570 = shufflevector <8 x half> %1534, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1570, <4 x float> %1569) #9, !dbg !14
  %1572 = shufflevector <8 x half> %1536, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1572, <4 x float> %1571) #9, !dbg !14
  %1574 = shufflevector <8 x half> %1538, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1575 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1574, <4 x float> %1573) #9, !dbg !14
  %1576 = extractvalue [2 x <8 x float>] %210, 1, !dbg !35
  %1577 = shufflevector <8 x float> %1576, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1578 = fmul <4 x float> %1577, %987, !dbg !35
  %1579 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1539, <4 x float> %1578) #9, !dbg !14
  %1580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1544, <4 x float> %1579) #9, !dbg !14
  %1581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1546, <4 x float> %1580) #9, !dbg !14
  %1582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1548, <4 x float> %1581) #9, !dbg !14
  %1583 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1550, <4 x float> %1582) #9, !dbg !14
  %1584 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1552, <4 x float> %1583) #9, !dbg !14
  %1585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1554, <4 x float> %1584) #9, !dbg !14
  %1586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1556, <4 x float> %1585) #9, !dbg !14
  %1587 = shufflevector <8 x float> %1576, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1588 = fmul <4 x float> %1587, %987, !dbg !35
  %1589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1558, <4 x float> %1588) #9, !dbg !14
  %1590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1562, <4 x float> %1589) #9, !dbg !14
  %1591 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1564, <4 x float> %1590) #9, !dbg !14
  %1592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed496, <4 x half> %1566, <4 x float> %1591) #9, !dbg !14
  %1593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed498, <4 x half> %1568, <4 x float> %1592) #9, !dbg !14
  %1594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed500, <4 x half> %1570, <4 x float> %1593) #9, !dbg !14
  %1595 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed502, <4 x half> %1572, <4 x float> %1594) #9, !dbg !14
  %1596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed504, <4 x half> %1574, <4 x float> %1595) #9, !dbg !14
  %1597 = shufflevector <4 x float> %1275, <4 x float> %1293, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1598 = insertvalue [2 x <8 x float>] poison, <8 x float> %1597, 0, !dbg !7
  %1599 = shufflevector <4 x float> %1304, <4 x float> %1314, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1600 = insertvalue [2 x <8 x float>] %1598, <8 x float> %1599, 1, !dbg !7
  %1601 = insertvalue [2 x <1 x float>] poison, <1 x float> %943, 0, !dbg !7
  %1602 = insertvalue [2 x <1 x float>] %1601, <1 x float> %956, 1, !dbg !7
  %1603 = shufflevector <4 x float> %1379, <4 x float> %1397, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1604 = insertvalue [2 x <8 x float>] poison, <8 x float> %1603, 0, !dbg !7
  %1605 = shufflevector <4 x float> %1408, <4 x float> %1418, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1606 = insertvalue [2 x <8 x float>] %1604, <8 x float> %1605, 1, !dbg !7
  %1607 = shufflevector <4 x float> %1483, <4 x float> %1501, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1608 = insertvalue [2 x <8 x float>] poison, <8 x float> %1607, 0, !dbg !7
  %1609 = shufflevector <4 x float> %1512, <4 x float> %1522, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1610 = insertvalue [2 x <8 x float>] %1608, <8 x float> %1609, 1, !dbg !7
  %1611 = shufflevector <4 x float> %1557, <4 x float> %1575, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1612 = insertvalue [2 x <8 x float>] poison, <8 x float> %1611, 0, !dbg !7
  %1613 = shufflevector <4 x float> %1586, <4 x float> %1596, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1614 = insertvalue [2 x <8 x float>] %1612, <8 x float> %1613, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1615 = add nuw nsw i64 %204, 1, !dbg !7
  br label %203, !dbg !7

1616:                                             ; preds = %203
  %lane.sum.field284 = extractvalue [2 x <1 x float>] %206, 1, !dbg !7
  %lane.sum.local285 = extractelement <1 x float> %lane.sum.field284, i64 0, !dbg !7
  %lane.sum.lane.lo286 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane287 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo286), !dbg !7
  %1617 = shl nuw nsw i32 %lane.sum.lane287, 2, !dbg !7
  %1618 = xor i32 %1617, 64, !dbg !7
  %1619 = bitcast float %lane.sum.local285 to i32, !dbg !7
  %lane.sum.shuffle288 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1618, i32 %1619), !dbg !7
  %1620 = bitcast i32 %lane.sum.shuffle288 to float, !dbg !7
  %lane.sum.final289 = fadd float %lane.sum.local285, %1620, !dbg !7
  %1621 = shl nuw nsw i32 %lane.sum.lane287, 2, !dbg !7
  %1622 = xor i32 %1621, 128, !dbg !7
  %1623 = bitcast float %lane.sum.final289 to i32, !dbg !7
  %lane.sum.shuffle290 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1622, i32 %1623), !dbg !7
  %1624 = bitcast i32 %lane.sum.shuffle290 to float, !dbg !7
  %lane.sum.final291 = fadd float %lane.sum.final289, %1624, !dbg !7
  %1625 = insertelement <1 x float> poison, float %lane.sum.final291, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %206, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %lane.sum.lane.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo), !dbg !7
  %1626 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1627 = xor i32 %1626, 64, !dbg !7
  %1628 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1627, i32 %1628), !dbg !7
  %1629 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1629, !dbg !7
  %1630 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1631 = xor i32 %1630, 128, !dbg !7
  %1632 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle282 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1631, i32 %1632), !dbg !7
  %1633 = bitcast i32 %lane.sum.shuffle282 to float, !dbg !7
  %lane.sum.final283 = fadd float %lane.sum.final, %1633, !dbg !7
  %1634 = insertelement <1 x float> poison, float %lane.sum.final283, i64 0, !dbg !7
  %1635 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %1635, 0, !dbg !7
  %1636 = and i32 %31, 32, !dbg !7
  %.not36 = icmp eq i32 %1636, 0, !dbg !7
  %1637 = extractvalue [2 x <8 x float>] %210, 0, !dbg !7
  %1638 = extractelement <8 x float> %1637, i64 0, !dbg !7
  %1639 = extractelement <8 x float> %1637, i64 1, !dbg !7
  %1640 = extractelement <8 x float> %1637, i64 2, !dbg !7
  %1641 = extractelement <8 x float> %1637, i64 3, !dbg !7
  %1642 = select i1 %.not, float %1639, float %1638, !dbg !7
  %1643 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1644 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1643), !dbg !7
  %1645 = add nuw nsw i32 %1644, 64, !dbg !7
  %1646 = xor i32 %1644, 16, !dbg !7
  %1647 = and i32 %1645, 192, !dbg !7
  %1648 = icmp samesign ult i32 %1646, %1647, !dbg !7
  %1649 = select i1 %1648, i32 %1646, i32 %1644, !dbg !7
  %1650 = shl nuw nsw i32 %1649, 2, !dbg !7
  %1651 = bitcast float %1642 to i32, !dbg !7
  %1652 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1651), !dbg !7
  %1653 = bitcast i32 %1652 to float, !dbg !7
  %1654 = select i1 %.not, float %1638, float %1653, !dbg !7
  %1655 = select i1 %.not, float %1653, float %1639, !dbg !7
  %1656 = select i1 %.not, float %1641, float %1640, !dbg !7
  %1657 = bitcast float %1656 to i32, !dbg !7
  %1658 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1657), !dbg !7
  %1659 = bitcast i32 %1658 to float, !dbg !7
  %1660 = select i1 %.not, float %1640, float %1659, !dbg !7
  %1661 = select i1 %.not, float %1659, float %1641, !dbg !7
  %1662 = select i1 %.not36, float %1660, float %1654, !dbg !7
  %1663 = xor i32 %1644, 32, !dbg !7
  %1664 = icmp samesign ult i32 %1663, %1647, !dbg !7
  %1665 = select i1 %1664, i32 %1663, i32 %1644, !dbg !7
  %1666 = shl nuw nsw i32 %1665, 2, !dbg !7
  %1667 = bitcast float %1662 to i32, !dbg !7
  %1668 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1667), !dbg !7
  %1669 = bitcast i32 %1668 to float, !dbg !7
  %1670 = select i1 %.not36, float %1654, float %1669, !dbg !7
  %1671 = select i1 %.not36, float %1669, float %1660, !dbg !7
  %1672 = select i1 %.not36, float %1661, float %1655, !dbg !7
  %1673 = bitcast float %1672 to i32, !dbg !7
  %1674 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1673), !dbg !7
  %1675 = bitcast i32 %1674 to float, !dbg !7
  %1676 = select i1 %.not36, float %1655, float %1675, !dbg !7
  %1677 = select i1 %.not36, float %1675, float %1661, !dbg !7
  %1678 = extractvalue [2 x <8 x float>] %209, 0, !dbg !7
  %1679 = extractelement <8 x float> %1678, i64 0, !dbg !7
  %1680 = extractelement <8 x float> %1678, i64 1, !dbg !7
  %1681 = extractelement <8 x float> %1678, i64 2, !dbg !7
  %1682 = extractelement <8 x float> %1678, i64 3, !dbg !7
  %1683 = select i1 %.not, float %1680, float %1679, !dbg !7
  %1684 = bitcast float %1683 to i32, !dbg !7
  %1685 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1684), !dbg !7
  %1686 = bitcast i32 %1685 to float, !dbg !7
  %1687 = select i1 %.not, float %1679, float %1686, !dbg !7
  %1688 = select i1 %.not, float %1686, float %1680, !dbg !7
  %1689 = select i1 %.not, float %1682, float %1681, !dbg !7
  %1690 = bitcast float %1689 to i32, !dbg !7
  %1691 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1690), !dbg !7
  %1692 = bitcast i32 %1691 to float, !dbg !7
  %1693 = select i1 %.not, float %1681, float %1692, !dbg !7
  %1694 = select i1 %.not, float %1692, float %1682, !dbg !7
  %1695 = select i1 %.not36, float %1693, float %1687, !dbg !7
  %1696 = bitcast float %1695 to i32, !dbg !7
  %1697 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1696), !dbg !7
  %1698 = bitcast i32 %1697 to float, !dbg !7
  %1699 = select i1 %.not36, float %1687, float %1698, !dbg !7
  %1700 = select i1 %.not36, float %1698, float %1693, !dbg !7
  %1701 = select i1 %.not36, float %1694, float %1688, !dbg !7
  %1702 = bitcast float %1701 to i32, !dbg !7
  %1703 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1702), !dbg !7
  %1704 = bitcast i32 %1703 to float, !dbg !7
  %1705 = select i1 %.not36, float %1688, float %1704, !dbg !7
  %1706 = select i1 %.not36, float %1704, float %1694, !dbg !7
  %1707 = extractvalue [2 x <8 x float>] %208, 0, !dbg !7
  %1708 = extractelement <8 x float> %1707, i64 0, !dbg !7
  %1709 = extractelement <8 x float> %1707, i64 1, !dbg !7
  %1710 = extractelement <8 x float> %1707, i64 2, !dbg !7
  %1711 = extractelement <8 x float> %1707, i64 3, !dbg !7
  %1712 = select i1 %.not, float %1709, float %1708, !dbg !7
  %1713 = bitcast float %1712 to i32, !dbg !7
  %1714 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1713), !dbg !7
  %1715 = bitcast i32 %1714 to float, !dbg !7
  %1716 = select i1 %.not, float %1708, float %1715, !dbg !7
  %1717 = select i1 %.not, float %1715, float %1709, !dbg !7
  %1718 = select i1 %.not, float %1711, float %1710, !dbg !7
  %1719 = bitcast float %1718 to i32, !dbg !7
  %1720 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1719), !dbg !7
  %1721 = bitcast i32 %1720 to float, !dbg !7
  %1722 = select i1 %.not, float %1710, float %1721, !dbg !7
  %1723 = select i1 %.not, float %1721, float %1711, !dbg !7
  %1724 = select i1 %.not36, float %1722, float %1716, !dbg !7
  %1725 = bitcast float %1724 to i32, !dbg !7
  %1726 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1725), !dbg !7
  %1727 = bitcast i32 %1726 to float, !dbg !7
  %1728 = select i1 %.not36, float %1716, float %1727, !dbg !7
  %1729 = select i1 %.not36, float %1727, float %1722, !dbg !7
  %1730 = select i1 %.not36, float %1723, float %1717, !dbg !7
  %1731 = bitcast float %1730 to i32, !dbg !7
  %1732 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1731), !dbg !7
  %1733 = bitcast i32 %1732 to float, !dbg !7
  %1734 = select i1 %.not36, float %1717, float %1733, !dbg !7
  %1735 = select i1 %.not36, float %1733, float %1723, !dbg !7
  %1736 = extractvalue [2 x <8 x float>] %205, 0, !dbg !7
  %1737 = extractelement <8 x float> %1736, i64 0, !dbg !7
  %1738 = extractelement <8 x float> %1736, i64 1, !dbg !7
  %1739 = extractelement <8 x float> %1736, i64 2, !dbg !7
  %1740 = extractelement <8 x float> %1736, i64 3, !dbg !7
  %1741 = select i1 %.not, float %1738, float %1737, !dbg !7
  %1742 = bitcast float %1741 to i32, !dbg !7
  %1743 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1742), !dbg !7
  %1744 = bitcast i32 %1743 to float, !dbg !7
  %1745 = select i1 %.not, float %1737, float %1744, !dbg !7
  %1746 = select i1 %.not, float %1744, float %1738, !dbg !7
  %1747 = select i1 %.not, float %1740, float %1739, !dbg !7
  %1748 = bitcast float %1747 to i32, !dbg !7
  %1749 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1748), !dbg !7
  %1750 = bitcast i32 %1749 to float, !dbg !7
  %1751 = select i1 %.not, float %1739, float %1750, !dbg !7
  %1752 = select i1 %.not, float %1750, float %1740, !dbg !7
  %1753 = select i1 %.not36, float %1751, float %1745, !dbg !7
  %1754 = bitcast float %1753 to i32, !dbg !7
  %1755 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1754), !dbg !7
  %1756 = select i1 %.not36, float %1752, float %1746, !dbg !7
  %1757 = bitcast float %1756 to i32, !dbg !7
  %1758 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1757), !dbg !7
  %1759 = shufflevector <1 x float> %1634, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1760 = extractelement <8 x float> %1637, i64 4, !dbg !7
  %1761 = extractelement <8 x float> %1637, i64 5, !dbg !7
  %1762 = extractelement <8 x float> %1637, i64 6, !dbg !7
  %1763 = extractelement <8 x float> %1637, i64 7, !dbg !7
  %1764 = select i1 %.not, float %1761, float %1760, !dbg !7
  %1765 = bitcast float %1764 to i32, !dbg !7
  %1766 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1765), !dbg !7
  %1767 = bitcast i32 %1766 to float, !dbg !7
  %1768 = select i1 %.not, float %1760, float %1767, !dbg !7
  %1769 = select i1 %.not, float %1767, float %1761, !dbg !7
  %1770 = select i1 %.not, float %1763, float %1762, !dbg !7
  %1771 = bitcast float %1770 to i32, !dbg !7
  %1772 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1771), !dbg !7
  %1773 = bitcast i32 %1772 to float, !dbg !7
  %1774 = select i1 %.not, float %1762, float %1773, !dbg !7
  %1775 = select i1 %.not, float %1773, float %1763, !dbg !7
  %1776 = select i1 %.not36, float %1774, float %1768, !dbg !7
  %1777 = bitcast float %1776 to i32, !dbg !7
  %1778 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1777), !dbg !7
  %1779 = bitcast i32 %1778 to float, !dbg !7
  %1780 = select i1 %.not36, float %1768, float %1779, !dbg !7
  %1781 = select i1 %.not36, float %1779, float %1774, !dbg !7
  %1782 = select i1 %.not36, float %1775, float %1769, !dbg !7
  %1783 = bitcast float %1782 to i32, !dbg !7
  %1784 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1783), !dbg !7
  %1785 = bitcast i32 %1784 to float, !dbg !7
  %1786 = select i1 %.not36, float %1769, float %1785, !dbg !7
  %1787 = select i1 %.not36, float %1785, float %1775, !dbg !7
  %1788 = extractelement <8 x float> %1678, i64 4, !dbg !7
  %1789 = extractelement <8 x float> %1678, i64 5, !dbg !7
  %1790 = extractelement <8 x float> %1678, i64 6, !dbg !7
  %1791 = extractelement <8 x float> %1678, i64 7, !dbg !7
  %1792 = select i1 %.not, float %1789, float %1788, !dbg !7
  %1793 = bitcast float %1792 to i32, !dbg !7
  %1794 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1793), !dbg !7
  %1795 = bitcast i32 %1794 to float, !dbg !7
  %1796 = select i1 %.not, float %1788, float %1795, !dbg !7
  %1797 = select i1 %.not, float %1795, float %1789, !dbg !7
  %1798 = select i1 %.not, float %1791, float %1790, !dbg !7
  %1799 = bitcast float %1798 to i32, !dbg !7
  %1800 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1799), !dbg !7
  %1801 = bitcast i32 %1800 to float, !dbg !7
  %1802 = select i1 %.not, float %1790, float %1801, !dbg !7
  %1803 = select i1 %.not, float %1801, float %1791, !dbg !7
  %1804 = select i1 %.not36, float %1802, float %1796, !dbg !7
  %1805 = bitcast float %1804 to i32, !dbg !7
  %1806 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1805), !dbg !7
  %1807 = bitcast i32 %1806 to float, !dbg !7
  %1808 = select i1 %.not36, float %1796, float %1807, !dbg !7
  %1809 = select i1 %.not36, float %1807, float %1802, !dbg !7
  %1810 = select i1 %.not36, float %1803, float %1797, !dbg !7
  %1811 = bitcast float %1810 to i32, !dbg !7
  %1812 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1811), !dbg !7
  %1813 = bitcast i32 %1812 to float, !dbg !7
  %1814 = select i1 %.not36, float %1797, float %1813, !dbg !7
  %1815 = select i1 %.not36, float %1813, float %1803, !dbg !7
  %1816 = extractelement <8 x float> %1707, i64 4, !dbg !7
  %1817 = extractelement <8 x float> %1707, i64 5, !dbg !7
  %1818 = extractelement <8 x float> %1707, i64 6, !dbg !7
  %1819 = extractelement <8 x float> %1707, i64 7, !dbg !7
  %1820 = select i1 %.not, float %1817, float %1816, !dbg !7
  %1821 = bitcast float %1820 to i32, !dbg !7
  %1822 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1821), !dbg !7
  %1823 = bitcast i32 %1822 to float, !dbg !7
  %1824 = select i1 %.not, float %1816, float %1823, !dbg !7
  %1825 = select i1 %.not, float %1823, float %1817, !dbg !7
  %1826 = select i1 %.not, float %1819, float %1818, !dbg !7
  %1827 = bitcast float %1826 to i32, !dbg !7
  %1828 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1827), !dbg !7
  %1829 = bitcast i32 %1828 to float, !dbg !7
  %1830 = select i1 %.not, float %1818, float %1829, !dbg !7
  %1831 = select i1 %.not, float %1829, float %1819, !dbg !7
  %1832 = select i1 %.not36, float %1830, float %1824, !dbg !7
  %1833 = bitcast float %1832 to i32, !dbg !7
  %1834 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1833), !dbg !7
  %1835 = bitcast i32 %1834 to float, !dbg !7
  %1836 = select i1 %.not36, float %1824, float %1835, !dbg !7
  %1837 = select i1 %.not36, float %1835, float %1830, !dbg !7
  %1838 = select i1 %.not36, float %1831, float %1825, !dbg !7
  %1839 = bitcast float %1838 to i32, !dbg !7
  %1840 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1839), !dbg !7
  %1841 = bitcast i32 %1840 to float, !dbg !7
  %1842 = select i1 %.not36, float %1825, float %1841, !dbg !7
  %1843 = select i1 %.not36, float %1841, float %1831, !dbg !7
  %1844 = extractelement <8 x float> %1736, i64 4, !dbg !7
  %1845 = extractelement <8 x float> %1736, i64 5, !dbg !7
  %1846 = extractelement <8 x float> %1736, i64 6, !dbg !7
  %1847 = extractelement <8 x float> %1736, i64 7, !dbg !7
  %1848 = select i1 %.not, float %1845, float %1844, !dbg !7
  %1849 = bitcast float %1848 to i32, !dbg !7
  %1850 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1849), !dbg !7
  %1851 = bitcast i32 %1850 to float, !dbg !7
  %1852 = select i1 %.not, float %1844, float %1851, !dbg !7
  %1853 = select i1 %.not, float %1851, float %1845, !dbg !7
  %1854 = select i1 %.not, float %1847, float %1846, !dbg !7
  %1855 = bitcast float %1854 to i32, !dbg !7
  %1856 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1855), !dbg !7
  %1857 = bitcast i32 %1856 to float, !dbg !7
  %1858 = select i1 %.not, float %1846, float %1857, !dbg !7
  %1859 = select i1 %.not, float %1857, float %1847, !dbg !7
  %1860 = select i1 %.not36, float %1858, float %1852, !dbg !7
  %1861 = bitcast float %1860 to i32, !dbg !7
  %1862 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1861), !dbg !7
  %1863 = select i1 %.not36, float %1859, float %1853, !dbg !7
  %1864 = bitcast float %1863 to i32, !dbg !7
  %1865 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1864), !dbg !7
  %1866 = extractvalue [2 x <8 x float>] %210, 1, !dbg !7
  %1867 = extractelement <8 x float> %1866, i64 0, !dbg !7
  %1868 = extractelement <8 x float> %1866, i64 1, !dbg !7
  %1869 = extractelement <8 x float> %1866, i64 2, !dbg !7
  %1870 = extractelement <8 x float> %1866, i64 3, !dbg !7
  %1871 = select i1 %.not, float %1868, float %1867, !dbg !7
  %1872 = bitcast float %1871 to i32, !dbg !7
  %1873 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1872), !dbg !7
  %1874 = bitcast i32 %1873 to float, !dbg !7
  %1875 = select i1 %.not, float %1867, float %1874, !dbg !7
  %1876 = select i1 %.not, float %1874, float %1868, !dbg !7
  %1877 = select i1 %.not, float %1870, float %1869, !dbg !7
  %1878 = bitcast float %1877 to i32, !dbg !7
  %1879 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1878), !dbg !7
  %1880 = bitcast i32 %1879 to float, !dbg !7
  %1881 = select i1 %.not, float %1869, float %1880, !dbg !7
  %1882 = select i1 %.not, float %1880, float %1870, !dbg !7
  %1883 = select i1 %.not36, float %1881, float %1875, !dbg !7
  %1884 = bitcast float %1883 to i32, !dbg !7
  %1885 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1884), !dbg !7
  %1886 = bitcast i32 %1885 to float, !dbg !7
  %1887 = select i1 %.not36, float %1875, float %1886, !dbg !7
  %1888 = select i1 %.not36, float %1886, float %1881, !dbg !7
  %1889 = select i1 %.not36, float %1882, float %1876, !dbg !7
  %1890 = bitcast float %1889 to i32, !dbg !7
  %1891 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1890), !dbg !7
  %1892 = bitcast i32 %1891 to float, !dbg !7
  %1893 = select i1 %.not36, float %1876, float %1892, !dbg !7
  %1894 = select i1 %.not36, float %1892, float %1882, !dbg !7
  %1895 = extractvalue [2 x <8 x float>] %209, 1, !dbg !7
  %1896 = extractelement <8 x float> %1895, i64 0, !dbg !7
  %1897 = extractelement <8 x float> %1895, i64 1, !dbg !7
  %1898 = extractelement <8 x float> %1895, i64 2, !dbg !7
  %1899 = extractelement <8 x float> %1895, i64 3, !dbg !7
  %1900 = select i1 %.not, float %1897, float %1896, !dbg !7
  %1901 = bitcast float %1900 to i32, !dbg !7
  %1902 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1901), !dbg !7
  %1903 = bitcast i32 %1902 to float, !dbg !7
  %1904 = select i1 %.not, float %1896, float %1903, !dbg !7
  %1905 = select i1 %.not, float %1903, float %1897, !dbg !7
  %1906 = select i1 %.not, float %1899, float %1898, !dbg !7
  %1907 = bitcast float %1906 to i32, !dbg !7
  %1908 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1907), !dbg !7
  %1909 = bitcast i32 %1908 to float, !dbg !7
  %1910 = select i1 %.not, float %1898, float %1909, !dbg !7
  %1911 = select i1 %.not, float %1909, float %1899, !dbg !7
  %1912 = select i1 %.not36, float %1910, float %1904, !dbg !7
  %1913 = bitcast float %1912 to i32, !dbg !7
  %1914 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1913), !dbg !7
  %1915 = bitcast i32 %1914 to float, !dbg !7
  %1916 = select i1 %.not36, float %1904, float %1915, !dbg !7
  %1917 = select i1 %.not36, float %1915, float %1910, !dbg !7
  %1918 = select i1 %.not36, float %1911, float %1905, !dbg !7
  %1919 = bitcast float %1918 to i32, !dbg !7
  %1920 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1919), !dbg !7
  %1921 = bitcast i32 %1920 to float, !dbg !7
  %1922 = select i1 %.not36, float %1905, float %1921, !dbg !7
  %1923 = select i1 %.not36, float %1921, float %1911, !dbg !7
  %1924 = extractvalue [2 x <8 x float>] %208, 1, !dbg !7
  %1925 = extractelement <8 x float> %1924, i64 0, !dbg !7
  %1926 = extractelement <8 x float> %1924, i64 1, !dbg !7
  %1927 = extractelement <8 x float> %1924, i64 2, !dbg !7
  %1928 = extractelement <8 x float> %1924, i64 3, !dbg !7
  %1929 = select i1 %.not, float %1926, float %1925, !dbg !7
  %1930 = bitcast float %1929 to i32, !dbg !7
  %1931 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1930), !dbg !7
  %1932 = bitcast i32 %1931 to float, !dbg !7
  %1933 = select i1 %.not, float %1925, float %1932, !dbg !7
  %1934 = select i1 %.not, float %1932, float %1926, !dbg !7
  %1935 = select i1 %.not, float %1928, float %1927, !dbg !7
  %1936 = bitcast float %1935 to i32, !dbg !7
  %1937 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1936), !dbg !7
  %1938 = bitcast i32 %1937 to float, !dbg !7
  %1939 = select i1 %.not, float %1927, float %1938, !dbg !7
  %1940 = select i1 %.not, float %1938, float %1928, !dbg !7
  %1941 = select i1 %.not36, float %1939, float %1933, !dbg !7
  %1942 = bitcast float %1941 to i32, !dbg !7
  %1943 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1942), !dbg !7
  %1944 = bitcast i32 %1943 to float, !dbg !7
  %1945 = select i1 %.not36, float %1933, float %1944, !dbg !7
  %1946 = select i1 %.not36, float %1944, float %1939, !dbg !7
  %1947 = select i1 %.not36, float %1940, float %1934, !dbg !7
  %1948 = bitcast float %1947 to i32, !dbg !7
  %1949 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1948), !dbg !7
  %1950 = bitcast i32 %1949 to float, !dbg !7
  %1951 = select i1 %.not36, float %1934, float %1950, !dbg !7
  %1952 = select i1 %.not36, float %1950, float %1940, !dbg !7
  %1953 = extractvalue [2 x <8 x float>] %205, 1, !dbg !7
  %1954 = extractelement <8 x float> %1953, i64 0, !dbg !7
  %1955 = extractelement <8 x float> %1953, i64 1, !dbg !7
  %1956 = extractelement <8 x float> %1953, i64 2, !dbg !7
  %1957 = extractelement <8 x float> %1953, i64 3, !dbg !7
  %1958 = select i1 %.not, float %1955, float %1954, !dbg !7
  %1959 = bitcast float %1958 to i32, !dbg !7
  %1960 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1959), !dbg !7
  %1961 = bitcast i32 %1960 to float, !dbg !7
  %1962 = select i1 %.not, float %1954, float %1961, !dbg !7
  %1963 = select i1 %.not, float %1961, float %1955, !dbg !7
  %1964 = select i1 %.not, float %1957, float %1956, !dbg !7
  %1965 = bitcast float %1964 to i32, !dbg !7
  %1966 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1965), !dbg !7
  %1967 = bitcast i32 %1966 to float, !dbg !7
  %1968 = select i1 %.not, float %1956, float %1967, !dbg !7
  %1969 = select i1 %.not, float %1967, float %1957, !dbg !7
  %1970 = select i1 %.not36, float %1968, float %1962, !dbg !7
  %1971 = bitcast float %1970 to i32, !dbg !7
  %1972 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1971), !dbg !7
  %1973 = select i1 %.not36, float %1969, float %1963, !dbg !7
  %1974 = bitcast float %1973 to i32, !dbg !7
  %1975 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1974), !dbg !7
  %1976 = shufflevector <1 x float> %1625, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1977 = extractelement <8 x float> %1866, i64 4, !dbg !7
  %1978 = extractelement <8 x float> %1866, i64 5, !dbg !7
  %1979 = extractelement <8 x float> %1866, i64 6, !dbg !7
  %1980 = extractelement <8 x float> %1866, i64 7, !dbg !7
  %1981 = select i1 %.not, float %1978, float %1977, !dbg !7
  %1982 = bitcast float %1981 to i32, !dbg !7
  %1983 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1982), !dbg !7
  %1984 = bitcast i32 %1983 to float, !dbg !7
  %1985 = select i1 %.not, float %1977, float %1984, !dbg !7
  %1986 = select i1 %.not, float %1984, float %1978, !dbg !7
  %1987 = select i1 %.not, float %1980, float %1979, !dbg !7
  %1988 = bitcast float %1987 to i32, !dbg !7
  %1989 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %1988), !dbg !7
  %1990 = bitcast i32 %1989 to float, !dbg !7
  %1991 = select i1 %.not, float %1979, float %1990, !dbg !7
  %1992 = select i1 %.not, float %1990, float %1980, !dbg !7
  %1993 = select i1 %.not36, float %1991, float %1985, !dbg !7
  %1994 = bitcast float %1993 to i32, !dbg !7
  %1995 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1994), !dbg !7
  %1996 = bitcast i32 %1995 to float, !dbg !7
  %1997 = select i1 %.not36, float %1985, float %1996, !dbg !7
  %1998 = select i1 %.not36, float %1996, float %1991, !dbg !7
  %1999 = select i1 %.not36, float %1992, float %1986, !dbg !7
  %2000 = bitcast float %1999 to i32, !dbg !7
  %2001 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %2000), !dbg !7
  %2002 = bitcast i32 %2001 to float, !dbg !7
  %2003 = select i1 %.not36, float %1986, float %2002, !dbg !7
  %2004 = select i1 %.not36, float %2002, float %1992, !dbg !7
  %2005 = extractelement <8 x float> %1895, i64 4, !dbg !7
  %2006 = extractelement <8 x float> %1895, i64 5, !dbg !7
  %2007 = extractelement <8 x float> %1895, i64 6, !dbg !7
  %2008 = extractelement <8 x float> %1895, i64 7, !dbg !7
  %2009 = select i1 %.not, float %2006, float %2005, !dbg !7
  %2010 = bitcast float %2009 to i32, !dbg !7
  %2011 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %2010), !dbg !7
  %2012 = bitcast i32 %2011 to float, !dbg !7
  %2013 = select i1 %.not, float %2005, float %2012, !dbg !7
  %2014 = select i1 %.not, float %2012, float %2006, !dbg !7
  %2015 = select i1 %.not, float %2008, float %2007, !dbg !7
  %2016 = bitcast float %2015 to i32, !dbg !7
  %2017 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %2016), !dbg !7
  %2018 = bitcast i32 %2017 to float, !dbg !7
  %2019 = select i1 %.not, float %2007, float %2018, !dbg !7
  %2020 = select i1 %.not, float %2018, float %2008, !dbg !7
  %2021 = select i1 %.not36, float %2019, float %2013, !dbg !7
  %2022 = bitcast float %2021 to i32, !dbg !7
  %2023 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %2022), !dbg !7
  %2024 = bitcast i32 %2023 to float, !dbg !7
  %2025 = select i1 %.not36, float %2013, float %2024, !dbg !7
  %2026 = select i1 %.not36, float %2024, float %2019, !dbg !7
  %2027 = select i1 %.not36, float %2020, float %2014, !dbg !7
  %2028 = bitcast float %2027 to i32, !dbg !7
  %2029 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %2028), !dbg !7
  %2030 = bitcast i32 %2029 to float, !dbg !7
  %2031 = select i1 %.not36, float %2014, float %2030, !dbg !7
  %2032 = select i1 %.not36, float %2030, float %2020, !dbg !7
  %2033 = extractelement <8 x float> %1924, i64 4, !dbg !7
  %2034 = extractelement <8 x float> %1924, i64 5, !dbg !7
  %2035 = extractelement <8 x float> %1924, i64 6, !dbg !7
  %2036 = extractelement <8 x float> %1924, i64 7, !dbg !7
  %2037 = select i1 %.not, float %2034, float %2033, !dbg !7
  %2038 = bitcast float %2037 to i32, !dbg !7
  %2039 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %2038), !dbg !7
  %2040 = bitcast i32 %2039 to float, !dbg !7
  %2041 = select i1 %.not, float %2033, float %2040, !dbg !7
  %2042 = select i1 %.not, float %2040, float %2034, !dbg !7
  %2043 = select i1 %.not, float %2036, float %2035, !dbg !7
  %2044 = bitcast float %2043 to i32, !dbg !7
  %2045 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %2044), !dbg !7
  %2046 = bitcast i32 %2045 to float, !dbg !7
  %2047 = select i1 %.not, float %2035, float %2046, !dbg !7
  %2048 = select i1 %.not, float %2046, float %2036, !dbg !7
  %2049 = select i1 %.not36, float %2047, float %2041, !dbg !7
  %2050 = bitcast float %2049 to i32, !dbg !7
  %2051 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %2050), !dbg !7
  %2052 = bitcast i32 %2051 to float, !dbg !7
  %2053 = select i1 %.not36, float %2041, float %2052, !dbg !7
  %2054 = select i1 %.not36, float %2052, float %2047, !dbg !7
  %2055 = select i1 %.not36, float %2048, float %2042, !dbg !7
  %2056 = bitcast float %2055 to i32, !dbg !7
  %2057 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %2056), !dbg !7
  %2058 = bitcast i32 %2057 to float, !dbg !7
  %2059 = select i1 %.not36, float %2042, float %2058, !dbg !7
  %2060 = select i1 %.not36, float %2058, float %2048, !dbg !7
  %2061 = extractelement <8 x float> %1953, i64 4, !dbg !7
  %2062 = extractelement <8 x float> %1953, i64 5, !dbg !7
  %2063 = extractelement <8 x float> %1953, i64 6, !dbg !7
  %2064 = extractelement <8 x float> %1953, i64 7, !dbg !7
  %2065 = select i1 %.not, float %2062, float %2061, !dbg !7
  %2066 = bitcast float %2065 to i32, !dbg !7
  %2067 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %2066), !dbg !7
  %2068 = bitcast i32 %2067 to float, !dbg !7
  %2069 = select i1 %.not, float %2061, float %2068, !dbg !7
  %2070 = select i1 %.not, float %2068, float %2062, !dbg !7
  %2071 = select i1 %.not, float %2064, float %2063, !dbg !7
  %2072 = bitcast float %2071 to i32, !dbg !7
  %2073 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1650, i32 %2072), !dbg !7
  %2074 = bitcast i32 %2073 to float, !dbg !7
  %2075 = select i1 %.not, float %2063, float %2074, !dbg !7
  %2076 = select i1 %.not, float %2074, float %2064, !dbg !7
  %2077 = select i1 %.not36, float %2075, float %2069, !dbg !7
  %2078 = bitcast float %2077 to i32, !dbg !7
  %2079 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %2078), !dbg !7
  %2080 = select i1 %.not36, float %2076, float %2070, !dbg !7
  %2081 = bitcast float %2080 to i32, !dbg !7
  %2082 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %2081), !dbg !7
  %2083 = bitcast i32 %2079 to float, !dbg !7
  %2084 = select i1 %.not36, float %2069, float %2083, !dbg !7
  %2085 = insertelement <4 x float> poison, float %2084, i64 0, !dbg !19
  %2086 = bitcast i32 %2082 to float, !dbg !7
  %2087 = select i1 %.not36, float %2070, float %2086, !dbg !7
  %2088 = insertelement <4 x float> %2085, float %2087, i64 1, !dbg !19
  %2089 = select i1 %.not36, float %2083, float %2075, !dbg !7
  %2090 = insertelement <4 x float> %2088, float %2089, i64 2, !dbg !19
  %2091 = select i1 %.not36, float %2086, float %2076, !dbg !7
  %2092 = insertelement <4 x float> %2090, float %2091, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1976, !dbg !19
  %normalized = fmul <4 x float> %2092, %row.reciprocal, !dbg !19
  %2093 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %2094 = extractelement <4 x half> %2093, i64 3, !dbg !20
  %2095 = extractelement <4 x half> %2093, i64 2, !dbg !20
  %2096 = extractelement <4 x half> %2093, i64 1, !dbg !20
  %2097 = extractelement <4 x half> %2093, i64 0, !dbg !20
  %2098 = bitcast i32 %1972 to float, !dbg !7
  %2099 = select i1 %.not36, float %1962, float %2098, !dbg !7
  %2100 = insertelement <4 x float> poison, float %2099, i64 0, !dbg !19
  %2101 = bitcast i32 %1975 to float, !dbg !7
  %2102 = select i1 %.not36, float %1963, float %2101, !dbg !7
  %2103 = insertelement <4 x float> %2100, float %2102, i64 1, !dbg !19
  %2104 = select i1 %.not36, float %2098, float %1968, !dbg !7
  %2105 = insertelement <4 x float> %2103, float %2104, i64 2, !dbg !19
  %2106 = select i1 %.not36, float %2101, float %1969, !dbg !7
  %2107 = insertelement <4 x float> %2105, float %2106, i64 3, !dbg !19
  %normalized104 = fmul <4 x float> %2107, %row.reciprocal, !dbg !19
  %2108 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !20
  %2109 = extractelement <4 x half> %2108, i64 3, !dbg !20
  %2110 = extractelement <4 x half> %2108, i64 2, !dbg !20
  %2111 = extractelement <4 x half> %2108, i64 1, !dbg !20
  %2112 = extractelement <4 x half> %2108, i64 0, !dbg !20
  %2113 = bitcast i32 %1862 to float, !dbg !7
  %2114 = select i1 %.not36, float %1852, float %2113, !dbg !7
  %2115 = insertelement <4 x float> poison, float %2114, i64 0, !dbg !19
  %2116 = bitcast i32 %1865 to float, !dbg !7
  %2117 = select i1 %.not36, float %1853, float %2116, !dbg !7
  %2118 = insertelement <4 x float> %2115, float %2117, i64 1, !dbg !19
  %2119 = select i1 %.not36, float %2113, float %1858, !dbg !7
  %2120 = insertelement <4 x float> %2118, float %2119, i64 2, !dbg !19
  %2121 = select i1 %.not36, float %2116, float %1859, !dbg !7
  %2122 = insertelement <4 x float> %2120, float %2121, i64 3, !dbg !19
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1759, !dbg !19
  %normalized106 = fmul <4 x float> %2122, %row.reciprocal105, !dbg !19
  %2123 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !20
  %2124 = extractelement <4 x half> %2123, i64 3, !dbg !20
  %2125 = extractelement <4 x half> %2123, i64 2, !dbg !20
  %2126 = extractelement <4 x half> %2123, i64 1, !dbg !20
  %2127 = extractelement <4 x half> %2123, i64 0, !dbg !20
  %2128 = bitcast i32 %1755 to float, !dbg !7
  %2129 = select i1 %.not36, float %1745, float %2128, !dbg !7
  %2130 = insertelement <4 x float> poison, float %2129, i64 0, !dbg !19
  %2131 = bitcast i32 %1758 to float, !dbg !7
  %2132 = select i1 %.not36, float %1746, float %2131, !dbg !7
  %2133 = insertelement <4 x float> %2130, float %2132, i64 1, !dbg !19
  %2134 = select i1 %.not36, float %2128, float %1751, !dbg !7
  %2135 = insertelement <4 x float> %2133, float %2134, i64 2, !dbg !19
  %2136 = select i1 %.not36, float %2131, float %1752, !dbg !7
  %2137 = insertelement <4 x float> %2135, float %2136, i64 3, !dbg !19
  %normalized107 = fmul <4 x float> %2137, %row.reciprocal105, !dbg !19
  %2138 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !20
  %2139 = extractelement <4 x half> %2138, i64 3, !dbg !20
  %2140 = extractelement <4 x half> %2138, i64 2, !dbg !20
  %2141 = extractelement <4 x half> %2138, i64 1, !dbg !20
  %2142 = extractelement <4 x half> %2138, i64 0, !dbg !20
  %2143 = add nsw i64 %16, %22, !dbg !21
  %2144 = shl i64 %2143, 19, !dbg !21
  %2145 = and i64 %2144, 16252928, !dbg !21
  %address.thread1100 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1101 = zext nneg i32 %address.thread1100 to i64, !dbg !8
  %address.local1102 = add i64 %25, %address.local1101, !dbg !21
  %2146 = lshr i64 %address.local1101, 2, !dbg !21
  %address.local1104 = and i64 %2146, 48, !dbg !21
  %address.local1105 = add i64 %address.local1102, %address.local1104, !dbg !21
  %address.local1106 = and i64 %address.local1101, 240, !dbg !21
  %address.local1107 = sub i64 %address.local1105, %address.local1106, !dbg !21
  %address.local1108 = add i64 %address.local1107, %26, !dbg !21
  %address.local1109 = shl nuw nsw i64 %address.local1108, 7, !dbg !21
  %address.local1110 = add nuw nsw i64 %2145, %address.local1109, !dbg !21
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1112 = lshr i64 %address.local1101, 2, !dbg !11
  %address.local1113 = and i64 %address.local1112, 12, !dbg !11
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1113, !dbg !21
  store half %2142, ptr addrspace(1) %address.local1114, align 2, !dbg !21
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1116 = or disjoint i64 %address.local1113, 1, !dbg !21
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1115, i64 %address.local1116, !dbg !21
  store half %2141, ptr addrspace(1) %address.local1117, align 2, !dbg !21
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1119 = or disjoint i64 %address.local1113, 2, !dbg !21
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1118, i64 %address.local1119, !dbg !21
  store half %2140, ptr addrspace(1) %address.local1120, align 2, !dbg !21
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1122 = or disjoint i64 %address.local1113, 3, !dbg !21
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1121, i64 %address.local1122, !dbg !21
  store half %2139, ptr addrspace(1) %address.local1123, align 2, !dbg !21
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1125 = or disjoint i64 %address.local1113, 16, !dbg !21
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1124, i64 %address.local1125, !dbg !21
  store half %2127, ptr addrspace(1) %address.local1126, align 2, !dbg !21
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1128 = or disjoint i64 %address.local1113, 17, !dbg !21
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1128, !dbg !21
  store half %2126, ptr addrspace(1) %address.local1129, align 2, !dbg !21
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1131 = or disjoint i64 %address.local1113, 18, !dbg !21
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1130, i64 %address.local1131, !dbg !21
  store half %2125, ptr addrspace(1) %address.local1132, align 2, !dbg !21
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !21
  %address.local1134 = or disjoint i64 %address.local1113, 19, !dbg !21
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1133, i64 %address.local1134, !dbg !21
  store half %2124, ptr addrspace(1) %address.local1135, align 2, !dbg !21
  %address.local1136 = shl i64 %address.local1108, 7, !dbg !21
  %address.local1137 = add i64 %address.local1136, 8192, !dbg !21
  %address.local1138 = add nuw nsw i64 %2145, %address.local1137, !dbg !21
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1113, !dbg !21
  store half %2112, ptr addrspace(1) %address.local1140, align 2, !dbg !21
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1116, !dbg !21
  store half %2111, ptr addrspace(1) %address.local1142, align 2, !dbg !21
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1119, !dbg !21
  store half %2110, ptr addrspace(1) %address.local1144, align 2, !dbg !21
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1122, !dbg !21
  store half %2109, ptr addrspace(1) %address.local1146, align 2, !dbg !21
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1125, !dbg !21
  store half %2097, ptr addrspace(1) %address.local1148, align 2, !dbg !21
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1128, !dbg !21
  store half %2096, ptr addrspace(1) %address.local1150, align 2, !dbg !21
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1131, !dbg !21
  store half %2095, ptr addrspace(1) %address.local1152, align 2, !dbg !21
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1138, !dbg !21
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1134, !dbg !21
  store half %2094, ptr addrspace(1) %address.local1154, align 2, !dbg !21
  %2147 = insertelement <4 x float> poison, float %2053, i64 0, !dbg !19
  %2148 = insertelement <4 x float> %2147, float %2059, i64 1, !dbg !19
  %2149 = insertelement <4 x float> %2148, float %2054, i64 2, !dbg !19
  %2150 = insertelement <4 x float> %2149, float %2060, i64 3, !dbg !19
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1976, !dbg !19
  %normalized109 = fmul <4 x float> %2150, %row.reciprocal108, !dbg !19
  %2151 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !20
  %2152 = extractelement <4 x half> %2151, i64 3, !dbg !20
  %2153 = extractelement <4 x half> %2151, i64 2, !dbg !20
  %2154 = extractelement <4 x half> %2151, i64 1, !dbg !20
  %2155 = extractelement <4 x half> %2151, i64 0, !dbg !20
  %2156 = insertelement <4 x float> poison, float %1945, i64 0, !dbg !19
  %2157 = insertelement <4 x float> %2156, float %1951, i64 1, !dbg !19
  %2158 = insertelement <4 x float> %2157, float %1946, i64 2, !dbg !19
  %2159 = insertelement <4 x float> %2158, float %1952, i64 3, !dbg !19
  %normalized110 = fmul <4 x float> %2159, %row.reciprocal108, !dbg !19
  %2160 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %2161 = extractelement <4 x half> %2160, i64 3, !dbg !20
  %2162 = extractelement <4 x half> %2160, i64 2, !dbg !20
  %2163 = extractelement <4 x half> %2160, i64 1, !dbg !20
  %2164 = extractelement <4 x half> %2160, i64 0, !dbg !20
  %2165 = insertelement <4 x float> poison, float %1836, i64 0, !dbg !19
  %2166 = insertelement <4 x float> %2165, float %1842, i64 1, !dbg !19
  %2167 = insertelement <4 x float> %2166, float %1837, i64 2, !dbg !19
  %2168 = insertelement <4 x float> %2167, float %1843, i64 3, !dbg !19
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1759, !dbg !19
  %normalized112 = fmul <4 x float> %2168, %row.reciprocal111, !dbg !19
  %2169 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !20
  %2170 = extractelement <4 x half> %2169, i64 3, !dbg !20
  %2171 = extractelement <4 x half> %2169, i64 2, !dbg !20
  %2172 = extractelement <4 x half> %2169, i64 1, !dbg !20
  %2173 = extractelement <4 x half> %2169, i64 0, !dbg !20
  %2174 = insertelement <4 x float> poison, float %1728, i64 0, !dbg !19
  %2175 = insertelement <4 x float> %2174, float %1734, i64 1, !dbg !19
  %2176 = insertelement <4 x float> %2175, float %1729, i64 2, !dbg !19
  %2177 = insertelement <4 x float> %2176, float %1735, i64 3, !dbg !19
  %normalized113 = fmul <4 x float> %2177, %row.reciprocal111, !dbg !19
  %2178 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %2179 = extractelement <4 x half> %2178, i64 3, !dbg !20
  %2180 = extractelement <4 x half> %2178, i64 2, !dbg !20
  %2181 = extractelement <4 x half> %2178, i64 1, !dbg !20
  %2182 = extractelement <4 x half> %2178, i64 0, !dbg !20
  %2183 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1155 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1156 = zext nneg i32 %address.thread1155 to i64, !dbg !8
  %address.local1157 = add i64 %25, %address.local1156, !dbg !21
  %2184 = lshr i64 %address.local1156, 2, !dbg !21
  %address.local1159 = and i64 %2184, 48, !dbg !21
  %address.local1160 = add i64 %address.local1157, %address.local1159, !dbg !21
  %address.local1161 = and i64 %address.local1156, 240, !dbg !21
  %address.local1162 = sub i64 %address.local1160, %address.local1161, !dbg !21
  %address.local1163 = add i64 %address.local1162, %26, !dbg !21
  %address.local1164 = shl nuw nsw i64 %address.local1163, 7, !dbg !21
  %address.local1165 = add nuw nsw i64 %2183, %address.local1164, !dbg !21
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1167 = lshr i64 %address.local1156, 2, !dbg !11
  %address.local1168 = and i64 %address.local1167, 12, !dbg !11
  %address.local1169 = or disjoint i64 %address.local1168, 32, !dbg !21
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1166, i64 %address.local1169, !dbg !21
  store half %2182, ptr addrspace(1) %address.local1170, align 2, !dbg !21
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1172 = or disjoint i64 %address.local1168, 33, !dbg !21
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1171, i64 %address.local1172, !dbg !21
  store half %2181, ptr addrspace(1) %address.local1173, align 2, !dbg !21
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1175 = or disjoint i64 %address.local1168, 34, !dbg !21
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1174, i64 %address.local1175, !dbg !21
  store half %2180, ptr addrspace(1) %address.local1176, align 2, !dbg !21
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1178 = or disjoint i64 %address.local1168, 35, !dbg !21
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1177, i64 %address.local1178, !dbg !21
  store half %2179, ptr addrspace(1) %address.local1179, align 2, !dbg !21
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1181 = or disjoint i64 %address.local1168, 48, !dbg !21
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1180, i64 %address.local1181, !dbg !21
  store half %2173, ptr addrspace(1) %address.local1182, align 2, !dbg !21
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1184 = or disjoint i64 %address.local1168, 49, !dbg !21
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1184, !dbg !21
  store half %2172, ptr addrspace(1) %address.local1185, align 2, !dbg !21
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1187 = or disjoint i64 %address.local1168, 50, !dbg !21
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1186, i64 %address.local1187, !dbg !21
  store half %2171, ptr addrspace(1) %address.local1188, align 2, !dbg !21
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1165, !dbg !21
  %address.local1190 = or i64 %address.local1167, 51, !dbg !21
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1189, i64 %address.local1190, !dbg !21
  store half %2170, ptr addrspace(1) %address.local1191, align 2, !dbg !21
  %address.local1192 = shl i64 %address.local1163, 7, !dbg !21
  %address.local1193 = add i64 %address.local1192, 8192, !dbg !21
  %address.local1194 = add nuw nsw i64 %2183, %address.local1193, !dbg !21
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1195, i64 %address.local1169, !dbg !21
  store half %2164, ptr addrspace(1) %address.local1196, align 2, !dbg !21
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1197, i64 %address.local1172, !dbg !21
  store half %2163, ptr addrspace(1) %address.local1198, align 2, !dbg !21
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1199, i64 %address.local1175, !dbg !21
  store half %2162, ptr addrspace(1) %address.local1200, align 2, !dbg !21
  %address.local1201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1201, i64 %address.local1178, !dbg !21
  store half %2161, ptr addrspace(1) %address.local1202, align 2, !dbg !21
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1203, i64 %address.local1181, !dbg !21
  store half %2155, ptr addrspace(1) %address.local1204, align 2, !dbg !21
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1205, i64 %address.local1184, !dbg !21
  store half %2154, ptr addrspace(1) %address.local1206, align 2, !dbg !21
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1207, i64 %address.local1187, !dbg !21
  store half %2153, ptr addrspace(1) %address.local1208, align 2, !dbg !21
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1194, !dbg !21
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1209, i64 %address.local1190, !dbg !21
  store half %2152, ptr addrspace(1) %address.local1210, align 2, !dbg !21
  %2185 = insertelement <4 x float> poison, float %2025, i64 0, !dbg !19
  %2186 = insertelement <4 x float> %2185, float %2031, i64 1, !dbg !19
  %2187 = insertelement <4 x float> %2186, float %2026, i64 2, !dbg !19
  %2188 = insertelement <4 x float> %2187, float %2032, i64 3, !dbg !19
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1976, !dbg !19
  %normalized115 = fmul <4 x float> %2188, %row.reciprocal114, !dbg !19
  %2189 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !20
  %2190 = extractelement <4 x half> %2189, i64 3, !dbg !20
  %2191 = extractelement <4 x half> %2189, i64 2, !dbg !20
  %2192 = extractelement <4 x half> %2189, i64 1, !dbg !20
  %2193 = extractelement <4 x half> %2189, i64 0, !dbg !20
  %2194 = insertelement <4 x float> poison, float %1916, i64 0, !dbg !19
  %2195 = insertelement <4 x float> %2194, float %1922, i64 1, !dbg !19
  %2196 = insertelement <4 x float> %2195, float %1917, i64 2, !dbg !19
  %2197 = insertelement <4 x float> %2196, float %1923, i64 3, !dbg !19
  %normalized116 = fmul <4 x float> %2197, %row.reciprocal114, !dbg !19
  %2198 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %2199 = extractelement <4 x half> %2198, i64 3, !dbg !20
  %2200 = extractelement <4 x half> %2198, i64 2, !dbg !20
  %2201 = extractelement <4 x half> %2198, i64 1, !dbg !20
  %2202 = extractelement <4 x half> %2198, i64 0, !dbg !20
  %2203 = insertelement <4 x float> poison, float %1808, i64 0, !dbg !19
  %2204 = insertelement <4 x float> %2203, float %1814, i64 1, !dbg !19
  %2205 = insertelement <4 x float> %2204, float %1809, i64 2, !dbg !19
  %2206 = insertelement <4 x float> %2205, float %1815, i64 3, !dbg !19
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1759, !dbg !19
  %normalized118 = fmul <4 x float> %2206, %row.reciprocal117, !dbg !19
  %2207 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !20
  %2208 = extractelement <4 x half> %2207, i64 3, !dbg !20
  %2209 = extractelement <4 x half> %2207, i64 2, !dbg !20
  %2210 = extractelement <4 x half> %2207, i64 1, !dbg !20
  %2211 = extractelement <4 x half> %2207, i64 0, !dbg !20
  %2212 = insertelement <4 x float> poison, float %1699, i64 0, !dbg !19
  %2213 = insertelement <4 x float> %2212, float %1705, i64 1, !dbg !19
  %2214 = insertelement <4 x float> %2213, float %1700, i64 2, !dbg !19
  %2215 = insertelement <4 x float> %2214, float %1706, i64 3, !dbg !19
  %normalized119 = fmul <4 x float> %2215, %row.reciprocal117, !dbg !19
  %2216 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %2217 = extractelement <4 x half> %2216, i64 3, !dbg !20
  %2218 = extractelement <4 x half> %2216, i64 2, !dbg !20
  %2219 = extractelement <4 x half> %2216, i64 1, !dbg !20
  %2220 = extractelement <4 x half> %2216, i64 0, !dbg !20
  %2221 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1211 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1212 = zext nneg i32 %address.thread1211 to i64, !dbg !8
  %address.local1213 = add i64 %25, %address.local1212, !dbg !21
  %2222 = lshr i64 %address.local1212, 2, !dbg !21
  %address.local1215 = and i64 %2222, 48, !dbg !21
  %address.local1216 = add i64 %address.local1213, %address.local1215, !dbg !21
  %address.local1217 = and i64 %address.local1212, 240, !dbg !21
  %address.local1218 = sub i64 %address.local1216, %address.local1217, !dbg !21
  %address.local1219 = add i64 %address.local1218, %26, !dbg !21
  %address.local1220 = shl nuw nsw i64 %address.local1219, 7, !dbg !21
  %address.local1221 = add nuw nsw i64 %2221, %address.local1220, !dbg !21
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1223 = lshr i64 %address.local1212, 2, !dbg !11
  %address.local1224 = and i64 %address.local1223, 12, !dbg !11
  %address.local1225 = or disjoint i64 %address.local1224, 64, !dbg !21
  %address.local1226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1222, i64 %address.local1225, !dbg !21
  store half %2220, ptr addrspace(1) %address.local1226, align 2, !dbg !21
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1228 = or disjoint i64 %address.local1224, 65, !dbg !21
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1227, i64 %address.local1228, !dbg !21
  store half %2219, ptr addrspace(1) %address.local1229, align 2, !dbg !21
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1231 = or disjoint i64 %address.local1224, 66, !dbg !21
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1230, i64 %address.local1231, !dbg !21
  store half %2218, ptr addrspace(1) %address.local1232, align 2, !dbg !21
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1234 = or disjoint i64 %address.local1224, 67, !dbg !21
  %address.local1235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1233, i64 %address.local1234, !dbg !21
  store half %2217, ptr addrspace(1) %address.local1235, align 2, !dbg !21
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1237 = or disjoint i64 %address.local1224, 80, !dbg !21
  %address.local1238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1236, i64 %address.local1237, !dbg !21
  store half %2211, ptr addrspace(1) %address.local1238, align 2, !dbg !21
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1240 = or disjoint i64 %address.local1224, 81, !dbg !21
  %address.local1241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1239, i64 %address.local1240, !dbg !21
  store half %2210, ptr addrspace(1) %address.local1241, align 2, !dbg !21
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1243 = or disjoint i64 %address.local1224, 82, !dbg !21
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1242, i64 %address.local1243, !dbg !21
  store half %2209, ptr addrspace(1) %address.local1244, align 2, !dbg !21
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1221, !dbg !21
  %address.local1246 = or disjoint i64 %address.local1224, 83, !dbg !21
  %address.local1247 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1245, i64 %address.local1246, !dbg !21
  store half %2208, ptr addrspace(1) %address.local1247, align 2, !dbg !21
  %address.local1248 = shl i64 %address.local1219, 7, !dbg !21
  %address.local1249 = add i64 %address.local1248, 8192, !dbg !21
  %address.local1250 = add nuw nsw i64 %2221, %address.local1249, !dbg !21
  %address.local1251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1252 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1251, i64 %address.local1225, !dbg !21
  store half %2202, ptr addrspace(1) %address.local1252, align 2, !dbg !21
  %address.local1253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1253, i64 %address.local1228, !dbg !21
  store half %2201, ptr addrspace(1) %address.local1254, align 2, !dbg !21
  %address.local1255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1255, i64 %address.local1231, !dbg !21
  store half %2200, ptr addrspace(1) %address.local1256, align 2, !dbg !21
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1257, i64 %address.local1234, !dbg !21
  store half %2199, ptr addrspace(1) %address.local1258, align 2, !dbg !21
  %address.local1259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1259, i64 %address.local1237, !dbg !21
  store half %2193, ptr addrspace(1) %address.local1260, align 2, !dbg !21
  %address.local1261 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1261, i64 %address.local1240, !dbg !21
  store half %2192, ptr addrspace(1) %address.local1262, align 2, !dbg !21
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1263, i64 %address.local1243, !dbg !21
  store half %2191, ptr addrspace(1) %address.local1264, align 2, !dbg !21
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1250, !dbg !21
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1265, i64 %address.local1246, !dbg !21
  store half %2190, ptr addrspace(1) %address.local1266, align 2, !dbg !21
  %2223 = insertelement <4 x float> poison, float %1997, i64 0, !dbg !19
  %2224 = insertelement <4 x float> %2223, float %2003, i64 1, !dbg !19
  %2225 = insertelement <4 x float> %2224, float %1998, i64 2, !dbg !19
  %2226 = insertelement <4 x float> %2225, float %2004, i64 3, !dbg !19
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1976, !dbg !19
  %normalized121 = fmul <4 x float> %2226, %row.reciprocal120, !dbg !19
  %2227 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !20
  %2228 = extractelement <4 x half> %2227, i64 3, !dbg !20
  %2229 = extractelement <4 x half> %2227, i64 2, !dbg !20
  %2230 = extractelement <4 x half> %2227, i64 1, !dbg !20
  %2231 = extractelement <4 x half> %2227, i64 0, !dbg !20
  %2232 = insertelement <4 x float> poison, float %1887, i64 0, !dbg !19
  %2233 = insertelement <4 x float> %2232, float %1893, i64 1, !dbg !19
  %2234 = insertelement <4 x float> %2233, float %1888, i64 2, !dbg !19
  %2235 = insertelement <4 x float> %2234, float %1894, i64 3, !dbg !19
  %normalized122 = fmul <4 x float> %2235, %row.reciprocal120, !dbg !19
  %2236 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %2237 = extractelement <4 x half> %2236, i64 3, !dbg !20
  %2238 = extractelement <4 x half> %2236, i64 2, !dbg !20
  %2239 = extractelement <4 x half> %2236, i64 1, !dbg !20
  %2240 = extractelement <4 x half> %2236, i64 0, !dbg !20
  %2241 = insertelement <4 x float> poison, float %1780, i64 0, !dbg !19
  %2242 = insertelement <4 x float> %2241, float %1786, i64 1, !dbg !19
  %2243 = insertelement <4 x float> %2242, float %1781, i64 2, !dbg !19
  %2244 = insertelement <4 x float> %2243, float %1787, i64 3, !dbg !19
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1759, !dbg !19
  %normalized124 = fmul <4 x float> %2244, %row.reciprocal123, !dbg !19
  %2245 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %2246 = extractelement <4 x half> %2245, i64 3, !dbg !20
  %2247 = extractelement <4 x half> %2245, i64 2, !dbg !20
  %2248 = extractelement <4 x half> %2245, i64 1, !dbg !20
  %2249 = extractelement <4 x half> %2245, i64 0, !dbg !20
  %2250 = insertelement <4 x float> poison, float %1670, i64 0, !dbg !19
  %2251 = insertelement <4 x float> %2250, float %1676, i64 1, !dbg !19
  %2252 = insertelement <4 x float> %2251, float %1671, i64 2, !dbg !19
  %2253 = insertelement <4 x float> %2252, float %1677, i64 3, !dbg !19
  %normalized125 = fmul <4 x float> %2253, %row.reciprocal123, !dbg !19
  %2254 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %2255 = extractelement <4 x half> %2254, i64 3, !dbg !20
  %2256 = extractelement <4 x half> %2254, i64 2, !dbg !20
  %2257 = extractelement <4 x half> %2254, i64 1, !dbg !20
  %2258 = extractelement <4 x half> %2254, i64 0, !dbg !20
  %2259 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1267 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1268 = zext nneg i32 %address.thread1267 to i64, !dbg !8
  %address.local1269 = add i64 %25, %address.local1268, !dbg !21
  %2260 = lshr i64 %address.local1268, 2, !dbg !21
  %address.local1271 = and i64 %2260, 48, !dbg !21
  %address.local1272 = add i64 %address.local1269, %address.local1271, !dbg !21
  %address.local1273 = and i64 %address.local1268, 240, !dbg !21
  %address.local1274 = sub i64 %address.local1272, %address.local1273, !dbg !21
  %address.local1275 = add i64 %address.local1274, %26, !dbg !21
  %address.local1276 = shl nuw nsw i64 %address.local1275, 7, !dbg !21
  %address.local1277 = add nuw nsw i64 %2259, %address.local1276, !dbg !21
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1279 = lshr i64 %address.local1268, 2, !dbg !11
  %address.local1280 = and i64 %address.local1279, 12, !dbg !11
  %address.local1281 = or disjoint i64 %address.local1280, 96, !dbg !21
  %address.local1282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1278, i64 %address.local1281, !dbg !21
  store half %2258, ptr addrspace(1) %address.local1282, align 2, !dbg !21
  %address.local1283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1284 = or disjoint i64 %address.local1280, 97, !dbg !21
  %address.local1285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1283, i64 %address.local1284, !dbg !21
  store half %2257, ptr addrspace(1) %address.local1285, align 2, !dbg !21
  %address.local1286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1287 = or disjoint i64 %address.local1280, 98, !dbg !21
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1286, i64 %address.local1287, !dbg !21
  store half %2256, ptr addrspace(1) %address.local1288, align 2, !dbg !21
  %address.local1289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1290 = or disjoint i64 %address.local1280, 99, !dbg !21
  %address.local1291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1289, i64 %address.local1290, !dbg !21
  store half %2255, ptr addrspace(1) %address.local1291, align 2, !dbg !21
  %address.local1292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1293 = or disjoint i64 %address.local1280, 112, !dbg !21
  %address.local1294 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1292, i64 %address.local1293, !dbg !21
  store half %2249, ptr addrspace(1) %address.local1294, align 2, !dbg !21
  %address.local1295 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1296 = or disjoint i64 %address.local1280, 113, !dbg !21
  %address.local1297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1295, i64 %address.local1296, !dbg !21
  store half %2248, ptr addrspace(1) %address.local1297, align 2, !dbg !21
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1299 = or disjoint i64 %address.local1280, 114, !dbg !21
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1298, i64 %address.local1299, !dbg !21
  store half %2247, ptr addrspace(1) %address.local1300, align 2, !dbg !21
  %address.local1301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1277, !dbg !21
  %address.local1302 = or i64 %address.local1279, 115, !dbg !21
  %address.local1303 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1301, i64 %address.local1302, !dbg !21
  store half %2246, ptr addrspace(1) %address.local1303, align 2, !dbg !21
  %address.local1304 = shl i64 %address.local1275, 7, !dbg !21
  %address.local1305 = add i64 %address.local1304, 8192, !dbg !21
  %address.local1306 = add nuw nsw i64 %2259, %address.local1305, !dbg !21
  %address.local1307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1307, i64 %address.local1281, !dbg !21
  store half %2240, ptr addrspace(1) %address.local1308, align 2, !dbg !21
  %address.local1309 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1309, i64 %address.local1284, !dbg !21
  store half %2239, ptr addrspace(1) %address.local1310, align 2, !dbg !21
  %address.local1311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1311, i64 %address.local1287, !dbg !21
  store half %2238, ptr addrspace(1) %address.local1312, align 2, !dbg !21
  %address.local1313 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1314 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1313, i64 %address.local1290, !dbg !21
  store half %2237, ptr addrspace(1) %address.local1314, align 2, !dbg !21
  %address.local1315 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1315, i64 %address.local1293, !dbg !21
  store half %2231, ptr addrspace(1) %address.local1316, align 2, !dbg !21
  %address.local1317 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1318 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1317, i64 %address.local1296, !dbg !21
  store half %2230, ptr addrspace(1) %address.local1318, align 2, !dbg !21
  %address.local1319 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1320 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1319, i64 %address.local1299, !dbg !21
  store half %2229, ptr addrspace(1) %address.local1320, align 2, !dbg !21
  %address.local1321 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1306, !dbg !21
  %address.local1322 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1321, i64 %address.local1302, !dbg !21
  store half %2228, ptr addrspace(1) %address.local1322, align 2, !dbg !21
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/gemm_result_layout_comparison/cfg_fusion_pass/run-20261027-151454-z6uud_gi/grid-000002")
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
