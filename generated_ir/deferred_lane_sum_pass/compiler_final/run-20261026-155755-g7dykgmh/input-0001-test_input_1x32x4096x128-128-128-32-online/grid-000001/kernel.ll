; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/deferred_lane_sum_pass/compiler_final/run-20261026-155755-g7dykgmh/input-0001-test_input_1x32x4096x128-128-128-32-online/grid-000001/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/deferred_lane_sum_pass/compiler_final/run-20261026-155755-g7dykgmh/input-0001-test_input_1x32x4096x128-128-128-32-online/grid-000001/input.mlir"

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
  %address.local495 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local496 = and i64 %address.local495, 1536, !dbg !8
  %address.local497 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local496, !dbg !8
  %address.local498 = ptrtoint ptr addrspace(3) %address.local497 to i64, !dbg !8
  %address.local499 = trunc i64 %address.local498 to i32, !dbg !8
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local499), !dbg !8
  %35 = zext i32 %34 to i64, !dbg !8
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !8
  %address.local500 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local501 = and i64 %address.local500, 16256, !dbg !8
  %address.local502 = lshr i64 %address.local, 4, !dbg !8
  %address.local503 = and i64 %address.local502, 8, !dbg !8
  %address.local504 = or disjoint i64 %address.local501, %address.local503, !dbg !8
  %address.local505 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local504, !dbg !8
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %38 = ptrtoint ptr addrspace(1) %address.local505 to i64, !dbg !8
  %39 = sub i64 %38, %37, !dbg !8
  %dma.buffer.offset1768 = trunc i64 %39 to i32, !dbg !8
  %40 = trunc i64 %37 to i32, !dbg !8
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !8
  %42 = lshr i64 %37, 32, !dbg !8
  %43 = trunc i64 %42 to i32, !dbg !8
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !8
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1769 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !8
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !8
  %dma.buffer.lds1770 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1770, i32 %dma.buffer.offset1768, <4 x i32> %dma.buffer.rsrc1769) #10, !dbg !8
  %address.local506 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local507 = shl nuw nsw i64 %address.local506, 3, !dbg !8
  %address.local508 = and i64 %address.local507, 3584, !dbg !8
  %address.local509 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local508, !dbg !8
  %address.local510 = ptrtoint ptr addrspace(3) %address.local509 to i64, !dbg !8
  %address.local511 = trunc i64 %address.local510 to i32, !dbg !8
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local511), !dbg !8
  %49 = zext i32 %48 to i64, !dbg !8
  %50 = inttoptr i64 %49 to ptr addrspace(3), !dbg !8
  %address.local512 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local513 = and i64 %address.local512, 16256, !dbg !8
  %address.local514 = lshr i64 %address.local506, 4, !dbg !8
  %address.local515 = and i64 %address.local514, 24, !dbg !8
  %address.local516 = or disjoint i64 %address.local513, %address.local515, !dbg !8
  %address.local517 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local516, !dbg !8
  %51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %52 = ptrtoint ptr addrspace(1) %address.local517 to i64, !dbg !8
  %53 = sub i64 %52, %51, !dbg !8
  %dma.buffer.offset1765 = trunc i64 %53 to i32, !dbg !8
  %54 = trunc i64 %51 to i32, !dbg !8
  %55 = insertelement <4 x i32> poison, i32 %54, i64 0, !dbg !8
  %56 = lshr i64 %51, 32, !dbg !8
  %57 = trunc i64 %56 to i32, !dbg !8
  %58 = insertelement <4 x i32> %55, i32 %57, i64 1, !dbg !8
  %59 = insertelement <4 x i32> %58, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1766 = insertelement <4 x i32> %59, i32 131072, i64 3, !dbg !8
  %60 = ptrtoint ptr addrspace(3) %50 to i32, !dbg !8
  %dma.buffer.lds1767 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !8
  %61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1767, i32 %dma.buffer.offset1765, <4 x i32> %dma.buffer.rsrc1766) #10, !dbg !8
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
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local496, !dbg !8
  %address.local519 = ptrtoint ptr addrspace(3) %address.local518 to i64, !dbg !8
  %address.local520 = trunc i64 %address.local519 to i32, !dbg !8
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local520), !dbg !8
  %69 = zext i32 %68 to i64, !dbg !8
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !8
  %address.local521 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local504, !dbg !8
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %72 = ptrtoint ptr addrspace(1) %address.local521 to i64, !dbg !8
  %73 = sub i64 %72, %71, !dbg !8
  %dma.buffer.offset1762 = trunc i64 %73 to i32, !dbg !8
  %74 = trunc i64 %71 to i32, !dbg !8
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !8
  %76 = lshr i64 %71, 32, !dbg !8
  %77 = trunc i64 %76 to i32, !dbg !8
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !8
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1763 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !8
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !8
  %dma.buffer.lds1764 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !8
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1764, i32 %dma.buffer.offset1762, <4 x i32> %dma.buffer.rsrc1763) #10, !dbg !8
  %address.local522 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local508, !dbg !8
  %address.local523 = ptrtoint ptr addrspace(3) %address.local522 to i64, !dbg !8
  %address.local524 = trunc i64 %address.local523 to i32, !dbg !8
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local524), !dbg !8
  %83 = zext i32 %82 to i64, !dbg !8
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !8
  %address.local525 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local516, !dbg !8
  %85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %86 = ptrtoint ptr addrspace(1) %address.local525 to i64, !dbg !8
  %87 = sub i64 %86, %85, !dbg !8
  %dma.buffer.offset1759 = trunc i64 %87 to i32, !dbg !8
  %88 = trunc i64 %85 to i32, !dbg !8
  %89 = insertelement <4 x i32> poison, i32 %88, i64 0, !dbg !8
  %90 = lshr i64 %85, 32, !dbg !8
  %91 = trunc i64 %90 to i32, !dbg !8
  %92 = insertelement <4 x i32> %89, i32 %91, i64 1, !dbg !8
  %93 = insertelement <4 x i32> %92, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1760 = insertelement <4 x i32> %93, i32 131072, i64 3, !dbg !8
  %94 = ptrtoint ptr addrspace(3) %84 to i32, !dbg !8
  %dma.buffer.lds1761 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !8
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1761, i32 %dma.buffer.offset1759, <4 x i32> %dma.buffer.rsrc1760) #10, !dbg !8
  %address.local526 = lshr i32 %address.thread, 2, !dbg !11
  %address.local527 = and i32 %address.local526, 4, !dbg !11
  %address.local528 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local529 = and i32 %address.local528, 384, !dbg !11
  %address.local530 = or disjoint i32 %address.local527, %address.local529, !dbg !11
  %address.local531 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local532 = and i32 %address.local531, 120, !dbg !11
  %address.local533 = or disjoint i32 %address.local530, %address.local532, !dbg !11
  %address.local534 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local535 = and i32 %address.local534, 1024, !dbg !11
  %address.local536 = or disjoint i32 %address.local533, %address.local535, !dbg !11
  %address.local537 = zext nneg i32 %address.local536 to i64, !dbg !11
  %address.local538 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local537, !dbg !11
  %96 = load <4 x half>, ptr addrspace(3) %address.local538, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local538, i64 4096, !dbg !11
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %98 = fmul <4 x half> %96, splat (half 0xH3015), !dbg !11
  %lds.const.gep1569 = getelementptr i8, ptr addrspace(3) %address.local538, i64 1024, !dbg !11
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1569, align 8, !dbg !11
  %100 = fmul <4 x half> %97, splat (half 0xH3015), !dbg !11
  %lds.const.gep1570 = getelementptr i8, ptr addrspace(3) %address.local538, i64 5120, !dbg !11
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1570, align 8, !dbg !11
  %102 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !11
  %103 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %105 = getelementptr i8, ptr addrspace(1) %104, i64 128, !dbg !8
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local499), !dbg !8
  %107 = zext i32 %106 to i64, !dbg !8
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !8
  %address.local581 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local504, !dbg !8
  %109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %110 = ptrtoint ptr addrspace(1) %address.local581 to i64, !dbg !8
  %111 = sub i64 %110, %109, !dbg !8
  %dma.buffer.offset1756 = trunc i64 %111 to i32, !dbg !8
  %112 = trunc i64 %109 to i32, !dbg !8
  %113 = insertelement <4 x i32> poison, i32 %112, i64 0, !dbg !8
  %114 = lshr i64 %109, 32, !dbg !8
  %115 = trunc i64 %114 to i32, !dbg !8
  %116 = insertelement <4 x i32> %113, i32 %115, i64 1, !dbg !8
  %117 = insertelement <4 x i32> %116, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1757 = insertelement <4 x i32> %117, i32 131072, i64 3, !dbg !8
  %118 = ptrtoint ptr addrspace(3) %108 to i32, !dbg !8
  %dma.buffer.lds1758 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !8
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1758, i32 %dma.buffer.offset1756, <4 x i32> %dma.buffer.rsrc1757) #10, !dbg !8
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local511), !dbg !8
  %121 = zext i32 %120 to i64, !dbg !8
  %122 = inttoptr i64 %121 to ptr addrspace(3), !dbg !8
  %address.local582 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local516, !dbg !8
  %123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %124 = ptrtoint ptr addrspace(1) %address.local582 to i64, !dbg !8
  %125 = sub i64 %124, %123, !dbg !8
  %dma.buffer.offset1753 = trunc i64 %125 to i32, !dbg !8
  %126 = trunc i64 %123 to i32, !dbg !8
  %127 = insertelement <4 x i32> poison, i32 %126, i64 0, !dbg !8
  %128 = lshr i64 %123, 32, !dbg !8
  %129 = trunc i64 %128 to i32, !dbg !8
  %130 = insertelement <4 x i32> %127, i32 %129, i64 1, !dbg !8
  %131 = insertelement <4 x i32> %130, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1754 = insertelement <4 x i32> %131, i32 131072, i64 3, !dbg !8
  %132 = ptrtoint ptr addrspace(3) %122 to i32, !dbg !8
  %dma.buffer.lds1755 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !8
  %133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1755, i32 %dma.buffer.offset1753, <4 x i32> %dma.buffer.rsrc1754) #10, !dbg !8
  %lds.const.gep1571 = getelementptr i8, ptr addrspace(3) %address.local538, i64 8192, !dbg !11
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1571, align 8, !dbg !11
  %lds.const.gep1572 = getelementptr i8, ptr addrspace(3) %address.local538, i64 12288, !dbg !11
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1572, align 8, !dbg !11
  %lds.const.gep1573 = getelementptr i8, ptr addrspace(3) %address.local538, i64 9216, !dbg !11
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1573, align 8, !dbg !11
  %137 = fmul <4 x half> %134, splat (half 0xH3015), !dbg !11
  %lds.const.gep1574 = getelementptr i8, ptr addrspace(3) %address.local538, i64 13312, !dbg !11
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep1574, align 8, !dbg !11
  %139 = fmul <4 x half> %135, splat (half 0xH3015), !dbg !11
  %140 = fmul <4 x half> %136, splat (half 0xH3015), !dbg !11
  %141 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %143 = getelementptr i8, ptr addrspace(1) %142, i64 192, !dbg !8
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local520), !dbg !8
  %145 = zext i32 %144 to i64, !dbg !8
  %146 = inttoptr i64 %145 to ptr addrspace(3), !dbg !8
  %address.local587 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local504, !dbg !8
  %147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %148 = ptrtoint ptr addrspace(1) %address.local587 to i64, !dbg !8
  %149 = sub i64 %148, %147, !dbg !8
  %dma.buffer.offset1750 = trunc i64 %149 to i32, !dbg !8
  %150 = trunc i64 %147 to i32, !dbg !8
  %151 = insertelement <4 x i32> poison, i32 %150, i64 0, !dbg !8
  %152 = lshr i64 %147, 32, !dbg !8
  %153 = trunc i64 %152 to i32, !dbg !8
  %154 = insertelement <4 x i32> %151, i32 %153, i64 1, !dbg !8
  %155 = insertelement <4 x i32> %154, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1751 = insertelement <4 x i32> %155, i32 131072, i64 3, !dbg !8
  %156 = ptrtoint ptr addrspace(3) %146 to i32, !dbg !8
  %dma.buffer.lds1752 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %156), !dbg !8
  %157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1752, i32 %dma.buffer.offset1750, <4 x i32> %dma.buffer.rsrc1751) #10, !dbg !8
  %158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local524), !dbg !8
  %159 = zext i32 %158 to i64, !dbg !8
  %160 = inttoptr i64 %159 to ptr addrspace(3), !dbg !8
  %address.local588 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local516, !dbg !8
  %161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %162 = ptrtoint ptr addrspace(1) %address.local588 to i64, !dbg !8
  %163 = sub i64 %162, %161, !dbg !8
  %dma.buffer.offset1747 = trunc i64 %163 to i32, !dbg !8
  %164 = trunc i64 %161 to i32, !dbg !8
  %165 = insertelement <4 x i32> poison, i32 %164, i64 0, !dbg !8
  %166 = lshr i64 %161, 32, !dbg !8
  %167 = trunc i64 %166 to i32, !dbg !8
  %168 = insertelement <4 x i32> %165, i32 %167, i64 1, !dbg !8
  %169 = insertelement <4 x i32> %168, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1748 = insertelement <4 x i32> %169, i32 131072, i64 3, !dbg !8
  %170 = ptrtoint ptr addrspace(3) %160 to i32, !dbg !8
  %dma.buffer.lds1749 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !8
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1749, i32 %dma.buffer.offset1747, <4 x i32> %dma.buffer.rsrc1748) #10, !dbg !8
  %172 = load <4 x half>, ptr addrspace(3) %address.local538, align 8, !dbg !11
  %lds.const.gep1575 = getelementptr i8, ptr addrspace(3) %address.local538, i64 4096, !dbg !11
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1575, align 8, !dbg !11
  %lds.const.gep1576 = getelementptr i8, ptr addrspace(3) %address.local538, i64 1024, !dbg !11
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1576, align 8, !dbg !11
  %175 = fmul <4 x half> %172, splat (half 0xH3015), !dbg !11
  %lds.const.gep1577 = getelementptr i8, ptr addrspace(3) %address.local538, i64 5120, !dbg !11
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1577, align 8, !dbg !11
  %177 = fmul <4 x half> %173, splat (half 0xH3015), !dbg !11
  %178 = fmul <4 x half> %174, splat (half 0xH3015), !dbg !11
  %179 = fmul <4 x half> %176, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1578 = getelementptr i8, ptr addrspace(3) %address.local538, i64 8192, !dbg !11
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1578, align 8, !dbg !11
  %lds.const.gep1579 = getelementptr i8, ptr addrspace(3) %address.local538, i64 12288, !dbg !11
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1579, align 8, !dbg !11
  %lds.const.gep1580 = getelementptr i8, ptr addrspace(3) %address.local538, i64 9216, !dbg !11
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1580, align 8, !dbg !11
  %183 = fmul <4 x half> %180, splat (half 0xH3015), !dbg !11
  %lds.const.gep1581 = getelementptr i8, ptr addrspace(3) %address.local538, i64 13312, !dbg !11
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1581, align 8, !dbg !11
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
  %index.compact435 = zext nneg i32 %200 to i64, !dbg !17
  %201 = or disjoint i64 %199, %index.compact435, !dbg !17
  %202 = add nuw nsw i64 %201, 1, !dbg !17
  %203 = sdiv i64 %202, 2, !dbg !17
  br label %204, !dbg !7

204:                                              ; preds = %845, %4
  %205 = phi i64 [ %1622, %845 ], [ 0, %4 ], !dbg !7
  %206 = phi [2 x <8 x float>] [ %1607, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %207 = phi [2 x <1 x float>] [ %1213, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %208 = phi [2 x <1 x float>] [ %1609, %845 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %209 = phi [2 x <8 x float>] [ %1613, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %210 = phi [2 x <8 x float>] [ %1617, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %211 = phi [2 x <8 x float>] [ %1621, %845 ], [ zeroinitializer, %4 ], !dbg !7
  %212 = icmp slt i64 %205, %191, !dbg !7
  br i1 %212, label %213, label %1623, !dbg !7

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
  %address.thread589 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local590 = zext nneg i32 %address.thread589 to i64, !dbg !8
  %address.local591 = shl nuw nsw i64 %address.local590, 3, !dbg !8
  %address.local592 = and i64 %address.local591, 1536, !dbg !8
  %address.local593 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local592, !dbg !8
  %address.local594 = ptrtoint ptr addrspace(3) %address.local593 to i64, !dbg !8
  %address.local595 = trunc i64 %address.local594 to i32, !dbg !8
  %227 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local595), !dbg !16
  %228 = zext i32 %227 to i64, !dbg !16
  %229 = inttoptr i64 %228 to ptr addrspace(3), !dbg !16
  %address.local596 = lshr i64 %address.local590, 4, !dbg !16
  %address.local597 = and i64 %address.local596, 8, !dbg !16
  %address.local598 = and i64 %address.local590, 123, !dbg !16
  %address.local599 = lshr i64 %address.local590, 2, !dbg !16
  %address.local600 = and i64 %address.local599, 4, !dbg !16
  %address.local601 = add nuw nsw i64 %address.local600, %address.local590, !dbg !16
  %address.local602 = and i64 %address.local601, 4, !dbg !16
  %address.local603 = or disjoint i64 %address.local598, %address.local602, !dbg !16
  %address.local604 = shl nuw nsw i64 %address.local603, 7, !dbg !16
  %address.local605 = or disjoint i64 %address.local597, %address.local604, !dbg !16
  %address.local606 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %address.local605, !dbg !16
  %230 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %231 = ptrtoint ptr addrspace(1) %address.local606 to i64, !dbg !16
  %232 = sub i64 %231, %230, !dbg !16
  %dma.buffer.offset1744 = trunc i64 %232 to i32, !dbg !16
  %233 = trunc i64 %230 to i32, !dbg !16
  %234 = insertelement <4 x i32> poison, i32 %233, i64 0, !dbg !16
  %235 = lshr i64 %230, 32, !dbg !16
  %236 = trunc i64 %235 to i32, !dbg !16
  %237 = insertelement <4 x i32> %234, i32 %236, i64 1, !dbg !16
  %238 = insertelement <4 x i32> %237, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1745 = insertelement <4 x i32> %238, i32 131072, i64 3, !dbg !16
  %239 = ptrtoint ptr addrspace(3) %229 to i32, !dbg !16
  %dma.buffer.lds1746 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %239), !dbg !16
  %240 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1746, i32 %dma.buffer.offset1744, <4 x i32> %dma.buffer.rsrc1745) #10, !dbg !16
  %address.local607 = or disjoint i64 %address.local590, 256, !dbg !8
  %address.local608 = shl nuw nsw i64 %address.local607, 3, !dbg !8
  %address.local609 = and i64 %address.local608, 3584, !dbg !8
  %address.local610 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local609, !dbg !8
  %address.local611 = ptrtoint ptr addrspace(3) %address.local610 to i64, !dbg !8
  %address.local612 = trunc i64 %address.local611 to i32, !dbg !8
  %241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local612), !dbg !16
  %242 = zext i32 %241 to i64, !dbg !16
  %243 = inttoptr i64 %242 to ptr addrspace(3), !dbg !16
  %address.local613 = lshr i64 %address.local607, 4, !dbg !16
  %address.local614 = and i64 %address.local613, 24, !dbg !16
  %address.local615 = and i64 %address.local590, 123, !dbg !16
  %address.local616 = lshr i64 %address.local590, 2, !dbg !16
  %address.local617 = and i64 %address.local616, 4, !dbg !16
  %address.local618 = add nuw nsw i64 %address.local617, %address.local590, !dbg !16
  %address.local619 = and i64 %address.local618, 4, !dbg !16
  %address.local620 = or disjoint i64 %address.local615, %address.local619, !dbg !16
  %address.local621 = shl nuw nsw i64 %address.local620, 7, !dbg !16
  %address.local622 = or disjoint i64 %address.local614, %address.local621, !dbg !16
  %address.local623 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %address.local622, !dbg !16
  %244 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %245 = ptrtoint ptr addrspace(1) %address.local623 to i64, !dbg !16
  %246 = sub i64 %245, %244, !dbg !16
  %dma.buffer.offset1741 = trunc i64 %246 to i32, !dbg !16
  %247 = trunc i64 %244 to i32, !dbg !16
  %248 = insertelement <4 x i32> poison, i32 %247, i64 0, !dbg !16
  %249 = lshr i64 %244, 32, !dbg !16
  %250 = trunc i64 %249 to i32, !dbg !16
  %251 = insertelement <4 x i32> %248, i32 %250, i64 1, !dbg !16
  %252 = insertelement <4 x i32> %251, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1742 = insertelement <4 x i32> %252, i32 131072, i64 3, !dbg !16
  %253 = ptrtoint ptr addrspace(3) %243 to i32, !dbg !16
  %dma.buffer.lds1743 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %253), !dbg !16
  %254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1743, i32 %dma.buffer.offset1741, <4 x i32> %dma.buffer.rsrc1742) #10, !dbg !16
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
  %address.local624 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local592, !dbg !8
  %address.local625 = ptrtoint ptr addrspace(3) %address.local624 to i64, !dbg !8
  %address.local626 = trunc i64 %address.local625 to i32, !dbg !8
  %279 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local626), !dbg !16
  %280 = zext i32 %279 to i64, !dbg !16
  %281 = inttoptr i64 %280 to ptr addrspace(3), !dbg !16
  %address.local627 = getelementptr [2 x i8], ptr addrspace(1) %278, i64 %address.local605, !dbg !16
  %282 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %283 = ptrtoint ptr addrspace(1) %address.local627 to i64, !dbg !16
  %284 = sub i64 %283, %282, !dbg !16
  %dma.buffer.offset1738 = trunc i64 %284 to i32, !dbg !16
  %285 = trunc i64 %282 to i32, !dbg !16
  %286 = insertelement <4 x i32> poison, i32 %285, i64 0, !dbg !16
  %287 = lshr i64 %282, 32, !dbg !16
  %288 = trunc i64 %287 to i32, !dbg !16
  %289 = insertelement <4 x i32> %286, i32 %288, i64 1, !dbg !16
  %290 = insertelement <4 x i32> %289, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1739 = insertelement <4 x i32> %290, i32 131072, i64 3, !dbg !16
  %291 = ptrtoint ptr addrspace(3) %281 to i32, !dbg !16
  %dma.buffer.lds1740 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %291), !dbg !16
  %292 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1740, i32 %dma.buffer.offset1738, <4 x i32> %dma.buffer.rsrc1739) #10, !dbg !16
  %address.local628 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local609, !dbg !8
  %address.local629 = ptrtoint ptr addrspace(3) %address.local628 to i64, !dbg !8
  %address.local630 = trunc i64 %address.local629 to i32, !dbg !8
  %293 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local630), !dbg !16
  %294 = zext i32 %293 to i64, !dbg !16
  %295 = inttoptr i64 %294 to ptr addrspace(3), !dbg !16
  %address.local631 = getelementptr [2 x i8], ptr addrspace(1) %278, i64 %address.local622, !dbg !16
  %296 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %297 = ptrtoint ptr addrspace(1) %address.local631 to i64, !dbg !16
  %298 = sub i64 %297, %296, !dbg !16
  %dma.buffer.offset1735 = trunc i64 %298 to i32, !dbg !16
  %299 = trunc i64 %296 to i32, !dbg !16
  %300 = insertelement <4 x i32> poison, i32 %299, i64 0, !dbg !16
  %301 = lshr i64 %296, 32, !dbg !16
  %302 = trunc i64 %301 to i32, !dbg !16
  %303 = insertelement <4 x i32> %300, i32 %302, i64 1, !dbg !16
  %304 = insertelement <4 x i32> %303, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1736 = insertelement <4 x i32> %304, i32 131072, i64 3, !dbg !16
  %305 = ptrtoint ptr addrspace(3) %295 to i32, !dbg !16
  %dma.buffer.lds1737 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %305), !dbg !16
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1737, i32 %dma.buffer.offset1735, <4 x i32> %dma.buffer.rsrc1736) #10, !dbg !16
  %address.local632 = mul nuw nsw i64 %address.local590, 160, !dbg !17
  %address.local633 = lshr i64 %address.local590, 2, !dbg !17
  %address.local634 = mul nsw i64 %address.local633, -640, !dbg !17
  %address.local635 = add nsw i64 %address.local632, %address.local634, !dbg !17
  %address.local636 = shl nuw nsw i64 %address.local590, 1, !dbg !17
  %address.local637 = and i64 %address.local636, 24, !dbg !17
  %address.local638 = or disjoint i64 %address.local635, %address.local637, !dbg !17
  %address.local639 = shl i64 %address.local590, 62, !dbg !17
  %address.local640 = ashr exact i64 %address.local639, 57, !dbg !17
  %address.local641 = and i64 %address.local640, -64, !dbg !17
  %address.local642 = add nsw i64 %address.local638, %address.local641, !dbg !17
  %address.local643 = lshr i64 %address.local590, 2, !dbg !11
  %address.local644 = and i64 %address.local643, 12, !dbg !11
  %address.local645 = add nsw i64 %address.local642, %address.local644, !dbg !17
  %address.local646 = and i64 %address.local590, 63, !dbg !11
  %address.local647 = lshr i64 %address.local646, 5, !dbg !11
  %address.local648 = mul nuw nsw i64 %address.local647, 1016, !dbg !11
  %address.local649 = add i64 %address.local645, %address.local648, !dbg !17
  %address.local650 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local649, !dbg !17
  %307 = load <4 x half>, ptr addrspace(3) %address.local650, align 8, !dbg !17
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %307, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1582 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4096, !dbg !17
  %309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1582, align 8, !dbg !17
  %address.local661 = shl nuw nsw i32 %address.thread589, 1, !dbg !17
  %address.local662 = and i32 %address.local661, 24, !dbg !17
  %address.local663 = shl nuw nsw i32 %address.thread589, 8, !dbg !17
  %address.local664 = and i32 %address.local663, 768, !dbg !17
  %address.local665 = or disjoint i32 %address.local662, %address.local664, !dbg !17
  %address.local666 = shl nuw nsw i32 %address.thread589, 5, !dbg !17
  %address.local667 = and i32 %address.local666, 96, !dbg !17
  %address.local668 = or disjoint i32 %address.local665, %address.local667, !dbg !17
  %address.local669 = zext nneg i32 %address.local668 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %203) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local670 = lshr i32 %address.thread589, 1, !dbg !17
  %address.local671 = and i32 %address.local670, 1, !dbg !17
  %address.local672 = zext nneg i32 %address.local671 to i64, !dbg !17
  %address.local673 = add nsw i64 %address.base, %address.local672, !dbg !17
  %address.local674 = shl i64 %address.local673, 6, !dbg !17
  %address.local675 = sub i64 %address.local669, %address.local674, !dbg !17
  %address.local676 = add i64 %address.local675, %address.local644, !dbg !17
  %address.local677 = add i64 %address.local676, %address.local648, !dbg !17
  %address.local678 = add i64 %address.local677, 32, !dbg !17
  %address.local679 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local678, !dbg !17
  %310 = load <4 x half>, ptr addrspace(3) %address.local679, align 8, !dbg !17
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %309, <4 x float> %308) #9, !dbg !17
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %310, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4096, !dbg !17
  %313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !17
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local650, i64 128, !dbg !17
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !17
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %313, <4 x float> %312) #9, !dbg !17
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %314, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1585 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4224, !dbg !17
  %317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1585, align 8, !dbg !17
  %lds.const.gep1586 = getelementptr i8, ptr addrspace(3) %address.local679, i64 128, !dbg !17
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1586, align 8, !dbg !17
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %317, <4 x float> %316) #9, !dbg !17
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %318, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1587 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4224, !dbg !17
  %321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1587, align 8, !dbg !17
  %lds.const.gep1588 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1024, !dbg !17
  %322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1588, align 8, !dbg !17
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %321, <4 x float> %320) #9, !dbg !17
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %322, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1589 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5120, !dbg !17
  %325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !17
  %lds.const.gep1590 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1024, !dbg !17
  %326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !17
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %325, <4 x float> %324) #9, !dbg !17
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %326, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1591 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5120, !dbg !17
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !17
  %lds.const.gep1592 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1152, !dbg !17
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !17
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %329, <4 x float> %328) #9, !dbg !17
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %330, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5248, !dbg !17
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !17
  %lds.const.gep1594 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1152, !dbg !17
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !17
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %333, <4 x float> %332) #9, !dbg !17
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %334, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1595 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5248, !dbg !17
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !17
  %338 = load <4 x half>, ptr addrspace(3) %address.local650, align 8, !dbg !17
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %337, <4 x float> %336) #9, !dbg !17
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %338, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1596 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4096, !dbg !17
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !17
  %342 = load <4 x half>, ptr addrspace(3) %address.local679, align 8, !dbg !17
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %341, <4 x float> %340) #9, !dbg !17
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %342, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1597 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4096, !dbg !17
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !17
  %lds.const.gep1598 = getelementptr i8, ptr addrspace(3) %address.local650, i64 128, !dbg !17
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !17
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %345, <4 x float> %344) #9, !dbg !17
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %346, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1599 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4224, !dbg !17
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !17
  %lds.const.gep1600 = getelementptr i8, ptr addrspace(3) %address.local679, i64 128, !dbg !17
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !17
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %349, <4 x float> %348) #9, !dbg !17
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %350, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1601 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4224, !dbg !17
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !17
  %lds.const.gep1602 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1024, !dbg !17
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !17
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %353, <4 x float> %352) #9, !dbg !17
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %354, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1603 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5120, !dbg !17
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !17
  %lds.const.gep1604 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1024, !dbg !17
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !17
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %357, <4 x float> %356) #9, !dbg !17
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %358, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1605 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5120, !dbg !17
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !17
  %lds.const.gep1606 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1152, !dbg !17
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1606, align 8, !dbg !17
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %361, <4 x float> %360) #9, !dbg !17
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %362, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1607 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5248, !dbg !17
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1607, align 8, !dbg !17
  %lds.const.gep1608 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1152, !dbg !17
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1608, align 8, !dbg !17
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %365, <4 x float> %364) #9, !dbg !17
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %366, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1609 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5248, !dbg !17
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1609, align 8, !dbg !17
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
  %394 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local595), !dbg !16
  %395 = zext i32 %394 to i64, !dbg !16
  %396 = inttoptr i64 %395 to ptr addrspace(3), !dbg !16
  %address.local988 = getelementptr [2 x i8], ptr addrspace(1) %393, i64 %address.local605, !dbg !16
  %397 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %398 = ptrtoint ptr addrspace(1) %address.local988 to i64, !dbg !16
  %399 = sub i64 %398, %397, !dbg !16
  %dma.buffer.offset1732 = trunc i64 %399 to i32, !dbg !16
  %400 = trunc i64 %397 to i32, !dbg !16
  %401 = insertelement <4 x i32> poison, i32 %400, i64 0, !dbg !16
  %402 = lshr i64 %397, 32, !dbg !16
  %403 = trunc i64 %402 to i32, !dbg !16
  %404 = insertelement <4 x i32> %401, i32 %403, i64 1, !dbg !16
  %405 = insertelement <4 x i32> %404, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1733 = insertelement <4 x i32> %405, i32 131072, i64 3, !dbg !16
  %406 = ptrtoint ptr addrspace(3) %396 to i32, !dbg !16
  %dma.buffer.lds1734 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %406), !dbg !16
  %407 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1734, i32 %dma.buffer.offset1732, <4 x i32> %dma.buffer.rsrc1733) #10, !dbg !16
  %408 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local612), !dbg !16
  %409 = zext i32 %408 to i64, !dbg !16
  %410 = inttoptr i64 %409 to ptr addrspace(3), !dbg !16
  %address.local989 = getelementptr [2 x i8], ptr addrspace(1) %393, i64 %address.local622, !dbg !16
  %411 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %412 = ptrtoint ptr addrspace(1) %address.local989 to i64, !dbg !16
  %413 = sub i64 %412, %411, !dbg !16
  %dma.buffer.offset1729 = trunc i64 %413 to i32, !dbg !16
  %414 = trunc i64 %411 to i32, !dbg !16
  %415 = insertelement <4 x i32> poison, i32 %414, i64 0, !dbg !16
  %416 = lshr i64 %411, 32, !dbg !16
  %417 = trunc i64 %416 to i32, !dbg !16
  %418 = insertelement <4 x i32> %415, i32 %417, i64 1, !dbg !16
  %419 = insertelement <4 x i32> %418, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1730 = insertelement <4 x i32> %419, i32 131072, i64 3, !dbg !16
  %420 = ptrtoint ptr addrspace(3) %410 to i32, !dbg !16
  %dma.buffer.lds1731 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %420), !dbg !16
  %421 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1731, i32 %dma.buffer.offset1729, <4 x i32> %dma.buffer.rsrc1730) #10, !dbg !16
  %lds.const.gep1610 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8192, !dbg !17
  %422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1610, align 8, !dbg !17
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %422, <4 x float> %311) #9, !dbg !17
  %lds.const.gep1611 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12288, !dbg !17
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1611, align 8, !dbg !17
  %lds.const.gep1612 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8192, !dbg !17
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1612, align 8, !dbg !17
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %424, <4 x float> %423) #9, !dbg !17
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %425, <4 x float> %315) #9, !dbg !17
  %lds.const.gep1613 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12288, !dbg !17
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1613, align 8, !dbg !17
  %lds.const.gep1614 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8320, !dbg !17
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1614, align 8, !dbg !17
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %428, <4 x float> %427) #9, !dbg !17
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %429, <4 x float> %319) #9, !dbg !17
  %lds.const.gep1615 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12416, !dbg !17
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1615, align 8, !dbg !17
  %lds.const.gep1616 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8320, !dbg !17
  %433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1616, align 8, !dbg !17
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %432, <4 x float> %431) #9, !dbg !17
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %433, <4 x float> %323) #9, !dbg !17
  %lds.const.gep1617 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12416, !dbg !17
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1617, align 8, !dbg !17
  %lds.const.gep1618 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9216, !dbg !17
  %437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1618, align 8, !dbg !17
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %436, <4 x float> %435) #9, !dbg !17
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %437, <4 x float> %327) #9, !dbg !17
  %lds.const.gep1619 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13312, !dbg !17
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1619, align 8, !dbg !17
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9216, !dbg !17
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !17
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %440, <4 x float> %439) #9, !dbg !17
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %441, <4 x float> %331) #9, !dbg !17
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13312, !dbg !17
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !17
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9344, !dbg !17
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !17
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %444, <4 x float> %443) #9, !dbg !17
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %445, <4 x float> %335) #9, !dbg !17
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13440, !dbg !17
  %448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !17
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9344, !dbg !17
  %449 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !17
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %448, <4 x float> %447) #9, !dbg !17
  %451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %449, <4 x float> %339) #9, !dbg !17
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13440, !dbg !17
  %452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !17
  %lds.const.gep1626 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8192, !dbg !17
  %453 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !17
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %452, <4 x float> %451) #9, !dbg !17
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %453, <4 x float> %343) #9, !dbg !17
  %lds.const.gep1627 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12288, !dbg !17
  %456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !17
  %lds.const.gep1628 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8192, !dbg !17
  %457 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !17
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %456, <4 x float> %455) #9, !dbg !17
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %457, <4 x float> %347) #9, !dbg !17
  %lds.const.gep1629 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12288, !dbg !17
  %460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !17
  %lds.const.gep1630 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8320, !dbg !17
  %461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !17
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %460, <4 x float> %459) #9, !dbg !17
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %461, <4 x float> %351) #9, !dbg !17
  %lds.const.gep1631 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12416, !dbg !17
  %464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !17
  %lds.const.gep1632 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8320, !dbg !17
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !17
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %464, <4 x float> %463) #9, !dbg !17
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %465, <4 x float> %355) #9, !dbg !17
  %lds.const.gep1633 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12416, !dbg !17
  %468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !17
  %lds.const.gep1634 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9216, !dbg !17
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !17
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %468, <4 x float> %467) #9, !dbg !17
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %469, <4 x float> %359) #9, !dbg !17
  %lds.const.gep1635 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13312, !dbg !17
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !17
  %lds.const.gep1636 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9216, !dbg !17
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1636, align 8, !dbg !17
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %472, <4 x float> %471) #9, !dbg !17
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %473, <4 x float> %363) #9, !dbg !17
  %lds.const.gep1637 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13312, !dbg !17
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1637, align 8, !dbg !17
  %lds.const.gep1638 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9344, !dbg !17
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1638, align 8, !dbg !17
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %476, <4 x float> %475) #9, !dbg !17
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %477, <4 x float> %367) #9, !dbg !17
  %lds.const.gep1639 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13440, !dbg !17
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1639, align 8, !dbg !17
  %lds.const.gep1640 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9344, !dbg !17
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1640, align 8, !dbg !17
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %480, <4 x float> %479) #9, !dbg !17
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %481, <4 x float> %370) #9, !dbg !17
  %lds.const.gep1641 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13440, !dbg !17
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1641, align 8, !dbg !17
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
  %509 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local626), !dbg !16
  %510 = zext i32 %509 to i64, !dbg !16
  %511 = inttoptr i64 %510 to ptr addrspace(3), !dbg !16
  %address.local1006 = getelementptr [2 x i8], ptr addrspace(1) %508, i64 %address.local605, !dbg !16
  %512 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %513 = ptrtoint ptr addrspace(1) %address.local1006 to i64, !dbg !16
  %514 = sub i64 %513, %512, !dbg !16
  %dma.buffer.offset1726 = trunc i64 %514 to i32, !dbg !16
  %515 = trunc i64 %512 to i32, !dbg !16
  %516 = insertelement <4 x i32> poison, i32 %515, i64 0, !dbg !16
  %517 = lshr i64 %512, 32, !dbg !16
  %518 = trunc i64 %517 to i32, !dbg !16
  %519 = insertelement <4 x i32> %516, i32 %518, i64 1, !dbg !16
  %520 = insertelement <4 x i32> %519, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1727 = insertelement <4 x i32> %520, i32 131072, i64 3, !dbg !16
  %521 = ptrtoint ptr addrspace(3) %511 to i32, !dbg !16
  %dma.buffer.lds1728 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %521), !dbg !16
  %522 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1728, i32 %dma.buffer.offset1726, <4 x i32> %dma.buffer.rsrc1727) #10, !dbg !16
  %523 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local630), !dbg !16
  %524 = zext i32 %523 to i64, !dbg !16
  %525 = inttoptr i64 %524 to ptr addrspace(3), !dbg !16
  %address.local1007 = getelementptr [2 x i8], ptr addrspace(1) %508, i64 %address.local622, !dbg !16
  %526 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %527 = ptrtoint ptr addrspace(1) %address.local1007 to i64, !dbg !16
  %528 = sub i64 %527, %526, !dbg !16
  %dma.buffer.offset1723 = trunc i64 %528 to i32, !dbg !16
  %529 = trunc i64 %526 to i32, !dbg !16
  %530 = insertelement <4 x i32> poison, i32 %529, i64 0, !dbg !16
  %531 = lshr i64 %526, 32, !dbg !16
  %532 = trunc i64 %531 to i32, !dbg !16
  %533 = insertelement <4 x i32> %530, i32 %532, i64 1, !dbg !16
  %534 = insertelement <4 x i32> %533, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1724 = insertelement <4 x i32> %534, i32 131072, i64 3, !dbg !16
  %535 = ptrtoint ptr addrspace(3) %525 to i32, !dbg !16
  %dma.buffer.lds1725 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %535), !dbg !16
  %536 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1725, i32 %dma.buffer.offset1723, <4 x i32> %dma.buffer.rsrc1724) #10, !dbg !16
  %537 = load <4 x half>, ptr addrspace(3) %address.local650, align 8, !dbg !17
  %538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %537, <4 x float> %426) #9, !dbg !17
  %lds.const.gep1642 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4096, !dbg !17
  %539 = load <4 x half>, ptr addrspace(3) %lds.const.gep1642, align 8, !dbg !17
  %540 = load <4 x half>, ptr addrspace(3) %address.local679, align 8, !dbg !17
  %541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %539, <4 x float> %538) #9, !dbg !17
  %542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %540, <4 x float> %430) #9, !dbg !17
  %lds.const.gep1643 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4096, !dbg !17
  %543 = load <4 x half>, ptr addrspace(3) %lds.const.gep1643, align 8, !dbg !17
  %lds.const.gep1644 = getelementptr i8, ptr addrspace(3) %address.local650, i64 128, !dbg !17
  %544 = load <4 x half>, ptr addrspace(3) %lds.const.gep1644, align 8, !dbg !17
  %545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %543, <4 x float> %542) #9, !dbg !17
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %544, <4 x float> %434) #9, !dbg !17
  %lds.const.gep1645 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4224, !dbg !17
  %547 = load <4 x half>, ptr addrspace(3) %lds.const.gep1645, align 8, !dbg !17
  %lds.const.gep1646 = getelementptr i8, ptr addrspace(3) %address.local679, i64 128, !dbg !17
  %548 = load <4 x half>, ptr addrspace(3) %lds.const.gep1646, align 8, !dbg !17
  %549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %547, <4 x float> %546) #9, !dbg !17
  %550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %548, <4 x float> %438) #9, !dbg !17
  %lds.const.gep1647 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4224, !dbg !17
  %551 = load <4 x half>, ptr addrspace(3) %lds.const.gep1647, align 8, !dbg !17
  %lds.const.gep1648 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1024, !dbg !17
  %552 = load <4 x half>, ptr addrspace(3) %lds.const.gep1648, align 8, !dbg !17
  %553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %551, <4 x float> %550) #9, !dbg !17
  %554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %552, <4 x float> %442) #9, !dbg !17
  %lds.const.gep1649 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5120, !dbg !17
  %555 = load <4 x half>, ptr addrspace(3) %lds.const.gep1649, align 8, !dbg !17
  %lds.const.gep1650 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1024, !dbg !17
  %556 = load <4 x half>, ptr addrspace(3) %lds.const.gep1650, align 8, !dbg !17
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %555, <4 x float> %554) #9, !dbg !17
  %558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %556, <4 x float> %446) #9, !dbg !17
  %lds.const.gep1651 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5120, !dbg !17
  %559 = load <4 x half>, ptr addrspace(3) %lds.const.gep1651, align 8, !dbg !17
  %lds.const.gep1652 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1152, !dbg !17
  %560 = load <4 x half>, ptr addrspace(3) %lds.const.gep1652, align 8, !dbg !17
  %561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %559, <4 x float> %558) #9, !dbg !17
  %562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %560, <4 x float> %450) #9, !dbg !17
  %lds.const.gep1653 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5248, !dbg !17
  %563 = load <4 x half>, ptr addrspace(3) %lds.const.gep1653, align 8, !dbg !17
  %lds.const.gep1654 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1152, !dbg !17
  %564 = load <4 x half>, ptr addrspace(3) %lds.const.gep1654, align 8, !dbg !17
  %565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %563, <4 x float> %562) #9, !dbg !17
  %566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %564, <4 x float> %454) #9, !dbg !17
  %lds.const.gep1655 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5248, !dbg !17
  %567 = load <4 x half>, ptr addrspace(3) %lds.const.gep1655, align 8, !dbg !17
  %568 = load <4 x half>, ptr addrspace(3) %address.local650, align 8, !dbg !17
  %569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %567, <4 x float> %566) #9, !dbg !17
  %570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %568, <4 x float> %458) #9, !dbg !17
  %lds.const.gep1656 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4096, !dbg !17
  %571 = load <4 x half>, ptr addrspace(3) %lds.const.gep1656, align 8, !dbg !17
  %572 = load <4 x half>, ptr addrspace(3) %address.local679, align 8, !dbg !17
  %573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %571, <4 x float> %570) #9, !dbg !17
  %574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %572, <4 x float> %462) #9, !dbg !17
  %lds.const.gep1657 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4096, !dbg !17
  %575 = load <4 x half>, ptr addrspace(3) %lds.const.gep1657, align 8, !dbg !17
  %lds.const.gep1658 = getelementptr i8, ptr addrspace(3) %address.local650, i64 128, !dbg !17
  %576 = load <4 x half>, ptr addrspace(3) %lds.const.gep1658, align 8, !dbg !17
  %577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %575, <4 x float> %574) #9, !dbg !17
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %576, <4 x float> %466) #9, !dbg !17
  %lds.const.gep1659 = getelementptr i8, ptr addrspace(3) %address.local650, i64 4224, !dbg !17
  %579 = load <4 x half>, ptr addrspace(3) %lds.const.gep1659, align 8, !dbg !17
  %lds.const.gep1660 = getelementptr i8, ptr addrspace(3) %address.local679, i64 128, !dbg !17
  %580 = load <4 x half>, ptr addrspace(3) %lds.const.gep1660, align 8, !dbg !17
  %581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %579, <4 x float> %578) #9, !dbg !17
  %582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %580, <4 x float> %470) #9, !dbg !17
  %lds.const.gep1661 = getelementptr i8, ptr addrspace(3) %address.local679, i64 4224, !dbg !17
  %583 = load <4 x half>, ptr addrspace(3) %lds.const.gep1661, align 8, !dbg !17
  %lds.const.gep1662 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1024, !dbg !17
  %584 = load <4 x half>, ptr addrspace(3) %lds.const.gep1662, align 8, !dbg !17
  %585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %583, <4 x float> %582) #9, !dbg !17
  %586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %584, <4 x float> %474) #9, !dbg !17
  %lds.const.gep1663 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5120, !dbg !17
  %587 = load <4 x half>, ptr addrspace(3) %lds.const.gep1663, align 8, !dbg !17
  %lds.const.gep1664 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1024, !dbg !17
  %588 = load <4 x half>, ptr addrspace(3) %lds.const.gep1664, align 8, !dbg !17
  %589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %587, <4 x float> %586) #9, !dbg !17
  %590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %588, <4 x float> %478) #9, !dbg !17
  %lds.const.gep1665 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5120, !dbg !17
  %591 = load <4 x half>, ptr addrspace(3) %lds.const.gep1665, align 8, !dbg !17
  %lds.const.gep1666 = getelementptr i8, ptr addrspace(3) %address.local650, i64 1152, !dbg !17
  %592 = load <4 x half>, ptr addrspace(3) %lds.const.gep1666, align 8, !dbg !17
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %591, <4 x float> %590) #9, !dbg !17
  %594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %592, <4 x float> %482) #9, !dbg !17
  %lds.const.gep1667 = getelementptr i8, ptr addrspace(3) %address.local650, i64 5248, !dbg !17
  %595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1667, align 8, !dbg !17
  %lds.const.gep1668 = getelementptr i8, ptr addrspace(3) %address.local679, i64 1152, !dbg !17
  %596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1668, align 8, !dbg !17
  %597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %595, <4 x float> %594) #9, !dbg !17
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %596, <4 x float> %485) #9, !dbg !17
  %lds.const.gep1669 = getelementptr i8, ptr addrspace(3) %address.local679, i64 5248, !dbg !17
  %599 = load <4 x half>, ptr addrspace(3) %lds.const.gep1669, align 8, !dbg !17
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
  %606 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local595), !dbg !18
  %607 = zext i32 %606 to i64, !dbg !18
  %608 = inttoptr i64 %607 to ptr addrspace(3), !dbg !18
  %address.local1008 = shl nuw nsw i64 %address.local590, 5, !dbg !18
  %address.local1009 = and i64 %address.local1008, 8064, !dbg !18
  %address.local1010 = shl nuw nsw i64 %address.local590, 3, !dbg !18
  %address.local1011 = and i64 %address.local1010, 24, !dbg !18
  %address.local1012 = or disjoint i64 %address.local1009, %address.local1011, !dbg !18
  %address.local1013 = getelementptr [2 x i8], ptr addrspace(1) %605, i64 %address.local1012, !dbg !18
  %609 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %610 = ptrtoint ptr addrspace(1) %address.local1013 to i64, !dbg !18
  %611 = sub i64 %610, %609, !dbg !18
  %dma.buffer.offset1720 = trunc i64 %611 to i32, !dbg !18
  %612 = trunc i64 %609 to i32, !dbg !18
  %613 = insertelement <4 x i32> poison, i32 %612, i64 0, !dbg !18
  %614 = lshr i64 %609, 32, !dbg !18
  %615 = trunc i64 %614 to i32, !dbg !18
  %616 = insertelement <4 x i32> %613, i32 %615, i64 1, !dbg !18
  %617 = insertelement <4 x i32> %616, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1721 = insertelement <4 x i32> %617, i32 131072, i64 3, !dbg !18
  %618 = ptrtoint ptr addrspace(3) %608 to i32, !dbg !18
  %dma.buffer.lds1722 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %618), !dbg !18
  %619 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1722, i32 %dma.buffer.offset1720, <4 x i32> %dma.buffer.rsrc1721) #10, !dbg !18
  %620 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local612), !dbg !18
  %621 = zext i32 %620 to i64, !dbg !18
  %622 = inttoptr i64 %621 to ptr addrspace(3), !dbg !18
  %address.local1014 = shl nuw nsw i64 %address.local607, 5, !dbg !18
  %address.local1015 = and i64 %address.local1014, 16256, !dbg !18
  %address.local1016 = shl nuw nsw i64 %address.local590, 3, !dbg !18
  %address.local1017 = and i64 %address.local1016, 24, !dbg !18
  %address.local1018 = or disjoint i64 %address.local1015, %address.local1017, !dbg !18
  %address.local1019 = getelementptr [2 x i8], ptr addrspace(1) %605, i64 %address.local1018, !dbg !18
  %623 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %624 = ptrtoint ptr addrspace(1) %address.local1019 to i64, !dbg !18
  %625 = sub i64 %624, %623, !dbg !18
  %dma.buffer.offset1717 = trunc i64 %625 to i32, !dbg !18
  %626 = trunc i64 %623 to i32, !dbg !18
  %627 = insertelement <4 x i32> poison, i32 %626, i64 0, !dbg !18
  %628 = lshr i64 %623, 32, !dbg !18
  %629 = trunc i64 %628 to i32, !dbg !18
  %630 = insertelement <4 x i32> %627, i32 %629, i64 1, !dbg !18
  %631 = insertelement <4 x i32> %630, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1718 = insertelement <4 x i32> %631, i32 131072, i64 3, !dbg !18
  %632 = ptrtoint ptr addrspace(3) %622 to i32, !dbg !18
  %dma.buffer.lds1719 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %632), !dbg !18
  %633 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1719, i32 %dma.buffer.offset1717, <4 x i32> %dma.buffer.rsrc1718) #10, !dbg !18
  %lds.const.gep1670 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8192, !dbg !17
  %634 = load <4 x half>, ptr addrspace(3) %lds.const.gep1670, align 8, !dbg !17
  %635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %634, <4 x float> %541) #9, !dbg !17
  %lds.const.gep1671 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12288, !dbg !17
  %636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1671, align 8, !dbg !17
  %lds.const.gep1672 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8192, !dbg !17
  %637 = load <4 x half>, ptr addrspace(3) %lds.const.gep1672, align 8, !dbg !17
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %636, <4 x float> %635) #9, !dbg !17
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %637, <4 x float> %545) #9, !dbg !17
  %lds.const.gep1673 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12288, !dbg !17
  %640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1673, align 8, !dbg !17
  %lds.const.gep1674 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8320, !dbg !17
  %641 = load <4 x half>, ptr addrspace(3) %lds.const.gep1674, align 8, !dbg !17
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %640, <4 x float> %639) #9, !dbg !17
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %641, <4 x float> %549) #9, !dbg !17
  %lds.const.gep1675 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12416, !dbg !17
  %644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1675, align 8, !dbg !17
  %lds.const.gep1676 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8320, !dbg !17
  %645 = load <4 x half>, ptr addrspace(3) %lds.const.gep1676, align 8, !dbg !17
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %644, <4 x float> %643) #9, !dbg !17
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %645, <4 x float> %553) #9, !dbg !17
  %lds.const.gep1677 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12416, !dbg !17
  %648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1677, align 8, !dbg !17
  %lds.const.gep1678 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9216, !dbg !17
  %649 = load <4 x half>, ptr addrspace(3) %lds.const.gep1678, align 8, !dbg !17
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %648, <4 x float> %647) #9, !dbg !17
  %651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %649, <4 x float> %557) #9, !dbg !17
  %lds.const.gep1679 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13312, !dbg !17
  %652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1679, align 8, !dbg !17
  %lds.const.gep1680 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9216, !dbg !17
  %653 = load <4 x half>, ptr addrspace(3) %lds.const.gep1680, align 8, !dbg !17
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %652, <4 x float> %651) #9, !dbg !17
  %655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %653, <4 x float> %561) #9, !dbg !17
  %lds.const.gep1681 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13312, !dbg !17
  %656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1681, align 8, !dbg !17
  %lds.const.gep1682 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9344, !dbg !17
  %657 = load <4 x half>, ptr addrspace(3) %lds.const.gep1682, align 8, !dbg !17
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %656, <4 x float> %655) #9, !dbg !17
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %657, <4 x float> %565) #9, !dbg !17
  %lds.const.gep1683 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13440, !dbg !17
  %660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1683, align 8, !dbg !17
  %lds.const.gep1684 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9344, !dbg !17
  %661 = load <4 x half>, ptr addrspace(3) %lds.const.gep1684, align 8, !dbg !17
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %660, <4 x float> %659) #9, !dbg !17
  %663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %661, <4 x float> %569) #9, !dbg !17
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13440, !dbg !17
  %664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1685, align 8, !dbg !17
  %lds.const.gep1686 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8192, !dbg !17
  %665 = load <4 x half>, ptr addrspace(3) %lds.const.gep1686, align 8, !dbg !17
  %666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %664, <4 x float> %663) #9, !dbg !17
  %667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %665, <4 x float> %573) #9, !dbg !17
  %lds.const.gep1687 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12288, !dbg !17
  %668 = load <4 x half>, ptr addrspace(3) %lds.const.gep1687, align 8, !dbg !17
  %lds.const.gep1688 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8192, !dbg !17
  %669 = load <4 x half>, ptr addrspace(3) %lds.const.gep1688, align 8, !dbg !17
  %670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %668, <4 x float> %667) #9, !dbg !17
  %671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %669, <4 x float> %577) #9, !dbg !17
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12288, !dbg !17
  %672 = load <4 x half>, ptr addrspace(3) %lds.const.gep1689, align 8, !dbg !17
  %lds.const.gep1690 = getelementptr i8, ptr addrspace(3) %address.local650, i64 8320, !dbg !17
  %673 = load <4 x half>, ptr addrspace(3) %lds.const.gep1690, align 8, !dbg !17
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %672, <4 x float> %671) #9, !dbg !17
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %673, <4 x float> %581) #9, !dbg !17
  %lds.const.gep1691 = getelementptr i8, ptr addrspace(3) %address.local650, i64 12416, !dbg !17
  %676 = load <4 x half>, ptr addrspace(3) %lds.const.gep1691, align 8, !dbg !17
  %lds.const.gep1692 = getelementptr i8, ptr addrspace(3) %address.local679, i64 8320, !dbg !17
  %677 = load <4 x half>, ptr addrspace(3) %lds.const.gep1692, align 8, !dbg !17
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %676, <4 x float> %675) #9, !dbg !17
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %677, <4 x float> %585) #9, !dbg !17
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local679, i64 12416, !dbg !17
  %680 = load <4 x half>, ptr addrspace(3) %lds.const.gep1693, align 8, !dbg !17
  %lds.const.gep1694 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9216, !dbg !17
  %681 = load <4 x half>, ptr addrspace(3) %lds.const.gep1694, align 8, !dbg !17
  %682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %680, <4 x float> %679) #9, !dbg !17
  %683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %681, <4 x float> %589) #9, !dbg !17
  %lds.const.gep1695 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13312, !dbg !17
  %684 = load <4 x half>, ptr addrspace(3) %lds.const.gep1695, align 8, !dbg !17
  %lds.const.gep1696 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9216, !dbg !17
  %685 = load <4 x half>, ptr addrspace(3) %lds.const.gep1696, align 8, !dbg !17
  %686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %684, <4 x float> %683) #9, !dbg !17
  %687 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %685, <4 x float> %593) #9, !dbg !17
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13312, !dbg !17
  %688 = load <4 x half>, ptr addrspace(3) %lds.const.gep1697, align 8, !dbg !17
  %lds.const.gep1698 = getelementptr i8, ptr addrspace(3) %address.local650, i64 9344, !dbg !17
  %689 = load <4 x half>, ptr addrspace(3) %lds.const.gep1698, align 8, !dbg !17
  %690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %688, <4 x float> %687) #9, !dbg !17
  %691 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %689, <4 x float> %597) #9, !dbg !17
  %lds.const.gep1699 = getelementptr i8, ptr addrspace(3) %address.local650, i64 13440, !dbg !17
  %692 = load <4 x half>, ptr addrspace(3) %lds.const.gep1699, align 8, !dbg !17
  %lds.const.gep1700 = getelementptr i8, ptr addrspace(3) %address.local679, i64 9344, !dbg !17
  %693 = load <4 x half>, ptr addrspace(3) %lds.const.gep1700, align 8, !dbg !17
  %694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %692, <4 x float> %691) #9, !dbg !17
  %695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %693, <4 x float> %600) #9, !dbg !17
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local679, i64 13440, !dbg !17
  %696 = load <4 x half>, ptr addrspace(3) %lds.const.gep1701, align 8, !dbg !17
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
  %compare.high1313 = trunc i64 %712 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1313, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1313, !dbg !24
  %compare.low.base = trunc i64 %710 to i32, !dbg !24
  %compare.low.base1314 = trunc i64 %709 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1314, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %713 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low1315 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result1316 = select i1 %compare.high.equal, i1 %compare.low1315, i1 %compare.high.order, !dbg !24
  %714 = select i1 %compare.result1316, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1317 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1318 = or i32 %compare.local.base1317, 2, !dbg !24
  %compare.low1319 = icmp ugt i32 %compare.low.base, %compare.local1318, !dbg !24
  %compare.result1320 = select i1 %compare.high.equal, i1 %compare.low1319, i1 %compare.high.order, !dbg !24
  %715 = select i1 %compare.result1320, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1321 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1322 = or i32 %compare.local.base1321, 3, !dbg !24
  %compare.low1323 = icmp ugt i32 %compare.low.base, %compare.local1322, !dbg !24
  %compare.result1324 = select i1 %compare.high.equal, i1 %compare.low1323, i1 %compare.high.order, !dbg !24
  %716 = select i1 %compare.result1324, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1325 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1326 = or i32 %compare.local.base1325, 4, !dbg !24
  %compare.low1327 = icmp ugt i32 %compare.low.base, %compare.local1326, !dbg !24
  %compare.result1328 = select i1 %compare.high.equal, i1 %compare.low1327, i1 %compare.high.order, !dbg !24
  %717 = select i1 %compare.result1328, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1330 = or i32 %compare.local.base1329, 5, !dbg !24
  %compare.low1331 = icmp ugt i32 %compare.low.base, %compare.local1330, !dbg !24
  %compare.result1332 = select i1 %compare.high.equal, i1 %compare.low1331, i1 %compare.high.order, !dbg !24
  %718 = select i1 %compare.result1332, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1334 = or i32 %compare.local.base1333, 6, !dbg !24
  %compare.low1335 = icmp ugt i32 %compare.low.base, %compare.local1334, !dbg !24
  %compare.result1336 = select i1 %compare.high.equal, i1 %compare.low1335, i1 %compare.high.order, !dbg !24
  %719 = select i1 %compare.result1336, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1338 = or i32 %compare.local.base1337, 7, !dbg !24
  %compare.low1339 = icmp ugt i32 %compare.low.base, %compare.local1338, !dbg !24
  %compare.result1340 = select i1 %compare.high.equal, i1 %compare.low1339, i1 %compare.high.order, !dbg !24
  %720 = select i1 %compare.result1340, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1342 = or i32 %compare.local.base1341, 8, !dbg !24
  %compare.low1343 = icmp ugt i32 %compare.low.base, %compare.local1342, !dbg !24
  %compare.result1344 = select i1 %compare.high.equal, i1 %compare.low1343, i1 %compare.high.order, !dbg !24
  %721 = select i1 %compare.result1344, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1346 = or i32 %compare.local.base1345, 9, !dbg !24
  %compare.low1347 = icmp ugt i32 %compare.low.base, %compare.local1346, !dbg !24
  %compare.result1348 = select i1 %compare.high.equal, i1 %compare.low1347, i1 %compare.high.order, !dbg !24
  %722 = select i1 %compare.result1348, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1350 = or i32 %compare.local.base1349, 10, !dbg !24
  %compare.low1351 = icmp ugt i32 %compare.low.base, %compare.local1350, !dbg !24
  %compare.result1352 = select i1 %compare.high.equal, i1 %compare.low1351, i1 %compare.high.order, !dbg !24
  %723 = select i1 %compare.result1352, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1354 = or i32 %compare.local.base1353, 11, !dbg !24
  %compare.low1355 = icmp ugt i32 %compare.low.base, %compare.local1354, !dbg !24
  %compare.result1356 = select i1 %compare.high.equal, i1 %compare.low1355, i1 %compare.high.order, !dbg !24
  %724 = select i1 %compare.result1356, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1358 = or i32 %compare.local.base1357, 12, !dbg !24
  %compare.low1359 = icmp ugt i32 %compare.low.base, %compare.local1358, !dbg !24
  %compare.result1360 = select i1 %compare.high.equal, i1 %compare.low1359, i1 %compare.high.order, !dbg !24
  %725 = select i1 %compare.result1360, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1362 = or i32 %compare.local.base1361, 13, !dbg !24
  %compare.low1363 = icmp ugt i32 %compare.low.base, %compare.local1362, !dbg !24
  %compare.result1364 = select i1 %compare.high.equal, i1 %compare.low1363, i1 %compare.high.order, !dbg !24
  %726 = select i1 %compare.result1364, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1366 = or i32 %compare.local.base1365, 14, !dbg !24
  %compare.low1367 = icmp ugt i32 %compare.low.base, %compare.local1366, !dbg !24
  %compare.result1368 = select i1 %compare.high.equal, i1 %compare.low1367, i1 %compare.high.order, !dbg !24
  %727 = select i1 %compare.result1368, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1370 = or i32 %compare.local.base1369, 15, !dbg !24
  %compare.low1371 = icmp ugt i32 %compare.low.base, %compare.local1370, !dbg !24
  %compare.result1372 = select i1 %compare.high.equal, i1 %compare.low1371, i1 %compare.high.order, !dbg !24
  %728 = select i1 %compare.result1372, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1374 = or i32 %compare.local.base1373, 64, !dbg !24
  %compare.low1375 = icmp ugt i32 %compare.low.base, %compare.local1374, !dbg !24
  %compare.result1376 = select i1 %compare.high.equal, i1 %compare.low1375, i1 %compare.high.order, !dbg !24
  %729 = select i1 %compare.result1376, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1378 = or i32 %compare.local.base1377, 65, !dbg !24
  %compare.low1379 = icmp ugt i32 %compare.low.base, %compare.local1378, !dbg !24
  %compare.result1380 = select i1 %compare.high.equal, i1 %compare.low1379, i1 %compare.high.order, !dbg !24
  %730 = select i1 %compare.result1380, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1382 = or i32 %compare.local.base1381, 66, !dbg !24
  %compare.low1383 = icmp ugt i32 %compare.low.base, %compare.local1382, !dbg !24
  %compare.result1384 = select i1 %compare.high.equal, i1 %compare.low1383, i1 %compare.high.order, !dbg !24
  %731 = select i1 %compare.result1384, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1385 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1386 = or i32 %compare.local.base1385, 67, !dbg !24
  %compare.low1387 = icmp ugt i32 %compare.low.base, %compare.local1386, !dbg !24
  %compare.result1388 = select i1 %compare.high.equal, i1 %compare.low1387, i1 %compare.high.order, !dbg !24
  %732 = select i1 %compare.result1388, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1389 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1390 = or i32 %compare.local.base1389, 68, !dbg !24
  %compare.low1391 = icmp ugt i32 %compare.low.base, %compare.local1390, !dbg !24
  %compare.result1392 = select i1 %compare.high.equal, i1 %compare.low1391, i1 %compare.high.order, !dbg !24
  %733 = select i1 %compare.result1392, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1393 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1394 = or i32 %compare.local.base1393, 69, !dbg !24
  %compare.low1395 = icmp ugt i32 %compare.low.base, %compare.local1394, !dbg !24
  %compare.result1396 = select i1 %compare.high.equal, i1 %compare.low1395, i1 %compare.high.order, !dbg !24
  %734 = select i1 %compare.result1396, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1397 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1398 = or i32 %compare.local.base1397, 70, !dbg !24
  %compare.low1399 = icmp ugt i32 %compare.low.base, %compare.local1398, !dbg !24
  %compare.result1400 = select i1 %compare.high.equal, i1 %compare.low1399, i1 %compare.high.order, !dbg !24
  %735 = select i1 %compare.result1400, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1401 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1402 = or i32 %compare.local.base1401, 71, !dbg !24
  %compare.low1403 = icmp ugt i32 %compare.low.base, %compare.local1402, !dbg !24
  %compare.result1404 = select i1 %compare.high.equal, i1 %compare.low1403, i1 %compare.high.order, !dbg !24
  %736 = select i1 %compare.result1404, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1405 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1406 = or i32 %compare.local.base1405, 72, !dbg !24
  %compare.low1407 = icmp ugt i32 %compare.low.base, %compare.local1406, !dbg !24
  %compare.result1408 = select i1 %compare.high.equal, i1 %compare.low1407, i1 %compare.high.order, !dbg !24
  %737 = select i1 %compare.result1408, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1409 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1410 = or i32 %compare.local.base1409, 73, !dbg !24
  %compare.low1411 = icmp ugt i32 %compare.low.base, %compare.local1410, !dbg !24
  %compare.result1412 = select i1 %compare.high.equal, i1 %compare.low1411, i1 %compare.high.order, !dbg !24
  %738 = select i1 %compare.result1412, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1413 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1414 = or i32 %compare.local.base1413, 74, !dbg !24
  %compare.low1415 = icmp ugt i32 %compare.low.base, %compare.local1414, !dbg !24
  %compare.result1416 = select i1 %compare.high.equal, i1 %compare.low1415, i1 %compare.high.order, !dbg !24
  %739 = select i1 %compare.result1416, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1417 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1418 = or i32 %compare.local.base1417, 75, !dbg !24
  %compare.low1419 = icmp ugt i32 %compare.low.base, %compare.local1418, !dbg !24
  %compare.result1420 = select i1 %compare.high.equal, i1 %compare.low1419, i1 %compare.high.order, !dbg !24
  %740 = select i1 %compare.result1420, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1421 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1422 = or i32 %compare.local.base1421, 76, !dbg !24
  %compare.low1423 = icmp ugt i32 %compare.low.base, %compare.local1422, !dbg !24
  %compare.result1424 = select i1 %compare.high.equal, i1 %compare.low1423, i1 %compare.high.order, !dbg !24
  %741 = select i1 %compare.result1424, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1425 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1426 = or i32 %compare.local.base1425, 77, !dbg !24
  %compare.low1427 = icmp ugt i32 %compare.low.base, %compare.local1426, !dbg !24
  %compare.result1428 = select i1 %compare.high.equal, i1 %compare.low1427, i1 %compare.high.order, !dbg !24
  %742 = select i1 %compare.result1428, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1429 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1430 = or i32 %compare.local.base1429, 78, !dbg !24
  %compare.low1431 = icmp ugt i32 %compare.low.base, %compare.local1430, !dbg !24
  %compare.result1432 = select i1 %compare.high.equal, i1 %compare.low1431, i1 %compare.high.order, !dbg !24
  %743 = select i1 %compare.result1432, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1433 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1314, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1434 = or i32 %compare.local.base1433, 79, !dbg !24
  %compare.low1435 = icmp ugt i32 %compare.low.base, %compare.local1434, !dbg !24
  %compare.result1436 = select i1 %compare.high.equal, i1 %compare.low1435, i1 %compare.high.order, !dbg !24
  %744 = select i1 %compare.result1436, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %745 = add i64 %707, 65, !dbg !23
  %746 = lshr i64 %745, 32, !dbg !24
  %compare.high1437 = trunc i64 %746 to i32, !dbg !24
  %747 = lshr i64 %709, 32, !dbg !24
  %compare.high1438 = trunc i64 %747 to i32, !dbg !24
  %compare.high.equal1439 = icmp eq i32 %compare.high1437, %compare.high1438, !dbg !24
  %compare.high.order1440 = icmp ugt i32 %compare.high1437, %compare.high1438, !dbg !24
  %compare.low.base1441 = trunc i64 %745 to i32, !dbg !24
  %compare.low.base1442 = trunc i64 %709 to i32, !dbg !24
  %compare.low1443 = icmp ugt i32 %compare.low.base1441, %compare.low.base1442, !dbg !24
  %compare.result1444 = select i1 %compare.high.equal1439, i1 %compare.low1443, i1 %compare.high.order1440, !dbg !24
  %748 = select i1 %compare.result1444, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1445 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1446 = or i32 %compare.local.base1445, 1, !dbg !24
  %compare.low1447 = icmp ugt i32 %compare.low.base1441, %compare.local1446, !dbg !24
  %compare.result1448 = select i1 %compare.high.equal1439, i1 %compare.low1447, i1 %compare.high.order1440, !dbg !24
  %749 = select i1 %compare.result1448, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1449 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1450 = or i32 %compare.local.base1449, 2, !dbg !24
  %compare.low1451 = icmp ugt i32 %compare.low.base1441, %compare.local1450, !dbg !24
  %compare.result1452 = select i1 %compare.high.equal1439, i1 %compare.low1451, i1 %compare.high.order1440, !dbg !24
  %750 = select i1 %compare.result1452, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1453 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1454 = or i32 %compare.local.base1453, 3, !dbg !24
  %compare.low1455 = icmp ugt i32 %compare.low.base1441, %compare.local1454, !dbg !24
  %compare.result1456 = select i1 %compare.high.equal1439, i1 %compare.low1455, i1 %compare.high.order1440, !dbg !24
  %751 = select i1 %compare.result1456, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1457 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1458 = or i32 %compare.local.base1457, 4, !dbg !24
  %compare.low1459 = icmp ugt i32 %compare.low.base1441, %compare.local1458, !dbg !24
  %compare.result1460 = select i1 %compare.high.equal1439, i1 %compare.low1459, i1 %compare.high.order1440, !dbg !24
  %752 = select i1 %compare.result1460, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1461 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1462 = or i32 %compare.local.base1461, 5, !dbg !24
  %compare.low1463 = icmp ugt i32 %compare.low.base1441, %compare.local1462, !dbg !24
  %compare.result1464 = select i1 %compare.high.equal1439, i1 %compare.low1463, i1 %compare.high.order1440, !dbg !24
  %753 = select i1 %compare.result1464, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1465 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1466 = or i32 %compare.local.base1465, 6, !dbg !24
  %compare.low1467 = icmp ugt i32 %compare.low.base1441, %compare.local1466, !dbg !24
  %compare.result1468 = select i1 %compare.high.equal1439, i1 %compare.low1467, i1 %compare.high.order1440, !dbg !24
  %754 = select i1 %compare.result1468, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1469 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1470 = or i32 %compare.local.base1469, 7, !dbg !24
  %compare.low1471 = icmp ugt i32 %compare.low.base1441, %compare.local1470, !dbg !24
  %compare.result1472 = select i1 %compare.high.equal1439, i1 %compare.low1471, i1 %compare.high.order1440, !dbg !24
  %755 = select i1 %compare.result1472, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1473 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1474 = or i32 %compare.local.base1473, 8, !dbg !24
  %compare.low1475 = icmp ugt i32 %compare.low.base1441, %compare.local1474, !dbg !24
  %compare.result1476 = select i1 %compare.high.equal1439, i1 %compare.low1475, i1 %compare.high.order1440, !dbg !24
  %756 = select i1 %compare.result1476, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1477 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1478 = or i32 %compare.local.base1477, 9, !dbg !24
  %compare.low1479 = icmp ugt i32 %compare.low.base1441, %compare.local1478, !dbg !24
  %compare.result1480 = select i1 %compare.high.equal1439, i1 %compare.low1479, i1 %compare.high.order1440, !dbg !24
  %757 = select i1 %compare.result1480, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1481 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1482 = or i32 %compare.local.base1481, 10, !dbg !24
  %compare.low1483 = icmp ugt i32 %compare.low.base1441, %compare.local1482, !dbg !24
  %compare.result1484 = select i1 %compare.high.equal1439, i1 %compare.low1483, i1 %compare.high.order1440, !dbg !24
  %758 = select i1 %compare.result1484, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1485 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1486 = or i32 %compare.local.base1485, 11, !dbg !24
  %compare.low1487 = icmp ugt i32 %compare.low.base1441, %compare.local1486, !dbg !24
  %compare.result1488 = select i1 %compare.high.equal1439, i1 %compare.low1487, i1 %compare.high.order1440, !dbg !24
  %759 = select i1 %compare.result1488, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1489 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1490 = or i32 %compare.local.base1489, 12, !dbg !24
  %compare.low1491 = icmp ugt i32 %compare.low.base1441, %compare.local1490, !dbg !24
  %compare.result1492 = select i1 %compare.high.equal1439, i1 %compare.low1491, i1 %compare.high.order1440, !dbg !24
  %760 = select i1 %compare.result1492, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1493 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1494 = or i32 %compare.local.base1493, 13, !dbg !24
  %compare.low1495 = icmp ugt i32 %compare.low.base1441, %compare.local1494, !dbg !24
  %compare.result1496 = select i1 %compare.high.equal1439, i1 %compare.low1495, i1 %compare.high.order1440, !dbg !24
  %761 = select i1 %compare.result1496, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1497 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1498 = or i32 %compare.local.base1497, 14, !dbg !24
  %compare.low1499 = icmp ugt i32 %compare.low.base1441, %compare.local1498, !dbg !24
  %compare.result1500 = select i1 %compare.high.equal1439, i1 %compare.low1499, i1 %compare.high.order1440, !dbg !24
  %762 = select i1 %compare.result1500, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1501 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1502 = or i32 %compare.local.base1501, 15, !dbg !24
  %compare.low1503 = icmp ugt i32 %compare.low.base1441, %compare.local1502, !dbg !24
  %compare.result1504 = select i1 %compare.high.equal1439, i1 %compare.low1503, i1 %compare.high.order1440, !dbg !24
  %763 = select i1 %compare.result1504, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1505 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1506 = or i32 %compare.local.base1505, 64, !dbg !24
  %compare.low1507 = icmp ugt i32 %compare.low.base1441, %compare.local1506, !dbg !24
  %compare.result1508 = select i1 %compare.high.equal1439, i1 %compare.low1507, i1 %compare.high.order1440, !dbg !24
  %764 = select i1 %compare.result1508, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1509 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1510 = or i32 %compare.local.base1509, 65, !dbg !24
  %compare.low1511 = icmp ugt i32 %compare.low.base1441, %compare.local1510, !dbg !24
  %compare.result1512 = select i1 %compare.high.equal1439, i1 %compare.low1511, i1 %compare.high.order1440, !dbg !24
  %765 = select i1 %compare.result1512, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1513 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1514 = or i32 %compare.local.base1513, 66, !dbg !24
  %compare.low1515 = icmp ugt i32 %compare.low.base1441, %compare.local1514, !dbg !24
  %compare.result1516 = select i1 %compare.high.equal1439, i1 %compare.low1515, i1 %compare.high.order1440, !dbg !24
  %766 = select i1 %compare.result1516, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1517 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1518 = or i32 %compare.local.base1517, 67, !dbg !24
  %compare.low1519 = icmp ugt i32 %compare.low.base1441, %compare.local1518, !dbg !24
  %compare.result1520 = select i1 %compare.high.equal1439, i1 %compare.low1519, i1 %compare.high.order1440, !dbg !24
  %767 = select i1 %compare.result1520, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1521 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1522 = or i32 %compare.local.base1521, 68, !dbg !24
  %compare.low1523 = icmp ugt i32 %compare.low.base1441, %compare.local1522, !dbg !24
  %compare.result1524 = select i1 %compare.high.equal1439, i1 %compare.low1523, i1 %compare.high.order1440, !dbg !24
  %768 = select i1 %compare.result1524, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1525 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1526 = or i32 %compare.local.base1525, 69, !dbg !24
  %compare.low1527 = icmp ugt i32 %compare.low.base1441, %compare.local1526, !dbg !24
  %compare.result1528 = select i1 %compare.high.equal1439, i1 %compare.low1527, i1 %compare.high.order1440, !dbg !24
  %769 = select i1 %compare.result1528, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1529 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1530 = or i32 %compare.local.base1529, 70, !dbg !24
  %compare.low1531 = icmp ugt i32 %compare.low.base1441, %compare.local1530, !dbg !24
  %compare.result1532 = select i1 %compare.high.equal1439, i1 %compare.low1531, i1 %compare.high.order1440, !dbg !24
  %770 = select i1 %compare.result1532, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1533 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1534 = or i32 %compare.local.base1533, 71, !dbg !24
  %compare.low1535 = icmp ugt i32 %compare.low.base1441, %compare.local1534, !dbg !24
  %compare.result1536 = select i1 %compare.high.equal1439, i1 %compare.low1535, i1 %compare.high.order1440, !dbg !24
  %771 = select i1 %compare.result1536, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1537 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1538 = or i32 %compare.local.base1537, 72, !dbg !24
  %compare.low1539 = icmp ugt i32 %compare.low.base1441, %compare.local1538, !dbg !24
  %compare.result1540 = select i1 %compare.high.equal1439, i1 %compare.low1539, i1 %compare.high.order1440, !dbg !24
  %772 = select i1 %compare.result1540, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1541 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1542 = or i32 %compare.local.base1541, 73, !dbg !24
  %compare.low1543 = icmp ugt i32 %compare.low.base1441, %compare.local1542, !dbg !24
  %compare.result1544 = select i1 %compare.high.equal1439, i1 %compare.low1543, i1 %compare.high.order1440, !dbg !24
  %773 = select i1 %compare.result1544, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1545 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1546 = or i32 %compare.local.base1545, 74, !dbg !24
  %compare.low1547 = icmp ugt i32 %compare.low.base1441, %compare.local1546, !dbg !24
  %compare.result1548 = select i1 %compare.high.equal1439, i1 %compare.low1547, i1 %compare.high.order1440, !dbg !24
  %774 = select i1 %compare.result1548, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1549 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1550 = or i32 %compare.local.base1549, 75, !dbg !24
  %compare.low1551 = icmp ugt i32 %compare.low.base1441, %compare.local1550, !dbg !24
  %compare.result1552 = select i1 %compare.high.equal1439, i1 %compare.low1551, i1 %compare.high.order1440, !dbg !24
  %775 = select i1 %compare.result1552, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1553 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1554 = or i32 %compare.local.base1553, 76, !dbg !24
  %compare.low1555 = icmp ugt i32 %compare.low.base1441, %compare.local1554, !dbg !24
  %compare.result1556 = select i1 %compare.high.equal1439, i1 %compare.low1555, i1 %compare.high.order1440, !dbg !24
  %776 = select i1 %compare.result1556, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1557 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1558 = or i32 %compare.local.base1557, 77, !dbg !24
  %compare.low1559 = icmp ugt i32 %compare.low.base1441, %compare.local1558, !dbg !24
  %compare.result1560 = select i1 %compare.high.equal1439, i1 %compare.low1559, i1 %compare.high.order1440, !dbg !24
  %777 = select i1 %compare.result1560, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1561 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1562 = or i32 %compare.local.base1561, 78, !dbg !24
  %compare.low1563 = icmp ugt i32 %compare.low.base1441, %compare.local1562, !dbg !24
  %compare.result1564 = select i1 %compare.high.equal1439, i1 %compare.low1563, i1 %compare.high.order1440, !dbg !24
  %778 = select i1 %compare.result1564, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1565 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1442, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1566 = or i32 %compare.local.base1565, 79, !dbg !24
  %compare.low1567 = icmp ugt i32 %compare.low.base1441, %compare.local1566, !dbg !24
  %compare.result1568 = select i1 %compare.high.equal1439, i1 %compare.low1567, i1 %compare.high.order1440, !dbg !24
  %779 = select i1 %compare.result1568, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
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
  %math.result465 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1006) #9, !dbg !34, !frisk.register_lifetimes !1
  %1007 = fptrunc <4 x float> %math.result465 to <4 x half>, !dbg !34
  %math.packed466 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1007) #9, !dbg !33, !frisk.register_lifetimes !1
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
  %math.result467 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1019) #9, !dbg !34, !frisk.register_lifetimes !1
  %1020 = fptrunc <4 x float> %math.result467 to <4 x half>, !dbg !34
  %math.packed468 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1020) #9, !dbg !33, !frisk.register_lifetimes !1
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
  %math.result469 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1032) #9, !dbg !34, !frisk.register_lifetimes !1
  %1033 = fptrunc <4 x float> %math.result469 to <4 x half>, !dbg !34
  %math.packed470 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1033) #9, !dbg !33, !frisk.register_lifetimes !1
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
  %math.result471 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1045) #9, !dbg !34, !frisk.register_lifetimes !1
  %1046 = fptrunc <4 x float> %math.result471 to <4 x half>, !dbg !34
  %math.packed472 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1046) #9, !dbg !33, !frisk.register_lifetimes !1
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
  %math.result473 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1058) #9, !dbg !34, !frisk.register_lifetimes !1
  %1059 = fptrunc <4 x float> %math.result473 to <4 x half>, !dbg !34
  %math.packed474 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1059) #9, !dbg !33, !frisk.register_lifetimes !1
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
  %math.result475 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1071) #9, !dbg !34, !frisk.register_lifetimes !1
  %1072 = fptrunc <4 x float> %math.result475 to <4 x half>, !dbg !34
  %math.packed476 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1072) #9, !dbg !33, !frisk.register_lifetimes !1
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
  %math.result477 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1084) #9, !dbg !34, !frisk.register_lifetimes !1
  %1085 = fptrunc <4 x float> %math.result477 to <4 x half>, !dbg !34
  %math.packed478 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1085) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce216 = fadd float %local.reduce212, %1083, !dbg !33, !frisk.local_reduction !1
  %local.reduce218 = fadd float %local.reduce215, %local.reduce216, !dbg !33, !frisk.local_reduction !1
  %1086 = fadd float %local.reduce217, %local.reduce218, !dbg !33, !frisk.local_reduction !1
  %1087 = extractelement <4 x float> %958, i64 0, !dbg !32
  %1088 = call float @llvm.amdgcn.exp2.f32(float %1087), !dbg !32
  %1089 = insertelement <4 x float> poison, float %1088, i64 0, !dbg !34
  %1090 = extractelement <4 x float> %958, i64 1, !dbg !32
  %1091 = call float @llvm.amdgcn.exp2.f32(float %1090), !dbg !32
  %1092 = insertelement <4 x float> %1089, float %1091, i64 1, !dbg !34
  %1093 = extractelement <4 x float> %958, i64 2, !dbg !32
  %1094 = call float @llvm.amdgcn.exp2.f32(float %1093), !dbg !32
  %1095 = insertelement <4 x float> %1092, float %1094, i64 2, !dbg !34
  %1096 = extractelement <4 x float> %958, i64 3, !dbg !32
  %1097 = call float @llvm.amdgcn.exp2.f32(float %1096), !dbg !32
  %1098 = insertelement <4 x float> %1095, float %1097, i64 3, !dbg !34
  %math.result479 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1098) #9, !dbg !34, !frisk.register_lifetimes !1
  %1099 = fptrunc <4 x float> %math.result479 to <4 x half>, !dbg !34
  %math.packed480 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1099) #9, !dbg !32, !frisk.register_lifetimes !1
  %1100 = extractelement <4 x float> %959, i64 0, !dbg !32
  %1101 = call float @llvm.amdgcn.exp2.f32(float %1100), !dbg !32
  %1102 = insertelement <4 x float> poison, float %1101, i64 0, !dbg !34
  %local.reduce = fadd float %1088, %1101, !dbg !33, !frisk.local_reduction !1
  %1103 = extractelement <4 x float> %959, i64 1, !dbg !32
  %1104 = call float @llvm.amdgcn.exp2.f32(float %1103), !dbg !32
  %1105 = insertelement <4 x float> %1102, float %1104, i64 1, !dbg !34
  %local.reduce159 = fadd float %1091, %1104, !dbg !33, !frisk.local_reduction !1
  %1106 = extractelement <4 x float> %959, i64 2, !dbg !32
  %1107 = call float @llvm.amdgcn.exp2.f32(float %1106), !dbg !32
  %1108 = insertelement <4 x float> %1105, float %1107, i64 2, !dbg !34
  %local.reduce160 = fadd float %1094, %1107, !dbg !33, !frisk.local_reduction !1
  %1109 = extractelement <4 x float> %959, i64 3, !dbg !32
  %1110 = call float @llvm.amdgcn.exp2.f32(float %1109), !dbg !32
  %1111 = insertelement <4 x float> %1108, float %1110, i64 3, !dbg !34
  %math.result481 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1111) #9, !dbg !34, !frisk.register_lifetimes !1
  %1112 = fptrunc <4 x float> %math.result481 to <4 x half>, !dbg !34
  %math.packed482 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1112) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce161 = fadd float %1097, %1110, !dbg !33, !frisk.local_reduction !1
  %1113 = extractelement <4 x float> %960, i64 0, !dbg !32
  %1114 = call float @llvm.amdgcn.exp2.f32(float %1113), !dbg !32
  %1115 = insertelement <4 x float> poison, float %1114, i64 0, !dbg !34
  %local.reduce162 = fadd float %local.reduce, %1114, !dbg !33, !frisk.local_reduction !1
  %1116 = extractelement <4 x float> %960, i64 1, !dbg !32
  %1117 = call float @llvm.amdgcn.exp2.f32(float %1116), !dbg !32
  %1118 = insertelement <4 x float> %1115, float %1117, i64 1, !dbg !34
  %local.reduce163 = fadd float %local.reduce159, %1117, !dbg !33, !frisk.local_reduction !1
  %1119 = extractelement <4 x float> %960, i64 2, !dbg !32
  %1120 = call float @llvm.amdgcn.exp2.f32(float %1119), !dbg !32
  %1121 = insertelement <4 x float> %1118, float %1120, i64 2, !dbg !34
  %local.reduce164 = fadd float %local.reduce160, %1120, !dbg !33, !frisk.local_reduction !1
  %1122 = extractelement <4 x float> %960, i64 3, !dbg !32
  %1123 = call float @llvm.amdgcn.exp2.f32(float %1122), !dbg !32
  %1124 = insertelement <4 x float> %1121, float %1123, i64 3, !dbg !34
  %math.result483 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1124) #9, !dbg !34, !frisk.register_lifetimes !1
  %1125 = fptrunc <4 x float> %math.result483 to <4 x half>, !dbg !34
  %math.packed484 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1125) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce165 = fadd float %local.reduce161, %1123, !dbg !33, !frisk.local_reduction !1
  %1126 = extractelement <4 x float> %961, i64 0, !dbg !32
  %1127 = call float @llvm.amdgcn.exp2.f32(float %1126), !dbg !32
  %1128 = insertelement <4 x float> poison, float %1127, i64 0, !dbg !34
  %local.reduce166 = fadd float %local.reduce162, %1127, !dbg !33, !frisk.local_reduction !1
  %1129 = extractelement <4 x float> %961, i64 1, !dbg !32
  %1130 = call float @llvm.amdgcn.exp2.f32(float %1129), !dbg !32
  %1131 = insertelement <4 x float> %1128, float %1130, i64 1, !dbg !34
  %local.reduce167 = fadd float %local.reduce163, %1130, !dbg !33, !frisk.local_reduction !1
  %1132 = extractelement <4 x float> %961, i64 2, !dbg !32
  %1133 = call float @llvm.amdgcn.exp2.f32(float %1132), !dbg !32
  %1134 = insertelement <4 x float> %1131, float %1133, i64 2, !dbg !34
  %local.reduce168 = fadd float %local.reduce164, %1133, !dbg !33, !frisk.local_reduction !1
  %1135 = extractelement <4 x float> %961, i64 3, !dbg !32
  %1136 = call float @llvm.amdgcn.exp2.f32(float %1135), !dbg !32
  %1137 = insertelement <4 x float> %1134, float %1136, i64 3, !dbg !34
  %math.result485 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1137) #9, !dbg !34, !frisk.register_lifetimes !1
  %1138 = fptrunc <4 x float> %math.result485 to <4 x half>, !dbg !34
  %math.packed486 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1138) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce169 = fadd float %local.reduce165, %1136, !dbg !33, !frisk.local_reduction !1
  %1139 = extractelement <4 x float> %962, i64 0, !dbg !32
  %1140 = call float @llvm.amdgcn.exp2.f32(float %1139), !dbg !32
  %1141 = insertelement <4 x float> poison, float %1140, i64 0, !dbg !34
  %local.reduce170 = fadd float %local.reduce166, %1140, !dbg !33, !frisk.local_reduction !1
  %1142 = extractelement <4 x float> %962, i64 1, !dbg !32
  %1143 = call float @llvm.amdgcn.exp2.f32(float %1142), !dbg !32
  %1144 = insertelement <4 x float> %1141, float %1143, i64 1, !dbg !34
  %local.reduce171 = fadd float %local.reduce167, %1143, !dbg !33, !frisk.local_reduction !1
  %1145 = extractelement <4 x float> %962, i64 2, !dbg !32
  %1146 = call float @llvm.amdgcn.exp2.f32(float %1145), !dbg !32
  %1147 = insertelement <4 x float> %1144, float %1146, i64 2, !dbg !34
  %local.reduce172 = fadd float %local.reduce168, %1146, !dbg !33, !frisk.local_reduction !1
  %1148 = extractelement <4 x float> %962, i64 3, !dbg !32
  %1149 = call float @llvm.amdgcn.exp2.f32(float %1148), !dbg !32
  %1150 = insertelement <4 x float> %1147, float %1149, i64 3, !dbg !34
  %math.result487 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1150) #9, !dbg !34, !frisk.register_lifetimes !1
  %1151 = fptrunc <4 x float> %math.result487 to <4 x half>, !dbg !34
  %math.packed488 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1151) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce173 = fadd float %local.reduce169, %1149, !dbg !33, !frisk.local_reduction !1
  %1152 = extractelement <4 x float> %963, i64 0, !dbg !32
  %1153 = call float @llvm.amdgcn.exp2.f32(float %1152), !dbg !32
  %1154 = insertelement <4 x float> poison, float %1153, i64 0, !dbg !34
  %local.reduce174 = fadd float %local.reduce170, %1153, !dbg !33, !frisk.local_reduction !1
  %1155 = extractelement <4 x float> %963, i64 1, !dbg !32
  %1156 = call float @llvm.amdgcn.exp2.f32(float %1155), !dbg !32
  %1157 = insertelement <4 x float> %1154, float %1156, i64 1, !dbg !34
  %local.reduce175 = fadd float %local.reduce171, %1156, !dbg !33, !frisk.local_reduction !1
  %1158 = extractelement <4 x float> %963, i64 2, !dbg !32
  %1159 = call float @llvm.amdgcn.exp2.f32(float %1158), !dbg !32
  %1160 = insertelement <4 x float> %1157, float %1159, i64 2, !dbg !34
  %local.reduce176 = fadd float %local.reduce172, %1159, !dbg !33, !frisk.local_reduction !1
  %1161 = extractelement <4 x float> %963, i64 3, !dbg !32
  %1162 = call float @llvm.amdgcn.exp2.f32(float %1161), !dbg !32
  %1163 = insertelement <4 x float> %1160, float %1162, i64 3, !dbg !34
  %math.result489 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1163) #9, !dbg !34, !frisk.register_lifetimes !1
  %1164 = fptrunc <4 x float> %math.result489 to <4 x half>, !dbg !34
  %math.packed490 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1164) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce177 = fadd float %local.reduce173, %1162, !dbg !33, !frisk.local_reduction !1
  %1165 = extractelement <4 x float> %964, i64 0, !dbg !32
  %1166 = call float @llvm.amdgcn.exp2.f32(float %1165), !dbg !32
  %1167 = insertelement <4 x float> poison, float %1166, i64 0, !dbg !34
  %local.reduce178 = fadd float %local.reduce174, %1166, !dbg !33, !frisk.local_reduction !1
  %1168 = extractelement <4 x float> %964, i64 1, !dbg !32
  %1169 = call float @llvm.amdgcn.exp2.f32(float %1168), !dbg !32
  %1170 = insertelement <4 x float> %1167, float %1169, i64 1, !dbg !34
  %local.reduce179 = fadd float %local.reduce175, %1169, !dbg !33, !frisk.local_reduction !1
  %1171 = extractelement <4 x float> %964, i64 2, !dbg !32
  %1172 = call float @llvm.amdgcn.exp2.f32(float %1171), !dbg !32
  %1173 = insertelement <4 x float> %1170, float %1172, i64 2, !dbg !34
  %local.reduce180 = fadd float %local.reduce176, %1172, !dbg !33, !frisk.local_reduction !1
  %1174 = extractelement <4 x float> %964, i64 3, !dbg !32
  %1175 = call float @llvm.amdgcn.exp2.f32(float %1174), !dbg !32
  %1176 = insertelement <4 x float> %1173, float %1175, i64 3, !dbg !34
  %math.result491 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1176) #9, !dbg !34, !frisk.register_lifetimes !1
  %1177 = fptrunc <4 x float> %math.result491 to <4 x half>, !dbg !34
  %math.packed492 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1177) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce181 = fadd float %local.reduce177, %1175, !dbg !33, !frisk.local_reduction !1
  %1178 = extractelement <4 x float> %965, i64 0, !dbg !32
  %1179 = call float @llvm.amdgcn.exp2.f32(float %1178), !dbg !32
  %1180 = insertelement <4 x float> poison, float %1179, i64 0, !dbg !34
  %local.reduce182 = fadd float %local.reduce178, %1179, !dbg !33, !frisk.local_reduction !1
  %1181 = extractelement <4 x float> %965, i64 1, !dbg !32
  %1182 = call float @llvm.amdgcn.exp2.f32(float %1181), !dbg !32
  %1183 = insertelement <4 x float> %1180, float %1182, i64 1, !dbg !34
  %local.reduce183 = fadd float %local.reduce179, %1182, !dbg !33, !frisk.local_reduction !1
  %local.reduce186 = fadd float %local.reduce182, %local.reduce183, !dbg !33, !frisk.local_reduction !1
  %1184 = extractelement <4 x float> %965, i64 2, !dbg !32
  %1185 = call float @llvm.amdgcn.exp2.f32(float %1184), !dbg !32
  %1186 = insertelement <4 x float> %1183, float %1185, i64 2, !dbg !34
  %local.reduce184 = fadd float %local.reduce180, %1185, !dbg !33, !frisk.local_reduction !1
  %1187 = extractelement <4 x float> %965, i64 3, !dbg !32
  %1188 = call float @llvm.amdgcn.exp2.f32(float %1187), !dbg !32
  %1189 = insertelement <4 x float> %1186, float %1188, i64 3, !dbg !34
  %math.result493 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1189) #9, !dbg !34, !frisk.register_lifetimes !1
  %1190 = fptrunc <4 x float> %math.result493 to <4 x half>, !dbg !34
  %math.packed494 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1190) #9, !dbg !33, !frisk.register_lifetimes !1
  %local.reduce185 = fadd float %local.reduce181, %1188, !dbg !33, !frisk.local_reduction !1
  %local.reduce187 = fadd float %local.reduce184, %local.reduce185, !dbg !33, !frisk.local_reduction !1
  %1191 = fadd float %local.reduce186, %local.reduce187, !dbg !33, !frisk.local_reduction !1
  %1192 = bitcast float %1086 to i32, !dbg !33
  %1193 = call i32 @llvm.amdgcn.ds.bpermute(i32 %919, i32 %1192), !dbg !33
  %1194 = bitcast i32 %1193 to float, !dbg !33
  %1195 = fadd float %1086, %1194, !dbg !33
  %1196 = bitcast float %1195 to i32, !dbg !33
  %1197 = call i32 @llvm.amdgcn.ds.bpermute(i32 %927, i32 %1196), !dbg !33
  %1198 = bitcast i32 %1197 to float, !dbg !33
  %1199 = fadd float %1195, %1198, !dbg !33
  %1200 = fadd float %1199, %973, !dbg !33
  %1201 = insertelement <1 x float> poison, float %1200, i64 0, !dbg !7
  %1202 = insertvalue [2 x <1 x float>] poison, <1 x float> %1201, 0, !dbg !7
  %1203 = bitcast float %1191 to i32, !dbg !33
  %1204 = call i32 @llvm.amdgcn.ds.bpermute(i32 %919, i32 %1203), !dbg !33
  %1205 = bitcast i32 %1204 to float, !dbg !33
  %1206 = fadd float %1191, %1205, !dbg !33
  %1207 = bitcast float %1206 to i32, !dbg !33
  %1208 = call i32 @llvm.amdgcn.ds.bpermute(i32 %927, i32 %1207), !dbg !33
  %1209 = bitcast i32 %1208 to float, !dbg !33
  %1210 = fadd float %1206, %1209, !dbg !33
  %1211 = fadd float %1210, %981, !dbg !33
  %1212 = insertelement <1 x float> poison, float %1211, i64 0, !dbg !7
  %1213 = insertvalue [2 x <1 x float>] %1202, <1 x float> %1212, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread1020 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !14, !range !39, !frisk.register_lifetimes !1
  %address.local1021 = shl nuw nsw i32 %address.thread1020, 3, !dbg !14
  %address.local1022 = and i32 %address.local1021, 120, !dbg !14
  %address.local1023 = shl nuw nsw i32 %address.thread1020, 5, !dbg !14
  %address.local1024 = and i32 %address.local1023, 1536, !dbg !14
  %address.local1025 = or disjoint i32 %address.local1022, %address.local1024, !dbg !14
  %address.local1026 = zext nneg i32 %address.local1025 to i64, !dbg !14
  %address.local1027 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1026, !dbg !14
  %1214 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #0, !dbg !14
  %1215 = bitcast <4 x i32> %1214 to <8 x half>, !dbg !14
  %address.local1028 = zext nneg i32 %address.thread1020 to i64, !dbg !8
  %address.local1029 = and i64 %address.local1028, 60, !dbg !14
  %address.local1030 = add nuw nsw i64 %address.local1029, %address.local1028, !dbg !14
  %address.local1031 = shl nuw nsw i64 %address.local1030, 3, !dbg !14
  %address.local1032 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1031, !dbg !14
  %address.local1033 = and i64 %address.local1028, 63, !dbg !11
  %address.local1034 = lshr i64 %address.local1033, 4, !dbg !14
  %address.local1035 = mul nuw nsw i64 %address.local1034, 384, !dbg !14
  %address.local1036 = getelementptr [2 x i8], ptr addrspace(3) %address.local1032, i64 %address.local1035, !dbg !14
  %address.local1037 = lshr i64 %address.local1028, 2, !dbg !17
  %address.local1038 = mul nsw i64 %address.local1037, -32, !dbg !14
  %address.local1039 = getelementptr [2 x i8], ptr addrspace(3) %address.local1036, i64 %address.local1038, !dbg !14
  %address.local1040 = getelementptr i8, ptr addrspace(3) %address.local1039, i64 256, !dbg !14
  %1216 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #0, !dbg !14
  %1217 = bitcast <4 x i32> %1216 to <8 x half>, !dbg !14
  %address.local1041 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1031, !dbg !14
  %address.local1042 = getelementptr [2 x i8], ptr addrspace(3) %address.local1041, i64 %address.local1035, !dbg !14
  %address.local1043 = getelementptr [2 x i8], ptr addrspace(3) %address.local1042, i64 %address.local1038, !dbg !14
  %address.local1044 = getelementptr i8, ptr addrspace(3) %address.local1043, i64 512, !dbg !14
  %1218 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #0, !dbg !14
  %1219 = bitcast <4 x i32> %1218 to <8 x half>, !dbg !14
  %address.local1045 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1031, !dbg !14
  %address.local1046 = getelementptr [2 x i8], ptr addrspace(3) %address.local1045, i64 %address.local1035, !dbg !14
  %address.local1047 = getelementptr [2 x i8], ptr addrspace(3) %address.local1046, i64 %address.local1038, !dbg !14
  %address.local1048 = getelementptr i8, ptr addrspace(3) %address.local1047, i64 768, !dbg !14
  %1220 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #0, !dbg !14
  %1221 = bitcast <4 x i32> %1220 to <8 x half>, !dbg !14
  %address.local1049 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1026, !dbg !14
  %address.local1050 = getelementptr i8, ptr addrspace(3) %address.local1049, i64 4096, !dbg !14
  %1222 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1223 = bitcast <4 x i32> %1222 to <8 x half>, !dbg !14
  %address.local1051 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1031, !dbg !14
  %address.local1052 = getelementptr [2 x i8], ptr addrspace(3) %address.local1051, i64 %address.local1035, !dbg !14
  %address.local1053 = getelementptr [2 x i8], ptr addrspace(3) %address.local1052, i64 %address.local1038, !dbg !14
  %address.local1054 = getelementptr i8, ptr addrspace(3) %address.local1053, i64 4352, !dbg !14
  %1224 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1225 = bitcast <4 x i32> %1224 to <8 x half>, !dbg !14
  %address.local1055 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1031, !dbg !14
  %address.local1056 = getelementptr [2 x i8], ptr addrspace(3) %address.local1055, i64 %address.local1035, !dbg !14
  %address.local1057 = getelementptr [2 x i8], ptr addrspace(3) %address.local1056, i64 %address.local1038, !dbg !14
  %address.local1058 = getelementptr i8, ptr addrspace(3) %address.local1057, i64 4608, !dbg !14
  %1226 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1227 = bitcast <4 x i32> %1226 to <8 x half>, !dbg !14
  %address.local1059 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1031, !dbg !14
  %address.local1060 = getelementptr [2 x i8], ptr addrspace(3) %address.local1059, i64 %address.local1035, !dbg !14
  %address.local1061 = getelementptr [2 x i8], ptr addrspace(3) %address.local1060, i64 %address.local1038, !dbg !14
  %address.local1062 = getelementptr i8, ptr addrspace(3) %address.local1061, i64 4864, !dbg !14
  %1228 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1062, i32 0) #0, !dbg !14
  %1229 = bitcast <4 x i32> %1228 to <8 x half>, !dbg !14
  %1230 = add nuw nsw i64 %217, %16, !dbg !18
  %1231 = shl i64 %1230, 19, !dbg !18
  %1232 = and i64 %1231, 16252928, !dbg !18
  %1233 = add i64 %1232, %223, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1234 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1233, !dbg !18
  %1235 = getelementptr i8, ptr addrspace(1) %1234, i64 64, !dbg !18
  %address.local1063 = shl nuw nsw i64 %address.local1028, 3, !dbg !8
  %address.local1064 = and i64 %address.local1063, 1536, !dbg !8
  %address.local1065 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1064, !dbg !8
  %address.local1066 = ptrtoint ptr addrspace(3) %address.local1065 to i64, !dbg !8
  %address.local1067 = trunc i64 %address.local1066 to i32, !dbg !8
  %1236 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1067), !dbg !18
  %1237 = zext i32 %1236 to i64, !dbg !18
  %1238 = inttoptr i64 %1237 to ptr addrspace(3), !dbg !18
  %address.local1068 = shl nuw nsw i64 %address.local1028, 5, !dbg !18
  %address.local1069 = and i64 %address.local1068, 8064, !dbg !18
  %address.local1070 = shl nuw nsw i64 %address.local1028, 3, !dbg !18
  %address.local1071 = and i64 %address.local1070, 24, !dbg !18
  %address.local1072 = or disjoint i64 %address.local1069, %address.local1071, !dbg !18
  %address.local1073 = getelementptr [2 x i8], ptr addrspace(1) %1235, i64 %address.local1072, !dbg !18
  %1239 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1240 = ptrtoint ptr addrspace(1) %address.local1073 to i64, !dbg !18
  %1241 = sub i64 %1240, %1239, !dbg !18
  %dma.buffer.offset1714 = trunc i64 %1241 to i32, !dbg !18
  %1242 = trunc i64 %1239 to i32, !dbg !18
  %1243 = insertelement <4 x i32> poison, i32 %1242, i64 0, !dbg !18
  %1244 = lshr i64 %1239, 32, !dbg !18
  %1245 = trunc i64 %1244 to i32, !dbg !18
  %1246 = insertelement <4 x i32> %1243, i32 %1245, i64 1, !dbg !18
  %1247 = insertelement <4 x i32> %1246, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1715 = insertelement <4 x i32> %1247, i32 131072, i64 3, !dbg !18
  %1248 = ptrtoint ptr addrspace(3) %1238 to i32, !dbg !18
  %dma.buffer.lds1716 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1248), !dbg !18
  %1249 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1716, i32 %dma.buffer.offset1714, <4 x i32> %dma.buffer.rsrc1715) #10, !dbg !18
  %address.local1074 = or disjoint i64 %address.local1028, 256, !dbg !8
  %address.local1075 = shl nuw nsw i64 %address.local1074, 3, !dbg !8
  %address.local1076 = and i64 %address.local1075, 3584, !dbg !8
  %address.local1077 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1076, !dbg !8
  %address.local1078 = ptrtoint ptr addrspace(3) %address.local1077 to i64, !dbg !8
  %address.local1079 = trunc i64 %address.local1078 to i32, !dbg !8
  %1250 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1079), !dbg !18
  %1251 = zext i32 %1250 to i64, !dbg !18
  %1252 = inttoptr i64 %1251 to ptr addrspace(3), !dbg !18
  %address.local1080 = shl nuw nsw i64 %address.local1074, 5, !dbg !18
  %address.local1081 = and i64 %address.local1080, 16256, !dbg !18
  %address.local1082 = shl nuw nsw i64 %address.local1028, 3, !dbg !18
  %address.local1083 = and i64 %address.local1082, 24, !dbg !18
  %address.local1084 = or disjoint i64 %address.local1081, %address.local1083, !dbg !18
  %address.local1085 = getelementptr [2 x i8], ptr addrspace(1) %1235, i64 %address.local1084, !dbg !18
  %1253 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1254 = ptrtoint ptr addrspace(1) %address.local1085 to i64, !dbg !18
  %1255 = sub i64 %1254, %1253, !dbg !18
  %dma.buffer.offset1711 = trunc i64 %1255 to i32, !dbg !18
  %1256 = trunc i64 %1253 to i32, !dbg !18
  %1257 = insertelement <4 x i32> poison, i32 %1256, i64 0, !dbg !18
  %1258 = lshr i64 %1253, 32, !dbg !18
  %1259 = trunc i64 %1258 to i32, !dbg !18
  %1260 = insertelement <4 x i32> %1257, i32 %1259, i64 1, !dbg !18
  %1261 = insertelement <4 x i32> %1260, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1712 = insertelement <4 x i32> %1261, i32 131072, i64 3, !dbg !18
  %1262 = ptrtoint ptr addrspace(3) %1252 to i32, !dbg !18
  %dma.buffer.lds1713 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1262), !dbg !18
  %1263 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1713, i32 %dma.buffer.offset1711, <4 x i32> %dma.buffer.rsrc1712) #10, !dbg !18
  %1264 = shufflevector <8 x half> %1215, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1265 = extractvalue [2 x <8 x float>] %206, 0, !dbg !35
  %1266 = shufflevector <8 x float> %1265, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1267 = fmul <4 x float> %1266, %970, !dbg !35
  %1268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1264, <4 x float> %1267) #9, !dbg !14
  %1269 = shufflevector <8 x half> %1217, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1269, <4 x float> %1268) #9, !dbg !14
  %1271 = shufflevector <8 x half> %1219, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1271, <4 x float> %1270) #9, !dbg !14
  %1273 = shufflevector <8 x half> %1221, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1273, <4 x float> %1272) #9, !dbg !14
  %1275 = shufflevector <8 x half> %1223, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %1275, <4 x float> %1274) #9, !dbg !14
  %1277 = shufflevector <8 x half> %1225, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %1277, <4 x float> %1276) #9, !dbg !14
  %1279 = shufflevector <8 x half> %1227, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1279, <4 x float> %1278) #9, !dbg !14
  %1281 = shufflevector <8 x half> %1229, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1281, <4 x float> %1280) #9, !dbg !14
  %1283 = shufflevector <8 x half> %1215, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1284 = shufflevector <8 x float> %1265, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1285 = fmul <4 x float> %1284, %970, !dbg !35
  %1286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1283, <4 x float> %1285) #9, !dbg !14
  %1287 = shufflevector <8 x half> %1217, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1287, <4 x float> %1286) #9, !dbg !14
  %1289 = shufflevector <8 x half> %1219, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1289, <4 x float> %1288) #9, !dbg !14
  %1291 = shufflevector <8 x half> %1221, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1291, <4 x float> %1290) #9, !dbg !14
  %1293 = shufflevector <8 x half> %1223, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %1293, <4 x float> %1292) #9, !dbg !14
  %1295 = shufflevector <8 x half> %1225, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %1295, <4 x float> %1294) #9, !dbg !14
  %1297 = shufflevector <8 x half> %1227, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1297, <4 x float> %1296) #9, !dbg !14
  %1299 = shufflevector <8 x half> %1229, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1299, <4 x float> %1298) #9, !dbg !14
  %1301 = extractvalue [2 x <8 x float>] %206, 1, !dbg !35
  %1302 = shufflevector <8 x float> %1301, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1303 = fmul <4 x float> %1302, %978, !dbg !35
  %1304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1264, <4 x float> %1303) #9, !dbg !14
  %1305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1269, <4 x float> %1304) #9, !dbg !14
  %1306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1271, <4 x float> %1305) #9, !dbg !14
  %1307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1273, <4 x float> %1306) #9, !dbg !14
  %1308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1275, <4 x float> %1307) #9, !dbg !14
  %1309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1277, <4 x float> %1308) #9, !dbg !14
  %1310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1279, <4 x float> %1309) #9, !dbg !14
  %1311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1281, <4 x float> %1310) #9, !dbg !14
  %1312 = shufflevector <8 x float> %1301, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1313 = fmul <4 x float> %1312, %978, !dbg !35
  %1314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1283, <4 x float> %1313) #9, !dbg !14
  %1315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1287, <4 x float> %1314) #9, !dbg !14
  %1316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1289, <4 x float> %1315) #9, !dbg !14
  %1317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1291, <4 x float> %1316) #9, !dbg !14
  %1318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1293, <4 x float> %1317) #9, !dbg !14
  %1319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1295, <4 x float> %1318) #9, !dbg !14
  %1320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1297, <4 x float> %1319) #9, !dbg !14
  %1321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1299, <4 x float> %1320) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1322 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #0, !dbg !14
  %1323 = bitcast <4 x i32> %1322 to <8 x half>, !dbg !14
  %1324 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #0, !dbg !14
  %1325 = bitcast <4 x i32> %1324 to <8 x half>, !dbg !14
  %1326 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #0, !dbg !14
  %1327 = bitcast <4 x i32> %1326 to <8 x half>, !dbg !14
  %1328 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #0, !dbg !14
  %1329 = bitcast <4 x i32> %1328 to <8 x half>, !dbg !14
  %1330 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1331 = bitcast <4 x i32> %1330 to <8 x half>, !dbg !14
  %1332 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1333 = bitcast <4 x i32> %1332 to <8 x half>, !dbg !14
  %1334 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1335 = bitcast <4 x i32> %1334 to <8 x half>, !dbg !14
  %1336 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1062, i32 0) #0, !dbg !14
  %1337 = bitcast <4 x i32> %1336 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1338 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1233, !dbg !18
  %1339 = getelementptr i8, ptr addrspace(1) %1338, i64 128, !dbg !18
  %1340 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1067), !dbg !18
  %1341 = zext i32 %1340 to i64, !dbg !18
  %1342 = inttoptr i64 %1341 to ptr addrspace(3), !dbg !18
  %address.local1086 = getelementptr [2 x i8], ptr addrspace(1) %1339, i64 %address.local1072, !dbg !18
  %1343 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1344 = ptrtoint ptr addrspace(1) %address.local1086 to i64, !dbg !18
  %1345 = sub i64 %1344, %1343, !dbg !18
  %dma.buffer.offset1708 = trunc i64 %1345 to i32, !dbg !18
  %1346 = trunc i64 %1343 to i32, !dbg !18
  %1347 = insertelement <4 x i32> poison, i32 %1346, i64 0, !dbg !18
  %1348 = lshr i64 %1343, 32, !dbg !18
  %1349 = trunc i64 %1348 to i32, !dbg !18
  %1350 = insertelement <4 x i32> %1347, i32 %1349, i64 1, !dbg !18
  %1351 = insertelement <4 x i32> %1350, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1709 = insertelement <4 x i32> %1351, i32 131072, i64 3, !dbg !18
  %1352 = ptrtoint ptr addrspace(3) %1342 to i32, !dbg !18
  %dma.buffer.lds1710 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1352), !dbg !18
  %1353 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1710, i32 %dma.buffer.offset1708, <4 x i32> %dma.buffer.rsrc1709) #10, !dbg !18
  %1354 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1079), !dbg !18
  %1355 = zext i32 %1354 to i64, !dbg !18
  %1356 = inttoptr i64 %1355 to ptr addrspace(3), !dbg !18
  %address.local1087 = getelementptr [2 x i8], ptr addrspace(1) %1339, i64 %address.local1084, !dbg !18
  %1357 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1358 = ptrtoint ptr addrspace(1) %address.local1087 to i64, !dbg !18
  %1359 = sub i64 %1358, %1357, !dbg !18
  %dma.buffer.offset1705 = trunc i64 %1359 to i32, !dbg !18
  %1360 = trunc i64 %1357 to i32, !dbg !18
  %1361 = insertelement <4 x i32> poison, i32 %1360, i64 0, !dbg !18
  %1362 = lshr i64 %1357, 32, !dbg !18
  %1363 = trunc i64 %1362 to i32, !dbg !18
  %1364 = insertelement <4 x i32> %1361, i32 %1363, i64 1, !dbg !18
  %1365 = insertelement <4 x i32> %1364, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1706 = insertelement <4 x i32> %1365, i32 131072, i64 3, !dbg !18
  %1366 = ptrtoint ptr addrspace(3) %1356 to i32, !dbg !18
  %dma.buffer.lds1707 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1366), !dbg !18
  %1367 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1707, i32 %dma.buffer.offset1705, <4 x i32> %dma.buffer.rsrc1706) #10, !dbg !18
  %1368 = shufflevector <8 x half> %1323, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1369 = extractvalue [2 x <8 x float>] %209, 0, !dbg !35
  %1370 = shufflevector <8 x float> %1369, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1371 = fmul <4 x float> %1370, %970, !dbg !35
  %1372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1368, <4 x float> %1371) #9, !dbg !14
  %1373 = shufflevector <8 x half> %1325, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1373, <4 x float> %1372) #9, !dbg !14
  %1375 = shufflevector <8 x half> %1327, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1375, <4 x float> %1374) #9, !dbg !14
  %1377 = shufflevector <8 x half> %1329, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1377, <4 x float> %1376) #9, !dbg !14
  %1379 = shufflevector <8 x half> %1331, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %1379, <4 x float> %1378) #9, !dbg !14
  %1381 = shufflevector <8 x half> %1333, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %1381, <4 x float> %1380) #9, !dbg !14
  %1383 = shufflevector <8 x half> %1335, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1383, <4 x float> %1382) #9, !dbg !14
  %1385 = shufflevector <8 x half> %1337, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1385, <4 x float> %1384) #9, !dbg !14
  %1387 = shufflevector <8 x half> %1323, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1388 = shufflevector <8 x float> %1369, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1389 = fmul <4 x float> %1388, %970, !dbg !35
  %1390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1387, <4 x float> %1389) #9, !dbg !14
  %1391 = shufflevector <8 x half> %1325, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1391, <4 x float> %1390) #9, !dbg !14
  %1393 = shufflevector <8 x half> %1327, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1393, <4 x float> %1392) #9, !dbg !14
  %1395 = shufflevector <8 x half> %1329, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1395, <4 x float> %1394) #9, !dbg !14
  %1397 = shufflevector <8 x half> %1331, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %1397, <4 x float> %1396) #9, !dbg !14
  %1399 = shufflevector <8 x half> %1333, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %1399, <4 x float> %1398) #9, !dbg !14
  %1401 = shufflevector <8 x half> %1335, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1401, <4 x float> %1400) #9, !dbg !14
  %1403 = shufflevector <8 x half> %1337, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1403, <4 x float> %1402) #9, !dbg !14
  %1405 = extractvalue [2 x <8 x float>] %209, 1, !dbg !35
  %1406 = shufflevector <8 x float> %1405, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1407 = fmul <4 x float> %1406, %978, !dbg !35
  %1408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1368, <4 x float> %1407) #9, !dbg !14
  %1409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1373, <4 x float> %1408) #9, !dbg !14
  %1410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1375, <4 x float> %1409) #9, !dbg !14
  %1411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1377, <4 x float> %1410) #9, !dbg !14
  %1412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1379, <4 x float> %1411) #9, !dbg !14
  %1413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1381, <4 x float> %1412) #9, !dbg !14
  %1414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1383, <4 x float> %1413) #9, !dbg !14
  %1415 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1385, <4 x float> %1414) #9, !dbg !14
  %1416 = shufflevector <8 x float> %1405, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1417 = fmul <4 x float> %1416, %978, !dbg !35
  %1418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1387, <4 x float> %1417) #9, !dbg !14
  %1419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1391, <4 x float> %1418) #9, !dbg !14
  %1420 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1393, <4 x float> %1419) #9, !dbg !14
  %1421 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1395, <4 x float> %1420) #9, !dbg !14
  %1422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1397, <4 x float> %1421) #9, !dbg !14
  %1423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1399, <4 x float> %1422) #9, !dbg !14
  %1424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1401, <4 x float> %1423) #9, !dbg !14
  %1425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1403, <4 x float> %1424) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1426 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #0, !dbg !14
  %1427 = bitcast <4 x i32> %1426 to <8 x half>, !dbg !14
  %1428 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #0, !dbg !14
  %1429 = bitcast <4 x i32> %1428 to <8 x half>, !dbg !14
  %1430 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #0, !dbg !14
  %1431 = bitcast <4 x i32> %1430 to <8 x half>, !dbg !14
  %1432 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #0, !dbg !14
  %1433 = bitcast <4 x i32> %1432 to <8 x half>, !dbg !14
  %1434 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1435 = bitcast <4 x i32> %1434 to <8 x half>, !dbg !14
  %1436 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1437 = bitcast <4 x i32> %1436 to <8 x half>, !dbg !14
  %1438 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1439 = bitcast <4 x i32> %1438 to <8 x half>, !dbg !14
  %1440 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1062, i32 0) #0, !dbg !14
  %1441 = bitcast <4 x i32> %1440 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1442 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1233, !dbg !18
  %1443 = getelementptr i8, ptr addrspace(1) %1442, i64 192, !dbg !18
  %1444 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1067), !dbg !18
  %1445 = zext i32 %1444 to i64, !dbg !18
  %1446 = inttoptr i64 %1445 to ptr addrspace(3), !dbg !18
  %address.local1088 = getelementptr [2 x i8], ptr addrspace(1) %1443, i64 %address.local1072, !dbg !18
  %1447 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1448 = ptrtoint ptr addrspace(1) %address.local1088 to i64, !dbg !18
  %1449 = sub i64 %1448, %1447, !dbg !18
  %dma.buffer.offset1702 = trunc i64 %1449 to i32, !dbg !18
  %1450 = trunc i64 %1447 to i32, !dbg !18
  %1451 = insertelement <4 x i32> poison, i32 %1450, i64 0, !dbg !18
  %1452 = lshr i64 %1447, 32, !dbg !18
  %1453 = trunc i64 %1452 to i32, !dbg !18
  %1454 = insertelement <4 x i32> %1451, i32 %1453, i64 1, !dbg !18
  %1455 = insertelement <4 x i32> %1454, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1703 = insertelement <4 x i32> %1455, i32 131072, i64 3, !dbg !18
  %1456 = ptrtoint ptr addrspace(3) %1446 to i32, !dbg !18
  %dma.buffer.lds1704 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1456), !dbg !18
  %1457 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1704, i32 %dma.buffer.offset1702, <4 x i32> %dma.buffer.rsrc1703) #10, !dbg !18
  %1458 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1079), !dbg !18
  %1459 = zext i32 %1458 to i64, !dbg !18
  %1460 = inttoptr i64 %1459 to ptr addrspace(3), !dbg !18
  %address.local1089 = getelementptr [2 x i8], ptr addrspace(1) %1443, i64 %address.local1084, !dbg !18
  %1461 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1462 = ptrtoint ptr addrspace(1) %address.local1089 to i64, !dbg !18
  %1463 = sub i64 %1462, %1461, !dbg !18
  %dma.buffer.offset = trunc i64 %1463 to i32, !dbg !18
  %1464 = trunc i64 %1461 to i32, !dbg !18
  %1465 = insertelement <4 x i32> poison, i32 %1464, i64 0, !dbg !18
  %1466 = lshr i64 %1461, 32, !dbg !18
  %1467 = trunc i64 %1466 to i32, !dbg !18
  %1468 = insertelement <4 x i32> %1465, i32 %1467, i64 1, !dbg !18
  %1469 = insertelement <4 x i32> %1468, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1469, i32 131072, i64 3, !dbg !18
  %1470 = ptrtoint ptr addrspace(3) %1460 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1470), !dbg !18
  %1471 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1472 = shufflevector <8 x half> %1427, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1473 = extractvalue [2 x <8 x float>] %210, 0, !dbg !35
  %1474 = shufflevector <8 x float> %1473, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1475 = fmul <4 x float> %1474, %970, !dbg !35
  %1476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1472, <4 x float> %1475) #9, !dbg !14
  %1477 = shufflevector <8 x half> %1429, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1477, <4 x float> %1476) #9, !dbg !14
  %1479 = shufflevector <8 x half> %1431, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1479, <4 x float> %1478) #9, !dbg !14
  %1481 = shufflevector <8 x half> %1433, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1481, <4 x float> %1480) #9, !dbg !14
  %1483 = shufflevector <8 x half> %1435, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %1483, <4 x float> %1482) #9, !dbg !14
  %1485 = shufflevector <8 x half> %1437, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %1485, <4 x float> %1484) #9, !dbg !14
  %1487 = shufflevector <8 x half> %1439, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1487, <4 x float> %1486) #9, !dbg !14
  %1489 = shufflevector <8 x half> %1441, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1489, <4 x float> %1488) #9, !dbg !14
  %1491 = shufflevector <8 x half> %1427, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1492 = shufflevector <8 x float> %1473, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1493 = fmul <4 x float> %1492, %970, !dbg !35
  %1494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1491, <4 x float> %1493) #9, !dbg !14
  %1495 = shufflevector <8 x half> %1429, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1495, <4 x float> %1494) #9, !dbg !14
  %1497 = shufflevector <8 x half> %1431, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1497, <4 x float> %1496) #9, !dbg !14
  %1499 = shufflevector <8 x half> %1433, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1499, <4 x float> %1498) #9, !dbg !14
  %1501 = shufflevector <8 x half> %1435, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %1501, <4 x float> %1500) #9, !dbg !14
  %1503 = shufflevector <8 x half> %1437, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %1503, <4 x float> %1502) #9, !dbg !14
  %1505 = shufflevector <8 x half> %1439, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1505, <4 x float> %1504) #9, !dbg !14
  %1507 = shufflevector <8 x half> %1441, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1507, <4 x float> %1506) #9, !dbg !14
  %1509 = extractvalue [2 x <8 x float>] %210, 1, !dbg !35
  %1510 = shufflevector <8 x float> %1509, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1511 = fmul <4 x float> %1510, %978, !dbg !35
  %1512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1472, <4 x float> %1511) #9, !dbg !14
  %1513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1477, <4 x float> %1512) #9, !dbg !14
  %1514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1479, <4 x float> %1513) #9, !dbg !14
  %1515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1481, <4 x float> %1514) #9, !dbg !14
  %1516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1483, <4 x float> %1515) #9, !dbg !14
  %1517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1485, <4 x float> %1516) #9, !dbg !14
  %1518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1487, <4 x float> %1517) #9, !dbg !14
  %1519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1489, <4 x float> %1518) #9, !dbg !14
  %1520 = shufflevector <8 x float> %1509, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1521 = fmul <4 x float> %1520, %978, !dbg !35
  %1522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1491, <4 x float> %1521) #9, !dbg !14
  %1523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1495, <4 x float> %1522) #9, !dbg !14
  %1524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1497, <4 x float> %1523) #9, !dbg !14
  %1525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1499, <4 x float> %1524) #9, !dbg !14
  %1526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1501, <4 x float> %1525) #9, !dbg !14
  %1527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1503, <4 x float> %1526) #9, !dbg !14
  %1528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1505, <4 x float> %1527) #9, !dbg !14
  %1529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1507, <4 x float> %1528) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1530 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #0, !dbg !14
  %1531 = bitcast <4 x i32> %1530 to <8 x half>, !dbg !14
  %1532 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1040, i32 0) #0, !dbg !14
  %1533 = bitcast <4 x i32> %1532 to <8 x half>, !dbg !14
  %1534 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1044, i32 0) #0, !dbg !14
  %1535 = bitcast <4 x i32> %1534 to <8 x half>, !dbg !14
  %1536 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1048, i32 0) #0, !dbg !14
  %1537 = bitcast <4 x i32> %1536 to <8 x half>, !dbg !14
  %1538 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1050, i32 0) #0, !dbg !14
  %1539 = bitcast <4 x i32> %1538 to <8 x half>, !dbg !14
  %1540 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1054, i32 0) #0, !dbg !14
  %1541 = bitcast <4 x i32> %1540 to <8 x half>, !dbg !14
  %1542 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1058, i32 0) #0, !dbg !14
  %1543 = bitcast <4 x i32> %1542 to <8 x half>, !dbg !14
  %1544 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1062, i32 0) #0, !dbg !14
  %1545 = bitcast <4 x i32> %1544 to <8 x half>, !dbg !14
  %1546 = shufflevector <8 x half> %1531, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1547 = extractvalue [2 x <8 x float>] %211, 0, !dbg !35
  %1548 = shufflevector <8 x float> %1547, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1549 = fmul <4 x float> %1548, %970, !dbg !35
  %1550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1546, <4 x float> %1549) #9, !dbg !14
  %1551 = shufflevector <8 x half> %1533, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1551, <4 x float> %1550) #9, !dbg !14
  %1553 = shufflevector <8 x half> %1535, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1553, <4 x float> %1552) #9, !dbg !14
  %1555 = shufflevector <8 x half> %1537, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1555, <4 x float> %1554) #9, !dbg !14
  %1557 = shufflevector <8 x half> %1539, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %1557, <4 x float> %1556) #9, !dbg !14
  %1559 = shufflevector <8 x half> %1541, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %1559, <4 x float> %1558) #9, !dbg !14
  %1561 = shufflevector <8 x half> %1543, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1561, <4 x float> %1560) #9, !dbg !14
  %1563 = shufflevector <8 x half> %1545, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1563, <4 x float> %1562) #9, !dbg !14
  %1565 = shufflevector <8 x half> %1531, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1566 = shufflevector <8 x float> %1547, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1567 = fmul <4 x float> %1566, %970, !dbg !35
  %1568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1565, <4 x float> %1567) #9, !dbg !14
  %1569 = shufflevector <8 x half> %1533, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1569, <4 x float> %1568) #9, !dbg !14
  %1571 = shufflevector <8 x half> %1535, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1571, <4 x float> %1570) #9, !dbg !14
  %1573 = shufflevector <8 x half> %1537, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1573, <4 x float> %1572) #9, !dbg !14
  %1575 = shufflevector <8 x half> %1539, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed472, <4 x half> %1575, <4 x float> %1574) #9, !dbg !14
  %1577 = shufflevector <8 x half> %1541, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed474, <4 x half> %1577, <4 x float> %1576) #9, !dbg !14
  %1579 = shufflevector <8 x half> %1543, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed476, <4 x half> %1579, <4 x float> %1578) #9, !dbg !14
  %1581 = shufflevector <8 x half> %1545, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed478, <4 x half> %1581, <4 x float> %1580) #9, !dbg !14
  %1583 = extractvalue [2 x <8 x float>] %211, 1, !dbg !35
  %1584 = shufflevector <8 x float> %1583, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1585 = fmul <4 x float> %1584, %978, !dbg !35
  %1586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1546, <4 x float> %1585) #9, !dbg !14
  %1587 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1551, <4 x float> %1586) #9, !dbg !14
  %1588 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1553, <4 x float> %1587) #9, !dbg !14
  %1589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1555, <4 x float> %1588) #9, !dbg !14
  %1590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1557, <4 x float> %1589) #9, !dbg !14
  %1591 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1559, <4 x float> %1590) #9, !dbg !14
  %1592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1561, <4 x float> %1591) #9, !dbg !14
  %1593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1563, <4 x float> %1592) #9, !dbg !14
  %1594 = shufflevector <8 x float> %1583, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1595 = fmul <4 x float> %1594, %978, !dbg !35
  %1596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed480, <4 x half> %1565, <4 x float> %1595) #9, !dbg !14
  %1597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed482, <4 x half> %1569, <4 x float> %1596) #9, !dbg !14
  %1598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed484, <4 x half> %1571, <4 x float> %1597) #9, !dbg !14
  %1599 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed486, <4 x half> %1573, <4 x float> %1598) #9, !dbg !14
  %1600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed488, <4 x half> %1575, <4 x float> %1599) #9, !dbg !14
  %1601 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed490, <4 x half> %1577, <4 x float> %1600) #9, !dbg !14
  %1602 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed492, <4 x half> %1579, <4 x float> %1601) #9, !dbg !14
  %1603 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed494, <4 x half> %1581, <4 x float> %1602) #9, !dbg !14
  %1604 = shufflevector <4 x float> %1282, <4 x float> %1300, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1605 = insertvalue [2 x <8 x float>] poison, <8 x float> %1604, 0, !dbg !7
  %1606 = shufflevector <4 x float> %1311, <4 x float> %1321, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1607 = insertvalue [2 x <8 x float>] %1605, <8 x float> %1606, 1, !dbg !7
  %1608 = insertvalue [2 x <1 x float>] poison, <1 x float> %934, 0, !dbg !7
  %1609 = insertvalue [2 x <1 x float>] %1608, <1 x float> %947, 1, !dbg !7
  %1610 = shufflevector <4 x float> %1386, <4 x float> %1404, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1611 = insertvalue [2 x <8 x float>] poison, <8 x float> %1610, 0, !dbg !7
  %1612 = shufflevector <4 x float> %1415, <4 x float> %1425, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1613 = insertvalue [2 x <8 x float>] %1611, <8 x float> %1612, 1, !dbg !7
  %1614 = shufflevector <4 x float> %1490, <4 x float> %1508, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1615 = insertvalue [2 x <8 x float>] poison, <8 x float> %1614, 0, !dbg !7
  %1616 = shufflevector <4 x float> %1519, <4 x float> %1529, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1617 = insertvalue [2 x <8 x float>] %1615, <8 x float> %1616, 1, !dbg !7
  %1618 = shufflevector <4 x float> %1564, <4 x float> %1582, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1619 = insertvalue [2 x <8 x float>] poison, <8 x float> %1618, 0, !dbg !7
  %1620 = shufflevector <4 x float> %1593, <4 x float> %1603, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1621 = insertvalue [2 x <8 x float>] %1619, <8 x float> %1620, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1622 = add nuw nsw i64 %205, 1, !dbg !7
  br label %204, !dbg !7

1623:                                             ; preds = %204
  %1624 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %1624, 0, !dbg !7
  %1625 = and i32 %31, 32, !dbg !7
  %.not36 = icmp eq i32 %1625, 0, !dbg !7
  %1626 = extractvalue [2 x <8 x float>] %211, 0, !dbg !7
  %1627 = extractelement <8 x float> %1626, i64 0, !dbg !7
  %1628 = extractelement <8 x float> %1626, i64 1, !dbg !7
  %1629 = extractelement <8 x float> %1626, i64 2, !dbg !7
  %1630 = extractelement <8 x float> %1626, i64 3, !dbg !7
  %1631 = select i1 %.not, float %1628, float %1627, !dbg !7
  %1632 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1633 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1632), !dbg !7
  %1634 = add nuw nsw i32 %1633, 64, !dbg !7
  %1635 = xor i32 %1633, 16, !dbg !7
  %1636 = and i32 %1634, 192, !dbg !7
  %1637 = icmp samesign ult i32 %1635, %1636, !dbg !7
  %1638 = select i1 %1637, i32 %1635, i32 %1633, !dbg !7
  %1639 = shl nuw nsw i32 %1638, 2, !dbg !7
  %1640 = bitcast float %1631 to i32, !dbg !7
  %1641 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1640), !dbg !7
  %1642 = bitcast i32 %1641 to float, !dbg !7
  %1643 = select i1 %.not, float %1627, float %1642, !dbg !7
  %1644 = select i1 %.not, float %1642, float %1628, !dbg !7
  %1645 = select i1 %.not, float %1630, float %1629, !dbg !7
  %1646 = bitcast float %1645 to i32, !dbg !7
  %1647 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1646), !dbg !7
  %1648 = bitcast i32 %1647 to float, !dbg !7
  %1649 = select i1 %.not, float %1629, float %1648, !dbg !7
  %1650 = select i1 %.not, float %1648, float %1630, !dbg !7
  %1651 = select i1 %.not36, float %1649, float %1643, !dbg !7
  %1652 = xor i32 %1633, 32, !dbg !7
  %1653 = icmp samesign ult i32 %1652, %1636, !dbg !7
  %1654 = select i1 %1653, i32 %1652, i32 %1633, !dbg !7
  %1655 = shl nuw nsw i32 %1654, 2, !dbg !7
  %1656 = bitcast float %1651 to i32, !dbg !7
  %1657 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1656), !dbg !7
  %1658 = bitcast i32 %1657 to float, !dbg !7
  %1659 = select i1 %.not36, float %1643, float %1658, !dbg !7
  %1660 = select i1 %.not36, float %1658, float %1649, !dbg !7
  %1661 = select i1 %.not36, float %1650, float %1644, !dbg !7
  %1662 = bitcast float %1661 to i32, !dbg !7
  %1663 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1662), !dbg !7
  %1664 = bitcast i32 %1663 to float, !dbg !7
  %1665 = select i1 %.not36, float %1644, float %1664, !dbg !7
  %1666 = select i1 %.not36, float %1664, float %1650, !dbg !7
  %1667 = extractvalue [2 x <8 x float>] %210, 0, !dbg !7
  %1668 = extractelement <8 x float> %1667, i64 0, !dbg !7
  %1669 = extractelement <8 x float> %1667, i64 1, !dbg !7
  %1670 = extractelement <8 x float> %1667, i64 2, !dbg !7
  %1671 = extractelement <8 x float> %1667, i64 3, !dbg !7
  %1672 = select i1 %.not, float %1669, float %1668, !dbg !7
  %1673 = bitcast float %1672 to i32, !dbg !7
  %1674 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1673), !dbg !7
  %1675 = bitcast i32 %1674 to float, !dbg !7
  %1676 = select i1 %.not, float %1668, float %1675, !dbg !7
  %1677 = select i1 %.not, float %1675, float %1669, !dbg !7
  %1678 = select i1 %.not, float %1671, float %1670, !dbg !7
  %1679 = bitcast float %1678 to i32, !dbg !7
  %1680 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1679), !dbg !7
  %1681 = bitcast i32 %1680 to float, !dbg !7
  %1682 = select i1 %.not, float %1670, float %1681, !dbg !7
  %1683 = select i1 %.not, float %1681, float %1671, !dbg !7
  %1684 = select i1 %.not36, float %1682, float %1676, !dbg !7
  %1685 = bitcast float %1684 to i32, !dbg !7
  %1686 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1685), !dbg !7
  %1687 = bitcast i32 %1686 to float, !dbg !7
  %1688 = select i1 %.not36, float %1676, float %1687, !dbg !7
  %1689 = select i1 %.not36, float %1687, float %1682, !dbg !7
  %1690 = select i1 %.not36, float %1683, float %1677, !dbg !7
  %1691 = bitcast float %1690 to i32, !dbg !7
  %1692 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1691), !dbg !7
  %1693 = bitcast i32 %1692 to float, !dbg !7
  %1694 = select i1 %.not36, float %1677, float %1693, !dbg !7
  %1695 = select i1 %.not36, float %1693, float %1683, !dbg !7
  %1696 = extractvalue [2 x <8 x float>] %209, 0, !dbg !7
  %1697 = extractelement <8 x float> %1696, i64 0, !dbg !7
  %1698 = extractelement <8 x float> %1696, i64 1, !dbg !7
  %1699 = extractelement <8 x float> %1696, i64 2, !dbg !7
  %1700 = extractelement <8 x float> %1696, i64 3, !dbg !7
  %1701 = select i1 %.not, float %1698, float %1697, !dbg !7
  %1702 = bitcast float %1701 to i32, !dbg !7
  %1703 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1702), !dbg !7
  %1704 = bitcast i32 %1703 to float, !dbg !7
  %1705 = select i1 %.not, float %1697, float %1704, !dbg !7
  %1706 = select i1 %.not, float %1704, float %1698, !dbg !7
  %1707 = select i1 %.not, float %1700, float %1699, !dbg !7
  %1708 = bitcast float %1707 to i32, !dbg !7
  %1709 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1708), !dbg !7
  %1710 = bitcast i32 %1709 to float, !dbg !7
  %1711 = select i1 %.not, float %1699, float %1710, !dbg !7
  %1712 = select i1 %.not, float %1710, float %1700, !dbg !7
  %1713 = select i1 %.not36, float %1711, float %1705, !dbg !7
  %1714 = bitcast float %1713 to i32, !dbg !7
  %1715 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1714), !dbg !7
  %1716 = bitcast i32 %1715 to float, !dbg !7
  %1717 = select i1 %.not36, float %1705, float %1716, !dbg !7
  %1718 = select i1 %.not36, float %1716, float %1711, !dbg !7
  %1719 = select i1 %.not36, float %1712, float %1706, !dbg !7
  %1720 = bitcast float %1719 to i32, !dbg !7
  %1721 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1720), !dbg !7
  %1722 = bitcast i32 %1721 to float, !dbg !7
  %1723 = select i1 %.not36, float %1706, float %1722, !dbg !7
  %1724 = select i1 %.not36, float %1722, float %1712, !dbg !7
  %1725 = extractvalue [2 x <8 x float>] %206, 0, !dbg !7
  %1726 = extractelement <8 x float> %1725, i64 0, !dbg !7
  %1727 = extractelement <8 x float> %1725, i64 1, !dbg !7
  %1728 = extractelement <8 x float> %1725, i64 2, !dbg !7
  %1729 = extractelement <8 x float> %1725, i64 3, !dbg !7
  %1730 = select i1 %.not, float %1727, float %1726, !dbg !7
  %1731 = bitcast float %1730 to i32, !dbg !7
  %1732 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1731), !dbg !7
  %1733 = bitcast i32 %1732 to float, !dbg !7
  %1734 = select i1 %.not, float %1726, float %1733, !dbg !7
  %1735 = select i1 %.not, float %1733, float %1727, !dbg !7
  %1736 = select i1 %.not, float %1729, float %1728, !dbg !7
  %1737 = bitcast float %1736 to i32, !dbg !7
  %1738 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1737), !dbg !7
  %1739 = bitcast i32 %1738 to float, !dbg !7
  %1740 = select i1 %.not, float %1728, float %1739, !dbg !7
  %1741 = select i1 %.not, float %1739, float %1729, !dbg !7
  %1742 = select i1 %.not36, float %1740, float %1734, !dbg !7
  %1743 = bitcast float %1742 to i32, !dbg !7
  %1744 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1743), !dbg !7
  %1745 = select i1 %.not36, float %1741, float %1735, !dbg !7
  %1746 = bitcast float %1745 to i32, !dbg !7
  %1747 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1746), !dbg !7
  %1748 = extractvalue [2 x <1 x float>] %207, 0, !dbg !19
  %1749 = shufflevector <1 x float> %1748, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1750 = extractelement <8 x float> %1626, i64 4, !dbg !7
  %1751 = extractelement <8 x float> %1626, i64 5, !dbg !7
  %1752 = extractelement <8 x float> %1626, i64 6, !dbg !7
  %1753 = extractelement <8 x float> %1626, i64 7, !dbg !7
  %1754 = select i1 %.not, float %1751, float %1750, !dbg !7
  %1755 = bitcast float %1754 to i32, !dbg !7
  %1756 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1755), !dbg !7
  %1757 = bitcast i32 %1756 to float, !dbg !7
  %1758 = select i1 %.not, float %1750, float %1757, !dbg !7
  %1759 = select i1 %.not, float %1757, float %1751, !dbg !7
  %1760 = select i1 %.not, float %1753, float %1752, !dbg !7
  %1761 = bitcast float %1760 to i32, !dbg !7
  %1762 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1761), !dbg !7
  %1763 = bitcast i32 %1762 to float, !dbg !7
  %1764 = select i1 %.not, float %1752, float %1763, !dbg !7
  %1765 = select i1 %.not, float %1763, float %1753, !dbg !7
  %1766 = select i1 %.not36, float %1764, float %1758, !dbg !7
  %1767 = bitcast float %1766 to i32, !dbg !7
  %1768 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1767), !dbg !7
  %1769 = bitcast i32 %1768 to float, !dbg !7
  %1770 = select i1 %.not36, float %1758, float %1769, !dbg !7
  %1771 = select i1 %.not36, float %1769, float %1764, !dbg !7
  %1772 = select i1 %.not36, float %1765, float %1759, !dbg !7
  %1773 = bitcast float %1772 to i32, !dbg !7
  %1774 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1773), !dbg !7
  %1775 = bitcast i32 %1774 to float, !dbg !7
  %1776 = select i1 %.not36, float %1759, float %1775, !dbg !7
  %1777 = select i1 %.not36, float %1775, float %1765, !dbg !7
  %1778 = extractelement <8 x float> %1667, i64 4, !dbg !7
  %1779 = extractelement <8 x float> %1667, i64 5, !dbg !7
  %1780 = extractelement <8 x float> %1667, i64 6, !dbg !7
  %1781 = extractelement <8 x float> %1667, i64 7, !dbg !7
  %1782 = select i1 %.not, float %1779, float %1778, !dbg !7
  %1783 = bitcast float %1782 to i32, !dbg !7
  %1784 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1783), !dbg !7
  %1785 = bitcast i32 %1784 to float, !dbg !7
  %1786 = select i1 %.not, float %1778, float %1785, !dbg !7
  %1787 = select i1 %.not, float %1785, float %1779, !dbg !7
  %1788 = select i1 %.not, float %1781, float %1780, !dbg !7
  %1789 = bitcast float %1788 to i32, !dbg !7
  %1790 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1789), !dbg !7
  %1791 = bitcast i32 %1790 to float, !dbg !7
  %1792 = select i1 %.not, float %1780, float %1791, !dbg !7
  %1793 = select i1 %.not, float %1791, float %1781, !dbg !7
  %1794 = select i1 %.not36, float %1792, float %1786, !dbg !7
  %1795 = bitcast float %1794 to i32, !dbg !7
  %1796 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1795), !dbg !7
  %1797 = bitcast i32 %1796 to float, !dbg !7
  %1798 = select i1 %.not36, float %1786, float %1797, !dbg !7
  %1799 = select i1 %.not36, float %1797, float %1792, !dbg !7
  %1800 = select i1 %.not36, float %1793, float %1787, !dbg !7
  %1801 = bitcast float %1800 to i32, !dbg !7
  %1802 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1801), !dbg !7
  %1803 = bitcast i32 %1802 to float, !dbg !7
  %1804 = select i1 %.not36, float %1787, float %1803, !dbg !7
  %1805 = select i1 %.not36, float %1803, float %1793, !dbg !7
  %1806 = extractelement <8 x float> %1696, i64 4, !dbg !7
  %1807 = extractelement <8 x float> %1696, i64 5, !dbg !7
  %1808 = extractelement <8 x float> %1696, i64 6, !dbg !7
  %1809 = extractelement <8 x float> %1696, i64 7, !dbg !7
  %1810 = select i1 %.not, float %1807, float %1806, !dbg !7
  %1811 = bitcast float %1810 to i32, !dbg !7
  %1812 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1811), !dbg !7
  %1813 = bitcast i32 %1812 to float, !dbg !7
  %1814 = select i1 %.not, float %1806, float %1813, !dbg !7
  %1815 = select i1 %.not, float %1813, float %1807, !dbg !7
  %1816 = select i1 %.not, float %1809, float %1808, !dbg !7
  %1817 = bitcast float %1816 to i32, !dbg !7
  %1818 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1817), !dbg !7
  %1819 = bitcast i32 %1818 to float, !dbg !7
  %1820 = select i1 %.not, float %1808, float %1819, !dbg !7
  %1821 = select i1 %.not, float %1819, float %1809, !dbg !7
  %1822 = select i1 %.not36, float %1820, float %1814, !dbg !7
  %1823 = bitcast float %1822 to i32, !dbg !7
  %1824 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1823), !dbg !7
  %1825 = bitcast i32 %1824 to float, !dbg !7
  %1826 = select i1 %.not36, float %1814, float %1825, !dbg !7
  %1827 = select i1 %.not36, float %1825, float %1820, !dbg !7
  %1828 = select i1 %.not36, float %1821, float %1815, !dbg !7
  %1829 = bitcast float %1828 to i32, !dbg !7
  %1830 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1829), !dbg !7
  %1831 = bitcast i32 %1830 to float, !dbg !7
  %1832 = select i1 %.not36, float %1815, float %1831, !dbg !7
  %1833 = select i1 %.not36, float %1831, float %1821, !dbg !7
  %1834 = extractelement <8 x float> %1725, i64 4, !dbg !7
  %1835 = extractelement <8 x float> %1725, i64 5, !dbg !7
  %1836 = extractelement <8 x float> %1725, i64 6, !dbg !7
  %1837 = extractelement <8 x float> %1725, i64 7, !dbg !7
  %1838 = select i1 %.not, float %1835, float %1834, !dbg !7
  %1839 = bitcast float %1838 to i32, !dbg !7
  %1840 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1839), !dbg !7
  %1841 = bitcast i32 %1840 to float, !dbg !7
  %1842 = select i1 %.not, float %1834, float %1841, !dbg !7
  %1843 = select i1 %.not, float %1841, float %1835, !dbg !7
  %1844 = select i1 %.not, float %1837, float %1836, !dbg !7
  %1845 = bitcast float %1844 to i32, !dbg !7
  %1846 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1845), !dbg !7
  %1847 = bitcast i32 %1846 to float, !dbg !7
  %1848 = select i1 %.not, float %1836, float %1847, !dbg !7
  %1849 = select i1 %.not, float %1847, float %1837, !dbg !7
  %1850 = select i1 %.not36, float %1848, float %1842, !dbg !7
  %1851 = bitcast float %1850 to i32, !dbg !7
  %1852 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1851), !dbg !7
  %1853 = select i1 %.not36, float %1849, float %1843, !dbg !7
  %1854 = bitcast float %1853 to i32, !dbg !7
  %1855 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1854), !dbg !7
  %1856 = extractvalue [2 x <8 x float>] %211, 1, !dbg !7
  %1857 = extractelement <8 x float> %1856, i64 0, !dbg !7
  %1858 = extractelement <8 x float> %1856, i64 1, !dbg !7
  %1859 = extractelement <8 x float> %1856, i64 2, !dbg !7
  %1860 = extractelement <8 x float> %1856, i64 3, !dbg !7
  %1861 = select i1 %.not, float %1858, float %1857, !dbg !7
  %1862 = bitcast float %1861 to i32, !dbg !7
  %1863 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1862), !dbg !7
  %1864 = bitcast i32 %1863 to float, !dbg !7
  %1865 = select i1 %.not, float %1857, float %1864, !dbg !7
  %1866 = select i1 %.not, float %1864, float %1858, !dbg !7
  %1867 = select i1 %.not, float %1860, float %1859, !dbg !7
  %1868 = bitcast float %1867 to i32, !dbg !7
  %1869 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1868), !dbg !7
  %1870 = bitcast i32 %1869 to float, !dbg !7
  %1871 = select i1 %.not, float %1859, float %1870, !dbg !7
  %1872 = select i1 %.not, float %1870, float %1860, !dbg !7
  %1873 = select i1 %.not36, float %1871, float %1865, !dbg !7
  %1874 = bitcast float %1873 to i32, !dbg !7
  %1875 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1874), !dbg !7
  %1876 = bitcast i32 %1875 to float, !dbg !7
  %1877 = select i1 %.not36, float %1865, float %1876, !dbg !7
  %1878 = select i1 %.not36, float %1876, float %1871, !dbg !7
  %1879 = select i1 %.not36, float %1872, float %1866, !dbg !7
  %1880 = bitcast float %1879 to i32, !dbg !7
  %1881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1880), !dbg !7
  %1882 = bitcast i32 %1881 to float, !dbg !7
  %1883 = select i1 %.not36, float %1866, float %1882, !dbg !7
  %1884 = select i1 %.not36, float %1882, float %1872, !dbg !7
  %1885 = extractvalue [2 x <8 x float>] %210, 1, !dbg !7
  %1886 = extractelement <8 x float> %1885, i64 0, !dbg !7
  %1887 = extractelement <8 x float> %1885, i64 1, !dbg !7
  %1888 = extractelement <8 x float> %1885, i64 2, !dbg !7
  %1889 = extractelement <8 x float> %1885, i64 3, !dbg !7
  %1890 = select i1 %.not, float %1887, float %1886, !dbg !7
  %1891 = bitcast float %1890 to i32, !dbg !7
  %1892 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1891), !dbg !7
  %1893 = bitcast i32 %1892 to float, !dbg !7
  %1894 = select i1 %.not, float %1886, float %1893, !dbg !7
  %1895 = select i1 %.not, float %1893, float %1887, !dbg !7
  %1896 = select i1 %.not, float %1889, float %1888, !dbg !7
  %1897 = bitcast float %1896 to i32, !dbg !7
  %1898 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1897), !dbg !7
  %1899 = bitcast i32 %1898 to float, !dbg !7
  %1900 = select i1 %.not, float %1888, float %1899, !dbg !7
  %1901 = select i1 %.not, float %1899, float %1889, !dbg !7
  %1902 = select i1 %.not36, float %1900, float %1894, !dbg !7
  %1903 = bitcast float %1902 to i32, !dbg !7
  %1904 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1903), !dbg !7
  %1905 = bitcast i32 %1904 to float, !dbg !7
  %1906 = select i1 %.not36, float %1894, float %1905, !dbg !7
  %1907 = select i1 %.not36, float %1905, float %1900, !dbg !7
  %1908 = select i1 %.not36, float %1901, float %1895, !dbg !7
  %1909 = bitcast float %1908 to i32, !dbg !7
  %1910 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1909), !dbg !7
  %1911 = bitcast i32 %1910 to float, !dbg !7
  %1912 = select i1 %.not36, float %1895, float %1911, !dbg !7
  %1913 = select i1 %.not36, float %1911, float %1901, !dbg !7
  %1914 = extractvalue [2 x <8 x float>] %209, 1, !dbg !7
  %1915 = extractelement <8 x float> %1914, i64 0, !dbg !7
  %1916 = extractelement <8 x float> %1914, i64 1, !dbg !7
  %1917 = extractelement <8 x float> %1914, i64 2, !dbg !7
  %1918 = extractelement <8 x float> %1914, i64 3, !dbg !7
  %1919 = select i1 %.not, float %1916, float %1915, !dbg !7
  %1920 = bitcast float %1919 to i32, !dbg !7
  %1921 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1920), !dbg !7
  %1922 = bitcast i32 %1921 to float, !dbg !7
  %1923 = select i1 %.not, float %1915, float %1922, !dbg !7
  %1924 = select i1 %.not, float %1922, float %1916, !dbg !7
  %1925 = select i1 %.not, float %1918, float %1917, !dbg !7
  %1926 = bitcast float %1925 to i32, !dbg !7
  %1927 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1926), !dbg !7
  %1928 = bitcast i32 %1927 to float, !dbg !7
  %1929 = select i1 %.not, float %1917, float %1928, !dbg !7
  %1930 = select i1 %.not, float %1928, float %1918, !dbg !7
  %1931 = select i1 %.not36, float %1929, float %1923, !dbg !7
  %1932 = bitcast float %1931 to i32, !dbg !7
  %1933 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1932), !dbg !7
  %1934 = bitcast i32 %1933 to float, !dbg !7
  %1935 = select i1 %.not36, float %1923, float %1934, !dbg !7
  %1936 = select i1 %.not36, float %1934, float %1929, !dbg !7
  %1937 = select i1 %.not36, float %1930, float %1924, !dbg !7
  %1938 = bitcast float %1937 to i32, !dbg !7
  %1939 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1938), !dbg !7
  %1940 = bitcast i32 %1939 to float, !dbg !7
  %1941 = select i1 %.not36, float %1924, float %1940, !dbg !7
  %1942 = select i1 %.not36, float %1940, float %1930, !dbg !7
  %1943 = extractvalue [2 x <8 x float>] %206, 1, !dbg !7
  %1944 = extractelement <8 x float> %1943, i64 0, !dbg !7
  %1945 = extractelement <8 x float> %1943, i64 1, !dbg !7
  %1946 = extractelement <8 x float> %1943, i64 2, !dbg !7
  %1947 = extractelement <8 x float> %1943, i64 3, !dbg !7
  %1948 = select i1 %.not, float %1945, float %1944, !dbg !7
  %1949 = bitcast float %1948 to i32, !dbg !7
  %1950 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1949), !dbg !7
  %1951 = bitcast i32 %1950 to float, !dbg !7
  %1952 = select i1 %.not, float %1944, float %1951, !dbg !7
  %1953 = select i1 %.not, float %1951, float %1945, !dbg !7
  %1954 = select i1 %.not, float %1947, float %1946, !dbg !7
  %1955 = bitcast float %1954 to i32, !dbg !7
  %1956 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1955), !dbg !7
  %1957 = bitcast i32 %1956 to float, !dbg !7
  %1958 = select i1 %.not, float %1946, float %1957, !dbg !7
  %1959 = select i1 %.not, float %1957, float %1947, !dbg !7
  %1960 = select i1 %.not36, float %1958, float %1952, !dbg !7
  %1961 = bitcast float %1960 to i32, !dbg !7
  %1962 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1961), !dbg !7
  %1963 = select i1 %.not36, float %1959, float %1953, !dbg !7
  %1964 = bitcast float %1963 to i32, !dbg !7
  %1965 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1964), !dbg !7
  %1966 = extractvalue [2 x <1 x float>] %207, 1, !dbg !19
  %1967 = shufflevector <1 x float> %1966, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1968 = extractelement <8 x float> %1856, i64 4, !dbg !7
  %1969 = extractelement <8 x float> %1856, i64 5, !dbg !7
  %1970 = extractelement <8 x float> %1856, i64 6, !dbg !7
  %1971 = extractelement <8 x float> %1856, i64 7, !dbg !7
  %1972 = select i1 %.not, float %1969, float %1968, !dbg !7
  %1973 = bitcast float %1972 to i32, !dbg !7
  %1974 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1973), !dbg !7
  %1975 = bitcast i32 %1974 to float, !dbg !7
  %1976 = select i1 %.not, float %1968, float %1975, !dbg !7
  %1977 = select i1 %.not, float %1975, float %1969, !dbg !7
  %1978 = select i1 %.not, float %1971, float %1970, !dbg !7
  %1979 = bitcast float %1978 to i32, !dbg !7
  %1980 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %1979), !dbg !7
  %1981 = bitcast i32 %1980 to float, !dbg !7
  %1982 = select i1 %.not, float %1970, float %1981, !dbg !7
  %1983 = select i1 %.not, float %1981, float %1971, !dbg !7
  %1984 = select i1 %.not36, float %1982, float %1976, !dbg !7
  %1985 = bitcast float %1984 to i32, !dbg !7
  %1986 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1985), !dbg !7
  %1987 = bitcast i32 %1986 to float, !dbg !7
  %1988 = select i1 %.not36, float %1976, float %1987, !dbg !7
  %1989 = select i1 %.not36, float %1987, float %1982, !dbg !7
  %1990 = select i1 %.not36, float %1983, float %1977, !dbg !7
  %1991 = bitcast float %1990 to i32, !dbg !7
  %1992 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %1991), !dbg !7
  %1993 = bitcast i32 %1992 to float, !dbg !7
  %1994 = select i1 %.not36, float %1977, float %1993, !dbg !7
  %1995 = select i1 %.not36, float %1993, float %1983, !dbg !7
  %1996 = extractelement <8 x float> %1885, i64 4, !dbg !7
  %1997 = extractelement <8 x float> %1885, i64 5, !dbg !7
  %1998 = extractelement <8 x float> %1885, i64 6, !dbg !7
  %1999 = extractelement <8 x float> %1885, i64 7, !dbg !7
  %2000 = select i1 %.not, float %1997, float %1996, !dbg !7
  %2001 = bitcast float %2000 to i32, !dbg !7
  %2002 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %2001), !dbg !7
  %2003 = bitcast i32 %2002 to float, !dbg !7
  %2004 = select i1 %.not, float %1996, float %2003, !dbg !7
  %2005 = select i1 %.not, float %2003, float %1997, !dbg !7
  %2006 = select i1 %.not, float %1999, float %1998, !dbg !7
  %2007 = bitcast float %2006 to i32, !dbg !7
  %2008 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %2007), !dbg !7
  %2009 = bitcast i32 %2008 to float, !dbg !7
  %2010 = select i1 %.not, float %1998, float %2009, !dbg !7
  %2011 = select i1 %.not, float %2009, float %1999, !dbg !7
  %2012 = select i1 %.not36, float %2010, float %2004, !dbg !7
  %2013 = bitcast float %2012 to i32, !dbg !7
  %2014 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %2013), !dbg !7
  %2015 = bitcast i32 %2014 to float, !dbg !7
  %2016 = select i1 %.not36, float %2004, float %2015, !dbg !7
  %2017 = select i1 %.not36, float %2015, float %2010, !dbg !7
  %2018 = select i1 %.not36, float %2011, float %2005, !dbg !7
  %2019 = bitcast float %2018 to i32, !dbg !7
  %2020 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %2019), !dbg !7
  %2021 = bitcast i32 %2020 to float, !dbg !7
  %2022 = select i1 %.not36, float %2005, float %2021, !dbg !7
  %2023 = select i1 %.not36, float %2021, float %2011, !dbg !7
  %2024 = extractelement <8 x float> %1914, i64 4, !dbg !7
  %2025 = extractelement <8 x float> %1914, i64 5, !dbg !7
  %2026 = extractelement <8 x float> %1914, i64 6, !dbg !7
  %2027 = extractelement <8 x float> %1914, i64 7, !dbg !7
  %2028 = select i1 %.not, float %2025, float %2024, !dbg !7
  %2029 = bitcast float %2028 to i32, !dbg !7
  %2030 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %2029), !dbg !7
  %2031 = bitcast i32 %2030 to float, !dbg !7
  %2032 = select i1 %.not, float %2024, float %2031, !dbg !7
  %2033 = select i1 %.not, float %2031, float %2025, !dbg !7
  %2034 = select i1 %.not, float %2027, float %2026, !dbg !7
  %2035 = bitcast float %2034 to i32, !dbg !7
  %2036 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %2035), !dbg !7
  %2037 = bitcast i32 %2036 to float, !dbg !7
  %2038 = select i1 %.not, float %2026, float %2037, !dbg !7
  %2039 = select i1 %.not, float %2037, float %2027, !dbg !7
  %2040 = select i1 %.not36, float %2038, float %2032, !dbg !7
  %2041 = bitcast float %2040 to i32, !dbg !7
  %2042 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %2041), !dbg !7
  %2043 = bitcast i32 %2042 to float, !dbg !7
  %2044 = select i1 %.not36, float %2032, float %2043, !dbg !7
  %2045 = select i1 %.not36, float %2043, float %2038, !dbg !7
  %2046 = select i1 %.not36, float %2039, float %2033, !dbg !7
  %2047 = bitcast float %2046 to i32, !dbg !7
  %2048 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %2047), !dbg !7
  %2049 = bitcast i32 %2048 to float, !dbg !7
  %2050 = select i1 %.not36, float %2033, float %2049, !dbg !7
  %2051 = select i1 %.not36, float %2049, float %2039, !dbg !7
  %2052 = extractelement <8 x float> %1943, i64 4, !dbg !7
  %2053 = extractelement <8 x float> %1943, i64 5, !dbg !7
  %2054 = extractelement <8 x float> %1943, i64 6, !dbg !7
  %2055 = extractelement <8 x float> %1943, i64 7, !dbg !7
  %2056 = select i1 %.not, float %2053, float %2052, !dbg !7
  %2057 = bitcast float %2056 to i32, !dbg !7
  %2058 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %2057), !dbg !7
  %2059 = bitcast i32 %2058 to float, !dbg !7
  %2060 = select i1 %.not, float %2052, float %2059, !dbg !7
  %2061 = select i1 %.not, float %2059, float %2053, !dbg !7
  %2062 = select i1 %.not, float %2055, float %2054, !dbg !7
  %2063 = bitcast float %2062 to i32, !dbg !7
  %2064 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1639, i32 %2063), !dbg !7
  %2065 = bitcast i32 %2064 to float, !dbg !7
  %2066 = select i1 %.not, float %2054, float %2065, !dbg !7
  %2067 = select i1 %.not, float %2065, float %2055, !dbg !7
  %2068 = select i1 %.not36, float %2066, float %2060, !dbg !7
  %2069 = bitcast float %2068 to i32, !dbg !7
  %2070 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %2069), !dbg !7
  %2071 = select i1 %.not36, float %2067, float %2061, !dbg !7
  %2072 = bitcast float %2071 to i32, !dbg !7
  %2073 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1655, i32 %2072), !dbg !7
  br i1 true, label %2074, label %2138, !dbg !21

2074:                                             ; preds = %1623
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
  %2090 = bitcast i32 %1962 to float, !dbg !7
  %2091 = select i1 %.not36, float %1952, float %2090, !dbg !7
  %2092 = insertelement <4 x float> poison, float %2091, i64 0, !dbg !19
  %2093 = bitcast i32 %1965 to float, !dbg !7
  %2094 = select i1 %.not36, float %1953, float %2093, !dbg !7
  %2095 = insertelement <4 x float> %2092, float %2094, i64 1, !dbg !19
  %2096 = select i1 %.not36, float %2090, float %1958, !dbg !7
  %2097 = insertelement <4 x float> %2095, float %2096, i64 2, !dbg !19
  %2098 = select i1 %.not36, float %2093, float %1959, !dbg !7
  %2099 = insertelement <4 x float> %2097, float %2098, i64 3, !dbg !19
  %normalized104 = fmul <4 x float> %2099, %row.reciprocal, !dbg !19
  %2100 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !20
  %2101 = extractelement <4 x half> %2100, i64 3, !dbg !20
  %2102 = extractelement <4 x half> %2100, i64 2, !dbg !20
  %2103 = extractelement <4 x half> %2100, i64 1, !dbg !20
  %2104 = extractelement <4 x half> %2100, i64 0, !dbg !20
  %2105 = bitcast i32 %1852 to float, !dbg !7
  %2106 = select i1 %.not36, float %1842, float %2105, !dbg !7
  %2107 = insertelement <4 x float> poison, float %2106, i64 0, !dbg !19
  %2108 = bitcast i32 %1855 to float, !dbg !7
  %2109 = select i1 %.not36, float %1843, float %2108, !dbg !7
  %2110 = insertelement <4 x float> %2107, float %2109, i64 1, !dbg !19
  %2111 = select i1 %.not36, float %2105, float %1848, !dbg !7
  %2112 = insertelement <4 x float> %2110, float %2111, i64 2, !dbg !19
  %2113 = select i1 %.not36, float %2108, float %1849, !dbg !7
  %2114 = insertelement <4 x float> %2112, float %2113, i64 3, !dbg !19
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1749, !dbg !19
  %normalized106 = fmul <4 x float> %2114, %row.reciprocal105, !dbg !19
  %2115 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !20
  %2116 = extractelement <4 x half> %2115, i64 3, !dbg !20
  %2117 = extractelement <4 x half> %2115, i64 2, !dbg !20
  %2118 = extractelement <4 x half> %2115, i64 1, !dbg !20
  %2119 = extractelement <4 x half> %2115, i64 0, !dbg !20
  %2120 = bitcast i32 %1744 to float, !dbg !7
  %2121 = select i1 %.not36, float %1734, float %2120, !dbg !7
  %2122 = insertelement <4 x float> poison, float %2121, i64 0, !dbg !19
  %2123 = bitcast i32 %1747 to float, !dbg !7
  %2124 = select i1 %.not36, float %1735, float %2123, !dbg !7
  %2125 = insertelement <4 x float> %2122, float %2124, i64 1, !dbg !19
  %2126 = select i1 %.not36, float %2120, float %1740, !dbg !7
  %2127 = insertelement <4 x float> %2125, float %2126, i64 2, !dbg !19
  %2128 = select i1 %.not36, float %2123, float %1741, !dbg !7
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
  %address.thread1090 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1091 = zext nneg i32 %address.thread1090 to i64, !dbg !8
  %address.local1092 = add i64 %25, %address.local1091, !dbg !21
  %address.local1093 = lshr i64 %address.local1091, 6, !dbg !11
  %address.local1094 = shl nuw nsw i64 %address.local1093, 4, !dbg !21
  %address.local1095 = add i64 %address.local1092, %address.local1094, !dbg !21
  %address.local1096 = and i64 %address.local1091, 240, !dbg !21
  %address.local1097 = sub i64 %address.local1095, %address.local1096, !dbg !21
  %address.local1098 = add i64 %address.local1097, %26, !dbg !21
  %address.local1099 = shl nuw nsw i64 %address.local1098, 7, !dbg !21
  %address.local1100 = add nuw nsw i64 %2137, %address.local1099, !dbg !21
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1102 = lshr i64 %address.local1091, 2, !dbg !11
  %address.local1103 = and i64 %address.local1102, 12, !dbg !11
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1103, !dbg !21
  store half %2134, ptr addrspace(1) %address.local1104, align 2, !dbg !21
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1106 = or disjoint i64 %address.local1103, 1, !dbg !21
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1105, i64 %address.local1106, !dbg !21
  store half %2133, ptr addrspace(1) %address.local1107, align 2, !dbg !21
  %address.local1108 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1109 = or disjoint i64 %address.local1103, 2, !dbg !21
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1108, i64 %address.local1109, !dbg !21
  store half %2132, ptr addrspace(1) %address.local1110, align 2, !dbg !21
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1112 = or disjoint i64 %address.local1103, 3, !dbg !21
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1112, !dbg !21
  store half %2131, ptr addrspace(1) %address.local1113, align 2, !dbg !21
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1115 = or disjoint i64 %address.local1103, 16, !dbg !21
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1114, i64 %address.local1115, !dbg !21
  store half %2119, ptr addrspace(1) %address.local1116, align 2, !dbg !21
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1118 = or disjoint i64 %address.local1103, 17, !dbg !21
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1118, !dbg !21
  store half %2118, ptr addrspace(1) %address.local1119, align 2, !dbg !21
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1121 = or disjoint i64 %address.local1103, 18, !dbg !21
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1120, i64 %address.local1121, !dbg !21
  store half %2117, ptr addrspace(1) %address.local1122, align 2, !dbg !21
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1100, !dbg !21
  %address.local1124 = or disjoint i64 %address.local1103, 19, !dbg !21
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1124, !dbg !21
  store half %2116, ptr addrspace(1) %address.local1125, align 2, !dbg !21
  %address.local1126 = shl i64 %address.local1098, 7, !dbg !21
  %address.local1127 = add i64 %address.local1126, 8192, !dbg !21
  %address.local1128 = add nuw nsw i64 %2137, %address.local1127, !dbg !21
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1103, !dbg !21
  store half %2104, ptr addrspace(1) %address.local1130, align 2, !dbg !21
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1131, i64 %address.local1106, !dbg !21
  store half %2103, ptr addrspace(1) %address.local1132, align 2, !dbg !21
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1133, i64 %address.local1109, !dbg !21
  store half %2102, ptr addrspace(1) %address.local1134, align 2, !dbg !21
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1112, !dbg !21
  store half %2101, ptr addrspace(1) %address.local1136, align 2, !dbg !21
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1137, i64 %address.local1115, !dbg !21
  store half %2089, ptr addrspace(1) %address.local1138, align 2, !dbg !21
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1118, !dbg !21
  store half %2088, ptr addrspace(1) %address.local1140, align 2, !dbg !21
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1121, !dbg !21
  store half %2087, ptr addrspace(1) %address.local1142, align 2, !dbg !21
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1128, !dbg !21
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1124, !dbg !21
  store half %2086, ptr addrspace(1) %address.local1144, align 2, !dbg !21
  br label %2138, !dbg !21

2138:                                             ; preds = %2074, %1623
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
  %2149 = insertelement <4 x float> poison, float %1935, i64 0, !dbg !19
  %2150 = insertelement <4 x float> %2149, float %1941, i64 1, !dbg !19
  %2151 = insertelement <4 x float> %2150, float %1936, i64 2, !dbg !19
  %2152 = insertelement <4 x float> %2151, float %1942, i64 3, !dbg !19
  %normalized110 = fmul <4 x float> %2152, %row.reciprocal108, !dbg !19
  %2153 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %2154 = extractelement <4 x half> %2153, i64 3, !dbg !20
  %2155 = extractelement <4 x half> %2153, i64 2, !dbg !20
  %2156 = extractelement <4 x half> %2153, i64 1, !dbg !20
  %2157 = extractelement <4 x half> %2153, i64 0, !dbg !20
  %2158 = insertelement <4 x float> poison, float %1826, i64 0, !dbg !19
  %2159 = insertelement <4 x float> %2158, float %1832, i64 1, !dbg !19
  %2160 = insertelement <4 x float> %2159, float %1827, i64 2, !dbg !19
  %2161 = insertelement <4 x float> %2160, float %1833, i64 3, !dbg !19
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1749, !dbg !19
  %normalized112 = fmul <4 x float> %2161, %row.reciprocal111, !dbg !19
  %2162 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !20
  %2163 = extractelement <4 x half> %2162, i64 3, !dbg !20
  %2164 = extractelement <4 x half> %2162, i64 2, !dbg !20
  %2165 = extractelement <4 x half> %2162, i64 1, !dbg !20
  %2166 = extractelement <4 x half> %2162, i64 0, !dbg !20
  %2167 = insertelement <4 x float> poison, float %1717, i64 0, !dbg !19
  %2168 = insertelement <4 x float> %2167, float %1723, i64 1, !dbg !19
  %2169 = insertelement <4 x float> %2168, float %1718, i64 2, !dbg !19
  %2170 = insertelement <4 x float> %2169, float %1724, i64 3, !dbg !19
  %normalized113 = fmul <4 x float> %2170, %row.reciprocal111, !dbg !19
  %2171 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %2172 = extractelement <4 x half> %2171, i64 3, !dbg !20
  %2173 = extractelement <4 x half> %2171, i64 2, !dbg !20
  %2174 = extractelement <4 x half> %2171, i64 1, !dbg !20
  %2175 = extractelement <4 x half> %2171, i64 0, !dbg !20
  %2176 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1145 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1146 = zext nneg i32 %address.thread1145 to i64, !dbg !8
  %address.local1147 = add i64 %25, %address.local1146, !dbg !21
  %address.local1148 = lshr i64 %address.local1146, 6, !dbg !11
  %address.local1149 = shl nuw nsw i64 %address.local1148, 4, !dbg !21
  %address.local1150 = add i64 %address.local1147, %address.local1149, !dbg !21
  %address.local1151 = and i64 %address.local1146, 240, !dbg !21
  %address.local1152 = sub i64 %address.local1150, %address.local1151, !dbg !21
  %address.local1153 = add i64 %address.local1152, %26, !dbg !21
  %address.local1154 = shl nuw nsw i64 %address.local1153, 7, !dbg !21
  %address.local1155 = add nuw nsw i64 %2176, %address.local1154, !dbg !21
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1157 = lshr i64 %address.local1146, 2, !dbg !11
  %address.local1158 = and i64 %address.local1157, 12, !dbg !11
  %address.local1159 = or disjoint i64 %address.local1158, 32, !dbg !21
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1156, i64 %address.local1159, !dbg !21
  store half %2175, ptr addrspace(1) %address.local1160, align 2, !dbg !21
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1162 = or disjoint i64 %address.local1158, 33, !dbg !21
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1161, i64 %address.local1162, !dbg !21
  store half %2174, ptr addrspace(1) %address.local1163, align 2, !dbg !21
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1165 = or disjoint i64 %address.local1158, 34, !dbg !21
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1164, i64 %address.local1165, !dbg !21
  store half %2173, ptr addrspace(1) %address.local1166, align 2, !dbg !21
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1168 = or disjoint i64 %address.local1158, 35, !dbg !21
  %address.local1169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1167, i64 %address.local1168, !dbg !21
  store half %2172, ptr addrspace(1) %address.local1169, align 2, !dbg !21
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1171 = or disjoint i64 %address.local1158, 48, !dbg !21
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1170, i64 %address.local1171, !dbg !21
  store half %2166, ptr addrspace(1) %address.local1172, align 2, !dbg !21
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1174 = or disjoint i64 %address.local1158, 49, !dbg !21
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1173, i64 %address.local1174, !dbg !21
  store half %2165, ptr addrspace(1) %address.local1175, align 2, !dbg !21
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1177 = or disjoint i64 %address.local1158, 50, !dbg !21
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1176, i64 %address.local1177, !dbg !21
  store half %2164, ptr addrspace(1) %address.local1178, align 2, !dbg !21
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1155, !dbg !21
  %address.local1180 = or i64 %address.local1157, 51, !dbg !21
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1179, i64 %address.local1180, !dbg !21
  store half %2163, ptr addrspace(1) %address.local1181, align 2, !dbg !21
  %address.local1182 = shl i64 %address.local1153, 7, !dbg !21
  %address.local1183 = add i64 %address.local1182, 8192, !dbg !21
  %address.local1184 = add nuw nsw i64 %2176, %address.local1183, !dbg !21
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1159, !dbg !21
  store half %2157, ptr addrspace(1) %address.local1186, align 2, !dbg !21
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1187, i64 %address.local1162, !dbg !21
  store half %2156, ptr addrspace(1) %address.local1188, align 2, !dbg !21
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1189, i64 %address.local1165, !dbg !21
  store half %2155, ptr addrspace(1) %address.local1190, align 2, !dbg !21
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1191, i64 %address.local1168, !dbg !21
  store half %2154, ptr addrspace(1) %address.local1192, align 2, !dbg !21
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1193, i64 %address.local1171, !dbg !21
  store half %2148, ptr addrspace(1) %address.local1194, align 2, !dbg !21
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1195, i64 %address.local1174, !dbg !21
  store half %2147, ptr addrspace(1) %address.local1196, align 2, !dbg !21
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1197, i64 %address.local1177, !dbg !21
  store half %2146, ptr addrspace(1) %address.local1198, align 2, !dbg !21
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1184, !dbg !21
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1199, i64 %address.local1180, !dbg !21
  store half %2145, ptr addrspace(1) %address.local1200, align 2, !dbg !21
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
  %2188 = insertelement <4 x float> poison, float %1906, i64 0, !dbg !19
  %2189 = insertelement <4 x float> %2188, float %1912, i64 1, !dbg !19
  %2190 = insertelement <4 x float> %2189, float %1907, i64 2, !dbg !19
  %2191 = insertelement <4 x float> %2190, float %1913, i64 3, !dbg !19
  %normalized116 = fmul <4 x float> %2191, %row.reciprocal114, !dbg !19
  %2192 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %2193 = extractelement <4 x half> %2192, i64 3, !dbg !20
  %2194 = extractelement <4 x half> %2192, i64 2, !dbg !20
  %2195 = extractelement <4 x half> %2192, i64 1, !dbg !20
  %2196 = extractelement <4 x half> %2192, i64 0, !dbg !20
  %2197 = insertelement <4 x float> poison, float %1798, i64 0, !dbg !19
  %2198 = insertelement <4 x float> %2197, float %1804, i64 1, !dbg !19
  %2199 = insertelement <4 x float> %2198, float %1799, i64 2, !dbg !19
  %2200 = insertelement <4 x float> %2199, float %1805, i64 3, !dbg !19
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1749, !dbg !19
  %normalized118 = fmul <4 x float> %2200, %row.reciprocal117, !dbg !19
  %2201 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !20
  %2202 = extractelement <4 x half> %2201, i64 3, !dbg !20
  %2203 = extractelement <4 x half> %2201, i64 2, !dbg !20
  %2204 = extractelement <4 x half> %2201, i64 1, !dbg !20
  %2205 = extractelement <4 x half> %2201, i64 0, !dbg !20
  %2206 = insertelement <4 x float> poison, float %1688, i64 0, !dbg !19
  %2207 = insertelement <4 x float> %2206, float %1694, i64 1, !dbg !19
  %2208 = insertelement <4 x float> %2207, float %1689, i64 2, !dbg !19
  %2209 = insertelement <4 x float> %2208, float %1695, i64 3, !dbg !19
  %normalized119 = fmul <4 x float> %2209, %row.reciprocal117, !dbg !19
  %2210 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %2211 = extractelement <4 x half> %2210, i64 3, !dbg !20
  %2212 = extractelement <4 x half> %2210, i64 2, !dbg !20
  %2213 = extractelement <4 x half> %2210, i64 1, !dbg !20
  %2214 = extractelement <4 x half> %2210, i64 0, !dbg !20
  %2215 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1201 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1202 = zext nneg i32 %address.thread1201 to i64, !dbg !8
  %address.local1203 = add i64 %25, %address.local1202, !dbg !21
  %address.local1204 = lshr i64 %address.local1202, 6, !dbg !11
  %address.local1205 = shl nuw nsw i64 %address.local1204, 4, !dbg !21
  %address.local1206 = add i64 %address.local1203, %address.local1205, !dbg !21
  %address.local1207 = and i64 %address.local1202, 240, !dbg !21
  %address.local1208 = sub i64 %address.local1206, %address.local1207, !dbg !21
  %address.local1209 = add i64 %address.local1208, %26, !dbg !21
  %address.local1210 = shl nuw nsw i64 %address.local1209, 7, !dbg !21
  %address.local1211 = add nuw nsw i64 %2215, %address.local1210, !dbg !21
  %address.local1212 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1213 = lshr i64 %address.local1202, 2, !dbg !11
  %address.local1214 = and i64 %address.local1213, 12, !dbg !11
  %address.local1215 = or disjoint i64 %address.local1214, 64, !dbg !21
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1212, i64 %address.local1215, !dbg !21
  store half %2214, ptr addrspace(1) %address.local1216, align 2, !dbg !21
  %address.local1217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1218 = or disjoint i64 %address.local1214, 65, !dbg !21
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1217, i64 %address.local1218, !dbg !21
  store half %2213, ptr addrspace(1) %address.local1219, align 2, !dbg !21
  %address.local1220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1221 = or disjoint i64 %address.local1214, 66, !dbg !21
  %address.local1222 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1220, i64 %address.local1221, !dbg !21
  store half %2212, ptr addrspace(1) %address.local1222, align 2, !dbg !21
  %address.local1223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1224 = or disjoint i64 %address.local1214, 67, !dbg !21
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1223, i64 %address.local1224, !dbg !21
  store half %2211, ptr addrspace(1) %address.local1225, align 2, !dbg !21
  %address.local1226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1227 = or disjoint i64 %address.local1214, 80, !dbg !21
  %address.local1228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1226, i64 %address.local1227, !dbg !21
  store half %2205, ptr addrspace(1) %address.local1228, align 2, !dbg !21
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1230 = or disjoint i64 %address.local1214, 81, !dbg !21
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1229, i64 %address.local1230, !dbg !21
  store half %2204, ptr addrspace(1) %address.local1231, align 2, !dbg !21
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1233 = or disjoint i64 %address.local1214, 82, !dbg !21
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1232, i64 %address.local1233, !dbg !21
  store half %2203, ptr addrspace(1) %address.local1234, align 2, !dbg !21
  %address.local1235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1211, !dbg !21
  %address.local1236 = or disjoint i64 %address.local1214, 83, !dbg !21
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1235, i64 %address.local1236, !dbg !21
  store half %2202, ptr addrspace(1) %address.local1237, align 2, !dbg !21
  %address.local1238 = shl i64 %address.local1209, 7, !dbg !21
  %address.local1239 = add i64 %address.local1238, 8192, !dbg !21
  %address.local1240 = add nuw nsw i64 %2215, %address.local1239, !dbg !21
  %address.local1241 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1241, i64 %address.local1215, !dbg !21
  store half %2196, ptr addrspace(1) %address.local1242, align 2, !dbg !21
  %address.local1243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1244 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1243, i64 %address.local1218, !dbg !21
  store half %2195, ptr addrspace(1) %address.local1244, align 2, !dbg !21
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1246 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1245, i64 %address.local1221, !dbg !21
  store half %2194, ptr addrspace(1) %address.local1246, align 2, !dbg !21
  %address.local1247 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1247, i64 %address.local1224, !dbg !21
  store half %2193, ptr addrspace(1) %address.local1248, align 2, !dbg !21
  %address.local1249 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1250 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1249, i64 %address.local1227, !dbg !21
  store half %2187, ptr addrspace(1) %address.local1250, align 2, !dbg !21
  %address.local1251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1252 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1251, i64 %address.local1230, !dbg !21
  store half %2186, ptr addrspace(1) %address.local1252, align 2, !dbg !21
  %address.local1253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1253, i64 %address.local1233, !dbg !21
  store half %2185, ptr addrspace(1) %address.local1254, align 2, !dbg !21
  %address.local1255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1240, !dbg !21
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1255, i64 %address.local1236, !dbg !21
  store half %2184, ptr addrspace(1) %address.local1256, align 2, !dbg !21
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
  %2227 = insertelement <4 x float> poison, float %1877, i64 0, !dbg !19
  %2228 = insertelement <4 x float> %2227, float %1883, i64 1, !dbg !19
  %2229 = insertelement <4 x float> %2228, float %1878, i64 2, !dbg !19
  %2230 = insertelement <4 x float> %2229, float %1884, i64 3, !dbg !19
  %normalized122 = fmul <4 x float> %2230, %row.reciprocal120, !dbg !19
  %2231 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %2232 = extractelement <4 x half> %2231, i64 3, !dbg !20
  %2233 = extractelement <4 x half> %2231, i64 2, !dbg !20
  %2234 = extractelement <4 x half> %2231, i64 1, !dbg !20
  %2235 = extractelement <4 x half> %2231, i64 0, !dbg !20
  %2236 = insertelement <4 x float> poison, float %1770, i64 0, !dbg !19
  %2237 = insertelement <4 x float> %2236, float %1776, i64 1, !dbg !19
  %2238 = insertelement <4 x float> %2237, float %1771, i64 2, !dbg !19
  %2239 = insertelement <4 x float> %2238, float %1777, i64 3, !dbg !19
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1749, !dbg !19
  %normalized124 = fmul <4 x float> %2239, %row.reciprocal123, !dbg !19
  %2240 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %2241 = extractelement <4 x half> %2240, i64 3, !dbg !20
  %2242 = extractelement <4 x half> %2240, i64 2, !dbg !20
  %2243 = extractelement <4 x half> %2240, i64 1, !dbg !20
  %2244 = extractelement <4 x half> %2240, i64 0, !dbg !20
  %2245 = insertelement <4 x float> poison, float %1659, i64 0, !dbg !19
  %2246 = insertelement <4 x float> %2245, float %1665, i64 1, !dbg !19
  %2247 = insertelement <4 x float> %2246, float %1660, i64 2, !dbg !19
  %2248 = insertelement <4 x float> %2247, float %1666, i64 3, !dbg !19
  %normalized125 = fmul <4 x float> %2248, %row.reciprocal123, !dbg !19
  %2249 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %2250 = extractelement <4 x half> %2249, i64 3, !dbg !20
  %2251 = extractelement <4 x half> %2249, i64 2, !dbg !20
  %2252 = extractelement <4 x half> %2249, i64 1, !dbg !20
  %2253 = extractelement <4 x half> %2249, i64 0, !dbg !20
  %2254 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1257 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1258 = zext nneg i32 %address.thread1257 to i64, !dbg !8
  %address.local1259 = add i64 %25, %address.local1258, !dbg !21
  %address.local1260 = lshr i64 %address.local1258, 6, !dbg !11
  %address.local1261 = shl nuw nsw i64 %address.local1260, 4, !dbg !21
  %address.local1262 = add i64 %address.local1259, %address.local1261, !dbg !21
  %address.local1263 = and i64 %address.local1258, 240, !dbg !21
  %address.local1264 = sub i64 %address.local1262, %address.local1263, !dbg !21
  %address.local1265 = add i64 %address.local1264, %26, !dbg !21
  %address.local1266 = shl nuw nsw i64 %address.local1265, 7, !dbg !21
  %address.local1267 = add nuw nsw i64 %2254, %address.local1266, !dbg !21
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1269 = lshr i64 %address.local1258, 2, !dbg !11
  %address.local1270 = and i64 %address.local1269, 12, !dbg !11
  %address.local1271 = or disjoint i64 %address.local1270, 96, !dbg !21
  %address.local1272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1268, i64 %address.local1271, !dbg !21
  store half %2253, ptr addrspace(1) %address.local1272, align 2, !dbg !21
  %address.local1273 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1274 = or disjoint i64 %address.local1270, 97, !dbg !21
  %address.local1275 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1273, i64 %address.local1274, !dbg !21
  store half %2252, ptr addrspace(1) %address.local1275, align 2, !dbg !21
  %address.local1276 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1277 = or disjoint i64 %address.local1270, 98, !dbg !21
  %address.local1278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1276, i64 %address.local1277, !dbg !21
  store half %2251, ptr addrspace(1) %address.local1278, align 2, !dbg !21
  %address.local1279 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1280 = or disjoint i64 %address.local1270, 99, !dbg !21
  %address.local1281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1279, i64 %address.local1280, !dbg !21
  store half %2250, ptr addrspace(1) %address.local1281, align 2, !dbg !21
  %address.local1282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1283 = or disjoint i64 %address.local1270, 112, !dbg !21
  %address.local1284 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1282, i64 %address.local1283, !dbg !21
  store half %2244, ptr addrspace(1) %address.local1284, align 2, !dbg !21
  %address.local1285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1286 = or disjoint i64 %address.local1270, 113, !dbg !21
  %address.local1287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1285, i64 %address.local1286, !dbg !21
  store half %2243, ptr addrspace(1) %address.local1287, align 2, !dbg !21
  %address.local1288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1289 = or disjoint i64 %address.local1270, 114, !dbg !21
  %address.local1290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1288, i64 %address.local1289, !dbg !21
  store half %2242, ptr addrspace(1) %address.local1290, align 2, !dbg !21
  %address.local1291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1267, !dbg !21
  %address.local1292 = or i64 %address.local1269, 115, !dbg !21
  %address.local1293 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1291, i64 %address.local1292, !dbg !21
  store half %2241, ptr addrspace(1) %address.local1293, align 2, !dbg !21
  %address.local1294 = shl i64 %address.local1265, 7, !dbg !21
  %address.local1295 = add i64 %address.local1294, 8192, !dbg !21
  %address.local1296 = add nuw nsw i64 %2254, %address.local1295, !dbg !21
  %address.local1297 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1297, i64 %address.local1271, !dbg !21
  store half %2235, ptr addrspace(1) %address.local1298, align 2, !dbg !21
  %address.local1299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1300 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1299, i64 %address.local1274, !dbg !21
  store half %2234, ptr addrspace(1) %address.local1300, align 2, !dbg !21
  %address.local1301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1301, i64 %address.local1277, !dbg !21
  store half %2233, ptr addrspace(1) %address.local1302, align 2, !dbg !21
  %address.local1303 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1304 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1303, i64 %address.local1280, !dbg !21
  store half %2232, ptr addrspace(1) %address.local1304, align 2, !dbg !21
  %address.local1305 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1306 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1305, i64 %address.local1283, !dbg !21
  store half %2226, ptr addrspace(1) %address.local1306, align 2, !dbg !21
  %address.local1307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1307, i64 %address.local1286, !dbg !21
  store half %2225, ptr addrspace(1) %address.local1308, align 2, !dbg !21
  %address.local1309 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1309, i64 %address.local1289, !dbg !21
  store half %2224, ptr addrspace(1) %address.local1310, align 2, !dbg !21
  %address.local1311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1296, !dbg !21
  %address.local1312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1311, i64 %address.local1292, !dbg !21
  store half %2223, ptr addrspace(1) %address.local1312, align 2, !dbg !21
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/deferred_lane_sum_pass/compiler_final/run-20261026-155755-g7dykgmh/input-0001-test_input_1x32x4096x128-128-128-32-online/grid-000001")
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
