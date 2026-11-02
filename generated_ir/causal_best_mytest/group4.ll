; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/causal_best_mytest/causal.input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/causal_best_mytest/causal.input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !36 {
  %kdma.tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !5
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !5
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !5
  %5 = and i32 %kdma.lane, 31, !dbg !5
  %kdma.owner = or disjoint i32 %kdma.wave, %5, !dbg !5
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !5
  %6 = shl nuw nsw i32 %kdma.tid, 3, !dbg !5
  %kdma.intra = and i32 %6, 8, !dbg !5
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %10 = sext i32 %9 to i64, !dbg !5
  %11 = shl nsw i64 %10, 5, !dbg !5
  %12 = add nsw i64 %11, %8, !dbg !5
  %13 = lshr i64 %12, 5, !dbg !5
  %14 = and i64 %13, 576460752303423484, !dbg !5
  %15 = and i64 %8, 3, !dbg !5
  %16 = or disjoint i64 %14, %15, !dbg !5
  %17 = lshr i64 %12, 2, !dbg !5
  %18 = lshr i64 %12, 10, !dbg !8
  %19 = add nuw nsw i64 %17, %18, !dbg !8
  %20 = shl i64 %16, 7, !dbg !8
  %21 = mul nsw i64 %18, -4096, !dbg !8
  %22 = add i64 %20, %21, !dbg !8
  %23 = shl i64 %19, 19, !dbg !8
  %24 = and i64 %23, 16252928, !dbg !8
  %25 = shl nsw i64 %22, 7, !dbg !8
  %26 = add i64 %24, %25, !dbg !8
  %27 = zext nneg i32 %kdma.tid to i64, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !8, !range !38, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local472 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local473 = and i64 %address.local472, 1536, !dbg !8
  %address.local474 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local473, !dbg !8
  %address.local475 = ptrtoint ptr addrspace(3) %address.local474 to i64, !dbg !8
  %address.local476 = trunc i64 %address.local475 to i32, !dbg !8
  %28 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local476), !dbg !8
  %address.local477 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local478 = and i64 %address.local477, 16256, !dbg !8
  %address.local479 = lshr i64 %address.local, 4, !dbg !8
  %address.local480 = and i64 %address.local479, 8, !dbg !8
  %address.local481 = or disjoint i64 %address.local478, %address.local480, !dbg !8
  %29 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1727 = or disjoint i64 %26, %address.local481, !dbg !8
  %.idx1727.tr = trunc i64 %.idx1727 to i32, !dbg !8
  %dma.buffer.offset1724 = shl i32 %.idx1727.tr, 1, !dbg !8
  %30 = insertelement <2 x i64> poison, i64 %29, i64 0, !dbg !8
  %31 = bitcast <2 x i64> %30 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1725 = shufflevector <4 x i32> %31, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1726 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %28), !dbg !8
  %32 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1724, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %address.local483 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local484 = shl nuw nsw i64 %address.local483, 3, !dbg !8
  %address.local485 = and i64 %address.local484, 3584, !dbg !8
  %address.local486 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local485, !dbg !8
  %address.local487 = ptrtoint ptr addrspace(3) %address.local486 to i64, !dbg !8
  %address.local488 = trunc i64 %address.local487 to i32, !dbg !8
  %33 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local488), !dbg !8
  %address.local491 = lshr i64 %address.local483, 4, !dbg !8
  %address.local492 = and i64 %address.local491, 24, !dbg !8
  %address.local493 = or disjoint i64 %address.local478, %address.local492, !dbg !8
  %.idx1728 = or disjoint i64 %26, %address.local493, !dbg !8
  %.idx1728.tr = trunc i64 %.idx1728 to i32, !dbg !8
  %dma.buffer.offset1721 = shl i32 %.idx1728.tr, 1, !dbg !8
  %dma.buffer.lds1723 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %33), !dbg !8
  %34 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset1721, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %35 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %36 = sext i32 %35 to i64, !dbg !5
  %37 = and i64 %27, 240, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local495 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local473, !dbg !8
  %address.local496 = ptrtoint ptr addrspace(3) %address.local495 to i64, !dbg !8
  %address.local497 = trunc i64 %address.local496 to i32, !dbg !8
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local497), !dbg !8
  %39 = or disjoint i64 %26, 32, !dbg !8
  %40 = or disjoint i64 %address.local481, %39, !dbg !8
  %.tr = trunc i64 %40 to i32, !dbg !8
  %dma.buffer.offset1718 = shl i32 %.tr, 1, !dbg !8
  %dma.buffer.lds1720 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %38), !dbg !8
  %41 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1720, i32 %dma.buffer.offset1718, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %address.local499 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local485, !dbg !8
  %address.local500 = ptrtoint ptr addrspace(3) %address.local499 to i64, !dbg !8
  %address.local501 = trunc i64 %address.local500 to i32, !dbg !8
  %42 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local501), !dbg !8
  %43 = or disjoint i64 %address.local493, %39, !dbg !8
  %.tr1729 = trunc i64 %43 to i32, !dbg !8
  %dma.buffer.offset1715 = shl i32 %.tr1729, 1, !dbg !8
  %dma.buffer.lds1717 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %42), !dbg !8
  %44 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1717, i32 %dma.buffer.offset1715, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %address.local503 = lshr i32 %address.thread, 2, !dbg !11
  %address.local504 = and i32 %address.local503, 4, !dbg !11
  %address.local505 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local506 = and i32 %address.local505, 384, !dbg !11
  %address.local507 = or disjoint i32 %address.local504, %address.local506, !dbg !11
  %address.local508 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local509 = and i32 %address.local508, 120, !dbg !11
  %address.local510 = or disjoint i32 %address.local507, %address.local509, !dbg !11
  %address.local511 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local512 = and i32 %address.local511, 1024, !dbg !11
  %address.local513 = or disjoint i32 %address.local510, %address.local512, !dbg !11
  %address.local514 = zext nneg i32 %address.local513 to i64, !dbg !11
  %address.local515 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local514, !dbg !11
  %45 = load <4 x half>, ptr addrspace(3) %address.local515, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local515, i64 4096, !dbg !11
  %46 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %47 = fmul <4 x half> %45, splat (half 0xH3015), !dbg !11
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local515, i64 1024, !dbg !11
  %48 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !11
  %49 = fmul <4 x half> %46, splat (half 0xH3015), !dbg !11
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local515, i64 5120, !dbg !11
  %50 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !11
  %51 = fmul <4 x half> %48, splat (half 0xH3015), !dbg !11
  %52 = fmul <4 x half> %50, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %53 = or disjoint i64 %26, 64, !dbg !8
  %54 = or disjoint i64 %address.local481, %53, !dbg !8
  %.tr1730 = trunc i64 %54 to i32, !dbg !8
  %dma.buffer.offset1712 = shl i32 %.tr1730, 1, !dbg !8
  %55 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1712, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %56 = or disjoint i64 %address.local493, %53, !dbg !8
  %.tr1731 = trunc i64 %56 to i32, !dbg !8
  %dma.buffer.offset1709 = shl i32 %.tr1731, 1, !dbg !8
  %57 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset1709, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local515, i64 8192, !dbg !11
  %58 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !11
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local515, i64 12288, !dbg !11
  %59 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !11
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local515, i64 9216, !dbg !11
  %60 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !11
  %61 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !11
  %lds.const.gep1530 = getelementptr i8, ptr addrspace(3) %address.local515, i64 13312, !dbg !11
  %62 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !11
  %63 = fmul <4 x half> %59, splat (half 0xH3015), !dbg !11
  %64 = fmul <4 x half> %60, splat (half 0xH3015), !dbg !11
  %65 = fmul <4 x half> %62, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %66 = or disjoint i64 %26, 96, !dbg !8
  %67 = or disjoint i64 %address.local481, %66, !dbg !8
  %.tr1732 = trunc i64 %67 to i32, !dbg !8
  %dma.buffer.offset1706 = shl i32 %.tr1732, 1, !dbg !8
  %68 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1720, i32 %dma.buffer.offset1706, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %69 = or disjoint i64 %address.local493, %66, !dbg !8
  %.tr1733 = trunc i64 %69 to i32, !dbg !8
  %dma.buffer.offset1703 = shl i32 %.tr1733, 1, !dbg !8
  %70 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1717, i32 %dma.buffer.offset1703, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %71 = load <4 x half>, ptr addrspace(3) %address.local515, align 8, !dbg !11
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !11
  %74 = fmul <4 x half> %71, splat (half 0xH3015), !dbg !11
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !11
  %76 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !11
  %77 = fmul <4 x half> %73, splat (half 0xH3015), !dbg !11
  %78 = fmul <4 x half> %75, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !11
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !11
  %81 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !11
  %82 = fmul <4 x half> %79, splat (half 0xH3015), !dbg !11
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !11
  %84 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !11
  %85 = fmul <4 x half> %81, splat (half 0xH3015), !dbg !11
  %86 = fmul <4 x half> %83, splat (half 0xH3015), !dbg !11
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %87 = shl i64 %17, 7, !dbg !14
  %88 = and i64 %87, 3968, !dbg !14
  %89 = shl nsw i64 %36, 12, !dbg !14
  %90 = or disjoint i64 %88, %89, !dbg !14
  %91 = shl nuw nsw i64 %27, 2, !dbg !15
  %92 = and i64 %91, 3084, !dbg !15
  %93 = and i32 %kdma.tid, 1, !dbg !15
  %index.compact412 = zext nneg i32 %93 to i64, !dbg !15
  %94 = or disjoint i64 %92, %index.compact412, !dbg !15
  %95 = add nuw nsw i64 %94, 1, !dbg !15
  %96 = lshr i64 %95, 1, !dbg !15
  %97 = add i64 %20, %27, !dbg !17
  %98 = lshr i64 %27, 2, !dbg !17
  %99 = and i64 %98, 48, !dbg !17
  %100 = add i64 %97, %99, !dbg !17
  %101 = sub i64 %100, %37, !dbg !17
  %102 = and i64 %27, 48, !dbg !17
  %103 = add i64 %101, 1, !dbg !18
  %104 = add i64 %101, 65, !dbg !18
  %k0.prime.address.thread566 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !14, !range !38, !frisk.register_lifetimes !1
  %k0.prime.address.local567 = zext nneg i32 %k0.prime.address.thread566 to i64, !dbg !8
  %k0.prime.address.local568 = shl nuw nsw i64 %k0.prime.address.local567, 3, !dbg !8
  %k0.prime.address.local569 = and i64 %k0.prime.address.local568, 1536, !dbg !8
  %k0.prime.address.local570 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.prime.address.local569, !dbg !8
  %k0.prime.address.local571 = ptrtoint ptr addrspace(3) %k0.prime.address.local570 to i64, !dbg !8
  %k0.prime.address.local572 = trunc i64 %k0.prime.address.local571 to i32, !dbg !8
  %k0.prime. = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local572), !dbg !14
  %k0.prime.dma.buffer.lds1702 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.), !dbg !14
  %k0.prime.2347 = lshr i64 0, 12, !dbg !14
  %k0.prime.2348 = add nuw nsw i64 %k0.prime.2347, %17, !dbg !14
  %k0.prime.2349 = shl i64 %k0.prime.2348, 19, !dbg !14
  %k0.prime.2350 = and i64 %k0.prime.2349, 16252928, !dbg !14
  %k0.prime.2351 = lshr i64 0, 5, !dbg !14
  %k0.prime.2352 = and i64 %k0.prime.2351, 127, !dbg !14
  %k0.prime..idx1736 = or disjoint i64 %k0.prime.2350, %k0.prime.2352, !dbg !14
  %k0.prime.2353 = shl i64 0, 14, !dbg !14
  %k0.prime.2354 = shl i64 %k0.prime.2351, 19, !dbg !14
  %k0.prime.2355 = sub i64 %k0.prime.2353, %k0.prime.2354, !dbg !14
  %k0.prime.2356 = add i64 %k0.prime..idx1736, %k0.prime.2355, !dbg !14
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !14
  %k0.prime.kdma.source1878 = lshr i64 %k0.prime.kdma.source, 4, !dbg !14
  %k0.prime.kdma.source1879 = and i64 %k0.prime.kdma.source1878, 8, !dbg !14
  %k0.prime.kdma.source1880 = and i64 %k0.prime.kdma.source, 91, !dbg !14
  %k0.prime.kdma.source1881 = lshr i64 %k0.prime.kdma.source, 2, !dbg !14
  %k0.prime.kdma.source1882 = and i64 %k0.prime.kdma.source1881, 4, !dbg !14
  %k0.prime.kdma.source1883 = add nuw nsw i64 %k0.prime.kdma.source1882, %k0.prime.kdma.source, !dbg !14
  %k0.prime.kdma.source1884 = and i64 %k0.prime.kdma.source1883, 4, !dbg !14
  %k0.prime.kdma.source1885 = or disjoint i64 %k0.prime.kdma.source1880, %k0.prime.kdma.source1884, !dbg !14
  %k0.prime.kdma.source1886 = shl nuw nsw i64 %k0.prime.kdma.source1885, 7, !dbg !14
  %k0.prime.kdma.source1887 = or disjoint i64 %k0.prime.kdma.source1879, %k0.prime.kdma.source1886, !dbg !14
  %k0.prime.kdma.source1888 = add nuw nsw i64 %k0.prime.2356, %k0.prime.kdma.source1887, !dbg !14
  %k0.prime.kdma.source1889 = trunc i64 %k0.prime.kdma.source1888 to i32, !dbg !14
  %k0.prime.kdma.source1890 = shl i32 %k0.prime.kdma.source1889, 1, !dbg !14
  %k0.prime.kdma.offset = add i32 %k0.prime.kdma.source1890, %kdma.intra, !dbg !14
  %k0.prime.2357 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !14
  %k0.prime.2358 = insertelement <2 x i64> poison, i64 %k0.prime.2357, i64 0, !dbg !14
  %k0.prime.2359 = bitcast <2 x i64> %k0.prime.2358 to <4 x i32>, !dbg !14
  %k0.prime.dma.buffer.rsrc1701 = shufflevector <4 x i32> %k0.prime.2359, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !14
  %k0.prime.2360 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.dma.buffer.lds1702, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #10, !dbg !14, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1906 = add i32 %k0.prime.dma.buffer.lds1702, 512, !dbg !14
  %k0.prime.kdma.source1891 = zext nneg i32 %kdma.owner.hi to i64, !dbg !14
  %k0.prime.kdma.source1892 = lshr i64 %k0.prime.kdma.source1891, 4, !dbg !14
  %k0.prime.kdma.source1893 = and i64 %k0.prime.kdma.source1892, 8, !dbg !14
  %k0.prime.kdma.source1894 = and i64 %k0.prime.kdma.source1891, 123, !dbg !14
  %k0.prime.kdma.source1895 = lshr i64 %k0.prime.kdma.source1891, 2, !dbg !14
  %k0.prime.kdma.source1896 = and i64 %k0.prime.kdma.source1895, 4, !dbg !14
  %k0.prime.kdma.source1897 = add nuw nsw i64 %k0.prime.kdma.source1896, %k0.prime.kdma.source1891, !dbg !14
  %k0.prime.kdma.source1898 = and i64 %k0.prime.kdma.source1897, 4, !dbg !14
  %k0.prime.kdma.source1899 = or disjoint i64 %k0.prime.kdma.source1894, %k0.prime.kdma.source1898, !dbg !14
  %k0.prime.kdma.source1900 = shl nuw nsw i64 %k0.prime.kdma.source1899, 7, !dbg !14
  %k0.prime.kdma.source1901 = or disjoint i64 %k0.prime.kdma.source1893, %k0.prime.kdma.source1900, !dbg !14
  %k0.prime.kdma.source1902 = add i64 %k0.prime.2356, %k0.prime.kdma.source1901, !dbg !14
  %k0.prime.kdma.source1903 = trunc i64 %k0.prime.kdma.source1902 to i32, !dbg !14
  %k0.prime.kdma.source1904 = shl i32 %k0.prime.kdma.source1903, 1, !dbg !14
  %k0.prime.kdma.offset1905 = add i32 %k0.prime.kdma.source1904, %kdma.intra.hi, !dbg !14
  %k0.prime.2361 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1906, i32 %k0.prime.kdma.offset1905, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #10, !dbg !14, !frisk.cross_bn_k0 !1
  %k0.prime.address.local584 = or disjoint i64 %k0.prime.address.local567, 256, !dbg !8
  %k0.prime.address.local585 = shl nuw nsw i64 %k0.prime.address.local584, 3, !dbg !8
  %k0.prime.address.local586 = and i64 %k0.prime.address.local585, 3584, !dbg !8
  %k0.prime.address.local587 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.prime.address.local586, !dbg !8
  %k0.prime.address.local588 = ptrtoint ptr addrspace(3) %k0.prime.address.local587 to i64, !dbg !8
  %k0.prime.address.local589 = trunc i64 %k0.prime.address.local588 to i32, !dbg !8
  %k0.prime.2362 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local589), !dbg !14
  %k0.prime.dma.buffer.lds1699 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.2362), !dbg !14
  %k0.prime.kdma.source1918 = or disjoint i64 %k0.prime.kdma.source1887, 16, !dbg !14
  %k0.prime.kdma.source1919 = add nuw nsw i64 %k0.prime.2356, %k0.prime.kdma.source1918, !dbg !14
  %k0.prime.kdma.source1920 = trunc i64 %k0.prime.kdma.source1919 to i32, !dbg !14
  %k0.prime.kdma.source1921 = shl i32 %k0.prime.kdma.source1920, 1, !dbg !14
  %k0.prime.kdma.offset1922 = add i32 %k0.prime.kdma.source1921, %kdma.intra, !dbg !14
  %k0.prime.2363 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.dma.buffer.lds1699, i32 %k0.prime.kdma.offset1922, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #10, !dbg !14, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1940 = add i32 %k0.prime.dma.buffer.lds1699, 512, !dbg !14
  %k0.prime.kdma.source1935 = or disjoint i64 %k0.prime.kdma.source1901, 16, !dbg !14
  %k0.prime.kdma.source1936 = add i64 %k0.prime.2356, %k0.prime.kdma.source1935, !dbg !14
  %k0.prime.kdma.source1937 = trunc i64 %k0.prime.kdma.source1936 to i32, !dbg !14
  %k0.prime.kdma.source1938 = shl i32 %k0.prime.kdma.source1937, 1, !dbg !14
  %k0.prime.kdma.offset1939 = add i32 %k0.prime.kdma.source1938, %kdma.intra.hi, !dbg !14
  %k0.prime.2364 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1940, i32 %k0.prime.kdma.offset1939, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #10, !dbg !14, !frisk.cross_bn_k0 !1
  br label %105, !dbg !7

105:                                              ; preds = %k0.join, %4
  %106 = phi i64 [ %1160, %k0.join ], [ 0, %4 ], !dbg !7
  %107 = phi [2 x <8 x float>] [ %1238, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %108 = phi [2 x <1 x float>] [ %1003, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %109 = phi [2 x <1 x float>] [ %1240, %k0.join ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %110 = phi [2 x <8 x float>] [ %1244, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %111 = phi [2 x <8 x float>] [ %1248, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %112 = phi [2 x <8 x float>] [ %1252, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %.not = icmp samesign ugt i64 %106, %16, !dbg !7
  br i1 %.not, label %1253, label %113, !dbg !7

113:                                              ; preds = %105
  %114 = lshr i64 %106, 12, !dbg !14
  %115 = add nuw nsw i64 %114, %17, !dbg !14
  %116 = lshr i64 %106, 5, !dbg !14
  %117 = and i64 %116, 127, !dbg !14
  %118 = shl nsw i64 %106, 7, !dbg !14
  %119 = shl i64 %115, 19, !dbg !14
  %120 = and i64 %119, 16252928, !dbg !14
  %121 = shl i64 %116, 19, !dbg !14
  %122 = shl i64 %106, 14, !dbg !14
  %123 = sub i64 %122, %121, !dbg !14
  %address.thread566 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !14, !range !38, !frisk.register_lifetimes !1
  %address.local567 = zext nneg i32 %address.thread566 to i64, !dbg !8
  %address.local568 = shl nuw nsw i64 %address.local567, 3, !dbg !8
  %address.local569 = and i64 %address.local568, 1536, !dbg !8
  %address.local570 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local569, !dbg !8
  %address.local571 = ptrtoint ptr addrspace(3) %address.local570 to i64, !dbg !8
  %address.local572 = trunc i64 %address.local571 to i32, !dbg !8
  %124 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local572), !dbg !14
  %125 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !14
  %.idx1736 = or disjoint i64 %120, %117, !dbg !14
  %126 = add i64 %.idx1736, %123, !dbg !14
  %127 = insertelement <2 x i64> poison, i64 %125, i64 0, !dbg !14
  %128 = bitcast <2 x i64> %127 to <4 x i32>, !dbg !14
  %dma.buffer.rsrc1701 = shufflevector <4 x i32> %128, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !14
  %dma.buffer.lds1702 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %124), !dbg !14
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !14
  %kdma.source1878 = lshr i64 %kdma.source, 4, !dbg !14
  %kdma.source1879 = and i64 %kdma.source1878, 8, !dbg !14
  %kdma.source1880 = and i64 %kdma.source, 91, !dbg !14
  %kdma.source1881 = lshr i64 %kdma.source, 2, !dbg !14
  %kdma.source1882 = and i64 %kdma.source1881, 4, !dbg !14
  %kdma.source1883 = add nuw nsw i64 %kdma.source1882, %kdma.source, !dbg !14
  %kdma.source1884 = and i64 %kdma.source1883, 4, !dbg !14
  %kdma.source1885 = or disjoint i64 %kdma.source1880, %kdma.source1884, !dbg !14
  %kdma.source1886 = shl nuw nsw i64 %kdma.source1885, 7, !dbg !14
  %kdma.source1887 = or disjoint i64 %kdma.source1879, %kdma.source1886, !dbg !14
  %kdma.source1888 = add nuw nsw i64 %126, %kdma.source1887, !dbg !14
  %kdma.source1889 = trunc i64 %kdma.source1888 to i32, !dbg !14
  %kdma.source1890 = shl i32 %kdma.source1889, 1, !dbg !14
  %kdma.offset = add i32 %kdma.source1890, %kdma.intra, !dbg !14
  %kdma.source1891 = zext nneg i32 %kdma.owner.hi to i64, !dbg !14
  %kdma.source1892 = lshr i64 %kdma.source1891, 4, !dbg !14
  %kdma.source1893 = and i64 %kdma.source1892, 8, !dbg !14
  %kdma.source1894 = and i64 %kdma.source1891, 123, !dbg !14
  %kdma.source1895 = lshr i64 %kdma.source1891, 2, !dbg !14
  %kdma.source1896 = and i64 %kdma.source1895, 4, !dbg !14
  %kdma.source1897 = add nuw nsw i64 %kdma.source1896, %kdma.source1891, !dbg !14
  %kdma.source1898 = and i64 %kdma.source1897, 4, !dbg !14
  %kdma.source1899 = or disjoint i64 %kdma.source1894, %kdma.source1898, !dbg !14
  %kdma.source1900 = shl nuw nsw i64 %kdma.source1899, 7, !dbg !14
  %kdma.source1901 = or disjoint i64 %kdma.source1893, %kdma.source1900, !dbg !14
  %kdma.source1902 = add i64 %126, %kdma.source1901, !dbg !14
  %kdma.source1903 = trunc i64 %kdma.source1902 to i32, !dbg !14
  %kdma.source1904 = shl i32 %kdma.source1903, 1, !dbg !14
  %kdma.offset1905 = add i32 %kdma.source1904, %kdma.intra.hi, !dbg !14
  %kdma.lds1906 = add i32 %dma.buffer.lds1702, 512, !dbg !14
  %address.local584 = or disjoint i64 %address.local567, 256, !dbg !8
  %address.local585 = shl nuw nsw i64 %address.local584, 3, !dbg !8
  %address.local586 = and i64 %address.local585, 3584, !dbg !8
  %address.local587 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local586, !dbg !8
  %address.local588 = ptrtoint ptr addrspace(3) %address.local587 to i64, !dbg !8
  %address.local589 = trunc i64 %address.local588 to i32, !dbg !8
  %129 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local589), !dbg !14
  %dma.buffer.lds1699 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %129), !dbg !14
  %kdma.source1918 = or disjoint i64 %kdma.source1887, 16, !dbg !14
  %kdma.source1919 = add nuw nsw i64 %126, %kdma.source1918, !dbg !14
  %kdma.source1920 = trunc i64 %kdma.source1919 to i32, !dbg !14
  %kdma.source1921 = shl i32 %kdma.source1920, 1, !dbg !14
  %kdma.offset1922 = add i32 %kdma.source1921, %kdma.intra, !dbg !14
  %kdma.source1935 = or disjoint i64 %kdma.source1901, 16, !dbg !14
  %kdma.source1936 = add i64 %126, %kdma.source1935, !dbg !14
  %kdma.source1937 = trunc i64 %kdma.source1936 to i32, !dbg !14
  %kdma.source1938 = shl i32 %kdma.source1937, 1, !dbg !14
  %kdma.offset1939 = add i32 %kdma.source1938, %kdma.intra.hi, !dbg !14
  %kdma.lds1940 = add i32 %dma.buffer.lds1699, 512, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %130 = add nuw i64 %106, 1024, !dbg !14
  %131 = icmp slt i64 %130, 0, !dbg !14
  %132 = sub nuw i64 -1025, %106, !dbg !14
  %133 = select i1 %131, i64 %132, i64 %130, !dbg !14
  %134 = sdiv i64 %133, 4096, !dbg !14
  %.lobit38 = ashr i64 %130, 63, !dbg !14
  %135 = xor i64 %134, %.lobit38, !dbg !14
  %136 = add nsw i64 %135, %17, !dbg !14
  %137 = add nsw i64 %116, %90, !dbg !14
  %138 = add nsw i64 %137, 32, !dbg !14
  %139 = icmp slt i64 %137, -32, !dbg !14
  %140 = sub nsw i64 -33, %137, !dbg !14
  %141 = select i1 %139, i64 %140, i64 %138, !dbg !14
  %142 = sdiv i64 %141, 128, !dbg !14
  %143 = sext i1 %139 to i64, !dbg !14
  %144 = xor i64 %142, %143, !dbg !14
  %145 = shl nsw i64 %144, 7, !dbg !14
  %146 = sub i64 %90, %145, !dbg !14
  %147 = shl i64 %136, 19, !dbg !14
  %148 = and i64 %147, 16252928, !dbg !14
  %address.local601 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local569, !dbg !8
  %address.local602 = ptrtoint ptr addrspace(3) %address.local601 to i64, !dbg !8
  %address.local603 = trunc i64 %address.local602 to i32, !dbg !8
  %149 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local603), !dbg !14
  %.idx1745 = add nuw nsw i64 %148, %116, !dbg !14
  %150 = add i64 %.idx1745, %146, !dbg !14
  %151 = add i64 %150, %123, !dbg !14
  %dma.buffer.lds1696 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %149), !dbg !14
  %kdma.source1952 = add i64 %151, %kdma.source1887, !dbg !14
  %kdma.source1953 = trunc i64 %kdma.source1952 to i32, !dbg !14
  %kdma.source1954 = shl i32 %kdma.source1953, 1, !dbg !14
  %kdma.source1955 = add i32 %kdma.source1954, 64, !dbg !14
  %kdma.offset1956 = add i32 %kdma.source1955, %kdma.intra, !dbg !14
  %152 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1696, i32 %kdma.offset1956, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source1969 = add i64 %151, %kdma.source1901, !dbg !14
  %kdma.source1970 = trunc i64 %kdma.source1969 to i32, !dbg !14
  %kdma.source1971 = shl i32 %kdma.source1970, 1, !dbg !14
  %kdma.source1972 = add i32 %kdma.source1971, 64, !dbg !14
  %kdma.offset1973 = add i32 %kdma.source1972, %kdma.intra.hi, !dbg !14
  %kdma.lds1974 = add i32 %dma.buffer.lds1696, 512, !dbg !14
  %153 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1974, i32 %kdma.offset1973, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %address.local605 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local586, !dbg !8
  %address.local606 = ptrtoint ptr addrspace(3) %address.local605 to i64, !dbg !8
  %address.local607 = trunc i64 %address.local606 to i32, !dbg !8
  %154 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local607), !dbg !14
  %dma.buffer.lds1693 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %154), !dbg !14
  %kdma.source1987 = add i64 %151, %kdma.source1918, !dbg !14
  %kdma.source1988 = trunc i64 %kdma.source1987 to i32, !dbg !14
  %kdma.source1989 = shl i32 %kdma.source1988, 1, !dbg !14
  %kdma.source1990 = add i32 %kdma.source1989, 64, !dbg !14
  %kdma.offset1991 = add i32 %kdma.source1990, %kdma.intra, !dbg !14
  %155 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1693, i32 %kdma.offset1991, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source2005 = add i64 %151, %kdma.source1935, !dbg !14
  %kdma.source2006 = trunc i64 %kdma.source2005 to i32, !dbg !14
  %kdma.source2007 = shl i32 %kdma.source2006, 1, !dbg !14
  %kdma.source2008 = add i32 %kdma.source2007, 64, !dbg !14
  %kdma.offset2009 = add i32 %kdma.source2008, %kdma.intra.hi, !dbg !14
  %kdma.lds2010 = add i32 %dma.buffer.lds1693, 512, !dbg !14
  %156 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds2010, i32 %kdma.offset2009, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %address.local610 = lshr i64 %address.local567, 2, !dbg !15
  %address.local613 = shl nuw nsw i64 %address.local567, 1, !dbg !15
  %address.local614 = and i64 %address.local613, 24, !dbg !15
  %address.local616 = shl i64 %address.local567, 62, !dbg !15
  %address.local617 = ashr exact i64 %address.local616, 57, !dbg !15
  %address.local618 = and i64 %address.local617, -64, !dbg !15
  %address.local621 = and i64 %address.local610, 12, !dbg !11
  %address.local623 = and i64 %address.local567, 63, !dbg !11
  %address.local624 = lshr i64 %address.local623, 5, !dbg !11
  %address.local625 = mul nuw nsw i64 %address.local624, 1016, !dbg !11
  %.idx = mul nuw nsw i64 %address.local567, 320, !dbg !15
  %157 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !15
  %.idx1752 = mul nsw i64 %address.local610, -1280, !dbg !15
  %158 = getelementptr i8, ptr addrspace(3) %157, i64 %.idx1752, !dbg !15
  %159 = getelementptr [2 x i8], ptr addrspace(3) %158, i64 %address.local614, !dbg !15
  %160 = getelementptr [2 x i8], ptr addrspace(3) %159, i64 %address.local618, !dbg !15
  %161 = getelementptr [2 x i8], ptr addrspace(3) %160, i64 %address.local621, !dbg !15
  %address.local627 = getelementptr [2 x i8], ptr addrspace(3) %161, i64 %address.local625, !dbg !15
  %162 = add nsw i64 %.idx, %.idx1752, !dbg !15
  %.idx2277 = shl nuw nsw i64 %address.local614, 1, !dbg !15
  %163 = or disjoint i64 %162, %.idx2277, !dbg !15
  %164 = or disjoint i64 %address.local618, %address.local621, !dbg !15
  %165 = shl nsw i64 %164, 1, !dbg !15
  %166 = add nsw i64 %165, %163, !dbg !15
  %address.local627.idx = mul nuw nsw i64 %address.local624, 2032, !dbg !15
  %167 = add nsw i64 %166, %address.local627.idx, !dbg !15
  %168 = lshr i64 %167, 6, !dbg !15
  %169 = and i64 %168, 8, !dbg !15
  %170 = xor i64 %167, %169, !dbg !15
  %171 = sub i64 %170, %167, !dbg !15
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local627, i64 %171, !dbg !15
  %172 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %172, <4 x float> zeroinitializer) #9, !dbg !15
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !15
  %174 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15
  %address.local638 = shl nuw nsw i32 %address.thread566, 1, !dbg !15
  %address.local639 = and i32 %address.local638, 24, !dbg !15
  %address.local640 = shl nuw nsw i32 %address.thread566, 8, !dbg !15
  %address.local641 = and i32 %address.local640, 768, !dbg !15
  %address.local642 = or disjoint i32 %address.local639, %address.local641, !dbg !15
  %address.local643 = shl nuw nsw i32 %address.thread566, 5, !dbg !15
  %address.local644 = and i32 %address.local643, 96, !dbg !15
  %address.local645 = or disjoint i32 %address.local642, %address.local644, !dbg !15
  %address.local646 = zext nneg i32 %address.local645 to i64, !dbg !15
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %96) #9, !dbg !15, !frisk.register_lifetimes !1
  %address.local647 = lshr i32 %address.thread566, 1, !dbg !15
  %address.local648 = and i32 %address.local647, 1, !dbg !15
  %address.local649 = zext nneg i32 %address.local648 to i64, !dbg !15
  %address.local650 = add nsw i64 %address.base, %address.local649, !dbg !15
  %address.local651 = shl i64 %address.local650, 6, !dbg !15
  %address.local652 = sub i64 %address.local646, %address.local651, !dbg !15
  %175 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local652, !dbg !15
  %176 = getelementptr [2 x i8], ptr addrspace(3) %175, i64 %address.local621, !dbg !15
  %177 = getelementptr [2 x i8], ptr addrspace(3) %176, i64 %address.local625, !dbg !15
  %address.local656 = getelementptr i8, ptr addrspace(3) %177, i64 64, !dbg !15
  %.idx22802283 = add i64 %address.local652, %address.local621, !dbg !15
  %178 = shl i64 %.idx22802283, 1, !dbg !15
  %179 = add i64 %178, %address.local627.idx, !dbg !15
  %180 = add i64 %179, 64, !dbg !15
  %181 = lshr i64 %180, 6, !dbg !15
  %182 = and i64 %181, 8, !dbg !15
  %183 = xor i64 %180, %182, !dbg !15
  %184 = sub i64 %183, %180, !dbg !15
  %kdma.swizzled2151 = getelementptr i8, ptr addrspace(3) %address.local656, i64 %184, !dbg !15
  %185 = load <4 x half>, ptr addrspace(3) %kdma.swizzled2151, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep2155 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 128, !dbg !15
  %186 = load <4 x half>, ptr addrspace(3) %kdma.gep2155, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep2152 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 4096, !dbg !15
  %187 = load <4 x half>, ptr addrspace(3) %kdma.gep2152, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep2156 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 4224, !dbg !15
  %188 = load <4 x half>, ptr addrspace(3) %kdma.gep2156, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep2153 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !15
  %189 = load <4 x half>, ptr addrspace(3) %kdma.gep2153, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep2157 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !15
  %190 = load <4 x half>, ptr addrspace(3) %kdma.gep2157, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep2154 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !15
  %191 = load <4 x half>, ptr addrspace(3) %kdma.gep2154, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep2158 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !15
  %192 = load <4 x half>, ptr addrspace(3) %kdma.gep2158, align 8, !dbg !15, !frisk.k_lds_pair !44
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %174, <4 x float> %173) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep2159 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 1024, !dbg !15
  %194 = load <4 x half>, ptr addrspace(3) %kdma.gep2159, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep2163 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 1152, !dbg !15
  %195 = load <4 x half>, ptr addrspace(3) %kdma.gep2163, align 8, !dbg !15, !frisk.k_lds_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %185, <4 x float> zeroinitializer) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep2160 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 5120, !dbg !15
  %197 = load <4 x half>, ptr addrspace(3) %kdma.gep2160, align 8, !dbg !15, !frisk.k_lds_pair !45
  %kdma.gep2164 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 5248, !dbg !15
  %198 = load <4 x half>, ptr addrspace(3) %kdma.gep2164, align 8, !dbg !15, !frisk.k_lds_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %187, <4 x float> %196) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep2161 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !15
  %200 = load <4 x half>, ptr addrspace(3) %kdma.gep2161, align 8, !dbg !15, !frisk.k_lds_pair !46
  %kdma.gep2162 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !15
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep2162, align 8, !dbg !15, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %189, <4 x float> zeroinitializer) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %191, <4 x float> %202) #9, !dbg !15
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %186, <4 x float> zeroinitializer) #9, !dbg !15
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %188, <4 x float> %204) #9, !dbg !15
  %206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %190, <4 x float> zeroinitializer) #9, !dbg !15
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %192, <4 x float> %206) #9, !dbg !15
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %194, <4 x float> zeroinitializer) #9, !dbg !15
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %197, <4 x float> %208) #9, !dbg !15
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %200, <4 x float> zeroinitializer) #9, !dbg !15
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %201, <4 x float> %210) #9, !dbg !15
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %195, <4 x float> zeroinitializer) #9, !dbg !15
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %198, <4 x float> %212) #9, !dbg !15
  %214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %172, <4 x float> zeroinitializer) #9, !dbg !15
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %174, <4 x float> %214) #9, !dbg !15
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %185, <4 x float> zeroinitializer) #9, !dbg !15
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %187, <4 x float> %216) #9, !dbg !15
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %189, <4 x float> zeroinitializer) #9, !dbg !15
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %191, <4 x float> %218) #9, !dbg !15
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %186, <4 x float> zeroinitializer) #9, !dbg !15
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %188, <4 x float> %220) #9, !dbg !15
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %190, <4 x float> zeroinitializer) #9, !dbg !15
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %192, <4 x float> %222) #9, !dbg !15
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %194, <4 x float> zeroinitializer) #9, !dbg !15
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %197, <4 x float> %224) #9, !dbg !15
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %200, <4 x float> zeroinitializer) #9, !dbg !15
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %201, <4 x float> %226) #9, !dbg !15
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %195, <4 x float> zeroinitializer) #9, !dbg !15
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %198, <4 x float> %228) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %230 = add nuw i64 %106, 2048, !dbg !14
  %231 = icmp slt i64 %230, 0, !dbg !14
  %232 = sub nuw i64 -2049, %106, !dbg !14
  %233 = select i1 %231, i64 %232, i64 %230, !dbg !14
  %234 = sdiv i64 %233, 4096, !dbg !14
  %.lobit39 = ashr i64 %230, 63, !dbg !14
  %235 = xor i64 %234, %.lobit39, !dbg !14
  %236 = add nsw i64 %235, %17, !dbg !14
  %237 = add nsw i64 %137, 64, !dbg !14
  %238 = icmp slt i64 %137, -64, !dbg !14
  %239 = sub nsw i64 -65, %137, !dbg !14
  %240 = select i1 %238, i64 %239, i64 %237, !dbg !14
  %241 = sdiv i64 %240, 128, !dbg !14
  %242 = sext i1 %238 to i64, !dbg !14
  %243 = xor i64 %241, %242, !dbg !14
  %244 = shl nsw i64 %243, 7, !dbg !14
  %245 = sub i64 %90, %244, !dbg !14
  %246 = shl i64 %236, 19, !dbg !14
  %247 = and i64 %246, 16252928, !dbg !14
  %.idx17531757 = add nuw nsw i64 %247, %116, !dbg !14
  %248 = add i64 %.idx17531757, %245, !dbg !14
  %249 = add i64 %248, %123, !dbg !14
  %kdma.source2022 = add i64 %249, %kdma.source1887, !dbg !14
  %kdma.source2023 = trunc i64 %kdma.source2022 to i32, !dbg !14
  %kdma.source2024 = shl i32 %kdma.source2023, 1, !dbg !14
  %kdma.source2025 = add i32 %kdma.source2024, 128, !dbg !14
  %kdma.offset2026 = add i32 %kdma.source2025, %kdma.intra, !dbg !14
  %250 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %kdma.offset2026, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source2039 = add i64 %249, %kdma.source1901, !dbg !14
  %kdma.source2040 = trunc i64 %kdma.source2039 to i32, !dbg !14
  %kdma.source2041 = shl i32 %kdma.source2040, 1, !dbg !14
  %kdma.source2042 = add i32 %kdma.source2041, 128, !dbg !14
  %kdma.offset2043 = add i32 %kdma.source2042, %kdma.intra.hi, !dbg !14
  %251 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1906, i32 %kdma.offset2043, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source2057 = add i64 %249, %kdma.source1918, !dbg !14
  %kdma.source2058 = trunc i64 %kdma.source2057 to i32, !dbg !14
  %kdma.source2059 = shl i32 %kdma.source2058, 1, !dbg !14
  %kdma.source2060 = add i32 %kdma.source2059, 128, !dbg !14
  %kdma.offset2061 = add i32 %kdma.source2060, %kdma.intra, !dbg !14
  %252 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %kdma.offset2061, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source2075 = add i64 %249, %kdma.source1935, !dbg !14
  %kdma.source2076 = trunc i64 %kdma.source2075 to i32, !dbg !14
  %kdma.source2077 = shl i32 %kdma.source2076, 1, !dbg !14
  %kdma.source2078 = add i32 %kdma.source2077, 128, !dbg !14
  %kdma.offset2079 = add i32 %kdma.source2078, %kdma.intra.hi, !dbg !14
  %253 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1940, i32 %kdma.offset2079, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.gep2181 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !15
  %254 = load <4 x half>, ptr addrspace(3) %kdma.gep2181, align 8, !dbg !15, !frisk.k_lds_pair !47
  %kdma.gep2185 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !15
  %255 = load <4 x half>, ptr addrspace(3) %kdma.gep2185, align 8, !dbg !15, !frisk.k_lds_pair !47
  %kdma.gep2182 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !15
  %256 = load <4 x half>, ptr addrspace(3) %kdma.gep2182, align 8, !dbg !15, !frisk.k_lds_pair !48
  %kdma.gep2186 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !15
  %257 = load <4 x half>, ptr addrspace(3) %kdma.gep2186, align 8, !dbg !15, !frisk.k_lds_pair !48
  %kdma.gep2183 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 8192, !dbg !15
  %258 = load <4 x half>, ptr addrspace(3) %kdma.gep2183, align 8, !dbg !15, !frisk.k_lds_pair !49
  %kdma.gep2187 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 8320, !dbg !15
  %259 = load <4 x half>, ptr addrspace(3) %kdma.gep2187, align 8, !dbg !15, !frisk.k_lds_pair !49
  %kdma.gep2184 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 12288, !dbg !15
  %260 = load <4 x half>, ptr addrspace(3) %kdma.gep2184, align 8, !dbg !15, !frisk.k_lds_pair !50
  %kdma.gep2188 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 12416, !dbg !15
  %261 = load <4 x half>, ptr addrspace(3) %kdma.gep2188, align 8, !dbg !15, !frisk.k_lds_pair !50
  %kdma.gep2189 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !15
  %262 = load <4 x half>, ptr addrspace(3) %kdma.gep2189, align 8, !dbg !15, !frisk.k_lds_pair !51
  %kdma.gep2193 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !15
  %263 = load <4 x half>, ptr addrspace(3) %kdma.gep2193, align 8, !dbg !15, !frisk.k_lds_pair !51
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %254, <4 x float> %193) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep2190 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !15
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep2190, align 8, !dbg !15, !frisk.k_lds_pair !52
  %kdma.gep2194 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !15
  %266 = load <4 x half>, ptr addrspace(3) %kdma.gep2194, align 8, !dbg !15, !frisk.k_lds_pair !52
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %256, <4 x float> %264) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep2191 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 9216, !dbg !15
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep2191, align 8, !dbg !15, !frisk.k_lds_pair !53
  %kdma.gep2195 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 9344, !dbg !15
  %269 = load <4 x half>, ptr addrspace(3) %kdma.gep2195, align 8, !dbg !15, !frisk.k_lds_pair !53
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %258, <4 x float> %199) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep2192 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 13312, !dbg !15
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep2192, align 8, !dbg !15, !frisk.k_lds_pair !54
  %kdma.gep2196 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 13440, !dbg !15
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep2196, align 8, !dbg !15, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %260, <4 x float> %270) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %255, <4 x float> %203) #9, !dbg !15
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %257, <4 x float> %274) #9, !dbg !15
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %259, <4 x float> %205) #9, !dbg !15
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %261, <4 x float> %276) #9, !dbg !15
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %262, <4 x float> %207) #9, !dbg !15
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %265, <4 x float> %278) #9, !dbg !15
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %268, <4 x float> %209) #9, !dbg !15
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %271, <4 x float> %280) #9, !dbg !15
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %263, <4 x float> %211) #9, !dbg !15
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %266, <4 x float> %282) #9, !dbg !15
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %269, <4 x float> %213) #9, !dbg !15
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %272, <4 x float> %284) #9, !dbg !15
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %254, <4 x float> %215) #9, !dbg !15
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %256, <4 x float> %286) #9, !dbg !15
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %258, <4 x float> %217) #9, !dbg !15
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %260, <4 x float> %288) #9, !dbg !15
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %255, <4 x float> %219) #9, !dbg !15
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %257, <4 x float> %290) #9, !dbg !15
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %259, <4 x float> %221) #9, !dbg !15
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %261, <4 x float> %292) #9, !dbg !15
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %262, <4 x float> %223) #9, !dbg !15
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %265, <4 x float> %294) #9, !dbg !15
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %268, <4 x float> %225) #9, !dbg !15
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %271, <4 x float> %296) #9, !dbg !15
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %263, <4 x float> %227) #9, !dbg !15
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %266, <4 x float> %298) #9, !dbg !15
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %269, <4 x float> %229) #9, !dbg !15
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %272, <4 x float> %300) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %302 = add nuw i64 %106, 3072, !dbg !14
  %303 = icmp slt i64 %302, 0, !dbg !14
  %304 = sub nuw i64 -3073, %106, !dbg !14
  %305 = select i1 %303, i64 %304, i64 %302, !dbg !14
  %306 = sdiv i64 %305, 4096, !dbg !14
  %.lobit40 = ashr i64 %302, 63, !dbg !14
  %307 = xor i64 %306, %.lobit40, !dbg !14
  %308 = add nsw i64 %307, %17, !dbg !14
  %309 = add nsw i64 %137, 96, !dbg !14
  %310 = icmp slt i64 %137, -96, !dbg !14
  %311 = sub nsw i64 -97, %137, !dbg !14
  %312 = select i1 %310, i64 %311, i64 %309, !dbg !14
  %313 = sdiv i64 %312, 128, !dbg !14
  %314 = sext i1 %310 to i64, !dbg !14
  %315 = xor i64 %313, %314, !dbg !14
  %316 = shl nsw i64 %315, 7, !dbg !14
  %317 = sub i64 %90, %316, !dbg !14
  %318 = shl i64 %308, 19, !dbg !14
  %319 = and i64 %318, 16252928, !dbg !14
  %.idx17651769 = add nuw nsw i64 %319, %116, !dbg !14
  %320 = add i64 %.idx17651769, %317, !dbg !14
  %321 = add i64 %320, %123, !dbg !14
  %kdma.source2092 = add i64 %321, %kdma.source1887, !dbg !14
  %kdma.source2093 = trunc i64 %kdma.source2092 to i32, !dbg !14
  %kdma.source2094 = shl i32 %kdma.source2093, 1, !dbg !14
  %kdma.source2095 = add i32 %kdma.source2094, 192, !dbg !14
  %kdma.offset2096 = add i32 %kdma.source2095, %kdma.intra, !dbg !14
  %322 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1696, i32 %kdma.offset2096, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source2109 = add i64 %321, %kdma.source1901, !dbg !14
  %kdma.source2110 = trunc i64 %kdma.source2109 to i32, !dbg !14
  %kdma.source2111 = shl i32 %kdma.source2110, 1, !dbg !14
  %kdma.source2112 = add i32 %kdma.source2111, 192, !dbg !14
  %kdma.offset2113 = add i32 %kdma.source2112, %kdma.intra.hi, !dbg !14
  %323 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1974, i32 %kdma.offset2113, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source2127 = add i64 %321, %kdma.source1918, !dbg !14
  %kdma.source2128 = trunc i64 %kdma.source2127 to i32, !dbg !14
  %kdma.source2129 = shl i32 %kdma.source2128, 1, !dbg !14
  %kdma.source2130 = add i32 %kdma.source2129, 192, !dbg !14
  %kdma.offset2131 = add i32 %kdma.source2130, %kdma.intra, !dbg !14
  %324 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1693, i32 %kdma.offset2131, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source2145 = add i64 %321, %kdma.source1935, !dbg !14
  %kdma.source2146 = trunc i64 %kdma.source2145 to i32, !dbg !14
  %kdma.source2147 = shl i32 %kdma.source2146, 1, !dbg !14
  %kdma.source2148 = add i32 %kdma.source2147, 192, !dbg !14
  %kdma.offset2149 = add i32 %kdma.source2148, %kdma.intra.hi, !dbg !14
  %325 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds2010, i32 %kdma.offset2149, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %326 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15, !frisk.k_lds_pair !55
  %327 = load <4 x half>, ptr addrspace(3) %kdma.gep2153, align 8, !dbg !15, !frisk.k_lds_pair !55
  %328 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !56
  %329 = load <4 x half>, ptr addrspace(3) %kdma.gep2154, align 8, !dbg !15, !frisk.k_lds_pair !56
  %330 = load <4 x half>, ptr addrspace(3) %kdma.swizzled2151, align 8, !dbg !15, !frisk.k_lds_pair !57
  %331 = load <4 x half>, ptr addrspace(3) %kdma.gep2155, align 8, !dbg !15, !frisk.k_lds_pair !57
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep2152, align 8, !dbg !15, !frisk.k_lds_pair !58
  %333 = load <4 x half>, ptr addrspace(3) %kdma.gep2156, align 8, !dbg !15, !frisk.k_lds_pair !58
  %334 = load <4 x half>, ptr addrspace(3) %kdma.gep2157, align 8, !dbg !15, !frisk.k_lds_pair !59
  %335 = load <4 x half>, ptr addrspace(3) %kdma.gep2161, align 8, !dbg !15, !frisk.k_lds_pair !59
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %326, <4 x float> %267) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %337 = load <4 x half>, ptr addrspace(3) %kdma.gep2158, align 8, !dbg !15, !frisk.k_lds_pair !60
  %338 = load <4 x half>, ptr addrspace(3) %kdma.gep2162, align 8, !dbg !15, !frisk.k_lds_pair !60
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %328, <4 x float> %336) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %340 = load <4 x half>, ptr addrspace(3) %kdma.gep2159, align 8, !dbg !15, !frisk.k_lds_pair !61
  %341 = load <4 x half>, ptr addrspace(3) %kdma.gep2163, align 8, !dbg !15, !frisk.k_lds_pair !61
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %330, <4 x float> %273) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %343 = load <4 x half>, ptr addrspace(3) %kdma.gep2160, align 8, !dbg !15, !frisk.k_lds_pair !62
  %344 = load <4 x half>, ptr addrspace(3) %kdma.gep2164, align 8, !dbg !15, !frisk.k_lds_pair !62
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %332, <4 x float> %342) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %327, <4 x float> %275) #9, !dbg !15
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %329, <4 x float> %346) #9, !dbg !15
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %331, <4 x float> %277) #9, !dbg !15
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %333, <4 x float> %348) #9, !dbg !15
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %334, <4 x float> %279) #9, !dbg !15
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %337, <4 x float> %350) #9, !dbg !15
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %340, <4 x float> %281) #9, !dbg !15
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %343, <4 x float> %352) #9, !dbg !15
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %335, <4 x float> %283) #9, !dbg !15
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %338, <4 x float> %354) #9, !dbg !15
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %341, <4 x float> %285) #9, !dbg !15
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %344, <4 x float> %356) #9, !dbg !15
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %326, <4 x float> %287) #9, !dbg !15
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %328, <4 x float> %358) #9, !dbg !15
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %330, <4 x float> %289) #9, !dbg !15
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %332, <4 x float> %360) #9, !dbg !15
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %327, <4 x float> %291) #9, !dbg !15
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %329, <4 x float> %362) #9, !dbg !15
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %331, <4 x float> %293) #9, !dbg !15
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %333, <4 x float> %364) #9, !dbg !15
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %334, <4 x float> %295) #9, !dbg !15
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %337, <4 x float> %366) #9, !dbg !15
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %340, <4 x float> %297) #9, !dbg !15
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %343, <4 x float> %368) #9, !dbg !15
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %335, <4 x float> %299) #9, !dbg !15
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %338, <4 x float> %370) #9, !dbg !15
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %341, <4 x float> %301) #9, !dbg !15
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %344, <4 x float> %372) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %374 = add nuw nsw i64 %116, %17, !dbg !16
  %375 = shl i64 %374, 19, !dbg !16
  %376 = and i64 %375, 16252928, !dbg !16
  %address.local985 = shl nuw nsw i64 %address.local567, 5, !dbg !16
  %address.local986 = and i64 %address.local985, 8064, !dbg !16
  %address.local988 = and i64 %address.local568, 24, !dbg !16
  %address.local989 = or disjoint i64 %address.local986, %address.local988, !dbg !16
  %377 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx17771779 = add i64 %376, %123, !dbg !16
  %378 = or disjoint i64 %.idx17771779, %address.local989, !dbg !16
  %.tr1780 = trunc i64 %378 to i32, !dbg !16
  %dma.buffer.offset1676 = shl i32 %.tr1780, 1, !dbg !16
  %379 = insertelement <2 x i64> poison, i64 %377, i64 0, !dbg !16
  %380 = bitcast <2 x i64> %379 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1677 = shufflevector <4 x i32> %380, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %381 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %dma.buffer.offset1676, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %address.local991 = shl nuw nsw i64 %address.local584, 5, !dbg !16
  %address.local992 = and i64 %address.local991, 16256, !dbg !16
  %address.local995 = or disjoint i64 %address.local992, %address.local988, !dbg !16
  %382 = or disjoint i64 %.idx17771779, %address.local995, !dbg !16
  %.tr1784 = trunc i64 %382 to i32, !dbg !16
  %dma.buffer.offset1673 = shl i32 %.tr1784, 1, !dbg !16
  %383 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %dma.buffer.offset1673, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %384 = load <4 x half>, ptr addrspace(3) %kdma.gep2181, align 8, !dbg !15
  %385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %384, <4 x float> %339) #9, !dbg !15
  %386 = or disjoint i64 %118, %102, !dbg !17
  %387 = lshr i64 %103, 32, !dbg !23
  %compare.high = trunc nuw i64 %387 to i32, !dbg !23
  %388 = lshr i64 %118, 32, !dbg !23
  %compare.high1269 = trunc nuw i64 %388 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1269, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1269, !dbg !23
  %compare.low.base = trunc i64 %103 to i32, !dbg !23
  %compare.low.base1270 = trunc i64 %386 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1270, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %389 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low1271 = icmp ult i32 %compare.local, %compare.low.base, !dbg !23
  %compare.result1272 = select i1 %compare.high.equal, i1 %compare.low1271, i1 %compare.high.order, !dbg !23
  %390 = select i1 %compare.result1272, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1273 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1274 = or i32 %compare.local.base1273, 2, !dbg !23
  %compare.low1275 = icmp ult i32 %compare.local1274, %compare.low.base, !dbg !23
  %compare.result1276 = select i1 %compare.high.equal, i1 %compare.low1275, i1 %compare.high.order, !dbg !23
  %391 = select i1 %compare.result1276, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1277 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1278 = or i32 %compare.local.base1277, 3, !dbg !23
  %compare.low1279 = icmp ult i32 %compare.local1278, %compare.low.base, !dbg !23
  %compare.result1280 = select i1 %compare.high.equal, i1 %compare.low1279, i1 %compare.high.order, !dbg !23
  %392 = select i1 %compare.result1280, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %393 = insertelement <4 x float> poison, float %389, i64 0, !dbg !25
  %394 = insertelement <4 x float> %393, float %390, i64 1, !dbg !25
  %395 = insertelement <4 x float> %394, float %391, i64 2, !dbg !25
  %396 = insertelement <4 x float> %395, float %392, i64 3, !dbg !25
  %compare.local.base1281 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1282 = or i32 %compare.local.base1281, 4, !dbg !23
  %compare.low1283 = icmp ult i32 %compare.local1282, %compare.low.base, !dbg !23
  %compare.result1284 = select i1 %compare.high.equal, i1 %compare.low1283, i1 %compare.high.order, !dbg !23
  %397 = select i1 %compare.result1284, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1285 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1286 = or i32 %compare.local.base1285, 5, !dbg !23
  %compare.low1287 = icmp ult i32 %compare.local1286, %compare.low.base, !dbg !23
  %compare.result1288 = select i1 %compare.high.equal, i1 %compare.low1287, i1 %compare.high.order, !dbg !23
  %398 = select i1 %compare.result1288, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1289 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1290 = or i32 %compare.local.base1289, 6, !dbg !23
  %compare.low1291 = icmp ult i32 %compare.local1290, %compare.low.base, !dbg !23
  %compare.result1292 = select i1 %compare.high.equal, i1 %compare.low1291, i1 %compare.high.order, !dbg !23
  %399 = select i1 %compare.result1292, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1293 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1294 = or i32 %compare.local.base1293, 7, !dbg !23
  %compare.low1295 = icmp ult i32 %compare.local1294, %compare.low.base, !dbg !23
  %compare.result1296 = select i1 %compare.high.equal, i1 %compare.low1295, i1 %compare.high.order, !dbg !23
  %400 = select i1 %compare.result1296, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %401 = insertelement <4 x float> poison, float %397, i64 0, !dbg !25
  %402 = insertelement <4 x float> %401, float %398, i64 1, !dbg !25
  %403 = insertelement <4 x float> %402, float %399, i64 2, !dbg !25
  %404 = insertelement <4 x float> %403, float %400, i64 3, !dbg !25
  %compare.local.base1297 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1298 = or i32 %compare.local.base1297, 8, !dbg !23
  %compare.low1299 = icmp ult i32 %compare.local1298, %compare.low.base, !dbg !23
  %compare.result1300 = select i1 %compare.high.equal, i1 %compare.low1299, i1 %compare.high.order, !dbg !23
  %405 = select i1 %compare.result1300, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1301 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1302 = or i32 %compare.local.base1301, 9, !dbg !23
  %compare.low1303 = icmp ult i32 %compare.local1302, %compare.low.base, !dbg !23
  %compare.result1304 = select i1 %compare.high.equal, i1 %compare.low1303, i1 %compare.high.order, !dbg !23
  %406 = select i1 %compare.result1304, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1305 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1306 = or i32 %compare.local.base1305, 10, !dbg !23
  %compare.low1307 = icmp ult i32 %compare.local1306, %compare.low.base, !dbg !23
  %compare.result1308 = select i1 %compare.high.equal, i1 %compare.low1307, i1 %compare.high.order, !dbg !23
  %407 = select i1 %compare.result1308, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1309 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1310 = or i32 %compare.local.base1309, 11, !dbg !23
  %compare.low1311 = icmp ult i32 %compare.local1310, %compare.low.base, !dbg !23
  %compare.result1312 = select i1 %compare.high.equal, i1 %compare.low1311, i1 %compare.high.order, !dbg !23
  %408 = select i1 %compare.result1312, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %409 = insertelement <4 x float> poison, float %405, i64 0, !dbg !25
  %410 = insertelement <4 x float> %409, float %406, i64 1, !dbg !25
  %411 = insertelement <4 x float> %410, float %407, i64 2, !dbg !25
  %412 = insertelement <4 x float> %411, float %408, i64 3, !dbg !25
  %compare.local.base1313 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1314 = or i32 %compare.local.base1313, 12, !dbg !23
  %compare.low1315 = icmp ult i32 %compare.local1314, %compare.low.base, !dbg !23
  %compare.result1316 = select i1 %compare.high.equal, i1 %compare.low1315, i1 %compare.high.order, !dbg !23
  %413 = select i1 %compare.result1316, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1317 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1318 = or i32 %compare.local.base1317, 13, !dbg !23
  %compare.low1319 = icmp ult i32 %compare.local1318, %compare.low.base, !dbg !23
  %compare.result1320 = select i1 %compare.high.equal, i1 %compare.low1319, i1 %compare.high.order, !dbg !23
  %414 = select i1 %compare.result1320, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1321 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1322 = or i32 %compare.local.base1321, 14, !dbg !23
  %compare.low1323 = icmp ult i32 %compare.local1322, %compare.low.base, !dbg !23
  %compare.result1324 = select i1 %compare.high.equal, i1 %compare.low1323, i1 %compare.high.order, !dbg !23
  %415 = select i1 %compare.result1324, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1325 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1326 = or i32 %compare.local.base1325, 15, !dbg !23
  %compare.low1327 = icmp ult i32 %compare.local1326, %compare.low.base, !dbg !23
  %compare.result1328 = select i1 %compare.high.equal, i1 %compare.low1327, i1 %compare.high.order, !dbg !23
  %416 = select i1 %compare.result1328, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %417 = insertelement <4 x float> poison, float %413, i64 0, !dbg !25
  %418 = insertelement <4 x float> %417, float %414, i64 1, !dbg !25
  %419 = insertelement <4 x float> %418, float %415, i64 2, !dbg !25
  %420 = insertelement <4 x float> %419, float %416, i64 3, !dbg !25
  %compare.local.base1329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 16) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1330 = or i32 %compare.local.base1329, 64, !dbg !23
  %compare.low1331 = icmp ult i32 %compare.local1330, %compare.low.base, !dbg !23
  %compare.result1332 = select i1 %compare.high.equal, i1 %compare.low1331, i1 %compare.high.order, !dbg !23
  %421 = select i1 %compare.result1332, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 17) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1334 = or i32 %compare.local.base1333, 65, !dbg !23
  %compare.low1335 = icmp ult i32 %compare.local1334, %compare.low.base, !dbg !23
  %compare.result1336 = select i1 %compare.high.equal, i1 %compare.low1335, i1 %compare.high.order, !dbg !23
  %422 = select i1 %compare.result1336, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 18) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1338 = or i32 %compare.local.base1337, 66, !dbg !23
  %compare.low1339 = icmp ult i32 %compare.local1338, %compare.low.base, !dbg !23
  %compare.result1340 = select i1 %compare.high.equal, i1 %compare.low1339, i1 %compare.high.order, !dbg !23
  %423 = select i1 %compare.result1340, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 19) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1342 = or i32 %compare.local.base1341, 67, !dbg !23
  %compare.low1343 = icmp ult i32 %compare.local1342, %compare.low.base, !dbg !23
  %compare.result1344 = select i1 %compare.high.equal, i1 %compare.low1343, i1 %compare.high.order, !dbg !23
  %424 = select i1 %compare.result1344, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %425 = insertelement <4 x float> poison, float %421, i64 0, !dbg !25
  %426 = insertelement <4 x float> %425, float %422, i64 1, !dbg !25
  %427 = insertelement <4 x float> %426, float %423, i64 2, !dbg !25
  %428 = insertelement <4 x float> %427, float %424, i64 3, !dbg !25
  %compare.local.base1345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 20) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1346 = or i32 %compare.local.base1345, 68, !dbg !23
  %compare.low1347 = icmp ult i32 %compare.local1346, %compare.low.base, !dbg !23
  %compare.result1348 = select i1 %compare.high.equal, i1 %compare.low1347, i1 %compare.high.order, !dbg !23
  %429 = select i1 %compare.result1348, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 21) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1350 = or i32 %compare.local.base1349, 69, !dbg !23
  %compare.low1351 = icmp ult i32 %compare.local1350, %compare.low.base, !dbg !23
  %compare.result1352 = select i1 %compare.high.equal, i1 %compare.low1351, i1 %compare.high.order, !dbg !23
  %430 = select i1 %compare.result1352, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 22) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1354 = or i32 %compare.local.base1353, 70, !dbg !23
  %compare.low1355 = icmp ult i32 %compare.local1354, %compare.low.base, !dbg !23
  %compare.result1356 = select i1 %compare.high.equal, i1 %compare.low1355, i1 %compare.high.order, !dbg !23
  %431 = select i1 %compare.result1356, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 23) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1358 = or i32 %compare.local.base1357, 71, !dbg !23
  %compare.low1359 = icmp ult i32 %compare.local1358, %compare.low.base, !dbg !23
  %compare.result1360 = select i1 %compare.high.equal, i1 %compare.low1359, i1 %compare.high.order, !dbg !23
  %432 = select i1 %compare.result1360, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %433 = insertelement <4 x float> poison, float %429, i64 0, !dbg !25
  %434 = insertelement <4 x float> %433, float %430, i64 1, !dbg !25
  %435 = insertelement <4 x float> %434, float %431, i64 2, !dbg !25
  %436 = insertelement <4 x float> %435, float %432, i64 3, !dbg !25
  %compare.local.base1361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 24) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1362 = or i32 %compare.local.base1361, 72, !dbg !23
  %compare.low1363 = icmp ult i32 %compare.local1362, %compare.low.base, !dbg !23
  %compare.result1364 = select i1 %compare.high.equal, i1 %compare.low1363, i1 %compare.high.order, !dbg !23
  %437 = select i1 %compare.result1364, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 25) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1366 = or i32 %compare.local.base1365, 73, !dbg !23
  %compare.low1367 = icmp ult i32 %compare.local1366, %compare.low.base, !dbg !23
  %compare.result1368 = select i1 %compare.high.equal, i1 %compare.low1367, i1 %compare.high.order, !dbg !23
  %438 = select i1 %compare.result1368, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 26) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1370 = or i32 %compare.local.base1369, 74, !dbg !23
  %compare.low1371 = icmp ult i32 %compare.local1370, %compare.low.base, !dbg !23
  %compare.result1372 = select i1 %compare.high.equal, i1 %compare.low1371, i1 %compare.high.order, !dbg !23
  %439 = select i1 %compare.result1372, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 27) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1374 = or i32 %compare.local.base1373, 75, !dbg !23
  %compare.low1375 = icmp ult i32 %compare.local1374, %compare.low.base, !dbg !23
  %compare.result1376 = select i1 %compare.high.equal, i1 %compare.low1375, i1 %compare.high.order, !dbg !23
  %440 = select i1 %compare.result1376, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %441 = insertelement <4 x float> poison, float %437, i64 0, !dbg !25
  %442 = insertelement <4 x float> %441, float %438, i64 1, !dbg !25
  %443 = insertelement <4 x float> %442, float %439, i64 2, !dbg !25
  %444 = insertelement <4 x float> %443, float %440, i64 3, !dbg !25
  %compare.local.base1377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 28) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1378 = or i32 %compare.local.base1377, 76, !dbg !23
  %compare.low1379 = icmp ult i32 %compare.local1378, %compare.low.base, !dbg !23
  %compare.result1380 = select i1 %compare.high.equal, i1 %compare.low1379, i1 %compare.high.order, !dbg !23
  %445 = select i1 %compare.result1380, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 29) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1382 = or i32 %compare.local.base1381, 77, !dbg !23
  %compare.low1383 = icmp ult i32 %compare.local1382, %compare.low.base, !dbg !23
  %compare.result1384 = select i1 %compare.high.equal, i1 %compare.low1383, i1 %compare.high.order, !dbg !23
  %446 = select i1 %compare.result1384, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1385 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 30) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1386 = or i32 %compare.local.base1385, 78, !dbg !23
  %compare.low1387 = icmp ult i32 %compare.local1386, %compare.low.base, !dbg !23
  %compare.result1388 = select i1 %compare.high.equal, i1 %compare.low1387, i1 %compare.high.order, !dbg !23
  %447 = select i1 %compare.result1388, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1389 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 31) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1390 = or i32 %compare.local.base1389, 79, !dbg !23
  %compare.low1391 = icmp ult i32 %compare.local1390, %compare.low.base, !dbg !23
  %compare.result1392 = select i1 %compare.high.equal, i1 %compare.low1391, i1 %compare.high.order, !dbg !23
  %448 = select i1 %compare.result1392, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %449 = insertelement <4 x float> poison, float %445, i64 0, !dbg !25
  %450 = insertelement <4 x float> %449, float %446, i64 1, !dbg !25
  %451 = insertelement <4 x float> %450, float %447, i64 2, !dbg !25
  %452 = insertelement <4 x float> %451, float %448, i64 3, !dbg !25
  %453 = lshr i64 %104, 32, !dbg !23
  %compare.high1393 = trunc nuw i64 %453 to i32, !dbg !23
  %compare.high.equal1395 = icmp eq i32 %compare.high1393, %compare.high1269, !dbg !23
  %compare.high.order1396 = icmp ugt i32 %compare.high1393, %compare.high1269, !dbg !23
  %compare.low.base1397 = trunc i64 %104 to i32, !dbg !23
  %compare.low1399 = icmp ugt i32 %compare.low.base1397, %compare.low.base1270, !dbg !23
  %compare.result1400 = select i1 %compare.high.equal1395, i1 %compare.low1399, i1 %compare.high.order1396, !dbg !23
  %454 = select i1 %compare.result1400, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1401 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 32) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1402 = or i32 %compare.local.base1401, 1, !dbg !23
  %compare.low1403 = icmp ult i32 %compare.local1402, %compare.low.base1397, !dbg !23
  %compare.result1404 = select i1 %compare.high.equal1395, i1 %compare.low1403, i1 %compare.high.order1396, !dbg !23
  %455 = select i1 %compare.result1404, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1405 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 33) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1406 = or i32 %compare.local.base1405, 2, !dbg !23
  %compare.low1407 = icmp ult i32 %compare.local1406, %compare.low.base1397, !dbg !23
  %compare.result1408 = select i1 %compare.high.equal1395, i1 %compare.low1407, i1 %compare.high.order1396, !dbg !23
  %456 = select i1 %compare.result1408, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1409 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 34) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1410 = or i32 %compare.local.base1409, 3, !dbg !23
  %compare.low1411 = icmp ult i32 %compare.local1410, %compare.low.base1397, !dbg !23
  %compare.result1412 = select i1 %compare.high.equal1395, i1 %compare.low1411, i1 %compare.high.order1396, !dbg !23
  %457 = select i1 %compare.result1412, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %458 = insertelement <4 x float> poison, float %454, i64 0, !dbg !25
  %459 = insertelement <4 x float> %458, float %455, i64 1, !dbg !25
  %460 = insertelement <4 x float> %459, float %456, i64 2, !dbg !25
  %461 = insertelement <4 x float> %460, float %457, i64 3, !dbg !25
  %compare.local.base1413 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 35) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1414 = or i32 %compare.local.base1413, 4, !dbg !23
  %compare.low1415 = icmp ult i32 %compare.local1414, %compare.low.base1397, !dbg !23
  %compare.result1416 = select i1 %compare.high.equal1395, i1 %compare.low1415, i1 %compare.high.order1396, !dbg !23
  %462 = select i1 %compare.result1416, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1417 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 36) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1418 = or i32 %compare.local.base1417, 5, !dbg !23
  %compare.low1419 = icmp ult i32 %compare.local1418, %compare.low.base1397, !dbg !23
  %compare.result1420 = select i1 %compare.high.equal1395, i1 %compare.low1419, i1 %compare.high.order1396, !dbg !23
  %463 = select i1 %compare.result1420, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1421 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 37) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1422 = or i32 %compare.local.base1421, 6, !dbg !23
  %compare.low1423 = icmp ult i32 %compare.local1422, %compare.low.base1397, !dbg !23
  %compare.result1424 = select i1 %compare.high.equal1395, i1 %compare.low1423, i1 %compare.high.order1396, !dbg !23
  %464 = select i1 %compare.result1424, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1425 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 38) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1426 = or i32 %compare.local.base1425, 7, !dbg !23
  %compare.low1427 = icmp ult i32 %compare.local1426, %compare.low.base1397, !dbg !23
  %compare.result1428 = select i1 %compare.high.equal1395, i1 %compare.low1427, i1 %compare.high.order1396, !dbg !23
  %465 = select i1 %compare.result1428, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %466 = insertelement <4 x float> poison, float %462, i64 0, !dbg !25
  %467 = insertelement <4 x float> %466, float %463, i64 1, !dbg !25
  %468 = insertelement <4 x float> %467, float %464, i64 2, !dbg !25
  %469 = insertelement <4 x float> %468, float %465, i64 3, !dbg !25
  %compare.local.base1429 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 39) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1430 = or i32 %compare.local.base1429, 8, !dbg !23
  %compare.low1431 = icmp ult i32 %compare.local1430, %compare.low.base1397, !dbg !23
  %compare.result1432 = select i1 %compare.high.equal1395, i1 %compare.low1431, i1 %compare.high.order1396, !dbg !23
  %470 = select i1 %compare.result1432, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1433 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 40) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1434 = or i32 %compare.local.base1433, 9, !dbg !23
  %compare.low1435 = icmp ult i32 %compare.local1434, %compare.low.base1397, !dbg !23
  %compare.result1436 = select i1 %compare.high.equal1395, i1 %compare.low1435, i1 %compare.high.order1396, !dbg !23
  %471 = select i1 %compare.result1436, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1437 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 41) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1438 = or i32 %compare.local.base1437, 10, !dbg !23
  %compare.low1439 = icmp ult i32 %compare.local1438, %compare.low.base1397, !dbg !23
  %compare.result1440 = select i1 %compare.high.equal1395, i1 %compare.low1439, i1 %compare.high.order1396, !dbg !23
  %472 = select i1 %compare.result1440, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1441 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 42) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1442 = or i32 %compare.local.base1441, 11, !dbg !23
  %compare.low1443 = icmp ult i32 %compare.local1442, %compare.low.base1397, !dbg !23
  %compare.result1444 = select i1 %compare.high.equal1395, i1 %compare.low1443, i1 %compare.high.order1396, !dbg !23
  %473 = select i1 %compare.result1444, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %474 = insertelement <4 x float> poison, float %470, i64 0, !dbg !25
  %475 = insertelement <4 x float> %474, float %471, i64 1, !dbg !25
  %476 = insertelement <4 x float> %475, float %472, i64 2, !dbg !25
  %477 = insertelement <4 x float> %476, float %473, i64 3, !dbg !25
  %compare.local.base1445 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 43) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1446 = or i32 %compare.local.base1445, 12, !dbg !23
  %compare.low1447 = icmp ult i32 %compare.local1446, %compare.low.base1397, !dbg !23
  %compare.result1448 = select i1 %compare.high.equal1395, i1 %compare.low1447, i1 %compare.high.order1396, !dbg !23
  %478 = select i1 %compare.result1448, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1449 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 44) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1450 = or i32 %compare.local.base1449, 13, !dbg !23
  %compare.low1451 = icmp ult i32 %compare.local1450, %compare.low.base1397, !dbg !23
  %compare.result1452 = select i1 %compare.high.equal1395, i1 %compare.low1451, i1 %compare.high.order1396, !dbg !23
  %479 = select i1 %compare.result1452, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1453 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 45) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1454 = or i32 %compare.local.base1453, 14, !dbg !23
  %compare.low1455 = icmp ult i32 %compare.local1454, %compare.low.base1397, !dbg !23
  %compare.result1456 = select i1 %compare.high.equal1395, i1 %compare.low1455, i1 %compare.high.order1396, !dbg !23
  %480 = select i1 %compare.result1456, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1457 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 46) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1458 = or i32 %compare.local.base1457, 15, !dbg !23
  %compare.low1459 = icmp ult i32 %compare.local1458, %compare.low.base1397, !dbg !23
  %compare.result1460 = select i1 %compare.high.equal1395, i1 %compare.low1459, i1 %compare.high.order1396, !dbg !23
  %481 = select i1 %compare.result1460, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %482 = insertelement <4 x float> poison, float %478, i64 0, !dbg !25
  %483 = insertelement <4 x float> %482, float %479, i64 1, !dbg !25
  %484 = insertelement <4 x float> %483, float %480, i64 2, !dbg !25
  %485 = insertelement <4 x float> %484, float %481, i64 3, !dbg !25
  %compare.local.base1461 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 47) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1462 = or i32 %compare.local.base1461, 64, !dbg !23
  %compare.low1463 = icmp ult i32 %compare.local1462, %compare.low.base1397, !dbg !23
  %compare.result1464 = select i1 %compare.high.equal1395, i1 %compare.low1463, i1 %compare.high.order1396, !dbg !23
  %486 = select i1 %compare.result1464, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1465 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 48) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1466 = or i32 %compare.local.base1465, 65, !dbg !23
  %compare.low1467 = icmp ult i32 %compare.local1466, %compare.low.base1397, !dbg !23
  %compare.result1468 = select i1 %compare.high.equal1395, i1 %compare.low1467, i1 %compare.high.order1396, !dbg !23
  %487 = select i1 %compare.result1468, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1469 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 49) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1470 = or i32 %compare.local.base1469, 66, !dbg !23
  %compare.low1471 = icmp ult i32 %compare.local1470, %compare.low.base1397, !dbg !23
  %compare.result1472 = select i1 %compare.high.equal1395, i1 %compare.low1471, i1 %compare.high.order1396, !dbg !23
  %488 = select i1 %compare.result1472, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1473 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 50) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1474 = or i32 %compare.local.base1473, 67, !dbg !23
  %compare.low1475 = icmp ult i32 %compare.local1474, %compare.low.base1397, !dbg !23
  %compare.result1476 = select i1 %compare.high.equal1395, i1 %compare.low1475, i1 %compare.high.order1396, !dbg !23
  %489 = select i1 %compare.result1476, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %490 = insertelement <4 x float> poison, float %486, i64 0, !dbg !25
  %491 = insertelement <4 x float> %490, float %487, i64 1, !dbg !25
  %492 = insertelement <4 x float> %491, float %488, i64 2, !dbg !25
  %493 = insertelement <4 x float> %492, float %489, i64 3, !dbg !25
  %compare.local.base1477 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 51) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1478 = or i32 %compare.local.base1477, 68, !dbg !23
  %compare.low1479 = icmp ult i32 %compare.local1478, %compare.low.base1397, !dbg !23
  %compare.result1480 = select i1 %compare.high.equal1395, i1 %compare.low1479, i1 %compare.high.order1396, !dbg !23
  %494 = select i1 %compare.result1480, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1481 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 52) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1482 = or i32 %compare.local.base1481, 69, !dbg !23
  %compare.low1483 = icmp ult i32 %compare.local1482, %compare.low.base1397, !dbg !23
  %compare.result1484 = select i1 %compare.high.equal1395, i1 %compare.low1483, i1 %compare.high.order1396, !dbg !23
  %495 = select i1 %compare.result1484, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1485 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 53) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1486 = or i32 %compare.local.base1485, 70, !dbg !23
  %compare.low1487 = icmp ult i32 %compare.local1486, %compare.low.base1397, !dbg !23
  %compare.result1488 = select i1 %compare.high.equal1395, i1 %compare.low1487, i1 %compare.high.order1396, !dbg !23
  %496 = select i1 %compare.result1488, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1489 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 54) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1490 = or i32 %compare.local.base1489, 71, !dbg !23
  %compare.low1491 = icmp ult i32 %compare.local1490, %compare.low.base1397, !dbg !23
  %compare.result1492 = select i1 %compare.high.equal1395, i1 %compare.low1491, i1 %compare.high.order1396, !dbg !23
  %497 = select i1 %compare.result1492, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %498 = insertelement <4 x float> poison, float %494, i64 0, !dbg !25
  %499 = insertelement <4 x float> %498, float %495, i64 1, !dbg !25
  %500 = insertelement <4 x float> %499, float %496, i64 2, !dbg !25
  %501 = insertelement <4 x float> %500, float %497, i64 3, !dbg !25
  %compare.local.base1493 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 55) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1494 = or i32 %compare.local.base1493, 72, !dbg !23
  %compare.low1495 = icmp ult i32 %compare.local1494, %compare.low.base1397, !dbg !23
  %compare.result1496 = select i1 %compare.high.equal1395, i1 %compare.low1495, i1 %compare.high.order1396, !dbg !23
  %502 = select i1 %compare.result1496, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1497 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 56) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1498 = or i32 %compare.local.base1497, 73, !dbg !23
  %compare.low1499 = icmp ult i32 %compare.local1498, %compare.low.base1397, !dbg !23
  %compare.result1500 = select i1 %compare.high.equal1395, i1 %compare.low1499, i1 %compare.high.order1396, !dbg !23
  %503 = select i1 %compare.result1500, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1501 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 57) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1502 = or i32 %compare.local.base1501, 74, !dbg !23
  %compare.low1503 = icmp ult i32 %compare.local1502, %compare.low.base1397, !dbg !23
  %compare.result1504 = select i1 %compare.high.equal1395, i1 %compare.low1503, i1 %compare.high.order1396, !dbg !23
  %504 = select i1 %compare.result1504, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1505 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 58) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1506 = or i32 %compare.local.base1505, 75, !dbg !23
  %compare.low1507 = icmp ult i32 %compare.local1506, %compare.low.base1397, !dbg !23
  %compare.result1508 = select i1 %compare.high.equal1395, i1 %compare.low1507, i1 %compare.high.order1396, !dbg !23
  %505 = select i1 %compare.result1508, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %506 = insertelement <4 x float> poison, float %502, i64 0, !dbg !25
  %507 = insertelement <4 x float> %506, float %503, i64 1, !dbg !25
  %508 = insertelement <4 x float> %507, float %504, i64 2, !dbg !25
  %509 = insertelement <4 x float> %508, float %505, i64 3, !dbg !25
  %compare.local.base1509 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 59) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1510 = or i32 %compare.local.base1509, 76, !dbg !23
  %compare.low1511 = icmp ult i32 %compare.local1510, %compare.low.base1397, !dbg !23
  %compare.result1512 = select i1 %compare.high.equal1395, i1 %compare.low1511, i1 %compare.high.order1396, !dbg !23
  %510 = select i1 %compare.result1512, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1513 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 60) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1514 = or i32 %compare.local.base1513, 77, !dbg !23
  %compare.low1515 = icmp ult i32 %compare.local1514, %compare.low.base1397, !dbg !23
  %compare.result1516 = select i1 %compare.high.equal1395, i1 %compare.low1515, i1 %compare.high.order1396, !dbg !23
  %511 = select i1 %compare.result1516, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1517 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 61) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1518 = or i32 %compare.local.base1517, 78, !dbg !23
  %compare.low1519 = icmp ult i32 %compare.local1518, %compare.low.base1397, !dbg !23
  %compare.result1520 = select i1 %compare.high.equal1395, i1 %compare.low1519, i1 %compare.high.order1396, !dbg !23
  %512 = select i1 %compare.result1520, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1521 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 62) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1522 = or i32 %compare.local.base1521, 79, !dbg !23
  %compare.low1523 = icmp ult i32 %compare.local1522, %compare.low.base1397, !dbg !23
  %compare.result1524 = select i1 %compare.high.equal1395, i1 %compare.low1523, i1 %compare.high.order1396, !dbg !23
  %513 = select i1 %compare.result1524, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %514 = insertelement <4 x float> poison, float %510, i64 0, !dbg !25
  %515 = insertelement <4 x float> %514, float %511, i64 1, !dbg !25
  %516 = insertelement <4 x float> %515, float %512, i64 2, !dbg !25
  %517 = insertelement <4 x float> %516, float %513, i64 3, !dbg !25
  %518 = load <4 x half>, ptr addrspace(3) %kdma.gep2182, align 8, !dbg !15, !frisk.k_tail_pair !41
  %519 = load <4 x half>, ptr addrspace(3) %kdma.gep2186, align 8, !dbg !15, !frisk.k_tail_pair !41
  %520 = load <4 x half>, ptr addrspace(3) %kdma.gep2183, align 8, !dbg !15, !frisk.k_tail_pair !42
  %521 = load <4 x half>, ptr addrspace(3) %kdma.gep2187, align 8, !dbg !15, !frisk.k_tail_pair !42
  %522 = load <4 x half>, ptr addrspace(3) %kdma.gep2184, align 8, !dbg !15, !frisk.k_tail_pair !43
  %523 = load <4 x half>, ptr addrspace(3) %kdma.gep2188, align 8, !dbg !15, !frisk.k_tail_pair !43
  %524 = load <4 x half>, ptr addrspace(3) %kdma.gep2185, align 8, !dbg !15, !frisk.k_tail_pair !44
  %525 = load <4 x half>, ptr addrspace(3) %kdma.gep2189, align 8, !dbg !15, !frisk.k_tail_pair !44
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %518, <4 x float> %385) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %520, <4 x float> %345) #9, !dbg !15
  %528 = fadd <4 x float> %526, %396, !dbg !25
  %529 = extractelement <4 x float> %528, i64 0, !dbg !25
  %530 = call float @llvm.maxnum.f32(float %529, float 0xFFF0000000000000), !dbg !26, !frisk.local_reduction !1
  %531 = extractelement <4 x float> %528, i64 1, !dbg !25
  %532 = extractelement <4 x float> %528, i64 2, !dbg !25
  %533 = extractelement <4 x float> %528, i64 3, !dbg !25
  %534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %522, <4 x float> %527) #9, !dbg !15
  %535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %524, <4 x float> %347) #9, !dbg !15
  %536 = fadd <4 x float> %534, %404, !dbg !25
  %537 = extractelement <4 x float> %536, i64 0, !dbg !25
  %local.reduce219 = call float @llvm.maxnum.f32(float %530, float %537), !dbg !26, !frisk.local_reduction !1
  %538 = extractelement <4 x float> %536, i64 1, !dbg !25
  %local.reduce220 = call float @llvm.maxnum.f32(float %531, float %538), !dbg !26, !frisk.local_reduction !1
  %539 = extractelement <4 x float> %536, i64 2, !dbg !25
  %local.reduce221 = call float @llvm.maxnum.f32(float %532, float %539), !dbg !26, !frisk.local_reduction !1
  %540 = extractelement <4 x float> %536, i64 3, !dbg !25
  %local.reduce222 = call float @llvm.maxnum.f32(float %533, float %540), !dbg !26, !frisk.local_reduction !1
  %541 = load <4 x half>, ptr addrspace(3) %kdma.gep2190, align 8, !dbg !15, !frisk.k_tail_pair !39
  %542 = load <4 x half>, ptr addrspace(3) %kdma.gep2194, align 8, !dbg !15, !frisk.k_tail_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %519, <4 x float> %535) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %544 = load <4 x half>, ptr addrspace(3) %kdma.gep2191, align 8, !dbg !15, !frisk.k_tail_pair !45
  %545 = load <4 x half>, ptr addrspace(3) %kdma.gep2195, align 8, !dbg !15, !frisk.k_tail_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %521, <4 x float> %349) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25, !frisk.k_tail_prefetch !1
  %547 = fadd <4 x float> %543, %412, !dbg !25
  %548 = extractelement <4 x float> %547, i64 0, !dbg !25
  %local.reduce223 = call float @llvm.maxnum.f32(float %local.reduce219, float %548), !dbg !26, !frisk.local_reduction !1
  %549 = extractelement <4 x float> %547, i64 1, !dbg !25
  %local.reduce224 = call float @llvm.maxnum.f32(float %local.reduce220, float %549), !dbg !26, !frisk.local_reduction !1
  %550 = extractelement <4 x float> %547, i64 2, !dbg !25
  %local.reduce225 = call float @llvm.maxnum.f32(float %local.reduce221, float %550), !dbg !26, !frisk.local_reduction !1
  %551 = extractelement <4 x float> %547, i64 3, !dbg !25
  %local.reduce226 = call float @llvm.maxnum.f32(float %local.reduce222, float %551), !dbg !26, !frisk.local_reduction !1
  %552 = load <4 x half>, ptr addrspace(3) %kdma.gep2192, align 8, !dbg !15, !frisk.k_tail_pair !46
  %553 = load <4 x half>, ptr addrspace(3) %kdma.gep2196, align 8, !dbg !15, !frisk.k_tail_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %523, <4 x float> %546) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %525, <4 x float> %351) #9, !dbg !15
  %556 = fadd <4 x float> %554, %420, !dbg !25
  %557 = extractelement <4 x float> %556, i64 0, !dbg !25
  %local.reduce227 = call float @llvm.maxnum.f32(float %local.reduce223, float %557), !dbg !26, !frisk.local_reduction !1
  %558 = extractelement <4 x float> %556, i64 1, !dbg !25
  %local.reduce228 = call float @llvm.maxnum.f32(float %local.reduce224, float %558), !dbg !26, !frisk.local_reduction !1
  %559 = extractelement <4 x float> %556, i64 2, !dbg !25
  %local.reduce229 = call float @llvm.maxnum.f32(float %local.reduce225, float %559), !dbg !26, !frisk.local_reduction !1
  %560 = extractelement <4 x float> %556, i64 3, !dbg !25
  %local.reduce230 = call float @llvm.maxnum.f32(float %local.reduce226, float %560), !dbg !26, !frisk.local_reduction !1
  %561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %541, <4 x float> %555) #9, !dbg !15
  %562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %544, <4 x float> %353) #9, !dbg !15
  %563 = fadd <4 x float> %561, %428, !dbg !25
  %564 = extractelement <4 x float> %563, i64 0, !dbg !25
  %local.reduce231 = call float @llvm.maxnum.f32(float %local.reduce227, float %564), !dbg !26, !frisk.local_reduction !1
  %565 = extractelement <4 x float> %563, i64 1, !dbg !25
  %local.reduce232 = call float @llvm.maxnum.f32(float %local.reduce228, float %565), !dbg !26, !frisk.local_reduction !1
  %566 = extractelement <4 x float> %563, i64 2, !dbg !25
  %local.reduce233 = call float @llvm.maxnum.f32(float %local.reduce229, float %566), !dbg !26, !frisk.local_reduction !1
  %567 = extractelement <4 x float> %563, i64 3, !dbg !25
  %local.reduce234 = call float @llvm.maxnum.f32(float %local.reduce230, float %567), !dbg !26, !frisk.local_reduction !1
  %568 = load <4 x half>, ptr addrspace(3) %kdma.gep2193, align 8, !dbg !15
  %569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %552, <4 x float> %562) #9, !dbg !15
  %570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %568, <4 x float> %355) #9, !dbg !15
  %571 = fadd <4 x float> %569, %436, !dbg !25
  %572 = extractelement <4 x float> %571, i64 0, !dbg !25
  %local.reduce235 = call float @llvm.maxnum.f32(float %local.reduce231, float %572), !dbg !26, !frisk.local_reduction !1
  %573 = extractelement <4 x float> %571, i64 1, !dbg !25
  %local.reduce236 = call float @llvm.maxnum.f32(float %local.reduce232, float %573), !dbg !26, !frisk.local_reduction !1
  %574 = extractelement <4 x float> %571, i64 2, !dbg !25
  %local.reduce237 = call float @llvm.maxnum.f32(float %local.reduce233, float %574), !dbg !26, !frisk.local_reduction !1
  %575 = extractelement <4 x float> %571, i64 3, !dbg !25
  %local.reduce238 = call float @llvm.maxnum.f32(float %local.reduce234, float %575), !dbg !26, !frisk.local_reduction !1
  %576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %542, <4 x float> %570) #9, !dbg !15
  %577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %545, <4 x float> %357) #9, !dbg !15
  %578 = fadd <4 x float> %576, %444, !dbg !25
  %579 = extractelement <4 x float> %578, i64 0, !dbg !25
  %local.reduce239 = call float @llvm.maxnum.f32(float %local.reduce235, float %579), !dbg !26, !frisk.local_reduction !1
  %580 = extractelement <4 x float> %578, i64 1, !dbg !25
  %local.reduce240 = call float @llvm.maxnum.f32(float %local.reduce236, float %580), !dbg !26, !frisk.local_reduction !1
  %581 = extractelement <4 x float> %578, i64 2, !dbg !25
  %local.reduce241 = call float @llvm.maxnum.f32(float %local.reduce237, float %581), !dbg !26, !frisk.local_reduction !1
  %582 = extractelement <4 x float> %578, i64 3, !dbg !25
  %local.reduce242 = call float @llvm.maxnum.f32(float %local.reduce238, float %582), !dbg !26, !frisk.local_reduction !1
  %583 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %553, <4 x float> %577) #9, !dbg !15
  %584 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %384, <4 x float> %359) #9, !dbg !15
  %585 = fadd <4 x float> %583, %452, !dbg !25
  %586 = extractelement <4 x float> %585, i64 0, !dbg !25
  %local.reduce243 = call float @llvm.maxnum.f32(float %local.reduce239, float %586), !dbg !26, !frisk.local_reduction !1
  %587 = extractelement <4 x float> %585, i64 1, !dbg !25
  %local.reduce244 = call float @llvm.maxnum.f32(float %local.reduce240, float %587), !dbg !26, !frisk.local_reduction !1
  %588 = extractelement <4 x float> %585, i64 2, !dbg !25
  %local.reduce245 = call float @llvm.maxnum.f32(float %local.reduce241, float %588), !dbg !26, !frisk.local_reduction !1
  %589 = extractelement <4 x float> %585, i64 3, !dbg !25
  %local.reduce246 = call float @llvm.maxnum.f32(float %local.reduce242, float %589), !dbg !26, !frisk.local_reduction !1
  %local.reduce247 = call float @llvm.maxnum.f32(float %local.reduce243, float %local.reduce244), !dbg !26, !frisk.local_reduction !1
  %local.reduce248 = call float @llvm.maxnum.f32(float %local.reduce245, float %local.reduce246), !dbg !26, !frisk.local_reduction !1
  %590 = call float @llvm.maxnum.f32(float %local.reduce247, float %local.reduce248), !dbg !26, !frisk.local_reduction !1
  %591 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %518, <4 x float> %584) #9, !dbg !15
  %592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %520, <4 x float> %361) #9, !dbg !15
  %593 = fadd <4 x float> %591, %461, !dbg !25
  %594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %522, <4 x float> %592) #9, !dbg !15
  %595 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %524, <4 x float> %363) #9, !dbg !15
  %596 = fadd <4 x float> %594, %469, !dbg !25
  %597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %519, <4 x float> %595) #9, !dbg !15
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %521, <4 x float> %365) #9, !dbg !15
  %599 = fadd <4 x float> %597, %477, !dbg !25
  %600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %523, <4 x float> %598) #9, !dbg !15
  %601 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %525, <4 x float> %367) #9, !dbg !15
  %602 = fadd <4 x float> %600, %485, !dbg !25
  %603 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %541, <4 x float> %601) #9, !dbg !15
  %604 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %544, <4 x float> %369) #9, !dbg !15
  %605 = fadd <4 x float> %603, %493, !dbg !25
  %606 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %552, <4 x float> %604) #9, !dbg !15
  %607 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %568, <4 x float> %371) #9, !dbg !15
  %608 = fadd <4 x float> %606, %501, !dbg !25
  %609 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %542, <4 x float> %607) #9, !dbg !15
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %545, <4 x float> %373) #9, !dbg !15
  %611 = fadd <4 x float> %609, %509, !dbg !25
  %612 = extractelement <4 x float> %593, i64 0, !dbg !25
  %613 = extractelement <4 x float> %593, i64 1, !dbg !25
  %614 = extractelement <4 x float> %593, i64 2, !dbg !25
  %615 = extractelement <4 x float> %593, i64 3, !dbg !25
  %616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %553, <4 x float> %610) #9, !dbg !15
  %617 = extractelement <4 x float> %596, i64 0, !dbg !25
  %618 = extractelement <4 x float> %596, i64 1, !dbg !25
  %619 = extractelement <4 x float> %596, i64 2, !dbg !25
  %620 = extractelement <4 x float> %596, i64 3, !dbg !25
  %621 = extractelement <4 x float> %599, i64 0, !dbg !25
  %622 = extractelement <4 x float> %599, i64 1, !dbg !25
  %623 = extractelement <4 x float> %599, i64 2, !dbg !25
  %624 = extractelement <4 x float> %599, i64 3, !dbg !25
  %625 = fadd <4 x float> %616, %517, !dbg !25
  %626 = extractelement <4 x float> %602, i64 0, !dbg !25
  %627 = extractelement <4 x float> %602, i64 1, !dbg !25
  %628 = extractelement <4 x float> %602, i64 2, !dbg !25
  %629 = extractelement <4 x float> %602, i64 3, !dbg !25
  %630 = extractelement <4 x float> %605, i64 0, !dbg !25
  %631 = extractelement <4 x float> %605, i64 1, !dbg !25
  %632 = extractelement <4 x float> %605, i64 2, !dbg !25
  %633 = extractelement <4 x float> %605, i64 3, !dbg !25
  %634 = extractelement <4 x float> %608, i64 0, !dbg !25
  %635 = extractelement <4 x float> %608, i64 1, !dbg !25
  %636 = extractelement <4 x float> %608, i64 2, !dbg !25
  %637 = extractelement <4 x float> %608, i64 3, !dbg !25
  %638 = extractelement <4 x float> %611, i64 0, !dbg !25
  %639 = extractelement <4 x float> %611, i64 1, !dbg !25
  %640 = extractelement <4 x float> %611, i64 2, !dbg !25
  %641 = extractelement <4 x float> %611, i64 3, !dbg !25
  %642 = extractelement <4 x float> %625, i64 0, !dbg !25
  %643 = extractelement <4 x float> %625, i64 1, !dbg !25
  %644 = extractelement <4 x float> %625, i64 2, !dbg !25
  %645 = extractelement <4 x float> %625, i64 3, !dbg !25
  %646 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %647 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %646), !dbg !26
  %648 = add nuw nsw i32 %647, 64, !dbg !26
  %649 = xor i32 %647, 16, !dbg !26
  %650 = and i32 %648, 192, !dbg !26
  %651 = icmp samesign ult i32 %649, %650, !dbg !26
  %652 = select i1 %651, i32 %649, i32 %647, !dbg !26
  %653 = shl nuw nsw i32 %652, 2, !dbg !26
  %654 = bitcast float %590 to i32, !dbg !26
  %655 = call i32 @llvm.amdgcn.ds.bpermute(i32 %653, i32 %654), !dbg !26
  %656 = bitcast i32 %655 to float, !dbg !26
  %657 = call float @llvm.maxnum.f32(float %590, float %656), !dbg !26
  %658 = xor i32 %647, 32, !dbg !26
  %659 = icmp samesign ult i32 %658, %650, !dbg !26
  %660 = select i1 %659, i32 %658, i32 %647, !dbg !26
  %661 = shl nuw nsw i32 %660, 2, !dbg !26
  %662 = bitcast float %657 to i32, !dbg !26
  %663 = call i32 @llvm.amdgcn.ds.bpermute(i32 %661, i32 %662), !dbg !26
  %664 = bitcast i32 %663 to float, !dbg !26
  %665 = call float @llvm.maxnum.f32(float %657, float %664), !dbg !26
  %666 = extractvalue [2 x <1 x float>] %109, 0, !dbg !26
  %667 = insertelement <1 x float> poison, float %665, i64 0, !dbg !26
  %668 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %667, <1 x float> %666), !dbg !26
  %669 = call float @llvm.maxnum.f32(float %612, float 0xFFF0000000000000), !dbg !26, !frisk.local_reduction !1
  %local.reduce188 = call float @llvm.maxnum.f32(float %669, float %617), !dbg !26, !frisk.local_reduction !1
  %local.reduce189 = call float @llvm.maxnum.f32(float %613, float %618), !dbg !26, !frisk.local_reduction !1
  %local.reduce190 = call float @llvm.maxnum.f32(float %614, float %619), !dbg !26, !frisk.local_reduction !1
  %local.reduce191 = call float @llvm.maxnum.f32(float %615, float %620), !dbg !26, !frisk.local_reduction !1
  %local.reduce192 = call float @llvm.maxnum.f32(float %local.reduce188, float %621), !dbg !26, !frisk.local_reduction !1
  %local.reduce193 = call float @llvm.maxnum.f32(float %local.reduce189, float %622), !dbg !26, !frisk.local_reduction !1
  %local.reduce194 = call float @llvm.maxnum.f32(float %local.reduce190, float %623), !dbg !26, !frisk.local_reduction !1
  %local.reduce195 = call float @llvm.maxnum.f32(float %local.reduce191, float %624), !dbg !26, !frisk.local_reduction !1
  %local.reduce196 = call float @llvm.maxnum.f32(float %local.reduce192, float %626), !dbg !26, !frisk.local_reduction !1
  %local.reduce197 = call float @llvm.maxnum.f32(float %local.reduce193, float %627), !dbg !26, !frisk.local_reduction !1
  %local.reduce198 = call float @llvm.maxnum.f32(float %local.reduce194, float %628), !dbg !26, !frisk.local_reduction !1
  %local.reduce199 = call float @llvm.maxnum.f32(float %local.reduce195, float %629), !dbg !26, !frisk.local_reduction !1
  %local.reduce200 = call float @llvm.maxnum.f32(float %local.reduce196, float %630), !dbg !26, !frisk.local_reduction !1
  %local.reduce201 = call float @llvm.maxnum.f32(float %local.reduce197, float %631), !dbg !26, !frisk.local_reduction !1
  %local.reduce202 = call float @llvm.maxnum.f32(float %local.reduce198, float %632), !dbg !26, !frisk.local_reduction !1
  %local.reduce203 = call float @llvm.maxnum.f32(float %local.reduce199, float %633), !dbg !26, !frisk.local_reduction !1
  %local.reduce204 = call float @llvm.maxnum.f32(float %local.reduce200, float %634), !dbg !26, !frisk.local_reduction !1
  %local.reduce205 = call float @llvm.maxnum.f32(float %local.reduce201, float %635), !dbg !26, !frisk.local_reduction !1
  %local.reduce206 = call float @llvm.maxnum.f32(float %local.reduce202, float %636), !dbg !26, !frisk.local_reduction !1
  %local.reduce207 = call float @llvm.maxnum.f32(float %local.reduce203, float %637), !dbg !26, !frisk.local_reduction !1
  %local.reduce208 = call float @llvm.maxnum.f32(float %local.reduce204, float %638), !dbg !26, !frisk.local_reduction !1
  %local.reduce209 = call float @llvm.maxnum.f32(float %local.reduce205, float %639), !dbg !26, !frisk.local_reduction !1
  %local.reduce210 = call float @llvm.maxnum.f32(float %local.reduce206, float %640), !dbg !26, !frisk.local_reduction !1
  %local.reduce211 = call float @llvm.maxnum.f32(float %local.reduce207, float %641), !dbg !26, !frisk.local_reduction !1
  %local.reduce212 = call float @llvm.maxnum.f32(float %local.reduce208, float %642), !dbg !26, !frisk.local_reduction !1
  %local.reduce213 = call float @llvm.maxnum.f32(float %local.reduce209, float %643), !dbg !26, !frisk.local_reduction !1
  %local.reduce214 = call float @llvm.maxnum.f32(float %local.reduce210, float %644), !dbg !26, !frisk.local_reduction !1
  %local.reduce215 = call float @llvm.maxnum.f32(float %local.reduce211, float %645), !dbg !26, !frisk.local_reduction !1
  %local.reduce216 = call float @llvm.maxnum.f32(float %local.reduce212, float %local.reduce213), !dbg !26, !frisk.local_reduction !1
  %local.reduce217 = call float @llvm.maxnum.f32(float %local.reduce214, float %local.reduce215), !dbg !26, !frisk.local_reduction !1
  %670 = call float @llvm.maxnum.f32(float %local.reduce216, float %local.reduce217), !dbg !26, !frisk.local_reduction !1
  %671 = bitcast float %670 to i32, !dbg !26
  %672 = call i32 @llvm.amdgcn.ds.bpermute(i32 %653, i32 %671), !dbg !26
  %673 = bitcast i32 %672 to float, !dbg !26
  %674 = call float @llvm.maxnum.f32(float %670, float %673), !dbg !26
  %675 = bitcast float %674 to i32, !dbg !26
  %676 = call i32 @llvm.amdgcn.ds.bpermute(i32 %661, i32 %675), !dbg !26
  %677 = bitcast i32 %676 to float, !dbg !26
  %678 = call float @llvm.maxnum.f32(float %674, float %677), !dbg !26
  %679 = extractvalue [2 x <1 x float>] %109, 1, !dbg !26
  %680 = insertelement <1 x float> poison, float %678, i64 0, !dbg !26
  %681 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %680, <1 x float> %679), !dbg !26
  %682 = shufflevector <1 x float> %668, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %683 = shufflevector <1 x float> %681, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %684 = fsub <1 x float> %666, %668, !dbg !28
  %685 = extractelement <1 x float> %684, i64 0, !dbg !29
  %686 = call float @llvm.amdgcn.exp2.f32(float %685), !dbg !29
  %687 = insertelement <4 x float> poison, float %686, i64 0, !dbg !34
  %688 = shufflevector <4 x float> %687, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %689 = extractvalue [2 x <1 x float>] %108, 0, !dbg !30
  %690 = extractelement <1 x float> %689, i64 0, !dbg !30
  %691 = fmul float %690, %686, !dbg !30
  %692 = fsub <1 x float> %679, %681, !dbg !28
  %693 = extractelement <1 x float> %692, i64 0, !dbg !29
  %694 = call float @llvm.amdgcn.exp2.f32(float %693), !dbg !29
  %695 = insertelement <4 x float> poison, float %694, i64 0, !dbg !34
  %696 = shufflevector <4 x float> %695, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %697 = extractvalue [2 x <1 x float>] %108, 1, !dbg !30
  %698 = extractelement <1 x float> %697, i64 0, !dbg !30
  %699 = fmul float %698, %694, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1802 = or disjoint i64 %.idx17771779, 32, !dbg !16
  %vdb.address1803 = or disjoint i64 %address.local989, %vdb.address1802, !dbg !16
  %vdb.address1804 = trunc i64 %vdb.address1803 to i32, !dbg !16
  %vdb.address1805 = shl i32 %vdb.address1804, 1, !dbg !16
  %vdb.address1813 = or disjoint i64 %address.local995, %vdb.address1802, !dbg !16
  %vdb.address1814 = trunc i64 %vdb.address1813 to i32, !dbg !16
  %vdb.address1815 = shl i32 %vdb.address1814, 1, !dbg !16
  %vdb.dma.base = add i32 %dma.buffer.lds1702, 8192, !dbg !13
  %700 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1805, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %vdb.dma.base1820 = add i32 %dma.buffer.lds1699, 8192, !dbg !13
  %701 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1820, i32 %vdb.address1815, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %address.local997 = shl nuw nsw i32 %address.thread566, 3, !dbg !13
  %address.local998 = and i32 %address.local997, 120, !dbg !13
  %address.local1000 = and i32 %address.local643, 1536, !dbg !13
  %address.local1001 = or disjoint i32 %address.local998, %address.local1000, !dbg !13
  %address.local1002 = zext nneg i32 %address.local1001 to i64, !dbg !13
  %address.local1003 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1002, !dbg !13
  %702 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1003, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %703 = bitcast <4 x i32> %702 to <8 x half>, !dbg !13
  %address.local1004 = and i64 %address.local567, 60, !dbg !13
  %address.local1005 = add nuw nsw i64 %address.local1004, %address.local567, !dbg !13
  %address.local1007.idx = shl nuw nsw i64 %address.local1005, 4, !dbg !13
  %address.local1007 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local1007.idx, !dbg !13
  %address.local1008 = lshr i64 %address.local623, 4, !dbg !13
  %address.local1010.idx = mul nuw nsw i64 %address.local1008, 768, !dbg !13
  %address.local1010 = getelementptr i8, ptr addrspace(3) %address.local1007, i64 %address.local1010.idx, !dbg !13
  %address.local1012.idx = mul nsw i64 %address.local610, -64, !dbg !13
  %address.local1012 = getelementptr i8, ptr addrspace(3) %address.local1010, i64 %address.local1012.idx, !dbg !13
  %address.local1013 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 256, !dbg !13
  %704 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1013, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %705 = bitcast <4 x i32> %704 to <8 x half>, !dbg !13
  %address.local1017 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 512, !dbg !13
  %706 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1017, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %707 = bitcast <4 x i32> %706 to <8 x half>, !dbg !13
  %address.local1021 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 768, !dbg !13
  %708 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1021, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %709 = bitcast <4 x i32> %708 to <8 x half>, !dbg !13
  %address.local1023 = getelementptr i8, ptr addrspace(3) %address.local1003, i64 4096, !dbg !13
  %710 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1023, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %711 = bitcast <4 x i32> %710 to <8 x half>, !dbg !13
  %address.local1027 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 4352, !dbg !13
  %712 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %713 = bitcast <4 x i32> %712 to <8 x half>, !dbg !13
  %address.local1031 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 4608, !dbg !13
  %714 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1031, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %715 = bitcast <4 x i32> %714 to <8 x half>, !dbg !13
  %address.local1035 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 4864, !dbg !13
  %716 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1035, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %717 = shufflevector <8 x half> %703, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %718 = extractvalue [2 x <8 x float>] %107, 0, !dbg !34
  %719 = shufflevector <8 x float> %718, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %720 = fmul <4 x float> %719, %688, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %721 = fsub <4 x float> %528, %682, !dbg !27
  %722 = extractelement <4 x float> %721, i64 0, !dbg !31
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !31
  %724 = insertelement <4 x float> poison, float %723, i64 0, !dbg !33
  %725 = extractelement <4 x float> %721, i64 1, !dbg !31
  %726 = call float @llvm.amdgcn.exp2.f32(float %725), !dbg !31
  %727 = insertelement <4 x float> %724, float %726, i64 1, !dbg !33
  %728 = extractelement <4 x float> %721, i64 2, !dbg !31
  %729 = call float @llvm.amdgcn.exp2.f32(float %728), !dbg !31
  %730 = insertelement <4 x float> %727, float %729, i64 2, !dbg !33
  %731 = extractelement <4 x float> %721, i64 3, !dbg !31
  %732 = call float @llvm.amdgcn.exp2.f32(float %731), !dbg !31
  %733 = insertelement <4 x float> %730, float %732, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %733) #9, !dbg !33, !frisk.register_lifetimes !1
  %734 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %734) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %735 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %717, <4 x float> %720) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %736 = shufflevector <8 x half> %703, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %737 = shufflevector <8 x float> %718, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %738 = fmul <4 x float> %737, %688, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %739 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %736, <4 x float> %738) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %740 = extractvalue [2 x <8 x float>] %107, 1, !dbg !34
  %741 = shufflevector <8 x float> %740, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %742 = fmul <4 x float> %741, %696, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %743 = fsub <4 x float> %593, %683, !dbg !27
  %744 = extractelement <4 x float> %743, i64 0, !dbg !31
  %745 = call float @llvm.amdgcn.exp2.f32(float %744), !dbg !31
  %746 = insertelement <4 x float> poison, float %745, i64 0, !dbg !33
  %747 = extractelement <4 x float> %743, i64 1, !dbg !31
  %748 = call float @llvm.amdgcn.exp2.f32(float %747), !dbg !31
  %749 = insertelement <4 x float> %746, float %748, i64 1, !dbg !33
  %750 = extractelement <4 x float> %743, i64 2, !dbg !31
  %751 = call float @llvm.amdgcn.exp2.f32(float %750), !dbg !31
  %752 = insertelement <4 x float> %749, float %751, i64 2, !dbg !33
  %753 = extractelement <4 x float> %743, i64 3, !dbg !31
  %754 = call float @llvm.amdgcn.exp2.f32(float %753), !dbg !31
  %755 = insertelement <4 x float> %752, float %754, i64 3, !dbg !33
  %math.result456 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %755) #9, !dbg !33, !frisk.register_lifetimes !1
  %756 = fptrunc <4 x float> %math.result456 to <4 x half>, !dbg !33
  %math.packed457 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %756) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %717, <4 x float> %742) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %758 = shufflevector <8 x float> %740, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %759 = fmul <4 x float> %758, %696, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %760 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %736, <4 x float> %759) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %761 = shufflevector <8 x half> %705, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %762 = fsub <4 x float> %536, %682, !dbg !27
  %763 = extractelement <4 x float> %762, i64 0, !dbg !31
  %764 = call float @llvm.amdgcn.exp2.f32(float %763), !dbg !31
  %local.reduce157 = fadd float %723, %764, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %local.reduce157) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %765 = insertelement <4 x float> poison, float %764, i64 0, !dbg !33
  %766 = extractelement <4 x float> %762, i64 1, !dbg !31
  %767 = call float @llvm.amdgcn.exp2.f32(float %766), !dbg !31
  %local.reduce158 = fadd float %726, %767, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2284 = call float asm sideeffect "", "=v,0"(float %local.reduce158) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %768 = insertelement <4 x float> %765, float %767, i64 1, !dbg !33
  %769 = extractelement <4 x float> %762, i64 2, !dbg !31
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !31
  %local.reduce159 = fadd float %729, %770, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2285 = call float asm sideeffect "", "=v,0"(float %local.reduce159) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %771 = insertelement <4 x float> %768, float %770, i64 2, !dbg !33
  %772 = extractelement <4 x float> %762, i64 3, !dbg !31
  %773 = call float @llvm.amdgcn.exp2.f32(float %772), !dbg !31
  %local.reduce160 = fadd float %732, %773, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2286 = call float asm sideeffect "", "=v,0"(float %local.reduce160) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %774 = insertelement <4 x float> %771, float %773, i64 3, !dbg !33
  %math.result442 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %774) #9, !dbg !33, !frisk.register_lifetimes !1
  %775 = fptrunc <4 x float> %math.result442 to <4 x half>, !dbg !33
  %math.packed443 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %775) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %776 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %761, <4 x float> %735) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %777 = shufflevector <8 x half> %705, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %778 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %777, <4 x float> %739) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %779 = fsub <4 x float> %596, %683, !dbg !27
  %780 = extractelement <4 x float> %779, i64 0, !dbg !31
  %781 = call float @llvm.amdgcn.exp2.f32(float %780), !dbg !31
  %local.reduce = fadd float %745, %781, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2287 = call float asm sideeffect "", "=v,0"(float %local.reduce) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %782 = insertelement <4 x float> poison, float %781, i64 0, !dbg !33
  %783 = extractelement <4 x float> %779, i64 1, !dbg !31
  %784 = call float @llvm.amdgcn.exp2.f32(float %783), !dbg !31
  %local.reduce127 = fadd float %748, %784, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2288 = call float asm sideeffect "", "=v,0"(float %local.reduce127) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %785 = insertelement <4 x float> %782, float %784, i64 1, !dbg !33
  %786 = extractelement <4 x float> %779, i64 2, !dbg !31
  %787 = call float @llvm.amdgcn.exp2.f32(float %786), !dbg !31
  %local.reduce128 = fadd float %751, %787, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2289 = call float asm sideeffect "", "=v,0"(float %local.reduce128) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %788 = insertelement <4 x float> %785, float %787, i64 2, !dbg !33
  %789 = extractelement <4 x float> %779, i64 3, !dbg !31
  %790 = call float @llvm.amdgcn.exp2.f32(float %789), !dbg !31
  %local.reduce129 = fadd float %754, %790, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2290 = call float asm sideeffect "", "=v,0"(float %local.reduce129) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %791 = insertelement <4 x float> %788, float %790, i64 3, !dbg !33
  %math.result458 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %791) #9, !dbg !33, !frisk.register_lifetimes !1
  %792 = fptrunc <4 x float> %math.result458 to <4 x half>, !dbg !33
  %math.packed459 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %792) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %793 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %761, <4 x float> %757) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %794 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %777, <4 x float> %760) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %795 = shufflevector <8 x half> %707, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %796 = fsub <4 x float> %547, %682, !dbg !27
  %797 = extractelement <4 x float> %796, i64 0, !dbg !31
  %798 = call float @llvm.amdgcn.exp2.f32(float %797), !dbg !31
  %local.reduce161 = fadd float %k0.sum.anchor, %798, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2291 = call float asm sideeffect "", "=v,0"(float %local.reduce161) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %799 = insertelement <4 x float> poison, float %798, i64 0, !dbg !33
  %800 = extractelement <4 x float> %796, i64 1, !dbg !31
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !31
  %local.reduce162 = fadd float %k0.sum.anchor2284, %801, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2292 = call float asm sideeffect "", "=v,0"(float %local.reduce162) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %802 = insertelement <4 x float> %799, float %801, i64 1, !dbg !33
  %803 = extractelement <4 x float> %796, i64 2, !dbg !31
  %804 = call float @llvm.amdgcn.exp2.f32(float %803), !dbg !31
  %local.reduce163 = fadd float %k0.sum.anchor2285, %804, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2293 = call float asm sideeffect "", "=v,0"(float %local.reduce163) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %805 = insertelement <4 x float> %802, float %804, i64 2, !dbg !33
  %806 = extractelement <4 x float> %796, i64 3, !dbg !31
  %807 = call float @llvm.amdgcn.exp2.f32(float %806), !dbg !31
  %local.reduce164 = fadd float %k0.sum.anchor2286, %807, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2294 = call float asm sideeffect "", "=v,0"(float %local.reduce164) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %808 = insertelement <4 x float> %805, float %807, i64 3, !dbg !33
  %math.result444 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %808) #9, !dbg !33, !frisk.register_lifetimes !1
  %809 = fptrunc <4 x float> %math.result444 to <4 x half>, !dbg !33
  %math.packed445 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %809) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %810 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %795, <4 x float> %776) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %811 = shufflevector <8 x half> %707, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %812 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %811, <4 x float> %778) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %813 = fsub <4 x float> %599, %683, !dbg !27
  %814 = extractelement <4 x float> %813, i64 0, !dbg !31
  %815 = call float @llvm.amdgcn.exp2.f32(float %814), !dbg !31
  %local.reduce130 = fadd float %k0.sum.anchor2287, %815, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2295 = call float asm sideeffect "", "=v,0"(float %local.reduce130) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %816 = insertelement <4 x float> poison, float %815, i64 0, !dbg !33
  %817 = extractelement <4 x float> %813, i64 1, !dbg !31
  %818 = call float @llvm.amdgcn.exp2.f32(float %817), !dbg !31
  %local.reduce131 = fadd float %k0.sum.anchor2288, %818, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2296 = call float asm sideeffect "", "=v,0"(float %local.reduce131) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %819 = insertelement <4 x float> %816, float %818, i64 1, !dbg !33
  %820 = extractelement <4 x float> %813, i64 2, !dbg !31
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !31
  %local.reduce132 = fadd float %k0.sum.anchor2289, %821, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2297 = call float asm sideeffect "", "=v,0"(float %local.reduce132) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %822 = insertelement <4 x float> %819, float %821, i64 2, !dbg !33
  %823 = extractelement <4 x float> %813, i64 3, !dbg !31
  %824 = call float @llvm.amdgcn.exp2.f32(float %823), !dbg !31
  %local.reduce133 = fadd float %k0.sum.anchor2290, %824, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2298 = call float asm sideeffect "", "=v,0"(float %local.reduce133) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %825 = insertelement <4 x float> %822, float %824, i64 3, !dbg !33
  %math.result460 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %825) #9, !dbg !33, !frisk.register_lifetimes !1
  %826 = fptrunc <4 x float> %math.result460 to <4 x half>, !dbg !33
  %math.packed461 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %826) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %795, <4 x float> %793) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %811, <4 x float> %794) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %829 = shufflevector <8 x half> %709, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %830 = fsub <4 x float> %556, %682, !dbg !27
  %831 = extractelement <4 x float> %830, i64 0, !dbg !31
  %832 = call float @llvm.amdgcn.exp2.f32(float %831), !dbg !31
  %local.reduce165 = fadd float %k0.sum.anchor2291, %832, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2299 = call float asm sideeffect "", "=v,0"(float %local.reduce165) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %833 = insertelement <4 x float> poison, float %832, i64 0, !dbg !33
  %834 = extractelement <4 x float> %830, i64 1, !dbg !31
  %835 = call float @llvm.amdgcn.exp2.f32(float %834), !dbg !31
  %local.reduce166 = fadd float %k0.sum.anchor2292, %835, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2300 = call float asm sideeffect "", "=v,0"(float %local.reduce166) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %836 = insertelement <4 x float> %833, float %835, i64 1, !dbg !33
  %837 = extractelement <4 x float> %830, i64 2, !dbg !31
  %838 = call float @llvm.amdgcn.exp2.f32(float %837), !dbg !31
  %local.reduce167 = fadd float %k0.sum.anchor2293, %838, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2301 = call float asm sideeffect "", "=v,0"(float %local.reduce167) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %839 = insertelement <4 x float> %836, float %838, i64 2, !dbg !33
  %840 = extractelement <4 x float> %830, i64 3, !dbg !31
  %841 = call float @llvm.amdgcn.exp2.f32(float %840), !dbg !31
  %local.reduce168 = fadd float %k0.sum.anchor2294, %841, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2302 = call float asm sideeffect "", "=v,0"(float %local.reduce168) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %842 = insertelement <4 x float> %839, float %841, i64 3, !dbg !33
  %math.result446 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %842) #9, !dbg !33, !frisk.register_lifetimes !1
  %843 = fptrunc <4 x float> %math.result446 to <4 x half>, !dbg !33
  %math.packed447 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %843) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %829, <4 x float> %810) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %845 = shufflevector <8 x half> %709, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %846 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %845, <4 x float> %812) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %847 = fsub <4 x float> %602, %683, !dbg !27
  %848 = extractelement <4 x float> %847, i64 0, !dbg !31
  %849 = call float @llvm.amdgcn.exp2.f32(float %848), !dbg !31
  %local.reduce134 = fadd float %k0.sum.anchor2295, %849, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2303 = call float asm sideeffect "", "=v,0"(float %local.reduce134) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %850 = insertelement <4 x float> poison, float %849, i64 0, !dbg !33
  %851 = extractelement <4 x float> %847, i64 1, !dbg !31
  %852 = call float @llvm.amdgcn.exp2.f32(float %851), !dbg !31
  %local.reduce135 = fadd float %k0.sum.anchor2296, %852, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2304 = call float asm sideeffect "", "=v,0"(float %local.reduce135) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %853 = insertelement <4 x float> %850, float %852, i64 1, !dbg !33
  %854 = extractelement <4 x float> %847, i64 2, !dbg !31
  %855 = call float @llvm.amdgcn.exp2.f32(float %854), !dbg !31
  %local.reduce136 = fadd float %k0.sum.anchor2297, %855, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2305 = call float asm sideeffect "", "=v,0"(float %local.reduce136) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %856 = insertelement <4 x float> %853, float %855, i64 2, !dbg !33
  %857 = extractelement <4 x float> %847, i64 3, !dbg !31
  %858 = call float @llvm.amdgcn.exp2.f32(float %857), !dbg !31
  %local.reduce137 = fadd float %k0.sum.anchor2298, %858, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2306 = call float asm sideeffect "", "=v,0"(float %local.reduce137) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %859 = insertelement <4 x float> %856, float %858, i64 3, !dbg !33
  %math.result462 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %859) #9, !dbg !33, !frisk.register_lifetimes !1
  %860 = fptrunc <4 x float> %math.result462 to <4 x half>, !dbg !33
  %math.packed463 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %860) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %829, <4 x float> %827) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %845, <4 x float> %828) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %863 = shufflevector <8 x half> %711, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %864 = fsub <4 x float> %563, %682, !dbg !27
  %865 = extractelement <4 x float> %864, i64 0, !dbg !31
  %866 = call float @llvm.amdgcn.exp2.f32(float %865), !dbg !31
  %local.reduce169 = fadd float %k0.sum.anchor2299, %866, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2307 = call float asm sideeffect "", "=v,0"(float %local.reduce169) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %867 = insertelement <4 x float> poison, float %866, i64 0, !dbg !33
  %868 = extractelement <4 x float> %864, i64 1, !dbg !31
  %869 = call float @llvm.amdgcn.exp2.f32(float %868), !dbg !31
  %local.reduce170 = fadd float %k0.sum.anchor2300, %869, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2308 = call float asm sideeffect "", "=v,0"(float %local.reduce170) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %870 = insertelement <4 x float> %867, float %869, i64 1, !dbg !33
  %871 = extractelement <4 x float> %864, i64 2, !dbg !31
  %872 = call float @llvm.amdgcn.exp2.f32(float %871), !dbg !31
  %local.reduce171 = fadd float %k0.sum.anchor2301, %872, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2309 = call float asm sideeffect "", "=v,0"(float %local.reduce171) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %873 = insertelement <4 x float> %870, float %872, i64 2, !dbg !33
  %874 = extractelement <4 x float> %864, i64 3, !dbg !31
  %875 = call float @llvm.amdgcn.exp2.f32(float %874), !dbg !31
  %local.reduce172 = fadd float %k0.sum.anchor2302, %875, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2310 = call float asm sideeffect "", "=v,0"(float %local.reduce172) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %876 = insertelement <4 x float> %873, float %875, i64 3, !dbg !33
  %math.result448 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %876) #9, !dbg !33, !frisk.register_lifetimes !1
  %877 = fptrunc <4 x float> %math.result448 to <4 x half>, !dbg !33
  %math.packed449 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %877) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %863, <4 x float> %844) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %879 = shufflevector <8 x half> %711, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %880 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %879, <4 x float> %846) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %881 = fsub <4 x float> %605, %683, !dbg !27
  %882 = extractelement <4 x float> %881, i64 0, !dbg !31
  %883 = call float @llvm.amdgcn.exp2.f32(float %882), !dbg !31
  %local.reduce138 = fadd float %k0.sum.anchor2303, %883, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2311 = call float asm sideeffect "", "=v,0"(float %local.reduce138) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %884 = insertelement <4 x float> poison, float %883, i64 0, !dbg !33
  %885 = extractelement <4 x float> %881, i64 1, !dbg !31
  %886 = call float @llvm.amdgcn.exp2.f32(float %885), !dbg !31
  %local.reduce139 = fadd float %k0.sum.anchor2304, %886, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2312 = call float asm sideeffect "", "=v,0"(float %local.reduce139) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %887 = insertelement <4 x float> %884, float %886, i64 1, !dbg !33
  %888 = extractelement <4 x float> %881, i64 2, !dbg !31
  %889 = call float @llvm.amdgcn.exp2.f32(float %888), !dbg !31
  %local.reduce140 = fadd float %k0.sum.anchor2305, %889, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2313 = call float asm sideeffect "", "=v,0"(float %local.reduce140) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %890 = insertelement <4 x float> %887, float %889, i64 2, !dbg !33
  %891 = extractelement <4 x float> %881, i64 3, !dbg !31
  %892 = call float @llvm.amdgcn.exp2.f32(float %891), !dbg !31
  %local.reduce141 = fadd float %k0.sum.anchor2306, %892, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2314 = call float asm sideeffect "", "=v,0"(float %local.reduce141) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %893 = insertelement <4 x float> %890, float %892, i64 3, !dbg !33
  %math.result464 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %893) #9, !dbg !33, !frisk.register_lifetimes !1
  %894 = fptrunc <4 x float> %math.result464 to <4 x half>, !dbg !33
  %math.packed465 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %894) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %863, <4 x float> %861) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %879, <4 x float> %862) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %897 = shufflevector <8 x half> %713, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %898 = fsub <4 x float> %571, %682, !dbg !27
  %899 = extractelement <4 x float> %898, i64 0, !dbg !31
  %900 = call float @llvm.amdgcn.exp2.f32(float %899), !dbg !31
  %local.reduce173 = fadd float %k0.sum.anchor2307, %900, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2315 = call float asm sideeffect "", "=v,0"(float %local.reduce173) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %901 = insertelement <4 x float> poison, float %900, i64 0, !dbg !33
  %902 = extractelement <4 x float> %898, i64 1, !dbg !31
  %903 = call float @llvm.amdgcn.exp2.f32(float %902), !dbg !31
  %local.reduce174 = fadd float %k0.sum.anchor2308, %903, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2316 = call float asm sideeffect "", "=v,0"(float %local.reduce174) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %904 = insertelement <4 x float> %901, float %903, i64 1, !dbg !33
  %905 = extractelement <4 x float> %898, i64 2, !dbg !31
  %906 = call float @llvm.amdgcn.exp2.f32(float %905), !dbg !31
  %local.reduce175 = fadd float %k0.sum.anchor2309, %906, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2317 = call float asm sideeffect "", "=v,0"(float %local.reduce175) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %907 = insertelement <4 x float> %904, float %906, i64 2, !dbg !33
  %908 = extractelement <4 x float> %898, i64 3, !dbg !31
  %909 = call float @llvm.amdgcn.exp2.f32(float %908), !dbg !31
  %local.reduce176 = fadd float %k0.sum.anchor2310, %909, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2318 = call float asm sideeffect "", "=v,0"(float %local.reduce176) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %910 = insertelement <4 x float> %907, float %909, i64 3, !dbg !33
  %math.result450 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %910) #9, !dbg !33, !frisk.register_lifetimes !1
  %911 = fptrunc <4 x float> %math.result450 to <4 x half>, !dbg !33
  %math.packed451 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %911) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %897, <4 x float> %878) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %913 = shufflevector <8 x half> %713, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %913, <4 x float> %880) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %915 = fsub <4 x float> %608, %683, !dbg !27
  %916 = extractelement <4 x float> %915, i64 0, !dbg !31
  %917 = call float @llvm.amdgcn.exp2.f32(float %916), !dbg !31
  %local.reduce142 = fadd float %k0.sum.anchor2311, %917, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2319 = call float asm sideeffect "", "=v,0"(float %local.reduce142) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %918 = insertelement <4 x float> poison, float %917, i64 0, !dbg !33
  %919 = extractelement <4 x float> %915, i64 1, !dbg !31
  %920 = call float @llvm.amdgcn.exp2.f32(float %919), !dbg !31
  %local.reduce143 = fadd float %k0.sum.anchor2312, %920, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2320 = call float asm sideeffect "", "=v,0"(float %local.reduce143) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %921 = insertelement <4 x float> %918, float %920, i64 1, !dbg !33
  %922 = extractelement <4 x float> %915, i64 2, !dbg !31
  %923 = call float @llvm.amdgcn.exp2.f32(float %922), !dbg !31
  %local.reduce144 = fadd float %k0.sum.anchor2313, %923, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2321 = call float asm sideeffect "", "=v,0"(float %local.reduce144) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %924 = insertelement <4 x float> %921, float %923, i64 2, !dbg !33
  %925 = extractelement <4 x float> %915, i64 3, !dbg !31
  %926 = call float @llvm.amdgcn.exp2.f32(float %925), !dbg !31
  %local.reduce145 = fadd float %k0.sum.anchor2314, %926, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2322 = call float asm sideeffect "", "=v,0"(float %local.reduce145) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %927 = insertelement <4 x float> %924, float %926, i64 3, !dbg !33
  %math.result466 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %927) #9, !dbg !33, !frisk.register_lifetimes !1
  %928 = fptrunc <4 x float> %math.result466 to <4 x half>, !dbg !33
  %math.packed467 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %928) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %897, <4 x float> %895) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %913, <4 x float> %896) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %931 = shufflevector <8 x half> %715, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %932 = fsub <4 x float> %578, %682, !dbg !27
  %933 = extractelement <4 x float> %932, i64 0, !dbg !31
  %934 = call float @llvm.amdgcn.exp2.f32(float %933), !dbg !31
  %local.reduce177 = fadd float %k0.sum.anchor2315, %934, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2323 = call float asm sideeffect "", "=v,0"(float %local.reduce177) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %935 = insertelement <4 x float> poison, float %934, i64 0, !dbg !33
  %936 = extractelement <4 x float> %932, i64 1, !dbg !31
  %937 = call float @llvm.amdgcn.exp2.f32(float %936), !dbg !31
  %local.reduce178 = fadd float %k0.sum.anchor2316, %937, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2324 = call float asm sideeffect "", "=v,0"(float %local.reduce178) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %938 = insertelement <4 x float> %935, float %937, i64 1, !dbg !33
  %939 = extractelement <4 x float> %932, i64 2, !dbg !31
  %940 = call float @llvm.amdgcn.exp2.f32(float %939), !dbg !31
  %local.reduce179 = fadd float %k0.sum.anchor2317, %940, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2325 = call float asm sideeffect "", "=v,0"(float %local.reduce179) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %941 = insertelement <4 x float> %938, float %940, i64 2, !dbg !33
  %942 = extractelement <4 x float> %932, i64 3, !dbg !31
  %943 = call float @llvm.amdgcn.exp2.f32(float %942), !dbg !31
  %local.reduce180 = fadd float %k0.sum.anchor2318, %943, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2326 = call float asm sideeffect "", "=v,0"(float %local.reduce180) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %944 = insertelement <4 x float> %941, float %943, i64 3, !dbg !33
  %math.result452 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %944) #9, !dbg !33, !frisk.register_lifetimes !1
  %945 = fptrunc <4 x float> %math.result452 to <4 x half>, !dbg !33
  %math.packed453 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %945) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %946 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %931, <4 x float> %912) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %947 = shufflevector <8 x half> %715, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %948 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %947, <4 x float> %914) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %949 = fsub <4 x float> %611, %683, !dbg !27
  %950 = extractelement <4 x float> %949, i64 0, !dbg !31
  %951 = call float @llvm.amdgcn.exp2.f32(float %950), !dbg !31
  %local.reduce146 = fadd float %k0.sum.anchor2319, %951, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2327 = call float asm sideeffect "", "=v,0"(float %local.reduce146) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %952 = insertelement <4 x float> poison, float %951, i64 0, !dbg !33
  %953 = extractelement <4 x float> %949, i64 1, !dbg !31
  %954 = call float @llvm.amdgcn.exp2.f32(float %953), !dbg !31
  %local.reduce147 = fadd float %k0.sum.anchor2320, %954, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2328 = call float asm sideeffect "", "=v,0"(float %local.reduce147) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %955 = insertelement <4 x float> %952, float %954, i64 1, !dbg !33
  %956 = extractelement <4 x float> %949, i64 2, !dbg !31
  %957 = call float @llvm.amdgcn.exp2.f32(float %956), !dbg !31
  %local.reduce148 = fadd float %k0.sum.anchor2321, %957, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2329 = call float asm sideeffect "", "=v,0"(float %local.reduce148) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %958 = insertelement <4 x float> %955, float %957, i64 2, !dbg !33
  %959 = extractelement <4 x float> %949, i64 3, !dbg !31
  %960 = call float @llvm.amdgcn.exp2.f32(float %959), !dbg !31
  %local.reduce149 = fadd float %k0.sum.anchor2322, %960, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2330 = call float asm sideeffect "", "=v,0"(float %local.reduce149) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %961 = insertelement <4 x float> %958, float %960, i64 3, !dbg !33
  %math.result468 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %961) #9, !dbg !33, !frisk.register_lifetimes !1
  %962 = fptrunc <4 x float> %math.result468 to <4 x half>, !dbg !33
  %math.packed469 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %962) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %931, <4 x float> %929) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %947, <4 x float> %930) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %965 = bitcast <4 x i32> %716 to <8 x half>, !dbg !13
  %966 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %967 = fsub <4 x float> %585, %682, !dbg !27
  %968 = extractelement <4 x float> %967, i64 0, !dbg !31
  %969 = call float @llvm.amdgcn.exp2.f32(float %968), !dbg !31
  %local.reduce181 = fadd float %k0.sum.anchor2323, %969, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2331 = call float asm sideeffect "", "=v,0"(float %local.reduce181) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %970 = insertelement <4 x float> poison, float %969, i64 0, !dbg !33
  %971 = extractelement <4 x float> %967, i64 1, !dbg !31
  %972 = call float @llvm.amdgcn.exp2.f32(float %971), !dbg !31
  %local.reduce182 = fadd float %k0.sum.anchor2324, %972, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2332 = call float asm sideeffect "", "=v,0"(float %local.reduce182) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %local.reduce185 = fadd float %k0.sum.anchor2331, %k0.sum.anchor2332, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2333 = call float asm sideeffect "", "=v,0"(float %local.reduce185) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %973 = insertelement <4 x float> %970, float %972, i64 1, !dbg !33
  %974 = extractelement <4 x float> %967, i64 2, !dbg !31
  %975 = call float @llvm.amdgcn.exp2.f32(float %974), !dbg !31
  %local.reduce183 = fadd float %k0.sum.anchor2325, %975, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2334 = call float asm sideeffect "", "=v,0"(float %local.reduce183) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %976 = insertelement <4 x float> %973, float %975, i64 2, !dbg !33
  %977 = extractelement <4 x float> %967, i64 3, !dbg !31
  %978 = call float @llvm.amdgcn.exp2.f32(float %977), !dbg !31
  %local.reduce184 = fadd float %k0.sum.anchor2326, %978, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2335 = call float asm sideeffect "", "=v,0"(float %local.reduce184) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %local.reduce186 = fadd float %k0.sum.anchor2334, %k0.sum.anchor2335, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2336 = call float asm sideeffect "", "=v,0"(float %local.reduce186) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %979 = fadd float %k0.sum.anchor2333, %k0.sum.anchor2336, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2337 = call float asm sideeffect "", "=v,0"(float %979) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %980 = fadd float %k0.sum.anchor2337, %691, !dbg !32
  %k0.sum.anchor2338 = call float asm sideeffect "", "=v,0"(float %980) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %981 = insertelement <1 x float> poison, float %k0.sum.anchor2338, i64 0, !dbg !7
  %982 = insertvalue [2 x <1 x float>] poison, <1 x float> %981, 0, !dbg !7
  %983 = insertelement <4 x float> %976, float %978, i64 3, !dbg !33
  %math.result454 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %983) #9, !dbg !33, !frisk.register_lifetimes !1
  %984 = fptrunc <4 x float> %math.result454 to <4 x half>, !dbg !33
  %math.packed455 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %984) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %966, <4 x float> %946) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %986 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %986, <4 x float> %948) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %988 = fsub <4 x float> %625, %683, !dbg !27
  %989 = extractelement <4 x float> %988, i64 0, !dbg !31
  %990 = call float @llvm.amdgcn.exp2.f32(float %989), !dbg !31
  %local.reduce150 = fadd float %k0.sum.anchor2327, %990, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2339 = call float asm sideeffect "", "=v,0"(float %local.reduce150) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %991 = insertelement <4 x float> poison, float %990, i64 0, !dbg !33
  %992 = extractelement <4 x float> %988, i64 1, !dbg !31
  %993 = call float @llvm.amdgcn.exp2.f32(float %992), !dbg !31
  %local.reduce151 = fadd float %k0.sum.anchor2328, %993, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2340 = call float asm sideeffect "", "=v,0"(float %local.reduce151) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %local.reduce154 = fadd float %k0.sum.anchor2339, %k0.sum.anchor2340, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2341 = call float asm sideeffect "", "=v,0"(float %local.reduce154) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %994 = insertelement <4 x float> %991, float %993, i64 1, !dbg !33
  %995 = extractelement <4 x float> %988, i64 2, !dbg !31
  %996 = call float @llvm.amdgcn.exp2.f32(float %995), !dbg !31
  %local.reduce152 = fadd float %k0.sum.anchor2329, %996, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2342 = call float asm sideeffect "", "=v,0"(float %local.reduce152) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %997 = insertelement <4 x float> %994, float %996, i64 2, !dbg !33
  %998 = extractelement <4 x float> %988, i64 3, !dbg !31
  %999 = call float @llvm.amdgcn.exp2.f32(float %998), !dbg !31
  %local.reduce153 = fadd float %k0.sum.anchor2330, %999, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2343 = call float asm sideeffect "", "=v,0"(float %local.reduce153) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %local.reduce155 = fadd float %k0.sum.anchor2342, %k0.sum.anchor2343, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2344 = call float asm sideeffect "", "=v,0"(float %local.reduce155) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %1000 = fadd float %k0.sum.anchor2341, %k0.sum.anchor2344, !dbg !32, !frisk.local_reduction !1
  %k0.sum.anchor2345 = call float asm sideeffect "", "=v,0"(float %1000) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %1001 = fadd float %k0.sum.anchor2345, %699, !dbg !32
  %k0.sum.anchor2346 = call float asm sideeffect "", "=v,0"(float %1001) #9, !dbg !32, !frisk.cross_bn_sum_anchor !1
  %1002 = insertelement <1 x float> poison, float %k0.sum.anchor2346, i64 0, !dbg !7
  %1003 = insertvalue [2 x <1 x float>] %982, <1 x float> %1002, 1, !dbg !7
  %1004 = insertelement <4 x float> %997, float %999, i64 3, !dbg !33
  %math.result470 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1004) #9, !dbg !33, !frisk.register_lifetimes !1
  %1005 = fptrunc <4 x float> %math.result470 to <4 x half>, !dbg !33
  %math.packed471 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1005) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %1006 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %966, <4 x float> %963) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1007 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %986, <4 x float> %964) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1830 = or disjoint i64 %.idx17771779, 64, !dbg !16
  %vdb.address1831 = or disjoint i64 %address.local989, %vdb.address1830, !dbg !16
  %vdb.address1832 = trunc i64 %vdb.address1831 to i32, !dbg !16
  %vdb.address1833 = shl i32 %vdb.address1832, 1, !dbg !16
  %vdb.address1841 = or disjoint i64 %address.local995, %vdb.address1830, !dbg !16
  %vdb.address1842 = trunc i64 %vdb.address1841 to i32, !dbg !16
  %vdb.address1843 = shl i32 %vdb.address1842, 1, !dbg !16
  %1008 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %vdb.address1833, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %1009 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %vdb.address1843, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local1003, i64 8192, !dbg !13
  %1010 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1011 = bitcast <4 x i32> %1010 to <8 x half>, !dbg !13
  %vdb.read1821 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 8448, !dbg !13
  %1012 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1821, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1013 = bitcast <4 x i32> %1012 to <8 x half>, !dbg !13
  %vdb.read1822 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 8704, !dbg !13
  %1014 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1822, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1015 = bitcast <4 x i32> %1014 to <8 x half>, !dbg !13
  %vdb.read1823 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 8960, !dbg !13
  %1016 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1823, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1017 = bitcast <4 x i32> %1016 to <8 x half>, !dbg !13
  %vdb.read1824 = getelementptr i8, ptr addrspace(3) %address.local1003, i64 12288, !dbg !13
  %1018 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1824, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1019 = bitcast <4 x i32> %1018 to <8 x half>, !dbg !13
  %vdb.read1825 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 12544, !dbg !13
  %1020 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1825, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1021 = bitcast <4 x i32> %1020 to <8 x half>, !dbg !13
  %vdb.read1826 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 12800, !dbg !13
  %1022 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1826, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1023 = bitcast <4 x i32> %1022 to <8 x half>, !dbg !13
  %vdb.read1827 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 13056, !dbg !13
  %1024 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1827, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1025 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1026 = extractvalue [2 x <8 x float>] %110, 0, !dbg !34
  %1027 = shufflevector <8 x float> %1026, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1028 = fmul <4 x float> %1027, %688, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1025, <4 x float> %1028) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1030 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1031 = shufflevector <8 x float> %1026, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1032 = fmul <4 x float> %1031, %688, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1030, <4 x float> %1032) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1034 = extractvalue [2 x <8 x float>] %110, 1, !dbg !34
  %1035 = shufflevector <8 x float> %1034, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1036 = fmul <4 x float> %1035, %696, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1025, <4 x float> %1036) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1038 = shufflevector <8 x float> %1034, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1039 = fmul <4 x float> %1038, %696, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1030, <4 x float> %1039) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1041 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1041, <4 x float> %1029) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1043 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1043, <4 x float> %1033) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1041, <4 x float> %1037) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1043, <4 x float> %1040) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1047 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1048 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1047, <4 x float> %1042) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1049 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1049, <4 x float> %1044) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1047, <4 x float> %1045) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1049, <4 x float> %1046) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1053 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1053, <4 x float> %1048) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1055 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1055, <4 x float> %1050) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1053, <4 x float> %1051) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1055, <4 x float> %1052) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1059 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1059, <4 x float> %1054) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1061 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1061, <4 x float> %1056) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1059, <4 x float> %1057) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1061, <4 x float> %1058) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1065 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1065, <4 x float> %1060) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1067 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1067, <4 x float> %1062) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1065, <4 x float> %1063) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1067, <4 x float> %1064) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1071 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1071, <4 x float> %1066) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1073 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1073, <4 x float> %1068) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1071, <4 x float> %1069) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1073, <4 x float> %1070) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1077 = bitcast <4 x i32> %1024 to <8 x half>, !dbg !13
  %1078 = shufflevector <8 x half> %1077, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1078, <4 x float> %1072) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1080 = shufflevector <8 x half> %1077, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1080, <4 x float> %1074) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1078, <4 x float> %1075) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1080, <4 x float> %1076) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1850 = or disjoint i64 %.idx17771779, 96, !dbg !16
  %vdb.address1851 = or disjoint i64 %address.local989, %vdb.address1850, !dbg !16
  %vdb.address1852 = trunc i64 %vdb.address1851 to i32, !dbg !16
  %vdb.address1853 = shl i32 %vdb.address1852, 1, !dbg !16
  %vdb.address1861 = or disjoint i64 %address.local995, %vdb.address1850, !dbg !16
  %vdb.address1862 = trunc i64 %vdb.address1861 to i32, !dbg !16
  %vdb.address1863 = shl i32 %vdb.address1862, 1, !dbg !16
  %1084 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1853, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %1085 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1820, i32 %vdb.address1863, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %1086 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1003, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1087 = bitcast <4 x i32> %1086 to <8 x half>, !dbg !13
  %1088 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1013, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1089 = bitcast <4 x i32> %1088 to <8 x half>, !dbg !13
  %1090 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1017, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1091 = bitcast <4 x i32> %1090 to <8 x half>, !dbg !13
  %1092 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1021, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1093 = bitcast <4 x i32> %1092 to <8 x half>, !dbg !13
  %1094 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1023, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1095 = bitcast <4 x i32> %1094 to <8 x half>, !dbg !13
  %1096 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1097 = bitcast <4 x i32> %1096 to <8 x half>, !dbg !13
  %1098 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1031, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1099 = bitcast <4 x i32> %1098 to <8 x half>, !dbg !13
  %1100 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1035, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1101 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1102 = extractvalue [2 x <8 x float>] %111, 0, !dbg !34
  %1103 = shufflevector <8 x float> %1102, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1104 = fmul <4 x float> %1103, %688, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1101, <4 x float> %1104) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1106 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1107 = shufflevector <8 x float> %1102, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1108 = fmul <4 x float> %1107, %688, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1109 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1106, <4 x float> %1108) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1110 = extractvalue [2 x <8 x float>] %111, 1, !dbg !34
  %1111 = shufflevector <8 x float> %1110, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1112 = fmul <4 x float> %1111, %696, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1113 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1101, <4 x float> %1112) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1114 = shufflevector <8 x float> %1110, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1115 = fmul <4 x float> %1114, %696, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1106, <4 x float> %1115) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1117 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1117, <4 x float> %1105) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1119 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1119, <4 x float> %1109) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1117, <4 x float> %1113) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1119, <4 x float> %1116) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1123 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1123, <4 x float> %1118) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1125 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1125, <4 x float> %1120) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1123, <4 x float> %1121) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1125, <4 x float> %1122) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1129 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1129, <4 x float> %1124) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1131 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1131, <4 x float> %1126) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1129, <4 x float> %1127) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1131, <4 x float> %1128) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1135 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1135, <4 x float> %1130) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1137 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1137, <4 x float> %1132) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1135, <4 x float> %1133) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1137, <4 x float> %1134) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1141 = shufflevector <8 x half> %1097, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1141, <4 x float> %1136) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1143 = shufflevector <8 x half> %1097, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1143, <4 x float> %1138) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1141, <4 x float> %1139) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1143, <4 x float> %1140) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1147 = shufflevector <8 x half> %1099, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1147, <4 x float> %1142) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1149 = shufflevector <8 x half> %1099, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1149, <4 x float> %1144) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1147, <4 x float> %1145) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1149, <4 x float> %1146) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1153 = bitcast <4 x i32> %1100 to <8 x half>, !dbg !13
  %1154 = shufflevector <8 x half> %1153, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1154, <4 x float> %1148) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1156 = shufflevector <8 x half> %1153, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1156, <4 x float> %1150) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1154, <4 x float> %1151) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1156, <4 x float> %1152) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1160 = add nuw nsw i64 %106, 1, !dbg !7
  %k0.has.next = icmp ule i64 %1160, %16, !dbg !13
  br i1 %k0.has.next, label %k0.issue, label %k0.join, !dbg !13

k0.issue:                                         ; preds = %113
  %k0.next. = lshr i64 %1160, 12, !dbg !14
  %k0.next.2365 = add nuw nsw i64 %k0.next., %17, !dbg !14
  %k0.next.2366 = shl i64 %k0.next.2365, 19, !dbg !14
  %k0.next.2367 = and i64 %k0.next.2366, 16252928, !dbg !14
  %k0.next.2368 = lshr i64 %1160, 5, !dbg !14
  %k0.next.2369 = and i64 %k0.next.2368, 127, !dbg !14
  %k0.next..idx1736 = or disjoint i64 %k0.next.2367, %k0.next.2369, !dbg !14
  %k0.next.2370 = shl i64 %1160, 14, !dbg !14
  %k0.next.2371 = shl i64 %k0.next.2368, 19, !dbg !14
  %k0.next.2372 = sub i64 %k0.next.2370, %k0.next.2371, !dbg !14
  %k0.next.2373 = add i64 %k0.next..idx1736, %k0.next.2372, !dbg !14
  %k0.next.kdma.source1888 = add nuw nsw i64 %k0.next.2373, %kdma.source1887, !dbg !14
  %k0.next.kdma.source1889 = trunc i64 %k0.next.kdma.source1888 to i32, !dbg !14
  %k0.next.kdma.source1890 = shl i32 %k0.next.kdma.source1889, 1, !dbg !14
  %k0.next.kdma.offset = add i32 %k0.next.kdma.source1890, %kdma.intra, !dbg !14
  %k0.next.2374 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %k0.next.kdma.offset, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1902 = add i64 %k0.next.2373, %kdma.source1901, !dbg !14
  %k0.next.kdma.source1903 = trunc i64 %k0.next.kdma.source1902 to i32, !dbg !14
  %k0.next.kdma.source1904 = shl i32 %k0.next.kdma.source1903, 1, !dbg !14
  %k0.next.kdma.offset1905 = add i32 %k0.next.kdma.source1904, %kdma.intra.hi, !dbg !14
  %k0.next.2375 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1906, i32 %k0.next.kdma.offset1905, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1919 = add nuw nsw i64 %k0.next.2373, %kdma.source1918, !dbg !14
  %k0.next.kdma.source1920 = trunc i64 %k0.next.kdma.source1919 to i32, !dbg !14
  %k0.next.kdma.source1921 = shl i32 %k0.next.kdma.source1920, 1, !dbg !14
  %k0.next.kdma.offset1922 = add i32 %k0.next.kdma.source1921, %kdma.intra, !dbg !14
  %k0.next.2376 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %k0.next.kdma.offset1922, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1936 = add i64 %k0.next.2373, %kdma.source1935, !dbg !14
  %k0.next.kdma.source1937 = trunc i64 %k0.next.kdma.source1936 to i32, !dbg !14
  %k0.next.kdma.source1938 = shl i32 %k0.next.kdma.source1937, 1, !dbg !14
  %k0.next.kdma.offset1939 = add i32 %k0.next.kdma.source1938, %kdma.intra.hi, !dbg !14
  %k0.next.2377 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1940, i32 %k0.next.kdma.offset1939, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14, !frisk.cross_bn_k0 !1
  br label %k0.join

k0.join:                                          ; preds = %113, %k0.issue
  %1161 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1162 = bitcast <4 x i32> %1161 to <8 x half>, !dbg !13
  %1163 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1821, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1164 = bitcast <4 x i32> %1163 to <8 x half>, !dbg !13
  %1165 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1822, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1166 = bitcast <4 x i32> %1165 to <8 x half>, !dbg !13
  %1167 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1823, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1168 = bitcast <4 x i32> %1167 to <8 x half>, !dbg !13
  %1169 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1824, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1170 = bitcast <4 x i32> %1169 to <8 x half>, !dbg !13
  %1171 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1825, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1172 = bitcast <4 x i32> %1171 to <8 x half>, !dbg !13
  %1173 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1826, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1174 = bitcast <4 x i32> %1173 to <8 x half>, !dbg !13
  %1175 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1827, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1176 = shufflevector <8 x half> %1162, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1177 = extractvalue [2 x <8 x float>] %112, 0, !dbg !34
  %1178 = shufflevector <8 x float> %1177, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1179 = fmul <4 x float> %1178, %688, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1176, <4 x float> %1179) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1181 = shufflevector <8 x half> %1162, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1182 = shufflevector <8 x float> %1177, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1183 = fmul <4 x float> %1182, %688, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1181, <4 x float> %1183) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1185 = extractvalue [2 x <8 x float>] %112, 1, !dbg !34
  %1186 = shufflevector <8 x float> %1185, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1187 = fmul <4 x float> %1186, %696, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1176, <4 x float> %1187) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1189 = shufflevector <8 x float> %1185, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1190 = fmul <4 x float> %1189, %696, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1181, <4 x float> %1190) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1192 = shufflevector <8 x half> %1164, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1192, <4 x float> %1180) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1194 = shufflevector <8 x half> %1164, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1194, <4 x float> %1184) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1192, <4 x float> %1188) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1194, <4 x float> %1191) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1198 = shufflevector <8 x half> %1166, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1198, <4 x float> %1193) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1200 = shufflevector <8 x half> %1166, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1200, <4 x float> %1195) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1198, <4 x float> %1196) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1200, <4 x float> %1197) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1204 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1204, <4 x float> %1199) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1206 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1206, <4 x float> %1201) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1204, <4 x float> %1202) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1206, <4 x float> %1203) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1210 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1210, <4 x float> %1205) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1212 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1212, <4 x float> %1207) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1210, <4 x float> %1208) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1212, <4 x float> %1209) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1216 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1216, <4 x float> %1211) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1218 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1218, <4 x float> %1213) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1216, <4 x float> %1214) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1218, <4 x float> %1215) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1222 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1222, <4 x float> %1217) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1224 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1224, <4 x float> %1219) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1222, <4 x float> %1220) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1224, <4 x float> %1221) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1228 = bitcast <4 x i32> %1175 to <8 x half>, !dbg !13
  %1229 = shufflevector <8 x half> %1228, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1229, <4 x float> %1223) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1231 = shufflevector <8 x half> %1228, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1231, <4 x float> %1225) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1229, <4 x float> %1226) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1231, <4 x float> %1227) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %1235 = shufflevector <4 x float> %985, <4 x float> %987, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1236 = insertvalue [2 x <8 x float>] poison, <8 x float> %1235, 0, !dbg !7
  %1237 = shufflevector <4 x float> %1006, <4 x float> %1007, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1238 = insertvalue [2 x <8 x float>] %1236, <8 x float> %1237, 1, !dbg !7
  %1239 = insertvalue [2 x <1 x float>] poison, <1 x float> %668, 0, !dbg !7
  %1240 = insertvalue [2 x <1 x float>] %1239, <1 x float> %681, 1, !dbg !7
  %1241 = shufflevector <4 x float> %1079, <4 x float> %1081, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1242 = insertvalue [2 x <8 x float>] poison, <8 x float> %1241, 0, !dbg !7
  %1243 = shufflevector <4 x float> %1082, <4 x float> %1083, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1244 = insertvalue [2 x <8 x float>] %1242, <8 x float> %1243, 1, !dbg !7
  %1245 = shufflevector <4 x float> %1155, <4 x float> %1157, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1246 = insertvalue [2 x <8 x float>] poison, <8 x float> %1245, 0, !dbg !7
  %1247 = shufflevector <4 x float> %1158, <4 x float> %1159, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1248 = insertvalue [2 x <8 x float>] %1246, <8 x float> %1247, 1, !dbg !7
  %1249 = shufflevector <4 x float> %1230, <4 x float> %1232, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1250 = insertvalue [2 x <8 x float>] poison, <8 x float> %1249, 0, !dbg !7
  %1251 = shufflevector <4 x float> %1233, <4 x float> %1234, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1252 = insertvalue [2 x <8 x float>] %1250, <8 x float> %1251, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  br label %105, !dbg !7

1253:                                             ; preds = %105
  %lane.sum.field252 = extractvalue [2 x <1 x float>] %108, 1, !dbg !7
  %lane.sum.local253 = extractelement <1 x float> %lane.sum.field252, i64 0, !dbg !7
  %lane.sum.lane.lo254 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane255 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo254), !dbg !7
  %1254 = shl nuw nsw i32 %lane.sum.lane255, 2, !dbg !7
  %1255 = xor i32 %1254, 64, !dbg !7
  %1256 = bitcast float %lane.sum.local253 to i32, !dbg !7
  %lane.sum.shuffle256 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1255, i32 %1256), !dbg !7
  %1257 = bitcast i32 %lane.sum.shuffle256 to float, !dbg !7
  %lane.sum.final257 = fadd float %lane.sum.local253, %1257, !dbg !7
  %1258 = xor i32 %1254, 128, !dbg !7
  %1259 = bitcast float %lane.sum.final257 to i32, !dbg !7
  %lane.sum.shuffle258 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1258, i32 %1259), !dbg !7
  %1260 = bitcast i32 %lane.sum.shuffle258 to float, !dbg !7
  %lane.sum.final259 = fadd float %lane.sum.final257, %1260, !dbg !7
  %1261 = insertelement <1 x float> poison, float %lane.sum.final259, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %108, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %1262 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1255, i32 %1262), !dbg !7
  %1263 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1263, !dbg !7
  %1264 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1258, i32 %1264), !dbg !7
  %1265 = bitcast i32 %lane.sum.shuffle250 to float, !dbg !7
  %lane.sum.final251 = fadd float %lane.sum.final, %1265, !dbg !7
  %1266 = insertelement <1 x float> poison, float %lane.sum.final251, i64 0, !dbg !7
  %1267 = and i32 %kdma.tid, 16, !dbg !7
  %.not36 = icmp eq i32 %1267, 0, !dbg !7
  %1268 = and i32 %kdma.tid, 32, !dbg !7
  %.not37 = icmp eq i32 %1268, 0, !dbg !7
  %1269 = extractvalue [2 x <8 x float>] %112, 0, !dbg !7
  %1270 = extractelement <8 x float> %1269, i64 0, !dbg !7
  %1271 = extractelement <8 x float> %1269, i64 1, !dbg !7
  %1272 = extractelement <8 x float> %1269, i64 2, !dbg !7
  %1273 = extractelement <8 x float> %1269, i64 3, !dbg !7
  %1274 = select i1 %.not36, float %1271, float %1270, !dbg !7
  %1275 = add nuw nsw i32 %lane.sum.lane255, 64, !dbg !7
  %1276 = xor i32 %lane.sum.lane255, 16, !dbg !7
  %1277 = and i32 %1275, 192, !dbg !7
  %1278 = icmp samesign ult i32 %1276, %1277, !dbg !7
  %1279 = select i1 %1278, i32 %1276, i32 %lane.sum.lane255, !dbg !7
  %1280 = shl nuw nsw i32 %1279, 2, !dbg !7
  %1281 = bitcast float %1274 to i32, !dbg !7
  %1282 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1281), !dbg !7
  %1283 = bitcast i32 %1282 to float, !dbg !7
  %1284 = select i1 %.not36, float %1270, float %1283, !dbg !7
  %1285 = select i1 %.not36, float %1283, float %1271, !dbg !7
  %1286 = select i1 %.not36, float %1273, float %1272, !dbg !7
  %1287 = bitcast float %1286 to i32, !dbg !7
  %1288 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1287), !dbg !7
  %1289 = bitcast i32 %1288 to float, !dbg !7
  %1290 = select i1 %.not36, float %1272, float %1289, !dbg !7
  %1291 = select i1 %.not36, float %1289, float %1273, !dbg !7
  %1292 = select i1 %.not37, float %1290, float %1284, !dbg !7
  %1293 = xor i32 %lane.sum.lane255, 32, !dbg !7
  %1294 = icmp samesign ult i32 %1293, %1277, !dbg !7
  %1295 = select i1 %1294, i32 %1293, i32 %lane.sum.lane255, !dbg !7
  %1296 = shl nuw nsw i32 %1295, 2, !dbg !7
  %1297 = bitcast float %1292 to i32, !dbg !7
  %1298 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1297), !dbg !7
  %1299 = bitcast i32 %1298 to float, !dbg !7
  %1300 = select i1 %.not37, float %1284, float %1299, !dbg !7
  %1301 = select i1 %.not37, float %1299, float %1290, !dbg !7
  %1302 = select i1 %.not37, float %1291, float %1285, !dbg !7
  %1303 = bitcast float %1302 to i32, !dbg !7
  %1304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1303), !dbg !7
  %1305 = bitcast i32 %1304 to float, !dbg !7
  %1306 = select i1 %.not37, float %1285, float %1305, !dbg !7
  %1307 = select i1 %.not37, float %1305, float %1291, !dbg !7
  %1308 = extractvalue [2 x <8 x float>] %111, 0, !dbg !7
  %1309 = extractelement <8 x float> %1308, i64 0, !dbg !7
  %1310 = extractelement <8 x float> %1308, i64 1, !dbg !7
  %1311 = extractelement <8 x float> %1308, i64 2, !dbg !7
  %1312 = extractelement <8 x float> %1308, i64 3, !dbg !7
  %1313 = select i1 %.not36, float %1310, float %1309, !dbg !7
  %1314 = bitcast float %1313 to i32, !dbg !7
  %1315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1314), !dbg !7
  %1316 = bitcast i32 %1315 to float, !dbg !7
  %1317 = select i1 %.not36, float %1309, float %1316, !dbg !7
  %1318 = select i1 %.not36, float %1316, float %1310, !dbg !7
  %1319 = select i1 %.not36, float %1312, float %1311, !dbg !7
  %1320 = bitcast float %1319 to i32, !dbg !7
  %1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1320), !dbg !7
  %1322 = bitcast i32 %1321 to float, !dbg !7
  %1323 = select i1 %.not36, float %1311, float %1322, !dbg !7
  %1324 = select i1 %.not36, float %1322, float %1312, !dbg !7
  %1325 = select i1 %.not37, float %1323, float %1317, !dbg !7
  %1326 = bitcast float %1325 to i32, !dbg !7
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1326), !dbg !7
  %1328 = bitcast i32 %1327 to float, !dbg !7
  %1329 = select i1 %.not37, float %1317, float %1328, !dbg !7
  %1330 = select i1 %.not37, float %1328, float %1323, !dbg !7
  %1331 = select i1 %.not37, float %1324, float %1318, !dbg !7
  %1332 = bitcast float %1331 to i32, !dbg !7
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1332), !dbg !7
  %1334 = bitcast i32 %1333 to float, !dbg !7
  %1335 = select i1 %.not37, float %1318, float %1334, !dbg !7
  %1336 = select i1 %.not37, float %1334, float %1324, !dbg !7
  %1337 = extractvalue [2 x <8 x float>] %110, 0, !dbg !7
  %1338 = extractelement <8 x float> %1337, i64 0, !dbg !7
  %1339 = extractelement <8 x float> %1337, i64 1, !dbg !7
  %1340 = extractelement <8 x float> %1337, i64 2, !dbg !7
  %1341 = extractelement <8 x float> %1337, i64 3, !dbg !7
  %1342 = select i1 %.not36, float %1339, float %1338, !dbg !7
  %1343 = bitcast float %1342 to i32, !dbg !7
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1343), !dbg !7
  %1345 = bitcast i32 %1344 to float, !dbg !7
  %1346 = select i1 %.not36, float %1338, float %1345, !dbg !7
  %1347 = select i1 %.not36, float %1345, float %1339, !dbg !7
  %1348 = select i1 %.not36, float %1341, float %1340, !dbg !7
  %1349 = bitcast float %1348 to i32, !dbg !7
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1349), !dbg !7
  %1351 = bitcast i32 %1350 to float, !dbg !7
  %1352 = select i1 %.not36, float %1340, float %1351, !dbg !7
  %1353 = select i1 %.not36, float %1351, float %1341, !dbg !7
  %1354 = select i1 %.not37, float %1352, float %1346, !dbg !7
  %1355 = bitcast float %1354 to i32, !dbg !7
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1355), !dbg !7
  %1357 = bitcast i32 %1356 to float, !dbg !7
  %1358 = select i1 %.not37, float %1346, float %1357, !dbg !7
  %1359 = select i1 %.not37, float %1357, float %1352, !dbg !7
  %1360 = select i1 %.not37, float %1353, float %1347, !dbg !7
  %1361 = bitcast float %1360 to i32, !dbg !7
  %1362 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1361), !dbg !7
  %1363 = bitcast i32 %1362 to float, !dbg !7
  %1364 = select i1 %.not37, float %1347, float %1363, !dbg !7
  %1365 = select i1 %.not37, float %1363, float %1353, !dbg !7
  %1366 = extractvalue [2 x <8 x float>] %107, 0, !dbg !7
  %1367 = extractelement <8 x float> %1366, i64 0, !dbg !7
  %1368 = extractelement <8 x float> %1366, i64 1, !dbg !7
  %1369 = extractelement <8 x float> %1366, i64 2, !dbg !7
  %1370 = extractelement <8 x float> %1366, i64 3, !dbg !7
  %1371 = select i1 %.not36, float %1368, float %1367, !dbg !7
  %1372 = bitcast float %1371 to i32, !dbg !7
  %1373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1372), !dbg !7
  %1374 = bitcast i32 %1373 to float, !dbg !7
  %1375 = select i1 %.not36, float %1367, float %1374, !dbg !7
  %1376 = select i1 %.not36, float %1374, float %1368, !dbg !7
  %1377 = select i1 %.not36, float %1370, float %1369, !dbg !7
  %1378 = bitcast float %1377 to i32, !dbg !7
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1378), !dbg !7
  %1380 = bitcast i32 %1379 to float, !dbg !7
  %1381 = select i1 %.not36, float %1369, float %1380, !dbg !7
  %1382 = select i1 %.not36, float %1380, float %1370, !dbg !7
  %1383 = select i1 %.not37, float %1381, float %1375, !dbg !7
  %1384 = bitcast float %1383 to i32, !dbg !7
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1384), !dbg !7
  %1386 = select i1 %.not37, float %1382, float %1376, !dbg !7
  %1387 = bitcast float %1386 to i32, !dbg !7
  %1388 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1387), !dbg !7
  %1389 = shufflevector <1 x float> %1266, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1390 = extractelement <8 x float> %1269, i64 4, !dbg !7
  %1391 = extractelement <8 x float> %1269, i64 5, !dbg !7
  %1392 = extractelement <8 x float> %1269, i64 6, !dbg !7
  %1393 = extractelement <8 x float> %1269, i64 7, !dbg !7
  %1394 = select i1 %.not36, float %1391, float %1390, !dbg !7
  %1395 = bitcast float %1394 to i32, !dbg !7
  %1396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1395), !dbg !7
  %1397 = bitcast i32 %1396 to float, !dbg !7
  %1398 = select i1 %.not36, float %1390, float %1397, !dbg !7
  %1399 = select i1 %.not36, float %1397, float %1391, !dbg !7
  %1400 = select i1 %.not36, float %1393, float %1392, !dbg !7
  %1401 = bitcast float %1400 to i32, !dbg !7
  %1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1401), !dbg !7
  %1403 = bitcast i32 %1402 to float, !dbg !7
  %1404 = select i1 %.not36, float %1392, float %1403, !dbg !7
  %1405 = select i1 %.not36, float %1403, float %1393, !dbg !7
  %1406 = select i1 %.not37, float %1404, float %1398, !dbg !7
  %1407 = bitcast float %1406 to i32, !dbg !7
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1407), !dbg !7
  %1409 = bitcast i32 %1408 to float, !dbg !7
  %1410 = select i1 %.not37, float %1398, float %1409, !dbg !7
  %1411 = select i1 %.not37, float %1409, float %1404, !dbg !7
  %1412 = select i1 %.not37, float %1405, float %1399, !dbg !7
  %1413 = bitcast float %1412 to i32, !dbg !7
  %1414 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1413), !dbg !7
  %1415 = bitcast i32 %1414 to float, !dbg !7
  %1416 = select i1 %.not37, float %1399, float %1415, !dbg !7
  %1417 = select i1 %.not37, float %1415, float %1405, !dbg !7
  %1418 = extractelement <8 x float> %1308, i64 4, !dbg !7
  %1419 = extractelement <8 x float> %1308, i64 5, !dbg !7
  %1420 = extractelement <8 x float> %1308, i64 6, !dbg !7
  %1421 = extractelement <8 x float> %1308, i64 7, !dbg !7
  %1422 = select i1 %.not36, float %1419, float %1418, !dbg !7
  %1423 = bitcast float %1422 to i32, !dbg !7
  %1424 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1423), !dbg !7
  %1425 = bitcast i32 %1424 to float, !dbg !7
  %1426 = select i1 %.not36, float %1418, float %1425, !dbg !7
  %1427 = select i1 %.not36, float %1425, float %1419, !dbg !7
  %1428 = select i1 %.not36, float %1421, float %1420, !dbg !7
  %1429 = bitcast float %1428 to i32, !dbg !7
  %1430 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1429), !dbg !7
  %1431 = bitcast i32 %1430 to float, !dbg !7
  %1432 = select i1 %.not36, float %1420, float %1431, !dbg !7
  %1433 = select i1 %.not36, float %1431, float %1421, !dbg !7
  %1434 = select i1 %.not37, float %1432, float %1426, !dbg !7
  %1435 = bitcast float %1434 to i32, !dbg !7
  %1436 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1435), !dbg !7
  %1437 = bitcast i32 %1436 to float, !dbg !7
  %1438 = select i1 %.not37, float %1426, float %1437, !dbg !7
  %1439 = select i1 %.not37, float %1437, float %1432, !dbg !7
  %1440 = select i1 %.not37, float %1433, float %1427, !dbg !7
  %1441 = bitcast float %1440 to i32, !dbg !7
  %1442 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1441), !dbg !7
  %1443 = bitcast i32 %1442 to float, !dbg !7
  %1444 = select i1 %.not37, float %1427, float %1443, !dbg !7
  %1445 = select i1 %.not37, float %1443, float %1433, !dbg !7
  %1446 = extractelement <8 x float> %1337, i64 4, !dbg !7
  %1447 = extractelement <8 x float> %1337, i64 5, !dbg !7
  %1448 = extractelement <8 x float> %1337, i64 6, !dbg !7
  %1449 = extractelement <8 x float> %1337, i64 7, !dbg !7
  %1450 = select i1 %.not36, float %1447, float %1446, !dbg !7
  %1451 = bitcast float %1450 to i32, !dbg !7
  %1452 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1451), !dbg !7
  %1453 = bitcast i32 %1452 to float, !dbg !7
  %1454 = select i1 %.not36, float %1446, float %1453, !dbg !7
  %1455 = select i1 %.not36, float %1453, float %1447, !dbg !7
  %1456 = select i1 %.not36, float %1449, float %1448, !dbg !7
  %1457 = bitcast float %1456 to i32, !dbg !7
  %1458 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1457), !dbg !7
  %1459 = bitcast i32 %1458 to float, !dbg !7
  %1460 = select i1 %.not36, float %1448, float %1459, !dbg !7
  %1461 = select i1 %.not36, float %1459, float %1449, !dbg !7
  %1462 = select i1 %.not37, float %1460, float %1454, !dbg !7
  %1463 = bitcast float %1462 to i32, !dbg !7
  %1464 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1463), !dbg !7
  %1465 = bitcast i32 %1464 to float, !dbg !7
  %1466 = select i1 %.not37, float %1454, float %1465, !dbg !7
  %1467 = select i1 %.not37, float %1465, float %1460, !dbg !7
  %1468 = select i1 %.not37, float %1461, float %1455, !dbg !7
  %1469 = bitcast float %1468 to i32, !dbg !7
  %1470 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1469), !dbg !7
  %1471 = bitcast i32 %1470 to float, !dbg !7
  %1472 = select i1 %.not37, float %1455, float %1471, !dbg !7
  %1473 = select i1 %.not37, float %1471, float %1461, !dbg !7
  %1474 = extractelement <8 x float> %1366, i64 4, !dbg !7
  %1475 = extractelement <8 x float> %1366, i64 5, !dbg !7
  %1476 = extractelement <8 x float> %1366, i64 6, !dbg !7
  %1477 = extractelement <8 x float> %1366, i64 7, !dbg !7
  %1478 = select i1 %.not36, float %1475, float %1474, !dbg !7
  %1479 = bitcast float %1478 to i32, !dbg !7
  %1480 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1479), !dbg !7
  %1481 = bitcast i32 %1480 to float, !dbg !7
  %1482 = select i1 %.not36, float %1474, float %1481, !dbg !7
  %1483 = select i1 %.not36, float %1481, float %1475, !dbg !7
  %1484 = select i1 %.not36, float %1477, float %1476, !dbg !7
  %1485 = bitcast float %1484 to i32, !dbg !7
  %1486 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1485), !dbg !7
  %1487 = bitcast i32 %1486 to float, !dbg !7
  %1488 = select i1 %.not36, float %1476, float %1487, !dbg !7
  %1489 = select i1 %.not36, float %1487, float %1477, !dbg !7
  %1490 = select i1 %.not37, float %1488, float %1482, !dbg !7
  %1491 = bitcast float %1490 to i32, !dbg !7
  %1492 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1491), !dbg !7
  %1493 = select i1 %.not37, float %1489, float %1483, !dbg !7
  %1494 = bitcast float %1493 to i32, !dbg !7
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1494), !dbg !7
  %1496 = extractvalue [2 x <8 x float>] %112, 1, !dbg !7
  %1497 = extractelement <8 x float> %1496, i64 0, !dbg !7
  %1498 = extractelement <8 x float> %1496, i64 1, !dbg !7
  %1499 = extractelement <8 x float> %1496, i64 2, !dbg !7
  %1500 = extractelement <8 x float> %1496, i64 3, !dbg !7
  %1501 = select i1 %.not36, float %1498, float %1497, !dbg !7
  %1502 = bitcast float %1501 to i32, !dbg !7
  %1503 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1502), !dbg !7
  %1504 = bitcast i32 %1503 to float, !dbg !7
  %1505 = select i1 %.not36, float %1497, float %1504, !dbg !7
  %1506 = select i1 %.not36, float %1504, float %1498, !dbg !7
  %1507 = select i1 %.not36, float %1500, float %1499, !dbg !7
  %1508 = bitcast float %1507 to i32, !dbg !7
  %1509 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1508), !dbg !7
  %1510 = bitcast i32 %1509 to float, !dbg !7
  %1511 = select i1 %.not36, float %1499, float %1510, !dbg !7
  %1512 = select i1 %.not36, float %1510, float %1500, !dbg !7
  %1513 = select i1 %.not37, float %1511, float %1505, !dbg !7
  %1514 = bitcast float %1513 to i32, !dbg !7
  %1515 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1514), !dbg !7
  %1516 = bitcast i32 %1515 to float, !dbg !7
  %1517 = select i1 %.not37, float %1505, float %1516, !dbg !7
  %1518 = select i1 %.not37, float %1516, float %1511, !dbg !7
  %1519 = select i1 %.not37, float %1512, float %1506, !dbg !7
  %1520 = bitcast float %1519 to i32, !dbg !7
  %1521 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1520), !dbg !7
  %1522 = bitcast i32 %1521 to float, !dbg !7
  %1523 = select i1 %.not37, float %1506, float %1522, !dbg !7
  %1524 = select i1 %.not37, float %1522, float %1512, !dbg !7
  %1525 = extractvalue [2 x <8 x float>] %111, 1, !dbg !7
  %1526 = extractelement <8 x float> %1525, i64 0, !dbg !7
  %1527 = extractelement <8 x float> %1525, i64 1, !dbg !7
  %1528 = extractelement <8 x float> %1525, i64 2, !dbg !7
  %1529 = extractelement <8 x float> %1525, i64 3, !dbg !7
  %1530 = select i1 %.not36, float %1527, float %1526, !dbg !7
  %1531 = bitcast float %1530 to i32, !dbg !7
  %1532 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1531), !dbg !7
  %1533 = bitcast i32 %1532 to float, !dbg !7
  %1534 = select i1 %.not36, float %1526, float %1533, !dbg !7
  %1535 = select i1 %.not36, float %1533, float %1527, !dbg !7
  %1536 = select i1 %.not36, float %1529, float %1528, !dbg !7
  %1537 = bitcast float %1536 to i32, !dbg !7
  %1538 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1537), !dbg !7
  %1539 = bitcast i32 %1538 to float, !dbg !7
  %1540 = select i1 %.not36, float %1528, float %1539, !dbg !7
  %1541 = select i1 %.not36, float %1539, float %1529, !dbg !7
  %1542 = select i1 %.not37, float %1540, float %1534, !dbg !7
  %1543 = bitcast float %1542 to i32, !dbg !7
  %1544 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1543), !dbg !7
  %1545 = bitcast i32 %1544 to float, !dbg !7
  %1546 = select i1 %.not37, float %1534, float %1545, !dbg !7
  %1547 = select i1 %.not37, float %1545, float %1540, !dbg !7
  %1548 = select i1 %.not37, float %1541, float %1535, !dbg !7
  %1549 = bitcast float %1548 to i32, !dbg !7
  %1550 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1549), !dbg !7
  %1551 = bitcast i32 %1550 to float, !dbg !7
  %1552 = select i1 %.not37, float %1535, float %1551, !dbg !7
  %1553 = select i1 %.not37, float %1551, float %1541, !dbg !7
  %1554 = extractvalue [2 x <8 x float>] %110, 1, !dbg !7
  %1555 = extractelement <8 x float> %1554, i64 0, !dbg !7
  %1556 = extractelement <8 x float> %1554, i64 1, !dbg !7
  %1557 = extractelement <8 x float> %1554, i64 2, !dbg !7
  %1558 = extractelement <8 x float> %1554, i64 3, !dbg !7
  %1559 = select i1 %.not36, float %1556, float %1555, !dbg !7
  %1560 = bitcast float %1559 to i32, !dbg !7
  %1561 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1560), !dbg !7
  %1562 = bitcast i32 %1561 to float, !dbg !7
  %1563 = select i1 %.not36, float %1555, float %1562, !dbg !7
  %1564 = select i1 %.not36, float %1562, float %1556, !dbg !7
  %1565 = select i1 %.not36, float %1558, float %1557, !dbg !7
  %1566 = bitcast float %1565 to i32, !dbg !7
  %1567 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1566), !dbg !7
  %1568 = bitcast i32 %1567 to float, !dbg !7
  %1569 = select i1 %.not36, float %1557, float %1568, !dbg !7
  %1570 = select i1 %.not36, float %1568, float %1558, !dbg !7
  %1571 = select i1 %.not37, float %1569, float %1563, !dbg !7
  %1572 = bitcast float %1571 to i32, !dbg !7
  %1573 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1572), !dbg !7
  %1574 = bitcast i32 %1573 to float, !dbg !7
  %1575 = select i1 %.not37, float %1563, float %1574, !dbg !7
  %1576 = select i1 %.not37, float %1574, float %1569, !dbg !7
  %1577 = select i1 %.not37, float %1570, float %1564, !dbg !7
  %1578 = bitcast float %1577 to i32, !dbg !7
  %1579 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1578), !dbg !7
  %1580 = bitcast i32 %1579 to float, !dbg !7
  %1581 = select i1 %.not37, float %1564, float %1580, !dbg !7
  %1582 = select i1 %.not37, float %1580, float %1570, !dbg !7
  %1583 = extractvalue [2 x <8 x float>] %107, 1, !dbg !7
  %1584 = extractelement <8 x float> %1583, i64 0, !dbg !7
  %1585 = extractelement <8 x float> %1583, i64 1, !dbg !7
  %1586 = extractelement <8 x float> %1583, i64 2, !dbg !7
  %1587 = extractelement <8 x float> %1583, i64 3, !dbg !7
  %1588 = select i1 %.not36, float %1585, float %1584, !dbg !7
  %1589 = bitcast float %1588 to i32, !dbg !7
  %1590 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1589), !dbg !7
  %1591 = bitcast i32 %1590 to float, !dbg !7
  %1592 = select i1 %.not36, float %1584, float %1591, !dbg !7
  %1593 = select i1 %.not36, float %1591, float %1585, !dbg !7
  %1594 = select i1 %.not36, float %1587, float %1586, !dbg !7
  %1595 = bitcast float %1594 to i32, !dbg !7
  %1596 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1595), !dbg !7
  %1597 = bitcast i32 %1596 to float, !dbg !7
  %1598 = select i1 %.not36, float %1586, float %1597, !dbg !7
  %1599 = select i1 %.not36, float %1597, float %1587, !dbg !7
  %1600 = select i1 %.not37, float %1598, float %1592, !dbg !7
  %1601 = bitcast float %1600 to i32, !dbg !7
  %1602 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1601), !dbg !7
  %1603 = select i1 %.not37, float %1599, float %1593, !dbg !7
  %1604 = bitcast float %1603 to i32, !dbg !7
  %1605 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1604), !dbg !7
  %1606 = shufflevector <1 x float> %1261, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1607 = extractelement <8 x float> %1496, i64 4, !dbg !7
  %1608 = extractelement <8 x float> %1496, i64 5, !dbg !7
  %1609 = extractelement <8 x float> %1496, i64 6, !dbg !7
  %1610 = extractelement <8 x float> %1496, i64 7, !dbg !7
  %1611 = select i1 %.not36, float %1608, float %1607, !dbg !7
  %1612 = bitcast float %1611 to i32, !dbg !7
  %1613 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1612), !dbg !7
  %1614 = bitcast i32 %1613 to float, !dbg !7
  %1615 = select i1 %.not36, float %1607, float %1614, !dbg !7
  %1616 = select i1 %.not36, float %1614, float %1608, !dbg !7
  %1617 = select i1 %.not36, float %1610, float %1609, !dbg !7
  %1618 = bitcast float %1617 to i32, !dbg !7
  %1619 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1618), !dbg !7
  %1620 = bitcast i32 %1619 to float, !dbg !7
  %1621 = select i1 %.not36, float %1609, float %1620, !dbg !7
  %1622 = select i1 %.not36, float %1620, float %1610, !dbg !7
  %1623 = select i1 %.not37, float %1621, float %1615, !dbg !7
  %1624 = bitcast float %1623 to i32, !dbg !7
  %1625 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1624), !dbg !7
  %1626 = bitcast i32 %1625 to float, !dbg !7
  %1627 = select i1 %.not37, float %1615, float %1626, !dbg !7
  %1628 = select i1 %.not37, float %1626, float %1621, !dbg !7
  %1629 = select i1 %.not37, float %1622, float %1616, !dbg !7
  %1630 = bitcast float %1629 to i32, !dbg !7
  %1631 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1630), !dbg !7
  %1632 = bitcast i32 %1631 to float, !dbg !7
  %1633 = select i1 %.not37, float %1616, float %1632, !dbg !7
  %1634 = select i1 %.not37, float %1632, float %1622, !dbg !7
  %1635 = extractelement <8 x float> %1525, i64 4, !dbg !7
  %1636 = extractelement <8 x float> %1525, i64 5, !dbg !7
  %1637 = extractelement <8 x float> %1525, i64 6, !dbg !7
  %1638 = extractelement <8 x float> %1525, i64 7, !dbg !7
  %1639 = select i1 %.not36, float %1636, float %1635, !dbg !7
  %1640 = bitcast float %1639 to i32, !dbg !7
  %1641 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1640), !dbg !7
  %1642 = bitcast i32 %1641 to float, !dbg !7
  %1643 = select i1 %.not36, float %1635, float %1642, !dbg !7
  %1644 = select i1 %.not36, float %1642, float %1636, !dbg !7
  %1645 = select i1 %.not36, float %1638, float %1637, !dbg !7
  %1646 = bitcast float %1645 to i32, !dbg !7
  %1647 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1646), !dbg !7
  %1648 = bitcast i32 %1647 to float, !dbg !7
  %1649 = select i1 %.not36, float %1637, float %1648, !dbg !7
  %1650 = select i1 %.not36, float %1648, float %1638, !dbg !7
  %1651 = select i1 %.not37, float %1649, float %1643, !dbg !7
  %1652 = bitcast float %1651 to i32, !dbg !7
  %1653 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1652), !dbg !7
  %1654 = bitcast i32 %1653 to float, !dbg !7
  %1655 = select i1 %.not37, float %1643, float %1654, !dbg !7
  %1656 = select i1 %.not37, float %1654, float %1649, !dbg !7
  %1657 = select i1 %.not37, float %1650, float %1644, !dbg !7
  %1658 = bitcast float %1657 to i32, !dbg !7
  %1659 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1658), !dbg !7
  %1660 = bitcast i32 %1659 to float, !dbg !7
  %1661 = select i1 %.not37, float %1644, float %1660, !dbg !7
  %1662 = select i1 %.not37, float %1660, float %1650, !dbg !7
  %1663 = extractelement <8 x float> %1554, i64 4, !dbg !7
  %1664 = extractelement <8 x float> %1554, i64 5, !dbg !7
  %1665 = extractelement <8 x float> %1554, i64 6, !dbg !7
  %1666 = extractelement <8 x float> %1554, i64 7, !dbg !7
  %1667 = select i1 %.not36, float %1664, float %1663, !dbg !7
  %1668 = bitcast float %1667 to i32, !dbg !7
  %1669 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1668), !dbg !7
  %1670 = bitcast i32 %1669 to float, !dbg !7
  %1671 = select i1 %.not36, float %1663, float %1670, !dbg !7
  %1672 = select i1 %.not36, float %1670, float %1664, !dbg !7
  %1673 = select i1 %.not36, float %1666, float %1665, !dbg !7
  %1674 = bitcast float %1673 to i32, !dbg !7
  %1675 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1674), !dbg !7
  %1676 = bitcast i32 %1675 to float, !dbg !7
  %1677 = select i1 %.not36, float %1665, float %1676, !dbg !7
  %1678 = select i1 %.not36, float %1676, float %1666, !dbg !7
  %1679 = select i1 %.not37, float %1677, float %1671, !dbg !7
  %1680 = bitcast float %1679 to i32, !dbg !7
  %1681 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1680), !dbg !7
  %1682 = bitcast i32 %1681 to float, !dbg !7
  %1683 = select i1 %.not37, float %1671, float %1682, !dbg !7
  %1684 = select i1 %.not37, float %1682, float %1677, !dbg !7
  %1685 = select i1 %.not37, float %1678, float %1672, !dbg !7
  %1686 = bitcast float %1685 to i32, !dbg !7
  %1687 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1686), !dbg !7
  %1688 = bitcast i32 %1687 to float, !dbg !7
  %1689 = select i1 %.not37, float %1672, float %1688, !dbg !7
  %1690 = select i1 %.not37, float %1688, float %1678, !dbg !7
  %1691 = extractelement <8 x float> %1583, i64 4, !dbg !7
  %1692 = extractelement <8 x float> %1583, i64 5, !dbg !7
  %1693 = extractelement <8 x float> %1583, i64 6, !dbg !7
  %1694 = extractelement <8 x float> %1583, i64 7, !dbg !7
  %1695 = select i1 %.not36, float %1692, float %1691, !dbg !7
  %1696 = bitcast float %1695 to i32, !dbg !7
  %1697 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1696), !dbg !7
  %1698 = bitcast i32 %1697 to float, !dbg !7
  %1699 = select i1 %.not36, float %1691, float %1698, !dbg !7
  %1700 = select i1 %.not36, float %1698, float %1692, !dbg !7
  %1701 = select i1 %.not36, float %1694, float %1693, !dbg !7
  %1702 = bitcast float %1701 to i32, !dbg !7
  %1703 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1280, i32 %1702), !dbg !7
  %1704 = bitcast i32 %1703 to float, !dbg !7
  %1705 = select i1 %.not36, float %1693, float %1704, !dbg !7
  %1706 = select i1 %.not36, float %1704, float %1694, !dbg !7
  %1707 = select i1 %.not37, float %1705, float %1699, !dbg !7
  %1708 = bitcast float %1707 to i32, !dbg !7
  %1709 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1708), !dbg !7
  %1710 = select i1 %.not37, float %1706, float %1700, !dbg !7
  %1711 = bitcast float %1710 to i32, !dbg !7
  %1712 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1711), !dbg !7
  %1713 = bitcast i32 %1709 to float, !dbg !7
  %1714 = select i1 %.not37, float %1699, float %1713, !dbg !7
  %1715 = insertelement <4 x float> poison, float %1714, i64 0, !dbg !19
  %1716 = bitcast i32 %1712 to float, !dbg !7
  %1717 = select i1 %.not37, float %1700, float %1716, !dbg !7
  %1718 = insertelement <4 x float> %1715, float %1717, i64 1, !dbg !19
  %1719 = select i1 %.not37, float %1713, float %1705, !dbg !7
  %1720 = insertelement <4 x float> %1718, float %1719, i64 2, !dbg !19
  %1721 = select i1 %.not37, float %1716, float %1706, !dbg !7
  %1722 = insertelement <4 x float> %1720, float %1721, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1606, !dbg !19
  %normalized = fmul <4 x float> %1722, %row.reciprocal, !dbg !19
  %1723 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1724 = extractelement <4 x half> %1723, i64 3, !dbg !20
  %1725 = extractelement <4 x half> %1723, i64 2, !dbg !20
  %1726 = extractelement <4 x half> %1723, i64 1, !dbg !20
  %1727 = extractelement <4 x half> %1723, i64 0, !dbg !20
  %1728 = bitcast i32 %1602 to float, !dbg !7
  %1729 = select i1 %.not37, float %1592, float %1728, !dbg !7
  %1730 = insertelement <4 x float> poison, float %1729, i64 0, !dbg !19
  %1731 = bitcast i32 %1605 to float, !dbg !7
  %1732 = select i1 %.not37, float %1593, float %1731, !dbg !7
  %1733 = insertelement <4 x float> %1730, float %1732, i64 1, !dbg !19
  %1734 = select i1 %.not37, float %1728, float %1598, !dbg !7
  %1735 = insertelement <4 x float> %1733, float %1734, i64 2, !dbg !19
  %1736 = select i1 %.not37, float %1731, float %1599, !dbg !7
  %1737 = insertelement <4 x float> %1735, float %1736, i64 3, !dbg !19
  %normalized105 = fmul <4 x float> %1737, %row.reciprocal, !dbg !19
  %1738 = fptrunc <4 x float> %normalized105 to <4 x half>, !dbg !20
  %1739 = extractelement <4 x half> %1738, i64 3, !dbg !20
  %1740 = extractelement <4 x half> %1738, i64 2, !dbg !20
  %1741 = extractelement <4 x half> %1738, i64 1, !dbg !20
  %1742 = extractelement <4 x half> %1738, i64 0, !dbg !20
  %1743 = bitcast i32 %1492 to float, !dbg !7
  %1744 = select i1 %.not37, float %1482, float %1743, !dbg !7
  %1745 = insertelement <4 x float> poison, float %1744, i64 0, !dbg !19
  %1746 = bitcast i32 %1495 to float, !dbg !7
  %1747 = select i1 %.not37, float %1483, float %1746, !dbg !7
  %1748 = insertelement <4 x float> %1745, float %1747, i64 1, !dbg !19
  %1749 = select i1 %.not37, float %1743, float %1488, !dbg !7
  %1750 = insertelement <4 x float> %1748, float %1749, i64 2, !dbg !19
  %1751 = select i1 %.not37, float %1746, float %1489, !dbg !7
  %1752 = insertelement <4 x float> %1750, float %1751, i64 3, !dbg !19
  %row.reciprocal106 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1389, !dbg !19
  %normalized107 = fmul <4 x float> %1752, %row.reciprocal106, !dbg !19
  %1753 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !20
  %1754 = extractelement <4 x half> %1753, i64 3, !dbg !20
  %1755 = extractelement <4 x half> %1753, i64 2, !dbg !20
  %1756 = extractelement <4 x half> %1753, i64 1, !dbg !20
  %1757 = extractelement <4 x half> %1753, i64 0, !dbg !20
  %1758 = bitcast i32 %1385 to float, !dbg !7
  %1759 = select i1 %.not37, float %1375, float %1758, !dbg !7
  %1760 = insertelement <4 x float> poison, float %1759, i64 0, !dbg !19
  %1761 = bitcast i32 %1388 to float, !dbg !7
  %1762 = select i1 %.not37, float %1376, float %1761, !dbg !7
  %1763 = insertelement <4 x float> %1760, float %1762, i64 1, !dbg !19
  %1764 = select i1 %.not37, float %1758, float %1381, !dbg !7
  %1765 = insertelement <4 x float> %1763, float %1764, i64 2, !dbg !19
  %1766 = select i1 %.not37, float %1761, float %1382, !dbg !7
  %1767 = insertelement <4 x float> %1765, float %1766, i64 3, !dbg !19
  %normalized108 = fmul <4 x float> %1767, %row.reciprocal106, !dbg !19
  %1768 = fptrunc <4 x float> %normalized108 to <4 x half>, !dbg !20
  %1769 = extractelement <4 x half> %1768, i64 3, !dbg !20
  %1770 = extractelement <4 x half> %1768, i64 2, !dbg !20
  %1771 = extractelement <4 x half> %1768, i64 1, !dbg !20
  %1772 = extractelement <4 x half> %1768, i64 0, !dbg !20
  %address.thread1042 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !21, !range !38, !frisk.register_lifetimes !1
  %address.local1043 = zext nneg i32 %address.thread1042 to i64, !dbg !8
  %address.local1044 = add nuw i64 %20, %address.local1043, !dbg !21
  %address.local1045 = lshr i64 %address.local1043, 2, !dbg !17
  %address.local1046 = and i64 %address.local1045, 48, !dbg !17
  %address.local1047 = add i64 %address.local1044, %address.local1046, !dbg !21
  %address.local1048 = and i64 %address.local1043, 240, !dbg !21
  %address.local1050 = sub i64 %address.local1047, %address.local1048, !dbg !21
  %address.local1051 = add i64 %address.local1050, %21, !dbg !21
  %address.local1052 = shl nuw nsw i64 %address.local1051, 7, !dbg !21
  %1773 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %24, !dbg !21
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1773, i64 %address.local1052, !dbg !21
  %address.local1056 = and i64 %address.local1045, 12, !dbg !11
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1056, !dbg !21
  store half %1772, ptr addrspace(1) %address.local1057, align 2, !dbg !21
  %address.local1059 = or disjoint i64 %address.local1056, 1, !dbg !21
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1059, !dbg !21
  store half %1771, ptr addrspace(1) %address.local1060, align 2, !dbg !21
  %address.local1062 = or disjoint i64 %address.local1056, 2, !dbg !21
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1062, !dbg !21
  store half %1770, ptr addrspace(1) %address.local1063, align 2, !dbg !21
  %address.local1065 = or disjoint i64 %address.local1056, 3, !dbg !21
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1065, !dbg !21
  store half %1769, ptr addrspace(1) %address.local1066, align 2, !dbg !21
  %address.local1068 = or disjoint i64 %address.local1056, 16, !dbg !21
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1068, !dbg !21
  store half %1757, ptr addrspace(1) %address.local1069, align 2, !dbg !21
  %address.local1071 = or disjoint i64 %address.local1056, 17, !dbg !21
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1071, !dbg !21
  store half %1756, ptr addrspace(1) %address.local1072, align 2, !dbg !21
  %address.local1074 = or disjoint i64 %address.local1056, 18, !dbg !21
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1074, !dbg !21
  store half %1755, ptr addrspace(1) %address.local1075, align 2, !dbg !21
  %address.local1077 = or disjoint i64 %address.local1056, 19, !dbg !21
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1077, !dbg !21
  store half %1754, ptr addrspace(1) %address.local1078, align 2, !dbg !21
  %address.local1080 = add nuw i64 %address.local1052, 8192, !dbg !21
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1773, i64 %address.local1080, !dbg !21
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1056, !dbg !21
  store half %1742, ptr addrspace(1) %address.local1083, align 2, !dbg !21
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1059, !dbg !21
  store half %1741, ptr addrspace(1) %address.local1085, align 2, !dbg !21
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1062, !dbg !21
  store half %1740, ptr addrspace(1) %address.local1087, align 2, !dbg !21
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1065, !dbg !21
  store half %1739, ptr addrspace(1) %address.local1089, align 2, !dbg !21
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1068, !dbg !21
  store half %1727, ptr addrspace(1) %address.local1091, align 2, !dbg !21
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1071, !dbg !21
  store half %1726, ptr addrspace(1) %address.local1093, align 2, !dbg !21
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1074, !dbg !21
  store half %1725, ptr addrspace(1) %address.local1095, align 2, !dbg !21
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1077, !dbg !21
  store half %1724, ptr addrspace(1) %address.local1097, align 2, !dbg !21
  %1774 = insertelement <4 x float> poison, float %1683, i64 0, !dbg !19
  %1775 = insertelement <4 x float> %1774, float %1689, i64 1, !dbg !19
  %1776 = insertelement <4 x float> %1775, float %1684, i64 2, !dbg !19
  %1777 = insertelement <4 x float> %1776, float %1690, i64 3, !dbg !19
  %normalized110 = fmul <4 x float> %1777, %row.reciprocal, !dbg !19
  %1778 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %1779 = extractelement <4 x half> %1778, i64 3, !dbg !20
  %1780 = extractelement <4 x half> %1778, i64 2, !dbg !20
  %1781 = extractelement <4 x half> %1778, i64 1, !dbg !20
  %1782 = extractelement <4 x half> %1778, i64 0, !dbg !20
  %1783 = insertelement <4 x float> poison, float %1575, i64 0, !dbg !19
  %1784 = insertelement <4 x float> %1783, float %1581, i64 1, !dbg !19
  %1785 = insertelement <4 x float> %1784, float %1576, i64 2, !dbg !19
  %1786 = insertelement <4 x float> %1785, float %1582, i64 3, !dbg !19
  %normalized111 = fmul <4 x float> %1786, %row.reciprocal, !dbg !19
  %1787 = fptrunc <4 x float> %normalized111 to <4 x half>, !dbg !20
  %1788 = extractelement <4 x half> %1787, i64 3, !dbg !20
  %1789 = extractelement <4 x half> %1787, i64 2, !dbg !20
  %1790 = extractelement <4 x half> %1787, i64 1, !dbg !20
  %1791 = extractelement <4 x half> %1787, i64 0, !dbg !20
  %1792 = insertelement <4 x float> poison, float %1466, i64 0, !dbg !19
  %1793 = insertelement <4 x float> %1792, float %1472, i64 1, !dbg !19
  %1794 = insertelement <4 x float> %1793, float %1467, i64 2, !dbg !19
  %1795 = insertelement <4 x float> %1794, float %1473, i64 3, !dbg !19
  %normalized113 = fmul <4 x float> %1795, %row.reciprocal106, !dbg !19
  %1796 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %1797 = extractelement <4 x half> %1796, i64 3, !dbg !20
  %1798 = extractelement <4 x half> %1796, i64 2, !dbg !20
  %1799 = extractelement <4 x half> %1796, i64 1, !dbg !20
  %1800 = extractelement <4 x half> %1796, i64 0, !dbg !20
  %1801 = insertelement <4 x float> poison, float %1358, i64 0, !dbg !19
  %1802 = insertelement <4 x float> %1801, float %1364, i64 1, !dbg !19
  %1803 = insertelement <4 x float> %1802, float %1359, i64 2, !dbg !19
  %1804 = insertelement <4 x float> %1803, float %1365, i64 3, !dbg !19
  %normalized114 = fmul <4 x float> %1804, %row.reciprocal106, !dbg !19
  %1805 = fptrunc <4 x float> %normalized114 to <4 x half>, !dbg !20
  %1806 = extractelement <4 x half> %1805, i64 3, !dbg !20
  %1807 = extractelement <4 x half> %1805, i64 2, !dbg !20
  %1808 = extractelement <4 x half> %1805, i64 1, !dbg !20
  %1809 = extractelement <4 x half> %1805, i64 0, !dbg !20
  %address.local1113 = or disjoint i64 %address.local1056, 32, !dbg !21
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1113, !dbg !21
  store half %1809, ptr addrspace(1) %address.local1114, align 2, !dbg !21
  %address.local1116 = or disjoint i64 %address.local1056, 33, !dbg !21
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1116, !dbg !21
  store half %1808, ptr addrspace(1) %address.local1117, align 2, !dbg !21
  %address.local1119 = or disjoint i64 %address.local1056, 34, !dbg !21
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1119, !dbg !21
  store half %1807, ptr addrspace(1) %address.local1120, align 2, !dbg !21
  %address.local1122 = or disjoint i64 %address.local1056, 35, !dbg !21
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1122, !dbg !21
  store half %1806, ptr addrspace(1) %address.local1123, align 2, !dbg !21
  %address.local1125 = or disjoint i64 %address.local1056, 48, !dbg !21
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1125, !dbg !21
  store half %1800, ptr addrspace(1) %address.local1126, align 2, !dbg !21
  %address.local1128 = or disjoint i64 %address.local1056, 49, !dbg !21
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1128, !dbg !21
  store half %1799, ptr addrspace(1) %address.local1129, align 2, !dbg !21
  %address.local1131 = or disjoint i64 %address.local1056, 50, !dbg !21
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1131, !dbg !21
  store half %1798, ptr addrspace(1) %address.local1132, align 2, !dbg !21
  %address.local1134 = or i64 %address.local1045, 51, !dbg !21
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1134, !dbg !21
  store half %1797, ptr addrspace(1) %address.local1135, align 2, !dbg !21
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1113, !dbg !21
  store half %1791, ptr addrspace(1) %address.local1140, align 2, !dbg !21
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1116, !dbg !21
  store half %1790, ptr addrspace(1) %address.local1142, align 2, !dbg !21
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1119, !dbg !21
  store half %1789, ptr addrspace(1) %address.local1144, align 2, !dbg !21
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1122, !dbg !21
  store half %1788, ptr addrspace(1) %address.local1146, align 2, !dbg !21
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1125, !dbg !21
  store half %1782, ptr addrspace(1) %address.local1148, align 2, !dbg !21
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1128, !dbg !21
  store half %1781, ptr addrspace(1) %address.local1150, align 2, !dbg !21
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1131, !dbg !21
  store half %1780, ptr addrspace(1) %address.local1152, align 2, !dbg !21
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1134, !dbg !21
  store half %1779, ptr addrspace(1) %address.local1154, align 2, !dbg !21
  %1810 = insertelement <4 x float> poison, float %1655, i64 0, !dbg !19
  %1811 = insertelement <4 x float> %1810, float %1661, i64 1, !dbg !19
  %1812 = insertelement <4 x float> %1811, float %1656, i64 2, !dbg !19
  %1813 = insertelement <4 x float> %1812, float %1662, i64 3, !dbg !19
  %normalized116 = fmul <4 x float> %1813, %row.reciprocal, !dbg !19
  %1814 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %1815 = extractelement <4 x half> %1814, i64 3, !dbg !20
  %1816 = extractelement <4 x half> %1814, i64 2, !dbg !20
  %1817 = extractelement <4 x half> %1814, i64 1, !dbg !20
  %1818 = extractelement <4 x half> %1814, i64 0, !dbg !20
  %1819 = insertelement <4 x float> poison, float %1546, i64 0, !dbg !19
  %1820 = insertelement <4 x float> %1819, float %1552, i64 1, !dbg !19
  %1821 = insertelement <4 x float> %1820, float %1547, i64 2, !dbg !19
  %1822 = insertelement <4 x float> %1821, float %1553, i64 3, !dbg !19
  %normalized117 = fmul <4 x float> %1822, %row.reciprocal, !dbg !19
  %1823 = fptrunc <4 x float> %normalized117 to <4 x half>, !dbg !20
  %1824 = extractelement <4 x half> %1823, i64 3, !dbg !20
  %1825 = extractelement <4 x half> %1823, i64 2, !dbg !20
  %1826 = extractelement <4 x half> %1823, i64 1, !dbg !20
  %1827 = extractelement <4 x half> %1823, i64 0, !dbg !20
  %1828 = insertelement <4 x float> poison, float %1438, i64 0, !dbg !19
  %1829 = insertelement <4 x float> %1828, float %1444, i64 1, !dbg !19
  %1830 = insertelement <4 x float> %1829, float %1439, i64 2, !dbg !19
  %1831 = insertelement <4 x float> %1830, float %1445, i64 3, !dbg !19
  %normalized119 = fmul <4 x float> %1831, %row.reciprocal106, !dbg !19
  %1832 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %1833 = extractelement <4 x half> %1832, i64 3, !dbg !20
  %1834 = extractelement <4 x half> %1832, i64 2, !dbg !20
  %1835 = extractelement <4 x half> %1832, i64 1, !dbg !20
  %1836 = extractelement <4 x half> %1832, i64 0, !dbg !20
  %1837 = insertelement <4 x float> poison, float %1329, i64 0, !dbg !19
  %1838 = insertelement <4 x float> %1837, float %1335, i64 1, !dbg !19
  %1839 = insertelement <4 x float> %1838, float %1330, i64 2, !dbg !19
  %1840 = insertelement <4 x float> %1839, float %1336, i64 3, !dbg !19
  %normalized120 = fmul <4 x float> %1840, %row.reciprocal106, !dbg !19
  %1841 = fptrunc <4 x float> %normalized120 to <4 x half>, !dbg !20
  %1842 = extractelement <4 x half> %1841, i64 3, !dbg !20
  %1843 = extractelement <4 x half> %1841, i64 2, !dbg !20
  %1844 = extractelement <4 x half> %1841, i64 1, !dbg !20
  %1845 = extractelement <4 x half> %1841, i64 0, !dbg !20
  %address.local1170 = or disjoint i64 %address.local1056, 64, !dbg !21
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1170, !dbg !21
  store half %1845, ptr addrspace(1) %address.local1171, align 2, !dbg !21
  %address.local1173 = or disjoint i64 %address.local1056, 65, !dbg !21
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1173, !dbg !21
  store half %1844, ptr addrspace(1) %address.local1174, align 2, !dbg !21
  %address.local1176 = or disjoint i64 %address.local1056, 66, !dbg !21
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1176, !dbg !21
  store half %1843, ptr addrspace(1) %address.local1177, align 2, !dbg !21
  %address.local1179 = or disjoint i64 %address.local1056, 67, !dbg !21
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1179, !dbg !21
  store half %1842, ptr addrspace(1) %address.local1180, align 2, !dbg !21
  %address.local1182 = or disjoint i64 %address.local1056, 80, !dbg !21
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1182, !dbg !21
  store half %1836, ptr addrspace(1) %address.local1183, align 2, !dbg !21
  %address.local1185 = or disjoint i64 %address.local1056, 81, !dbg !21
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1185, !dbg !21
  store half %1835, ptr addrspace(1) %address.local1186, align 2, !dbg !21
  %address.local1188 = or disjoint i64 %address.local1056, 82, !dbg !21
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1188, !dbg !21
  store half %1834, ptr addrspace(1) %address.local1189, align 2, !dbg !21
  %address.local1191 = or disjoint i64 %address.local1056, 83, !dbg !21
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1191, !dbg !21
  store half %1833, ptr addrspace(1) %address.local1192, align 2, !dbg !21
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1170, !dbg !21
  store half %1827, ptr addrspace(1) %address.local1197, align 2, !dbg !21
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1173, !dbg !21
  store half %1826, ptr addrspace(1) %address.local1199, align 2, !dbg !21
  %address.local1201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1176, !dbg !21
  store half %1825, ptr addrspace(1) %address.local1201, align 2, !dbg !21
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1179, !dbg !21
  store half %1824, ptr addrspace(1) %address.local1203, align 2, !dbg !21
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1182, !dbg !21
  store half %1818, ptr addrspace(1) %address.local1205, align 2, !dbg !21
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1185, !dbg !21
  store half %1817, ptr addrspace(1) %address.local1207, align 2, !dbg !21
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1188, !dbg !21
  store half %1816, ptr addrspace(1) %address.local1209, align 2, !dbg !21
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1191, !dbg !21
  store half %1815, ptr addrspace(1) %address.local1211, align 2, !dbg !21
  %1846 = insertelement <4 x float> poison, float %1627, i64 0, !dbg !19
  %1847 = insertelement <4 x float> %1846, float %1633, i64 1, !dbg !19
  %1848 = insertelement <4 x float> %1847, float %1628, i64 2, !dbg !19
  %1849 = insertelement <4 x float> %1848, float %1634, i64 3, !dbg !19
  %normalized122 = fmul <4 x float> %1849, %row.reciprocal, !dbg !19
  %1850 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %1851 = extractelement <4 x half> %1850, i64 3, !dbg !20
  %1852 = extractelement <4 x half> %1850, i64 2, !dbg !20
  %1853 = extractelement <4 x half> %1850, i64 1, !dbg !20
  %1854 = extractelement <4 x half> %1850, i64 0, !dbg !20
  %1855 = insertelement <4 x float> poison, float %1517, i64 0, !dbg !19
  %1856 = insertelement <4 x float> %1855, float %1523, i64 1, !dbg !19
  %1857 = insertelement <4 x float> %1856, float %1518, i64 2, !dbg !19
  %1858 = insertelement <4 x float> %1857, float %1524, i64 3, !dbg !19
  %normalized123 = fmul <4 x float> %1858, %row.reciprocal, !dbg !19
  %1859 = fptrunc <4 x float> %normalized123 to <4 x half>, !dbg !20
  %1860 = extractelement <4 x half> %1859, i64 3, !dbg !20
  %1861 = extractelement <4 x half> %1859, i64 2, !dbg !20
  %1862 = extractelement <4 x half> %1859, i64 1, !dbg !20
  %1863 = extractelement <4 x half> %1859, i64 0, !dbg !20
  %1864 = insertelement <4 x float> poison, float %1410, i64 0, !dbg !19
  %1865 = insertelement <4 x float> %1864, float %1416, i64 1, !dbg !19
  %1866 = insertelement <4 x float> %1865, float %1411, i64 2, !dbg !19
  %1867 = insertelement <4 x float> %1866, float %1417, i64 3, !dbg !19
  %normalized125 = fmul <4 x float> %1867, %row.reciprocal106, !dbg !19
  %1868 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %1869 = extractelement <4 x half> %1868, i64 3, !dbg !20
  %1870 = extractelement <4 x half> %1868, i64 2, !dbg !20
  %1871 = extractelement <4 x half> %1868, i64 1, !dbg !20
  %1872 = extractelement <4 x half> %1868, i64 0, !dbg !20
  %1873 = insertelement <4 x float> poison, float %1300, i64 0, !dbg !19
  %1874 = insertelement <4 x float> %1873, float %1306, i64 1, !dbg !19
  %1875 = insertelement <4 x float> %1874, float %1301, i64 2, !dbg !19
  %1876 = insertelement <4 x float> %1875, float %1307, i64 3, !dbg !19
  %normalized126 = fmul <4 x float> %1876, %row.reciprocal106, !dbg !19
  %1877 = fptrunc <4 x float> %normalized126 to <4 x half>, !dbg !20
  %1878 = extractelement <4 x half> %1877, i64 3, !dbg !20
  %1879 = extractelement <4 x half> %1877, i64 2, !dbg !20
  %1880 = extractelement <4 x half> %1877, i64 1, !dbg !20
  %1881 = extractelement <4 x half> %1877, i64 0, !dbg !20
  %address.local1227 = or disjoint i64 %address.local1056, 96, !dbg !21
  %address.local1228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1227, !dbg !21
  store half %1881, ptr addrspace(1) %address.local1228, align 2, !dbg !21
  %address.local1230 = or disjoint i64 %address.local1056, 97, !dbg !21
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1230, !dbg !21
  store half %1880, ptr addrspace(1) %address.local1231, align 2, !dbg !21
  %address.local1233 = or disjoint i64 %address.local1056, 98, !dbg !21
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1233, !dbg !21
  store half %1879, ptr addrspace(1) %address.local1234, align 2, !dbg !21
  %address.local1236 = or disjoint i64 %address.local1056, 99, !dbg !21
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1236, !dbg !21
  store half %1878, ptr addrspace(1) %address.local1237, align 2, !dbg !21
  %address.local1239 = or disjoint i64 %address.local1056, 112, !dbg !21
  %address.local1240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1239, !dbg !21
  store half %1872, ptr addrspace(1) %address.local1240, align 2, !dbg !21
  %address.local1242 = or disjoint i64 %address.local1056, 113, !dbg !21
  %address.local1243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1242, !dbg !21
  store half %1871, ptr addrspace(1) %address.local1243, align 2, !dbg !21
  %address.local1245 = or disjoint i64 %address.local1056, 114, !dbg !21
  %address.local1246 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1245, !dbg !21
  store half %1870, ptr addrspace(1) %address.local1246, align 2, !dbg !21
  %address.local1248 = or i64 %address.local1045, 115, !dbg !21
  %address.local1249 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1248, !dbg !21
  store half %1869, ptr addrspace(1) %address.local1249, align 2, !dbg !21
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1227, !dbg !21
  store half %1863, ptr addrspace(1) %address.local1254, align 2, !dbg !21
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1230, !dbg !21
  store half %1862, ptr addrspace(1) %address.local1256, align 2, !dbg !21
  %address.local1258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1233, !dbg !21
  store half %1861, ptr addrspace(1) %address.local1258, align 2, !dbg !21
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1236, !dbg !21
  store half %1860, ptr addrspace(1) %address.local1260, align 2, !dbg !21
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1239, !dbg !21
  store half %1854, ptr addrspace(1) %address.local1262, align 2, !dbg !21
  %address.local1264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1242, !dbg !21
  store half %1853, ptr addrspace(1) %address.local1264, align 2, !dbg !21
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1245, !dbg !21
  store half %1852, ptr addrspace(1) %address.local1266, align 2, !dbg !21
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1248, !dbg !21
  store half %1851, ptr addrspace(1) %address.local1268, align 2, !dbg !21
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
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
!llvm.module.flags = !{!37}

!0 = !DIFile(filename: "causal.input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/causal_best_mytest")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 70, column: 22, scope: !4)
!11 = !DILocation(line: 72, column: 24, scope: !4)
!13 = !DILocation(line: 102, column: 23, scope: !4)
!14 = !DILocation(line: 71, column: 22, scope: !4)
!15 = !DILocation(line: 73, column: 26, scope: !4)
!16 = !DILocation(line: 101, column: 20, scope: !4)
!17 = !DILocation(line: 77, column: 19, scope: !4)
!18 = !DILocation(line: 79, column: 24, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
!23 = !DILocation(line: 80, column: 23, scope: !4)
!24 = !DILocation(line: 81, column: 26, scope: !4)
!25 = !DILocation(line: 88, column: 21, scope: !4)
!26 = !DILocation(line: 89, column: 20, scope: !4)
!27 = !DILocation(line: 90, column: 22, scope: !4)
!28 = !DILocation(line: 91, column: 23, scope: !4)
!29 = !DILocation(line: 92, column: 20, scope: !4)
!30 = !DILocation(line: 93, column: 22, scope: !4)
!31 = !DILocation(line: 94, column: 22, scope: !4)
!32 = !DILocation(line: 95, column: 20, scope: !4)
!33 = !DILocation(line: 96, column: 16, scope: !4)
!34 = !DILocation(line: 103, column: 22, scope: !4)
!36 = !{i32 256, i32 1, i32 1}
!37 = !{i32 2, !"Debug Info Version", i32 3}
!38 = !{i32 0, i32 256}
!39 = !{i32 4}
!40 = !{i64 8192}
!41 = !{i32 0}
!42 = !{i32 1}
!43 = !{i32 2}
!44 = !{i32 3}
!45 = !{i32 5}
!46 = !{i32 6}
!47 = !{i32 7}
!48 = !{i32 8}
!49 = !{i32 9}
!50 = !{i32 10}
!51 = !{i32 11}
!52 = !{i32 12}
!53 = !{i32 13}
!54 = !{i32 14}
!55 = !{i32 15}
!56 = !{i32 16}
!57 = !{i32 17}
!58 = !{i32 18}
!59 = !{i32 19}
!60 = !{i32 20}
!61 = !{i32 21}
!62 = !{i32 22}
